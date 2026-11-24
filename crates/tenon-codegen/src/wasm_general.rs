//! General-Purpose Tenon -> Standalone Binary WebAssembly (`.wasm` 1.0 MVP) Compiler.
//! Zero external toolchain dependencies: directly compiles Tenon `pub fn`s (scalar & refined
//! `Int`/`Float`/`Bool`, locals, arithmetic, bitwise, `if/else`, `while`, `break`/`continue`,
//! `return`, direct & recursive calls) into valid LEB128 `.wasm` binaries runnable in V8/Node/browsers.

use crate::lower_interfaces::prepare_program_for_codegen;
use std::collections::BTreeMap;
use tenon_syntax::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WasmTy {
    I64,
    F64,
}

impl WasmTy {
    fn byte(self) -> u8 {
        match self {
            WasmTy::I64 => 0x7E,
            WasmTy::F64 => 0x7C,
        }
    }
}

struct FnSigWasm {
    idx: u32,
    ret: Option<WasmTy>,
}

/// Compiles all scalar/numeric concrete functions in `program` into a standalone `.wasm` binary.
pub fn compile_program_to_wasm(program: &Program) -> Result<Vec<u8>, String> {
    let prepared = prepare_program_for_codegen(program);
    let mut alias_bases: BTreeMap<String, TypeExpr> = BTreeMap::new();
    for item in &prepared.program.items {
        if let Item::TypeAlias(ta) = item {
            alias_bases.insert(ta.name.clone(), ta.base.clone());
        }
    }

    let mut fns: Vec<&FnDecl> = Vec::new();
    for item in &prepared.program.items {
        if let Item::Fn(f) = item {
            if !f.is_ghost && !f.is_async && !f.is_test && f.type_params.is_empty() {
                fns.push(f);
            }
        }
    }
    if fns.is_empty() {
        return Err("no compilable functions found for wasm32 target".into());
    }

    let mut sigs: BTreeMap<String, FnSigWasm> = BTreeMap::new();
    let mut ordered_sigs: Vec<(Vec<WasmTy>, Option<WasmTy>)> = Vec::new();
    for (idx, f) in fns.iter().enumerate() {
        let params: Vec<WasmTy> = f
            .params
            .iter()
            .map(|p| ty_to_wasm(&p.ty, &alias_bases))
            .collect();
        let ret = f.ret_ty.as_ref().map(|t| ty_to_wasm(t, &alias_bases));
        sigs.insert(
            f.name.clone(),
            FnSigWasm {
                idx: idx as u32,
                ret,
            },
        );
        ordered_sigs.push((params, ret));
    }

    let mut wasm = Vec::new();
    // Magic `\0asm` + Version 1
    wasm.extend_from_slice(&[0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00]);

    // Section 1: Type Section (1 signature entry per function)
    let mut type_sec = Vec::new();
    push_uleb(&mut type_sec, ordered_sigs.len() as u32);
    for (ps, rt) in &ordered_sigs {
        type_sec.push(0x60);
        push_uleb(&mut type_sec, ps.len() as u32);
        for p in ps {
            type_sec.push(p.byte());
        }
        if let Some(r) = rt {
            type_sec.push(0x01);
            type_sec.push(r.byte());
        } else {
            type_sec.push(0x00);
        }
    }
    emit_section(&mut wasm, 1, &type_sec);

    // Section 3: Function Section
    let mut func_sec = Vec::new();
    push_uleb(&mut func_sec, fns.len() as u32);
    for i in 0..fns.len() {
        push_uleb(&mut func_sec, i as u32);
    }
    emit_section(&mut wasm, 3, &func_sec);

    // Section 5: Memory Section (1 page = 64 KiB)
    let mut mem_sec = Vec::new();
    push_uleb(&mut mem_sec, 1);
    mem_sec.push(0x00);
    push_uleb(&mut mem_sec, 1);
    emit_section(&mut wasm, 5, &mem_sec);

    // Section 7: Export Section (memory + all `pub fn`s + `main`)
    let mut exports: Vec<(String, u8, u32)> = vec![("memory".into(), 0x02, 0)];
    for (idx, f) in fns.iter().enumerate() {
        if f.is_pub || f.name == "main" {
            exports.push((f.name.clone(), 0x00, idx as u32));
        }
    }
    let mut exp_sec = Vec::new();
    push_uleb(&mut exp_sec, exports.len() as u32);
    for (name, kind, idx) in &exports {
        push_uleb(&mut exp_sec, name.len() as u32);
        exp_sec.extend_from_slice(name.as_bytes());
        exp_sec.push(*kind);
        push_uleb(&mut exp_sec, *idx);
    }
    emit_section(&mut wasm, 7, &exp_sec);

    // Section 10: Code Section
    let mut code_sec = Vec::new();
    push_uleb(&mut code_sec, fns.len() as u32);
    for f in &fns {
        let body = compile_wasm_fn(f, &sigs, &alias_bases)?;
        push_uleb(&mut code_sec, body.len() as u32);
        code_sec.extend_from_slice(&body);
    }
    emit_section(&mut wasm, 10, &code_sec);

    Ok(wasm)
}

fn resolve_alias_base<'a>(ty: &'a TypeExpr, aliases: &'a BTreeMap<String, TypeExpr>) -> &'a TypeExpr {
    let mut cur = ty;
    for _ in 0..16 {
        if let TypeExpr::Named { name, .. } = cur {
            if let Some(next) = aliases.get(name) {
                cur = next;
                continue;
            }
        }
        break;
    }
    cur
}

fn ty_to_wasm(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> WasmTy {
    match resolve_alias_base(ty, aliases) {
        TypeExpr::Named { name, .. } if matches!(name.as_str(), "Float" | "Float64" | "Float32") => {
            WasmTy::F64
        }
        _ => WasmTy::I64,
    }
}

struct WasmFnEmitter<'a> {
    sigs: &'a BTreeMap<String, FnSigWasm>,
    aliases: &'a BTreeMap<String, TypeExpr>,
    locals: BTreeMap<String, (u32, WasmTy)>,
    extra_locals: Vec<WasmTy>,
    next_local: u32,
    code: Vec<u8>,
    /// Depth stack for `while` loops: `(break_depth, continue_depth)` relative to current nesting
    nesting: u32,
    loop_depths: Vec<(u32, u32)>,
}

impl<'a> WasmFnEmitter<'a> {
    fn alloc_local(&mut self, name: &str, ty: WasmTy) -> u32 {
        let idx = self.next_local;
        self.next_local += 1;
        self.extra_locals.push(ty);
        self.locals.insert(name.to_string(), (idx, ty));
        idx
    }

    fn emit_block(&mut self, blk: &Block, want_val: bool) -> Result<WasmTy, String> {
        for s in &blk.stmts {
            self.emit_stmt(s)?;
        }
        if let Some(t) = &blk.tail {
            let ty = self.emit_expr(t)?;
            if !want_val {
                self.code.push(0x1A); // drop
            }
            Ok(ty)
        } else {
            if want_val {
                self.code.push(0x42); // i64.const 0
                push_sleb64(&mut self.code, 0);
            }
            Ok(WasmTy::I64)
        }
    }

    fn emit_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Let { name, ty, value, .. } => {
                let vty = self.emit_expr(value)?;
                let target_ty = ty
                    .as_ref()
                    .map(|t| ty_to_wasm(t, self.aliases))
                    .unwrap_or(vty);
                if name == "_" {
                    self.code.push(0x1A); // drop
                } else {
                    let idx = self.alloc_local(name, target_ty);
                    self.code.push(0x21); // local.set
                    push_uleb(&mut self.code, idx);
                }
            }
            Stmt::Assign { target, op, value, .. } => {
                let ExprKind::Ident(vname) = &target.kind else {
                    return Err("wasm32 backend supports scalar local assignment".into());
                };
                let Some(&(idx, vty)) = self.locals.get(vname) else {
                    return Err(format!("unknown local `{}` in wasm32 assign", vname));
                };
                if *op == AssignOp::Set {
                    let _ = self.emit_expr(value)?;
                } else {
                    self.code.push(0x20); // local.get
                    push_uleb(&mut self.code, idx);
                    let _ = self.emit_expr(value)?;
                    match (vty, op) {
                        (WasmTy::I64, AssignOp::AddSet) => self.code.push(0x7C), // i64.add
                        (WasmTy::I64, AssignOp::SubSet) => self.code.push(0x7D), // i64.sub
                        (WasmTy::I64, AssignOp::MulSet) => self.code.push(0x7E), // i64.mul
                        (WasmTy::I64, AssignOp::DivSet) => self.code.push(0x7F), // i64.div_s
                        (WasmTy::I64, AssignOp::BitAndSet) => self.code.push(0x83), // i64.and
                        (WasmTy::I64, AssignOp::BitOrSet) => self.code.push(0x84), // i64.or
                        (WasmTy::I64, AssignOp::BitXorSet) => self.code.push(0x85), // i64.xor
                        (WasmTy::I64, AssignOp::ShlSet) => self.code.push(0x86), // i64.shl
                        (WasmTy::I64, AssignOp::ShrSet) => self.code.push(0x88), // i64.shr_u
                        (WasmTy::F64, AssignOp::AddSet) => self.code.push(0xA0), // f64.add
                        (WasmTy::F64, AssignOp::SubSet) => self.code.push(0xA1), // f64.sub
                        (WasmTy::F64, AssignOp::MulSet) => self.code.push(0xA2), // f64.mul
                        (WasmTy::F64, AssignOp::DivSet) => self.code.push(0xA3), // f64.div
                        _ => return Err(format!("unsupported assign op {:?} in wasm32", op)),
                    }
                }
                self.code.push(0x21); // local.set
                push_uleb(&mut self.code, idx);
            }
            Stmt::Return { value, .. } => {
                if let Some(v) = value {
                    let _ = self.emit_expr(v)?;
                }
                self.code.push(0x0F); // return
            }
            Stmt::While { cond, body, .. } => {
                // block $break (void)
                //   loop $cont (void)
                //     <cond> -> i32; i32.eqz; br_if $break (depth 1)
                //     <body>
                //     br $cont (depth 0)
                //   end
                // end
                self.code.extend_from_slice(&[0x02, 0x40]); // block void
                self.code.extend_from_slice(&[0x03, 0x40]); // loop void
                let break_nest = self.nesting + 1;
                let cont_nest = self.nesting + 2;
                self.nesting += 2;
                self.loop_depths.push((break_nest, cont_nest));

                let _ = self.emit_expr(cond)?;
                self.code.push(0x50); // i64.eqz (cond == 0 => exit loop)
                self.code.push(0x0D); // br_if 1 ($break)
                push_uleb(&mut self.code, 1);

                let _ = self.emit_block(body, false)?;
                self.code.push(0x0C); // br 0 ($cont)
                push_uleb(&mut self.code, 0);

                self.loop_depths.pop();
                self.nesting -= 2;
                self.code.push(0x0B); // end loop
                self.code.push(0x0B); // end block
            }
            Stmt::Break { .. } => {
                if let Some(&(break_nest, _)) = self.loop_depths.last() {
                    let rel = self.nesting - break_nest;
                    self.code.push(0x0C); // br
                    push_uleb(&mut self.code, rel);
                }
            }
            Stmt::Continue { .. } => {
                if let Some(&(_, cont_nest)) = self.loop_depths.last() {
                    let rel = self.nesting - cont_nest;
                    self.code.push(0x0C); // br
                    push_uleb(&mut self.code, rel);
                }
            }
            Stmt::Assert { cond, .. } => {
                let _ = self.emit_expr(cond)?;
                self.code.push(0x50); // i64.eqz
                self.code.extend_from_slice(&[0x04, 0x40, 0x00, 0x0B]); // if (void) unreachable end
            }
            Stmt::Expr { expr, .. } => {
                let _ = self.emit_expr(expr)?;
                self.code.push(0x1A); // drop
            }
            Stmt::Ghost { .. } | Stmt::Defer { .. } | Stmt::LetTuple { .. } | Stmt::For { .. } | Stmt::Mutate { .. } => {}
        }
        Ok(())
    }

    fn emit_expr(&mut self, expr: &Expr) -> Result<WasmTy, String> {
        match &expr.kind {
            ExprKind::Int(n) => {
                self.code.push(0x42); // i64.const
                push_sleb64(&mut self.code, *n);
                Ok(WasmTy::I64)
            }
            ExprKind::Float(x) => {
                self.code.push(0x44); // f64.const
                self.code.extend_from_slice(&x.to_le_bytes());
                Ok(WasmTy::F64)
            }
            ExprKind::Bool(b) => {
                self.code.push(0x42); // i64.const 0/1
                push_sleb64(&mut self.code, if *b { 1 } else { 0 });
                Ok(WasmTy::I64)
            }
            ExprKind::Ident(s) => {
                let Some(&(idx, ty)) = self.locals.get(s) else {
                    return Err(format!("unknown identifier `{}` in wasm32 backend", s));
                };
                self.code.push(0x20); // local.get
                push_uleb(&mut self.code, idx);
                Ok(ty)
            }
            ExprKind::Unary { op, expr: inner } => {
                let ty = self.emit_expr(inner)?;
                match (ty, op) {
                    (WasmTy::I64, UnOp::Neg) => {
                        let tmp = self.alloc_local("__neg_tmp", WasmTy::I64);
                        self.code.push(0x21);
                        push_uleb(&mut self.code, tmp);
                        self.code.push(0x42);
                        push_sleb64(&mut self.code, 0);
                        self.code.push(0x20);
                        push_uleb(&mut self.code, tmp);
                        self.code.push(0x7D); // i64.sub
                        Ok(WasmTy::I64)
                    }
                    (WasmTy::F64, UnOp::Neg) => {
                        self.code.push(0x9A); // f64.neg
                        Ok(WasmTy::F64)
                    }
                    (WasmTy::I64, UnOp::Not) => {
                        self.code.push(0x50); // i64.eqz -> i32
                        self.code.push(0xAD); // i64.extend_i32_u
                        Ok(WasmTy::I64)
                    }
                    (WasmTy::I64, UnOp::BitNot) => {
                        self.code.push(0x42);
                        push_sleb64(&mut self.code, -1);
                        self.code.push(0x85); // i64.xor
                        Ok(WasmTy::I64)
                    }
                    _ => Err(format!("unsupported unary op {:?} in wasm32", op)),
                }
            }
            ExprKind::Binary { op, left, right } => {
                let lt = self.emit_expr(left)?;
                let _rt = self.emit_expr(right)?;
                match (lt, op) {
                    (WasmTy::I64, BinOp::Add) => { self.code.push(0x7C); Ok(WasmTy::I64) }
                    (WasmTy::I64, BinOp::Sub) => { self.code.push(0x7D); Ok(WasmTy::I64) }
                    (WasmTy::I64, BinOp::Mul) => { self.code.push(0x7E); Ok(WasmTy::I64) }
                    (WasmTy::I64, BinOp::Div) => { self.code.push(0x7F); Ok(WasmTy::I64) }
                    (WasmTy::I64, BinOp::Mod) => { self.code.push(0x81); Ok(WasmTy::I64) }
                    (WasmTy::I64, BinOp::BitAnd) | (WasmTy::I64, BinOp::And) => { self.code.push(0x83); Ok(WasmTy::I64) }
                    (WasmTy::I64, BinOp::BitOr) | (WasmTy::I64, BinOp::Or) => { self.code.push(0x84); Ok(WasmTy::I64) }
                    (WasmTy::I64, BinOp::BitXor) => { self.code.push(0x85); Ok(WasmTy::I64) }
                    (WasmTy::I64, BinOp::Shl) => { self.code.push(0x86); Ok(WasmTy::I64) }
                    (WasmTy::I64, BinOp::Shr) => { self.code.push(0x88); Ok(WasmTy::I64) }
                    (WasmTy::I64, BinOp::Eq) => { self.code.extend_from_slice(&[0x51, 0xAD]); Ok(WasmTy::I64) }
                    (WasmTy::I64, BinOp::Ne) => { self.code.extend_from_slice(&[0x52, 0xAD]); Ok(WasmTy::I64) }
                    (WasmTy::I64, BinOp::Lt) => { self.code.extend_from_slice(&[0x53, 0xAD]); Ok(WasmTy::I64) }
                    (WasmTy::I64, BinOp::Gt) => { self.code.extend_from_slice(&[0x55, 0xAD]); Ok(WasmTy::I64) }
                    (WasmTy::I64, BinOp::Le) => { self.code.extend_from_slice(&[0x57, 0xAD]); Ok(WasmTy::I64) }
                    (WasmTy::I64, BinOp::Ge) => { self.code.extend_from_slice(&[0x59, 0xAD]); Ok(WasmTy::I64) }
                    (WasmTy::F64, BinOp::Add) => { self.code.push(0xA0); Ok(WasmTy::F64) }
                    (WasmTy::F64, BinOp::Sub) => { self.code.push(0xA1); Ok(WasmTy::F64) }
                    (WasmTy::F64, BinOp::Mul) => { self.code.push(0xA2); Ok(WasmTy::F64) }
                    (WasmTy::F64, BinOp::Div) => { self.code.push(0xA3); Ok(WasmTy::F64) }
                    (WasmTy::F64, BinOp::Eq) => { self.code.extend_from_slice(&[0x61, 0xAD]); Ok(WasmTy::I64) }
                    (WasmTy::F64, BinOp::Ne) => { self.code.extend_from_slice(&[0x62, 0xAD]); Ok(WasmTy::I64) }
                    (WasmTy::F64, BinOp::Lt) => { self.code.extend_from_slice(&[0x63, 0xAD]); Ok(WasmTy::I64) }
                    (WasmTy::F64, BinOp::Gt) => { self.code.extend_from_slice(&[0x64, 0xAD]); Ok(WasmTy::I64) }
                    (WasmTy::F64, BinOp::Le) => { self.code.extend_from_slice(&[0x65, 0xAD]); Ok(WasmTy::I64) }
                    (WasmTy::F64, BinOp::Ge) => { self.code.extend_from_slice(&[0x66, 0xAD]); Ok(WasmTy::I64) }
                    _ => Err(format!("unsupported binary op {:?} in wasm32", op)),
                }
            }
            ExprKind::Cast { expr: inner, ty } | ExprKind::TryCast { expr: inner, ty } => {
                let src = self.emit_expr(inner)?;
                let dst = ty_to_wasm(ty, self.aliases);
                match (src, dst) {
                    (WasmTy::I64, WasmTy::F64) => self.code.push(0xB9), // f64.convert_i64_s
                    (WasmTy::F64, WasmTy::I64) => self.code.push(0xB0), // i64.trunc_f64_s
                    _ => {}
                }
                Ok(dst)
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                let _ = self.emit_expr(cond)?;
                self.code.extend_from_slice(&[0x42, 0x00, 0x52]); // i64.const 0; i64.ne -> i32
                let has_else = else_branch.is_some();
                let res_ty = if has_else { WasmTy::I64 } else { WasmTy::I64 };
                if has_else {
                    self.code.extend_from_slice(&[0x04, res_ty.byte()]); // if (result i64)
                } else {
                    self.code.extend_from_slice(&[0x04, 0x40]); // if (void)
                }
                self.nesting += 1;
                let t_ty = self.emit_block(then_block, has_else)?;
                if let Some(eb) = else_branch {
                    self.code.push(0x05); // else
                    let _ = self.emit_expr(eb)?;
                }
                self.nesting -= 1;
                self.code.push(0x0B); // end
                if !has_else {
                    self.code.push(0x42);
                    push_sleb64(&mut self.code, 0);
                }
                Ok(t_ty)
            }
            ExprKind::Block(b) | ExprKind::Unsafe(b) => self.emit_block(b, true),
            ExprKind::Call { callee, args } => {
                let ExprKind::Ident(fname) = &callee.kind else {
                    return Err("indirect calls not yet supported in general wasm32 MVP".into());
                };
                for a in args {
                    let _ = self.emit_expr(&a.value)?;
                }
                let Some(sig) = self.sigs.get(fname) else {
                    return Err(format!("unknown wasm32 function `{}`", fname));
                };
                self.code.push(0x10); // call
                push_uleb(&mut self.code, sig.idx);
                if let Some(rt) = sig.ret {
                    Ok(rt)
                } else {
                    self.code.push(0x42);
                    push_sleb64(&mut self.code, 0);
                    Ok(WasmTy::I64)
                }
            }
            _ => Err(format!("unsupported expr in general wasm32 MVP: {:?}", expr.kind)),
        }
    }
}

fn compile_wasm_fn(
    f: &FnDecl,
    sigs: &BTreeMap<String, FnSigWasm>,
    aliases: &BTreeMap<String, TypeExpr>,
) -> Result<Vec<u8>, String> {
    let mut locals = BTreeMap::new();
    for (i, p) in f.params.iter().enumerate() {
        locals.insert(p.name.clone(), (i as u32, ty_to_wasm(&p.ty, aliases)));
    }
    let mut emitter = WasmFnEmitter {
        sigs,
        aliases,
        locals,
        extra_locals: Vec::new(),
        next_local: f.params.len() as u32,
        code: Vec::new(),
        nesting: 0,
        loop_depths: Vec::new(),
    };
    let want_ret = f.ret_ty.is_some();
    let _ = emitter.emit_block(&f.body, want_ret)?;
    emitter.code.push(0x0B); // end function

    let mut body = Vec::new();
    if emitter.extra_locals.is_empty() {
        push_uleb(&mut body, 0);
    } else {
        push_uleb(&mut body, emitter.extra_locals.len() as u32);
        for lt in &emitter.extra_locals {
            push_uleb(&mut body, 1);
            body.push(lt.byte());
        }
    }
    body.extend_from_slice(&emitter.code);
    Ok(body)
}

fn emit_section(out: &mut Vec<u8>, id: u8, payload: &[u8]) {
    out.push(id);
    push_uleb(out, payload.len() as u32);
    out.extend_from_slice(payload);
}

fn push_uleb(out: &mut Vec<u8>, mut v: u32) {
    loop {
        let mut b = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            b |= 0x80;
        }
        out.push(b);
        if v == 0 {
            break;
        }
    }
}

fn push_sleb64(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let b = (v & 0x7F) as u8;
        v >>= 7;
        let done = (v == 0 && (b & 0x40) == 0) || (v == -1 && (b & 0x40) != 0);
        out.push(if done { b } else { b | 0x80 });
        if done {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    fn general_wasm32_compiles_and_executes_fib_while_and_refinements_in_v8() {
        let prog = tenon_syntax::parse(r#"
pub type Range[LO Int, HI Int] = Int where self >= LO && self <= HI

pub pure fn fib(n Int) Int {
    if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
}

pub pure fn sum_to(n Int) Int {
    var i = 1
    var acc = 0
    while i <= n {
        acc += i
        i += 1
    }
    acc
}

pub pure fn widen_val(x Range[3, 5]) Range[2, 6] {
    refine_cast[Range[2, 6]](x)
}
"#)
        .unwrap();
        let rep = tenon_verify::check_program(&prog);
        assert!(rep.ok, "verify failed: {:?}", rep.diagnostics);

        let bytes = compile_program_to_wasm(&prog).expect("general wasm32 compile");
        assert!(bytes.starts_with(b"\0asm"));

        let tmp = std::env::temp_dir().join(format!("tenon_gen_wasm_{}.wasm", std::process::id()));
        std::fs::write(&tmp, &bytes).unwrap();
        let js = format!(
            "const fs=require('fs');const buf=fs.readFileSync({:?});WebAssembly.instantiate(buf).then(({{instance}})=>{{const e=instance.exports;const r=[Number(e.fib(10n)),Number(e.sum_to(100n)),Number(e.widen_val(4n))];console.log(JSON.stringify(r));if(r[0]!==55||r[1]!==5050||r[2]!==4)process.exit(1);}});",
            tmp.to_str().unwrap()
        );
        let out = Command::new("node").args(["-e", &js]).output().unwrap();
        let _ = std::fs::remove_file(&tmp);
        assert!(
            out.status.success(),
            "Node V8 execution of general wasm32 failed: stdout={} stderr={}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
