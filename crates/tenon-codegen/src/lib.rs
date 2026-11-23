use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{AbiParam, InstBuilder, Type, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI64, Ordering};
use tenon_syntax::*;

static LAST_ASSERT_FAILED: AtomicI64 = AtomicI64::new(0);
static LAST_PRINTED: AtomicI64 = AtomicI64::new(0);

extern "C" fn tenon_rt_assert_fail(line: i64) {
    LAST_ASSERT_FAILED.store(line, Ordering::SeqCst);
}

extern "C" fn tenon_rt_print_int(val: i64) -> i64 {
    LAST_PRINTED.store(val, Ordering::SeqCst);
    println!("{}", val);
    val
}

pub struct JitProgram {
    module: JITModule,
    func_ids: BTreeMap<String, FuncId>,
    pub compiled_fns: Vec<String>,
}

impl JitProgram {
    pub fn compile(program: &Program) -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|e| e.to_string())?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| e.to_string())?;
        let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| e.to_string())?;

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        jit_builder.symbol("tenon_rt_assert_fail", tenon_rt_assert_fail as *const u8);
        jit_builder.symbol("tenon_rt_print_int", tenon_rt_print_int as *const u8);

        let mut module = JITModule::new(jit_builder);

        // Declare runtime helpers
        let mut assert_sig = module.make_signature();
        assert_sig.params.push(AbiParam::new(types::I64));
        let assert_fail_id = module
            .declare_function("tenon_rt_assert_fail", Linkage::Import, &assert_sig)
            .map_err(|e| e.to_string())?;

        let mut print_sig = module.make_signature();
        print_sig.params.push(AbiParam::new(types::I64));
        print_sig.returns.push(AbiParam::new(types::I64));
        let print_int_id = module
            .declare_function("tenon_rt_print_int", Linkage::Import, &print_sig)
            .map_err(|e| e.to_string())?;

        // Collect compilable functions
        let fns: Vec<&FnDecl> = program
            .items
            .iter()
            .filter_map(|i| match i {
                Item::Fn(f) if is_jit_supported_fn(f) => Some(f),
                _ => None,
            })
            .collect();

        // First pass: declare all signatures
        let mut func_ids = BTreeMap::new();
        func_ids.insert("print_int".into(), print_int_id);

        for f in &fns {
            let mut sig = module.make_signature();
            for _ in &f.params {
                sig.params.push(AbiParam::new(types::I64));
            }
            sig.returns.push(AbiParam::new(types::I64));
            let id = module
                .declare_function(&f.name, Linkage::Export, &sig)
                .map_err(|e| e.to_string())?;
            func_ids.insert(f.name.clone(), id);
        }

        // Second pass: compile bodies to Cranelift IR -> native machine code!
        let mut ctx = module.make_context();
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut compiled_fns = Vec::new();

        for f in &fns {
            let id = func_ids[&f.name];
            ctx.func.signature = module.declarations().get_function_decl(id).signature.clone();

            {
                let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
                let entry_block = builder.create_block();
                builder.append_block_params_for_function_params(entry_block);
                builder.switch_to_block(entry_block);
                builder.seal_block(entry_block);

                let mut vars = BTreeMap::new();
                for (idx, p) in f.params.iter().enumerate() {
                    let var = builder.declare_var(types::I64);
                    let val = builder.block_params(entry_block)[idx];
                    builder.def_var(var, val);
                    vars.insert(p.name.clone(), var);
                }

                let mut lower = FnLower {
                    module: &mut module,
                    builder: &mut builder,
                    func_ids: &func_ids,
                    assert_fail_id,
                    vars,
                    terminated: false,
                };
                let ret_val = lower.lower_block(&f.body)?;
                if !lower.terminated {
                    lower.builder.ins().return_(&[ret_val]);
                }
                builder.finalize();
            }

            module
                .define_function(id, &mut ctx)
                .map_err(|e| format!("Cranelift codegen failed for `{}`: {}", f.name, e))?;
            module.clear_context(&mut ctx);
            compiled_fns.push(f.name.clone());
        }

        module.finalize_definitions().map_err(|e| e.to_string())?;
        Ok(Self {
            module,
            func_ids,
            compiled_fns,
        })
    }

    pub fn call_0(&self, name: &str) -> Result<i64, String> {
        LAST_ASSERT_FAILED.store(0, Ordering::SeqCst);
        let id = *self
            .func_ids
            .get(name)
            .ok_or_else(|| format!("function `{}` not compiled", name))?;
        let ptr = self.module.get_finalized_function(id);
        let f: extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
        let res = f();
        let failed = LAST_ASSERT_FAILED.load(Ordering::SeqCst);
        if failed != 0 {
            return Err(format!("assertion failed at line {}", failed));
        }
        Ok(res)
    }

    pub fn call_2(&self, name: &str, a: i64, b: i64) -> Result<i64, String> {
        LAST_ASSERT_FAILED.store(0, Ordering::SeqCst);
        let id = *self
            .func_ids
            .get(name)
            .ok_or_else(|| format!("function `{}` not compiled", name))?;
        let ptr = self.module.get_finalized_function(id);
        let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        let res = f(a, b);
        let failed = LAST_ASSERT_FAILED.load(Ordering::SeqCst);
        if failed != 0 {
            return Err(format!("assertion failed at line {}", failed));
        }
        Ok(res)
    }

    pub fn last_printed() -> i64 {
        LAST_PRINTED.load(Ordering::SeqCst)
    }
}

fn is_jit_supported_fn(f: &FnDecl) -> bool {
    if f.is_async || !f.type_params.is_empty() {
        return false;
    }
    for p in &f.params {
        if p.is_inout {
            return false;
        }
        if matches!(p.ty.display_name().as_str(), "Date" | "DateRange" | "Order" | "Text") {
            return false;
        }
    }
    if let Some(rt) = &f.ret_ty {
        if matches!(rt.display_name().as_str(), "Date" | "DateRange" | "Order" | "Text") {
            return false;
        }
    }
    true
}

struct FnLower<'a, 'b> {
    module: &'a mut JITModule,
    builder: &'a mut FunctionBuilder<'b>,
    func_ids: &'a BTreeMap<String, FuncId>,
    assert_fail_id: FuncId,
    vars: BTreeMap<String, Variable>,
    terminated: bool,
}

impl<'a, 'b> FnLower<'a, 'b> {
    fn declare_var(&mut self, name: &str, ty: Type) -> Variable {
        if let Some(&v) = self.vars.get(name) {
            return v;
        }
        let v = self.builder.declare_var(ty);
        self.vars.insert(name.to_string(), v);
        v
    }

    fn lower_block(&mut self, blk: &Block) -> Result<cranelift_codegen::ir::Value, String> {
        let mut last = self.builder.ins().iconst(types::I64, 0);
        for stmt in &blk.stmts {
            if self.terminated {
                break;
            }
            match stmt {
                Stmt::Let { name, value, .. } => {
                    let v = self.lower_expr(value)?;
                    let var = self.declare_var(name, types::I64);
                    self.builder.def_var(var, v);
                }
                Stmt::Assign {
                    target, op, value, ..
                } => {
                    if let ExprKind::Ident(name) = &target.kind {
                        let rhs = self.lower_expr(value)?;
                        let var = self.declare_var(name, types::I64);
                        let new_v = match op {
                            AssignOp::Set => rhs,
                            AssignOp::AddSet => {
                                let cur = self.builder.use_var(var);
                                self.builder.ins().iadd(cur, rhs)
                            }
                            AssignOp::SubSet => {
                                let cur = self.builder.use_var(var);
                                self.builder.ins().isub(cur, rhs)
                            }
                        };
                        self.builder.def_var(var, new_v);
                    }
                }
                Stmt::Return { value, .. } => {
                    let v = match value {
                        Some(e) => self.lower_expr(e)?,
                        None => self.builder.ins().iconst(types::I64, 0),
                    };
                    self.builder.ins().return_(&[v]);
                    self.terminated = true;
                    return Ok(v);
                }
                Stmt::Assert { cond, span } => {
                    let c = self.lower_expr(cond)?;
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    let is_ok = self.builder.ins().icmp(IntCC::NotEqual, c, zero);
                    let fail_blk = self.builder.create_block();
                    let cont_blk = self.builder.create_block();
                    self.builder
                        .ins()
                        .brif(is_ok, cont_blk, &[], fail_blk, &[]);
                    self.builder.switch_to_block(fail_blk);
                    self.builder.seal_block(fail_blk);
                    let callee = self
                        .module
                        .declare_func_in_func(self.assert_fail_id, self.builder.func);
                    let line_val = self.builder.ins().iconst(types::I64, span.line as i64);
                    self.builder.ins().call(callee, &[line_val]);
                    self.builder.ins().jump(cont_blk, &[]);
                    self.builder.switch_to_block(cont_blk);
                    self.builder.seal_block(cont_blk);
                }
                Stmt::While { cond, body, .. } => {
                    let hdr = self.builder.create_block();
                    let loop_blk = self.builder.create_block();
                    let exit_blk = self.builder.create_block();
                    self.builder.ins().jump(hdr, &[]);
                    self.builder.switch_to_block(hdr);
                    let c = self.lower_expr(cond)?;
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    let cmp = self.builder.ins().icmp(IntCC::NotEqual, c, zero);
                    self.builder
                        .ins()
                        .brif(cmp, loop_blk, &[], exit_blk, &[]);
                    self.builder.switch_to_block(loop_blk);
                    self.builder.seal_block(loop_blk);
                    let _ = self.lower_block(body)?;
                    if !self.terminated {
                        self.builder.ins().jump(hdr, &[]);
                    }
                    self.terminated = false;
                    self.builder.seal_block(hdr);
                    self.builder.switch_to_block(exit_blk);
                    self.builder.seal_block(exit_blk);
                }
                Stmt::Expr { expr, .. } => {
                    last = self.lower_expr(expr)?;
                }
                _ => {}
            }
        }
        if !self.terminated {
            if let Some(tail) = &blk.tail {
                last = self.lower_expr(tail)?;
            }
        }
        Ok(last)
    }

    fn lower_expr(&mut self, expr: &Expr) -> Result<cranelift_codegen::ir::Value, String> {
        match &expr.kind {
            ExprKind::Int(n) => Ok(self.builder.ins().iconst(types::I64, *n)),
            ExprKind::Bool(b) => Ok(self
                .builder
                .ins()
                .iconst(types::I64, if *b { 1 } else { 0 })),
            ExprKind::Ident(s) => {
                let var = *self
                    .vars
                    .get(s)
                    .ok_or_else(|| format!("undefined variable `{}` in codegen", s))?;
                Ok(self.builder.use_var(var))
            }
            ExprKind::Unary { op, expr } => {
                let v = self.lower_expr(expr)?;
                match op {
                    UnOp::Neg => Ok(self.builder.ins().ineg(v)),
                    UnOp::Not => {
                        let zero = self.builder.ins().iconst(types::I64, 0);
                        let eq = self.builder.ins().icmp(IntCC::Equal, v, zero);
                        Ok(self.builder.ins().uextend(types::I64, eq))
                    }
                }
            }
            ExprKind::Binary { op, left, right } => {
                let l = self.lower_expr(left)?;
                let r = self.lower_expr(right)?;
                let v = match op {
                    BinOp::Add => self.builder.ins().iadd(l, r),
                    BinOp::Sub => self.builder.ins().isub(l, r),
                    BinOp::Mul => self.builder.ins().imul(l, r),
                    BinOp::Div => self.builder.ins().sdiv(l, r),
                    BinOp::Mod => self.builder.ins().srem(l, r),
                    BinOp::BitAnd => self.builder.ins().band(l, r),
                    BinOp::BitOr => self.builder.ins().bor(l, r),
                    BinOp::Eq => {
                        let c = self.builder.ins().icmp(IntCC::Equal, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::Ne => {
                        let c = self.builder.ins().icmp(IntCC::NotEqual, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::Lt => {
                        let c = self.builder.ins().icmp(IntCC::SignedLessThan, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::Le => {
                        let c = self.builder.ins().icmp(IntCC::SignedLessThanOrEqual, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::Gt => {
                        let c = self.builder.ins().icmp(IntCC::SignedGreaterThan, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::Ge => {
                        let c = self
                            .builder
                            .ins()
                            .icmp(IntCC::SignedGreaterThanOrEqual, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::And => {
                        let z = self.builder.ins().iconst(types::I64, 0);
                        let lb = self.builder.ins().icmp(IntCC::NotEqual, l, z);
                        let rb = self.builder.ins().icmp(IntCC::NotEqual, r, z);
                        let and = self.builder.ins().band(lb, rb);
                        self.builder.ins().uextend(types::I64, and)
                    }
                    BinOp::Or => {
                        let z = self.builder.ins().iconst(types::I64, 0);
                        let lb = self.builder.ins().icmp(IntCC::NotEqual, l, z);
                        let rb = self.builder.ins().icmp(IntCC::NotEqual, r, z);
                        let or = self.builder.ins().bor(lb, rb);
                        self.builder.ins().uextend(types::I64, or)
                    }
                    other => return Err(format!("unsupported binary op {:?} in codegen", other)),
                };
                Ok(v)
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                let c = self.lower_expr(cond)?;
                let z = self.builder.ins().iconst(types::I64, 0);
                let cmp = self.builder.ins().icmp(IntCC::NotEqual, c, z);
                let then_b = self.builder.create_block();
                let else_b = self.builder.create_block();
                let merge_b = self.builder.create_block();
                self.builder.append_block_param(merge_b, types::I64);
                self.builder
                    .ins()
                    .brif(cmp, then_b, &[], else_b, &[]);

                self.builder.switch_to_block(then_b);
                self.builder.seal_block(then_b);
                let tv = self.lower_block(then_block)?;
                if !self.terminated {
                    self.builder.ins().jump(merge_b, &[tv.into()]);
                }
                self.terminated = false;

                self.builder.switch_to_block(else_b);
                self.builder.seal_block(else_b);
                let ev = match else_branch {
                    Some(eb) => self.lower_expr(eb)?,
                    None => self.builder.ins().iconst(types::I64, 0),
                };
                if !self.terminated {
                    self.builder.ins().jump(merge_b, &[ev.into()]);
                }
                self.terminated = false;

                self.builder.switch_to_block(merge_b);
                self.builder.seal_block(merge_b);
                Ok(self.builder.block_params(merge_b)[0])
            }
            ExprKind::Match { scrutinee, arms } => {
                let sv = self.lower_expr(scrutinee)?;
                let merge_b = self.builder.create_block();
                self.builder.append_block_param(merge_b, types::I64);

                for arm in arms {
                    let is_wild = arm
                        .patterns
                        .iter()
                        .any(|p| matches!(p, Pattern::Wildcard(_)));
                    if is_wild {
                        let v = self.lower_expr(&arm.body)?;
                        self.builder.ins().jump(merge_b, &[v.into()]);
                        self.builder.switch_to_block(merge_b);
                        self.builder.seal_block(merge_b);
                        return Ok(self.builder.block_params(merge_b)[0]);
                    }
                    let mut cond: Option<cranelift_codegen::ir::Value> = None;
                    for p in &arm.patterns {
                        let pv = match p {
                            Pattern::Int(n, _) => self.builder.ins().iconst(types::I64, *n),
                            Pattern::Bool(b, _) => {
                                self.builder
                                    .ins()
                                    .iconst(types::I64, if *b { 1 } else { 0 })
                            }
                            _ => continue,
                        };
                        let eq = self.builder.ins().icmp(IntCC::Equal, sv, pv);
                        cond = Some(match cond {
                            Some(prev) => self.builder.ins().bor(prev, eq),
                            None => eq,
                        });
                    }
                    let matched_b = self.builder.create_block();
                    let next_b = self.builder.create_block();
                    let c = cond.unwrap_or_else(|| self.builder.ins().iconst(types::I8, 0));
                    self.builder
                        .ins()
                        .brif(c, matched_b, &[], next_b, &[]);

                    self.builder.switch_to_block(matched_b);
                    self.builder.seal_block(matched_b);
                    let v = self.lower_expr(&arm.body)?;
                    self.builder.ins().jump(merge_b, &[v.into()]);

                    self.builder.switch_to_block(next_b);
                    self.builder.seal_block(next_b);
                }

                let zero = self.builder.ins().iconst(types::I64, 0);
                self.builder.ins().jump(merge_b, &[zero.into()]);
                self.builder.switch_to_block(merge_b);
                self.builder.seal_block(merge_b);
                Ok(self.builder.block_params(merge_b)[0])
            }
            ExprKind::Block(blk) => self.lower_block(blk),
            ExprKind::Call { callee, args } => {
                if let ExprKind::Ident(fname) = &callee.kind {
                    let fid = *self
                        .func_ids
                        .get(fname)
                        .ok_or_else(|| format!("unknown function `{}` in codegen", fname))?;
                    let mut arg_vals = Vec::new();
                    for a in args {
                        arg_vals.push(self.lower_expr(&a.value)?);
                    }
                    let local_callee = self.module.declare_func_in_func(fid, self.builder.func);
                    let call = self.builder.ins().call(local_callee, &arg_vals);
                    Ok(self.builder.inst_results(call)[0])
                } else {
                    Err("unsupported indirect call in M0 codegen".into())
                }
            }
            _ => Ok(self.builder.ins().iconst(types::I64, 0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenon_syntax::parse;

    #[test]
    fn cranelift_jits_and_runs_days_in_month_and_add_odds() {
        let src = r#"
pub type Odd = Int64 where self % 2 != 0
pub type Even = Int64 where self % 2 == 0

pub pure fn add_odds(a Odd, b Odd) Even {
    a + b
}

pub pure fn is_leap_year(y Int) Bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

pub pure fn days_in_month(y Int, m Int) Int {
    match m {
        2 => if is_leap_year(y) { 29 } else { 28 }
        4 | 6 | 9 | 11 => 30
        _ => 31
    }
}

pub fn main() Int {
    assert days_in_month(2024, 2) == 29
    assert days_in_month(2023, 2) == 28
    assert days_in_month(1900, 2) == 28
    assert days_in_month(2000, 2) == 29
    assert add_odds(3, 5) == 8
    days_in_month(2024, 2) + add_odds(3, 5)
}
"#;
        let prog = parse(src).unwrap();
        let jit = JitProgram::compile(&prog).unwrap();
        assert_eq!(jit.call_2("days_in_month", 2024, 2).unwrap(), 29);
        assert_eq!(jit.call_2("days_in_month", 1900, 2).unwrap(), 28);
        assert_eq!(jit.call_2("add_odds", 7, 9).unwrap(), 16);
        assert_eq!(jit.call_0("main").unwrap(), 37);
    }
}
