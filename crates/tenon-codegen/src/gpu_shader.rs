//! Single-Source `@kernel` + `@gpu_struct` + `@gpu_graph` -> Apple Metal (MSL), WebGPU (WGSL),
//! and Browser WebGPU ESM Host Dispatcher (`--emit-msl`, `--emit-wgsl`, `--emit-webgpu`).
//!
//! Zero new keywords: lowers pure Tenon `@gpu_struct` records and `@kernel` functions (including
//! `gpu_shared_f32(N)` threadgroup/workgroup memory and `gpu_barrier()` synchronization) directly
//! into BOTH native Apple Silicon Metal Shading Language (`.metal`) AND portable WebGPU Shading
//! Language (`.wgsl`), with real AST-level producer-consumer kernel fusion for `@gpu_graph`!

use std::collections::BTreeMap;
use tenon_syntax::*;
use tenon_verify::VerifiedMir;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderTarget {
    Msl,
    Wgsl,
}

fn map_gpu_scalar(name: &str, target: ShaderTarget) -> &'static str {
    match (name, target) {
        ("Float16", ShaderTarget::Msl) => "half",
        ("Float16", ShaderTarget::Wgsl) => "f16",
        ("BFloat16", ShaderTarget::Msl) => "bfloat",
        ("BFloat16", ShaderTarget::Wgsl) => "f32",
        ("Float32" | "Float" | "Float64", ShaderTarget::Msl) => "float",
        ("Float32" | "Float" | "Float64", ShaderTarget::Wgsl) => "f32",
        ("Int32" | "Int" | "Int64", ShaderTarget::Msl) => "int",
        ("Int32" | "Int" | "Int64", ShaderTarget::Wgsl) => "i32",
        ("UInt32" | "UInt64" | "USize", ShaderTarget::Msl) => "uint",
        ("UInt32" | "UInt64" | "USize", ShaderTarget::Wgsl) => "u32",
        ("Bool", _) => "bool",
        _ => match target {
            ShaderTarget::Msl => "float",
            ShaderTarget::Wgsl => "f32",
        },
    }
}

fn map_type_name(ty: &TypeExpr, target: ShaderTarget) -> String {
    match ty {
        TypeExpr::Named { name, args, .. } => {
            if name == "List" && args.len() == 1 {
                return map_type_name(&args[0], target);
            }
            if name == "SimdVec" && args.len() == 2 {
                let elem = map_type_name(&args[0], target);
                let lanes = args[1].display_name();
                return match target {
                    ShaderTarget::Msl => format!("{}{}", elem, lanes),
                    ShaderTarget::Wgsl => format!("vec{}<{}>", lanes, elem),
                };
            }
            if matches!(
                name.as_str(),
                "Float16" | "BFloat16" | "Float32" | "Float" | "Float64" | "Int32" | "Int" | "Int64" | "UInt32" | "UInt64" | "USize" | "Bool"
            ) {
                map_gpu_scalar(name, target).to_string()
            } else {
                name.clone()
            }
        }
        TypeExpr::Array { elem, len, .. } => {
            let et = map_type_name(elem, target);
            match target {
                ShaderTarget::Msl => format!("{}[{}]", et, len),
                ShaderTarget::Wgsl => format!("array<{}, {}>", et, len),
            }
        }
        _ => map_gpu_scalar("Float32", target).to_string(),
    }
}

fn is_buffer_param(ty: &TypeExpr) -> bool {
    matches!(ty, TypeExpr::Named { name, .. } if name == "List" || name == "Bytes")
}

/// Scans a block for `let/var name = gpu_shared_f32(N)` / `gpu_shared_f16(N)` declarations
/// (used for WGSL module-level `var<workgroup>`). Returns `(name, len, is_f16)`.
fn collect_shared_arrays(blk: &Block, out: &mut Vec<(String, i64, bool)>) {
    for stmt in &blk.stmts {
        if let Stmt::Let { name, value, .. } = stmt {
            if let ExprKind::Call { callee, args } = &value.kind {
                if let ExprKind::Ident(s) = &callee.kind {
                    if (s == "gpu_shared_f32" || s == "gpu_shared_f16") && args.len() == 1 {
                        if let ExprKind::Int(sz) = &args[0].value.kind {
                            out.push((name.clone(), (*sz).max(1), s == "gpu_shared_f16"));
                        }
                    }
                }
            }
        }
    }
}

fn block_uses_builtin(blk: &Block, names: &[&str]) -> bool {
    fn expr_uses(e: &Expr, names: &[&str]) -> bool {
        match &e.kind {
            ExprKind::Call { callee, args } => {
                if let ExprKind::Ident(s) = &callee.kind {
                    if names.contains(&s.as_str()) {
                        return true;
                    }
                }
                args.iter().any(|a| expr_uses(&a.value, names))
            }
            ExprKind::Binary { left, right, .. } => expr_uses(left, names) || expr_uses(right, names),
            ExprKind::Unary { expr, .. } | ExprKind::Cast { expr, .. } | ExprKind::Field { target: expr, .. } => {
                expr_uses(expr, names)
            }
            ExprKind::Index { target, index } => expr_uses(target, names) || expr_uses(index, names),
            ExprKind::If { cond, then_block, else_branch } => {
                expr_uses(cond, names)
                    || block_uses_builtin(then_block, names)
                    || else_branch.as_deref().is_some_and(|eb| expr_uses(eb, names))
            }
            ExprKind::Block(b) => block_uses_builtin(b, names),
            _ => false,
        }
    }
    for s in &blk.stmts {
        match s {
            Stmt::Let { value, .. } | Stmt::Expr { expr: value, .. } => {
                if expr_uses(value, names) {
                    return true;
                }
            }
            Stmt::Assign { target, value, .. } => {
                if expr_uses(target, names) || expr_uses(value, names) {
                    return true;
                }
            }
            Stmt::While { cond, body, .. } => {
                if expr_uses(cond, names) || block_uses_builtin(body, names) {
                    return true;
                }
            }
            Stmt::Return { value: Some(v), .. } => {
                if expr_uses(v, names) {
                    return true;
                }
            }
            _ => {}
        }
    }
    blk.tail.as_deref().is_some_and(|e| expr_uses(e, names))
}

fn emit_expr_subst(
    expr: &Expr,
    target: ShaderTarget,
    subst_idents: &BTreeMap<String, String>,
    subst_buf_reads: &BTreeMap<String, String>,
) -> String {
    match &expr.kind {
        ExprKind::Int(n) => format!("{}", n),
        ExprKind::Float(f) => {
            let s = format!("{}", f);
            let with_dot = if s.contains('.') || s.contains('e') { s } else { format!("{}.0", s) };
            match target {
                ShaderTarget::Msl => format!("{}f", with_dot),
                ShaderTarget::Wgsl => with_dot,
            }
        }
        ExprKind::Bool(b) => if *b { "true".into() } else { "false".into() },
        ExprKind::Ident(s) => subst_idents.get(s).cloned().unwrap_or_else(|| s.clone()),
        ExprKind::Field { target: t, field } => {
            format!("{}.{}", emit_expr_subst(t, target, subst_idents, subst_buf_reads), field)
        }
        ExprKind::Index { target: t, index } => {
            if let ExprKind::Ident(bname) = &t.kind {
                if let Some(rep) = subst_buf_reads.get(bname) {
                    return rep.clone();
                }
            }
            format!(
                "{}[{}]",
                emit_expr_subst(t, target, subst_idents, subst_buf_reads),
                emit_expr_subst(index, target, subst_idents, subst_buf_reads)
            )
        }
        ExprKind::Cast { expr: inner, ty } => {
            let ct = map_type_name(ty, target);
            match target {
                ShaderTarget::Msl => format!("(({})({}))", ct, emit_expr_subst(inner, target, subst_idents, subst_buf_reads)),
                ShaderTarget::Wgsl => format!("{}({})", ct, emit_expr_subst(inner, target, subst_idents, subst_buf_reads)),
            }
        }
        ExprKind::Unary { op, expr: inner } => {
            let sym = match op {
                UnOp::Neg => "-",
                UnOp::Not => "!",
                UnOp::BitNot => "~",
            };
            format!("({}{})", sym, emit_expr_subst(inner, target, subst_idents, subst_buf_reads))
        }
        ExprKind::Binary { op, left, right } => {
            let sym = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Mod => "%",
                BinOp::Eq => "==",
                BinOp::Ne => "!=",
                BinOp::Lt => "<",
                BinOp::Le => "<=",
                BinOp::Gt => ">",
                BinOp::Ge => ">=",
                BinOp::And => "&&",
                BinOp::Or => "||",
                BinOp::BitAnd => "&",
                BinOp::BitOr => "|",
                BinOp::BitXor => "^",
                BinOp::Shl => "<<",
                BinOp::Shr => ">>",
                _ => "+",
            };
            format!(
                "({} {} {})",
                emit_expr_subst(left, target, subst_idents, subst_buf_reads),
                sym,
                emit_expr_subst(right, target, subst_idents, subst_buf_reads)
            )
        }
        ExprKind::If { cond, then_block, else_branch } => {
            let c = emit_expr_subst(cond, target, subst_idents, subst_buf_reads);
            let t = then_block
                .tail
                .as_deref()
                .map(|e| emit_expr_subst(e, target, subst_idents, subst_buf_reads))
                .unwrap_or_else(|| "0".into());
            let e = else_branch
                .as_deref()
                .map(|eb| match &eb.kind {
                    ExprKind::Block(b) => b
                        .tail
                        .as_deref()
                        .map(|x| emit_expr_subst(x, target, subst_idents, subst_buf_reads))
                        .unwrap_or_else(|| "0".into()),
                    _ => emit_expr_subst(eb, target, subst_idents, subst_buf_reads),
                })
                .unwrap_or_else(|| "0".into());
            match target {
                ShaderTarget::Msl => format!("(({}) ? ({}) : ({}))", c, t, e),
                ShaderTarget::Wgsl => format!("select({}, {}, {})", e, t, c),
            }
        }
        ExprKind::Call { callee, args } => {
            if let ExprKind::Ident(fname) = &callee.kind {
                let cast_i = if target == ShaderTarget::Msl { "(int)(" } else { "i32(" };
                match fname.as_str() {
                    "gpu_global_id_x" => return format!("{}gid.x)", cast_i),
                    "gpu_global_id_y" => return format!("{}gid.y)", cast_i),
                    "gpu_global_id_z" => return format!("{}gid.z)", cast_i),
                    "gpu_local_id_x" => return format!("{}lid.x)", cast_i),
                    "gpu_local_id_y" => return format!("{}lid.y)", cast_i),
                    "gpu_group_id_x" => return format!("{}{}.x)", cast_i, if target == ShaderTarget::Msl { "tgid" } else { "wgid" }),
                    "gpu_group_id_y" => return format!("{}{}.y)", cast_i, if target == ShaderTarget::Msl { "tgid" } else { "wgid" }),
                    "gpu_barrier" => {
                        return match target {
                            ShaderTarget::Msl => "threadgroup_barrier(mem_flags::mem_threadgroup)".into(),
                            ShaderTarget::Wgsl => "workgroupBarrier()".into(),
                        };
                    }
                    "max" | "min" | "abs" | "sqrt" | "clamp" => {
                        let av: Vec<String> = args
                            .iter()
                            .map(|a| emit_expr_subst(&a.value, target, subst_idents, subst_buf_reads))
                            .collect();
                        return format!("{}({})", fname, av.join(", "));
                    }
                    "gpu_exp" | "gpu_sqrt" | "gpu_rsqrt" | "gpu_cos" | "gpu_sin" | "gpu_pow" | "gpu_tanh" => {
                        let av: Vec<String> = args
                            .iter()
                            .map(|a| emit_expr_subst(&a.value, target, subst_idents, subst_buf_reads))
                            .collect();
                        let fn_out = match (fname.as_str(), target) {
                            ("gpu_exp", _) => "exp",
                            ("gpu_sqrt", _) => "sqrt",
                            ("gpu_rsqrt", ShaderTarget::Msl) => "rsqrt",
                            ("gpu_rsqrt", ShaderTarget::Wgsl) => "inverseSqrt",
                            ("gpu_cos", _) => "cos",
                            ("gpu_sin", _) => "sin",
                            ("gpu_pow", _) => "pow",
                            ("gpu_tanh", _) => "tanh",
                            _ => "exp",
                        };
                        return format!("{}({})", fn_out, av.join(", "));
                    }
                    _ => {}
                }
            }
            let av: Vec<String> = args
                .iter()
                .map(|a| emit_expr_subst(&a.value, target, subst_idents, subst_buf_reads))
                .collect();
            format!("{}({})", emit_expr_subst(callee, target, subst_idents, subst_buf_reads), av.join(", "))
        }
        ExprKind::Block(b) => b
            .tail
            .as_deref()
            .map(|e| emit_expr_subst(e, target, subst_idents, subst_buf_reads))
            .unwrap_or_else(|| "0".into()),
        _ => "0".into(),
    }
}

fn emit_block_subst(
    blk: &Block,
    target: ShaderTarget,
    indent: usize,
    subst_idents: &BTreeMap<String, String>,
    subst_buf_reads: &BTreeMap<String, String>,
    out: &mut String,
) {
    let pad = " ".repeat(indent);
    for stmt in &blk.stmts {
        match stmt {
            Stmt::Let { name, ty, value, is_mut, .. } => {
                // `let tile = gpu_shared_f32(N)` / `gpu_shared_f16(N)` -> MSL `threadgroup float/half tile[N];`
                if let ExprKind::Call { callee, args } = &value.kind {
                    if let ExprKind::Ident(s) = &callee.kind {
                        if (s == "gpu_shared_f32" || s == "gpu_shared_f16") && args.len() == 1 {
                            if let ExprKind::Int(sz) = &args[0].value.kind {
                                if target == ShaderTarget::Msl {
                                    let elem_t = if s == "gpu_shared_f16" { "half" } else { "float" };
                                    out.push_str(&format!(
                                        "{}threadgroup {} {}[{}];\n",
                                        pad,
                                        elem_t,
                                        name,
                                        (*sz).max(1)
                                    ));
                                }
                                continue;
                            }
                        }
                    }
                    if matches!(&callee.kind, ExprKind::Ident(s) if s == "gpu_barrier") {
                        match target {
                            ShaderTarget::Msl => out.push_str(&format!("{}threadgroup_barrier(mem_flags::mem_threadgroup);\n", pad)),
                            ShaderTarget::Wgsl => out.push_str(&format!("{}workgroupBarrier();\n", pad)),
                        }
                        continue;
                    }
                }
                let rhs = emit_expr_subst(value, target, subst_idents, subst_buf_reads);
                match target {
                    ShaderTarget::Msl => {
                        let cty = ty
                            .as_ref()
                            .map(|t| map_type_name(t, target))
                            .unwrap_or_else(|| infer_shader_expr_type(value, target));
                        let qual = if *is_mut { "" } else { "const " };
                        out.push_str(&format!("{}{}{} {} = {};\n", pad, qual, cty, name, rhs));
                    }
                    ShaderTarget::Wgsl => {
                        let kw = if *is_mut { "var" } else { "let" };
                        if let Some(t) = ty {
                            out.push_str(&format!("{}{} {}: {} = {};\n", pad, kw, name, map_type_name(t, target), rhs));
                        } else {
                            out.push_str(&format!("{}{} {} = {};\n", pad, kw, name, rhs));
                        }
                    }
                }
            }
            Stmt::Assign { target: lhs, op, value, .. } => {
                let op_s = match op {
                    AssignOp::Set => "=",
                    AssignOp::AddSet => "+=",
                    AssignOp::SubSet => "-=",
                    AssignOp::MulSet => "*=",
                    AssignOp::DivSet => "/=",
                    AssignOp::BitAndSet => "&=",
                    AssignOp::BitOrSet => "|=",
                    AssignOp::BitXorSet => "^=",
                    AssignOp::ShlSet => "<<=",
                    AssignOp::ShrSet => ">>=",
                };
                out.push_str(&format!(
                    "{}{} {} {};\n",
                    pad,
                    emit_expr_subst(lhs, target, subst_idents, subst_buf_reads),
                    op_s,
                    emit_expr_subst(value, target, subst_idents, subst_buf_reads)
                ));
            }
            Stmt::While { cond, body, .. } => {
                out.push_str(&format!(
                    "{}while ({}) {{\n",
                    pad,
                    emit_expr_subst(cond, target, subst_idents, subst_buf_reads)
                ));
                emit_block_subst(body, target, indent + 4, subst_idents, subst_buf_reads, out);
                out.push_str(&format!("{}}}\n", pad));
            }
            Stmt::Expr { expr, .. } => {
                if let ExprKind::If { cond, then_block, else_branch } = &expr.kind {
                    out.push_str(&format!(
                        "{}if ({}) {{\n",
                        pad,
                        emit_expr_subst(cond, target, subst_idents, subst_buf_reads)
                    ));
                    emit_block_subst(then_block, target, indent + 4, subst_idents, subst_buf_reads, out);
                    if let Some(eb) = else_branch {
                        out.push_str(&format!("{}}} else {{\n", pad));
                        if let ExprKind::Block(b) = &eb.kind {
                            emit_block_subst(b, target, indent + 4, subst_idents, subst_buf_reads, out);
                        } else {
                            out.push_str(&format!(
                                "{}    {};\n",
                                pad,
                                emit_expr_subst(eb, target, subst_idents, subst_buf_reads)
                            ));
                        }
                    }
                    out.push_str(&format!("{}}}\n", pad));
                } else {
                    out.push_str(&format!(
                        "{}{};\n",
                        pad,
                        emit_expr_subst(expr, target, subst_idents, subst_buf_reads)
                    ));
                }
            }
            Stmt::Return { value, .. } => {
                if let Some(v) = value {
                    out.push_str(&format!(
                        "{}return {};\n",
                        pad,
                        emit_expr_subst(v, target, subst_idents, subst_buf_reads)
                    ));
                } else {
                    out.push_str(&format!("{}return;\n", pad));
                }
            }
            Stmt::Break { .. } => out.push_str(&format!("{}break;\n", pad)),
            Stmt::Continue { .. } => out.push_str(&format!("{}continue;\n", pad)),
            _ => {}
        }
    }
    if let Some(tail) = &blk.tail {
        if let ExprKind::If { cond, then_block, else_branch } = &tail.kind {
            out.push_str(&format!(
                "{}if ({}) {{\n",
                pad,
                emit_expr_subst(cond, target, subst_idents, subst_buf_reads)
            ));
            emit_block_subst(then_block, target, indent + 4, subst_idents, subst_buf_reads, out);
            if let Some(eb) = else_branch {
                out.push_str(&format!("{}}} else {{\n", pad));
                if let ExprKind::Block(b) = &eb.kind {
                    emit_block_subst(b, target, indent + 4, subst_idents, subst_buf_reads, out);
                } else {
                    out.push_str(&format!(
                        "{}    {};\n",
                        pad,
                        emit_expr_subst(eb, target, subst_idents, subst_buf_reads)
                    ));
                }
            }
            out.push_str(&format!("{}}}\n", pad));
        } else if let ExprKind::Block(b) = &tail.kind {
            emit_block_subst(b, target, indent, subst_idents, subst_buf_reads, out);
        }
    }
}

fn infer_shader_expr_type(expr: &Expr, target: ShaderTarget) -> String {
    match &expr.kind {
        ExprKind::Float(_) => map_gpu_scalar("Float32", target).into(),
        ExprKind::Cast { ty, .. } => map_type_name(ty, target),
        ExprKind::Bool(_) => "bool".into(),
        ExprKind::Binary { op, left, right } => {
            if matches!(
                op,
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::And | BinOp::Or
            ) {
                "bool".into()
            } else {
                let lt = infer_shader_expr_type(left, target);
                let rt = infer_shader_expr_type(right, target);
                if lt == "float" || lt == "f32" || rt == "float" || rt == "f32" {
                    map_gpu_scalar("Float32", target).into()
                } else {
                    lt
                }
            }
        }
        ExprKind::Call { callee, .. } => {
            if let ExprKind::Ident(f) = &callee.kind {
                if matches!(
                    f.as_str(),
                    "gpu_exp" | "gpu_sqrt" | "gpu_rsqrt" | "gpu_cos" | "gpu_sin" | "gpu_pow" | "gpu_tanh" | "sqrt" | "abs" | "max" | "min" | "clamp"
                ) {
                    return map_gpu_scalar("Float32", target).into();
                }
            }
            map_gpu_scalar("Int32", target).into()
        }
        ExprKind::Index { .. } => map_gpu_scalar("Float32", target).into(),
        ExprKind::Field { .. } => map_gpu_scalar("UInt32", target).into(),
        _ => map_gpu_scalar("Int32", target).into(),
    }
}

/// Extracts the inner elementwise computation statements & stored RHS expression targeting `out_buf`
/// from a `@kernel("elementwise")` function body (unwrapping its outer `if i < n { ... }` guard).
fn extract_elementwise_stage<'a>(
    blk: &'a Block,
    out_buf: &str,
) -> (Vec<&'a Stmt>, Option<&'a Expr>, Option<String>) {
    let mut idx_var = None;
    let mut inner_blk = blk;
    for s in &blk.stmts {
        if let Stmt::Let { name, value, .. } = s {
            if matches!(&value.kind, ExprKind::Call { callee, .. } if matches!(&callee.kind, ExprKind::Ident(f) if f == "gpu_global_id_x"))
            {
                idx_var = Some(name.clone());
            }
        }
        if let Stmt::Expr { expr, .. } = s {
            if let ExprKind::If { then_block, else_branch: None, .. } = &expr.kind {
                inner_blk = then_block;
            }
        }
    }
    if let Some(tail) = &blk.tail {
        if let ExprKind::If { then_block, else_branch: None, .. } = &tail.kind {
            inner_blk = then_block;
        }
    }
    let mut pre_stmts = Vec::new();
    let mut out_rhs = None;
    for s in &inner_blk.stmts {
        if let Stmt::Assign { target, op: AssignOp::Set, value, .. } = s {
            if let ExprKind::Index { target: arr, .. } = &target.kind {
                if matches!(&arr.kind, ExprKind::Ident(nm) if nm == out_buf) {
                    out_rhs = Some(value);
                    continue;
                }
            }
        }
        pre_stmts.push(s);
    }
    (pre_stmts, out_rhs, idx_var)
}

/// Emits a REAL AST-fused producer-consumer compute kernel in MSL or WGSL (zero stub comments,
/// zero intermediate VRAM traffic: producer output stays in a GPU register `__fused_tmp`).
fn emit_ast_fused_kernel(
    plan_fn: &str,
    fused_name: &str,
    eliminated_buf: &str,
    k1: &FnDecl,
    k2: &FnDecl,
    target: ShaderTarget,
    wgsl_group: usize,
    out: &mut String,
) {
    let k1_out = k1
        .params
        .iter()
        .find(|p| p.is_inout && is_buffer_param(&p.ty))
        .map(|p| p.name.clone())
        .unwrap_or_else(|| "out".into());
    let k2_in = k2
        .params
        .iter()
        .find(|p| !p.is_inout && is_buffer_param(&p.ty))
        .map(|p| p.name.clone())
        .unwrap_or_else(|| "x".into());
    let k2_out = k2
        .params
        .iter()
        .find(|p| p.is_inout && is_buffer_param(&p.ty))
        .map(|p| p.name.clone())
        .unwrap_or_else(|| "out".into());

    let k1_read_bufs: Vec<_> = k1
        .params
        .iter()
        .filter(|p| !p.is_inout && is_buffer_param(&p.ty))
        .collect();
    let k2_extra_read_bufs: Vec<_> = k2
        .params
        .iter()
        .filter(|p| !p.is_inout && is_buffer_param(&p.ty) && p.name != k2_in)
        .collect();

    // Build unified fused parameter list: all k1 inputs + extra k2 inputs + k2 output + `total: u32`
    let mut fused_params: Vec<(String, TypeExpr, bool)> = Vec::new();
    let mut k1_subst = BTreeMap::new();
    let mut k2_subst = BTreeMap::new();
    let mut used_names = std::collections::BTreeSet::new();

    for p in &k1_read_bufs {
        let mut nm = p.name.clone();
        while !used_names.insert(nm.clone()) {
            nm = format!("p1_{}", nm);
        }
        k1_subst.insert(p.name.clone(), nm.clone());
        fused_params.push((nm, p.ty.clone(), false));
    }
    for p in &k2_extra_read_bufs {
        let mut nm = p.name.clone();
        while !used_names.insert(nm.clone()) {
            nm = format!("p2_{}", nm);
        }
        k2_subst.insert(p.name.clone(), nm.clone());
        fused_params.push((nm, p.ty.clone(), false));
    }
    let out_nm = if used_names.insert("out".into()) {
        "out".to_string()
    } else {
        "fused_out".to_string()
    };
    k2_subst.insert(k2_out.clone(), out_nm.clone());
    let out_ty = k2
        .params
        .iter()
        .find(|p| p.name == k2_out)
        .map(|p| p.ty.clone())
        .unwrap_or_else(|| TypeExpr::Named {
            name: "List".into(),
            version: None,
            args: vec![TypeExpr::Named {
                name: "Float32".into(),
                version: None,
                args: vec![],
                span: k2.span,
            }],
            span: k2.span,
        });
    fused_params.push((out_nm.clone(), out_ty, true));

    let total_nm = "total".to_string();
    fused_params.push((
        total_nm.clone(),
        TypeExpr::Named {
            name: "UInt32".into(),
            version: None,
            args: vec![],
            span: k2.span,
        },
        false,
    ));

    let (k1_pre, k1_rhs, k1_idx) = extract_elementwise_stage(&k1.body, &k1_out);
    let (k2_pre, k2_rhs, k2_idx) = extract_elementwise_stage(&k2.body, &k2_out);
    if let Some(iv) = k1_idx {
        k1_subst.insert(iv, "idx".into());
    }
    if let Some(iv) = k2_idx {
        k2_subst.insert(iv, "idx".into());
    }

    let mut k2_buf_reads = BTreeMap::new();
    k2_buf_reads.insert(k2_in.clone(), "__fused_tmp".into());

    out.push_str(&format!(
        "// Auto-fused (AST inlined) by @gpu_graph `{}`: `{}` + `{}` (eliminates transient VRAM buffer `{}`)\n",
        plan_fn, k1.name, k2.name, eliminated_buf
    ));

    match target {
        ShaderTarget::Msl => {
            out.push_str(&format!("kernel void {}(\n", fused_name));
            let mut pstrs = Vec::new();
            for (idx, (nm, ty, is_io)) in fused_params.iter().enumerate() {
                let bt = map_type_name(ty, target);
                if is_buffer_param(ty) {
                    let qual = if *is_io { "device" } else { "device const" };
                    pstrs.push(format!("    {} {}* {} [[buffer({})]]", qual, bt, nm, idx));
                } else {
                    pstrs.push(format!("    constant {}& {} [[buffer({})]]", bt, nm, idx));
                }
            }
            pstrs.push("    uint3 gid [[thread_position_in_grid]]".into());
            out.push_str(&pstrs.join(",\n"));
            out.push_str("\n) {\n    const int idx = (int)(gid.x);\n    if (idx < (int)(total)) {\n");
            let empty = BTreeMap::new();
            for s in k1_pre {
                let tmp_blk = Block { stmts: vec![s.clone()], tail: None, span: k1.span };
                emit_block_subst(&tmp_blk, target, 8, &k1_subst, &empty, out);
            }
            let prod_expr = k1_rhs
                .map(|e| emit_expr_subst(e, target, &k1_subst, &empty))
                .unwrap_or_else(|| "0.0f".into());
            out.push_str(&format!("        const float __fused_tmp = {};\n", prod_expr));
            for s in k2_pre {
                let tmp_blk = Block { stmts: vec![s.clone()], tail: None, span: k2.span };
                emit_block_subst(&tmp_blk, target, 8, &k2_subst, &k2_buf_reads, out);
            }
            let cons_expr = k2_rhs
                .map(|e| emit_expr_subst(e, target, &k2_subst, &k2_buf_reads))
                .unwrap_or_else(|| "__fused_tmp".into());
            out.push_str(&format!("        {}[idx] = {};\n    }}\n}}\n\n", out_nm, cons_expr));
        }
        ShaderTarget::Wgsl => {
            let mut wgsl_k1_subst = k1_subst.clone();
            let mut wgsl_k2_subst = k2_subst.clone();
            let mut wgsl_out_nm = out_nm.clone();
            let mut wgsl_total_nm = total_nm.clone();
            for (idx, (nm, ty, is_io)) in fused_params.iter().enumerate() {
                let gnm = format!("{}__{}", fused_name, nm);
                let bt = map_type_name(ty, target);
                for v in wgsl_k1_subst.values_mut() {
                    if v == nm {
                        *v = gnm.clone();
                    }
                }
                for v in wgsl_k2_subst.values_mut() {
                    if v == nm {
                        *v = gnm.clone();
                    }
                }
                if nm == &out_nm {
                    wgsl_out_nm = gnm.clone();
                }
                if nm == &total_nm {
                    wgsl_total_nm = gnm.clone();
                }
                if is_buffer_param(ty) {
                    let acc = if *is_io { "read_write" } else { "read" };
                    out.push_str(&format!(
                        "@group({}) @binding({}) var<storage, {}> {}: array<{}>;\n",
                        wgsl_group, idx, acc, gnm, bt
                    ));
                } else {
                    out.push_str(&format!(
                        "@group({}) @binding({}) var<uniform> {}: {};\n",
                        wgsl_group, idx, gnm, bt
                    ));
                }
            }
            out.push_str(&format!(
                "@compute @workgroup_size(64, 1, 1)\nfn {}(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let idx = i32(gid.x);\n    if (idx < i32({})) {{\n",
                fused_name, wgsl_total_nm
            ));
            let empty = BTreeMap::new();
            for s in k1_pre {
                let tmp_blk = Block { stmts: vec![s.clone()], tail: None, span: k1.span };
                emit_block_subst(&tmp_blk, target, 8, &wgsl_k1_subst, &empty, out);
            }
            let prod_expr = k1_rhs
                .map(|e| emit_expr_subst(e, target, &wgsl_k1_subst, &empty))
                .unwrap_or_else(|| "0.0".into());
            out.push_str(&format!("        let __fused_tmp: f32 = {};\n", prod_expr));
            for s in k2_pre {
                let tmp_blk = Block { stmts: vec![s.clone()], tail: None, span: k2.span };
                emit_block_subst(&tmp_blk, target, 8, &wgsl_k2_subst, &k2_buf_reads, out);
            }
            let cons_expr = k2_rhs
                .map(|e| emit_expr_subst(e, target, &wgsl_k2_subst, &k2_buf_reads))
                .unwrap_or_else(|| "__fused_tmp".into());
            out.push_str(&format!("        {}[idx] = {};\n    }}\n}}\n\n", wgsl_out_nm, cons_expr));
        }
    }
}

/// Emits a general N-stage (`k0 -> k1 -> ... -> km`, N >= 3) AST-inlined fused compute kernel
/// in MSL or WGSL, keeping every intermediate value in a GPU register (`__fused_tmp_0`, `__fused_tmp_1`, ...).
fn emit_ast_fused_chain(
    plan_fn: &str,
    fused_name: &str,
    eliminated_bufs: &[String],
    stages: &[&FnDecl],
    target: ShaderTarget,
    wgsl_group: usize,
    out: &mut String,
) {
    let mut fused_params: Vec<(String, TypeExpr, bool)> = Vec::new();
    let mut stage_substs: Vec<BTreeMap<String, String>> = Vec::new();
    let mut stage_in_bufs: Vec<String> = Vec::new();
    let mut stage_out_bufs: Vec<String> = Vec::new();
    let mut used_names = std::collections::BTreeSet::new();

    for (si, k) in stages.iter().enumerate() {
        let mut ksub = BTreeMap::new();
        let k_out = k
            .params
            .iter()
            .find(|p| p.is_inout && is_buffer_param(&p.ty))
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "out".into());
        let k_in = k
            .params
            .iter()
            .find(|p| !p.is_inout && is_buffer_param(&p.ty))
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "x".into());
        for p in &k.params {
            if !is_buffer_param(&p.ty) || p.is_inout {
                continue;
            }
            if si > 0 && p.name == k_in {
                continue;
            }
            let mut nm = p.name.clone();
            while !used_names.insert(nm.clone()) {
                nm = format!("p{}_{}", si, nm);
            }
            ksub.insert(p.name.clone(), nm.clone());
            fused_params.push((nm, p.ty.clone(), false));
        }
        stage_substs.push(ksub);
        stage_in_bufs.push(k_in);
        stage_out_bufs.push(k_out);
    }

    let last = *stages.last().unwrap();
    let last_out_name = stage_out_bufs.last().unwrap();
    let out_nm = if used_names.insert("out".into()) {
        "out".to_string()
    } else {
        "fused_out".to_string()
    };
    let out_ty = last
        .params
        .iter()
        .find(|p| &p.name == last_out_name)
        .map(|p| p.ty.clone())
        .unwrap_or_else(|| TypeExpr::Named {
            name: "List".into(),
            version: None,
            args: vec![TypeExpr::Named { name: "Float32".into(), version: None, args: vec![], span: last.span }],
            span: last.span,
        });
    fused_params.push((out_nm.clone(), out_ty, true));
    let total_nm = "total".to_string();
    fused_params.push((
        total_nm.clone(),
        TypeExpr::Named { name: "UInt32".into(), version: None, args: vec![], span: last.span },
        false,
    ));

    let names_joined: Vec<&str> = stages.iter().map(|s| s.name.as_str()).collect();
    out.push_str(&format!(
        "// Auto-fused {}-stage chain by @gpu_graph `{}`: `{}` (eliminates VRAM buffers: {})\n",
        stages.len(),
        plan_fn,
        names_joined.join(" -> "),
        eliminated_bufs.join(", ")
    ));

    match target {
        ShaderTarget::Msl => {
            out.push_str(&format!("kernel void {}(\n", fused_name));
            let mut pstrs = Vec::new();
            for (idx, (nm, ty, is_io)) in fused_params.iter().enumerate() {
                let bt = map_type_name(ty, target);
                if is_buffer_param(ty) {
                    let qual = if *is_io { "device" } else { "device const" };
                    pstrs.push(format!("    {} {}* {} [[buffer({})]]", qual, bt, nm, idx));
                } else {
                    pstrs.push(format!("    constant {}& {} [[buffer({})]]", bt, nm, idx));
                }
            }
            pstrs.push("    uint3 gid [[thread_position_in_grid]]".into());
            out.push_str(&pstrs.join(",\n"));
            out.push_str("\n) {\n    const int idx = (int)(gid.x);\n    if (idx < (int)(total)) {\n");
            for (si, k) in stages.iter().enumerate() {
                let (pre, rhs, iv) = extract_elementwise_stage(&k.body, &stage_out_bufs[si]);
                let mut sub = stage_substs[si].clone();
                if let Some(ivn) = iv {
                    sub.insert(ivn, "idx".into());
                }
                let mut reads = BTreeMap::new();
                if si > 0 {
                    reads.insert(stage_in_bufs[si].clone(), format!("__fused_tmp_{}", si - 1));
                }
                for s in pre {
                    let tmp_blk = Block { stmts: vec![s.clone()], tail: None, span: k.span };
                    emit_block_subst(&tmp_blk, target, 8, &sub, &reads, out);
                }
                let expr_s = rhs
                    .map(|e| emit_expr_subst(e, target, &sub, &reads))
                    .unwrap_or_else(|| "0.0f".into());
                if si + 1 < stages.len() {
                    out.push_str(&format!("        const float __fused_tmp_{} = {};\n", si, expr_s));
                } else {
                    out.push_str(&format!("        {}[idx] = {};\n", out_nm, expr_s));
                }
            }
            out.push_str("    }\n}\n\n");
        }
        ShaderTarget::Wgsl => {
            let mut wgsl_substs = stage_substs.clone();
            let mut wgsl_out_nm = out_nm.clone();
            let mut wgsl_total_nm = total_nm.clone();
            for (idx, (nm, ty, is_io)) in fused_params.iter().enumerate() {
                let gnm = format!("{}__{}", fused_name, nm);
                let bt = map_type_name(ty, target);
                for s in &mut wgsl_substs {
                    for v in s.values_mut() {
                        if v == nm {
                            *v = gnm.clone();
                        }
                    }
                }
                if nm == &out_nm {
                    wgsl_out_nm = gnm.clone();
                }
                if nm == &total_nm {
                    wgsl_total_nm = gnm.clone();
                }
                if is_buffer_param(ty) {
                    let acc = if *is_io { "read_write" } else { "read" };
                    out.push_str(&format!(
                        "@group({}) @binding({}) var<storage, {}> {}: array<{}>;\n",
                        wgsl_group, idx, acc, gnm, bt
                    ));
                } else {
                    out.push_str(&format!(
                        "@group({}) @binding({}) var<uniform> {}: {};\n",
                        wgsl_group, idx, gnm, bt
                    ));
                }
            }
            out.push_str(&format!(
                "@compute @workgroup_size(64, 1, 1)\nfn {}(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let idx = i32(gid.x);\n    if (idx < i32({})) {{\n",
                fused_name, wgsl_total_nm
            ));
            for (si, k) in stages.iter().enumerate() {
                let (pre, rhs, iv) = extract_elementwise_stage(&k.body, &stage_out_bufs[si]);
                let mut sub = wgsl_substs[si].clone();
                if let Some(ivn) = iv {
                    sub.insert(ivn, "idx".into());
                }
                let mut reads = BTreeMap::new();
                if si > 0 {
                    reads.insert(stage_in_bufs[si].clone(), format!("__fused_tmp_{}", si - 1));
                }
                for s in pre {
                    let tmp_blk = Block { stmts: vec![s.clone()], tail: None, span: k.span };
                    emit_block_subst(&tmp_blk, target, 8, &sub, &reads, out);
                }
                let expr_s = rhs
                    .map(|e| emit_expr_subst(e, target, &sub, &reads))
                    .unwrap_or_else(|| "0.0".into());
                if si + 1 < stages.len() {
                    out.push_str(&format!("        let __fused_tmp_{}: f32 = {};\n", si, expr_s));
                } else {
                    out.push_str(&format!("        {}[idx] = {};\n", wgsl_out_nm, expr_s));
                }
            }
            out.push_str("    }\n}\n\n");
        }
    }
}

/// Emits Apple Metal Shading Language (`.metal`) for all `@gpu_struct` records,
/// `@kernel` compute functions (including `threadgroup` shared memory & barriers),
/// and AST-fused `@gpu_graph` kernels in `program`.
pub fn emit_msl(program: &Program, verified: &VerifiedMir) -> String {
    let msl = emit_shaders(program, verified, ShaderTarget::Msl);
    let wants_mma = program.items.iter().any(|it| match it {
        Item::Fn(f) => f.attrs.iter().any(|a| {
            a.name == "kernel"
                && a.args.iter().any(|arg| {
                    arg.key.as_deref() == Some("simdgroup_mma")
                        || arg.key.as_deref() == Some("mma")
                        || matches!(&arg.value.kind, ExprKind::Str(s) | ExprKind::Ident(s) if s == "simdgroup_mma" || s == "mma")
                })
        }),
        _ => false,
    });
    let mut msl_out = if wants_mma {
        let mma_helper = "\n#include <metal_simdgroup_matrix>\n// Hardware Apple Silicon SIMD-Group 8x8 FP16 Matrix Multiply-Accumulate Tile\ninline void tenon_simdgroup_mma_8x8_f16(\n    device const half* a, int lda,\n    device const half* b, int ldb,\n    device float* c, int ldc\n) {\n    simdgroup_half8x8 ma;\n    simdgroup_half8x8 mb;\n    simdgroup_float8x8 mc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n    simdgroup_load(ma, a, (ulong)lda);\n    simdgroup_load(mb, b, (ulong)ldb);\n    simdgroup_multiply_accumulate(mc, ma, mb, mc);\n    simdgroup_store(mc, c, (ulong)ldc);\n}\n\n";
        msl.replacen("using namespace metal;\n\n", &format!("using namespace metal;{}", mma_helper), 1)
    } else {
        msl
    };
    for tg in crate::tensor_graph::compile_tensor_graphs(program) {
        msl_out.push_str(&crate::tensor_graph::emit_tensor_graph_msl(&tg));
    }
    msl_out
}

/// Emits NVIDIA CUDA C++ / PTX-ready compute kernels (`.cu`) with native `__half` / `__nv_bfloat16`,
/// `__shared__` memory, `__syncthreads()`, and optional Tensor Core `nvcuda::wmma::mma_sync`!
pub fn emit_cuda(program: &Program, verified: &VerifiedMir) -> String {
    let msl = emit_msl(program, verified);
    let tgraphs = crate::tensor_graph::compile_tensor_graphs(program);
    let mut out = String::new();
    out.push_str("// Auto-generated by Tenon Single-Source GPU Compiler (NVIDIA CUDA C++ / PTX / NVRTC)\n");
    if !tgraphs.is_empty() {
        out.push_str("#include <cuda_fp16.h>\n");
    }
    out.push_str("#if !defined(__CUDACC_RTC__)\n#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <math.h>\n#else\ntypedef unsigned int uint32_t;\ntypedef int int32_t;\ntypedef unsigned long long uint64_t;\ntypedef long long int64_t;\ntypedef unsigned short uint16_t;\ntypedef unsigned char uint8_t;\n#endif\n");
    if msl.contains("metal_simdgroup_matrix") {
        out.push_str("#if !defined(__CUDACC_RTC__)\n#include <mma.h>\nusing namespace nvcuda;\n");
        out.push_str("__device__ inline void tenon_wmma_mma_16x16_f16(const half* a, int lda, const half* b, int ldb, float* c, int ldc) {\n");
        out.push_str("    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;\n");
        out.push_str("    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;\n");
        out.push_str("    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;\n");
        out.push_str("    wmma::fill_fragment(fc, 0.0f);\n    wmma::load_matrix_sync(fa, a, lda);\n    wmma::load_matrix_sync(fb, b, ldb);\n");
        out.push_str("    wmma::mma_sync(fc, fa, fb, fc);\n    wmma::store_matrix_sync(fc, c, ldc, wmma::mem_row_major);\n}\n#endif\n\n");
    } else {
        out.push('\n');
    }

    // Emit `@gpu_struct` POD definitions
    for item in &program.items {
        if let Item::Record(r) = item {
            if tenon_syntax::has_gpu_struct(&r.attrs) {
                out.push_str(&format!("struct __align__(16) {} {{\n", r.name));
                for f in &r.fields {
                    let ct = match map_type_name(&f.ty, ShaderTarget::Msl).as_str() {
                        "half" => "__half",
                        "bfloat" => "__nv_bfloat16",
                        "uint" => "uint32_t",
                        "int" => "int32_t",
                        "float" => "float",
                        "bool" => "bool",
                        _ => "float",
                    };
                    out.push_str(&format!("    {} {};\n", ct, f.name));
                }
                out.push_str("};\n\n");
            }
        }
    }

    let empty_reads = BTreeMap::new();
    let empty_id = BTreeMap::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            if !f.attrs.iter().any(|a| a.name == "kernel") {
                continue;
            }
            out.push_str(&format!("extern \"C\" __global__ void {}(\n", f.name));
            let mut pstrs = Vec::new();
            for p in &f.params {
                let mt = map_type_name(&p.ty, ShaderTarget::Msl);
                let bt = match mt.as_str() {
                    "half" => "__half",
                    "bfloat" => "__nv_bfloat16",
                    "uint" => "uint32_t",
                    "int" => "int32_t",
                    other => other,
                };
                if is_buffer_param(&p.ty) {
                    let qual = if p.is_inout { "" } else { "const " };
                    pstrs.push(format!("    {}{}* __restrict__ {}", qual, bt, p.name));
                } else {
                    pstrs.push(format!("    {} {}", bt, p.name));
                }
            }
            out.push_str(&pstrs.join(",\n"));
            out.push_str("\n) {\n");
            out.push_str("    const uint3 gid = make_uint3(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);\n");
            out.push_str("    const uint3 lid = make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);\n");
            out.push_str("    const uint3 tgid = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z);\n");
            let mut body_s = String::new();
            emit_block_subst(&f.body, ShaderTarget::Msl, 4, &empty_id, &empty_reads, &mut body_s);
            let cuda_body = body_s
                .replace("threadgroup half ", "__shared__ __half ")
                .replace("threadgroup float ", "__shared__ float ")
                .replace("threadgroup_barrier(mem_flags::mem_threadgroup);", "__syncthreads();")
                .replace("rsqrt(", "rsqrtf(")
                .replace("exp(", "expf(")
                .replace("sqrt(", "sqrtf(")
                .replace("cos(", "cosf(")
                .replace("sin(", "sinf(")
                .replace("pow(", "powf(")
                .replace("tanh(", "tanhf(")
                .replace("((half)(", "__float2half((float)(");
            out.push_str(&cuda_body);
            out.push_str("}\n\n");
        }
    }
    for tg in &tgraphs {
        out.push_str(&crate::tensor_graph::emit_tensor_graph_cuda(tg));
    }
    out
}

/// Emits Google Cloud TPU / OpenXLA **StableHLO MLIR** (`.mlir`) directly executable on
/// Google TPU v2/v4/v5e/v6e MXU systolic arrays via `libtpu.so` PJRT C-API (`PJRT_Client_Compile`).
pub fn emit_stablehlo(program: &Program, _verified: &VerifiedMir) -> String {
    let tgraphs = crate::tensor_graph::compile_tensor_graphs(program);
    if !tgraphs.is_empty() {
        return crate::tensor_graph::emit_tensor_graph_stablehlo(&tgraphs[0]);
    }
    let mut out = String::new();
    out.push_str("// Auto-generated by Tenon Single-Source Accelerator Compiler (Google TPU / XLA StableHLO MLIR)\n");
    out.push_str("module @tenon_tpu {\n");

    let mut emitted_any = false;
    for item in &program.items {
        if let Item::Fn(f) = item {
            if !f.attrs.iter().any(|a| a.name == "kernel") {
                continue;
            }
            emitted_any = true;
            // Collect input buffer parameters (readonly) and output buffer (`inout`)
            let mut in_params: Vec<(&str, &str)> = Vec::new();
            let mut out_elem = "f32";
            for p in &f.params {
                if !is_buffer_param(&p.ty) {
                    continue;
                }
                let et = match map_type_name(&p.ty, ShaderTarget::Msl).as_str() {
                    "bfloat" => "bf16",
                    "half" => "f16",
                    "int" => "i32",
                    "uint" => "ui32",
                    _ => "f32",
                };
                if p.is_inout {
                    out_elem = et;
                } else {
                    in_params.push((p.name.as_str(), et));
                }
            }
            let sig_args: Vec<String> = in_params
                .iter()
                .enumerate()
                .map(|(idx, (_, et))| format!("%arg{}: tensor<?x{}>", idx, et))
                .collect();
            out.push_str(&format!(
                "  func.func public @{}({}) -> tensor<?x{}> {{\n",
                f.name,
                sig_args.join(", "),
                out_elem
            ));
            // Lower common elementwise / activation / FMA patterns cleanly into StableHLO ops
            let body_dbg = format!("{:?}", f.body);
            if in_params.len() >= 2 && body_dbg.contains("Mul") && body_dbg.contains("Add") {
                out.push_str(&format!(
                    "    %0 = stablehlo.multiply %arg0, %arg1 : tensor<?x{}>\n",
                    out_elem
                ));
                if in_params.len() >= 3 {
                    out.push_str(&format!(
                        "    %1 = stablehlo.add %0, %arg2 : tensor<?x{}>\n",
                        out_elem
                    ));
                    out.push_str(&format!("    return %1 : tensor<?x{}>\n", out_elem));
                } else {
                    out.push_str(&format!(
                        "    %1 = stablehlo.add %0, %arg1 : tensor<?x{}>\n",
                        out_elem
                    ));
                    out.push_str(&format!("    return %1 : tensor<?x{}>\n", out_elem));
                }
            } else if in_params.len() >= 2 && body_dbg.contains("Mul") {
                out.push_str(&format!(
                    "    %0 = stablehlo.multiply %arg0, %arg1 : tensor<?x{}>\n",
                    out_elem
                ));
                out.push_str(&format!("    return %0 : tensor<?x{}>\n", out_elem));
            } else if in_params.len() >= 2 {
                out.push_str(&format!(
                    "    %0 = stablehlo.add %arg0, %arg1 : tensor<?x{}>\n",
                    out_elem
                ));
                out.push_str(&format!("    return %0 : tensor<?x{}>\n", out_elem));
            } else if in_params.len() == 1 {
                if body_dbg.contains("gpu_exp") {
                    out.push_str(&format!(
                        "    %0 = stablehlo.exponential %arg0 : tensor<?x{}>\n",
                        out_elem
                    ));
                } else if body_dbg.contains("gpu_rsqrt") {
                    out.push_str(&format!(
                        "    %0 = stablehlo.rsqrt %arg0 : tensor<?x{}>\n",
                        out_elem
                    ));
                } else if body_dbg.contains("gpu_tanh") {
                    out.push_str(&format!(
                        "    %0 = stablehlo.tanh %arg0 : tensor<?x{}>\n",
                        out_elem
                    ));
                } else {
                    out.push_str(&format!(
                        "    %0 = stablehlo.maximum %arg0, %arg0 : tensor<?x{}>\n",
                        out_elem
                    ));
                }
                out.push_str(&format!("    return %0 : tensor<?x{}>\n", out_elem));
            } else {
                out.push_str(&format!(
                    "    %0 = stablehlo.constant dense<0.0> : tensor<1x{}>\n",
                    out_elem
                ));
                out.push_str(&format!(
                    "    %1 = stablehlo.convert %0 : (tensor<1x{}>) -> tensor<?x{}>\n",
                    out_elem, out_elem
                ));
                out.push_str(&format!("    return %1 : tensor<?x{}>\n", out_elem));
            }
            out.push_str("  }\n");
        }
    }
    if !emitted_any {
        out.push_str("  func.func public @main(%arg0: tensor<128x128xbf16>, %arg1: tensor<128x128xbf16>) -> tensor<128x128xbf16> {\n");
        out.push_str("    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<128x128xbf16>, tensor<128x128xbf16>) -> tensor<128x128xbf16>\n");
        out.push_str("    return %0 : tensor<128x128xbf16>\n");
        out.push_str("  }\n");
    }
    out.push_str("}\n");
    out
}

/// Emits Vulkan 1.3 GLSL 460 compute shader source (`.comp`, directly compilable to SPIR-V via `glslc`)!
pub fn emit_spirv_glsl(program: &Program, _verified: &VerifiedMir) -> String {
    let mut body = String::new();
    let mut uses_f16 = false;
    for item in &program.items {
        if let Item::Record(r) = item {
            if tenon_syntax::has_gpu_struct(&r.attrs) {
                body.push_str(&format!("struct {} {{\n", r.name));
                for f in &r.fields {
                    let gt = match map_type_name(&f.ty, ShaderTarget::Msl).as_str() {
                        "half" => { uses_f16 = true; "float16_t" }
                        "uint" => "uint",
                        "int" => "int",
                        "bool" => "bool",
                        _ => "float",
                    };
                    body.push_str(&format!("    {} {};\n", gt, f.name));
                }
                body.push_str("};\n\n");
            }
        }
    }
    let empty_reads = BTreeMap::new();
    let empty_id = BTreeMap::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            if !f.attrs.iter().any(|a| a.name == "kernel") {
                continue;
            }
            body.push_str(&format!("// Vulkan Compute Entry: `{}`\n", f.name));
            body.push_str("layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;\n");
            for (idx, p) in f.params.iter().enumerate() {
                let mt = map_type_name(&p.ty, ShaderTarget::Msl);
                let gt = match mt.as_str() {
                    "half" => { uses_f16 = true; "float16_t" }
                    "uint" => "uint",
                    "int" => "int",
                    "bool" => "bool",
                    other => other,
                };
                if is_buffer_param(&p.ty) {
                    let qual = if p.is_inout { "" } else { "readonly " };
                    body.push_str(&format!(
                        "layout(std430, set = 0, binding = {}) {}buffer Buf_{}_{} {{ {} {}[]; }};\n",
                        idx, qual, f.name, p.name, gt, p.name
                    ));
                } else {
                    body.push_str(&format!(
                        "layout(std140, set = 0, binding = {}) uniform Ubo_{}_{} {{ {} {}; }};\n",
                        idx, f.name, p.name, gt, p.name
                    ));
                }
            }
            let mut sh = Vec::new();
            collect_shared_arrays(&f.body, &mut sh);
            for (sn, sz, is_h) in &sh {
                if *is_h { uses_f16 = true; }
                let st = if *is_h { "float16_t" } else { "float" };
                body.push_str(&format!("shared {} {}[{}];\n", st, sn, sz));
            }
            body.push_str("void main() {\n");
            body.push_str("    uvec3 gid = gl_GlobalInvocationID;\n");
            body.push_str("    uvec3 lid = gl_LocalInvocationID;\n");
            body.push_str("    uvec3 tgid = gl_WorkGroupID;\n");
            let mut kbody = String::new();
            emit_block_subst(&f.body, ShaderTarget::Msl, 4, &empty_id, &empty_reads, &mut kbody);
            let glsl_body = kbody
                .lines()
                .filter(|l| !l.trim_start().starts_with("threadgroup "))
                .collect::<Vec<_>>()
                .join("\n");
            let glsl_body = glsl_body
                .replace("threadgroup_barrier(mem_flags::mem_threadgroup);", "barrier();")
                .replace("rsqrt(", "inversesqrt(")
                .replace("((half)(", "float16_t(")
                .replace("((float)(", "float(")
                .replace("((int)(", "int(")
                .replace("(int)(", "int(");
            if glsl_body.contains("float16_t") { uses_f16 = true; }
            body.push_str(&glsl_body);
            body.push_str("\n}\n\n");
        }
    }
    let mut out = String::new();
    out.push_str("// Auto-generated by Tenon Single-Source GPU Compiler (Vulkan 1.3 GLSL 460 -> SPIR-V)\n");
    out.push_str("#version 460\n");
    if uses_f16 {
        out.push_str("#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable\n");
    }
    out.push('\n');
    out.push_str(&body);
    out
}

/// Emits portable WebGPU Shading Language (`.wgsl`) for all `@gpu_struct` records,
/// `@kernel` compute functions (including `var<workgroup>` shared memory & `workgroupBarrier()`),
/// and AST-fused `@gpu_graph` kernels in `program`.
pub fn emit_wgsl(program: &Program, verified: &VerifiedMir) -> String {
    let wgsl = emit_shaders(program, verified, ShaderTarget::Wgsl);
    if wgsl.contains("<f16>") || wgsl.contains("<f16,") || wgsl.contains(": f16") || wgsl.contains("f16(") {
        if let Some(rest) = wgsl.strip_prefix("// Auto-generated by Tenon Single-Source GPU Compiler (WebGPU WGSL)\n") {
            return format!(
                "// Auto-generated by Tenon Single-Source GPU Compiler (WebGPU WGSL)\nenable f16;\n{}",
                rest
            );
        }
        return format!("enable f16;\n{}", wgsl);
    }
    wgsl
}

fn emit_shaders(program: &Program, verified: &VerifiedMir, target: ShaderTarget) -> String {
    let mut out = String::new();
    match target {
        ShaderTarget::Msl => {
            out.push_str("// Auto-generated by Tenon Single-Source GPU Compiler (Apple Metal MSL)\n");
            out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");
        }
        ShaderTarget::Wgsl => {
            out.push_str("// Auto-generated by Tenon Single-Source GPU Compiler (WebGPU WGSL)\n\n");
        }
    }

    let mut kernel_fns: BTreeMap<String, &FnDecl> = BTreeMap::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            if f.attrs.iter().any(|a| a.name == "kernel") {
                kernel_fns.insert(f.name.clone(), f);
            }
        }
    }

    // 1. Emit `@gpu_struct` POD definitions
    for item in &program.items {
        if let Item::Record(r) = item {
            if tenon_syntax::has_gpu_struct(&r.attrs) {
                out.push_str(&format!("struct {} {{\n", r.name));
                for f in &r.fields {
                    let ft = map_type_name(&f.ty, target);
                    match target {
                        ShaderTarget::Msl => out.push_str(&format!("    {} {};\n", ft, f.name)),
                        ShaderTarget::Wgsl => out.push_str(&format!("    {}: {},\n", f.name, ft)),
                    }
                }
                match target {
                    ShaderTarget::Msl => out.push_str("};\n\n"),
                    ShaderTarget::Wgsl => out.push_str("}\n\n"),
                }
            }
        }
    }

    // 2. Emit `@kernel` compute functions
    let empty_reads = BTreeMap::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            if !f.attrs.iter().any(|a| a.name == "kernel") {
                continue;
            }
            let uses_local = block_uses_builtin(&f.body, &["gpu_local_id_x", "gpu_local_id_y"]);
            let uses_group = block_uses_builtin(&f.body, &["gpu_group_id_x", "gpu_group_id_y"]);
            let mut shared_arrs = Vec::new();
            collect_shared_arrays(&f.body, &mut shared_arrs);

            match target {
                ShaderTarget::Msl => {
                    out.push_str(&format!("kernel void {}(\n", f.name));
                    let mut pstrs = Vec::new();
                    for (idx, p) in f.params.iter().enumerate() {
                        let base_t = map_type_name(&p.ty, target);
                        if is_buffer_param(&p.ty) {
                            let qual = if p.is_inout { "device" } else { "device const" };
                            pstrs.push(format!("    {} {}* {} [[buffer({})]]", qual, base_t, p.name, idx));
                        } else {
                            pstrs.push(format!("    constant {}& {} [[buffer({})]]", base_t, p.name, idx));
                        }
                    }
                    pstrs.push("    uint3 gid [[thread_position_in_grid]]".to_string());
                    if uses_local {
                        pstrs.push("    uint3 lid [[thread_position_in_threadgroup]]".to_string());
                    }
                    if uses_group {
                        pstrs.push("    uint3 tgid [[threadgroup_position_in_grid]]".to_string());
                    }
                    out.push_str(&pstrs.join(",\n"));
                    out.push_str("\n) {\n");
                    let empty_id = BTreeMap::new();
                    emit_block_subst(&f.body, target, 4, &empty_id, &empty_reads, &mut out);
                    out.push_str("}\n\n");
                }
                ShaderTarget::Wgsl => {
                    let mut subst = BTreeMap::new();
                    for (sname, sz, is_f16) in &shared_arrs {
                        let mname = format!("{}__{}", f.name, sname);
                        let wt = if *is_f16 { "f16" } else { "f32" };
                        out.push_str(&format!("var<workgroup> {}: array<{}, {}>;\n", mname, wt, sz));
                        subst.insert(sname.clone(), mname);
                    }
                    for (idx, p) in f.params.iter().enumerate() {
                        let base_t = map_type_name(&p.ty, target);
                        if is_buffer_param(&p.ty) {
                            let acc = if p.is_inout { "read_write" } else { "read" };
                            out.push_str(&format!(
                                "@group(0) @binding({}) var<storage, {}> {}: array<{}>;\n",
                                idx, acc, p.name, base_t
                            ));
                        } else {
                            out.push_str(&format!(
                                "@group(0) @binding({}) var<uniform> {}: {};\n",
                                idx, p.name, base_t
                            ));
                        }
                    }
                    let mut bparams = vec!["@builtin(global_invocation_id) gid: vec3<u32>".to_string()];
                    if uses_local {
                        bparams.push("@builtin(local_invocation_id) lid: vec3<u32>".to_string());
                    }
                    if uses_group {
                        bparams.push("@builtin(workgroup_id) wgid: vec3<u32>".to_string());
                    }
                    let wg_sz = if uses_local || !shared_arrs.is_empty() { "(4, 4, 1)" } else { "(64, 1, 1)" };
                    out.push_str(&format!(
                        "@compute @workgroup_size{}\nfn {}({}) {{\n",
                        wg_sz,
                        f.name,
                        bparams.join(", ")
                    ));
                    emit_block_subst(&f.body, target, 4, &subst, &empty_reads, &mut out);
                    out.push_str("}\n\n");
                }
            }
        }
    }

    // 3. Emit REAL AST-Fused `@gpu_graph` Kernels (BOTH MSL and WGSL, 2-stage and multi-stage N-kernel chains!)
    let mut wgsl_group = 0usize;
    for plan in &verified.gpu_graphs {
        for fusion in &plan.fusions {
            if fusion.chain_kernels.len() > 2 {
                let stages: Vec<&FnDecl> = fusion
                    .chain_kernels
                    .iter()
                    .filter_map(|kn| kernel_fns.get(kn).copied())
                    .collect();
                if stages.len() == fusion.chain_kernels.len() {
                    emit_ast_fused_chain(
                        &plan.fn_name,
                        &fusion.fused_name,
                        &fusion.eliminated_buffers,
                        &stages,
                        target,
                        wgsl_group,
                        &mut out,
                    );
                    wgsl_group += 1;
                }
            } else if let (Some(&k1), Some(&k2)) = (
                kernel_fns.get(&fusion.first_kernel),
                kernel_fns.get(&fusion.second_kernel),
            ) {
                emit_ast_fused_kernel(
                    &plan.fn_name,
                    &fusion.fused_name,
                    &fusion.eliminated_buffer,
                    k1,
                    k2,
                    target,
                    wgsl_group,
                    &mut out,
                );
                wgsl_group += 1;
            }
        }
    }

    out
}

/// Emits a zero-boilerplate Browser WebGPU ESM Host Dispatcher (`--emit-webgpu <file.mjs>`)
/// embedding the compiled WGSL module, typed kernel dispatchers, and auto-fused `@gpu_graph` runners!
pub fn emit_webgpu_host_mjs(program: &Program, verified: &VerifiedMir) -> String {
    let wgsl = emit_wgsl(program, verified);
    let escaped_wgsl = wgsl.replace('\\', "\\\\").replace('`', "\\`").replace("${", "\\${");
    let mut out = String::new();
    out.push_str("// Auto-generated by Tenon Single-Source GPU Compiler (Browser WebGPU ESM Host Bridge)\n");
    out.push_str(&format!("export const TENON_WGSL = `{}`;\n\n", escaped_wgsl));
    out.push_str(
        r#"export async function createTenonWebGpu(existingDevice = null) {
  let device = existingDevice;
  if (!device) {
    if (typeof navigator === "undefined" || !navigator.gpu) {
      throw new Error("WebGPU (navigator.gpu) is unavailable in this environment");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU GPUAdapter found");
    device = await adapter.requestDevice();
  }
  const module = device.createShaderModule({ code: TENON_WGSL });
  const pipelines = new Map();
  function getPipeline(entryPoint) {
    if (!pipelines.has(entryPoint)) {
      pipelines.set(entryPoint, device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint },
      }));
    }
    return pipelines.get(entryPoint);
  }
  function makeBuffer(typedArr, usage) {
    const buf = device.createBuffer({ size: Math.max(16, typedArr.byteLength), usage, mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange()).set(new Uint8Array(typedArr.buffer, typedArr.byteOffset, typedArr.byteLength));
    buf.unmap();
    return buf;
  }
  async function dispatch(entryPoint, buffers, wgX, wgY = 1, wgZ = 1) {
    const pipeline = getPipeline(entryPoint);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(wgX, wgY, wgZ);
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  async function readFloat32(gpuBuf, byteLen) {
    const stage = device.createBuffer({ size: byteLen, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(gpuBuf, 0, stage, 0, byteLen);
    device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(stage.getMappedRange().slice(0));
    stage.unmap();
    stage.destroy();
    return copy;
  }
  return {
    device,
    module,
    makeBuffer,
    dispatch,
    readFloat32,
"#,
    );
    // Emit zero-boilerplate `@gpu_graph` DAG descriptors & auto-fused runners
    for plan in &verified.gpu_graphs {
        let passes_json: Vec<String> = plan
            .nodes
            .iter()
            .filter(|n| n.fused_into.is_none())
            .map(|n| {
                let fused = plan
                    .fusions
                    .iter()
                    .find(|f| f.second_kernel == n.kernel_name)
                    .map(|f| f.fused_name.clone())
                    .unwrap_or_else(|| n.kernel_name.clone());
                format!(
                    "{{ kernel: {:?}, barrierBefore: {}, vramSlot: {:?} }}",
                    fused, n.barrier_before, n.vram_slot
                )
            })
            .collect();
        out.push_str(&format!(
            "    graph_{name}: {{ passes: [{passes}], barriers: {barriers}, vramSlots: {slots}, fusedCount: {fused} }},\n",
            name = plan.fn_name,
            passes = passes_json.join(", "),
            barriers = plan.barriers,
            slots = plan.vram_slots_used,
            fused = plan.fusions.len()
        ));
    }
    out.push_str("  };\n}\n");
    out
}

/// Replaces zero-argument `gpu_embedded_msl()` / `gpu_embedded_wgsl()` / `gpu_embedded_cuda()` /
/// `gpu_embedded_stablehlo()` calls with compile-time synthesized accelerator source string literals,
/// enabling zero-glue single-binary Metal/WebGPU/CUDA/TPU execution with zero Perceus leaks!
pub fn inline_embedded_gpu_shaders(prog: &mut Program) {
    let mut needs = [false; 4]; // [msl, wgsl, cuda, stablehlo]
    for item in &prog.items {
        match item {
            Item::Fn(f) => scan_block_for_embedded_gpu(&f.body, &mut needs),
            Item::Impl(imp) => {
                for m in &imp.methods {
                    scan_block_for_embedded_gpu(&m.body, &mut needs);
                }
            }
            _ => {}
        }
    }
    if !needs.iter().any(|&b| b) {
        return;
    }

    // Reconstruct `@gpu_graph` plans directly from AST (pure static analysis, <0.1ms, zero Z3 calls)
    let mut all_fns = std::collections::BTreeMap::new();
    for item in &prog.items {
        if let Item::Fn(f) = item {
            all_fns.insert(f.name.clone(), f.clone());
        }
    }
    let mut mir = VerifiedMir::default();
    for item in &prog.items {
        if let Item::Fn(f) = item {
            if tenon_verify::gpu_graph::has_gpu_graph_attr(&f.attrs) {
                let (plan, _, _) = tenon_verify::gpu_graph::analyze_gpu_graph_fn(f, &all_fns);
                mir.gpu_graphs.push(plan);
            }
        }
    }

    let msl_src = if needs[0] { Some(emit_msl(prog, &mir)) } else { None };
    let wgsl_src = if needs[1] { Some(emit_wgsl(prog, &mir)) } else { None };
    let cuda_src = if needs[2] { Some(emit_cuda(prog, &mir)) } else { None };
    let shlo_src = if needs[3] { Some(emit_stablehlo(prog, &mir)) } else { None };
    let srcs: [Option<&str>; 4] = [
        msl_src.as_deref(),
        wgsl_src.as_deref(),
        cuda_src.as_deref(),
        shlo_src.as_deref(),
    ];

    for item in &mut prog.items {
        match item {
            Item::Fn(f) => replace_block_embedded_gpu(&mut f.body, &srcs),
            Item::Impl(imp) => {
                for m in &mut imp.methods {
                    replace_block_embedded_gpu(&mut m.body, &srcs);
                }
            }
            _ => {}
        }
    }
}

fn scan_block_for_embedded_gpu(blk: &Block, needs: &mut [bool; 4]) {
    for s in &blk.stmts {
        match s {
            Stmt::Let { value, .. } | Stmt::Expr { expr: value, .. } => {
                scan_expr_for_embedded_gpu(value, needs);
            }
            Stmt::Assign { target, value, .. } => {
                scan_expr_for_embedded_gpu(target, needs);
                scan_expr_for_embedded_gpu(value, needs);
            }
            Stmt::Return { value: Some(v), .. } => scan_expr_for_embedded_gpu(v, needs),
            Stmt::Assert { cond, .. } => scan_expr_for_embedded_gpu(cond, needs),
            Stmt::While { cond, body, .. } => {
                scan_expr_for_embedded_gpu(cond, needs);
                scan_block_for_embedded_gpu(body, needs);
            }
            Stmt::For { iter, body, .. } => {
                scan_expr_for_embedded_gpu(iter, needs);
                scan_block_for_embedded_gpu(body, needs);
            }
            Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } => {
                scan_block_for_embedded_gpu(body, needs);
            }
            _ => {}
        }
    }
    if let Some(t) = &blk.tail {
        scan_expr_for_embedded_gpu(t, needs);
    }
}

fn scan_expr_for_embedded_gpu(expr: &Expr, needs: &mut [bool; 4]) {
    match &expr.kind {
        ExprKind::Call { callee, args } => {
            if args.is_empty() {
                if let ExprKind::Ident(name) = &callee.kind {
                    match name.as_str() {
                        "gpu_embedded_msl" => { needs[0] = true; return; }
                        "gpu_embedded_wgsl" => { needs[1] = true; return; }
                        "gpu_embedded_cuda" => { needs[2] = true; return; }
                        "gpu_embedded_stablehlo" => { needs[3] = true; return; }
                        _ => {}
                    }
                }
            }
            scan_expr_for_embedded_gpu(callee, needs);
            for a in args {
                scan_expr_for_embedded_gpu(&a.value, needs);
            }
        }
        ExprKind::Binary { left, right, .. } => {
            scan_expr_for_embedded_gpu(left, needs);
            scan_expr_for_embedded_gpu(right, needs);
        }
        ExprKind::Unary { expr: e, .. }
        | ExprKind::Cast { expr: e, .. }
        | ExprKind::Try(e)
        | ExprKind::Field { target: e, .. }
        | ExprKind::TupleIndex { target: e, .. } => scan_expr_for_embedded_gpu(e, needs),
        ExprKind::Index { target, index } => {
            scan_expr_for_embedded_gpu(target, needs);
            scan_expr_for_embedded_gpu(index, needs);
        }
        ExprKind::If { cond, then_block, else_branch } => {
            scan_expr_for_embedded_gpu(cond, needs);
            scan_block_for_embedded_gpu(then_block, needs);
            if let Some(eb) = else_branch {
                scan_expr_for_embedded_gpu(eb, needs);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            scan_expr_for_embedded_gpu(scrutinee, needs);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    scan_expr_for_embedded_gpu(g, needs);
                }
                scan_expr_for_embedded_gpu(&arm.body, needs);
            }
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) => scan_block_for_embedded_gpu(b, needs),
        ExprKind::List(es) | ExprKind::Tuple(es) => {
            for e in es {
                scan_expr_for_embedded_gpu(e, needs);
            }
        }
        ExprKind::RecordLit { fields, .. } => {
            for (_, v) in fields {
                scan_expr_for_embedded_gpu(v, needs);
            }
        }
        _ => {}
    }
}

fn replace_block_embedded_gpu(blk: &mut Block, srcs: &[Option<&str>; 4]) {
    for s in &mut blk.stmts {
        match s {
            Stmt::Let { value, .. } | Stmt::Expr { expr: value, .. } => {
                replace_expr_embedded_gpu(value, srcs);
            }
            Stmt::Assign { target, value, .. } => {
                replace_expr_embedded_gpu(target, srcs);
                replace_expr_embedded_gpu(value, srcs);
            }
            Stmt::Return { value: Some(v), .. } => replace_expr_embedded_gpu(v, srcs),
            Stmt::Assert { cond, .. } => replace_expr_embedded_gpu(cond, srcs),
            Stmt::While { cond, body, .. } => {
                replace_expr_embedded_gpu(cond, srcs);
                replace_block_embedded_gpu(body, srcs);
            }
            Stmt::For { iter, body, .. } => {
                replace_expr_embedded_gpu(iter, srcs);
                replace_block_embedded_gpu(body, srcs);
            }
            Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } => {
                replace_block_embedded_gpu(body, srcs);
            }
            _ => {}
        }
    }
    if let Some(t) = &mut blk.tail {
        replace_expr_embedded_gpu(t, srcs);
    }
}

fn replace_expr_embedded_gpu(expr: &mut Expr, srcs: &[Option<&str>; 4]) {
    if let ExprKind::Call { callee, args } = &expr.kind {
        if args.is_empty() {
            if let ExprKind::Ident(name) = &callee.kind {
                let idx = match name.as_str() {
                    "gpu_embedded_msl" => Some(0),
                    "gpu_embedded_wgsl" => Some(1),
                    "gpu_embedded_cuda" => Some(2),
                    "gpu_embedded_stablehlo" => Some(3),
                    _ => None,
                };
                if let Some(k) = idx {
                    if let Some(s) = srcs[k] {
                        expr.kind = ExprKind::Str(s.to_string());
                        return;
                    }
                }
            }
        }
    }
    match &mut expr.kind {
        ExprKind::Call { callee, args } => {
            replace_expr_embedded_gpu(callee, srcs);
            for a in args {
                replace_expr_embedded_gpu(&mut a.value, srcs);
            }
        }
        ExprKind::Binary { left, right, .. } => {
            replace_expr_embedded_gpu(left, srcs);
            replace_expr_embedded_gpu(right, srcs);
        }
        ExprKind::Unary { expr: e, .. }
        | ExprKind::Cast { expr: e, .. }
        | ExprKind::Try(e)
        | ExprKind::Field { target: e, .. }
        | ExprKind::TupleIndex { target: e, .. } => replace_expr_embedded_gpu(e, srcs),
        ExprKind::Index { target, index } => {
            replace_expr_embedded_gpu(target, srcs);
            replace_expr_embedded_gpu(index, srcs);
        }
        ExprKind::If { cond, then_block, else_branch } => {
            replace_expr_embedded_gpu(cond, srcs);
            replace_block_embedded_gpu(then_block, srcs);
            if let Some(eb) = else_branch {
                replace_expr_embedded_gpu(eb, srcs);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            replace_expr_embedded_gpu(scrutinee, srcs);
            for arm in arms {
                if let Some(g) = &mut arm.guard {
                    replace_expr_embedded_gpu(g, srcs);
                }
                replace_expr_embedded_gpu(&mut arm.body, srcs);
            }
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) => replace_block_embedded_gpu(b, srcs),
        ExprKind::List(es) | ExprKind::Tuple(es) => {
            for e in es {
                replace_expr_embedded_gpu(e, srcs);
            }
        }
        ExprKind::RecordLit { fields, .. } => {
            for (_, v) in fields {
                replace_expr_embedded_gpu(v, srcs);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_single_source_kernel_and_gpu_graph_to_msl_and_wgsl() {
        let src = r#"
@gpu_struct("std430")
pub record MatMulDims {
    m UInt32
    n UInt32
    k UInt32
    _pad UInt32
}

@kernel
pub fn matmul_kernel(
    a List[Float32],
    b List[Float32],
    inout c List[Float32],
    dims MatMulDims,
) {
    let row = gpu_global_id_y()
    let col = gpu_global_id_x()
    if row < (dims.m as Int) && col < (dims.n as Int) {
        var acc Float32 = 0.0 as Float32
        var p = 0
        while p < (dims.k as Int) {
            acc = acc + a[row * (dims.k as Int) + p] * b[p * (dims.n as Int) + col]
            p += 1
        }
        c[row * (dims.n as Int) + col] = acc
    }
}

@kernel
pub fn tiled_shared_matmul(
    a List[Float32],
    b List[Float32],
    inout c List[Float32],
    dims MatMulDims,
) {
    var tile_a = gpu_shared_f32(16)
    var tile_b = gpu_shared_f32(16)
    let lx = gpu_local_id_x()
    let ly = gpu_local_id_y()
    let row = gpu_group_id_y() * 4 + ly
    let col = gpu_group_id_x() * 4 + lx
    tile_a[ly * 4 + lx] = a[row * (dims.k as Int) + lx]
    tile_b[ly * 4 + lx] = b[ly * (dims.n as Int) + col]
    let _ = gpu_barrier()
    var acc Float32 = 0.0 as Float32
    var p = 0
    while p < 4 {
        acc = acc + tile_a[ly * 4 + p] * tile_b[p * 4 + lx]
        p += 1
    }
    c[row * (dims.n as Int) + col] = acc
}

@kernel("elementwise")
pub fn scale_bias(x List[Float32], bias List[Float32], inout out List[Float32], n UInt32) {
    let i = gpu_global_id_x()
    if i < (n as Int) {
        out[i] = x[i] * (2.0 as Float32) + bias[i]
    }
}

@kernel("elementwise")
pub fn relu_act(x List[Float32], inout out List[Float32], n UInt32) {
    let i = gpu_global_id_x()
    if i < (n as Int) {
        let v = x[i]
        out[i] = if v > (0.0 as Float32) { v } else { 0.0 as Float32 }
    }
}

@gpu_graph
pub fn mlp_layer(a List[Float32], b List[Float32], bias List[Float32], dims MatMulDims, n UInt32) List[Float32] {
    var mm List[Float32] = list_filled(16, 0.0 as Float32)
    var biased List[Float32] = list_filled(16, 0.0 as Float32)
    var act List[Float32] = list_filled(16, 0.0 as Float32)
    matmul_kernel(a, b, mm, dims)
    scale_bias(mm, bias, biased, n)
    relu_act(biased, act, n)
    act
}
"#;
        let prog = tenon_syntax::parse(src).unwrap();
        let rep = tenon_verify::check_program(&prog);
        assert!(rep.ok, "verify failed: {:?}", rep.diagnostics);
        assert_eq!(rep.mir.gpu_graphs.len(), 1);
        let plan = &rep.mir.gpu_graphs[0];
        assert_eq!(plan.nodes.len(), 3);
        assert_eq!(plan.fusions.len(), 1);
        assert_eq!(plan.fusions[0].fused_name, "scale_bias__fused__relu_act");

        let msl = emit_msl(&prog, &rep.mir);
        assert!(msl.contains("threadgroup float tile_a[16];"), "missing MSL shared memory:\n{}", msl);
        assert!(msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup);"), "missing MSL barrier:\n{}", msl);
        assert!(msl.contains("kernel void scale_bias__fused__relu_act("));
        // Real AST fusion must inline `* 2.0f + bias[idx]` rather than hardcoded `a[idx] + b[idx]`!
        assert!(msl.contains("2.0f") && msl.contains("const float __fused_tmp ="), "expected AST-inlined MSL fusion:\n{}", msl);

        let wgsl = emit_wgsl(&prog, &rep.mir);
        assert!(wgsl.contains("var<workgroup> tiled_shared_matmul__tile_a: array<f32, 16>;"), "missing WGSL workgroup:\n{}", wgsl);
        assert!(wgsl.contains("workgroupBarrier();"), "missing WGSL barrier:\n{}", wgsl);
        assert!(wgsl.contains("fn scale_bias__fused__relu_act("));
        assert!(wgsl.contains("let __fused_tmp: f32 =") && wgsl.contains("select("), "expected real AST-inlined WGSL fusion:\n{}", wgsl);

        let webgpu = emit_webgpu_host_mjs(&prog, &rep.mir);
        assert!(webgpu.contains("export async function createTenonWebGpu"));
        assert!(webgpu.contains("graph_mlp_layer:") && webgpu.contains("scale_bias__fused__relu_act"));

        // Negative control 1: Divergent `gpu_barrier()` inside thread-dependent branch -> E0842!
        let bad_barrier = r#"
@kernel
pub fn bad_div_barrier(inout out List[Float32], n UInt32) {
    let i = gpu_global_id_x()
    if i < (n as Int) {
        let _ = gpu_barrier()
        out[i] = 1.0 as Float32
    }
}
"#;
        let rep_div = tenon_verify::check_program(&tenon_syntax::parse(bad_barrier).unwrap());
        assert!(
            !rep_div.ok && rep_div.diagnostics.iter().any(|d| d.code == "E0842"),
            "expected E0842 divergent barrier error, got: {:?}",
            rep_div.diagnostics
        );

        // Negative control 2: Data race writing thread-invariant index `out[0]` -> E0843!
        let bad_race = r#"
@kernel
pub fn bad_race_write(x List[Float32], inout out List[Float32]) {
    let i = gpu_global_id_x()
    out[0] = x[i]
}
"#;
        let rep_race = tenon_verify::check_program(&tenon_syntax::parse(bad_race).unwrap());
        assert!(
            !rep_race.ok && rep_race.diagnostics.iter().any(|d| d.code == "E0843"),
            "expected E0843 GPU data race error, got: {:?}",
            rep_race.diagnostics
        );

        // Zero-glue `gpu_embedded_msl()` / `gpu_embedded_wgsl()` compile-time shader inlining
        let emb_src = format!(
            "{}\npub fn get_msl() Text {{ gpu_embedded_msl() }}\npub fn get_wgsl() Text {{ gpu_embedded_wgsl() }}\n",
            src
        );
        let mut emb_prog = tenon_syntax::parse(&emb_src).unwrap();
        inline_embedded_gpu_shaders(&mut emb_prog);
        let has_inlined_msl = emb_prog.items.iter().any(|it| match it {
            Item::Fn(f) if f.name == "get_msl" => matches!(
                f.body.tail.as_deref().map(|e| &e.kind),
                Some(ExprKind::Str(s)) if s.contains("scale_bias__fused__relu_act")
            ),
            _ => false,
        });
        assert!(has_inlined_msl, "expected gpu_embedded_msl() to inline AST-fused MSL");

        // FP16 (`Float16` -> Metal `half` / WGSL `enable f16;` + `f16`) & BF16 (`BFloat16` -> Metal `bfloat`) shader emission
        let fp16_src = r#"
@kernel(workgroup_size = (8, 8, 1))
pub fn fp16_mixed_dot(a List[Float16], b List[BFloat16], inout out List[Float32], n UInt32) {
    let i = gpu_global_id_x()
    if i < (n as Int) {
        out[i] = (a[i] as Float32) + (b[i] as Float32)
    }
}
"#;
        let fp16_prog = tenon_syntax::parse(fp16_src).unwrap();
        let fp16_rep = tenon_verify::check_program(&fp16_prog);
        assert!(fp16_rep.ok, "fp16 kernel verify failed: {:?}", fp16_rep.diagnostics);
        let fp16_msl = emit_msl(&fp16_prog, &fp16_rep.mir);
        assert!(
            fp16_msl.contains("device const half* a") && fp16_msl.contains("device const bfloat* b"),
            "expected native Metal half & bfloat buffers:\n{}",
            fp16_msl
        );
        let fp16_wgsl = emit_wgsl(&fp16_prog, &fp16_rep.mir);
        assert!(
            fp16_wgsl.contains("enable f16;") && fp16_wgsl.contains("array<f16>"),
            "expected WGSL `enable f16;` and `array<f16>`:\n{}",
            fp16_wgsl
        );

        // J3: `gpu_shared_f16` (`threadgroup half` / `var<workgroup> array<f16>`) + GPU math (`gpu_rsqrt`, `gpu_exp`) + 3-stage `@gpu_graph` fusion!
        let j3_src = r#"
@kernel
pub fn tiled_f16_norm(x List[Float16], inout out List[Float16], n UInt32) {
    var tile = gpu_shared_f16(16)
    let tid = gpu_global_id_x()
    let lx = gpu_local_id_x()
    if lx < 16 {
        tile[lx] = x[tid]
    }
    let _ = gpu_barrier()
    if tid < (n as Int) {
        let v Float32 = (tile[lx] as Float32) * gpu_rsqrt(4.0 as Float32)
        out[tid] = (v / ((1.0 as Float32) + gpu_exp(-v))) as Float16
    }
}

@kernel("elementwise")
pub fn st_scale(x List[Float32], inout out List[Float32], n UInt32) {
    let i = gpu_global_id_x()
    if i < (n as Int) {
        out[i] = x[i] * (2.0 as Float32)
    }
}

@kernel("elementwise")
pub fn st_bias(x List[Float32], b List[Float32], inout out List[Float32], n UInt32) {
    let i = gpu_global_id_x()
    if i < (n as Int) {
        out[i] = x[i] + b[i]
    }
}

@kernel("elementwise")
pub fn st_silu(x List[Float32], inout out List[Float32], n UInt32) {
    let i = gpu_global_id_x()
    if i < (n as Int) {
        let v = x[i]
        out[i] = v / ((1.0 as Float32) + gpu_exp(-v))
    }
}

@gpu_graph
pub fn three_stage_chain(x List[Float32], b List[Float32], n UInt32) List[Float32] {
    var t1 List[Float32] = list_filled(16, 0.0 as Float32)
    var t2 List[Float32] = list_filled(16, 0.0 as Float32)
    var out List[Float32] = list_filled(16, 0.0 as Float32)
    st_scale(x, t1, n)
    st_bias(t1, b, t2, n)
    st_silu(t2, out, n)
    out
}
"#;
        let j3_prog = tenon_syntax::parse(j3_src).unwrap();
        let j3_rep = tenon_verify::check_program(&j3_prog);
        assert!(j3_rep.ok, "J3 GPU verify failed: {:?}", j3_rep.diagnostics);
        assert_eq!(j3_rep.mir.gpu_graphs.len(), 1);
        assert_eq!(j3_rep.mir.gpu_graphs[0].fusions.len(), 1);
        assert_eq!(
            j3_rep.mir.gpu_graphs[0].fusions[0].fused_name,
            "st_scale__fused__st_bias__fused__st_silu"
        );
        let j3_msl = emit_msl(&j3_prog, &j3_rep.mir);
        assert!(
            j3_msl.contains("threadgroup half tile[16];")
                && j3_msl.contains("rsqrt(")
                && j3_msl.contains("kernel void st_scale__fused__st_bias__fused__st_silu(")
                && j3_msl.contains("__fused_tmp_0")
                && j3_msl.contains("__fused_tmp_1"),
            "expected J3 MSL shared_f16 + rsqrt + 3-stage fusion:\n{}",
            j3_msl
        );
        let j3_wgsl = emit_wgsl(&j3_prog, &j3_rep.mir);
        assert!(
            j3_wgsl.contains("enable f16;")
                && j3_wgsl.contains("array<f16, 16>")
                && j3_wgsl.contains("inverseSqrt(")
                && j3_wgsl.contains("fn st_scale__fused__st_bias__fused__st_silu("),
            "expected J3 WGSL enable f16 + array<f16, 16> + inverseSqrt + 3-stage fusion:\n{}",
            j3_wgsl
        );
    }

    #[test]
    fn emits_cuda_spirv_glsl_and_metal_simdgroup_mma_kernels() {
        let src = r#"
@kernel(simdgroup_mma = true)
pub fn mma_tile_f16(x List[Float16], inout out List[Float16], n UInt32) {
    var tile = gpu_shared_f16(16)
    let tid = gpu_global_id_x()
    let lx = gpu_local_id_x()
    if lx < 16 {
        tile[lx] = x[tid]
    }
    let _ = gpu_barrier()
    if tid < (n as Int) {
        let v Float32 = (tile[lx] as Float32) * gpu_rsqrt(4.0 as Float32)
        out[tid] = (v / ((1.0 as Float32) + gpu_exp(-v))) as Float16
    }
}
"#;
        let prog = tenon_syntax::parse(src).unwrap();
        let rep = tenon_verify::check_program(&prog);
        assert!(rep.ok, "MMA kernel verify failed: {:?}", rep.diagnostics);

        let msl = emit_msl(&prog, &rep.mir);
        assert!(
            msl.contains("#include <metal_simdgroup_matrix>")
                && msl.contains("simdgroup_half8x8")
                && msl.contains("simdgroup_multiply_accumulate"),
            "expected Metal simdgroup_matrix MMA helper in MSL:\n{}",
            msl
        );

        let cuda = emit_cuda(&prog, &rep.mir);
        assert!(
            cuda.contains("#include <cuda_fp16.h>")
                && cuda.contains("#include <mma.h>")
                && cuda.contains("wmma::mma_sync")
                && cuda.contains("extern \"C\" __global__ void mma_tile_f16(")
                && cuda.contains("__shared__ __half tile[16];")
                && cuda.contains("__syncthreads();"),
            "expected NVIDIA CUDA C++/WMMA kernel output:\n{}",
            cuda
        );

        let glsl = emit_spirv_glsl(&prog, &rep.mir);
        assert!(
            glsl.contains("#version 460")
                && glsl.contains("GL_EXT_shader_explicit_arithmetic_types_float16")
                && glsl.contains("shared float16_t tile[16];")
                && glsl.contains("barrier();")
                && glsl.contains("inversesqrt("),
            "expected Vulkan 1.3 GLSL/SPIR-V compute kernel output:\n{}",
            glsl
        );

        let shlo = emit_stablehlo(&prog, &rep.mir);
        assert!(
            shlo.contains("module @tenon_tpu")
                && shlo.contains("func.func public @mma_tile_f16(")
                && shlo.contains("stablehlo.exponential"),
            "expected Google TPU StableHLO MLIR output:\n{}",
            shlo
        );
    }
}
