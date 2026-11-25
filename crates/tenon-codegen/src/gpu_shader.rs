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
                "Float32" | "Float" | "Float64" | "Int32" | "Int" | "Int64" | "UInt32" | "UInt64" | "USize" | "Bool"
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

/// Scans a block for `let/var name = gpu_shared_f32(N)` declarations (used for WGSL module-level `var<workgroup>`).
fn collect_shared_arrays(blk: &Block, out: &mut Vec<(String, i64)>) {
    for stmt in &blk.stmts {
        if let Stmt::Let { name, value, .. } = stmt {
            if let ExprKind::Call { callee, args } = &value.kind {
                if matches!(&callee.kind, ExprKind::Ident(s) if s == "gpu_shared_f32")
                    && args.len() == 1
                {
                    if let ExprKind::Int(sz) = &args[0].value.kind {
                        out.push((name.clone(), (*sz).max(1)));
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
                // `let tile = gpu_shared_f32(N)` -> MSL `threadgroup float tile[N];` (WGSL hoisted to module scope!)
                if let ExprKind::Call { callee, args } = &value.kind {
                    if matches!(&callee.kind, ExprKind::Ident(s) if s == "gpu_shared_f32") && args.len() == 1 {
                        if let ExprKind::Int(sz) = &args[0].value.kind {
                            if target == ShaderTarget::Msl {
                                out.push_str(&format!("{}threadgroup float {}[{}];\n", pad, name, (*sz).max(1)));
                            }
                            continue;
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

/// Emits Apple Metal Shading Language (`.metal`) for all `@gpu_struct` records,
/// `@kernel` compute functions (including `threadgroup` shared memory & barriers),
/// and AST-fused `@gpu_graph` kernels in `program`.
pub fn emit_msl(program: &Program, verified: &VerifiedMir) -> String {
    emit_shaders(program, verified, ShaderTarget::Msl)
}

/// Emits portable WebGPU Shading Language (`.wgsl`) for all `@gpu_struct` records,
/// `@kernel` compute functions (including `var<workgroup>` shared memory & `workgroupBarrier()`),
/// and AST-fused `@gpu_graph` kernels in `program`.
pub fn emit_wgsl(program: &Program, verified: &VerifiedMir) -> String {
    emit_shaders(program, verified, ShaderTarget::Wgsl)
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
                    for (sname, sz) in &shared_arrs {
                        let mname = format!("{}__{}", f.name, sname);
                        out.push_str(&format!("var<workgroup> {}: array<f32, {}>;\n", mname, sz));
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

    // 3. Emit REAL AST-Fused `@gpu_graph` Kernels (BOTH MSL and WGSL!)
    let mut wgsl_group = 0usize;
    for plan in &verified.gpu_graphs {
        for fusion in &plan.fusions {
            if let (Some(&k1), Some(&k2)) = (
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

/// Replaces zero-argument `gpu_embedded_msl()` / `gpu_embedded_wgsl()` calls with compile-time
/// synthesized MSL/WGSL string literals (including AST-fused `@gpu_graph` kernels), enabling
/// zero-glue single-binary GPU execution with zero external shader files or Perceus leaks!
pub fn inline_embedded_gpu_shaders(prog: &mut Program) {
    let mut needs_msl = false;
    let mut needs_wgsl = false;
    for item in &prog.items {
        match item {
            Item::Fn(f) => scan_block_for_embedded_gpu(&f.body, &mut needs_msl, &mut needs_wgsl),
            Item::Impl(imp) => {
                for m in &imp.methods {
                    scan_block_for_embedded_gpu(&m.body, &mut needs_msl, &mut needs_wgsl);
                }
            }
            _ => {}
        }
    }
    if !needs_msl && !needs_wgsl {
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

    let msl_src = if needs_msl { Some(emit_msl(prog, &mir)) } else { None };
    let wgsl_src = if needs_wgsl { Some(emit_wgsl(prog, &mir)) } else { None };

    for item in &mut prog.items {
        match item {
            Item::Fn(f) => replace_block_embedded_gpu(&mut f.body, msl_src.as_deref(), wgsl_src.as_deref()),
            Item::Impl(imp) => {
                for m in &mut imp.methods {
                    replace_block_embedded_gpu(&mut m.body, msl_src.as_deref(), wgsl_src.as_deref());
                }
            }
            _ => {}
        }
    }
}

fn scan_block_for_embedded_gpu(blk: &Block, needs_msl: &mut bool, needs_wgsl: &mut bool) {
    for s in &blk.stmts {
        match s {
            Stmt::Let { value, .. } | Stmt::Expr { expr: value, .. } => {
                scan_expr_for_embedded_gpu(value, needs_msl, needs_wgsl);
            }
            Stmt::Assign { target, value, .. } => {
                scan_expr_for_embedded_gpu(target, needs_msl, needs_wgsl);
                scan_expr_for_embedded_gpu(value, needs_msl, needs_wgsl);
            }
            Stmt::Return { value: Some(v), .. } => scan_expr_for_embedded_gpu(v, needs_msl, needs_wgsl),
            Stmt::Assert { cond, .. } => scan_expr_for_embedded_gpu(cond, needs_msl, needs_wgsl),
            Stmt::While { cond, body, .. } => {
                scan_expr_for_embedded_gpu(cond, needs_msl, needs_wgsl);
                scan_block_for_embedded_gpu(body, needs_msl, needs_wgsl);
            }
            Stmt::For { iter, body, .. } => {
                scan_expr_for_embedded_gpu(iter, needs_msl, needs_wgsl);
                scan_block_for_embedded_gpu(body, needs_msl, needs_wgsl);
            }
            Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } => {
                scan_block_for_embedded_gpu(body, needs_msl, needs_wgsl);
            }
            _ => {}
        }
    }
    if let Some(t) = &blk.tail {
        scan_expr_for_embedded_gpu(t, needs_msl, needs_wgsl);
    }
}

fn scan_expr_for_embedded_gpu(expr: &Expr, needs_msl: &mut bool, needs_wgsl: &mut bool) {
    match &expr.kind {
        ExprKind::Call { callee, args } => {
            if args.is_empty() {
                if let ExprKind::Ident(name) = &callee.kind {
                    if name == "gpu_embedded_msl" {
                        *needs_msl = true;
                        return;
                    }
                    if name == "gpu_embedded_wgsl" {
                        *needs_wgsl = true;
                        return;
                    }
                }
            }
            scan_expr_for_embedded_gpu(callee, needs_msl, needs_wgsl);
            for a in args {
                scan_expr_for_embedded_gpu(&a.value, needs_msl, needs_wgsl);
            }
        }
        ExprKind::Binary { left, right, .. } => {
            scan_expr_for_embedded_gpu(left, needs_msl, needs_wgsl);
            scan_expr_for_embedded_gpu(right, needs_msl, needs_wgsl);
        }
        ExprKind::Unary { expr: e, .. }
        | ExprKind::Cast { expr: e, .. }
        | ExprKind::Try(e)
        | ExprKind::Field { target: e, .. }
        | ExprKind::TupleIndex { target: e, .. } => scan_expr_for_embedded_gpu(e, needs_msl, needs_wgsl),
        ExprKind::Index { target, index } => {
            scan_expr_for_embedded_gpu(target, needs_msl, needs_wgsl);
            scan_expr_for_embedded_gpu(index, needs_msl, needs_wgsl);
        }
        ExprKind::If { cond, then_block, else_branch } => {
            scan_expr_for_embedded_gpu(cond, needs_msl, needs_wgsl);
            scan_block_for_embedded_gpu(then_block, needs_msl, needs_wgsl);
            if let Some(eb) = else_branch {
                scan_expr_for_embedded_gpu(eb, needs_msl, needs_wgsl);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            scan_expr_for_embedded_gpu(scrutinee, needs_msl, needs_wgsl);
            for arm in arms {
                if let Some(g) = &arm.guard {
                    scan_expr_for_embedded_gpu(g, needs_msl, needs_wgsl);
                }
                scan_expr_for_embedded_gpu(&arm.body, needs_msl, needs_wgsl);
            }
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) => scan_block_for_embedded_gpu(b, needs_msl, needs_wgsl),
        ExprKind::List(es) | ExprKind::Tuple(es) => {
            for e in es {
                scan_expr_for_embedded_gpu(e, needs_msl, needs_wgsl);
            }
        }
        ExprKind::RecordLit { fields, .. } => {
            for (_, v) in fields {
                scan_expr_for_embedded_gpu(v, needs_msl, needs_wgsl);
            }
        }
        _ => {}
    }
}

fn replace_block_embedded_gpu(blk: &mut Block, msl: Option<&str>, wgsl: Option<&str>) {
    for s in &mut blk.stmts {
        match s {
            Stmt::Let { value, .. } | Stmt::Expr { expr: value, .. } => {
                replace_expr_embedded_gpu(value, msl, wgsl);
            }
            Stmt::Assign { target, value, .. } => {
                replace_expr_embedded_gpu(target, msl, wgsl);
                replace_expr_embedded_gpu(value, msl, wgsl);
            }
            Stmt::Return { value: Some(v), .. } => replace_expr_embedded_gpu(v, msl, wgsl),
            Stmt::Assert { cond, .. } => replace_expr_embedded_gpu(cond, msl, wgsl),
            Stmt::While { cond, body, .. } => {
                replace_expr_embedded_gpu(cond, msl, wgsl);
                replace_block_embedded_gpu(body, msl, wgsl);
            }
            Stmt::For { iter, body, .. } => {
                replace_expr_embedded_gpu(iter, msl, wgsl);
                replace_block_embedded_gpu(body, msl, wgsl);
            }
            Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } => {
                replace_block_embedded_gpu(body, msl, wgsl);
            }
            _ => {}
        }
    }
    if let Some(t) = &mut blk.tail {
        replace_expr_embedded_gpu(t, msl, wgsl);
    }
}

fn replace_expr_embedded_gpu(expr: &mut Expr, msl: Option<&str>, wgsl: Option<&str>) {
    if let ExprKind::Call { callee, args } = &expr.kind {
        if args.is_empty() {
            if let ExprKind::Ident(name) = &callee.kind {
                if name == "gpu_embedded_msl" {
                    if let Some(s) = msl {
                        expr.kind = ExprKind::Str(s.to_string());
                        return;
                    }
                }
                if name == "gpu_embedded_wgsl" {
                    if let Some(s) = wgsl {
                        expr.kind = ExprKind::Str(s.to_string());
                        return;
                    }
                }
            }
        }
    }
    match &mut expr.kind {
        ExprKind::Call { callee, args } => {
            replace_expr_embedded_gpu(callee, msl, wgsl);
            for a in args {
                replace_expr_embedded_gpu(&mut a.value, msl, wgsl);
            }
        }
        ExprKind::Binary { left, right, .. } => {
            replace_expr_embedded_gpu(left, msl, wgsl);
            replace_expr_embedded_gpu(right, msl, wgsl);
        }
        ExprKind::Unary { expr: e, .. }
        | ExprKind::Cast { expr: e, .. }
        | ExprKind::Try(e)
        | ExprKind::Field { target: e, .. }
        | ExprKind::TupleIndex { target: e, .. } => replace_expr_embedded_gpu(e, msl, wgsl),
        ExprKind::Index { target, index } => {
            replace_expr_embedded_gpu(target, msl, wgsl);
            replace_expr_embedded_gpu(index, msl, wgsl);
        }
        ExprKind::If { cond, then_block, else_branch } => {
            replace_expr_embedded_gpu(cond, msl, wgsl);
            replace_block_embedded_gpu(then_block, msl, wgsl);
            if let Some(eb) = else_branch {
                replace_expr_embedded_gpu(eb, msl, wgsl);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            replace_expr_embedded_gpu(scrutinee, msl, wgsl);
            for arm in arms {
                if let Some(g) = &mut arm.guard {
                    replace_expr_embedded_gpu(g, msl, wgsl);
                }
                replace_expr_embedded_gpu(&mut arm.body, msl, wgsl);
            }
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) => replace_block_embedded_gpu(b, msl, wgsl),
        ExprKind::List(es) | ExprKind::Tuple(es) => {
            for e in es {
                replace_expr_embedded_gpu(e, msl, wgsl);
            }
        }
        ExprKind::RecordLit { fields, .. } => {
            for (_, v) in fields {
                replace_expr_embedded_gpu(v, msl, wgsl);
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
    }
}
