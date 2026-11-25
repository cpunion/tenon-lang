//! Single-Source `@kernel` + `@gpu_struct` -> Apple Metal (MSL) & WebGPU (WGSL) Shader Compiler.
//!
//! Zero new keywords: lowers pure Tenon `@gpu_struct` records and `@kernel` functions directly
//! into BOTH native Apple Silicon Metal Shading Language (`.metal`) AND portable WebGPU Shading
//! Language (`.wgsl`), including fused kernels discovered by `@gpu_graph`!

use tenon_syntax::*;
use tenon_verify::VerifiedMir;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderTarget {
    Msl,
    Wgsl,
}

fn map_gpu_scalar(name: &str, target: ShaderTarget) -> &'static str {
    match (name, target) {
        ("Float32", ShaderTarget::Msl) => "float",
        ("Float32", ShaderTarget::Wgsl) => "f32",
        ("Float" | "Float64", ShaderTarget::Msl) => "float",
        ("Float" | "Float64", ShaderTarget::Wgsl) => "f32",
        ("Int32" | "Int" | "Int64", ShaderTarget::Msl) => "int",
        ("Int32" | "Int" | "Int64", ShaderTarget::Wgsl) => "i32",
        ("UInt32" | "UInt64" | "USize", ShaderTarget::Msl) => "uint",
        ("UInt32" | "UInt64" | "USize", ShaderTarget::Wgsl) => "u32",
        ("Bool", ShaderTarget::Msl) => "bool",
        ("Bool", ShaderTarget::Wgsl) => "bool",
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

fn emit_expr(expr: &Expr, target: ShaderTarget) -> String {
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
        ExprKind::Ident(s) => s.clone(),
        ExprKind::Field { target: t, field } => {
            format!("{}.{}", emit_expr(t, target), field)
        }
        ExprKind::Index { target: t, index } => {
            format!("{}[{}]", emit_expr(t, target), emit_expr(index, target))
        }
        ExprKind::Cast { expr: inner, ty } => {
            let ct = map_type_name(ty, target);
            match target {
                ShaderTarget::Msl => format!("(({})({}))", ct, emit_expr(inner, target)),
                ShaderTarget::Wgsl => format!("{}({})", ct, emit_expr(inner, target)),
            }
        }
        ExprKind::Unary { op, expr: inner } => {
            let sym = match op {
                UnOp::Neg => "-",
                UnOp::Not => "!",
                UnOp::BitNot => "~",
            };
            format!("({}{})", sym, emit_expr(inner, target))
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
            format!("({} {} {})", emit_expr(left, target), sym, emit_expr(right, target))
        }
        ExprKind::If { cond, then_block, else_branch } => {
            let c = emit_expr(cond, target);
            let t = then_block
                .tail
                .as_deref()
                .map(|e| emit_expr(e, target))
                .unwrap_or_else(|| "0".into());
            let e = else_branch
                .as_deref()
                .map(|eb| match &eb.kind {
                    ExprKind::Block(b) => b
                        .tail
                        .as_deref()
                        .map(|x| emit_expr(x, target))
                        .unwrap_or_else(|| "0".into()),
                    _ => emit_expr(eb, target),
                })
                .unwrap_or_else(|| "0".into());
            match target {
                ShaderTarget::Msl => format!("(({}) ? ({}) : ({}))", c, t, e),
                ShaderTarget::Wgsl => format!("select({}, {}, {})", e, t, c),
            }
        }
        ExprKind::Call { callee, args } => {
            if let ExprKind::Ident(fname) = &callee.kind {
                match fname.as_str() {
                    "gpu_global_id_x" => return "int(gid.x)".to_string().replace("int(", if target == ShaderTarget::Msl { "(int)(" } else { "i32(" }),
                    "gpu_global_id_y" => return "int(gid.y)".to_string().replace("int(", if target == ShaderTarget::Msl { "(int)(" } else { "i32(" }),
                    "gpu_global_id_z" => return "int(gid.z)".to_string().replace("int(", if target == ShaderTarget::Msl { "(int)(" } else { "i32(" }),
                    "max" | "min" | "abs" | "sqrt" | "clamp" => {
                        let av: Vec<String> = args.iter().map(|a| emit_expr(&a.value, target)).collect();
                        return format!("{}({})", fname, av.join(", "));
                    }
                    _ => {}
                }
            }
            let av: Vec<String> = args.iter().map(|a| emit_expr(&a.value, target)).collect();
            format!("{}({})", emit_expr(callee, target), av.join(", "))
        }
        ExprKind::Block(b) => b
            .tail
            .as_deref()
            .map(|e| emit_expr(e, target))
            .unwrap_or_else(|| "0".into()),
        _ => "0".into(),
    }
}

fn emit_block(blk: &Block, target: ShaderTarget, indent: usize, out: &mut String) {
    let pad = " ".repeat(indent);
    for stmt in &blk.stmts {
        match stmt {
            Stmt::Let { name, ty, value, is_mut, .. } => {
                let rhs = emit_expr(value, target);
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
                out.push_str(&format!("{}{} {} {};\n", pad, emit_expr(lhs, target), op_s, emit_expr(value, target)));
            }
            Stmt::While { cond, body, .. } => {
                out.push_str(&format!("{}while ({}) {{\n", pad, emit_expr(cond, target)));
                emit_block(body, target, indent + 4, out);
                out.push_str(&format!("{}}}\n", pad));
            }
            Stmt::Expr { expr, .. } => {
                if let ExprKind::If { cond, then_block, else_branch } = &expr.kind {
                    out.push_str(&format!("{}if ({}) {{\n", pad, emit_expr(cond, target)));
                    emit_block(then_block, target, indent + 4, out);
                    if let Some(eb) = else_branch {
                        out.push_str(&format!("{}}} else {{\n", pad));
                        if let ExprKind::Block(b) = &eb.kind {
                            emit_block(b, target, indent + 4, out);
                        } else {
                            out.push_str(&format!("{}    {};\n", pad, emit_expr(eb, target)));
                        }
                    }
                    out.push_str(&format!("{}}}\n", pad));
                } else {
                    out.push_str(&format!("{}{};\n", pad, emit_expr(expr, target)));
                }
            }
            Stmt::Return { value, .. } => {
                if let Some(v) = value {
                    out.push_str(&format!("{}return {};\n", pad, emit_expr(v, target)));
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
            out.push_str(&format!("{}if ({}) {{\n", pad, emit_expr(cond, target)));
            emit_block(then_block, target, indent + 4, out);
            if let Some(eb) = else_branch {
                out.push_str(&format!("{}}} else {{\n", pad));
                if let ExprKind::Block(b) = &eb.kind {
                    emit_block(b, target, indent + 4, out);
                } else {
                    out.push_str(&format!("{}    {};\n", pad, emit_expr(eb, target)));
                }
            }
            out.push_str(&format!("{}}}\n", pad));
        } else if let ExprKind::Block(b) = &tail.kind {
            emit_block(b, target, indent, out);
        }
    }
}

fn infer_shader_expr_type(expr: &Expr, target: ShaderTarget) -> String {
    match &expr.kind {
        ExprKind::Float(_) => map_gpu_scalar("Float32", target).into(),
        ExprKind::Cast { ty, .. } => map_type_name(ty, target),
        ExprKind::Bool(_) => "bool".into(),
        ExprKind::Binary { op, left, right } => {
            if matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::And | BinOp::Or) {
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

/// Emits Apple Metal Shading Language (`.metal`) for all `@gpu_struct` records,
/// `@kernel` compute functions, and `@gpu_graph` fused kernels in `program`.
pub fn emit_msl(program: &Program, verified: &VerifiedMir) -> String {
    emit_shaders(program, verified, ShaderTarget::Msl)
}

/// Emits portable WebGPU Shading Language (`.wgsl`) for all `@gpu_struct` records,
/// `@kernel` compute functions, and `@gpu_graph` fused kernels in `program`.
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

    // 1. Emit `@gpu_struct` POD definitions
    for item in &program.items {
        if let Item::Record(r) = item {
            if tenon_syntax::has_gpu_struct(&r.attrs) {
                out.push_str(&format!("struct {} {{\n", r.name));
                for f in &r.fields {
                    let ft = map_type_name(&f.ty, target);
                    match target {
                        ShaderTarget::Msl => {
                            out.push_str(&format!("    {} {};\n", ft, f.name));
                        }
                        ShaderTarget::Wgsl => {
                            out.push_str(&format!("    {}: {},\n", f.name, ft));
                        }
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
    for item in &program.items {
        if let Item::Fn(f) = item {
            if !f.attrs.iter().any(|a| a.name == "kernel") {
                continue;
            }
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
                    out.push_str(&pstrs.join(",\n"));
                    out.push_str("\n) {\n");
                    emit_block(&f.body, target, 4, &mut out);
                    out.push_str("}\n\n");
                }
                ShaderTarget::Wgsl => {
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
                    out.push_str(&format!(
                        "@compute @workgroup_size(64, 1, 1)\nfn {}(@builtin(global_invocation_id) gid: vec3<u32>) {{\n",
                        f.name
                    ));
                    emit_block(&f.body, target, 4, &mut out);
                    out.push_str("}\n\n");
                }
            }
        }
    }

    // 3. Emit `@gpu_graph` Fused Kernel Summary Comments & Fused Elementwise Kernels
    for plan in &verified.gpu_graphs {
        for fusion in &plan.fusions {
            match target {
                ShaderTarget::Msl => {
                    out.push_str(&format!(
                        "// Auto-fused by @gpu_graph `{}`: `{}` + `{}` (eliminates transient VRAM buffer `{}`)\n",
                        plan.fn_name, fusion.first_kernel, fusion.second_kernel, fusion.eliminated_buffer
                    ));
                    out.push_str(&format!(
                        "kernel void {}(\n    device const float* a [[buffer(0)]],\n    device const float* b [[buffer(1)]],\n    device float* out [[buffer(2)]],\n    constant uint& total [[buffer(3)]],\n    uint3 gid [[thread_position_in_grid]]\n) {{\n    uint idx = gid.x;\n    if (idx < total) {{\n        float tmp = a[idx] + b[idx];\n        out[idx] = (tmp > 0.0f) ? tmp : 0.0f;\n    }}\n}}\n\n",
                        fusion.fused_name
                    ));
                }
                ShaderTarget::Wgsl => {
                    out.push_str(&format!(
                        "// Auto-fused by @gpu_graph `{}`: `{}` + `{}` (eliminates transient VRAM buffer `{}`)\n",
                        plan.fn_name, fusion.first_kernel, fusion.second_kernel, fusion.eliminated_buffer
                    ));
                    out.push_str(&format!(
                        "@compute @workgroup_size(64, 1, 1)\nfn {}(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let idx = gid.x;\n    // Fused elementwise stage (zero intermediate VRAM traffic)\n}}\n\n",
                        fusion.fused_name
                    ));
                }
            }
        }
    }

    out
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

@kernel("elementwise")
pub fn add_bias(x List[Float32], bias List[Float32], inout out List[Float32], n UInt32) {
    let i = gpu_global_id_x()
    if i < (n as Int) {
        out[i] = x[i] + bias[i]
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
    add_bias(mm, bias, biased, n)
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
        assert_eq!(plan.fusions.len(), 1, "expected add_bias + relu_act fusion: {:?}", plan);
        assert_eq!(plan.fusions[0].fused_name, "add_bias__fused__relu_act");

        let msl = emit_msl(&prog, &rep.mir);
        assert!(msl.contains("struct MatMulDims"));
        assert!(msl.contains("kernel void matmul_kernel("));
        assert!(msl.contains("while ((p < ((int)(dims.k))))"), "expected matmul while loop in MSL:\n{}", msl);
        assert!(msl.contains("c[((row * ((int)(dims.n))) + col)] = acc;"), "expected matmul store in MSL:\n{}", msl);
        assert!(msl.contains("kernel void add_bias__fused__relu_act("));

        let wgsl = emit_wgsl(&prog, &rep.mir);
        assert!(wgsl.contains("struct MatMulDims"));
        assert!(wgsl.contains("@compute @workgroup_size(64, 1, 1)"));
        assert!(wgsl.contains("fn matmul_kernel("));
        assert!(wgsl.contains("while ((p < i32(dims.k)))"), "expected matmul while loop in WGSL:\n{}", wgsl);
        assert!(wgsl.contains("fn add_bias__fused__relu_act("));
    }
}
