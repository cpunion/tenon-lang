//! Cross-Platform Multi-Target Binding Generators (ADR-0001 D08: Single-Source -> Multi-End).
//! Emits rich, idiomatic bindings directly from verified Tenon AST signatures:
//! - TypeScript (`.d.ts` + runtime Wasm linear-memory UTF-8/Record/Enum/Result loader)
//! - Swift (`.swift` structs, `TenonError`, `@_silgen_name` C-ABI wrappers)
//! - Kotlin (`.kt` `data class`es + JNI/FFI `TenonNative`)
//! - Go (`cgo` `.go` wrapper calling static/shared C-ABI Tenon libraries!)

use std::collections::BTreeMap;
use tenon_syntax::*;

fn resolve_base<'a>(ty: &'a TypeExpr, aliases: &'a BTreeMap<String, TypeExpr>) -> &'a TypeExpr {
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

fn collect_aliases(program: &Program) -> BTreeMap<String, TypeExpr> {
    let mut map = BTreeMap::new();
    for item in &program.items {
        if let Item::TypeAlias(ta) = item {
            map.insert(ta.name.clone(), ta.base.clone());
        }
    }
    map
}

fn export_symbol_name(f: &FnDecl) -> String {
    for a in &f.attrs {
        if a.name == "export" {
            if let Some(arg) = a.args.first() {
                if let ExprKind::Str(s) = &arg.value.kind {
                    return s.clone();
                }
            }
        }
    }
    f.name.clone()
}

fn ts_type(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> String {
    match resolve_base(ty, aliases) {
        TypeExpr::Named { name, args, .. } => match name.as_str() {
            "Float" | "Float64" | "Float32" => "number".into(),
            "Bool" => "boolean".into(),
            "Text" => "string".into(),
            "Option" if args.len() == 1 => format!("{} | null", ts_type(&args[0], aliases)),
            "Result" if args.len() == 2 => {
                format!("TenonResult<{}, {}>", ts_type(&args[0], aliases), ts_type(&args[1], aliases))
            }
            "List" if args.len() == 1 => format!("Array<{}>", ts_type(&args[0], aliases)),
            "Int" | "Int8" | "Int16" | "Int32" | "Int64" | "UInt8" | "UInt16" | "UInt32"
            | "UInt64" | "Byte" | "CInt" | "CUInt" | "CSize" => "bigint".into(),
            other => other.to_string(),
        },
        _ => "bigint".into(),
    }
}

fn ts_wasm_abi_type(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> &'static str {
    match resolve_base(ty, aliases) {
        TypeExpr::Named { name, .. } => match name.as_str() {
            "Float" | "Float64" | "Float32" => "number",
            "Bool" => "boolean",
            _ => "bigint",
        },
        _ => "bigint",
    }
}

fn swift_type(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> String {
    match resolve_base(ty, aliases) {
        TypeExpr::Named { name, args, .. } => match name.as_str() {
            "Int8" => "Int8".into(),
            "Int16" => "Int16".into(),
            "Int32" | "CInt" => "Int32".into(),
            "UInt8" | "Byte" => "UInt8".into(),
            "UInt16" => "UInt16".into(),
            "UInt32" | "CUInt" => "UInt32".into(),
            "UInt64" | "USize" | "CSize" => "UInt64".into(),
            "Float32" => "Float".into(),
            "Float" | "Float64" => "Double".into(),
            "Bool" => "Bool".into(),
            "Text" => "String".into(),
            "Option" if args.len() == 1 => format!("{}?", swift_type(&args[0], aliases)),
            "List" if args.len() == 1 => format!("[{}]", swift_type(&args[0], aliases)),
            "Int" | "Int64" => "Int64".into(),
            other => other.to_string(),
        },
        _ => "Int64".into(),
    }
}

fn kotlin_type(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> String {
    match resolve_base(ty, aliases) {
        TypeExpr::Named { name, args, .. } => match name.as_str() {
            "Int8" | "UInt8" | "Byte" => "Byte".into(),
            "Int16" | "UInt16" => "Short".into(),
            "Int32" | "UInt32" | "CInt" | "CUInt" => "Int".into(),
            "Float32" => "Float".into(),
            "Float" | "Float64" => "Double".into(),
            "Bool" => "Boolean".into(),
            "Text" => "String".into(),
            "Option" if args.len() == 1 => format!("{}?", kotlin_type(&args[0], aliases)),
            "List" if args.len() == 1 => format!("List<{}>", kotlin_type(&args[0], aliases)),
            "Int" | "Int64" | "UInt64" | "USize" | "CSize" => "Long".into(),
            other => other.to_string(),
        },
        _ => "Long".into(),
    }
}

fn go_cgo_type(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> (&'static str, &'static str) {
    match resolve_base(ty, aliases) {
        TypeExpr::Named { name, .. } => match name.as_str() {
            "Int32" | "CInt" => ("int32", "C.int32_t"),
            "UInt32" | "CUInt" => ("uint32", "C.uint32_t"),
            "UInt64" | "USize" | "CSize" => ("uint64", "C.uint64_t"),
            "Float32" => ("float32", "C.float"),
            "Float" | "Float64" => ("float64", "C.double"),
            "Bool" => ("bool", "C.bool"),
            _ => ("int64", "C.int64_t"),
        },
        _ => ("int64", "C.int64_t"),
    }
}

/// Emits rich TypeScript declarations (Record interfaces, Enum tagged unions, `TenonResult`,
/// and Wasm32 linear-memory UTF-8 runtime bridge) for all exported `pub fn`s.
pub fn emit_typescript_bindings(program: &Program) -> String {
    let aliases = collect_aliases(program);
    let mut out = String::from("// Auto-generated by Tenon (`--emit-ts`). Do not edit manually.\n\n");
    out.push_str("export type TenonResult<T, E> = { readonly ok: true; readonly value: T } | { readonly ok: false; readonly error: E };\n\n");

    for item in &program.items {
        match item {
            Item::Record(r) if r.is_pub && r.type_params.is_empty() => {
                out.push_str(&format!("export interface {} {{\n", r.name));
                for f in &r.fields {
                    out.push_str(&format!("  readonly {}: {};\n", f.name, ts_type(&f.ty, &aliases)));
                }
                out.push_str("}\n\n");
            }
            Item::Enum(e) if e.is_pub && e.type_params.is_empty() => {
                let mut vars = Vec::new();
                for v in &e.variants {
                    if v.fields.is_empty() {
                        vars.push(format!("{{ readonly tag: '{}' }}", v.name));
                    } else {
                        let fs: Vec<String> = v
                            .fields
                            .iter()
                            .enumerate()
                            .map(|(i, fl)| format!("readonly f{}: {}", i, ts_type(&fl.ty, &aliases)))
                            .collect();
                        vars.push(format!("{{ readonly tag: '{}'; {} }}", v.name, fs.join("; ")));
                    }
                }
                if !vars.is_empty() {
                    out.push_str(&format!("export type {} = {};\n\n", e.name, vars.join(" | ")));
                }
            }
            _ => {}
        }
    }

    out.push_str("export interface TenonWasmExports extends WebAssembly.Exports {\n");
    out.push_str("  readonly memory: WebAssembly.Memory;\n");
    out.push_str("  tenon_wasm_live_objects?(): bigint;\n");
    out.push_str("  tenon_wasm_alloc_bytes?(n: number): number;\n");
    out.push_str("  tenon_wasm_text_from_mem?(ptr: number, len: number): bigint;\n");
    out.push_str("  tenon_wasm_text_ptr?(handle: bigint): number;\n");
    out.push_str("  tenon_wasm_text_len?(handle: bigint): number;\n");
    out.push_str("  tenon_wasm_rc_dec?(handle: bigint): void;\n");
    for item in &program.items {
        if let Item::Fn(f) = item {
            if (f.is_pub || f.name == "main") && !f.is_ghost && !f.is_test && f.type_params.is_empty() {
                let ps: Vec<String> = f
                    .params
                    .iter()
                    .map(|p| format!("{}: {}", p.name, ts_wasm_abi_type(&p.ty, &aliases)))
                    .collect();
                let rt = f
                    .ret_ty
                    .as_ref()
                    .map(|t| ts_wasm_abi_type(t, &aliases))
                    .unwrap_or("void");
                out.push_str(&format!("  {}({}): {};\n", export_symbol_name(f), ps.join(", "), rt));
            }
        }
    }
    out.push_str("}\n\n");
    out.push_str("export interface TenonWasmModule {\n");
    out.push_str("  readonly exports: TenonWasmExports;\n");
    out.push_str("  writeText(s: string): bigint;\n");
    out.push_str("  readText(handle: bigint): string;\n");
    out.push_str("  freeHandle(handle: bigint): void;\n");
    out.push_str("  liveObjects(): number;\n");
    for item in &program.items {
        if let Item::Fn(f) = item {
            if (f.is_pub || f.name == "main") && !f.is_ghost && !f.is_test && f.type_params.is_empty() {
                let ps: Vec<String> = f
                    .params
                    .iter()
                    .map(|p| format!("{}: {}", p.name, ts_high_level_type(&p.ty, &aliases)))
                    .collect();
                let rt = f
                    .ret_ty
                    .as_ref()
                    .map(|t| ts_high_level_type(t, &aliases))
                    .unwrap_or_else(|| "void".into());
                out.push_str(&format!("  {}({}): {};\n", export_symbol_name(f), ps.join(", "), rt));
            }
        }
    }
    out.push_str("}\n\n");
    out.push_str("export declare function loadTenonWasm(bytes: BufferSource): Promise<TenonWasmModule>;\n");
    out
}

fn ts_high_level_type(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> String {
    match resolve_base(ty, aliases) {
        TypeExpr::Named { name, .. } => match name.as_str() {
            "Text" => "string".into(),
            "Bool" => "boolean".into(),
            "Float" | "Float64" | "Float32" => "number".into(),
            "Int" | "Int8" | "Int16" | "Int32" | "Int64" | "UInt8" | "UInt16" | "UInt32"
            | "UInt64" | "Byte" | "CInt" | "CUInt" | "CSize" => "number | bigint".into(),
            _ => "bigint".into(),
        },
        _ => "bigint".into(),
    }
}

fn is_text_ty(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> bool {
    matches!(resolve_base(ty, aliases), TypeExpr::Named { name, .. } if name == "Text")
}

fn is_bool_ty(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> bool {
    matches!(resolve_base(ty, aliases), TypeExpr::Named { name, .. } if name == "Bool")
}

fn is_int_ty(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> bool {
    matches!(
        resolve_base(ty, aliases),
        TypeExpr::Named { name, .. }
            if matches!(
                name.as_str(),
                "Int" | "Int8" | "Int16" | "Int32" | "Int64" | "ISize"
                | "UInt8" | "UInt16" | "UInt32" | "UInt64" | "USize" | "Byte"
                | "CInt" | "CUInt" | "CSize"
            )
    )
}

/// Emits a zero-dependency ES Module (`.mjs`) Wasm32 Auto-Object Bridge (`--emit-mjs`):
/// automatically marshals JS `string`, `boolean`, and `number | bigint` across Wasm linear memory
/// and deterministically releases all temporary Perceus RC handles in `finally` blocks (`liveObjects() === 0`).
pub fn emit_wasm_mjs_loader(program: &Program) -> String {
    let aliases = collect_aliases(program);
    let mut out = String::from(
        "// Auto-generated by Tenon Wasm32 Auto-Object Bridge (`--emit-mjs`). Do not edit manually.\n\n\
const _enc = new TextEncoder();\n\
const _dec = new TextDecoder('utf-8');\n\n\
export async function loadTenonWasm(bytes, imports = {}) {\n\
  const { instance } = await WebAssembly.instantiate(bytes, imports);\n\
  const ex = instance.exports;\n\
  function writeText(s) {\n\
    const raw = _enc.encode(String(s));\n\
    const ptr = ex.tenon_wasm_alloc_bytes(raw.length);\n\
    new Uint8Array(ex.memory.buffer, ptr, raw.length).set(raw);\n\
    return ex.tenon_wasm_text_from_mem(ptr, raw.length);\n\
  }\n\
  function readText(handle) {\n\
    const ptr = ex.tenon_wasm_text_ptr(handle);\n\
    const len = ex.tenon_wasm_text_len(handle);\n\
    return _dec.decode(new Uint8Array(ex.memory.buffer, ptr, len));\n\
  }\n\
  function freeHandle(handle) {\n\
    if (handle && ex.tenon_wasm_rc_dec) ex.tenon_wasm_rc_dec(handle);\n\
  }\n\
  function liveObjects() {\n\
    return ex.tenon_wasm_live_objects ? Number(ex.tenon_wasm_live_objects()) : 0;\n\
  }\n\
  return {\n\
    exports: ex,\n\
    writeText,\n\
    readText,\n\
    freeHandle,\n\
    liveObjects,\n",
    );

    for item in &program.items {
        if let Item::Fn(f) = item {
            if (f.is_pub || f.name == "main") && !f.is_ghost && !f.is_test && f.type_params.is_empty() {
                let sym = export_symbol_name(f);
                let pnames: Vec<String> = f.params.iter().map(|p| p.name.clone()).collect();
                out.push_str(&format!("    {}({}) {{\n", sym, pnames.join(", ")));
                let mut cleanup_vars = Vec::new();
                let mut call_args = Vec::new();
                for p in &f.params {
                    if is_text_ty(&p.ty, &aliases) {
                        let hv = format!("__h_{}", p.name);
                        out.push_str(&format!("      const {} = writeText({});\n", hv, p.name));
                        cleanup_vars.push(hv.clone());
                        call_args.push(hv);
                    } else if is_bool_ty(&p.ty, &aliases) {
                        call_args.push(format!("({} ? 1n : 0n)", p.name));
                    } else if is_int_ty(&p.ty, &aliases) {
                        call_args.push(format!("BigInt({})", p.name));
                    } else {
                        call_args.push(p.name.clone());
                    }
                }
                out.push_str("      try {\n");
                let call_expr = format!("ex.{}({})", sym, call_args.join(", "));
                match &f.ret_ty {
                    Some(rt) if is_text_ty(rt, &aliases) => {
                        out.push_str(&format!("        const __ret = {};\n", call_expr));
                        out.push_str("        try { return readText(__ret); } finally { freeHandle(__ret); }\n");
                    }
                    Some(rt) if is_bool_ty(rt, &aliases) => {
                        out.push_str(&format!("        return ({}) !== 0n;\n", call_expr));
                    }
                    Some(rt) if is_int_ty(rt, &aliases) => {
                        out.push_str(&format!("        const __r = {};\n", call_expr));
                        out.push_str("        return (__r >= -9007199254740991n && __r <= 9007199254740991n) ? Number(__r) : __r;\n");
                    }
                    Some(_) => {
                        out.push_str(&format!("        return {};\n", call_expr));
                    }
                    None => {
                        out.push_str(&format!("        {};\n", call_expr));
                    }
                }
                if !cleanup_vars.is_empty() {
                    out.push_str("      } finally {\n");
                    for cv in cleanup_vars {
                        out.push_str(&format!("        freeHandle({});\n", cv));
                    }
                    out.push_str("      }\n");
                } else {
                    out.push_str("      } finally {}\n");
                }
                out.push_str("    },\n");
            }
        }
    }
    out.push_str("  };\n}\n");
    out
}

/// Emits idiomatic Swift C-ABI bindings (`.swift`) including `struct`s, `TenonError`, and `@_silgen_name`.
pub fn emit_swift_bindings(program: &Program) -> String {
    let aliases = collect_aliases(program);
    let mut out = String::from("// Auto-generated by Tenon (`--emit-swift`). Do not edit manually.\nimport Foundation\n\n");
    out.push_str("public enum TenonError: Error {\n    case contractViolation(String)\n}\n\n");
    for item in &program.items {
        if let Item::Record(r) = item {
            if r.is_pub && r.type_params.is_empty() {
                out.push_str(&format!("public struct {}: Codable, Equatable {{\n", r.name));
                for f in &r.fields {
                    out.push_str(&format!("    public let {}: {}\n", f.name, swift_type(&f.ty, &aliases)));
                }
                out.push_str("}\n\n");
            }
        }
    }
    for item in &program.items {
        if let Item::Fn(f) = item {
            if (f.is_pub || f.attrs.iter().any(|a| a.name == "export"))
                && !f.is_ghost
                && !f.is_test
                && f.type_params.is_empty()
            {
                let csym = export_symbol_name(f);
                let ps: Vec<String> = f
                    .params
                    .iter()
                    .map(|p| format!("_ {}: {}", p.name, swift_type(&p.ty, &aliases)))
                    .collect();
                let rt = f
                    .ret_ty
                    .as_ref()
                    .map(|t| format!(" -> {}", swift_type(t, &aliases)))
                    .unwrap_or_default();
                out.push_str(&format!("@_silgen_name(\"{}\")\n", csym));
                out.push_str(&format!("public func {}({}){}\n\n", f.name, ps.join(", "), rt));
            }
        }
    }
    out
}

/// Emits Kotlin native/JNI wrapper bindings (`.kt`) including `data class`es & `TenonNative`.
pub fn emit_kotlin_bindings(program: &Program, package_name: &str) -> String {
    let aliases = collect_aliases(program);
    let mut out = format!(
        "// Auto-generated by Tenon (`--emit-kotlin`). Do not edit manually.\npackage {}\n\n",
        package_name
    );
    for item in &program.items {
        if let Item::Record(r) = item {
            if r.is_pub && r.type_params.is_empty() && !r.fields.is_empty() {
                let fs: Vec<String> = r
                    .fields
                    .iter()
                    .map(|f| format!("val {}: {}", f.name, kotlin_type(&f.ty, &aliases)))
                    .collect();
                out.push_str(&format!("data class {}({})\n\n", r.name, fs.join(", ")));
            }
        }
    }
    out.push_str("object TenonNative {\n    init { System.loadLibrary(\"tenon_native\") }\n\n");
    for item in &program.items {
        if let Item::Fn(f) = item {
            if (f.is_pub || f.attrs.iter().any(|a| a.name == "export"))
                && !f.is_ghost
                && !f.is_test
                && f.type_params.is_empty()
            {
                let ps: Vec<String> = f
                    .params
                    .iter()
                    .map(|p| format!("{}: {}", p.name, kotlin_type(&p.ty, &aliases)))
                    .collect();
                let rt = f
                    .ret_ty
                    .as_ref()
                    .map(|t| format!(": {}", kotlin_type(t, &aliases)))
                    .unwrap_or_else(|| ": Unit".into());
                out.push_str(&format!(
                    "    @JvmStatic external fun {}({}){}\n",
                    export_symbol_name(f),
                    ps.join(", "),
                    rt
                ));
            }
        }
    }
    out.push_str("}\n");
    out
}

/// Emits a ready-to-compile Go `cgo` wrapper (`.go`) linking against a Tenon C-ABI static library & header.
pub fn emit_go_cgo_bindings(program: &Program, package_name: &str, header_name: &str, lib_name: &str) -> String {
    let aliases = collect_aliases(program);
    let mut out = format!(
        "// Auto-generated by Tenon (`--emit-go`). Do not edit manually.\npackage {}\n\n/*\n#cgo LDFLAGS: -L. -l{} -lm\n#include \"{}\"\n*/\nimport \"C\"\n\n",
        package_name, lib_name, header_name
    );
    for item in &program.items {
        if let Item::Fn(f) = item {
            if f.attrs.iter().any(|a| a.name == "export")
                && !f.is_ghost
                && !f.is_test
                && f.type_params.is_empty()
            {
                let csym = export_symbol_name(f);
                let go_name: String = {
                    let mut chars = f.name.chars();
                    match chars.next() {
                        Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
                        None => "Fn".into(),
                    }
                };
                let mut go_params = Vec::new();
                let mut c_args = Vec::new();
                for p in &f.params {
                    let (gt, ct) = go_cgo_type(&p.ty, &aliases);
                    go_params.push(format!("{} {}", p.name, gt));
                    c_args.push(format!("{}({})", ct, p.name));
                }
                if let Some(rt) = &f.ret_ty {
                    let (grt, _) = go_cgo_type(rt, &aliases);
                    out.push_str(&format!(
                        "func {}({}) {} {{\n\treturn {}(C.{}({}))\n}}\n\n",
                        go_name,
                        go_params.join(", "),
                        grt,
                        grt,
                        csym,
                        c_args.join(", ")
                    ));
                } else {
                    out.push_str(&format!(
                        "func {}({}) {{\n\tC.{}({})\n}}\n\n",
                        go_name,
                        go_params.join(", "),
                        csym,
                        c_args.join(", ")
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
    fn emits_valid_typescript_swift_and_kotlin_bindings() {
        let prog = tenon_syntax::parse(r#"
pub type Price = Int where self > 0

pub record Tick {
    price Price
    qty   Int
}

@export("tenon_calc_fee")
pub pure fn calc_fee(px Price, bps Int) Int {
    (px * bps) / 10000
}
"#)
        .unwrap();
        let ts = emit_typescript_bindings(&prog);
        let sw = emit_swift_bindings(&prog);
        let kt = emit_kotlin_bindings(&prog, "com.tenon.generated");
        let go = emit_go_cgo_bindings(&prog, "tenonbind", "quant.h", "quant");
        assert!(ts.contains("export interface Tick"));
        assert!(ts.contains("tenon_calc_fee(px: bigint, bps: bigint): bigint;"));
        assert!(sw.contains("public struct Tick: Codable, Equatable"));
        assert!(sw.contains("@_silgen_name(\"tenon_calc_fee\")"));
        assert!(sw.contains("public func calc_fee(_ px: Int64, _ bps: Int64) -> Int64"));
        assert!(kt.contains("data class Tick(val price: Long, val qty: Long)"));
        assert!(kt.contains("external fun tenon_calc_fee(px: Long, bps: Long): Long"));
        assert!(go.contains("func Calc_fee(px int64, bps int64) int64"));
        assert!(go.contains("C.tenon_calc_fee(C.int64_t(px), C.int64_t(bps))"));
        let mjs = emit_wasm_mjs_loader(&prog);
        assert!(mjs.contains("export async function loadTenonWasm("));
        assert!(mjs.contains("tenon_calc_fee(px, bps)"));
        assert!(mjs.contains("freeHandle"));
    }
}
