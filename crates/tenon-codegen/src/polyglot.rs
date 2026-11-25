//! Industrial Polyglot Native Bridge & Extension Generators (Exceeding PyO3 Ergonomics).
//!
//! From ONE single zero-boilerplate verified Tenon `.tn` file (`@export` functions + `pub record`s + `impl` methods),
//! Tenon automatically emits ready-to-compile native bridges with deterministic Perceus RC (`live_objects == 0`):
//! - **Python 3 (CPython C-API `.so` + `.pyi`)**: native `PyTypeObject` classes, properties, `impl` methods,
//!   UTF-8 `str`, `list[int]`/`list[float]`, and automatic GIL release (`Py_BEGIN_ALLOW_THREADS`)!
//! - **Node.js (Zero-`node-gyp` N-API v8 `.node` + `.d.ts`)**: native JS classes, functions, UTF-8 strings, finalizers!
//! - **Swift 6 (iOS/macOS `.swift` + `module.modulemap`)**: ARC `deinit` classes + `String` marshaling!
//! - **Java & Kotlin / Android (JNI `.c` + `.java` + `.kt`)**: real `JNIEXPORT` C glue + `jstring` UTF-8 marshaling!
//! - **C++17 (`.hpp`)**: move-only RAII classes + `std::string_view`/`std::string` marshaling!
//! - **Rust (`.rs`) & Zig (`.zig`)**: safe idiomatic wrappers with `Drop`/`defer` Perceus cleanup!

use std::collections::{BTreeMap, BTreeSet};
use tenon_syntax::*;

#[derive(Debug, Clone, PartialEq, Eq)]
enum BridgeTy {
    Int,
    Float,
    Float32,
    Bool,
    Text,
    ListInt,
    ListFloat,
    Record(String),
    Void,
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

fn is_user_source_file(program: &Program, file_id: u32) -> bool {
    if file_id == 0 {
        return true;
    }
    if program.file_packages.get(file_id as usize).copied() == Some(1) {
        return true;
    }
    program
        .source_files
        .get(file_id as usize)
        .is_some_and(|p| !p.contains("/std/") && !p.starts_with("std/"))
}

fn collect_export_records(program: &Program) -> Vec<&RecordDecl> {
    let mut out = Vec::new();
    for item in &program.items {
        if let Item::Record(r) = item {
            let exported = r.attrs.iter().any(|a| a.name == "export");
            let is_c = r.attrs.iter().any(|a| a.name == "repr");
            if (exported || (r.is_pub && is_user_source_file(program, r.span.file_id)))
                && !is_c
                && r.type_params.is_empty()
                && !r.fields.is_empty()
            {
                out.push(r);
            }
        }
    }
    out
}

fn collect_record_methods<'a>(program: &'a Program, rec_name: &str) -> Vec<&'a FnDecl> {
    let mut out = Vec::new();
    for item in &program.items {
        if let Item::Impl(imp) = item {
            if imp.interface.is_none()
                && matches!(&imp.target_ty, TypeExpr::Named { name, .. } if name == rec_name)
            {
                for m in &imp.methods {
                    if m.is_pub && !m.is_ghost && !m.is_test && m.type_params.is_empty() {
                        if let Some(first) = m.params.first() {
                            if first.name == "self" && !first.is_inout {
                                out.push(m);
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

fn collect_export_fns(program: &Program) -> Vec<(&FnDecl, String)> {
    let mut out = Vec::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            if f.attrs.iter().any(|a| a.name == "export")
                && !f.is_ghost
                && !f.is_test
                && f.type_params.is_empty()
            {
                let mut csym = f.name.clone();
                for a in &f.attrs {
                    if a.name == "export" {
                        if let Some(arg) = a.args.first() {
                            if let ExprKind::Str(s) = &arg.value.kind {
                                csym = s.clone();
                            }
                        }
                    }
                }
                out.push((f, csym));
            }
        }
    }
    out
}

fn resolve_bridge_ty(
    ty: &TypeExpr,
    aliases: &BTreeMap<String, TypeExpr>,
    rec_names: &BTreeSet<String>,
) -> BridgeTy {
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
    match cur {
        TypeExpr::Named { name, args, .. } => match name.as_str() {
            "Float" | "Float64" => BridgeTy::Float,
            "Float32" => BridgeTy::Float32,
            "Bool" => BridgeTy::Bool,
            "Text" => BridgeTy::Text,
            "List" if args.len() == 1 => match resolve_bridge_ty(&args[0], aliases, rec_names) {
                BridgeTy::Float => BridgeTy::ListFloat,
                _ => BridgeTy::ListInt,
            },
            "CVoid" | "Unit" => BridgeTy::Void,
            other if rec_names.contains(other) => BridgeTy::Record(other.to_string()),
            _ => BridgeTy::Int,
        },
        _ => BridgeTy::Int,
    }
}

fn c_abi_ty(bt: &BridgeTy) -> &'static str {
    match bt {
        BridgeTy::Float => "double",
        BridgeTy::Float32 => "float",
        BridgeTy::Bool => "bool",
        BridgeTy::Void => "void",
        _ => "int64_t",
    }
}

fn is_rc_bridge_ty(bt: &BridgeTy) -> bool {
    matches!(bt, BridgeTy::Text | BridgeTy::ListInt | BridgeTy::ListFloat | BridgeTy::Record(_))
}

fn rec_rc_mask(r: &RecordDecl, aliases: &BTreeMap<String, TypeExpr>, rec_names: &BTreeSet<String>) -> u64 {
    let mut mask = 0u64;
    for (i, f) in r.fields.iter().enumerate() {
        if i < 64 && is_rc_bridge_ty(&resolve_bridge_ty(&f.ty, aliases, rec_names)) {
            mask |= 1u64 << i;
        }
    }
    mask
}

fn sanitize_ident(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

// ============================================================================
// 1. PYTHON 3 NATIVE C-API EXTENSION (EXCEEDING PyO3 — ZERO BOILERPLATE!)
// ============================================================================

/// Generates a complete, self-contained CPython 3 C-API native extension source (`.c`)
/// exposing ALL `@export` functions, `pub record` classes (with `__init__`, properties, `__repr__`,
/// `tp_dealloc`, and `impl` methods!), UTF-8 strings, `list[int]`/`list[float]`, and automatic GIL release!
pub fn emit_python_c_ext(program: &Program, mod_name: &str) -> String {
    let aliases = collect_aliases(program);
    let records = collect_export_records(program);
    let rec_names: BTreeSet<String> = records.iter().map(|r| r.name.clone()).collect();
    let fns = collect_export_fns(program);
    let mident = sanitize_ident(mod_name);

    let mut out = String::new();
    out.push_str(&format!(
        "/* Auto-generated by Tenon Polyglot Bridge (`--export python` -> `{}.so`) */\n\
         #define PY_SSIZE_T_CLEAN\n\
         #include <Python.h>\n\
         #include <stdbool.h>\n\
         #include <stdint.h>\n\
         #include <string.h>\n\n\
         /* Tenon Perceus Runtime C ABI */\n\
         extern int64_t tenon_rc_alloc(int64_t tag, int64_t len);\n\
         extern void tenon_rc_set_rc_mask(int64_t ptr, int64_t mask);\n\
         extern int64_t tenon_rc_inc(int64_t ptr);\n\
         extern void tenon_rc_dec(int64_t ptr);\n\
         extern int64_t tenon_rc_get_field(int64_t ptr, int64_t idx);\n\
         extern void tenon_rc_set_field(int64_t ptr, int64_t idx, int64_t val);\n\
         extern int64_t tenon_rc_get_len(int64_t ptr);\n\
         extern int64_t tenon_text_new(int64_t data_ptr, int64_t byte_len);\n\
         extern const char* tenon_text_data_ptr(int64_t ptr);\n\
         extern int64_t tenon_text_byte_len(int64_t ptr);\n\
         extern int64_t tenon_list_filled(int64_t len, int64_t init_val, int64_t elem_is_rc);\n\
         extern void* tenon_obj_fields_ptr(int64_t ptr);\n\
         extern int64_t tenon_rt_live_objects(void);\n\n\
         static inline int64_t tn_f64_to_bits(double v) {{ int64_t b; memcpy(&b, &v, 8); return b; }}\n\
         static inline double tn_bits_to_f64(int64_t b) {{ double v; memcpy(&v, &b, 8); return v; }}\n\n\
         /* PEP 3118 Zero-Copy/Bulk Buffer Protocol Fast-Path (NumPy/PyTorch/array.array/memoryview -> memcpy!) */\n\
         static int64_t tn_py_to_list_int(PyObject* seq) {{\n\
             if (PyObject_CheckBuffer(seq)) {{\n\
                 Py_buffer view;\n\
                 if (PyObject_GetBuffer(seq, &view, PyBUF_CONTIG_RO | PyBUF_FORMAT) == 0) {{\n\
                     if (view.itemsize == 8 && view.format && (view.format[0] == 'q' || view.format[0] == 'l')) {{\n\
                         int64_t n = (int64_t)(view.len / 8);\n\
                         int64_t h = tenon_list_filled(n, 0, 0);\n\
                         if (n > 0) memcpy(tenon_obj_fields_ptr(h), view.buf, (size_t)view.len);\n\
                         PyBuffer_Release(&view);\n\
                         return h;\n\
                     }}\n\
                     PyBuffer_Release(&view);\n\
                 }} else {{\n\
                     PyErr_Clear();\n\
                 }}\n\
             }}\n\
             PyObject* fast = PySequence_Fast(seq, \"expected sequence or int64 buffer\");\n\
             if (!fast) return 0;\n\
             Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);\n\
             int64_t h = tenon_list_filled((int64_t)n, 0, 0);\n\
             int64_t* dst = (int64_t*)tenon_obj_fields_ptr(h);\n\
             PyObject** items = PySequence_Fast_ITEMS(fast);\n\
             for (Py_ssize_t i = 0; i < n; i++) {{\n\
                 dst[i] = (int64_t)PyLong_AsLongLong(items[i]);\n\
             }}\n\
             Py_DECREF(fast);\n\
             return h;\n\
         }}\n\
         static PyObject* tn_list_int_to_py(int64_t h) {{\n\
             int64_t n = tenon_rc_get_len(h);\n\
             PyObject* lst = PyList_New((Py_ssize_t)n);\n\
             if (!lst) {{ tenon_rc_dec(h); return NULL; }}\n\
             const int64_t* src = (const int64_t*)tenon_obj_fields_ptr(h);\n\
             for (int64_t i = 0; i < n; i++) {{\n\
                 PyList_SET_ITEM(lst, (Py_ssize_t)i, PyLong_FromLongLong(src[i]));\n\
             }}\n\
             tenon_rc_dec(h);\n\
             return lst;\n\
         }}\n\
         static int64_t tn_py_to_list_f64(PyObject* seq) {{\n\
             if (PyObject_CheckBuffer(seq)) {{\n\
                 Py_buffer view;\n\
                 if (PyObject_GetBuffer(seq, &view, PyBUF_CONTIG_RO | PyBUF_FORMAT) == 0) {{\n\
                     if (view.itemsize == 8 && view.format && view.format[0] == 'd') {{\n\
                         int64_t n = (int64_t)(view.len / 8);\n\
                         int64_t h = tenon_list_filled(n, 0, 0);\n\
                         if (n > 0) memcpy(tenon_obj_fields_ptr(h), view.buf, (size_t)view.len);\n\
                         PyBuffer_Release(&view);\n\
                         return h;\n\
                     }}\n\
                     PyBuffer_Release(&view);\n\
                 }} else {{\n\
                     PyErr_Clear();\n\
                 }}\n\
             }}\n\
             PyObject* fast = PySequence_Fast(seq, \"expected sequence or float64 buffer\");\n\
             if (!fast) return 0;\n\
             Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);\n\
             int64_t h = tenon_list_filled((int64_t)n, 0, 0);\n\
             double* dst = (double*)tenon_obj_fields_ptr(h);\n\
             PyObject** items = PySequence_Fast_ITEMS(fast);\n\
             for (Py_ssize_t i = 0; i < n; i++) {{\n\
                 dst[i] = PyFloat_AsDouble(items[i]);\n\
             }}\n\
             Py_DECREF(fast);\n\
             return h;\n\
         }}\n\
         static PyObject* tn_list_f64_to_py(int64_t h) {{\n\
             int64_t n = tenon_rc_get_len(h);\n\
             PyObject* lst = PyList_New((Py_ssize_t)n);\n\
             if (!lst) {{ tenon_rc_dec(h); return NULL; }}\n\
             const double* src = (const double*)tenon_obj_fields_ptr(h);\n\
             for (int64_t i = 0; i < n; i++) {{\n\
                 PyList_SET_ITEM(lst, (Py_ssize_t)i, PyFloat_FromDouble(src[i]));\n\
             }}\n\
             tenon_rc_dec(h);\n\
             return lst;\n\
         }}\n\n",
        mident
    ));

    // Forward declarations of exported C functions & impl methods
    for (f, csym) in &fns {
        let rt = f
            .ret_ty
            .as_ref()
            .map(|t| resolve_bridge_ty(t, &aliases, &rec_names))
            .unwrap_or(BridgeTy::Void);
        let ps: Vec<String> = if f.params.is_empty() {
            vec!["void".into()]
        } else {
            f.params
                .iter()
                .map(|p| c_abi_ty(&resolve_bridge_ty(&p.ty, &aliases, &rec_names)).to_string())
                .collect()
        };
        out.push_str(&format!("extern {} {}({});\n", c_abi_ty(&rt), csym, ps.join(", ")));
    }
    for r in &records {
        for m in collect_record_methods(program, &r.name) {
            let ps = vec!["int64_t"; m.params.len()].join(", ");
            out.push_str(&format!(
                "extern int64_t tenon_fn___m__{}__{}({});\n",
                sanitize_ident(&r.name),
                sanitize_ident(&m.name),
                ps
            ));
        }
    }
    out.push('\n');

    // Emit PyTypeObject structs & forward decls for every exported pub record
    for r in &records {
        let rn = sanitize_ident(&r.name);
        out.push_str(&format!(
            "typedef struct {{ PyObject_HEAD int64_t handle; }} PyTenon_{rn};\n\
             static PyTypeObject PyTenon_{rn}_Type;\n\
             static PyObject* tn_wrap_{rn}(int64_t h) {{\n\
                 PyTenon_{rn}* obj = PyObject_New(PyTenon_{rn}, &PyTenon_{rn}_Type);\n\
                 if (!obj) {{ tenon_rc_dec(h); return NULL; }}\n\
                 obj->handle = h;\n\
                 return (PyObject*)obj;\n\
             }}\n\n"
        ));
    }

    // Implement each record PyTypeObject (dealloc, __init__, properties, methods, __repr__)
    for r in &records {
        let rn = sanitize_ident(&r.name);
        let mask = rec_rc_mask(r, &aliases, &rec_names);
        out.push_str(&format!(
            "static void PyTenon_{rn}_dealloc(PyTenon_{rn}* self) {{\n\
                 if (self->handle) {{ tenon_rc_dec(self->handle); self->handle = 0; }}\n\
                 Py_TYPE(self)->tp_free((PyObject*)self);\n\
             }}\n\n"
        ));

        // Getters & Setters for every field
        for (fi, f) in r.fields.iter().enumerate() {
            let ft = resolve_bridge_ty(&f.ty, &aliases, &rec_names);
            let fnm = sanitize_ident(&f.name);
            out.push_str(&format!(
                "static PyObject* PyTenon_{rn}_get_{fnm}(PyTenon_{rn}* self, void* closure) {{\n\
                     (void)closure;\n\
                     int64_t raw = tenon_rc_get_field(self->handle, {fi});\n"
            ));
            match &ft {
                BridgeTy::Float | BridgeTy::Float32 => {
                    out.push_str("    return PyFloat_FromDouble(tn_bits_to_f64(raw));\n");
                }
                BridgeTy::Bool => {
                    out.push_str("    return PyBool_FromLong(raw != 0);\n");
                }
                BridgeTy::Text => {
                    out.push_str("    return PyUnicode_FromStringAndSize(tenon_text_data_ptr(raw), (Py_ssize_t)tenon_text_byte_len(raw));\n");
                }
                BridgeTy::ListInt => {
                    out.push_str("    tenon_rc_inc(raw); return tn_list_int_to_py(raw);\n");
                }
                BridgeTy::ListFloat => {
                    out.push_str("    tenon_rc_inc(raw); return tn_list_f64_to_py(raw);\n");
                }
                BridgeTy::Record(sub) => {
                    out.push_str(&format!("    tenon_rc_inc(raw); return tn_wrap_{}(raw);\n", sanitize_ident(sub)));
                }
                _ => {
                    out.push_str("    return PyLong_FromLongLong(raw);\n");
                }
            }
            out.push_str("}\n");

            out.push_str(&format!(
                "static int PyTenon_{rn}_set_{fnm}(PyTenon_{rn}* self, PyObject* val, void* closure) {{\n\
                     (void)closure;\n\
                     if (!val) {{ PyErr_SetString(PyExc_AttributeError, \"cannot delete field\"); return -1; }}\n"
            ));
            match &ft {
                BridgeTy::Float | BridgeTy::Float32 => {
                    out.push_str(&format!(
                        "    double dv = PyFloat_AsDouble(val);\n\
                             if (PyErr_Occurred()) return -1;\n\
                             tenon_rc_set_field(self->handle, {fi}, tn_f64_to_bits(dv));\n\
                             return 0;\n"
                    ));
                }
                BridgeTy::Bool => {
                    out.push_str(&format!(
                        "    int bv = PyObject_IsTrue(val);\n\
                             if (bv < 0) return -1;\n\
                             tenon_rc_set_field(self->handle, {fi}, bv ? 1 : 0);\n\
                             return 0;\n"
                    ));
                }
                BridgeTy::Text => {
                    out.push_str(&format!(
                        "    Py_ssize_t sz = 0;\n\
                             const char* s = PyUnicode_AsUTF8AndSize(val, &sz);\n\
                             if (!s) return -1;\n\
                             int64_t old = tenon_rc_get_field(self->handle, {fi});\n\
                             tenon_rc_set_field(self->handle, {fi}, tenon_text_new((int64_t)(uintptr_t)s, (int64_t)sz));\n\
                             if (old) tenon_rc_dec(old);\n\
                             return 0;\n"
                    ));
                }
                _ => {
                    out.push_str(&format!(
                        "    long long iv = PyLong_AsLongLong(val);\n\
                             if (PyErr_Occurred()) return -1;\n\
                             tenon_rc_set_field(self->handle, {fi}, (int64_t)iv);\n\
                             return 0;\n"
                    ));
                }
            }
            out.push_str("}\n\n");
        }

        // __init__ supporting positional AND keyword args!
        let kw_list: Vec<String> = r.fields.iter().map(|f| format!("\"{}\"", f.name)).collect();
        let mut fmt_str = String::new();
        for f in &r.fields {
            match resolve_bridge_ty(&f.ty, &aliases, &rec_names) {
                BridgeTy::Float | BridgeTy::Float32 => fmt_str.push('d'),
                BridgeTy::Bool => fmt_str.push('p'),
                BridgeTy::Text => fmt_str.push_str("s#"),
                BridgeTy::ListInt | BridgeTy::ListFloat | BridgeTy::Record(_) => fmt_str.push('O'),
                _ => fmt_str.push('L'),
            }
        }
        out.push_str(&format!(
            "static int PyTenon_{rn}_init(PyTenon_{rn}* self, PyObject* args, PyObject* kwds) {{\n\
                 static char* kwlist[] = {{ {}, NULL }};\n",
            kw_list.join(", ")
        ));
        let mut parse_args = Vec::new();
        for (fi, f) in r.fields.iter().enumerate() {
            match resolve_bridge_ty(&f.ty, &aliases, &rec_names) {
                BridgeTy::Float | BridgeTy::Float32 => {
                    out.push_str(&format!("    double v{fi} = 0.0;\n"));
                    parse_args.push(format!("&v{fi}"));
                }
                BridgeTy::Bool => {
                    out.push_str(&format!("    int v{fi} = 0;\n"));
                    parse_args.push(format!("&v{fi}"));
                }
                BridgeTy::Text => {
                    out.push_str(&format!("    const char* s{fi} = \"\"; Py_ssize_t sl{fi} = 0;\n"));
                    parse_args.push(format!("&s{fi}, &sl{fi}"));
                }
                BridgeTy::ListInt | BridgeTy::ListFloat | BridgeTy::Record(_) => {
                    out.push_str(&format!("    PyObject* o{fi} = NULL;\n"));
                    parse_args.push(format!("&o{fi}"));
                }
                _ => {
                    out.push_str(&format!("    long long v{fi} = 0;\n"));
                    parse_args.push(format!("&v{fi}"));
                }
            }
        }
        out.push_str(&format!(
            "    if (!PyArg_ParseTupleAndKeywords(args, kwds, \"{fmt_str}\", kwlist, {})) return -1;\n\
                 if (self->handle) tenon_rc_dec(self->handle);\n\
                 self->handle = tenon_rc_alloc(100, {});\n\
                 tenon_rc_set_rc_mask(self->handle, {}LL);\n",
            parse_args.join(", "),
            r.fields.len(),
            mask
        ));
        for (fi, f) in r.fields.iter().enumerate() {
            match resolve_bridge_ty(&f.ty, &aliases, &rec_names) {
                BridgeTy::Float | BridgeTy::Float32 => {
                    out.push_str(&format!("    tenon_rc_set_field(self->handle, {fi}, tn_f64_to_bits(v{fi}));\n"));
                }
                BridgeTy::Bool => {
                    out.push_str(&format!("    tenon_rc_set_field(self->handle, {fi}, v{fi} ? 1 : 0);\n"));
                }
                BridgeTy::Text => {
                    out.push_str(&format!("    tenon_rc_set_field(self->handle, {fi}, tenon_text_new((int64_t)(uintptr_t)s{fi}, (int64_t)sl{fi}));\n"));
                }
                BridgeTy::ListInt => {
                    out.push_str(&format!("    tenon_rc_set_field(self->handle, {fi}, tn_py_to_list_int(o{fi}));\n"));
                }
                BridgeTy::ListFloat => {
                    out.push_str(&format!("    tenon_rc_set_field(self->handle, {fi}, tn_py_to_list_f64(o{fi}));\n"));
                }
                BridgeTy::Record(sub) => {
                    let sr = sanitize_ident(&sub);
                    out.push_str(&format!(
                        "    if (!PyObject_TypeCheck(o{fi}, &PyTenon_{sr}_Type)) {{ PyErr_SetString(PyExc_TypeError, \"expected {sub}\"); return -1; }}\n\
                             tenon_rc_set_field(self->handle, {fi}, tenon_rc_inc(((PyTenon_{sr}*)o{fi})->handle));\n"
                    ));
                }
                _ => {
                    out.push_str(&format!("    tenon_rc_set_field(self->handle, {fi}, (int64_t)v{fi});\n"));
                }
            }
        }
        out.push_str("    return 0;\n}\n\n");

        // __repr__
        out.push_str(&format!(
            "static PyObject* PyTenon_{rn}_repr(PyTenon_{rn}* self) {{\n\
                 (void)self;\n\
                 return PyUnicode_FromFormat(\"<{rn} handle=%p>\", (void*)(uintptr_t)self->handle);\n\
             }}\n\n"
        ));

        // impl methods on this record — PEP 590 METH_FASTCALL (ZERO PyTuple allocation!)
        let methods = collect_record_methods(program, &r.name);
        for m in &methods {
            let mn = sanitize_ident(&m.name);
            let extra_params = &m.params[1..];
            let rt = m
                .ret_ty
                .as_ref()
                .map(|t| resolve_bridge_ty(t, &aliases, &rec_names))
                .unwrap_or(BridgeTy::Void);
            if extra_params.is_empty() {
                out.push_str(&format!(
                    "static PyObject* PyTenon_{rn}_m_{mn}(PyTenon_{rn}* self, PyObject* Py_UNUSED(ignored)) {{\n"
                ));
            } else {
                out.push_str(&format!(
                    "static PyObject* PyTenon_{rn}_m_{mn}(PyTenon_{rn}* self, PyObject* const* args, Py_ssize_t nargs) {{\n\
                         if (nargs != {}) {{ PyErr_Format(PyExc_TypeError, \"{}.{}() takes {} argument(s) (%zd given)\", nargs); return NULL; }}\n",
                    extra_params.len(),
                    r.name,
                    m.name,
                    extra_params.len()
                ));
            }
            let mut mcall = vec!["self->handle".to_string()];
            let mut mclean = Vec::new();
            for (pi, p) in extra_params.iter().enumerate() {
                let pt = resolve_bridge_ty(&p.ty, &aliases, &rec_names);
                match &pt {
                    BridgeTy::Float | BridgeTy::Float32 => {
                        out.push_str(&format!("    double a{pi} = PyFloat_AsDouble(args[{pi}]); if (a{pi} == -1.0 && PyErr_Occurred()) return NULL;\n"));
                        mcall.push(format!("tn_f64_to_bits(a{pi})"));
                    }
                    BridgeTy::Bool => {
                        out.push_str(&format!("    int a{pi} = PyObject_IsTrue(args[{pi}]); if (a{pi} < 0) return NULL;\n"));
                        mcall.push(format!("(int64_t)(a{pi} != 0 ? 1 : 0)"));
                    }
                    BridgeTy::Text => {
                        out.push_str(&format!("    Py_ssize_t sl{pi} = 0; const char* s{pi} = PyUnicode_AsUTF8AndSize(args[{pi}], &sl{pi}); if (!s{pi}) return NULL;\n"));
                        mcall.push(format!("h{pi}"));
                        mclean.push((pi, BridgeTy::Text));
                    }
                    BridgeTy::ListInt => {
                        mcall.push(format!("h{pi}"));
                        mclean.push((pi, BridgeTy::ListInt));
                    }
                    BridgeTy::ListFloat => {
                        mcall.push(format!("h{pi}"));
                        mclean.push((pi, BridgeTy::ListFloat));
                    }
                    BridgeTy::Record(sub) => {
                        let sr = sanitize_ident(sub);
                        out.push_str(&format!("    if (!PyObject_TypeCheck(args[{pi}], &PyTenon_{sr}_Type)) {{ PyErr_SetString(PyExc_TypeError, \"expected {sub}\"); return NULL; }}\n"));
                        mcall.push(format!("((PyTenon_{sr}*)args[{pi}])->handle"));
                    }
                    _ => {
                        out.push_str(&format!("    long long a{pi} = PyLong_AsLongLong(args[{pi}]); if (a{pi} == -1 && PyErr_Occurred()) return NULL;\n"));
                        mcall.push(format!("(int64_t)a{pi}"));
                    }
                }
            }
            for (pi, cty) in &mclean {
                match cty {
                    BridgeTy::Text => {
                        out.push_str(&format!("    int64_t h{pi} = tenon_text_new((int64_t)(uintptr_t)s{pi}, (int64_t)sl{pi});\n"));
                    }
                    BridgeTy::ListInt => {
                        out.push_str(&format!("    int64_t h{pi} = tn_py_to_list_int(args[{pi}]);\n    if (PyErr_Occurred()) {{ if (h{pi}) tenon_rc_dec(h{pi}); return NULL; }}\n"));
                    }
                    BridgeTy::ListFloat => {
                        out.push_str(&format!("    int64_t h{pi} = tn_py_to_list_f64(args[{pi}]);\n    if (PyErr_Occurred()) {{ if (h{pi}) tenon_rc_dec(h{pi}); return NULL; }}\n"));
                    }
                    _ => {}
                }
            }
            out.push_str(&format!(
                "    int64_t raw = 0;\n\
                     Py_BEGIN_ALLOW_THREADS\n\
                     raw = tenon_fn___m__{rn}__{mn}({});\n\
                     Py_END_ALLOW_THREADS\n",
                mcall.join(", ")
            ));
            for (pi, _) in &mclean {
                out.push_str(&format!("    tenon_rc_dec(h{pi});\n"));
            }
            emit_py_return(&mut out, &rt, true);
            out.push_str("}\n\n");
        }

        // GetSetDef & MethodDef tables
        out.push_str(&format!("static PyGetSetDef PyTenon_{rn}_getset[] = {{\n"));
        for f in &r.fields {
            let fnm = sanitize_ident(&f.name);
            out.push_str(&format!(
                "    {{\"{}\", (getter)PyTenon_{rn}_get_{fnm}, (setter)PyTenon_{rn}_set_{fnm}, \"Tenon field `{}`\", NULL}},\n",
                f.name, f.name
            ));
        }
        out.push_str("    {NULL, NULL, NULL, NULL, NULL}\n};\n\n");

        out.push_str(&format!("static PyMethodDef PyTenon_{rn}_methods[] = {{\n"));
        for m in &methods {
            let mn = sanitize_ident(&m.name);
            let flags = if m.params.len() <= 1 { "METH_NOARGS" } else { "METH_FASTCALL" };
            out.push_str(&format!(
                "    {{\"{}\", (PyCFunction)PyTenon_{rn}_m_{mn}, {flags}, \"Tenon verified method `{}.{}`\"}},\n",
                m.name, r.name, m.name
            ));
        }
        out.push_str("    {NULL, NULL, 0, NULL}\n};\n\n");

        out.push_str(&format!(
            "static PyTypeObject PyTenon_{rn}_Type = {{\n\
                 PyVarObject_HEAD_INIT(NULL, 0)\n\
                 .tp_name = \"{mident}.{}\",\n\
                 .tp_basicsize = sizeof(PyTenon_{rn}),\n\
                 .tp_dealloc = (destructor)PyTenon_{rn}_dealloc,\n\
                 .tp_repr = (reprfunc)PyTenon_{rn}_repr,\n\
                 .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,\n\
                 .tp_doc = \"Tenon verified native record `{}` (Perceus deterministic RC)\",\n\
                 .tp_methods = PyTenon_{rn}_methods,\n\
                 .tp_getset = PyTenon_{rn}_getset,\n\
                 .tp_init = (initproc)PyTenon_{rn}_init,\n\
                 .tp_new = PyType_GenericNew,\n\
             }};\n\n",
            r.name, r.name
        ));
    }

    // Module-level @export functions — PEP 590 METH_FASTCALL (ZERO PyTuple / PyArg_ParseTuple!)
    for (f, csym) in &fns {
        let fn_ident = sanitize_ident(&f.name);
        let rt = f
            .ret_ty
            .as_ref()
            .map(|t| resolve_bridge_ty(t, &aliases, &rec_names))
            .unwrap_or(BridgeTy::Void);
        if f.params.is_empty() {
            out.push_str(&format!("static PyObject* py_tn_{fn_ident}(PyObject* self, PyObject* Py_UNUSED(ignored)) {{\n    (void)self;\n"));
        } else {
            out.push_str(&format!(
                "static PyObject* py_tn_{fn_ident}(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {{\n\
                     (void)self;\n\
                     if (nargs != {}) {{ PyErr_Format(PyExc_TypeError, \"{}() takes {} argument(s) (%zd given)\", nargs); return NULL; }}\n",
                f.params.len(),
                f.name,
                f.params.len()
            ));
        }
        let mut call_args = Vec::new();
        let mut cleanups = Vec::new();
        for (pi, p) in f.params.iter().enumerate() {
            let pt = resolve_bridge_ty(&p.ty, &aliases, &rec_names);
            match &pt {
                BridgeTy::Float => {
                    out.push_str(&format!("    double a{pi} = PyFloat_AsDouble(args[{pi}]); if (a{pi} == -1.0 && PyErr_Occurred()) return NULL;\n"));
                    call_args.push(format!("a{pi}"));
                }
                BridgeTy::Float32 => {
                    out.push_str(&format!("    double ad{pi} = PyFloat_AsDouble(args[{pi}]); if (ad{pi} == -1.0 && PyErr_Occurred()) return NULL; float a{pi} = (float)ad{pi};\n"));
                    call_args.push(format!("a{pi}"));
                }
                BridgeTy::Bool => {
                    out.push_str(&format!("    int a{pi} = PyObject_IsTrue(args[{pi}]); if (a{pi} < 0) return NULL;\n"));
                    call_args.push(format!("(a{pi} != 0)"));
                }
                BridgeTy::Text => {
                    out.push_str(&format!("    Py_ssize_t sl{pi} = 0; const char* s{pi} = PyUnicode_AsUTF8AndSize(args[{pi}], &sl{pi}); if (!s{pi}) return NULL;\n"));
                    call_args.push(format!("h{pi}"));
                    cleanups.push((pi, BridgeTy::Text));
                }
                BridgeTy::ListInt => {
                    call_args.push(format!("h{pi}"));
                    cleanups.push((pi, BridgeTy::ListInt));
                }
                BridgeTy::ListFloat => {
                    call_args.push(format!("h{pi}"));
                    cleanups.push((pi, BridgeTy::ListFloat));
                }
                BridgeTy::Record(sub) => {
                    let sr = sanitize_ident(sub);
                    out.push_str(&format!("    if (!PyObject_TypeCheck(args[{pi}], &PyTenon_{sr}_Type)) {{ PyErr_SetString(PyExc_TypeError, \"expected {sub}\"); return NULL; }}\n"));
                    call_args.push(format!("((PyTenon_{sr}*)args[{pi}])->handle"));
                }
                _ => {
                    out.push_str(&format!("    long long a{pi} = PyLong_AsLongLong(args[{pi}]); if (a{pi} == -1 && PyErr_Occurred()) return NULL;\n"));
                    call_args.push(format!("(int64_t)a{pi}"));
                }
            }
        }
        for (pi, cty) in &cleanups {
            match cty {
                BridgeTy::Text => {
                    out.push_str(&format!("    int64_t h{pi} = tenon_text_new((int64_t)(uintptr_t)s{pi}, (int64_t)sl{pi});\n"));
                }
                BridgeTy::ListInt => {
                    out.push_str(&format!("    int64_t h{pi} = tn_py_to_list_int(args[{pi}]);\n    if (PyErr_Occurred()) {{ if (h{pi}) tenon_rc_dec(h{pi}); return NULL; }}\n"));
                }
                BridgeTy::ListFloat => {
                    out.push_str(&format!("    int64_t h{pi} = tn_py_to_list_f64(args[{pi}]);\n    if (PyErr_Occurred()) {{ if (h{pi}) tenon_rc_dec(h{pi}); return NULL; }}\n"));
                }
                _ => {}
            }
        }
        match &rt {
            BridgeTy::Void => {
                out.push_str(&format!(
                    "    Py_BEGIN_ALLOW_THREADS\n    {csym}({});\n    Py_END_ALLOW_THREADS\n",
                    call_args.join(", ")
                ));
            }
            _ => {
                out.push_str(&format!(
                    "    {} res;\n    Py_BEGIN_ALLOW_THREADS\n    res = {csym}({});\n    Py_END_ALLOW_THREADS\n",
                    c_abi_ty(&rt),
                    call_args.join(", ")
                ));
            }
        }
        for (pi, _) in &cleanups {
            out.push_str(&format!("    tenon_rc_dec(h{pi});\n"));
        }
        emit_py_return(&mut out, &rt, false);
        out.push_str("}\n\n");
    }

    // Built-in `live_objects()` leak-audit function
    out.push_str(
        "static PyObject* py_tn_live_objects(PyObject* self, PyObject* args) {\n\
             (void)self; (void)args;\n\
             return PyLong_FromLongLong(tenon_rt_live_objects());\n\
         }\n\n",
    );

    out.push_str("static PyMethodDef TenonModuleMethods[] = {\n");
    for (f, _) in &fns {
        let fn_ident = sanitize_ident(&f.name);
        let flags = if f.params.is_empty() { "METH_NOARGS" } else { "METH_FASTCALL" };
        out.push_str(&format!(
            "    {{\"{}\", (PyCFunction)py_tn_{fn_ident}, {flags}, \"Tenon verified native function `{}`\"}},\n",
            f.name, f.name
        ));
    }
    out.push_str("    {\"live_objects\", py_tn_live_objects, METH_NOARGS, \"Return live Perceus heap object count (0 == zero leaks)\"},\n");
    out.push_str("    {NULL, NULL, 0, NULL}\n};\n\n");

    out.push_str(&format!(
        "static struct PyModuleDef tenon_module_def = {{\n\
             PyModuleDef_HEAD_INIT, \"{mident}\", \"Tenon Verified Native Python Extension\", -1, TenonModuleMethods\n\
         }};\n\n\
         PyMODINIT_FUNC PyInit_{mident}(void) {{\n"
    ));
    for r in &records {
        let rn = sanitize_ident(&r.name);
        out.push_str(&format!("    if (PyType_Ready(&PyTenon_{rn}_Type) < 0) return NULL;\n"));
    }
    out.push_str("    PyObject* m = PyModule_Create(&tenon_module_def);\n    if (!m) return NULL;\n");
    for r in &records {
        let rn = sanitize_ident(&r.name);
        out.push_str(&format!(
            "    Py_INCREF(&PyTenon_{rn}_Type);\n\
                 if (PyModule_AddObject(m, \"{}\", (PyObject*)&PyTenon_{rn}_Type) < 0) {{ Py_DECREF(&PyTenon_{rn}_Type); Py_DECREF(m); return NULL; }}\n",
            r.name
        ));
    }
    out.push_str("    return m;\n}\n");
    out
}

fn emit_py_return(out: &mut String, rt: &BridgeTy, raw_bits: bool) {
    let var = if raw_bits { "raw" } else { "res" };
    match rt {
        BridgeTy::Void => out.push_str("    Py_RETURN_NONE;\n"),
        BridgeTy::Float => {
            if raw_bits {
                out.push_str("    return PyFloat_FromDouble(tn_bits_to_f64(raw));\n");
            } else {
                out.push_str("    return PyFloat_FromDouble(res);\n");
            }
        }
        BridgeTy::Float32 => {
            if raw_bits {
                out.push_str("    return PyFloat_FromDouble(tn_bits_to_f64(raw));\n");
            } else {
                out.push_str("    return PyFloat_FromDouble((double)res);\n");
            }
        }
        BridgeTy::Bool => out.push_str(&format!("    return PyBool_FromLong({var} != 0);\n")),
        BridgeTy::Text => out.push_str(&format!(
            "    PyObject* s = PyUnicode_FromStringAndSize(tenon_text_data_ptr({var}), (Py_ssize_t)tenon_text_byte_len({var}));\n\
                 tenon_rc_dec({var});\n\
                 return s;\n"
        )),
        BridgeTy::ListInt => out.push_str(&format!("    return tn_list_int_to_py({var});\n")),
        BridgeTy::ListFloat => out.push_str(&format!("    return tn_list_f64_to_py({var});\n")),
        BridgeTy::Record(sub) => {
            out.push_str(&format!("    return tn_wrap_{}({var});\n", sanitize_ident(sub)));
        }
        BridgeTy::Int => out.push_str(&format!("    return PyLong_FromLongLong({var});\n")),
    }
}

fn py_stub_ty(bt: &BridgeTy) -> String {
    match bt {
        BridgeTy::Int => "int".into(),
        BridgeTy::Float | BridgeTy::Float32 => "float".into(),
        BridgeTy::Bool => "bool".into(),
        BridgeTy::Text => "str".into(),
        BridgeTy::ListInt => "list[int]".into(),
        BridgeTy::ListFloat => "list[float]".into(),
        BridgeTy::Record(r) => r.clone(),
        BridgeTy::Void => "None".into(),
    }
}

/// Generates PEP-484/561 `.pyi` typing stubs for IDE autocomplete & mypy/pyright verification.
pub fn emit_python_pyi(program: &Program, mod_name: &str) -> String {
    let aliases = collect_aliases(program);
    let records = collect_export_records(program);
    let rec_names: BTreeSet<String> = records.iter().map(|r| r.name.clone()).collect();
    let fns = collect_export_fns(program);

    let mut out = format!("# Auto-generated PEP-484 typing stub by Tenon (`--export python` -> `{}`)\n\n", mod_name);
    for r in &records {
        out.push_str(&format!("class {}:\n", r.name));
        for f in &r.fields {
            let pt = py_stub_ty(&resolve_bridge_ty(&f.ty, &aliases, &rec_names));
            out.push_str(&format!("    {}: {}\n", f.name, pt));
        }
        let init_args: Vec<String> = r
            .fields
            .iter()
            .map(|f| format!("{}: {}", f.name, py_stub_ty(&resolve_bridge_ty(&f.ty, &aliases, &rec_names))))
            .collect();
        out.push_str(&format!("    def __init__(self, {}) -> None: ...\n", init_args.join(", ")));
        for m in collect_record_methods(program, &r.name) {
            let ps: Vec<String> = m.params[1..]
                .iter()
                .map(|p| format!("{}: {}", p.name, py_stub_ty(&resolve_bridge_ty(&p.ty, &aliases, &rec_names))))
                .collect();
            let rt = m
                .ret_ty
                .as_ref()
                .map(|t| py_stub_ty(&resolve_bridge_ty(t, &aliases, &rec_names)))
                .unwrap_or_else(|| "None".into());
            let self_args = if ps.is_empty() {
                "self".to_string()
            } else {
                format!("self, {}", ps.join(", "))
            };
            out.push_str(&format!("    def {}({}) -> {}: ...\n", m.name, self_args, rt));
        }
        out.push('\n');
    }
    for (f, _) in &fns {
        let ps: Vec<String> = f
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, py_stub_ty(&resolve_bridge_ty(&p.ty, &aliases, &rec_names))))
            .collect();
        let rt = f
            .ret_ty
            .as_ref()
            .map(|t| py_stub_ty(&resolve_bridge_ty(t, &aliases, &rec_names)))
            .unwrap_or_else(|| "None".into());
        out.push_str(&format!("def {}({}) -> {}: ...\n", f.name, ps.join(", "), rt));
    }
    out.push_str("def live_objects() -> int: ...\n");
    out
}

// ============================================================================
// 2. NODE.JS ZERO-`node-gyp` NATIVE N-API v8 ADDON (`.c` -> `.node` + `.d.ts`)
// ============================================================================

/// Generates a zero-dependency Node-API v8 native C addon (`NAPI_MODULE`) that compiles directly
/// to `<mod>.node` via `clang -shared` with ZERO `node-gyp` and ZERO external Node headers!
pub fn emit_napi_c_addon(program: &Program, mod_name: &str) -> String {
    let aliases = collect_aliases(program);
    let records = collect_export_records(program);
    let rec_names: BTreeSet<String> = records.iter().map(|r| r.name.clone()).collect();
    let fns = collect_export_fns(program);
    let mident = sanitize_ident(mod_name);

    let mut out = String::new();
    out.push_str(&format!(
        "/* Auto-generated by Tenon Polyglot Bridge (`--export node` -> `{mident}.node`) */\n\
         /* Zero-`node-gyp`: uses stable Node-API v8 ABI directly! */\n\
         #include <stdbool.h>\n\
         #include <stddef.h>\n\
         #include <stdint.h>\n\
         #include <stdlib.h>\n\
         #include <string.h>\n\n\
         typedef struct napi_env__* napi_env;\n\
         typedef struct napi_value__* napi_value;\n\
         typedef struct napi_callback_info__* napi_callback_info;\n\
         typedef struct napi_ref__* napi_ref;\n\
         typedef int napi_status;\n\
         typedef napi_value (*napi_callback)(napi_env env, napi_callback_info info);\n\
         typedef void (*napi_finalize)(napi_env env, void* finalize_data, void* finalize_hint);\n\
         typedef struct {{\n\
             const char* utf8name;\n\
             napi_value name;\n\
             napi_callback method;\n\
             napi_callback getter;\n\
             napi_callback setter;\n\
             napi_value value;\n\
             int attributes;\n\
             void* data;\n\
         }} napi_property_descriptor;\n\
         typedef struct {{\n\
             int nm_version;\n\
             unsigned int nm_flags;\n\
             const char* nm_filename;\n\
             napi_value (*nm_register_func)(napi_env env, napi_value exports);\n\
             const char* nm_modname;\n\
             void* nm_priv;\n\
             void* reserved[4];\n\
         }} napi_module;\n\n\
         extern void napi_module_register(napi_module* mod);\n\
         extern napi_status napi_get_cb_info(napi_env, napi_callback_info, size_t*, napi_value*, napi_value*, void**);\n\
         extern napi_status napi_create_function(napi_env, const char*, size_t, napi_callback, void*, napi_value*);\n\
         extern napi_status napi_set_named_property(napi_env, napi_value, const char*, napi_value);\n\
         extern napi_status napi_create_int64(napi_env, int64_t, napi_value*);\n\
         extern napi_status napi_get_value_int64(napi_env, napi_value, int64_t*);\n\
         extern napi_status napi_create_double(napi_env, double, napi_value*);\n\
         extern napi_status napi_get_value_double(napi_env, napi_value, double*);\n\
         extern napi_status napi_get_boolean(napi_env, bool, napi_value*);\n\
         extern napi_status napi_get_value_bool(napi_env, napi_value, bool*);\n\
         extern napi_status napi_get_undefined(napi_env, napi_value*);\n\
         extern napi_status napi_create_string_utf8(napi_env, const char*, size_t, napi_value*);\n\
         extern napi_status napi_get_value_string_utf8(napi_env, napi_value, char*, size_t, size_t*);\n\
         extern napi_status napi_define_class(napi_env, const char*, size_t, napi_callback, void*, size_t, const napi_property_descriptor*, napi_value*);\n\
         extern napi_status napi_wrap(napi_env, napi_value, void*, napi_finalize, void*, napi_ref*);\n\
         extern napi_status napi_unwrap(napi_env, napi_value, void**);\n\
         extern napi_status napi_is_typedarray(napi_env, napi_value, bool*);\n\
         extern napi_status napi_get_typedarray_info(napi_env, napi_value, int*, size_t*, void**, napi_value*, size_t*);\n\
         extern napi_status napi_create_arraybuffer(napi_env, size_t, void**, napi_value*);\n\
         extern napi_status napi_create_typedarray(napi_env, int, size_t, napi_value, size_t, napi_value*);\n\
         extern napi_status napi_get_array_length(napi_env, napi_value, uint32_t*);\n\
         extern napi_status napi_get_element(napi_env, napi_value, uint32_t, napi_value*);\n\
         extern napi_status napi_create_array_with_length(napi_env, size_t, napi_value*);\n\
         extern napi_status napi_set_element(napi_env, napi_value, uint32_t, napi_value);\n\n\
         extern int64_t tenon_rc_alloc(int64_t tag, int64_t len);\n\
         extern void tenon_rc_set_rc_mask(int64_t ptr, int64_t mask);\n\
         extern void tenon_rc_dec(int64_t ptr);\n\
         extern int64_t tenon_rc_get_field(int64_t ptr, int64_t idx);\n\
         extern void tenon_rc_set_field(int64_t ptr, int64_t idx, int64_t val);\n\
         extern int64_t tenon_rc_get_len(int64_t ptr);\n\
         extern int64_t tenon_list_filled(int64_t len, int64_t init_val, int64_t elem_is_rc);\n\
         extern void* tenon_obj_fields_ptr(int64_t ptr);\n\
         extern int64_t tenon_text_new(int64_t data_ptr, int64_t byte_len);\n\
         extern const char* tenon_text_data_ptr(int64_t ptr);\n\
         extern int64_t tenon_text_byte_len(int64_t ptr);\n\
         extern int64_t tenon_rt_live_objects(void);\n\n\
         static inline int64_t tn_f64_to_bits(double v) {{ int64_t b; memcpy(&b, &v, 8); return b; }}\n\
         static inline double tn_bits_to_f64(int64_t b) {{ double v; memcpy(&v, &b, 8); return v; }}\n\
         static int64_t tn_napi_to_list_f64(napi_env env, napi_value v) {{\n\
             bool is_ta = false;\n\
             napi_is_typedarray(env, v, &is_ta);\n\
             if (is_ta) {{\n\
                 int ty = 0; size_t len = 0, off = 0; void* data = NULL; napi_value ab = NULL;\n\
                 if (napi_get_typedarray_info(env, v, &ty, &len, &data, &ab, &off) == 0 && ty == 8) {{\n\
                     int64_t h = tenon_list_filled((int64_t)len, 0, 0);\n\
                     if (len > 0) memcpy(tenon_obj_fields_ptr(h), data, len * sizeof(double));\n\
                     return h;\n\
                 }}\n\
             }}\n\
             uint32_t len = 0; napi_get_array_length(env, v, &len);\n\
             int64_t h = tenon_list_filled((int64_t)len, 0, 0);\n\
             double* dst = (double*)tenon_obj_fields_ptr(h);\n\
             for (uint32_t i = 0; i < len; i++) {{\n\
                 napi_value el = NULL; double dv = 0.0;\n\
                 napi_get_element(env, v, i, &el); napi_get_value_double(env, el, &dv);\n\
                 dst[i] = dv;\n\
             }}\n\
             return h;\n\
         }}\n\
         static int64_t tn_napi_to_list_int(napi_env env, napi_value v) {{\n\
             bool is_ta = false;\n\
             napi_is_typedarray(env, v, &is_ta);\n\
             if (is_ta) {{\n\
                 int ty = 0; size_t len = 0, off = 0; void* data = NULL; napi_value ab = NULL;\n\
                 if (napi_get_typedarray_info(env, v, &ty, &len, &data, &ab, &off) == 0 && ty == 9) {{\n\
                     int64_t h = tenon_list_filled((int64_t)len, 0, 0);\n\
                     if (len > 0) memcpy(tenon_obj_fields_ptr(h), data, len * sizeof(int64_t));\n\
                     return h;\n\
                 }}\n\
             }}\n\
             uint32_t len = 0; napi_get_array_length(env, v, &len);\n\
             int64_t h = tenon_list_filled((int64_t)len, 0, 0);\n\
             int64_t* dst = (int64_t*)tenon_obj_fields_ptr(h);\n\
             for (uint32_t i = 0; i < len; i++) {{\n\
                 napi_value el = NULL; int64_t iv = 0;\n\
                 napi_get_element(env, v, i, &el); napi_get_value_int64(env, el, &iv);\n\
                 dst[i] = iv;\n\
             }}\n\
             return h;\n\
         }}\n\
         static napi_value tn_napi_from_list_f64(napi_env env, int64_t h) {{\n\
             size_t len = (size_t)tenon_rc_get_len(h);\n\
             void* ab_data = NULL; napi_value ab = NULL; napi_value ta = NULL;\n\
             napi_create_arraybuffer(env, len * sizeof(double), &ab_data, &ab);\n\
             if (len > 0) memcpy(ab_data, tenon_obj_fields_ptr(h), len * sizeof(double));\n\
             tenon_rc_dec(h);\n\
             napi_create_typedarray(env, 8, len, ab, 0, &ta);\n\
             return ta;\n\
         }}\n\
         static napi_value tn_napi_from_list_int(napi_env env, int64_t h) {{\n\
             size_t len = (size_t)tenon_rc_get_len(h);\n\
             napi_value arr = NULL; napi_create_array_with_length(env, len, &arr);\n\
             const int64_t* src = (const int64_t*)tenon_obj_fields_ptr(h);\n\
             for (size_t i = 0; i < len; i++) {{\n\
                 napi_value el = NULL; napi_create_int64(env, src[i], &el);\n\
                 napi_set_element(env, arr, (uint32_t)i, el);\n\
             }}\n\
             tenon_rc_dec(h);\n\
             return arr;\n\
         }}\n\
         static int64_t tn_napi_to_text(napi_env env, napi_value v) {{\n\
             size_t len = 0;\n\
             napi_get_value_string_utf8(env, v, NULL, 0, &len);\n\
             char* buf = (char*)malloc(len + 1);\n\
             napi_get_value_string_utf8(env, v, buf, len + 1, &len);\n\
             int64_t h = tenon_text_new((int64_t)(uintptr_t)buf, (int64_t)len);\n\
             free(buf);\n\
             return h;\n\
         }}\n\
         static void tn_napi_finalize_obj(napi_env env, void* data, void* hint) {{\n\
             (void)env; (void)hint;\n\
             if (data) tenon_rc_dec((int64_t)(uintptr_t)data);\n\
         }}\n\n"
    ));

    for (f, csym) in &fns {
        let rt = f
            .ret_ty
            .as_ref()
            .map(|t| resolve_bridge_ty(t, &aliases, &rec_names))
            .unwrap_or(BridgeTy::Void);
        let ps: Vec<String> = if f.params.is_empty() {
            vec!["void".into()]
        } else {
            f.params
                .iter()
                .map(|p| c_abi_ty(&resolve_bridge_ty(&p.ty, &aliases, &rec_names)).to_string())
                .collect()
        };
        out.push_str(&format!("extern {} {}({});\n", c_abi_ty(&rt), csym, ps.join(", ")));
    }
    for r in &records {
        for m in collect_record_methods(program, &r.name) {
            let ps = vec!["int64_t"; m.params.len()].join(", ");
            out.push_str(&format!(
                "extern int64_t tenon_fn___m__{}__{}({});\n",
                sanitize_ident(&r.name),
                sanitize_ident(&m.name),
                ps
            ));
        }
    }
    out.push('\n');

    // Emit N-API classes for exported records
    for r in &records {
        let rn = sanitize_ident(&r.name);
        let mask = rec_rc_mask(r, &aliases, &rec_names);
        let nf = r.fields.len();
        out.push_str(&format!(
            "static napi_value napi_ctor_{rn}(napi_env env, napi_callback_info info) {{\n\
                 size_t argc = {nf}; napi_value argv[{max_a}]; napi_value jsthis = NULL;\n\
                 napi_get_cb_info(env, info, &argc, argv, &jsthis, NULL);\n\
                 int64_t h = tenon_rc_alloc(100, {nf});\n\
                 tenon_rc_set_rc_mask(h, {mask}LL);\n",
            max_a = nf.max(1)
        ));
        for (fi, f) in r.fields.iter().enumerate() {
            match resolve_bridge_ty(&f.ty, &aliases, &rec_names) {
                BridgeTy::Float | BridgeTy::Float32 => {
                    out.push_str(&format!("    {{ double v = 0; if (argc > {fi}) napi_get_value_double(env, argv[{fi}], &v); tenon_rc_set_field(h, {fi}, tn_f64_to_bits(v)); }}\n"));
                }
                BridgeTy::Bool => {
                    out.push_str(&format!("    {{ bool v = false; if (argc > {fi}) napi_get_value_bool(env, argv[{fi}], &v); tenon_rc_set_field(h, {fi}, v ? 1 : 0); }}\n"));
                }
                BridgeTy::Text => {
                    out.push_str(&format!("    if (argc > {fi}) tenon_rc_set_field(h, {fi}, tn_napi_to_text(env, argv[{fi}]));\n"));
                }
                _ => {
                    out.push_str(&format!("    {{ int64_t v = 0; if (argc > {fi}) napi_get_value_int64(env, argv[{fi}], &v); tenon_rc_set_field(h, {fi}, v); }}\n"));
                }
            }
        }
        out.push_str("    napi_wrap(env, jsthis, (void*)(uintptr_t)h, tn_napi_finalize_obj, NULL, NULL);\n    return jsthis;\n}\n\n");

        // Property getters
        for (fi, f) in r.fields.iter().enumerate() {
            let fnm = sanitize_ident(&f.name);
            out.push_str(&format!(
                "static napi_value napi_get_{rn}_{fnm}(napi_env env, napi_callback_info info) {{\n\
                     napi_value jsthis = NULL; napi_get_cb_info(env, info, NULL, NULL, &jsthis, NULL);\n\
                     void* rawp = NULL; napi_unwrap(env, jsthis, &rawp);\n\
                     int64_t val = tenon_rc_get_field((int64_t)(uintptr_t)rawp, {fi});\n\
                     napi_value out = NULL;\n"
            ));
            match resolve_bridge_ty(&f.ty, &aliases, &rec_names) {
                BridgeTy::Float | BridgeTy::Float32 => out.push_str("    napi_create_double(env, tn_bits_to_f64(val), &out);\n"),
                BridgeTy::Bool => out.push_str("    napi_get_boolean(env, val != 0, &out);\n"),
                BridgeTy::Text => out.push_str("    napi_create_string_utf8(env, tenon_text_data_ptr(val), (size_t)tenon_text_byte_len(val), &out);\n"),
                _ => out.push_str("    napi_create_int64(env, val, &out);\n"),
            }
            out.push_str("    return out;\n}\n");
        }

        // Methods
        for m in collect_record_methods(program, &r.name) {
            let mn = sanitize_ident(&m.name);
            let extra = &m.params[1..];
            let rt = m
                .ret_ty
                .as_ref()
                .map(|t| resolve_bridge_ty(t, &aliases, &rec_names))
                .unwrap_or(BridgeTy::Void);
            out.push_str(&format!(
                "static napi_value napi_m_{rn}_{mn}(napi_env env, napi_callback_info info) {{\n\
                     size_t argc = {}; napi_value argv[{}]; napi_value jsthis = NULL;\n\
                     napi_get_cb_info(env, info, &argc, argv, &jsthis, NULL);\n\
                     void* rawp = NULL; napi_unwrap(env, jsthis, &rawp);\n",
                extra.len(),
                extra.len().max(1)
            ));
            let mut mcall = vec!["(int64_t)(uintptr_t)rawp".to_string()];
            let mut mclean = Vec::new();
            for (pi, p) in extra.iter().enumerate() {
                match resolve_bridge_ty(&p.ty, &aliases, &rec_names) {
                    BridgeTy::Float | BridgeTy::Float32 => {
                        out.push_str(&format!("    double a{pi} = 0; if (argc > {pi}) napi_get_value_double(env, argv[{pi}], &a{pi});\n"));
                        mcall.push(format!("tn_f64_to_bits(a{pi})"));
                    }
                    BridgeTy::Bool => {
                        out.push_str(&format!("    bool a{pi} = false; if (argc > {pi}) napi_get_value_bool(env, argv[{pi}], &a{pi});\n"));
                        mcall.push(format!("(int64_t)(a{pi} ? 1 : 0)"));
                    }
                    BridgeTy::Text => {
                        out.push_str(&format!("    int64_t h{pi} = (argc > {pi}) ? tn_napi_to_text(env, argv[{pi}]) : tenon_text_new(0, 0);\n"));
                        mcall.push(format!("h{pi}"));
                        mclean.push(pi);
                    }
                    _ => {
                        out.push_str(&format!("    int64_t a{pi} = 0; if (argc > {pi}) napi_get_value_int64(env, argv[{pi}], &a{pi});\n"));
                        mcall.push(format!("a{pi}"));
                    }
                }
            }
            out.push_str(&format!("    int64_t raw = tenon_fn___m__{rn}__{mn}({});\n", mcall.join(", ")));
            for pi in mclean {
                out.push_str(&format!("    tenon_rc_dec(h{pi});\n"));
            }
            emit_napi_return(&mut out, &rt, true);
            out.push_str("}\n");
        }
        out.push('\n');
    }

    // Module-level @export functions
    for (f, csym) in &fns {
        let fn_ident = sanitize_ident(&f.name);
        let rt = f
            .ret_ty
            .as_ref()
            .map(|t| resolve_bridge_ty(t, &aliases, &rec_names))
            .unwrap_or(BridgeTy::Void);
        let np = f.params.len();
        out.push_str(&format!(
            "static napi_value napi_fn_{fn_ident}(napi_env env, napi_callback_info info) {{\n\
                 size_t argc = {np}; napi_value argv[{max_a}];\n\
                 napi_get_cb_info(env, info, &argc, argv, NULL, NULL);\n",
            max_a = np.max(1)
        ));
        let mut call_args = Vec::new();
        let mut cleanups = Vec::new();
        for (pi, p) in f.params.iter().enumerate() {
            match resolve_bridge_ty(&p.ty, &aliases, &rec_names) {
                BridgeTy::Float => {
                    out.push_str(&format!("    double a{pi} = 0; if (argc > {pi}) napi_get_value_double(env, argv[{pi}], &a{pi});\n"));
                    call_args.push(format!("a{pi}"));
                }
                BridgeTy::Float32 => {
                    out.push_str(&format!("    double ad{pi} = 0; if (argc > {pi}) napi_get_value_double(env, argv[{pi}], &ad{pi}); float a{pi} = (float)ad{pi};\n"));
                    call_args.push(format!("a{pi}"));
                }
                BridgeTy::Bool => {
                    out.push_str(&format!("    bool a{pi} = false; if (argc > {pi}) napi_get_value_bool(env, argv[{pi}], &a{pi});\n"));
                    call_args.push(format!("a{pi}"));
                }
                BridgeTy::Text => {
                    out.push_str(&format!("    int64_t h{pi} = (argc > {pi}) ? tn_napi_to_text(env, argv[{pi}]) : tenon_text_new(0, 0);\n"));
                    call_args.push(format!("h{pi}"));
                    cleanups.push(pi);
                }
                BridgeTy::ListFloat => {
                    out.push_str(&format!("    int64_t h{pi} = (argc > {pi}) ? tn_napi_to_list_f64(env, argv[{pi}]) : tenon_list_filled(0, 0, 0);\n"));
                    call_args.push(format!("h{pi}"));
                    cleanups.push(pi);
                }
                BridgeTy::ListInt => {
                    out.push_str(&format!("    int64_t h{pi} = (argc > {pi}) ? tn_napi_to_list_int(env, argv[{pi}]) : tenon_list_filled(0, 0, 0);\n"));
                    call_args.push(format!("h{pi}"));
                    cleanups.push(pi);
                }
                BridgeTy::Record(_) => {
                    out.push_str(&format!("    void* rp{pi} = NULL; if (argc > {pi}) napi_unwrap(env, argv[{pi}], &rp{pi});\n"));
                    call_args.push(format!("(int64_t)(uintptr_t)rp{pi}"));
                }
                _ => {
                    out.push_str(&format!("    int64_t a{pi} = 0; if (argc > {pi}) napi_get_value_int64(env, argv[{pi}], &a{pi});\n"));
                    call_args.push(format!("a{pi}"));
                }
            }
        }
        match &rt {
            BridgeTy::Void => out.push_str(&format!("    {csym}({});\n", call_args.join(", "))),
            _ => out.push_str(&format!("    {} res = {csym}({});\n", c_abi_ty(&rt), call_args.join(", "))),
        }
        for pi in cleanups {
            out.push_str(&format!("    tenon_rc_dec(h{pi});\n"));
        }
        emit_napi_return(&mut out, &rt, false);
        out.push_str("}\n\n");
    }

    out.push_str(
        "static napi_value napi_fn_live_objects(napi_env env, napi_callback_info info) {\n\
             (void)info;\n\
             napi_value out = NULL;\n\
             napi_create_int64(env, tenon_rt_live_objects(), &out);\n\
             return out;\n\
         }\n\n\
         static napi_value tenon_napi_init(napi_env env, napi_value exports) {\n\
             napi_value fn = NULL;\n",
    );
    for (f, _) in &fns {
        let fn_ident = sanitize_ident(&f.name);
        out.push_str(&format!(
            "    napi_create_function(env, \"{}\", (size_t)-1, napi_fn_{fn_ident}, NULL, &fn);\n\
                 napi_set_named_property(env, exports, \"{}\", fn);\n",
            f.name, f.name
        ));
    }
    out.push_str(
        "    napi_create_function(env, \"liveObjects\", (size_t)-1, napi_fn_live_objects, NULL, &fn);\n\
             napi_set_named_property(env, exports, \"liveObjects\", fn);\n",
    );
    for r in &records {
        let rn = sanitize_ident(&r.name);
        let methods = collect_record_methods(program, &r.name);
        let nprops = r.fields.len() + methods.len();
        out.push_str(&format!("    {{\n        napi_property_descriptor props[{max_p}] = {{\n", max_p = nprops.max(1)));
        for f in &r.fields {
            let fnm = sanitize_ident(&f.name);
            out.push_str(&format!(
                "            {{\"{}\", NULL, NULL, napi_get_{rn}_{fnm}, NULL, NULL, 0, NULL}},\n",
                f.name
            ));
        }
        for m in &methods {
            let mn = sanitize_ident(&m.name);
            out.push_str(&format!(
                "            {{\"{}\", NULL, napi_m_{rn}_{mn}, NULL, NULL, NULL, 0, NULL}},\n",
                m.name
            ));
        }
        out.push_str(&format!(
            "        }};\n\
                 napi_value cls = NULL;\n\
                 napi_define_class(env, \"{}\", (size_t)-1, napi_ctor_{rn}, NULL, {nprops}, props, &cls);\n\
                 napi_set_named_property(env, exports, \"{}\", cls);\n\
             }}\n",
            r.name, r.name
        ));
    }
    out.push_str(&format!(
        "    return exports;\n}}\n\n\
         static napi_module tenon_napi_mod = {{ 1, 0, __FILE__, tenon_napi_init, \"{mident}\", NULL, {{0}} }};\n\
         __attribute__((constructor)) static void tenon_napi_register_ctor(void) {{ napi_module_register(&tenon_napi_mod); }}\n"
    ));
    out
}

fn emit_napi_return(out: &mut String, rt: &BridgeTy, raw_bits: bool) {
    let var = if raw_bits { "raw" } else { "res" };
    out.push_str("    napi_value out = NULL;\n");
    match rt {
        BridgeTy::Void => out.push_str("    napi_get_undefined(env, &out);\n"),
        BridgeTy::Float | BridgeTy::Float32 => {
            if raw_bits {
                out.push_str("    napi_create_double(env, tn_bits_to_f64(raw), &out);\n");
            } else {
                out.push_str(&format!("    napi_create_double(env, (double){var}, &out);\n"));
            }
        }
        BridgeTy::Bool => out.push_str(&format!("    napi_get_boolean(env, {var} != 0, &out);\n")),
        BridgeTy::Text => out.push_str(&format!(
            "    napi_create_string_utf8(env, tenon_text_data_ptr({var}), (size_t)tenon_text_byte_len({var}), &out);\n\
                 tenon_rc_dec({var});\n"
        )),
        BridgeTy::ListFloat => out.push_str(&format!("    return tn_napi_from_list_f64(env, {var});\n")),
        BridgeTy::ListInt => out.push_str(&format!("    return tn_napi_from_list_int(env, {var});\n")),
        _ => out.push_str(&format!("    napi_create_int64(env, (int64_t){var}, &out);\n")),
    }
    out.push_str("    return out;\n");
}

pub fn emit_napi_dts(program: &Program, mod_name: &str) -> String {
    let aliases = collect_aliases(program);
    let records = collect_export_records(program);
    let rec_names: BTreeSet<String> = records.iter().map(|r| r.name.clone()).collect();
    let fns = collect_export_fns(program);
    let mut out = format!("// Auto-generated TypeScript declarations for Node-API addon `{}.node`\n\n", mod_name);
    let map_ts = |bt: &BridgeTy| -> String {
        match bt {
            BridgeTy::Int | BridgeTy::Float | BridgeTy::Float32 => "number".into(),
            BridgeTy::Bool => "boolean".into(),
            BridgeTy::Text => "string".into(),
            BridgeTy::ListInt | BridgeTy::ListFloat => "number[]".into(),
            BridgeTy::Record(r) => r.clone(),
            BridgeTy::Void => "void".into(),
        }
    };
    for r in &records {
        out.push_str(&format!("export declare class {} {{\n", r.name));
        for f in &r.fields {
            out.push_str(&format!("  readonly {}: {};\n", f.name, map_ts(&resolve_bridge_ty(&f.ty, &aliases, &rec_names))));
        }
        let cargs: Vec<String> = r.fields.iter().map(|f| format!("{}: {}", f.name, map_ts(&resolve_bridge_ty(&f.ty, &aliases, &rec_names)))).collect();
        out.push_str(&format!("  constructor({});\n", cargs.join(", ")));
        for m in collect_record_methods(program, &r.name) {
            let ps: Vec<String> = m.params[1..].iter().map(|p| format!("{}: {}", p.name, map_ts(&resolve_bridge_ty(&p.ty, &aliases, &rec_names)))).collect();
            let rt = m.ret_ty.as_ref().map(|t| map_ts(&resolve_bridge_ty(t, &aliases, &rec_names))).unwrap_or_else(|| "void".into());
            out.push_str(&format!("  {}({}): {};\n", m.name, ps.join(", "), rt));
        }
        out.push_str("}\n\n");
    }
    for (f, _) in &fns {
        let ps: Vec<String> = f.params.iter().map(|p| format!("{}: {}", p.name, map_ts(&resolve_bridge_ty(&p.ty, &aliases, &rec_names)))).collect();
        let rt = f.ret_ty.as_ref().map(|t| map_ts(&resolve_bridge_ty(t, &aliases, &rec_names))).unwrap_or_else(|| "void".into());
        out.push_str(&format!("export declare function {}({}): {};\n", f.name, ps.join(", "), rt));
    }
    out.push_str("export declare function liveObjects(): number;\n");
    out
}

// ============================================================================
// 3. JAVA & KOTLIN / ANDROID JNI C GLUE + JAVA CLASS
// ============================================================================

fn jni_c_ty(bt: &BridgeTy) -> &'static str {
    match bt {
        BridgeTy::Float => "jdouble",
        BridgeTy::Float32 => "jfloat",
        BridgeTy::Bool => "jboolean",
        BridgeTy::Text => "jstring",
        BridgeTy::Void => "void",
        _ => "jlong",
    }
}

fn java_ty(bt: &BridgeTy) -> &'static str {
    match bt {
        BridgeTy::Float => "double",
        BridgeTy::Float32 => "float",
        BridgeTy::Bool => "boolean",
        BridgeTy::Text => "String",
        BridgeTy::Void => "void",
        _ => "long",
    }
}

pub fn emit_jni_c_bridge(program: &Program, java_pkg: &str, class_name: &str) -> String {
    let aliases = collect_aliases(program);
    let records = collect_export_records(program);
    let rec_names: BTreeSet<String> = records.iter().map(|r| r.name.clone()).collect();
    let fns = collect_export_fns(program);
    let pkg_prefix = if java_pkg.is_empty() {
        format!("Java_{}", class_name.replace('_', "_1"))
    } else {
        format!("Java_{}_{}", java_pkg.replace('.', "_"), class_name.replace('_', "_1"))
    };

    let mut out = String::from(
        "/* Auto-generated JNI C Bridge by Tenon (`--export java` / `--export kotlin`) */\n\
         #include <jni.h>\n\
         #include <stdbool.h>\n\
         #include <stdint.h>\n\
         #include <string.h>\n\n\
         extern void tenon_rc_dec(int64_t ptr);\n\
         extern int64_t tenon_text_new(int64_t data_ptr, int64_t byte_len);\n\
         extern const char* tenon_text_data_ptr(int64_t ptr);\n\
         extern int64_t tenon_rt_live_objects(void);\n\n",
    );
    for (f, csym) in &fns {
        let rt = f.ret_ty.as_ref().map(|t| resolve_bridge_ty(t, &aliases, &rec_names)).unwrap_or(BridgeTy::Void);
        let ps: Vec<String> = if f.params.is_empty() {
            vec!["void".into()]
        } else {
            f.params.iter().map(|p| c_abi_ty(&resolve_bridge_ty(&p.ty, &aliases, &rec_names)).to_string()).collect()
        };
        out.push_str(&format!("extern {} {}({});\n", c_abi_ty(&rt), csym, ps.join(", ")));
    }
    out.push('\n');

    for (f, csym) in &fns {
        let rt = f.ret_ty.as_ref().map(|t| resolve_bridge_ty(t, &aliases, &rec_names)).unwrap_or(BridgeTy::Void);
        let jni_sym = format!("{}_{}", pkg_prefix, f.name.replace('_', "_1"));
        let mut jparams = vec!["JNIEnv* env".to_string(), "jclass cls".to_string()];
        let mut cargs = Vec::new();
        let mut cleanups = Vec::new();
        for (pi, p) in f.params.iter().enumerate() {
            let pt = resolve_bridge_ty(&p.ty, &aliases, &rec_names);
            jparams.push(format!("{} a{pi}", jni_c_ty(&pt)));
            match pt {
                BridgeTy::Text => {
                    cargs.push(format!("h{pi}"));
                    cleanups.push(pi);
                }
                BridgeTy::Bool => cargs.push(format!("(a{pi} != 0)")),
                BridgeTy::Float => cargs.push(format!("(double)a{pi}")),
                BridgeTy::Float32 => cargs.push(format!("(float)a{pi}")),
                _ => cargs.push(format!("(int64_t)a{pi}")),
            }
        }
        out.push_str(&format!(
            "JNIEXPORT {} JNICALL {}({}) {{\n    (void)env; (void)cls;\n",
            jni_c_ty(&rt),
            jni_sym,
            jparams.join(", ")
        ));
        for &pi in &cleanups {
            out.push_str(&format!(
                "    const char* s{pi} = (*env)->GetStringUTFChars(env, a{pi}, NULL);\n\
                     int64_t h{pi} = tenon_text_new((int64_t)(uintptr_t)s{pi}, (int64_t)strlen(s{pi}));\n\
                     (*env)->ReleaseStringUTFChars(env, a{pi}, s{pi});\n"
            ));
        }
        match &rt {
            BridgeTy::Void => {
                out.push_str(&format!("    {csym}({});\n", cargs.join(", ")));
                for pi in cleanups { out.push_str(&format!("    tenon_rc_dec(h{pi});\n")); }
            }
            BridgeTy::Text => {
                out.push_str(&format!("    int64_t res = {csym}({});\n", cargs.join(", ")));
                for pi in cleanups { out.push_str(&format!("    tenon_rc_dec(h{pi});\n")); }
                out.push_str("    jstring js = (*env)->NewStringUTF(env, tenon_text_data_ptr(res));\n    tenon_rc_dec(res);\n    return js;\n");
            }
            BridgeTy::Bool => {
                out.push_str(&format!("    bool res = {csym}({});\n", cargs.join(", ")));
                for pi in cleanups { out.push_str(&format!("    tenon_rc_dec(h{pi});\n")); }
                out.push_str("    return res ? JNI_TRUE : JNI_FALSE;\n");
            }
            _ => {
                out.push_str(&format!("    {} res = {csym}({});\n", c_abi_ty(&rt), cargs.join(", ")));
                for pi in cleanups { out.push_str(&format!("    tenon_rc_dec(h{pi});\n")); }
                out.push_str(&format!("    return ({})res;\n", jni_c_ty(&rt)));
            }
        }
        out.push_str("}\n\n");
    }
    out.push_str(&format!(
        "JNIEXPORT jlong JNICALL {pkg_prefix}_liveObjects(JNIEnv* env, jclass cls) {{\n\
             (void)env; (void)cls; return (jlong)tenon_rt_live_objects();\n\
         }}\n"
    ));
    out
}

pub fn emit_java_bindings(program: &Program, java_pkg: &str, class_name: &str, lib_name: &str) -> String {
    let aliases = collect_aliases(program);
    let records = collect_export_records(program);
    let rec_names: BTreeSet<String> = records.iter().map(|r| r.name.clone()).collect();
    let fns = collect_export_fns(program);

    let mut out = String::from("// Auto-generated Java/Android JNI bindings by Tenon (`--export java`).\n");
    if !java_pkg.is_empty() {
        out.push_str(&format!("package {};\n\n", java_pkg));
    }
    out.push_str(&format!(
        "public final class {} {{\n    static {{ System.loadLibrary(\"{}\"); }}\n\n",
        class_name, lib_name
    ));
    for (f, _) in &fns {
        let rt = f.ret_ty.as_ref().map(|t| resolve_bridge_ty(t, &aliases, &rec_names)).unwrap_or(BridgeTy::Void);
        let ps: Vec<String> = f.params.iter().map(|p| format!("{} {}", java_ty(&resolve_bridge_ty(&p.ty, &aliases, &rec_names)), p.name)).collect();
        out.push_str(&format!("    public static native {} {}({});\n", java_ty(&rt), f.name, ps.join(", ")));
    }
    out.push_str("    public static native long liveObjects();\n}\n");
    out
}

// ============================================================================
// 4. MODERN C++17 RAII HEADER (`.hpp`)
// ============================================================================

pub fn emit_cpp_header(program: &Program, ns: &str) -> String {
    let aliases = collect_aliases(program);
    let records = collect_export_records(program);
    let rec_names: BTreeSet<String> = records.iter().map(|r| r.name.clone()).collect();
    let fns = collect_export_fns(program);

    let mut out = String::from(
        "// Auto-generated C++17 RAII Bridge by Tenon (`--export cpp`)\n\
         #pragma once\n\
         #include <cstdint>\n\
         #include <cstring>\n\
         #include <string>\n\
         #include <string_view>\n\n\
         extern \"C\" {\n\
         int64_t tenon_rc_alloc(int64_t tag, int64_t len);\n\
         void tenon_rc_set_rc_mask(int64_t ptr, int64_t mask);\n\
         void tenon_rc_dec(int64_t ptr);\n\
         int64_t tenon_rc_get_field(int64_t ptr, int64_t idx);\n\
         void tenon_rc_set_field(int64_t ptr, int64_t idx, int64_t val);\n\
         int64_t tenon_text_new(int64_t data_ptr, int64_t byte_len);\n\
         const char* tenon_text_data_ptr(int64_t ptr);\n\
         int64_t tenon_text_byte_len(int64_t ptr);\n\
         int64_t tenon_rt_live_objects(void);\n",
    );
    for (f, csym) in &fns {
        let rt = f.ret_ty.as_ref().map(|t| resolve_bridge_ty(t, &aliases, &rec_names)).unwrap_or(BridgeTy::Void);
        let ps: Vec<String> = f.params.iter().map(|p| c_abi_ty(&resolve_bridge_ty(&p.ty, &aliases, &rec_names)).to_string()).collect();
        out.push_str(&format!("{} {}({});\n", c_abi_ty(&rt), csym, ps.join(", ")));
    }
    out.push_str("}\n\n");

    out.push_str(&format!("namespace {} {{\n\n", sanitize_ident(ns)));
    out.push_str(
        "class ScopedText {\n\
             int64_t h_;\n\
         public:\n\
             explicit ScopedText(std::string_view sv) : h_(tenon_text_new(reinterpret_cast<int64_t>(sv.data()), static_cast<int64_t>(sv.size()))) {}\n\
             ~ScopedText() { if (h_) tenon_rc_dec(h_); }\n\
             ScopedText(const ScopedText&) = delete;\n\
             int64_t raw() const noexcept { return h_; }\n\
         };\n\n\
         inline std::string take_text(int64_t h) {\n\
             std::string s(tenon_text_data_ptr(h), static_cast<size_t>(tenon_text_byte_len(h)));\n\
             tenon_rc_dec(h);\n\
             return s;\n\
         }\n\n\
         inline int64_t live_objects() noexcept { return tenon_rt_live_objects(); }\n\n",
    );

    let cpp_ty = |bt: &BridgeTy, is_param: bool| -> &'static str {
        match bt {
            BridgeTy::Float => "double",
            BridgeTy::Float32 => "float",
            BridgeTy::Bool => "bool",
            BridgeTy::Text => if is_param { "std::string_view" } else { "std::string" },
            BridgeTy::Void => "void",
            _ => "int64_t",
        }
    };
    for (f, csym) in &fns {
        let rt = f.ret_ty.as_ref().map(|t| resolve_bridge_ty(t, &aliases, &rec_names)).unwrap_or(BridgeTy::Void);
        let ps: Vec<String> = f.params.iter().map(|p| format!("{} {}", cpp_ty(&resolve_bridge_ty(&p.ty, &aliases, &rec_names), true), p.name)).collect();
        out.push_str(&format!("inline {} {}({}) {{\n", cpp_ty(&rt, false), f.name, ps.join(", ")));
        let mut cargs = Vec::new();
        for p in &f.params {
            if matches!(resolve_bridge_ty(&p.ty, &aliases, &rec_names), BridgeTy::Text) {
                out.push_str(&format!("    ScopedText __t_{}({});\n", p.name, p.name));
                cargs.push(format!("__t_{}.raw()", p.name));
            } else {
                cargs.push(p.name.clone());
            }
        }
        match &rt {
            BridgeTy::Void => out.push_str(&format!("    ::{}({});\n", csym, cargs.join(", "))),
            BridgeTy::Text => out.push_str(&format!("    return take_text(::{}({}));\n", csym, cargs.join(", "))),
            _ => out.push_str(&format!("    return ::{}({});\n", csym, cargs.join(", "))),
        }
        out.push_str("}\n\n");
    }
    out.push_str("} // namespace\n");
    out
}

// ============================================================================
// 5. SWIFT 6 (iOS/macOS) RICH WRAPPER WITH UTF-8 `String` & ARC `deinit`
// ============================================================================

pub fn emit_swift_rich_bindings(program: &Program) -> String {
    let aliases = collect_aliases(program);
    let records = collect_export_records(program);
    let rec_names: BTreeSet<String> = records.iter().map(|r| r.name.clone()).collect();
    let fns = collect_export_fns(program);

    let mut out = String::from(
        "// Auto-generated Swift 6 / iOS / macOS Native Bridge by Tenon (`--export swift`)\n\
         import Foundation\n\n\
         @_silgen_name(\"tenon_rc_dec\") private func _tn_rc_dec(_ p: Int64)\n\
         @_silgen_name(\"tenon_text_new\") private func _tn_text_new(_ ptr: Int64, _ len: Int64) -> Int64\n\
         @_silgen_name(\"tenon_text_data_ptr\") private func _tn_text_data_ptr(_ p: Int64) -> UnsafePointer<CChar>\n\
         @_silgen_name(\"tenon_text_byte_len\") private func _tn_text_byte_len(_ p: Int64) -> Int64\n\
         @_silgen_name(\"tenon_rt_live_objects\") public func tenonLiveObjects() -> Int64\n\n\
         private func _tnMakeText(_ s: String) -> Int64 {\n\
             var copy = s\n\
             return copy.withUTF8 { buf in\n\
                 guard let base = buf.baseAddress else { return _tn_text_new(0, 0) }\n\
                 return _tn_text_new(Int64(Int(bitPattern: base)), Int64(buf.count))\n\
             }\n\
         }\n\
         private func _tnTakeText(_ h: Int64) -> String {\n\
             defer { _tn_rc_dec(h) }\n\
             let ptr = _tn_text_data_ptr(h)\n\
             let len = Int(_tn_text_byte_len(h))\n\
             let raw = UnsafeBufferPointer(start: ptr, count: len)\n\
             return String(decoding: raw.map { UInt8(bitPattern: $0) }, as: UTF8.self)\n\
         }\n\n",
    );

    let sw_abi = |bt: &BridgeTy| -> &'static str {
        match bt {
            BridgeTy::Float => "Double",
            BridgeTy::Float32 => "Float",
            BridgeTy::Bool => "Bool",
            BridgeTy::Void => "Void",
            _ => "Int64",
        }
    };
    let sw_hi = |bt: &BridgeTy| -> &'static str {
        match bt {
            BridgeTy::Float => "Double",
            BridgeTy::Float32 => "Float",
            BridgeTy::Bool => "Bool",
            BridgeTy::Text => "String",
            BridgeTy::Void => "Void",
            _ => "Int64",
        }
    };

    for (f, csym) in &fns {
        let rt = f.ret_ty.as_ref().map(|t| resolve_bridge_ty(t, &aliases, &rec_names)).unwrap_or(BridgeTy::Void);
        let raw_ps: Vec<String> = f.params.iter().map(|p| format!("_ {}: {}", p.name, sw_abi(&resolve_bridge_ty(&p.ty, &aliases, &rec_names)))).collect();
        let hi_ps: Vec<String> = f.params.iter().map(|p| format!("_ {}: {}", p.name, sw_hi(&resolve_bridge_ty(&p.ty, &aliases, &rec_names)))).collect();
        out.push_str(&format!("@_silgen_name(\"{csym}\")\nprivate func _raw_{}({}) -> {}\n", f.name, raw_ps.join(", "), sw_abi(&rt)));
        out.push_str(&format!("public func {}({}) -> {} {{\n", f.name, hi_ps.join(", "), sw_hi(&rt)));
        let mut cargs = Vec::new();
        for p in &f.params {
            if matches!(resolve_bridge_ty(&p.ty, &aliases, &rec_names), BridgeTy::Text) {
                out.push_str(&format!("    let __h_{} = _tnMakeText({})\n    defer {{ _tn_rc_dec(__h_{}) }}\n", p.name, p.name, p.name));
                cargs.push(format!("__h_{}", p.name));
            } else {
                cargs.push(p.name.clone());
            }
        }
        match &rt {
            BridgeTy::Void => out.push_str(&format!("    _raw_{}({})\n", f.name, cargs.join(", "))),
            BridgeTy::Text => out.push_str(&format!("    return _tnTakeText(_raw_{}({}))\n", f.name, cargs.join(", "))),
            _ => out.push_str(&format!("    return _raw_{}({})\n", f.name, cargs.join(", "))),
        }
        out.push_str("}\n\n");
    }
    out
}

pub fn emit_swift_modulemap(mod_name: &str, header_name: &str) -> String {
    format!(
        "module {} {{\n    header \"{}\"\n    export *\n}}\n",
        sanitize_ident(mod_name),
        header_name
    )
}

// ============================================================================
// 6. SAFE RUST FFI WRAPPER (`.rs`) & ZIG FFI WRAPPER (`.zig`)
// ============================================================================

pub fn emit_rust_bindings(program: &Program, lib_name: &str) -> String {
    let aliases = collect_aliases(program);
    let records = collect_export_records(program);
    let rec_names: BTreeSet<String> = records.iter().map(|r| r.name.clone()).collect();
    let fns = collect_export_fns(program);

    let mut out = format!(
        "// Auto-generated Safe Rust FFI Bindings by Tenon (`--export rust`)\n\
         #![allow(dead_code, non_snake_case)]\n\
         mod sys {{\n\
             use std::ffi::c_char;\n\n\
             #[link(name = \"{}\")]\n\
             extern \"C\" {{\n\
                 pub fn tenon_rc_dec(p: i64);\n\
                 pub fn tenon_text_new(data_ptr: i64, byte_len: i64) -> i64;\n\
                 pub fn tenon_text_data_ptr(p: i64) -> *const c_char;\n\
                 pub fn tenon_text_byte_len(p: i64) -> i64;\n\
                 pub fn tenon_rt_live_objects() -> i64;\n",
        lib_name
    );
    let rs_abi = |bt: &BridgeTy| -> &'static str {
        match bt {
            BridgeTy::Float => "f64",
            BridgeTy::Float32 => "f32",
            BridgeTy::Bool => "bool",
            BridgeTy::Void => "()",
            _ => "i64",
        }
    };
    let rs_hi = |bt: &BridgeTy, is_param: bool| -> &'static str {
        match bt {
            BridgeTy::Float => "f64",
            BridgeTy::Float32 => "f32",
            BridgeTy::Bool => "bool",
            BridgeTy::Text => if is_param { "&str" } else { "String" },
            BridgeTy::Void => "()",
            _ => "i64",
        }
    };
    for (f, csym) in &fns {
        let rt = f.ret_ty.as_ref().map(|t| resolve_bridge_ty(t, &aliases, &rec_names)).unwrap_or(BridgeTy::Void);
        let ps: Vec<String> = f.params.iter().map(|p| format!("{}: {}", p.name, rs_abi(&resolve_bridge_ty(&p.ty, &aliases, &rec_names)))).collect();
        out.push_str(&format!("        pub fn {}({}) -> {};\n", csym, ps.join(", "), rs_abi(&rt)));
    }
    out.push_str(
        "    }\n\
         }\n\n\
         struct ScopedTenonText(i64);\n\
         impl ScopedTenonText {\n\
             fn new(s: &str) -> Self { Self(unsafe { sys::tenon_text_new(s.as_ptr() as i64, s.len() as i64) }) }\n\
         }\n\
         impl Drop for ScopedTenonText { fn drop(&mut self) { unsafe { sys::tenon_rc_dec(self.0); } } }\n\n\
         fn take_tenon_text(h: i64) -> String {\n\
             unsafe {\n\
                 let ptr = sys::tenon_text_data_ptr(h) as *const u8;\n\
                 let len = sys::tenon_text_byte_len(h).max(0) as usize;\n\
                 let s = String::from_utf8_lossy(std::slice::from_raw_parts(ptr, len)).into_owned();\n\
                 sys::tenon_rc_dec(h);\n\
                 s\n\
             }\n\
         }\n\n\
         pub fn live_objects() -> i64 { unsafe { sys::tenon_rt_live_objects() } }\n\n",
    );
    for (f, csym) in &fns {
        let rt = f.ret_ty.as_ref().map(|t| resolve_bridge_ty(t, &aliases, &rec_names)).unwrap_or(BridgeTy::Void);
        let ps: Vec<String> = f.params.iter().map(|p| format!("{}: {}", p.name, rs_hi(&resolve_bridge_ty(&p.ty, &aliases, &rec_names), true))).collect();
        out.push_str(&format!("pub fn {}({}) -> {} {{\n", f.name, ps.join(", "), rs_hi(&rt, false)));
        let mut cargs = Vec::new();
        for p in &f.params {
            if matches!(resolve_bridge_ty(&p.ty, &aliases, &rec_names), BridgeTy::Text) {
                out.push_str(&format!("    let __t_{} = ScopedTenonText::new({});\n", p.name, p.name));
                cargs.push(format!("__t_{}.0", p.name));
            } else {
                cargs.push(p.name.clone());
            }
        }
        match &rt {
            BridgeTy::Void => out.push_str(&format!("    unsafe {{ sys::{}({}); }}\n", csym, cargs.join(", "))),
            BridgeTy::Text => out.push_str(&format!("    take_tenon_text(unsafe {{ sys::{}({}) }})\n", csym, cargs.join(", "))),
            _ => out.push_str(&format!("    unsafe {{ sys::{}({}) }}\n", csym, cargs.join(", "))),
        }
        out.push_str("}\n\n");
    }
    out
}

pub fn emit_zig_bindings(program: &Program) -> String {
    let aliases = collect_aliases(program);
    let records = collect_export_records(program);
    let rec_names: BTreeSet<String> = records.iter().map(|r| r.name.clone()).collect();
    let fns = collect_export_fns(program);

    let zig_abi = |bt: &BridgeTy| -> &'static str {
        match bt {
            BridgeTy::Float => "f64",
            BridgeTy::Float32 => "f32",
            BridgeTy::Bool => "bool",
            BridgeTy::Void => "void",
            _ => "i64",
        }
    };
    let mut out = String::from(
        "// Auto-generated Zig Native FFI Bindings by Tenon (`--export zig`)\n\
         const std = @import(\"std\");\n\n\
         const sys = struct {\n\
             pub extern \"c\" fn tenon_rc_dec(p: i64) void;\n\
             pub extern \"c\" fn tenon_text_new(data_ptr: i64, byte_len: i64) i64;\n\
             pub extern \"c\" fn tenon_text_data_ptr(p: i64) [*]const u8;\n\
             pub extern \"c\" fn tenon_text_byte_len(p: i64) i64;\n\
             pub extern \"c\" fn tenon_rt_live_objects() i64;\n",
    );
    for (f, csym) in &fns {
        let rt = f.ret_ty.as_ref().map(|t| resolve_bridge_ty(t, &aliases, &rec_names)).unwrap_or(BridgeTy::Void);
        let ps: Vec<String> = f.params.iter().map(|p| format!("{}: {}", p.name, zig_abi(&resolve_bridge_ty(&p.ty, &aliases, &rec_names)))).collect();
        out.push_str(&format!("    pub extern \"c\" fn {}({}) {};\n", csym, ps.join(", "), zig_abi(&rt)));
    }
    out.push_str("};\n\npub fn liveObjects() i64 { return sys.tenon_rt_live_objects(); }\n\n");
    for (f, csym) in &fns {
        let rt = f.ret_ty.as_ref().map(|t| resolve_bridge_ty(t, &aliases, &rec_names)).unwrap_or(BridgeTy::Void);
        let has_text_ret = matches!(rt, BridgeTy::Text);
        let mut ps: Vec<String> = Vec::new();
        if has_text_ret {
            ps.push("allocator: std.mem.Allocator".into());
        }
        for p in &f.params {
            let pt = resolve_bridge_ty(&p.ty, &aliases, &rec_names);
            let zt = match pt {
                BridgeTy::Text => "[]const u8",
                _ => zig_abi(&pt),
            };
            ps.push(format!("{}: {}", p.name, zt));
        }
        let zret = if has_text_ret { "![]u8" } else { zig_abi(&rt) };
        out.push_str(&format!("pub fn {}({}) {} {{\n", f.name, ps.join(", "), zret));
        let mut cargs = Vec::new();
        for p in &f.params {
            if matches!(resolve_bridge_ty(&p.ty, &aliases, &rec_names), BridgeTy::Text) {
                out.push_str(&format!(
                    "    const __h_{} = sys.tenon_text_new(@intCast(@intFromPtr({}.ptr)), @intCast({}.len));\n\
                         defer sys.tenon_rc_dec(__h_{});\n",
                    p.name, p.name, p.name, p.name
                ));
                cargs.push(format!("__h_{}", p.name));
            } else {
                cargs.push(p.name.clone());
            }
        }
        if has_text_ret {
            out.push_str(&format!(
                "    const __r = sys.{}({});\n\
                     defer sys.tenon_rc_dec(__r);\n\
                     const __len: usize = @intCast(sys.tenon_text_byte_len(__r));\n\
                     return try allocator.dupe(u8, sys.tenon_text_data_ptr(__r)[0..__len]);\n",
                csym,
                cargs.join(", ")
            ));
        } else if matches!(rt, BridgeTy::Void) {
            out.push_str(&format!("    sys.{}({});\n", csym, cargs.join(", ")));
        } else {
            out.push_str(&format!("    return sys.{}({});\n", csym, cargs.join(", ")));
        }
        out.push_str("}\n\n");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenon_syntax::parse;

    #[test]
    fn emits_all_polyglot_bridges_cleanly() {
        let src = r#"
pub record Box2D {
    name Text
    width Float
}
impl Box2D {
    pub fn area(self, height Float) Float { self.width * height }
}
@export
pub fn greet(who Text) Text { "hi " ++ who }
@export
pub fn add_i(a Int, b Int) Int { a + b }
"#;
        let prog = parse(src).expect("parse");
        let py_c = emit_python_c_ext(&prog, "demo");
        assert!(py_c.contains("PyInit_demo"));
        assert!(py_c.contains("PyTenon_Box2D_Type"));
        assert!(py_c.contains("tenon_fn___m__Box2D__area"));
        assert!(py_c.contains("Py_BEGIN_ALLOW_THREADS"));
        assert!(py_c.contains("METH_FASTCALL"));
        assert!(py_c.contains("PyObject_GetBuffer"));

        let pyi = emit_python_pyi(&prog, "demo");
        assert!(pyi.contains("class Box2D:"));
        assert!(pyi.contains("def greet(who: str) -> str:"));

        let napi = emit_napi_c_addon(&prog, "demo");
        assert!(napi.contains("napi_define_class"));
        assert!(napi.contains("napi_is_typedarray"));
        assert!(napi.contains("tenon_fn___m__Box2D__area"));

        let jni = emit_jni_c_bridge(&prog, "", "Demo");
        assert!(jni.contains("Java_Demo_greet"));
        assert!(jni.contains("Java_Demo_add_1i"));

        let hpp = emit_cpp_header(&prog, "demo");
        assert!(hpp.contains("class ScopedText"));
        assert!(hpp.contains("namespace demo"));

        let sw = emit_swift_rich_bindings(&prog);
        assert!(sw.contains("public func greet(_ who: String) -> String"));

        let rs = emit_rust_bindings(&prog, "demo");
        assert!(rs.contains("pub fn greet(who: &str) -> String"));

        let zg = emit_zig_bindings(&prog);
        assert!(zg.contains("pub fn add_i(a: i64, b: i64) i64"));
    }
}
