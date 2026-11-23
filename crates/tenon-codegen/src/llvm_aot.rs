use crate::perceus::{TAG_ERR, TAG_LIST, TAG_NONE, TAG_OK, TAG_RECORD_BASE, TAG_SOME};
use crate::split_async_segments;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::process::Command;
use tenon_syntax::*;

#[derive(Debug, Clone)]
pub struct BuildArtifact {
    pub output_path: String,
    pub llvm_ir_bytes: usize,
    pub binary_bytes: u64,
    pub opt_level: String,
}

#[derive(Debug, Clone)]
struct LlvmRecordLayout {
    tag: i64,
    fields: Vec<(String, Vec<Expr>)>,
    where_clauses: Vec<Expr>,
}

#[derive(Debug, Clone)]
struct LlvmFnMeta {
    params_inout: Vec<bool>,
    returns_rc: bool,
}

const TENON_AOT_RUNTIME_C: &str = r#"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int64_t rc;
    int64_t tag;
    int64_t len;
    int64_t cap;
    int64_t fields[];
} TenonObj;

static int64_t g_live = 0;
static int64_t g_allocs = 0;
static int64_t g_reuse = 0;
static int64_t g_cow = 0;
static int64_t g_polls = 0;
static int64_t g_suspends = 0;

void tenon_rt_assert_fail(int64_t line) {
    fprintf(stderr, "TENON ASSERTION FAILED at line %lld\n", (long long)line);
    exit(1);
}

int64_t tenon_rt_print_int(int64_t v) {
    printf("%lld\n", (long long)v);
    return v;
}

int64_t tenon_rc_alloc(int64_t tag, int64_t len) {
    int64_t lu = len < 0 ? 0 : len;
    int64_t cap = lu < 1 ? 1 : lu;
    TenonObj* obj = (TenonObj*)calloc(1, sizeof(TenonObj) + (size_t)cap * sizeof(int64_t));
    if (!obj) abort();
    obj->rc = 1;
    obj->tag = tag;
    obj->len = lu;
    obj->cap = cap;
    g_live++;
    g_allocs++;
    return (int64_t)(uintptr_t)obj;
}

int64_t tenon_rc_inc(int64_t ptr) {
    if (ptr == 0) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    obj->rc++;
    return ptr;
}

void tenon_rc_dec(int64_t ptr) {
    if (ptr == 0) return;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    obj->rc--;
    if (obj->rc <= 0) {
        g_live--;
        free(obj);
    }
}

int64_t tenon_rc_ensure_unique(int64_t ptr) {
    if (ptr == 0) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    if (obj->rc == 1) {
        g_reuse++;
        return ptr;
    }
    g_cow++;
    int64_t fresh_p = tenon_rc_alloc(obj->tag, obj->len);
    TenonObj* fresh = (TenonObj*)(uintptr_t)fresh_p;
    for (int64_t i = 0; i < obj->len; i++) fresh->fields[i] = obj->fields[i];
    tenon_rc_dec(ptr);
    return fresh_p;
}

int64_t tenon_rc_get_field(int64_t ptr, int64_t idx) {
    if (ptr == 0 || idx < 0) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    if (idx >= obj->len) return 0;
    return obj->fields[idx];
}

void tenon_rc_set_field(int64_t ptr, int64_t idx, int64_t val) {
    if (ptr == 0 || idx < 0) return;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    if (idx < obj->len) obj->fields[idx] = val;
}

int64_t tenon_rc_get_tag(int64_t ptr) {
    if (ptr == 0) return -1;
    return ((TenonObj*)(uintptr_t)ptr)->tag;
}

int64_t tenon_rc_get_len(int64_t ptr) {
    if (ptr == 0) return 0;
    return ((TenonObj*)(uintptr_t)ptr)->len;
}

int64_t tenon_text_new(int64_t data_ptr, int64_t byte_len) {
    int64_t blen = byte_len < 0 ? 0 : byte_len;
    int64_t words = (blen + 7) / 8;
    if (words < 1) words = 1;
    int64_t p = tenon_rc_alloc(5, words);
    TenonObj* obj = (TenonObj*)(uintptr_t)p;
    obj->len = blen;
    if (blen > 0 && data_ptr != 0) {
        memcpy((void*)obj->fields, (const void*)(uintptr_t)data_ptr, (size_t)blen);
    }
    return p;
}

int64_t tenon_text_eq(int64_t a, int64_t b) {
    if (a == b) return 1;
    if (a == 0 || b == 0) return 0;
    TenonObj* ha = (TenonObj*)(uintptr_t)a;
    TenonObj* hb = (TenonObj*)(uintptr_t)b;
    if (ha->len != hb->len) return 0;
    return memcmp((const void*)ha->fields, (const void*)hb->fields, (size_t)ha->len) == 0 ? 1 : 0;
}

int64_t tenon_list_push(int64_t ptr, int64_t val) {
    if (ptr == 0) {
        int64_t f = tenon_rc_alloc(4, 1);
        tenon_rc_set_field(f, 0, val);
        return f;
    }
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    int64_t old_len = obj->len;
    if (obj->rc == 1 && old_len < obj->cap) {
        g_reuse++;
        obj->fields[old_len] = val;
        obj->len = old_len + 1;
        return ptr;
    }
    int64_t new_cap = obj->cap * 2;
    if (new_cap < old_len + 1) new_cap = old_len + 1;
    int64_t fp = tenon_rc_alloc(4, new_cap);
    TenonObj* fresh = (TenonObj*)(uintptr_t)fp;
    fresh->len = old_len + 1;
    for (int64_t i = 0; i < old_len; i++) fresh->fields[i] = obj->fields[i];
    fresh->fields[old_len] = val;
    if (obj->rc > 1) g_cow++; else g_reuse++;
    tenon_rc_dec(ptr);
    return fp;
}

int64_t tenon_async_frame_alloc(int64_t total_slots, int64_t poll_fn_ptr) {
    int64_t slots = total_slots < 4 ? 4 : total_slots;
    int64_t f = tenon_rc_alloc(6, slots);
    tenon_rc_set_field(f, 0, 0);
    tenon_rc_set_field(f, 1, poll_fn_ptr);
    tenon_rc_set_field(f, 2, 0);
    tenon_rc_set_field(f, 3, 0);
    return f;
}

static int64_t tenon_async_yield_poll(int64_t frame) {
    int64_t st = tenon_rc_get_field(frame, 0);
    if (st == 0) {
        tenon_rc_set_field(frame, 0, 1);
        g_suspends++;
        return 0;
    } else {
        int64_t val = tenon_rc_get_field(frame, 4);
        tenon_rc_set_field(frame, 2, val);
        tenon_rc_set_field(frame, 0, -1);
        return 1;
    }
}

int64_t tenon_async_yield_create(int64_t val) {
    int64_t f = tenon_async_frame_alloc(5, (int64_t)(uintptr_t)&tenon_async_yield_poll);
    tenon_rc_set_field(f, 4, val);
    return f;
}

int64_t tenon_async_poll_frame(int64_t frame) {
    if (frame == 0) return 1;
    g_polls++;
    int64_t pp = tenon_rc_get_field(frame, 1);
    if (pp == 0) return 1;
    int64_t (*poll_fn)(int64_t) = (int64_t (*)(int64_t))(uintptr_t)pp;
    return poll_fn(frame);
}

int64_t tenon_async_block_on(int64_t frame) {
    for (;;) {
        if (tenon_async_poll_frame(frame) != 0) {
            int64_t res = tenon_rc_get_field(frame, 2);
            tenon_rc_dec(frame);
            return res;
        }
    }
}

extern int64_t tenon_fn_main(void);

int main(void) {
    int64_t res = tenon_fn_main();
    printf("main() => %lld\n", (long long)res);
    if (g_allocs > 0 || g_polls > 0) {
        printf("Perceus/Async stats: allocs=%lld, unique_reuse_hits=%lld, cow_clones=%lld, live_objects=%lld, async_polls=%lld, async_suspends=%lld\n",
               (long long)g_allocs, (long long)g_reuse, (long long)g_cow,
               (long long)g_live, (long long)g_polls, (long long)g_suspends);
    }
    return g_live == 0 ? 0 : 2;
}
"#;

/// Compile a verified Tenon `Program` directly into LLVM 22 IR (`.ll`) and link with `clang`
/// into a standalone native Mach-O/ELF executable!
pub fn compile_to_native_binary(
    program: &Program,
    output_path: &Path,
    emit_ll_path: Option<&Path>,
    opt_level: &str,
) -> Result<BuildArtifact, String> {
    let ir = emit_llvm_ir(program)?;
    let pid = std::process::id();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp_ll = std::env::temp_dir().join(format!("tenon_aot_{}_{}.ll", pid, nonce));
    let tmp_rt = std::env::temp_dir().join(format!("tenon_rt_{}_{}.c", pid, nonce));

    fs::write(&tmp_ll, &ir).map_err(|e| e.to_string())?;
    fs::write(&tmp_rt, TENON_AOT_RUNTIME_C).map_err(|e| e.to_string())?;
    if let Some(p) = emit_ll_path {
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(p, &ir).map_err(|e| e.to_string())?;
    }
    if let Some(parent) = output_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let opt_flag = match opt_level {
        "O0" | "-O0" => "-O0",
        "O2" | "-O2" => "-O2",
        _ => "-O3",
    };

    let out = Command::new("clang")
        .args([
            opt_flag,
            "-Wno-override-module",
            tmp_ll.to_str().unwrap(),
            tmp_rt.to_str().unwrap(),
            "-o",
            output_path.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| format!("failed to invoke LLVM clang: {}", e))?;

    let _ = fs::remove_file(&tmp_ll);
    let _ = fs::remove_file(&tmp_rt);

    if !out.status.success() {
        return Err(format!(
            "LLVM clang compilation failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }

    let binary_bytes = fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
    Ok(BuildArtifact {
        output_path: output_path.display().to_string(),
        llvm_ir_bytes: ir.len(),
        binary_bytes,
        opt_level: opt_flag.to_string(),
    })
}

/// Emit clean, self-contained LLVM 22 IR (opaque `ptr`) for a Tenon `Program`.
pub fn emit_llvm_ir(program: &Program) -> Result<String, String> {
    let mut alias_preds: BTreeMap<String, Expr> = BTreeMap::new();
    for item in &program.items {
        if let Item::TypeAlias(ta) = item {
            if let Some(p) = &ta.predicate {
                alias_preds.insert(ta.name.clone(), p.clone());
            }
        }
    }

    let mut records: BTreeMap<String, LlvmRecordLayout> = BTreeMap::new();
    let mut field_to_idx: BTreeMap<String, usize> = BTreeMap::new();
    let mut enum_tags: BTreeMap<String, i64> = BTreeMap::new();
    enum_tags.insert("Err".into(), TAG_ERR);
    enum_tags.insert("Ok".into(), TAG_OK);
    enum_tags.insert("None".into(), TAG_NONE);
    enum_tags.insert("Some".into(), TAG_SOME);

    let mut next_rec_tag = TAG_RECORD_BASE;
    let mut next_enum_tag = 10i64;
    for item in &program.items {
        match item {
            Item::Record(r) if r.type_params.is_empty() => {
                let mut fields = Vec::new();
                for (idx, f) in r.fields.iter().enumerate() {
                    let mut preds = Vec::new();
                    if let Some(ap) = alias_preds.get(&f.ty.display_name()) {
                        preds.push(ap.clone());
                    }
                    if let Some(fp) = &f.predicate {
                        preds.push(fp.clone());
                    }
                    fields.push((f.name.clone(), preds));
                    field_to_idx.entry(f.name.clone()).or_insert(idx);
                }
                records.insert(
                    r.name.clone(),
                    LlvmRecordLayout {
                        tag: next_rec_tag,
                        fields,
                        where_clauses: r.where_clauses.clone(),
                    },
                );
                next_rec_tag += 1;
            }
            Item::Enum(e) => {
                for v in &e.variants {
                    enum_tags.insert(v.name.clone(), next_enum_tag);
                    enum_tags.insert(format!("{}.{}", e.name, v.name), next_enum_tag);
                    next_enum_tag += 1;
                }
            }
            _ => {}
        }
    }

    let sync_fns: Vec<&FnDecl> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Fn(f) if !f.is_async && f.type_params.is_empty() => Some(f),
            _ => None,
        })
        .collect();
    let async_fns: Vec<&FnDecl> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Fn(f) if f.is_async && f.type_params.is_empty() => Some(f),
            _ => None,
        })
        .collect();

    let mut fn_metas: BTreeMap<String, LlvmFnMeta> = BTreeMap::new();
    for f in &sync_fns {
        fn_metas.insert(
            f.name.clone(),
            LlvmFnMeta {
                params_inout: f.params.iter().map(|p| p.is_inout).collect(),
                returns_rc: f.ret_ty.as_ref().is_some_and(|t| is_rc_type(t, &records)),
            },
        );
    }
    for f in &async_fns {
        fn_metas.insert(
            f.name.clone(),
            LlvmFnMeta {
                params_inout: vec![false; f.params.len()],
                returns_rc: true,
            },
        );
    }
    for (rname, layout) in &records {
        if !layout.fields.is_empty() {
            fn_metas.insert(
                format!("{}__parse", rname),
                LlvmFnMeta {
                    params_inout: vec![false; layout.fields.len()],
                    returns_rc: true,
                },
            );
        }
    }

    let mut string_globals: Vec<(String, String)> = Vec::new();
    let mut fn_defs = String::new();

    // 1. Emit all sync functions
    for f in &sync_fns {
        let mut cg = LlvmFnGen::new(&records, &field_to_idx, &enum_tags, &fn_metas, &mut string_globals);
        fn_defs.push_str(&cg.emit_sync_fn(f)?);
        fn_defs.push('\n');
    }

    // 2. Emit auto-derived `<Record>__parse(...)` parsers
    for (rname, layout) in &records {
        if layout.fields.is_empty() {
            continue;
        }
        let mut cg = LlvmFnGen::new(&records, &field_to_idx, &enum_tags, &fn_metas, &mut string_globals);
        fn_defs.push_str(&cg.emit_record_parser(rname, layout)?);
        fn_defs.push('\n');
    }

    // 3. Emit stackless `async fn` constructors + state-machine poll functions!
    for f in &async_fns {
        let mut cg = LlvmFnGen::new(&records, &field_to_idx, &enum_tags, &fn_metas, &mut string_globals);
        fn_defs.push_str(&cg.emit_async_fn(f)?);
        fn_defs.push('\n');
    }

    let mut ir = String::new();
    ir.push_str("; Generated by Tenon LLVM 22 AOT Release Backend\n\n");
    ir.push_str("declare void @tenon_rt_assert_fail(i64)\n");
    ir.push_str("declare i64 @tenon_rt_print_int(i64)\n");
    ir.push_str("declare i64 @tenon_rc_alloc(i64, i64)\n");
    ir.push_str("declare i64 @tenon_rc_inc(i64)\n");
    ir.push_str("declare void @tenon_rc_dec(i64)\n");
    ir.push_str("declare i64 @tenon_rc_ensure_unique(i64)\n");
    ir.push_str("declare i64 @tenon_rc_get_field(i64, i64)\n");
    ir.push_str("declare void @tenon_rc_set_field(i64, i64, i64)\n");
    ir.push_str("declare i64 @tenon_rc_get_tag(i64)\n");
    ir.push_str("declare i64 @tenon_rc_get_len(i64)\n");
    ir.push_str("declare i64 @tenon_text_new(i64, i64)\n");
    ir.push_str("declare i64 @tenon_text_eq(i64, i64)\n");
    ir.push_str("declare i64 @tenon_list_push(i64, i64)\n");
    ir.push_str("declare i64 @tenon_async_frame_alloc(i64, i64)\n");
    ir.push_str("declare i64 @tenon_async_yield_create(i64)\n");
    ir.push_str("declare i64 @tenon_async_poll_frame(i64)\n");
    ir.push_str("declare i64 @tenon_async_block_on(i64)\n\n");

    for (gname, text) in &string_globals {
        let escaped = escape_llvm_bytes(text.as_bytes());
        ir.push_str(&format!(
            "@{} = private unnamed_addr constant [{} x i8] c\"{}\"\n",
            gname,
            text.len(),
            escaped
        ));
    }
    if !string_globals.is_empty() {
        ir.push('\n');
    }

    ir.push_str(&fn_defs);
    Ok(ir)
}

fn escape_llvm_bytes(bytes: &[u8]) -> String {
    let mut out = String::new();
    for &b in bytes {
        if (0x20..=0x7E).contains(&b) && b != b'"' && b != b'\\' {
            out.push(b as char);
        } else {
            out.push_str(&format!("\\{:02X}", b));
        }
    }
    out
}

fn is_rc_type(ty: &TypeExpr, records: &BTreeMap<String, LlvmRecordLayout>) -> bool {
    let name = ty.display_name();
    records.contains_key(&name)
        || name == "Text"
        || name.starts_with("List[")
        || name.starts_with("Option[")
        || name.starts_with("Result[")
}

struct LlvmFnGen<'a> {
    records: &'a BTreeMap<String, LlvmRecordLayout>,
    field_to_idx: &'a BTreeMap<String, usize>,
    enum_tags: &'a BTreeMap<String, i64>,
    fn_metas: &'a BTreeMap<String, LlvmFnMeta>,
    string_globals: &'a mut Vec<(String, String)>,
    tmp: usize,
    lbl: usize,
    entry_allocas: String,
    code: String,
    var_ptrs: BTreeMap<String, String>,
    var_records: BTreeMap<String, String>,
    inout_vars: BTreeSet<String>,
    owned_rc_vars: BTreeSet<String>,
    text_vars: BTreeSet<String>,
    self_var: Option<String>,
    terminated: bool,
}

impl<'a> LlvmFnGen<'a> {
    fn new(
        records: &'a BTreeMap<String, LlvmRecordLayout>,
        field_to_idx: &'a BTreeMap<String, usize>,
        enum_tags: &'a BTreeMap<String, i64>,
        fn_metas: &'a BTreeMap<String, LlvmFnMeta>,
        string_globals: &'a mut Vec<(String, String)>,
    ) -> Self {
        Self {
            records,
            field_to_idx,
            enum_tags,
            fn_metas,
            string_globals,
            tmp: 0,
            lbl: 0,
            entry_allocas: String::new(),
            code: String::new(),
            var_ptrs: BTreeMap::new(),
            var_records: BTreeMap::new(),
            inout_vars: BTreeSet::new(),
            owned_rc_vars: BTreeSet::new(),
            text_vars: BTreeSet::new(),
            self_var: None,
            terminated: false,
        }
    }

    fn fresh_tmp(&mut self) -> String {
        self.tmp += 1;
        format!("%t{}", self.tmp)
    }

    fn fresh_lbl(&mut self, prefix: &str) -> String {
        self.lbl += 1;
        format!("{}_{}", prefix, self.lbl)
    }

    fn ensure_var_slot(&mut self, name: &str) -> String {
        if let Some(p) = self.var_ptrs.get(name) {
            return p.clone();
        }
        let slot = format!("%v_{}_{}", sanitize(name), self.var_ptrs.len());
        self.entry_allocas
            .push_str(&format!("  {} = alloca i64\n  store i64 0, ptr {}\n", slot, slot));
        self.var_ptrs.insert(name.to_string(), slot.clone());
        slot
    }

    fn emit_sync_fn(&mut self, f: &FnDecl) -> Result<String, String> {
        let mut param_list = Vec::new();
        for (i, p) in f.params.iter().enumerate() {
            let pname = format!("%arg_{}", i);
            param_list.push(format!("i64 {}", pname));
            let slot = self.ensure_var_slot(&p.name);
            self.code
                .push_str(&format!("  store i64 {}, ptr {}\n", pname, slot));
            let tname = p.ty.display_name();
            if self.records.contains_key(&tname) {
                self.var_records.insert(p.name.clone(), tname.clone());
            }
            if tname == "Text" {
                self.text_vars.insert(p.name.clone());
            }
            if p.is_inout {
                self.inout_vars.insert(p.name.clone());
            }
        }

        let ret_ident = f.body.tail.as_deref().and_then(|t| match &t.kind {
            ExprKind::Ident(s) => Some(s.clone()),
            _ => None,
        });
        let ret_val = self.lower_block(&f.body)?;
        if !self.terminated {
            self.emit_cleanup(ret_ident.as_deref());
            self.code.push_str(&format!("  ret i64 {}\n", ret_val));
        }

        Ok(format!(
            "define i64 @tenon_fn_{}({}) {{\nentry:\n{}{}}}\n",
            sanitize(&f.name),
            param_list.join(", "),
            self.entry_allocas,
            self.code
        ))
    }

    fn emit_record_parser(
        &mut self,
        rname: &str,
        layout: &LlvmRecordLayout,
    ) -> Result<String, String> {
        let mut param_list = Vec::new();
        for (i, (fname, _)) in layout.fields.iter().enumerate() {
            let pname = format!("%arg_{}", i);
            param_list.push(format!("i64 {}", pname));
            let slot = self.ensure_var_slot(fname);
            self.code
                .push_str(&format!("  store i64 {}, ptr {}\n", pname, slot));
        }

        for (idx, (fname, preds)) in layout.fields.iter().enumerate() {
            for pred in preds {
                self.self_var = Some(fname.clone());
                let cv = self.lower_expr(pred)?;
                self.self_var = None;
                let cond = self.fresh_tmp();
                let fail_l = self.fresh_lbl("pfail");
                let ok_l = self.fresh_lbl("pok");
                self.code.push_str(&format!(
                    "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n",
                    cond, cv, cond, ok_l, fail_l
                ));
                self.code.push_str(&format!("{}:\n", fail_l));
                let eo = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 1)\n  call void @tenon_rc_set_field(i64 {}, i64 0, i64 {})\n  ret i64 {}\n",
                    eo, TAG_ERR, eo, idx + 1, eo
                ));
                self.code.push_str(&format!("{}:\n", ok_l));
            }
        }

        for wc in &layout.where_clauses {
            let cv = self.lower_expr(wc)?;
            let cond = self.fresh_tmp();
            let fail_l = self.fresh_lbl("wfail");
            let ok_l = self.fresh_lbl("wok");
            self.code.push_str(&format!(
                "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n",
                cond, cv, cond, ok_l, fail_l
            ));
            self.code.push_str(&format!("{}:\n", fail_l));
            let eo = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 1)\n  call void @tenon_rc_set_field(i64 {}, i64 0, i64 99)\n  ret i64 {}\n",
                eo, TAG_ERR, eo, eo
            ));
            self.code.push_str(&format!("{}:\n", ok_l));
        }

        let rec = self.fresh_tmp();
        self.code.push_str(&format!(
            "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 {})\n",
            rec,
            layout.tag,
            layout.fields.len()
        ));
        for (idx, (fname, _)) in layout.fields.iter().enumerate() {
            let slot = self.var_ptrs[fname].clone();
            let fv = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = load i64, ptr {}\n  call void @tenon_rc_set_field(i64 {}, i64 {}, i64 {})\n",
                fv, slot, rec, idx, fv
            ));
        }
        let ok_obj = self.fresh_tmp();
        self.code.push_str(&format!(
            "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 1)\n  call void @tenon_rc_set_field(i64 {}, i64 0, i64 {})\n  ret i64 {}\n",
            ok_obj, TAG_OK, ok_obj, rec, ok_obj
        ));

        Ok(format!(
            "define i64 @tenon_fn_{}__parse({}) {{\nentry:\n{}{}}}\n",
            sanitize(rname),
            param_list.join(", "),
            self.entry_allocas,
            self.code
        ))
    }

    fn emit_async_fn(&mut self, f: &FnDecl) -> Result<String, String> {
        let mut frame_slots: BTreeMap<String, i64> = BTreeMap::new();
        let mut next_slot = 4i64;
        for p in &f.params {
            frame_slots.insert(p.name.clone(), next_slot);
            next_slot += 1;
        }
        for s in &f.body.stmts {
            if let Stmt::Let { name, .. } = s {
                if !frame_slots.contains_key(name) {
                    frame_slots.insert(name.clone(), next_slot);
                    next_slot += 1;
                }
            }
        }
        let segments = split_async_segments(&f.body, &mut frame_slots, &mut next_slot);
        let total_slots = next_slot;
        let sname = sanitize(&f.name);

        // 1) Constructor `@tenon_fn_<name>(params...) -> frame_ptr`
        let mut ctor_params = Vec::new();
        let mut ctor_body = String::new();
        ctor_body.push_str(&format!(
            "  %pp = ptrtoint ptr @tenon_fn_{}__poll to i64\n  %fr = call i64 @tenon_async_frame_alloc(i64 {}, i64 %pp)\n",
            sname, total_slots
        ));
        for (i, p) in f.params.iter().enumerate() {
            ctor_params.push(format!("i64 %arg_{}", i));
            ctor_body.push_str(&format!(
                "  call void @tenon_rc_set_field(i64 %fr, i64 {}, i64 %arg_{})\n",
                frame_slots[&p.name], i
            ));
        }
        ctor_body.push_str("  ret i64 %fr\n");
        let ctor_ir = format!(
            "define i64 @tenon_fn_{}({}) {{\nentry:\n{}}}\n\n",
            sname,
            ctor_params.join(", "),
            ctor_body
        );

        // 2) Poll state machine `@tenon_fn_<name>__poll(i64 %frame) -> i64`
        for (vname, &slot) in &frame_slots {
            let vp = self.ensure_var_slot(vname);
            let lv = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = call i64 @tenon_rc_get_field(i64 %frame, i64 {})\n  store i64 {}, ptr {}\n",
                lv, slot, lv, vp
            ));
        }
        let cur_st = self.fresh_tmp();
        self.code
            .push_str(&format!("  {} = call i64 @tenon_rc_get_field(i64 %frame, i64 0)\n", cur_st));

        let n_awaits = segments.await_steps.len();
        let mut start_lbls = Vec::new();
        let mut poll_lbls = Vec::new();
        for s in 0..n_awaits {
            start_lbls.push(format!("st_start_{}", s));
            poll_lbls.push(format!("st_poll_{}", s));
        }
        let final_lbl = "st_final".to_string();

        self.code.push_str(&format!("  switch i64 {}, label %{} [\n", cur_st, final_lbl));
        for s in 0..n_awaits {
            self.code.push_str(&format!("    i64 {}, label %{}\n", 2 * s, start_lbls[s]));
            self.code.push_str(&format!("    i64 {}, label %{}\n", 2 * s + 1, poll_lbls[s]));
        }
        self.code.push_str("  ]\n");

        for (s, step) in segments.await_steps.iter().enumerate() {
            self.code.push_str(&format!("{}:\n", start_lbls[s]));
            self.terminated = false;
            let pre_blk = Block {
                stmts: step.pre_stmts.clone(),
                tail: None,
                span: f.span,
            };
            let _ = self.lower_block(&pre_blk)?;
            for (vname, &slot) in &frame_slots {
                let vp = self.var_ptrs[vname].clone();
                let lv = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = load i64, ptr {}\n  call void @tenon_rc_set_field(i64 %frame, i64 {}, i64 {})\n",
                    lv, vp, slot, lv
                ));
            }
            let cframe = self.lower_expr(&step.child_expr)?;
            self.code.push_str(&format!(
                "  call void @tenon_rc_set_field(i64 %frame, i64 3, i64 {})\n  call void @tenon_rc_set_field(i64 %frame, i64 0, i64 {})\n  br label %{}\n",
                cframe,
                2 * s + 1,
                poll_lbls[s]
            ));

            self.code.push_str(&format!("{}:\n", poll_lbls[s]));
            self.terminated = false;
            let cf = self.fresh_tmp();
            let rdy = self.fresh_tmp();
            let cmp = self.fresh_tmp();
            let pend_l = self.fresh_lbl("pend");
            let res_l = self.fresh_lbl("res");
            self.code.push_str(&format!(
                "  {} = call i64 @tenon_rc_get_field(i64 %frame, i64 3)\n  {} = call i64 @tenon_async_poll_frame(i64 {})\n  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n",
                cf, rdy, cf, cmp, rdy, cmp, res_l, pend_l
            ));
            self.code.push_str(&format!("{}:\n  ret i64 0\n", pend_l));
            self.code.push_str(&format!("{}:\n", res_l));
            let cres = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = call i64 @tenon_rc_get_field(i64 {}, i64 2)\n  call void @tenon_rc_dec(i64 {})\n  call void @tenon_rc_set_field(i64 %frame, i64 3, i64 0)\n",
                cres, cf, cf
            ));
            if let Some(bname) = &step.bind_var {
                let slot = frame_slots[bname];
                let vp = self.var_ptrs[bname].clone();
                self.code.push_str(&format!(
                    "  call void @tenon_rc_set_field(i64 %frame, i64 {}, i64 {})\n  store i64 {}, ptr {}\n",
                    slot, cres, cres, vp
                ));
            }
            let next_lbl = if s + 1 < n_awaits {
                start_lbls[s + 1].as_str()
            } else {
                final_lbl.as_str()
            };
            self.code.push_str(&format!(
                "  call void @tenon_rc_set_field(i64 %frame, i64 0, i64 {})\n  br label %{}\n",
                2 * (s + 1),
                next_lbl
            ));
        }

        self.code.push_str(&format!("{}:\n", final_lbl));
        self.terminated = false;
        let final_blk = Block {
            stmts: segments.final_stmts.clone(),
            tail: segments.final_tail.clone().map(Box::new),
            span: f.span,
        };
        let fval = self.lower_block(&final_blk)?;
        self.code.push_str(&format!(
            "  call void @tenon_rc_set_field(i64 %frame, i64 2, i64 {})\n  call void @tenon_rc_set_field(i64 %frame, i64 0, i64 -1)\n  ret i64 1\n",
            fval
        ));

        let poll_ir = format!(
            "define i64 @tenon_fn_{}__poll(i64 %frame) {{\nentry:\n{}{}}}\n",
            sname, self.entry_allocas, self.code
        );
        Ok(format!("{}{}", ctor_ir, poll_ir))
    }

    fn emit_cleanup(&mut self, keep_var: Option<&str>) {
        let to_dec: Vec<String> = self
            .owned_rc_vars
            .iter()
            .filter(|v| keep_var != Some(v.as_str()) && !self.inout_vars.contains(*v))
            .cloned()
            .collect();
        for vname in to_dec {
            if let Some(slot) = self.var_ptrs.get(&vname).cloned() {
                let h = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = load i64, ptr {}\n  call void @tenon_rc_dec(i64 {})\n",
                    h, slot, h
                ));
            }
        }
    }

    fn resolve_field_idx(&self, obj_expr: &Expr, field: &str) -> Result<usize, String> {
        if let ExprKind::Ident(vname) = &obj_expr.kind {
            if let Some(rname) = self.var_records.get(vname) {
                if let Some(layout) = self.records.get(rname) {
                    if let Some(pos) = layout.fields.iter().position(|(f, _)| f == field) {
                        return Ok(pos);
                    }
                }
            }
        }
        self.field_to_idx
            .get(field)
            .copied()
            .ok_or_else(|| format!("unknown field `{}` in LLVM codegen", field))
    }

    fn expr_produces_rc(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::RecordLit { .. } | ExprKind::List(_) | ExprKind::Str(_) => true,
            ExprKind::Ident(s) => {
                s == "None" || self.owned_rc_vars.contains(s) || self.var_records.contains_key(s)
            }
            ExprKind::Call { callee, .. } => match &callee.kind {
                ExprKind::Ident(f) => {
                    matches!(f.as_str(), "Ok" | "Err" | "Some" | "push")
                        || self.enum_tags.contains_key(f)
                        || self.fn_metas.get(f).is_some_and(|m| m.returns_rc)
                }
                ExprKind::Field { field, .. } => field == "parse",
                _ => false,
            },
            _ => false,
        }
    }

    fn is_text_expr(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Str(_) => true,
            ExprKind::Ident(s) => self.text_vars.contains(s),
            _ => false,
        }
    }

    fn lower_block(&mut self, blk: &Block) -> Result<String, String> {
        let mut last = "0".to_string();
        for stmt in &blk.stmts {
            if self.terminated {
                break;
            }
            match stmt {
                Stmt::Let {
                    name, ty, value, ..
                } => {
                    let is_copy_rc = matches!(&value.kind, ExprKind::Ident(src) if self.owned_rc_vars.contains(src) || self.var_records.contains_key(src));
                    let mut v = self.lower_expr(value)?;
                    if is_copy_rc {
                        let inc = self.fresh_tmp();
                        self.code
                            .push_str(&format!("  {} = call i64 @tenon_rc_inc(i64 {})\n", inc, v));
                        v = inc;
                    }
                    let slot = self.ensure_var_slot(name);
                    self.code
                        .push_str(&format!("  store i64 {}, ptr {}\n", v, slot));

                    if let Some(t) = ty {
                        let tname = t.display_name();
                        if self.records.contains_key(&tname) {
                            self.var_records.insert(name.clone(), tname);
                            self.owned_rc_vars.insert(name.clone());
                        } else if tname == "Text" {
                            self.text_vars.insert(name.clone());
                            self.owned_rc_vars.insert(name.clone());
                        } else if is_rc_type(t, self.records) {
                            self.owned_rc_vars.insert(name.clone());
                        }
                    } else {
                        if let ExprKind::RecordLit { name: rname, .. } = &value.kind {
                            self.var_records.insert(name.clone(), rname.clone());
                        } else if let ExprKind::Ident(src) = &value.kind {
                            if let Some(rname) = self.var_records.get(src).cloned() {
                                self.var_records.insert(name.clone(), rname);
                            }
                            if self.text_vars.contains(src) {
                                self.text_vars.insert(name.clone());
                            }
                        } else if matches!(&value.kind, ExprKind::Str(_)) {
                            self.text_vars.insert(name.clone());
                        }
                        if self.expr_produces_rc(value) {
                            self.owned_rc_vars.insert(name.clone());
                        }
                    }
                }
                Stmt::Assign {
                    target, op, value, ..
                } => match &target.kind {
                    ExprKind::Ident(name) => {
                        let consuming_push = matches!(
                            &value.kind,
                            ExprKind::Call { callee, args }
                            if matches!(&callee.kind, ExprKind::Ident(f) if f == "push")
                                && args.first().is_some_and(|a| matches!(&a.value.kind, ExprKind::Ident(s) if s == name))
                        );
                        let rhs = self.lower_expr(value)?;
                        let slot = self.ensure_var_slot(name);
                        if self.owned_rc_vars.contains(name) && !consuming_push {
                            let old = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = load i64, ptr {}\n  call void @tenon_rc_dec(i64 {})\n",
                                old, slot, old
                            ));
                        }
                        let new_v = match op {
                            AssignOp::Set => rhs,
                            AssignOp::AddSet => {
                                let cur = self.fresh_tmp();
                                let res = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = load i64, ptr {}\n  {} = add i64 {}, {}\n",
                                    cur, slot, res, cur, rhs
                                ));
                                res
                            }
                            AssignOp::SubSet => {
                                let cur = self.fresh_tmp();
                                let res = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = load i64, ptr {}\n  {} = sub i64 {}, {}\n",
                                    cur, slot, res, cur, rhs
                                ));
                                res
                            }
                        };
                        self.code
                            .push_str(&format!("  store i64 {}, ptr {}\n", new_v, slot));
                    }
                    ExprKind::Field { target: obj, field } => {
                        if let ExprKind::Ident(vname) = &obj.kind {
                            let fidx = self.resolve_field_idx(obj, field)?;
                            let rhs = self.lower_expr(value)?;
                            let slot = self.var_ptrs[vname].clone();
                            let is_inout = self.inout_vars.contains(vname);
                            let cur_h = if is_inout {
                                let addr_i = self.fresh_tmp();
                                let addr_p = self.fresh_tmp();
                                let val = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  {} = load i64, ptr {}\n",
                                    addr_i, slot, addr_p, addr_i, val, addr_p
                                ));
                                val
                            } else {
                                let val = self.fresh_tmp();
                                self.code
                                    .push_str(&format!("  {} = load i64, ptr {}\n", val, slot));
                                val
                            };
                            let uniq = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = call i64 @tenon_rc_ensure_unique(i64 {})\n",
                                uniq, cur_h
                            ));
                            if is_inout {
                                let addr_i = self.fresh_tmp();
                                let addr_p = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                    addr_i, slot, addr_p, addr_i, uniq, addr_p
                                ));
                            } else {
                                self.code
                                    .push_str(&format!("  store i64 {}, ptr {}\n", uniq, slot));
                            }
                            let new_fv = match op {
                                AssignOp::Set => rhs,
                                AssignOp::AddSet => {
                                    let cf = self.fresh_tmp();
                                    let res = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = call i64 @tenon_rc_get_field(i64 {}, i64 {})\n  {} = add i64 {}, {}\n",
                                        cf, uniq, fidx, res, cf, rhs
                                    ));
                                    res
                                }
                                AssignOp::SubSet => {
                                    let cf = self.fresh_tmp();
                                    let res = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = call i64 @tenon_rc_get_field(i64 {}, i64 {})\n  {} = sub i64 {}, {}\n",
                                        cf, uniq, fidx, res, cf, rhs
                                    ));
                                    res
                                }
                            };
                            self.code.push_str(&format!(
                                "  call void @tenon_rc_set_field(i64 {}, i64 {}, i64 {})\n",
                                uniq, fidx, new_fv
                            ));
                        }
                    }
                    _ => {}
                },
                Stmt::Mutate { body, .. } => {
                    let _ = self.lower_block(body)?;
                }
                Stmt::Return { value, .. } => {
                    let keep = value.as_ref().and_then(|e| match &e.kind {
                        ExprKind::Ident(s) => Some(s.clone()),
                        _ => None,
                    });
                    let v = match value {
                        Some(e) => self.lower_expr(e)?,
                        None => "0".into(),
                    };
                    self.emit_cleanup(keep.as_deref());
                    self.code.push_str(&format!("  ret i64 {}\n", v));
                    self.terminated = true;
                    return Ok(v);
                }
                Stmt::Assert { cond, span } => {
                    let c = self.lower_expr(cond)?;
                    let ok = self.fresh_tmp();
                    let fl = self.fresh_lbl("afail");
                    let cl = self.fresh_lbl("acont");
                    self.code.push_str(&format!(
                        "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n  call void @tenon_rt_assert_fail(i64 {})\n  br label %{}\n{}:\n",
                        ok, c, ok, cl, fl, fl, span.line, cl, cl
                    ));
                }
                Stmt::While { cond, body, .. } => {
                    let hdr = self.fresh_lbl("whdr");
                    let lp = self.fresh_lbl("wbody");
                    let ex = self.fresh_lbl("wexit");
                    self.code.push_str(&format!("  br label %{}\n{}:\n", hdr, hdr));
                    let c = self.lower_expr(cond)?;
                    let cmp = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n",
                        cmp, c, cmp, lp, ex, lp
                    ));
                    let _ = self.lower_block(body)?;
                    if !self.terminated {
                        self.code.push_str(&format!("  br label %{}\n", hdr));
                    }
                    self.terminated = false;
                    self.code.push_str(&format!("{}:\n", ex));
                }
                Stmt::Expr { expr, .. } => {
                    last = self.lower_expr(expr)?;
                }
            }
        }
        if !self.terminated {
            if let Some(tail) = &blk.tail {
                last = self.lower_expr(tail)?;
            }
        }
        Ok(last)
    }

    fn lower_expr(&mut self, expr: &Expr) -> Result<String, String> {
        match &expr.kind {
            ExprKind::Int(n) => Ok(n.to_string()),
            ExprKind::Bool(b) => Ok(if *b { "1" } else { "0" }.into()),
            ExprKind::Str(s) => {
                let gname = format!(".str.{}", self.string_globals.len());
                self.string_globals.push((gname.clone(), s.clone()));
                let pi = self.fresh_tmp();
                let obj = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = ptrtoint ptr @{} to i64\n  {} = call i64 @tenon_text_new(i64 {}, i64 {})\n",
                    pi, gname, obj, pi, s.len()
                ));
                Ok(obj)
            }
            ExprKind::Ident(s) => {
                if s == "self" {
                    if let Some(sv) = self.self_var.clone() {
                        let slot = self.var_ptrs[&sv].clone();
                        let v = self.fresh_tmp();
                        self.code
                            .push_str(&format!("  {} = load i64, ptr {}\n", v, slot));
                        return Ok(v);
                    }
                }
                if s == "None" {
                    let o = self.fresh_tmp();
                    self.code
                        .push_str(&format!("  {} = call i64 @tenon_rc_alloc(i64 {}, i64 0)\n", o, TAG_NONE));
                    return Ok(o);
                }
                if let Some(&tag) = self.enum_tags.get(s) {
                    if !self.var_ptrs.contains_key(s) {
                        let o = self.fresh_tmp();
                        self.code
                            .push_str(&format!("  {} = call i64 @tenon_rc_alloc(i64 {}, i64 0)\n", o, tag));
                        return Ok(o);
                    }
                }
                let slot = self
                    .var_ptrs
                    .get(s)
                    .cloned()
                    .ok_or_else(|| format!("undefined variable `{}` in LLVM codegen", s))?;
                if self.inout_vars.contains(s) {
                    let ai = self.fresh_tmp();
                    let ap = self.fresh_tmp();
                    let v = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  {} = load i64, ptr {}\n",
                        ai, slot, ap, ai, v, ap
                    ));
                    Ok(v)
                } else {
                    let v = self.fresh_tmp();
                    self.code
                        .push_str(&format!("  {} = load i64, ptr {}\n", v, slot));
                    Ok(v)
                }
            }
            ExprKind::RecordLit { name, fields, .. } => {
                let layout = self
                    .records
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown record `{}`", name))?;
                let obj = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 {})\n",
                    obj,
                    layout.tag,
                    layout.fields.len()
                ));
                for (fname, fexpr) in fields {
                    let fv = self.lower_expr(fexpr)?;
                    if let Some(idx) = layout.fields.iter().position(|(f, _)| f == fname) {
                        self.code.push_str(&format!(
                            "  call void @tenon_rc_set_field(i64 {}, i64 {}, i64 {})\n",
                            obj, idx, fv
                        ));
                    }
                }
                Ok(obj)
            }
            ExprKind::List(elems) => {
                let obj = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 {})\n",
                    obj,
                    TAG_LIST,
                    elems.len()
                ));
                for (idx, e) in elems.iter().enumerate() {
                    let ev = self.lower_expr(e)?;
                    self.code.push_str(&format!(
                        "  call void @tenon_rc_set_field(i64 {}, i64 {}, i64 {})\n",
                        obj, idx, ev
                    ));
                }
                Ok(obj)
            }
            ExprKind::Index { target, index } => {
                let t = self.lower_expr(target)?;
                let i = self.lower_expr(index)?;
                let res = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @tenon_rc_get_field(i64 {}, i64 {})\n",
                    res, t, i
                ));
                Ok(res)
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    let key = format!("{}.{}", ename, field);
                    if let Some(&tag) = self.enum_tags.get(&key) {
                        let o = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 0)\n",
                            o, tag
                        ));
                        return Ok(o);
                    }
                }
                let idx = self.resolve_field_idx(target, field)?;
                let obj = self.lower_expr(target)?;
                let res = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @tenon_rc_get_field(i64 {}, i64 {})\n",
                    res, obj, idx
                ));
                Ok(res)
            }
            ExprKind::Unary { op, expr } => {
                let v = self.lower_expr(expr)?;
                let r = self.fresh_tmp();
                match op {
                    UnOp::Neg => self.code.push_str(&format!("  {} = sub i64 0, {}\n", r, v)),
                    UnOp::Not => {
                        let c = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = icmp eq i64 {}, 0\n  {} = zext i1 {} to i64\n",
                            c, v, r, c
                        ));
                    }
                }
                Ok(r)
            }
            ExprKind::Binary { op, left, right } => {
                let tcmp = self.is_text_expr(left) || self.is_text_expr(right);
                let l = self.lower_expr(left)?;
                let r = self.lower_expr(right)?;
                if tcmp && matches!(op, BinOp::Eq | BinOp::Ne) {
                    let eq = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @tenon_text_eq(i64 {}, i64 {})\n",
                        eq, l, r
                    ));
                    if *op == BinOp::Eq {
                        return Ok(eq);
                    }
                    let c = self.fresh_tmp();
                    let res = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = icmp eq i64 {}, 0\n  {} = zext i1 {} to i64\n",
                        c, eq, res, c
                    ));
                    return Ok(res);
                }
                let res = self.fresh_tmp();
                match op {
                    BinOp::Add => self.code.push_str(&format!("  {} = add i64 {}, {}\n", res, l, r)),
                    BinOp::Sub => self.code.push_str(&format!("  {} = sub i64 {}, {}\n", res, l, r)),
                    BinOp::Mul => self.code.push_str(&format!("  {} = mul i64 {}, {}\n", res, l, r)),
                    BinOp::Div => self.code.push_str(&format!("  {} = sdiv i64 {}, {}\n", res, l, r)),
                    BinOp::Mod => self.code.push_str(&format!("  {} = srem i64 {}, {}\n", res, l, r)),
                    BinOp::BitAnd => self.code.push_str(&format!("  {} = and i64 {}, {}\n", res, l, r)),
                    BinOp::BitOr => self.code.push_str(&format!("  {} = or i64 {}, {}\n", res, l, r)),
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        let pred = match op {
                            BinOp::Eq => "eq",
                            BinOp::Ne => "ne",
                            BinOp::Lt => "slt",
                            BinOp::Le => "sle",
                            BinOp::Gt => "sgt",
                            BinOp::Ge => "sge",
                            _ => unreachable!(),
                        };
                        let c = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = icmp {} i64 {}, {}\n  {} = zext i1 {} to i64\n",
                            c, pred, l, r, res, c
                        ));
                    }
                    BinOp::And | BinOp::Or => {
                        let lb = self.fresh_tmp();
                        let rb = self.fresh_tmp();
                        let comb = self.fresh_tmp();
                        let iop = if *op == BinOp::And { "and" } else { "or" };
                        self.code.push_str(&format!(
                            "  {} = icmp ne i64 {}, 0\n  {} = icmp ne i64 {}, 0\n  {} = {} i1 {}, {}\n  {} = zext i1 {} to i64\n",
                            lb, l, rb, r, comb, iop, lb, rb, res, comb
                        ));
                    }
                    other => return Err(format!("unsupported op {:?} in LLVM codegen", other)),
                }
                Ok(res)
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                let c = self.lower_expr(cond)?;
                let cmp = self.fresh_tmp();
                let res_slot = format!("%ifres_{}", self.lbl + 1);
                self.entry_allocas.push_str(&format!(
                    "  {} = alloca i64\n  store i64 0, ptr {}\n",
                    res_slot, res_slot
                ));
                let tl = self.fresh_lbl("then");
                let el = self.fresh_lbl("else");
                let ml = self.fresh_lbl("endif");
                self.code.push_str(&format!(
                    "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n",
                    cmp, c, cmp, tl, el, tl
                ));
                let tv = self.lower_block(then_block)?;
                if !self.terminated {
                    self.code.push_str(&format!(
                        "  store i64 {}, ptr {}\n  br label %{}\n",
                        tv, res_slot, ml
                    ));
                }
                self.terminated = false;
                self.code.push_str(&format!("{}:\n", el));
                let ev = match else_branch {
                    Some(eb) => self.lower_expr(eb)?,
                    None => "0".into(),
                };
                if !self.terminated {
                    self.code.push_str(&format!(
                        "  store i64 {}, ptr {}\n  br label %{}\n",
                        ev, res_slot, ml
                    ));
                }
                self.terminated = false;
                self.code.push_str(&format!("{}:\n", ml));
                let out = self.fresh_tmp();
                self.code
                    .push_str(&format!("  {} = load i64, ptr {}\n", out, res_slot));
                Ok(out)
            }
            ExprKind::Match { scrutinee, arms } => {
                let is_temp_parse = matches!(
                    &scrutinee.kind,
                    ExprKind::Call { callee, .. }
                    if matches!(&callee.kind, ExprKind::Field { field, .. } if field == "parse")
                );
                let parse_rname = match &scrutinee.kind {
                    ExprKind::Call { callee, .. } => match &callee.kind {
                        ExprKind::Field { target, field } if field == "parse" => {
                            match &target.kind {
                                ExprKind::Ident(rn) => Some(rn.clone()),
                                _ => None,
                            }
                        }
                        _ => None,
                    },
                    _ => None,
                };
                let is_variant = arms
                    .iter()
                    .any(|a| a.patterns.iter().any(|p| matches!(p, Pattern::Variant { .. })));

                let sv = self.lower_expr(scrutinee)?;
                let cmp_val = if is_variant {
                    let tg = self.fresh_tmp();
                    self.code
                        .push_str(&format!("  {} = call i64 @tenon_rc_get_tag(i64 {})\n", tg, sv));
                    tg
                } else {
                    sv.clone()
                };

                let res_slot = format!("%mres_{}", self.lbl + 1);
                self.entry_allocas.push_str(&format!(
                    "  {} = alloca i64\n  store i64 0, ptr {}\n",
                    res_slot, res_slot
                ));
                let end_l = self.fresh_lbl("mend");

                for arm in arms {
                    let is_wild = arm
                        .patterns
                        .iter()
                        .any(|p| matches!(p, Pattern::Wildcard(_)));
                    if is_wild {
                        let v = self.lower_expr(&arm.body)?;
                        if is_temp_parse {
                            self.code.push_str(&format!("  call void @tenon_rc_dec(i64 {})\n", sv));
                        }
                        self.code.push_str(&format!(
                            "  store i64 {}, ptr {}\n  br label %{}\n",
                            v, res_slot, end_l
                        ));
                        self.code.push_str(&format!("{}:\n", end_l));
                        let out = self.fresh_tmp();
                        self.code
                            .push_str(&format!("  {} = load i64, ptr {}\n", out, res_slot));
                        return Ok(out);
                    }

                    let mut cond_var: Option<String> = None;
                    let mut vbind: Option<(String, bool)> = None;
                    for p in &arm.patterns {
                        let pval = match p {
                            Pattern::Int(n, _) => *n,
                            Pattern::Bool(b, _) => if *b { 1 } else { 0 },
                            Pattern::Variant { name, binding, .. } => {
                                let t = *self.enum_tags.get(name).ok_or_else(|| {
                                    format!("unknown variant `{}` in LLVM match", name)
                                })?;
                                if let Some(bn) = binding {
                                    vbind = Some((bn.clone(), name == "Ok"));
                                }
                                t
                            }
                            _ => continue,
                        };
                        let eq = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = icmp eq i64 {}, {}\n",
                            eq, cmp_val, pval
                        ));
                        cond_var = Some(match cond_var {
                            Some(prev) => {
                                let o = self.fresh_tmp();
                                self.code
                                    .push_str(&format!("  {} = or i1 {}, {}\n", o, prev, eq));
                                o
                            }
                            None => eq,
                        });
                    }
                    let mat_l = self.fresh_lbl("marm");
                    let nxt_l = self.fresh_lbl("mnxt");
                    let c = cond_var.unwrap_or_else(|| "false".into());
                    self.code.push_str(&format!(
                        "  br i1 {}, label %{}, label %{}\n{}:\n",
                        c, mat_l, nxt_l, mat_l
                    ));
                    let mut inner_rec: Option<String> = None;
                    if let Some((bn, is_ok)) = vbind {
                        let pay = self.fresh_tmp();
                        let bslot = self.ensure_var_slot(&bn);
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_rc_get_field(i64 {}, i64 0)\n  store i64 {}, ptr {}\n",
                            pay, sv, pay, bslot
                        ));
                        if is_ok {
                            if let Some(rn) = &parse_rname {
                                self.var_records.insert(bn, rn.clone());
                                inner_rec = Some(pay);
                            }
                        }
                    }
                    let av = self.lower_expr(&arm.body)?;
                    if is_temp_parse {
                        if let Some(ir) = inner_rec {
                            if !self.expr_produces_rc(&arm.body) {
                                self.code
                                    .push_str(&format!("  call void @tenon_rc_dec(i64 {})\n", ir));
                            }
                        }
                        self.code
                            .push_str(&format!("  call void @tenon_rc_dec(i64 {})\n", sv));
                    }
                    self.code.push_str(&format!(
                        "  store i64 {}, ptr {}\n  br label %{}\n{}:\n",
                        av, res_slot, end_l, nxt_l
                    ));
                }

                if is_temp_parse {
                    self.code
                        .push_str(&format!("  call void @tenon_rc_dec(i64 {})\n", sv));
                }
                self.code
                    .push_str(&format!("  br label %{}\n{}:\n", end_l, end_l));
                let out = self.fresh_tmp();
                self.code
                    .push_str(&format!("  {} = load i64, ptr {}\n", out, res_slot));
                Ok(out)
            }
            ExprKind::Block(blk) => self.lower_block(blk),
            ExprKind::Call { callee, args } => match &callee.kind {
                ExprKind::Field { target, field } if field == "parse" => {
                    if let ExprKind::Ident(rname) = &target.kind {
                        let mut avals = Vec::new();
                        for a in args {
                            avals.push(format!("i64 {}", self.lower_expr(&a.value)?));
                        }
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_fn_{}__parse({})\n",
                            res,
                            sanitize(rname),
                            avals.join(", ")
                        ));
                        return Ok(res);
                    }
                    Err("unsupported `.parse()` target in LLVM".into())
                }
                ExprKind::Ident(fname) => {
                    if fname == "print_int" && args.len() == 1 {
                        let v = self.lower_expr(&args[0].value)?;
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_rt_print_int(i64 {})\n",
                            res, v
                        ));
                        return Ok(res);
                    }
                    if fname == "len" && args.len() == 1 {
                        let v = self.lower_expr(&args[0].value)?;
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_rc_get_len(i64 {})\n",
                            res, v
                        ));
                        return Ok(res);
                    }
                    if fname == "push" && args.len() == 2 {
                        let xs = self.lower_expr(&args[0].value)?;
                        let val = self.lower_expr(&args[1].value)?;
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_list_push(i64 {}, i64 {})\n",
                            res, xs, val
                        ));
                        return Ok(res);
                    }
                    if fname == "yield_val" && args.len() == 1 {
                        let v = self.lower_expr(&args[0].value)?;
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_async_yield_create(i64 {})\n",
                            res, v
                        ));
                        return Ok(res);
                    }
                    if fname == "block_on" && args.len() == 1 {
                        let f = self.lower_expr(&args[0].value)?;
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_async_block_on(i64 {})\n",
                            res, f
                        ));
                        return Ok(res);
                    }
                    if let Some(&tag) = self.enum_tags.get(fname) {
                        if !self.fn_metas.contains_key(fname) {
                            let o = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 {})\n",
                                o,
                                tag,
                                args.len()
                            ));
                            for (i, a) in args.iter().enumerate() {
                                let av = self.lower_expr(&a.value)?;
                                self.code.push_str(&format!(
                                    "  call void @tenon_rc_set_field(i64 {}, i64 {}, i64 {})\n",
                                    o, i, av
                                ));
                            }
                            return Ok(o);
                        }
                    }

                    let meta = self.fn_metas.get(fname).cloned();
                    let mut avals = Vec::new();
                    let mut writebacks: Vec<(String, String)> = Vec::new();
                    for (idx, a) in args.iter().enumerate() {
                        let is_inout = meta
                            .as_ref()
                            .and_then(|m| m.params_inout.get(idx).copied())
                            .unwrap_or(false);
                        if is_inout {
                            if let ExprKind::Ident(vn) = &a.value.kind {
                                let cur = self.lower_expr(&a.value)?;
                                let tmp_slot = format!("%io_{}_{}", self.tmp + 1, idx);
                                self.tmp += 1;
                                self.entry_allocas
                                    .push_str(&format!("  {} = alloca i64\n", tmp_slot));
                                let addr_i = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  store i64 {}, ptr {}\n  {} = ptrtoint ptr {} to i64\n",
                                    cur, tmp_slot, addr_i, tmp_slot
                                ));
                                avals.push(format!("i64 {}", addr_i));
                                writebacks.push((vn.clone(), tmp_slot));
                            } else {
                                return Err("inout arg must be variable".into());
                            }
                        } else {
                            let av = self.lower_expr(&a.value)?;
                            avals.push(format!("i64 {}", av));
                        }
                    }
                    let res = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @tenon_fn_{}({})\n",
                        res,
                        sanitize(fname),
                        avals.join(", ")
                    ));
                    for (vn, tmp_slot) in writebacks {
                        let updated = self.fresh_tmp();
                        let vslot = self.var_ptrs[&vn].clone();
                        self.code
                            .push_str(&format!("  {} = load i64, ptr {}\n", updated, tmp_slot));
                        if self.inout_vars.contains(&vn) {
                            let ai = self.fresh_tmp();
                            let ap = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                ai, vslot, ap, ai, updated, ap
                            ));
                        } else {
                            self.code
                                .push_str(&format!("  store i64 {}, ptr {}\n", updated, vslot));
                        }
                    }
                    Ok(res)
                }
                _ => Err("unsupported call target in LLVM codegen".into()),
            },
            _ => Ok("0".into()),
        }
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenon_syntax::parse;

    #[test]
    fn llvm_aot_compiles_and_runs_records_mutate_parse_and_stackless_async() {
        let src = r#"
pub type Year  = Int where 1 <= self && self <= 9999
pub type Month = Int where 1 <= self && self <= 12

pub pure fn is_leap_year(y Int) Bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

pub pure fn days_in_month(y Year, m Month) Int {
    match m {
        2 => if is_leap_year(y) { 29 } else { 28 }
        4 | 6 | 9 | 11 => 30
        _ => 31
    }
}

pub record Date {
    year  Year
    month Month
    day   Int where 1 <= self && self <= days_in_month(year, month)
}

pub fn shift_prev(inout d Date) {
    mutate d {
        d.year -= 1
        if d.month == 2 && d.day == 29 && !is_leap_year(d.year) {
            d.day = 28
        }
    }
}

pub async fn async_add(v Int) Int {
    let a = await yield_val(v + 5)
    let b = await yield_val(a + 7)
    b
}

pub fn main() Int {
    var d Date = Date { year = 2024, month = 2, day = 29 }
    shift_prev(d)
    assert d.year == 2023
    assert d.day == 28

    let ok_day = match Date.parse(2024, 2, 29) {
        Ok(dt) => dt.day
        Err(c) => -c
    }
    let bad_m = match Date.parse(2024, 13, 1) {
        Ok(dt) => dt.day
        Err(c) => -c
    }
    assert ok_day == 29
    assert bad_m == -2

    let ar = block_on(async_add(10))
    assert ar == 22
    ok_day + d.day + ar
}
"#;
        let prog = parse(src).unwrap();
        let bin = std::env::temp_dir().join(format!("tenon_llvm_aot_test_{}", std::process::id()));
        let art = compile_to_native_binary(&prog, &bin, None, "O3").unwrap();
        assert!(art.binary_bytes > 1024);

        let run = Command::new(&bin).output().unwrap();
        let _ = fs::remove_file(&bin);
        let stdout = String::from_utf8_lossy(&run.stdout);
        assert!(
            run.status.success(),
            "native LLVM binary failed: stdout={} stderr={}",
            stdout,
            String::from_utf8_lossy(&run.stderr)
        );
        assert!(stdout.contains("main() => 79"), "unexpected stdout: {}", stdout);
        assert!(stdout.contains("live_objects=0"), "leaked objects: {}", stdout);
        assert!(stdout.contains("async_suspends=2"), "missing suspensions: {}", stdout);
    }
}
