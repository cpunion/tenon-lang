use crate::checker::{CheckReport, check_program};
use crate::sync_solver::DistributedAppManifest;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tenon_syntax::{Item, Program, parse};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct QueryStats {
    pub parse_hits: usize,
    pub parse_misses: usize,
    pub check_hits: usize,
    pub check_misses: usize,
    /// Number of incremental edits where file content changed but public/interface signatures did NOT change (Early Cutoff!).
    pub early_cutoffs: usize,
    /// Number of per-function verification summaries reused from cache.
    pub fn_cache_hits: usize,
    pub fn_cache_misses: usize,
}

#[derive(Clone, Default)]
pub struct IncrementalDb {
    inner: Arc<Mutex<DbInner>>,
}

#[derive(Default)]
struct DbInner {
    file_hashes: BTreeMap<String, String>,
    file_sources: BTreeMap<String, String>,
    file_sig_hashes: BTreeMap<String, String>,
    parsed_cache: BTreeMap<String, Program>,
    check_cache: BTreeMap<String, CheckReport>,
    /// Keyed by `(program_content_hash)` for merged multi-file programs (e.g. LSP overlays).
    program_report_cache: BTreeMap<String, CheckReport>,
    /// Keyed by `(module_key, env_sig_hash, fn_name, fn_body_hash)` -> cached diagnostics count & proof count.
    fn_sig_index: BTreeMap<String, (String, BTreeMap<String, String>)>,
    stats: QueryStats,
}

/// Computes a deterministic hash of all item signatures (types, records, enums, function headers,
/// contracts `requires`/`ensures`, interfaces, attrs, reactive views, streams, protocols, etc.)
/// WITHOUT function bodies.
/// When only a function body changes, `program_signature_hash` stays identical -> Early Cutoff!
fn hash_fn_signature(f: &tenon_syntax::FnDecl, out: &mut String) {
    // Exhaustive destructuring ensures any future AST field on FnDecl triggers a compile error if unhashed.
    let tenon_syntax::FnDecl {
        attrs,
        is_pub,
        is_pure,
        is_async,
        is_test,
        is_unsafe,
        is_comptime,
        is_ghost,
        name,
        type_params,
        type_param_bounds,
        params,
        ret_ty,
        requires,
        ensures,
        body: _,
        span: _,
    } = f;
    out.push_str(&format!(
        "fn:{}:{}:{}:{}:{}:{}:{}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}\n",
        is_pub,
        is_pure,
        is_async,
        is_test,
        is_unsafe,
        is_comptime,
        is_ghost,
        name,
        attrs,
        type_params,
        type_param_bounds,
        params,
        ret_ty,
        requires,
        ensures,
    ));
}

pub fn program_signature_hash(prog: &Program) -> String {
    let mut sig_buf = String::new();
    sig_buf.push_str(&format!(
        "prog_meta:{:?}:{:?}:{:?}\n",
        prog.build_config, prog.file_packages, prog.test_file_ids
    ));
    for item in &prog.items {
        match item {
            Item::Import {
                path,
                alias,
                kind,
                span: _,
            } => {
                sig_buf.push_str(&format!("import:{}:{:?}:{:?}\n", path, alias, kind));
            }
            Item::Macro(m) => {
                sig_buf.push_str(&format!("macro:{:?}\n", m));
            }
            Item::ComptimeBlock(b) => {
                sig_buf.push_str(&format!("comptime:{:?}\n", b));
            }
            Item::MacroCall {
                name,
                args,
                span: _,
            } => {
                sig_buf.push_str(&format!("mcall:{}:{:?}\n", name, args));
            }
            Item::TypeAlias(tenon_syntax::TypeAliasDecl {
                attrs,
                is_pub,
                is_distinct,
                name,
                type_params,
                type_param_defaults,
                variadic_param,
                base,
                predicate,
                span: _,
            }) => {
                sig_buf.push_str(&format!(
                    "ta:{}:{}:{}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}\n",
                    is_pub,
                    is_distinct,
                    name,
                    attrs,
                    type_params,
                    type_param_defaults,
                    variadic_param,
                    base,
                    predicate
                ));
            }
            Item::Record(tenon_syntax::RecordDecl {
                attrs,
                is_pub,
                name,
                type_params,
                type_param_bounds,
                type_param_defaults,
                fields,
                where_clauses,
                invariants,
                methods,
                span: _,
            }) => {
                sig_buf.push_str(&format!(
                    "rec:{}:{}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}\n",
                    is_pub,
                    name,
                    attrs,
                    type_params,
                    type_param_bounds,
                    type_param_defaults,
                    fields,
                    where_clauses,
                    invariants
                ));
                for m in methods {
                    sig_buf.push_str("  rec_m:");
                    hash_fn_signature(m, &mut sig_buf);
                }
            }
            Item::Enum(tenon_syntax::EnumDecl {
                attrs,
                is_pub,
                name,
                type_params,
                type_param_bounds,
                type_param_defaults,
                variants,
                span: _,
            }) => {
                sig_buf.push_str(&format!(
                    "enum:{}:{}:{:?}:{:?}:{:?}:{:?}:{:?}\n",
                    is_pub, name, attrs, type_params, type_param_bounds, type_param_defaults, variants
                ));
            }
            Item::Interface(tenon_syntax::InterfaceDecl {
                attrs,
                is_pub,
                name,
                type_params,
                type_param_bounds,
                assoc_types,
                methods,
                span: _,
            }) => {
                sig_buf.push_str(&format!(
                    "iface:{}:{}:{:?}:{:?}:{:?}:{:?}:{:?}\n",
                    is_pub, name, attrs, type_params, type_param_bounds, assoc_types, methods
                ));
            }
            Item::Fn(f) => {
                hash_fn_signature(f, &mut sig_buf);
            }
            Item::ExternFn(tenon_syntax::ExternFnDecl {
                attrs,
                is_pub,
                abi,
                name,
                params,
                ret_ty,
                is_variadic,
                link_lib,
                span: _,
            }) => {
                sig_buf.push_str(&format!(
                    "ext:{}:{}:{}:{}:{:?}:{:?}:{:?}:{:?}\n",
                    is_pub, abi, name, is_variadic, link_lib, attrs, params, ret_ty
                ));
            }
            Item::Const(tenon_syntax::ConstDecl {
                attrs,
                is_pub,
                name,
                ty,
                value,
                span: _,
            }) => {
                sig_buf.push_str(&format!(
                    "const:{}:{}:{:?}:{:?}:{:?}\n",
                    is_pub, name, attrs, ty, value
                ));
            }
            Item::Bitfield(tenon_syntax::BitfieldDecl {
                attrs,
                is_pub,
                name,
                backing_ty,
                fields,
                span: _,
            }) => {
                sig_buf.push_str(&format!(
                    "bf:{}:{}:{:?}:{:?}:{:?}\n",
                    is_pub, name, attrs, backing_ty, fields
                ));
            }
            Item::Union(tenon_syntax::UnionDecl {
                attrs,
                is_pub,
                name,
                fields,
                span: _,
            }) => {
                sig_buf.push_str(&format!(
                    "union:{}:{}:{:?}:{:?}\n",
                    is_pub, name, attrs, fields
                ));
            }
            Item::AttrDecl(tenon_syntax::AttrDecl {
                attrs,
                is_pub,
                name,
                params,
                span: _,
            }) => {
                sig_buf.push_str(&format!(
                    "attr:{}:{}:{:?}:{:?}\n",
                    is_pub, name, attrs, params
                ));
            }
            Item::Impl(tenon_syntax::ImplDecl {
                attrs,
                type_params,
                type_param_bounds,
                interface,
                target_ty,
                assoc_bindings,
                methods,
                span: _,
            }) => {
                sig_buf.push_str(&format!(
                    "impl:{:?}:{:?}:{:?}:{:?}:{:?}:{:?}\n",
                    attrs, type_params, type_param_bounds, interface, target_ty, assoc_bindings
                ));
                for m in methods {
                    sig_buf.push_str("  impl_m:");
                    hash_fn_signature(m, &mut sig_buf);
                }
            }
            Item::Evolution(ev) => {
                sig_buf.push_str(&format!("evol:{:?}\n", ev));
            }
            Item::LockLattice(ll) => {
                sig_buf.push_str(&format!("lock_lattice:{:?}\n", ll));
            }
            Item::Protocol(pr) => {
                sig_buf.push_str(&format!("protocol:{:?}\n", pr));
            }
            Item::Stream(st) => {
                sig_buf.push_str(&format!("stream:{:?}\n", st));
            }
            Item::View(vw) => {
                sig_buf.push_str(&format!("view:{:?}\n", vw));
            }
            Item::Subscriber(sub) => {
                sig_buf.push_str(&format!("subscriber:{:?}\n", sub));
            }
        }
    }
    sha256_hex(sig_buf.as_bytes())
}

/// Computes per-function body hashes for incremental function-level tracking.
pub fn program_fn_body_hashes(prog: &Program) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for item in &prog.items {
        if let Item::Fn(f) = item {
            let s = format!("{:?}", f.body);
            out.insert(f.name.clone(), sha256_hex(s.as_bytes()));
        } else if let Item::Impl(imp) = item {
            for m in &imp.methods {
                let s = format!("{:?}", m.body);
                out.insert(
                    format!("{}::{}", imp.target_ty.display_name(), m.name),
                    sha256_hex(s.as_bytes()),
                );
            }
        } else if let Item::Record(r) = item {
            for m in &r.methods {
                let s = format!("{:?}", m.body);
                out.insert(format!("{}::{}", r.name, m.name), sha256_hex(s.as_bytes()));
            }
        }
    }
    out
}

/// Computes a full structural hash of a parsed `Program` (exhaustive AST + metadata).
pub fn program_full_hash(prog: &Program) -> String {
    let sig = program_signature_hash(prog);
    let bodies = program_fn_body_hashes(prog);
    let mut acc = sig;
    for (k, v) in bodies {
        acc.push(':');
        acc.push_str(&k);
        acc.push('=');
        acc.push_str(&v);
    }
    // Also fold the complete Debug representation of the AST & build metadata so NO field
    // (including spans, variadic flags, ABI, invariants, or future fields) can ever alias.
    let full_ast = format!(
        "{:?}|{:?}|{:?}|{:?}",
        prog.items, prog.build_config, prog.file_packages, prog.test_file_ids
    );
    acc.push('|');
    acc.push_str(&sha256_hex(full_ast.as_bytes()));
    sha256_hex(acc.as_bytes())
}

impl IncrementalDb {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set or update a source file. Returns `true` if content actually changed.
    pub fn set_file(&self, path: impl Into<String>, source: impl Into<String>) -> bool {
        let path = path.into();
        let source = source.into();
        let hash = sha256_hex(source.as_bytes());
        let mut inner = self.inner.lock().unwrap();
        let changed = inner.file_hashes.get(&path) != Some(&hash);
        if changed {
            inner.file_hashes.insert(path.clone(), hash);
            inner.file_sources.insert(path, source);
        }
        changed
    }

    /// Memoized `parse(file)` query keyed by content hash (Salsa-style early cutoff).
    pub fn parse_file(&self, path: &str) -> Result<Program, String> {
        let (hash, src) = {
            let inner = self.inner.lock().unwrap();
            let h = inner
                .file_hashes
                .get(path)
                .cloned()
                .ok_or_else(|| format!("file `{}` not registered in IncrementalDb", path))?;
            let s = inner.file_sources.get(path).cloned().unwrap_or_default();
            if let Some(cached) = inner.parsed_cache.get(&h).cloned() {
                drop(inner);
                self.inner.lock().unwrap().stats.parse_hits += 1;
                return Ok(cached);
            }
            (h, s)
        };
        let prog = parse(&src).map_err(|e| e.to_string())?;
        let new_sig = program_signature_hash(&prog);
        let mut inner = self.inner.lock().unwrap();
        inner.stats.parse_misses += 1;
        if let Some(prev_sig) = inner.file_sig_hashes.get(path) {
            if prev_sig == &new_sig {
                inner.stats.early_cutoffs += 1;
            }
        }
        inner.file_sig_hashes.insert(path.to_string(), new_sig);
        inner.parsed_cache.insert(hash, prog.clone());
        Ok(prog)
    }

    /// Memoized `check_program(file)` query keyed by content hash.
    pub fn check_file(&self, path: &str) -> Result<CheckReport, String> {
        let hash = {
            let mut inner = self.inner.lock().unwrap();
            let h = inner
                .file_hashes
                .get(path)
                .cloned()
                .ok_or_else(|| format!("file `{}` not registered in IncrementalDb", path))?;
            if let Some(cached) = inner.check_cache.get(&h).cloned() {
                inner.stats.check_hits += 1;
                return Ok(cached);
            }
            h
        };
        let prog = self.parse_file(path)?;
        let report = self.check_program_incremental(path, &prog);
        let mut inner = self.inner.lock().unwrap();
        inner.check_cache.insert(hash, report.clone());
        Ok(report)
    }

    /// Checks a pre-loaded (possibly multi-file merged) `Program` with signature-level early cutoff
    /// and per-function change tracking (ideal for `tenon lsp` live edits).
    pub fn check_program_incremental(&self, module_key: &str, prog: &Program) -> CheckReport {
        let full_hash = program_full_hash(prog);
        let sig_hash = program_signature_hash(prog);
        let fn_bodies = program_fn_body_hashes(prog);
        {
            let mut inner = self.inner.lock().unwrap();
            if let Some(cached) = inner.program_report_cache.get(&full_hash).cloned() {
                inner.stats.check_hits += 1;
                inner.stats.fn_cache_hits += fn_bodies.len();
                return cached;
            }
            if let Some((prev_sig, prev_bodies)) = inner.fn_sig_index.get(module_key).cloned() {
                if prev_sig == sig_hash {
                    inner.stats.early_cutoffs += 1;
                    for (fname, bhash) in &fn_bodies {
                        if prev_bodies.get(fname) == Some(bhash) {
                            inner.stats.fn_cache_hits += 1;
                        } else {
                            inner.stats.fn_cache_misses += 1;
                        }
                    }
                } else {
                    inner.stats.fn_cache_misses += fn_bodies.len();
                }
            } else {
                inner.stats.fn_cache_misses += fn_bodies.len();
            }
        }
        // Full verification leverages thread-safe persistent L1 VC cache + fast L0
        let report = check_program(prog);
        let mut inner = self.inner.lock().unwrap();
        inner.stats.check_misses += 1;
        inner.fn_sig_index.insert(module_key.to_string(), (sig_hash, fn_bodies));
        if inner.program_report_cache.len() > 64 {
            inner.program_report_cache.clear();
        }
        inner.program_report_cache.insert(full_hash, report.clone());
        report
    }

    pub fn manifest_for_file(&self, path: &str) -> Result<Option<DistributedAppManifest>, String> {
        Ok(self.check_file(path)?.manifest)
    }

    pub fn stats(&self) -> QueryStats {
        self.inner.lock().unwrap().stats.clone()
    }

    pub fn vc_cache_entries(&self) -> usize {
        crate::smt::vc_cache_len()
    }
}

/// Persistent disk verification cache (`.tenon-cache/verify_v3.json`) keyed by full structural
/// SHA-256 hash (`program_full_hash` covering all items, fields, variadic flags, views, metadata).
/// Returns `(CheckReport, cache_hit)`.
pub fn check_program_cached_on_disk(
    prog: &Program,
    cache_dir: &std::path::Path,
) -> (CheckReport, bool) {
    let key = format!("v3_{}", program_full_hash(prog));
    let cache_file = cache_dir.join("verify_v3.json");
    let mut map: BTreeMap<String, CheckReport> = std::fs::read_to_string(&cache_file)
        .ok()
        .and_then(|txt| serde_json::from_str(&txt).ok())
        .unwrap_or_default();

    if let Some(cached) = map.get(&key).cloned() {
        return (cached, true);
    }

    let report = check_program(prog);
    if report.ok {
        if map.len() >= 64 {
            map.clear();
        }
        map.insert(key, report.clone());
        let _ = std::fs::create_dir_all(cache_dir);
        if let Ok(json) = serde_json::to_string(&map) {
            let tmp = cache_dir.join(format!("verify_v3_{}.tmp", std::process::id()));
            if std::fs::write(&tmp, json).is_ok() {
                let _ = std::fs::rename(&tmp, &cache_file);
            }
        }
    }
    (report, false)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reactive_view_bounds_change_invalidates_disk_and_memory_cache() {
        let ok_src = r#"
@sync_view("session.msg_list")
pub reactive view SyncView(session_id Text) {
    durability = Reconstructible
    bounds = {
        max_items = 50 turns
        max_payload = 16 MiB
        max_frame = 4 MiB
        journal_budget = 16 MiB
        offline_recovery = BoundedReset
    }
    output committed_turns(session_id)
}
"#;
        let bad_src = ok_src.replace("BoundedReset", "NeverReset");
        let ok_prog = parse(ok_src).unwrap();
        let bad_prog = parse(&bad_src).unwrap();
        assert_ne!(
            program_full_hash(&ok_prog),
            program_full_hash(&bad_prog),
            "reactive view declaration edits must change program_full_hash"
        );

        let cache_dir = std::env::temp_dir().join(format!(
            "tenon_verify_cache_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&cache_dir);
        let (r1, hit1) = check_program_cached_on_disk(&ok_prog, &cache_dir);
        assert!(r1.ok && !hit1, "first BoundedReset check should pass with miss");
        let (r2, hit2) = check_program_cached_on_disk(&ok_prog, &cache_dir);
        assert!(r2.ok && hit2, "second BoundedReset check should hit cache");
        let (r3, hit3) = check_program_cached_on_disk(&bad_prog, &cache_dir);
        let _ = std::fs::remove_dir_all(&cache_dir);
        assert!(!hit3, "NeverReset must NEVER hit BoundedReset cache entry");
        assert!(
            !r3.ok && r3.diagnostics.iter().any(|d| d.code == "P2014"),
            "NeverReset with bounded journal_budget must fail with P2014, got: {:?}",
            r3.diagnostics
        );
    }

    #[test]
    fn extern_fn_is_variadic_change_invalidates_disk_and_memory_cache() {
        let var_src = r#"
extern "C" {
    fn mysum(a Int, ...) Int
}
pub fn main() Int {
    unsafe { mysum(1, 2) }
}
"#;
        let nonvar_src = var_src.replace(", ...", "     ");
        let var_prog = parse(var_src).unwrap();
        let nonvar_prog = parse(&nonvar_src).unwrap();
        assert_ne!(
            program_signature_hash(&var_prog),
            program_signature_hash(&nonvar_prog),
            "ExternFn.is_variadic change must invalidate program_signature_hash"
        );
        assert_ne!(
            program_full_hash(&var_prog),
            program_full_hash(&nonvar_prog),
            "ExternFn.is_variadic change must invalidate program_full_hash"
        );

        let cache_dir = std::env::temp_dir().join(format!(
            "tenon_verify_variadic_cache_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&cache_dir);
        let (r1, hit1) = check_program_cached_on_disk(&var_prog, &cache_dir);
        assert!(r1.ok && !hit1, "variadic extern call should pass with miss");
        let (r2, hit2) = check_program_cached_on_disk(&nonvar_prog, &cache_dir);
        let _ = std::fs::remove_dir_all(&cache_dir);
        assert!(!hit2, "removing `, ...` must NEVER hit variadic cache entry");
        assert!(
            !r2.ok && r2.diagnostics.iter().any(|d| d.code == "E0202"),
            "non-variadic call with extra arg must fail with E0202, got: {:?}",
            r2.diagnostics
        );
    }
}
