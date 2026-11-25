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
/// contracts `requires`/`ensures`, interfaces, attrs) WITHOUT function bodies.
/// When only a function body changes, `program_signature_hash` stays identical -> Early Cutoff!
pub fn program_signature_hash(prog: &Program) -> String {
    let mut sig_buf = String::new();
    for item in &prog.items {
        match item {
            Item::TypeAlias(ta) => {
                sig_buf.push_str(&format!(
                    "ta:{}:{}:{:?}:{:?}:{:?}\n",
                    ta.is_pub, ta.name, ta.type_params, ta.base, ta.predicate
                ));
            }
            Item::Record(r) => {
                sig_buf.push_str(&format!(
                    "rec:{}:{}:{:?}:{:?}:{:?}\n",
                    r.is_pub, r.name, r.type_params, r.fields, r.where_clauses
                ));
            }
            Item::Enum(e) => {
                sig_buf.push_str(&format!(
                    "enum:{}:{}:{:?}:{:?}\n",
                    e.is_pub, e.name, e.type_params, e.variants
                ));
            }
            Item::Interface(i) => {
                sig_buf.push_str(&format!(
                    "iface:{}:{}:{:?}:{:?}\n",
                    i.is_pub, i.name, i.type_params, i.methods
                ));
            }
            Item::Fn(f) => {
                sig_buf.push_str(&format!(
                    "fn:{}:{}:{}:{}:{:?}:{:?}:{:?}:{:?}:{:?}\n",
                    f.is_pub,
                    f.is_pure,
                    f.is_async,
                    f.is_ghost,
                    f.name,
                    f.type_params,
                    f.params,
                    f.ret_ty,
                    (&f.requires, &f.ensures, &f.attrs)
                ));
            }
            Item::ExternFn(ef) => {
                sig_buf.push_str(&format!("ext:{}:{:?}:{:?}\n", ef.name, ef.params, ef.ret_ty));
            }
            Item::Const(c) => {
                sig_buf.push_str(&format!("const:{}:{:?}:{:?}\n", c.name, c.ty, c.value));
            }
            Item::Bitfield(b) => {
                sig_buf.push_str(&format!("bf:{}:{:?}:{:?}\n", b.name, b.backing_ty, b.fields));
            }
            Item::Union(u) => {
                sig_buf.push_str(&format!("union:{}:{:?}\n", u.name, u.fields));
            }
            Item::AttrDecl(a) => {
                sig_buf.push_str(&format!("attr:{}:{:?}\n", a.name, a.params));
            }
            Item::Impl(imp) => {
                sig_buf.push_str(&format!("impl:{:?}:{:?}\n", imp.interface, imp.target_ty));
                for m in &imp.methods {
                    sig_buf.push_str(&format!(
                        "  m:{}:{:?}:{:?}:{:?}\n",
                        m.name, m.params, m.ret_ty, (&m.requires, &m.ensures)
                    ));
                }
            }
            _ => {}
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
                out.insert(format!("{}::{}", imp.target_ty.display_name(), m.name), sha256_hex(s.as_bytes()));
            }
        }
    }
    out
}

/// Computes a full structural hash of a parsed `Program` (signatures + all bodies).
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

/// Persistent disk verification cache (`.tenon-cache/verify_v1.json`) keyed by full structural
/// SHA-256 hash (`program_full_hash`). Returns `(CheckReport, cache_hit)`.
pub fn check_program_cached_on_disk(
    prog: &Program,
    cache_dir: &std::path::Path,
) -> (CheckReport, bool) {
    let key = format!("v1_{}", program_full_hash(prog));
    let cache_file = cache_dir.join("verify_v1.json");
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
            let tmp = cache_dir.join(format!("verify_v1_{}.tmp", std::process::id()));
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
