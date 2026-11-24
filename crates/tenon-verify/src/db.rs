use crate::checker::{CheckReport, check_program};
use crate::sync_solver::DistributedAppManifest;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tenon_syntax::{Program, parse};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct QueryStats {
    pub parse_hits: usize,
    pub parse_misses: usize,
    pub check_hits: usize,
    pub check_misses: usize,
}

#[derive(Clone, Default)]
pub struct IncrementalDb {
    inner: Arc<Mutex<DbInner>>,
}

#[derive(Default)]
struct DbInner {
    file_hashes: BTreeMap<String, String>,
    file_sources: BTreeMap<String, String>,
    parsed_cache: BTreeMap<String, Program>,
    check_cache: BTreeMap<String, CheckReport>,
    stats: QueryStats,
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
        let mut inner = self.inner.lock().unwrap();
        inner.stats.parse_misses += 1;
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
        let report = check_program(&prog);
        let mut inner = self.inner.lock().unwrap();
        inner.stats.check_misses += 1;
        inner.check_cache.insert(hash, report.clone());
        Ok(report)
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

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}
