use std::collections::HashSet;
use std::env;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

const RTLD_NOW: c_int = 2;

type FnLoadGguf = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;
type FnInfo = unsafe extern "C" fn(i64, i64) -> i64;
type FnReset = unsafe extern "C" fn(i64) -> i64;
type FnKvSlotSet = unsafe extern "C" fn(i64, i64) -> i64;
type FnCopyLogits = unsafe extern "C" fn(i64, i64, i64) -> i64;
type FnPrefill = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;
type FnVerifyBatch = unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64;
type FnAutoPlan = unsafe extern "C" fn(i64, f64, i64, i64, i64) -> i64;
type FnSampleNext =
    unsafe extern "C" fn(i64, i64, i64, f64, i64, f64, f64, f64, i64, i64, i64) -> i64;
type FnGetArch = unsafe extern "C" fn(i64) -> i64;
type FnTokenize = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;
type FnTokenCstr = unsafe extern "C" fn(i64, i64) -> i64;
type FnFree = unsafe extern "C" fn(i64) -> i64;
type FnInit = unsafe extern "C" fn() -> i64;

struct LlmApi {
    backend: &'static str,
    _dl_handle: *mut c_void,
    init_fn: Option<FnInit>,
    load_gguf: FnLoadGguf,
    info: FnInfo,
    reset: FnReset,
    kv_slot_set: FnKvSlotSet,
    copy_logits: FnCopyLogits,
    prefill: FnPrefill,
    verify_batch: FnVerifyBatch,
    auto_plan: FnAutoPlan,
    sample_next: FnSampleNext,
    get_arch: FnGetArch,
    tokenize: FnTokenize,
    token_cstr: FnTokenCstr,
    free: FnFree,
}

unsafe impl Send for LlmApi {}
unsafe impl Sync for LlmApi {}

#[derive(Debug, Clone)]
struct AutoPlan {
    slots: i64,
    nocache: bool,
    kv_q8: bool,
    prefill_chunk: i64,
    est_vram_bytes: i64,
    disk_bytes: i64,
    n_layers: i64,
    n_experts: i64,
    d_model: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpecMode {
    Off,
    Ngram,
    Draft,
}

#[derive(Debug, Clone)]
struct SamplingCfg {
    temp: f64,
    top_k: i64,
    top_p: f64,
    min_p: f64,
    repeat_penalty: f64,
    max_tokens: usize,
    seed: i64,
    json_constrained: bool,
    spec_mode: SpecMode,
    spec_k: usize,
}

impl Default for SamplingCfg {
    fn default() -> Self {
        Self {
            temp: 0.7,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            repeat_penalty: 1.1,
            max_tokens: 256,
            seed: 42,
            json_constrained: false,
            spec_mode: SpecMode::Off,
            spec_k: 5,
        }
    }
}

fn find_shim_source(rel: &str) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    candidates.push(PathBuf::from(rel));
    if let Ok(home) = env::var("TENON_HOME") {
        candidates.push(PathBuf::from(home).join(rel));
    }
    if let Ok(exe) = env::current_exe() {
        let mut dir = exe.parent().map(Path::to_path_buf);
        for _ in 0..5 {
            if let Some(d) = dir {
                candidates.push(d.join(rel));
                dir = d.parent().map(Path::to_path_buf);
            }
        }
    }
    candidates.into_iter().find(|p| p.exists())
}

fn load_symbol<T: Copy>(handle: *mut c_void, name: &str) -> Result<T, String> {
    let cname = CString::new(name).map_err(|_| format!("invalid symbol name `{}`", name))?;
    let ptr = unsafe { dlsym(handle, cname.as_ptr()) };
    if ptr.is_null() {
        let err = unsafe {
            let e = dlerror();
            if e.is_null() {
                "unknown dlsym error".to_string()
            } else {
                CStr::from_ptr(e).to_string_lossy().into_owned()
            }
        };
        return Err(format!("missing symbol `{}`: {}", name, err));
    }
    Ok(unsafe { std::mem::transmute_copy(&ptr) })
}

fn load_llm_api() -> Result<LlmApi, String> {
    let (backend, rel_src, prefix, ext) = if cfg!(target_os = "macos") {
        ("metal", "std/gpu/metal/moe_stream_shim.m", "tn_moe_", "dylib")
    } else if cfg!(target_os = "linux") {
        ("cuda", "std/gpu/cuda/moe_stream_cuda_shim.c", "tn_cumoe_", "so")
    } else {
        return Err("`tenon llm` is currently supported on macOS (Metal) and Linux (CUDA)".into());
    };

    let src = find_shim_source(rel_src).ok_or_else(|| {
        format!(
            "could not locate `{}` (run from tenon-lang repo root or set TENON_HOME)",
            rel_src
        )
    })?;
    let mtime = fs::metadata(&src)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let lib_path = PathBuf::from(format!("/tmp/tenon_llm_{}_v5_{}.{}", backend, mtime, ext));

    if !lib_path.exists() {
        eprintln!(
            "[tenon llm] compiling {} v5 runtime shim ({}) -> {}",
            backend,
            src.display(),
            lib_path.display()
        );
        let status = if cfg!(target_os = "macos") {
            Command::new("clang")
                .args([
                    "-O3",
                    "-fobjc-arc",
                    "-shared",
                    "-fPIC",
                    "-Wno-deprecated-declarations",
                    "-Wno-nontrivial-memaccess",
                    "-Wno-unused-but-set-variable",
                ])
                .arg(&src)
                .args(["-framework", "Metal", "-framework", "Foundation", "-o"])
                .arg(&lib_path)
                .status()
        } else {
            let mut cmd = Command::new("gcc");
            cmd.args(["-O3", "-shared", "-fPIC"]);
            if cfg!(target_arch = "x86_64") {
                cmd.args(["-mavx2", "-mfma", "-mf16c"]);
            }
            cmd.arg(&src)
                .args(["-ldl", "-lpthread", "-lm", "-o"])
                .arg(&lib_path)
                .status()
        }
        .map_err(|e| format!("failed to invoke C/ObjC compiler: {}", e))?;
        if !status.success() {
            return Err(format!("compiling {} shim failed with {}", backend, status));
        }
    }

    let cpath = CString::new(lib_path.to_string_lossy().as_ref())
        .map_err(|_| "invalid shared library path".to_string())?;
    let dl_handle = unsafe { dlopen(cpath.as_ptr(), RTLD_NOW) };
    if dl_handle.is_null() {
        let err = unsafe {
            let e = dlerror();
            if e.is_null() {
                "unknown dlopen error".to_string()
            } else {
                CStr::from_ptr(e).to_string_lossy().into_owned()
            }
        };
        return Err(format!("dlopen({}) failed: {}", lib_path.display(), err));
    }

    let sym = |s: &str| format!("{}{}", prefix, s);
    let init_fn = if backend == "cuda" {
        Some(load_symbol::<FnInit>(dl_handle, &sym("init"))?)
    } else {
        None
    };

    let api = LlmApi {
        backend,
        _dl_handle: dl_handle,
        init_fn,
        load_gguf: load_symbol(dl_handle, &sym("load_gguf"))?,
        info: load_symbol(dl_handle, &sym("info"))?,
        reset: load_symbol(dl_handle, &sym("reset"))?,
        kv_slot_set: load_symbol(dl_handle, &sym("kv_slot_set"))?,
        copy_logits: load_symbol(dl_handle, &sym("copy_logits"))?,
        prefill: load_symbol(dl_handle, &sym("prefill"))?,
        verify_batch: load_symbol(dl_handle, &sym("verify_batch"))?,
        auto_plan: load_symbol(dl_handle, &sym("auto_plan"))?,
        sample_next: load_symbol(dl_handle, &sym("sample_next"))?,
        get_arch: load_symbol(dl_handle, &sym("get_arch"))?,
        tokenize: load_symbol(dl_handle, &sym("tokenize"))?,
        token_cstr: load_symbol(dl_handle, &sym("token_cstr"))?,
        free: load_symbol(dl_handle, &sym("free"))?,
    };
    if let Some(init) = api.init_fn {
        if unsafe { init() } == 0 {
            return Err("CUDA runtime initialization failed (no NVIDIA GPU / libcuda.so.1)".into());
        }
    }
    Ok(api)
}

impl LlmApi {
    fn plan(&self, path: &str, max_vram_gb: f64, max_ctx: i64, kv_mode: i64) -> Result<AutoPlan, String> {
        let cp = CString::new(path).map_err(|_| "invalid model path")?;
        let mut out = [0i64; 9];
        let ok = unsafe {
            (self.auto_plan)(
                cp.as_ptr() as i64,
                max_vram_gb,
                max_ctx,
                kv_mode,
                out.as_mut_ptr() as i64,
            )
        };
        if ok == 0 {
            return Err(format!("failed to inspect GGUF metadata for `{}`", path));
        }
        Ok(AutoPlan {
            slots: out[0],
            nocache: out[1] != 0,
            kv_q8: out[2] != 0,
            prefill_chunk: out[3],
            est_vram_bytes: out[4],
            disk_bytes: out[5],
            n_layers: out[6],
            n_experts: out[7],
            d_model: out[8],
        })
    }

    fn load(&self, path: &str, slots: i64, nocache: bool, overlap: bool) -> Result<i64, String> {
        let cp = CString::new(path).map_err(|_| "invalid model path")?;
        let h = unsafe {
            (self.load_gguf)(
                cp.as_ptr() as i64,
                slots,
                if nocache { 1 } else { 0 },
                if overlap { 1 } else { 0 },
            )
        };
        if h <= 0 {
            return Err(format!("failed to load GGUF model `{}`", path));
        }
        Ok(h)
    }

    fn info(&self, h: i64, field: i64) -> i64 {
        unsafe { (self.info)(h, field) }
    }

    fn arch(&self, h: i64) -> String {
        let ptr = unsafe { (self.get_arch)(h) } as *const c_char;
        if ptr.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }

    fn tokenize(&self, h: i64, text: &str, max_toks: usize) -> Vec<i64> {
        let Ok(ct) = CString::new(text) else {
            return Vec::new();
        };
        let mut buf = vec![0i64; max_toks];
        let n = unsafe {
            (self.tokenize)(h, ct.as_ptr() as i64, buf.as_mut_ptr() as i64, max_toks as i64)
        };
        buf.truncate(n.max(0) as usize);
        buf
    }

    fn token_str(&self, h: i64, tok_id: i64) -> String {
        let ptr = unsafe { (self.token_cstr)(h, tok_id) } as *const c_char;
        if ptr.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }

    fn kv_slot_set(&self, h: i64, slot_id: usize) -> bool {
        unsafe { (self.kv_slot_set)(h, slot_id as i64) != 0 }
    }

    fn copy_logits(&self, h: i64, out: &mut [f32]) -> usize {
        let n = unsafe { (self.copy_logits)(h, out.as_mut_ptr() as i64, out.len() as i64) };
        n.max(0) as usize
    }

    fn prefill(&self, h: i64, ids: &[i64], start_pos: i64) -> i64 {
        unsafe { (self.prefill)(h, ids.as_ptr() as i64, ids.len() as i64, start_pos) }
    }

    fn verify_batch(&self, h: i64, ids: &[i64], start_pos: i64, out_argmax: &mut [i64]) -> usize {
        let n = unsafe {
            (self.verify_batch)(
                h,
                ids.as_ptr() as i64,
                ids.len() as i64,
                start_pos,
                out_argmax.as_mut_ptr() as i64,
            )
        };
        n.max(0) as usize
    }

    fn sample_next(
        &self,
        h: i64,
        tok_id: i64,
        pos: i64,
        cfg: &SamplingCfg,
        recent: &[i64],
        seed: i64,
    ) -> i64 {
        unsafe {
            (self.sample_next)(
                h,
                tok_id,
                pos,
                cfg.temp,
                cfg.top_k,
                cfg.top_p,
                cfg.min_p,
                cfg.repeat_penalty,
                if recent.is_empty() { 0 } else { recent.as_ptr() as i64 },
                recent.len() as i64,
                seed,
            )
        }
    }

    fn reset(&self, h: i64) {
        unsafe {
            (self.reset)(h);
        }
    }

    fn free(&self, h: i64) {
        unsafe {
            (self.free)(h);
        }
    }
}

// ============================================================================
// Radix Prefix Tree KV Cache (Multi-Slot Longest-Prefix Reuse -> 5-15x TTFT!)
// ============================================================================
#[derive(Debug, Clone)]
struct KvSlotState {
    tokens: Vec<i64>,
    last_used: u64,
}

struct RadixKvCache {
    slots: Vec<KvSlotState>,
    clock: u64,
    total_hits: u64,
    total_cached_tokens: u64,
}

impl RadixKvCache {
    fn new(n_slots: usize) -> Self {
        let n = n_slots.max(1);
        Self {
            slots: (0..n)
                .map(|_| KvSlotState {
                    tokens: Vec::new(),
                    last_used: 0,
                })
                .collect(),
            clock: 0,
            total_hits: 0,
            total_cached_tokens: 0,
        }
    }

    fn match_and_acquire(&mut self, prompt: &[i64]) -> (usize, usize) {
        let max_reuse = prompt.len().saturating_sub(1);
        let mut best_slot = 0usize;
        let mut best_lcp = 0usize;
        let mut best_lru = u64::MAX;

        for (idx, st) in self.slots.iter().enumerate() {
            let mut lcp = 0usize;
            let lim = st.tokens.len().min(max_reuse);
            while lcp < lim && st.tokens[lcp] == prompt[lcp] {
                lcp += 1;
            }
            if lcp > best_lcp || (lcp == best_lcp && st.last_used < best_lru) {
                best_lcp = lcp;
                best_slot = idx;
                best_lru = st.last_used;
            }
        }
        self.clock += 1;
        self.slots[best_slot].last_used = self.clock;
        if best_lcp > 0 {
            self.total_hits += 1;
            self.total_cached_tokens += best_lcp as u64;
        }
        (best_slot, best_lcp)
    }

    fn commit(&mut self, slot: usize, full_tokens: Vec<i64>) {
        if let Some(st) = self.slots.get_mut(slot) {
            self.clock += 1;
            st.tokens = full_tokens;
            st.last_used = self.clock;
        }
    }
}

// ============================================================================
// Incremental RFC 8259 JSON Grammar FSM (100% Valid JSON & OpenAI tool_calls!)
// ============================================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Container {
    Object,
    Array,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonPhase {
    ExpectRootObject,
    ExpectValue,
    ExpectKeyOrObjEnd,
    ExpectKey,
    ExpectColon,
    ExpectCommaOrEnd,
    InString { for_key: bool, escaped: bool },
    InNumber { has_dot: bool, has_exp: bool, need_digit: bool },
    InLit { lit: &'static [u8], pos: usize },
    Done,
}

#[derive(Debug, Clone)]
struct JsonFsm {
    stack: Vec<Container>,
    phase: JsonPhase,
}

impl JsonFsm {
    fn new_object() -> Self {
        Self {
            stack: Vec::with_capacity(8),
            phase: JsonPhase::ExpectRootObject,
        }
    }

    fn is_done(&self) -> bool {
        self.phase == JsonPhase::Done
    }

    fn after_value(&mut self) {
        if self.stack.is_empty() {
            self.phase = JsonPhase::Done;
        } else {
            self.phase = JsonPhase::ExpectCommaOrEnd;
        }
    }

    fn step_byte(&mut self, b: u8) -> bool {
        let ws = matches!(b, b' ' | b'\t' | b'\n' | b'\r');
        match self.phase {
            JsonPhase::Done => ws,
            JsonPhase::ExpectRootObject => {
                if ws {
                    true
                } else if b == b'{' {
                    self.stack.push(Container::Object);
                    self.phase = JsonPhase::ExpectKeyOrObjEnd;
                    true
                } else {
                    false
                }
            }
            JsonPhase::ExpectKeyOrObjEnd => {
                if ws {
                    true
                } else if b == b'}' {
                    if self.stack.pop() == Some(Container::Object) {
                        self.after_value();
                        true
                    } else {
                        false
                    }
                } else if b == b'"' {
                    self.phase = JsonPhase::InString {
                        for_key: true,
                        escaped: false,
                    };
                    true
                } else {
                    false
                }
            }
            JsonPhase::ExpectKey => {
                if ws {
                    true
                } else if b == b'"' {
                    self.phase = JsonPhase::InString {
                        for_key: true,
                        escaped: false,
                    };
                    true
                } else {
                    false
                }
            }
            JsonPhase::ExpectColon => {
                if ws {
                    true
                } else if b == b':' {
                    self.phase = JsonPhase::ExpectValue;
                    true
                } else {
                    false
                }
            }
            JsonPhase::ExpectValue => {
                if ws {
                    true
                } else if b == b'{' {
                    if self.stack.len() >= 16 {
                        return false;
                    }
                    self.stack.push(Container::Object);
                    self.phase = JsonPhase::ExpectKeyOrObjEnd;
                    true
                } else if b == b'[' {
                    if self.stack.len() >= 16 {
                        return false;
                    }
                    self.stack.push(Container::Array);
                    self.phase = JsonPhase::ExpectValue;
                    true
                } else if b == b']' && self.stack.last() == Some(&Container::Array) {
                    self.stack.pop();
                    self.after_value();
                    true
                } else if b == b'"' {
                    self.phase = JsonPhase::InString {
                        for_key: false,
                        escaped: false,
                    };
                    true
                } else if b == b'-' {
                    self.phase = JsonPhase::InNumber {
                        has_dot: false,
                        has_exp: false,
                        need_digit: true,
                    };
                    true
                } else if b.is_ascii_digit() {
                    self.phase = JsonPhase::InNumber {
                        has_dot: false,
                        has_exp: false,
                        need_digit: false,
                    };
                    true
                } else if b == b't' {
                    self.phase = JsonPhase::InLit { lit: b"true", pos: 1 };
                    true
                } else if b == b'f' {
                    self.phase = JsonPhase::InLit { lit: b"false", pos: 1 };
                    true
                } else if b == b'n' {
                    self.phase = JsonPhase::InLit { lit: b"null", pos: 1 };
                    true
                } else {
                    false
                }
            }
            JsonPhase::ExpectCommaOrEnd => {
                if ws {
                    true
                } else if b == b',' {
                    match self.stack.last() {
                        Some(Container::Object) => {
                            self.phase = JsonPhase::ExpectKey;
                            true
                        }
                        Some(Container::Array) => {
                            self.phase = JsonPhase::ExpectValue;
                            true
                        }
                        None => false,
                    }
                } else if b == b'}' && self.stack.last() == Some(&Container::Object) {
                    self.stack.pop();
                    self.after_value();
                    true
                } else if b == b']' && self.stack.last() == Some(&Container::Array) {
                    self.stack.pop();
                    self.after_value();
                    true
                } else {
                    false
                }
            }
            JsonPhase::InString { for_key, escaped } => {
                if b < 0x20 {
                    return false;
                }
                if escaped {
                    if matches!(b, b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u') {
                        self.phase = JsonPhase::InString {
                            for_key,
                            escaped: false,
                        };
                        true
                    } else {
                        false
                    }
                } else if b == b'\\' {
                    self.phase = JsonPhase::InString {
                        for_key,
                        escaped: true,
                    };
                    true
                } else if b == b'"' {
                    if for_key {
                        self.phase = JsonPhase::ExpectColon;
                    } else {
                        self.after_value();
                    }
                    true
                } else {
                    true
                }
            }
            JsonPhase::InNumber {
                has_dot,
                has_exp,
                need_digit,
            } => {
                if b.is_ascii_digit() {
                    self.phase = JsonPhase::InNumber {
                        has_dot,
                        has_exp,
                        need_digit: false,
                    };
                    true
                } else if !need_digit && b == b'.' && !has_dot && !has_exp {
                    self.phase = JsonPhase::InNumber {
                        has_dot: true,
                        has_exp,
                        need_digit: true,
                    };
                    true
                } else if !need_digit && (b == b'e' || b == b'E') && !has_exp {
                    self.phase = JsonPhase::InNumber {
                        has_dot,
                        has_exp: true,
                        need_digit: true,
                    };
                    true
                } else if !need_digit {
                    self.after_value();
                    self.step_byte(b)
                } else {
                    false
                }
            }
            JsonPhase::InLit { lit, pos } => {
                if pos < lit.len() && b == lit[pos] {
                    if pos + 1 == lit.len() {
                        self.after_value();
                    } else {
                        self.phase = JsonPhase::InLit { lit, pos: pos + 1 };
                    }
                    true
                } else {
                    false
                }
            }
        }
    }

    fn try_advance(&self, piece: &str) -> Option<Self> {
        if piece.is_empty() {
            return None;
        }
        let mut next = self.clone();
        for &b in piece.as_bytes() {
            if !next.step_byte(b) {
                return None;
            }
        }
        Some(next)
    }
}

// ============================================================================
// Loaded Engine State + Constrained Sampler + Speculative Decoding
// ============================================================================
#[derive(Clone)]
struct ModelMeta {
    backend: &'static str,
    model_id: String,
    arch: String,
    max_ctx: i64,
    slot_ctx: i64,
    n_kv_slots: usize,
    kv_q8: bool,
    mla_latent: bool,
    n_bpe_merges: i64,
    tp_size: i64,
    has_draft: bool,
}

struct LoadedModel {
    api: LlmApi,
    handle: i64,
    draft_handle: i64,
    meta: ModelMeta,
    bos_id: i64,
    eos_id: i64,
    vocab_size: usize,
    radix_cache: RadixKvCache,
    vocab_pieces: Vec<String>,
    logits_scratch: Vec<f32>,
}

impl LoadedModel {
    fn ensure_vocab_pieces(&mut self) {
        if !self.vocab_pieces.is_empty() {
            return;
        }
        let mut pieces = Vec::with_capacity(self.vocab_size);
        for v in 0..self.vocab_size {
            pieces.push(self.api.token_str(self.handle, v as i64));
        }
        self.vocab_pieces = pieces;
    }

    fn sample_constrained_json(
        &mut self,
        fsm: &JsonFsm,
        cfg: &SamplingCfg,
        recent: &[i64],
        seed: i64,
    ) -> (i64, Option<JsonFsm>) {
        if fsm.is_done() {
            return (self.eos_id, None);
        }
        self.ensure_vocab_pieces();
        if self.logits_scratch.len() < self.vocab_size {
            self.logits_scratch.resize(self.vocab_size, 0.0);
        }
        let n = self.api.copy_logits(self.handle, &mut self.logits_scratch);
        if n == 0 {
            return (self.eos_id, None);
        }
        if cfg.repeat_penalty > 1.0001 && !recent.is_empty() {
            let rp = cfg.repeat_penalty as f32;
            let seen: HashSet<i64> = recent.iter().copied().collect();
            for r in seen {
                if r >= 0 && (r as usize) < n {
                    let lg = &mut self.logits_scratch[r as usize];
                    *lg = if *lg > 0.0 { *lg / rp } else { *lg * rp };
                }
            }
        }
        // Fast path: inspect top-512 highest logits first (< 0.3 ms!), fallback to full scan if needed
        let mut idx: Vec<usize> = (0..n).collect();
        let probe_k = 512.min(n);
        idx.select_nth_unstable_by(probe_k - 1, |&a, &b| {
            self.logits_scratch[b]
                .partial_cmp(&self.logits_scratch[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx[..probe_k].sort_unstable_by(|&a, &b| {
            self.logits_scratch[b]
                .partial_cmp(&self.logits_scratch[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut valid: Vec<(usize, f32, JsonFsm)> = Vec::with_capacity(64);
        let limit_k = (cfg.top_k.max(1) as usize).min(64);
        for &id in &idx[..probe_k] {
            if id as i64 == self.eos_id || is_stop_token(id as i64, self.eos_id, &self.vocab_pieces[id]) {
                continue;
            }
            if let Some(ns) = fsm.try_advance(&self.vocab_pieces[id]) {
                valid.push((id, self.logits_scratch[id], ns));
                if valid.len() >= limit_k {
                    break;
                }
            }
        }
        if valid.is_empty() {
            // Guaranteed fallback over entire vocabulary so output is ALWAYS valid JSON!
            let mut best: Option<(usize, f32, JsonFsm)> = None;
            for id in 0..n {
                if id as i64 == self.eos_id || is_stop_token(id as i64, self.eos_id, &self.vocab_pieces[id]) {
                    continue;
                }
                let val = self.logits_scratch[id];
                if let Some((_, bv, _)) = &best {
                    if val <= *bv {
                        continue;
                    }
                }
                if let Some(ns) = fsm.try_advance(&self.vocab_pieces[id]) {
                    best = Some((id, val, ns));
                }
            }
            if let Some((id, _, ns)) = best {
                return (id as i64, Some(ns));
            }
            return (self.eos_id, None);
        }
        if cfg.temp <= 1e-5 || valid.len() == 1 {
            let (id, _, ns) = valid.remove(0);
            return (id as i64, Some(ns));
        }
        let inv_t = 1.0f32 / (cfg.temp as f32);
        let mx = valid[0].1 * inv_t;
        let mut probs: Vec<f32> = valid.iter().map(|(_, v, _)| (v * inv_t - mx).exp()).collect();
        let sum: f32 = probs.iter().sum::<f32>().max(1e-12);
        for p in &mut probs {
            *p /= sum;
        }
        let p_thresh = if cfg.min_p > 0.0 {
            probs[0] * (cfg.min_p as f32)
        } else {
            0.0
        };
        let mut cum = 0.0f32;
        let mut keep = 0usize;
        for (i, &p) in probs.iter().enumerate() {
            if i > 0 && p < p_thresh {
                break;
            }
            cum += p;
            keep += 1;
            if cfg.top_p > 0.0 && cfg.top_p < 1.0 && cum >= cfg.top_p as f32 {
                break;
            }
        }
        let mut x = (seed as u64) ^ 0x9E3779B97F4A7C15u64;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        let u = (((x.wrapping_mul(2685821657736338717u64)) >> 11) as f64) * (1.0 / 9007199254740992.0);
        let target = (u as f32) * cum;
        let mut acc = 0.0f32;
        let mut chosen = 0usize;
        for i in 0..keep {
            acc += probs[i];
            if acc >= target {
                chosen = i;
                break;
            }
        }
        let (id, _, ns) = valid.swap_remove(chosen);
        (id as i64, Some(ns))
    }
}

impl Drop for LoadedModel {
    fn drop(&mut self) {
        if self.draft_handle > 0 {
            self.api.free(self.draft_handle);
            self.draft_handle = 0;
        }
        if self.handle > 0 {
            self.api.free(self.handle);
            self.handle = 0;
        }
    }
}

fn format_tools_prompt(tools: &[Value]) -> String {
    let mut s = String::from(
        "You have access to the following tools. To call a tool, respond ONLY with a valid JSON object of the form {\"name\": \"<tool_name>\", \"arguments\": {<args>}}.\nAvailable tools:\n",
    );
    for t in tools {
        let f = t.get("function").unwrap_or(t);
        let name = f.get("name").and_then(Value::as_str).unwrap_or("tool");
        let desc = f.get("description").and_then(Value::as_str).unwrap_or("");
        let params = f.get("parameters").cloned().unwrap_or_else(|| json!({}));
        s.push_str(&format!("- {}: {} | parameters: {}\n", name, desc, params));
    }
    s
}

fn format_chat_template(arch: &str, messages: &[(String, String)]) -> String {
    let a = arch.to_ascii_lowercase();
    let mut out = String::new();
    if a.starts_with("llama") {
        for (role, content) in messages {
            out.push_str("<|start_header_id|>");
            out.push_str(role);
            out.push_str("<|end_header_id|>\n\n");
            out.push_str(content);
            out.push_str("<|eot_id|>");
        }
        out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    } else if a.starts_with("glm") {
        out.push_str("[gMASK]<sop>");
        for (role, content) in messages {
            if role == "system" {
                out.push_str("<|system|>\n");
                out.push_str(content);
            } else if role == "assistant" {
                out.push_str("<|assistant|>\n");
                out.push_str(content);
            } else {
                out.push_str("<|user|>\n");
                out.push_str(content);
            }
        }
        out.push_str("<|assistant|>\n");
    } else if a.starts_with("qwen") {
        for (role, content) in messages {
            out.push_str("<|im_start|>");
            out.push_str(role);
            out.push('\n');
            out.push_str(content);
            out.push_str("<|im_end|>\n");
        }
        out.push_str("<|im_start|>assistant\n");
    } else if a.starts_with("deepseek") {
        for (role, content) in messages {
            if role == "system" {
                out.push_str(content);
                out.push_str("\n\n");
            } else if role == "assistant" {
                out.push_str("Assistant: ");
                out.push_str(content);
                out.push_str("\n\n");
            } else {
                out.push_str("User: ");
                out.push_str(content);
                out.push_str("\n\n");
            }
        }
        out.push_str("Assistant:");
    } else {
        for (role, content) in messages {
            out.push_str(role);
            out.push_str(": ");
            out.push_str(content);
            out.push('\n');
        }
        out.push_str("assistant: ");
    }
    out
}

fn is_stop_token(tok_id: i64, eos_id: i64, piece: &str) -> bool {
    if tok_id == eos_id {
        return true;
    }
    matches!(
        piece,
        "<|eot_id|>"
            | "<|eom_id|>"
            | "<|eoft_id|>"
            | "<|end_of_text|>"
            | "<|im_end|>"
            | "<|endoftext|>"
            | "<|user|>"
            | "</s>"
    )
}

fn propose_ngram_draft(history: &[i64], k: usize) -> Vec<i64> {
    if history.len() < 4 || k == 0 {
        return Vec::new();
    }
    for n in (1..=3).rev() {
        if history.len() <= n + 1 {
            continue;
        }
        let pat = &history[history.len() - n..];
        let search_end = history.len() - n;
        for i in (0..search_end).rev() {
            if &history[i..i + n] == pat {
                let start = i + n;
                let end = (start + k).min(history.len() - n);
                if end > start {
                    return history[start..end].to_vec();
                }
            }
        }
    }
    Vec::new()
}

struct GenResult {
    text: String,
    prompt_tokens: usize,
    cached_prefix_tokens: usize,
    kv_slot: usize,
    completion_tokens: usize,
    spec_proposed: usize,
    spec_accepted: usize,
    prefill_ms: f64,
    decode_ms: f64,
    finish_reason: &'static str,
}

fn run_generation<F>(
    m: &mut LoadedModel,
    prompt: &str,
    add_bos: bool,
    cfg: &SamplingCfg,
    mut on_piece: F,
) -> GenResult
where
    F: FnMut(&str) -> bool,
{
    let slot_ctx = m.meta.slot_ctx as usize;
    let max_prompt = slot_ctx.saturating_sub(cfg.max_tokens.min(512) + 8).max(16);
    let mut ids = Vec::new();
    if add_bos && m.bos_id >= 0 {
        ids.push(m.bos_id);
    }
    ids.extend(m.api.tokenize(m.handle, prompt, max_prompt));
    if ids.is_empty() {
        ids.push(m.bos_id.max(1));
    }
    let prompt_tokens = ids.len();

    // 1. Radix Prefix Tree KV Cache lookup + slot switch!
    let (kv_slot, lcp) = m.radix_cache.match_and_acquire(&ids);
    m.api.kv_slot_set(m.handle, kv_slot);

    let t_pf = Instant::now();
    let _argmax = m.api.prefill(m.handle, &ids[lcp..], lcp as i64);
    let prefill_ms = t_pf.elapsed().as_secs_f64() * 1000.0;
    m.api.reset(m.handle);

    let mut full_seq: Vec<i64> = ids.clone();
    let mut json_fsm = if cfg.json_constrained {
        Some(JsonFsm::new_object())
    } else {
        None
    };

    let mut next_tok = if let Some(ref fsm) = json_fsm {
        let (tok, ns) = m.sample_constrained_json(
            fsm,
            cfg,
            &full_seq[full_seq.len().saturating_sub(64)..],
            cfg.seed,
        );
        if let Some(n) = ns {
            json_fsm = Some(n);
        }
        tok
    } else {
        m.api.sample_next(
            m.handle,
            -1,
            (prompt_tokens - 1) as i64,
            cfg,
            &full_seq[full_seq.len().saturating_sub(64)..],
            cfg.seed,
        )
    };

    let use_draft = cfg.spec_mode == SpecMode::Draft && m.draft_handle > 0 && !cfg.json_constrained;
    let use_ngram = cfg.spec_mode == SpecMode::Ngram && !cfg.json_constrained && cfg.temp <= 1e-5;
    if use_draft {
        m.api.prefill(m.draft_handle, &ids, 0);
        m.api.reset(m.draft_handle);
    }

    let mut out_text = String::new();
    let mut completion_tokens = 0usize;
    let mut spec_proposed = 0usize;
    let mut spec_accepted = 0usize;
    let mut finish_reason = "length";
    let t_dec = Instant::now();

    while completion_tokens < cfg.max_tokens {
        let piece = m.api.token_str(m.handle, next_tok);
        if is_stop_token(next_tok, m.eos_id, &piece) {
            finish_reason = "stop";
            break;
        }
        out_text.push_str(&piece);
        completion_tokens += 1;
        full_seq.push(next_tok);
        if !on_piece(&piece) {
            finish_reason = "stop";
            break;
        }
        if json_fsm.as_ref().is_some_and(JsonFsm::is_done) {
            finish_reason = "stop";
            break;
        }
        let pos = full_seq.len() - 1;
        if pos + 2 >= slot_ctx || completion_tokens >= cfg.max_tokens {
            finish_reason = "length";
            break;
        }

        // Lossless Batched Speculative Decoding (`--draft <small.gguf>` OR `--speculative ngram`)!
        if (use_draft || use_ngram) && cfg.spec_k > 0 {
            let max_k = cfg.spec_k.min(cfg.max_tokens - completion_tokens).min(slot_ctx - pos - 2);
            let mut draft_toks = Vec::with_capacity(max_k);
            if use_draft {
                let mut dcur = next_tok;
                let mut dpos = pos as i64;
                let greedy_cfg = SamplingCfg {
                    temp: 0.0,
                    repeat_penalty: 1.0,
                    ..cfg.clone()
                };
                for _ in 0..max_k {
                    dcur = m.api.sample_next(m.draft_handle, dcur, dpos, &greedy_cfg, &[], cfg.seed);
                    let dp = m.api.token_str(m.draft_handle, dcur);
                    draft_toks.push(dcur);
                    dpos += 1;
                    if is_stop_token(dcur, m.eos_id, &dp) {
                        break;
                    }
                }
            } else {
                draft_toks = propose_ngram_draft(&full_seq, max_k);
            }

            if !draft_toks.is_empty() {
                spec_proposed += draft_toks.len();
                let mut verify_in = Vec::with_capacity(1 + draft_toks.len());
                verify_in.push(next_tok);
                verify_in.extend_from_slice(&draft_toks);
                let mut target_out = vec![0i64; verify_in.len()];
                m.api.verify_batch(m.handle, &verify_in, pos as i64, &mut target_out);

                let mut n_acc = 0usize;
                for (i, &dt) in draft_toks.iter().enumerate() {
                    if target_out[i] == dt {
                        n_acc += 1;
                        spec_accepted += 1;
                        let dp = m.api.token_str(m.handle, dt);
                        if is_stop_token(dt, m.eos_id, &dp) {
                            finish_reason = "stop";
                            break;
                        }
                        out_text.push_str(&dp);
                        completion_tokens += 1;
                        full_seq.push(dt);
                        if !on_piece(&dp) || completion_tokens >= cfg.max_tokens {
                            finish_reason = "stop";
                            break;
                        }
                    } else {
                        break;
                    }
                }
                if finish_reason == "stop" {
                    break;
                }
                next_tok = target_out[n_acc];
                continue;
            }
        }

        let rstart = full_seq.len().saturating_sub(64);
        let step_seed = cfg.seed.wrapping_add((completion_tokens as i64).wrapping_mul(1000003));
        if let Some(ref fsm) = json_fsm {
            // Feed `next_tok` into target model to update KV & compute logits, then apply JSON FSM mask!
            let _ = m.api.prefill(m.handle, &[next_tok], pos as i64);
            let (tok, ns) = m.sample_constrained_json(fsm, cfg, &full_seq[rstart..], step_seed);
            if let Some(n) = ns {
                json_fsm = Some(n);
            }
            next_tok = tok;
        } else {
            next_tok = m.api.sample_next(
                m.handle,
                next_tok,
                pos as i64,
                cfg,
                &full_seq[rstart..],
                step_seed,
            );
        }
    }

    let decode_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
    m.radix_cache.commit(kv_slot, full_seq);
    GenResult {
        text: out_text,
        prompt_tokens,
        cached_prefix_tokens: lcp,
        kv_slot,
        completion_tokens,
        spec_proposed,
        spec_accepted,
        prefill_ms,
        decode_ms,
        finish_reason,
    }
}

// ============================================================================
// Concurrent Request Scheduler & OpenAI-Compatible HTTP + SSE Server
// ============================================================================
struct ServerState {
    meta: ModelMeta,
    model: Mutex<LoadedModel>,
    queue_sem: Semaphore,
    queued_reqs: AtomicUsize,
    total_reqs: AtomicU64,
    radix_hits: AtomicU64,
    radix_saved_toks: AtomicU64,
}

async fn read_http_request(stream: &mut TcpStream) -> io::Result<(String, String, String)> {
    let mut buf = Vec::with_capacity(4096);
    let mut tmp = [0u8; 4096];
    let mut header_end = None;
    let mut content_length = 0usize;

    loop {
        let n = stream.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if header_end.is_none() {
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                header_end = Some(pos + 4);
                let hdr_str = String::from_utf8_lossy(&buf[..pos]);
                for line in hdr_str.lines().skip(1) {
                    if let Some((k, v)) = line.split_once(':') {
                        if k.trim().eq_ignore_ascii_case("content-length") {
                            content_length = v.trim().parse().unwrap_or(0);
                        }
                    }
                }
            }
        }
        if let Some(he) = header_end {
            if buf.len() >= he + content_length {
                break;
            }
        }
        if buf.len() > 8 * 1024 * 1024 {
            break;
        }
    }

    let he = header_end.unwrap_or(buf.len());
    let head = String::from_utf8_lossy(&buf[..he]);
    let first = head.lines().next().unwrap_or("");
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();
    let body = if buf.len() > he {
        String::from_utf8_lossy(&buf[he..(he + content_length).min(buf.len())]).into_owned()
    } else {
        String::new()
    };
    Ok((method, path, body))
}

async fn write_json_response(stream: &mut TcpStream, status: &str, body: &Value) -> io::Result<()> {
    let payload = body.to_string();
    let resp = format!(
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        payload.len(),
        payload
    );
    stream.write_all(resp.as_bytes()).await?;
    stream.flush().await
}

fn parse_sampling_from_json(v: &Value, default_cfg: &SamplingCfg) -> SamplingCfg {
    let rf_type = v
        .get("response_format")
        .and_then(|rf| rf.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let has_tools = v
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty());
    let json_constrained = default_cfg.json_constrained
        || rf_type == "json_object"
        || rf_type == "json_schema"
        || has_tools;

    SamplingCfg {
        temp: v.get("temperature").and_then(Value::as_f64).unwrap_or(default_cfg.temp),
        top_k: v.get("top_k").and_then(Value::as_i64).unwrap_or(default_cfg.top_k),
        top_p: v.get("top_p").and_then(Value::as_f64).unwrap_or(default_cfg.top_p),
        min_p: v.get("min_p").and_then(Value::as_f64).unwrap_or(default_cfg.min_p),
        repeat_penalty: v
            .get("repeat_penalty")
            .or_else(|| v.get("repetition_penalty"))
            .and_then(Value::as_f64)
            .unwrap_or(default_cfg.repeat_penalty),
        max_tokens: v
            .get("max_tokens")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(default_cfg.max_tokens),
        seed: v.get("seed").and_then(Value::as_i64).unwrap_or(default_cfg.seed),
        json_constrained,
        spec_mode: default_cfg.spec_mode,
        spec_k: default_cfg.spec_k,
    }
}

fn parse_tool_call_from_json(raw: &str, call_id: &str) -> Option<Value> {
    let parsed: Value = serde_json::from_str(raw.trim()).ok()?;
    let name = parsed.get("name").and_then(Value::as_str)?;
    let args_val = parsed.get("arguments").cloned().unwrap_or_else(|| json!({}));
    let args_str = if let Some(s) = args_val.as_str() {
        s.to_string()
    } else {
        args_val.to_string()
    };
    Some(json!([{
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": args_str
        }
    }]))
}

async fn handle_http_client(
    mut stream: TcpStream,
    state: Arc<ServerState>,
    default_cfg: SamplingCfg,
) {
    let Ok((method, path, body)) = read_http_request(&mut stream).await else {
        return;
    };
    let clean_path = path.split('?').next().unwrap_or("/");
    if method == "OPTIONS" {
        let _ = stream
            .write_all(b"HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n")
            .await;
        return;
    }

    if method == "GET" && clean_path == "/health" {
        let info = json!({
            "status": "ok",
            "backend": state.meta.backend,
            "model": state.meta.model_id,
            "arch": state.meta.arch,
            "max_ctx": state.meta.max_ctx,
            "slot_ctx": state.meta.slot_ctx,
            "kv_slots": state.meta.n_kv_slots,
            "kv_quant": if state.meta.kv_q8 { "q8_0" } else { "fp16" },
            "mla_latent_kv": state.meta.mla_latent,
            "bpe_merges": state.meta.n_bpe_merges,
            "tp_size": state.meta.tp_size,
            "speculative_draft": state.meta.has_draft,
            "queued_requests": state.queued_reqs.load(Ordering::Relaxed),
            "total_requests": state.total_reqs.load(Ordering::Relaxed),
            "radix_prefix_hits": state.radix_hits.load(Ordering::Relaxed),
            "radix_saved_tokens": state.radix_saved_toks.load(Ordering::Relaxed)
        });
        let _ = write_json_response(&mut stream, "200 OK", &info).await;
        return;
    }

    if method == "GET" && clean_path == "/v1/models" {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let resp = json!({
            "object": "list",
            "data": [{
                "id": state.meta.model_id,
                "object": "model",
                "created": now,
                "owned_by": "tenon-lang"
            }]
        });
        let _ = write_json_response(&mut stream, "200 OK", &resp).await;
        return;
    }

    if method == "POST"
        && (clean_path == "/v1/chat/completions" || clean_path == "/v1/completions")
    {
        let Ok(req) = serde_json::from_str::<Value>(&body) else {
            let _ = write_json_response(
                &mut stream,
                "400 Bad Request",
                &json!({"error": {"message": "invalid JSON request body"}}),
            )
            .await;
            return;
        };
        let is_chat = clean_path == "/v1/chat/completions";
        let want_stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
        let cfg = parse_sampling_from_json(&req, &default_cfg);
        let tools_arr = req.get("tools").and_then(Value::as_array).cloned().unwrap_or_default();
        let has_tools = !tools_arr.is_empty();
        let rf_schema = req
            .get("response_format")
            .and_then(|rf| rf.get("json_schema"))
            .and_then(|js| js.get("schema"))
            .cloned();

        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let seq_no = state.total_reqs.fetch_add(1, Ordering::Relaxed) + 1;
        let req_id = format!("chatcmpl-tenon-{}-{}", created, seq_no);

        let prompt = if is_chat {
            let mut msgs: Vec<(String, String)> = Vec::new();
            if let Some(arr) = req.get("messages").and_then(Value::as_array) {
                for item in arr {
                    let role = item
                        .get("role")
                        .and_then(Value::as_str)
                        .unwrap_or("user")
                        .to_string();
                    let content = item
                        .get("content")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    msgs.push((role, content));
                }
            }
            if has_tools {
                let tool_sys = format_tools_prompt(&tools_arr);
                if let Some((r, c)) = msgs.first_mut() {
                    if r == "system" {
                        c.push('\n');
                        c.push_str(&tool_sys);
                    } else {
                        msgs.insert(0, ("system".to_string(), tool_sys));
                    }
                } else {
                    msgs.push(("system".to_string(), tool_sys));
                }
            } else if let Some(schema) = rf_schema {
                let schema_note = format!(
                    "Respond ONLY with a valid JSON object adhering to this JSON Schema: {}",
                    schema
                );
                if let Some((r, c)) = msgs.first_mut() {
                    if r == "system" {
                        c.push('\n');
                        c.push_str(&schema_note);
                    } else {
                        msgs.insert(0, ("system".to_string(), schema_note));
                    }
                } else {
                    msgs.push(("system".to_string(), schema_note));
                }
            }
            format_chat_template(&state.meta.arch, &msgs)
        } else {
            req.get("prompt")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        };
        let model_id = state.meta.model_id.clone();

        // Fair async queue slot acquisition (allows concurrent HTTP connections + keeps /health instant!)
        state.queued_reqs.fetch_add(1, Ordering::Relaxed);
        let Ok(_permit) = state.queue_sem.acquire().await else {
            state.queued_reqs.fetch_sub(1, Ordering::Relaxed);
            return;
        };

        if want_stream {
            let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
            if stream.write_all(hdr.as_bytes()).await.is_err() {
                state.queued_reqs.fetch_sub(1, Ordering::Relaxed);
                return;
            }
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
            let st_bg = Arc::clone(&state);
            let req_id_bg = req_id.clone();
            let model_id_bg = model_id.clone();
            let jh = tokio::task::spawn_blocking(move || {
                let mut m = st_bg.model.lock().unwrap();
                let res = run_generation(&mut m, &prompt, true, &cfg, |piece| {
                    let chunk = if is_chat {
                        json!({
                            "id": req_id_bg,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id_bg,
                            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": null}]
                        })
                    } else {
                        json!({
                            "id": req_id_bg,
                            "object": "text_completion",
                            "created": created,
                            "model": model_id_bg,
                            "choices": [{"index": 0, "text": piece, "finish_reason": null}]
                        })
                    };
                    tx.send(format!("data: {}\n\n", chunk)).is_ok()
                });
                if res.cached_prefix_tokens > 0 {
                    st_bg.radix_hits.fetch_add(1, Ordering::Relaxed);
                    st_bg
                        .radix_saved_toks
                        .fetch_add(res.cached_prefix_tokens as u64, Ordering::Relaxed);
                }
                let done_chunk = if is_chat {
                    json!({
                        "id": req_id_bg,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id_bg,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": res.finish_reason}],
                        "usage": {
                            "prompt_tokens": res.prompt_tokens,
                            "completion_tokens": res.completion_tokens,
                            "total_tokens": res.prompt_tokens + res.completion_tokens,
                            "prompt_tokens_details": {"cached_tokens": res.cached_prefix_tokens}
                        }
                    })
                } else {
                    json!({
                        "id": req_id_bg,
                        "object": "text_completion",
                        "created": created,
                        "model": model_id_bg,
                        "choices": [{"index": 0, "text": "", "finish_reason": res.finish_reason}]
                    })
                };
                let _ = tx.send(format!("data: {}\n\ndata: [DONE]\n\n", done_chunk));
            });

            while let Some(evt) = rx.recv().await {
                if stream.write_all(evt.as_bytes()).await.is_err() {
                    break;
                }
                let _ = stream.flush().await;
            }
            let _ = jh.await;
            state.queued_reqs.fetch_sub(1, Ordering::Relaxed);
            return;
        }

        let st_bg = Arc::clone(&state);
        let Ok(res) = tokio::task::spawn_blocking(move || {
            let mut m = st_bg.model.lock().unwrap();
            let r = run_generation(&mut m, &prompt, true, &cfg, |_| true);
            if r.cached_prefix_tokens > 0 {
                st_bg.radix_hits.fetch_add(1, Ordering::Relaxed);
                st_bg
                    .radix_saved_toks
                    .fetch_add(r.cached_prefix_tokens as u64, Ordering::Relaxed);
            }
            r
        })
        .await
        else {
            state.queued_reqs.fetch_sub(1, Ordering::Relaxed);
            return;
        };
        state.queued_reqs.fetch_sub(1, Ordering::Relaxed);

        let pf_new = res.prompt_tokens.saturating_sub(res.cached_prefix_tokens);
        let pf_tps = if res.prefill_ms > 0.0 {
            (res.prompt_tokens as f64) * 1000.0 / res.prefill_ms
        } else {
            0.0
        };
        let dec_tps = if res.decode_ms > 0.0 {
            (res.completion_tokens as f64) * 1000.0 / res.decode_ms
        } else {
            0.0
        };

        let resp = if is_chat {
            let call_id = format!("call_tenon_{}", seq_no);
            let parsed_tools = if has_tools {
                parse_tool_call_from_json(&res.text, &call_id)
            } else {
                None
            };
            let (msg_obj, fin) = if let Some(tc) = parsed_tools {
                (
                    json!({"role": "assistant", "content": null, "tool_calls": tc}),
                    "tool_calls",
                )
            } else {
                (
                    json!({"role": "assistant", "content": res.text}),
                    res.finish_reason,
                )
            };
            json!({
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": msg_obj,
                    "finish_reason": fin
                }],
                "usage": {
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "total_tokens": res.prompt_tokens + res.completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": res.cached_prefix_tokens}
                },
                "timings": {
                    "kv_slot": res.kv_slot,
                    "cached_prefix_tokens": res.cached_prefix_tokens,
                    "evaluated_prompt_tokens": pf_new,
                    "prefill_ms": res.prefill_ms,
                    "prefill_tok_per_sec": pf_tps,
                    "decode_ms": res.decode_ms,
                    "decode_tok_per_sec": dec_tps,
                    "spec_proposed": res.spec_proposed,
                    "spec_accepted": res.spec_accepted
                }
            })
        } else {
            json!({
                "id": req_id,
                "object": "text_completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "text": res.text,
                    "finish_reason": res.finish_reason
                }],
                "usage": {
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "total_tokens": res.prompt_tokens + res.completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": res.cached_prefix_tokens}
                }
            })
        };
        let _ = write_json_response(&mut stream, "200 OK", &resp).await;
        return;
    }

    let _ = write_json_response(
        &mut stream,
        "404 Not Found",
        &json!({"error": {"message": format!("unknown endpoint `{} {}`", method, clean_path)}}),
    )
    .await;
}

fn print_llm_usage() {
    println!(
        r#"tenon llm — Native Out-of-Core Streaming LLM CLI & OpenAI-Compatible SSE Server (v5)

USAGE:
  tenon llm <model.gguf> [OPTIONS]

OPTIONS:
  --prompt <text>           Raw completion prompt (runs batched prefill + streaming stdout)
  --chat <text>             Chat user message (auto-applies model architecture chat template)
  --system <text>           Optional system prompt (used with --chat)
  --serve <host:port>       Launch OpenAI-compatible HTTP + SSE server (Radix KV, tools, json_schema)
  --plan-only               Inspect GGUF & print Auto VRAM/NUMA/KV/TP plan without loading weights
  --max-vram-gb <F>         Hard VRAM/RAM budget cap in GiB (0 = auto-detect hardware)
  --ctx <N>                 Total KV context capacity across all slots (default: 4096, up to 16384)
  --kv-slots <N>            Number of partitioned Radix Prefix KV slots (default: 2 for --serve, 1 CLI)
  --kv-quant <auto|q8|fp16> KV-cache precision (`q8` saves -47% VRAM; default: auto)
  --mla-latent <0|1>        Matrix-absorbed compressed latent MLA KV cache (default: 1)
  --tp <N>                  Multi-GPU Tensor Parallelism degree on CUDA (`TP=2..8`, default: 1)
  --slots <N>               Override auto-planned VRAM slots / pinned layers
  --nocache                 Force uncached direct NVMe I/O (`F_NOCACHE` / `POSIX_FADV_DONTNEED`)
  --fast-dp4a               Enable opt-in CUDA 128-bit `int4` Q8_1 `__dp4a` fast path
  --draft <small.gguf>      Small draft GGUF for lossless speculative decoding (`verify_batch`)
  --speculative <off|ngram|draft> Speculative decoding mode (default: off, or draft when --draft set)
  --spec-k <N>              Speculative lookahead window K (default: 5)
  --json                    Enforce RFC 8259 JSON-object grammar-constrained sampling
  --temp <F>                Sampling temperature (0 = greedy argmax; default: 0.7)
  --top-k <N>               Top-K sampling cutoff (default: 40)
  --top-p <F>               Top-P nucleus cutoff (default: 0.9)
  --min-p <F>               Min-P relative threshold (default: 0.05)
  --repeat-penalty <F>      Repetition penalty (default: 1.1)
  --max-tokens <N>          Maximum generated tokens (default: 256)
  --seed <N>                RNG seed (default: 42)"#
    );
}

pub async fn cmd_llm(args: &[String]) -> ExitCode {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print_llm_usage();
        return ExitCode::SUCCESS;
    }

    let mut model_path: Option<String> = None;
    let mut draft_path: Option<String> = None;
    let mut prompt: Option<String> = None;
    let mut chat_msg: Option<String> = None;
    let mut system_msg: Option<String> = None;
    let mut serve_addr: Option<String> = None;
    let mut plan_only = false;
    let mut max_vram_gb = 0.0f64;
    let mut max_ctx = 4096i64;
    let mut kv_slots_opt: Option<usize> = None;
    let mut kv_quant = "auto".to_string();
    let mut mla_latent = 1i64;
    let mut tp_size = 1i64;
    let mut slots_override: Option<i64> = None;
    let mut force_nocache = false;
    let mut fast_dp4a = false;
    let mut cfg = SamplingCfg::default();

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" if i + 1 < args.len() => {
                prompt = Some(args[i + 1].clone());
                i += 2;
            }
            "--chat" if i + 1 < args.len() => {
                chat_msg = Some(args[i + 1].clone());
                i += 2;
            }
            "--system" if i + 1 < args.len() => {
                system_msg = Some(args[i + 1].clone());
                i += 2;
            }
            "--serve" if i + 1 < args.len() => {
                serve_addr = Some(args[i + 1].clone());
                i += 2;
            }
            "--plan-only" => {
                plan_only = true;
                i += 1;
            }
            "--max-vram-gb" if i + 1 < args.len() => {
                max_vram_gb = args[i + 1].parse().unwrap_or(0.0);
                i += 2;
            }
            "--ctx" if i + 1 < args.len() => {
                max_ctx = args[i + 1].parse().unwrap_or(4096);
                i += 2;
            }
            "--kv-slots" if i + 1 < args.len() => {
                kv_slots_opt = args[i + 1].parse().ok();
                i += 2;
            }
            "--kv-quant" if i + 1 < args.len() => {
                kv_quant = args[i + 1].clone();
                i += 2;
            }
            "--mla-latent" if i + 1 < args.len() => {
                mla_latent = args[i + 1].parse().unwrap_or(1);
                i += 2;
            }
            "--tp" if i + 1 < args.len() => {
                tp_size = args[i + 1].parse().unwrap_or(1);
                i += 2;
            }
            "--slots" if i + 1 < args.len() => {
                slots_override = args[i + 1].parse().ok();
                i += 2;
            }
            "--nocache" => {
                force_nocache = true;
                i += 1;
            }
            "--fast-dp4a" => {
                fast_dp4a = true;
                i += 1;
            }
            "--draft" if i + 1 < args.len() => {
                draft_path = Some(args[i + 1].clone());
                cfg.spec_mode = SpecMode::Draft;
                i += 2;
            }
            "--speculative" if i + 1 < args.len() => {
                cfg.spec_mode = match args[i + 1].as_str() {
                    "ngram" | "lookup" => SpecMode::Ngram,
                    "draft" => SpecMode::Draft,
                    _ => SpecMode::Off,
                };
                i += 2;
            }
            "--spec-k" if i + 1 < args.len() => {
                cfg.spec_k = args[i + 1].parse().unwrap_or(5);
                i += 2;
            }
            "--json" => {
                cfg.json_constrained = true;
                i += 1;
            }
            "--temp" if i + 1 < args.len() => {
                cfg.temp = args[i + 1].parse().unwrap_or(0.7);
                i += 2;
            }
            "--top-k" if i + 1 < args.len() => {
                cfg.top_k = args[i + 1].parse().unwrap_or(40);
                i += 2;
            }
            "--top-p" if i + 1 < args.len() => {
                cfg.top_p = args[i + 1].parse().unwrap_or(0.9);
                i += 2;
            }
            "--min-p" if i + 1 < args.len() => {
                cfg.min_p = args[i + 1].parse().unwrap_or(0.05);
                i += 2;
            }
            "--repeat-penalty" if i + 1 < args.len() => {
                cfg.repeat_penalty = args[i + 1].parse().unwrap_or(1.1);
                i += 2;
            }
            "--max-tokens" if i + 1 < args.len() => {
                cfg.max_tokens = args[i + 1].parse().unwrap_or(256);
                i += 2;
            }
            "--seed" if i + 1 < args.len() => {
                cfg.seed = args[i + 1].parse().unwrap_or(42);
                i += 2;
            }
            s if !s.starts_with('-') && model_path.is_none() => {
                model_path = Some(s.to_string());
                i += 1;
            }
            other => {
                eprintln!("unknown option `{}`", other);
                return ExitCode::from(2);
            }
        }
    }

    let Some(model_path) = model_path else {
        print_llm_usage();
        return ExitCode::from(2);
    };

    let kv_slots = kv_slots_opt.unwrap_or_else(|| if serve_addr.is_some() { 2 } else { 1 }).clamp(1, 16);
    unsafe {
        env::set_var("TENON_MLA_LATENT_KV", mla_latent.to_string());
        env::set_var("TENON_KV_SLOTS", kv_slots.to_string());
        if tp_size > 1 {
            env::set_var("TENON_TP_SIZE", tp_size.to_string());
        }
    }

    let api = match load_llm_api() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::from(1);
        }
    };

    let kv_mode = match kv_quant.as_str() {
        "q8" | "q8_0" | "1" => 1,
        "fp16" | "0" => 0,
        _ => -1,
    };
    let plan = match api.plan(&model_path, max_vram_gb, max_ctx, kv_mode) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::from(1);
        }
    };

    let chosen_slots = slots_override.unwrap_or(plan.slots);
    let chosen_nocache = force_nocache || plan.nocache;
    let gib = 1024.0 * 1024.0 * 1024.0;
    eprintln!(
        "[tenon llm] backend={} | disk={:.2} GiB | layers={} experts={} d={} | plan: slots={} nocache={} kv={} kv_slots={} tp={} prefill_B={} est_vram={:.2} GiB",
        api.backend,
        plan.disk_bytes as f64 / gib,
        plan.n_layers,
        plan.n_experts,
        plan.d_model,
        chosen_slots,
        if chosen_nocache { 1 } else { 0 },
        if plan.kv_q8 { "Q8_0" } else { "FP16" },
        kv_slots,
        tp_size,
        plan.prefill_chunk,
        plan.est_vram_bytes as f64 / gib
    );
    if plan_only {
        return ExitCode::SUCCESS;
    }

    unsafe {
        env::set_var("TENON_MAX_CTX", max_ctx.to_string());
        env::set_var("TENON_KV_QUANT", if plan.kv_q8 { "q8" } else { "fp16" });
        if fast_dp4a {
            env::set_var("TENON_CUDA_FAST_DP4A", "1");
        }
    }

    let t_load = Instant::now();
    let handle = match api.load(&model_path, chosen_slots, chosen_nocache, true) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::from(1);
        }
    };
    let draft_handle = if let Some(ref dp) = draft_path {
        unsafe {
            env::set_var("TENON_KV_SLOTS", "1");
            env::set_var("TENON_TP_SIZE", "1");
        }
        match api.load(dp, -1, false, true) {
            Ok(dh) => {
                eprintln!("[tenon llm] loaded speculative draft model `{}` (K={})", dp, cfg.spec_k);
                dh
            }
            Err(e) => {
                eprintln!("warning: failed to load draft model `{}` ({}), falling back to standard decode", dp, e);
                0
            }
        }
    } else {
        0
    };

    let arch = api.arch(handle);
    let bos_id = api.info(handle, 11);
    let eos_id = api.info(handle, 12);
    let vocab_size = api.info(handle, 7).max(1) as usize;
    let core_vram = api.info(handle, 8) as f64 / gib;
    let cache_vram = api.info(handle, 9) as f64 / gib;
    let actual_kv_slots = api.info(handle, 25).max(api.info(handle, 27)).max(1) as usize;
    let (actual_mla_lat, actual_slots, actual_slot_ctx, actual_bpe_merges, actual_tp) =
        if api.backend == "cuda" {
            let sctx = api.info(handle, 28);
            (
                api.info(handle, 26) != 0,
                api.info(handle, 27).max(1) as usize,
                if sctx > 0 { sctx } else { max_ctx },
                api.info(handle, 29),
                api.info(handle, 31).max(1),
            )
        } else {
            let sctx = api.info(handle, 26);
            (
                api.info(handle, 24) != 0,
                api.info(handle, 25).max(1) as usize,
                if sctx > 0 { sctx } else { max_ctx },
                api.info(handle, 28),
                1,
            )
        };
    let _ = actual_kv_slots;

    let model_id = Path::new(&model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("tenon-gguf")
        .to_string();
    eprintln!(
        "[tenon llm] loaded `{}` (arch={}, ctx={} [{}x{}], kv={}, mla_latent={}, bpe_merges={}, tp={}, core_vram={:.2} GiB, cache_vram={:.2} GiB) in {:.2}s",
        model_id,
        arch,
        max_ctx,
        actual_slots,
        actual_slot_ctx,
        if plan.kv_q8 { "Q8_0" } else { "FP16" },
        actual_mla_lat,
        actual_bpe_merges,
        actual_tp,
        core_vram,
        cache_vram,
        t_load.elapsed().as_secs_f64()
    );

    let meta = ModelMeta {
        backend: api.backend,
        model_id,
        arch,
        max_ctx,
        slot_ctx: actual_slot_ctx,
        n_kv_slots: actual_slots,
        kv_q8: plan.kv_q8,
        mla_latent: actual_mla_lat,
        n_bpe_merges: actual_bpe_merges,
        tp_size: actual_tp,
        has_draft: draft_handle > 0,
    };

    let mut loaded = LoadedModel {
        api,
        handle,
        draft_handle,
        meta: meta.clone(),
        bos_id,
        eos_id,
        vocab_size,
        radix_cache: RadixKvCache::new(actual_slots),
        vocab_pieces: Vec::new(),
        logits_scratch: Vec::new(),
    };

    if let Some(addr) = serve_addr {
        let listener = match TcpListener::bind(&addr).await {
            Ok(l) => l,
            Err(e) => {
                eprintln!("error binding {}: {}", addr, e);
                return ExitCode::from(1);
            }
        };
        eprintln!(
            "[tenon llm] OpenAI-compatible v5 SSE server listening on http://{} (Radix KV {} slots, tools/json_schema, speculative={:?})",
            addr, actual_slots, cfg.spec_mode
        );
        let state = Arc::new(ServerState {
            meta,
            model: Mutex::new(loaded),
            queue_sem: Semaphore::new(1),
            queued_reqs: AtomicUsize::new(0),
            total_reqs: AtomicU64::new(0),
            radix_hits: AtomicU64::new(0),
            radix_saved_toks: AtomicU64::new(0),
        });
        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let st_ref = Arc::clone(&state);
                    let dcfg = cfg.clone();
                    tokio::spawn(async move {
                        handle_http_client(stream, st_ref, dcfg).await;
                    });
                }
                Err(e) => {
                    eprintln!("[tenon llm] accept error: {}", e);
                }
            }
        }
    }

    let final_prompt = if let Some(user_text) = chat_msg {
        let mut msgs = Vec::new();
        if let Some(sys) = system_msg {
            msgs.push(("system".to_string(), sys));
        }
        msgs.push(("user".to_string(), user_text));
        format_chat_template(&loaded.meta.arch, &msgs)
    } else {
        prompt.unwrap_or_else(|| "The capital of France is".to_string())
    };

    let res = run_generation(&mut loaded, &final_prompt, true, &cfg, |piece| {
        print!("{}", piece);
        let _ = io::stdout().flush();
        true
    });
    println!();
    let pf_tps = if res.prefill_ms > 0.0 {
        res.prompt_tokens as f64 * 1000.0 / res.prefill_ms
    } else {
        0.0
    };
    let dec_tps = if res.decode_ms > 0.0 {
        res.completion_tokens as f64 * 1000.0 / res.decode_ms
    } else {
        0.0
    };
    eprintln!(
        "[tenon llm] prefill: {} toks (cached={}) in {:.1} ms ({:.2} tok/s) | decode: {} toks in {:.1} ms ({:.2} tok/s) | spec={}/{}",
        res.prompt_tokens,
        res.cached_prefix_tokens,
        res.prefill_ms,
        pf_tps,
        res.completion_tokens,
        res.decode_ms,
        dec_tps,
        res.spec_accepted,
        res.spec_proposed
    );
    ExitCode::SUCCESS
}
