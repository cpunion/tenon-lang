use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, mpsc as std_mpsc};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

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
type FnCopyLogits2 = unsafe extern "C" fn(i64, i64) -> i64;
type FnCopyLogits3 = unsafe extern "C" fn(i64, i64, i64) -> i64;
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
type FnStep = unsafe extern "C" fn(i64, i64, i64) -> i64;

// Tenon v6 FFI function signatures (identical across Metal & CUDA!)
type FnPagedKvInfo = unsafe extern "C" fn(i64, i64) -> i64;
type FnPagedSeqFree = unsafe extern "C" fn(i64, i64) -> i64;
type FnPagedSeqFork = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;
type FnPagedPrefill = unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64;
type FnStepBatch = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64;
type FnCopyLogitsBatch = unsafe extern "C" fn(i64, i64, i64) -> i64;
type FnVerifyTree = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64;
type FnCommitTreePath = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;
type FnMtpPredict = unsafe extern "C" fn(i64, i64, i64) -> i64;
type FnEmbedText = unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64;
type FnStepEmbd = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;
type FnMmprojLoad = unsafe extern "C" fn(i64) -> i64;
type FnMmprojInfo = unsafe extern "C" fn(i64, i64) -> i64;
type FnMmprojEncodeRgb = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64;
type FnMmprojFree = unsafe extern "C" fn(i64) -> i64;

// Tenon v7 FFI function signatures (identical across Metal & CUDA!)
type FnMmprojEncodeAnyres = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64;
type FnSetMropePos = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;
type FnLoraLoad = unsafe extern "C" fn(i64, i64, f64) -> i64;
type FnLoraSet = unsafe extern "C" fn(i64, i64, f64) -> i64;
type FnLoraFree = unsafe extern "C" fn(i64, i64) -> i64;
type FnLoraInfo = unsafe extern "C" fn(i64, i64, i64) -> i64;

// Tenon v8 FFI function signatures (identical across Metal & CUDA!)
type FnPagedVerifyBatch = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64;
type FnStepEmbdMrope = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64;
type FnAudioLoad = unsafe extern "C" fn(i64) -> i64;
type FnAudioInfo = unsafe extern "C" fn(i64, i64) -> i64;
type FnAudioEncodePcm16 = unsafe extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64;
type FnAudioFree = unsafe extern "C" fn(i64) -> i64;

struct LlmApi {
    backend: &'static str,
    _dl_handle: *mut c_void,
    init_fn: Option<FnInit>,
    load_gguf: FnLoadGguf,
    info: FnInfo,
    reset: FnReset,
    kv_slot_set: FnKvSlotSet,
    copy_logits_2: Option<FnCopyLogits2>,
    copy_logits_3: Option<FnCopyLogits3>,
    step_fn: FnStep,
    prefill: FnPrefill,
    verify_batch: FnVerifyBatch,
    auto_plan: FnAutoPlan,
    sample_next: FnSampleNext,
    get_arch: FnGetArch,
    tokenize: FnTokenize,
    token_cstr: FnTokenCstr,
    free: FnFree,
    // v6 APIs
    paged_kv_info: FnPagedKvInfo,
    paged_seq_free: FnPagedSeqFree,
    paged_seq_fork: FnPagedSeqFork,
    paged_prefill: FnPagedPrefill,
    step_batch: FnStepBatch,
    copy_logits_batch: FnCopyLogitsBatch,
    verify_tree: FnVerifyTree,
    commit_tree_path: FnCommitTreePath,
    mtp_predict: FnMtpPredict,
    embed_text: FnEmbedText,
    step_embd: FnStepEmbd,
    mmproj_load: FnMmprojLoad,
    mmproj_info: FnMmprojInfo,
    mmproj_encode_rgb: FnMmprojEncodeRgb,
    mmproj_free: FnMmprojFree,
    // v7 APIs
    mmproj_encode_anyres: FnMmprojEncodeAnyres,
    set_mrope_pos: FnSetMropePos,
    lora_load: FnLoraLoad,
    lora_set: FnLoraSet,
    lora_free: FnLoraFree,
    lora_info: FnLoraInfo,
    // v8 APIs
    paged_verify_batch: FnPagedVerifyBatch,
    step_embd_mrope: FnStepEmbdMrope,
    audio_load: FnAudioLoad,
    audio_info: FnAudioInfo,
    audio_encode_pcm16: FnAudioEncodePcm16,
    audio_free: FnAudioFree,
}

unsafe impl Send for LlmApi {}
unsafe impl Sync for LlmApi {}

#[derive(Debug, Clone)]
struct AutoPlan {
    slots: i64,
    nocache: bool,
    kv_q8: bool,
    #[allow(dead_code)]
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
    Mtp,
    Tree,
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
    json_schema: Option<Value>,
    gbnf_grammar: Option<String>,
    spec_mode: SpecMode,
    spec_k: usize,
    lora: Option<String>,
    anyres: bool,
    mrope_grid: Option<(i64, i64, i64)>,
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
            json_schema: None,
            gbnf_grammar: None,
            spec_mode: SpecMode::Off,
            spec_k: 5,
            lora: None,
            anyres: false,
            mrope_grid: None,
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
    let lib_path = PathBuf::from(format!("/tmp/tenon_llm_{}_v8_{}.{}", backend, mtime, ext));

    if !lib_path.exists() {
        eprintln!(
            "[tenon llm] compiling {} v8 runtime shim ({}) -> {}",
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
    let (copy_logits_2, copy_logits_3) = if backend == "cuda" {
        (None, Some(load_symbol::<FnCopyLogits3>(dl_handle, &sym("copy_logits"))?))
    } else {
        (Some(load_symbol::<FnCopyLogits2>(dl_handle, &sym("copy_logits"))?), None)
    };

    let api = LlmApi {
        backend,
        _dl_handle: dl_handle,
        init_fn,
        load_gguf: load_symbol(dl_handle, &sym("load_gguf"))?,
        info: load_symbol(dl_handle, &sym("info"))?,
        reset: load_symbol(dl_handle, &sym("reset"))?,
        kv_slot_set: load_symbol(dl_handle, &sym("kv_slot_set"))?,
        copy_logits_2,
        copy_logits_3,
        step_fn: load_symbol(dl_handle, &sym("step"))?,
        prefill: load_symbol(dl_handle, &sym("prefill"))?,
        verify_batch: load_symbol(dl_handle, &sym("verify_batch"))?,
        auto_plan: load_symbol(dl_handle, &sym("auto_plan"))?,
        sample_next: load_symbol(dl_handle, &sym("sample_next"))?,
        get_arch: load_symbol(dl_handle, &sym("get_arch"))?,
        tokenize: load_symbol(dl_handle, &sym("tokenize"))?,
        token_cstr: load_symbol(dl_handle, &sym("token_cstr"))?,
        free: load_symbol(dl_handle, &sym("free"))?,
        paged_kv_info: load_symbol(dl_handle, &sym("paged_kv_info"))?,
        paged_seq_free: load_symbol(dl_handle, &sym("paged_seq_free"))?,
        paged_seq_fork: load_symbol(dl_handle, &sym("paged_seq_fork"))?,
        paged_prefill: load_symbol(dl_handle, &sym("paged_prefill"))?,
        step_batch: load_symbol(dl_handle, &sym("step_batch"))?,
        copy_logits_batch: load_symbol(dl_handle, &sym("copy_logits_batch"))?,
        verify_tree: load_symbol(dl_handle, &sym("verify_tree"))?,
        commit_tree_path: load_symbol(dl_handle, &sym("commit_tree_path"))?,
        mtp_predict: load_symbol(dl_handle, &sym("mtp_predict"))?,
        embed_text: load_symbol(dl_handle, &sym("embed_text"))?,
        step_embd: load_symbol(dl_handle, &sym("step_embd"))?,
        mmproj_load: load_symbol(dl_handle, &sym("mmproj_load"))?,
        mmproj_info: load_symbol(dl_handle, &sym("mmproj_info"))?,
        mmproj_encode_rgb: load_symbol(dl_handle, &sym("mmproj_encode_rgb"))?,
        mmproj_free: load_symbol(dl_handle, &sym("mmproj_free"))?,
        mmproj_encode_anyres: load_symbol(dl_handle, &sym("mmproj_encode_anyres"))?,
        set_mrope_pos: load_symbol(dl_handle, &sym("set_mrope_pos"))?,
        lora_load: load_symbol(dl_handle, &sym("lora_load"))?,
        lora_set: load_symbol(dl_handle, &sym("lora_set"))?,
        lora_free: load_symbol(dl_handle, &sym("lora_free"))?,
        lora_info: load_symbol(dl_handle, &sym("lora_info"))?,
        paged_verify_batch: load_symbol(dl_handle, &sym("paged_verify_batch"))?,
        step_embd_mrope: load_symbol(dl_handle, &sym("step_embd_mrope"))?,
        audio_load: load_symbol(dl_handle, &sym("audio_load"))?,
        audio_info: load_symbol(dl_handle, &sym("audio_info"))?,
        audio_encode_pcm16: load_symbol(dl_handle, &sym("audio_encode_pcm16"))?,
        audio_free: load_symbol(dl_handle, &sym("audio_free"))?,
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
        let n = if let Some(f3) = self.copy_logits_3 {
            unsafe { f3(h, out.as_mut_ptr() as i64, out.len() as i64) }
        } else if let Some(f2) = self.copy_logits_2 {
            unsafe { f2(h, out.as_mut_ptr() as i64) }
        } else {
            0
        };
        n.max(0) as usize
    }

    fn step(&self, h: i64, tok_id: i64, pos: i64) -> i64 {
        unsafe { (self.step_fn)(h, tok_id, pos) }
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

    fn paged_kv_info(&self, h: i64, field: i64) -> i64 {
        unsafe { (self.paged_kv_info)(h, field) }
    }

    fn paged_seq_free(&self, h: i64, slot: usize) -> i64 {
        unsafe { (self.paged_seq_free)(h, slot as i64) }
    }

    fn paged_seq_fork(&self, h: i64, src_slot: usize, dst_slot: usize, prefix_len: usize) -> i64 {
        unsafe { (self.paged_seq_fork)(h, src_slot as i64, dst_slot as i64, prefix_len as i64) }
    }

    fn paged_prefill(&self, h: i64, slot: usize, ids: &[i64], start_pos: i64) -> i64 {
        unsafe { (self.paged_prefill)(h, slot as i64, ids.as_ptr() as i64, ids.len() as i64, start_pos) }
    }

    fn step_batch(
        &self,
        h: i64,
        ids: &[i64],
        pos: &[i64],
        slots: &[i64],
        out_toks: &mut [i64],
    ) -> usize {
        let n = unsafe {
            (self.step_batch)(
                h,
                ids.as_ptr() as i64,
                pos.as_ptr() as i64,
                slots.as_ptr() as i64,
                ids.len() as i64,
                out_toks.as_mut_ptr() as i64,
            )
        };
        n.max(0) as usize
    }

    fn copy_logits_batch(&self, h: i64, batch_idx: usize, dst: &mut [f32]) -> usize {
        let n = unsafe { (self.copy_logits_batch)(h, batch_idx as i64, dst.as_mut_ptr() as i64) };
        n.max(0) as usize
    }

    fn verify_tree(
        &self,
        h: i64,
        ids: &[i64],
        parents: &[i64],
        start_pos: i64,
        out_toks: &mut [i64],
    ) -> usize {
        let n = unsafe {
            (self.verify_tree)(
                h,
                ids.as_ptr() as i64,
                parents.as_ptr() as i64,
                ids.len() as i64,
                start_pos,
                out_toks.as_mut_ptr() as i64,
            )
        };
        n.max(0) as usize
    }

    fn commit_tree_path(&self, h: i64, nodes: &[i64], start_pos: i64) -> i64 {
        unsafe { (self.commit_tree_path)(h, nodes.as_ptr() as i64, nodes.len() as i64, start_pos) }
    }

    fn mtp_predict(&self, h: i64, tok_id: i64, pos: i64) -> i64 {
        unsafe { (self.mtp_predict)(h, tok_id, pos) }
    }

    fn embed_text(&self, h: i64, ids: &[i64], out: &mut [f32]) -> usize {
        let n = unsafe {
            (self.embed_text)(
                h,
                ids.as_ptr() as i64,
                ids.len() as i64,
                out.as_mut_ptr() as i64,
                out.len() as i64,
            )
        };
        n.max(0) as usize
    }

    fn step_embd(&self, h: i64, vecs: &[f32], n_vecs: usize, start_pos: i64) -> i64 {
        unsafe { (self.step_embd)(h, vecs.as_ptr() as i64, n_vecs as i64, start_pos) }
    }

    fn mmproj_load(&self, path: &str) -> Result<i64, String> {
        let cp = CString::new(path).map_err(|_| "invalid mmproj path")?;
        let h = unsafe { (self.mmproj_load)(cp.as_ptr() as i64) };
        if h <= 0 {
            return Err(format!("failed to load mmproj GGUF `{}`", path));
        }
        Ok(h)
    }

    fn mmproj_info(&self, mm_h: i64, field: i64) -> i64 {
        unsafe { (self.mmproj_info)(mm_h, field) }
    }

    fn mmproj_encode_rgb(&self, mm_h: i64, rgb: &[u8], w: usize, h: usize, out: &mut [f32], max_vecs: usize) -> usize {
        let n = unsafe {
            (self.mmproj_encode_rgb)(
                mm_h,
                rgb.as_ptr() as i64,
                w as i64,
                h as i64,
                out.as_mut_ptr() as i64,
                max_vecs as i64,
            )
        };
        n.max(0) as usize
    }

    fn mmproj_free(&self, mm_h: i64) {
        unsafe {
            (self.mmproj_free)(mm_h);
        }
    }

    fn mmproj_encode_anyres(
        &self,
        mm_h: i64,
        rgb: &[u8],
        w: usize,
        h: usize,
        max_tiles: usize,
        out: &mut [f32],
        max_vecs: usize,
    ) -> usize {
        let n = unsafe {
            (self.mmproj_encode_anyres)(
                mm_h,
                rgb.as_ptr() as i64,
                w as i64,
                h as i64,
                max_tiles as i64,
                out.as_mut_ptr() as i64,
                max_vecs as i64,
            )
        };
        n.max(0) as usize
    }

    #[allow(dead_code)]
    fn set_mrope_pos(&self, h: i64, pos_t: i64, pos_h: i64, pos_w: i64) -> i64 {
        unsafe { (self.set_mrope_pos)(h, pos_t, pos_h, pos_w) }
    }

    fn lora_load(&self, h: i64, path: &str, scale: f64) -> Result<i64, String> {
        let cp = CString::new(path).map_err(|_| "invalid lora path")?;
        let id = unsafe { (self.lora_load)(h, cp.as_ptr() as i64, scale) };
        if id <= 0 {
            return Err(format!("failed to load GGUF LoRA adapter `{}`", path));
        }
        Ok(id)
    }

    fn lora_set(&self, h: i64, lora_id: i64, scale: f64) -> i64 {
        unsafe { (self.lora_set)(h, lora_id, scale) }
    }

    fn lora_free(&self, h: i64, lora_id: i64) -> i64 {
        unsafe { (self.lora_free)(h, lora_id) }
    }

    fn lora_info(&self, h: i64, lora_id: i64, field: i64) -> i64 {
        unsafe { (self.lora_info)(h, lora_id, field) }
    }

    fn paged_verify_batch(
        &self,
        h: i64,
        slot: usize,
        ids: &[i64],
        start_pos: i64,
        out_argmax: &mut [i64],
    ) -> usize {
        let n = unsafe {
            (self.paged_verify_batch)(
                h,
                slot as i64,
                ids.as_ptr() as i64,
                ids.len() as i64,
                start_pos,
                out_argmax.as_mut_ptr() as i64,
            )
        };
        n.max(0) as usize
    }

    fn step_embd_mrope(
        &self,
        h: i64,
        vecs: &[f32],
        n_vecs: usize,
        start_pos: i64,
        grid_t: i64,
        grid_h: i64,
        grid_w: i64,
    ) -> i64 {
        unsafe {
            (self.step_embd_mrope)(
                h,
                vecs.as_ptr() as i64,
                n_vecs as i64,
                start_pos,
                grid_t,
                grid_h,
                grid_w,
            )
        }
    }

    fn audio_load(&self, path: &str) -> Result<i64, String> {
        let cp = CString::new(path).map_err(|_| "invalid audio encoder path")?;
        let h = unsafe { (self.audio_load)(cp.as_ptr() as i64) };
        if h <= 0 {
            return Err(format!("failed to load audio encoder `{}`", path));
        }
        Ok(h)
    }

    fn audio_info(&self, audio_h: i64, field: i64) -> i64 {
        unsafe { (self.audio_info)(audio_h, field) }
    }

    fn audio_encode_pcm16(
        &self,
        audio_h: i64,
        pcm16: &[i16],
        sample_rate: u32,
        out: &mut [f32],
        max_vecs: usize,
    ) -> usize {
        let n = unsafe {
            (self.audio_encode_pcm16)(
                audio_h,
                pcm16.as_ptr() as i64,
                pcm16.len() as i64,
                sample_rate as i64,
                out.as_mut_ptr() as i64,
                max_vecs as i64,
            )
        };
        n.max(0) as usize
    }

    fn audio_free(&self, audio_h: i64) {
        unsafe {
            (self.audio_free)(audio_h);
        }
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
// Tenon v6: Full Recursive JSON Schema + GBNF Pushdown Automaton Grammar Compiler
// ============================================================================
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CharClass {
    ranges: Vec<(u8, u8)>,
    negated: bool,
}

impl CharClass {
    fn matches(&self, b: u8) -> bool {
        let hit = self.ranges.iter().any(|&(lo, hi)| b >= lo && b <= hi);
        if self.negated { !hit } else { hit }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum GbnfAtom {
    Byte(u8),
    Class(usize),
    Rule(usize),
}

#[derive(Debug, Clone)]
struct GbnfGrammar {
    rules: Vec<Vec<Vec<GbnfAtom>>>,
    classes: Vec<CharClass>,
    root: usize,
}

impl GbnfGrammar {
    fn new() -> Self {
        Self {
            rules: Vec::new(),
            classes: Vec::new(),
            root: 0,
        }
    }

    fn add_rule(&mut self, alts: Vec<Vec<GbnfAtom>>) -> usize {
        let id = self.rules.len();
        self.rules.push(alts);
        id
    }

    fn reserve_rule(&mut self) -> usize {
        let id = self.rules.len();
        self.rules.push(Vec::new());
        id
    }

    fn add_class(&mut self, ranges: Vec<(u8, u8)>, negated: bool) -> usize {
        let cc = CharClass { ranges, negated };
        if let Some(pos) = self.classes.iter().position(|c| c == &cc) {
            return pos;
        }
        let id = self.classes.len();
        self.classes.push(cc);
        id
    }

    fn lit_atoms(s: &str) -> Vec<GbnfAtom> {
        s.bytes().map(GbnfAtom::Byte).collect()
    }

    fn make_opt(&mut self, seq: Vec<GbnfAtom>) -> usize {
        self.add_rule(vec![vec![], seq])
    }

    fn make_star(&mut self, seq: Vec<GbnfAtom>) -> usize {
        let id = self.reserve_rule();
        let mut rec = seq;
        rec.push(GbnfAtom::Rule(id));
        self.rules[id] = vec![vec![], rec];
        id
    }

    fn make_plus(&mut self, seq: Vec<GbnfAtom>) -> usize {
        let star = self.make_star(seq.clone());
        let mut alt = seq;
        alt.push(GbnfAtom::Rule(star));
        self.add_rule(vec![alt])
    }

    /// Parse standard llama.cpp GBNF grammar (`root ::= ...`) into a runnable PDA grammar!
    fn parse_gbnf(src: &str) -> Result<Self, String> {
        let mut g = GbnfGrammar::new();
        let mut name_to_id: HashMap<String, usize> = HashMap::new();

        // Pre-scan rule names (`name ::=`) so forward & mutually recursive references resolve cleanly.
        let mut raw_rules: Vec<(String, String)> = Vec::new();
        let mut cur_name: Option<String> = None;
        let mut cur_rhs = String::new();
        for raw_line in src.lines() {
            // Strip comments outside quotes/brackets
            let mut line = String::new();
            let mut in_q = false;
            let mut in_br = false;
            let mut esc = false;
            for ch in raw_line.chars() {
                if esc {
                    line.push(ch);
                    esc = false;
                    continue;
                }
                if ch == '\\' && (in_q || in_br) {
                    line.push(ch);
                    esc = true;
                    continue;
                }
                if ch == '"' && !in_br {
                    in_q = !in_q;
                } else if ch == '[' && !in_q {
                    in_br = true;
                } else if ch == ']' && !in_q {
                    in_br = false;
                } else if ch == '#' && !in_q && !in_br {
                    break;
                }
                line.push(ch);
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(pos) = trimmed.find("::=") {
                if let Some(n) = cur_name.take() {
                    raw_rules.push((n, cur_rhs.clone()));
                    cur_rhs.clear();
                }
                let nm = trimmed[..pos].trim().to_string();
                let rhs = trimmed[pos + 3..].trim().to_string();
                cur_name = Some(nm);
                cur_rhs = rhs;
            } else if cur_name.is_some() {
                cur_rhs.push(' ');
                cur_rhs.push_str(trimmed);
            }
        }
        if let Some(n) = cur_name.take() {
            raw_rules.push((n, cur_rhs));
        }
        if raw_rules.is_empty() {
            return Err("empty GBNF grammar".into());
        }
        for (nm, _) in &raw_rules {
            if !name_to_id.contains_key(nm) {
                let id = g.reserve_rule();
                name_to_id.insert(nm.clone(), id);
            }
        }
        g.root = *name_to_id.get("root").unwrap_or(&0);

        for (nm, rhs) in raw_rules {
            let rid = name_to_id[&nm];
            let mut p = GbnfParser {
                b: rhs.as_bytes(),
                pos: 0,
                g: &mut g,
                names: &mut name_to_id,
            };
            let alts = p.parse_alts()?;
            g.rules[rid] = alts;
        }
        Ok(g)
    }

    /// Full Recursive JSON Schema -> GBNF Grammar Compiler!
    /// Supports nested `object`/`properties`/`required`/`additionalProperties`, `array`/`items`/`minItems`/`maxItems`,
    /// `string`/`enum`/`const`/`pattern` (regex!), `integer`, `number`, `boolean`, `null`, `$ref`/`$defs`, `anyOf`/`oneOf`!
    fn from_json_schema(schema: &Value) -> Result<Self, String> {
        let mut b = JsonSchemaBuilder::new(schema);
        let root_id = b.compile(schema)?;
        let ws = b.ws;
        b.g.root = b.g.add_rule(vec![vec![GbnfAtom::Rule(ws), GbnfAtom::Rule(root_id), GbnfAtom::Rule(ws)]]);
        Ok(b.g)
    }

    /// Generic RFC 8259 JSON Object grammar (for `response_format: {type: "json_object"}` or `--json`).
    fn json_object() -> Self {
        Self::from_json_schema(&json!({"type": "object"})).expect("json_object schema")
    }
}

struct GbnfParser<'a> {
    b: &'a [u8],
    pos: usize,
    g: &'a mut GbnfGrammar,
    names: &'a mut HashMap<String, usize>,
}

impl<'a> GbnfParser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.b.len() && self.b[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn parse_escape(&mut self) -> Result<u8, String> {
        if self.pos >= self.b.len() {
            return Err("unterminated escape".into());
        }
        let c = self.b[self.pos];
        self.pos += 1;
        match c {
            b'n' => Ok(b'\n'),
            b'r' => Ok(b'\r'),
            b't' => Ok(b'\t'),
            b'"' => Ok(b'"'),
            b'\\' => Ok(b'\\'),
            b'x' if self.pos + 2 <= self.b.len() => {
                let hex = std::str::from_utf8(&self.b[self.pos..self.pos + 2]).unwrap_or("00");
                self.pos += 2;
                u8::from_str_radix(hex, 16).map_err(|_| "invalid hex escape".into())
            }
            other => Ok(other),
        }
    }

    fn parse_alts(&mut self) -> Result<Vec<Vec<GbnfAtom>>, String> {
        let mut alts = Vec::new();
        loop {
            alts.push(self.parse_seq()?);
            self.skip_ws();
            if self.pos < self.b.len() && self.b[self.pos] == b'|' {
                self.pos += 1;
            } else {
                break;
            }
        }
        Ok(alts)
    }

    fn parse_seq(&mut self) -> Result<Vec<GbnfAtom>, String> {
        let mut seq = Vec::new();
        loop {
            self.skip_ws();
            if self.pos >= self.b.len() || self.b[self.pos] == b'|' || self.b[self.pos] == b')' {
                break;
            }
            let atom_seq = self.parse_primary()?;
            self.skip_ws();
            if self.pos < self.b.len() {
                match self.b[self.pos] {
                    b'?' => {
                        self.pos += 1;
                        let r = self.g.make_opt(atom_seq);
                        seq.push(GbnfAtom::Rule(r));
                        continue;
                    }
                    b'*' => {
                        self.pos += 1;
                        let r = self.g.make_star(atom_seq);
                        seq.push(GbnfAtom::Rule(r));
                        continue;
                    }
                    b'+' => {
                        self.pos += 1;
                        let r = self.g.make_plus(atom_seq);
                        seq.push(GbnfAtom::Rule(r));
                        continue;
                    }
                    _ => {}
                }
            }
            seq.extend(atom_seq);
        }
        Ok(seq)
    }

    fn parse_primary(&mut self) -> Result<Vec<GbnfAtom>, String> {
        self.skip_ws();
        if self.pos >= self.b.len() {
            return Err("unexpected end of GBNF expression".into());
        }
        match self.b[self.pos] {
            b'"' => {
                self.pos += 1;
                let mut out = Vec::new();
                while self.pos < self.b.len() && self.b[self.pos] != b'"' {
                    if self.b[self.pos] == b'\\' {
                        self.pos += 1;
                        out.push(GbnfAtom::Byte(self.parse_escape()?));
                    } else {
                        out.push(GbnfAtom::Byte(self.b[self.pos]));
                        self.pos += 1;
                    }
                }
                if self.pos >= self.b.len() {
                    return Err("unterminated GBNF string literal".into());
                }
                self.pos += 1;
                Ok(out)
            }
            b'[' => {
                self.pos += 1;
                let mut neg = false;
                if self.pos < self.b.len() && self.b[self.pos] == b'^' {
                    neg = true;
                    self.pos += 1;
                }
                let mut ranges = Vec::new();
                while self.pos < self.b.len() && self.b[self.pos] != b']' {
                    let lo = if self.b[self.pos] == b'\\' {
                        self.pos += 1;
                        self.parse_escape()?
                    } else {
                        let c = self.b[self.pos];
                        self.pos += 1;
                        c
                    };
                    if self.pos + 1 < self.b.len() && self.b[self.pos] == b'-' && self.b[self.pos + 1] != b']' {
                        self.pos += 1;
                        let hi = if self.b[self.pos] == b'\\' {
                            self.pos += 1;
                            self.parse_escape()?
                        } else {
                            let c = self.b[self.pos];
                            self.pos += 1;
                            c
                        };
                        ranges.push((lo.min(hi), lo.max(hi)));
                    } else {
                        ranges.push((lo, lo));
                    }
                }
                if self.pos >= self.b.len() {
                    return Err("unterminated GBNF character class".into());
                }
                self.pos += 1;
                let cid = self.g.add_class(ranges, neg);
                Ok(vec![GbnfAtom::Class(cid)])
            }
            b'(' => {
                self.pos += 1;
                let alts = self.parse_alts()?;
                self.skip_ws();
                if self.pos >= self.b.len() || self.b[self.pos] != b')' {
                    return Err("missing closing `)` in GBNF".into());
                }
                self.pos += 1;
                let r = self.g.add_rule(alts);
                Ok(vec![GbnfAtom::Rule(r)])
            }
            _ => {
                let start = self.pos;
                while self.pos < self.b.len()
                    && (self.b[self.pos].is_ascii_alphanumeric()
                        || self.b[self.pos] == b'-'
                        || self.b[self.pos] == b'_')
                {
                    self.pos += 1;
                }
                if self.pos == start {
                    return Err(format!("unexpected byte `{}` in GBNF", self.b[self.pos] as char));
                }
                let name = String::from_utf8_lossy(&self.b[start..self.pos]).into_owned();
                let rid = if let Some(&id) = self.names.get(&name) {
                    id
                } else {
                    let id = self.g.reserve_rule();
                    self.names.insert(name, id);
                    id
                };
                Ok(vec![GbnfAtom::Rule(rid)])
            }
        }
    }
}

struct JsonSchemaBuilder<'a> {
    root_doc: &'a Value,
    g: GbnfGrammar,
    ws: usize,
    j_str: usize,
    j_int: usize,
    j_num: usize,
    j_bool: usize,
    j_null: usize,
    j_val: usize,
    j_obj: usize,
    ref_cache: HashMap<String, usize>,
}

impl<'a> JsonSchemaBuilder<'a> {
    fn new(root_doc: &'a Value) -> Self {
        let mut g = GbnfGrammar::new();
        let ws_cls = g.add_class(vec![(b' ', b' '), (b'\t', b'\t'), (b'\n', b'\n'), (b'\r', b'\r')], false);
        let ws = g.make_star(vec![GbnfAtom::Class(ws_cls)]);

        // JSON string: `"` (plain_char | `\` escape_char)* `"`
        let plain_cls = g.add_class(vec![(0x00, 0x1F), (b'"', b'"'), (b'\\', b'\\')], true);
        let esc_cls = g.add_class(
            vec![
                (b'"', b'"'), (b'\\', b'\\'), (b'/', b'/'), (b'b', b'b'),
                (b'f', b'f'), (b'n', b'n'), (b'r', b'r'), (b't', b't'),
            ],
            false,
        );
        let hex_cls = g.add_class(vec![(b'0', b'9'), (b'a', b'f'), (b'A', b'F')], false);
        let str_char = g.add_rule(vec![
            vec![GbnfAtom::Class(plain_cls)],
            vec![GbnfAtom::Byte(b'\\'), GbnfAtom::Class(esc_cls)],
            vec![
                GbnfAtom::Byte(b'\\'), GbnfAtom::Byte(b'u'),
                GbnfAtom::Class(hex_cls), GbnfAtom::Class(hex_cls),
                GbnfAtom::Class(hex_cls), GbnfAtom::Class(hex_cls),
            ],
        ]);
        let str_chars = g.make_star(vec![GbnfAtom::Rule(str_char)]);
        let j_str = g.add_rule(vec![vec![GbnfAtom::Byte(b'"'), GbnfAtom::Rule(str_chars), GbnfAtom::Byte(b'"')]]);

        // JSON integer: `-?` (`0` | `[1-9][0-9]*`)
        let dig_cls = g.add_class(vec![(b'0', b'9')], false);
        let nz_cls = g.add_class(vec![(b'1', b'9')], false);
        let digs_star = g.make_star(vec![GbnfAtom::Class(dig_cls)]);
        let digs_plus = g.make_plus(vec![GbnfAtom::Class(dig_cls)]);
        let opt_minus = g.make_opt(vec![GbnfAtom::Byte(b'-')]);
        let int_body = g.add_rule(vec![
            vec![GbnfAtom::Byte(b'0')],
            vec![GbnfAtom::Class(nz_cls), GbnfAtom::Rule(digs_star)],
        ]);
        let j_int = g.add_rule(vec![vec![GbnfAtom::Rule(opt_minus), GbnfAtom::Rule(int_body)]]);

        // JSON number: int (`.` `[0-9]+`)? (`[eE]` `[+-]?` `[0-9]+`)?
        let frac = g.make_opt(vec![GbnfAtom::Byte(b'.'), GbnfAtom::Rule(digs_plus)]);
        let exp_ch = g.add_class(vec![(b'e', b'e'), (b'E', b'E')], false);
        let sign_ch = g.add_class(vec![(b'+', b'+'), (b'-', b'-')], false);
        let opt_sign = g.make_opt(vec![GbnfAtom::Class(sign_ch)]);
        let exp_part = g.make_opt(vec![GbnfAtom::Class(exp_ch), GbnfAtom::Rule(opt_sign), GbnfAtom::Rule(digs_plus)]);
        let j_num = g.add_rule(vec![vec![GbnfAtom::Rule(j_int), GbnfAtom::Rule(frac), GbnfAtom::Rule(exp_part)]]);

        let j_bool = g.add_rule(vec![GbnfGrammar::lit_atoms("true"), GbnfGrammar::lit_atoms("false")]);
        let j_null = g.add_rule(vec![GbnfGrammar::lit_atoms("null")]);

        let j_val = g.reserve_rule();
        let j_obj = g.reserve_rule();
        let j_arr = g.reserve_rule();

        let kv_pair = g.add_rule(vec![vec![
            GbnfAtom::Rule(j_str), GbnfAtom::Rule(ws), GbnfAtom::Byte(b':'), GbnfAtom::Rule(ws), GbnfAtom::Rule(j_val),
        ]]);
        let more_kvs = g.make_star(vec![GbnfAtom::Rule(ws), GbnfAtom::Byte(b','), GbnfAtom::Rule(ws), GbnfAtom::Rule(kv_pair)]);
        let obj_body = g.make_opt(vec![GbnfAtom::Rule(kv_pair), GbnfAtom::Rule(more_kvs)]);
        g.rules[j_obj] = vec![vec![
            GbnfAtom::Byte(b'{'), GbnfAtom::Rule(ws), GbnfAtom::Rule(obj_body), GbnfAtom::Rule(ws), GbnfAtom::Byte(b'}'),
        ]];

        let more_vals = g.make_star(vec![GbnfAtom::Rule(ws), GbnfAtom::Byte(b','), GbnfAtom::Rule(ws), GbnfAtom::Rule(j_val)]);
        let arr_body = g.make_opt(vec![GbnfAtom::Rule(j_val), GbnfAtom::Rule(more_vals)]);
        g.rules[j_arr] = vec![vec![
            GbnfAtom::Byte(b'['), GbnfAtom::Rule(ws), GbnfAtom::Rule(arr_body), GbnfAtom::Rule(ws), GbnfAtom::Byte(b']'),
        ]];

        g.rules[j_val] = vec![
            vec![GbnfAtom::Rule(j_obj)],
            vec![GbnfAtom::Rule(j_arr)],
            vec![GbnfAtom::Rule(j_str)],
            vec![GbnfAtom::Rule(j_num)],
            vec![GbnfAtom::Rule(j_bool)],
            vec![GbnfAtom::Rule(j_null)],
        ];

        Self {
            root_doc,
            g,
            ws,
            j_str,
            j_int,
            j_num,
            j_bool,
            j_null,
            j_val,
            j_obj,
            ref_cache: HashMap::new(),
        }
    }

    fn resolve_ref(&self, ref_str: &str) -> Option<&'a Value> {
        let ptr = ref_str.strip_prefix("#/")?;
        let mut cur = self.root_doc;
        for part in ptr.split('/') {
            cur = cur.get(part)?;
        }
        Some(cur)
    }

    fn compile_regex_inner(&mut self, pat: &str) -> Result<usize, String> {
        let mut s = pat;
        if let Some(rest) = s.strip_prefix('^') {
            s = rest;
        }
        if let Some(rest) = s.strip_suffix('$') {
            s = rest;
        }
        let mut rp = RegexCompiler { b: s.as_bytes(), pos: 0, g: &mut self.g };
        let alts = rp.parse_alts()?;
        Ok(self.g.add_rule(alts))
    }

    fn compile(&mut self, s: &Value) -> Result<usize, String> {
        // 1. `$ref` resolution with recursion memoization
        if let Some(r) = s.get("$ref").and_then(Value::as_str) {
            if let Some(&id) = self.ref_cache.get(r) {
                return Ok(id);
            }
            let target = self
                .resolve_ref(r)
                .ok_or_else(|| format!("unresolved JSON Schema $ref `{}`", r))?;
            let id = self.g.reserve_rule();
            self.ref_cache.insert(r.to_string(), id);
            let inner = self.compile(target)?;
            self.g.rules[id] = vec![vec![GbnfAtom::Rule(inner)]];
            return Ok(id);
        }

        // 2. `const`
        if let Some(cv) = s.get("const") {
            let lit = serde_json::to_string(cv).unwrap_or_else(|_| "null".into());
            return Ok(self.g.add_rule(vec![GbnfGrammar::lit_atoms(&lit)]));
        }

        // 3. `enum`
        if let Some(arr) = s.get("enum").and_then(Value::as_array) {
            if !arr.is_empty() {
                let alts: Vec<Vec<GbnfAtom>> = arr
                    .iter()
                    .map(|v| {
                        let lit = serde_json::to_string(v).unwrap_or_else(|_| "null".into());
                        GbnfGrammar::lit_atoms(&lit)
                    })
                    .collect();
                return Ok(self.g.add_rule(alts));
            }
        }

        // 4. `anyOf` / `oneOf`
        if let Some(arr) = s.get("anyOf").or_else(|| s.get("oneOf")).and_then(Value::as_array) {
            if !arr.is_empty() {
                let mut alts = Vec::with_capacity(arr.len());
                for sub in arr {
                    let sub_r = self.compile(sub)?;
                    alts.push(vec![GbnfAtom::Rule(sub_r)]);
                }
                return Ok(self.g.add_rule(alts));
            }
        }

        // 5. Type dispatch (supports string type OR array of types like `["string", "null"]`)
        if let Some(tarr) = s.get("type").and_then(Value::as_array) {
            let mut alts = Vec::new();
            for tv in tarr {
                if let Some(ts) = tv.as_str() {
                    let mut clone = s.clone();
                    clone["type"] = Value::String(ts.to_string());
                    let r = self.compile(&clone)?;
                    alts.push(vec![GbnfAtom::Rule(r)]);
                }
            }
            if !alts.is_empty() {
                return Ok(self.g.add_rule(alts));
            }
        }

        let ty = s.get("type").and_then(Value::as_str).unwrap_or_else(|| {
            if s.get("properties").is_some() {
                "object"
            } else if s.get("items").is_some() {
                "array"
            } else {
                "any"
            }
        });

        match ty {
            "null" => Ok(self.j_null),
            "boolean" => Ok(self.j_bool),
            "integer" => Ok(self.j_int),
            "number" => Ok(self.j_num),
            "string" => {
                if let Some(pat) = s.get("pattern").and_then(Value::as_str) {
                    let body = self.compile_regex_inner(pat)?;
                    Ok(self.g.add_rule(vec![vec![
                        GbnfAtom::Byte(b'"'), GbnfAtom::Rule(body), GbnfAtom::Byte(b'"'),
                    ]]))
                } else {
                    Ok(self.j_str)
                }
            }
            "array" => {
                let item_r = if let Some(it) = s.get("items") {
                    self.compile(it)?
                } else {
                    self.j_val
                };
                let min_items = s.get("minItems").and_then(Value::as_u64).unwrap_or(0) as usize;
                let max_items = s.get("maxItems").and_then(Value::as_u64).map(|x| x as usize);
                let ws = self.ws;
                let mut seq = vec![GbnfAtom::Byte(b'['), GbnfAtom::Rule(ws)];
                if min_items == 0 && max_items.is_none() {
                    let more = self.g.make_star(vec![
                        GbnfAtom::Rule(ws), GbnfAtom::Byte(b','), GbnfAtom::Rule(ws), GbnfAtom::Rule(item_r),
                    ]);
                    let body = self.g.make_opt(vec![GbnfAtom::Rule(item_r), GbnfAtom::Rule(more)]);
                    seq.push(GbnfAtom::Rule(body));
                } else {
                    for idx in 0..min_items {
                        if idx > 0 {
                            seq.extend([GbnfAtom::Rule(ws), GbnfAtom::Byte(b','), GbnfAtom::Rule(ws)]);
                        }
                        seq.push(GbnfAtom::Rule(item_r));
                    }
                    if let Some(mx) = max_items {
                        let extra = mx.saturating_sub(min_items);
                        if min_items == 0 && extra > 0 {
                            let mut opt_chain = self.g.add_rule(vec![vec![]]);
                            for _ in 1..extra {
                                let prev = opt_chain;
                                opt_chain = self.g.make_opt(vec![
                                    GbnfAtom::Rule(ws), GbnfAtom::Byte(b','), GbnfAtom::Rule(ws),
                                    GbnfAtom::Rule(item_r), GbnfAtom::Rule(prev),
                                ]);
                            }
                            let top = self.g.make_opt(vec![GbnfAtom::Rule(item_r), GbnfAtom::Rule(opt_chain)]);
                            seq.push(GbnfAtom::Rule(top));
                        } else {
                            let mut opt_chain = self.g.add_rule(vec![vec![]]);
                            for _ in 0..extra {
                                let prev = opt_chain;
                                opt_chain = self.g.make_opt(vec![
                                    GbnfAtom::Rule(ws), GbnfAtom::Byte(b','), GbnfAtom::Rule(ws),
                                    GbnfAtom::Rule(item_r), GbnfAtom::Rule(prev),
                                ]);
                            }
                            seq.push(GbnfAtom::Rule(opt_chain));
                        }
                    } else {
                        let more = self.g.make_star(vec![
                            GbnfAtom::Rule(ws), GbnfAtom::Byte(b','), GbnfAtom::Rule(ws), GbnfAtom::Rule(item_r),
                        ]);
                        seq.push(GbnfAtom::Rule(more));
                    }
                }
                seq.extend([GbnfAtom::Rule(ws), GbnfAtom::Byte(b']')]);
                Ok(self.g.add_rule(vec![seq]))
            }
            "object" => {
                let Some(props) = s.get("properties").and_then(Value::as_object) else {
                    return Ok(self.j_obj);
                };
                if props.is_empty() {
                    return Ok(self.j_obj);
                }
                let req_list: Vec<&str> = s
                    .get("required")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(Value::as_str).collect())
                    .unwrap_or_default();
                let req_set: HashSet<&str> = req_list.iter().copied().collect();

                // Required properties in declared `required` order first, then optional properties!
                let mut ordered: Vec<(&str, &Value, bool)> = Vec::new();
                for rk in &req_list {
                    if let Some(v) = props.get(*rk) {
                        ordered.push((*rk, v, true));
                    }
                }
                for (k, v) in props {
                    if !req_set.contains(k.as_str()) {
                        ordered.push((k.as_str(), v, false));
                    }
                }
                let mut prop_rules: Vec<(usize, bool)> = Vec::with_capacity(ordered.len());
                let ws = self.ws;
                for (k, v, is_req) in ordered {
                    let vr = self.compile(v)?;
                    let klit = serde_json::to_string(k).unwrap_or_else(|_| format!("\"{}\"", k));
                    let mut pseq = GbnfGrammar::lit_atoms(&klit);
                    pseq.extend([GbnfAtom::Rule(ws), GbnfAtom::Byte(b':'), GbnfAtom::Rule(ws), GbnfAtom::Rule(vr)]);
                    let pr = self.g.add_rule(vec![pseq]);
                    prop_rules.push((pr, is_req));
                }

                // Build exact comma-handling chain over required & optional properties!
                // State `(idx, emitted_any)` -> rule ID!
                let n = prop_rules.len();
                let mut tail_no = self.g.add_rule(vec![vec![]]);
                let mut tail_yes = self.g.add_rule(vec![vec![]]);
                if s.get("additionalProperties").and_then(Value::as_bool) == Some(true) {
                    let kv = self.g.add_rule(vec![vec![
                        GbnfAtom::Rule(self.j_str), GbnfAtom::Rule(ws), GbnfAtom::Byte(b':'), GbnfAtom::Rule(ws), GbnfAtom::Rule(self.j_val),
                    ]]);
                    let more = self.g.make_star(vec![GbnfAtom::Rule(ws), GbnfAtom::Byte(b','), GbnfAtom::Rule(ws), GbnfAtom::Rule(kv)]);
                    tail_no = self.g.make_opt(vec![GbnfAtom::Rule(kv), GbnfAtom::Rule(more)]);
                    tail_yes = more;
                }
                for idx in (0..n).rev() {
                    let (pr, is_req) = prop_rules[idx];
                    let with_no = vec![GbnfAtom::Rule(pr), GbnfAtom::Rule(tail_yes)];
                    let with_yes = vec![GbnfAtom::Rule(ws), GbnfAtom::Byte(b','), GbnfAtom::Rule(ws), GbnfAtom::Rule(pr), GbnfAtom::Rule(tail_yes)];
                    if is_req {
                        tail_no = self.g.add_rule(vec![with_no]);
                        tail_yes = self.g.add_rule(vec![with_yes]);
                    } else {
                        tail_no = self.g.add_rule(vec![with_no, vec![GbnfAtom::Rule(tail_no)]]);
                        tail_yes = self.g.add_rule(vec![with_yes, vec![GbnfAtom::Rule(tail_yes)]]);
                    }
                }
                Ok(self.g.add_rule(vec![vec![
                    GbnfAtom::Byte(b'{'), GbnfAtom::Rule(ws), GbnfAtom::Rule(tail_no), GbnfAtom::Rule(ws), GbnfAtom::Byte(b'}'),
                ]]))
            }
            _ => Ok(self.j_val),
        }
    }
}

struct RegexCompiler<'a> {
    b: &'a [u8],
    pos: usize,
    g: &'a mut GbnfGrammar,
}

impl<'a> RegexCompiler<'a> {
    fn parse_alts(&mut self) -> Result<Vec<Vec<GbnfAtom>>, String> {
        let mut alts = Vec::new();
        loop {
            alts.push(self.parse_seq()?);
            if self.pos < self.b.len() && self.b[self.pos] == b'|' {
                self.pos += 1;
            } else {
                break;
            }
        }
        Ok(alts)
    }

    fn parse_seq(&mut self) -> Result<Vec<GbnfAtom>, String> {
        let mut seq = Vec::new();
        while self.pos < self.b.len() && self.b[self.pos] != b'|' && self.b[self.pos] != b')' {
            let atom = self.parse_atom()?;
            if self.pos < self.b.len() {
                match self.b[self.pos] {
                    b'?' => {
                        self.pos += 1;
                        let r = self.g.make_opt(atom);
                        seq.push(GbnfAtom::Rule(r));
                        continue;
                    }
                    b'*' => {
                        self.pos += 1;
                        let r = self.g.make_star(atom);
                        seq.push(GbnfAtom::Rule(r));
                        continue;
                    }
                    b'+' => {
                        self.pos += 1;
                        let r = self.g.make_plus(atom);
                        seq.push(GbnfAtom::Rule(r));
                        continue;
                    }
                    b'{' => {
                        if let Some((mn, mx)) = self.try_parse_brace() {
                            for _ in 0..mn {
                                seq.extend(atom.iter().copied());
                            }
                            if let Some(m) = mx {
                                for _ in mn..m {
                                    let r = self.g.make_opt(atom.clone());
                                    seq.push(GbnfAtom::Rule(r));
                                }
                            } else {
                                let r = self.g.make_star(atom);
                                seq.push(GbnfAtom::Rule(r));
                            }
                            continue;
                        }
                    }
                    _ => {}
                }
            }
            seq.extend(atom);
        }
        Ok(seq)
    }

    fn try_parse_brace(&mut self) -> Option<(usize, Option<usize>)> {
        let save = self.pos;
        self.pos += 1;
        let s1 = self.pos;
        while self.pos < self.b.len() && self.b[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        let mn: usize = std::str::from_utf8(&self.b[s1..self.pos]).ok()?.parse().ok()?;
        if self.pos < self.b.len() && self.b[self.pos] == b'}' {
            self.pos += 1;
            return Some((mn, Some(mn)));
        }
        if self.pos < self.b.len() && self.b[self.pos] == b',' {
            self.pos += 1;
            let s2 = self.pos;
            while self.pos < self.b.len() && self.b[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
            if self.pos < self.b.len() && self.b[self.pos] == b'}' {
                let mx = if self.pos > s2 {
                    std::str::from_utf8(&self.b[s2..self.pos]).ok()?.parse().ok()
                } else {
                    None
                };
                self.pos += 1;
                return Some((mn, mx));
            }
        }
        self.pos = save;
        None
    }

    fn parse_atom(&mut self) -> Result<Vec<GbnfAtom>, String> {
        match self.b[self.pos] {
            b'(' => {
                self.pos += 1;
                let alts = self.parse_alts()?;
                if self.pos < self.b.len() && self.b[self.pos] == b')' {
                    self.pos += 1;
                }
                let r = self.g.add_rule(alts);
                Ok(vec![GbnfAtom::Rule(r)])
            }
            b'[' => {
                self.pos += 1;
                let mut neg = false;
                if self.pos < self.b.len() && self.b[self.pos] == b'^' {
                    neg = true;
                    self.pos += 1;
                }
                let mut ranges = Vec::new();
                while self.pos < self.b.len() && self.b[self.pos] != b']' {
                    let lo = self.b[self.pos];
                    self.pos += 1;
                    if self.pos + 1 < self.b.len() && self.b[self.pos] == b'-' && self.b[self.pos + 1] != b']' {
                        self.pos += 1;
                        let hi = self.b[self.pos];
                        self.pos += 1;
                        ranges.push((lo.min(hi), lo.max(hi)));
                    } else {
                        ranges.push((lo, lo));
                    }
                }
                if self.pos < self.b.len() {
                    self.pos += 1;
                }
                let cid = self.g.add_class(ranges, neg);
                Ok(vec![GbnfAtom::Class(cid)])
            }
            b'.' => {
                self.pos += 1;
                let cid = self.g.add_class(vec![(0x00, 0x1F), (b'"', b'"'), (b'\\', b'\\')], true);
                Ok(vec![GbnfAtom::Class(cid)])
            }
            b'\\' if self.pos + 1 < self.b.len() => {
                self.pos += 1;
                let c = self.b[self.pos];
                self.pos += 1;
                let cid = match c {
                    b'd' => self.g.add_class(vec![(b'0', b'9')], false),
                    b'w' => self.g.add_class(vec![(b'0', b'9'), (b'a', b'z'), (b'A', b'Z'), (b'_', b'_')], false),
                    b's' => self.g.add_class(vec![(b' ', b' '), (b'\t', b'\t')], false),
                    _ => return Ok(vec![GbnfAtom::Byte(c)]),
                };
                Ok(vec![GbnfAtom::Class(cid)])
            }
            c => {
                self.pos += 1;
                Ok(vec![GbnfAtom::Byte(c)])
            }
        }
    }
}

#[derive(Debug, Clone)]
struct GrammarState {
    grammar: Arc<GbnfGrammar>,
    /// Active PDA stacks (top of stack is `.last()`). Empty stack => accepting!
    stacks: Vec<Vec<GbnfAtom>>,
}

impl GrammarState {
    fn new(grammar: Arc<GbnfGrammar>) -> Self {
        let mut st = Self {
            stacks: vec![vec![GbnfAtom::Rule(grammar.root)]],
            grammar,
        };
        st.epsilon_close();
        st
    }

    fn epsilon_close(&mut self) {
        let mut out: Vec<Vec<GbnfAtom>> = Vec::new();
        let mut seen: HashSet<Vec<GbnfAtom>> = HashSet::new();
        let mut work: Vec<(Vec<GbnfAtom>, Vec<usize>)> = self
            .stacks
            .drain(..)
            .map(|st| (st, Vec::new()))
            .collect();

        while let Some((mut st, expanded)) = work.pop() {
            if out.len() >= 96 {
                break;
            }
            match st.last().copied() {
                Some(GbnfAtom::Rule(rid)) => {
                    if expanded.contains(&rid) || st.len() > 96 {
                        continue;
                    }
                    st.pop();
                    let mut next_exp = expanded;
                    next_exp.push(rid);
                    if let Some(alts) = self.grammar.rules.get(rid) {
                        for alt in alts {
                            let mut nst = st.clone();
                            for &atom in alt.iter().rev() {
                                nst.push(atom);
                            }
                            work.push((nst, next_exp.clone()));
                        }
                    }
                }
                _ => {
                    if seen.insert(st.clone()) {
                        out.push(st);
                    }
                }
            }
        }
        self.stacks = out;
    }

    fn is_accepting(&self) -> bool {
        self.stacks.iter().any(|s| s.is_empty())
    }

    fn is_strictly_done(&self) -> bool {
        !self.stacks.is_empty() && self.stacks.iter().all(|s| s.is_empty())
    }

    fn step_byte(&mut self, b: u8) -> bool {
        let mut next = Vec::new();
        for st in &self.stacks {
            match st.last().copied() {
                Some(GbnfAtom::Byte(x)) if x == b => {
                    let mut nst = st.clone();
                    nst.pop();
                    next.push(nst);
                }
                Some(GbnfAtom::Class(cid)) if self.grammar.classes[cid].matches(b) => {
                    let mut nst = st.clone();
                    nst.pop();
                    next.push(nst);
                }
                _ => {}
            }
        }
        if next.is_empty() {
            self.stacks.clear();
            return false;
        }
        self.stacks = next;
        self.epsilon_close();
        !self.stacks.is_empty()
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

fn compile_grammar_for_cfg(cfg: &SamplingCfg) -> Result<Option<GrammarState>, String> {
    if let Some(ref gbnf) = cfg.gbnf_grammar {
        let g = GbnfGrammar::parse_gbnf(gbnf)?;
        return Ok(Some(GrammarState::new(Arc::new(g))));
    }
    if let Some(ref schema) = cfg.json_schema {
        let g = GbnfGrammar::from_json_schema(schema)?;
        return Ok(Some(GrammarState::new(Arc::new(g))));
    }
    if cfg.json_constrained {
        let g = GbnfGrammar::json_object();
        return Ok(Some(GrammarState::new(Arc::new(g))));
    }
    Ok(None)
}

// ============================================================================
// Tenon v6: Zero-Dependency Base64 + PNG (RFC 1951 Inflate) + BMP + PPM Decoder
// ============================================================================
#[derive(Debug, Clone)]
struct RgbImage {
    width: usize,
    height: usize,
    rgb: Vec<u8>,
}

fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0u32;
    for b in input.bytes() {
        let val = match b {
            b'A'..=b'Z' => (b - b'A') as u32,
            b'a'..=b'z' => (b - b'a' + 26) as u32,
            b'0'..=b'9' => (b - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            b'=' | b'\r' | b'\n' | b' ' | b'\t' => continue,
            _ => return Err(format!("invalid base64 byte {}", b)),
        };
        acc = (acc << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((acc >> bits) & 0xFF) as u8);
        }
    }
    Ok(out)
}

// Minimal RFC 1951 Inflate for PNG zlib IDAT streams (stored, fixed Huffman, dynamic Huffman)
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buf: u32,
    bit_cnt: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, bit_buf: 0, bit_cnt: 0 }
    }
    fn read_bits(&mut self, n: u32) -> Result<u32, String> {
        while self.bit_cnt < n {
            if self.pos >= self.data.len() {
                return Err("unexpected EOF in deflate stream".into());
            }
            self.bit_buf |= (self.data[self.pos] as u32) << self.bit_cnt;
            self.pos += 1;
            self.bit_cnt += 8;
        }
        let v = self.bit_buf & ((1u32 << n) - 1);
        self.bit_buf >>= n;
        self.bit_cnt -= n;
        Ok(v)
    }
    fn align_byte(&mut self) {
        let rem = self.bit_cnt & 7;
        self.bit_buf >>= rem;
        self.bit_cnt -= rem;
    }
}

struct HuffTable {
    counts: [u16; 16],
    syms: Vec<u16>,
}

impl HuffTable {
    fn build(lengths: &[u8]) -> Result<Self, String> {
        let mut counts = [0u16; 16];
        for &l in lengths {
            if l > 15 {
                return Err("huffman length > 15".into());
            }
            if l > 0 {
                counts[l as usize] += 1;
            }
        }
        let mut offs = [0usize; 16];
        let mut total = 0usize;
        for bits in 1..16 {
            offs[bits] = total;
            total += counts[bits] as usize;
        }
        let mut syms = vec![0u16; total];
        for (sym, &l) in lengths.iter().enumerate() {
            if l > 0 {
                syms[offs[l as usize]] = sym as u16;
                offs[l as usize] += 1;
            }
        }
        Ok(Self { counts, syms })
    }

    fn decode(&self, br: &mut BitReader<'_>) -> Result<u16, String> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0usize;
        for len in 1..16 {
            code |= br.read_bits(1)? as i32;
            let count = self.counts[len] as i32;
            if code - count < first {
                return Ok(self.syms[index + (code - first) as usize]);
            }
            index += count as usize;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err("invalid huffman code".into())
    }
}

fn zlib_inflate(zdata: &[u8]) -> Result<Vec<u8>, String> {
    if zdata.len() < 6 {
        return Err("zlib stream too short".into());
    }
    let cmf = zdata[0];
    let flg = zdata[1];
    if (cmf & 0x0F) != 8 || (((cmf as u16) * 256 + (flg as u16)) % 31) != 0 {
        return Err("invalid zlib header".into());
    }
    let start = if (flg & 0x20) != 0 { 6 } else { 2 };
    let mut br = BitReader::new(&zdata[start..]);
    let mut out = Vec::new();

    const LEN_BASE: [u16; 29] = [
        3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258,
    ];
    const LEN_EXT: [u8; 29] = [
        0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,
    ];
    const DIST_BASE: [u16; 30] = [
        1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577,
    ];
    const DIST_EXT: [u8; 30] = [
        0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,
    ];
    const CL_ORDER: [usize; 19] = [16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15];

    loop {
        let bfinal = br.read_bits(1)?;
        let btype = br.read_bits(2)?;
        if btype == 0 {
            br.align_byte();
            let len = br.read_bits(16)? as usize;
            let _nlen = br.read_bits(16)?;
            for _ in 0..len {
                out.push(br.read_bits(8)? as u8);
            }
        } else if btype == 1 || btype == 2 {
            let (lit_tbl, dist_tbl) = if btype == 1 {
                let mut ll = [0u8; 288];
                for i in 0..144 { ll[i] = 8; }
                for i in 144..256 { ll[i] = 9; }
                for i in 256..280 { ll[i] = 7; }
                for i in 280..288 { ll[i] = 8; }
                let dl = [5u8; 32];
                (HuffTable::build(&ll)?, HuffTable::build(&dl)?)
            } else {
                let hlit = br.read_bits(5)? as usize + 257;
                let hdist = br.read_bits(5)? as usize + 1;
                let hclen = br.read_bits(4)? as usize + 4;
                let mut cl_lens = [0u8; 19];
                for i in 0..hclen {
                    cl_lens[CL_ORDER[i]] = br.read_bits(3)? as u8;
                }
                let cl_tbl = HuffTable::build(&cl_lens)?;
                let mut lens = Vec::with_capacity(hlit + hdist);
                while lens.len() < hlit + hdist {
                    let sym = cl_tbl.decode(&mut br)?;
                    if sym < 16 {
                        lens.push(sym as u8);
                    } else if sym == 16 {
                        let rep = br.read_bits(2)? as usize + 3;
                        let prev = *lens.last().ok_or("invalid rle 16")?;
                        for _ in 0..rep { lens.push(prev); }
                    } else if sym == 17 {
                        let rep = br.read_bits(3)? as usize + 3;
                        for _ in 0..rep { lens.push(0); }
                    } else if sym == 18 {
                        let rep = br.read_bits(7)? as usize + 11;
                        for _ in 0..rep { lens.push(0); }
                    }
                }
                (HuffTable::build(&lens[..hlit])?, HuffTable::build(&lens[hlit..hlit + hdist])?)
            };
            loop {
                let sym = lit_tbl.decode(&mut br)?;
                if sym < 256 {
                    out.push(sym as u8);
                } else if sym == 256 {
                    break;
                } else {
                    let li = (sym - 257) as usize;
                    if li >= LEN_BASE.len() {
                        return Err("invalid deflate length symbol".into());
                    }
                    let length = LEN_BASE[li] as usize + br.read_bits(LEN_EXT[li] as u32)? as usize;
                    let di = dist_tbl.decode(&mut br)? as usize;
                    if di >= DIST_BASE.len() {
                        return Err("invalid deflate dist symbol".into());
                    }
                    let dist = DIST_BASE[di] as usize + br.read_bits(DIST_EXT[di] as u32)? as usize;
                    if dist == 0 || dist > out.len() {
                        return Err("invalid deflate back-reference".into());
                    }
                    for _ in 0..length {
                        let b = out[out.len() - dist];
                        out.push(b);
                    }
                }
            }
        } else {
            return Err("invalid deflate block type".into());
        }
        if bfinal != 0 {
            break;
        }
    }
    Ok(out)
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (ia, ib, ic) = (a as i32, b as i32, c as i32);
    let p = ia + ib - ic;
    let pa = (p - ia).abs();
    let pb = (p - ib).abs();
    let pc = (p - ic).abs();
    if pa <= pb && pa <= pc { a } else if pb <= pc { b } else { c }
}

fn decode_png(data: &[u8]) -> Result<RgbImage, String> {
    if data.len() < 33 || &data[..8] != b"\x89PNG\r\n\x1a\n" {
        return Err("not a PNG image".into());
    }
    let mut pos = 8usize;
    let mut width = 0usize;
    let mut height = 0usize;
    let mut color_type = 2u8;
    let mut idat = Vec::new();
    while pos + 12 <= data.len() {
        let len = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let ctype = &data[pos + 4..pos + 8];
        let body_start = pos + 8;
        let body_end = body_start + len;
        if body_end + 4 > data.len() {
            break;
        }
        let body = &data[body_start..body_end];
        if ctype == b"IHDR" && len >= 13 {
            width = u32::from_be_bytes([body[0], body[1], body[2], body[3]]) as usize;
            height = u32::from_be_bytes([body[4], body[5], body[6], body[7]]) as usize;
            let bit_depth = body[8];
            color_type = body[9];
            if bit_depth != 8 {
                return Err(format!("unsupported PNG bit_depth {}", bit_depth));
            }
        } else if ctype == b"IDAT" {
            idat.extend_from_slice(body);
        } else if ctype == b"IEND" {
            break;
        }
        pos = body_end + 4;
    }
    if width == 0 || height == 0 {
        return Err("invalid PNG dimensions".into());
    }
    let bpp: usize = match color_type {
        0 => 1,
        2 => 3,
        6 => 4,
        _ => return Err(format!("unsupported PNG color_type {}", color_type)),
    };
    let raw = zlib_inflate(&idat)?;
    let stride = width * bpp;
    if raw.len() < height * (stride + 1) {
        return Err("truncated PNG IDAT payload".into());
    }
    let mut recon = vec![0u8; height * stride];
    for y in 0..height {
        let flt = raw[y * (stride + 1)];
        let src = &raw[y * (stride + 1) + 1..y * (stride + 1) + 1 + stride];
        for x in 0..stride {
            let left = if x >= bpp { recon[y * stride + x - bpp] } else { 0 };
            let up = if y > 0 { recon[(y - 1) * stride + x] } else { 0 };
            let ul = if y > 0 && x >= bpp { recon[(y - 1) * stride + x - bpp] } else { 0 };
            let v = match flt {
                0 => src[x],
                1 => src[x].wrapping_add(left),
                2 => src[x].wrapping_add(up),
                3 => src[x].wrapping_add(((left as u16 + up as u16) >> 1) as u8),
                4 => src[x].wrapping_add(paeth(left, up, ul)),
                _ => src[x],
            };
            recon[y * stride + x] = v;
        }
    }
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let p = y * stride + x * bpp;
            match bpp {
                1 => { let g = recon[p]; rgb.extend_from_slice(&[g, g, g]); }
                3 | 4 => rgb.extend_from_slice(&[recon[p], recon[p + 1], recon[p + 2]]),
                _ => {}
            }
        }
    }
    Ok(RgbImage { width, height, rgb })
}

fn decode_bmp(data: &[u8]) -> Result<RgbImage, String> {
    if data.len() < 54 || &data[..2] != b"BM" {
        return Err("not a BMP image".into());
    }
    let pix_off = u32::from_le_bytes([data[10], data[11], data[12], data[13]]) as usize;
    let w = i32::from_le_bytes([data[18], data[19], data[20], data[21]]).unsigned_abs() as usize;
    let h_raw = i32::from_le_bytes([data[22], data[23], data[24], data[25]]);
    let top_down = h_raw < 0;
    let h = h_raw.unsigned_abs() as usize;
    let bpp = u16::from_le_bytes([data[28], data[29]]) as usize / 8;
    if w == 0 || h == 0 || (bpp != 3 && bpp != 4) {
        return Err("unsupported BMP format".into());
    }
    let row_stride = (w * bpp + 3) & !3;
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        let src_y = if top_down { y } else { h - 1 - y };
        let row = pix_off + src_y * row_stride;
        if row + w * bpp > data.len() {
            return Err("truncated BMP data".into());
        }
        for x in 0..w {
            let b = data[row + x * bpp];
            let g = data[row + x * bpp + 1];
            let r = data[row + x * bpp + 2];
            let dst = (y * w + x) * 3;
            rgb[dst] = r;
            rgb[dst + 1] = g;
            rgb[dst + 2] = b;
        }
    }
    Ok(RgbImage { width: w, height: h, rgb })
}

fn decode_ppm(data: &[u8]) -> Result<RgbImage, String> {
    if data.len() < 11 || &data[..2] != b"P6" {
        return Err("not a P6 PPM image".into());
    }
    let mut i = 2usize;
    let mut nums = Vec::new();
    while nums.len() < 3 && i < data.len() {
        while i < data.len() && data[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < data.len() && data[i] == b'#' {
            while i < data.len() && data[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let s = i;
        while i < data.len() && data[i].is_ascii_digit() {
            i += 1;
        }
        if i > s {
            nums.push(std::str::from_utf8(&data[s..i]).unwrap_or("0").parse::<usize>().unwrap_or(0));
        } else {
            break;
        }
    }
    if nums.len() < 3 || i >= data.len() {
        return Err("invalid PPM header".into());
    }
    i += 1; // single whitespace byte after maxval
    let (w, h) = (nums[0], nums[1]);
    if i + w * h * 3 > data.len() {
        return Err("truncated PPM pixel payload".into());
    }
    Ok(RgbImage { width: w, height: h, rgb: data[i..i + w * h * 3].to_vec() })
}

fn decode_image_bytes(data: &[u8]) -> Result<RgbImage, String> {
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        decode_png(data)
    } else if data.starts_with(b"BM") {
        decode_bmp(data)
    } else if data.starts_with(b"P6") {
        decode_ppm(data)
    } else {
        Err("unsupported image format (supported: PNG, BMP, P6 PPM)".into())
    }
}

fn decode_image_url_to_rgb(url: &str) -> Result<RgbImage, String> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (_, payload) = rest.split_once(',').ok_or("malformed data: URL")?;
        let bytes = base64_decode(payload)?;
        decode_image_bytes(&bytes)
    } else {
        let path = url.strip_prefix("file://").unwrap_or(url);
        let bytes = fs::read(path).map_err(|e| format!("failed to read image `{}`: {}", path, e))?;
        decode_image_bytes(&bytes)
    }
}

// ============================================================================
// Radix Prefix Tree KV Cache (Multi-Slot Longest-Prefix Reuse -> 5-15x TTFT!)
// ============================================================================
#[derive(Debug, Clone)]
struct KvSlotState {
    tokens: Vec<i64>,
    last_used: u64,
    in_use: bool,
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
                    in_use: false,
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
            if st.in_use {
                continue;
            }
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

    /// PagedAttention Zero-Copy COW Fork Matcher: finds the longest prefix across ALL slots (even active ones!)
    /// and acquires a free slot to fork into! Returns `(dst_slot, src_slot, lcp)`.
    fn match_for_cow_fork(&mut self, prompt: &[i64]) -> Option<(usize, usize, usize)> {
        let max_reuse = prompt.len().saturating_sub(1);
        let mut best_src = 0usize;
        let mut best_lcp = 0usize;
        for (idx, st) in self.slots.iter().enumerate() {
            let mut lcp = 0usize;
            let lim = st.tokens.len().min(max_reuse);
            while lcp < lim && st.tokens[lcp] == prompt[lcp] {
                lcp += 1;
            }
            if lcp > best_lcp {
                best_lcp = lcp;
                best_src = idx;
            }
        }
        // Pick an unused slot (preferring empty, then LRU inactive; never overwriting `best_src` if another free slot exists!)
        let mut dst_slot: Option<usize> = None;
        let mut best_lru = u64::MAX;
        for (idx, st) in self.slots.iter().enumerate() {
            if st.in_use {
                continue;
            }
            if best_lcp > 0 && idx == best_src && self.slots.iter().filter(|s| !s.in_use).count() > 1 {
                continue;
            }
            if st.last_used < best_lru {
                best_lru = st.last_used;
                dst_slot = Some(idx);
            }
        }
        let dst = dst_slot?;
        self.clock += 1;
        self.slots[dst].in_use = true;
        self.slots[dst].last_used = self.clock;
        if best_lcp > 0 {
            self.total_hits += 1;
            self.total_cached_tokens += best_lcp as u64;
        }
        Some((dst, best_src, best_lcp))
    }

    fn set_active_prefix(&mut self, slot: usize, tokens: &[i64]) {
        if let Some(st) = self.slots.get_mut(slot) {
            st.tokens = tokens.to_vec();
        }
    }

    fn release(&mut self, slot: usize, full_tokens: Vec<i64>) {
        if let Some(st) = self.slots.get_mut(slot) {
            self.clock += 1;
            st.tokens = full_tokens;
            st.last_used = self.clock;
            st.in_use = false;
        }
    }

    fn commit(&mut self, slot: usize, full_tokens: Vec<i64>) {
        self.release(slot, full_tokens);
    }
}

// ============================================================================
// Loaded Engine State + Unified Grammar Sampler + Speculative Decoding
// ============================================================================
#[derive(Clone)]
struct ModelMeta {
    backend: &'static str,
    model_id: String,
    arch: String,
    d_model: usize,
    max_ctx: i64,
    slot_ctx: i64,
    n_kv_slots: usize,
    kv_q8: bool,
    mla_latent: bool,
    n_bpe_merges: i64,
    tp_size: i64,
    ep_size: i64,
    hybrid_2d: bool,
    nccl_active: bool,
    has_draft: bool,
    has_mtp: i64,
    has_mmproj: bool,
    mmproj_n_tokens: usize,
    has_audio: bool,
    audio_proj_dim: usize,
    supports_cb: bool,
    prefill_chunk_size: usize,
    lora_adapters: Vec<String>,
}

struct LoadedModel {
    api: LlmApi,
    handle: i64,
    draft_handle: i64,
    mmproj_handle: i64,
    audio_handle: i64,
    meta: ModelMeta,
    bos_id: i64,
    eos_id: i64,
    vocab_size: usize,
    radix_cache: RadixKvCache,
    cb_radix: RadixKvCache,
    vocab_pieces: Vec<String>,
    logits_scratch: Vec<f32>,
    lora_map: HashMap<String, (i64, f64)>,
    active_lora_id: i64,
    prefill_chunk_size: usize,
    vision_cache: HashMap<u128, (usize, Vec<f32>)>,
    vision_cache_order: Vec<u128>,
    vision_cache_hits: u64,
}

impl LoadedModel {
    fn resolve_lora(&self, name: Option<&str>) -> (i64, f64) {
        if let Some(n) = name {
            if let Some(&(id, sc)) = self.lora_map.get(n) {
                return (id, sc);
            }
        }
        (0, 1.0)
    }

    /// Zero-copy runtime LoRA hot-swap (~1 us GPU pointer switch); returns `(lora_id, swapped)`.
    fn apply_lora(&mut self, name: Option<&str>) -> (i64, bool) {
        let (target_id, scale) = self.resolve_lora(name);
        if target_id != self.active_lora_id {
            self.api.lora_set(self.handle, target_id, scale);
            self.active_lora_id = target_id;
            return (target_id, true);
        }
        (target_id, false)
    }

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

    /// Sample a token directly from `self.logits_scratch` with optional GrammarState (JSON Schema / GBNF),
    /// repetition penalty, temperature, top-k, top-p, and min-p!
    fn sample_from_scratch_logits(
        &mut self,
        grammar: Option<&GrammarState>,
        cfg: &SamplingCfg,
        recent: &[i64],
        seed: i64,
    ) -> (i64, Option<GrammarState>) {
        let n = self.vocab_size.min(self.logits_scratch.len());
        if n == 0 {
            return (self.eos_id, None);
        }
        if let Some(g) = grammar {
            if g.is_strictly_done() {
                return (self.eos_id, None);
            }
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

        // Fast unconstrained sampling path (no grammar)
        let Some(gs) = grammar else {
            if cfg.temp <= 1e-5 {
                let mut best = 0usize;
                let mut bv = self.logits_scratch[0];
                for i in 1..n {
                    if self.logits_scratch[i] > bv {
                        bv = self.logits_scratch[i];
                        best = i;
                    }
                }
                return (best as i64, None);
            }
            let k = (cfg.top_k.max(1) as usize).min(n).min(128);
            let mut idx: Vec<usize> = (0..n).collect();
            idx.select_nth_unstable_by(k - 1, |&a, &b| {
                self.logits_scratch[b]
                    .partial_cmp(&self.logits_scratch[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            idx[..k].sort_unstable_by(|&a, &b| {
                self.logits_scratch[b]
                    .partial_cmp(&self.logits_scratch[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let cands: Vec<(usize, f32)> = idx[..k].iter().map(|&id| (id, self.logits_scratch[id])).collect();
            return (nucleus_sample(&cands, cfg, seed) as i64, None);
        };

        // Grammar-constrained sampling path (Full Recursive JSON Schema / GBNF PDA!)
        self.ensure_vocab_pieces();
        let allow_stop = gs.is_accepting();
        let mut idx: Vec<usize> = (0..n).collect();
        let probe_k = 1024.min(n);
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

        let mut valid: Vec<(usize, f32, Option<GrammarState>)> = Vec::with_capacity(64);
        let limit_k = (cfg.top_k.max(1) as usize).min(64);
        for &id in &idx[..probe_k] {
            let is_stp = id as i64 == self.eos_id || is_stop_token(id as i64, self.eos_id, &self.vocab_pieces[id]);
            if is_stp {
                if allow_stop {
                    valid.push((id, self.logits_scratch[id], None));
                    if valid.len() >= limit_k {
                        break;
                    }
                }
                continue;
            }
            if let Some(ns) = gs.try_advance(&self.vocab_pieces[id]) {
                valid.push((id, self.logits_scratch[id], Some(ns)));
                if valid.len() >= limit_k {
                    break;
                }
            }
        }
        if valid.is_empty() {
            // Guaranteed full-vocab fallback so output ALWAYS obeys schema/grammar!
            let mut best: Option<(usize, f32, GrammarState)> = None;
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
                if let Some(ns) = gs.try_advance(&self.vocab_pieces[id]) {
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
            return (id as i64, ns);
        }
        let pairs: Vec<(usize, f32)> = valid.iter().enumerate().map(|(i, (_, v, _))| (i, *v)).collect();
        let chosen = nucleus_sample(&pairs, cfg, seed);
        let (id, _, ns) = valid.swap_remove(chosen);
        (id as i64, ns)
    }

    fn sample_constrained(
        &mut self,
        grammar: &GrammarState,
        cfg: &SamplingCfg,
        recent: &[i64],
        seed: i64,
    ) -> (i64, Option<GrammarState>) {
        if self.logits_scratch.len() < self.vocab_size {
            self.logits_scratch.resize(self.vocab_size, 0.0);
        }
        let n = self.api.copy_logits(self.handle, &mut self.logits_scratch);
        if n == 0 {
            return (self.eos_id, None);
        }
        self.sample_from_scratch_logits(Some(grammar), cfg, recent, seed)
    }
}

fn nucleus_sample(cands: &[(usize, f32)], cfg: &SamplingCfg, seed: i64) -> usize {
    if cands.len() <= 1 || cfg.temp <= 1e-5 {
        return cands[0].0;
    }
    let inv_t = 1.0f32 / (cfg.temp as f32);
    let mx = cands[0].1 * inv_t;
    let mut probs: Vec<f32> = cands.iter().map(|(_, v)| (v * inv_t - mx).exp()).collect();
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
    for i in 0..keep {
        acc += probs[i];
        if acc >= target {
            return cands[i].0;
        }
    }
    cands[0].0
}

impl Drop for LoadedModel {
    fn drop(&mut self) {
        if self.handle > 0 {
            if self.active_lora_id > 0 {
                self.api.lora_set(self.handle, 0, 1.0);
                self.active_lora_id = 0;
            }
            for &(lid, _) in self.lora_map.values() {
                if lid > 0 {
                    self.api.lora_free(self.handle, lid);
                }
            }
        }
        if self.mmproj_handle > 0 {
            self.api.mmproj_free(self.mmproj_handle);
            self.mmproj_handle = 0;
        }
        if self.audio_handle > 0 {
            self.api.audio_free(self.audio_handle);
            self.audio_handle = 0;
        }
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
            | "<end_of_utterance>"
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

#[derive(Debug, Clone)]
struct AudioPcm16 {
    samples: Vec<i16>,
    sample_rate: u32,
}

fn decode_wav_pcm16(raw: &[u8]) -> Result<AudioPcm16, String> {
    if raw.len() >= 44 && &raw[0..4] == b"RIFF" && &raw[8..12] == b"WAVE" {
        let mut pos = 12usize;
        let mut channels = 1u16;
        let mut sample_rate = 16000u32;
        let mut bits_per_sample = 16u16;
        let mut data_slice: Option<&[u8]> = None;
        while pos + 8 <= raw.len() {
            let cid = &raw[pos..pos + 4];
            let csz = u32::from_le_bytes([raw[pos + 4], raw[pos + 5], raw[pos + 6], raw[pos + 7]]) as usize;
            let body_start = pos + 8;
            let body_end = (body_start + csz).min(raw.len());
            if cid == b"fmt " && body_end - body_start >= 16 {
                let fmt_code = u16::from_le_bytes([raw[body_start], raw[body_start + 1]]);
                if fmt_code != 1 {
                    return Err(format!("unsupported WAV audio format {} (only PCM=1 supported)", fmt_code));
                }
                channels = u16::from_le_bytes([raw[body_start + 2], raw[body_start + 3]]).max(1);
                sample_rate = u32::from_le_bytes([
                    raw[body_start + 4],
                    raw[body_start + 5],
                    raw[body_start + 6],
                    raw[body_start + 7],
                ]);
                bits_per_sample = u16::from_le_bytes([raw[body_start + 14], raw[body_start + 15]]);
            } else if cid == b"data" {
                data_slice = Some(&raw[body_start..body_end]);
            }
            pos = body_start + csz + (csz & 1);
        }
        let data = data_slice.ok_or_else(|| "WAV missing `data` chunk".to_string())?;
        if bits_per_sample != 16 {
            return Err(format!("unsupported WAV bit depth {} (expected 16-bit PCM)", bits_per_sample));
        }
        let ch = channels as usize;
        let frame_bytes = ch * 2;
        let n_frames = data.len() / frame_bytes;
        let mut samples = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let mut acc = 0i32;
            for c in 0..ch {
                let o = f * frame_bytes + c * 2;
                acc += i16::from_le_bytes([data[o], data[o + 1]]) as i32;
            }
            samples.push((acc / ch as i32) as i16);
        }
        return Ok(AudioPcm16 { samples, sample_rate: sample_rate.max(8000) });
    }
    if raw.len() >= 2 {
        let mut samples = Vec::with_capacity(raw.len() / 2);
        for chunk in raw.chunks_exact(2) {
            samples.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        return Ok(AudioPcm16 { samples, sample_rate: 16000 });
    }
    Err("empty audio payload".into())
}

fn fnv1a_128_image(img: &RgbImage, use_anyres: bool) -> u128 {
    let mut h1: u64 = 0xcbf29ce484222325;
    let mut h2: u64 = 0x100000001b3u64.wrapping_mul(0x9e3779b97f4a7c15);
    for b in (img.width as u64).to_le_bytes().iter()
        .chain((img.height as u64).to_le_bytes().iter())
        .chain([use_anyres as u8].iter())
    {
        h1 ^= *b as u64;
        h1 = h1.wrapping_mul(0x100000001b3);
        h2 ^= (*b as u64).rotate_left(3);
        h2 = h2.wrapping_mul(0x9e3779b185ebca87);
    }
    for &b in &img.rgb {
        h1 ^= b as u64;
        h1 = h1.wrapping_mul(0x100000001b3);
        h2 ^= b as u64;
        h2 = h2.wrapping_mul(0x9e3779b185ebca87);
    }
    ((h1 as u128) << 64) | (h2 as u128)
}

#[derive(Debug, Clone)]
enum MmPart {
    Text(String),
    Image(RgbImage),
    Audio(AudioPcm16),
}

fn run_multimodal_generation<F>(
    m: &mut LoadedModel,
    parts: &[MmPart],
    cfg: &SamplingCfg,
    mut on_piece: F,
) -> GenResult
where
    F: FnMut(&str) -> bool,
{
    let _ = m.apply_lora(cfg.lora.as_deref());
    let t_pf = Instant::now();
    let img_size = if m.mmproj_handle > 0 {
        m.api.mmproj_info(m.mmproj_handle, 0).max(1) as usize
    } else {
        384
    };
    let n_vis_base = m.meta.mmproj_n_tokens.max(1);
    let pdim = if m.mmproj_handle > 0 {
        m.api.mmproj_info(m.mmproj_handle, 5).max(m.meta.d_model as i64) as usize
    } else {
        m.meta.d_model
    };
    let auto_mrope = {
        let a = m.meta.arch.to_ascii_lowercase();
        a.contains("qwen2vl") || a.contains("qwen2_5_vl") || a.contains("qwen2.5-vl") || cfg.mrope_grid.is_some()
    };

    // SmolVLM / Idefics3 multi-image chat template vs generic LLaVA / Audio template
    let is_smolvlm = m.api.tokenize(m.handle, "<fake_token_around_image>", 8).len() == 1;
    let mut pos = 0i64;
    let mut next_tok = m.bos_id;
    let mut seg_buf = String::from(if is_smolvlm { "<|im_start|>User:" } else { "USER:" });

    for part in parts {
        match part {
            MmPart::Text(txt) => {
                if !txt.is_empty() {
                    if !seg_buf.ends_with(' ') && !txt.starts_with(' ') {
                        seg_buf.push(' ');
                    }
                    seg_buf.push_str(txt);
                }
            }
            MmPart::Image(img) => {
                if m.mmproj_handle <= 0 {
                    continue;
                }
                if is_smolvlm {
                    seg_buf.push_str("<fake_token_around_image><global-img>");
                }
                let tids = m.api.tokenize(m.handle, &seg_buf, 512);
                seg_buf.clear();
                if !tids.is_empty() {
                    next_tok = m.api.prefill(m.handle, &tids, pos);
                    pos += tids.len() as i64;
                }
                let use_anyres = cfg.anyres || img.width > img_size || img.height > img_size;
                let hash_key = fnv1a_128_image(img, use_anyres);
                let (n_encoded, vis_emb) = if let Some((nc, cached)) = m.vision_cache.get(&hash_key) {
                    m.vision_cache_hits += 1;
                    (*nc, cached.clone())
                } else {
                    let max_vecs = if use_anyres { n_vis_base * 5 } else { n_vis_base };
                    let mut buf = vec![0.0f32; max_vecs * pdim];
                    let n_enc = if use_anyres {
                        m.api.mmproj_encode_anyres(
                            m.mmproj_handle,
                            &img.rgb,
                            img.width,
                            img.height,
                            5,
                            &mut buf,
                            max_vecs,
                        )
                    } else {
                        m.api.mmproj_encode_rgb(
                            m.mmproj_handle,
                            &img.rgb,
                            img.width,
                            img.height,
                            &mut buf,
                            n_vis_base,
                        )
                    };
                    buf.truncate(n_enc * pdim);
                    if n_enc > 0 {
                        if m.vision_cache_order.len() >= 32 {
                            let old = m.vision_cache_order.remove(0);
                            m.vision_cache.remove(&old);
                        }
                        m.vision_cache_order.push(hash_key);
                        m.vision_cache.insert(hash_key, (n_enc, buf.clone()));
                    }
                    (n_enc, buf)
                };
                if n_encoded > 0 {
                    if auto_mrope {
                        let (gt, gh, gw) = cfg.mrope_grid.unwrap_or_else(|| {
                            let side = (n_encoded as f64).sqrt().round().max(1.0) as i64;
                            let w = ((n_encoded as i64) + side - 1) / side;
                            (1, side, w.max(1))
                        });
                        next_tok = m.api.step_embd_mrope(m.handle, &vis_emb, n_encoded, pos, gt, gh, gw);
                    } else {
                        next_tok = m.api.step_embd(m.handle, &vis_emb, n_encoded, pos);
                    }
                    pos += n_encoded as i64;
                }
                if is_smolvlm {
                    seg_buf.push_str("<fake_token_around_image>");
                }
            }
            MmPart::Audio(aud) => {
                if m.audio_handle <= 0 {
                    continue;
                }
                let tids = m.api.tokenize(m.handle, &seg_buf, 512);
                seg_buf.clear();
                if !tids.is_empty() {
                    next_tok = m.api.prefill(m.handle, &tids, pos);
                    pos += tids.len() as i64;
                }
                let adim = m.api.audio_info(m.audio_handle, 5).max(m.meta.d_model as i64) as usize;
                let max_vecs = 512usize;
                let mut aud_emb = vec![0.0f32; max_vecs * adim];
                let n_aud = m.api.audio_encode_pcm16(
                    m.audio_handle,
                    &aud.samples,
                    aud.sample_rate,
                    &mut aud_emb,
                    max_vecs,
                );
                if n_aud > 0 {
                    next_tok = m.api.step_embd(m.handle, &aud_emb, n_aud, pos);
                    pos += n_aud as i64;
                }
            }
        }
    }

    seg_buf.push_str(if is_smolvlm { "<end_of_utterance>\nAssistant:" } else { "\nASSISTANT:" });
    let close_ids = m.api.tokenize(m.handle, &seg_buf, 512);
    if !close_ids.is_empty() {
        next_tok = m.api.prefill(m.handle, &close_ids, pos);
        pos += close_ids.len() as i64;
    }
    let prefill_ms = t_pf.elapsed().as_secs_f64() * 1000.0;
    let prompt_tokens = pos as usize;

    let mut grammar = compile_grammar_for_cfg(cfg).ok().flatten();
    if let Some(ref gs) = grammar {
        let (tok, ns) = m.sample_constrained(gs, cfg, &[], cfg.seed);
        grammar = ns;
        next_tok = tok;
    } else if cfg.temp > 1e-5 {
        next_tok = m.api.sample_next(m.handle, -1, pos - 1, cfg, &[], cfg.seed);
    }

    let mut out_text = String::new();
    let mut completion_tokens = 0usize;
    let mut finish_reason = "length";
    let mut recent: Vec<i64> = Vec::new();
    let t_dec = Instant::now();
    let slot_ctx = m.meta.slot_ctx as usize;

    while completion_tokens < cfg.max_tokens {
        let piece = m.api.token_str(m.handle, next_tok);
        if is_stop_token(next_tok, m.eos_id, &piece) {
            finish_reason = "stop";
            break;
        }
        out_text.push_str(&piece);
        completion_tokens += 1;
        recent.push(next_tok);
        if !on_piece(&piece) {
            finish_reason = "stop";
            break;
        }
        if grammar.as_ref().is_some_and(GrammarState::is_strictly_done) {
            finish_reason = "stop";
            break;
        }
        if (pos as usize) + 2 >= slot_ctx || completion_tokens >= cfg.max_tokens {
            finish_reason = "length";
            break;
        }
        let rstart = recent.len().saturating_sub(64);
        let step_seed = cfg.seed.wrapping_add((completion_tokens as i64).wrapping_mul(1000003));
        if let Some(ref gs) = grammar {
            let _ = m.api.step(m.handle, next_tok, pos);
            pos += 1;
            let (tok, ns) = m.sample_constrained(gs, cfg, &recent[rstart..], step_seed);
            grammar = ns;
            next_tok = tok;
        } else {
            next_tok = m.api.sample_next(m.handle, next_tok, pos, cfg, &recent[rstart..], step_seed);
            pos += 1;
        }
    }
    let decode_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
    GenResult {
        text: out_text,
        prompt_tokens,
        cached_prefix_tokens: 0,
        kv_slot: 0,
        completion_tokens,
        spec_proposed: 0,
        spec_accepted: 0,
        prefill_ms,
        decode_ms,
        finish_reason,
    }
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
    let _ = m.apply_lora(cfg.lora.as_deref());
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
    let mut grammar = compile_grammar_for_cfg(cfg).ok().flatten();

    let mut next_tok = if let Some(ref gs) = grammar {
        let (tok, ns) = m.sample_constrained(
            gs,
            cfg,
            &full_seq[full_seq.len().saturating_sub(64)..],
            cfg.seed,
        );
        grammar = ns;
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

    let unconstrained = grammar.is_none();
    let use_draft = cfg.spec_mode == SpecMode::Draft && m.draft_handle > 0 && unconstrained && cfg.temp <= 1e-5;
    let use_tree = cfg.spec_mode == SpecMode::Tree && m.draft_handle > 0 && unconstrained && cfg.temp <= 1e-5;
    let use_mtp = cfg.spec_mode == SpecMode::Mtp && m.meta.has_mtp > 0 && unconstrained && cfg.temp <= 1e-5;
    let use_ngram = cfg.spec_mode == SpecMode::Ngram && unconstrained && cfg.temp <= 1e-5;
    if use_draft || use_tree {
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
        if grammar.as_ref().is_some_and(GrammarState::is_strictly_done) {
            finish_reason = "stop";
            break;
        }
        let pos = full_seq.len() - 1;
        if pos + 2 >= slot_ctx || completion_tokens >= cfg.max_tokens {
            finish_reason = "length";
            break;
        }

        // Lossless Tree-Attention Speculative Decoding (`--speculative tree --draft ...`)!
        if use_tree && cfg.spec_k >= 2 {
            let max_d = cfg.spec_k.min(4).min(cfg.max_tokens - completion_tokens).min(slot_ctx - pos - 2);
            // Build a 2-branch speculative tree from draft model's top-2 at depth 1 + greedy chain along both branches
            let mut tree_ids = vec![next_tok];
            let mut parents = vec![-1i64];
            let d1_top = m.api.step(m.draft_handle, next_tok, pos as i64);
            let mut dlogits = vec![0.0f32; m.vocab_size];
            let dn = m.api.copy_logits(m.draft_handle, &mut dlogits);
            let mut second_tok = d1_top;
            if dn > 2 {
                let mut best2 = f32::NEG_INFINITY;
                for v in 0..dn {
                    if v as i64 != d1_top && dlogits[v] > best2 {
                        best2 = dlogits[v];
                        second_tok = v as i64;
                    }
                }
            }
            // Branch 1 (primary greedy chain of length `max_d`): nodes 1..=max_d
            tree_ids.push(d1_top);
            parents.push(0);
            let mut cur = d1_top;
            let mut dpos = (pos + 1) as i64;
            for step_i in 1..max_d {
                let nxt = m.api.step(m.draft_handle, cur, dpos);
                tree_ids.push(nxt);
                parents.push(step_i as i64);
                cur = nxt;
                dpos += 1;
            }
            // Branch 2 (alt token at depth 1): node max_d + 1
            if second_tok != d1_top {
                tree_ids.push(second_tok);
                parents.push(0);
            }
            spec_proposed += tree_ids.len() - 1;
            let mut tree_out = vec![0i64; tree_ids.len()];
            m.api.verify_tree(m.handle, &tree_ids, &parents, pos as i64, &mut tree_out);

            // Walk longest accepted path from root 0!
            let mut acc_path: Vec<i64> = vec![0];
            let mut cur_node = 0usize;
            let mut bonus_tok = tree_out[0];
            loop {
                let want = tree_out[cur_node];
                let mut matched_child = None;
                for c in (cur_node + 1)..tree_ids.len() {
                    if parents[c] == cur_node as i64 && tree_ids[c] == want {
                        matched_child = Some(c);
                        break;
                    }
                }
                if let Some(ch) = matched_child {
                    acc_path.push(ch as i64);
                    spec_accepted += 1;
                    let dp = m.api.token_str(m.handle, want);
                    if is_stop_token(want, m.eos_id, &dp) {
                        finish_reason = "stop";
                        break;
                    }
                    out_text.push_str(&dp);
                    completion_tokens += 1;
                    full_seq.push(want);
                    if !on_piece(&dp) || completion_tokens >= cfg.max_tokens {
                        finish_reason = "stop";
                        break;
                    }
                    cur_node = ch;
                    bonus_tok = tree_out[ch];
                } else {
                    break;
                }
            }
            m.api.commit_tree_path(m.handle, &acc_path, pos as i64);
            if finish_reason == "stop" {
                break;
            }
            // Resync draft KV strictly for the accepted branch tokens
            if acc_path.len() > 1 {
                let acc_toks: Vec<i64> = acc_path[1..].iter().map(|&ni| tree_ids[ni as usize]).collect();
                m.api.prefill(m.draft_handle, &acc_toks, (pos + 1) as i64);
            }
            next_tok = bonus_tok;
            continue;
        }

        // Lossless Batched Speculative Decoding (`--draft`, `--speculative mtp`, OR `--speculative ngram`)!
        if (use_draft || use_mtp || use_ngram) && cfg.spec_k > 0 {
            let max_k = cfg.spec_k.min(cfg.max_tokens - completion_tokens).min(slot_ctx - pos - 2);
            let mut draft_toks = Vec::with_capacity(max_k);
            if use_draft {
                let mut dcur = next_tok;
                let mut dpos = pos as i64;
                for _ in 0..max_k {
                    dcur = m.api.step(m.draft_handle, dcur, dpos);
                    let dp = m.api.token_str(m.draft_handle, dcur);
                    draft_toks.push(dcur);
                    dpos += 1;
                    if is_stop_token(dcur, m.eos_id, &dp) {
                        break;
                    }
                }
            } else if use_mtp {
                let p_tok = m.api.mtp_predict(m.handle, next_tok, pos as i64);
                if p_tok >= 0 && (p_tok as usize) < m.vocab_size {
                    draft_toks.push(p_tok);
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
                // CRITICAL: if ALL K draft tokens were accepted, feed the last accepted draft token
                // into `draft_handle` at `pos + n_acc` so draft KV has ZERO gap before bonus token!
                if use_draft && n_acc == draft_toks.len() {
                    let last_dt = draft_toks[n_acc - 1];
                    let _ = m.api.step(m.draft_handle, last_dt, (pos + n_acc) as i64);
                }
                next_tok = target_out[n_acc];
                continue;
            }
        }

        let rstart = full_seq.len().saturating_sub(64);
        let step_seed = cfg.seed.wrapping_add((completion_tokens as i64).wrapping_mul(1000003));
        if let Some(ref gs) = grammar {
            let _ = m.api.step(m.handle, next_tok, pos as i64);
            let (tok, ns) = m.sample_constrained(gs, cfg, &full_seq[rstart..], step_seed);
            grammar = ns;
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
// Tenon v7: Chunked Prefill + Continuous Batching + Adaptive Speculative Fusion + Multi-LoRA
// ============================================================================
enum StreamEvent {
    Piece(String),
    Done(GenResult),
}

enum EngineCmd {
    Generate {
        prompt: String,
        add_bos: bool,
        cfg: SamplingCfg,
        mm_parts: Vec<MmPart>,
        tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
    },
    Embed {
        inputs: Vec<String>,
        reply: tokio::sync::oneshot::Sender<Result<(Vec<Vec<f32>>, usize), String>>,
    },
    Rerank {
        query: String,
        documents: Vec<String>,
        top_n: usize,
        reply: tokio::sync::oneshot::Sender<Result<Vec<(usize, f32)>, String>>,
    },
    PagedStats {
        reply: tokio::sync::oneshot::Sender<(i64, i64, i64, i64)>,
    },
}

struct PendingGenReq {
    prompt: String,
    add_bos: bool,
    cfg: SamplingCfg,
    mm_parts: Vec<MmPart>,
    tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
}

struct PrefillingCbSeq {
    slot: usize,
    cfg: SamplingCfg,
    lora_id: i64,
    ids: Vec<i64>,
    actual_lcp: usize,
    cursor: usize,
    is_chunked: bool,
    t_pf: Instant,
    tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
}

struct ActiveCbSeq {
    slot: usize,
    cfg: SamplingCfg,
    lora_id: i64,
    grammar: Option<GrammarState>,
    full_seq: Vec<i64>,
    next_tok: i64,
    out_text: String,
    prompt_tokens: usize,
    cached_prefix_tokens: usize,
    completion_tokens: usize,
    spec_proposed: usize,
    spec_accepted: usize,
    prefill_ms: f64,
    t_dec: Instant,
    tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
}

/// Single-client Adaptive Speculative Decoding sequence running in linear KV slot 0;
/// seamlessly migrates to PagedKV Continuous Batching the instant a 2nd request arrives!
struct ActiveSpecSeq {
    kv_slot: usize,
    cfg: SamplingCfg,
    full_seq: Vec<i64>,
    next_tok: i64,
    out_text: String,
    prompt_tokens: usize,
    cached_prefix_tokens: usize,
    completion_tokens: usize,
    spec_proposed: usize,
    spec_accepted: usize,
    prefill_ms: f64,
    t_dec: Instant,
    tx: tokio::sync::mpsc::UnboundedSender<StreamEvent>,
}

struct ServerState {
    meta: ModelMeta,
    engine_tx: std_mpsc::Sender<EngineCmd>,
    queued_reqs: AtomicUsize,
    active_batch: AtomicUsize,
    cb_batches_run: AtomicU64,
    cb_max_batch: AtomicUsize,
    total_reqs: AtomicU64,
    radix_hits: AtomicU64,
    radix_saved_toks: AtomicU64,
    chunked_prefill_chunks: AtomicU64,
    adaptive_spec_steps: AtomicU64,
    adaptive_spec_proposed: AtomicU64,
    adaptive_spec_accepted: AtomicU64,
    spec_zero_migrations: AtomicU64,
    vision_cache_hits: AtomicU64,
    vision_cache_entries: AtomicUsize,
    lora_swaps: AtomicU64,
}

fn finish_cb_seq(
    m: &mut LoadedModel,
    seq: ActiveCbSeq,
    finish_reason: &'static str,
    state: &ServerState,
) {
    let decode_ms = seq.t_dec.elapsed().as_secs_f64() * 1000.0;
    if seq.cached_prefix_tokens > 0 {
        state.radix_hits.fetch_add(1, Ordering::Relaxed);
        state
            .radix_saved_toks
            .fetch_add(seq.cached_prefix_tokens as u64, Ordering::Relaxed);
    }
    m.cb_radix.release(seq.slot, seq.full_seq);
    let _ = seq.tx.send(StreamEvent::Done(GenResult {
        text: seq.out_text,
        prompt_tokens: seq.prompt_tokens,
        cached_prefix_tokens: seq.cached_prefix_tokens,
        kv_slot: seq.slot,
        completion_tokens: seq.completion_tokens,
        spec_proposed: seq.spec_proposed,
        spec_accepted: seq.spec_accepted,
        prefill_ms: seq.prefill_ms,
        decode_ms,
        finish_reason,
    }));
}

fn finish_spec_seq(
    m: &mut LoadedModel,
    seq: ActiveSpecSeq,
    finish_reason: &'static str,
    state: &ServerState,
) {
    let decode_ms = seq.t_dec.elapsed().as_secs_f64() * 1000.0;
    if seq.cached_prefix_tokens > 0 {
        state.radix_hits.fetch_add(1, Ordering::Relaxed);
        state
            .radix_saved_toks
            .fetch_add(seq.cached_prefix_tokens as u64, Ordering::Relaxed);
    }
    m.radix_cache.commit(seq.kv_slot, seq.full_seq);
    let _ = seq.tx.send(StreamEvent::Done(GenResult {
        text: seq.out_text,
        prompt_tokens: seq.prompt_tokens,
        cached_prefix_tokens: seq.cached_prefix_tokens,
        kv_slot: seq.kv_slot,
        completion_tokens: seq.completion_tokens,
        spec_proposed: seq.spec_proposed,
        spec_accepted: seq.spec_accepted,
        prefill_ms: seq.prefill_ms,
        decode_ms,
        finish_reason,
    }));
}

fn finalize_prefilled_cb(
    m: &mut LoadedModel,
    pf: PrefillingCbSeq,
    argmax0: i64,
    active: &mut Vec<ActiveCbSeq>,
    state: &ServerState,
) {
    m.cb_radix.set_active_prefix(pf.slot, &pf.ids);
    let prefill_ms = pf.t_pf.elapsed().as_secs_f64() * 1000.0;

    let mut grammar = compile_grammar_for_cfg(&pf.cfg).ok().flatten();
    let first_tok = if let Some(ref gs) = grammar {
        let (tok, ns) = m.sample_constrained(gs, &pf.cfg, &pf.ids[pf.ids.len().saturating_sub(64)..], pf.cfg.seed);
        grammar = ns;
        tok
    } else if pf.cfg.temp <= 1e-5 && pf.cfg.repeat_penalty <= 1.0001 {
        argmax0
    } else {
        if m.logits_scratch.len() < m.vocab_size {
            m.logits_scratch.resize(m.vocab_size, 0.0);
        }
        m.api.copy_logits(m.handle, &mut m.logits_scratch);
        m.sample_from_scratch_logits(None, &pf.cfg, &pf.ids[pf.ids.len().saturating_sub(64)..], pf.cfg.seed).0
    };

    let mut seq = ActiveCbSeq {
        slot: pf.slot,
        cfg: pf.cfg,
        lora_id: pf.lora_id,
        grammar,
        prompt_tokens: pf.ids.len(),
        cached_prefix_tokens: pf.actual_lcp,
        full_seq: pf.ids,
        next_tok: first_tok,
        out_text: String::new(),
        completion_tokens: 0,
        spec_proposed: 0,
        spec_accepted: 0,
        prefill_ms,
        t_dec: Instant::now(),
        tx: pf.tx,
    };

    let piece = m.api.token_str(m.handle, seq.next_tok);
    if is_stop_token(seq.next_tok, m.eos_id, &piece) || seq.cfg.max_tokens == 0 {
        finish_cb_seq(m, seq, "stop", state);
    } else {
        seq.out_text.push_str(&piece);
        seq.completion_tokens += 1;
        seq.full_seq.push(seq.next_tok);
        let alive = seq.tx.send(StreamEvent::Piece(piece)).is_ok();
        if !alive || seq.grammar.as_ref().is_some_and(GrammarState::is_strictly_done) {
            finish_cb_seq(m, seq, "stop", state);
        } else if seq.completion_tokens >= seq.cfg.max_tokens
            || seq.full_seq.len() + 2 >= m.meta.max_ctx as usize
        {
            finish_cb_seq(m, seq, "length", state);
        } else {
            active.push(seq);
        }
    }
}

/// Runs ONE lossless speculative decode iteration (`tree`, `draft`, `mtp`, or `ngram`) for `spec_active`.
/// Returns `Some(seq)` if continuing, or `None` if finished.
fn step_adaptive_spec(
    m: &mut LoadedModel,
    mut seq: ActiveSpecSeq,
    state: &ServerState,
) -> Option<ActiveSpecSeq> {
    let (_, swapped) = m.apply_lora(seq.cfg.lora.as_deref());
    if swapped {
        state.lora_swaps.fetch_add(1, Ordering::Relaxed);
    }
    m.api.kv_slot_set(m.handle, seq.kv_slot);
    let slot_ctx = m.meta.slot_ctx as usize;
    let pos = seq.full_seq.len() - 1;
    if pos + 2 >= slot_ctx || seq.completion_tokens >= seq.cfg.max_tokens {
        finish_spec_seq(m, seq, "length", state);
        return None;
    }

    let use_tree = seq.cfg.spec_mode == SpecMode::Tree && m.draft_handle > 0 && seq.cfg.spec_k >= 2;
    let use_draft = seq.cfg.spec_mode == SpecMode::Draft && m.draft_handle > 0;
    let use_mtp = seq.cfg.spec_mode == SpecMode::Mtp && m.meta.has_mtp > 0;
    let use_ngram = seq.cfg.spec_mode == SpecMode::Ngram;

    if use_tree {
        let max_d = seq.cfg.spec_k.min(4).min(seq.cfg.max_tokens - seq.completion_tokens).min(slot_ctx - pos - 2);
        let mut tree_ids = vec![seq.next_tok];
        let mut parents = vec![-1i64];
        let d1_top = m.api.step(m.draft_handle, seq.next_tok, pos as i64);
        let mut dlogits = vec![0.0f32; m.vocab_size];
        let dn = m.api.copy_logits(m.draft_handle, &mut dlogits);
        let mut second_tok = d1_top;
        if dn > 2 {
            let mut best2 = f32::NEG_INFINITY;
            for v in 0..dn {
                if v as i64 != d1_top && dlogits[v] > best2 {
                    best2 = dlogits[v];
                    second_tok = v as i64;
                }
            }
        }
        tree_ids.push(d1_top);
        parents.push(0);
        let mut cur = d1_top;
        let mut dpos = (pos + 1) as i64;
        for step_i in 1..max_d {
            let nxt = m.api.step(m.draft_handle, cur, dpos);
            tree_ids.push(nxt);
            parents.push(step_i as i64);
            cur = nxt;
            dpos += 1;
        }
        if second_tok != d1_top {
            tree_ids.push(second_tok);
            parents.push(0);
        }
        let prop = tree_ids.len() - 1;
        seq.spec_proposed += prop;
        state.adaptive_spec_steps.fetch_add(1, Ordering::Relaxed);
        state.adaptive_spec_proposed.fetch_add(prop as u64, Ordering::Relaxed);

        let mut tree_out = vec![0i64; tree_ids.len()];
        m.api.verify_tree(m.handle, &tree_ids, &parents, pos as i64, &mut tree_out);
        let mut acc_path: Vec<i64> = vec![0];
        let mut cur_node = 0usize;
        let mut bonus_tok = tree_out[0];
        let mut done_reason: Option<&'static str> = None;
        loop {
            let want = tree_out[cur_node];
            let mut matched_child = None;
            for c in (cur_node + 1)..tree_ids.len() {
                if parents[c] == cur_node as i64 && tree_ids[c] == want {
                    matched_child = Some(c);
                    break;
                }
            }
            if let Some(ch) = matched_child {
                acc_path.push(ch as i64);
                seq.spec_accepted += 1;
                state.adaptive_spec_accepted.fetch_add(1, Ordering::Relaxed);
                let dp = m.api.token_str(m.handle, want);
                if is_stop_token(want, m.eos_id, &dp) {
                    done_reason = Some("stop");
                    break;
                }
                seq.out_text.push_str(&dp);
                seq.completion_tokens += 1;
                seq.full_seq.push(want);
                if seq.tx.send(StreamEvent::Piece(dp)).is_err() {
                    done_reason = Some("stop");
                    break;
                }
                if seq.completion_tokens >= seq.cfg.max_tokens {
                    done_reason = Some("length");
                    break;
                }
                cur_node = ch;
                bonus_tok = tree_out[ch];
            } else {
                break;
            }
        }
        m.api.commit_tree_path(m.handle, &acc_path, pos as i64);
        if let Some(reason) = done_reason {
            finish_spec_seq(m, seq, reason, state);
            return None;
        }
        if acc_path.len() > 1 {
            let acc_toks: Vec<i64> = acc_path[1..].iter().map(|&ni| tree_ids[ni as usize]).collect();
            m.api.prefill(m.draft_handle, &acc_toks, (pos + 1) as i64);
        }
        seq.next_tok = bonus_tok;
    } else {
        let max_k = seq.cfg.spec_k.min(seq.cfg.max_tokens - seq.completion_tokens).min(slot_ctx - pos - 2);
        let mut draft_toks = Vec::new();
        if use_draft && max_k > 0 {
            let mut dcur = seq.next_tok;
            let mut dpos = pos as i64;
            for _ in 0..max_k {
                dcur = m.api.step(m.draft_handle, dcur, dpos);
                let dp = m.api.token_str(m.draft_handle, dcur);
                draft_toks.push(dcur);
                dpos += 1;
                if is_stop_token(dcur, m.eos_id, &dp) {
                    break;
                }
            }
        } else if use_mtp && max_k > 0 {
            let p_tok = m.api.mtp_predict(m.handle, seq.next_tok, pos as i64);
            if p_tok >= 0 && (p_tok as usize) < m.vocab_size {
                draft_toks.push(p_tok);
            }
        } else if use_ngram && max_k > 0 {
            draft_toks = propose_ngram_draft(&seq.full_seq, max_k);
        }

        if !draft_toks.is_empty() {
            seq.spec_proposed += draft_toks.len();
            state.adaptive_spec_steps.fetch_add(1, Ordering::Relaxed);
            state.adaptive_spec_proposed.fetch_add(draft_toks.len() as u64, Ordering::Relaxed);
            let mut verify_in = Vec::with_capacity(1 + draft_toks.len());
            verify_in.push(seq.next_tok);
            verify_in.extend_from_slice(&draft_toks);
            let mut target_out = vec![0i64; verify_in.len()];
            m.api.verify_batch(m.handle, &verify_in, pos as i64, &mut target_out);

            let mut n_acc = 0usize;
            for &dt in &draft_toks {
                if target_out[n_acc] == dt {
                    n_acc += 1;
                    seq.spec_accepted += 1;
                    state.adaptive_spec_accepted.fetch_add(1, Ordering::Relaxed);
                    let dp = m.api.token_str(m.handle, dt);
                    if is_stop_token(dt, m.eos_id, &dp) {
                        finish_spec_seq(m, seq, "stop", state);
                        return None;
                    }
                    seq.out_text.push_str(&dp);
                    seq.completion_tokens += 1;
                    seq.full_seq.push(dt);
                    if seq.tx.send(StreamEvent::Piece(dp)).is_err() {
                        finish_spec_seq(m, seq, "stop", state);
                        return None;
                    }
                    if seq.completion_tokens >= seq.cfg.max_tokens {
                        finish_spec_seq(m, seq, "length", state);
                        return None;
                    }
                } else {
                    break;
                }
            }
            if use_draft && n_acc == draft_toks.len() {
                let last_dt = draft_toks[n_acc - 1];
                let _ = m.api.step(m.draft_handle, last_dt, (pos + n_acc) as i64);
            }
            seq.next_tok = target_out[n_acc];
        } else {
            seq.next_tok = m.api.step(m.handle, seq.next_tok, pos as i64);
        }
    }

    // Emit the newly verified bonus/step token
    let piece = m.api.token_str(m.handle, seq.next_tok);
    if is_stop_token(seq.next_tok, m.eos_id, &piece) {
        finish_spec_seq(m, seq, "stop", state);
        return None;
    }
    seq.out_text.push_str(&piece);
    seq.completion_tokens += 1;
    seq.full_seq.push(seq.next_tok);
    if seq.tx.send(StreamEvent::Piece(piece)).is_err() {
        finish_spec_seq(m, seq, "stop", state);
        return None;
    }
    if seq.completion_tokens >= seq.cfg.max_tokens || seq.full_seq.len() + 2 >= slot_ctx {
        finish_spec_seq(m, seq, "length", state);
        return None;
    }
    Some(seq)
}

/// Tenon v8: Native Zero-Migration In-Slot Speculative Decoding directly on PagedAttention `ActiveCbSeq`!
/// Uses `paged_verify_batch(handle, seq.slot, ...)` so when a 2nd client arrives, `seq` stays in-place
/// inside its PagedKV slot with ZERO `paged_prefill` replay!
fn step_paged_spec_in_slot(
    m: &mut LoadedModel,
    mut seq: ActiveCbSeq,
    state: &ServerState,
) -> Option<ActiveCbSeq> {
    let (_, swapped) = m.apply_lora(seq.cfg.lora.as_deref());
    if swapped {
        state.lora_swaps.fetch_add(1, Ordering::Relaxed);
    }
    let max_ctx = m.meta.max_ctx as usize;
    let pos = seq.full_seq.len().saturating_sub(1);
    if pos + 2 >= max_ctx || seq.completion_tokens >= seq.cfg.max_tokens {
        finish_cb_seq(m, seq, "length", state);
        return None;
    }
    let use_draft = (seq.cfg.spec_mode == SpecMode::Draft || seq.cfg.spec_mode == SpecMode::Tree) && m.draft_handle > 0;
    let use_mtp = seq.cfg.spec_mode == SpecMode::Mtp && m.meta.has_mtp > 0;
    let use_ngram = seq.cfg.spec_mode == SpecMode::Ngram
        || (seq.cfg.spec_mode == SpecMode::Draft && m.draft_handle <= 0)
        || (seq.cfg.spec_mode == SpecMode::Tree && m.draft_handle <= 0);
    let max_k = seq.cfg.spec_k.min(seq.cfg.max_tokens - seq.completion_tokens).min(max_ctx - pos - 2);
    let mut draft_toks = Vec::new();
    if use_draft && max_k > 0 {
        let mut dcur = seq.next_tok;
        let mut dpos = pos as i64;
        for _ in 0..max_k {
            dcur = m.api.step(m.draft_handle, dcur, dpos);
            let dp = m.api.token_str(m.draft_handle, dcur);
            draft_toks.push(dcur);
            dpos += 1;
            if is_stop_token(dcur, m.eos_id, &dp) {
                break;
            }
        }
    } else if use_mtp && max_k > 0 {
        let p_tok = m.api.mtp_predict(m.handle, seq.next_tok, pos as i64);
        if p_tok >= 0 && (p_tok as usize) < m.vocab_size {
            draft_toks.push(p_tok);
        }
    } else if use_ngram && max_k > 0 {
        draft_toks = propose_ngram_draft(&seq.full_seq, max_k);
    }

    if !draft_toks.is_empty() {
        seq.spec_proposed += draft_toks.len();
        state.adaptive_spec_steps.fetch_add(1, Ordering::Relaxed);
        state.adaptive_spec_proposed.fetch_add(draft_toks.len() as u64, Ordering::Relaxed);
        let mut verify_in = Vec::with_capacity(1 + draft_toks.len());
        verify_in.push(seq.next_tok);
        verify_in.extend_from_slice(&draft_toks);
        let mut target_out = vec![0i64; verify_in.len()];
        m.api.paged_verify_batch(m.handle, seq.slot, &verify_in, pos as i64, &mut target_out);

        let mut n_acc = 0usize;
        for &dt in &draft_toks {
            if target_out[n_acc] == dt {
                n_acc += 1;
                seq.spec_accepted += 1;
                state.adaptive_spec_accepted.fetch_add(1, Ordering::Relaxed);
                let dp = m.api.token_str(m.handle, dt);
                if is_stop_token(dt, m.eos_id, &dp) {
                    finish_cb_seq(m, seq, "stop", state);
                    return None;
                }
                seq.out_text.push_str(&dp);
                seq.completion_tokens += 1;
                seq.full_seq.push(dt);
                if seq.tx.send(StreamEvent::Piece(dp)).is_err() {
                    finish_cb_seq(m, seq, "stop", state);
                    return None;
                }
                if seq.completion_tokens >= seq.cfg.max_tokens {
                    finish_cb_seq(m, seq, "length", state);
                    return None;
                }
            } else {
                break;
            }
        }
        if use_draft && n_acc == draft_toks.len() {
            let last_dt = draft_toks[n_acc - 1];
            let _ = m.api.step(m.draft_handle, last_dt, (pos + n_acc) as i64);
        }
        seq.next_tok = target_out[n_acc];
    } else {
        let ids1 = [seq.next_tok];
        let pos1 = [pos as i64];
        let slt1 = [seq.slot as i64];
        let mut out1 = [0i64];
        m.api.step_batch(m.handle, &ids1, &pos1, &slt1, &mut out1);
        seq.next_tok = out1[0];
    }

    let piece = m.api.token_str(m.handle, seq.next_tok);
    if is_stop_token(seq.next_tok, m.eos_id, &piece) {
        finish_cb_seq(m, seq, "stop", state);
        return None;
    }
    seq.out_text.push_str(&piece);
    seq.completion_tokens += 1;
    seq.full_seq.push(seq.next_tok);
    if seq.tx.send(StreamEvent::Piece(piece)).is_err() {
        finish_cb_seq(m, seq, "stop", state);
        return None;
    }
    if seq.completion_tokens >= seq.cfg.max_tokens || seq.full_seq.len() + 2 >= max_ctx {
        finish_cb_seq(m, seq, "length", state);
        return None;
    }
    Some(seq)
}

fn cb_engine_loop(mut m: LoadedModel, rx: std_mpsc::Receiver<EngineCmd>, state: Arc<ServerState>) {
    let mut active: Vec<ActiveCbSeq> = Vec::new();
    let mut prefill_queue: Vec<PrefillingCbSeq> = Vec::new();
    let mut spec_active: Option<ActiveSpecSeq> = None;
    let mut pending_gen: Vec<PendingGenReq> = Vec::new();
    let max_cb_batch = 16usize;
    let chunk_sz = m.prefill_chunk_size.max(16);

    loop {
        if active.is_empty() && prefill_queue.is_empty() && spec_active.is_none() && pending_gen.is_empty() {
            match rx.recv() {
                Ok(cmd) => {
                    if !handle_engine_cmd(&mut m, cmd, &mut pending_gen, &state) {
                        continue;
                    }
                }
                Err(_) => break,
            }
        }
        while let Ok(cmd) = rx.try_recv() {
            handle_engine_cmd(&mut m, cmd, &mut pending_gen, &state);
        }

        // Legacy linear fallback migration (only used if a model ran outside PagedAttention)
        if spec_active.is_some() && !pending_gen.is_empty() && m.meta.supports_cb {
            if let Some(sseq) = spec_active.take() {
                let (lora_id, swapped) = m.apply_lora(sseq.cfg.lora.as_deref());
                if swapped {
                    state.lora_swaps.fetch_add(1, Ordering::Relaxed);
                }
                m.radix_cache.commit(sseq.kv_slot, sseq.full_seq.clone());
                if let Some((dst_slot, src_slot, lcp)) = m.cb_radix.match_for_cow_fork(&sseq.full_seq) {
                    if lcp > 0 && src_slot != dst_slot {
                        m.api.paged_seq_fork(m.handle, src_slot, dst_slot, lcp);
                    } else {
                        m.api.paged_seq_free(m.handle, dst_slot);
                    }
                    let actual_lcp = if lcp > 0 && src_slot != dst_slot { lcp } else { 0 };
                    let prefill_end = sseq.full_seq.len().saturating_sub(1);
                    if prefill_end > actual_lcp {
                        let _ = m.api.paged_prefill(
                            m.handle,
                            dst_slot,
                            &sseq.full_seq[actual_lcp..prefill_end],
                            actual_lcp as i64,
                        );
                    }
                    m.cb_radix.set_active_prefix(dst_slot, &sseq.full_seq);
                    active.push(ActiveCbSeq {
                        slot: dst_slot,
                        cfg: sseq.cfg,
                        lora_id,
                        grammar: None,
                        full_seq: sseq.full_seq,
                        next_tok: sseq.next_tok,
                        out_text: sseq.out_text,
                        prompt_tokens: sseq.prompt_tokens,
                        cached_prefix_tokens: sseq.cached_prefix_tokens,
                        completion_tokens: sseq.completion_tokens,
                        spec_proposed: sseq.spec_proposed,
                        spec_accepted: sseq.spec_accepted,
                        prefill_ms: sseq.prefill_ms,
                        t_dec: sseq.t_dec,
                        tx: sseq.tx,
                    });
                } else {
                    spec_active = Some(sseq);
                }
            }
        }

        if let Some(sseq) = spec_active.take() {
            state.active_batch.store(1, Ordering::Relaxed);
            spec_active = step_adaptive_spec(&mut m, sseq, &state);
            continue;
        }

        // Admit waiting requests into Universal PagedAttention CB (with Native Zero-Migration In-Slot Spec!)
        while !pending_gen.is_empty() {
            let front = &pending_gen[0];
            let has_mm = !front.mm_parts.is_empty();
            // When `supports_cb` is true, speculative requests enter PagedAttention CB directly (ZERO migration!).
            // Linear `ActiveSpecSeq` is only used when `!supports_cb`.
            let want_legacy_spec = !m.meta.supports_cb
                && !has_mm
                && front.cfg.spec_mode != SpecMode::Off
                && front.cfg.temp <= 1e-5
                && !front.cfg.json_constrained
                && front.cfg.json_schema.is_none()
                && front.cfg.gbnf_grammar.is_none()
                && active.is_empty()
                && prefill_queue.is_empty()
                && pending_gen.len() == 1;

            if want_legacy_spec {
                let req = pending_gen.remove(0);
                let (_, swapped) = m.apply_lora(req.cfg.lora.as_deref());
                if swapped {
                    state.lora_swaps.fetch_add(1, Ordering::Relaxed);
                }
                let slot_ctx = m.meta.slot_ctx as usize;
                let max_prompt = slot_ctx.saturating_sub(req.cfg.max_tokens.min(512) + 8).max(16);
                let mut ids = Vec::new();
                if req.add_bos && m.bos_id >= 0 {
                    ids.push(m.bos_id);
                }
                ids.extend(m.api.tokenize(m.handle, &req.prompt, max_prompt));
                if ids.is_empty() {
                    ids.push(m.bos_id.max(1));
                }
                let prompt_tokens = ids.len();
                let (kv_slot, lcp) = m.radix_cache.match_and_acquire(&ids);
                m.api.kv_slot_set(m.handle, kv_slot);
                let t_pf = Instant::now();
                let first_tok = m.api.prefill(m.handle, &ids[lcp..], lcp as i64);
                let prefill_ms = t_pf.elapsed().as_secs_f64() * 1000.0;
                m.api.reset(m.handle);
                if (req.cfg.spec_mode == SpecMode::Draft || req.cfg.spec_mode == SpecMode::Tree)
                    && m.draft_handle > 0
                {
                    m.api.prefill(m.draft_handle, &ids, 0);
                    m.api.reset(m.draft_handle);
                }
                let mut sseq = ActiveSpecSeq {
                    kv_slot,
                    cfg: req.cfg,
                    full_seq: ids,
                    next_tok: first_tok,
                    out_text: String::new(),
                    prompt_tokens,
                    cached_prefix_tokens: lcp,
                    completion_tokens: 0,
                    spec_proposed: 0,
                    spec_accepted: 0,
                    prefill_ms,
                    t_dec: Instant::now(),
                    tx: req.tx,
                };
                let piece0 = m.api.token_str(m.handle, first_tok);
                if is_stop_token(first_tok, m.eos_id, &piece0) || sseq.cfg.max_tokens == 0 {
                    finish_spec_seq(&mut m, sseq, "stop", &state);
                } else {
                    sseq.out_text.push_str(&piece0);
                    sseq.completion_tokens += 1;
                    sseq.full_seq.push(first_tok);
                    if sseq.tx.send(StreamEvent::Piece(piece0)).is_err() {
                        finish_spec_seq(&mut m, sseq, "stop", &state);
                    } else if sseq.completion_tokens >= sseq.cfg.max_tokens {
                        finish_spec_seq(&mut m, sseq, "length", &state);
                    } else {
                        spec_active = Some(sseq);
                    }
                }
                break;
            }

            let need_legacy = !m.meta.supports_cb || has_mm;
            if need_legacy {
                if !active.is_empty() || !prefill_queue.is_empty() {
                    break;
                }
                let req = pending_gen.remove(0);
                let res = if !req.mm_parts.is_empty() && (m.mmproj_handle > 0 || m.audio_handle > 0) {
                    let prev_vhits = m.vision_cache_hits;
                    let r = run_multimodal_generation(&mut m, &req.mm_parts, &req.cfg, |p| {
                        req.tx.send(StreamEvent::Piece(p.to_string())).is_ok()
                    });
                    if m.vision_cache_hits > prev_vhits {
                        state
                            .vision_cache_hits
                            .fetch_add(m.vision_cache_hits - prev_vhits, Ordering::Relaxed);
                    }
                    state.vision_cache_entries.store(m.vision_cache.len(), Ordering::Relaxed);
                    r
                } else {
                    run_generation(&mut m, &req.prompt, req.add_bos, &req.cfg, |p| {
                        req.tx.send(StreamEvent::Piece(p.to_string())).is_ok()
                    })
                };
                if res.cached_prefix_tokens > 0 {
                    state.radix_hits.fetch_add(1, Ordering::Relaxed);
                    state
                        .radix_saved_toks
                        .fetch_add(res.cached_prefix_tokens as u64, Ordering::Relaxed);
                }
                let _ = req.tx.send(StreamEvent::Done(res));
                continue;
            }

            if active.len() + prefill_queue.len() >= max_cb_batch {
                break;
            }
            let req = pending_gen.remove(0);
            let max_prompt = (m.meta.max_ctx as usize).saturating_sub(req.cfg.max_tokens.min(512) + 8).max(16);
            let mut ids = Vec::new();
            if req.add_bos && m.bos_id >= 0 {
                ids.push(m.bos_id);
            }
            ids.extend(m.api.tokenize(m.handle, &req.prompt, max_prompt));
            if ids.is_empty() {
                ids.push(m.bos_id.max(1));
            }
            // If a single speculative request was active and a 2nd request just arrived, record ZERO-migration transition!
            if active.len() == 1 && active[0].spec_proposed > 0 {
                state.spec_zero_migrations.fetch_add(1, Ordering::Relaxed);
            }
            if (req.cfg.spec_mode == SpecMode::Draft || req.cfg.spec_mode == SpecMode::Tree)
                && m.draft_handle > 0
                && active.is_empty()
                && prefill_queue.is_empty()
            {
                m.api.prefill(m.draft_handle, &ids, 0);
                m.api.reset(m.draft_handle);
            }
            let Some((dst_slot, src_slot, lcp)) = m.cb_radix.match_for_cow_fork(&ids) else {
                pending_gen.insert(0, req);
                break;
            };
            let (lora_id, _) = m.resolve_lora(req.cfg.lora.as_deref());
            // Do not COW-fork KV across incompatible LoRA adapters!
            let allow_cow = lcp > 0 && src_slot != dst_slot && lora_id == 0;
            if allow_cow {
                m.api.paged_seq_fork(m.handle, src_slot, dst_slot, lcp);
            } else {
                m.api.paged_seq_free(m.handle, dst_slot);
            }
            let actual_lcp = if allow_cow { lcp } else { 0 };
            let remaining = ids.len().saturating_sub(actual_lcp);
            prefill_queue.push(PrefillingCbSeq {
                slot: dst_slot,
                cfg: req.cfg,
                lora_id,
                ids,
                actual_lcp,
                cursor: actual_lcp,
                is_chunked: remaining > chunk_sz,
                t_pf: Instant::now(),
                tx: req.tx,
            });
        }

        if spec_active.is_some() {
            continue;
        }

        // Chunked Prefill Interleaving: advance AT MOST `chunk_sz` tokens per tick when decoders are active
        // (or finish short <= chunk_sz prompts immediately when idle so bursts batch together on tick 1!).
        let mut chunk_budget = chunk_sz;
        let pfi = 0usize;
        while pfi < prefill_queue.len() && chunk_budget > 0 {
            let rem = prefill_queue[pfi].ids.len().saturating_sub(prefill_queue[pfi].cursor);
            let take = rem.min(chunk_budget);
            let mut pf = prefill_queue.remove(pfi);
            let (_, swapped) = m.apply_lora(pf.cfg.lora.as_deref());
            if swapped {
                state.lora_swaps.fetch_add(1, Ordering::Relaxed);
            }
            let end = (pf.cursor + take).min(pf.ids.len());
            let argmax = if end > pf.cursor {
                m.api.paged_prefill(m.handle, pf.slot, &pf.ids[pf.cursor..end], pf.cursor as i64)
            } else {
                m.bos_id
            };
            pf.cursor = end;
            if pf.is_chunked {
                state.chunked_prefill_chunks.fetch_add(1, Ordering::Relaxed);
            }
            chunk_budget = chunk_budget.saturating_sub(take);
            if pf.cursor >= pf.ids.len() {
                finalize_prefilled_cb(&mut m, pf, argmax, &mut active, &state);
            } else {
                prefill_queue.insert(pfi, pf);
                break; // Yield remaining prompt chunks to next tick so active decoders stream smoothly!
            }
            if !active.is_empty() && chunk_budget == 0 {
                break;
            }
        }

        if active.is_empty() {
            state.active_batch.store(0, Ordering::Relaxed);
            continue;
        }

        // Tenon v8 Native Zero-Migration In-Slot Speculative Step:
        // When `active.len() == 1 && prefill_queue.is_empty()` and speculative mode is active,
        // verify K+1 tokens directly IN-PLACE inside `active[0].slot` using `paged_verify_batch`!
        if active.len() == 1
            && prefill_queue.is_empty()
            && active[0].cfg.spec_mode != SpecMode::Off
            && active[0].cfg.temp <= 1e-5
            && active[0].grammar.is_none()
        {
            state.active_batch.store(1, Ordering::Relaxed);
            let seq = active.remove(0);
            if let Some(next_seq) = step_paged_spec_in_slot(&mut m, seq, &state) {
                active.push(next_seq);
            }
            continue;
        }

        // ONE Iteration-Level Continuous Batching Decode Step across ALL `B = active.len()` clients!
        // Partition seamlessly by `lora_id` if concurrent clients use different LoRAs (zero cross-client interference!).
        let b_sz = active.len();
        state.active_batch.store(b_sz, Ordering::Relaxed);
        state.cb_batches_run.fetch_add(1, Ordering::Relaxed);
        let prev_max = state.cb_max_batch.load(Ordering::Relaxed);
        if b_sz > prev_max {
            state.cb_max_batch.store(b_sz, Ordering::Relaxed);
        }

        if m.logits_scratch.len() < m.vocab_size {
            m.logits_scratch.resize(m.vocab_size, 0.0);
        }

        let mut unique_loras: Vec<i64> = Vec::new();
        for s in &active {
            if !unique_loras.contains(&s.lora_id) {
                unique_loras.push(s.lora_id);
            }
        }

        let mut chosen_toks = vec![0i64; b_sz];
        let mut next_grammars: Vec<Option<GrammarState>> = (0..b_sz).map(|_| None).collect();

        for lid in unique_loras {
            let group_idx: Vec<usize> = active
                .iter()
                .enumerate()
                .filter_map(|(i, s)| if s.lora_id == lid { Some(i) } else { None })
                .collect();
            let lora_name = active[group_idx[0]].cfg.lora.as_deref();
            let (_, swapped) = m.apply_lora(lora_name);
            if swapped {
                state.lora_swaps.fetch_add(1, Ordering::Relaxed);
            }
            let gb = group_idx.len();
            let tok_ids: Vec<i64> = group_idx.iter().map(|&i| active[i].next_tok).collect();
            let pos_arr: Vec<i64> = group_idx.iter().map(|&i| (active[i].full_seq.len() - 1) as i64).collect();
            let slot_arr: Vec<i64> = group_idx.iter().map(|&i| active[i].slot as i64).collect();
            let mut argmax_out = vec![0i64; gb];
            m.api.step_batch(m.handle, &tok_ids, &pos_arr, &slot_arr, &mut argmax_out);

            for (gbi, &orig_i) in group_idx.iter().enumerate() {
                let seq = &active[orig_i];
                let need_logits = seq.grammar.is_some()
                    || seq.cfg.temp > 1e-5
                    || seq.cfg.repeat_penalty > 1.0001;
                if need_logits {
                    m.api.copy_logits_batch(m.handle, gbi, &mut m.logits_scratch);
                    let rstart = seq.full_seq.len().saturating_sub(64);
                    let step_seed = seq
                        .cfg
                        .seed
                        .wrapping_add((seq.completion_tokens as i64).wrapping_mul(1000003));
                    let (tok, ns) = m.sample_from_scratch_logits(
                        seq.grammar.as_ref(),
                        &seq.cfg,
                        &seq.full_seq[rstart..],
                        step_seed,
                    );
                    chosen_toks[orig_i] = tok;
                    next_grammars[orig_i] = ns;
                } else {
                    chosen_toks[orig_i] = argmax_out[gbi];
                }
            }
        }

        let mut next_active = Vec::with_capacity(b_sz);
        for (bi, mut seq) in active.drain(..).enumerate() {
            let chosen = chosen_toks[bi];
            if seq.grammar.is_some() {
                seq.grammar = next_grammars[bi].take();
            }

            let piece = m.api.token_str(m.handle, chosen);
            if is_stop_token(chosen, m.eos_id, &piece) {
                finish_cb_seq(&mut m, seq, "stop", &state);
                continue;
            }
            seq.next_tok = chosen;
            seq.out_text.push_str(&piece);
            seq.completion_tokens += 1;
            seq.full_seq.push(chosen);
            let alive = seq.tx.send(StreamEvent::Piece(piece)).is_ok();
            if !alive || seq.grammar.as_ref().is_some_and(GrammarState::is_strictly_done) {
                finish_cb_seq(&mut m, seq, "stop", &state);
            } else if seq.completion_tokens >= seq.cfg.max_tokens
                || seq.full_seq.len() + 2 >= m.meta.max_ctx as usize
            {
                finish_cb_seq(&mut m, seq, "length", &state);
            } else {
                next_active.push(seq);
            }
        }
        active = next_active;
    }
}

fn handle_engine_cmd(
    m: &mut LoadedModel,
    cmd: EngineCmd,
    pending_gen: &mut Vec<PendingGenReq>,
    _state: &ServerState,
) -> bool {
    match cmd {
        EngineCmd::Generate { prompt, add_bos, cfg, mm_parts, tx } => {
            pending_gen.push(PendingGenReq { prompt, add_bos, cfg, mm_parts, tx });
            true
        }
        EngineCmd::Embed { inputs, reply } => {
            let dm = m.meta.d_model;
            let mut vecs = Vec::with_capacity(inputs.len());
            let mut total_toks = 0usize;
            for txt in &inputs {
                let mut ids = Vec::new();
                if m.bos_id >= 0 {
                    ids.push(m.bos_id);
                }
                ids.extend(m.api.tokenize(m.handle, txt, 512));
                if ids.is_empty() {
                    ids.push(m.bos_id.max(1));
                }
                total_toks += ids.len();
                let mut emb = vec![0.0f32; dm];
                let got = m.api.embed_text(m.handle, &ids, &mut emb);
                emb.truncate(got);
                vecs.push(emb);
            }
            let _ = reply.send(Ok((vecs, total_toks)));
            false
        }
        EngineCmd::Rerank { query, documents, top_n, reply } => {
            // Cross-encoder LLM reranking via P("Yes") / (P("Yes") + P("No")) logit calibration
            // (matches BGE-Reranker-v2-Gemma / Qwen3-Reranker / vLLM score semantics on causal LLMs),
            // with automatic fallback to L2-normalized hidden-state cosine similarity if Yes/No tokens are absent.
            let mut yes_ids: Vec<usize> = Vec::new();
            let mut no_ids: Vec<usize> = Vec::new();
            for s in ["Yes", " Yes", "yes", " yes"] {
                let toks = m.api.tokenize(m.handle, s, 4);
                if toks.len() == 1 && (toks[0] as usize) < m.vocab_size {
                    let id = toks[0] as usize;
                    if !yes_ids.contains(&id) {
                        yes_ids.push(id);
                    }
                }
            }
            for s in ["No", " No", "no", " no"] {
                let toks = m.api.tokenize(m.handle, s, 4);
                if toks.len() == 1 && (toks[0] as usize) < m.vocab_size {
                    let id = toks[0] as usize;
                    if !no_ids.contains(&id) {
                        no_ids.push(id);
                    }
                }
            }
            let mut scored: Vec<(usize, f32)> = Vec::with_capacity(documents.len());
            if !yes_ids.is_empty() && !no_ids.is_empty() {
                if m.logits_scratch.len() < m.vocab_size {
                    m.logits_scratch.resize(m.vocab_size, 0.0);
                }
                for (idx, doc) in documents.iter().enumerate() {
                    let prompt = format!(
                        "Determine whether the Document is relevant to the Query. Respond with Yes or No.\nQuery: {}\nDocument: {}\nRelevant:",
                        query, doc
                    );
                    let mut ids = Vec::new();
                    if m.bos_id >= 0 {
                        ids.push(m.bos_id);
                    }
                    ids.extend(m.api.tokenize(m.handle, &prompt, 512));
                    if ids.is_empty() {
                        ids.push(m.bos_id.max(1));
                    }
                    let _ = m.api.prefill(m.handle, &ids, 0);
                    m.api.copy_logits(m.handle, &mut m.logits_scratch);
                    let mut y_max = f32::NEG_INFINITY;
                    for &id in &yes_ids {
                        y_max = y_max.max(m.logits_scratch[id]);
                    }
                    let mut n_max = f32::NEG_INFINITY;
                    for &id in &no_ids {
                        n_max = n_max.max(m.logits_scratch[id]);
                    }
                    let prob = 1.0 / (1.0 + (n_max - y_max).exp());
                    scored.push((idx, prob));
                }
            } else {
                let dm = m.meta.d_model;
                let embed_one = |m: &LoadedModel, txt: &str| -> Vec<f32> {
                    let mut ids = Vec::new();
                    if m.bos_id >= 0 {
                        ids.push(m.bos_id);
                    }
                    ids.extend(m.api.tokenize(m.handle, txt, 512));
                    if ids.is_empty() {
                        ids.push(m.bos_id.max(1));
                    }
                    let mut emb = vec![0.0f32; dm];
                    let got = m.api.embed_text(m.handle, &ids, &mut emb);
                    emb.truncate(got);
                    emb
                };
                let qv = embed_one(m, &query);
                for (idx, doc) in documents.iter().enumerate() {
                    let dv = embed_one(m, doc);
                    let dot: f32 = qv.iter().zip(dv.iter()).map(|(a, b)| a * b).sum();
                    scored.push((idx, dot));
                }
            }
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            if top_n > 0 && scored.len() > top_n {
                scored.truncate(top_n);
            }
            let _ = reply.send(Ok(scored));
            false
        }
        EngineCmd::PagedStats { reply } => {
            let total = m.api.paged_kv_info(m.handle, 1);
            let free = m.api.paged_kv_info(m.handle, 2);
            let forks = m.api.paged_kv_info(m.handle, 4);
            let copies = m.api.paged_kv_info(m.handle, 5);
            let _ = reply.send((total, free, forks, copies));
            false
        }
    }
}

async fn read_http_request(stream: &mut TcpStream) -> io::Result<(String, String, String)> {
    let mut buf = Vec::with_capacity(4096);
    let mut tmp = [0u8; 8192];
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
        if buf.len() > 24 * 1024 * 1024 {
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
    let rf_schema = v
        .get("response_format")
        .and_then(|rf| rf.get("json_schema"))
        .and_then(|js| js.get("schema").or(Some(js)))
        .cloned()
        .filter(|s| s.is_object());
    let gbnf_grammar = v
        .get("grammar")
        .and_then(Value::as_str)
        .map(String::from)
        .or_else(|| default_cfg.gbnf_grammar.clone());
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
        json_schema: rf_schema.or_else(|| default_cfg.json_schema.clone()),
        gbnf_grammar,
        spec_mode: default_cfg.spec_mode,
        spec_k: default_cfg.spec_k,
        lora: v
            .get("lora")
            .or_else(|| v.get("adapter"))
            .and_then(Value::as_str)
            .map(String::from)
            .or_else(|| default_cfg.lora.clone()),
        anyres: v.get("anyres").and_then(Value::as_bool).unwrap_or(default_cfg.anyres),
        mrope_grid: v
            .get("mrope_grid")
            .and_then(Value::as_array)
            .and_then(|a| {
                if a.len() == 3 {
                    Some((a[0].as_i64()?, a[1].as_i64()?, a[2].as_i64()?))
                } else {
                    None
                }
            })
            .or(default_cfg.mrope_grid),
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
        let (pa_tot, pa_free, pa_forks, pa_copies) = {
            let (tx, rx) = tokio::sync::oneshot::channel();
            if state.engine_tx.send(EngineCmd::PagedStats { reply: tx }).is_ok() {
                rx.await.unwrap_or((0, 0, 0, 0))
            } else {
                (0, 0, 0, 0)
            }
        };
        let mut info = json!({
            "status": "ok",
            "version": "tenon-v8",
            "backend": state.meta.backend,
            "model": state.meta.model_id,
            "arch": state.meta.arch,
            "d_model": state.meta.d_model,
            "max_ctx": state.meta.max_ctx,
            "slot_ctx": state.meta.slot_ctx,
            "kv_slots": state.meta.n_kv_slots,
            "kv_quant": if state.meta.kv_q8 { "q8_0" } else { "fp16" },
            "mla_latent_kv": state.meta.mla_latent,
            "bpe_merges": state.meta.n_bpe_merges,
            "tp_size": state.meta.tp_size,
            "ep_size": state.meta.ep_size,
            "hybrid_2d_tp_ep": state.meta.hybrid_2d,
            "nccl_active": state.meta.nccl_active,
            "has_mtp": state.meta.has_mtp,
            "has_mmproj": state.meta.has_mmproj,
            "has_audio": state.meta.has_audio,
            "audio_proj_dim": state.meta.audio_proj_dim,
            "speculative_draft": state.meta.has_draft,
            "continuous_batching": state.meta.supports_cb
        });
        if let Some(obj) = info.as_object_mut() {
            obj.insert("vision_cache_hits".into(), json!(state.vision_cache_hits.load(Ordering::Relaxed)));
            obj.insert("vision_cache_entries".into(), json!(state.vision_cache_entries.load(Ordering::Relaxed)));
            obj.insert("prefill_chunk_size".into(), json!(state.meta.prefill_chunk_size));
            obj.insert("chunked_prefill_chunks".into(), json!(state.chunked_prefill_chunks.load(Ordering::Relaxed)));
            obj.insert("adaptive_spec_steps".into(), json!(state.adaptive_spec_steps.load(Ordering::Relaxed)));
            obj.insert("adaptive_spec_proposed".into(), json!(state.adaptive_spec_proposed.load(Ordering::Relaxed)));
            obj.insert("adaptive_spec_accepted".into(), json!(state.adaptive_spec_accepted.load(Ordering::Relaxed)));
            obj.insert("spec_zero_migrations".into(), json!(state.spec_zero_migrations.load(Ordering::Relaxed)));
            obj.insert("lora_adapters".into(), json!(state.meta.lora_adapters));
            obj.insert("lora_swaps".into(), json!(state.lora_swaps.load(Ordering::Relaxed)));
            obj.insert("active_batch_size".into(), json!(state.active_batch.load(Ordering::Relaxed)));
            obj.insert("cb_batches_run".into(), json!(state.cb_batches_run.load(Ordering::Relaxed)));
            obj.insert("cb_max_batch_seen".into(), json!(state.cb_max_batch.load(Ordering::Relaxed)));
            obj.insert("paged_kv_blocks".into(), json!(pa_tot));
            obj.insert("paged_kv_free_blocks".into(), json!(pa_free));
            obj.insert("paged_cow_forks".into(), json!(pa_forks));
            obj.insert("paged_cow_copies".into(), json!(pa_copies));
            obj.insert("queued_requests".into(), json!(state.queued_reqs.load(Ordering::Relaxed)));
            obj.insert("total_requests".into(), json!(state.total_reqs.load(Ordering::Relaxed)));
            obj.insert("radix_prefix_hits".into(), json!(state.radix_hits.load(Ordering::Relaxed)));
            obj.insert("radix_saved_tokens".into(), json!(state.radix_saved_toks.load(Ordering::Relaxed)));
        }
        let _ = write_json_response(&mut stream, "200 OK", &info).await;
        return;
    }

    if method == "GET" && clean_path == "/metrics" {
        let (pa_tot, pa_free, pa_forks, pa_copies) = {
            let (tx, rx) = tokio::sync::oneshot::channel();
            if state.engine_tx.send(EngineCmd::PagedStats { reply: tx }).is_ok() {
                rx.await.unwrap_or((0, 0, 0, 0))
            } else {
                (0, 0, 0, 0)
            }
        };
        let body = format!(
            "# HELP tenon_requests_total Total HTTP generation requests\n\
             # TYPE tenon_requests_total counter\n\
             tenon_requests_total {}\n\
             # HELP tenon_active_batch_size Current active continuous batch size\n\
             # TYPE tenon_active_batch_size gauge\n\
             tenon_active_batch_size {}\n\
             # HELP tenon_cb_batches_total Total continuous batching decode iterations\n\
             # TYPE tenon_cb_batches_total counter\n\
             tenon_cb_batches_total {}\n\
             # HELP tenon_paged_kv_blocks_total Total PagedAttention physical KV blocks\n\
             # TYPE tenon_paged_kv_blocks_total gauge\n\
             tenon_paged_kv_blocks_total {}\n\
             # HELP tenon_paged_kv_blocks_free Free PagedAttention physical KV blocks\n\
             # TYPE tenon_paged_kv_blocks_free gauge\n\
             tenon_paged_kv_blocks_free {}\n\
             # HELP tenon_paged_cow_forks_total Zero-copy COW prefix forks\n\
             # TYPE tenon_paged_cow_forks_total counter\n\
             tenon_paged_cow_forks_total {}\n\
             # HELP tenon_paged_cow_copies_total Tail block COW splits\n\
             # TYPE tenon_paged_cow_copies_total counter\n\
             tenon_paged_cow_copies_total {}\n\
             # HELP tenon_chunked_prefill_chunks_total Chunked prefill ticks executed\n\
             # TYPE tenon_chunked_prefill_chunks_total counter\n\
             tenon_chunked_prefill_chunks_total {}\n\
             # HELP tenon_spec_zero_migrations_total Zero-migration Spec-to-CB transitions\n\
             # TYPE tenon_spec_zero_migrations_total counter\n\
             tenon_spec_zero_migrations_total {}\n\
             # HELP tenon_vision_cache_hits_total Vision content-hash prefix cache hits\n\
             # TYPE tenon_vision_cache_hits_total counter\n\
             tenon_vision_cache_hits_total {}\n\
             # HELP tenon_lora_swaps_total Zero-copy runtime LoRA adapter swaps\n\
             # TYPE tenon_lora_swaps_total counter\n\
             tenon_lora_swaps_total {}\n",
            state.total_reqs.load(Ordering::Relaxed),
            state.active_batch.load(Ordering::Relaxed),
            state.cb_batches_run.load(Ordering::Relaxed),
            pa_tot,
            pa_free,
            pa_forks,
            pa_copies,
            state.chunked_prefill_chunks.load(Ordering::Relaxed),
            state.spec_zero_migrations.load(Ordering::Relaxed),
            state.vision_cache_hits.load(Ordering::Relaxed),
            state.lora_swaps.load(Ordering::Relaxed),
        );
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nAccess-Control-Allow-Origin: *\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let _ = stream.write_all(resp.as_bytes()).await;
        let _ = stream.flush().await;
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

    // Tenon v6: OpenAI-compatible `POST /v1/embeddings`
    if method == "POST" && clean_path == "/v1/embeddings" {
        let Ok(req) = serde_json::from_str::<Value>(&body) else {
            let _ = write_json_response(&mut stream, "400 Bad Request", &json!({"error": {"message": "invalid JSON"}})).await;
            return;
        };
        let mut inputs = Vec::new();
        match req.get("input") {
            Some(Value::String(s)) => inputs.push(s.clone()),
            Some(Value::Array(arr)) => {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        inputs.push(s.to_string());
                    }
                }
            }
            _ => {}
        }
        if inputs.is_empty() {
            let _ = write_json_response(&mut stream, "400 Bad Request", &json!({"error": {"message": "missing `input`"}})).await;
            return;
        }
        let (tx, rx) = tokio::sync::oneshot::channel();
        if state.engine_tx.send(EngineCmd::Embed { inputs, reply: tx }).is_err() {
            let _ = write_json_response(&mut stream, "500 Internal Server Error", &json!({"error": {"message": "engine stopped"}})).await;
            return;
        }
        match rx.await {
            Ok(Ok((vecs, total_toks))) => {
                let data: Vec<Value> = vecs
                    .into_iter()
                    .enumerate()
                    .map(|(idx, emb)| json!({"object": "embedding", "index": idx, "embedding": emb}))
                    .collect();
                let resp = json!({
                    "object": "list",
                    "model": state.meta.model_id,
                    "data": data,
                    "usage": {"prompt_tokens": total_toks, "total_tokens": total_toks}
                });
                let _ = write_json_response(&mut stream, "200 OK", &resp).await;
            }
            _ => {
                let _ = write_json_response(&mut stream, "500 Internal Server Error", &json!({"error": {"message": "embedding failed"}})).await;
            }
        }
        return;
    }

    // Tenon v6: Cohere/vLLM-compatible `POST /v1/rerank`
    if method == "POST" && clean_path == "/v1/rerank" {
        let Ok(req) = serde_json::from_str::<Value>(&body) else {
            let _ = write_json_response(&mut stream, "400 Bad Request", &json!({"error": {"message": "invalid JSON"}})).await;
            return;
        };
        let query = req.get("query").and_then(Value::as_str).unwrap_or("").to_string();
        let docs: Vec<String> = req
            .get("documents")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|d| d.as_str().or_else(|| d.get("text").and_then(Value::as_str)).map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let top_n = req.get("top_n").and_then(Value::as_u64).unwrap_or(0) as usize;
        if query.is_empty() || docs.is_empty() {
            let _ = write_json_response(&mut stream, "400 Bad Request", &json!({"error": {"message": "missing `query` or `documents`"}})).await;
            return;
        }
        let docs_copy = docs.clone();
        let (tx, rx) = tokio::sync::oneshot::channel();
        if state.engine_tx.send(EngineCmd::Rerank { query, documents: docs, top_n, reply: tx }).is_err() {
            let _ = write_json_response(&mut stream, "500 Internal Server Error", &json!({"error": {"message": "engine stopped"}})).await;
            return;
        }
        match rx.await {
            Ok(Ok(scored)) => {
                let results: Vec<Value> = scored
                    .into_iter()
                    .map(|(idx, score)| json!({
                        "index": idx,
                        "relevance_score": score,
                        "document": {"text": docs_copy[idx]}
                    }))
                    .collect();
                let resp = json!({
                    "model": state.meta.model_id,
                    "results": results
                });
                let _ = write_json_response(&mut stream, "200 OK", &resp).await;
            }
            _ => {
                let _ = write_json_response(&mut stream, "500 Internal Server Error", &json!({"error": {"message": "rerank failed"}})).await;
            }
        }
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
        if let Some(ref lname) = cfg.lora {
            if !state.meta.lora_adapters.iter().any(|a| a == lname) {
                let _ = write_json_response(
                    &mut stream,
                    "400 Bad Request",
                    &json!({"error": {"message": format!("unknown LoRA adapter `{}` (loaded: {:?})", lname, state.meta.lora_adapters)}}),
                )
                .await;
                return;
            }
        }
        if let Err(e) = compile_grammar_for_cfg(&cfg) {
            let _ = write_json_response(&mut stream, "400 Bad Request", &json!({"error": {"message": format!("grammar/schema error: {}", e)}})).await;
            return;
        }
        let tools_arr = req.get("tools").and_then(Value::as_array).cloned().unwrap_or_default();
        let has_tools = !tools_arr.is_empty();

        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let seq_no = state.total_reqs.fetch_add(1, Ordering::Relaxed) + 1;
        let req_id = format!("chatcmpl-tenon-{}-{}", created, seq_no);

        let mut mm_parts: Vec<MmPart> = Vec::new();
        let mut has_any_mm = false;
        let prompt = if is_chat {
            let mut msgs: Vec<(String, String)> = Vec::new();
            if let Some(arr) = req.get("messages").and_then(Value::as_array) {
                for item in arr {
                    let role = item
                        .get("role")
                        .and_then(Value::as_str)
                        .unwrap_or("user")
                        .to_string();
                    let content = if let Some(s) = item.get("content").and_then(Value::as_str) {
                        if !s.is_empty() && role != "assistant" {
                            mm_parts.push(MmPart::Text(s.to_string()));
                        }
                        s.to_string()
                    } else if let Some(parts) = item.get("content").and_then(Value::as_array) {
                        // OpenAI Multimodal Content Array: interleaves arbitrary `text`, `image_url`, & `input_audio`!
                        let mut text_acc = String::new();
                        for part in parts {
                            match part.get("type").and_then(Value::as_str) {
                                Some("text") => {
                                    if let Some(t) = part.get("text").and_then(Value::as_str) {
                                        if !text_acc.is_empty() {
                                            text_acc.push(' ');
                                        }
                                        text_acc.push_str(t);
                                        mm_parts.push(MmPart::Text(t.to_string()));
                                    }
                                }
                                Some("image_url") => {
                                    let url = part
                                        .get("image_url")
                                        .and_then(|u| u.as_str().or_else(|| u.get("url").and_then(Value::as_str)));
                                    if let Some(u) = url {
                                        match decode_image_url_to_rgb(u) {
                                            Ok(img) => {
                                                has_any_mm = true;
                                                mm_parts.push(MmPart::Image(img));
                                            }
                                            Err(e) => {
                                                let _ = write_json_response(&mut stream, "400 Bad Request", &json!({"error": {"message": format!("image_url decode failed: {}", e)}})).await;
                                                return;
                                            }
                                        }
                                    }
                                }
                                Some("input_audio") => {
                                    let b64 = part
                                        .get("input_audio")
                                        .and_then(|a| a.get("data").and_then(Value::as_str))
                                        .or_else(|| part.get("data").and_then(Value::as_str));
                                    if let Some(d) = b64 {
                                        match base64_decode(d).and_then(|raw| decode_wav_pcm16(&raw)) {
                                            Ok(aud) => {
                                                has_any_mm = true;
                                                mm_parts.push(MmPart::Audio(aud));
                                            }
                                            Err(e) => {
                                                let _ = write_json_response(&mut stream, "400 Bad Request", &json!({"error": {"message": format!("input_audio decode failed: {}", e)}})).await;
                                                return;
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        text_acc
                    } else {
                        String::new()
                    };
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
            } else if let Some(ref schema) = cfg.json_schema {
                let schema_note = format!(
                    "Respond ONLY with valid JSON adhering strictly to this JSON Schema: {}",
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
            if !has_any_mm || (!state.meta.has_mmproj && !state.meta.has_audio) {
                mm_parts.clear();
            }
            format_chat_template(&state.meta.arch, &msgs)
        } else {
            req.get("prompt")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        };
        let model_id = state.meta.model_id.clone();

        state.queued_reqs.fetch_add(1, Ordering::Relaxed);
        let (ev_tx, mut ev_rx) = tokio::sync::mpsc::unbounded_channel::<StreamEvent>();
        if state
            .engine_tx
            .send(EngineCmd::Generate {
                prompt,
                add_bos: true,
                cfg,
                mm_parts,
                tx: ev_tx,
            })
            .is_err()
        {
            state.queued_reqs.fetch_sub(1, Ordering::Relaxed);
            return;
        }

        if want_stream {
            let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
            if stream.write_all(hdr.as_bytes()).await.is_err() {
                state.queued_reqs.fetch_sub(1, Ordering::Relaxed);
                return;
            }
            while let Some(ev) = ev_rx.recv().await {
                match ev {
                    StreamEvent::Piece(piece) => {
                        let chunk = if is_chat {
                            json!({
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": null}]
                            })
                        } else {
                            json!({
                                "id": req_id,
                                "object": "text_completion",
                                "created": created,
                                "model": model_id,
                                "choices": [{"index": 0, "text": piece, "finish_reason": null}]
                            })
                        };
                        let line = format!("data: {}\n\n", chunk);
                        if stream.write_all(line.as_bytes()).await.is_err() {
                            break;
                        }
                        let _ = stream.flush().await;
                    }
                    StreamEvent::Done(res) => {
                        let done_chunk = if is_chat {
                            json!({
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
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
                                "id": req_id,
                                "object": "text_completion",
                                "created": created,
                                "model": model_id,
                                "choices": [{"index": 0, "text": "", "finish_reason": res.finish_reason}]
                            })
                        };
                        let tail = format!("data: {}\n\ndata: [DONE]\n\n", done_chunk);
                        let _ = stream.write_all(tail.as_bytes()).await;
                        let _ = stream.flush().await;
                        break;
                    }
                }
            }
            state.queued_reqs.fetch_sub(1, Ordering::Relaxed);
            return;
        }

        let mut final_res: Option<GenResult> = None;
        while let Some(ev) = ev_rx.recv().await {
            if let StreamEvent::Done(r) = ev {
                final_res = Some(r);
                break;
            }
        }
        state.queued_reqs.fetch_sub(1, Ordering::Relaxed);
        let Some(res) = final_res else {
            return;
        };

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

fn parse_lora_spec(spec: &str) -> (String, String, f64) {
    let (name_opt, rest) = match spec.split_once('=') {
        Some((n, r)) if !n.is_empty() => (Some(n.to_string()), r),
        _ => (None, spec),
    };
    let (path, scale) = if let Some((p, sc_s)) = rest.rsplit_once(':') {
        if let Ok(sc) = sc_s.parse::<f64>() {
            (p.to_string(), sc)
        } else {
            (rest.to_string(), 1.0)
        }
    } else {
        (rest.to_string(), 1.0)
    };
    let name = name_opt.unwrap_or_else(|| {
        Path::new(&path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("lora")
            .to_string()
    });
    (name, path, scale)
}

fn print_llm_usage() {
    println!(
        r#"tenon llm — Native Out-of-Core Streaming LLM CLI & Continuous-Batching OpenAI Server (v7)

USAGE:
  tenon llm <model.gguf> [OPTIONS]

OPTIONS:
  --prompt <text>           Raw completion prompt (runs batched prefill + streaming stdout)
  --chat <text>             Chat user message (auto-applies model architecture chat template)
  --system <text>           Optional system prompt (used with --chat)
  --image <path>            Local PNG/BMP/PPM image for multimodal VLM (repeatable for multi-image!)
  --anyres                  Force AnyRes high-res 5-tile encoding on VLM images (auto for hi-res!)
  --mmproj <mmproj.gguf>    CLIP/SigLIP/SmolVLM multimodal GPU vision encoder GGUF
  --audio-mmproj <path|mel:D> Whisper 80-bin Log-Mel audio encoder GGUF or built-in `mel:<dim>`
  --audio <file.wav>        Local 16-bit PCM WAV audio file for multimodal speech/audio input
  --mrope-grid <T,H,W>      Explicit 2D/3D MRoPE spatial grid for Qwen2.5-VL/video embeddings
  --lora <name=path[:sc]>   Load zero-copy runtime GGUF LoRA adapter (repeatable; HTTP `"lora":"name"`)
  --serve <host:port>       Launch OpenAI-compatible v8 Universal CB + Zero-Migration Spec + Multi-LoRA + /metrics server
  --prefill-chunk <N>       Chunked prefill token budget per decode tick in --serve mode (default: 128)
  --plan-only               Inspect GGUF & print Auto VRAM/NUMA/KV/TP/EP plan without loading weights
  --max-vram-gb <F>         Hard VRAM/RAM budget cap in GiB (0 = auto-detect hardware)
  --ctx <N>                 Total KV context capacity across all slots (default: 4096, up to 16384)
  --kv-slots <N>            Number of partitioned Radix Prefix KV slots (default: 2 for --serve, 1 CLI)
  --kv-quant <auto|q8|fp16> KV-cache precision (`q8` saves -47% VRAM; default: auto)
  --mla-latent <0|1>        Matrix-absorbed compressed latent MLA KV cache (default: 1)
  --tp <N>                  Multi-GPU Tensor Parallelism degree on CUDA (`TP=2..8`, default: 1)
  --ep <N>                  Multi-GPU MoE Expert Parallelism degree on CUDA (`EP=2..8`, 2D `TPxEP` supported!)
  --slots <N>               Override auto-planned VRAM slots / pinned layers
  --nocache                 Force uncached direct NVMe I/O (`F_NOCACHE` / `POSIX_FADV_DONTNEED`)
  --fast-dp4a               Enable opt-in CUDA 128-bit `int4` Q8_1 `__dp4a` fast path
  --draft <small.gguf>      Small draft GGUF for lossless speculative decoding (`verify_batch` / `verify_tree`)
  --speculative <off|ngram|draft|mtp|tree> Speculative decoding mode (adaptive fusion with CB in --serve!)
  --spec-k <N>              Speculative lookahead window K (default: 5)
  --json                    Enforce RFC 8259 JSON-object grammar-constrained sampling
  --json-schema <json>      Enforce full recursive JSON Schema grammar-constrained sampling
  --grammar <gbnf>          Enforce custom GBNF grammar rules (`root ::= ...`)
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
    let mut mmproj_path: Option<String> = None;
    let mut audio_mmproj_path: Option<String> = None;
    let mut image_paths: Vec<String> = Vec::new();
    let mut audio_paths: Vec<String> = Vec::new();
    let mut lora_specs: Vec<(String, String, f64)> = Vec::new();
    let mut prompt: Option<String> = None;
    let mut chat_msg: Option<String> = None;
    let mut system_msg: Option<String> = None;
    let mut serve_addr: Option<String> = None;
    let mut prefill_chunk_size = 128usize;
    let mut plan_only = false;
    let mut max_vram_gb = 0.0f64;
    let mut max_ctx = 4096i64;
    let mut kv_slots_opt: Option<usize> = None;
    let mut kv_quant = "auto".to_string();
    let mut mla_latent = 1i64;
    let mut tp_size = 1i64;
    let mut ep_size = 1i64;
    let mut slots_override: Option<i64> = None;
    let mut force_nocache = false;
    let mut fast_dp4a = false;
    let mut cfg = SamplingCfg::default();

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--prompt" if i + 1 < args.len() => { prompt = Some(args[i + 1].clone()); i += 2; }
            "--chat" if i + 1 < args.len() => { chat_msg = Some(args[i + 1].clone()); i += 2; }
            "--system" if i + 1 < args.len() => { system_msg = Some(args[i + 1].clone()); i += 2; }
            "--image" if i + 1 < args.len() => { image_paths.push(args[i + 1].clone()); i += 2; }
            "--audio" if i + 1 < args.len() => { audio_paths.push(args[i + 1].clone()); i += 2; }
            "--anyres" => { cfg.anyres = true; i += 1; }
            "--mmproj" if i + 1 < args.len() => { mmproj_path = Some(args[i + 1].clone()); i += 2; }
            "--audio-mmproj" if i + 1 < args.len() => { audio_mmproj_path = Some(args[i + 1].clone()); i += 2; }
            "--mrope-grid" if i + 1 < args.len() => {
                let parts: Vec<i64> = args[i + 1].split(',').filter_map(|p| p.trim().parse().ok()).collect();
                if parts.len() == 3 {
                    cfg.mrope_grid = Some((parts[0], parts[1], parts[2]));
                }
                i += 2;
            }
            "--lora" if i + 1 < args.len() => { lora_specs.push(parse_lora_spec(&args[i + 1])); i += 2; }
            "--serve" if i + 1 < args.len() => { serve_addr = Some(args[i + 1].clone()); i += 2; }
            "--prefill-chunk" if i + 1 < args.len() => {
                prefill_chunk_size = args[i + 1].parse::<usize>().unwrap_or(128).clamp(16, 4096);
                i += 2;
            }
            "--plan-only" => { plan_only = true; i += 1; }
            "--max-vram-gb" if i + 1 < args.len() => { max_vram_gb = args[i + 1].parse().unwrap_or(0.0); i += 2; }
            "--ctx" if i + 1 < args.len() => { max_ctx = args[i + 1].parse().unwrap_or(4096); i += 2; }
            "--kv-slots" if i + 1 < args.len() => { kv_slots_opt = args[i + 1].parse().ok(); i += 2; }
            "--kv-quant" if i + 1 < args.len() => { kv_quant = args[i + 1].clone(); i += 2; }
            "--mla-latent" if i + 1 < args.len() => { mla_latent = args[i + 1].parse().unwrap_or(1); i += 2; }
            "--tp" if i + 1 < args.len() => { tp_size = args[i + 1].parse().unwrap_or(1); i += 2; }
            "--ep" if i + 1 < args.len() => { ep_size = args[i + 1].parse().unwrap_or(1); i += 2; }
            "--slots" if i + 1 < args.len() => { slots_override = args[i + 1].parse().ok(); i += 2; }
            "--nocache" => { force_nocache = true; i += 1; }
            "--fast-dp4a" => { fast_dp4a = true; i += 1; }
            "--draft" if i + 1 < args.len() => {
                draft_path = Some(args[i + 1].clone());
                if cfg.spec_mode == SpecMode::Off {
                    cfg.spec_mode = SpecMode::Draft;
                }
                i += 2;
            }
            "--speculative" if i + 1 < args.len() => {
                cfg.spec_mode = match args[i + 1].as_str() {
                    "ngram" | "lookup" => SpecMode::Ngram,
                    "draft" => SpecMode::Draft,
                    "mtp" => SpecMode::Mtp,
                    "tree" => SpecMode::Tree,
                    _ => SpecMode::Off,
                };
                i += 2;
            }
            "--spec-k" if i + 1 < args.len() => { cfg.spec_k = args[i + 1].parse().unwrap_or(5); i += 2; }
            "--json" => { cfg.json_constrained = true; i += 1; }
            "--json-schema" if i + 1 < args.len() => {
                match serde_json::from_str::<Value>(&args[i + 1]) {
                    Ok(v) => { cfg.json_schema = Some(v); cfg.json_constrained = true; }
                    Err(e) => { eprintln!("invalid --json-schema: {}", e); return ExitCode::from(2); }
                }
                i += 2;
            }
            "--grammar" if i + 1 < args.len() => { cfg.gbnf_grammar = Some(args[i + 1].clone()); i += 2; }
            "--temp" if i + 1 < args.len() => { cfg.temp = args[i + 1].parse().unwrap_or(0.7); i += 2; }
            "--top-k" if i + 1 < args.len() => { cfg.top_k = args[i + 1].parse().unwrap_or(40); i += 2; }
            "--top-p" if i + 1 < args.len() => { cfg.top_p = args[i + 1].parse().unwrap_or(0.9); i += 2; }
            "--min-p" if i + 1 < args.len() => { cfg.min_p = args[i + 1].parse().unwrap_or(0.05); i += 2; }
            "--repeat-penalty" if i + 1 < args.len() => { cfg.repeat_penalty = args[i + 1].parse().unwrap_or(1.1); i += 2; }
            "--max-tokens" if i + 1 < args.len() => { cfg.max_tokens = args[i + 1].parse().unwrap_or(256); i += 2; }
            "--seed" if i + 1 < args.len() => { cfg.seed = args[i + 1].parse().unwrap_or(42); i += 2; }
            s if !s.starts_with('-') && model_path.is_none() => { model_path = Some(s.to_string()); i += 1; }
            other => { eprintln!("unknown option `{}`", other); return ExitCode::from(2); }
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
        if ep_size > 1 {
            env::set_var("TENON_EP_SIZE", ep_size.to_string());
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
        "[tenon llm] backend={} | disk={:.2} GiB | layers={} experts={} d={} | plan: slots={} nocache={} kv={} kv_slots={} tp={} ep={} prefill_chunk={} est_vram={:.2} GiB",
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
        ep_size,
        prefill_chunk_size,
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
            env::set_var("TENON_EP_SIZE", "1");
        }
        match api.load(dp, -1, false, true) {
            Ok(dh) => {
                eprintln!("[tenon llm] loaded speculative draft model `{}` (mode={:?}, K={})", dp, cfg.spec_mode, cfg.spec_k);
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
    let mmproj_handle = if let Some(ref mp) = mmproj_path {
        match api.mmproj_load(mp) {
            Ok(mh) => {
                eprintln!(
                    "[tenon llm] loaded mmproj vision encoder `{}` ({}x{} patch={} -> {} vision tokens x {}d, AnyRes ready)",
                    mp,
                    api.mmproj_info(mh, 0),
                    api.mmproj_info(mh, 0),
                    api.mmproj_info(mh, 1),
                    api.mmproj_info(mh, 6),
                    api.mmproj_info(mh, 5),
                );
                mh
            }
            Err(e) => {
                eprintln!("error: {}", e);
                api.free(handle);
                return ExitCode::from(1);
            }
        }
    } else {
        0
    };

    let mut lora_map: HashMap<String, (i64, f64)> = HashMap::new();
    let mut lora_adapters: Vec<String> = Vec::new();
    for (lname, lpath, lscale) in &lora_specs {
        match api.lora_load(handle, lpath, *lscale) {
            Ok(lid) => {
                let rank = api.lora_info(handle, lid, 1);
                let nt = api.lora_info(handle, lid, 2);
                eprintln!(
                    "[tenon llm] loaded zero-copy GGUF LoRA `{}` (id={}, rank={}, tensors={}, scale={:.2}) from `{}`",
                    lname, lid, rank, nt, lscale, lpath
                );
                lora_map.insert(lname.clone(), (lid, *lscale));
                lora_adapters.push(lname.clone());
            }
            Err(e) => {
                eprintln!("error: {}", e);
                api.free(handle);
                return ExitCode::from(1);
            }
        }
    }
    if serve_addr.is_none() && !lora_adapters.is_empty() && cfg.lora.is_none() {
        cfg.lora = Some(lora_adapters[0].clone());
    }

    let arch = api.arch(handle);
    let d_model = api.info(handle, 1).max(1) as usize;
    let n_experts = api.info(handle, 5);
    let bos_id = api.info(handle, 11);
    let eos_id = api.info(handle, 12);
    let vocab_size = api.info(handle, 7).max(1) as usize;
    let core_vram = api.info(handle, 8) as f64 / gib;
    let cache_vram = api.info(handle, 9) as f64 / gib;
    let (actual_mla_lat, actual_slots, actual_slot_ctx, actual_bpe_merges, actual_tp, actual_ep, hybrid_2d, nccl_active, has_mtp, is_mla) =
        if api.backend == "cuda" {
            let sctx = api.info(handle, 28);
            (
                api.info(handle, 26) != 0,
                api.info(handle, 27).max(1) as usize,
                if sctx > 0 { sctx } else { max_ctx },
                api.info(handle, 29),
                api.info(handle, 31).max(1),
                api.info(handle, 34).max(1),
                api.info(handle, 38) != 0,
                api.info(handle, 33) != 0,
                api.info(handle, 30),
                api.info(handle, 24) != 0,
            )
        } else {
            let sctx = api.info(handle, 26);
            (
                api.info(handle, 24) != 0,
                api.info(handle, 25).max(1) as usize,
                if sctx > 0 { sctx } else { max_ctx },
                api.info(handle, 28),
                1,
                1,
                false,
                false,
                api.info(handle, 27),
                api.info(handle, 23) != 0,
            )
        };
    let pa_blocks = api.paged_kv_info(handle, 1);
    let audio_handle = if let Some(ref ap) = audio_mmproj_path {
        match api.audio_load(ap) {
            Ok(ah) => {
                eprintln!(
                    "[tenon llm] loaded Whisper 80-bin Log-Mel audio encoder `{}` (n_mels={}, sr={}Hz, proj_dim={})",
                    ap,
                    api.audio_info(ah, 0),
                    api.audio_info(ah, 1),
                    api.audio_info(ah, 5),
                );
                ah
            }
            Err(e) => {
                eprintln!("error: {}", e);
                api.free(handle);
                return ExitCode::from(1);
            }
        }
    } else if !audio_paths.is_empty() {
        let auto_spec = format!("mel:{}", d_model);
        api.audio_load(&auto_spec).unwrap_or(0)
    } else {
        0
    };
    let is_resident_moe = n_experts > 0 && (chosen_slots <= 0 || chosen_slots >= n_experts || hybrid_2d);
    let supports_cb = pa_blocks >= 4
        && (!is_mla || actual_mla_lat)
        && (n_experts == 0 || is_resident_moe);

    let model_id = Path::new(&model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("tenon-gguf")
        .to_string();
    eprintln!(
        "[tenon llm] loaded `{}` (arch={}, ctx={} [{}x{}], kv={}, mla_latent={}, bpe_merges={}, tp={}, ep={}, hybrid_2d={}, nccl={}, mtp={}, pa_blks={}, cb={}, loras={}, core_vram={:.2} GiB, cache_vram={:.2} GiB) in {:.2}s",
        model_id,
        arch,
        max_ctx,
        actual_slots,
        actual_slot_ctx,
        if plan.kv_q8 { "Q8_0" } else { "FP16" },
        actual_mla_lat,
        actual_bpe_merges,
        actual_tp,
        actual_ep,
        hybrid_2d,
        nccl_active,
        has_mtp,
        pa_blocks,
        supports_cb,
        lora_adapters.len(),
        core_vram,
        cache_vram,
        t_load.elapsed().as_secs_f64()
    );

    let meta = ModelMeta {
        backend: api.backend,
        model_id,
        arch,
        d_model,
        max_ctx,
        slot_ctx: actual_slot_ctx,
        n_kv_slots: actual_slots,
        kv_q8: plan.kv_q8,
        mla_latent: actual_mla_lat,
        n_bpe_merges: actual_bpe_merges,
        tp_size: actual_tp,
        ep_size: actual_ep,
        hybrid_2d,
        nccl_active,
        has_draft: draft_handle > 0,
        has_mtp,
        has_mmproj: mmproj_handle > 0,
        mmproj_n_tokens: if mmproj_handle > 0 { api.mmproj_info(mmproj_handle, 6).max(0) as usize } else { 0 },
        has_audio: audio_handle > 0,
        audio_proj_dim: if audio_handle > 0 { api.audio_info(audio_handle, 5).max(0) as usize } else { 0 },
        supports_cb,
        prefill_chunk_size,
        lora_adapters,
    };

    let mut loaded = LoadedModel {
        api,
        handle,
        draft_handle,
        mmproj_handle,
        audio_handle,
        meta: meta.clone(),
        bos_id,
        eos_id,
        vocab_size,
        radix_cache: RadixKvCache::new(actual_slots),
        cb_radix: RadixKvCache::new(16),
        vocab_pieces: Vec::new(),
        logits_scratch: Vec::new(),
        lora_map,
        active_lora_id: 0,
        prefill_chunk_size,
        vision_cache: HashMap::new(),
        vision_cache_order: Vec::new(),
        vision_cache_hits: 0,
    };

    if let Some(addr) = serve_addr {
        let listener = match TcpListener::bind(&addr).await {
            Ok(l) => l,
            Err(e) => {
                eprintln!("error binding {}: {}", addr, e);
                return ExitCode::from(1);
            }
        };
        let (engine_tx, engine_rx) = std_mpsc::channel::<EngineCmd>();
        let state = Arc::new(ServerState {
            meta: meta.clone(),
            engine_tx,
            queued_reqs: AtomicUsize::new(0),
            active_batch: AtomicUsize::new(0),
            cb_batches_run: AtomicU64::new(0),
            cb_max_batch: AtomicUsize::new(0),
            total_reqs: AtomicU64::new(0),
            radix_hits: AtomicU64::new(0),
            radix_saved_toks: AtomicU64::new(0),
            chunked_prefill_chunks: AtomicU64::new(0),
            adaptive_spec_steps: AtomicU64::new(0),
            adaptive_spec_proposed: AtomicU64::new(0),
            adaptive_spec_accepted: AtomicU64::new(0),
            spec_zero_migrations: AtomicU64::new(0),
            vision_cache_hits: AtomicU64::new(0),
            vision_cache_entries: AtomicUsize::new(0),
            lora_swaps: AtomicU64::new(0),
        });
        let st_eng = Arc::clone(&state);
        std::thread::Builder::new()
            .name("tenon-cb-engine".into())
            .spawn(move || cb_engine_loop(loaded, engine_rx, st_eng))
            .expect("spawn continuous batching engine thread");

        eprintln!(
            "[tenon llm] OpenAI-compatible v8 Server listening on http://{} (PagedKV {} blks, UniversalCB={}, ChunkedPrefill={}, ZeroMigSpec={:?}, MultiLoRA={:?}, mmproj={}, audio={})",
            addr, pa_blocks, meta.supports_cb, meta.prefill_chunk_size, cfg.spec_mode, meta.lora_adapters, meta.has_mmproj, meta.has_audio
        );
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

    if !image_paths.is_empty() || !audio_paths.is_empty() {
        if !image_paths.is_empty() && loaded.mmproj_handle <= 0 {
            eprintln!("error: --image requires --mmproj <mmproj.gguf>");
            return ExitCode::from(2);
        }
        let mut parts: Vec<MmPart> = Vec::new();
        for (idx, img_p) in image_paths.iter().enumerate() {
            let img = match decode_image_url_to_rgb(img_p) {
                Ok(im) => im,
                Err(e) => {
                    eprintln!("error decoding image `{}`: {}", img_p, e);
                    return ExitCode::from(1);
                }
            };
            if image_paths.len() > 1 {
                parts.push(MmPart::Text(format!("Image {}:", idx + 1)));
            }
            parts.push(MmPart::Image(img));
        }
        for aud_p in &audio_paths {
            let raw = match fs::read(aud_p) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("error reading audio `{}`: {}", aud_p, e);
                    return ExitCode::from(1);
                }
            };
            let aud = match decode_wav_pcm16(&raw) {
                Ok(a) => a,
                Err(e) => {
                    eprintln!("error decoding WAV `{}`: {}", aud_p, e);
                    return ExitCode::from(1);
                }
            };
            parts.push(MmPart::Audio(aud));
        }
        let q = chat_msg.or(prompt).unwrap_or_else(|| "Describe this input.".to_string());
        parts.push(MmPart::Text(q));
        let res = run_multimodal_generation(&mut loaded, &parts, &cfg, |piece| {
            print!("{}", piece);
            let _ = io::stdout().flush();
            true
        });
        println!();
        eprintln!(
            "[tenon llm] multimodal prefill ({} imgs, {} audio, vis_cache_hits={}): {} toks in {:.1} ms | decode: {} toks in {:.1} ms ({:.2} tok/s)",
            image_paths.len(),
            audio_paths.len(),
            loaded.vision_cache_hits,
            res.prompt_tokens,
            res.prefill_ms,
            res.completion_tokens,
            res.decode_ms,
            if res.decode_ms > 0.0 { res.completion_tokens as f64 * 1000.0 / res.decode_ms } else { 0.0 }
        );
        return ExitCode::SUCCESS;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn accepts(g: &Arc<GbnfGrammar>, s: &str) -> bool {
        let mut st = GrammarState::new(Arc::clone(g));
        for &b in s.as_bytes() {
            if !st.step_byte(b) {
                return false;
            }
        }
        st.is_accepting()
    }

    #[test]
    fn test_recursive_json_schema_compiler() {
        let schema = json!({
            "$defs": {
                "Tag": { "type": "string", "enum": ["gpu", "metal", "cuda"] }
            },
            "type": "object",
            "required": ["name", "version", "tags", "meta"],
            "properties": {
                "name": { "type": "string", "pattern": "^[A-Z][a-z]+-[0-9]+$" },
                "version": { "type": "integer" },
                "tags": { "type": "array", "items": { "$ref": "#/$defs/Tag" }, "minItems": 1, "maxItems": 3 },
                "meta": {
                    "type": "object",
                    "required": ["ok"],
                    "properties": {
                        "ok": { "type": "boolean" },
                        "note": { "anyOf": [{"type": "string"}, {"type": "null"}] }
                    }
                }
            }
        });
        let g = Arc::new(GbnfGrammar::from_json_schema(&schema).expect("compile schema"));
        assert!(accepts(&g, r#"{"name":"Tenon-6","version":6,"tags":["metal","cuda"],"meta":{"ok":true,"note":null}}"#));
        assert!(accepts(&g, r#"{"name":"Tenon-6","version":6,"tags":["gpu"],"meta":{"ok":false}}"#));
        // Invalid regex pattern (lowercase first char)
        assert!(!accepts(&g, r#"{"name":"tenon-6","version":6,"tags":["gpu"],"meta":{"ok":true}}"#));
        // Invalid float for integer field
        assert!(!accepts(&g, r#"{"name":"Tenon-6","version":6.5,"tags":["gpu"],"meta":{"ok":true}}"#));
        // Invalid enum value
        assert!(!accepts(&g, r#"{"name":"Tenon-6","version":6,"tags":["rocm"],"meta":{"ok":true}}"#));
        // Violates minItems=1
        assert!(!accepts(&g, r#"{"name":"Tenon-6","version":6,"tags":[],"meta":{"ok":true}}"#));
    }

    #[test]
    fn test_custom_gbnf_grammar_parser() {
        let gbnf = r#"
            # Simple SQL-like select grammar
            root ::= "SELECT " cols " FROM " ident ";"
            cols ::= ident (", " ident)*
            ident ::= [a-z_][a-z0-9_]*
        "#;
        let g = Arc::new(GbnfGrammar::parse_gbnf(gbnf).expect("parse gbnf"));
        assert!(accepts(&g, "SELECT id, name_2 FROM users;"));
        assert!(accepts(&g, "SELECT col FROM t;"));
        assert!(!accepts(&g, "DROP TABLE users;"));
        assert!(!accepts(&g, "SELECT 123 FROM t;"));
    }

    #[test]
    fn test_ppm_and_bmp_and_base64_image_decoders() {
        // 2x1 red+blue PPM P6 via base64 data URL (`\xFF\x00\x00\x00\x00\xFF` -> 17 bytes)
        let ppm = b"P6\n2 1\n255\n\xFF\x00\x00\x00\x00\xFF";
        let b64 = "UDYKMiAxCjI1NQr/AAAAAP8=";
        assert_eq!(base64_decode(b64).unwrap(), ppm);
        let img = decode_image_url_to_rgb(&format!("data:image/x-portable-pixmap;base64,{}", b64)).unwrap();
        assert_eq!((img.width, img.height), (2, 1));
        assert_eq!(img.rgb, vec![255, 0, 0, 0, 0, 255]);
    }

    #[test]
    fn test_v7_lora_spec_and_sampling_json() {
        assert_eq!(
            parse_lora_spec("coder=/tmp/code_lora.gguf:1.5"),
            ("coder".to_string(), "/tmp/code_lora.gguf".to_string(), 1.5)
        );
        assert_eq!(
            parse_lora_spec("/tmp/my_adapter.gguf"),
            ("my_adapter".to_string(), "/tmp/my_adapter.gguf".to_string(), 1.0)
        );
        let req = json!({
            "temperature": 0.0,
            "max_tokens": 32,
            "lora": "coder",
            "anyres": true
        });
        let cfg = parse_sampling_from_json(&req, &SamplingCfg::default());
        assert_eq!(cfg.lora.as_deref(), Some("coder"));
        assert!(cfg.anyres);
        assert_eq!(cfg.max_tokens, 32);
    }
}
