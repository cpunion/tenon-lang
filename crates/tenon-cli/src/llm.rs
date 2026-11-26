use std::env;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::sync::{Arc, Mutex};
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
type FnPrefill = unsafe extern "C" fn(i64, i64, i64, i64) -> i64;
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
    prefill: FnPrefill,
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

#[derive(Debug, Clone)]
struct SamplingCfg {
    temp: f64,
    top_k: i64,
    top_p: f64,
    min_p: f64,
    repeat_penalty: f64,
    max_tokens: usize,
    seed: i64,
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
    let lib_path = PathBuf::from(format!("/tmp/tenon_llm_{}_v4_{}.{}", backend, mtime, ext));

    if !lib_path.exists() {
        eprintln!(
            "[tenon llm] compiling {} runtime shim ({}) -> {}",
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
        prefill: load_symbol(dl_handle, &sym("prefill"))?,
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

    fn prefill(&self, h: i64, ids: &[i64], start_pos: i64) -> i64 {
        unsafe { (self.prefill)(h, ids.as_ptr() as i64, ids.len() as i64, start_pos) }
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

struct LoadedModel {
    api: LlmApi,
    handle: i64,
    model_id: String,
    arch: String,
    bos_id: i64,
    eos_id: i64,
    max_ctx: i64,
    kv_q8: bool,
}

impl Drop for LoadedModel {
    fn drop(&mut self) {
        if self.handle > 0 {
            self.api.free(self.handle);
            self.handle = 0;
        }
    }
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

struct GenResult {
    text: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    prefill_ms: f64,
    decode_ms: f64,
    finish_reason: &'static str,
}

fn run_generation<F>(
    m: &LoadedModel,
    prompt: &str,
    add_bos: bool,
    cfg: &SamplingCfg,
    mut on_piece: F,
) -> GenResult
where
    F: FnMut(&str) -> bool,
{
    let max_prompt = (m.max_ctx as usize).saturating_sub(cfg.max_tokens.min(512) + 8).max(16);
    let mut ids = Vec::new();
    if add_bos && m.bos_id >= 0 {
        ids.push(m.bos_id);
    }
    ids.extend(m.api.tokenize(m.handle, prompt, max_prompt));
    if ids.is_empty() {
        ids.push(m.bos_id.max(1));
    }
    let prompt_tokens = ids.len();

    let t_pf = Instant::now();
    let _argmax = m.api.prefill(m.handle, &ids, 0);
    let prefill_ms = t_pf.elapsed().as_secs_f64() * 1000.0;
    m.api.reset(m.handle);

    let mut recent: Vec<i64> = ids.clone();
    let mut next_tok = m.api.sample_next(
        m.handle,
        -1,
        (prompt_tokens - 1) as i64,
        cfg,
        &recent[recent.len().saturating_sub(64)..],
        cfg.seed,
    );

    let mut out_text = String::new();
    let mut completion_tokens = 0usize;
    let mut finish_reason = "length";
    let t_dec = Instant::now();

    for step_idx in 0..cfg.max_tokens {
        let piece = m.api.token_str(m.handle, next_tok);
        if is_stop_token(next_tok, m.eos_id, &piece) {
            finish_reason = "stop";
            break;
        }
        out_text.push_str(&piece);
        completion_tokens += 1;
        if !on_piece(&piece) {
            finish_reason = "stop";
            break;
        }
        let pos = (prompt_tokens + step_idx) as i64;
        if pos + 1 >= m.max_ctx {
            finish_reason = "length";
            break;
        }
        recent.push(next_tok);
        let rstart = recent.len().saturating_sub(64);
        next_tok = m.api.sample_next(
            m.handle,
            next_tok,
            pos,
            cfg,
            &recent[rstart..],
            cfg.seed.wrapping_add((step_idx as i64 + 1).wrapping_mul(1000003)),
        );
    }
    let decode_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
    GenResult {
        text: out_text,
        prompt_tokens,
        completion_tokens,
        prefill_ms,
        decode_ms,
        finish_reason,
    }
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
    }
}

async fn handle_http_client(
    mut stream: TcpStream,
    model: Arc<Mutex<LoadedModel>>,
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
        let info = {
            let m = model.lock().unwrap();
            json!({
                "status": "ok",
                "backend": m.api.backend,
                "model": m.model_id,
                "arch": m.arch,
                "max_ctx": m.max_ctx,
                "kv_quant": if m.kv_q8 { "q8_0" } else { "fp16" }
            })
        };
        let _ = write_json_response(&mut stream, "200 OK", &info).await;
        return;
    }

    if method == "GET" && clean_path == "/v1/models" {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let resp = {
            let m = model.lock().unwrap();
            json!({
                "object": "list",
                "data": [{
                    "id": m.model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": "tenon-lang"
                }]
            })
        };
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
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let req_id = format!("chatcmpl-tenon-{}", created);

        let (prompt, model_id) = {
            let m = model.lock().unwrap();
            let p = if is_chat {
                let mut msgs = Vec::new();
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
                format_chat_template(&m.arch, &msgs)
            } else {
                req.get("prompt")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string()
            };
            (p, m.model_id.clone())
        };

        if want_stream {
            let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
            if stream.write_all(hdr.as_bytes()).await.is_err() {
                return;
            }
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
            let model_bg = Arc::clone(&model);
            let req_id_bg = req_id.clone();
            let model_id_bg = model_id.clone();
            let jh = tokio::task::spawn_blocking(move || {
                let m = model_bg.lock().unwrap();
                let res = run_generation(&m, &prompt, true, &cfg, |piece| {
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
                            "total_tokens": res.prompt_tokens + res.completion_tokens
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
            return;
        }

        let model_bg = Arc::clone(&model);
        let Ok(res) = tokio::task::spawn_blocking(move || {
            let m = model_bg.lock().unwrap();
            run_generation(&m, &prompt, true, &cfg, |_| true)
        })
        .await
        else {
            return;
        };

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
            json!({
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": res.text},
                    "finish_reason": res.finish_reason
                }],
                "usage": {
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "total_tokens": res.prompt_tokens + res.completion_tokens
                },
                "timings": {
                    "prefill_ms": res.prefill_ms,
                    "prefill_tok_per_sec": pf_tps,
                    "decode_ms": res.decode_ms,
                    "decode_tok_per_sec": dec_tps
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
                    "total_tokens": res.prompt_tokens + res.completion_tokens
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
        r#"tenon llm — Native Out-of-Core Streaming LLM CLI & OpenAI-Compatible SSE Server

USAGE:
  tenon llm <model.gguf> [OPTIONS]

OPTIONS:
  --prompt <text>           Raw completion prompt (runs batched prefill + streaming stdout)
  --chat <text>             Chat user message (auto-applies model architecture chat template)
  --system <text>           Optional system prompt (used with --chat)
  --serve <host:port>       Launch OpenAI-compatible HTTP + SSE server (e.g. 127.0.0.1:8080)
  --plan-only               Inspect GGUF & print Auto VRAM/NUMA/KV plan without loading weights
  --max-vram-gb <F>         Hard VRAM/RAM budget cap in GiB (0 = auto-detect hardware)
  --ctx <N>                 Max KV context length (default: 4096, up to 16384)
  --kv-quant <auto|q8|fp16> KV-cache precision (`q8` saves -47% VRAM; default: auto)
  --slots <N>               Override auto-planned VRAM slots / pinned layers
  --nocache                 Force uncached direct NVMe I/O (`F_NOCACHE` / `POSIX_FADV_DONTNEED`)
  --fast-dp4a               Enable opt-in CUDA 128-bit `int4` Q8_1 `__dp4a` fast path
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
    let mut prompt: Option<String> = None;
    let mut chat_msg: Option<String> = None;
    let mut system_msg: Option<String> = None;
    let mut serve_addr: Option<String> = None;
    let mut plan_only = false;
    let mut max_vram_gb = 0.0f64;
    let mut max_ctx = 4096i64;
    let mut kv_quant = "auto".to_string();
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
            "--kv-quant" if i + 1 < args.len() => {
                kv_quant = args[i + 1].clone();
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
        "[tenon llm] backend={} | disk={:.2} GiB | layers={} experts={} d={} | plan: slots={} nocache={} kv={} prefill_B={} est_vram={:.2} GiB",
        api.backend,
        plan.disk_bytes as f64 / gib,
        plan.n_layers,
        plan.n_experts,
        plan.d_model,
        chosen_slots,
        if chosen_nocache { 1 } else { 0 },
        if plan.kv_q8 { "Q8_0" } else { "FP16" },
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
    let arch = api.arch(handle);
    let bos_id = api.info(handle, 11);
    let eos_id = api.info(handle, 12);
    let core_vram = api.info(handle, 8) as f64 / gib;
    let cache_vram = api.info(handle, 9) as f64 / gib;
    let model_id = Path::new(&model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("tenon-gguf")
        .to_string();
    eprintln!(
        "[tenon llm] loaded `{}` (arch={}, ctx={}, kv={}, core_vram={:.2} GiB, cache_vram={:.2} GiB) in {:.2}s",
        model_id,
        arch,
        max_ctx,
        if plan.kv_q8 { "Q8_0" } else { "FP16" },
        core_vram,
        cache_vram,
        t_load.elapsed().as_secs_f64()
    );

    let loaded = LoadedModel {
        api,
        handle,
        model_id,
        arch,
        bos_id,
        eos_id,
        max_ctx,
        kv_q8: plan.kv_q8,
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
            "[tenon llm] OpenAI-compatible SSE server listening on http://{} (/v1/chat/completions, /v1/completions, /v1/models, /health)",
            addr
        );
        let shared = Arc::new(Mutex::new(loaded));
        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let m_ref = Arc::clone(&shared);
                    let dcfg = cfg.clone();
                    tokio::spawn(async move {
                        handle_http_client(stream, m_ref, dcfg).await;
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
        format_chat_template(&loaded.arch, &msgs)
    } else {
        prompt.unwrap_or_else(|| "The capital of France is".to_string())
    };

    let res = run_generation(&loaded, &final_prompt, true, &cfg, |piece| {
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
        "[tenon llm] prefill: {} toks in {:.1} ms ({:.2} tok/s) | decode: {} toks in {:.1} ms ({:.2} tok/s)",
        res.prompt_tokens, res.prefill_ms, pf_tps, res.completion_tokens, res.decode_ms, dec_tps
    );
    ExitCode::SUCCESS
}
