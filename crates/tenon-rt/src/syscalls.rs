//! Minimal Tenon Core Byte, Buffer & OS/HTTP Syscall Primitives (24 primitives total).
//!
//! By design, this module contains ZERO high-level stdlib or vendor logic:
//! - NO JSON parsing/serialization/schema (written in pure Tenon in `std/encoding/json.tn` + `@derive`)
//! - NO string utilities like trim/find/replace/parse_int (written in `std/text/text.tn` & `std/text/builder.tn`)
//! - NO `.env` parsing (written in `std/encoding/dotenv.tn`)
//! - NO file edit/workspace-guard/tool logic (written in `std/os/{path,fs}.tn` & `examples/agent/lib/tools.tn`)
//! - NO OpenAI protocol or agent ReAct loop logic (written in `std/os/http.tn`, `examples/agent/lib/llm.tn` & `examples/agent/cli_agent.tn`)

use crate::perceus::{make_text, make_text_bytes, read_text, read_text_bytes};
use std::collections::BTreeMap;
use std::ffi::{CString, c_char, c_int, c_long, c_void};
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};

static LAST_STDIN_EOF: AtomicBool = AtomicBool::new(false);
static LAST_FILE_OK: AtomicBool = AtomicBool::new(true);
static LAST_EXEC_EXIT_CODE: AtomicI64 = AtomicI64::new(0);

struct SseStreamHandle {
    rx: mpsc::Receiver<String>,
    eof: bool,
    err: Arc<Mutex<Option<String>>>,
}

static NEXT_SSE_HANDLE: AtomicI64 = AtomicI64::new(1);
static NEXT_BUF_HANDLE: AtomicI64 = AtomicI64::new(1);

fn sse_table() -> &'static Mutex<BTreeMap<i64, SseStreamHandle>> {
    static TABLE: OnceLock<Mutex<BTreeMap<i64, SseStreamHandle>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn buf_table() -> &'static Mutex<BTreeMap<i64, Vec<u8>>> {
    static TABLE: OnceLock<Mutex<BTreeMap<i64, Vec<u8>>>> = OnceLock::new();
    TABLE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

// =========================================================================
// 1b. O(n) Byte Buffer Primitives (4 — used by std/text/builder & std/encoding/json)
// =========================================================================

#[unsafe(no_mangle)]
pub extern "C" fn tenon_buf_new() -> i64 {
    let id = NEXT_BUF_HANDLE.fetch_add(1, Ordering::SeqCst);
    buf_table().lock().unwrap().insert(id, Vec::with_capacity(64));
    id
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_buf_push_str(handle: i64, s: i64) -> i64 {
    let bytes = read_text_bytes(s);
    if let Some(buf) = buf_table().lock().unwrap().get_mut(&handle) {
        buf.extend_from_slice(bytes);
        buf.len() as i64
    } else {
        0
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_buf_push_byte(handle: i64, b: i64) -> i64 {
    if let Some(buf) = buf_table().lock().unwrap().get_mut(&handle) {
        buf.push((b & 0xFF) as u8);
        buf.len() as i64
    } else {
        0
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_buf_finish(handle: i64) -> i64 {
    let bytes = buf_table().lock().unwrap().remove(&handle).unwrap_or_default();
    make_text_bytes(&bytes)
}

// =========================================================================
// 1. Minimal Core Byte / Text Primitives (4)
// =========================================================================

#[unsafe(no_mangle)]
pub extern "C" fn tenon_byte_at(s: i64, idx: i64) -> i64 {
    if idx < 0 {
        return -1;
    }
    let bytes = read_text_bytes(s);
    let i = idx as usize;
    if i >= bytes.len() {
        return -1;
    }
    bytes[i] as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_str_slice(s: i64, start: i64, end: i64) -> i64 {
    let bytes = read_text_bytes(s);
    let len = bytes.len();
    let st = (start.max(0) as usize).min(len);
    let en = (end.max(0) as usize).min(len);
    if st >= en {
        return make_text_bytes(&[]);
    }
    make_text_bytes(&bytes[st..en])
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_byte_to_text(b: i64) -> i64 {
    let byte = [(b & 0xFF) as u8];
    make_text_bytes(&byte)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_str_concat(a: i64, b: i64) -> i64 {
    let ba = read_text_bytes(a);
    let bb = read_text_bytes(b);
    let mut out = Vec::with_capacity(ba.len() + bb.len());
    out.extend_from_slice(ba);
    out.extend_from_slice(bb);
    make_text_bytes(&out)
}

// =========================================================================
// 1c. Minimal IEEE-754 Float64 Primitives (6)
// =========================================================================

#[unsafe(no_mangle)]
pub extern "C" fn tenon_int_to_float(v: i64) -> i64 {
    (v as f64).to_bits() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_float_to_int(bits: i64) -> i64 {
    f64::from_bits(bits as u64) as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_float_to_text(bits: i64) -> i64 {
    let v = f64::from_bits(bits as u64);
    let clean = if v == 0.0 { 0.0 } else { v };
    make_text(&format!("{}", clean))
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_parse_float(s: i64) -> i64 {
    let txt = read_text(s);
    match txt.trim().parse::<f64>() {
        Ok(v) => v.to_bits() as i64,
        Err(_) => f64::NAN.to_bits() as i64,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_float_is_nan(bits: i64) -> i64 {
    if f64::from_bits(bits as u64).is_nan() { 1 } else { 0 }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_float_bits(bits: i64) -> i64 {
    let v = f64::from_bits(bits as u64);
    if v == 0.0 {
        0.0f64.to_bits() as i64
    } else {
        bits
    }
}

// =========================================================================
// 2. Minimal OS / I/O / Filesystem / Process Syscalls (10)
// =========================================================================

#[unsafe(no_mangle)]
pub extern "C" fn tenon_sys_write_stdout(s: i64) -> i64 {
    let bytes = read_text_bytes(s);
    let mut stdout = io::stdout().lock();
    let _ = stdout.write_all(bytes);
    let _ = stdout.flush();
    bytes.len() as i64
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_sys_read_line() -> i64 {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut line = String::new();
    match reader.read_line(&mut line) {
        Ok(0) => {
            LAST_STDIN_EOF.store(true, Ordering::SeqCst);
            make_text("")
        }
        Ok(_) => {
            LAST_STDIN_EOF.store(false, Ordering::SeqCst);
            while line.ends_with('\n') || line.ends_with('\r') {
                line.pop();
            }
            make_text(&line)
        }
        Err(_) => {
            LAST_STDIN_EOF.store(true, Ordering::SeqCst);
            make_text("")
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_sys_is_eof() -> i64 {
    if LAST_STDIN_EOF.load(Ordering::SeqCst) {
        1
    } else {
        0
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_sys_env(key: i64) -> i64 {
    let k = read_text(key);
    match std::env::var(&k) {
        Ok(v) => make_text(&v),
        Err(_) => make_text(""),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_sys_cwd() -> i64 {
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".into());
    make_text(&cwd)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_sys_read_file(path: i64) -> i64 {
    let p = read_text(path);
    match fs::read(&p) {
        Ok(bytes) => {
            LAST_FILE_OK.store(true, Ordering::SeqCst);
            make_text_bytes(&bytes)
        }
        Err(e) => {
            LAST_FILE_OK.store(false, Ordering::SeqCst);
            make_text(&e.to_string())
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_sys_file_ok() -> i64 {
    if LAST_FILE_OK.load(Ordering::SeqCst) {
        1
    } else {
        0
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_sys_write_file(path: i64, content: i64) -> i64 {
    let p = read_text(path);
    let path_ref = Path::new(&p);
    if let Some(parent) = path_ref.parent() {
        if !parent.as_os_str().is_empty() {
            let _ = fs::create_dir_all(parent);
        }
    }
    let bytes = read_text_bytes(content);
    match fs::write(path_ref, bytes) {
        Ok(()) => 1,
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_sys_exec(cmd: i64) -> i64 {
    let command = read_text(cmd);
    let output = Command::new("/bin/sh").arg("-c").arg(&command).output();
    match output {
        Ok(out) => {
            let code = out.status.code().unwrap_or(-1) as i64;
            LAST_EXEC_EXIT_CODE.store(code, Ordering::SeqCst);
            let mut combined = Vec::with_capacity(out.stdout.len() + out.stderr.len() + 1);
            combined.extend_from_slice(&out.stdout);
            if !out.stderr.is_empty() {
                if !combined.is_empty() && !combined.ends_with(b"\n") {
                    combined.push(b'\n');
                }
                combined.extend_from_slice(&out.stderr);
            }
            make_text_bytes(&combined)
        }
        Err(e) => {
            LAST_EXEC_EXIT_CODE.store(-1, Ordering::SeqCst);
            make_text(&format!("exec error: {}", e))
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_sys_exec_exit_code() -> i64 {
    LAST_EXEC_EXIT_CODE.load(Ordering::SeqCst)
}

// =========================================================================
// 3. Minimal Raw HTTPS Line-Stream Transport Syscalls (6)
//    Pure transport ONLY — zero JSON, zero OpenAI protocol logic!
// =========================================================================

/// Generic HTTP/SSE stream opener: accepts HTTP method, URL, newline-separated `Key: Value` headers,
/// and raw body bytes. Zero vendor/OpenAI specifics inside `tenon-rt`.
#[unsafe(no_mangle)]
pub extern "C" fn tenon_sys_http_open(method: i64, url: i64, headers: i64, body: i64) -> i64 {
    let method_s = read_text(method).trim().to_uppercase();
    let url_s = read_text(url);
    let headers_s = read_text(headers);
    let body_bytes = read_text_bytes(body).to_vec();
    spawn_http_line_stream(method_s, url_s, headers_s, body_bytes)
}

#[link(name = "curl")]
unsafe extern "C" {
    fn curl_easy_init() -> *mut c_void;
    fn curl_easy_setopt(curl: *mut c_void, option: c_int, ...) -> c_int;
    fn curl_easy_perform(curl: *mut c_void) -> c_int;
    fn curl_easy_getinfo(curl: *mut c_void, info: c_int, ...) -> c_int;
    fn curl_easy_strerror(code: c_int) -> *const c_char;
    fn curl_easy_cleanup(curl: *mut c_void);
    fn curl_slist_append(list: *mut c_void, string: *const c_char) -> *mut c_void;
    fn curl_slist_free_all(list: *mut c_void);
}

const CURLOPT_WRITEDATA: c_int = 10001;
const CURLOPT_URL: c_int = 10002;
const CURLOPT_WRITEFUNCTION: c_int = 20011;
const CURLOPT_TIMEOUT: c_int = 13;
const CURLOPT_POSTFIELDS: c_int = 10015;
const CURLOPT_HTTPHEADER: c_int = 10023;
const CURLOPT_CUSTOMREQUEST: c_int = 10036;
const CURLOPT_FOLLOWLOCATION: c_int = 52;
const CURLOPT_POSTFIELDSIZE: c_int = 60;
const CURLINFO_RESPONSE_CODE: c_int = 0x200002;

struct CurlStreamCtx {
    tx: mpsc::Sender<String>,
    buf: Vec<u8>,
    err_preview: Vec<u8>,
    closed: bool,
}

unsafe extern "C" fn curl_write_cb(
    ptr: *const c_char,
    size: usize,
    nmemb: usize,
    userdata: *mut c_void,
) -> usize {
    let total = size.saturating_mul(nmemb);
    if total == 0 || ptr.is_null() || userdata.is_null() {
        return total;
    }
    let ctx = unsafe { &mut *(userdata as *mut CurlStreamCtx) };
    let chunk = unsafe { std::slice::from_raw_parts(ptr as *const u8, total) };
    if ctx.err_preview.len() < 4096 {
        let take = (4096 - ctx.err_preview.len()).min(chunk.len());
        ctx.err_preview.extend_from_slice(&chunk[..take]);
    }
    ctx.buf.extend_from_slice(chunk);
    while let Some(nl_pos) = ctx.buf.iter().position(|&b| b == b'\n') {
        let mut line_bytes: Vec<u8> = ctx.buf.drain(..=nl_pos).collect();
        if line_bytes.ends_with(b"\n") {
            line_bytes.pop();
        }
        if line_bytes.ends_with(b"\r") {
            line_bytes.pop();
        }
        let line = String::from_utf8_lossy(&line_bytes).into_owned();
        if ctx.tx.send(line).is_err() {
            ctx.closed = true;
            return 0;
        }
    }
    total
}

fn spawn_http_line_stream(
    method_s: String,
    url_s: String,
    headers_s: String,
    body_bytes: Vec<u8>,
) -> i64 {
    let (tx, rx) = mpsc::channel::<String>();
    let err_slot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let err_bg = err_slot.clone();

    std::thread::spawn(move || {
        let Ok(c_url) = CString::new(url_s) else {
            *err_bg.lock().unwrap() = Some("invalid URL (interior NUL)".into());
            return;
        };
        let Ok(c_method) = CString::new(if method_s.is_empty() { "POST".to_string() } else { method_s }) else {
            *err_bg.lock().unwrap() = Some("invalid HTTP method".into());
            return;
        };
        unsafe {
            let curl = curl_easy_init();
            if curl.is_null() {
                *err_bg.lock().unwrap() = Some("curl_easy_init failed".into());
                return;
            }
            let mut slist: *mut c_void = std::ptr::null_mut();
            let mut header_cstrs = Vec::new();
            for line in headers_s.lines() {
                let trimmed = line.trim();
                if !trimmed.is_empty() && trimmed.contains(':') {
                    if let Ok(ch) = CString::new(trimmed) {
                        slist = curl_slist_append(slist, ch.as_ptr());
                        header_cstrs.push(ch);
                    }
                }
            }
            let mut ctx = Box::new(CurlStreamCtx {
                tx,
                buf: Vec::new(),
                err_preview: Vec::new(),
                closed: false,
            });
            let cb: unsafe extern "C" fn(*const c_char, usize, usize, *mut c_void) -> usize = curl_write_cb;
            curl_easy_setopt(curl, CURLOPT_URL, c_url.as_ptr());
            curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, c_method.as_ptr());
            if !slist.is_null() {
                curl_easy_setopt(curl, CURLOPT_HTTPHEADER, slist);
            }
            if !body_bytes.is_empty() {
                curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_bytes.as_ptr() as *const c_char);
                curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, body_bytes.len() as c_long);
            }
            curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300 as c_long);
            curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1 as c_long);
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, cb);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &mut *ctx as *mut CurlStreamCtx as *mut c_void);

            let res = curl_easy_perform(curl);
            let mut http_code: c_long = 0;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &mut http_code as *mut c_long);

            if !ctx.buf.is_empty() && !ctx.closed {
                if ctx.buf.ends_with(b"\r") {
                    ctx.buf.pop();
                }
                let line = String::from_utf8_lossy(&ctx.buf).into_owned();
                let _ = ctx.tx.send(line);
            }

            if res != 0 && !ctx.closed {
                let msg = std::ffi::CStr::from_ptr(curl_easy_strerror(res))
                    .to_string_lossy()
                    .into_owned();
                *err_bg.lock().unwrap() = Some(format!("http request failed: {}", msg));
            } else if http_code >= 400 {
                let preview = String::from_utf8_lossy(&ctx.err_preview).into_owned();
                *err_bg.lock().unwrap() = Some(format!("HTTP {}: {}", http_code, preview));
            }

            if !slist.is_null() {
                curl_slist_free_all(slist);
            }
            drop(header_cstrs);
            curl_easy_cleanup(curl);
        }
    });

    let id = NEXT_SSE_HANDLE.fetch_add(1, Ordering::SeqCst);
    sse_table().lock().unwrap().insert(
        id,
        SseStreamHandle {
            rx,
            eof: false,
            err: err_slot,
        },
    );
    id
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_sys_http_sse_open(url: i64, bearer: i64, body: i64) -> i64 {
    let url_s = read_text(url);
    let bearer_s = read_text(bearer);
    let body_bytes = read_text_bytes(body).to_vec();
    let mut headers = String::from("Content-Type: application/json\nAccept: text/event-stream\n");
    if !bearer_s.is_empty() {
        headers.push_str(&format!("Authorization: Bearer {}\n", bearer_s));
    }
    spawn_http_line_stream("POST".to_string(), url_s, headers, body_bytes)
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_sys_http_sse_next_line(handle: i64) -> i64 {
    let recv_res = {
        let mut tbl = sse_table().lock().unwrap();
        let Some(h) = tbl.get_mut(&handle) else {
            return make_text("");
        };
        if h.eof {
            return make_text("");
        }
        // Release mutex only if we want concurrent handles; here each handle is polled sequentially.
        // Using std::sync::mpsc::Receiver::recv while holding lock is safe because background thread
        // never acquires `sse_table()`.
        match h.rx.recv() {
            Ok(line) => Some(line),
            Err(_) => {
                h.eof = true;
                None
            }
        }
    };
    match recv_res {
        Some(line) => make_text(&line),
        None => make_text(""),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_sys_http_sse_eof(handle: i64) -> i64 {
    let tbl = sse_table().lock().unwrap();
    match tbl.get(&handle) {
        Some(h) if !h.eof => 0,
        _ => 1,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_sys_http_sse_error(handle: i64) -> i64 {
    let tbl = sse_table().lock().unwrap();
    if let Some(h) = tbl.get(&handle) {
        if let Some(e) = h.err.lock().unwrap().as_deref() {
            return make_text(e);
        }
    }
    make_text("")
}

#[unsafe(no_mangle)]
pub extern "C" fn tenon_sys_http_sse_close(handle: i64) -> i64 {
    sse_table().lock().unwrap().remove(&handle);
    0
}
