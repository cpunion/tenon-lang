//! Minimal Tenon Core Byte & OS/Network Syscall Primitives (19 primitives total).
//!
//! By design, this module contains ZERO high-level stdlib logic:
//! - NO JSON parsing or serialization (written in pure Tenon in `std/json.tn`)
//! - NO string utilities like trim/index_of/replace/int_to_text (written in `std/text.tn`)
//! - NO `.env` parsing (written in `std/env.tn`)
//! - NO file edit/line-numbering/workspace-guard logic (written in `std/tools.tn`)
//! - NO OpenAI protocol or agent ReAct loop logic (written in `std/openai.tn` & `examples/agent/cli_agent.tn`)

use crate::perceus::{make_text, make_text_bytes, read_text, read_text_bytes};
use futures_util::StreamExt;
use std::collections::BTreeMap;
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
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(r) => r,
            Err(e) => {
                *err_bg.lock().unwrap() = Some(format!("runtime error: {}", e));
                return;
            }
        };
        rt.block_on(async move {
            let client = match reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()
            {
                Ok(c) => c,
                Err(e) => {
                    *err_bg.lock().unwrap() = Some(format!("http client error: {}", e));
                    return;
                }
            };
            let http_method = reqwest::Method::from_bytes(method_s.as_bytes())
                .unwrap_or(reqwest::Method::POST);
            let mut req = client.request(http_method, &url_s);
            for line in headers_s.lines() {
                if let Some((k, v)) = line.split_once(':') {
                    let key = k.trim();
                    let val = v.trim();
                    if !key.is_empty() {
                        req = req.header(key, val);
                    }
                }
            }
            if !body_bytes.is_empty() {
                req = req.body(body_bytes);
            }
            let resp = match req.send().await {
                Ok(r) => r,
                Err(e) => {
                    *err_bg.lock().unwrap() = Some(format!("http request failed: {}", e));
                    return;
                }
            };
            let status = resp.status();
            if !status.is_success() {
                let err_body = resp.text().await.unwrap_or_default();
                *err_bg.lock().unwrap() = Some(format!("HTTP {}: {}", status.as_u16(), err_body));
                return;
            }
            let mut byte_stream = resp.bytes_stream();
            let mut buf = Vec::<u8>::new();
            while let Some(chunk_res) = byte_stream.next().await {
                let chunk = match chunk_res {
                    Ok(c) => c,
                    Err(e) => {
                        *err_bg.lock().unwrap() = Some(format!("stream read error: {}", e));
                        return;
                    }
                };
                buf.extend_from_slice(&chunk);
                while let Some(nl_pos) = buf.iter().position(|&b| b == b'\n') {
                    let mut line_bytes = buf.drain(..=nl_pos).collect::<Vec<u8>>();
                    if line_bytes.ends_with(b"\n") {
                        line_bytes.pop();
                    }
                    if line_bytes.ends_with(b"\r") {
                        line_bytes.pop();
                    }
                    let line = String::from_utf8_lossy(&line_bytes).into_owned();
                    if tx.send(line).is_err() {
                        return;
                    }
                }
            }
            if !buf.is_empty() {
                if buf.ends_with(b"\r") {
                    buf.pop();
                }
                let line = String::from_utf8_lossy(&buf).into_owned();
                let _ = tx.send(line);
            }
        });
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
