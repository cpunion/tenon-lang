use crate::sync_kernel::{MuxFrame, UniversalSyncKernel};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

/// Minimal HTTP/1.1 + Unified Multiplexed SSE Gateway (`GET /api/client/stream`)
/// + live Web UI demonstrating compiled `coalesce_window(33ms)` frame batching & Anti-Join overlay!
pub async fn serve_http(kernel: Arc<UniversalSyncKernel>, addr: &str) -> Result<String, String> {
    let listener = TcpListener::bind(addr).await.map_err(|e| e.to_string())?;
    let bound = listener.local_addr().map_err(|e| e.to_string())?.to_string();

    // Background worker loop automatically draining runnable sessions
    let wk = kernel.clone();
    tokio::spawn(async move {
        loop {
            let _ = wk.run_worker_once("main", "worker-auto-1", 40).await;
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    });

    tokio::spawn(async move {
        loop {
            if let Ok((stream, _)) = listener.accept().await {
                let k = kernel.clone();
                tokio::spawn(async move {
                    let _ = handle_conn(stream, k).await;
                });
            }
        }
    });

    Ok(bound)
}

async fn handle_conn(
    stream: TcpStream,
    kernel: Arc<UniversalSyncKernel>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut reader = BufReader::new(stream);
    let mut req_line = String::new();
    if reader.read_line(&mut req_line).await? == 0 {
        return Ok(());
    }
    let parts: Vec<&str> = req_line.split_whitespace().collect();
    if parts.len() < 2 {
        return Ok(());
    }
    let method = parts[0];
    let path = parts[1];

    let mut content_len = 0usize;
    loop {
        let mut h = String::new();
        reader.read_line(&mut h).await?;
        let trimmed = h.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_len = v.trim().parse().unwrap_or(0);
            }
        }
    }

    let mut body = vec![0u8; content_len];
    if content_len > 0 {
        reader.read_exact(&mut body).await?;
    }
    let body_str = String::from_utf8_lossy(&body).to_string();
    let mut sock = reader.into_inner();

    if method == "GET" && path == "/" {
        let html = render_web_ui(kernel.web_coalesce_ms());
        write_http(&mut sock, 200, "text/html; charset=utf-8", &html).await?;
        return Ok(());
    }

    if method == "GET" && path == "/api/manifest" {
        let json = serde_json::to_string_pretty(&kernel.manifest)?;
        write_http(&mut sock, 200, "application/json", &json).await?;
        return Ok(());
    }

    if method == "GET" && path == "/api/channels" {
        let stats = kernel.channel_stats.lock().unwrap().clone();
        let json = serde_json::to_string_pretty(&stats)?;
        write_http(&mut sock, 200, "application/json", &json).await?;
        return Ok(());
    }

    if method == "GET" && path.starts_with("/api/client/stream") {
        // Unified Multiplexed SSE: Subscribe BEFORE first authoritative read (dast-os-kernel docs/11 §6.1!)
        let mut rx = kernel.mux_tx.subscribe();
        let initial = kernel.build_snapshot_frame("main")?;
        let hello = MuxFrame::Hello {
            conn_id: format!("conn_{}", std::process::id()),
            coalesce_ms: kernel.web_coalesce_ms(),
        };

        sock.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\n\r\n",
        )
        .await?;
        write_sse(&mut sock, &hello).await?;
        write_sse(&mut sock, &initial).await?;

        let mut hb = tokio::time::interval(std::time::Duration::from_secs(15));
        loop {
            tokio::select! {
                msg = rx.recv() => {
                    match msg {
                        Ok(frame) => {
                            if write_sse(&mut sock, &frame).await.is_err() {
                                break;
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                            // Slow consumer recovery: send fresh authoritative snapshot!
                            if let Ok(snap) = kernel.build_snapshot_frame("main") {
                                if write_sse(&mut sock, &snap).await.is_err() { break; }
                            }
                        }
                        Err(_) => break,
                    }
                }
                _ = hb.tick() => {
                    let frame = MuxFrame::Heartbeat {
                        session_id: "main".into(),
                        stream_id: None,
                        high_water_seq: 0,
                    };
                    if write_sse(&mut sock, &frame).await.is_err() { break; }
                }
            }
        }
        return Ok(());
    }

    if method == "POST" && path == "/api/sessions/main/send" {
        let v: serde_json::Value = serde_json::from_str(&body_str).unwrap_or_default();
        let text = v["text"].as_str().unwrap_or("hello");
        let cid = v["client_msg_id"].as_str().unwrap_or("c1");
        let (id, started) = kernel.authority.send_message("main", cid, text)?;
        kernel.publish_snapshot("main", None);
        let resp = serde_json::json!({ "ok": true, "id": id, "started": started }).to_string();
        write_http(&mut sock, 200, "application/json", &resp).await?;
        return Ok(());
    }

    if method == "POST" && path == "/api/sessions/main/steer" {
        let v: serde_json::Value = serde_json::from_str(&body_str).unwrap_or_default();
        let text = v["text"].as_str().unwrap_or("be concise");
        let cid = v["client_msg_id"].as_str().unwrap_or("s1");
        kernel.authority.enqueue_steer("main", cid, text)?;
        kernel.publish_snapshot("main", None);
        write_http(&mut sock, 200, "application/json", r#"{"ok":true}"#).await?;
        return Ok(());
    }

    if method == "POST" && path == "/api/sessions/main/followup" {
        let v: serde_json::Value = serde_json::from_str(&body_str).unwrap_or_default();
        let text = v["text"].as_str().unwrap_or("next task");
        let cid = v["client_msg_id"].as_str().unwrap_or("f1");
        kernel.authority.enqueue_followup("main", cid, text)?;
        kernel.publish_snapshot("main", None);
        write_http(&mut sock, 200, "application/json", r#"{"ok":true}"#).await?;
        return Ok(());
    }

    if method == "POST" && path == "/api/sessions/main/interrupt" {
        let v: serde_json::Value = serde_json::from_str(&body_str).unwrap_or_default();
        let text = v["text"].as_str().unwrap_or("stop and switch topic");
        let cid = v["cmd_id"].as_str().unwrap_or("int1");
        match kernel.authority.interrupt("main", cid, text) {
            Ok(out) => {
                kernel.publish_snapshot("main", None);
                write_http(&mut sock, 200, "application/json", &serde_json::to_string(&out)?).await?;
            }
            Err(e) => {
                let err = serde_json::json!({ "ok": false, "error": e.to_string() }).to_string();
                write_http(&mut sock, 409, "application/json", &err).await?;
            }
        }
        return Ok(());
    }

    write_http(&mut sock, 404, "text/plain", "not found").await?;
    Ok(())
}

async fn write_http(
    sock: &mut TcpStream,
    status: u16,
    ctype: &str,
    body: &str,
) -> std::io::Result<()> {
    let header = format!(
        "HTTP/1.1 {} OK\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status,
        ctype,
        body.len()
    );
    sock.write_all(header.as_bytes()).await?;
    sock.write_all(body.as_bytes()).await?;
    Ok(())
}

async fn write_sse(sock: &mut TcpStream, frame: &MuxFrame) -> std::io::Result<()> {
    let json = serde_json::to_string(frame).unwrap_or_default();
    let msg = format!("data: {}\n\n", json);
    sock.write_all(msg.as_bytes()).await?;
    sock.flush().await
}

fn render_web_ui(coalesce_ms: u64) -> String {
    format!(
        r#"<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Tenon Distributed Prototype (SessionMsgList + 33ms Coalesce)</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 820px; margin: 24px auto; padding: 0 16px; background: #0f172a; color: #e2e8f0; }}
  .bar {{ display: flex; gap: 12px; font-size: 13px; background: #1e293b; padding: 10px 14px; border-radius: 8px; margin-bottom: 14px; }}
  .msg {{ padding: 10px 14px; margin: 8px 0; border-radius: 8px; }}
  .user {{ background: #1d4ed8; }}
  .assistant {{ background: #1e293b; border: 1px solid #334155; }}
  .live {{ background: #172554; border: 1px dashed #38bdf8; }}
  .controls {{ display: flex; gap: 8px; margin-top: 14px; }}
  input {{ flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #475569; background: #0f172a; color: #fff; }}
  button {{ padding: 10px 14px; border-radius: 6px; border: none; cursor: pointer; font-weight: 600; }}
  .b-send {{ background: #22c55e; color: #052e16; }}
  .b-int {{ background: #ef4444; color: #fff; }}
  .b-steer {{ background: #f59e0b; color: #451a03; }}
  .b-follow {{ background: #38bdf8; color: #082f49; }}
</style>
</head>
<body>
  <h2>Tenon — Compiled Distributed Prototype</h2>
  <div class="bar" id="status">Connecting to unified /api/client/stream (coalesce={coalesce_ms}ms)...</div>
  <div id="timeline"></div>
  <div class="controls">
    <input id="inp" placeholder="Type message / steer / interrupt..." value="Explain Tenon fencing" />
    <button class="b-send" onclick="post('/api/sessions/main/send', 'c')">Send</button>
    <button class="b-int" onclick="post('/api/sessions/main/interrupt', 'i')">Interrupt (New Branch)</button>
    <button class="b-steer" onclick="post('/api/sessions/main/steer', 's')">Steer (P0)</button>
    <button class="b-follow" onclick="post('/api/sessions/main/followup', 'f')">Followup (P1)</button>
  </div>
<script>
const COALESCE_MS = {coalesce_ms};
let state = {{ history: [], queued: [], live: null, phase: 'idle', branch: 1, epoch: 0, renders: 0, deltas: 0 }};
let pendingDelta = '';
let coalesceTimer = null;

function render() {{
  state.renders++;
  document.getElementById('status').textContent =
    `branch=#${{state.branch}} | exec_epoch=${{state.epoch}} | phase=${{state.phase}} | deltas=${{state.deltas}} | coalesced_renders=${{state.renders}} (window=${{COALESCE_MS}}ms)`;
  const el = document.getElementById('timeline');
  el.innerHTML = '';
  for (const t of state.history) {{
    const d = document.createElement('div');
    d.className = 'msg ' + t.role;
    d.textContent = `[${{t.role}} b#${{t.branch_id}}] ${{t.content}}`;
    el.appendChild(d);
  }}
  if (state.live) {{
    const d = document.createElement('div');
    d.className = 'msg live';
    d.textContent = `[streaming b#${{state.live.branch_id}} seq=${{state.live.through_seq}}] ${{state.live.text}}`;
    el.appendChild(d);
  }}
  for (const q of state.queued) {{
    const d = document.createElement('div');
    d.className = 'msg';
    d.textContent = `[queued:${{q.kind}}] ${{q.text}}`;
    el.appendChild(d);
  }}
}}

function flushCoalesced() {{
  coalesceTimer = null;
  if (pendingDelta && state.live) {{
    state.live.text += pendingDelta;
    pendingDelta = '';
    render();
  }}
}}

const es = new EventSource('/api/client/stream');
es.onmessage = (ev) => {{
  const f = JSON.parse(ev.data);
  if (f.event === 'session.snapshot') {{
    pendingDelta = '';
    if (coalesceTimer) {{ clearTimeout(coalesceTimer); coalesceTimer = null; }}
    state.history = f.view.history;
    state.queued = f.view.queued;
    state.phase = f.view.phase;
    state.branch = f.view.active_branch;
    state.epoch = f.view.exec_epoch;
    state.live = f.live_bubble;
    render(); // immediate bypass of debounce!
  }} else if (f.event === 'session.delta') {{
    state.deltas++;
    if (!state.live || state.live.stream_id !== f.stream_id) {{
      state.live = {{ branch_id: f.branch_id, run_id: f.run_id, stream_id: f.stream_id, through_seq: f.seq, text: '' }};
    }}
    state.live.through_seq = f.seq;
    pendingDelta += f.delta;
    if (!coalesceTimer) {{
      coalesceTimer = setTimeout(flushCoalesced, COALESCE_MS);
    }}
  }}
}};

let seqId = 1;
async function post(url, prefix) {{
  const text = document.getElementById('inp').value;
  const id = prefix + '_' + Date.now() + '_' + (seqId++);
  const body = prefix === 'i' ? {{ cmd_id: id, text }} : {{ client_msg_id: id, text }};
  await fetch(url, {{ method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify(body) }});
}}
</script>
</body>
</html>
"#
    )
}
