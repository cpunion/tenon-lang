use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tenon_verify::ResolvedSubscriberSpec;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelWebhookEvent {
    pub channel_id: String,
    pub mode: String, // "draft_edit" | "final_send"
    pub chunk_index: usize,
    pub total_chunks: usize,
    pub text: String,
}

/// Real Outbound Channel Adapter enforcing compiled manifest policies:
/// - `dead_stream_eliminated`: non-editable channels (e.g. Email) NEVER receive streaming deltas
/// - `throttle_ms`: trailing throttle window for editable channels (e.g. Telegram 1200ms)
/// - `split_bytes`: UTF-8-safe auto chunking (e.g. Telegram 4096 bytes)
pub struct ChannelWebhookAdapter {
    webhook_addr: String,
    specs: Vec<ResolvedSubscriberSpec>,
    last_emit: Mutex<BTreeMap<String, Instant>>,
}

impl ChannelWebhookAdapter {
    pub fn new(webhook_addr: impl Into<String>, specs: Vec<ResolvedSubscriberSpec>) -> Self {
        let addr = webhook_addr
            .into()
            .trim_start_matches("http://")
            .trim_end_matches('/')
            .to_string();
        Self {
            webhook_addr: addr,
            specs,
            last_emit: Mutex::new(BTreeMap::new()),
        }
    }

    pub async fn on_streaming_prefix(&self, prefix_text: &str) {
        let now = Instant::now();
        for spec in &self.specs {
            if spec.kind != "channel" || spec.dead_stream_eliminated {
                continue;
            }
            let cid = spec.channel_id.as_deref().unwrap_or(&spec.name);
            let throttle_ms = spec.throttle_ms.unwrap_or(0);
            let allowed = {
                let mut map = self.last_emit.lock().unwrap();
                match map.get(cid) {
                    Some(prev) if (now.duration_since(*prev).as_millis() as u64) < throttle_ms => {
                        false
                    }
                    _ => {
                        map.insert(cid.to_string(), now);
                        true
                    }
                }
            };
            if !allowed {
                continue;
            }
            let chunks = split_utf8_chunks(prefix_text, spec.split_bytes.unwrap_or(65536) as usize);
            let total = chunks.len();
            for (idx, chunk) in chunks.into_iter().enumerate() {
                let ev = ChannelWebhookEvent {
                    channel_id: cid.to_string(),
                    mode: "draft_edit".into(),
                    chunk_index: idx,
                    total_chunks: total,
                    text: chunk,
                };
                let _ = post_webhook(&self.webhook_addr, &ev).await;
            }
        }
    }

    pub async fn on_final_committed(&self, final_text: &str) {
        for spec in &self.specs {
            if spec.kind != "channel" {
                continue;
            }
            let cid = spec.channel_id.as_deref().unwrap_or(&spec.name);
            let chunks = split_utf8_chunks(final_text, spec.split_bytes.unwrap_or(65536) as usize);
            let total = chunks.len();
            for (idx, chunk) in chunks.into_iter().enumerate() {
                let ev = ChannelWebhookEvent {
                    channel_id: cid.to_string(),
                    mode: "final_send".into(),
                    chunk_index: idx,
                    total_chunks: total,
                    text: chunk,
                };
                let _ = post_webhook(&self.webhook_addr, &ev).await;
            }
        }
    }
}

pub fn split_utf8_chunks(text: &str, max_bytes: usize) -> Vec<String> {
    let limit = max_bytes.max(1);
    if text.len() <= limit {
        return vec![text.to_string()];
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if cur.len() + ch.len_utf8() > limit && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        cur.push(ch);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

async fn post_webhook(addr: &str, ev: &ChannelWebhookEvent) -> Result<(), String> {
    let mut sock = TcpStream::connect(addr).await.map_err(|e| e.to_string())?;
    let body = serde_json::to_string(ev).unwrap_or_default();
    let req = format!(
        "POST /webhook/channel HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        addr,
        body.len(),
        body
    );
    sock.write_all(req.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    let mut sink = Vec::new();
    let _ = sock.read_to_end(&mut sink).await;
    Ok(())
}

/// Built-in Mock Webhook Receiver Server for reproducible live HTTP channel delivery tests!
pub async fn spawn_mock_webhook_receiver()
-> Result<(String, Arc<Mutex<Vec<ChannelWebhookEvent>>>), String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| e.to_string())?;
    let bound = listener.local_addr().map_err(|e| e.to_string())?.to_string();
    let received: Arc<Mutex<Vec<ChannelWebhookEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let rec_clone = received.clone();

    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let store = rec_clone.clone();
            tokio::spawn(async move {
                let mut reader = BufReader::new(stream);
                let mut clen = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
                        return;
                    }
                    let t = line.trim();
                    if t.is_empty() {
                        break;
                    }
                    if let Some((k, v)) = t.split_once(':') {
                        if k.eq_ignore_ascii_case("content-length") {
                            clen = v.trim().parse().unwrap_or(0);
                        }
                    }
                }
                let mut body = vec![0u8; clen];
                if clen > 0 && reader.read_exact(&mut body).await.is_err() {
                    return;
                }
                if let Ok(ev) = serde_json::from_slice::<ChannelWebhookEvent>(&body) {
                    store.lock().unwrap().push(ev);
                }
                let mut sock = reader.into_inner();
                let _ = sock
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                    .await;
            });
        }
    });

    Ok((bound, received))
}
