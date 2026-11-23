use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

#[derive(Debug, Clone)]
pub struct LlmConfig {
    /// Host:port (e.g. `127.0.0.1:11434` or OpenAI-compatible local gateway)
    pub base_addr: String,
    pub path: String,
    pub api_key: Option<String>,
    pub model: String,
}

impl LlmConfig {
    pub fn from_env() -> Option<Self> {
        let base = std::env::var("TENON_LLM_BASE_URL").ok()?;
        let trimmed = base
            .trim()
            .trim_start_matches("http://")
            .trim_end_matches('/');
        let (addr, prefix) = match trimmed.split_once('/') {
            Some((h, p)) => (h.to_string(), format!("/{}", p)),
            None => (trimmed.to_string(), String::new()),
        };
        let path = if prefix.ends_with("/chat/completions") {
            prefix
        } else if prefix.ends_with("/v1") {
            format!("{}/chat/completions", prefix)
        } else {
            format!("{}/v1/chat/completions", prefix)
        };
        Some(Self {
            base_addr: addr,
            path,
            api_key: std::env::var("TENON_LLM_API_KEY").ok(),
            model: std::env::var("TENON_LLM_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into()),
        })
    }
}

/// Real OpenAI-compatible HTTP/1.1 Streaming SSE Client (`stream: true`):
/// Parses `data: {"choices":[{"delta":{"content":"..."}}]}` chunks token-by-token.
/// If `on_token` returns `Err` (e.g. Hard EffectProxy Fencing!), immediately drops the TCP stream!
pub async fn stream_openai_sse<F>(
    cfg: &LlmConfig,
    prompt: &str,
    mut on_token: F,
) -> Result<String, String>
where
    F: FnMut(u64, &str) -> Result<(), String>,
{
    let mut sock = TcpStream::connect(&cfg.base_addr)
        .await
        .map_err(|e| format!("LLM connect {} failed: {}", cfg.base_addr, e))?;

    let payload = serde_json::json!({
        "model": cfg.model,
        "stream": true,
        "messages": [
            { "role": "user", "content": prompt }
        ]
    })
    .to_string();

    let auth_header = match &cfg.api_key {
        Some(k) if !k.is_empty() => format!("Authorization: Bearer {}\r\n", k),
        _ => String::new(),
    };

    let req = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\n{}Content-Type: application/json\r\nAccept: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        cfg.path,
        cfg.base_addr,
        auth_header,
        payload.len(),
        payload
    );
    sock.write_all(req.as_bytes())
        .await
        .map_err(|e| e.to_string())?;

    let mut reader = BufReader::new(sock);
    // Read HTTP status & headers
    let mut status_line = String::new();
    reader
        .read_line(&mut status_line)
        .await
        .map_err(|e| e.to_string())?;
    if !status_line.contains(" 200 ") {
        return Err(format!("LLM server returned non-200: {}", status_line.trim()));
    }
    loop {
        let mut h = String::new();
        reader.read_line(&mut h).await.map_err(|e| e.to_string())?;
        if h.trim().is_empty() {
            break;
        }
    }

    let mut full = String::new();
    let mut seq = 0u64;
    loop {
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .await
            .map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        let Some(data) = trimmed.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
            if let Some(tok) = v["choices"][0]["delta"]["content"].as_str() {
                if !tok.is_empty() {
                    seq += 1;
                    on_token(seq, tok)?;
                    full.push_str(tok);
                }
            }
        }
    }
    Ok(full)
}

/// Built-in Mock OpenAI SSE Server (`POST /v1/chat/completions`) for reproducible live E2E testing!
pub async fn spawn_mock_openai_sse_server(tokens: Vec<String>) -> Result<String, String> {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| e.to_string())?;
    let bound = listener.local_addr().map_err(|e| e.to_string())?.to_string();

    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let toks = tokens.clone();
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
                if clen > 0 {
                    let mut body = vec![0u8; clen];
                    let _ = tokio::io::AsyncReadExt::read_exact(&mut reader, &mut body).await;
                }
                let mut sock = reader.into_inner();
                let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
                if sock.write_all(hdr.as_bytes()).await.is_err() {
                    return;
                }
                for tok in &toks {
                    let chunk = serde_json::json!({
                        "id": "chatcmpl-tenon",
                        "object": "chat.completion.chunk",
                        "choices": [{ "index": 0, "delta": { "content": tok }, "finish_reason": null }]
                    })
                    .to_string();
                    let frame = format!("data: {}\n\n", chunk);
                    if sock.write_all(frame.as_bytes()).await.is_err() {
                        return;
                    }
                    let _ = sock.flush().await;
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }
                let _ = sock.write_all(b"data: [DONE]\n\n").await;
                let _ = sock.flush().await;
            });
        }
    });

    Ok(bound)
}
