use crate::authority::ExecSlice;
use crate::sandbox::run_fenced_sandbox;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Standalone External Worker OS Process loop communicating with Authority over real TCP HTTP RPC!
pub async fn run_external_worker(
    authority_addr: &str,
    worker_id: &str,
    session_id: &str,
    token_delay_ms: u64,
    force_takeover: bool,
    once: bool,
) -> Result<(), String> {
    let host_port = authority_addr
        .trim_start_matches("http://")
        .trim_end_matches('/');
    let mut takeover_flag = force_takeover;

    loop {
        let claim_body = serde_json::json!({
            "session_id": session_id,
            "worker_id": worker_id,
            "force_takeover": takeover_flag
        })
        .to_string();
        let (status, resp_body) = http_rpc(host_port, "POST", "/rpc/claim", &claim_body).await?;
        if status == 200 {
            takeover_flag = false;
            let slice: ExecSlice =
                serde_json::from_str(&resp_body).map_err(|e| format!("bad slice json: {}", e))?;
            println!(
                "[worker {} pid={}] claimed session={} branch={} epoch={} run={}",
                worker_id,
                std::process::id(),
                slice.session_id,
                slice.branch_id,
                slice.exec_epoch,
                slice.run_id
            );

            // Step 1: Execute isolated Fabric Sandbox child process!
            let sb = run_fenced_sandbox(
                &slice.session_id,
                slice.exec_epoch,
                &slice.prompt,
                Duration::from_secs(3),
            )?;
            println!(
                "[worker {} pid={}] fabric sandbox child_pid={} exited={} ({})",
                worker_id,
                std::process::id(),
                sb.child_pid,
                sb.exit_code,
                sb.stdout
            );

            // Step 2: Stream tokens through Authority's Hard-Fenced EffectProxy!
            let tokens = [
                format!("[worker:{} pid={}] ", worker_id, std::process::id()),
                format!("[{}] ", sb.stdout),
                format!("answering [{}] ", slice.prompt),
                format!("on branch#{} (epoch {})... ", slice.branch_id, slice.exec_epoch),
                "verified ".into(),
                "and ".into(),
                "complete.".into(),
            ];
            let mut full = String::new();
            let mut seq = 0u64;
            let mut fenced_out = false;

            for (idx, tok) in tokens.iter().enumerate() {
                if token_delay_ms > 0 {
                    tokio::time::sleep(Duration::from_millis(token_delay_ms)).await;
                }
                if idx == 3 {
                    // Check P0 Steering at mid-run safe point via fenced RPC!
                    let steer_req = serde_json::json!({ "slice": slice }).to_string();
                    let (st_code, st_body) =
                        http_rpc(host_port, "POST", "/rpc/steer/drain", &steer_req).await?;
                    if st_code == 409 {
                        fenced_out = true;
                        break;
                    }
                    if st_code == 200 {
                        if let Ok(steers) = serde_json::from_str::<Vec<String>>(&st_body) {
                            for s in steers {
                                seq += 1;
                                let note = format!(" [STEERED: {}] ", s);
                                let req = serde_json::json!({
                                    "slice": slice,
                                    "seq": seq,
                                    "delta": note
                                })
                                .to_string();
                                let (pcode, _) =
                                    http_rpc(host_port, "POST", "/rpc/proxy/token", &req).await?;
                                if pcode == 409 {
                                    fenced_out = true;
                                    break;
                                }
                                full.push_str(&note);
                            }
                        }
                    }
                    if fenced_out {
                        break;
                    }
                }
                seq += 1;
                let req = serde_json::json!({
                    "slice": slice,
                    "seq": seq,
                    "delta": tok
                })
                .to_string();
                let (pcode, _) = http_rpc(host_port, "POST", "/rpc/proxy/token", &req).await?;
                if pcode == 409 {
                    fenced_out = true;
                    break;
                }
                full.push_str(tok);
            }

            if fenced_out {
                println!(
                    "[worker {} pid={}] HARD-FENCED out on epoch={}; recording late audit receipt",
                    worker_id,
                    std::process::id(),
                    slice.exec_epoch
                );
                let rec = serde_json::json!({
                    "attempt_id": format!("att_{}_e{}", slice.run_id, slice.exec_epoch),
                    "session_id": slice.session_id,
                    "branch_id": slice.branch_id,
                    "exec_epoch": slice.exec_epoch,
                    "outcome": "aborted_by_proxy_fence",
                    "payload": format!("partial_seq={}", seq)
                })
                .to_string();
                let _ = http_rpc(host_port, "POST", "/rpc/receipt", &rec).await;
            } else {
                let commit_req = serde_json::json!({
                    "slice": slice,
                    "assistant_text": full
                })
                .to_string();
                let (ccode, cbody) =
                    http_rpc(host_port, "POST", "/rpc/commit", &commit_req).await?;
                println!(
                    "[worker {} pid={}] commit HTTP {} ({})",
                    worker_id,
                    std::process::id(),
                    ccode,
                    cbody.trim()
                );
            }

            if once {
                return Ok(());
            }
        } else {
            tokio::time::sleep(Duration::from_millis(30)).await;
        }
    }
}

pub async fn http_rpc(
    host_port: &str,
    method: &str,
    path: &str,
    body: &str,
) -> Result<(u16, String), String> {
    let mut sock = TcpStream::connect(host_port)
        .await
        .map_err(|e| format!("connect {} failed: {}", host_port, e))?;
    let req = format!(
        "{} {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        method,
        path,
        host_port,
        body.len(),
        body
    );
    sock.write_all(req.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    sock.read_to_end(&mut buf)
        .await
        .map_err(|e| e.to_string())?;
    let raw = String::from_utf8_lossy(&buf).to_string();
    let status = raw
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(500);
    let body_out = raw
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_default();
    Ok((status, body_out))
}
