use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxResult {
    pub child_pid: u32,
    pub exit_code: i32,
    pub stdout: String,
    pub duration_ms: u64,
}

/// Fabric Sandbox isolated child-process executor:
/// - Clears inherited environment (`env_clear`) to prevent secret leakage
/// - Restricts working directory to `/tmp`
/// - Caps execution time and stdout size
pub fn run_fenced_sandbox(
    session_id: &str,
    exec_epoch: i64,
    prompt: &str,
    timeout: Duration,
) -> Result<SandboxResult, String> {
    let start = Instant::now();
    // Safe deterministic sandbox diagnostic command (never interpolates untrusted shell metacharacters!)
    let mut child = Command::new("/bin/sh")
        .arg("-c")
        .arg(r#"printf "sandbox_pid=%s session=%s epoch=%s len=%s" "$$" "$TENON_SESSION" "$TENON_EPOCH" "${#TENON_PROMPT}""#)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("TENON_SESSION", session_id)
        .env("TENON_EPOCH", exec_epoch.to_string())
        .env("TENON_PROMPT", prompt)
        .current_dir("/tmp")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn fabric sandbox: {}", e))?;

    let pid = child.id();
    loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            let output = child.wait_with_output().map_err(|e| e.to_string())?;
            let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
            if stdout.len() > 4096 {
                stdout.truncate(4096);
            }
            return Ok(SandboxResult {
                child_pid: pid,
                exit_code: status.code().unwrap_or(-1),
                stdout,
                duration_ms: start.elapsed().as_millis() as u64,
            });
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "fabric sandbox pid={} timed out after {}ms",
                pid,
                timeout.as_millis()
            ));
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}
