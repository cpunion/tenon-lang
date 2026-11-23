use crate::openai_responses::ToolSpec;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExecResult {
    pub tool: String,
    pub ok: bool,
    pub summary: String,
    pub output_json: String,
    pub duration_ms: u64,
}

#[derive(Debug, Clone)]
pub struct AgentToolkit {
    pub workspace_root: PathBuf,
    pub allow_outside_workspace: bool,
}

impl AgentToolkit {
    pub fn new(workspace_root: PathBuf, allow_outside_workspace: bool) -> Self {
        let canonical = fs::canonicalize(&workspace_root).unwrap_or(workspace_root);
        Self {
            workspace_root: canonical,
            allow_outside_workspace,
        }
    }

    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        vec![
            ToolSpec {
                name: "shell".into(),
                description: "Run a shell command in the workspace directory via /bin/sh -c. Returns exit_code, stdout, and stderr.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": { "type": "string", "description": "Shell command to execute" },
                        "timeout_sec": { "type": "integer", "description": "Optional timeout in seconds (default 30, max 120)" }
                    },
                    "required": ["command"]
                }),
            },
            ToolSpec {
                name: "read_file".into(),
                description: "Read a UTF-8 text file with optional 1-based start_line and end_line range.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path relative to workspace root or absolute path" },
                        "start_line": { "type": "integer", "description": "Optional 1-based start line (inclusive)" },
                        "end_line": { "type": "integer", "description": "Optional 1-based end line (inclusive)" }
                    },
                    "required": ["path"]
                }),
            },
            ToolSpec {
                name: "write_file".into(),
                description: "Create or overwrite a text file (parent directories are created automatically).".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path to write" },
                        "content": { "type": "string", "description": "Full text content to write" }
                    },
                    "required": ["path", "content"]
                }),
            },
            ToolSpec {
                name: "edit_file".into(),
                description: "Replace an exact substring `target` with `replacement` inside an existing file.".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path to modify" },
                        "target": { "type": "string", "description": "Exact existing text to replace" },
                        "replacement": { "type": "string", "description": "Replacement text" }
                    },
                    "required": ["path", "target", "replacement"]
                }),
            },
            ToolSpec {
                name: "list_dir".into(),
                description: "List files and directories under a path (defaults to workspace root, skips .git/target/node_modules).".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Optional subdirectory path (default '.')" },
                        "max_depth": { "type": "integer", "description": "Optional recursion depth (default 2, max 4)" }
                    }
                }),
            },
            ToolSpec {
                name: "grep_search".into(),
                description: "Search for a literal substring across workspace text files (returns file:line: text).".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Substring to search for" },
                        "path": { "type": "string", "description": "Optional subdirectory or file to search (default '.')" }
                    },
                    "required": ["query"]
                }),
            },
        ]
    }

    pub async fn execute(&self, name: &str, raw_args: &str) -> ToolExecResult {
        let start = Instant::now();
        let args: Value = serde_json::from_str(raw_args).unwrap_or_else(|_| json!({}));
        let (ok, summary, payload) = match name {
            "shell" => self.exec_shell(&args).await,
            "read_file" => self.exec_read_file(&args),
            "write_file" => self.exec_write_file(&args),
            "edit_file" => self.exec_edit_file(&args),
            "list_dir" => self.exec_list_dir(&args),
            "grep_search" => self.exec_grep_search(&args),
            other => (
                false,
                format!("unknown tool `{}`", other),
                json!({ "ok": false, "error": format!("unknown tool `{}`", other) }),
            ),
        };
        ToolExecResult {
            tool: name.to_string(),
            ok,
            summary,
            output_json: payload.to_string(),
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }

    fn resolve_path(&self, raw: &str) -> Result<PathBuf, String> {
        let p = Path::new(raw);
        let joined = if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.workspace_root.join(p)
        };
        let normalized = normalize_path(&joined);
        if !self.allow_outside_workspace && !normalized.starts_with(&self.workspace_root) {
            return Err(format!(
                "path `{}` escapes workspace root `{}`",
                normalized.display(),
                self.workspace_root.display()
            ));
        }
        Ok(normalized)
    }

    async fn exec_shell(&self, args: &Value) -> (bool, String, Value) {
        let Some(cmd) = args["command"].as_str() else {
            return (false, "missing command".into(), json!({"ok": false, "error": "missing `command`"}));
        };
        let timeout_sec = args["timeout_sec"].as_u64().unwrap_or(30).clamp(1, 120);
        let fut = Command::new("/bin/sh")
            .arg("-c")
            .arg(cmd)
            .current_dir(&self.workspace_root)
            .output();

        match tokio::time::timeout(Duration::from_secs(timeout_sec), fut).await {
            Ok(Ok(out)) => {
                let code = out.status.code().unwrap_or(-1);
                let stdout = truncate_utf8(&String::from_utf8_lossy(&out.stdout), 24 * 1024);
                let stderr = truncate_utf8(&String::from_utf8_lossy(&out.stderr), 8 * 1024);
                let ok = out.status.success();
                let summary = format!("exit={} ({}B stdout)", code, stdout.len());
                (
                    ok,
                    summary,
                    json!({ "ok": ok, "exit_code": code, "stdout": stdout, "stderr": stderr }),
                )
            }
            Ok(Err(e)) => (
                false,
                format!("spawn error: {}", e),
                json!({ "ok": false, "error": e.to_string() }),
            ),
            Err(_) => (
                false,
                format!("timed out after {}s", timeout_sec),
                json!({ "ok": false, "error": format!("command timed out after {}s", timeout_sec) }),
            ),
        }
    }

    fn exec_read_file(&self, args: &Value) -> (bool, String, Value) {
        let Some(raw) = args["path"].as_str() else {
            return (false, "missing path".into(), json!({"ok": false, "error": "missing `path`"}));
        };
        let path = match self.resolve_path(raw) {
            Ok(p) => p,
            Err(e) => return (false, e.clone(), json!({"ok": false, "error": e})),
        };
        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => return (false, format!("read error: {}", e), json!({"ok": false, "error": e.to_string()})),
        };
        let lines: Vec<&str> = content.lines().collect();
        let total = lines.len();
        let start = args["start_line"].as_u64().unwrap_or(1).max(1) as usize;
        let end = args["end_line"]
            .as_u64()
            .map(|n| (n as usize).min(total))
            .unwrap_or_else(|| (start + 399).min(total));
        let mut numbered = String::new();
        if start <= end && start <= total {
            for (idx, line) in lines[(start - 1)..end].iter().enumerate() {
                numbered.push_str(&format!("{}: {}\n", start + idx, line));
            }
        }
        let numbered = truncate_utf8(&numbered, 32 * 1024);
        let summary = format!("{} (lines {}-{} of {})", raw, start, end, total);
        (
            true,
            summary,
            json!({ "ok": true, "path": raw, "total_lines": total, "content": numbered }),
        )
    }

    fn exec_write_file(&self, args: &Value) -> (bool, String, Value) {
        let (Some(raw), Some(content)) = (args["path"].as_str(), args["content"].as_str()) else {
            return (
                false,
                "missing path/content".into(),
                json!({"ok": false, "error": "missing `path` or `content`"}),
            );
        };
        let path = match self.resolve_path(raw) {
            Ok(p) => p,
            Err(e) => return (false, e.clone(), json!({"ok": false, "error": e})),
        };
        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                return (false, e.to_string(), json!({"ok": false, "error": e.to_string()}));
            }
        }
        match fs::write(&path, content) {
            Ok(()) => (
                true,
                format!("wrote {} ({} bytes)", raw, content.len()),
                json!({ "ok": true, "path": raw, "bytes": content.len() }),
            ),
            Err(e) => (false, e.to_string(), json!({"ok": false, "error": e.to_string()})),
        }
    }

    fn exec_edit_file(&self, args: &Value) -> (bool, String, Value) {
        let (Some(raw), Some(target), Some(replacement)) = (
            args["path"].as_str(),
            args["target"].as_str(),
            args["replacement"].as_str(),
        ) else {
            return (
                false,
                "missing path/target/replacement".into(),
                json!({"ok": false, "error": "missing `path`, `target`, or `replacement`"}),
            );
        };
        let path = match self.resolve_path(raw) {
            Ok(p) => p,
            Err(e) => return (false, e.clone(), json!({"ok": false, "error": e})),
        };
        let cur = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => return (false, e.to_string(), json!({"ok": false, "error": e.to_string()})),
        };
        let matches = cur.matches(target).count();
        if matches == 0 {
            return (
                false,
                format!("target not found in {}", raw),
                json!({"ok": false, "error": "target string not found in file"}),
            );
        }
        let updated = cur.replace(target, replacement);
        match fs::write(&path, &updated) {
            Ok(()) => (
                true,
                format!("edited {} ({} replacement(s))", raw, matches),
                json!({ "ok": true, "path": raw, "replacements": matches }),
            ),
            Err(e) => (false, e.to_string(), json!({"ok": false, "error": e.to_string()})),
        }
    }

    fn exec_list_dir(&self, args: &Value) -> (bool, String, Value) {
        let raw = args["path"].as_str().unwrap_or(".");
        let max_depth = args["max_depth"].as_u64().unwrap_or(2).clamp(1, 4) as usize;
        let root = match self.resolve_path(raw) {
            Ok(p) => p,
            Err(e) => return (false, e.clone(), json!({"ok": false, "error": e})),
        };
        let mut entries = Vec::new();
        walk_dir(&root, &self.workspace_root, 0, max_depth, &mut entries);
        let summary = format!("{} ({} entries)", raw, entries.len());
        (true, summary, json!({ "ok": true, "entries": entries }))
    }

    fn exec_grep_search(&self, args: &Value) -> (bool, String, Value) {
        let Some(query) = args["query"].as_str() else {
            return (false, "missing query".into(), json!({"ok": false, "error": "missing `query`"}));
        };
        let raw = args["path"].as_str().unwrap_or(".");
        let root = match self.resolve_path(raw) {
            Ok(p) => p,
            Err(e) => return (false, e.clone(), json!({"ok": false, "error": e})),
        };
        let mut hits = Vec::new();
        grep_recursive(&root, &self.workspace_root, query, &mut hits);
        let summary = format!("`{}` -> {} hit(s)", query, hits.len());
        (true, summary, json!({ "ok": true, "hits": hits }))
    }
}

fn walk_dir(dir: &Path, root: &Path, depth: usize, max_depth: usize, out: &mut Vec<String>) {
    if depth > max_depth || out.len() >= 200 {
        return;
    }
    let Ok(rd) = fs::read_dir(dir) else {
        return;
    };
    let mut items: Vec<PathBuf> = rd.filter_map(Result::ok).map(|e| e.path()).collect();
    items.sort();
    for p in items {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if matches!(name, ".git" | "target" | "node_modules" | ".DS_Store") {
            continue;
        }
        let rel = p.strip_prefix(root).unwrap_or(&p).display().to_string();
        if p.is_dir() {
            out.push(format!("{}/", rel));
            walk_dir(&p, root, depth + 1, max_depth, out);
        } else {
            out.push(rel);
        }
        if out.len() >= 200 {
            break;
        }
    }
}

fn grep_recursive(path: &Path, root: &Path, query: &str, out: &mut Vec<String>) {
    if out.len() >= 100 {
        return;
    }
    if path.is_file() {
        if let Ok(content) = fs::read_to_string(path) {
            let rel = path.strip_prefix(root).unwrap_or(path).display().to_string();
            for (idx, line) in content.lines().enumerate() {
                if line.contains(query) {
                    out.push(format!("{}:{}: {}", rel, idx + 1, truncate_utf8(line, 240)));
                    if out.len() >= 100 {
                        break;
                    }
                }
            }
        }
        return;
    }
    let Ok(rd) = fs::read_dir(path) else {
        return;
    };
    let mut items: Vec<PathBuf> = rd.filter_map(Result::ok).map(|e| e.path()).collect();
    items.sort();
    for p in items {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if matches!(name, ".git" | "target" | "node_modules") {
            continue;
        }
        grep_recursive(&p, root, query, out);
    }
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn truncate_utf8(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n...[truncated {} bytes]", &s[..end], s.len() - end)
}
