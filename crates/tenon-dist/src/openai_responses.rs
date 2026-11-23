use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// OpenAI Direct Responses API configuration aligned with `dast-os` (`docs/openai-model-routing-20261105.md`).
/// Default models:
/// - `gpt-6-luna`   (Main / simple,   reasoning=low,    $0.10 / $0.50 / $0.01 per 1M)
/// - `gpt-6.1-terra`(general/visual,  reasoning=medium, $1.00 / $5.00 / $0.05 per 1M)
/// - `gpt-6.1-sol`  (coding,          reasoning=medium, $2.00 / $10.00 / $0.10 per 1M)
/// - `gpt-6-astra`  (reasoning,       reasoning=high,   $10.00 / $50.00 / $1.00 per 1M)
#[derive(Debug, Clone)]
pub struct OpenAIResponsesConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub reasoning_effort: String,
    pub max_output_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallItem {
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponseUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cached_tokens: u64,
    pub estimated_cost_usd: f64,
}

#[derive(Debug, Clone)]
pub struct ResponsesTurnOutput {
    pub response_id: String,
    pub model: String,
    pub text: String,
    pub tool_calls: Vec<FunctionCallItem>,
    /// Verbatim raw output items (reasoning with encrypted_content, message, function_call) for faithful replay!
    pub raw_output_items: Vec<Value>,
    pub usage: ResponseUsage,
}

#[derive(Debug, Clone)]
pub enum ResponsesStreamDelta {
    Text(String),
    Thinking(String),
    ToolArgsProgress,
}

impl OpenAIResponsesConfig {
    /// Resolve model alias (`luna`/`simple`, `terra`/`general`, `sol`/`coding`, `astra`/`reasoning`, or explicit name).
    pub fn resolve_model_and_effort(model_or_alias: &str) -> (String, String) {
        match model_or_alias.trim().to_ascii_lowercase().as_str() {
            "" | "luna" | "main" | "simple" | "gpt-6-luna" => {
                ("gpt-6-luna".into(), "low".into())
            }
            "terra" | "general" | "visual" | "gpt-6.1-terra" => {
                ("gpt-6.1-terra".into(), "medium".into())
            }
            "sol" | "coding" | "gpt-6.1-sol" => ("gpt-6.1-sol".into(), "medium".into()),
            "astra" | "reasoning" | "gpt-6-astra" => ("gpt-6-astra".into(), "high".into()),
            other => (other.to_string(), "low".into()),
        }
    }

    /// Resolve `OPENAI_API_KEY` securely from:
    /// 1) Process environment (`OPENAI_API_KEY`)
    /// 2) Explicit `--env-file <path>` if supplied
    /// 3) `./.env` in current working directory
    /// 4) Local `dast-os` `.env` / `.data/secrets/local.env` if present on disk
    pub fn load(
        model_or_alias: &str,
        explicit_env_file: Option<&Path>,
        effort_override: Option<&str>,
    ) -> Result<Self, String> {
        let mut env_map: BTreeMap<String, String> = BTreeMap::new();

        // Load fallback env files from lowest priority to highest priority (never overwriting process env!)
        let mut candidate_paths: Vec<std::path::PathBuf> = vec![
            std::path::PathBuf::from("/Users/lijie/source/dast.ai/dast-os/.data/secrets/local.env"),
            std::path::PathBuf::from("/Users/lijie/source/dast.ai/dast-os/.env"),
            std::path::PathBuf::from(".env"),
        ];
        if let Some(p) = explicit_env_file {
            candidate_paths.push(p.to_path_buf());
        }
        for path in candidate_paths {
            if let Ok(content) = fs::read_to_string(&path) {
                parse_dotenv_into(&content, &mut env_map);
            }
        }

        let api_key = std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| env_map.get("OPENAI_API_KEY").cloned())
            .ok_or_else(|| {
                "OPENAI_API_KEY not found in environment, .env, or dast-os/.env".to_string()
            })?;

        let base_url = std::env::var("OPENAI_BASE_URL")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| env_map.get("OPENAI_BASE_URL").cloned())
            .unwrap_or_else(|| "https://api.openai.com/v1".into())
            .trim_end_matches('/')
            .to_string();

        let (model, default_effort) = Self::resolve_model_and_effort(model_or_alias);
        let reasoning_effort = effort_override
            .map(|s| s.to_string())
            .unwrap_or(default_effort);

        Ok(Self {
            base_url,
            api_key,
            model,
            reasoning_effort,
            max_output_tokens: 4096,
        })
    }
}

/// Robust `.env` parser splitting on first `=` and stripping optional quotes (handles unquoted values like `Name <email>` safely!).
pub fn parse_dotenv_into(content: &str, out: &mut BTreeMap<String, String>) {
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        if let Some((k, v)) = line.split_once('=') {
            let key = k.trim();
            if key.is_empty() {
                continue;
            }
            let mut val = v.trim().to_string();
            if (val.starts_with('"') && val.ends_with('"') && val.len() >= 2)
                || (val.starts_with('\'') && val.ends_with('\'') && val.len() >= 2)
            {
                val = val[1..val.len() - 1].to_string();
            }
            if !val.is_empty() {
                out.insert(key.to_string(), val);
            }
        }
    }
}

pub fn estimate_cost_usd(model: &str, u: &ResponseUsage) -> f64 {
    // dast-os model-cost-v2-20261105 rates ($ per 1M tokens: input, output, cached_input)
    let (in_rate, out_rate, cache_rate) = match model {
        m if m.starts_with("gpt-6-luna") => (0.10, 0.50, 0.01),
        m if m.starts_with("gpt-6.1-terra") => (1.00, 5.00, 0.05),
        m if m.starts_with("gpt-6.1-sol") => (2.00, 10.00, 0.10),
        m if m.starts_with("gpt-6-astra") => (10.00, 50.00, 1.00),
        _ => (1.00, 5.00, 0.10),
    };
    let uncached_in = u.input_tokens.saturating_sub(u.cached_tokens) as f64;
    let cached_in = u.cached_tokens as f64;
    let out_tok = u.output_tokens as f64;
    (uncached_in * in_rate + cached_in * cache_rate + out_tok * out_rate) / 1_000_000.0
}

/// Stream one OpenAI Direct Responses API turn (`POST /v1/responses`, `stream: true`, `store: false`, `strict: false`).
/// Supports live token callback with cancellation / Fencing abort!
pub async fn stream_openai_responses<F>(
    client: &reqwest::Client,
    cfg: &OpenAIResponsesConfig,
    session_id: &str,
    input_items: &[Value],
    tools: &[ToolSpec],
    mut on_delta: F,
) -> Result<ResponsesTurnOutput, String>
where
    F: FnMut(ResponsesStreamDelta) -> Result<(), String>,
{
    let resp_tools: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "strict": false
            })
        })
        .collect();

    let mut body = json!({
        "model": cfg.model,
        "input": input_items,
        "stream": true,
        "store": false,
        "max_output_tokens": cfg.max_output_tokens,
        "prompt_cache_key": session_id
    });
    if !resp_tools.is_empty() {
        body["tools"] = Value::Array(resp_tools);
    }
    if !cfg.reasoning_effort.is_empty() && cfg.model.contains("gpt-6") {
        body["reasoning"] = json!({ "effort": cfg.reasoning_effort });
        body["include"] = json!(["reasoning.encrypted_content"]);
    }

    let url = format!("{}/responses", cfg.base_url);
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {}", cfg.api_key))
        .header("Content-Type", "application/json")
        .header("Accept", "text/event-stream")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("OpenAI Responses request failed: {}", e))?;

    let status = resp.status();
    if !status.is_success() {
        let err_text = resp.text().await.unwrap_or_default();
        let snippet: String = err_text.chars().take(600).collect();
        return Err(format!("OpenAI Responses HTTP {}: {}", status.as_u16(), snippet));
    }

    let mut response_id = String::new();
    let mut response_model = cfg.model.clone();
    let mut streamed_text = String::new();
    let mut tool_calls: Vec<FunctionCallItem> = Vec::new();
    let mut raw_output_items: Vec<Value> = Vec::new();
    let mut terminal_output: Option<Vec<Value>> = None;
    let mut usage = ResponseUsage::default();

    let mut byte_stream = resp.bytes_stream();
    let mut buf = String::new();

    while let Some(chunk_res) = byte_stream.next().await {
        let chunk = chunk_res.map_err(|e| format!("SSE stream read error: {}", e))?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(nl_pos) = buf.find('\n') {
            let line = buf[..nl_pos].trim().to_string();
            buf.drain(..=nl_pos);

            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let Ok(ev) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            let ev_type = ev["type"].as_str().unwrap_or("");

            match ev_type {
                "response.created" => {
                    if let Some(id) = ev["response"]["id"].as_str() {
                        response_id = id.to_string();
                    }
                    if let Some(m) = ev["response"]["model"].as_str() {
                        response_model = m.to_string();
                    }
                }
                "response.output_text.delta" => {
                    if let Some(d) = ev["delta"].as_str() {
                        if !d.is_empty() {
                            streamed_text.push_str(d);
                            on_delta(ResponsesStreamDelta::Text(d.to_string()))?;
                        }
                    }
                }
                "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                    if let Some(d) = ev["delta"].as_str() {
                        if !d.is_empty() {
                            on_delta(ResponsesStreamDelta::Thinking(d.to_string()))?;
                        }
                    }
                }
                "response.function_call_arguments.delta" => {
                    on_delta(ResponsesStreamDelta::ToolArgsProgress)?;
                }
                "response.output_item.done" => {
                    if let Some(item) = ev.get("item") {
                        raw_output_items.push(item.clone());
                        if item["type"].as_str() == Some("function_call") {
                            let call_id = item["call_id"]
                                .as_str()
                                .or_else(|| item["id"].as_str())
                                .unwrap_or("")
                                .to_string();
                            let name = item["name"].as_str().unwrap_or("").to_string();
                            let arguments = item["arguments"].as_str().unwrap_or("{}").to_string();
                            if !name.is_empty() {
                                tool_calls.push(FunctionCallItem {
                                    call_id,
                                    name,
                                    arguments,
                                });
                            }
                        }
                    }
                }
                "response.completed" => {
                    if let Some(r) = ev.get("response") {
                        if let Some(id) = r["id"].as_str() {
                            response_id = id.to_string();
                        }
                        if let Some(m) = r["model"].as_str() {
                            response_model = m.to_string();
                        }
                        if let Some(u) = r.get("usage") {
                            usage.input_tokens = u["input_tokens"].as_u64().unwrap_or(0);
                            usage.output_tokens = u["output_tokens"].as_u64().unwrap_or(0);
                            usage.total_tokens = u["total_tokens"].as_u64().unwrap_or(0);
                            usage.cached_tokens = u["input_tokens_details"]["cached_tokens"]
                                .as_u64()
                                .unwrap_or(0);
                            usage.estimated_cost_usd = estimate_cost_usd(&response_model, &usage);
                        }
                        if let Some(arr) = r["output"].as_array() {
                            terminal_output = Some(arr.clone());
                        }
                    }
                }
                "response.failed" | "error" => {
                    let msg = ev["error"]["message"]
                        .as_str()
                        .or_else(|| ev["response"]["error"]["message"].as_str())
                        .or_else(|| ev["message"].as_str())
                        .unwrap_or("unknown OpenAI Responses error");
                    return Err(format!("OpenAI Responses stream error: {}", msg));
                }
                _ => {}
            }
        }
    }

    // Prefer terminal `response.output` when present (contains complete reasoning encrypted_content + items)
    let final_items = terminal_output.unwrap_or(raw_output_items);

    // If tool_calls wasn't populated via output_item.done (e.g. non-standard proxy), extract from final_items
    if tool_calls.is_empty() {
        for item in &final_items {
            if item["type"].as_str() == Some("function_call") {
                let call_id = item["call_id"]
                    .as_str()
                    .or_else(|| item["id"].as_str())
                    .unwrap_or("")
                    .to_string();
                let name = item["name"].as_str().unwrap_or("").to_string();
                let arguments = item["arguments"].as_str().unwrap_or("{}").to_string();
                if !name.is_empty() {
                    tool_calls.push(FunctionCallItem {
                        call_id,
                        name,
                        arguments,
                    });
                }
            }
        }
    }

    // Fallback text extraction if provider sent full text only in completed output
    if streamed_text.is_empty() {
        for item in &final_items {
            if item["type"].as_str() == Some("message") {
                if let Some(parts) = item["content"].as_array() {
                    for p in parts {
                        if let Some(t) = p["text"].as_str() {
                            streamed_text.push_str(t);
                        }
                    }
                }
            }
        }
    }

    Ok(ResponsesTurnOutput {
        response_id,
        model: response_model,
        text: streamed_text,
        tool_calls,
        raw_output_items: final_items,
        usage,
    })
}
