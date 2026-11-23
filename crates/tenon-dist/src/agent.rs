use crate::agent_tools::{AgentToolkit, ToolExecResult};
use crate::openai_responses::{
    OpenAIResponsesConfig, ResponseUsage, ResponsesStreamDelta, stream_openai_responses,
};
use crate::sync_kernel::UniversalSyncKernel;
use serde_json::{Value, json};
use std::io::{self, Write};
use std::sync::Arc;
use tenon_verify::TraceEvent;

pub struct TenonAgentRuntime {
    pub kernel: Arc<UniversalSyncKernel>,
    pub llm_cfg: OpenAIResponsesConfig,
    pub toolkit: AgentToolkit,
    pub client: reqwest::Client,
    pub session_id: String,
    /// OpenAI Responses API replay input items (preserves reasoning encrypted_content + tool history cleanly!)
    pub conversation_items: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct AgentTurnReport {
    pub branch_id: i64,
    pub exec_epoch: i64,
    pub final_text: String,
    pub tool_calls_executed: Vec<ToolExecResult>,
    pub usage: ResponseUsage,
    pub rounds: usize,
}

impl TenonAgentRuntime {
    pub fn new(
        kernel: Arc<UniversalSyncKernel>,
        llm_cfg: OpenAIResponsesConfig,
        toolkit: AgentToolkit,
        session_id: impl Into<String>,
    ) -> Self {
        let sid = session_id.into();
        let sys_prompt = format!(
            "You are Tenon Agent, an interactive coding & systems agent running on the verified Tenon distributed kernel (SQLite WAL partitioned authority + hard exec_epoch fencing + Branch-Aware Turn DAG + Perceus zero-leak runtime).\n\
             Workspace root: {}\n\
             Available tools: shell, read_file, write_file, edit_file, list_dir, grep_search.\n\
             Be concise, inspect real files/commands before making claims, and execute requested edits/commands directly.",
            toolkit.workspace_root.display()
        );
        let conversation_items = vec![json!({
            "role": "system",
            "content": [{ "type": "input_text", "text": sys_prompt }]
        })];
        Self {
            kernel,
            llm_cfg,
            toolkit,
            client: reqwest::Client::new(),
            session_id: sid,
            conversation_items,
        }
    }

    /// Execute one complete user turn (possibly spanning multiple LLM + shell/file tool rounds),
    /// strictly fenced by `exec_epoch` at Proxy + Tool + Storage Commit!
    pub async fn run_turn(
        &mut self,
        user_text: &str,
        stream_to_stdout: bool,
    ) -> Result<AgentTurnReport, String> {
        let msg_id = format!(
            "cli_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        );
        self.kernel
            .authority
            .send_message(&self.session_id, &msg_id, user_text)?;
        let slice = self
            .kernel
            .authority
            .claim_slice(&self.session_id, "tenon-cli-agent", false)?
            .ok_or_else(|| "failed to claim fenced execution slice".to_string())?;
        self.kernel.record_trace(TraceEvent::ClaimSlice {
            session_id: slice.session_id.clone(),
            worker_id: slice.worker_id.clone(),
            branch_id: slice.branch_id,
            exec_epoch: slice.exec_epoch,
        });
        self.kernel.publish_snapshot(&self.session_id, None);

        self.conversation_items.push(json!({
            "role": "user",
            "content": [{ "type": "input_text", "text": user_text }]
        }));

        let tools = self.toolkit.tool_specs();
        let mut total_usage = ResponseUsage::default();
        let mut executed_tools = Vec::new();
        let mut final_text = String::new();
        let mut seq = 0u64;
        let mut rounds = 0usize;

        const MAX_ROUNDS: usize = 12;

        loop {
            rounds += 1;
            if rounds > MAX_ROUNDS {
                return Err(format!("agent exceeded max tool rounds ({})", MAX_ROUNDS));
            }

            // Check P0 Steering at every round boundary (hard-fenced!)
            if let Ok(steers) = self.kernel.authority.drain_steering_fenced(&slice) {
                for st in steers {
                    self.conversation_items.push(json!({
                        "role": "user",
                        "content": [{ "type": "input_text", "text": format!("[P0 STEERING]: {}", st) }]
                    }));
                }
            }

            let kernel_ref = self.kernel.clone();
            let slice_ref = slice.clone();
            let mut printed_any = false;

            let turn_out = stream_openai_responses(
                &self.client,
                &self.llm_cfg,
                &self.session_id,
                &self.conversation_items,
                &tools,
                |delta| match delta {
                    ResponsesStreamDelta::Text(tok) => {
                        seq += 1;
                        kernel_ref
                            .proxy_emit_token(&slice_ref, seq, &tok)
                            .map_err(|e| e.to_string())?;
                        if stream_to_stdout {
                            print!("{}", tok);
                            let _ = io::stdout().flush();
                            printed_any = true;
                        }
                        Ok(())
                    }
                    ResponsesStreamDelta::Thinking(_) | ResponsesStreamDelta::ToolArgsProgress => {
                        Ok(())
                    }
                },
            )
            .await?;

            if stream_to_stdout && printed_any {
                println!();
            }

            // Record durable usage/model audit receipt independently in SQLite `receipts`!
            let _ = self.kernel.authority.append_late_receipt(
                &format!("llm_{}_r{}", slice.run_id, rounds),
                &slice.session_id,
                slice.branch_id,
                slice.exec_epoch,
                "llm_completed",
                &format!(
                    "model={} resp_id={} in={} out={} cached={} cost=${:.6}",
                    turn_out.model,
                    turn_out.response_id,
                    turn_out.usage.input_tokens,
                    turn_out.usage.output_tokens,
                    turn_out.usage.cached_tokens,
                    turn_out.usage.estimated_cost_usd
                ),
            );

            total_usage.input_tokens += turn_out.usage.input_tokens;
            total_usage.output_tokens += turn_out.usage.output_tokens;
            total_usage.total_tokens += turn_out.usage.total_tokens;
            total_usage.cached_tokens += turn_out.usage.cached_tokens;
            total_usage.estimated_cost_usd += turn_out.usage.estimated_cost_usd;

            // Append raw assistant output items (reasoning + message + function_call) for faithful Responses replay!
            for item in &turn_out.raw_output_items {
                self.conversation_items.push(item.clone());
            }
            if !turn_out.text.is_empty() {
                if !final_text.is_empty() {
                    final_text.push('\n');
                }
                final_text.push_str(&turn_out.text);
            }

            if turn_out.tool_calls.is_empty() {
                break;
            }

            // Execute each requested tool call with pre/post Epoch Fencing!
            for tc in &turn_out.tool_calls {
                let cur_epoch = self.kernel.authority.current_epoch(&self.session_id);
                if cur_epoch != slice.exec_epoch {
                    return Err(format!(
                        "FENCING ABORT before tool `{}`: epoch {} != {}",
                        tc.name, slice.exec_epoch, cur_epoch
                    ));
                }
                if stream_to_stdout {
                    let arg_preview: String = tc.arguments.chars().take(120).collect();
                    println!("  [tool] {}({})", tc.name, arg_preview);
                }
                let res = self.toolkit.execute(&tc.name, &tc.arguments).await;
                if stream_to_stdout {
                    println!(
                        "  [tool:{}] {} ({}ms)",
                        if res.ok { "ok" } else { "err" },
                        res.summary,
                        res.duration_ms
                    );
                }
                let _ = self.kernel.authority.append_late_receipt(
                    &format!("tool_{}_{}", slice.run_id, tc.call_id),
                    &slice.session_id,
                    slice.branch_id,
                    slice.exec_epoch,
                    if res.ok { "tool_ok" } else { "tool_err" },
                    &format!("{} => {}", res.tool, res.summary),
                );
                self.conversation_items.push(json!({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": res.output_json
                }));
                executed_tools.push(res);
            }
        }

        // Fenced Turn Commit to SQLite Authority!
        self.kernel
            .authority
            .commit_turn_fenced(&slice, &final_text)
            .map_err(|e| e.to_string())?;
        self.kernel.record_trace(TraceEvent::CommitTurn {
            session_id: slice.session_id.clone(),
            branch_id: slice.branch_id,
            exec_epoch: slice.exec_epoch,
            run_id: slice.run_id.clone(),
        });
        self.kernel
            .publish_snapshot(&self.session_id, Some(&final_text));

        Ok(AgentTurnReport {
            branch_id: slice.branch_id,
            exec_epoch: slice.exec_epoch,
            final_text,
            tool_calls_executed: executed_tools,
            usage: total_usage,
            rounds,
        })
    }

    pub fn switch_model(&mut self, model_or_alias: &str) {
        let (model, effort) = OpenAIResponsesConfig::resolve_model_and_effort(model_or_alias);
        self.llm_cfg.model = model;
        self.llm_cfg.reasoning_effort = effort;
    }

    pub fn reset_conversation(&mut self) {
        self.conversation_items.truncate(1); // keep system prompt only
    }
}
