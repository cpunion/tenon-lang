use crate::authority::{AuthorityStore, ExecSlice, FencingError, SessionStateView};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use tenon_verify::DistributedAppManifest;
use tokio::sync::broadcast;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveOverlay {
    pub session_id: String,
    pub conv_gen: i64,
    pub branch_id: i64,
    pub run_id: String,
    pub stream_id: String,
    pub exec_epoch: i64,
    pub through_seq: u64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event")]
pub enum MuxFrame {
    #[serde(rename = "hello")]
    Hello {
        conn_id: String,
        coalesce_ms: u64,
    },
    #[serde(rename = "session.snapshot")]
    Snapshot {
        view: SessionStateView,
        /// Anti-Join computed live_bubble: `None` if `history` already contains `run_id`!
        live_bubble: Option<ActiveOverlay>,
        head_seq: u64,
    },
    #[serde(rename = "session.delta")]
    Delta {
        session_id: String,
        branch_id: i64,
        run_id: String,
        stream_id: String,
        seq: u64,
        delta: String,
    },
    #[serde(rename = "head")]
    Head {
        scope: String,
        epoch: i64,
        seq: u64,
    },
    #[serde(rename = "heartbeat")]
    Heartbeat {
        session_id: String,
        stream_id: Option<String>,
        high_water_seq: u64,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChannelDeliveryStats {
    pub channel_id: String,
    pub dead_stream_eliminated: bool,
    pub throttle_ms: Option<u64>,
    pub token_events_received: usize,
    pub outbound_edits_or_sends: usize,
    pub messages: Vec<String>,
}

/// Universal Minimal SyncKernel (1 kernel driven by compiled `DistributedAppManifest`!).
/// Holds:
/// - In-memory `ActiveTurnStream` prefix (strictly 0 DB writes per token!)
/// - Unified `MuxStream` broadcast bus
/// - Fenced LLM/Fabric Effect Proxy
/// - Compiled Channel Subscribers (Telegram trailing-throttle + Email dead-stream-eliminated)
#[derive(Clone)]
pub struct UniversalSyncKernel {
    pub authority: AuthorityStore,
    pub manifest: DistributedAppManifest,
    pub mux_tx: broadcast::Sender<MuxFrame>,
    overlays: Arc<Mutex<BTreeMap<String, ActiveOverlay>>>,
    head_seq: Arc<Mutex<u64>>,
    pub channel_stats: Arc<Mutex<BTreeMap<String, ChannelDeliveryStats>>>,
    pub db_token_writes: Arc<Mutex<usize>>,
    pub fenced_proxy_rejections: Arc<Mutex<usize>>,
}

impl UniversalSyncKernel {
    pub fn new(authority: AuthorityStore, manifest: DistributedAppManifest) -> Self {
        let (mux_tx, _) = broadcast::channel(512);
        let mut ch_map = BTreeMap::new();
        for sub in &manifest.subscribers {
            if sub.kind == "channel" {
                let cid = sub.channel_id.clone().unwrap_or_else(|| sub.name.clone());
                ch_map.insert(
                    cid.clone(),
                    ChannelDeliveryStats {
                        channel_id: cid,
                        dead_stream_eliminated: sub.dead_stream_eliminated,
                        throttle_ms: sub.throttle_ms,
                        token_events_received: 0,
                        outbound_edits_or_sends: 0,
                        messages: Vec::new(),
                    },
                );
            }
        }
        Self {
            authority,
            manifest,
            mux_tx,
            overlays: Arc::new(Mutex::new(BTreeMap::new())),
            head_seq: Arc::new(Mutex::new(1)),
            channel_stats: Arc::new(Mutex::new(ch_map)),
            db_token_writes: Arc::new(Mutex::new(0)),
            fenced_proxy_rejections: Arc::new(Mutex::new(0)),
        }
    }

    pub fn web_coalesce_ms(&self) -> u64 {
        self.manifest
            .subscribers
            .iter()
            .find(|s| s.kind == "client")
            .and_then(|s| s.coalesce_ms)
            .unwrap_or(33)
    }

    /// Fenced Effect Proxy: emits one volatile token delta ONLY IF `slice.exec_epoch == current_epoch`!
    /// Strictly 0 DB writes!
    pub fn proxy_emit_token(
        &self,
        slice: &ExecSlice,
        seq: u64,
        delta: &str,
    ) -> Result<(), FencingError> {
        let current = self.authority.current_epoch(&slice.session_id);
        if current != slice.exec_epoch {
            *self.fenced_proxy_rejections.lock().unwrap() += 1;
            return Err(FencingError::StaleEpoch {
                expected: slice.exec_epoch,
                actual: current,
            });
        }

        // Update volatile in-memory prefix (0 DB writes!)
        {
            let mut ov = self.overlays.lock().unwrap();
            let entry = ov
                .entry(slice.session_id.clone())
                .or_insert_with(|| ActiveOverlay {
                    session_id: slice.session_id.clone(),
                    conv_gen: slice.conv_gen,
                    branch_id: slice.branch_id,
                    run_id: slice.run_id.clone(),
                    stream_id: slice.stream_id.clone(),
                    exec_epoch: slice.exec_epoch,
                    through_seq: 0,
                    text: String::new(),
                });
            if entry.stream_id != slice.stream_id {
                *entry = ActiveOverlay {
                    session_id: slice.session_id.clone(),
                    conv_gen: slice.conv_gen,
                    branch_id: slice.branch_id,
                    run_id: slice.run_id.clone(),
                    stream_id: slice.stream_id.clone(),
                    exec_epoch: slice.exec_epoch,
                    through_seq: 0,
                    text: String::new(),
                };
            }
            entry.through_seq = seq;
            entry.text.push_str(delta);
        }

        // Fan-out to MuxStream (Web/Native clients)
        let _ = self.mux_tx.send(MuxFrame::Delta {
            session_id: slice.session_id.clone(),
            branch_id: slice.branch_id,
            run_id: slice.run_id.clone(),
            stream_id: slice.stream_id.clone(),
            seq,
            delta: delta.to_string(),
        });

        // Dispatch to channels respecting Dead-Stream Elimination!
        let mut ch = self.channel_stats.lock().unwrap();
        for stats in ch.values_mut() {
            if stats.dead_stream_eliminated {
                // Dead-Stream Eliminated (e.g. Email): ZERO token events delivered!
                continue;
            }
            stats.token_events_received += 1;
            // Trailing throttle simulation: emit edit on first token and every Nth token
            if seq == 1 || seq % 4 == 0 {
                stats.outbound_edits_or_sends += 1;
            }
        }
        Ok(())
    }

    /// Compute `SessionMsgList` with Anti-Join overlay handoff:
    /// `live_bubble` is ONLY included if its lineage matches AND `history` does NOT contain `run_id`!
    pub fn build_snapshot_frame(&self, session_id: &str) -> Result<MuxFrame, String> {
        let view = self.authority.read_session_view(session_id)?;
        let live_bubble = {
            let ov = self.overlays.lock().unwrap();
            ov.get(session_id).and_then(|o| {
                let lineage_ok = o.conv_gen == view.conv_gen
                    && o.branch_id == view.active_branch
                    && view.active_stream_id.as_deref() == Some(o.stream_id.as_str());
                // Anti-Join: once `CommittedTurn` with `o.run_id` lands in `history`, overlay vanishes atomically!
                let already_committed = view
                    .history
                    .iter()
                    .any(|t| t.role == "assistant" && t.run_id == o.run_id);
                if lineage_ok && !already_committed {
                    Some(o.clone())
                } else {
                    None
                }
            })
        };
        let seq = *self.head_seq.lock().unwrap();
        Ok(MuxFrame::Snapshot {
            view,
            live_bubble,
            head_seq: seq,
        })
    }

    /// Broadcast authoritative snapshot after state transition (commit / interrupt / steer / followup).
    pub fn publish_snapshot(&self, session_id: &str, finalized_assistant_text: Option<&str>) {
        // Clear stale overlay if committed or branch switched
        if finalized_assistant_text.is_some() {
            self.overlays.lock().unwrap().remove(session_id);
            let mut ch = self.channel_stats.lock().unwrap();
            for stats in ch.values_mut() {
                stats.outbound_edits_or_sends += 1;
                if let Some(txt) = finalized_assistant_text {
                    stats.messages.push(txt.to_string());
                }
            }
        }
        let seq = {
            let mut s = self.head_seq.lock().unwrap();
            *s += 1;
            *s
        };
        if let Ok(frame) = self.build_snapshot_frame(session_id) {
            let epoch = self.authority.current_epoch(session_id);
            let _ = self.mux_tx.send(MuxFrame::Head {
                scope: format!("session:{}", session_id),
                epoch,
                seq,
            });
            let _ = self.mux_tx.send(frame);
        }
    }

    /// Execute a worker run step-by-step via Fenced Proxy, checking P0 Steering at safe points!
    pub async fn run_worker_once(
        &self,
        session_id: &str,
        worker_id: &str,
        token_delay_ms: u64,
    ) -> Result<bool, String> {
        let Some(slice) = self.authority.claim_slice(session_id, worker_id, false)? else {
            return Ok(false);
        };
        self.publish_snapshot(session_id, None);
        self.drive_claimed_slice(slice, token_delay_ms).await?;
        Ok(true)
    }

    pub async fn drive_claimed_slice(
        &self,
        slice: ExecSlice,
        token_delay_ms: u64,
    ) -> Result<bool, String> {
        let base_tokens = [
            "Analyzing ",
            &format!("[{}] ", slice.prompt),
            "on branch#",
            &slice.branch_id.to_string(),
            " (epoch ",
            &slice.exec_epoch.to_string(),
            ")... ",
            "verified ",
            "and ",
            "complete.",
        ];
        let mut full = String::new();
        let mut seq = 0u64;

        for (idx, tok) in base_tokens.iter().enumerate() {
            if token_delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(token_delay_ms)).await;
            }
            // Mid-run safe point: check P0 Steering queue!
            if idx == 4 {
                match self.authority.drain_steering_fenced(&slice) {
                    Ok(steers) => {
                        for st in steers {
                            seq += 1;
                            let note = format!(" [STEERED: {}] ", st);
                            if self.proxy_emit_token(&slice, seq, &note).is_err() {
                                return Ok(false);
                            }
                            full.push_str(&note);
                        }
                    }
                    Err(_) => return Ok(false),
                }
            }
            seq += 1;
            if self.proxy_emit_token(&slice, seq, tok).is_err() {
                // Fenced out by newer Worker takeover or User Interrupt!
                // Record late fact to ReceiptPort independently (audit/billing, ZERO branch pollution!)
                let _ = self.authority.append_late_receipt(
                    &format!("att_{}_e{}", slice.run_id, slice.exec_epoch),
                    &slice.session_id,
                    slice.branch_id,
                    slice.exec_epoch,
                    "aborted_by_fence",
                    &format!("partial_tokens={}", seq - 1),
                );
                return Ok(false);
            }
            full.push_str(tok);
        }

        match self.authority.commit_turn_fenced(&slice, &full) {
            Ok(_promoted_followup) => {
                self.publish_snapshot(&slice.session_id, Some(&full));
                Ok(true)
            }
            Err(_) => {
                let _ = self.authority.append_late_receipt(
                    &format!("att_late_{}_e{}", slice.run_id, slice.exec_epoch),
                    &slice.session_id,
                    slice.branch_id,
                    slice.exec_epoch,
                    "late_commit_fenced",
                    &full,
                );
                Ok(false)
            }
        }
    }
}
