pub mod authority;
pub mod channels;
pub mod llm;
pub mod sandbox;
pub mod server;
pub mod sync_kernel;
pub mod worker;

use authority::{AuthorityStore, FencingError};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use sync_kernel::{ChannelDeliveryStats, MuxFrame, UniversalSyncKernel};
use tenon_verify::{DistributedAppManifest, TraceEvent, verify_runtime_trace};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualificationSummary {
    pub single_executor_hard_fenced: bool,
    pub stale_proxy_rejections: usize,
    pub stale_storage_commit_rejected: bool,
    pub interrupt_opened_new_branch: i64,
    pub zero_cross_branch_pollution: bool,
    pub late_receipt_preserved_for_audit: usize,
    pub steer_applied_safely: bool,
    pub followup_promoted_and_completed: bool,
    pub zero_db_writes_per_token: bool,
    pub anti_join_zero_duplicate_bubbles: bool,
    pub trace_refinement_passed: bool,
    pub trace_events_verified: usize,
    pub web_coalesce_ms: u64,
    pub channel_stats: Vec<ChannelDeliveryStats>,
}

/// End-to-End Automated Distributed Qualification Suite:
/// Proves ALL core guarantees live on real SQLite + Async Multi-Worker + Fenced Proxy + MuxStream!
pub async fn run_distributed_qualification(
    manifest: DistributedAppManifest,
) -> Result<QualificationSummary, String> {
    let store = AuthorityStore::open(Path::new(":memory:"))?;
    let kernel = Arc::new(UniversalSyncKernel::new(store, manifest));
    let mut mux_rx = kernel.mux_tx.subscribe();

    // 1. Send initial message -> Worker 1 claims Epoch 1 on Branch 1
    kernel.authority.send_message("main", "m1", "initial task")?;
    let slice_w1 = kernel
        .authority
        .claim_slice("main", "worker-1", false)?
        .expect("worker-1 should claim slice");
    kernel.record_trace(TraceEvent::ClaimSlice {
        session_id: slice_w1.session_id.clone(),
        worker_id: slice_w1.worker_id.clone(),
        branch_id: slice_w1.branch_id,
        exec_epoch: slice_w1.exec_epoch,
    });
    assert_eq!(slice_w1.exec_epoch, 1);
    assert_eq!(slice_w1.branch_id, 1);

    // Worker 1 emits first token safely on Epoch 1 (0 DB writes!)
    kernel
        .proxy_emit_token(&slice_w1, 1, "Part1 ")
        .map_err(|e| e.to_string())?;

    // 2. Competing Worker 2 takes over (Epoch advances 1 -> 2)!
    let slice_w2 = kernel
        .authority
        .claim_slice("main", "worker-2", true)?
        .expect("worker-2 takeover");
    kernel.record_trace(TraceEvent::ClaimSlice {
        session_id: slice_w2.session_id.clone(),
        worker_id: slice_w2.worker_id.clone(),
        branch_id: slice_w2.branch_id,
        exec_epoch: slice_w2.exec_epoch,
    });
    assert_eq!(slice_w2.exec_epoch, 2);

    // 3. Stale Worker 1 tries BOTH Proxy emission AND Storage commit -> BOTH MUST BE HARD FENCED!
    let proxy_err = kernel.proxy_emit_token(&slice_w1, 2, "stale token");
    assert!(matches!(proxy_err, Err(FencingError::StaleEpoch { expected: 1, actual: 2 })));
    let commit_err = kernel.authority.commit_turn_fenced(&slice_w1, "stale commit");
    let stale_storage_commit_rejected =
        matches!(commit_err, Err(FencingError::StaleEpoch { expected: 1, actual: 2 }));
    assert!(stale_storage_commit_rejected);
    kernel.record_trace(TraceEvent::StorageCommitRejectedStale {
        session_id: "main".into(),
        worker_epoch: slice_w1.exec_epoch,
        auth_epoch: 2,
    });

    // Late receipt from stale Worker 1 is still preserved in ReceiptPort for audit!
    kernel.authority.append_late_receipt(
        "att_stale_w1",
        "main",
        slice_w1.branch_id,
        slice_w1.exec_epoch,
        "fenced_late_fact",
        "usage=12tok",
    )?;
    kernel.record_trace(TraceEvent::LateReceiptAudit {
        session_id: "main".into(),
        branch_id: slice_w1.branch_id,
        exec_epoch: slice_w1.exec_epoch,
    });

    // 4. Enqueue P0 Steer AND P1 Followup while Worker 2 is executing, then trigger Branch-Aware INTERRUPT!
    kernel.authority.enqueue_steer("main", "old_steer", "old steer to supersede")?;
    kernel.authority.enqueue_followup("main", "keep_follow", "follow up after interrupt")?;

    let int_out = kernel
        .authority
        .interrupt("main", "cmd_int_1", "switched topic")
        .map_err(|e| e.to_string())?;
    kernel.record_trace(TraceEvent::Interrupt {
        session_id: "main".into(),
        new_branch: int_out.new_branch,
        new_epoch: int_out.new_epoch,
    });
    assert_eq!(int_out.new_branch, 2);
    assert_eq!(int_out.superseded_steers, 1);
    assert_eq!(int_out.preserved_followups, 1);

    // Stale Worker 2 (Epoch 2, Branch 1) is now ALSO hard-fenced by Interrupt (Epoch 3, Branch 2)!
    assert!(matches!(
        kernel.proxy_emit_token(&slice_w2, 1, "old branch token"),
        Err(FencingError::StaleEpoch { expected: 2, actual: 3 })
    ));

    // 5. New Worker 3 claims Branch 2, receives a NEW P0 Steer mid-run, finishes, AND promotes P1 Followup!
    let slice_w3 = kernel
        .authority
        .claim_slice("main", "worker-3", false)?
        .expect("worker-3 claims branch 2");
    kernel.record_trace(TraceEvent::ClaimSlice {
        session_id: slice_w3.session_id.clone(),
        worker_id: slice_w3.worker_id.clone(),
        branch_id: slice_w3.branch_id,
        exec_epoch: slice_w3.exec_epoch,
    });
    assert_eq!(slice_w3.branch_id, 2);
    assert_eq!(slice_w3.exec_epoch, 4);

    kernel.authority.enqueue_steer("main", "new_steer", "stay concise")?;
    let committed = kernel.drive_claimed_slice(slice_w3, 1).await?;
    assert!(committed);

    // Run promoted P1 Followup to completion!
    let ran_followup = kernel.run_worker_once("main", "worker-3", 1).await?;
    assert!(ran_followup);

    // 6. Inspect final authoritative view & Anti-Join invariant & channel stats
    let snap_frame = kernel.build_snapshot_frame("main")?;
    let (view, anti_join_ok) = match snap_frame {
        MuxFrame::Snapshot {
            view, live_bubble, ..
        } => (view, live_bubble.is_none()),
        _ => unreachable!(),
    };

    // Check zero cross-branch pollution: NO turn from branch 1 ("initial task") appears in branch 2 history!
    let zero_cross_branch_pollution = view.history.iter().all(|t| t.branch_id == 2)
        && !view.history.iter().any(|t| t.content.contains("initial task"));
    let steer_applied_safely = view
        .history
        .iter()
        .any(|t| t.role == "assistant" && t.content.contains("[STEERED: stay concise]"));
    let followup_promoted_and_completed = view
        .history
        .iter()
        .any(|t| t.role == "assistant" && t.content.contains("follow up after interrupt"));

    // Drain mux frames to confirm streaming worked
    let mut saw_delta = false;
    while let Ok(f) = mux_rx.try_recv() {
        if matches!(f, MuxFrame::Delta { .. }) {
            saw_delta = true;
        }
    }
    assert!(saw_delta);

    let channel_stats: Vec<ChannelDeliveryStats> = kernel
        .channel_stats
        .lock()
        .unwrap()
        .values()
        .cloned()
        .collect();

    // Verify Dead-Stream Elimination for non-editable channel (Email)
    for cs in &channel_stats {
        if cs.dead_stream_eliminated {
            assert_eq!(
                cs.token_events_received, 0,
                "dead-stream-eliminated channel `{}` must receive 0 token events!",
                cs.channel_id
            );
        }
    }

    // Verify Runtime Trace Refinement against formal protocol invariants!
    let trace_rep = verify_runtime_trace(&kernel.snapshot_trace());
    if !trace_rep.ok {
        return Err(format!(
            "runtime trace refinement failed: {:?}",
            trace_rep.violations
        ));
    }

    Ok(QualificationSummary {
        single_executor_hard_fenced: true,
        stale_proxy_rejections: *kernel.fenced_proxy_rejections.lock().unwrap(),
        stale_storage_commit_rejected,
        interrupt_opened_new_branch: int_out.new_branch,
        zero_cross_branch_pollution,
        late_receipt_preserved_for_audit: view.late_audit_receipts,
        steer_applied_safely,
        followup_promoted_and_completed,
        zero_db_writes_per_token: *kernel.db_token_writes.lock().unwrap() == 0,
        anti_join_zero_duplicate_bubbles: anti_join_ok,
        trace_refinement_passed: trace_rep.ok,
        trace_events_verified: trace_rep.events_checked,
        web_coalesce_ms: kernel.web_coalesce_ms(),
        channel_stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenon_verify::{
        ResolvedStreamSpec, ResolvedSubscriberSpec, ResolvedViewSpec, SyncArchetypeKind,
    };

    #[tokio::test]
    async fn full_distributed_qualification_passes() {
        let manifest = DistributedAppManifest {
            partitioned_entities: vec!["AgentSession".into()],
            streams: vec![ResolvedStreamSpec {
                name: "ActiveTurnStream".into(),
                durability: "Volatile".into(),
                db_writes_per_delta: 0,
                slow_consumer_policy: "DropAndResync".into(),
            }],
            views: vec![ResolvedViewSpec {
                name: "SessionMsgList".into(),
                sync_id: "session.msg_list".into(),
                archetype: SyncArchetypeKind::FencedIncremental,
                max_items: 50,
                max_payload_bytes: 16 * 1024 * 1024,
                max_frame_bytes: 4 * 1024 * 1024,
                journal_budget_bytes: Some(16 * 1024 * 1024),
                bounded_reset: true,
                uses_anti_join_overlay: true,
            }],
            subscribers: vec![
                ResolvedSubscriberSpec {
                    name: "SessionChatStore".into(),
                    kind: "client".into(),
                    channel_id: None,
                    bound_view: "SessionMsgList".into(),
                    coalesce_ms: Some(33),
                    throttle_ms: None,
                    split_bytes: None,
                    debounce_cards_ms: Some(100),
                    dead_stream_eliminated: false,
                    persist_overlay_forbidden: true,
                },
                ResolvedSubscriberSpec {
                    name: "OutboundChannelSink_telegram".into(),
                    kind: "channel".into(),
                    channel_id: Some("telegram".into()),
                    bound_view: "SessionMsgList".into(),
                    coalesce_ms: None,
                    throttle_ms: Some(1200),
                    split_bytes: Some(4096),
                    debounce_cards_ms: None,
                    dead_stream_eliminated: false,
                    persist_overlay_forbidden: true,
                },
                ResolvedSubscriberSpec {
                    name: "OutboundChannelSink_email".into(),
                    kind: "channel".into(),
                    channel_id: Some("email".into()),
                    bound_view: "SessionMsgList".into(),
                    coalesce_ms: None,
                    throttle_ms: None,
                    split_bytes: None,
                    debounce_cards_ms: None,
                    dead_stream_eliminated: true,
                    persist_overlay_forbidden: true,
                },
            ],
        };
        let summary = run_distributed_qualification(manifest).await.unwrap();
        assert!(summary.single_executor_hard_fenced);
        assert!(summary.stale_proxy_rejections >= 2);
        assert!(summary.zero_cross_branch_pollution);
        assert!(summary.steer_applied_safely);
        assert!(summary.followup_promoted_and_completed);
        assert!(summary.zero_db_writes_per_token);
        assert!(summary.anti_join_zero_duplicate_bubbles);
        assert!(summary.trace_refinement_passed);
        assert!(summary.trace_events_verified >= 10);
    }

    #[test]
    fn fabric_sandbox_and_journal_410_reset_work() {
        let sb = sandbox::run_fenced_sandbox(
            "main",
            2,
            "check sandbox",
            std::time::Duration::from_secs(3),
        )
        .unwrap();
        assert_eq!(sb.exit_code, 0);
        assert!(sb.stdout.contains("epoch=2"));
        assert!(sb.child_pid > 0);

        let store = AuthorityStore::open(Path::new(":memory:")).unwrap();
        store.send_message("main", "m1", "first").unwrap();
        let s1 = store.claim_slice("main", "w1", false).unwrap().unwrap();
        store.commit_turn_fenced(&s1, "reply1").unwrap();
        store.send_message("main", "m2", "second").unwrap();
        let s2 = store.claim_slice("main", "w1", false).unwrap().unwrap();
        store.commit_turn_fenced(&s2, "reply2").unwrap();

        let (floor, head) = store.gc_journal_floor("main", 1).unwrap();
        assert!(floor >= 2 && head >= floor);
        match store.sync_since("main", 0).unwrap() {
            authority::SyncSinceResult::ResetRequired { floor_seq, .. } => {
                assert_eq!(floor_seq, floor);
            }
            other => panic!("expected 410 ResetRequired, got {:?}", other),
        }
        match store.sync_since("main", floor).unwrap() {
            authority::SyncSinceResult::Incremental { .. } => {}
            other => panic!("expected Incremental catchup, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn live_openai_sse_stream_and_webhook_channels_work_end_to_end() {
        let llm_addr = llm::spawn_mock_openai_sse_server(vec![
            "Tenon ".into(),
            "streams ".into(),
            "verified ".into(),
            "tokens ".into(),
            "safely.".into(),
        ])
        .await
        .unwrap();
        let (wh_addr, wh_events) = channels::spawn_mock_webhook_receiver().await.unwrap();

        let manifest = DistributedAppManifest {
            partitioned_entities: vec!["AgentSession".into()],
            streams: vec![],
            views: vec![ResolvedViewSpec {
                name: "SessionMsgList".into(),
                sync_id: "session.msg_list".into(),
                archetype: SyncArchetypeKind::FencedIncremental,
                max_items: 50,
                max_payload_bytes: 16 * 1024 * 1024,
                max_frame_bytes: 4 * 1024 * 1024,
                journal_budget_bytes: Some(16 * 1024 * 1024),
                bounded_reset: true,
                uses_anti_join_overlay: true,
            }],
            subscribers: vec![
                ResolvedSubscriberSpec {
                    name: "OutboundChannelSink_telegram".into(),
                    kind: "channel".into(),
                    channel_id: Some("telegram".into()),
                    bound_view: "SessionMsgList".into(),
                    coalesce_ms: None,
                    throttle_ms: Some(1200),
                    split_bytes: Some(18), // Force multi-chunk split verification!
                    debounce_cards_ms: None,
                    dead_stream_eliminated: false,
                    persist_overlay_forbidden: true,
                },
                ResolvedSubscriberSpec {
                    name: "OutboundChannelSink_email".into(),
                    kind: "channel".into(),
                    channel_id: Some("email".into()),
                    bound_view: "SessionMsgList".into(),
                    coalesce_ms: None,
                    throttle_ms: None,
                    split_bytes: None,
                    debounce_cards_ms: None,
                    dead_stream_eliminated: true,
                    persist_overlay_forbidden: true,
                },
            ],
        };

        let wh = channels::ChannelWebhookAdapter::new(&wh_addr, manifest.subscribers.clone());
        let store = AuthorityStore::open(Path::new(":memory:")).unwrap();
        let kernel = UniversalSyncKernel::new(store, manifest);
        kernel.authority.send_message("main", "llm_1", "hello llm").unwrap();
        let slice = kernel.authority.claim_slice("main", "w-llm", false).unwrap().unwrap();
        kernel.record_trace(TraceEvent::ClaimSlice {
            session_id: slice.session_id.clone(),
            worker_id: slice.worker_id.clone(),
            branch_id: slice.branch_id,
            exec_epoch: slice.exec_epoch,
        });

        let llm_cfg = llm::LlmConfig {
            base_addr: llm_addr,
            path: "/v1/chat/completions".into(),
            api_key: None,
            model: "tenon-mock".into(),
        };
        let committed = kernel
            .drive_claimed_slice_with_llm(slice, Some(&llm_cfg), Some(&wh), 0)
            .await
            .unwrap();
        assert!(committed);
        assert_eq!(*kernel.db_token_writes.lock().unwrap(), 0);
        assert!(verify_runtime_trace(&kernel.snapshot_trace()).ok);

        let evs = wh_events.lock().unwrap().clone();
        // Email dead-stream elimination: 0 draft_edit events, 1 final_send event!
        let email_drafts = evs
            .iter()
            .filter(|e| e.channel_id == "email" && e.mode == "draft_edit")
            .count();
        let email_finals = evs
            .iter()
            .filter(|e| e.channel_id == "email" && e.mode == "final_send")
            .count();
        assert_eq!(email_drafts, 0);
        assert_eq!(email_finals, 1);
        // Telegram: throttled to 1 draft_edit + chunk-split final_send (37 bytes / 18 = 3 chunks)!
        let tg_drafts = evs
            .iter()
            .filter(|e| e.channel_id == "telegram" && e.mode == "draft_edit")
            .count();
        let tg_final_chunks = evs
            .iter()
            .filter(|e| e.channel_id == "telegram" && e.mode == "final_send")
            .count();
        assert_eq!(tg_drafts, 1);
        assert_eq!(tg_final_chunks, 3);
    }
}
