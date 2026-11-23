use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecSlice {
    pub session_id: String,
    pub worker_id: String,
    pub conv_gen: i64,
    pub branch_id: i64,
    pub run_id: String,
    pub stream_id: String,
    pub exec_epoch: i64,
    pub prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnRecord {
    pub turn_id: i64,
    pub session_id: String,
    pub conv_gen: i64,
    pub branch_id: i64,
    pub parent_turn_id: Option<i64>,
    pub run_id: String,
    pub role: String,
    pub content: String,
    pub closed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueuedItem {
    pub id: i64,
    pub kind: String, // "steer" | "followup"
    pub client_msg_id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    pub seq: i64,
    pub session_id: String,
    pub event_kind: String,
    pub payload: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum SyncSinceResult {
    #[serde(rename = "ok")]
    Incremental {
        head_seq: i64,
        floor_seq: i64,
        entries: Vec<JournalEntry>,
    },
    #[serde(rename = "reset_required")]
    ResetRequired {
        floor_seq: i64,
        head_seq: i64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStateView {
    pub session_id: String,
    pub conv_gen: i64,
    pub active_branch: i64,
    pub exec_epoch: i64,
    pub phase: String,
    pub active_run_id: Option<String>,
    pub active_stream_id: Option<String>,
    pub history: Vec<TurnRecord>,
    pub queued: Vec<QueuedItem>,
    pub late_audit_receipts: usize,
    pub journal_head_seq: i64,
    pub journal_floor_seq: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptOutcome {
    pub new_branch: i64,
    pub new_epoch: i64,
    pub base_turn_id: Option<i64>,
    pub superseded_steers: usize,
    pub preserved_followups: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FencingError {
    StaleEpoch { expected: i64, actual: i64 },
    SessionIdle,
    Storage(String),
}

impl std::fmt::Display for FencingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FencingError::StaleEpoch { expected, actual } => write!(
                f,
                "FENCING REJECTED: stale exec_epoch={} (current authoritative epoch={})",
                expected, actual
            ),
            FencingError::SessionIdle => write!(f, "cannot interrupt an idle session"),
            FencingError::Storage(s) => write!(f, "storage error: {}", s),
        }
    }
}

/// Partitioned Single-Writer Authority + SQLite Persistence with Hard `exec_epoch` Fencing
/// and Immutable Branch-Aware Turn DAG (`Interrupt` = O(1) new branch from last closed turn!).
#[derive(Clone)]
pub struct AuthorityStore {
    conn: Arc<Mutex<Connection>>,
}

impl AuthorityStore {
    pub fn open(path: &Path) -> Result<Self, String> {
        let conn = if path.to_str() == Some(":memory:") {
            Connection::open_in_memory().map_err(|e| e.to_string())?
        } else {
            Connection::open(path).map_err(|e| e.to_string())?
        };
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                conv_gen INTEGER NOT NULL DEFAULT 1,
                active_branch INTEGER NOT NULL DEFAULT 1,
                exec_epoch INTEGER NOT NULL DEFAULT 0,
                phase TEXT NOT NULL DEFAULT 'idle',
                active_run_id TEXT,
                active_stream_id TEXT,
                active_worker TEXT,
                journal_seq INTEGER NOT NULL DEFAULT 0,
                journal_floor INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS journal_entries (
                seq INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                event_kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(session_id, seq)
            );

            CREATE TABLE IF NOT EXISTS turns (
                turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                conv_gen INTEGER NOT NULL,
                branch_id INTEGER NOT NULL,
                parent_turn_id INTEGER,
                run_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                closed INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS inbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                branch_id INTEGER NOT NULL,
                kind TEXT NOT NULL, -- 'steer' | 'followup'
                client_msg_id TEXT NOT NULL,
                text TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending', -- 'pending' | 'consumed' | 'superseded'
                UNIQUE(session_id, client_msg_id)
            );

            CREATE TABLE IF NOT EXISTS receipts (
                attempt_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                branch_id INTEGER NOT NULL,
                exec_epoch INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                payload TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS commands (
                cmd_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                result_json TEXT NOT NULL
            );
            "#,
        )
        .map_err(|e| e.to_string())?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    pub fn ensure_session(&self, session_id: &str) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO sessions(session_id) VALUES (?1)",
            params![session_id],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn current_epoch(&self, session_id: &str) -> i64 {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT exec_epoch FROM sessions WHERE session_id = ?1",
            params![session_id],
            |r| r.get(0),
        )
        .unwrap_or(0)
    }

    /// Submit a user message to an idle session (or enqueue as followup if busy).
    pub fn send_message(
        &self,
        session_id: &str,
        client_msg_id: &str,
        text: &str,
    ) -> Result<(String, bool), String> {
        self.ensure_session(session_id)?;
        let conn = self.conn.lock().unwrap();
        let (conv_gen, branch_id, phase): (i64, i64, String) = conn
            .query_row(
                "SELECT conv_gen, active_branch, phase FROM sessions WHERE session_id = ?1",
                params![session_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .map_err(|e| e.to_string())?;

        if phase == "idle" {
            let run_id = format!("run_{}", client_msg_id);
            let parent: Option<i64> = conn
                .query_row(
                    "SELECT turn_id FROM turns WHERE session_id = ?1 AND conv_gen = ?2 AND branch_id = ?3 AND closed = 1 ORDER BY turn_id DESC LIMIT 1",
                    params![session_id, conv_gen, branch_id],
                    |r| r.get(0),
                )
                .ok();
            conn.execute(
                "INSERT INTO turns(session_id, conv_gen, branch_id, parent_turn_id, run_id, role, content, closed) VALUES (?1, ?2, ?3, ?4, ?5, 'user', ?6, 0)",
                params![session_id, conv_gen, branch_id, parent, run_id, text],
            )
            .map_err(|e| e.to_string())?;
            conn.execute(
                "UPDATE sessions SET phase = 'runnable', active_run_id = ?2, journal_seq = journal_seq + 1 WHERE session_id = ?1",
                params![session_id, run_id],
            )
            .map_err(|e| e.to_string())?;
            let seq: i64 = conn
                .query_row(
                    "SELECT journal_seq FROM sessions WHERE session_id = ?1",
                    params![session_id],
                    |r| r.get(0),
                )
                .unwrap_or(1);
            let _ = conn.execute(
                "INSERT INTO journal_entries(seq, session_id, event_kind, payload) VALUES (?1, ?2, 'send_message', ?3)",
                params![seq, session_id, text],
            );
            Ok((run_id, true))
        } else {
            conn.execute(
                "INSERT OR IGNORE INTO inbox(session_id, branch_id, kind, client_msg_id, text) VALUES (?1, ?2, 'followup', ?3, ?4)",
                params![session_id, branch_id, client_msg_id, text],
            )
            .map_err(|e| e.to_string())?;
            Ok((format!("queued_{}", client_msg_id), false))
        }
    }

    pub fn enqueue_steer(
        &self,
        session_id: &str,
        client_msg_id: &str,
        text: &str,
    ) -> Result<(), String> {
        self.ensure_session(session_id)?;
        let conn = self.conn.lock().unwrap();
        let branch_id: i64 = conn
            .query_row(
                "SELECT active_branch FROM sessions WHERE session_id = ?1",
                params![session_id],
                |r| r.get(0),
            )
            .map_err(|e| e.to_string())?;
        conn.execute(
            "INSERT OR IGNORE INTO inbox(session_id, branch_id, kind, client_msg_id, text) VALUES (?1, ?2, 'steer', ?3, ?4)",
            params![session_id, branch_id, client_msg_id, text],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn enqueue_followup(
        &self,
        session_id: &str,
        client_msg_id: &str,
        text: &str,
    ) -> Result<(), String> {
        self.ensure_session(session_id)?;
        let conn = self.conn.lock().unwrap();
        let branch_id: i64 = conn
            .query_row(
                "SELECT active_branch FROM sessions WHERE session_id = ?1",
                params![session_id],
                |r| r.get(0),
            )
            .map_err(|e| e.to_string())?;
        conn.execute(
            "INSERT OR IGNORE INTO inbox(session_id, branch_id, kind, client_msg_id, text) VALUES (?1, ?2, 'followup', ?3, ?4)",
            params![session_id, branch_id, client_msg_id, text],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Branch-Aware Interrupt:
    /// 1) Finds last `closed = 1` Turn on current branch (never inherits unclosed partial input!)
    /// 2) Opens a brand-new `branch_id` + advances `exec_epoch` (fencing any active Worker at Storage + Proxy!)
    /// 3) Supersedes old-branch pending `steer`, PRESERVES `followup`!
    pub fn interrupt(
        &self,
        session_id: &str,
        cmd_id: &str,
        new_msg: &str,
    ) -> Result<InterruptOutcome, FencingError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn
            .transaction()
            .map_err(|e| FencingError::Storage(e.to_string()))?;

        // Idempotent replay by cmd_id
        if let Ok(cached) = tx.query_row(
            "SELECT result_json FROM commands WHERE cmd_id = ?1",
            params![cmd_id],
            |r| r.get::<_, String>(0),
        ) {
            if let Ok(out) = serde_json::from_str::<InterruptOutcome>(&cached) {
                return Ok(out);
            }
        }

        let (conv_gen, old_branch, old_epoch, phase): (i64, i64, i64, String) = tx
            .query_row(
                "SELECT conv_gen, active_branch, exec_epoch, phase FROM sessions WHERE session_id = ?1",
                params![session_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
            )
            .map_err(|e| FencingError::Storage(e.to_string()))?;

        if phase == "idle" {
            return Err(FencingError::SessionIdle);
        }

        let base_turn: Option<i64> = tx
            .query_row(
                "SELECT turn_id FROM turns WHERE session_id = ?1 AND conv_gen = ?2 AND branch_id = ?3 AND closed = 1 ORDER BY turn_id DESC LIMIT 1",
                params![session_id, conv_gen, old_branch],
                |r| r.get(0),
            )
            .ok();

        let new_branch = old_branch + 1;
        let new_epoch = old_epoch + 1;
        let new_run_id = format!("run_int_{}", cmd_id);

        // Supersede ONLY old-branch pending steering; keep followups intact!
        let superseded_steers = tx
            .execute(
                "UPDATE inbox SET state = 'superseded' WHERE session_id = ?1 AND kind = 'steer' AND state = 'pending'",
                params![session_id],
            )
            .map_err(|e| FencingError::Storage(e.to_string()))?;

        let preserved_followups: i64 = tx
            .query_row(
                "SELECT COUNT(*) FROM inbox WHERE session_id = ?1 AND kind = 'followup' AND state = 'pending'",
                params![session_id],
                |r| r.get(0),
            )
            .unwrap_or(0);

        // Copy closed history prefix onto new branch lineage (lightweight reference chain)
        if let Some(bt) = base_turn {
            tx.execute(
                "INSERT INTO turns(session_id, conv_gen, branch_id, parent_turn_id, run_id, role, content, closed)
                 SELECT session_id, conv_gen, ?4, parent_turn_id, run_id, role, content, 1
                 FROM turns WHERE session_id = ?1 AND conv_gen = ?2 AND branch_id = ?3 AND closed = 1 AND turn_id <= ?5 ORDER BY turn_id ASC",
                params![session_id, conv_gen, old_branch, new_branch, bt],
            )
            .map_err(|e| FencingError::Storage(e.to_string()))?;
        }

        tx.execute(
            "INSERT INTO turns(session_id, conv_gen, branch_id, parent_turn_id, run_id, role, content, closed) VALUES (?1, ?2, ?3, ?4, ?5, 'user', ?6, 0)",
            params![session_id, conv_gen, new_branch, base_turn, new_run_id, new_msg],
        )
        .map_err(|e| FencingError::Storage(e.to_string()))?;

        tx.execute(
            "UPDATE sessions SET active_branch = ?2, exec_epoch = ?3, phase = 'runnable', active_run_id = ?4, active_stream_id = NULL, active_worker = NULL WHERE session_id = ?1",
            params![session_id, new_branch, new_epoch, new_run_id],
        )
        .map_err(|e| FencingError::Storage(e.to_string()))?;

        let out = InterruptOutcome {
            new_branch,
            new_epoch,
            base_turn_id: base_turn,
            superseded_steers,
            preserved_followups: preserved_followups as usize,
        };
        let out_json = serde_json::to_string(&out).unwrap();
        tx.execute(
            "INSERT INTO commands(cmd_id, session_id, result_json) VALUES (?1, ?2, ?3)",
            params![cmd_id, session_id, out_json],
        )
        .map_err(|e| FencingError::Storage(e.to_string()))?;
        tx.commit()
            .map_err(|e| FencingError::Storage(e.to_string()))?;
        Ok(out)
    }

    /// Claim (or take over) an execution slice: atomically advances `exec_epoch` so ANY prior worker is hard-fenced!
    pub fn claim_slice(
        &self,
        session_id: &str,
        worker_id: &str,
        force_takeover: bool,
    ) -> Result<Option<ExecSlice>, String> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction().map_err(|e| e.to_string())?;
        let (conv_gen, branch_id, old_epoch, phase, run_id): (i64, i64, i64, String, Option<String>) =
            tx.query_row(
                "SELECT conv_gen, active_branch, exec_epoch, phase, active_run_id FROM sessions WHERE session_id = ?1",
                params![session_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )
            .map_err(|e| e.to_string())?;

        if phase != "runnable" && !(force_takeover && phase == "executing") {
            return Ok(None);
        }

        let new_epoch = old_epoch + 1;
        let run_id = run_id.unwrap_or_else(|| format!("run_e{}", new_epoch));
        let stream_id = format!("stream_{}_b{}_e{}", run_id, branch_id, new_epoch);

        let prompt: String = tx
            .query_row(
                "SELECT content FROM turns WHERE session_id = ?1 AND branch_id = ?2 AND role = 'user' ORDER BY turn_id DESC LIMIT 1",
                params![session_id, branch_id],
                |r| r.get(0),
            )
            .unwrap_or_else(|_| "continue".into());

        tx.execute(
            "UPDATE sessions SET exec_epoch = ?2, phase = 'executing', active_stream_id = ?3, active_worker = ?4 WHERE session_id = ?1",
            params![session_id, new_epoch, stream_id, worker_id],
        )
        .map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;

        Ok(Some(ExecSlice {
            session_id: session_id.to_string(),
            worker_id: worker_id.to_string(),
            conv_gen,
            branch_id,
            run_id,
            stream_id,
            exec_epoch: new_epoch,
            prompt,
        }))
    }

    /// Consume pending P0 steering messages at a safe point (only if worker still holds current `exec_epoch`!).
    pub fn drain_steering_fenced(&self, slice: &ExecSlice) -> Result<Vec<String>, FencingError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn
            .transaction()
            .map_err(|e| FencingError::Storage(e.to_string()))?;
        let actual: i64 = tx
            .query_row(
                "SELECT exec_epoch FROM sessions WHERE session_id = ?1",
                params![slice.session_id],
                |r| r.get(0),
            )
            .map_err(|e| FencingError::Storage(e.to_string()))?;
        if actual != slice.exec_epoch {
            return Err(FencingError::StaleEpoch {
                expected: slice.exec_epoch,
                actual,
            });
        }
        let mut stmt = tx
            .prepare("SELECT id, text FROM inbox WHERE session_id = ?1 AND branch_id = ?2 AND kind = 'steer' AND state = 'pending' ORDER BY id ASC")
            .map_err(|e| FencingError::Storage(e.to_string()))?;
        let items: Vec<(i64, String)> = stmt
            .query_map(params![slice.session_id, slice.branch_id], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .map_err(|e| FencingError::Storage(e.to_string()))?
            .filter_map(Result::ok)
            .collect();
        drop(stmt);
        for (id, _) in &items {
            tx.execute(
                "UPDATE inbox SET state = 'consumed' WHERE id = ?1",
                params![id],
            )
            .map_err(|e| FencingError::Storage(e.to_string()))?;
        }
        tx.commit()
            .map_err(|e| FencingError::Storage(e.to_string()))?;
        Ok(items.into_iter().map(|(_, t)| t).collect())
    }

    /// Fenced Turn Commit:
    /// Atomically verifies `exec_epoch == slice.exec_epoch` and `active_branch == slice.branch_id`,
    /// closes the user turn, appends the committed assistant turn, and promotes any pending P1 followup!
    pub fn commit_turn_fenced(
        &self,
        slice: &ExecSlice,
        assistant_text: &str,
    ) -> Result<Option<String>, FencingError> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn
            .transaction()
            .map_err(|e| FencingError::Storage(e.to_string()))?;

        let (actual_epoch, actual_branch): (i64, i64) = tx
            .query_row(
                "SELECT exec_epoch, active_branch FROM sessions WHERE session_id = ?1",
                params![slice.session_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(|e| FencingError::Storage(e.to_string()))?;

        if actual_epoch != slice.exec_epoch || actual_branch != slice.branch_id {
            return Err(FencingError::StaleEpoch {
                expected: slice.exec_epoch,
                actual: actual_epoch,
            });
        }

        // Mark user turn closed and append committed assistant turn
        tx.execute(
            "UPDATE turns SET closed = 1 WHERE session_id = ?1 AND branch_id = ?2 AND run_id = ?3",
            params![slice.session_id, slice.branch_id, slice.run_id],
        )
        .map_err(|e| FencingError::Storage(e.to_string()))?;

        let parent: Option<i64> = tx
            .query_row(
                "SELECT turn_id FROM turns WHERE session_id = ?1 AND branch_id = ?2 ORDER BY turn_id DESC LIMIT 1",
                params![slice.session_id, slice.branch_id],
                |r| r.get(0),
            )
            .ok();

        tx.execute(
            "INSERT INTO turns(session_id, conv_gen, branch_id, parent_turn_id, run_id, role, content, closed) VALUES (?1, ?2, ?3, ?4, ?5, 'assistant', ?6, 1)",
            params![
                slice.session_id,
                slice.conv_gen,
                slice.branch_id,
                parent,
                slice.run_id,
                assistant_text
            ],
        )
        .map_err(|e| FencingError::Storage(e.to_string()))?;

        // Check if a P1 Followup is waiting; if so, promote it into a fresh runnable Run on the current branch!
        let next_followup: Option<(i64, String, String)> = tx
            .query_row(
                "SELECT id, client_msg_id, text FROM inbox WHERE session_id = ?1 AND kind = 'followup' AND state = 'pending' ORDER BY id ASC LIMIT 1",
                params![slice.session_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .ok();

        let promoted_run = if let Some((fid, cmsg, ftext)) = next_followup {
            let next_run = format!("run_follow_{}", cmsg);
            tx.execute(
                "UPDATE inbox SET state = 'consumed' WHERE id = ?1",
                params![fid],
            )
            .map_err(|e| FencingError::Storage(e.to_string()))?;
            let p2: Option<i64> = tx
                .query_row(
                    "SELECT turn_id FROM turns WHERE session_id = ?1 AND branch_id = ?2 ORDER BY turn_id DESC LIMIT 1",
                    params![slice.session_id, slice.branch_id],
                    |r| r.get(0),
                )
                .ok();
            tx.execute(
                "INSERT INTO turns(session_id, conv_gen, branch_id, parent_turn_id, run_id, role, content, closed) VALUES (?1, ?2, ?3, ?4, ?5, 'user', ?6, 0)",
                params![
                    slice.session_id,
                    slice.conv_gen,
                    slice.branch_id,
                    p2,
                    next_run,
                    ftext
                ],
            )
            .map_err(|e| FencingError::Storage(e.to_string()))?;
            tx.execute(
                "UPDATE sessions SET phase = 'runnable', active_run_id = ?2, active_stream_id = NULL, active_worker = NULL WHERE session_id = ?1",
                params![slice.session_id, next_run],
            )
            .map_err(|e| FencingError::Storage(e.to_string()))?;
            Some(next_run)
        } else {
            tx.execute(
                "UPDATE sessions SET phase = 'idle', active_run_id = NULL, active_stream_id = NULL, active_worker = NULL, journal_seq = journal_seq + 1 WHERE session_id = ?1",
                params![slice.session_id],
            )
            .map_err(|e| FencingError::Storage(e.to_string()))?;
            None
        };

        let jseq: i64 = tx
            .query_row(
                "SELECT journal_seq FROM sessions WHERE session_id = ?1",
                params![slice.session_id],
                |r| r.get(0),
            )
            .unwrap_or(1);
        let _ = tx.execute(
            "INSERT OR REPLACE INTO journal_entries(seq, session_id, event_kind, payload) VALUES (?1, ?2, 'commit_turn', ?3)",
            params![jseq, slice.session_id, assistant_text],
        );

        tx.commit()
            .map_err(|e| FencingError::Storage(e.to_string()))?;
        Ok(promoted_run)
    }

    /// Bounded Journal GC: deletes old incremental journal entries and advances `journal_floor`,
    /// causing stale offline checkpoints (`since_seq < journal_floor`) to receive `410 reset_required`!
    pub fn gc_journal_floor(
        &self,
        session_id: &str,
        retain_last: i64,
    ) -> Result<(i64, i64), String> {
        self.ensure_session(session_id)?;
        let conn = self.conn.lock().unwrap();
        let (head, cur_floor): (i64, i64) = conn
            .query_row(
                "SELECT journal_seq, journal_floor FROM sessions WHERE session_id = ?1",
                params![session_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(|e| e.to_string())?;
        let new_floor = cur_floor.max((head - retain_last).max(1));
        conn.execute(
            "DELETE FROM journal_entries WHERE session_id = ?1 AND seq < ?2",
            params![session_id, new_floor],
        )
        .map_err(|e| e.to_string())?;
        conn.execute(
            "UPDATE sessions SET journal_floor = ?2 WHERE session_id = ?1",
            params![session_id, new_floor],
        )
        .map_err(|e| e.to_string())?;
        Ok((new_floor, head))
    }

    /// Incremental catch-up query (`GET /api/sessions/<id>/sync?since_seq=N`):
    /// Enforces `NoSilentGap`: returns `ResetRequired` (HTTP 410) whenever `since_seq < journal_floor`!
    pub fn sync_since(&self, session_id: &str, since_seq: i64) -> Result<SyncSinceResult, String> {
        self.ensure_session(session_id)?;
        let conn = self.conn.lock().unwrap();
        let (head_seq, floor_seq): (i64, i64) = conn
            .query_row(
                "SELECT journal_seq, journal_floor FROM sessions WHERE session_id = ?1",
                params![session_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(|e| e.to_string())?;
        if since_seq < floor_seq {
            return Ok(SyncSinceResult::ResetRequired {
                floor_seq,
                head_seq,
            });
        }
        let mut stmt = conn
            .prepare("SELECT seq, session_id, event_kind, payload FROM journal_entries WHERE session_id = ?1 AND seq > ?2 ORDER BY seq ASC")
            .map_err(|e| e.to_string())?;
        let entries: Vec<JournalEntry> = stmt
            .query_map(params![session_id, since_seq], |r| {
                Ok(JournalEntry {
                    seq: r.get(0)?,
                    session_id: r.get(1)?,
                    event_kind: r.get(2)?,
                    payload: r.get(3)?,
                })
            })
            .map_err(|e| e.to_string())?
            .filter_map(Result::ok)
            .collect();
        Ok(SyncSinceResult::Incremental {
            head_seq,
            floor_seq,
            entries,
        })
    }

    /// ReceiptPort.Append:
    /// Always accepts immutable external facts from stale epochs / old branches for billing/audit,
    /// NEVER mutating the active branch's `msg_list` or reviving cancelled intents!
    pub fn append_late_receipt(
        &self,
        attempt_id: &str,
        session_id: &str,
        branch_id: i64,
        exec_epoch: i64,
        outcome: &str,
        payload: &str,
    ) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO receipts(attempt_id, session_id, branch_id, exec_epoch, outcome, payload) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![attempt_id, session_id, branch_id, exec_epoch, outcome, payload],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Read authoritative projection along the active `(conv_gen, active_branch)` chain!
    pub fn read_session_view(&self, session_id: &str) -> Result<SessionStateView, String> {
        self.ensure_session(session_id)?;
        let conn = self.conn.lock().unwrap();
        let (conv_gen, active_branch, exec_epoch, phase, active_run_id, active_stream_id, journal_head_seq, journal_floor_seq): (
            i64,
            i64,
            i64,
            String,
            Option<String>,
            Option<String>,
            i64,
            i64,
        ) = conn
            .query_row(
                "SELECT conv_gen, active_branch, exec_epoch, phase, active_run_id, active_stream_id, journal_seq, journal_floor FROM sessions WHERE session_id = ?1",
                params![session_id],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                        r.get(7)?,
                    ))
                },
            )
            .map_err(|e| e.to_string())?;

        let mut stmt = conn
            .prepare("SELECT turn_id, session_id, conv_gen, branch_id, parent_turn_id, run_id, role, content, closed FROM turns WHERE session_id = ?1 AND conv_gen = ?2 AND branch_id = ?3 ORDER BY turn_id ASC")
            .map_err(|e| e.to_string())?;
        let history: Vec<TurnRecord> = stmt
            .query_map(params![session_id, conv_gen, active_branch], |r| {
                Ok(TurnRecord {
                    turn_id: r.get(0)?,
                    session_id: r.get(1)?,
                    conv_gen: r.get(2)?,
                    branch_id: r.get(3)?,
                    parent_turn_id: r.get(4)?,
                    run_id: r.get(5)?,
                    role: r.get(6)?,
                    content: r.get(7)?,
                    closed: r.get::<_, i64>(8)? != 0,
                })
            })
            .map_err(|e| e.to_string())?
            .filter_map(Result::ok)
            .collect();

        let mut qstmt = conn
            .prepare("SELECT id, kind, client_msg_id, text FROM inbox WHERE session_id = ?1 AND state = 'pending' ORDER BY id ASC")
            .map_err(|e| e.to_string())?;
        let queued: Vec<QueuedItem> = qstmt
            .query_map(params![session_id], |r| {
                Ok(QueuedItem {
                    id: r.get(0)?,
                    kind: r.get(1)?,
                    client_msg_id: r.get(2)?,
                    text: r.get(3)?,
                })
            })
            .map_err(|e| e.to_string())?
            .filter_map(Result::ok)
            .collect();

        let late_audit_receipts: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM receipts WHERE session_id = ?1",
                params![session_id],
                |r| r.get(0),
            )
            .unwrap_or(0);

        Ok(SessionStateView {
            session_id: session_id.to_string(),
            conv_gen,
            active_branch,
            exec_epoch,
            phase,
            active_run_id,
            active_stream_id,
            history,
            queued,
            late_audit_receipts: late_audit_receipts as usize,
            journal_head_seq,
            journal_floor_seq,
        })
    }
}
