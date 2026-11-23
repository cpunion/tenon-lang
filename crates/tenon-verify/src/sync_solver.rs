use crate::checker::Diagnostic;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tenon_syntax::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncArchetypeKind {
    SnapshotReplace,
    FencedIncremental,
    VolatilePrefixDelta,
    DigestRefBlob,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedViewSpec {
    pub name: String,
    pub sync_id: String,
    pub archetype: SyncArchetypeKind,
    pub max_items: u64,
    pub max_payload_bytes: u64,
    pub max_frame_bytes: u64,
    pub journal_budget_bytes: Option<u64>,
    pub bounded_reset: bool,
    pub uses_anti_join_overlay: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedStreamSpec {
    pub name: String,
    pub durability: String,
    pub db_writes_per_delta: u64,
    pub slow_consumer_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedSubscriberSpec {
    pub name: String,
    pub kind: String, // "client" | "channel"
    pub channel_id: Option<String>,
    pub bound_view: String,
    pub coalesce_ms: Option<u64>,
    pub throttle_ms: Option<u64>,
    pub split_bytes: Option<u64>,
    pub debounce_cards_ms: Option<u64>,
    pub dead_stream_eliminated: bool,
    pub persist_overlay_forbidden: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedAppManifest {
    pub partitioned_entities: Vec<String>,
    pub streams: Vec<ResolvedStreamSpec>,
    pub views: Vec<ResolvedViewSpec>,
    pub subscribers: Vec<ResolvedSubscriberSpec>,
}

/// Protocol Constraint Solver:
/// Compiles declarative `@sync_view`, `@stream`, `@client_binding`, `@channel_binding`
/// into ONE minimal Universal SyncKernel manifest — rejecting contradictory bounds at compile time!
pub fn solve_distributed_manifest(
    program: &Program,
) -> Result<Option<DistributedAppManifest>, Vec<Diagnostic>> {
    let mut errors = Vec::new();
    let mut partitioned_entities = Vec::new();
    let mut streams = Vec::new();
    let mut views = Vec::new();
    let mut subscribers = Vec::new();

    for item in &program.items {
        match item {
            Item::Record(r) => {
                if r.attrs.iter().any(|a| a.name == "partition") {
                    partitioned_entities.push(r.name.clone());
                }
            }
            Item::Stream(s) => {
                let durability = attr_ident_or_str(&s.attrs, "stream", "durability")
                    .unwrap_or_else(|| "Volatile".into());
                let db_writes = attr_int(&s.attrs, "stream", "db_writes").unwrap_or(0);
                let slow_consumer = attr_ident_or_str(&s.attrs, "stream", "slow_consumer")
                    .unwrap_or_else(|| "DropAndResync".into());
                if durability == "Volatile" && db_writes > 0 {
                    errors.push(Diagnostic {
                        code: "P2010".into(),
                        message: format!(
                            "volatile stream `{}` forbids per-delta DB writes (found db_writes = {})",
                            s.name, db_writes
                        ),
                        span: s.span,
                        notes: vec!["set `db_writes = 0` to keep hot token streams off AuthorityTxn".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
                streams.push(ResolvedStreamSpec {
                    name: s.name.clone(),
                    durability,
                    db_writes_per_delta: db_writes as u64,
                    slow_consumer_policy: slow_consumer,
                });
            }
            Item::View(v) => {
                let sync_id = v
                    .attrs
                    .iter()
                    .find(|a| a.name == "sync_view")
                    .and_then(|a| a.args.first())
                    .and_then(|arg| match &arg.value.kind {
                        ExprKind::Str(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| v.name.clone());

                let durability = prop_str(&v.properties, "durability")
                    .unwrap_or_else(|| "Reconstructible".into());
                let max_items = bound_u64(&v.bounds, "max_items")
                    .or_else(|| bound_u64(&v.bounds, "window"))
                    .unwrap_or(64);
                let max_frame = bound_bytes(&v.bounds, "max_frame").unwrap_or(4 * 1024 * 1024);
                let max_payload = bound_bytes(&v.bounds, "max_payload").unwrap_or(max_frame);
                let journal_budget = bound_bytes(&v.bounds, "journal_budget");
                let offline_recovery = prop_str(&v.bounds, "offline_recovery")
                    .unwrap_or_else(|| "BoundedReset".into());

                // Contradiction check P2014: bounded journal_budget vs NeverReset!
                if journal_budget.is_some() && offline_recovery == "NeverReset" {
                    errors.push(Diagnostic {
                        code: "P2014".into(),
                        message: format!(
                            "unsatisfiable sync protocol constraints on `{}`: `offline_recovery = NeverReset` contradicts bounded `journal_budget`",
                            v.name
                        ),
                        span: v.span,
                        notes: vec![
                            "under arbitrary offline duration, a bounded journal will advance `floor` past stale checkpoints".into(),
                            "change `offline_recovery` to `BoundedReset` (enables 410 reset_required + staging rebuild)".into(),
                        ],
                        counterexample: BTreeMap::new(),
                    });
                }

                let archetype = if durability == "Volatile" {
                    SyncArchetypeKind::VolatilePrefixDelta
                } else if durability == "ImmutableByDigest" {
                    SyncArchetypeKind::DigestRefBlob
                } else if max_payload > max_frame || journal_budget.is_some() {
                    SyncArchetypeKind::FencedIncremental
                } else {
                    SyncArchetypeKind::SnapshotReplace
                };

                let uses_anti_join_overlay = v
                    .body
                    .as_ref()
                    .is_some_and(|b| format!("{:?}", b).contains("anti_join"));

                views.push(ResolvedViewSpec {
                    name: v.name.clone(),
                    sync_id,
                    archetype,
                    max_items,
                    max_payload_bytes: max_payload,
                    max_frame_bytes: max_frame,
                    journal_budget_bytes: journal_budget,
                    bounded_reset: offline_recovery == "BoundedReset",
                    uses_anti_join_overlay,
                });
            }
            Item::Subscriber(sub) => {
                let bound_view = sub.view_name.clone().unwrap_or_default();
                let is_client = sub.attrs.iter().any(|a| a.name == "client_binding");
                let channel_attrs: Vec<&Attr> = sub
                    .attrs
                    .iter()
                    .filter(|a| a.name == "channel_binding")
                    .collect();

                if is_client {
                    let coalesce_ms = find_pacing_call_ms(&sub.pacing, "coalesce_window");
                    let debounce_cards_ms = find_pacing_call_ms(&sub.pacing, "debounce");
                    subscribers.push(ResolvedSubscriberSpec {
                        name: sub.name.clone(),
                        kind: "client".into(),
                        channel_id: None,
                        bound_view: bound_view.clone(),
                        coalesce_ms,
                        throttle_ms: None,
                        split_bytes: None,
                        debounce_cards_ms,
                        dead_stream_eliminated: false,
                        persist_overlay_forbidden: true,
                    });
                }

                for ch_attr in channel_attrs {
                    let ch_id = ch_attr.args.first().and_then(|a| match &a.value.kind {
                        ExprKind::Str(s) => Some(s.clone()),
                        _ => None,
                    });
                    let supports_edit = ch_attr
                        .args
                        .iter()
                        .find(|a| a.key.as_deref() == Some("supports_edit"))
                        .and_then(|a| match &a.value.kind {
                            ExprKind::Bool(b) => Some(*b),
                            _ => None,
                        })
                        .unwrap_or(true);
                    let max_bytes = ch_attr
                        .args
                        .iter()
                        .find(|a| a.key.as_deref() == Some("max_bytes"))
                        .and_then(|a| expr_bytes(&a.value));
                    let rate_ms = ch_attr
                        .args
                        .iter()
                        .find(|a| a.key.as_deref() == Some("rate_limit_ms"))
                        .and_then(|a| match &a.value.kind {
                            ExprKind::Int(n) => Some(*n as u64),
                            ExprKind::UnitValue { value, unit } if unit == "ms" => {
                                Some(*value as u64)
                            }
                            _ => None,
                        });

                    // Dead-Stream Elimination: non-editable channels (e.g. Email) eliminate ActiveTurnStream subscription!
                    subscribers.push(ResolvedSubscriberSpec {
                        name: format!("{}_{}", sub.name, ch_id.as_deref().unwrap_or("ch")),
                        kind: "channel".into(),
                        channel_id: ch_id,
                        bound_view: bound_view.clone(),
                        coalesce_ms: None,
                        throttle_ms: if supports_edit { rate_ms } else { None },
                        split_bytes: max_bytes,
                        debounce_cards_ms: None,
                        dead_stream_eliminated: !supports_edit,
                        persist_overlay_forbidden: true,
                    });
                }
            }
            _ => {}
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    if partitioned_entities.is_empty()
        && streams.is_empty()
        && views.is_empty()
        && subscribers.is_empty()
    {
        return Ok(None);
    }

    Ok(Some(DistributedAppManifest {
        partitioned_entities,
        streams,
        views,
        subscribers,
    }))
}

fn attr_ident_or_str(attrs: &[Attr], attr_name: &str, key: &str) -> Option<String> {
    let a = attrs.iter().find(|x| x.name == attr_name)?;
    let arg = a.args.iter().find(|x| x.key.as_deref() == Some(key))?;
    match &arg.value.kind {
        ExprKind::Ident(s) | ExprKind::Str(s) => Some(s.clone()),
        _ => None,
    }
}

fn attr_int(attrs: &[Attr], attr_name: &str, key: &str) -> Option<i64> {
    let a = attrs.iter().find(|x| x.name == attr_name)?;
    let arg = a.args.iter().find(|x| x.key.as_deref() == Some(key))?;
    match &arg.value.kind {
        ExprKind::Int(n) => Some(*n),
        _ => None,
    }
}

fn prop_str(props: &[(String, Expr)], key: &str) -> Option<String> {
    let (_, expr) = props.iter().find(|(k, _)| k == key)?;
    match &expr.kind {
        ExprKind::Ident(s) | ExprKind::Str(s) => Some(s.clone()),
        _ => None,
    }
}

fn bound_u64(bounds: &[(String, Expr)], key: &str) -> Option<u64> {
    let (_, expr) = bounds.iter().find(|(k, _)| k == key)?;
    match &expr.kind {
        ExprKind::Int(n) => Some(*n as u64),
        ExprKind::UnitValue { value, .. } => Some(*value as u64),
        _ => None,
    }
}

fn bound_bytes(bounds: &[(String, Expr)], key: &str) -> Option<u64> {
    let (_, expr) = bounds.iter().find(|(k, _)| k == key)?;
    expr_bytes(expr)
}

fn expr_bytes(expr: &Expr) -> Option<u64> {
    match &expr.kind {
        ExprKind::Int(n) => Some(*n as u64),
        ExprKind::UnitValue { value, unit } => {
            let v = *value as u64;
            match unit.as_str() {
                "B" => Some(v),
                "KiB" => Some(v * 1024),
                "MiB" => Some(v * 1024 * 1024),
                "GiB" => Some(v * 1024 * 1024 * 1024),
                _ => Some(v),
            }
        }
        ExprKind::Tuple(elems) => elems.iter().find_map(expr_bytes),
        _ => None,
    }
}

fn find_pacing_call_ms(pacing: &[(String, Expr)], fn_name: &str) -> Option<u64> {
    for (_, expr) in pacing {
        if let ExprKind::Call { callee, args } = &expr.kind {
            if matches!(&callee.kind, ExprKind::Ident(s) if s == fn_name) {
                if let Some(first) = args.first() {
                    match &first.value.kind {
                        ExprKind::UnitValue { value, unit } if unit == "ms" => {
                            return Some(*value as u64);
                        }
                        ExprKind::Int(n) => return Some(*n as u64),
                        _ => {}
                    }
                }
            }
        }
    }
    None
}
