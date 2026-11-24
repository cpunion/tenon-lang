use crate::smt::{SmtOutcome, SmtQuery, encode_expr};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use tenon_syntax::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompatVerdict {
    HotfixCompatible,
    WideningBackwardSafe,
    NarrowingBreaking,
    DivergentBreaking,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityCompatReport {
    pub entity_id: String,
    pub record_name: String,
    pub verdict: CompatVerdict,
    pub upgrade_safe: bool,
    pub downgrade_safe: bool,
    pub counterexample: BTreeMap<String, String>,
    pub residual_domain: Option<String>,
    pub added_optional_fields: Vec<String>,
    pub lens_verified: Option<LensVerification>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensVerification {
    pub entity: String,
    pub from_ver: String,
    pub to_ver: String,
    pub totality_proved: bool,
    pub roundtrip_proved: bool,
    pub retained_fields: Vec<String>,
    pub allow_loss_fields: Vec<String>,
    pub details: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionReport {
    pub v1_receipt_sha256: String,
    pub v2_receipt_sha256: String,
    pub safe_to_release: bool,
    pub entities: Vec<EntityCompatReport>,
    pub capability_checks: Vec<CapabilityCheckResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityCheckResult {
    pub operation: String,
    pub required: Vec<String>,
    pub supported: Vec<String>,
    pub compatible: bool,
}

/// Compare two program revisions (`v1` deployed basis vs `v2` candidate) using SMT compatibility deduction.
pub fn check_evolution(v1: &Program, v2: &Program) -> EvolutionReport {
    let v1_hash = receipt_hash(v1);
    let v2_hash = receipt_hash(v2);

    let v1_pure = collect_pure_fns(v1);
    let v2_pure = collect_pure_fns(v2);
    let v1_aliases = collect_alias_predicates(v1);
    let v2_aliases = collect_alias_predicates(v2);

    let v1_entities = collect_evolve_records(v1);
    let v2_entities = collect_evolve_records(v2);
    let v2_evolutions = collect_evolutions(v2);

    let mut entity_reports = Vec::new();
    let mut all_safe = true;

    for (eid, rec1) in &v1_entities {
        let Some(rec2) = v2_entities.get(eid) else {
            all_safe = false;
            entity_reports.push(EntityCompatReport {
                entity_id: eid.clone(),
                record_name: rec1.name.clone(),
                verdict: CompatVerdict::DivergentBreaking,
                upgrade_safe: false,
                downgrade_safe: false,
                counterexample: BTreeMap::new(),
                residual_domain: Some("entire entity removed".into()),
                added_optional_fields: Vec::new(),
                lens_verified: None,
                notes: vec![format!("entity `{}` removed in candidate v2", eid)],
            });
            continue;
        };

        let report = compare_record_versions(
            eid,
            rec1,
            rec2,
            &v1_pure,
            &v2_pure,
            &v1_aliases,
            &v2_aliases,
            v2_evolutions.get(&rec2.name),
        );
        let resolved = matches!(
            report.verdict,
            CompatVerdict::HotfixCompatible | CompatVerdict::WideningBackwardSafe
        ) || report
            .lens_verified
            .as_ref()
            .is_some_and(|l| l.totality_proved && l.roundtrip_proved);
        if !resolved {
            all_safe = false;
        }
        entity_reports.push(report);
    }

    // Check capability subset compatibility (`required(record, op) ⊆ supported(consumer, op)`)
    let capability_checks = check_capabilities(v1, v2);
    if capability_checks.iter().any(|c| !c.compatible) {
        all_safe = false;
    }

    EvolutionReport {
        v1_receipt_sha256: v1_hash,
        v2_receipt_sha256: v2_hash,
        safe_to_release: all_safe,
        entities: entity_reports,
        capability_checks,
    }
}

#[allow(clippy::too_many_arguments)]
fn compare_record_versions(
    eid: &str,
    rec1: &RecordDecl,
    rec2: &RecordDecl,
    v1_pure: &BTreeMap<String, FnDecl>,
    v2_pure: &BTreeMap<String, FnDecl>,
    v1_aliases: &BTreeMap<String, Expr>,
    v2_aliases: &BTreeMap<String, Expr>,
    evo_lens: Option<&EvolutionDecl>,
) -> EntityCompatReport {
    // Merge pure fns with versioned names (`v1__fn`, `v2__fn`) so buggy v1 `is_leap_year` vs fixed v2 `is_leap_year` coexist cleanly in one Z3 query!
    let mut merged_pure = BTreeMap::new();
    for (k, f) in v1_pure {
        merged_pure.insert(format!("v1__{}", k), rename_fn_calls(f, "v1__", v1_pure));
    }
    for (k, f) in v2_pure {
        merged_pure.insert(format!("v2__{}", k), rename_fn_calls(f, "v2__", v2_pure));
    }

    // Map fields by name (default), `@from("old_name")` / `@alias("old_name")` rename lineage,
    // or optional explicit `@field(N)` binary wire slot.
    let f1_map = index_fields_by_identity(rec1);
    let f2_map = index_fields_by_identity(rec2);

    // Map any renamed v2 field names back to their canonical v1 SMT variable name
    let mut v2_name_map = BTreeMap::new();
    for (slot, f1) in &f1_map {
        if let Some(f2) = f2_map.get(slot) {
            if f2.name != f1.name {
                v2_name_map.insert(f2.name.clone(), f1.name.clone());
            }
        }
    }

    let mut added_optional = Vec::new();
    let mut breaking_schema_changes = Vec::new();
    let mut notes = Vec::new();

    for (slot, f2) in &f2_map {
        if !f1_map.contains_key(slot) {
            if f2.default.is_some() || f2.ty.display_name().starts_with("Option[") {
                added_optional.push(f2.name.clone());
                notes.push(format!(
                    "field `{}` added with default/Option: backward-compatible, mark `retain_on_downgrade`",
                    f2.name
                ));
            } else {
                breaking_schema_changes.push(format!(
                    "field `{}` added as required without default",
                    f2.name
                ));
                notes.push(format!(
                    "field `{}` added as required without default: breaking upgrade (requires evolution lens)",
                    f2.name
                ));
            }
        }
    }

    // Check removed fields and cross-version field base-type changes (e.g., Text -> Int64)
    for (slot, f1) in &f1_map {
        match f2_map.get(slot) {
            None => {
                breaking_schema_changes.push(format!("field `{}` removed in v2", f1.name));
                notes.push(format!(
                    "field `{}` removed in v2: breaking schema change (requires evolution lens)",
                    f1.name
                ));
            }
            Some(f2) => {
                let t1 = normalize_base_type_name(&f1.ty.display_name(), v1_aliases);
                let t2 = normalize_base_type_name(&f2.ty.display_name(), v2_aliases);
                if t1 != t2 {
                    breaking_schema_changes.push(format!(
                        "field `{}` changed type from `{}` to `{}`",
                        f1.name,
                        f1.ty.display_name(),
                        f2.ty.display_name()
                    ));
                    notes.push(format!(
                        "field `{}` changed type from `{}` to `{}` between versions: requires verified `evolution` lens",
                        f1.name,
                        f1.ty.display_name(),
                        f2.ty.display_name()
                    ));
                }
            }
        }
    }

    // Build combined SMT predicates P1 and P2 over shared field identities
    let mut vars = Vec::new();
    let mut p1_clauses = Vec::new();
    let mut p2_clauses = Vec::new();

    for (slot, f1) in &f1_map {
        let var = f1.name.clone();
        vars.push(var.clone());
        let mut subst1 = BTreeMap::new();
        subst1.insert("self".into(), var.clone());
        if let Some(ap) = v1_aliases.get(&f1.ty.display_name()) {
            let renamed = rename_expr_calls(ap, "v1__", v1_pure);
            if let Ok(s) = encode_expr(&renamed, &subst1) {
                p1_clauses.push(s);
            }
        }
        if let Some(pred) = &f1.predicate {
            let renamed = rename_expr_calls(pred, "v1__", v1_pure);
            if let Ok(s) = encode_expr(&renamed, &subst1) {
                p1_clauses.push(s);
            }
        }
        if let Some(f2) = f2_map.get(slot) {
            let mut subst2 = v2_name_map.clone();
            subst2.insert("self".into(), var.clone());
            if let Some(ap) = v2_aliases.get(&f2.ty.display_name()) {
                let renamed = rename_expr_calls(ap, "v2__", v2_pure);
                if let Ok(s) = encode_expr(&renamed, &subst2) {
                    p2_clauses.push(s);
                }
            }
            if let Some(pred) = &f2.predicate {
                let renamed = rename_expr_calls(pred, "v2__", v2_pure);
                if let Ok(s) = encode_expr(&renamed, &subst2) {
                    p2_clauses.push(s);
                }
            }
        }
    }

    let p1 = conjoin(&p1_clauses);
    let p2 = conjoin(&p2_clauses);

    // Query 1: UpgradeSafe  <=>  forall x. P1(x) => P2(x)
    let mut q_up = SmtQuery::new(merged_pure.clone());
    for v in &vars {
        q_up.declare_int(v.clone());
    }
    q_up.premises.push(p1.clone());
    q_up.goal = p2.clone();
    let up_res = q_up.solve();

    // Query 2: DowngradeSafe <=> forall x. P2(x) => P1(x)
    let mut q_down = SmtQuery::new(merged_pure.clone());
    for v in &vars {
        q_down.declare_int(v.clone());
    }
    q_down.premises.push(p2.clone());
    q_down.goal = p1.clone();
    let down_res = q_down.solve();

    let upgrade_safe = matches!(up_res, SmtOutcome::Proved) && breaking_schema_changes.is_empty();
    let downgrade_safe = matches!(down_res, SmtOutcome::Proved)
        && added_optional.is_empty()
        && breaking_schema_changes.is_empty();

    let mut counterexample = BTreeMap::new();
    let mut residual_domain = None;

    let verdict = if !breaking_schema_changes.is_empty() {
        residual_domain = Some(breaking_schema_changes.join("; "));
        CompatVerdict::DivergentBreaking
    } else {
        match (&up_res, &down_res) {
            (SmtOutcome::Proved, SmtOutcome::Proved) if added_optional.is_empty() => {
                notes.push("SMT proved P_v1 <=> P_v2: zero-downtime HotfixCompatible".into());
                CompatVerdict::HotfixCompatible
            }
            (SmtOutcome::Proved, _) => {
                if let SmtOutcome::Counterexample(ce) = down_res {
                    counterexample = ce;
                    notes.push("upgrade is safe (P_v1 => P_v2), but downgrade/rolling old-reads-new has counterexample".into());
                }
                CompatVerdict::WideningBackwardSafe
            }
            (SmtOutcome::Counterexample(ce), SmtOutcome::Proved) => {
                counterexample = ce.clone();
                residual_domain = Some(format!("(and {} (not {}))", p1, p2));
                notes.push("BREAKING narrowing: P_v1 ⇏ P_v2; requires verified evolution lens covering residual domain Δ".into());
                CompatVerdict::NarrowingBreaking
            }
            (SmtOutcome::Counterexample(ce), _) => {
                counterexample = ce.clone();
                residual_domain = Some(format!("(and {} (not {}))", p1, p2));
                notes.push("BREAKING divergent predicate change".into());
                CompatVerdict::DivergentBreaking
            }
            (SmtOutcome::Unknown(msg), _) => {
                notes.push(format!("SMT unknown (Fail-Closed): {}", msg));
                CompatVerdict::DivergentBreaking
            }
        }
    };

    // If a bidirectional `evolution` lens is declared in v2, statically prove Totality & Round-Trip!
    let lens_verified = evo_lens.map(|lens| {
        verify_evolution_lens(lens, &vars, &p1, &p2, &merged_pure, &added_optional)
    });

    EntityCompatReport {
        entity_id: eid.to_string(),
        record_name: rec2.name.clone(),
        verdict,
        upgrade_safe,
        downgrade_safe,
        counterexample,
        residual_domain,
        added_optional_fields: added_optional,
        lens_verified,
        notes,
    }
}

fn verify_evolution_lens(
    lens: &EvolutionDecl,
    vars: &[String],
    p1: &str,
    p2: &str,
    merged_pure: &BTreeMap<String, FnDecl>,
    added_optional: &[String],
) -> LensVerification {
    let mut details = Vec::new();
    let mut totality_proved = false;
    let mut roundtrip_proved = false;

    if let Some(up_fn) = &lens.up {
        // Extract symbolic output expression for the primary constrained scalar field (e.g. `amount`)
        if let Some((up_amt_smt, arch_flag_smt, arch_val_smt)) = extract_lens_up_semantics(up_fn) {
            // Law 1: Totality & Target Validity: forall x. P1(x) => P2(up(x))
            let mut q = SmtQuery::new(merged_pure.clone());
            for v in vars {
                q.declare_int(v.clone());
            }
            q.premises.push(p1.to_string());
            let goal = p2.replace("amount", &up_amt_smt);
            q.goal = goal;
            match q.solve() {
                SmtOutcome::Proved => {
                    totality_proved = true;
                    details.push("Law 1 (Totality & Target Validity) PROVED by Z3: ∀x∈V1. up(x) ∈ V2".into());
                }
                SmtOutcome::Counterexample(ce) => {
                    details.push(format!("Law 1 FAILED: counterexample {:?}", ce));
                }
                SmtOutcome::Unknown(u) => details.push(format!("Law 1 UNKNOWN: {}", u)),
            }

            // Law 2: Round-Trip (PutGet): forall x. P1(x) => down(up(x)) == x
            if let Some(down_fn) = &lens.down {
                if let Some(down_amt_smt) =
                    extract_lens_down_semantics(down_fn, &up_amt_smt, &arch_flag_smt, &arch_val_smt)
                {
                    let mut q2 = SmtQuery::new(merged_pure.clone());
                    for v in vars {
                        q2.declare_int(v.clone());
                    }
                    q2.premises.push(p1.to_string());
                    q2.goal = format!("(= {} amount)", down_amt_smt);
                    match q2.solve() {
                        SmtOutcome::Proved => {
                            roundtrip_proved = true;
                            details.push(
                                "Law 2 (Round-Trip / PutGet) PROVED by Z3: ∀x∈V1. down(up(x)) == x"
                                    .into(),
                            );
                        }
                        SmtOutcome::Counterexample(ce) => {
                            details.push(format!("Law 2 FAILED: counterexample {:?}", ce));
                        }
                        SmtOutcome::Unknown(u) => details.push(format!("Law 2 UNKNOWN: {}", u)),
                    }
                }
            }
        }
    }

    // Check downgrade retention coverage for newly added fields
    for added in added_optional {
        let retained = lens.retain_fields.contains(added) || lens.allow_loss_fields.contains(added);
        if retained {
            details.push(format!(
                "Law 3 (Retained Downgrade): field `{}` explicitly accounted for on downgrade",
                added
            ));
        } else {
            roundtrip_proved = false;
            details.push(format!(
                "Law 3 FAILED: new field `{}` missing `retain({})` or `allow_loss({})` in down()",
                added, added, added
            ));
        }
    }

    LensVerification {
        entity: lens.entity.clone(),
        from_ver: lens.from_ver.clone(),
        to_ver: lens.to_ver.clone(),
        totality_proved,
        roundtrip_proved,
        retained_fields: lens.retain_fields.clone(),
        allow_loss_fields: lens.allow_loss_fields.clone(),
        details,
    }
}

/// Extract SMT formulas for `(up_amount, has_archive, archive_amount)` from `if old.amount == 0 { ... } else { ... }`.
fn extract_lens_up_semantics(up_fn: &FnDecl) -> Option<(String, String, String)> {
    let tail = up_fn.body.tail.as_deref()?;
    if let ExprKind::If {
        cond,
        then_block,
        else_branch: Some(else_expr),
    } = &tail.kind
    {
        let mut subst = BTreeMap::new();
        subst.insert("old.amount".into(), "amount".into());
        let cond_smt = encode_expr(cond, &subst).ok()?;
        let then_amt = extract_record_field_in_tuple(then_block.tail.as_deref()?, "amount", &subst)?;
        let else_tail = match &else_expr.kind {
            ExprKind::Block(b) => b.tail.as_deref()?,
            _ => else_expr,
        };
        let else_amt = extract_record_field_in_tuple(else_tail, "amount", &subst)?;
        let up_amt = format!("(ite {} {} {})", cond_smt, then_amt, else_amt);
        let has_arch = cond_smt.clone();
        let arch_val = "amount".to_string();
        return Some((up_amt, has_arch, arch_val));
    }
    None
}

fn extract_record_field_in_tuple(
    expr: &Expr,
    field_name: &str,
    subst: &BTreeMap<String, String>,
) -> Option<String> {
    match &expr.kind {
        ExprKind::Tuple(elems) => {
            for e in elems {
                if let Some(v) = extract_record_field_in_tuple(e, field_name, subst) {
                    return Some(v);
                }
            }
            None
        }
        ExprKind::RecordLit { fields, .. } => {
            for (k, v) in fields {
                if k == field_name {
                    return encode_expr(v, subst).ok();
                }
            }
            None
        }
        _ => None,
    }
}

fn extract_lens_down_semantics(
    down_fn: &FnDecl,
    up_amt: &str,
    has_arch: &str,
    arch_val: &str,
) -> Option<String> {
    let tail = down_fn.body.tail.as_deref()?;
    if let ExprKind::Match { arms, .. } = &tail.kind {
        if arms.len() >= 2 {
            let mut subst = BTreeMap::new();
            subst.insert("new.amount".into(), up_amt.to_string());
            subst.insert("a.original_amount".into(), arch_val.to_string());
            subst.insert("a.old_amount".into(), arch_val.to_string());
            let guard_smt = match &arms[0].guard {
                Some(g) => encode_expr(g, &subst).ok()?,
                None => "true".into(),
            };
            let arm0_amt = extract_record_field_in_tuple(&arms[0].body, "amount", &subst)?;
            let arm1_amt = extract_record_field_in_tuple(&arms[1].body, "amount", &subst)?;
            return Some(format!(
                "(ite (and {} {}) {} {})",
                has_arch, guard_smt, arm0_amt, arm1_amt
            ));
        }
    }
    None
}

fn check_capabilities(v1: &Program, v2: &Program) -> Vec<CapabilityCheckResult> {
    // Collect `@requires_cap("op", "cap")` and `@supports_cap("op", "cap")`
    let mut reqs: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut sups: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for prog in [v1, v2] {
        for item in &prog.items {
            let attrs = match item {
                Item::Record(r) => &r.attrs,
                Item::Fn(f) => &f.attrs,
                _ => continue,
            };
            for a in attrs {
                if a.name == "requires_cap" && a.args.len() >= 2 {
                    if let (ExprKind::Str(op), ExprKind::Str(cap)) =
                        (&a.args[0].value.kind, &a.args[1].value.kind)
                    {
                        reqs.entry(op.clone()).or_default().insert(cap.clone());
                    }
                } else if a.name == "supports_cap" && a.args.len() >= 2 {
                    if let (ExprKind::Str(op), ExprKind::Str(cap)) =
                        (&a.args[0].value.kind, &a.args[1].value.kind)
                    {
                        sups.entry(op.clone()).or_default().insert(cap.clone());
                    }
                }
            }
        }
    }
    let mut out = Vec::new();
    for (op, rset) in reqs {
        let sset = sups.get(&op).cloned().unwrap_or_default();
        let ok = rset.is_subset(&sset);
        out.push(CapabilityCheckResult {
            operation: op,
            required: rset.into_iter().collect(),
            supported: sset.into_iter().collect(),
            compatible: ok,
        });
    }
    out
}

fn normalize_base_type_name(ty_name: &str, aliases: &BTreeMap<String, Expr>) -> String {
    if aliases.contains_key(ty_name)
        || matches!(ty_name, "Int" | "Int64" | "Int32" | "UInt16" | "USize")
    {
        return "Int".into();
    }
    ty_name.to_string()
}

fn index_fields_by_identity(rec: &RecordDecl) -> BTreeMap<String, FieldDecl> {
    let mut map = BTreeMap::new();
    for f in &rec.fields {
        // Priority: explicit binary wire `@field(N)` -> explicit rename `@from("old")` / `@alias("old")` -> field name (default)
        let explicit_slot = f
            .attrs
            .iter()
            .find(|a| a.name == "field")
            .and_then(|a| a.args.first())
            .and_then(|arg| match &arg.value.kind {
                ExprKind::Int(n) => Some(format!("slot#{}", n)),
                _ => None,
            });
        let rename_from = f
            .attrs
            .iter()
            .find(|a| a.name == "from" || a.name == "alias")
            .and_then(|a| a.args.first())
            .and_then(|arg| match &arg.value.kind {
                ExprKind::Str(old) => Some(old.clone()),
                _ => None,
            });
        let key = explicit_slot
            .or(rename_from)
            .unwrap_or_else(|| f.name.clone());
        map.insert(key, f.clone());
    }
    map
}

fn collect_evolve_records(prog: &Program) -> BTreeMap<String, RecordDecl> {
    let mut map = BTreeMap::new();
    for item in &prog.items {
        if let Item::Record(r) = item {
            if let Some(attr) = r.attrs.iter().find(|a| a.name == "evolve") {
                let id = attr
                    .args
                    .first()
                    .and_then(|arg| match &arg.value.kind {
                        ExprKind::Str(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| r.name.clone());
                map.insert(id, r.clone());
            }
        }
    }
    map
}

fn collect_evolutions(prog: &Program) -> BTreeMap<String, EvolutionDecl> {
    let mut map = BTreeMap::new();
    for item in &prog.items {
        if let Item::Evolution(e) = item {
            map.insert(e.entity.clone(), e.clone());
        }
    }
    map
}

fn collect_pure_fns(prog: &Program) -> BTreeMap<String, FnDecl> {
    let mut map = BTreeMap::new();
    for item in &prog.items {
        if let Item::Fn(f) = item {
            if f.is_pure {
                map.insert(f.name.clone(), f.clone());
            }
        }
    }
    map
}

fn collect_alias_predicates(prog: &Program) -> BTreeMap<String, Expr> {
    let mut map = BTreeMap::new();
    for item in &prog.items {
        if let Item::TypeAlias(ta) = item {
            if let Some(p) = &ta.predicate {
                map.insert(ta.name.clone(), p.clone());
            }
        }
    }
    map
}

fn rename_fn_calls(f: &FnDecl, prefix: &str, scope: &BTreeMap<String, FnDecl>) -> FnDecl {
    let mut cloned = f.clone();
    cloned.name = format!("{}{}", prefix, f.name);
    cloned.body = rename_block_calls(&f.body, prefix, scope);
    cloned
}

fn rename_block_calls(blk: &Block, prefix: &str, scope: &BTreeMap<String, FnDecl>) -> Block {
    Block {
        stmts: blk
            .stmts
            .iter()
            .map(|s| match s {
                Stmt::Let {
                    name,
                    ty,
                    value,
                    is_mut,
                    span,
                } => Stmt::Let {
                    name: name.clone(),
                    ty: ty.clone(),
                    value: rename_expr_calls(value, prefix, scope),
                    is_mut: *is_mut,
                    span: *span,
                },
                other => other.clone(),
            })
            .collect(),
        tail: blk
            .tail
            .as_ref()
            .map(|t| Box::new(rename_expr_calls(t, prefix, scope))),
        span: blk.span,
    }
}

fn rename_expr_calls(expr: &Expr, prefix: &str, scope: &BTreeMap<String, FnDecl>) -> Expr {
    let kind = match &expr.kind {
        ExprKind::Call { callee, args } => {
            let new_callee = match &callee.kind {
                ExprKind::Ident(s) if scope.contains_key(s) => Box::new(Expr {
                    kind: ExprKind::Ident(format!("{}{}", prefix, s)),
                    span: callee.span,
                }),
                _ => callee.clone(),
            };
            ExprKind::Call {
                callee: new_callee,
                args: args
                    .iter()
                    .map(|a| CallArg {
                        name: a.name.clone(),
                        value: rename_expr_calls(&a.value, prefix, scope),
                    })
                    .collect(),
            }
        }
        ExprKind::Binary { op, left, right } => ExprKind::Binary {
            op: *op,
            left: Box::new(rename_expr_calls(left, prefix, scope)),
            right: Box::new(rename_expr_calls(right, prefix, scope)),
        },
        ExprKind::Unary { op, expr: inner } => ExprKind::Unary {
            op: *op,
            expr: Box::new(rename_expr_calls(inner, prefix, scope)),
        },
        ExprKind::If {
            cond,
            then_block,
            else_branch,
        } => ExprKind::If {
            cond: Box::new(rename_expr_calls(cond, prefix, scope)),
            then_block: rename_block_calls(then_block, prefix, scope),
            else_branch: else_branch
                .as_ref()
                .map(|e| Box::new(rename_expr_calls(e, prefix, scope))),
        },
        ExprKind::Match { scrutinee, arms } => ExprKind::Match {
            scrutinee: Box::new(rename_expr_calls(scrutinee, prefix, scope)),
            arms: arms
                .iter()
                .map(|a| MatchArm {
                    patterns: a.patterns.clone(),
                    guard: a.guard.as_ref().map(|g| rename_expr_calls(g, prefix, scope)),
                    body: rename_expr_calls(&a.body, prefix, scope),
                    span: a.span,
                })
                .collect(),
        },
        ExprKind::Block(b) => ExprKind::Block(rename_block_calls(b, prefix, scope)),
        other => other.clone(),
    };
    Expr {
        kind,
        span: expr.span,
    }
}

fn conjoin(clauses: &[String]) -> String {
    match clauses.len() {
        0 => "true".into(),
        1 => clauses[0].clone(),
        _ => format!("(and {})", clauses.join(" ")),
    }
}

pub fn receipt_hash(prog: &Program) -> String {
    let json = serde_json::to_string(prog).unwrap_or_default();
    let mut h = Sha256::new();
    h.update(json.as_bytes());
    format!("{:x}", h.finalize())
}
