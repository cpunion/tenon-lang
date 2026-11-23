use crate::l0::{AbsInt, Interval, L0Env, L0Result};
use crate::smt::{SmtOutcome, SmtQuery};
use crate::sync_solver::{DistributedAppManifest, solve_distributed_manifest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use tenon_syntax::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofTier {
    L0Fast,
    L1Smt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofRecord {
    pub obligation: String,
    pub tier: ProofTier,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: String,
    pub message: String,
    pub span: Span,
    pub notes: Vec<String>,
    pub counterexample: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckReport {
    pub ok: bool,
    pub proofs: Vec<ProofRecord>,
    pub diagnostics: Vec<Diagnostic>,
    pub sql_checks: BTreeMap<String, String>,
    pub manifest: Option<DistributedAppManifest>,
}

pub fn check_program(program: &Program) -> CheckReport {
    let mut ctx = CheckerCtx::new(program);
    ctx.check_all(program);

    // Run Protocol Constraint Solver if any distributed views/streams/subscribers exist
    let manifest = match solve_distributed_manifest(program) {
        Ok(m) => m,
        Err(errs) => {
            for e in errs {
                ctx.diagnostics.push(e);
            }
            None
        }
    };

    // Derive SQL CHECK constraints for visible refined records
    let sql_checks = derive_sqlite_checks(program);

    CheckReport {
        ok: ctx.diagnostics.is_empty(),
        proofs: ctx.proofs,
        diagnostics: ctx.diagnostics,
        sql_checks,
        manifest,
    }
}

struct CheckerCtx {
    type_aliases: BTreeMap<String, TypeAliasDecl>,
    records: BTreeMap<String, RecordDecl>,
    pure_fns: BTreeMap<String, FnDecl>,
    all_fns: BTreeMap<String, FnDecl>,
    proofs: Vec<ProofRecord>,
    diagnostics: Vec<Diagnostic>,
}

impl CheckerCtx {
    fn new(program: &Program) -> Self {
        let mut type_aliases = BTreeMap::new();
        let mut records = BTreeMap::new();
        let mut pure_fns = BTreeMap::new();
        let mut all_fns = BTreeMap::new();

        for item in &program.items {
            match item {
                Item::TypeAlias(t) => {
                    type_aliases.insert(t.name.clone(), t.clone());
                }
                Item::Record(r) => {
                    records.insert(r.name.clone(), r.clone());
                    for m in &r.methods {
                        all_fns.insert(format!("{}.{}", r.name, m.name), m.clone());
                    }
                }
                Item::Fn(f) => {
                    if f.is_pure {
                        pure_fns.insert(f.name.clone(), f.clone());
                    }
                    all_fns.insert(f.name.clone(), f.clone());
                }
                _ => {}
            }
        }

        Self {
            type_aliases,
            records,
            pure_fns,
            all_fns,
            proofs: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    fn check_all(&mut self, program: &Program) {
        for item in &program.items {
            if let Item::Fn(f) = item {
                self.check_fn(f);
            }
        }
    }

    fn check_fn(&mut self, f: &FnDecl) {
        let mut env = L0Env {
            pure_fns: self.pure_fns.clone(),
            ..Default::default()
        };
        let mut smt_premises: Vec<(Expr, BTreeMap<String, String>)> = Vec::new();
        let mut smt_int_vars: BTreeSet<String> = BTreeSet::new();
        let mut var_types: BTreeMap<String, String> = BTreeMap::new();

        // 1. Register parameters, their type refinements & record invariants
        for p in &f.params {
            let tname = p.ty.display_name();
            var_types.insert(p.name.clone(), tname.clone());
            self.assume_type_facts(
                &p.name,
                &p.ty,
                &mut env,
                &mut smt_int_vars,
                &mut smt_premises,
            );
            if let Some(pred) = &p.predicate {
                env.assume_expr(pred, Some(&p.name));
                let mut subst = BTreeMap::new();
                subst.insert("self".into(), p.name.clone());
                smt_premises.push((pred.clone(), subst));
            }
        }

        // 2. Assume function `requires`
        for req in &f.requires {
            env.assume_expr(req, None);
            smt_premises.push((req.clone(), BTreeMap::new()));
        }

        // 3. Walk statements & tail expression checking record literals, calls, div-by-zero, mutate blocks
        let mut active_mutate: Option<String> = None;
        let tail_expr = self.check_block(
            &f.body,
            &mut env,
            &mut smt_int_vars,
            &mut smt_premises,
            &mut var_types,
            &mut active_mutate,
            f.is_async,
        );

        // 4. Verify return type refinement if return type is refined (e.g., `Odd + Odd -> Even`)
        if let (Some(ret_ty), Some(ret_expr)) = (&f.ret_ty, &tail_expr) {
            self.check_expr_satisfies_type(
                ret_expr,
                ret_ty,
                &env,
                &smt_int_vars,
                &smt_premises,
                &format!("return value of `{}`", f.name),
                ret_expr.span,
            );
        }

        // 5. Verify every declared `ensures` clause (strict Fail-Closed: L0 -> L1 Z3!)
        for ens in &f.ensures {
            self.check_ensures(f, ens, tail_expr.as_ref(), &env, &smt_int_vars, &smt_premises);
        }
    }

    fn assume_type_facts(
        &self,
        var_name: &str,
        ty: &TypeExpr,
        env: &mut L0Env,
        smt_int_vars: &mut BTreeSet<String>,
        smt_premises: &mut Vec<(Expr, BTreeMap<String, String>)>,
    ) {
        if let TypeExpr::Named { name, .. } = ty {
            if matches!(
                name.as_str(),
                "Int" | "Int64" | "Int32" | "UInt16" | "USize"
            ) {
                smt_int_vars.insert(var_name.to_string());
                match name.as_str() {
                    "UInt16" => env.set_int(
                        var_name,
                        AbsInt::top().refine_interval(Interval { min: 0, max: 65535 }),
                    ),
                    "USize" => env.set_int(
                        var_name,
                        AbsInt::top().refine_interval(Interval {
                            min: 0,
                            max: i64::MAX as i128,
                        }),
                    ),
                    _ => {}
                }
            } else if let Some(alias) = self.type_aliases.get(name).cloned() {
                self.assume_type_facts(var_name, &alias.base, env, smt_int_vars, smt_premises);
                if let Some(pred) = &alias.predicate {
                    env.assume_expr(pred, Some(var_name));
                    let mut subst = BTreeMap::new();
                    subst.insert("self".into(), var_name.to_string());
                    smt_premises.push((pred.clone(), subst));
                }
            } else if let Some(rec) = self.records.get(name).cloned() {
                let mut field_subst = BTreeMap::new();
                field_subst.insert("self".into(), var_name.to_string());
                for field in &rec.fields {
                    let fvar = format!("{}.{}", var_name, field.name);
                    field_subst.insert(field.name.clone(), fvar.clone());
                    self.assume_type_facts(&fvar, &field.ty, env, smt_int_vars, smt_premises);
                    if let Some(fpred) = &field.predicate {
                        let rewritten = substitute_field_idents(fpred, &rec.fields, var_name);
                        env.assume_expr(&rewritten, Some(&fvar));
                        let mut s = field_subst.clone();
                        s.insert("self".into(), fvar.clone());
                        smt_premises.push((fpred.clone(), s));
                    }
                }
                for wc in &rec.where_clauses {
                    let rewritten = substitute_field_idents(wc, &rec.fields, var_name);
                    env.assume_expr(&rewritten, Some(var_name));
                    smt_premises.push((wc.clone(), field_subst.clone()));
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_block(
        &mut self,
        blk: &Block,
        env: &mut L0Env,
        smt_int_vars: &mut BTreeSet<String>,
        smt_premises: &mut Vec<(Expr, BTreeMap<String, String>)>,
        var_types: &mut BTreeMap<String, String>,
        active_mutate: &mut Option<String>,
        _is_async: bool,
    ) -> Option<Expr> {
        for stmt in &blk.stmts {
            match stmt {
                Stmt::Let {
                    name,
                    ty,
                    value,
                    span,
                    ..
                } => {
                    self.check_expr(value, env, smt_int_vars, smt_premises, active_mutate);
                    if let Some(t) = ty {
                        var_types.insert(name.clone(), t.display_name());
                        self.check_expr_satisfies_type(
                            value,
                            t,
                            env,
                            smt_int_vars,
                            smt_premises,
                            &format!("binding `{}`", name),
                            *span,
                        );
                        self.assume_type_facts(name, t, env, smt_int_vars, smt_premises);
                    }
                    let abs = env.eval_abs_int(value, None);
                    env.set_int(name.clone(), abs);
                    smt_int_vars.insert(name.clone());
                    // Add equality premise `name == value`
                    let eq = Expr {
                        kind: ExprKind::Binary {
                            op: BinOp::Eq,
                            left: Box::new(Expr {
                                kind: ExprKind::Ident(name.clone()),
                                span: *span,
                            }),
                            right: Box::new(value.clone()),
                        },
                        span: *span,
                    };
                    smt_premises.push((eq, BTreeMap::new()));
                }
                Stmt::Assert { cond, .. } => {
                    self.check_expr(cond, env, smt_int_vars, smt_premises, active_mutate);
                    env.assume_expr(cond, None);
                    smt_premises.push((cond.clone(), BTreeMap::new()));
                }
                Stmt::Assign {
                    target,
                    op,
                    value,
                    span,
                } => {
                    self.check_expr(value, env, smt_int_vars, smt_premises, active_mutate);
                    if let ExprKind::Field {
                        target: obj,
                        field,
                    } = &target.kind
                    {
                        if let ExprKind::Ident(var) = &obj.kind {
                            let is_inside_mutate = active_mutate.as_deref() == Some(var.as_str());
                            if let Some(rec_name) = var_types.get(var).cloned() {
                                if let Some(rec) = self.records.get(&rec_name).cloned() {
                                    let has_cross_inv = !rec.where_clauses.is_empty()
                                        || rec.fields.iter().any(|f| f.predicate.is_some());
                                    if has_cross_inv && !is_inside_mutate {
                                        self.diagnostics.push(Diagnostic {
                                            code: "E0420".into(),
                                            message: format!(
                                                "direct field mutation `{}.{}` on invariant record `{}` must be wrapped in `mutate {} {{ ... }}`",
                                                var, field, rec_name, var
                                            ),
                                            span: *span,
                                            notes: vec![
                                                "mutating one field directly could expose an intermediate invalid invariant state".into(),
                                            ],
                                            counterexample: BTreeMap::new(),
                                        });
                                    }
                                    // Update local field abstract value inside mutate block
                                    let fkey = format!("{}.{}", var, field);
                                    let new_val_expr = match op {
                                        AssignOp::Set => value.clone(),
                                        AssignOp::AddSet => Expr {
                                            kind: ExprKind::Binary {
                                                op: BinOp::Add,
                                                left: Box::new(target.clone()),
                                                right: Box::new(value.clone()),
                                            },
                                            span: *span,
                                        },
                                        AssignOp::SubSet => Expr {
                                            kind: ExprKind::Binary {
                                                op: BinOp::Sub,
                                                left: Box::new(target.clone()),
                                                right: Box::new(value.clone()),
                                            },
                                            span: *span,
                                        },
                                    };
                                    let new_abs = env.eval_abs_int(&new_val_expr, None);
                                    env.set_int(fkey, new_abs);
                                }
                            }
                        }
                    }
                }
                Stmt::Mutate { target, body, span } => {
                    let prev = active_mutate.replace(target.clone());
                    let _ = self.check_block(
                        body,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        active_mutate,
                        _is_async,
                    );
                    *active_mutate = prev;

                    // Verify record invariants hold upon exiting `mutate target { ... }` using symbolic post-state!
                    if let Some(rec_name) = var_types.get(target).cloned() {
                        if let Some(rec) = self.records.get(&rec_name).cloned() {
                            self.verify_record_invariants_after_mutate(
                                target,
                                &rec,
                                body,
                                env,
                                smt_int_vars,
                                smt_premises,
                                *span,
                            );
                        }
                    }
                }
                Stmt::Expr { expr, .. } => {
                    self.check_expr(expr, env, smt_int_vars, smt_premises, active_mutate);
                }
                Stmt::Return { value: Some(e), .. } => {
                    self.check_expr(e, env, smt_int_vars, smt_premises, active_mutate);
                }
                _ => {}
            }
        }
        if let Some(tail) = &blk.tail {
            self.check_expr(tail, env, smt_int_vars, smt_premises, active_mutate);
            return Some((**tail).clone());
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_record_invariants_after_mutate(
        &mut self,
        var: &str,
        rec: &RecordDecl,
        body: &Block,
        env: &L0Env,
        smt_int_vars: &BTreeSet<String>,
        smt_premises: &[(Expr, BTreeMap<String, String>)],
        span: Span,
    ) {
        // Build symbolic SSA post-state map for every field of `var` across assignments & `if` branches in `body`
        let mut sym_fields: BTreeMap<String, String> = BTreeMap::new();
        for f in &rec.fields {
            sym_fields.insert(
                f.name.clone(),
                crate::smt::smt_ident(&format!("{}.{}", var, f.name)),
            );
        }
        let mut mutated_any = false;
        apply_mutate_block_symbolic(var, body, &mut sym_fields, &mut mutated_any);

        let mut subst = BTreeMap::new();
        subst.insert("self".into(), var.to_string());
        for f in &rec.fields {
            let post_expr = sym_fields
                .get(&f.name)
                .cloned()
                .unwrap_or_else(|| crate::smt::smt_ident(&format!("{}.{}", var, f.name)));
            subst.insert(f.name.clone(), post_expr.clone());
            subst.insert(format!("{}.{}", var, f.name), post_expr);
        }

        for f in &rec.fields {
            if let Some(pred) = &f.predicate {
                let fkey = format!("{}.{}", var, f.name);
                let rewritten = substitute_field_idents(pred, &rec.fields, var);
                let l0_res = if mutated_any {
                    L0Result::Unknown // Escalate symbolic post-state directly to L1 Z3 for exact relational proof
                } else {
                    env.check_predicate(&rewritten, Some(&fkey))
                };
                match l0_res {
                    L0Result::Proved => {
                        self.proofs.push(ProofRecord {
                            obligation: format!("mutate({}) invariant `{}.{}`", var, rec.name, f.name),
                            tier: ProofTier::L0Fast,
                            span,
                        });
                    }
                    L0Result::Disproved(reason) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0421".into(),
                            message: format!(
                                "mutate `{}` violates field invariant `{}.{}`",
                                var, rec.name, f.name
                            ),
                            span,
                            notes: vec![reason],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    L0Result::Unknown => {
                        let mut q = SmtQuery::new(self.pure_fns.clone());
                        for v in smt_int_vars {
                            q.declare_int(v.clone());
                        }
                        for (p, s) in smt_premises {
                            let _ = q.add_premise_expr(p, s);
                        }
                        let mut fs = subst.clone();
                        fs.insert(
                            "self".into(),
                            sym_fields
                                .get(&f.name)
                                .cloned()
                                .unwrap_or_else(|| crate::smt::smt_ident(&fkey)),
                        );
                        if q.set_goal_expr(pred, &fs).is_ok() {
                            match q.solve() {
                                SmtOutcome::Proved => {
                                    self.proofs.push(ProofRecord {
                                        obligation: format!(
                                            "mutate({}) invariant `{}.{}`",
                                            var, rec.name, f.name
                                        ),
                                        tier: ProofTier::L1Smt,
                                        span,
                                    });
                                }
                                SmtOutcome::Counterexample(ce) => {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0421".into(),
                                        message: format!(
                                            "mutate `{}` may violate field invariant `{}.{}`",
                                            var, rec.name, f.name
                                        ),
                                        span,
                                        notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                                        counterexample: ce,
                                    });
                                }
                                SmtOutcome::Unknown(msg) => {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0422".into(),
                                        message: format!(
                                            "unproved mutate `{}` invariant `{}.{}` (Fail-Closed)",
                                            var, rec.name, f.name
                                        ),
                                        span,
                                        notes: vec![msg],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Also verify record-level `where` cross-field invariants on exit from `mutate`!
        for wc in &rec.where_clauses {
            let mut q = SmtQuery::new(self.pure_fns.clone());
            for v in smt_int_vars {
                q.declare_int(v.clone());
            }
            for (p, s) in smt_premises {
                let _ = q.add_premise_expr(p, s);
            }
            if q.set_goal_expr(wc, &subst).is_ok() {
                match q.solve() {
                    SmtOutcome::Proved => {
                        self.proofs.push(ProofRecord {
                            obligation: format!("mutate({}) record invariant `{}`", var, rec.name),
                            tier: ProofTier::L1Smt,
                            span,
                        });
                    }
                    SmtOutcome::Counterexample(ce) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0421".into(),
                            message: format!(
                                "mutate `{}` may violate record invariant on `{}`",
                                var, rec.name
                            ),
                            span,
                            notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                            counterexample: ce,
                        });
                    }
                    SmtOutcome::Unknown(msg) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0422".into(),
                            message: format!(
                                "unproved mutate `{}` record invariant on `{}` (Fail-Closed)",
                                var, rec.name
                            ),
                            span,
                            notes: vec![msg],
                            counterexample: BTreeMap::new(),
                        });
                    }
                }
            }
        }
    }

    fn check_expr(
        &mut self,
        expr: &Expr,
        env: &L0Env,
        smt_int_vars: &BTreeSet<String>,
        smt_premises: &[(Expr, BTreeMap<String, String>)],
        active_mutate: &Option<String>,
    ) {
        match &expr.kind {
            ExprKind::Binary { op, left, right } => {
                self.check_expr(left, env, smt_int_vars, smt_premises, active_mutate);
                self.check_expr(right, env, smt_int_vars, smt_premises, active_mutate);
                if matches!(op, BinOp::Div | BinOp::Mod) {
                    let denom = env.eval_abs_int(right, None);
                    if !denom.non_zero {
                        // Try proving `right != 0` via L0/L1
                        let ne_zero = Expr {
                            kind: ExprKind::Binary {
                                op: BinOp::Ne,
                                left: right.clone(),
                                right: Box::new(Expr {
                                    kind: ExprKind::Int(0),
                                    span: right.span,
                                }),
                            },
                            span: right.span,
                        };
                        if matches!(env.check_predicate(&ne_zero, None), L0Result::Proved) {
                            self.proofs.push(ProofRecord {
                                obligation: "division denominator != 0".into(),
                                tier: ProofTier::L0Fast,
                                span: expr.span,
                            });
                        } else {
                            self.diagnostics.push(Diagnostic {
                                code: "E0430".into(),
                                message: "potential division/modulo by zero: denominator is not proved NonZero".into(),
                                span: expr.span,
                                notes: vec!["refine denominator with `where self != 0` or guard with `if denom != 0`".into()],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    } else {
                        self.proofs.push(ProofRecord {
                            obligation: "division denominator != 0 (elided)".into(),
                            tier: ProofTier::L0Fast,
                            span: expr.span,
                        });
                    }
                }
            }
            ExprKind::RecordLit { name, fields, .. } => {
                for (_, val) in fields {
                    self.check_expr(val, env, smt_int_vars, smt_premises, active_mutate);
                }
                if let Some(rec) = self.records.get(name).cloned() {
                    self.check_record_construction(&rec, fields, env, smt_int_vars, smt_premises, expr.span);
                }
            }
            ExprKind::Call { callee, args } => {
                // Check `inout` exclusivity: cannot pass same identifier to two `inout` params!
                if let ExprKind::Ident(fname) = &callee.kind {
                    if let Some(fdecl) = self.all_fns.get(fname).cloned() {
                        let mut inout_vars = BTreeSet::new();
                        for (p, a) in fdecl.params.iter().zip(args.iter()) {
                            if p.is_inout {
                                if let ExprKind::Ident(v) = &a.value.kind {
                                    if !inout_vars.insert(v.clone()) {
                                        self.diagnostics.push(Diagnostic {
                                            code: "E0501".into(),
                                            message: format!(
                                                "MVS exclusivity violation: variable `{}` passed to multiple `inout` parameters of `{}`",
                                                v, fname
                                            ),
                                            span: a.value.span,
                                            notes: vec!["Tenon forbids overlapping mutable aliases".into()],
                                            counterexample: BTreeMap::new(),
                                        });
                                    }
                                    if active_mutate.as_deref() == Some(v.as_str()) {
                                        self.diagnostics.push(Diagnostic {
                                            code: "E0502".into(),
                                            message: format!(
                                                "cannot pass `{}` (currently unsealed inside `mutate {}`) to `{}`",
                                                v, v, fname
                                            ),
                                            span: a.value.span,
                                            notes: vec!["record invariant is temporarily suspended inside `mutate`".into()],
                                            counterexample: BTreeMap::new(),
                                        });
                                    }
                                }
                            }
                            self.check_expr_satisfies_type(
                                &a.value,
                                &p.ty,
                                env,
                                smt_int_vars,
                                smt_premises,
                                &format!("argument `{}` of `{}`", p.name, fname),
                                a.value.span,
                            );
                        }
                    }
                }
                for a in args {
                    self.check_expr(&a.value, env, smt_int_vars, smt_premises, active_mutate);
                }
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                self.check_expr(cond, env, smt_int_vars, smt_premises, active_mutate);
                // Flow-sensitive narrowing in then-branch
                let mut then_env = env.clone();
                then_env.assume_expr(cond, None);
                let mut then_premises = smt_premises.to_vec();
                then_premises.push(((**cond).clone(), BTreeMap::new()));
                let mut then_vars = BTreeMap::new();
                let mut m = active_mutate.clone();
                self.check_block(
                    then_block,
                    &mut then_env,
                    &mut smt_int_vars.clone(),
                    &mut then_premises,
                    &mut then_vars,
                    &mut m,
                    false,
                );
                if let Some(eb) = else_branch {
                    let neg_cond = Expr {
                        kind: ExprKind::Unary {
                            op: UnOp::Not,
                            expr: cond.clone(),
                        },
                        span: cond.span,
                    };
                    let mut else_env = env.clone();
                    else_env.assume_expr(&neg_cond, None);
                    let mut else_premises = smt_premises.to_vec();
                    else_premises.push((neg_cond, BTreeMap::new()));
                    self.check_expr(eb, &else_env, smt_int_vars, &else_premises, active_mutate);
                }
            }
            ExprKind::Block(blk) => {
                let mut b_env = env.clone();
                let mut b_vars = BTreeMap::new();
                let mut m = active_mutate.clone();
                self.check_block(
                    blk,
                    &mut b_env,
                    &mut smt_int_vars.clone(),
                    &mut smt_premises.to_vec(),
                    &mut b_vars,
                    &mut m,
                    false,
                );
            }
            _ => {}
        }
    }

    fn check_record_construction(
        &mut self,
        rec: &RecordDecl,
        provided: &[(String, Expr)],
        env: &L0Env,
        smt_int_vars: &BTreeSet<String>,
        smt_premises: &[(Expr, BTreeMap<String, String>)],
        span: Span,
    ) {
        let mut local_env = env.clone();
        let mut local_vars = smt_int_vars.clone();
        let mut local_premises = smt_premises.to_vec();
        let mut field_exprs: BTreeMap<String, Expr> = BTreeMap::new();

        for (fname, fexpr) in provided {
            field_exprs.insert(fname.clone(), fexpr.clone());
        }

        // Check each field in topological declaration order (`year`, then `month`, then `day`!)
        for f in &rec.fields {
            let Some(val_expr) = field_exprs.get(&f.name) else {
                continue;
            };
            let abs = local_env.eval_abs_int(val_expr, None);
            local_env.set_int(f.name.clone(), abs);
            local_vars.insert(f.name.clone());
            // Bind field name in SMT premises
            let eq = Expr {
                kind: ExprKind::Binary {
                    op: BinOp::Eq,
                    left: Box::new(Expr {
                        kind: ExprKind::Ident(f.name.clone()),
                        span: val_expr.span,
                    }),
                    right: Box::new(val_expr.clone()),
                },
                span: val_expr.span,
            };
            local_premises.push((eq, BTreeMap::new()));

            // Check type alias constraint on field type
            self.check_expr_satisfies_type(
                val_expr,
                &f.ty,
                &local_env,
                &local_vars,
                &local_premises,
                &format!("field `{}` of `{}`", f.name, rec.name),
                val_expr.span,
            );

            // Check field-level `where` predicate (e.g., `1 <= self && self <= days_in_month(year, month)`)
            if let Some(pred) = &f.predicate {
                match local_env.check_predicate(pred, Some(&f.name)) {
                    L0Result::Proved => {
                        self.proofs.push(ProofRecord {
                            obligation: format!("record `{}` field `{}` refinement", rec.name, f.name),
                            tier: ProofTier::L0Fast,
                            span: val_expr.span,
                        });
                    }
                    L0Result::Disproved(reason) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0412".into(),
                            message: format!(
                                "refinement constraint not satisfied for field `{}` of `{}`",
                                f.name, rec.name
                            ),
                            span: val_expr.span,
                            notes: vec![
                                reason,
                                format!("hint: use `{}.parse(...)` for unchecked runtime inputs", rec.name),
                            ],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    L0Result::Unknown => {
                        // Escalate to L1 Z3!
                        let mut q = SmtQuery::new(self.pure_fns.clone());
                        for v in &local_vars {
                            q.declare_int(v.clone());
                        }
                        for (p, s) in &local_premises {
                            let _ = q.add_premise_expr(p, s);
                        }
                        let mut subst = BTreeMap::new();
                        subst.insert("self".into(), f.name.clone());
                        if let Err(e) = q.set_goal_expr(pred, &subst) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0413".into(),
                                message: format!(
                                    "cannot statically prove refinement for `{}.{}` (Fail-Closed)",
                                    rec.name, f.name
                                ),
                                span: val_expr.span,
                                notes: vec![e],
                                counterexample: BTreeMap::new(),
                            });
                            continue;
                        }
                        match q.solve() {
                            SmtOutcome::Proved => {
                                self.proofs.push(ProofRecord {
                                    obligation: format!(
                                        "record `{}` field `{}` refinement",
                                        rec.name, f.name
                                    ),
                                    tier: ProofTier::L1Smt,
                                    span: val_expr.span,
                                });
                            }
                            SmtOutcome::Counterexample(ce) => {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0412".into(),
                                    message: format!(
                                        "refinement constraint not satisfied for field `{}` of `{}`",
                                        f.name, rec.name
                                    ),
                                    span: val_expr.span,
                                    notes: vec![
                                        "counterexample found by L1 SMT (Z3)".into(),
                                        format!("hint: use `{}.parse(...)` for unchecked runtime inputs", rec.name),
                                    ],
                                    counterexample: ce,
                                });
                            }
                            SmtOutcome::Unknown(msg) => {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0413".into(),
                                    message: format!(
                                        "unproved refinement for `{}.{}` (Fail-Closed)",
                                        rec.name, f.name
                                    ),
                                    span: val_expr.span,
                                    notes: vec![msg],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        }
                    }
                }
            }
        }

        // Check record-level `where` clauses (e.g., `self.start <= self.end`)
        for wc in &rec.where_clauses {
            let rewritten = substitute_self_fields(wc);
            match local_env.check_predicate(&rewritten, None) {
                L0Result::Proved => {
                    self.proofs.push(ProofRecord {
                        obligation: format!("record `{}` invariant", rec.name),
                        tier: ProofTier::L0Fast,
                        span,
                    });
                }
                L0Result::Disproved(reason) => {
                    self.diagnostics.push(Diagnostic {
                        code: "E0414".into(),
                        message: format!("record invariant not satisfied for `{}`", rec.name),
                        span,
                        notes: vec![reason],
                        counterexample: BTreeMap::new(),
                    });
                }
                L0Result::Unknown => {
                    let mut q = SmtQuery::new(self.pure_fns.clone());
                    for v in &local_vars {
                        q.declare_int(v.clone());
                    }
                    for (p, s) in &local_premises {
                        let _ = q.add_premise_expr(p, s);
                    }
                    if q.set_goal_expr(&rewritten, &BTreeMap::new()).is_ok() {
                        match q.solve() {
                            SmtOutcome::Proved => self.proofs.push(ProofRecord {
                                obligation: format!("record `{}` invariant", rec.name),
                                tier: ProofTier::L1Smt,
                                span,
                            }),
                            SmtOutcome::Counterexample(ce) => self.diagnostics.push(Diagnostic {
                                code: "E0414".into(),
                                message: format!("record invariant not satisfied for `{}`", rec.name),
                                span,
                                notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                                counterexample: ce,
                            }),
                            SmtOutcome::Unknown(msg) => self.diagnostics.push(Diagnostic {
                                code: "E0415".into(),
                                message: format!("unproved record invariant for `{}` (Fail-Closed)", rec.name),
                                span,
                                notes: vec![msg],
                                counterexample: BTreeMap::new(),
                            }),
                        }
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_expr_satisfies_type(
        &mut self,
        expr: &Expr,
        ty: &TypeExpr,
        env: &L0Env,
        smt_int_vars: &BTreeSet<String>,
        smt_premises: &[(Expr, BTreeMap<String, String>)],
        context: &str,
        span: Span,
    ) {
        let TypeExpr::Named { name, .. } = ty else {
            return;
        };
        let Some(alias) = self.type_aliases.get(name).cloned() else {
            return;
        };
        let Some(pred) = &alias.predicate else {
            return;
        };

        // Bind a fresh symbolic variable `__val` to `expr` in L0
        let mut check_env = env.clone();
        let abs = env.eval_abs_int(expr, None);
        check_env.set_int("__val", abs);

        match check_env.check_predicate(pred, Some("__val")) {
            L0Result::Proved => {
                self.proofs.push(ProofRecord {
                    obligation: format!("{} satisfies `{}`", context, name),
                    tier: ProofTier::L0Fast,
                    span,
                });
            }
            L0Result::Disproved(reason) => {
                self.diagnostics.push(Diagnostic {
                    code: "E0410".into(),
                    message: format!("{} violates refinement type `{}`", context, name),
                    span,
                    notes: vec![reason],
                    counterexample: BTreeMap::new(),
                });
            }
            L0Result::Unknown => {
                let mut q = SmtQuery::new(self.pure_fns.clone());
                for v in smt_int_vars {
                    q.declare_int(v.clone());
                }
                q.declare_int("__val");
                for (p, s) in smt_premises {
                    let _ = q.add_premise_expr(p, s);
                }
                let eq = Expr {
                    kind: ExprKind::Binary {
                        op: BinOp::Eq,
                        left: Box::new(Expr {
                            kind: ExprKind::Ident("__val".into()),
                            span,
                        }),
                        right: Box::new(expr.clone()),
                    },
                    span,
                };
                if q.add_premise_expr(&eq, &BTreeMap::new()).is_err() {
                    self.diagnostics.push(Diagnostic {
                        code: "E0411".into(),
                        message: format!("unproved refinement `{}` for {} (Fail-Closed)", name, context),
                        span,
                        notes: vec!["expression outside supported SMT fragment".into()],
                        counterexample: BTreeMap::new(),
                    });
                    return;
                }
                let mut subst = BTreeMap::new();
                subst.insert("self".into(), "__val".into());
                if let Err(e) = q.set_goal_expr(pred, &subst) {
                    self.diagnostics.push(Diagnostic {
                        code: "E0411".into(),
                        message: format!("unproved refinement `{}` for {} (Fail-Closed)", name, context),
                        span,
                        notes: vec![e],
                        counterexample: BTreeMap::new(),
                    });
                    return;
                }
                match q.solve() {
                    SmtOutcome::Proved => self.proofs.push(ProofRecord {
                        obligation: format!("{} satisfies `{}`", context, name),
                        tier: ProofTier::L1Smt,
                        span,
                    }),
                    SmtOutcome::Counterexample(ce) => self.diagnostics.push(Diagnostic {
                        code: "E0410".into(),
                        message: format!("{} may violate refinement type `{}`", context, name),
                        span,
                        notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                        counterexample: ce,
                    }),
                    SmtOutcome::Unknown(msg) => self.diagnostics.push(Diagnostic {
                        code: "E0411".into(),
                        message: format!("unproved refinement `{}` for {} (Fail-Closed)", name, context),
                        span,
                        notes: vec![msg],
                        counterexample: BTreeMap::new(),
                    }),
                }
            }
        }
    }

    fn check_ensures(
        &mut self,
        f: &FnDecl,
        ens: &Expr,
        tail_expr: Option<&Expr>,
        env: &L0Env,
        smt_int_vars: &BTreeSet<String>,
        smt_premises: &[(Expr, BTreeMap<String, String>)],
    ) {
        let mut check_env = env.clone();
        if let Some(te) = tail_expr {
            let abs = env.eval_abs_int(te, None);
            check_env.set_int("result", abs);
        }

        match check_env.check_predicate(ens, Some("result")) {
            L0Result::Proved => {
                self.proofs.push(ProofRecord {
                    obligation: format!("ensures on `{}`", f.name),
                    tier: ProofTier::L0Fast,
                    span: ens.span,
                });
                return;
            }
            L0Result::Disproved(reason) => {
                self.diagnostics.push(Diagnostic {
                    code: "V1004".into(),
                    message: format!("postcondition `ensures` violated on `{}`", f.name),
                    span: ens.span,
                    notes: vec![reason],
                    counterexample: BTreeMap::new(),
                });
                return;
            }
            L0Result::Unknown => {}
        }

        // Escalate to L1 Z3 (strict Fail-Closed!)
        let mut q = SmtQuery::new(self.pure_fns.clone());
        for v in smt_int_vars {
            q.declare_int(v.clone());
        }
        q.declare_int("result");
        for (p, s) in smt_premises {
            let _ = q.add_premise_expr(p, s);
        }
        if let Ok(body_smt) = crate::smt::encode_block(&f.body, &BTreeMap::new()) {
            q.premises.push(format!("(= result {})", body_smt));
        }
        if let Err(e) = q.set_goal_expr(ens, &BTreeMap::new()) {
            self.diagnostics.push(Diagnostic {
                code: "V1005".into(),
                message: format!("unproved `ensures` on `{}` (Fail-Closed)", f.name),
                span: ens.span,
                notes: vec![e],
                counterexample: BTreeMap::new(),
            });
            return;
        }
        match q.solve() {
            SmtOutcome::Proved => self.proofs.push(ProofRecord {
                obligation: format!("ensures on `{}`", f.name),
                tier: ProofTier::L1Smt,
                span: ens.span,
            }),
            SmtOutcome::Counterexample(ce) => self.diagnostics.push(Diagnostic {
                code: "V1004".into(),
                message: format!("postcondition `ensures` may be violated on `{}`", f.name),
                span: ens.span,
                notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                counterexample: ce,
            }),
            SmtOutcome::Unknown(msg) => self.diagnostics.push(Diagnostic {
                code: "V1005".into(),
                message: format!("unproved `ensures` on `{}` (Fail-Closed)", f.name),
                span: ens.span,
                notes: vec![msg],
                counterexample: BTreeMap::new(),
            }),
        }
    }
}

fn substitute_field_idents(expr: &Expr, fields: &[FieldDecl], prefix: &str) -> Expr {
    let field_set: BTreeSet<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    fn rewrite(e: &Expr, fields: &BTreeSet<&str>, prefix: &str) -> Expr {
        let kind = match &e.kind {
            ExprKind::Ident(s) if fields.contains(s.as_str()) => ExprKind::Field {
                target: Box::new(Expr {
                    kind: ExprKind::Ident(prefix.to_string()),
                    span: e.span,
                }),
                field: s.clone(),
            },
            ExprKind::Binary { op, left, right } => ExprKind::Binary {
                op: *op,
                left: Box::new(rewrite(left, fields, prefix)),
                right: Box::new(rewrite(right, fields, prefix)),
            },
            ExprKind::Unary { op, expr } => ExprKind::Unary {
                op: *op,
                expr: Box::new(rewrite(expr, fields, prefix)),
            },
            ExprKind::Call { callee, args } => ExprKind::Call {
                callee: callee.clone(),
                args: args
                    .iter()
                    .map(|a| CallArg {
                        name: a.name.clone(),
                        value: rewrite(&a.value, fields, prefix),
                    })
                    .collect(),
            },
            other => other.clone(),
        };
        Expr { kind, span: e.span }
    }
    rewrite(expr, &field_set, prefix)
}

fn substitute_self_fields(expr: &Expr) -> Expr {
    let kind = match &expr.kind {
        ExprKind::Field { target, field } => {
            if let ExprKind::Ident(s) = &target.kind {
                if s == "self" {
                    return Expr {
                        kind: ExprKind::Ident(field.clone()),
                        span: expr.span,
                    };
                }
            }
            expr.kind.clone()
        }
        ExprKind::Binary { op, left, right } => ExprKind::Binary {
            op: *op,
            left: Box::new(substitute_self_fields(left)),
            right: Box::new(substitute_self_fields(right)),
        },
        ExprKind::Unary { op, expr: inner } => ExprKind::Unary {
            op: *op,
            expr: Box::new(substitute_self_fields(inner)),
        },
        other => other.clone(),
    };
    Expr {
        kind,
        span: expr.span,
    }
}

/// Automatically derive SQLite CHECK clauses from direct scalar predicates on records.
pub fn derive_sqlite_checks(program: &Program) -> BTreeMap<String, String> {
    let mut checks = BTreeMap::new();
    for item in &program.items {
        if let Item::Record(r) = item {
            let mut clauses = Vec::new();
            for f in &r.fields {
                if let Some(pred) = &f.predicate {
                    if let Some(sql) = expr_to_sql(pred, Some(&f.name)) {
                        clauses.push(format!("({})", sql));
                    }
                }
            }
            for wc in &r.where_clauses {
                if let Some(sql) = expr_to_sql(&substitute_self_fields(wc), None) {
                    clauses.push(format!("({})", sql));
                }
            }
            if !clauses.is_empty() {
                checks.insert(r.name.clone(), format!("CHECK (({}) IS 1)", clauses.join(" AND ")));
            }
        }
    }
    checks
}

fn expr_to_sql(expr: &Expr, self_col: Option<&str>) -> Option<String> {
    match &expr.kind {
        ExprKind::Int(n) => Some(n.to_string()),
        ExprKind::Bool(b) => Some(if *b { "1".into() } else { "0".into() }),
        ExprKind::Ident(s) => {
            let col = if s == "self" {
                self_col?
            } else {
                s.as_str()
            };
            Some(format!("\"{}\"", col))
        }
        ExprKind::Unary { op, expr } => {
            let inner = expr_to_sql(expr, self_col)?;
            match op {
                UnOp::Neg => Some(format!("(-{})", inner)),
                UnOp::Not => Some(format!("(NOT {})", inner)),
            }
        }
        ExprKind::Binary { op, left, right } => {
            let l = expr_to_sql(left, self_col)?;
            let r = expr_to_sql(right, self_col)?;
            let sym = match op {
                BinOp::And => "AND",
                BinOp::Or => "OR",
                BinOp::Eq => "=",
                BinOp::Ne => "!=",
                BinOp::Lt => "<",
                BinOp::Le => "<=",
                BinOp::Gt => ">",
                BinOp::Ge => ">=",
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Mod => "%",
                _ => return None,
            };
            Some(format!("({} {} {})", l, sym, r))
        }
        _ => None,
    }
}

fn apply_mutate_block_symbolic(
    var: &str,
    blk: &Block,
    sym_fields: &mut BTreeMap<String, String>,
    mutated_any: &mut bool,
) {
    for stmt in &blk.stmts {
        match stmt {
            Stmt::Assign {
                target, op, value, ..
            } => {
                if let ExprKind::Field { target: obj, field } = &target.kind {
                    if matches!(&obj.kind, ExprKind::Ident(s) if s == var) {
                        let mut subst = BTreeMap::new();
                        for (k, v) in sym_fields.iter() {
                            subst.insert(format!("{}.{}", var, k), v.clone());
                            subst.insert(k.clone(), v.clone());
                        }
                        if let Ok(rhs) = crate::smt::encode_expr(value, &subst) {
                            let cur = sym_fields
                                .get(field)
                                .cloned()
                                .unwrap_or_else(|| crate::smt::smt_ident(&format!("{}.{}", var, field)));
                            let updated = match op {
                                AssignOp::Set => rhs,
                                AssignOp::AddSet => format!("(+ {} {})", cur, rhs),
                                AssignOp::SubSet => format!("(- {} {})", cur, rhs),
                            };
                            sym_fields.insert(field.clone(), updated);
                            *mutated_any = true;
                        }
                    }
                }
            }
            Stmt::Expr { expr, .. } => {
                apply_mutate_expr_symbolic(var, expr, sym_fields, mutated_any);
            }
            _ => {}
        }
    }
    if let Some(tail) = &blk.tail {
        apply_mutate_expr_symbolic(var, tail, sym_fields, mutated_any);
    }
}

fn apply_mutate_expr_symbolic(
    var: &str,
    expr: &Expr,
    sym_fields: &mut BTreeMap<String, String>,
    mutated_any: &mut bool,
) {
    match &expr.kind {
        ExprKind::If {
            cond,
            then_block,
            else_branch,
        } => {
            let mut subst = BTreeMap::new();
            for (k, v) in sym_fields.iter() {
                subst.insert(format!("{}.{}", var, k), v.clone());
                subst.insert(k.clone(), v.clone());
            }
            let Ok(cond_smt) = crate::smt::encode_expr(cond, &subst) else {
                return;
            };
            let mut then_fields = sym_fields.clone();
            let mut else_fields = sym_fields.clone();
            let mut m1 = false;
            let mut m2 = false;
            apply_mutate_block_symbolic(var, then_block, &mut then_fields, &mut m1);
            if let Some(eb) = else_branch {
                apply_mutate_expr_symbolic(var, eb, &mut else_fields, &mut m2);
            }
            if m1 || m2 {
                *mutated_any = true;
                let keys: Vec<String> = sym_fields.keys().cloned().collect();
                for k in keys {
                    let t = &then_fields[&k];
                    let e = &else_fields[&k];
                    if t != e {
                        sym_fields.insert(k, format!("(ite {} {} {})", cond_smt, t, e));
                    }
                }
            }
        }
        ExprKind::Block(b) => apply_mutate_block_symbolic(var, b, sym_fields, mutated_any),
        _ => {}
    }
}
