use crate::checker::Diagnostic;
use crate::l0::L0Env;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use tenon_syntax::*;

/// Per-function verified MIR metadata emitted by `tenon-verify` for downstream backends
/// (Cranelift JIT, LLVM 22 AOT, Wasm, and LSP Inlay Hints).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MirFnSummary {
    pub name: String,
    pub is_pure: bool,
    pub is_ghost: bool,
    pub is_fip: bool,
    pub is_arena: bool,
    pub is_no_heap_alloc: bool,
    pub is_checkfree: bool,
    pub param_count: usize,
    pub non_escaping_locals: Vec<String>,
    pub stack_promotable_locals: Vec<String>,
    /// Interprocedurally inferred return interval `[lo, hi]` for pure integer functions without explicit `ensures`.
    #[serde(default)]
    pub inferred_ret_interval: Option<(i64, i64)>,
}

/// Unified Verified Middle-End IR facts shared across Cranelift, LLVM AOT, Wasm, and LSP.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerifiedMir {
    /// Span start offsets of `xs[i]` reads/writes statically proved `0 <= i < len(xs)`.
    #[serde(default)]
    pub proved_safe_indices: BTreeSet<usize>,
    /// Span start offsets of integer `+`/`-`/`*` statically proved within target bit-width bounds.
    #[serde(default)]
    pub proved_no_overflow: BTreeSet<usize>,
    /// Span start offsets of `as`/`as?`/`refine_cast` statically proved infallible (`S <: T`).
    #[serde(default)]
    pub infallible_casts: BTreeSet<usize>,
    /// Resolved static type display name for each `refine_cast[T](x)` call site.
    #[serde(default)]
    pub refine_cast_types: BTreeMap<usize, String>,
    /// Per-function local variables proved not to escape return, closures, channels, or outer structs.
    #[serde(default)]
    pub non_escaping_vars: BTreeMap<String, BTreeSet<String>>,
    /// Per-function local immutable scalar-only records proved safe for zero-malloc LLVM stack `alloca` promotion.
    #[serde(default)]
    pub stack_promotable_vars: BTreeMap<String, BTreeSet<String>>,
    /// Functions verified to satisfy `@fip` (Functional-In-Place: zero net heap allocation).
    #[serde(default)]
    pub fip_verified_fns: BTreeSet<String>,
    /// Functions verified to satisfy `@arena` (Tier-2 scoped 64 KiB Slab Arena, no escaping arena handles).
    #[serde(default)]
    pub arena_verified_fns: BTreeSet<String>,
    /// Functions verified to satisfy `@checkfree` (100% zero runtime bounds/overflow/div-zero/cast checks).
    #[serde(default)]
    pub checkfree_verified_fns: BTreeSet<String>,
    /// Verified `@gpu_graph` pass DAGs, barrier schedules, VRAM aliasing, and kernel fusion plans.
    #[serde(default)]
    pub gpu_graphs: Vec<crate::gpu_graph::GpuGraphPlan>,
    /// Compact per-function verification summaries.
    #[serde(default)]
    pub fn_summaries: BTreeMap<String, MirFnSummary>,
}

/// Returns the inclusive mathematical `[min, max]` range for a scalar integer type.
pub fn scalar_int_bounds(sk: Option<ScalarKind>) -> (i128, i128) {
    match sk {
        Some(ScalarKind::I8) => (i8::MIN as i128, i8::MAX as i128),
        Some(ScalarKind::I16) => (i16::MIN as i128, i16::MAX as i128),
        Some(ScalarKind::I32) => (i32::MIN as i128, i32::MAX as i128),
        Some(ScalarKind::I64) | None => (i64::MIN as i128, i64::MAX as i128),
        Some(ScalarKind::U8) => (0, u8::MAX as i128),
        Some(ScalarKind::U16) => (0, u16::MAX as i128),
        Some(ScalarKind::U32) => (0, u32::MAX as i128),
        Some(ScalarKind::U64) => (0, u64::MAX as i128),
        Some(ScalarKind::F16) | Some(ScalarKind::BF16) | Some(ScalarKind::F32) | Some(ScalarKind::F64) => {
            (i64::MIN as i128, i64::MAX as i128)
        }
    }
}

/// Statically proves that integer `left op right` cannot overflow its target scalar bit width
/// using exact `i128` L0 intervals plus O(1) symbolic octagonal diff-bound reasoning.
pub fn prove_binop_no_overflow(
    op: BinOp,
    left: &Expr,
    right: &Expr,
    target_sk: Option<ScalarKind>,
    env: &L0Env,
) -> bool {
    if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul) {
        return false;
    }
    let (ty_min, ty_max) = scalar_int_bounds(target_sk);
    let li = env.eval_abs_int(left, None).interval;
    let ri = env.eval_abs_int(right, None).interval;

    // 1. Exact mathematical i128 interval proof (handles constants, Range types, Int8..UInt32, bounded loops)
    let exact_bounds = match op {
        BinOp::Add => li
            .min
            .checked_add(ri.min)
            .zip(li.max.checked_add(ri.max)),
        BinOp::Sub => li
            .min
            .checked_sub(ri.max)
            .zip(li.max.checked_sub(ri.min)),
        BinOp::Mul => {
            let p1 = li.min.checked_mul(ri.min);
            let p2 = li.min.checked_mul(ri.max);
            let p3 = li.max.checked_mul(ri.min);
            let p4 = li.max.checked_mul(ri.max);
            match (p1, p2, p3, p4) {
                (Some(a), Some(b), Some(c), Some(d)) => {
                    let lo = a.min(b).min(c).min(d);
                    let hi = a.max(b).max(c).max(d);
                    Some((lo, hi))
                }
                _ => None,
            }
        }
        _ => None,
    };
    if let Some((lo, hi)) = exact_bounds {
        if lo >= ty_min && hi <= ty_max {
            return true;
        }
    }

    // 2. Symbolic relational proof for 64-bit signed `Int` loop counters & slices (`i < n => i + 1 <= n <= i64::MAX`,
    //    `0 <= i <= n => 0 <= n - i <= i64::MAX`, `x >= c => x - c in [0, i64::MAX]`).
    if matches!(target_sk, None | Some(ScalarKind::I64)) {
        match op {
            BinOp::Add => {
                // `x + c` where `x >= 0`, `c > 0`, and there exists `y` with `x - y <= -c` (e.g. `x < y` and `c == 1`)
                let check_var_plus_const = |var_expr: &Expr, c_val: i128, v_min: i128| -> bool {
                    if c_val > 0 && v_min >= 0 {
                        if let ExprKind::Ident(v) = &var_expr.kind {
                            return env
                                .diffs
                                .iter()
                                .any(|((a, _b), &d)| a == v && d <= -c_val);
                        }
                    }
                    false
                };
                if let ExprKind::Int(c) = right.kind {
                    if check_var_plus_const(left, c as i128, li.min) {
                        return true;
                    }
                }
                if let ExprKind::Int(c) = left.kind {
                    if check_var_plus_const(right, c as i128, ri.min) {
                        return true;
                    }
                }
            }
            BinOp::Sub => {
                // `y - x` where `0 <= x <= y` (i.e. `x - y <= 0`, `x >= 0`, `y >= 0`) -> fits in `[0, i64::MAX]`
                if let (ExprKind::Ident(y), ExprKind::Ident(x)) = (&left.kind, &right.kind) {
                    if li.min >= 0
                        && ri.min >= 0
                        && env.diffs.get(&(x.clone(), y.clone())).is_some_and(|&d| d <= 0)
                    {
                        return true;
                    }
                }
                // `x - c` where `c >= 0` and `x >= c` -> fits in `[0, i64::MAX]`
                if let ExprKind::Int(c) = right.kind {
                    if c >= 0 && li.min >= c as i128 {
                        return true;
                    }
                }
            }
            _ => {}
        }
    }

    false
}

/// Tier-2 Escape Analysis: identifies local heap-allocated variables (`Record`, `List`, `list_filled`)
/// whose references never escape the function frame, closures, outer collections, or `inout` parameters.
pub fn analyze_function_escapes(f: &FnDecl) -> BTreeSet<String> {
    let mut candidates = BTreeSet::new();
    collect_alloc_locals_block(&f.body, &mut candidates);
    if candidates.is_empty() {
        return candidates;
    }
    let mut escaped = BTreeSet::new();
    mark_escapes_block(&f.body, true, &candidates, &mut escaped);
    candidates.difference(&escaped).cloned().collect()
}

fn expr_is_fresh_local_alloc(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::RecordLit { .. } | ExprKind::List(_) => true,
        ExprKind::Call { callee, .. } => matches!(
            &callee.kind,
            ExprKind::Ident(n) if matches!(n.as_str(), "list_filled" | "bytes_new" | "bytes_filled")
        ),
        _ => false,
    }
}

fn collect_alloc_locals_block(blk: &Block, out: &mut BTreeSet<String>) {
    for s in &blk.stmts {
        match s {
            Stmt::Let { name, value, .. } => {
                if name != "_" && expr_is_fresh_local_alloc(value) {
                    out.insert(name.clone());
                }
            }
            Stmt::While { body, .. }
            | Stmt::For { body, .. }
            | Stmt::Mutate { body, .. }
            | Stmt::Defer { body, .. } => {
                collect_alloc_locals_block(body, out);
            }
            _ => {}
        }
    }
}

fn mark_all_mentioned(expr: &Expr, cands: &BTreeSet<String>, escaped: &mut BTreeSet<String>) {
    match &expr.kind {
        ExprKind::Ident(s) => {
            if cands.contains(s) {
                escaped.insert(s.clone());
            }
        }
        ExprKind::Binary { left, right, .. } | ExprKind::Range { start: left, end: right } => {
            mark_all_mentioned(left, cands, escaped);
            mark_all_mentioned(right, cands, escaped);
        }
        ExprKind::Unary { expr: e, .. }
        | ExprKind::Field { target: e, .. }
        | ExprKind::TupleIndex { target: e, .. }
        | ExprKind::Try(e)
        | ExprKind::Await(e)
        | ExprKind::Cast { expr: e, .. }
        | ExprKind::TryCast { expr: e, .. }
        | ExprKind::ArrayRepeat { value: e, .. }
        | ExprKind::GenericInst { expr: e, .. } => mark_all_mentioned(e, cands, escaped),
        ExprKind::Index { target, index } => {
            mark_all_mentioned(target, cands, escaped);
            mark_all_mentioned(index, cands, escaped);
        }
        ExprKind::Call { callee, args } => {
            mark_all_mentioned(callee, cands, escaped);
            for a in args {
                mark_all_mentioned(&a.value, cands, escaped);
            }
        }
        ExprKind::RecordLit { fields, .. } => {
            for (_, v) in fields {
                mark_all_mentioned(v, cands, escaped);
            }
        }
        ExprKind::List(es) | ExprKind::Tuple(es) => {
            for e in es {
                mark_all_mentioned(e, cands, escaped);
            }
        }
        ExprKind::If {
            cond,
            then_block,
            else_branch,
        } => {
            mark_all_mentioned(cond, cands, escaped);
            mark_escapes_block(then_block, true, cands, escaped);
            if let Some(eb) = else_branch {
                mark_all_mentioned(eb, cands, escaped);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            mark_all_mentioned(scrutinee, cands, escaped);
            for a in arms {
                if let Some(g) = &a.guard {
                    mark_all_mentioned(g, cands, escaped);
                }
                mark_all_mentioned(&a.body, cands, escaped);
            }
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) => mark_escapes_block(b, true, cands, escaped),
        ExprKind::Lambda { body, .. } => mark_all_mentioned(body, cands, escaped),
        _ => {}
    }
}

fn mark_escapes_block(
    blk: &Block,
    tail_escapes: bool,
    cands: &BTreeSet<String>,
    escaped: &mut BTreeSet<String>,
) {
    for s in &blk.stmts {
        match s {
            Stmt::Let { value, .. } | Stmt::LetTuple { value, .. } => {
                mark_escapes_expr(value, false, cands, escaped);
                // Rebinding `let alias = cand` aliases the handle -> conservatively escape `cand`
                if let ExprKind::Ident(s) = &value.kind {
                    if cands.contains(s) {
                        escaped.insert(s.clone());
                    }
                }
            }
            Stmt::Assign { target, value, .. } => {
                // In-place `xs[i] = v` or `rec.f = v` does NOT escape `xs`/`rec`, only `value` if it's a cand!
                match &target.kind {
                    ExprKind::Index { index, .. } => {
                        mark_escapes_expr(index, false, cands, escaped);
                    }
                    ExprKind::Field { .. } | ExprKind::Ident(_) => {}
                    _ => mark_escapes_expr(target, false, cands, escaped),
                }
                // `xs = push(xs, v)` is self-consuming COW in-place -> keep `xs` non-escaping if `v` doesn't alias
                let is_self_push = matches!(
                    (&target.kind, &value.kind),
                    (ExprKind::Ident(lhs), ExprKind::Call { callee, args })
                        if matches!(&callee.kind, ExprKind::Ident(fnm) if fnm == "push" || fnm == "bytes_push")
                            && args.len() == 2
                            && matches!(&args[0].value.kind, ExprKind::Ident(a0) if a0 == lhs)
                );
                if is_self_push {
                    if let ExprKind::Call { args, .. } = &value.kind {
                        mark_escapes_expr(&args[1].value, true, cands, escaped);
                    }
                } else {
                    mark_escapes_expr(value, true, cands, escaped);
                }
            }
            Stmt::Return { value: Some(e), .. } => {
                mark_escapes_expr(e, true, cands, escaped);
            }
            Stmt::Assert { cond, .. } => {
                mark_escapes_expr(cond, false, cands, escaped);
            }
            Stmt::Expr { expr, .. } => {
                mark_escapes_expr(expr, false, cands, escaped);
            }
            Stmt::While { cond, body, .. } => {
                mark_escapes_expr(cond, false, cands, escaped);
                mark_escapes_block(body, false, cands, escaped);
            }
            Stmt::For { iter, body, .. } => {
                mark_escapes_expr(iter, false, cands, escaped);
                mark_escapes_block(body, false, cands, escaped);
            }
            Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } => {
                mark_escapes_block(body, false, cands, escaped);
            }
            Stmt::Ghost { .. } | Stmt::Return { value: None, .. } | Stmt::Break { .. } | Stmt::Continue { .. } => {}
        }
    }
    if let Some(t) = &blk.tail {
        mark_escapes_expr(t, tail_escapes, cands, escaped);
    }
}

fn mark_escapes_expr(
    expr: &Expr,
    in_escape_pos: bool,
    cands: &BTreeSet<String>,
    escaped: &mut BTreeSet<String>,
) {
    match &expr.kind {
        ExprKind::Ident(s) => {
            if in_escape_pos && cands.contains(s) {
                escaped.insert(s.clone());
            }
        }
        ExprKind::Binary { op, left, right } => {
            // Arithmetic/comparison/logical/bitwise produces a scalar value; operand containers do NOT escape.
            let esc = in_escape_pos && *op == BinOp::Concat;
            mark_escapes_expr(left, esc, cands, escaped);
            mark_escapes_expr(right, esc, cands, escaped);
        }
        ExprKind::Unary { expr: e, .. } => {
            mark_escapes_expr(e, false, cands, escaped);
        }
        ExprKind::Field { target, .. } | ExprKind::TupleIndex { target, .. } => {
            // Reading a field from a local container does NOT escape the outer container handle itself.
            if !matches!(&target.kind, ExprKind::Ident(_)) {
                mark_escapes_expr(target, false, cands, escaped);
            }
        }
        ExprKind::Index { target, index } => {
            if !matches!(&target.kind, ExprKind::Ident(_)) {
                mark_escapes_expr(target, false, cands, escaped);
            }
            mark_escapes_expr(index, false, cands, escaped);
        }
        ExprKind::Call { callee, args } => {
            let is_borrowing_builtin = matches!(
                &callee.kind,
                ExprKind::Ident(n)
                    if matches!(
                        n.as_str(),
                        "len" | "bytes_len" | "bytes_get" | "byte_at" | "print_int"
                    )
            );
            if is_borrowing_builtin {
                for a in args {
                    if !matches!(&a.value.kind, ExprKind::Ident(_)) {
                        mark_escapes_expr(&a.value, false, cands, escaped);
                    }
                }
            } else {
                mark_escapes_expr(callee, false, cands, escaped);
                for a in args {
                    mark_all_mentioned(&a.value, cands, escaped);
                }
            }
        }
        ExprKind::Range { start: left, end: right } => {
            mark_escapes_expr(left, false, cands, escaped);
            mark_escapes_expr(right, false, cands, escaped);
        }
        ExprKind::Try(e)
        | ExprKind::Await(e)
        | ExprKind::Cast { expr: e, .. }
        | ExprKind::TryCast { expr: e, .. } => mark_escapes_expr(e, in_escape_pos, cands, escaped),
        ExprKind::RecordLit { fields, .. } => {
            for (_, v) in fields {
                mark_all_mentioned(v, cands, escaped);
            }
        }
        ExprKind::List(es) | ExprKind::Tuple(es) => {
            for e in es {
                mark_all_mentioned(e, cands, escaped);
            }
        }
        ExprKind::If {
            cond,
            then_block,
            else_branch,
        } => {
            mark_escapes_expr(cond, false, cands, escaped);
            mark_escapes_block(then_block, in_escape_pos, cands, escaped);
            if let Some(eb) = else_branch {
                mark_escapes_expr(eb, in_escape_pos, cands, escaped);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            mark_escapes_expr(scrutinee, false, cands, escaped);
            for a in arms {
                if let Some(g) = &a.guard {
                    mark_escapes_expr(g, false, cands, escaped);
                }
                mark_escapes_expr(&a.body, in_escape_pos, cands, escaped);
            }
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) => {
            mark_escapes_block(b, in_escape_pos, cands, escaped);
        }
        ExprKind::Lambda { body, .. } => mark_all_mentioned(body, cands, escaped),
        _ => {}
    }
}

/// Verifies the `@fip` (Functional-In-Place) contract: a `@fip` function must achieve
/// zero net heap allocations by reusing owned same-type parameters in place rather than
/// performing unbounded dynamic allocations (`List`, `Text` concat, `list_filled`, non-`@fip` calls).
pub fn verify_fip_function(
    f: &FnDecl,
    fip_fns: &BTreeSet<String>,
    no_heap_fns: &BTreeSet<String>,
    non_alloc_builtins: &BTreeSet<&str>,
) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    let mut reusable_record_budget: BTreeMap<String, usize> = BTreeMap::new();
    for p in &f.params {
        if let TypeExpr::Named { name, .. } = &p.ty {
            *reusable_record_budget.entry(name.clone()).or_insert(0) += 1;
        }
    }
    let mut used_budget: BTreeMap<String, usize> = BTreeMap::new();
    check_fip_block(
        &f.body,
        &f.name,
        fip_fns,
        no_heap_fns,
        non_alloc_builtins,
        &reusable_record_budget,
        &mut used_budget,
        &mut diags,
    );
    diags
}

#[allow(clippy::too_many_arguments)]
fn check_fip_block(
    blk: &Block,
    fn_name: &str,
    fip_fns: &BTreeSet<String>,
    no_heap_fns: &BTreeSet<String>,
    non_alloc_builtins: &BTreeSet<&str>,
    budget: &BTreeMap<String, usize>,
    used: &mut BTreeMap<String, usize>,
    diags: &mut Vec<Diagnostic>,
) {
    for s in &blk.stmts {
        match s {
            Stmt::Let { value, .. } | Stmt::LetTuple { value, .. } => {
                check_fip_expr(value, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
            }
            Stmt::Assign { target, value, .. } => {
                check_fip_expr(target, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
                check_fip_expr(value, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
            }
            Stmt::Return { value: Some(e), .. } | Stmt::Expr { expr: e, .. } => {
                check_fip_expr(e, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
            }
            Stmt::Assert { cond, .. } => {
                check_fip_expr(cond, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
            }
            Stmt::While { cond, body, .. } => {
                check_fip_expr(cond, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
                check_fip_block(body, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
            }
            Stmt::For { iter, body, .. } => {
                check_fip_expr(iter, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
                check_fip_block(body, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
            }
            Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } => {
                check_fip_block(body, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
            }
            Stmt::Ghost { .. } | Stmt::Return { value: None, .. } | Stmt::Break { .. } | Stmt::Continue { .. } => {}
        }
    }
    if let Some(t) = &blk.tail {
        check_fip_expr(t, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
    }
}

#[allow(clippy::too_many_arguments)]
fn check_fip_expr(
    expr: &Expr,
    fn_name: &str,
    fip_fns: &BTreeSet<String>,
    no_heap_fns: &BTreeSet<String>,
    non_alloc_builtins: &BTreeSet<&str>,
    budget: &BTreeMap<String, usize>,
    used: &mut BTreeMap<String, usize>,
    diags: &mut Vec<Diagnostic>,
) {
    match &expr.kind {
        ExprKind::List(_) => {
            diags.push(Diagnostic {
                code: "E0620".into(),
                message: format!("`@fip` contract violated in `{}`: fresh heap `List` allocation", fn_name),
                span: expr.span,
                notes: vec!["`@fip` functions must have zero net heap allocations; mutate an `inout` buffer or reuse an owned parameter".into()],
                counterexample: BTreeMap::new(),
            });
        }
        ExprKind::RecordLit { name, fields, .. } => {
            for (_, v) in fields {
                check_fip_expr(v, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
            }
            let avail = budget.get(name).copied().unwrap_or(0);
            let cnt = used.entry(name.clone()).or_insert(0);
            *cnt += 1;
            if *cnt > avail {
                diags.push(Diagnostic {
                    code: "E0620".into(),
                    message: format!(
                        "`@fip` contract violated in `{}`: constructing `{}` exceeds reusable parameter credit ({} > {})",
                        fn_name, name, *cnt, avail
                    ),
                    span: expr.span,
                    notes: vec![format!("pass an owned `{}` parameter to reuse its Perceus cell in place, or use `inout` mutation", name)],
                    counterexample: BTreeMap::new(),
                });
            }
        }
        ExprKind::Binary { op, left, right } => {
            if *op == BinOp::Concat {
                diags.push(Diagnostic {
                    code: "E0620".into(),
                    message: format!("`@fip` contract violated in `{}`: heap string concatenation `++`", fn_name),
                    span: expr.span,
                    notes: vec!["string concatenation allocates on the heap and violates `@fip`".into()],
                    counterexample: BTreeMap::new(),
                });
            }
            check_fip_expr(left, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
            check_fip_expr(right, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
        }
        ExprKind::Call { callee, args } => {
            for a in args {
                check_fip_expr(&a.value, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
            }
            if let ExprKind::Ident(cname) = &callee.kind {
                if !fip_fns.contains(cname)
                    && !no_heap_fns.contains(cname)
                    && !non_alloc_builtins.contains(cname.as_str())
                {
                    diags.push(Diagnostic {
                        code: "E0620".into(),
                        message: format!(
                            "`@fip` contract violated in `{}`: call to non-`@fip` function `{}` may allocate",
                            fn_name, cname
                        ),
                        span: expr.span,
                        notes: vec![format!("annotate `{}` with `@fip` or `@no_heap_alloc`", cname)],
                        counterexample: BTreeMap::new(),
                    });
                }
            }
        }
        ExprKind::If {
            cond,
            then_block,
            else_branch,
        } => {
            check_fip_expr(cond, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
            // Independent branches share the same reuse credit snapshot!
            let snap = used.clone();
            let mut u_then = snap.clone();
            check_fip_block(then_block, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, &mut u_then, diags);
            let mut u_else = snap;
            if let Some(eb) = else_branch {
                check_fip_expr(eb, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, &mut u_else, diags);
            }
            for (k, v) in u_then.into_iter().chain(u_else.into_iter()) {
                let entry = used.entry(k).or_insert(0);
                *entry = (*entry).max(v);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            check_fip_expr(scrutinee, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
            let snap = used.clone();
            for a in arms {
                let mut u_arm = snap.clone();
                if let Some(g) = &a.guard {
                    check_fip_expr(g, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, &mut u_arm, diags);
                }
                check_fip_expr(&a.body, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, &mut u_arm, diags);
                for (k, v) in u_arm {
                    let entry = used.entry(k).or_insert(0);
                    *entry = (*entry).max(v);
                }
            }
        }
        ExprKind::Unary { expr: e, .. }
        | ExprKind::Field { target: e, .. }
        | ExprKind::TupleIndex { target: e, .. }
        | ExprKind::Try(e)
        | ExprKind::Cast { expr: e, .. }
        | ExprKind::TryCast { expr: e, .. } => {
            check_fip_expr(e, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
        }
        ExprKind::Index { target, index } => {
            check_fip_expr(target, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
            check_fip_expr(index, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) => {
            check_fip_block(b, fn_name, fip_fns, no_heap_fns, non_alloc_builtins, budget, used, diags);
        }
        _ => {}
    }
}

/// Identifies local `let`/`var x = Rec { ... }` bindings whose record type contains only
/// scalar (non-RC) fields and whose ONLY uses in `f.body` are field reads `x.field` or
/// in-place scalar field writes `x.field = ...`.
/// Such locals are 100% sound to promote directly onto the LLVM stack (`alloca`), eliminating
/// heap allocation, `rc_inc`/`rc_dec`, and enabling LLVM SROA into CPU registers!
pub fn compute_stack_promotable_vars(
    f: &FnDecl,
    non_escaping: &BTreeSet<String>,
    scalar_only_records: &BTreeSet<String>,
) -> BTreeSet<String> {
    let mut candidates = BTreeSet::new();
    let mut disqualified = BTreeSet::new();
    collect_stack_candidates(&f.body, non_escaping, scalar_only_records, &mut candidates, &mut disqualified);
    check_stack_uses_block(&f.body, &candidates, &mut disqualified);
    candidates.difference(&disqualified).cloned().collect()
}

fn collect_stack_candidates(
    blk: &Block,
    non_escaping: &BTreeSet<String>,
    scalar_only_records: &BTreeSet<String>,
    candidates: &mut BTreeSet<String>,
    disqualified: &mut BTreeSet<String>,
) {
    for s in &blk.stmts {
        match s {
            Stmt::Let { name, value, .. } => {
                if non_escaping.contains(name) {
                    if let ExprKind::RecordLit { name: rname, .. } = &value.kind {
                        if scalar_only_records.contains(rname) {
                            if !candidates.insert(name.clone()) {
                                // Shadowed name -> disqualify conservatively
                                disqualified.insert(name.clone());
                            }
                            continue;
                        }
                    }
                }
                if candidates.contains(name) {
                    disqualified.insert(name.clone());
                }
            }
            Stmt::Assign { target, .. } => {
                // Whole-record reassignment (`x = ...`) disqualifies; in-place scalar field mutation (`x.f = ...`) is 100% stack-safe!
                if let ExprKind::Ident(n) = &target.kind {
                    disqualified.insert(n.clone());
                }
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } => {
                collect_stack_candidates(body, non_escaping, scalar_only_records, candidates, disqualified);
            }
            _ => {}
        }
    }
}

fn check_stack_uses_block(blk: &Block, candidates: &BTreeSet<String>, disqualified: &mut BTreeSet<String>) {
    for s in &blk.stmts {
        match s {
            Stmt::Let { value, .. } | Stmt::LetTuple { value, .. } => {
                check_stack_uses_expr(value, candidates, disqualified, false);
            }
            Stmt::Assign { target, value, .. } => {
                check_stack_uses_expr(target, candidates, disqualified, false);
                check_stack_uses_expr(value, candidates, disqualified, false);
            }
            Stmt::Return { value: Some(e), .. } | Stmt::Expr { expr: e, .. } | Stmt::Assert { cond: e, .. } => {
                check_stack_uses_expr(e, candidates, disqualified, false);
            }
            Stmt::While { cond, body, .. } => {
                check_stack_uses_expr(cond, candidates, disqualified, false);
                check_stack_uses_block(body, candidates, disqualified);
            }
            Stmt::For { iter, body, .. } => {
                check_stack_uses_expr(iter, candidates, disqualified, false);
                check_stack_uses_block(body, candidates, disqualified);
            }
            Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } => {
                check_stack_uses_block(body, candidates, disqualified);
            }
            _ => {}
        }
    }
    if let Some(t) = &blk.tail {
        check_stack_uses_expr(t, candidates, disqualified, false);
    }
}

fn check_stack_uses_expr(
    expr: &Expr,
    candidates: &BTreeSet<String>,
    disqualified: &mut BTreeSet<String>,
    in_field_read: bool,
) {
    match &expr.kind {
        ExprKind::Ident(n) => {
            if !in_field_read && candidates.contains(n) {
                disqualified.insert(n.clone());
            }
        }
        ExprKind::Field { target, .. } => {
            check_stack_uses_expr(target, candidates, disqualified, true);
        }
        ExprKind::Binary { left, right, .. } => {
            check_stack_uses_expr(left, candidates, disqualified, false);
            check_stack_uses_expr(right, candidates, disqualified, false);
        }
        ExprKind::Unary { expr: e, .. }
        | ExprKind::TupleIndex { target: e, .. }
        | ExprKind::Try(e)
        | ExprKind::Cast { expr: e, .. }
        | ExprKind::TryCast { expr: e, .. } => {
            check_stack_uses_expr(e, candidates, disqualified, false);
        }
        ExprKind::Index { target, index } => {
            check_stack_uses_expr(target, candidates, disqualified, false);
            check_stack_uses_expr(index, candidates, disqualified, false);
        }
        ExprKind::Call { callee, args } => {
            check_stack_uses_expr(callee, candidates, disqualified, false);
            for a in args {
                check_stack_uses_expr(&a.value, candidates, disqualified, false);
            }
        }
        ExprKind::RecordLit { fields, .. } => {
            for (_, v) in fields {
                check_stack_uses_expr(v, candidates, disqualified, false);
            }
        }
        ExprKind::List(es) | ExprKind::Tuple(es) => {
            for e in es {
                check_stack_uses_expr(e, candidates, disqualified, false);
            }
        }
        ExprKind::If { cond, then_block, else_branch } => {
            check_stack_uses_expr(cond, candidates, disqualified, false);
            check_stack_uses_block(then_block, candidates, disqualified);
            if let Some(eb) = else_branch {
                check_stack_uses_expr(eb, candidates, disqualified, false);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            check_stack_uses_expr(scrutinee, candidates, disqualified, false);
            for a in arms {
                if let Some(g) = &a.guard {
                    check_stack_uses_expr(g, candidates, disqualified, false);
                }
                check_stack_uses_expr(&a.body, candidates, disqualified, false);
            }
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) => check_stack_uses_block(b, candidates, disqualified),
        ExprKind::Lambda { body, .. } => check_stack_uses_expr(body, candidates, disqualified, false),
        _ => {}
    }
}

/// Verifies that an `@arena` function safely uses a scoped Tier-2 64 KiB Slab Arena:
/// 1. Cannot return a heap-allocated (RC) value (arena objects are bulk-freed on function exit).
/// 2. Cannot accept `inout` RC parameters (mutating them could store arena pointers outside the arena).
/// 3. Cannot spawn threads or send objects over channels inside the arena frame.
pub fn verify_arena_function(
    f: &FnDecl,
    is_rc_type: &dyn Fn(&TypeExpr) -> bool,
) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    if let Some(rt) = &f.ret_ty {
        if is_rc_type(rt) {
            diags.push(Diagnostic {
                code: "E0621".into(),
                message: format!(
                    "`@arena` contract violated in `{}`: cannot return heap type `{}` from a scoped slab arena",
                    f.name,
                    rt.display_name()
                ),
                span: f.span,
                notes: vec![
                    "Tier-2 `@arena` bulk-frees all allocations on function exit; return a scalar/primitive summary or remove `@arena`".into(),
                ],
                counterexample: BTreeMap::new(),
            });
        }
    }
    for p in &f.params {
        if p.is_inout && is_rc_type(&p.ty) {
            diags.push(Diagnostic {
                code: "E0621".into(),
                message: format!(
                    "`@arena` contract violated in `{}`: `inout` heap parameter `{}` could escape arena pointers",
                    f.name, p.name
                ),
                span: p.span,
                notes: vec!["only read-only or scalar `inout` parameters are allowed in `@arena` functions".into()],
                counterexample: BTreeMap::new(),
            });
        }
    }
    check_arena_escape_block(&f.body, &f.name, &mut diags);
    diags
}

fn check_arena_escape_block(blk: &Block, fn_name: &str, diags: &mut Vec<Diagnostic>) {
    for s in &blk.stmts {
        match s {
            Stmt::Let { value, .. } | Stmt::LetTuple { value, .. } => {
                check_arena_escape_expr(value, fn_name, diags);
            }
            Stmt::Assign { target, value, .. } => {
                check_arena_escape_expr(target, fn_name, diags);
                check_arena_escape_expr(value, fn_name, diags);
            }
            Stmt::Return { value: Some(e), .. } | Stmt::Expr { expr: e, .. } | Stmt::Assert { cond: e, .. } => {
                check_arena_escape_expr(e, fn_name, diags);
            }
            Stmt::While { cond, body, .. } => {
                check_arena_escape_expr(cond, fn_name, diags);
                check_arena_escape_block(body, fn_name, diags);
            }
            Stmt::For { iter, body, .. } => {
                check_arena_escape_expr(iter, fn_name, diags);
                check_arena_escape_block(body, fn_name, diags);
            }
            Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } => {
                check_arena_escape_block(body, fn_name, diags);
            }
            _ => {}
        }
    }
    if let Some(t) = &blk.tail {
        check_arena_escape_expr(t, fn_name, diags);
    }
}

fn check_arena_escape_expr(expr: &Expr, fn_name: &str, diags: &mut Vec<Diagnostic>) {
    match &expr.kind {
        ExprKind::Call { callee, args } => {
            if let ExprKind::Ident(c) = &callee.kind {
                if matches!(c.as_str(), "thread_spawn" | "channel_send" | "channel_try_send") {
                    diags.push(Diagnostic {
                        code: "E0621".into(),
                        message: format!(
                            "`@arena` contract violated in `{}`: `{}` may escape arena-allocated values across threads",
                            fn_name, c
                        ),
                        span: expr.span,
                        notes: vec!["do not spawn threads or send values on channels inside an `@arena` scope".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
            }
            for a in args {
                check_arena_escape_expr(&a.value, fn_name, diags);
            }
        }
        ExprKind::If { cond, then_block, else_branch } => {
            check_arena_escape_expr(cond, fn_name, diags);
            check_arena_escape_block(then_block, fn_name, diags);
            if let Some(eb) = else_branch {
                check_arena_escape_expr(eb, fn_name, diags);
            }
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) => check_arena_escape_block(b, fn_name, diags),
        ExprKind::Binary { left, right, .. } => {
            check_arena_escape_expr(left, fn_name, diags);
            check_arena_escape_expr(right, fn_name, diags);
        }
        ExprKind::Unary { expr: e, .. } | ExprKind::Field { target: e, .. } | ExprKind::Try(e) => {
            check_arena_escape_expr(e, fn_name, diags);
        }
        _ => {}
    }
}

/// Verifies that a `@checkfree` function has ZERO unproved runtime checks:
/// - Every `xs[i]` index read/write must be in `proved_safe_indices`
/// - Every integer `+`, `-`, `*` must be in `proved_no_overflow`
/// - Every integer `/`, `%` must be in `proved_nonzero_divs`
/// - Every `x as? T` fallible cast must be in `infallible_casts`
/// Emits `E0622` for any operation requiring a runtime safety check.
pub fn verify_checkfree_function(
    f: &FnDecl,
    proved_safe_indices: &BTreeSet<usize>,
    proved_no_overflow: &BTreeSet<usize>,
    proved_nonzero_divs: &BTreeSet<usize>,
    infallible_casts: &BTreeSet<usize>,
    is_float_expr: &dyn Fn(&Expr) -> bool,
) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    check_checkfree_block(
        &f.body,
        &f.name,
        proved_safe_indices,
        proved_no_overflow,
        proved_nonzero_divs,
        infallible_casts,
        is_float_expr,
        &mut diags,
    );
    diags
}

#[allow(clippy::too_many_arguments)]
fn check_checkfree_block(
    blk: &Block,
    fn_name: &str,
    safe_idx: &BTreeSet<usize>,
    no_ovf: &BTreeSet<usize>,
    nz_div: &BTreeSet<usize>,
    infallible: &BTreeSet<usize>,
    is_float: &dyn Fn(&Expr) -> bool,
    diags: &mut Vec<Diagnostic>,
) {
    for s in &blk.stmts {
        match s {
            Stmt::Let { value, .. } | Stmt::LetTuple { value, .. } => {
                check_checkfree_expr(value, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
            }
            Stmt::Assign { target, op, value, .. } => {
                check_checkfree_expr(target, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
                check_checkfree_expr(value, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
                if matches!(op, AssignOp::AddSet | AssignOp::SubSet | AssignOp::MulSet)
                    && !is_float(target)
                    && !is_float(value)
                    && !no_ovf.contains(&target.span.start)
                    && !no_ovf.contains(&value.span.start)
                {
                    // Compound assignment like `i += 1` records its synthetic Binary at `target.span.start` or `stmt.span.start`;
                    // allow if either target or value span was proved overflow-free.
                }
            }
            Stmt::Return { value: Some(e), .. } | Stmt::Expr { expr: e, .. } | Stmt::Assert { cond: e, .. } => {
                check_checkfree_expr(e, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
            }
            Stmt::While { cond, body, .. } => {
                check_checkfree_expr(cond, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
                check_checkfree_block(body, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
            }
            Stmt::For { iter, body, .. } => {
                check_checkfree_expr(iter, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
                check_checkfree_block(body, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
            }
            Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } => {
                check_checkfree_block(body, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
            }
            Stmt::Ghost { .. } | Stmt::Return { value: None, .. } | Stmt::Break { .. } | Stmt::Continue { .. } => {}
        }
    }
    if let Some(t) = &blk.tail {
        check_checkfree_expr(t, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
    }
}

#[allow(clippy::too_many_arguments)]
fn check_checkfree_expr(
    expr: &Expr,
    fn_name: &str,
    safe_idx: &BTreeSet<usize>,
    no_ovf: &BTreeSet<usize>,
    nz_div: &BTreeSet<usize>,
    infallible: &BTreeSet<usize>,
    is_float: &dyn Fn(&Expr) -> bool,
    diags: &mut Vec<Diagnostic>,
) {
    match &expr.kind {
        ExprKind::Index { target, index } => {
            if !safe_idx.contains(&expr.span.start) {
                diags.push(Diagnostic {
                    code: "E0622".into(),
                    message: format!(
                        "`@checkfree` contract violated in `{}`: unproved runtime bounds check on index access",
                        fn_name
                    ),
                    span: expr.span,
                    notes: vec![
                        "add a loop `invariant 0 <= i && i <= len(xs)` or `requires` bound so bounds check is statically elided".into(),
                    ],
                    counterexample: BTreeMap::new(),
                });
            }
            check_checkfree_expr(target, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
            check_checkfree_expr(index, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
        }
        ExprKind::Binary { op, left, right } => {
            let flt = is_float(left) || is_float(right);
            if !flt && matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul) && !no_ovf.contains(&expr.span.start) {
                diags.push(Diagnostic {
                    code: "E0622".into(),
                    message: format!(
                        "`@checkfree` contract violated in `{}`: integer `{:?}` is not statically proved overflow-free",
                        fn_name, op
                    ),
                    span: expr.span,
                    notes: vec![
                        "constrain operand ranges with `where` / `requires` or use `wrapping_add` / `wrapping_sub` / `wrapping_mul`".into(),
                    ],
                    counterexample: BTreeMap::new(),
                });
            }
            if !flt && matches!(op, BinOp::Div | BinOp::Mod) && !nz_div.contains(&expr.span.start) {
                diags.push(Diagnostic {
                    code: "E0622".into(),
                    message: format!(
                        "`@checkfree` contract violated in `{}`: divisor is not statically proved non-zero",
                        fn_name
                    ),
                    span: expr.span,
                    notes: vec!["constrain divisor with `where self != 0`".into()],
                    counterexample: BTreeMap::new(),
                });
            }
            check_checkfree_expr(left, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
            check_checkfree_expr(right, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
        }
        ExprKind::TryCast { expr: inner, .. } => {
            if !infallible.contains(&expr.span.start) {
                diags.push(Diagnostic {
                    code: "E0622".into(),
                    message: format!(
                        "`@checkfree` contract violated in `{}`: fallible runtime refinement cast `as?` cannot be eliminated",
                        fn_name
                    ),
                    span: expr.span,
                    notes: vec!["narrow the source refinement so `src <: dst` is statically proved infallible".into()],
                    counterexample: BTreeMap::new(),
                });
            }
            check_checkfree_expr(inner, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
        }
        ExprKind::Unary { expr: e, .. }
        | ExprKind::Field { target: e, .. }
        | ExprKind::TupleIndex { target: e, .. }
        | ExprKind::Try(e)
        | ExprKind::Await(e)
        | ExprKind::Cast { expr: e, .. }
        | ExprKind::ArrayRepeat { value: e, .. }
        | ExprKind::GenericInst { expr: e, .. } => {
            check_checkfree_expr(e, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
        }
        ExprKind::Call { callee, args } => {
            check_checkfree_expr(callee, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
            for a in args {
                check_checkfree_expr(&a.value, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
            }
        }
        ExprKind::RecordLit { fields, .. } => {
            for (_, v) in fields {
                check_checkfree_expr(v, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
            }
        }
        ExprKind::List(es) | ExprKind::Tuple(es) => {
            for e in es {
                check_checkfree_expr(e, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
            }
        }
        ExprKind::If { cond, then_block, else_branch } => {
            check_checkfree_expr(cond, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
            check_checkfree_block(then_block, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
            if let Some(eb) = else_branch {
                check_checkfree_expr(eb, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            check_checkfree_expr(scrutinee, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
            for a in arms {
                if let Some(g) = &a.guard {
                    check_checkfree_expr(g, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
                }
                check_checkfree_expr(&a.body, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
            }
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) => {
            check_checkfree_block(b, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
        }
        ExprKind::Lambda { body, .. } => {
            check_checkfree_expr(body, fn_name, safe_idx, no_ovf, nz_div, infallible, is_float, diags);
        }
        _ => {}
    }
}

/// Extracts a structural memory access path (`r.a.b` -> `["r", "a", "b"]`, `xs[i]` -> `["xs", "*"]`)
/// for Law-of-Exclusivity aliasing checks (`E0820`).
pub fn extract_access_path(expr: &Expr) -> Option<Vec<String>> {
    match &expr.kind {
        ExprKind::Ident(s) => Some(vec![s.clone()]),
        ExprKind::Field { target, field } => {
            let mut p = extract_access_path(target)?;
            p.push(field.clone());
            Some(p)
        }
        ExprKind::TupleIndex { target, index } => {
            let mut p = extract_access_path(target)?;
            p.push(index.to_string());
            Some(p)
        }
        ExprKind::Index { target, .. } => {
            let mut p = extract_access_path(target)?;
            p.push("*".into());
            Some(p)
        }
        _ => None,
    }
}

/// Returns `true` iff two memory access paths may alias (prefix overlap or wildcard index).
pub fn access_paths_overlap(a: &[String], b: &[String]) -> bool {
    if a.is_empty() || b.is_empty() || a[0] != b[0] {
        return false;
    }
    let min_len = a.len().min(b.len());
    for i in 0..min_len {
        if a[i] == "*" || b[i] == "*" {
            return true;
        }
        if a[i] != b[i] {
            return false;
        }
    }
    true
}

/// Verifies `@linear` / `@must_use` resource token linearity across a function body:
/// Checks whether a linear type name represents either a direct `@linear` record `T`
/// or a linear-carrying wrapper `Result[T]` / `Option[T]`.
fn is_linear_ty_name(tname: &str, linear_records: &BTreeSet<String>) -> bool {
    if linear_records.contains(tname) {
        return true;
    }
    if let Some(inner) = tname
        .strip_prefix("Result[")
        .and_then(|s| s.strip_suffix(']'))
    {
        return linear_records.contains(inner);
    }
    if let Some(inner) = tname
        .strip_prefix("Option[")
        .and_then(|s| s.strip_suffix(']'))
    {
        return linear_records.contains(inner);
    }
    false
}

/// Extracts the inner linear type name from `Result[T]` / `Option[T]` when matching `Ok(v)` / `Some(v)`.
fn unwrap_linear_wrapper_for_variant(scrut_ty: &str, variant_name: &str) -> Option<String> {
    let short = variant_name.rsplit('.').next().unwrap_or(variant_name);
    if short == "Ok" {
        if let Some(inner) = scrut_ty
            .strip_prefix("Result[")
            .and_then(|s| s.strip_suffix(']'))
        {
            return Some(inner.to_string());
        }
    }
    if short == "Some" {
        if let Some(inner) = scrut_ty
            .strip_prefix("Option[")
            .and_then(|s| s.strip_suffix(']'))
        {
            return Some(inner.to_string());
        }
    }
    None
}

/// Linear / Must-Use Typestate Verification (`@linear` / `@must_use`):
/// every binding of a `@linear`/`@must_use` record type (or `Result[LinearT, E]` / `Option[LinearT]`)
/// must be consumed exactly once (passed to a consuming function/method, matched & consumed,
/// moved into a struct/enum, or returned) before scope exit.
/// Borrowing via `inout` parameter or `@borrow` method keeps the resource live without consuming it.
/// Emits `E0821` on dropped unconsumed tokens or use-after-consume.
pub fn verify_linear_tokens(
    f: &FnDecl,
    linear_records: &BTreeSet<String>,
    infer_ty_name: &dyn Fn(&Expr, &BTreeMap<String, String>) -> Option<String>,
    is_param_borrowing: &dyn Fn(&Expr, usize, &BTreeMap<String, String>) -> bool,
) -> Vec<Diagnostic> {
    if linear_records.is_empty() {
        return Vec::new();
    }
    let mut diags = Vec::new();
    let mut live: BTreeMap<String, (String, Span)> = BTreeMap::new();
    let mut consumed: BTreeMap<String, (String, Span)> = BTreeMap::new();
    let mut var_ty_names: BTreeMap<String, String> = BTreeMap::new();
    let is_consumer_fn = f.attrs.iter().any(|a| a.name == "consume");
    let is_borrow_fn = f.attrs.iter().any(|a| a.name == "borrow");
    for p in &f.params {
        let tn = type_expr_to_linear_name(&p.ty, linear_records).unwrap_or_default();
        if !tn.is_empty() {
            var_ty_names.insert(p.name.clone(), tn.clone());
        }
        let is_own_method = !tn.is_empty()
            && (f.name.starts_with(&format!("{}.", tn))
                || f.name.starts_with(&format!("{}::", tn)));
        if !p.is_inout
            && !is_borrow_fn
            && is_linear_ty_name(&tn, linear_records)
            && !is_consumer_fn
            && !is_own_method
        {
            live.insert(p.name.clone(), (tn, p.span));
        }
    }
    let mut returned = false;
    check_linear_block(
        &f.body,
        &f.name,
        linear_records,
        infer_ty_name,
        is_param_borrowing,
        &mut var_ty_names,
        &mut live,
        &mut consumed,
        &mut returned,
        &mut diags,
    );
    if !returned {
        for (v, (ty, sp)) in &live {
            diags.push(Diagnostic {
                code: "E0821".into(),
                message: format!(
                    "linear resource `{}` of type `{}` in `{}` is dropped without being consumed",
                    v, ty, f.name
                ),
                span: *sp,
                notes: vec![
                    format!(
                        "`{}` is marked `@linear` / `@must_use`: pass it to a consuming function/method or return it",
                        ty
                    ),
                ],
                counterexample: BTreeMap::new(),
            });
        }
    }
    diags
}

pub fn type_expr_to_linear_name(
    ty: &TypeExpr,
    linear_records: &BTreeSet<String>,
) -> Option<String> {
    if let TypeExpr::Named { name, args, .. } = ty {
        if linear_records.contains(name) {
            return Some(name.clone());
        }
        if (name == "Result" || name == "Option") && !args.is_empty() {
            if let TypeExpr::Named { name: inner, .. } = &args[0] {
                if linear_records.contains(inner) {
                    return Some(format!("{}[{}]", name, inner));
                }
            }
        }
        return Some(name.clone());
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn check_linear_block(
    blk: &Block,
    fn_name: &str,
    linear_records: &BTreeSet<String>,
    infer_ty_name: &dyn Fn(&Expr, &BTreeMap<String, String>) -> Option<String>,
    is_param_borrowing: &dyn Fn(&Expr, usize, &BTreeMap<String, String>) -> bool,
    var_ty_names: &mut BTreeMap<String, String>,
    live: &mut BTreeMap<String, (String, Span)>,
    consumed: &mut BTreeMap<String, (String, Span)>,
    returned: &mut bool,
    diags: &mut Vec<Diagnostic>,
) {
    let mut block_locals: Vec<String> = Vec::new();
    for s in &blk.stmts {
        match s {
            Stmt::Let { name, ty, value, span, .. } => {
                consume_linear_expr(value, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, true, diags);
                let tn = ty
                    .as_ref()
                    .and_then(|t| type_expr_to_linear_name(t, linear_records))
                    .or_else(|| infer_ty_name(value, var_ty_names));
                if let Some(tname) = tn {
                    var_ty_names.insert(name.clone(), tname.clone());
                    if is_linear_ty_name(&tname, linear_records) {
                        if name == "_" {
                            diags.push(Diagnostic {
                                code: "E0821".into(),
                                message: format!(
                                    "cannot discard `@linear` / `@must_use` value of type `{}` with `_` in `{}`",
                                    tname, fn_name
                                ),
                                span: *span,
                                notes: vec!["bind and explicitly consume or return the linear resource".into()],
                                counterexample: BTreeMap::new(),
                            });
                        } else {
                            live.insert(name.clone(), (tname, *span));
                            consumed.remove(name);
                            block_locals.push(name.clone());
                        }
                    }
                }
            }
            Stmt::Assign { target, value, .. } => {
                consume_linear_expr(value, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, true, diags);
                if let ExprKind::Ident(n) = &target.kind {
                    if let Some((ty, sp)) = live.get(n) {
                        diags.push(Diagnostic {
                            code: "E0821".into(),
                            message: format!(
                                "cannot overwrite unconsumed linear resource `{}` of type `{}` in `{}`",
                                n, ty, fn_name
                            ),
                            span: *sp,
                            notes: vec!["consume the previous linear handle before reassigning".into()],
                            counterexample: BTreeMap::new(),
                        });
                    }
                }
            }
            Stmt::Return { value, .. } => {
                if let Some(v) = value {
                    consume_linear_expr(v, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, true, diags);
                }
                for (v, (ty, sp)) in live.iter() {
                    diags.push(Diagnostic {
                        code: "E0821".into(),
                        message: format!(
                            "early return in `{}` drops unconsumed linear resource `{}` of type `{}`",
                            fn_name, v, ty
                        ),
                        span: *sp,
                        notes: vec!["consume or return all `@linear` handles on every return path".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
                live.clear();
                *returned = true;
                return;
            }
            Stmt::Expr { expr, span } => {
                consume_linear_expr(expr, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, false, diags);
                if let Some(tname) = infer_ty_name(expr, var_ty_names) {
                    if is_linear_ty_name(&tname, linear_records) {
                        diags.push(Diagnostic {
                            code: "E0821".into(),
                            message: format!(
                                "unused `@linear` / `@must_use` expression value of type `{}` in `{}`",
                                tname, fn_name
                            ),
                            span: *span,
                            notes: vec!["bind and explicitly consume or return this linear resource".into()],
                            counterexample: BTreeMap::new(),
                        });
                    }
                }
            }
            Stmt::Assert { cond, .. } => {
                consume_linear_expr(cond, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, false, diags);
            }
            Stmt::While { cond, body, .. } => {
                consume_linear_expr(cond, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, false, diags);
                let mut r_sub = false;
                check_linear_block(body, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, &mut r_sub, diags);
            }
            Stmt::For { iter, body, .. } => {
                consume_linear_expr(iter, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, false, diags);
                let mut r_sub = false;
                check_linear_block(body, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, &mut r_sub, diags);
            }
            Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } => {
                let mut r_sub = false;
                check_linear_block(body, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, &mut r_sub, diags);
            }
            _ => {}
        }
    }
    if let Some(t) = &blk.tail {
        consume_linear_expr(t, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, true, diags);
    }
    for local in block_locals {
        if let Some((ty, sp)) = live.remove(&local) {
            diags.push(Diagnostic {
                code: "E0821".into(),
                message: format!(
                    "linear resource `{}` of type `{}` in `{}` is dropped without being consumed",
                    local, ty, fn_name
                ),
                span: sp,
                notes: vec![
                    format!(
                        "`{}` is marked `@linear` / `@must_use`: pass it to a consuming function/method or return it",
                        ty
                    ),
                ],
                counterexample: BTreeMap::new(),
            });
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn consume_linear_expr(
    expr: &Expr,
    fn_name: &str,
    linear_records: &BTreeSet<String>,
    infer_ty_name: &dyn Fn(&Expr, &BTreeMap<String, String>) -> Option<String>,
    is_param_borrowing: &dyn Fn(&Expr, usize, &BTreeMap<String, String>) -> bool,
    var_ty_names: &mut BTreeMap<String, String>,
    live: &mut BTreeMap<String, (String, Span)>,
    consumed: &mut BTreeMap<String, (String, Span)>,
    in_move_pos: bool,
    diags: &mut Vec<Diagnostic>,
) {
    match &expr.kind {
        ExprKind::Ident(s) => {
            if let Some((ty, _)) = consumed.get(s) {
                diags.push(Diagnostic {
                    code: "E0821".into(),
                    message: format!(
                        "use-after-consume of linear resource `{}` of type `{}` in `{}`",
                        s, ty, fn_name
                    ),
                    span: expr.span,
                    notes: vec!["a `@linear` resource can be consumed at most once".into()],
                    counterexample: BTreeMap::new(),
                });
            } else if in_move_pos {
                if let Some((ty, sp)) = live.remove(s) {
                    consumed.insert(s.clone(), (ty, sp));
                }
            }
        }
        ExprKind::Field { target, .. } | ExprKind::TupleIndex { target, .. } => {
            // Reading a field (`tok.fd`) borrows the token; still reject use-after-consume!
            consume_linear_expr(target, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, false, diags);
        }
        ExprKind::Call { callee, args } => {
            // Distinguish borrowing (`inout` param / `@borrow` fn/method) vs consuming (by-value)!
            if let ExprKind::Field { target, .. } = &callee.kind {
                let recv_borrows = is_param_borrowing(expr, 0, var_ty_names);
                consume_linear_expr(target, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, !recv_borrows, diags);
                for (i, a) in args.iter().enumerate() {
                    let arg_borrows = is_param_borrowing(expr, i + 1, var_ty_names);
                    consume_linear_expr(&a.value, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, !arg_borrows, diags);
                }
            } else {
                consume_linear_expr(callee, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, false, diags);
                for (i, a) in args.iter().enumerate() {
                    let arg_borrows = is_param_borrowing(expr, i, var_ty_names);
                    consume_linear_expr(&a.value, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, !arg_borrows, diags);
                }
            }
        }
        ExprKind::RecordLit { fields, .. } => {
            for (_, v) in fields {
                consume_linear_expr(v, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, true, diags);
            }
        }
        ExprKind::List(es) | ExprKind::Tuple(es) => {
            for e in es {
                consume_linear_expr(e, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, true, diags);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            let scrut_ty = infer_ty_name(scrutinee, var_ty_names);
            let scrut_is_linear = scrut_ty
                .as_deref()
                .is_some_and(|t| is_linear_ty_name(t, linear_records));
            consume_linear_expr(
                scrutinee,
                fn_name,
                linear_records,
                infer_ty_name,
                is_param_borrowing,
                var_ty_names,
                live,
                consumed,
                scrut_is_linear,
                diags,
            );
            let pre_live = live.clone();
            let mut merged_live: BTreeMap<String, (String, Span)> = BTreeMap::new();
            let mut all_cons: Vec<BTreeMap<String, (String, Span)>> = Vec::new();
            let mut any_non_returning = false;
            for arm in arms {
                let mut arm_live = pre_live.clone();
                let mut arm_cons = consumed.clone();
                let mut arm_tys = var_ty_names.clone();
                let mut arm_locals: Vec<String> = Vec::new();
                if let Some(ref sty) = scrut_ty {
                    for pat in &arm.patterns {
                        match pat {
                            Pattern::Variant { name, binding: Some(b), span } => {
                                if let Some(inner_lin) = unwrap_linear_wrapper_for_variant(sty, name) {
                                    arm_tys.insert(b.clone(), inner_lin.clone());
                                    if b == "_" {
                                        diags.push(Diagnostic {
                                            code: "E0821".into(),
                                            message: format!(
                                                "cannot discard `@linear` payload of `{}` in `match` inside `{}`",
                                                inner_lin, fn_name
                                            ),
                                            span: *span,
                                            notes: vec!["bind and explicitly consume or return the linear payload".into()],
                                            counterexample: BTreeMap::new(),
                                        });
                                    } else {
                                        arm_live.insert(b.clone(), (inner_lin, *span));
                                        arm_cons.remove(b);
                                        arm_locals.push(b.clone());
                                    }
                                }
                            }
                            Pattern::Ident(b, span) if scrut_is_linear => {
                                arm_tys.insert(b.clone(), sty.clone());
                                arm_live.insert(b.clone(), (sty.clone(), *span));
                                arm_cons.remove(b);
                                arm_locals.push(b.clone());
                            }
                            Pattern::Wildcard(span) if scrut_is_linear => {
                                // Only flag wildcard discard if scrutinee is a direct linear type
                                // (for Result[T]/Option[T], `_` may match Err/None after Ok/Some!)
                                if linear_records.contains(sty) {
                                    diags.push(Diagnostic {
                                        code: "E0821".into(),
                                        message: format!(
                                            "cannot discard `@linear` value of type `{}` with `_` in `match` inside `{}`",
                                            sty, fn_name
                                        ),
                                        span: *span,
                                        notes: vec!["bind and consume the linear value".into()],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                }
                let mut arm_ret = false;
                if let ExprKind::Block(ablk) = &arm.body.kind {
                    check_linear_block(
                        ablk,
                        fn_name,
                        linear_records,
                        infer_ty_name,
                        is_param_borrowing,
                        &mut arm_tys,
                        &mut arm_live,
                        &mut arm_cons,
                        &mut arm_ret,
                        diags,
                    );
                } else {
                    consume_linear_expr(
                        &arm.body,
                        fn_name,
                        linear_records,
                        infer_ty_name,
                        is_param_borrowing,
                        &mut arm_tys,
                        &mut arm_live,
                        &mut arm_cons,
                        in_move_pos,
                        diags,
                    );
                }
                if !arm_ret {
                    for local in arm_locals {
                        if let Some((ty, sp)) = arm_live.remove(&local) {
                            diags.push(Diagnostic {
                                code: "E0821".into(),
                                message: format!(
                                    "linear resource `{}` of type `{}` bound in `match` arm of `{}` is dropped without being consumed",
                                    local, ty, fn_name
                                ),
                                span: sp,
                                notes: vec![format!("consume or return `{}` inside the `match` arm", local)],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                    any_non_returning = true;
                    merged_live.extend(arm_live);
                    all_cons.push(arm_cons);
                }
            }
            if any_non_returning {
                for cmap in all_cons {
                    for (k, v) in cmap {
                        if !merged_live.contains_key(&k) {
                            consumed.insert(k, v);
                        }
                    }
                }
                *live = merged_live;
            }
        }
        ExprKind::If { cond, then_block, else_branch } => {
            consume_linear_expr(cond, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, false, diags);
            let mut live_t = live.clone();
            let mut cons_t = consumed.clone();
            let mut ret_t = false;
            check_linear_block(then_block, fn_name, linear_records, infer_ty_name, is_param_borrowing, &mut var_ty_names.clone(), &mut live_t, &mut cons_t, &mut ret_t, diags);
            if let Some(eb) = else_branch {
                let mut live_e = live.clone();
                let mut cons_e = consumed.clone();
                let mut ret_e = false;
                if let ExprKind::Block(eblk) = &eb.kind {
                    check_linear_block(eblk, fn_name, linear_records, infer_ty_name, is_param_borrowing, &mut var_ty_names.clone(), &mut live_e, &mut cons_e, &mut ret_e, diags);
                } else {
                    consume_linear_expr(eb, fn_name, linear_records, infer_ty_name, is_param_borrowing, &mut var_ty_names.clone(), &mut live_e, &mut cons_e, in_move_pos, diags);
                }
                // After `if/else`, a token is still unconsumed if it remains live on EITHER non-returning branch.
                let mut merged_live = BTreeMap::new();
                if !ret_t {
                    merged_live.extend(live_t);
                }
                if !ret_e {
                    merged_live.extend(live_e);
                }
                for (k, v) in cons_t {
                    if !merged_live.contains_key(&k) {
                        consumed.insert(k, v);
                    }
                }
                for (k, v) in cons_e {
                    if !merged_live.contains_key(&k) {
                        consumed.insert(k, v);
                    }
                }
                *live = merged_live;
            } else if !ret_t {
                // `if cond { consume(tok) }` without `else` leaves `tok` consumed on one path and live on the other!
                for (k, (ty, sp)) in live.iter() {
                    if !live_t.contains_key(k) {
                        diags.push(Diagnostic {
                            code: "E0821".into(),
                            message: format!(
                                "linear resource `{}` of type `{}` is consumed inside `if` branch without `else` in `{}`",
                                k, ty, fn_name
                            ),
                            span: *sp,
                            notes: vec!["consume the resource on both `if` and `else` branches or return inside the `if` branch".into()],
                            counterexample: BTreeMap::new(),
                        });
                    }
                }
            }
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) => {
            let mut r = false;
            check_linear_block(b, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, &mut r, diags);
        }
        ExprKind::Binary { left, right, .. } => {
            consume_linear_expr(left, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, false, diags);
            consume_linear_expr(right, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, false, diags);
        }
        ExprKind::Unary { expr: e, .. }
        | ExprKind::Try(e)
        | ExprKind::Await(e)
        | ExprKind::Cast { expr: e, .. }
        | ExprKind::TryCast { expr: e, .. } => {
            consume_linear_expr(e, fn_name, linear_records, infer_ty_name, is_param_borrowing, var_ty_names, live, consumed, in_move_pos, diags);
        }
        _ => {}
    }
}
