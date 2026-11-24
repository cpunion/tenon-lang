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
    pub is_no_heap_alloc: bool,
    pub param_count: usize,
    pub non_escaping_locals: Vec<String>,
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
    /// Functions verified to satisfy `@fip` (Functional-In-Place: zero net heap allocation).
    #[serde(default)]
    pub fip_verified_fns: BTreeSet<String>,
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
        Some(ScalarKind::F32) | Some(ScalarKind::F64) => (i64::MIN as i128, i64::MAX as i128),
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
                        mark_all_mentioned(&args[1].value, cands, escaped);
                    }
                } else {
                    mark_all_mentioned(value, cands, escaped);
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
