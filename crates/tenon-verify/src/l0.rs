use std::collections::{BTreeMap, BTreeSet};
use tenon_syntax::*;

/// Closed integer interval [min, max] in mathematical integers (i128 to avoid overflow during analysis).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval {
    pub min: i128,
    pub max: i128,
}

impl Interval {
    pub const FULL_I64: Interval = Interval {
        min: i64::MIN as i128,
        max: i64::MAX as i128,
    };

    pub fn exact(v: i128) -> Self {
        Self { min: v, max: v }
    }

    pub fn is_empty(&self) -> bool {
        self.min > self.max
    }

    pub fn intersect(self, other: Interval) -> Interval {
        Interval {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        }
    }

    pub fn join(self, other: Interval) -> Interval {
        Interval {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    pub fn fits_i64(&self) -> bool {
        self.min >= i64::MIN as i128 && self.max <= i64::MAX as i128
    }

    pub fn add(self, other: Interval) -> Interval {
        Interval {
            min: self.min.saturating_add(other.min),
            max: self.max.saturating_add(other.max),
        }
    }

    pub fn sub(self, other: Interval) -> Interval {
        Interval {
            min: self.min.saturating_sub(other.max),
            max: self.max.saturating_sub(other.min),
        }
    }

    pub fn mul(self, other: Interval) -> Interval {
        let products = [
            self.min.saturating_mul(other.min),
            self.min.saturating_mul(other.max),
            self.max.saturating_mul(other.min),
            self.max.saturating_mul(other.max),
        ];
        Interval {
            min: *products.iter().min().unwrap(),
            max: *products.iter().max().unwrap(),
        }
    }
}

/// Congruence domain: `x ≡ rem (mod modulus)`. `modulus == 1` means TOP (unknown).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Congruence {
    pub modulus: i128,
    pub rem: i128,
}

impl Congruence {
    pub const TOP: Congruence = Congruence { modulus: 1, rem: 0 };

    pub fn new(modulus: i128, rem: i128) -> Self {
        let m = modulus.abs().max(1);
        let r = rem.rem_euclid(m);
        Self { modulus: m, rem: r }
    }

    pub fn exact(v: i128) -> Self {
        Self {
            modulus: 0, // 0 denotes exact constant `rem`
            rem: v,
        }
    }

    pub fn gcd(mut a: i128, mut b: i128) -> i128 {
        a = a.abs();
        b = b.abs();
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
        }
        a
    }

    pub fn add(self, other: Congruence) -> Congruence {
        if self.modulus == 0 && other.modulus == 0 {
            return Congruence::exact(self.rem.saturating_add(other.rem));
        }
        let m = Self::gcd(self.modulus, other.modulus);
        if m == 0 {
            Congruence::TOP
        } else {
            Congruence::new(m, self.rem + other.rem)
        }
    }

    pub fn sub(self, other: Congruence) -> Congruence {
        if self.modulus == 0 && other.modulus == 0 {
            return Congruence::exact(self.rem.saturating_sub(other.rem));
        }
        let m = Self::gcd(self.modulus, other.modulus);
        if m == 0 {
            Congruence::TOP
        } else {
            Congruence::new(m, self.rem - other.rem)
        }
    }

    pub fn mul(self, other: Congruence) -> Congruence {
        if self.modulus == 0 && other.modulus == 0 {
            return Congruence::exact(self.rem.saturating_mul(other.rem));
        }
        if self.modulus == 0 {
            let c = self.rem;
            return Congruence::new((c.abs() * other.modulus).max(1), c * other.rem);
        }
        if other.modulus == 0 {
            let c = other.rem;
            return Congruence::new((c.abs() * self.modulus).max(1), c * self.rem);
        }
        let g = Self::gcd(
            Self::gcd(self.modulus * other.modulus, self.modulus * other.rem),
            other.modulus * self.rem,
        );
        if g == 0 {
            Congruence::TOP
        } else {
            Congruence::new(g, self.rem * other.rem)
        }
    }

    pub fn implies_mod_eq(self, m: i128, r: i128) -> Option<bool> {
        if m <= 0 {
            return None;
        }
        let target = r.rem_euclid(m);
        if self.modulus == 0 {
            return Some(self.rem.rem_euclid(m) == target);
        }
        if self.modulus % m == 0 {
            Some(self.rem.rem_euclid(m) == target)
        } else {
            None
        }
    }

    pub fn meet(self, other: Congruence) -> Congruence {
        if self.modulus == 0 {
            return self;
        }
        if other.modulus == 0 {
            return other;
        }
        if self.modulus >= other.modulus {
            self
        } else {
            other
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbsInt {
    pub interval: Interval,
    pub cong: Congruence,
    pub non_zero: bool,
}

impl AbsInt {
    pub fn top() -> Self {
        Self {
            interval: Interval::FULL_I64,
            cong: Congruence::TOP,
            non_zero: false,
        }
    }

    pub fn exact(v: i128) -> Self {
        Self {
            interval: Interval::exact(v),
            cong: Congruence::exact(v),
            non_zero: v != 0,
        }
    }

    pub fn refine_interval(mut self, iv: Interval) -> Self {
        self.interval = self.interval.intersect(iv);
        if self.interval.min > 0 || self.interval.max < 0 {
            self.non_zero = true;
        }
        if self.interval.min == self.interval.max {
            self.cong = Congruence::exact(self.interval.min);
        }
        self
    }

    pub fn refine_cong(mut self, c: Congruence) -> Self {
        self.cong = self.cong.meet(c);
        if c.modulus > 1 && c.rem != 0 {
            self.non_zero = true;
        }
        self
    }

    pub fn join(self, other: Self) -> Self {
        let interval = self.interval.join(other.interval);
        let cong = if self.cong == other.cong {
            self.cong
        } else {
            Congruence::TOP
        };
        let non_zero = (self.non_zero && other.non_zero)
            || interval.min > 0
            || interval.max < 0;
        Self {
            interval,
            cong,
            non_zero,
        }
    }
}

/// Abstract environment for L0 fast verification.
#[derive(Debug, Clone, Default)]
pub struct L0Env {
    pub ints: BTreeMap<String, AbsInt>,
    pub bools: BTreeMap<String, bool>,
    /// Closed floating-point interval `[min, max]` for fast L0 float refinement checking.
    pub floats: BTreeMap<String, (f64, f64)>,
    /// Float variables statically proved finite (non-NaN, non-Inf) via `is_finite(x)` or bounded intervals.
    pub finite_floats: BTreeSet<String>,
    /// Float variables statically proved strictly positive (`x > 0.0`) or strictly negative (`x < 0.0`).
    pub nonzero_floats: BTreeSet<String>,
    /// Difference bounds `a - b <= c` stored as `(a, b) -> c`.
    pub diffs: BTreeMap<(String, String), i128>,
    /// Symbolic product lower bounds `var >= a * b` (used to prove 2D row-major `y * w + x < len(grid)` in L0!).
    pub product_lbs: BTreeMap<String, Vec<(String, String)>>,
    /// Element abstract integer intervals for refined `List[RefinedT]` / `[T; N]` bindings (`list_var -> AbsInt`).
    pub elem_ints: BTreeMap<String, AbsInt>,
    /// Flow-sensitively assumed pure predicate calls (e.g. `is_safe_rel_path(p)`).
    pub proved_calls: BTreeSet<String>,
    /// Pure function declarations available for bounded CTFE and ensures summaries.
    pub pure_fns: BTreeMap<String, FnDecl>,
    /// Refined type alias predicates available for callee return-type summaries.
    pub type_aliases: BTreeMap<String, TypeAliasDecl>,
    /// Higher-order function parameter/local return types (`f -> RetTy`) for L0 refinement summaries.
    pub fn_var_ret_tys: BTreeMap<String, TypeExpr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum L0Result {
    Proved,
    Disproved(String),
    Unknown,
}

impl L0Env {
    pub fn set_int(&mut self, name: impl Into<String>, val: AbsInt) {
        self.ints.insert(name.into(), val);
    }

    pub fn record_product_lb(&mut self, var: String, a: String, b: String) {
        let entry = self.product_lbs.entry(var).or_default();
        if !entry.iter().any(|(x, y)| (x == &a && y == &b) || (x == &b && y == &a)) {
            entry.push((a, b));
        }
    }

    fn var_lt(&self, a: &str, b: &str) -> bool {
        if let Some(&d) = self.diffs.get(&(a.to_string(), b.to_string())) {
            if d <= -1 {
                return true;
            }
        }
        // Check 1-hop transitivity: `a < mid` and `mid <= b`, or `a <= mid` and `mid < b`
        for ((x, mid), &d1) in &self.diffs {
            if x == a && d1 <= 0 {
                if let Some(&d2) = self.diffs.get(&(mid.clone(), b.to_string())) {
                    if d1 + d2 <= -1 {
                        return true;
                    }
                }
            }
        }
        let ai = self.get_int(a).interval;
        let bi = self.get_int(b).interval;
        ai.max < bi.min
    }

    fn var_non_neg(&self, v: &str) -> bool {
        self.get_int(v).interval.min >= 0
    }

    /// Proves `row * stride + col < bound` when `bound >= stride * height` (or `height * stride`)
    /// and `0 <= row < height && 0 <= col < stride`.
    fn try_prove_2d_index_lt(&self, left: &Expr, right: &Expr, self_var: Option<&str>) -> bool {
        // Decompose `left` as `(p1 * p2) + c` or `c + (p1 * p2)`
        let (mul_a, mul_b, add_c) = match &left.kind {
            ExprKind::Binary {
                op: BinOp::Add,
                left: l,
                right: r,
            } => {
                if let ExprKind::Binary {
                    op: BinOp::Mul,
                    left: ma,
                    right: mb,
                } = &l.kind
                {
                    (ma.as_ref(), mb.as_ref(), r.as_ref())
                } else if let ExprKind::Binary {
                    op: BinOp::Mul,
                    left: ma,
                    right: mb,
                } = &r.kind
                {
                    (ma.as_ref(), mb.as_ref(), l.as_ref())
                } else {
                    return false;
                }
            }
            _ => return false,
        };
        let (Some(a), Some(b), Some(c)) = (
            expr_var_name(mul_a, self_var),
            expr_var_name(mul_b, self_var),
            expr_var_name(add_c, self_var),
        ) else {
            return false;
        };
        if !self.var_non_neg(&a) || !self.var_non_neg(&b) || !self.var_non_neg(&c) {
            return false;
        }
        // Gather candidate `(d1, d2)` products such that `right >= d1 * d2`
        let mut candidates: Vec<(String, String)> = Vec::new();
        if let ExprKind::Binary {
            op: BinOp::Mul,
            left: ra,
            right: rb,
        } = &right.kind
        {
            if let (Some(x), Some(y)) =
                (expr_var_name(ra, self_var), expr_var_name(rb, self_var))
            {
                candidates.push((x, y));
            }
        }
        if let Some(rname) = expr_var_name(right, self_var) {
            if let Some(prods) = self.product_lbs.get(&rname) {
                candidates.extend(prods.iter().cloned());
            }
            // Also check vars `m` with `m <= rname` (diffs[(m, rname)] <= 0)
            for ((m, target), &d) in &self.diffs {
                if target == &rname && d <= 0 {
                    if let Some(prods) = self.product_lbs.get(m) {
                        candidates.extend(prods.iter().cloned());
                    }
                }
            }
        }
        for (d1, d2) in candidates {
            // Orientation 1: `stride` is `b`, `row` is `a`, `col` is `c` -> need `a < h && c < b`
            if b == d1 && self.var_lt(&a, &d2) && self.var_lt(&c, &b) {
                return true;
            }
            if b == d2 && self.var_lt(&a, &d1) && self.var_lt(&c, &b) {
                return true;
            }
            // Orientation 2: `stride` is `a`, `row` is `b`, `col` is `c` -> need `b < h && c < a`
            if a == d1 && self.var_lt(&b, &d2) && self.var_lt(&c, &a) {
                return true;
            }
            if a == d2 && self.var_lt(&b, &d1) && self.var_lt(&c, &a) {
                return true;
            }
        }
        false
    }

    pub fn get_int(&self, name: &str) -> AbsInt {
        if let Some(v) = self.ints.get(name) {
            return *v;
        }
        if name.starts_with("__len__") {
            return AbsInt {
                interval: Interval {
                    min: 0,
                    max: i64::MAX as i128,
                },
                cong: Congruence::TOP,
                non_zero: false,
            };
        }
        AbsInt::top()
    }

    pub fn set_float(&mut self, name: impl Into<String>, lo: f64, hi: f64) {
        if lo.is_finite() && hi.is_finite() && lo <= hi {
            let n = name.into();
            self.finite_floats.insert(n.clone());
            self.floats.insert(n, (lo, hi));
        }
    }

    /// Static NaN/Inf-freedom evaluator for `is_finite(expr)` (Flexbox layout & GPU uniforms).
    pub fn eval_is_finite(&self, expr: &Expr, self_var: Option<&str>) -> bool {
        match &expr.kind {
            ExprKind::Float(x) => x.is_finite(),
            ExprKind::Int(_) => true,
            ExprKind::Ident(_) | ExprKind::Field { .. } | ExprKind::Old(_) => {
                if let Some(k) = expr_var_name(expr, self_var) {
                    if self.finite_floats.contains(&k) {
                        return true;
                    }
                    if let Some((lo, hi)) = self.floats.get(&k) {
                        return lo.is_finite() && hi.is_finite();
                    }
                }
                false
            }
            ExprKind::Cast { expr: inner, .. } => self.eval_is_finite(inner, self_var),
            ExprKind::Unary { op: UnOp::Neg, expr: inner } => self.eval_is_finite(inner, self_var),
            ExprKind::Binary { op, left, right } => match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul => {
                    self.eval_is_finite(left, self_var) && self.eval_is_finite(right, self_var)
                }
                BinOp::Div => {
                    let denom_nz = self
                        .eval_abs_float(right, self_var)
                        .is_some_and(|(r0, r1)| r0 > 0.0 || r1 < 0.0)
                        || expr_var_name(right, self_var)
                            .is_some_and(|rv| self.nonzero_floats.contains(&rv))
                        || self.eval_abs_int(right, self_var).non_zero;
                    self.eval_is_finite(left, self_var)
                        && self.eval_is_finite(right, self_var)
                        && denom_nz
                }
                _ => false,
            },
            ExprKind::Call { callee, args } => {
                if let ExprKind::Ident(fname) = &callee.kind {
                    if fname == "int_to_float" && args.len() == 1 {
                        return true;
                    }
                    if let Some(fdecl) = self.pure_fns.get(fname) {
                        if let Some(TypeExpr::Named { name: rty, args: rargs, .. }) = &fdecl.ret_ty {
                            if let Some(alias) = self.type_aliases.get(rty) {
                                if let Some(pred) = instantiate_alias_predicate(alias, rargs) {
                                    let mut senv = L0Env {
                                        pure_fns: self.pure_fns.clone(),
                                        type_aliases: self.type_aliases.clone(),
                                        ..Default::default()
                                    };
                                    senv.assume_expr(&pred, Some("result"));
                                    if senv.finite_floats.contains("result") {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
                false
            }
            ExprKind::If { cond, then_block, else_branch } => {
                let mut te = self.clone();
                te.assume_expr(cond, self_var);
                te.assume_float_expr(cond, self_var);
                let ok_t = then_block
                    .tail
                    .as_deref()
                    .is_some_and(|e| te.eval_is_finite(e, self_var));
                if !ok_t {
                    return false;
                }
                if let Some(eb) = else_branch {
                    let mut ee = self.clone();
                    let neg = Expr {
                        kind: ExprKind::Unary {
                            op: UnOp::Not,
                            expr: cond.clone(),
                        },
                        span: cond.span,
                    };
                    ee.assume_expr(&neg, self_var);
                    ee.eval_is_finite(eb, self_var)
                } else {
                    false
                }
            }
            ExprKind::Block(b) => b
                .tail
                .as_deref()
                .is_some_and(|e| self.eval_is_finite(e, self_var)),
            _ => false,
        }
    }

    pub fn eval_abs_float(&self, expr: &Expr, self_var: Option<&str>) -> Option<(f64, f64)> {
        match &expr.kind {
            ExprKind::Float(x) if x.is_finite() => Some((*x, *x)),
            ExprKind::Int(n) => {
                let f = *n as f64;
                Some((f, f))
            }
            ExprKind::Ident(s) => {
                let k = resolve_self(s, self_var);
                self.floats.get(&k).copied()
            }
            ExprKind::Old(inner) => {
                let k = expr_var_name(expr, self_var)?;
                self.floats.get(&k).copied().or_else(|| self.eval_abs_float(inner, self_var))
            }
            ExprKind::Cast { expr: inner, .. } => self.eval_abs_float(inner, self_var),
            ExprKind::Unary { op: UnOp::Neg, expr: inner } => {
                let (lo, hi) = self.eval_abs_float(inner, self_var)?;
                Some((-hi, -lo))
            }
            ExprKind::Binary { op, left, right } => {
                let (l0, l1) = self.eval_abs_float(left, self_var)?;
                let (r0, r1) = self.eval_abs_float(right, self_var)?;
                match op {
                    BinOp::Add => Some((l0 + r0, l1 + r1)),
                    BinOp::Sub => Some((l0 - r1, l1 - r0)),
                    BinOp::Mul => {
                        if l0 >= 0.0 && r0 >= 0.0 {
                            let hi = if l1.is_finite() && r1.is_finite() { l1 * r1 } else { f64::INFINITY };
                            return Some((l0 * r0, hi));
                        }
                        if l0.is_finite() && l1.is_finite() && r0.is_finite() && r1.is_finite() {
                            let ps = [l0 * r0, l0 * r1, l1 * r0, l1 * r1];
                            let mn = ps.iter().copied().fold(f64::INFINITY, f64::min);
                            let mx = ps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            Some((mn, mx))
                        } else {
                            None
                        }
                    }
                    BinOp::Div if r0 > 0.0 || r1 < 0.0 => {
                        if l0 >= 0.0 && r0 > 0.0 {
                            let hi = if l1.is_finite() { l1 / r0 } else { f64::INFINITY };
                            return Some((0.0, hi));
                        }
                        if l0.is_finite() && l1.is_finite() && r0.is_finite() && r1.is_finite() {
                            let qs = [l0 / r0, l0 / r1, l1 / r0, l1 / r1];
                            let mn = qs.iter().copied().fold(f64::INFINITY, f64::min);
                            let mx = qs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            Some((mn, mx))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            ExprKind::Call { callee, args } => {
                if let ExprKind::Ident(fname) = &callee.kind {
                    if fname == "int_to_float" && args.len() == 1 {
                        let a = self.eval_abs_int(&args[0].value, self_var);
                        let lo = if a.interval.min <= i64::MIN as i128 { f64::NEG_INFINITY } else { a.interval.min as f64 };
                        let hi = if a.interval.max >= i64::MAX as i128 { f64::INFINITY } else { a.interval.max as f64 };
                        return Some((lo, hi));
                    }
                    if let Some(fdecl) = self.pure_fns.get(fname) {
                        if let Some(TypeExpr::Named { name: rty, args: rargs, .. }) = &fdecl.ret_ty {
                            if let Some(alias) = self.type_aliases.get(rty) {
                                if let Some(pred) = instantiate_alias_predicate(alias, rargs) {
                                    let mut senv = L0Env {
                                        pure_fns: self.pure_fns.clone(),
                                        type_aliases: self.type_aliases.clone(),
                                        ..Default::default()
                                    };
                                    senv.assume_float_expr(&pred, Some("result"));
                                    return senv.floats.get("result").copied();
                                }
                            }
                        }
                    }
                }
                None
            }
            ExprKind::If { cond, then_block, else_branch } => {
                let mut te = self.clone();
                te.assume_expr(cond, self_var);
                te.assume_float_expr(cond, self_var);
                let (t0, t1) = then_block
                    .tail
                    .as_deref()
                    .and_then(|e| te.eval_abs_float(e, self_var))?;
                let eb = else_branch.as_deref()?;
                let mut ee = self.clone();
                let neg = Expr {
                    kind: ExprKind::Unary { op: UnOp::Not, expr: cond.clone() },
                    span: cond.span,
                };
                ee.assume_expr(&neg, self_var);
                ee.assume_float_expr(&neg, self_var);
                let (e0, e1) = ee.eval_abs_float(eb, self_var)?;
                Some((t0.min(e0), t1.max(e1)))
            }
            ExprKind::Block(b) => b.tail.as_deref().and_then(|e| self.eval_abs_float(e, self_var)),
            _ => None,
        }
    }

    pub fn assume_float_expr(&mut self, expr: &Expr, self_var: Option<&str>) {
        match &expr.kind {
            ExprKind::Unary { op: UnOp::Not, expr: inner } => {
                if let ExprKind::Binary { op, left, right } = &inner.kind {
                    let flipped = match op {
                        BinOp::Lt => Some(BinOp::Ge),
                        BinOp::Le => Some(BinOp::Gt),
                        BinOp::Gt => Some(BinOp::Le),
                        BinOp::Ge => Some(BinOp::Lt),
                        _ => None,
                    };
                    if let Some(fop) = flipped {
                        let rewritten = Expr {
                            kind: ExprKind::Binary {
                                op: fop,
                                left: left.clone(),
                                right: right.clone(),
                            },
                            span: inner.span,
                        };
                        self.assume_float_expr(&rewritten, self_var);
                    }
                }
            }
            ExprKind::Binary { op: BinOp::And, left, right } => {
                self.assume_float_expr(left, self_var);
                self.assume_float_expr(right, self_var);
            }
            ExprKind::Binary { op, left, right } => {
                if let (Some(v), Some((r0, r1))) = (
                    expr_var_name(left, self_var),
                    self.eval_abs_float(right, self_var),
                ) {
                    let (cur_lo, cur_hi) = self.floats.get(&v).copied().unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
                    let (nlo, nhi) = match op {
                        BinOp::Ge | BinOp::Gt => {
                            if (*op == BinOp::Gt && r0 >= 0.0) || (*op == BinOp::Ge && r0 > 0.0) {
                                self.nonzero_floats.insert(v.clone());
                            }
                            (cur_lo.max(r0), cur_hi)
                        }
                        BinOp::Le | BinOp::Lt => {
                            if (*op == BinOp::Lt && r1 <= 0.0) || (*op == BinOp::Le && r1 < 0.0) {
                                self.nonzero_floats.insert(v.clone());
                            }
                            (cur_lo, cur_hi.min(r1))
                        }
                        BinOp::Eq => (cur_lo.max(r0), cur_hi.min(r1)),
                        _ => (cur_lo, cur_hi),
                    };
                    if nlo <= nhi {
                        self.floats.insert(v, (nlo, nhi));
                    }
                } else if let (Some((l0, l1)), Some(v)) = (
                    self.eval_abs_float(left, self_var),
                    expr_var_name(right, self_var),
                ) {
                    let (cur_lo, cur_hi) = self.floats.get(&v).copied().unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
                    let (nlo, nhi) = match op {
                        BinOp::Le | BinOp::Lt => (cur_lo.max(l0), cur_hi),
                        BinOp::Ge | BinOp::Gt => (cur_lo, cur_hi.min(l1)),
                        BinOp::Eq => (cur_lo.max(l0), cur_hi.min(l1)),
                        _ => (cur_lo, cur_hi),
                    };
                    if nlo <= nhi {
                        self.floats.insert(v, (nlo, nhi));
                    }
                }
            }
            _ => {}
        }
    }

    /// Assume a predicate holds (for parameter refinements, `requires`, `if` guards, `assert`).
    pub fn assume_expr(&mut self, expr: &Expr, self_var: Option<&str>) {
        match &expr.kind {
            ExprKind::Binary { op, left, right } => match op {
                BinOp::And => {
                    self.assume_expr(left, self_var);
                    self.assume_expr(right, self_var);
                }
                BinOp::Gt | BinOp::Ge | BinOp::Lt | BinOp::Le | BinOp::Eq | BinOp::Ne => {
                    self.assume_cmp(*op, left, right, self_var);
                }
                _ => {}
            },
            ExprKind::Ident(name) => {
                let key = resolve_self(name, self_var);
                self.bools.insert(key, true);
            }
            ExprKind::Call { callee, args } => {
                if matches!(&callee.kind, ExprKind::Ident(f) if f == "is_finite") && args.len() == 1 {
                    if let Some(v) = expr_var_name(&args[0].value, self_var) {
                        self.finite_floats.insert(v);
                    }
                }
                if let Some(ck) = canonical_pred_call(callee, args, self_var) {
                    self.proved_calls.insert(ck);
                }
            }
            ExprKind::Unary {
                op: UnOp::Not,
                expr: inner,
            } => match &inner.kind {
                ExprKind::Ident(name) => {
                    let key = resolve_self(name, self_var);
                    self.bools.insert(key, false);
                }
                ExprKind::Unary {
                    op: UnOp::Not,
                    expr: double,
                } => self.assume_expr(double, self_var),
                ExprKind::Binary { op, left, right } => match op {
                    BinOp::Or => {
                        let nl = Expr {
                            kind: ExprKind::Unary {
                                op: UnOp::Not,
                                expr: left.clone(),
                            },
                            span: left.span,
                        };
                        let nr = Expr {
                            kind: ExprKind::Unary {
                                op: UnOp::Not,
                                expr: right.clone(),
                            },
                            span: right.span,
                        };
                        self.assume_expr(&nl, self_var);
                        self.assume_expr(&nr, self_var);
                    }
                    BinOp::Lt => self.assume_cmp(BinOp::Ge, left, right, self_var),
                    BinOp::Le => self.assume_cmp(BinOp::Gt, left, right, self_var),
                    BinOp::Gt => self.assume_cmp(BinOp::Le, left, right, self_var),
                    BinOp::Ge => self.assume_cmp(BinOp::Lt, left, right, self_var),
                    BinOp::Eq => self.assume_cmp(BinOp::Ne, left, right, self_var),
                    BinOp::Ne => self.assume_cmp(BinOp::Eq, left, right, self_var),
                    _ => {}
                },
                _ => {}
            },
            _ => {}
        }
    }

    fn assume_cmp(&mut self, op: BinOp, left: &Expr, right: &Expr, self_var: Option<&str>) {
        // Pattern: `x % m == r` or `x % 2 != 0`
        if let ExprKind::Binary {
            op: BinOp::Mod,
            left: mod_lhs,
            right: mod_rhs,
        } = &left.kind
        {
            if let (Some(var), Some(m), Some(r)) = (
                expr_var_name(mod_lhs, self_var),
                self.eval_const_int(mod_rhs, self_var),
                self.eval_const_int(right, self_var),
            ) {
                if op == BinOp::Eq && m > 0 {
                    let cur = self.get_int(&var);
                    self.set_int(var, cur.refine_cong(Congruence::new(m, r)));
                    return;
                }
                if op == BinOp::Ne && m == 2 && r == 0 {
                    let cur = self.get_int(&var);
                    self.set_int(var, cur.refine_cong(Congruence::new(2, 1)));
                    return;
                }
                if op == BinOp::Ne && m == 2 && r == 1 {
                    let cur = self.get_int(&var);
                    self.set_int(var, cur.refine_cong(Congruence::new(2, 0)));
                    return;
                }
            }
        }

        // Pattern: `var <op> const` or `const <op> var`
        if let (Some(var), Some(c)) = (
            expr_var_name(left, self_var),
            self.eval_const_int(right, self_var),
        ) {
            let mut cur = self.get_int(&var);
            match op {
                BinOp::Gt => cur = cur.refine_interval(Interval { min: c + 1, max: i64::MAX as i128 }),
                BinOp::Ge => cur = cur.refine_interval(Interval { min: c, max: i64::MAX as i128 }),
                BinOp::Lt => cur = cur.refine_interval(Interval { min: i64::MIN as i128, max: c - 1 }),
                BinOp::Le => cur = cur.refine_interval(Interval { min: i64::MIN as i128, max: c }),
                BinOp::Eq => cur = cur.refine_interval(Interval::exact(c)),
                BinOp::Ne if c == 0 => cur.non_zero = true,
                _ => {}
            }
            self.set_int(var, cur);
            return;
        }
        if let (Some(c), Some(var)) = (
            self.eval_const_int(left, self_var),
            expr_var_name(right, self_var),
        ) {
            let flipped = match op {
                BinOp::Gt => BinOp::Lt,
                BinOp::Ge => BinOp::Le,
                BinOp::Lt => BinOp::Gt,
                BinOp::Le => BinOp::Ge,
                other => other,
            };
            self.assume_cmp(flipped, right, left, self_var);
            let _ = (c, var);
            return;
        }

        // Difference bound: `a <= b` => `a - b <= 0`, `a < b` => `a - b <= -1`
        if let (Some(a), Some(b)) = (
            expr_var_name(left, self_var),
            expr_var_name(right, self_var),
        ) {
            match op {
                BinOp::Le => {
                    self.diffs.insert((a, b), 0);
                }
                BinOp::Lt => {
                    self.diffs.insert((a, b), -1);
                }
                BinOp::Ge => {
                    self.diffs.insert((b, a), 0);
                }
                BinOp::Gt => {
                    self.diffs.insert((b, a), -1);
                }
                BinOp::Eq => {
                    self.diffs.insert((a.clone(), b.clone()), 0);
                    self.diffs.insert((b, a), 0);
                }
                _ => {}
            }
        }

        // Symbolic product lower bounds: `var == a * b` / `var >= a * b` (and flipped `a * b == var` / `a * b <= var`)
        if matches!(op, BinOp::Eq | BinOp::Ge) {
            if let Some(lhs_v) = expr_var_name(left, self_var) {
                if let ExprKind::Binary {
                    op: BinOp::Mul,
                    left: m1,
                    right: m2,
                } = &right.kind
                {
                    if let (Some(a), Some(b)) =
                        (expr_var_name(m1, self_var), expr_var_name(m2, self_var))
                    {
                        self.record_product_lb(lhs_v, a, b);
                    }
                }
            }
        }
        if matches!(op, BinOp::Eq | BinOp::Le) {
            if let Some(rhs_v) = expr_var_name(right, self_var) {
                if let ExprKind::Binary {
                    op: BinOp::Mul,
                    left: m1,
                    right: m2,
                } = &left.kind
                {
                    if let (Some(a), Some(b)) =
                        (expr_var_name(m1, self_var), expr_var_name(m2, self_var))
                    {
                        self.record_product_lb(rhs_v, a, b);
                    }
                }
            }
        }

        // General interval refinement: `var <op> expr` refines `var` using `eval_abs_int(expr).interval`
        if let Some(var) = expr_var_name(left, self_var) {
            let ri = self.eval_abs_int(right, self_var).interval;
            let mut cur = self.get_int(&var);
            match op {
                BinOp::Le => cur = cur.refine_interval(Interval { min: i64::MIN as i128, max: ri.max }),
                BinOp::Lt => cur = cur.refine_interval(Interval { min: i64::MIN as i128, max: ri.max.saturating_sub(1) }),
                BinOp::Ge => cur = cur.refine_interval(Interval { min: ri.min, max: i64::MAX as i128 }),
                BinOp::Gt => cur = cur.refine_interval(Interval { min: ri.min.saturating_add(1), max: i64::MAX as i128 }),
                BinOp::Eq => cur = cur.refine_interval(ri),
                _ => {}
            }
            self.set_int(var, cur);
        } else if let Some(var) = expr_var_name(right, self_var) {
            let li = self.eval_abs_int(left, self_var).interval;
            let mut cur = self.get_int(&var);
            match op {
                BinOp::Ge => cur = cur.refine_interval(Interval { min: i64::MIN as i128, max: li.max }),
                BinOp::Gt => cur = cur.refine_interval(Interval { min: i64::MIN as i128, max: li.max.saturating_sub(1) }),
                BinOp::Le => cur = cur.refine_interval(Interval { min: li.min, max: i64::MAX as i128 }),
                BinOp::Lt => cur = cur.refine_interval(Interval { min: li.min.saturating_add(1), max: i64::MAX as i128 }),
                BinOp::Eq => cur = cur.refine_interval(li),
                _ => {}
            }
            self.set_int(var, cur);
        }
    }

    /// Evaluate abstract integer semantics of an expression.
    pub fn eval_abs_int(&self, expr: &Expr, self_var: Option<&str>) -> AbsInt {
        match &expr.kind {
            ExprKind::Int(n) => AbsInt::exact(*n as i128),
            ExprKind::Cast { expr: inner, ty } => {
                let a = self.eval_abs_int(inner, self_var);
                if let TypeExpr::Named { name, .. } = ty {
                    if let Some(sk) = tenon_syntax::ScalarKind::from_name(name) {
                        if sk.is_int() {
                            if let (Some(lo), Some(hi)) = (sk.min_i128(), sk.max_i128()) {
                                if a.interval.min >= lo && a.interval.max <= hi {
                                    return a;
                                }
                                if sk == tenon_syntax::ScalarKind::U64 {
                                    if let ExprKind::Int(n) = &inner.kind {
                                        return AbsInt::exact((*n as u64) as i128);
                                    }
                                }
                                return AbsInt {
                                    interval: Interval { min: lo, max: hi },
                                    cong: Congruence::TOP,
                                    non_zero: false,
                                };
                            }
                        }
                    }
                }
                a
            }
            ExprKind::Ident(s) => {
                let key = resolve_self(s, self_var);
                self.get_int(&key)
            }
            ExprKind::Old(_) => {
                if let Some(k) = expr_var_name(expr, self_var) {
                    self.get_int(&k)
                } else {
                    AbsInt::top()
                }
            }
            ExprKind::Call { callee, args }
                if matches!(&callee.kind, ExprKind::Ident(s) if s == "len") && args.len() == 1 =>
            {
                if let Some(k) = expr_var_name(expr, self_var) {
                    self.get_int(&k)
                } else {
                    AbsInt {
                        interval: Interval {
                            min: 0,
                            max: i64::MAX as i128,
                        },
                        cong: Congruence::TOP,
                        non_zero: false,
                    }
                }
            }
            ExprKind::Index { target, .. } => {
                if let Some(base) = expr_var_name(target, self_var) {
                    if let Some(elem_abs) = self.elem_ints.get(&base) {
                        return *elem_abs;
                    }
                }
                AbsInt::top()
            }
            ExprKind::Field { target, field } => {
                if let Some(base) = expr_var_name(target, self_var) {
                    let key = format!("{}.{}", base, field);
                    self.get_int(&key)
                } else {
                    AbsInt::top()
                }
            }
            ExprKind::Unary {
                op: UnOp::Neg,
                expr: inner,
            } => {
                let a = self.eval_abs_int(inner, self_var);
                AbsInt {
                    interval: Interval {
                        min: a.interval.max.saturating_neg(),
                        max: a.interval.min.saturating_neg(),
                    },
                    cong: Congruence::exact(-1).mul(a.cong),
                    non_zero: a.non_zero,
                }
            }
            ExprKind::Binary { op, left, right } => {
                let l = self.eval_abs_int(left, self_var);
                let r = self.eval_abs_int(right, self_var);
                match op {
                    BinOp::Add => {
                        let iv = l.interval.add(r.interval);
                        let cg = l.cong.add(r.cong);
                        let nz = iv.min > 0 || iv.max < 0 || (cg.modulus > 1 && cg.rem != 0);
                        AbsInt {
                            interval: iv,
                            cong: cg,
                            non_zero: nz,
                        }
                    }
                    BinOp::Sub => {
                        let iv = l.interval.sub(r.interval);
                        let cg = l.cong.sub(r.cong);
                        let nz = iv.min > 0 || iv.max < 0 || (cg.modulus > 1 && cg.rem != 0);
                        AbsInt {
                            interval: iv,
                            cong: cg,
                            non_zero: nz,
                        }
                    }
                    BinOp::Mul => {
                        let iv = l.interval.mul(r.interval);
                        let cg = l.cong.mul(r.cong);
                        let nz = (l.non_zero && r.non_zero)
                            || iv.min > 0
                            || iv.max < 0
                            || (cg.modulus > 1 && cg.rem != 0);
                        AbsInt {
                            interval: iv,
                            cong: cg,
                            non_zero: nz,
                        }
                    }
                    BinOp::Div => {
                        if r.interval.min == r.interval.max && r.interval.min > 0 {
                            let d = r.interval.min;
                            return AbsInt {
                                interval: Interval {
                                    min: l.interval.min / d,
                                    max: l.interval.max / d,
                                },
                                cong: if l.interval.min == l.interval.max {
                                    Congruence::exact(l.interval.min / d)
                                } else {
                                    Congruence::TOP
                                },
                                non_zero: (l.interval.min / d) > 0 || (l.interval.max / d) < 0,
                            };
                        }
                        AbsInt::top()
                    }
                    BinOp::Mod => {
                        if r.interval.min == r.interval.max && r.interval.min > 0 {
                            let m = r.interval.min;
                            if l.cong.modulus == 0 {
                                return AbsInt::exact(l.cong.rem % m);
                            }
                            if l.cong.modulus % m == 0 {
                                let rem = l.cong.rem.rem_euclid(m);
                                if l.interval.min >= 0 {
                                    return AbsInt::exact(rem);
                                }
                                return AbsInt {
                                    interval: Interval {
                                        min: -(m - 1),
                                        max: m - 1,
                                    },
                                    cong: Congruence::new(m, rem),
                                    non_zero: rem != 0,
                                };
                            }
                            return AbsInt {
                                interval: Interval {
                                    min: if l.interval.min >= 0 { 0 } else { -(m - 1) },
                                    max: m - 1,
                                },
                                cong: Congruence::TOP,
                                non_zero: false,
                            };
                        }
                        AbsInt::top()
                    }
                    BinOp::BitAnd => {
                        if r.interval.min >= 0 && r.interval.max <= i64::MAX as i128 {
                            return AbsInt {
                                interval: Interval {
                                    min: 0,
                                    max: r.interval.max,
                                },
                                cong: Congruence::TOP,
                                non_zero: false,
                            };
                        }
                        if l.interval.min >= 0 && l.interval.max <= i64::MAX as i128 {
                            return AbsInt {
                                interval: Interval {
                                    min: 0,
                                    max: l.interval.max,
                                },
                                cong: Congruence::TOP,
                                non_zero: false,
                            };
                        }
                        AbsInt::top()
                    }
                    _ => AbsInt::top(),
                }
            }
            ExprKind::Call { callee, args } => {
                if let ExprKind::GenericInst { expr: inner, .. } = &callee.kind {
                    if matches!(&inner.kind, ExprKind::Ident(n) if n == "refine_cast") && args.len() == 1 {
                        return self.eval_abs_int(&args[0].value, self_var);
                    }
                }
                if let ExprKind::Field { field, .. } = &callee.kind {
                    if matches!(field.as_str(), "len" | "byte_len" | "size") && args.is_empty() {
                        return AbsInt::top().refine_interval(Interval {
                            min: 0,
                            max: i64::MAX as i128,
                        });
                    }
                    if field == "byte_at" && args.len() == 1 {
                        return AbsInt::top().refine_interval(Interval { min: -1, max: 255 });
                    }
                }
                if let ExprKind::Ident(fname) = &callee.kind {
                    // Built-in core domain facts: `len(x) >= 0`, `byte_at(s, i) in [-1, 255]`
                    if fname == "len" && args.len() == 1 {
                        return AbsInt::top().refine_interval(Interval {
                            min: 0,
                            max: i64::MAX as i128,
                        });
                    }
                    if fname == "byte_at" && args.len() == 2 {
                        return AbsInt::top().refine_interval(Interval { min: -1, max: 255 });
                    }
                    // 1. Try exact bounded CTFE if all args evaluate to exact constants!
                    let mut const_args = Vec::new();
                    let mut all_const = true;
                    for a in args {
                        if let Some(v) = self.eval_const_int(&a.value, self_var) {
                            const_args.push(CtfeVal::Int(v));
                        } else if let Some(b) = self.eval_const_bool(&a.value, self_var) {
                            const_args.push(CtfeVal::Bool(b));
                        } else {
                            all_const = false;
                            break;
                        }
                    }
                    if all_const {
                        if let Some(CtfeVal::Int(v)) = self.ctfe_call(fname, &const_args, 256) {
                            return AbsInt::exact(v);
                        }
                    }
                    // 2. Otherwise use callee return-type refinement & `ensures` summary bounds on `result`!
                    if let Some(fdecl) = self.pure_fns.get(fname) {
                        let mut summary_env = L0Env {
                            pure_fns: self.pure_fns.clone(),
                            type_aliases: self.type_aliases.clone(),
                            ..Default::default()
                        };
                        if let Some(TypeExpr::Named { name: rty, args: rargs, .. }) = &fdecl.ret_ty {
                            if let Some(alias) = self.type_aliases.get(rty) {
                                if let Some(pred) = instantiate_alias_predicate(alias, rargs) {
                                    summary_env.assume_expr(&pred, Some("result"));
                                }
                            }
                        }
                        for ens in &fdecl.ensures {
                            summary_env.assume_expr(ens, Some("result"));
                        }
                        return summary_env.get_int("result");
                    }
                    // 3. Higher-order function value call `f(args)` -> apply declared return refinement!
                    if let Some(TypeExpr::Named { name: rty, args: rargs, .. }) =
                        self.fn_var_ret_tys.get(fname)
                    {
                        if let Some(alias) = self.type_aliases.get(rty) {
                            if let Some(pred) = instantiate_alias_predicate(alias, rargs) {
                                let mut summary_env = L0Env {
                                    pure_fns: self.pure_fns.clone(),
                                    type_aliases: self.type_aliases.clone(),
                                    ..Default::default()
                                };
                                summary_env.assume_expr(&pred, Some("result"));
                                return summary_env.get_int("result");
                            }
                        }
                    }
                }
                AbsInt::top()
            }
            ExprKind::Block(blk) => {
                if let Some(tail) = &blk.tail {
                    self.eval_abs_int(tail, self_var)
                } else {
                    AbsInt::top()
                }
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                let mut then_env = self.clone();
                then_env.assume_expr(cond, self_var);
                let then_val = then_block
                    .tail
                    .as_deref()
                    .map(|e| then_env.eval_abs_int(e, self_var))
                    .unwrap_or_else(AbsInt::top);
                if let Some(eb) = else_branch {
                    let mut else_env = self.clone();
                    let neg_cond = Expr {
                        kind: ExprKind::Unary {
                            op: UnOp::Not,
                            expr: cond.clone(),
                        },
                        span: cond.span,
                    };
                    else_env.assume_expr(&neg_cond, self_var);
                    let else_val = else_env.eval_abs_int(eb, self_var);
                    then_val.join(else_val)
                } else {
                    AbsInt::top()
                }
            }
            ExprKind::Match { arms, .. } => {
                let mut joined: Option<AbsInt> = None;
                for arm in arms {
                    // Skip diverging arm blocks (`{ return ... }`, `{ continue }`, `{ break }`)
                    if let ExprKind::Block(b) = &arm.body.kind {
                        if b.tail.is_none()
                            && matches!(
                                b.stmts.last(),
                                Some(Stmt::Return { .. } | Stmt::Break { .. } | Stmt::Continue { .. })
                            )
                        {
                            continue;
                        }
                    }
                    let av = self.eval_abs_int(&arm.body, self_var);
                    joined = Some(match joined {
                        Some(acc) => acc.join(av),
                        None => av,
                    });
                }
                joined.unwrap_or_else(AbsInt::top)
            }
            _ => AbsInt::top(),
        }
    }

    pub fn eval_const_int(&self, expr: &Expr, self_var: Option<&str>) -> Option<i128> {
        let abs = self.eval_abs_int(expr, self_var);
        if abs.interval.min == abs.interval.max {
            Some(abs.interval.min)
        } else {
            None
        }
    }

    pub fn eval_const_bool(&self, expr: &Expr, self_var: Option<&str>) -> Option<bool> {
        match &expr.kind {
            ExprKind::Bool(b) => Some(*b),
            ExprKind::Ident(s) => self.bools.get(&resolve_self(s, self_var)).copied(),
            _ => match self.check_predicate(expr, self_var) {
                L0Result::Proved => Some(true),
                L0Result::Disproved(_) => Some(false),
                L0Result::Unknown => None,
            },
        }
    }

    /// Check whether `pred` is statically proved or disproved by L0 (<1ms, zero Z3!).
    pub fn check_predicate(&self, pred: &Expr, self_var: Option<&str>) -> L0Result {
        match &pred.kind {
            ExprKind::Bool(true) => L0Result::Proved,
            ExprKind::Bool(false) => L0Result::Disproved("constant `false`".into()),
            ExprKind::Ident(name) => {
                let key = resolve_self(name, self_var);
                match self.bools.get(&key) {
                    Some(true) => L0Result::Proved,
                    Some(false) => L0Result::Disproved(format!("`{}` is known false", key)),
                    None => L0Result::Unknown,
                }
            }
            ExprKind::Unary {
                op: UnOp::Not,
                expr: inner,
            } => match self.check_predicate(inner, self_var) {
                L0Result::Proved => L0Result::Disproved("negated predicate is true".into()),
                L0Result::Disproved(_) => L0Result::Proved,
                L0Result::Unknown => L0Result::Unknown,
            },
            ExprKind::Binary { op, left, right } => match op {
                BinOp::And => match self.check_predicate(left, self_var) {
                    L0Result::Disproved(r) => L0Result::Disproved(r),
                    L0Result::Proved => self.check_predicate(right, self_var),
                    L0Result::Unknown => match self.check_predicate(right, self_var) {
                        L0Result::Disproved(r) => L0Result::Disproved(r),
                        _ => L0Result::Unknown,
                    },
                },
                BinOp::Or => match self.check_predicate(left, self_var) {
                    L0Result::Proved => L0Result::Proved,
                    L0Result::Disproved(r1) => match self.check_predicate(right, self_var) {
                        L0Result::Proved => L0Result::Proved,
                        L0Result::Disproved(r2) => {
                            L0Result::Disproved(format!("both branches false: {}; {}", r1, r2))
                        }
                        L0Result::Unknown => L0Result::Unknown,
                    },
                    L0Result::Unknown => match self.check_predicate(right, self_var) {
                        L0Result::Proved => L0Result::Proved,
                        _ => L0Result::Unknown,
                    },
                },
                BinOp::Implies => {
                    let mut branch = self.clone();
                    branch.assume_expr(left, self_var);
                    branch.check_predicate(right, self_var)
                }
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                    self.check_cmp(*op, left, right, self_var)
                }
                _ => L0Result::Unknown,
            },
            ExprKind::Call { callee, args } => {
                if matches!(&callee.kind, ExprKind::Ident(f) if f == "is_finite") && args.len() == 1 {
                    if self.eval_is_finite(&args[0].value, self_var) {
                        return L0Result::Proved;
                    }
                    if let ExprKind::Float(x) = args[0].value.kind {
                        if !x.is_finite() {
                            return L0Result::Disproved("floating-point literal is NaN or Inf".into());
                        }
                    }
                }
                if let Some(ck) = canonical_pred_call(callee, args, self_var) {
                    if self.proved_calls.contains(&ck) {
                        return L0Result::Proved;
                    }
                }
                if let ExprKind::Ident(fname) = &callee.kind {
                    let mut const_args = Vec::new();
                    for a in args {
                        if let Some(v) = self.eval_const_int(&a.value, self_var) {
                            const_args.push(CtfeVal::Int(v));
                        } else if let Some(b) = self.eval_const_bool(&a.value, self_var) {
                            const_args.push(CtfeVal::Bool(b));
                        } else {
                            return L0Result::Unknown;
                        }
                    }
                    if let Some(CtfeVal::Bool(b)) = self.ctfe_call(fname, &const_args, 256) {
                        return if b {
                            L0Result::Proved
                        } else {
                            L0Result::Disproved(format!("evaluated `{}(...) == false`", fname))
                        };
                    }
                }
                L0Result::Unknown
            }
            _ => L0Result::Unknown,
        }
    }

    fn check_cmp(&self, op: BinOp, left: &Expr, right: &Expr, self_var: Option<&str>) -> L0Result {
        // 1. Congruence check for `expr % m == r` or `expr % m != r`
        if let ExprKind::Binary {
            op: BinOp::Mod,
            left: mod_lhs,
            right: mod_rhs,
        } = &left.kind
        {
            if let (Some(m), Some(r)) = (
                self.eval_const_int(mod_rhs, self_var),
                self.eval_const_int(right, self_var),
            ) {
                let lhs_abs = self.eval_abs_int(mod_lhs, self_var);
                if let Some(is_eq) = lhs_abs.cong.implies_mod_eq(m, r) {
                    return match op {
                        BinOp::Eq => {
                            if is_eq {
                                L0Result::Proved
                            } else {
                                L0Result::Disproved(format!(
                                    "congruence mismatch: expected ≡ {} (mod {}), got ≡ {} (mod {})",
                                    r, m, lhs_abs.cong.rem, lhs_abs.cong.modulus
                                ))
                            }
                        }
                        BinOp::Ne => {
                            if !is_eq {
                                L0Result::Proved
                            } else {
                                L0Result::Disproved(format!(
                                    "congruence violation: value is ≡ {} (mod {})",
                                    r, m
                                ))
                            }
                        }
                        _ => L0Result::Unknown,
                    };
                }
            }
        }

        // 2. Difference bound check (`a - b <= c`)
        if let (Some(a), Some(b)) = (
            expr_var_name(left, self_var),
            expr_var_name(right, self_var),
        ) {
            if a == b {
                return match op {
                    BinOp::Eq | BinOp::Le | BinOp::Ge => L0Result::Proved,
                    BinOp::Ne | BinOp::Lt | BinOp::Gt => {
                        L0Result::Disproved(format!("`{} {} {}` is always false", a, op.symbol(), b))
                    }
                    _ => L0Result::Unknown,
                };
            }
            if let Some(&bound) = self.diffs.get(&(a.clone(), b.clone())) {
                match op {
                    BinOp::Le if bound <= 0 => return L0Result::Proved,
                    BinOp::Lt if bound <= -1 => return L0Result::Proved,
                    _ => {}
                }
            }
            if let Some(&bound) = self.diffs.get(&(b.clone(), a.clone())) {
                match op {
                    BinOp::Ge if bound <= 0 => return L0Result::Proved,
                    BinOp::Gt if bound <= -1 => return L0Result::Proved,
                    _ => {}
                }
            }
        }

        // 2b. 2D Row-Major Index In-Bounds Theorem:
        // If `0 <= row < h` and `0 <= col < w`, then `row * w + col < w * h` (and `< bound` whenever `bound >= w * h`)!
        if op == BinOp::Lt && self.try_prove_2d_index_lt(left, right, self_var) {
            return L0Result::Proved;
        }
        if op == BinOp::Gt && self.try_prove_2d_index_lt(right, left, self_var) {
            return L0Result::Proved;
        }

        // 2c. Float interval comparison check (`NonNegFloat` / `FiniteFloat` bounds in <0.01ms!)
        if let (Some((l0, l1)), Some((r0, r1))) = (
            self.eval_abs_float(left, self_var),
            self.eval_abs_float(right, self_var),
        ) {
            let is_float_cmp = !matches!(&left.kind, ExprKind::Int(_))
                || !matches!(&right.kind, ExprKind::Int(_));
            if is_float_cmp
                && (matches!(&left.kind, ExprKind::Float(_))
                    || matches!(&right.kind, ExprKind::Float(_))
                    || expr_var_name(left, self_var).is_some_and(|v| self.floats.contains_key(&v) || self.finite_floats.contains(&v))
                    || expr_var_name(right, self_var).is_some_and(|v| self.floats.contains_key(&v) || self.finite_floats.contains(&v)))
            {
                match op {
                    BinOp::Ge if l0 >= r1 => return L0Result::Proved,
                    BinOp::Gt if l0 > r1 => return L0Result::Proved,
                    BinOp::Le if l1 <= r0 => return L0Result::Proved,
                    BinOp::Lt if l1 < r0 => return L0Result::Proved,
                    _ => {}
                }
            }
        }

        // 3. Interval check
        let l = self.eval_abs_int(left, self_var);
        let r = self.eval_abs_int(right, self_var);
        let li = l.interval;
        let ri = r.interval;

        match op {
            BinOp::Lt => {
                if li.max < ri.min {
                    L0Result::Proved
                } else if li.min >= ri.max {
                    L0Result::Disproved(format!("`{} < {}` is false", li.min, ri.max))
                } else {
                    L0Result::Unknown
                }
            }
            BinOp::Le => {
                if li.max <= ri.min {
                    L0Result::Proved
                } else if li.min > ri.max {
                    L0Result::Disproved(format!("`{} <= {}` is false", li.min, ri.max))
                } else {
                    L0Result::Unknown
                }
            }
            BinOp::Gt => {
                if li.min > ri.max {
                    L0Result::Proved
                } else if li.max <= ri.min {
                    L0Result::Disproved(format!("`{} > {}` is false", li.max, ri.min))
                } else {
                    L0Result::Unknown
                }
            }
            BinOp::Ge => {
                if li.min >= ri.max {
                    L0Result::Proved
                } else if li.max < ri.min {
                    L0Result::Disproved(format!("`{} >= {}` is false", li.max, ri.min))
                } else {
                    L0Result::Unknown
                }
            }
            BinOp::Eq => {
                if li.min == li.max && ri.min == ri.max && li.min == ri.min {
                    L0Result::Proved
                } else if li.max < ri.min || ri.max < li.min {
                    L0Result::Disproved(format!(
                        "disjoint intervals `[{}..={}] == [{}..={}]`",
                        li.min, li.max, ri.min, ri.max
                    ))
                } else {
                    L0Result::Unknown
                }
            }
            BinOp::Ne => {
                if (ri.min == 0 && ri.max == 0 && l.non_zero)
                    || (li.min == 0 && li.max == 0 && r.non_zero)
                    || li.max < ri.min
                    || ri.max < li.min
                {
                    L0Result::Proved
                } else if li.min == li.max && ri.min == ri.max && li.min == ri.min {
                    L0Result::Disproved(format!("`{} != {}` is false", li.min, ri.min))
                } else {
                    L0Result::Unknown
                }
            }
            _ => L0Result::Unknown,
        }
    }

    /// Bounded CTFE evaluator for pure Tenon functions (handles `is_leap_year`, `days_in_month`, etc.).
    pub fn ctfe_call(&self, name: &str, args: &[CtfeVal], budget: usize) -> Option<CtfeVal> {
        if budget == 0 {
            return None;
        }
        let f = self.pure_fns.get(name)?;
        if !f.is_pure || f.params.len() != args.len() {
            return None;
        }
        let mut locals = BTreeMap::new();
        for (p, v) in f.params.iter().zip(args.iter()) {
            locals.insert(p.name.clone(), v.clone());
        }
        self.ctfe_block(&f.body, &mut locals, budget - 1)
    }

    fn ctfe_block(
        &self,
        block: &Block,
        locals: &mut BTreeMap<String, CtfeVal>,
        budget: usize,
    ) -> Option<CtfeVal> {
        if budget == 0 {
            return None;
        }
        for stmt in &block.stmts {
            match stmt {
                Stmt::Let { name, value, .. } => {
                    let v = self.ctfe_expr(value, locals, budget - 1)?;
                    locals.insert(name.clone(), v);
                }
                Stmt::Return { value, .. } => {
                    return match value {
                        Some(e) => self.ctfe_expr(e, locals, budget - 1),
                        None => Some(CtfeVal::Unit),
                    };
                }
                _ => return None,
            }
        }
        match &block.tail {
            Some(e) => self.ctfe_expr(e, locals, budget - 1),
            None => Some(CtfeVal::Unit),
        }
    }

    fn ctfe_expr(
        &self,
        expr: &Expr,
        locals: &mut BTreeMap<String, CtfeVal>,
        budget: usize,
    ) -> Option<CtfeVal> {
        if budget == 0 {
            return None;
        }
        match &expr.kind {
            ExprKind::Int(n) => Some(CtfeVal::Int(*n as i128)),
            ExprKind::Bool(b) => Some(CtfeVal::Bool(*b)),
            ExprKind::Ident(s) => locals.get(s).cloned(),
            ExprKind::Unary { op, expr } => match (op, self.ctfe_expr(expr, locals, budget - 1)?) {
                (UnOp::Neg, CtfeVal::Int(n)) => Some(CtfeVal::Int(-n)),
                (UnOp::Not, CtfeVal::Bool(b)) => Some(CtfeVal::Bool(!b)),
                (UnOp::BitNot, CtfeVal::Int(n)) => Some(CtfeVal::Int(!n)),
                _ => None,
            },
            ExprKind::Binary { op, left, right } => {
                let l = self.ctfe_expr(left, locals, budget - 1)?;
                let r = self.ctfe_expr(right, locals, budget - 1)?;
                match (op, l, r) {
                    (BinOp::Add, CtfeVal::Int(a), CtfeVal::Int(b)) => Some(CtfeVal::Int(a + b)),
                    (BinOp::Sub, CtfeVal::Int(a), CtfeVal::Int(b)) => Some(CtfeVal::Int(a - b)),
                    (BinOp::Mul, CtfeVal::Int(a), CtfeVal::Int(b)) => Some(CtfeVal::Int(a * b)),
                    (BinOp::Div, CtfeVal::Int(a), CtfeVal::Int(b)) if b != 0 => {
                        Some(CtfeVal::Int(a / b))
                    }
                    (BinOp::Mod, CtfeVal::Int(a), CtfeVal::Int(b)) if b != 0 => {
                        Some(CtfeVal::Int(a % b))
                    }
                    (BinOp::Eq, CtfeVal::Int(a), CtfeVal::Int(b)) => Some(CtfeVal::Bool(a == b)),
                    (BinOp::Ne, CtfeVal::Int(a), CtfeVal::Int(b)) => Some(CtfeVal::Bool(a != b)),
                    (BinOp::Lt, CtfeVal::Int(a), CtfeVal::Int(b)) => Some(CtfeVal::Bool(a < b)),
                    (BinOp::Le, CtfeVal::Int(a), CtfeVal::Int(b)) => Some(CtfeVal::Bool(a <= b)),
                    (BinOp::Gt, CtfeVal::Int(a), CtfeVal::Int(b)) => Some(CtfeVal::Bool(a > b)),
                    (BinOp::Ge, CtfeVal::Int(a), CtfeVal::Int(b)) => Some(CtfeVal::Bool(a >= b)),
                    (BinOp::And, CtfeVal::Bool(a), CtfeVal::Bool(b)) => Some(CtfeVal::Bool(a && b)),
                    (BinOp::Or, CtfeVal::Bool(a), CtfeVal::Bool(b)) => Some(CtfeVal::Bool(a || b)),
                    (BinOp::Implies, CtfeVal::Bool(a), CtfeVal::Bool(b)) => {
                        Some(CtfeVal::Bool(!a || b))
                    }
                    _ => None,
                }
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => match self.ctfe_expr(cond, locals, budget - 1)? {
                CtfeVal::Bool(true) => self.ctfe_block(then_block, locals, budget - 1),
                CtfeVal::Bool(false) => match else_branch {
                    Some(e) => self.ctfe_expr(e, locals, budget - 1),
                    None => Some(CtfeVal::Unit),
                },
                _ => None,
            },
            ExprKind::Match { scrutinee, arms } => {
                let sv = self.ctfe_expr(scrutinee, locals, budget - 1)?;
                for arm in arms {
                    let matched = arm.patterns.iter().any(|p| match (p, &sv) {
                        (Pattern::Wildcard(_), _) => true,
                        (Pattern::Int(n, _), CtfeVal::Int(v)) => (*n as i128) == *v,
                        (Pattern::Bool(b, _), CtfeVal::Bool(v)) => b == v,
                        _ => false,
                    });
                    if matched {
                        return self.ctfe_expr(&arm.body, locals, budget - 1);
                    }
                }
                None
            }
            ExprKind::Block(blk) => self.ctfe_block(blk, locals, budget - 1),
            ExprKind::Call { callee, args } => {
                if let ExprKind::GenericInst { expr: inner, .. } = &callee.kind {
                    if matches!(&inner.kind, ExprKind::Ident(n) if n == "refine_cast") && args.len() == 1 {
                        return self.ctfe_expr(&args[0].value, locals, budget - 1);
                    }
                }
                if let ExprKind::Ident(fname) = &callee.kind {
                    let mut vals = Vec::new();
                    for a in args {
                        vals.push(self.ctfe_expr(&a.value, locals, budget - 1)?);
                    }
                    return self.ctfe_call(fname, &vals, budget - 1);
                }
                None
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CtfeVal {
    Int(i128),
    Bool(bool),
    Unit,
}

fn canonical_pred_call(callee: &Expr, args: &[CallArg], self_var: Option<&str>) -> Option<String> {
    let ExprKind::Ident(fname) = &callee.kind else {
        return None;
    };
    let mut parts = Vec::with_capacity(args.len());
    for a in args {
        if let Some(v) = expr_var_name(&a.value, self_var) {
            parts.push(v);
        } else if let ExprKind::Int(n) = &a.value.kind {
            parts.push(n.to_string());
        } else {
            return None;
        }
    }
    Some(format!("{}({})", fname, parts.join(",")))
}

fn resolve_self(name: &str, self_var: Option<&str>) -> String {
    if name == "self" {
        self_var.unwrap_or("self").to_string()
    } else {
        name.to_string()
    }
}

fn expr_var_name(expr: &Expr, self_var: Option<&str>) -> Option<String> {
    match &expr.kind {
        ExprKind::Ident(s) => Some(resolve_self(s, self_var)),
        ExprKind::Field { target, field } => {
            let base = expr_var_name(target, self_var)?;
            Some(format!("{}.{}", base, field))
        }
        ExprKind::Old(inner) => {
            let base = expr_var_name(inner, self_var)?;
            Some(format!("old__{}", base.replace('.', "__")))
        }
        ExprKind::Call { callee, args }
            if matches!(&callee.kind, ExprKind::Ident(s) if s == "len") && args.len() == 1 =>
        {
            let target = expr_var_name(&args[0].value, self_var)?;
            Some(format!("__len__{}", target.replace('.', "__")))
        }
        _ => None,
    }
}
