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
    /// Difference bounds `a - b <= c` stored as `(a, b) -> c`.
    pub diffs: BTreeMap<(String, String), i128>,
    /// Flow-sensitively assumed pure predicate calls (e.g. `is_safe_rel_path(p)`).
    pub proved_calls: BTreeSet<String>,
    /// Pure function declarations available for bounded CTFE and ensures summaries.
    pub pure_fns: BTreeMap<String, FnDecl>,
    /// Refined type alias predicates available for callee return-type summaries.
    pub type_aliases: BTreeMap<String, TypeAliasDecl>,
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

    pub fn get_int(&self, name: &str) -> AbsInt {
        self.ints.get(name).copied().unwrap_or_else(AbsInt::top)
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
                        if let Some(TypeExpr::Named { name: rty, .. }) = &fdecl.ret_ty {
                            if let Some(alias) = self.type_aliases.get(rty) {
                                if let Some(pred) = &alias.predicate {
                                    summary_env.assume_expr(pred, Some("result"));
                                }
                            }
                        }
                        for ens in &fdecl.ensures {
                            summary_env.assume_expr(ens, Some("result"));
                        }
                        return summary_env.get_int("result");
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
        _ => None,
    }
}
