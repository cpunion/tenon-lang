use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use tenon_syntax::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtOutcome {
    Proved,
    Counterexample(BTreeMap<String, String>),
    Unknown(String),
}

static VC_CACHE: OnceLock<Mutex<BTreeMap<String, SmtOutcome>>> = OnceLock::new();

fn cache() -> &'static Mutex<BTreeMap<String, SmtOutcome>> {
    VC_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub struct SmtQuery {
    pub pure_fns: BTreeMap<String, FnDecl>,
    pub vars: BTreeMap<String, SmtSort>,
    pub premises: Vec<String>,
    pub goal: String,
    pub rlimit: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmtSort {
    Int,
    Bool,
}

impl SmtSort {
    fn as_str(self) -> &'static str {
        match self {
            SmtSort::Int => "Int",
            SmtSort::Bool => "Bool",
        }
    }
}

impl SmtQuery {
    pub fn new(pure_fns: BTreeMap<String, FnDecl>) -> Self {
        Self {
            pure_fns,
            vars: BTreeMap::new(),
            premises: Vec::new(),
            goal: "true".into(),
            rlimit: 500_000,
        }
    }

    pub fn declare_int(&mut self, name: impl Into<String>) {
        self.vars.insert(name.into(), SmtSort::Int);
    }

    pub fn declare_bool(&mut self, name: impl Into<String>) {
        self.vars.insert(name.into(), SmtSort::Bool);
    }

    pub fn add_premise_expr(&mut self, expr: &Expr, subst: &BTreeMap<String, String>) -> Result<(), String> {
        let s = encode_expr(expr, subst)?;
        self.premises.push(s);
        Ok(())
    }

    pub fn set_goal_expr(&mut self, expr: &Expr, subst: &BTreeMap<String, String>) -> Result<(), String> {
        self.goal = encode_expr(expr, subst)?;
        Ok(())
    }

    pub fn render_smtlib(&self) -> Result<String, String> {
        let mut out = String::new();
        out.push_str("(set-logic ALL)\n");
        out.push_str(&format!("(set-option :rlimit {})\n", self.rlimit));
        out.push_str("(set-option :smt.random_seed 42)\n");

        // Topologically emit pure functions (simple dependency sort by name length / two-pass)
        let ordered = order_pure_fns(&self.pure_fns);
        for f in ordered {
            if let Ok(def) = encode_pure_fn(&f) {
                out.push_str(&def);
                out.push('\n');
            }
        }

        for (name, sort) in &self.vars {
            out.push_str(&format!("(declare-const {} {})\n", smt_ident(name), sort.as_str()));
        }
        for p in &self.premises {
            out.push_str(&format!("(assert {})\n", p));
        }
        out.push_str(&format!("(assert (not {}))\n", self.goal));
        out.push_str("(check-sat)\n");
        out.push_str("(get-model)\n");
        Ok(out)
    }

    /// Execute Z3 with Canonical SHA-256 VC caching and strict Fail-Closed semantics.
    pub fn solve(&self) -> SmtOutcome {
        let script = match self.render_smtlib() {
            Ok(s) => s,
            Err(e) => return SmtOutcome::Unknown(format!("SMT encoding unsupported: {}", e)),
        };

        let mut hasher = Sha256::new();
        hasher.update(script.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        if let Some(hit) = cache().lock().unwrap().get(&hash).cloned() {
            return hit;
        }

        let outcome = run_z3(&script, &self.vars);
        // Only cache definite answers (Proved / Counterexample); never cache transient errors!
        if matches!(outcome, SmtOutcome::Proved | SmtOutcome::Counterexample(_)) {
            cache().lock().unwrap().insert(hash, outcome.clone());
        }
        outcome
    }
}

fn run_z3(script: &str, vars: &BTreeMap<String, SmtSort>) -> SmtOutcome {
    let mut child = match Command::new("z3")
        .args(["-in", "-smt2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return SmtOutcome::Unknown(format!(
                "failed to launch z3 (Fail-Closed: solver required for L1 obligations): {}",
                e
            ));
        }
    };

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(script.as_bytes());
    }

    let output = match child.wait_with_output() {
        Ok(o) => o,
        Err(e) => return SmtOutcome::Unknown(format!("z3 wait failed: {}", e)),
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout.lines().next().unwrap_or("").trim();
    match first_line {
        "unsat" => SmtOutcome::Proved,
        "sat" => {
            let model = parse_z3_model(&stdout, vars);
            SmtOutcome::Counterexample(model)
        }
        other => SmtOutcome::Unknown(format!(
            "z3 returned `{}` (Fail-Closed): {}",
            other,
            String::from_utf8_lossy(&output.stderr).trim()
        )),
    }
}

fn parse_z3_model(stdout: &str, vars: &BTreeMap<String, SmtSort>) -> BTreeMap<String, String> {
    let mut result = BTreeMap::new();
    let lines: Vec<&str> = stdout.lines().collect();
    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("(define-fun ") {
            let mut parts = rest.split_whitespace();
            let Some(sym) = parts.next() else {
                continue;
            };
            let Some((orig, _)) = vars.iter().find(|(k, _)| smt_ident(k) == sym) else {
                continue;
            };
            // Handle both single-line `(define-fun x () Int 1900)` and multi-line `(define-fun x () Int\n  1900)`
            let _params = parts.next();
            let _sort = parts.next();
            let inline_val: Vec<&str> = parts.collect();
            let val_raw = if !inline_val.is_empty() {
                inline_val.join(" ")
            } else if let Some(next_line) = lines.get(idx + 1) {
                next_line.trim().to_string()
            } else {
                continue;
            };
            let stripped = val_raw.trim().trim_end_matches(')').trim();
            let val = if let Some(neg) = stripped.strip_prefix("(- ") {
                format!("-{}", neg.trim())
            } else {
                stripped.to_string()
            };
            result.insert(orig.clone(), val);
        }
    }
    result
}

pub fn smt_ident(name: &str) -> String {
    name.replace('.', "__")
}

fn order_pure_fns(fns: &BTreeMap<String, FnDecl>) -> Vec<FnDecl> {
    let mut visited = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    fn visit(
        name: &str,
        fns: &BTreeMap<String, FnDecl>,
        visited: &mut std::collections::BTreeSet<String>,
        out: &mut Vec<FnDecl>,
    ) {
        if visited.contains(name) {
            return;
        }
        visited.insert(name.to_string());
        if let Some(f) = fns.get(name) {
            let mut deps = Vec::new();
            collect_calls_in_block(&f.body, &mut deps);
            for d in deps {
                if fns.contains_key(&d) {
                    visit(&d, fns, visited, out);
                }
            }
            out.push(f.clone());
        }
    }
    for name in fns.keys() {
        visit(name, fns, &mut visited, &mut out);
    }
    out
}

fn collect_calls_in_block(blk: &Block, out: &mut Vec<String>) {
    for s in &blk.stmts {
        match s {
            Stmt::Let { value, .. } => collect_calls_in_expr(value, out),
            Stmt::Return { value: Some(e), .. } => collect_calls_in_expr(e, out),
            Stmt::Expr { expr, .. } => collect_calls_in_expr(expr, out),
            _ => {}
        }
    }
    if let Some(t) = &blk.tail {
        collect_calls_in_expr(t, out);
    }
}

fn collect_calls_in_expr(e: &Expr, out: &mut Vec<String>) {
    match &e.kind {
        ExprKind::Call { callee, args } => {
            if let ExprKind::Ident(s) = &callee.kind {
                out.push(s.clone());
            }
            for a in args {
                collect_calls_in_expr(&a.value, out);
            }
        }
        ExprKind::Binary { left, right, .. } => {
            collect_calls_in_expr(left, out);
            collect_calls_in_expr(right, out);
        }
        ExprKind::Unary { expr, .. } => collect_calls_in_expr(expr, out),
        ExprKind::If {
            cond,
            then_block,
            else_branch,
        } => {
            collect_calls_in_expr(cond, out);
            collect_calls_in_block(then_block, out);
            if let Some(eb) = else_branch {
                collect_calls_in_expr(eb, out);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            collect_calls_in_expr(scrutinee, out);
            for a in arms {
                collect_calls_in_expr(&a.body, out);
            }
        }
        ExprKind::Block(b) => collect_calls_in_block(b, out),
        _ => {}
    }
}

fn type_to_smt_sort(ty: &TypeExpr) -> Result<&'static str, String> {
    match ty {
        TypeExpr::Named { name, args, .. } if args.is_empty() => match name.as_str() {
            "Bool" => Ok("Bool"),
            "Text" | "String" | "Any" => Err(format!("non-scalar SMT sort `{}`", name)),
            _ => Ok("Int"),
        },
        _ => Err(format!("unsupported SMT sort {:?}", ty)),
    }
}

pub fn can_encode_pure_fn(f: &FnDecl) -> bool {
    encode_pure_fn(f).is_ok()
}

fn encode_pure_fn(f: &FnDecl) -> Result<String, String> {
    if !f.is_pure {
        return Err("non-pure function".into());
    }
    let ret_sort = match &f.ret_ty {
        Some(t) => type_to_smt_sort(t)?,
        None => return Err("unit function".into()),
    };
    let mut params_str = String::new();
    for p in &f.params {
        let s = type_to_smt_sort(&p.ty)?;
        params_str.push_str(&format!("({} {}) ", smt_ident(&p.name), s));
    }
    let subst = BTreeMap::new();
    let body = encode_block(&f.body, &subst)?;
    Ok(format!(
        "(define-fun {} ({}) {} {})",
        smt_ident(&f.name),
        params_str.trim(),
        ret_sort,
        body
    ))
}

pub fn encode_block(blk: &Block, subst: &BTreeMap<String, String>) -> Result<String, String> {
    encode_stmts(&blk.stmts, blk.tail.as_deref(), subst)
}

fn encode_stmts(
    stmts: &[Stmt],
    tail: Option<&Expr>,
    subst: &BTreeMap<String, String>,
) -> Result<String, String> {
    let Some((first, rest)) = stmts.split_first() else {
        return match tail {
            Some(e) => encode_expr(e, subst),
            None => Err("empty block in pure SMT function".into()),
        };
    };
    match first {
        Stmt::Let { name, value, .. } => {
            let v = encode_expr(value, subst)?;
            let body = encode_stmts(rest, tail, subst)?;
            Ok(format!("(let (({} {})) {})", smt_ident(name), v, body))
        }
        Stmt::Return { value: Some(e), .. } => encode_expr(e, subst),
        Stmt::Expr {
            expr:
                Expr {
                    kind:
                        ExprKind::If {
                            cond,
                            then_block,
                            else_branch: None,
                        },
                    ..
                },
            ..
        } => {
            let c = encode_expr(cond, subst)?;
            let t = encode_block(then_block, subst)?;
            let e = encode_stmts(rest, tail, subst)?;
            Ok(format!("(ite {} {} {})", c, t, e))
        }
        _ => Err("unsupported statement in pure SMT function".into()),
    }
}

pub fn encode_expr(expr: &Expr, subst: &BTreeMap<String, String>) -> Result<String, String> {
    match &expr.kind {
        ExprKind::Int(n) => {
            if *n < 0 {
                Ok(format!("(- {})", -(*n as i128)))
            } else {
                Ok(n.to_string())
            }
        }
        ExprKind::Bool(b) => Ok(if *b { "true".into() } else { "false".into() }),
        ExprKind::Ident(s) => {
            if let Some(rep) = subst.get(s) {
                if rep.starts_with('(') {
                    Ok(rep.clone())
                } else {
                    Ok(smt_ident(rep))
                }
            } else {
                Ok(smt_ident(s))
            }
        }
        ExprKind::Field { target, field } => {
            if let ExprKind::Ident(base) = &target.kind {
                let full = format!("{}.{}", base, field);
                if let Some(rep) = subst.get(&full) {
                    return Ok(rep.clone());
                }
                if base == "self" {
                    if let Some(prefix) = subst.get("self") {
                        return Ok(smt_ident(&format!("{}.{}", prefix, field)));
                    }
                    return Ok(smt_ident(field));
                }
                return Ok(smt_ident(&full));
            }
            Err("complex field projection in SMT".into())
        }
        ExprKind::Unary { op, expr } => {
            let inner = encode_expr(expr, subst)?;
            match op {
                UnOp::Neg => Ok(format!("(- {})", inner)),
                UnOp::Not => Ok(format!("(not {})", inner)),
            }
        }
        ExprKind::Binary { op, left, right } => {
            let l = encode_expr(left, subst)?;
            let r = encode_expr(right, subst)?;
            let sym = match op {
                BinOp::And => "and",
                BinOp::Or => "or",
                BinOp::Implies => "=>",
                BinOp::Eq => "=",
                BinOp::Ne => return Ok(format!("(not (= {} {}))", l, r)),
                BinOp::Lt => "<",
                BinOp::Le => "<=",
                BinOp::Gt => ">",
                BinOp::Ge => ">=",
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "div",
                BinOp::Mod => "mod",
                _ => return Err(format!("unsupported SMT op {:?}", op)),
            };
            Ok(format!("({} {} {})", sym, l, r))
        }
        ExprKind::If {
            cond,
            then_block,
            else_branch,
        } => {
            let c = encode_expr(cond, subst)?;
            let t = encode_block(then_block, subst)?;
            let e = match else_branch {
                Some(eb) => encode_expr(eb, subst)?,
                None => return Err("if without else in pure SMT expression".into()),
            };
            Ok(format!("(ite {} {} {})", c, t, e))
        }
        ExprKind::Match { scrutinee, arms } => {
            let sc = encode_expr(scrutinee, subst)?;
            encode_match_arms(&sc, arms, subst)
        }
        ExprKind::Block(blk) => encode_block(blk, subst),
        ExprKind::Call { callee, args } => {
            if let ExprKind::Ident(fname) = &callee.kind {
                let mut enc_args = Vec::new();
                for a in args {
                    enc_args.push(encode_expr(&a.value, subst)?);
                }
                if enc_args.is_empty() {
                    Ok(smt_ident(fname))
                } else {
                    Ok(format!("({} {})", smt_ident(fname), enc_args.join(" ")))
                }
            } else {
                Err("indirect call in SMT".into())
            }
        }
        _ => Err(format!("unsupported expr in SMT: {:?}", expr.kind)),
    }
}

fn encode_match_arms(
    scrutinee: &str,
    arms: &[MatchArm],
    subst: &BTreeMap<String, String>,
) -> Result<String, String> {
    if arms.is_empty() {
        return Err("empty match".into());
    }
    let last = arms.last().unwrap();
    let mut acc = encode_expr(&last.body, subst)?;
    for arm in arms[..arms.len() - 1].iter().rev() {
        let body = encode_expr(&arm.body, subst)?;
        let mut conds = Vec::new();
        for pat in &arm.patterns {
            match pat {
                Pattern::Int(n, _) => conds.push(format!("(= {} {})", scrutinee, n)),
                Pattern::Bool(b, _) => {
                    conds.push(format!("(= {} {})", scrutinee, if *b { "true" } else { "false" }))
                }
                Pattern::Wildcard(_) => conds.push("true".into()),
                _ => return Err("unsupported pattern in SMT match".into()),
            }
        }
        let cond = if conds.len() == 1 {
            conds.pop().unwrap()
        } else {
            format!("(or {})", conds.join(" "))
        };
        acc = format!("(ite {} {} {})", cond, body, acc);
    }
    Ok(acc)
}
