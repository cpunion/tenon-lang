use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use tenon_syntax::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmtOutcome {
    Proved,
    Counterexample(BTreeMap<String, String>),
    Unknown(String),
}

static VC_CACHE: OnceLock<Mutex<BTreeMap<String, SmtOutcome>>> = OnceLock::new();

fn disk_cache_path() -> Option<PathBuf> {
    if std::env::var_os("TENON_NO_DISK_CACHE").is_some() {
        return None;
    }
    let cwd = std::env::current_dir().ok()?;
    Some(cwd.join(".tenon-cache").join("vc_cache_v1.json"))
}

fn cache() -> &'static Mutex<BTreeMap<String, SmtOutcome>> {
    VC_CACHE.get_or_init(|| {
        let mut map = BTreeMap::new();
        if let Some(p) = disk_cache_path() {
            if let Ok(bytes) = std::fs::read(&p) {
                if let Ok(loaded) = serde_json::from_slice::<BTreeMap<String, SmtOutcome>>(&bytes) {
                    map = loaded;
                }
            }
        }
        Mutex::new(map)
    })
}

fn persist_disk_cache(map: &BTreeMap<String, SmtOutcome>) {
    let Some(path) = disk_cache_path() else {
        return;
    };
    let Some(parent) = path.parent() else {
        return;
    };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    if let Ok(json) = serde_json::to_vec(map) {
        let tmp = parent.join(format!("vc_cache_v1.{}.tmp", std::process::id()));
        if std::fs::write(&tmp, &json).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }
}

pub fn vc_cache_len() -> usize {
    cache().lock().unwrap().len()
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
    Real,
}

impl SmtSort {
    fn as_str(self) -> &'static str {
        match self {
            SmtSort::Int => "Int",
            SmtSort::Bool => "Bool",
            SmtSort::Real => "Real",
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

    pub fn declare_real(&mut self, name: impl Into<String>) {
        self.vars.insert(name.into(), SmtSort::Real);
    }

    fn real_var_set(&self) -> BTreeSet<String> {
        self.vars
            .iter()
            .filter_map(|(k, v)| if *v == SmtSort::Real { Some(k.clone()) } else { None })
            .collect()
    }

    pub fn add_premise_expr(&mut self, expr: &Expr, subst: &BTreeMap<String, String>) -> Result<(), String> {
        let rvars = self.real_var_set();
        let s = encode_expr_with_reals(expr, subst, &rvars)?;
        self.premises.push(s);
        Ok(())
    }

    pub fn set_goal_expr(&mut self, expr: &Expr, subst: &BTreeMap<String, String>) -> Result<(), String> {
        let rvars = self.real_var_set();
        self.goal = encode_expr_with_reals(expr, subst, &rvars)?;
        Ok(())
    }

    pub fn render_smtlib(&self) -> Result<String, String> {
        let mut out = String::new();
        out.push_str("(set-logic ALL)\n");
        out.push_str(&format!("(set-option :rlimit {})\n", self.rlimit));
        out.push_str("(set-option :smt.random_seed 42)\n");

        // Topologically emit pure functions; recursive pure functions get sound bounded fuel unrolling (k=4).
        let ordered = order_pure_fns(&self.pure_fns);
        let mut emitted = BTreeSet::new();
        for f in ordered {
            let mut deps = Vec::new();
            collect_calls_in_block(&f.body, &mut deps);
            let is_rec = deps.iter().any(|d| d == &f.name);
            let other_deps_ready = deps.iter().all(|d| d == &f.name || emitted.contains(d));
            if !other_deps_ready {
                continue;
            }
            if !is_rec {
                if let Ok(def) = encode_pure_fn(&f) {
                    out.push_str(&def);
                    out.push('\n');
                    emitted.insert(f.name.clone());
                }
            } else if let Ok(defs) = encode_recursive_pure_fn(&f, 4) {
                out.push_str(&defs);
                out.push('\n');
                emitted.insert(f.name.clone());
            }
        }

        // Auto-declare non-negative `__len__*` variables referenced in premises or goal
        let mut auto_len_vars = BTreeSet::new();
        for s in self.premises.iter().chain(std::iter::once(&self.goal)) {
            collect_len_symbols(s, &mut auto_len_vars);
        }
        let mut all_vars = self.vars.clone();
        for lv in &auto_len_vars {
            all_vars.entry(lv.clone()).or_insert(SmtSort::Int);
        }

        for (name, sort) in &all_vars {
            out.push_str(&format!("(declare-const {} {})\n", smt_ident(name), sort.as_str()));
        }
        for lv in &auto_len_vars {
            out.push_str(&format!("(assert (>= {} 0))\n", smt_ident(lv)));
        }
        for p in &self.premises {
            out.push_str(&format!("(assert {})\n", p));
        }
        out.push_str(&format!("(assert (not {}))\n", self.goal));
        out.push_str("(check-sat)\n");
        out.push_str("(get-model)\n");
        Ok(out)
    }

    /// Execute Z3 with Canonical SHA-256 VC caching (memory + `.tenon-cache/vc_cache_v1.json`)
    /// and strict Fail-Closed semantics.
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
            let mut guard = cache().lock().unwrap();
            guard.insert(hash, outcome.clone());
            persist_disk_cache(&guard);
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
            "Float" | "Float32" | "Float64" => Ok("Real"),
            "Text" | "String" | "Any" => Err(format!("non-scalar SMT sort `{}`", name)),
            _ => Ok("Int"),
        },
        _ => Err(format!("unsupported SMT sort {:?}", ty)),
    }
}

pub fn can_encode_pure_fn(f: &FnDecl) -> bool {
    encode_pure_fn(f).is_ok()
}

pub fn pure_fn_callees(f: &FnDecl) -> Vec<String> {
    let mut deps = Vec::new();
    collect_calls_in_block(&f.body, &mut deps);
    deps
}

fn collect_len_symbols(s: &str, out: &mut BTreeSet<String>) {
    let bytes = s.as_bytes();
    let needle = b"__len__";
    let mut i = 0;
    while i + needle.len() <= bytes.len() {
        if &bytes[i..i + needle.len()] == needle {
            let mut end = i + needle.len();
            while end < bytes.len()
                && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_')
            {
                end += 1;
            }
            if end > i + needle.len() {
                if let Ok(tok) = std::str::from_utf8(&bytes[i..end]) {
                    out.insert(tok.to_string());
                }
            }
            i = end;
        } else {
            i += 1;
        }
    }
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
    let mut real_vars = BTreeSet::new();
    for p in &f.params {
        let s = type_to_smt_sort(&p.ty)?;
        if s == "Real" {
            real_vars.insert(p.name.clone());
        }
        params_str.push_str(&format!("({} {}) ", smt_ident(&p.name), s));
    }
    let subst = BTreeMap::new();
    let body = encode_block_with_reals(&f.body, &subst, &real_vars)?;
    Ok(format!(
        "(define-fun {} ({}) {} {})",
        smt_ident(&f.name),
        params_str.trim(),
        ret_sort,
        body
    ))
}

fn encode_recursive_pure_fn(f: &FnDecl, fuel: usize) -> Result<String, String> {
    if !f.is_pure {
        return Err("non-pure function".into());
    }
    let ret_sort = match &f.ret_ty {
        Some(t) => type_to_smt_sort(t)?,
        None => return Err("unit function".into()),
    };
    let mut param_sorts = Vec::new();
    let mut param_decls = Vec::new();
    let mut param_names = Vec::new();
    let mut real_vars = BTreeSet::new();
    for p in &f.params {
        let s = type_to_smt_sort(&p.ty)?;
        if s == "Real" {
            real_vars.insert(p.name.clone());
        }
        param_sorts.push(s.to_string());
        param_decls.push(format!("({} {})", smt_ident(&p.name), s));
        param_names.push(smt_ident(&p.name));
    }
    let fname = smt_ident(&f.name);
    let mut out = String::new();
    out.push_str(&format!(
        "(declare-fun {}__0 ({}) {})\n",
        fname,
        param_sorts.join(" "),
        ret_sort
    ));
    for k in 1..=fuel {
        let mut subst = BTreeMap::new();
        subst.insert(format!("__fn__{}", f.name), format!("{}__{}", fname, k - 1));
        let body = encode_block_with_reals(&f.body, &subst, &real_vars)?;
        out.push_str(&format!(
            "(define-fun {}__{} ({}) {} {})\n",
            fname,
            k,
            param_decls.join(" "),
            ret_sort,
            body
        ));
    }
    let call_top = if param_names.is_empty() {
        format!("{}__{}", fname, fuel)
    } else {
        format!("({}__{} {})", fname, fuel, param_names.join(" "))
    };
    out.push_str(&format!(
        "(define-fun {} ({}) {} {})",
        fname,
        param_decls.join(" "),
        ret_sort,
        call_top
    ));
    Ok(out)
}

pub fn encode_block(blk: &Block, subst: &BTreeMap<String, String>) -> Result<String, String> {
    encode_block_with_reals(blk, subst, &BTreeSet::new())
}

pub fn encode_block_with_reals(
    blk: &Block,
    subst: &BTreeMap<String, String>,
    real_vars: &BTreeSet<String>,
) -> Result<String, String> {
    encode_stmts(&blk.stmts, blk.tail.as_deref(), subst, real_vars)
}

fn encode_stmts(
    stmts: &[Stmt],
    tail: Option<&Expr>,
    subst: &BTreeMap<String, String>,
    real_vars: &BTreeSet<String>,
) -> Result<String, String> {
    let Some((first, rest)) = stmts.split_first() else {
        return match tail {
            Some(e) => encode_expr_with_reals(e, subst, real_vars),
            None => Err("empty block in pure SMT function".into()),
        };
    };
    match first {
        Stmt::Let { name, value, .. } => {
            let v = encode_expr_with_reals(value, subst, real_vars)?;
            let body = encode_stmts(rest, tail, subst, real_vars)?;
            Ok(format!("(let (({} {})) {})", smt_ident(name), v, body))
        }
        Stmt::Return { value: Some(e), .. } => encode_expr_with_reals(e, subst, real_vars),
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
            let c = encode_expr_with_reals(cond, subst, real_vars)?;
            let t = encode_block_with_reals(then_block, subst, real_vars)?;
            let e = encode_stmts(rest, tail, subst, real_vars)?;
            Ok(format!("(ite {} {} {})", c, t, e))
        }
        _ => Err("unsupported statement in pure SMT function".into()),
    }
}

fn format_smt_real(x: f64) -> Result<String, String> {
    if !x.is_finite() {
        return Err("non-finite float in SMT".into());
    }
    let abs = x.abs();
    let s = format!("{:.12}", abs);
    let trimmed = s.trim_end_matches('0');
    let dec = if trimmed.ends_with('.') {
        format!("{}0", trimmed)
    } else {
        trimmed.to_string()
    };
    if x < 0.0 {
        Ok(format!("(- {})", dec))
    } else {
        Ok(dec)
    }
}

fn expr_is_real(expr: &Expr, real_vars: &BTreeSet<String>) -> bool {
    match &expr.kind {
        ExprKind::Float(_) => true,
        ExprKind::Ident(s) => real_vars.contains(s),
        ExprKind::Old(inner) => expr_is_real(inner, real_vars),
        ExprKind::Unary { expr: inner, .. } => expr_is_real(inner, real_vars),
        ExprKind::Binary { left, right, .. } => {
            expr_is_real(left, real_vars) || expr_is_real(right, real_vars)
        }
        ExprKind::Cast { ty, .. } => matches!(
            ty.display_name().as_str(),
            "Float" | "Float32" | "Float64"
        ),
        _ => false,
    }
}

pub fn encode_expr(expr: &Expr, subst: &BTreeMap<String, String>) -> Result<String, String> {
    encode_expr_with_reals(expr, subst, &BTreeSet::new())
}

pub fn encode_expr_with_reals(
    expr: &Expr,
    subst: &BTreeMap<String, String>,
    real_vars: &BTreeSet<String>,
) -> Result<String, String> {
    match &expr.kind {
        ExprKind::Int(n) => {
            if *n < 0 {
                Ok(format!("(- {})", -(*n as i128)))
            } else {
                Ok(n.to_string())
            }
        }
        ExprKind::Float(x) => format_smt_real(*x),
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
        ExprKind::Old(inner) => match &inner.kind {
            ExprKind::Ident(s) => Ok(smt_ident(&format!("old__{}", s))),
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(base) = &target.kind {
                    Ok(smt_ident(&format!("old__{}__{}", base, field)))
                } else {
                    Err("complex old(...) target in SMT".into())
                }
            }
            _ => Err("old(...) expects variable or field in SMT".into()),
        },
        ExprKind::Cast { expr: inner, ty } => {
            let s = encode_expr_with_reals(inner, subst, real_vars)?;
            match ty.display_name().as_str() {
                "Float" | "Float32" | "Float64" => {
                    if expr_is_real(inner, real_vars) {
                        Ok(s)
                    } else {
                        Ok(format!("(to_real {})", s))
                    }
                }
                "Int" | "Int64" | "Int32" | "Int16" | "Int8" | "UInt8" | "UInt16" | "UInt32"
                | "UInt64" | "USize" | "ISize" => {
                    if expr_is_real(inner, real_vars) {
                        Ok(format!("(to_int {})", s))
                    } else {
                        Ok(s)
                    }
                }
                _ => Ok(s),
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
            let inner = encode_expr_with_reals(expr, subst, real_vars)?;
            match op {
                UnOp::Neg => Ok(format!("(- {})", inner)),
                UnOp::Not => Ok(format!("(not {})", inner)),
                UnOp::BitNot => Ok(format!("(bv2nat (bvnot ((_ int2bv 64) {})))", inner)),
            }
        }
        ExprKind::Binary { op, left, right } => {
            let is_real_op = expr_is_real(left, real_vars) || expr_is_real(right, real_vars);
            let mut l = encode_expr_with_reals(left, subst, real_vars)?;
            let mut r = encode_expr_with_reals(right, subst, real_vars)?;
            if is_real_op {
                if !expr_is_real(left, real_vars) {
                    l = format!("(to_real {})", l);
                }
                if !expr_is_real(right, real_vars) {
                    r = format!("(to_real {})", r);
                }
            }
            match op {
                BinOp::BitAnd => {
                    return Ok(format!(
                        "(bv2nat (bvand ((_ int2bv 64) {}) ((_ int2bv 64) {})))",
                        l, r
                    ));
                }
                BinOp::BitOr => {
                    return Ok(format!(
                        "(bv2nat (bvor ((_ int2bv 64) {}) ((_ int2bv 64) {})))",
                        l, r
                    ));
                }
                BinOp::BitXor => {
                    return Ok(format!(
                        "(bv2nat (bvxor ((_ int2bv 64) {}) ((_ int2bv 64) {})))",
                        l, r
                    ));
                }
                BinOp::Shl => {
                    return Ok(format!(
                        "(bv2nat (bvshl ((_ int2bv 64) {}) ((_ int2bv 64) {})))",
                        l, r
                    ));
                }
                BinOp::Shr => {
                    return Ok(format!(
                        "(bv2nat (bvlshr ((_ int2bv 64) {}) ((_ int2bv 64) {})))",
                        l, r
                    ));
                }
                _ => {}
            }
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
                BinOp::Div => if is_real_op { "/" } else { "div" },
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
            let c = encode_expr_with_reals(cond, subst, real_vars)?;
            let t = encode_block_with_reals(then_block, subst, real_vars)?;
            let e = match else_branch {
                Some(eb) => encode_expr_with_reals(eb, subst, real_vars)?,
                None => return Err("if without else in pure SMT expression".into()),
            };
            Ok(format!("(ite {} {} {})", c, t, e))
        }
        ExprKind::Match { scrutinee, arms } => {
            let sc = encode_expr_with_reals(scrutinee, subst, real_vars)?;
            encode_match_arms(&sc, arms, subst, real_vars)
        }
        ExprKind::Block(blk) => encode_block_with_reals(blk, subst, real_vars),
        ExprKind::Call { callee, args } => {
            if let ExprKind::Ident(fname) = &callee.kind {
                if fname == "len" && args.len() == 1 {
                    if let ExprKind::Ident(var) = &args[0].value.kind {
                        let resolved = subst.get(var).map(|s| s.as_str()).unwrap_or(var.as_str());
                        return Ok(smt_ident(&format!("__len__{}", resolved)));
                    }
                }
                let call_name = subst
                    .get(&format!("__fn__{}", fname))
                    .cloned()
                    .unwrap_or_else(|| smt_ident(fname));
                let mut enc_args = Vec::new();
                for a in args {
                    enc_args.push(encode_expr_with_reals(&a.value, subst, real_vars)?);
                }
                if enc_args.is_empty() {
                    Ok(call_name)
                } else {
                    Ok(format!("({} {})", call_name, enc_args.join(" ")))
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
    real_vars: &BTreeSet<String>,
) -> Result<String, String> {
    if arms.is_empty() {
        return Err("empty match".into());
    }
    let last = arms.last().unwrap();
    let mut acc = encode_expr_with_reals(&last.body, subst, real_vars)?;
    for arm in arms[..arms.len() - 1].iter().rev() {
        let body = encode_expr_with_reals(&arm.body, subst, real_vars)?;
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
