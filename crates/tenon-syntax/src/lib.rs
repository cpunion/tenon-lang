pub mod ast;
pub mod derive;
pub mod lexer;
pub mod parser;
pub mod span;

pub use ast::*;
pub use derive::expand_derives;
pub use parser::parse;
pub use span::{ParseError, Span};

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Load a `.tn` source file and recursively resolve all `import "path.tn" [as alias]` declarations
/// in topological order with cycle-safe deduplication, then expand `@derive(...)` macros.
pub fn load_program_with_imports(entry: &Path) -> Result<Program, String> {
    let mut visited = BTreeSet::new();
    let mut module_exports: BTreeMap<PathBuf, (String, BTreeSet<String>)> = BTreeMap::new();
    let mut items = Vec::new();
    load_module_recursive(entry, &mut visited, &mut module_exports, &mut items)?;
    let mut prog = Program { items };
    expand_derives(&mut prog);
    Ok(prog)
}

fn resolve_import_path(current_file: &Path, import_rel: &str) -> Result<PathBuf, String> {
    let rel = Path::new(import_rel);
    if rel.is_absolute() && rel.exists() {
        return Ok(rel.to_path_buf());
    }
    let parent = current_file.parent().unwrap_or_else(|| Path::new("."));
    let direct = parent.join(rel);
    if direct.exists() {
        return Ok(direct);
    }
    // Walk up ancestor directories so `import "std/text.tn"` resolves from any example/subdir
    let mut cur = fs::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf());
    for _ in 0..8 {
        let cand = cur.join(rel);
        if cand.exists() {
            return Ok(cand);
        }
        if !cur.pop() {
            break;
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let cand = cwd.join(rel);
        if cand.exists() {
            return Ok(cand);
        }
    }
    Err(format!(
        "cannot resolve import `{}` from `{}`",
        import_rel,
        current_file.display()
    ))
}

fn load_module_recursive(
    file: &Path,
    visited: &mut BTreeSet<PathBuf>,
    module_exports: &mut BTreeMap<PathBuf, (String, BTreeSet<String>)>,
    out_items: &mut Vec<Item>,
) -> Result<(), String> {
    let canon = fs::canonicalize(file).unwrap_or_else(|_| file.to_path_buf());
    if !visited.insert(canon.clone()) {
        return Ok(());
    }
    let src = fs::read_to_string(&canon)
        .map_err(|e| format!("failed to read `{}`: {}", file.display(), e))?;
    let prog = parse(&src).map_err(|e| format!("{}: {}", file.display(), e))?;

    let mut local_aliases: BTreeMap<String, (String, BTreeSet<String>)> = BTreeMap::new();
    let mut local_items = Vec::new();
    let mut exported_fns = BTreeSet::new();

    for item in prog.items {
        match item {
            Item::Import { path, alias, .. } => {
                let target = resolve_import_path(&canon, &path)?;
                let target_canon = fs::canonicalize(&target).unwrap_or_else(|_| target.clone());
                load_module_recursive(&target, visited, module_exports, out_items)?;
                if let Some(a) = alias {
                    if let Some(exp) = module_exports.get(&target_canon) {
                        local_aliases.insert(a, exp.clone());
                    }
                }
            }
            other => {
                if let Item::Fn(f) = &other {
                    exported_fns.insert(f.name.clone());
                }
                local_items.push(other);
            }
        }
    }

    let stem = canon
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();
    module_exports.insert(canon, (stem, exported_fns));

    if !local_aliases.is_empty() {
        for item in &mut local_items {
            rewrite_item_aliases(item, &local_aliases);
        }
    }
    out_items.extend(local_items);
    Ok(())
}

fn rewrite_item_aliases(item: &mut Item, aliases: &BTreeMap<String, (String, BTreeSet<String>)>) {
    match item {
        Item::Fn(f) => rewrite_fn_aliases(f, aliases),
        Item::Impl(imp) => {
            for m in &mut imp.methods {
                rewrite_fn_aliases(m, aliases);
            }
        }
        Item::Interface(iface) => {
            for m in &mut iface.methods {
                if let Some(b) = &mut m.default_body {
                    rewrite_block_aliases(b, aliases);
                }
            }
        }
        Item::Record(r) => {
            for m in &mut r.methods {
                rewrite_fn_aliases(m, aliases);
            }
        }
        _ => {}
    }
}

fn rewrite_fn_aliases(f: &mut FnDecl, aliases: &BTreeMap<String, (String, BTreeSet<String>)>) {
    for r in &mut f.requires {
        rewrite_expr_aliases(r, aliases);
    }
    for e in &mut f.ensures {
        rewrite_expr_aliases(e, aliases);
    }
    rewrite_block_aliases(&mut f.body, aliases);
}

fn rewrite_block_aliases(blk: &mut Block, aliases: &BTreeMap<String, (String, BTreeSet<String>)>) {
    for s in &mut blk.stmts {
        match s {
            Stmt::Let { value, .. } => rewrite_expr_aliases(value, aliases),
            Stmt::Assign { target, value, .. } => {
                rewrite_expr_aliases(target, aliases);
                rewrite_expr_aliases(value, aliases);
            }
            Stmt::Return { value: Some(e), .. }
            | Stmt::Assert { cond: e, .. }
            | Stmt::Expr { expr: e, .. } => rewrite_expr_aliases(e, aliases),
            Stmt::Mutate { body, .. } => rewrite_block_aliases(body, aliases),
            Stmt::While { cond, body, .. } => {
                rewrite_expr_aliases(cond, aliases);
                rewrite_block_aliases(body, aliases);
            }
            _ => {}
        }
    }
    if let Some(t) = &mut blk.tail {
        rewrite_expr_aliases(t, aliases);
    }
}

fn rewrite_expr_aliases(expr: &mut Expr, aliases: &BTreeMap<String, (String, BTreeSet<String>)>) {
    match &mut expr.kind {
        ExprKind::Field { target, field } => {
            if let ExprKind::Ident(m) = &target.kind {
                if let Some((stem, exports)) = aliases.get(m) {
                    if exports.contains(field.as_str()) {
                        let resolved = field.clone();
                        expr.kind = ExprKind::Ident(resolved);
                        return;
                    }
                    let prefixed = format!("{}_{}", stem, field);
                    if exports.contains(&prefixed) {
                        expr.kind = ExprKind::Ident(prefixed);
                        return;
                    }
                }
            }
            rewrite_expr_aliases(target, aliases);
        }
        ExprKind::GenericInst { expr: inner, .. } => rewrite_expr_aliases(inner, aliases),
        ExprKind::Call { callee, args } => {
            rewrite_expr_aliases(callee, aliases);
            for a in args {
                rewrite_expr_aliases(&mut a.value, aliases);
            }
        }
        ExprKind::Binary { left, right, .. } => {
            rewrite_expr_aliases(left, aliases);
            rewrite_expr_aliases(right, aliases);
        }
        ExprKind::Unary { expr: inner, .. } | ExprKind::Await(inner) | ExprKind::Old(inner) => {
            rewrite_expr_aliases(inner, aliases);
        }
        ExprKind::Index { target, index } => {
            rewrite_expr_aliases(target, aliases);
            rewrite_expr_aliases(index, aliases);
        }
        ExprKind::RecordLit { fields, .. } => {
            for (_, v) in fields {
                rewrite_expr_aliases(v, aliases);
            }
        }
        ExprKind::Tuple(es) | ExprKind::List(es) => {
            for e in es {
                rewrite_expr_aliases(e, aliases);
            }
        }
        ExprKind::If {
            cond,
            then_block,
            else_branch,
        } => {
            rewrite_expr_aliases(cond, aliases);
            rewrite_block_aliases(then_block, aliases);
            if let Some(eb) = else_branch {
                rewrite_expr_aliases(eb, aliases);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            rewrite_expr_aliases(scrutinee, aliases);
            for a in arms {
                if let Some(g) = &mut a.guard {
                    rewrite_expr_aliases(g, aliases);
                }
                rewrite_expr_aliases(&mut a.body, aliases);
            }
        }
        ExprKind::Block(b) => rewrite_block_aliases(b, aliases),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_date_and_refinements() {
        let src = r#"
pub type Amount = Int64 where self > 0
pub type Odd = Int64 where self % 2 != 0
pub type Even = Int64 where self % 2 == 0

pub pure fn is_leap_year(y Int) Bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

pub pure fn days_in_month(y Int, m Int where 1 <= self && self <= 12) Int
    ensures 28 <= result && result <= 31
{
    match m {
        2 => if is_leap_year(y) { 29 } else { 28 }
        4 | 6 | 9 | 11 => 30
        _ => 31
    }
}

pub record Date {
    year  Int where 1 <= self && self <= 9999
    month Int where 1 <= self && self <= 12
    day   Int where 1 <= self && self <= days_in_month(year, month)
}
"#;
        let prog = parse(src).expect("should parse Date example");
        assert_eq!(prog.items.len(), 6);
    }

    #[test]
    fn parses_evolution_and_distributed_views() {
        let src = r#"
@evolve("billing.Order", storage = "orders")
pub record Order {
    @field(1) id Int64
    @field(2) amount Int64 where self > 0
}

evolution Order: v1 <-> v2 {
    archive ZeroOrder {
        order_id Int64
        amount Int64 where self == 0
    }
    up(old Order@v1) (Order@v2, Option[ZeroOrder]) {
        if old.amount == 0 {
            (Order@v2 { id = old.id, amount = 1 }, Some(ZeroOrder { order_id = old.id, amount = 0 }))
        } else {
            (Order@v2 { id = old.id, amount = old.amount }, None)
        }
    }
    down(new Order@v2, arch Option[ZeroOrder]) Order@v1
        retain(new.note)
    {
        Order@v1 { id = new.id, amount = new.amount }
    }
}

@sync_view("tasks.list")
view TasksListView from TaskAuthority {
    durability = Reconstructible
    bounds {
        max_items = 1000
        max_payload = 16 MiB
        max_frame = 4 MiB
        journal_budget = 16 MiB
        offline_recovery = BoundedReset
    }
}
"#;
        let prog = parse(src).expect("should parse evolution and sync view");
        assert_eq!(prog.items.len(), 3);
    }

    #[test]
    fn resolves_recursive_imports_utf8_and_loop_control() {
        let dir = std::env::temp_dir().join(format!("tenon_import_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("std")).unwrap();
        fs::write(
            dir.join("std/base.tn"),
            "pub fn greet() Text { \"你好 Tenon\" }\n",
        )
        .unwrap();
        fs::write(
            dir.join("std/helper.tn"),
            "import \"std/base.tn\"\npub fn count_to(n Int) Int {\n  var i = 0\n  while i < n {\n    i += 1\n    if i == 2 { continue }\n    if i == 5 { break }\n  }\n  i\n}\n",
        )
        .unwrap();
        let entry = dir.join("app.tn");
        fs::write(
            &entry,
            "import \"std/helper.tn\"\nimport \"std/base.tn\"\npub fn main() Int { count_to(10) }\n",
        )
        .unwrap();

        let prog = load_program_with_imports(&entry).expect("should load with imports");
        let _ = fs::remove_dir_all(&dir);
        // base.tn deduplicated: greet, count_to, main
        assert_eq!(prog.items.len(), 3);
        if let Item::Fn(f) = &prog.items[0] {
            assert_eq!(f.name, "greet");
            if let Some(tail) = &f.body.tail {
                assert_eq!(tail.kind, ExprKind::Str("你好 Tenon".into()));
            } else {
                panic!("expected tail str");
            }
        } else {
            panic!("expected greet first");
        }
    }
}
