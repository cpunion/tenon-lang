pub mod ast;
pub mod derive;
pub mod layout;
pub mod lexer;
pub mod parser;
pub mod span;

pub use ast::*;
pub use derive::expand_derives;
pub use layout::*;
pub use parser::{parse, parse_with_file_id};
pub use span::{ParseError, Span};

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Evaluate `@cfg(...)` conditional compilation attributes and drop disabled items.
pub fn eval_cfg_attrs(program: &mut Program) {
    program.items.retain(|item| {
        let attrs: &[Attr] = match item {
            Item::TypeAlias(d) => &d.attrs,
            Item::Record(d) => &d.attrs,
            Item::Enum(d) => &d.attrs,
            Item::Interface(d) => &d.attrs,
            Item::Impl(d) => &d.attrs,
            Item::Fn(d) => &d.attrs,
            Item::ExternFn(d) => &d.attrs,
            Item::Bitfield(d) => &d.attrs,
            Item::Union(d) => &d.attrs,
            Item::Const(d) => &d.attrs,
            Item::Evolution(d) => &d.attrs,
            Item::Protocol(d) => &d.attrs,
            Item::Stream(d) => &d.attrs,
            Item::View(d) => &d.attrs,
            Item::Subscriber(d) => &d.attrs,
            Item::Import { .. } | Item::LockLattice(_) => return true,
        };
        attrs.iter().filter(|a| a.name == "cfg").all(eval_cfg_attr)
    });
}

fn eval_cfg_attr(attr: &Attr) -> bool {
    for arg in &attr.args {
        if let Some(k) = &arg.key {
            let val = match &arg.value.kind {
                ExprKind::Str(s) | ExprKind::Ident(s) => s.as_str(),
                ExprKind::Int(n) => {
                    if k == "target_pointer_width" {
                        if (*n as usize) != std::mem::size_of::<usize>() * 8 {
                            return false;
                        }
                        continue;
                    }
                    ""
                }
                _ => "",
            };
            match k.as_str() {
                "os" | "target_os" => {
                    if val != std::env::consts::OS {
                        return false;
                    }
                }
                "arch" | "target_arch" => {
                    if val != std::env::consts::ARCH {
                        return false;
                    }
                }
                "family" | "target_family" => {
                    if val != std::env::consts::FAMILY {
                        return false;
                    }
                }
                "target_pointer_width" => {
                    let expected = (std::mem::size_of::<usize>() * 8).to_string();
                    if val != expected {
                        return false;
                    }
                }
                _ => {}
            }
        } else if let ExprKind::Call { callee, args } = &arg.value.kind {
            if let ExprKind::Ident(fname) = &callee.kind {
                if fname == "not" && args.len() == 1 {
                    let inner_attr = Attr {
                        name: "cfg".into(),
                        args: vec![AttrArg {
                            key: args[0].name.clone(),
                            value: args[0].value.clone(),
                        }],
                        span: attr.span,
                    };
                    if eval_cfg_attr(&inner_attr) {
                        return false;
                    }
                }
            }
        } else if let ExprKind::Ident(flag) = &arg.value.kind {
            match flag.as_str() {
                "unix" => {
                    if std::env::consts::FAMILY != "unix" {
                        return false;
                    }
                }
                "windows" => {
                    if std::env::consts::OS != "windows" {
                        return false;
                    }
                }
                "macos" => {
                    if std::env::consts::OS != "macos" {
                        return false;
                    }
                }
                "linux" => {
                    if std::env::consts::OS != "linux" {
                        return false;
                    }
                }
                "debug" => {
                    if !cfg!(debug_assertions) {
                        return false;
                    }
                }
                _ => {}
            }
        }
    }
    true
}

/// Load a `.tn` source file and recursively resolve all `import "path.tn" [as alias]` declarations
/// in topological order with cycle-safe deduplication, then evaluate `@cfg` and expand `@derive(...)` macros.
pub fn load_program_with_imports(entry: &Path) -> Result<Program, String> {
    let mut visited = BTreeSet::new();
    let mut module_exports: BTreeMap<PathBuf, (String, BTreeSet<String>)> = BTreeMap::new();
    let mut source_files = vec!["<inline>".to_string()];
    let mut items = Vec::new();
    load_module_recursive(
        entry,
        &mut visited,
        &mut module_exports,
        &mut source_files,
        &mut items,
    )?;
    let mut prog = Program {
        items,
        source_files,
    };
    eval_cfg_attrs(&mut prog);
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
    source_files: &mut Vec<String>,
    out_items: &mut Vec<Item>,
) -> Result<(), String> {
    let canon = fs::canonicalize(file).unwrap_or_else(|_| file.to_path_buf());
    if !visited.insert(canon.clone()) {
        return Ok(());
    }
    let file_id = source_files.len() as u32;
    source_files.push(canon.to_string_lossy().into_owned());
    let src = fs::read_to_string(&canon)
        .map_err(|e| format!("failed to read `{}`: {}", file.display(), e))?;
    let prog = parse_with_file_id(&src, file_id).map_err(|e| format!("{}: {}", file.display(), e))?;

    let mut local_aliases: BTreeMap<String, (String, BTreeSet<String>)> = BTreeMap::new();
    let mut local_items = Vec::new();
    let mut exported_fns = BTreeSet::new();

    for item in prog.items {
        match item {
            Item::Import { path, alias, .. } => {
                let target = resolve_import_path(&canon, &path)?;
                let target_canon = fs::canonicalize(&target).unwrap_or_else(|_| target.clone());
                load_module_recursive(&target, visited, module_exports, source_files, out_items)?;
                if let Some(a) = alias {
                    if let Some(exp) = module_exports.get(&target_canon) {
                        local_aliases.insert(a, exp.clone());
                    }
                }
            }
            other => {
                match &other {
                    Item::Fn(f) => {
                        exported_fns.insert(f.name.clone());
                    }
                    Item::ExternFn(ef) => {
                        exported_fns.insert(ef.name.clone());
                    }
                    Item::Const(c) => {
                        exported_fns.insert(c.name.clone());
                    }
                    _ => {}
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
        Item::Const(c) => rewrite_expr_aliases(&mut c.value, aliases),
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
            Stmt::Let { value, .. } | Stmt::LetTuple { value, .. } => {
                rewrite_expr_aliases(value, aliases)
            }
            Stmt::Assign { target, value, .. } => {
                rewrite_expr_aliases(target, aliases);
                rewrite_expr_aliases(value, aliases);
            }
            Stmt::Return { value: Some(e), .. }
            | Stmt::Assert { cond: e, .. }
            | Stmt::Expr { expr: e, .. } => rewrite_expr_aliases(e, aliases),
            Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } => {
                rewrite_block_aliases(body, aliases)
            }
            Stmt::While { cond, body, .. } => {
                rewrite_expr_aliases(cond, aliases);
                rewrite_block_aliases(body, aliases);
            }
            Stmt::For { iter, body, .. } => {
                rewrite_expr_aliases(iter, aliases);
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
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Await(inner)
        | ExprKind::Old(inner)
        | ExprKind::Try(inner)
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::ArrayRepeat { value: inner, .. }
        | ExprKind::TupleIndex { target: inner, .. } => {
            rewrite_expr_aliases(inner, aliases);
        }
        ExprKind::Range { start, end } => {
            rewrite_expr_aliases(start, aliases);
            rewrite_expr_aliases(end, aliases);
        }
        ExprKind::Lambda { body, .. } => rewrite_expr_aliases(body, aliases),
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
        ExprKind::Block(b) | ExprKind::Unsafe(b) => rewrite_block_aliases(b, aliases),
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

    #[test]
    fn parses_systems_syntax_and_computes_exact_c_layouts() {
        let src = r#"
pub const MAGIC UInt32 = 0xDEAD_BEEF

@repr(C)
pub record PacketHdr {
    ver  UInt8
    len  UInt32
    tag  UInt16
}

@packed
pub record PackedHdr {
    ver  UInt8
    len  UInt32
    tag  UInt16
}

@align(64)
pub record CacheAligned {
    val Int64
}

@repr(C)
pub union FloatBits {
    f Float32
    u UInt32
}

pub bitfield TcpFlags : UInt16 {
    data_offset : 4
    reserved    : 3
    syn         : 1
    ack         : 1
}

@link("m")
extern "C" {
    fn sin(x Float64) Float64
    fn printf(fmt Ptr[UInt8], ...) Int32
}

@cfg(os = "nonexistent_os")
pub fn dropped_by_cfg() Int { 1 }

pub unsafe fn demo(p Ptr[UInt8]) Int {
    let arr [UInt8; 16] = [0; 16]
    defer { let _ = arr[0] }
    let x = (~0b1010) as UInt32
    let y = unsafe { sin(1.0) }
    0
}
"#;
        let prog = parse(src).expect("systems syntax should parse cleanly");
        assert!(!prog.items.iter().any(|i| matches!(i, Item::Fn(f) if f.name == "dropped_by_cfg")));
        let eng = LayoutEngine::from_program(&prog);
        let c_hdr = eng.struct_layout("PacketHdr").unwrap();
        assert_eq!(c_hdr.size, 12);
        assert_eq!(c_hdr.align, 4);
        assert_eq!(eng.offset_of("PacketHdr", "ver"), Some(0));
        assert_eq!(eng.offset_of("PacketHdr", "len"), Some(4));
        assert_eq!(eng.offset_of("PacketHdr", "tag"), Some(8));

        let p_hdr = eng.struct_layout("PackedHdr").unwrap();
        assert_eq!(p_hdr.size, 7);
        assert_eq!(p_hdr.align, 1);
        assert_eq!(eng.offset_of("PackedHdr", "len"), Some(1));
        assert_eq!(eng.offset_of("PackedHdr", "tag"), Some(5));

        let ca = eng.struct_layout("CacheAligned").unwrap();
        assert_eq!(ca.size, 64);
        assert_eq!(ca.align, 64);

        let ul = eng.union_layout("FloatBits").unwrap();
        assert_eq!(ul.size, 4);
        assert_eq!(ul.align, 4);

        let bf = eng.bitfield_layout("TcpFlags").unwrap();
        assert_eq!(bf.backing, ScalarKind::U16);
        assert_eq!(bf.total_bits, 9);
        assert_eq!(bf.fields[2].name, "syn");
        assert_eq!(bf.fields[2].bit_offset, 7);
        assert_eq!(bf.fields[2].mask, 1);
    }
}
