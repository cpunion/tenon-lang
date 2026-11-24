pub mod ast;
pub mod comptime;
pub mod derive;
pub mod layout;
pub mod lexer;
pub mod parser;
pub mod project;
pub mod span;

pub use ast::*;
pub use comptime::{ComptimeContext, eval_comptime_in_fn, expand_comptime_and_macros};
pub use derive::expand_derives;
pub use layout::*;
pub use parser::{parse, parse_expr_str, parse_with_file_id};
pub use project::{
    DepSpec, LoadMode, TenonManifest, discover_test_packages, find_enclosing_manifest,
    is_test_source_file, load_package, load_package_with_config, load_package_with_overlays,
};
pub use span::{ParseError, Span};

use std::path::Path;

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
            Item::Macro(d) => &d.attrs,
            Item::AttrDecl(d) => &d.attrs,
            Item::Import { .. }
            | Item::LockLattice(_)
            | Item::ComptimeBlock(_)
            | Item::MacroCall { .. } => return true,
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

/// Load a `.tn` source file or package directory and recursively resolve all imports
/// (`import "path.tn"`, `import pkg.subpkg [as alias]`, `import pkg.subpkg.*`,
/// `import pkg.subpkg.{A, B as C}`), then evaluate `@cfg`, expand `comptime`/macros, and `@derive(...)`.
pub fn load_program_with_imports(entry: &Path) -> Result<Program, String> {
    load_package(entry, LoadMode::Production)
}

/// Load a `.tn` source file or package directory in test mode (including `*_test.tn` and `test fn`
/// for the root package while loading imported dependencies in production mode).
pub fn load_program_for_test(entry: &Path) -> Result<Program, String> {
    load_package(entry, LoadMode::Test)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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

    #[test]
    fn parses_and_expands_comptime_macros_and_reflection() {
        let src = r#"
pub record User {
    id   Int
    name Text
}

pub macro double_val(x) {
    (x + x)
}

comptime fn gen_const(name Text, val Int) Text {
    "pub const " ++ name ++ " Int = " ++ int_to_text(val)
}

gen_const!("MAGIC_NUM", 42)

comptime {
    "pub pure fn comptime_generated() Int { " ++ int_to_text(field_count[User]()) ++ " }"
}

@get("/ping")
pub fn ping() Int { 200 }

pub fn main() Int {
    let a = comptime { 20 + 1 }
    let b = double_val!(a)
    let c = comptime if has_field[User]("name") { 1 } else { 0 }
    let r = comptime { len(routes_of()) }
    b + c + r + MAGIC_NUM + comptime_generated()
}
"#;
        let mut prog = parse(src).expect("comptime/macro syntax must parse");
        expand_comptime_and_macros(&mut prog).expect("comptime expansion must succeed");
        assert!(
            prog.items.iter().any(|i| matches!(i, Item::Const(c) if c.name == "MAGIC_NUM")),
            "typed generator macro must emit MAGIC_NUM const"
        );
        assert!(
            prog.items.iter().any(|i| matches!(i, Item::Fn(f) if f.name == "comptime_generated")),
            "top-level comptime block must emit comptime_generated fn"
        );
        assert!(
            !prog.items.iter().any(|i| matches!(i, Item::Fn(f) if f.is_comptime)),
            "comptime-only generator fns must be stripped after expansion"
        );
    }

    #[test]
    fn loads_directory_packages_and_discovers_tests() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("examples/modules_demo");
        let pkgs = discover_test_packages(&root).expect("discover_test_packages must succeed");
        // `mathlib/` has its own `tenon.toml` (nested module boundary), so root discovers `app` and `util`
        assert_eq!(pkgs.len(), 2, "expected app & util packages under root boundary; got {:?}", pkgs);
        let nested_pkgs = discover_test_packages(&root.join("mathlib"))
            .expect("nested mathlib module discovery must succeed");
        assert_eq!(nested_pkgs.len(), 1, "expected mathlib package; got {:?}", nested_pkgs);

        let prod = load_package(&root.join("app"), LoadMode::Production)
            .expect("production load of modules_demo/app must succeed");
        assert!(
            !prod.items.iter().any(|i| matches!(i, Item::Fn(f) if f.is_test)),
            "production mode must strip inline test fns"
        );
        assert!(prod.items.iter().any(|i| matches!(i, Item::Fn(f) if f.name == "main")));

        let test_prog = load_package(&root.join("mathlib"), LoadMode::Test)
            .expect("test load of modules_demo/mathlib must succeed");
        assert!(!test_prog.test_file_ids.is_empty(), "mathlib_test.tn must be marked as test file");
        assert!(
            test_prog.items.iter().any(|i| matches!(i, Item::Fn(f) if f.is_test)),
            "test mode must load test fn items"
        );
    }
}
