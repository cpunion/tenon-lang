use super::*;
use tenon_syntax::parse;

#[test]
fn llvm_aot_core_records_async_syscalls_and_interfaces_zero_leaks() {
    let src = r#"
pub type Year  = Int where 1 <= self && self <= 9999
pub type Month = Int where 1 <= self && self <= 12

pub pure fn is_leap_year(y Int) Bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

pub pure fn days_in_month(y Year, m Month) Int {
    match m {
        2 => if is_leap_year(y) { 29 } else { 28 }
        4 | 6 | 9 | 11 => 30
        _ => 31
    }
}

pub record Date {
    year  Year
    month Month
    day   Int where 1 <= self && self <= days_in_month(year, month)
}

pub fn shift_prev(inout d Date) {
    mutate d {
        d.year -= 1
        if d.month == 2 && d.day == 29 && !is_leap_year(d.year) {
            d.day = 28
        }
    }
}

pub async fn async_add(v Int) Int {
    let a = await yield_val(v + 5)
    let b = await yield_val(a + 7)
    b
}

pub record Msg { tag Text, body Text }

pub enum BoxVal {
    BEmpty
    BMsg(Msg)
    BList(List[Msg])
}

pub fn build_msg(t Text, b Text) Msg {
    Msg { tag = "[" ++ t ++ "]", body = b }
}

pub interface Greeter {
    pure fn greet(self, who Text) Text
}

pub interface Accumulator {
    fn add(inout self, delta Int) Int
    pure fn current(self) Int
}

pub record EnGreeter { prefix Text }
pub record ZhGreeter { prefix Text }
impl Greeter for EnGreeter { pure fn greet(self, who Text) Text { self.prefix ++ " " ++ who } }
impl Greeter for ZhGreeter { pure fn greet(self, who Text) Text { self.prefix ++ ", " ++ who } }

pub record Counter { val Int }
impl Accumulator for Counter {
    fn add(inout self, delta Int) Int { self.val += delta; self.val }
    pure fn current(self) Int { self.val }
}

pub pure fn run_static[T Greeter](g T, name Text) Text { g.greet(name) }
pub pure fn run_dyn(g dyn Greeter, name Text) Text { g.greet(name) }

pub fn main() Int {
    // 1. Records, mutate, Date.parse, stackless async
    var d Date = Date { year = 2024, month = 2, day = 29 }
    shift_prev(d)
    assert d.year == 2023 && d.day == 28
    let ok_day = match Date.parse(2024, 2, 29) { Ok(dt) => dt.day Err(c) => -c }
    let bad_m = match Date.parse(2024, 13, 1) { Ok(dt) => dt.day Err(c) => -c }
    assert ok_day == 29 && bad_m == -2
    let ar = block_on(async_add(10))
    assert ar == 22

    // 2. Lists, enums, text slicing, core syscalls
    var xs List[Msg] = []
    xs = push(xs, build_msg("OK", "hello tenon"))
    let bv = BList(xs)
    let out = match bv {
        BList(items) => { let m = items[0]; m.tag ++ " " ++ str_slice(m.body, 0, 5) }
        BMsg(m) => m.tag
        BEmpty => ""
    }
    assert out == "[OK] hello"

    // 3. Static monomorph & dyn vtable interfaces + COW
    let en = EnGreeter { prefix = "Hello" }
    let zh = ZhGreeter { prefix = "NiHao" }
    assert run_static(en, "Tenon") == "Hello Tenon"
    assert run_static(zh, "Tenon") == "NiHao, Tenon"
    var list List[dyn Greeter] = []
    list = push(list, en)
    list = push(list, zh)
    let b = buf_new()
    let _ = buf_push_str(b, run_dyn(list[0], "World"))
    let _ = buf_push_byte(b, 124)
    let _ = buf_push_str(b, list[1].greet("World"))
    let joined = buf_finish(b)
    assert joined == "Hello World|NiHao, World"
    var c dyn Accumulator = Counter { val = 10 }
    let c2 dyn Accumulator = c
    assert c.add(5) == 15 && c.current() == 15 && c2.current() == 10

    let _ = sys_write_stdout(out ++ "|" ++ joined ++ "\n")
    0
}
"#;
    let prog = parse(src).unwrap();
    let bin = std::env::temp_dir().join(format!("tenon_llvm_core_test_{}", std::process::id()));
    let opts = BuildOptions {
        opt_level: "O3".into(),
        ..Default::default()
    };
    let (art, run) = compile_and_run_for_test(&prog, &bin, &opts).unwrap();
    assert!(art.binary_bytes > 4096);
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        run.status.success(),
        "native LLVM core suite binary failed: stdout={} stderr={}",
        stdout,
        String::from_utf8_lossy(&run.stderr)
    );
    assert_eq!(stdout.trim(), "[OK] hello|Hello World|NiHao, World");
}

#[test]
fn llvm_aot_p0_expressiveness_zero_leaks() {
    let src = r#"
pub type NonNegative = Int where self >= 0

pub interface Display {
pure fn display(self) Text
}
pub interface Eq {
pure fn eq(self, other Self) Bool
}
pub interface Hash {
pure fn hash(self) NonNegative
}

impl Display for Text {
pub pure fn display(self) Text { self }
}
impl Display for Float {
pub pure fn display(self) Text { float_to_text(self) }
}
impl Eq for Int {
pub pure fn eq(self, other Int) Bool { self == other }
}
impl Hash for Int {
pub pure fn hash(self) NonNegative { self & 9223372036854775807 }
}
impl Eq for Text {
pub pure fn eq(self, other Text) Bool { self == other }
}
impl Hash for Text {
pub pure fn hash(self) NonNegative { len(self) }
}

@derive(Eq, Hash)
pub record TagPt {
x   Int
tag Text
}

pub enum Tree[T] {
Empty
Leaf(T)
Node(Tree[T], T, Tree[T])
}

pub pure fn sum_tree(t Tree[Int]) Int {
match t {
    Empty => 0
    Leaf(v) => v
    Node(Leaf(a), v, Leaf(b)) => a + v + b
    Node(l, v, r) => sum_tree(l) + v + sum_tree(r)
}
}

pub pure fn join_tree(t Tree[Text]) Text {
match t {
    Empty => ""
    Leaf(s) => s
    Node(l, s, r) => join_tree(l) ++ s ++ join_tree(r)
}
}

pub pure fn make_greeter(prefix Text, bump Int) fn(Text) Text {
|who| prefix ++ who ++ if bump > 0 { "!" } else { "." }
}

pub pure fn try_chain_res(ok_flag Bool) Result[Int, Text] {
let alive = "keep_me_rc"
let first Result[Int, Text] = if ok_flag { Ok(10) } else { Err("fail:" ++ alive) }
let a = first?
let second Result[Int, Text] = Ok(a + len(alive))
let b = second?
Ok(b)
}

pub pure fn try_chain_opt(present Bool) Option[Int] {
let alive = "opt_rc"
let o Option[Int] = if present { Some(20) } else { None }
let v = o?
Some(v + len(alive))
}

pub fn main() Int {
var f = 2.0
f += 1.5
f *= 2.0
f /= 3.5
assert f == 2.0
assert float_to_int(3.9) == 3
assert int_to_float(5) == 5.0
assert parse_float("1.25") == 1.25
assert !float_is_nan(f)

let g = make_greeter("hi ", 1)
assert g("tenon") == "hi tenon!"

var acc = 0
for i in 0..8 {
    if i == 2 { continue }
    if i == 6 { break }
    acc += i
}
assert acc == 13

let r = 1..5
var rsum = 0
for k in r { rsum += k }
assert rsum == 10

var lsum = 0
let ws = ["a", "bb", "ccc"]
for w in ws { lsum += len(w) }
assert lsum == 6

var bits = 4
bits <<= 2
bits >>= 1
bits |= 3
bits &= 10
bits ^= 6
bits *= 3
bits /= 4
assert bits == 9

assert match try_chain_res(true) { Ok(v) => v Err(_) => -1 } == 20
assert match try_chain_res(false) { Ok(_) => "" Err(e) => e } == "fail:keep_me_rc"
assert match try_chain_opt(true) { Some(v) => v None => -1 } == 26
assert match try_chain_opt(false) { Some(_) => 1 None => 0 } == 0

let tup = (7, "abc", 1.5)
let (tn, ts, tf) = tup
assert tn == 7 && ts == "abc" && tf == 1.5 && tup.0 == 7
let nested Option[Result[(Int, Text), Text]] = Some(Ok((4, "xy")))
let nv = match nested {
    Some(Ok((n, s))) => n + len(s)
    Some(Err(_)) => -1
    None => -2
}
assert nv == 6

let it Tree[Int] = Node(Node(Leaf(1), 2, Leaf(3)), 4, Leaf(5))
assert sum_tree(it) == 15
let tt Tree[Text] = Node(Leaf("te"), "n", Leaf("on"))
assert join_tree(tt) == "tenon"

let fs = f"{ts}={tf}"
assert fs == "abc=1.5"

let p1 = TagPt { x = 1, tag = "k" }
let p2 = TagPt { x = 1, tag = "k" }
let p3 = TagPt { x = 2, tag = "k" }
assert p1 == p2 && p1 != p3
42
}
"#;
    let prog = parse(src).unwrap();
    let bin = std::env::temp_dir().join(format!("tenon_llvm_p0_test_{}", std::process::id()));
    let opts = BuildOptions {
        opt_level: "O1".into(),
        ..Default::default()
    };
    let (art, run) = compile_and_run_for_test(&prog, &bin, &opts).unwrap();
    assert!(art.binary_bytes > 4096);
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        run.status.success(),
        "native LLVM P0 expressiveness binary failed: stdout={} stderr={}",
        stdout,
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(stdout.contains("main() => 42"), "unexpected stdout: {}", stdout);
    assert!(stdout.contains("live_objects=0"), "leaked objects in LLVM AOT: {}", stdout);
}

#[test]
fn llvm_aot_systems_suite_simd_threads_tcp_zero_leaks() {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let entry = repo_root.join("examples/systems_test.tn");
    let prog = load_program_with_imports(&entry).unwrap();

    let pid = std::process::id();
    let bin = std::env::temp_dir().join(format!("tenon_llvm_systems_suite_{}", pid));
    let ir_path = std::env::temp_dir().join(format!("tenon_llvm_systems_suite_{}.ll", pid));
    let opts = BuildOptions {
        opt_level: "O1".into(),
        emit_ll_path: Some(ir_path.clone()),
        ..Default::default()
    };
    let (art, run) = compile_and_run_for_test(&prog, &bin, &opts).unwrap();
    assert!(art.binary_bytes > 4096);

    let ir = fs::read_to_string(&ir_path).unwrap();
    let _ = fs::remove_file(&ir_path);
    assert!(
        ir.contains("<4 x float>")
            && ir.contains("@llvm.fma.v4f32")
            && ir.contains("@llvm.vector.reduce.fadd.v4f32"),
        "LLVM AOT must emit native SIMD vector IR for Float32x4"
    );
    assert!(ir.contains("<4 x double>"), "LLVM AOT must emit native SIMD vector IR for Float64x4");
    assert!(ir.contains("<16 x i8>"), "LLVM AOT must emit native SIMD vector IR for UInt8x16");

    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        run.status.success(),
        "native LLVM systems suite binary failed: stdout={} stderr={}",
        stdout,
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(stdout.contains("main() => 1110"), "unexpected stdout: {}", stdout);
    assert!(stdout.contains("live_objects=0"), "leaked objects in LLVM AOT: {}", stdout);
}

#[test]
fn llvm_aot_bidirectional_c_ffi_and_header_gen() {
    let pid = std::process::id();
    let tmp = std::env::temp_dir();

    // Direction 1: Tenon calls real C code (`--c-src`) passing `@packed` and `@repr(C)` struct pointers!
    let c_helper = tmp.join(format!("tenon_ffi_helper_{}.c", pid));
    fs::write(
        &c_helper,
        r#"
#include <stddef.h>
#include <stdint.h>

typedef struct __attribute__((packed)) PackedHdr {
uint8_t ver;
uint16_t port;
uint32_t seq;
} PackedHdr;

typedef struct CPoint {
uint8_t tag;
int32_t x;
float y;
} CPoint;

int64_t c_mutate_and_verify(const PackedHdr* h, CPoint* p) {
if (sizeof(PackedHdr) != 7 || offsetof(PackedHdr, seq) != 3) return -1;
if (sizeof(CPoint) != 12 || offsetof(CPoint, x) != 4 || offsetof(CPoint, y) != 8) return -2;
p->x += (int32_t)h->seq;
return (int64_t)h->ver * 10000 + (int64_t)h->port + (int64_t)p->tag;
}
"#,
    )
    .unwrap();

    let tenon_caller = parse(
        r#"
@packed
pub record PackedHdr {
ver  UInt8
port UInt16
seq  UInt32
}

@repr(C)
pub record CPoint {
tag UInt8
x   Int32
y   Float32
}

extern "C" {
fn c_mutate_and_verify(h Ptr[PackedHdr], p MutPtr[CPoint]) Int64
}

pub fn main() Int {
let hdr = PackedHdr { ver = 4, port = 8080, seq = 100 }
var pt = CPoint { tag = 7, x = 23, y = 1.5 }
let res = unsafe {
    c_mutate_and_verify(ptr_of(hdr), mut_ptr_of(pt))
}
assert res == 48087
assert pt.x == 123
(res as Int) + (pt.x as Int)
}
"#,
    )
    .unwrap();

    let bin1 = tmp.join(format!("tenon_ffi_caller_{}", pid));
    let opts1 = BuildOptions {
        opt_level: "O1".into(),
        extra_c_sources: vec![c_helper.clone()],
        ..Default::default()
    };
    let (_art1, run1) = compile_and_run_for_test(&tenon_caller, &bin1, &opts1).unwrap();
    let _ = fs::remove_file(&c_helper);
    let out1 = String::from_utf8_lossy(&run1.stdout);
    assert!(
        run1.status.success(),
        "Tenon -> C FFI binary failed: stdout={} stderr={}",
        out1,
        String::from_utf8_lossy(&run1.stderr)
    );
    assert!(out1.contains("main() => 48210"), "unexpected Tenon->C output: {}", out1);
    assert!(out1.contains("live_objects=0"), "Tenon->C leaked objects: {}", out1);

    // Direction 2: C calls Tenon static library (`--lib static` + `--emit-header`) via `@export`!
    let tenon_lib = parse(
        r#"
@repr(C)
pub record CPoint {
tag UInt8
x   Int32
y   Float32
}

@export("tn_compute_point")
pub fn compute_point(p Ptr[CPoint], scale Int32) Int32 {
unsafe {
    let pt = ptr_read(p)
    pt.x * scale + (pt.tag as Int32)
}
}
"#,
    )
    .unwrap();

    let lib_path = tmp.join(format!("libtenon_export_{}.a", pid));
    let hdr_path = tmp.join(format!("tenon_export_{}.h", pid));
    let opts2 = BuildOptions {
        opt_level: "O2".into(),
        output_kind: OutputKind::StaticLib,
        emit_header_path: Some(hdr_path.clone()),
        ..Default::default()
    };
    compile_with_options(&tenon_lib, &lib_path, &opts2).unwrap();

    let hdr_text = fs::read_to_string(&hdr_path).unwrap();
    assert!(
        hdr_text.contains("typedef struct CPoint")
            && hdr_text.contains("extern int32_t tn_compute_point(const CPoint* p, int32_t scale);"),
        "unexpected generated C header:\n{}",
        hdr_text
    );

    let c_main = tmp.join(format!("tenon_c_driver_{}.c", pid));
    let c_bin = tmp.join(format!("tenon_c_driver_{}", pid));
    fs::write(
        &c_main,
        format!(
            r#"
#include <stdio.h>
#include "{}"

int main(void) {{
CPoint pt = {{ 5, 14, 2.5f }};
int32_t r = tn_compute_point(&pt, 3);
printf("c_to_tenon=%d\n", r);
return r == 47 ? 0 : 1;
}}
"#,
            hdr_path.display()
        ),
    )
    .unwrap();

    let cc = Command::new("clang")
        .args([
            "-O2",
            c_main.to_str().unwrap(),
            lib_path.to_str().unwrap(),
            "-o",
            c_bin.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        cc.status.success(),
        "clang linking libtenon_export.a failed: {}",
        String::from_utf8_lossy(&cc.stderr)
    );
    let run2 = run_c_driver_against_tenon_lib_for_test(&tenon_lib, &c_main, &c_bin, &opts2).unwrap();
    let _ = fs::remove_file(&c_main);
    let _ = fs::remove_file(&c_bin);
    let _ = fs::remove_file(&lib_path);
    let _ = fs::remove_file(&hdr_path);
    let out2 = String::from_utf8_lossy(&run2.stdout);
    assert!(
        run2.status.success() && out2.contains("c_to_tenon=47"),
        "C -> Tenon static library execution failed: status={:?} stdout={} stderr={}",
        run2.status,
        out2,
        String::from_utf8_lossy(&run2.stderr)
    );
}

#[test]
fn llvm_aot_dwarf_debug_info_emits_valid_metadata() {
    let mut prog = parse(
        r#"
pub fn add_pair(a Int, b Int) Int {
let s = a + b
s * 2
}

pub fn main() Int {
add_pair(10, 11)
}
"#,
    )
    .unwrap();
    prog.source_files.push("debug_demo.tn".into());

    let pid = std::process::id();
    let obj = std::env::temp_dir().join(format!("tenon_llvm_dbg_{}.o", pid));
    let ir_path = std::env::temp_dir().join(format!("tenon_llvm_dbg_{}.ll", pid));
    let opts = BuildOptions {
        opt_level: "O0".into(),
        debug_info: true,
        emit_ll_path: Some(ir_path.clone()),
        ..Default::default()
    };
    compile_with_options(&prog, &obj, &opts).unwrap();
    let ir = fs::read_to_string(&ir_path).unwrap();
    let _ = fs::remove_file(&ir_path);
    assert!(ir.contains("!DICompileUnit"), "missing !DICompileUnit in DWARF IR");
    assert!(ir.contains("!DIFile"), "missing !DIFile in DWARF IR");
    assert!(ir.contains("!DISubprogram"), "missing !DISubprogram in DWARF IR");
    assert!(ir.contains("!DILocation"), "missing !DILocation in DWARF IR");
    assert!(fs::metadata(&obj).unwrap().len() > 256, "expected valid DWARF .o object");
    let _ = fs::remove_file(&obj);
}

#[test]
fn llvm_aot_web_app_and_modules_demo_zero_leaks() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let pid = std::process::id();

    let o1 = BuildOptions {
        opt_level: "O1".into(),
        ..Default::default()
    };

    // 1. Multi-package modules_demo/app (Go/glob/selective imports + comptime + typed macros)
    let mod_prog = tenon_syntax::load_package(
        &repo_root.join("examples/modules_demo/app"),
        tenon_syntax::LoadMode::Production,
    )
    .expect("modules_demo/app must load");
    let mod_bin = std::env::temp_dir().join(format!("tenon_llvm_mod_{}", pid));
    let (_a1, run1) = compile_and_run_for_test(&mod_prog, &mod_bin, &o1).expect("modules_demo LLVM AOT build");
    let out1 = String::from_utf8_lossy(&run1.stdout);
    assert!(
        run1.status.success() && out1.contains("main() => 107"),
        "modules_demo AOT failed: stdout={} stderr={}",
        out1,
        String::from_utf8_lossy(&run1.stderr)
    );

    // 2. Metaprogramming Web Framework (examples/web_app: kqueue + HTTP/1.1 Keep-Alive + @get/@post + OpenAPI + JSON refinement)
    let web_prog = tenon_syntax::load_package(
        &repo_root.join("examples/web_app"),
        tenon_syntax::LoadMode::Production,
    )
    .expect("examples/web_app must load");
    let web_bin = std::env::temp_dir().join(format!("tenon_llvm_web_{}", pid));
    let (_a2, run2) = compile_and_run_for_test(&web_prog, &web_bin, &o1).expect("examples/web_app LLVM AOT build");
    let out2 = String::from_utf8_lossy(&run2.stdout);
    assert!(
        run2.status.success()
            && out2.contains("main() => 200")
            && out2.contains("live_objects=0"),
        "web_app AOT failed: stdout={} stderr={}",
        out2,
        String::from_utf8_lossy(&run2.stderr)
    );
}

#[test]
fn llvm_aot_advanced_verification_unchecked_indexing_ghost_erasure_and_zero_leaks() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let adv_path = repo_root.join("examples/refinements/advanced_verification.tn");
    let prog = tenon_syntax::load_program_with_imports(&adv_path)
        .expect("advanced_verification.tn must load");
    let rep = tenon_verify::check_program(&prog);
    assert!(
        rep.ok,
        "advanced_verification.tn must pass verification: {:?}",
        rep.diagnostics
    );
    assert!(
        rep.proved_safe_indices.len() >= 2,
        "expected >= 2 proved in-bounds indices, got {:?}",
        rep.proved_safe_indices
    );

    let pid = std::process::id();
    let bin = std::env::temp_dir().join(format!("tenon_llvm_adv_ver_{}", pid));
    let ir_path = std::env::temp_dir().join(format!("tenon_llvm_adv_ver_{}.ll", pid));
    let opts = BuildOptions {
        opt_level: "O1".into(),
        emit_ll_path: Some(ir_path.clone()),
        proved_safe_indices: rep.proved_safe_indices.clone(),
        ..Default::default()
    };
    let (_art, run) = compile_and_run_for_test(&prog, &bin, &opts).expect("advanced_verification LLVM AOT build");
    let ir = fs::read_to_string(&ir_path).unwrap();
    let _ = fs::remove_file(&ir_path);
    assert!(
        ir.contains("call i64 @__tn_list_get_unchecked("),
        "expected @__tn_list_get_unchecked call in LLVM IR"
    );
    assert!(
        ir.contains("call void @__tn_list_set_prim_unchecked("),
        "expected @__tn_list_set_prim_unchecked call in LLVM IR"
    );
    assert!(
        !ir.contains("nonneg_product"),
        "ghost lemma `nonneg_product` must be completely erased from LLVM IR"
    );

    let out = String::from_utf8_lossy(&run.stdout);
    assert!(
        run.status.success()
            && out.contains("main() => 102")
            && out.contains("live_objects=0"),
        "advanced_verification AOT failed: stdout={} stderr={}",
        out,
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn llvm_aot_checked_overflow_flat_lists_mir_moves_and_xcompile_zero_leaks() {
    let prog = parse(r#"
pub type Small = Int where self >= 0 && self <= 1000

pub pure fn safe_add(a Small, b Small) Int {
    a + b
}

pub pure fn unchecked_add(a Int, b Int) Int {
    a + b
}

pub record Boxed {
    tag Int
    label Text
}

pub fn wrap_last_use(n Int) Boxed {
    let b = Boxed { tag = n, label = "tenon-mir-move" }
    b
}

pub fn flat_int32_sum(n Int where self >= 1 && self <= 16) Int {
    var xs List[Int32] = list_filled(n, -3 as Int32)
    let copy = xs
    xs[0] = 10 as Int32
    xs = push(xs, 5 as Int32)
    var s = 0
    var i = 0
    while i < len(xs) {
        s = s + (xs[i] as Int)
        i += 1
    }
    s + (copy[0] as Int)
}

pub fn flat_float32_dot() Int {
    var a List[Float32] = [1.5 as Float32, 2.5 as Float32, 3.0 as Float32]
    a[1] = 4.0 as Float32
    a = push(a, 2.0 as Float32)
    let sum = (a[0] as Float) + (a[1] as Float) + (a[2] as Float) + (a[3] as Float)
    float_to_int(sum * 2.0)
}

pub fn main() Int {
    let s = safe_add(10, 20)
    let u = unchecked_add(10, 2)
    let b = wrap_last_use(100)
    let s32 = flat_int32_sum(4)
    let f32s = flat_float32_dot()
    // 42 + 100 + 3 + 21 = 166
    s + u + b.tag + s32 + f32s
}
"#)
    .unwrap();
    let rep = tenon_verify::check_program(&prog);
    assert!(rep.ok, "verify failed: {:?}", rep.diagnostics);
    let ssa_mir = tenon_mir::lower_program_to_ssa_mir(&prog, &rep.mir);
    let mir_txt = tenon_mir::format_mir(&ssa_mir);
    assert!(mir_txt.contains("move.local $b"), "expected move.local $b in MIR:\n{}", mir_txt);

    let pid = std::process::id();
    let bin = std::env::temp_dir().join(format!("tenon_llvm_ov_mir_{}", pid));
    let ir_path = std::env::temp_dir().join(format!("tenon_llvm_ov_mir_{}.ll", pid));
    let mut opts = BuildOptions {
        opt_level: "O3".into(),
        emit_ll_path: Some(ir_path.clone()),
        checked_overflow: true,
        ..Default::default()
    };
    opts.apply_verified_mir(&rep.mir);
    opts.apply_ssa_mir(&ssa_mir);
    let (_art, run) = compile_and_run_for_test(&prog, &bin, &opts).expect("checked overflow + MIR AOT build");
    let ir = fs::read_to_string(&ir_path).unwrap();
    let _ = fs::remove_file(&ir_path);
    assert!(ir.contains("add nsw i64"), "expected `add nsw i64` for proved overflow-free `safe_add`");
    assert!(
        ir.contains("@llvm.sadd.with.overflow.i64"),
        "expected `@llvm.sadd.with.overflow.i64` for unproved `unchecked_add`"
    );
    assert!(
        ir.contains("@tenon_flat_list_filled(i64") && ir.contains(", i64 4, ptr"),
        "expected 4-byte flat list allocation in IR"
    );
    assert!(
        ir.contains("load i32, ptr") && ir.contains("store i32"),
        "expected native 32-bit packed element load/store in IR"
    );
    let wrap_start = ir.find("@tenon_fn_wrap_last_use(").expect("wrap_last_use IR");
    let wrap_end = ir[wrap_start..].find("\n}\n").unwrap() + wrap_start;
    let wrap_ir = &ir[wrap_start..wrap_end];
    assert!(
        !wrap_ir.contains("@__tn_rc_inc"),
        "expected Last-Use Move to eliminate @__tn_rc_inc in wrap_last_use:\n{}",
        wrap_ir
    );

    let out = String::from_utf8_lossy(&run.stdout);
    assert!(
        run.status.success() && out.contains("main() => 166") && out.contains("live_objects=0"),
        "checked overflow + MIR run failed: stdout={} stderr={}",
        out,
        String::from_utf8_lossy(&run.stderr)
    );

    // Cross-compile object emission test (`x86_64-unknown-linux-gnu`)
    let obj_path = std::env::temp_dir().join(format!("tenon_xcompile_{}.o", pid));
    let mut xopts = BuildOptions {
        opt_level: "O2".into(),
        target_triple: Some("x86_64-unknown-linux-gnu".into()),
        ..Default::default()
    };
    xopts.apply_verified_mir(&rep.mir);
    compile_with_options(&prog, &obj_path, &xopts).expect("cross-compile x86_64-linux .o");
    assert!(fs::metadata(&obj_path).unwrap().len() > 100);
    let _ = fs::remove_file(&obj_path);
}

#[test]
fn llvm_aot_sqlite3_check_constraints_and_openssl_tls_zero_leaks() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let pid = std::process::id();

    let (has_openssl, link_dirs, cert_path, key_path, gen_ok) =
        if let Ok(pc) = Command::new("pkg-config").args(["--libs-only-L", "openssl"]).output() {
            if pc.status.success() {
                let lflags = String::from_utf8_lossy(&pc.stdout);
                let dirs: Vec<String> = lflags
                    .split_whitespace()
                    .filter_map(|t| t.strip_prefix("-L").map(|s| s.to_string()))
                    .collect();
                let cp = std::env::temp_dir().join(format!("tenon_tls_cert_{}.pem", pid));
                let kp = std::env::temp_dir().join(format!("tenon_tls_key_{}.pem", pid));
                let ok = Command::new("openssl")
                    .args([
                        "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                        "-keyout", &kp.to_string_lossy(),
                        "-out", &cp.to_string_lossy(),
                        "-days", "1", "-subj", "/CN=localhost",
                    ])
                    .output()
                    .is_ok_and(|o| o.status.success());
                (true, dirs, cp, kp, ok)
            } else {
                (false, vec![], PathBuf::new(), PathBuf::new(), false)
            }
        } else {
            (false, vec![], PathBuf::new(), PathBuf::new(), false)
        };

    // Combine real SQLite3 CHECK constraint verification + OpenSSL TLS 1.3 handshake into ONE native binary!
    let tls_imports = if has_openssl {
        "import \"std/net/tls.tn\"\nimport \"std/sync/sync.tn\"\n"
    } else {
        ""
    };
    let tls_check = if has_openssl && gen_ok {
        format!(
            r#"
    let ver = tls_openssl_version()
    let sctx = match tls_server_context() {{ Ok(c) => c Err(_) => {{ return -11 }} }}
    match sctx.load_cert_key("{}", "{}") {{
        Ok(_) => {{}}
        Err(_) => {{ sctx.close(); return -12 }}
    }}
    let cctx = match tls_client_context() {{ Ok(c) => c Err(_) => {{ sctx.close(); return -13 }} }}
    let lfd = net_tcp_listen("127.0.0.1:0")
    if lfd < 0 {{ cctx.close(); sctx.close(); return -14 }}
    let port = net_tcp_local_port(lfd)
    let srv = thread_spawn(|| {{
        let cfd = net_tcp_accept(lfd)
        if cfd < 0 {{ return -1 }}
        let served = match sctx.accept_fd(cfd) {{
            Ok(st) => {{
                let n = https_serve_stream(st, |req| http_ok("tls:" ++ req.path))
                st.close()
                n
            }}
            Err(_) => -1
        }}
        let _ = net_close(cfd)
        served
    }})
    let tls_ok = match https_get_with_context(cctx, f"127.0.0.1:{{port}}", "localhost", "/secure") {{
        Ok(res) => if res.status == 200 && res.body == "tls:/secure" && ver > 0 {{ 1 }} else {{ -15 }}
        Err(_) => -16
    }}
    let srv_n = thread_join(srv)
    let _ = net_close(lfd)
    cctx.close()
    sctx.close()
    if tls_ok != 1 || srv_n != 1 {{ return tls_ok }}
"#,
            cert_path.to_string_lossy().replace('\\', "\\\\"),
            key_path.to_string_lossy().replace('\\', "\\\\")
        )
    } else {
        String::new()
    };

    let combined_src = format!(
        r#"
import "std/db/sqlite.tn"
{tls_imports}

pub fn main() Int {{
    let db = match sqlite_open(":memory:") {{
        Ok(d) => d
        Err(_) => {{ return -1 }}
    }}
    let ddl = sql_create_table_ddl(
        "orders",
        ["id INTEGER PRIMARY KEY", "sym TEXT NOT NULL", "qty INTEGER NOT NULL", "px INTEGER NOT NULL"],
        ["qty > 0", "px >= 1"]
    )
    match db.exec(ddl) {{
        Ok(_) => {{}}
        Err(_) => {{ db.close(); return -2 }}
    }}
    let _ = db.exec("INSERT INTO orders(id, sym, qty, px) VALUES (1, 'BTC', 10, 50000);")
    let _ = db.exec("INSERT INTO orders(id, sym, qty, px) VALUES (2, 'ETH', 25, 3000);")
    let rejected = match db.exec("INSERT INTO orders(id, sym, qty, px) VALUES (3, 'BAD', -5, 100);") {{
        Ok(_) => 0
        Err(_) => 1
    }}
    let sum_qty = match db.query_int("SELECT SUM(qty) FROM orders;") {{
        Ok(v) => v
        Err(_) => -3
    }}
    let first_sym = match db.query_text("SELECT sym FROM orders WHERE id = 1;") {{
        Ok(s) => s
        Err(_) => ""
    }}
    db.close()
    if rejected != 1 || first_sym != "BTC" {{ return -99 }}
{tls_check}
    sum_qty
}}
"#,
        tls_imports = tls_imports,
        tls_check = tls_check
    );

    let tmp_tn = repo_root.join(format!("target/tenon_sqlite_tls_test_{}.tn", pid));
    let _ = fs::create_dir_all(tmp_tn.parent().unwrap());
    fs::write(&tmp_tn, combined_src).unwrap();
    let prog = tenon_syntax::load_program_with_imports(&tmp_tn).unwrap();
    let _ = fs::remove_file(&tmp_tn);
    let rep = tenon_verify::check_program(&prog);
    assert!(rep.ok, "sqlite+tls test verify failed: {:?}", rep.diagnostics);

    let bin = std::env::temp_dir().join(format!("tenon_llvm_sqlite_tls_{}", pid));
    let opts = BuildOptions {
        opt_level: "O1".into(),
        link_dirs,
        ..Default::default()
    };
    let (_art, run) = compile_and_run_for_test(&prog, &bin, &opts).expect("SQLite3+OpenSSL LLVM AOT link failed");
    if has_openssl && gen_ok {
        let _ = fs::remove_file(&cert_path);
        let _ = fs::remove_file(&key_path);
    }
    let out = String::from_utf8_lossy(&run.stdout);
    assert!(
        run.status.success() && out.contains("main() => 35") && out.contains("live_objects=0"),
        "SQLite3+OpenSSL AOT run failed: stdout={} stderr={}",
        out,
        String::from_utf8_lossy(&run.stderr)
    );
}

#[test]
fn llvm_aot_dogfood_matching_engine_and_tenon_docgen_zero_leaks() {
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let pid = std::process::id();

    // 1. Dogfood #1: Verified Limit-Order-Book Matching Engine + Const Range[LO,HI] + forall Array + @fip + SQLite3
    let me_prog = tenon_syntax::load_program_with_imports(
        &repo_root.join("examples/dogfood/matching_engine.tn"),
    )
    .expect("matching_engine.tn must load");
    let me_rep = tenon_verify::check_program(&me_prog);
    assert!(me_rep.ok, "matching_engine verify failed: {:?}", me_rep.diagnostics);
    assert!(me_rep.mir.fip_verified_fns.contains("adjust_order_qty"));
    assert!(me_rep.mir.proved_safe_indices.len() >= 2);
    let me_bin = std::env::temp_dir().join(format!("tenon_dogfood_me_{}", pid));
    let mut opts = BuildOptions {
        opt_level: "O1".into(),
        ..Default::default()
    };
    opts.apply_verified_mir(&me_rep.mir);
    let (_a1, run1) = compile_and_run_for_test(&me_prog, &me_bin, &opts).expect("matching_engine LLVM AOT build");
    let out1 = String::from_utf8_lossy(&run1.stdout);
    assert!(
        run1.status.success() && out1.contains("main() => 5730") && out1.contains("live_objects=0"),
        "matching_engine AOT run failed: stdout={} stderr={}",
        out1,
        String::from_utf8_lossy(&run1.stderr)
    );

    // 2. Dogfood #2: Self-hosting Tenon API Documentation Generator (`tenon_docgen.tn`)
    let doc_prog = tenon_syntax::load_program_with_imports(
        &repo_root.join("examples/dogfood/tenon_docgen.tn"),
    )
    .expect("tenon_docgen.tn must load");
    let doc_rep = tenon_verify::check_program(&doc_prog);
    assert!(doc_rep.ok, "tenon_docgen verify failed: {:?}", doc_rep.diagnostics);
    let doc_bin = std::env::temp_dir().join(format!("tenon_dogfood_doc_{}", pid));
    let mut doc_opts = BuildOptions {
        opt_level: "O1".into(),
        ..Default::default()
    };
    doc_opts.apply_verified_mir(&doc_rep.mir);
    let (_a2, run2) = compile_and_run_for_test(&doc_prog, &doc_bin, &doc_opts).expect("tenon_docgen LLVM AOT build");
    let out2 = String::from_utf8_lossy(&run2.stdout);
    assert!(
        run2.status.success() && out2.contains("main() => 4") && out2.contains("live_objects=0"),
        "tenon_docgen AOT run failed: stdout={} stderr={}",
        out2,
        String::from_utf8_lossy(&run2.stderr)
    );
}

#[test]
fn cross_gpu_link_bundle_uses_strictly_relative_paths_and_copies_local_headers() {
    let handle = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .canonicalize()
                .unwrap();
            let srv_src = repo_root.join("examples/showcase/tenon_llm_server_cuda.tn");
            let prog = tenon_syntax::load_program_with_imports(&srv_src).expect("load cuda server");
            let tmp_dir = std::env::temp_dir().join(format!("tenon_bundle_test_{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&tmp_dir);
            std::fs::create_dir_all(&tmp_dir).unwrap();

            for (target_name, ext) in [("windows-x86_64-cuda", "bat"), ("linux-x86_64-cuda", "sh")] {
                let out_path = tmp_dir.join(format!("srv_{}", target_name));
                let target = crate::targets::resolve_target(target_name).expect("resolve target");
                let opts = BuildOptions {
                    opt_level: "O2".into(),
                    resolved_target: Some(target),
                    ..Default::default()
                };
                compile_with_options(&prog, &out_path, &opts).expect("cross-compile bundle");
                let script = std::fs::read_to_string(tmp_dir.join(format!("srv_{}_link.{}", target_name, ext)))
                    .expect("link script exists");
                assert!(
                    !script.contains(repo_root.to_str().unwrap()),
                    "link script {} must NOT contain host absolute path:\n{}",
                    target_name,
                    script
                );
                assert!(
                    !script.contains("/Users/") && !script.contains("/home/"),
                    "link script {} leaked host path:\n{}",
                    target_name,
                    script
                );
                let src_dir = tmp_dir.join(format!("srv_{}_src", target_name));
                assert!(
                    src_dir.join("cuda/moe_stream_cuda_shim.c").exists(),
                    "bundled CUDA shim missing in {:?}",
                    src_dir
                );
                assert!(
                    src_dir.join("common/tenon_io_hal.h").exists(),
                    "bundled local header tenon_io_hal.h missing in {:?}",
                    src_dir
                );
            }
            let _ = std::fs::remove_dir_all(&tmp_dir);
        })
        .expect("spawn test thread");
    handle.join().expect("cross_gpu_link_bundle test panicked");
}


