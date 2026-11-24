pub mod checker;
pub mod db;
pub mod evolve;
pub mod l0;
pub mod modelcheck;
pub mod smt;
pub mod sync_solver;

pub use checker::{CheckReport, Diagnostic, ProofRecord, ProofTier, check_program};
pub use db::{IncrementalDb, QueryStats};
pub use evolve::{CompatVerdict, EvolutionReport, check_evolution};
pub use modelcheck::{
    ModelCheckSuiteReport, TraceEvent, TraceRefinementReport, check_program_protocols,
    export_program_tla_specs, export_tla_specs, run_all_protocol_checks, verify_runtime_trace,
};
pub use sync_solver::{
    DistributedAppManifest, ResolvedStreamSpec, ResolvedSubscriberSpec, ResolvedViewSpec,
    SyncArchetypeKind,
};

#[cfg(test)]
mod tests {
    use super::*;
    use tenon_syntax::parse;

    #[test]
    fn proves_odd_plus_odd_and_date_in_l0_without_z3() {
        let src = r#"
pub type Odd  = Int64 where self % 2 != 0
pub type Even = Int64 where self % 2 == 0

pub pure fn add_odds(a Odd, b Odd) Even {
    a + b
}

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

pub fn valid_leap() Date {
    Date { year = 2024, month = 2, day = 29 }
}
"#;
        let prog = parse(src).unwrap();
        let report = check_program(&prog);
        assert!(report.ok, "unexpected diagnostics: {:?}", report.diagnostics);
        assert!(
            report.proofs.iter().any(|p| p.tier == ProofTier::L0Fast && p.obligation.contains("Even")),
            "Odd + Odd -> Even must be proved in L0"
        );
        assert!(
            report.proofs.iter().any(|p| p.tier == ProofTier::L0Fast && p.obligation.contains("Date")),
            "Date 2024-02-29 must be proved in L0"
        );
    }

    #[test]
    fn rejects_invalid_non_leap_feb_29() {
        let src = r#"
pub pure fn is_leap_year(y Int) Bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

pub pure fn days_in_month(y Int, m Int where 1 <= self && self <= 12) Int {
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

pub fn bad_date() Date {
    Date { year = 2023, month = 2, day = 29 }
}
"#;
        let prog = parse(src).unwrap();
        let report = check_program(&prog);
        assert!(!report.ok, "2023-02-29 must be rejected");
        assert!(report.diagnostics[0].notes.iter().any(|n| n.contains("29 <= 28")));
    }

    #[test]
    fn evolution_discovers_century_leap_year_counterexample_1900() {
        let v1 = parse(r#"
pub pure fn is_leap_year(y Int) Bool { y % 4 == 0 }
pub pure fn days_in_month(y Int, m Int) Int {
    match m {
        2 => if is_leap_year(y) { 29 } else { 28 }
        4 | 6 | 9 | 11 => 30
        _ => 31
    }
}
@evolve("calendar.Date")
pub record Date {
    @field(1) year  Int where 1 <= self && self <= 9999
    @field(2) month Int where 1 <= self && self <= 12
    @field(3) day   Int where 1 <= self && self <= days_in_month(year, month)
}
"#).unwrap();
        let v2 = parse(r#"
pub pure fn is_leap_year(y Int) Bool { (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0) }
pub pure fn days_in_month(y Int, m Int) Int {
    match m {
        2 => if is_leap_year(y) { 29 } else { 28 }
        4 | 6 | 9 | 11 => 30
        _ => 31
    }
}
@evolve("calendar.Date")
pub record Date {
    @field(1) year  Int where 1 <= self && self <= 9999
    @field(2) month Int where 1 <= self && self <= 12
    @field(3) day   Int where 1 <= self && self <= days_in_month(year, month)
}
"#).unwrap();
        let rep = check_evolution(&v1, &v2);
        assert!(!rep.safe_to_release);
        assert_eq!(rep.entities[0].verdict, CompatVerdict::NarrowingBreaking);
        let ce = &rep.entities[0].counterexample;
        assert_eq!(ce.get("month").map(|s| s.as_str()), Some("2"));
        assert_eq!(ce.get("day").map(|s| s.as_str()), Some("29"));
        let yr: i64 = ce.get("year").unwrap().parse().unwrap();
        assert_eq!(yr % 100, 0);
        assert_ne!(yr % 400, 0);
    }

    #[test]
    fn protocol_model_checker_passes_all_positive_and_negative_controls() {
        let rep = run_all_protocol_checks();
        assert!(rep.ok, "modelcheck failed: {:?}", rep.cases);
        assert_eq!(rep.cases.len(), 9);
    }

    #[test]
    fn z3_proves_safe_mutate_and_rejects_buggy_mutate() {
        let safe_src = r#"
pub pure fn is_leap_year(y Int) Bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}
pub pure fn days_in_month(y Int, m Int where 1 <= self && self <= 12) Int {
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
pub fn shift_safe(inout d Date)
    requires d.year > 1
{
    mutate d {
        d.year -= 1
        if d.month == 2 && d.day == 29 && !is_leap_year(d.year) {
            d.day = 28
        }
    }
}
"#;
        let rep = check_program(&parse(safe_src).unwrap());
        assert!(rep.ok, "safe mutate should verify cleanly: {:?}", rep.diagnostics);

        let buggy_src = safe_src.replace(
            "if d.month == 2 && d.day == 29 && !is_leap_year(d.year) {\n            d.day = 28\n        }",
            "",
        );
        let bad_rep = check_program(&parse(&buggy_src).unwrap());
        assert!(!bad_rep.ok, "buggy mutate without Feb-29 clamp MUST be rejected by Z3!");
        assert!(bad_rep.diagnostics.iter().any(|d| d.code == "E0421"));
    }

    #[test]
    fn incremental_db_memoizes_and_invalidates_on_change() {
        let db = IncrementalDb::new();
        let src = "pub type Odd = Int64 where self % 2 != 0\npub pure fn id_odd(x Odd) Odd { x }\n";
        assert!(db.set_file("a.tn", src));
        let r1 = db.check_file("a.tn").unwrap();
        assert!(r1.ok);
        let r2 = db.check_file("a.tn").unwrap();
        assert!(r2.ok);
        let st = db.stats();
        assert_eq!(st.check_misses, 1);
        assert_eq!(st.check_hits, 1);
        assert!(!db.set_file("a.tn", src)); // unchanged -> early cutoff
        assert!(db.set_file("a.tn", "pub type Even = Int64 where self % 2 == 0\n"));
        let _ = db.check_file("a.tn").unwrap();
        assert_eq!(db.stats().check_misses, 2);
    }

    #[test]
    fn custom_tn_protocol_ast_bfs_lock_lattice_and_trace_refinement() {
        let src = r#"
pub lock_lattice KernelLocks {
    SessionLock < StorageTxnLock < ReceiptLock
}

@acquires("SessionLock", "StorageTxnLock")
pub fn ok_locks(x Int) Int { x }

pub protocol state_machine CustomFencedLease {
    state {
        auth_epoch   Int  = 1
        w1_epoch     Int  = 1
        stale_commit Bool = false
    }
    action Takeover when auth_epoch < 3 {
        auth_epoch = auth_epoch + 1
    }
    fault DropFence when w1_epoch < auth_epoch {
        stale_commit = true
    }
    laws {
        invariant NoStale: !stale_commit
    }
}
"#;
        let prog = parse(src).unwrap();
        let rep = check_program_protocols(&prog);
        assert!(rep.ok, "AST protocol checks failed: {:?}", rep.cases);
        assert!(rep.cases.iter().any(|c| c.protocol == "CustomFencedLease" && c.case_name == "safety_proof" && c.passed));
        assert!(rep.cases.iter().any(|c| c.protocol == "CustomFencedLease" && c.case_name == "negative_fault_DropFence" && c.passed));

        // Negative control: bad lock order MUST fail!
        let bad_lock = parse(r#"
pub lock_lattice KernelLocks {
    SessionLock < StorageTxnLock
}
@acquires("StorageTxnLock", "SessionLock")
pub fn bad_fn(x Int) Int { x }
"#).unwrap();
        let bad_rep = check_program_protocols(&bad_lock);
        assert!(!bad_rep.ok, "inverted lock acquisition order must fail");

        // Runtime Trace Refinement positive + negative check!
        let good_trace = vec![
            TraceEvent::ClaimSlice { session_id: "s1".into(), worker_id: "w1".into(), branch_id: 1, exec_epoch: 1 },
            TraceEvent::ProxyEmit { session_id: "s1".into(), branch_id: 1, exec_epoch: 1, run_id: "r1".into(), seq: 1 },
            TraceEvent::CommitTurn { session_id: "s1".into(), branch_id: 1, exec_epoch: 1, run_id: "r1".into() },
            TraceEvent::SnapshotBuilt { session_id: "s1".into(), active_branch: 1, live_run_id: None, committed_run_ids: vec!["r1".into()] },
        ];
        assert!(verify_runtime_trace(&good_trace).ok);

        let bad_trace = vec![
            TraceEvent::ClaimSlice { session_id: "s1".into(), worker_id: "w1".into(), branch_id: 1, exec_epoch: 1 },
            TraceEvent::ClaimSlice { session_id: "s1".into(), worker_id: "w2".into(), branch_id: 1, exec_epoch: 2 },
            TraceEvent::ProxyEmit { session_id: "s1".into(), branch_id: 1, exec_epoch: 1, run_id: "r1".into(), seq: 2 },
        ];
        let bad_tr = verify_runtime_trace(&bad_trace);
        assert!(!bad_tr.ok);
        assert!(bad_tr.violations[0].contains("NoStaleProxyCall"));
    }

    #[test]
    fn enforces_async_effect_discipline_and_no_heap_alloc_contract() {
        let bad = parse(r#"
pub record Point { x Int, y Int }
pub async fn a(v Int) Int { await yield_val(v) }
pub fn bad_sync(v Int) Int { await a(v) }
@no_heap_alloc
pub fn bad_alloc(v Int) Point { Point { x = v, y = v } }
"#).unwrap();
        let rep = check_program(&bad);
        assert!(!rep.ok);
        assert!(rep.diagnostics.iter().any(|d| d.code == "E0601"));
        assert!(rep.diagnostics.iter().any(|d| d.code == "E0701"));
    }

    #[test]
    fn enforces_exhaustiveness_requires_purity_immutability_and_type_checks() {
        // 1. Positive: exhaustive enum/Option/Result match + diverging-if narrowing + call-site requires + pure fn!
        let good = parse(r#"
pub type NonNegative = Int where self >= 0

pub enum JsonVal {
    JNull
    JInt(Int)
    JStr(Text)
}

@no_heap_alloc
pub pure fn clamp_non_neg(n Int) NonNegative
    ensures result >= 0
{
    if n < 0 {
        return 0
    }
    n
}

pub pure fn slice_len(start NonNegative, end NonNegative) NonNegative
    requires start <= end
    ensures result >= 0
{
    end - start
}

pub pure fn describe(v JsonVal) Text {
    match v {
        JNull => "null"
        JInt(_) => "int"
        JStr(s) => s
    }
}

pub fn ok_caller(x Int) NonNegative {
    let a = clamp_non_neg(x)
    slice_len(a, a + 5)
}
"#).unwrap();
        let good_rep = check_program(&good);
        assert!(
            good_rep.ok,
            "valid contracts + exhaustive match should pass: {:?}",
            good_rep.diagnostics
        );
        assert!(good_rep.proofs.iter().any(|p| p.obligation.contains("exhaustive match on `JsonVal`")));
        assert!(good_rep.proofs.iter().any(|p| p.obligation.contains("precondition `requires` of `slice_len`")));

        // 2. Negative: missing variant in match (E0301), violated call-site requires (E0440),
        //    impure syscall in pure fn (E0610), immutable assign (E0203), type mismatch (E0205), undefined fn (E0201)!
        let bad = parse(r#"
pub type NonNegative = Int where self >= 0
pub enum JsonVal { JNull, JInt(Int), JStr(Text) }

pub pure fn slice_len(start NonNegative, end NonNegative) NonNegative
    requires start <= end
{
    end - start
}

pub pure fn bad_pure() Text {
    sys_cwd()
}

pub fn bad_checks(v JsonVal) Int {
    let x = 10
    x = 20
    let _ = slice_len(10, 3)
    let _ = unknown_fn(1)
    let _ Int = "not_an_int"
    match v {
        JNull => 0
        JInt(n) => n
    }
}
"#).unwrap();
        let bad_rep = check_program(&bad);
        assert!(!bad_rep.ok);
        let codes: Vec<&str> = bad_rep.diagnostics.iter().map(|d| d.code.as_str()).collect();
        assert!(codes.contains(&"E0610"), "expected E0610 purity error, got {:?}", codes);
        assert!(codes.contains(&"E0203"), "expected E0203 immutability error, got {:?}", codes);
        assert!(codes.contains(&"E0440"), "expected E0440 requires violation, got {:?}", codes);
        assert!(codes.contains(&"E0201"), "expected E0201 undefined fn, got {:?}", codes);
        assert!(codes.contains(&"E0205"), "expected E0205 type mismatch, got {:?}", codes);
        assert!(codes.contains(&"E0301"), "expected E0301 non-exhaustive match, got {:?}", codes);
    }

    #[test]
    fn rejects_stdlib_v2_contract_violations() {
        let bad = parse(r#"
pub type NonNegative = Int where self >= 0
pub type HexNibble   = Int where 0 <= self && self <= 15
pub type MaxBytes    = Int where self >= 64

pub record SliceRange {
    start NonNegative
    end   NonNegative
    where start <= end
}

pub enum JsonValue {
    JNull
    JStr(Text)
}

pub pure fn hex_nibble(n HexNibble) Text {
    byte_to_text(48 + n)
}

pub pure fn truncate_output(s Text, max_bytes MaxBytes) Text {
    str_slice(s, 0, max_bytes)
}

@no_heap_alloc
pub pure fn bad_no_heap(a Text, b Text) JsonValue {
    JStr(a ++ b)
}

pub fn bad_uses(opt Option[Int]) Int {
    let _ = SliceRange { start = 8, end = 3 }
    let _ = SliceRange { start = 0 }
    let _ = hex_nibble(20)
    let _ = truncate_output("hello", 10)
    let _ = hex_nibble(1, 2)
    match opt {
        Some(v) => v
    }
}
"#).unwrap();
        let rep = check_program(&bad);
        assert!(!rep.ok);
        let codes: Vec<&str> = rep.diagnostics.iter().map(|d| d.code.as_str()).collect();
        assert!(codes.contains(&"E0701"), "expected E0701 @no_heap_alloc violation, got {:?}", codes);
        assert!(codes.contains(&"E0414"), "expected E0414 SliceRange invariant violation, got {:?}", codes);
        assert!(codes.contains(&"E0204"), "expected E0204 missing record field, got {:?}", codes);
        assert!(codes.contains(&"E0410"), "expected E0410 HexNibble/MaxBytes violation, got {:?}", codes);
        assert!(codes.contains(&"E0202"), "expected E0202 arity mismatch, got {:?}", codes);
        assert!(codes.contains(&"E0301"), "expected E0301 non-exhaustive Option match, got {:?}", codes);
    }

    #[test]
    fn evolve_matches_by_field_name_and_from_rename_without_field_ids() {
        let v1 = parse(r#"
@evolve
pub record Order {
    id     Int64
    amount Int64 where self >= 0
}
"#).unwrap();
        let v2_safe = parse(r#"
@evolve
pub record Order {
    id Int64
    @from("amount") total_cents Int64 where self >= 0
    note Option[Text] = None
}
"#).unwrap();
        let rep_safe = check_evolution(&v1, &v2_safe);
        assert!(rep_safe.safe_to_release, "name-first @evolve + @from rename + optional field must be upgrade-safe");
        assert_eq!(rep_safe.entities[0].verdict, CompatVerdict::WideningBackwardSafe);
        assert_eq!(rep_safe.entities[0].added_optional_fields, vec!["note".to_string()]);

        let v2_narrow = parse(r#"
@evolve
pub record Order {
    id Int64
    @from("amount") total_cents Int64 where self > 0
}
"#).unwrap();
        let rep_narrow = check_evolution(&v1, &v2_narrow);
        assert!(!rep_narrow.safe_to_release, "narrowing renamed field must be caught by Z3");
        assert_eq!(rep_narrow.entities[0].verdict, CompatVerdict::NarrowingBreaking);
        assert_eq!(rep_narrow.entities[0].counterexample.get("amount").map(|s| s.as_str()), Some("0"));

        // Version-first field base type change (`Int64 -> Text`) caught automatically!
        let v2_type_change = parse(r#"
@evolve
pub record Order {
    id     Int64
    amount Text
}
"#).unwrap();
        let rep_tc = check_evolution(&v1, &v2_type_change);
        assert!(!rep_tc.safe_to_release, "field type change across versions must be flagged");
        assert_eq!(rep_tc.entities[0].verdict, CompatVerdict::DivergentBreaking);
    }

    #[test]
    fn verifies_interfaces_impls_liskov_contracts_static_bounds_and_dyn_object_safety() {
        // 1. Positive: interface with `ensures`, `pure`, static `[T Scorer]` AND `dyn Scorer`!
        let good = parse(r#"
pub type NonNegative = Int where self >= 0

pub interface Scorer {
    pure fn score(self, bonus NonNegative) Int
        ensures result >= 0
}

pub record FlatScorer {
    base NonNegative
}

impl Scorer for FlatScorer {
    pure fn score(self, bonus NonNegative) Int {
        self.base + bonus
    }
}

pub pure fn run_static[T Scorer](s T, b NonNegative) Int {
    s.score(b)
}

pub pure fn run_dynamic(s dyn Scorer, b NonNegative) Int {
    s.score(b)
}

pub fn demo() Int {
    let f = FlatScorer { base = 10 }
    let a = run_static(f, 5)
    let d dyn Scorer = f
    let b = run_dynamic(d, 3)
    var list List[dyn Scorer] = []
    list = push(list, f)
    a + b
}
"#).unwrap();
        let good_rep = check_program(&good);
        assert!(
            good_rep.ok,
            "valid interface + static bound + dyn dispatch should pass: {:?}",
            good_rep.diagnostics
        );
        assert!(good_rep
            .proofs
            .iter()
            .any(|p| p.obligation.contains("impl `Scorer` for `FlatScorer`")));

        // 2. Negative controls: missing method (E0501), purity mismatch (E0503),
        //    Liskov ensures violation (V1004), unsatisfied bound / dyn coercion (E0505),
        //    non-object-safe dyn (E0506)!
        let bad = parse(r#"
pub interface Scorer {
    pure fn score(self, x Int) Int
        ensures result >= 0
}

pub interface Factory {
    pure fn make() Self
}

pub record MissingImpl { v Int }
pub record ImpureImpl { v Int }
pub record BadContractImpl { v Int }
pub record NoImpl { v Int }

impl Scorer for MissingImpl {}

impl Scorer for ImpureImpl {
    fn score(self, x Int) Int { 1 }
}

impl Scorer for BadContractImpl {
    pure fn score(self, x Int) Int { -1 }
}

pub fn run_static[T Scorer](s T) Int { s.score(1) }

pub fn bad_uses(f dyn Factory) Int {
    let n = NoImpl { v = 1 }
    let _ = run_static(n)
    let _ dyn Scorer = n
    0
}
"#).unwrap();
        let bad_rep = check_program(&bad);
        assert!(!bad_rep.ok);
        let codes: Vec<&str> = bad_rep.diagnostics.iter().map(|d| d.code.as_str()).collect();
        assert!(codes.contains(&"E0501"), "expected E0501 missing method, got {:?}", codes);
        assert!(codes.contains(&"E0503"), "expected E0503 purity mismatch, got {:?}", codes);
        assert!(codes.contains(&"V1004"), "expected V1004 Liskov ensures violation, got {:?}", codes);
        assert!(codes.contains(&"E0505"), "expected E0505 unsatisfied interface bound/dyn, got {:?}", codes);
        assert!(codes.contains(&"E0506"), "expected E0506 non-object-safe dyn, got {:?}", codes);
    }

    #[test]
    fn enforces_oop_list_push_mutability_and_dyn_element_bounds() {
        let src = r#"
pub interface Tool {
    pure fn name(self) Text
}

pub record GoodTool { id Int }
pub record NotATool { id Int }

impl Tool for GoodTool {
    pure fn name(self) Text { "good" }
}

impl[T] List[T] {
    pub pure fn push(inout self, item T) {
        self = push(self, item)
    }
}

pub record Holder {
    tools List[dyn Tool]
}

impl Holder {
    pub pure fn bad_imm_field(self, t dyn Tool) {
        self.tools.push(t)
    }
}

pub fn bad_uses() Int {
    let frozen List[dyn Tool] = []
    frozen.push(GoodTool { id = 1 })

    var ok List[dyn Tool] = []
    ok.push(NotATool { id = 2 })
    0
}
"#;
        let prog = parse(src).unwrap();
        let rep = check_program(&prog);
        assert!(!rep.ok);
        let codes: Vec<&str> = rep.diagnostics.iter().map(|d| d.code.as_str()).collect();
        let e0203_count = codes.iter().filter(|&&c| c == "E0203").count();
        assert!(
            e0203_count >= 2,
            "expected >=2 E0203 (immutable `let` and immutable `self.tools.push`), got {:?}",
            rep.diagnostics
        );
        assert!(
            codes.contains(&"E0505"),
            "expected E0505 when pushing non-Tool into List[dyn Tool], got {:?}",
            rep.diagnostics
        );
    }

    #[test]
    fn enforces_index_assignment_mutability_and_type_checks() {
        let src = r#"
pub interface Item {
    pure fn id(self) Int
}
pub record GoodItem { v Int }
pub record BadItem  { v Int }

impl Item for GoodItem {
    pure fn id(self) Int { self.v }
}

pub fn bad_index_assign() Int {
    let frozen List[Int] = [1, 2, 3]
    frozen[0] = 99

    var xs List[Int] = [1, 2]
    xs[0] = "wrong_type"

    var dyn_xs List[dyn Item] = []
    dyn_xs[0] = BadItem { v = 7 }
    0
}
"#;
        let prog = parse(src).unwrap();
        let rep = check_program(&prog);
        assert!(!rep.ok);
        let codes: Vec<&str> = rep.diagnostics.iter().map(|d| d.code.as_str()).collect();
        assert!(codes.contains(&"E0203"), "expected E0203 on immutable index assign, got {:?}", codes);
        assert!(codes.contains(&"E0205"), "expected E0205 on mismatched element type, got {:?}", codes);
        assert!(codes.contains(&"E0505"), "expected E0505 on non-Item into List[dyn Item], got {:?}", codes);
    }
}
