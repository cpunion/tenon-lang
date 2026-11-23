pub mod checker;
pub mod evolve;
pub mod l0;
pub mod modelcheck;
pub mod smt;
pub mod sync_solver;

pub use checker::{CheckReport, Diagnostic, ProofRecord, ProofTier, check_program};
pub use evolve::{CompatVerdict, EvolutionReport, check_evolution};
pub use modelcheck::{ModelCheckSuiteReport, export_tla_specs, run_all_protocol_checks};
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
}
