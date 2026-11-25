mod lsp;

use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use tenon_codegen::JitProgram;
use tenon_dist::agent::TenonAgentRuntime;
use tenon_dist::agent_tools::AgentToolkit;
use tenon_dist::authority::{AuthorityStore, ExecSlice};
use tenon_dist::openai_responses::OpenAIResponsesConfig;
use tenon_dist::run_distributed_qualification;
use tenon_dist::server::{serve_http, serve_http_with_options};
use tenon_dist::sync_kernel::UniversalSyncKernel;
use tenon_dist::worker::{http_rpc, run_external_worker};
use tenon_syntax::{Item, discover_test_packages, load_program_for_test, load_program_with_imports};
use tenon_verify::{
    ProofTier, check_evolution, check_program, check_program_protocols, export_program_tla_specs,
    export_tla_specs, run_all_protocol_checks, verify_runtime_trace,
};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

#[tokio::main]
async fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        return ExitCode::from(2);
    }

    match args[1].as_str() {
        "check" => cmd_check(&args[2..]),
        "run" => cmd_run(&args[2..]),
        "test" => cmd_test(&args[2..]),
        "evolve" => cmd_evolve(&args[2..]),
        "modelcheck" => cmd_modelcheck(&args[2..]),
        "serve" => cmd_serve(&args[2..]).await,
        "authority" => cmd_authority(&args[2..]).await,
        "worker" => cmd_worker(&args[2..]).await,
        "cluster-demo" => cmd_cluster_demo(&args[2..]).await,
        "agent" => cmd_agent(&args[2..]).await,
        "build" => cmd_build_aot(&args[2..]),
        "build-wasm" => cmd_build_wasm(&args[2..]),
        "verify-proofs" => cmd_verify_proofs(&args[2..]),
        "pkg" => cmd_pkg(&args[2..]),
        "new" => cmd_new(&args[2..]),
        "doc" => cmd_doc(&args[2..]),
        "bench" => cmd_bench(&args[2..]),
        "fmt" => cmd_fmt(&args[2..]),
        "lsp" => lsp::run_lsp_stdio(),
        "-h" | "--help" | "help" => {
            print_usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown subcommand `{}`", other);
            print_usage();
            ExitCode::from(2)
        }
    }
}

fn print_usage() {
    println!(
        r#"tenon — Verified Systems, Web & Distributed Application Language

USAGE:
  tenon check <file.tn> [--json] [--emit-proofs <dir>] [--emit-repro-test <out.tn>]
                                                  Run L0 + L1 (Z3) Fail-Closed verification & Protocol Solver
  tenon verify-proofs <dir> [--solver z3|cvc5|auto]
                                                  Independently audit exported SMT-LIB2 proof certificates
  tenon run <file.tn>                             Verify and execute `main()` via Cranelift native JIT
  tenon test [file.tn|pkg_dir] [--filter <substr>] [-r|--recursive] [--no-leak-check]
                                                  Verify and run `test fn` tests with per-test Perceus zero-leak checks
  tenon build <file.tn> -o <out> [--opt O0|O2|O3] [--target <triple|wasm32>] [--lib static|shared]
        [--checked-overflow] [--emit-header <h>] [--emit-llvm <ll>] [--emit-ts <dts>]
        [--emit-swift <sw>] [--emit-kotlin <kt>] [--emit-go <go>] [-g|--debug] [-l <lib>] [-L <dir>] [--c-src <c>]
                                                  Verify (Fail-Closed) & AOT-compile to native/Wasm32/C-ABI libraries
  tenon build-wasm <app.tn> [-o <out.wasm>]       Compile declarative .tn view/pacing to binary WebAssembly store
  tenon evolve check <v1.tn> <v2.tn> [--lock <file>] [--emit-migration <sql>] [--json]
                                                  Verify schema evolution compatibility, ledger & Expand-Contract SQL
  tenon modelcheck [file.tn] [--export-tla <dir>] [--replay <trace.jsonl>] [--json]
                                                  Run L2-b protocol safety/liveness model checker & trace replay
  tenon pkg <lock|verify|tree|add|sync> [dir]     Manage `tenon.toml`, `tenon.lock` SHA-256 ledger & git deps
  tenon new <name> [--template cli|lib|web|gpui|openui-web|tui|gpu]
                                                  Scaffold a new verified Tenon project with tests & manifest
  tenon doc <file_or_dir> [-o <dir>] [--format md|html|both]
                                                  Generate Markdown & HTML API reference from verified AST
  tenon bench <file.tn> [--iters <N>]             Compile with LLVM -O3 and run timed benchmark iterations
  tenon serve|authority|worker|cluster-demo|agent Distributed runtime, Fenced Authority/Worker & CLI Agent
  tenon fmt <file.tn> [--check]                   Format Tenon source file with canonical indentation
  tenon lsp                                       Run Language Server Protocol (JSON-RPC 2.0 over stdio)"#
    );
}

fn load_and_parse(path: &str) -> Result<tenon_syntax::Program, String> {
    load_program_with_imports(Path::new(path))
}

fn format_rich_diagnostic(file: &str, prog: &tenon_syntax::Program, d: &tenon_verify::Diagnostic) -> String {
    let mut out = String::new();
    let diag_file = prog
        .source_files
        .get(d.span.file_id as usize)
        .map(|s| s.as_str())
        .unwrap_or(file);
    out.push_str(&format!(
        "error[{}]: {}\n  --> {}:{}:{}\n",
        d.code, d.message, diag_file, d.span.line, d.span.col
    ));
    if d.span.line > 0 {
        if let Ok(src) = fs::read_to_string(diag_file) {
            if let Some(src_line) = src.lines().nth(d.span.line - 1) {
                let lno = format!("{}", d.span.line);
                let pad = " ".repeat(lno.len());
                let col_off = d.span.col.saturating_sub(1);
                let width = d.span.end.saturating_sub(d.span.start).clamp(1, 32);
                out.push_str(&format!("  {} |\n", pad));
                out.push_str(&format!("  {} | {}\n", lno, src_line));
                out.push_str(&format!("  {} | {}{}\n", pad, " ".repeat(col_off), "^".repeat(width)));
            }
        }
    }
    for n in &d.notes {
        out.push_str(&format!("  = note: {}\n", n));
    }
    if !d.counterexample.is_empty() {
        let pairs: Vec<String> = d
            .counterexample
            .iter()
            .map(|(k, v)| format!("{} = {}", k, v))
            .collect();
        out.push_str(&format!("  = counterexample: {}\n", pairs.join(", ")));
    }
    out
}

fn emit_sarif_report(file: &str, prog: &tenon_syntax::Program, report: &tenon_verify::CheckReport) -> String {
    let results: Vec<serde_json::Value> = report
        .diagnostics
        .iter()
        .map(|d| {
            let diag_file = prog
                .source_files
                .get(d.span.file_id as usize)
                .map(|s| s.as_str())
                .unwrap_or(file);
            let mut msg = d.message.clone();
            if !d.counterexample.is_empty() {
                let pairs: Vec<String> = d
                    .counterexample
                    .iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect();
                msg.push_str(&format!(" (counterexample: {})", pairs.join(", ")));
            }
            serde_json::json!({
                "ruleId": d.code,
                "level": "error",
                "message": { "text": msg },
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": { "uri": diag_file },
                        "region": {
                            "startLine": d.span.line.max(1),
                            "startColumn": d.span.col.max(1)
                        }
                    }
                }]
            })
        })
        .collect();
    let doc = serde_json::json!({
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "tenon",
                    "version": "1.0.0",
                    "informationUri": "https://github.com/cpunion/tenon-lang"
                }
            },
            "results": results
        }]
    });
    serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".into())
}

fn cmd_check(args: &[String]) -> ExitCode {
    let mut file_opt: Option<&str> = None;
    let mut emit_proofs: Option<PathBuf> = None;
    let mut emit_repro_test: Option<PathBuf> = None;
    let mut format = "text".to_string();
    let as_json = args.iter().any(|a| a == "--json");
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--emit-proofs" if i + 1 < args.len() => {
                emit_proofs = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--emit-repro-test" if i + 1 < args.len() => {
                emit_repro_test = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--format" if i + 1 < args.len() => {
                format = args[i + 1].clone();
                i += 2;
            }
            "--json" => i += 1,
            s if !s.starts_with('-') && file_opt.is_none() => {
                file_opt = Some(s);
                i += 1;
            }
            _ => i += 1,
        }
    }
    let Some(file) = file_opt else {
        eprintln!("usage: tenon check <file.tn> [--json] [--format text|json|sarif] [--emit-proofs <dir>] [--emit-repro-test <out.tn>]");
        return ExitCode::from(2);
    };

    let prog = match load_and_parse(file) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::from(1);
        }
    };

    let use_disk_cache = emit_proofs.is_none()
        && emit_repro_test.is_none()
        && std::env::var("TENON_NO_CACHE").as_deref() != Ok("1");
    if emit_proofs.is_some() {
        tenon_verify::begin_proof_recording();
    }
    let (report, disk_cache_hit) = if use_disk_cache {
        tenon_verify::check_program_cached_on_disk(&prog, Path::new(".tenon-cache"))
    } else {
        (check_program(&prog), false)
    };
    let _ = disk_cache_hit;
    if let Some(pdir) = emit_proofs {
        let certs = tenon_verify::end_proof_recording();
        match tenon_verify::export_proof_certificates(&pdir, &certs) {
            Ok(n) => println!(
                "Exported {} independent SMT-LIB2 proof certificate(s) -> `{}`",
                n,
                pdir.display()
            ),
            Err(e) => eprintln!("warning: failed to export proof certificates: {}", e),
        }
    }
    if let Some(rpath) = emit_repro_test {
        let repro = tenon_verify::emit_repro_test_from_diagnostics(file, Some(&prog), &report.diagnostics);
        if let Err(e) = fs::write(&rpath, repro) {
            eprintln!("warning: failed to write repro test `{}`: {}", rpath.display(), e);
        } else {
            println!("Emitted counterexample regression test -> `{}`", rpath.display());
        }
    }
    if format == "sarif" {
        println!("{}", emit_sarif_report(file, &prog, &report));
        return if report.ok {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(1)
        };
    }
    if as_json || format == "json" {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        return if report.ok {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(1)
        };
    }

    let l0_count = report
        .proofs
        .iter()
        .filter(|p| p.tier == ProofTier::L0Fast)
        .count();
    let l1_count = report
        .proofs
        .iter()
        .filter(|p| p.tier == ProofTier::L1Smt)
        .count();

    if report.ok {
        println!(
            "PASS  {}  (proofs: {} L0-fast, {} L1-Z3)",
            file, l0_count, l1_count
        );
        for p in &report.proofs {
            println!("  [OK:{:?}] {} (line {})", p.tier, p.obligation, p.span.line);
        }
    } else {
        println!(
            "FAIL  {}  ({} diagnostic(s); {} L0, {} L1 proved)",
            file,
            report.diagnostics.len(),
            l0_count,
            l1_count
        );
        for d in &report.diagnostics {
            eprint!("{}", format_rich_diagnostic(file, &prog, d));
        }
    }

    if !report.sql_checks.is_empty() {
        println!("Derived SQLite CHECK constraints:");
        for (rec, sql) in &report.sql_checks {
            println!("  record {} => {}", rec, sql);
        }
    }

    if let Some(m) = &report.manifest {
        println!("Derived DistributedAppManifest (Universal SyncKernel):");
        println!("  partitioned entities: {:?}", m.partitioned_entities);
        for s in &m.streams {
            println!(
                "  stream `{}`: durability={}, db_writes={}, slow_consumer={}",
                s.name, s.durability, s.db_writes_per_delta, s.slow_consumer_policy
            );
        }
        for v in &m.views {
            println!(
                "  view `{}` ({}): archetype={:?}, max_items={}, max_payload={}B, max_frame={}B, bounded_reset={}, anti_join={}",
                v.name,
                v.sync_id,
                v.archetype,
                v.max_items,
                v.max_payload_bytes,
                v.max_frame_bytes,
                v.bounded_reset,
                v.uses_anti_join_overlay
            );
        }
        for sub in &m.subscribers {
            let coalesce = sub
                .coalesce_ms
                .map(|v| format!("{}ms", v))
                .unwrap_or_else(|| "-".into());
            let throttle = sub
                .throttle_ms
                .map(|v| format!("{}ms", v))
                .unwrap_or_else(|| "-".into());
            let split = sub
                .split_bytes
                .map(|v| format!("{}B", v))
                .unwrap_or_else(|| "-".into());
            println!(
                "  subscriber `{}` [{} -> {}]: coalesce={}, throttle={}, split={}, dead_stream_eliminated={}",
                sub.name,
                sub.kind,
                sub.bound_view,
                coalesce,
                throttle,
                split,
                sub.dead_stream_eliminated
            );
        }
    }

    if report.ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn cmd_run(args: &[String]) -> ExitCode {
    let Some(file) = args.first() else {
        eprintln!("usage: tenon run <file.tn>");
        return ExitCode::from(2);
    };
    let prog = match load_and_parse(file) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::from(1);
        }
    };
    // Fail-Closed: verify BEFORE compiling/executing native machine code!
    let report = check_program(&prog);
    if !report.ok {
        eprintln!("FAIL-CLOSED: verification failed for `{}`; refusing to execute.", file);
        for d in &report.diagnostics {
            eprintln!("  error[{}] line {}: {}", d.code, d.span.line, d.message);
            for n in &d.notes {
                eprintln!("    note: {}", n);
            }
        }
        return ExitCode::from(1);
    }

    tenon_codegen::perceus::reset_stats();
    let jit = match JitProgram::compile(&prog) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("codegen error: {}", e);
            return ExitCode::from(1);
        }
    };
    println!(
        "Verified ({} proofs) & JIT-compiled {} function(s) (derived parsers: {:?}) -> executing main()...",
        report.proofs.len(),
        jit.compiled_fns.len(),
        jit.derived_parsers
    );
    match jit.call_0("main") {
        Ok(val) => {
            let st = tenon_codegen::perceus::snapshot_stats();
            println!("main() => {}", val);
            if st.total_allocs > 0 || st.async_polls > 0 {
                println!(
                    "Perceus/Async stats: allocs={}, unique_reuse_hits={}, cow_clones={}, live_objects={}, async_polls={}, async_suspends={}",
                    st.total_allocs,
                    st.reuse_hits,
                    st.cow_clones,
                    st.live_objects,
                    st.async_polls,
                    st.async_suspends
                );
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("runtime error: {}", e);
            ExitCode::from(1)
        }
    }
}

fn cmd_test(args: &[String]) -> ExitCode {
    let mut target: Option<String> = None;
    let mut filter = String::new();
    let mut recursive = false;
    let mut check_leaks = true;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-r" | "--recursive" => {
                recursive = true;
                i += 1;
            }
            "--filter" | "-f" => {
                if i + 1 < args.len() {
                    filter = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("error: `--filter` requires a pattern argument");
                    return ExitCode::from(2);
                }
            }
            "--no-leak-check" => {
                check_leaks = false;
                i += 1;
            }
            s if s.starts_with("--filter=") => {
                filter = s["--filter=".len()..].to_string();
                i += 1;
            }
            "./..." => {
                recursive = true;
                target = Some(".".into());
                i += 1;
            }
            other if !other.starts_with('-') => {
                if let Some(prefix) = other.strip_suffix("/...") {
                    recursive = true;
                    target = Some(if prefix.is_empty() { ".".into() } else { prefix.into() });
                } else {
                    target = Some(other.to_string());
                }
                i += 1;
            }
            other => {
                eprintln!("unknown option `{}` for `tenon test`", other);
                eprintln!("usage: tenon test [file.tn|pkg_dir] [--filter <substr>] [-r|--recursive]");
                return ExitCode::from(2);
            }
        }
    }

    let root_str = target.unwrap_or_else(|| ".".to_string());
    let root_path = Path::new(&root_str);
    let packages: Vec<PathBuf> = if recursive {
        match discover_test_packages(root_path) {
            Ok(ps) if !ps.is_empty() => ps,
            Ok(_) => {
                eprintln!("No Tenon packages found under `{}`.", root_str);
                return ExitCode::from(1);
            }
            Err(e) => {
                eprintln!("error: {}", e);
                return ExitCode::from(1);
            }
        }
    } else {
        vec![root_path.to_path_buf()]
    };

    let mut total_passed = 0usize;
    let mut total_failed = 0usize;
    let mut total_filtered = 0usize;

    for pkg in &packages {
        let prog = match load_program_for_test(pkg) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("FAIL `{}`: {}", pkg.display(), e);
                total_failed += 1;
                continue;
            }
        };
        let report = check_program(&prog);
        if !report.ok {
            eprintln!("FAIL-CLOSED: verification failed for `{}`.", pkg.display());
            for d in &report.diagnostics {
                let fname = prog
                    .source_files
                    .get(d.span.file_id as usize)
                    .map(|s| s.as_str())
                    .unwrap_or("<input>");
                eprintln!(
                    "  error[{}] {}:{}:{}: {}",
                    d.code, fname, d.span.line, d.span.col, d.message
                );
            }
            total_failed += 1;
            continue;
        }

        let all_tests: Vec<String> = prog
            .items
            .iter()
            .filter_map(|it| match it {
                Item::Fn(f) if f.is_test => Some(f.name.clone()),
                _ => None,
            })
            .collect();
        if all_tests.is_empty() {
            if !recursive {
                println!("No `test fn` items found in `{}`.", pkg.display());
            }
            continue;
        }

        let selected: Vec<String> = all_tests
            .iter()
            .filter(|n| filter.is_empty() || n.contains(&filter))
            .cloned()
            .collect();
        total_filtered += all_tests.len() - selected.len();
        if selected.is_empty() {
            continue;
        }

        let jit = match JitProgram::compile(&prog) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("FAIL `{}` codegen error: {}", pkg.display(), e);
                total_failed += selected.len();
                continue;
            }
        };

        println!(
            "running {} test(s) in `{}` ({} proofs verified)",
            selected.len(),
            pkg.display(),
            report.proofs.len()
        );
        for name in &selected {
            let t0 = std::time::Instant::now();
            tenon_codegen::perceus::reset_stats();
            match jit.call_0(name) {
                Ok(_) => {
                    let elapsed = t0.elapsed();
                    let st = tenon_codegen::perceus::snapshot_stats();
                    if check_leaks && st.live_objects != 0 {
                        eprintln!(
                            "  test {} ... FAILED (Perceus memory leak: live_objects={})",
                            name, st.live_objects
                        );
                        total_failed += 1;
                    } else {
                        println!(
                            "  test {} ... ok ({:.2}ms, allocs={}, live=0)",
                            name,
                            elapsed.as_secs_f64() * 1000.0,
                            st.total_allocs
                        );
                        total_passed += 1;
                    }
                }
                Err(e) => {
                    eprintln!("  test {} ... FAILED ({})", name, e);
                    total_failed += 1;
                }
            }
        }
    }

    println!(
        "\ntest result: {}. {} passed; {} failed; {} filtered out",
        if total_failed == 0 { "ok" } else { "FAILED" },
        total_passed,
        total_failed,
        total_filtered
    );
    if total_failed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn cmd_evolve(args: &[String]) -> ExitCode {
    if args.len() < 3 || args[0] != "check" {
        eprintln!("usage: tenon evolve check <v1.tn> <v2.tn> [--json] [--lock <evolution.lock>] [--emit-migration <out.sql>]");
        return ExitCode::from(2);
    }
    let v1_path = &args[1];
    let v2_path = &args[2];
    let as_json = args.iter().any(|a| a == "--json");
    let lock_path = args
        .windows(2)
        .find(|w| w[0] == "--lock")
        .map(|w| PathBuf::from(&w[1]));
    let migration_path = args
        .windows(2)
        .find(|w| w[0] == "--emit-migration")
        .map(|w| PathBuf::from(&w[1]));

    let v1 = match load_and_parse(v1_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::from(1);
        }
    };
    let v2 = match load_and_parse(v2_path) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::from(1);
        }
    };

    let rep = check_evolution(&v1, &v2);
    if let Some(lp) = &lock_path {
        match tenon_verify::append_evolution_lock(lp, &rep, v1_path, v2_path) {
            Ok(ent) => println!(
                "Appended evolution receipt #{} ({}…) -> `{}`",
                ent.seq,
                &ent.entry_sha256[..12],
                lp.display()
            ),
            Err(e) => eprintln!("warning: failed to update evolution lock: {}", e),
        }
    }
    if let Some(mp) = &migration_path {
        let sql = tenon_verify::emit_expand_contract_sql(&v1, &v2, &rep);
        if let Some(parent) = mp.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        if let Err(e) = fs::write(mp, sql) {
            eprintln!("failed to write `{}`: {}", mp.display(), e);
            return ExitCode::from(1);
        }
        println!("Emitted verified Expand-Contract SQL migration -> `{}`", mp.display());
    }
    if as_json {
        println!("{}", serde_json::to_string_pretty(&rep).unwrap());
        return if rep.safe_to_release {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(1)
        };
    }

    println!("Evolution Compatibility Check: {} -> {}", v1_path, v2_path);
    println!("  v1 receipt sha256: {}", &rep.v1_receipt_sha256[..16]);
    println!("  v2 receipt sha256: {}", &rep.v2_receipt_sha256[..16]);
    for ent in &rep.entities {
        println!(
            "  entity `{}` ({}): verdict={:?} upgrade_safe={} downgrade_safe={}",
            ent.entity_id, ent.record_name, ent.verdict, ent.upgrade_safe, ent.downgrade_safe
        );
        if !ent.counterexample.is_empty() {
            println!("    Z3 counterexample: {:?}", ent.counterexample);
        }
        if let Some(delta) = &ent.residual_domain {
            println!("    residual domain Δ: {}", delta);
        }
        for n in &ent.notes {
            println!("    note: {}", n);
        }
        if let Some(lens) = &ent.lens_verified {
            println!(
                "    evolution lens {} ({} <-> {}): totality={} roundtrip={} retained={:?}",
                lens.entity,
                lens.from_ver,
                lens.to_ver,
                lens.totality_proved,
                lens.roundtrip_proved,
                lens.retained_fields
            );
            for d in &lens.details {
                println!("      * {}", d);
            }
        }
    }
    for cap in &rep.capability_checks {
        println!(
            "  capability `{}`: required={:?} supported={:?} compatible={}",
            cap.operation, cap.required, cap.supported, cap.compatible
        );
    }

    if rep.safe_to_release {
        println!("VERDICT: SAFE TO RELEASE");
        ExitCode::SUCCESS
    } else {
        println!("VERDICT: BLOCKED (breaking schema evolution without verified lens)");
        ExitCode::from(1)
    }
}

fn cmd_modelcheck(args: &[String]) -> ExitCode {
    let as_json = args.iter().any(|a| a == "--json");
    let export_dir = args
        .windows(2)
        .find(|w| w[0] == "--export-tla")
        .map(|w| w[1].as_str());
    let replay_trace = args
        .windows(2)
        .find(|w| w[0] == "--replay")
        .map(|w| w[1].as_str());
    if let Some(tpath) = replay_trace {
        return match tenon_verify::replay_trace_jsonl_file(Path::new(tpath)) {
            Ok(tr) => {
                if as_json {
                    println!("{}", serde_json::to_string_pretty(&tr).unwrap());
                } else {
                    println!(
                        "Tenon Trace Refinement Replay (`{}`): {} ({} event(s) checked)",
                        tpath,
                        if tr.ok { "PASS" } else { "FAIL" },
                        tr.events_checked
                    );
                    for v in &tr.violations {
                        println!("  violation: {}", v);
                    }
                }
                if tr.ok { ExitCode::SUCCESS } else { ExitCode::from(1) }
            }
            Err(e) => {
                eprintln!("error: {}", e);
                ExitCode::from(1)
            }
        };
    }
    // Optional `.tn` file argument (skipping flag values like `--export-tla <dir>`)
    let mut file_arg: Option<&str> = None;
    let mut skip_next = false;
    for a in args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if a == "--export-tla" || a == "--replay" {
            skip_next = true;
            continue;
        }
        if !a.starts_with('-') {
            file_arg = Some(a.as_str());
            break;
        }
    }

    let (report, exported) = if let Some(path) = file_arg {
        let prog = match load_and_parse(path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("error: {}", e);
                return ExitCode::from(1);
            }
        };
        let rep = check_program_protocols(&prog);
        let exp = if let Some(dir) = export_dir {
            match export_program_tla_specs(&prog, Path::new(dir)) {
                Ok(files) => files,
                Err(e) => {
                    eprintln!("failed to export TLA+ specs: {}", e);
                    return ExitCode::from(1);
                }
            }
        } else {
            Vec::new()
        };
        (rep, exp)
    } else {
        let rep = run_all_protocol_checks();
        let exp = if let Some(dir) = export_dir {
            match export_tla_specs(Path::new(dir)) {
                Ok(files) => files,
                Err(e) => {
                    eprintln!("failed to export TLA+ specs: {}", e);
                    return ExitCode::from(1);
                }
            }
        } else {
            Vec::new()
        };
        (rep, exp)
    };

    if as_json {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
        return if report.ok {
            ExitCode::SUCCESS
        } else {
            ExitCode::from(1)
        };
    }

    println!(
        "Tenon L2-b Protocol Model Checker ({}):",
        file_arg.unwrap_or("built-in core protocols")
    );
    for c in &report.cases {
        let status = if c.passed { "PASS" } else { "FAIL" };
        let kind = if c.expect_violation {
            "negative-control"
        } else {
            "safety-proof"
        };
        println!(
            "  [{}] {} :: {} ({}) — {} states explored",
            status, c.protocol, c.case_name, kind, c.states_explored
        );
        if let Some(inv) = &c.violated_invariant {
            println!(
                "         {} `{}` via trace: {}",
                if c.passed { "caught" } else { "VIOLATED" },
                inv,
                c.counterexample_trace.join(" -> ")
            );
        }
    }
    if !exported.is_empty() {
        println!("Exported TLA+ specs for TLC cross-checking:");
        for f in &exported {
            println!("  - {}", f);
        }
    }

    if report.ok {
        println!("ALL {} PROTOCOL CHECKS PASSED", report.cases.len());
        ExitCode::SUCCESS
    } else {
        println!("MODEL CHECK FAILED");
        ExitCode::from(1)
    }
}

async fn cmd_serve(args: &[String]) -> ExitCode {
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: tenon serve <app.tn> [--addr 127.0.0.1:8787] [--db :memory:] [--demo]");
        return ExitCode::from(2);
    };
    let demo_mode = args.iter().any(|a| a == "--demo");
    let addr = args
        .windows(2)
        .find(|w| w[0] == "--addr")
        .map(|w| w[1].as_str())
        .unwrap_or("127.0.0.1:8787");
    let db_path = args
        .windows(2)
        .find(|w| w[0] == "--db")
        .map(|w| w[1].as_str())
        .unwrap_or(":memory:");

    let prog = match load_and_parse(file) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::from(1);
        }
    };
    let report = check_program(&prog);
    if !report.ok {
        eprintln!("FAIL-CLOSED: cannot serve `{}` due to verification/protocol errors:", file);
        for d in &report.diagnostics {
            eprintln!("  error[{}] line {}: {}", d.code, d.span.line, d.message);
            for n in &d.notes {
                eprintln!("    note: {}", n);
            }
        }
        return ExitCode::from(1);
    }

    let Some(manifest) = report.manifest else {
        eprintln!("error: `{}` defines no distributed `@sync_view`/`@stream`/`@partition` items.", file);
        return ExitCode::from(1);
    };

    if demo_mode {
        println!("=== Tenon Distributed Prototype Demo ({}) ===", file);
        println!(
            "1) Protocol Constraint Solver compiled {} view(s), {} stream(s), {} subscriber(s) -> Universal SyncKernel",
            manifest.views.len(),
            manifest.streams.len(),
            manifest.subscribers.len()
        );
        match run_distributed_qualification(manifest.clone()).await {
            Ok(summary) => {
                println!("2) Live Multi-Worker Qualification (SQLite + Fencing + Turn DAG + Anti-Join + Channel Pacing):");
                println!("{}", serde_json::to_string_pretty(&summary).unwrap());
            }
            Err(e) => {
                eprintln!("distributed qualification FAILED: {}", e);
                return ExitCode::from(1);
            }
        }

        // 3) Boot real TCP HTTP/SSE Gateway + verify live SSE streaming & Web UI over loopback!
        let store = match AuthorityStore::open(Path::new(":memory:")) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("sqlite open failed: {}", e);
                return ExitCode::from(1);
            }
        };
        let kernel = Arc::new(UniversalSyncKernel::new(store, manifest));
        let bound = match serve_http(kernel, "127.0.0.1:0").await {
            Ok(b) => b,
            Err(e) => {
                eprintln!("failed to start HTTP/SSE server: {}", e);
                return ExitCode::from(1);
            }
        };
        match smoke_test_live_http_sse(&bound).await {
            Ok((ui_bytes, manifest_ok, sse_events)) => {
                println!(
                    "3) Live TCP HTTP/SSE Gateway smoke test on http://{}/ PASSED:",
                    bound
                );
                println!("   - GET / (Web UI with 33ms coalesce_window): {} bytes", ui_bytes);
                println!("   - GET /api/manifest: ok={}", manifest_ok);
                println!(
                    "   - POST /api/sessions/main/send + GET /api/client/stream received {} live SSE frames (hello + snapshot + streaming deltas + committed handoff)",
                    sse_events
                );
            }
            Err(e) => {
                eprintln!("live HTTP/SSE smoke test FAILED: {}", e);
                return ExitCode::from(1);
            }
        }
        return ExitCode::SUCCESS;
    }

    let store = match AuthorityStore::open(Path::new(db_path)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to open authority store `{}`: {}", db_path, e);
            return ExitCode::from(1);
        }
    };
    let kernel = Arc::new(UniversalSyncKernel::new(store, manifest));
    let bound = match serve_http(kernel, addr).await {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to bind `{}`: {}", addr, e);
            return ExitCode::from(1);
        }
    };
    println!("Tenon Distributed Runtime serving `{}`:", file);
    println!("  Web UI:      http://{}/", bound);
    println!("  SSE Stream:  http://{}/api/client/stream", bound);
    println!("  Manifest:    http://{}/api/manifest", bound);
    println!("  Channels:    http://{}/api/channels", bound);
    println!("Press Ctrl-C to stop.");
    let _ = tokio::signal::ctrl_c().await;
    ExitCode::SUCCESS
}

async fn smoke_test_live_http_sse(addr: &str) -> Result<(usize, bool, usize), String> {
    // 1. GET / & GET /tenon_client_store.wasm
    let ui_resp = http_request(addr, "GET", "/", None).await?;
    if !ui_resp.starts_with("HTTP/1.1 200") || !ui_resp.contains("tenon_client_store.wasm") {
        return Err("GET / did not serve expected WASM-backed 33ms Web UI".into());
    }
    let wasm_resp = http_request(addr, "GET", "/tenon_client_store.wasm", None).await?;
    if !wasm_resp.starts_with("HTTP/1.1 200") || !wasm_resp.contains("application/wasm") {
        return Err("GET /tenon_client_store.wasm did not return valid application/wasm".into());
    }

    // 2. GET /api/manifest
    let man_resp = http_request(addr, "GET", "/api/manifest", None).await?;
    let manifest_ok = man_resp.starts_with("HTTP/1.1 200") && man_resp.contains("FencedIncremental");
    if !manifest_ok {
        return Err("GET /api/manifest missing FencedIncremental archetype".into());
    }

    // 3. Open live SSE stream BEFORE sending message (subscribe-before-read invariant!)
    let mut sse_sock = TcpStream::connect(addr).await.map_err(|e| e.to_string())?;
    let sse_req = format!(
        "GET /api/client/stream HTTP/1.1\r\nHost: {}\r\nAccept: text/event-stream\r\n\r\n",
        addr
    );
    sse_sock
        .write_all(sse_req.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    let mut sse_reader = BufReader::new(sse_sock);

    // Read HTTP headers
    loop {
        let mut line = String::new();
        sse_reader
            .read_line(&mut line)
            .await
            .map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            break;
        }
    }

    // Read initial hello + snapshot frames
    let mut saw_hello = false;
    let mut saw_delta = false;
    let mut saw_final_snapshot = false;
    let mut frames = 0usize;

    // Trigger POST /api/sessions/main/send in parallel so background worker streams tokens!
    let addr_owned = addr.to_string();
    tokio::spawn(async move {
        let _ = http_request(
            &addr_owned,
            "POST",
            "/api/sessions/main/send",
            Some(r#"{"client_msg_id":"demo_1","text":"hello tenon live sse"}"#),
        )
        .await;
    });

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(4);
    while tokio::time::Instant::now() < deadline {
        let mut line = String::new();
        let read_res = tokio::time::timeout(
            std::time::Duration::from_millis(800),
            sse_reader.read_line(&mut line),
        )
        .await;
        match read_res {
            Ok(Ok(0)) => break,
            Ok(Ok(_)) => {
                if let Some(payload) = line.strip_prefix("data: ") {
                    frames += 1;
                    if payload.contains(r#""event":"hello""#) {
                        saw_hello = true;
                    }
                    if payload.contains(r#""event":"session.delta""#) {
                        saw_delta = true;
                    }
                    if payload.contains(r#""event":"session.snapshot""#)
                        && payload.contains("verified and complete.")
                        && payload.contains(r#""live_bubble":null"#)
                    {
                        saw_final_snapshot = true;
                        break;
                    }
                }
            }
            _ => break,
        }
    }

    if !saw_hello || !saw_delta || !saw_final_snapshot {
        return Err(format!(
            "incomplete SSE flow (hello={}, delta={}, final_snapshot={}, frames={})",
            saw_hello, saw_delta, saw_final_snapshot, frames
        ));
    }

    Ok((ui_resp.len(), manifest_ok, frames))
}

async fn http_request(
    addr: &str,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> Result<String, String> {
    let mut sock = TcpStream::connect(addr).await.map_err(|e| e.to_string())?;
    let payload = body.unwrap_or("");
    let req = format!(
        "{} {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        method,
        path,
        addr,
        payload.len(),
        payload
    );
    sock.write_all(req.as_bytes())
        .await
        .map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    sock.read_to_end(&mut buf)
        .await
        .map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

async fn cmd_authority(args: &[String]) -> ExitCode {
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: tenon authority <app.tn> [--addr 127.0.0.1:8787] [--db :memory:]");
        return ExitCode::from(2);
    };
    let addr = args
        .windows(2)
        .find(|w| w[0] == "--addr")
        .map(|w| w[1].as_str())
        .unwrap_or("127.0.0.1:8787");
    let db_path = args
        .windows(2)
        .find(|w| w[0] == "--db")
        .map(|w| w[1].as_str())
        .unwrap_or(":memory:");

    let prog = match load_and_parse(file) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::from(1);
        }
    };
    let report = check_program(&prog);
    if !report.ok {
        eprintln!("FAIL-CLOSED: verification failed for `{}`.", file);
        return ExitCode::from(1);
    }
    let Some(manifest) = report.manifest else {
        eprintln!("error: `{}` defines no distributed manifest.", file);
        return ExitCode::from(1);
    };
    let store = match AuthorityStore::open(Path::new(db_path)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to open authority store `{}`: {}", db_path, e);
            return ExitCode::from(1);
        }
    };
    let kernel = Arc::new(UniversalSyncKernel::new(store, manifest));
    let bound = match serve_http_with_options(kernel, addr, false).await {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to bind `{}`: {}", addr, e);
            return ExitCode::from(1);
        }
    };
    println!(
        "[authority pid={}] listening on http://{} (external worker mode, db={})",
        std::process::id(),
        bound,
        db_path
    );
    let _ = tokio::signal::ctrl_c().await;
    ExitCode::SUCCESS
}

async fn cmd_worker(args: &[String]) -> ExitCode {
    let authority = args
        .windows(2)
        .find(|w| w[0] == "--authority")
        .map(|w| w[1].as_str())
        .unwrap_or("http://127.0.0.1:8787");
    let worker_id = args
        .windows(2)
        .find(|w| w[0] == "--id")
        .map(|w| w[1].as_str())
        .unwrap_or("worker-1");
    let session = args
        .windows(2)
        .find(|w| w[0] == "--session")
        .map(|w| w[1].as_str())
        .unwrap_or("main");
    let delay_ms: u64 = args
        .windows(2)
        .find(|w| w[0] == "--delay-ms")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(40);
    let takeover = args.iter().any(|a| a == "--takeover");
    let once = args.iter().any(|a| a == "--once");

    match run_external_worker(authority, worker_id, session, delay_ms, takeover, once).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[worker {}] error: {}", worker_id, e);
            ExitCode::from(1)
        }
    }
}

/// Real Multi-Process Distributed Chaos Qualification:
/// Spawns separate OS processes for Authority, Worker-1 (killed with real SIGKILL mid-flight!),
/// Worker-2 (live takeover + Fabric Sandbox child process!), verifies HTTP 409 Fencing,
/// HTTP 410 Gone (reset_required) after Journal GC, and Runtime Trace Refinement!
async fn cmd_cluster_demo(args: &[String]) -> ExitCode {
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: tenon cluster-demo <app.tn>");
        return ExitCode::from(2);
    };
    let exe = match env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("failed to locate current exe: {}", e);
            return ExitCode::from(1);
        }
    };

    // Reserve a free loopback TCP port cleanly
    let port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        l.local_addr().unwrap().port()
    };
    let addr = format!("127.0.0.1:{}", port);
    let db_file = format!("/tmp/tenon_cluster_demo_{}.db", std::process::id());
    let _ = fs::remove_file(&db_file);

    println!("=== Tenon Multi-Process Distributed Chaos Demo ({}) ===", file);
    let mut auth_proc = match std::process::Command::new(&exe)
        .args(["authority", file, "--addr", &addr, "--db", &db_file])
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to spawn authority process: {}", e);
            return ExitCode::from(1);
        }
    };
    let auth_pid = auth_proc.id();

    // Wait until Authority TCP endpoint is ready
    let mut ready = false;
    for _ in 0..40 {
        if let Ok((200, _)) = http_rpc(&addr, "GET", "/api/state", "").await {
            ready = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    if !ready {
        let _ = auth_proc.kill();
        eprintln!("authority process did not become ready");
        return ExitCode::from(1);
    }

    // 1. Spawn Worker-1 OS process (slow streaming so we can SIGKILL it mid-flight!)
    let mut w1_proc = std::process::Command::new(&exe)
        .args([
            "worker",
            "--authority",
            &format!("http://{}", addr),
            "--id",
            "worker-1",
            "--delay-ms",
            "250",
        ])
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .expect("spawn worker-1");
    let w1_pid = w1_proc.id();

    // 2. Send user task -> wait for Worker-1 to claim Epoch 1 and enter `executing`!
    let (sc, _) = http_rpc(
        &addr,
        "POST",
        "/api/sessions/main/send",
        r#"{"client_msg_id":"cluster_m1","text":"survive sigkill failover"}"#,
    )
    .await
    .unwrap();
    assert_eq!(sc, 200);

    let mut w1_claimed = false;
    for _ in 0..50 {
        if let Ok((200, body)) = http_rpc(&addr, "GET", "/api/state", "").await {
            if body.contains(r#""phase": "executing""#) && body.contains(r#""exec_epoch": 1"#) {
                w1_claimed = true;
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(40)).await;
    }
    if !w1_claimed {
        let _ = w1_proc.kill();
        let _ = auth_proc.kill();
        eprintln!("worker-1 did not claim epoch 1");
        return ExitCode::from(1);
    }

    // 3. REAL SIGKILL on Worker-1 mid-flight!
    println!(
        ">>> Injecting REAL SIGKILL (kill -9) on Worker-1 (pid={}) mid-execution on epoch=1!",
        w1_pid
    );
    let _ = w1_proc.kill();
    let _ = w1_proc.wait();

    // 4. Spawn Worker-2 OS process to take over Epoch 1 -> Epoch 2, execute Fabric Sandbox, and commit!
    let mut w2_proc = std::process::Command::new(&exe)
        .args([
            "worker",
            "--authority",
            &format!("http://{}", addr),
            "--id",
            "worker-2",
            "--takeover",
            "--once",
            "--delay-ms",
            "10",
        ])
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .expect("spawn worker-2");
    let w2_pid = w2_proc.id();
    let w2_status = w2_proc.wait().expect("wait worker-2");
    assert!(w2_status.success(), "worker-2 failed");

    // 5. Verify zombie stale Epoch 1 RPCs get HTTP 409 Conflict from Authority (BOTH Proxy AND Storage)!
    let stale_slice = ExecSlice {
        session_id: "main".into(),
        worker_id: "zombie-worker-1".into(),
        conv_gen: 1,
        branch_id: 1,
        run_id: "run_cluster_m1".into(),
        stream_id: "stream_stale".into(),
        exec_epoch: 1,
        prompt: "stale".into(),
    };
    let (proxy_409, proxy_body) = http_rpc(
        &addr,
        "POST",
        "/rpc/proxy/token",
        &serde_json::json!({"slice": stale_slice, "seq": 99, "delta": "zombie"}).to_string(),
    )
    .await
    .unwrap();
    let (commit_409, commit_body) = http_rpc(
        &addr,
        "POST",
        "/rpc/commit",
        &serde_json::json!({"slice": stale_slice, "assistant_text": "zombie commit"}).to_string(),
    )
    .await
    .unwrap();

    // 6. Trigger Journal GC Floor and verify offline client receives HTTP 410 Gone (reset_required)!
    let (_, gc_body) = http_rpc(&addr, "POST", "/api/admin/gc_floor", r#"{"retain_last":0}"#)
        .await
        .unwrap();
    let (sync_410, sync_body) = http_rpc(&addr, "GET", "/api/sessions/main/sync?since_seq=0", "")
        .await
        .unwrap();

    // 7. Read authoritative state & live Trace Refinement verification report!
    let (_, state_body) = http_rpc(&addr, "GET", "/api/state", "").await.unwrap();
    let (_, trace_body) = http_rpc(&addr, "GET", "/api/trace", "").await.unwrap();

    let _ = auth_proc.kill();
    let _ = auth_proc.wait();
    let _ = fs::remove_file(&db_file);
    let _ = fs::remove_file(format!("{}-wal", db_file));
    let _ = fs::remove_file(format!("{}-shm", db_file));

    println!("=== Multi-Process Cluster Qualification Summary ===");
    println!("  OS Processes: authority pid={}, killed worker-1 pid={}, takeover worker-2 pid={}", auth_pid, w1_pid, w2_pid);
    println!("  Stale Proxy RPC (epoch=1 vs 2):   HTTP {} -> {}", proxy_409, proxy_body.trim());
    println!("  Stale Storage RPC (epoch=1 vs 2): HTTP {} -> {}", commit_409, commit_body.trim());
    println!("  Journal GC floor advance:         {}", gc_body.trim());
    println!("  Offline Sync (since_seq=0):       HTTP {} -> {}", sync_410, sync_body.trim());
    println!("  Trace Refinement Verification:    {}", trace_body.trim());

    let ok = proxy_409 == 409
        && commit_409 == 409
        && sync_410 == 410
        && state_body.contains("worker-2")
        && state_body.contains("sandbox_pid=")
        && trace_body.contains(r#""ok": true"#);

    if ok {
        println!("MULTI-PROCESS CLUSTER DEMO PASSED 100%");
        ExitCode::SUCCESS
    } else {
        eprintln!("MULTI-PROCESS CLUSTER DEMO FAILED; state={}", state_body);
        ExitCode::from(1)
    }
}

fn cmd_build_wasm(args: &[String]) -> ExitCode {
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: tenon build-wasm <app.tn> [-o <out.wasm>]");
        return ExitCode::from(2);
    };
    let out_path = args
        .windows(2)
        .find(|w| w[0] == "-o")
        .map(|w| w[1].as_str())
        .unwrap_or("target/tenon_client_store.wasm");

    let prog = match load_and_parse(file) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::from(1);
        }
    };
    let emit_ts = args
        .windows(2)
        .find(|w| w[0] == "--emit-ts")
        .map(|w| PathBuf::from(&w[1]));
    let report = check_program(&prog);
    if !report.ok {
        eprintln!("FAIL-CLOSED: verification failed for `{}`.", file);
        return ExitCode::from(1);
    }
    if let Some(ts_path) = &emit_ts {
        let ts = tenon_codegen::bindings::emit_typescript_bindings(&prog);
        if let Some(parent) = ts_path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        if let Err(e) = fs::write(ts_path, ts) {
            eprintln!("failed to write `{}`: {}", ts_path.display(), e);
            return ExitCode::from(1);
        }
        println!("Emitted TypeScript Wasm bindings -> `{}`", ts_path.display());
    }
    let Some(manifest) = report.manifest else {
        // General-purpose Tenon -> standalone `.wasm` compilation!
        let bytes = match tenon_codegen::wasm_general::compile_program_to_wasm(&prog) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("general wasm32 compilation failed: {}", e);
                return ExitCode::from(1);
            }
        };
        if let Some(parent) = Path::new(out_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
        if let Err(e) = fs::write(out_path, &bytes) {
            eprintln!("failed to write `{}`: {}", out_path, e);
            return ExitCode::from(1);
        }
        println!(
            "Compiled general Wasm32 `{}` -> `{}` ({} bytes, {} proofs verified)",
            file,
            out_path,
            bytes.len(),
            report.proofs.len()
        );
        return ExitCode::SUCCESS;
    };

    let coalesce_ms = manifest
        .subscribers
        .iter()
        .find(|s| s.kind == "client")
        .and_then(|s| s.coalesce_ms)
        .unwrap_or(33) as u32;
    let debounce_cards_ms = manifest
        .subscribers
        .iter()
        .find(|s| s.kind == "client")
        .and_then(|s| s.debounce_cards_ms)
        .unwrap_or(100) as u32;
    let anti_join = manifest.views.iter().any(|v| v.uses_anti_join_overlay);

    let bytes = tenon_codegen::wasm_client::compile_wasm_client_store(
        tenon_codegen::wasm_client::WasmClientConfig {
            coalesce_ms,
            anti_join,
            debounce_cards_ms,
        },
    );
    if let Some(parent) = Path::new(out_path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Err(e) = fs::write(out_path, &bytes) {
        eprintln!("failed to write `{}`: {}", out_path, e);
        return ExitCode::from(1);
    }
    println!(
        "Compiled `{}` -> `{}` ({} bytes, coalesce={}ms, debounce_cards={}ms, anti_join={})",
        file,
        out_path,
        bytes.len(),
        coalesce_ms,
        debounce_cards_ms,
        anti_join
    );
    ExitCode::SUCCESS
}

fn cmd_build_aot(args: &[String]) -> ExitCode {
    let mut file_arg: Option<String> = None;
    let mut out_bin = "target/tenon_out".to_string();
    let mut opts = tenon_codegen::llvm_aot::BuildOptions::default();
    let mut emit_mir: Option<PathBuf> = None;
    let mut emit_msl: Option<PathBuf> = None;
    let mut emit_wgsl: Option<PathBuf> = None;
    let mut emit_webgpu: Option<PathBuf> = None;
    let mut emit_ts: Option<PathBuf> = None;
    let mut emit_mjs: Option<PathBuf> = None;
    let mut emit_swift: Option<PathBuf> = None;
    let mut emit_kotlin: Option<PathBuf> = None;
    let mut emit_go: Option<PathBuf> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-o" if i + 1 < args.len() => {
                out_bin = args[i + 1].clone();
                i += 2;
            }
            "--emit-llvm" if i + 1 < args.len() => {
                opts.emit_ll_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--emit-mir" if i + 1 < args.len() => {
                emit_mir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--emit-msl" if i + 1 < args.len() => {
                emit_msl = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--emit-wgsl" if i + 1 < args.len() => {
                emit_wgsl = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--emit-webgpu" if i + 1 < args.len() => {
                emit_webgpu = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--emit-header" if i + 1 < args.len() => {
                opts.emit_header_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--emit-ts" if i + 1 < args.len() => {
                emit_ts = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--emit-mjs" if i + 1 < args.len() => {
                emit_mjs = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--emit-swift" if i + 1 < args.len() => {
                emit_swift = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--emit-kotlin" if i + 1 < args.len() => {
                emit_kotlin = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--emit-go" if i + 1 < args.len() => {
                emit_go = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--opt" if i + 1 < args.len() => {
                opts.opt_level = args[i + 1].clone();
                i += 2;
            }
            "--lib" if i + 1 < args.len() => {
                opts.output_kind = match args[i + 1].as_str() {
                    "static" => tenon_codegen::llvm_aot::OutputKind::StaticLib,
                    "shared" | "dylib" | "so" => tenon_codegen::llvm_aot::OutputKind::SharedLib,
                    other => {
                        eprintln!("unknown --lib kind `{}` (expected `static` or `shared`)", other);
                        return ExitCode::from(2);
                    }
                };
                i += 2;
            }
            "-g" | "--debug" => {
                opts.debug_info = true;
                i += 1;
            }
            "--checked-overflow" => {
                opts.checked_overflow = true;
                i += 1;
            }
            "--target" if i + 1 < args.len() => {
                opts.target_triple = Some(args[i + 1].clone());
                i += 2;
            }
            "-l" if i + 1 < args.len() => {
                opts.link_libs.push(args[i + 1].clone());
                i += 2;
            }
            "-L" if i + 1 < args.len() => {
                opts.link_dirs.push(args[i + 1].clone());
                i += 2;
            }
            "--c-src" if i + 1 < args.len() => {
                opts.extra_c_sources.push(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            s if s.starts_with("-l") && s.len() > 2 => {
                opts.link_libs.push(s[2..].to_string());
                i += 1;
            }
            s if s.starts_with("-L") && s.len() > 2 => {
                opts.link_dirs.push(s[2..].to_string());
                i += 1;
            }
            s if !s.starts_with('-') && file_arg.is_none() => {
                file_arg = Some(s.to_string());
                i += 1;
            }
            other => {
                eprintln!("unknown build argument `{}`", other);
                return ExitCode::from(2);
            }
        }
    }
    let Some(file) = file_arg else {
        eprintln!(
            "usage: tenon build <file.tn> [-o <out>] [--opt O3] [--lib static|shared] [--emit-header <h>] [--emit-llvm <ll>] [-g] [-l <lib>] [-L <dir>] [--c-src <c>]"
        );
        return ExitCode::from(2);
    };

    let prog = match load_and_parse(&file) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::from(1);
        }
    };
    let report = check_program(&prog);
    if !report.ok {
        eprintln!("FAIL-CLOSED: verification failed for `{}`; refusing AOT build.", file);
        for d in &report.diagnostics {
            eprintln!("  error[{}] line {}: {}", d.code, d.span.line, d.message);
            for n in &d.notes {
                eprintln!("    note: {}", n);
            }
        }
        return ExitCode::from(1);
    }

    if let Some(p) = &emit_ts {
        let _ = fs::write(p, tenon_codegen::bindings::emit_typescript_bindings(&prog));
        println!("Emitted TypeScript bindings -> `{}`", p.display());
    }
    if let Some(p) = &emit_mjs {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let _ = fs::write(p, tenon_codegen::bindings::emit_wasm_mjs_loader(&prog));
        println!("Emitted Wasm32 ESM Auto-Bridge -> `{}`", p.display());
    }
    if let Some(p) = &emit_swift {
        let _ = fs::write(p, tenon_codegen::bindings::emit_swift_bindings(&prog));
        println!("Emitted Swift C-ABI bindings -> `{}`", p.display());
    }
    if let Some(p) = &emit_kotlin {
        let _ = fs::write(p, tenon_codegen::bindings::emit_kotlin_bindings(&prog, "com.tenon.generated"));
        println!("Emitted Kotlin bindings -> `{}`", p.display());
    }
    if let Some(p) = &emit_go {
        let hdr = opts
            .emit_header_path
            .as_ref()
            .and_then(|h| h.file_name())
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "tenon.h".into());
        let lib_stem = Path::new(&out_bin)
            .file_stem()
            .map(|s| s.to_string_lossy().trim_start_matches("lib").to_string())
            .unwrap_or_else(|| "tenon".into());
        let _ = fs::write(
            p,
            tenon_codegen::bindings::emit_go_cgo_bindings(&prog, "main", &hdr, &lib_stem),
        );
        println!("Emitted Go cgo bindings -> `{}`", p.display());
    }
    opts.apply_verified_mir(&report.mir);
    let ssa_mir = tenon_mir::lower_program_to_ssa_mir(&prog, &report.mir);
    opts.apply_ssa_mir(&ssa_mir);
    if let Some(p) = &emit_mir {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let _ = fs::write(p, tenon_mir::format_mir(&ssa_mir));
        println!("Emitted Tenon SSA/CFG MIR -> `{}`", p.display());
    }
    if let Some(p) = &emit_msl {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let msl = tenon_codegen::gpu_shader::emit_msl(&prog, &report.mir);
        let _ = fs::write(p, &msl);
        println!("Emitted Apple Metal MSL ({} bytes) -> `{}`", msl.len(), p.display());
    }
    if let Some(p) = &emit_wgsl {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let wgsl = tenon_codegen::gpu_shader::emit_wgsl(&prog, &report.mir);
        let _ = fs::write(p, &wgsl);
        println!("Emitted WebGPU WGSL ({} bytes) -> `{}`", wgsl.len(), p.display());
    }
    if let Some(p) = &emit_webgpu {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = fs::create_dir_all(parent);
            }
        }
        let mjs = tenon_codegen::gpu_shader::emit_webgpu_host_mjs(&prog, &report.mir);
        let _ = fs::write(p, &mjs);
        println!("Emitted Browser WebGPU ESM Host ({} bytes) -> `{}`", mjs.len(), p.display());
    }
    if opts
        .target_triple
        .as_deref()
        .is_some_and(|t| t == "wasm32" || t == "wasm32-wasip2" || t == "wasi")
    {
        if tenon_codegen::wasm_llvm::has_llvm_wasm32_toolchain() {
            return match tenon_codegen::wasm_llvm::compile_program_to_wasm32_llvm(
                &prog,
                Path::new(&out_bin),
                &opts,
            ) {
                Ok(art) => {
                    println!(
                        "Wasm32 (Full LLVM + Perceus) Built `{}` -> `{}` ({} bytes, {} proofs verified)",
                        file,
                        art.output_path,
                        art.binary_bytes,
                        report.proofs.len()
                    );
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("Wasm32 LLVM build failed: {}", e);
                    ExitCode::from(1)
                }
            };
        }
        return match tenon_codegen::wasm_general::compile_program_to_wasm(&prog) {
            Ok(bytes) => {
                if let Some(parent) = Path::new(&out_bin).parent() {
                    let _ = fs::create_dir_all(parent);
                }
                if let Err(e) = fs::write(&out_bin, &bytes) {
                    eprintln!("failed to write `{}`: {}", out_bin, e);
                    return ExitCode::from(1);
                }
                println!("Wasm32 Built `{}` -> `{}` ({} bytes, {} proofs verified)", file, out_bin, bytes.len(), report.proofs.len());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("Wasm32 build failed: {}", e);
                ExitCode::from(1)
            }
        };
    }

    match tenon_codegen::llvm_aot::compile_with_options(&prog, Path::new(&out_bin), &opts) {
        Ok(art) => {
            let kind_str = match opts.output_kind {
                tenon_codegen::llvm_aot::OutputKind::Executable => "native binary",
                tenon_codegen::llvm_aot::OutputKind::StaticLib => "static library",
                tenon_codegen::llvm_aot::OutputKind::SharedLib => "shared library",
            };
            println!(
                "LLVM AOT Built `{}` -> `{}` ({} bytes {}, {} bytes LLVM IR, opt={}, debug={}, {} proofs verified, {} bounds checks eliminated, {} overflow-free ops)",
                file,
                art.output_path,
                art.binary_bytes,
                kind_str,
                art.llvm_ir_bytes,
                art.opt_level,
                opts.debug_info,
                report.proofs.len(),
                report.mir.proved_safe_indices.len(),
                report.mir.proved_no_overflow.len()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("LLVM AOT build failed: {}", e);
            ExitCode::from(1)
        }
    }
}

fn cmd_verify_proofs(args: &[String]) -> ExitCode {
    let mut dir_arg: Option<&str> = None;
    let mut solver = "auto";
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--solver" if i + 1 < args.len() => {
                solver = args[i + 1].as_str();
                i += 2;
            }
            s if !s.starts_with('-') && dir_arg.is_none() => {
                dir_arg = Some(s);
                i += 1;
            }
            _ => i += 1,
        }
    }
    let Some(dir) = dir_arg else {
        eprintln!("usage: tenon verify-proofs <proof_dir> [--solver z3|cvc5|auto]");
        return ExitCode::from(2);
    };
    match tenon_verify::verify_proof_certificates_dir(Path::new(dir), solver) {
        Ok(rep) => {
            println!(
                "Tenon Independent Proof Auditor (`{}` via `{}`): {} ({}/{} certificates proved unsat)",
                dir,
                rep.solver,
                if rep.ok { "PASS" } else { "FAIL" },
                rep.verified,
                rep.total
            );
            for f in &rep.failed {
                eprintln!("  FAILED: {}", f);
            }
            if rep.ok { ExitCode::SUCCESS } else { ExitCode::from(1) }
        }
        Err(e) => {
            eprintln!("error: {}", e);
            ExitCode::from(1)
        }
    }
}

fn cmd_pkg(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("usage: tenon pkg <lock|verify|tree|add> [dir]");
        return ExitCode::from(2);
    }
    match args[0].as_str() {
        "lock" => {
            let dir = PathBuf::from(args.get(1).map(|s| s.as_str()).unwrap_or("."));
            match tenon_syntax::generate_lockfile(&dir) {
                Ok((content, count)) => {
                    let lock_path = dir.join("tenon.lock");
                    if let Err(e) = fs::write(&lock_path, content) {
                        eprintln!("failed to write `{}`: {}", lock_path.display(), e);
                        return ExitCode::from(1);
                    }
                    println!("Locked {} package(s) -> `{}`", count, lock_path.display());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {}", e);
                    ExitCode::from(1)
                }
            }
        }
        "verify" => {
            let dir = PathBuf::from(args.get(1).map(|s| s.as_str()).unwrap_or("."));
            match tenon_syntax::verify_lockfile(&dir) {
                Ok(count) => {
                    println!("PASS: verified {} locked package(s) in `{}`", count, dir.display());
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("{}", e);
                    ExitCode::from(1)
                }
            }
        }
        "tree" => {
            let dir = PathBuf::from(args.get(1).map(|s| s.as_str()).unwrap_or("."));
            match tenon_syntax::format_dep_tree(&dir) {
                Ok(tree) => {
                    print!("{}", tree);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {}", e);
                    ExitCode::from(1)
                }
            }
        }
        "add" => {
            if args.len() < 2 {
                eprintln!("usage: tenon pkg add <name> --path <dir> [--version <v>] [--git <url>] [--dir <pkg_dir>]");
                return ExitCode::from(2);
            }
            let dep_name = &args[1];
            let mut dep_path = String::new();
            let mut dep_git: Option<String> = None;
            let mut dep_ver: Option<String> = None;
            let mut pkg_dir = PathBuf::from(".");
            let mut i = 2usize;
            while i < args.len() {
                match args[i].as_str() {
                    "--path" if i + 1 < args.len() => {
                        dep_path = args[i + 1].clone();
                        i += 2;
                    }
                    "--git" if i + 1 < args.len() => {
                        dep_git = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--version" if i + 1 < args.len() => {
                        dep_ver = Some(args[i + 1].clone());
                        i += 2;
                    }
                    "--dir" if i + 1 < args.len() => {
                        pkg_dir = PathBuf::from(&args[i + 1]);
                        i += 2;
                    }
                    _ => i += 1,
                }
            }
            let toml_path = pkg_dir.join("tenon.toml");
            let mut existing = fs::read_to_string(&toml_path).unwrap_or_default();
            let mut block = format!("\n[dependencies.{}]\n", dep_name);
            if !dep_path.is_empty() {
                block.push_str(&format!("path = \"{}\"\n", dep_path));
            }
            if let Some(g) = dep_git {
                block.push_str(&format!("git = \"{}\"\n", g));
            }
            if let Some(v) = dep_ver {
                block.push_str(&format!("version = \"{}\"\n", v));
            }
            existing.push_str(&block);
            if let Err(e) = fs::write(&toml_path, existing) {
                eprintln!("failed to update `{}`: {}", toml_path.display(), e);
                return ExitCode::from(1);
            }
            println!("Added dependency `{}` to `{}`", dep_name, toml_path.display());
            ExitCode::SUCCESS
        }
        "sync" => {
            let dir = PathBuf::from(args.get(1).map(|s| s.as_str()).unwrap_or("."));
            match tenon_syntax::sync_git_dependencies(&dir) {
                Ok(synced) => {
                    println!("Synced {} remote git dependency(ies) in `{}`", synced.len(), dir.display());
                    for (name, p) in synced {
                        println!("  {} -> {}", name, p.display());
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("error: {}", e);
                    ExitCode::from(1)
                }
            }
        }
        other => {
            eprintln!("unknown `tenon pkg` command `{}` (expected lock|verify|tree|add|sync)", other);
            ExitCode::from(2)
        }
    }
}

fn cmd_new(args: &[String]) -> ExitCode {
    let mut name_opt: Option<&str> = None;
    let mut template = "cli";
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--template" | "-t" if i + 1 < args.len() => {
                template = args[i + 1].as_str();
                i += 2;
            }
            "--web" => {
                template = "web";
                i += 1;
            }
            "--lib" => {
                template = "lib";
                i += 1;
            }
            "--cli" => {
                template = "cli";
                i += 1;
            }
            "--gpui" => {
                template = "gpui";
                i += 1;
            }
            "--openui-web" => {
                template = "openui-web";
                i += 1;
            }
            "--tui" => {
                template = "tui";
                i += 1;
            }
            "--gpu" => {
                template = "gpu";
                i += 1;
            }
            s if !s.starts_with('-') && name_opt.is_none() => {
                name_opt = Some(s);
                i += 1;
            }
            _ => i += 1,
        }
    }
    let Some(name) = name_opt else {
        eprintln!("usage: tenon new <project_dir> [--template cli|lib|web|gpui|openui-web|tui|gpu]");
        return ExitCode::from(2);
    };
    let dir = PathBuf::from(name);
    let pkg_name = dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("tenon_app");
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("failed to create `{}`: {}", dir.display(), e);
        return ExitCode::from(1);
    }
    let manifest = format!(
        "[package]\nname = \"{}\"\nversion = \"0.1.0\"\n",
        pkg_name
    );
    let _ = fs::write(dir.join("tenon.toml"), manifest);
    match template {
        "lib" => {
            let lib_src = format!(
                "// {} — Verified Tenon Library\n\npub type Positive = Int where self > 0\n\npub pure fn add_pos(a Positive, b Positive) Positive {{\n    a + b\n}}\n",
                pkg_name
            );
            let test_src = "test fn test_add_pos() {\n    assert add_pos(2, 3) == 5\n}\n";
            let _ = fs::write(dir.join("lib.tn"), lib_src);
            let _ = fs::write(dir.join("lib_test.tn"), test_src);
        }
        "web" => {
            let web_src = format!(
                "// {} — Tenon Web Service\nimport \"std/web/app.tn\"\n\n@get(\"/health\")\npub pure fn health(ctx Context) HttpResponse {{\n    ctx.ok(\"ok\")\n}}\n\ngen!(mount_annotated_routes())\n\npub fn main() Int {{\n    let app = mount_routes(web_app_new())\n    let h = app.into_handler()\n    let req = HttpRequest {{ method = \"GET\", target = \"/health\", path = \"/health\", query = \"\", headers = [], body = \"\", keep_alive = true }}\n    h(req).status\n}}\n",
                pkg_name
            );
            let test_src = "test fn test_health() {\n    assert main() == 200\n}\n";
            let _ = fs::write(dir.join("main.tn"), web_src);
            let _ = fs::write(dir.join("main_test.tn"), test_src);
        }
        "gpui" => {
            let gpui_src = format!(
                "// {} — Native Apple Silicon Metal Desktop App (`tenon-gpui`)\n// Build & run:\n//   tenon build --c-src std/gpu/window/metal_window.m -o target/{} main.tn && ./target/{}\nimport \"std/core/types.tn\"\nimport \"std/ui/vdom.tn\"\nimport \"std/ui/openui.tn\"\nimport \"std/ui/layout.tn\"\nimport \"std/ui/scene.tn\"\nimport \"std/gpu/window/window.tn\"\n\npure fn push_q(inout buf List[Float32], q GpuQuad) {{\n    buf = push(buf, q.x)\n    buf = push(buf, q.y)\n    buf = push(buf, q.w)\n    buf = push(buf, q.h)\n    buf = push(buf, q.r)\n    buf = push(buf, q.g)\n    buf = push(buf, q.b)\n    buf = push(buf, q.a)\n    buf = push(buf, q.radius)\n    buf = push(buf, q.border_w)\n    buf = push(buf, q.br)\n    buf = push(buf, q.bg)\n}}\n\npub fn main() Int {{\n    let src = \"root = Card([hdr, row])\\nhdr = Header(\\\"{}\\\", \\\"Native Metal GPUI + R8 Glyph Atlas\\\")\\nrow = Row([m1, m2])\\nm1 = Metric(\\\"FPS\\\", \\\"120Hz\\\", \\\"VSync\\\")\\nm2 = Metric(\\\"Leaks\\\", \\\"0\\\", \\\"Perceus\\\")\\n\"\n    let doc = openui_parse_stream(src, true)\n    let vdom = openui_to_vdom(doc, ui_store_new())\n    let boxes = layout_vdom(vdom, 760.0, 440.0)\n    let scene = layout_to_scene(boxes, 760, 440)\n    let smoke = sys_env(\"TENON_GUI_SMOKE\") == \"1\"\n    if !win_open(\"{}\", 760, 440, smoke) {{ return 1 }}\n    var qf List[Float32] = []\n    var i = 0\n    while i < len(scene.quads) {{\n        push_q(qf, scene.quads[i])\n        i += 1\n    }}\n    win_begin_text()\n    win_draw_text(\"{} — Ready on \" ++ win_gpu_name(), 32, 24, 18, true, 248, 250, 252)\n    if smoke {{\n        let _ = win_save_png(qf, len(scene.quads), 11, 17, 32, \"/tmp/{}_preview.png\")\n        return 0\n    }}\n    let _ = win_present(qf, len(scene.quads), 11, 17, 32)\n    var running = true\n    while running {{\n        let ev = win_poll_event(16)\n        if ev.kind == 3 || (ev.kind == 1 && (ev.a == 27 || ev.a == 113)) {{ running = false }}\n    }}\n    0\n}}\n",
                pkg_name, pkg_name, pkg_name, pkg_name, pkg_name, pkg_name, pkg_name
            );
            let _ = fs::write(dir.join("main.tn"), gpui_src);
        }
        "openui-web" => {
            let wasm_src = format!(
                "// {} — OpenUI + Wasm32 Surgical VDOM Kernel\n// Build:\n//   tenon build --target wasm32 --emit-mjs dist/app.mjs -o dist/app.wasm main.tn\nimport \"std/core/types.tn\"\nimport \"std/ui/vdom.tn\"\nimport \"std/ui/openui.tn\"\n\n@export\npub pure fn render_openui_html(src Text, value Text) Text {{\n    let doc = openui_parse_stream(src, true)\n    let store = ui_store_set(ui_store_new(), \"val\", value)\n    vnode_to_html(openui_to_vdom(doc, store))\n}}\n\n@export\npub pure fn diff_openui_json(src Text, old_val Text, new_val Text) Text {{\n    let doc = openui_parse_stream(src, true)\n    let t1 = openui_to_vdom(doc, ui_store_set(ui_store_new(), \"val\", old_val))\n    let t2 = openui_to_vdom(doc, ui_store_set(ui_store_new(), \"val\", new_val))\n    patches_to_json(diff_vdom(t1, t2))\n}}\n\npub fn main() Int {{ 0 }}\n",
                pkg_name
            );
            let html_src = format!(
                "<!doctype html>\n<html><head><meta charset=\"utf-8\"/><title>{} — Tenon OpenUI Web</title></head>\n<body style=\"background:#0b1120;color:#e2e8f0;font-family:system-ui;padding:24px\">\n<h1>{} (Tenon Wasm32 OpenUI)</h1>\n<div id=\"app\">Build with <code>tenon build --target wasm32 --emit-mjs dist/app.mjs -o dist/app.wasm main.tn</code></div>\n<script type=\"module\">\nimport {{ loadTenonWasm }} from './dist/app.mjs';\nconst api = await loadTenonWasm(await (await fetch('./dist/app.wasm')).arrayBuffer());\nconst src = 'root = Card([hdr, m1])\\nhdr = Header(\"{}\", \"Wasm32 ESM Bridge\")\\nm1 = Metric(\"Live\", $val, \"+100%\")\\n';\ndocument.getElementById('app').innerHTML = api.render_openui_html(src, '42');\n</script>\n</body></html>\n",
                pkg_name, pkg_name, pkg_name
            );
            let _ = fs::write(dir.join("main.tn"), wasm_src);
            let _ = fs::write(dir.join("index.html"), html_src);
        }
        "tui" => {
            let tui_src = format!(
                "// {} — Verified Double-Buffered ANSI TUI App\nimport \"std/core/types.tn\"\nimport \"std/ui/vdom.tn\"\nimport \"std/ui/openui.tn\"\nimport \"std/ui/tui.tn\"\n\npub fn main() Int {{\n    let src = \"root = Card([hdr, row])\\nhdr = Header(\\\"{}\\\", \\\"OpenUI -> VDOM -> Double-Buffered TUI\\\")\\nrow = Row([m1, m2])\\nm1 = Metric(\\\"Status\\\", $status, \\\"OK\\\")\\nm2 = Metric(\\\"Leaks\\\", \\\"0\\\", \\\"Perceus\\\")\\n\"\n    let doc = openui_parse_stream(src, true)\n    let b1 = render_vdom_to_tui(openui_to_vdom(doc, ui_store_set(ui_store_new(), \"status\", \"BOOT\")), 64, 14)\n    let b2 = render_vdom_to_tui(openui_to_vdom(doc, ui_store_set(ui_store_new(), \"status\", \"LIVE\")), 64, 14)\n    let diff = tui_diff_ansi(b1, b2)\n    println(tui_buffer_to_plain_text(b2))\n    println(\"ANSI minimal diff bytes = \" ++ int_to_text(len(diff)))\n    0\n}}\n",
                pkg_name, pkg_name
            );
            let test_src = "test fn test_tui_main() {\n    assert main() == 0\n}\n";
            let _ = fs::write(dir.join("main.tn"), tui_src);
            let _ = fs::write(dir.join("main_test.tn"), test_src);
        }
        "gpu" => {
            let gpu_src = format!(
                "// {} — Single-Source Declarative GPU Compute App (`@kernel` + `@gpu_graph`)\n// Emit MSL + WGSL + Browser WebGPU ESM:\n//   tenon build --emit-msl dist/kernels.metal --emit-wgsl dist/kernels.wgsl --emit-webgpu dist/gpu.mjs -o target/{} main.tn\n\n@gpu_struct\npub record ScaleParams {{\n    n    UInt32\n    gain Float32\n}}\n\n@kernel(\"elementwise\")\npub pure fn scale_bias(x List[Float32], bias List[Float32], inout out List[Float32], total UInt32) {{\n    let i = gpu_global_id_x() as Int\n    if i < (total as Int) {{\n        out[i] = x[i] + bias[i]\n    }}\n}}\n\n@kernel(\"elementwise\")\npub pure fn relu_clamp(x List[Float32], inout out List[Float32], total UInt32) {{\n    let i = gpu_global_id_x() as Int\n    if i < (total as Int) {{\n        let v = x[i]\n        out[i] = if v > (0.0 as Float32) {{ v }} else {{ 0.0 as Float32 }}\n    }}\n}}\n\n@gpu_graph\npub pure fn fused_pipeline(x List[Float32], bias List[Float32], total UInt32) List[Float32] {{\n    var biased List[Float32] = list_filled(16, 0.0 as Float32)\n    var out List[Float32] = list_filled(16, 0.0 as Float32)\n    scale_bias(x, bias, biased, total)\n    relu_clamp(biased, out, total)\n    out\n}}\n\npub fn main() Int {{\n    let _ = sys_write_stdout(\"{}: verified @kernel (E0842/E0843) & fused @gpu_graph ready for MSL/WGSL/WebGPU!\\n\")\n    0\n}}\n",
                pkg_name, pkg_name, pkg_name
            );
            let test_src = "test fn test_gpu_main() {\n    assert main() == 0\n}\n";
            let _ = fs::write(dir.join("main.tn"), gpu_src);
            let _ = fs::write(dir.join("main_test.tn"), test_src);
        }
        "cli" => {
            let cli_src = format!(
                "// {} — Verified Tenon CLI Application\n\npub type Range[LO Int, HI Int] = Int where self >= LO && self <= HI\n\npub pure fn clamp_score(x Range[0, 100]) Range[0, 200] {{\n    x * 2\n}}\n\npub fn main() Int {{\n    clamp_score(21)\n}}\n",
                pkg_name
            );
            let test_src = "test fn test_clamp_score() {\n    assert clamp_score(21) == 42\n}\n";
            let _ = fs::write(dir.join("main.tn"), cli_src);
            let _ = fs::write(dir.join("main_test.tn"), test_src);
        }
        other => {
            eprintln!("unknown template `{}` (expected cli|lib|web|gpui|openui-web|tui|gpu)", other);
            return ExitCode::from(2);
        }
    }
    println!("Created Tenon `{}` project -> `{}`", template, dir.display());
    ExitCode::SUCCESS
}

fn cmd_doc(args: &[String]) -> ExitCode {
    let mut target_opt: Option<&str> = None;
    let mut out_dir = PathBuf::from("target/tenon-doc");
    let mut format = "both";
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-o" if i + 1 < args.len() => {
                out_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--format" if i + 1 < args.len() => {
                format = args[i + 1].as_str();
                i += 2;
            }
            s if !s.starts_with('-') && target_opt.is_none() => {
                target_opt = Some(s);
                i += 1;
            }
            _ => i += 1,
        }
    }
    let Some(target) = target_opt else {
        eprintln!("usage: tenon doc <file.tn|pkg_dir> [-o <out_dir>] [--format md|html|both]");
        return ExitCode::from(2);
    };
    let prog = match load_and_parse(target) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::from(1);
        }
    };
    if let Err(e) = fs::create_dir_all(&out_dir) {
        eprintln!("failed to create `{}`: {}", out_dir.display(), e);
        return ExitCode::from(1);
    }

    let target_canon = fs::canonicalize(target)
        .unwrap_or_else(|_| PathBuf::from(target))
        .to_string_lossy()
        .to_string();
    let target_fid: Option<u32> = prog
        .source_files
        .iter()
        .position(|sf| sf == &target_canon)
        .map(|idx| idx as u32);

    let mut md = format!("# Tenon API Reference — `{}`\n\n", target);
    let mut html_items = String::new();
    let mut count = 0usize;
    for item in &prog.items {
        let fid = match item {
            Item::TypeAlias(d) => d.span.file_id,
            Item::Record(d) => d.span.file_id,
            Item::Enum(d) => d.span.file_id,
            Item::Interface(d) => d.span.file_id,
            Item::Fn(d) => d.span.file_id,
            _ => 0,
        };
        if let Some(tfid) = target_fid {
            if fid != tfid {
                continue;
            }
        }
        match item {
            Item::TypeAlias(ta) if ta.is_pub => {
                count += 1;
                let sig = format!("pub type {} = {}", ta.name, ta.base.display_name());
                md.push_str(&format!("### `{}`\n\n", sig));
                html_items.push_str(&format!("<section class='item'><h3><code>{}</code></h3><span class='badge'>type</span></section>\n", sig));
            }
            Item::Record(r) if r.is_pub => {
                count += 1;
                md.push_str(&format!("### `pub record {}`\n", r.name));
                let mut fields_html = String::new();
                for fl in &r.fields {
                    md.push_str(&format!("- `{}`: `{}`\n", fl.name, fl.ty.display_name()));
                    fields_html.push_str(&format!("<li><code>{}</code>: <code>{}</code></li>", fl.name, fl.ty.display_name()));
                }
                md.push('\n');
                html_items.push_str(&format!("<section class='item'><h3><code>pub record {}</code></h3><span class='badge'>record</span><ul>{}</ul></section>\n", r.name, fields_html));
            }
            Item::Enum(e) if e.is_pub => {
                count += 1;
                let vars: Vec<String> = e.variants.iter().map(|v| v.name.clone()).collect();
                md.push_str(&format!("### `pub enum {}`\nVariants: `{}`\n\n", e.name, vars.join(" | ")));
                html_items.push_str(&format!("<section class='item'><h3><code>pub enum {}</code></h3><span class='badge'>enum</span><p>Variants: <code>{}</code></p></section>\n", e.name, vars.join(" | ")));
            }
            Item::Interface(iface) if iface.is_pub => {
                count += 1;
                md.push_str(&format!("### `pub interface {}`\n\n", iface.name));
                html_items.push_str(&format!("<section class='item'><h3><code>pub interface {}</code></h3><span class='badge'>interface</span></section>\n", iface.name));
            }
            Item::Fn(f) if f.is_pub && !f.is_test => {
                count += 1;
                let ps: Vec<String> = f.params.iter().map(|p| format!("{} {}", p.name, p.ty.display_name())).collect();
                let rt = f.ret_ty.as_ref().map(|t| format!(" {}", t.display_name())).unwrap_or_default();
                let effs = f.declared_effects();
                let eff_str = if effs.is_empty() { String::new() } else { format!(" / {}", effs.join(", ")) };
                let prefix = if f.is_pure { "pub pure fn" } else if f.is_async { "pub async fn" } else { "pub fn" };
                let sig = format!("{} {}({}){}{}", prefix, f.name, ps.join(", "), rt, eff_str);
                md.push_str(&format!("### `{}`\n\n", sig));
                html_items.push_str(&format!("<section class='item'><h3><code>{}</code></h3><span class='badge'>fn</span></section>\n", sig));
            }
            _ => {}
        }
    }
    if format == "md" || format == "both" {
        let _ = fs::write(out_dir.join("API.md"), &md);
    }
    if format == "html" || format == "both" {
        let html = format!(
            "<!doctype html><html><head><meta charset='utf-8'><title>Tenon Docs: {}</title><style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;background:#0f172a;color:#e2e8f0}}h1{{color:#38bdf8}}.item{{background:#1e293b;border-radius:10px;padding:1rem 1.2rem;margin:1rem 0;border:1px solid #334155}}.badge{{display:inline-block;background:#0284c7;color:#fff;font-size:.75rem;padding:2px 8px;border-radius:999px;margin-bottom:.4rem}}code{{color:#f8fafc}}</style></head><body><h1>Tenon API Reference: {}</h1>{}</body></html>",
            target, target, html_items
        );
        let _ = fs::write(out_dir.join("index.html"), html);
    }
    println!("Generated Tenon API docs ({} public item(s)) -> `{}`", count, out_dir.display());
    ExitCode::SUCCESS
}

fn cmd_bench(args: &[String]) -> ExitCode {
    let mut file_opt: Option<&str> = None;
    let mut iters = 5usize;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" if i + 1 < args.len() => {
                iters = args[i + 1].parse().unwrap_or(5).max(1);
                i += 2;
            }
            s if !s.starts_with('-') && file_opt.is_none() => {
                file_opt = Some(s);
                i += 1;
            }
            _ => i += 1,
        }
    }
    let Some(file) = file_opt else {
        eprintln!("usage: tenon bench <file.tn> [--iters <N>]");
        return ExitCode::from(2);
    };
    let prog = match load_and_parse(file) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::from(1);
        }
    };
    let report = check_program(&prog);
    if !report.ok {
        eprintln!("FAIL-CLOSED: verification failed for `{}`", file);
        return ExitCode::from(1);
    }
    let bin = std::env::temp_dir().join(format!("tenon_bench_{}", std::process::id()));
    let mut opts = tenon_codegen::llvm_aot::BuildOptions::default();
    opts.apply_verified_mir(&report.mir);
    let ssa_mir = tenon_mir::lower_program_to_ssa_mir(&prog, &report.mir);
    opts.apply_ssa_mir(&ssa_mir);
    let art = match tenon_codegen::llvm_aot::compile_with_options(&prog, &bin, &opts) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("benchmark build failed: {}", e);
            return ExitCode::from(1);
        }
    };
    // 1 warmup + `iters` measured runs
    let _ = std::process::Command::new(&bin).output();
    let mut samples_ms = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = std::time::Instant::now();
        let out = std::process::Command::new(&bin).output();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        match out {
            Ok(o) if o.status.success() => samples_ms.push(ms),
            Ok(o) => {
                eprintln!("benchmark exited with {:?}: {}", o.status, String::from_utf8_lossy(&o.stderr));
                let _ = fs::remove_file(&bin);
                return ExitCode::from(1);
            }
            Err(e) => {
                eprintln!("benchmark failed to execute: {}", e);
                let _ = fs::remove_file(&bin);
                return ExitCode::from(1);
            }
        }
    }
    let _ = fs::remove_file(&bin);
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = samples_ms[samples_ms.len() / 2];
    let min_ms = samples_ms[0];
    let max_ms = *samples_ms.last().unwrap();
    println!(
        "BENCH `{}` ({} iters, LLVM -O3, {:.1} KB, {} proofs): median={:.2}ms min={:.2}ms max={:.2}ms",
        file,
        iters,
        art.binary_bytes as f64 / 1024.0,
        report.proofs.len(),
        median,
        min_ms,
        max_ms
    );
    ExitCode::SUCCESS
}

fn format_tenon_source(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut indent: usize = 0;
    let mut prev_blank = false;

    for raw_line in src.lines() {
        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            if !out.is_empty() && !prev_blank {
                out.push('\n');
                prev_blank = true;
            }
            continue;
        }
        prev_blank = false;

        // Count leading closing braces on this line to dedent before printing
        let mut leading_closes = 0usize;
        for ch in trimmed.chars() {
            if ch == '}' {
                leading_closes += 1;
            } else {
                break;
            }
        }
        let line_indent = indent.saturating_sub(leading_closes);
        for _ in 0..line_indent {
            out.push_str("    ");
        }
        out.push_str(trimmed);
        out.push('\n');

        // Scan net `{` / `}` delta outside string literals and `//` comments
        let mut opens = 0usize;
        let mut closes = 0usize;
        let bytes = trimmed.as_bytes();
        let mut idx = 0usize;
        let mut in_str = false;
        while idx < bytes.len() {
            let b = bytes[idx];
            if in_str {
                if b == b'\\' && idx + 1 < bytes.len() {
                    idx += 2;
                    continue;
                }
                if b == b'"' {
                    in_str = false;
                }
            } else {
                if b == b'/' && idx + 1 < bytes.len() && bytes[idx + 1] == b'/' {
                    break;
                }
                if b == b'"' {
                    in_str = true;
                } else if b == b'{' {
                    opens += 1;
                } else if b == b'}' {
                    closes += 1;
                }
            }
            idx += 1;
        }
        indent = indent.saturating_add(opens).saturating_sub(closes);
    }
    while out.ends_with("\n\n") {
        out.pop();
    }
    out
}

fn cmd_fmt(args: &[String]) -> ExitCode {
    let check_only = args.iter().any(|a| a == "--check");
    let files: Vec<&String> = args.iter().filter(|a| !a.starts_with('-')).collect();
    if files.is_empty() {
        eprintln!("usage: tenon fmt <file.tn> [--check]");
        return ExitCode::from(2);
    }
    let mut dirty = false;
    for file in files {
        let src = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error reading `{}`: {}", file, e);
                return ExitCode::from(1);
            }
        };
        if let Err(e) = tenon_syntax::parse(&src) {
            eprintln!("syntax error in `{}`: {}", file, e);
            return ExitCode::from(1);
        }
        let formatted = format_tenon_source(&src);
        if formatted != src {
            if check_only {
                eprintln!("would reformat `{}`", file);
                dirty = true;
            } else {
                if let Err(e) = fs::write(file, &formatted) {
                    eprintln!("error writing `{}`: {}", file, e);
                    return ExitCode::from(1);
                }
                println!("Formatted `{}`", file);
            }
        }
    }
    if dirty {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

async fn cmd_agent(args: &[String]) -> ExitCode {
    let model_arg = args
        .windows(2)
        .find(|w| w[0] == "--model" || w[0] == "-m")
        .map(|w| w[1].as_str())
        .unwrap_or("gpt-6-luna");
    let effort_arg = args
        .windows(2)
        .find(|w| w[0] == "--effort")
        .map(|w| w[1].as_str());
    let spec_file = args
        .windows(2)
        .find(|w| w[0] == "--spec")
        .map(|w| w[1].as_str())
        .unwrap_or("examples/dast_os_kernel/kernel_v2.tn");
    let db_path = args
        .windows(2)
        .find(|w| w[0] == "--db")
        .map(|w| w[1].as_str())
        .unwrap_or(":memory:");
    let session_id = args
        .windows(2)
        .find(|w| w[0] == "--session")
        .map(|w| w[1].as_str())
        .unwrap_or("main");
    let env_file = args
        .windows(2)
        .find(|w| w[0] == "--env-file")
        .map(|w| Path::new(w[1].as_str()));
    let workspace = args
        .windows(2)
        .find(|w| w[0] == "--workspace" || w[0] == "-C")
        .map(|w| PathBuf::from(&w[1]))
        .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let allow_outside = args.iter().any(|a| a == "--allow-outside-workspace");
    let single_prompt = args
        .windows(2)
        .find(|w| w[0] == "--prompt" || w[0] == "-p")
        .map(|w| w[1].clone());

    // 1. Load & verify `.tn` distributed kernel manifest (or fall back to default verified manifest if run outside repo)
    let manifest = if Path::new(spec_file).exists() {
        let prog = match load_and_parse(spec_file) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("error loading spec `{}`: {}", spec_file, e);
                return ExitCode::from(1);
            }
        };
        let rep = check_program(&prog);
        if !rep.ok {
            eprintln!("FAIL-CLOSED: spec `{}` failed verification.", spec_file);
            return ExitCode::from(1);
        }
        rep.manifest.unwrap_or_else(default_agent_manifest)
    } else {
        default_agent_manifest()
    };

    // 2. Load OpenAI Responses API config (dast-os models: gpt-6-luna / gpt-6.1-sol / gpt-6.1-terra / gpt-6-astra)
    let llm_cfg = match OpenAIResponsesConfig::load(model_arg, env_file, effort_arg) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("OpenAI configuration error: {}", e);
            eprintln!("Hint: export OPENAI_API_KEY=... or pass --env-file /path/to/.env");
            return ExitCode::from(1);
        }
    };

    let store = match AuthorityStore::open(Path::new(db_path)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to open SQLite authority `{}`: {}", db_path, e);
            return ExitCode::from(1);
        }
    };
    let kernel = Arc::new(UniversalSyncKernel::new(store, manifest));
    let toolkit = AgentToolkit::new(workspace, allow_outside);
    let mut agent = TenonAgentRuntime::new(kernel, llm_cfg, toolkit, session_id);

    // Single-shot mode (`-p "..."`)
    if let Some(prompt) = single_prompt {
        println!(
            "Tenon Agent [model={} effort={} workspace={}]",
            agent.llm_cfg.model,
            agent.llm_cfg.reasoning_effort,
            agent.toolkit.workspace_root.display()
        );
        match agent.run_turn(&prompt, true).await {
            Ok(rep) => {
                let trace_ok = verify_runtime_trace(&agent.kernel.snapshot_trace()).ok;
                println!(
                    "[done] branch=#{} epoch={} rounds={} tools={} tokens(in={} cached={} out={}) cost=${:.6} trace_refinement={}",
                    rep.branch_id,
                    rep.exec_epoch,
                    rep.rounds,
                    rep.tool_calls_executed.len(),
                    rep.usage.input_tokens,
                    rep.usage.cached_tokens,
                    rep.usage.output_tokens,
                    rep.usage.estimated_cost_usd,
                    trace_ok
                );
                return ExitCode::SUCCESS;
            }
            Err(e) => {
                eprintln!("agent error: {}", e);
                return ExitCode::from(1);
            }
        }
    }

    // Interactive CLI REPL Mode!
    println!("==================================================================");
    println!("  Tenon Interactive CLI Agent (dast-os OpenAI Responses API)");
    println!("  Model:     {} (reasoning={})", agent.llm_cfg.model, agent.llm_cfg.reasoning_effort);
    println!("  Workspace: {}", agent.toolkit.workspace_root.display());
    println!("  Tools:     shell, read_file, write_file, edit_file, list_dir, grep_search");
    println!("  Authority: SQLite WAL ({}) | Hard exec_epoch Fencing | Turn DAG", db_path);
    println!("  Commands:  /model <luna|sol|terra|astra>  /state  /trace  /clear  /help  /exit");
    println!("==================================================================");

    let stdin = io::stdin();
    let mut reader = stdin.lock();
    loop {
        print!("\ntenon[{}]> ", agent.llm_cfg.model);
        let _ = io::stdout().flush();

        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                println!("\nBye.");
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("stdin error: {}", e);
                break;
            }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if matches!(trimmed, "/exit" | "/quit" | "exit" | "quit") {
            println!("Bye.");
            break;
        }
        if trimmed == "/help" {
            println!("Commands:");
            println!("  /model <luna|terra|sol|astra|<id>>  Switch OpenAI model (gpt-6-luna / gpt-6.1-terra / gpt-6.1-sol / gpt-6-astra)");
            println!("  /state                              Inspect SQLite Turn DAG branch, epoch, history & receipts");
            println!("  /trace                              Verify live Runtime Trace Refinement invariants");
            println!("  /steer <msg>                        Enqueue P0 steering instruction for next turn");
            println!("  /clear                              Reset conversation context");
            println!("  /exit                               Quit interactive agent");
            continue;
        }
        if let Some(m) = trimmed.strip_prefix("/model ") {
            agent.switch_model(m.trim());
            println!(
                "Switched model -> {} (reasoning={})",
                agent.llm_cfg.model, agent.llm_cfg.reasoning_effort
            );
            continue;
        }
        if trimmed == "/state" || trimmed == "/branch" {
            match agent.kernel.authority.read_session_view(&agent.session_id) {
                Ok(v) => println!(
                    "session={} branch=#{} epoch={} phase={} history_turns={} audit_receipts={} journal_head={}",
                    v.session_id,
                    v.active_branch,
                    v.exec_epoch,
                    v.phase,
                    v.history.len(),
                    v.late_audit_receipts,
                    v.journal_head_seq
                ),
                Err(e) => eprintln!("state error: {}", e),
            }
            continue;
        }
        if trimmed == "/trace" {
            let tr = verify_runtime_trace(&agent.kernel.snapshot_trace());
            println!("{}", serde_json::to_string_pretty(&tr).unwrap());
            continue;
        }
        if let Some(st) = trimmed.strip_prefix("/steer ") {
            let cid = format!("steer_{}", agent.conversation_items.len());
            match agent
                .kernel
                .authority
                .enqueue_steer(&agent.session_id, &cid, st.trim())
            {
                Ok(()) => println!("Enqueued P0 steering instruction."),
                Err(e) => eprintln!("steer error: {}", e),
            }
            continue;
        }
        if trimmed == "/clear" {
            agent.reset_conversation();
            println!("Cleared conversation context.");
            continue;
        }

        // Run live fenced agent turn with Ctrl-C cancellation support!
        let run_fut = agent.run_turn(trimmed, true);
        tokio::select! {
            res = run_fut => {
                match res {
                    Ok(rep) => {
                        println!(
                            "\n[branch=#{} epoch={} rounds={} tools={} in={} cached={} out={} cost=${:.6}]",
                            rep.branch_id,
                            rep.exec_epoch,
                            rep.rounds,
                            rep.tool_calls_executed.len(),
                            rep.usage.input_tokens,
                            rep.usage.cached_tokens,
                            rep.usage.output_tokens,
                            rep.usage.estimated_cost_usd
                        );
                    }
                    Err(e) => eprintln!("\n[agent error] {}", e),
                }
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\n[interrupted] Ctrl-C received; fencing active epoch and opening clean branch...");
                let cid = format!("ctrlc_{}", std::process::id());
                let _ = agent.kernel.authority.interrupt(&agent.session_id, &cid, "(interrupted by Ctrl-C)");
            }
        }
    }
    ExitCode::SUCCESS
}

fn default_agent_manifest() -> tenon_verify::DistributedAppManifest {
    tenon_verify::DistributedAppManifest {
        partitioned_entities: vec!["AgentSession".into()],
        streams: vec![tenon_verify::ResolvedStreamSpec {
            name: "ActiveTurnStream".into(),
            durability: "Volatile".into(),
            db_writes_per_delta: 0,
            slow_consumer_policy: "DropAndResync".into(),
        }],
        views: vec![tenon_verify::ResolvedViewSpec {
            name: "SessionMsgList".into(),
            sync_id: "session.msg_list".into(),
            archetype: tenon_verify::SyncArchetypeKind::FencedIncremental,
            max_items: 50,
            max_payload_bytes: 16 * 1024 * 1024,
            max_frame_bytes: 4 * 1024 * 1024,
            journal_budget_bytes: Some(16 * 1024 * 1024),
            bounded_reset: true,
            uses_anti_join_overlay: true,
        }],
        subscribers: vec![],
    }
}
