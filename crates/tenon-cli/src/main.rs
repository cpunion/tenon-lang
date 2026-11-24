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
        r#"tenon — Verified Systems & Distributed Application Language (M0 Prototype)

USAGE:
  tenon check <file.tn> [--json]                  Run L0 + L1 (Z3) Fail-Closed verification & Protocol Solver
  tenon run <file.tn>                             Verify and execute `main()` via Cranelift native JIT
  tenon test [file.tn|pkg_dir] [--filter <substr>] [-r|--recursive] [--no-leak-check]
                                                  Verify and run `test fn` tests with per-test Perceus zero-leak checks
  tenon evolve check <v1.tn> <v2.tn> [--json]     Verify schema evolution compatibility & bidirectional lenses
  tenon modelcheck [file.tn] [--export-tla <dir>] [--json]
                                                  Run L2-b protocol model checker (built-in or .tn AST)
  tenon serve <app.tn> [--addr 127.0.0.1:8787] [--db <path>] [--demo]
                                                  Compile & run single-process distributed runtime + SSE Web UI
  tenon authority <app.tn> [--addr 127.0.0.1:8787] [--db <path>]
                                                  Run standalone Authority + Fenced Proxy + SSE Gateway
  tenon worker --authority http://127.0.0.1:8787 --id worker-1 [--session main] [--delay-ms 40] [--takeover] [--once]
                                                  Run standalone Worker OS process with Fabric Sandbox
  tenon cluster-demo <app.tn>                     Spawn real multi-process cluster (Authority + 2 Workers + SIGKILL failover + 410 reset)
  tenon agent [--model luna|sol|terra|astra|<id>] [--workspace <dir>] [--spec <app.tn>] [--db <sqlite>] [--env-file <path>] [-p "<prompt>"]
                                                  Interactive CLI Agent powered by real OpenAI Responses API (dast-os models) + shell & file tools
  tenon build <file.tn> -o <out> [--opt O0|O2|O3] [--lib static|shared] [--emit-header <h>] [--emit-llvm <ll>] [-g|--debug] [-l <lib>] [-L <dir>] [--c-src <file.c>]
                                                  Verify (Fail-Closed) & AOT-compile to native binary / C static or shared library via LLVM
  tenon build-wasm <app.tn> [-o <out.wasm>]       Compile declarative .tn view/pacing directly to binary WebAssembly client store
  tenon fmt <file.tn> [--check]                   Format Tenon source file with canonical indentation & spacing
  tenon lsp                                       Run Language Server Protocol (JSON-RPC 2.0 over stdio) for VS Code / editors"#
    );
}

fn load_and_parse(path: &str) -> Result<tenon_syntax::Program, String> {
    load_program_with_imports(Path::new(path))
}

fn cmd_check(args: &[String]) -> ExitCode {
    let Some(file) = args.iter().find(|a| !a.starts_with('-')) else {
        eprintln!("usage: tenon check <file.tn> [--json]");
        return ExitCode::from(2);
    };
    let as_json = args.iter().any(|a| a == "--json");

    let prog = match load_and_parse(file) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            return ExitCode::from(1);
        }
    };

    let report = check_program(&prog);
    if as_json {
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
            eprintln!(
                "  error[{}] {}:{}:{}: {}",
                d.code, file, d.span.line, d.span.col, d.message
            );
            for n in &d.notes {
                eprintln!("    note: {}", n);
            }
            if !d.counterexample.is_empty() {
                eprintln!("    counterexample: {:?}", d.counterexample);
            }
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
        eprintln!("usage: tenon evolve check <v1.tn> <v2.tn> [--json]");
        return ExitCode::from(2);
    }
    let v1_path = &args[1];
    let v2_path = &args[2];
    let as_json = args.iter().any(|a| a == "--json");

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
    // Optional `.tn` file argument (skipping flag values like `--export-tla <dir>`)
    let mut file_arg: Option<&str> = None;
    let mut skip_next = false;
    for a in args {
        if skip_next {
            skip_next = false;
            continue;
        }
        if a == "--export-tla" {
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
    let report = check_program(&prog);
    if !report.ok {
        eprintln!("FAIL-CLOSED: verification failed for `{}`.", file);
        return ExitCode::from(1);
    }
    let Some(manifest) = report.manifest else {
        eprintln!("error: `{}` has no distributed views/subscribers.", file);
        return ExitCode::from(1);
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
            "--emit-header" if i + 1 < args.len() => {
                opts.emit_header_path = Some(PathBuf::from(&args[i + 1]));
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

    match tenon_codegen::llvm_aot::compile_with_options(&prog, Path::new(&out_bin), &opts) {
        Ok(art) => {
            let kind_str = match opts.output_kind {
                tenon_codegen::llvm_aot::OutputKind::Executable => "native binary",
                tenon_codegen::llvm_aot::OutputKind::StaticLib => "static library",
                tenon_codegen::llvm_aot::OutputKind::SharedLib => "shared library",
            };
            println!(
                "LLVM AOT Built `{}` -> `{}` ({} bytes {}, {} bytes LLVM IR, opt={}, debug={}, {} proofs verified)",
                file,
                art.output_path,
                art.binary_bytes,
                kind_str,
                art.llvm_ir_bytes,
                art.opt_level,
                opts.debug_info,
                report.proofs.len()
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("LLVM AOT build failed: {}", e);
            ExitCode::from(1)
        }
    }
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
