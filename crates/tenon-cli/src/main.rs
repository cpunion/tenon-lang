use std::env;
use std::fs;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use tenon_codegen::JitProgram;
use tenon_dist::authority::{AuthorityStore, ExecSlice};
use tenon_dist::run_distributed_qualification;
use tenon_dist::server::{serve_http, serve_http_with_options};
use tenon_dist::sync_kernel::UniversalSyncKernel;
use tenon_dist::worker::{http_rpc, run_external_worker};
use tenon_syntax::{Item, parse};
use tenon_verify::{
    ProofTier, check_evolution, check_program, check_program_protocols, export_program_tla_specs,
    export_tla_specs, run_all_protocol_checks,
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
        "build" => cmd_build_aot(&args[2..]),
        "build-wasm" => cmd_build_wasm(&args[2..]),
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
  tenon test <file.tn>                            Verify and run all `test fn` functions via Cranelift JIT
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
  tenon build <file.tn> -o <bin> [--emit-llvm <file.ll>] [--opt O0|O2|O3]
                                                  Verify (Fail-Closed) & AOT-compile to standalone native binary via LLVM
  tenon build-wasm <app.tn> [-o <out.wasm>]       Compile declarative .tn view/pacing directly to binary WebAssembly client store"#
    );
}

fn load_and_parse(path: &str) -> Result<tenon_syntax::Program, String> {
    let src = fs::read_to_string(path).map_err(|e| format!("failed to read `{}`: {}", path, e))?;
    parse(&src).map_err(|e| format!("{}: {}", path, e))
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
    let Some(file) = args.first() else {
        eprintln!("usage: tenon test <file.tn>");
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
        eprintln!("FAIL-CLOSED: verification failed for `{}`.", file);
        for d in &report.diagnostics {
            eprintln!("  error[{}] line {}: {}", d.code, d.span.line, d.message);
        }
        return ExitCode::from(1);
    }
    let jit = match JitProgram::compile(&prog) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("codegen error: {}", e);
            return ExitCode::from(1);
        }
    };
    let test_fns: Vec<String> = prog
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Fn(f) if f.is_test => Some(f.name.clone()),
            _ => None,
        })
        .collect();
    if test_fns.is_empty() {
        println!("No `test fn` items found in `{}`.", file);
        return ExitCode::SUCCESS;
    }
    let mut all_ok = true;
    for name in &test_fns {
        match jit.call_0(name) {
            Ok(_) => println!("  test {} ... ok", name),
            Err(e) => {
                eprintln!("  test {} ... FAILED ({})", name, e);
                all_ok = false;
            }
        }
    }
    if all_ok {
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
    let mut file_arg: Option<&str> = None;
    let mut skip = false;
    for a in args {
        if skip {
            skip = false;
            continue;
        }
        if a == "-o" || a == "--emit-llvm" || a == "--opt" {
            skip = true;
            continue;
        }
        if !a.starts_with('-') {
            file_arg = Some(a.as_str());
            break;
        }
    }
    let Some(file) = file_arg else {
        eprintln!("usage: tenon build <file.tn> [-o <bin>] [--emit-llvm <file.ll>] [--opt O3]");
        return ExitCode::from(2);
    };
    let out_bin = args
        .windows(2)
        .find(|w| w[0] == "-o")
        .map(|w| w[1].as_str())
        .unwrap_or("target/tenon_out");
    let emit_ll = args
        .windows(2)
        .find(|w| w[0] == "--emit-llvm")
        .map(|w| Path::new(w[1].as_str()));
    let opt = args
        .windows(2)
        .find(|w| w[0] == "--opt")
        .map(|w| w[1].as_str())
        .unwrap_or("O3");

    let prog = match load_and_parse(file) {
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

    match tenon_codegen::llvm_aot::compile_to_native_binary(&prog, Path::new(out_bin), emit_ll, opt)
    {
        Ok(art) => {
            println!(
                "LLVM AOT Built `{}` -> `{}` ({} bytes native binary, {} bytes LLVM IR, opt={}, {} proofs verified)",
                file,
                art.output_path,
                art.binary_bytes,
                art.llvm_ir_bytes,
                art.opt_level,
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
