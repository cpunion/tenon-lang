use crate::llvm_aot_rt::{
    TENON_AOT_HTTP_C, TENON_AOT_MAIN_CLI_C, TENON_AOT_MAIN_STATS_C, TENON_AOT_OS_C,
    TENON_AOT_RUNTIME_C, TENON_INLINE_IR_HELPERS, TENON_NONATOMIC_RC_IR_HELPERS,
    TENON_TBAA_METADATA,
};
use crate::llvm_runtime::{
    TENON_AOT_NET_C, TENON_AOT_SYNC_C, TENON_AOT_SYSTEMS_CORE_C, TENON_ATOMIC_RC_IR_HELPERS,
};
pub use crate::llvm_runtime::emit_c_header;
use crate::lower_interfaces::{DynInterfaceLayout, prepare_program_for_codegen};
use crate::perceus::{
    RC_MASK_ALL, TAG_CLOSURE, TAG_CSTRUCT, TAG_DYN, TAG_ENUM_BASE, TAG_ERR, TAG_LIST, TAG_NONE,
    TAG_OK, TAG_RECORD_BASE, TAG_SOME, TAG_TUPLE,
};
use crate::{
    CORE_SYSCALLS, generic_ty, is_consuming_cow_builtin, is_f32_type_expr, is_float_type_expr,
    is_rc_type_expr, is_text_type_expr, named_ty, parse_syscall_ret_ty, resolve_alias_ty,
    scalar_kind_of_ty, split_async_segments,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use tenon_syntax::*;

static AOT_TMP_SEQ: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputKind {
    Executable,
    StaticLib,
    SharedLib,
}

#[derive(Debug, Clone)]
pub struct BuildOptions {
    pub opt_level: String,
    pub output_kind: OutputKind,
    pub emit_ll_path: Option<PathBuf>,
    pub emit_header_path: Option<PathBuf>,
    pub debug_info: bool,
    pub link_libs: Vec<String>,
    pub link_dirs: Vec<String>,
    pub extra_c_sources: Vec<PathBuf>,
    /// Span start offsets of `List`/`Bytes` index reads/writes statically proved in-bounds (`0 <= i < len(xs)`) by `tenon-verify`.
    pub proved_safe_indices: BTreeSet<usize>,
    /// Span start offsets of integer `+`/`-`/`*` statically proved overflow-free by `tenon-verify` (`VerifiedMir`).
    pub proved_no_overflow: BTreeSet<usize>,
    /// Per-function local variables proved non-escaping (Tier-2 escape analysis -> eliminates COW uniqueness checks).
    pub non_escaping_vars: BTreeMap<String, BTreeSet<String>>,
    /// Per-function local immutable scalar-only records proved safe for zero-malloc LLVM stack `alloca` promotion.
    pub stack_promotable_vars: BTreeMap<String, BTreeSet<String>>,
    /// Per-function span offsets of owned RC local reads proved by `tenon-mir` liveness to be Last-Use Moves.
    pub last_use_move_spans: BTreeMap<String, BTreeSet<usize>>,
    /// Emit checked signed integer overflow traps for arithmetic NOT statically proved overflow-free.
    pub checked_overflow: bool,
    /// Optional target triple override for cross-compilation (e.g. `x86_64-unknown-linux-gnu`).
    pub target_triple: Option<String>,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            opt_level: "O3".into(),
            output_kind: OutputKind::Executable,
            emit_ll_path: None,
            emit_header_path: None,
            debug_info: false,
            link_libs: Vec::new(),
            link_dirs: Vec::new(),
            extra_c_sources: Vec::new(),
            proved_safe_indices: BTreeSet::new(),
            proved_no_overflow: BTreeSet::new(),
            non_escaping_vars: BTreeMap::new(),
            stack_promotable_vars: BTreeMap::new(),
            last_use_move_spans: BTreeMap::new(),
            checked_overflow: false,
            target_triple: None,
        }
    }
}

impl BuildOptions {
    /// Applies all proof-guided optimization facts from `tenon_verify::VerifiedMir`.
    pub fn apply_verified_mir(&mut self, mir: &tenon_verify::VerifiedMir) {
        self.proved_safe_indices = mir.proved_safe_indices.clone();
        self.proved_no_overflow = mir.proved_no_overflow.clone();
        self.non_escaping_vars = mir.non_escaping_vars.clone();
        self.stack_promotable_vars = mir.stack_promotable_vars.clone();
    }

    /// Applies SSA/CFG liveness Last-Use Move spans from `tenon_mir::MirProgram`.
    pub fn apply_ssa_mir(&mut self, mir: &tenon_mir::MirProgram) {
        self.last_use_move_spans = tenon_mir::collect_last_use_move_spans(mir);
    }
}

#[derive(Debug, Clone)]
pub struct BuildArtifact {
    pub output_path: String,
    pub llvm_ir_bytes: usize,
    pub binary_bytes: u64,
    pub opt_level: String,
}

#[derive(Debug, Clone)]
struct LlvmFieldLayout {
    name: String,
    ty: TypeExpr,
    preds: Vec<Expr>,
    is_rc: bool,
}

#[derive(Debug, Clone)]
struct LlvmRecordLayout {
    tag: i64,
    fields: Vec<LlvmFieldLayout>,
    rc_mask: i64,
    where_clauses: Vec<Expr>,
    c_layout: Option<StructByteLayout>,
}

#[derive(Debug, Clone)]
struct LlvmEnumVariantLayout {
    enum_name: String,
    payload_tys: Vec<TypeExpr>,
}

#[derive(Debug, Clone)]
struct LlvmFnMeta {
    params_inout: Vec<bool>,
    param_tys: Vec<TypeExpr>,
    ret_ty: Option<TypeExpr>,
    returns_rc: bool,
}

fn ir_uses_os_syscalls(ir: &str) -> bool {
    ir.contains("call i64 @tenon_sys_")
}

fn ir_uses_http_syscalls(ir: &str) -> bool {
    ir.contains("call i64 @tenon_sys_http_")
}

fn ir_uses_sync_runtime(ir: &str) -> bool {
    ir.contains("call i64 @tenon_thread_")
        || ir.contains("call i64 @tenon_atomic_")
        || ir.contains("call i64 @tenon_mutex_")
        || ir.contains("call i64 @tenon_chan_")
        || ir.contains("@tenon_sched_")
        || ir.contains("@tenon_preempt_flag")
}

fn ir_uses_net_runtime(ir: &str) -> bool {
    ir.contains("call i64 @tenon_net_")
        || ir.contains("call i64 @tenon_reactor_")
        || ir.contains("call i64 @tenon_time_")
}

fn ir_uses_systems_core(ir: &str) -> bool {
    ir.contains("call i64 @tenon_bytes_")
        || ir.contains("call i64 @tenon_slice_")
        || ir.contains("call i64 @tenon_cstruct_")
        || ir.contains("call i64 @tenon_raw_")
        || ir.contains("call i64 @tenon_f32_")
        || ir.contains("call i64 @tenon_parse_f32")
        || ir.contains("call i64 @tenon_text_cstr_")
        || ir.contains("call i64 @tenon_cstr_")
        || ir.contains("call i64 @tenon_text_to_bytes")
        || ir.contains("call i64 @tenon_clz")
        || ir.contains("call i64 @tenon_ctz")
        || ir.contains("call i64 @tenon_popcnt")
        || ir.contains("call i64 @tenon_bswap")
        || ir.contains("call i64 @tenon_rot")
        || ir.contains("call i64 @tenon_gpu_")
        || ir_uses_net_runtime(ir)
}

pub fn compile_to_native_binary(
    program: &Program,
    output_path: &Path,
    emit_ll_path: Option<&Path>,
    opt_level: &str,
) -> Result<BuildArtifact, String> {
    let opts = BuildOptions {
        opt_level: opt_level.to_string(),
        output_kind: OutputKind::Executable,
        emit_ll_path: emit_ll_path.map(PathBuf::from),
        ..BuildOptions::default()
    };
    compile_with_options(program, output_path, &opts)
}

pub fn compile_with_options(
    program: &Program,
    output_path: &Path,
    opts: &BuildOptions,
) -> Result<BuildArtifact, String> {
    if let Some(hpath) = &opts.emit_header_path {
        if let Some(parent) = hpath.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let header = emit_c_header(program);
        fs::write(hpath, header).map_err(|e| e.to_string())?;
    }

    let ir = emit_llvm_ir_with_build_options(program, opts)?;
    let pid = std::process::id();
    let seq = AOT_TMP_SEQ.fetch_add(1, Ordering::SeqCst);
    let tmp_ll = std::env::temp_dir().join(format!("tenon_aot_{}_{}.ll", pid, seq));
    let tmp_rt = std::env::temp_dir().join(format!("tenon_rt_{}_{}.c", pid, seq));

    fs::write(&tmp_ll, &ir).map_err(|e| e.to_string())?;
    let uses_os = ir_uses_os_syscalls(&ir);
    let uses_http = ir_uses_http_syscalls(&ir);
    let uses_sync = ir_uses_sync_runtime(&ir);
    let uses_net = ir_uses_net_runtime(&ir);
    let uses_sys_core = ir_uses_systems_core(&ir);

    let mut rt_c = String::with_capacity(TENON_AOT_RUNTIME_C.len() + 16384);
    if uses_sync {
        rt_c.push_str("#define TENON_THREAD_SAFE 1\n#define TENON_USE_SYNC 1\n");
    }
    rt_c.push_str(TENON_AOT_RUNTIME_C);
    if uses_sys_core {
        rt_c.push_str(TENON_AOT_SYSTEMS_CORE_C);
    }
    if uses_sync {
        rt_c.push_str(TENON_AOT_SYNC_C);
    }
    if uses_net {
        rt_c.push_str(TENON_AOT_NET_C);
    }
    if uses_os {
        rt_c.push_str(TENON_AOT_OS_C);
    }
    if uses_http {
        rt_c.push_str(TENON_AOT_HTTP_C);
    }
    if opts.output_kind == OutputKind::Executable {
        if uses_os {
            rt_c.push_str(TENON_AOT_MAIN_CLI_C);
        } else {
            rt_c.push_str(TENON_AOT_MAIN_STATS_C);
        }
    }
    fs::write(&tmp_rt, &rt_c).map_err(|e| e.to_string())?;

    if let Some(p) = &opts.emit_ll_path {
        if let Some(parent) = p.parent() {
            let _ = fs::create_dir_all(parent);
        }
        fs::write(p, &ir).map_err(|e| e.to_string())?;
    }
    if let Some(parent) = output_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let opt_flag = match opts.opt_level.as_str() {
        "O0" | "-O0" => "-O0",
        "O1" | "-O1" => "-O1",
        "O2" | "-O2" => "-O2",
        _ => "-O3",
    };

    // Collect `@link("lib")`, `@link(framework = "Metal")`, and `@c_src("companion.m")` attributes
    let mut all_link_libs: Vec<String> = Vec::new();
    let mut all_link_frameworks: Vec<String> = Vec::new();
    let mut all_c_sources: Vec<PathBuf> = Vec::new();
    for item in &program.items {
        let (attrs_opt, span_fid) = match item {
            Item::ExternFn(ef) => (Some(&ef.attrs), ef.span.file_id),
            Item::Fn(f) => (Some(&f.attrs), f.span.file_id),
            _ => (None, 0),
        };
        if let Some(attrs) = attrs_opt {
            for lib in extract_link_libs(attrs) {
                if !all_link_libs.contains(&lib) {
                    all_link_libs.push(lib);
                }
            }
            for fw in extract_link_frameworks(attrs) {
                if !all_link_frameworks.contains(&fw) {
                    all_link_frameworks.push(fw);
                }
            }
            let src_dir = program
                .source_files
                .get(span_fid as usize)
                .and_then(|sf| Path::new(sf).parent().map(|p| p.to_path_buf()));
            for rel in extract_c_sources(attrs) {
                let candidate = if let Some(dir) = &src_dir {
                    let p = dir.join(&rel);
                    if p.exists() { p } else { PathBuf::from(&rel) }
                } else {
                    PathBuf::from(&rel)
                };
                let canon = fs::canonicalize(&candidate).unwrap_or(candidate);
                if !all_c_sources.contains(&canon) {
                    all_c_sources.push(canon);
                }
            }
        }
    }
    for lib in &opts.link_libs {
        if !all_link_libs.contains(lib) {
            all_link_libs.push(lib.clone());
        }
    }
    for csrc in &opts.extra_c_sources {
        let canon = fs::canonicalize(csrc).unwrap_or_else(|_| csrc.clone());
        if !all_c_sources.contains(&canon) {
            all_c_sources.push(canon);
        }
    }

    let has_objc_sources = all_c_sources.iter().any(|p| {
        p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e == "m" || e == "mm")
    });

    // Content-addressed compiled object cache for the Tenon C runtime (`rt_c`)!
    // Avoids recompiling ~1,500 lines of runtime C on repeated `tenon build` and `cargo test` runs.
    let mut rt_hash: u64 = 0xcbf29ce484222325;
    for &b in rt_c.as_bytes() {
        rt_hash ^= b as u64;
        rt_hash = rt_hash.wrapping_mul(0x100000001b3);
    }
    let use_thin_lto = opts.output_kind == OutputKind::Executable
        && opt_flag == "-O3"
        && !opts.debug_info
        && opts.target_triple.is_none()
        && !has_objc_sources;
    let rt_cache_key = format!(
        "tenon_rt_cache_{:016x}_{}_{}_{}_{}.o",
        rt_hash,
        opt_flag.trim_start_matches('-'),
        if opts.debug_info { "g" } else { "ng" },
        if use_thin_lto { "lto" } else { "nolto" },
        opts.target_triple.as_deref().unwrap_or("host").replace(['/', ':', ' '], "_")
    );
    let rt_cache_obj = std::env::temp_dir().join(&rt_cache_key);
    let rt_input_path = if opts.output_kind != OutputKind::SharedLib {
        if !rt_cache_obj.exists() {
            let rt_tmp_obj = std::env::temp_dir().join(format!("tenon_rt_build_{}_{}.o", pid, seq));
            let mut crt = Command::new("clang");
            crt.arg(opt_flag).arg("-c");
            if use_thin_lto {
                crt.arg("-flto=thin");
            }
            if let Some(triple) = &opts.target_triple {
                crt.arg(format!("--target={}", triple));
            }
            if opts.debug_info {
                crt.arg("-g");
            }
            crt.args([tmp_rt.to_str().unwrap(), "-o", rt_tmp_obj.to_str().unwrap()]);
            if crt.output().is_ok_and(|o| o.status.success()) {
                let _ = fs::rename(&rt_tmp_obj, &rt_cache_obj);
            }
        }
        if rt_cache_obj.exists() {
            rt_cache_obj.clone()
        } else {
            tmp_rt.clone()
        }
    } else {
        tmp_rt.clone()
    };

    let res = match opts.output_kind {
        OutputKind::StaticLib => {
            let tmp_obj_ll = std::env::temp_dir().join(format!("tenon_ll_{}_{}.o", pid, seq));
            let tmp_obj_rt = std::env::temp_dir().join(format!("tenon_rt_{}_{}.o", pid, seq));
            let mut extra_objs: Vec<PathBuf> = Vec::new();

            let mut c1 = Command::new("clang");
            c1.arg(opt_flag).arg("-c").arg("-Wno-override-module");
            if let Some(triple) = &opts.target_triple {
                c1.arg(format!("--target={}", triple));
            }
            if opts.debug_info {
                c1.arg("-g");
            }
            c1.args([tmp_ll.to_str().unwrap(), "-o", tmp_obj_ll.to_str().unwrap()]);
            let o1 = c1.output().map_err(|e| format!("clang failed: {}", e))?;
            if !o1.status.success() {
                let _ = fs::remove_file(&tmp_ll);
                let _ = fs::remove_file(&tmp_rt);
                return Err(format!(
                    "LLVM clang -c (IR) failed:\n{}",
                    String::from_utf8_lossy(&o1.stderr)
                ));
            }

            let mut c2 = Command::new("clang");
            c2.arg(opt_flag).arg("-c");
            if opts.debug_info {
                c2.arg("-g");
            }
            c2.args([tmp_rt.to_str().unwrap(), "-o", tmp_obj_rt.to_str().unwrap()]);
            let o2 = c2.output().map_err(|e| format!("clang failed: {}", e))?;
            if !o2.status.success() {
                let _ = fs::remove_file(&tmp_ll);
                let _ = fs::remove_file(&tmp_rt);
                let _ = fs::remove_file(&tmp_obj_ll);
                return Err(format!(
                    "LLVM clang -c (RT) failed:\n{}",
                    String::from_utf8_lossy(&o2.stderr)
                ));
            }

            for (idx, csrc) in all_c_sources.iter().enumerate() {
                let eo = std::env::temp_dir().join(format!("tenon_ex_{}_{}_{}.o", pid, seq, idx));
                let mut ce = Command::new("clang");
                ce.arg(opt_flag).arg("-c");
                if csrc.extension().and_then(|e| e.to_str()).is_some_and(|e| e == "m" || e == "mm") {
                    ce.arg("-fobjc-arc");
                }
                if opts.debug_info {
                    ce.arg("-g");
                }
                ce.args([csrc.to_str().unwrap(), "-o", eo.to_str().unwrap()]);
                let oe = ce.output().map_err(|e| format!("clang failed: {}", e))?;
                if !oe.status.success() {
                    return Err(format!(
                        "LLVM clang -c ({}) failed:\n{}",
                        csrc.display(),
                        String::from_utf8_lossy(&oe.stderr)
                    ));
                }
                extra_objs.push(eo);
            }

            let _ = fs::remove_file(output_path);
            let mut ar = Command::new("ar");
            ar.args([
                "rcs",
                output_path.to_str().unwrap(),
                tmp_obj_ll.to_str().unwrap(),
                tmp_obj_rt.to_str().unwrap(),
            ]);
            for eo in &extra_objs {
                ar.arg(eo.to_str().unwrap());
            }
            let oar = ar.output().map_err(|e| format!("ar rcs failed: {}", e))?;
            let _ = fs::remove_file(&tmp_obj_ll);
            let _ = fs::remove_file(&tmp_obj_rt);
            for eo in &extra_objs {
                let _ = fs::remove_file(eo);
            }
            if !oar.status.success() {
                Err(format!("ar rcs failed:\n{}", String::from_utf8_lossy(&oar.stderr)))
            } else {
                Ok(())
            }
        }
        OutputKind::SharedLib | OutputKind::Executable => {
            let emit_obj_only = output_path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e == "o" || e == "obj");
            let mut cmd = Command::new("clang");
            cmd.arg(opt_flag);
            if let Some(triple) = &opts.target_triple {
                cmd.arg(format!("--target={}", triple));
            }
            if opts.debug_info {
                cmd.arg("-g");
            }
            if emit_obj_only {
                cmd.args(["-c", "-Wno-override-module", tmp_ll.to_str().unwrap()]);
            } else {
                if opts.output_kind == OutputKind::SharedLib {
                    cmd.args(["-shared", "-fPIC"]);
                } else if use_thin_lto {
                    cmd.arg("-flto=thin");
                }
                if has_objc_sources {
                    cmd.arg("-fobjc-arc");
                }
                cmd.args([
                    "-Wno-override-module",
                    tmp_ll.to_str().unwrap(),
                    rt_input_path.to_str().unwrap(),
                ]);
                for csrc in &all_c_sources {
                    cmd.arg(csrc.to_str().unwrap());
                }
                for dir in &opts.link_dirs {
                    cmd.arg(format!("-L{}", dir));
                }
                if uses_http {
                    cmd.args(["-lcurl", "-lpthread"]);
                } else if uses_sync {
                    cmd.arg("-lpthread");
                }
                for lib in &all_link_libs {
                    cmd.arg(format!("-l{}", lib));
                }
                for fw in &all_link_frameworks {
                    cmd.args(["-framework", fw]);
                }
            }
            cmd.args(["-o", output_path.to_str().unwrap()]);
            let out = cmd
                .output()
                .map_err(|e| format!("failed to invoke LLVM clang: {}", e))?;
            if !out.status.success() {
                Err(format!(
                    "LLVM clang compilation failed:\nstdout: {}\nstderr: {}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                ))
            } else {
                Ok(())
            }
        }
    };

    let _ = fs::remove_file(&tmp_ll);
    let _ = fs::remove_file(&tmp_rt);
    res?;

    let binary_bytes = fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
    Ok(BuildArtifact {
        output_path: output_path.display().to_string(),
        llvm_ir_bytes: ir.len(),
        binary_bytes,
        opt_level: opt_flag.to_string(),
    })
}

/// Returns true if LLVM's `lli` JIT executor is available on PATH (e.g. Homebrew LLVM 22).
pub fn has_llvm_lli() -> bool {
    static HAS_LLI: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *HAS_LLI.get_or_init(|| {
        Command::new("lli")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

fn resolve_dylib_for_lli(lib: &str, link_dirs: &[String]) -> Option<PathBuf> {
    let fname = format!("lib{}.dylib", lib);
    for dir in link_dirs {
        let p = Path::new(dir).join(&fname);
        if p.exists() {
            return Some(p);
        }
    }
    for sys_dir in ["/opt/homebrew/lib", "/usr/local/lib", "/usr/lib"] {
        let p = Path::new(sys_dir).join(&fname);
        if p.exists() {
            return Some(p);
        }
    }
    // On modern macOS, system dylibs (e.g. libsqlite3.dylib, libcurl.dylib) live inside the
    // dyld shared cache so Path::exists() is false, yet dlopen("/usr/lib/libX.dylib") succeeds.
    Some(PathBuf::from(format!("/usr/lib/{}", fname)))
}

fn build_or_get_cached_rt_bc(rt_c: &str, pid: u32, seq: u64) -> Result<PathBuf, String> {
    let mut rt_hash: u64 = 0xcbf29ce484222325;
    for &b in rt_c.as_bytes() {
        rt_hash ^= b as u64;
        rt_hash = rt_hash.wrapping_mul(0x100000001b3);
    }
    let cache_bc = std::env::temp_dir().join(format!("tenon_rt_cache_lli_{:016x}.bc", rt_hash));
    if !cache_bc.exists() {
        let tmp_c = std::env::temp_dir().join(format!("tenon_rt_bc_{}_{}.c", pid, seq));
        let tmp_bc = std::env::temp_dir().join(format!("tenon_rt_bc_{}_{}.bc", pid, seq));
        fs::write(&tmp_c, rt_c).map_err(|e| e.to_string())?;
        let out = Command::new("clang")
            .args([
                "-O1",
                "-DTENON_LLI_MODE=1",
                "-emit-llvm",
                "-c",
                tmp_c.to_str().unwrap(),
                "-o",
                tmp_bc.to_str().unwrap(),
            ])
            .output()
            .map_err(|e| format!("clang -emit-llvm failed: {}", e))?;
        let _ = fs::remove_file(&tmp_c);
        if !out.status.success() {
            let _ = fs::remove_file(&tmp_bc);
            return Err(format!(
                "clang -emit-llvm for runtime failed:\n{}",
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        let _ = fs::rename(&tmp_bc, &cache_bc);
    }
    Ok(cache_bc)
}

/// Runs a C driver file linked against a Tenon library program via `lli` (fast macOS test mode)
/// or executes `native_bin` directly when `TENON_TEST_NATIVE_EXEC=1` or on non-macOS hosts.
pub fn run_c_driver_against_tenon_lib_for_test(
    tenon_lib: &Program,
    c_main_path: &Path,
    native_bin: &Path,
    opts: &BuildOptions,
) -> Result<std::process::Output, String> {
    let use_lli = cfg!(target_os = "macos")
        && std::env::var("TENON_TEST_NATIVE_EXEC").as_deref() != Ok("1")
        && has_llvm_lli();
    if !use_lli {
        return Command::new(native_bin)
            .output()
            .map_err(|e| format!("failed to run {}: {}", native_bin.display(), e));
    }
    let pid = std::process::id();
    let seq = AOT_TMP_SEQ.fetch_add(1, Ordering::SeqCst);
    let ir = emit_llvm_ir_with_build_options(tenon_lib, opts)?;
    let tmp_ll = std::env::temp_dir().join(format!("tenon_lli_lib_{}_{}.ll", pid, seq));
    let tmp_c_bc = std::env::temp_dir().join(format!("tenon_lli_cdrv_{}_{}.bc", pid, seq));
    fs::write(&tmp_ll, &ir).map_err(|e| e.to_string())?;
    let cc = Command::new("clang")
        .args([
            "-O1",
            "-emit-llvm",
            "-c",
            c_main_path.to_str().unwrap(),
            "-o",
            tmp_c_bc.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| format!("clang -emit-llvm c_driver failed: {}", e))?;
    if !cc.status.success() {
        let _ = fs::remove_file(&tmp_ll);
        return Err(format!(
            "clang -emit-llvm c_driver failed:\n{}",
            String::from_utf8_lossy(&cc.stderr)
        ));
    }
    let mut rt_src = String::from(TENON_AOT_RUNTIME_C);
    if ir_uses_systems_core(&ir) {
        rt_src.push_str(TENON_AOT_SYSTEMS_CORE_C);
    }
    let rt_bc = build_or_get_cached_rt_bc(&rt_src, pid, seq)?;
    let out = Command::new("lli")
        .arg(format!("--extra-module={}", tmp_ll.display()))
        .arg(format!("--extra-module={}", rt_bc.display()))
        .arg(&tmp_c_bc)
        .output()
        .map_err(|e| format!("lli failed: {}", e));
    let _ = fs::remove_file(&tmp_ll);
    let _ = fs::remove_file(&tmp_c_bc);
    out
}

/// Compiles a Tenon program to a real native binary with `clang` (verifying full IR emission,
/// optimization, and native linking) and executes it for test verification.
/// On macOS, executes the exact emitted LLVM IR + cached C runtime bitcode through LLVM's `lli`
/// ORC JIT (unless `TENON_TEST_NATIVE_EXEC=1`), bypassing the ~4.3s/binary macOS `syspolicyd`
/// first-launch scan while preserving 100% real ARM64 LLVM execution and native `clang` link verification.
pub fn compile_and_run_for_test(
    program: &Program,
    bin_path: &Path,
    opts: &BuildOptions,
) -> Result<(BuildArtifact, std::process::Output), String> {
    let use_lli = cfg!(target_os = "macos")
        && std::env::var("TENON_TEST_NATIVE_EXEC").as_deref() != Ok("1")
        && has_llvm_lli();

    let pid = std::process::id();
    let seq = AOT_TMP_SEQ.fetch_add(1, Ordering::SeqCst);
    let tmp_keep_ll = if use_lli && opts.emit_ll_path.is_none() {
        Some(std::env::temp_dir().join(format!("tenon_lli_prog_{}_{}.ll", pid, seq)))
    } else {
        None
    };
    let mut build_opts = opts.clone();
    if let Some(p) = &tmp_keep_ll {
        build_opts.emit_ll_path = Some(p.clone());
    }

    // 1. Always verify full real native compilation & linking with `clang`!
    let art = compile_with_options(program, bin_path, &build_opts)?;

    if !use_lli {
        let out = Command::new(bin_path)
            .output()
            .map_err(|e| format!("failed to execute {}: {}", bin_path.display(), e))?;
        let _ = fs::remove_file(bin_path);
        return Ok((art, out));
    }

    // 2. Execute the exact emitted LLVM IR + cached runtime bitcode via `lli` (20-40ms in parallel!).
    let ll_path = build_opts.emit_ll_path.as_ref().unwrap();
    let ir = fs::read_to_string(ll_path).map_err(|e| e.to_string())?;
    let uses_os = ir_uses_os_syscalls(&ir);
    let uses_http = ir_uses_http_syscalls(&ir);
    let uses_sync = ir_uses_sync_runtime(&ir);
    let uses_net = ir_uses_net_runtime(&ir);
    let uses_sys_core = ir_uses_systems_core(&ir);

    let mut rt_c = String::with_capacity(TENON_AOT_RUNTIME_C.len() + 16384);
    if uses_sync {
        rt_c.push_str("#define TENON_THREAD_SAFE 1\n#define TENON_USE_SYNC 1\n");
    }
    rt_c.push_str(TENON_AOT_RUNTIME_C);
    if uses_sys_core {
        rt_c.push_str(TENON_AOT_SYSTEMS_CORE_C);
    }
    if uses_sync {
        rt_c.push_str(TENON_AOT_SYNC_C);
    }
    if uses_net {
        rt_c.push_str(TENON_AOT_NET_C);
    }
    if uses_os {
        rt_c.push_str(TENON_AOT_OS_C);
    }
    if uses_http {
        rt_c.push_str(TENON_AOT_HTTP_C);
    }
    if uses_os {
        rt_c.push_str(TENON_AOT_MAIN_CLI_C);
    } else {
        rt_c.push_str(TENON_AOT_MAIN_STATS_C);
    }

    let rt_bc = build_or_get_cached_rt_bc(&rt_c, pid, seq)?;

    let mut extra_bcs: Vec<PathBuf> = Vec::new();
    for (idx, csrc) in opts.extra_c_sources.iter().enumerate() {
        let ebc = std::env::temp_dir().join(format!("tenon_lli_ex_{}_{}_{}.bc", pid, seq, idx));
        let co = Command::new("clang")
            .args([
                "-O1",
                "-emit-llvm",
                "-c",
                csrc.to_str().unwrap(),
                "-o",
                ebc.to_str().unwrap(),
            ])
            .output()
            .map_err(|e| format!("clang -emit-llvm extra C failed: {}", e))?;
        if !co.status.success() {
            let _ = fs::remove_file(bin_path);
            if let Some(p) = &tmp_keep_ll {
                let _ = fs::remove_file(p);
            }
            return Err(format!(
                "clang -emit-llvm ({}) failed:\n{}",
                csrc.display(),
                String::from_utf8_lossy(&co.stderr)
            ));
        }
        extra_bcs.push(ebc);
    }

    // Collect `@link("lib")` libraries + load `crypto` before `ssl` for clean symbol resolution.
    let mut all_libs: Vec<String> = Vec::new();
    for item in &program.items {
        let attrs_opt = match item {
            Item::ExternFn(ef) => Some(&ef.attrs),
            Item::Fn(f) => Some(&f.attrs),
            _ => None,
        };
        if let Some(attrs) = attrs_opt {
            for lib in extract_link_libs(attrs) {
                if !all_libs.contains(&lib) {
                    all_libs.push(lib);
                }
            }
        }
    }
    for lib in &opts.link_libs {
        if !all_libs.contains(lib) {
            all_libs.push(lib.clone());
        }
    }
    if uses_http && !all_libs.iter().any(|l| l == "curl") {
        all_libs.push("curl".into());
    }
    all_libs.sort_by_key(|l| if l == "crypto" { 0 } else if l == "ssl" { 1 } else { 2 });

    let mut lli = Command::new("lli");
    for lib in &all_libs {
        if let Some(dylib) = resolve_dylib_for_lli(lib, &opts.link_dirs) {
            lli.arg(format!("--dlopen={}", dylib.display()));
        }
    }
    lli.arg(format!("--extra-module={}", ll_path.display()));
    for ebc in &extra_bcs {
        lli.arg(format!("--extra-module={}", ebc.display()));
    }
    lli.arg(&rt_bc);

    let out = lli
        .output()
        .map_err(|e| format!("failed to run lli: {}", e))?;

    let _ = fs::remove_file(bin_path);
    if let Some(p) = &tmp_keep_ll {
        let _ = fs::remove_file(p);
    }
    for ebc in &extra_bcs {
        let _ = fs::remove_file(ebc);
    }
    Ok((art, out))
}

fn llvm_c_abi_type(ty: &TypeExpr, aliases: &BTreeMap<String, TypeExpr>) -> &'static str {
    match resolve_alias_ty(ty, aliases) {
        TypeExpr::Named { name, .. } => match ScalarKind::from_name(&name) {
            Some(ScalarKind::I8 | ScalarKind::U8) => "i8",
            Some(ScalarKind::I16 | ScalarKind::U16 | ScalarKind::F16 | ScalarKind::BF16) => "i16",
            Some(ScalarKind::I32 | ScalarKind::U32) => "i32",
            Some(ScalarKind::I64 | ScalarKind::U64) => "i64",
            Some(ScalarKind::F32) => "float",
            Some(ScalarKind::F64) => "double",
            None => match name.as_str() {
                "Bool" => "i1",
                "CVoid" => "void",
                "Ptr" | "MutPtr" | "CStr" => "ptr",
                _ => "i64",
            },
        },
        _ => "i64",
    }
}

fn emit_export_wrapper(
    f: &FnDecl,
    cname: &str,
    aliases: &BTreeMap<String, TypeExpr>,
) -> String {
    let ret_llvm = match &f.ret_ty {
        Some(rt) => llvm_c_abi_type(rt, aliases),
        None => "void",
    };
    let mut sig_params = Vec::new();
    let mut conv_lines = String::new();
    let mut call_args = Vec::new();
    for (i, p) in f.params.iter().enumerate() {
        let cty = llvm_c_abi_type(&p.ty, aliases);
        let arg = format!("%c_arg_{}", i);
        sig_params.push(format!("{} {}", cty, arg));
        let sk = scalar_kind_of_ty(&p.ty, aliases);
        let tn_val = format!("%tn_arg_{}", i);
        match cty {
            "i8" | "i16" | "i32" => {
                let ext = if sk.is_some_and(|k| k.is_unsigned()) { "zext" } else { "sext" };
                conv_lines.push_str(&format!("  {} = {} {} {} to i64\n", tn_val, ext, cty, arg));
            }
            "i1" => {
                conv_lines.push_str(&format!("  {} = zext i1 {} to i64\n", tn_val, arg));
            }
            "float" => {
                let b32 = format!("%fb32_{}", i);
                conv_lines.push_str(&format!(
                    "  {} = bitcast float {} to i32\n  {} = zext i32 {} to i64\n",
                    b32, arg, tn_val, b32
                ));
            }
            "double" => {
                conv_lines.push_str(&format!("  {} = bitcast double {} to i64\n", tn_val, arg));
            }
            "ptr" => {
                conv_lines.push_str(&format!("  {} = ptrtoint ptr {} to i64\n", tn_val, arg));
            }
            _ => {
                call_args.push(format!("i64 {}", arg));
                continue;
            }
        }
        call_args.push(format!("i64 {}", tn_val));
    }
    let mut body = conv_lines;
    body.push_str(&format!(
        "  %raw_ret = call i64 @tenon_fn_{}({})\n",
        sanitize(&f.name),
        call_args.join(", ")
    ));
    match ret_llvm {
        "void" => body.push_str("  ret void\n"),
        "i8" | "i16" | "i32" => {
            body.push_str(&format!(
                "  %out = trunc i64 %raw_ret to {}\n  ret {} %out\n",
                ret_llvm, ret_llvm
            ));
        }
        "i1" => {
            body.push_str("  %out = icmp ne i64 %raw_ret, 0\n  ret i1 %out\n");
        }
        "float" => {
            body.push_str(
                "  %r32 = trunc i64 %raw_ret to i32\n  %out = bitcast i32 %r32 to float\n  ret float %out\n",
            );
        }
        "double" => {
            body.push_str("  %out = bitcast i64 %raw_ret to double\n  ret double %out\n");
        }
        "ptr" => {
            body.push_str("  %out = inttoptr i64 %raw_ret to ptr\n  ret ptr %out\n");
        }
        _ => {
            body.push_str("  ret i64 %raw_ret\n");
        }
    }
    format!(
        "define {} @{}({}) {{\nentry:\n{}}}\n",
        ret_llvm,
        cname,
        sig_params.join(", "),
        body
    )
}

pub fn emit_llvm_ir(program: &Program) -> Result<String, String> {
    emit_llvm_ir_with_build_options(program, &BuildOptions::default())
}

pub fn emit_llvm_ir_with_options(program: &Program, debug_info: bool) -> Result<String, String> {
    let mut opts = BuildOptions::default();
    opts.debug_info = debug_info;
    emit_llvm_ir_with_build_options(program, &opts)
}

pub fn emit_llvm_ir_with_build_options(
    program: &Program,
    opts: &BuildOptions,
) -> Result<String, String> {
    let debug_info = opts.debug_info;
    let proved_safe_indices = &opts.proved_safe_indices;
    let proved_no_overflow = &opts.proved_no_overflow;
    let non_escaping_vars = &opts.non_escaping_vars;
    let stack_promotable_vars = &opts.stack_promotable_vars;
    let last_use_move_spans = &opts.last_use_move_spans;
    let checked_overflow = opts.checked_overflow;
    let target_triple = opts.target_triple.as_deref();
    let prepared = prepare_program_for_codegen(program);
    let program = &prepared.program;
    let layout_engine = LayoutEngine::from_program(program);
    let vtable_ifaces: BTreeMap<String, String> = prepared
        .vtables
        .iter()
        .map(|v| (v.sym_name.clone(), v.interface.clone()))
        .collect();
    let dyn_interfaces = &prepared.dyn_interfaces;
    let mut alias_preds: BTreeMap<String, Expr> = BTreeMap::new();
    let mut alias_bases: BTreeMap<String, TypeExpr> = BTreeMap::new();
    for item in &program.items {
        if let Item::TypeAlias(ta) = item {
            alias_bases.insert(ta.name.clone(), ta.base.clone());
            if let Some(p) = &ta.predicate {
                alias_preds.insert(ta.name.clone(), p.clone());
            }
        }
    }

    let record_names: BTreeSet<String> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Record(r) if r.type_params.is_empty() => Some(r.name.clone()),
            Item::Union(u) => Some(u.name.clone()),
            _ => None,
        })
        .collect();
    let enum_names: BTreeSet<String> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Enum(e) => Some(e.name.clone()),
            _ => None,
        })
        .collect();

    let mut records: BTreeMap<String, LlvmRecordLayout> = BTreeMap::new();
    let mut field_to_idx: BTreeMap<String, usize> = BTreeMap::new();
    let mut enum_tags: BTreeMap<String, i64> = BTreeMap::new();
    let mut enum_variants: BTreeMap<String, LlvmEnumVariantLayout> = BTreeMap::new();
    let mut extern_fns: BTreeMap<String, ExternFnDecl> = BTreeMap::new();
    enum_tags.insert("Err".into(), TAG_ERR);
    enum_tags.insert("Ok".into(), TAG_OK);
    enum_tags.insert("None".into(), TAG_NONE);
    enum_tags.insert("Some".into(), TAG_SOME);

    let mut next_rec_tag = TAG_RECORD_BASE;
    let mut next_enum_tag = TAG_ENUM_BASE;
    for item in &program.items {
        match item {
            Item::Record(r) if r.type_params.is_empty() => {
                let c_layout = if is_c_layout_attrs(&r.attrs) {
                    layout_engine.struct_layout(&r.name)
                } else {
                    None
                };
                let mut fields = Vec::new();
                let mut rc_mask = 0u64;
                for (idx, f) in r.fields.iter().enumerate() {
                    let tname = f.ty.display_name();
                    let mut preds = Vec::new();
                    if let Some(ap) = alias_preds.get(&tname) {
                        preds.push(ap.clone());
                    }
                    if let Some(fp) = &f.predicate {
                        preds.push(fp.clone());
                    }
                    let is_rc = c_layout.is_none()
                        && is_rc_type_expr(&f.ty, &alias_bases, &record_names, &enum_names);
                    if is_rc && idx < 64 {
                        rc_mask |= 1u64 << idx;
                    }
                    fields.push(LlvmFieldLayout {
                        name: f.name.clone(),
                        ty: f.ty.clone(),
                        preds,
                        is_rc,
                    });
                    field_to_idx.entry(f.name.clone()).or_insert(idx);
                }
                records.insert(
                    r.name.clone(),
                    LlvmRecordLayout {
                        tag: next_rec_tag,
                        fields,
                        rc_mask: rc_mask as i64,
                        where_clauses: r.where_clauses.clone(),
                        c_layout,
                    },
                );
                next_rec_tag += 1;
            }
            Item::Union(u) => {
                let c_layout = layout_engine.union_layout(&u.name);
                let mut fields = Vec::new();
                for (idx, f) in u.fields.iter().enumerate() {
                    fields.push(LlvmFieldLayout {
                        name: f.name.clone(),
                        ty: f.ty.clone(),
                        preds: Vec::new(),
                        is_rc: false,
                    });
                    field_to_idx.entry(f.name.clone()).or_insert(idx);
                }
                records.insert(
                    u.name.clone(),
                    LlvmRecordLayout {
                        tag: TAG_CSTRUCT,
                        fields,
                        rc_mask: 0,
                        where_clauses: Vec::new(),
                        c_layout,
                    },
                );
            }
            Item::ExternFn(ef) => {
                extern_fns.insert(ef.name.clone(), ef.clone());
            }
            Item::Enum(e) => {
                for v in &e.variants {
                    let payload_tys: Vec<TypeExpr> =
                        v.fields.iter().map(|f| f.ty.clone()).collect();
                    let vl = LlvmEnumVariantLayout {
                        enum_name: e.name.clone(),
                        payload_tys,
                    };
                    enum_tags.insert(v.name.clone(), next_enum_tag);
                    enum_tags.insert(format!("{}.{}", e.name, v.name), next_enum_tag);
                    enum_variants.insert(v.name.clone(), vl.clone());
                    enum_variants.insert(format!("{}.{}", e.name, v.name), vl);
                    next_enum_tag += 1;
                }
            }
            _ => {}
        }
    }

    let sync_fns: Vec<&FnDecl> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Fn(f) if !f.is_async && f.type_params.is_empty() => Some(f),
            _ => None,
        })
        .collect();
    let async_fns: Vec<&FnDecl> = program
        .items
        .iter()
        .filter_map(|i| match i {
            Item::Fn(f) if f.is_async && f.type_params.is_empty() => Some(f),
            _ => None,
        })
        .collect();

    let mut fn_metas: BTreeMap<String, LlvmFnMeta> = BTreeMap::new();
    let mut syscall_map: BTreeMap<String, &'static str> = BTreeMap::new();
    for spec in CORE_SYSCALLS {
        syscall_map.insert(spec.tenon_name.to_string(), spec.c_symbol);
        fn_metas.insert(
            spec.tenon_name.to_string(),
            LlvmFnMeta {
                params_inout: vec![false; spec.arity],
                param_tys: vec![named_ty("Any"); spec.arity],
                ret_ty: Some(parse_syscall_ret_ty(spec.ret_ty)),
                returns_rc: spec.returns_rc,
            },
        );
    }
    for (ename, ef) in &extern_fns {
        fn_metas.insert(
            ename.clone(),
            LlvmFnMeta {
                params_inout: vec![false; ef.params.len()],
                param_tys: ef.params.iter().map(|p| p.ty.clone()).collect(),
                ret_ty: ef.ret_ty.clone(),
                returns_rc: false,
            },
        );
    }
    for f in &sync_fns {
        fn_metas.insert(
            f.name.clone(),
            LlvmFnMeta {
                params_inout: f.params.iter().map(|p| p.is_inout).collect(),
                param_tys: f.params.iter().map(|p| p.ty.clone()).collect(),
                ret_ty: f.ret_ty.clone(),
                returns_rc: f.ret_ty.as_ref().is_some_and(|t| {
                    is_rc_type_expr(t, &alias_bases, &record_names, &enum_names)
                }),
            },
        );
    }
    for f in &async_fns {
        fn_metas.insert(
            f.name.clone(),
            LlvmFnMeta {
                params_inout: vec![false; f.params.len()],
                param_tys: f.params.iter().map(|p| p.ty.clone()).collect(),
                ret_ty: f.ret_ty.clone(),
                returns_rc: true,
            },
        );
    }
    for (rname, layout) in &records {
        if !layout.fields.is_empty() && layout.c_layout.is_none() {
            fn_metas.insert(
                format!("{}__parse", rname),
                LlvmFnMeta {
                    params_inout: vec![false; layout.fields.len()],
                    param_tys: layout.fields.iter().map(|f| f.ty.clone()).collect(),
                    ret_ty: Some(generic_ty("Result", vec![named_ty(rname), named_ty("Int")])),
                    returns_rc: true,
                },
            );
        }
    }

    let mut string_globals: Vec<(String, String)> = Vec::new();
    let mut string_intern: BTreeMap<String, String> = BTreeMap::new();
    let mut unit_enum_tags: BTreeSet<i64> = BTreeSet::new();
    let mut fn_defs = String::new();

    // DWARF debug info metadata builder (allocated IDs >= 100 so TBAA !0..!10 stay untouched!)
    let mut next_md_id: u32 = 110;
    let mut dbg_files: Vec<(u32, String, String)> = Vec::new();
    let mut dbg_subprograms: Vec<(u32, String, String, u32, u32)> = Vec::new();
    let mut dbg_locs: Vec<(u32, u32, u32, u32)> = Vec::new();
    if debug_info {
        let files = if program.source_files.is_empty() {
            vec!["main.tn".to_string()]
        } else {
            program.source_files.clone()
        };
        for fpath in files {
            let p = Path::new(&fpath);
            let fname = p
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| fpath.clone());
            let dir = p
                .parent()
                .map(|d| d.to_string_lossy().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| ".".to_string());
            let fid = next_md_id;
            next_md_id += 1;
            dbg_files.push((fid, fname, dir));
        }
    }

    for f in &sync_fns {
        let sp_id = if debug_info {
            let id = next_md_id;
            next_md_id += 1;
            let file_md = dbg_files
                .get(f.span.file_id as usize)
                .or_else(|| dbg_files.first())
                .map(|(id, _, _)| *id)
                .unwrap_or(110);
            let line = f.span.line.max(1) as u32;
            dbg_subprograms.push((
                id,
                f.name.clone(),
                format!("tenon_fn_{}", sanitize(&f.name)),
                file_md,
                line,
            ));
            Some(id)
        } else {
            None
        };
        let mut cg = LlvmFnGen::new(
            &alias_bases,
            &records,
            &record_names,
            &enum_names,
            &field_to_idx,
            &enum_tags,
            &enum_variants,
            &fn_metas,
            &syscall_map,
            &layout_engine,
            &extern_fns,
            dyn_interfaces,
            &vtable_ifaces,
            proved_safe_indices,
            proved_no_overflow,
            non_escaping_vars,
            stack_promotable_vars,
            last_use_move_spans,
            checked_overflow,
            &mut string_globals,
            &mut string_intern,
            &mut unit_enum_tags,
            sp_id,
            &mut dbg_locs,
            &mut next_md_id,
        );
        fn_defs.push_str(&cg.emit_sync_fn(f)?);
        fn_defs.push('\n');
        if let Some(cname) = extract_export_name(&f.attrs, &f.name) {
            fn_defs.push_str(&emit_export_wrapper(f, &cname, &alias_bases));
            fn_defs.push('\n');
        }
    }

    for (rname, layout) in &records {
        if layout.fields.is_empty() || layout.c_layout.is_some() {
            continue;
        }
        let mut cg = LlvmFnGen::new(
            &alias_bases,
            &records,
            &record_names,
            &enum_names,
            &field_to_idx,
            &enum_tags,
            &enum_variants,
            &fn_metas,
            &syscall_map,
            &layout_engine,
            &extern_fns,
            dyn_interfaces,
            &vtable_ifaces,
            proved_safe_indices,
            proved_no_overflow,
            non_escaping_vars,
            stack_promotable_vars,
            last_use_move_spans,
            checked_overflow,
            &mut string_globals,
            &mut string_intern,
            &mut unit_enum_tags,
            None,
            &mut dbg_locs,
            &mut next_md_id,
        );
        fn_defs.push_str(&cg.emit_record_parser(rname, layout)?);
        fn_defs.push('\n');
    }

    for f in &async_fns {
        let mut cg = LlvmFnGen::new(
            &alias_bases,
            &records,
            &record_names,
            &enum_names,
            &field_to_idx,
            &enum_tags,
            &enum_variants,
            &fn_metas,
            &syscall_map,
            &layout_engine,
            &extern_fns,
            dyn_interfaces,
            &vtable_ifaces,
            proved_safe_indices,
            proved_no_overflow,
            non_escaping_vars,
            stack_promotable_vars,
            last_use_move_spans,
            checked_overflow,
            &mut string_globals,
            &mut string_intern,
            &mut unit_enum_tags,
            None,
            &mut dbg_locs,
            &mut next_md_id,
        );
        fn_defs.push_str(&cg.emit_async_fn(f)?);
        fn_defs.push('\n');
    }

    let uses_sync = ir_uses_sync_runtime(&fn_defs);

    let mut ir = String::new();
    ir.push_str("; Generated by Tenon LLVM 22 AOT Release Backend\n");
    if let Some(triple) = target_triple {
        ir.push_str(&format!("target triple = \"{}\"\n", triple));
    }
    ir.push('\n');
    ir.push_str("declare void @llvm.assume(i1)\n");
    ir.push_str("declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)\n");
    ir.push_str("declare i32 @memcmp(ptr, ptr, i64)\n");
    // Emit only the LLVM bit/saturating/SIMD intrinsics actually referenced in `fn_defs`
    let optional_intrinsics: &[(&str, &str)] = &[
        ("@llvm.ctlz.i64", "declare i64 @llvm.ctlz.i64(i64, i1)\n"),
        ("@llvm.cttz.i64", "declare i64 @llvm.cttz.i64(i64, i1)\n"),
        ("@llvm.ctpop.i64", "declare i64 @llvm.ctpop.i64(i64)\n"),
        ("@llvm.bswap.i16", "declare i16 @llvm.bswap.i16(i16)\n"),
        ("@llvm.bswap.i32", "declare i32 @llvm.bswap.i32(i32)\n"),
        ("@llvm.bswap.i64", "declare i64 @llvm.bswap.i64(i64)\n"),
        ("@llvm.fshl.i64", "declare i64 @llvm.fshl.i64(i64, i64, i64)\n"),
        ("@llvm.fshr.i64", "declare i64 @llvm.fshr.i64(i64, i64, i64)\n"),
        ("@llvm.sadd.sat.i8", "declare i8 @llvm.sadd.sat.i8(i8, i8)\n"),
        ("@llvm.uadd.sat.i8", "declare i8 @llvm.uadd.sat.i8(i8, i8)\n"),
        ("@llvm.ssub.sat.i8", "declare i8 @llvm.ssub.sat.i8(i8, i8)\n"),
        ("@llvm.usub.sat.i8", "declare i8 @llvm.usub.sat.i8(i8, i8)\n"),
        ("@llvm.sadd.sat.i16", "declare i16 @llvm.sadd.sat.i16(i16, i16)\n"),
        ("@llvm.uadd.sat.i16", "declare i16 @llvm.uadd.sat.i16(i16, i16)\n"),
        ("@llvm.ssub.sat.i16", "declare i16 @llvm.ssub.sat.i16(i16, i16)\n"),
        ("@llvm.usub.sat.i16", "declare i16 @llvm.usub.sat.i16(i16, i16)\n"),
        ("@llvm.sadd.sat.i32", "declare i32 @llvm.sadd.sat.i32(i32, i32)\n"),
        ("@llvm.uadd.sat.i32", "declare i32 @llvm.uadd.sat.i32(i32, i32)\n"),
        ("@llvm.ssub.sat.i32", "declare i32 @llvm.ssub.sat.i32(i32, i32)\n"),
        ("@llvm.usub.sat.i32", "declare i32 @llvm.usub.sat.i32(i32, i32)\n"),
        ("@llvm.sadd.sat.i64", "declare i64 @llvm.sadd.sat.i64(i64, i64)\n"),
        ("@llvm.uadd.sat.i64", "declare i64 @llvm.uadd.sat.i64(i64, i64)\n"),
        ("@llvm.ssub.sat.i64", "declare i64 @llvm.ssub.sat.i64(i64, i64)\n"),
        ("@llvm.usub.sat.i64", "declare i64 @llvm.usub.sat.i64(i64, i64)\n"),
        ("@llvm.fma.v4f32", "declare <4 x float> @llvm.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)\n"),
        ("@llvm.fma.v4f64", "declare <4 x double> @llvm.fma.v4f64(<4 x double>, <4 x double>, <4 x double>)\n"),
        ("@llvm.minnum.v4f32", "declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>)\n"),
        ("@llvm.maxnum.v4f32", "declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>)\n"),
        ("@llvm.minnum.v4f64", "declare <4 x double> @llvm.minnum.v4f64(<4 x double>, <4 x double>)\n"),
        ("@llvm.maxnum.v4f64", "declare <4 x double> @llvm.maxnum.v4f64(<4 x double>, <4 x double>)\n"),
        ("@llvm.vector.reduce.fadd.v4f32", "declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>)\n"),
        ("@llvm.vector.reduce.fadd.v4f64", "declare double @llvm.vector.reduce.fadd.v4f64(double, <4 x double>)\n"),
        ("@llvm.vector.reduce.add.v4i32", "declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>)\n"),
        ("@llvm.vector.reduce.add.v4i64", "declare i64 @llvm.vector.reduce.add.v4i64(<4 x i64>)\n"),
        ("@llvm.sadd.with.overflow.i64", "declare { i64, i1 } @llvm.sadd.with.overflow.i64(i64, i64)\n"),
        ("@llvm.ssub.with.overflow.i64", "declare { i64, i1 } @llvm.ssub.with.overflow.i64(i64, i64)\n"),
        ("@llvm.smul.with.overflow.i64", "declare { i64, i1 } @llvm.smul.with.overflow.i64(i64, i64)\n"),
        ("@llvm.vector.reduce.add.v16i8", "declare i8 @llvm.vector.reduce.add.v16i8(<16 x i8>)\n"),
    ];
    for (needle, decl) in optional_intrinsics {
        if fn_defs.contains(needle) {
            ir.push_str(decl);
        }
    }
    ir.push_str("declare void @tenon_rt_assert_fail(i64) cold\n");
    ir.push_str("declare void @tenon_rt_overflow_trap(i64) cold\n");
    ir.push_str("declare i64 @tenon_rt_print_int(i64)\n");
    ir.push_str("declare i64 @tenon_rc_alloc(i64, i64)\n");
    ir.push_str("declare void @tenon_rc_set_rc_mask(i64, i64)\n");
    ir.push_str("declare i64 @tenon_rc_inc(i64)\n");
    ir.push_str("declare void @tenon_rc_dec(i64)\n");
    ir.push_str("declare void @tenon_rc_free_obj(i64) cold\n");
    ir.push_str("declare i64 @tenon_rc_ensure_unique(i64) cold\n");
    ir.push_str("declare i64 @tenon_rc_get_field(i64, i64)\n");
    ir.push_str("declare void @tenon_rc_set_field(i64, i64, i64)\n");
    ir.push_str("declare void @tenon_rc_set_field_raw(i64, i64, i64)\n");
    ir.push_str("declare i64 @tenon_rc_get_tag(i64)\n");
    ir.push_str("declare i64 @tenon_rc_get_len(i64)\n");
    ir.push_str("declare i64 @tenon_text_new(i64, i64)\n");
    ir.push_str("declare i64 @tenon_text_eq(i64, i64)\n");
    ir.push_str("declare i64 @tenon_list_push(i64, i64)\n");
    ir.push_str("declare i64 @tenon_list_filled(i64, i64, i64)\n");
    ir.push_str("declare i64 @tenon_flat_list_filled(i64, i64, ptr)\n");
    ir.push_str("declare i64 @tenon_flat_list_push(i64, i64, ptr)\n");
    ir.push_str("declare i64 @tenon_arena_enter()\n");
    ir.push_str("declare void @tenon_arena_exit(i64)\n");
    ir.push_str("declare i64 @tenon_list_pop_shrink(i64)\n");
    ir.push_str("declare i64 @tenon_async_frame_alloc(i64, i64)\n");
    ir.push_str("declare i64 @tenon_async_yield_create(i64)\n");
    ir.push_str("declare i64 @tenon_async_poll_frame(i64)\n");
    ir.push_str("declare i64 @tenon_async_block_on(i64)\n");
    ir.push_str("declare i64 @tenon_cstruct_alloc(i64)\n");
    ir.push_str("declare i64 @tenon_cstruct_data_ptr(i64)\n");
    if fn_defs.contains("@tenon_preempt_flag") {
        ir.push_str("@tenon_preempt_flag = external global i32\n");
        ir.push_str("declare void @tenon_sched_preempt_point() cold\n");
    }
    for spec in CORE_SYSCALLS {
        let params = vec!["i64"; spec.arity].join(", ");
        ir.push_str(&format!("declare i64 @{}({})\n", spec.c_symbol, params));
    }
    for ef in extern_fns.values() {
        let ret_llvm = match &ef.ret_ty {
            Some(rt) => llvm_c_abi_type(rt, &alias_bases),
            None => "void",
        };
        let mut ptypes: Vec<&str> = ef
            .params
            .iter()
            .map(|p| llvm_c_abi_type(&p.ty, &alias_bases))
            .collect();
        if ef.is_variadic {
            ptypes.push("...");
        }
        ir.push_str(&format!(
            "declare {} @{}({})\n",
            ret_llvm,
            ef.name,
            ptypes.join(", ")
        ));
    }
    if uses_sync {
        ir.push_str(TENON_ATOMIC_RC_IR_HELPERS);
    } else {
        ir.push_str(TENON_NONATOMIC_RC_IR_HELPERS);
    }
    ir.push_str(TENON_INLINE_IR_HELPERS);
    ir.push('\n');

    for (gname, text) in &string_globals {
        let blen = text.len();
        let words = ((blen + 8) / 8).max(1);
        if blen == 0 {
            ir.push_str(&format!(
                "@{} = private unnamed_addr constant {{ i64, i64, i64, i64, i64, [1 x i8] }} {{ i64 -1, i64 5, i64 0, i64 1, i64 0, [1 x i8] zeroinitializer }}, align 8\n",
                gname
            ));
        } else {
            let escaped = escape_llvm_bytes(text.as_bytes());
            ir.push_str(&format!(
                "@{} = private unnamed_addr constant {{ i64, i64, i64, i64, i64, [{} x i8] }} {{ i64 -1, i64 5, i64 {}, i64 {}, i64 0, [{} x i8] c\"{}\\00\" }}, align 8\n",
                gname,
                blen + 1,
                blen,
                words,
                blen + 1,
                escaped
            ));
        }
    }
    if !string_globals.is_empty() {
        ir.push('\n');
    }

    for &tag in &unit_enum_tags {
        ir.push_str(&format!(
            "@.unit_enum.{} = private unnamed_addr constant {{ i64, i64, i64, i64, i64, i64 }} {{ i64 -1, i64 {}, i64 0, i64 1, i64 0, i64 0 }}, align 8\n",
            tag, tag
        ));
    }
    if !unit_enum_tags.is_empty() {
        ir.push('\n');
    }

    for vt in &prepared.vtables {
        let entries: Vec<String> = vt
            .fn_names
            .iter()
            .map(|fn_name| format!("ptr @tenon_fn_{}", sanitize(fn_name)))
            .collect();
        ir.push_str(&format!(
            "@{} = private constant [{} x ptr] [{}]\n",
            sanitize(&vt.sym_name),
            entries.len(),
            entries.join(", ")
        ));
    }
    if !prepared.vtables.is_empty() {
        ir.push('\n');
    }

    ir.push_str(&fn_defs);
    ir.push_str(TENON_TBAA_METADATA);

    if debug_info {
        let primary_file_id = dbg_files.first().map(|(id, _, _)| *id).unwrap_or(110);
        ir.push_str("\n!llvm.dbg.cu = !{!100}\n");
        ir.push_str("!llvm.module.flags = !{!101, !102}\n");
        ir.push_str(&format!(
            "!100 = distinct !DICompileUnit(language: DW_LANG_C99, file: !{}, producer: \"Tenon LLVM 22 AOT\", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)\n",
            primary_file_id
        ));
        ir.push_str("!101 = !{i32 7, !\"Dwarf Version\", i32 4}\n");
        ir.push_str("!102 = !{i32 2, !\"Debug Info Version\", i32 3}\n");
        ir.push_str("!103 = !DISubroutineType(types: !104)\n");
        ir.push_str("!104 = !{null}\n");
        for (fid, fname, dir) in &dbg_files {
            ir.push_str(&format!(
                "!{} = !DIFile(filename: \"{}\", directory: \"{}\")\n",
                fid,
                fname.replace('\\', "\\\\").replace('"', "\\\""),
                dir.replace('\\', "\\\\").replace('"', "\\\"")
            ));
        }
        for (sp_id, name, linkage, file_id, line) in &dbg_subprograms {
            ir.push_str(&format!(
                "!{} = distinct !DISubprogram(name: \"{}\", linkageName: \"{}\", scope: !{}, file: !{}, line: {}, type: !103, scopeLine: {}, spFlags: DISPFlagDefinition, unit: !100)\n",
                sp_id, name, linkage, file_id, file_id, line, line
            ));
        }
        for (loc_id, line, col, scope_id) in &dbg_locs {
            ir.push_str(&format!(
                "!{} = !DILocation(line: {}, column: {}, scope: !{})\n",
                loc_id, line, col, scope_id
            ));
        }
    }

    Ok(ir)
}

fn escape_llvm_bytes(bytes: &[u8]) -> String {
    let mut out = String::new();
    for &b in bytes {
        if (0x20..=0x7E).contains(&b) && b != b'"' && b != b'\\' {
            out.push(b as char);
        } else {
            out.push_str(&format!("\\{:02X}", b));
        }
    }
    out
}

struct LlvmLoopCtx {
    hdr_lbl: String,
    exit_lbl: String,
    rc_scope_depth: usize,
    temp_scope_depth: usize,
}

struct LlvmSavedLexical {
    var_ptrs: BTreeMap<String, String>,
    var_types: BTreeMap<String, TypeExpr>,
    owned_rc_vars: BTreeSet<String>,
    borrowed_rc_vars: BTreeSet<String>,
}

struct LlvmFnGen<'a> {
    alias_bases: &'a BTreeMap<String, TypeExpr>,
    records: &'a BTreeMap<String, LlvmRecordLayout>,
    record_names: &'a BTreeSet<String>,
    enum_names: &'a BTreeSet<String>,
    field_to_idx: &'a BTreeMap<String, usize>,
    enum_tags: &'a BTreeMap<String, i64>,
    enum_variants: &'a BTreeMap<String, LlvmEnumVariantLayout>,
    fn_metas: &'a BTreeMap<String, LlvmFnMeta>,
    syscall_map: &'a BTreeMap<String, &'static str>,
    layout_engine: &'a LayoutEngine,
    extern_fns: &'a BTreeMap<String, ExternFnDecl>,
    dyn_interfaces: &'a BTreeMap<String, DynInterfaceLayout>,
    vtable_ifaces: &'a BTreeMap<String, String>,
    proved_safe_indices: &'a BTreeSet<usize>,
    proved_no_overflow: &'a BTreeSet<usize>,
    non_escaping_vars: &'a BTreeMap<String, BTreeSet<String>>,
    stack_promotable_vars: &'a BTreeMap<String, BTreeSet<String>>,
    last_use_move_spans: &'a BTreeMap<String, BTreeSet<usize>>,
    checked_overflow: bool,
    string_globals: &'a mut Vec<(String, String)>,
    string_intern: &'a mut BTreeMap<String, String>,
    unit_enum_tags: &'a mut BTreeSet<i64>,
    debug_sp_id: Option<u32>,
    dbg_locs: &'a mut Vec<(u32, u32, u32, u32)>,
    next_md_id: &'a mut u32,
    tmp: usize,
    lbl: usize,
    slot_id: usize,
    entry_allocas: String,
    code: String,
    var_ptrs: BTreeMap<String, String>,
    var_types: BTreeMap<String, TypeExpr>,
    inout_vars: BTreeSet<String>,
    owned_rc_vars: BTreeSet<String>,
    borrowed_rc_vars: BTreeSet<String>,
    always_unique_vars: BTreeSet<String>,
    cur_stack_promotable: BTreeSet<String>,
    cur_last_use_move_spans: BTreeSet<usize>,
    active_arena_mark: Option<String>,
    rc_scopes: Vec<Vec<(String, String)>>,
    temp_rc_scopes: Vec<Vec<String>>,
    loop_stack: Vec<LlvmLoopCtx>,
    fn_returns_rc: bool,
    fn_ret_ty: Option<TypeExpr>,
    self_var: Option<String>,
    cur_fn_preemptible: bool,
    terminated: bool,
}

impl<'a> LlvmFnGen<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        alias_bases: &'a BTreeMap<String, TypeExpr>,
        records: &'a BTreeMap<String, LlvmRecordLayout>,
        record_names: &'a BTreeSet<String>,
        enum_names: &'a BTreeSet<String>,
        field_to_idx: &'a BTreeMap<String, usize>,
        enum_tags: &'a BTreeMap<String, i64>,
        enum_variants: &'a BTreeMap<String, LlvmEnumVariantLayout>,
        fn_metas: &'a BTreeMap<String, LlvmFnMeta>,
        syscall_map: &'a BTreeMap<String, &'static str>,
        layout_engine: &'a LayoutEngine,
        extern_fns: &'a BTreeMap<String, ExternFnDecl>,
        dyn_interfaces: &'a BTreeMap<String, DynInterfaceLayout>,
        vtable_ifaces: &'a BTreeMap<String, String>,
        proved_safe_indices: &'a BTreeSet<usize>,
        proved_no_overflow: &'a BTreeSet<usize>,
        non_escaping_vars: &'a BTreeMap<String, BTreeSet<String>>,
        stack_promotable_vars: &'a BTreeMap<String, BTreeSet<String>>,
        last_use_move_spans: &'a BTreeMap<String, BTreeSet<usize>>,
        checked_overflow: bool,
        string_globals: &'a mut Vec<(String, String)>,
        string_intern: &'a mut BTreeMap<String, String>,
        unit_enum_tags: &'a mut BTreeSet<i64>,
        debug_sp_id: Option<u32>,
        dbg_locs: &'a mut Vec<(u32, u32, u32, u32)>,
        next_md_id: &'a mut u32,
    ) -> Self {
        Self {
            alias_bases,
            records,
            record_names,
            enum_names,
            field_to_idx,
            enum_tags,
            enum_variants,
            fn_metas,
            syscall_map,
            layout_engine,
            extern_fns,
            dyn_interfaces,
            vtable_ifaces,
            proved_safe_indices,
            proved_no_overflow,
            non_escaping_vars,
            stack_promotable_vars,
            last_use_move_spans,
            checked_overflow,
            string_globals,
            string_intern,
            unit_enum_tags,
            debug_sp_id,
            dbg_locs,
            next_md_id,
            tmp: 0,
            lbl: 0,
            slot_id: 0,
            entry_allocas: String::new(),
            code: String::new(),
            var_ptrs: BTreeMap::new(),
            var_types: BTreeMap::new(),
            inout_vars: BTreeSet::new(),
            owned_rc_vars: BTreeSet::new(),
            borrowed_rc_vars: BTreeSet::new(),
            always_unique_vars: BTreeSet::new(),
            cur_stack_promotable: BTreeSet::new(),
            cur_last_use_move_spans: BTreeSet::new(),
            active_arena_mark: None,
            rc_scopes: Vec::new(),
            temp_rc_scopes: Vec::new(),
            loop_stack: Vec::new(),
            fn_returns_rc: false,
            fn_ret_ty: None,
            self_var: None,
            cur_fn_preemptible: false,
            terminated: false,
        }
    }

    fn save_lexical(&self) -> LlvmSavedLexical {
        LlvmSavedLexical {
            var_ptrs: self.var_ptrs.clone(),
            var_types: self.var_types.clone(),
            owned_rc_vars: self.owned_rc_vars.clone(),
            borrowed_rc_vars: self.borrowed_rc_vars.clone(),
        }
    }

    fn restore_lexical(&mut self, saved: LlvmSavedLexical) {
        self.var_ptrs = saved.var_ptrs;
        self.var_types = saved.var_types;
        self.owned_rc_vars = saved.owned_rc_vars;
        self.borrowed_rc_vars = saved.borrowed_rc_vars;
    }

    fn fresh_tmp(&mut self) -> String {
        self.tmp += 1;
        format!("%t{}", self.tmp)
    }

    fn fresh_lbl(&mut self, prefix: &str) -> String {
        self.lbl += 1;
        format!("{}_{}", prefix, self.lbl)
    }

    fn fresh_var_slot(&mut self, name: &str) -> String {
        self.slot_id += 1;
        let slot = format!("%v_{}_{}", sanitize(name), self.slot_id);
        self.entry_allocas
            .push_str(&format!("  {} = alloca i64\n  store i64 0, ptr {}\n", slot, slot));
        self.var_ptrs.insert(name.to_string(), slot.clone());
        slot
    }

    fn ensure_var_slot(&mut self, name: &str) -> String {
        if let Some(p) = self.var_ptrs.get(name) {
            return p.clone();
        }
        self.fresh_var_slot(name)
    }

    fn intern_string_global(&mut self, s: &str) -> String {
        if let Some(g) = self.string_intern.get(s) {
            return g.clone();
        }
        let gname = format!(".str.{}", self.string_globals.len());
        self.string_globals.push((gname.clone(), s.to_string()));
        self.string_intern.insert(s.to_string(), gname.clone());
        gname
    }

    fn emit_unit_enum_const(&mut self, tag: i64) -> String {
        self.unit_enum_tags.insert(tag);
        let o = self.fresh_tmp();
        self.code.push_str(&format!(
            "  {} = ptrtoint ptr @.unit_enum.{} to i64\n",
            o, tag
        ));
        o
    }

    fn flush_temps(&mut self, temps: &[String]) {
        for v in temps.iter().rev() {
            self.code
                .push_str(&format!("  call void @__tn_rc_dec(i64 {})\n", v));
        }
    }

    fn emit_cleanup_scopes_from(&mut self, from_depth: usize, skip_slot: Option<&str>) {
        let scopes: Vec<Vec<(String, String)>> = self.rc_scopes[from_depth..].to_vec();
        for scope in scopes.iter().rev() {
            for (_, slot) in scope.iter().rev() {
                if skip_slot == Some(slot.as_str()) {
                    continue;
                }
                let h = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = load i64, ptr {}\n  call void @__tn_rc_dec(i64 {})\n",
                    h, slot, h
                ));
            }
        }
    }

    fn emit_cleanup_temp_scopes_from(&mut self, from_depth: usize) {
        let scopes: Vec<Vec<String>> = self.temp_rc_scopes[from_depth..].to_vec();
        for scope in scopes.iter().rev() {
            for val in scope.iter().rev() {
                self.code
                    .push_str(&format!("  call void @__tn_rc_dec(i64 {})\n", val));
            }
        }
    }

    fn find_owned_slot_in_scopes(&self, slot: &str, from_depth: usize) -> bool {
        self.rc_scopes[from_depth..]
            .iter()
            .any(|sc| sc.iter().any(|(_, s)| s == slot))
    }

    fn type_is_rc(&self, ty: &TypeExpr) -> bool {
        is_rc_type_expr(ty, self.alias_bases, self.record_names, self.enum_names)
    }

    fn dbg_suffix(&mut self, span: Span) -> String {
        if let Some(sp_id) = self.debug_sp_id {
            let line = span.line.max(1) as u32;
            let col = span.col.max(1) as u32;
            let loc_id = *self.next_md_id;
            *self.next_md_id += 1;
            self.dbg_locs.push((loc_id, line, col, sp_id));
            format!(", !dbg !{}", loc_id)
        } else {
            String::new()
        }
    }

    fn list_elem_ty(&self, ty: &TypeExpr) -> Option<TypeExpr> {
        let resolved = resolve_alias_ty(ty, self.alias_bases);
        match resolved {
            TypeExpr::Array { elem, .. } => Some(*elem),
            TypeExpr::Named { name, args, .. } if name == "List" && !args.is_empty() => {
                Some(args[0].clone())
            }
            _ => None,
        }
    }

    fn array_layout(&self, ty: &TypeExpr) -> Option<(TypeExpr, usize, usize, bool)> {
        if let TypeExpr::Array { elem, len, .. } = resolve_alias_ty(ty, self.alias_bases) {
            let is_rc = self.type_is_rc(&elem) && !self.is_c_value_type(&elem);
            let stride = if is_rc {
                8
            } else {
                self.layout_engine.size_and_align(&elem).0.max(1)
            };
            return Some((*elem, len, stride, is_rc));
        }
        None
    }

    fn is_c_value_type(&self, ty: &TypeExpr) -> bool {
        match resolve_alias_ty(ty, self.alias_bases) {
            TypeExpr::Named { name, .. } => self
                .records
                .get(&name)
                .is_some_and(|r| r.c_layout.is_some()),
            TypeExpr::Array { elem, .. } => !self.type_is_rc(&elem) || self.is_c_value_type(&elem),
            _ => false,
        }
    }

    fn ptr_pointee_ty(&self, ty: &TypeExpr) -> Option<TypeExpr> {
        if let TypeExpr::Named { name, args, .. } = resolve_alias_ty(ty, self.alias_bases) {
            if (name == "Ptr" || name == "MutPtr") && !args.is_empty() {
                return Some(args[0].clone());
            }
            if name == "CStr" {
                return Some(named_ty("UInt8"));
            }
        }
        None
    }

    fn scalar_kind_of_expr(&self, expr: &Expr) -> Option<ScalarKind> {
        let ty = self.infer_expr_type(expr);
        if let Some(sk) = scalar_kind_of_ty(&ty, self.alias_bases) {
            return Some(sk);
        }
        match &expr.kind {
            ExprKind::Float(_) => Some(ScalarKind::F64),
            ExprKind::Cast { ty, .. } => scalar_kind_of_ty(ty, self.alias_bases),
            ExprKind::TryCast { .. } => None,
            ExprKind::Unary { op: UnOp::Neg | UnOp::BitNot, expr: inner } => {
                self.scalar_kind_of_expr(inner)
            }
            ExprKind::Binary { op, left, right }
                if matches!(
                    op,
                    BinOp::Add
                        | BinOp::Sub
                        | BinOp::Mul
                        | BinOp::Div
                        | BinOp::Mod
                        | BinOp::BitAnd
                        | BinOp::BitOr
                        | BinOp::BitXor
                        | BinOp::Shl
                        | BinOp::Shr
                ) =>
            {
                let lk = self.scalar_kind_of_expr(left);
                let rk = self.scalar_kind_of_expr(right);
                match (lk, rk) {
                    (Some(ScalarKind::I64), Some(k)) if k != ScalarKind::I64 && matches!(left.kind, ExprKind::Int(_)) => Some(k),
                    (Some(k), _) => Some(k),
                    (None, k) => k,
                }
            }
            _ => None,
        }
    }

    fn normalize_scalar(&mut self, val: &str, kind: ScalarKind) -> String {
        match kind {
            ScalarKind::I8 | ScalarKind::U8 | ScalarKind::I16 | ScalarKind::U16 | ScalarKind::I32 | ScalarKind::U32 => {
                let ity = match kind {
                    ScalarKind::I8 | ScalarKind::U8 => "i8",
                    ScalarKind::I16 | ScalarKind::U16 => "i16",
                    _ => "i32",
                };
                let ext = if kind.is_unsigned() { "zext" } else { "sext" };
                let tr = self.fresh_tmp();
                let ex = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = trunc i64 {} to {}\n  {} = {} {} {} to i64\n",
                    tr, val, ity, ex, ext, ity, tr
                ));
                ex
            }
            ScalarKind::I64 | ScalarKind::U64 | ScalarKind::F16 | ScalarKind::BF16 | ScalarKind::F32 | ScalarKind::F64 => val.to_string(),
        }
    }

    fn emit_mem_load(
        &mut self,
        base_ptr_i64: &str,
        byte_off: i64,
        ty: &TypeExpr,
        volatile: bool,
    ) -> Result<String, String> {
        let resolved = resolve_alias_ty(ty, self.alias_bases);
        let bp = self.fresh_tmp();
        self.code
            .push_str(&format!("  {} = inttoptr i64 {} to ptr\n", bp, base_ptr_i64));
        let ptr = if byte_off == 0 {
            bp
        } else {
            let gp = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = getelementptr inbounds i8, ptr {}, i64 {}\n",
                gp, bp, byte_off
            ));
            gp
        };
        if self.is_c_value_type(&resolved) {
            let (sz, _) = self.layout_engine.size_and_align(&resolved);
            let obj = self.fresh_tmp();
            let dst_i = self.fresh_tmp();
            let dst_p = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = call i64 @tenon_cstruct_alloc(i64 {})\n  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n  {} = inttoptr i64 {} to ptr\n  call void @llvm.memcpy.p0.p0.i64(ptr {}, ptr {}, i64 {}, i1 false)\n",
                obj, sz, dst_i, obj, dst_p, dst_i, dst_p, ptr, sz
            ));
            return Ok(obj);
        }
        let vol = if volatile { "volatile " } else { "" };
        let sk = scalar_kind_of_ty(&resolved, self.alias_bases);
        let (ity, ext) = match sk {
            Some(ScalarKind::I8) => ("i8", Some("sext")),
            Some(ScalarKind::U8) => ("i8", Some("zext")),
            Some(ScalarKind::I16) => ("i16", Some("sext")),
            Some(ScalarKind::U16 | ScalarKind::F16 | ScalarKind::BF16) => ("i16", Some("zext")),
            Some(ScalarKind::I32) => ("i32", Some("sext")),
            Some(ScalarKind::U32 | ScalarKind::F32) => ("i32", Some("zext")),
            None if matches!(&resolved, TypeExpr::Named { name, .. } if name == "Bool") => {
                ("i8", Some("zext"))
            }
            _ => ("i64", None),
        };
        let ld = self.fresh_tmp();
        self.code.push_str(&format!(
            "  {} = load {}{}, ptr {}, align 1\n",
            ld, vol, ity, ptr
        ));
        if let Some(ex_op) = ext {
            let out = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = {} {} {} to i64\n",
                out, ex_op, ity, ld
            ));
            Ok(out)
        } else {
            Ok(ld)
        }
    }

    fn emit_mem_store(
        &mut self,
        base_ptr_i64: &str,
        byte_off: i64,
        val_i64: &str,
        ty: &TypeExpr,
        volatile: bool,
    ) -> Result<(), String> {
        let resolved = resolve_alias_ty(ty, self.alias_bases);
        let bp = self.fresh_tmp();
        self.code
            .push_str(&format!("  {} = inttoptr i64 {} to ptr\n", bp, base_ptr_i64));
        let ptr = if byte_off == 0 {
            bp
        } else {
            let gp = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = getelementptr inbounds i8, ptr {}, i64 {}\n",
                gp, bp, byte_off
            ));
            gp
        };
        if self.is_c_value_type(&resolved) {
            let (sz, _) = self.layout_engine.size_and_align(&resolved);
            let src_i = self.fresh_tmp();
            let src_p = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n  {} = inttoptr i64 {} to ptr\n  call void @llvm.memcpy.p0.p0.i64(ptr {}, ptr {}, i64 {}, i1 false)\n",
                src_i, val_i64, src_p, src_i, ptr, src_p, sz
            ));
            return Ok(());
        }
        let vol = if volatile { "volatile " } else { "" };
        let sk = scalar_kind_of_ty(&resolved, self.alias_bases);
        let ity = match sk {
            Some(ScalarKind::I8 | ScalarKind::U8) => "i8",
            Some(ScalarKind::I16 | ScalarKind::U16 | ScalarKind::F16 | ScalarKind::BF16) => "i16",
            Some(ScalarKind::I32 | ScalarKind::U32 | ScalarKind::F32) => "i32",
            None if matches!(&resolved, TypeExpr::Named { name, .. } if name == "Bool") => "i8",
            _ => "i64",
        };
        if ity == "i64" {
            self.code.push_str(&format!(
                "  store {}i64 {}, ptr {}, align 1\n",
                vol, val_i64, ptr
            ));
        } else {
            let tr = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = trunc i64 {} to {}\n  store {}{} {}, ptr {}, align 1\n",
                tr, val_i64, ity, vol, ity, tr, ptr
            ));
        }
        Ok(())
    }

    fn emit_f16_to_float(&mut self, val_i64: &str) -> String {
        let t = self.fresh_tmp();
        let h = self.fresh_tmp();
        let f = self.fresh_tmp();
        self.code.push_str(&format!(
            "  {} = trunc i64 {} to i16\n  {} = bitcast i16 {} to half\n  {} = fpext half {} to float\n",
            t, val_i64, h, t, f, h
        ));
        f
    }

    fn emit_float_to_f16(&mut self, val_f32: &str) -> String {
        let h = self.fresh_tmp();
        let b = self.fresh_tmp();
        let z = self.fresh_tmp();
        self.code.push_str(&format!(
            "  {} = fptrunc float {} to half\n  {} = bitcast half {} to i16\n  {} = zext i16 {} to i64\n",
            h, val_f32, b, h, z, b
        ));
        z
    }

    fn emit_bf16_to_float(&mut self, val_i64: &str) -> String {
        let t = self.fresh_tmp();
        let z = self.fresh_tmp();
        let s = self.fresh_tmp();
        let f = self.fresh_tmp();
        self.code.push_str(&format!(
            "  {} = trunc i64 {} to i16\n  {} = zext i16 {} to i32\n  {} = shl i32 {}, 16\n  {} = bitcast i32 {} to float\n",
            t, val_i64, z, t, s, z, f, s
        ));
        f
    }

    fn emit_float_to_bf16(&mut self, val_f32: &str) -> String {
        let b = self.fresh_tmp();
        let hi = self.fresh_tmp();
        let lsb = self.fresh_tmp();
        let bias = self.fresh_tmp();
        let rnd = self.fresh_tmp();
        let sh = self.fresh_tmp();
        let m = self.fresh_tmp();
        let z = self.fresh_tmp();
        self.code.push_str(&format!(
            "  {} = bitcast float {} to i32\n  {} = lshr i32 {}, 16\n  {} = and i32 {}, 1\n  {} = add i32 {}, 32767\n  {} = add i32 {}, {}\n  {} = lshr i32 {}, 16\n  {} = and i32 {}, 65535\n  {} = zext i32 {} to i64\n",
            b, val_f32, hi, b, lsb, hi, bias, lsb, rnd, b, bias, sh, rnd, m, sh, z, m
        ));
        z
    }

    fn emit_cast(&mut self, val: &str, src_ty: &TypeExpr, dst_ty: &TypeExpr) -> String {
        let src_base = resolve_alias_ty(src_ty, self.alias_bases);
        let dst_base = resolve_alias_ty(dst_ty, self.alias_bases);
        let src_sk = scalar_kind_of_ty(&src_base, self.alias_bases);
        let dst_sk = scalar_kind_of_ty(&dst_base, self.alias_bases);

        if matches!(&dst_base, TypeExpr::Named { name, .. } if name == "Bool") {
            let c = self.fresh_tmp();
            let z = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = icmp ne i64 {}, 0\n  {} = zext i1 {} to i64\n",
                c, val, z, c
            ));
            return z;
        }
        match (src_sk, dst_sk) {
            (Some(ScalarKind::F16), Some(ScalarKind::F16))
            | (Some(ScalarKind::BF16), Some(ScalarKind::BF16)) => val.to_string(),
            (Some(ScalarKind::F16), Some(ScalarKind::F32)) => {
                let f = self.emit_f16_to_float(val);
                let b = self.fresh_tmp();
                let z = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = bitcast float {} to i32\n  {} = zext i32 {} to i64\n",
                    b, f, z, b
                ));
                z
            }
            (Some(ScalarKind::BF16), Some(ScalarKind::F32)) => {
                let f = self.emit_bf16_to_float(val);
                let b = self.fresh_tmp();
                let z = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = bitcast float {} to i32\n  {} = zext i32 {} to i64\n",
                    b, f, z, b
                ));
                z
            }
            (Some(ScalarKind::F16), Some(ScalarKind::F64)) => {
                let f = self.emit_f16_to_float(val);
                let d = self.fresh_tmp();
                let b = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = fpext float {} to double\n  {} = bitcast double {} to i64\n",
                    d, f, b, d
                ));
                b
            }
            (Some(ScalarKind::BF16), Some(ScalarKind::F64)) => {
                let f = self.emit_bf16_to_float(val);
                let d = self.fresh_tmp();
                let b = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = fpext float {} to double\n  {} = bitcast double {} to i64\n",
                    d, f, b, d
                ));
                b
            }
            (Some(ScalarKind::F64), Some(ScalarKind::F16)) => {
                let d = self.fresh_tmp();
                let f = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = bitcast i64 {} to double\n  {} = fptrunc double {} to float\n",
                    d, val, f, d
                ));
                self.emit_float_to_f16(&f)
            }
            (Some(ScalarKind::F64), Some(ScalarKind::BF16)) => {
                let d = self.fresh_tmp();
                let f = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = bitcast i64 {} to double\n  {} = fptrunc double {} to float\n",
                    d, val, f, d
                ));
                self.emit_float_to_bf16(&f)
            }
            (Some(ScalarKind::F32), Some(ScalarKind::F16)) => {
                let t = self.fresh_tmp();
                let f = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = trunc i64 {} to i32\n  {} = bitcast i32 {} to float\n",
                    t, val, f, t
                ));
                self.emit_float_to_f16(&f)
            }
            (Some(ScalarKind::F32), Some(ScalarKind::BF16)) => {
                let t = self.fresh_tmp();
                let f = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = trunc i64 {} to i32\n  {} = bitcast i32 {} to float\n",
                    t, val, f, t
                ));
                self.emit_float_to_bf16(&f)
            }
            (Some(ScalarKind::F16), Some(ScalarKind::BF16)) => {
                let f = self.emit_f16_to_float(val);
                self.emit_float_to_bf16(&f)
            }
            (Some(ScalarKind::BF16), Some(ScalarKind::F16)) => {
                let f = self.emit_bf16_to_float(val);
                self.emit_float_to_f16(&f)
            }
            (Some(ScalarKind::F16), Some(dk)) if dk.is_int() => {
                let f = self.emit_f16_to_float(val);
                let i = self.fresh_tmp();
                let op = if dk.is_unsigned() { "fptoui" } else { "fptosi" };
                self.code.push_str(&format!("  {} = {} float {} to i64\n", i, op, f));
                self.normalize_scalar(&i, dk)
            }
            (Some(ScalarKind::BF16), Some(dk)) if dk.is_int() => {
                let f = self.emit_bf16_to_float(val);
                let i = self.fresh_tmp();
                let op = if dk.is_unsigned() { "fptoui" } else { "fptosi" };
                self.code.push_str(&format!("  {} = {} float {} to i64\n", i, op, f));
                self.normalize_scalar(&i, dk)
            }
            (sk, Some(ScalarKind::F16)) if sk != Some(ScalarKind::F16) => {
                let f = self.fresh_tmp();
                let op = if sk.is_some_and(|k| k.is_unsigned()) { "uitofp" } else { "sitofp" };
                self.code.push_str(&format!("  {} = {} i64 {} to float\n", f, op, val));
                self.emit_float_to_f16(&f)
            }
            (sk, Some(ScalarKind::BF16)) if sk != Some(ScalarKind::BF16) => {
                let f = self.fresh_tmp();
                let op = if sk.is_some_and(|k| k.is_unsigned()) { "uitofp" } else { "sitofp" };
                self.code.push_str(&format!("  {} = {} i64 {} to float\n", f, op, val));
                self.emit_float_to_bf16(&f)
            }
            (Some(ScalarKind::F64), Some(ScalarKind::F32)) => {
                let d = self.fresh_tmp();
                let f = self.fresh_tmp();
                let b = self.fresh_tmp();
                let z = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = bitcast i64 {} to double\n  {} = fptrunc double {} to float\n  {} = bitcast float {} to i32\n  {} = zext i32 {} to i64\n",
                    d, val, f, d, b, f, z, b
                ));
                z
            }
            (Some(ScalarKind::F32), Some(ScalarKind::F64)) => {
                let t = self.fresh_tmp();
                let f = self.fresh_tmp();
                let d = self.fresh_tmp();
                let b = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = trunc i64 {} to i32\n  {} = bitcast i32 {} to float\n  {} = fpext float {} to double\n  {} = bitcast double {} to i64\n",
                    t, val, f, t, d, f, b, d
                ));
                b
            }
            (Some(ScalarKind::F64), Some(dk)) if dk.is_int() => {
                let d = self.fresh_tmp();
                let i = self.fresh_tmp();
                let op = if dk.is_unsigned() { "fptoui" } else { "fptosi" };
                self.code.push_str(&format!(
                    "  {} = bitcast i64 {} to double\n  {} = {} double {} to i64\n",
                    d, val, i, op, d
                ));
                self.normalize_scalar(&i, dk)
            }
            (Some(ScalarKind::F32), Some(dk)) if dk.is_int() => {
                let t = self.fresh_tmp();
                let f = self.fresh_tmp();
                let i = self.fresh_tmp();
                let op = if dk.is_unsigned() { "fptoui" } else { "fptosi" };
                self.code.push_str(&format!(
                    "  {} = trunc i64 {} to i32\n  {} = bitcast i32 {} to float\n  {} = {} float {} to i64\n",
                    t, val, f, t, i, op, f
                ));
                self.normalize_scalar(&i, dk)
            }
            (sk, Some(ScalarKind::F64)) if sk != Some(ScalarKind::F64) => {
                let d = self.fresh_tmp();
                let b = self.fresh_tmp();
                let op = if sk.is_some_and(|k| k.is_unsigned()) { "uitofp" } else { "sitofp" };
                self.code.push_str(&format!(
                    "  {} = {} i64 {} to double\n  {} = bitcast double {} to i64\n",
                    d, op, val, b, d
                ));
                b
            }
            (sk, Some(ScalarKind::F32)) if sk != Some(ScalarKind::F32) => {
                let f = self.fresh_tmp();
                let b = self.fresh_tmp();
                let z = self.fresh_tmp();
                let op = if sk.is_some_and(|k| k.is_unsigned()) { "uitofp" } else { "sitofp" };
                self.code.push_str(&format!(
                    "  {} = {} i64 {} to float\n  {} = bitcast float {} to i32\n  {} = zext i32 {} to i64\n",
                    f, op, val, b, f, z, b
                ));
                z
            }
            (_, Some(dk)) => self.normalize_scalar(val, dk),
            _ => val.to_string(),
        }
    }

    fn emit_extern_c_call(
        &mut self,
        ef: &ExternFnDecl,
        args: &[CallArg],
    ) -> Result<String, String> {
        let mut temps = Vec::new();
        let mut c_args = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            let raw = self.lower_expr_borrowed(&a.value, &mut temps)?;
            let cty = if let Some(p) = ef.params.get(i) {
                llvm_c_abi_type(&p.ty, self.alias_bases)
            } else {
                // Variadic extra argument: promote C default promotions
                let aty = self.infer_expr_type(&a.value);
                match llvm_c_abi_type(&aty, self.alias_bases) {
                    "float" => "double",
                    "i8" | "i16" | "i1" => "i32",
                    other => other,
                }
            };
            let cv = match cty {
                "i8" | "i16" | "i32" => {
                    let t = self.fresh_tmp();
                    self.code.push_str(&format!("  {} = trunc i64 {} to {}\n", t, raw, cty));
                    t
                }
                "i1" => {
                    let t = self.fresh_tmp();
                    self.code.push_str(&format!("  {} = icmp ne i64 {}, 0\n", t, raw));
                    t
                }
                "float" => {
                    let t = self.fresh_tmp();
                    let f = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = trunc i64 {} to i32\n  {} = bitcast i32 {} to float\n",
                        t, raw, f, t
                    ));
                    f
                }
                "double" => {
                    let d = self.fresh_tmp();
                    self.code.push_str(&format!("  {} = bitcast i64 {} to double\n", d, raw));
                    d
                }
                "ptr" => {
                    let p = self.fresh_tmp();
                    self.code.push_str(&format!("  {} = inttoptr i64 {} to ptr\n", p, raw));
                    p
                }
                _ => raw,
            };
            c_args.push(format!("{} {}", cty, cv));
        }
        let ret_llvm = match &ef.ret_ty {
            Some(rt) => llvm_c_abi_type(rt, self.alias_bases),
            None => "void",
        };
        let sig_str = if ef.is_variadic {
            let mut ps: Vec<&str> = ef
                .params
                .iter()
                .map(|p| llvm_c_abi_type(&p.ty, self.alias_bases))
                .collect();
            ps.push("...");
            format!("{} ({})", ret_llvm, ps.join(", "))
        } else {
            ret_llvm.to_string()
        };
        let out = if ret_llvm == "void" {
            self.code.push_str(&format!(
                "  call {} @{}({})\n",
                sig_str,
                ef.name,
                c_args.join(", ")
            ));
            "0".to_string()
        } else {
            let rv = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = call {} @{}({})\n",
                rv,
                sig_str,
                ef.name,
                c_args.join(", ")
            ));
            let sk = ef
                .ret_ty
                .as_ref()
                .and_then(|t| scalar_kind_of_ty(t, self.alias_bases));
            match ret_llvm {
                "i8" | "i16" | "i32" => {
                    let ext = if sk.is_some_and(|k| k.is_unsigned()) { "zext" } else { "sext" };
                    let o = self.fresh_tmp();
                    self.code.push_str(&format!("  {} = {} {} {} to i64\n", o, ext, ret_llvm, rv));
                    o
                }
                "i1" => {
                    let o = self.fresh_tmp();
                    self.code.push_str(&format!("  {} = zext i1 {} to i64\n", o, rv));
                    o
                }
                "float" => {
                    let b = self.fresh_tmp();
                    let o = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = bitcast float {} to i32\n  {} = zext i32 {} to i64\n",
                        b, rv, o, b
                    ));
                    o
                }
                "double" => {
                    let o = self.fresh_tmp();
                    self.code.push_str(&format!("  {} = bitcast double {} to i64\n", o, rv));
                    o
                }
                "ptr" => {
                    let o = self.fresh_tmp();
                    self.code.push_str(&format!("  {} = ptrtoint ptr {} to i64\n", o, rv));
                    o
                }
                _ => rv,
            }
        };
        self.flush_temps(&temps);
        Ok(out)
    }

    /// Returns `(llvm_vec_ty, elem_byte_size, flat_tag)` if `ty` is `List[SimdVec]`
    /// eligible for contiguous unboxed value-type specialization (zero pointer chasing, zero child RC).
    fn flat_simd_list_info(&self, ty: &TypeExpr) -> Option<(&'static str, usize, i64)> {
        let elem = self.list_elem_ty(ty)?;
        self.simd_elem_info(&elem)
    }

    /// Returns `Some(is_signed)` if `ty` is a compact 2-byte scalar list
    /// (`List[Float16]`, `List[BFloat16]`, `List[Int16]`, `List[UInt16]`, stored flat with tag 20, 2 bytes/elem).
    fn flat_scalar2_list_info(&self, ty: &TypeExpr) -> Option<bool> {
        let elem = self.list_elem_ty(ty)?;
        self.scalar2_elem_info(&elem)
    }

    fn scalar2_elem_info(&self, ty: &TypeExpr) -> Option<bool> {
        match resolve_alias_ty(ty, self.alias_bases) {
            TypeExpr::Named { name, .. } => match name.as_str() {
                "Int16" => Some(true),
                "UInt16" | "Float16" | "BFloat16" => Some(false),
                _ => None,
            },
            _ => None,
        }
    }

    fn emit_flat_scalar2_load(
        &mut self,
        list_val: &str,
        idx_val: &str,
        is_signed: bool,
        is_safe: bool,
    ) -> String {
        let ext = if is_signed { "sext" } else { "zext" };
        if is_safe {
            let base = self.fresh_tmp();
            let data = self.fresh_tmp();
            let ep = self.fresh_tmp();
            let v16 = self.fresh_tmp();
            let res = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 40\n  {} = getelementptr inbounds i16, ptr {}, i64 {}\n  {} = load i16, ptr {}, align 2, !tbaa !7\n  {} = {} i16 {} to i64\n",
                base, list_val, data, base, ep, data, idx_val, v16, ep, res, ext, v16
            ));
            res
        } else {
            self.slot_id += 1;
            let rslot = format!("%fs2_{}", self.slot_id);
            self.entry_allocas.push_str(&format!("  {} = alloca i64, align 8\n", rslot));
            let chk_l = self.fresh_lbl("fs2_chk");
            let ok_l = self.fresh_lbl("fs2_ok");
            let oob_l = self.fresh_lbl("fs2_oob");
            let end_l = self.fresh_lbl("fs2_end");
            let nz = self.fresh_tmp();
            let base = self.fresh_tmp();
            let lp = self.fresh_tmp();
            let len_v = self.fresh_tmp();
            let in_b = self.fresh_tmp();
            let data = self.fresh_tmp();
            let ep = self.fresh_tmp();
            let v16 = self.fresh_tmp();
            let v64 = self.fresh_tmp();
            let out = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 16\n  {} = load i64, ptr {}, align 8, !tbaa !8\n  {} = icmp ult i64 {}, {}\n  br i1 {}, label %{}, label %{}\n{}:\n  {} = getelementptr inbounds i8, ptr {}, i64 40\n  {} = getelementptr inbounds i16, ptr {}, i64 {}\n  {} = load i16, ptr {}, align 2, !tbaa !7\n  {} = {} i16 {} to i64\n  store i64 {}, ptr {}, align 8\n  br label %{}\n{}:\n  store i64 0, ptr {}, align 8\n  br label %{}\n{}:\n  {} = load i64, ptr {}, align 8\n",
                nz, list_val, nz, chk_l, oob_l,
                chk_l, base, list_val, lp, base, len_v, lp, in_b, idx_val, len_v, in_b, ok_l, oob_l,
                ok_l, data, base, ep, data, idx_val, v16, ep, v64, ext, v16, v64, rslot, end_l,
                oob_l, rslot, end_l,
                end_l, out, rslot
            ));
            out
        }
    }

    fn emit_flat_scalar2_store(
        &mut self,
        list_val: &str,
        idx_val: &str,
        val_i64: &str,
        is_safe: bool,
    ) {
        let v16 = self.fresh_tmp();
        self.code.push_str(&format!("  {} = trunc i64 {} to i16\n", v16, val_i64));
        let base = self.fresh_tmp();
        let data = self.fresh_tmp();
        let ep = self.fresh_tmp();
        if is_safe {
            self.code.push_str(&format!(
                "  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 40\n  {} = getelementptr inbounds i16, ptr {}, i64 {}\n  store i16 {}, ptr {}, align 2, !tbaa !7\n",
                base, list_val, data, base, ep, data, idx_val, v16, ep
            ));
        } else {
            let ok_l = self.fresh_lbl("fs2s_ok");
            let end_l = self.fresh_lbl("fs2s_end");
            let lp = self.fresh_tmp();
            let len_v = self.fresh_tmp();
            let in_b = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 16\n  {} = load i64, ptr {}, align 8, !tbaa !8\n  {} = icmp ult i64 {}, {}\n  br i1 {}, label %{}, label %{}\n{}:\n  {} = getelementptr inbounds i8, ptr {}, i64 40\n  {} = getelementptr inbounds i16, ptr {}, i64 {}\n  store i16 {}, ptr {}, align 2, !tbaa !7\n  br label %{}\n{}:\n",
                base, list_val, lp, base, len_v, lp, in_b, idx_val, len_v, in_b, ok_l, end_l,
                ok_l, data, base, ep, data, idx_val, v16, ep, end_l, end_l
            ));
        }
    }

    /// Returns `Some(is_signed)` if `ty` is a compact 4-byte scalar list
    /// (`List[Int32]`, `List[UInt32]`, or `List[Float32]`, stored flat with tag 19, 4 bytes/elem).
    fn flat_scalar4_list_info(&self, ty: &TypeExpr) -> Option<bool> {
        let elem = self.list_elem_ty(ty)?;
        self.scalar4_elem_info(&elem)
    }

    fn scalar4_elem_info(&self, ty: &TypeExpr) -> Option<bool> {
        match resolve_alias_ty(ty, self.alias_bases) {
            TypeExpr::Named { name, .. } => match name.as_str() {
                "Int32" => Some(true),
                "UInt32" | "Float32" => Some(false),
                _ => None,
            },
            _ => None,
        }
    }

    fn emit_flat_scalar4_load(
        &mut self,
        list_val: &str,
        idx_val: &str,
        is_signed: bool,
        is_safe: bool,
    ) -> String {
        let ext = if is_signed { "sext" } else { "zext" };
        if is_safe {
            let base = self.fresh_tmp();
            let data = self.fresh_tmp();
            let ep = self.fresh_tmp();
            let v32 = self.fresh_tmp();
            let res = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 40\n  {} = getelementptr inbounds i32, ptr {}, i64 {}\n  {} = load i32, ptr {}, align 4, !tbaa !7\n  {} = {} i32 {} to i64\n",
                base, list_val, data, base, ep, data, idx_val, v32, ep, res, ext, v32
            ));
            res
        } else {
            self.slot_id += 1;
            let rslot = format!("%fs4_{}", self.slot_id);
            self.entry_allocas.push_str(&format!("  {} = alloca i64, align 8\n", rslot));
            let chk_l = self.fresh_lbl("fs4_chk");
            let ok_l = self.fresh_lbl("fs4_ok");
            let oob_l = self.fresh_lbl("fs4_oob");
            let end_l = self.fresh_lbl("fs4_end");
            let nz = self.fresh_tmp();
            let base = self.fresh_tmp();
            let lp = self.fresh_tmp();
            let len_v = self.fresh_tmp();
            let in_b = self.fresh_tmp();
            let data = self.fresh_tmp();
            let ep = self.fresh_tmp();
            let v32 = self.fresh_tmp();
            let v64 = self.fresh_tmp();
            let out = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 16\n  {} = load i64, ptr {}, align 8, !tbaa !8\n  {} = icmp ult i64 {}, {}\n  br i1 {}, label %{}, label %{}\n{}:\n  {} = getelementptr inbounds i8, ptr {}, i64 40\n  {} = getelementptr inbounds i32, ptr {}, i64 {}\n  {} = load i32, ptr {}, align 4, !tbaa !7\n  {} = {} i32 {} to i64\n  store i64 {}, ptr {}, align 8\n  br label %{}\n{}:\n  store i64 0, ptr {}, align 8\n  br label %{}\n{}:\n  {} = load i64, ptr {}, align 8\n",
                nz, list_val, nz, chk_l, oob_l,
                chk_l, base, list_val, lp, base, len_v, lp, in_b, idx_val, len_v, in_b, ok_l, oob_l,
                ok_l, data, base, ep, data, idx_val, v32, ep, v64, ext, v32, v64, rslot, end_l,
                oob_l, rslot, end_l,
                end_l, out, rslot
            ));
            out
        }
    }

    fn emit_flat_scalar4_store(
        &mut self,
        list_val: &str,
        idx_val: &str,
        val_i64: &str,
        is_safe: bool,
    ) {
        let v32 = self.fresh_tmp();
        self.code.push_str(&format!("  {} = trunc i64 {} to i32\n", v32, val_i64));
        let base = self.fresh_tmp();
        let data = self.fresh_tmp();
        let ep = self.fresh_tmp();
        if is_safe {
            self.code.push_str(&format!(
                "  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 40\n  {} = getelementptr inbounds i32, ptr {}, i64 {}\n  store i32 {}, ptr {}, align 4, !tbaa !7\n",
                base, list_val, data, base, ep, data, idx_val, v32, ep
            ));
        } else {
            let ok_l = self.fresh_lbl("fs4s_ok");
            let end_l = self.fresh_lbl("fs4s_end");
            let lp = self.fresh_tmp();
            let len_v = self.fresh_tmp();
            let in_b = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 16\n  {} = load i64, ptr {}, align 8, !tbaa !8\n  {} = icmp ult i64 {}, {}\n  br i1 {}, label %{}, label %{}\n{}:\n  {} = getelementptr inbounds i8, ptr {}, i64 40\n  {} = getelementptr inbounds i32, ptr {}, i64 {}\n  store i32 {}, ptr {}, align 4, !tbaa !7\n  br label %{}\n{}:\n",
                base, list_val, lp, base, len_v, lp, in_b, idx_val, len_v, in_b, ok_l, end_l,
                ok_l, data, base, ep, data, idx_val, v32, ep, end_l, end_l
            ));
        }
    }

    fn simd_elem_info(&self, ty: &TypeExpr) -> Option<(&'static str, usize, i64)> {
        match resolve_alias_ty(ty, self.alias_bases) {
            TypeExpr::Named { name, .. } => match name.as_str() {
                "Float64x4" => Some(("<4 x double>", 32, 18)),
                "Int64x4" => Some(("<4 x i64>", 32, 18)),
                "Float32x4" => Some(("<4 x float>", 16, 17)),
                "Float16x8" => Some(("<8 x half>", 16, 17)),
                "Int32x4" => Some(("<4 x i32>", 16, 17)),
                "UInt8x16" => Some(("<16 x i8>", 16, 17)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Evaluates a SIMD vector expression directly into an LLVM vector SSA register (`<4 x double>`, etc.)
    /// without intermediate heap boxing:
    /// - `a[j]` on a flat `List[SimdVec]` emits a direct contiguous vector `load` (`!tbaa !7`)!
    /// - `simd_*_new(...)` / `simd_*_splat(...)` constructs the vector directly in registers!
    fn lower_simd_vec_reg(
        &mut self,
        expr: &Expr,
        vec_ty: &str,
        elem_bytes: usize,
        temps: &mut Vec<String>,
    ) -> Result<String, String> {
        // Fast path 1: contiguous unboxed vector load from flat `List[SimdVec]`
        if let ExprKind::Index { target, index } = &expr.kind {
            let tty = self.infer_expr_type(target);
            if let Some((f_vty, f_bytes, _)) = self.flat_simd_list_info(&tty) {
                if f_vty == vec_ty && f_bytes == elem_bytes {
                    let t = self.lower_expr_borrowed(target, temps)?;
                    let idx = self.lower_expr_borrowed(index, temps)?;
                    let is_safe = self.proved_safe_indices.contains(&expr.span.start);
                    if is_safe {
                        let base = self.fresh_tmp();
                        let data = self.fresh_tmp();
                        let ep = self.fresh_tmp();
                        let v = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 40\n  {} = getelementptr inbounds {}, ptr {}, i64 {}\n  {} = load {}, ptr {}, align 8, !tbaa !7\n",
                            base, t, data, base, ep, vec_ty, data, idx, v, vec_ty, ep
                        ));
                        return Ok(v);
                    } else {
                        let res_slot = format!("%fsimd_{}", { self.slot_id += 1; self.slot_id });
                        self.entry_allocas
                            .push_str(&format!("  {} = alloca {}, align 8\n", res_slot, vec_ty));
                        let ok_l = self.fresh_lbl("fsimd_ok");
                        let oob_l = self.fresh_lbl("fsimd_oob");
                        let end_l = self.fresh_lbl("fsimd_end");
                        let nz = self.fresh_tmp();
                        let base = self.fresh_tmp();
                        let lp = self.fresh_tmp();
                        let len_v = self.fresh_tmp();
                        let in_b = self.fresh_tmp();
                        let ok_all = self.fresh_tmp();
                        let data = self.fresh_tmp();
                        let ep = self.fresh_tmp();
                        let lv = self.fresh_tmp();
                        let out = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}_chk, label %{}\n{}_chk:\n  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 16\n  {} = load i64, ptr {}, align 8, !tbaa !8\n  {} = icmp ult i64 {}, {}\n  {} = and i1 {}, {}\n  br i1 {}, label %{}, label %{}\n{}:\n  {} = getelementptr inbounds i8, ptr {}, i64 40\n  {} = getelementptr inbounds {}, ptr {}, i64 {}\n  {} = load {}, ptr {}, align 8, !tbaa !7\n  store {} {}, ptr {}, align 8\n  br label %{}\n{}:\n  store {} zeroinitializer, ptr {}, align 8\n  br label %{}\n{}:\n  {} = load {}, ptr {}, align 8\n",
                            nz, t, nz, ok_l, oob_l,
                            ok_l, base, t, lp, base, len_v, lp, in_b, idx, len_v, ok_all, nz, in_b, ok_all, ok_l, oob_l,
                            ok_l, data, base, ep, vec_ty, data, idx, lv, vec_ty, ep, vec_ty, lv, res_slot, end_l,
                            oob_l, vec_ty, res_slot, end_l,
                            end_l, out, vec_ty, res_slot
                        ));
                        return Ok(out);
                    }
                }
            }
        }
        // Fast path 2: construct SIMD vector directly in registers (zero heap allocation!)
        if let ExprKind::Call { callee, args } = &expr.kind {
            if let ExprKind::Ident(fname) = &callee.kind {
                match (fname.as_str(), vec_ty) {
                    ("simd_f64x4_new", "<4 x double>") if args.len() == 4 => {
                        let mut ds = Vec::new();
                        for a in args {
                            let raw = self.lower_expr_borrowed(&a.value, temps)?;
                            let d = self.fresh_tmp();
                            self.code.push_str(&format!("  {} = bitcast i64 {} to double\n", d, raw));
                            ds.push(d);
                        }
                        let v0 = self.fresh_tmp();
                        let v1 = self.fresh_tmp();
                        let v2 = self.fresh_tmp();
                        let v3 = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = insertelement <4 x double> undef, double {}, i32 0\n  {} = insertelement <4 x double> {}, double {}, i32 1\n  {} = insertelement <4 x double> {}, double {}, i32 2\n  {} = insertelement <4 x double> {}, double {}, i32 3\n",
                            v0, ds[0], v1, v0, ds[1], v2, v1, ds[2], v3, v2, ds[3]
                        ));
                        return Ok(v3);
                    }
                    ("simd_f64x4_splat", "<4 x double>") if args.len() == 1 => {
                        let raw = self.lower_expr_borrowed(&args[0].value, temps)?;
                        let d = self.fresh_tmp();
                        let v0 = self.fresh_tmp();
                        let vs = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = bitcast i64 {} to double\n  {} = insertelement <4 x double> undef, double {}, i32 0\n  {} = shufflevector <4 x double> {}, <4 x double> undef, <4 x i32> zeroinitializer\n",
                            d, raw, v0, d, vs, v0
                        ));
                        return Ok(vs);
                    }
                    ("simd_f32x4_new", "<4 x float>") if args.len() == 4 => {
                        let mut fs = Vec::new();
                        for a in args {
                            let raw = self.lower_expr_borrowed(&a.value, temps)?;
                            let t = self.fresh_tmp();
                            let f = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = trunc i64 {} to i32\n  {} = bitcast i32 {} to float\n",
                                t, raw, f, t
                            ));
                            fs.push(f);
                        }
                        let v0 = self.fresh_tmp();
                        let v1 = self.fresh_tmp();
                        let v2 = self.fresh_tmp();
                        let v3 = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = insertelement <4 x float> undef, float {}, i32 0\n  {} = insertelement <4 x float> {}, float {}, i32 1\n  {} = insertelement <4 x float> {}, float {}, i32 2\n  {} = insertelement <4 x float> {}, float {}, i32 3\n",
                            v0, fs[0], v1, v0, fs[1], v2, v1, fs[2], v3, v2, fs[3]
                        ));
                        return Ok(v3);
                    }
                    ("simd_f32x4_splat", "<4 x float>") if args.len() == 1 => {
                        let raw = self.lower_expr_borrowed(&args[0].value, temps)?;
                        let t = self.fresh_tmp();
                        let f = self.fresh_tmp();
                        let v0 = self.fresh_tmp();
                        let vs = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = trunc i64 {} to i32\n  {} = bitcast i32 {} to float\n  {} = insertelement <4 x float> undef, float {}, i32 0\n  {} = shufflevector <4 x float> {}, <4 x float> undef, <4 x i32> zeroinitializer\n",
                            t, raw, f, t, v0, f, vs, v0
                        ));
                        return Ok(vs);
                    }
                    ("simd_i64x4_new", "<4 x i64>") if args.len() == 4 => {
                        let mut vs = Vec::new();
                        for a in args {
                            vs.push(self.lower_expr_borrowed(&a.value, temps)?);
                        }
                        let v0 = self.fresh_tmp();
                        let v1 = self.fresh_tmp();
                        let v2 = self.fresh_tmp();
                        let v3 = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = insertelement <4 x i64> undef, i64 {}, i32 0\n  {} = insertelement <4 x i64> {}, i64 {}, i32 1\n  {} = insertelement <4 x i64> {}, i64 {}, i32 2\n  {} = insertelement <4 x i64> {}, i64 {}, i32 3\n",
                            v0, vs[0], v1, v0, vs[1], v2, v1, vs[2], v3, v2, vs[3]
                        ));
                        return Ok(v3);
                    }
                    ("simd_i64x4_splat", "<4 x i64>") if args.len() == 1 => {
                        let raw = self.lower_expr_borrowed(&args[0].value, temps)?;
                        let v0 = self.fresh_tmp();
                        let vs = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = insertelement <4 x i64> undef, i64 {}, i32 0\n  {} = shufflevector <4 x i64> {}, <4 x i64> undef, <4 x i32> zeroinitializer\n",
                            v0, raw, vs, v0
                        ));
                        return Ok(vs);
                    }
                    ("simd_i32x4_new", "<4 x i32>") if args.len() == 4 => {
                        let mut vs = Vec::new();
                        for a in args {
                            let raw = self.lower_expr_borrowed(&a.value, temps)?;
                            let t = self.fresh_tmp();
                            self.code.push_str(&format!("  {} = trunc i64 {} to i32\n", t, raw));
                            vs.push(t);
                        }
                        let v0 = self.fresh_tmp();
                        let v1 = self.fresh_tmp();
                        let v2 = self.fresh_tmp();
                        let v3 = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = insertelement <4 x i32> undef, i32 {}, i32 0\n  {} = insertelement <4 x i32> {}, i32 {}, i32 1\n  {} = insertelement <4 x i32> {}, i32 {}, i32 2\n  {} = insertelement <4 x i32> {}, i32 {}, i32 3\n",
                            v0, vs[0], v1, v0, vs[1], v2, v1, vs[2], v3, v2, vs[3]
                        ));
                        return Ok(v3);
                    }
                    ("simd_i32x4_splat", "<4 x i32>") if args.len() == 1 => {
                        let raw = self.lower_expr_borrowed(&args[0].value, temps)?;
                        let t = self.fresh_tmp();
                        let v0 = self.fresh_tmp();
                        let vs = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = trunc i64 {} to i32\n  {} = insertelement <4 x i32> undef, i32 {}, i32 0\n  {} = shufflevector <4 x i32> {}, <4 x i32> undef, <4 x i32> zeroinitializer\n",
                            t, raw, v0, t, vs, v0
                        ));
                        return Ok(vs);
                    }
                    ("simd_u8x16_splat", "<16 x i8>") if args.len() == 1 => {
                        let raw = self.lower_expr_borrowed(&args[0].value, temps)?;
                        let t = self.fresh_tmp();
                        let v0 = self.fresh_tmp();
                        let vs = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = trunc i64 {} to i8\n  {} = insertelement <16 x i8> undef, i8 {}, i32 0\n  {} = shufflevector <16 x i8> {}, <16 x i8> undef, <16 x i32> zeroinitializer\n",
                            t, raw, v0, t, vs, v0
                        ));
                        return Ok(vs);
                    }
                    _ => {}
                }
            }
        }
        // Fallback: load vector from boxed SIMD handle
        let vh = self.lower_expr_borrowed(expr, temps)?;
        let di = self.fresh_tmp();
        let dp = self.fresh_tmp();
        let v = self.fresh_tmp();
        self.code.push_str(&format!(
            "  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n  {} = inttoptr i64 {} to ptr\n  {} = load {}, ptr {}, align 8\n",
            di, vh, dp, di, v, vec_ty, dp
        ));
        Ok(v)
    }

    /// Native LLVM SIMD Vector IR emitter (`<4 x float>`, `<4 x double>`, `<4 x i32>`, `<4 x i64>`, `<16 x i8>`).
    fn try_emit_simd_builtin(
        &mut self,
        fname: &str,
        args: &[CallArg],
        expected_ty: Option<&TypeExpr>,
    ) -> Result<Option<String>, String> {
        if !fname.starts_with("simd_") {
            return Ok(None);
        }
        let mut temps = Vec::new();
        let alloc_vec = |cg: &mut Self, bytes: usize| -> (String, String) {
            let obj = cg.fresh_tmp();
            let di = cg.fresh_tmp();
            let dp = cg.fresh_tmp();
            cg.code.push_str(&format!(
                "  {} = call i64 @tenon_cstruct_alloc(i64 {})\n  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n  {} = inttoptr i64 {} to ptr\n",
                obj, bytes, di, obj, dp, di
            ));
            (obj, dp)
        };
        let load_vec = |cg: &mut Self, h: &str, vty: &str| -> String {
            let di = cg.fresh_tmp();
            let dp = cg.fresh_tmp();
            let v = cg.fresh_tmp();
            cg.code.push_str(&format!(
                "  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n  {} = inttoptr i64 {} to ptr\n  {} = load {}, ptr {}, align 8\n",
                di, h, dp, di, v, vty, dp
            ));
            v
        };

        let out = match fname {
            "simd_f32x4_new" if args.len() == 4 => {
                let mut fs = Vec::new();
                for a in args {
                    let raw = self.lower_expr_borrowed(&a.value, &mut temps)?;
                    let t = self.fresh_tmp();
                    let f = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = trunc i64 {} to i32\n  {} = bitcast i32 {} to float\n",
                        t, raw, f, t
                    ));
                    fs.push(f);
                }
                let (obj, dp) = alloc_vec(self, 16);
                let v0 = self.fresh_tmp();
                let v1 = self.fresh_tmp();
                let v2 = self.fresh_tmp();
                let v3 = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = insertelement <4 x float> undef, float {}, i32 0\n  {} = insertelement <4 x float> {}, float {}, i32 1\n  {} = insertelement <4 x float> {}, float {}, i32 2\n  {} = insertelement <4 x float> {}, float {}, i32 3\n  store <4 x float> {}, ptr {}, align 8\n",
                    v0, fs[0], v1, v0, fs[1], v2, v1, fs[2], v3, v2, fs[3], v3, dp
                ));
                obj
            }
            "simd_f32x4_splat" if args.len() == 1 => {
                let raw = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                let t = self.fresh_tmp();
                let f = self.fresh_tmp();
                let v0 = self.fresh_tmp();
                let vs = self.fresh_tmp();
                let (obj, dp) = alloc_vec(self, 16);
                self.code.push_str(&format!(
                    "  {} = trunc i64 {} to i32\n  {} = bitcast i32 {} to float\n  {} = insertelement <4 x float> undef, float {}, i32 0\n  {} = shufflevector <4 x float> {}, <4 x float> undef, <4 x i32> zeroinitializer\n  store <4 x float> {}, ptr {}, align 8\n",
                    t, raw, f, t, v0, f, vs, v0, vs, dp
                ));
                obj
            }
            "simd_f32x4_lane" if args.len() == 2 => {
                let v = self.lower_simd_vec_reg(&args[0].value, "<4 x float>", 16, &mut temps)?;
                let idx = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                let i32x = self.fresh_tmp();
                let f = self.fresh_tmp();
                let b = self.fresh_tmp();
                let z = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = trunc i64 {} to i32\n  {} = extractelement <4 x float> {}, i32 {}\n  {} = bitcast float {} to i32\n  {} = zext i32 {} to i64\n",
                    i32x, idx, f, v, i32x, b, f, z, b
                ));
                z
            }
            "simd_f32x4_binop" if args.len() == 3 => {
                let va = self.lower_simd_vec_reg(&args[0].value, "<4 x float>", 16, &mut temps)?;
                let vb = self.lower_simd_vec_reg(&args[1].value, "<4 x float>", 16, &mut temps)?;
                let (obj, dp) = alloc_vec(self, 16);
                let op_const = match &args[2].value.kind {
                    ExprKind::Int(n) => Some(*n),
                    _ => None,
                };
                let res = self.fresh_tmp();
                match op_const {
                    Some(0) => self.code.push_str(&format!("  {} = fadd <4 x float> {}, {}\n", res, va, vb)),
                    Some(1) => self.code.push_str(&format!("  {} = fsub <4 x float> {}, {}\n", res, va, vb)),
                    Some(2) => self.code.push_str(&format!("  {} = fmul <4 x float> {}, {}\n", res, va, vb)),
                    Some(3) => self.code.push_str(&format!("  {} = fdiv <4 x float> {}, {}\n", res, va, vb)),
                    Some(4) => self.code.push_str(&format!("  {} = call <4 x float> @llvm.minnum.v4f32(<4 x float> {}, <4 x float> {})\n", res, va, vb)),
                    Some(_) => self.code.push_str(&format!("  {} = call <4 x float> @llvm.maxnum.v4f32(<4 x float> {}, <4 x float> {})\n", res, va, vb)),
                    None => {
                        let opv = self.lower_expr_borrowed(&args[2].value, &mut temps)?;
                        let r0 = self.fresh_tmp();
                        let r1 = self.fresh_tmp();
                        let r2 = self.fresh_tmp();
                        let r3 = self.fresh_tmp();
                        let r4 = self.fresh_tmp();
                        let r5 = self.fresh_tmp();
                        let c0 = self.fresh_tmp();
                        let c1 = self.fresh_tmp();
                        let c2 = self.fresh_tmp();
                        let c3 = self.fresh_tmp();
                        let c4 = self.fresh_tmp();
                        let s1 = self.fresh_tmp();
                        let s2 = self.fresh_tmp();
                        let s3 = self.fresh_tmp();
                        let s4 = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = fadd <4 x float> {}, {}\n  {} = fsub <4 x float> {}, {}\n  {} = fmul <4 x float> {}, {}\n  {} = fdiv <4 x float> {}, {}\n  {} = call <4 x float> @llvm.minnum.v4f32(<4 x float> {}, <4 x float> {})\n  {} = call <4 x float> @llvm.maxnum.v4f32(<4 x float> {}, <4 x float> {})\n  {} = icmp eq i64 {}, 0\n  {} = icmp eq i64 {}, 1\n  {} = icmp eq i64 {}, 2\n  {} = icmp eq i64 {}, 3\n  {} = icmp eq i64 {}, 4\n  {} = select i1 {}, <4 x float> {}, <4 x float> {}\n  {} = select i1 {}, <4 x float> {}, <4 x float> {}\n  {} = select i1 {}, <4 x float> {}, <4 x float> {}\n  {} = select i1 {}, <4 x float> {}, <4 x float> {}\n  {} = select i1 {}, <4 x float> {}, <4 x float> {}\n",
                            r0, va, vb, r1, va, vb, r2, va, vb, r3, va, vb, r4, va, vb, r5, va, vb,
                            c0, opv, c1, opv, c2, opv, c3, opv, c4, opv,
                            s1, c4, r4, r5, s2, c3, r3, s1, s3, c2, r2, s2, s4, c1, r1, s3, res, c0, r0, s4
                        ));
                    }
                }
                self.code.push_str(&format!("  store <4 x float> {}, ptr {}, align 8\n", res, dp));
                obj
            }
            "simd_f32x4_fma" if args.len() == 3 => {
                let va = self.lower_simd_vec_reg(&args[0].value, "<4 x float>", 16, &mut temps)?;
                let vb = self.lower_simd_vec_reg(&args[1].value, "<4 x float>", 16, &mut temps)?;
                let vc = self.lower_simd_vec_reg(&args[2].value, "<4 x float>", 16, &mut temps)?;
                let (obj, dp) = alloc_vec(self, 16);
                let res = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call <4 x float> @llvm.fma.v4f32(<4 x float> {}, <4 x float> {}, <4 x float> {})\n  store <4 x float> {}, ptr {}, align 8\n",
                    res, va, vb, vc, res, dp
                ));
                obj
            }
            "simd_f32x4_reduce_sum" if args.len() == 1 => {
                let v = self.lower_simd_vec_reg(&args[0].value, "<4 x float>", 16, &mut temps)?;
                let sum = self.fresh_tmp();
                let b = self.fresh_tmp();
                let z = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call float @llvm.vector.reduce.fadd.v4f32(float -0.0, <4 x float> {})\n  {} = bitcast float {} to i32\n  {} = zext i32 {} to i64\n",
                    sum, v, b, sum, z, b
                ));
                z
            }
            "simd_f32x4_dot" if args.len() == 2 => {
                let va = self.lower_simd_vec_reg(&args[0].value, "<4 x float>", 16, &mut temps)?;
                let vb = self.lower_simd_vec_reg(&args[1].value, "<4 x float>", 16, &mut temps)?;
                let prod = self.fresh_tmp();
                let lo = self.fresh_tmp();
                let hi = self.fresh_tmp();
                let p2 = self.fresh_tmp();
                let e0 = self.fresh_tmp();
                let e1 = self.fresh_tmp();
                let sum = self.fresh_tmp();
                let b = self.fresh_tmp();
                let z = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = fmul <4 x float> {}, {}\n  {} = shufflevector <4 x float> {}, <4 x float> poison, <2 x i32> <i32 0, i32 1>\n  {} = shufflevector <4 x float> {}, <4 x float> poison, <2 x i32> <i32 2, i32 3>\n  {} = fadd <2 x float> {}, {}\n  {} = extractelement <2 x float> {}, i32 0\n  {} = extractelement <2 x float> {}, i32 1\n  {} = fadd float {}, {}\n  {} = bitcast float {} to i32\n  {} = zext i32 {} to i64\n",
                    prod, va, vb, lo, prod, hi, prod, p2, lo, hi, e0, p2, e1, p2, sum, e0, e1, b, sum, z, b
                ));
                z
            }
            "simd_f64x4_new" if args.len() == 4 => {
                let mut ds = Vec::new();
                for a in args {
                    let raw = self.lower_expr_borrowed(&a.value, &mut temps)?;
                    let d = self.fresh_tmp();
                    self.code.push_str(&format!("  {} = bitcast i64 {} to double\n", d, raw));
                    ds.push(d);
                }
                let (obj, dp) = alloc_vec(self, 32);
                let v0 = self.fresh_tmp();
                let v1 = self.fresh_tmp();
                let v2 = self.fresh_tmp();
                let v3 = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = insertelement <4 x double> undef, double {}, i32 0\n  {} = insertelement <4 x double> {}, double {}, i32 1\n  {} = insertelement <4 x double> {}, double {}, i32 2\n  {} = insertelement <4 x double> {}, double {}, i32 3\n  store <4 x double> {}, ptr {}, align 8\n",
                    v0, ds[0], v1, v0, ds[1], v2, v1, ds[2], v3, v2, ds[3], v3, dp
                ));
                obj
            }
            "simd_f64x4_splat" if args.len() == 1 => {
                let raw = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                let d = self.fresh_tmp();
                let v0 = self.fresh_tmp();
                let vs = self.fresh_tmp();
                let (obj, dp) = alloc_vec(self, 32);
                self.code.push_str(&format!(
                    "  {} = bitcast i64 {} to double\n  {} = insertelement <4 x double> undef, double {}, i32 0\n  {} = shufflevector <4 x double> {}, <4 x double> undef, <4 x i32> zeroinitializer\n  store <4 x double> {}, ptr {}, align 8\n",
                    d, raw, v0, d, vs, v0, vs, dp
                ));
                obj
            }
            "simd_f64x4_lane" if args.len() == 2 => {
                let v = self.lower_simd_vec_reg(&args[0].value, "<4 x double>", 32, &mut temps)?;
                let idx = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                let i32x = self.fresh_tmp();
                let d = self.fresh_tmp();
                let b = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = trunc i64 {} to i32\n  {} = extractelement <4 x double> {}, i32 {}\n  {} = bitcast double {} to i64\n",
                    i32x, idx, d, v, i32x, b, d
                ));
                b
            }
            "simd_f64x4_binop" if args.len() == 3 => {
                let va = self.lower_simd_vec_reg(&args[0].value, "<4 x double>", 32, &mut temps)?;
                let vb = self.lower_simd_vec_reg(&args[1].value, "<4 x double>", 32, &mut temps)?;
                let (obj, dp) = alloc_vec(self, 32);
                let op_const = match &args[2].value.kind {
                    ExprKind::Int(n) => Some(*n),
                    _ => None,
                };
                let res = self.fresh_tmp();
                match op_const {
                    Some(0) => self.code.push_str(&format!("  {} = fadd <4 x double> {}, {}\n", res, va, vb)),
                    Some(1) => self.code.push_str(&format!("  {} = fsub <4 x double> {}, {}\n", res, va, vb)),
                    Some(2) => self.code.push_str(&format!("  {} = fmul <4 x double> {}, {}\n", res, va, vb)),
                    Some(3) => self.code.push_str(&format!("  {} = fdiv <4 x double> {}, {}\n", res, va, vb)),
                    Some(4) => self.code.push_str(&format!("  {} = call <4 x double> @llvm.minnum.v4f64(<4 x double> {}, <4 x double> {})\n", res, va, vb)),
                    Some(_) => self.code.push_str(&format!("  {} = call <4 x double> @llvm.maxnum.v4f64(<4 x double> {}, <4 x double> {})\n", res, va, vb)),
                    None => {
                        let opv = self.lower_expr_borrowed(&args[2].value, &mut temps)?;
                        let r0 = self.fresh_tmp();
                        let r1 = self.fresh_tmp();
                        let r2 = self.fresh_tmp();
                        let r3 = self.fresh_tmp();
                        let r4 = self.fresh_tmp();
                        let r5 = self.fresh_tmp();
                        let c0 = self.fresh_tmp();
                        let c1 = self.fresh_tmp();
                        let c2 = self.fresh_tmp();
                        let c3 = self.fresh_tmp();
                        let c4 = self.fresh_tmp();
                        let s1 = self.fresh_tmp();
                        let s2 = self.fresh_tmp();
                        let s3 = self.fresh_tmp();
                        let s4 = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = fadd <4 x double> {}, {}\n  {} = fsub <4 x double> {}, {}\n  {} = fmul <4 x double> {}, {}\n  {} = fdiv <4 x double> {}, {}\n  {} = call <4 x double> @llvm.minnum.v4f64(<4 x double> {}, <4 x double> {})\n  {} = call <4 x double> @llvm.maxnum.v4f64(<4 x double> {}, <4 x double> {})\n  {} = icmp eq i64 {}, 0\n  {} = icmp eq i64 {}, 1\n  {} = icmp eq i64 {}, 2\n  {} = icmp eq i64 {}, 3\n  {} = icmp eq i64 {}, 4\n  {} = select i1 {}, <4 x double> {}, <4 x double> {}\n  {} = select i1 {}, <4 x double> {}, <4 x double> {}\n  {} = select i1 {}, <4 x double> {}, <4 x double> {}\n  {} = select i1 {}, <4 x double> {}, <4 x double> {}\n  {} = select i1 {}, <4 x double> {}, <4 x double> {}\n",
                            r0, va, vb, r1, va, vb, r2, va, vb, r3, va, vb, r4, va, vb, r5, va, vb,
                            c0, opv, c1, opv, c2, opv, c3, opv, c4, opv,
                            s1, c4, r4, r5, s2, c3, r3, s1, s3, c2, r2, s2, s4, c1, r1, s3, res, c0, r0, s4
                        ));
                    }
                }
                self.code.push_str(&format!("  store <4 x double> {}, ptr {}, align 8\n", res, dp));
                obj
            }
            "simd_f64x4_fma" if args.len() == 3 => {
                let va = self.lower_simd_vec_reg(&args[0].value, "<4 x double>", 32, &mut temps)?;
                let vb = self.lower_simd_vec_reg(&args[1].value, "<4 x double>", 32, &mut temps)?;
                let vc = self.lower_simd_vec_reg(&args[2].value, "<4 x double>", 32, &mut temps)?;
                let (obj, dp) = alloc_vec(self, 32);
                let res = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call <4 x double> @llvm.fma.v4f64(<4 x double> {}, <4 x double> {}, <4 x double> {})\n  store <4 x double> {}, ptr {}, align 8\n",
                    res, va, vb, vc, res, dp
                ));
                obj
            }
            "simd_f64x4_reduce_sum" if args.len() == 1 => {
                let v = self.lower_simd_vec_reg(&args[0].value, "<4 x double>", 32, &mut temps)?;
                let sum = self.fresh_tmp();
                let b = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call double @llvm.vector.reduce.fadd.v4f64(double -0.0, <4 x double> {})\n  {} = bitcast double {} to i64\n",
                    sum, v, b, sum
                ));
                b
            }
            "simd_f64x4_dot" if args.len() == 2 => {
                let va = self.lower_simd_vec_reg(&args[0].value, "<4 x double>", 32, &mut temps)?;
                let vb = self.lower_simd_vec_reg(&args[1].value, "<4 x double>", 32, &mut temps)?;
                let prod = self.fresh_tmp();
                let lo = self.fresh_tmp();
                let hi = self.fresh_tmp();
                let p2 = self.fresh_tmp();
                let e0 = self.fresh_tmp();
                let e1 = self.fresh_tmp();
                let sum = self.fresh_tmp();
                let b = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = fmul <4 x double> {}, {}\n  {} = shufflevector <4 x double> {}, <4 x double> poison, <2 x i32> <i32 0, i32 1>\n  {} = shufflevector <4 x double> {}, <4 x double> poison, <2 x i32> <i32 2, i32 3>\n  {} = fadd <2 x double> {}, {}\n  {} = extractelement <2 x double> {}, i32 0\n  {} = extractelement <2 x double> {}, i32 1\n  {} = fadd double {}, {}\n  {} = bitcast double {} to i64\n",
                    prod, va, vb, lo, prod, hi, prod, p2, lo, hi, e0, p2, e1, p2, sum, e0, e1, b, sum
                ));
                b
            }
            "simd_i32x4_new" if args.len() == 4 => {
                let mut vs = Vec::new();
                for a in args {
                    let raw = self.lower_expr_borrowed(&a.value, &mut temps)?;
                    let t = self.fresh_tmp();
                    self.code.push_str(&format!("  {} = trunc i64 {} to i32\n", t, raw));
                    vs.push(t);
                }
                let (obj, dp) = alloc_vec(self, 16);
                let v0 = self.fresh_tmp();
                let v1 = self.fresh_tmp();
                let v2 = self.fresh_tmp();
                let v3 = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = insertelement <4 x i32> undef, i32 {}, i32 0\n  {} = insertelement <4 x i32> {}, i32 {}, i32 1\n  {} = insertelement <4 x i32> {}, i32 {}, i32 2\n  {} = insertelement <4 x i32> {}, i32 {}, i32 3\n  store <4 x i32> {}, ptr {}, align 8\n",
                    v0, vs[0], v1, v0, vs[1], v2, v1, vs[2], v3, v2, vs[3], v3, dp
                ));
                obj
            }
            "simd_i32x4_splat" if args.len() == 1 => {
                let raw = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                let t = self.fresh_tmp();
                let v0 = self.fresh_tmp();
                let vs = self.fresh_tmp();
                let (obj, dp) = alloc_vec(self, 16);
                self.code.push_str(&format!(
                    "  {} = trunc i64 {} to i32\n  {} = insertelement <4 x i32> undef, i32 {}, i32 0\n  {} = shufflevector <4 x i32> {}, <4 x i32> undef, <4 x i32> zeroinitializer\n  store <4 x i32> {}, ptr {}, align 8\n",
                    t, raw, v0, t, vs, v0, vs, dp
                ));
                obj
            }
            "simd_i32x4_lane" if args.len() == 2 => {
                let v = self.lower_simd_vec_reg(&args[0].value, "<4 x i32>", 16, &mut temps)?;
                let idx = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                let i32x = self.fresh_tmp();
                let el = self.fresh_tmp();
                let ex = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = trunc i64 {} to i32\n  {} = extractelement <4 x i32> {}, i32 {}\n  {} = sext i32 {} to i64\n",
                    i32x, idx, el, v, i32x, ex, el
                ));
                ex
            }
            "simd_i32x4_binop" if args.len() == 3 => {
                let va = self.lower_simd_vec_reg(&args[0].value, "<4 x i32>", 16, &mut temps)?;
                let vb = self.lower_simd_vec_reg(&args[1].value, "<4 x i32>", 16, &mut temps)?;
                let (obj, dp) = alloc_vec(self, 16);
                let op_const = match &args[2].value.kind {
                    ExprKind::Int(n) => Some(*n),
                    _ => None,
                };
                let res = self.fresh_tmp();
                match op_const {
                    Some(0) => self.code.push_str(&format!("  {} = add <4 x i32> {}, {}\n", res, va, vb)),
                    Some(1) => self.code.push_str(&format!("  {} = sub <4 x i32> {}, {}\n", res, va, vb)),
                    Some(_) => self.code.push_str(&format!("  {} = mul <4 x i32> {}, {}\n", res, va, vb)),
                    None => {
                        let opv = self.lower_expr_borrowed(&args[2].value, &mut temps)?;
                        let r0 = self.fresh_tmp();
                        let r1 = self.fresh_tmp();
                        let r2 = self.fresh_tmp();
                        let c0 = self.fresh_tmp();
                        let c1 = self.fresh_tmp();
                        let s1 = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = add <4 x i32> {}, {}\n  {} = sub <4 x i32> {}, {}\n  {} = mul <4 x i32> {}, {}\n  {} = icmp eq i64 {}, 0\n  {} = icmp eq i64 {}, 1\n  {} = select i1 {}, <4 x i32> {}, <4 x i32> {}\n  {} = select i1 {}, <4 x i32> {}, <4 x i32> {}\n",
                            r0, va, vb, r1, va, vb, r2, va, vb, c0, opv, c1, opv, s1, c1, r1, r2, res, c0, r0, s1
                        ));
                    }
                }
                self.code.push_str(&format!("  store <4 x i32> {}, ptr {}, align 8\n", res, dp));
                obj
            }
            "simd_i32x4_reduce_sum" if args.len() == 1 => {
                let v = self.lower_simd_vec_reg(&args[0].value, "<4 x i32>", 16, &mut temps)?;
                let sum = self.fresh_tmp();
                let ex = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> {})\n  {} = sext i32 {} to i64\n",
                    sum, v, ex, sum
                ));
                ex
            }
            "simd_i64x4_new" if args.len() == 4 => {
                let mut vs = Vec::new();
                for a in args {
                    vs.push(self.lower_expr_borrowed(&a.value, &mut temps)?);
                }
                let (obj, dp) = alloc_vec(self, 32);
                let v0 = self.fresh_tmp();
                let v1 = self.fresh_tmp();
                let v2 = self.fresh_tmp();
                let v3 = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = insertelement <4 x i64> undef, i64 {}, i32 0\n  {} = insertelement <4 x i64> {}, i64 {}, i32 1\n  {} = insertelement <4 x i64> {}, i64 {}, i32 2\n  {} = insertelement <4 x i64> {}, i64 {}, i32 3\n  store <4 x i64> {}, ptr {}, align 8\n",
                    v0, vs[0], v1, v0, vs[1], v2, v1, vs[2], v3, v2, vs[3], v3, dp
                ));
                obj
            }
            "simd_i64x4_splat" if args.len() == 1 => {
                let raw = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                let v0 = self.fresh_tmp();
                let vs = self.fresh_tmp();
                let (obj, dp) = alloc_vec(self, 32);
                self.code.push_str(&format!(
                    "  {} = insertelement <4 x i64> undef, i64 {}, i32 0\n  {} = shufflevector <4 x i64> {}, <4 x i64> undef, <4 x i32> zeroinitializer\n  store <4 x i64> {}, ptr {}, align 8\n",
                    v0, raw, vs, v0, vs, dp
                ));
                obj
            }
            "simd_i64x4_lane" if args.len() == 2 => {
                let v = self.lower_simd_vec_reg(&args[0].value, "<4 x i64>", 32, &mut temps)?;
                let idx = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                let i32x = self.fresh_tmp();
                let el = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = trunc i64 {} to i32\n  {} = extractelement <4 x i64> {}, i32 {}\n",
                    i32x, idx, el, v, i32x
                ));
                el
            }
            "simd_i64x4_binop" if args.len() == 3 => {
                let va = self.lower_simd_vec_reg(&args[0].value, "<4 x i64>", 32, &mut temps)?;
                let vb = self.lower_simd_vec_reg(&args[1].value, "<4 x i64>", 32, &mut temps)?;
                let (obj, dp) = alloc_vec(self, 32);
                let op_const = match &args[2].value.kind {
                    ExprKind::Int(n) => Some(*n),
                    _ => None,
                };
                let res = self.fresh_tmp();
                match op_const {
                    Some(0) => self.code.push_str(&format!("  {} = add <4 x i64> {}, {}\n", res, va, vb)),
                    Some(1) => self.code.push_str(&format!("  {} = sub <4 x i64> {}, {}\n", res, va, vb)),
                    Some(_) => self.code.push_str(&format!("  {} = mul <4 x i64> {}, {}\n", res, va, vb)),
                    None => {
                        let opv = self.lower_expr_borrowed(&args[2].value, &mut temps)?;
                        let r0 = self.fresh_tmp();
                        let r1 = self.fresh_tmp();
                        let r2 = self.fresh_tmp();
                        let c0 = self.fresh_tmp();
                        let c1 = self.fresh_tmp();
                        let s1 = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = add <4 x i64> {}, {}\n  {} = sub <4 x i64> {}, {}\n  {} = mul <4 x i64> {}, {}\n  {} = icmp eq i64 {}, 0\n  {} = icmp eq i64 {}, 1\n  {} = select i1 {}, <4 x i64> {}, <4 x i64> {}\n  {} = select i1 {}, <4 x i64> {}, <4 x i64> {}\n",
                            r0, va, vb, r1, va, vb, r2, va, vb, c0, opv, c1, opv, s1, c1, r1, r2, res, c0, r0, s1
                        ));
                    }
                }
                self.code.push_str(&format!("  store <4 x i64> {}, ptr {}, align 8\n", res, dp));
                obj
            }
            "simd_i64x4_reduce_sum" if args.len() == 1 => {
                let v = self.lower_simd_vec_reg(&args[0].value, "<4 x i64>", 32, &mut temps)?;
                let sum = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @llvm.vector.reduce.add.v4i64(<4 x i64> {})\n",
                    sum, v
                ));
                sum
            }
            "simd_u8x16_splat" if args.len() == 1 => {
                let raw = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                let t = self.fresh_tmp();
                let v0 = self.fresh_tmp();
                let vs = self.fresh_tmp();
                let (obj, dp) = alloc_vec(self, 16);
                self.code.push_str(&format!(
                    "  {} = trunc i64 {} to i8\n  {} = insertelement <16 x i8> undef, i8 {}, i32 0\n  {} = shufflevector <16 x i8> {}, <16 x i8> undef, <16 x i32> zeroinitializer\n  store <16 x i8> {}, ptr {}, align 8\n",
                    t, raw, v0, t, vs, v0, vs, dp
                ));
                obj
            }
            "simd_u8x16_lane" if args.len() == 2 => {
                let vh = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                let idx = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                let v = load_vec(self, &vh, "<16 x i8>");
                let i32x = self.fresh_tmp();
                let el = self.fresh_tmp();
                let ex = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = trunc i64 {} to i32\n  {} = extractelement <16 x i8> {}, i32 {}\n  {} = zext i8 {} to i64\n",
                    i32x, idx, el, v, i32x, ex, el
                ));
                ex
            }
            "simd_u8x16_set_lane" if args.len() == 3 => {
                let vh = self.lower_expr_owned(&args[0].value, expected_ty)?;
                let idx = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                let val = self.lower_expr_borrowed(&args[2].value, &mut temps)?;
                let uniq = self.fresh_tmp();
                let di = self.fresh_tmp();
                let dp = self.fresh_tmp();
                let ep = self.fresh_tmp();
                let b8 = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 {}\n  {} = trunc i64 {} to i8\n  store i8 {}, ptr {}, align 1\n",
                    uniq, vh, di, uniq, dp, di, ep, dp, idx, b8, val, b8, ep
                ));
                uniq
            }
            "simd_u8x16_binop" if args.len() == 3 => {
                let ah = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                let bh = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                let va = load_vec(self, &ah, "<16 x i8>");
                let vb = load_vec(self, &bh, "<16 x i8>");
                let (obj, dp) = alloc_vec(self, 16);
                let op_const = match &args[2].value.kind {
                    ExprKind::Int(n) => Some(*n),
                    _ => None,
                };
                let res = self.fresh_tmp();
                match op_const {
                    Some(0) => self.code.push_str(&format!("  {} = add <16 x i8> {}, {}\n", res, va, vb)),
                    Some(1) => self.code.push_str(&format!("  {} = sub <16 x i8> {}, {}\n", res, va, vb)),
                    Some(_) => self.code.push_str(&format!("  {} = xor <16 x i8> {}, {}\n", res, va, vb)),
                    None => {
                        let opv = self.lower_expr_borrowed(&args[2].value, &mut temps)?;
                        let r0 = self.fresh_tmp();
                        let r1 = self.fresh_tmp();
                        let r2 = self.fresh_tmp();
                        let c0 = self.fresh_tmp();
                        let c1 = self.fresh_tmp();
                        let s1 = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = add <16 x i8> {}, {}\n  {} = sub <16 x i8> {}, {}\n  {} = xor <16 x i8> {}, {}\n  {} = icmp eq i64 {}, 0\n  {} = icmp eq i64 {}, 1\n  {} = select i1 {}, <16 x i8> {}, <16 x i8> {}\n  {} = select i1 {}, <16 x i8> {}, <16 x i8> {}\n",
                            r0, va, vb, r1, va, vb, r2, va, vb, c0, opv, c1, opv, s1, c1, r1, r2, res, c0, r0, s1
                        ));
                    }
                }
                self.code.push_str(&format!("  store <16 x i8> {}, ptr {}, align 8\n", res, dp));
                obj
            }
            "simd_u8x16_reduce_sum" if args.len() == 1 => {
                let vh = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                let v = load_vec(self, &vh, "<16 x i8>");
                let ext = self.fresh_tmp();
                // Zero-extend <16 x i8> -> <16 x i32> so sum doesn't wrap at 255!
                let sum32 = self.fresh_tmp();
                let sum64 = self.fresh_tmp();
                let mut acc = "0".to_string();
                self.code.push_str(&format!("  {} = zext <16 x i8> {} to <16 x i32>\n", ext, v));
                for lane in 0..16 {
                    let el = self.fresh_tmp();
                    let nxt = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = extractelement <16 x i32> {}, i32 {}\n  {} = add i32 {}, {}\n",
                        el, ext, lane, nxt, acc, el
                    ));
                    acc = nxt;
                }
                self.code.push_str(&format!(
                    "  {} = add i32 {}, 0\n  {} = zext i32 {} to i64\n",
                    sum32, acc, sum64, sum32
                ));
                sum64
            }
            _ => return Ok(None),
        };
        self.flush_temps(&temps);
        Ok(Some(out))
    }

    fn infer_expr_type(&self, expr: &Expr) -> TypeExpr {
        self.infer_expr_type_with(expr, &BTreeMap::new())
    }

    fn infer_expr_type_with(&self, expr: &Expr, extra_vars: &BTreeMap<String, TypeExpr>) -> TypeExpr {
        match &expr.kind {
            ExprKind::Int(_) => named_ty("Int"),
            ExprKind::Float(_) => named_ty("Float"),
            ExprKind::Bool(_) => named_ty("Bool"),
            ExprKind::Str(_) => named_ty("Text"),
            ExprKind::Ident(s) => {
                if s == "None" {
                    return generic_ty("Option", vec![named_ty("Any")]);
                }
                if let Some(ty) = extra_vars.get(s) {
                    return ty.clone();
                }
                if let Some(ty) = self.var_types.get(s) {
                    return ty.clone();
                }
                if let Some(vl) = self.enum_variants.get(s) {
                    return named_ty(&vl.enum_name);
                }
                named_ty("Any")
            }
            ExprKind::RecordLit { name, .. } => named_ty(name),
            ExprKind::List(elems) => {
                if let Some(first) = elems.first() {
                    generic_ty("List", vec![self.infer_expr_type_with(first, extra_vars)])
                } else {
                    generic_ty("List", vec![named_ty("Any")])
                }
            }
            ExprKind::Tuple(elems) => TypeExpr::Tuple(
                elems.iter().map(|e| self.infer_expr_type_with(e, extra_vars)).collect(),
                expr.span,
            ),
            ExprKind::TupleIndex { target, index } => {
                let tt = resolve_alias_ty(&self.infer_expr_type_with(target, extra_vars), self.alias_bases);
                if let TypeExpr::Tuple(es, _) = tt {
                    if let Some(et) = es.get(*index) {
                        return et.clone();
                    }
                }
                named_ty("Any")
            }
            ExprKind::Cast { ty, .. } => ty.clone(),
            ExprKind::TryCast { ty, .. } => generic_ty("Result", vec![ty.clone(), named_ty("Text")]),
            ExprKind::ArrayRepeat { value, len } => TypeExpr::Array {
                elem: Box::new(self.infer_expr_type_with(value, extra_vars)),
                len: *len,
                span: expr.span,
            },
            ExprKind::Unsafe(b) => self.infer_block_type_with(b, extra_vars),
            ExprKind::Binary { op: BinOp::Concat, .. } => named_ty("Text"),
            ExprKind::Binary {
                op: BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::And | BinOp::Or,
                ..
            } => named_ty("Bool"),
            ExprKind::Binary { op, left, right } => {
                let lt = self.infer_expr_type_with(left, extra_vars);
                let rt = self.infer_expr_type_with(right, extra_vars);
                if matches!(left.kind, ExprKind::Int(_) | ExprKind::Float(_))
                    && scalar_kind_of_ty(&rt, self.alias_bases).is_some()
                {
                    rt
                } else if scalar_kind_of_ty(&lt, self.alias_bases).is_some() {
                    lt
                } else if scalar_kind_of_ty(&rt, self.alias_bases).is_some() {
                    rt
                } else if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod)
                    && (is_float_type_expr(&lt, self.alias_bases)
                        || is_float_type_expr(&rt, self.alias_bases))
                {
                    named_ty("Float")
                } else {
                    named_ty("Int")
                }
            }
            ExprKind::Unary { op: UnOp::Not, .. } => named_ty("Bool"),
            ExprKind::Unary { op: op @ (UnOp::Neg | UnOp::BitNot), expr: inner } => {
                let it = self.infer_expr_type_with(inner, extra_vars);
                if scalar_kind_of_ty(&it, self.alias_bases).is_some() {
                    it
                } else if matches!(op, UnOp::Neg) && is_float_type_expr(&it, self.alias_bases) {
                    named_ty("Float")
                } else {
                    named_ty("Int")
                }
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    if let Some(vl) = self.enum_variants.get(&format!("{}.{}", ename, field)) {
                        return named_ty(&vl.enum_name);
                    }
                }
                let obj_ty = resolve_alias_ty(&self.infer_expr_type_with(target, extra_vars), self.alias_bases);
                if let TypeExpr::Named { name: rname, .. } = obj_ty {
                    if let Some(layout) = self.records.get(&rname) {
                        if let Some(fl) = layout.fields.iter().find(|f| f.name == *field) {
                            return fl.ty.clone();
                        }
                    }
                }
                named_ty("Any")
            }
            ExprKind::Index { target, .. } => {
                let tt = self.infer_expr_type_with(target, extra_vars);
                self.list_elem_ty(&tt).unwrap_or_else(|| named_ty("Any"))
            }
            ExprKind::Call { callee, args } => match &callee.kind {
                ExprKind::Field { target, field } if field == "parse" => {
                    if let ExprKind::Ident(rname) = &target.kind {
                        return generic_ty("Result", vec![named_ty(rname), named_ty("Int")]);
                    }
                    named_ty("Any")
                }
                ExprKind::Field { target, field } => {
                    if let ExprKind::Ident(ename) = &target.kind {
                        if let Some(vl) = self.enum_variants.get(&format!("{}.{}", ename, field)) {
                            return named_ty(&vl.enum_name);
                        }
                    }
                    let recv_ty = resolve_alias_ty(
                        &self.infer_expr_type_with(target, extra_vars),
                        self.alias_bases,
                    );
                    if let TypeExpr::Dyn { interface, .. } = recv_ty {
                        if let Some(layout) = self.dyn_interfaces.get(&interface) {
                            if let Some(dm) = layout.methods.iter().find(|m| m.name == *field) {
                                return dm.ret_ty.clone().unwrap_or_else(|| named_ty("Any"));
                            }
                        }
                    }
                    named_ty("Any")
                }
                ExprKind::Ident(f) => {
                    if f == "__make_closure" {
                        if let Some(a0) = args.first() {
                            if let ExprKind::Str(csym) = &a0.value.kind {
                                if let Some(meta) = self.fn_metas.get(csym) {
                                    return TypeExpr::Fn {
                                        params: meta.param_tys.iter().skip(1).cloned().collect(),
                                        ret: meta.ret_ty.clone().map(Box::new),
                                        span: expr.span,
                                    };
                                }
                            }
                        }
                    }
                    if f == "__call_closure" {
                        if let Some(a0) = args.first() {
                            let cty = resolve_alias_ty(
                                &self.infer_expr_type_with(&a0.value, extra_vars),
                                self.alias_bases,
                            );
                            if let TypeExpr::Fn { ret, .. } = cty {
                                return ret.map(|r| *r).unwrap_or_else(|| named_ty("Any"));
                            }
                        }
                    }
                    if f == "__box_dyn" {
                        if let Some(a1) = args.get(1) {
                            if let ExprKind::Str(vsym) = &a1.value.kind {
                                if let Some(iface) = self.vtable_ifaces.get(vsym) {
                                    return TypeExpr::Dyn {
                                        interface: iface.clone(),
                                        args: vec![],
                                        span: expr.span,
                                    };
                                }
                            }
                        }
                    }
                    if f == "Ok" && args.len() == 1 {
                        return generic_ty("Result", vec![self.infer_expr_type_with(&args[0].value, extra_vars), named_ty("Any")]);
                    }
                    if f == "Err" && args.len() == 1 {
                        return generic_ty("Result", vec![named_ty("Any"), self.infer_expr_type_with(&args[0].value, extra_vars)]);
                    }
                    if f == "Some" && args.len() == 1 {
                        return generic_ty("Option", vec![self.infer_expr_type_with(&args[0].value, extra_vars)]);
                    }
                    if f == "push" && args.len() == 2 {
                        let lt = self.infer_expr_type_with(&args[0].value, extra_vars);
                        if self.list_elem_ty(&lt).is_some_and(|e| e.display_name() != "Any") {
                            return lt;
                        }
                        return generic_ty("List", vec![self.infer_expr_type_with(&args[1].value, extra_vars)]);
                    }
                    if f == "list_filled" && args.len() == 2 {
                        return generic_ty("List", vec![self.infer_expr_type_with(&args[1].value, extra_vars)]);
                    }
                    if f == "list_pop_shrink" && args.len() == 1 {
                        return self.infer_expr_type_with(&args[0].value, extra_vars);
                    }
                    if f == "mutex_get" && args.len() == 1 {
                        if let TypeExpr::Named { name, args: ma, .. } =
                            resolve_alias_ty(&self.infer_expr_type_with(&args[0].value, extra_vars), self.alias_bases)
                        {
                            if name == "Mutex" && !ma.is_empty() {
                                return ma[0].clone();
                            }
                        }
                    }
                    if (f == "chan_recv" || f == "chan_try_recv") && args.len() == 1 {
                        if let TypeExpr::Named { name, args: ca, .. } =
                            resolve_alias_ty(&self.infer_expr_type_with(&args[0].value, extra_vars), self.alias_bases)
                        {
                            if name == "Channel" && !ca.is_empty() {
                                return generic_ty("Option", vec![ca[0].clone()]);
                            }
                        }
                        return generic_ty("Option", vec![named_ty("Any")]);
                    }
                    if (f == "ptr_read" || f == "volatile_read") && args.len() == 1 {
                        let pty = self.infer_expr_type_with(&args[0].value, extra_vars);
                        if let Some(elem) = self.ptr_pointee_ty(&pty) {
                            return elem;
                        }
                    }
                    if (f == "__ptr_add" || f == "ptr_add") && !args.is_empty() {
                        return self.infer_expr_type_with(&args[0].value, extra_vars);
                    }
                    if (f == "ptr_of" || f == "mut_ptr_of") && args.len() == 1 {
                        let raw_inner = self.infer_expr_type_with(&args[0].value, extra_vars);
                        let inner = match resolve_alias_ty(&raw_inner, self.alias_bases) {
                            TypeExpr::Array { elem, .. } => *elem,
                            _ => raw_inner,
                        };
                        let pname = if f == "mut_ptr_of" { "MutPtr" } else { "Ptr" };
                        return generic_ty(pname, vec![inner]);
                    }
                    if matches!(
                        f.as_str(),
                        "wrapping_add" | "wrapping_sub" | "wrapping_mul" | "saturating_add" | "saturating_sub"
                    ) && !args.is_empty()
                    {
                        return self.infer_expr_type_with(&args[0].value, extra_vars);
                    }
                    if let Some(meta) = self.fn_metas.get(f) {
                        if let Some(rt) = &meta.ret_ty {
                            return rt.clone();
                        }
                    }
                    if let Some(vl) = self.enum_variants.get(f) {
                        return named_ty(&vl.enum_name);
                    }
                    named_ty("Any")
                }
                _ => named_ty("Any"),
            },
            ExprKind::If { then_block, else_branch, .. } => {
                let then_ty = self.infer_block_type_with(then_block, extra_vars);
                if then_ty.display_name() != "Any" {
                    return then_ty;
                }
                if let Some(eb) = else_branch {
                    return self.infer_expr_type_with(eb, extra_vars);
                }
                named_ty("Any")
            }
            ExprKind::Match { scrutinee, arms } => {
                let st = self.infer_expr_type_with(scrutinee, extra_vars);
                for arm in arms {
                    let mut arm_vars = extra_vars.clone();
                    for p in &arm.patterns {
                        self.bind_pattern_types_for_infer(&st, p, &mut arm_vars);
                    }
                    let t = self.infer_expr_type_with(&arm.body, &arm_vars);
                    if t.display_name() != "Any" {
                        return t;
                    }
                }
                named_ty("Any")
            }
            ExprKind::Block(blk) => self.infer_block_type_with(blk, extra_vars),
            _ => named_ty("Any"),
        }
    }

    fn bind_pattern_types_for_infer(&self, sty: &TypeExpr, pat: &Pattern, out: &mut BTreeMap<String, TypeExpr>) {
        let base = resolve_alias_ty(sty, self.alias_bases);
        match pat {
            Pattern::Ident(s, _) if s != "_" => {
                out.insert(s.clone(), sty.clone());
            }
            Pattern::Tuple(ps, _) => {
                let elems = match &base {
                    TypeExpr::Tuple(es, _) => es.clone(),
                    _ => Vec::new(),
                };
                for (i, sp) in ps.iter().enumerate() {
                    let ety = elems.get(i).cloned().unwrap_or_else(|| named_ty("Any"));
                    self.bind_pattern_types_for_infer(&ety, sp, out);
                }
            }
            Pattern::Variant { name, binding: Some(b), .. } if b != "_" => {
                let ptys = self.resolve_variant_payload_tys(&base, name);
                out.insert(b.clone(), ptys.into_iter().next().unwrap_or_else(|| named_ty("Any")));
            }
            Pattern::VariantMulti { name, patterns, .. } => {
                let ptys = self.resolve_variant_payload_tys(&base, name);
                for (i, sp) in patterns.iter().enumerate() {
                    let pty = ptys.get(i).cloned().unwrap_or_else(|| named_ty("Any"));
                    self.bind_pattern_types_for_infer(&pty, sp, out);
                }
            }
            _ => {}
        }
    }

    fn infer_block_type_with(&self, blk: &Block, extra_vars: &BTreeMap<String, TypeExpr>) -> TypeExpr {
        let mut local = extra_vars.clone();
        for s in &blk.stmts {
            if let Stmt::Let { name, ty, value, .. } = s {
                let lt = ty.clone().unwrap_or_else(|| self.infer_expr_type_with(value, &local));
                local.insert(name.clone(), lt);
            }
        }
        blk.tail
            .as_deref()
            .map(|t| self.infer_expr_type_with(t, &local))
            .unwrap_or_else(|| named_ty("Any"))
    }

    fn resolve_field_layout(
        &self,
        obj_expr: &Expr,
        field: &str,
    ) -> Result<(usize, Option<LlvmFieldLayout>), String> {
        let obj_ty = resolve_alias_ty(&self.infer_expr_type(obj_expr), self.alias_bases);
        if let TypeExpr::Named { name: rname, .. } = obj_ty {
            if let Some(layout) = self.records.get(&rname) {
                if let Some((pos, fl)) = layout
                    .fields
                    .iter()
                    .enumerate()
                    .find(|(_, f)| f.name == field)
                {
                    return Ok((pos, Some(fl.clone())));
                }
            }
        }
        let idx = self
            .field_to_idx
            .get(field)
            .copied()
            .ok_or_else(|| format!("unknown field `{}` in LLVM codegen", field))?;
        Ok((idx, None))
    }

    fn is_rc_ident(&self, s: &str) -> bool {
        self.owned_rc_vars.contains(s)
            || self.borrowed_rc_vars.contains(s)
            || self.var_types.get(s).is_some_and(|t| self.type_is_rc(t))
    }

    fn expr_produces_rc(&self, expr: &Expr) -> bool {
        if self.type_is_rc(&self.infer_expr_type(expr)) {
            return true;
        }
        match &expr.kind {
            ExprKind::RecordLit { .. }
            | ExprKind::List(_)
            | ExprKind::ArrayRepeat { .. }
            | ExprKind::Tuple(_)
            | ExprKind::Str(_) => true,
            ExprKind::Binary { op: BinOp::Concat, .. } => true,
            ExprKind::Ident(s) => {
                s == "None"
                    || self.is_rc_ident(s)
                    || (self.enum_tags.contains_key(s) && !self.var_ptrs.contains_key(s))
            }
            ExprKind::TupleIndex { target, index } => {
                let tt = resolve_alias_ty(&self.infer_expr_type(target), self.alias_bases);
                if let TypeExpr::Tuple(es, _) = tt {
                    es.get(*index).is_some_and(|et| self.type_is_rc(et))
                } else {
                    false
                }
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    if self.enum_tags.contains_key(&format!("{}.{}", ename, field)) {
                        return true;
                    }
                }
                if let TypeExpr::Named { name: rname, .. } =
                    resolve_alias_ty(&self.infer_expr_type(target), self.alias_bases)
                {
                    if let Some(cfl) = self
                        .records
                        .get(&rname)
                        .and_then(|rl| rl.c_layout.as_ref())
                        .and_then(|cl| cl.fields.iter().find(|f| f.name == *field))
                    {
                        return self.is_c_value_type(&cfl.ty);
                    }
                }
                if let Ok((_, Some(fl))) = self.resolve_field_layout(target, field) {
                    return fl.is_rc;
                }
                false
            }
            ExprKind::Index { target, .. } => {
                let tt = self.infer_expr_type(target);
                self.list_elem_ty(&tt).is_some_and(|et| self.type_is_rc(&et))
            }
            ExprKind::Call { callee, args } => match &callee.kind {
                ExprKind::Ident(f) => {
                    if f == "mutex_get" {
                        if let Some(a0) = args.first() {
                            if let TypeExpr::Named { name, args: ma, .. } =
                                resolve_alias_ty(&self.infer_expr_type(&a0.value), self.alias_bases)
                            {
                                if name == "Mutex" && !ma.is_empty() {
                                    return self.type_is_rc(&ma[0]);
                                }
                            }
                        }
                        return false;
                    }
                    if f == "ptr_read" || f == "volatile_read" {
                        if let Some(a0) = args.first() {
                            let pty = self.infer_expr_type(&a0.value);
                            if let Some(pt) = self.ptr_pointee_ty(&pty) {
                                return self.is_c_value_type(&pt);
                            }
                        }
                        return false;
                    }
                    matches!(
                        f.as_str(),
                        "Ok" | "Err" | "Some" | "push" | "list_filled" | "list_pop_shrink" | "yield_val" | "__box_dyn" | "__make_closure"
                    )
                        || (self.enum_tags.contains_key(f) && !self.fn_metas.contains_key(f))
                        || self.fn_metas.get(f).is_some_and(|m| m.returns_rc)
                }
                ExprKind::Field { target, field } => {
                    if field == "parse"
                        || matches!(&target.kind, ExprKind::Ident(e) if self.enum_tags.contains_key(&format!("{}.{}", e, field)))
                    {
                        return true;
                    }
                    let recv_ty = resolve_alias_ty(&self.infer_expr_type(target), self.alias_bases);
                    if let TypeExpr::Dyn { interface, .. } = recv_ty {
                        if let Some(layout) = self.dyn_interfaces.get(&interface) {
                            if let Some(dm) = layout.methods.iter().find(|m| m.name == *field) {
                                return dm.returns_rc;
                            }
                        }
                    }
                    false
                }
                _ => false,
            },
            ExprKind::If { then_block, else_branch, .. } => {
                else_branch.is_some()
                    && (then_block.tail.as_deref().is_some_and(|t| self.expr_produces_rc(t))
                        || else_branch.as_deref().is_some_and(|e| self.expr_produces_rc(e)))
            }
            ExprKind::Match { arms, .. } => arms.iter().any(|a| self.expr_produces_rc(&a.body)),
            ExprKind::Block(blk) => blk.tail.as_deref().is_some_and(|t| self.expr_produces_rc(t)),
            _ => false,
        }
    }

    fn is_text_expr(&self, expr: &Expr) -> bool {
        let ty = self.infer_expr_type(expr);
        if is_text_type_expr(&ty, self.alias_bases) {
            return true;
        }
        match &expr.kind {
            ExprKind::Str(_) | ExprKind::Binary { op: BinOp::Concat, .. } => true,
            ExprKind::If { then_block, else_branch, .. } => {
                then_block.tail.as_deref().is_some_and(|t| self.is_text_expr(t))
                    || else_branch.as_deref().is_some_and(|e| self.is_text_expr(e))
            }
            ExprKind::Match { arms, .. } => arms.iter().any(|a| self.is_text_expr(&a.body)),
            ExprKind::Block(blk) => blk.tail.as_deref().is_some_and(|t| self.is_text_expr(t)),
            _ => false,
        }
    }

    fn is_float_expr(&self, expr: &Expr) -> bool {
        let ty = self.infer_expr_type(expr);
        if is_float_type_expr(&ty, self.alias_bases) {
            return true;
        }
        match &expr.kind {
            ExprKind::Float(_) => true,
            ExprKind::Unary { op: UnOp::Neg, expr: inner } => self.is_float_expr(inner),
            ExprKind::Binary { op, left, right }
                if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod) =>
            {
                self.is_float_expr(left) || self.is_float_expr(right)
            }
            ExprKind::If { then_block, else_branch, .. } => {
                then_block.tail.as_deref().is_some_and(|t| self.is_float_expr(t))
                    || else_branch.as_deref().is_some_and(|e| self.is_float_expr(e))
            }
            ExprKind::Match { arms, .. } => arms.iter().any(|a| self.is_float_expr(&a.body)),
            ExprKind::Block(blk) => blk.tail.as_deref().is_some_and(|t| self.is_float_expr(t)),
            _ => false,
        }
    }

    fn resolve_variant_payload_tys(&self, scrutinee_base: &TypeExpr, variant_name: &str) -> Vec<TypeExpr> {
        if variant_name == "Some" {
            if let TypeExpr::Named { name, args, .. } = scrutinee_base {
                if name == "Option" && !args.is_empty() {
                    return vec![args[0].clone()];
                }
            }
            return vec![named_ty("Any")];
        }
        if variant_name == "Ok" {
            if let TypeExpr::Named { name, args, .. } = scrutinee_base {
                if name == "Result" && !args.is_empty() {
                    return vec![args[0].clone()];
                }
            }
            return vec![named_ty("Any")];
        }
        if variant_name == "Err" {
            if let TypeExpr::Named { name, args, .. } = scrutinee_base {
                if name == "Result" && args.len() >= 2 {
                    return vec![args[1].clone()];
                }
            }
            return vec![named_ty("Any")];
        }
        if let Some(vl) = self.enum_variants.get(variant_name) {
            return vl.payload_tys.clone();
        }
        Vec::new()
    }

    fn emit_compound_op(
        &mut self,
        cur: &str,
        rhs: &str,
        op: AssignOp,
        is_float: bool,
        sk: Option<ScalarKind>,
    ) -> String {
        if matches!(sk, Some(ScalarKind::F16 | ScalarKind::BF16)) {
            let fop = match op {
                AssignOp::AddSet => "fadd",
                AssignOp::SubSet => "fsub",
                AssignOp::MulSet => "fmul",
                AssignOp::DivSet => "fdiv",
                _ => return rhs.to_string(),
            };
            let is_f16 = sk == Some(ScalarKind::F16);
            let cf = if is_f16 { self.emit_f16_to_float(cur) } else { self.emit_bf16_to_float(cur) };
            let rf = if is_f16 { self.emit_f16_to_float(rhs) } else { self.emit_bf16_to_float(rhs) };
            let fres = self.fresh_tmp();
            self.code.push_str(&format!("  {} = {} float {}, {}\n", fres, fop, cf, rf));
            return if is_f16 { self.emit_float_to_f16(&fres) } else { self.emit_float_to_bf16(&fres) };
        }
        if sk == Some(ScalarKind::F32) {
            let fop = match op {
                AssignOp::AddSet => "fadd",
                AssignOp::SubSet => "fsub",
                AssignOp::MulSet => "fmul",
                AssignOp::DivSet => "fdiv",
                _ => return rhs.to_string(),
            };
            let c32 = self.fresh_tmp();
            let r32 = self.fresh_tmp();
            let cf = self.fresh_tmp();
            let rf = self.fresh_tmp();
            let fres = self.fresh_tmp();
            let b32 = self.fresh_tmp();
            let res = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = trunc i64 {} to i32\n  {} = trunc i64 {} to i32\n  {} = bitcast i32 {} to float\n  {} = bitcast i32 {} to float\n  {} = {} float {}, {}\n  {} = bitcast float {} to i32\n  {} = zext i32 {} to i64\n",
                c32, cur, r32, rhs, cf, c32, rf, r32, fres, fop, cf, rf, b32, fres, res, b32
            ));
            return res;
        }
        if is_float {
            let cf = self.fresh_tmp();
            let rf = self.fresh_tmp();
            let fop = match op {
                AssignOp::AddSet => "fadd",
                AssignOp::SubSet => "fsub",
                AssignOp::MulSet => "fmul",
                AssignOp::DivSet => "fdiv",
                _ => return rhs.to_string(),
            };
            let fres = self.fresh_tmp();
            let res = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = bitcast i64 {} to double\n  {} = bitcast i64 {} to double\n  {} = {} double {}, {}\n  {} = bitcast double {} to i64\n",
                cf, cur, rf, rhs, fres, fop, cf, rf, res, fres
            ));
            return res;
        }
        let is_unsigned = sk.is_some_and(|k| k.is_unsigned());
        let is_signed_narrow = matches!(sk, Some(ScalarKind::I8 | ScalarKind::I16 | ScalarKind::I32));
        let iop = match op {
            AssignOp::Set => return rhs.to_string(),
            AssignOp::AddSet => "add",
            AssignOp::SubSet => "sub",
            AssignOp::MulSet => "mul",
            AssignOp::DivSet => if is_unsigned { "udiv" } else { "sdiv" },
            AssignOp::BitAndSet => "and",
            AssignOp::BitOrSet => "or",
            AssignOp::BitXorSet => "xor",
            AssignOp::ShlSet => "shl",
            AssignOp::ShrSet => if is_signed_narrow { "ashr" } else { "lshr" },
        };
        let res = self.fresh_tmp();
        self.code
            .push_str(&format!("  {} = {} i64 {}, {}\n", res, iop, cur, rhs));
        if let Some(k) = sk {
            self.normalize_scalar(&res, k)
        } else {
            res
        }
    }

    fn infer_match_binding_type(&self, scrutinee_ty: &TypeExpr, variant_name: &str) -> TypeExpr {
        let base = resolve_alias_ty(scrutinee_ty, self.alias_bases);
        if variant_name == "Some" {
            if let TypeExpr::Named { name, args, .. } = &base {
                if name == "Option" && !args.is_empty() {
                    return args[0].clone();
                }
            }
        }
        if variant_name == "Ok" {
            if let TypeExpr::Named { name, args, .. } = &base {
                if name == "Result" && !args.is_empty() {
                    return args[0].clone();
                }
            }
        }
        if variant_name == "Err" {
            if let TypeExpr::Named { name, args, .. } = &base {
                if name == "Result" && args.len() >= 2 {
                    return args[1].clone();
                }
            }
        }
        if let Some(vl) = self.enum_variants.get(variant_name) {
            if let Some(pty) = vl.payload_tys.first() {
                return pty.clone();
            }
        }
        named_ty("Any")
    }

    fn emit_sync_fn(&mut self, f: &FnDecl) -> Result<String, String> {
        let mut param_list = Vec::new();
        for (i, p) in f.params.iter().enumerate() {
            let pname = format!("%arg_{}", i);
            param_list.push(format!("i64 {}", pname));
            let slot = self.fresh_var_slot(&p.name);
            self.code
                .push_str(&format!("  store i64 {}, ptr {}\n", pname, slot));
            self.var_types.insert(p.name.clone(), p.ty.clone());
            if self.type_is_rc(&p.ty) {
                self.borrowed_rc_vars.insert(p.name.clone());
            }
            if p.is_inout {
                self.inout_vars.insert(p.name.clone());
            }
        }

        self.fn_returns_rc = self.fn_metas.get(&f.name).is_some_and(|m| m.returns_rc);
        self.fn_ret_ty = f.ret_ty.clone();
        self.always_unique_vars = collect_always_unique_locals(&f.body);
        if let Some(ne) = self.non_escaping_vars.get(&f.name) {
            self.always_unique_vars.extend(ne.iter().cloned());
        }
        self.cur_stack_promotable = self
            .stack_promotable_vars
            .get(&f.name)
            .cloned()
            .unwrap_or_default();
        self.cur_last_use_move_spans = self
            .last_use_move_spans
            .get(&f.name)
            .cloned()
            .unwrap_or_default();
        self.cur_fn_preemptible = f.attrs.iter().any(|a| a.name == "preemptible")
            || f.name.starts_with("__closure_");
        if f.attrs.iter().any(|a| a.name == "arena") {
            let am = self.fresh_tmp();
            self.code
                .push_str(&format!("  {} = call i64 @tenon_arena_enter()\n", am));
            self.active_arena_mark = Some(am);
        } else {
            self.active_arena_mark = None;
        }

        let ret_val = self.lower_block(&f.body, self.fn_returns_rc, f.ret_ty.as_ref())?;
        if !self.terminated {
            if let Some(am) = &self.active_arena_mark {
                self.code
                    .push_str(&format!("  call void @tenon_arena_exit(i64 {})\n", am));
            }
            let dbg = self.dbg_suffix(f.body.span);
            self.code.push_str(&format!("  ret i64 {}{}\n", ret_val, dbg));
        }

        let linkage = if f.name == "main" { "" } else { "internal " };
        let mut fn_attrs = String::new();
        for attr in &f.attrs {
            if attr.name == "inline" {
                match attr.args.first().map(|a| &a.value.kind) {
                    Some(ExprKind::Ident(s) | ExprKind::Str(s)) if s == "always" => {
                        fn_attrs.push_str(" alwaysinline");
                    }
                    Some(ExprKind::Ident(s) | ExprKind::Str(s)) if s == "never" => {
                        fn_attrs.push_str(" noinline");
                    }
                    None => fn_attrs.push_str(" alwaysinline"),
                    _ => {}
                }
            } else if attr.name == "cold" {
                fn_attrs.push_str(" cold");
            }
        }
        let sp_dbg = self
            .debug_sp_id
            .map(|id| format!(" !dbg !{}", id))
            .unwrap_or_default();
        Ok(format!(
            "define {}i64 @tenon_fn_{}({}){}{} {{\nentry:\n{}{}}}\n",
            linkage,
            sanitize(&f.name),
            param_list.join(", "),
            fn_attrs,
            sp_dbg,
            self.entry_allocas,
            self.code
        ))
    }

    fn emit_record_parser(
        &mut self,
        rname: &str,
        layout: &LlvmRecordLayout,
    ) -> Result<String, String> {
        let mut param_list = Vec::new();
        for (i, fl) in layout.fields.iter().enumerate() {
            let pname = format!("%arg_{}", i);
            param_list.push(format!("i64 {}", pname));
            let slot = self.fresh_var_slot(&fl.name);
            self.code
                .push_str(&format!("  store i64 {}, ptr {}\n", pname, slot));
            self.var_types.insert(fl.name.clone(), fl.ty.clone());
            if fl.is_rc {
                self.borrowed_rc_vars.insert(fl.name.clone());
            }
        }

        for (idx, fl) in layout.fields.iter().enumerate() {
            for pred in &fl.preds {
                self.self_var = Some(fl.name.clone());
                let mut temps = Vec::new();
                let cv = self.lower_expr_borrowed(pred, &mut temps)?;
                self.flush_temps(&temps);
                self.self_var = None;
                let cond = self.fresh_tmp();
                let fail_l = self.fresh_lbl("pfail");
                let ok_l = self.fresh_lbl("pok");
                self.code.push_str(&format!(
                    "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n",
                    cond, cv, cond, ok_l, fail_l
                ));
                self.code.push_str(&format!("{}:\n", fail_l));
                let eo = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 1)\n  call void @__tn_slot_set_prim(i64 {}, i64 0, i64 {})\n  ret i64 {}\n",
                    eo, TAG_ERR, eo, idx + 1, eo
                ));
                self.code.push_str(&format!("{}:\n", ok_l));
            }
        }

        for wc in &layout.where_clauses {
            let mut temps = Vec::new();
            let cv = self.lower_expr_borrowed(wc, &mut temps)?;
            self.flush_temps(&temps);
            let cond = self.fresh_tmp();
            let fail_l = self.fresh_lbl("wfail");
            let ok_l = self.fresh_lbl("wok");
            self.code.push_str(&format!(
                "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n",
                cond, cv, cond, ok_l, fail_l
            ));
            self.code.push_str(&format!("{}:\n", fail_l));
            let eo = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 1)\n  call void @__tn_slot_set_prim(i64 {}, i64 0, i64 99)\n  ret i64 {}\n",
                eo, TAG_ERR, eo, eo
            ));
            self.code.push_str(&format!("{}:\n", ok_l));
        }

        let rec = self.fresh_tmp();
        self.code.push_str(&format!(
            "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 {})\n",
            rec,
            layout.tag,
            layout.fields.len()
        ));
        if layout.rc_mask != 0 {
            self.code.push_str(&format!(
                "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                rec, layout.rc_mask
            ));
        }
        for (idx, fl) in layout.fields.iter().enumerate() {
            let slot = self.var_ptrs[&fl.name].clone();
            let mut fv = self.fresh_tmp();
            self.code
                .push_str(&format!("  {} = load i64, ptr {}\n", fv, slot));
            if fl.is_rc {
                let inc = self.fresh_tmp();
                self.code
                    .push_str(&format!("  {} = call i64 @__tn_rc_inc(i64 {})\n", inc, fv));
                fv = inc;
            }
            self.code.push_str(&format!(
                "  call void @__tn_slot_set_prim(i64 {}, i64 {}, i64 {})\n",
                rec, idx, fv
            ));
        }
        let ok_obj = self.fresh_tmp();
        self.code.push_str(&format!(
            "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 1)\n  call void @__tn_rc_set_rc_mask(i64 {}, i64 1)\n  call void @__tn_slot_set_prim(i64 {}, i64 0, i64 {})\n  ret i64 {}\n",
            ok_obj, TAG_OK, ok_obj, ok_obj, rec, ok_obj
        ));

        Ok(format!(
            "define internal i64 @tenon_fn_{}__parse({}) {{\nentry:\n{}{}}}\n",
            sanitize(rname),
            param_list.join(", "),
            self.entry_allocas,
            self.code
        ))
    }

    fn emit_async_fn(&mut self, f: &FnDecl) -> Result<String, String> {
        let mut frame_slots: BTreeMap<String, i64> = BTreeMap::new();
        let mut next_slot = 4i64;
        for p in &f.params {
            frame_slots.insert(p.name.clone(), next_slot);
            next_slot += 1;
        }
        for s in &f.body.stmts {
            if let Stmt::Let { name, .. } = s {
                if !frame_slots.contains_key(name) {
                    frame_slots.insert(name.clone(), next_slot);
                    next_slot += 1;
                }
            }
        }
        let segments = split_async_segments(&f.body, &mut frame_slots, &mut next_slot);
        let total_slots = next_slot;
        let sname = sanitize(&f.name);

        let mut ctor_params = Vec::new();
        let mut ctor_body = String::new();
        ctor_body.push_str(&format!(
            "  %pp = ptrtoint ptr @tenon_fn_{}__poll to i64\n  %fr = call i64 @tenon_async_frame_alloc(i64 {}, i64 %pp)\n",
            sname, total_slots
        ));
        for (i, p) in f.params.iter().enumerate() {
            ctor_params.push(format!("i64 %arg_{}", i));
            ctor_body.push_str(&format!(
                "  call void @__tn_slot_set_prim(i64 %fr, i64 {}, i64 %arg_{})\n",
                frame_slots[&p.name], i
            ));
        }
        ctor_body.push_str("  ret i64 %fr\n");
        let ctor_ir = format!(
            "define internal i64 @tenon_fn_{}({}) {{\nentry:\n{}}}\n\n",
            sname,
            ctor_params.join(", "),
            ctor_body
        );

        for (vname, &slot) in &frame_slots {
            let vp = self.ensure_var_slot(vname);
            let lv = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = call i64 @__tn_slot_get(i64 %frame, i64 {})\n  store i64 {}, ptr {}\n",
                lv, slot, lv, vp
            ));
        }
        let cur_st = self.fresh_tmp();
        self.code
            .push_str(&format!("  {} = call i64 @__tn_slot_get(i64 %frame, i64 0)\n", cur_st));

        let n_awaits = segments.await_steps.len();
        let mut start_lbls = Vec::new();
        let mut poll_lbls = Vec::new();
        for s in 0..n_awaits {
            start_lbls.push(format!("st_start_{}", s));
            poll_lbls.push(format!("st_poll_{}", s));
        }
        let final_lbl = "st_final".to_string();

        self.code.push_str(&format!("  switch i64 {}, label %{} [\n", cur_st, final_lbl));
        for s in 0..n_awaits {
            self.code.push_str(&format!("    i64 {}, label %{}\n", 2 * s, start_lbls[s]));
            self.code.push_str(&format!("    i64 {}, label %{}\n", 2 * s + 1, poll_lbls[s]));
        }
        self.code.push_str("  ]\n");

        for (s, step) in segments.await_steps.iter().enumerate() {
            self.code.push_str(&format!("{}:\n", start_lbls[s]));
            self.terminated = false;
            let pre_blk = Block {
                stmts: step.pre_stmts.clone(),
                tail: None,
                span: f.span,
            };
            let _ = self.lower_block(&pre_blk, false, None)?;
            for (vname, &slot) in &frame_slots {
                let vp = self.var_ptrs[vname].clone();
                let lv = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = load i64, ptr {}\n  call void @__tn_slot_set_prim(i64 %frame, i64 {}, i64 {})\n",
                    lv, vp, slot, lv
                ));
            }
            let cframe = self.lower_expr_owned(&step.child_expr, None)?;
            self.code.push_str(&format!(
                "  call void @__tn_slot_set_prim(i64 %frame, i64 3, i64 {})\n  call void @__tn_slot_set_prim(i64 %frame, i64 0, i64 {})\n  br label %{}\n",
                cframe,
                2 * s + 1,
                poll_lbls[s]
            ));

            self.code.push_str(&format!("{}:\n", poll_lbls[s]));
            self.terminated = false;
            let cf = self.fresh_tmp();
            let rdy = self.fresh_tmp();
            let cmp = self.fresh_tmp();
            let pend_l = self.fresh_lbl("pend");
            let res_l = self.fresh_lbl("res");
            self.code.push_str(&format!(
                "  {} = call i64 @__tn_slot_get(i64 %frame, i64 3)\n  {} = call i64 @tenon_async_poll_frame(i64 {})\n  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n",
                cf, rdy, cf, cmp, rdy, cmp, res_l, pend_l
            ));
            self.code.push_str(&format!("{}:\n  ret i64 0\n", pend_l));
            self.code.push_str(&format!("{}:\n", res_l));
            let cres = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = call i64 @__tn_slot_get(i64 {}, i64 2)\n  call void @__tn_rc_dec(i64 {})\n  call void @__tn_slot_set_prim(i64 %frame, i64 3, i64 0)\n",
                cres, cf, cf
            ));
            if let Some(bname) = &step.bind_var {
                let slot = frame_slots[bname];
                let vp = self.var_ptrs[bname].clone();
                self.code.push_str(&format!(
                    "  call void @__tn_slot_set_prim(i64 %frame, i64 {}, i64 {})\n  store i64 {}, ptr {}\n",
                    slot, cres, cres, vp
                ));
            }
            let next_lbl = if s + 1 < n_awaits {
                start_lbls[s + 1].as_str()
            } else {
                final_lbl.as_str()
            };
            self.code.push_str(&format!(
                "  call void @__tn_slot_set_prim(i64 %frame, i64 0, i64 {})\n  br label %{}\n",
                2 * (s + 1),
                next_lbl
            ));
        }

        self.code.push_str(&format!("{}:\n", final_lbl));
        self.terminated = false;
        let final_blk = Block {
            stmts: segments.final_stmts.clone(),
            tail: segments.final_tail.clone().map(Box::new),
            span: f.span,
        };
        let fval = self.lower_block(&final_blk, true, None)?;
        self.code.push_str(&format!(
            "  call void @__tn_slot_set_prim(i64 %frame, i64 2, i64 {})\n  call void @__tn_slot_set_prim(i64 %frame, i64 0, i64 -1)\n  ret i64 1\n",
            fval
        ));

        let poll_ir = format!(
            "define internal i64 @tenon_fn_{}__poll(i64 %frame) {{\nentry:\n{}{}}}\n",
            sname, self.entry_allocas, self.code
        );
        Ok(format!("{}{}", ctor_ir, poll_ir))
    }

    fn load_ident_val(&mut self, s: &str) -> Result<String, String> {
        let slot = self
            .var_ptrs
            .get(s)
            .cloned()
            .ok_or_else(|| format!("undefined variable `{}` in LLVM codegen", s))?;
        if self.inout_vars.contains(s) {
            let ai = self.fresh_tmp();
            let ap = self.fresh_tmp();
            let v = self.fresh_tmp();
            self.code.push_str(&format!(
                "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  {} = load i64, ptr {}\n",
                ai, slot, ap, ai, v, ap
            ));
            Ok(v)
        } else {
            let v = self.fresh_tmp();
            self.code
                .push_str(&format!("  {} = load i64, ptr {}\n", v, slot));
            Ok(v)
        }
    }

    fn lower_block(
        &mut self,
        blk: &Block,
        want_owned_tail: bool,
        expected_tail_ty: Option<&TypeExpr>,
    ) -> Result<String, String> {
        let saved = self.save_lexical();
        let scope_idx = self.rc_scopes.len();
        self.rc_scopes.push(Vec::new());

        let mut last = "0".to_string();
        for stmt in &blk.stmts {
            if self.terminated {
                break;
            }
            match stmt {
                Stmt::Let {
                    name, ty, value, span, ..
                } => {
                    let effective_ty = ty.clone().unwrap_or_else(|| self.infer_expr_type(value));
                    // Tier-2 Stack Promotion: non-escaping immutable scalar-only RecordLit -> LLVM entry `alloca`!
                    if self.cur_stack_promotable.contains(name) {
                        if let ExprKind::RecordLit {
                            name: rname,
                            fields,
                            ..
                        } = &value.kind
                        {
                            if let Some(layout) = self.records.get(rname).cloned() {
                                if layout.rc_mask == 0 && layout.c_layout.is_none() {
                                    let nwords = 5 + layout.fields.len().max(1);
                                    self.slot_id += 1;
                                    let stk_arr = format!("%stkrec_{}", self.slot_id);
                                    self.entry_allocas.push_str(&format!(
                                        "  {} = alloca [{} x i64], align 8\n",
                                        stk_arr, nwords
                                    ));
                                    let obj = self.fresh_tmp();
                                    let p_rc = self.fresh_tmp();
                                    let p_tag = self.fresh_tmp();
                                    let p_len = self.fresh_tmp();
                                    let p_cap = self.fresh_tmp();
                                    let p_mask = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = getelementptr inbounds [{} x i64], ptr {}, i64 0, i64 0\n  store i64 -2, ptr {}, align 8\n  {} = getelementptr inbounds [{} x i64], ptr {}, i64 0, i64 1\n  store i64 {}, ptr {}, align 8\n  {} = getelementptr inbounds [{} x i64], ptr {}, i64 0, i64 2\n  store i64 {}, ptr {}, align 8\n  {} = getelementptr inbounds [{} x i64], ptr {}, i64 0, i64 3\n  store i64 {}, ptr {}, align 8\n  {} = getelementptr inbounds [{} x i64], ptr {}, i64 0, i64 4\n  store i64 0, ptr {}, align 8\n",
                                        p_rc, nwords, stk_arr, p_rc,
                                        p_tag, nwords, stk_arr, layout.tag, p_tag,
                                        p_len, nwords, stk_arr, layout.fields.len(), p_len,
                                        p_cap, nwords, stk_arr, layout.fields.len().max(1), p_cap,
                                        p_mask, nwords, stk_arr, p_mask
                                    ));
                                    for (fname, fexpr) in fields {
                                        if let Some((idx, _)) = layout
                                            .fields
                                            .iter()
                                            .enumerate()
                                            .find(|(_, f)| f.name == *fname)
                                        {
                                            let mut temps = Vec::new();
                                            let fv = self.lower_expr_borrowed(fexpr, &mut temps)?;
                                            self.flush_temps(&temps);
                                            let fp = self.fresh_tmp();
                                            self.code.push_str(&format!(
                                                "  {} = getelementptr inbounds [{} x i64], ptr {}, i64 0, i64 {}\n  store i64 {}, ptr {}, align 8, !tbaa !7\n",
                                                fp, nwords, stk_arr, 5 + idx, fv, fp
                                            ));
                                        }
                                    }
                                    self.code.push_str(&format!(
                                        "  {} = ptrtoint ptr {} to i64\n",
                                        obj, stk_arr
                                    ));
                                    let slot = self.fresh_var_slot(name);
                                    let dbg = self.dbg_suffix(*span);
                                    self.code.push_str(&format!(
                                        "  store i64 {}, ptr {}{}\n",
                                        obj, slot, dbg
                                    ));
                                    self.var_types.insert(name.clone(), effective_ty);
                                    self.owned_rc_vars.remove(name);
                                    self.borrowed_rc_vars.insert(name.clone());
                                    continue;
                                }
                            }
                        }
                    }
                    let produces_rc = self.type_is_rc(&effective_ty) || self.expr_produces_rc(value);
                    let mut v = if produces_rc {
                        self.lower_expr_owned(value, ty.as_ref())?
                    } else {
                        let mut temps = Vec::new();
                        let val = self.lower_expr_borrowed(value, &mut temps)?;
                        self.flush_temps(&temps);
                        val
                    };
                    if let Some(sk) = scalar_kind_of_ty(&effective_ty, self.alias_bases) {
                        if sk.is_int() {
                            v = self.normalize_scalar(&v, sk);
                        }
                    }
                    let slot = self.fresh_var_slot(name);
                    let dbg = self.dbg_suffix(*span);
                    self.code
                        .push_str(&format!("  store i64 {}, ptr {}{}\n", v, slot, dbg));

                    self.var_types.insert(name.clone(), effective_ty);
                    self.owned_rc_vars.remove(name);
                    self.borrowed_rc_vars.remove(name);
                    if produces_rc {
                        self.owned_rc_vars.insert(name.clone());
                        self.rc_scopes[scope_idx].push((name.clone(), slot));
                    }
                }
                Stmt::Assign {
                    target, op, value, ..
                } => match &target.kind {
                    ExprKind::Ident(name) => {
                        let consuming_push = matches!(
                            &value.kind,
                            ExprKind::Call { callee, args }
                            if matches!(&callee.kind, ExprKind::Ident(f) if f == "push")
                                && args.len() == 2
                                && args.first().is_some_and(|a| matches!(&a.value.kind, ExprKind::Ident(s) if s == name))
                        );
                        let consuming_pop = matches!(
                            &value.kind,
                            ExprKind::Call { callee, args }
                            if matches!(&callee.kind, ExprKind::Ident(f) if f == "list_pop_shrink")
                                && args.len() == 1
                                && args.first().is_some_and(|a| matches!(&a.value.kind, ExprKind::Ident(s) if s == name))
                        );
                        let consuming_cow = matches!(
                            &value.kind,
                            ExprKind::Call { callee, args }
                            if matches!(&callee.kind, ExprKind::Ident(f) if is_consuming_cow_builtin(f))
                                && !args.is_empty()
                                && args.first().is_some_and(|a| matches!(&a.value.kind, ExprKind::Ident(s) if s == name))
                        );
                        let consuming_in_place = consuming_push || consuming_pop || consuming_cow;
                        let target_ty = self.var_types.get(name).cloned();
                        let is_inout = self.inout_vars.contains(name);
                        let target_is_rc = self.owned_rc_vars.contains(name);
                        let inout_is_rc = is_inout
                            && target_ty.as_ref().is_some_and(|t| self.type_is_rc(t));
                        let rhs = if consuming_push {
                            let ExprKind::Call { args, .. } = &value.kind else {
                                unreachable!()
                            };
                            if let Some((vec_ty, elem_bytes, _)) = target_ty
                                .as_ref()
                                .and_then(|t| self.flat_simd_list_info(t))
                                .or_else(|| self.simd_elem_info(&self.infer_expr_type(&args[1].value)))
                            {
                                let xs_val = self.load_ident_val(name)?;
                                let mut temps = Vec::new();
                                let vv = self.lower_simd_vec_reg(&args[1].value, vec_ty, elem_bytes, &mut temps)?;
                                self.slot_id += 1;
                                let vslot = format!("%fpush_{}", self.slot_id);
                                self.entry_allocas
                                    .push_str(&format!("  {} = alloca {}, align 8\n", vslot, vec_ty));
                                let res = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  store {} {}, ptr {}, align 8\n  {} = call i64 @tenon_flat_list_push(i64 {}, i64 {}, ptr {})\n",
                                    vec_ty, vv, vslot, res, xs_val, elem_bytes, vslot
                                ));
                                self.flush_temps(&temps);
                                res
                            } else if target_ty
                                .as_ref()
                                .and_then(|t| self.flat_scalar2_list_info(t))
                                .or_else(|| self.scalar2_elem_info(&self.infer_expr_type(&args[1].value)))
                                .is_some()
                            {
                                let xs_val = self.load_ident_val(name)?;
                                let mut temps = Vec::new();
                                let vv = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                                let v16 = self.fresh_tmp();
                                self.slot_id += 1;
                                let vslot = format!("%fs2push_{}", self.slot_id);
                                self.entry_allocas.push_str(&format!("  {} = alloca i16, align 2\n", vslot));
                                let res = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = trunc i64 {} to i16\n  store i16 {}, ptr {}, align 2\n  {} = call i64 @tenon_flat_list_push(i64 {}, i64 2, ptr {})\n",
                                    v16, vv, v16, vslot, res, xs_val, vslot
                                ));
                                self.flush_temps(&temps);
                                res
                            } else if target_ty
                                .as_ref()
                                .and_then(|t| self.flat_scalar4_list_info(t))
                                .or_else(|| self.scalar4_elem_info(&self.infer_expr_type(&args[1].value)))
                                .is_some()
                            {
                                let xs_val = self.load_ident_val(name)?;
                                let mut temps = Vec::new();
                                let vv = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                                let v32 = self.fresh_tmp();
                                self.slot_id += 1;
                                let vslot = format!("%fs4push_{}", self.slot_id);
                                self.entry_allocas.push_str(&format!("  {} = alloca i32, align 4\n", vslot));
                                let res = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = trunc i64 {} to i32\n  store i32 {}, ptr {}, align 4\n  {} = call i64 @tenon_flat_list_push(i64 {}, i64 4, ptr {})\n",
                                    v32, vv, v32, vslot, res, xs_val, vslot
                                ));
                                self.flush_temps(&temps);
                                res
                            } else {
                            let elem_is_rc = target_ty
                                .as_ref()
                                .and_then(|t| self.list_elem_ty(t))
                                .is_some_and(|et| self.type_is_rc(&et))
                                || self.expr_produces_rc(&args[1].value);
                            let xs_val = self.load_ident_val(name)?;
                            let val_v = if elem_is_rc {
                                self.code.push_str(&format!(
                                    "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                                    xs_val, RC_MASK_ALL as i64
                                ));
                                self.lower_expr_owned(&args[1].value, None)?
                            } else {
                                let mut temps = Vec::new();
                                let v = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                                self.flush_temps(&temps);
                                v
                            };
                            let res = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = call i64 @tenon_list_push(i64 {}, i64 {})\n",
                                res, xs_val, val_v
                            ));
                            res
                            }
                        } else if consuming_pop {
                            let xs_val = self.load_ident_val(name)?;
                            let res = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = call i64 @tenon_list_pop_shrink(i64 {})\n",
                                res, xs_val
                            ));
                            res
                        } else if consuming_cow {
                            let ExprKind::Call { callee, args } = &value.kind else {
                                unreachable!()
                            };
                            let ExprKind::Ident(fname) = &callee.kind else {
                                unreachable!()
                            };
                            if fname == "simd_u8x16_set_lane" && args.len() == 3 {
                                let vh = self.load_ident_val(name)?;
                                let mut temps = Vec::new();
                                let idx = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                                let val = self.lower_expr_borrowed(&args[2].value, &mut temps)?;
                                let uniq = self.fresh_tmp();
                                let di = self.fresh_tmp();
                                let dp = self.fresh_tmp();
                                let ep = self.fresh_tmp();
                                let b8 = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 {}\n  {} = trunc i64 {} to i8\n  store i8 {}, ptr {}, align 1\n",
                                    uniq, vh, di, uniq, dp, di, ep, dp, idx, b8, val, b8, ep
                                ));
                                self.flush_temps(&temps);
                                uniq
                            } else {
                                let csym = *self
                                    .syscall_map
                                    .get(fname)
                                    .ok_or_else(|| format!("unknown cow builtin `{}`", fname))?;
                                let mut temps = Vec::new();
                                let mut avals = Vec::with_capacity(args.len());
                                avals.push(format!("i64 {}", self.load_ident_val(name)?));
                                for a in &args[1..] {
                                    avals.push(format!(
                                        "i64 {}",
                                        self.lower_expr_borrowed(&a.value, &mut temps)?
                                    ));
                                }
                                let res = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = call i64 @{}({})\n",
                                    res,
                                    csym,
                                    avals.join(", ")
                                ));
                                self.flush_temps(&temps);
                                res
                            }
                        } else if (target_is_rc || inout_is_rc) && *op == AssignOp::Set {
                            self.lower_expr_owned(value, target_ty.as_ref())?
                        } else {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(value, &mut temps)?;
                            self.flush_temps(&temps);
                            v
                        };
                        let slot = self
                            .var_ptrs
                            .get(name)
                            .cloned()
                            .ok_or_else(|| format!("undefined variable `{}` in LLVM assign", name))?;
                        let sk = target_ty
                            .as_ref()
                            .and_then(|t| scalar_kind_of_ty(t, self.alias_bases))
                            .or_else(|| self.scalar_kind_of_expr(value));
                        if target_is_rc && !consuming_in_place && !is_inout {
                            let old = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = load i64, ptr {}\n  store i64 {}, ptr {}\n  call void @__tn_rc_dec(i64 {})\n",
                                old, slot, rhs, slot, old
                            ));
                        } else if is_inout {
                            let ai = self.fresh_tmp();
                            let ap = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n",
                                ai, slot, ap, ai
                            ));
                            if inout_is_rc && !consuming_in_place && *op == AssignOp::Set {
                                let old = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = load i64, ptr {}\n  store i64 {}, ptr {}\n  call void @__tn_rc_dec(i64 {})\n",
                                    old, ap, rhs, ap, old
                                ));
                            } else {
                                let is_flt = target_ty
                                    .as_ref()
                                    .is_some_and(|t| is_float_type_expr(t, self.alias_bases))
                                    || self.is_float_expr(value);
                                let new_v = if *op == AssignOp::Set {
                                    if let Some(k) = sk {
                                        if k.is_int() { self.normalize_scalar(&rhs, k) } else { rhs }
                                    } else {
                                        rhs
                                    }
                                } else {
                                    let cur = self.fresh_tmp();
                                    self.code.push_str(&format!("  {} = load i64, ptr {}\n", cur, ap));
                                    self.emit_compound_op(&cur, &rhs, *op, is_flt, sk)
                                };
                                self.code
                                    .push_str(&format!("  store i64 {}, ptr {}\n", new_v, ap));
                            }
                        } else {
                            let is_flt = target_ty
                                .as_ref()
                                .is_some_and(|t| is_float_type_expr(t, self.alias_bases))
                                || self.is_float_expr(value);
                            let new_v = if *op == AssignOp::Set {
                                if let Some(k) = sk {
                                    if k.is_int() { self.normalize_scalar(&rhs, k) } else { rhs }
                                } else {
                                    rhs
                                }
                            } else {
                                let cur = self.fresh_tmp();
                                self.code.push_str(&format!("  {} = load i64, ptr {}\n", cur, slot));
                                self.emit_compound_op(&cur, &rhs, *op, is_flt, sk)
                            };
                            self.code
                                .push_str(&format!("  store i64 {}, ptr {}\n", new_v, slot));
                        }
                    }
                    ExprKind::Field { target: obj, field } => {
                        if let ExprKind::Ident(vname) = &obj.kind {
                            let c_field_info = match resolve_alias_ty(&self.infer_expr_type(obj), self.alias_bases) {
                                TypeExpr::Named { name: rname, .. } => self
                                    .records
                                    .get(&rname)
                                    .and_then(|rl| rl.c_layout.as_ref())
                                    .and_then(|cl| cl.fields.iter().find(|f| f.name == *field))
                                    .cloned(),
                                _ => None,
                            };
                            if let Some(cfl) = c_field_info {
                                let mut temps = Vec::new();
                                let rhs = self.lower_expr_borrowed(value, &mut temps)?;
                                let cur_h = self.load_ident_val(vname)?;
                                let uniq = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n",
                                    uniq, cur_h
                                ));
                                let slot = self.var_ptrs[vname].clone();
                                if self.inout_vars.contains(vname) {
                                    let addr_i = self.fresh_tmp();
                                    let addr_p = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                        addr_i, slot, addr_p, addr_i, uniq, addr_p
                                    ));
                                } else {
                                    self.code.push_str(&format!("  store i64 {}, ptr {}\n", uniq, slot));
                                }
                                let base = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n",
                                    base, uniq
                                ));
                                let sk = scalar_kind_of_ty(&cfl.ty, self.alias_bases);
                                let is_flt = is_float_type_expr(&cfl.ty, self.alias_bases) || self.is_float_expr(value);
                                let new_fv = if *op == AssignOp::Set {
                                    rhs
                                } else {
                                    let cf = self.emit_mem_load(&base, cfl.byte_offset as i64, &cfl.ty, false)?;
                                    self.emit_compound_op(&cf, &rhs, *op, is_flt, sk)
                                };
                                self.emit_mem_store(&base, cfl.byte_offset as i64, &new_fv, &cfl.ty, false)?;
                                self.flush_temps(&temps);
                            } else {
                                let (fidx, fl) = self.resolve_field_layout(obj, field)?;
                                let field_is_rc = fl.as_ref().is_some_and(|f| f.is_rc);
                                let field_ty = fl.as_ref().map(|f| &f.ty);
                                let rhs = if field_is_rc && *op == AssignOp::Set {
                                    self.lower_expr_owned(value, field_ty)?
                                } else {
                                    let mut temps = Vec::new();
                                    let v = self.lower_expr_borrowed(value, &mut temps)?;
                                    self.flush_temps(&temps);
                                    v
                                };
                                let slot = self.var_ptrs[vname].clone();
                                let is_inout = self.inout_vars.contains(vname);
                                let cur_h = if is_inout {
                                    let addr_i = self.fresh_tmp();
                                    let addr_p = self.fresh_tmp();
                                    let val = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  {} = load i64, ptr {}\n",
                                        addr_i, slot, addr_p, addr_i, val, addr_p
                                    ));
                                    val
                                } else {
                                    let val = self.fresh_tmp();
                                    self.code
                                        .push_str(&format!("  {} = load i64, ptr {}\n", val, slot));
                                    val
                                };
                                let is_always_uniq = self.always_unique_vars.contains(vname)
                                    || self.cur_stack_promotable.contains(vname);
                                let uniq = if is_always_uniq {
                                    cur_h
                                } else {
                                    let u = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n",
                                        u, cur_h
                                    ));
                                    if is_inout {
                                        let addr_i = self.fresh_tmp();
                                        let addr_p = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                            addr_i, slot, addr_p, addr_i, u, addr_p
                                        ));
                                    } else {
                                        self.code.push_str(&format!(
                                            "  store i64 {}, ptr {}\n",
                                            u, slot
                                        ));
                                    }
                                    u
                                };
                                let sk = field_ty.and_then(|t| scalar_kind_of_ty(t, self.alias_bases));
                                let is_flt = field_ty
                                    .is_some_and(|t| is_float_type_expr(t, self.alias_bases))
                                    || self.is_float_expr(value);
                                let new_fv = if *op == AssignOp::Set {
                                    rhs
                                } else {
                                    let cf = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n",
                                        cf, uniq, fidx
                                    ));
                                    self.emit_compound_op(&cf, &rhs, *op, is_flt, sk)
                                };
                                let setter = if *op != AssignOp::Set {
                                    "@__tn_slot_set_prim"
                                } else if let Some(ref f) = fl {
                                    if f.is_rc {
                                        "@__tn_slot_set_rc"
                                    } else {
                                        "@__tn_slot_set_prim"
                                    }
                                } else {
                                    "@tenon_rc_set_field"
                                };
                                self.code.push_str(&format!(
                                    "  call void {}(i64 {}, i64 {}, i64 {})\n",
                                    setter, uniq, fidx, new_fv
                                ));
                            }
                        }
                    }
                    ExprKind::Index {
                        target: list_expr,
                        index,
                    } => {
                        let mut idx_temps = Vec::new();
                        let idx_v = self.lower_expr_borrowed(index, &mut idx_temps)?;
                        self.flush_temps(&idx_temps);
                        match &list_expr.kind {
                            ExprKind::Ident(vname) => {
                                let arr_info = self
                                    .var_types
                                    .get(vname)
                                    .cloned()
                                    .and_then(|t| self.array_layout(&t));
                                if let Some((elem_ty, _len, stride, false)) = arr_info {
                                    let mut temps = Vec::new();
                                    let rhs = self.lower_expr_borrowed(value, &mut temps)?;
                                    let cur_arr = self.load_ident_val(vname)?;
                                    let uniq_arr = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n",
                                        uniq_arr, cur_arr
                                    ));
                                    let vslot = self.var_ptrs[vname].clone();
                                    if self.inout_vars.contains(vname) {
                                        let ai = self.fresh_tmp();
                                        let ap = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                            ai, vslot, ap, ai, uniq_arr, ap
                                        ));
                                    } else {
                                        self.code.push_str(&format!(
                                            "  store i64 {}, ptr {}\n",
                                            uniq_arr, vslot
                                        ));
                                    }
                                    let base = self.fresh_tmp();
                                    let boff = self.fresh_tmp();
                                    let eptr = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n  {} = mul i64 {}, {}\n  {} = add i64 {}, {}\n",
                                        base, uniq_arr, boff, idx_v, stride, eptr, base, boff
                                    ));
                                    let sk = scalar_kind_of_ty(&elem_ty, self.alias_bases);
                                    let is_flt = is_float_type_expr(&elem_ty, self.alias_bases) || self.is_float_expr(value);
                                    let new_val = if *op == AssignOp::Set {
                                        rhs
                                    } else {
                                        let ce = self.emit_mem_load(&eptr, 0, &elem_ty, false)?;
                                        self.emit_compound_op(&ce, &rhs, *op, is_flt, sk)
                                    };
                                    self.emit_mem_store(&eptr, 0, &new_val, &elem_ty, false)?;
                                    self.flush_temps(&temps);
                                } else if let Some((vec_ty, elem_bytes, _)) = self
                                    .var_types
                                    .get(vname)
                                    .and_then(|t| self.flat_simd_list_info(t))
                                {
                                    // Unboxed contiguous flat `List[SimdVec]` element store (zero heap boxing!)
                                    let mut temps = Vec::new();
                                    let vv = self.lower_simd_vec_reg(value, vec_ty, elem_bytes, &mut temps)?;
                                    let cur_list = self.load_ident_val(vname)?;
                                    let is_always_uniq = self.always_unique_vars.contains(vname);
                                    let uniq_list = if is_always_uniq {
                                        cur_list
                                    } else {
                                        let u = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n",
                                            u, cur_list
                                        ));
                                        let vslot = self.var_ptrs[vname].clone();
                                        if self.inout_vars.contains(vname) {
                                            let ai = self.fresh_tmp();
                                            let ap = self.fresh_tmp();
                                            self.code.push_str(&format!(
                                                "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                                ai, vslot, ap, ai, u, ap
                                            ));
                                        } else {
                                            self.code.push_str(&format!(
                                                "  store i64 {}, ptr {}\n",
                                                u, vslot
                                            ));
                                        }
                                        u
                                    };
                                    let base = self.fresh_tmp();
                                    let data = self.fresh_tmp();
                                    let ep = self.fresh_tmp();
                                    if self.proved_safe_indices.contains(&target.span.start) {
                                        self.code.push_str(&format!(
                                            "  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 40\n  {} = getelementptr inbounds {}, ptr {}, i64 {}\n  store {} {}, ptr {}, align 8, !tbaa !7\n",
                                            base, uniq_list, data, base, ep, vec_ty, data, idx_v, vec_ty, vv, ep
                                        ));
                                    } else {
                                        let ok_l = self.fresh_lbl("fset_ok");
                                        let end_l = self.fresh_lbl("fset_end");
                                        let lp = self.fresh_tmp();
                                        let len_v = self.fresh_tmp();
                                        let in_b = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 16\n  {} = load i64, ptr {}, align 8, !tbaa !8\n  {} = icmp ult i64 {}, {}\n  br i1 {}, label %{}, label %{}\n{}:\n  {} = getelementptr inbounds i8, ptr {}, i64 40\n  {} = getelementptr inbounds {}, ptr {}, i64 {}\n  store {} {}, ptr {}, align 8, !tbaa !7\n  br label %{}\n{}:\n",
                                            base, uniq_list, lp, base, len_v, lp, in_b, idx_v, len_v, in_b, ok_l, end_l,
                                            ok_l, data, base, ep, vec_ty, data, idx_v, vec_ty, vv, ep, end_l, end_l
                                        ));
                                    }
                                    self.flush_temps(&temps);
                                } else if let Some(is_signed) = self
                                    .var_types
                                    .get(vname)
                                    .and_then(|t| self.flat_scalar2_list_info(t))
                                {
                                    let elem_ty = self
                                        .var_types
                                        .get(vname)
                                        .and_then(|t| self.list_elem_ty(t));
                                    let mut temps = Vec::new();
                                    let rhs = self.lower_expr_borrowed(value, &mut temps)?;
                                    let cur_list = self.load_ident_val(vname)?;
                                    let is_always_uniq = self.always_unique_vars.contains(vname);
                                    let uniq_list = if is_always_uniq {
                                        cur_list
                                    } else {
                                        let u = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n",
                                            u, cur_list
                                        ));
                                        let vslot = self.var_ptrs[vname].clone();
                                        if self.inout_vars.contains(vname) {
                                            let ai = self.fresh_tmp();
                                            let ap = self.fresh_tmp();
                                            self.code.push_str(&format!(
                                                "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                                ai, vslot, ap, ai, u, ap
                                            ));
                                        } else {
                                            self.code.push_str(&format!(
                                                "  store i64 {}, ptr {}\n",
                                                u, vslot
                                            ));
                                        }
                                        u
                                    };
                                    let is_safe = self.proved_safe_indices.contains(&target.span.start);
                                    let final_v = if *op == AssignOp::Set {
                                        rhs
                                    } else {
                                        let cur = self.emit_flat_scalar2_load(&uniq_list, &idx_v, is_signed, is_safe);
                                        let sk = elem_ty.as_ref().and_then(|t| scalar_kind_of_ty(t, self.alias_bases));
                                        let is_flt = elem_ty.as_ref().is_some_and(|t| is_float_type_expr(t, self.alias_bases));
                                        self.emit_compound_op(&cur, &rhs, *op, is_flt, sk)
                                    };
                                    self.emit_flat_scalar2_store(&uniq_list, &idx_v, &final_v, is_safe);
                                    self.flush_temps(&temps);
                                } else if let Some(is_signed) = self
                                    .var_types
                                    .get(vname)
                                    .and_then(|t| self.flat_scalar4_list_info(t))
                                {
                                    let elem_ty = self
                                        .var_types
                                        .get(vname)
                                        .and_then(|t| self.list_elem_ty(t));
                                    let mut temps = Vec::new();
                                    let rhs = self.lower_expr_borrowed(value, &mut temps)?;
                                    let cur_list = self.load_ident_val(vname)?;
                                    let is_always_uniq = self.always_unique_vars.contains(vname);
                                    let uniq_list = if is_always_uniq {
                                        cur_list
                                    } else {
                                        let u = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n",
                                            u, cur_list
                                        ));
                                        let vslot = self.var_ptrs[vname].clone();
                                        if self.inout_vars.contains(vname) {
                                            let ai = self.fresh_tmp();
                                            let ap = self.fresh_tmp();
                                            self.code.push_str(&format!(
                                                "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                                ai, vslot, ap, ai, u, ap
                                            ));
                                        } else {
                                            self.code.push_str(&format!(
                                                "  store i64 {}, ptr {}\n",
                                                u, vslot
                                            ));
                                        }
                                        u
                                    };
                                    let is_safe = self.proved_safe_indices.contains(&target.span.start);
                                    let final_v = if *op == AssignOp::Set {
                                        rhs
                                    } else {
                                        let cur = self.emit_flat_scalar4_load(&uniq_list, &idx_v, is_signed, is_safe);
                                        let sk = elem_ty.as_ref().and_then(|t| scalar_kind_of_ty(t, self.alias_bases));
                                        let is_flt = elem_ty.as_ref().is_some_and(|t| is_float_type_expr(t, self.alias_bases));
                                        self.emit_compound_op(&cur, &rhs, *op, is_flt, sk)
                                    };
                                    self.emit_flat_scalar4_store(&uniq_list, &idx_v, &final_v, is_safe);
                                    self.flush_temps(&temps);
                                } else {
                                    let elem_ty = self
                                        .var_types
                                        .get(vname)
                                        .and_then(|t| self.list_elem_ty(t));
                                    let elem_is_rc = elem_ty
                                        .as_ref()
                                        .is_some_and(|et| self.type_is_rc(et))
                                        || self.expr_produces_rc(value);
                                    let rhs = if elem_is_rc && *op == AssignOp::Set {
                                        self.lower_expr_owned(value, elem_ty.as_ref())?
                                    } else {
                                        let mut temps = Vec::new();
                                        let v = self.lower_expr_borrowed(value, &mut temps)?;
                                        self.flush_temps(&temps);
                                        v
                                    };
                                    let cur_list = self.load_ident_val(vname)?;
                                    let is_always_uniq = self.always_unique_vars.contains(vname);
                                    let uniq_list = if is_always_uniq {
                                        cur_list
                                    } else {
                                        let u = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n",
                                            u, cur_list
                                        ));
                                        let vslot = self.var_ptrs[vname].clone();
                                        if self.inout_vars.contains(vname) {
                                            let ai = self.fresh_tmp();
                                            let ap = self.fresh_tmp();
                                            self.code.push_str(&format!(
                                                "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                                ai, vslot, ap, ai, u, ap
                                            ));
                                        } else {
                                            self.code.push_str(&format!(
                                                "  store i64 {}, ptr {}\n",
                                                u, vslot
                                            ));
                                        }
                                        u
                                    };
                                    if elem_is_rc {
                                        self.code.push_str(&format!(
                                            "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                                            uniq_list, RC_MASK_ALL as i64
                                        ));
                                    }
                                    let is_proved_safe =
                                        self.proved_safe_indices.contains(&target.span.start);
                                    let getter = if is_proved_safe {
                                        "@__tn_list_get_unchecked"
                                    } else if is_always_uniq {
                                        "@__tn_list_get_nonnull"
                                    } else {
                                        "@__tn_list_get"
                                    };
                                    let sk = elem_ty.as_ref().and_then(|t| scalar_kind_of_ty(t, self.alias_bases));
                                    let is_flt = elem_ty
                                        .as_ref()
                                        .is_some_and(|t| is_float_type_expr(t, self.alias_bases))
                                        || self.is_float_expr(value);
                                    let new_val = if *op == AssignOp::Set {
                                        rhs
                                    } else {
                                        let ce = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = call i64 {}(i64 {}, i64 {})\n",
                                            ce, getter, uniq_list, idx_v
                                        ));
                                        self.emit_compound_op(&ce, &rhs, *op, is_flt, sk)
                                    };
                                    let setter = if *op != AssignOp::Set {
                                        if is_proved_safe {
                                            "@__tn_list_set_prim_unchecked"
                                        } else if is_always_uniq {
                                            "@__tn_list_set_prim_nonnull"
                                        } else {
                                            "@__tn_list_set_prim"
                                        }
                                    } else if elem_is_rc {
                                        "@__tn_list_set_rc"
                                    } else if elem_ty
                                        .as_ref()
                                        .is_some_and(|et| et.display_name() != "Any")
                                    {
                                        if is_proved_safe {
                                            "@__tn_list_set_prim_unchecked"
                                        } else if is_always_uniq {
                                            "@__tn_list_set_prim_nonnull"
                                        } else {
                                            "@__tn_list_set_prim"
                                        }
                                    } else {
                                        "@tenon_rc_set_field"
                                    };
                                    self.code.push_str(&format!(
                                        "  call void {}(i64 {}, i64 {}, i64 {})\n",
                                        setter, uniq_list, idx_v, new_val
                                    ));
                                }
                            }
                            ExprKind::Field { target: obj, field }
                                if matches!(&obj.kind, ExprKind::Ident(_)) =>
                            {
                                let ExprKind::Ident(vname) = &obj.kind else {
                                    unreachable!()
                                };
                                let (fidx, fl) = self.resolve_field_layout(obj, field)?;
                                let elem_ty = fl.as_ref().and_then(|f| self.list_elem_ty(&f.ty));
                                let elem_is_rc = elem_ty
                                    .as_ref()
                                    .is_some_and(|et| self.type_is_rc(et))
                                    || self.expr_produces_rc(value);
                                let rhs = if elem_is_rc && *op == AssignOp::Set {
                                    self.lower_expr_owned(value, elem_ty.as_ref())?
                                } else {
                                    let mut temps = Vec::new();
                                    let v = self.lower_expr_borrowed(value, &mut temps)?;
                                    self.flush_temps(&temps);
                                    v
                                };
                                let cur_obj = self.load_ident_val(vname)?;
                                let uniq_obj = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n",
                                    uniq_obj, cur_obj
                                ));
                                let vslot = self.var_ptrs[vname].clone();
                                if self.inout_vars.contains(vname) {
                                    let ai = self.fresh_tmp();
                                    let ap = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                        ai, vslot, ap, ai, uniq_obj, ap
                                    ));
                                } else {
                                    self.code.push_str(&format!(
                                        "  store i64 {}, ptr {}\n",
                                        uniq_obj, vslot
                                    ));
                                }
                                let cur_list = self.fresh_tmp();
                                let uniq_list = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n  call void @__tn_slot_set_prim(i64 {}, i64 {}, i64 {})\n",
                                    cur_list, uniq_obj, fidx, uniq_list, cur_list, uniq_obj, fidx, uniq_list
                                ));
                                if elem_is_rc {
                                    self.code.push_str(&format!(
                                        "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                                        uniq_list, RC_MASK_ALL as i64
                                    ));
                                }
                                let sk = elem_ty.as_ref().and_then(|t| scalar_kind_of_ty(t, self.alias_bases));
                                let is_flt = elem_ty
                                    .as_ref()
                                    .is_some_and(|t| is_float_type_expr(t, self.alias_bases))
                                    || self.is_float_expr(value);
                                let new_val = if *op == AssignOp::Set {
                                    rhs
                                } else {
                                    let ce = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = call i64 @__tn_list_get(i64 {}, i64 {})\n",
                                        ce, uniq_list, idx_v
                                    ));
                                    self.emit_compound_op(&ce, &rhs, *op, is_flt, sk)
                                };
                                let setter = if *op != AssignOp::Set {
                                    "@__tn_list_set_prim"
                                } else if elem_is_rc {
                                    "@__tn_list_set_rc"
                                } else if elem_ty
                                    .as_ref()
                                    .is_some_and(|et| et.display_name() != "Any")
                                {
                                    "@__tn_list_set_prim"
                                } else {
                                    "@tenon_rc_set_field"
                                };
                                self.code.push_str(&format!(
                                    "  call void {}(i64 {}, i64 {}, i64 {})\n",
                                    setter, uniq_list, idx_v, new_val
                                ));
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                },
                Stmt::Mutate { body, .. } => {
                    let _ = self.lower_block(body, false, None)?;
                }
                Stmt::Return { value, .. } => {
                    let mut move_skip_slot: Option<String> = None;
                    let ret_ty_opt = self.fn_ret_ty.clone();
                    let v = match value {
                        Some(e) => {
                            if self.fn_returns_rc {
                                if let ExprKind::Ident(rname) = &e.kind {
                                    if !self.inout_vars.contains(rname) {
                                        if let Some(slot) = self.var_ptrs.get(rname).cloned() {
                                            if self.find_owned_slot_in_scopes(&slot, 0) {
                                                move_skip_slot = Some(slot.clone());
                                                let lv = self.fresh_tmp();
                                                self.code.push_str(&format!(
                                                    "  {} = load i64, ptr {}\n",
                                                    lv, slot
                                                ));
                                                lv
                                            } else {
                                                self.lower_expr_owned(e, ret_ty_opt.as_ref())?
                                            }
                                        } else {
                                            self.lower_expr_owned(e, ret_ty_opt.as_ref())?
                                        }
                                    } else {
                                        self.lower_expr_owned(e, ret_ty_opt.as_ref())?
                                    }
                                } else {
                                    self.lower_expr_owned(e, ret_ty_opt.as_ref())?
                                }
                            } else {
                                let mut temps = Vec::new();
                                let val = self.lower_expr_borrowed(e, &mut temps)?;
                                self.flush_temps(&temps);
                                val
                            }
                        }
                        None => "0".into(),
                    };
                    self.emit_cleanup_temp_scopes_from(0);
                    self.emit_cleanup_scopes_from(0, move_skip_slot.as_deref());
                    if let Some(am) = &self.active_arena_mark {
                        self.code
                            .push_str(&format!("  call void @tenon_arena_exit(i64 {})\n", am));
                    }
                    self.code.push_str(&format!("  ret i64 {}\n", v));
                    self.terminated = true;
                    last = v;
                    break;
                }
                Stmt::Break { .. } => {
                    let (exit_lbl, rc_depth, temp_depth) = self
                        .loop_stack
                        .last()
                        .map(|l| (l.exit_lbl.clone(), l.rc_scope_depth, l.temp_scope_depth))
                        .ok_or_else(|| "`break` used outside of `while` loop".to_string())?;
                    self.emit_cleanup_temp_scopes_from(temp_depth);
                    self.emit_cleanup_scopes_from(rc_depth, None);
                    self.code.push_str(&format!("  br label %{}\n", exit_lbl));
                    self.terminated = true;
                    break;
                }
                Stmt::Continue { .. } => {
                    let (hdr_lbl, rc_depth, temp_depth) = self
                        .loop_stack
                        .last()
                        .map(|l| (l.hdr_lbl.clone(), l.rc_scope_depth, l.temp_scope_depth))
                        .ok_or_else(|| "`continue` used outside of `while` loop".to_string())?;
                    self.emit_cleanup_temp_scopes_from(temp_depth);
                    self.emit_cleanup_scopes_from(rc_depth, None);
                    self.code.push_str(&format!("  br label %{}\n", hdr_lbl));
                    self.terminated = true;
                    break;
                }
                Stmt::Assert { cond, span } => {
                    let mut temps = Vec::new();
                    let c = self.lower_expr_borrowed(cond, &mut temps)?;
                    self.flush_temps(&temps);
                    let ok = self.fresh_tmp();
                    let fl = self.fresh_lbl("afail");
                    let cl = self.fresh_lbl("acont");
                    self.code.push_str(&format!(
                        "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n  call void @tenon_rt_assert_fail(i64 {})\n  br label %{}\n{}:\n",
                        ok, c, ok, cl, fl, fl, span.line, cl, cl
                    ));
                }
                Stmt::While { cond, body, .. } => {
                    let hdr = self.fresh_lbl("whdr");
                    let lp = self.fresh_lbl("wbody");
                    let ex = self.fresh_lbl("wexit");
                    self.code.push_str(&format!("  br label %{}\n{}:\n", hdr, hdr));
                    let mut ctemps = Vec::new();
                    let c = self.lower_expr_borrowed(cond, &mut ctemps)?;
                    self.flush_temps(&ctemps);
                    let cmp = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n",
                        cmp, c, cmp, lp, ex, lp
                    ));
                    self.loop_stack.push(LlvmLoopCtx {
                        hdr_lbl: hdr.clone(),
                        exit_lbl: ex.clone(),
                        rc_scope_depth: self.rc_scopes.len(),
                        temp_scope_depth: self.temp_rc_scopes.len(),
                    });
                    let _ = self.lower_block(body, false, None)?;
                    self.loop_stack.pop();
                    if !self.terminated {
                        if self.cur_fn_preemptible {
                            let pf = self.fresh_tmp();
                            let need = self.fresh_tmp();
                            let pslow = self.fresh_lbl("preempt_slow");
                            self.code.push_str(&format!(
                                "  {} = load atomic i32, ptr @tenon_preempt_flag monotonic, align 4\n  {} = icmp ne i32 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n  call void @tenon_sched_preempt_point()\n  br label %{}\n",
                                pf, need, pf, need, pslow, hdr, pslow, hdr
                            ));
                        } else {
                            self.code.push_str(&format!("  br label %{}\n", hdr));
                        }
                    }
                    self.terminated = false;
                    self.code.push_str(&format!("{}:\n", ex));
                }
                Stmt::Expr { expr, .. } => {
                    let mut temps = Vec::new();
                    last = self.lower_expr_borrowed(expr, &mut temps)?;
                    if !self.terminated {
                        self.flush_temps(&temps);
                    }
                }
                Stmt::LetTuple { .. } | Stmt::For { .. } | Stmt::Defer { .. } | Stmt::Ghost { .. } => {}
            }
        }

        let mut move_skip_slot: Option<String> = None;
        if !self.terminated {
            if let Some(tail) = &blk.tail {
                if want_owned_tail {
                    if let ExprKind::Ident(tname) = &tail.kind {
                        if !self.inout_vars.contains(tname) {
                            if let Some(slot) = self.var_ptrs.get(tname).cloned() {
                                if self.find_owned_slot_in_scopes(&slot, scope_idx) {
                                    move_skip_slot = Some(slot.clone());
                                    let lv = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = load i64, ptr {}\n",
                                        lv, slot
                                    ));
                                    last = lv;
                                } else {
                                    last = self.lower_expr_owned(tail, expected_tail_ty)?;
                                }
                            } else {
                                last = self.lower_expr_owned(tail, expected_tail_ty)?;
                            }
                        } else {
                            last = self.lower_expr_owned(tail, expected_tail_ty)?;
                        }
                    } else {
                        last = self.lower_expr_owned(tail, expected_tail_ty)?;
                    }
                } else {
                    let mut temps = Vec::new();
                    last = self.lower_expr_borrowed(tail, &mut temps)?;
                    if !self.terminated {
                        self.flush_temps(&temps);
                    }
                }
            }
            if !self.terminated {
                self.emit_cleanup_scopes_from(scope_idx, move_skip_slot.as_deref());
            }
        }

        self.rc_scopes.pop();
        self.restore_lexical(saved);
        Ok(last)
    }

    fn lower_expr_owned(
        &mut self,
        expr: &Expr,
        expected_ty: Option<&TypeExpr>,
    ) -> Result<String, String> {
        match &expr.kind {
            ExprKind::Ident(s) => {
                if s == "self" {
                    if let Some(sv) = self.self_var.clone() {
                        let slot = self.var_ptrs[&sv].clone();
                        let v = self.fresh_tmp();
                        self.code
                            .push_str(&format!("  {} = load i64, ptr {}\n", v, slot));
                        return Ok(v);
                    }
                }
                if s == "None" {
                    return Ok(self.emit_unit_enum_const(TAG_NONE));
                }
                if let Some(&tag) = self.enum_tags.get(s) {
                    if !self.var_ptrs.contains_key(s) {
                        return Ok(self.emit_unit_enum_const(tag));
                    }
                }
                let v = self.load_ident_val(s)?;
                if self.is_rc_ident(s) {
                    if self.owned_rc_vars.contains(s)
                        && !self.inout_vars.contains(s)
                        && expr.span.start > 0
                        && self.cur_last_use_move_spans.contains(&expr.span.start)
                    {
                        if let Some(slot) = self.var_ptrs.get(s) {
                            self.code.push_str(&format!("  store i64 0, ptr {}\n", slot));
                        }
                        return Ok(v);
                    }
                    let inc = self.fresh_tmp();
                    self.code
                        .push_str(&format!("  {} = call i64 @__tn_rc_inc(i64 {})\n", inc, v));
                    Ok(inc)
                } else {
                    Ok(v)
                }
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    let key = format!("{}.{}", ename, field);
                    if let Some(&tag) = self.enum_tags.get(&key) {
                        return Ok(self.emit_unit_enum_const(tag));
                    }
                }
                let c_field_info = match resolve_alias_ty(&self.infer_expr_type(target), self.alias_bases) {
                    TypeExpr::Named { name: rname, .. } => self
                        .records
                        .get(&rname)
                        .and_then(|rl| rl.c_layout.as_ref())
                        .and_then(|cl| cl.fields.iter().find(|f| f.name == *field))
                        .cloned(),
                    _ => None,
                };
                if let Some(cfl) = c_field_info {
                    let mut temps = Vec::new();
                    let obj = self.lower_expr_borrowed(target, &mut temps)?;
                    let base = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n",
                        base, obj
                    ));
                    let val = self.emit_mem_load(&base, cfl.byte_offset as i64, &cfl.ty, false)?;
                    self.flush_temps(&temps);
                    return Ok(val);
                }
                let (idx, fl) = self.resolve_field_layout(target, field)?;
                let mut temps = Vec::new();
                let obj = self.lower_expr_borrowed(target, &mut temps)?;
                let mut res = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n",
                    res, obj, idx
                ));
                if fl.as_ref().is_some_and(|f| f.is_rc) {
                    let inc = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @__tn_rc_inc(i64 {})\n",
                        inc, res
                    ));
                    res = inc;
                }
                self.flush_temps(&temps);
                Ok(res)
            }
            ExprKind::Index { target, index } => {
                let target_ty = self.infer_expr_type(target);
                if let Some((elem_ty, _len, stride, false)) = self.array_layout(&target_ty) {
                    let mut temps = Vec::new();
                    let t = self.lower_expr_borrowed(target, &mut temps)?;
                    let i = self.lower_expr_borrowed(index, &mut temps)?;
                    let base = self.fresh_tmp();
                    let boff = self.fresh_tmp();
                    let eptr = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n  {} = mul i64 {}, {}\n  {} = add i64 {}, {}\n",
                        base, t, boff, i, stride, eptr, base, boff
                    ));
                    let res = self.emit_mem_load(&eptr, 0, &elem_ty, false)?;
                    self.flush_temps(&temps);
                    return Ok(res);
                }
                if let Some((vec_ty, elem_bytes, _)) = self.flat_simd_list_info(&target_ty) {
                    let mut temps = Vec::new();
                    let vv = self.lower_simd_vec_reg(expr, vec_ty, elem_bytes, &mut temps)?;
                    let obj = self.fresh_tmp();
                    let di = self.fresh_tmp();
                    let dp = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @tenon_cstruct_alloc(i64 {})\n  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n  {} = inttoptr i64 {} to ptr\n  store {} {}, ptr {}, align 8\n",
                        obj, elem_bytes, di, obj, dp, di, vec_ty, vv, dp
                    ));
                    self.flush_temps(&temps);
                    return Ok(obj);
                }
                if let Some(is_signed) = self.flat_scalar2_list_info(&target_ty) {
                    let mut temps = Vec::new();
                    let t = self.lower_expr_borrowed(target, &mut temps)?;
                    let i = self.lower_expr_borrowed(index, &mut temps)?;
                    let is_safe = self.proved_safe_indices.contains(&expr.span.start);
                    let res = self.emit_flat_scalar2_load(&t, &i, is_signed, is_safe);
                    self.flush_temps(&temps);
                    return Ok(res);
                }
                if let Some(is_signed) = self.flat_scalar4_list_info(&target_ty) {
                    let mut temps = Vec::new();
                    let t = self.lower_expr_borrowed(target, &mut temps)?;
                    let i = self.lower_expr_borrowed(index, &mut temps)?;
                    let is_safe = self.proved_safe_indices.contains(&expr.span.start);
                    let res = self.emit_flat_scalar4_load(&t, &i, is_signed, is_safe);
                    self.flush_temps(&temps);
                    return Ok(res);
                }
                let elem_is_rc = self
                    .list_elem_ty(&target_ty)
                    .is_some_and(|et| self.type_is_rc(&et));
                let mut temps = Vec::new();
                let t = self.lower_expr_borrowed(target, &mut temps)?;
                let i = self.lower_expr_borrowed(index, &mut temps)?;
                let mut res = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @__tn_list_get(i64 {}, i64 {})\n",
                    res, t, i
                ));
                if elem_is_rc {
                    let inc = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @__tn_rc_inc(i64 {})\n",
                        inc, res
                    ));
                    res = inc;
                }
                self.flush_temps(&temps);
                Ok(res)
            }
            ExprKind::TupleIndex { target, index } => {
                let elem_is_rc = match resolve_alias_ty(&self.infer_expr_type(target), self.alias_bases) {
                    TypeExpr::Tuple(es, _) => es.get(*index).is_some_and(|et| self.type_is_rc(et)),
                    _ => expected_ty.is_some_and(|t| self.type_is_rc(t)),
                };
                let mut temps = Vec::new();
                let t = self.lower_expr_borrowed(target, &mut temps)?;
                let mut res = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n",
                    res, t, index
                ));
                if elem_is_rc {
                    let inc = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @__tn_rc_inc(i64 {})\n",
                        inc, res
                    ));
                    res = inc;
                }
                self.flush_temps(&temps);
                Ok(res)
            }
            ExprKind::Call { callee, args }
                if matches!(&callee.kind, ExprKind::Ident(f) if f == "__closure_get") && args.len() == 2 =>
            {
                let mut temps = Vec::new();
                let env = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                let idx = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                let mut res = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n",
                    res, env, idx
                ));
                if expected_ty.is_some_and(|t| self.type_is_rc(t)) {
                    let inc = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @__tn_rc_inc(i64 {})\n",
                        inc, res
                    ));
                    res = inc;
                }
                self.flush_temps(&temps);
                Ok(res)
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => self.lower_if(cond, then_block, else_branch.as_deref(), true, expected_ty),
            ExprKind::Match { scrutinee, arms } => {
                self.lower_match(scrutinee, arms, true, expected_ty)
            }
            ExprKind::Block(blk) | ExprKind::Unsafe(blk) => self.lower_block(blk, true, expected_ty),
            _ => self.lower_expr_fresh(expr, expected_ty),
        }
    }

    fn lower_expr_borrowed(
        &mut self,
        expr: &Expr,
        temps: &mut Vec<String>,
    ) -> Result<String, String> {
        match &expr.kind {
            ExprKind::Int(n) => Ok(n.to_string()),
            ExprKind::Float(x) => Ok((x.to_bits() as i64).to_string()),
            ExprKind::Bool(b) => Ok(if *b { "1" } else { "0" }.into()),
            ExprKind::Str(s) => {
                let gname = self.intern_string_global(s);
                let obj = self.fresh_tmp();
                self.code
                    .push_str(&format!("  {} = ptrtoint ptr @{} to i64\n", obj, gname));
                Ok(obj)
            }
            ExprKind::Ident(s) => {
                if s == "self" {
                    if let Some(sv) = self.self_var.clone() {
                        let slot = self.var_ptrs[&sv].clone();
                        let v = self.fresh_tmp();
                        self.code
                            .push_str(&format!("  {} = load i64, ptr {}\n", v, slot));
                        return Ok(v);
                    }
                }
                if s == "None" {
                    return Ok(self.emit_unit_enum_const(TAG_NONE));
                }
                if let Some(&tag) = self.enum_tags.get(s) {
                    if !self.var_ptrs.contains_key(s) {
                        return Ok(self.emit_unit_enum_const(tag));
                    }
                }
                self.load_ident_val(s)
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    let key = format!("{}.{}", ename, field);
                    if let Some(&tag) = self.enum_tags.get(&key) {
                        return Ok(self.emit_unit_enum_const(tag));
                    }
                }
                let c_field_info = match resolve_alias_ty(&self.infer_expr_type(target), self.alias_bases) {
                    TypeExpr::Named { name: rname, .. } => self
                        .records
                        .get(&rname)
                        .and_then(|rl| rl.c_layout.as_ref())
                        .and_then(|cl| cl.fields.iter().find(|f| f.name == *field))
                        .cloned(),
                    _ => None,
                };
                if let Some(cfl) = c_field_info {
                    let obj = self.lower_expr_borrowed(target, temps)?;
                    let base = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n",
                        base, obj
                    ));
                    let val = self.emit_mem_load(&base, cfl.byte_offset as i64, &cfl.ty, false)?;
                    if self.is_c_value_type(&cfl.ty) {
                        temps.push(val.clone());
                    }
                    return Ok(val);
                }
                let (idx, _) = self.resolve_field_layout(target, field)?;
                let obj = self.lower_expr_borrowed(target, temps)?;
                let res = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n",
                    res, obj, idx
                ));
                Ok(res)
            }
            ExprKind::Index { target, index } => {
                let target_ty = self.infer_expr_type(target);
                if let Some((elem_ty, _len, stride, false)) = self.array_layout(&target_ty) {
                    let t = self.lower_expr_borrowed(target, temps)?;
                    let i = self.lower_expr_borrowed(index, temps)?;
                    let base = self.fresh_tmp();
                    let boff = self.fresh_tmp();
                    let eptr = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n  {} = mul i64 {}, {}\n  {} = add i64 {}, {}\n",
                        base, t, boff, i, stride, eptr, base, boff
                    ));
                    let val = self.emit_mem_load(&eptr, 0, &elem_ty, false)?;
                    if self.is_c_value_type(&elem_ty) {
                        temps.push(val.clone());
                    }
                    return Ok(val);
                }
                if let Some((vec_ty, elem_bytes, _)) = self.flat_simd_list_info(&target_ty) {
                    let vv = self.lower_simd_vec_reg(expr, vec_ty, elem_bytes, temps)?;
                    let obj = self.fresh_tmp();
                    let di = self.fresh_tmp();
                    let dp = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @tenon_cstruct_alloc(i64 {})\n  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n  {} = inttoptr i64 {} to ptr\n  store {} {}, ptr {}, align 8\n",
                        obj, elem_bytes, di, obj, dp, di, vec_ty, vv, dp
                    ));
                    temps.push(obj.clone());
                    return Ok(obj);
                }
                if let Some(is_signed) = self.flat_scalar2_list_info(&target_ty) {
                    let t = self.lower_expr_borrowed(target, temps)?;
                    let i = self.lower_expr_borrowed(index, temps)?;
                    let is_safe = self.proved_safe_indices.contains(&expr.span.start);
                    return Ok(self.emit_flat_scalar2_load(&t, &i, is_signed, is_safe));
                }
                if let Some(is_signed) = self.flat_scalar4_list_info(&target_ty) {
                    let t = self.lower_expr_borrowed(target, temps)?;
                    let i = self.lower_expr_borrowed(index, temps)?;
                    let is_safe = self.proved_safe_indices.contains(&expr.span.start);
                    return Ok(self.emit_flat_scalar4_load(&t, &i, is_signed, is_safe));
                }
                let is_proved_safe = self.proved_safe_indices.contains(&expr.span.start);
                let is_nonnull = matches!(&target.kind, ExprKind::Ident(n) if self.always_unique_vars.contains(n));
                let t = self.lower_expr_borrowed(target, temps)?;
                let i = self.lower_expr_borrowed(index, temps)?;
                let res = self.fresh_tmp();
                let getter = if is_proved_safe {
                    "@__tn_list_get_unchecked"
                } else if is_nonnull {
                    "@__tn_list_get_nonnull"
                } else {
                    "@__tn_list_get"
                };
                self.code.push_str(&format!(
                    "  {} = call i64 {}(i64 {}, i64 {})\n",
                    res, getter, t, i
                ));
                Ok(res)
            }
            ExprKind::TupleIndex { target, index } => {
                let t = self.lower_expr_borrowed(target, temps)?;
                let res = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n",
                    res, t, index
                ));
                Ok(res)
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                let want_owned = self.expr_produces_rc(expr);
                let v = self.lower_if(cond, then_block, else_branch.as_deref(), want_owned, None)?;
                if want_owned && !self.terminated {
                    temps.push(v.clone());
                }
                Ok(v)
            }
            ExprKind::Match { scrutinee, arms } => {
                let want_owned = self.expr_produces_rc(expr);
                let v = self.lower_match(scrutinee, arms, want_owned, None)?;
                if want_owned && !self.terminated {
                    temps.push(v.clone());
                }
                Ok(v)
            }
            ExprKind::Block(blk) | ExprKind::Unsafe(blk) => {
                let want_owned = self.expr_produces_rc(expr);
                let v = self.lower_block(blk, want_owned, None)?;
                if want_owned && !self.terminated {
                    temps.push(v.clone());
                }
                Ok(v)
            }
            ExprKind::Call { callee, args }
                if args.len() == 1
                    && matches!(
                        &callee.kind,
                        ExprKind::Ident(f) if f == "text_cstr_ptr" || f == "bytes_ptr" || f == "list_data_ptr"
                    ) =>
            {
                let ExprKind::Ident(fname) = &callee.kind else {
                    unreachable!()
                };
                // Keep the backing Text/Bytes/List temporary alive in caller's `temps`
                // until the enclosing call completes (prevents UAF on dynamic buffers)!
                let arg_v = self.lower_expr_borrowed(&args[0].value, temps)?;
                let csym = if fname == "text_cstr_ptr" {
                    "tenon_text_cstr_ptr"
                } else if fname == "list_data_ptr" {
                    "tenon_list_data_ptr"
                } else {
                    "tenon_bytes_ptr"
                };
                let res = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @{}(i64 {})\n",
                    res, csym, arg_v
                ));
                Ok(res)
            }
            _ => {
                let v = self.lower_expr_fresh(expr, None)?;
                if self.expr_produces_rc(expr) {
                    temps.push(v.clone());
                }
                Ok(v)
            }
        }
    }

    fn lower_if(
        &mut self,
        cond: &Expr,
        then_block: &Block,
        else_branch: Option<&Expr>,
        want_owned: bool,
        expected_ty: Option<&TypeExpr>,
    ) -> Result<String, String> {
        let mut ctemps = Vec::new();
        let c = self.lower_expr_borrowed(cond, &mut ctemps)?;
        self.flush_temps(&ctemps);

        let cmp = self.fresh_tmp();
        self.slot_id += 1;
        let res_slot = format!("%ifres_{}", self.slot_id);
        self.entry_allocas.push_str(&format!(
            "  {} = alloca i64\n  store i64 0, ptr {}\n",
            res_slot, res_slot
        ));
        let tl = self.fresh_lbl("then");
        let el = self.fresh_lbl("else");
        let ml = self.fresh_lbl("endif");
        self.code.push_str(&format!(
            "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n",
            cmp, c, cmp, tl, el, tl
        ));
        let tv = self.lower_block(then_block, want_owned, expected_ty)?;
        let then_term = self.terminated;
        if !then_term {
            self.code.push_str(&format!(
                "  store i64 {}, ptr {}\n  br label %{}\n",
                tv, res_slot, ml
            ));
        }
        self.terminated = false;
        self.code.push_str(&format!("{}:\n", el));
        let ev = match else_branch {
            Some(eb) => {
                if want_owned {
                    self.lower_expr_owned(eb, expected_ty)?
                } else {
                    let mut etemps = Vec::new();
                    let v = self.lower_expr_borrowed(eb, &mut etemps)?;
                    if !self.terminated {
                        self.flush_temps(&etemps);
                    }
                    v
                }
            }
            None => "0".into(),
        };
        let else_term = self.terminated;
        if !else_term {
            self.code.push_str(&format!(
                "  store i64 {}, ptr {}\n  br label %{}\n",
                ev, res_slot, ml
            ));
        }
        self.code.push_str(&format!("{}:\n", ml));
        if then_term && else_term {
            self.code.push_str("  unreachable\n");
            self.terminated = true;
            return Ok("0".into());
        }
        self.terminated = false;
        let out = self.fresh_tmp();
        self.code
            .push_str(&format!("  {} = load i64, ptr {}\n", out, res_slot));
        Ok(out)
    }

    fn emit_pattern_check(
        &mut self,
        val: &str,
        ty: &TypeExpr,
        pat: &Pattern,
        fail_lbl: &str,
        bindings: &mut Vec<(String, String, TypeExpr)>,
    ) -> Result<(), String> {
        let base = resolve_alias_ty(ty, self.alias_bases);
        match pat {
            Pattern::Wildcard(_) => Ok(()),
            Pattern::Ident(name, _) => {
                if name != "_" {
                    bindings.push((name.clone(), val.to_string(), ty.clone()));
                }
                Ok(())
            }
            Pattern::Int(n, _) => {
                let eq = self.fresh_tmp();
                let ok_l = self.fresh_lbl("pok");
                self.code.push_str(&format!(
                    "  {} = icmp eq i64 {}, {}\n  br i1 {}, label %{}, label %{}\n{}:\n",
                    eq, val, n, eq, ok_l, fail_lbl, ok_l
                ));
                Ok(())
            }
            Pattern::Bool(b, _) => {
                let eq = self.fresh_tmp();
                let ok_l = self.fresh_lbl("pok");
                self.code.push_str(&format!(
                    "  {} = icmp eq i64 {}, {}\n  br i1 {}, label %{}, label %{}\n{}:\n",
                    eq,
                    val,
                    if *b { 1 } else { 0 },
                    eq,
                    ok_l,
                    fail_lbl,
                    ok_l
                ));
                Ok(())
            }
            Pattern::Str(s, _) => {
                let gname = self.intern_string_global(s);
                let ptxt = self.fresh_tmp();
                let teq = self.fresh_tmp();
                let eq = self.fresh_tmp();
                let ok_l = self.fresh_lbl("pok");
                self.code.push_str(&format!(
                    "  {} = ptrtoint ptr @{} to i64\n  {} = call i64 @__tn_text_eq(i64 {}, i64 {})\n  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n",
                    ptxt, gname, teq, val, ptxt, eq, teq, eq, ok_l, fail_lbl, ok_l
                ));
                Ok(())
            }
            Pattern::Tuple(subpats, _) => {
                let elems = match &base {
                    TypeExpr::Tuple(es, _) => es.clone(),
                    _ => Vec::new(),
                };
                for (idx, sp) in subpats.iter().enumerate() {
                    if matches!(sp, Pattern::Wildcard(_)) {
                        continue;
                    }
                    let ev = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n",
                        ev, val, idx
                    ));
                    let ety = elems.get(idx).cloned().unwrap_or_else(|| named_ty("Any"));
                    self.emit_pattern_check(&ev, &ety, sp, fail_lbl, bindings)?;
                }
                Ok(())
            }
            Pattern::Variant { name, binding, .. } => {
                let t = *self
                    .enum_tags
                    .get(name)
                    .ok_or_else(|| format!("unknown variant `{}` in LLVM match", name))?;
                let tg = self.fresh_tmp();
                let eq = self.fresh_tmp();
                let ok_l = self.fresh_lbl("vok");
                self.code.push_str(&format!(
                    "  {} = call i64 @__tn_rc_tag(i64 {})\n  {} = icmp eq i64 {}, {}\n  br i1 {}, label %{}, label %{}\n{}:\n",
                    tg, val, eq, tg, t, eq, ok_l, fail_lbl, ok_l
                ));
                if let Some(bn) = binding {
                    if bn != "_" {
                        let pay = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @__tn_slot_get(i64 {}, i64 0)\n",
                            pay, val
                        ));
                        let bty = self.infer_match_binding_type(ty, name);
                        bindings.push((bn.clone(), pay, bty));
                    }
                }
                Ok(())
            }
            Pattern::VariantMulti { name, patterns, .. } => {
                let t = *self
                    .enum_tags
                    .get(name)
                    .ok_or_else(|| format!("unknown variant `{}` in LLVM match", name))?;
                let tg = self.fresh_tmp();
                let eq = self.fresh_tmp();
                let ok_l = self.fresh_lbl("vok");
                self.code.push_str(&format!(
                    "  {} = call i64 @__tn_rc_tag(i64 {})\n  {} = icmp eq i64 {}, {}\n  br i1 {}, label %{}, label %{}\n{}:\n",
                    tg, val, eq, tg, t, eq, ok_l, fail_lbl, ok_l
                ));
                let ptys = self.resolve_variant_payload_tys(&base, name);
                for (idx, sp) in patterns.iter().enumerate() {
                    if matches!(sp, Pattern::Wildcard(_)) {
                        continue;
                    }
                    let fv = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n",
                        fv, val, idx
                    ));
                    let pty = ptys.get(idx).cloned().unwrap_or_else(|| named_ty("Any"));
                    self.emit_pattern_check(&fv, &pty, sp, fail_lbl, bindings)?;
                }
                Ok(())
            }
        }
    }

    fn lower_match(
        &mut self,
        scrutinee: &Expr,
        arms: &[MatchArm],
        want_owned: bool,
        expected_ty: Option<&TypeExpr>,
    ) -> Result<String, String> {
        let scrutinee_ty = self.infer_expr_type(scrutinee);

        let mut mtemps = Vec::new();
        let sv = self.lower_expr_borrowed(scrutinee, &mut mtemps)?;
        let temp_idx = self.temp_rc_scopes.len();
        self.temp_rc_scopes.push(mtemps);

        self.slot_id += 1;
        let res_slot = format!("%mres_{}", self.slot_id);
        self.entry_allocas.push_str(&format!(
            "  {} = alloca i64\n  store i64 0, ptr {}\n",
            res_slot, res_slot
        ));
        let end_l = self.fresh_lbl("mend");
        let mut any_reachable = false;

        for arm in arms {
            let is_wild = arm.guard.is_none()
                && arm.patterns.len() == 1
                && matches!(arm.patterns[0], Pattern::Wildcard(_));
            if is_wild {
                let v = if want_owned {
                    self.lower_expr_owned(&arm.body, expected_ty)?
                } else {
                    let mut atemps = Vec::new();
                    let av = self.lower_expr_borrowed(&arm.body, &mut atemps)?;
                    if !self.terminated {
                        self.flush_temps(&atemps);
                    }
                    av
                };
                if !self.terminated {
                    self.code.push_str(&format!(
                        "  store i64 {}, ptr {}\n  br label %{}\n",
                        v, res_slot, end_l
                    ));
                    any_reachable = true;
                }
                self.terminated = false;
                self.code.push_str(&format!("{}:\n", end_l));
                let mt = self.temp_rc_scopes.pop().unwrap();
                if !any_reachable {
                    self.code.push_str("  unreachable\n");
                    self.terminated = true;
                    return Ok("0".into());
                }
                self.flush_temps(&mt);
                let out = self.fresh_tmp();
                self.code
                    .push_str(&format!("  {} = load i64, ptr {}\n", out, res_slot));
                return Ok(out);
            }

            let nxt_l = self.fresh_lbl("mnxt");
            let mut bindings: Vec<(String, String, TypeExpr)> = Vec::new();

            if arm.patterns.len() == 1 {
                self.emit_pattern_check(&sv, &scrutinee_ty, &arm.patterns[0], &nxt_l, &mut bindings)?;
            } else {
                let mat_l = self.fresh_lbl("marm");
                for (pi, p) in arm.patterns.iter().enumerate() {
                    let fail_target = if pi + 1 < arm.patterns.len() {
                        self.fresh_lbl("mor")
                    } else {
                        nxt_l.clone()
                    };
                    let mut pbinds = Vec::new();
                    self.emit_pattern_check(&sv, &scrutinee_ty, p, &fail_target, &mut pbinds)?;
                    self.code.push_str(&format!("  br label %{}\n", mat_l));
                    if pi + 1 < arm.patterns.len() {
                        self.code.push_str(&format!("{}:\n", fail_target));
                    }
                }
                self.code.push_str(&format!("{}:\n", mat_l));
            }

            let saved_arm = self.save_lexical();
            for (bn, bval, bty) in bindings {
                let bslot = self.fresh_var_slot(&bn);
                self.code
                    .push_str(&format!("  store i64 {}, ptr {}\n", bval, bslot));
                self.owned_rc_vars.remove(&bn);
                if self.type_is_rc(&bty) {
                    self.borrowed_rc_vars.insert(bn.clone());
                } else {
                    self.borrowed_rc_vars.remove(&bn);
                }
                self.var_types.insert(bn, bty);
            }
            if let Some(guard_expr) = &arm.guard {
                let mut gtemps = Vec::new();
                let gv = self.lower_expr_borrowed(guard_expr, &mut gtemps)?;
                self.flush_temps(&gtemps);
                let gc = self.fresh_tmp();
                let body_l = self.fresh_lbl("mgbody");
                self.code.push_str(&format!(
                    "  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n",
                    gc, gv, gc, body_l, nxt_l, body_l
                ));
            }
            let av = if want_owned {
                self.lower_expr_owned(&arm.body, expected_ty)?
            } else {
                let mut atemps = Vec::new();
                let v = self.lower_expr_borrowed(&arm.body, &mut atemps)?;
                if !self.terminated {
                    self.flush_temps(&atemps);
                }
                v
            };
            self.restore_lexical(saved_arm);
            if !self.terminated {
                self.code.push_str(&format!(
                    "  store i64 {}, ptr {}\n  br label %{}\n",
                    av, res_slot, end_l
                ));
                any_reachable = true;
            }
            self.terminated = false;
            self.code.push_str(&format!("{}:\n", nxt_l));
        }

        self.code
            .push_str(&format!("  br label %{}\n{}:\n", end_l, end_l));
        let mt = self.temp_rc_scopes.remove(temp_idx);
        self.flush_temps(&mt);
        let out = self.fresh_tmp();
        self.code
            .push_str(&format!("  {} = load i64, ptr {}\n", out, res_slot));
        Ok(out)
    }

    fn lower_expr_fresh(
        &mut self,
        expr: &Expr,
        expected_ty: Option<&TypeExpr>,
    ) -> Result<String, String> {
        match &expr.kind {
            ExprKind::Int(n) => Ok(n.to_string()),
            ExprKind::Float(x) => Ok((x.to_bits() as i64).to_string()),
            ExprKind::Bool(b) => Ok(if *b { "1" } else { "0" }.into()),
            ExprKind::Str(s) => {
                let gname = self.intern_string_global(s);
                let obj = self.fresh_tmp();
                self.code
                    .push_str(&format!("  {} = ptrtoint ptr @{} to i64\n", obj, gname));
                Ok(obj)
            }
            ExprKind::RecordLit { name, fields, .. } => {
                let layout = self
                    .records
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown record `{}`", name))?;
                if let Some(c_lay) = &layout.c_layout {
                    let obj = self.fresh_tmp();
                    let base = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @tenon_cstruct_alloc(i64 {})\n  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n",
                        obj, c_lay.size, base, obj
                    ));
                    for (fname, fexpr) in fields {
                        if let Some(cfl) = c_lay.fields.iter().find(|f| f.name == *fname) {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(fexpr, &mut temps)?;
                            self.emit_mem_store(&base, cfl.byte_offset as i64, &v, &cfl.ty, false)?;
                            self.flush_temps(&temps);
                        }
                    }
                    return Ok(obj);
                }
                let obj = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 {})\n",
                    obj,
                    layout.tag,
                    layout.fields.len()
                ));
                if layout.rc_mask != 0 {
                    self.code.push_str(&format!(
                        "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                        obj, layout.rc_mask
                    ));
                }
                for (fname, fexpr) in fields {
                    if let Some((idx, fl)) = layout
                        .fields
                        .iter()
                        .enumerate()
                        .find(|(_, f)| f.name == *fname)
                    {
                        let fv = if fl.is_rc {
                            self.lower_expr_owned(fexpr, Some(&fl.ty))?
                        } else {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(fexpr, &mut temps)?;
                            self.flush_temps(&temps);
                            v
                        };
                        self.code.push_str(&format!(
                            "  call void @__tn_slot_set_prim(i64 {}, i64 {}, i64 {})\n",
                            obj, idx, fv
                        ));
                    }
                }
                Ok(obj)
            }
            ExprKind::ArrayRepeat { value, len } => {
                let arr_ty = expected_ty
                    .cloned()
                    .filter(|t| matches!(resolve_alias_ty(t, self.alias_bases), TypeExpr::Array { .. }))
                    .unwrap_or_else(|| TypeExpr::Array {
                        elem: Box::new(self.infer_expr_type(value)),
                        len: *len,
                        span: expr.span,
                    });
                let (elem_ty, alen, stride, elem_is_rc) = self.array_layout(&arr_ty).unwrap();
                if !elem_is_rc {
                    let total_bytes = alen * stride;
                    let obj = self.fresh_tmp();
                    let base = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @tenon_cstruct_alloc(i64 {})\n  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n",
                        obj, total_bytes, base, obj
                    ));
                    let mut temps = Vec::new();
                    let val = self.lower_expr_borrowed(value, &mut temps)?;
                    if alen <= 16 {
                        for idx in 0..alen {
                            self.emit_mem_store(&base, (idx * stride) as i64, &val, &elem_ty, false)?;
                        }
                    } else {
                        self.slot_id += 1;
                        let islot = format!("%arrep_{}", self.slot_id);
                        self.entry_allocas.push_str(&format!("  {} = alloca i64\n", islot));
                        let hdr = self.fresh_lbl("arr_hdr");
                        let bdy = self.fresh_lbl("arr_bdy");
                        let end = self.fresh_lbl("arr_end");
                        self.code.push_str(&format!(
                            "  store i64 0, ptr {}\n  br label %{}\n{}:\n",
                            islot, hdr, hdr
                        ));
                        let cur = self.fresh_tmp();
                        let cmp = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = load i64, ptr {}\n  {} = icmp slt i64 {}, {}\n  br i1 {}, label %{}, label %{}\n{}:\n",
                            cur, islot, cmp, cur, alen, cmp, bdy, end, bdy
                        ));
                        let boff = self.fresh_tmp();
                        let eptr = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = mul i64 {}, {}\n  {} = add i64 {}, {}\n",
                            boff, cur, stride, eptr, base, boff
                        ));
                        self.emit_mem_store(&eptr, 0, &val, &elem_ty, false)?;
                        let nxt = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = add i64 {}, 1\n  store i64 {}, ptr {}\n  br label %{}\n{}:\n",
                            nxt, cur, nxt, islot, hdr, end
                        ));
                    }
                    self.flush_temps(&temps);
                    return Ok(obj);
                }
                let mut temps = Vec::new();
                let init_v = self.lower_expr_borrowed(value, &mut temps)?;
                let res = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @tenon_list_filled(i64 {}, i64 {}, i64 1)\n",
                    res, alen, init_v
                ));
                self.flush_temps(&temps);
                Ok(res)
            }
            ExprKind::Cast { expr: inner, ty } | ExprKind::TryCast { expr: inner, ty } => {
                if let ExprKind::Float(x) = &inner.kind {
                    if is_f32_type_expr(ty, self.alias_bases) {
                        return Ok(((*x as f32).to_bits() as i64).to_string());
                    }
                }
                let src_ty = self.infer_expr_type(inner);
                let mut temps = Vec::new();
                let v = self.lower_expr_borrowed(inner, &mut temps)?;
                self.flush_temps(&temps);
                Ok(self.emit_cast(&v, &src_ty, ty))
            }
            ExprKind::List(elems) => {
                if let Some(exp) = expected_ty {
                    if let Some((elem_ty, alen, stride, false)) = self.array_layout(exp) {
                        let total_bytes = alen * stride;
                        let obj = self.fresh_tmp();
                        let base = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_cstruct_alloc(i64 {})\n  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n",
                            obj, total_bytes, base, obj
                        ));
                        for (idx, e) in elems.iter().enumerate() {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(e, &mut temps)?;
                            self.emit_mem_store(&base, (idx * stride) as i64, &v, &elem_ty, false)?;
                            self.flush_temps(&temps);
                        }
                        return Ok(obj);
                    }
                }
                if let Some((vec_ty, elem_bytes, _)) = expected_ty
                    .and_then(|t| self.flat_simd_list_info(t))
                    .or_else(|| elems.first().and_then(|e| self.simd_elem_info(&self.infer_expr_type(e))))
                {
                    let obj = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @tenon_flat_list_filled(i64 {}, i64 {}, ptr null)\n",
                        obj,
                        elems.len(),
                        elem_bytes
                    ));
                    let base = self.fresh_tmp();
                    let data = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 40\n",
                        base, obj, data, base
                    ));
                    for (idx, e) in elems.iter().enumerate() {
                        let mut temps = Vec::new();
                        let vv = self.lower_simd_vec_reg(e, vec_ty, elem_bytes, &mut temps)?;
                        let ep = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = getelementptr inbounds {}, ptr {}, i64 {}\n  store {} {}, ptr {}, align 8, !tbaa !7\n",
                            ep, vec_ty, data, idx, vec_ty, vv, ep
                        ));
                        self.flush_temps(&temps);
                    }
                    return Ok(obj);
                }
                if expected_ty
                    .and_then(|t| self.flat_scalar2_list_info(t))
                    .or_else(|| elems.first().and_then(|e| self.scalar2_elem_info(&self.infer_expr_type(e))))
                    .is_some()
                {
                    let obj = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @tenon_flat_list_filled(i64 {}, i64 2, ptr null)\n",
                        obj,
                        elems.len()
                    ));
                    let base = self.fresh_tmp();
                    let data = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 40\n",
                        base, obj, data, base
                    ));
                    for (idx, e) in elems.iter().enumerate() {
                        let mut temps = Vec::new();
                        let vv = self.lower_expr_borrowed(e, &mut temps)?;
                        let v16 = self.fresh_tmp();
                        let ep = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = trunc i64 {} to i16\n  {} = getelementptr inbounds i16, ptr {}, i64 {}\n  store i16 {}, ptr {}, align 2, !tbaa !7\n",
                            v16, vv, ep, data, idx, v16, ep
                        ));
                        self.flush_temps(&temps);
                    }
                    return Ok(obj);
                }
                if expected_ty
                    .and_then(|t| self.flat_scalar4_list_info(t))
                    .or_else(|| elems.first().and_then(|e| self.scalar4_elem_info(&self.infer_expr_type(e))))
                    .is_some()
                {
                    let obj = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @tenon_flat_list_filled(i64 {}, i64 4, ptr null)\n",
                        obj,
                        elems.len()
                    ));
                    let base = self.fresh_tmp();
                    let data = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 40\n",
                        base, obj, data, base
                    ));
                    for (idx, e) in elems.iter().enumerate() {
                        let mut temps = Vec::new();
                        let vv = self.lower_expr_borrowed(e, &mut temps)?;
                        let v32 = self.fresh_tmp();
                        let ep = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = trunc i64 {} to i32\n  {} = getelementptr inbounds i32, ptr {}, i64 {}\n  store i32 {}, ptr {}, align 4, !tbaa !7\n",
                            v32, vv, ep, data, idx, v32, ep
                        ));
                        self.flush_temps(&temps);
                    }
                    return Ok(obj);
                }
                let exp_elem_ty = expected_ty.and_then(|t| self.list_elem_ty(t));
                let elem_is_rc = exp_elem_ty
                    .as_ref()
                    .is_some_and(|et| self.type_is_rc(et))
                    || elems.iter().any(|e| self.expr_produces_rc(e));
                let obj = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 {})\n",
                    obj,
                    TAG_LIST,
                    elems.len()
                ));
                if elem_is_rc {
                    self.code.push_str(&format!(
                        "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                        obj, RC_MASK_ALL as i64
                    ));
                }
                for (idx, e) in elems.iter().enumerate() {
                    let ev = if elem_is_rc {
                        self.lower_expr_owned(e, exp_elem_ty.as_ref())?
                    } else {
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(e, &mut temps)?;
                        self.flush_temps(&temps);
                        v
                    };
                    self.code.push_str(&format!(
                        "  call void @__tn_slot_set_prim(i64 {}, i64 {}, i64 {})\n",
                        obj, idx, ev
                    ));
                }
                Ok(obj)
            }
            ExprKind::Tuple(elems) => {
                let exp_elems = expected_ty.and_then(|t| match resolve_alias_ty(t, self.alias_bases) {
                    TypeExpr::Tuple(es, _) => Some(es),
                    _ => None,
                });
                let obj = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 {})\n",
                    obj,
                    TAG_TUPLE,
                    elems.len()
                ));
                let mut rc_mask = 0u64;
                for (idx, e) in elems.iter().enumerate() {
                    let ety = exp_elems.as_ref().and_then(|es| es.get(idx));
                    let is_rc = ety.is_some_and(|t| self.type_is_rc(t)) || self.expr_produces_rc(e);
                    let ev = if is_rc {
                        if idx < 64 {
                            rc_mask |= 1u64 << idx;
                        }
                        self.lower_expr_owned(e, ety)?
                    } else {
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(e, &mut temps)?;
                        self.flush_temps(&temps);
                        v
                    };
                    self.code.push_str(&format!(
                        "  call void @__tn_slot_set_prim(i64 {}, i64 {}, i64 {})\n",
                        obj, idx, ev
                    ));
                }
                if rc_mask != 0 {
                    self.code.push_str(&format!(
                        "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                        obj, rc_mask as i64
                    ));
                }
                Ok(obj)
            }
            ExprKind::Unary { op, expr } => {
                let sk = self.scalar_kind_of_expr(expr);
                let is_flt = self.is_float_expr(expr);
                let mut temps = Vec::new();
                let v = self.lower_expr_borrowed(expr, &mut temps)?;
                self.flush_temps(&temps);
                let r = self.fresh_tmp();
                match op {
                    UnOp::Neg => {
                        if matches!(sk, Some(ScalarKind::F16 | ScalarKind::BF16)) {
                            self.code.push_str(&format!("  {} = xor i64 {}, 32768\n", r, v));
                        } else if sk == Some(ScalarKind::F32) {
                            let t32 = self.fresh_tmp();
                            let fv = self.fresh_tmp();
                            let nf = self.fresh_tmp();
                            let b32 = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = trunc i64 {} to i32\n  {} = bitcast i32 {} to float\n  {} = fneg float {}\n  {} = bitcast float {} to i32\n  {} = zext i32 {} to i64\n",
                                t32, v, fv, t32, nf, fv, b32, nf, r, b32
                            ));
                        } else if is_flt {
                            let fv = self.fresh_tmp();
                            let nf = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = bitcast i64 {} to double\n  {} = fneg double {}\n  {} = bitcast double {} to i64\n",
                                fv, v, nf, fv, r, nf
                            ));
                        } else {
                            self.code.push_str(&format!("  {} = sub i64 0, {}\n", r, v));
                            if let Some(k) = sk {
                                return Ok(self.normalize_scalar(&r, k));
                            }
                        }
                    }
                    UnOp::BitNot => {
                        self.code.push_str(&format!("  {} = xor i64 {}, -1\n", r, v));
                        if let Some(k) = sk {
                            return Ok(self.normalize_scalar(&r, k));
                        }
                    }
                    UnOp::Not => {
                        let c = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = icmp eq i64 {}, 0\n  {} = zext i1 {} to i64\n",
                            c, v, r, c
                        ));
                    }
                }
                Ok(r)
            }
            ExprKind::Binary {
                op: BinOp::And,
                left,
                right,
            } => {
                self.slot_id += 1;
                let slot = format!("%and_{}", self.slot_id);
                self.entry_allocas.push_str(&format!(
                    "  {} = alloca i64\n  store i64 0, ptr {}\n",
                    slot, slot
                ));
                let mut ltemps = Vec::new();
                let l = self.lower_expr_borrowed(left, &mut ltemps)?;
                self.flush_temps(&ltemps);
                let lb = self.fresh_tmp();
                let rhs_l = self.fresh_lbl("and_rhs");
                let end_l = self.fresh_lbl("and_end");
                self.code.push_str(&format!(
                    "  store i64 0, ptr {}\n  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n",
                    slot, lb, l, lb, rhs_l, end_l, rhs_l
                ));
                let mut rtemps = Vec::new();
                let r = self.lower_expr_borrowed(right, &mut rtemps)?;
                self.flush_temps(&rtemps);
                let rb = self.fresh_tmp();
                let rz = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = icmp ne i64 {}, 0\n  {} = zext i1 {} to i64\n  store i64 {}, ptr {}\n  br label %{}\n{}:\n",
                    rb, r, rz, rb, rz, slot, end_l, end_l
                ));
                let out = self.fresh_tmp();
                self.code
                    .push_str(&format!("  {} = load i64, ptr {}\n", out, slot));
                Ok(out)
            }
            ExprKind::Binary {
                op: BinOp::Or,
                left,
                right,
            } => {
                self.slot_id += 1;
                let slot = format!("%or_{}", self.slot_id);
                self.entry_allocas.push_str(&format!(
                    "  {} = alloca i64\n  store i64 0, ptr {}\n",
                    slot, slot
                ));
                let mut ltemps = Vec::new();
                let l = self.lower_expr_borrowed(left, &mut ltemps)?;
                self.flush_temps(&ltemps);
                let lb = self.fresh_tmp();
                let rhs_l = self.fresh_lbl("or_rhs");
                let end_l = self.fresh_lbl("or_end");
                self.code.push_str(&format!(
                    "  store i64 1, ptr {}\n  {} = icmp ne i64 {}, 0\n  br i1 {}, label %{}, label %{}\n{}:\n",
                    slot, lb, l, lb, end_l, rhs_l, rhs_l
                ));
                let mut rtemps = Vec::new();
                let r = self.lower_expr_borrowed(right, &mut rtemps)?;
                self.flush_temps(&rtemps);
                let rb = self.fresh_tmp();
                let rz = self.fresh_tmp();
                self.code.push_str(&format!(
                    "  {} = icmp ne i64 {}, 0\n  {} = zext i1 {} to i64\n  store i64 {}, ptr {}\n  br label %{}\n{}:\n",
                    rb, r, rz, rb, rz, slot, end_l, end_l
                ));
                let out = self.fresh_tmp();
                self.code
                    .push_str(&format!("  {} = load i64, ptr {}\n", out, slot));
                Ok(out)
            }
            ExprKind::Binary { op, left, right } => {
                let tcmp = self.is_text_expr(left) || self.is_text_expr(right);
                let lk = self.scalar_kind_of_expr(left);
                let rk = self.scalar_kind_of_expr(right);
                let eff_sk = match (lk, rk) {
                    (Some(ScalarKind::I64), Some(k)) if k != ScalarKind::I64 && matches!(left.kind, ExprKind::Int(_)) => Some(k),
                    (Some(ScalarKind::F32), _) | (_, Some(ScalarKind::F32)) => Some(ScalarKind::F32),
                    (Some(ScalarKind::F16), _) | (_, Some(ScalarKind::F16)) => Some(ScalarKind::F16),
                    (Some(ScalarKind::BF16), _) | (_, Some(ScalarKind::BF16)) => Some(ScalarKind::BF16),
                    (Some(k), _) => Some(k),
                    (None, k) => k,
                };
                let float_op = self.is_float_expr(left) || self.is_float_expr(right);
                let mut temps = Vec::new();
                let l = self.lower_expr_borrowed(left, &mut temps)?;
                let r = self.lower_expr_borrowed(right, &mut temps)?;
                if *op == BinOp::Concat {
                    let res = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @tenon_str_concat(i64 {}, i64 {})\n",
                        res, l, r
                    ));
                    self.flush_temps(&temps);
                    return Ok(res);
                }
                if tcmp && matches!(op, BinOp::Eq | BinOp::Ne) {
                    let eq = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @__tn_text_eq(i64 {}, i64 {})\n",
                        eq, l, r
                    ));
                    self.flush_temps(&temps);
                    if *op == BinOp::Eq {
                        return Ok(eq);
                    }
                    let c = self.fresh_tmp();
                    let res = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = icmp eq i64 {}, 0\n  {} = zext i1 {} to i64\n",
                        c, eq, res, c
                    ));
                    return Ok(res);
                }
                self.flush_temps(&temps);
                if matches!(eff_sk, Some(ScalarKind::F16 | ScalarKind::BF16)) {
                    let is_f16 = eff_sk == Some(ScalarKind::F16);
                    let lf = if is_f16 { self.emit_f16_to_float(&l) } else { self.emit_bf16_to_float(&l) };
                    let rf = if is_f16 { self.emit_f16_to_float(&r) } else { self.emit_bf16_to_float(&r) };
                    match op {
                        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                            let fop = match op {
                                BinOp::Add => "fadd",
                                BinOp::Sub => "fsub",
                                BinOp::Mul => "fmul",
                                BinOp::Div => "fdiv",
                                _ => unreachable!(),
                            };
                            let fres = self.fresh_tmp();
                            self.code.push_str(&format!("  {} = {} float {}, {}\n", fres, fop, lf, rf));
                            let out = if is_f16 {
                                self.emit_float_to_f16(&fres)
                            } else {
                                self.emit_float_to_bf16(&fres)
                            };
                            return Ok(out);
                        }
                        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                            let pred = match op {
                                BinOp::Eq => "oeq",
                                BinOp::Ne => "une",
                                BinOp::Lt => "olt",
                                BinOp::Le => "ole",
                                BinOp::Gt => "ogt",
                                BinOp::Ge => "oge",
                                _ => unreachable!(),
                            };
                            let c = self.fresh_tmp();
                            let res = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = fcmp {} float {}, {}\n  {} = zext i1 {} to i64\n",
                                c, pred, lf, rf, res, c
                            ));
                            return Ok(res);
                        }
                        other => return Err(format!("unsupported float16/bfloat16 op {:?} in LLVM codegen", other)),
                    }
                }
                if eff_sk == Some(ScalarKind::F32) {
                    let l32 = self.fresh_tmp();
                    let r32 = self.fresh_tmp();
                    let lf = self.fresh_tmp();
                    let rf = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = trunc i64 {} to i32\n  {} = trunc i64 {} to i32\n  {} = bitcast i32 {} to float\n  {} = bitcast i32 {} to float\n",
                        l32, l, r32, r, lf, l32, rf, r32
                    ));
                    let res = self.fresh_tmp();
                    match op {
                        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                            let fop = match op {
                                BinOp::Add => "fadd",
                                BinOp::Sub => "fsub",
                                BinOp::Mul => "fmul",
                                BinOp::Div => "fdiv",
                                _ => unreachable!(),
                            };
                            let fres = self.fresh_tmp();
                            let b32 = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = {} float {}, {}\n  {} = bitcast float {} to i32\n  {} = zext i32 {} to i64\n",
                                fres, fop, lf, rf, b32, fres, res, b32
                            ));
                        }
                        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                            let pred = match op {
                                BinOp::Eq => "oeq",
                                BinOp::Ne => "une",
                                BinOp::Lt => "olt",
                                BinOp::Le => "ole",
                                BinOp::Gt => "ogt",
                                BinOp::Ge => "oge",
                                _ => unreachable!(),
                            };
                            let c = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = fcmp {} float {}, {}\n  {} = zext i1 {} to i64\n",
                                c, pred, lf, rf, res, c
                            ));
                        }
                        other => return Err(format!("unsupported float32 op {:?} in LLVM codegen", other)),
                    }
                    return Ok(res);
                }
                if float_op {
                    let lf = self.fresh_tmp();
                    let rf = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = bitcast i64 {} to double\n  {} = bitcast i64 {} to double\n",
                        lf, l, rf, r
                    ));
                    let res = self.fresh_tmp();
                    match op {
                        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                            let fop = match op {
                                BinOp::Add => "fadd",
                                BinOp::Sub => "fsub",
                                BinOp::Mul => "fmul",
                                BinOp::Div => "fdiv",
                                _ => unreachable!(),
                            };
                            let fres = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = {} double {}, {}\n  {} = bitcast double {} to i64\n",
                                fres, fop, lf, rf, res, fres
                            ));
                        }
                        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                            let pred = match op {
                                BinOp::Eq => "oeq",
                                BinOp::Ne => "une",
                                BinOp::Lt => "olt",
                                BinOp::Le => "ole",
                                BinOp::Gt => "ogt",
                                BinOp::Ge => "oge",
                                _ => unreachable!(),
                            };
                            let c = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = fcmp {} double {}, {}\n  {} = zext i1 {} to i64\n",
                                c, pred, lf, rf, res, c
                            ));
                        }
                        other => return Err(format!("unsupported float op {:?} in LLVM codegen", other)),
                    }
                    return Ok(res);
                }
                let is_unsigned = eff_sk.is_some_and(|k| k.is_unsigned());
                let is_signed_narrow = matches!(eff_sk, Some(ScalarKind::I8 | ScalarKind::I16 | ScalarKind::I32));
                let is_cmp = matches!(
                    op,
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
                );
                let is_i64_signed = matches!(eff_sk, None | Some(ScalarKind::I64));
                let is_proved_no_ov = self.proved_no_overflow.contains(&expr.span.start);
                let emit_checked_i64 = self.checked_overflow && !is_proved_no_ov && is_i64_signed;
                let res = self.fresh_tmp();
                match op {
                    BinOp::Add => {
                        if emit_checked_i64 {
                            let ov_tup = self.fresh_tmp();
                            let ov_bit = self.fresh_tmp();
                            let ok_lbl = self.fresh_lbl("ov_ok");
                            let trap_lbl = self.fresh_lbl("ov_trap");
                            self.code.push_str(&format!(
                                "  {} = call {{ i64, i1 }} @llvm.sadd.with.overflow.i64(i64 {}, i64 {})\n  {} = extractvalue {{ i64, i1 }} {}, 1\n  br i1 {}, label %{}, label %{}\n{}:\n  call void @tenon_rt_overflow_trap(i64 {})\n  unreachable\n{}:\n  {} = extractvalue {{ i64, i1 }} {}, 0\n",
                                ov_tup, l, r, ov_bit, ov_tup, ov_bit, trap_lbl, ok_lbl, trap_lbl, expr.span.line, ok_lbl, res, ov_tup
                            ));
                        } else if is_proved_no_ov && is_i64_signed {
                            self.code.push_str(&format!("  {} = add nsw i64 {}, {}\n", res, l, r));
                        } else {
                            self.code.push_str(&format!("  {} = add i64 {}, {}\n", res, l, r));
                        }
                    }
                    BinOp::Sub => {
                        if emit_checked_i64 {
                            let ov_tup = self.fresh_tmp();
                            let ov_bit = self.fresh_tmp();
                            let ok_lbl = self.fresh_lbl("ov_ok");
                            let trap_lbl = self.fresh_lbl("ov_trap");
                            self.code.push_str(&format!(
                                "  {} = call {{ i64, i1 }} @llvm.ssub.with.overflow.i64(i64 {}, i64 {})\n  {} = extractvalue {{ i64, i1 }} {}, 1\n  br i1 {}, label %{}, label %{}\n{}:\n  call void @tenon_rt_overflow_trap(i64 {})\n  unreachable\n{}:\n  {} = extractvalue {{ i64, i1 }} {}, 0\n",
                                ov_tup, l, r, ov_bit, ov_tup, ov_bit, trap_lbl, ok_lbl, trap_lbl, expr.span.line, ok_lbl, res, ov_tup
                            ));
                        } else if is_proved_no_ov && is_i64_signed {
                            self.code.push_str(&format!("  {} = sub nsw i64 {}, {}\n", res, l, r));
                        } else {
                            self.code.push_str(&format!("  {} = sub i64 {}, {}\n", res, l, r));
                        }
                    }
                    BinOp::Mul => {
                        if emit_checked_i64 {
                            let ov_tup = self.fresh_tmp();
                            let ov_bit = self.fresh_tmp();
                            let ok_lbl = self.fresh_lbl("ov_ok");
                            let trap_lbl = self.fresh_lbl("ov_trap");
                            self.code.push_str(&format!(
                                "  {} = call {{ i64, i1 }} @llvm.smul.with.overflow.i64(i64 {}, i64 {})\n  {} = extractvalue {{ i64, i1 }} {}, 1\n  br i1 {}, label %{}, label %{}\n{}:\n  call void @tenon_rt_overflow_trap(i64 {})\n  unreachable\n{}:\n  {} = extractvalue {{ i64, i1 }} {}, 0\n",
                                ov_tup, l, r, ov_bit, ov_tup, ov_bit, trap_lbl, ok_lbl, trap_lbl, expr.span.line, ok_lbl, res, ov_tup
                            ));
                        } else if is_proved_no_ov && is_i64_signed {
                            self.code.push_str(&format!("  {} = mul nsw i64 {}, {}\n", res, l, r));
                        } else {
                            self.code.push_str(&format!("  {} = mul i64 {}, {}\n", res, l, r));
                        }
                    }
                    BinOp::Div => {
                        let iop = if is_unsigned { "udiv" } else { "sdiv" };
                        self.code.push_str(&format!("  {} = {} i64 {}, {}\n", res, iop, l, r));
                    }
                    BinOp::Mod => {
                        let iop = if is_unsigned { "urem" } else { "srem" };
                        self.code.push_str(&format!("  {} = {} i64 {}, {}\n", res, iop, l, r));
                    }
                    BinOp::BitAnd => self.code.push_str(&format!("  {} = and i64 {}, {}\n", res, l, r)),
                    BinOp::BitOr => self.code.push_str(&format!("  {} = or i64 {}, {}\n", res, l, r)),
                    BinOp::BitXor => self.code.push_str(&format!("  {} = xor i64 {}, {}\n", res, l, r)),
                    BinOp::Shl => self.code.push_str(&format!("  {} = shl i64 {}, {}\n", res, l, r)),
                    BinOp::Shr => {
                        let iop = if is_signed_narrow { "ashr" } else { "lshr" };
                        self.code.push_str(&format!("  {} = {} i64 {}, {}\n", res, iop, l, r));
                    }
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        let pred = match op {
                            BinOp::Eq => "eq",
                            BinOp::Ne => "ne",
                            BinOp::Lt => if is_unsigned { "ult" } else { "slt" },
                            BinOp::Le => if is_unsigned { "ule" } else { "sle" },
                            BinOp::Gt => if is_unsigned { "ugt" } else { "sgt" },
                            BinOp::Ge => if is_unsigned { "uge" } else { "sge" },
                            _ => unreachable!(),
                        };
                        let c = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = icmp {} i64 {}, {}\n  {} = zext i1 {} to i64\n",
                            c, pred, l, r, res, c
                        ));
                    }
                    other => return Err(format!("unsupported op {:?} in LLVM codegen", other)),
                }
                if !is_cmp {
                    if let Some(k) = eff_sk {
                        return Ok(self.normalize_scalar(&res, k));
                    }
                }
                Ok(res)
            }
            ExprKind::Call { callee, args } => match &callee.kind {
                ExprKind::Field { target, field } if field == "parse" => {
                    if let ExprKind::Ident(rname) = &target.kind {
                        let mut temps = Vec::new();
                        let mut avals = Vec::new();
                        for a in args {
                            avals.push(format!(
                                "i64 {}",
                                self.lower_expr_borrowed(&a.value, &mut temps)?
                            ));
                        }
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_fn_{}__parse({})\n",
                            res,
                            sanitize(rname),
                            avals.join(", ")
                        ));
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    Err("unsupported `.parse()` target in LLVM".into())
                }
                ExprKind::Field { target, field } => {
                    if let ExprKind::Ident(ename) = &target.kind {
                        let key = format!("{}.{}", ename, field);
                        if let Some(&tag) = self.enum_tags.get(&key) {
                            let ptys = self
                                .enum_variants
                                .get(&key)
                                .map(|vl| vl.payload_tys.clone())
                                .unwrap_or_default();
                            return self.lower_enum_ctor(tag, &ptys, args);
                        }
                    }
                    // Dynamic interface method dispatch: `d.method(args)` where `d: dyn Interface`!
                    let recv_ty = resolve_alias_ty(&self.infer_expr_type(target), self.alias_bases);
                    if let TypeExpr::Dyn { interface, .. } = recv_ty {
                        let dm = self
                            .dyn_interfaces
                            .get(&interface)
                            .and_then(|l| l.methods.iter().find(|m| m.name == *field))
                            .cloned()
                            .ok_or_else(|| {
                                format!("unknown method `{}` on `dyn {}` in LLVM codegen", field, interface)
                            })?;
                        let mut temps = Vec::new();
                        let mut d_val = self.lower_expr_borrowed(target, &mut temps)?;
                        let mut call_args = Vec::new();
                        let mut inout_dyn_slot: Option<(String, String, Option<String>)> = None;

                        if dm.is_inout_self {
                            let d_uniq = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n",
                                d_uniq, d_val
                            ));
                            if let ExprKind::Ident(vname) = &target.kind {
                                if let Some(ptr) = self.var_ptrs.get(vname).cloned() {
                                    if self.inout_vars.contains(vname) {
                                        let slot_addr = self.fresh_tmp();
                                        let p = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                            slot_addr, ptr, p, slot_addr, d_uniq, p
                                        ));
                                    } else {
                                        self.code
                                            .push_str(&format!("  store i64 {}, ptr {}\n", d_uniq, ptr));
                                    }
                                }
                            }
                            d_val = d_uniq.clone();
                            let data_val = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = call i64 @__tn_slot_get(i64 {}, i64 0)\n",
                                data_val, d_val
                            ));
                            self.slot_id += 1;
                            let slot = format!("%dyn_inout_{}", self.slot_id);
                            self.entry_allocas.push_str(&format!(
                                "  {} = alloca i64\n  store i64 0, ptr {}\n",
                                slot, slot
                            ));
                            let slot_i64 = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  store i64 {}, ptr {}\n  {} = ptrtoint ptr {} to i64\n",
                                data_val, slot, slot_i64, slot
                            ));
                            call_args.push(format!("i64 {}", slot_i64));
                            inout_dyn_slot = Some((d_uniq, slot, None));
                        } else {
                            let data_val = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = call i64 @__tn_slot_get(i64 {}, i64 0)\n",
                                data_val, d_val
                            ));
                            call_args.push(format!("i64 {}", data_val));
                        }

                        for a in args {
                            let av = self.lower_expr_borrowed(&a.value, &mut temps)?;
                            call_args.push(format!("i64 {}", av));
                        }

                        let vt_i64 = self.fresh_tmp();
                        let vt_ptr = self.fresh_tmp();
                        let slot_ptr = self.fresh_tmp();
                        let fn_ptr = self.fresh_tmp();
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @__tn_slot_get(i64 {}, i64 1)\n  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds ptr, ptr {}, i64 {}\n  {} = load ptr, ptr {}\n  {} = call i64 {}({})\n",
                            vt_i64, d_val,
                            vt_ptr, vt_i64,
                            slot_ptr, vt_ptr, dm.slot_idx,
                            fn_ptr, slot_ptr,
                            res, fn_ptr, call_args.join(", ")
                        ));
                        if let Some((d_uniq, slot, _)) = inout_dyn_slot {
                            let updated = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = load i64, ptr {}\n  call void @__tn_slot_set_prim(i64 {}, i64 0, i64 {})\n",
                                updated, slot, d_uniq, updated
                            ));
                        }
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    Err(format!(
                        "unsupported method call `{}` on `{}` in LLVM codegen",
                        field,
                        recv_ty.display_name()
                    ))
                }
                ExprKind::Ident(fname) => {
                    if fname == "__make_closure" && args.len() >= 2 {
                        let csym = match &args[0].value.kind {
                            ExprKind::Str(s) => s.clone(),
                            _ => return Err("__make_closure expects closure fn symbol".into()),
                        };
                        let rc_mask = match &args[1].value.kind {
                            ExprKind::Int(m) => *m,
                            _ => 0,
                        };
                        let captures = &args[2..];
                        let obj = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 {})\n",
                            obj,
                            TAG_CLOSURE,
                            1 + captures.len()
                        ));
                        if rc_mask != 0 {
                            self.code.push_str(&format!(
                                "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                                obj, rc_mask
                            ));
                        }
                        let fptr = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = ptrtoint ptr @tenon_fn_{} to i64\n  call void @__tn_slot_set_prim(i64 {}, i64 0, i64 {})\n",
                            fptr,
                            sanitize(&csym),
                            obj,
                            fptr
                        ));
                        for (k, cap) in captures.iter().enumerate() {
                            let bit = 1i64.checked_shl((k + 1) as u32).unwrap_or(0);
                            let is_rc = (rc_mask & bit) != 0;
                            let cv = if is_rc {
                                self.lower_expr_owned(&cap.value, None)?
                            } else {
                                let mut temps = Vec::new();
                                let v = self.lower_expr_borrowed(&cap.value, &mut temps)?;
                                self.flush_temps(&temps);
                                v
                            };
                            self.code.push_str(&format!(
                                "  call void @__tn_slot_set_prim(i64 {}, i64 {}, i64 {})\n",
                                obj,
                                k + 1,
                                cv
                            ));
                        }
                        return Ok(obj);
                    }
                    if fname == "__closure_get" && args.len() == 2 {
                        let mut temps = Vec::new();
                        let env = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let idx = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n",
                            res, env, idx
                        ));
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    if fname == "__call_closure" && !args.is_empty() {
                        let mut temps = Vec::new();
                        let cl_val = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let mut call_args = Vec::with_capacity(args.len());
                        call_args.push(format!("i64 {}", cl_val));
                        for a in &args[1..] {
                            let av = self.lower_expr_borrowed(&a.value, &mut temps)?;
                            call_args.push(format!("i64 {}", av));
                        }
                        let fn_i64 = self.fresh_tmp();
                        let fn_ptr = self.fresh_tmp();
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @__tn_slot_get(i64 {}, i64 0)\n  {} = inttoptr i64 {} to ptr\n  {} = call i64 {}({})\n",
                            fn_i64,
                            cl_val,
                            fn_ptr,
                            fn_i64,
                            res,
                            fn_ptr,
                            call_args.join(", ")
                        ));
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    if fname == "__box_dyn" && args.len() == 3 {
                        let vsym = match &args[1].value.kind {
                            ExprKind::Str(s) => s.clone(),
                            _ => return Err("__box_dyn expects vtable symbol string".into()),
                        };
                        let data_is_rc = matches!(&args[2].value.kind, ExprKind::Int(n) if *n != 0);
                        let data_val = if data_is_rc {
                            self.lower_expr_owned(&args[0].value, None)?
                        } else {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                            self.flush_temps(&temps);
                            v
                        };
                        let obj = self.fresh_tmp();
                        let vt_i64 = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 2)\n",
                            obj, TAG_DYN
                        ));
                        if data_is_rc {
                            self.code.push_str(&format!(
                                "  call void @__tn_rc_set_rc_mask(i64 {}, i64 1)\n",
                                obj
                            ));
                        }
                        self.code.push_str(&format!(
                            "  call void @__tn_slot_set_prim(i64 {}, i64 0, i64 {})\n  {} = ptrtoint ptr @{} to i64\n  call void @__tn_slot_set_prim(i64 {}, i64 1, i64 {})\n",
                            obj, data_val,
                            vt_i64, sanitize(&vsym),
                            obj, vt_i64
                        ));
                        return Ok(obj);
                    }
                    if fname == "print_int" && args.len() == 1 {
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        self.flush_temps(&temps);
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_rt_print_int(i64 {})\n",
                            res, v
                        ));
                        return Ok(res);
                    }
                    if fname == "len" && args.len() == 1 {
                        let arg_ty = self.infer_expr_type(&args[0].value);
                        if let Some((_, alen, _, _)) = self.array_layout(&arg_ty) {
                            let mut temps = Vec::new();
                            let _ = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                            self.flush_temps(&temps);
                            return Ok(alen.to_string());
                        }
                        if matches!(resolve_alias_ty(&arg_ty, self.alias_bases), TypeExpr::Named { ref name, .. } if name == "BytesSlice") {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                            let res = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = call i64 @tenon_slice_len(i64 {})\n",
                                res, v
                            ));
                            self.flush_temps(&temps);
                            return Ok(res);
                        }
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @__tn_rc_len(i64 {})\n",
                            res, v
                        ));
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    // Native LLVM SIMD vector IR (takes priority over scalar syscall fallback!)
                    if let Some(simd_res) = self.try_emit_simd_builtin(fname, args, expected_ty)? {
                        return Ok(simd_res);
                    }
                    // Native LLVM hardware bit intrinsics
                    if matches!(
                        fname.as_str(),
                        "clz" | "ctz" | "popcnt" | "bswap16" | "bswap32" | "bswap64"
                    ) && args.len() == 1
                    {
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        self.flush_temps(&temps);
                        let res = self.fresh_tmp();
                        match fname.as_str() {
                            "clz" => self.code.push_str(&format!(
                                "  {} = call i64 @llvm.ctlz.i64(i64 {}, i1 false)\n",
                                res, v
                            )),
                            "ctz" => self.code.push_str(&format!(
                                "  {} = call i64 @llvm.cttz.i64(i64 {}, i1 false)\n",
                                res, v
                            )),
                            "popcnt" => self.code.push_str(&format!(
                                "  {} = call i64 @llvm.ctpop.i64(i64 {})\n",
                                res, v
                            )),
                            "bswap16" => {
                                let t = self.fresh_tmp();
                                let b = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = trunc i64 {} to i16\n  {} = call i16 @llvm.bswap.i16(i16 {})\n  {} = zext i16 {} to i64\n",
                                    t, v, b, t, res, b
                                ));
                            }
                            "bswap32" => {
                                let t = self.fresh_tmp();
                                let b = self.fresh_tmp();
                                self.code.push_str(&format!(
                                    "  {} = trunc i64 {} to i32\n  {} = call i32 @llvm.bswap.i32(i32 {})\n  {} = zext i32 {} to i64\n",
                                    t, v, b, t, res, b
                                ));
                            }
                            _ => self.code.push_str(&format!(
                                "  {} = call i64 @llvm.bswap.i64(i64 {})\n",
                                res, v
                            )),
                        }
                        return Ok(res);
                    }
                    if (fname == "rotl64" || fname == "rotr64") && args.len() == 2 {
                        let mut temps = Vec::new();
                        let x = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let n = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                        self.flush_temps(&temps);
                        let res = self.fresh_tmp();
                        let intr = if fname == "rotl64" { "@llvm.fshl.i64" } else { "@llvm.fshr.i64" };
                        self.code.push_str(&format!(
                            "  {} = call i64 {}(i64 {}, i64 {}, i64 {})\n",
                            res, intr, x, x, n
                        ));
                        return Ok(res);
                    }
                    if matches!(
                        fname.as_str(),
                        "wrapping_add" | "wrapping_sub" | "wrapping_mul" | "saturating_add" | "saturating_sub"
                    ) && args.len() == 2
                    {
                        let sk = self
                            .scalar_kind_of_expr(&args[0].value)
                            .or_else(|| self.scalar_kind_of_expr(&args[1].value))
                            .unwrap_or(ScalarKind::I64);
                        let mut temps = Vec::new();
                        let a = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let b = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                        self.flush_temps(&temps);
                        if matches!(fname.as_str(), "wrapping_add" | "wrapping_sub" | "wrapping_mul") {
                            let iop = match fname.as_str() {
                                "wrapping_add" => "add",
                                "wrapping_sub" => "sub",
                                _ => "mul",
                            };
                            let r = self.fresh_tmp();
                            self.code.push_str(&format!("  {} = {} i64 {}, {}\n", r, iop, a, b));
                            return Ok(self.normalize_scalar(&r, sk));
                        }
                        let (ity, bits) = match sk {
                            ScalarKind::I8 | ScalarKind::U8 => ("i8", 8),
                            ScalarKind::I16 | ScalarKind::U16 => ("i16", 16),
                            ScalarKind::I32 | ScalarKind::U32 => ("i32", 32),
                            _ => ("i64", 64),
                        };
                        let prefix = match (fname.as_str(), sk.is_unsigned()) {
                            ("saturating_add", false) => "sadd",
                            ("saturating_add", true) => "uadd",
                            (_, false) => "ssub",
                            (_, true) => "usub",
                        };
                        let res = self.fresh_tmp();
                        if bits == 64 {
                            self.code.push_str(&format!(
                                "  {} = call i64 @llvm.{}.sat.i64(i64 {}, i64 {})\n",
                                res, prefix, a, b
                            ));
                        } else {
                            let ta = self.fresh_tmp();
                            let tb = self.fresh_tmp();
                            let sv = self.fresh_tmp();
                            let ext = if sk.is_unsigned() { "zext" } else { "sext" };
                            self.code.push_str(&format!(
                                "  {} = trunc i64 {} to {}\n  {} = trunc i64 {} to {}\n  {} = call {} @llvm.{}.sat.{}({} {}, {} {})\n  {} = {} {} {} to i64\n",
                                ta, a, ity, tb, b, ity, sv, ity, prefix, ity, ity, ta, ity, tb, res, ext, ity, sv
                            ));
                        }
                        return Ok(res);
                    }
                    if is_consuming_cow_builtin(fname) && !args.is_empty() {
                        let csym = *self
                            .syscall_map
                            .get(fname)
                            .ok_or_else(|| format!("unknown cow builtin `{}`", fname))?;
                        let mut temps = Vec::new();
                        let mut avals = Vec::with_capacity(args.len());
                        avals.push(format!(
                            "i64 {}",
                            self.lower_expr_owned(&args[0].value, expected_ty)?
                        ));
                        for a in &args[1..] {
                            avals.push(format!(
                                "i64 {}",
                                self.lower_expr_borrowed(&a.value, &mut temps)?
                            ));
                        }
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @{}({})\n",
                            res,
                            csym,
                            avals.join(", ")
                        ));
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    if fname == "null_ptr" && args.is_empty() {
                        return Ok("0".to_string());
                    }
                    if fname == "ptr_is_null" && args.len() == 1 {
                        let mut temps = Vec::new();
                        let p = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        self.flush_temps(&temps);
                        let c = self.fresh_tmp();
                        let r = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = icmp eq i64 {}, 0\n  {} = zext i1 {} to i64\n",
                            c, p, r, c
                        ));
                        return Ok(r);
                    }
                    if (fname == "ptr_of" || fname == "mut_ptr_of") && args.len() == 1 {
                        let arg = &args[0].value;
                        let aty = self.infer_expr_type(arg);
                        let abase = resolve_alias_ty(&aty, self.alias_bases);
                        let is_buf = self.is_c_value_type(&abase)
                            || matches!(&abase, TypeExpr::Named { name, .. } if name == "Bytes" || name == "Text");
                        if is_buf {
                            if fname == "mut_ptr_of" {
                                if let ExprKind::Ident(vname) = &arg.kind {
                                    let cur = self.load_ident_val(vname)?;
                                    let uniq = self.fresh_tmp();
                                    let dp = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n",
                                        uniq, cur
                                    ));
                                    let vslot = self.var_ptrs[vname].clone();
                                    if self.inout_vars.contains(vname) {
                                        let ai = self.fresh_tmp();
                                        let ap = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                            ai, vslot, ap, ai, uniq, ap
                                        ));
                                    } else {
                                        self.code.push_str(&format!(
                                            "  store i64 {}, ptr {}\n",
                                            uniq, vslot
                                        ));
                                    }
                                    self.code.push_str(&format!(
                                        "  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n",
                                        dp, uniq
                                    ));
                                    return Ok(dp);
                                }
                            }
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(arg, &mut temps)?;
                            let dp = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = call i64 @__tn_cstruct_data_ptr(i64 {})\n",
                                dp, v
                            ));
                            self.flush_temps(&temps);
                            return Ok(dp);
                        }
                        if let ExprKind::Ident(vname) = &arg.kind {
                            if let Some(slot) = self.var_ptrs.get(vname).cloned() {
                                if self.inout_vars.contains(vname) {
                                    let p = self.fresh_tmp();
                                    self.code.push_str(&format!("  {} = load i64, ptr {}\n", p, slot));
                                    return Ok(p);
                                }
                                let p = self.fresh_tmp();
                                self.code.push_str(&format!("  {} = ptrtoint ptr {} to i64\n", p, slot));
                                return Ok(p);
                            }
                        }
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(arg, &mut temps)?;
                        self.slot_id += 1;
                        let s = format!("%ptrof_{}", self.slot_id);
                        self.entry_allocas.push_str(&format!("  {} = alloca i64\n", s));
                        let p = self.fresh_tmp();
                        self.code.push_str(&format!("  {} = ptrtoint ptr {} to i64\n", p, s));
                        self.emit_mem_store(&p, 0, &v, &abase, false)?;
                        self.flush_temps(&temps);
                        return Ok(p);
                    }
                    if fname == "__ptr_add" && args.len() == 3 {
                        let mut temps = Vec::new();
                        let p = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let n = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                        let sz = self.lower_expr_borrowed(&args[2].value, &mut temps)?;
                        self.flush_temps(&temps);
                        let off = self.fresh_tmp();
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = mul i64 {}, {}\n  {} = add i64 {}, {}\n",
                            off, n, sz, res, p, off
                        ));
                        return Ok(res);
                    }
                    if (fname == "ptr_read" || fname == "volatile_read") && args.len() == 1 {
                        let pty = self.infer_expr_type(&args[0].value);
                        let elem_ty = self
                            .ptr_pointee_ty(&pty)
                            .unwrap_or_else(|| named_ty("Int64"));
                        let mut temps = Vec::new();
                        let p = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let v = self.emit_mem_load(&p, 0, &elem_ty, fname == "volatile_read")?;
                        self.flush_temps(&temps);
                        return Ok(v);
                    }
                    if (fname == "ptr_write" || fname == "volatile_write") && args.len() == 2 {
                        let pty = self.infer_expr_type(&args[0].value);
                        let elem_ty = self
                            .ptr_pointee_ty(&pty)
                            .unwrap_or_else(|| self.infer_expr_type(&args[1].value));
                        let mut temps = Vec::new();
                        let p = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let v = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                        self.emit_mem_store(&p, 0, &v, &elem_ty, fname == "volatile_write")?;
                        self.flush_temps(&temps);
                        return Ok("0".into());
                    }
                    if let Some(ef) = self.extern_fns.get(fname).cloned() {
                        return self.emit_extern_c_call(&ef, args);
                    }
                    if fname == "push" && args.len() == 2 {
                        let xs_ty = self.infer_expr_type(&args[0].value);
                        if let Some((vec_ty, elem_bytes, _)) = self
                            .flat_simd_list_info(&xs_ty)
                            .or_else(|| expected_ty.and_then(|t| self.flat_simd_list_info(t)))
                            .or_else(|| self.simd_elem_info(&self.infer_expr_type(&args[1].value)))
                        {
                            let xs = self.lower_expr_owned(&args[0].value, expected_ty)?;
                            let mut temps = Vec::new();
                            let vv = self.lower_simd_vec_reg(&args[1].value, vec_ty, elem_bytes, &mut temps)?;
                            self.slot_id += 1;
                            let vslot = format!("%fpush_{}", self.slot_id);
                            self.entry_allocas
                                .push_str(&format!("  {} = alloca {}, align 8\n", vslot, vec_ty));
                            let res = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  store {} {}, ptr {}, align 8\n  {} = call i64 @tenon_flat_list_push(i64 {}, i64 {}, ptr {})\n",
                                vec_ty, vv, vslot, res, xs, elem_bytes, vslot
                            ));
                            self.flush_temps(&temps);
                            return Ok(res);
                        }
                        if self
                            .flat_scalar2_list_info(&xs_ty)
                            .or_else(|| expected_ty.and_then(|t| self.flat_scalar2_list_info(t)))
                            .or_else(|| self.scalar2_elem_info(&self.infer_expr_type(&args[1].value)))
                            .is_some()
                        {
                            let xs = self.lower_expr_owned(&args[0].value, expected_ty)?;
                            let mut temps = Vec::new();
                            let vv = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                            let v16 = self.fresh_tmp();
                            self.slot_id += 1;
                            let vslot = format!("%fs2push_{}", self.slot_id);
                            self.entry_allocas.push_str(&format!("  {} = alloca i16, align 2\n", vslot));
                            let res = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = trunc i64 {} to i16\n  store i16 {}, ptr {}, align 2\n  {} = call i64 @tenon_flat_list_push(i64 {}, i64 2, ptr {})\n",
                                v16, vv, v16, vslot, res, xs, vslot
                            ));
                            self.flush_temps(&temps);
                            return Ok(res);
                        }
                        if self
                            .flat_scalar4_list_info(&xs_ty)
                            .or_else(|| expected_ty.and_then(|t| self.flat_scalar4_list_info(t)))
                            .or_else(|| self.scalar4_elem_info(&self.infer_expr_type(&args[1].value)))
                            .is_some()
                        {
                            let xs = self.lower_expr_owned(&args[0].value, expected_ty)?;
                            let mut temps = Vec::new();
                            let vv = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                            let v32 = self.fresh_tmp();
                            self.slot_id += 1;
                            let vslot = format!("%fs4push_{}", self.slot_id);
                            self.entry_allocas.push_str(&format!("  {} = alloca i32, align 4\n", vslot));
                            let res = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = trunc i64 {} to i32\n  store i32 {}, ptr {}, align 4\n  {} = call i64 @tenon_flat_list_push(i64 {}, i64 4, ptr {})\n",
                                v32, vv, v32, vslot, res, xs, vslot
                            ));
                            self.flush_temps(&temps);
                            return Ok(res);
                        }
                        let elem_is_rc = self
                            .list_elem_ty(&xs_ty)
                            .is_some_and(|et| self.type_is_rc(&et))
                            || self.expr_produces_rc(&args[1].value);
                        let xs = self.lower_expr_owned(&args[0].value, expected_ty)?;
                        let val = if elem_is_rc {
                            self.code.push_str(&format!(
                                "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                                xs, RC_MASK_ALL as i64
                            ));
                            self.lower_expr_owned(&args[1].value, None)?
                        } else {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                            self.flush_temps(&temps);
                            v
                        };
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_list_push(i64 {}, i64 {})\n",
                            res, xs, val
                        ));
                        return Ok(res);
                    }
                    if fname == "list_filled" && args.len() == 2 {
                        if let Some((vec_ty, elem_bytes, _)) = expected_ty
                            .and_then(|t| self.flat_simd_list_info(t))
                            .or_else(|| self.simd_elem_info(&self.infer_expr_type(&args[1].value)))
                        {
                            let mut temps = Vec::new();
                            let count_v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                            let vv = self.lower_simd_vec_reg(&args[1].value, vec_ty, elem_bytes, &mut temps)?;
                            self.slot_id += 1;
                            let islot = format!("%ffill_{}", self.slot_id);
                            self.entry_allocas
                                .push_str(&format!("  {} = alloca {}, align 8\n", islot, vec_ty));
                            let res = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  store {} {}, ptr {}, align 8\n  {} = call i64 @tenon_flat_list_filled(i64 {}, i64 {}, ptr {})\n",
                                vec_ty, vv, islot, res, count_v, elem_bytes, islot
                            ));
                            let lf_pos = self.fresh_tmp();
                            let lf_exp = self.fresh_tmp();
                            let lf_nz = self.fresh_tmp();
                            let lf_obj = self.fresh_tmp();
                            let lf_lp = self.fresh_tmp();
                            let lf_len = self.fresh_tmp();
                            let lf_eq = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = icmp sgt i64 {}, 0\n  {} = select i1 {}, i64 {}, i64 0\n  {} = icmp ne i64 {}, 0\n  call void @llvm.assume(i1 {})\n  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 16\n  {} = load i64, ptr {}, align 8, !tbaa !8\n  {} = icmp eq i64 {}, {}\n  call void @llvm.assume(i1 {})\n",
                                lf_pos, count_v,
                                lf_exp, lf_pos, count_v,
                                lf_nz, res,
                                lf_nz,
                                lf_obj, res,
                                lf_lp, lf_obj,
                                lf_len, lf_lp,
                                lf_eq, lf_len, lf_exp,
                                lf_eq
                            ));
                            self.flush_temps(&temps);
                            return Ok(res);
                        }
                        if expected_ty
                            .and_then(|t| self.flat_scalar2_list_info(t))
                            .or_else(|| self.scalar2_elem_info(&self.infer_expr_type(&args[1].value)))
                            .is_some()
                        {
                            let mut temps = Vec::new();
                            let count_v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                            let vv = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                            let v16 = self.fresh_tmp();
                            self.slot_id += 1;
                            let islot = format!("%fs2fill_{}", self.slot_id);
                            self.entry_allocas.push_str(&format!("  {} = alloca i16, align 2\n", islot));
                            let res = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = trunc i64 {} to i16\n  store i16 {}, ptr {}, align 2\n  {} = call i64 @tenon_flat_list_filled(i64 {}, i64 2, ptr {})\n",
                                v16, vv, v16, islot, res, count_v, islot
                            ));
                            self.flush_temps(&temps);
                            return Ok(res);
                        }
                        if expected_ty
                            .and_then(|t| self.flat_scalar4_list_info(t))
                            .or_else(|| self.scalar4_elem_info(&self.infer_expr_type(&args[1].value)))
                            .is_some()
                        {
                            let mut temps = Vec::new();
                            let count_v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                            let vv = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                            let v32 = self.fresh_tmp();
                            self.slot_id += 1;
                            let islot = format!("%fs4fill_{}", self.slot_id);
                            self.entry_allocas.push_str(&format!("  {} = alloca i32, align 4\n", islot));
                            let res = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = trunc i64 {} to i32\n  store i32 {}, ptr {}, align 4\n  {} = call i64 @tenon_flat_list_filled(i64 {}, i64 4, ptr {})\n",
                                v32, vv, v32, islot, res, count_v, islot
                            ));
                            self.flush_temps(&temps);
                            return Ok(res);
                        }
                        let elem_is_rc = expected_ty
                            .and_then(|t| self.list_elem_ty(t))
                            .is_some_and(|et| self.type_is_rc(&et))
                            || self.type_is_rc(&self.infer_expr_type(&args[1].value))
                            || self.expr_produces_rc(&args[1].value);
                        let mut temps = Vec::new();
                        let count_v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let init_v = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_list_filled(i64 {}, i64 {}, i64 {})\n",
                            res,
                            count_v,
                            init_v,
                            if elem_is_rc { 1 } else { 0 }
                        ));
                        let lf_pos = self.fresh_tmp();
                        let lf_exp = self.fresh_tmp();
                        let lf_nz = self.fresh_tmp();
                        let lf_obj = self.fresh_tmp();
                        let lf_lp = self.fresh_tmp();
                        let lf_len = self.fresh_tmp();
                        let lf_eq = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = icmp sgt i64 {}, 0\n  {} = select i1 {}, i64 {}, i64 0\n  {} = icmp ne i64 {}, 0\n  call void @llvm.assume(i1 {})\n  {} = inttoptr i64 {} to ptr\n  {} = getelementptr inbounds i8, ptr {}, i64 16\n  {} = load i64, ptr {}, align 8, !tbaa !8\n  {} = icmp eq i64 {}, {}\n  call void @llvm.assume(i1 {})\n",
                            lf_pos, count_v,
                            lf_exp, lf_pos, count_v,
                            lf_nz, res,
                            lf_nz,
                            lf_obj, res,
                            lf_lp, lf_obj,
                            lf_len, lf_lp,
                            lf_eq, lf_len, lf_exp,
                            lf_eq
                        ));
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    if fname == "list_pop_shrink" && args.len() == 1 {
                        let xs = self.lower_expr_owned(&args[0].value, expected_ty)?;
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_list_pop_shrink(i64 {})\n",
                            res, xs
                        ));
                        return Ok(res);
                    }
                    if fname == "yield_val" && args.len() == 1 {
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        self.flush_temps(&temps);
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_async_yield_create(i64 {})\n",
                            res, v
                        ));
                        return Ok(res);
                    }
                    if fname == "block_on" && args.len() == 1 {
                        let f = self.lower_expr_owned(&args[0].value, None)?;
                        let res = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = call i64 @tenon_async_block_on(i64 {})\n",
                            res, f
                        ));
                        return Ok(res);
                    }
                    if let Some(&csym) = self.syscall_map.get(fname) {
                        let mut temps = Vec::new();
                        let mut avals = Vec::new();
                        for a in args {
                            avals.push(format!(
                                "i64 {}",
                                self.lower_expr_borrowed(&a.value, &mut temps)?
                            ));
                        }
                        let res = self.fresh_tmp();
                        let target_sym = match csym {
                            "tenon_byte_at" => "__tn_byte_at",
                            "tenon_buf_push_byte" => "__tn_buf_push_byte",
                            _ => csym,
                        };
                        self.code.push_str(&format!(
                            "  {} = call i64 @{}({})\n",
                            res,
                            target_sym,
                            avals.join(", ")
                        ));
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    if let Some(&tag) = self.enum_tags.get(fname) {
                        if !self.fn_metas.contains_key(fname) {
                            let mut ptys = self
                                .enum_variants
                                .get(fname)
                                .map(|vl| vl.payload_tys.clone())
                                .unwrap_or_default();
                            if ptys.is_empty() {
                                if let Some(exp) = expected_ty {
                                    let resolved = resolve_alias_ty(exp, self.alias_bases);
                                    if let TypeExpr::Named { name: tn, args: targs, .. } = resolved {
                                        if fname == "Some" && tn == "Option" && !targs.is_empty() {
                                            ptys.push(targs[0].clone());
                                        } else if fname == "Ok" && tn == "Result" && !targs.is_empty() {
                                            ptys.push(targs[0].clone());
                                        } else if fname == "Err" && tn == "Result" && targs.len() >= 2 {
                                            ptys.push(targs[1].clone());
                                        }
                                    }
                                }
                            }
                            return self.lower_enum_ctor(tag, &ptys, args);
                        }
                    }

                    let meta = self.fn_metas.get(fname).cloned();
                    let mut temps = Vec::new();
                    let mut avals = Vec::new();
                    let mut writebacks: Vec<(String, String)> = Vec::new();
                    let mut field_writebacks: Vec<(String, usize, String)> = Vec::new();
                    for (idx, a) in args.iter().enumerate() {
                        let is_inout = meta
                            .as_ref()
                            .and_then(|m| m.params_inout.get(idx).copied())
                            .unwrap_or(false);
                        let pty = meta.as_ref().and_then(|m| m.param_tys.get(idx));
                        if is_inout {
                            match &a.value.kind {
                                ExprKind::Ident(vn) => {
                                    let cur = self.load_ident_val(vn)?;
                                    self.slot_id += 1;
                                    let tmp_slot = format!("%io_{}_{}", self.slot_id, idx);
                                    self.entry_allocas
                                        .push_str(&format!("  {} = alloca i64\n", tmp_slot));
                                    let addr_i = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  store i64 {}, ptr {}\n  {} = ptrtoint ptr {} to i64\n",
                                        cur, tmp_slot, addr_i, tmp_slot
                                    ));
                                    avals.push(format!("i64 {}", addr_i));
                                    writebacks.push((vn.clone(), tmp_slot));
                                }
                                ExprKind::Field { target: obj, field }
                                    if matches!(&obj.kind, ExprKind::Ident(_)) =>
                                {
                                    let ExprKind::Ident(vn) = &obj.kind else {
                                        unreachable!()
                                    };
                                    let (fidx, _) = self.resolve_field_layout(obj, field)?;
                                    let cur_obj = self.load_ident_val(vn)?;
                                    let uniq_obj = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = call i64 @__tn_rc_ensure_unique(i64 {})\n",
                                        uniq_obj, cur_obj
                                    ));
                                    let vslot = self.var_ptrs[vn].clone();
                                    if self.inout_vars.contains(vn) {
                                        let ai = self.fresh_tmp();
                                        let ap = self.fresh_tmp();
                                        self.code.push_str(&format!(
                                            "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                            ai, vslot, ap, ai, uniq_obj, ap
                                        ));
                                    } else {
                                        self.code.push_str(&format!(
                                            "  store i64 {}, ptr {}\n",
                                            uniq_obj, vslot
                                        ));
                                    }
                                    let cur_fv = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  {} = call i64 @__tn_slot_get(i64 {}, i64 {})\n",
                                        cur_fv, uniq_obj, fidx
                                    ));
                                    self.slot_id += 1;
                                    let tmp_slot = format!("%iof_{}_{}", self.slot_id, idx);
                                    self.entry_allocas
                                        .push_str(&format!("  {} = alloca i64\n", tmp_slot));
                                    let addr_i = self.fresh_tmp();
                                    self.code.push_str(&format!(
                                        "  store i64 {}, ptr {}\n  {} = ptrtoint ptr {} to i64\n",
                                        cur_fv, tmp_slot, addr_i, tmp_slot
                                    ));
                                    avals.push(format!("i64 {}", addr_i));
                                    field_writebacks.push((uniq_obj, fidx, tmp_slot));
                                }
                                _ => {
                                    return Err("inout arg must be variable or record field".into());
                                }
                            }
                        } else if matches!(&a.value.kind, ExprKind::List(es) if es.is_empty())
                            && pty.is_some()
                        {
                            let lv = self.lower_expr_owned(&a.value, pty)?;
                            temps.push(lv.clone());
                            avals.push(format!("i64 {}", lv));
                        } else {
                            let av = self.lower_expr_borrowed(&a.value, &mut temps)?;
                            avals.push(format!("i64 {}", av));
                        }
                    }
                    let res = self.fresh_tmp();
                    self.code.push_str(&format!(
                        "  {} = call i64 @tenon_fn_{}({})\n",
                        res,
                        sanitize(fname),
                        avals.join(", ")
                    ));
                    self.flush_temps(&temps);
                    for (vn, tmp_slot) in writebacks {
                        let updated = self.fresh_tmp();
                        let vslot = self.var_ptrs[&vn].clone();
                        self.code
                            .push_str(&format!("  {} = load i64, ptr {}\n", updated, tmp_slot));
                        if self.inout_vars.contains(&vn) {
                            let ai = self.fresh_tmp();
                            let ap = self.fresh_tmp();
                            self.code.push_str(&format!(
                                "  {} = load i64, ptr {}\n  {} = inttoptr i64 {} to ptr\n  store i64 {}, ptr {}\n",
                                ai, vslot, ap, ai, updated, ap
                            ));
                        } else {
                            self.code
                                .push_str(&format!("  store i64 {}, ptr {}\n", updated, vslot));
                        }
                    }
                    for (uniq_obj, fidx, tmp_slot) in field_writebacks {
                        let updated = self.fresh_tmp();
                        self.code.push_str(&format!(
                            "  {} = load i64, ptr {}\n  call void @__tn_slot_set_prim(i64 {}, i64 {}, i64 {})\n",
                            updated, tmp_slot, uniq_obj, fidx, updated
                        ));
                    }
                    Ok(res)
                }
                _ => Err("unsupported call target in LLVM codegen".into()),
            },
            _ => Ok("0".into()),
        }
    }

    fn lower_enum_ctor(
        &mut self,
        tag: i64,
        payload_tys: &[TypeExpr],
        args: &[CallArg],
    ) -> Result<String, String> {
        if args.is_empty() {
            return Ok(self.emit_unit_enum_const(tag));
        }
        let o = self.fresh_tmp();
        self.code.push_str(&format!(
            "  {} = call i64 @tenon_rc_alloc(i64 {}, i64 {})\n",
            o,
            tag,
            args.len()
        ));
        let mut rc_mask = 0u64;
        for (i, a) in args.iter().enumerate() {
            let pty = payload_tys.get(i);
            let is_rc = pty.is_some_and(|t| self.type_is_rc(t)) || self.expr_produces_rc(&a.value);
            let av = if is_rc {
                if i < 64 {
                    rc_mask |= 1u64 << i;
                }
                self.lower_expr_owned(&a.value, pty)?
            } else {
                let mut temps = Vec::new();
                let v = self.lower_expr_borrowed(&a.value, &mut temps)?;
                self.flush_temps(&temps);
                v
            };
            self.code.push_str(&format!(
                "  call void @__tn_slot_set_prim(i64 {}, i64 {}, i64 {})\n",
                o, i, av
            ));
        }
        if rc_mask != 0 {
            self.code.push_str(&format!(
                "  call void @__tn_rc_set_rc_mask(i64 {}, i64 {})\n",
                o, rc_mask as i64
            ));
        }
        Ok(o)
    }
}

fn collect_always_unique_locals(body: &Block) -> BTreeSet<String> {
    let mut candidates: Vec<(usize, String)> = Vec::new();
    for (idx, stmt) in body.stmts.iter().enumerate() {
        if let Stmt::Let { name, value, .. } = stmt {
            let is_fresh_unique = match &value.kind {
                ExprKind::List(_) | ExprKind::RecordLit { .. } => true,
                ExprKind::Call { callee, .. } => matches!(&callee.kind, ExprKind::Ident(f) if f == "list_filled"),
                _ => false,
            };
            if is_fresh_unique {
                candidates.push((idx, name.clone()));
            }
        }
    }
    let mut result = BTreeSet::new();
    for (decl_idx, name) in candidates {
        let mut ok = true;
        for stmt in &body.stmts[(decl_idx + 1)..] {
            if !stmt_keeps_unique(stmt, &name) {
                ok = false;
                break;
            }
        }
        if ok {
            if let Some(tail) = &body.tail {
                if !expr_keeps_unique(tail, &name) {
                    ok = false;
                }
            }
        }
        if ok {
            result.insert(name);
        }
    }
    result
}

fn block_keeps_unique(blk: &Block, name: &str) -> bool {
    for stmt in &blk.stmts {
        if !stmt_keeps_unique(stmt, name) {
            return false;
        }
    }
    blk.tail.as_deref().is_none_or(|t| expr_keeps_unique(t, name))
}

fn stmt_keeps_unique(stmt: &Stmt, name: &str) -> bool {
    match stmt {
        Stmt::Let { name: ln, value, .. } => ln != name && expr_keeps_unique(value, name),
        Stmt::Assign { target, op, value, .. } => match &target.kind {
            ExprKind::Ident(vn) if vn == name => {
                if *op != AssignOp::Set {
                    return false;
                }
                match &value.kind {
                    ExprKind::Call { callee, args }
                        if matches!(&callee.kind, ExprKind::Ident(f) if f == "push")
                            && args.len() == 2
                            && matches!(&args[0].value.kind, ExprKind::Ident(s) if s == name) =>
                    {
                        expr_keeps_unique(&args[1].value, name)
                    }
                    ExprKind::Call { callee, args }
                        if matches!(&callee.kind, ExprKind::Ident(f) if f == "list_pop_shrink")
                            && args.len() == 1
                            && matches!(&args[0].value.kind, ExprKind::Ident(s) if s == name) =>
                    {
                        true
                    }
                    _ => false,
                }
            }
            ExprKind::Index { target: lt, index } => {
                let t_ok = matches!(&lt.kind, ExprKind::Ident(s) if s == name)
                    || expr_keeps_unique(lt, name);
                t_ok && expr_keeps_unique(index, name) && expr_keeps_unique(value, name)
            }
            ExprKind::Field { target: ot, .. } => {
                let t_ok = matches!(&ot.kind, ExprKind::Ident(s) if s == name)
                    || expr_keeps_unique(ot, name);
                t_ok && expr_keeps_unique(value, name)
            }
            _ => expr_keeps_unique(target, name) && expr_keeps_unique(value, name),
        },
        Stmt::Mutate { body, .. } => block_keeps_unique(body, name),
        Stmt::Return { value, .. } => value.as_ref().is_none_or(|e| expr_keeps_unique(e, name)),
        Stmt::Assert { cond, .. } => expr_keeps_unique(cond, name),
        Stmt::While { cond, body, .. } => {
            expr_keeps_unique(cond, name) && block_keeps_unique(body, name)
        }
        Stmt::Expr { expr, .. } => expr_keeps_unique(expr, name),
        Stmt::Break { .. } | Stmt::Continue { .. } => true,
        Stmt::Defer { body, .. } | Stmt::Ghost { body, .. } => block_keeps_unique(body, name),
        Stmt::LetTuple { .. } | Stmt::For { .. } => false,
    }
}

fn pattern_mentions_name(pat: &Pattern, name: &str) -> bool {
    match pat {
        Pattern::Ident(s, _) => s == name,
        Pattern::Variant { binding: Some(b), .. } => b == name,
        Pattern::VariantMulti { patterns, .. } | Pattern::Tuple(patterns, _) => {
            patterns.iter().any(|p| pattern_mentions_name(p, name))
        }
        _ => false,
    }
}

fn expr_keeps_unique(expr: &Expr, name: &str) -> bool {
    match &expr.kind {
        ExprKind::Ident(s) => s != name,
        ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Str(_) => true,
        ExprKind::Index { target, index } => {
            let t_ok = matches!(&target.kind, ExprKind::Ident(s) if s == name)
                || expr_keeps_unique(target, name);
            t_ok && expr_keeps_unique(index, name)
        }
        ExprKind::Field { target, .. } | ExprKind::TupleIndex { target, .. } => {
            matches!(&target.kind, ExprKind::Ident(s) if s == name)
                || expr_keeps_unique(target, name)
        }
        ExprKind::Call { callee, args } => {
            if matches!(&callee.kind, ExprKind::Ident(f) if f == "len")
                && args.len() == 1
                && matches!(&args[0].value.kind, ExprKind::Ident(s) if s == name)
            {
                return true;
            }
            expr_keeps_unique(callee, name)
                && args.iter().all(|a| expr_keeps_unique(&a.value, name))
        }
        ExprKind::Unary { expr: inner, .. } => expr_keeps_unique(inner, name),
        ExprKind::Binary { left, right, .. } => {
            expr_keeps_unique(left, name) && expr_keeps_unique(right, name)
        }
        ExprKind::RecordLit { fields, .. } => {
            fields.iter().all(|(_, e)| expr_keeps_unique(e, name))
        }
        ExprKind::List(es) | ExprKind::Tuple(es) => es.iter().all(|e| expr_keeps_unique(e, name)),
        ExprKind::If {
            cond,
            then_block,
            else_branch,
        } => {
            expr_keeps_unique(cond, name)
                && block_keeps_unique(then_block, name)
                && else_branch
                    .as_deref()
                    .is_none_or(|e| expr_keeps_unique(e, name))
        }
        ExprKind::Match { scrutinee, arms } => {
            if !expr_keeps_unique(scrutinee, name) {
                return false;
            }
            arms.iter().all(|a| {
                let shadows = a.patterns.iter().any(|p| pattern_mentions_name(p, name));
                !shadows
                    && a.guard.as_ref().is_none_or(|g| expr_keeps_unique(g, name))
                    && expr_keeps_unique(&a.body, name)
            })
        }
        ExprKind::Block(blk) | ExprKind::Unsafe(blk) => block_keeps_unique(blk, name),
        ExprKind::Cast { expr: e, .. }
        | ExprKind::TryCast { expr: e, .. }
        | ExprKind::ArrayRepeat { value: e, .. }
        | ExprKind::Await(e) => expr_keeps_unique(e, name),
        _ => false,
    }
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "llvm_aot_tests.rs"]
mod tests;
