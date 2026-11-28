//! Declarative Cross-Compilation Target Specification Engine (inspired by `llgo/targets`).
//!
//! Supports composable JSON target files (`targets/*.json`) with recursive `"inherits"` merging
//! across Desktop/Server (macOS/Linux/Windows x86_64/arm64/riscv64), Mobile (iOS/Android),
//! WebAssembly (wasm32-wasi / freestanding), and Bare-Metal Embedded MCUs (ARM Cortex-M0/M4/M7,
//! RP2040, STM32F4, RISC-V ESP32-C3, AVR).

use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct RawTargetSpec {
    #[serde(default)]
    pub inherits: Vec<String>,
    pub llvm_target: Option<String>,
    pub cpu: Option<String>,
    pub features: Option<String>,
    pub os: Option<String>,
    pub arch: Option<String>,
    pub pointer_bits: Option<u32>,
    pub freestanding: Option<bool>,
    pub linker: Option<String>,
    pub binary_format: Option<String>,
    pub flash_method: Option<String>,
    pub linkerscript: Option<String>,
    #[serde(default)]
    pub build_tags: Vec<String>,
    #[serde(default)]
    pub cflags: Vec<String>,
    #[serde(default)]
    pub ldflags: Vec<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ResolvedTarget {
    pub name: String,
    pub llvm_target: String,
    pub cpu: Option<String>,
    pub features: Option<String>,
    pub os: String,
    pub arch: String,
    pub pointer_bits: u32,
    pub freestanding: bool,
    pub linker: Option<String>,
    pub binary_format: String,
    pub flash_method: Option<String>,
    pub linkerscript: Option<String>,
    pub build_tags: Vec<String>,
    pub cflags: Vec<String>,
    pub ldflags: Vec<String>,
    pub description: String,
}

impl ResolvedTarget {
    pub fn is_wasm(&self) -> bool {
        self.arch == "wasm32" || self.llvm_target.starts_with("wasm32")
    }

    pub fn is_apple(&self) -> bool {
        self.llvm_target.contains("apple") || matches!(self.os.as_str(), "macos" | "ios" | "darwin")
    }

    pub fn is_windows(&self) -> bool {
        self.os == "windows" || self.llvm_target.contains("windows")
    }

    pub fn is_host(&self) -> bool {
        let host_os = std::env::consts::OS;
        let host_arch = std::env::consts::ARCH;
        let same_os = (host_os == "macos" && (self.os == "macos" || self.os == "darwin"))
            || (host_os == "linux" && self.os == "linux")
            || (host_os == "windows" && self.os == "windows");
        let same_arch = (host_arch == "aarch64" && (self.arch == "arm64" || self.arch == "aarch64"))
            || (host_arch == "x86_64" && self.arch == "x86_64")
            || (host_arch == "riscv64" && self.arch == "riscv64");
        same_os && same_arch && !self.freestanding
    }

    /// Clang codegen flags for compiling LLVM IR / C sources to this target.
    pub fn clang_compile_args(&self) -> Vec<String> {
        let mut args = vec![format!("--target={}", self.llvm_target)];
        if let Some(cpu) = &self.cpu {
            if !cpu.is_empty() && cpu != "generic" && !self.arch.starts_with("riscv") && self.arch != "x86_64" {
                args.push(format!("-mcpu={}", cpu));
            } else if self.arch == "x86_64" && !cpu.is_empty() && cpu != "generic" {
                args.push(format!("-march={}", cpu));
            }
        }
        if self.freestanding {
            args.push("-ffreestanding".into());
            args.push("-DTENON_FREESTANDING=1".into());
        }
        for f in &self.cflags {
            if !args.contains(f) {
                args.push(f.clone());
            }
        }
        args
    }
}

pub struct TargetSummary {
    pub name: String,
    pub llvm_target: String,
    pub category: &'static str,
    pub freestanding: bool,
    pub description: String,
}

const BUILTIN_TARGETS: &[(&str, &'static str, &str)] = &[
    // Desktop / Server
    ("macos-arm64", "desktop", include_str!("../../../targets/macos-arm64.json")),
    ("macos-x86_64", "desktop", include_str!("../../../targets/macos-x86_64.json")),
    ("linux-x86_64", "desktop", include_str!("../../../targets/linux-x86_64.json")),
    ("linux-x86_64-cuda", "gpu-server", include_str!("../../../targets/linux-x86_64-cuda.json")),
    ("linux-arm64", "desktop", include_str!("../../../targets/linux-arm64.json")),
    ("linux-riscv64", "desktop", include_str!("../../../targets/linux-riscv64.json")),
    ("windows-x86_64", "desktop", include_str!("../../../targets/windows-x86_64.json")),
    ("windows-arm64", "desktop", include_str!("../../../targets/windows-arm64.json")),
    // Mobile
    ("ios-arm64", "mobile", include_str!("../../../targets/ios-arm64.json")),
    ("ios-sim-arm64", "mobile", include_str!("../../../targets/ios-sim-arm64.json")),
    ("android-arm64", "mobile", include_str!("../../../targets/android-arm64.json")),
    ("android-x86_64", "mobile", include_str!("../../../targets/android-x86_64.json")),
    // WebAssembly
    ("wasm32-wasi", "wasm", include_str!("../../../targets/wasm32-wasi.json")),
    ("wasm32-freestanding", "wasm", include_str!("../../../targets/wasm32-freestanding.json")),
    // Embedded / Bare-Metal (inspired by llgo/targets)
    ("cortex-m", "embedded-base", include_str!("../../../targets/cortex-m.json")),
    ("cortex-m0", "embedded", include_str!("../../../targets/cortex-m0.json")),
    ("cortex-m4", "embedded", include_str!("../../../targets/cortex-m4.json")),
    ("cortex-m7", "embedded", include_str!("../../../targets/cortex-m7.json")),
    ("rp2040", "embedded", include_str!("../../../targets/rp2040.json")),
    ("stm32f4", "embedded", include_str!("../../../targets/stm32f4.json")),
    ("riscv32", "embedded", include_str!("../../../targets/riscv32.json")),
    ("esp32c3", "embedded", include_str!("../../../targets/esp32c3.json")),
    ("avr-arduino", "embedded", include_str!("../../../targets/avr-arduino.json")),
];

fn load_raw_spec(name_or_path: &str, extra_dir: Option<&Path>) -> Option<RawTargetSpec> {
    // 1. Direct JSON file path
    let p = Path::new(name_or_path);
    if p.extension().and_then(|e| e.to_str()) == Some("json") && p.exists() {
        let text = fs::read_to_string(p).ok()?;
        return serde_json::from_str(&text).ok();
    }
    // 2. Directory lookup (`extra_dir/<name>.json`, `./targets/<name>.json`, `$TENON_TARGETS_DIR/<name>.json`)
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = extra_dir {
        dirs.push(d.to_path_buf());
    }
    if let Ok(env_d) = std::env::var("TENON_TARGETS_DIR") {
        dirs.push(PathBuf::from(env_d));
    }
    dirs.push(PathBuf::from("targets"));
    for dir in dirs {
        let cand = dir.join(format!("{}.json", name_or_path));
        if cand.exists() {
            if let Ok(text) = fs::read_to_string(&cand) {
                if let Ok(raw) = serde_json::from_str::<RawTargetSpec>(&text) {
                    return Some(raw);
                }
            }
        }
    }
    // 3. Embedded built-in target specs
    for &(bname, _, json_src) in BUILTIN_TARGETS {
        if bname == name_or_path {
            return serde_json::from_str(json_src).ok();
        }
    }
    None
}

fn merge_spec_recursive(
    name: &str,
    extra_dir: Option<&Path>,
    visited: &mut BTreeSet<String>,
) -> Result<RawTargetSpec, String> {
    if !visited.insert(name.to_string()) {
        return Err(format!("cyclic target inheritance involving `{}`", name));
    }
    let child = load_raw_spec(name, extra_dir)
        .ok_or_else(|| format!("unknown target spec `{}`", name))?;
    let mut merged = RawTargetSpec::default();
    for parent_name in &child.inherits {
        let parent = merge_spec_recursive(parent_name, extra_dir, visited)?;
        apply_overlay(&mut merged, parent);
    }
    apply_overlay(&mut merged, child);
    Ok(merged)
}

fn apply_overlay(dst: &mut RawTargetSpec, src: RawTargetSpec) {
    if src.llvm_target.is_some() { dst.llvm_target = src.llvm_target; }
    if src.cpu.is_some() { dst.cpu = src.cpu; }
    if src.features.is_some() { dst.features = src.features; }
    if src.os.is_some() { dst.os = src.os; }
    if src.arch.is_some() { dst.arch = src.arch; }
    if src.pointer_bits.is_some() { dst.pointer_bits = src.pointer_bits; }
    if src.freestanding.is_some() { dst.freestanding = src.freestanding; }
    if src.linker.is_some() { dst.linker = src.linker; }
    if src.binary_format.is_some() { dst.binary_format = src.binary_format; }
    if src.flash_method.is_some() { dst.flash_method = src.flash_method; }
    if src.linkerscript.is_some() { dst.linkerscript = src.linkerscript; }
    if src.description.is_some() { dst.description = src.description; }
    for t in src.build_tags {
        if !dst.build_tags.contains(&t) { dst.build_tags.push(t); }
    }
    for f in src.cflags {
        if !dst.cflags.contains(&f) { dst.cflags.push(f); }
    }
    for f in src.ldflags {
        if !dst.ldflags.contains(&f) { dst.ldflags.push(f); }
    }
}

/// Resolves a target name (e.g. `linux-x86_64`, `esp32c3`, `rp2040`, `ios-arm64`), a `.json` file path,
/// or a raw LLVM target triple (e.g. `aarch64-unknown-linux-musl`).
pub fn resolve_target(spec: &str) -> Result<ResolvedTarget, String> {
    let norm = match spec {
        "wasm32" | "wasi" | "wasm32-wasip2" => "wasm32-wasi",
        "darwin-arm64" => "macos-arm64",
        "darwin-x86_64" => "macos-x86_64",
        "linux-aarch64" => "linux-arm64",
        "android-aarch64" => "android-arm64",
        "windows-aarch64" => "windows-arm64",
        other => other,
    };
    let extra_dir = Path::new(norm).parent().filter(|p| !p.as_os_str().is_empty());
    let mut visited = BTreeSet::new();
    if let Ok(raw) = merge_spec_recursive(norm, extra_dir, &mut visited) {
        let llvm_target = raw
            .llvm_target
            .ok_or_else(|| format!("target `{}` missing `llvm-target`", norm))?;
        let os = raw.os.unwrap_or_else(|| infer_os(&llvm_target));
        let arch = raw.arch.unwrap_or_else(|| infer_arch(&llvm_target));
        let freestanding = raw.freestanding.unwrap_or(os == "none" || os == "baremetal");
        let pointer_bits = raw.pointer_bits.unwrap_or_else(|| infer_ptr_bits(&arch));
        return Ok(ResolvedTarget {
            name: norm.to_string(),
            llvm_target,
            cpu: raw.cpu,
            features: raw.features,
            os,
            arch,
            pointer_bits,
            freestanding,
            linker: raw.linker,
            binary_format: raw.binary_format.unwrap_or_else(|| "native".into()),
            flash_method: raw.flash_method,
            linkerscript: raw.linkerscript,
            build_tags: raw.build_tags,
            cflags: raw.cflags,
            ldflags: raw.ldflags,
            description: raw.description.unwrap_or_default(),
        });
    }

    // Fallback: treat any hyphenated string as a raw LLVM target triple
    if spec.contains('-') {
        let os = infer_os(spec);
        let arch = infer_arch(spec);
        let freestanding = os == "none" || os == "baremetal";
        return Ok(ResolvedTarget {
            name: spec.to_string(),
            llvm_target: spec.to_string(),
            cpu: None,
            features: None,
            os,
            arch: arch.clone(),
            pointer_bits: infer_ptr_bits(&arch),
            freestanding,
            linker: None,
            binary_format: "native".into(),
            flash_method: None,
            linkerscript: None,
            build_tags: Vec::new(),
            cflags: Vec::new(),
            ldflags: Vec::new(),
            description: "Raw LLVM target triple".into(),
        });
    }

    Err(format!(
        "unknown target `{}` (run `tenon targets` to list built-in targets or pass a `.json` / LLVM triple)",
        spec
    ))
}

fn infer_os(triple: &str) -> String {
    if triple.contains("ios") { "ios".into() }
    else if triple.contains("darwin") || triple.contains("macos") { "macos".into() }
    else if triple.contains("android") { "android".into() }
    else if triple.contains("linux") { "linux".into() }
    else if triple.contains("windows") { "windows".into() }
    else if triple.contains("wasi") { "wasi".into() }
    else if triple.contains("none") || triple.contains("eabi") || triple.contains("unknown-unknown") { "baremetal".into() }
    else { "unknown".into() }
}

fn infer_arch(triple: &str) -> String {
    let first = triple.split('-').next().unwrap_or(triple);
    if first.starts_with("thumb") || first.starts_with("arm") && !first.starts_with("arm64") { "arm".into() }
    else if first == "aarch64" || first == "arm64" { "arm64".into() }
    else if first == "x86_64" || first == "amd64" { "x86_64".into() }
    else if first.starts_with("riscv32") { "riscv32".into() }
    else if first.starts_with("riscv64") { "riscv64".into() }
    else if first.starts_with("wasm32") { "wasm32".into() }
    else if first.starts_with("avr") { "avr".into() }
    else { first.into() }
}

fn infer_ptr_bits(arch: &str) -> u32 {
    match arch {
        "arm" | "riscv32" | "wasm32" | "x86" => 32,
        "avr" => 16,
        _ => 64,
    }
}

pub fn list_builtin_targets() -> Vec<TargetSummary> {
    let mut out = Vec::new();
    for &(name, cat, _) in BUILTIN_TARGETS {
        if cat == "embedded-base" {
            continue;
        }
        if let Ok(rt) = resolve_target(name) {
            out.push(TargetSummary {
                name: rt.name,
                llvm_target: rt.llvm_target,
                category: cat,
                freestanding: rt.freestanding,
                description: rt.description,
            });
        }
    }
    out
}

/// Zero-OS, zero-libc Freestanding Perceus Runtime for Bare-Metal MCUs
/// (ARM Cortex-M0/M4/M7, RP2040, STM32F4, RISC-V ESP32-C3, AVR, Wasm32 freestanding).
pub const TENON_AOT_FREESTANDING_RT_C: &str = r#"
/* ===== Tenon Freestanding Bare-Metal Perceus Runtime (Zero libc / Zero OS!) ===== */
#include <stddef.h>
#include <stdint.h>

#ifndef TENON_EMBEDDED_HEAP_BYTES
#define TENON_EMBEDDED_HEAP_BYTES 32768
#endif

typedef struct {
    int64_t rc;
    int64_t tag;
    int64_t len;
    int64_t cap;
    uint64_t rc_mask;
    int64_t fields[];
} TenonObj;

static uint8_t g_tn_static_heap[TENON_EMBEDDED_HEAP_BYTES] __attribute__((aligned(8)));
static size_t g_tn_heap_off = 0;
static int64_t g_tn_live = 0;

static void* tn_fs_alloc(size_t bytes) {
    size_t aligned = (bytes + 7u) & ~(size_t)7u;
    if (g_tn_heap_off + aligned > sizeof(g_tn_static_heap)) {
        for (;;) {} /* OOM trap on bare-metal MCU */
    }
    void* p = &g_tn_static_heap[g_tn_heap_off];
    g_tn_heap_off += aligned;
    return p;
}

static void tn_fs_zero(void* dst, size_t n) {
    uint8_t* p = (uint8_t*)dst;
    for (size_t i = 0; i < n; i++) p[i] = 0;
}

static void tn_fs_copy(void* dst, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    for (size_t i = 0; i < n; i++) d[i] = s[i];
}

void tenon_rt_assert_fail(int64_t line) { (void)line; for (;;) {} }
void tenon_rt_overflow_trap(int64_t line) { (void)line; for (;;) {} }
int64_t tenon_rt_print_int(int64_t v) { return v; }
int64_t tenon_arena_enter(void) { return (int64_t)g_tn_heap_off; }
void tenon_arena_exit(int64_t mark) {
    if (mark >= 0 && (size_t)mark <= g_tn_heap_off) g_tn_heap_off = (size_t)mark;
}

int64_t tenon_rc_alloc(int64_t tag, int64_t len) {
    int64_t lu = len < 0 ? 0 : len;
    int64_t cap = lu < 1 ? 1 : lu;
    TenonObj* obj = (TenonObj*)tn_fs_alloc(sizeof(TenonObj) + (size_t)cap * sizeof(int64_t));
    obj->rc = 1;
    obj->tag = tag;
    obj->len = lu;
    obj->cap = cap;
    obj->rc_mask = 0;
    tn_fs_zero(obj->fields, (size_t)cap * sizeof(int64_t));
    g_tn_live++;
    return (int64_t)(uintptr_t)obj;
}

void tenon_rc_set_rc_mask(int64_t ptr, int64_t mask) {
    if (ptr != 0) ((TenonObj*)(uintptr_t)ptr)->rc_mask = (uint64_t)mask;
}

int64_t tenon_rc_inc(int64_t ptr) {
    if (ptr == 0) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    if (obj->rc > 0) obj->rc++;
    return ptr;
}

void tenon_rc_dec(int64_t ptr);
void tenon_rc_free_obj(int64_t ptr) {
    if (ptr == 0) return;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    uint64_t mask = obj->rc_mask;
    if (mask == UINT64_MAX) {
        for (int64_t i = 0; i < obj->len; i++) {
            if (obj->fields[i] != 0) tenon_rc_dec(obj->fields[i]);
        }
    } else if (mask != 0) {
        int64_t n = obj->len < 64 ? obj->len : 64;
        for (int64_t i = 0; i < n; i++) {
            if ((mask & (1ULL << i)) && obj->fields[i] != 0) tenon_rc_dec(obj->fields[i]);
        }
    }
    g_tn_live--;
}

void tenon_rc_dec(int64_t ptr) {
    if (ptr == 0) return;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    if (obj->rc <= 0) return;
    obj->rc--;
    if (obj->rc == 0) tenon_rc_free_obj(ptr);
}

int64_t tenon_rc_ensure_unique(int64_t ptr) {
    if (ptr == 0) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    if (obj->rc <= 1) return ptr;
    int64_t fp = tenon_rc_alloc(obj->tag, obj->cap);
    TenonObj* fresh = (TenonObj*)(uintptr_t)fp;
    fresh->len = obj->len;
    fresh->rc_mask = obj->rc_mask;
    tn_fs_copy(fresh->fields, obj->fields, (size_t)obj->len * sizeof(int64_t));
    tenon_rc_dec(ptr);
    return fp;
}

int64_t tenon_rc_get_field(int64_t ptr, int64_t idx) {
    if (ptr == 0) return 0;
    return ((TenonObj*)(uintptr_t)ptr)->fields[idx];
}
void tenon_rc_set_field(int64_t ptr, int64_t idx, int64_t val) {
    if (ptr != 0) ((TenonObj*)(uintptr_t)ptr)->fields[idx] = val;
}
void tenon_rc_set_field_raw(int64_t ptr, int64_t idx, int64_t val) {
    if (ptr != 0) ((TenonObj*)(uintptr_t)ptr)->fields[idx] = val;
}
int64_t tenon_rc_get_tag(int64_t ptr) { return ptr ? ((TenonObj*)(uintptr_t)ptr)->tag : 0; }
int64_t tenon_rc_get_len(int64_t ptr) { return ptr ? ((TenonObj*)(uintptr_t)ptr)->len : 0; }

int64_t tenon_text_new(int64_t data_ptr, int64_t byte_len) {
    int64_t lu = byte_len < 0 ? 0 : byte_len;
    int64_t words = (lu + 8) / 8;
    int64_t p = tenon_rc_alloc(1, words);
    TenonObj* obj = (TenonObj*)(uintptr_t)p;
    obj->len = lu;
    if (data_ptr != 0 && lu > 0) tn_fs_copy(obj->fields, (const void*)(uintptr_t)data_ptr, (size_t)lu);
    return p;
}
int64_t tenon_text_eq(int64_t a, int64_t b) {
    if (a == b) return 1;
    if (!a || !b) return 0;
    TenonObj* oa = (TenonObj*)(uintptr_t)a;
    TenonObj* ob = (TenonObj*)(uintptr_t)b;
    if (oa->len != ob->len) return 0;
    const uint8_t* pa = (const uint8_t*)oa->fields;
    const uint8_t* pb = (const uint8_t*)ob->fields;
    for (int64_t i = 0; i < oa->len; i++) if (pa[i] != pb[i]) return 0;
    return 1;
}
int64_t tenon_list_filled(int64_t len, int64_t init_val, int64_t elem_is_rc) {
    int64_t p = tenon_rc_alloc(4, len);
    TenonObj* obj = (TenonObj*)(uintptr_t)p;
    if (elem_is_rc) obj->rc_mask = UINT64_MAX;
    for (int64_t i = 0; i < len; i++) obj->fields[i] = init_val;
    return p;
}
int64_t tenon_list_push(int64_t ptr, int64_t val) {
    if (ptr == 0) { int64_t f = tenon_rc_alloc(4, 1); ((TenonObj*)(uintptr_t)f)->fields[0] = val; return f; }
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    if (obj->rc <= 1 && obj->len < obj->cap) { obj->fields[obj->len++] = val; return ptr; }
    int64_t nc = obj->cap < 4 ? 4 : obj->cap * 2;
    int64_t fp = tenon_rc_alloc(4, nc);
    TenonObj* fresh = (TenonObj*)(uintptr_t)fp;
    fresh->len = obj->len + 1;
    fresh->rc_mask = obj->rc_mask;
    tn_fs_copy(fresh->fields, obj->fields, (size_t)obj->len * sizeof(int64_t));
    fresh->fields[obj->len] = val;
    tenon_rc_dec(ptr);
    return fp;
}
int64_t tenon_list_pop_shrink(int64_t ptr) {
    if (ptr == 0) return 0;
    TenonObj* obj = (TenonObj*)(uintptr_t)ptr;
    if (obj->len > 0) obj->len--;
    return ptr;
}
int64_t tenon_flat_list_filled(int64_t len, int64_t eb, const void* init_ptr) {
    int64_t words = (len * eb + 7) / 8;
    int64_t p = tenon_rc_alloc(18, words);
    TenonObj* obj = (TenonObj*)(uintptr_t)p;
    obj->len = len;
    uint8_t* dst = (uint8_t*)obj->fields;
    if (init_ptr) {
        for (int64_t i = 0; i < len; i++) tn_fs_copy(dst + i * eb, init_ptr, (size_t)eb);
    }
    return p;
}
int64_t tenon_flat_list_push(int64_t ptr, int64_t eb, const void* val_ptr) {
    (void)eb; (void)val_ptr;
    return ptr;
}
int64_t tenon_cstruct_alloc(int64_t bytes) {
    int64_t words = (bytes + 7) / 8;
    return tenon_rc_alloc(16, words);
}
int64_t tenon_cstruct_data_ptr(int64_t ptr) {
    return ptr ? (int64_t)(uintptr_t)((TenonObj*)(uintptr_t)ptr)->fields : 0;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_all_builtin_targets_and_inheritance() {
        let all = list_builtin_targets();
        assert!(all.len() >= 20, "expected >= 20 built-in targets, got {}", all.len());

        let rp = resolve_target("rp2040").expect("resolve rp2040");
        assert_eq!(rp.llvm_target, "thumbv6m-none-eabi");
        assert_eq!(rp.cpu.as_deref(), Some("cortex-m0plus"));
        assert!(rp.freestanding);
        assert!(rp.build_tags.contains(&"cortexm".to_string()));
        assert!(rp.build_tags.contains(&"rp2040".to_string()));

        let esp = resolve_target("esp32c3").expect("resolve esp32c3");
        assert_eq!(esp.llvm_target, "riscv32-unknown-none-elf");
        assert!(esp.freestanding);
        assert!(esp.cflags.iter().any(|f| f.contains("rv32imc")));

        let lnx = resolve_target("linux-x86_64").expect("resolve linux-x86_64");
        assert_eq!(lnx.llvm_target, "x86_64-unknown-linux-gnu");
        assert!(!lnx.freestanding);

        let lnx_cu = resolve_target("linux-x86_64-cuda").expect("resolve linux-x86_64-cuda");
        assert_eq!(lnx_cu.llvm_target, "x86_64-unknown-linux-gnu");
        assert!(!lnx_cu.freestanding);
        assert!(lnx_cu.build_tags.contains(&"cuda".to_string()));
        assert!(lnx_cu.cflags.iter().any(|f| f.contains("TENON_TARGET_CUDA")));

        let raw = resolve_target("aarch64-unknown-linux-musl").expect("resolve raw triple");
        assert_eq!(raw.os, "linux");
        assert_eq!(raw.arch, "arm64");
    }
}
