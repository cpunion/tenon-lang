# ADR-0003: Zero-Keyword Declarative CPU/GPU Integration & Unified OpenUI + VDOM (Web / GPU / TUI) Architecture

- **Status**: Accepted (2026-11-25)
- **Supersedes**: Prototype A2UI demo (`examples/dogfood/a2ui_wasm_kernel.tn`)

## Context

Tenon's core philosophy is *"minimal orthogonal language core + static refinement proofs (L0/L1/L2) + Perceus/Arena deterministic memory + declarative reconciliation"*.
Two strategic priorities emerged for v1.0+:
1. Replace the prototype A2UI kernel with **OpenUI (`thesysdev/openui`)** — a compact, line-oriented, streaming-first Generative UI language (`OpenUI Lang`) backed by a React-style unified VDOM reconciler that targets **Web (Wasm32)**, **Terminal TUI**, and **Native GPU (`tenon-gpui`)** from a single pure-Tenon source.
2. Build a **zero-Rust native GPU & GUI stack** (`tenon-gpu` / `tenon-gpui`, rivaling and surpassing Rust's `wgpu` + `gpui`) and gradually self-host the Tenon compiler, **without bloating the core lexer with domain-specific hard keywords**.

## Decisions

### 1. Grammar Slimming — Zero New Keywords & Contextual Domain Words
- **Rule**: GPU compute/graphics, OpenUI, VDOM, and TUI introduce **zero new hard lexer keywords**. All capabilities compose from Tenon's orthogonal core (`record`, `enum`, `fn`, `inout`, `where`/`requires`/`ensures`, `comptime`) plus minimal `@attr` annotations (`@gpu_struct`, `@kernel`, `@gpu_graph`, `@linear`, `@checkfree`, `@arena`).
- Existing domain keywords (`bounds`, `output`, `view`, `stream`, `from`, `bind`, `policy`, `pacing`, `protocol`, `reactive`, `laws`, `subscriber`, `archive`, `retain`, `allow_loss`, `lock_lattice`, `satisfies`, `fault_model`, `local_storage`, `evolution`) are permitted as ordinary identifiers in expressions, parameters, and struct fields while keeping legacy top-level declarations 100% backward-compatible.

### 2. Zero-Keyword Declarative CPU/GPU Unification
- **`@gpu_struct(std140|std430)`**: Single-source host/shader POD structs. Statically forbids RC fields (`E0830`), verifies 4/8/16-byte GPU alignment, and emits matching Metal MSL / WebGPU WGSL struct definitions.
- **`@kernel pure fn`**: Single-source CPU/GPU compute/graphics kernels. Unit-testable on CPU JIT, statically verified `@checkfree` by L0/Z3 (zero OOB / overflow / `NaN`), and compiled from `tenon-mir` to **Metal MSL**, **WebGPU WGSL**, and **CPU SIMD**.
- **`@gpu_graph fn`**: Ordinary Tenon functions whose `let` bindings (reads) + `inout` parameters (exclusive writes, enforced by `E0820`) + non-escaping local buffers/textures form an exact SSA Dataflow DAG. The compiler reconciles this DAG to:
  1. Prove tensor shape compatibility (`requires a.cols == b.rows`), acyclicity (`E0840`), and RAW/WAW hazard safety (`E0841`);
  2. Synthesize minimal GPU memory barriers & layout transitions;
  3. Alias non-overlapping transient VRAM allocations (GPU `@arena`);
  4. Fuse consecutive pure elementwise `@kernel` dispatches into a single GPU pass.
- **`@linear` Imperative Escape Hatch (`std/gpu/`)**: For low-level driver control, `@linear` handles (`CommandEncoder`, `RenderPass`, `GpuBuffer`, `GpuTexture`) distinguish `inout self` **borrow** calls (`set_pipeline`, `draw`) from by-value `self` **consume** calls (`end_pass`, `submit`), preventing resource leaks and invalid pass ordering at compile time (`E0821`).

### 3. Unified 4-Layer UI Stack (OpenUI + VDOM → Web / TUI / Native Metal GPU)
1. **Declarative Components / OpenUI Lang** (`std/ui/openui.tn`): Line-oriented streaming parser with forward-reference hoisting, bounded-depth cycle protection, `$var` state, `@action` events, and `comptime` Schema/Prompt generation.
2. **Core VDOM Reconciler** (`std/ui/vdom.tn`): Keyed `VNode` diff → `List[VDomPatch]`, reactive signals, running in `@arena` + Perceus FBIP.
3. **Three Native Renderers**:
   - **Web**: Wasm32 + `--emit-mjs` auto-object bridge streaming `VDomPatch` to HTML DOM (`liveObjects() === 0`).
   - **TUI** (`std/ui/tui.tn`): `@checkfree` double-buffered 2D cell grid (`cells[y * w + x]` statically proved in-bounds) emitting minimal ANSI/VT100 diffs.
   - **Native GPU (`tenon-gpui`)**: `@arena` Flexbox layout (`std/ui/layout.tn`) + batched `QuadInstance` scene graph (`std/ui/scene.tn`) rendered via zero-Rust macOS Metal backend (`std/gpu/metal_shim.m`).
