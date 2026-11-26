//! Stage 1 (Pillars 1-5) + Stage 2 (Warp-Spec & PGO Candidates) + Stage 3 (Ragged/Sparse/MXFP4):
//! Universal Producer-Consumer Fusion, Whole-Graph Layout Propagation, Static Arena Liveness
//! Graph-Coloring Allocator, and Tile-IR Schedule Planner (`fusion.rs`).

use super::*;
use std::collections::{BTreeMap, BTreeSet};

/// Post-contraction register-level epilogue operation fused directly into tile store.
#[derive(Debug, Clone)]
pub enum EpilogueOp {
    AddTensor { src: String, broadcast_row: bool },
    SubTensor { src: String },
    MulTensor { src: String },
    Silu,
    Gelu,
    Relu,
    /// Exact fused Reverse-Mode AD VJP epilogue: `val *= silu'(saved_x)`.
    SiluBwd { saved_x: String },
    /// Exact fused Reverse-Mode AD VJP epilogue: `val *= gelu'(saved_x)`.
    GeluBwd { saved_x: String },
    /// Exact fused Reverse-Mode AD VJP epilogue: `val = (saved_x > 0) ? val : 0`.
    ReluBwd { saved_x: String },
    Scale(f32),
}

/// Optional fused prologue reduction folded before a contraction (eliminating intermediate HBM write).
#[derive(Debug, Clone)]
pub struct PrologueNorm {
    pub kind: ReduceOp,
    pub x: String,
    pub weight: String,
}

/// Concrete hardware Tile-IR schedule parameters (also used as PGO Auto-Tune candidates!).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileSchedule {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub num_sgs: usize,
    /// Output columns handled per SIMD-group in GEMV (`1`, `2`, or `4` -> shares activation loads across rows!).
    pub rows_per_sg: usize,
    pub split_k: usize,
    pub double_buf: bool,
    pub swizzle_pad: usize,
    pub use_mma: bool,
    pub stream_k: bool,
    pub warp_specialized: bool,
    /// Facility #1: 2D L2/SLC Tile Swizzling (`GROUP_M = 8` block rasterization order).
    pub tile_swizzle: bool,
}

impl TileSchedule {
    pub fn tag(&self) -> String {
        format!(
            "m{}n{}k{}_sg{}_r{}_sk{}{}{}{}",
            self.bm,
            self.bn,
            self.bk,
            self.num_sgs,
            self.rows_per_sg,
            self.split_k,
            if self.double_buf { "_db" } else { "" },
            if self.swizzle_pad > 0 { "_swz" } else { "" },
            if self.tile_swizzle { "_l2" } else { "" }
        )
    }

    pub fn smem_bytes(&self, elem_bytes: usize) -> usize {
        if !self.use_mma {
            return 0;
        }
        let bufs = if self.double_buf { 2 } else { 1 };
        let a_bytes = self.bm * (self.bk + self.swizzle_pad) * elem_bytes;
        let b_bytes = self.bk * (self.bn + self.swizzle_pad) * elem_bytes;
        bufs * (a_bytes + b_bytes)
    }
}

/// Unified lowered Tile-IR execution stage (backend-agnostic!).
#[derive(Debug, Clone)]
pub enum TileStage {
    /// General Contraction (`C = op(A) @ op(B)`) with optional:
    ///   - Fused Prologue Norm (`RMSNorm` / `LayerNorm`)
    ///   - Arbitrary Register-Level Epilogue Chain (`+ residual/bias`, `silu`, `gelu`, `relu`, `scale`)
    ///   - Dual Horizontal Sibling Fusion (`w_up` + `SwiGlu` or parallel outputs!)
    ///   - N-Way Horizontal Sibling Fusion (`extra_weights` + `extra_outs`!)
    ///   - Layout (`RowMajor` vs `ColMajorNK` vs `MXFP4Block32NK`)
    ///   - PGO Auto-Tune candidate schedules!
    Contraction {
        stage_id: usize,
        name: String,
        a: String,
        b: String,
        trans_a: bool,
        trans_b: bool,
        m: usize,
        k: usize,
        n: usize,
        /// Facility #10 (`auto_pad`): physical padded K/N (`>= k, n`, aligned to 64 for 128-bit & MMA fast paths!).
        padded_k: usize,
        padded_n: usize,
        act_dtype: IrDType,
        weight_dtype: IrDType,
        weight_layout: LayoutTag,
        prologue: Option<PrologueNorm>,
        /// Optional horizontal sibling weight (e.g. `w_up` fused with `w_gate` into SwiGLU).
        dual_b: Option<String>,
        dual_swiglu: bool,
        /// Optional extra horizontal sibling contractions sharing input `a` (N-way general fusion!).
        extra_siblings: Vec<(String, String, usize)>, // (weight_name, out_name, n_cols)
        epilogue: Vec<EpilogueOp>,
        out: String,
        schedule: TileSchedule,
        pgo_candidates: Vec<TileSchedule>,
        /// Optional Arena scratch buffer name if `schedule.split_k > 1`.
        split_k_scratch: Option<String>,
    },
    /// General Row Reduction + optional fused post-elementwise ops (`RmsNorm`, `LayerNorm`, `Softmax`).
    RowReduce {
        stage_id: usize,
        name: String,
        op: ReduceOp,
        x: String,
        weight: Option<String>,
        rows: usize,
        cols: usize,
        dtype: IrDType,
        epilogue: Vec<EpilogueOp>,
        out: String,
        vectorized: bool,
    },
    /// General Vectorized Elementwise Kernel (standalone or fused chain!).
    Elementwise {
        stage_id: usize,
        name: String,
        inputs: Vec<String>,
        ops: Vec<ElemOp>,
        numel: usize,
        dtype: IrDType,
        out: String,
        vectorized: bool,
        /// Facility #12 (`streaming_cache`): 64B full-cacheline 4x128-bit streaming burst / CUDA `__ldcs`/`__stcs`.
        streaming_cache: bool,
    },
    /// General Attention: either single-pass **Tiled Online-Softmax FlashAttention** (`online_tiled=true`,
    /// ZERO O(S^2) HBM!) or naive 3-stage $O(S^2)$ materialization (`online_tiled=false` for ablation!).
    AttentionStage {
        stage_id: usize,
        name: String,
        q: String,
        k: String,
        v: String,
        seq: usize,
        q_dim: usize,
        kv_dim: usize,
        dtype: IrDType,
        causal: bool,
        paged: bool,
        fused_rope: bool,
        online_tiled: bool,
        /// Hardware Matrix Unit FlashAttention-2 (`simdgroup_half8x8` / Tensor Core `mma.sync`) when `tile_mma && online_tiled`!
        use_mma: bool,
        out: String,
        /// Scratch buffer name `[seq, seq]` when `online_tiled == false` (naive HBM materialization!).
        naive_scores_scratch: Option<String>,
    },
    /// Stage 3 — Compact Ragged-Batch MatMul (skips padding tokens via `valid_rows`).
    RaggedContraction {
        stage_id: usize,
        name: String,
        a: String,
        b: String,
        m_padded: usize,
        m_valid: usize,
        k: usize,
        n: usize,
        dtype: IrDType,
        weight_layout: LayoutTag,
        compact_enabled: bool,
        schedule: TileSchedule,
        out: String,
    },
    /// Stage 3 — Structured Block-Sparse MatMul with Device-Side Persistent Work-Stealing Queue.
    BlockSparseContraction {
        stage_id: usize,
        name: String,
        a: String,
        b: String,
        m: usize,
        k: usize,
        n: usize,
        block_size: usize,
        density_pct: usize,
        dtype: IrDType,
        weight_layout: LayoutTag,
        sparse_enabled: bool,
        schedule: TileSchedule,
        out: String,
    },
    /// Explicit Transpose `[rows, cols] -> [cols, rows]` (only survives if E-Graph is OFF!).
    TransposeStage {
        stage_id: usize,
        name: String,
        x: String,
        rows: usize,
        cols: usize,
        dtype: IrDType,
        out: String,
    },
    /// SPMD Collective stage (`AllReduceSum` / `ReduceScatter` / `AllGather`).
    CollectiveStage {
        stage_id: usize,
        name: String,
        kind: CollectiveKind,
        x: String,
        numel: usize,
        dtype: IrDType,
        devices: usize,
        out: String,
    },
    /// Facility #2: Horizontal Weight-Packed N-way Projection (`PACK_CONCAT_NK` -> 1 wide `64x64` AMX GEMM).
    ConcatContraction {
        stage_id: usize,
        name: String,
        a: String,
        weights: Vec<String>,
        m: usize,
        k: usize,
        n_per: usize,
        total_n: usize,
        dtype: IrDType,
        packed_enabled: bool,
        schedule: TileSchedule,
        out: String,
    },
    /// Facility #3/#11/#12/#13: Zero-Copy Buffer-Donated Fused 128-bit AdamW Update (`llm.c` `adamw_kernel3`!).
    InPlaceAdamW {
        stage_id: usize,
        name: String,
        param: String,
        grad: String,
        m: String,
        v: String,
        numel: usize,
        dtype: IrDType,
        donate_enabled: bool,
        /// Facility #11: Stateless Counter-Based Philox BF16/FP16 stochastic rounding (no FP32 master weights!).
        stochastic_round: bool,
        /// Facility #12: Streaming cache hints (`__ldcs`/`__stcs` / 64B burst).
        streaming_cache: bool,
        /// Facility #13: ZeRO-1 optimizer shard factor (`1` = full, `D > 1` = `1/D` local shard update!).
        zero1_shards: usize,
        out: String,
    },
    /// Facility #5/#10/#12: Online Monoid Fused Cross-Entropy Loss + In-Place Backward `dlogits` (`llm.c` `fused_classifier3`!).
    CrossEntropyStage {
        stage_id: usize,
        name: String,
        logits: String,
        targets: String,
        rows: usize,
        vocab: usize,
        /// Facility #10: physical aligned vocab (`50257 -> 50304`, multiple of 64 when `auto_pad` ON!).
        padded_vocab: usize,
        dtype: IrDType,
        online_fused: bool,
        donate_enabled: bool,
        streaming_cache: bool,
        scratch_probs: Option<String>,
        out: String,
    },
    /// Facility #6: Auto Mixed-Precision Fused Single-Pass RMSNorm + Dynamic Quant W8A8/FP8 Contraction.
    AmpQuantContraction {
        stage_id: usize,
        name: String,
        x: String,
        nw: String,
        w: String,
        m: usize,
        k: usize,
        n: usize,
        dtype: IrDType,
        quant_enabled: bool,
        weight_layout: LayoutTag,
        schedule: TileSchedule,
        out: String,
    },
    /// Facility #7: Symbolic Dynamic-Shape Multi-Version Bucketed Contraction (O(1) dispatch, zero padding!).
    DynBucketContraction {
        stage_id: usize,
        name: String,
        a: String,
        b: String,
        max_m: usize,
        k: usize,
        n: usize,
        dtype: IrDType,
        weight_layout: LayoutTag,
        bucket_enabled: bool,
        tile_swizzle: bool,
        out: String,
    },
    /// Facility #8A: Implicit GEMM 3x3 Conv2d (zero `im2col` HBM expansion!) vs explicit `im2col` + MatMul.
    Conv2dStage {
        stage_id: usize,
        name: String,
        x: String,
        w: String,
        in_h: usize,
        in_w: usize,
        c_in: usize,
        c_out: usize,
        kh: usize,
        kw: usize,
        dtype: IrDType,
        weight_layout: LayoutTag,
        implicit_enabled: bool,
        im2col_scratch: Option<String>,
        out: String,
    },
    /// Facility #8B: First-Class Associative Parallel Scan (`h_t = a_t * h_{t-1} + b_t`, Mamba/SSM/Linear-Attn).
    ParallelScanStage {
        stage_id: usize,
        name: String,
        a: String,
        b: String,
        batch: usize,
        seq_len: usize,
        dtype: IrDType,
        parallel_enabled: bool,
        out: String,
    },
}

/// Static compile-time Arena memory layout computed via interval graph coloring!
#[derive(Debug, Clone)]
pub struct StaticArenaPlan {
    pub enabled: bool,
    pub total_arena_bytes: usize,
    pub naive_total_bytes: usize,
    pub vram_saved_pct: f64,
    /// Bytes saved via Facility #3 Zero-Copy `inout` Buffer Donation (`DONATE_INOUT`).
    pub donated_bytes: usize,
    /// Maps each intermediate tensor name -> `(byte_offset_in_arena, byte_size)`.
    pub allocations: BTreeMap<String, (usize, usize)>,
}

/// Lowers `TensorIrGraph` into fused `TileStage`s + Layout Repacks + Auto-Pads + Cache Policies + Static Arena Plan.
pub fn lower_to_tile_stages(
    g: &mut TensorIrGraph,
    cfg: GpuOptConfig,
) -> (
    Vec<TileStage>,
    StaticArenaPlan,
    Vec<(String, LayoutTag)>,
    Vec<PadAlignSpec>,
    BTreeMap<String, CachePolicy>,
) {
    let input_set: BTreeSet<String> = g.params.iter().map(|(k, _)| k.clone()).collect();
    let output_set: BTreeSet<String> = g.outputs.iter().cloned().collect();

    // Compute consumer counts AND identify tensors read as activations/LHS/elementwise inputs.
    // A parameter used as LHS `a` (e.g. `x` or `__d_out` in Reverse-Mode AD!) MUST stay `RowMajor`
    // so repacking RHS weights to `ColMajorNK`/`MXFP4` never corrupts an LHS consumer!
    let mut use_count: BTreeMap<String, usize> = BTreeMap::new();
    let mut non_weight_uses: BTreeSet<String> = BTreeSet::new();
    for n in &g.nodes {
        bump_node_uses(&n.op, &mut use_count);
        match &n.op {
            IrOp::MatMul { a, .. }
            | IrOp::Conv2d { x: a, .. }
            | IrOp::RaggedMatMul { a, .. }
            | IrOp::BlockSparseMatMul { a, .. }
            | IrOp::DynMatMul { a, .. }
            | IrOp::ConcatMatMul { a, .. } => {
                non_weight_uses.insert(a.clone());
            }
            IrOp::AmpNormMatMul { x, nw, .. } => {
                non_weight_uses.insert(x.clone());
                non_weight_uses.insert(nw.clone());
            }
            IrOp::AdamWStep { param, grad, m, v } => {
                non_weight_uses.insert(param.clone());
                non_weight_uses.insert(grad.clone());
                non_weight_uses.insert(m.clone());
                non_weight_uses.insert(v.clone());
            }
            IrOp::CrossEntropy { logits, targets } => {
                non_weight_uses.insert(logits.clone());
                non_weight_uses.insert(targets.clone());
            }
            IrOp::ParallelScan { a, b } => {
                non_weight_uses.insert(a.clone());
                non_weight_uses.insert(b.clone());
            }
            IrOp::Binary { a, b, .. } => {
                non_weight_uses.insert(a.clone());
                non_weight_uses.insert(b.clone());
            }
            IrOp::Unary { x, .. }
            | IrOp::Reduce { x, .. }
            | IrOp::Rope { x }
            | IrOp::Transpose { x }
            | IrOp::Collective { x, .. } => {
                non_weight_uses.insert(x.clone());
            }
            IrOp::Attention { q, k, v, .. } => {
                non_weight_uses.insert(q.clone());
                non_weight_uses.insert(k.clone());
                non_weight_uses.insert(v.clone());
            }
        }
    }
    for o in &g.outputs {
        *use_count.entry(o.clone()).or_default() += 1;
    }

    // Track which weights should be repacked at load time (`input_repacks`)
    let mut weight_layouts: BTreeMap<String, LayoutTag> = BTreeMap::new();
    let mut bsr32_weights: Vec<String> = Vec::new();
    let mut int8_weights: Vec<String> = Vec::new();
    let mut input_pads: Vec<PadAlignSpec> = Vec::new();
    let mut donated_bytes_total: usize = 0;

    // Facility #12: Automatic Liveness-Driven Cache Policy (`Streaming` vs `Persistent`)!
    let mut cache_policies: BTreeMap<String, CachePolicy> = BTreeMap::new();
    if cfg.streaming_cache {
        for (tname, tdesc) in &g.tensors {
            let uses = use_count.get(tname).copied().unwrap_or(0);
            let policy = if tdesc.alloc_bytes() >= 4 * 1024 * 1024 && uses <= 1 {
                CachePolicy::Streaming
            } else if uses >= 2 {
                CachePolicy::Persistent
            } else {
                CachePolicy::Default
            };
            cache_policies.insert(tname.clone(), policy);
        }
    }
    let mut resolve_weight_layout = |w_name: &str, m: usize, k: usize, n: usize, w_dtype: IrDType, trans_b: bool| -> (IrDType, LayoutTag) {
        // If this parameter was already assigned a physical layout by an earlier stage (e.g. Forward
        // repacked `w1` to `ColMajorNK` and Backward reads `w1^T`), ALWAYS return its TRUE physical layout
        // so emitters compute `b_is_nk = trans_b ^ (w_layout == ColMajorNK)` with zero double-transpose!
        if let Some(&existing) = weight_layouts.get(w_name) {
            let eff_dt = if existing.is_mxfp4() { IrDType::MXFP4 } else { w_dtype };
            return (eff_dt, existing);
        }
        if trans_b {
            return (w_dtype, LayoutTag::RowMajor);
        }
        let is_pure_weight = input_set.contains(w_name) && !non_weight_uses.contains(w_name);
        // Stage 3: MXFP4 auto-quantization on memory-bound weights (`M <= 8` & `K % 32 == 0`) or explicit MXFP4 dtype
        // Uses 16-block interleaved SuperBlock16 (272B, 16B aligned, single DRAM stream) whenever `K % 512 == 0`!
        if cfg.mxfp4_quant
            && is_pure_weight
            && (w_dtype == IrDType::MXFP4
                || (matches!(w_dtype, IrDType::F32 | IrDType::F16 | IrDType::BF16)
                    && m <= 8
                    && k >= 256
                    && k % 32 == 0
                    && n >= 64))
        {
            let lay = if k % 512 == 0 {
                LayoutTag::MXFP4Super16NK
            } else {
                LayoutTag::MXFP4Block32NK
            };
            weight_layouts.insert(w_name.to_string(), lay);
            return (IrDType::MXFP4, lay);
        }
        if cfg.layout_prop && is_pure_weight && !w_dtype.is_sub_byte() {
            weight_layouts.insert(w_name.to_string(), LayoutTag::ColMajorNK);
            return (w_dtype, LayoutTag::ColMajorNK);
        }
        (w_dtype, LayoutTag::RowMajor)
    };

    let mut consumed = vec![false; g.nodes.len()];
    let mut stages = Vec::new();
    let mut extra_scratch_sizes: BTreeMap<String, usize> = BTreeMap::new();

    for i in 0..g.nodes.len() {
        if consumed[i] {
            continue;
        }
        let node = &g.nodes[i];
        let stage_id = stages.len();

        // === Pattern A (when `general_fusion` ON): Prologue Norm -> Contraction / Dual SwiGLU / N-Way Siblings ===
        if cfg.general_fusion {
            if let IrOp::Reduce { op: rkind @ (ReduceOp::RmsNorm | ReduceOp::LayerNorm), x, weight: Some(nw) } = &node.op {
                let h_var = &node.out;
                let h_uses = use_count.get(h_var).copied().unwrap_or(0);
                let mut mm_consumers = Vec::new();
                for j in (i + 1)..g.nodes.len() {
                    if !consumed[j] {
                        if let IrOp::MatMul { a, trans_a: false, .. } = &g.nodes[j].op {
                            if a == h_var {
                                mm_consumers.push(j);
                            }
                        }
                    }
                }
                if !mm_consumers.is_empty() && mm_consumers.len() == h_uses {
                    let first_j = mm_consumers[0];
                    let IrOp::MatMul { b: w0, trans_b: tb0, weight_dtype: wdt0, .. } = &g.nodes[first_j].op else { unreachable!() };
                    let sx = &g.tensors[x];
                    let m = sx.rows();
                    let k = sx.cols();
                    let n0 = g.nodes[first_j].desc.cols();

                    // Check 2-way Dual MatMul -> SwiGLU super-fusion (`g = h @ w_gate`, `u = h @ w_up`, `a = swiglu(g, u)`)
                    let mut dual_b = None;
                    let mut dual_swiglu = false;
                    let mut out_var = g.nodes[first_j].out.clone();
                    let mut extra_siblings = Vec::new();

                    if mm_consumers.len() == 2 {
                        let second_j = mm_consumers[1];
                        let IrOp::MatMul { b: w1, trans_b: tb1, .. } = &g.nodes[second_j].op else { unreachable!() };
                        if *tb0 == *tb1 && g.nodes[second_j].desc.cols() == n0 {
                            let v0 = &g.nodes[first_j].out;
                            let v1 = &g.nodes[second_j].out;
                            let mut swi_idx = None;
                            for sk in (second_j + 1)..g.nodes.len() {
                                if !consumed[sk] {
                                    if let IrOp::Binary { op: ElemOp::SwiGlu, a: sa, b: sb } = &g.nodes[sk].op {
                                        if (sa == v0 && sb == v1) || (sa == v1 && sb == v0) {
                                            swi_idx = Some(sk);
                                            break;
                                        }
                                    }
                                }
                            }
                            if let Some(sk) = swi_idx {
                                consumed[i] = true;
                                consumed[first_j] = true;
                                consumed[second_j] = true;
                                consumed[sk] = true;
                                dual_b = Some(w1.clone());
                                dual_swiglu = true;
                                out_var = g.nodes[sk].out.clone();
                            }
                        }
                    }

                    // If not SwiGLU, check general N-way sibling contractions (`Q || K || V`, or single `Norm -> MatMul`)
                    if !consumed[i] {
                        consumed[i] = true;
                        consumed[first_j] = true;
                        for &sib_j in &mm_consumers[1..] {
                            if let IrOp::MatMul { b: wsib, .. } = &g.nodes[sib_j].op {
                                consumed[sib_j] = true;
                                extra_siblings.push((wsib.clone(), g.nodes[sib_j].out.clone(), g.nodes[sib_j].desc.cols()));
                            }
                        }
                    }

                    // Absorb any trailing single-use epilogue ops on `out_var`!
                    let (epilogue, final_out) = if extra_siblings.is_empty() {
                        absorb_epilogue_chain(g, &mut consumed, &use_count, &output_set, out_var)
                    } else {
                        (Vec::new(), out_var)
                    };

                    let (eff_wdt, wlayout) = resolve_weight_layout(w0, m, k, n0, *wdt0, *tb0);
                    if let Some(ref wb) = dual_b {
                        let _ = resolve_weight_layout(wb, m, k, n0, *wdt0, *tb0);
                    }
                    for (ws, _, ns) in &extra_siblings {
                        let _ = resolve_weight_layout(ws, m, k, *ns, *wdt0, *tb0);
                    }

                    let (schedule, pgo_candidates) = solve_tile_schedule(m, k, n0, sx.dtype, eff_wdt, wlayout, dual_b.is_some(), cfg);
                    let max_sk = pgo_candidates.iter().map(|c| c.split_k).max().unwrap_or(schedule.split_k).max(schedule.split_k);
                    let split_k_scratch = if max_sk > 1 {
                        let sc_name = format!("__sk_scratch_{}", stage_id);
                        extra_scratch_sizes.insert(sc_name.clone(), max_sk * m * n0 * 4);
                        Some(sc_name)
                    } else {
                        None
                    };

                    stages.push(TileStage::Contraction {
                        stage_id,
                        name: format!("{}_s{}_fused_norm_mm", g.fn_name, stage_id),
                        a: x.clone(),
                        b: w0.clone(),
                        trans_a: false,
                        trans_b: *tb0,
                        m,
                        k,
                        n: n0,
                        padded_k: k,
                        padded_n: n0,
                        act_dtype: sx.dtype,
                        weight_dtype: eff_wdt,
                        weight_layout: wlayout,
                        prologue: Some(PrologueNorm { kind: *rkind, x: x.clone(), weight: nw.clone() }),
                        dual_b,
                        dual_swiglu,
                        extra_siblings,
                        epilogue,
                        out: final_out,
                        schedule,
                        pgo_candidates,
                        split_k_scratch,
                    });
                    continue;
                }
            }
        }

        // === Pattern B: General Contraction (`MatMul` or `Conv2d` lowered to im2col Contraction) ===
        match &node.op {
            IrOp::MatMul { a, b, trans_a, trans_b, weight_dtype } => {
                let sa = &g.tensors[a];
                let sb = &g.tensors[b];
                let m = if *trans_a { sa.cols() } else { sa.rows() };
                let k = if *trans_a { sa.rows() } else { sa.cols() };
                let n = if *trans_b { sb.rows() } else { sb.cols() };
                consumed[i] = true;

                // When `general_fusion` ON: check Dual SwiGLU sibling without norm, AND absorb trailing Epilogue chain!
                let mut dual_b = None;
                let mut dual_swiglu = false;
                let mut out_var = node.out.clone();
                if cfg.general_fusion && !*trans_a && !*trans_b {
                    for j in (i + 1)..g.nodes.len() {
                        if !consumed[j] {
                            if let IrOp::MatMul { a: a2, b: b2, trans_a: false, trans_b: false, .. } = &g.nodes[j].op {
                                if a2 == a && g.nodes[j].desc.cols() == n {
                                    let v0 = &node.out;
                                    let v1 = &g.nodes[j].out;
                                    for sk in (j + 1)..g.nodes.len() {
                                        if !consumed[sk] {
                                            if let IrOp::Binary { op: ElemOp::SwiGlu, a: sa_s, b: sb_s } = &g.nodes[sk].op {
                                                if (sa_s == v0 && sb_s == v1) || (sa_s == v1 && sb_s == v0) {
                                                    consumed[j] = true;
                                                    consumed[sk] = true;
                                                    dual_b = Some(b2.clone());
                                                    dual_swiglu = true;
                                                    out_var = g.nodes[sk].out.clone();
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    if dual_swiglu { break; }
                                }
                            }
                        }
                    }
                }

                let (epilogue, final_out) = if cfg.general_fusion {
                    absorb_epilogue_chain(g, &mut consumed, &use_count, &output_set, out_var)
                } else {
                    (Vec::new(), out_var)
                };

                // Facility #10 (`auto_pad`): Automatic logical-vs-padded alignment for odd/unaligned MatMul dims!
                // Zero-pads `a[m, k] -> [m, pad_k]` and `b[k, n] -> [pad_n, pad_k]` (NK) ONCE on upload so the
                // entire AMX inner loop executes branchless 128-bit `uint4` double-buffered 64x64 tiles!
                let can_auto_pad = cfg.auto_pad
                    && cfg.tile_mma
                    && !*trans_a
                    && !*trans_b
                    && dual_b.is_none()
                    && epilogue.is_empty()
                    && sa.dtype == IrDType::F16
                    && sb.dtype == IrDType::F16
                    && m >= 16
                    && (k % 8 != 0 || n % 8 != 0)
                    && input_set.contains(a)
                    && input_set.contains(b);
                let (padded_k, padded_n, eff_wdt, wlayout) = if can_auto_pad {
                    let pk = (k + 63) & !63;
                    let pn = (n + 63) & !63;
                    if !input_pads.iter().any(|p| p.name == *a) {
                        input_pads.push(PadAlignSpec { name: a.clone(), pad_rows: m, pad_cols: pk, is_nk: false });
                    }
                    if !input_pads.iter().any(|p| p.name == *b) {
                        input_pads.push(PadAlignSpec { name: b.clone(), pad_rows: pk, pad_cols: pn, is_nk: true });
                    }
                    (pk, pn, *weight_dtype, LayoutTag::ColMajorNK)
                } else {
                    let (ew, wl) = resolve_weight_layout(b, m, k, n, *weight_dtype, *trans_b);
                    (k, n, ew, wl)
                };
                if let Some(ref wb) = dual_b {
                    let _ = resolve_weight_layout(wb, m, k, n, *weight_dtype, *trans_b);
                }

                let (schedule, pgo_candidates) = solve_tile_schedule(m, padded_k, padded_n, sa.dtype, eff_wdt, wlayout, dual_b.is_some(), cfg);
                let max_sk = pgo_candidates.iter().map(|c| c.split_k).max().unwrap_or(schedule.split_k).max(schedule.split_k);
                let split_k_scratch = if max_sk > 1 {
                    let sc_name = format!("__sk_scratch_{}", stage_id);
                    extra_scratch_sizes.insert(sc_name.clone(), max_sk * m * n * 4);
                    Some(sc_name)
                } else {
                    None
                };

                stages.push(TileStage::Contraction {
                    stage_id,
                    name: format!("{}_s{}_mm", g.fn_name, stage_id),
                    a: a.clone(),
                    b: b.clone(),
                    trans_a: *trans_a,
                    trans_b: *trans_b,
                    m,
                    k,
                    n,
                    padded_k,
                    padded_n,
                    act_dtype: sa.dtype,
                    weight_dtype: eff_wdt,
                    weight_layout: wlayout,
                    prologue: None,
                    dual_b,
                    dual_swiglu,
                    extra_siblings: Vec::new(),
                    epilogue,
                    out: final_out,
                    schedule,
                    pgo_candidates,
                    split_k_scratch,
                });
            }
            IrOp::Conv2d { x, w, in_h, in_w, kh, kw } => {
                let sx = &g.tensors[x];
                let sw = &g.tensors[w];
                let c_in = sx.cols();
                let c_out = sw.cols();
                consumed[i] = true;
                if *kh == 1 && *kw == 1 {
                    let (m, k, n) = (sx.rows(), sx.cols(), sw.cols());
                    let (epilogue, final_out) = if cfg.general_fusion {
                        absorb_epilogue_chain(g, &mut consumed, &use_count, &output_set, node.out.clone())
                    } else {
                        (Vec::new(), node.out.clone())
                    };
                    let (eff_wdt, wlayout) = resolve_weight_layout(w, m, k, n, sw.dtype, false);
                    let (schedule, pgo_candidates) = solve_tile_schedule(m, k, n, sx.dtype, eff_wdt, wlayout, false, cfg);
                    stages.push(TileStage::Contraction {
                        stage_id,
                        name: format!("{}_s{}_conv2d_mm", g.fn_name, stage_id),
                        a: x.clone(),
                        b: w.clone(),
                        trans_a: false,
                        trans_b: false,
                        m,
                        k,
                        n,
                        padded_k: k,
                        padded_n: n,
                        act_dtype: sx.dtype,
                        weight_dtype: eff_wdt,
                        weight_layout: wlayout,
                        prologue: None,
                        dual_b: None,
                        dual_swiglu: false,
                        extra_siblings: Vec::new(),
                        epilogue,
                        out: final_out,
                        schedule,
                        pgo_candidates,
                        split_k_scratch: None,
                    });
                } else {
                    let k_eff = c_in * (*kh) * (*kw);
                    let m_eff = (*in_h) * (*in_w);
                    let (_, wlayout) = resolve_weight_layout(w, m_eff, k_eff, c_out, sw.dtype, false);
                    let implicit_enabled = cfg.non_tf_ops;
                    let im2col_scratch = if !implicit_enabled {
                        let sc = format!("__im2col_{}", stage_id);
                        extra_scratch_sizes.insert(sc.clone(), m_eff * k_eff * (sx.dtype.bytes_f64() as usize));
                        Some(sc)
                    } else {
                        None
                    };
                    stages.push(TileStage::Conv2dStage {
                        stage_id,
                        name: format!("{}_s{}_conv2d", g.fn_name, stage_id),
                        x: x.clone(),
                        w: w.clone(),
                        in_h: *in_h,
                        in_w: *in_w,
                        c_in,
                        c_out,
                        kh: *kh,
                        kw: *kw,
                        dtype: sx.dtype,
                        weight_layout: wlayout,
                        implicit_enabled,
                        im2col_scratch,
                        out: node.out.clone(),
                    });
                }
            }
            IrOp::ConcatMatMul { a, weights } => {
                consumed[i] = true;
                let sa = &g.tensors[a];
                let m = sa.rows();
                let k = sa.cols();
                let n_per = g.tensors[&weights[0]].cols();
                let total_n = n_per * weights.len();
                // Repack each weight NK when not horiz-packed; when horiz-packed `CONCAT_NK` packs all of them into ONE buffer!
                if !cfg.horiz_pack {
                    for w in weights {
                        let _ = resolve_weight_layout(w, m, k, n_per, g.tensors[w].dtype, false);
                    }
                }
                let (schedule, _) = solve_tile_schedule(m, k, total_n, sa.dtype, sa.dtype, LayoutTag::ColMajorNK, false, cfg);
                stages.push(TileStage::ConcatContraction {
                    stage_id,
                    name: format!("{}_s{}_concat_mm", g.fn_name, stage_id),
                    a: a.clone(),
                    weights: weights.clone(),
                    m,
                    k,
                    n_per,
                    total_n,
                    dtype: sa.dtype,
                    packed_enabled: cfg.horiz_pack,
                    schedule,
                    out: node.out.clone(),
                });
            }
            IrOp::AdamWStep { param, grad, m, v } => {
                consumed[i] = true;
                let sp = &g.tensors[param];
                let numel = sp.numel();
                let donate_enabled = cfg.buffer_donation;
                let stochastic_round = cfg.stochastic_round && matches!(sp.dtype, IrDType::F16 | IrDType::BF16);
                let streaming_cache = cfg.streaming_cache;
                let zero1_shards = if cfg.zero1_spmd && cfg.spmd_devices > 1 { cfg.spmd_devices } else { 1 };
                if donate_enabled {
                    // Donates param + m + v in-place (ZERO extra output allocation!)
                    donated_bytes_total += sp.alloc_bytes() * 3;
                }
                stages.push(TileStage::InPlaceAdamW {
                    stage_id,
                    name: format!("{}_s{}_adamw", g.fn_name, stage_id),
                    param: param.clone(),
                    grad: grad.clone(),
                    m: m.clone(),
                    v: v.clone(),
                    numel,
                    dtype: sp.dtype,
                    donate_enabled,
                    stochastic_round,
                    streaming_cache,
                    zero1_shards,
                    out: node.out.clone(),
                });
            }
            IrOp::CrossEntropy { logits, targets } => {
                consumed[i] = true;
                let sl = &g.tensors[logits];
                let rows = sl.rows();
                let vocab = sl.cols();
                let online_fused = cfg.online_monoid;
                let donate_enabled = cfg.buffer_donation;
                let streaming_cache = cfg.streaming_cache;
                // Facility #10 (`auto_pad`): align odd vocabulary (e.g. GPT-2 `50257 -> 50304`) to 64 for 128-bit `float4`!
                let padded_vocab = if cfg.auto_pad && vocab % 4 != 0 && input_set.contains(logits) {
                    let pv = (vocab + 63) & !63;
                    if !input_pads.iter().any(|p| p.name == *logits) {
                        input_pads.push(PadAlignSpec { name: logits.clone(), pad_rows: rows, pad_cols: pv, is_nk: false });
                    }
                    pv
                } else {
                    vocab
                };
                if donate_enabled && padded_vocab == vocab {
                    donated_bytes_total += sl.alloc_bytes();
                }
                let scratch_probs = if !online_fused {
                    let sc = format!("__ce_probs_{}", stage_id);
                    extra_scratch_sizes.insert(sc.clone(), rows * vocab * 4);
                    Some(sc)
                } else {
                    None
                };
                stages.push(TileStage::CrossEntropyStage {
                    stage_id,
                    name: format!("{}_s{}_xent", g.fn_name, stage_id),
                    logits: logits.clone(),
                    targets: targets.clone(),
                    rows,
                    vocab,
                    padded_vocab,
                    dtype: sl.dtype,
                    online_fused,
                    donate_enabled,
                    streaming_cache,
                    scratch_probs,
                    out: node.out.clone(),
                });
            }
            IrOp::AmpNormMatMul { x, nw, w } => {
                consumed[i] = true;
                let sx = &g.tensors[x];
                let sw = &g.tensors[w];
                let (m, k, n) = (sx.rows(), sx.cols(), sw.cols());
                let quant_enabled = cfg.amp_dyn_quant;
                let wlayout = if quant_enabled && input_set.contains(w) && k % 8 == 0 && n % 8 == 0 {
                    int8_weights.push(w.clone());
                    LayoutTag::Int8RowScaleNK
                } else {
                    let (_, wl) = resolve_weight_layout(w, m, k, n, sw.dtype, false);
                    wl
                };
                let (schedule, _) = solve_tile_schedule(m, k, n, sx.dtype, sw.dtype, wlayout, false, cfg);
                stages.push(TileStage::AmpQuantContraction {
                    stage_id,
                    name: format!("{}_s{}_amp_mm", g.fn_name, stage_id),
                    x: x.clone(),
                    nw: nw.clone(),
                    w: w.clone(),
                    m,
                    k,
                    n,
                    dtype: sx.dtype,
                    quant_enabled,
                    weight_layout: wlayout,
                    schedule,
                    out: node.out.clone(),
                });
            }
            IrOp::DynMatMul { a, b, max_m } => {
                consumed[i] = true;
                let sa = &g.tensors[a];
                let sb = &g.tensors[b];
                let (k, n) = (sa.cols(), sb.cols());
                let (_, wlayout) = resolve_weight_layout(b, *max_m, k, n, sb.dtype, false);
                stages.push(TileStage::DynBucketContraction {
                    stage_id,
                    name: format!("{}_s{}_dyn_mm", g.fn_name, stage_id),
                    a: a.clone(),
                    b: b.clone(),
                    max_m: *max_m,
                    k,
                    n,
                    dtype: sa.dtype,
                    weight_layout: wlayout,
                    bucket_enabled: cfg.dyn_bucket,
                    tile_swizzle: cfg.tile_swizzle,
                    out: node.out.clone(),
                });
            }
            IrOp::ParallelScan { a, b } => {
                consumed[i] = true;
                let sa = &g.tensors[a];
                stages.push(TileStage::ParallelScanStage {
                    stage_id,
                    name: format!("{}_s{}_scan", g.fn_name, stage_id),
                    a: a.clone(),
                    b: b.clone(),
                    batch: sa.rows(),
                    seq_len: sa.cols(),
                    dtype: sa.dtype,
                    parallel_enabled: cfg.non_tf_ops,
                    out: node.out.clone(),
                });
            }
            IrOp::RaggedMatMul { a, b, valid_pct } => {
                let sa = &g.tensors[a];
                let sb = &g.tensors[b];
                let (m_pad, k, n) = (sa.rows(), sa.cols(), sb.cols());
                let m_valid = ((m_pad * *valid_pct) / 100).max(1);
                consumed[i] = true;
                let (_, wlayout) = resolve_weight_layout(b, m_valid, k, n, sb.dtype, false);
                let eff_m = if cfg.ragged_sparse { m_valid } else { m_pad };
                let (schedule, _) = solve_tile_schedule(eff_m, k, n, sa.dtype, sb.dtype, wlayout, false, cfg);
                stages.push(TileStage::RaggedContraction {
                    stage_id,
                    name: format!("{}_s{}_ragged_mm", g.fn_name, stage_id),
                    a: a.clone(),
                    b: b.clone(),
                    m_padded: m_pad,
                    m_valid,
                    k,
                    n,
                    dtype: sa.dtype,
                    weight_layout: wlayout,
                    compact_enabled: cfg.ragged_sparse,
                    schedule,
                    out: node.out.clone(),
                });
            }
            IrOp::BlockSparseMatMul { a, b, block_size, density_pct } => {
                let sa = &g.tensors[a];
                let sb = &g.tensors[b];
                let (m, k, n) = (sa.rows(), sa.cols(), sb.cols());
                consumed[i] = true;
                // Stage 3: True compressed 32x32 BSR layout (`PACK_BSR32_NK`) eliminates zero blocks from VRAM & DRAM!
                let wlayout = if cfg.ragged_sparse && input_set.contains(b) && *block_size == 32 && k % 32 == 0 && n % 32 == 0 && sb.dtype == IrDType::F16 {
                    bsr32_weights.push(b.clone());
                    LayoutTag::Bsr32NK
                } else {
                    let (_, wl) = resolve_weight_layout(b, m, k, n, sb.dtype, false);
                    wl
                };
                let (schedule, _) = solve_tile_schedule(m, k, n, sa.dtype, sb.dtype, wlayout, false, cfg);
                stages.push(TileStage::BlockSparseContraction {
                    stage_id,
                    name: format!("{}_s{}_sparse_mm", g.fn_name, stage_id),
                    a: a.clone(),
                    b: b.clone(),
                    m,
                    k,
                    n,
                    block_size: *block_size,
                    density_pct: *density_pct,
                    dtype: sa.dtype,
                    weight_layout: wlayout,
                    sparse_enabled: cfg.ragged_sparse,
                    schedule,
                    out: node.out.clone(),
                });
            }
            IrOp::Rope { x: q_in } if cfg.general_fusion => {
                // Fuse RoPE(Q), RoPE(K) + Attention into ONE FlashAttention kernel!
                let rq_var = &node.out;
                let mut fused_attn = false;
                if use_count.get(rq_var).copied().unwrap_or(0) == 1 {
                    let mut k_pair = None;
                    let mut attn_match = None;
                    for j in (i + 1)..g.nodes.len() {
                        if consumed[j] { continue; }
                        match &g.nodes[j].op {
                            IrOp::Rope { x: k_in } if k_pair.is_none() => {
                                k_pair = Some((j, k_in.clone(), g.nodes[j].out.clone()));
                            }
                            IrOp::Attention { q, k, v, q_dim, kv_dim, causal, paged } if q == rq_var => {
                                if let Some((_, _, ref rk_var)) = k_pair {
                                    if k == rk_var {
                                        attn_match = Some((j, v.clone(), *q_dim, *kv_dim, *causal, *paged));
                                        break;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    if let (Some((kj, k_in, _)), Some((aj, v_in, q_dim, kv_dim, causal, paged))) = (k_pair, attn_match) {
                        consumed[i] = true;
                        consumed[kj] = true;
                        consumed[aj] = true;
                        let sq = &g.tensors[q_in];
                        stages.push(TileStage::AttentionStage {
                            stage_id,
                            name: format!("{}_s{}_flash_attn", g.fn_name, stage_id),
                            q: q_in.clone(),
                            k: k_in,
                            v: v_in,
                            seq: sq.rows(),
                            q_dim,
                            kv_dim,
                            dtype: sq.dtype,
                            causal,
                            paged,
                            fused_rope: true,
                            online_tiled: true,
                            use_mma: cfg.tile_mma,
                            out: g.nodes[aj].out.clone(),
                            naive_scores_scratch: None,
                        });
                        fused_attn = true;
                    }
                }
                if !fused_attn {
                    consumed[i] = true;
                    stages.push(TileStage::Elementwise {
                        stage_id,
                        name: format!("{}_s{}_rope", g.fn_name, stage_id),
                        inputs: vec![q_in.clone()],
                        ops: vec![ElemOp::Scale { scale_bits: 1.0f32.to_bits() }],
                        numel: node.desc.numel(),
                        dtype: node.desc.dtype,
                        out: node.out.clone(),
                        vectorized: cfg.tile_mma,
                        streaming_cache: false,
                    });
                }
            }
            IrOp::Attention { q, k, v, q_dim, kv_dim, causal, paged } => {
                consumed[i] = true;
                let sq = &g.tensors[q];
                let seq = sq.rows();
                let online_tiled = cfg.general_fusion;
                let naive_scores_scratch = if !online_tiled {
                    let sc_name = format!("__attn_scores_{}", stage_id);
                    // Naive multi-head attention materializes [n_heads, seq, seq] FP32 scores!
                    let n_heads = (*q_dim / 64).max(1);
                    extra_scratch_sizes.insert(sc_name.clone(), n_heads * seq * seq * 4);
                    Some(sc_name)
                } else {
                    None
                };
                stages.push(TileStage::AttentionStage {
                    stage_id,
                    name: format!("{}_s{}_attn", g.fn_name, stage_id),
                    q: q.clone(),
                    k: k.clone(),
                    v: v.clone(),
                    seq,
                    q_dim: *q_dim,
                    kv_dim: *kv_dim,
                    dtype: sq.dtype,
                    causal: *causal,
                    paged: *paged,
                    fused_rope: false,
                    online_tiled,
                    use_mma: cfg.tile_mma,
                    out: node.out.clone(),
                    naive_scores_scratch,
                });
            }
            IrOp::Reduce { op, x, weight } => {
                consumed[i] = true;
                let sx = &g.tensors[x];
                let (epilogue, final_out) = if cfg.general_fusion {
                    absorb_epilogue_chain(g, &mut consumed, &use_count, &output_set, node.out.clone())
                } else {
                    (Vec::new(), node.out.clone())
                };
                stages.push(TileStage::RowReduce {
                    stage_id,
                    name: format!("{}_s{}_reduce", g.fn_name, stage_id),
                    op: *op,
                    x: x.clone(),
                    weight: weight.clone(),
                    rows: sx.rows(),
                    cols: sx.cols(),
                    dtype: sx.dtype,
                    epilogue,
                    out: final_out,
                    vectorized: cfg.tile_mma,
                });
            }
            IrOp::Unary { op, x } => {
                consumed[i] = true;
                let mut ops = vec![*op];
                let mut cur_out = node.out.clone();
                // Fuse chained unary ops into one elementwise kernel when `general_fusion` ON!
                if cfg.general_fusion {
                    loop {
                        if output_set.contains(&cur_out) || use_count.get(&cur_out).copied().unwrap_or(0) != 1 {
                            break;
                        }
                        let mut found = None;
                        for j in (i + 1)..g.nodes.len() {
                            if !consumed[j] {
                                if let IrOp::Unary { op: next_op, x: nx } = &g.nodes[j].op {
                                    if nx == &cur_out {
                                        found = Some((j, *next_op, g.nodes[j].out.clone()));
                                        break;
                                    }
                                }
                            }
                        }
                        if let Some((j, nop, nout)) = found {
                            consumed[j] = true;
                            ops.push(nop);
                            cur_out = nout;
                        } else {
                            break;
                        }
                    }
                }
                stages.push(TileStage::Elementwise {
                    stage_id,
                    name: format!("{}_s{}_elem", g.fn_name, stage_id),
                    inputs: vec![x.clone()],
                    ops,
                    numel: node.desc.numel(),
                    dtype: node.desc.dtype,
                    out: cur_out,
                    vectorized: cfg.tile_mma,
                    streaming_cache: cfg.streaming_cache && node.desc.numel() >= 16384,
                });
            }
            IrOp::Binary { op, a, b } => {
                consumed[i] = true;
                stages.push(TileStage::Elementwise {
                    stage_id,
                    name: format!("{}_s{}_bin", g.fn_name, stage_id),
                    inputs: vec![a.clone(), b.clone()],
                    ops: vec![*op],
                    numel: node.desc.numel(),
                    dtype: node.desc.dtype,
                    out: node.out.clone(),
                    vectorized: cfg.tile_mma,
                    streaming_cache: cfg.streaming_cache && node.desc.numel() >= 16384,
                });
            }
            IrOp::Rope { x } => {
                consumed[i] = true;
                stages.push(TileStage::Elementwise {
                    stage_id,
                    name: format!("{}_s{}_rope", g.fn_name, stage_id),
                    inputs: vec![x.clone()],
                    ops: vec![ElemOp::Scale { scale_bits: 1.0f32.to_bits() }],
                    numel: node.desc.numel(),
                    dtype: node.desc.dtype,
                    out: node.out.clone(),
                    vectorized: cfg.tile_mma,
                    streaming_cache: false,
                });
            }
            IrOp::Transpose { x } => {
                consumed[i] = true;
                let sx = &g.tensors[x];
                stages.push(TileStage::TransposeStage {
                    stage_id,
                    name: format!("{}_s{}_transpose", g.fn_name, stage_id),
                    x: x.clone(),
                    rows: sx.rows(),
                    cols: sx.cols(),
                    dtype: sx.dtype,
                    out: node.out.clone(),
                });
            }
            IrOp::Collective { kind, x, devices } => {
                consumed[i] = true;
                stages.push(TileStage::CollectiveStage {
                    stage_id,
                    name: format!("{}_s{}_collective", g.fn_name, stage_id),
                    kind: *kind,
                    x: x.clone(),
                    numel: node.desc.numel(),
                    dtype: node.desc.dtype,
                    devices: *devices,
                    out: node.out.clone(),
                });
            }
        }
    }

    // Apply solved physical layouts back to graph tensor descriptors
    for bname in bsr32_weights {
        weight_layouts.insert(bname, LayoutTag::Bsr32NK);
    }
    for iname in int8_weights {
        weight_layouts.insert(iname, LayoutTag::Int8RowScaleNK);
    }
    let mut input_repacks = Vec::new();
    for (p_name, p_desc) in &mut g.params {
        if let Some(&lay) = weight_layouts.get(p_name) {
            p_desc.layout = lay;
            if lay.is_mxfp4() {
                p_desc.dtype = IrDType::MXFP4;
            }
            if let Some(t) = g.tensors.get_mut(p_name) {
                t.layout = lay;
                if lay.is_mxfp4() {
                    t.dtype = IrDType::MXFP4;
                }
            }
            input_repacks.push((p_name.clone(), lay));
        }
    }

    // Build Static Arena Plan via interval graph coloring across all stage intermediates & scratch!
    let mut arena = build_static_arena(&stages, g, &input_set, &output_set, &extra_scratch_sizes, cfg.arena_async);
    arena.donated_bytes = donated_bytes_total;

    (stages, arena, input_repacks, input_pads, cache_policies)
}

/// Absorbs an arbitrary chain of single-use elementwise ops (`+ bias/residual`, `silu`, `gelu`, `relu`, `scale`)
/// straight into the preceding stage's register epilogue!
fn absorb_epilogue_chain(
    g: &TensorIrGraph,
    consumed: &mut [bool],
    use_count: &BTreeMap<String, usize>,
    output_set: &BTreeSet<String>,
    mut cur_var: String,
) -> (Vec<EpilogueOp>, String) {
    let mut ep = Vec::new();
    loop {
        if output_set.contains(&cur_var) || use_count.get(&cur_var).copied().unwrap_or(0) != 1 {
            break;
        }
        let mut matched = None;
        for j in 0..g.nodes.len() {
            if consumed[j] {
                continue;
            }
            match &g.nodes[j].op {
                IrOp::Unary { op, x } if x == &cur_var => {
                    let eop = match op {
                        ElemOp::Silu => Some(EpilogueOp::Silu),
                        ElemOp::Gelu => Some(EpilogueOp::Gelu),
                        ElemOp::Relu => Some(EpilogueOp::Relu),
                        ElemOp::Scale { scale_bits } => Some(EpilogueOp::Scale(f32::from_bits(*scale_bits))),
                        _ => None,
                    };
                    if let Some(e) = eop {
                        matched = Some((j, e, g.nodes[j].out.clone()));
                        break;
                    }
                }
                IrOp::Binary { op: ElemOp::Add, a, b } if a == &cur_var || b == &cur_var => {
                    let other = if a == &cur_var { b.clone() } else { a.clone() };
                    let bcast = g.tensors.get(&other).map(|t| t.rows() == 1).unwrap_or(false);
                    matched = Some((j, EpilogueOp::AddTensor { src: other, broadcast_row: bcast }, g.nodes[j].out.clone()));
                    break;
                }
                IrOp::Binary { op: ElemOp::Sub, a, b } if a == &cur_var => {
                    matched = Some((j, EpilogueOp::SubTensor { src: b.clone() }, g.nodes[j].out.clone()));
                    break;
                }
                IrOp::Binary { op: ElemOp::Mul, a, b } if a == &cur_var || b == &cur_var => {
                    let other = if a == &cur_var { b.clone() } else { a.clone() };
                    matched = Some((j, EpilogueOp::MulTensor { src: other }, g.nodes[j].out.clone()));
                    break;
                }
                IrOp::Binary { op: ElemOp::SiluBwd, a, b } if a == &cur_var => {
                    matched = Some((j, EpilogueOp::SiluBwd { saved_x: b.clone() }, g.nodes[j].out.clone()));
                    break;
                }
                IrOp::Binary { op: ElemOp::GeluBwd, a, b } if a == &cur_var => {
                    matched = Some((j, EpilogueOp::GeluBwd { saved_x: b.clone() }, g.nodes[j].out.clone()));
                    break;
                }
                IrOp::Binary { op: ElemOp::ReluBwd, a, b } if a == &cur_var => {
                    matched = Some((j, EpilogueOp::ReluBwd { saved_x: b.clone() }, g.nodes[j].out.clone()));
                    break;
                }
                _ => {}
            }
        }
        if let Some((j, eop, next_var)) = matched {
            consumed[j] = true;
            ep.push(eop);
            cur_var = next_var;
        } else {
            break;
        }
    }
    (ep, cur_var)
}

/// Hardware-aware Tile-IR Schedule & PGO Candidate Solver (`TargetSpec::M4_MAX` / `RTX_4090`).
/// Accounts for register pressure (`has_dual` doubles accumulator footprint) and 32 KB SRAM occupancy!
pub fn solve_tile_schedule(
    m: usize,
    k: usize,
    n: usize,
    _act_dtype: IrDType,
    _weight_dtype: IrDType,
    layout: LayoutTag,
    has_dual: bool,
    cfg: GpuOptConfig,
) -> (TileSchedule, Vec<TileSchedule>) {
    if !cfg.tile_mma {
        let sched = TileSchedule {
            bm: 16,
            bn: 16,
            bk: 1,
            num_sgs: 4,
            rows_per_sg: 1,
            split_k: 1,
            double_buf: false,
            swizzle_pad: 0,
            use_mma: false,
            stream_k: false,
            warp_specialized: false,
            tile_swizzle: false,
        };
        return (sched, vec![sched]);
    }

    let swizzle_pad = if cfg.layout_prop { 8 } else { 0 };
    let double_buf = cfg.warp_spec;

    // Memory-bound regime (`M <= 8`): 128-bit coalesced SIMD-group GEMV schedule
    // Supports multi-row shared-activation GEMV (`rows_per_sg = 1, 2, 4`) so activation loads are amortized
    // and multiple independent DRAM streams stay in flight per SIMD-group!
    if m <= 8 {
        let def_rps = if layout.is_mxfp4() && !has_dual && n >= 256 && n % 2 == 0 { 2 } else { 1 };
        let n_tiles = (n + 15) / 16;
        let split_k = if cfg.stream_k && k >= 1024 && n_tiles < 128 {
            if k >= 4096 && n_tiles < 64 { 4 } else { 2 }
        } else {
            1
        };
        let sched = TileSchedule {
            bm: m,
            bn: 16,
            bk: 32,
            num_sgs: 4,
            rows_per_sg: def_rps,
            split_k,
            double_buf,
            swizzle_pad,
            use_mma: true,
            stream_k: cfg.stream_k && split_k > 1,
            warp_specialized: cfg.warp_spec,
            tile_swizzle: false,
        };
        let mut cands = vec![sched];
        if cfg.pgo_autotune {
            cands.push(TileSchedule { bn: 8, num_sgs: 4, rows_per_sg: 1, split_k: 1, stream_k: false, ..sched });
            cands.push(TileSchedule { bn: 16, num_sgs: 8, rows_per_sg: 1, split_k: 1, stream_k: false, ..sched });
            if !has_dual && n % 2 == 0 {
                cands.push(TileSchedule { bn: 16, num_sgs: 4, rows_per_sg: 2, split_k: 1, stream_k: false, ..sched });
                cands.push(TileSchedule { bn: 32, num_sgs: 8, rows_per_sg: 2, split_k: 1, stream_k: false, ..sched });
            }
            if !has_dual && n % 4 == 0 && layout.is_mxfp4() {
                cands.push(TileSchedule { bn: 32, num_sgs: 4, rows_per_sg: 4, split_k: 1, stream_k: false, ..sched });
                if cfg.stream_k && k >= 1024 {
                    cands.push(TileSchedule { bn: 32, num_sgs: 4, rows_per_sg: 4, split_k: 2, stream_k: true, ..sched });
                }
            }
            if cfg.stream_k && k >= 1024 {
                cands.push(TileSchedule { bn: 16, num_sgs: 4, rows_per_sg: def_rps, split_k: 2, stream_k: true, ..sched });
            }
        }
        return (sched, dedup_schedules(cands));
    }

    // Compute-bound / Batch regime (`M >= 9`): Hardware AMX `simdgroup_half8x8` / Tensor Core Tiled MMA!
    // Register-pressure & SRAM-occupancy model:
    //   - `has_dual`: 2x accumulators (`mc` + `mc2`), so `32x32 (4SG, 8 acc)` prevents register spill!
    //   - `double_buf` (`warp_spec`): `BK=16` keeps 2x ping-pong SMEM at 12 KB (2 TGs/core concurrent!)
    //   - Single-buf: `64x64x32 (8SG)` or `128x64x16 (8SG, 9.2 KB SMEM, 42.7 FLOPs/byte!)`
    let (bm, bn, bk, num_sgs) = if let Some((fm, fn_, fk)) = cfg.force_tile {
        (fm, fn_, fk, if fm >= 64 && fn_ >= 64 { 8 } else { 4 })
    } else if has_dual {
        (32, 32, 32, 4)
    } else if m >= 64 && n >= 64 {
        if double_buf {
            (64, 64, 16, 4) // 12 KB double-buf SMEM -> 2 TGs/core + 2.0 mma/load ratio!
        } else {
            (64, 64, 32, 8)
        }
    } else {
        (32, 32, 32, 4)
    };
    let tiles_total = ((m + bm - 1) / bm) * ((n + bn - 1) / bn);
    let split_k = if cfg.stream_k && k >= 1024 && tiles_total < 80 {
        if tiles_total < 24 && k >= 2048 { 4 } else { 2 }
    } else {
        1
    };

    let default_sched = TileSchedule {
        bm,
        bn,
        bk,
        num_sgs,
        rows_per_sg: 1,
        split_k,
        double_buf,
        swizzle_pad,
        use_mma: true,
        stream_k: cfg.stream_k && split_k > 1,
        warp_specialized: cfg.warp_spec,
        tile_swizzle: cfg.tile_swizzle,
    };

    let mut cands = vec![default_sched];
    if cfg.pgo_autotune {
        // High-value PGO candidate space covering occupancy vs arithmetic-intensity Pareto frontier!
        cands.push(TileSchedule { bm: 32, bn: 32, bk: 32, num_sgs: 4, double_buf: false, split_k: 1, stream_k: false, ..default_sched });
        if m >= 64 && n >= 64 && !has_dual {
            cands.push(TileSchedule { bm: 64, bn: 64, bk: 16, num_sgs: 4, double_buf: false, split_k: 1, stream_k: false, ..default_sched });
            cands.push(TileSchedule { bm: 64, bn: 64, bk: 16, num_sgs: 4, double_buf: true, split_k: 1, stream_k: false, ..default_sched });
            cands.push(TileSchedule { bm: 64, bn: 64, bk: 32, num_sgs: 4, double_buf: false, split_k: 1, stream_k: false, ..default_sched });
            cands.push(TileSchedule { bm: 64, bn: 64, bk: 32, num_sgs: 8, double_buf: false, split_k: 1, stream_k: false, ..default_sched });
            if m >= 128 && n >= 128 {
                // High arithmetic-intensity 8-SG tiles (`42.7 FLOPs/byte`, only `9.2 KB` single-buf / `18.4 KB` double-buf SMEM!)
                cands.push(TileSchedule { bm: 128, bn: 64, bk: 16, num_sgs: 8, double_buf: false, split_k: 1, stream_k: false, ..default_sched });
                cands.push(TileSchedule { bm: 64, bn: 128, bk: 16, num_sgs: 8, double_buf: false, split_k: 1, stream_k: false, ..default_sched });
            }
        } else if m >= 64 && n >= 64 && has_dual {
            cands.push(TileSchedule { bm: 32, bn: 32, bk: 16, num_sgs: 4, double_buf: false, split_k: 1, stream_k: false, ..default_sched });
            cands.push(TileSchedule { bm: 64, bn: 64, bk: 32, num_sgs: 8, double_buf: false, split_k: 1, stream_k: false, ..default_sched });
        }
        if cfg.stream_k && k >= 1024 && tiles_total < 80 {
            cands.push(TileSchedule { bm: 32, bn: 32, bk: 32, num_sgs: 4, double_buf: false, split_k: 2, stream_k: true, ..default_sched });
        }
    }

    (default_sched, dedup_schedules(cands))
}

fn dedup_schedules(v: Vec<TileSchedule>) -> Vec<TileSchedule> {
    let mut out = Vec::new();
    for s in v {
        if !out.contains(&s) {
            out.push(s);
        }
    }
    out
}

/// Static Liveness Graph-Coloring Arena Allocator (256-byte aligned for Metal/CUDA/TPU DMA!).
fn build_static_arena(
    stages: &[TileStage],
    g: &TensorIrGraph,
    input_set: &BTreeSet<String>,
    output_set: &BTreeSet<String>,
    extra_scratch: &BTreeMap<String, usize>,
    enabled: bool,
) -> StaticArenaPlan {
    // Collect `[first_def, last_use]` for every non-input, non-output intermediate buffer
    let mut intervals: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();

    for (idx, st) in stages.iter().enumerate() {
        let mut outs = Vec::new();
        let mut ins = Vec::new();
        collect_stage_io(st, &mut ins, &mut outs);

        for o in &outs {
            if !input_set.contains(o) && !output_set.contains(o) {
                let sz = extra_scratch
                    .get(o)
                    .copied()
                    .or_else(|| g.tensors.get(o).map(|t| t.alloc_bytes()))
                    .unwrap_or(256);
                let aligned = (sz + 255) & !255;
                intervals.entry(o.clone()).or_insert((idx, idx, aligned));
            }
        }
        for inp in &ins {
            if let Some(entry) = intervals.get_mut(inp) {
                entry.1 = entry.1.max(idx);
            }
        }
        // Scratch buffers live ONLY during their own stage `[idx, idx]`!
        if let TileStage::Contraction { split_k_scratch: Some(sc), .. } = st {
            let sz = extra_scratch.get(sc).copied().unwrap_or(4096);
            intervals.insert(sc.clone(), (idx, idx, (sz + 255) & !255));
        }
        if let TileStage::AttentionStage { naive_scores_scratch: Some(sc), .. } = st {
            let sz = extra_scratch.get(sc).copied().unwrap_or(4096);
            intervals.insert(sc.clone(), (idx, idx, (sz + 255) & !255));
        }
        if let TileStage::CrossEntropyStage { scratch_probs: Some(sc), .. } = st {
            let sz = extra_scratch.get(sc).copied().unwrap_or(4096);
            intervals.insert(sc.clone(), (idx, idx, (sz + 255) & !255));
        }
        if let TileStage::Conv2dStage { im2col_scratch: Some(sc), .. } = st {
            let sz = extra_scratch.get(sc).copied().unwrap_or(4096);
            intervals.insert(sc.clone(), (idx, idx, (sz + 255) & !255));
        }
    }

    let naive_total: usize = intervals.values().map(|(_, _, sz)| *sz).sum();
    let mut allocations = BTreeMap::new();

    if !enabled {
        // Unoptimized bump layout (no reuse!)
        let mut cur = 0usize;
        for (name, (_, _, sz)) in &intervals {
            allocations.insert(name.clone(), (cur, *sz));
            cur += *sz;
        }
        return StaticArenaPlan {
            enabled: false,
            total_arena_bytes: cur.max(256),
            naive_total_bytes: naive_total,
            vram_saved_pct: 0.0,
            donated_bytes: 0,
            allocations,
        };
    }

    // Greedy interval coloring (sort by descending size for tightest Arena packing)
    let mut sorted: Vec<(String, usize, usize, usize)> = intervals
        .into_iter()
        .map(|(k, (s, e, sz))| (k, s, e, sz))
        .collect();
    sorted.sort_by(|a, b| b.3.cmp(&a.3).then(a.1.cmp(&b.1)));

    // Placed intervals: `(offset, size, start_stage, end_stage)`
    let mut placed: Vec<(usize, usize, usize, usize)> = Vec::new();
    let mut peak_bytes = 0usize;

    for (name, s, e, sz) in sorted {
        let mut cand_off = 0usize;
        loop {
            let conflict = placed.iter().find(|&&(po, psz, ps, pe)| {
                let time_overlap = !(e < ps || s > pe);
                let mem_overlap = !(cand_off + sz <= po || cand_off >= po + psz);
                time_overlap && mem_overlap
            });
            if let Some(&(po, psz, _, _)) = conflict {
                cand_off = (po + psz + 255) & !255;
            } else {
                break;
            }
        }
        placed.push((cand_off, sz, s, e));
        allocations.insert(name, (cand_off, sz));
        peak_bytes = peak_bytes.max(cand_off + sz);
    }

    let saved_pct = if naive_total > 0 {
        ((naive_total.saturating_sub(peak_bytes)) as f64 / naive_total as f64) * 100.0
    } else {
        0.0
    };

    StaticArenaPlan {
        enabled: true,
        total_arena_bytes: peak_bytes.max(256),
        naive_total_bytes: naive_total,
        vram_saved_pct: saved_pct,
        donated_bytes: 0,
        allocations,
    }
}

pub fn collect_stage_io(st: &TileStage, ins: &mut Vec<String>, outs: &mut Vec<String>) {
    match st {
        TileStage::Contraction { a, b, prologue, dual_b, extra_siblings, epilogue, out, .. } => {
            ins.push(a.clone());
            ins.push(b.clone());
            if let Some(p) = prologue {
                ins.push(p.x.clone());
                ins.push(p.weight.clone());
            }
            if let Some(db) = dual_b {
                ins.push(db.clone());
            }
            for (ws, os, _) in extra_siblings {
                ins.push(ws.clone());
                outs.push(os.clone());
            }
            for ep in epilogue {
                match ep {
                    EpilogueOp::AddTensor { src, .. }
                    | EpilogueOp::SubTensor { src }
                    | EpilogueOp::MulTensor { src }
                    | EpilogueOp::SiluBwd { saved_x: src }
                    | EpilogueOp::GeluBwd { saved_x: src }
                    | EpilogueOp::ReluBwd { saved_x: src } => ins.push(src.clone()),
                    _ => {}
                }
            }
            outs.push(out.clone());
        }
        TileStage::RowReduce { x, weight, epilogue, out, .. } => {
            ins.push(x.clone());
            if let Some(w) = weight { ins.push(w.clone()); }
            for ep in epilogue {
                match ep {
                    EpilogueOp::AddTensor { src, .. }
                    | EpilogueOp::SubTensor { src }
                    | EpilogueOp::MulTensor { src }
                    | EpilogueOp::SiluBwd { saved_x: src }
                    | EpilogueOp::GeluBwd { saved_x: src }
                    | EpilogueOp::ReluBwd { saved_x: src } => ins.push(src.clone()),
                    _ => {}
                }
            }
            outs.push(out.clone());
        }
        TileStage::Elementwise { inputs, out, .. } => {
            ins.extend(inputs.iter().cloned());
            outs.push(out.clone());
        }
        TileStage::AttentionStage { q, k, v, out, .. } => {
            ins.push(q.clone());
            ins.push(k.clone());
            ins.push(v.clone());
            outs.push(out.clone());
        }
        TileStage::RaggedContraction { a, b, out, .. }
        | TileStage::BlockSparseContraction { a, b, out, .. } => {
            ins.push(a.clone());
            ins.push(b.clone());
            outs.push(out.clone());
        }
        TileStage::TransposeStage { x, out, .. } | TileStage::CollectiveStage { x, out, .. } => {
            ins.push(x.clone());
            outs.push(out.clone());
        }
        TileStage::ConcatContraction { a, weights, out, .. } => {
            ins.push(a.clone());
            ins.extend(weights.iter().cloned());
            outs.push(out.clone());
        }
        TileStage::InPlaceAdamW { param, grad, m, v, out, .. } => {
            ins.push(param.clone());
            ins.push(grad.clone());
            ins.push(m.clone());
            ins.push(v.clone());
            outs.push(out.clone());
        }
        TileStage::CrossEntropyStage { logits, targets, out, .. } => {
            ins.push(logits.clone());
            ins.push(targets.clone());
            outs.push(out.clone());
        }
        TileStage::AmpQuantContraction { x, nw, w, out, .. } => {
            ins.push(x.clone());
            ins.push(nw.clone());
            ins.push(w.clone());
            outs.push(out.clone());
        }
        TileStage::DynBucketContraction { a, b, out, .. }
        | TileStage::Conv2dStage { x: a, w: b, out, .. }
        | TileStage::ParallelScanStage { a, b, out, .. } => {
            ins.push(a.clone());
            ins.push(b.clone());
            outs.push(out.clone());
        }
    }
}

pub fn estimate_fused_bytes(stages: &[TileStage]) -> f64 {
    let mut total = 0.0;
    for st in stages {
        match st {
            TileStage::Contraction { m, k, n, act_dtype, weight_dtype, dual_b, extra_siblings, epilogue, .. } => {
                let ab = act_dtype.bytes_f64();
                let wb = weight_dtype.bytes_f64();
                let mut total_n = *n;
                if dual_b.is_some() { total_n += *n; }
                for (_, _, sn) in extra_siblings { total_n += *sn; }
                let ep_reads: f64 = epilogue.iter().map(|e| match e {
                    EpilogueOp::AddTensor { broadcast_row: false, .. }
                    | EpilogueOp::SubTensor { .. }
                    | EpilogueOp::MulTensor { .. }
                    | EpilogueOp::SiluBwd { .. }
                    | EpilogueOp::GeluBwd { .. }
                    | EpilogueOp::ReluBwd { .. } => (*m * *n) as f64 * ab,
                    EpilogueOp::AddTensor { broadcast_row: true, .. } => (*n) as f64 * ab,
                    _ => 0.0,
                }).sum();
                total += (*m * *k) as f64 * ab + (*k * total_n) as f64 * wb + (*m * *n) as f64 * ab + ep_reads;
            }
            TileStage::RaggedContraction { m_padded, m_valid, k, n, dtype, compact_enabled, .. } => {
                let m = if *compact_enabled { *m_valid } else { *m_padded };
                let b = dtype.bytes_f64();
                total += ((m * k + k * n + m * n) as f64) * b;
            }
            TileStage::BlockSparseContraction { m, k, n, density_pct, dtype, sparse_enabled, .. } => {
                let b = dtype.bytes_f64();
                let f = if *sparse_enabled { *density_pct as f64 / 100.0 } else { 1.0 };
                total += ((m * k + m * n) as f64 + (k * n) as f64 * f) * b;
            }
            TileStage::AttentionStage { seq, q_dim, kv_dim, dtype, online_tiled, .. } => {
                let b = dtype.bytes_f64();
                let base = (*seq * (*q_dim * 2 + *kv_dim * 2)) as f64 * b;
                let s2 = if *online_tiled { 0.0 } else { 2.0 * (*seq * *seq) as f64 * 4.0 * ((*q_dim / 64).max(1) as f64) };
                total += base + s2;
            }
            TileStage::RowReduce { rows, cols, dtype, .. } => {
                total += 2.0 * (*rows * *cols) as f64 * dtype.bytes_f64();
            }
            TileStage::Elementwise { inputs, numel, dtype, .. } => {
                total += ((inputs.len() + 1) * *numel) as f64 * dtype.bytes_f64();
            }
            TileStage::TransposeStage { rows, cols, dtype, .. } => {
                total += 2.0 * (*rows * *cols) as f64 * dtype.bytes_f64();
            }
            TileStage::CollectiveStage { numel, dtype, .. } => {
                total += 2.0 * (*numel as f64) * dtype.bytes_f64();
            }
            TileStage::ConcatContraction { m, k, total_n, dtype, .. } => {
                let b = dtype.bytes_f64();
                total += ((*m * *k + *k * *total_n + *m * *total_n) as f64) * b;
            }
            TileStage::InPlaceAdamW { numel, dtype, donate_enabled, .. } => {
                let mult = if *donate_enabled { 7.0 } else { 10.0 };
                total += (*numel as f64) * dtype.bytes_f64() * mult;
            }
            TileStage::CrossEntropyStage { rows, vocab, dtype, online_fused, .. } => {
                let mult = if *online_fused { 2.0 } else { 5.0 };
                total += (*rows * *vocab) as f64 * dtype.bytes_f64() * mult;
            }
            TileStage::AmpQuantContraction { m, k, n, dtype, quant_enabled, .. } => {
                let ab = dtype.bytes_f64();
                let wb = if *quant_enabled { 1.0 } else { ab };
                total += (*m * *k) as f64 * ab + (*k * *n) as f64 * wb + (*m * *n) as f64 * ab;
            }
            TileStage::DynBucketContraction { max_m, k, n, dtype, .. } => {
                let b = dtype.bytes_f64();
                total += ((*max_m * *k + *k * *n + *max_m * *n) as f64) * b;
            }
            TileStage::Conv2dStage { in_h, in_w, c_in, c_out, kh, kw, dtype, implicit_enabled, .. } => {
                let b = dtype.bytes_f64();
                let m = (*in_h) * (*in_w);
                let im2col_b = if *implicit_enabled { 0.0 } else { 2.0 * (m * *c_in * *kh * *kw) as f64 * b };
                total += ((m * *c_in + *c_in * *kh * *kw * *c_out + m * *c_out) as f64) * b + im2col_b;
            }
            TileStage::ParallelScanStage { batch, seq_len, dtype, .. } => {
                total += 3.0 * (*batch * *seq_len) as f64 * dtype.bytes_f64();
            }
        }
    }
    total
}

fn bump_node_uses(op: &IrOp, uses: &mut BTreeMap<String, usize>) {
    let mut b = |s: &str| *uses.entry(s.to_string()).or_default() += 1;
    match op {
        IrOp::MatMul { a, b: wb, .. }
        | IrOp::Conv2d { x: a, w: wb, .. }
        | IrOp::RaggedMatMul { a, b: wb, .. }
        | IrOp::BlockSparseMatMul { a, b: wb, .. }
        | IrOp::DynMatMul { a, b: wb, .. }
        | IrOp::ParallelScan { a, b: wb }
        | IrOp::CrossEntropy { logits: a, targets: wb }
        | IrOp::Binary { a, b: wb, .. } => {
            b(a);
            b(wb);
        }
        IrOp::ConcatMatMul { a, weights } => {
            b(a);
            for w in weights { b(w); }
        }
        IrOp::AdamWStep { param, grad, m, v } => {
            b(param); b(grad); b(m); b(v);
        }
        IrOp::AmpNormMatMul { x, nw, w } => {
            b(x); b(nw); b(w);
        }
        IrOp::Unary { x, .. } | IrOp::Rope { x } | IrOp::Transpose { x } | IrOp::Collective { x, .. } => b(x),
        IrOp::Reduce { x, weight, .. } => {
            b(x);
            if let Some(w) = weight { b(w); }
        }
        IrOp::Attention { q, k, v, .. } => {
            b(q);
            b(k);
            b(v);
        }
    }
}
