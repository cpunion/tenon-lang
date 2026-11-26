//! Tenon Universal Tile-IR & Multi-Stage Hardware-Ceiling Compiler (`tile_ir`).
//!
//! Architecture-agnostic, operator-agnostic declarative compiler pipeline that lowers ANY
//! `@gpu_graph` dependent-typed tensor dataflow into near-silicon-ceiling code across:
//!   - Apple Silicon Metal (`simdgroup_half8x8` AMX, 128-bit coalesced SIMD, XOR swizzle, Single-CB)
//!   - NVIDIA CUDA / PTX (`sm_89` FP8/FP16 MMA + `cp.async`, `sm_90` TMA/WGMMA Warp-Specialization)
//!   - Google Cloud TPU (StableHLO MLIR, Algebraic Fused RMSNorm+INT8, Online FlashAttn, Buffer Donation, SPMD)
//!
//! Every optimization pass across Stage 1, Stage 2, and Stage 3 is individually toggleable via
//! `GpuOptConfig` (`--gpu-opt`, `TENON_GPU_OPT`, or `gpu_embedded_*_opt("...")`) for strict
//! ablation verification!

pub mod ad_spmd;
pub mod egraph;
pub mod emit_cuda_tpu;
pub mod emit_metal;
pub mod fusion;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use tenon_syntax::*;

/// Per-pass toggleable compiler optimization switches spanning Stages 1, 2, and 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuOptConfig {
    // --- Stage 1: Foundational 6 Pillars ---
    /// Pillar 1: Tile-IR hardware matrix/vector lowering (`simdgroup_half8x8` AMX / `mma.sync` / MXU + 128-bit vec).
    pub tile_mma: bool,
    /// Pillar 2: Universal producer-consumer / prologue / epilogue / N-way horizontal / online-reduction fusion.
    pub general_fusion: bool,
    /// Pillar 3: Whole-graph layout propagation (`W[N,K]` coalesced repack + bank-conflict XOR swizzle).
    pub layout_prop: bool,
    /// Pillar 4: Stream-K / adaptive Split-K load balancing for short-fat shapes.
    pub stream_k: bool,
    /// Pillar 5: Static liveness Arena allocation + in-place aliasing + single-CB/Graph/PJRT async execution plan.
    pub arena_async: bool,
    /// Pillar 6: Compile-time Reverse-Mode AD (VJP + checkpointing) & declarative SPMD auto-sharding.
    pub ad_and_spmd: bool,

    // --- Stage 2: Squeezing the Last 10%-25% ---
    /// PGO live hardware auto-tuner over pruned Tile-IR candidates + persistent `.tenon-gpu-cache`.
    pub pgo_autotune: bool,
    /// Producer/Consumer double-buffered ping-pong pipeline (`sA[2]`/`sB[2]`, Hopper `sm_90` TMA+WGMMA roles).
    pub warp_spec: bool,
    /// E-Graph equality saturation & algebraic superoptimization (matrix-chain reassociation, scale commutation).
    pub egraph_rewrite: bool,

    // --- Stage 3: Breaking Physical Walls & Dynamic Production ---
    /// First-class Ragged / Paged / Block-Sparse lowering + device-side persistent work-stealing queue.
    pub ragged_sparse: bool,
    /// Microscaling MXFP4 (E2M1 + shared E8M0 per 32 elems) block-scaled numerics & mixed-precision propagation.
    pub mxfp4_quant: bool,

    /// Optional SPMD device count (`1` = single device, `>1` = auto-sharded multi-device SPMD).
    pub spmd_devices: usize,
    /// Optional explicit tile override `(BM, BN, BK)` for deterministic ablation/tuning.
    pub force_tile: Option<(usize, usize, usize)>,
}

impl Default for GpuOptConfig {
    fn default() -> Self {
        Self::all()
    }
}

impl GpuOptConfig {
    pub fn none() -> Self {
        Self {
            tile_mma: false,
            general_fusion: false,
            layout_prop: false,
            stream_k: false,
            arena_async: false,
            ad_and_spmd: false,
            pgo_autotune: false,
            warp_spec: false,
            egraph_rewrite: false,
            ragged_sparse: false,
            mxfp4_quant: false,
            spmd_devices: 1,
            force_tile: None,
        }
    }

    pub fn stage1() -> Self {
        Self {
            tile_mma: true,
            general_fusion: true,
            layout_prop: true,
            stream_k: true,
            arena_async: true,
            ad_and_spmd: true,
            pgo_autotune: false,
            warp_spec: false,
            egraph_rewrite: false,
            ragged_sparse: false,
            mxfp4_quant: false,
            spmd_devices: 1,
            force_tile: None,
        }
    }

    pub fn stage2() -> Self {
        let mut c = Self::stage1();
        c.pgo_autotune = true;
        c.warp_spec = true;
        c.egraph_rewrite = true;
        c
    }

    pub fn all() -> Self {
        let mut c = Self::stage2();
        c.ragged_sparse = true;
        c.mxfp4_quant = true;
        c
    }

    /// Parses comma-separated preset & toggle specs, e.g.:
    ///   `"none"`, `"stage1"`, `"stage2"`, `"all"`,
    ///   `"none,+tile_mma,+layout_prop"`, `"all,-warp_spec,-pgo_autotune"`,
    ///   `"all,spmd=4"`, `"tile_mma,layout_prop,fusion"`.
    pub fn parse_spec(spec: &str) -> Self {
        let trimmed = spec.trim();
        if trimmed.is_empty() {
            return Self::all();
        }
        let tokens: Vec<&str> = trimmed
            .split(|c| c == ',' || c == ';' || c == ' ')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if tokens.is_empty() {
            return Self::all();
        }

        // If first token is a preset, start from it; if it starts with '-', start from `all`;
        // otherwise start from `none` so `"tile_mma,layout_prop"` enables ONLY those passes!
        let mut cfg = match tokens[0] {
            "none" | "baseline" | "O0" => Self::none(),
            "stage1" | "s1" => Self::stage1(),
            "stage2" | "s2" => Self::stage2(),
            "stage3" | "s3" | "all" | "O3" => Self::all(),
            t if t.starts_with('-') => Self::all(),
            _ => Self::none(),
        };

        for tok in tokens {
            match tok {
                "none" | "baseline" | "O0" => cfg = Self::none(),
                "stage1" | "s1" => cfg = Self::stage1(),
                "stage2" | "s2" => cfg = Self::stage2(),
                "stage3" | "s3" | "all" | "O3" => cfg = Self::all(),
                _ => {
                    if let Some(rest) = tok.strip_prefix("spmd=") {
                        if let Ok(n) = rest.parse::<usize>() {
                            cfg.spmd_devices = n.max(1);
                            if n > 1 {
                                cfg.ad_and_spmd = true;
                            }
                        }
                        continue;
                    }
                    let (enable, key) = if let Some(k) = tok.strip_prefix('-') {
                        (false, k)
                    } else if let Some(k) = tok.strip_prefix('+') {
                        (true, k)
                    } else {
                        (true, tok)
                    };
                    match key {
                        "tile_mma" | "tile" | "mma" => cfg.tile_mma = enable,
                        "general_fusion" | "fusion" => cfg.general_fusion = enable,
                        "layout_prop" | "layout" | "swizzle" => cfg.layout_prop = enable,
                        "stream_k" | "streamk" | "split_k" => cfg.stream_k = enable,
                        "arena_async" | "arena" | "async" | "graph" => cfg.arena_async = enable,
                        "ad_and_spmd" | "ad" | "spmd" => cfg.ad_and_spmd = enable,
                        "pgo_autotune" | "pgo" | "autotune" => cfg.pgo_autotune = enable,
                        "warp_spec" | "warpspec" | "double_buf" => cfg.warp_spec = enable,
                        "egraph_rewrite" | "egraph" | "algebraic" => cfg.egraph_rewrite = enable,
                        "ragged_sparse" | "ragged" | "sparse" => cfg.ragged_sparse = enable,
                        "mxfp4_quant" | "mxfp4" | "fp4" => cfg.mxfp4_quant = enable,
                        _ => {}
                    }
                }
            }
        }
        cfg
    }

    pub fn from_env() -> Self {
        match std::env::var("TENON_GPU_OPT") {
            Ok(s) if !s.trim().is_empty() => Self::parse_spec(&s),
            _ => Self::all(),
        }
    }

    pub fn summary_tag(&self) -> String {
        let mut on = Vec::new();
        if self.tile_mma { on.push("tile_mma"); }
        if self.general_fusion { on.push("fusion"); }
        if self.layout_prop { on.push("layout"); }
        if self.stream_k { on.push("stream_k"); }
        if self.arena_async { on.push("arena_async"); }
        if self.ad_and_spmd { on.push("ad_spmd"); }
        if self.pgo_autotune { on.push("pgo"); }
        if self.warp_spec { on.push("warp_spec"); }
        if self.egraph_rewrite { on.push("egraph"); }
        if self.ragged_sparse { on.push("ragged_sparse"); }
        if self.mxfp4_quant { on.push("mxfp4"); }
        if self.spmd_devices > 1 {
            on.push("spmd_multi");
        }
        if on.is_empty() {
            "none(scalar)".to_string()
        } else {
            on.join("+")
        }
    }
}

/// Hardware target capability descriptor driving Tile-IR tiling, SRAM budgeting, and Roofline math.
#[derive(Debug, Clone, Copy)]
pub struct TargetSpec {
    pub name: &'static str,
    pub nominal_bw_gbs: f64,
    pub real_dram_bw_gbs: f64,
    pub peak_fp16_tflops: f64,
    pub peak_fp8_or_i8_tflops: f64,
    pub sram_per_block_bytes: usize,
    pub core_or_sm_count: usize,
    pub simd_width: usize,
    pub mma_m: usize,
    pub mma_n: usize,
    pub mma_k: usize,
    pub supports_tma_wgmma: bool,
}

impl TargetSpec {
    pub const M4_MAX: Self = Self {
        name: "Apple M4 Max",
        nominal_bw_gbs: 546.0,
        real_dram_bw_gbs: 468.4,
        // 40 GPU cores x 128 ALUs x 2 FMA x 1.58 GHz = 16.2 TFLOPS (Apple official MPS 4096^3 FP16 = 14.80 TFLOPS)
        peak_fp16_tflops: 16.2,
        peak_fp8_or_i8_tflops: 16.2,
        sram_per_block_bytes: 32 * 1024,
        core_or_sm_count: 40,
        simd_width: 32,
        mma_m: 8,
        mma_n: 8,
        mma_k: 8,
        supports_tma_wgmma: false,
    };

    pub const RTX_4090: Self = Self {
        name: "NVIDIA RTX 4090 sm_89",
        nominal_bw_gbs: 1008.1,
        real_dram_bw_gbs: 935.0,
        peak_fp16_tflops: 330.3,
        peak_fp8_or_i8_tflops: 660.6,
        sram_per_block_bytes: 100 * 1024,
        core_or_sm_count: 128,
        simd_width: 32,
        mma_m: 16,
        mma_n: 8,
        mma_k: 16,
        supports_tma_wgmma: false,
    };

    pub const H100_SXM: Self = Self {
        name: "NVIDIA H100 SXM sm_90",
        nominal_bw_gbs: 3350.0,
        real_dram_bw_gbs: 3050.0,
        peak_fp16_tflops: 989.0,
        peak_fp8_or_i8_tflops: 1979.0,
        sram_per_block_bytes: 228 * 1024,
        core_or_sm_count: 132,
        simd_width: 32,
        mma_m: 64,
        mma_n: 64,
        mma_k: 32,
        supports_tma_wgmma: true,
    };

    pub const TPU_V5E: Self = Self {
        name: "Google Cloud TPU v5e-1",
        nominal_bw_gbs: 819.0,
        real_dram_bw_gbs: 775.0,
        peak_fp16_tflops: 197.0,
        peak_fp8_or_i8_tflops: 394.0,
        sram_per_block_bytes: 128 * 1024 * 1024, // 128 MB VMEM/HBM-tiled
        core_or_sm_count: 4, // 4 MXUs on v5e
        simd_width: 128,
        mma_m: 128,
        mma_n: 128,
        mma_k: 128,
        supports_tma_wgmma: true,
    };
}

/// Element numeric type (extended with Stage 3 OCP Microscaling `MXFP4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IrDType {
    F32,
    F16,
    BF16,
    FP8,
    I8,
    I4,
    Q8_0,
    Q4_0,
    /// OCP Microscaling FP4: 32 E2M1 4-bit elements + 1 E8M0 shared exponent = 17 bytes / 32 elems (0.53125 B/elem).
    MXFP4,
}

impl IrDType {
    pub fn bytes_f64(self) -> f64 {
        match self {
            Self::F32 => 4.0,
            Self::F16 | Self::BF16 => 2.0,
            Self::FP8 | Self::I8 => 1.0,
            Self::I4 => 0.5,
            Self::Q8_0 => 34.0 / 32.0,
            Self::Q4_0 => 18.0 / 32.0,
            Self::MXFP4 => 17.0 / 32.0,
        }
    }

    pub fn is_sub_byte(self) -> bool {
        matches!(self, Self::I4 | Self::Q4_0 | Self::MXFP4)
    }

    pub fn msl_ty(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::F16 => "half",
            Self::BF16 => "bfloat",
            Self::FP8 | Self::I4 | Self::Q8_0 | Self::Q4_0 | Self::MXFP4 => "uchar",
            Self::I8 => "char",
        }
    }

    pub fn cuda_ty(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::F16 => "__half",
            Self::BF16 => "__nv_bfloat16",
            Self::I8 => "int8_t",
            Self::FP8 | Self::I4 | Self::Q8_0 | Self::Q4_0 | Self::MXFP4 => "uint8_t",
        }
    }

    pub fn mlir_ty(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::FP8 => "f8E4M3FN",
            Self::I8 | Self::Q8_0 => "i8",
            Self::I4 | Self::Q4_0 | Self::MXFP4 => "i4",
        }
    }
}

/// Symbolic or static tensor dimension (supports both Z3 compile-time constants and runtime symbols!).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DimExpr {
    Const(usize),
    Sym { name: String, max_bound: usize },
}

impl DimExpr {
    pub fn static_val(&self) -> usize {
        match self {
            Self::Const(v) => *v,
            Self::Sym { max_bound, .. } => *max_bound,
        }
    }
    pub fn is_symbolic(&self) -> bool {
        matches!(self, Self::Sym { .. })
    }
}

/// Physical storage layout solved by the Layout Propagation Pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutTag {
    /// Standard contiguous row-major `[rows, cols]`.
    RowMajor,
    /// Transposed / K-major `[cols, rows]` so SIMD lanes coalesce along K during GEMV/MMA!
    ColMajorNK,
    /// 128-bit aligned XOR-swizzled SRAM/HBM tile layout (bank-conflict free).
    Swizzled128,
    /// OCP MXFP4 block-32 packed (`[cols, rows/32]` of 17-byte blocks: 16B E2M1 nibbles + 1B E8M0 scale).
    MXFP4Block32NK,
    /// OCP MXFP4 Interleaved SuperBlock16 (`16` blocks = 512 weights = 256B `uint4` nibbles + 16B `uint4` scales = 272B, 16B aligned, single DRAM stream!).
    MXFP4Super16NK,
    /// True compressed 32x32 Block Sparse Row (BSR) layout: header + row_ptr + k_idx + contiguous non-zero [32_n][32_k] FP16 blocks!
    Bsr32NK,
}

impl LayoutTag {
    pub fn is_mxfp4(self) -> bool {
        matches!(self, Self::MXFP4Block32NK | Self::MXFP4Super16NK)
    }

    pub fn plan_tag(self) -> &'static str {
        match self {
            Self::RowMajor => "ROW_MAJOR",
            Self::ColMajorNK => "TRANSPOSE_NK",
            Self::Swizzled128 => "SWIZZLE_128",
            Self::MXFP4Block32NK => "PACK_MXFP4_NK",
            Self::MXFP4Super16NK => "PACK_MXFP4_NK_SB16",
            Self::Bsr32NK => "PACK_BSR32_NK",
        }
    }
}

/// Storage sparsity/irregularity kind (Stage 3 first-class ragged/paged/block-sparse tensors).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageKind {
    Dense,
    /// Variable-length compact batch (`offsets[B+1]` packing valid tokens without zero-padding!).
    Ragged { valid_pct: usize },
    /// Paged KV-Cache blocks (`page_size` tokens per block, indirect `block_table`).
    Paged { page_size: usize },
    /// Structured `block_m x block_n` block-sparse tensor with `density_pct` non-zero blocks.
    BlockSparse { block_m: usize, block_n: usize, density_pct: usize },
}

/// Declarative SPMD sharding annotation (`Replicate`, `Shard(axis)`, or `PartialSum`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShardSpec {
    Replicate,
    Shard(usize),
    PartialSum,
}

/// Full tensor descriptor in the Universal Tensor IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDesc {
    pub dims: Vec<DimExpr>,
    pub dtype: IrDType,
    pub layout: LayoutTag,
    pub storage: StorageKind,
    pub sharding: ShardSpec,
}

impl TensorDesc {
    pub fn matrix(rows: usize, cols: usize, dtype: IrDType) -> Self {
        Self {
            dims: vec![DimExpr::Const(rows), DimExpr::Const(cols)],
            dtype,
            layout: LayoutTag::RowMajor,
            storage: StorageKind::Dense,
            sharding: ShardSpec::Replicate,
        }
    }

    pub fn rows(&self) -> usize {
        if self.dims.len() >= 2 {
            self.dims[..self.dims.len() - 1]
                .iter()
                .map(|d| d.static_val())
                .product()
        } else {
            1
        }
    }

    pub fn cols(&self) -> usize {
        self.dims.last().map(|d| d.static_val()).unwrap_or(1)
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().map(|d| d.static_val()).product::<usize>().max(1)
    }

    pub fn bytes_f64(&self) -> f64 {
        let base = (self.numel() as f64) * self.dtype.bytes_f64();
        match self.storage {
            StorageKind::Dense => base,
            StorageKind::Ragged { valid_pct } => base * (valid_pct as f64 / 100.0),
            StorageKind::Paged { .. } => base,
            StorageKind::BlockSparse { density_pct, .. } => base * (density_pct as f64 / 100.0),
        }
    }

    pub fn alloc_bytes(&self) -> usize {
        match self.layout {
            LayoutTag::MXFP4Block32NK | LayoutTag::MXFP4Super16NK => {
                let r = self.rows();
                let c = self.cols();
                let blocks = (r + 31) / 32;
                (c * blocks * 17).max(64)
            }
            LayoutTag::Bsr32NK => {
                // Compressed size depends on runtime non-zero blocks; upper bound by dense size + BSR index header
                ((self.numel() * 2) + 4096).max(64)
            }
            _ => ((self.numel() as f64 * self.dtype.bytes_f64()).ceil() as usize).max(4),
        }
    }
}

/// Elementwise unary/binary operator for general fusion trees (including exact Reverse-Mode AD VJPs!).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElemOp {
    Add,
    Sub,
    Mul,
    Silu,
    Gelu,
    Relu,
    SwiGlu,
    /// Exact VJP `dy * silu'(saved_x)` where `silu'(x) = sig(x) * (1 + x * (1 - sig(x)))`.
    SiluBwd,
    /// Exact VJP `dy * gelu'(saved_x)` (tanh approximation derivative).
    GeluBwd,
    /// Exact VJP `(saved_x > 0) ? dy : 0`.
    ReluBwd,
    Scale { scale_bits: u32 },
    Cast(IrDType),
}

/// Row/axis reduction kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
    RmsNorm,
    LayerNorm,
    Softmax,
}

/// Universal Tensor IR operation (general across LLM, Conv, SSM, MoE, Training AD, SPMD).
#[derive(Debug, Clone)]
pub enum IrOp {
    /// General matrix contraction `C = op(A) @ op(B)` (supports transpose flags for AD VJP!).
    MatMul {
        a: String,
        b: String,
        trans_a: bool,
        trans_b: bool,
        weight_dtype: IrDType,
    },
    /// General unary elementwise op.
    Unary { op: ElemOp, x: String },
    /// General binary elementwise op (`Add`, `Sub`, `Mul`, `SwiGlu`).
    Binary { op: ElemOp, a: String, b: String },
    /// General row reduction (`RmsNorm`, `LayerNorm`, `Softmax`, `Sum`, `Max`).
    Reduce { op: ReduceOp, x: String, weight: Option<String> },
    /// General Rotary Embedding (in-place or fused into Attention).
    Rope { x: String },
    /// General Scaled Dot-Product / Grouped-Query / Paged Attention (`Q, K, V`).
    Attention {
        q: String,
        k: String,
        v: String,
        q_dim: usize,
        kv_dim: usize,
        causal: bool,
        paged: bool,
    },
    /// General 2D Convolution lowered into implicit im2col contraction (`[B*H*W, C_in*KH*KW] @ [C_in*KH*KW, C_out]`).
    Conv2d {
        x: String,
        w: String,
        in_h: usize,
        in_w: usize,
        kh: usize,
        kw: usize,
    },
    /// Dynamic Ragged-Batch MatMul (`valid_pct` real tokens packed via `offsets`).
    RaggedMatMul {
        a: String,
        b: String,
        valid_pct: usize,
    },
    /// Structured Block-Sparse MatMul (`density_pct` non-zero `32x32` tiles, persistent work-stealing).
    BlockSparseMatMul {
        a: String,
        b: String,
        block_size: usize,
        density_pct: usize,
    },
    /// Transpose `[R, C] -> [C, R]`.
    Transpose { x: String },
    /// SPMD Collective inserted automatically by the sharding pass.
    Collective {
        kind: CollectiveKind,
        x: String,
        devices: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollectiveKind {
    AllReduceSum,
    ReduceScatter,
    AllGather,
}

#[derive(Debug, Clone)]
pub struct IrNode {
    pub id: usize,
    pub out: String,
    pub op: IrOp,
    pub desc: TensorDesc,
    /// Optional provenance for backward AD checkpointing.
    pub is_backward: bool,
}

/// Full Universal Tensor IR Graph extracted from `@gpu_graph`.
#[derive(Debug, Clone)]
pub struct TensorIrGraph {
    pub fn_name: String,
    pub attr_mode: String,
    pub params: Vec<(String, TensorDesc)>,
    pub nodes: Vec<IrNode>,
    pub outputs: Vec<String>,
    pub tensors: BTreeMap<String, TensorDesc>,
    pub opt_notes: Vec<String>,
}

impl TensorIrGraph {
    pub fn new(fn_name: String, attr_mode: String) -> Self {
        Self {
            fn_name,
            attr_mode,
            params: Vec::new(),
            nodes: Vec::new(),
            outputs: Vec::new(),
            tensors: BTreeMap::new(),
            opt_notes: Vec::new(),
        }
    }

    pub fn push_node(&mut self, out: String, op: IrOp, desc: TensorDesc, is_backward: bool) {
        let id = self.nodes.len();
        self.tensors.insert(out.clone(), desc.clone());
        self.nodes.push(IrNode { id, out, op, desc, is_backward });
    }
}

/// Complete compiled multi-stage pipeline artifact (IR + Fused Tile Stages + Static Arena + Roofline).
#[derive(Debug, Clone)]
pub struct CompiledUniversalGraph {
    pub graph: TensorIrGraph,
    pub config: GpuOptConfig,
    pub stages: Vec<fusion::TileStage>,
    pub arena: fusion::StaticArenaPlan,
    pub input_repacks: Vec<(String, LayoutTag)>,
    pub roofline: UniversalRoofline,
    pub egraph_stats: egraph::EGraphStats,
    pub ad_stats: ad_spmd::AdSpmdStats,
}

#[derive(Debug, Clone)]
pub struct UniversalRoofline {
    pub raw_flops: f64,
    pub opt_flops: f64,
    pub naive_bytes: f64,
    pub fused_bytes: f64,
    pub bytes_saved_pct: f64,
    pub flops_saved_pct: f64,
    pub arithmetic_intensity: f64,
    pub m4_t_min_ms: f64,
    pub rtx4090_t_min_ms: f64,
    pub tpu_v5e_t_min_ms: f64,
}

// =========================================================================
// AST -> Universal TensorIrGraph Frontend Extraction
// =========================================================================

fn parse_dim_expr(ty: &TypeExpr) -> Option<DimExpr> {
    if let TypeExpr::Named { name, args, .. } = ty {
        if args.is_empty() {
            if let Ok(v) = name.parse::<usize>() {
                if v > 0 {
                    return Some(DimExpr::Const(v));
                }
            }
            // Symbolic dimension identifier (e.g. `S`, `B`, `Seq`) -> default upper bound 1024
            if name.chars().next().is_some_and(|c| c.is_ascii_uppercase())
                && !matches!(
                    name.as_str(),
                    "Float32" | "F32" | "Float16" | "F16" | "BFloat16" | "BF16"
                        | "FP8" | "Float8" | "Int8" | "I8" | "Int4" | "I4"
                        | "Q8_0" | "Q4_0" | "MXFP4" | "FP4"
                )
            {
                return Some(DimExpr::Sym { name: name.clone(), max_bound: 1024 });
            }
        }
    }
    None
}

fn parse_ir_dtype(ty: &TypeExpr) -> Option<IrDType> {
    if let TypeExpr::Named { name, .. } = ty {
        match name.as_str() {
            "Float32" | "F32" => Some(IrDType::F32),
            "Float16" | "F16" => Some(IrDType::F16),
            "BFloat16" | "BF16" => Some(IrDType::BF16),
            "FP8" | "Float8" | "F8E4M3" | "Float8E4M3" => Some(IrDType::FP8),
            "Int8" | "I8" => Some(IrDType::I8),
            "Int4" | "I4" => Some(IrDType::I4),
            "Q8_0" => Some(IrDType::Q8_0),
            "Q4_0" => Some(IrDType::Q4_0),
            "MXFP4" | "FP4" | "BlockFloat4" => Some(IrDType::MXFP4),
            _ => None,
        }
    } else {
        None
    }
}

/// Extracts `Matrix[R, C, DType]`, `Tensor[D0, D1, ..., DType]`, `Tensor3D`, `Tensor4D`.
pub fn extract_tensor_desc(ty: &TypeExpr) -> Option<TensorDesc> {
    let TypeExpr::Named { name, args, .. } = ty else { return None };
    if !matches!(name.as_str(), "Matrix" | "Tensor" | "Tensor3D" | "Tensor4D" | "RaggedMatrix" | "SparseMatrix") {
        return None;
    }
    if args.is_empty() {
        return None;
    }
    let mut dims = Vec::new();
    let mut dtype = IrDType::F32;
    for arg in args {
        if let Some((kn, val_ty)) = arg.as_named_arg() {
            if kn == "DType" {
                if let Some(dt) = parse_ir_dtype(val_ty) {
                    dtype = dt;
                }
            }
        } else if let Some(dt) = parse_ir_dtype(arg) {
            dtype = dt;
        } else if let Some(d) = parse_dim_expr(arg) {
            dims.push(d);
        }
    }
    if dims.is_empty() {
        return None;
    }
    if dims.len() == 1 {
        dims.insert(0, DimExpr::Const(1));
    }
    let storage = match name.as_str() {
        "RaggedMatrix" => StorageKind::Ragged { valid_pct: 50 },
        "SparseMatrix" => StorageKind::BlockSparse { block_m: 32, block_n: 32, density_pct: 25 },
        _ => StorageKind::Dense,
    };
    Some(TensorDesc {
        dims,
        dtype,
        layout: LayoutTag::RowMajor,
        storage,
        sharding: ShardSpec::Replicate,
    })
}

fn call_arg_ident(args: &[CallArg], idx: usize) -> Option<String> {
    match &args.get(idx)?.value.kind {
        ExprKind::Ident(s) => Some(s.clone()),
        _ => None,
    }
}

fn call_arg_int(args: &[CallArg], idx: usize) -> Option<usize> {
    match &args.get(idx)?.value.kind {
        ExprKind::Int(v) if *v > 0 => Some(*v as usize),
        _ => None,
    }
}

fn call_arg_float(args: &[CallArg], idx: usize) -> Option<f32> {
    match &args.get(idx)?.value.kind {
        ExprKind::Float(v) => Some(*v as f32),
        ExprKind::Int(v) => Some(*v as f32),
        _ => None,
    }
}

/// Extracts all `@gpu_graph` declarative functions into Universal `TensorIrGraph`s and compiles
/// them through the full toggleable Stage 1 -> Stage 2 -> Stage 3 optimization pipeline!
pub fn compile_universal_graphs(program: &Program, config: GpuOptConfig) -> Vec<CompiledUniversalGraph> {
    let mut out = Vec::new();
    for item in &program.items {
        let Item::Fn(f) = item else { continue };
        let Some(attr) = f.attrs.iter().find(|a| a.name == "gpu_graph") else { continue };
        let attr_mode = attr
            .args
            .first()
            .and_then(|a| match &a.value.kind {
                ExprKind::Str(s) | ExprKind::Ident(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_else(|| "tensor_dataflow".to_string());

        let mut g = TensorIrGraph::new(f.name.clone(), attr_mode.clone());
        let mut all_tensor_params = !f.params.is_empty();
        for p in &f.params {
            if let Some(td) = extract_tensor_desc(&p.ty) {
                g.params.push((p.name.clone(), td.clone()));
                g.tensors.insert(p.name.clone(), td);
            } else {
                all_tensor_params = false;
                break;
            }
        }
        if !all_tensor_params {
            continue;
        }

        let parse_stmt_op = |expr: &Expr, out_var: &str, g: &mut TensorIrGraph| -> bool {
            let ExprKind::Call { callee, args } = &expr.kind else { return false };
            let ExprKind::Ident(cname) = &callee.kind else { return false };
            match cname.as_str() {
                "tg_matmul" | "matmul" | "tg_qmatmul" | "qmatmul" if args.len() >= 2 => {
                    let (Some(a), Some(b)) = (call_arg_ident(args, 0), call_arg_ident(args, 1)) else { return false };
                    let (Some(sa), Some(sb)) = (g.tensors.get(&a).cloned(), g.tensors.get(&b).cloned()) else { return false };
                    if sa.cols() != sb.rows() { return false; }
                    let mut out_desc = TensorDesc::matrix(sa.rows(), sb.cols(), sa.dtype);
                    out_desc.storage = sa.storage;
                    g.push_node(out_var.to_string(), IrOp::MatMul {
                        a,
                        b,
                        trans_a: false,
                        trans_b: false,
                        weight_dtype: sb.dtype,
                    }, out_desc, false);
                    true
                }
                "tg_ragged_matmul" if args.len() >= 2 => {
                    let (Some(a), Some(b)) = (call_arg_ident(args, 0), call_arg_ident(args, 1)) else { return false };
                    let (Some(sa), Some(sb)) = (g.tensors.get(&a).cloned(), g.tensors.get(&b).cloned()) else { return false };
                    if sa.cols() != sb.rows() { return false; }
                    let valid_pct = call_arg_int(args, 2).unwrap_or(50).clamp(1, 100);
                    let mut out_desc = TensorDesc::matrix(sa.rows(), sb.cols(), sa.dtype);
                    out_desc.storage = StorageKind::Ragged { valid_pct };
                    g.push_node(out_var.to_string(), IrOp::RaggedMatMul { a, b, valid_pct }, out_desc, false);
                    true
                }
                "tg_block_sparse_matmul" if args.len() >= 2 => {
                    let (Some(a), Some(b)) = (call_arg_ident(args, 0), call_arg_ident(args, 1)) else { return false };
                    let (Some(sa), Some(sb)) = (g.tensors.get(&a).cloned(), g.tensors.get(&b).cloned()) else { return false };
                    if sa.cols() != sb.rows() { return false; }
                    let density_pct = call_arg_int(args, 2).unwrap_or(25).clamp(1, 100);
                    let block_size = call_arg_int(args, 3).unwrap_or(32);
                    let mut out_desc = TensorDesc::matrix(sa.rows(), sb.cols(), sa.dtype);
                    out_desc.storage = StorageKind::BlockSparse { block_m: block_size, block_n: block_size, density_pct };
                    g.push_node(out_var.to_string(), IrOp::BlockSparseMatMul { a, b, block_size, density_pct }, out_desc, false);
                    true
                }
                "tg_rmsnorm" | "rmsnorm" if args.len() >= 2 => {
                    let (Some(x), Some(w)) = (call_arg_ident(args, 0), call_arg_ident(args, 1)) else { return false };
                    let Some(sx) = g.tensors.get(&x).cloned() else { return false };
                    g.push_node(out_var.to_string(), IrOp::Reduce { op: ReduceOp::RmsNorm, x, weight: Some(w) }, sx, false);
                    true
                }
                "tg_layernorm" | "layernorm" if args.len() >= 2 => {
                    let (Some(x), Some(w)) = (call_arg_ident(args, 0), call_arg_ident(args, 1)) else { return false };
                    let Some(sx) = g.tensors.get(&x).cloned() else { return false };
                    g.push_node(out_var.to_string(), IrOp::Reduce { op: ReduceOp::LayerNorm, x, weight: Some(w) }, sx, false);
                    true
                }
                "tg_softmax" | "softmax" if !args.is_empty() => {
                    let Some(x) = call_arg_ident(args, 0) else { return false };
                    let Some(sx) = g.tensors.get(&x).cloned() else { return false };
                    g.push_node(out_var.to_string(), IrOp::Reduce { op: ReduceOp::Softmax, x, weight: None }, sx, false);
                    true
                }
                "tg_rope" | "rope" if !args.is_empty() => {
                    let Some(x) = call_arg_ident(args, 0) else { return false };
                    let Some(sx) = g.tensors.get(&x).cloned() else { return false };
                    g.push_node(out_var.to_string(), IrOp::Rope { x }, sx, false);
                    true
                }
                "tg_gqa_attention" | "gqa_attention" | "tg_attention" | "tg_paged_attention" if args.len() >= 3 => {
                    let (Some(q), Some(k), Some(v)) = (call_arg_ident(args, 0), call_arg_ident(args, 1), call_arg_ident(args, 2)) else { return false };
                    let (Some(sq), Some(sk), Some(sv)) = (g.tensors.get(&q).cloned(), g.tensors.get(&k).cloned(), g.tensors.get(&v).cloned()) else { return false };
                    if sq.rows() != sk.rows() || sk.rows() != sv.rows() || sk.cols() != sv.cols() || sk.cols() == 0 || sq.cols() % sk.cols() != 0 {
                        return false;
                    }
                    let paged = cname == "tg_paged_attention";
                    g.push_node(out_var.to_string(), IrOp::Attention {
                        q,
                        k,
                        v,
                        q_dim: sq.cols(),
                        kv_dim: sk.cols(),
                        causal: true,
                        paged,
                    }, sq, false);
                    true
                }
                "tg_swiglu" | "swiglu" if args.len() >= 2 => {
                    let (Some(a), Some(b)) = (call_arg_ident(args, 0), call_arg_ident(args, 1)) else { return false };
                    let Some(sa) = g.tensors.get(&a).cloned() else { return false };
                    g.push_node(out_var.to_string(), IrOp::Binary { op: ElemOp::SwiGlu, a, b }, sa, false);
                    true
                }
                "tg_add" | "add" if args.len() >= 2 => {
                    let (Some(a), Some(b)) = (call_arg_ident(args, 0), call_arg_ident(args, 1)) else { return false };
                    let Some(sa) = g.tensors.get(&a).cloned() else { return false };
                    g.push_node(out_var.to_string(), IrOp::Binary { op: ElemOp::Add, a, b }, sa, false);
                    true
                }
                "tg_sub" | "sub" if args.len() >= 2 => {
                    let (Some(a), Some(b)) = (call_arg_ident(args, 0), call_arg_ident(args, 1)) else { return false };
                    let Some(sa) = g.tensors.get(&a).cloned() else { return false };
                    g.push_node(out_var.to_string(), IrOp::Binary { op: ElemOp::Sub, a, b }, sa, false);
                    true
                }
                "tg_mul" | "mul" if args.len() >= 2 => {
                    let (Some(a), Some(b)) = (call_arg_ident(args, 0), call_arg_ident(args, 1)) else { return false };
                    let Some(sa) = g.tensors.get(&a).cloned() else { return false };
                    g.push_node(out_var.to_string(), IrOp::Binary { op: ElemOp::Mul, a, b }, sa, false);
                    true
                }
                "tg_silu" | "silu" if !args.is_empty() => {
                    let Some(x) = call_arg_ident(args, 0) else { return false };
                    let Some(sx) = g.tensors.get(&x).cloned() else { return false };
                    g.push_node(out_var.to_string(), IrOp::Unary { op: ElemOp::Silu, x }, sx, false);
                    true
                }
                "tg_gelu" | "gelu" if !args.is_empty() => {
                    let Some(x) = call_arg_ident(args, 0) else { return false };
                    let Some(sx) = g.tensors.get(&x).cloned() else { return false };
                    g.push_node(out_var.to_string(), IrOp::Unary { op: ElemOp::Gelu, x }, sx, false);
                    true
                }
                "tg_relu" | "relu" if !args.is_empty() => {
                    let Some(x) = call_arg_ident(args, 0) else { return false };
                    let Some(sx) = g.tensors.get(&x).cloned() else { return false };
                    g.push_node(out_var.to_string(), IrOp::Unary { op: ElemOp::Relu, x }, sx, false);
                    true
                }
                "tg_scale" | "scale" if args.len() >= 2 => {
                    let Some(x) = call_arg_ident(args, 0) else { return false };
                    let Some(sx) = g.tensors.get(&x).cloned() else { return false };
                    let sf = call_arg_float(args, 1).unwrap_or(1.0);
                    g.push_node(out_var.to_string(), IrOp::Unary { op: ElemOp::Scale { scale_bits: sf.to_bits() }, x }, sx, false);
                    true
                }
                "tg_transpose" | "transpose" if !args.is_empty() => {
                    let Some(x) = call_arg_ident(args, 0) else { return false };
                    let Some(sx) = g.tensors.get(&x).cloned() else { return false };
                    let td = TensorDesc::matrix(sx.cols(), sx.rows(), sx.dtype);
                    g.push_node(out_var.to_string(), IrOp::Transpose { x }, td, false);
                    true
                }
                "tg_conv2d" if args.len() >= 2 => {
                    let (Some(x), Some(w)) = (call_arg_ident(args, 0), call_arg_ident(args, 1)) else { return false };
                    let (Some(sx), Some(sw)) = (g.tensors.get(&x).cloned(), g.tensors.get(&w).cloned()) else { return false };
                    if sx.cols() != sw.rows() { return false; }
                    let out_desc = TensorDesc::matrix(sx.rows(), sw.cols(), sx.dtype);
                    g.push_node(out_var.to_string(), IrOp::Conv2d { x, w, in_h: sx.rows(), in_w: 1, kh: 1, kw: 1 }, out_desc, false);
                    true
                }
                _ => false,
            }
        };

        for stmt in &f.body.stmts {
            if let Stmt::Let { name, value, .. } = stmt {
                let _ = parse_stmt_op(value, name, &mut g);
            }
        }
        let mut ret_var = g.nodes.last().map(|n| n.out.clone()).unwrap_or_default();
        if let Some(tail) = &f.body.tail {
            match &tail.kind {
                ExprKind::Ident(s) => ret_var = s.clone(),
                ExprKind::Call { .. } => {
                    if parse_stmt_op(tail, "__tg_out", &mut g) {
                        ret_var = "__tg_out".to_string();
                    }
                }
                _ => {}
            }
        }
        if g.nodes.is_empty() || !g.tensors.contains_key(&ret_var) {
            continue;
        }
        g.outputs.push(ret_var);

        out.push(run_optimization_pipeline(g, config));
    }
    out
}

/// Runs the full toggleable Stage 1 -> Stage 2 -> Stage 3 compilation pipeline on any `TensorIrGraph`.
pub fn run_optimization_pipeline(mut g: TensorIrGraph, mut cfg: GpuOptConfig) -> CompiledUniversalGraph {
    let raw_flops = estimate_graph_flops(&g);
    let naive_bytes = estimate_naive_bytes(&g);

    // Honor graph-level attribute hints (`@gpu_graph("autograd")`, `@gpu_graph("spmd:4")`)
    let wants_autograd = g.attr_mode.contains("autograd") || g.attr_mode.contains("train");
    if let Some(rest) = g.attr_mode.strip_prefix("spmd:") {
        if let Ok(n) = rest.parse::<usize>() {
            if n > 1 && cfg.spmd_devices <= 1 {
                cfg.spmd_devices = n;
            }
        }
    }

    // --- Stage 2 Pass: E-Graph Equality Saturation & Algebraic Superoptimization ---
    let egraph_stats = if cfg.egraph_rewrite {
        egraph::run_egraph_superopt(&mut g)
    } else {
        egraph::EGraphStats::default()
    };

    // --- Stage 1 Pillar 6: Compile-Time Reverse-Mode AD (VJP) + Declarative SPMD ---
    let ad_stats = if cfg.ad_and_spmd {
        ad_spmd::run_ad_and_spmd(&mut g, wants_autograd, cfg.spmd_devices)
    } else {
        ad_spmd::AdSpmdStats::default()
    };

    // --- Stage 1 Pillars 1-5 + Stage 2 (Warp-Spec/PGO) + Stage 3 (Ragged/Sparse/MXFP4) ---
    let (stages, arena, input_repacks) = fusion::lower_to_tile_stages(&mut g, cfg);

    let opt_flops = estimate_graph_flops(&g);
    let fused_bytes = fusion::estimate_fused_bytes(&stages);
    let bytes_saved_pct = if naive_bytes > 0.0 {
        ((naive_bytes - fused_bytes).max(0.0) / naive_bytes) * 100.0
    } else {
        0.0
    };
    let flops_saved_pct = if raw_flops > 0.0 {
        ((raw_flops - opt_flops).max(0.0) / raw_flops) * 100.0
    } else {
        0.0
    };
    let ai = if fused_bytes > 0.0 { opt_flops / fused_bytes } else { 0.0 };

    let t_min = |spec: TargetSpec, uses_low_bit: bool| -> f64 {
        let t_mem = (fused_bytes / (spec.real_dram_bw_gbs * 1e9)) * 1e3;
        let peak_t = if uses_low_bit { spec.peak_fp8_or_i8_tflops } else { spec.peak_fp16_tflops };
        let t_cmp = (opt_flops / (peak_t * 1e12)) * 1e3;
        t_mem.max(t_cmp)
    };
    let uses_low_bit = g.tensors.values().any(|t| matches!(t.dtype, IrDType::FP8 | IrDType::I8 | IrDType::I4 | IrDType::Q4_0 | IrDType::MXFP4));

    let roofline = UniversalRoofline {
        raw_flops,
        opt_flops,
        naive_bytes,
        fused_bytes,
        bytes_saved_pct,
        flops_saved_pct,
        arithmetic_intensity: ai,
        m4_t_min_ms: t_min(TargetSpec::M4_MAX, false),
        rtx4090_t_min_ms: t_min(TargetSpec::RTX_4090, uses_low_bit),
        tpu_v5e_t_min_ms: t_min(TargetSpec::TPU_V5E, uses_low_bit),
    };

    CompiledUniversalGraph {
        graph: g,
        config: cfg,
        stages,
        arena,
        input_repacks,
        roofline,
        egraph_stats,
        ad_stats,
    }
}

pub fn estimate_graph_flops(g: &TensorIrGraph) -> f64 {
    let mut flops = 0.0;
    for n in &g.nodes {
        match &n.op {
            IrOp::MatMul { a, b, trans_a, trans_b, .. } => {
                let sa = &g.tensors[a];
                let sb = &g.tensors[b];
                let m = if *trans_a { sa.cols() } else { sa.rows() };
                let k = if *trans_a { sa.rows() } else { sa.cols() };
                let nn = if *trans_b { sb.rows() } else { sb.cols() };
                flops += 2.0 * (m as f64) * (k as f64) * (nn as f64);
            }
            IrOp::Conv2d { x, w, .. } => {
                let sx = &g.tensors[x];
                let sw = &g.tensors[w];
                flops += 2.0 * (sx.rows() as f64) * (sx.cols() as f64) * (sw.cols() as f64);
            }
            IrOp::RaggedMatMul { a, b, valid_pct } => {
                let sa = &g.tensors[a];
                let sb = &g.tensors[b];
                let eff_m = (sa.rows() as f64) * (*valid_pct as f64 / 100.0);
                flops += 2.0 * eff_m * (sa.cols() as f64) * (sb.cols() as f64);
            }
            IrOp::BlockSparseMatMul { a, b, density_pct, .. } => {
                let sa = &g.tensors[a];
                let sb = &g.tensors[b];
                flops += 2.0 * (sa.rows() as f64) * (sa.cols() as f64) * (sb.cols() as f64) * (*density_pct as f64 / 100.0);
            }
            IrOp::Attention { q, q_dim, .. } => {
                let sq = &g.tensors[q];
                let s = sq.rows() as f64;
                flops += 4.0 * s * s * (*q_dim as f64);
            }
            IrOp::Reduce { .. } => flops += 5.0 * (n.desc.numel() as f64),
            IrOp::Unary { .. } | IrOp::Binary { .. } | IrOp::Rope { .. } => {
                flops += 4.0 * (n.desc.numel() as f64);
            }
            IrOp::Transpose { .. } | IrOp::Collective { .. } => {}
        }
    }
    flops
}

pub fn estimate_naive_bytes(g: &TensorIrGraph) -> f64 {
    let mut bytes = 0.0;
    for n in &g.nodes {
        match &n.op {
            IrOp::MatMul { a, b, .. }
            | IrOp::Conv2d { x: a, w: b, .. }
            | IrOp::RaggedMatMul { a, b, .. }
            | IrOp::BlockSparseMatMul { a, b, .. } => {
                bytes += g.tensors[a].bytes_f64() + g.tensors[b].bytes_f64() + n.desc.bytes_f64();
            }
            IrOp::Binary { a, b, .. } => {
                bytes += g.tensors[a].bytes_f64() + g.tensors[b].bytes_f64() + n.desc.bytes_f64();
            }
            IrOp::Reduce { x, weight, .. } => {
                bytes += g.tensors[x].bytes_f64()
                    + weight.as_ref().map(|w| g.tensors[w].bytes_f64()).unwrap_or(0.0)
                    + n.desc.bytes_f64();
            }
            IrOp::Attention { q, k, v, .. } => {
                let sq = &g.tensors[q];
                let s = sq.rows() as f64;
                // Naive attention materializes full S x S score & prob matrices!
                let naive_s2 = 2.0 * s * s * 4.0;
                bytes += sq.bytes_f64() + g.tensors[k].bytes_f64() + g.tensors[v].bytes_f64() + n.desc.bytes_f64() + naive_s2;
            }
            IrOp::Unary { x, .. } | IrOp::Rope { x } | IrOp::Transpose { x } | IrOp::Collective { x, .. } => {
                bytes += g.tensors[x].bytes_f64() + n.desc.bytes_f64();
            }
        }
    }
    bytes
}
