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
pub mod hetero_stream;
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

    // --- Stage 4: Full General Compiler & Runtime Facilities (Non-Chip-Specific!) ---
    /// #1: 2D L2/SLC Tile Swizzling (`GROUP_M = 8` rasterization order for higher L2/SLC hit rate).
    pub tile_swizzle: bool,
    /// #2: Horizontal Weight Packing (`Q||K||V` / `Gate||Up` concatenated into one contiguous wide GEMM).
    pub horiz_pack: bool,
    /// #3: Zero-Copy `inout` Buffer Donation / In-Place Aliasing (`tf.aliasing_output` / in-place AdamW & dlogits).
    pub buffer_donation: bool,
    /// #4: Device-Side Graph Replay (`MTLIndirectCommandBuffer` / CUDA Graph Exec / TPU `stablehlo.while`).
    pub graph_replay: bool,
    /// #5: Online Monoid Fused Cross-Entropy (`llm.c` `fused_classifier3`) + Roofline Selective Checkpointing.
    pub online_monoid: bool,
    /// #6: Auto Mixed-Precision + Fused Single-Pass Norm & Dynamic Quantization (W8A8 / FP8 Contraction).
    pub amp_dyn_quant: bool,
    /// #7: Symbolic Dynamic-Shape Multi-Version Bucketing (`M<=8`, `9..64`, `>64` O(1) dispatch, zero padding/JIT).
    pub dyn_bucket: bool,
    /// #8: Non-Transformer General Primitives (Zero-`im2col` Implicit GEMM Conv2d + Blelloch Parallel Associative Scan).
    pub non_tf_ops: bool,
    /// #9: Micro-Tile Compute-Communication SPMD Overlap + Dual-Queue Async Weight Prefetch (`MTLEvent` / NCCL stream).
    pub async_overlap: bool,

    // --- Stage 5: `llm.c`-Inspired Universal Compiler Facilities (ALL Default-ON!) ---
    /// #10: Automatic logical-vs-padded dimension alignment (`50257->50304`, `1003->1024`) restoring 128-bit & MMA fast paths!
    pub auto_pad: bool,
    /// #11: Stateless Counter-Based Philox Stochastic Rounding on BF16/FP16 optimizer updates (master-weight-free training!).
    pub stochastic_round: bool,
    /// #12: Liveness-driven Streaming Cache Hints (`CachePolicy::Streaming` -> CUDA `__ldcs`/`__stcs`, Metal 64B burst, TPU MSA).
    pub streaming_cache: bool,
    /// #13: Automatic ZeRO-1 Optimizer State Sharding (`ReduceScatter` + `1/D` local AdamW + `AllGather`) + Gradient Bucketing.
    pub zero1_spmd: bool,

    // --- Stage 6: Universal Heterogeneous Out-of-Core Streaming Facilities (ALL Default-ON!) ---
    /// #14: Automatic Hard-VRAM-Cap Budget Planner (Core Attn + Q8_0 KV-Cache + Prefix FFN / Per-Layer LRU + NVMe F_NOCACHE + Batched Prefill).
    pub vram_budget_plan: bool,
    /// #15: NUMA-Local Parallel First-Touch 4-Row Contiguous Weight Repacking (`Q4_0_4R` 144B/72B, `Q4_K_4R`).
    pub numa_repack_layout: bool,
    /// #16: Hardware Dual-Stream `Compute || DMA` Pipeline (`cuStreamWaitEvent` / `MTLEvent` + `MTLPurgeableStateEmpty` + Async Promoter).
    pub dual_stream_dma: bool,
    /// #17: Roofline-Driven GPU+CPU Hybrid Dispatch (AVX2/NEON INT16 `Main=tid0` container-cgroup-aware worker pool when `BW_DDR > 1.2 * BW_PCIe`).
    pub hetero_dispatch: bool,
    /// #22: Iteration-Level Continuous Batching (`step_batch`) + PagedAttention 16-Token Block-Table KV + Zero-Copy COW Prefix Fork.
    pub paged_cont_batch: bool,
    /// #23: Marlin-Style 2-Stage `cp.async` + FP16x2 Magic-Bit `lop3` W4A16 Tensor Core Prefill.
    pub marlin_w4a16_mma: bool,
    /// #24: Multi-GPU MoE Expert Parallelism (`EP=2..8`) + Zero-Link `dlopen("libnccl.so.2")` Collective Auto-Dispatch.
    pub moe_ep_nccl: bool,
    /// #25: Tree-Attention Speculative Masking (`verify_tree`) + Native MTP `nextn` Block + Vision `mmproj` Soft-Prompt Injection.
    pub tree_spec_mtp_mm: bool,
    /// #27: Cross-Chip Small-Batch (`B=2..8`) Coalesced Shared-X `half4`/`half2` W4A16/Q4_1/Q4_K/Q6_K Dispatch (Metal + CUDA + TPU).
    pub small_batch_shared_x: bool,
    /// #28: 2D Hybrid Tensor × Expert Parallelism (`TP × EP`) Co-Located/Orthogonal Mesh + Resident MoE Direct Fast-Path.
    pub hybrid_2d_tp_ep: bool,
    /// #29: Chunked Prefill Interleaved Continuous Batching + Adaptive Low-Concurrency Speculative Fusion.
    pub chunked_prefill_cb: bool,
    /// #30: Zero-Copy Runtime GGUF Multi-Adapter LoRA Hot-Swapping (`y += (scale/r) * B(Ax)` sharing base weights & PagedKV).
    pub multi_lora_zero_copy: bool,
    /// #31: Multi-Image AnyRes Vision + 2D/3D MRoPE + Native `FP8_E4M3` & OCP `MXFP4` (`E2M1`+`E8M0`) Cross-Chip Kernels.
    pub anyres_mrope_fp8_mxfp4: bool,
    /// #32: Hardware Tensor Core (`mma.sync`) & Apple AMX (`simdgroup_half8x8`) FP8_E4M3 + MXFP4 Kernels (Metal + CUDA + TPU).
    pub fp8_mxfp4_tc_amx: bool,
    /// #33: `Q4_K`/`Q6_K` 2-Stage `cp.async` Marlin Tensor Core & Metal AMX `simdgroup_float8x8` Pipeline.
    pub q4k_q6k_marlin_pipeline: bool,
    /// #34: Universal PagedAttention Continuous Batching (`TP=2..8`, Latent MLA, `2D TPxEP`) + Native Zero-Migration `paged_verify_batch`.
    pub paged_cb_tp_mla_spec: bool,
    /// #35: MoE `B=2..16` Batched Routing + Counting-Sort Grouped MoE GEMM Fusion (Single Weight Read per Active Expert).
    pub moe_grouped_gemm_batch: bool,
    /// #36: Auto `Qwen2.5-VL` 2D/3D MRoPE Grid + Native Audio/Whisper Mel-Spectrogram Encoder + Vision Content-Hash Prefix Cache + Prometheus Metrics.
    pub vlm_mrope_audio_prefix_cache: bool,
    /// #37: Zero-Host-Sync GPU-Resident Persistent Grouped MoE Mega-Kernel (GPU TopK + Counting-Sort + Single-Launch Grouped GEMM).
    pub moe_gpu_persistent_mega: bool,
    /// #38: Native Ada/Hopper `mma.sync.m16n8k32.f32.e4m3.e4m3.f32` + Apple Metal `simdgroup_half8x8->float8x8` AMX + TPU W4A8/FP8 Fusion.
    pub fp8_e4m3_m16n8k32_amx_f16: bool,
    /// #39: Mixed-Batch Heterogeneous Multi-Tenant LoRA `BGMV` (S-LoRA / Punica) Zero-Split Batched Kernel.
    pub hetero_lora_bgmv_batch: bool,
    /// #40: Real `Qwen2.5-VL` Spatial Merger ViT + Real Whisper Transformer Encoder GGUF + PagedKV-Block-Level Multimodal COW Prefix Cache.
    pub multimodal_real_vit_whisper_kv_cache: bool,
    /// #41: Disaggregated Prefill-Decode (PD) PagedKV Block Serialization & Zero-Copy Cross-Node KV Transfer.
    pub disaggregated_pd_kv_transfer: bool,
    /// #42: Native Tensor Core (`mma.sync`) / Apple FP16-AMX / TPU `ragged_dot` Adaptive Grouped MoE Mega-Kernel v2.
    pub moe_mega_tc_amx_ragged: bool,
    /// #43: Device-Side Persistent Decode Graphs (`CUDA Graphs` `cuGraphLaunch` + Metal `MTLIndirectCommandBuffer` + TPU `while_loop`).
    pub device_graph_capture: bool,
    /// #44: Zero-Copy High-Speed Disaggregated PD Transport (Raw Binary Stream + POSIX `shm_open` + CUDA `cuIpc`).
    pub pd_zerocopy_binary_shm_ipc: bool,
    /// #45: `FP8_E4M3`/`Q4_0` PagedKV Compression (`2x-3.56x`) + Sliding Window Attention (SWA) & Attention Sink + `128K+` Context.
    pub paged_kv_fp8_q4_swa_longctx: bool,
    /// #46: Prefix-Affinity Disaggregated `PD Router / Gateway` + Real Open-Source Multimodal E2E + Prometheus TTFT/TPOT Histograms.
    pub pd_router_prefix_affinity_telemetry: bool,
    /// #47: Multi-GPU `TP=2..8` & `2D TPxEP` Persistent Device Decode Graphs (`CUDA Graphs` + NCCL Capture) & Metal Multi-Queue Async ICB Pipeline.
    pub multi_gpu_graph_and_async_icb: bool,
    /// #48: Native `sm_89+` FP8 (`e4m3×e4m3`) `mma.sync.m16n8k32` Grouped MoE Mega-Kernel v3 & Apple Metal Register-Direct `half4` AMX Mega-Kernel v3.
    pub moe_mega_v3_fp8_tc_and_reg_amx: bool,
    /// #49: Multi-GPU `TP=2..8` GPU-Direct `CUDA IPC` Sharded PD + Layer-Wise Streaming Pipelined PD (`/v1/pd/layer`).
    pub pd_multigpu_ipc_and_layer_stream: bool,
    /// #50: Hardware-Native `FlashDecoding` Split-KV Parallel Long-Context Decode (`32K-128K`, 4-Mode GQA + SWA/Sink + Latent MLA).
    pub flash_decoding_split_kv_lse: bool,
    /// #51: Production HA `PD Router` (Health Checks / Circuit Breaker / Failover / Backpressure) + SSE Stream Cancel + Dynamic `/v1/lora/load` Hot-Swap.
    pub production_ha_cancel_and_lora_hotswap: bool,

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
            tile_swizzle: false,
            horiz_pack: false,
            buffer_donation: false,
            graph_replay: false,
            online_monoid: false,
            amp_dyn_quant: false,
            dyn_bucket: false,
            non_tf_ops: false,
            async_overlap: false,
            auto_pad: false,
            stochastic_round: false,
            streaming_cache: false,
            zero1_spmd: false,
            vram_budget_plan: false,
            numa_repack_layout: false,
            dual_stream_dma: false,
            hetero_dispatch: false,
            paged_cont_batch: false,
            marlin_w4a16_mma: false,
            moe_ep_nccl: false,
            tree_spec_mtp_mm: false,
            small_batch_shared_x: false,
            hybrid_2d_tp_ep: false,
            chunked_prefill_cb: false,
            multi_lora_zero_copy: false,
            anyres_mrope_fp8_mxfp4: false,
            fp8_mxfp4_tc_amx: false,
            q4k_q6k_marlin_pipeline: false,
            paged_cb_tp_mla_spec: false,
            moe_grouped_gemm_batch: false,
            vlm_mrope_audio_prefix_cache: false,
            moe_gpu_persistent_mega: false,
            fp8_e4m3_m16n8k32_amx_f16: false,
            hetero_lora_bgmv_batch: false,
            multimodal_real_vit_whisper_kv_cache: false,
            disaggregated_pd_kv_transfer: false,
            moe_mega_tc_amx_ragged: false,
            device_graph_capture: false,
            pd_zerocopy_binary_shm_ipc: false,
            paged_kv_fp8_q4_swa_longctx: false,
            pd_router_prefix_affinity_telemetry: false,
            multi_gpu_graph_and_async_icb: false,
            moe_mega_v3_fp8_tc_and_reg_amx: false,
            pd_multigpu_ipc_and_layer_stream: false,
            flash_decoding_split_kv_lse: false,
            production_ha_cancel_and_lora_hotswap: false,
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
            ..Self::none()
        }
    }

    pub fn stage2() -> Self {
        let mut c = Self::stage1();
        c.pgo_autotune = true;
        c.warp_spec = true;
        c.egraph_rewrite = true;
        c
    }

    pub fn stage3() -> Self {
        let mut c = Self::stage2();
        c.ragged_sparse = true;
        c.mxfp4_quant = true;
        c
    }

    pub fn stage4() -> Self {
        let mut c = Self::stage3();
        c.tile_swizzle = true;
        c.horiz_pack = true;
        c.buffer_donation = true;
        c.graph_replay = true;
        c.online_monoid = true;
        c.amp_dyn_quant = true;
        c.dyn_bucket = true;
        c.non_tf_ops = true;
        c.async_overlap = true;
        c
    }

    pub fn stage5() -> Self {
        let mut c = Self::stage4();
        c.auto_pad = true;
        c.stochastic_round = true;
        c.streaming_cache = true;
        c.zero1_spmd = true;
        c
    }

    pub fn stage6() -> Self {
        let mut c = Self::stage5();
        c.vram_budget_plan = true;
        c.numa_repack_layout = true;
        c.dual_stream_dma = true;
        c.hetero_dispatch = true;
        c.paged_cont_batch = true;
        c.marlin_w4a16_mma = true;
        c.moe_ep_nccl = true;
        c.tree_spec_mtp_mm = true;
        c.small_batch_shared_x = true;
        c.hybrid_2d_tp_ep = true;
        c.chunked_prefill_cb = true;
        c.multi_lora_zero_copy = true;
        c.anyres_mrope_fp8_mxfp4 = true;
        c.fp8_mxfp4_tc_amx = true;
        c.q4k_q6k_marlin_pipeline = true;
        c.paged_cb_tp_mla_spec = true;
        c.moe_grouped_gemm_batch = true;
        c.vlm_mrope_audio_prefix_cache = true;
        c.moe_gpu_persistent_mega = true;
        c.fp8_e4m3_m16n8k32_amx_f16 = true;
        c.hetero_lora_bgmv_batch = true;
        c.multimodal_real_vit_whisper_kv_cache = true;
        c.disaggregated_pd_kv_transfer = true;
        c.moe_mega_tc_amx_ragged = true;
        c.device_graph_capture = true;
        c.pd_zerocopy_binary_shm_ipc = true;
        c.paged_kv_fp8_q4_swa_longctx = true;
        c.pd_router_prefix_affinity_telemetry = true;
        c.multi_gpu_graph_and_async_icb = true;
        c.moe_mega_v3_fp8_tc_and_reg_amx = true;
        c.pd_multigpu_ipc_and_layer_stream = true;
        c.flash_decoding_split_kv_lse = true;
        c.production_ha_cancel_and_lora_hotswap = true;
        c
    }

    pub fn all() -> Self {
        Self::stage6()
    }

    /// Parses comma-separated preset & toggle specs, e.g.:
    ///   `"none"`, `"stage1"`, `"stage2"`, `"stage3"`, `"stage4"`, `"all"`,
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
            "stage3" | "s3" => Self::stage3(),
            "stage4" | "s4" => Self::stage4(),
            "stage5" | "s5" => Self::stage5(),
            "stage6" | "s6" | "all" | "O3" => Self::all(),
            t if t.starts_with('-') => Self::all(),
            _ => Self::none(),
        };

        for tok in tokens {
            match tok {
                "none" | "baseline" | "O0" => cfg = Self::none(),
                "stage1" | "s1" => cfg = Self::stage1(),
                "stage2" | "s2" => cfg = Self::stage2(),
                "stage3" | "s3" => cfg = Self::stage3(),
                "stage4" | "s4" => cfg = Self::stage4(),
                "stage5" | "s5" => cfg = Self::stage5(),
                "stage6" | "s6" | "all" | "O3" => cfg = Self::all(),
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
                        "tile_swizzle" | "l2_swizzle" => cfg.tile_swizzle = enable,
                        "horiz_pack" | "qkv_pack" | "concat_pack" => cfg.horiz_pack = enable,
                        "buffer_donation" | "donate" | "inplace" => cfg.buffer_donation = enable,
                        "graph_replay" | "icb" | "replay" => cfg.graph_replay = enable,
                        "online_monoid" | "crossentropy" | "monoid" => cfg.online_monoid = enable,
                        "amp_dyn_quant" | "amp" | "w8a8" => cfg.amp_dyn_quant = enable,
                        "dyn_bucket" | "bucket" | "dynamic" => cfg.dyn_bucket = enable,
                        "non_tf_ops" | "conv_scan" | "implicit_conv" => cfg.non_tf_ops = enable,
                        "async_overlap" | "overlap" | "prefetch" => cfg.async_overlap = enable,
                        "auto_pad" | "pad" | "align_pad" => cfg.auto_pad = enable,
                        "stochastic_round" | "sr" | "philox" => cfg.stochastic_round = enable,
                        "streaming_cache" | "ldcs" | "stcs" => cfg.streaming_cache = enable,
                        "zero1_spmd" | "zero1" => cfg.zero1_spmd = enable,
                        "vram_budget_plan" | "vram_plan" | "budget" => cfg.vram_budget_plan = enable,
                        "numa_repack_layout" | "numa_repack" | "repack_4r" => cfg.numa_repack_layout = enable,
                        "dual_stream_dma" | "dma_ring" | "dual_stream" => cfg.dual_stream_dma = enable,
                        "hetero_dispatch" | "hetero" | "hybrid_cpu" => cfg.hetero_dispatch = enable,
                        "paged_cont_batch" | "paged_kv" | "cont_batch" => cfg.paged_cont_batch = enable,
                        "marlin_w4a16_mma" | "marlin" => cfg.marlin_w4a16_mma = enable,
                        "moe_ep_nccl" | "ep" | "nccl" => cfg.moe_ep_nccl = enable,
                        "tree_spec_mtp_mm" | "tree_spec" | "mtp" | "mmproj" => cfg.tree_spec_mtp_mm = enable,
                        "small_batch_shared_x" | "small_b" | "shared_x" => cfg.small_batch_shared_x = enable,
                        "hybrid_2d_tp_ep" | "tp_ep" | "2d_mesh" => cfg.hybrid_2d_tp_ep = enable,
                        "chunked_prefill_cb" | "chunked_prefill" => cfg.chunked_prefill_cb = enable,
                        "multi_lora_zero_copy" | "lora" => cfg.multi_lora_zero_copy = enable,
                        "anyres_mrope_fp8_mxfp4" | "mrope" | "fp8_mxfp4" => cfg.anyres_mrope_fp8_mxfp4 = enable,
                        "fp8_mxfp4_tc_amx" | "fp8_tc" | "mxfp4_amx" => cfg.fp8_mxfp4_tc_amx = enable,
                        "q4k_q6k_marlin_pipeline" | "q4k_marlin" | "q6k_amx" => cfg.q4k_q6k_marlin_pipeline = enable,
                        "paged_cb_tp_mla_spec" | "paged_verify" | "mla_cb" => cfg.paged_cb_tp_mla_spec = enable,
                        "moe_grouped_gemm_batch" | "moe_grouped" => cfg.moe_grouped_gemm_batch = enable,
                        "vlm_mrope_audio_prefix_cache" | "audio_vlm" | "vision_cache" => cfg.vlm_mrope_audio_prefix_cache = enable,
                        "moe_gpu_persistent_mega" | "moe_mega" => cfg.moe_gpu_persistent_mega = enable,
                        "fp8_e4m3_m16n8k32_amx_f16" | "fp8_k32" | "amx_f16" => cfg.fp8_e4m3_m16n8k32_amx_f16 = enable,
                        "hetero_lora_bgmv_batch" | "hetero_lora" | "bgmv" => cfg.hetero_lora_bgmv_batch = enable,
                        "multimodal_real_vit_whisper_kv_cache" | "mm_kv_cow" => cfg.multimodal_real_vit_whisper_kv_cache = enable,
                        "disaggregated_pd_kv_transfer" | "pd_kv" => cfg.disaggregated_pd_kv_transfer = enable,
                        "moe_mega_tc_amx_ragged" | "moe_mega_v2" => cfg.moe_mega_tc_amx_ragged = enable,
                        "device_graph_capture" | "dev_graph" | "cuda_graph" | "metal_icb" => cfg.device_graph_capture = enable,
                        "pd_zerocopy_binary_shm_ipc" | "pd_zc" | "pd_shm" => cfg.pd_zerocopy_binary_shm_ipc = enable,
                        "paged_kv_fp8_q4_swa_longctx" | "kv_fp8_q4_swa" | "swa" => cfg.paged_kv_fp8_q4_swa_longctx = enable,
                        "pd_router_prefix_affinity_telemetry" | "pd_router" => cfg.pd_router_prefix_affinity_telemetry = enable,
                        "multi_gpu_graph_and_async_icb" | "tp_graph" | "async_icb" => cfg.multi_gpu_graph_and_async_icb = enable,
                        "moe_mega_v3_fp8_tc_and_reg_amx" | "moe_mega_v3" => cfg.moe_mega_v3_fp8_tc_and_reg_amx = enable,
                        "pd_multigpu_ipc_and_layer_stream" | "pd_tp_ipc" | "pd_layer_stream" => cfg.pd_multigpu_ipc_and_layer_stream = enable,
                        "flash_decoding_split_kv_lse" | "flash_decode" | "split_kv" => cfg.flash_decoding_split_kv_lse = enable,
                        "production_ha_cancel_and_lora_hotswap" | "router_ha" | "lora_hotswap" => cfg.production_ha_cancel_and_lora_hotswap = enable,
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
        if self.tile_swizzle { on.push("l2swz"); }
        if self.horiz_pack { on.push("hpack"); }
        if self.buffer_donation { on.push("donate"); }
        if self.graph_replay { on.push("replay"); }
        if self.online_monoid { on.push("monoid"); }
        if self.amp_dyn_quant { on.push("amp_q8"); }
        if self.dyn_bucket { on.push("dyn_bkt"); }
        if self.non_tf_ops { on.push("conv_scan"); }
        if self.async_overlap { on.push("overlap"); }
        if self.auto_pad { on.push("autopad"); }
        if self.stochastic_round { on.push("sr"); }
        if self.streaming_cache { on.push("ldcs"); }
        if self.zero1_spmd { on.push("zero1"); }
        if self.vram_budget_plan { on.push("vram_plan"); }
        if self.numa_repack_layout { on.push("numa4r"); }
        if self.dual_stream_dma { on.push("dma_ring"); }
        if self.hetero_dispatch { on.push("hetero"); }
        if self.paged_cont_batch { on.push("paged_cb"); }
        if self.marlin_w4a16_mma { on.push("marlin"); }
        if self.moe_ep_nccl { on.push("ep_nccl"); }
        if self.tree_spec_mtp_mm { on.push("tree_mtp_mm"); }
        if self.small_batch_shared_x { on.push("sb_shx"); }
        if self.hybrid_2d_tp_ep { on.push("2d_tpep"); }
        if self.chunked_prefill_cb { on.push("chunk_pf"); }
        if self.multi_lora_zero_copy { on.push("lora"); }
        if self.anyres_mrope_fp8_mxfp4 { on.push("mrope_fp8_fp4"); }
        if self.fp8_mxfp4_tc_amx { on.push("fp8_tc_amx"); }
        if self.q4k_q6k_marlin_pipeline { on.push("q4k_marlin"); }
        if self.paged_cb_tp_mla_spec { on.push("paged_tp_mla_spec"); }
        if self.moe_grouped_gemm_batch { on.push("moe_grouped"); }
        if self.vlm_mrope_audio_prefix_cache { on.push("vlm_audio_cache"); }
        if self.moe_gpu_persistent_mega { on.push("moe_mega"); }
        if self.fp8_e4m3_m16n8k32_amx_f16 { on.push("fp8_k32_amxf16"); }
        if self.hetero_lora_bgmv_batch { on.push("hetero_lora_bgmv"); }
        if self.multimodal_real_vit_whisper_kv_cache { on.push("mm_kv_cow"); }
        if self.disaggregated_pd_kv_transfer { on.push("pd_kv"); }
        if self.moe_mega_tc_amx_ragged { on.push("moe_mega_v2"); }
        if self.device_graph_capture { on.push("dev_graph"); }
        if self.pd_zerocopy_binary_shm_ipc { on.push("pd_zc"); }
        if self.paged_kv_fp8_q4_swa_longctx { on.push("kv_fp8_q4_swa"); }
        if self.pd_router_prefix_affinity_telemetry { on.push("pd_router"); }
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
#[allow(non_camel_case_types)]
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
    Q5_0,
    Q5_1,
    /// GGML K-Quant & I-Quant super-blocks (256 weights: Q2_K=84B, Q3_K=110B, IQ4_XS=136B, Q4_K=144B, Q5_K=176B, Q6_K=210B).
    Q2_K,
    Q3_K,
    IQ4_XS,
    Q4_K,
    Q5_K,
    Q6_K,
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
            Self::Q5_0 => 22.0 / 32.0,
            Self::Q5_1 => 24.0 / 32.0,
            Self::Q2_K => 84.0 / 256.0,
            Self::Q3_K => 110.0 / 256.0,
            Self::IQ4_XS => 136.0 / 256.0,
            Self::Q4_K => 144.0 / 256.0,
            Self::Q5_K => 176.0 / 256.0,
            Self::Q6_K => 210.0 / 256.0,
            Self::MXFP4 => 17.0 / 32.0,
        }
    }

    pub fn is_sub_byte(self) -> bool {
        matches!(self, Self::I4 | Self::Q4_0 | Self::Q5_0 | Self::Q5_1 | Self::Q2_K | Self::Q3_K | Self::IQ4_XS | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::MXFP4)
    }

    pub fn msl_ty(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::F16 => "half",
            Self::BF16 => "bfloat",
            Self::FP8 | Self::I4 | Self::Q8_0 | Self::Q4_0 | Self::Q5_0 | Self::Q5_1 | Self::Q2_K | Self::Q3_K | Self::IQ4_XS | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::MXFP4 => "uchar",
            Self::I8 => "char",
        }
    }

    pub fn cuda_ty(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::F16 => "__half",
            Self::BF16 => "__nv_bfloat16",
            Self::I8 => "int8_t",
            Self::FP8 | Self::I4 | Self::Q8_0 | Self::Q4_0 | Self::Q5_0 | Self::Q5_1 | Self::Q2_K | Self::Q3_K | Self::IQ4_XS | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::MXFP4 => "uint8_t",
        }
    }

    pub fn mlir_ty(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::FP8 => "f8E4M3FN",
            Self::I8 | Self::Q8_0 | Self::Q5_0 | Self::Q5_1 | Self::Q5_K | Self::Q6_K => "i8",
            Self::I4 | Self::Q4_0 | Self::Q2_K | Self::Q3_K | Self::IQ4_XS | Self::Q4_K | Self::MXFP4 => "i4",
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
#[allow(non_camel_case_types)]
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
    /// Per-row INT8 dynamic-quantized weights (`[N, K]` int8 + `[N]` FP32 row scales -> -50% DRAM!).
    Int8RowScaleNK,
    /// Stage 6 Facility #15: NUMA-local parallel first-touch 4-row interleaved Q4_0 (`72B/blk` or `144B/blk` fused gate+up).
    NumaRepackQ4_4R,
    /// Stage 6 Facility #15: NUMA-local parallel first-touch 4-row interleaved K-Quant (`Q4_K`/`Q5_K`/`Q6_K`).
    NumaRepackQ4K_4R,
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
            Self::Int8RowScaleNK => "PACK_INT8_ROW_NK",
            Self::NumaRepackQ4_4R => "NUMA_REPACK_Q4_4R",
            Self::NumaRepackQ4K_4R => "NUMA_REPACK_Q4K_4R",
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
            LayoutTag::Int8RowScaleNK => {
                let r = self.rows();
                let c = self.cols();
                (c * r + c * 4 + 256).max(64)
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
    /// Facility #2: Horizontal Weight-Packed N-way Projection (`X @ [W0 || W1 || ...]` -> 1 wide GEMM).
    ConcatMatMul {
        a: String,
        weights: Vec<String>,
    },
    /// Facility #3: Zero-Copy Buffer-Donated Fused 128-bit AdamW Step (`param, grad, m, v`).
    AdamWStep {
        param: String,
        grad: String,
        m: String,
        v: String,
    },
    /// Facility #5: Online Monoid Fused Cross-Entropy Loss + In-Place Backward `dlogits` (`llm.c` `fused_classifier3`).
    CrossEntropy {
        logits: String,
        targets: String,
    },
    /// Facility #6: Auto Mixed-Precision Fused Single-Pass RMSNorm + Dynamic Quant W8A8/FP8 Contraction.
    AmpNormMatMul {
        x: String,
        nw: String,
        w: String,
    },
    /// Facility #7: Symbolic Dynamic-Shape Multi-Version Bucketed MatMul (`M_actual` O(1) routed at runtime).
    DynMatMul {
        a: String,
        b: String,
        max_m: usize,
    },
    /// Facility #8B: First-Class Associative Parallel Scan (`h_t = a_t * h_{t-1} + b_t`, Mamba/SSM/Linear-Attn).
    ParallelScan {
        a: String,
        b: String,
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

/// Automatic Liveness-Driven Cache Policy (`llm.c` Facility #12: `__ldcs`/`__stcs` streaming cache hints!).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CachePolicy {
    Default,
    /// Multi-consumer hot activation/weight kept L2/SLC-resident.
    Persistent,
    /// Large single-use tensor (`>= 4 MB`, `use_count <= 1`) marked evict-first (`__ldcs`/`__stcs` / 64B streaming burst).
    Streaming,
}

/// Automatic Logical-vs-Padded Alignment directive (`llm.c` Facility #10: e.g. `50257 -> 50304`, `1003 -> 1024`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PadAlignSpec {
    pub name: String,
    pub pad_rows: usize,
    pub pad_cols: usize,
    /// `true` -> zero-pad AND transpose `[K, N] -> [pad_N, pad_K]` for coalesced 128-bit weight loads!
    pub is_nk: bool,
}

/// Complete compiled multi-stage pipeline artifact (IR + Fused Tile Stages + Static Arena + Roofline).
#[derive(Debug, Clone)]
pub struct CompiledUniversalGraph {
    pub graph: TensorIrGraph,
    pub config: GpuOptConfig,
    pub stages: Vec<fusion::TileStage>,
    pub arena: fusion::StaticArenaPlan,
    pub input_repacks: Vec<(String, LayoutTag)>,
    pub input_pads: Vec<PadAlignSpec>,
    pub cache_policies: BTreeMap<String, CachePolicy>,
    pub roofline: UniversalRoofline,
    pub egraph_stats: egraph::EGraphStats,
    pub ad_stats: ad_spmd::AdSpmdStats,
    pub hetero_plan: hetero_stream::HeteroStreamPlan,
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
                        | "Q8_0" | "Q4_0" | "Q5_0" | "Q5_1" | "Q2_K" | "Q3_K" | "IQ4_XS" | "Q4_K" | "Q5_K" | "Q6_K" | "MXFP4" | "FP4"
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
            "Q5_0" => Some(IrDType::Q5_0),
            "Q5_1" => Some(IrDType::Q5_1),
            "Q2_K" => Some(IrDType::Q2_K),
            "Q3_K" | "Q3_K_M" => Some(IrDType::Q3_K),
            "IQ4_XS" => Some(IrDType::IQ4_XS),
            "Q4_K" | "Q4_K_M" => Some(IrDType::Q4_K),
            "Q5_K" | "Q5_K_M" => Some(IrDType::Q5_K),
            "Q6_K" => Some(IrDType::Q6_K),
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
                "tg_conv2d_3x3" if args.len() >= 4 => {
                    let (Some(x), Some(w)) = (call_arg_ident(args, 0), call_arg_ident(args, 1)) else { return false };
                    let (Some(sx), Some(sw)) = (g.tensors.get(&x).cloned(), g.tensors.get(&w).cloned()) else { return false };
                    let in_h = call_arg_int(args, 2).unwrap_or(32);
                    let in_w = call_arg_int(args, 3).unwrap_or(32);
                    let out_desc = TensorDesc::matrix(in_h * in_w, sw.cols(), sx.dtype);
                    g.push_node(out_var.to_string(), IrOp::Conv2d { x, w, in_h, in_w, kh: 3, kw: 3 }, out_desc, false);
                    true
                }
                "tg_qkv_proj" if args.len() >= 4 => {
                    let (Some(a), Some(w0), Some(w1), Some(w2)) = (
                        call_arg_ident(args, 0),
                        call_arg_ident(args, 1),
                        call_arg_ident(args, 2),
                        call_arg_ident(args, 3),
                    ) else { return false };
                    let (Some(sa), Some(sw0)) = (g.tensors.get(&a).cloned(), g.tensors.get(&w0).cloned()) else { return false };
                    let out_desc = TensorDesc::matrix(sa.rows(), sw0.cols() * 3, sa.dtype);
                    g.push_node(out_var.to_string(), IrOp::ConcatMatMul { a, weights: vec![w0, w1, w2] }, out_desc, false);
                    true
                }
                "tg_adamw_step" if args.len() >= 4 => {
                    let (Some(param), Some(grad), Some(m), Some(v)) = (
                        call_arg_ident(args, 0),
                        call_arg_ident(args, 1),
                        call_arg_ident(args, 2),
                        call_arg_ident(args, 3),
                    ) else { return false };
                    let Some(sp) = g.tensors.get(&param).cloned() else { return false };
                    g.push_node(out_var.to_string(), IrOp::AdamWStep { param, grad, m, v }, sp, false);
                    true
                }
                "tg_cross_entropy" if args.len() >= 2 => {
                    let (Some(logits), Some(targets)) = (call_arg_ident(args, 0), call_arg_ident(args, 1)) else { return false };
                    let Some(sl) = g.tensors.get(&logits).cloned() else { return false };
                    g.push_node(out_var.to_string(), IrOp::CrossEntropy { logits, targets }, sl, false);
                    true
                }
                "tg_amp_matmul" if args.len() >= 3 => {
                    let (Some(x), Some(nw), Some(w)) = (call_arg_ident(args, 0), call_arg_ident(args, 1), call_arg_ident(args, 2)) else { return false };
                    let (Some(sx), Some(sw)) = (g.tensors.get(&x).cloned(), g.tensors.get(&w).cloned()) else { return false };
                    let out_desc = TensorDesc::matrix(sx.rows(), sw.cols(), sx.dtype);
                    g.push_node(out_var.to_string(), IrOp::AmpNormMatMul { x, nw, w }, out_desc, false);
                    true
                }
                "tg_dyn_matmul" if args.len() >= 2 => {
                    let (Some(a), Some(b)) = (call_arg_ident(args, 0), call_arg_ident(args, 1)) else { return false };
                    let (Some(sa), Some(sb)) = (g.tensors.get(&a).cloned(), g.tensors.get(&b).cloned()) else { return false };
                    let out_desc = TensorDesc::matrix(sa.rows(), sb.cols(), sa.dtype);
                    g.push_node(out_var.to_string(), IrOp::DynMatMul { a, b, max_m: sa.rows() }, out_desc, false);
                    true
                }
                "tg_scan" if args.len() >= 2 => {
                    let (Some(a), Some(b)) = (call_arg_ident(args, 0), call_arg_ident(args, 1)) else { return false };
                    let Some(sa) = g.tensors.get(&a).cloned() else { return false };
                    g.push_node(out_var.to_string(), IrOp::ParallelScan { a, b }, sa, false);
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

    // --- Stage 1 Pillar 6 + Stage 5 Facility #13: Reverse-Mode AD (VJP) + Declarative SPMD + ZeRO-1 ---
    let mut ad_stats = if cfg.ad_and_spmd {
        ad_spmd::run_ad_and_spmd(&mut g, wants_autograd, cfg.spmd_devices, cfg.zero1_spmd)
    } else {
        ad_spmd::AdSpmdStats::default()
    };
    if cfg.async_overlap && ad_stats.spmd_devices > 1 {
        ad_stats.micro_tile_chunks = 4;
    }
    if cfg.online_monoid && wants_autograd {
        ad_stats.selective_recomputed = ad_stats.checkpointed_activations;
    }

    // --- Stage 1-5: Fusion, Layout, Auto-Pad, Streaming Cache Policy, Static Arena & Schedule ---
    let (stages, arena, mut input_repacks, input_pads, cache_policies) = fusion::lower_to_tile_stages(&mut g, cfg);

    // --- Stage 6: Universal Heterogeneous Out-of-Core Streaming Planner (VRAM Budget, NUMA 4R, Dual-Stream DMA, Hybrid CPU+GPU) ---
    let hetero_plan = hetero_stream::plan_hetero_stream(&g, cfg, &mut input_repacks);

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
    let uses_low_bit = g.tensors.values().any(|t| matches!(t.dtype, IrDType::FP8 | IrDType::I8 | IrDType::I4 | IrDType::Q4_0 | IrDType::Q5_0 | IrDType::Q5_1 | IrDType::Q2_K | IrDType::Q3_K | IrDType::IQ4_XS | IrDType::Q4_K | IrDType::Q5_K | IrDType::Q6_K | IrDType::MXFP4));

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
        input_pads,
        cache_policies,
        roofline,
        egraph_stats,
        ad_stats,
        hetero_plan,
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
            IrOp::Conv2d { x, w, kh, kw, .. } => {
                let sx = &g.tensors[x];
                let sw = &g.tensors[w];
                let k_eff = sx.cols() * (*kh) * (*kw);
                flops += 2.0 * (sx.rows() as f64) * (k_eff as f64) * (sw.cols() as f64);
            }
            IrOp::ConcatMatMul { a, weights } => {
                let sa = &g.tensors[a];
                let total_n: usize = weights.iter().map(|w| g.tensors[w].cols()).sum();
                flops += 2.0 * (sa.rows() as f64) * (sa.cols() as f64) * (total_n as f64);
            }
            IrOp::AmpNormMatMul { x, w, .. } => {
                let sx = &g.tensors[x];
                let sw = &g.tensors[w];
                flops += 2.0 * (sx.rows() as f64) * (sx.cols() as f64) * (sw.cols() as f64);
            }
            IrOp::DynMatMul { a, b, .. } => {
                let sa = &g.tensors[a];
                let sb = &g.tensors[b];
                flops += 2.0 * (sa.rows() as f64) * (sa.cols() as f64) * (sb.cols() as f64);
            }
            IrOp::AdamWStep { .. } => flops += 12.0 * (n.desc.numel() as f64),
            IrOp::CrossEntropy { .. } => flops += 8.0 * (n.desc.numel() as f64),
            IrOp::ParallelScan { .. } => flops += 4.0 * (n.desc.numel() as f64),
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
            | IrOp::BlockSparseMatMul { a, b, .. }
            | IrOp::DynMatMul { a, b, .. }
            | IrOp::ParallelScan { a, b } => {
                bytes += g.tensors[a].bytes_f64() + g.tensors[b].bytes_f64() + n.desc.bytes_f64();
            }
            IrOp::ConcatMatMul { a, weights } => {
                let w_bytes: f64 = weights.iter().map(|w| g.tensors[w].bytes_f64()).sum();
                bytes += g.tensors[a].bytes_f64() + w_bytes + n.desc.bytes_f64();
            }
            IrOp::AmpNormMatMul { x, nw, w } => {
                bytes += g.tensors[x].bytes_f64() + g.tensors[nw].bytes_f64() + g.tensors[w].bytes_f64() + n.desc.bytes_f64();
            }
            IrOp::AdamWStep { param, grad, m, v } => {
                bytes += g.tensors[param].bytes_f64() * 2.0
                    + g.tensors[grad].bytes_f64()
                    + g.tensors[m].bytes_f64() * 2.0
                    + g.tensors[v].bytes_f64() * 2.0;
            }
            IrOp::CrossEntropy { logits, targets } => {
                // Naive materializes probs[BT, V] FP32 in HBM -> 3x logits!
                bytes += g.tensors[logits].bytes_f64() * 4.0 + g.tensors[targets].bytes_f64();
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
