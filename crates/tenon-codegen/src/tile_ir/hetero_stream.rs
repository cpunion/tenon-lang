//! Stage 6 Universal Heterogeneous Out-of-Core Streaming Compiler Pass (`hetero_stream.rs`).
//!
//! Unifies Tenon's hard-VRAM-capped multi-tier streaming architecture (Metal + CUDA + AVX2/NEON)
//! directly inside the declarative Tile-IR compiler pipeline (`@gpu_graph("stream:...")`):
//!   - **Facility #14 (`vram_budget_plan`)**: Hard-VRAM-Cap Budget Solver — partitions Core Attention,
//!     KV-Cache (`Q8_0` 1.0625 B/elem vs `FP16` 2.0 B/elem), Dense prefix GPU FFNs or Sparse MoE
//!     Per-Layer LRU slots, `F_NOCACHE`/`POSIX_FADV_DONTNEED` NVMe mode, and Batched Prefill chunk size!
//!   - **Facility #15 (`numa_repack_layout`)**: NUMA-Local Parallel First-Touch 4-Row Contiguous
//!     Weight Repacking (`NumaRepackQ4_4R` 144B/72B, `NumaRepackQ4K_4R`) eliminating cross-socket stalls!
//!   - **Facility #16 (`dual_stream_dma`)**: Hardware Dual-Stream `Compute || DMA` Pipeline
//!     (`cuStreamWaitEvent` / `MTLEvent` ring barriers + `MTLPurgeableStateEmpty` immediate page reclaim
//!     + off-thread background hot-expert promoter).
//!   - **Facility #17 (`hetero_dispatch`)**: Roofline-Driven GPU+CPU Hybrid Dispatch — evaluates
//!     `BW_DDR > 1.2 * BW_PCIe` to route unpinned Dense FFNs / missed MoE experts onto AVX2/NEON
//!     INT16 (`_mm256_madd_epi16`) `Main=tid0` container-cgroup-aware worker pool with zero PCIe traffic!

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeteroDispatchMode {
    /// Entire model fits in VRAM -> single CUDA Graph / Metal Single-CB resident execution.
    ResidentGpu,
    /// Unpinned FFNs / missed MoE experts execute in-place on NUMA-local AVX2/NEON INT16 CPU pool (`DDR >> PCIe`).
    HybridCpuInt16,
    /// Hardware Dual-Stream `Compute || PCIe/Unified DMA` ring (`cuStreamWaitEvent` / `MTLEvent`).
    AsyncPcieRing,
    /// Host RAM < model size -> strict `F_NOCACHE` / `POSIX_FADV_DONTNEED` direct NVMe streaming.
    DirectNvmeStream,
}

impl HeteroDispatchMode {
    pub fn tag(self) -> &'static str {
        match self {
            Self::ResidentGpu => "RESIDENT_GPU",
            Self::HybridCpuInt16 => "HYBRID_CPU_INT16",
            Self::AsyncPcieRing => "ASYNC_DMA_RING",
            Self::DirectNvmeStream => "DIRECT_NVME_NOCACHE",
        }
    }
}

/// Complete Stage 6 Heterogeneous Streaming Execution Plan attached to `CompiledUniversalGraph`.
#[derive(Debug, Clone)]
pub struct HeteroStreamPlan {
    pub active: bool,
    pub vram_cap_bytes: u64,
    pub ram_cap_bytes: u64,
    pub n_layers: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub max_ctx: usize,
    pub kv_quant_q8: bool,
    pub kv_cache_bytes: u64,
    pub kv_bytes_saved: u64,
    pub core_vram_bytes: u64,
    pub total_model_bytes: u64,
    pub pinned_gpu_ffn_layers: usize,
    pub lru_slots_per_layer: usize,
    pub cpu_offload_layers: usize,
    pub use_nocache_nvme: bool,
    pub prefill_chunk_tokens: usize,
    pub dma_ring_slots: usize,
    pub async_promoter: bool,
    pub purgeable_reclaim: bool,
    pub cgroup_main_tid0: bool,
    pub tp_devices: usize,
    pub ep_devices: usize,
    pub nccl_collectives: bool,
    pub marlin_cp_async: bool,
    pub cont_batch_slots: usize,
    pub cow_prefix_blocks: bool,
    pub mla_latent_dim: usize,
    pub paged_kv_block_size: usize,
    pub radix_prefix_cache: bool,
    pub speculative_k: usize,
    pub tree_spec_width: usize,
    pub mtp_nextn_layers: usize,
    pub vision_mmproj_tokens: usize,
    pub small_batch_shared_x: bool,
    pub hybrid_2d_tp_ep: bool,
    pub chunked_prefill_size: usize,
    pub lora_max_rank: usize,
    pub vision_anyres_tiles: usize,
    pub vision_mrope_2d: bool,
    pub fp8_mxfp4_native: bool,
    pub fp8_mxfp4_tc_amx: bool,
    pub q4k_q6k_marlin: bool,
    pub paged_cb_tp_mla_spec: bool,
    pub moe_grouped_gemm: bool,
    pub audio_mel_tokens: usize,
    pub vision_prefix_cache: bool,
    pub dispatch_mode: HeteroDispatchMode,
    pub numa_repack_weights: Vec<(String, LayoutTag)>,
    pub projected_decode_tok_s: f64,
    pub projected_prefill_tok_s: f64,
}

impl Default for HeteroStreamPlan {
    fn default() -> Self {
        Self {
            active: false,
            vram_cap_bytes: 0,
            ram_cap_bytes: 0,
            n_layers: 1,
            n_experts: 0,
            top_k: 0,
            max_ctx: 2048,
            kv_quant_q8: false,
            kv_cache_bytes: 0,
            kv_bytes_saved: 0,
            core_vram_bytes: 0,
            total_model_bytes: 0,
            pinned_gpu_ffn_layers: 1,
            lru_slots_per_layer: 0,
            cpu_offload_layers: 0,
            use_nocache_nvme: false,
            prefill_chunk_tokens: 1,
            dma_ring_slots: 1,
            async_promoter: false,
            purgeable_reclaim: false,
            cgroup_main_tid0: false,
            tp_devices: 1,
            ep_devices: 1,
            nccl_collectives: false,
            marlin_cp_async: false,
            cont_batch_slots: 1,
            cow_prefix_blocks: false,
            mla_latent_dim: 0,
            paged_kv_block_size: 16,
            radix_prefix_cache: false,
            speculative_k: 0,
            tree_spec_width: 0,
            mtp_nextn_layers: 0,
            vision_mmproj_tokens: 0,
            small_batch_shared_x: false,
            hybrid_2d_tp_ep: false,
            chunked_prefill_size: 0,
            lora_max_rank: 0,
            vision_anyres_tiles: 0,
            vision_mrope_2d: false,
            fp8_mxfp4_native: false,
            fp8_mxfp4_tc_amx: false,
            q4k_q6k_marlin: false,
            paged_cb_tp_mla_spec: false,
            moe_grouped_gemm: false,
            audio_mel_tokens: 0,
            vision_prefix_cache: false,
            dispatch_mode: HeteroDispatchMode::ResidentGpu,
            numa_repack_weights: Vec::new(),
            projected_decode_tok_s: 0.0,
            projected_prefill_tok_s: 0.0,
        }
    }
}

fn parse_bytes_token(s: &str) -> Option<u64> {
    let u = s.trim().to_ascii_uppercase();
    if let Some(v) = u.strip_suffix("GIB").or_else(|| u.strip_suffix("GB")).or_else(|| u.strip_suffix('G')) {
        v.parse::<f64>().ok().map(|x| (x * 1024.0 * 1024.0 * 1024.0) as u64)
    } else if let Some(v) = u.strip_suffix("MIB").or_else(|| u.strip_suffix("MB")).or_else(|| u.strip_suffix('M')) {
        v.parse::<f64>().ok().map(|x| (x * 1024.0 * 1024.0) as u64)
    } else {
        u.parse::<u64>().ok()
    }
}

/// Runs Stage 6 Passes (#14 VRAM Budget Planner, #15 NUMA 4R Repack Layout,
/// #16 Dual-Stream DMA Pipeline, #17 Roofline Hybrid CPU+GPU Dispatch).
pub fn plan_hetero_stream(
    g: &TensorIrGraph,
    cfg: GpuOptConfig,
    input_repacks: &mut Vec<(String, LayoutTag)>,
) -> HeteroStreamPlan {
    let any_stage6 = cfg.vram_budget_plan || cfg.numa_repack_layout || cfg.dual_stream_dma || cfg.hetero_dispatch;
    let is_stream_graph = g.attr_mode.starts_with("stream");
    if !any_stage6 && !is_stream_graph {
        return HeteroStreamPlan::default();
    }

    // Parse optional `@gpu_graph("stream:vram=24G,ram=64G,layers=80,experts=0,ctx=16384,kv=auto,ddr=70,pcie=24")`
    let mut vram_cap_bytes: u64 = 24u64 * 1024 * 1024 * 1024;
    let mut ram_cap_bytes: u64 = 64u64 * 1024 * 1024 * 1024;
    let mut n_layers: usize = 1;
    let mut n_experts: usize = 0;
    let mut top_k: usize = 0;
    let mut max_ctx: usize = 4096;
    let mut kv_mode = "auto";
    let mut ddr_bw_gbs: f64 = 70.0;
    let mut pcie_bw_gbs: f64 = 24.0;
    let mut gpu_bw_gbs: f64 = 854.0;
    let mut explicit_kv_dim: Option<usize> = None;
    let mut tp_devices: usize = cfg.spmd_devices.max(1);
    let mut ep_devices: usize = 1;
    let mut cont_batch_slots: usize = if cfg.paged_cont_batch { 4 } else { 1 };
    let mut mla_latent_dim: usize = 0;
    let mut speculative_k: usize = 0;
    let mut tree_spec_width: usize = 0;
    let mut mtp_nextn_layers: usize = 0;
    let mut vision_mmproj_tokens: usize = 0;
    let mut radix_prefix_cache: bool = cfg.vram_budget_plan;
    let mut chunked_prefill_size: usize = if cfg.chunked_prefill_cb { 128 } else { 0 };
    let mut lora_max_rank: usize = 0;
    let mut vision_anyres_tiles: usize = 1;
    let mut vision_mrope_2d: bool = false;
    let mut fp8_mxfp4_native: bool = cfg.anyres_mrope_fp8_mxfp4;
    let mut audio_mel_tokens: usize = 0;
    let mut vision_prefix_cache: bool = cfg.vlm_mrope_audio_prefix_cache;

    if let Some(rest) = g.attr_mode.strip_prefix("stream:") {
        for part in rest.split(',') {
            let mut kv = part.splitn(2, '=');
            let k = kv.next().unwrap_or("").trim();
            let v = kv.next().unwrap_or("").trim();
            match k {
                "vram" | "vram_cap" => {
                    if let Some(b) = parse_bytes_token(v) {
                        vram_cap_bytes = b;
                    }
                }
                "ram" | "ram_cap" => {
                    if let Some(b) = parse_bytes_token(v) {
                        ram_cap_bytes = b;
                    }
                }
                "layers" | "L" => {
                    if let Ok(n) = v.parse::<usize>() {
                        n_layers = n.max(1);
                    }
                }
                "experts" | "E" => {
                    if let Ok(n) = v.parse::<usize>() {
                        n_experts = n;
                    }
                }
                "topk" | "K" => {
                    if let Ok(n) = v.parse::<usize>() {
                        top_k = n;
                    }
                }
                "ctx" | "seq" => {
                    if let Ok(n) = v.parse::<usize>() {
                        max_ctx = n.max(256);
                    }
                }
                "kv" => kv_mode = v,
                "kv_dim" => {
                    if let Ok(n) = v.parse::<usize>() {
                        explicit_kv_dim = Some(n);
                    }
                }
                "ddr" | "ddr_bw" => {
                    if let Ok(f) = v.parse::<f64>() {
                        ddr_bw_gbs = f.max(1.0);
                    }
                }
                "pcie" | "pcie_bw" => {
                    if let Ok(f) = v.parse::<f64>() {
                        pcie_bw_gbs = f.max(1.0);
                    }
                }
                "gpu_bw" => {
                    if let Ok(f) = v.parse::<f64>() {
                        gpu_bw_gbs = f.max(10.0);
                    }
                }
                "tp" | "tp_devices" => {
                    if let Ok(n) = v.parse::<usize>() {
                        tp_devices = n.max(1);
                    }
                }
                "ep" | "ep_devices" => {
                    if let Ok(n) = v.parse::<usize>() {
                        ep_devices = n.max(1);
                    }
                }
                "cb" | "cont_batch" | "slots" => {
                    if let Ok(n) = v.parse::<usize>() {
                        cont_batch_slots = n.max(1);
                    }
                }
                "mla" | "mla_latent" => {
                    if let Ok(n) = v.parse::<usize>() {
                        mla_latent_dim = n;
                    }
                }
                "spec" => {
                    if let Ok(n) = v.parse::<usize>() {
                        speculative_k = n;
                    }
                }
                "mtp" => {
                    if let Ok(n) = v.parse::<usize>() {
                        mtp_nextn_layers = n;
                        if speculative_k == 0 { speculative_k = n; }
                    }
                }
                "tree" | "tree_spec" => {
                    if let Ok(n) = v.parse::<usize>() {
                        tree_spec_width = n;
                    }
                }
                "mmproj" | "vision" => {
                    if let Ok(n) = v.parse::<usize>() {
                        vision_mmproj_tokens = n;
                    }
                }
                "chunk_pf" | "prefill_chunk" => {
                    if let Ok(n) = v.parse::<usize>() {
                        chunked_prefill_size = n;
                    }
                }
                "lora" | "lora_rank" => {
                    if let Ok(n) = v.parse::<usize>() {
                        lora_max_rank = n;
                    }
                }
                "anyres" => {
                    if let Ok(n) = v.parse::<usize>() {
                        vision_anyres_tiles = n.max(1);
                    }
                }
                "mrope" => vision_mrope_2d = v != "0" && v != "false",
                "fp8_mxfp4" => fp8_mxfp4_native = v != "0" && v != "false",
                "audio" | "audio_mel" => {
                    if let Ok(n) = v.parse::<usize>() {
                        audio_mel_tokens = n;
                    }
                }
                "vis_cache" => vision_prefix_cache = v != "0" && v != "false",
                "prefix" => radix_prefix_cache = v != "0" && v != "false",
                _ => {}
            }
        }
    }

    // Measure single-layer Attn vs FFN weight sizes directly from the declarative `@gpu_graph` params!
    let mut single_layer_attn_bytes: u64 = 0;
    let mut single_layer_ffn_bytes: u64 = 0;
    let mut inferred_kv_dim: usize = 1024;
    let mut numa_repack_weights: Vec<(String, LayoutTag)> = Vec::new();

    for n in &g.nodes {
        if let IrOp::Attention { kv_dim, .. } = &n.op {
            inferred_kv_dim = *kv_dim;
        }
    }
    let kv_dim = explicit_kv_dim.unwrap_or(inferred_kv_dim);

    for (pname, pdesc) in &g.params {
        let lower = pname.to_ascii_lowercase();
        let sz = pdesc.alloc_bytes() as u64;
        let is_act = lower == "x" || lower.starts_with("act") || lower.starts_with("in_") || lower == "q" || lower == "k" || lower == "v";
        if is_act {
            continue;
        }
        let is_ffn = lower.contains("gate") || lower.contains("up") || lower.contains("down") || lower.contains("ffn") || lower.contains("exp") || lower.contains("w1") || lower.contains("w2") || lower.contains("w3");
        if is_ffn {
            single_layer_ffn_bytes += sz;
            if cfg.numa_repack_layout {
                let tag = match pdesc.dtype {
                    IrDType::Q2_K | IrDType::Q3_K | IrDType::IQ4_XS | IrDType::Q4_K | IrDType::Q5_K | IrDType::Q6_K => Some(LayoutTag::NumaRepackQ4K_4R),
                    IrDType::Q4_0 | IrDType::Q5_0 | IrDType::Q5_1 | IrDType::I4 | IrDType::MXFP4 => Some(LayoutTag::NumaRepackQ4_4R),
                    _ => None,
                };
                if let Some(t) = tag {
                    numa_repack_weights.push((pname.clone(), t));
                }
            }
        } else {
            single_layer_attn_bytes += sz;
        }
    }

    // Total streamable model bytes across `n_layers` (and `n_experts` for Sparse MoE)
    let ffn_per_layer_total = if n_experts > 0 {
        single_layer_ffn_bytes.saturating_mul(n_experts as u64)
    } else {
        single_layer_ffn_bytes
    };
    let single_expert_bytes = if n_experts > 0 { single_layer_ffn_bytes.max(1) } else { 0 };
    let total_attn_bytes = single_layer_attn_bytes.saturating_mul(n_layers as u64);
    let total_ffn_bytes = ffn_per_layer_total.saturating_mul(n_layers as u64);
    // Estimate vocab/norm overhead (~1.2 GiB for 70B/106B class if multi-layer, else 0)
    let head_overhead_bytes: u64 = if n_layers >= 32 { 1_100_000_000 } else { 0 };
    let total_model_bytes = total_attn_bytes + total_ffn_bytes + head_overhead_bytes;

    // Facility #14: Hard-VRAM-Cap Budget Solver + Compressed Latent MLA KV + Q8_0 KV-Cache + Multi-GPU TP!
    let kv_fp16_bytes = (n_layers as u64) * 2 * (max_ctx as u64) * (kv_dim as u64) * 2;
    let kv_mla_bytes = if mla_latent_dim > 0 {
        // Matrix-Absorbed MLA stores ONLY compressed `[c_kv || k_pe]` (single FP16 vector, NO separate V cache!)
        (n_layers as u64) * (max_ctx as u64) * (mla_latent_dim as u64) * 2
    } else {
        0
    };
    let kv_q8_bytes = ((n_layers as u64) * 2 * (max_ctx as u64) * (kv_dim as u64) * 17) / 16; // 34B per 32 elems = 1.0625 B/elem!
    let kv_quant_q8 = if mla_latent_dim > 0 {
        false
    } else if !cfg.vram_budget_plan {
        kv_mode == "q8"
    } else {
        match kv_mode {
            "q8" | "q8_0" => true,
            "fp16" | "f16" => false,
            _ => max_ctx > 2048 || kv_fp16_bytes > (vram_cap_bytes / 10),
        }
    };
    let kv_cache_bytes = if mla_latent_dim > 0 {
        kv_mla_bytes
    } else if kv_quant_q8 {
        kv_q8_bytes
    } else {
        kv_fp16_bytes
    };
    let kv_bytes_saved = kv_fp16_bytes.saturating_sub(kv_cache_bytes);

    // Multi-GPU Tensor + Expert Parallelism (`tp_devices * ep_devices`): aggregate VRAM scales linearly across GPUs!
    let world_gpus = tp_devices.saturating_mul(ep_devices).max(1);
    let total_vram_cap = vram_cap_bytes.saturating_mul(world_gpus as u64);
    let nccl_collectives = cfg.moe_ep_nccl && world_gpus > 1;
    let marlin_cp_async = cfg.marlin_w4a16_mma && cfg.vram_budget_plan;
    let cow_prefix_blocks = cfg.paged_cont_batch && radix_prefix_cache;
    let usable_vram = (total_vram_cap as f64 * 0.94) as u64;
    let base_core = total_attn_bytes + kv_cache_bytes + head_overhead_bytes + (64 * 1024 * 1024);

    let (pinned_gpu_ffn_layers, lru_slots_per_layer, cpu_offload_layers, core_vram_bytes) = if !cfg.vram_budget_plan {
        // Without Stage 6 VRAM Budget Planner: naive static allocation (cannot fit oversized models!)
        let naive_pinned = if total_model_bytes <= usable_vram { n_layers } else { n_layers / 4 };
        (naive_pinned, if n_experts > 0 { n_experts / 8 } else { 0 }, n_layers.saturating_sub(naive_pinned), base_core)
    } else if n_experts > 0 {
        // Sparse MoE: Per-Layer GPU VRAM LRU Expert Pool!
        let avail_for_lru = usable_vram.saturating_sub(base_core);
        let bytes_per_slot_all_layers = single_expert_bytes.saturating_mul(n_layers as u64).max(1);
        let slots = ((avail_for_lru / bytes_per_slot_all_layers) as usize).min(n_experts);
        let pool_bytes = (slots as u64) * bytes_per_slot_all_layers;
        (0, slots, if slots < n_experts { n_layers } else { 0 }, base_core + pool_bytes)
    } else {
        // Dense LLM: Pin All-Layers Attention + Maximal Prefix GPU FFNs!
        let avail_for_ffn = usable_vram.saturating_sub(base_core);
        let pinned = if single_layer_ffn_bytes > 0 {
            ((avail_for_ffn / single_layer_ffn_bytes) as usize).min(n_layers)
        } else {
            n_layers
        };
        let offload = n_layers.saturating_sub(pinned);
        (pinned, 0, offload, base_core + (pinned as u64) * single_layer_ffn_bytes)
    };

    // Host RAM check -> strict `F_NOCACHE` / `POSIX_FADV_DONTNEED` NVMe mode if model exceeds 85% host RAM!
    let use_nocache_nvme = cfg.vram_budget_plan && total_model_bytes > ((ram_cap_bytes as f64 * 0.85) as u64);

    // Facility #16: Dual-Stream `Compute || DMA` Pipeline
    let dma_ring_slots = if !cfg.dual_stream_dma {
        1
    } else if n_experts > 0 {
        16
    } else {
        4
    };
    let async_promoter = cfg.dual_stream_dma && n_experts > 0 && lru_slots_per_layer < n_experts;
    let purgeable_reclaim = cfg.dual_stream_dma && (use_nocache_nvme || cpu_offload_layers > 0);

    // Facility #17: Roofline-Driven GPU+CPU Heterogeneous Dispatch
    let fits_resident = if n_experts > 0 {
        lru_slots_per_layer >= n_experts
    } else {
        pinned_gpu_ffn_layers >= n_layers
    };
    let dispatch_mode = if fits_resident {
        HeteroDispatchMode::ResidentGpu
    } else if use_nocache_nvme {
        HeteroDispatchMode::DirectNvmeStream
    } else if cfg.hetero_dispatch && ddr_bw_gbs > pcie_bw_gbs * 1.2 {
        HeteroDispatchMode::HybridCpuInt16
    } else {
        HeteroDispatchMode::AsyncPcieRing
    };

    // Promote NUMA 4R repack directives into `input_repacks` when hybrid CPU offload is active!
    if cfg.numa_repack_layout && dispatch_mode == HeteroDispatchMode::HybridCpuInt16 {
        for (wname, tag) in &numa_repack_weights {
            if let Some(slot) = input_repacks.iter_mut().find(|(k, _)| k == wname) {
                slot.1 = *tag;
            } else {
                input_repacks.push((wname.clone(), *tag));
            }
        }
    }

    // Batched Prefill Chunk Solver (`B = 32 / 64 / 128` within remaining scratch VRAM!)
    let headroom = usable_vram.saturating_sub(core_vram_bytes);
    let prefill_chunk_tokens = if !cfg.vram_budget_plan {
        1
    } else if headroom >= 512 * 1024 * 1024 {
        128
    } else if headroom >= 256 * 1024 * 1024 {
        64
    } else {
        32
    };

    // Analytical Multi-Tier Roofline Decode & Batched Prefill Projections
    let gpu_bytes_per_tok = if n_experts > 0 {
        let hit_ratio = (lru_slots_per_layer as f64 / n_experts.max(1) as f64).min(1.0);
        let active_exp_bytes = (top_k.max(1) as f64) * (single_expert_bytes as f64) * (n_layers as f64);
        total_attn_bytes as f64 + head_overhead_bytes as f64 + active_exp_bytes * hit_ratio
    } else {
        total_attn_bytes as f64 + head_overhead_bytes as f64 + (pinned_gpu_ffn_layers as f64) * (single_layer_ffn_bytes as f64)
    };
    let offload_bytes_per_tok = if n_experts > 0 {
        let miss_ratio = 1.0 - (lru_slots_per_layer as f64 / n_experts.max(1) as f64).min(1.0);
        (top_k.max(1) as f64) * (single_expert_bytes as f64) * (n_layers as f64) * miss_ratio
    } else {
        (cpu_offload_layers as f64) * (single_layer_ffn_bytes as f64)
    };
    let t_gpu_s = gpu_bytes_per_tok / (gpu_bw_gbs * (world_gpus as f64) * 1e9);
    let offload_bw = match dispatch_mode {
        HeteroDispatchMode::ResidentGpu => gpu_bw_gbs,
        HeteroDispatchMode::HybridCpuInt16 => ddr_bw_gbs,
        HeteroDispatchMode::AsyncPcieRing => pcie_bw_gbs,
        HeteroDispatchMode::DirectNvmeStream => 6.0, // Gen4 NVMe ~6 GB/s
    };
    let t_offload_s = if offload_bytes_per_tok > 0.0 {
        offload_bytes_per_tok / (offload_bw * 1e9)
    } else {
        0.0
    };
    let t_tok_s = (t_gpu_s + t_offload_s).max(1e-6);
    let projected_decode_tok_s = 1.0 / t_tok_s;
    // Batched Prefill amortizes weight reads across `prefill_chunk_tokens`!
    let projected_prefill_tok_s = projected_decode_tok_s * (prefill_chunk_tokens as f64 * 0.65).max(1.0);

    HeteroStreamPlan {
        active: is_stream_graph || !numa_repack_weights.is_empty(),
        vram_cap_bytes,
        ram_cap_bytes,
        n_layers,
        n_experts,
        top_k,
        max_ctx,
        kv_quant_q8,
        kv_cache_bytes,
        kv_bytes_saved,
        core_vram_bytes,
        total_model_bytes,
        pinned_gpu_ffn_layers,
        lru_slots_per_layer,
        cpu_offload_layers,
        use_nocache_nvme,
        prefill_chunk_tokens,
        dma_ring_slots,
        async_promoter,
        purgeable_reclaim,
        cgroup_main_tid0: cfg.hetero_dispatch,
        tp_devices,
        ep_devices,
        nccl_collectives,
        marlin_cp_async,
        cont_batch_slots,
        cow_prefix_blocks,
        mla_latent_dim,
        paged_kv_block_size: 16,
        radix_prefix_cache,
        speculative_k,
        tree_spec_width,
        mtp_nextn_layers,
        vision_mmproj_tokens,
        small_batch_shared_x: cfg.small_batch_shared_x,
        hybrid_2d_tp_ep: cfg.hybrid_2d_tp_ep && tp_devices > 1 && ep_devices > 1,
        chunked_prefill_size: if cfg.chunked_prefill_cb { chunked_prefill_size } else { 0 },
        lora_max_rank: if cfg.multi_lora_zero_copy { lora_max_rank } else { 0 },
        vision_anyres_tiles: if vision_mmproj_tokens > 0 { vision_anyres_tiles } else { 0 },
        vision_mrope_2d: cfg.anyres_mrope_fp8_mxfp4 && vision_mrope_2d,
        fp8_mxfp4_native: cfg.anyres_mrope_fp8_mxfp4 && fp8_mxfp4_native,
        fp8_mxfp4_tc_amx: cfg.fp8_mxfp4_tc_amx && fp8_mxfp4_native,
        q4k_q6k_marlin: cfg.q4k_q6k_marlin_pipeline,
        paged_cb_tp_mla_spec: cfg.paged_cb_tp_mla_spec && cfg.paged_cont_batch,
        moe_grouped_gemm: cfg.moe_grouped_gemm_batch && n_experts > 0,
        audio_mel_tokens: if cfg.vlm_mrope_audio_prefix_cache { audio_mel_tokens } else { 0 },
        vision_prefix_cache: cfg.vlm_mrope_audio_prefix_cache && vision_prefix_cache && vision_mmproj_tokens > 0,
        dispatch_mode,
        numa_repack_weights,
        projected_decode_tok_s,
        projected_prefill_tok_s,
    }
}
