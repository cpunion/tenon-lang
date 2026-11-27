use super::*;

#[test]
fn stage1_to_stage3_toggles_and_ablation_specs_parse_cleanly() {
    let none = GpuOptConfig::parse_spec("none");
    assert!(!none.tile_mma && !none.general_fusion && !none.layout_prop && !none.stream_k);
    assert!(!none.arena_async && !none.pgo_autotune && !none.warp_spec && !none.egraph_rewrite);

    let s1 = GpuOptConfig::parse_spec("stage1");
    assert!(s1.tile_mma && s1.general_fusion && s1.layout_prop && s1.stream_k && s1.arena_async && s1.ad_and_spmd);
    assert!(!s1.pgo_autotune && !s1.warp_spec && !s1.egraph_rewrite && !s1.ragged_sparse && !s1.mxfp4_quant);

    let s2 = GpuOptConfig::parse_spec("stage2");
    assert!(s2.pgo_autotune && s2.warp_spec && s2.egraph_rewrite);
    assert!(!s2.ragged_sparse && !s2.mxfp4_quant);

    let all_minus = GpuOptConfig::parse_spec("all,-warp_spec,-mxfp4,spmd=4");
    assert!(all_minus.tile_mma && !all_minus.warp_spec && !all_minus.mxfp4_quant && all_minus.spmd_devices == 4);
}

#[test]
fn universal_fusion_and_arena_and_layout_toggle_ablation() {
    let src = r#"
@gpu_graph("tensor_dataflow")
pub fn generic_mlp_block(
    x    Matrix[64, 512, Float16],
    nw   Matrix[1, 512, Float16],
    w1   Matrix[512, 2048, Float16],
    b1   Matrix[1, 2048, Float16],
    w2   Matrix[2048, 512, Float16],
) Matrix[64, 512, Float16] {
    let h = tg_layernorm(x, nw)
    let m1 = tg_matmul(h, w1)
    let a1 = tg_add(m1, b1)
    let g1 = tg_gelu(a1)
    let m2 = tg_matmul(g1, w2)
    tg_add(x, m2)
}
"#;
    let prog = tenon_syntax::parse(src).expect("parse");
    let unopt = compile_universal_graphs(&prog, GpuOptConfig::none());
    let opt = compile_universal_graphs(&prog, GpuOptConfig::stage1());
    assert_eq!(unopt.len(), 1);
    assert_eq!(opt.len(), 1);
    // Unoptimized has 6 separate stages; Stage 1 fuses LayerNorm+MatMul+Bias+GELU and MatMul+Residual -> 2 stages!
    assert_eq!(unopt[0].stages.len(), 6);
    assert_eq!(opt[0].stages.len(), 2);
    assert!(opt[0].roofline.bytes_saved_pct > unopt[0].roofline.bytes_saved_pct);
    assert!(!opt[0].input_repacks.is_empty(), "expected layout propagation repacks");
    assert!(unopt[0].input_repacks.is_empty());

    let (msl_none, _) = emit_metal::emit_metal_graph_and_plan(&unopt[0]);
    let (msl_opt, plan_opt) = emit_metal::emit_metal_graph_and_plan(&opt[0]);
    assert!(!msl_none.contains("simdgroup_multiply_accumulate"));
    assert!(msl_opt.contains("simdgroup_multiply_accumulate"));
    assert!(msl_opt.contains("tanh(0.7978845608f")); // In-register fused GELU epilogue!
    assert!(plan_opt.contains("TRANSPOSE_NK"));
    assert!(plan_opt.contains("single_cb=1"));
}

#[test]
fn stage2_egraph_reassociates_bottleneck_chain_for_massive_flop_reduction() {
    // MLA / LoRA / Bottleneck chain: ([512, 256] @ [256, 2048]) @ [2048, 128]
    // Left-assoc:  2*(512*256*2048 + 512*2048*128) = 805M FLOPs
    // Right-assoc: 2*(256*2048*128 + 512*256*128)  = 168M FLOPs (4.8x reduction!)
    let src = r#"
@gpu_graph("tensor_dataflow")
pub fn bottleneck_chain(
    x  Matrix[512, 256, Float16],
    w1 Matrix[256, 2048, Float16],
    w2 Matrix[2048, 128, Float16],
) Matrix[512, 128, Float16] {
    let t = tg_matmul(x, w1)
    tg_matmul(t, w2)
}
"#;
    let prog = tenon_syntax::parse(src).expect("parse");
    let no_eg = compile_universal_graphs(&prog, GpuOptConfig::parse_spec("stage1"));
    let with_eg = compile_universal_graphs(&prog, GpuOptConfig::parse_spec("stage1,+egraph"));
    assert_eq!(no_eg[0].egraph_stats.reassociated_chains, 0);
    assert_eq!(with_eg[0].egraph_stats.reassociated_chains, 1);
    assert!(with_eg[0].roofline.flops_saved_pct > 70.0, "expected >70% FLOPs eliminated: {:.1}%", with_eg[0].roofline.flops_saved_pct);
    assert!(with_eg[0].roofline.opt_flops < no_eg[0].roofline.opt_flops * 0.3);
}

#[test]
fn stage1_reverse_ad_and_spmd_sharding_across_metal_cuda_tpu() {
    let src = r#"
@gpu_graph("autograd")
pub fn train_mlp(
    x  Matrix[64, 512, Float16],
    w1 Matrix[512, 1024, Float16],
    w2 Matrix[1024, 512, Float16],
) Matrix[64, 512, Float16] {
    let h = tg_matmul(x, w1)
    let a = tg_silu(h)
    tg_matmul(a, w2)
}
"#;
    let prog = tenon_syntax::parse(src).expect("parse");
    let cg = compile_universal_graphs(&prog, GpuOptConfig::parse_spec("stage2,spmd=4"));
    assert_eq!(cg.len(), 1);
    assert!(cg[0].ad_stats.backward_nodes_synthesized >= 4, "expected forward+backward VJP nodes");
    assert!(cg[0].ad_stats.checkpointed_activations >= 1);
    assert!(cg[0].ad_stats.collectives_inserted >= 1);

    let cuda = emit_cuda_tpu::emit_universal_cuda(&cg[0]);
    assert!(cuda.contains("Warp-Specialized Producer/Consumer Pipeline"));
    assert!(cuda.contains("ncclAllReduce"));

    let shlo = emit_cuda_tpu::emit_universal_stablehlo(&cg[0]);
    assert!(shlo.contains("mhlo.sharding"));
    assert!(shlo.contains("stablehlo.all_reduce"));
}

#[test]
fn stage3_mxfp4_and_ragged_sparse_and_online_flash_attention() {
    let src = r#"
@gpu_graph("tensor_dataflow")
pub fn decode_mxfp4_and_sparse(
    x  Matrix[1, 2048, Float16],
    w  Matrix[2048, 2048, Float16],
) Matrix[1, 2048, Float16] {
    tg_matmul(x, w)
}

@gpu_graph("tensor_dataflow")
pub fn long_attn(
    q Matrix[512, 512, Float16],
    k Matrix[512, 512, Float16],
    v Matrix[512, 512, Float16],
) Matrix[512, 512, Float16] {
    tg_gqa_attention(q, k, v)
}

@gpu_graph("tensor_dataflow")
pub fn sparse_and_ragged(
    a Matrix[512, 512, Float16],
    b Matrix[512, 512, Float16],
) Matrix[512, 512, Float16] {
    let r = tg_ragged_matmul(a, b, 50)
    tg_block_sparse_matmul(r, b, 25, 32)
}
"#;
    let prog = tenon_syntax::parse(src).expect("parse");
    let off = compile_universal_graphs(&prog, GpuOptConfig::parse_spec("stage1,-fusion"));
    let on = compile_universal_graphs(&prog, GpuOptConfig::all());

    // MXFP4 interleaved SuperBlock16 auto-repack on B=1 memory-bound decode
    let (msl_mx, plan_mx) = emit_metal::emit_metal_graph_and_plan(&on[0]);
    assert!(msl_mx.contains("mxfp4_e2m1_x4"));
    assert!(plan_mx.contains("PACK_MXFP4_NK_SB16"));

    // FlashAttention hardware AMX FlashAttention-2 vs naive O(S^2)
    let (msl_fa, _) = emit_metal::emit_metal_graph_and_plan(&on[1]);
    assert!(msl_fa.contains("simdgroup_multiply_accumulate"), "expected AMX FlashAttention-2!");
    assert!(off[1].arena.total_arena_bytes > on[1].arena.total_arena_bytes * 10, "expected FlashAttention to eliminate O(S^2) HBM!");

    // Ragged + True BSR-32 Compressed BlockSparse persistent work-stealing
    let (msl_sp, plan_sp) = emit_metal::emit_metal_graph_and_plan(&on[2]);
    assert!(msl_sp.contains("atomic_fetch_add_explicit"));
    assert!(plan_sp.contains("PACK_BSR32_NK"), "expected true BSR-32 compressed weight repack!");
    assert!(on[2].roofline.fused_bytes < off[2].roofline.fused_bytes);
}

#[test]
fn stage4_all_9_general_facilities_toggle_and_emit_metal_cuda_tpu() {
    let cfg_s3 = GpuOptConfig::parse_spec("stage3");
    assert!(cfg_s3.mxfp4_quant && !cfg_s3.tile_swizzle && !cfg_s3.horiz_pack && !cfg_s3.buffer_donation);
    let cfg_all = GpuOptConfig::parse_spec("all,-tile_swizzle,-async_overlap");
    assert!(!cfg_all.tile_swizzle && !cfg_all.async_overlap);
    assert!(cfg_all.horiz_pack && cfg_all.buffer_donation && cfg_all.graph_replay);
    assert!(cfg_all.online_monoid && cfg_all.amp_dyn_quant && cfg_all.dyn_bucket && cfg_all.non_tf_ops);

    let src = r#"
@gpu_graph("tensor_dataflow")
pub fn qkv_packed(
    x  Matrix[512, 2048, Float16],
    wq Matrix[2048, 2048, Float16],
    wk Matrix[2048, 2048, Float16],
    wv Matrix[2048, 2048, Float16],
) Matrix[512, 6144, Float16] {
    tg_qkv_proj(x, wq, wk, wv)
}

@gpu_graph("tensor_dataflow")
pub fn opt_adamw(
    p Matrix[512, 512, Float32],
    g Matrix[512, 512, Float32],
    m Matrix[512, 512, Float32],
    v Matrix[512, 512, Float32],
) Matrix[512, 512, Float32] {
    tg_adamw_step(p, g, m, v)
}

@gpu_graph("tensor_dataflow")
pub fn fused_xent(
    logits  Matrix[128, 4096, Float32],
    targets Matrix[128, 1, Float32],
) Matrix[128, 4096, Float32] {
    tg_cross_entropy(logits, targets)
}

@gpu_graph("tensor_dataflow")
pub fn amp_int8_mm(
    x  Matrix[512, 2048, Float16],
    nw Matrix[1, 2048, Float16],
    w  Matrix[2048, 2048, Float16],
) Matrix[512, 2048, Float16] {
    tg_amp_matmul(x, nw, w)
}

@gpu_graph("tensor_dataflow")
pub fn dyn_mm(
    a Matrix[512, 2048, Float16],
    b Matrix[2048, 2048, Float16],
) Matrix[512, 2048, Float16] {
    tg_dyn_matmul(a, b)
}

@gpu_graph("tensor_dataflow")
pub fn conv3x3_implicit(
    x Matrix[1024, 128, Float16],
    w Matrix[1152, 128, Float16],
) Matrix[1024, 128, Float16] {
    tg_conv2d_3x3(x, w, 32, 32)
}

@gpu_graph("tensor_dataflow")
pub fn ssm_scan(
    a Matrix[256, 512, Float32],
    b Matrix[256, 512, Float32],
) Matrix[256, 512, Float32] {
    tg_scan(a, b)
}
"#;
    let prog = tenon_syntax::parse(src).expect("parse stage4");
    let off = compile_universal_graphs(&prog, GpuOptConfig::stage3());
    let on = compile_universal_graphs(&prog, GpuOptConfig::all());
    assert_eq!(on.len(), 7);

    // #1 tile_swizzle + #2 horiz_pack
    let (msl_qkv_on, plan_qkv_on) = emit_metal::emit_metal_graph_and_plan(&on[0]);
    let (_, plan_qkv_off) = emit_metal::emit_metal_graph_and_plan(&off[0]);
    assert!(plan_qkv_on.contains("CONCAT_NK|"));
    assert!(msl_qkv_on.contains("_grp_sz = 8u"));
    assert!(!plan_qkv_off.contains("CONCAT_NK|"));

    // #3 buffer_donation (AdamW in-place)
    let (_, plan_adam_on) = emit_metal::emit_metal_graph_and_plan(&on[1]);
    let (_, plan_adam_off) = emit_metal::emit_metal_graph_and_plan(&off[1]);
    assert!(plan_adam_on.contains("DONATE|0|0"));
    assert!(on[1].arena.donated_bytes > 0);
    assert_eq!(off[1].arena.donated_bytes, 0);
    assert!(!plan_adam_off.contains("DONATE|0|0"));

    // #4 graph_replay + #5 online_monoid (Cross-Entropy zero probs scratch!)
    let (_, plan_ce_on) = emit_metal::emit_metal_graph_and_plan(&on[2]);
    let (_, plan_ce_off) = emit_metal::emit_metal_graph_and_plan(&off[2]);
    assert!(plan_ce_on.contains("replay=1|overlap=1"));
    assert!(off[2].arena.total_arena_bytes > on[2].arena.total_arena_bytes * 10);
    assert!(!plan_ce_off.contains("DONATE|0|0"));

    // #6 amp_dyn_quant (PACK_INT8_ROW_NK -50% weight DRAM)
    let (msl_amp_on, plan_amp_on) = emit_metal::emit_metal_graph_and_plan(&on[3]);
    assert!(plan_amp_on.contains("PACK_INT8_ROW_NK"));
    assert!(msl_amp_on.contains("w_scale"));

    // #7 dyn_bucket (3-bucket O(1) dispatch)
    let (_, plan_dyn_on) = emit_metal::emit_metal_graph_and_plan(&on[4]);
    let (_, plan_dyn_off) = emit_metal::emit_metal_graph_and_plan(&off[4]);
    assert_eq!(plan_dyn_on.matches("DYN_BUCKET|").count(), 3);
    assert_eq!(plan_dyn_off.matches("DYN_BUCKET|").count(), 0);

    // #8A non_tf_ops Conv2d Implicit GEMM vs explicit im2col
    let (msl_conv_on, _) = emit_metal::emit_metal_graph_and_plan(&on[5]);
    assert!(msl_conv_on.contains("Implicit GEMM"));
    assert!(off[5].arena.total_arena_bytes > on[5].arena.total_arena_bytes * 10);

    // #8B non_tf_ops Parallel Scan vs Sequential
    let (msl_scan_on, _) = emit_metal::emit_metal_graph_and_plan(&on[6]);
    let (msl_scan_off, _) = emit_metal::emit_metal_graph_and_plan(&off[6]);
    assert!(msl_scan_on.contains("simd_shuffle_up"));
    assert!(!msl_scan_off.contains("simd_shuffle_up"));

    // Multi-backend CUDA & TPU StableHLO emission check across Stage-4 ops — ZERO stub kernels!
    for cg in &on {
        let cuda = emit_cuda_tpu::emit_universal_cuda(cg);
        let tpu = emit_cuda_tpu::emit_universal_stablehlo(cg);
        assert!(cuda.contains("__global__"));
        assert!(!cuda.contains("(void)a;") && !cuda.contains("(void)x;") && !cuda.contains("(void)logits;"),
            "found stub CUDA kernel in {}: {}", cg.graph.fn_name, cuda);
        assert!(tpu.contains("func.func"));
    }
}

#[test]
fn stage5_llmc_4_facilities_default_on_and_multi_backend() {
    // Verify `default()` enables ALL 24 general compiler passes out of the box!
    let def = GpuOptConfig::default();
    assert_eq!(def, GpuOptConfig::all());
    assert!(def.auto_pad && def.stochastic_round && def.streaming_cache && def.zero1_spmd);

    let s4 = GpuOptConfig::parse_spec("stage4");
    assert!(!s4.auto_pad && !s4.stochastic_round && !s4.streaming_cache && !s4.zero1_spmd);

    let src = r#"
@gpu_graph("tensor_dataflow")
pub fn odd_gemm(
    a Matrix[512, 1003, Float16],
    b Matrix[1003, 2003, Float16],
) Matrix[512, 2003, Float16] {
    tg_matmul(a, b)
}

@gpu_graph("tensor_dataflow")
pub fn gpt2_vocab_xent(
    logits  Matrix[128, 50257, Float32],
    targets Matrix[128, 1, Float32],
) Matrix[128, 50257, Float32] {
    tg_cross_entropy(logits, targets)
}

@gpu_graph("tensor_dataflow")
pub fn sr_adamw_f16(
    p Matrix[1024, 4096, Float16],
    g Matrix[1024, 4096, Float16],
    m Matrix[1024, 4096, Float16],
    v Matrix[1024, 4096, Float16],
) Matrix[1024, 4096, Float16] {
    tg_adamw_step(p, g, m, v)
}

@gpu_graph("tensor_dataflow")
pub fn stream_add_64mb(
    a Matrix[4096, 4096, Float32],
    b Matrix[4096, 4096, Float32],
) Matrix[4096, 4096, Float32] {
    tg_add(a, b)
}
"#;
    let prog = tenon_syntax::parse(src).expect("parse stage5");
    let off = compile_universal_graphs(&prog, GpuOptConfig::stage4());
    let on = compile_universal_graphs(&prog, GpuOptConfig::default());
    let z1 = compile_universal_graphs(&prog, GpuOptConfig::parse_spec("all,spmd=4"));

    // #10A Odd-Dim MatMul Auto-Pad (`1003->1024`, `2003->2048`)
    let (msl_mm, plan_mm) = emit_metal::emit_metal_graph_and_plan(&on[0]);
    assert!(plan_mm.contains("PAD_ALIGN|a|512|1024|0"));
    assert!(plan_mm.contains("PAD_ALIGN|b|1024|2048|1"));
    assert!(msl_mm.contains("K_pad=1024, N_pad=2048 -> logical N=2003"));
    assert!(off[0].input_pads.is_empty());

    // #10B GPT-2 Odd-Vocab Cross-Entropy Auto-Pad (`50257 -> 50304`)
    let (_, plan_ce_on) = emit_metal::emit_metal_graph_and_plan(&on[1]);
    let (msl_ce_off, _) = emit_metal::emit_metal_graph_and_plan(&off[1]);
    assert!(plan_ce_on.contains("PAD_ALIGN|logits|128|50304|0"));
    assert!(msl_ce_off.contains("50257u"));

    // #11 Philox Stochastic Rounding on FP16 AdamW
    let (msl_sr_on, plan_sr_on) = emit_metal::emit_metal_graph_and_plan(&on[2]);
    let (msl_sr_off, _) = emit_metal::emit_metal_graph_and_plan(&off[2]);
    assert!(msl_sr_on.contains("_philox") && msl_sr_on.contains("_sr_f16"));
    assert!(plan_sr_on.contains("__sr_seed"));
    assert!(!msl_sr_off.contains("_philox"));

    // #12 Liveness-Driven Streaming Cache Hints (`CachePolicy::Streaming` -> 64B Metal burst / CUDA `__ldcs` / TPU `hbm_streaming`)
    assert_eq!(on[3].cache_policies.get("a"), Some(&CachePolicy::Streaming));
    let (msl_st_on, _) = emit_metal::emit_metal_graph_and_plan(&on[3]);
    assert!(msl_st_on.contains("64B full-cacheline streaming burst"));
    let cuda_sr = emit_cuda_tpu::emit_universal_cuda(&on[2]);
    assert!(cuda_sr.contains("__ldcs") && cuda_sr.contains("__stcs") && cuda_sr.contains("stochastic_rounding"));
    let tpu_st = emit_cuda_tpu::emit_universal_stablehlo(&on[3]);
    assert!(tpu_st.contains("hbm_streaming"));

    // #13 ZeRO-1 Optimizer Sharding + Gradient Bucketing (`spmd=4`: ReduceScatter + 1/4 AdamW + AllGather!)
    assert_eq!(z1[2].ad_stats.zero1_shards, 4);
    assert!(z1[2].ad_stats.zero1_sharded_opt_bytes > 0);
    let cuda_z1 = emit_cuda_tpu::emit_universal_cuda(&z1[2]);
    assert!(cuda_z1.contains("ncclReduceScatter") && cuda_z1.contains("ncclAllGather"));
    let tpu_z1 = emit_cuda_tpu::emit_universal_stablehlo(&z1[2]);
    assert!(tpu_z1.contains("stablehlo.reduce_scatter") && tpu_z1.contains("stablehlo.all_gather"));
}

#[test]
fn stage6_hetero_streaming_vram_budget_numa4r_dual_dma_and_hybrid_dispatch() {
    let def = GpuOptConfig::default();
    assert_eq!(def, GpuOptConfig::stage6());
    assert!(def.vram_budget_plan && def.numa_repack_layout && def.dual_stream_dma && def.hetero_dispatch);

    let s5 = GpuOptConfig::parse_spec("stage5");
    assert!(s5.auto_pad && s5.zero1_spmd);
    assert!(!s5.vram_budget_plan && !s5.numa_repack_layout && !s5.dual_stream_dma && !s5.hetero_dispatch);

    let src = r#"
@gpu_graph("stream:vram=24G,ram=64G,layers=80,experts=0,ctx=16384,kv=auto,kv_dim=1024,ddr=70,pcie=24")
pub fn llama70b_layer_24g(
    x       Matrix[1, 8192, Float16],
    wq      Matrix[8192, 8192, Q4_0],
    wk      Matrix[8192, 1024, Q4_0],
    wv      Matrix[8192, 1024, Q4_0],
    wo      Matrix[8192, 8192, Q4_0],
    w_gate  Matrix[8192, 28672, Q4_K],
    w_up    Matrix[8192, 28672, Q4_K],
    w_down  Matrix[28672, 8192, Q4_K],
) Matrix[1, 8192, Float16] {
    let g = tg_matmul(x, w_gate)
    let u = tg_matmul(x, w_up)
    let h = tg_swiglu(g, u)
    tg_matmul(h, w_down)
}

@gpu_graph("stream:vram=24G,ram=32G,layers=46,experts=128,topk=8,ctx=8192,kv=auto,kv_dim=1024")
pub fn glm106b_moe_nvme(
    x        Matrix[1, 4096, Float16],
    wq       Matrix[4096, 4096, Q4_0],
    exp_gate Matrix[4096, 1408, Q4_0],
    exp_up   Matrix[4096, 1408, Q4_0],
    exp_down Matrix[1408, 4096, Q4_0],
) Matrix[1, 4096, Float16] {
    let g = tg_matmul(x, exp_gate)
    let u = tg_matmul(x, exp_up)
    let h = tg_swiglu(g, u)
    tg_matmul(h, exp_down)
}
"#;
    let prog = tenon_syntax::parse(src).expect("parse stage6");
    let off = compile_universal_graphs(&prog, GpuOptConfig::stage5());
    let on = compile_universal_graphs(&prog, GpuOptConfig::default());
    assert_eq!(on.len(), 2);

    // 1. 70B Dense @ 24GB VRAM, 64GB RAM, 16K ctx -> HybridCpuInt16 + Q8_0 KV + NUMA Q4K_4R + 30..45 pinned GPU FFNs!
    let hp_dense = &on[0].hetero_plan;
    assert!(hp_dense.active);
    assert!(hp_dense.kv_quant_q8, "16K ctx on 24G VRAM must auto-enable Q8_0 KV-Cache");
    assert!(hp_dense.kv_bytes_saved > 2_000_000_000, "Q8_0 KV should save >2GB on 70B @ 16K ctx");
    assert!(hp_dense.pinned_gpu_ffn_layers >= 30 && hp_dense.pinned_gpu_ffn_layers <= 48,
        "expected ~35-42 pinned GPU FFNs on 24G 4090, got {}", hp_dense.pinned_gpu_ffn_layers);
    assert_eq!(hp_dense.dispatch_mode, hetero_stream::HeteroDispatchMode::HybridCpuInt16);
    assert!(!hp_dense.use_nocache_nvme);
    assert_eq!(hp_dense.dma_ring_slots, 4);
    assert!(hp_dense.prefill_chunk_tokens >= 32);
    assert!(on[0].input_repacks.iter().any(|(_, l)| *l == LayoutTag::NumaRepackQ4K_4R));
    assert!(hp_dense.projected_prefill_tok_s > hp_dense.projected_decode_tok_s * 10.0);

    // Ablation: Stage 5 (without Stage 6) has naive ring=1, no Q8 KV upgrade, no NUMA 4R repack!
    assert_eq!(off[0].hetero_plan.dma_ring_slots, 1);
    assert_eq!(off[0].hetero_plan.prefill_chunk_tokens, 1);
    assert!(off[0].hetero_plan.numa_repack_weights.is_empty());

    // 2. 106B MoE @ 24GB VRAM, 32GB Host RAM (< 58GB model) -> DirectNvmeStream + Per-Layer LRU + 16-slot DMA + Promoter!
    let hp_moe = &on[1].hetero_plan;
    assert!(hp_moe.use_nocache_nvme, "32G host RAM < 58G MoE must trigger F_NOCACHE direct NVMe");
    assert_eq!(hp_moe.dispatch_mode, hetero_stream::HeteroDispatchMode::DirectNvmeStream);
    assert!(hp_moe.lru_slots_per_layer >= 30 && hp_moe.lru_slots_per_layer <= 90,
        "expected ~40-60 LRU expert slots on 24G VRAM, got {}", hp_moe.lru_slots_per_layer);
    assert_eq!(hp_moe.dma_ring_slots, 16);
    assert!(hp_moe.async_promoter && hp_moe.purgeable_reclaim);

    // 3. Multi-backend Metal Plan + CUDA + TPU StableHLO emission check!
    let (msl_d, plan_d) = emit_metal::emit_metal_graph_and_plan(&on[0]);
    assert!(plan_d.contains("HETERO_PLAN|mode=HYBRID_CPU_INT16|") && plan_d.contains("NUMA_REPACK_Q4K_4R"));
    assert!(msl_d.contains("Stage 6 Heterogeneous Streaming") && msl_d.contains("MTLPurgeableStateEmpty"));
    let cuda_d = emit_cuda_tpu::emit_universal_cuda(&on[0]);
    assert!(cuda_d.contains("cuStreamWaitEvent") && cuda_d.contains("_mm256_madd_epi16") && cuda_d.contains("d_step_args"));
    let tpu_d = emit_cuda_tpu::emit_universal_stablehlo(&on[0]);
    assert!(tpu_d.contains("mhlo.memory_kind=\"pinned_host\""));
}

#[test]
fn stage6_v5_tp_multi_gpu_mla_latent_kv_and_extreme_low_bit_quants() {
    let src = r#"
@gpu_graph("stream:vram=24G,ram=64G,layers=80,experts=0,ctx=16384,kv=auto,tp=2,spec=4")
fn llama70b_tp2_iq4xs(
    x       Matrix[1, 8192, Float16],
    w_qkv   Matrix[8192, 10240, IQ4_XS],
    w_gate  Matrix[8192, 28672, Q3_K],
    w_up    Matrix[8192, 28672, Q2_K],
    w_down  Matrix[28672, 8192, IQ4_XS],
) Matrix[1, 8192, Float16] {
    let qkv = tg_matmul(x, w_qkv)
    let g = tg_silu(tg_matmul(x, w_gate))
    let u = tg_matmul(x, w_up)
    let h = tg_mul(g, u)
    tg_matmul(h, w_down)
}

@gpu_graph("stream:vram=24G,ram=128G,layers=61,experts=256,topk=8,ctx=16384,kv_dim=16384,mla=576,spec=4")
fn deepseek_v3_mla_latent(
    x        Matrix[1, 7168, Float16],
    w_kv_a   Matrix[7168, 576, Q4_K],
    exp_gate Matrix[7168, 2048, Q4_K],
    exp_up   Matrix[7168, 2048, Q4_K],
    exp_down Matrix[2048, 7168, Q4_K],
) Matrix[1, 7168, Float16] {
    let g = tg_silu(tg_matmul(x, exp_gate))
    let u = tg_matmul(x, exp_up)
    let h = tg_mul(g, u)
    tg_matmul(h, exp_down)
}
"#;
    let prog = tenon_syntax::parse(src).expect("parse v5 stage6");
    let on = compile_universal_graphs(&prog, GpuOptConfig::default());
    assert_eq!(on.len(), 2);

    // 1. 2x 24GB GPUs (TP=2 -> 48GB aggregate VRAM!) with IQ4_XS / Q3_K / Q2_K fits 70B 100% ResidentGpu!
    let hp_tp2 = &on[0].hetero_plan;
    assert_eq!(hp_tp2.tp_devices, 2);
    assert_eq!(hp_tp2.speculative_k, 4);
    assert_eq!(hp_tp2.dispatch_mode, hetero_stream::HeteroDispatchMode::ResidentGpu);
    assert_eq!(hp_tp2.pinned_gpu_ffn_layers, 80);

    // 2. DeepSeek-V3 Compressed Latent MLA KV-Cache (`mla=576` vs `2 * 16384` FP16 -> >50x KV VRAM reduction!)
    let hp_mla = &on[1].hetero_plan;
    assert_eq!(hp_mla.mla_latent_dim, 576);
    assert!(hp_mla.kv_bytes_saved > 50_000_000_000, "MLA latent KV should save >50 GB on 61L 16K ctx!");
    let cuda_tp2 = emit_cuda_tpu::emit_universal_cuda(&on[0]);
    assert!(cuda_tp2.contains("tp=2 (P2P AllReduce)") && cuda_tp2.contains("mma.sync m16n8k16"));
    let (_, plan_mla) = emit_metal::emit_metal_graph_and_plan(&on[1]);
    assert!(plan_mla.contains("mla_latent=576") && plan_mla.contains("spec_k=4"));
}

#[test]
fn stage6_v6_paged_cont_batch_marlin_ep_nccl_tree_mtp_mmproj() {
    let def = GpuOptConfig::default();
    assert!(def.paged_cont_batch && def.marlin_w4a16_mma && def.moe_ep_nccl && def.tree_spec_mtp_mm);
    let src = r#"
@gpu_graph("stream:vram=48G,ram=128G,layers=27,experts=64,topk=6,ctx=16384,kv_dim=2048,mla=576,ep=2,cb=8,tree=7,mtp=1,mmproj=256")
fn deepseek_v2_lite_ep2_v6(
    x        Matrix[1, 2048, Float16],
    w_kv_a   Matrix[2048, 576, Q4_K],
    exp_gate Matrix[2048, 1408, Q4_K],
    exp_up   Matrix[2048, 1408, Q4_K],
    exp_down Matrix[1408, 2048, Q5_0],
) Matrix[1, 2048, Float16] {
    let g = tg_silu(tg_matmul(x, exp_gate))
    let u = tg_matmul(x, exp_up)
    let h = tg_mul(g, u)
    tg_matmul(h, exp_down)
}
"#;
    let prog = tenon_syntax::parse(src).expect("parse v6 stage6");
    let on = compile_universal_graphs(&prog, GpuOptConfig::default());
    assert_eq!(on.len(), 1);
    let hp = &on[0].hetero_plan;
    assert_eq!(hp.ep_devices, 2);
    assert!(hp.nccl_collectives && hp.marlin_cp_async && hp.cow_prefix_blocks);
    assert_eq!(hp.cont_batch_slots, 8);
    assert_eq!(hp.tree_spec_width, 7);
    assert_eq!(hp.mtp_nextn_layers, 1);
    assert_eq!(hp.vision_mmproj_tokens, 256);
    assert_eq!(hp.dispatch_mode, hetero_stream::HeteroDispatchMode::ResidentGpu);
    let cuda = emit_cuda_tpu::emit_universal_cuda(&on[0]);
    assert!(cuda.contains("ep=2 (MoE EP + NCCL=true)") && cuda.contains("Marlin cp.async+lop3=true") && cuda.contains("tree_w=7 mtp=1 mmproj=256"));
    let (_, plan) = emit_metal::emit_metal_graph_and_plan(&on[0]);
    assert!(plan.contains("ep=2|nccl=1|marlin=1|cb_slots=8|cow_kv=1") && plan.contains("tree_w=7|mtp=1|mmproj=256"));
}

#[test]
fn stage6_v7_cross_chip_small_batch_2d_tpep_chunked_lora_mrope_fp8_mxfp4() {
    let def = GpuOptConfig::default();
    assert!(def.small_batch_shared_x && def.hybrid_2d_tp_ep && def.chunked_prefill_cb && def.multi_lora_zero_copy && def.anyres_mrope_fp8_mxfp4);
    let src = r#"
@gpu_graph("stream:vram=48G,ram=128G,layers=46,experts=128,topk=8,ctx=16384,kv_dim=1024,tp=2,ep=2,cb=8,chunk_pf=128,lora=16,mmproj=256,anyres=4,mrope=1,fp8_mxfp4=1")
fn glm45_2d_hybrid_v7(
    x        Matrix[1, 4096, Float16],
    exp_gate Matrix[4096, 1408, MXFP4],
    exp_up   Matrix[4096, 1408, MXFP4],
    exp_down Matrix[1408, 4096, FP8],
) Matrix[1, 4096, Float16] {
    let g = tg_silu(tg_matmul(x, exp_gate))
    let u = tg_matmul(x, exp_up)
    let h = tg_mul(g, u)
    tg_matmul(h, exp_down)
}
"#;
    let prog = tenon_syntax::parse(src).expect("parse v7 stage6");
    let on = compile_universal_graphs(&prog, GpuOptConfig::default());
    assert_eq!(on.len(), 1);
    let hp = &on[0].hetero_plan;
    assert!(hp.small_batch_shared_x && hp.hybrid_2d_tp_ep && hp.vision_mrope_2d && hp.fp8_mxfp4_native);
    assert_eq!(hp.chunked_prefill_size, 128);
    assert_eq!(hp.lora_max_rank, 16);
    assert_eq!(hp.vision_anyres_tiles, 4);
    let cuda = emit_cuda_tpu::emit_universal_cuda(&on[0]);
    assert!(cuda.contains("hybrid_2d=true") && cuda.contains("chunk_pf=128") && cuda.contains("sb_shx=true") && cuda.contains("lora_r=16") && cuda.contains("fp8_mxfp4=true"));
    let tpu = emit_cuda_tpu::emit_universal_stablehlo(&on[0]);
    assert!(tpu.contains("hybrid_2d=true") && tpu.contains("sb_shx=true") && tpu.contains("lora_r=16"));
    let (_, plan) = emit_metal::emit_metal_graph_and_plan(&on[0]);
    assert!(plan.contains("sb_shx=1|hybrid_2d=1|chunk_pf=128|lora_r=16|anyres=4|mrope=1|fp8_mxfp4=1"));
}

#[test]
fn test_stage6_v8_tc_amx_q4k_marlin_universal_cb_moe_grouped_audio() {
    let src = r#"
@gpu_graph("stream:vram=48G,ram=128G,layers=27,experts=64,topk=6,ctx=8192,tp=2,ep=2,mla=576,cb=8,mmproj=64,audio=128,mrope=1,fp8_mxfp4=1")
fn dsv2_v8_multimodal_block(x Matrix[8, 2048, Float16], w_q Matrix[2048, 2048, Q4_K], w_down Matrix[2048, 1408, Q6_K]) Matrix[8, 1408, Float16] {
    let h = tg_matmul(x, w_q)
    tg_matmul(h, w_down)
}
"#;
    let prog = tenon_syntax::parse(src).expect("parse v8 graph");
    let on = compile_universal_graphs(&prog, GpuOptConfig::default());
    assert_eq!(on.len(), 1);
    let hp = &on[0].hetero_plan;
    assert!(hp.fp8_mxfp4_tc_amx && hp.q4k_q6k_marlin && hp.paged_cb_tp_mla_spec && hp.moe_grouped_gemm && hp.vision_prefix_cache);
    assert_eq!(hp.audio_mel_tokens, 128);
    let cuda = emit_cuda_tpu::emit_universal_cuda(&on[0]);
    assert!(cuda.contains("fp8_tc=true") && cuda.contains("q4k_marlin=true") && cuda.contains("paged_univ=true") && cuda.contains("moe_grp=true") && cuda.contains("audio_mel=128") && cuda.contains("vis_cache=true"));
    let tpu = emit_cuda_tpu::emit_universal_stablehlo(&on[0]);
    assert!(tpu.contains("fp8_tc=true") && tpu.contains("q4k_marlin=true") && tpu.contains("moe_grp=true") && tpu.contains("audio_mel=128"));
    let (_, plan) = emit_metal::emit_metal_graph_and_plan(&on[0]);
    assert!(plan.contains("fp8_tc=1|q4k_marlin=1|paged_univ=1|moe_grp=1|audio_mel=128|vis_cache=1"));
}

#[test]
fn test_stage6_v9_moe_mega_fp8_k32_hetero_lora_mm_kv_cow_pd_kv() {
    let src = r#"
@gpu_graph("stream:vram=48G,ram=128G,layers=27,experts=64,topk=6,ctx=8192,tp=2,ep=2,mla=576,cb=8,lora=16,mmproj=64,audio=128,mrope=1,fp8_mxfp4=1")
fn dsv2_v9_full_stack(x Matrix[8, 2048, Float16], w_q Matrix[2048, 2048, FP8], w_down Matrix[2048, 1408, Q6_K]) Matrix[8, 1408, Float16] {
    let h = tg_matmul(x, w_q)
    tg_matmul(h, w_down)
}
"#;
    let prog = tenon_syntax::parse(src).expect("parse v9 graph");
    let on = compile_universal_graphs(&prog, GpuOptConfig::default());
    assert_eq!(on.len(), 1);
    let hp = &on[0].hetero_plan;
    assert!(hp.moe_gpu_persistent_mega && hp.fp8_e4m3_m16n8k32_amx_f16 && hp.hetero_lora_bgmv && hp.multimodal_kv_cow_cache && hp.disaggregated_pd_kv);
    let cuda = emit_cuda_tpu::emit_universal_cuda(&on[0]);
    assert!(cuda.contains("moe_mega=true") && cuda.contains("fp8_k32=true") && cuda.contains("hetero_lora=true") && cuda.contains("mm_kv_cow=true") && cuda.contains("pd_kv=true"));
    let tpu = emit_cuda_tpu::emit_universal_stablehlo(&on[0]);
    assert!(tpu.contains("moe_mega=true") && tpu.contains("fp8_k32=true") && tpu.contains("hetero_lora=true") && tpu.contains("mm_kv_cow=true") && tpu.contains("pd_kv=true"));
    let (_, plan) = emit_metal::emit_metal_graph_and_plan(&on[0]);
    assert!(plan.contains("moe_mega=1|fp8_k32=1|hetero_lora=1|mm_kv_cow=1|pd_kv=1"));
}

#[test]
fn test_stage6_v10_moe_mega_v2_dev_graph_pd_zc_kv_q4_swa_router() {
    let def = GpuOptConfig::default();
    assert!(
        def.moe_mega_tc_amx_ragged
            && def.device_graph_capture
            && def.pd_zerocopy_binary_shm_ipc
            && def.paged_kv_fp8_q4_swa_longctx
            && def.pd_router_prefix_affinity_telemetry
    );
    let src = r#"
@gpu_graph("stream:vram=48G,ram=128G,layers=27,experts=64,topk=6,ctx=131072,kv=q4,swa=4096,tp=2,ep=2,mla=576,cb=16,lora=16,mmproj=64,audio=128,pd_router=1")
fn dsv2_v10_full_stack(x Matrix[16, 2048, Float16], w_q Matrix[2048, 2048, FP8], w_down Matrix[2048, 1408, Q6_K]) Matrix[16, 1408, Float16] {
    let h = tg_matmul(x, w_q)
    tg_matmul(h, w_down)
}
"#;
    let prog = tenon_syntax::parse(src).expect("parse v10 graph");
    let on = compile_universal_graphs(&prog, GpuOptConfig::default());
    assert_eq!(on.len(), 1);
    let hp = &on[0].hetero_plan;
    assert!(hp.moe_mega_tc_amx_ragged && hp.device_graph_capture && hp.pd_zerocopy_binary_shm_ipc && hp.pd_router_affinity);
    assert_eq!(hp.kv_quant_mode, 3);
    assert_eq!(hp.swa_window, 4096);
    let cuda = emit_cuda_tpu::emit_universal_cuda(&on[0]);
    assert!(cuda.contains("moe_mega_v2=true") && cuda.contains("dev_graph=true") && cuda.contains("pd_zc=true") && cuda.contains("kv_mode=3") && cuda.contains("swa=4096") && cuda.contains("pd_router=true"));
    let tpu = emit_cuda_tpu::emit_universal_stablehlo(&on[0]);
    assert!(tpu.contains("moe_mega_v2=true") && tpu.contains("dev_graph=true") && tpu.contains("pd_zc=true") && tpu.contains("kv_mode=3") && tpu.contains("swa=4096") && tpu.contains("pd_router=true"));
    let (_, plan) = emit_metal::emit_metal_graph_and_plan(&on[0]);
    assert!(plan.contains("moe_mega_v2=1|dev_graph=1|pd_zc=1|kv_mode=3|swa=4096|pd_router=1"));
}

#[test]
fn test_stage6_v11_tp_graph_mega_v3_pd_layer_flash_decode_ha_hotswap() {
    let def = GpuOptConfig::default();
    assert!(
        def.multi_gpu_graph_and_async_icb
            && def.moe_mega_v3_fp8_tc_and_reg_amx
            && def.pd_multigpu_ipc_and_layer_stream
            && def.flash_decoding_split_kv_lse
            && def.production_ha_cancel_and_lora_hotswap
    );
    let src = r#"
@gpu_graph("stream:vram=48G,ram=128G,layers=27,experts=64,topk=6,ctx=131072,kv=fp8,swa=4096,tp=2,ep=2,mla=576,cb=16,lora=16,mmproj=64,audio=128,pd_router=1")
fn dsv2_v11_full_stack(x Matrix[16, 2048, Float16], w_q Matrix[2048, 2048, FP8], w_down Matrix[2048, 1408, Q6_K]) Matrix[16, 1408, Float16] {
    let h = tg_matmul(x, w_q)
    tg_matmul(h, w_down)
}
"#;
    let prog = tenon_syntax::parse(src).expect("parse v11 graph");
    let on = compile_universal_graphs(&prog, GpuOptConfig::default());
    assert_eq!(on.len(), 1);
    let hp = &on[0].hetero_plan;
    assert!(hp.multi_gpu_graph_async_icb && hp.moe_mega_v3_fp8_amx && hp.pd_multigpu_ipc_layer_stream && hp.router_ha_cancel_hotswap);
    assert_eq!(hp.flash_decode_splits, 16);
    let cuda = emit_cuda_tpu::emit_universal_cuda(&on[0]);
    assert!(cuda.contains("tp_graph=true") && cuda.contains("moe_mega_v3=true") && cuda.contains("pd_layer=true") && cuda.contains("fd_splits=16") && cuda.contains("ha_hotswap=true"));
    let tpu = emit_cuda_tpu::emit_universal_stablehlo(&on[0]);
    assert!(tpu.contains("tp_graph=true") && tpu.contains("moe_mega_v3=true") && tpu.contains("pd_layer=true") && tpu.contains("fd_splits=16") && tpu.contains("ha_hotswap=true"));
    let (_, plan) = emit_metal::emit_metal_graph_and_plan(&on[0]);
    assert!(plan.contains("tp_graph=1|moe_mega_v3=1|pd_layer=1|fd_splits=16|ha_hotswap=1"));
}
