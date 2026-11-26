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

    // MXFP4 auto-repack on B=1 memory-bound decode
    let (msl_mx, plan_mx) = emit_metal::emit_metal_graph_and_plan(&on[0]);
    assert!(msl_mx.contains("mxfp4_e2m1"));
    assert!(plan_mx.contains("PACK_MXFP4_NK"));

    // FlashAttention online tiled vs naive O(S^2)
    assert!(off[1].arena.total_arena_bytes > on[1].arena.total_arena_bytes * 10, "expected FlashAttention to eliminate O(S^2) HBM!");

    // Ragged + BlockSparse persistent work-stealing
    let (msl_sp, _) = emit_metal::emit_metal_graph_and_plan(&on[2]);
    assert!(msl_sp.contains("atomic_fetch_add_explicit"));
    assert!(on[2].roofline.fused_bytes < off[2].roofline.fused_bytes);
}
