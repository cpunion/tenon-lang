//! Universal Apple Silicon Metal MSL & Executable Host Plan Emitter (`emit_metal.rs`).
//!
//! Translates ANY lowered `CompiledUniversalGraph` into:
//!   1. Directly compilable, zero-warning Apple Metal 3.x MSL kernels:
//!      - Baseline scalar loop (`tile_mma = false`) for exact ablation reference
//!      - Hardware AMX `simdgroup_half8x8` Tiled GEMM (`BM=32/64, BN=32/64, BK=32`) with:
//!        * All 4 transpose modes (`NN`, `NT`, `TN`, `TT`) for Inference + Reverse-Mode AD!
//!        * Coalesced `half4` / `uint4` loads (`layout_prop`) + bank-conflict-free SRAM padding
//!        * Producer/Consumer ping-pong double buffering (`warp_spec`)
//!        * Split-K / Stream-K parallel reduction (`stream_k`)
//!        * Arbitrary in-register Epilogue fusion (`+ bias/residual`, `silu`, `gelu`, `relu`, `scale`, `SwiGLU`)
//!        * Fused Prologue RMSNorm / LayerNorm in threadgroup SRAM (`general_fusion`)
//!        * PGO Auto-Tune candidate variants (`_cfg0`, `_cfg1`, ...) (`pgo_autotune`)
//!      - Memory-bound 128-bit Coalesced SIMD-Group GEMV (`M <= 8`) + OCP MXFP4 (E2M1 + E8M0) register unpack (`mxfp4_quant`)
//!      - Single-pass Tiled Online-Softmax FlashAttention vs Naive $O(S^2)$ Attention (`general_fusion`)
//!      - Compact Ragged-Batch AMX GEMM & Persistent Work-Stealing Block-Sparse AMX GEMM (`ragged_sparse`)
//!   2. A compact, op-agnostic **Execution Plan** consumed directly by `metal_shim.m`'s generic
//!      Graph Executor (`tn_metal_exec_plan`), containing zero hand-coded kernel names or shapes!

use super::fusion::*;
use super::*;

/// Emits BOTH the MSL shader source and the machine-readable Execution Plan lines for `cg`.
pub fn emit_metal_graph_and_plan(cg: &CompiledUniversalGraph) -> (String, String) {
    let mut msl = String::new();
    let mut plan = String::new();

    let r = &cg.roofline;
    msl.push_str(&format!(
        "// =========================================================================\n\
         // Tenon Universal Tile-IR Metal Emitter (`@gpu_graph {}` | opts={})\n\
         // FLOPs: {:.3}G -> {:.3}G (E-Graph saved {:.1}%) | Bytes: {:.2}MB -> {:.2}MB (Fusion saved {:.1}%)\n\
         // Arena: {}B vs naive {}B (VRAM saved {:.1}%) | Silicon T_min(M4_Max)={:.4}ms\n\
         // =========================================================================\n",
        cg.graph.fn_name,
        cg.config.summary_tag(),
        r.raw_flops / 1e9,
        r.opt_flops / 1e9,
        r.flops_saved_pct,
        r.naive_bytes / 1e6,
        r.fused_bytes / 1e6,
        r.bytes_saved_pct,
        cg.arena.total_arena_bytes,
        cg.arena.naive_total_bytes,
        cg.arena.vram_saved_pct,
        r.m4_t_min_ms,
    ));

    // Plan header line for this graph
    plan.push_str(&format!(
        "GRAPH|{}|opts={}|arena={}|naive_arena={}|donated={}|zero1_saved={}|single_cb={}|pgo={}|replay={}|overlap={}|flops={:.0}|bytes={:.0}|t_min_ms={:.6}\n",
        cg.graph.fn_name,
        cg.config.summary_tag(),
        cg.arena.total_arena_bytes,
        cg.arena.naive_total_bytes,
        cg.arena.donated_bytes,
        cg.ad_stats.zero1_sharded_opt_bytes,
        if cg.config.arena_async { 1 } else { 0 },
        if cg.config.pgo_autotune { 1 } else { 0 },
        if cg.config.graph_replay { 1 } else { 0 },
        if cg.config.async_overlap { 1 } else { 0 },
        r.opt_flops,
        r.fused_bytes,
        r.m4_t_min_ms,
    ));

    // Emit input descriptors (including automatic load-time repack directives!)
    for (idx, (pname, pdesc)) in cg.graph.params.iter().enumerate() {
        let repack = cg
            .input_repacks
            .iter()
            .find(|(k, _)| k == pname)
            .map(|(_, l)| l.plan_tag())
            .unwrap_or("NONE");
        plan.push_str(&format!(
            "INPUT|{}|{}|{}|{}|{}|{}|{}\n",
            idx,
            pname,
            pdesc.rows(),
            pdesc.cols(),
            pdesc.dtype.msl_ty(),
            pdesc.alloc_bytes(),
            repack,
        ));
    }

    // Facility #10 (`auto_pad`): Emit automatic logical-vs-padded alignment directives (`50257->50304`, `1003->1024`)!
    for pad in &cg.input_pads {
        plan.push_str(&format!(
            "PAD_ALIGN|{}|{}|{}|{}\n",
            pad.name,
            pad.pad_rows,
            pad.pad_cols,
            if pad.is_nk { 1 } else { 0 },
        ));
    }

    // Emit output descriptors
    for (idx, oname) in cg.graph.outputs.iter().enumerate() {
        let odesc = &cg.graph.tensors[oname];
        plan.push_str(&format!(
            "OUTPUT|{}|{}|{}|{}|{}|{}\n",
            idx,
            oname,
            odesc.rows(),
            odesc.cols(),
            odesc.dtype.msl_ty(),
            odesc.alloc_bytes(),
        ));
    }

    // Emit static Arena intermediate bindings
    for (tname, (off, sz)) in &cg.arena.allocations {
        plan.push_str(&format!("ARENA|{}|{}|{}\n", tname, off, sz));
    }

    // Stage 6 Heterogeneous Out-of-Core Streaming Plan Directive (Facilities #14-#17)
    if cg.hetero_plan.active {
        let hp = &cg.hetero_plan;
        plan.push_str(&format!(
            "HETERO_PLAN|mode={}|vram_cap={}|core_vram={}|pinned_ffn={}|lru_slots={}|cpu_offload={}|kv_q8={}|kv_saved={}|nocache={}|prefill_b={}|dma_ring={}|purgeable={}|promoter={}|tp={}|ep={}|nccl={}|marlin={}|cb_slots={}|cow_kv={}|mla_latent={}|paged_blk={}|radix_prefix={}|spec_k={}|tree_w={}|mtp={}|mmproj={}|sb_shx={}|hybrid_2d={}|chunk_pf={}|lora_r={}|anyres={}|mrope={}|fp8_mxfp4={}|fp8_tc={}|q4k_marlin={}|paged_univ={}|moe_grp={}|audio_mel={}|vis_cache={}|moe_mega={}|fp8_k32={}|hetero_lora={}|mm_kv_cow={}|pd_kv={}|moe_mega_v2={}|dev_graph={}|pd_zc={}|kv_mode={}|swa={}|pd_router={}|tp_graph={}|moe_mega_v3={}|pd_layer={}|fd_splits={}|ha_hotswap={}|dec_tok_s={:.2}|pref_tok_s={:.2}\n",
            hp.dispatch_mode.tag(),
            hp.vram_cap_bytes,
            hp.core_vram_bytes,
            hp.pinned_gpu_ffn_layers,
            hp.lru_slots_per_layer,
            hp.cpu_offload_layers,
            if hp.kv_quant_q8 { 1 } else { 0 },
            hp.kv_bytes_saved,
            if hp.use_nocache_nvme { 1 } else { 0 },
            hp.prefill_chunk_tokens,
            hp.dma_ring_slots,
            if hp.purgeable_reclaim { 1 } else { 0 },
            if hp.async_promoter { 1 } else { 0 },
            hp.tp_devices,
            hp.ep_devices,
            if hp.nccl_collectives { 1 } else { 0 },
            if hp.marlin_cp_async { 1 } else { 0 },
            hp.cont_batch_slots,
            if hp.cow_prefix_blocks { 1 } else { 0 },
            hp.mla_latent_dim,
            hp.paged_kv_block_size,
            if hp.radix_prefix_cache { 1 } else { 0 },
            hp.speculative_k,
            hp.tree_spec_width,
            hp.mtp_nextn_layers,
            hp.vision_mmproj_tokens,
            if hp.small_batch_shared_x { 1 } else { 0 },
            if hp.hybrid_2d_tp_ep { 1 } else { 0 },
            hp.chunked_prefill_size,
            hp.lora_max_rank,
            hp.vision_anyres_tiles,
            if hp.vision_mrope_2d { 1 } else { 0 },
            if hp.fp8_mxfp4_native { 1 } else { 0 },
            if hp.fp8_mxfp4_tc_amx { 1 } else { 0 },
            if hp.q4k_q6k_marlin { 1 } else { 0 },
            if hp.paged_cb_tp_mla_spec { 1 } else { 0 },
            if hp.moe_grouped_gemm { 1 } else { 0 },
            hp.audio_mel_tokens,
            if hp.vision_prefix_cache { 1 } else { 0 },
            if hp.moe_gpu_persistent_mega { 1 } else { 0 },
            if hp.fp8_e4m3_m16n8k32_amx_f16 { 1 } else { 0 },
            if hp.hetero_lora_bgmv { 1 } else { 0 },
            if hp.multimodal_kv_cow_cache { 1 } else { 0 },
            if hp.disaggregated_pd_kv { 1 } else { 0 },
            if hp.moe_mega_tc_amx_ragged { 1 } else { 0 },
            if hp.device_graph_capture { 1 } else { 0 },
            if hp.pd_zerocopy_binary_shm_ipc { 1 } else { 0 },
            hp.kv_quant_mode,
            hp.swa_window,
            if hp.pd_router_affinity { 1 } else { 0 },
            if hp.multi_gpu_graph_async_icb { 1 } else { 0 },
            if hp.moe_mega_v3_fp8_amx { 1 } else { 0 },
            if hp.pd_multigpu_ipc_layer_stream { 1 } else { 0 },
            hp.flash_decode_splits,
            if hp.router_ha_cancel_hotswap { 1 } else { 0 },
            hp.projected_decode_tok_s,
            hp.projected_prefill_tok_s,
        ));
        msl.push_str(&format!(
            "// [Stage 6 Heterogeneous Streaming] mode={} | tp={} ep={} hybrid_2d={} cb_slots={} cow_kv={} chunk_pf={} mla_latent={} | pinned_ffn={} lru_slots={} | kv_q8={} (saved {:.1}MB) | dma_ring={} (MTLEvent + MTLPurgeableStateEmpty={}) | prefill_chunk={} sb_shx={} spec_k={} tree_w={} mtp={} mmproj={} anyres={} mrope={} lora_r={} fp8_mxfp4={} fp8_tc={} q4k_marlin={} paged_univ={} moe_grp={} audio_mel={} vis_cache={} moe_mega={} fp8_k32={} hetero_lora={} mm_kv_cow={} pd_kv={} moe_mega_v2={} dev_graph={} pd_zc={} kv_mode={} swa={} pd_router={} tp_graph={} moe_mega_v3={} pd_layer={} fd_splits={} ha_hotswap={}\n",
            hp.dispatch_mode.tag(),
            hp.tp_devices,
            hp.ep_devices,
            hp.hybrid_2d_tp_ep,
            hp.cont_batch_slots,
            hp.cow_prefix_blocks,
            hp.chunked_prefill_size,
            hp.mla_latent_dim,
            hp.pinned_gpu_ffn_layers,
            hp.lru_slots_per_layer,
            hp.kv_quant_q8,
            hp.kv_bytes_saved as f64 / 1e6,
            hp.dma_ring_slots,
            hp.purgeable_reclaim,
            hp.prefill_chunk_tokens,
            hp.small_batch_shared_x,
            hp.speculative_k,
            hp.tree_spec_width,
            hp.mtp_nextn_layers,
            hp.vision_mmproj_tokens,
            hp.vision_anyres_tiles,
            hp.vision_mrope_2d,
            hp.lora_max_rank,
            hp.fp8_mxfp4_native,
            hp.fp8_mxfp4_tc_amx,
            hp.q4k_q6k_marlin,
            hp.paged_cb_tp_mla_spec,
            hp.moe_grouped_gemm,
            hp.audio_mel_tokens,
            hp.vision_prefix_cache,
            hp.moe_gpu_persistent_mega,
            hp.fp8_e4m3_m16n8k32_amx_f16,
            hp.hetero_lora_bgmv,
            hp.multimodal_kv_cow_cache,
            hp.disaggregated_pd_kv,
            hp.moe_mega_tc_amx_ragged,
            hp.device_graph_capture,
            hp.pd_zerocopy_binary_shm_ipc,
            hp.kv_quant_mode,
            hp.swa_window,
            hp.pd_router_affinity,
            hp.multi_gpu_graph_async_icb,
            hp.moe_mega_v3_fp8_amx,
            hp.pd_multigpu_ipc_layer_stream,
            hp.flash_decode_splits,
            hp.router_ha_cancel_hotswap,
        ));
    }

    // Emit each TileStage -> MSL kernel(s) + Plan dispatch directive(s)
    for st in &cg.stages {
        match st {
            TileStage::Contraction {
                stage_id,
                name,
                a,
                b,
                trans_a,
                trans_b,
                m,
                k,
                n,
                padded_k,
                padded_n,
                act_dtype,
                weight_dtype,
                weight_layout,
                prologue,
                dual_b,
                dual_swiglu,
                extra_siblings,
                epilogue,
                out,
                schedule,
                pgo_candidates,
                split_k_scratch,
            } => {
                // If fused Prologue Norm is present for compute-bound M >= 16, emit fused norm-into-temp or inline in GEMV!
                // For M <= 8, prologue RMSNorm is fused IN-KERNEL inside the GEMV threadgroup!
                // For M >= 16, emit a vectorized single-pass SRAM RMSNorm stage before AMX MMA (or inline if M == 1).
                let mut eff_a = a.clone();
                if let Some(prol) = prologue {
                    if *m > 8 || !schedule.use_mma {
                        let norm_tmp = format!("__prol_norm_{}", stage_id);
                        let norm_kname = format!("{}_prol_norm", name);
                        emit_msl_row_reduce(
                            &mut msl,
                            &norm_kname,
                            prol.kind,
                            *m,
                            *k,
                            *act_dtype,
                            &[],
                            schedule.use_mma,
                        );
                        let tgx = *m;
                        let tx = if schedule.use_mma { 128 } else { 64 };
                        // Allocate small scratch for prologue output in plan
                        let nb = ((*m * *k * act_dtype.bytes_f64() as usize) + 255) & !255;
                        plan.push_str(&format!("SCRATCH|{}|{}\n", norm_tmp, nb));
                        plan.push_str(&format!(
                            "DISPATCH|{}|tg={},{},1|th={},{},1|smem=0|args={},{},{}\n",
                            norm_kname, tgx, 1, tx, 1, prol.x, prol.weight, norm_tmp
                        ));
                        eff_a = norm_tmp;
                    }
                }

                // Emit primary contraction (+ PGO candidates!)
                let mut cand_specs = Vec::new();
                let cands = if pgo_candidates.is_empty() { vec![*schedule] } else { pgo_candidates.clone() };

                for (ci, cand) in cands.iter().enumerate() {
                    let kname = if ci == 0 {
                        name.clone()
                    } else {
                        format!("{}_cfg{}", name, ci)
                    };

                    if !cand.use_mma {
                        // Baseline naive scalar loop (`tile_mma = false`)
                        emit_msl_contraction_scalar(
                            &mut msl,
                            &kname,
                            *m,
                            *k,
                            *n,
                            *trans_a,
                            *trans_b,
                            *act_dtype,
                            *weight_dtype,
                            *weight_layout,
                            dual_b.is_some(),
                            *dual_swiglu,
                            epilogue,
                        );
                        let (tgx, tgy, tx, ty) = ((*n + 15) / 16, (*m + 15) / 16, 16, 16);
                        let args = build_contraction_args(&eff_a, b, dual_b.as_deref(), epilogue, out, None);
                        cand_specs.push((kname, tgx, tgy, 1usize, tx, ty, 1usize, 0usize, args, None));
                    } else if *m <= 8 {
                        // Memory-bound 128-bit Coalesced SIMD-Group GEMV (+ inline Prologue RMSNorm + MXFP4 + Epilogue!)
                        let inline_prol = prologue.as_ref();
                        emit_msl_gemv_vectorized(
                            &mut msl,
                            &kname,
                            *m,
                            *k,
                            *n,
                            *act_dtype,
                            *weight_dtype,
                            *weight_layout,
                            inline_prol,
                            dual_b.is_some(),
                            *dual_swiglu,
                            epilogue,
                            *cand,
                        );
                        let rows_per_tg = cand.num_sgs * cand.rows_per_sg.max(1);
                        let tgx = (*n + rows_per_tg - 1) / rows_per_tg;
                        let tgy = *m;
                        let tgz = cand.split_k;
                        let tx = cand.num_sgs * 32;
                        let dst_buf = if cand.split_k > 1 {
                            split_k_scratch.clone().unwrap_or_else(|| out.clone())
                        } else {
                            out.clone()
                        };
                        let args = build_contraction_args_gemv(
                            &eff_a,
                            b,
                            inline_prol.map(|p| p.weight.as_str()),
                            dual_b.as_deref(),
                            if cand.split_k > 1 { &[] } else { epilogue.as_slice() },
                            &dst_buf,
                        );
                        let red_spec = if cand.split_k > 1 {
                            let rkname = format!("{}_sk_red", kname);
                            emit_msl_splitk_reduce(&mut msl, &rkname, *m, *n, cand.split_k, *act_dtype, epilogue);
                            let mut rargs = vec![dst_buf.clone()];
                            push_epilogue_arg_names(epilogue, &mut rargs);
                            rargs.push(out.clone());
                            Some((rkname, (*m * *n + 255) / 256, 1usize, 256usize, rargs))
                        } else {
                            None
                        };
                        cand_specs.push((kname, tgx, tgy, tgz, tx, 1, 1, 0, args, red_spec));
                    } else if *padded_k > *k || *padded_n > *n {
                        // Facility #10 (`auto_pad`): Zero-branch 128-bit `uint4` double-buffered AMX over aligned `[m, padded_k] @ [padded_n, padded_k]^T`!
                        emit_msl_amx_gemm_padded(
                            &mut msl,
                            &kname,
                            *m,
                            *padded_k,
                            *padded_n,
                            *n,
                            *act_dtype,
                            *cand,
                        );
                        let tgx = (*padded_n + cand.bn - 1) / cand.bn;
                        let tgy = (*m + cand.bm - 1) / cand.bm;
                        let tx = cand.num_sgs * 32;
                        let args = vec![eff_a.clone(), b.clone(), out.clone()];
                        cand_specs.push((kname, tgx, tgy, 1usize, tx, 1, 1, 0, args, None));
                    } else {
                        // Compute-bound Hardware AMX `simdgroup_half8x8` Tiled MMA!
                        emit_msl_amx_gemm(
                            &mut msl,
                            &kname,
                            *m,
                            *k,
                            *n,
                            *trans_a,
                            *trans_b,
                            *act_dtype,
                            *weight_layout,
                            dual_b.is_some(),
                            *dual_swiglu,
                            if cand.split_k > 1 { &[] } else { epilogue.as_slice() },
                            *cand,
                        );
                        let tgx = (*n + cand.bn - 1) / cand.bn;
                        let tgy = (*m + cand.bm - 1) / cand.bm;
                        let tgz = cand.split_k;
                        let tx = cand.num_sgs * 32;
                        let dst_buf = if cand.split_k > 1 {
                            split_k_scratch.clone().unwrap_or_else(|| out.clone())
                        } else {
                            out.clone()
                        };
                        let args = build_contraction_args(
                            &eff_a,
                            b,
                            dual_b.as_deref(),
                            if cand.split_k > 1 { &[] } else { epilogue.as_slice() },
                            &dst_buf,
                            None,
                        );
                        let red_spec = if cand.split_k > 1 {
                            let rkname = format!("{}_sk_red", kname);
                            emit_msl_splitk_reduce(&mut msl, &rkname, *m, *n, cand.split_k, *act_dtype, epilogue);
                            let mut rargs = vec![dst_buf.clone()];
                            push_epilogue_arg_names(epilogue, &mut rargs);
                            rargs.push(out.clone());
                            Some((rkname, (*m * *n + 255) / 256, 1usize, 256usize, rargs))
                        } else {
                            None
                        };
                        cand_specs.push((kname, tgx, tgy, tgz, tx, 1, 1, 0, args, red_spec));
                    }
                }

                // Emit primary dispatch + optional PGO_GROUP if multiple candidates exist!
                if cand_specs.len() > 1 && cg.config.pgo_autotune {
                    // Only include candidates without split_k mismatch in PGO group (all valid!)
                    plan.push_str(&format!("PGO_BEGIN|{}|{}\n", name, cand_specs.len()));
                    for (kn, tgx, tgy, tgz, tx, ty, tz, smem, args, red) in &cand_specs {
                        plan.push_str(&format!(
                            "PGO_CAND|{}|tg={},{},{}|th={},{},{}|smem={}|args={}\n",
                            kn, tgx, tgy, tgz, tx, ty, tz, smem, args.join(",")
                        ));
                        if let Some((rk, rtgx, rtgy, rtx, rargs)) = red {
                            plan.push_str(&format!(
                                "PGO_POST|{}|tg={},{},1|th={},1,1|smem=0|args={}\n",
                                rk, rtgx, rtgy, rtx, rargs.join(",")
                            ));
                        }
                    }
                    plan.push_str("PGO_END\n");
                } else {
                    let (kn, tgx, tgy, tgz, tx, ty, tz, smem, args, red) = &cand_specs[0];
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},{},{}|th={},{},{}|smem={}|args={}\n",
                        kn, tgx, tgy, tgz, tx, ty, tz, smem, args.join(",")
                    ));
                    if let Some((rk, rtgx, rtgy, rtx, rargs)) = red {
                        plan.push_str(&format!(
                            "DISPATCH|{}|tg={},{},1|th={},1,1|smem=0|args={}\n",
                            rk, rtgx, rtgy, rtx, rargs.join(",")
                        ));
                    }
                }

                // Emit any extra horizontal sibling outputs (e.g. K, V from 3-way QKV fusion) sharing `eff_a`!
                for (si, (wsib, osib, nsib)) in extra_siblings.iter().enumerate() {
                    let skname = format!("{}_sib{}", name, si);
                    let mut sib_sched = *schedule;
                    sib_sched.split_k = 1;
                    sib_sched.rows_per_sg = 1;
                    if !sib_sched.use_mma {
                        emit_msl_contraction_scalar(
                            &mut msl, &skname, *m, *k, *nsib, false, false,
                            *act_dtype, *weight_dtype, *weight_layout, false, false, &[],
                        );
                        plan.push_str(&format!(
                            "DISPATCH|{}|tg={},{},1|th=16,16,1|smem=0|args={},{},{}\n",
                            skname, (*nsib + 15) / 16, (*m + 15) / 16, eff_a, wsib, osib
                        ));
                    } else if *m <= 8 {
                        emit_msl_gemv_vectorized(
                            &mut msl, &skname, *m, *k, *nsib, *act_dtype, *weight_dtype,
                            *weight_layout, prologue.as_ref(), false, false, &[], sib_sched,
                        );
                        let args = build_contraction_args_gemv(
                            &eff_a, wsib, prologue.as_ref().map(|p| p.weight.as_str()), None, &[], osib,
                        );
                        plan.push_str(&format!(
                            "DISPATCH|{}|tg={},{},1|th={},1,1|smem=0|args={}\n",
                            skname, (*nsib + sib_sched.num_sgs - 1) / sib_sched.num_sgs, *m,
                            sib_sched.num_sgs * 32, args.join(",")
                        ));
                    } else {
                        emit_msl_amx_gemm(
                            &mut msl, &skname, *m, *k, *nsib, false, false,
                            *act_dtype, *weight_layout, false, false, &[], sib_sched,
                        );
                        plan.push_str(&format!(
                            "DISPATCH|{}|tg={},{},1|th={},1,1|smem=0|args={},{},{}\n",
                            skname, (*nsib + sib_sched.bn - 1) / sib_sched.bn,
                            (*m + sib_sched.bm - 1) / sib_sched.bm,
                            sib_sched.num_sgs * 32, eff_a, wsib, osib
                        ));
                    }
                }
            }
            TileStage::RowReduce { name, op, x, weight, rows, cols, dtype, epilogue, out, vectorized, .. } => {
                emit_msl_row_reduce(&mut msl, name, *op, *rows, *cols, *dtype, epilogue, *vectorized);
                let mut args = vec![x.clone()];
                if let Some(w) = weight { args.push(w.clone()); }
                push_epilogue_arg_names(epilogue, &mut args);
                args.push(out.clone());
                let tx = if *vectorized { 128 } else { 64 };
                plan.push_str(&format!(
                    "DISPATCH|{}|tg={},1,1|th={},1,1|smem=0|args={}\n",
                    name, rows, tx, args.join(",")
                ));
            }
            TileStage::Elementwise { name, inputs, ops, numel, dtype, out, vectorized, streaming_cache, .. } => {
                let use_stream_burst = *streaming_cache
                    && *vectorized
                    && *numel % 16 == 0
                    && matches!(*dtype, IrDType::F32 | IrDType::F16);
                if use_stream_burst {
                    emit_msl_elementwise_streaming(&mut msl, name, inputs.len(), ops, *numel, *dtype);
                } else {
                    emit_msl_elementwise(&mut msl, name, inputs.len(), ops, *numel, *dtype, *vectorized);
                }
                let mut args = inputs.clone();
                args.push(out.clone());
                let elems_per_thr = if use_stream_burst {
                    16
                } else if *vectorized && *numel % 4 == 0 {
                    4
                } else {
                    1
                };
                let n_thrs = (*numel + elems_per_thr - 1) / elems_per_thr;
                let tgx = (n_thrs + 255) / 256;
                plan.push_str(&format!(
                    "DISPATCH|{}|tg={},1,1|th=256,1,1|smem=0|args={}\n",
                    name, tgx, args.join(",")
                ));
            }
            TileStage::AttentionStage { name, q, k, v, seq, q_dim, kv_dim, dtype, causal, online_tiled, use_mma, out, naive_scores_scratch, .. } => {
                let head_dim = 64usize.min(*kv_dim);
                let n_heads = (*q_dim / head_dim).max(1);
                let n_kv = (*kv_dim / head_dim).max(1);
                let can_mma_fa = *online_tiled
                    && *use_mma
                    && *dtype == IrDType::F16
                    && (head_dim == 64 || head_dim == 128)
                    && *q_dim % 8 == 0
                    && *kv_dim % 8 == 0;
                if can_mma_fa {
                    // Hardware AMX `simdgroup_half8x8` FlashAttention-2 (Br=32, Bc=32, 4 SGs, 15.4 KB < 16 KB -> 2 TGs/core,
                    // ZERO register spill, 2 barriers/iter, exp2 SFU, float4/half4 Softmax, simd_any rescale skip!)
                    emit_msl_flash_attention_mma(&mut msl, name, *seq, n_heads, n_kv, head_dim, *dtype, *causal);
                    let tgx = (*seq + 31) / 32;
                    let tgy = n_heads;
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},{},1|th=128,1,1|smem=0|args={},{},{},{}\n",
                        name, tgx, tgy, q, k, v, out
                    ));
                } else if *online_tiled {
                    emit_msl_flash_attention(&mut msl, name, *seq, n_heads, n_kv, head_dim, *dtype, *causal);
                    // 8 simdgroups (256 threads) cooperatively load K/V once for 8 query rows -> 2x-4x less K/V bandwidth!
                    let tgx = (*seq + 7) / 8;
                    let tgy = n_heads;
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},{},1|th=256,1,1|smem=0|args={},{},{},{}\n",
                        name, tgx, tgy, q, k, v, out
                    ));
                } else {
                    let sc_buf = naive_scores_scratch.clone().unwrap_or_else(|| format!("__scores_{}", name));
                    let k1 = format!("{}_qk_soft", name);
                    let k2 = format!("{}_pv", name);
                    emit_msl_naive_attention(&mut msl, &k1, &k2, *seq, n_heads, n_kv, head_dim, *dtype, *causal);
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},{},1|th=128,1,1|smem=0|args={},{},{}\n",
                        k1, *seq, n_heads, q, k, sc_buf
                    ));
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},{},1|th=64,1,1|smem=0|args={},{},{}\n",
                        k2, *seq, n_heads, sc_buf, v, out
                    ));
                }
            }
            TileStage::RaggedContraction { name, a, b, m_padded, m_valid, k, n, dtype, weight_layout, compact_enabled, schedule, out, .. } => {
                let eff_m = if *compact_enabled { *m_valid } else { *m_padded };
                let mut sc = *schedule;
                sc.split_k = 1;
                if sc.use_mma && eff_m >= 16 {
                    emit_msl_amx_gemm(&mut msl, name, eff_m, *k, *n, false, false, *dtype, *weight_layout, false, false, &[], sc);
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},{},1|th={},1,1|smem=0|args={},{},{}\n",
                        name, (*n + sc.bn - 1) / sc.bn, (eff_m + sc.bm - 1) / sc.bm, sc.num_sgs * 32, a, b, out
                    ));
                } else {
                    emit_msl_contraction_scalar(&mut msl, name, eff_m, *k, *n, false, false, *dtype, *dtype, *weight_layout, false, false, &[]);
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},{},1|th=16,16,1|smem=0|args={},{},{}\n",
                        name, (*n + 15) / 16, (eff_m + 15) / 16, a, b, out
                    ));
                }
            }
            TileStage::BlockSparseContraction { name, a, b, m, k, n, block_size, density_pct, dtype, weight_layout, sparse_enabled, schedule, out, .. } => {
                let mut sc = *schedule;
                sc.split_k = 1;
                if *sparse_enabled && sc.use_mma {
                    emit_msl_block_sparse_amx(&mut msl, name, *m, *k, *n, *block_size, *density_pct, *dtype, *weight_layout, sc);
                    let use_bsr64 = *weight_layout == LayoutTag::Bsr32NK && *dtype == IrDType::F16 && *m % 64 == 0 && *n % 32 == 0 && *k % 32 == 0;
                    let tiles_m = if use_bsr64 { *m / 64 } else { (*m + 31) / 32 };
                    let tiles_n = (*n + 31) / 32;
                    // Persistent work-stealing grid saturating all 40 cores!
                    let n_work = tiles_m * tiles_n;
                    let persistent_tgs = n_work.min(160);
                    let ctr_buf = format!("__sp_ctr_{}", name);
                    plan.push_str(&format!("ZERO_SCRATCH|{}|64\n", ctr_buf));
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},1,1|th=128,1,1|smem=0|args={},{},{},{}\n",
                        name, persistent_tgs, a, b, ctr_buf, out
                    ));
                } else if sc.use_mma {
                    emit_msl_amx_gemm(&mut msl, name, *m, *k, *n, false, false, *dtype, *weight_layout, false, false, &[], sc);
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},{},1|th={},1,1|smem=0|args={},{},{}\n",
                        name, (*n + sc.bn - 1) / sc.bn, (*m + sc.bm - 1) / sc.bm, sc.num_sgs * 32, a, b, out
                    ));
                } else {
                    emit_msl_contraction_scalar(&mut msl, name, *m, *k, *n, false, false, *dtype, *dtype, *weight_layout, false, false, &[]);
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},{},1|th=16,16,1|smem=0|args={},{},{}\n",
                        name, (*n + 15) / 16, (*m + 15) / 16, a, b, out
                    ));
                }
            }
            TileStage::TransposeStage { name, x, rows, cols, dtype, out, .. } => {
                let mt = dtype.msl_ty();
                msl.push_str(&format!(
                    "kernel void {name}(device const {mt}* x [[buffer(0)]], device {mt}* out [[buffer(1)]], uint2 gid [[thread_position_in_grid]]) {{\n\
                     \tif (gid.y < {rows}u && gid.x < {cols}u) out[gid.x * {rows}u + gid.y] = x[gid.y * {cols}u + gid.x];\n\
                     }}\n\n"
                ));
                plan.push_str(&format!(
                    "DISPATCH|{}|tg={},{},1|th=16,16,1|smem=0|args={},{}\n",
                    name, (*cols + 15) / 16, (*rows + 15) / 16, x, out
                ));
            }
            TileStage::CollectiveStage { name, kind, x, numel, dtype, devices, out, .. } => {
                // Single-host Metal simulation for SPMD Collectives (AllReduce / ZeRO-1 ReduceScatter / AllGather)
                let mt = dtype.msl_ty();
                let is_zero1_ag = matches!(kind, CollectiveKind::AllGather) && *devices > 1 && x.ends_with("_zero1_shard");
                if is_zero1_ag {
                    // When ZeRO-1 AdamW donated `param`, `out` already aliases `param` (1-thread sync on single host!)
                    msl.push_str(&format!(
                        "kernel void {name}(device const {mt}* x [[buffer(0)]], device {mt}* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n\
                         \tif (gid == 0u) {{ (void)x; (void)out; }}\n\
                         }}\n\n"
                    ));
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg=1,1,1|th=32,1,1|smem=0|args={},{}\n",
                        name, x, out
                    ));
                } else {
                    let eff_n = if matches!(kind, CollectiveKind::ReduceScatter) && *devices > 1 {
                        (*numel / *devices).max(1)
                    } else {
                        *numel
                    };
                    if eff_n % 4 == 0 && (*dtype == IrDType::F32 || *dtype == IrDType::F16) {
                        let vt = if *dtype == IrDType::F32 { "float4" } else { "half4" };
                        let n4 = eff_n / 4;
                        msl.push_str(&format!(
                            "kernel void {name}(device const {vt}* x [[buffer(0)]], device {vt}* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n\
                             \tif (gid < {n4}u) out[gid] = x[gid];\n\
                             }}\n\n"
                        ));
                        plan.push_str(&format!(
                            "DISPATCH|{}|tg={},1,1|th=256,1,1|smem=0|args={},{}\n",
                            name, (n4 + 255) / 256, x, out
                        ));
                    } else {
                        msl.push_str(&format!(
                            "kernel void {name}(device const {mt}* x [[buffer(0)]], device {mt}* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n\
                             \tif (gid < {eff_n}u) out[gid] = x[gid];\n\
                             }}\n\n"
                        ));
                        plan.push_str(&format!(
                            "DISPATCH|{}|tg={},1,1|th=256,1,1|smem=0|args={},{}\n",
                            name, (eff_n + 255) / 256, x, out
                        ));
                    }
                }
            }
            TileStage::ConcatContraction { stage_id, name, a, weights, m, k, n_per, total_n, dtype, packed_enabled, schedule, out } => {
                let mut sc = *schedule;
                sc.split_k = 1;
                if *packed_enabled {
                    // Facility #2: Pack `[W0 || W1 || W2]` into ONE contiguous `[total_n, k]` buffer at load time -> 1 wide 64x64 AMX GEMM!
                    let packed_w = format!("__hpack_{}", stage_id);
                    plan.push_str(&format!(
                        "CONCAT_NK|{}|{}|{}|{}\n",
                        packed_w, k, n_per, weights.join(",")
                    ));
                    emit_msl_amx_gemm(&mut msl, name, *m, *k, *total_n, false, false, *dtype, LayoutTag::ColMajorNK, false, false, &[], sc);
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},{},1|th={},1,1|smem=0|args={},{},{}\n",
                        name, (*total_n + sc.bn - 1) / sc.bn, (*m + sc.bm - 1) / sc.bm, sc.num_sgs * 32, a, packed_w, out
                    ));
                } else {
                    // Unpacked baseline: N separate GEMMs writing strided column slices of `out[m, total_n]`!
                    for (wi, wname) in weights.iter().enumerate() {
                        let kslice = format!("{}_w{}", name, wi);
                        let col_off = wi * (*n_per);
                        emit_msl_amx_gemm_col_slice(&mut msl, &kslice, *m, *k, *n_per, *total_n, col_off, *dtype, sc);
                        plan.push_str(&format!(
                            "DISPATCH|{}|tg={},{},1|th={},1,1|smem=0|args={},{},{}\n",
                            kslice, (*n_per + sc.bn - 1) / sc.bn, (*m + sc.bm - 1) / sc.bm, sc.num_sgs * 32, a, wname, out
                        ));
                    }
                }
            }
            TileStage::InPlaceAdamW { stage_id, name, param, grad, m, v, numel, dtype, donate_enabled, stochastic_round, streaming_cache: _, zero1_shards, out } => {
                let eff_numel = (*numel / (*zero1_shards).max(1)).max(1);
                // Keep 1x 128-bit `float4`/`half4` per thread on Apple Silicon Metal (hits 489 GB/s = 100% DRAM
                // without 4x-unroll register pressure; CUDA lowers `streaming_cache` to `__ldcs`/`__stcs`!).
                let use_burst = false;
                emit_msl_adamw(&mut msl, name, eff_numel, *dtype, *donate_enabled, *stochastic_round, use_burst);
                let elems_per_thr = if use_burst { 16 } else { 4 };
                let n_thrs = (eff_numel + elems_per_thr - 1) / elems_per_thr;
                let tgx = (n_thrs + 255) / 256;
                let sr_suffix = if *stochastic_round { ",__sr_seed" } else { "" };
                if *donate_enabled {
                    // Facility #3/#11/#13: Zero-Copy Buffer Donation (`out` aliases `param`, `m`/`v` updated in-place -> 0 B extra VRAM!)
                    plan.push_str("DONATE|0|0\n");
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},1,1|th=256,1,1|smem=0|args={},{},{},{}{}\n",
                        name, tgx, param, grad, m, v, sr_suffix
                    ));
                } else {
                    let elem_b = dtype.bytes_f64() as usize;
                    let sz = ((eff_numel * elem_b) + 255) & !255;
                    let nm = format!("__adamw_nm_{}", stage_id);
                    let nv = format!("__adamw_nv_{}", stage_id);
                    plan.push_str(&format!("SCRATCH|{}|{}\nSCRATCH|{}|{}\n", nm, sz, nv, sz));
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},1,1|th=256,1,1|smem=0|args={},{},{},{},{},{},{}{}\n",
                        name, tgx, param, grad, m, v, out, nm, nv, sr_suffix
                    ));
                }
            }
            TileStage::CrossEntropyStage { name, logits, targets, rows, vocab, padded_vocab, dtype, online_fused, donate_enabled, streaming_cache, scratch_probs, out, .. } => {
                if *online_fused {
                    // Facility #5/#10/#12: Single-Pass Online Monoid Cross-Entropy + Auto-Padded 128-bit (`50257->50304`)!
                    emit_msl_online_cross_entropy(&mut msl, name, *rows, *vocab, *padded_vocab, *dtype, *streaming_cache);
                    if *donate_enabled && *padded_vocab == *vocab {
                        plan.push_str("DONATE|0|0\n");
                        plan.push_str(&format!(
                            "DISPATCH|{}|tg={},1,1|th=256,1,1|smem=0|args={},{},{}\n",
                            name, rows, logits, targets, logits
                        ));
                    } else {
                        plan.push_str(&format!(
                            "DISPATCH|{}|tg={},1,1|th=256,1,1|smem=0|args={},{},{}\n",
                            name, rows, logits, targets, out
                        ));
                    }
                } else {
                    let pb = scratch_probs.clone().unwrap_or_else(|| format!("__ce_probs_{}", name));
                    let k1 = format!("{}_probs", name);
                    let k2 = format!("{}_dlogits", name);
                    emit_msl_naive_cross_entropy(&mut msl, &k1, &k2, *rows, *vocab, *dtype);
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},1,1|th=256,1,1|smem=0|args={},{}\n",
                        k1, rows, logits, pb
                    ));
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},1,1|th=256,1,1|smem=0|args={},{},{}\n",
                        k2, rows, pb, targets, out
                    ));
                }
            }
            TileStage::AmpQuantContraction { stage_id, name, x, nw, w, m, k, n, dtype, quant_enabled, weight_layout, schedule, out } => {
                let norm_tmp = format!("__amp_norm_{}", stage_id);
                let norm_k = format!("{}_norm", name);
                emit_msl_row_reduce(&mut msl, &norm_k, ReduceOp::RmsNorm, *m, *k, *dtype, &[], true);
                let nb = ((*m * *k * (dtype.bytes_f64() as usize)) + 255) & !255;
                plan.push_str(&format!("SCRATCH|{}|{}\n", norm_tmp, nb));
                plan.push_str(&format!(
                    "DISPATCH|{}|tg={},1,1|th=128,1,1|smem=0|args={},{},{}\n",
                    norm_k, m, x, nw, norm_tmp
                ));
                let mut sc = *schedule;
                sc.split_k = 1;
                if *quant_enabled && *weight_layout == LayoutTag::Int8RowScaleNK && *dtype == IrDType::F16 {
                    // Facility #6: W8A8/W8A16 INT8-Weight AMX Contraction (-50% weight DRAM + per-row dynamic scale epilogue!)
                    let (bm_w8, bn_w8) = if *m < 64 { (32usize, 32usize) } else { (64usize, 64usize) };
                    emit_msl_amx_w8_gemm(&mut msl, name, *m, *k, *n, bm_w8, bn_w8, sc);
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},{},1|th=128,1,1|smem=0|args={},{},{}\n",
                        name, (*n + bn_w8 - 1) / bn_w8, (*m + bm_w8 - 1) / bm_w8, norm_tmp, w, out
                    ));
                } else {
                    emit_msl_amx_gemm(&mut msl, name, *m, *k, *n, false, false, *dtype, *weight_layout, false, false, &[], sc);
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},{},1|th={},1,1|smem=0|args={},{},{}\n",
                        name, (*n + sc.bn - 1) / sc.bn, (*m + sc.bm - 1) / sc.bm, sc.num_sgs * 32, norm_tmp, w, out
                    ));
                }
            }
            TileStage::DynBucketContraction { stage_id, name, a, b, max_m, k, n, dtype, weight_layout, bucket_enabled, tile_swizzle, out } => {
                let ctrl = format!("__dyn_ctrl_{}", stage_id);
                plan.push_str(&format!("DYN_CTRL|{}|{}\n", ctrl, max_m));
                if *bucket_enabled {
                    // Facility #7: 3-Bucket O(1) Multi-Version Dynamic Dispatch (ZERO padding, ZERO JIT recompile!)
                    let kb0 = format!("{}_b0_gemv", name);
                    let kb1 = format!("{}_b1_amx32", name);
                    let kb2 = format!("{}_b2_amx64", name);
                    emit_msl_dyn_gemv(&mut msl, &kb0, *k, *n, *dtype);
                    emit_msl_dyn_amx(&mut msl, &kb1, 32, 32, 32, *k, *n, *dtype, false);
                    emit_msl_dyn_amx(&mut msl, &kb2, 64, 64, 16, *k, *n, *dtype, *tile_swizzle);
                    plan.push_str(&format!(
                        "DYN_BUCKET|1|8|{}|tgx={}|tgy_mode=DYN_M|th=128,1,1|args={},{},{},{}\n",
                        kb0, (*n + 3) / 4, a, b, ctrl, out
                    ));
                    plan.push_str(&format!(
                        "DYN_BUCKET|9|64|{}|tgx={}|tgy_mode=CEIL32|th=128,1,1|args={},{},{},{}\n",
                        kb1, (*n + 31) / 32, a, b, ctrl, out
                    ));
                    plan.push_str(&format!(
                        "DYN_BUCKET|65|{}|{}|tgx={}|tgy_mode=CEIL64|th=128,1,1|args={},{},{},{}\n",
                        max_m, kb2, (*n + 63) / 64, a, b, ctrl, out
                    ));
                } else {
                    // Static padded baseline: ALWAYS launches full `max_m` rows!
                    let sc = TileSchedule {
                        bm: 64, bn: 64, bk: 16, num_sgs: 4, rows_per_sg: 1, split_k: 1,
                        double_buf: true, swizzle_pad: 8, use_mma: true, stream_k: false,
                        warp_specialized: true, tile_swizzle: *tile_swizzle,
                    };
                    emit_msl_amx_gemm(&mut msl, name, *max_m, *k, *n, false, false, *dtype, *weight_layout, false, false, &[], sc);
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},{},1|th=128,1,1|smem=0|args={},{},{}\n",
                        name, (*n + 63) / 64, (*max_m + 63) / 64, a, b, out
                    ));
                }
            }
            TileStage::Conv2dStage { name, x, w, in_h, in_w, c_in, c_out, kh, kw, dtype, weight_layout, implicit_enabled, im2col_scratch, out, .. } => {
                let m = (*in_h) * (*in_w);
                let k_eff = (*c_in) * (*kh) * (*kw);
                if *implicit_enabled {
                    // Facility #8A: Zero-`im2col` Implicit GEMM 3x3 Conv2d AMX Kernel (0 KB im2col VRAM!)
                    emit_msl_implicit_conv2d_3x3(&mut msl, name, *in_h, *in_w, *c_in, *c_out, *dtype);
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},{},1|th=128,1,1|smem=0|args={},{},{}\n",
                        name, (*c_out + 63) / 64, (m + 63) / 64, x, w, out
                    ));
                } else {
                    let sc_buf = im2col_scratch.clone().unwrap_or_else(|| format!("__im2col_{}", name));
                    let k_i2c = format!("{}_im2col", name);
                    let k_mm = format!("{}_gemm", name);
                    emit_msl_explicit_im2col_3x3(&mut msl, &k_i2c, *in_h, *in_w, *c_in, *dtype);
                    let sc = TileSchedule {
                        bm: 64, bn: 64, bk: 16, num_sgs: 4, rows_per_sg: 1, split_k: 1,
                        double_buf: true, swizzle_pad: 8, use_mma: true, stream_k: false,
                        warp_specialized: true, tile_swizzle: false,
                    };
                    emit_msl_amx_gemm(&mut msl, &k_mm, m, k_eff, *c_out, false, false, *dtype, *weight_layout, false, false, &[], sc);
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},1,1|th=256,1,1|smem=0|args={},{}\n",
                        k_i2c, (m * k_eff + 255) / 256, x, sc_buf
                    ));
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},{},1|th=128,1,1|smem=0|args={},{},{}\n",
                        k_mm, (*c_out + 63) / 64, (m + 63) / 64, sc_buf, w, out
                    ));
                }
            }
            TileStage::ParallelScanStage { name, a, b, batch, seq_len, dtype, parallel_enabled, out, .. } => {
                if *parallel_enabled {
                    // Facility #8B: 2-Level SIMD+TG Associative Parallel Prefix Scan (`O(log L)` depth, 128-bit float4!)
                    emit_msl_parallel_scan(&mut msl, name, *batch, *seq_len, *dtype);
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},1,1|th=256,1,1|smem=0|args={},{},{}\n",
                        name, batch, a, b, out
                    ));
                } else {
                    // Sequential O(L) baseline
                    emit_msl_sequential_scan(&mut msl, name, *batch, *seq_len, *dtype);
                    plan.push_str(&format!(
                        "DISPATCH|{}|tg={},1,1|th=64,1,1|smem=0|args={},{},{}\n",
                        name, (*batch + 63) / 64, a, b, out
                    ));
                }
            }
        }
    }
    plan.push_str("END_GRAPH\n");
    (msl, plan)
}

fn push_epilogue_arg_names(ep: &[EpilogueOp], args: &mut Vec<String>) {
    for e in ep {
        match e {
            EpilogueOp::AddTensor { src, .. }
            | EpilogueOp::SubTensor { src }
            | EpilogueOp::MulTensor { src }
            | EpilogueOp::SiluBwd { saved_x: src }
            | EpilogueOp::GeluBwd { saved_x: src }
            | EpilogueOp::ReluBwd { saved_x: src } => args.push(src.clone()),
            _ => {}
        }
    }
}

fn build_contraction_args(
    a: &str,
    b: &str,
    dual_b: Option<&str>,
    ep: &[EpilogueOp],
    out: &str,
    extra_opt: Option<&str>,
) -> Vec<String> {
    let mut args = vec![a.to_string(), b.to_string()];
    if let Some(db) = dual_b { args.push(db.to_string()); }
    if let Some(ex) = extra_opt { args.push(ex.to_string()); }
    push_epilogue_arg_names(ep, &mut args);
    args.push(out.to_string());
    args
}

fn build_contraction_args_gemv(
    a: &str,
    b: &str,
    nw: Option<&str>,
    dual_b: Option<&str>,
    ep: &[EpilogueOp],
    out: &str,
) -> Vec<String> {
    let mut args = vec![a.to_string(), b.to_string()];
    if let Some(w) = nw { args.push(w.to_string()); }
    if let Some(db) = dual_b { args.push(db.to_string()); }
    push_epilogue_arg_names(ep, &mut args);
    args.push(out.to_string());
    args
}

fn emit_epilogue_params_and_code(
    ep: &[EpilogueOp],
    mt: &str,
    start_buf_idx: usize,
    row_expr: &str,
    col_expr: &str,
    n_cols: usize,
    val_var: &str,
) -> (String, String, usize) {
    let mut params = String::new();
    let mut code = String::new();
    let mut bi = start_buf_idx;
    for (ei, e) in ep.iter().enumerate() {
        match e {
            EpilogueOp::AddTensor { broadcast_row, .. } => {
                params.push_str(&format!(", device const {mt}* ep_{ei} [[buffer({bi})]]"));
                if *broadcast_row {
                    code.push_str(&format!("{val_var} += (float)ep_{ei}[{col_expr}]; "));
                } else {
                    code.push_str(&format!("{val_var} += (float)ep_{ei}[({row_expr}) * {n_cols}u + ({col_expr})]; "));
                }
                bi += 1;
            }
            EpilogueOp::SubTensor { .. } => {
                params.push_str(&format!(", device const {mt}* ep_{ei} [[buffer({bi})]]"));
                code.push_str(&format!("{val_var} -= (float)ep_{ei}[({row_expr}) * {n_cols}u + ({col_expr})]; "));
                bi += 1;
            }
            EpilogueOp::MulTensor { .. } => {
                params.push_str(&format!(", device const {mt}* ep_{ei} [[buffer({bi})]]"));
                code.push_str(&format!("{val_var} *= (float)ep_{ei}[({row_expr}) * {n_cols}u + ({col_expr})]; "));
                bi += 1;
            }
            EpilogueOp::Silu => {
                code.push_str(&format!("{val_var} = {val_var} / (1.0f + exp(-{val_var})); "));
            }
            EpilogueOp::Gelu => {
                code.push_str(&format!(
                    "{{ float _g = {val_var}; {val_var} = 0.5f * _g * (1.0f + tanh(0.7978845608f * (_g + 0.044715f * _g * _g * _g))); }} "
                ));
            }
            EpilogueOp::Relu => {
                code.push_str(&format!("{val_var} = max({val_var}, 0.0f); "));
            }
            EpilogueOp::SiluBwd { .. } => {
                params.push_str(&format!(", device const {mt}* ep_{ei} [[buffer({bi})]]"));
                code.push_str(&format!(
                    "{{ float _x = (float)ep_{ei}[({row_expr}) * {n_cols}u + ({col_expr})]; float _s = 1.0f / (1.0f + exp(-_x)); {val_var} *= _s * (1.0f + _x * (1.0f - _s)); }} "
                ));
                bi += 1;
            }
            EpilogueOp::GeluBwd { .. } => {
                params.push_str(&format!(", device const {mt}* ep_{ei} [[buffer({bi})]]"));
                code.push_str(&format!(
                    "{{ float _x = (float)ep_{ei}[({row_expr}) * {n_cols}u + ({col_expr})]; float _k = 0.7978845608f * (_x + 0.044715f * _x * _x * _x); float _t = tanh(_k); float _dk = 0.7978845608f * (1.0f + 0.134145f * _x * _x); {val_var} *= 0.5f * (1.0f + _t) + 0.5f * _x * (1.0f - _t * _t) * _dk; }} "
                ));
                bi += 1;
            }
            EpilogueOp::ReluBwd { .. } => {
                params.push_str(&format!(", device const {mt}* ep_{ei} [[buffer({bi})]]"));
                code.push_str(&format!(
                    "{{ float _x = (float)ep_{ei}[({row_expr}) * {n_cols}u + ({col_expr})]; {val_var} = (_x > 0.0f) ? {val_var} : 0.0f; }} "
                ));
                bi += 1;
            }
            EpilogueOp::Scale(s) => {
                code.push_str(&format!("{val_var} *= {:.8}f; ", s));
            }
        }
    }
    (params, code, bi)
}

/// 1. Baseline naive scalar MatMul (`tile_mma = false`) — exact reference baseline!
#[allow(clippy::too_many_arguments)]
fn emit_msl_contraction_scalar(
    msl: &mut String,
    kname: &str,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
    act_dt: IrDType,
    _w_dt: IrDType,
    w_layout: LayoutTag,
    has_dual: bool,
    dual_swiglu: bool,
    ep: &[EpilogueOp],
) {
    let mt = act_dt.msl_ty();
    let mut bi = 2usize;
    let dual_param = if has_dual {
        let s = format!(", device const {mt}* b2 [[buffer({bi})]]");
        bi += 1;
        s
    } else {
        String::new()
    };
    let (ep_params, ep_code, out_bi) = emit_epilogue_params_and_code(ep, mt, bi, "row", "col", n, "acc");
    let load_a = if trans_a {
        format!("(float)a[kk * {m}u + row]")
    } else {
        format!("(float)a[row * {k}u + kk]")
    };
    let b_nk = trans_b ^ (w_layout == LayoutTag::ColMajorNK);
    let load_b = if b_nk {
        format!("(float)b[col * {k}u + kk]")
    } else {
        format!("(float)b[kk * {n}u + col]")
    };
    let dual_acc_decl = if has_dual { "float acc2 = 0.0f;\n" } else { "" };
    let dual_fma = if has_dual {
        let lb2 = if b_nk {
            format!("(float)b2[col * {k}u + kk]")
        } else {
            format!("(float)b2[kk * {n}u + col]")
        };
        format!("acc2 = fma(av, {lb2}, acc2); ")
    } else {
        String::new()
    };
    let dual_fin = if has_dual && dual_swiglu {
        "acc = (acc / (1.0f + exp(-acc))) * acc2; "
    } else {
        ""
    };
    msl.push_str(&format!(
        "kernel void {kname}(\n\
         \tdevice const {mt}* a [[buffer(0)]], device const {mt}* b [[buffer(1)]]{dual_param}{ep_params},\n\
         \tdevice {mt}* out [[buffer({out_bi})]],\n\
         \tuint2 gid [[thread_position_in_grid]]\n\
         ) {{\n\
         \tuint row = gid.y, col = gid.x;\n\
         \tif (row >= {m}u || col >= {n}u) return;\n\
         \tfloat acc = 0.0f; {dual_acc_decl}\
         \tfor (uint kk = 0; kk < {k}u; kk++) {{\n\
         \t\tfloat av = {load_a};\n\
         \t\tacc = fma(av, {load_b}, acc); {dual_fma}\n\
         \t}}\n\
         \t{dual_fin}{ep_code}\n\
         \tout[row * {n}u + col] = ({mt})acc;\n\
         }}\n\n"
    ));
}

/// 2. Compute-Bound Hardware AMX `simdgroup_half8x8` Tiled MMA GEMM (`BM=32/64, BN=32/64, BK=32`)!
/// Supports: NN/NT/TN/TT, 128-bit `uint4` Coalesced Loads + Hardware `transpose_matrix=true` (`layout_prop`),
/// Register-Prefetched Software Pipeline (`warp_spec`), 8-SG Zero-Spill Register Allocation, Split-K (`stream_k`),
/// Dual SwiGLU, and Arbitrary In-Register Epilogue!
#[allow(clippy::too_many_arguments)]
fn emit_msl_amx_gemm(
    msl: &mut String,
    kname: &str,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
    act_dt: IrDType,
    w_layout: LayoutTag,
    has_dual: bool,
    dual_swiglu: bool,
    ep: &[EpilogueOp],
    sc: TileSchedule,
) {
    let mt = act_dt.msl_ty();
    let out_mt = if sc.split_k > 1 { "float" } else { mt };
    let bm = sc.bm;
    let bn = sc.bn;
    let bk = sc.bk;
    let n_sgs = sc.num_sgs; // 4 (2x2) or 8 (4x2 or 2x4)
    let threads = n_sgs * 32;
    let (sg_rows, sg_cols, sg_r_expr, sg_c_expr) = if n_sgs == 8 && bn > bm {
        (2usize, 4usize, "(sg >> 2)", "(sg & 3u)")
    } else if n_sgs == 8 {
        (4usize, 2usize, "(sg >> 1)", "(sg & 1u)")
    } else {
        (2usize, 2usize, "(sg >> 1)", "(sg & 1u)")
    };
    let sg_m = bm / sg_rows;
    let sg_n = bn / sg_cols;
    let tm = (sg_m / 8).max(1);
    let tn = (sg_n / 8).max(1);

    let pad = sc.swizzle_pad;
    // Exact XOR law for shared weights (Forward `trans_b=false` + Backward `trans_b=true` on same repacked buffer!):
    // If B was physically repacked to `ColMajorNK` (`[C, R]`), then Backward `B^T` (`trans_b=true`) reads it row-major!
    let b_is_nk = trans_b ^ (w_layout == LayoutTag::ColMajorNK);
    // When A is transposed (`A^T @ B`, i.e. ALL weight-gradient MatMuls `dW = A^T @ dOut` in Reverse-Mode AD!),
    // store `sA` as `[BK][BM + pad]` for ZERO-SCATTER 128-bit `uint4` coalesced loads + hardware `transpose=true`!
    let sa_transposed_smem = trans_a;
    // When B is NK (`[N, K]`), store `sB` as `[BN][BK + pad]` for ZERO-SCATTER 128-bit `uint4` copy
    // + zero-cost hardware `simdgroup_load(..., transpose_matrix = true)`!
    let sb_transposed_smem = b_is_nk;
    let sa_stride = if sa_transposed_smem { bm + pad } else { bk + pad };
    let sa_rows = if sa_transposed_smem { bk } else { bm };
    let sb_stride = if sb_transposed_smem { bk + pad } else { bn + pad };
    let sb_rows = if sb_transposed_smem { bn } else { bk };

    let mut bi = 2usize;
    let dual_param = if has_dual {
        let s = format!(", device const {mt}* b2 [[buffer({bi})]]");
        bi += 1;
        s
    } else {
        String::new()
    };
    let (ep_params, ep_code, out_bi) = emit_epilogue_params_and_code(ep, mt, bi, "gr", "gc", n, "val");

    let fast_a_u4 = !sa_transposed_smem && act_dt == IrDType::F16 && k % 8 == 0 && bk % 8 == 0 && sa_stride % 8 == 0 && m % bm == 0;
    let fast_a_t_u4 = sa_transposed_smem && act_dt == IrDType::F16 && m % 8 == 0 && bm % 8 == 0 && sa_stride % 8 == 0 && k % bk == 0;
    let fast_b_nk_u4 = sb_transposed_smem && act_dt == IrDType::F16 && k % 8 == 0 && bk % 8 == 0 && sb_stride % 8 == 0 && n % bn == 0;
    let fast_b_rm_u4 = !sb_transposed_smem && act_dt == IrDType::F16 && n % 8 == 0 && bn % 8 == 0 && sb_stride % 8 == 0 && k % bk == 0;

    let emit_load_a = |buf_expr: &str, k_expr: &str| -> String {
        if fast_a_u4 {
            let n_u4 = (bm * bk) / 8;
            let k8 = bk / 8;
            format!(
                "for (uint idx = lid; idx < {n_u4}u; idx += {threads}u) {{\n\
                 \tuint r = idx / {k8}u, c8 = (idx % {k8}u) * 8u;\n\
                 \t*(threadgroup uint4*)(&sA[{buf_expr}][r * {sa_stride}u + c8]) = *(device const uint4*)(&a[(m0 + r) * {k}u + ({k_expr}) + c8]);\n\
                 }}\n"
            )
        } else if fast_a_t_u4 {
            // ZERO-SCATTER 128-bit `uint4` contiguous copy from A[K, M] into sA[BK][BM + pad] for `A^T @ B`!
            let n_u4 = (bk * bm) / 8;
            let m8 = bm / 8;
            format!(
                "for (uint idx = lid; idx < {n_u4}u; idx += {threads}u) {{\n\
                 \tuint ck = idx / {m8}u, cm8 = (idx % {m8}u) * 8u;\n\
                 \t*(threadgroup uint4*)(&sA[{buf_expr}][ck * {sa_stride}u + cm8]) = *(device const uint4*)(&a[(({k_expr}) + ck) * {m}u + m0 + cm8]);\n\
                 }}\n"
            )
        } else if sa_transposed_smem && act_dt == IrDType::F16 && m % 4 == 0 && bm % 4 == 0 {
            let n_vec = (bk * bm) / 4;
            let m4 = bm / 4;
            format!(
                "for (uint idx = lid; idx < {n_vec}u; idx += {threads}u) {{\n\
                 \tuint ck = idx / {m4}u, cm4 = (idx % {m4}u) * 4u;\n\
                 \tuint gk = ({k_expr}) + ck, gr = m0 + cm4;\n\
                 \t*(threadgroup half4*)(&sA[{buf_expr}][ck * {sa_stride}u + cm4]) = (gk < {k}u && gr + 3u < {m}u) ? *(device const half4*)(&a[gk * {m}u + gr]) : half4(0.0h);\n\
                 }}\n"
            )
        } else if !sa_transposed_smem && act_dt == IrDType::F16 && k % 4 == 0 && bk % 4 == 0 {
            let n_vec = (bm * bk) / 4;
            let k4 = bk / 4;
            format!(
                "for (uint idx = lid; idx < {n_vec}u; idx += {threads}u) {{\n\
                 \tuint r = idx / {k4}u, c4 = (idx % {k4}u) * 4u;\n\
                 \tuint gr = m0 + r, gk = ({k_expr}) + c4;\n\
                 \t*(threadgroup half4*)(&sA[{buf_expr}][r * {sa_stride}u + c4]) = (gr < {m}u && gk + 3u < {k}u) ? *(device const half4*)(&a[gr * {k}u + gk]) : half4(0.0h);\n\
                 }}\n"
            )
        } else if sa_transposed_smem {
            let n_el = bk * bm;
            format!(
                "for (uint idx = lid; idx < {n_el}u; idx += {threads}u) {{\n\
                 \tuint ck = idx / {bm}u, cm = idx % {bm}u;\n\
                 \tuint gk = ({k_expr}) + ck, gr = m0 + cm;\n\
                 \tsA[{buf_expr}][ck * {sa_stride}u + cm] = (gk < {k}u && gr < {m}u) ? (half)a[gk * {m}u + gr] : half(0.0h);\n\
                 }}\n"
            )
        } else {
            let n_el = bm * bk;
            format!(
                "for (uint idx = lid; idx < {n_el}u; idx += {threads}u) {{\n\
                 \tuint r = idx / {bk}u, c = idx % {bk}u;\n\
                 \tuint gr = m0 + r, gk = ({k_expr}) + c;\n\
                 \tsA[{buf_expr}][r * {sa_stride}u + c] = (gr < {m}u && gk < {k}u) ? (half)a[gr * {k}u + gk] : half(0.0h);\n\
                 }}\n"
            )
        }
    };

    let emit_load_b = |b_ptr: &str, sb_arr: &str, buf_expr: &str, k_expr: &str| -> String {
        if fast_b_nk_u4 {
            // ZERO-SCATTER 128-bit `uint4` contiguous copy from B_NK[N, K] into sB[BN][BK + pad]!
            let n_u4 = (bn * bk) / 8;
            let k8 = bk / 8;
            format!(
                "for (uint idx = lid; idx < {n_u4}u; idx += {threads}u) {{\n\
                 \tuint cn = idx / {k8}u, ck8 = (idx % {k8}u) * 8u;\n\
                 \t*(threadgroup uint4*)(&{sb_arr}[{buf_expr}][cn * {sb_stride}u + ck8]) = *(device const uint4*)(&{b_ptr}[(n0 + cn) * {k}u + ({k_expr}) + ck8]);\n\
                 }}\n"
            )
        } else if sb_transposed_smem && act_dt == IrDType::F16 && k % 4 == 0 && bk % 4 == 0 {
            let n_vec = (bn * bk) / 4;
            let k4 = bk / 4;
            format!(
                "for (uint idx = lid; idx < {n_vec}u; idx += {threads}u) {{\n\
                 \tuint cn = idx / {k4}u, ck4 = (idx % {k4}u) * 4u;\n\
                 \tuint gn = n0 + cn, gk = ({k_expr}) + ck4;\n\
                 \t*(threadgroup half4*)(&{sb_arr}[{buf_expr}][cn * {sb_stride}u + ck4]) = (gn < {n}u && gk + 3u < {k}u) ? *(device const half4*)(&{b_ptr}[gn * {k}u + gk]) : half4(0.0h);\n\
                 }}\n"
            )
        } else if fast_b_rm_u4 {
            let n_u4 = (bk * bn) / 8;
            let n8 = bn / 8;
            format!(
                "for (uint idx = lid; idx < {n_u4}u; idx += {threads}u) {{\n\
                 \tuint ck = idx / {n8}u, cn8 = (idx % {n8}u) * 8u;\n\
                 \t*(threadgroup uint4*)(&{sb_arr}[{buf_expr}][ck * {sb_stride}u + cn8]) = *(device const uint4*)(&{b_ptr}[(({k_expr}) + ck) * {n}u + n0 + cn8]);\n\
                 }}\n"
            )
        } else if !sb_transposed_smem && act_dt == IrDType::F16 && n % 4 == 0 && bn % 4 == 0 {
            let n_vec = (bk * bn) / 4;
            let n4 = bn / 4;
            format!(
                "for (uint idx = lid; idx < {n_vec}u; idx += {threads}u) {{\n\
                 \tuint ck = idx / {n4}u, cn4 = (idx % {n4}u) * 4u;\n\
                 \tuint gk = ({k_expr}) + ck, gn = n0 + cn4;\n\
                 \t*(threadgroup half4*)(&{sb_arr}[{buf_expr}][ck * {sb_stride}u + cn4]) = (gk < {k}u && gn + 3u < {n}u) ? *(device const half4*)(&{b_ptr}[gk * {n}u + gn]) : half4(0.0h);\n\
                 }}\n"
            )
        } else {
            let n_el = bk * bn;
            if sb_transposed_smem {
                format!(
                    "for (uint idx = lid; idx < {n_el}u; idx += {threads}u) {{\n\
                     \tuint cn = idx / {bk}u, ck = idx % {bk}u;\n\
                     \tuint gn = n0 + cn, gk = ({k_expr}) + ck;\n\
                     \t{sb_arr}[{buf_expr}][cn * {sb_stride}u + ck] = (gn < {n}u && gk < {k}u) ? (half){b_ptr}[gn * {k}u + gk] : half(0.0h);\n\
                     }}\n"
                )
            } else {
                format!(
                    "for (uint idx = lid; idx < {n_el}u; idx += {threads}u) {{\n\
                     \tuint ck = idx / {bn}u, cn = idx % {bn}u;\n\
                     \tuint gk = ({k_expr}) + ck, gn = n0 + cn;\n\
                     \t{sb_arr}[{buf_expr}][ck * {sb_stride}u + cn] = (gk < {k}u && gn < {n}u) ? (half){b_ptr}[gk * {n}u + gn] : half(0.0h);\n\
                     }}\n"
                )
            }
        }
    };

    let nbufs = if sc.double_buf && !has_dual { 2usize } else { 1usize };
    let sa_elems = sa_rows * sa_stride;
    let sb_elems = sb_rows * sb_stride;
    let sb2_decl = if has_dual {
        format!("\tthreadgroup half sB2[1][{sb_elems}];\n")
    } else {
        String::new()
    };
    let acc2_decl = if has_dual {
        format!(
            "\tsimdgroup_float8x8 mc2[{tm}][{tn}];\n\
             \t#pragma unroll\n\
             \tfor (uint r = 0; r < {tm}u; r++) for (uint c = 0; c < {tn}u; c++) mc2[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n"
        )
    } else {
        String::new()
    };

    let load_ma_expr = |buf_expr: &str| -> String {
        if sa_transposed_smem {
            format!("simdgroup_load(ma[r], &sA[{buf_expr}][kk * {sa_stride}u + sg_r + r * 8u], {sa_stride}ul, ulong2(0, 0), true);")
        } else {
            format!("simdgroup_load(ma[r], &sA[{buf_expr}][(sg_r + r * 8u) * {sa_stride}u + kk], {sa_stride}ul);")
        }
    };
    let load_mb_expr = |sb_arr: &str, buf_expr: &str| -> String {
        if sb_transposed_smem {
            format!("simdgroup_load(mb[c], &{sb_arr}[{buf_expr}][(sg_c + c * 8u) * {sb_stride}u + kk], {sb_stride}ul, ulong2(0, 0), true);")
        } else {
            format!("simdgroup_load(mb[c], &{sb_arr}[{buf_expr}][kk * {sb_stride}u + sg_c + c * 8u], {sb_stride}ul);")
        }
    };

    let mma_inner = |buf_expr: &str| -> String {
        let lma = load_ma_expr(buf_expr);
        let lmb = load_mb_expr("sB", buf_expr);
        let mut s = format!(
            "#pragma unroll\n\
             for (uint kk = 0; kk < {bk}u; kk += 8u) {{\n\
             \tsimdgroup_half8x8 ma[{tm}], mb[{tn}];\n\
             \t#pragma unroll\n\
             \tfor (uint r = 0; r < {tm}u; r++) {lma}\n\
             \t#pragma unroll\n\
             \tfor (uint c = 0; c < {tn}u; c++) {lmb}\n\
             \t#pragma unroll\n\
             \tfor (uint r = 0; r < {tm}u; r++) for (uint c = 0; c < {tn}u; c++) simdgroup_multiply_accumulate(mc[r][c], ma[r], mb[c], mc[r][c]);\n"
        );
        if has_dual {
            let lmb2 = load_mb_expr("sB2", "0").replace("mb[c]", "mb2[c]");
            s.push_str(&format!(
                "\tsimdgroup_half8x8 mb2[{tn}];\n\
                 \t#pragma unroll\n\
                 \tfor (uint c = 0; c < {tn}u; c++) {lmb2}\n\
                 \t#pragma unroll\n\
                 \tfor (uint r = 0; r < {tm}u; r++) for (uint c = 0; c < {tn}u; c++) simdgroup_multiply_accumulate(mc2[r][c], ma[r], mb2[c], mc2[r][c]);\n"
            ));
        }
        s.push_str("}\n");
        s
    };

    // Split-K range calculation
    let k_per_split = ((k + sc.split_k - 1) / sc.split_k + bk - 1) / bk * bk;
    let k_loop_code = if nbufs == 2 {
        // Producer/Consumer Ping-Pong Double-Buffered Pipeline (`warp_spec = true`) — ONLY 1 barrier per K-tile!
        format!(
            "\tuint k_start = tgid.z * {k_per_split}u;\n\
             \tuint k_end = min(k_start + {k_per_split}u, {k}u);\n\
             \tif (k_start < k_end) {{\n\
             \t\t{la0}{lb0}\t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
             \t\tuint buf = 0u;\n\
             \t\tfor (uint kt = k_start; kt < k_end; kt += {bk}u) {{\n\
             \t\t\tuint next_kt = kt + {bk}u;\n\
             \t\t\tuint nbuf = buf ^ 1u;\n\
             \t\t\tif (next_kt < k_end) {{\n\
             \t\t\t\t{la_next}{lb_next}\t\t\t}}\n\
             \t\t\t{mma_cur}\n\
             \t\t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
             \t\t\tbuf = nbuf;\n\
             \t\t}}\n\
             \t}}\n",
            la0 = emit_load_a("0", "k_start"),
            lb0 = emit_load_b("b", "sB", "0", "k_start"),
            la_next = emit_load_a("nbuf", "next_kt"),
            lb_next = emit_load_b("b", "sB", "nbuf", "next_kt"),
            mma_cur = mma_inner("buf"),
        )
    } else {
        let lb2 = if has_dual {
            emit_load_b("b2", "sB2", "0", "kt")
        } else {
            String::new()
        };
        format!(
            "\tuint k_start = tgid.z * {k_per_split}u;\n\
             \tuint k_end = min(k_start + {k_per_split}u, {k}u);\n\
             \tfor (uint kt = k_start; kt < k_end; kt += {bk}u) {{\n\
             \t\t{la}{lb}{lb2}\t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
             \t\t{mma}\n\
             \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
             \t}}\n",
            la = emit_load_a("0", "kt"),
            lb = emit_load_b("b", "sB", "0", "kt"),
            mma = mma_inner("0"),
        )
    };

    // Always accumulate in FP32 (`simdgroup_float8x8`) for exact numerical accuracy even at K=8192+!
    // When `tn % 2 == 0`, store 2 adjacent 8x8 tiles (8x16 = 128 floats) into sC[sg] and write ONE coalesced
    // 64-bit `half4` per lane -> 2x fewer barriers + 4x wider global stores!
    let fast_vec_store = !has_dual && ep.is_empty() && sc.split_k == 1 && act_dt == IrDType::F16 && tn >= 2 && tn % 2 == 0 && m % bm == 0 && n % bn == 0 && n % 4 == 0;
    let out_off_expr = if sc.split_k > 1 {
        format!("tgid.z * ({m}u * {n}u) + gr * {n}u + gc")
    } else {
        format!("gr * {n}u + gc")
    };
    let sc_decls = if fast_vec_store {
        format!("\tthreadgroup float sC[{n_sgs}][128];\n")
    } else if has_dual && dual_swiglu {
        format!("\tthreadgroup float sC[{n_sgs}][64];\n\tthreadgroup float sC2[{n_sgs}][64];\n")
    } else {
        format!("\tthreadgroup float sC[{n_sgs}][64];\n")
    };
    let dual_store_stage = if has_dual && dual_swiglu {
        "\tsimdgroup_store(mc2[r][c], &sC2[sg][0], 8ul);\n\
         \tsimdgroup_barrier(mem_flags::mem_threadgroup);\n"
    } else {
        ""
    };
    let dual_combine_expr = if has_dual && dual_swiglu {
        "float uval = sC2[sg][e]; val = (val / (1.0f + exp(-val))) * uval; "
    } else {
        ""
    };

    let acc_ty = "simdgroup_float8x8";
    let acc_zero = "make_filled_simdgroup_matrix<float, 8, 8>(0.0f)";
    let store_code = if fast_vec_store {
        format!(
            "\tuint er = lane >> 2, ec4 = (lane & 3u) * 4u;\n\
             \t#pragma unroll\n\
             \tfor (uint r = 0; r < {tm}u; r++) {{\n\
             \t\t#pragma unroll\n\
             \t\tfor (uint c = 0; c < {tn}u; c += 2u) {{\n\
             \t\t\tsimdgroup_store(mc[r][c], &sC[sg][0], 16ul);\n\
             \t\t\tsimdgroup_store(mc[r][c + 1u], &sC[sg][8], 16ul);\n\
             \t\t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
             \t\t\tuint gr = m0 + sg_r + r * 8u + er;\n\
             \t\t\tuint gc = n0 + sg_c + c * 8u + ec4;\n\
             \t\t\tfloat4 v4 = *(threadgroup const float4*)(&sC[sg][er * 16u + ec4]);\n\
             \t\t\t*(device half4*)(&out[gr * {n}u + gc]) = half4(v4);\n\
             \t\t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
             \t\t}}\n\
             \t}}\n"
        )
    } else {
        format!(
            "\tfor (uint r = 0; r < {tm}u; r++) {{\n\
             \t\tfor (uint c = 0; c < {tn}u; c++) {{\n\
             \t\t\tsimdgroup_store(mc[r][c], &sC[sg][0], 8ul);\n\
             {dual_store_stage}\t\t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
             \t\t\tfor (uint e = lane; e < 64u; e += 32u) {{\n\
             \t\t\t\tuint gr = m0 + sg_r + r * 8u + (e >> 3);\n\
             \t\t\t\tuint gc = n0 + sg_c + c * 8u + (e & 7u);\n\
             \t\t\t\tif (gr < {m}u && gc < {n}u) {{\n\
             \t\t\t\t\tfloat val = sC[sg][e]; {dual_combine_expr}{ep_code}\n\
             \t\t\t\t\tout[{out_off_expr}] = ({out_mt})val;\n\
             \t\t\t\t}}\n\
             \t\t\t}}\n\
             \t\t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
             \t\t}}\n\
             \t}}\n"
        )
    };

    let tiles_m_grid = (m + bm - 1) / bm;
    let tiles_n_grid = (n + bn - 1) / bn;
    let group_m = 8usize.min(tiles_m_grid);
    let tile_coord_code = if sc.tile_swizzle && tiles_m_grid >= 8 && tiles_n_grid >= 8 {
        // Facility #1: 2D Serpentine L2/SLC Tile Swizzling (`GROUP_M = 8` + alternating N-sweep!)
        // Keeps concurrently executing TGs on shared A/B tiles AND reuses hot SLC `B` tiles across M-group boundaries!
        format!(
            "\tuint _pid = tgid.y * {tiles_n_grid}u + tgid.x;\n\
             \tuint _grp_sz = {group_m}u * {tiles_n_grid}u;\n\
             \tuint _gid = _pid / _grp_sz;\n\
             \tuint _first_m = _gid * {group_m}u;\n\
             \tuint _gm = min({tiles_m_grid}u - _first_m, {group_m}u);\n\
             \tuint _rem = _pid % _grp_sz;\n\
             \tuint _nt_raw = _rem / _gm;\n\
             \tuint _nt = ((_gid & 1u) != 0u) ? ({tiles_n_grid}u - 1u - _nt_raw) : _nt_raw;\n\
             \tuint m0 = (_first_m + (_rem % _gm)) * {bm}u, n0 = _nt * {bn}u;\n"
        )
    } else {
        format!("\tuint m0 = tgid.y * {bm}u, n0 = tgid.x * {bn}u;\n")
    };

    msl.push_str(&format!(
        "kernel void {kname}(\n\
         \tdevice const {mt}* a [[buffer(0)]], device const {mt}* b [[buffer(1)]]{dual_param}{ep_params},\n\
         \tdevice {out_mt}* out [[buffer({out_bi})]],\n\
         \tuint3 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
         \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
         ) {{\n\
         \tthreadgroup half sA[{nbufs}][{sa_elems}];\n\
         \tthreadgroup half sB[{nbufs}][{sb_elems}];\n\
         {sb2_decl}{sc_decls}{tile_coord_code}\
         \tuint sg_r = {sg_r_expr} * {sg_m}u, sg_c = {sg_c_expr} * {sg_n}u;\n\
         \t{acc_ty} mc[{tm}][{tn}];\n\
         \t#pragma unroll\n\
         \tfor (uint r = 0; r < {tm}u; r++) for (uint c = 0; c < {tn}u; c++) mc[r][c] = {acc_zero};\n\
         {acc2_decl}{k_loop_code}{store_code}}}\n\n"
    ));
}

/// 3. Memory-Bound 128-bit Coalesced SIMD-Group GEMV (`M <= 8`) with optional:
///    - Inline Threadgroup SRAM Prologue `RMSNorm`/`LayerNorm` (`general_fusion`)
///    - Coalesced `W_NK[N, K]` `half4` loads (`layout_prop`)
///    - Stage 3 OCP `MXFP4` E2M1+E8M0 4-bit block-32 hardware register unpack (`mxfp4_quant`)
///    - Dual Horizontal `SwiGLU` + Arbitrary Register Epilogue + Split-K (`stream_k`)!
#[allow(clippy::too_many_arguments)]
fn emit_msl_gemv_vectorized(
    msl: &mut String,
    kname: &str,
    m: usize,
    k: usize,
    n: usize,
    act_dt: IrDType,
    _w_dt: IrDType,
    w_layout: LayoutTag,
    prologue: Option<&PrologueNorm>,
    has_dual: bool,
    dual_swiglu: bool,
    ep: &[EpilogueOp],
    sc: TileSchedule,
) {
    let mt = act_dt.msl_ty();
    let out_mt = if sc.split_k > 1 { "float" } else { mt };
    let w_mt = if w_layout.is_mxfp4() { "uchar" } else { mt };
    let mut bi = 2usize;
    let nw_param = if prologue.is_some() {
        let s = format!(", device const {mt}* nw [[buffer({bi})]]");
        bi += 1;
        s
    } else {
        String::new()
    };
    let dual_param = if has_dual {
        let s = format!(", device const {w_mt}* b2 [[buffer({bi})]]");
        bi += 1;
        s
    } else {
        String::new()
    };
    let rps = if !has_dual
        && (w_layout == LayoutTag::MXFP4Super16NK
            || (w_layout == LayoutTag::ColMajorNK && act_dt == IrDType::F16 && k % 8 == 0))
    {
        sc.rows_per_sg.max(1).min(4)
    } else {
        1usize
    };
    let (ep_params, ep_code, out_bi) = emit_epilogue_params_and_code(ep, mt, bi, "row", "col", n, "acc");

    let n_sgs = sc.num_sgs;
    let threads = n_sgs * 32;
    let sx_is_half = k > 4096;
    let sx_ty = if sx_is_half { "half" } else { "float" };
    let prologue_code = if let Some(p) = prologue {
        let is_ln = p.kind == ReduceOp::LayerNorm;
        format!(
            "\tthreadgroup {sx_ty} sx[{k}];\n\
             \tthreadgroup float s_red[16];\n\
             \tfloat lsum = 0.0f, lsq = 0.0f;\n\
             \tfor (uint kk = lid; kk < {k}u; kk += {threads}u) {{\n\
             \t\tfloat v = (float)a[row * {k}u + kk]; sx[kk] = ({sx_ty})v; lsum += v; lsq = fma(v, v, lsq);\n\
             \t}}\n\
             \tlsum = simd_sum(lsum); lsq = simd_sum(lsq);\n\
             \tif (lane == 0u) {{ s_red[sg] = lsq; s_red[8u + sg] = lsum; }}\n\
             \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
             \tfloat tsq = 0.0f, tsum = 0.0f;\n\
             \tfor (uint s = 0; s < {n_sgs}u; s++) {{ tsq += s_red[s]; tsum += s_red[8u + s]; }}\n\
             \tfloat mean = {mean_expr};\n\
             \tfloat inv = rsqrt(max(tsq / (float){k}u - mean * mean, 0.0f) + 1e-5f);\n\
             \tfor (uint kk = lid; kk < {k}u; kk += {threads}u) sx[kk] = ({sx_ty})(((float)sx[kk] - mean) * inv * (float)nw[kk]);\n\
             \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n",
            mean_expr = if is_ln { format!("tsum / (float){k}u") } else { "0.0f".to_string() },
        )
    } else {
        String::new()
    };

    let out_idx_expr = if sc.split_k > 1 {
        format!("tgid.z * ({m}u * {n}u) + row * {n}u + col")
    } else {
        format!("row * {n}u + col")
    };
    let dual_fin = if has_dual && dual_swiglu {
        "acc = (acc / (1.0f + exp(-acc))) * acc2; "
    } else {
        ""
    };

    let load_x4 = |off: &str| -> String {
        if prologue.is_some() {
            if sx_is_half {
                format!("float4(*(threadgroup const half4*)(&sx[{off}]))")
            } else {
                format!("*(threadgroup const float4*)(&sx[{off}])")
            }
        } else {
            format!("float4(*(device const half4*)(&a[row * {k}u + ({off})]))")
        }
    };

    // Fast path: Multi-Row Shared-Activation GEMV (`rps = 2 or 4` output rows per SIMD-group)!
    // Amortizes activation loads by `rps`x AND keeps `rps` independent DRAM streams in flight per lane!
    if rps >= 2 && w_layout == LayoutTag::MXFP4Super16NK {
        let n_blk = k / 32;
        let n_sb = k / 512;
        let blk_per_split = (n_blk + sc.split_k - 1) / sc.split_k;
        let mut pref = String::new();
        for r in 0..rps {
            pref.push_str(&format!(
                "\t\tulong off{r} = ((ulong)min(col_base + {r}u, {n_m1}u) * {n_sb}ul + (ulong)sb) * 272ul;\n\
                 \t\tuint4 q{r} = *(const device uint4*)(b + off{r} + (ulong)bi * 16ul);\n\
                 \t\tfloat sc{r} = as_type<float>(((uint)b[off{r} + 256ul + (ulong)bi]) << 23) * 16384.0f;\n\
                 \t\tfloat4 dv{r} = float4(0.0f);\n",
                n_m1 = n - 1,
            ));
        }
        let mut words_body = String::new();
        for (wi, comp) in ["x", "y", "z", "w"].iter().enumerate() {
            let boff = wi * 8;
            words_body.push_str(&format!(
                "\t\t{{ float4 xv0 = {lx0}; float4 xv1 = {lx1};\n",
                lx0 = load_x4(&format!("base_k + {boff}u")),
                lx1 = load_x4(&format!("base_k + {}u", boff + 4)),
            ));
            for r in 0..rps {
                words_body.push_str(&format!(
                    "\t\t\tdv{r} = fma(xv0, float4(mxfp4_e2m1_x4(q{r}.{comp})), dv{r});\n\
                     \t\t\tdv{r} = fma(xv1, float4(mxfp4_e2m1_x4(q{r}.{comp} >> 16)), dv{r});\n"
                ));
            }
            words_body.push_str("\t\t}\n");
        }
        let mut acc_init = String::new();
        let mut acc_fma = String::new();
        let mut acc_fin = String::new();
        for r in 0..rps {
            acc_init.push_str(&format!("\tfloat acc{r} = 0.0f;\n"));
            acc_fma.push_str(&format!("\t\tacc{r} = fma((dv{r}.x + dv{r}.y) + (dv{r}.z + dv{r}.w), sc{r}, acc{r});\n"));
            acc_fin.push_str(&format!(
                "\tacc{r} = simd_sum(acc{r});\n\
                 \tif (lane == 0u) {{\n\
                 \t\tuint col = col_base + {r}u;\n\
                 \t\tif (col < {n}u) {{ float acc = acc{r}; {ep_code} out[{out_idx_expr}] = ({out_mt})acc; }}\n\
                 \t}}\n"
            ));
        }
        msl.push_str(&format!(
            "kernel void {kname}(\n\
             \tdevice const {mt}* a [[buffer(0)]], device const uchar* b [[buffer(1)]]{nw_param}{ep_params},\n\
             \tdevice {out_mt}* out [[buffer({out_bi})]],\n\
             \tuint3 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
             \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
             ) {{\n\
             \tuint row = tgid.y, col_base = (tgid.x * {n_sgs}u + sg) * {rps}u;\n\
             {prologue_code}\
             \tif (row >= {m}u || col_base >= {n}u) return;\n\
             {acc_init}\
             \tuint b_start = tgid.z * {blk_per_split}u;\n\
             \tuint b_end = min(b_start + {blk_per_split}u, {n_blk}u);\n\
             \tfor (uint blk = b_start + lane; blk < b_end; blk += 32u) {{\n\
             \t\tuint sb = blk >> 4, bi = blk & 15u;\n\
             \t\tuint base_k = blk * 32u;\n\
             {pref}{words_body}{acc_fma}\t}}\n\
             {acc_fin}}}\n\n"
        ));
        return;
    }

    if rps >= 2 && w_layout == LayoutTag::ColMajorNK && act_dt == IrDType::F16 && k % 8 == 0 {
        let k8 = k / 8;
        let k8_per_split = (k8 + sc.split_k - 1) / sc.split_k;
        let load_x8 = if prologue.is_some() {
            if sx_is_half {
                "float4 x0 = float4(*(threadgroup const half4*)(&sx[kk8 * 8u])); float4 x1 = float4(*(threadgroup const half4*)(&sx[kk8 * 8u + 4u]));"
            } else {
                "float4 x0 = *(threadgroup const float4*)(&sx[kk8 * 8u]); float4 x1 = *(threadgroup const float4*)(&sx[kk8 * 8u + 4u]);"
            }
        } else {
            "float4 x0 = float4(*(device const half4*)(&a[row * K + kk8 * 8u])); float4 x1 = float4(*(device const half4*)(&a[row * K + kk8 * 8u + 4u]));"
        };
        let mut acc_init = String::new();
        let mut dots = String::new();
        let mut acc_fin = String::new();
        for r in 0..rps {
            acc_init.push_str(&format!("\tfloat acc{r} = 0.0f; uint c{r} = min(col_base + {r}u, {n_m1}u);\n", n_m1 = n - 1));
            dots.push_str(&format!(
                "\t\tfloat4 w{r}_0 = float4(*(device const half4*)(&b[c{r} * K + kk8 * 8u]));\n\
                 \t\tfloat4 w{r}_1 = float4(*(device const half4*)(&b[c{r} * K + kk8 * 8u + 4u]));\n\
                 \t\tacc{r} += dot(x0, w{r}_0) + dot(x1, w{r}_1);\n"
            ));
            acc_fin.push_str(&format!(
                "\tacc{r} = simd_sum(acc{r});\n\
                 \tif (lane == 0u) {{\n\
                 \t\tuint col = col_base + {r}u;\n\
                 \t\tif (col < {n}u) {{ float acc = acc{r}; {ep_code} out[{out_idx_expr}] = ({out_mt})acc; }}\n\
                 \t}}\n"
            ));
        }
        msl.push_str(&format!(
            "kernel void {kname}(\n\
             \tdevice const {mt}* a [[buffer(0)]], device const {mt}* b [[buffer(1)]]{nw_param}{ep_params},\n\
             \tdevice {out_mt}* out [[buffer({out_bi})]],\n\
             \tuint3 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
             \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
             ) {{\n\
             \tconst uint K = {k}u;\n\
             \tuint row = tgid.y, col_base = (tgid.x * {n_sgs}u + sg) * {rps}u;\n\
             {prologue_code}\
             \tif (row >= {m}u || col_base >= {n}u) return;\n\
             {acc_init}\
             \tuint s0 = tgid.z * {k8_per_split}u, s1 = min(s0 + {k8_per_split}u, {k8}u);\n\
             \tfor (uint kk8 = s0 + lane; kk8 < s1; kk8 += 32u) {{\n\
             \t\t{load_x8}\n\
             {dots}\t}}\n\
             {acc_fin}}}\n\n"
        ));
        return;
    }

    let dot_body = if w_layout == LayoutTag::MXFP4Super16NK {
        let n_blk = k / 32;
        let n_sb = k / 512;
        let blk_per_split = (n_blk + sc.split_k - 1) / sc.split_k;
        let emit_u32_dot = |pk_expr: &str, base_expr: &str, acc_v: &str| -> String {
            format!(
                "{{ uint pk = {pk_expr}; uint b0 = {base_expr};\n\
                 \tfloat4 xv0 = {lx0}; float4 xv1 = {lx1};\n\
                 \t{acc_v} += dot(xv0, float4(mxfp4_e2m1_x4(pk)));\n\
                 \t{acc_v} += dot(xv1, float4(mxfp4_e2m1_x4(pk >> 16))); }}\n",
                lx0 = load_x4("b0"),
                lx1 = load_x4("b0 + 4u"),
            )
        };
        let dual_mxfp4 = if has_dual {
            format!(
                "\t\tuint4 q2 = *(const device uint4*)(b2 + sb_off + (ulong)bi * 16ul);\n\
                 \t\tfloat s2 = as_type<float>(((uint)b2[sb_off + 256ul + (ulong)bi]) << 23) * 16384.0f;\n\
                 \t\tfloat d2 = 0.0f;\n\
                 \t\t{d20}{d21}{d22}{d23}\
                 \t\tacc2 = fma(d2, s2, acc2);\n",
                d20 = emit_u32_dot("q2.x", "base_k", "d2"),
                d21 = emit_u32_dot("q2.y", "base_k + 8u", "d2"),
                d22 = emit_u32_dot("q2.z", "base_k + 16u", "d2"),
                d23 = emit_u32_dot("q2.w", "base_k + 24u", "d2"),
            )
        } else {
            String::new()
        };
        format!(
            "\tuint b_start = tgid.z * {blk_per_split}u;\n\
             \tuint b_end = min(b_start + {blk_per_split}u, {n_blk}u);\n\
             \tfor (uint blk = b_start + lane; blk < b_end; blk += 32u) {{\n\
             \t\tuint sb = blk >> 4, bi = blk & 15u;\n\
             \t\tulong sb_off = ((ulong)col * {n_sb}ul + (ulong)sb) * 272ul;\n\
             \t\tuint4 q = *(const device uint4*)(b + sb_off + (ulong)bi * 16ul);\n\
             \t\tfloat sc = as_type<float>(((uint)b[sb_off + 256ul + (ulong)bi]) << 23) * 16384.0f;\n\
             \t\tuint base_k = blk * 32u;\n\
             \t\tfloat d1 = 0.0f;\n\
             \t\t{d10}{d11}{d12}{d13}\
             \t\tacc = fma(d1, sc, acc);\n\
             {dual_mxfp4}\t}}\n",
            d10 = emit_u32_dot("q.x", "base_k", "d1"),
            d11 = emit_u32_dot("q.y", "base_k + 8u", "d1"),
            d12 = emit_u32_dot("q.z", "base_k + 16u", "d1"),
            d13 = emit_u32_dot("q.w", "base_k + 24u", "d1"),
        )
    } else if w_layout == LayoutTag::MXFP4Block32NK {
        // Stage 3: OCP MXFP4 (32 E2M1 nibbles packed in 16-byte aligned uint4 + E8M0 shared exponent byte)!
        // Uses zero-branch IEEE-754 `half4` bit-synthesis `mxfp4_e2m1_x4` (exact 2^-14 scale folded into E8M0 * 16384.0f!)
        // + 128-bit `half4` vectorized activation loads!
        let n_blk = k / 32;
        let nib_bytes = n * n_blk * 16;
        let blk_per_split = (n_blk + sc.split_k - 1) / sc.split_k;
        let emit_u32_dot = |pk_expr: &str, base_expr: &str, acc_v: &str| -> String {
            format!(
                "{{ uint pk = {pk_expr}; uint b0 = {base_expr};\n\
                 \tfloat4 xv0 = {lx0}; float4 xv1 = {lx1};\n\
                 \t{acc_v} += dot(xv0, float4(mxfp4_e2m1_x4(pk)));\n\
                 \t{acc_v} += dot(xv1, float4(mxfp4_e2m1_x4(pk >> 16))); }}\n",
                lx0 = load_x4("b0"),
                lx1 = load_x4("b0 + 4u"),
            )
        };
        let dual_mxfp4 = if has_dual {
            format!(
                "\t\tconst device uint4* nib2 = (const device uint4*)b2;\n\
                 \t\tconst device uchar* sc2 = b2 + {nib_bytes}ul;\n\
                 \t\tuint4 q2 = nib2[col * {n_blk}u + blk];\n\
                 \t\tfloat s2 = as_type<float>(((uint)sc2[col * {n_blk}u + blk]) << 23) * 16384.0f;\n\
                 \t\tfloat d2 = 0.0f;\n\
                 \t\t{d20}{d21}{d22}{d23}\
                 \t\tacc2 = fma(d2, s2, acc2);\n",
                d20 = emit_u32_dot("q2.x", "base_k", "d2"),
                d21 = emit_u32_dot("q2.y", "base_k + 8u", "d2"),
                d22 = emit_u32_dot("q2.z", "base_k + 16u", "d2"),
                d23 = emit_u32_dot("q2.w", "base_k + 24u", "d2"),
            )
        } else {
            String::new()
        };
        format!(
            "\tconst device uint4* nib1 = (const device uint4*)b;\n\
             \tconst device uchar* sc1 = b + {nib_bytes}ul;\n\
             \tuint b_start = tgid.z * {blk_per_split}u;\n\
             \tuint b_end = min(b_start + {blk_per_split}u, {n_blk}u);\n\
             \tfor (uint blk = b_start + lane; blk < b_end; blk += 32u) {{\n\
             \t\tuint4 q = nib1[col * {n_blk}u + blk];\n\
             \t\tfloat sc = as_type<float>(((uint)sc1[col * {n_blk}u + blk]) << 23) * 16384.0f;\n\
             \t\tuint base_k = blk * 32u;\n\
             \t\tfloat d1 = 0.0f;\n\
             \t\t{d10}{d11}{d12}{d13}\
             \t\tacc = fma(d1, sc, acc);\n\
             {dual_mxfp4}\t}}\n",
            d10 = emit_u32_dot("q.x", "base_k", "d1"),
            d11 = emit_u32_dot("q.y", "base_k + 8u", "d1"),
            d12 = emit_u32_dot("q.z", "base_k + 16u", "d1"),
            d13 = emit_u32_dot("q.w", "base_k + 24u", "d1"),
        )
    } else if w_layout == LayoutTag::ColMajorNK && act_dt == IrDType::F16 && k % 8 == 0 {
        // 128-bit coalesced `half4 x 2` (8 halfs/lane = 256 halfs/simdgroup) on transposed W_NK[N, K]!
        let k8 = k / 8;
        let k8_per_split = (k8 + sc.split_k - 1) / sc.split_k;
        let load_x8 = if prologue.is_some() {
            if sx_is_half {
                "float4 x0 = float4(*(threadgroup const half4*)(&sx[kk8 * 8u])); float4 x1 = float4(*(threadgroup const half4*)(&sx[kk8 * 8u + 4u]));"
            } else {
                "float4 x0 = *(threadgroup const float4*)(&sx[kk8 * 8u]); float4 x1 = *(threadgroup const float4*)(&sx[kk8 * 8u + 4u]);"
            }
        } else {
            "float4 x0 = float4(*(device const half4*)(&a[row * K + kk8 * 8u])); float4 x1 = float4(*(device const half4*)(&a[row * K + kk8 * 8u + 4u]));"
        };
        let dual_vec = if has_dual {
            "float4 u0 = float4(*(device const half4*)(&b2[col * K + kk8 * 8u])); float4 u1 = float4(*(device const half4*)(&b2[col * K + kk8 * 8u + 4u])); acc2 += dot(x0, u0) + dot(x1, u1);"
        } else {
            ""
        };
        format!(
            "\tconst uint K = {k}u;\n\
             \tuint s0 = tgid.z * {k8_per_split}u, s1 = min(s0 + {k8_per_split}u, {k8}u);\n\
             \tfor (uint kk8 = s0 + lane; kk8 < s1; kk8 += 32u) {{\n\
             \t\t{load_x8}\n\
             \t\tfloat4 w0 = float4(*(device const half4*)(&b[col * K + kk8 * 8u]));\n\
             \t\tfloat4 w1 = float4(*(device const half4*)(&b[col * K + kk8 * 8u + 4u]));\n\
             \t\tacc += dot(x0, w0) + dot(x1, w1);\n\
             \t\t{dual_vec}\n\
             \t}}\n"
        )
    } else {
        // Un-transposed row-major W[K, N] (shows real coalescing penalty when `layout_prop = false`!)
        let k_per_split = (k + sc.split_k - 1) / sc.split_k;
        let load_x = if prologue.is_some() { "sx[kk]" } else { "(float)a[row * K + kk]" };
        let dual_sc = if has_dual { "acc2 = fma(xv, (float)b2[kk * N + col], acc2);" } else { "" };
        format!(
            "\tconst uint K = {k}u, N = {n}u;\n\
             \tuint s0 = tgid.z * {k_per_split}u, s1 = min(s0 + {k_per_split}u, K);\n\
             \tfor (uint kk = s0 + lane; kk < s1; kk += 32u) {{\n\
             \t\tfloat xv = {load_x};\n\
             \t\tacc = fma(xv, (float)b[kk * N + col], acc); {dual_sc}\n\
             \t}}\n"
        )
    };

    let acc2_decl = if has_dual { "float acc2 = 0.0f;" } else { "" };
    let acc2_sum = if has_dual { "acc2 = simd_sum(acc2);" } else { "" };

    msl.push_str(&format!(
        "kernel void {kname}(\n\
         \tdevice const {mt}* a [[buffer(0)]], device const {w_mt}* b [[buffer(1)]]{nw_param}{dual_param}{ep_params},\n\
         \tdevice {out_mt}* out [[buffer({out_bi})]],\n\
         \tuint3 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
         \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
         ) {{\n\
         \tuint row = tgid.y, col = tgid.x * {n_sgs}u + sg;\n\
         {prologue_code}\
         \tif (row >= {m}u || col >= {n}u) return;\n\
         \tfloat acc = 0.0f; {acc2_decl}\n\
         {dot_body}\
         \tacc = simd_sum(acc); {acc2_sum}\n\
         \tif (lane == 0u) {{\n\
         \t\t{dual_fin}{ep_code}\n\
         \t\tout[{out_idx_expr}] = ({out_mt})acc;\n\
         \t}}\n\
         }}\n\n"
    ));
}

/// Deterministic two-pass Split-K / Stream-K reduction + fused arbitrary epilogue!
fn emit_msl_splitk_reduce(
    msl: &mut String,
    kname: &str,
    m: usize,
    n: usize,
    split_k: usize,
    act_dt: IrDType,
    ep: &[EpilogueOp],
) {
    let mt = act_dt.msl_ty();
    let total = m * n;
    let (ep_params, ep_code, out_bi) = emit_epilogue_params_and_code(ep, mt, 1, "row", "col", n, "acc");
    let coord_decl = if ep.is_empty() {
        String::new()
    } else {
        format!("\tuint row = gid / {n}u, col = gid % {n}u;\n")
    };
    msl.push_str(&format!(
        "kernel void {kname}(\n\
         \tdevice const float* partials [[buffer(0)]]{ep_params}, device {mt}* out [[buffer({out_bi})]],\n\
         \tuint gid [[thread_position_in_grid]]\n\
         ) {{\n\
         \tif (gid >= {total}u) return;\n\
         \tfloat acc = 0.0f;\n\
         \tfor (uint s = 0; s < {split_k}u; s++) acc += partials[s * {total}u + gid];\n\
         {coord_decl}\t{ep_code}\n\
         \tout[gid] = ({mt})acc;\n\
         }}\n\n"
    ));
}

/// 4. General Single-Pass SIMD Row Reduction (`RmsNorm`, `LayerNorm`, `Softmax`, `Sum`, `Max`) + Epilogue!
fn emit_msl_row_reduce(
    msl: &mut String,
    kname: &str,
    op: ReduceOp,
    rows: usize,
    cols: usize,
    dt: IrDType,
    ep: &[EpilogueOp],
    vectorized: bool,
) {
    let mt = dt.msl_ty();
    let has_w = matches!(op, ReduceOp::RmsNorm | ReduceOp::LayerNorm);
    let mut bi = 1usize;
    let w_param = if has_w {
        let s = format!(", device const {mt}* w [[buffer({bi})]]");
        bi += 1;
        s
    } else {
        String::new()
    };
    let (ep_params, ep_code, out_bi) = emit_epilogue_params_and_code(ep, mt, bi, "row", "c", cols, "val");
    let tx = if vectorized { 128usize } else { 64usize };
    let n_sg = tx / 32;

    match op {
        ReduceOp::RmsNorm | ReduceOp::LayerNorm => {
            let is_ln = op == ReduceOp::LayerNorm;
            msl.push_str(&format!(
                "kernel void {kname}(\n\
                 \tdevice const {mt}* x [[buffer(0)]]{w_param}{ep_params}, device {mt}* out [[buffer({out_bi})]],\n\
                 \tuint row [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
                 \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
                 ) {{\n\
                 \tif (row >= {rows}u) return;\n\
                 \tthreadgroup float s_sq[8]; threadgroup float s_sm[8];\n\
                 \tfloat lsq = 0.0f, lsm = 0.0f;\n\
                 \tfor (uint c = lid; c < {cols}u; c += {tx}u) {{ float v = (float)x[row * {cols}u + c]; lsm += v; lsq = fma(v, v, lsq); }}\n\
                 \tlsq = simd_sum(lsq); lsm = simd_sum(lsm);\n\
                 \tif (lane == 0u) {{ s_sq[sg] = lsq; s_sm[sg] = lsm; }}\n\
                 \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
                 \tfloat tsq = 0.0f, tsm = 0.0f;\n\
                 \tfor (uint s = 0; s < {n_sg}u; s++) {{ tsq += s_sq[s]; tsm += s_sm[s]; }}\n\
                 \tfloat mean = {mean_expr};\n\
                 \tfloat inv = rsqrt(max(tsq / (float){cols}u - mean * mean, 0.0f) + 1e-5f);\n\
                 \tfor (uint c = lid; c < {cols}u; c += {tx}u) {{\n\
                 \t\tfloat val = ((float)x[row * {cols}u + c] - mean) * inv * (float)w[c];\n\
                 \t\t{ep_code}\n\
                 \t\tout[row * {cols}u + c] = ({mt})val;\n\
                 \t}}\n\
                 }}\n\n",
                mean_expr = if is_ln { format!("tsm / (float){cols}u") } else { "0.0f".to_string() },
            ));
        }
        ReduceOp::Softmax => {
            msl.push_str(&format!(
                "kernel void {kname}(\n\
                 \tdevice const {mt}* x [[buffer(0)]]{ep_params}, device {mt}* out [[buffer({out_bi})]],\n\
                 \tuint row [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
                 \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
                 ) {{\n\
                 \tif (row >= {rows}u) return;\n\
                 \tthreadgroup float s_m[8]; threadgroup float s_l[8];\n\
                 \tfloat lm = -1e30f;\n\
                 \tfor (uint c = lid; c < {cols}u; c += {tx}u) lm = max(lm, (float)x[row * {cols}u + c]);\n\
                 \tlm = simd_max(lm);\n\
                 \tif (lane == 0u) s_m[sg] = lm;\n\
                 \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
                 \tfloat gm = -1e30f; for (uint s = 0; s < {n_sg}u; s++) gm = max(gm, s_m[s]);\n\
                 \tfloat ll = 0.0f;\n\
                 \tfor (uint c = lid; c < {cols}u; c += {tx}u) ll += exp((float)x[row * {cols}u + c] - gm);\n\
                 \tll = simd_sum(ll);\n\
                 \tif (lane == 0u) s_l[sg] = ll;\n\
                 \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
                 \tfloat gl = 0.0f; for (uint s = 0; s < {n_sg}u; s++) gl += s_l[s];\n\
                 \tfloat inv = 1.0f / max(gl, 1e-12f);\n\
                 \tfor (uint c = lid; c < {cols}u; c += {tx}u) {{\n\
                 \t\tfloat val = exp((float)x[row * {cols}u + c] - gm) * inv;\n\
                 \t\t{ep_code}\n\
                 \t\tout[row * {cols}u + c] = ({mt})val;\n\
                 \t}}\n\
                 }}\n\n"
            ));
        }
        ReduceOp::Sum | ReduceOp::Max => {
            msl.push_str(&format!(
                "kernel void {kname}(device const {mt}* x [[buffer(0)]], device {mt}* out [[buffer(1)]], uint row [[threadgroup_position_in_grid]]) {{\n\
                 \tif (row >= {rows}u) return;\n\
                 \tfloat acc = 0.0f; for (uint c = 0; c < {cols}u; c++) acc += (float)x[row * {cols}u + c];\n\
                 \tout[row] = ({mt})acc;\n\
                 }}\n\n"
            ));
        }
    }
}

/// 5B. Facility #12 (`streaming_cache`): 64-Byte Full-Cacheline 4x128-Bit Streaming Elementwise Burst!
/// Reads/writes 4 independent `float4`/`half4` vectors per thread (64B full cacheline on FP32, ILP=4)!
fn emit_msl_elementwise_streaming(
    msl: &mut String,
    kname: &str,
    n_inputs: usize,
    ops: &[ElemOp],
    numel: usize,
    dt: IrDType,
) {
    let vt = if dt == IrDType::F32 { "float4" } else { "half4" };
    let in2 = if n_inputs >= 2 { format!(", device const {vt}* b [[buffer(1)]]") } else { String::new() };
    let out_bi = n_inputs.max(1);
    let n16 = numel / 16;

    let apply_vec = |vname: &str, bname: &str| -> String {
        let mut s = String::new();
        for (idx, op) in ops.iter().enumerate() {
            match op {
                ElemOp::Add if idx == 0 && n_inputs >= 2 => s.push_str(&format!("{vname} = {vname} + {bname}; ")),
                ElemOp::Sub if idx == 0 && n_inputs >= 2 => s.push_str(&format!("{vname} = {vname} - {bname}; ")),
                ElemOp::Mul if idx == 0 && n_inputs >= 2 => s.push_str(&format!("{vname} = {vname} * {bname}; ")),
                ElemOp::SwiGlu if idx == 0 && n_inputs >= 2 => s.push_str(&format!("{vname} = ({vname} / (1.0f + exp(-{vname}))) * {bname}; ")),
                ElemOp::Silu => s.push_str(&format!("{vname} = {vname} / (1.0f + exp(-{vname})); ")),
                ElemOp::Gelu => s.push_str(&format!("{vname} = 0.5f * {vname} * (1.0f + tanh(0.7978845608f * ({vname} + 0.044715f * {vname} * {vname} * {vname}))); ")),
                ElemOp::Relu => s.push_str(&format!("{vname} = max({vname}, float4(0.0f)); ")),
                ElemOp::Scale { scale_bits } => s.push_str(&format!("{vname} *= {:.8}f; ", f32::from_bits(*scale_bits))),
                _ => {}
            }
        }
        s
    };

    let mut loads = String::new();
    let mut computes = String::new();
    let mut stores = String::new();
    for u in 0..4 {
        loads.push_str(&format!("\tfloat4 v{u} = float4(a[base + {u}u]);\n"));
        if n_inputs >= 2 {
            loads.push_str(&format!("\tfloat4 w{u} = float4(b[base + {u}u]);\n"));
        }
        let body = apply_vec(&format!("v{u}"), &format!("w{u}"));
        computes.push_str(&format!("\t{body}\n"));
        stores.push_str(&format!("\tout[base + {u}u] = {vt}(v{u});\n"));
    }

    msl.push_str(&format!(
        "kernel void {kname}(device const {vt}* a [[buffer(0)]]{in2}, device {vt}* out [[buffer({out_bi})]], uint gid [[thread_position_in_grid]]) {{\n\
         \tif (gid >= {n16}u) return;\n\
         \tulong base = (ulong)gid * 4ul;\n\
         \t// 64B full-cacheline streaming burst (ILP=4 independent 128-bit vector transactions!)\n\
         {loads}{computes}{stores}}}\n\n"
    ));
}

/// 5. General Elementwise Vectorized Kernel (supports arbitrary unary chains & binary ops!).
fn emit_msl_elementwise(
    msl: &mut String,
    kname: &str,
    n_inputs: usize,
    ops: &[ElemOp],
    numel: usize,
    dt: IrDType,
    vectorized: bool,
) {
    let mt = dt.msl_ty();
    let in2 = if n_inputs >= 2 { format!(", device const {mt}* b [[buffer(1)]]") } else { String::new() };
    let out_bi = n_inputs.max(1);

    let mut body = String::new();
    for (idx, op) in ops.iter().enumerate() {
        match op {
            ElemOp::Add if idx == 0 && n_inputs >= 2 => body.push_str("v = v + (float)b[i]; "),
            ElemOp::Sub if idx == 0 && n_inputs >= 2 => body.push_str("v = v - (float)b[i]; "),
            ElemOp::Mul if idx == 0 && n_inputs >= 2 => body.push_str("v = v * (float)b[i]; "),
            ElemOp::SwiGlu if idx == 0 && n_inputs >= 2 => body.push_str("v = (v / (1.0f + exp(-v))) * (float)b[i]; "),
            ElemOp::SiluBwd if idx == 0 && n_inputs >= 2 => body.push_str("{ float _x = (float)b[i]; float _s = 1.0f / (1.0f + exp(-_x)); v *= _s * (1.0f + _x * (1.0f - _s)); } "),
            ElemOp::GeluBwd if idx == 0 && n_inputs >= 2 => body.push_str("{ float _x = (float)b[i]; float _k = 0.7978845608f * (_x + 0.044715f * _x * _x * _x); float _t = tanh(_k); float _dk = 0.7978845608f * (1.0f + 0.134145f * _x * _x); v *= 0.5f * (1.0f + _t) + 0.5f * _x * (1.0f - _t * _t) * _dk; } "),
            ElemOp::ReluBwd if idx == 0 && n_inputs >= 2 => body.push_str("v = ((float)b[i] > 0.0f) ? v : 0.0f; "),
            ElemOp::Silu => body.push_str("v = v / (1.0f + exp(-v)); "),
            ElemOp::Gelu => body.push_str("v = 0.5f * v * (1.0f + tanh(0.7978845608f * (v + 0.044715f * v * v * v))); "),
            ElemOp::Relu => body.push_str("v = max(v, 0.0f); "),
            ElemOp::Scale { scale_bits } => body.push_str(&format!("v *= {:.8}f; ", f32::from_bits(*scale_bits))),
            _ => {}
        }
    }

    if vectorized && numel % 4 == 0 {
        let n4 = numel / 4;
        msl.push_str(&format!(
            "kernel void {kname}(device const {mt}* a [[buffer(0)]]{in2}, device {mt}* out [[buffer({out_bi})]], uint gid [[thread_position_in_grid]]) {{\n\
             \tif (gid >= {n4}u) return;\n\
             \tuint base = gid * 4u;\n\
             \tfor (uint u = 0; u < 4u; u++) {{\n\
             \t\tuint i = base + u;\n\
             \t\tfloat v = (float)a[i];\n\
             \t\t{body}\n\
             \t\tout[i] = ({mt})v;\n\
             \t}}\n\
             }}\n\n"
        ));
    } else {
        msl.push_str(&format!(
            "kernel void {kname}(device const {mt}* a [[buffer(0)]]{in2}, device {mt}* out [[buffer({out_bi})]], uint i [[thread_position_in_grid]]) {{\n\
             \tif (i >= {numel}u) return;\n\
             \tfloat v = (float)a[i];\n\
             \t{body}\n\
             \tout[i] = ({mt})v;\n\
             }}\n\n"
        ));
    }
}

/// 6A-MMA. Hardware AMX `simdgroup_half8x8` FlashAttention-2 (`general_fusion && tile_mma`, ZERO O(S^2) HBM!).
/// 4 simdgroups (128 threads) own Br=32 query rows (8 rows/SG, ZERO register spill!) and stream Bc=32 K/V blocks
/// through separate bank-conflict-free `sK` + `sV` (`15.4 KB < 16 KB` -> 2 concurrent TGs/core, ONLY 2 TG barriers/iter!),
/// using native 1-cycle `exp2()` SFU, `float4`/`half4` vectorized Softmax, and `simd_any(al != 1.0f)` exact rescale-skip!
#[allow(clippy::too_many_arguments)]
fn emit_msl_flash_attention_mma(
    msl: &mut String,
    kname: &str,
    seq: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    dt: IrDType,
    causal: bool,
) {
    let mt = dt.msl_ty();
    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv * head_dim;
    let grp = (n_heads / n_kv).max(1);
    let scale_log2 = (1.0f32 / (head_dim as f32).sqrt()) * 1.4426950408889634f32;
    let d8 = head_dim / 8;
    let d_stride = head_dim + 8; // 72 (D=64) or 136 (D=128) -> multiple of 8 for 128-bit uint4 + zero bank conflicts!
    let n_u4 = (32 * head_dim) / 8;
    let d_u4 = head_dim / 8;
    let max_k_tg = if causal { "min(m0 + 32u, SEQ)" } else { "SEQ" };
    let causal_cond = if causal {
        "if (grow >= SEQ || gk >= SEQ || gk > grow) sc = -1e30f;"
    } else {
        "if (grow >= SEQ || gk >= SEQ) sc = -1e30f;"
    };
    let unmasked = if causal {
        "(kb + 31u <= grow) && (grow < SEQ)"
    } else {
        "(kb + 31u < SEQ) && (grow < SEQ)"
    };

    msl.push_str(&format!(
        "kernel void {kname}(\n\
         \tdevice const {mt}* Q [[buffer(0)]], device const {mt}* K [[buffer(1)]],\n\
         \tdevice const {mt}* V [[buffer(2)]], device {mt}* O [[buffer(3)]],\n\
         \tuint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
         \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
         ) {{\n\
         \tconst uint SEQ = {seq}u, D = {head_dim}u, QS = {q_stride}u, KVS = {kv_stride}u;\n\
         \tuint m0 = tgid.x * 32u, h = tgid.y, kv_h = h / {grp}u;\n\
         \tthreadgroup half sK[32 * {d_stride}];\n\
         \tthreadgroup half sV[32 * {d_stride}];\n\
         \tthreadgroup float sS[4][256];\n\
         \tthreadgroup half sP[4][256];\n\
         \tthreadgroup float s_m[4][8];\n\
         \tthreadgroup float s_l[4][8];\n\
         \tthreadgroup float s_alpha[4][8];\n\
         \t// Load Q[m0..m0+31, 0..D-1] via sK once into per-SG AMX register tiles `mq[0..{d8}-1]` (ZERO sQ SMEM!)\n\
         \tfor (uint idx = lid; idx < {n_u4}u; idx += 128u) {{\n\
         \t\tuint r = idx / {d_u4}u, c8 = (idx % {d_u4}u) * 8u;\n\
         \t\tuint gq = m0 + r;\n\
         \t\t*(threadgroup uint4*)(&sK[r * {d_stride}u + c8]) = (gq < SEQ) ? *(device const uint4*)(&Q[gq * QS + h * D + c8]) : uint4(0u);\n\
         \t}}\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \tsimdgroup_half8x8 mq[{d8}];\n\
         \t#pragma unroll\n\
         \tfor (uint d = 0; d < {d8}u; d++) simdgroup_load(mq[d], &sK[(sg * 8u) * {d_stride}u + d * 8u], {d_stride}ul);\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \tsimdgroup_float8x8 mo[{d8}];\n\
         \t#pragma unroll\n\
         \tfor (uint d = 0; d < {d8}u; d++) mo[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n\
         \tif (lane < 8u) {{ s_m[sg][lane] = -1e30f; s_l[sg][lane] = 0.0f; s_alpha[sg][lane] = 1.0f; }}\n\
         \tuint r_sm = lane >> 2, c_sm = (lane & 3u) * 8u;\n\
         \tuint grow = m0 + sg * 8u + r_sm;\n\
         \tuint max_k = {max_k_tg};\n\
         \tfor (uint kb = 0; kb < max_k; kb += 32u) {{\n\
         \t\t// 128-bit coalesced uint4 cooperative load of K and V (32 x D) — ONLY 2 barriers per iteration!\n\
         \t\tfor (uint idx = lid; idx < {n_u4}u; idx += 128u) {{\n\
         \t\t\tuint tr = idx / {d_u4}u, tc8 = (idx % {d_u4}u) * 8u;\n\
         \t\t\tuint gk = kb + tr;\n\
         \t\t\tif (gk < SEQ) {{\n\
         \t\t\t\tuint koff = gk * KVS + kv_h * D + tc8;\n\
         \t\t\t\t*(threadgroup uint4*)(&sK[tr * {d_stride}u + tc8]) = *(device const uint4*)(&K[koff]);\n\
         \t\t\t\t*(threadgroup uint4*)(&sV[tr * {d_stride}u + tc8]) = *(device const uint4*)(&V[koff]);\n\
         \t\t\t}} else {{\n\
         \t\t\t\t*(threadgroup uint4*)(&sK[tr * {d_stride}u + tc8]) = uint4(0u);\n\
         \t\t\t\t*(threadgroup uint4*)(&sV[tr * {d_stride}u + tc8]) = uint4(0u);\n\
         \t\t\t}}\n\
         \t\t}}\n\
         \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\t// Stage A: AMX `S[8x32] = Q[8xD] @ K[32xD]^T` with hardware transpose_matrix=true!\n\
         \t\tsimdgroup_float8x8 ms[4];\n\
         \t\t#pragma unroll\n\
         \t\tfor (uint t = 0; t < 4u; t++) ms[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n\
         \t\t#pragma unroll\n\
         \t\tfor (uint d = 0; d < {d8}u; d++) {{\n\
         \t\t\t#pragma unroll\n\
         \t\t\tfor (uint t = 0; t < 4u; t++) {{\n\
         \t\t\t\tsimdgroup_half8x8 mk;\n\
         \t\t\t\tsimdgroup_load(mk, &sK[(t * 8u) * {d_stride}u + d * 8u], {d_stride}ul, ulong2(0, 0), true);\n\
         \t\t\t\tsimdgroup_multiply_accumulate(ms[t], mq[d], mk, ms[t]);\n\
         \t\t\t}}\n\
         \t\t}}\n\
         \t\t#pragma unroll\n\
         \t\tfor (uint t = 0; t < 4u; t++) simdgroup_store(ms[t], &sS[sg][t * 8u], 32ul);\n\
         \t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\t// Stage B: Vectorized float4/half4 + single-cycle `exp2()` Online Block-Row Softmax (<16KB SMEM = 2 TGs/core!)\n\
         \t\tfloat4 sv0 = *(threadgroup const float4*)(&sS[sg][r_sm * 32u + c_sm]) * {scale_log2:.8}f;\n\
         \t\tfloat4 sv1 = *(threadgroup const float4*)(&sS[sg][r_sm * 32u + c_sm + 4u]) * {scale_log2:.8}f;\n\
         \t\tif (!({unmasked})) {{\n\
         \t\t\tfloat sc;\n\
         \t\t\t{{ uint gk = kb + c_sm + 0u; sc = sv0.x; {causal_cond} sv0.x = sc; }}\n\
         \t\t\t{{ uint gk = kb + c_sm + 1u; sc = sv0.y; {causal_cond} sv0.y = sc; }}\n\
         \t\t\t{{ uint gk = kb + c_sm + 2u; sc = sv0.z; {causal_cond} sv0.z = sc; }}\n\
         \t\t\t{{ uint gk = kb + c_sm + 3u; sc = sv0.w; {causal_cond} sv0.w = sc; }}\n\
         \t\t\t{{ uint gk = kb + c_sm + 4u; sc = sv1.x; {causal_cond} sv1.x = sc; }}\n\
         \t\t\t{{ uint gk = kb + c_sm + 5u; sc = sv1.y; {causal_cond} sv1.y = sc; }}\n\
         \t\t\t{{ uint gk = kb + c_sm + 6u; sc = sv1.z; {causal_cond} sv1.z = sc; }}\n\
         \t\t\t{{ uint gk = kb + c_sm + 7u; sc = sv1.w; {causal_cond} sv1.w = sc; }}\n\
         \t\t}}\n\
         \t\tfloat4 mx4 = max(sv0, sv1);\n\
         \t\tfloat lmax = max(max(mx4.x, mx4.y), max(mx4.z, mx4.w));\n\
         \t\tlmax = max(lmax, simd_shuffle_xor(lmax, 1u));\n\
         \t\tlmax = max(lmax, simd_shuffle_xor(lmax, 2u));\n\
         \t\tfloat m_old = s_m[sg][r_sm];\n\
         \t\tbool valid = (lmax > -1e20f);\n\
         \t\tfloat m_new = valid ? max(m_old, lmax) : m_old;\n\
         \t\tfloat alpha = (valid && m_old > -1e20f) ? exp2(m_old - m_new) : (valid ? 0.0f : 1.0f);\n\
         \t\tfloat4 p0 = valid ? select(float4(0.0f), exp2(sv0 - m_new), sv0 > -1e20f) : float4(0.0f);\n\
         \t\tfloat4 p1 = valid ? select(float4(0.0f), exp2(sv1 - m_new), sv1 > -1e20f) : float4(0.0f);\n\
         \t\t*(threadgroup half4*)(&sP[sg][r_sm * 32u + c_sm]) = half4(p0);\n\
         \t\t*(threadgroup half4*)(&sP[sg][r_sm * 32u + c_sm + 4u]) = half4(p1);\n\
         \t\tfloat4 ps = p0 + p1;\n\
         \t\tfloat lsum = (ps.x + ps.y) + (ps.z + ps.w);\n\
         \t\tlsum += simd_shuffle_xor(lsum, 1u);\n\
         \t\tlsum += simd_shuffle_xor(lsum, 2u);\n\
         \t\tif ((lane & 3u) == 0u) {{\n\
         \t\t\ts_m[sg][r_sm] = m_new;\n\
         \t\t\ts_l[sg][r_sm] = s_l[sg][r_sm] * alpha + lsum;\n\
         \t\t\ts_alpha[sg][r_sm] = alpha;\n\
         \t\t}}\n\
         \t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\t// Rescale AMX accumulator tiles ONLY when row max increased (`simd_any`, keeping `mo` pinned in AMX regs!)\n\
         \t\tfloat al = s_alpha[sg][r_sm];\n\
         \t\tif (kb > 0u && simd_any(al != 1.0f)) {{\n\
         \t\t\t#pragma unroll\n\
         \t\t\tfor (uint d0 = 0; d0 < {d8}u; d0 += 4u) {{\n\
         \t\t\t\t#pragma unroll\n\
         \t\t\t\tfor (uint dd = 0; dd < 4u; dd++) simdgroup_store(mo[d0 + dd], &sS[sg][dd * 8u], 32ul);\n\
         \t\t\t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\t\t\tif (al != 1.0f) {{\n\
         \t\t\t\t\t*(threadgroup float4*)(&sS[sg][r_sm * 32u + c_sm]) *= al;\n\
         \t\t\t\t\t*(threadgroup float4*)(&sS[sg][r_sm * 32u + c_sm + 4u]) *= al;\n\
         \t\t\t\t}}\n\
         \t\t\t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\t\t\t#pragma unroll\n\
         \t\t\t\tfor (uint dd = 0; dd < 4u; dd++) simdgroup_load(mo[d0 + dd], &sS[sg][dd * 8u], 32ul);\n\
         \t\t\t}}\n\
         \t\t}}\n\
         \t\t// Stage C: AMX `O[8xD] += P[8x32] @ V[32xD]`!\n\
         \t\t#pragma unroll\n\
         \t\tfor (uint bk = 0; bk < 32u; bk += 8u) {{\n\
         \t\t\tsimdgroup_half8x8 mp;\n\
         \t\t\tsimdgroup_load(mp, &sP[sg][bk], 32ul);\n\
         \t\t\t#pragma unroll\n\
         \t\t\tfor (uint d = 0; d < {d8}u; d++) {{\n\
         \t\t\t\tsimdgroup_half8x8 mv;\n\
         \t\t\t\tsimdgroup_load(mv, &sV[bk * {d_stride}u + d * 8u], {d_stride}ul);\n\
         \t\t\t\tsimdgroup_multiply_accumulate(mo[d], mp, mv, mo[d]);\n\
         \t\t\t}}\n\
         \t\t}}\n\
         \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t}}\n\
         \t// Final normalize & coalesced 128-bit `half4 x 2` output store!\n\
         \tfloat inv = 1.0f / max(s_l[sg][r_sm], 1e-12f);\n\
         \t#pragma unroll\n\
         \tfor (uint d0 = 0; d0 < {d8}u; d0 += 4u) {{\n\
         \t\t#pragma unroll\n\
         \t\tfor (uint dd = 0; dd < 4u; dd++) simdgroup_store(mo[d0 + dd], &sS[sg][dd * 8u], 32ul);\n\
         \t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\tif (grow < SEQ) {{\n\
         \t\t\tuint ooff = grow * QS + h * D + d0 * 8u + c_sm;\n\
         \t\t\t*(device half4*)(&O[ooff]) = half4(*(threadgroup const float4*)(&sS[sg][r_sm * 32u + c_sm]) * inv);\n\
         \t\t\t*(device half4*)(&O[ooff + 4u]) = half4(*(threadgroup const float4*)(&sS[sg][r_sm * 32u + c_sm + 4u]) * inv);\n\
         \t\t}}\n\
         \t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t}}\n\
         }}\n\n"
    ));
}

/// 6A. Tiled Online-Softmax FlashAttention (`general_fusion = true`, ZERO O(S^2) HBM allocation!).
/// 4 simdgroups (128 threads) cooperatively stream Bc=32 K/V blocks through threadgroup SRAM,
/// maintaining exact online `(m_i, l_i, O_i)` monoid rescaling in registers!
fn emit_msl_flash_attention(
    msl: &mut String,
    kname: &str,
    seq: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    dt: IrDType,
    causal: bool,
) {
    let mt = dt.msl_ty();
    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv * head_dim;
    let grp = (n_heads / n_kv).max(1);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let k_lim = if causal { "min(row + 1u, SEQ)" } else { "SEQ" };

    // 8 simdgroups (256 threads) share each loaded 32x64 K/V SRAM tile across 8 query rows -> 2x-4x less K/V bandwidth!
    msl.push_str(&format!(
        "kernel void {kname}(\n\
         \tdevice const {mt}* Q [[buffer(0)]], device const {mt}* K [[buffer(1)]],\n\
         \tdevice const {mt}* V [[buffer(2)]], device {mt}* O [[buffer(3)]],\n\
         \tuint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
         \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
         ) {{\n\
         \tconst uint SEQ = {seq}u, D = {head_dim}u, QS = {q_stride}u, KVS = {kv_stride}u;\n\
         \tuint h = tgid.y, kv_h = h / {grp}u;\n\
         \tuint row = tgid.x * 8u + sg;\n\
         \tthreadgroup half sK[32][64];\n\
         \tthreadgroup half sV[32][64];\n\
         \tfloat q0 = 0.0f, q1 = 0.0f;\n\
         \tif (row < SEQ) {{\n\
         \t\tuint qoff = row * QS + h * D;\n\
         \t\tq0 = (float)Q[qoff + lane * 2u];\n\
         \t\tq1 = (float)Q[qoff + lane * 2u + 1u];\n\
         \t}}\n\
         \tfloat m_run = -1e30f, l_run = 0.0f, o0 = 0.0f, o1 = 0.0f;\n\
         \tuint max_k_tg = {max_k_tg};\n\
         \tfor (uint kb = 0; kb < max_k_tg; kb += 32u) {{\n\
         \t\t// 256 threads x 8 halfs = 2048 halfs: ONE 128-bit uint4 load per thread for K and V!\n\
         \t\tuint tr = lid >> 3, tc8 = (lid & 7u) * 8u;\n\
         \t\tuint gk = kb + tr;\n\
         \t\tif (gk < SEQ) {{\n\
         \t\t\tuint koff = gk * KVS + kv_h * D + tc8;\n\
         \t\t\t*(threadgroup uint4*)(&sK[tr][tc8]) = *(device const uint4*)(&K[koff]);\n\
         \t\t\t*(threadgroup uint4*)(&sV[tr][tc8]) = *(device const uint4*)(&V[koff]);\n\
         \t\t}} else {{\n\
         \t\t\t*(threadgroup uint4*)(&sK[tr][tc8]) = uint4(0u);\n\
         \t\t\t*(threadgroup uint4*)(&sV[tr][tc8]) = uint4(0u);\n\
         \t\t}}\n\
         \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\tif (row < SEQ) {{\n\
         \t\t\tuint lim = {k_lim};\n\
         \t\t\tuint t_end = min(32u, (lim > kb) ? (lim - kb) : 0u);\n\
         \t\t\tfor (uint t = 0; t < t_end; t++) {{\n\
         \t\t\t\tfloat dot = fma(q0, (float)sK[t][lane * 2u], q1 * (float)sK[t][lane * 2u + 1u]);\n\
         \t\t\t\tfloat sc = simd_sum(dot) * {scale:.8}f;\n\
         \t\t\t\tfloat mn = max(m_run, sc);\n\
         \t\t\t\tfloat al = exp(m_run - mn);\n\
         \t\t\t\tfloat wt = exp(sc - mn);\n\
         \t\t\t\tl_run = l_run * al + wt;\n\
         \t\t\t\tm_run = mn;\n\
         \t\t\t\to0 = fma(o0, al, wt * (float)sV[t][lane * 2u]);\n\
         \t\t\t\to1 = fma(o1, al, wt * (float)sV[t][lane * 2u + 1u]);\n\
         \t\t\t}}\n\
         \t\t}}\n\
         \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t}}\n\
         \tif (row < SEQ) {{\n\
         \t\tfloat inv = 1.0f / max(l_run, 1e-12f);\n\
         \t\tuint ooff = row * QS + h * D;\n\
         \t\tO[ooff + lane * 2u] = ({mt})(o0 * inv);\n\
         \t\tO[ooff + lane * 2u + 1u] = ({mt})(o1 * inv);\n\
         \t}}\n\
         }}\n\n",
        max_k_tg = if causal { "min(tgid.x * 8u + 8u, SEQ)" } else { "SEQ" },
    ));
}

/// 6B. Naive Materialized O(S^2) Attention (`general_fusion = false`) for strict live ablation!
fn emit_msl_naive_attention(
    msl: &mut String,
    k1: &str,
    k2: &str,
    seq: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    dt: IrDType,
    causal: bool,
) {
    let mt = dt.msl_ty();
    let q_stride = n_heads * head_dim;
    let kv_stride = n_kv * head_dim;
    let grp = (n_heads / n_kv).max(1);
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let k_lim = if causal { "row + 1u" } else { "SEQ" };

    msl.push_str(&format!(
        "kernel void {k1}(\n\
         \tdevice const {mt}* Q [[buffer(0)]], device const {mt}* K [[buffer(1)]], device float* S [[buffer(2)]],\n\
         \tuint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
         \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
         ) {{\n\
         \tconst uint SEQ = {seq}u, D = {head_dim}u, QS = {q_stride}u, KVS = {kv_stride}u;\n\
         \tuint row = tgid.x, h = tgid.y, kv_h = h / {grp}u;\n\
         \tuint base = (h * SEQ + row) * SEQ;\n\
         \tuint lim = {k_lim};\n\
         \tthreadgroup float sm[4]; threadgroup float sl[4];\n\
         \tfloat lm = -1e30f;\n\
         \tfor (uint t = lid; t < SEQ; t += 128u) {{\n\
         \t\tfloat sc = -1e30f;\n\
         \t\tif (t < lim) {{\n\
         \t\t\tfloat dot = 0.0f;\n\
         \t\t\tfor (uint d = 0; d < D; d++) dot = fma((float)Q[row * QS + h * D + d], (float)K[t * KVS + kv_h * D + d], dot);\n\
         \t\t\tsc = dot * {scale:.8}f;\n\
         \t\t}}\n\
         \t\tS[base + t] = sc;\n\
         \t\tlm = max(lm, sc);\n\
         \t}}\n\
         \tlm = simd_max(lm); if (lane == 0u) sm[sg] = lm;\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \tfloat gm = max(max(sm[0], sm[1]), max(sm[2], sm[3]));\n\
         \tfloat ll = 0.0f;\n\
         \tfor (uint t = lid; t < lim; t += 128u) {{ float p = exp(S[base + t] - gm); S[base + t] = p; ll += p; }}\n\
         \tfor (uint t = lim + lid; t < SEQ; t += 128u) S[base + t] = 0.0f;\n\
         \tll = simd_sum(ll); if (lane == 0u) sl[sg] = ll;\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \tfloat inv = 1.0f / max(sl[0] + sl[1] + sl[2] + sl[3], 1e-12f);\n\
         \tfor (uint t = lid; t < lim; t += 128u) S[base + t] *= inv;\n\
         }}\n\n\
         kernel void {k2}(\n\
         \tdevice const float* S [[buffer(0)]], device const {mt}* V [[buffer(1)]], device {mt}* O [[buffer(2)]],\n\
         \tuint2 tgid [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]\n\
         ) {{\n\
         \tconst uint SEQ = {seq}u, D = {head_dim}u, QS = {q_stride}u, KVS = {kv_stride}u;\n\
         \tuint row = tgid.x, h = tgid.y, kv_h = h / {grp}u;\n\
         \tif (d >= D) return;\n\
         \tuint base = (h * SEQ + row) * SEQ;\n\
         \tuint lim = {k_lim};\n\
         \tfloat acc = 0.0f;\n\
         \tfor (uint t = 0; t < lim; t++) acc = fma(S[base + t], (float)V[t * KVS + kv_h * D + d], acc);\n\
         \tO[row * QS + h * D + d] = ({mt})acc;\n\
         }}\n\n"
    ));
}

/// 7. Stage 3 — Device-Side Persistent Work-Stealing Block-Sparse AMX GEMM (`ragged_sparse = true`)!
/// Supports BOTH true compressed BSR-32 (`LayoutTag::Bsr32NK` — zero blocks omitted from VRAM & DRAM,
/// works for ANY arbitrary sparsity pattern with 100% coalesced 128-bit `uint4` loads!) and structured fallback!
#[allow(clippy::too_many_arguments)]
fn emit_msl_block_sparse_amx(
    msl: &mut String,
    kname: &str,
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
    density_pct: usize,
    dt: IrDType,
    w_layout: LayoutTag,
    sc: TileSchedule,
) {
    let mt = dt.msl_ty();
    let tiles_m = (m + 31) / 32;
    let tiles_n = (n + 31) / 32;
    let total_tiles = tiles_m * tiles_n;
    let pad = sc.swizzle_pad;
    let sa_st = 32 + pad;
    let sb_st = 32 + pad;

    if w_layout == LayoutTag::Bsr32NK && dt == IrDType::F16 && m % 32 == 0 && n % 32 == 0 && k % 32 == 0 {
        // True Compressed BSR-32 Persistent Work-Stealing AMX: ZERO dense weight reads, works on ANY sparse mask!
        // Uses `BM=64, BN=32` (4 SGs, `tm=4, tn=2`) when `m % 64 == 0` -> 2x compressed BSR weight reuse & 2x fewer BSR DRAM reads!
        let use_m64 = m % 64 == 0;
        let bm_sp = if use_m64 { 64usize } else { 32usize };
        let tm_sp = if use_m64 { 4usize } else { 2usize };
        let sg_m_sp = bm_sp / 2;
        let tiles_m_sp = m / bm_sp;
        let total_tiles_sp = tiles_m_sp * tiles_n;
        let load_a_sp = if use_m64 {
            format!(
                "\t\t\t*(threadgroup uint4*)(&sA[r_ld * {sa_st}u + c8_ld]) = *(device const uint4*)(&a[(m0 + r_ld) * {k}u + kt + c8_ld]);\n\
                 \t\t\t*(threadgroup uint4*)(&sA[(r_ld + 32u) * {sa_st}u + c8_ld]) = *(device const uint4*)(&a[(m0 + r_ld + 32u) * {k}u + kt + c8_ld]);\n"
            )
        } else {
            format!(
                "\t\t\t*(threadgroup uint4*)(&sA[r_ld * {sa_st}u + c8_ld]) = *(device const uint4*)(&a[(m0 + r_ld) * {k}u + kt + c8_ld]);\n"
            )
        };
        msl.push_str(&format!(
            "kernel void {kname}(\n\
             \tdevice const half* a [[buffer(0)]], device const uint* b_bsr [[buffer(1)]],\n\
             \tdevice atomic_uint* work_ctr [[buffer(2)]], device half* out [[buffer(3)]],\n\
             \tuint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
             ) {{\n\
             \tthreadgroup half sA[{bm_sp} * {sa_st}];\n\
             \tthreadgroup half sB[32 * {sb_st}];\n\
             \tthreadgroup float sC[4][128];\n\
             \tthreadgroup uint s_tile_id;\n\
             \tuint kidx_off = b_bsr[0], data_off_h = b_bsr[1];\n\
             \tconst device uint* row_ptr = b_bsr + 4u;\n\
             \tconst device uint* k_idx = b_bsr + kidx_off;\n\
             \tconst device half* data_b = ((const device half*)b_bsr) + data_off_h;\n\
             \tuint sg_r = (sg >> 1) * {sg_m_sp}u, sg_c = (sg & 1u) * 16u;\n\
             \tuint r_ld = lid >> 2, c8_ld = (lid & 3u) * 8u;\n\
             \tuint er = lane >> 2, ec4 = (lane & 3u) * 4u;\n\
             \twhile (true) {{\n\
             \t\tif (lid == 0u) s_tile_id = atomic_fetch_add_explicit(work_ctr, 1u, memory_order_relaxed);\n\
             \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
             \t\tuint tid = s_tile_id;\n\
             \t\tif (tid >= {total_tiles_sp}u) break;\n\
             \t\tuint tm = tid / {tiles_n}u, tn = tid % {tiles_n}u;\n\
             \t\tuint m0 = tm * {bm_sp}u, n0 = tn * 32u;\n\
             \t\tsimdgroup_float8x8 mc[{tm_sp}][2];\n\
             \t\t#pragma unroll\n\
             \t\tfor (uint r = 0; r < {tm_sp}u; r++) for (uint c = 0; c < 2u; c++) mc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n\
             \t\tuint p_start = row_ptr[tn], p_end = row_ptr[tn + 1u];\n\
             \t\tfor (uint p = p_start; p < p_end; p++) {{\n\
             \t\t\tuint kt = k_idx[p];\n\
             \t\t\tconst device half* blk_b = data_b + (ulong)p * 1024ul;\n\
             \t\t\t// 128-bit coalesced uint4 loads: {bm_sp}x32 A reused against ONE compressed 32x32 BSR block!\n\
             {load_a_sp}\t\t\t*(threadgroup uint4*)(&sB[r_ld * {sb_st}u + c8_ld]) = *(device const uint4*)(&blk_b[r_ld * 32u + c8_ld]);\n\
             \t\t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
             \t\t\t#pragma unroll\n\
             \t\t\tfor (uint kk = 0; kk < 32u; kk += 8u) {{\n\
             \t\t\t\tsimdgroup_half8x8 ma[{tm_sp}], mb[2];\n\
             \t\t\t\tfor (uint r = 0; r < {tm_sp}u; r++) simdgroup_load(ma[r], &sA[(sg_r + r * 8u) * {sa_st}u + kk], {sa_st}ul);\n\
             \t\t\t\tfor (uint c = 0; c < 2u; c++) simdgroup_load(mb[c], &sB[(sg_c + c * 8u) * {sb_st}u + kk], {sb_st}ul, ulong2(0, 0), true);\n\
             \t\t\t\tfor (uint r = 0; r < {tm_sp}u; r++) for (uint c = 0; c < 2u; c++) simdgroup_multiply_accumulate(mc[r][c], ma[r], mb[c], mc[r][c]);\n\
             \t\t\t}}\n\
             \t\t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
             \t\t}}\n\
             \t\t// Coalesced half4 output store (2 tiles per barrier!)\n\
             \t\t#pragma unroll\n\
             \t\tfor (uint r = 0; r < {tm_sp}u; r++) {{\n\
             \t\t\tsimdgroup_store(mc[r][0], &sC[sg][0], 16ul);\n\
             \t\t\tsimdgroup_store(mc[r][1], &sC[sg][8], 16ul);\n\
             \t\t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
             \t\t\tuint gr = m0 + sg_r + r * 8u + er, gc = n0 + sg_c + ec4;\n\
             \t\t\t*(device half4*)(&out[gr * {n}u + gc]) = half4(*(threadgroup const float4*)(&sC[sg][er * 16u + ec4]));\n\
             \t\t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
             \t\t}}\n\
             \t}}\n\
             }}\n\n"
        ));
        return;
    }

    let k_blocks = (k + block_size - 1) / block_size;
    let stride_mod = (100 / density_pct.max(1)).max(1);
    let b_nk = w_layout == LayoutTag::ColMajorNK;

    let load_tile = if b_nk && dt == IrDType::F16 && k % 8 == 0 && m % 32 == 0 && n % 32 == 0 {
        format!(
            "\t\t\t// 128-bit uint4 coalesced loads: 128 threads x 8 halfs = 1024 halfs (1 instruction per thread!)\n\
             \t\t\tuint r = lid >> 2, c8 = (lid & 3u) * 8u;\n\
             \t\t\t*(threadgroup uint4*)(&sA[r * {sa_st}u + c8]) = *(device const uint4*)(&a[(m0 + r) * {k}u + kt + c8]);\n\
             \t\t\t*(threadgroup uint4*)(&sB[r * {sb_st}u + c8]) = *(device const uint4*)(&b[(n0 + r) * {k}u + kt + c8]);\n"
        )
    } else {
        let lb = if b_nk { format!("(half)b[gn * {k}u + gk2]") } else { format!("(half)b[gk2 * {n}u + gn]") };
        format!(
            "\t\t\tfor (uint idx = lid; idx < 1024u; idx += 128u) {{\n\
             \t\t\t\tuint r = idx >> 5, c = idx & 31u;\n\
             \t\t\t\tuint gr = m0 + r, gk = kt + c;\n\
             \t\t\t\tsA[r * {sa_st}u + c] = (gr < {m}u && gk < {k}u) ? (half)a[gr * {k}u + gk] : half(0.0h);\n\
             \t\t\t\tuint gn = n0 + c, gk2 = kt + r;\n\
             \t\t\t\tsB[r * {sb_st}u + c] = (gk2 < {k}u && gn < {n}u) ? {lb} : half(0.0h);\n\
             \t\t\t}}\n"
        )
    };
    let load_mb = if b_nk && dt == IrDType::F16 && k % 8 == 0 && m % 32 == 0 && n % 32 == 0 {
        format!("simdgroup_load(mb[c], &sB[(sg_c + c * 8u) * {sb_st}u + kk], {sb_st}ul, ulong2(0, 0), true);")
    } else {
        format!("simdgroup_load(mb[c], &sB[kk * {sb_st}u + sg_c + c * 8u], {sb_st}ul);")
    };

    msl.push_str(&format!(
        "kernel void {kname}(\n\
         \tdevice const {mt}* a [[buffer(0)]], device const {mt}* b [[buffer(1)]],\n\
         \tdevice atomic_uint* work_ctr [[buffer(2)]], device {mt}* out [[buffer(3)]],\n\
         \tuint lid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
         ) {{\n\
         \tthreadgroup half sA[32 * {sa_st}];\n\
         \tthreadgroup half sB[32 * {sb_st}];\n\
         \tthreadgroup float sC[4][64];\n\
         \tthreadgroup uint s_tile_id;\n\
         \tuint sg_r = (sg >> 1) * 16u, sg_c = (sg & 1u) * 16u;\n\
         \twhile (true) {{\n\
         \t\tif (lid == 0u) s_tile_id = atomic_fetch_add_explicit(work_ctr, 1u, memory_order_relaxed);\n\
         \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\tuint tid = s_tile_id;\n\
         \t\tif (tid >= {total_tiles}u) break;\n\
         \t\tuint tm = tid / {tiles_n}u, tn = tid % {tiles_n}u;\n\
         \t\tuint m0 = tm * 32u, n0 = tn * 32u;\n\
         \t\tsimdgroup_float8x8 mc[2][2];\n\
         \t\t#pragma unroll\n\
         \t\tfor (uint r = 0; r < 2u; r++) for (uint c = 0; c < 2u; c++) mc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n\
         \t\t// Skip zero structured blocks directly on GPU (exact match to sparse weight mask!)\n\
         \t\tfor (uint kb = 0; kb < {k_blocks}u; kb++) {{\n\
         \t\t\tif (((kb + tn) % {stride_mod}u) != 0u) continue;\n\
         \t\t\tuint kt = kb * 32u;\n\
         {load_tile}\t\t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\t\t#pragma unroll\n\
         \t\t\tfor (uint kk = 0; kk < 32u; kk += 8u) {{\n\
         \t\t\t\tsimdgroup_half8x8 ma[2], mb[2];\n\
         \t\t\t\tfor (uint r = 0; r < 2u; r++) simdgroup_load(ma[r], &sA[(sg_r + r * 8u) * {sa_st}u + kk], {sa_st}ul);\n\
         \t\t\t\tfor (uint c = 0; c < 2u; c++) {load_mb}\n\
         \t\t\t\tfor (uint r = 0; r < 2u; r++) for (uint c = 0; c < 2u; c++) simdgroup_multiply_accumulate(mc[r][c], ma[r], mb[c], mc[r][c]);\n\
         \t\t\t}}\n\
         \t\t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\t}}\n\
         \t\tfor (uint r = 0; r < 2u; r++) for (uint c = 0; c < 2u; c++) {{\n\
         \t\t\tsimdgroup_store(mc[r][c], &sC[sg][0], 8ul);\n\
         \t\t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\t\tfor (uint e = lane; e < 64u; e += 32u) {{\n\
         \t\t\t\tuint gr = m0 + sg_r + r * 8u + (e >> 3), gc = n0 + sg_c + c * 8u + (e & 7u);\n\
         \t\t\t\tif (gr < {m}u && gc < {n}u) out[gr * {n}u + gc] = ({mt})sC[sg][e];\n\
         \t\t\t}}\n\
         \t\t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\t}}\n\
         \t}}\n\
         }}\n\n"
    ));
}

// =========================================================================
// Stage 4 Universal Compiler Facility Kernels (#2, #3, #5, #6, #7, #8A, #8B)
// =========================================================================

/// Facility #2 (Unpacked Baseline): AMX GEMM writing to strided column slice `[col_off .. col_off + n_slice]` of `[m, total_n]`.
#[allow(clippy::too_many_arguments)]
fn emit_msl_amx_gemm_col_slice(
    msl: &mut String,
    kname: &str,
    m: usize,
    k: usize,
    n_slice: usize,
    total_n: usize,
    col_off: usize,
    dt: IrDType,
    sc: TileSchedule,
) {
    let mt = dt.msl_ty();
    let bm = 64usize;
    let bn = 64usize;
    let bk = 16usize;
    let sa_st = bk + sc.swizzle_pad;
    let sb_st = bk + sc.swizzle_pad;
    msl.push_str(&format!(
        "kernel void {kname}(\n\
         \tdevice const {mt}* a [[buffer(0)]], device const {mt}* b [[buffer(1)]], device {mt}* out [[buffer(2)]],\n\
         \tuint3 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
         \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
         ) {{\n\
         \tthreadgroup half sA[64 * {sa_st}];\n\
         \tthreadgroup half sB[64 * {sb_st}];\n\
         \tthreadgroup float sC[4][128];\n\
         \tuint m0 = tgid.y * {bm}u, n0 = tgid.x * {bn}u;\n\
         \tuint sg_r = (sg >> 1) * 32u, sg_c = (sg & 1u) * 32u;\n\
         \tsimdgroup_float8x8 mc[4][4];\n\
         \t#pragma unroll\n\
         \tfor (uint r = 0; r < 4u; r++) for (uint c = 0; c < 4u; c++) mc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n\
         \tfor (uint kt = 0; kt < {k}u; kt += 16u) {{\n\
         \t\tuint r = lid >> 1, c8 = (lid & 1u) * 8u;\n\
         \t\t*(threadgroup uint4*)(&sA[r * {sa_st}u + c8]) = *(device const uint4*)(&a[(m0 + r) * {k}u + kt + c8]);\n\
         \t\t*(threadgroup uint4*)(&sB[r * {sb_st}u + c8]) = *(device const uint4*)(&b[(n0 + r) * {k}u + kt + c8]);\n\
         \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\t#pragma unroll\n\
         \t\tfor (uint kk = 0; kk < 16u; kk += 8u) {{\n\
         \t\t\tsimdgroup_half8x8 ma[4], mb[4];\n\
         \t\t\tfor (uint rr = 0; rr < 4u; rr++) simdgroup_load(ma[rr], &sA[(sg_r + rr * 8u) * {sa_st}u + kk], {sa_st}ul);\n\
         \t\t\tfor (uint cc = 0; cc < 4u; cc++) simdgroup_load(mb[cc], &sB[(sg_c + cc * 8u) * {sb_st}u + kk], {sb_st}ul, ulong2(0, 0), true);\n\
         \t\t\tfor (uint rr = 0; rr < 4u; rr++) for (uint cc = 0; cc < 4u; cc++) simdgroup_multiply_accumulate(mc[rr][cc], ma[rr], mb[cc], mc[rr][cc]);\n\
         \t\t}}\n\
         \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t}}\n\
         \tuint er = lane >> 2, ec4 = (lane & 3u) * 4u;\n\
         \tfor (uint r = 0; r < 4u; r++) for (uint c = 0; c < 4u; c += 2u) {{\n\
         \t\tsimdgroup_store(mc[r][c], &sC[sg][0], 16ul);\n\
         \t\tsimdgroup_store(mc[r][c + 1u], &sC[sg][8], 16ul);\n\
         \t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\tuint gr = m0 + sg_r + r * 8u + er, gc = n0 + sg_c + c * 8u + ec4;\n\
         \t\tif (gr < {m}u && gc < {n_slice}u) *(device half4*)(&out[gr * {total_n}u + {col_off}u + gc]) = half4(*(threadgroup const float4*)(&sC[sg][er * 16u + ec4]));\n\
         \t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t}}\n\
         }}\n\n"
    ));
}

/// Facility #10 (`auto_pad`): Zero-Branch 128-bit Double-Buffered 64x64 AMX GEMM for Odd Logical Dimensions!
/// Operates on load-time zero-padded `[m, padded_k] @ [padded_n, padded_k]^T` (all multiples of 64!)
/// and masks ONLY the final output write (`gr < m && gc < logical_n`)!
#[allow(clippy::too_many_arguments)]
fn emit_msl_amx_gemm_padded(
    msl: &mut String,
    kname: &str,
    m: usize,
    padded_k: usize,
    padded_n: usize,
    logical_n: usize,
    dt: IrDType,
    _sc: TileSchedule,
) {
    let mt = dt.msl_ty();
    let sa_st = 24usize;
    let sb_st = 24usize;
    msl.push_str(&format!(
        "kernel void {kname}(\n\
         \tdevice const {mt}* a [[buffer(0)]], device const {mt}* b [[buffer(1)]], device {mt}* out [[buffer(2)]],\n\
         \tuint3 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
         \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
         ) {{\n\
         \t// Facility #10 (`auto_pad`): Branchless 128-bit uint4 double-buffered 64x64 AMX (`K_pad={padded_k}, N_pad={padded_n} -> logical N={logical_n}`)\n\
         \tthreadgroup half sA[2][64 * {sa_st}];\n\
         \tthreadgroup half sB[2][64 * {sb_st}];\n\
         \tthreadgroup float sC[4][64];\n\
         \tuint m0 = tgid.y * 64u, n0 = tgid.x * 64u;\n\
         \tuint sg_r = (sg >> 1) * 32u, sg_c = (sg & 1u) * 32u;\n\
         \tsimdgroup_float8x8 mc[4][4];\n\
         \t#pragma unroll\n\
         \tfor (uint r = 0; r < 4u; r++) for (uint c = 0; c < 4u; c++) mc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n\
         \tuint r_ld = lid >> 1, c8_ld = (lid & 1u) * 8u;\n\
         \t*(threadgroup uint4*)(&sA[0][r_ld * {sa_st}u + c8_ld]) = (m0 + r_ld < {m}u) ? *(device const uint4*)(&a[(m0 + r_ld) * {padded_k}u + c8_ld]) : uint4(0u);\n\
         \t*(threadgroup uint4*)(&sB[0][r_ld * {sb_st}u + c8_ld]) = *(device const uint4*)(&b[(n0 + r_ld) * {padded_k}u + c8_ld]);\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \tuint buf = 0u;\n\
         \tfor (uint kt = 0; kt < {padded_k}u; kt += 16u) {{\n\
         \t\tuint nkt = kt + 16u, nbuf = buf ^ 1u;\n\
         \t\tif (nkt < {padded_k}u) {{\n\
         \t\t\t*(threadgroup uint4*)(&sA[nbuf][r_ld * {sa_st}u + c8_ld]) = (m0 + r_ld < {m}u) ? *(device const uint4*)(&a[(m0 + r_ld) * {padded_k}u + nkt + c8_ld]) : uint4(0u);\n\
         \t\t\t*(threadgroup uint4*)(&sB[nbuf][r_ld * {sb_st}u + c8_ld]) = *(device const uint4*)(&b[(n0 + r_ld) * {padded_k}u + nkt + c8_ld]);\n\
         \t\t}}\n\
         \t\t#pragma unroll\n\
         \t\tfor (uint kk = 0; kk < 16u; kk += 8u) {{\n\
         \t\t\tsimdgroup_half8x8 ma[4], mb[4];\n\
         \t\t\tfor (uint rr = 0; rr < 4u; rr++) simdgroup_load(ma[rr], &sA[buf][(sg_r + rr * 8u) * {sa_st}u + kk], {sa_st}ul);\n\
         \t\t\tfor (uint cc = 0; cc < 4u; cc++) simdgroup_load(mb[cc], &sB[buf][(sg_c + cc * 8u) * {sb_st}u + kk], {sb_st}ul, ulong2(0, 0), true);\n\
         \t\t\tfor (uint rr = 0; rr < 4u; rr++) for (uint cc = 0; cc < 4u; cc++) simdgroup_multiply_accumulate(mc[rr][cc], ma[rr], mb[cc], mc[rr][cc]);\n\
         \t\t}}\n\
         \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\tbuf = nbuf;\n\
         \t}}\n\
         \tfor (uint r = 0; r < 4u; r++) for (uint c = 0; c < 4u; c++) {{\n\
         \t\tsimdgroup_store(mc[r][c], &sC[sg][0], 8ul);\n\
         \t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\tfor (uint e = lane; e < 64u; e += 32u) {{\n\
         \t\t\tuint gr = m0 + sg_r + r * 8u + (e >> 3), gc = n0 + sg_c + c * 8u + (e & 7u);\n\
         \t\t\tif (gr < {m}u && gc < {logical_n}u) out[gr * {logical_n}u + gc] = ({mt})sC[sg][e];\n\
         \t\t}}\n\
         \t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t}}\n\
         }}\n\n"
    ));
}

/// Facility #3/#11/#12/#13: `llm.c` `x128` 128-bit Vectorized Fused AdamW:
///   - `in_place`: Zero-Copy Buffer Donation (`0 B` extra VRAM)
///   - `stochastic_round`: Stateless Counter-Based Philox BF16/FP16 Stochastic Rounding (master-weight-free training!)
///   - `streaming_burst`: 64B full-cacheline 4x128-bit streaming burst (ILP=4)!
fn emit_msl_adamw(
    msl: &mut String,
    kname: &str,
    numel: usize,
    dt: IrDType,
    in_place: bool,
    stochastic_round: bool,
    streaming_burst: bool,
) {
    let n4 = numel / 4;
    let is_f32 = dt == IrDType::F32;
    let is_bf16 = dt == IrDType::BF16;
    let vec_ty = if is_f32 { "float4" } else if is_bf16 { "ushort4" } else { "half4" };

    if stochastic_round && !is_f32 {
        // Facility #11: Stateless Counter-Based Philox/Murmur Hash + Exact Unbiased Low-Bit Stochastic Rounding!
        let sr_helper = format!(
            "static inline uint {kname}_philox(uint seed, uint idx) {{\n\
             \tuint h = (idx * 0x9E3779B1u) ^ (seed * 0x85EBCA77u + 0xC2B2AE3Du);\n\
             \th ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;\n\
             \treturn h;\n\
             }}\n\
             static inline half {kname}_sr_f16(float x, uint rng) {{\n\
             \tuint bits = as_type<uint>(x);\n\
             \tuint exp_b = (bits >> 23) & 0xFFu;\n\
             \tif (exp_b >= 113u && exp_b <= 142u) {{\n\
             \t\tbits = (bits + (rng & 0x1FFFu)) & ~0x1FFFu;\n\
             \t\treturn half(as_type<float>(bits));\n\
             \t}}\n\
             \treturn half(x);\n\
             }}\n\
             static inline half4 {kname}_sr_f16x4(float4 v, uint seed, uint gid) {{\n\
             \tuint r0 = {kname}_philox(seed, gid * 2u);\n\
             \tuint r1 = {kname}_philox(seed, gid * 2u + 1u);\n\
             \treturn half4({kname}_sr_f16(v.x, r0), {kname}_sr_f16(v.y, r0 >> 16), {kname}_sr_f16(v.z, r1), {kname}_sr_f16(v.w, r1 >> 16));\n\
             }}\n"
        );
        msl.push_str(&sr_helper);
        if in_place {
            msl.push_str(&format!(
                "kernel void {kname}(\n\
                 \tdevice half4* param [[buffer(0)]], device const half4* grad [[buffer(1)]],\n\
                 \tdevice half4* m_buf [[buffer(2)]], device half4* v_buf [[buffer(3)]],\n\
                 \tconstant uint& sr_seed [[buffer(4)]],\n\
                 \tuint gid [[thread_position_in_grid]]\n\
                 ) {{\n\
                 \tif (gid >= {n4}u) return;\n\
                 \tfloat4 p = float4(param[gid]);\n\
                 \tfloat4 g = float4(grad[gid]);\n\
                 \tfloat4 m = float4(m_buf[gid]);\n\
                 \tfloat4 v = float4(v_buf[gid]);\n\
                 \tm = fma(float4(0.9f), m, 0.1f * g);\n\
                 \tv = fma(float4(0.999f), v, 0.001f * g * g);\n\
                 \tfloat4 m_hat = m * 10.0f;\n\
                 \tfloat4 v_hat = v * 1000.0f;\n\
                 \tp = p * (1.0f - 1e-5f) - 1e-3f * m_hat / (sqrt(v_hat) + 1e-8f);\n\
                 \tm_buf[gid] = {kname}_sr_f16x4(m, sr_seed ^ 0x11111111u, gid);\n\
                 \tv_buf[gid] = {kname}_sr_f16x4(v, sr_seed ^ 0x22222222u, gid);\n\
                 \tparam[gid] = {kname}_sr_f16x4(p, sr_seed, gid);\n\
                 }}\n\n"
            ));
        } else {
            msl.push_str(&format!(
                "kernel void {kname}(\n\
                 \tdevice const half4* param [[buffer(0)]], device const half4* grad [[buffer(1)]],\n\
                 \tdevice const half4* m_buf [[buffer(2)]], device const half4* v_buf [[buffer(3)]],\n\
                 \tdevice half4* out_p [[buffer(4)]], device half4* out_m [[buffer(5)]], device half4* out_v [[buffer(6)]],\n\
                 \tconstant uint& sr_seed [[buffer(7)]],\n\
                 \tuint gid [[thread_position_in_grid]]\n\
                 ) {{\n\
                 \tif (gid >= {n4}u) return;\n\
                 \tfloat4 p = float4(param[gid]);\n\
                 \tfloat4 g = float4(grad[gid]);\n\
                 \tfloat4 m = float4(m_buf[gid]);\n\
                 \tfloat4 v = float4(v_buf[gid]);\n\
                 \tm = fma(float4(0.9f), m, 0.1f * g);\n\
                 \tv = fma(float4(0.999f), v, 0.001f * g * g);\n\
                 \tfloat4 m_hat = m * 10.0f;\n\
                 \tfloat4 v_hat = v * 1000.0f;\n\
                 \tp = p * (1.0f - 1e-5f) - 1e-3f * m_hat / (sqrt(v_hat) + 1e-8f);\n\
                 \tout_m[gid] = {kname}_sr_f16x4(m, sr_seed ^ 0x11111111u, gid);\n\
                 \tout_v[gid] = {kname}_sr_f16x4(v, sr_seed ^ 0x22222222u, gid);\n\
                 \tout_p[gid] = {kname}_sr_f16x4(p, sr_seed, gid);\n\
                 }}\n\n"
            ));
        }
        return;
    }

    if streaming_burst && numel % 16 == 0 && in_place {
        let n16 = numel / 16;
        let vt = if is_f32 { "float4" } else { "half4" };
        msl.push_str(&format!(
            "kernel void {kname}(\n\
             \tdevice {vt}* param [[buffer(0)]], device const {vt}* grad [[buffer(1)]],\n\
             \tdevice {vt}* m_buf [[buffer(2)]], device {vt}* v_buf [[buffer(3)]],\n\
             \tuint gid [[thread_position_in_grid]]\n\
             ) {{\n\
             \tif (gid >= {n16}u) return;\n\
             \tulong base = (ulong)gid * 4ul;\n\
             \t// Facility #12: 64B full-cacheline 4x128-bit streaming burst AdamW (ILP=4!)\n\
             \t#pragma unroll\n\
             \tfor (uint u = 0; u < 4u; u++) {{\n\
             \t\tulong i = base + u;\n\
             \t\tfloat4 p = float4(param[i]), g = float4(grad[i]), m = float4(m_buf[i]), v = float4(v_buf[i]);\n\
             \t\tm = fma(float4(0.9f), m, 0.1f * g);\n\
             \t\tv = fma(float4(0.999f), v, 0.001f * g * g);\n\
             \t\tp = p * (1.0f - 1e-5f) - 1e-3f * (m * 10.0f) / (sqrt(v * 1000.0f) + 1e-8f);\n\
             \t\tm_buf[i] = {vt}(m); v_buf[i] = {vt}(v); param[i] = {vt}(p);\n\
             \t}}\n\
             }}\n\n"
        ));
        return;
    }

    if in_place {
        msl.push_str(&format!(
            "kernel void {kname}(\n\
             \tdevice {vec_ty}* param [[buffer(0)]], device const {vec_ty}* grad [[buffer(1)]],\n\
             \tdevice {vec_ty}* m_buf [[buffer(2)]], device {vec_ty}* v_buf [[buffer(3)]],\n\
             \tuint gid [[thread_position_in_grid]]\n\
             ) {{\n\
             \tif (gid >= {n4}u) return;\n\
             \tfloat4 p = float4(param[gid]);\n\
             \tfloat4 g = float4(grad[gid]);\n\
             \tfloat4 m = float4(m_buf[gid]);\n\
             \tfloat4 v = float4(v_buf[gid]);\n\
             \tm = fma(float4(0.9f), m, 0.1f * g);\n\
             \tv = fma(float4(0.999f), v, 0.001f * g * g);\n\
             \tfloat4 m_hat = m * 10.0f;\n\
             \tfloat4 v_hat = v * 1000.0f;\n\
             \tp = p * (1.0f - 1e-5f) - 1e-3f * m_hat / (sqrt(v_hat) + 1e-8f);\n\
             \tm_buf[gid] = {vec_ty}(m);\n\
             \tv_buf[gid] = {vec_ty}(v);\n\
             \tparam[gid] = {vec_ty}(p);\n\
             }}\n\n"
        ));
    } else {
        msl.push_str(&format!(
            "kernel void {kname}(\n\
             \tdevice const {vec_ty}* param [[buffer(0)]], device const {vec_ty}* grad [[buffer(1)]],\n\
             \tdevice const {vec_ty}* m_buf [[buffer(2)]], device const {vec_ty}* v_buf [[buffer(3)]],\n\
             \tdevice {vec_ty}* out_p [[buffer(4)]], device {vec_ty}* out_m [[buffer(5)]], device {vec_ty}* out_v [[buffer(6)]],\n\
             \tuint gid [[thread_position_in_grid]]\n\
             ) {{\n\
             \tif (gid >= {n4}u) return;\n\
             \tfloat4 p = float4(param[gid]);\n\
             \tfloat4 g = float4(grad[gid]);\n\
             \tfloat4 m = float4(m_buf[gid]);\n\
             \tfloat4 v = float4(v_buf[gid]);\n\
             \tm = fma(float4(0.9f), m, 0.1f * g);\n\
             \tv = fma(float4(0.999f), v, 0.001f * g * g);\n\
             \tfloat4 m_hat = m * 10.0f;\n\
             \tfloat4 v_hat = v * 1000.0f;\n\
             \tp = p * (1.0f - 1e-5f) - 1e-3f * m_hat / (sqrt(v_hat) + 1e-8f);\n\
             \tout_m[gid] = {vec_ty}(m);\n\
             \tout_v[gid] = {vec_ty}(v);\n\
             \tout_p[gid] = {vec_ty}(p);\n\
             }}\n\n"
        ));
    }
}

/// Facility #5/#10/#12: Online Monoid Fused Cross-Entropy + Auto-Padded Odd Vocab (`50257 -> 50304`, ZERO O(BT*V) HBM!).
fn emit_msl_online_cross_entropy(
    msl: &mut String,
    kname: &str,
    rows: usize,
    vocab: usize,
    padded_vocab: usize,
    dt: IrDType,
    _streaming_cache: bool,
) {
    let inv_rows = 1.0f32 / (rows as f32);
    let vec_ty = if dt == IrDType::F32 { "float4" } else { "half4" };
    let sc_ty = dt.msl_ty();

    if padded_vocab == vocab && vocab % 4 != 0 {
        // Unpadded odd vocab (`auto_pad = false`, e.g. raw GPT-2 `50257`): scalar loop required because rows are unaligned!
        msl.push_str(&format!(
            "kernel void {kname}(\n\
             \tdevice const {sc_ty}* logits [[buffer(0)]], device const {sc_ty}* targets [[buffer(1)]],\n\
             \tdevice {sc_ty}* dlogits [[buffer(2)]],\n\
             \tuint row [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
             \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
             ) {{\n\
             \tif (row >= {rows}u) return;\n\
             \tthreadgroup float s_m[8]; threadgroup float s_l[8];\n\
             \tulong base = (ulong)row * {vocab}ul;\n\
             \tfloat lm = -1e30f, ll = 0.0f;\n\
             \tfor (uint c = lid; c < {vocab}u; c += 256u) {{\n\
             \t\tfloat v = (float)logits[base + c];\n\
             \t\tfloat nm = max(lm, v);\n\
             \t\tll = ll * exp(lm - nm) + exp(v - nm);\n\
             \t\tlm = nm;\n\
             \t}}\n\
             \tfloat sg_m = simd_max(lm);\n\
             \tfloat sg_l = simd_sum(ll * exp(lm - sg_m));\n\
             \tif (lane == 0u) {{ s_m[sg] = sg_m; s_l[sg] = sg_l; }}\n\
             \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
             \tfloat gm = -1e30f; for (uint s = 0; s < 8u; s++) gm = max(gm, s_m[s]);\n\
             \tfloat gl = 0.0f; for (uint s = 0; s < 8u; s++) gl += s_l[s] * exp(s_m[s] - gm);\n\
             \tfloat inv_l = 1.0f / max(gl, 1e-12f);\n\
             \tuint tgt = (uint)((float)targets[row]);\n\
             \tfor (uint c = lid; c < {vocab}u; c += 256u) {{\n\
             \t\tfloat p = exp((float)logits[base + c] - gm) * inv_l;\n\
             \t\tif (c == tgt) p -= 1.0f;\n\
             \t\tdlogits[base + c] = ({sc_ty})(p * {inv_rows:.10}f);\n\
             \t}}\n\
             }}\n\n"
        ));
        return;
    }

    if padded_vocab > vocab {
        // Facility #10 (`auto_pad`, e.g. GPT-2 `50257 -> 50304`): 100% 128-bit `float4` coalesced loads with exact tail masking!
        let pv4 = padded_vocab / 4;
        let full4 = vocab / 4;
        msl.push_str(&format!(
            "kernel void {kname}(\n\
             \tdevice const {vec_ty}* logits_pad [[buffer(0)]], device const {sc_ty}* targets [[buffer(1)]],\n\
             \tdevice {sc_ty}* dlogits [[buffer(2)]],\n\
             \tuint row [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
             \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
             ) {{\n\
             \tif (row >= {rows}u) return;\n\
             \tthreadgroup float s_m[8]; threadgroup float s_l[8];\n\
             \tulong pbase4 = (ulong)row * {pv4}ul;\n\
             \tulong obase = (ulong)row * {vocab}ul;\n\
             \tfloat lm = -1e30f, ll = 0.0f;\n\
             \tfor (uint c4 = lid; c4 < {full4}u; c4 += 256u) {{\n\
             \t\tfloat4 v = float4(logits_pad[pbase4 + c4]);\n\
             \t\tfloat bm = max(max(v.x, v.y), max(v.z, v.w));\n\
             \t\tfloat nm = max(lm, bm);\n\
             \t\tll = ll * exp(lm - nm) + exp(v.x - nm) + exp(v.y - nm) + exp(v.z - nm) + exp(v.w - nm);\n\
             \t\tlm = nm;\n\
             \t}}\n\
             \tif (lid == 0u) {{\n\
             \t\tfloat4 vt = float4(logits_pad[pbase4 + {full4}u]);\n\
             \t\tuint bcol = {full4}u * 4u;\n\
             \t\tfloat arr[4] = {{ vt.x, vt.y, vt.z, vt.w }};\n\
             \t\tfor (uint j = 0; j < 4u && bcol + j < {vocab}u; j++) {{\n\
             \t\t\tfloat nm = max(lm, arr[j]);\n\
             \t\t\tll = ll * exp(lm - nm) + exp(arr[j] - nm);\n\
             \t\t\tlm = nm;\n\
             \t\t}}\n\
             \t}}\n\
             \tfloat sg_m = simd_max(lm);\n\
             \tfloat sg_l = simd_sum(ll * exp(lm - sg_m));\n\
             \tif (lane == 0u) {{ s_m[sg] = sg_m; s_l[sg] = sg_l; }}\n\
             \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
             \tfloat gm = -1e30f; for (uint s = 0; s < 8u; s++) gm = max(gm, s_m[s]);\n\
             \tfloat gl = 0.0f; for (uint s = 0; s < 8u; s++) gl += s_l[s] * exp(s_m[s] - gm);\n\
             \tfloat inv_l = 1.0f / max(gl, 1e-12f);\n\
             \tuint tgt = (uint)((float)targets[row]);\n\
             \tfor (uint c4 = lid; c4 < {full4}u; c4 += 256u) {{\n\
             \t\tfloat4 p = exp(float4(logits_pad[pbase4 + c4]) - gm) * inv_l;\n\
             \t\tuint bcol = c4 * 4u;\n\
             \t\tif (tgt >= bcol && tgt < bcol + 4u) {{\n\
             \t\t\tuint r = tgt - bcol;\n\
             \t\t\tif (r == 0u) p.x -= 1.0f; else if (r == 1u) p.y -= 1.0f; else if (r == 2u) p.z -= 1.0f; else p.w -= 1.0f;\n\
             \t\t}}\n\
             \t\tp *= {inv_rows:.10}f;\n\
             \t\tdlogits[obase + bcol + 0u] = ({sc_ty})p.x; dlogits[obase + bcol + 1u] = ({sc_ty})p.y;\n\
             \t\tdlogits[obase + bcol + 2u] = ({sc_ty})p.z; dlogits[obase + bcol + 3u] = ({sc_ty})p.w;\n\
             \t}}\n\
             \tif (lid == 0u) {{\n\
             \t\tfloat4 vt = float4(logits_pad[pbase4 + {full4}u]);\n\
             \t\tuint bcol = {full4}u * 4u;\n\
             \t\tfloat arr[4] = {{ vt.x, vt.y, vt.z, vt.w }};\n\
             \t\tfor (uint j = 0; j < 4u && bcol + j < {vocab}u; j++) {{\n\
             \t\t\tfloat p = exp(arr[j] - gm) * inv_l;\n\
             \t\t\tif (bcol + j == tgt) p -= 1.0f;\n\
             \t\t\tdlogits[obase + bcol + j] = ({sc_ty})(p * {inv_rows:.10}f);\n\
             \t\t}}\n\
             \t}}\n\
             }}\n\n"
        ));
        return;
    }

    let v4 = vocab / 4;
    msl.push_str(&format!(
        "kernel void {kname}(\n\
         \tdevice const {vec_ty}* logits [[buffer(0)]], device const {sc_ty}* targets [[buffer(1)]],\n\
         \tdevice {vec_ty}* dlogits [[buffer(2)]],\n\
         \tuint row [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
         \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
         ) {{\n\
         \tif (row >= {rows}u) return;\n\
         \tthreadgroup float s_m[8];\n\
         \tthreadgroup float s_l[8];\n\
         \tulong base4 = (ulong)row * {v4}ul;\n\
         \t// Pass 1: Single-pass 128-bit Online Monoid (m, l) — ZERO O(BT*V) HBM probs materialization!\n\
         \tfloat lm = -1e30f, ll = 0.0f;\n\
         \tfor (uint c4 = lid; c4 < {v4}u; c4 += 256u) {{\n\
         \t\tfloat4 v = float4(logits[base4 + c4]);\n\
         \t\tfloat bm = max(max(v.x, v.y), max(v.z, v.w));\n\
         \t\tfloat nm = max(lm, bm);\n\
         \t\tll = ll * exp(lm - nm) + exp(v.x - nm) + exp(v.y - nm) + exp(v.z - nm) + exp(v.w - nm);\n\
         \t\tlm = nm;\n\
         \t}}\n\
         \tfloat sg_m = simd_max(lm);\n\
         \tfloat sg_l = simd_sum(ll * exp(lm - sg_m));\n\
         \tif (lane == 0u) {{ s_m[sg] = sg_m; s_l[sg] = sg_l; }}\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \tfloat gm = -1e30f;\n\
         \tfor (uint s = 0; s < 8u; s++) gm = max(gm, s_m[s]);\n\
         \tfloat gl = 0.0f;\n\
         \tfor (uint s = 0; s < 8u; s++) gl += s_l[s] * exp(s_m[s] - gm);\n\
         \tfloat inv_l = 1.0f / max(gl, 1e-12f);\n\
         \tuint tgt = (uint)((float)targets[row]);\n\
         \t// Pass 2: Fused 128-bit in-place dlogits = (softmax(logits) - one_hot(tgt)) / BT!\n\
         \tfor (uint c4 = lid; c4 < {v4}u; c4 += 256u) {{\n\
         \t\tfloat4 v = float4(logits[base4 + c4]);\n\
         \t\tfloat4 p = exp(v - gm) * inv_l;\n\
         \t\tuint bcol = c4 * 4u;\n\
         \t\tif (tgt >= bcol && tgt < bcol + 4u) {{\n\
         \t\t\tuint r = tgt - bcol;\n\
         \t\t\tif (r == 0u) p.x -= 1.0f; else if (r == 1u) p.y -= 1.0f; else if (r == 2u) p.z -= 1.0f; else p.w -= 1.0f;\n\
         \t\t}}\n\
         \t\tdlogits[base4 + c4] = {vec_ty}(p * {inv_rows:.10}f);\n\
         \t}}\n\
         }}\n\n"
    ));
}

/// Facility #5 Baseline: Naive 2-stage materializing `[BT, V]` FP32 probs in Arena HBM!
fn emit_msl_naive_cross_entropy(msl: &mut String, k1: &str, k2: &str, rows: usize, vocab: usize, dt: IrDType) {
    let v4 = vocab / 4;
    let inv_rows = 1.0f32 / (rows as f32);
    let vec_ty = if dt == IrDType::F32 { "float4" } else { "half4" };
    let sc_ty = dt.msl_ty();
    msl.push_str(&format!(
        "kernel void {k1}(\n\
         \tdevice const {vec_ty}* logits [[buffer(0)]], device float4* probs [[buffer(1)]],\n\
         \tuint row [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
         \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
         ) {{\n\
         \tif (row >= {rows}u) return;\n\
         \tthreadgroup float s_m[8]; threadgroup float s_l[8];\n\
         \tulong base4 = (ulong)row * {v4}ul;\n\
         \tfloat lm = -1e30f;\n\
         \tfor (uint c4 = lid; c4 < {v4}u; c4 += 256u) {{ float4 v = float4(logits[base4 + c4]); lm = max(lm, max(max(v.x, v.y), max(v.z, v.w))); }}\n\
         \tlm = simd_max(lm); if (lane == 0u) s_m[sg] = lm;\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \tfloat gm = -1e30f; for (uint s = 0; s < 8u; s++) gm = max(gm, s_m[s]);\n\
         \tfloat ll = 0.0f;\n\
         \tfor (uint c4 = lid; c4 < {v4}u; c4 += 256u) {{\n\
         \t\tfloat4 e = exp(float4(logits[base4 + c4]) - gm);\n\
         \t\tprobs[base4 + c4] = e;\n\
         \t\tll += (e.x + e.y) + (e.z + e.w);\n\
         \t}}\n\
         \tll = simd_sum(ll); if (lane == 0u) s_l[sg] = ll;\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \tfloat gl = 0.0f; for (uint s = 0; s < 8u; s++) gl += s_l[s];\n\
         \tfloat inv_l = 1.0f / max(gl, 1e-12f);\n\
         \tfor (uint c4 = lid; c4 < {v4}u; c4 += 256u) probs[base4 + c4] *= inv_l;\n\
         }}\n\n\
         kernel void {k2}(\n\
         \tdevice const float4* probs [[buffer(0)]], device const {sc_ty}* targets [[buffer(1)]],\n\
         \tdevice {vec_ty}* dlogits [[buffer(2)]],\n\
         \tuint row [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]\n\
         ) {{\n\
         \tif (row >= {rows}u) return;\n\
         \tulong base4 = (ulong)row * {v4}ul;\n\
         \tuint tgt = (uint)((float)targets[row]);\n\
         \tfor (uint c4 = lid; c4 < {v4}u; c4 += 256u) {{\n\
         \t\tfloat4 p = probs[base4 + c4];\n\
         \t\tuint bcol = c4 * 4u;\n\
         \t\tif (tgt >= bcol && tgt < bcol + 4u) {{\n\
         \t\t\tuint r = tgt - bcol;\n\
         \t\t\tif (r == 0u) p.x -= 1.0f; else if (r == 1u) p.y -= 1.0f; else if (r == 2u) p.z -= 1.0f; else p.w -= 1.0f;\n\
         \t\t}}\n\
         \t\tdlogits[base4 + c4] = {vec_ty}(p * {inv_rows:.10}f);\n\
         \t}}\n\
         }}\n\n"
    ));
}

/// Facility #6: Auto Mixed-Precision W8A16/W8A8 AMX Contraction reading 1-byte INT8 weights (-50% weight DRAM!)
/// via 128-bit `char16` (`uint4`) loads + double-buffered BK=32 pipeline + per-row FP32 dequant epilogue!
#[allow(clippy::too_many_arguments)]
fn emit_msl_amx_w8_gemm(msl: &mut String, kname: &str, m: usize, k: usize, n: usize, bm: usize, bn: usize, sc: TileSchedule) {
    let bk = if bm == 32 { 32usize } else { 16usize };
    let sa_st = bk + 8;
    let sb_st = bk + 8;
    let tm = (bm / 2) / 8;
    let tn = (bn / 2) / 8;
    let sg_m = bm / 2;
    let sg_n = bn / 2;
    let q_bytes_aligned = ((n * k) + 15) & !15;
    let tiles_m = (m + bm - 1) / bm;
    let tiles_n = (n + bn - 1) / bn;
    let group_m = 8usize.min(tiles_m);
    let tile_coord = if sc.tile_swizzle && tiles_m >= 8 && tiles_n >= 8 {
        format!(
            "\tuint _pid = tgid.y * {tiles_n}u + tgid.x; uint _gsz = {group_m}u * {tiles_n}u;\n\
             \tuint _fm = (_pid / _gsz) * {group_m}u; uint _gm = min({tiles_m}u - _fm, {group_m}u); uint _rem = _pid % _gsz;\n\
             \tuint m0 = (_fm + (_rem % _gm)) * {bm}u, n0 = (_rem / _gm) * {bn}u;\n"
        )
    } else {
        format!("\tuint m0 = tgid.y * {bm}u, n0 = tgid.x * {bn}u;\n")
    };

    if bm == 32 && bn == 32 && bk == 32 {
        // High-throughput Double-Buffered 32x32x32 W8A16 AMX (10.2 KB SMEM -> 2 TGs/core, 1 barrier/K-tile,
        // 128-bit `uint4` loads of 16x int8 = HALF the load instructions & HALF the weight DRAM bytes of FP16!)
        let load_a = |buf: &str, k_ex: &str| {
            format!(
                "\t{{ uint r = lid >> 2, c8 = (lid & 3u) * 8u;\n\
                 \t\t*(threadgroup uint4*)(&sA[{buf}][r * {sa_st}u + c8]) = (m0 + r < {m}u) ? *(device const uint4*)(&a[(m0 + r) * {k}u + ({k_ex}) + c8]) : uint4(0u); }}\n"
            )
        };
        // Only 64 threads needed to load 32x32 int8 (1024 bytes = 64 x 16B `uint4` = 1 instruction per thread!)
        let load_b = |buf: &str, k_ex: &str| {
            format!(
                "\tif (lid < 64u) {{\n\
                 \t\tuint r = lid >> 1, c16 = (lid & 1u) * 16u;\n\
                 \t\tuint4 raw = *(const device uint4*)(&w_i8[(n0 + r) * {k}u + ({k_ex}) + c16]);\n\
                 \t\t*(threadgroup half4*)(&sB[{buf}][r * {sb_st}u + c16 + 0u])  = half4(as_type<char4>(raw.x));\n\
                 \t\t*(threadgroup half4*)(&sB[{buf}][r * {sb_st}u + c16 + 4u])  = half4(as_type<char4>(raw.y));\n\
                 \t\t*(threadgroup half4*)(&sB[{buf}][r * {sb_st}u + c16 + 8u])  = half4(as_type<char4>(raw.z));\n\
                 \t\t*(threadgroup half4*)(&sB[{buf}][r * {sb_st}u + c16 + 12u]) = half4(as_type<char4>(raw.w));\n\
                 \t}}\n"
            )
        };
        msl.push_str(&format!(
            "kernel void {kname}(\n\
             \tdevice const half* a [[buffer(0)]], device const char* w_i8 [[buffer(1)]], device half* out [[buffer(2)]],\n\
             \tuint3 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
             \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
             ) {{\n\
             \tconst device float* w_scale = (const device float*)(w_i8 + {q_bytes_aligned}ul);\n\
             \tthreadgroup half sA[2][32 * {sa_st}];\n\
             \tthreadgroup half sB[2][32 * {sb_st}];\n\
             \tthreadgroup float sC[4][128];\n\
             {tile_coord}\
             \tuint sg_r = (sg >> 1) * 16u, sg_c = (sg & 1u) * 16u;\n\
             \tsimdgroup_float8x8 mc[2][2];\n\
             \t#pragma unroll\n\
             \tfor (uint r = 0; r < 2u; r++) for (uint c = 0; c < 2u; c++) mc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n\
             {la0}{lb0}\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
             \tuint buf = 0u;\n\
             \tfor (uint kt = 0; kt < {k}u; kt += 32u) {{\n\
             \t\tuint nkt = kt + 32u, nbuf = buf ^ 1u;\n\
             \t\tif (nkt < {k}u) {{\n\
             {lan}{lbn}\t\t}}\n\
             \t\t#pragma unroll\n\
             \t\tfor (uint kk = 0; kk < 32u; kk += 8u) {{\n\
             \t\t\tsimdgroup_half8x8 ma[2], mb[2];\n\
             \t\t\tfor (uint rr = 0; rr < 2u; rr++) simdgroup_load(ma[rr], &sA[buf][(sg_r + rr * 8u) * {sa_st}u + kk], {sa_st}ul);\n\
             \t\t\tfor (uint cc = 0; cc < 2u; cc++) simdgroup_load(mb[cc], &sB[buf][(sg_c + cc * 8u) * {sb_st}u + kk], {sb_st}ul, ulong2(0, 0), true);\n\
             \t\t\tfor (uint rr = 0; rr < 2u; rr++) for (uint cc = 0; cc < 2u; cc++) simdgroup_multiply_accumulate(mc[rr][cc], ma[rr], mb[cc], mc[rr][cc]);\n\
             \t\t}}\n\
             \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
             \t\tbuf = nbuf;\n\
             \t}}\n\
             \tuint er = lane >> 2, ec4 = (lane & 3u) * 4u;\n\
             \t#pragma unroll\n\
             \tfor (uint r = 0; r < 2u; r++) {{\n\
             \t\tsimdgroup_store(mc[r][0], &sC[sg][0], 16ul);\n\
             \t\tsimdgroup_store(mc[r][1], &sC[sg][8], 16ul);\n\
             \t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
             \t\tuint gr = m0 + sg_r + r * 8u + er, gc = n0 + sg_c + ec4;\n\
             \t\tif (gr < {m}u && gc < {n}u) {{\n\
             \t\t\tfloat4 v4 = *(threadgroup const float4*)(&sC[sg][er * 16u + ec4]) * *(const device float4*)(&w_scale[gc]);\n\
             \t\t\t*(device half4*)(&out[gr * {n}u + gc]) = half4(v4);\n\
             \t\t}}\n\
             \t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
             \t}}\n\
             }}\n\n",
            la0 = load_a("0", "0u"),
            lb0 = load_b("0", "0u"),
            lan = load_a("nbuf", "nkt"),
            lbn = load_b("nbuf", "nkt"),
        ));
        return;
    }

    // 64x64x16 Double-Buffered W8A16 AMX (`128-bit uint4` loads of 16x int8 -> 64 threads load entire 64x16 B tile in 1 op!)
    let load_a64 = |buf: &str, k_ex: &str| {
        format!(
            "\t{{ uint r = lid >> 1, c8 = (lid & 1u) * 8u;\n\
             \t\t*(threadgroup uint4*)(&sA[{buf}][r * {sa_st}u + c8]) = (m0 + r < {m}u) ? *(device const uint4*)(&a[(m0 + r) * {k}u + ({k_ex}) + c8]) : uint4(0u); }}\n"
        )
    };
    let load_b64 = |buf: &str, k_ex: &str| {
        format!(
            "\tif (lid < 64u) {{\n\
             \t\tuint4 raw = *(const device uint4*)(&w_i8[(n0 + lid) * {k}u + ({k_ex})]);\n\
             \t\t*(threadgroup half4*)(&sB[{buf}][lid * {sb_st}u + 0u])  = half4(as_type<char4>(raw.x));\n\
             \t\t*(threadgroup half4*)(&sB[{buf}][lid * {sb_st}u + 4u])  = half4(as_type<char4>(raw.y));\n\
             \t\t*(threadgroup half4*)(&sB[{buf}][lid * {sb_st}u + 8u])  = half4(as_type<char4>(raw.z));\n\
             \t\t*(threadgroup half4*)(&sB[{buf}][lid * {sb_st}u + 12u]) = half4(as_type<char4>(raw.w));\n\
             \t}}\n"
        )
    };
    msl.push_str(&format!(
        "kernel void {kname}(\n\
         \tdevice const half* a [[buffer(0)]], device const char* w_i8 [[buffer(1)]], device half* out [[buffer(2)]],\n\
         \tuint3 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
         \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
         ) {{\n\
         \tconst device float* w_scale = (const device float*)(w_i8 + {q_bytes_aligned}ul);\n\
         \tthreadgroup half sA[2][{bm} * {sa_st}];\n\
         \tthreadgroup half sB[2][{bn} * {sb_st}];\n\
         \tthreadgroup float sC[4][128];\n\
         {tile_coord}\
         \tuint sg_r = (sg >> 1) * {sg_m}u, sg_c = (sg & 1u) * {sg_n}u;\n\
         \tsimdgroup_float8x8 mc[{tm}][{tn}];\n\
         \t#pragma unroll\n\
         \tfor (uint r = 0; r < {tm}u; r++) for (uint c = 0; c < {tn}u; c++) mc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n\
         {la0}{lb0}\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \tuint buf = 0u;\n\
         \tfor (uint kt = 0; kt < {k}u; kt += 16u) {{\n\
         \t\tuint nkt = kt + 16u, nbuf = buf ^ 1u;\n\
         \t\tif (nkt < {k}u) {{\n\
         {lan}{lbn}\t\t}}\n\
         \t\t#pragma unroll\n\
         \t\tfor (uint kk = 0; kk < 16u; kk += 8u) {{\n\
         \t\t\tsimdgroup_half8x8 ma[{tm}], mb[{tn}];\n\
         \t\t\tfor (uint rr = 0; rr < {tm}u; rr++) simdgroup_load(ma[rr], &sA[buf][(sg_r + rr * 8u) * {sa_st}u + kk], {sa_st}ul);\n\
         \t\t\tfor (uint cc = 0; cc < {tn}u; cc++) simdgroup_load(mb[cc], &sB[buf][(sg_c + cc * 8u) * {sb_st}u + kk], {sb_st}ul, ulong2(0, 0), true);\n\
         \t\t\tfor (uint rr = 0; rr < {tm}u; rr++) for (uint cc = 0; cc < {tn}u; cc++) simdgroup_multiply_accumulate(mc[rr][cc], ma[rr], mb[cc], mc[rr][cc]);\n\
         \t\t}}\n\
         \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\tbuf = nbuf;\n\
         \t}}\n\
         \tuint er = lane >> 2, ec4 = (lane & 3u) * 4u;\n\
         \t#pragma unroll\n\
         \tfor (uint r = 0; r < {tm}u; r++) for (uint c = 0; c < {tn}u; c += 2u) {{\n\
         \t\tsimdgroup_store(mc[r][c], &sC[sg][0], 16ul);\n\
         \t\tsimdgroup_store(mc[r][c + 1u], &sC[sg][8], 16ul);\n\
         \t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\tuint gr = m0 + sg_r + r * 8u + er, gc = n0 + sg_c + c * 8u + ec4;\n\
         \t\tif (gr < {m}u && gc < {n}u) {{\n\
         \t\t\tfloat4 v4 = *(threadgroup const float4*)(&sC[sg][er * 16u + ec4]) * *(const device float4*)(&w_scale[gc]);\n\
         \t\t\t*(device half4*)(&out[gr * {n}u + gc]) = half4(v4);\n\
         \t\t}}\n\
         \t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t}}\n\
         }}\n\n",
        la0 = load_a64("0", "0u"),
        lb0 = load_b64("0", "0u"),
        lan = load_a64("nbuf", "nkt"),
        lbn = load_b64("nbuf", "nkt"),
    ));
}

/// Facility #7 (Bucket 0: `M <= 8`): Dynamic-Shape Coalesced SIMD GEMV (`m_actual` read from ctrl buffer!).
fn emit_msl_dyn_gemv(msl: &mut String, kname: &str, k: usize, n: usize, dt: IrDType) {
    let mt = dt.msl_ty();
    let k8 = k / 8;
    msl.push_str(&format!(
        "kernel void {kname}(\n\
         \tdevice const {mt}* a [[buffer(0)]], device const {mt}* b [[buffer(1)]],\n\
         \tdevice const uint* ctrl [[buffer(2)]], device {mt}* out [[buffer(3)]],\n\
         \tuint3 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
         ) {{\n\
         \tuint m_act = ctrl[0], row = tgid.y, col = tgid.x * 4u + sg;\n\
         \tif (row >= m_act || col >= {n}u) return;\n\
         \tfloat4 acc = float4(0.0f);\n\
         \tfor (uint kk8 = lane; kk8 < {k8}u; kk8 += 32u) {{\n\
         \t\tfloat4 x0 = float4(*(device const half4*)(&a[row * {k}u + kk8 * 8u]));\n\
         \t\tfloat4 x1 = float4(*(device const half4*)(&a[row * {k}u + kk8 * 8u + 4u]));\n\
         \t\tfloat4 w0 = float4(*(device const half4*)(&b[col * {k}u + kk8 * 8u]));\n\
         \t\tfloat4 w1 = float4(*(device const half4*)(&b[col * {k}u + kk8 * 8u + 4u]));\n\
         \t\tacc = fma(x0, w0, acc); acc = fma(x1, w1, acc);\n\
         \t}}\n\
         \tfloat res = simd_sum((acc.x + acc.y) + (acc.z + acc.w));\n\
         \tif (lane == 0u) out[row * {n}u + col] = ({mt})res;\n\
         }}\n\n"
    ));
}

/// Facility #7 (Bucket 1/2: `9..64` / `65..M_max`): Dynamic-Shape AMX GEMM with exact `m_actual` masking (ZERO padding!).
#[allow(clippy::too_many_arguments)]
fn emit_msl_dyn_amx(msl: &mut String, kname: &str, bm: usize, bn: usize, bk: usize, k: usize, n: usize, dt: IrDType, _swz: bool) {
    let mt = dt.msl_ty();
    let sa_st = bk + 8;
    let sb_st = bk + 8;
    let tm = (bm / 2) / 8;
    let tn = (bn / 2) / 8;
    let n_u4_a = (bm * bk) / 8;
    let n_u4_b = (bn * bk) / 8;
    let k8 = bk / 8;
    msl.push_str(&format!(
        "kernel void {kname}(\n\
         \tdevice const {mt}* a [[buffer(0)]], device const {mt}* b [[buffer(1)]],\n\
         \tdevice const uint* ctrl [[buffer(2)]], device {mt}* out [[buffer(3)]],\n\
         \tuint3 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
         \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
         ) {{\n\
         \tuint m_act = ctrl[0];\n\
         \tuint m0 = tgid.y * {bm}u, n0 = tgid.x * {bn}u;\n\
         \tif (m0 >= m_act) return;\n\
         \tthreadgroup half sA[{bm} * {sa_st}];\n\
         \tthreadgroup half sB[{bn} * {sb_st}];\n\
         \tthreadgroup float sC[4][128];\n\
         \tuint sg_r = (sg >> 1) * {sg_m}u, sg_c = (sg & 1u) * {sg_n}u;\n\
         \tsimdgroup_float8x8 mc[{tm}][{tn}];\n\
         \t#pragma unroll\n\
         \tfor (uint r = 0; r < {tm}u; r++) for (uint c = 0; c < {tn}u; c++) mc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n\
         \tfor (uint kt = 0; kt < {k}u; kt += {bk}u) {{\n\
         \t\tfor (uint idx = lid; idx < {n_u4_a}u; idx += 128u) {{\n\
         \t\t\tuint r = idx / {k8}u, c8 = (idx % {k8}u) * 8u;\n\
         \t\t\tuint gr = m0 + r;\n\
         \t\t\t*(threadgroup uint4*)(&sA[r * {sa_st}u + c8]) = (gr < m_act) ? *(device const uint4*)(&a[gr * {k}u + kt + c8]) : uint4(0u);\n\
         \t\t}}\n\
         \t\tfor (uint idx = lid; idx < {n_u4_b}u; idx += 128u) {{\n\
         \t\t\tuint r = idx / {k8}u, c8 = (idx % {k8}u) * 8u;\n\
         \t\t\t*(threadgroup uint4*)(&sB[r * {sb_st}u + c8]) = *(device const uint4*)(&b[(n0 + r) * {k}u + kt + c8]);\n\
         \t\t}}\n\
         \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\t#pragma unroll\n\
         \t\tfor (uint kk = 0; kk < {bk}u; kk += 8u) {{\n\
         \t\t\tsimdgroup_half8x8 ma[{tm}], mb[{tn}];\n\
         \t\t\tfor (uint rr = 0; rr < {tm}u; rr++) simdgroup_load(ma[rr], &sA[(sg_r + rr * 8u) * {sa_st}u + kk], {sa_st}ul);\n\
         \t\t\tfor (uint cc = 0; cc < {tn}u; cc++) simdgroup_load(mb[cc], &sB[(sg_c + cc * 8u) * {sb_st}u + kk], {sb_st}ul, ulong2(0, 0), true);\n\
         \t\t\tfor (uint rr = 0; rr < {tm}u; rr++) for (uint cc = 0; cc < {tn}u; cc++) simdgroup_multiply_accumulate(mc[rr][cc], ma[rr], mb[cc], mc[rr][cc]);\n\
         \t\t}}\n\
         \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t}}\n\
         \tuint er = lane >> 2, ec4 = (lane & 3u) * 4u;\n\
         \tfor (uint r = 0; r < {tm}u; r++) for (uint c = 0; c < {tn}u; c += 2u) {{\n\
         \t\tsimdgroup_store(mc[r][c], &sC[sg][0], 16ul);\n\
         \t\tsimdgroup_store(mc[r][c + 1u], &sC[sg][8], 16ul);\n\
         \t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\tuint gr = m0 + sg_r + r * 8u + er, gc = n0 + sg_c + c * 8u + ec4;\n\
         \t\tif (gr < m_act && gc < {n}u) *(device half4*)(&out[gr * {n}u + gc]) = half4(*(threadgroup const float4*)(&sC[sg][er * 16u + ec4]));\n\
         \t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t}}\n\
         }}\n\n",
        sg_m = bm / 2,
        sg_n = bn / 2,
    ));
}

/// Facility #8A: Zero-`im2col` Implicit GEMM 3x3 Conv2d AMX Kernel (0 KB `im2col` HBM expansion!).
fn emit_msl_implicit_conv2d_3x3(msl: &mut String, kname: &str, in_h: usize, in_w: usize, c_in: usize, c_out: usize, dt: IrDType) {
    let mt = dt.msl_ty();
    let m = in_h * in_w;
    let k_eff = c_in * 9;
    let sa_st = 24usize;
    let sb_st = 24usize;
    msl.push_str(&format!(
        "kernel void {kname}(\n\
         \tdevice const {mt}* x [[buffer(0)]], device const {mt}* w [[buffer(1)]], device {mt}* out [[buffer(2)]],\n\
         \tuint3 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
         \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
         ) {{\n\
         \tthreadgroup half sA[64 * {sa_st}];\n\
         \tthreadgroup half sB[64 * {sb_st}];\n\
         \tthreadgroup float sC[4][128];\n\
         \tuint m0 = tgid.y * 64u, n0 = tgid.x * 64u;\n\
         \tuint sg_r = (sg >> 1) * 32u, sg_c = (sg & 1u) * 32u;\n\
         \tsimdgroup_float8x8 mc[4][4];\n\
         \t#pragma unroll\n\
         \tfor (uint r = 0; r < 4u; r++) for (uint c = 0; c < 4u; c++) mc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);\n\
         \tuint r_ld = lid >> 1, c8_ld = (lid & 1u) * 8u;\n\
         \tuint gm = m0 + r_ld;\n\
         \tint oh = (int)(gm / {in_w}u), ow = (int)(gm % {in_w}u);\n\
         \tfor (uint kt = 0; kt < {k_eff}u; kt += 16u) {{\n\
         \t\t// Implicit GEMM: compute (ih, iw, ci) on the fly for 128-bit uint4 load -> ZERO im2col buffer!\n\
         \t\tuint gk = kt + c8_ld;\n\
         \t\tuint tap = gk / {c_in}u, ci = gk % {c_in}u;\n\
         \t\tint ih = oh + (int)(tap / 3u) - 1, iw = ow + (int)(tap % 3u) - 1;\n\
         \t\tbool inb = (gm < {m}u) && (ih >= 0) && (ih < {in_h}) && (iw >= 0) && (iw < {in_w});\n\
         \t\t*(threadgroup uint4*)(&sA[r_ld * {sa_st}u + c8_ld]) = inb ? *(device const uint4*)(&x[((uint)ih * {in_w}u + (uint)iw) * {c_in}u + ci]) : uint4(0u);\n\
         \t\t*(threadgroup uint4*)(&sB[r_ld * {sb_st}u + c8_ld]) = *(device const uint4*)(&w[(n0 + r_ld) * {k_eff}u + gk]);\n\
         \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\t#pragma unroll\n\
         \t\tfor (uint kk = 0; kk < 16u; kk += 8u) {{\n\
         \t\t\tsimdgroup_half8x8 ma[4], mb[4];\n\
         \t\t\tfor (uint rr = 0; rr < 4u; rr++) simdgroup_load(ma[rr], &sA[(sg_r + rr * 8u) * {sa_st}u + kk], {sa_st}ul);\n\
         \t\t\tfor (uint cc = 0; cc < 4u; cc++) simdgroup_load(mb[cc], &sB[(sg_c + cc * 8u) * {sb_st}u + kk], {sb_st}ul, ulong2(0, 0), true);\n\
         \t\t\tfor (uint rr = 0; rr < 4u; rr++) for (uint cc = 0; cc < 4u; cc++) simdgroup_multiply_accumulate(mc[rr][cc], ma[rr], mb[cc], mc[rr][cc]);\n\
         \t\t}}\n\
         \t\tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t}}\n\
         \tuint er = lane >> 2, ec4 = (lane & 3u) * 4u;\n\
         \tfor (uint r = 0; r < 4u; r++) for (uint c = 0; c < 4u; c += 2u) {{\n\
         \t\tsimdgroup_store(mc[r][c], &sC[sg][0], 16ul);\n\
         \t\tsimdgroup_store(mc[r][c + 1u], &sC[sg][8], 16ul);\n\
         \t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t\tuint gr = m0 + sg_r + r * 8u + er, gc = n0 + sg_c + c * 8u + ec4;\n\
         \t\tif (gr < {m}u && gc < {c_out}u) *(device half4*)(&out[gr * {c_out}u + gc]) = half4(*(threadgroup const float4*)(&sC[sg][er * 16u + ec4]));\n\
         \t\tsimdgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t}}\n\
         }}\n\n"
    ));
}

/// Facility #8A Baseline: Explicit `im2col` materialization `[H*W, 9*C_in]` in Arena HBM!
fn emit_msl_explicit_im2col_3x3(msl: &mut String, kname: &str, in_h: usize, in_w: usize, c_in: usize, dt: IrDType) {
    let mt = dt.msl_ty();
    let m = in_h * in_w;
    let k_eff = c_in * 9;
    let total = m * k_eff;
    msl.push_str(&format!(
        "kernel void {kname}(device const {mt}* x [[buffer(0)]], device {mt}* col [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n\
         \tif (gid >= {total}u) return;\n\
         \tuint m_idx = gid / {k_eff}u, k_idx = gid % {k_eff}u;\n\
         \tint oh = (int)(m_idx / {in_w}u), ow = (int)(m_idx % {in_w}u);\n\
         \tuint tap = k_idx / {c_in}u, ci = k_idx % {c_in}u;\n\
         \tint ih = oh + (int)(tap / 3u) - 1, iw = ow + (int)(tap % 3u) - 1;\n\
         \tbool inb = (ih >= 0) && (ih < {in_h}) && (iw >= 0) && (iw < {in_w});\n\
         \tcol[gid] = inb ? x[((uint)ih * {in_w}u + (uint)iw) * {c_in}u + ci] : ({mt})0;\n\
         }}\n\n"
    ));
}

/// Facility #8B: Work-Efficient 2-Level SIMD+TG Associative Parallel Scan (`h_t = a_t*h_{t-1} + b_t`, `O(log L)` depth!).
fn emit_msl_parallel_scan(msl: &mut String, kname: &str, batch: usize, seq_len: usize, dt: IrDType) {
    let l4 = seq_len / 4;
    let vec_ty = if dt == IrDType::F32 { "float4" } else { "half4" };
    msl.push_str(&format!(
        "kernel void {kname}(\n\
         \tdevice const {vec_ty}* a_in [[buffer(0)]], device const {vec_ty}* b_in [[buffer(1)]], device {vec_ty}* out [[buffer(2)]],\n\
         \tuint row [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
         \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
         ) {{\n\
         \tif (row >= {batch}u) return;\n\
         \tthreadgroup float s_a[8];\n\
         \tthreadgroup float s_b[8];\n\
         \tthreadgroup float s_pa[8];\n\
         \tthreadgroup float s_pb[8];\n\
         \tulong base4 = (ulong)row * {l4}ul;\n\
         \tfloat4 va = (lid < {l4}u) ? float4(a_in[base4 + lid]) : float4(1.0f);\n\
         \tfloat4 vb = (lid < {l4}u) ? float4(b_in[base4 + lid]) : float4(0.0f);\n\
         \t// Step 1: Local 4-element associative prefix in registers\n\
         \tfloat a0 = va.x, b0 = vb.x;\n\
         \tfloat a1 = va.y * a0, b1 = fma(va.y, b0, vb.y);\n\
         \tfloat a2 = va.z * a1, b2 = fma(va.z, b1, vb.z);\n\
         \tfloat a3 = va.w * a2, b3 = fma(va.w, b2, vb.w);\n\
         \t// Step 2: 5-step SIMD associative prefix scan (`simd_shuffle_up`)\n\
         \tfloat ia = a3, ib = b3;\n\
         \tfor (uint off = 1u; off < 32u; off <<= 1) {{\n\
         \t\tfloat pa = simd_shuffle_up(ia, off);\n\
         \t\tfloat pb = simd_shuffle_up(ib, off);\n\
         \t\tif (lane >= off) {{ ib = fma(ia, pb, ib); ia = ia * pa; }}\n\
         \t}}\n\
         \tfloat ex_a = simd_shuffle_up(ia, 1u), ex_b = simd_shuffle_up(ib, 1u);\n\
         \tif (lane == 0u) {{ ex_a = 1.0f; ex_b = 0.0f; }}\n\
         \tif (lane == 31u) {{ s_a[sg] = ia; s_b[sg] = ib; }}\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t// Step 3: 3-step 8-SG prefix scan in SG 0\n\
         \tif (sg == 0u) {{\n\
         \t\tfloat ga = (lane < 8u) ? s_a[lane] : 1.0f;\n\
         \t\tfloat gb = (lane < 8u) ? s_b[lane] : 0.0f;\n\
         \t\tfor (uint off = 1u; off < 8u; off <<= 1) {{\n\
         \t\t\tfloat pa = simd_shuffle_up(ga, off);\n\
         \t\t\tfloat pb = simd_shuffle_up(gb, off);\n\
         \t\t\tif (lane >= off && lane < 8u) {{ gb = fma(ga, pb, gb); ga = ga * pa; }}\n\
         \t\t}}\n\
         \t\tfloat pga = simd_shuffle_up(ga, 1u), pgb = simd_shuffle_up(gb, 1u);\n\
         \t\tif (lane < 8u) {{ s_pa[lane] = (lane == 0u) ? 1.0f : pga; s_pb[lane] = (lane == 0u) ? 0.0f : pgb; }}\n\
         \t}}\n\
         \tthreadgroup_barrier(mem_flags::mem_threadgroup);\n\
         \t// Step 4: Apply exact incoming state `h_prev = ex_a * s_pb[sg] + ex_b` to all 4 local elements!\n\
         \tfloat h_prev = fma(ex_a, s_pb[sg], ex_b);\n\
         \tif (lid < {l4}u) {{\n\
         \t\tout[base4 + lid] = {vec_ty}(fma(a0, h_prev, b0), fma(a1, h_prev, b1), fma(a2, h_prev, b2), fma(a3, h_prev, b3));\n\
         \t}}\n\
         }}\n\n"
    ));
}

/// Facility #8B Baseline: Sequential `O(L)` recurrence per batch row!
fn emit_msl_sequential_scan(msl: &mut String, kname: &str, batch: usize, seq_len: usize, dt: IrDType) {
    let mt = dt.msl_ty();
    msl.push_str(&format!(
        "kernel void {kname}(device const {mt}* a_in [[buffer(0)]], device const {mt}* b_in [[buffer(1)]], device {mt}* out [[buffer(2)]], uint row [[thread_position_in_grid]]) {{\n\
         \tif (row >= {batch}u) return;\n\
         \tulong base = (ulong)row * {seq_len}ul;\n\
         \tfloat h = 0.0f;\n\
         \tfor (uint t = 0; t < {seq_len}u; t++) {{\n\
         \t\th = fma((float)a_in[base + t], h, (float)b_in[base + t]);\n\
         \t\tout[base + t] = ({mt})h;\n\
         \t}}\n\
         }}\n\n"
    ));
}
