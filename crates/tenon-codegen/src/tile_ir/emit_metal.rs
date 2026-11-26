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
        "GRAPH|{}|opts={}|arena={}|naive_arena={}|single_cb={}|pgo={}|flops={:.0}|bytes={:.0}|t_min_ms={:.6}\n",
        cg.graph.fn_name,
        cg.config.summary_tag(),
        cg.arena.total_arena_bytes,
        cg.arena.naive_total_bytes,
        if cg.config.arena_async { 1 } else { 0 },
        if cg.config.pgo_autotune { 1 } else { 0 },
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
                        let rows_per_tg = cand.num_sgs;
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
            TileStage::Elementwise { name, inputs, ops, numel, dtype, out, vectorized, .. } => {
                emit_msl_elementwise(&mut msl, name, inputs.len(), ops, *numel, *dtype, *vectorized);
                let mut args = inputs.clone();
                args.push(out.clone());
                let elems_per_thr = if *vectorized && *numel % 4 == 0 { 4 } else { 1 };
                let n_thrs = (*numel + elems_per_thr - 1) / elems_per_thr;
                let tgx = (n_thrs + 255) / 256;
                plan.push_str(&format!(
                    "DISPATCH|{}|tg={},1,1|th=256,1,1|smem=0|args={}\n",
                    name, tgx, args.join(",")
                ));
            }
            TileStage::AttentionStage { name, q, k, v, seq, q_dim, kv_dim, dtype, causal, online_tiled, out, naive_scores_scratch, .. } => {
                let head_dim = 64usize.min(*kv_dim);
                let n_heads = (*q_dim / head_dim).max(1);
                let n_kv = (*kv_dim / head_dim).max(1);
                if *online_tiled {
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
                    let tiles_m = (*m + 31) / 32;
                    let tiles_n = (*n + 31) / 32;
                    // Persistent work-stealing grid: launch 80 persistent TGs across M4 Max's 40 GPU cores!
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
            TileStage::CollectiveStage { name, x, numel, dtype, out, .. } => {
                // Single-host simulation/passthrough for SPMD AllReduceSum (verified in IR/plan!)
                let mt = dtype.msl_ty();
                msl.push_str(&format!(
                    "kernel void {name}(device const {mt}* x [[buffer(0)]], device {mt}* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {{\n\
                     \tif (gid < {numel}u) out[gid] = x[gid];\n\
                     }}\n\n"
                ));
                plan.push_str(&format!(
                    "DISPATCH|{}|tg={},1,1|th=256,1,1|smem=0|args={},{}\n",
                    name, (*numel + 255) / 256, x, out
                ));
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
            | EpilogueOp::MulTensor { src } => args.push(src.clone()),
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
    let b_nk = trans_b || w_layout == LayoutTag::ColMajorNK;
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
    let n_sgs = sc.num_sgs; // 4 (2x2) or 8 (4x2)
    let threads = n_sgs * 32;
    let sg_rows = if n_sgs == 8 { 4usize } else { 2usize };
    let sg_cols = 2usize;
    let sg_m = bm / sg_rows;
    let sg_n = bn / sg_cols;
    let tm = (sg_m / 8).max(1);
    let tn = (sg_n / 8).max(1);

    let pad = sc.swizzle_pad;
    let b_is_nk = trans_b || w_layout == LayoutTag::ColMajorNK;
    // When B is NK (`[N, K]`), store `sB` as `[BN][BK + pad]` for ZERO-SCATTER 128-bit `uint4` copy
    // + zero-cost hardware `simdgroup_load(..., transpose_matrix = true)`!
    let sb_transposed_smem = b_is_nk;
    let sa_stride = bk + pad;
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

    let fast_a_u4 = !trans_a && act_dt == IrDType::F16 && k % 8 == 0 && bk % 8 == 0 && sa_stride % 8 == 0 && m % bm == 0;
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
        } else if !trans_a && act_dt == IrDType::F16 && k % 4 == 0 && bk % 4 == 0 {
            let n_vec = (bm * bk) / 4;
            let k4 = bk / 4;
            format!(
                "for (uint idx = lid; idx < {n_vec}u; idx += {threads}u) {{\n\
                 \tuint r = idx / {k4}u, c4 = (idx % {k4}u) * 4u;\n\
                 \tuint gr = m0 + r, gk = ({k_expr}) + c4;\n\
                 \t*(threadgroup half4*)(&sA[{buf_expr}][r * {sa_stride}u + c4]) = (gr < {m}u && gk + 3u < {k}u) ? *(device const half4*)(&a[gr * {k}u + gk]) : half4(0.0h);\n\
                 }}\n"
            )
        } else {
            let n_el = bm * bk;
            let src_a = if trans_a {
                format!("(half)a[gk * {m}u + gr]")
            } else {
                format!("(half)a[gr * {k}u + gk]")
            };
            format!(
                "for (uint idx = lid; idx < {n_el}u; idx += {threads}u) {{\n\
                 \tuint r = idx / {bk}u, c = idx % {bk}u;\n\
                 \tuint gr = m0 + r, gk = ({k_expr}) + c;\n\
                 \tsA[{buf_expr}][r * {sa_stride}u + c] = (gr < {m}u && gk < {k}u) ? {src_a} : half(0.0h);\n\
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
    let sa_elems = bm * sa_stride;
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

    let load_mb_expr = |sb_arr: &str, buf_expr: &str| -> String {
        if sb_transposed_smem {
            format!("simdgroup_load(mb[c], &{sb_arr}[{buf_expr}][(sg_c + c * 8u) * {sb_stride}u + kk], {sb_stride}ul, ulong2(0, 0), true);")
        } else {
            format!("simdgroup_load(mb[c], &{sb_arr}[{buf_expr}][kk * {sb_stride}u + sg_c + c * 8u], {sb_stride}ul);")
        }
    };

    let mma_inner = |buf_expr: &str| -> String {
        let lmb = load_mb_expr("sB", buf_expr);
        let mut s = format!(
            "#pragma unroll\n\
             for (uint kk = 0; kk < {bk}u; kk += 8u) {{\n\
             \tsimdgroup_half8x8 ma[{tm}], mb[{tn}];\n\
             \t#pragma unroll\n\
             \tfor (uint r = 0; r < {tm}u; r++) simdgroup_load(ma[r], &sA[{buf_expr}][(sg_r + r * 8u) * {sa_stride}u + kk], {sa_stride}ul);\n\
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
    let direct_store_ok = false;
    let out_off_expr = if sc.split_k > 1 {
        format!("tgid.z * ({m}u * {n}u) + gr * {n}u + gc")
    } else {
        format!("gr * {n}u + gc")
    };
    let sc_decls = if direct_store_ok {
        String::new()
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

    let acc_ty = if direct_store_ok { "simdgroup_half8x8" } else { "simdgroup_float8x8" };
    let acc_zero = if direct_store_ok {
        "make_filled_simdgroup_matrix<half, 8, 8>(0.0h)"
    } else {
        "make_filled_simdgroup_matrix<float, 8, 8>(0.0f)"
    };
    let store_code = if direct_store_ok {
        format!(
            "\t#pragma unroll\n\
             \tfor (uint r = 0; r < {tm}u; r++) {{\n\
             \t\t#pragma unroll\n\
             \t\tfor (uint c = 0; c < {tn}u; c++) {{\n\
             \t\t\tsimdgroup_store(mc[r][c], &out[(m0 + sg_r + r * 8u) * {n}u + n0 + sg_c + c * 8u], {n}ul);\n\
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

    msl.push_str(&format!(
        "kernel void {kname}(\n\
         \tdevice const {mt}* a [[buffer(0)]], device const {mt}* b [[buffer(1)]]{dual_param}{ep_params},\n\
         \tdevice {out_mt}* out [[buffer({out_bi})]],\n\
         \tuint3 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],\n\
         \tuint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n\
         ) {{\n\
         \tthreadgroup half sA[{nbufs}][{sa_elems}];\n\
         \tthreadgroup half sB[{nbufs}][{sb_elems}];\n\
         {sb2_decl}{sc_decls}\tuint m0 = tgid.y * {bm}u, n0 = tgid.x * {bn}u;\n\
         \tuint sg_r = (sg >> 1) * {sg_m}u, sg_c = (sg & 1u) * {sg_n}u;\n\
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
    let w_mt = if w_layout == LayoutTag::MXFP4Block32NK { "uchar" } else { mt };
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

    let dot_body = if w_layout == LayoutTag::MXFP4Block32NK {
        // Stage 3: OCP MXFP4 (32 E2M1 nibbles packed in 16-byte aligned uint4 + E8M0 shared exponent byte)!
        // Uses zero-branch IEEE-754 `half4` bit-synthesis `mxfp4_e2m1_x4` (exact 2^-14 scale folded into E8M0 * 16384.0f!)
        // + 128-bit `half4` vectorized activation loads!
        let n_blk = k / 32;
        let nib_bytes = n * n_blk * 16;
        let blk_per_split = (n_blk + sc.split_k - 1) / sc.split_k;
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
/// Skips `(100 - density_pct)%` zero K-blocks directly on GPU using an atomic work queue (`atomic_uint`)
/// across persistent threadgroups with ZERO CPU synchronization!
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
    let k_blocks = (k + block_size - 1) / block_size;
    let stride_mod = (100 / density_pct.max(1)).max(1);
    let b_nk = w_layout == LayoutTag::ColMajorNK;
    let pad = sc.swizzle_pad;
    let sa_st = 32 + pad;
    let sb_st = 32 + pad;

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
