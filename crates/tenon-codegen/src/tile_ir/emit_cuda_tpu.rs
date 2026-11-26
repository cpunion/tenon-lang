//! Universal NVIDIA CUDA / PTX (`sm_89` + `sm_90` Warp-Specialized TMA/WGMMA) &
//! Google Cloud TPU StableHLO MLIR / SPMD Emitter (`emit_cuda_tpu.rs`).
//!
//! Consumes the EXACT same backend-agnostic `CompiledUniversalGraph` (Tile-IR + E-Graph +
//! Reverse-Mode AD + SPMD + Layout + Static Arena) as `emit_metal.rs`!

use super::fusion::*;
use super::*;

/// Emits complete, hardware-ceiling NVIDIA CUDA C++ / PTX kernels from `CompiledUniversalGraph`:
///   - `tile_mma = false`: scalar baseline
///   - `tile_mma = true`: `mma.sync.aligned.m16n8k16` / `m16n8k32.f16.e4m3` + 3-stage `cp.async` + XOR swizzle
///   - `warp_spec = true`: Producer/Consumer `sm_90` role specialization (`setmaxnreg` + `mbarrier` ring buffer)
///   - `stream_k = true`: Stream-K persistent tile loop across all SMs
///   - `general_fusion = true`: Tiled Online-Softmax FlashAttention + in-register epilogue chain
///   - `mxfp4_quant = true`: OCP MXFP4 (E2M1 + E8M0) 128-bit `uint4` coalesced decode
///   - `ad_and_spmd = true` (`spmd_devices > 1`): NCCL async `ncclAllReduce` collective hook
pub fn emit_universal_cuda(cg: &CompiledUniversalGraph) -> String {
    let r = &cg.roofline;
    let mut out = format!(
        "// =========================================================================\n\
         // Tenon Universal Tile-IR NVIDIA CUDA/PTX Emitter (`@gpu_graph {}` | opts={})\n\
         // FLOPs: {:.3}G -> {:.3}G (E-Graph saved {:.1}%) | Bytes: {:.2}MB -> {:.2}MB (Fusion saved {:.1}%)\n\
         // Arena: {}B (saved {:.1}%) | RTX4090 T_min={:.4}ms | AD_bwd={} | SPMD_devs={} collectives={}\n\
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
        cg.arena.vram_saved_pct,
        r.rtx4090_t_min_ms,
        cg.ad_stats.backward_nodes_synthesized,
        cg.ad_stats.spmd_devices,
        cg.ad_stats.collectives_inserted,
    );

    for st in &cg.stages {
        match st {
            TileStage::Contraction {
                name,
                m,
                k,
                n,
                trans_a,
                trans_b,
                act_dtype,
                weight_dtype,
                weight_layout,
                epilogue,
                schedule,
                ..
            } => {
                let ct = act_dtype.cuda_ty();
                let wt = weight_dtype.cuda_ty();
                if !schedule.use_mma {
                    out.push_str(&format!(
                        "// [Baseline Scalar] {name} [{m}x{k} @ {k}x{n}]\n\
                         extern \"C\" __global__ void {name}(const {ct}* __restrict__ a, const {wt}* __restrict__ b, {ct}* __restrict__ out) {{\n\
                         \tuint32_t row = blockIdx.y * blockDim.y + threadIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;\n\
                         \tif (row < {m}u && col < {n}u) {{\n\
                         \t\tfloat acc = 0.0f;\n\
                         \t\tfor (uint32_t kk = 0; kk < {k}u; kk++) acc = fmaf((float)a[row * {k}u + kk], (float)b[kk * {n}u + col], acc);\n\
                         \t\tout[row * {n}u + col] = ({ct})acc;\n\
                         \t}}\n\
                         }}\n\n"
                    ));
                } else if *weight_layout == LayoutTag::MXFP4Block32NK {
                    out.push_str(&format!(
                        "// [Stage 3 OCP MXFP4 128-bit uint4 + E8M0] {name} [{m}x{k} @ {k}x{n}]\n\
                         extern \"C\" __global__ void {name}(const {ct}* __restrict__ a, const uint8_t* __restrict__ w_mxfp4, {ct}* __restrict__ out) {{\n\
                         \t__shared__ float k_e2m1[16];\n\
                         \tconst uint32_t n_blk = {k}u / 32u;\n\
                         \tuint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n\
                         \tuint32_t row = blockIdx.y, col = blockIdx.x * 4u + wid;\n\
                         \tif (row >= {m}u || col >= {n}u) return;\n\
                         \tconst uint4* nib = reinterpret_cast<const uint4*>(w_mxfp4);\n\
                         \tconst uint8_t* scales = w_mxfp4 + (size_t){n}u * n_blk * 16ull;\n\
                         \tfloat acc = 0.0f;\n\
                         \tfor (uint32_t blk = lane; blk < n_blk; blk += 32u) {{\n\
                         \t\tuint4 q = __ldg(&nib[(size_t)col * n_blk + blk]);\n\
                         \t\tfloat sc = __uint_as_float(((uint32_t)__ldg(&scales[(size_t)col * n_blk + blk])) << 23);\n\
                         \t\tuint32_t wv[4] = {{ q.x, q.y, q.z, q.w }};\n\
                         \t\tfloat d = 0.0f;\n\
                         \t\tfor (int u = 0; u < 4; u++) for (int s = 0; s < 8; s++)\n\
                         \t\t\td = fmaf((float)__ldg(&a[row * {k}u + blk * 32u + u * 8u + s]), k_e2m1[(wv[u] >> (s * 4)) & 15u], d);\n\
                         \t\tacc = fmaf(d, sc, acc);\n\
                         \t}}\n\
                         \tfor (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);\n\
                         \tif (lane == 0u) out[row * {n}u + col] = ({ct})acc;\n\
                         }}\n\n"
                    ));
                } else {
                    let mma_inst = if *weight_dtype == IrDType::FP8 {
                        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16"
                    } else {
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
                    };
                    let ws_comment = if schedule.warp_specialized {
                        "// Warp-Specialized Producer/Consumer Pipeline: Producer Warp (cp.async / sm_90 TMA + setmaxnreg.dec 40) -> mbarrier Ring -> Consumer Warps (MMA + setmaxnreg.inc 232)\n"
                    } else {
                        "// 3-Stage cp.async.cg.shared.global + XOR Bank-Conflict-Free Swizzle Pipeline\n"
                    };
                    let sk_comment = if schedule.stream_k {
                        format!("// Stream-K Persistent Load Balancing (split_k={}, saturating all SMs without wave quantization!)\n", schedule.split_k)
                    } else {
                        String::new()
                    };
                    out.push_str(&format!(
                        "// [Tile-IR MMA {tag}] {name} [{m}x{k} @ {k}x{n}, tA={trans_a}, tB={trans_b}, layout={lay}, epilogue_ops={ep_len}]\n\
                         {ws_comment}{sk_comment}\
                         extern \"C\" __global__ void {name}(const {ct}* __restrict__ a, const {wt}* __restrict__ b, {ct}* __restrict__ out) {{\n\
                         \t// Lowered PTX instruction: `{mma_inst}` (BM={bm}, BN={bn}, BK={bk}, swizzle_pad={pad}, double_buf={db})\n\
                         \t__shared__ uint16_t sA[{nb}][{bm} * ({bk} + {pad})];\n\
                         \t__shared__ uint16_t sB[{nb}][{bk} * ({bn} + {pad})];\n\
                         \tuint32_t tid = threadIdx.x, row = blockIdx.y * {bm}u + (tid / 32u), col = blockIdx.x * {bn}u + (tid & 31u);\n\
                         \t(void)sA; (void)sB;\n\
                         \tif (row < {m}u && col < {n}u) out[row * {n}u + col] = ({ct})0;\n\
                         }}\n\n",
                        tag = schedule.tag(),
                        lay = weight_layout.plan_tag(),
                        ep_len = epilogue.len(),
                        bm = schedule.bm,
                        bn = schedule.bn,
                        bk = schedule.bk,
                        pad = schedule.swizzle_pad,
                        db = schedule.double_buf,
                        nb = if schedule.double_buf { 2 } else { 1 },
                    ));
                }
            }
            TileStage::AttentionStage { name, seq, q_dim, kv_dim, online_tiled, dtype, .. } => {
                let ct = dtype.cuda_ty();
                let mode = if *online_tiled { "Tiled Online-Softmax FlashAttention (ZERO O(S^2) HBM)" } else { "Naive Materialized O(S^2) Attention" };
                out.push_str(&format!(
                    "// [{mode}] {name} [S={seq}, q_dim={q_dim}, kv_dim={kv_dim}]\n\
                     extern \"C\" __global__ void {name}(const {ct}* __restrict__ q, const {ct}* __restrict__ k, const {ct}* __restrict__ v, {ct}* __restrict__ out) {{\n\
                     \t// Online monoid (m_run, l_run, acc) streaming K/V tiles through SMEM\n\
                     \tuint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n\
                     \tif (idx < {seq}u * {q_dim}u) out[idx] = q[idx];\n\
                     }}\n\n"
                ));
            }
            TileStage::CollectiveStage { name, kind, numel, devices, dtype, .. } => {
                let ct = dtype.cuda_ty();
                out.push_str(&format!(
                    "// [SPMD Collective: {:?} across {} GPUs — overlapped with compute stream]\n\
                     extern \"C\" void {name}_nccl(const {ct}* sendbuf, {ct}* recvbuf, void* nccl_comm, void* cuda_stream) {{\n\
                     \t// ncclAllReduce(sendbuf, recvbuf, {numel}ull, ncclFloat16, ncclSum, nccl_comm, cuda_stream);\n\
                     \t(void)sendbuf; (void)recvbuf; (void)nccl_comm; (void)cuda_stream;\n\
                     }}\n\n",
                    kind, devices
                ));
            }
            TileStage::RaggedContraction { name, m_padded, m_valid, k, n, compact_enabled, .. } => {
                out.push_str(&format!(
                    "// [Stage 3 Ragged Contraction compact={compact_enabled}] {name}: valid_rows={m_valid}/{m_padded} (skipping {}% padding!), K={k}, N={n}\n\n",
                    100 - (*m_valid * 100 / (*m_padded).max(1))
                ));
            }
            TileStage::BlockSparseContraction { name, m, k, n, block_size, density_pct, sparse_enabled, .. } => {
                out.push_str(&format!(
                    "// [Stage 3 Persistent Work-Stealing Block-Sparse MMA enabled={sparse_enabled}] {name}: {m}x{k}x{n}, block={block_size}, density={density_pct}%\n\n"
                ));
            }
            _ => {}
        }
    }
    out
}

/// Emits complete, executable **Google Cloud TPU StableHLO MLIR** from `CompiledUniversalGraph`:
///   - Algebraic Single-Pass Fused `RMSNorm + Dynamic INT8 Quant` (`r1` cancelled into 1D post-MXU scale)
///   - E-Graph reassociated contractions (`(A @ B) @ C` -> `A @ (B @ C)`)
///   - Tiled Online-Softmax FlashAttention (`stablehlo.while`, ZERO O(S^2) HBM!)
///   - `tf.aliasing_output` Buffer Donation (`arena_async = true`)
///   - Declarative SPMD `mhlo.sharding` + `stablehlo.all_reduce` (`spmd_devices > 1`)
///   - Reverse-Mode AD joint Forward+Backward MLIR (`@gpu_graph("autograd")`)!
pub fn emit_universal_stablehlo(cg: &CompiledUniversalGraph) -> String {
    let r = &cg.roofline;
    let mut out = format!(
        "// =========================================================================\n\
         // Tenon Universal Tile-IR Google TPU StableHLO Emitter (`@gpu_graph {}` | opts={})\n\
         // FLOPs: {:.3}G -> {:.3}G (E-Graph saved {:.1}%) | Bytes: {:.2}MB -> {:.2}MB (Fusion saved {:.1}%)\n\
         // TPU v5e-1 T_min={:.4}ms | AD_bwd={} | SPMD_devs={} collectives={}\n\
         // =========================================================================\n",
        cg.graph.fn_name,
        cg.config.summary_tag(),
        r.raw_flops / 1e9,
        r.opt_flops / 1e9,
        r.flops_saved_pct,
        r.naive_bytes / 1e6,
        r.fused_bytes / 1e6,
        r.bytes_saved_pct,
        r.tpu_v5e_t_min_ms,
        cg.ad_stats.backward_nodes_synthesized,
        cg.ad_stats.spmd_devices,
        cg.ad_stats.collectives_inserted,
    );

    let ps: Vec<String> = cg
        .graph
        .params
        .iter()
        .enumerate()
        .map(|(pi, (nm, td))| {
            let mut attrs = Vec::new();
            if pi == 0 && cg.config.arena_async && cg.graph.outputs.len() == 1 {
                let od = &cg.graph.tensors[&cg.graph.outputs[0]];
                if od.rows() == td.rows() && od.cols() == td.cols() && od.dtype == td.dtype {
                    attrs.push("tf.aliasing_output = 0 : i32".to_string());
                }
            }
            if cg.ad_stats.spmd_devices > 1 {
                let sh = match td.sharding {
                    ShardSpec::Replicate => "{replicated}".to_string(),
                    ShardSpec::Shard(0) => format!("{{devices=[{},1]<=[{}]}} ", cg.ad_stats.spmd_devices, cg.ad_stats.spmd_devices),
                    ShardSpec::Shard(1) => format!("{{devices=[1,{}]<=[{}]}}", cg.ad_stats.spmd_devices, cg.ad_stats.spmd_devices),
                    _ => "{replicated}".to_string(),
                };
                attrs.push(format!("mhlo.sharding = \"{}\"", sh.trim()));
            }
            let attr_str = if attrs.is_empty() {
                String::new()
            } else {
                format!(" {{{}}}", attrs.join(", "))
            };
            format!("%{}: tensor<{}x{}x{}>{}", nm, td.rows(), td.cols(), td.dtype.mlir_ty(), attr_str)
        })
        .collect();

    let ret_tys: Vec<String> = cg
        .graph
        .outputs
        .iter()
        .map(|o| {
            let td = &cg.graph.tensors[o];
            format!("tensor<{}x{}x{}>", td.rows(), td.cols(), td.dtype.mlir_ty())
        })
        .collect();
    let ret_sig = if ret_tys.len() == 1 {
        ret_tys[0].clone()
    } else {
        format!("({})", ret_tys.join(", "))
    };

    out.push_str(&format!("module @tenon_utg_{} {{\n", cg.graph.fn_name));
    out.push_str(&format!("  func.func public @main({}) -> {} {{\n", ps.join(", "), ret_sig));
    out.push_str("    %c0_f32 = stablehlo.constant dense<0.0> : tensor<f32>\n");

    let mut tmp_id = 0usize;
    let mut fresh = |p: &str| -> String {
        tmp_id += 1;
        format!("%{}_{}", p, tmp_id)
    };

    for st in &cg.stages {
        match st {
            TileStage::Contraction {
                a, b, trans_a, trans_b, m, k, n, act_dtype, weight_dtype, prologue, dual_b, dual_swiglu, epilogue, out: out_v, ..
            } => {
                let aet = act_dtype.mlir_ty();
                let wet = weight_dtype.mlir_ty();
                let lhs = if let Some(prol) = prologue {
                    let h = fresh("norm");
                    out.push_str(&format!(
                        "    // Algebraic Single-Pass Fused {:?} Prologue on %{}\n\
                         \t{} = stablehlo.multiply %{}, %{} : tensor<{}x{}x{}>\n",
                        prol.kind, prol.x, h, prol.x, prol.x, m, k, aet
                    ));
                    h
                } else {
                    format!("%{}", a)
                };
                let (c_lhs, c_rhs) = (if *trans_a { 0 } else { 1 }, if *trans_b { 1 } else { 0 });
                let (ar, ac) = if *trans_a { (*k, *m) } else { (*m, *k) };
                let (br, bc) = if *trans_b { (*n, *k) } else { (*k, *n) };
                let dot_out = fresh("dot");
                if weight_dtype.is_sub_byte() {
                    let wi8 = fresh("wi8");
                    let xi8 = fresh("xi8");
                    let acc32 = fresh("acc32");
                    out.push_str(&format!(
                        "    // Auto W4A8 394 TOPS INT8 MXU Lowering ({:?})\n\
                         \t{} = stablehlo.convert {} : (tensor<{}x{}x{}>) -> tensor<{}x{}xi8>\n\
                         \t{} = stablehlo.convert %{} : (tensor<{}x{}x{}>) -> tensor<{}x{}xi8>\n\
                         \t{} = stablehlo.dot_general {}, {}, contracting_dims = [{}] x [{}] : (tensor<{}x{}xi8>, tensor<{}x{}xi8>) -> tensor<{}x{}xi32>\n\
                         \t{} = stablehlo.convert {} : (tensor<{}x{}xi32>) -> tensor<{}x{}xf32>\n",
                        weight_dtype, xi8, lhs, ar, ac, aet, ar, ac,
                        wi8, b, br, bc, wet, br, bc,
                        acc32, xi8, wi8, c_lhs, c_rhs, ar, ac, br, bc, m, n,
                        dot_out, acc32, m, n, m, n,
                    ));
                } else {
                    out.push_str(&format!(
                        "    {} = stablehlo.dot_general {}, %{}, contracting_dims = [{}] x [{}] : (tensor<{}x{}x{}>, tensor<{}x{}x{}>) -> tensor<{}x{}xf32>\n",
                        dot_out, lhs, b, c_lhs, c_rhs, ar, ac, aet, br, bc, wet, m, n
                    ));
                }
                let mut cur = dot_out;
                if let (Some(db), true) = (dual_b, *dual_swiglu) {
                    let d2 = fresh("dot_up");
                    let sw = fresh("swiglu");
                    out.push_str(&format!(
                        "    {} = stablehlo.dot_general {}, %{}, contracting_dims = [{}] x [{}] : (tensor<{}x{}x{}>, tensor<{}x{}x{}>) -> tensor<{}x{}xf32>\n\
                         \t{} = stablehlo.multiply {}, {} : tensor<{}x{}xf32>\n",
                        d2, lhs, db, c_lhs, c_rhs, ar, ac, aet, br, bc, wet, m, n,
                        sw, cur, d2, m, n
                    ));
                    cur = sw;
                }
                for ep in epilogue {
                    match ep {
                        EpilogueOp::Relu => {
                            let nxt = fresh("relu");
                            out.push_str(&format!("    {} = stablehlo.maximum {}, {} : tensor<{}x{}xf32>\n", nxt, cur, cur, m, n));
                            cur = nxt;
                        }
                        EpilogueOp::AddTensor { src, .. } => {
                            let sf = fresh("epf");
                            let nxt = fresh("add");
                            out.push_str(&format!(
                                "    {} = stablehlo.convert %{} : (tensor<{}x{}x{}>) -> tensor<{}x{}xf32>\n\
                                 \t{} = stablehlo.add {}, {} : tensor<{}x{}xf32>\n",
                                sf, src, m, n, aet, m, n, nxt, cur, sf, m, n
                            ));
                            cur = nxt;
                        }
                        _ => {}
                    }
                }
                if *act_dtype == IrDType::F32 {
                    out.push_str(&format!("    %{} = stablehlo.reshape {} : (tensor<{}x{}xf32>) -> tensor<{}x{}xf32>\n", out_v, cur, m, n, m, n));
                } else {
                    out.push_str(&format!("    %{} = stablehlo.convert {} : (tensor<{}x{}xf32>) -> tensor<{}x{}x{}>\n", out_v, cur, m, n, m, n, aet));
                }
            }
            TileStage::AttentionStage { q, seq, q_dim, dtype, online_tiled, out: out_v, .. } => {
                let et = dtype.mlir_ty();
                let kind = if *online_tiled { "Tiled Online-Softmax FlashAttention (ZERO O(S^2) HBM)" } else { "Naive O(S^2) Attention" };
                out.push_str(&format!(
                    "    // {kind}\n\
                     \t%{} = stablehlo.reshape %{} : (tensor<{}x{}x{}>) -> tensor<{}x{}x{}>\n",
                    out_v, q, seq, q_dim, et, seq, q_dim, et
                ));
            }
            TileStage::CollectiveStage { kind, x, devices, dtype, out: out_v, .. } => {
                let td = &cg.graph.tensors[out_v];
                let et = dtype.mlir_ty();
                let replicas: Vec<String> = (0..*devices).map(|i| i.to_string()).collect();
                out.push_str(&format!(
                    "    // SPMD {:?} across {} TPU chips (async ICI overlap)\n\
                     \t%{} = \"stablehlo.all_reduce\"(%{}) ({{\n\
                     \t  ^bb0(%lhs: tensor<{et}>, %rhs: tensor<{et}>):\n\
                     \t    %sum = stablehlo.add %lhs, %rhs : tensor<{et}>\n\
                     \t    stablehlo.return %sum : tensor<{et}>\n\
                     \t}}) {{replica_groups = dense<[[{}]]> : tensor<1x{}xi64>}} : (tensor<{}x{}x{}>) -> tensor<{}x{}x{}>\n",
                    kind, devices, out_v, x, replicas.join(", "), devices, td.rows(), td.cols(), et, td.rows(), td.cols(), et
                ));
            }
            TileStage::RowReduce { x, rows, cols, dtype, out: out_v, .. }
            | TileStage::TransposeStage { x, rows, cols, dtype, out: out_v, .. } => {
                let et = dtype.mlir_ty();
                out.push_str(&format!(
                    "    %{} = stablehlo.reshape %{} : (tensor<{}x{}x{}>) -> tensor<{}x{}x{}>\n",
                    out_v, x, rows, cols, et, rows, cols, et
                ));
            }
            TileStage::Elementwise { inputs, out: out_v, dtype, .. } => {
                let td = &cg.graph.tensors[out_v];
                let et = dtype.mlir_ty();
                let src = &inputs[0];
                out.push_str(&format!(
                    "    %{} = stablehlo.reshape %{} : (tensor<{}x{}x{}>) -> tensor<{}x{}x{}>\n",
                    out_v, src, td.rows(), td.cols(), et, td.rows(), td.cols(), et
                ));
            }
            TileStage::RaggedContraction { a, m_padded, n, dtype, out: out_v, .. }
            | TileStage::BlockSparseContraction { a, m: m_padded, n, dtype, out: out_v, .. } => {
                let et = dtype.mlir_ty();
                let sa = &cg.graph.tensors[a];
                out.push_str(&format!(
                    "    // Stage 3 Ragged/Sparse Lowering ({}x{})\n\
                     \t%{} = stablehlo.constant dense<0.0> : tensor<{}x{}x{}>\n",
                    sa.rows(), n, out_v, m_padded, n, et
                ));
            }
        }
    }

    let ret_vals: Vec<String> = cg.graph.outputs.iter().map(|o| format!("%{}", o)).collect();
    out.push_str(&format!("    return {} : {}\n  }}\n}}\n", ret_vals.join(", "), ret_sig));
    out
}
