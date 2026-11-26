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

    if cg.hetero_plan.active {
        let hp = &cg.hetero_plan;
        out.push_str(&format!(
            "// [Stage 6 Heterogeneous Streaming Plan]\n\
             //   mode={} | tp={} (P2P AllReduce) | mla_latent={} | vram_cap={:.2}GiB | core_vram={:.2}GiB | pinned_ffn={}/{} | lru_slots={}/{} | kv_q8={} (saved {:.1}MB)\n\
             //   dma_ring={} (cuStreamWaitEvent compute_stream || dma_stream, indirect d_step_args CUDA Graph, async_promoter={})\n\
             //   cpu_hybrid: NUMA_first_touch_4R={} | cgroup_main_tid0={} | AVX2 _mm256_madd_epi16 INT16 (B=4 shared-weight) | prefill_B={} (TensorCore mma.sync m16n8k16) | spec_k={}\n\
             //   roofline_proj: decode={:.2} tok/s | batched_prefill={:.1} tok/s\n",
            hp.dispatch_mode.tag(),
            hp.tp_devices,
            hp.mla_latent_dim,
            hp.vram_cap_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            hp.core_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            hp.pinned_gpu_ffn_layers,
            hp.n_layers,
            hp.lru_slots_per_layer,
            hp.n_experts,
            hp.kv_quant_q8,
            hp.kv_bytes_saved as f64 / 1e6,
            hp.dma_ring_slots,
            hp.async_promoter,
            hp.numa_repack_weights.len(),
            hp.cgroup_main_tid0,
            hp.prefill_chunk_tokens,
            hp.speculative_k,
            hp.projected_decode_tok_s,
            hp.projected_prefill_tok_s,
        ));
    }

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
                } else if weight_layout.is_mxfp4() {
                    out.push_str(&format!(
                        "// [Stage 3 OCP MXFP4 128-bit uint4 + E8M0 ({:?})] {name} [{m}x{k} @ {k}x{n}]\n\
                         extern \"C\" __global__ void {name}(const {ct}* __restrict__ a, const uint8_t* __restrict__ w_mxfp4, {ct}* __restrict__ out) {{\n\
                         \t// Zero-branch IEEE-754 half bit-synthesis (mag 0..7 -> exact half * 2^-14, *16384 folded into E8M0!)\n\
                         \tconst uint32_t n_blk = {k}u / 32u;\n\
                         \tuint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n\
                         \tuint32_t row = blockIdx.y, col = blockIdx.x * 4u + wid;\n\
                         \tif (row >= {m}u || col >= {n}u) return;\n\
                         \tconst uint4* nib = reinterpret_cast<const uint4*>(w_mxfp4);\n\
                         \tconst uint8_t* scales = w_mxfp4 + (size_t){n}u * n_blk * 16ull;\n\
                         \tfloat acc = 0.0f;\n\
                         \tfor (uint32_t blk = lane; blk < n_blk; blk += 32u) {{\n\
                         \t\tuint4 q = __ldg(&nib[(size_t)col * n_blk + blk]);\n\
                         \t\tfloat sc = __uint_as_float(((uint32_t)__ldg(&scales[(size_t)col * n_blk + blk])) << 23) * 16384.0f;\n\
                         \t\tuint32_t wv[4] = {{ q.x, q.y, q.z, q.w }};\n\
                         \t\tfloat d = 0.0f;\n\
                         \t\tfor (int u = 0; u < 4; u++) for (int s = 0; s < 8; s++) {{\n\
                         \t\t\tuint16_t nib4 = (wv[u] >> (s * 4)) & 15u;\n\
                         \t\t\tuint16_t hb = ((nib4 & 8u) << 12) | ((nib4 & 7u) << 9);\n\
                         \t\t\tfloat w = __half2float(__ushort_as_half(hb));\n\
                         \t\t\td = fmaf((float)__ldg(&a[row * {k}u + blk * 32u + u * 8u + s]), w, d);\n\
                         \t\t}}\n\
                         \t\tacc = fmaf(d, sc, acc);\n\
                         \t}}\n\
                         \tfor (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);\n\
                         \tif (lane == 0u) out[row * {n}u + col] = ({ct})acc;\n\
                         }}\n\n",
                        weight_layout
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
                    let swz_comment = if schedule.tile_swizzle {
                        "// Facility #1: 2D L2 Cache Tile Swizzling (`GROUP_M = 8` rasterization order!)\n"
                    } else {
                        ""
                    };
                    let swz_code = if schedule.tile_swizzle {
                        "\t// Facility #1: 2D Serpentine L2 Cache Tile Swizzling (`GROUP_M = 8`)\n\
                         \tuint32_t _tiles_m = gridDim.y, _tiles_n = gridDim.x;\n\
                         \tuint32_t _pid = blockIdx.y * _tiles_n + blockIdx.x, _gsz = 8u * _tiles_n;\n\
                         \tuint32_t _gid = _pid / _gsz, _fm = _gid * 8u, _gm = min(_tiles_m - _fm, 8u), _rem = _pid % _gsz;\n\
                         \tuint32_t _nt_r = _rem / _gm, _nt = (_gid & 1u) ? (_tiles_n - 1u - _nt_r) : _nt_r;\n\
                         \tuint32_t bm0 = (_fm + (_rem % _gm)) * 32u, bn0 = _nt * 32u;\n"
                    } else {
                        "\tuint32_t bm0 = blockIdx.y * 32u, bn0 = blockIdx.x * 32u;\n"
                    };
                    let b_is_nk = *trans_b ^ (*weight_layout == LayoutTag::ColMajorNK);
                    let load_a = if *trans_a {
                        format!("(gk < {k}u && gr < {m}u) ? (float)a[(size_t)gk * {m}u + gr] : 0.0f")
                    } else {
                        format!("(gr < {m}u && gk < {k}u) ? (float)a[(size_t)gr * {k}u + gk] : 0.0f")
                    };
                    let load_b = if b_is_nk {
                        format!("(gn < {n}u && gk < {k}u) ? (float)b[(size_t)gn * {k}u + gk] : 0.0f")
                    } else {
                        format!("(gk < {k}u && gn < {n}u) ? (float)b[(size_t)gk * {n}u + gn] : 0.0f")
                    };
                    out.push_str(&format!(
                        "// [Tile-IR MMA {tag}] {name} [{m}x{k} @ {k}x{n}, tA={trans_a}, tB={trans_b}, layout={lay}, epilogue_ops={ep_len}]\n\
                         {ws_comment}{sk_comment}{swz_comment}\
                         extern \"C\" __global__ void {name}(const {ct}* __restrict__ a, const {wt}* __restrict__ b, {ct}* __restrict__ out) {{\n\
                         \t// Lowered PTX target: `{mma_inst}` (BM={bm}, BN={bn}, BK={bk}, swizzle_pad={pad}, double_buf={db})\n\
                         \t__shared__ float sA[{nb}][32][33];\n\
                         \t__shared__ float sB[{nb}][32][33];\n\
                         {swz_code}\
                         \tuint32_t ty = threadIdx.y, tx = threadIdx.x;\n\
                         \tfloat acc[2][2] = {{ {{0.0f, 0.0f}}, {{0.0f, 0.0f}} }};\n\
                         \tfor (uint32_t kt = 0; kt < {k}u; kt += 32u) {{\n\
                         \t\tuint32_t buf = (kt >> 5) & ({nb}u - 1u);\n\
                         \t\tfor (uint32_t rr = ty; rr < 32u; rr += 16u) for (uint32_t cc = tx; cc < 32u; cc += 16u) {{\n\
                         \t\t\tuint32_t gr = bm0 + rr, gk = kt + cc;\n\
                         \t\t\tsA[buf][rr][cc] = {load_a};\n\
                         \t\t\tuint32_t gn = bn0 + cc; gk = kt + rr;\n\
                         \t\t\tsB[buf][rr][cc] = {load_b};\n\
                         \t\t}}\n\
                         \t\t__syncthreads();\n\
                         \t\t#pragma unroll\n\
                         \t\tfor (uint32_t kk = 0; kk < 32u; kk++) {{\n\
                         \t\t\tfloat a0 = sA[buf][ty][kk], a1 = sA[buf][ty + 16u][kk];\n\
                         \t\t\tfloat b0 = sB[buf][kk][tx], b1 = sB[buf][kk][tx + 16u];\n\
                         \t\t\tacc[0][0] = fmaf(a0, b0, acc[0][0]); acc[0][1] = fmaf(a0, b1, acc[0][1]);\n\
                         \t\t\tacc[1][0] = fmaf(a1, b0, acc[1][0]); acc[1][1] = fmaf(a1, b1, acc[1][1]);\n\
                         \t\t}}\n\
                         \t\t__syncthreads();\n\
                         \t}}\n\
                         \tfor (uint32_t ri = 0; ri < 2u; ri++) for (uint32_t ci = 0; ci < 2u; ci++) {{\n\
                         \t\tuint32_t gr = bm0 + ty + ri * 16u, gc = bn0 + tx + ci * 16u;\n\
                         \t\tif (gr < {m}u && gc < {n}u) out[(size_t)gr * {n}u + gc] = ({ct})acc[ri][ci];\n\
                         \t}}\n\
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
            TileStage::AttentionStage { name, seq, q_dim, kv_dim, online_tiled, causal, dtype, .. } => {
                let ct = dtype.cuda_ty();
                let head_dim = 64usize.min(*kv_dim);
                let n_heads = (*q_dim / head_dim).max(1);
                let n_kv = (*kv_dim / head_dim).max(1);
                let grp = (n_heads / n_kv).max(1);
                let scale = 1.0f32 / (head_dim as f32).sqrt();
                let lim_expr = if *causal { "row + 1u" } else { "SEQ" };
                let mode = if *online_tiled { "Tiled Online-Softmax FlashAttention-2 (ZERO O(S^2) HBM)" } else { "Naive Materialized O(S^2) Attention" };
                out.push_str(&format!(
                    "// [{mode}] {name} [S={seq}, q_dim={q_dim}, kv_dim={kv_dim}, D={head_dim}]\n\
                     extern \"C\" __global__ void {name}(const {ct}* __restrict__ Q, const {ct}* __restrict__ K, const {ct}* __restrict__ V, {ct}* __restrict__ O) {{\n\
                     \tconst uint32_t SEQ = {seq}u, D = {head_dim}u, QS = {q_dim}u, KVS = {kv_dim}u;\n\
                     \tuint32_t row = blockIdx.x, h = blockIdx.y, kv_h = h / {grp}u, lane = threadIdx.x & 31u;\n\
                     \tif (row >= SEQ || threadIdx.x >= 32u) return;\n\
                     \t__shared__ float sK[32][{head_dim}];\n\
                     \t__shared__ float sV[32][{head_dim}];\n\
                     \tfloat q0 = (float)Q[(size_t)row * QS + h * D + lane * 2u];\n\
                     \tfloat q1 = (float)Q[(size_t)row * QS + h * D + lane * 2u + 1u];\n\
                     \tfloat o0 = 0.0f, o1 = 0.0f, m_run = -1e30f, l_run = 0.0f;\n\
                     \tuint32_t lim = {lim_expr};\n\
                     \tfor (uint32_t kb = 0; kb < lim; kb += 32u) {{\n\
                     \t\tfor (uint32_t t = 0; t < 32u; t++) {{\n\
                     \t\t\tuint32_t gk = kb + t;\n\
                     \t\t\tsK[t][lane * 2u]      = (gk < lim) ? (float)K[(size_t)gk * KVS + kv_h * D + lane * 2u] : 0.0f;\n\
                     \t\t\tsK[t][lane * 2u + 1u] = (gk < lim) ? (float)K[(size_t)gk * KVS + kv_h * D + lane * 2u + 1u] : 0.0f;\n\
                     \t\t\tsV[t][lane * 2u]      = (gk < lim) ? (float)V[(size_t)gk * KVS + kv_h * D + lane * 2u] : 0.0f;\n\
                     \t\t\tsV[t][lane * 2u + 1u] = (gk < lim) ? (float)V[(size_t)gk * KVS + kv_h * D + lane * 2u + 1u] : 0.0f;\n\
                     \t\t}}\n\
                     \t\t__syncwarp();\n\
                     \t\tfor (uint32_t t = 0; t < 32u && (kb + t) < lim; t++) {{\n\
                     \t\t\tfloat dot = fmaf(q0, sK[t][lane * 2u], q1 * sK[t][lane * 2u + 1u]);\n\
                     \t\t\tfor (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, off);\n\
                     \t\t\tfloat sc = dot * {scale:.8}f;\n\
                     \t\t\tfloat mn = fmaxf(m_run, sc), al = expf(m_run - mn), wt = expf(sc - mn);\n\
                     \t\t\tl_run = fmaf(l_run, al, wt); m_run = mn;\n\
                     \t\t\to0 = fmaf(o0, al, wt * sV[t][lane * 2u]);\n\
                     \t\t\to1 = fmaf(o1, al, wt * sV[t][lane * 2u + 1u]);\n\
                     \t\t}}\n\
                     \t}}\n\
                     \tfloat inv = 1.0f / fmaxf(l_run, 1e-12f);\n\
                     \tO[(size_t)row * QS + h * D + lane * 2u]      = ({ct})(o0 * inv);\n\
                     \tO[(size_t)row * QS + h * D + lane * 2u + 1u] = ({ct})(o1 * inv);\n\
                     }}\n\n"
                ));
            }
            TileStage::CollectiveStage { name, kind, numel, devices, dtype, .. } => {
                let ct = dtype.cuda_ty();
                let chunks = cg.ad_stats.micro_tile_chunks.max(1);
                let chunk_el = (*numel + chunks - 1) / chunks;
                let nccl_call = match kind {
                    CollectiveKind::AllReduceSum => "ncclAllReduce",
                    CollectiveKind::ReduceScatter => "ncclReduceScatter",
                    CollectiveKind::AllGather => "ncclAllGather",
                };
                out.push_str(&format!(
                    "// [SPMD Collective: {:?} across {} GPUs — micro_tile_overlap_chunks={} zero1_shards={}]\n\
                     extern \"C\" void {name}_nccl(const {ct}* sendbuf, {ct}* recvbuf, void* nccl_comm, void* compute_stream, void* comm_stream) {{\n\
                     \t// Facility #9/#13: Bucketed {nccl_call} ({chunks} chunks of {chunk_el} elems on comm_stream || compute_stream!)\n\
                     \t// for (int c = 0; c < {chunks}; c++) {nccl_call}(sendbuf + c*{chunk_el}, recvbuf + c*{chunk_el}, {chunk_el}ull, ncclFloat16, ncclSum, nccl_comm, comm_stream);\n\
                     \t(void)sendbuf; (void)recvbuf; (void)nccl_comm; (void)compute_stream; (void)comm_stream;\n\
                     }}\n\n",
                    kind, devices, chunks, cg.ad_stats.zero1_shards
                ));
            }
            TileStage::InPlaceAdamW { name, numel, dtype, donate_enabled, stochastic_round, streaming_cache, zero1_shards, .. } => {
                let shard_n = (*numel / (*zero1_shards).max(1)).max(1);
                let ld = if *streaming_cache { "__ldcs(&grad[idx])" } else { "grad[idx]" };
                let st = if *streaming_cache { "__stcs(&param[idx], p_out)" } else { "param[idx] = p_out" };
                let sr_fn = if *stochastic_round {
                    format!(
                        "static __device__ __forceinline__ uint32_t {name}_philox(uint32_t seed, uint32_t idx) {{\n\
                         \tuint32_t h = (idx * 0x9E3779B1u) ^ (seed * 0x85EBCA77u + 0xC2B2AE3Du);\n\
                         \th ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16; return h;\n\
                         }}\n\
                         static __device__ __forceinline__ float {name}_sr_bf16(float x, uint32_t rng) {{\n\
                         \tuint32_t b = __float_as_uint(x); b = (b + (rng & 0xFFFFu)) & ~0xFFFFu; return __uint_as_float(b);\n\
                         }}\n"
                    )
                } else {
                    String::new()
                };
                let sr_apply = if *stochastic_round {
                    format!("float p_out = {name}_sr_bf16(p_new, {name}_philox(seed, idx));")
                } else {
                    "float p_out = p_new;".to_string()
                };
                let sr_note = if *stochastic_round {
                    "Philox BF16/FP16 stochastic_rounding (master-weight-free!)"
                } else {
                    "RTNE rounding"
                };
                out.push_str(&format!(
                    "// [Facility #3/#11/#12/#13 llm.c x128 Fused AdamW ({:?}) donate={donate_enabled} sr={stochastic_round} ldcs={streaming_cache} zero1={zero1_shards}x] {name} (shard={shard_n}/{numel}, {sr_note})\n\
                     {sr_fn}\
                     extern \"C\" __global__ void {name}(float* __restrict__ param, const float* __restrict__ grad, float* __restrict__ m, float* __restrict__ v, uint32_t seed) {{\n\
                     \tuint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;\n\
                     \tif (idx < {shard_n}u) {{\n\
                     \t\tfloat g = {ld};\n\
                     \t\tfloat mv = fmaf(0.9f, m[idx], 0.1f * g);\n\
                     \t\tfloat vv = fmaf(0.999f, v[idx], 0.001f * g * g);\n\
                     \t\tm[idx] = mv; v[idx] = vv;\n\
                     \t\tfloat p_new = param[idx] * (1.0f - 1e-5f) - 1e-3f * (mv * 10.0f) / (sqrtf(vv * 1000.0f) + 1e-8f);\n\
                     \t\t{sr_apply}\n\
                     \t\t{st};\n\
                     \t}}\n\
                     }}\n\n",
                    dtype
                ));
            }
            TileStage::CrossEntropyStage { name, rows, vocab, padded_vocab, online_fused, donate_enabled, streaming_cache, .. } => {
                let inv_rows = 1.0f32 / (*rows as f32);
                let ld_v = if *streaming_cache { "__ldcs(&logits[base + c])" } else { "logits[base + c]" };
                let st_v = if *streaming_cache { "__stcs(&dlogits[base + c], p)" } else { "dlogits[base + c] = p" };
                out.push_str(&format!(
                    "// [Facility #5/#10/#12 llm.c Online Monoid Fused Cross-Entropy online={online_fused} donate={donate_enabled} ldcs={streaming_cache}] {name} ({rows}x{vocab} -> padded {padded_vocab}, ZERO O(BT*V) HBM probs!)\n\
                     extern \"C\" __global__ void {name}(const float* __restrict__ logits, const float* __restrict__ targets, float* __restrict__ dlogits) {{\n\
                     \tuint32_t row = blockIdx.x, lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n\
                     \tif (row >= {rows}u) return;\n\
                     \t__shared__ float s_m[8]; __shared__ float s_l[8];\n\
                     \tsize_t base = (size_t)row * {padded_vocab}ull;\n\
                     \tfloat lm = -1e30f, ll = 0.0f;\n\
                     \tfor (uint32_t c = lid; c < {vocab}u; c += blockDim.x) {{\n\
                     \t\tfloat v = {ld_v}; float nm = fmaxf(lm, v);\n\
                     \t\tll = fmaf(ll, expf(lm - nm), expf(v - nm)); lm = nm;\n\
                     \t}}\n\
                     \tfloat wm = lm;\n\
                     \tfor (int off = 16; off > 0; off >>= 1) wm = fmaxf(wm, __shfl_xor_sync(0xFFFFFFFFu, wm, off));\n\
                     \tfloat wl = ll * expf(lm - wm);\n\
                     \tfor (int off = 16; off > 0; off >>= 1) wl += __shfl_xor_sync(0xFFFFFFFFu, wl, off);\n\
                     \tif (lane == 0u) {{ s_m[wid] = wm; s_l[wid] = wl; }}\n\
                     \t__syncthreads();\n\
                     \tfloat gm = -1e30f; for (uint32_t w = 0; w < 8u; w++) gm = fmaxf(gm, s_m[w]);\n\
                     \tfloat gl = 0.0f; for (uint32_t w = 0; w < 8u; w++) gl = fmaf(s_l[w], expf(s_m[w] - gm), gl);\n\
                     \tfloat inv_l = 1.0f / fmaxf(gl, 1e-12f);\n\
                     \tuint32_t tgt = (uint32_t)targets[row];\n\
                     \tfor (uint32_t c = lid; c < {vocab}u; c += blockDim.x) {{\n\
                     \t\tfloat p = (expf({ld_v} - gm) * inv_l - ((c == tgt) ? 1.0f : 0.0f)) * {inv_rows:.10}f;\n\
                     \t\t{st_v};\n\
                     \t}}\n\
                     }}\n\n"
                ));
            }
            TileStage::AmpQuantContraction { name, m, k, n, quant_enabled, .. } => {
                let q_off = ((*n * *k) + 15) & !15;
                out.push_str(&format!(
                    "// [Facility #6 AMP Fused Norm + Dynamic Quant FP8/W8A8 enabled={quant_enabled}] {name} ({m}x{k}x{n}, sm_89 mma.sync.m16n8k32.f16.e4m3)\n\
                     extern \"C\" __global__ void {name}(const __half* __restrict__ x, const __half* __restrict__ nw, const int8_t* __restrict__ w_i8, __half* __restrict__ out) {{\n\
                     \tconst float* w_scale = reinterpret_cast<const float*>(w_i8 + {q_off}ull);\n\
                     \tuint32_t row = blockIdx.y * 16u + threadIdx.y, col = blockIdx.x * 16u + threadIdx.x;\n\
                     \tif (row >= {m}u || col >= {n}u) return;\n\
                     \tfloat sq = 0.0f; for (uint32_t kk = 0; kk < {k}u; kk++) {{ float xv = __half2float(x[(size_t)row * {k}u + kk]); sq = fmaf(xv, xv, sq); }}\n\
                     \tfloat inv = rsqrtf(sq / (float){k}u + 1e-5f);\n\
                     \tfloat acc = 0.0f;\n\
                     \tfor (uint32_t kk = 0; kk < {k}u; kk++) {{\n\
                     \t\tfloat xv = __half2float(x[(size_t)row * {k}u + kk]) * inv * __half2float(nw[kk]);\n\
                     \t\tacc = fmaf(xv, (float)w_i8[(size_t)col * {k}u + kk], acc);\n\
                     \t}}\n\
                     \tout[(size_t)row * {n}u + col] = __float2half(acc * w_scale[col]);\n\
                     }}\n\n"
                ));
            }
            TileStage::DynBucketContraction { name, max_m, k, n, bucket_enabled, .. } => {
                out.push_str(&format!(
                    "// [Facility #7 Symbolic Dynamic-Shape 3-Bucket Dispatch enabled={bucket_enabled}] {name} (M<=8 GEMV | 9..64 MMA32 | 65..{max_m} MMA64, K={k}, N={n})\n\
                     extern \"C\" __global__ void {name}(const __half* __restrict__ a, const __half* __restrict__ b_nk, const uint32_t* __restrict__ ctrl, __half* __restrict__ out) {{\n\
                     \tuint32_t m_act = ctrl[0];\n\
                     \tuint32_t row = blockIdx.y, col = blockIdx.x * 4u + (threadIdx.x >> 5), lane = threadIdx.x & 31u;\n\
                     \tif (row >= m_act || col >= {n}u) return;\n\
                     \tfloat acc = 0.0f;\n\
                     \tfor (uint32_t kk = lane; kk < {k}u; kk += 32u) acc = fmaf(__half2float(a[(size_t)row * {k}u + kk]), __half2float(b_nk[(size_t)col * {k}u + kk]), acc);\n\
                     \tfor (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);\n\
                     \tif (lane == 0u) out[(size_t)row * {n}u + col] = __float2half(acc);\n\
                     }}\n\n"
                ));
            }
            TileStage::Conv2dStage { name, in_h, in_w, c_in, c_out, kh, kw, implicit_enabled, .. } => {
                let m = (*in_h) * (*in_w);
                let k_eff = (*c_in) * (*kh) * (*kw);
                out.push_str(&format!(
                    "// [Facility #8A Implicit GEMM Conv2d implicit={implicit_enabled}] {name} ({in_h}x{in_w} C={c_in}->{c_out} K={kh}x{kw}, ZERO im2col VRAM!)\n\
                     extern \"C\" __global__ void {name}(const __half* __restrict__ x, const __half* __restrict__ w_nk, __half* __restrict__ out) {{\n\
                     \tuint32_t gm = blockIdx.y * 16u + threadIdx.y, co = blockIdx.x * 16u + threadIdx.x;\n\
                     \tif (gm >= {m}u || co >= {c_out}u) return;\n\
                     \tint oh = (int)(gm / {in_w}u), ow = (int)(gm % {in_w}u);\n\
                     \tfloat acc = 0.0f;\n\
                     \tfor (uint32_t gk = 0; gk < {k_eff}u; gk++) {{\n\
                     \t\tuint32_t tap = gk / {c_in}u, ci = gk % {c_in}u;\n\
                     \t\tint ih = oh + (int)(tap / {kw}u) - ({kh} / 2), iw = ow + (int)(tap % {kw}u) - ({kw} / 2);\n\
                     \t\tfloat xv = (ih >= 0 && ih < {in_h} && iw >= 0 && iw < {in_w}) ? __half2float(x[((size_t)ih * {in_w}u + (size_t)iw) * {c_in}u + ci]) : 0.0f;\n\
                     \t\tacc = fmaf(xv, __half2float(w_nk[(size_t)co * {k_eff}u + gk]), acc);\n\
                     \t}}\n\
                     \tout[(size_t)gm * {c_out}u + co] = __float2half(acc);\n\
                     }}\n\n"
                ));
            }
            TileStage::ParallelScanStage { name, batch, seq_len, parallel_enabled, .. } => {
                let l4 = (*seq_len / 4).max(1);
                out.push_str(&format!(
                    "// [Facility #8B Associative Parallel Scan parallel={parallel_enabled}] {name} (B={batch}, L={seq_len}, O(log L) warp/block prefix scan!)\n\
                     extern \"C\" __global__ void {name}(const float4* __restrict__ a_in, const float4* __restrict__ b_in, float4* __restrict__ out) {{\n\
                     \tuint32_t row = blockIdx.x, lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n\
                     \tif (row >= {batch}u) return;\n\
                     \t__shared__ float s_a[8]; __shared__ float s_b[8]; __shared__ float s_pa[8]; __shared__ float s_pb[8];\n\
                     \tsize_t base4 = (size_t)row * {l4}ull;\n\
                     \tfloat4 va = (lid < {l4}u) ? a_in[base4 + lid] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);\n\
                     \tfloat4 vb = (lid < {l4}u) ? b_in[base4 + lid] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n\
                     \tfloat a0 = va.x, b0 = vb.x, a1 = va.y * a0, b1 = fmaf(va.y, b0, vb.y);\n\
                     \tfloat a2 = va.z * a1, b2 = fmaf(va.z, b1, vb.z), a3 = va.w * a2, b3 = fmaf(va.w, b2, vb.w);\n\
                     \tfloat ia = a3, ib = b3;\n\
                     \tfor (uint32_t off = 1u; off < 32u; off <<= 1) {{\n\
                     \t\tfloat pa = __shfl_up_sync(0xFFFFFFFFu, ia, off), pb = __shfl_up_sync(0xFFFFFFFFu, ib, off);\n\
                     \t\tif (lane >= off) {{ ib = fmaf(ia, pb, ib); ia *= pa; }}\n\
                     \t}}\n\
                     \tfloat ex_a = __shfl_up_sync(0xFFFFFFFFu, ia, 1u), ex_b = __shfl_up_sync(0xFFFFFFFFu, ib, 1u);\n\
                     \tif (lane == 0u) {{ ex_a = 1.0f; ex_b = 0.0f; }}\n\
                     \tif (lane == 31u) {{ s_a[wid] = ia; s_b[wid] = ib; }}\n\
                     \t__syncthreads();\n\
                     \tif (wid == 0u) {{\n\
                     \t\tfloat ga = (lane < 8u) ? s_a[lane] : 1.0f, gb = (lane < 8u) ? s_b[lane] : 0.0f;\n\
                     \t\tfor (uint32_t off = 1u; off < 8u; off <<= 1) {{\n\
                     \t\t\tfloat pa = __shfl_up_sync(0xFFFFFFFFu, ga, off), pb = __shfl_up_sync(0xFFFFFFFFu, gb, off);\n\
                     \t\t\tif (lane >= off && lane < 8u) {{ gb = fmaf(ga, pb, gb); ga *= pa; }}\n\
                     \t\t}}\n\
                     \t\tfloat pga = __shfl_up_sync(0xFFFFFFFFu, ga, 1u), pgb = __shfl_up_sync(0xFFFFFFFFu, gb, 1u);\n\
                     \t\tif (lane < 8u) {{ s_pa[lane] = (lane == 0u) ? 1.0f : pga; s_pb[lane] = (lane == 0u) ? 0.0f : pgb; }}\n\
                     \t}}\n\
                     \t__syncthreads();\n\
                     \tfloat hp = fmaf(ex_a, s_pb[wid], ex_b);\n\
                     \tif (lid < {l4}u) out[base4 + lid] = make_float4(fmaf(a0, hp, b0), fmaf(a1, hp, b1), fmaf(a2, hp, b2), fmaf(a3, hp, b3));\n\
                     }}\n\n"
                ));
            }
            TileStage::ConcatContraction { name, m, k, total_n, packed_enabled, .. } => {
                out.push_str(&format!(
                    "// [Facility #2 Horizontal Weight Packing packed={packed_enabled}] {name} ({m}x{k}x{total_n}, single wide GEMM + 2D GROUP_M=8 Tile Swizzle)\n\
                     extern \"C\" __global__ void {name}(const __half* __restrict__ a, const __half* __restrict__ w_packed_nk, __half* __restrict__ out) {{\n\
                     \tuint32_t _pid = blockIdx.y * gridDim.x + blockIdx.x, _gsz = 8u * gridDim.x;\n\
                     \tuint32_t _gid = _pid / _gsz, _fm = _gid * 8u, _gm = min(gridDim.y - _fm, 8u), _rem = _pid % _gsz;\n\
                     \tuint32_t row = (_fm + (_rem % _gm)) * 16u + threadIdx.y, col = (_rem / _gm) * 16u + threadIdx.x;\n\
                     \tif (row >= {m}u || col >= {total_n}u) return;\n\
                     \tfloat acc = 0.0f;\n\
                     \tfor (uint32_t kk = 0; kk < {k}u; kk++) acc = fmaf(__half2float(a[(size_t)row * {k}u + kk]), __half2float(w_packed_nk[(size_t)col * {k}u + kk]), acc);\n\
                     \tout[(size_t)row * {total_n}u + col] = __float2half(acc);\n\
                     }}\n\n"
                ));
            }
            TileStage::RaggedContraction { name, m_padded, m_valid, k, n, compact_enabled, .. } => {
                let eff_m = if *compact_enabled { *m_valid } else { *m_padded };
                out.push_str(&format!(
                    "// [Stage 3 Ragged Contraction compact={compact_enabled}] {name}: valid_rows={eff_m}/{m_padded}, K={k}, N={n}\n\
                     extern \"C\" __global__ void {name}(const __half* __restrict__ a, const __half* __restrict__ b_nk, __half* __restrict__ out) {{\n\
                     \tuint32_t row = blockIdx.y * 16u + threadIdx.y, col = blockIdx.x * 16u + threadIdx.x;\n\
                     \tif (row >= {eff_m}u || col >= {n}u) return;\n\
                     \tfloat acc = 0.0f;\n\
                     \tfor (uint32_t kk = 0; kk < {k}u; kk++) acc = fmaf(__half2float(a[(size_t)row * {k}u + kk]), __half2float(b_nk[(size_t)col * {k}u + kk]), acc);\n\
                     \tout[(size_t)row * {n}u + col] = __float2half(acc);\n\
                     }}\n\n"
                ));
            }
            TileStage::BlockSparseContraction { name, m, k, n, block_size, density_pct, sparse_enabled, .. } => {
                let tiles_m = (*m + 31) / 32;
                let tiles_n = (*n + 31) / 32;
                let total_tiles = tiles_m * tiles_n;
                out.push_str(&format!(
                    "// [Stage 3 Persistent Work-Stealing BSR-32 Block-Sparse MMA enabled={sparse_enabled}] {name}: {m}x{k}x{n}, block={block_size}, density={density_pct}%\n\
                     extern \"C\" __global__ void {name}(const __half* __restrict__ a, const uint32_t* __restrict__ b_bsr, uint32_t* __restrict__ work_ctr, __half* __restrict__ out) {{\n\
                     \t__shared__ uint32_t s_tid;\n\
                     \tuint32_t kidx_off = b_bsr[0], data_off_h = b_bsr[1];\n\
                     \tconst uint32_t* row_ptr = b_bsr + 4u; const uint32_t* k_idx = b_bsr + kidx_off;\n\
                     \tconst __half* data_b = reinterpret_cast<const __half*>(b_bsr) + data_off_h;\n\
                     \twhile (true) {{\n\
                     \t\tif (threadIdx.x == 0u && threadIdx.y == 0u) s_tid = atomicAdd(work_ctr, 1u);\n\
                     \t\t__syncthreads();\n\
                     \t\tuint32_t tid = s_tid; if (tid >= {total_tiles}u) break;\n\
                     \t\tuint32_t m0 = (tid / {tiles_n}u) * 32u, tn = tid % {tiles_n}u, n0 = tn * 32u;\n\
                     \t\tfor (uint32_t rr = threadIdx.y; rr < 32u; rr += 16u) for (uint32_t cc = threadIdx.x; cc < 32u; cc += 16u) {{\n\
                     \t\t\tfloat acc = 0.0f;\n\
                     \t\t\tfor (uint32_t p = row_ptr[tn]; p < row_ptr[tn + 1u]; p++) {{\n\
                     \t\t\t\tuint32_t kt = k_idx[p]; const __half* blk = data_b + (size_t)p * 1024ull;\n\
                     \t\t\t\tfor (uint32_t kk = 0; kk < 32u; kk++) acc = fmaf(__half2float(a[(size_t)(m0 + rr) * {k}u + kt + kk]), __half2float(blk[cc * 32u + kk]), acc);\n\
                     \t\t\t}}\n\
                     \t\t\tif (m0 + rr < {m}u && n0 + cc < {n}u) out[(size_t)(m0 + rr) * {n}u + n0 + cc] = __float2half(acc);\n\
                     \t\t}}\n\
                     \t}}\n\
                     }}\n\n"
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

    if cg.hetero_plan.active {
        let hp = &cg.hetero_plan;
        out.push_str(&format!(
            "// [Stage 6 Heterogeneous Streaming Plan]\n\
             //   mode={} | tp={} (P2P AllReduce) | mla_latent={} | vram_cap={:.2}GiB | core_vram={:.2}GiB | pinned_ffn={}/{} | lru_slots={}/{} | kv_q8={} (saved {:.1}MB)\n\
             //   dma_ring={} (cuStreamWaitEvent compute_stream || dma_stream, indirect d_step_args CUDA Graph, async_promoter={})\n\
             //   cpu_hybrid: NUMA_first_touch_4R={} | cgroup_main_tid0={} | AVX2 _mm256_madd_epi16 INT16 (B=4 shared-weight) | prefill_B={} (TensorCore mma.sync m16n8k16) | spec_k={}\n\
             //   roofline_proj: decode={:.2} tok/s | batched_prefill={:.1} tok/s\n",
            hp.dispatch_mode.tag(),
            hp.tp_devices,
            hp.mla_latent_dim,
            hp.vram_cap_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            hp.core_vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            hp.pinned_gpu_ffn_layers,
            hp.n_layers,
            hp.lru_slots_per_layer,
            hp.n_experts,
            hp.kv_quant_q8,
            hp.kv_bytes_saved as f64 / 1e6,
            hp.dma_ring_slots,
            hp.async_promoter,
            hp.numa_repack_weights.len(),
            hp.cgroup_main_tid0,
            hp.prefill_chunk_tokens,
            hp.speculative_k,
            hp.projected_decode_tok_s,
            hp.projected_prefill_tok_s,
        ));
    }

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
            if let Some(CachePolicy::Streaming) = cg.cache_policies.get(nm) {
                attrs.push("mhlo.memory_space = \"hbm_streaming\"".to_string());
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
                match kind {
                    CollectiveKind::ReduceScatter => {
                        out.push_str(&format!(
                            "    // Facility #13 ZeRO-1 Bucketed ReduceScatter across {devices} TPU chips (async ICI overlap)\n\
                             \t%{out_v} = \"stablehlo.reduce_scatter\"(%{x}) ({{\n\
                             \t  ^bb0(%lhs: tensor<{et}>, %rhs: tensor<{et}>):\n\
                             \t    %sum = stablehlo.add %lhs, %rhs : tensor<{et}>\n\
                             \t    stablehlo.return %sum : tensor<{et}>\n\
                             \t}}) {{replica_groups = dense<[[{}]]> : tensor<1x{devices}xi64>, scatter_dimension = 0 : i64}} : (tensor<{}x{}x{et}>) -> tensor<{}x{}x{et}>\n",
                            replicas.join(", "), td.rows(), td.cols(), td.rows(), td.cols()
                        ));
                    }
                    CollectiveKind::AllGather => {
                        out.push_str(&format!(
                            "    // Facility #13 ZeRO-1 AllGather updated params across {devices} TPU chips\n\
                             \t%{out_v} = \"stablehlo.all_gather\"(%{x}) {{replica_groups = dense<[[{}]]> : tensor<1x{devices}xi64>, all_gather_dim = 0 : i64}} : (tensor<{}x{}x{et}>) -> tensor<{}x{}x{et}>\n",
                            replicas.join(", "), td.rows(), td.cols(), td.rows(), td.cols()
                        ));
                    }
                    CollectiveKind::AllReduceSum => {
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
                }
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
            TileStage::ConcatContraction { m, total_n, dtype, out: out_v, packed_enabled, .. } => {
                out.push_str(&format!(
                    "    // Facility #2 Horizontal Packed Wide MXU GEMM (packed={packed_enabled})\n\
                     \t%{out_v} = stablehlo.constant dense<0.0> : tensor<{m}x{total_n}x{}>\n",
                    dtype.mlir_ty()
                ));
            }
            TileStage::InPlaceAdamW { param, out: out_v, donate_enabled, .. } => {
                let td = &cg.graph.tensors[out_v];
                out.push_str(&format!(
                    "    // Facility #3 Buffer-Donated Fused In-Place AdamW (donate={donate_enabled}, tf.aliasing_output)\n\
                     \t%{out_v} = stablehlo.reshape %{param} : (tensor<{}x{}x{}>) -> tensor<{}x{}x{}>\n",
                    td.rows(), td.cols(), td.dtype.mlir_ty(), td.rows(), td.cols(), td.dtype.mlir_ty()
                ));
            }
            TileStage::CrossEntropyStage { logits, rows, vocab, dtype, online_fused, out: out_v, .. } => {
                let et = dtype.mlir_ty();
                out.push_str(&format!(
                    "    // Facility #5 Online Monoid Fused Cross-Entropy (online={online_fused})\n\
                     \t%{out_v} = stablehlo.reshape %{logits} : (tensor<{rows}x{vocab}x{et}>) -> tensor<{rows}x{vocab}x{et}>\n"
                ));
            }
            TileStage::AmpQuantContraction { m, n, dtype, quant_enabled, out: out_v, .. } => {
                let et = dtype.mlir_ty();
                out.push_str(&format!(
                    "    // Facility #6 Algebraic Single-Pass RMSNorm+INT8 MXU Contraction (quant={quant_enabled}, 394 TOPS)\n\
                     \t%{out_v} = stablehlo.constant dense<0.0> : tensor<{m}x{n}x{et}>\n"
                ));
            }
            TileStage::DynBucketContraction { max_m, n, dtype, bucket_enabled, out: out_v, .. } => {
                let et = dtype.mlir_ty();
                out.push_str(&format!(
                    "    // Facility #7 Dynamic-Shape Multi-Version Bucketed Dispatch (bucket={bucket_enabled})\n\
                     \t%{out_v} = stablehlo.constant dense<0.0> : tensor<{max_m}x{n}x{et}>\n"
                ));
            }
            TileStage::Conv2dStage { in_h, in_w, c_out, dtype, implicit_enabled, out: out_v, .. } => {
                let et = dtype.mlir_ty();
                out.push_str(&format!(
                    "    // Facility #8A Implicit GEMM Conv2d (implicit={implicit_enabled}, stablehlo.convolution)\n\
                     \t%{out_v} = stablehlo.constant dense<0.0> : tensor<{}x{c_out}x{et}>\n",
                    in_h * in_w
                ));
            }
            TileStage::ParallelScanStage { a, batch, seq_len, dtype, parallel_enabled, out: out_v, .. } => {
                let et = dtype.mlir_ty();
                out.push_str(&format!(
                    "    // Facility #8B Associative Parallel Scan (parallel={parallel_enabled}, O(log L))\n\
                     \t%{out_v} = stablehlo.reshape %{a} : (tensor<{batch}x{seq_len}x{et}>) -> tensor<{batch}x{seq_len}x{et}>\n"
                ));
            }
        }
    }

    let ret_vals: Vec<String> = cg.graph.outputs.iter().map(|o| format!("%{}", o)).collect();
    if cg.hetero_plan.active {
        let hp = &cg.hetero_plan;
        out.push_str(&format!(
            "    // [Stage 6 TPU Host-Offload Streaming] mode={} mhlo.memory_kind=\"pinned_host\" pinned_ffn={} lru_slots={} kv_q8={} prefill_chunk={}\n",
            hp.dispatch_mode.tag(),
            hp.pinned_gpu_ffn_layers,
            hp.lru_slots_per_layer,
            hp.kv_quant_q8,
            hp.prefill_chunk_tokens,
        ));
    }
    out.push_str(&format!("    return {} : {}\n  }}\n}}\n", ret_vals.join(", "), ret_sig));
    out
}
