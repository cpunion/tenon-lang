// std/gpu/metal/moe_stream_shim.m — Universal Hard-Memory-Capped Native Metal Multi-Shard Streaming Engine
//
// Supports BOTH Oversized Sparse MoE (GLM-4.5-Air 106B-A12B, OLMoE, Qwen3-MoE, Mixtral)
// AND Oversized Dense LLMs (Llama-3.3-70B, Mistral-Large-123B) strictly inside bounded RAM/VRAM (e.g. <= 16 GiB)!
//   - Multi-Shard GGUF v2/v3 (`-00001-of-0000N.gguf` auto-discovery) + `F_NOCACHE` raw NVMe SSD I/O
//   - All 8 GGML quant formats: F32, F16, Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K
//   - Zero-VRAM On-Demand Embedding Row `pread()` (saves 350-563 MB RAM!)
//   - Sparse MoE Mode: Per-Layer GPU LRU Expert Cache + Parallel NVMe `pread()` + Intra-Layer (SharedExp+Hit)||Miss Overlap + L+1 Prefetch
//   - Dense Mode: 2-Layer Ping-Pong Double-Buffer Ring Pool (GPU Layer L || Parallel Chunked NVMe Layer L+1, < 2.5 GiB total RAM!)
//   - Kernel-Verified macOS `task_vm_info.phys_footprint` + `rusage.ru_maxrss` Memory Telemetry!

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define MOE_MAX_MODELS  8
#define MOE_MAX_SHARDS  16
#define MOE_MAX_LAYERS  128
#define MOE_MAX_EXPERTS 256
#define MOE_MAX_TOPK    16
#define MOE_MAX_SEQ     1024

// GGML Quantization Types
#define TN_GGML_F32   0
#define TN_GGML_F16   1
#define TN_GGML_Q4_0  2
#define TN_GGML_Q4_1  3
#define TN_GGML_Q8_0  8
#define TN_GGML_Q4_K  12
#define TN_GGML_Q5_K  13
#define TN_GGML_Q6_K  14

// GGUF Metadata Value Types
enum {
    GGUF_T_U8 = 0, GGUF_T_I8 = 1, GGUF_T_U16 = 2, GGUF_T_I16 = 3,
    GGUF_T_U32 = 4, GGUF_T_I32 = 5, GGUF_T_F32 = 6, GGUF_T_BOOL = 7,
    GGUF_T_STR = 8, GGUF_T_ARR = 9, GGUF_T_U64 = 10, GGUF_T_I64 = 11, GGUF_T_F64 = 12
};

typedef struct {
    char name[128];
    uint32_t n_dims;
    uint64_t ne[4];
    uint32_t ggml_type;
    uint64_t rel_off;
    uint64_t file_off;
    uint64_t byte_size;
    int shard_idx;
} MoeTensorInfo;

typedef struct {
    id<MTLBuffer> buf;
    uint64_t offset;
    uint32_t ggml_type;
    uint64_t ne0, ne1, ne2;
    uint64_t file_off;
    uint64_t byte_size;
    int shard_idx;
} MoeWeightRef;

typedef struct {
    int is_dense; // 1 = Dense FFN layer (Llama-3.3 all layers, GLM-4.5 blk.0), 0 = MoE routed layer
    int is_pinned_dense; // For Dense Streaming mode: 1 = weights permanently resident in GPU VRAM, 0 = streamed via Ping-Pong!

    MoeWeightRef attn_norm;
    MoeWeightRef wq, wk, wv, wo;
    MoeWeightRef bq, bk, bv; // Optional Q/K/V biases (GLM-4.5 / Qwen2)
    int has_qkv_bias;
    MoeWeightRef q_norm, k_norm;
    int has_qk_norm;
    int qk_norm_per_head; // 0=full vector (OLMoE), 1=per-head d_head (Qwen3)

    MoeWeightRef ffn_norm;
    float* ffn_norm_host;

    // Dense FFN weights (used when is_dense == 1)
    MoeWeightRef ffn_gate, ffn_up, ffn_down;

    // MoE Router + Routed Experts + Optional Shared Expert (used when is_dense == 0)
    MoeWeightRef router;
    float* router_host;
    MoeWeightRef exp_bias;    // Optional exp_probs_b.bias (GLM-4.5 / DeepSeek-V3)
    float* exp_bias_host;
    int has_exp_bias;
    MoeWeightRef gate_exps;
    MoeWeightRef up_exps;
    MoeWeightRef down_exps;
    int has_shared_exp;
    MoeWeightRef sh_gate, sh_up, sh_down;
    uint64_t gate_exp_bytes;
    uint64_t up_exp_bytes;
    uint64_t down_exp_bytes;
    uint64_t slot_stride_bytes;

    // Mode 1 MoE Per-Layer GPU LRU Expert Cache
    id<MTLBuffer> cache_buf;
    int exp_to_slot[MOE_MAX_EXPERTS];
    int slot_to_exp[MOE_MAX_EXPERTS];
    uint64_t slot_lru[MOE_MAX_EXPERTS];
    int slot_prefetched[MOE_MAX_EXPERTS];

    // KV Cache (FP16) [MOE_MAX_SEQ, n_kv * d_head]
    id<MTLBuffer> k_cache;
    id<MTLBuffer> v_cache;
} MoeLayer;

typedef struct {
    int active;
    int n_shards;
    int fds[MOE_MAX_SHARDS];
    size_t shard_sizes[MOE_MAX_SHARDS];
    uint64_t total_disk_bytes;
    void* mmap_ptrs[MOE_MAX_SHARDS];
    size_t mmap_lens[MOE_MAX_SHARDS];
    id<MTLBuffer> mmap_mtl_bufs[MOE_MAX_SHARDS];

    int use_nocache;
    int enable_overlap;
    int resident_mode;
    int dense_model; // 1 = pure Dense model (n_experts == 0), 0 = MoE model (n_experts > 0)

    char arch[64];
    int n_layers;
    int nextn_layers;
    int d_model;
    int d_ff_dense; // Dense FFN dim (e.g. 28672 for Llama-3.3-70B, 10944 for GLM-4.5 blk.0)
    int d_ff_exp;   // Expert FFN dim (e.g. 1408 for GLM-4.5, 1024 for OLMoE)
    int n_heads;
    int n_kv;
    int d_head;
    int rope_dim;
    int rope_neox;
    int n_experts;
    int top_k;
    int vocab_size;
    int bos_id;
    int eos_id;
    int gating_func; // 0/1 = Softmax, 2 = Sigmoid (GLM-4.5 / DeepSeek-V3)
    int expert_weights_norm;
    float expert_weights_scale;
    float rope_theta;
    float rms_eps;
    int cache_slots_per_layer; // For MoE: LRU slots/layer; For Dense: pinned prefix layers

    char** vocab_tokens;

    MoeWeightRef tok_embd;
    MoeWeightRef out_norm;
    MoeWeightRef lm_head;
    MoeWeightRef rope_freqs;
    int has_rope_freqs;
    MoeLayer layers[MOE_MAX_LAYERS];

    // Dense Mode Streaming Pools:
    //   - Ultra-low (<2.5 GiB, cache_slots==0): 2-Layer Ping-Pong (streams Attn+FFN, ~0.92 GiB pool)
    //   - Role-Aware (cache_slots>0, <=16 GiB): Pins ALL Attention on GPU + N FFNs + 4-Slot Continuous FFN Ring (~1.6 GiB)!
    int dense_pin_all_attn;
    id<MTLBuffer> dense_ping[4];
    uint64_t dense_ping_stride;
    uint64_t dense_off_wq, dense_off_wk, dense_off_wv, dense_off_wo;
    uint64_t dense_off_gate, dense_off_up, dense_off_down;
    dispatch_group_t dense_io_grp;
    dispatch_queue_t ring_prod_q;

    // Working activations in GPU Unified Memory
    id<MTLBuffer> buf_emb_row; // Tiny 64KB buffer for zero-VRAM on-demand `token_embd` row pread!
    id<MTLBuffer> buf_x;       // [d_model] F32
    id<MTLBuffer> buf_xn;      // [d_model] F32
    id<MTLBuffer> buf_q;       // [n_heads * d_head] F32
    id<MTLBuffer> buf_k;       // [n_kv * d_head] F32
    id<MTLBuffer> buf_v;       // [n_kv * d_head] F32
    id<MTLBuffer> buf_ctx;     // [n_heads * d_head] F32
    id<MTLBuffer> buf_rlogits; // [n_experts] F32
    id<MTLBuffer> buf_eidx;    // [top_k] uint32
    id<MTLBuffer> buf_ewt;     // [top_k] F32
    id<MTLBuffer> buf_h;       // [max(top_k * d_ff_exp, d_ff_dense)] F32
    id<MTLBuffer> buf_sh_g;    // [max(d_ff_exp, d_ff_dense)] F32
    id<MTLBuffer> buf_sh_u;    // [max(d_ff_exp, d_ff_dense)] F32
    id<MTLBuffer> buf_logits;  // [vocab_size] F32
    id<MTLBuffer> buf_argmax;  // [2] uint32

    // Stats & counters
    uint64_t lru_clock;
    int64_t core_vram_bytes;
    int64_t cache_vram_bytes;
    int64_t total_streamable_bytes;
    int64_t cache_hits;
    int64_t cache_misses;
    int64_t prefetch_hits;
    int64_t ssd_bytes_read;
    double last_total_ms;
    double last_ssd_ms;
    double last_gpu_ms;
    dispatch_group_t prefetch_grp;
    dispatch_queue_t io_queue;
} MoeModel;

static MoeModel g_moe[MOE_MAX_MODELS];
static id<MTLDevice> g_moe_dev = nil;
static id<MTLCommandQueue> g_moe_q = nil;
static id<MTLLibrary> g_moe_lib = nil;
static id<MTLComputePipelineState> g_pso_embed = nil;
static id<MTLComputePipelineState> g_pso_rmsnorm = nil;
static id<MTLComputePipelineState> g_pso_add_bias = nil;
static id<MTLComputePipelineState> g_pso_swiglu_mul = nil;
static id<MTLComputePipelineState> g_pso_gemv_q4 = nil;
static id<MTLComputePipelineState> g_pso_gemv_q4_1 = nil;
static id<MTLComputePipelineState> g_pso_gemv_q8 = nil;
static id<MTLComputePipelineState> g_pso_gemv_q4k = nil;
static id<MTLComputePipelineState> g_pso_gemv_q5k = nil;
static id<MTLComputePipelineState> g_pso_gemv_q6k = nil;
static id<MTLComputePipelineState> g_pso_gemv_f16 = nil;
static id<MTLComputePipelineState> g_pso_gemv_f32 = nil;
static id<MTLComputePipelineState> g_pso_dense_gu_q4 = nil;
static id<MTLComputePipelineState> g_pso_rope_kv = nil;
static id<MTLComputePipelineState> g_pso_flash_dec = nil;
static id<MTLComputePipelineState> g_pso_router = nil;
static id<MTLComputePipelineState> g_pso_moe_gu_q4 = nil;
static id<MTLComputePipelineState> g_pso_moe_down_q4 = nil;
static id<MTLComputePipelineState> g_pso_moe_down_q4_1 = nil;
static id<MTLComputePipelineState> g_pso_moe_slot_gu_q4 = nil;
static id<MTLComputePipelineState> g_pso_moe_slot_down_q4 = nil;
static id<MTLComputePipelineState> g_pso_moe_slot_down_q4_1 = nil;
static id<MTLComputePipelineState> g_pso_argmax = nil;

static double moe_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static int64_t get_os_phys_footprint_bytes(void) {
    task_vm_info_data_t vm_info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vm_info, &count) == KERN_SUCCESS) {
        return (int64_t)vm_info.phys_footprint;
    }
    return 0;
}

static int64_t get_os_peak_rss_bytes(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        return (int64_t)ru.ru_maxrss; // macOS reports bytes directly!
    }
    return 0;
}

static uint64_t ggml_type_bytes(uint32_t t, uint64_t n_elems) {
    switch (t) {
        case TN_GGML_F32:  return n_elems * 4ull;
        case TN_GGML_F16:  return n_elems * 2ull;
        case TN_GGML_Q4_0: return (n_elems / 32ull) * 18ull;
        case TN_GGML_Q4_1: return (n_elems / 32ull) * 20ull;
        case TN_GGML_Q8_0: return (n_elems / 32ull) * 34ull;
        case TN_GGML_Q4_K: return (n_elems / 256ull) * 144ull;
        case TN_GGML_Q5_K: return (n_elems / 256ull) * 176ull;
        case TN_GGML_Q6_K: return (n_elems / 256ull) * 210ull;
        default:           return n_elems * 2ull;
    }
}

static NSString* const kMoeMslSource = @R"(
#include <metal_stdlib>
using namespace metal;

// Load 32 floats into 8 float4 registers ONCE per lane -> reuse across 4 rows (4x fewer activation loads!)
static inline void load_x32_reg(const device float* x, thread float4 xv[8]) {
    const device float4* x4 = (const device float4*)x;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) xv[i] = x4[i];
}

// Fast vectorized GGML Q4_0 block dot against register-cached `xv[8]` (18 bytes / 32 elems)
static inline float dot_q4_0_reg(const device uchar* blk, thread const float4 xv[8]) {
    float d = float(*(const device half*)blk);
    const device uchar* qs = blk + 2u;
    float4 s0 = float4(0.0f), s1 = float4(0.0f);
    #pragma unroll
    for (uint g = 0; g < 4u; g++) {
        uint off = g * 4u;
        uchar b0 = qs[off], b1 = qs[off + 1u], b2 = qs[off + 2u], b3 = qs[off + 3u];
        float4 lo = float4(float(int(b0 & 0xFu) - 8), float(int(b1 & 0xFu) - 8),
                           float(int(b2 & 0xFu) - 8), float(int(b3 & 0xFu) - 8));
        float4 hi = float4(float(int(b0 >> 4) - 8),   float(int(b1 >> 4) - 8),
                           float(int(b2 >> 4) - 8),   float(int(b3 >> 4) - 8));
        s0 = fma(lo, xv[g],      s0);
        s1 = fma(hi, xv[g + 4u], s1);
    }
    float4 t = s0 + s1;
    return d * ((t.x + t.y) + (t.z + t.w));
}

// Fast vectorized GGML Q4_1 block dot (20 bytes / 32 elems: half d + half m + uchar qs[16], val = d*q + m)
static inline float dot_q4_1_reg(const device uchar* blk, thread const float4 xv[8], float sum_x) {
    float d = float(*(const device half*)(blk));
    float m = float(*(const device half*)(blk + 2u));
    const device uchar* qs = blk + 4u;
    float4 s0 = float4(0.0f), s1 = float4(0.0f);
    #pragma unroll
    for (uint g = 0; g < 4u; g++) {
        uint off = g * 4u;
        uchar b0 = qs[off], b1 = qs[off + 1u], b2 = qs[off + 2u], b3 = qs[off + 3u];
        float4 lo = float4(float(b0 & 0xFu), float(b1 & 0xFu), float(b2 & 0xFu), float(b3 & 0xFu));
        float4 hi = float4(float(b0 >> 4),   float(b1 >> 4),   float(b2 >> 4),   float(b3 >> 4));
        s0 = fma(lo, xv[g],      s0);
        s1 = fma(hi, xv[g + 4u], s1);
    }
    float4 t = s0 + s1;
    return fma(d, (t.x + t.y) + (t.z + t.w), m * sum_x);
}

static inline float sum_x32_reg(thread const float4 xv[8]) {
    float4 s = (xv[0] + xv[1]) + (xv[2] + xv[3]) + (xv[4] + xv[5]) + (xv[6] + xv[7]);
    return (s.x + s.y) + (s.z + s.w);
}

// Exact GGML Q8_0 block (34 bytes / 32 elems: half d + 32 int8)
static inline float dot_q8_0_reg(const device uchar* blk, thread const float4 xv[8]) {
    float d = float(*(const device half*)blk);
    const device char* qs = (const device char*)(blk + 2u);
    float acc = 0.0f;
    #pragma unroll
    for (uint g = 0; g < 8u; g++) {
        uint j = g * 4u;
        float4 qv = float4(float(qs[j]), float(qs[j+1]), float(qs[j+2]), float(qs[j+3]));
        acc += dot(qv, xv[g]);
    }
    return d * acc;
}

// Exact GGML Q4_K block (144 bytes / 256 elems)
static inline float dot_q4_k_block(const device uchar* blk, const device float* x) {
    float d    = float(*(const device half*)(blk));
    float dmin = float(*(const device half*)(blk + 2u));
    const device uchar* sc = blk + 4u;
    const device uchar* qs = blk + 16u;
    float sum = 0.0f;
    uint is = 0u;
    for (uint j = 0; j < 256u; j += 64u) {
        uchar sc0, m0, sc1, m1;
        if (is < 4u) {
            sc0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u;
            sc1 = sc[is + 1u] & 63u; m1 = sc[is + 5u] & 63u;
        } else {
            sc0 = (sc[is + 4u] & 0xFu) | ((sc[is - 4u] >> 6) << 4);
            m0  = (sc[is + 4u] >> 4)   | ((sc[is]      >> 6) << 4);
            sc1 = (sc[is + 5u] & 0xFu) | ((sc[is - 3u] >> 6) << 4);
            m1  = (sc[is + 5u] >> 4)   | ((sc[is + 1u] >> 6) << 4);
        }
        float d1 = d * float(sc0), dm1 = dmin * float(m0);
        float d2 = d * float(sc1), dm2 = dmin * float(m1);
        const device uchar* q = qs + (j >> 1);
        for (uint l = 0; l < 32u; l++) {
            sum = fma(d1 * float(q[l] & 0xFu) - dm1, x[j + l], sum);
            sum = fma(d2 * float(q[l] >> 4)   - dm2, x[j + 32u + l], sum);
        }
        is += 2u;
    }
    return sum;
}

// Exact GGML Q5_K quarter-block (176 bytes / 256 elems: half d, half dmin, scales[12], qh[32], qs[128])
static inline float dot_q5_k_quarter(const device uchar* blk, const device float* x, uint qtr) {
    float d    = float(*(const device half*)(blk));
    float dmin = float(*(const device half*)(blk + 2u));
    const device uchar* sc = blk + 4u;
    const device uchar* qh = blk + 16u;
    const device uchar* qs = blk + 48u;
    uint l0 = qtr * 8u;
    float sum = 0.0f;
    uint is = 0u;
    uint u1 = 1u, u2 = 2u;
    #pragma unroll
    for (uint j = 0; j < 256u; j += 64u) {
        uchar sc0, m0, sc1, m1;
        if (is < 4u) {
            sc0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u;
            sc1 = sc[is + 1u] & 63u; m1 = sc[is + 5u] & 63u;
        } else {
            sc0 = (sc[is + 4u] & 0xFu) | ((sc[is - 4u] >> 6) << 4);
            m0  = (sc[is + 4u] >> 4)   | ((sc[is]      >> 6) << 4);
            sc1 = (sc[is + 5u] & 0xFu) | ((sc[is - 3u] >> 6) << 4);
            m1  = (sc[is + 5u] >> 4)   | ((sc[is + 1u] >> 6) << 4);
        }
        float d1 = d * float(sc0), dm1 = dmin * float(m0);
        float d2 = d * float(sc1), dm2 = dmin * float(m1);
        const device uchar* q = qs + (j >> 1);
        float p0 = 0.0f, p1 = 0.0f, sx0 = 0.0f, sx1 = 0.0f;
        #pragma unroll
        for (uint l = l0; l < l0 + 8u; l++) {
            uchar qv = q[l];
            uchar hv = qh[l];
            float qlo = float((qv & 0xFu) + ((hv & u1) ? 16u : 0u));
            float qhi = float((qv >> 4)   + ((hv & u2) ? 16u : 0u));
            float x0 = x[j + l], x1 = x[j + 32u + l];
            p0 = fma(qlo, x0, p0); sx0 += x0;
            p1 = fma(qhi, x1, p1); sx1 += x1;
        }
        sum += fma(d1, p0, -dm1 * sx0) + fma(d2, p1, -dm2 * sx1);
        is += 2u; u1 <<= 2; u2 <<= 2;
    }
    return sum;
}

// Exact GGML Q6_K quarter-block (210 bytes / 256 elems, 4 lanes/superblock = 100% SIMD occupancy!)
static inline float dot_q6_k_quarter(const device uchar* blk, const device float* x, uint qtr) {
    const device uchar* ql = blk;
    const device uchar* qh = blk + 128u;
    const device char*  sc = (const device char*)(blk + 192u);
    float d = float(*(const device half*)(blk + 208u));
    uint l0 = qtr * 8u;
    uint is = l0 >> 4;
    float sum = 0.0f;
    #pragma unroll
    for (uint n = 0; n < 256u; n += 128u) {
        const device uchar* qlp = ql + (n >> 1);
        const device uchar* qhp = qh + (n >> 2);
        const device char*  scp = sc + (n >> 4);
        const device float* xp  = x + n;
        float s0 = float(scp[is + 0]), s2 = float(scp[is + 2]);
        float s4 = float(scp[is + 4]), s6 = float(scp[is + 6]);
        float p0 = 0.0f, p2 = 0.0f, p4 = 0.0f, p6 = 0.0f;
        #pragma unroll
        for (uint l = l0; l < l0 + 8u; l++) {
            uchar ql_a = qlp[l], ql_b = qlp[l + 32u], qhv = qhp[l];
            int q1 = int((ql_a & 0xFu) | (((qhv >> 0) & 3u) << 4)) - 32;
            int q2 = int((ql_b & 0xFu) | (((qhv >> 2) & 3u) << 4)) - 32;
            int q3 = int((ql_a >> 4)   | (((qhv >> 4) & 3u) << 4)) - 32;
            int q4 = int((ql_b >> 4)   | (((qhv >> 6) & 3u) << 4)) - 32;
            p0 = fma(float(q1), xp[l + 0u],  p0);
            p2 = fma(float(q2), xp[l + 32u], p2);
            p4 = fma(float(q3), xp[l + 64u], p4);
            p6 = fma(float(q4), xp[l + 96u], p6);
        }
        sum += s0 * p0 + s2 * p2 + s4 * p4 + s6 * p6;
    }
    return d * sum;
}

struct MoeEmbedArgs { uint tok_id; uint d_model; uint ggml_type; };

kernel void moe_embed_lookup(
    const device uchar* emb [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant MoeEmbedArgs& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.d_model) return;
    if (p.ggml_type == 0u) {
        out[gid] = ((const device float*)emb)[(ulong)p.tok_id * p.d_model + gid];
    } else if (p.ggml_type == 1u) {
        out[gid] = float(((const device half*)emb)[(ulong)p.tok_id * p.d_model + gid]);
    } else if (p.ggml_type == 2u) {
        ulong row_bytes = (ulong)(p.d_model / 32u) * 18ul;
        uint blk = gid >> 5, sub = gid & 31u;
        const device uchar* bp = emb + (ulong)p.tok_id * row_bytes + (ulong)blk * 18ul;
        float d = float(*(const device half*)bp);
        uchar byte_v = bp[2u + (sub & 15u)];
        int q = (sub < 16u) ? int(byte_v & 0xFu) - 8 : int(byte_v >> 4) - 8;
        out[gid] = d * float(q);
    } else if (p.ggml_type == 3u) {
        ulong row_bytes = (ulong)(p.d_model / 32u) * 20ul;
        uint blk = gid >> 5, sub = gid & 31u;
        const device uchar* bp = emb + (ulong)p.tok_id * row_bytes + (ulong)blk * 20ul;
        float d = float(*(const device half*)bp);
        float m = float(*(const device half*)(bp + 2u));
        uchar byte_v = bp[4u + (sub & 15u)];
        uint q = (sub < 16u) ? (byte_v & 0xFu) : (byte_v >> 4);
        out[gid] = fma(d, float(q), m);
    } else if (p.ggml_type == 8u) {
        ulong row_bytes = (ulong)(p.d_model / 32u) * 34ul;
        uint blk = gid >> 5, sub = gid & 31u;
        const device uchar* bp = emb + (ulong)p.tok_id * row_bytes + (ulong)blk * 34ul;
        float d = float(*(const device half*)bp);
        out[gid] = d * float(((const device char*)(bp + 2u))[sub]);
    } else if (p.ggml_type == 14u) {
        ulong row_bytes = (ulong)(p.d_model / 256u) * 210ul;
        uint blk = gid >> 8, rem = gid & 255u;
        const device uchar* bp = emb + (ulong)p.tok_id * row_bytes + (ulong)blk * 210ul;
        uint n = (rem >= 128u) ? 128u : 0u;
        uint r = rem - n;
        uint qpart = r >> 5, l = r & 31u;
        const device uchar* qlp = bp + (n >> 1);
        const device uchar* qhp = bp + 128u + (n >> 2);
        const device char*  scp = (const device char*)(bp + 192u) + (n >> 4);
        float d = float(*(const device half*)(bp + 208u));
        uint is = l >> 4;
        uchar lo = (qpart == 0u) ? (qlp[l] & 0xFu) : (qpart == 1u) ? (qlp[l + 32u] & 0xFu) : (qpart == 2u) ? (qlp[l] >> 4) : (qlp[l + 32u] >> 4);
        uchar hi = (qhp[l] >> (qpart * 2u)) & 3u;
        int q = int(lo | (hi << 4)) - 32;
        out[gid] = d * float(scp[is + qpart * 2u]) * float(q);
    }
}

struct MoeNormArgs { uint dim; uint n_groups; float eps; uint shared_w; };

kernel void moe_rmsnorm(
    const device float* x [[buffer(0)]],
    const device float* w [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant MoeNormArgs& p [[buffer(3)]],
    uint g [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (g >= p.n_groups) return;
    threadgroup float s_sq[8];
    uint base = g * p.dim;
    float lsq = 0.0f;
    for (uint i = lid; i < p.dim; i += 256u) {
        float v = x[base + i];
        lsq = fma(v, v, lsq);
    }
    lsq = simd_sum(lsq);
    if (lane == 0u) s_sq[sg] = lsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < 8u; s++) tot += s_sq[s];
    float inv = rsqrt(tot / float(p.dim) + p.eps);
    uint wbase = p.shared_w ? 0u : base;
    for (uint i = lid; i < p.dim; i += 256u) {
        out[base + i] = x[base + i] * inv * w[wbase + i];
    }
}

kernel void moe_add_bias(
    device float* x [[buffer(0)]],
    const device float* bias [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) x[gid] += bias[gid];
}

kernel void moe_swiglu_mul(
    device float* g [[buffer(0)]],
    const device float* u [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        float gv = g[gid];
        g[gid] = (gv / (1.0f + exp(-gv))) * u[gid];
    }
}

struct MoeGemvArgs { uint out_dim; uint in_dim; uint add_res; };

// 4-row-per-SG register-reused Q4_0 GEMV (16 rows/TG)
kernel void moe_gemv_q4(
    const device uchar* w [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 4u + sg) * 4u;
    if (row0 >= p.out_dim) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 18ul;
    const device uchar* wp0 = w + (ulong)row0 * rb;
    const device uchar* wp1 = (row0 + 1u < p.out_dim) ? (wp0 + rb) : wp0;
    const device uchar* wp2 = (row0 + 2u < p.out_dim) ? (wp0 + rb * 2ul) : wp0;
    const device uchar* wp3 = (row0 + 3u < p.out_dim) ? (wp0 + rb * 3ul) : wp0;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        thread float4 xv[8];
        load_x32_reg(x + (b << 5), xv);
        ulong boff = (ulong)b * 18ul;
        a0 += dot_q4_0_reg(wp0 + boff, xv);
        a1 += dot_q4_0_reg(wp1 + boff, xv);
        a2 += dot_q4_0_reg(wp2 + boff, xv);
        a3 += dot_q4_0_reg(wp3 + boff, xv);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row0 + 1u < p.out_dim) y[row0 + 1u] = p.add_res ? (y[row0 + 1u] + a1) : a1;
        if (row0 + 2u < p.out_dim) y[row0 + 2u] = p.add_res ? (y[row0 + 2u] + a2) : a2;
        if (row0 + 3u < p.out_dim) y[row0 + 3u] = p.add_res ? (y[row0 + 3u] + a3) : a3;
    }
}

// 4-row-per-SG register-reused Q4_1 GEMV (16 rows/TG)
kernel void moe_gemv_q4_1(
    const device uchar* w [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 4u + sg) * 4u;
    if (row0 >= p.out_dim) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 20ul;
    const device uchar* wp0 = w + (ulong)row0 * rb;
    const device uchar* wp1 = (row0 + 1u < p.out_dim) ? (wp0 + rb) : wp0;
    const device uchar* wp2 = (row0 + 2u < p.out_dim) ? (wp0 + rb * 2ul) : wp0;
    const device uchar* wp3 = (row0 + 3u < p.out_dim) ? (wp0 + rb * 3ul) : wp0;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        thread float4 xv[8];
        load_x32_reg(x + (b << 5), xv);
        float sx = sum_x32_reg(xv);
        ulong boff = (ulong)b * 20ul;
        a0 += dot_q4_1_reg(wp0 + boff, xv, sx);
        a1 += dot_q4_1_reg(wp1 + boff, xv, sx);
        a2 += dot_q4_1_reg(wp2 + boff, xv, sx);
        a3 += dot_q4_1_reg(wp3 + boff, xv, sx);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row0 + 1u < p.out_dim) y[row0 + 1u] = p.add_res ? (y[row0 + 1u] + a1) : a1;
        if (row0 + 2u < p.out_dim) y[row0 + 2u] = p.add_res ? (y[row0 + 2u] + a2) : a2;
        if (row0 + 3u < p.out_dim) y[row0 + 3u] = p.add_res ? (y[row0 + 3u] + a3) : a3;
    }
}

// 2-row-per-SG Q8_0 GEMV (8 rows/TG)
kernel void moe_gemv_q8(
    const device uchar* w [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 4u + sg) * 2u;
    uint row1 = row0 + 1u;
    if (row0 >= p.out_dim) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 34ul;
    const device uchar* wp0 = w + (ulong)row0 * rb;
    const device uchar* wp1 = (row1 < p.out_dim) ? (wp0 + rb) : wp0;
    float a0 = 0.0f, a1 = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        thread float4 xv[8];
        load_x32_reg(x + (b << 5), xv);
        ulong boff = (ulong)b * 34ul;
        a0 += dot_q8_0_reg(wp0 + boff, xv);
        if (row1 < p.out_dim) a1 += dot_q8_0_reg(wp1 + boff, xv);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row1 < p.out_dim) y[row1] = p.add_res ? (y[row1] + a1) : a1;
    }
}

kernel void moe_gemv_q4k(
    const device uchar* w [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4u + sg;
    if (row >= p.out_dim) return;
    uint nb = p.in_dim >> 8;
    const device uchar* wp = w + (ulong)row * (ulong)nb * 144ul;
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32u) acc += dot_q4_k_block(wp + (ulong)b * 144ul, x + (b << 8));
    acc = simd_sum(acc);
    if (lane == 0u) y[row] = p.add_res ? (y[row] + acc) : acc;
}

// Full 32-lane SIMD Q5_K GEMV (4 lanes/superblock, 2 rows/SG = 8 rows/TG)
kernel void moe_gemv_q5k(
    const device uchar* w [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 4u + sg) * 2u;
    uint row1 = row0 + 1u;
    if (row0 >= p.out_dim) return;
    uint nb = p.in_dim >> 8;
    ulong rb = (ulong)nb * 176ul;
    const device uchar* wp0 = w + (ulong)row0 * rb;
    const device uchar* wp1 = (row1 < p.out_dim) ? (wp0 + rb) : wp0;
    uint qtr = lane & 3u;
    float a0 = 0.0f, a1 = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float* xp = x + (b << 8);
        ulong boff = (ulong)b * 176ul;
        a0 += dot_q5_k_quarter(wp0 + boff, xp, qtr);
        if (row1 < p.out_dim) a1 += dot_q5_k_quarter(wp1 + boff, xp, qtr);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row1 < p.out_dim) y[row1] = p.add_res ? (y[row1] + a1) : a1;
    }
}

// Full 32-lane SIMD Q6_K GEMV (4 lanes/superblock, 2 rows/SG = 8 rows/TG)
kernel void moe_gemv_q6k(
    const device uchar* w [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 4u + sg) * 2u;
    uint row1 = row0 + 1u;
    if (row0 >= p.out_dim) return;
    uint nb = p.in_dim >> 8;
    ulong rb = (ulong)nb * 210ul;
    const device uchar* wp0 = w + (ulong)row0 * rb;
    const device uchar* wp1 = (row1 < p.out_dim) ? (wp0 + rb) : wp0;
    uint qtr = lane & 3u;
    float a0 = 0.0f, a1 = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float* xp = x + (b << 8);
        ulong boff = (ulong)b * 210ul;
        a0 += dot_q6_k_quarter(wp0 + boff, xp, qtr);
        if (row1 < p.out_dim) a1 += dot_q6_k_quarter(wp1 + boff, xp, qtr);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row1 < p.out_dim) y[row1] = p.add_res ? (y[row1] + a1) : a1;
    }
}

kernel void moe_gemv_f16(
    const device half* w [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4u + sg;
    if (row >= p.out_dim) return;
    const device half4* wp = (const device half4*)(w + (ulong)row * p.in_dim);
    const device float4* xp = (const device float4*)x;
    uint n4 = p.in_dim >> 2;
    float4 acc = float4(0.0f);
    for (uint i = lane; i < n4; i += 32u) acc = fma(float4(wp[i]), xp[i], acc);
    float res = simd_sum((acc.x + acc.y) + (acc.z + acc.w));
    if (lane == 0u) y[row] = p.add_res ? (y[row] + res) : res;
}

kernel void moe_gemv_f32(
    const device float* w [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4u + sg;
    if (row >= p.out_dim) return;
    const device float4* wp = (const device float4*)(w + (ulong)row * p.in_dim);
    const device float4* xp = (const device float4*)x;
    uint n4 = p.in_dim >> 2;
    float4 acc = float4(0.0f);
    for (uint i = lane; i < n4; i += 32u) acc = fma(wp[i], xp[i], acc);
    float res = simd_sum((acc.x + acc.y) + (acc.z + acc.w));
    if (lane == 0u) y[row] = p.add_res ? (y[row] + res) : res;
}

// Fused Dense SwiGLU Gate || Up Q4_0 (2 rows/SG = 8 rows/TG, shares xn across 4 weight rows!)
kernel void moe_dense_gate_up_q4(
    const device uchar* wg [[buffer(0)]],
    const device uchar* wu [[buffer(1)]],
    const device float* xn [[buffer(2)]],
    device float* h_out [[buffer(3)]],
    constant MoeGemvArgs& p [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 4u + sg) * 2u;
    uint row1 = row0 + 1u;
    if (row0 >= p.out_dim) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 18ul;
    const device uchar* g0 = wg + (ulong)row0 * rb;
    const device uchar* g1 = (row1 < p.out_dim) ? (g0 + rb) : g0;
    const device uchar* u0 = wu + (ulong)row0 * rb;
    const device uchar* u1 = (row1 < p.out_dim) ? (u0 + rb) : u0;
    float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        thread float4 xv[8];
        load_x32_reg(xn + (b << 5), xv);
        ulong boff = (ulong)b * 18ul;
        ag0 += dot_q4_0_reg(g0 + boff, xv);
        au0 += dot_q4_0_reg(u0 + boff, xv);
        ag1 += dot_q4_0_reg(g1 + boff, xv);
        au1 += dot_q4_0_reg(u1 + boff, xv);
    }
    ag0 = simd_sum(ag0); au0 = simd_sum(au0);
    ag1 = simd_sum(ag1); au1 = simd_sum(au1);
    if (lane == 0u) {
        h_out[row0] = (ag0 / (1.0f + exp(-ag0))) * au0;
        if (row1 < p.out_dim) h_out[row1] = (ag1 / (1.0f + exp(-ag1))) * au1;
    }
}

struct MoeRopeArgs {
    uint n_heads;
    uint n_kv;
    uint d_head;
    uint rope_dim;
    uint pos;
    float theta;
    uint rope_neox;
    uint has_freqs;
};

// Universal RoPE (Full/Partial, NORM/NEOX, Optional Llama-3 rope_freqs divisors) + FP16 KV-cache!
kernel void moe_rope_and_kv(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    const device float* v [[buffer(2)]],
    device half* k_cache [[buffer(3)]],
    device half* v_cache [[buffer(4)]],
    const device float* rope_freqs [[buffer(5)]],
    constant MoeRopeArgs& p [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x, h = gid.y;
    uint half_d = p.d_head >> 1;
    uint half_r = p.rope_dim >> 1;
    if (i >= half_d) return;

    uint idx0, idx1;
    if (i < half_r) {
        idx0 = p.rope_neox ? i : (2u * i);
        idx1 = p.rope_neox ? (i + half_r) : (2u * i + 1u);
        float inv_f = 1.0f / pow(p.theta, float(2u * i) / float(p.rope_dim));
        if (p.has_freqs) inv_f /= rope_freqs[i];
        float ang = float(p.pos) * inv_f;
        float cs = cos(ang), sn = sin(ang);
        if (h < p.n_heads) {
            uint qoff = h * p.d_head;
            float q0 = q[qoff + idx0], q1 = q[qoff + idx1];
            q[qoff + idx0] = q0 * cs - q1 * sn;
            q[qoff + idx1] = q0 * sn + q1 * cs;
        }
        if (h < p.n_kv) {
            uint koff = h * p.d_head;
            float k0 = k[koff + idx0], k1 = k[koff + idx1];
            float rk0 = k0 * cs - k1 * sn;
            float rk1 = k0 * sn + k1 * cs;
            k[koff + idx0] = rk0; k[koff + idx1] = rk1;
            ulong cbase = (ulong)p.pos * (p.n_kv * p.d_head) + koff;
            k_cache[cbase + idx0] = half(rk0);
            k_cache[cbase + idx1] = half(rk1);
            v_cache[cbase + idx0] = half(v[koff + idx0]);
            v_cache[cbase + idx1] = half(v[koff + idx1]);
        }
    } else {
        // Unrotated tail dimensions for Partial RoPE (e.g. GLM-4.5 rope_dim=64 of d_head=128)
        idx0 = 2u * i;
        idx1 = 2u * i + 1u;
        if (h < p.n_kv) {
            uint koff = h * p.d_head;
            ulong cbase = (ulong)p.pos * (p.n_kv * p.d_head) + koff;
            k_cache[cbase + idx0] = half(k[koff + idx0]);
            k_cache[cbase + idx1] = half(k[koff + idx1]);
            v_cache[cbase + idx0] = half(v[koff + idx0]);
            v_cache[cbase + idx1] = half(v[koff + idx1]);
        }
    }
}

struct MoeAttnArgs { uint n_heads; uint n_kv; uint d_head; uint kv_len; };

kernel void moe_flash_decode(
    const device float* q [[buffer(0)]],
    const device half* k_cache [[buffer(1)]],
    const device half* v_cache [[buffer(2)]],
    device float* ctx [[buffer(3)]],
    constant MoeAttnArgs& p [[buffer(4)]],
    uint h [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (h >= p.n_heads) return;
    uint grp = max(p.n_heads / p.n_kv, 1u);
    uint kvh = h / grp;
    uint kv_stride = p.n_kv * p.d_head;
    float scale = rsqrt(float(p.d_head));

    threadgroup float s_q[256];
    threadgroup float s_m[4];
    threadgroup float s_l[4];
    threadgroup float s_o[4][256];

    for (uint d = lid; d < p.d_head; d += 128u) s_q[d] = q[h * p.d_head + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_run = -1e30f, l_run = 0.0f;
    float o_acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    uint d_per_lane = (p.d_head + 31u) >> 5;

    for (uint t = sg; t < p.kv_len; t += 4u) {
        const device half* kp = k_cache + (ulong)t * kv_stride + kvh * p.d_head;
        const device half* vp = v_cache + (ulong)t * kv_stride + kvh * p.d_head;
        float dot = 0.0f;
        for (uint k = 0; k < d_per_lane; k++) {
            uint d = lane + k * 32u;
            if (d < p.d_head) dot = fma(s_q[d], float(kp[d]), dot);
        }
        float sc = simd_sum(dot) * scale;
        float mn = max(m_run, sc);
        float al = exp(m_run - mn);
        float wt = exp(sc - mn);
        l_run = fma(l_run, al, wt);
        m_run = mn;
        for (uint k = 0; k < d_per_lane; k++) {
            uint d = lane + k * 32u;
            if (d < p.d_head) o_acc[k] = fma(o_acc[k], al, wt * float(vp[d]));
        }
    }
    if (lane == 0u) { s_m[sg] = m_run; s_l[sg] = l_run; }
    for (uint k = 0; k < d_per_lane; k++) {
        uint d = lane + k * 32u;
        if (d < p.d_head) s_o[sg][d] = o_acc[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gm = max(max(s_m[0], s_m[1]), max(s_m[2], s_m[3]));
    float gl = 0.0f;
    for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * exp(s_m[s] - gm);
    float inv = 1.0f / max(gl, 1e-12f);
    for (uint d = lid; d < p.d_head; d += 128u) {
        float ov = 0.0f;
        for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fma(s_o[s][d], exp(s_m[s] - gm), ov);
        ctx[h * p.d_head + d] = ov * inv;
    }
}

struct MoeRouterArgs { uint n_experts; uint top_k; uint norm_weights; uint gating_func; uint has_bias; float w_scale; };

kernel void moe_router_topk(
    const device float* logits [[buffer(0)]],
    const device float* exp_bias [[buffer(1)]],
    device uint* out_idx [[buffer(2)]],
    device float* out_wt [[buffer(3)]],
    constant MoeRouterArgs& p [[buffer(4)]],
    uint lid [[thread_index_in_threadgroup]]
) {
    if (lid > 0u) return;
    float probs[256];
    float scores[256];
    if (p.gating_func == 2u) {
        for (uint e = 0; e < p.n_experts; e++) {
            float s = 1.0f / (1.0f + exp(-logits[e]));
            probs[e] = s;
            scores[e] = p.has_bias ? (s + exp_bias[e]) : s;
        }
    } else {
        float mx = -1e30f;
        for (uint e = 0; e < p.n_experts; e++) mx = max(mx, logits[e]);
        float sum = 0.0f;
        for (uint e = 0; e < p.n_experts; e++) { float v = exp(logits[e] - mx); probs[e] = v; sum += v; }
        float inv = 1.0f / max(sum, 1e-12f);
        for (uint e = 0; e < p.n_experts; e++) { probs[e] *= inv; scores[e] = probs[e]; }
    }
    float wsum = 0.0f;
    for (uint k = 0; k < p.top_k; k++) {
        uint best_e = 0u;
        float best_s = -1e30f;
        for (uint e = 0; e < p.n_experts; e++) {
            if (scores[e] > best_s) { best_s = scores[e]; best_e = e; }
        }
        out_idx[k] = best_e;
        out_wt[k] = probs[best_e];
        wsum += probs[best_e];
        scores[best_e] = -1e30f;
    }
    float mult = p.w_scale;
    if (p.norm_weights && wsum > 1e-12f) mult /= wsum;
    for (uint k = 0; k < p.top_k; k++) out_wt[k] *= mult;
}

struct MoeExpArgs {
    uint d_model;
    uint d_ff;
    uint n_active;
    ulong gate_exp_bytes;
    ulong up_exp_bytes;
    ulong down_exp_bytes;
    ulong slot_stride_bytes;
};

kernel void moe_topk_gate_up_q4(
    const device uchar* gate_exps [[buffer(0)]],
    const device uchar* up_exps [[buffer(1)]],
    const device float* xn [[buffer(2)]],
    const device uint* e_idx [[buffer(3)]],
    device float* h_out [[buffer(4)]],
    constant MoeExpArgs& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint slot = tgid.y;
    uint row0 = (tgid.x * 4u + sg) * 2u;
    uint row1 = row0 + 1u;
    if (slot >= p.n_active || row0 >= p.d_ff) return;
    uint eid = e_idx[slot];
    uint nb = p.d_model >> 5;
    ulong rb = (ulong)nb * 18ul;
    const device uchar* wg0 = gate_exps + (ulong)eid * p.gate_exp_bytes + (ulong)row0 * rb;
    const device uchar* wg1 = (row1 < p.d_ff) ? (wg0 + rb) : wg0;
    const device uchar* wu0 = up_exps   + (ulong)eid * p.up_exp_bytes   + (ulong)row0 * rb;
    const device uchar* wu1 = (row1 < p.d_ff) ? (wu0 + rb) : wu0;
    float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        thread float4 xv[8];
        load_x32_reg(xn + (b << 5), xv);
        ulong boff = (ulong)b * 18ul;
        ag0 += dot_q4_0_reg(wg0 + boff, xv);
        au0 += dot_q4_0_reg(wu0 + boff, xv);
        ag1 += dot_q4_0_reg(wg1 + boff, xv);
        au1 += dot_q4_0_reg(wu1 + boff, xv);
    }
    ag0 = simd_sum(ag0); au0 = simd_sum(au0);
    ag1 = simd_sum(ag1); au1 = simd_sum(au1);
    if (lane == 0u) {
        h_out[(ulong)slot * p.d_ff + row0] = (ag0 / (1.0f + exp(-ag0))) * au0;
        if (row1 < p.d_ff) h_out[(ulong)slot * p.d_ff + row1] = (ag1 / (1.0f + exp(-ag1))) * au1;
    }
}

kernel void moe_topk_down_accum_q4(
    const device uchar* down_exps [[buffer(0)]],
    const device float* h_in [[buffer(1)]],
    const device uint* e_idx [[buffer(2)]],
    const device float* e_wt [[buffer(3)]],
    device float* x_res [[buffer(4)]],
    constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 4u + sg) * 4u;
    if (row0 >= p.d_model) return;
    uint nb = p.d_ff >> 5;
    ulong rb = (ulong)nb * 18ul;
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    for (uint s = 0; s < p.n_active; s++) {
        uint eid = e_idx[s];
        const device uchar* wd0 = down_exps + (ulong)eid * p.down_exp_bytes + (ulong)row0 * rb;
        const device uchar* wd1 = (row0 + 1u < p.d_model) ? (wd0 + rb) : wd0;
        const device uchar* wd2 = (row0 + 2u < p.d_model) ? (wd0 + rb * 2ul) : wd0;
        const device uchar* wd3 = (row0 + 3u < p.d_model) ? (wd0 + rb * 3ul) : wd0;
        const device float* hp = h_in + (ulong)s * p.d_ff;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint b = lane; b < nb; b += 32u) {
            thread float4 hv[8];
            load_x32_reg(hp + (b << 5), hv);
            ulong boff = (ulong)b * 18ul;
            a0 += dot_q4_0_reg(wd0 + boff, hv);
            a1 += dot_q4_0_reg(wd1 + boff, hv);
            a2 += dot_q4_0_reg(wd2 + boff, hv);
            a3 += dot_q4_0_reg(wd3 + boff, hv);
        }
        float w = e_wt[s];
        t0 = fma(w, simd_sum(a0), t0); t1 = fma(w, simd_sum(a1), t1);
        t2 = fma(w, simd_sum(a2), t2); t3 = fma(w, simd_sum(a3), t3);
    }
    if (lane == 0u) {
        x_res[row0] += t0;
        if (row0 + 1u < p.d_model) x_res[row0 + 1u] += t1;
        if (row0 + 2u < p.d_model) x_res[row0 + 2u] += t2;
        if (row0 + 3u < p.d_model) x_res[row0 + 3u] += t3;
    }
}

kernel void moe_topk_down_accum_q4_1(
    const device uchar* down_exps [[buffer(0)]],
    const device float* h_in [[buffer(1)]],
    const device uint* e_idx [[buffer(2)]],
    const device float* e_wt [[buffer(3)]],
    device float* x_res [[buffer(4)]],
    constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 4u + sg) * 4u;
    if (row0 >= p.d_model) return;
    uint nb = p.d_ff >> 5;
    ulong rb = (ulong)nb * 20ul;
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    for (uint s = 0; s < p.n_active; s++) {
        uint eid = e_idx[s];
        const device uchar* wd0 = down_exps + (ulong)eid * p.down_exp_bytes + (ulong)row0 * rb;
        const device uchar* wd1 = (row0 + 1u < p.d_model) ? (wd0 + rb) : wd0;
        const device uchar* wd2 = (row0 + 2u < p.d_model) ? (wd0 + rb * 2ul) : wd0;
        const device uchar* wd3 = (row0 + 3u < p.d_model) ? (wd0 + rb * 3ul) : wd0;
        const device float* hp = h_in + (ulong)s * p.d_ff;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint b = lane; b < nb; b += 32u) {
            thread float4 hv[8];
            load_x32_reg(hp + (b << 5), hv);
            float sh = sum_x32_reg(hv);
            ulong boff = (ulong)b * 20ul;
            a0 += dot_q4_1_reg(wd0 + boff, hv, sh);
            a1 += dot_q4_1_reg(wd1 + boff, hv, sh);
            a2 += dot_q4_1_reg(wd2 + boff, hv, sh);
            a3 += dot_q4_1_reg(wd3 + boff, hv, sh);
        }
        float w = e_wt[s];
        t0 = fma(w, simd_sum(a0), t0); t1 = fma(w, simd_sum(a1), t1);
        t2 = fma(w, simd_sum(a2), t2); t3 = fma(w, simd_sum(a3), t3);
    }
    if (lane == 0u) {
        x_res[row0] += t0;
        if (row0 + 1u < p.d_model) x_res[row0 + 1u] += t1;
        if (row0 + 2u < p.d_model) x_res[row0 + 2u] += t2;
        if (row0 + 3u < p.d_model) x_res[row0 + 3u] += t3;
    }
}

kernel void moe_slot_gate_up_q4(
    const device uchar* cache_pool [[buffer(0)]],
    const device float* xn [[buffer(1)]],
    const device uint* slot_ids [[buffer(2)]],
    device float* h_out [[buffer(3)]],
    constant MoeExpArgs& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint s = tgid.y;
    uint row0 = (tgid.x * 4u + sg) * 2u;
    uint row1 = row0 + 1u;
    if (s >= p.n_active || row0 >= p.d_ff) return;
    uint cslot = slot_ids[s];
    uint nb = p.d_model >> 5;
    ulong rb = (ulong)nb * 18ul;
    const device uchar* base = cache_pool + (ulong)cslot * p.slot_stride_bytes;
    const device uchar* wg0 = base + (ulong)row0 * rb;
    const device uchar* wg1 = (row1 < p.d_ff) ? (wg0 + rb) : wg0;
    const device uchar* wu0 = base + p.gate_exp_bytes + (ulong)row0 * rb;
    const device uchar* wu1 = (row1 < p.d_ff) ? (wu0 + rb) : wu0;
    float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        thread float4 xv[8];
        load_x32_reg(xn + (b << 5), xv);
        ulong boff = (ulong)b * 18ul;
        ag0 += dot_q4_0_reg(wg0 + boff, xv);
        au0 += dot_q4_0_reg(wu0 + boff, xv);
        ag1 += dot_q4_0_reg(wg1 + boff, xv);
        au1 += dot_q4_0_reg(wu1 + boff, xv);
    }
    ag0 = simd_sum(ag0); au0 = simd_sum(au0);
    ag1 = simd_sum(ag1); au1 = simd_sum(au1);
    if (lane == 0u) {
        h_out[(ulong)s * p.d_ff + row0] = (ag0 / (1.0f + exp(-ag0))) * au0;
        if (row1 < p.d_ff) h_out[(ulong)s * p.d_ff + row1] = (ag1 / (1.0f + exp(-ag1))) * au1;
    }
}

kernel void moe_slot_down_accum_q4(
    const device uchar* cache_pool [[buffer(0)]],
    const device float* h_in [[buffer(1)]],
    const device uint* slot_ids [[buffer(2)]],
    const device float* e_wt [[buffer(3)]],
    device float* x_res [[buffer(4)]],
    constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 4u + sg) * 4u;
    if (row0 >= p.d_model) return;
    uint nb = p.d_ff >> 5;
    ulong rb = (ulong)nb * 18ul;
    ulong down_off = p.gate_exp_bytes + p.up_exp_bytes;
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    for (uint s = 0; s < p.n_active; s++) {
        uint cslot = slot_ids[s];
        const device uchar* wd0 = cache_pool + (ulong)cslot * p.slot_stride_bytes + down_off + (ulong)row0 * rb;
        const device uchar* wd1 = (row0 + 1u < p.d_model) ? (wd0 + rb) : wd0;
        const device uchar* wd2 = (row0 + 2u < p.d_model) ? (wd0 + rb * 2ul) : wd0;
        const device uchar* wd3 = (row0 + 3u < p.d_model) ? (wd0 + rb * 3ul) : wd0;
        const device float* hp = h_in + (ulong)s * p.d_ff;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint b = lane; b < nb; b += 32u) {
            thread float4 hv[8];
            load_x32_reg(hp + (b << 5), hv);
            ulong boff = (ulong)b * 18ul;
            a0 += dot_q4_0_reg(wd0 + boff, hv);
            a1 += dot_q4_0_reg(wd1 + boff, hv);
            a2 += dot_q4_0_reg(wd2 + boff, hv);
            a3 += dot_q4_0_reg(wd3 + boff, hv);
        }
        float w = e_wt[s];
        t0 = fma(w, simd_sum(a0), t0); t1 = fma(w, simd_sum(a1), t1);
        t2 = fma(w, simd_sum(a2), t2); t3 = fma(w, simd_sum(a3), t3);
    }
    if (lane == 0u) {
        x_res[row0] += t0;
        if (row0 + 1u < p.d_model) x_res[row0 + 1u] += t1;
        if (row0 + 2u < p.d_model) x_res[row0 + 2u] += t2;
        if (row0 + 3u < p.d_model) x_res[row0 + 3u] += t3;
    }
}

kernel void moe_slot_down_accum_q4_1(
    const device uchar* cache_pool [[buffer(0)]],
    const device float* h_in [[buffer(1)]],
    const device uint* slot_ids [[buffer(2)]],
    const device float* e_wt [[buffer(3)]],
    device float* x_res [[buffer(4)]],
    constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 4u + sg) * 4u;
    if (row0 >= p.d_model) return;
    uint nb = p.d_ff >> 5;
    ulong rb = (ulong)nb * 20ul;
    ulong down_off = p.gate_exp_bytes + p.up_exp_bytes;
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    for (uint s = 0; s < p.n_active; s++) {
        uint cslot = slot_ids[s];
        const device uchar* wd0 = cache_pool + (ulong)cslot * p.slot_stride_bytes + down_off + (ulong)row0 * rb;
        const device uchar* wd1 = (row0 + 1u < p.d_model) ? (wd0 + rb) : wd0;
        const device uchar* wd2 = (row0 + 2u < p.d_model) ? (wd0 + rb * 2ul) : wd0;
        const device uchar* wd3 = (row0 + 3u < p.d_model) ? (wd0 + rb * 3ul) : wd0;
        const device float* hp = h_in + (ulong)s * p.d_ff;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint b = lane; b < nb; b += 32u) {
            thread float4 hv[8];
            load_x32_reg(hp + (b << 5), hv);
            float sh = sum_x32_reg(hv);
            ulong boff = (ulong)b * 20ul;
            a0 += dot_q4_1_reg(wd0 + boff, hv, sh);
            a1 += dot_q4_1_reg(wd1 + boff, hv, sh);
            a2 += dot_q4_1_reg(wd2 + boff, hv, sh);
            a3 += dot_q4_1_reg(wd3 + boff, hv, sh);
        }
        float w = e_wt[s];
        t0 = fma(w, simd_sum(a0), t0); t1 = fma(w, simd_sum(a1), t1);
        t2 = fma(w, simd_sum(a2), t2); t3 = fma(w, simd_sum(a3), t3);
    }
    if (lane == 0u) {
        x_res[row0] += t0;
        if (row0 + 1u < p.d_model) x_res[row0 + 1u] += t1;
        if (row0 + 2u < p.d_model) x_res[row0 + 2u] += t2;
        if (row0 + 3u < p.d_model) x_res[row0 + 3u] += t3;
    }
}

kernel void moe_argmax_f32(
    const device float* logits [[buffer(0)]],
    device uint* out_tok [[buffer(1)]],
    constant uint& vocab [[buffer(2)]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float s_v[8];
    threadgroup uint  s_i[8];
    float bv = -1e30f;
    uint bi = 0u;
    for (uint i = lid; i < vocab; i += 256u) {
        float v = logits[i];
        if (v > bv) { bv = v; bi = i; }
    }
    for (uint off = 16u; off > 0u; off >>= 1) {
        float ov = simd_shuffle_xor(bv, off);
        uint oi = simd_shuffle_xor(bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0u) { s_v[sg] = bv; s_i[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        float best_v = s_v[0]; uint best_i = s_i[0];
        for (uint s = 1u; s < 8u; s++) {
            if (s_v[s] > best_v) { best_v = s_v[s]; best_i = s_i[s]; }
        }
        out_tok[0] = best_i;
    }
}
)";

static int moe_init_metal(void) {
    if (g_moe_dev != nil) return 1;
    g_moe_dev = MTLCreateSystemDefaultDevice();
    if (!g_moe_dev) return 0;
    g_moe_q = [g_moe_dev newCommandQueue];
    NSError* err = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.fastMathEnabled = YES;
    g_moe_lib = [g_moe_dev newLibraryWithSource:kMoeMslSource options:opts error:&err];
    if (!g_moe_lib) {
        fprintf(stderr, "[tenon-stream] MSL compile error: %s\n", [[err localizedDescription] UTF8String]);
        return 0;
    }
    id<MTLComputePipelineState> (^mk)(NSString*) = ^id<MTLComputePipelineState>(NSString* nm) {
        NSError* pe = nil;
        id<MTLFunction> fn = [g_moe_lib newFunctionWithName:nm];
        if (!fn) { fprintf(stderr, "[tenon-stream] missing kernel %s\n", [nm UTF8String]); return nil; }
        id<MTLComputePipelineState> pso = [g_moe_dev newComputePipelineStateWithFunction:fn error:&pe];
        if (!pso) fprintf(stderr, "[tenon-stream] PSO error %s: %s\n", [nm UTF8String], [[pe localizedDescription] UTF8String]);
        return pso;
    };
    g_pso_embed             = mk(@"moe_embed_lookup");
    g_pso_rmsnorm           = mk(@"moe_rmsnorm");
    g_pso_add_bias          = mk(@"moe_add_bias");
    g_pso_swiglu_mul        = mk(@"moe_swiglu_mul");
    g_pso_gemv_q4           = mk(@"moe_gemv_q4");
    g_pso_gemv_q4_1         = mk(@"moe_gemv_q4_1");
    g_pso_gemv_q8           = mk(@"moe_gemv_q8");
    g_pso_gemv_q4k          = mk(@"moe_gemv_q4k");
    g_pso_gemv_q5k          = mk(@"moe_gemv_q5k");
    g_pso_gemv_q6k          = mk(@"moe_gemv_q6k");
    g_pso_gemv_f16          = mk(@"moe_gemv_f16");
    g_pso_gemv_f32          = mk(@"moe_gemv_f32");
    g_pso_dense_gu_q4       = mk(@"moe_dense_gate_up_q4");
    g_pso_rope_kv           = mk(@"moe_rope_and_kv");
    g_pso_flash_dec         = mk(@"moe_flash_decode");
    g_pso_router            = mk(@"moe_router_topk");
    g_pso_moe_gu_q4         = mk(@"moe_topk_gate_up_q4");
    g_pso_moe_down_q4       = mk(@"moe_topk_down_accum_q4");
    g_pso_moe_down_q4_1     = mk(@"moe_topk_down_accum_q4_1");
    g_pso_moe_slot_gu_q4    = mk(@"moe_slot_gate_up_q4");
    g_pso_moe_slot_down_q4  = mk(@"moe_slot_down_accum_q4");
    g_pso_moe_slot_down_q4_1= mk(@"moe_slot_down_accum_q4_1");
    g_pso_argmax            = mk(@"moe_argmax_f32");
    return (g_pso_gemv_q4 != nil && g_pso_gemv_q5k != nil && g_pso_moe_slot_down_q4_1 != nil);
}

typedef struct { FILE* f; } GgufReader;
static uint32_t gr_u32(GgufReader* r) { uint32_t v = 0; fread(&v, 4, 1, r->f); return v; }
static int32_t  gr_i32(GgufReader* r) { int32_t  v = 0; fread(&v, 4, 1, r->f); return v; }
static uint64_t gr_u64(GgufReader* r) { uint64_t v = 0; fread(&v, 8, 1, r->f); return v; }
static float    gr_f32(GgufReader* r) { float    v = 0; fread(&v, 4, 1, r->f); return v; }

static char* gr_str(GgufReader* r) {
    uint64_t len = gr_u64(r);
    char* s = (char*)malloc((size_t)len + 1);
    if (len > 0) fread(s, 1, (size_t)len, r->f);
    s[len] = 0;
    return s;
}

static void gr_skip_val(GgufReader* r, uint32_t t) {
    switch (t) {
        case GGUF_T_U8: case GGUF_T_I8: case GGUF_T_BOOL: fseeko(r->f, 1, SEEK_CUR); break;
        case GGUF_T_U16: case GGUF_T_I16: fseeko(r->f, 2, SEEK_CUR); break;
        case GGUF_T_U32: case GGUF_T_I32: case GGUF_T_F32: fseeko(r->f, 4, SEEK_CUR); break;
        case GGUF_T_U64: case GGUF_T_I64: case GGUF_T_F64: fseeko(r->f, 8, SEEK_CUR); break;
        case GGUF_T_STR: { uint64_t l = gr_u64(r); fseeko(r->f, (off_t)l, SEEK_CUR); break; }
        case GGUF_T_ARR: {
            uint32_t et = gr_u32(r);
            uint64_t n = gr_u64(r);
            for (uint64_t i = 0; i < n; i++) gr_skip_val(r, et);
            break;
        }
        default: break;
    }
}

static uint64_t gr_read_int(GgufReader* r, uint32_t t) {
    switch (t) {
        case GGUF_T_U8:  { uint8_t v = 0; fread(&v, 1, 1, r->f); return v; }
        case GGUF_T_I8:  { int8_t v = 0; fread(&v, 1, 1, r->f); return (uint64_t)(int64_t)v; }
        case GGUF_T_U16: { uint16_t v = 0; fread(&v, 2, 1, r->f); return v; }
        case GGUF_T_I16: { int16_t v = 0; fread(&v, 2, 1, r->f); return (uint64_t)(int64_t)v; }
        case GGUF_T_U32: return gr_u32(r);
        case GGUF_T_I32: return (uint64_t)(int64_t)gr_i32(r);
        case GGUF_T_U64: return gr_u64(r);
        case GGUF_T_I64: { int64_t v = 0; fread(&v, 8, 1, r->f); return (uint64_t)v; }
        case GGUF_T_BOOL:{ uint8_t v = 0; fread(&v, 1, 1, r->f); return v; }
        default: gr_skip_val(r, t); return 0;
    }
}

static int kv_ends_with(const char* s, const char* suf) {
    size_t ls = strlen(s), lf = strlen(suf);
    return ls >= lf && strcmp(s + ls - lf, suf) == 0;
}

static void pread_exact(int fd, void* dst, size_t nbytes, uint64_t off) {
    uint8_t* p = (uint8_t*)dst;
    size_t done = 0;
    while (done < nbytes) {
        ssize_t r = pread(fd, p + done, nbytes - done, (off_t)(off + done));
        if (r <= 0) break;
        done += (size_t)r;
    }
}

static void record_weight_meta(MoeWeightRef* ref, const MoeTensorInfo* ti) {
    ref->ggml_type = ti->ggml_type;
    ref->ne0 = ti->ne[0]; ref->ne1 = ti->ne[1]; ref->ne2 = ti->ne[2];
    ref->file_off = ti->file_off;
    ref->byte_size = ti->byte_size;
    ref->shard_idx = ti->shard_idx;
}

static void load_pinned_weight(MoeModel* m, MoeWeightRef* ref, const MoeTensorInfo* ti) {
    record_weight_meta(ref, ti);
    int s = ti->shard_idx;
    if (m->resident_mode && m->mmap_mtl_bufs[s] != nil) {
        ref->buf = m->mmap_mtl_bufs[s];
        ref->offset = ti->file_off;
    } else {
        size_t sz = (size_t)(ti->byte_size > 64 ? ti->byte_size : 64);
        ref->buf = [g_moe_dev newBufferWithLength:sz options:MTLResourceStorageModeShared];
        pread_exact(m->fds[s], [ref->buf contents], (size_t)ti->byte_size, ti->file_off);
        ref->offset = 0;
        m->core_vram_bytes += (int64_t)sz;
    }
}

static uint64_t align64(uint64_t v) { return (v + 63ull) & ~63ull; }

static void load_expert_into_slot(MoeModel* m, int L, int eid, int slot) {
    MoeLayer* ly = &m->layers[L];
    uint8_t* dst = ((uint8_t*)[ly->cache_buf contents]) + (size_t)slot * (size_t)ly->slot_stride_bytes;
    uint64_t g_off = ly->gate_exps.file_off + (uint64_t)eid * ly->gate_exp_bytes;
    uint64_t u_off = ly->up_exps.file_off   + (uint64_t)eid * ly->up_exp_bytes;
    uint64_t d_off = ly->down_exps.file_off + (uint64_t)eid * ly->down_exp_bytes;
    pread_exact(m->fds[ly->gate_exps.shard_idx], dst, (size_t)ly->gate_exp_bytes, g_off);
    pread_exact(m->fds[ly->up_exps.shard_idx],   dst + ly->gate_exp_bytes, (size_t)ly->up_exp_bytes, u_off);
    pread_exact(m->fds[ly->down_exps.shard_idx], dst + ly->gate_exp_bytes + ly->up_exp_bytes, (size_t)ly->down_exp_bytes, d_off);
    __sync_fetch_and_add(&m->ssd_bytes_read, (int64_t)(ly->gate_exp_bytes + ly->up_exp_bytes + ly->down_exp_bytes));
}

// Parallel multi-chunk NVMe `pread()` of Dense layer weights into ring slot `pidx`!
// When `m->dense_pin_all_attn` is 1, streams ONLY FFN (`gate/up/down`, ~396 MB) since all Attention is GPU-pinned!
static void load_dense_layer_into_ping(MoeModel* m, int L, int pidx, int parallel) {
    MoeLayer* ly = &m->layers[L];
    uint8_t* base = (uint8_t*)[m->dense_ping[pidx] contents];
    typedef struct { int fd; uint8_t* dst; size_t len; uint64_t off; } IoChunk;
    IoChunk chunks[32];
    int nc = 0;

    MoeWeightRef* refs[7] = { &ly->wq, &ly->wk, &ly->wv, &ly->wo, &ly->ffn_gate, &ly->ffn_up, &ly->ffn_down };
    uint64_t offs[7] = { m->dense_off_wq, m->dense_off_wk, m->dense_off_wv, m->dense_off_wo,
                         m->dense_off_gate, m->dense_off_up, m->dense_off_down };
    int start_i = m->dense_pin_all_attn ? 4 : 0;
    uint64_t total_b = 0;
    for (int i = start_i; i < 7; i++) {
        uint64_t sz = refs[i]->byte_size;
        if (sz == 0) continue;
        total_b += sz;
        int splits = (parallel && sz >= 64ull * 1024ull * 1024ull) ? 6 : 1;
        size_t chunk_sz = (size_t)((sz + (uint64_t)splits - 1ull) / (uint64_t)splits);
        chunk_sz = (chunk_sz + 4095ull) & ~4095ull; // 4KB page-aligned chunks for raw NVMe DMA!
        for (int s = 0; s < splits && nc < 32; s++) {
            uint64_t rel = (uint64_t)s * (uint64_t)chunk_sz;
            if (rel >= sz) break;
            size_t len = (rel + chunk_sz <= sz) ? chunk_sz : (size_t)(sz - rel);
            chunks[nc].fd  = m->fds[refs[i]->shard_idx];
            chunks[nc].dst = base + offs[i] + rel;
            chunks[nc].len = len;
            chunks[nc].off = refs[i]->file_off + rel;
            nc++;
        }
    }
    if (parallel && nc > 1) {
        IoChunk* cp = chunks;
        dispatch_apply((size_t)nc, m->io_queue, ^(size_t idx) {
            pread_exact(cp[idx].fd, cp[idx].dst, cp[idx].len, cp[idx].off);
        });
    } else {
        for (int i = 0; i < nc; i++) pread_exact(chunks[i].fd, chunks[i].dst, chunks[i].len, chunks[i].off);
    }
    __sync_fetch_and_add(&m->ssd_bytes_read, (int64_t)total_b);
}

// Parse one GGUF shard header; if `is_primary`, also extracts model KV metadata & vocab.
static int parse_gguf_shard(
    const char* path, int shard_idx, int is_primary, MoeModel* m,
    MoeTensorInfo** tlist_io, uint64_t* tcount_io, int* split_count_out
) {
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "[tenon-stream] cannot open %s\n", path); return 0; }
    GgufReader gr = { fp };
    uint32_t magic = gr_u32(&gr);
    uint32_t ver = gr_u32(&gr);
    if (magic != 0x46554747u || ver < 2 || ver > 3) {
        fprintf(stderr, "[tenon-stream] invalid GGUF magic/ver in %s: 0x%x v%u\n", path, magic, ver);
        fclose(fp);
        return 0;
    }
    uint64_t tensor_count = gr_u64(&gr);
    uint64_t kv_count = gr_u64(&gr);
    uint64_t alignment = 32;
    int norm_wt_set = 0;

    for (uint64_t i = 0; i < kv_count; i++) {
        char* key = gr_str(&gr);
        uint32_t vt = gr_u32(&gr);
        if (strcmp(key, "general.alignment") == 0) {
            alignment = gr_read_int(&gr, vt);
        } else if (strcmp(key, "split.count") == 0) {
            int sc = (int)gr_read_int(&gr, vt);
            if (is_primary && split_count_out && sc > 0) *split_count_out = sc;
        } else if (!is_primary) {
            gr_skip_val(&gr, vt);
        } else if (strcmp(key, "general.architecture") == 0 && vt == GGUF_T_STR) {
            char* av = gr_str(&gr);
            strncpy(m->arch, av, sizeof(m->arch) - 1);
            free(av);
        } else if (kv_ends_with(key, ".block_count")) {
            m->n_layers = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".nextn_predict_layers")) {
            m->nextn_layers = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".embedding_length")) {
            m->d_model = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".expert_feed_forward_length")) {
            m->d_ff_exp = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".feed_forward_length")) {
            m->d_ff_dense = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".attention.head_count")) {
            m->n_heads = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".attention.head_count_kv")) {
            m->n_kv = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".attention.key_length")) {
            m->d_head = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".rope.dimension_count")) {
            m->rope_dim = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".expert_count")) {
            m->n_experts = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".expert_used_count")) {
            m->top_k = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".expert_gating_func")) {
            m->gating_func = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".expert_weights_scale") && vt == GGUF_T_F32) {
            m->expert_weights_scale = gr_f32(&gr);
        } else if (kv_ends_with(key, ".rope.freq_base") && vt == GGUF_T_F32) {
            m->rope_theta = gr_f32(&gr);
        } else if (kv_ends_with(key, ".attention.layer_norm_rms_epsilon") && vt == GGUF_T_F32) {
            m->rms_eps = gr_f32(&gr);
        } else if (kv_ends_with(key, ".expert_weights_norm")) {
            m->expert_weights_norm = (int)gr_read_int(&gr, vt);
            norm_wt_set = 1;
        } else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) {
            m->bos_id = (int)gr_read_int(&gr, vt);
        } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) {
            m->eos_id = (int)gr_read_int(&gr, vt);
        } else if (strcmp(key, "tokenizer.ggml.tokens") == 0 && vt == GGUF_T_ARR) {
            uint32_t et = gr_u32(&gr);
            uint64_t n = gr_u64(&gr);
            if (et == GGUF_T_STR && n > 0 && n <= 262144) {
                m->vocab_size = (int)n;
                m->vocab_tokens = (char**)calloc((size_t)n, sizeof(char*));
                for (uint64_t ti = 0; ti < n; ti++) m->vocab_tokens[ti] = gr_str(&gr);
            } else {
                for (uint64_t ti = 0; ti < n; ti++) gr_skip_val(&gr, et);
            }
        } else {
            gr_skip_val(&gr, vt);
        }
        free(key);
    }

    if (is_primary && !norm_wt_set) {
        m->expert_weights_norm = (strcmp(m->arch, "olmoe") == 0) ? 0 : 1;
    }

    uint64_t old_n = *tcount_io;
    *tlist_io = (MoeTensorInfo*)realloc(*tlist_io, (size_t)(old_n + tensor_count) * sizeof(MoeTensorInfo));
    MoeTensorInfo* dst = (*tlist_io) + old_n;
    memset(dst, 0, (size_t)tensor_count * sizeof(MoeTensorInfo));
    for (uint64_t i = 0; i < tensor_count; i++) {
        char* tn = gr_str(&gr);
        strncpy(dst[i].name, tn, sizeof(dst[i].name) - 1);
        free(tn);
        dst[i].n_dims = gr_u32(&gr);
        uint64_t numel = 1;
        for (uint32_t d = 0; d < dst[i].n_dims; d++) {
            uint64_t dim = gr_u64(&gr);
            if (d < 4) dst[i].ne[d] = dim;
            numel *= dim;
        }
        dst[i].ggml_type = gr_u32(&gr);
        dst[i].rel_off = gr_u64(&gr);
        dst[i].byte_size = ggml_type_bytes(dst[i].ggml_type, numel);
        dst[i].shard_idx = shard_idx;
    }
    uint64_t hdr_end = (uint64_t)ftello(fp);
    uint64_t data_off = (hdr_end + alignment - 1) & ~(alignment - 1);
    fclose(fp);
    for (uint64_t i = 0; i < tensor_count; i++) {
        dst[i].file_off = data_off + dst[i].rel_off;
    }
    *tcount_io = old_n + tensor_count;
    return 1;
}

int64_t tn_moe_load_gguf(int64_t path_cstr, int64_t cache_slots_per_layer, int64_t use_nocache, int64_t enable_overlap) {
    if (!moe_init_metal() || !path_cstr) return 0;
    @autoreleasepool {
        int slot = -1;
        for (int i = 1; i < MOE_MAX_MODELS; i++) { if (!g_moe[i].active) { slot = i; break; } }
        if (slot < 0) return 0;
        MoeModel* m = &g_moe[slot];
        memset(m, 0, sizeof(MoeModel));
        for (int s = 0; s < MOE_MAX_SHARDS; s++) m->fds[s] = -1;
        m->rope_theta = 10000.0f;
        m->rms_eps = 1e-5f;
        m->bos_id = 1;
        m->eos_id = 2;
        m->expert_weights_scale = 1.0f;
        m->use_nocache = (int)use_nocache;
        m->enable_overlap = (int)enable_overlap;

        const char* path0 = (const char*)(uintptr_t)path_cstr;
        char shard_paths[MOE_MAX_SHARDS][1024];
        strncpy(shard_paths[0], path0, sizeof(shard_paths[0]) - 1);

        MoeTensorInfo* tlist = NULL;
        uint64_t tcount = 0;
        int split_count = 1;
        if (!parse_gguf_shard(shard_paths[0], 0, 1, m, &tlist, &tcount, &split_count)) {
            free(tlist);
            return 0;
        }
        if (split_count > MOE_MAX_SHARDS) split_count = MOE_MAX_SHARDS;
        m->n_shards = split_count;

        // Auto-discover remaining shards (`...-00001-of-0000N.gguf` -> `00002..0000N`)
        for (int s = 1; s < split_count; s++) {
            const char* marker = strstr(path0, "-00001-of-");
            if (!marker) { fprintf(stderr, "[tenon-stream] split GGUF missing -00001-of- pattern\n"); free(tlist); return 0; }
            size_t pref = (size_t)(marker - path0);
            snprintf(shard_paths[s], sizeof(shard_paths[s]), "%.*s-%05d-of-%s",
                     (int)pref, path0, s + 1, marker + 10);
            if (!parse_gguf_shard(shard_paths[s], s, 0, m, &tlist, &tcount, NULL)) {
                free(tlist);
                return 0;
            }
        }

        // Finalize architecture dimensions
        if (m->nextn_layers > 0 && m->n_layers > m->nextn_layers) m->n_layers -= m->nextn_layers;
        if (m->n_layers > MOE_MAX_LAYERS) m->n_layers = MOE_MAX_LAYERS;
        if (m->n_kv <= 0) m->n_kv = m->n_heads;
        if (m->d_head <= 0 && m->n_heads > 0) m->d_head = m->d_model / m->n_heads;
        if (m->rope_dim <= 0) m->rope_dim = m->d_head;
        if (m->d_ff_exp <= 0) m->d_ff_exp = m->d_ff_dense;
        if (m->d_ff_dense <= 0) m->d_ff_dense = m->d_ff_exp;
        m->rope_neox = (strncmp(m->arch, "qwen", 4) == 0 || strstr(m->arch, "neox") != NULL) ? 1 : 0;
        m->dense_model = (m->n_experts <= 0) ? 1 : 0;

        if (m->dense_model) {
            m->resident_mode = (cache_slots_per_layer < 0 || cache_slots_per_layer >= m->n_layers) ? 1 : 0;
            m->cache_slots_per_layer = m->resident_mode ? m->n_layers : (int)cache_slots_per_layer;
            // Role-Aware Attention Pinning: when budget allows >0 pinned FFN layers, pin ALL Attention blocks on GPU
            // (~17% of model size) and stream ONLY FFN through a 4-slot continuous ring buffer!
            m->dense_pin_all_attn = (!m->resident_mode && m->cache_slots_per_layer > 0) ? 1 : 0;
        } else {
            m->resident_mode = (cache_slots_per_layer <= 0 || cache_slots_per_layer >= m->n_experts) ? 1 : 0;
            m->cache_slots_per_layer = m->resident_mode ? m->n_experts : (int)cache_slots_per_layer;
        }

        // Open all shard file descriptors & set F_NOCACHE in streaming mode (ZERO OS page cache pollution!)
        for (int s = 0; s < m->n_shards; s++) {
            struct stat st;
            if (stat(shard_paths[s], &st) != 0) { free(tlist); return 0; }
            m->shard_sizes[s] = (size_t)st.st_size;
            m->total_disk_bytes += (uint64_t)st.st_size;
            m->fds[s] = open(shard_paths[s], O_RDONLY);
            if (m->fds[s] < 0) { free(tlist); return 0; }
            if (m->resident_mode) {
                size_t psz = (size_t)getpagesize();
                m->mmap_lens[s] = (m->shard_sizes[s] + psz - 1) & ~(psz - 1);
                m->mmap_ptrs[s] = mmap(NULL, m->mmap_lens[s], PROT_READ, MAP_PRIVATE, m->fds[s], 0);
                if (m->mmap_ptrs[s] != MAP_FAILED) {
                    m->mmap_mtl_bufs[s] = [g_moe_dev newBufferWithBytesNoCopy:m->mmap_ptrs[s]
                                                                       length:m->mmap_lens[s]
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];
                    if (m->mmap_mtl_bufs[s] != nil) m->core_vram_bytes += (int64_t)m->shard_sizes[s];
                }
            } else if (m->use_nocache) {
                fcntl(m->fds[s], F_NOCACHE, 1);
            }
        }

        // First mark which layers are dense vs MoE and pinned vs streamed
        for (int L = 0; L < m->n_layers; L++) {
            m->layers[L].is_dense = m->dense_model ? 1 : 0;
            m->layers[L].is_pinned_dense = (m->dense_model && (m->resident_mode || L < m->cache_slots_per_layer)) ? 1 : 0;
        }
        // Detect leading dense blocks in hybrid MoE (e.g., GLM-4.5 blk.0 has ffn_gate.weight!)
        for (uint64_t i = 0; i < tcount; i++) {
            if (strncmp(tlist[i].name, "blk.", 4) == 0 && strstr(tlist[i].name, ".ffn_gate.weight") != NULL) {
                int L = atoi(tlist[i].name + 4);
                if (L >= 0 && L < m->n_layers) { m->layers[L].is_dense = 1; if (!m->dense_model) m->layers[L].is_pinned_dense = 1; }
            }
        }

        // Bind pinned tensors and record streaming file offsets!
        for (uint64_t i = 0; i < tcount; i++) {
            const MoeTensorInfo* ti = &tlist[i];
            if (strcmp(ti->name, "token_embd.weight") == 0) {
                if (m->vocab_size <= 0) m->vocab_size = (int)ti->ne[1];
                if (m->resident_mode) load_pinned_weight(m, &m->tok_embd, ti);
                else record_weight_meta(&m->tok_embd, ti); // ZERO-VRAM on-demand 1-row pread!
            } else if (strcmp(ti->name, "output_norm.weight") == 0) {
                load_pinned_weight(m, &m->out_norm, ti);
            } else if (strcmp(ti->name, "output.weight") == 0) {
                load_pinned_weight(m, &m->lm_head, ti);
            } else if (strcmp(ti->name, "rope_freqs.weight") == 0) {
                load_pinned_weight(m, &m->rope_freqs, ti);
                m->has_rope_freqs = 1;
            } else if (strncmp(ti->name, "blk.", 4) == 0) {
                int L = atoi(ti->name + 4);
                const char* dot = strchr(ti->name + 4, '.');
                if (L < 0 || L >= m->n_layers || !dot) continue;
                const char* sub = dot + 1;
                MoeLayer* ly = &m->layers[L];
                int stream_dense_w = (m->dense_model && !ly->is_pinned_dense);

                if (strcmp(sub, "attn_norm.weight") == 0) load_pinned_weight(m, &ly->attn_norm, ti);
                else if (strcmp(sub, "ffn_norm.weight") == 0 || strcmp(sub, "post_attention_norm.weight") == 0) {
                    load_pinned_weight(m, &ly->ffn_norm, ti);
                    if (!m->dense_model) {
                        ly->ffn_norm_host = (float*)malloc((size_t)ti->byte_size);
                        pread_exact(m->fds[ti->shard_idx], ly->ffn_norm_host, (size_t)ti->byte_size, ti->file_off);
                    }
                } else if (strcmp(sub, "attn_q.bias") == 0) { load_pinned_weight(m, &ly->bq, ti); ly->has_qkv_bias = 1; }
                else if (strcmp(sub, "attn_k.bias") == 0)   { load_pinned_weight(m, &ly->bk, ti); ly->has_qkv_bias = 1; }
                else if (strcmp(sub, "attn_v.bias") == 0)   { load_pinned_weight(m, &ly->bv, ti); ly->has_qkv_bias = 1; }
                else if (strcmp(sub, "attn_q_norm.weight") == 0) {
                    load_pinned_weight(m, &ly->q_norm, ti);
                    ly->has_qk_norm = 1;
                    ly->qk_norm_per_head = ((int)ti->ne[0] == m->d_head) ? 1 : 0;
                } else if (strcmp(sub, "attn_k_norm.weight") == 0) {
                    load_pinned_weight(m, &ly->k_norm, ti);
                } else if (strcmp(sub, "attn_q.weight") == 0) {
                    if (stream_dense_w && !m->dense_pin_all_attn) { record_weight_meta(&ly->wq, ti); m->total_streamable_bytes += (int64_t)ti->byte_size; }
                    else load_pinned_weight(m, &ly->wq, ti);
                } else if (strcmp(sub, "attn_k.weight") == 0) {
                    if (stream_dense_w && !m->dense_pin_all_attn) { record_weight_meta(&ly->wk, ti); m->total_streamable_bytes += (int64_t)ti->byte_size; }
                    else load_pinned_weight(m, &ly->wk, ti);
                } else if (strcmp(sub, "attn_v.weight") == 0) {
                    if (stream_dense_w && !m->dense_pin_all_attn) { record_weight_meta(&ly->wv, ti); m->total_streamable_bytes += (int64_t)ti->byte_size; }
                    else load_pinned_weight(m, &ly->wv, ti);
                } else if (strcmp(sub, "attn_output.weight") == 0) {
                    if (stream_dense_w && !m->dense_pin_all_attn) { record_weight_meta(&ly->wo, ti); m->total_streamable_bytes += (int64_t)ti->byte_size; }
                    else load_pinned_weight(m, &ly->wo, ti);
                } else if (strcmp(sub, "ffn_gate.weight") == 0) {
                    if (stream_dense_w) { record_weight_meta(&ly->ffn_gate, ti); m->total_streamable_bytes += (int64_t)ti->byte_size; }
                    else load_pinned_weight(m, &ly->ffn_gate, ti);
                } else if (strcmp(sub, "ffn_up.weight") == 0) {
                    if (stream_dense_w) { record_weight_meta(&ly->ffn_up, ti); m->total_streamable_bytes += (int64_t)ti->byte_size; }
                    else load_pinned_weight(m, &ly->ffn_up, ti);
                } else if (strcmp(sub, "ffn_down.weight") == 0) {
                    if (stream_dense_w) { record_weight_meta(&ly->ffn_down, ti); m->total_streamable_bytes += (int64_t)ti->byte_size; }
                    else load_pinned_weight(m, &ly->ffn_down, ti);
                } else if (strcmp(sub, "ffn_gate_inp.weight") == 0) {
                    load_pinned_weight(m, &ly->router, ti);
                    ly->router_host = (float*)malloc((size_t)ti->byte_size);
                    pread_exact(m->fds[ti->shard_idx], ly->router_host, (size_t)ti->byte_size, ti->file_off);
                } else if (strcmp(sub, "exp_probs_b.bias") == 0) {
                    load_pinned_weight(m, &ly->exp_bias, ti);
                    ly->exp_bias_host = (float*)malloc((size_t)ti->byte_size);
                    pread_exact(m->fds[ti->shard_idx], ly->exp_bias_host, (size_t)ti->byte_size, ti->file_off);
                    ly->has_exp_bias = 1;
                } else if (strcmp(sub, "ffn_gate_exps.weight") == 0) {
                    record_weight_meta(&ly->gate_exps, ti);
                    ly->gate_exp_bytes = ti->byte_size / (uint64_t)m->n_experts;
                    if (m->resident_mode) { ly->gate_exps.buf = m->mmap_mtl_bufs[ti->shard_idx]; ly->gate_exps.offset = ti->file_off; }
                    m->total_streamable_bytes += (int64_t)ti->byte_size;
                } else if (strcmp(sub, "ffn_up_exps.weight") == 0) {
                    record_weight_meta(&ly->up_exps, ti);
                    ly->up_exp_bytes = ti->byte_size / (uint64_t)m->n_experts;
                    if (m->resident_mode) { ly->up_exps.buf = m->mmap_mtl_bufs[ti->shard_idx]; ly->up_exps.offset = ti->file_off; }
                    m->total_streamable_bytes += (int64_t)ti->byte_size;
                } else if (strcmp(sub, "ffn_down_exps.weight") == 0) {
                    record_weight_meta(&ly->down_exps, ti);
                    ly->down_exp_bytes = ti->byte_size / (uint64_t)m->n_experts;
                    if (m->resident_mode) { ly->down_exps.buf = m->mmap_mtl_bufs[ti->shard_idx]; ly->down_exps.offset = ti->file_off; }
                    m->total_streamable_bytes += (int64_t)ti->byte_size;
                } else if (strcmp(sub, "ffn_gate_shexp.weight") == 0) {
                    load_pinned_weight(m, &ly->sh_gate, ti); ly->has_shared_exp = 1;
                } else if (strcmp(sub, "ffn_up_shexp.weight") == 0) {
                    load_pinned_weight(m, &ly->sh_up, ti);
                } else if (strcmp(sub, "ffn_down_shexp.weight") == 0) {
                    load_pinned_weight(m, &ly->sh_down, ti);
                }
            }
        }
        free(tlist);
        if (m->lm_head.buf == nil && m->tok_embd.buf != nil) m->lm_head = m->tok_embd;

        // Allocate KV caches + Streaming Pools (Dense 2-Layer Ping-Pong OR MoE Per-Layer LRU Pool!)
        size_t kv_bytes = (size_t)MOE_MAX_SEQ * (size_t)(m->n_kv * m->d_head) * 2;
        uint64_t max_wq = 0, max_wk = 0, max_wv = 0, max_wo = 0, max_fg = 0, max_fu = 0, max_fd = 0;
        for (int L = 0; L < m->n_layers; L++) {
            MoeLayer* ly = &m->layers[L];
            ly->k_cache = [g_moe_dev newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
            ly->v_cache = [g_moe_dev newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
            m->core_vram_bytes += (int64_t)(kv_bytes * 2);
            for (int e = 0; e < MOE_MAX_EXPERTS; e++) ly->exp_to_slot[e] = -1;
            for (int s = 0; s < MOE_MAX_EXPERTS; s++) { ly->slot_to_exp[s] = -1; ly->slot_lru[s] = 0; ly->slot_prefetched[s] = 0; }

            if (m->dense_model && !ly->is_pinned_dense) {
                if (ly->wq.byte_size > max_wq) max_wq = ly->wq.byte_size;
                if (ly->wk.byte_size > max_wk) max_wk = ly->wk.byte_size;
                if (ly->wv.byte_size > max_wv) max_wv = ly->wv.byte_size;
                if (ly->wo.byte_size > max_wo) max_wo = ly->wo.byte_size;
                if (ly->ffn_gate.byte_size > max_fg) max_fg = ly->ffn_gate.byte_size;
                if (ly->ffn_up.byte_size > max_fu) max_fu = ly->ffn_up.byte_size;
                if (ly->ffn_down.byte_size > max_fd) max_fd = ly->ffn_down.byte_size;
            } else if (!m->dense_model && !m->resident_mode && !ly->is_dense) {
                ly->slot_stride_bytes = align64(ly->gate_exp_bytes + ly->up_exp_bytes + ly->down_exp_bytes);
                size_t cbytes = (size_t)m->cache_slots_per_layer * (size_t)ly->slot_stride_bytes;
                ly->cache_buf = [g_moe_dev newBufferWithLength:cbytes options:MTLResourceStorageModeShared];
                m->cache_vram_bytes += (int64_t)cbytes;
            }
        }
        if (m->dense_model && !m->resident_mode) {
            uint64_t off = 0;
            if (!m->dense_pin_all_attn) {
                m->dense_off_wq = off;   off += align64(max_wq);
                m->dense_off_wk = off;   off += align64(max_wk);
                m->dense_off_wv = off;   off += align64(max_wv);
                m->dense_off_wo = off;   off += align64(max_wo);
            }
            m->dense_off_gate = off; off += align64(max_fg);
            m->dense_off_up = off;   off += align64(max_fu);
            m->dense_off_down = off; off += align64(max_fd);
            m->dense_ping_stride = off;
            int n_slots = m->dense_pin_all_attn ? 4 : 2;
            for (int s = 0; s < n_slots; s++) {
                m->dense_ping[s] = [g_moe_dev newBufferWithLength:(size_t)off options:MTLResourceStorageModeShared];
            }
            m->cache_vram_bytes = (int64_t)(off * (uint64_t)n_slots);
        } else if (m->resident_mode) {
            m->cache_vram_bytes = m->total_streamable_bytes;
        }

        // Working activation buffers
        size_t h_elems = (size_t)m->d_ff_dense;
        if ((size_t)(m->top_k * m->d_ff_exp) > h_elems) h_elems = (size_t)(m->top_k * m->d_ff_exp);
        size_t sh_elems = (size_t)(m->d_ff_dense > m->d_ff_exp ? m->d_ff_dense : m->d_ff_exp);
        m->buf_emb_row = [g_moe_dev newBufferWithLength:65536 options:MTLResourceStorageModeShared];
        m->buf_x       = [g_moe_dev newBufferWithLength:(size_t)m->d_model * 4 options:MTLResourceStorageModeShared];
        m->buf_xn      = [g_moe_dev newBufferWithLength:(size_t)m->d_model * 4 options:MTLResourceStorageModeShared];
        m->buf_q       = [g_moe_dev newBufferWithLength:(size_t)(m->n_heads * m->d_head) * 4 options:MTLResourceStorageModeShared];
        m->buf_k       = [g_moe_dev newBufferWithLength:(size_t)(m->n_kv * m->d_head) * 4 options:MTLResourceStorageModeShared];
        m->buf_v       = [g_moe_dev newBufferWithLength:(size_t)(m->n_kv * m->d_head) * 4 options:MTLResourceStorageModeShared];
        m->buf_ctx     = [g_moe_dev newBufferWithLength:(size_t)(m->n_heads * m->d_head) * 4 options:MTLResourceStorageModeShared];
        m->buf_rlogits = [g_moe_dev newBufferWithLength:(size_t)(m->n_experts > 16 ? m->n_experts : 16) * 4 options:MTLResourceStorageModeShared];
        m->buf_eidx    = [g_moe_dev newBufferWithLength:(size_t)(m->top_k > 8 ? m->top_k : 8) * 4 options:MTLResourceStorageModeShared];
        m->buf_ewt     = [g_moe_dev newBufferWithLength:(size_t)(m->top_k > 8 ? m->top_k : 8) * 4 options:MTLResourceStorageModeShared];
        m->buf_h       = [g_moe_dev newBufferWithLength:h_elems * 4 options:MTLResourceStorageModeShared];
        m->buf_sh_g    = [g_moe_dev newBufferWithLength:sh_elems * 4 options:MTLResourceStorageModeShared];
        m->buf_sh_u    = [g_moe_dev newBufferWithLength:sh_elems * 4 options:MTLResourceStorageModeShared];
        m->buf_logits  = [g_moe_dev newBufferWithLength:(size_t)m->vocab_size * 4 options:MTLResourceStorageModeShared];
        m->buf_argmax  = [g_moe_dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        m->prefetch_grp = dispatch_group_create();
        m->dense_io_grp = dispatch_group_create();
        m->io_queue = dispatch_queue_create("tenon.stream.nvme", DISPATCH_QUEUE_CONCURRENT);
        m->ring_prod_q = dispatch_queue_create("tenon.stream.ring_prod", DISPATCH_QUEUE_SERIAL);
        m->active = 1;
        return (int64_t)slot;
    }
}

static void encode_gemv_raw(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> buf, uint64_t off, uint32_t gtype,
                            id<MTLBuffer> x, id<MTLBuffer> y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {
    struct { uint32_t out_dim, in_dim, add_res; } args = { out_dim, in_dim, add_res };
    id<MTLComputePipelineState> pso = g_pso_gemv_q4;
    NSUInteger rows_per_tg = 4;
    switch (gtype) {
        case TN_GGML_Q4_0: pso = g_pso_gemv_q4;   rows_per_tg = 16; break;
        case TN_GGML_Q4_1: pso = g_pso_gemv_q4_1; rows_per_tg = 16; break;
        case TN_GGML_Q8_0: pso = g_pso_gemv_q8;   rows_per_tg = 8;  break;
        case TN_GGML_Q4_K: pso = g_pso_gemv_q4k;  rows_per_tg = 4;  break;
        case TN_GGML_Q5_K: pso = g_pso_gemv_q5k;  rows_per_tg = 8;  break;
        case TN_GGML_Q6_K: pso = g_pso_gemv_q6k;  rows_per_tg = 8;  break;
        case TN_GGML_F16:  pso = g_pso_gemv_f16;  rows_per_tg = 4;  break;
        default:           pso = g_pso_gemv_f32;  rows_per_tg = 4;  break;
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf offset:(NSUInteger)off atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1];
    [enc setBuffer:y offset:0 atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((out_dim + rows_per_tg - 1) / rows_per_tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

static void encode_gemv(id<MTLComputeCommandEncoder> enc, const MoeWeightRef* w, id<MTLBuffer> x, id<MTLBuffer> y,
                        uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {
    encode_gemv_raw(enc, w->buf, w->offset, w->ggml_type, x, y, out_dim, in_dim, add_res);
}

static void encode_rmsnorm(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, const MoeWeightRef* w, id<MTLBuffer> out,
                           uint32_t dim, uint32_t n_groups, float eps, uint32_t shared_w) {
    struct { uint32_t dim, n_groups; float eps; uint32_t shared_w; } args = { dim, n_groups, eps, shared_w };
    [enc setComputePipelineState:g_pso_rmsnorm];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:w->buf offset:(NSUInteger)w->offset atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(n_groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encode_add_bias(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, const MoeWeightRef* b, uint32_t n) {
    if (b->buf == nil) return;
    [enc setComputePipelineState:g_pso_add_bias];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:b->buf offset:(NSUInteger)b->offset atIndex:1];
    [enc setBytes:&n length:4 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encode_dense_ffn_raw(
    MoeModel* m, id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> g_buf, uint64_t g_off, uint32_t g_type,
    id<MTLBuffer> u_buf, uint64_t u_off, uint32_t u_type,
    id<MTLBuffer> d_buf, uint64_t d_off, uint32_t d_type,
    uint32_t d_ff
) {
    uint32_t dm = (uint32_t)m->d_model;
    if (g_type == TN_GGML_Q4_0 && u_type == TN_GGML_Q4_0) {
        struct { uint32_t out_dim, in_dim, add_res; } args = { d_ff, dm, 0 };
        [enc setComputePipelineState:g_pso_dense_gu_q4];
        [enc setBuffer:g_buf offset:(NSUInteger)g_off atIndex:0];
        [enc setBuffer:u_buf offset:(NSUInteger)u_off atIndex:1];
        [enc setBuffer:m->buf_xn offset:0 atIndex:2];
        [enc setBuffer:m->buf_sh_g offset:0 atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((d_ff + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        encode_gemv_raw(enc, g_buf, g_off, g_type, m->buf_xn, m->buf_sh_g, d_ff, dm, 0);
        encode_gemv_raw(enc, u_buf, u_off, u_type, m->buf_xn, m->buf_sh_u, d_ff, dm, 0);
        [enc setComputePipelineState:g_pso_swiglu_mul];
        [enc setBuffer:m->buf_sh_g offset:0 atIndex:0];
        [enc setBuffer:m->buf_sh_u offset:0 atIndex:1];
        [enc setBytes:&d_ff length:4 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(d_ff, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    encode_gemv_raw(enc, d_buf, d_off, d_type, m->buf_sh_g, m->buf_x, dm, d_ff, 1);
}

static void encode_layer_attn_body(
    MoeModel* m, id<MTLComputeCommandEncoder> enc, int L, uint32_t pos,
    id<MTLBuffer> ping_buf // nil -> use ly->wq/wk/wv/wo pinned buffers; non-nil -> use ping_buf offsets!
) {
    MoeLayer* ly = &m->layers[L];
    uint32_t dm = (uint32_t)m->d_model;
    uint32_t qdim = (uint32_t)(m->n_heads * m->d_head);
    uint32_t kvdim = (uint32_t)(m->n_kv * m->d_head);
    encode_rmsnorm(enc, m->buf_x, &ly->attn_norm, m->buf_xn, dm, 1, m->rms_eps, 0);

    if (ping_buf != nil) {
        encode_gemv_raw(enc, ping_buf, m->dense_off_wq, ly->wq.ggml_type, m->buf_xn, m->buf_q, qdim, dm, 0);
        encode_gemv_raw(enc, ping_buf, m->dense_off_wk, ly->wk.ggml_type, m->buf_xn, m->buf_k, kvdim, dm, 0);
        encode_gemv_raw(enc, ping_buf, m->dense_off_wv, ly->wv.ggml_type, m->buf_xn, m->buf_v, kvdim, dm, 0);
    } else {
        encode_gemv(enc, &ly->wq, m->buf_xn, m->buf_q, qdim, dm, 0);
        encode_gemv(enc, &ly->wk, m->buf_xn, m->buf_k, kvdim, dm, 0);
        encode_gemv(enc, &ly->wv, m->buf_xn, m->buf_v, kvdim, dm, 0);
    }
    if (ly->has_qkv_bias) {
        encode_add_bias(enc, m->buf_q, &ly->bq, qdim);
        encode_add_bias(enc, m->buf_k, &ly->bk, kvdim);
        encode_add_bias(enc, m->buf_v, &ly->bv, kvdim);
    }
    if (ly->has_qk_norm) {
        if (ly->qk_norm_per_head) {
            encode_rmsnorm(enc, m->buf_q, &ly->q_norm, m->buf_q, (uint32_t)m->d_head, (uint32_t)m->n_heads, m->rms_eps, 1);
            encode_rmsnorm(enc, m->buf_k, &ly->k_norm, m->buf_k, (uint32_t)m->d_head, (uint32_t)m->n_kv, m->rms_eps, 1);
        } else {
            encode_rmsnorm(enc, m->buf_q, &ly->q_norm, m->buf_q, qdim, 1, m->rms_eps, 0);
            encode_rmsnorm(enc, m->buf_k, &ly->k_norm, m->buf_k, kvdim, 1, m->rms_eps, 0);
        }
    }

    struct { uint32_t n_heads, n_kv, d_head, rope_dim, pos; float theta; uint32_t rope_neox, has_freqs; } rargs = {
        (uint32_t)m->n_heads, (uint32_t)m->n_kv, (uint32_t)m->d_head, (uint32_t)m->rope_dim,
        pos, m->rope_theta, (uint32_t)m->rope_neox, (uint32_t)m->has_rope_freqs
    };
    [enc setComputePipelineState:g_pso_rope_kv];
    [enc setBuffer:m->buf_q offset:0 atIndex:0];
    [enc setBuffer:m->buf_k offset:0 atIndex:1];
    [enc setBuffer:m->buf_v offset:0 atIndex:2];
    [enc setBuffer:ly->k_cache offset:0 atIndex:3];
    [enc setBuffer:ly->v_cache offset:0 atIndex:4];
    [enc setBuffer:(m->has_rope_freqs ? m->rope_freqs.buf : m->buf_argmax)
            offset:(m->has_rope_freqs ? (NSUInteger)m->rope_freqs.offset : 0) atIndex:5];
    [enc setBytes:&rargs length:sizeof(rargs) atIndex:6];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)(m->d_head / 2), (NSUInteger)m->n_heads, 1)
   threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    struct { uint32_t n_heads, n_kv, d_head, kv_len; } aargs = {
        (uint32_t)m->n_heads, (uint32_t)m->n_kv, (uint32_t)m->d_head, pos + 1u
    };
    [enc setComputePipelineState:g_pso_flash_dec];
    [enc setBuffer:m->buf_q offset:0 atIndex:0];
    [enc setBuffer:ly->k_cache offset:0 atIndex:1];
    [enc setBuffer:ly->v_cache offset:0 atIndex:2];
    [enc setBuffer:m->buf_ctx offset:0 atIndex:3];
    [enc setBytes:&aargs length:sizeof(aargs) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    if (ping_buf != nil) {
        encode_gemv_raw(enc, ping_buf, m->dense_off_wo, ly->wo.ggml_type, m->buf_ctx, m->buf_x, dm, qdim, 1);
    } else {
        encode_gemv(enc, &ly->wo, m->buf_ctx, m->buf_x, dm, qdim, 1);
    }
    encode_rmsnorm(enc, m->buf_x, &ly->ffn_norm, m->buf_xn, dm, 1, m->rms_eps, 0);
}

static int pick_lru_victim(MoeModel* m, MoeLayer* ly, const int* pinned, int n_pinned) {
    int victim = -1;
    uint64_t best_ts = UINT64_MAX;
    for (int s = 0; s < m->cache_slots_per_layer; s++) {
        int is_pin = 0;
        for (int p = 0; p < n_pinned; p++) if (pinned[p] == s) { is_pin = 1; break; }
        if (is_pin) continue;
        if (ly->slot_to_exp[s] < 0) return s;
        if (ly->slot_lru[s] < best_ts) { best_ts = ly->slot_lru[s]; victim = s; }
    }
    return (victim >= 0) ? victim : 0;
}

static void cpu_router_topk_from_logits(
    const float* logits, const float* exp_bias, int n_exp, int top_k,
    int gating_func, int norm_wt, float w_scale, int* out_e, float* out_w
) {
    float probs[MOE_MAX_EXPERTS];
    float scores[MOE_MAX_EXPERTS];
    if (gating_func == 2) {
        for (int e = 0; e < n_exp; e++) {
            float s = 1.0f / (1.0f + expf(-logits[e]));
            probs[e] = s;
            scores[e] = exp_bias ? (s + exp_bias[e]) : s;
        }
    } else {
        float mx = -1e30f;
        for (int e = 0; e < n_exp; e++) if (logits[e] > mx) mx = logits[e];
        float sum = 0.0f;
        for (int e = 0; e < n_exp; e++) { probs[e] = expf(logits[e] - mx); sum += probs[e]; }
        float inv = 1.0f / (sum > 1e-12f ? sum : 1e-12f);
        for (int e = 0; e < n_exp; e++) { probs[e] *= inv; scores[e] = probs[e]; }
    }
    float wsum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        int be = 0; float bs = -1e30f;
        for (int e = 0; e < n_exp; e++) if (scores[e] > bs) { bs = scores[e]; be = e; }
        out_e[k] = be; out_w[k] = probs[be]; wsum += probs[be]; scores[be] = -1e30f;
    }
    float mult = w_scale;
    if (norm_wt && wsum > 1e-12f) mult /= wsum;
    for (int k = 0; k < top_k; k++) out_w[k] *= mult;
}

static void schedule_next_layer_prefetch(MoeModel* m, int next_L) {
    if (!m->enable_overlap || next_L >= m->n_layers) return;
    if (m->cache_slots_per_layer < m->top_k + 4) return; // Don't thrash extreme minimal Top-K caches!
    MoeLayer* nly = &m->layers[next_L];
    if (nly->is_dense || !nly->router_host) return;
    const float* x_cur = (const float*)[m->buf_x contents];
    int dm = m->d_model;
    float x_norm[8192];
    if (dm <= 8192 && nly->ffn_norm_host) {
        float sq = 0.0f;
        for (int i = 0; i < dm; i++) sq += x_cur[i] * x_cur[i];
        float inv = 1.0f / sqrtf(sq / (float)dm + m->rms_eps);
        for (int i = 0; i < dm; i++) x_norm[i] = x_cur[i] * inv * nly->ffn_norm_host[i];
    } else {
        for (int i = 0; i < dm && i < 8192; i++) x_norm[i] = x_cur[i];
    }
    float pred_logits[MOE_MAX_EXPERTS];
    for (int e = 0; e < m->n_experts; e++) {
        const float* wr = nly->router_host + (size_t)e * (size_t)dm;
        float acc = 0.0f;
        for (int i = 0; i < dm; i++) acc += wr[i] * x_norm[i];
        pred_logits[e] = acc;
    }
    int pe[MOE_MAX_TOPK]; float pw[MOE_MAX_TOPK];
    cpu_router_topk_from_logits(pred_logits, nly->exp_bias_host, m->n_experts, m->top_k,
                                m->gating_func, m->expert_weights_norm, m->expert_weights_scale, pe, pw);
    int pinned[MOE_MAX_TOPK]; int npin = 0;
    for (int k = 0; k < m->top_k; k++) {
        int sl = nly->exp_to_slot[pe[k]];
        if (sl >= 0) pinned[npin++] = sl;
    }
    int n_pref = 0;
    for (int k = 0; k < m->top_k && n_pref < 2; k++) {
        int eid = pe[k];
        if (nly->exp_to_slot[eid] >= 0) continue;
        int vs = pick_lru_victim(m, nly, pinned, npin);
        pinned[npin++] = vs;
        int old_e = nly->slot_to_exp[vs];
        if (old_e >= 0) nly->exp_to_slot[old_e] = -1;
        nly->slot_to_exp[vs] = eid;
        nly->exp_to_slot[eid] = vs;
        nly->slot_lru[vs] = ++m->lru_clock;
        nly->slot_prefetched[vs] = 1;
        n_pref++;
        dispatch_group_async(m->prefetch_grp, m->io_queue, ^{
            load_expert_into_slot(m, next_L, eid, vs);
        });
    }
}

static void encode_embed_step(MoeModel* m, id<MTLComputeCommandEncoder> enc, uint32_t tok_id) {
    if (m->resident_mode) {
        struct { uint32_t tok_id, d_model, ggml_type; } eargs = { tok_id, (uint32_t)m->d_model, m->tok_embd.ggml_type };
        [enc setComputePipelineState:g_pso_embed];
        [enc setBuffer:m->tok_embd.buf offset:(NSUInteger)m->tok_embd.offset atIndex:0];
        [enc setBuffer:m->buf_x offset:0 atIndex:1];
        [enc setBytes:&eargs length:sizeof(eargs) atIndex:2];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)m->d_model, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    } else {
        uint64_t row_b = ggml_type_bytes(m->tok_embd.ggml_type, (uint64_t)m->d_model);
        pread_exact(m->fds[m->tok_embd.shard_idx], [m->buf_emb_row contents], (size_t)row_b,
                    m->tok_embd.file_off + (uint64_t)tok_id * row_b);
        struct { uint32_t tok_id, d_model, ggml_type; } eargs = { 0u, (uint32_t)m->d_model, m->tok_embd.ggml_type };
        [enc setComputePipelineState:g_pso_embed];
        [enc setBuffer:m->buf_emb_row offset:0 atIndex:0];
        [enc setBuffer:m->buf_x offset:0 atIndex:1];
        [enc setBytes:&eargs length:sizeof(eargs) atIndex:2];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)m->d_model, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
}

static void encode_final_lm_head(MoeModel* m, id<MTLComputeCommandEncoder> enc) {
    encode_rmsnorm(enc, m->buf_x, &m->out_norm, m->buf_xn, (uint32_t)m->d_model, 1, m->rms_eps, 0);
    encode_gemv(enc, &m->lm_head, m->buf_xn, m->buf_logits, (uint32_t)m->vocab_size, (uint32_t)m->d_model, 0);
    uint32_t v_u = (uint32_t)m->vocab_size;
    [enc setComputePipelineState:g_pso_argmax];
    [enc setBuffer:m->buf_logits offset:0 atIndex:0];
    [enc setBuffer:m->buf_argmax offset:0 atIndex:1];
    [enc setBytes:&v_u length:4 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

int64_t tn_moe_step(int64_t handle, int64_t tok_id, int64_t pos) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return 0;
    @autoreleasepool {
        MoeModel* m = &g_moe[handle];
        double t0 = moe_now_ms();
        double ssd_acc_ms = 0.0;

        // =========================================================================
        // PATH A: FULL RESIDENT MODE (Single CommandBuffer, Zero CPU Sync!)
        // =========================================================================
        if (m->resident_mode) {
            id<MTLCommandBuffer> cmd = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            encode_embed_step(m, enc, (uint32_t)tok_id);

            for (int L = 0; L < m->n_layers; L++) {
                MoeLayer* ly = &m->layers[L];
                encode_layer_attn_body(m, enc, L, (uint32_t)pos, nil);
                if (ly->is_dense) {
                    encode_dense_ffn_raw(m, enc,
                        ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                        ly->ffn_up.buf,   ly->ffn_up.offset,   ly->ffn_up.ggml_type,
                        ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type,
                        (uint32_t)m->d_ff_dense);
                } else {
                    encode_gemv(enc, &ly->router, m->buf_xn, m->buf_rlogits, (uint32_t)m->n_experts, (uint32_t)m->d_model, 0);
                    struct { uint32_t ne, tk, nw, gf, hb; float ws; } rt = {
                        (uint32_t)m->n_experts, (uint32_t)m->top_k, (uint32_t)m->expert_weights_norm,
                        (uint32_t)m->gating_func, (uint32_t)ly->has_exp_bias, m->expert_weights_scale
                    };
                    [enc setComputePipelineState:g_pso_router];
                    [enc setBuffer:m->buf_rlogits offset:0 atIndex:0];
                    [enc setBuffer:(ly->has_exp_bias ? ly->exp_bias.buf : m->buf_argmax)
                            offset:(ly->has_exp_bias ? (NSUInteger)ly->exp_bias.offset : 0) atIndex:1];
                    [enc setBuffer:m->buf_eidx offset:0 atIndex:2];
                    [enc setBuffer:m->buf_ewt offset:0 atIndex:3];
                    [enc setBytes:&rt length:sizeof(rt) atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                    if (ly->has_shared_exp) {
                        encode_dense_ffn_raw(m, enc,
                            ly->sh_gate.buf, ly->sh_gate.offset, ly->sh_gate.ggml_type,
                            ly->sh_up.buf,   ly->sh_up.offset,   ly->sh_up.ggml_type,
                            ly->sh_down.buf, ly->sh_down.offset, ly->sh_down.ggml_type,
                            (uint32_t)m->d_ff_exp);
                    }
                    struct { uint32_t dm, df, na; uint64_t gb, ub, db, sb; } margs = {
                        (uint32_t)m->d_model, (uint32_t)m->d_ff_exp, (uint32_t)m->top_k,
                        ly->gate_exp_bytes, ly->up_exp_bytes, ly->down_exp_bytes, 0
                    };
                    [enc setComputePipelineState:g_pso_moe_gu_q4];
                    [enc setBuffer:ly->gate_exps.buf offset:(NSUInteger)ly->gate_exps.offset atIndex:0];
                    [enc setBuffer:ly->up_exps.buf offset:(NSUInteger)ly->up_exps.offset atIndex:1];
                    [enc setBuffer:m->buf_xn offset:0 atIndex:2];
                    [enc setBuffer:m->buf_eidx offset:0 atIndex:3];
                    [enc setBuffer:m->buf_h offset:0 atIndex:4];
                    [enc setBytes:&margs length:sizeof(margs) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((m->d_ff_exp + 7) / 8), (NSUInteger)m->top_k, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

                    [enc setComputePipelineState:(ly->down_exps.ggml_type == TN_GGML_Q4_1 ? g_pso_moe_down_q4_1 : g_pso_moe_down_q4)];
                    [enc setBuffer:ly->down_exps.buf offset:(NSUInteger)ly->down_exps.offset atIndex:0];
                    [enc setBuffer:m->buf_h offset:0 atIndex:1];
                    [enc setBuffer:m->buf_eidx offset:0 atIndex:2];
                    [enc setBuffer:m->buf_ewt offset:0 atIndex:3];
                    [enc setBuffer:m->buf_x offset:0 atIndex:4];
                    [enc setBytes:&margs length:sizeof(margs) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((m->d_model + 15) / 16), 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    m->cache_hits += m->top_k;
                }
            }
            encode_final_lm_head(m, enc);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            double tot = moe_now_ms() - t0;
            m->last_total_ms = tot; m->last_ssd_ms = 0.0; m->last_gpu_ms = tot;
            return (int64_t)((uint32_t*)[m->buf_argmax contents])[0];
        }

        // =========================================================================
        // PATH B: DENSE STREAMING MODE (2-Slot Ultra-Low <2.5 GiB OR Role-Aware All-Attn + 4-Slot Continuous FFN Ring!)
        // =========================================================================
        if (m->dense_model) {
            int first_stream = m->cache_slots_per_layer;
            int n_ring = m->dense_pin_all_attn ? 4 : 2;
            dispatch_semaphore_t sem_empty = nil;
            dispatch_semaphore_t sem_full = nil;

            // If overlap is enabled and there are layers to stream, kick off a CONTINUOUS producer thread right away!
            // It immediately starts filling ring slots `0..n_ring-1` WHILE GPU runs pinned prefix layers 0..first_stream-1!
            if (m->enable_overlap && first_stream < m->n_layers) {
                sem_empty = dispatch_semaphore_create(n_ring);
                sem_full  = dispatch_semaphore_create(0);
                int n_layers_cap = m->n_layers;
                dispatch_group_async(m->dense_io_grp, m->ring_prod_q, ^{
                    for (int pL = first_stream; pL < n_layers_cap; pL++) {
                        dispatch_semaphore_wait(sem_empty, DISPATCH_TIME_FOREVER);
                        load_dense_layer_into_ping(m, pL, (pL - first_stream) % n_ring, 1);
                        dispatch_semaphore_signal(sem_full);
                    }
                });
            }

            // 1. Run Embedding + all pinned prefix layers (0 .. first_stream-1) in ONE CommandBuffer!
            {
                id<MTLCommandBuffer> pcmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> penc = [pcmd computeCommandEncoder];
                encode_embed_step(m, penc, (uint32_t)tok_id);
                for (int L = 0; L < first_stream && L < m->n_layers; L++) {
                    MoeLayer* ly = &m->layers[L];
                    encode_layer_attn_body(m, penc, L, (uint32_t)pos, nil);
                    encode_dense_ffn_raw(m, penc,
                        ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                        ly->ffn_up.buf,   ly->ffn_up.offset,   ly->ffn_up.ggml_type,
                        ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type,
                        (uint32_t)m->d_ff_dense);
                    m->cache_hits++;
                }
                [penc endEncoding];
                [pcmd commit];
                [pcmd waitUntilCompleted];
            }

            // 2. Consume remaining layers from continuous 2-slot or 4-slot Ring Buffer (instant slot recycling!)
            for (int L = first_stream; L < m->n_layers; L++) {
                MoeLayer* ly = &m->layers[L];
                int cur = (L - first_stream) % n_ring;
                if (!m->enable_overlap) {
                    double io0 = moe_now_ms();
                    load_dense_layer_into_ping(m, L, cur, 0);
                    ssd_acc_ms += (moe_now_ms() - io0);
                    m->cache_misses++;
                } else {
                    m->cache_misses++;
                    dispatch_semaphore_wait(sem_full, DISPATCH_TIME_FOREVER);
                }

                id<MTLCommandBuffer> lcmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> lenc = [lcmd computeCommandEncoder];
                // When `dense_pin_all_attn == 1`, Attention reads directly from pinned GPU VRAM (`nil`), only FFN reads ring!
                encode_layer_attn_body(m, lenc, L, (uint32_t)pos, m->dense_pin_all_attn ? nil : m->dense_ping[cur]);
                encode_dense_ffn_raw(m, lenc,
                    m->dense_ping[cur], m->dense_off_gate, ly->ffn_gate.ggml_type,
                    m->dense_ping[cur], m->dense_off_up,   ly->ffn_up.ggml_type,
                    m->dense_ping[cur], m->dense_off_down, ly->ffn_down.ggml_type,
                    (uint32_t)m->d_ff_dense);
                [lenc endEncoding];
                [lcmd commit];
                [lcmd waitUntilCompleted];
                if (m->enable_overlap) dispatch_semaphore_signal(sem_empty); // Instantly recycle slot for producer!
            }
            if (m->enable_overlap) dispatch_group_wait(m->dense_io_grp, DISPATCH_TIME_FOREVER);

            // 3. Final Norm + LM Head + Argmax
            id<MTLCommandBuffer> fcmd = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> fenc = [fcmd computeCommandEncoder];
            encode_final_lm_head(m, fenc);
            [fenc endEncoding];
            [fcmd commit];
            [fcmd waitUntilCompleted];

            double tot = moe_now_ms() - t0;
            m->last_total_ms = tot; m->last_ssd_ms = ssd_acc_ms;
            m->last_gpu_ms = (tot > ssd_acc_ms) ? (tot - ssd_acc_ms) : 0.0;
            return (int64_t)((uint32_t*)[m->buf_argmax contents])[0];
        }

        // =========================================================================
        // PATH C: SPARSE MoE ON-DEMAND STREAMING MODE (Bounded Per-Layer LRU Pool + Hit||Miss Overlap!)
        // =========================================================================
        {
            id<MTLCommandBuffer> ecmd = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [ecmd computeCommandEncoder];
            encode_embed_step(m, enc, (uint32_t)tok_id);
            [enc endEncoding];
            [ecmd commit];
        }

        for (int L = 0; L < m->n_layers; L++) {
            MoeLayer* ly = &m->layers[L];
            if (ly->is_dense) {
                // Leading Dense block (e.g. GLM-4.5 blk.0) — pinned in core VRAM!
                id<MTLCommandBuffer> dcmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> denc = [dcmd computeCommandEncoder];
                encode_layer_attn_body(m, denc, L, (uint32_t)pos, nil);
                encode_dense_ffn_raw(m, denc,
                    ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                    ly->ffn_up.buf,   ly->ffn_up.offset,   ly->ffn_up.ggml_type,
                    ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type,
                    (uint32_t)m->d_ff_dense);
                [denc endEncoding];
                [dcmd commit];
                [dcmd waitUntilCompleted];
                continue;
            }

            // Step 1: Attention + FFN-Norm + Router logits on GPU
            id<MTLCommandBuffer> acmd = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> aenc = [acmd computeCommandEncoder];
            encode_layer_attn_body(m, aenc, L, (uint32_t)pos, nil);
            encode_gemv(aenc, &ly->router, m->buf_xn, m->buf_rlogits, (uint32_t)m->n_experts, (uint32_t)m->d_model, 0);
            [aenc endEncoding];
            [acmd commit];
            [acmd waitUntilCompleted];

            // Ensure any speculative prefetch targeting Layer L has finished writing cache slots!
            dispatch_group_wait(m->prefetch_grp, DISPATCH_TIME_FOREVER);

            // Step 2: Top-K Router selection + HIT / MISS partition
            int top_e[MOE_MAX_TOPK]; float top_w[MOE_MAX_TOPK];
            cpu_router_topk_from_logits((const float*)[m->buf_rlogits contents], ly->exp_bias_host,
                                        m->n_experts, m->top_k, m->gating_func,
                                        m->expert_weights_norm, m->expert_weights_scale, top_e, top_w);

            int hit_slots[MOE_MAX_TOPK]; float hit_w[MOE_MAX_TOPK]; int n_hits = 0;
            int miss_eids[MOE_MAX_TOPK]; int miss_slots[MOE_MAX_TOPK]; float miss_w[MOE_MAX_TOPK]; int n_misses = 0;
            int pinned[MOE_MAX_TOPK]; int npin = 0;

            for (int k = 0; k < m->top_k; k++) {
                int eid = top_e[k];
                int cslot = ly->exp_to_slot[eid];
                if (cslot >= 0) {
                    m->cache_hits++;
                    if (ly->slot_prefetched[cslot]) { m->prefetch_hits++; ly->slot_prefetched[cslot] = 0; }
                    ly->slot_lru[cslot] = ++m->lru_clock;
                    hit_slots[n_hits] = cslot; hit_w[n_hits] = top_w[k]; n_hits++;
                    pinned[npin++] = cslot;
                } else {
                    m->cache_misses++;
                    miss_eids[n_misses] = eid; miss_w[n_misses] = top_w[k]; n_misses++;
                }
            }
            for (int mi = 0; mi < n_misses; mi++) {
                int vs = pick_lru_victim(m, ly, pinned, npin);
                pinned[npin++] = vs;
                int old_e = ly->slot_to_exp[vs];
                if (old_e >= 0) ly->exp_to_slot[old_e] = -1;
                ly->slot_to_exp[vs] = miss_eids[mi];
                ly->exp_to_slot[miss_eids[mi]] = vs;
                ly->slot_lru[vs] = ++m->lru_clock;
                ly->slot_prefetched[vs] = 0;
                miss_slots[mi] = vs;
            }

            id<MTLComputePipelineState> down_pso = (ly->down_exps.ggml_type == TN_GGML_Q4_1)
                ? g_pso_moe_slot_down_q4_1 : g_pso_moe_slot_down_q4;

            if (!m->enable_overlap) {
                // Naive Synchronous Serial Baseline
                double io0 = moe_now_ms();
                for (int mi = 0; mi < n_misses; mi++) load_expert_into_slot(m, L, miss_eids[mi], miss_slots[mi]);
                ssd_acc_ms += (moe_now_ms() - io0);

                uint32_t all_s[MOE_MAX_TOPK]; float all_w[MOE_MAX_TOPK]; int tot_a = 0;
                for (int hi = 0; hi < n_hits; hi++) { all_s[tot_a] = (uint32_t)hit_slots[hi]; all_w[tot_a] = hit_w[hi]; tot_a++; }
                for (int mi = 0; mi < n_misses; mi++) { all_s[tot_a] = (uint32_t)miss_slots[mi]; all_w[tot_a] = miss_w[mi]; tot_a++; }
                memcpy([m->buf_eidx contents], all_s, (size_t)tot_a * 4);
                memcpy([m->buf_ewt contents], all_w, (size_t)tot_a * 4);

                id<MTLCommandBuffer> mcmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> menc = [mcmd computeCommandEncoder];
                if (ly->has_shared_exp) {
                    encode_dense_ffn_raw(m, menc,
                        ly->sh_gate.buf, ly->sh_gate.offset, ly->sh_gate.ggml_type,
                        ly->sh_up.buf,   ly->sh_up.offset,   ly->sh_up.ggml_type,
                        ly->sh_down.buf, ly->sh_down.offset, ly->sh_down.ggml_type,
                        (uint32_t)m->d_ff_exp);
                }
                struct { uint32_t dm, df, na; uint64_t gb, ub, db, sb; } margs = {
                    (uint32_t)m->d_model, (uint32_t)m->d_ff_exp, (uint32_t)tot_a,
                    ly->gate_exp_bytes, ly->up_exp_bytes, ly->down_exp_bytes, ly->slot_stride_bytes
                };
                [menc setComputePipelineState:g_pso_moe_slot_gu_q4];
                [menc setBuffer:ly->cache_buf offset:0 atIndex:0];
                [menc setBuffer:m->buf_xn offset:0 atIndex:1];
                [menc setBuffer:m->buf_eidx offset:0 atIndex:2];
                [menc setBuffer:m->buf_h offset:0 atIndex:3];
                [menc setBytes:&margs length:sizeof(margs) atIndex:4];
                [menc dispatchThreadgroups:MTLSizeMake((NSUInteger)((m->d_ff_exp + 7) / 8), (NSUInteger)tot_a, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [menc setComputePipelineState:down_pso];
                [menc setBuffer:ly->cache_buf offset:0 atIndex:0];
                [menc setBuffer:m->buf_h offset:0 atIndex:1];
                [menc setBuffer:m->buf_eidx offset:0 atIndex:2];
                [menc setBuffer:m->buf_ewt offset:0 atIndex:3];
                [menc setBuffer:m->buf_x offset:0 atIndex:4];
                [menc setBytes:&margs length:sizeof(margs) atIndex:5];
                [menc dispatchThreadgroups:MTLSizeMake((NSUInteger)((m->d_model + 15) / 16), 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                [menc endEncoding];
                [mcmd commit];
                [mcmd waitUntilCompleted];
            } else {
                // TENON SMART ASYNC STREAMER: (SharedExp + HIT Experts) GPU Compute || Parallel NVMe `pread()` for MISSES!
                id<MTLCommandBuffer> hcmd = nil;
                if (n_hits > 0 || ly->has_shared_exp) {
                    if (n_hits > 0) {
                        uint32_t hs[MOE_MAX_TOPK];
                        for (int hi = 0; hi < n_hits; hi++) hs[hi] = (uint32_t)hit_slots[hi];
                        memcpy([m->buf_eidx contents], hs, (size_t)n_hits * 4);
                        memcpy([m->buf_ewt contents], hit_w, (size_t)n_hits * 4);
                    }
                    hcmd = [g_moe_q commandBuffer];
                    id<MTLComputeCommandEncoder> henc = [hcmd computeCommandEncoder];
                    if (ly->has_shared_exp) {
                        encode_dense_ffn_raw(m, henc,
                            ly->sh_gate.buf, ly->sh_gate.offset, ly->sh_gate.ggml_type,
                            ly->sh_up.buf,   ly->sh_up.offset,   ly->sh_up.ggml_type,
                            ly->sh_down.buf, ly->sh_down.offset, ly->sh_down.ggml_type,
                            (uint32_t)m->d_ff_exp);
                    }
                    if (n_hits > 0) {
                        struct { uint32_t dm, df, na; uint64_t gb, ub, db, sb; } margs = {
                            (uint32_t)m->d_model, (uint32_t)m->d_ff_exp, (uint32_t)n_hits,
                            ly->gate_exp_bytes, ly->up_exp_bytes, ly->down_exp_bytes, ly->slot_stride_bytes
                        };
                        [henc setComputePipelineState:g_pso_moe_slot_gu_q4];
                        [henc setBuffer:ly->cache_buf offset:0 atIndex:0];
                        [henc setBuffer:m->buf_xn offset:0 atIndex:1];
                        [henc setBuffer:m->buf_eidx offset:0 atIndex:2];
                        [henc setBuffer:m->buf_h offset:0 atIndex:3];
                        [henc setBytes:&margs length:sizeof(margs) atIndex:4];
                        [henc dispatchThreadgroups:MTLSizeMake((NSUInteger)((m->d_ff_exp + 7) / 8), (NSUInteger)n_hits, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [henc setComputePipelineState:down_pso];
                        [henc setBuffer:ly->cache_buf offset:0 atIndex:0];
                        [henc setBuffer:m->buf_h offset:0 atIndex:1];
                        [henc setBuffer:m->buf_eidx offset:0 atIndex:2];
                        [henc setBuffer:m->buf_ewt offset:0 atIndex:3];
                        [henc setBuffer:m->buf_x offset:0 atIndex:4];
                        [henc setBytes:&margs length:sizeof(margs) atIndex:5];
                        [henc dispatchThreadgroups:MTLSizeMake((NSUInteger)((m->d_model + 15) / 16), 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    }
                    [henc endEncoding];
                    [hcmd commit];
                }

                if (n_misses > 0) {
                    double io0 = moe_now_ms();
                    int* p_eids = miss_eids;
                    int* p_slots = miss_slots;
                    dispatch_apply((size_t)n_misses, m->io_queue, ^(size_t mi) {
                        load_expert_into_slot(m, L, p_eids[mi], p_slots[mi]);
                    });
                    ssd_acc_ms += (moe_now_ms() - io0);
                }

                if (hcmd) [hcmd waitUntilCompleted];

                if (n_misses == 0) {
                    schedule_next_layer_prefetch(m, L + 1);
                } else {
                    uint32_t ms_u[MOE_MAX_TOPK];
                    for (int mi = 0; mi < n_misses; mi++) ms_u[mi] = (uint32_t)miss_slots[mi];
                    memcpy([m->buf_eidx contents], ms_u, (size_t)n_misses * 4);
                    memcpy([m->buf_ewt contents], miss_w, (size_t)n_misses * 4);
                    id<MTLCommandBuffer> mcmd = [g_moe_q commandBuffer];
                    id<MTLComputeCommandEncoder> menc = [mcmd computeCommandEncoder];
                    struct { uint32_t dm, df, na; uint64_t gb, ub, db, sb; } margs = {
                        (uint32_t)m->d_model, (uint32_t)m->d_ff_exp, (uint32_t)n_misses,
                        ly->gate_exp_bytes, ly->up_exp_bytes, ly->down_exp_bytes, ly->slot_stride_bytes
                    };
                    [menc setComputePipelineState:g_pso_moe_slot_gu_q4];
                    [menc setBuffer:ly->cache_buf offset:0 atIndex:0];
                    [menc setBuffer:m->buf_xn offset:0 atIndex:1];
                    [menc setBuffer:m->buf_eidx offset:0 atIndex:2];
                    [menc setBuffer:m->buf_h offset:0 atIndex:3];
                    [menc setBytes:&margs length:sizeof(margs) atIndex:4];
                    [menc dispatchThreadgroups:MTLSizeMake((NSUInteger)((m->d_ff_exp + 7) / 8), (NSUInteger)n_misses, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [menc setComputePipelineState:down_pso];
                    [menc setBuffer:ly->cache_buf offset:0 atIndex:0];
                    [menc setBuffer:m->buf_h offset:0 atIndex:1];
                    [menc setBuffer:m->buf_eidx offset:0 atIndex:2];
                    [menc setBuffer:m->buf_ewt offset:0 atIndex:3];
                    [menc setBuffer:m->buf_x offset:0 atIndex:4];
                    [menc setBytes:&margs length:sizeof(margs) atIndex:5];
                    [menc dispatchThreadgroups:MTLSizeMake((NSUInteger)((m->d_model + 15) / 16), 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                    [menc endEncoding];
                    [mcmd commit];
                    schedule_next_layer_prefetch(m, L + 1);
                    [mcmd waitUntilCompleted];
                }
            }
        }

        id<MTLCommandBuffer> fcmd = [g_moe_q commandBuffer];
        id<MTLComputeCommandEncoder> fenc = [fcmd computeCommandEncoder];
        encode_final_lm_head(m, fenc);
        [fenc endEncoding];
        [fcmd commit];
        [fcmd waitUntilCompleted];

        double tot = moe_now_ms() - t0;
        m->last_total_ms = tot;
        m->last_ssd_ms = ssd_acc_ms;
        m->last_gpu_ms = (tot > ssd_acc_ms) ? (tot - ssd_acc_ms) : 0.0;
        return (int64_t)((uint32_t*)[m->buf_argmax contents])[0];
    }
}

int64_t tn_moe_info(int64_t handle, int64_t field_id) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return 0;
    MoeModel* m = &g_moe[handle];
    switch (field_id) {
        case 0: return m->n_layers;
        case 1: return m->d_model;
        case 2: return m->dense_model ? m->d_ff_dense : m->d_ff_exp;
        case 3: return m->n_heads;
        case 4: return m->n_kv;
        case 5: return m->n_experts;
        case 6: return m->top_k;
        case 7: return m->vocab_size;
        case 8: return m->core_vram_bytes;
        case 9: return m->cache_vram_bytes;
        case 10: return m->total_streamable_bytes;
        case 11: return m->bos_id;
        case 12: return m->eos_id;
        case 13: return m->cache_hits;
        case 14: return m->cache_misses;
        case 15: return m->prefetch_hits;
        case 16: return m->ssd_bytes_read;
        case 17: return get_os_phys_footprint_bytes();
        case 18: return get_os_peak_rss_bytes();
        case 19: return (int64_t)m->total_disk_bytes;
        case 20: return m->n_shards;
        default: return 0;
    }
}

int64_t tn_moe_reset(int64_t handle) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return 0;
    MoeModel* m = &g_moe[handle];
    dispatch_group_wait(m->prefetch_grp, DISPATCH_TIME_FOREVER);
    dispatch_group_wait(m->dense_io_grp, DISPATCH_TIME_FOREVER);
    m->cache_hits = 0;
    m->cache_misses = 0;
    m->prefetch_hits = 0;
    m->ssd_bytes_read = 0;
    return 1;
}

int64_t tn_moe_token_cstr(int64_t handle, int64_t tok_id) {
    static char g_tok_buf[512];
    g_tok_buf[0] = 0;
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return (int64_t)(uintptr_t)g_tok_buf;
    MoeModel* m = &g_moe[handle];
    if (!m->vocab_tokens || tok_id < 0 || tok_id >= m->vocab_size || !m->vocab_tokens[tok_id]) {
        return (int64_t)(uintptr_t)g_tok_buf;
    }
    const unsigned char* s = (const unsigned char*)m->vocab_tokens[tok_id];
    size_t o = 0;
    for (size_t i = 0; s[i] && o + 2 < sizeof(g_tok_buf); ) {
        if (s[i] == 0xC4 && s[i + 1] == 0xA0) { g_tok_buf[o++] = ' '; i += 2; }
        else if (s[i] == 0xC4 && s[i + 1] == 0x8A) { g_tok_buf[o++] = '\n'; i += 2; }
        else if (s[i] == 0xE2 && s[i + 1] == 0x96 && s[i + 2] == 0x81) { g_tok_buf[o++] = ' '; i += 3; }
        else { g_tok_buf[o++] = (char)s[i++]; }
    }
    g_tok_buf[o] = 0;
    return (int64_t)(uintptr_t)g_tok_buf;
}

int64_t tn_moe_tokenize(int64_t handle, int64_t text_cstr, int64_t out_ids_ptr, int64_t max_tokens) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !text_cstr || !out_ids_ptr) return 0;
    MoeModel* m = &g_moe[handle];
    if (!m->vocab_tokens) return 0;
    const char* raw = (const char*)(uintptr_t)text_cstr;
    char enc[2048];
    size_t elen = 0;
    for (size_t i = 0; raw[i] && elen + 3 < sizeof(enc); i++) {
        if (raw[i] == ' ') { enc[elen++] = (char)0xC4; enc[elen++] = (char)0xA0; }
        else if (raw[i] == '\n') { enc[elen++] = (char)0xC4; enc[elen++] = (char)0x8A; }
        else { enc[elen++] = raw[i]; }
    }
    enc[elen] = 0;
    int64_t* out = (int64_t*)(uintptr_t)out_ids_ptr;
    int64_t n_out = 0;
    size_t pos = 0;
    while (pos < elen && n_out < max_tokens) {
        int best_id = -1;
        size_t best_len = 0;
        for (int v = 0; v < m->vocab_size; v++) {
            const char* tv = m->vocab_tokens[v];
            if (!tv || !tv[0]) continue;
            size_t tl = strlen(tv);
            if (tl > best_len && pos + tl <= elen && memcmp(enc + pos, tv, tl) == 0) {
                best_len = tl;
                best_id = v;
            }
        }
        if (best_id >= 0) { out[n_out++] = (int64_t)best_id; pos += best_len; }
        else { pos++; }
    }
    return n_out;
}

double tn_moe_now_ms(void) { return moe_now_ms(); }

double tn_moe_last_timing_ms(int64_t handle, int64_t which) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return 0.0;
    MoeModel* m = &g_moe[handle];
    if (which == 0) return m->last_total_ms;
    if (which == 1) return m->last_ssd_ms;
    return m->last_gpu_ms;
}

int64_t tn_moe_free(int64_t handle) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return 0;
    MoeModel* m = &g_moe[handle];
    if (m->prefetch_grp) dispatch_group_wait(m->prefetch_grp, DISPATCH_TIME_FOREVER);
    if (m->dense_io_grp) dispatch_group_wait(m->dense_io_grp, DISPATCH_TIME_FOREVER);
    for (int L = 0; L < m->n_layers; L++) {
        MoeLayer* ly = &m->layers[L];
        if (ly->router_host) { free(ly->router_host); ly->router_host = NULL; }
        if (ly->ffn_norm_host) { free(ly->ffn_norm_host); ly->ffn_norm_host = NULL; }
        if (ly->exp_bias_host) { free(ly->exp_bias_host); ly->exp_bias_host = NULL; }
        ly->cache_buf = nil; ly->k_cache = nil; ly->v_cache = nil;
        ly->attn_norm.buf = nil; ly->wq.buf = nil; ly->wk.buf = nil; ly->wv.buf = nil; ly->wo.buf = nil;
        ly->bq.buf = nil; ly->bk.buf = nil; ly->bv.buf = nil;
        ly->q_norm.buf = nil; ly->k_norm.buf = nil; ly->ffn_norm.buf = nil; ly->router.buf = nil;
        ly->exp_bias.buf = nil; ly->ffn_gate.buf = nil; ly->ffn_up.buf = nil; ly->ffn_down.buf = nil;
        ly->gate_exps.buf = nil; ly->up_exps.buf = nil; ly->down_exps.buf = nil;
        ly->sh_gate.buf = nil; ly->sh_up.buf = nil; ly->sh_down.buf = nil;
    }
    if (m->vocab_tokens) {
        for (int i = 0; i < m->vocab_size; i++) free(m->vocab_tokens[i]);
        free(m->vocab_tokens);
        m->vocab_tokens = NULL;
    }
    for (int s = 0; s < 4; s++) m->dense_ping[s] = nil;
    m->tok_embd.buf = nil; m->out_norm.buf = nil; m->lm_head.buf = nil; m->rope_freqs.buf = nil;
    m->buf_emb_row = nil; m->buf_x = nil; m->buf_xn = nil; m->buf_q = nil; m->buf_k = nil; m->buf_v = nil;
    m->buf_ctx = nil; m->buf_rlogits = nil; m->buf_eidx = nil; m->buf_ewt = nil;
    m->buf_h = nil; m->buf_sh_g = nil; m->buf_sh_u = nil; m->buf_logits = nil; m->buf_argmax = nil;
    for (int s = 0; s < m->n_shards; s++) {
        m->mmap_mtl_bufs[s] = nil;
        if (m->mmap_ptrs[s] && m->mmap_ptrs[s] != MAP_FAILED) { munmap(m->mmap_ptrs[s], m->mmap_lens[s]); m->mmap_ptrs[s] = NULL; }
        if (m->fds[s] >= 0) { close(m->fds[s]); m->fds[s] = -1; }
    }
    m->active = 0;
    return 1;
}
