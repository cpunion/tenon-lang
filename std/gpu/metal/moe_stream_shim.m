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
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

#define MOE_MAX_MODELS  8
#define MOE_MAX_SHARDS  16
#define MOE_MAX_LAYERS  128
#define MOE_MAX_EXPERTS 256
#define MOE_MAX_TOPK    16
#define MOE_MAX_SEQ     16384
#define MOE_DEFAULT_SEQ 2048
#define MOE_PREFILL_MAX_B 64
#define PA_BLOCK_SIZE   16
#define PA_MAX_SEQS     32
#define PA_MAX_LOGICAL_BLKS (MOE_MAX_SEQ / PA_BLOCK_SIZE)
#define TREE_MAX_NODES  64

// GGML Quantization Types
#define TN_GGML_F32   0
#define TN_GGML_F16   1
#define TN_GGML_Q4_0  2
#define TN_GGML_Q4_1  3
#define TN_GGML_Q5_0  6
#define TN_GGML_Q5_1  7
#define TN_GGML_Q8_0   8
#define TN_GGML_Q2_K   10
#define TN_GGML_Q3_K   11
#define TN_GGML_Q4_K   12
#define TN_GGML_Q5_K   13
#define TN_GGML_Q6_K   14
#define TN_GGML_IQ4_XS 23

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
    // Optional DeepSeek-V2/V3/R1 Multi-Head Latent Attention (MLA) weights
    int is_mla;
    MoeWeightRef wq_a, q_a_norm, wq_b;
    MoeWeightRef wkv_a, kv_a_norm, wkv_b;
    // Unpacked FP16 MLA projection views for Matrix-Absorbed Compressed Latent KV-Cache!
    id<MTLBuffer> mla_wuk_f16; // [n_heads, kv_lora_rank, nope_d]
    id<MTLBuffer> mla_wuv_f16; // [n_heads, v_head_dim, kv_lora_rank]
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
    int n_expert_groups; // DeepSeek-V2/V3 group top-K routing (0 = disabled)
    int n_group_used;
    int kv_lora_rank;    // DeepSeek MLA latent dimensions
    int q_lora_rank;
    int v_head_dim;
    float rope_theta;
    float rms_eps;
    float yarn_factor;
    float yarn_orig_ctx;
    float yarn_log_m;
    float attn_mscale;
    int max_ctx;         // Dynamic KV context limit (default 2048, up to 16384!)
    int kv_quant_q8;     // 0 = FP16 KV (2.0 B/elem), 1 = Q8_0 KV (1.0625 B/elem -> -47% KV VRAM!)
    int mla_latent_kv;   // 1 = Compressed Latent MLA KV-Cache (`c_kv || k_pe`, -8x to -85x KV VRAM!)
    int n_kv_slots;      // Multi-slot Paged/Partitioned KV slots (default 1, env TENON_KV_SLOTS)
    int cur_kv_slot;     // Active KV slot index (0..n_kv_slots-1)
    int slot_ctx;        // Per-slot max context (`max_ctx / n_kv_slots`)
    int cache_slots_per_layer; // For MoE: LRU slots/layer; For Dense: pinned prefix layers

    char** vocab_tokens;
    // Standard GGUF BPE Merge Tokenizer (`tokenizer.ggml.merges` + O(1) hash tables)
    char** bpe_merges;
    int n_bpe_merges;
    int bpe_init;
    int bpe_use_sp;
    uint32_t tok_ht_mask;
    uint64_t* tok_ht_hash;
    int32_t*  tok_ht_id;
    uint32_t merge_ht_mask;
    uint64_t* merge_ht_key;
    int32_t*  merge_ht_rank;
    // Optional MTP (`nextn_layers`) self-speculative head + full MTP transformer layer
    int has_mtp;
    int has_mtp_layer;
    MoeWeightRef mtp_enorm, mtp_hnorm, mtp_eh_proj;
    MoeLayer* mtp_layer;
    id<MTLBuffer> buf_last_hidden; // [d_model] F32 snapshot before out_norm for MTP!
    id<MTLBuffer> buf_mtp_cat;     // [2 * d_model] F32 `[enorm(e) || hnorm(h)]`

    // PagedAttention 16-Token Block-Table KV Engine + Zero-Copy COW Prefix Fork + Continuous Batching
    int pa_num_blocks;
    int pa_free_blocks;
    int* pa_refcnt;
    int32_t pa_block_table[PA_MAX_SEQS][PA_MAX_LOGICAL_BLKS];
    int pa_seq_len[PA_MAX_SEQS];
    int64_t pa_cow_forks;
    int64_t pa_cow_copies;
    id<MTLBuffer> buf_pa_bt;        // [PA_MAX_SEQS * PA_MAX_LOGICAL_BLKS] int32
    id<MTLBuffer> buf_cb_pos;       // [MOE_PREFILL_MAX_B] uint32
    id<MTLBuffer> buf_cb_slot;      // [MOE_PREFILL_MAX_B] uint32
    id<MTLBuffer> buf_batch_logits; // [batch_logits_cap * vocab_size] F32
    int batch_logits_cap;
    // Tree-Attention Speculative Verification buffers
    id<MTLBuffer> buf_tree_depth;   // [TREE_MAX_NODES] uint32
    id<MTLBuffer> buf_tree_mask;    // [TREE_MAX_NODES] uint64

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

    // Lazy batched prefill working activations ([MOE_PREFILL_MAX_B, ...], allocated only when tn_moe_prefill is called!)
    int prefill_cap;
    id<MTLBuffer> pbuf_x;
    id<MTLBuffer> pbuf_xn;
    id<MTLBuffer> pbuf_q;
    id<MTLBuffer> pbuf_k;
    id<MTLBuffer> pbuf_v;
    id<MTLBuffer> pbuf_ctx;
    id<MTLBuffer> pbuf_g;
    id<MTLBuffer> pbuf_u;
    id<MTLBuffer> pbuf_mla_q;
    id<MTLBuffer> pbuf_mla_kv;
    id<MTLBuffer> pbuf_mla_qabs; // [n_heads * kv_lora_rank] F32
    id<MTLBuffer> pbuf_mla_olat; // [n_heads * kv_lora_rank] F32
    id<MTLBuffer> buf_mtp_x;     // [d_model] F32 scratch for MTP self-speculative head

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
static id<MTLComputePipelineState> g_pso_gemv_q5_0 = nil;
static id<MTLComputePipelineState> g_pso_gemv_q5_1 = nil;
static id<MTLComputePipelineState> g_pso_gemv_q8 = nil;
static id<MTLComputePipelineState> g_pso_gemv_q2k = nil;
static id<MTLComputePipelineState> g_pso_gemv_q3k = nil;
static id<MTLComputePipelineState> g_pso_gemv_iq4xs = nil;
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
static id<MTLComputePipelineState> g_pso_moe_down_q5_0 = nil;
static id<MTLComputePipelineState> g_pso_moe_down_q5_1 = nil;
static id<MTLComputePipelineState> g_pso_moe_down_q8 = nil;
static id<MTLComputePipelineState> g_pso_moe_down_q6k = nil;
static id<MTLComputePipelineState> g_pso_moe_slot_gu_q4 = nil;
static id<MTLComputePipelineState> g_pso_moe_slot_down_q4 = nil;
static id<MTLComputePipelineState> g_pso_moe_slot_down_q4_1 = nil;
static id<MTLComputePipelineState> g_pso_moe_slot_down_q5_0 = nil;
static id<MTLComputePipelineState> g_pso_moe_slot_down_q5_1 = nil;
static id<MTLComputePipelineState> g_pso_moe_slot_down_q8 = nil;
static id<MTLComputePipelineState> g_pso_moe_slot_down_q6k = nil;
static id<MTLComputePipelineState> g_pso_moe_slot_gu_q4k = nil;
static id<MTLComputePipelineState> g_pso_moe_slot_down_q4k = nil;
static id<MTLComputePipelineState> g_pso_moe_gu_q4k = nil;
static id<MTLComputePipelineState> g_pso_moe_down_q4k = nil;
static id<MTLComputePipelineState> g_pso_rope_kv_q8 = nil;
static id<MTLComputePipelineState> g_pso_flash_dec_q8 = nil;
static id<MTLComputePipelineState> g_pso_gemm_q4_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q4_sb = nil;
static id<MTLComputePipelineState> g_pso_gemm_q4_1_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q5_0_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q5_1_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q8_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q2k_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q3k_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_iq4xs_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q4k_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q5k_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q6k_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_f16_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_f32_b = nil;
static id<MTLComputePipelineState> g_pso_add_bias_b = nil;
static id<MTLComputePipelineState> g_pso_rope_kv_b = nil;
static id<MTLComputePipelineState> g_pso_rope_kv_q8_b = nil;
static id<MTLComputePipelineState> g_pso_flash_prefill = nil;
static id<MTLComputePipelineState> g_pso_flash_prefill_q8 = nil;
static id<MTLComputePipelineState> g_pso_mla_split_kv = nil;
static id<MTLComputePipelineState> g_pso_mla_rope_pe = nil;
static id<MTLComputePipelineState> g_pso_mla_store_lat = nil;
static id<MTLComputePipelineState> g_pso_mla_unpack_kvb = nil;
static id<MTLComputePipelineState> g_pso_mla_absorb_q = nil;
static id<MTLComputePipelineState> g_pso_mla_flash_lat = nil;
static id<MTLComputePipelineState> g_pso_mla_proj_v = nil;
static id<MTLComputePipelineState> g_pso_rope_kv_paged = nil;
static id<MTLComputePipelineState> g_pso_rope_kv_q8_paged = nil;
static id<MTLComputePipelineState> g_pso_flash_dec_paged = nil;
static id<MTLComputePipelineState> g_pso_flash_dec_q8_paged = nil;
static id<MTLComputePipelineState> g_pso_rope_kv_tree = nil;
static id<MTLComputePipelineState> g_pso_rope_kv_q8_tree = nil;
static id<MTLComputePipelineState> g_pso_flash_tree = nil;
static id<MTLComputePipelineState> g_pso_flash_tree_q8 = nil;
static id<MTLComputePipelineState> g_pso_vit_ln = nil;
static id<MTLComputePipelineState> g_pso_vit_gelu = nil;
static id<MTLComputePipelineState> g_pso_vit_add = nil;
static id<MTLComputePipelineState> g_pso_vit_attn = nil;
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
        case TN_GGML_Q5_0: return (n_elems / 32ull) * 22ull;
        case TN_GGML_Q5_1: return (n_elems / 32ull) * 24ull;
        case TN_GGML_Q8_0:   return (n_elems / 32ull) * 34ull;
        case TN_GGML_Q2_K:   return (n_elems / 256ull) * 84ull;
        case TN_GGML_Q3_K:   return (n_elems / 256ull) * 110ull;
        case TN_GGML_Q4_K:   return (n_elems / 256ull) * 144ull;
        case TN_GGML_Q5_K:   return (n_elems / 256ull) * 176ull;
        case TN_GGML_Q6_K:   return (n_elems / 256ull) * 210ull;
        case TN_GGML_IQ4_XS: return (n_elems / 256ull) * 136ull;
        default:             return n_elems * 2ull;
    }
}

static NSString* const kMoeMslSource = @R"(
#include <metal_stdlib>
using namespace metal;

// Load 32 floats into float4 registers (100% exact FP32, shared across 4 weight rows!)
static inline float load_x32_q4(const device float* x, thread float4 xv_lo[4], thread float4 xv_hi[4]) {
    const device float4* x4 = (const device float4*)x;
    float4 xs = float4(0.0f);
    #pragma unroll
    for (uint g = 0; g < 4u; g++) {
        float4 lo = x4[g];
        float4 hi = x4[g + 4u];
        xs += lo + hi;
        xv_lo[g] = lo;
        xv_hi[g] = hi;
    }
    return (xs.x + xs.y) + (xs.z + xs.w);
}

static inline void load_x32_f4(const device float* x, thread float4 xv[8]) {
    const device float4* x4 = (const device float4*)x;
    #pragma unroll
    for (uint i = 0; i < 8u; i++) xv[i] = x4[i];
}

// Ultra-fast 64-bit `packed_ushort4` zero-shift GGML Q4_0 dot (2 loads vs 16, exact FP32, 2-byte safe for GGUF mmap!)
static inline float dot_q4_0_fast(const device uchar* blk, thread const float4 xv_lo[4], thread const float4 xv_hi[4], float x_sum) {
    float d = float(*(const device half*)blk);
    const device packed_ushort4* q64 = (const device packed_ushort4*)(blk + 2u);
    ushort4 p0 = q64[0];
    ushort4 p1 = q64[1];
    uchar4 c0 = as_type<uchar4>(p0.xy);
    uchar4 c1 = as_type<uchar4>(p0.zw);
    uchar4 c2 = as_type<uchar4>(p1.xy);
    uchar4 c3 = as_type<uchar4>(p1.zw);
    float4 s0 = xv_lo[0] * float4(c0 & uchar4(0x0Fu));
    float4 s1 = xv_hi[0] * float4(c0 & uchar4(0xF0u));
    s0 = fma(xv_lo[1], float4(c1 & uchar4(0x0Fu)), s0);
    s1 = fma(xv_hi[1], float4(c1 & uchar4(0xF0u)), s1);
    s0 = fma(xv_lo[2], float4(c2 & uchar4(0x0Fu)), s0);
    s1 = fma(xv_hi[2], float4(c2 & uchar4(0xF0u)), s1);
    s0 = fma(xv_lo[3], float4(c3 & uchar4(0x0Fu)), s0);
    s1 = fma(xv_hi[3], float4(c3 & uchar4(0xF0u)), s1);
    float4 t = fma(s1, 0.0625f, s0);
    return d * (((t.x + t.y) + (t.z + t.w)) - 8.0f * x_sum);
}

// Ultra-fast 64-bit `packed_ushort4` zero-shift GGML Q4_1 dot (20 bytes / 32 elems: half d + half m + uchar qs[16])
static inline float dot_q4_1_fast(const device uchar* blk, thread const float4 xv_lo[4], thread const float4 xv_hi[4], float x_sum) {
    float d = float(*(const device half*)(blk));
    float m = float(*(const device half*)(blk + 2u));
    const device packed_ushort4* q64 = (const device packed_ushort4*)(blk + 4u);
    ushort4 p0 = q64[0];
    ushort4 p1 = q64[1];
    uchar4 c0 = as_type<uchar4>(p0.xy);
    uchar4 c1 = as_type<uchar4>(p0.zw);
    uchar4 c2 = as_type<uchar4>(p1.xy);
    uchar4 c3 = as_type<uchar4>(p1.zw);
    float4 s0 = xv_lo[0] * float4(c0 & uchar4(0x0Fu));
    float4 s1 = xv_hi[0] * float4(c0 & uchar4(0xF0u));
    s0 = fma(xv_lo[1], float4(c1 & uchar4(0x0Fu)), s0);
    s1 = fma(xv_hi[1], float4(c1 & uchar4(0xF0u)), s1);
    s0 = fma(xv_lo[2], float4(c2 & uchar4(0x0Fu)), s0);
    s1 = fma(xv_hi[2], float4(c2 & uchar4(0xF0u)), s1);
    s0 = fma(xv_lo[3], float4(c3 & uchar4(0x0Fu)), s0);
    s1 = fma(xv_hi[3], float4(c3 & uchar4(0xF0u)), s1);
    float4 t = fma(s1, 0.0625f, s0);
    return fma(d, (t.x + t.y) + (t.z + t.w), m * x_sum);
}

// Vectorized 64-bit `packed_ushort4 -> char4` GGML Q8_0 dot (34 bytes / 32 elems: half d + 32 int8)
static inline float dot_q8_0_fast(const device uchar* blk, thread const float4 xv[8]) {
    float d = float(*(const device half*)blk);
    const device packed_ushort4* q64 = (const device packed_ushort4*)(blk + 2u);
    ushort4 p0 = q64[0], p1 = q64[1], p2 = q64[2], p3 = q64[3];
    float4 s0 = xv[0] * float4(as_type<char4>(p0.xy));
    float4 s1 = xv[1] * float4(as_type<char4>(p0.zw));
    s0 = fma(xv[2], float4(as_type<char4>(p1.xy)), s0);
    s1 = fma(xv[3], float4(as_type<char4>(p1.zw)), s1);
    s0 = fma(xv[4], float4(as_type<char4>(p2.xy)), s0);
    s1 = fma(xv[5], float4(as_type<char4>(p2.zw)), s1);
    s0 = fma(xv[6], float4(as_type<char4>(p3.xy)), s0);
    s1 = fma(xv[7], float4(as_type<char4>(p3.zw)), s1);
    float4 t = s0 + s1;
    return d * ((t.x + t.y) + (t.z + t.w));
}

// Exact GGML Q5_0 dot (22 bytes / 32 elems: half d + uchar qh[4] + uchar qs[16])
static inline float dot_q5_0_fast(const device uchar* blk, const device float* x) {
    float d = float(*(const device half*)blk);
    uint qh = uint(blk[2u]) | (uint(blk[3u]) << 8) | (uint(blk[4u]) << 16) | (uint(blk[5u]) << 24);
    const device uchar* qs = blk + 6u;
    float s = 0.0f;
    #pragma unroll
    for (uint j = 0; j < 16u; j++) {
        uchar bv = qs[j];
        int qlo = int((bv & 0xFu) | (((qh >> j) & 1u) << 4)) - 16;
        int qhi = int((bv >> 4)   | (((qh >> (j + 16u)) & 1u) << 4)) - 16;
        s = fma(float(qlo), x[j], fma(float(qhi), x[j + 16u], s));
    }
    return d * s;
}

// Exact GGML Q5_1 dot (24 bytes / 32 elems: half d + half m + uchar qh[4] + uchar qs[16])
static inline float dot_q5_1_fast(const device uchar* blk, const device float* x) {
    float d = float(*(const device half*)blk);
    float m = float(*(const device half*)(blk + 2u));
    uint qh = uint(blk[4u]) | (uint(blk[5u]) << 8) | (uint(blk[6u]) << 16) | (uint(blk[7u]) << 24);
    const device uchar* qs = blk + 8u;
    float s = 0.0f, sx = 0.0f;
    #pragma unroll
    for (uint j = 0; j < 16u; j++) {
        uchar bv = qs[j];
        uint qlo = (bv & 0xFu) | (((qh >> j) & 1u) << 4);
        uint qhi = (bv >> 4)   | (((qh >> (j + 16u)) & 1u) << 4);
        float x0 = x[j], x1 = x[j + 16u];
        s = fma(float(qlo), x0, fma(float(qhi), x1, s));
        sx += x0 + x1;
    }
    return fma(d, s, m * sx);
}

// Vectorized 4-lane/superblock GGML Q4_K quarter-block (144 bytes / 256 elems, float4 x + uchar4 q!)
static inline float dot_q4_k_quarter(const device uchar* blk, const device float* x, uint qtr) {
    float d    = float(*(const device half*)(blk));
    float dmin = float(*(const device half*)(blk + 2u));
    const device uchar* sc = blk + 4u;
    const device uchar* qs = blk + 16u;
    uint l0 = qtr * 8u;
    float sum = 0.0f;
    uint is = 0u;
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
        const device uchar* q = qs + (j >> 1) + l0;
        const device float4* xl = (const device float4*)(x + j + l0);
        const device float4* xh = (const device float4*)(x + j + 32u + l0);
        float p0 = 0.0f, p1 = 0.0f, sx0 = 0.0f, sx1 = 0.0f;
        #pragma unroll
        for (uint g = 0; g < 2u; g++) {
            float4 vlo = xl[g], vhi = xh[g];
            sx0 += (vlo.x + vlo.y) + (vlo.z + vlo.w);
            sx1 += (vhi.x + vhi.y) + (vhi.z + vhi.w);
            uchar b0 = q[g * 4u + 0u], b1 = q[g * 4u + 1u], b2 = q[g * 4u + 2u], b3 = q[g * 4u + 3u];
            p0 = fma(float(b0 & 0xFu), vlo.x, fma(float(b1 & 0xFu), vlo.y, fma(float(b2 & 0xFu), vlo.z, fma(float(b3 & 0xFu), vlo.w, p0))));
            p1 = fma(float(b0 >> 4),   vhi.x, fma(float(b1 >> 4),   vhi.y, fma(float(b2 >> 4),   vhi.z, fma(float(b3 >> 4),   vhi.w, p1))));
        }
        sum += fma(d1, p0, -dm1 * sx0) + fma(d2, p1, -dm2 * sx1);
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

// Exact GGML Q2_K quarter-block (84 bytes / 256 elems: scales[16], qs[64], half d, half dmin)
static inline float dot_q2_k_quarter(const device uchar* blk, const device float* x, uint qtr) {
    const device uchar* sc = blk;
    const device uchar* qs = blk + 16u;
    float d    = float(*(const device half*)(blk + 80u));
    float dmin = float(*(const device half*)(blk + 82u));
    uint l0 = qtr * 4u; // 4 of 16 elems per sub-block -> 4 lanes cover all 16!
    float sum = 0.0f;
    uint is = 0u;
    #pragma unroll
    for (uint n = 0; n < 256u; n += 128u) {
        const device uchar* q = qs + (n >> 2);
        #pragma unroll
        for (uint shift = 0; shift < 8u; shift += 2u) {
            uint base = n + (shift << 4);
            float d0 = d * float(sc[is] & 0xFu),      m0 = dmin * float(sc[is] >> 4);
            float d1 = d * float(sc[is + 1u] & 0xFu), m1 = dmin * float(sc[is + 1u] >> 4);
            float p0 = 0.0f, p1 = 0.0f, sx0 = 0.0f, sx1 = 0.0f;
            #pragma unroll
            for (uint l = l0; l < l0 + 4u; l++) {
                float x0 = x[base + l], x1 = x[base + 16u + l];
                p0 = fma(float((q[l] >> shift) & 3u), x0, p0); sx0 += x0;
                p1 = fma(float((q[l + 16u] >> shift) & 3u), x1, p1); sx1 += x1;
            }
            sum += fma(d0, p0, -m0 * sx0) + fma(d1, p1, -m1 * sx1);
            is += 2u;
        }
    }
    return sum;
}

// Exact GGML Q3_K quarter-block (110 bytes / 256 elems: hmask[32], qs[64], scales[12], half d)
static inline int q3k_scale(const device uchar* sc, uint idx) {
    uint lo4 = (idx < 8u) ? (sc[idx] & 0xFu) : (sc[idx - 8u] >> 4);
    uint hi2 = (sc[8u + (idx & 3u)] >> (2u * (idx >> 2))) & 3u;
    return int(lo4 | (hi2 << 4)) - 32;
}

static inline float dot_q3_k_quarter(const device uchar* blk, const device float* x, uint qtr) {
    const device uchar* hm = blk;
    const device uchar* qs = blk + 32u;
    const device uchar* sc = blk + 96u;
    float d = float(*(const device half*)(blk + 108u));
    uint l0 = qtr * 4u;
    float sum = 0.0f;
    uint is = 0u;
    uchar mbit = 1u;
    #pragma unroll
    for (uint n = 0; n < 256u; n += 128u) {
        const device uchar* q = qs + (n >> 2);
        #pragma unroll
        for (uint shift = 0; shift < 8u; shift += 2u) {
            uint base = n + (shift << 4);
            float s0 = float(q3k_scale(sc, is)), s1 = float(q3k_scale(sc, is + 1u));
            float p0 = 0.0f, p1 = 0.0f;
            #pragma unroll
            for (uint l = l0; l < l0 + 4u; l++) {
                int q0 = int((q[l] >> shift) & 3u) - ((hm[l] & mbit) ? 0 : 4);
                int q1 = int((q[l + 16u] >> shift) & 3u) - ((hm[l + 16u] & mbit) ? 0 : 4);
                p0 = fma(float(q0), x[base + l], p0);
                p1 = fma(float(q1), x[base + 16u + l], p1);
            }
            sum += s0 * p0 + s1 * p1;
            is += 2u; mbit <<= 1;
        }
    }
    return d * sum;
}

// Exact GGML IQ4_XS quarter-block (136 bytes / 256 elems: half d, ushort scales_h, uchar scales_l[4], uchar qs[128])
constant int8_t kIq4Nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

static inline float dot_iq4_xs_quarter(const device uchar* blk, const device float* x, uint qtr) {
    float d = float(*(const device half*)blk);
    uint scales_h = uint(*(const device ushort*)(blk + 2u));
    const device uchar* scales_l = blk + 4u;
    const device uchar* qs = blk + 8u;
    uint ib0 = qtr * 2u; // 2 of 8 sub-blocks (64 weights) per quarter lane!
    float sum = 0.0f;
    #pragma unroll
    for (uint ib = ib0; ib < ib0 + 2u; ib++) {
        uint ls = ((scales_l[ib >> 1] >> (4u * (ib & 1u))) & 0xFu) | (((scales_h >> (2u * ib)) & 3u) << 4);
        float dl = d * float(int(ls) - 32);
        const device uchar* q = qs + ib * 16u;
        const device float* xp = x + ib * 32u;
        float acc = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 16u; j++) {
            uchar bv = q[j];
            acc = fma(float(kIq4Nl[bv & 0xFu]), xp[j], acc);
            acc = fma(float(kIq4Nl[bv >> 4]),   xp[j + 16u], acc);
        }
        sum += dl * acc;
    }
    return sum;
}

// Universal single-element dequantizer (used by moe_embed_lookup & one-time MLA wkv_b unpack!)
static inline float dequant_any_elem(const device uchar* w, ulong row, uint col, uint in_dim, uint gtype) {
    if (gtype == 0u) return ((const device float*)w)[row * in_dim + col];
    if (gtype == 1u) return float(((const device half*)w)[row * in_dim + col]);
    if (gtype == 2u) {
        ulong rb = (ulong)(in_dim >> 5) * 18ul;
        const device uchar* bp = w + row * rb + (ulong)(col >> 5) * 18ul;
        uint sub = col & 31u; uchar bv = bp[2u + (sub & 15u)];
        return float(*(const device half*)bp) * float((sub < 16u ? int(bv & 0xFu) : int(bv >> 4)) - 8);
    }
    if (gtype == 3u) {
        ulong rb = (ulong)(in_dim >> 5) * 20ul;
        const device uchar* bp = w + row * rb + (ulong)(col >> 5) * 20ul;
        uint sub = col & 31u; uchar bv = bp[4u + (sub & 15u)];
        return fma(float(*(const device half*)bp), float(sub < 16u ? (bv & 0xFu) : (bv >> 4)), float(*(const device half*)(bp + 2u)));
    }
    if (gtype == 6u) { // Q5_0
        ulong rb = (ulong)(in_dim >> 5) * 22ul;
        const device uchar* bp = w + row * rb + (ulong)(col >> 5) * 22ul;
        uint sub = col & 31u;
        uint qh = uint(bp[2u]) | (uint(bp[3u]) << 8) | (uint(bp[4u]) << 16) | (uint(bp[5u]) << 24);
        uchar bv = bp[6u + (sub & 15u)];
        uint lo4 = (sub < 16u) ? (bv & 0xFu) : (bv >> 4);
        int qv = int(lo4 | (((qh >> sub) & 1u) << 4)) - 16;
        return float(*(const device half*)bp) * float(qv);
    }
    if (gtype == 7u) { // Q5_1
        ulong rb = (ulong)(in_dim >> 5) * 24ul;
        const device uchar* bp = w + row * rb + (ulong)(col >> 5) * 24ul;
        uint sub = col & 31u;
        uint qh = uint(bp[4u]) | (uint(bp[5u]) << 8) | (uint(bp[6u]) << 16) | (uint(bp[7u]) << 24);
        uchar bv = bp[8u + (sub & 15u)];
        uint lo4 = (sub < 16u) ? (bv & 0xFu) : (bv >> 4);
        uint qv = lo4 | (((qh >> sub) & 1u) << 4);
        return fma(float(*(const device half*)bp), float(qv), float(*(const device half*)(bp + 2u)));
    }
    if (gtype == 8u) {
        ulong rb = (ulong)(in_dim >> 5) * 34ul;
        const device uchar* bp = w + row * rb + (ulong)(col >> 5) * 34ul;
        return float(*(const device half*)bp) * float(((const device char*)(bp + 2u))[col & 31u]);
    }
    if (gtype == 10u) { // Q2_K
        ulong rb = (ulong)(in_dim >> 8) * 84ul;
        const device uchar* bp = w + row * rb + (ulong)(col >> 8) * 84ul;
        uint rem = col & 255u, n = (rem >= 128u) ? 128u : 0u, r = rem - n;
        uint chunk = r >> 5, sub = r & 31u, is = (n >> 4) + chunk * 2u + (sub >> 4);
        uint shift = chunk * 2u;
        float d = float(*(const device half*)(bp + 80u)), dm = float(*(const device half*)(bp + 82u));
        uchar qv = (bp[16u + (n >> 2) + sub] >> shift) & 3u;
        return fma(d * float(bp[is] & 0xFu), float(qv), -dm * float(bp[is] >> 4));
    }
    if (gtype == 11u) { // Q3_K
        ulong rb = (ulong)(in_dim >> 8) * 110ul;
        const device uchar* bp = w + row * rb + (ulong)(col >> 8) * 110ul;
        uint rem = col & 255u, n = (rem >= 128u) ? 128u : 0u, r = rem - n;
        uint chunk = r >> 5, sub = r & 31u, is = (n >> 4) + chunk * 2u + (sub >> 4);
        uint shift = chunk * 2u; uchar mbit = uchar(1u << ((n ? 4u : 0u) + chunk));
        int qv = int((bp[32u + (n >> 2) + sub] >> shift) & 3u) - ((bp[sub] & mbit) ? 0 : 4);
        return float(*(const device half*)(bp + 108u)) * float(q3k_scale(bp + 96u, is)) * float(qv);
    }
    if (gtype == 12u) { // Q4_K
        ulong rb = (ulong)(in_dim >> 8) * 144ul;
        const device uchar* bp = w + row * rb + (ulong)(col >> 8) * 144ul;
        const device uchar* sc = bp + 4u;
        uint rem = col & 255u, j = rem & ~63u, r = rem & 63u, hi = r >> 5, l = r & 31u, is = (j >> 5) + hi;
        uchar scv, mv;
        if (is < 4u) { scv = sc[is] & 63u; mv = sc[is + 4u] & 63u; }
        else { scv = (sc[is + 4u] & 0xFu) | ((sc[is - 4u] >> 6) << 4); mv = (sc[is + 4u] >> 4) | ((sc[is] >> 6) << 4); }
        uchar bv = bp[16u + (j >> 1) + l];
        uint qv = hi ? (bv >> 4) : (bv & 0xFu);
        return fma(float(*(const device half*)bp) * float(scv), float(qv), -float(*(const device half*)(bp + 2u)) * float(mv));
    }
    if (gtype == 13u) { // Q5_K
        ulong rb = (ulong)(in_dim >> 8) * 176ul;
        const device uchar* bp = w + row * rb + (ulong)(col >> 8) * 176ul;
        const device uchar* sc = bp + 4u;
        uint rem = col & 255u, j = rem & ~63u, r = rem & 63u, hi = r >> 5, l = r & 31u, is = (j >> 5) + hi;
        uchar scv, mv;
        if (is < 4u) { scv = sc[is] & 63u; mv = sc[is + 4u] & 63u; }
        else { scv = (sc[is + 4u] & 0xFu) | ((sc[is - 4u] >> 6) << 4); mv = (sc[is + 4u] >> 4) | ((sc[is] >> 6) << 4); }
        uchar bv = bp[48u + (j >> 1) + l], hv = bp[16u + l];
        uint qv = (hi ? (bv >> 4) : (bv & 0xFu)) + ((hv & (1u << is)) ? 16u : 0u);
        return fma(float(*(const device half*)bp) * float(scv), float(qv), -float(*(const device half*)(bp + 2u)) * float(mv));
    }
    if (gtype == 14u) { // Q6_K
        ulong rb = (ulong)(in_dim >> 8) * 210ul;
        const device uchar* bp = w + row * rb + (ulong)(col >> 8) * 210ul;
        uint rem = col & 255u, n = (rem >= 128u) ? 128u : 0u, r = rem - n, qpart = r >> 5, l = r & 31u;
        const device uchar* qlp = bp + (n >> 1); const device uchar* qhp = bp + 128u + (n >> 2);
        const device char* scp = (const device char*)(bp + 192u) + (n >> 4);
        uchar lo = (qpart == 0u) ? (qlp[l] & 0xFu) : (qpart == 1u) ? (qlp[l + 32u] & 0xFu) : (qpart == 2u) ? (qlp[l] >> 4) : (qlp[l + 32u] >> 4);
        uchar hi = (qhp[l] >> (qpart * 2u)) & 3u;
        return float(*(const device half*)(bp + 208u)) * float(scp[(l >> 4) + qpart * 2u]) * float(int(lo | (hi << 4)) - 32);
    }
    if (gtype == 23u) { // IQ4_XS
        ulong rb = (ulong)(in_dim >> 8) * 136ul;
        const device uchar* bp = w + row * rb + (ulong)(col >> 8) * 136ul;
        uint rem = col & 255u, ib = rem >> 5, sub = rem & 31u;
        uint scales_h = uint(*(const device ushort*)(bp + 2u));
        uint ls = ((bp[4u + (ib >> 1)] >> (4u * (ib & 1u))) & 0xFu) | (((scales_h >> (2u * ib)) & 3u) << 4);
        uchar bv = bp[8u + ib * 16u + (sub & 15u)];
        uint qi = (sub < 16u) ? (bv & 0xFu) : (bv >> 4);
        return float(*(const device half*)bp) * float(int(ls) - 32) * float(kIq4Nl[qi]);
    }
    return 0.0f;
}

struct MoeEmbedArgs { uint tok_id; uint d_model; uint ggml_type; };

kernel void moe_embed_lookup(
    const device uchar* emb [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant MoeEmbedArgs& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.d_model) return;
    out[gid] = dequant_any_elem(emb, (ulong)p.tok_id, gid, p.d_model, p.ggml_type);
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

// 4-row-per-SG 64-bit vectorized Q4_0 GEMV (NSG=2 -> 8 rows/TG, 64 thr/TG for peak M4 Max occupancy!)
kernel void moe_gemv_q4(
    const device uchar* w [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 2u + sg) * 4u;
    if (row0 >= p.out_dim) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 18ul;
    const device uchar* wp0 = w + (ulong)row0 * rb;
    const device uchar* wp1 = (row0 + 1u < p.out_dim) ? (wp0 + rb) : wp0;
    const device uchar* wp2 = (row0 + 2u < p.out_dim) ? (wp0 + rb * 2ul) : wp0;
    const device uchar* wp3 = (row0 + 3u < p.out_dim) ? (wp0 + rb * 3ul) : wp0;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        thread float4 xv_lo[4], xv_hi[4];
        float xs = load_x32_q4(x + (b << 5), xv_lo, xv_hi);
        ulong boff = (ulong)b * 18ul;
        a0 += dot_q4_0_fast(wp0 + boff, xv_lo, xv_hi, xs);
        a1 += dot_q4_0_fast(wp1 + boff, xv_lo, xv_hi, xs);
        a2 += dot_q4_0_fast(wp2 + boff, xv_lo, xv_hi, xs);
        a3 += dot_q4_0_fast(wp3 + boff, xv_lo, xv_hi, xs);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row0 + 1u < p.out_dim) y[row0 + 1u] = p.add_res ? (y[row0 + 1u] + a1) : a1;
        if (row0 + 2u < p.out_dim) y[row0 + 2u] = p.add_res ? (y[row0 + 2u] + a2) : a2;
        if (row0 + 3u < p.out_dim) y[row0 + 3u] = p.add_res ? (y[row0 + 3u] + a3) : a3;
    }
}

// 4-row-per-SG 64-bit vectorized Q4_1 GEMV (NSG=2 -> 8 rows/TG, 64 thr/TG)
kernel void moe_gemv_q4_1(
    const device uchar* w [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 2u + sg) * 4u;
    if (row0 >= p.out_dim) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 20ul;
    const device uchar* wp0 = w + (ulong)row0 * rb;
    const device uchar* wp1 = (row0 + 1u < p.out_dim) ? (wp0 + rb) : wp0;
    const device uchar* wp2 = (row0 + 2u < p.out_dim) ? (wp0 + rb * 2ul) : wp0;
    const device uchar* wp3 = (row0 + 3u < p.out_dim) ? (wp0 + rb * 3ul) : wp0;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        thread float4 xv_lo[4], xv_hi[4];
        float sx = load_x32_q4(x + (b << 5), xv_lo, xv_hi);
        ulong boff = (ulong)b * 20ul;
        a0 += dot_q4_1_fast(wp0 + boff, xv_lo, xv_hi, sx);
        a1 += dot_q4_1_fast(wp1 + boff, xv_lo, xv_hi, sx);
        a2 += dot_q4_1_fast(wp2 + boff, xv_lo, xv_hi, sx);
        a3 += dot_q4_1_fast(wp3 + boff, xv_lo, xv_hi, sx);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row0 + 1u < p.out_dim) y[row0 + 1u] = p.add_res ? (y[row0 + 1u] + a1) : a1;
        if (row0 + 2u < p.out_dim) y[row0 + 2u] = p.add_res ? (y[row0 + 2u] + a2) : a2;
        if (row0 + 3u < p.out_dim) y[row0 + 3u] = p.add_res ? (y[row0 + 3u] + a3) : a3;
    }
}

// 2-row-per-SG Q5_0 GEMV (NSG=2 -> 4 rows/TG, 64 thr/TG)
kernel void moe_gemv_q5_0(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 2u + sg) * 2u, row1 = row0 + 1u;
    if (row0 >= p.out_dim) return;
    uint nb = p.in_dim >> 5; ulong rb = (ulong)nb * 22ul;
    const device uchar* wp0 = w + (ulong)row0 * rb;
    const device uchar* wp1 = (row1 < p.out_dim) ? (wp0 + rb) : wp0;
    float a0 = 0.0f, a1 = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        const device float* xp = x + (b << 5); ulong boff = (ulong)b * 22ul;
        a0 += dot_q5_0_fast(wp0 + boff, xp);
        if (row1 < p.out_dim) a1 += dot_q5_0_fast(wp1 + boff, xp);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row1 < p.out_dim) y[row1] = p.add_res ? (y[row1] + a1) : a1;
    }
}

// 2-row-per-SG Q5_1 GEMV (NSG=2 -> 4 rows/TG, 64 thr/TG)
kernel void moe_gemv_q5_1(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 2u + sg) * 2u, row1 = row0 + 1u;
    if (row0 >= p.out_dim) return;
    uint nb = p.in_dim >> 5; ulong rb = (ulong)nb * 24ul;
    const device uchar* wp0 = w + (ulong)row0 * rb;
    const device uchar* wp1 = (row1 < p.out_dim) ? (wp0 + rb) : wp0;
    float a0 = 0.0f, a1 = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        const device float* xp = x + (b << 5); ulong boff = (ulong)b * 24ul;
        a0 += dot_q5_1_fast(wp0 + boff, xp);
        if (row1 < p.out_dim) a1 += dot_q5_1_fast(wp1 + boff, xp);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row1 < p.out_dim) y[row1] = p.add_res ? (y[row1] + a1) : a1;
    }
}

// 2-row-per-SG 64-bit vectorized Q8_0 GEMV (NSG=2 -> 4 rows/TG, 64 thr/TG)
kernel void moe_gemv_q8(
    const device uchar* w [[buffer(0)]],
    const device float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 2u + sg) * 2u;
    uint row1 = row0 + 1u;
    if (row0 >= p.out_dim) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 34ul;
    const device uchar* wp0 = w + (ulong)row0 * rb;
    const device uchar* wp1 = (row1 < p.out_dim) ? (wp0 + rb) : wp0;
    float a0 = 0.0f, a1 = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        thread float4 xv[8];
        load_x32_f4(x + (b << 5), xv);
        ulong boff = (ulong)b * 34ul;
        a0 += dot_q8_0_fast(wp0 + boff, xv);
        if (row1 < p.out_dim) a1 += dot_q8_0_fast(wp1 + boff, xv);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row1 < p.out_dim) y[row1] = p.add_res ? (y[row1] + a1) : a1;
    }
}

// Full 32-lane SIMD Q4_K GEMV (4 lanes/superblock, 2 rows/SG = 8 rows/TG -> 4x higher SIMD occupancy!)
kernel void moe_gemv_q4k(
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
    ulong rb = (ulong)nb * 144ul;
    const device uchar* wp0 = w + (ulong)row0 * rb;
    const device uchar* wp1 = (row1 < p.out_dim) ? (wp0 + rb) : wp0;
    uint qtr = lane & 3u;
    float a0 = 0.0f, a1 = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float* xp = x + (b << 8);
        ulong boff = (ulong)b * 144ul;
        a0 += dot_q4_k_quarter(wp0 + boff, xp, qtr);
        if (row1 < p.out_dim) a1 += dot_q4_k_quarter(wp1 + boff, xp, qtr);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row1 < p.out_dim) y[row1] = p.add_res ? (y[row1] + a1) : a1;
    }
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

// Fused Dense SwiGLU Gate || Up Q4_0 (NSG=2, 2 rows/SG = 4 rows/TG, shares compact half4 xn across 4 weight rows!)
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
    uint row0 = (tgid * 2u + sg) * 2u;
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
        thread float4 xv_lo[4], xv_hi[4];
        float xs = load_x32_q4(xn + (b << 5), xv_lo, xv_hi);
        ulong boff = (ulong)b * 18ul;
        ag0 += dot_q4_0_fast(g0 + boff, xv_lo, xv_hi, xs);
        au0 += dot_q4_0_fast(u0 + boff, xv_lo, xv_hi, xs);
        ag1 += dot_q4_0_fast(g1 + boff, xv_lo, xv_hi, xs);
        au1 += dot_q4_0_fast(u1 + boff, xv_lo, xv_hi, xs);
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

struct MoeAttnArgs { uint n_heads; uint n_kv; uint d_head; uint kv_len; uint out_stride; float scale; };

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
    float scale = p.scale;

    threadgroup float s_q[256];
    threadgroup float s_m[4];
    threadgroup float s_l[4];
    threadgroup float s_o[4][256];

    for (uint d = lid; d < p.d_head; d += 128u) s_q[d] = q[h * p.d_head + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_run = -1e30f, l_run = 0.0f;
    float o_acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
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
    for (uint d = lid; d < p.out_stride; d += 128u) {
        float ov = 0.0f;
        for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fma(s_o[s][d], exp(s_m[s] - gm), ov);
        ctx[h * p.out_stride + d] = ov * inv;
    }
}

struct MoeRouterArgs { uint n_experts; uint top_k; uint norm_weights; uint gating_func; uint has_bias; float w_scale; uint n_groups; uint n_group_used; };

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
    // DeepSeek-V2/V3/R1 Group Top-K routing mask!
    if (p.n_groups > 1u && p.n_group_used > 0u && p.n_group_used < p.n_groups) {
        uint gsz = p.n_experts / p.n_groups;
        float gsc[32];
        bool gkeep[32];
        for (uint g = 0; g < p.n_groups && g < 32u; g++) {
            float m1 = -1e30f, m2 = -1e30f;
            for (uint i = 0; i < gsz; i++) {
                float sv = scores[g * gsz + i];
                if (sv > m1) { m2 = m1; m1 = sv; } else if (sv > m2) { m2 = sv; }
            }
            gsc[g] = (p.gating_func == 2u) ? (m1 + m2) : m1;
            gkeep[g] = false;
        }
        for (uint u = 0; u < p.n_group_used; u++) {
            uint bg = 0u; float bs = -1e30f;
            for (uint g = 0; g < p.n_groups && g < 32u; g++) {
                if (!gkeep[g] && gsc[g] > bs) { bs = gsc[g]; bg = g; }
            }
            gkeep[bg] = true;
        }
        for (uint g = 0; g < p.n_groups && g < 32u; g++) {
            if (!gkeep[g]) for (uint i = 0; i < gsz; i++) scores[g * gsz + i] = -1e30f;
        }
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
    uint row0 = (tgid.x * 2u + sg) * 2u;
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
        thread float4 xv_lo[4], xv_hi[4];
        float xs = load_x32_q4(xn + (b << 5), xv_lo, xv_hi);
        ulong boff = (ulong)b * 18ul;
        ag0 += dot_q4_0_fast(wg0 + boff, xv_lo, xv_hi, xs);
        au0 += dot_q4_0_fast(wu0 + boff, xv_lo, xv_hi, xs);
        ag1 += dot_q4_0_fast(wg1 + boff, xv_lo, xv_hi, xs);
        au1 += dot_q4_0_fast(wu1 + boff, xv_lo, xv_hi, xs);
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
    uint row0 = (tgid * 2u + sg) * 4u;
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
            thread float4 hv_lo[4], hv_hi[4];
            float hs = load_x32_q4(hp + (b << 5), hv_lo, hv_hi);
            ulong boff = (ulong)b * 18ul;
            a0 += dot_q4_0_fast(wd0 + boff, hv_lo, hv_hi, hs);
            a1 += dot_q4_0_fast(wd1 + boff, hv_lo, hv_hi, hs);
            a2 += dot_q4_0_fast(wd2 + boff, hv_lo, hv_hi, hs);
            a3 += dot_q4_0_fast(wd3 + boff, hv_lo, hv_hi, hs);
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
    uint row0 = (tgid * 2u + sg) * 4u;
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
            thread float4 hv_lo[4], hv_hi[4];
            float sh = load_x32_q4(hp + (b << 5), hv_lo, hv_hi);
            ulong boff = (ulong)b * 20ul;
            a0 += dot_q4_1_fast(wd0 + boff, hv_lo, hv_hi, sh);
            a1 += dot_q4_1_fast(wd1 + boff, hv_lo, hv_hi, sh);
            a2 += dot_q4_1_fast(wd2 + boff, hv_lo, hv_hi, sh);
            a3 += dot_q4_1_fast(wd3 + boff, hv_lo, hv_hi, sh);
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
    uint row0 = (tgid.x * 2u + sg) * 2u;
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
        thread float4 xv_lo[4], xv_hi[4];
        float xs = load_x32_q4(xn + (b << 5), xv_lo, xv_hi);
        ulong boff = (ulong)b * 18ul;
        ag0 += dot_q4_0_fast(wg0 + boff, xv_lo, xv_hi, xs);
        au0 += dot_q4_0_fast(wu0 + boff, xv_lo, xv_hi, xs);
        ag1 += dot_q4_0_fast(wg1 + boff, xv_lo, xv_hi, xs);
        au1 += dot_q4_0_fast(wu1 + boff, xv_lo, xv_hi, xs);
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
    uint row0 = (tgid * 2u + sg) * 4u;
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
            thread float4 hv_lo[4], hv_hi[4];
            float hs = load_x32_q4(hp + (b << 5), hv_lo, hv_hi);
            ulong boff = (ulong)b * 18ul;
            a0 += dot_q4_0_fast(wd0 + boff, hv_lo, hv_hi, hs);
            a1 += dot_q4_0_fast(wd1 + boff, hv_lo, hv_hi, hs);
            a2 += dot_q4_0_fast(wd2 + boff, hv_lo, hv_hi, hs);
            a3 += dot_q4_0_fast(wd3 + boff, hv_lo, hv_hi, hs);
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
    uint row0 = (tgid * 2u + sg) * 4u;
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
            thread float4 hv_lo[4], hv_hi[4];
            float sh = load_x32_q4(hp + (b << 5), hv_lo, hv_hi);
            ulong boff = (ulong)b * 20ul;
            a0 += dot_q4_1_fast(wd0 + boff, hv_lo, hv_hi, sh);
            a1 += dot_q4_1_fast(wd1 + boff, hv_lo, hv_hi, sh);
            a2 += dot_q4_1_fast(wd2 + boff, hv_lo, hv_hi, sh);
            a3 += dot_q4_1_fast(wd3 + boff, hv_lo, hv_hi, sh);
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

// =========================================================================
// Q4_K MoE Resident & Streaming-Slot Kernels (Qwen3-MoE / DeepSeek-R1 Q4_K_M!)
// =========================================================================
kernel void moe_topk_gate_up_q4k(
    const device uchar* gate_exps [[buffer(0)]], const device uchar* up_exps [[buffer(1)]],
    const device float* xn [[buffer(2)]], const device uint* eidx [[buffer(3)]],
    device float* h_out [[buffer(4)]], constant MoeExpArgs& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 4u + sg; uint k = tgid.y;
    if (row >= p.d_ff || k >= p.n_active) return;
    uint eid = eidx[k]; uint nb = p.d_model >> 8; ulong rb = (ulong)nb * 144ul;
    const device uchar* g0 = gate_exps + (ulong)eid * p.gate_exp_bytes + (ulong)row * rb;
    const device uchar* u0 = up_exps   + (ulong)eid * p.up_exp_bytes   + (ulong)row * rb;
    uint qtr = lane & 3u; float ag = 0.0f, au = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float* xp = xn + (b << 8); ulong boff = (ulong)b * 144ul;
        ag += dot_q4_k_quarter(g0 + boff, xp, qtr); au += dot_q4_k_quarter(u0 + boff, xp, qtr);
    }
    ag = simd_sum(ag); au = simd_sum(au);
    if (lane == 0u) h_out[k * p.d_ff + row] = (ag / (1.0f + exp(-ag))) * au;
}

kernel void moe_topk_down_accum_q4k(
    const device uchar* down_exps [[buffer(0)]], const device float* h_in [[buffer(1)]],
    const device uint* eidx [[buffer(2)]], const device float* ewt [[buffer(3)]],
    device float* x_io [[buffer(4)]], constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4u + sg; if (row >= p.d_model) return;
    uint nb = p.d_ff >> 8; ulong rb = (ulong)nb * 144ul; uint qtr = lane & 3u;
    float acc = 0.0f;
    for (uint k = 0; k < p.n_active; k++) {
        const device uchar* wd = down_exps + (ulong)eidx[k] * p.down_exp_bytes + (ulong)row * rb;
        const device float* hk = h_in + (ulong)k * p.d_ff;
        float s = 0.0f;
        for (uint b = (lane >> 2); b < nb; b += 8u) s += dot_q4_k_quarter(wd + (ulong)b * 144ul, hk + (b << 8), qtr);
        acc += ewt[k] * simd_sum(s);
    }
    if (lane == 0u) x_io[row] += acc;
}

kernel void moe_slot_gate_up_q4k(
    const device uchar* pool [[buffer(0)]], const device float* xn [[buffer(1)]],
    const device uint* slots [[buffer(2)]], device float* h_out [[buffer(3)]],
    constant MoeExpArgs& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 4u + sg; uint k = tgid.y;
    if (row >= p.d_ff || k >= p.n_active) return;
    const device uchar* base = pool + (ulong)slots[k] * p.slot_stride_bytes;
    uint nb = p.d_model >> 8; ulong rb = (ulong)nb * 144ul;
    const device uchar* g0 = base + (ulong)row * rb;
    const device uchar* u0 = base + p.gate_exp_bytes + (ulong)row * rb;
    uint qtr = lane & 3u; float ag = 0.0f, au = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float* xp = xn + (b << 8); ulong boff = (ulong)b * 144ul;
        ag += dot_q4_k_quarter(g0 + boff, xp, qtr); au += dot_q4_k_quarter(u0 + boff, xp, qtr);
    }
    ag = simd_sum(ag); au = simd_sum(au);
    if (lane == 0u) h_out[k * p.d_ff + row] = (ag / (1.0f + exp(-ag))) * au;
}

kernel void moe_slot_down_accum_q4k(
    const device uchar* pool [[buffer(0)]], const device float* h_in [[buffer(1)]],
    const device uint* slots [[buffer(2)]], const device float* ewt [[buffer(3)]],
    device float* x_io [[buffer(4)]], constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4u + sg; if (row >= p.d_model) return;
    uint nb = p.d_ff >> 8; ulong rb = (ulong)nb * 144ul; uint qtr = lane & 3u;
    float acc = 0.0f;
    for (uint k = 0; k < p.n_active; k++) {
        const device uchar* wd = pool + (ulong)slots[k] * p.slot_stride_bytes + p.gate_exp_bytes + p.up_exp_bytes + (ulong)row * rb;
        const device float* hk = h_in + (ulong)k * p.d_ff;
        float s = 0.0f;
        for (uint b = (lane >> 2); b < nb; b += 8u) s += dot_q4_k_quarter(wd + (ulong)b * 144ul, hk + (b << 8), qtr);
        acc += ewt[k] * simd_sum(s);
    }
    if (lane == 0u) x_io[row] += acc;
}

kernel void moe_topk_down_accum_q5_0(
    const device uchar* down_exps [[buffer(0)]], const device float* h_in [[buffer(1)]],
    const device uint* e_idx [[buffer(2)]], const device float* e_wt [[buffer(3)]],
    device float* x_res [[buffer(4)]], constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4u + sg; if (row >= p.d_model) return;
    uint nb = p.d_ff >> 5; ulong rb = (ulong)nb * 22ul; float acc = 0.0f;
    for (uint k = 0; k < p.n_active; k++) {
        const device uchar* wd = down_exps + (ulong)e_idx[k] * p.down_exp_bytes + (ulong)row * rb;
        const device float* hk = h_in + (ulong)k * p.d_ff; float s = 0.0f;
        for (uint b = lane; b < nb; b += 32u) s += dot_q5_0_fast(wd + (ulong)b * 22ul, hk + (b << 5));
        acc += e_wt[k] * simd_sum(s);
    }
    if (lane == 0u) x_res[row] += acc;
}

kernel void moe_topk_down_accum_q5_1(
    const device uchar* down_exps [[buffer(0)]], const device float* h_in [[buffer(1)]],
    const device uint* e_idx [[buffer(2)]], const device float* e_wt [[buffer(3)]],
    device float* x_res [[buffer(4)]], constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4u + sg; if (row >= p.d_model) return;
    uint nb = p.d_ff >> 5; ulong rb = (ulong)nb * 24ul; float acc = 0.0f;
    for (uint k = 0; k < p.n_active; k++) {
        const device uchar* wd = down_exps + (ulong)e_idx[k] * p.down_exp_bytes + (ulong)row * rb;
        const device float* hk = h_in + (ulong)k * p.d_ff; float s = 0.0f;
        for (uint b = lane; b < nb; b += 32u) s += dot_q5_1_fast(wd + (ulong)b * 24ul, hk + (b << 5));
        acc += e_wt[k] * simd_sum(s);
    }
    if (lane == 0u) x_res[row] += acc;
}

kernel void moe_topk_down_accum_q8(
    const device uchar* down_exps [[buffer(0)]], const device float* h_in [[buffer(1)]],
    const device uint* e_idx [[buffer(2)]], const device float* e_wt [[buffer(3)]],
    device float* x_res [[buffer(4)]], constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4u + sg; if (row >= p.d_model) return;
    uint nb = p.d_ff >> 5; ulong rb = (ulong)nb * 34ul; float acc = 0.0f;
    for (uint k = 0; k < p.n_active; k++) {
        const device uchar* wd = down_exps + (ulong)e_idx[k] * p.down_exp_bytes + (ulong)row * rb;
        const device float* hk = h_in + (ulong)k * p.d_ff; float s = 0.0f;
        for (uint b = lane; b < nb; b += 32u) { thread float4 xv[8]; load_x32_f4(hk + (b << 5), xv); s += dot_q8_0_fast(wd + (ulong)b * 34ul, xv); }
        acc += e_wt[k] * simd_sum(s);
    }
    if (lane == 0u) x_res[row] += acc;
}

kernel void moe_topk_down_accum_q6k(
    const device uchar* down_exps [[buffer(0)]], const device float* h_in [[buffer(1)]],
    const device uint* e_idx [[buffer(2)]], const device float* e_wt [[buffer(3)]],
    device float* x_res [[buffer(4)]], constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4u + sg; if (row >= p.d_model) return;
    uint nb = p.d_ff >> 8; ulong rb = (ulong)nb * 210ul; uint qtr = lane & 3u; float acc = 0.0f;
    for (uint k = 0; k < p.n_active; k++) {
        const device uchar* wd = down_exps + (ulong)e_idx[k] * p.down_exp_bytes + (ulong)row * rb;
        const device float* hk = h_in + (ulong)k * p.d_ff; float s = 0.0f;
        for (uint b = (lane >> 2); b < nb; b += 8u) s += dot_q6_k_quarter(wd + (ulong)b * 210ul, hk + (b << 8), qtr);
        acc += e_wt[k] * simd_sum(s);
    }
    if (lane == 0u) x_res[row] += acc;
}

kernel void moe_slot_down_accum_q5_0(
    const device uchar* pool [[buffer(0)]], const device float* h_in [[buffer(1)]],
    const device uint* slots [[buffer(2)]], const device float* ewt [[buffer(3)]],
    device float* x_io [[buffer(4)]], constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4u + sg; if (row >= p.d_model) return;
    uint nb = p.d_ff >> 5; ulong rb = (ulong)nb * 22ul; float acc = 0.0f;
    for (uint k = 0; k < p.n_active; k++) {
        const device uchar* wd = pool + (ulong)slots[k] * p.slot_stride_bytes + p.gate_exp_bytes + p.up_exp_bytes + (ulong)row * rb;
        const device float* hk = h_in + (ulong)k * p.d_ff; float s = 0.0f;
        for (uint b = lane; b < nb; b += 32u) s += dot_q5_0_fast(wd + (ulong)b * 22ul, hk + (b << 5));
        acc += ewt[k] * simd_sum(s);
    }
    if (lane == 0u) x_io[row] += acc;
}

kernel void moe_slot_down_accum_q5_1(
    const device uchar* pool [[buffer(0)]], const device float* h_in [[buffer(1)]],
    const device uint* slots [[buffer(2)]], const device float* ewt [[buffer(3)]],
    device float* x_io [[buffer(4)]], constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4u + sg; if (row >= p.d_model) return;
    uint nb = p.d_ff >> 5; ulong rb = (ulong)nb * 24ul; float acc = 0.0f;
    for (uint k = 0; k < p.n_active; k++) {
        const device uchar* wd = pool + (ulong)slots[k] * p.slot_stride_bytes + p.gate_exp_bytes + p.up_exp_bytes + (ulong)row * rb;
        const device float* hk = h_in + (ulong)k * p.d_ff; float s = 0.0f;
        for (uint b = lane; b < nb; b += 32u) s += dot_q5_1_fast(wd + (ulong)b * 24ul, hk + (b << 5));
        acc += ewt[k] * simd_sum(s);
    }
    if (lane == 0u) x_io[row] += acc;
}

kernel void moe_slot_down_accum_q8(
    const device uchar* pool [[buffer(0)]], const device float* h_in [[buffer(1)]],
    const device uint* slots [[buffer(2)]], const device float* ewt [[buffer(3)]],
    device float* x_io [[buffer(4)]], constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4u + sg; if (row >= p.d_model) return;
    uint nb = p.d_ff >> 5; ulong rb = (ulong)nb * 34ul; float acc = 0.0f;
    for (uint k = 0; k < p.n_active; k++) {
        const device uchar* wd = pool + (ulong)slots[k] * p.slot_stride_bytes + p.gate_exp_bytes + p.up_exp_bytes + (ulong)row * rb;
        const device float* hk = h_in + (ulong)k * p.d_ff; float s = 0.0f;
        for (uint b = lane; b < nb; b += 32u) { thread float4 xv[8]; load_x32_f4(hk + (b << 5), xv); s += dot_q8_0_fast(wd + (ulong)b * 34ul, xv); }
        acc += ewt[k] * simd_sum(s);
    }
    if (lane == 0u) x_io[row] += acc;
}

kernel void moe_slot_down_accum_q6k(
    const device uchar* pool [[buffer(0)]], const device float* h_in [[buffer(1)]],
    const device uint* slots [[buffer(2)]], const device float* ewt [[buffer(3)]],
    device float* x_io [[buffer(4)]], constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4u + sg; if (row >= p.d_model) return;
    uint nb = p.d_ff >> 8; ulong rb = (ulong)nb * 210ul; uint qtr = lane & 3u; float acc = 0.0f;
    for (uint k = 0; k < p.n_active; k++) {
        const device uchar* wd = pool + (ulong)slots[k] * p.slot_stride_bytes + p.gate_exp_bytes + p.up_exp_bytes + (ulong)row * rb;
        const device float* hk = h_in + (ulong)k * p.d_ff; float s = 0.0f;
        for (uint b = (lane >> 2); b < nb; b += 8u) s += dot_q6_k_quarter(wd + (ulong)b * 210ul, hk + (b << 8), qtr);
        acc += ewt[k] * simd_sum(s);
    }
    if (lane == 0u) x_io[row] += acc;
}

// =========================================================================
// Q8_0 Quantized KV-Cache Kernels (-47% KV VRAM, 16K ctx!) + Batched Prefill Kernels
// =========================================================================
struct MoeRopeBatchArgs {
    uint n_heads; uint n_kv; uint d_head; uint rope_dim;
    uint start_pos; float theta; uint rope_neox; uint has_freqs; uint batch_b;
};

kernel void moe_rope_and_kv_q8_batch(
    device float* q [[buffer(0)]], device float* k [[buffer(1)]], const device float* v [[buffer(2)]],
    device uchar* k_cache [[buffer(3)]], device uchar* v_cache [[buffer(4)]],
    const device float* rope_freqs [[buffer(5)]], constant MoeRopeBatchArgs& p [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    uint h = tgid.x, b = tgid.y;
    if (h >= p.n_heads || b >= p.batch_b) return;
    uint pos = p.start_pos + b;
    uint half_r = p.rope_dim >> 1;
    uint qbase = b * (p.n_heads * p.d_head) + h * p.d_head;
    uint kbase = b * (p.n_kv * p.d_head) + h * p.d_head;
    uint lid = sg * 32u + lane;
    for (uint i = lid; i < half_r; i += 128u) {
        uint idx0 = p.rope_neox ? i : (2u * i);
        uint idx1 = p.rope_neox ? (i + half_r) : (2u * i + 1u);
        float inv_f = 1.0f / pow(p.theta, float(2u * i) / float(p.rope_dim));
        if (p.has_freqs) inv_f /= rope_freqs[i];
        float ang = float(pos) * inv_f;
        float cs = cos(ang), sn = sin(ang);
        float q0 = q[qbase + idx0], q1 = q[qbase + idx1];
        q[qbase + idx0] = q0 * cs - q1 * sn; q[qbase + idx1] = q0 * sn + q1 * cs;
        if (h < p.n_kv) {
            float k0 = k[kbase + idx0], k1 = k[kbase + idx1];
            k[kbase + idx0] = k0 * cs - k1 * sn; k[kbase + idx1] = k0 * sn + k1 * cs;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (h < p.n_kv) {
        uint nblk = p.d_head >> 5;
        ulong row_bytes = (ulong)p.n_kv * (ulong)nblk * 34ul;
        ulong h_off = (ulong)pos * row_bytes + (ulong)h * (ulong)nblk * 34ul;
        for (uint blk = sg; blk < nblk; blk += 4u) {
            float kv = k[kbase + blk * 32u + lane];
            float vv = v[kbase + blk * 32u + lane];
            float kam = simd_max(fabs(kv));
            float vam = simd_max(fabs(vv));
            float kd = kam / 127.0f, kid = (kam > 0.0f) ? (127.0f / kam) : 0.0f;
            float vd = vam / 127.0f, vid = (vam > 0.0f) ? (127.0f / vam) : 0.0f;
            device uchar* kp = k_cache + h_off + (ulong)blk * 34ul;
            device uchar* vp = v_cache + h_off + (ulong)blk * 34ul;
            if (lane == 0u) { *(device half*)kp = half(kd); *(device half*)vp = half(vd); }
            ((device char*)(kp + 2u))[lane] = char(rint(kv * kid));
            ((device char*)(vp + 2u))[lane] = char(rint(vv * vid));
        }
    }
}

kernel void moe_rope_and_kv_batch(
    device float* q [[buffer(0)]], device float* k [[buffer(1)]], const device float* v [[buffer(2)]],
    device half* k_cache [[buffer(3)]], device half* v_cache [[buffer(4)]],
    const device float* rope_freqs [[buffer(5)]], constant MoeRopeBatchArgs& p [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint i = gid.x, h = gid.y, b = gid.z;
    uint half_d = p.d_head >> 1, half_r = p.rope_dim >> 1;
    if (i >= half_d || h >= p.n_heads || b >= p.batch_b) return;
    uint pos = p.start_pos + b;
    uint qoff = b * (p.n_heads * p.d_head) + h * p.d_head;
    uint koff = b * (p.n_kv * p.d_head) + h * p.d_head;
    if (i < half_r) {
        uint idx0 = p.rope_neox ? i : (2u * i);
        uint idx1 = p.rope_neox ? (i + half_r) : (2u * i + 1u);
        float inv_f = 1.0f / pow(p.theta, float(2u * i) / float(p.rope_dim));
        if (p.has_freqs) inv_f /= rope_freqs[i];
        float ang = float(pos) * inv_f;
        float cs = cos(ang), sn = sin(ang);
        float q0 = q[qoff + idx0], q1 = q[qoff + idx1];
        q[qoff + idx0] = q0 * cs - q1 * sn; q[qoff + idx1] = q0 * sn + q1 * cs;
        if (h < p.n_kv) {
            float k0 = k[koff + idx0], k1 = k[koff + idx1];
            float rk0 = k0 * cs - k1 * sn, rk1 = k0 * sn + k1 * cs;
            k[koff + idx0] = rk0; k[koff + idx1] = rk1;
            ulong cbase = (ulong)pos * (p.n_kv * p.d_head) + h * p.d_head;
            k_cache[cbase + idx0] = half(rk0); k_cache[cbase + idx1] = half(rk1);
            v_cache[cbase + idx0] = half(v[koff + idx0]); v_cache[cbase + idx1] = half(v[koff + idx1]);
        }
    } else if (h < p.n_kv) {
        uint idx0 = 2u * i, idx1 = 2u * i + 1u;
        ulong cbase = (ulong)pos * (p.n_kv * p.d_head) + h * p.d_head;
        k_cache[cbase + idx0] = half(k[koff + idx0]); k_cache[cbase + idx1] = half(k[koff + idx1]);
        v_cache[cbase + idx0] = half(v[koff + idx0]); v_cache[cbase + idx1] = half(v[koff + idx1]);
    }
}

struct MoeAttnBatchArgs { uint n_heads; uint n_kv; uint d_head; uint start_pos; uint batch_b; uint out_stride; float scale; };

kernel void moe_flash_attn_q8(
    const device float* q [[buffer(0)]], const device uchar* k_cache [[buffer(1)]], const device uchar* v_cache [[buffer(2)]],
    device float* ctx [[buffer(3)]], constant MoeAttnBatchArgs& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x, b = tgid.y;
    if (h >= p.n_heads || b >= p.batch_b) return;
    uint kv_len = p.start_pos + b + 1u;
    uint grp = max(p.n_heads / p.n_kv, 1u);
    uint kvh = h / grp;
    uint nblk = p.d_head >> 5;
    ulong row_bytes = (ulong)p.n_kv * (ulong)nblk * 34ul;
    float scale = p.scale;

    threadgroup float s_q[256]; threadgroup float s_m[4]; threadgroup float s_l[4]; threadgroup float s_o[4][256];
    ulong qbase = (ulong)b * (p.n_heads * p.d_head) + (ulong)h * p.d_head;
    ulong cbase = (ulong)b * (p.n_heads * p.out_stride) + (ulong)h * p.out_stride;
    for (uint d = lid; d < p.d_head; d += 128u) s_q[d] = q[qbase + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_run = -1e30f, l_run = 0.0f;
    float o_acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    for (uint t = sg; t < kv_len; t += 4u) {
        ulong h_off = (ulong)t * row_bytes + (ulong)kvh * (ulong)nblk * 34ul;
        const device uchar* kp = k_cache + h_off;
        const device uchar* vp = v_cache + h_off;
        float dot = 0.0f;
        for (uint blk = 0; blk < nblk; blk++) {
            const device uchar* kb = kp + (ulong)blk * 34ul;
            float kd = float(*(const device half*)kb);
            dot = fma(s_q[blk * 32u + lane], float(((const device char*)(kb + 2u))[lane]) * kd, dot);
        }
        float sc = simd_sum(dot) * scale;
        float mn = max(m_run, sc); float al = exp(m_run - mn); float wt = exp(sc - mn);
        l_run = fma(l_run, al, wt); m_run = mn;
        for (uint blk = 0; blk < nblk; blk++) {
            const device uchar* vb = vp + (ulong)blk * 34ul;
            float vd = float(*(const device half*)vb);
            o_acc[blk] = fma(o_acc[blk], al, wt * float(((const device char*)(vb + 2u))[lane]) * vd);
        }
    }
    if (lane == 0u) { s_m[sg] = m_run; s_l[sg] = l_run; }
    for (uint blk = 0; blk < nblk; blk++) s_o[sg][blk * 32u + lane] = o_acc[blk];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gm = max(max(s_m[0], s_m[1]), max(s_m[2], s_m[3]));
    float gl = 0.0f;
    for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * exp(s_m[s] - gm);
    float inv = 1.0f / max(gl, 1e-12f);
    for (uint d = lid; d < p.out_stride; d += 128u) {
        float ov = 0.0f;
        for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fma(s_o[s][d], exp(s_m[s] - gm), ov);
        ctx[cbase + d] = ov * inv;
    }
}

kernel void moe_flash_prefill_fp16(
    const device float* q [[buffer(0)]], const device half* k_cache [[buffer(1)]], const device half* v_cache [[buffer(2)]],
    device float* ctx [[buffer(3)]], constant MoeAttnBatchArgs& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x, b = tgid.y;
    if (h >= p.n_heads || b >= p.batch_b) return;
    uint kv_len = p.start_pos + b + 1u;
    uint grp = max(p.n_heads / p.n_kv, 1u);
    uint kvh = h / grp;
    uint kv_stride = p.n_kv * p.d_head;
    float scale = p.scale;

    threadgroup float s_q[256]; threadgroup float s_m[4]; threadgroup float s_l[4]; threadgroup float s_o[4][256];
    ulong qbase = (ulong)b * (p.n_heads * p.d_head) + (ulong)h * p.d_head;
    ulong cbase = (ulong)b * (p.n_heads * p.out_stride) + (ulong)h * p.out_stride;
    for (uint d = lid; d < p.d_head; d += 128u) s_q[d] = q[qbase + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_run = -1e30f, l_run = 0.0f;
    float o_acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    uint d_per_lane = (p.d_head + 31u) >> 5;
    for (uint t = sg; t < kv_len; t += 4u) {
        const device half* kp = k_cache + (ulong)t * kv_stride + kvh * p.d_head;
        const device half* vp = v_cache + (ulong)t * kv_stride + kvh * p.d_head;
        float dot = 0.0f;
        for (uint k = 0; k < d_per_lane; k++) { uint d = lane + k * 32u; if (d < p.d_head) dot = fma(s_q[d], float(kp[d]), dot); }
        float sc = simd_sum(dot) * scale;
        float mn = max(m_run, sc); float al = exp(m_run - mn); float wt = exp(sc - mn);
        l_run = fma(l_run, al, wt); m_run = mn;
        for (uint k = 0; k < d_per_lane; k++) { uint d = lane + k * 32u; if (d < p.d_head) o_acc[k] = fma(o_acc[k], al, wt * float(vp[d])); }
    }
    if (lane == 0u) { s_m[sg] = m_run; s_l[sg] = l_run; }
    for (uint k = 0; k < d_per_lane; k++) { uint d = lane + k * 32u; if (d < p.d_head) s_o[sg][d] = o_acc[k]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gm = max(max(s_m[0], s_m[1]), max(s_m[2], s_m[3]));
    float gl = 0.0f;
    for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * exp(s_m[s] - gm);
    float inv = 1.0f / max(gl, 1e-12f);
    for (uint d = lid; d < p.out_stride; d += 128u) {
        float ov = 0.0f;
        for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fma(s_o[s][d], exp(s_m[s] - gm), ov);
        ctx[cbase + d] = ov * inv;
    }
}

struct MoeGemmBatchArgs { uint out_dim; uint in_dim; uint batch_b; uint add_res; };

kernel void moe_add_bias_batch(
    device float* x [[buffer(0)]], const device float* bias [[buffer(1)]],
    constant uint2& dim_b [[buffer(2)]], uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x < dim_b.x && gid.y < dim_b.y) x[(ulong)gid.y * dim_b.x + gid.x] += bias[gid.x];
}

// Ultra-fast Register-Direct Small-Batch Q4_0 GEMM (`B <= 8`: 1 packed_ushort4 DRAM read shared across B tokens -> 4x faster Speculative Verify!)
// Coalesced Shared-X `half4` Small-Batch Q4_0 GEMM (`B <= 8`, 16 rows/TG, 128 thr/TG = 16 rows x 8 contiguous blocks!):
//   - Double-buffered / single-barrier coalesced float4 -> half4 staging of `sX[8][256]` (4 KB TG mem -> 16x X reuse!)
//   - Pre-subtract 8.0h ONCE per weight block outside the `t` loop -> eliminates `sXsum` & saves 2 barriers/tile!
//   - Native Apple GPU `half4` FMA -> ~380-430 GB/s DRAM saturation across B=2..8!
kernel void moe_gemm_q4_small_b(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]
) {
    uint r_local = lid >> 3;        // 0..15 (16 output rows per TG)
    uint blk_off = lid & 7u;        // 0..7  (8 consecutive Q4_0 blocks = 256 K-elements)
    uint row = tgid.x * 16u + r_local;
    uint nb = p.in_dim >> 5, B = min(8u, p.batch_b);
    ulong rb = (ulong)nb * 18ul;
    const device uchar* wp = w + (ulong)(row < p.out_dim ? row : (p.out_dim - 1u)) * rb;

    threadgroup half sX[8][256];
    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    // Map 128 threads to 64 float4 chunks per token (threads 0..63 load token 2*k, 64..127 load token 2*k+1!)
    uint half_lid = lid & 63u;      // 0..63 -> 64 float4 = 256 floats
    uint tok_grp  = lid >> 6;       // 0 or 1

    for (uint bk0 = 0; bk0 < nb; bk0 += 8u) {
        uint k_off = (bk0 << 5) + (half_lid << 2);
        for (uint t0 = 0; t0 < B; t0 += 2u) {
            uint t = t0 + tok_grp;
            if (t < B) {
                half4 hv = (k_off < p.in_dim) ? half4(*(const device float4*)(x + (ulong)t * p.in_dim + k_off)) : half4(0.0h);
                *(threadgroup half4*)(&sX[t][half_lid << 2]) = hv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint b = bk0 + blk_off;
        if (row < p.out_dim && b < nb) {
            const device uchar* blk = wp + (ulong)b * 18ul;
            float d = float(*(const device half*)blk);
            const device packed_ushort4* q64 = (const device packed_ushort4*)(blk + 2u);
            ushort4 p0 = q64[0], p1 = q64[1];
            uchar4 c0 = as_type<uchar4>(p0.xy), c1 = as_type<uchar4>(p0.zw);
            uchar4 c2 = as_type<uchar4>(p1.xy), c3 = as_type<uchar4>(p1.zw);
            half4 l0 = half4(c0 & uchar4(0x0Fu)) - 8.0h, h0 = half4(c0 >> 4) - 8.0h;
            half4 l1 = half4(c1 & uchar4(0x0Fu)) - 8.0h, h1 = half4(c1 >> 4) - 8.0h;
            half4 l2 = half4(c2 & uchar4(0x0Fu)) - 8.0h, h2 = half4(c2 >> 4) - 8.0h;
            half4 l3 = half4(c3 & uchar4(0x0Fu)) - 8.0h, h3 = half4(c3 >> 4) - 8.0h;
            uint xoff = blk_off << 5;
            for (uint t = 0; t < B; t++) {
                const threadgroup half4* xp = (const threadgroup half4*)(&sX[t][xoff]);
                half4 s0 = xp[0] * l0 + xp[4] * h0;
                s0 = fma(xp[1], l1, fma(xp[5], h1, s0));
                s0 = fma(xp[2], l2, fma(xp[6], h2, s0));
                s0 = fma(xp[3], l3, fma(xp[7], h3, s0));
                float4 sf = float4(s0);
                acc[t] = fma(d, (sf.x + sf.y) + (sf.z + sf.w), acc[t]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint t = 0; t < B; t++) {
        float v = acc[t];
        v += simd_shuffle_xor(v, 4u);
        v += simd_shuffle_xor(v, 2u);
        v += simd_shuffle_xor(v, 1u);
        if (blk_off == 0u && row < p.out_dim) {
            ulong idx = (ulong)t * p.out_dim + row;
            y[idx] = p.add_res ? (y[idx] + v) : v;
        }
    }
}

// Hardware AMX `simdgroup_float8x8` Batched Q4_0 GEMM (BM=32 tokens x BN=32 rows x BK=32 -> 1 DRAM pass + 16 TFLOPS AMX!)
kernel void moe_gemm_q4_batch(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 32u, b0 = tgid.y * 32u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 18ul;
    threadgroup half sX[32][32];
    threadgroup half sW[32][32];
    threadgroup float sO[4][64];

    uint n_tb = min(4u, (p.batch_b - b0 + 7u) >> 3);
    simdgroup_float8x8 acc[4];
    #pragma unroll
    for (uint tb = 0; tb < 4u; tb++) acc[tb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint tr = lid >> 2;        // 0..31 (token or output row)
    uint g  = lid & 3u;        // 0..3  (chunk of 4 bytes / 8 floats)
    uint c0 = g * 8u;
    uint glob_b = b0 + tr;
    uint glob_r = r0 + tr;
    const device uchar* wrow = w + (ulong)(glob_r < p.out_dim ? glob_r : (p.out_dim - 1u)) * rb;

    for (uint bk = 0; bk < nb; bk++) {
        // 1. Load sX[32][32] (half)
        if (glob_b < p.batch_b) {
            const device float4* xp = (const device float4*)(x + (ulong)glob_b * p.in_dim + (bk << 5) + c0);
            float4 v0 = xp[0], v1 = xp[1];
            sX[tr][c0 + 0u] = half(v0.x); sX[tr][c0 + 1u] = half(v0.y); sX[tr][c0 + 2u] = half(v0.z); sX[tr][c0 + 3u] = half(v0.w);
            sX[tr][c0 + 4u] = half(v1.x); sX[tr][c0 + 5u] = half(v1.y); sX[tr][c0 + 6u] = half(v1.z); sX[tr][c0 + 7u] = half(v1.w);
        } else if (tr < n_tb * 8u) {
            #pragma unroll
            for (uint j = 0; j < 8u; j++) sX[tr][c0 + j] = half(0.0f);
        }
        // 2. Dequantize Q4_0 block `bk` for 32 rows into sW[32][32] (4 threads per row = 4 bytes/thread!)
        if (glob_r < p.out_dim) {
            const device uchar* blk = wrow + (ulong)bk * 18ul;
            float d = float(*(const device half*)blk);
            uint qoff = g * 4u;
            #pragma unroll
            for (uint j = 0; j < 4u; j++) {
                uchar bv = blk[2u + qoff + j];
                sW[tr][qoff + j]       = half(float(int(bv & 0xFu) - 8) * d);
                sW[tr][16u + qoff + j] = half(float(int(bv >> 4) - 8) * d);
            }
        } else {
            #pragma unroll
            for (uint j = 0; j < 4u; j++) { sW[tr][g * 4u + j] = half(0.0f); sW[tr][16u + g * 4u + j] = half(0.0f); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. AMX `simdgroup_multiply_accumulate` (`sg` owns 8 output rows x active token tiles across K=32!)
        uint sg_r = sg * 8u;
        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 mW;
            simdgroup_load(mW, &sW[sg_r][kk], 32, ulong2(0, 0), true);
            for (uint tb = 0; tb < n_tb; tb++) {
                simdgroup_half8x8 mX;
                simdgroup_load(mX, &sX[tb * 8u][kk], 32, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(acc[tb], mX, mW, acc[tb]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint sg_r = sg * 8u;
    for (uint tb = 0; tb < n_tb; tb++) {
        simdgroup_store(acc[tb], &sO[sg][0], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = lane; idx < 64u; idx += 32u) {
            uint lt = idx >> 3, lr = idx & 7u;
            uint ob = b0 + tb * 8u + lt, orow = r0 + sg_r + lr;
            if (ob < p.batch_b && orow < p.out_dim) {
                ulong oidx = (ulong)ob * p.out_dim + orow;
                float val = sO[sg][idx];
                y[oidx] = p.add_res ? (y[oidx] + val) : val;
            }
        }
    }
}

// Hardware AMX `simdgroup_float8x8` Batched Q4_1 GEMM (BM=32 x BN=32 x BK=32!)
kernel void moe_gemm_q4_1_batch(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 32u, b0 = tgid.y * 32u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 20ul;
    threadgroup half sX[32][32];
    threadgroup half sW[32][32];
    threadgroup float sO[4][64];

    uint n_tb = min(4u, (p.batch_b - b0 + 7u) >> 3);
    simdgroup_float8x8 acc[4];
    #pragma unroll
    for (uint tb = 0; tb < 4u; tb++) acc[tb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint tr = lid >> 2, g = lid & 3u, c0 = g * 8u;
    uint glob_b = b0 + tr, glob_r = r0 + tr;
    const device uchar* wrow = w + (ulong)(glob_r < p.out_dim ? glob_r : (p.out_dim - 1u)) * rb;

    for (uint bk = 0; bk < nb; bk++) {
        if (glob_b < p.batch_b) {
            const device float4* xp = (const device float4*)(x + (ulong)glob_b * p.in_dim + (bk << 5) + c0);
            float4 v0 = xp[0], v1 = xp[1];
            sX[tr][c0 + 0u] = half(v0.x); sX[tr][c0 + 1u] = half(v0.y); sX[tr][c0 + 2u] = half(v0.z); sX[tr][c0 + 3u] = half(v0.w);
            sX[tr][c0 + 4u] = half(v1.x); sX[tr][c0 + 5u] = half(v1.y); sX[tr][c0 + 6u] = half(v1.z); sX[tr][c0 + 7u] = half(v1.w);
        } else if (tr < n_tb * 8u) {
            #pragma unroll
            for (uint j = 0; j < 8u; j++) sX[tr][c0 + j] = half(0.0f);
        }
        if (glob_r < p.out_dim) {
            const device uchar* blk = wrow + (ulong)bk * 20ul;
            float d = float(*(const device half*)blk), mv = float(*(const device half*)(blk + 2u));
            uint qoff = g * 4u;
            #pragma unroll
            for (uint j = 0; j < 4u; j++) {
                uchar bv = blk[4u + qoff + j];
                sW[tr][qoff + j]       = half(fma(float(bv & 0xFu), d, mv));
                sW[tr][16u + qoff + j] = half(fma(float(bv >> 4), d, mv));
            }
        } else {
            #pragma unroll
            for (uint j = 0; j < 4u; j++) { sW[tr][g * 4u + j] = half(0.0f); sW[tr][16u + g * 4u + j] = half(0.0f); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint sg_r = sg * 8u;
        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 mW;
            simdgroup_load(mW, &sW[sg_r][kk], 32, ulong2(0, 0), true);
            for (uint tb = 0; tb < n_tb; tb++) {
                simdgroup_half8x8 mX;
                simdgroup_load(mX, &sX[tb * 8u][kk], 32, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(acc[tb], mX, mW, acc[tb]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint sg_r = sg * 8u;
    for (uint tb = 0; tb < n_tb; tb++) {
        simdgroup_store(acc[tb], &sO[sg][0], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = lane; idx < 64u; idx += 32u) {
            uint lt = idx >> 3, lr = idx & 7u;
            uint ob = b0 + tb * 8u + lt, orow = r0 + sg_r + lr;
            if (ob < p.batch_b && orow < p.out_dim) {
                ulong oidx = (ulong)ob * p.out_dim + orow;
                float val = sO[sg][idx];
                y[oidx] = p.add_res ? (y[oidx] + val) : val;
            }
        }
    }
}

kernel void moe_gemm_q5_0_batch(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 4u + sg; uint b0 = tgid.y * 8u;
    if (row >= p.out_dim || b0 >= p.batch_b) return;
    uint bt = min(8u, p.batch_b - b0); uint nb = p.in_dim >> 5;
    const device uchar* wp = w + (ulong)row * (ulong)nb * 22ul;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (uint b = lane; b < nb; b += 32u) {
        const device uchar* blk = wp + (ulong)b * 22ul;
        for (uint t = 0; t < bt; t++) acc[t] += dot_q5_0_fast(blk, x + (ulong)(b0 + t) * p.in_dim + (b << 5));
    }
    for (uint t = 0; t < bt; t++) {
        float res = simd_sum(acc[t]);
        if (lane == 0u) { ulong idx = (ulong)(b0 + t) * p.out_dim + row; y[idx] = p.add_res ? (y[idx] + res) : res; }
    }
}

kernel void moe_gemm_q5_1_batch(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 4u + sg; uint b0 = tgid.y * 8u;
    if (row >= p.out_dim || b0 >= p.batch_b) return;
    uint bt = min(8u, p.batch_b - b0); uint nb = p.in_dim >> 5;
    const device uchar* wp = w + (ulong)row * (ulong)nb * 24ul;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (uint b = lane; b < nb; b += 32u) {
        const device uchar* blk = wp + (ulong)b * 24ul;
        for (uint t = 0; t < bt; t++) acc[t] += dot_q5_1_fast(blk, x + (ulong)(b0 + t) * p.in_dim + (b << 5));
    }
    for (uint t = 0; t < bt; t++) {
        float res = simd_sum(acc[t]);
        if (lane == 0u) { ulong idx = (ulong)(b0 + t) * p.out_dim + row; y[idx] = p.add_res ? (y[idx] + res) : res; }
    }
}

kernel void moe_gemm_q8_batch(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 4u + sg; uint b0 = tgid.y * 8u;
    if (row >= p.out_dim || b0 >= p.batch_b) return;
    uint bt = min(8u, p.batch_b - b0); uint nb = p.in_dim >> 5;
    const device uchar* wp = w + (ulong)row * (ulong)nb * 34ul;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (uint b = lane; b < nb; b += 32u) {
        const device uchar* blk = wp + (ulong)b * 34ul;
        float d = float(*(const device half*)blk);
        const device char* qs = (const device char*)(blk + 2u);
        for (uint t = 0; t < bt; t++) {
            const device float* xt = x + (ulong)(b0 + t) * p.in_dim + (b << 5);
            float s = 0.0f;
            for (uint i = 0; i < 32u; i++) s = fma(float(qs[i]), xt[i], s);
            acc[t] += d * s;
        }
    }
    for (uint t = 0; t < bt; t++) {
        float res = simd_sum(acc[t]);
        if (lane == 0u) { ulong idx = (ulong)(b0 + t) * p.out_dim + row; y[idx] = p.add_res ? (y[idx] + res) : res; }
    }
}

// AMX simdgroup_float8x8 Batched Q4_K GEMM (32 tokens x 32 rows -> 6x faster Q4_K_M prefill on M4 Max!)
kernel void moe_gemm_q4k_batch(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 32u, b0 = tgid.y * 32u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 8;
    ulong rb = (ulong)nb * 144ul;
    threadgroup float sX[32][32];
    threadgroup float sW[32][32];
    threadgroup float sO[4][64];
    uint n_tb = min(4u, (p.batch_b - b0 + 7u) >> 3);
    simdgroup_float8x8 acc[4];
    for (uint tb = 0; tb < 4u; tb++) acc[tb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint tr = lid >> 2, g = lid & 3u, c0 = g * 8u;
    uint gb = b0 + tr, gr = r0 + tr;
    const device uchar* wrow = w + (ulong)min(gr, p.out_dim - 1u) * rb;
    for (uint bk = 0; bk < nb; bk++) {
        const device uchar* blk = wrow + (ulong)bk * 144ul;
        float d = float(*(const device half*)blk), dm = float(*(const device half*)(blk + 2u));
        const device uchar* sc = blk + 4u;
        const device uchar* qs = blk + 16u;
        for (uint sb = 0; sb < 8u; sb++) {
            uint kbase = (bk << 8) + (sb << 5);
            if (gb < p.batch_b) {
                const device float4* xp = (const device float4*)(x + (ulong)gb * p.in_dim + kbase + c0);
                float4 v0 = xp[0], v1 = xp[1];
                sX[tr][c0+0]=v0.x; sX[tr][c0+1]=v0.y; sX[tr][c0+2]=v0.z; sX[tr][c0+3]=v0.w;
                sX[tr][c0+4]=v1.x; sX[tr][c0+5]=v1.y; sX[tr][c0+6]=v1.z; sX[tr][c0+7]=v1.w;
            } else if (tr < n_tb * 8u) { for (uint j = 0; j < 8u; j++) sX[tr][c0 + j] = 0.0f; }
            if (gr < p.out_dim) {
                uchar scv, mv;
                if (sb < 4u) { scv = sc[sb] & 63u; mv = sc[sb + 4u] & 63u; }
                else { scv = (sc[sb + 4u] & 0xFu) | ((sc[sb - 4u] >> 6) << 4); mv = (sc[sb + 4u] >> 4) | ((sc[sb] >> 6) << 4); }
                float dl = d * float(scv), ml = dm * float(mv);
                uint j64 = sb >> 1, hi = sb & 1u;
                const device uchar* qp = qs + j64 * 32u + c0;
                #pragma unroll
                for (uint j = 0; j < 8u; j++) {
                    uchar bv = qp[j];
                    uint qv = hi ? (bv >> 4) : (bv & 0xFu);
                    sW[tr][c0 + j] = fma(dl, float(qv), -ml);
                }
            } else { for (uint j = 0; j < 8u; j++) sW[tr][c0 + j] = 0.0f; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint sg_r = sg * 8u;
            #pragma unroll
            for (uint kk = 0; kk < 32u; kk += 8u) {
                simdgroup_float8x8 mW; simdgroup_load(mW, &sW[sg_r][kk], 32, ulong2(0, 0), true);
                for (uint tb = 0; tb < n_tb; tb++) {
                    simdgroup_float8x8 mX; simdgroup_load(mX, &sX[tb * 8u][kk], 32, ulong2(0, 0), false);
                    simdgroup_multiply_accumulate(acc[tb], mX, mW, acc[tb]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    uint sg_r = sg * 8u;
    for (uint tb = 0; tb < n_tb; tb++) {
        simdgroup_store(acc[tb], &sO[sg][0], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = lane; idx < 64u; idx += 32u) {
            uint lt = idx >> 3, lr = idx & 7u;
            uint ob = b0 + tb * 8u + lt, orow = r0 + sg_r + lr;
            if (ob < p.batch_b && orow < p.out_dim) {
                ulong oidx = (ulong)ob * p.out_dim + orow;
                float val = sO[sg][idx];
                y[oidx] = p.add_res ? (y[oidx] + val) : val;
            }
        }
    }
}

kernel void moe_gemv_q2k(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 4u + sg) * 2u, row1 = row0 + 1u;
    if (row0 >= p.out_dim) return;
    uint nb = p.in_dim >> 8, qtr = lane & 3u;
    ulong rb = (ulong)nb * 84ul;
    const device uchar* wp0 = w + (ulong)row0 * rb;
    const device uchar* wp1 = (row1 < p.out_dim) ? (wp0 + rb) : wp0;
    float a0 = 0.0f, a1 = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float* xp = x + (b << 8);
        a0 += dot_q2_k_quarter(wp0 + (ulong)b * 84ul, xp, qtr);
        a1 += dot_q2_k_quarter(wp1 + (ulong)b * 84ul, xp, qtr);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row1 < p.out_dim) y[row1] = p.add_res ? (y[row1] + a1) : a1;
    }
}

kernel void moe_gemv_q3k(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 4u + sg) * 2u, row1 = row0 + 1u;
    if (row0 >= p.out_dim) return;
    uint nb = p.in_dim >> 8, qtr = lane & 3u;
    ulong rb = (ulong)nb * 110ul;
    const device uchar* wp0 = w + (ulong)row0 * rb;
    const device uchar* wp1 = (row1 < p.out_dim) ? (wp0 + rb) : wp0;
    float a0 = 0.0f, a1 = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float* xp = x + (b << 8);
        a0 += dot_q3_k_quarter(wp0 + (ulong)b * 110ul, xp, qtr);
        a1 += dot_q3_k_quarter(wp1 + (ulong)b * 110ul, xp, qtr);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row1 < p.out_dim) y[row1] = p.add_res ? (y[row1] + a1) : a1;
    }
}

kernel void moe_gemv_iq4xs(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 4u + sg) * 2u, row1 = row0 + 1u;
    if (row0 >= p.out_dim) return;
    uint nb = p.in_dim >> 8, qtr = lane & 3u;
    ulong rb = (ulong)nb * 136ul;
    const device uchar* wp0 = w + (ulong)row0 * rb;
    const device uchar* wp1 = (row1 < p.out_dim) ? (wp0 + rb) : wp0;
    float a0 = 0.0f, a1 = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float* xp = x + (b << 8);
        a0 += dot_iq4_xs_quarter(wp0 + (ulong)b * 136ul, xp, qtr);
        a1 += dot_iq4_xs_quarter(wp1 + (ulong)b * 136ul, xp, qtr);
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row1 < p.out_dim) y[row1] = p.add_res ? (y[row1] + a1) : a1;
    }
}

kernel void moe_gemm_q2k_batch(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 4u + sg, b0 = tgid.y * 4u;
    if (row >= p.out_dim || b0 >= p.batch_b) return;
    uint bt = min(4u, p.batch_b - b0), nb = p.in_dim >> 8, qtr = lane & 3u;
    const device uchar* wp = w + (ulong)row * (ulong)nb * 84ul;
    float acc[4] = {0,0,0,0};
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device uchar* blk = wp + (ulong)b * 84ul;
        for (uint t = 0; t < bt; t++) acc[t] += dot_q2_k_quarter(blk, x + (ulong)(b0 + t) * p.in_dim + (b << 8), qtr);
    }
    for (uint t = 0; t < bt; t++) { float r = simd_sum(acc[t]); if (lane == 0u) { ulong idx = (ulong)(b0 + t) * p.out_dim + row; y[idx] = p.add_res ? (y[idx] + r) : r; } }
}

kernel void moe_gemm_q3k_batch(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 4u + sg, b0 = tgid.y * 4u;
    if (row >= p.out_dim || b0 >= p.batch_b) return;
    uint bt = min(4u, p.batch_b - b0), nb = p.in_dim >> 8, qtr = lane & 3u;
    const device uchar* wp = w + (ulong)row * (ulong)nb * 110ul;
    float acc[4] = {0,0,0,0};
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device uchar* blk = wp + (ulong)b * 110ul;
        for (uint t = 0; t < bt; t++) acc[t] += dot_q3_k_quarter(blk, x + (ulong)(b0 + t) * p.in_dim + (b << 8), qtr);
    }
    for (uint t = 0; t < bt; t++) { float r = simd_sum(acc[t]); if (lane == 0u) { ulong idx = (ulong)(b0 + t) * p.out_dim + row; y[idx] = p.add_res ? (y[idx] + r) : r; } }
}

kernel void moe_gemm_iq4xs_batch(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 4u + sg, b0 = tgid.y * 4u;
    if (row >= p.out_dim || b0 >= p.batch_b) return;
    uint bt = min(4u, p.batch_b - b0), nb = p.in_dim >> 8, qtr = lane & 3u;
    const device uchar* wp = w + (ulong)row * (ulong)nb * 136ul;
    float acc[4] = {0,0,0,0};
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device uchar* blk = wp + (ulong)b * 136ul;
        for (uint t = 0; t < bt; t++) acc[t] += dot_iq4_xs_quarter(blk, x + (ulong)(b0 + t) * p.in_dim + (b << 8), qtr);
    }
    for (uint t = 0; t < bt; t++) { float r = simd_sum(acc[t]); if (lane == 0u) { ulong idx = (ulong)(b0 + t) * p.out_dim + row; y[idx] = p.add_res ? (y[idx] + r) : r; } }
}

kernel void moe_gemm_q5k_batch(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 4u + sg; uint b0 = tgid.y * 4u;
    if (row >= p.out_dim || b0 >= p.batch_b) return;
    uint bt = min(4u, p.batch_b - b0); uint nb = p.in_dim >> 8; uint qtr = lane & 3u;
    const device uchar* wp = w + (ulong)row * (ulong)nb * 176ul;
    float acc[4] = {0,0,0,0};
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device uchar* blk = wp + (ulong)b * 176ul;
        for (uint t = 0; t < bt; t++) acc[t] += dot_q5_k_quarter(blk, x + (ulong)(b0 + t) * p.in_dim + (b << 8), qtr);
    }
    for (uint t = 0; t < bt; t++) {
        float res = simd_sum(acc[t]);
        if (lane == 0u) { ulong idx = (ulong)(b0 + t) * p.out_dim + row; y[idx] = p.add_res ? (y[idx] + res) : res; }
    }
}

kernel void moe_gemm_q6k_batch(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 4u + sg; uint b0 = tgid.y * 4u;
    if (row >= p.out_dim || b0 >= p.batch_b) return;
    uint bt = min(4u, p.batch_b - b0); uint nb = p.in_dim >> 8; uint qtr = lane & 3u;
    const device uchar* wp = w + (ulong)row * (ulong)nb * 210ul;
    float acc[4] = {0,0,0,0};
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device uchar* blk = wp + (ulong)b * 210ul;
        for (uint t = 0; t < bt; t++) acc[t] += dot_q6_k_quarter(blk, x + (ulong)(b0 + t) * p.in_dim + (b << 8), qtr);
    }
    for (uint t = 0; t < bt; t++) {
        float res = simd_sum(acc[t]);
        if (lane == 0u) { ulong idx = (ulong)(b0 + t) * p.out_dim + row; y[idx] = p.add_res ? (y[idx] + res) : res; }
    }
}

kernel void moe_gemm_f16_batch(
    const device half* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 4u + sg; uint b0 = tgid.y * 8u;
    if (row >= p.out_dim || b0 >= p.batch_b) return;
    uint bt = min(8u, p.batch_b - b0); uint n4 = p.in_dim >> 2;
    const device half4* wp = (const device half4*)(w + (ulong)row * p.in_dim);
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (uint i = lane; i < n4; i += 32u) {
        float4 wf = float4(wp[i]);
        for (uint t = 0; t < bt; t++) acc[t] += dot(wf, ((const device float4*)(x + (ulong)(b0 + t) * p.in_dim))[i]);
    }
    for (uint t = 0; t < bt; t++) {
        float res = simd_sum(acc[t]);
        if (lane == 0u) { ulong idx = (ulong)(b0 + t) * p.out_dim + row; y[idx] = p.add_res ? (y[idx] + res) : res; }
    }
}

kernel void moe_gemm_f32_batch(
    const device float* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 4u + sg; uint b0 = tgid.y * 8u;
    if (row >= p.out_dim || b0 >= p.batch_b) return;
    uint bt = min(8u, p.batch_b - b0); uint n4 = p.in_dim >> 2;
    const device float4* wp = (const device float4*)(w + (ulong)row * p.in_dim);
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (uint i = lane; i < n4; i += 32u) {
        float4 wf = wp[i];
        for (uint t = 0; t < bt; t++) acc[t] += dot(wf, ((const device float4*)(x + (ulong)(b0 + t) * p.in_dim))[i]);
    }
    for (uint t = 0; t < bt; t++) {
        float res = simd_sum(acc[t]);
        if (lane == 0u) { ulong idx = (ulong)(b0 + t) * p.out_dim + row; y[idx] = p.add_res ? (y[idx] + res) : res; }
    }
}

struct MlaSplitArgs { uint n_heads; uint nope_d; uint rope_d; uint v_d; uint d_head; uint batch_b; };
kernel void moe_mla_split_kv(
    const device float* kv_b_out [[buffer(0)]], const device float* k_pe [[buffer(1)]],
    device float* k_out [[buffer(2)]], device float* v_out [[buffer(3)]],
    constant MlaSplitArgs& p [[buffer(4)]], uint3 gid [[thread_position_in_grid]]
) {
    uint d = gid.x, h = gid.y, b = gid.z;
    if (d >= p.d_head || h >= p.n_heads || b >= p.batch_b) return;
    ulong kb_base = ((ulong)b * p.n_heads + h) * (p.nope_d + p.v_d);
    ulong out_base = ((ulong)b * p.n_heads + h) * p.d_head;
    if (d < p.nope_d) k_out[out_base + d] = kv_b_out[kb_base + d];
    else if (d < p.nope_d + p.rope_d) k_out[out_base + d] = k_pe[(ulong)b * p.rope_d + (d - p.nope_d)];
    else k_out[out_base + d] = 0.0f;
    v_out[out_base + d] = (d < p.v_d) ? kv_b_out[kb_base + p.nope_d + d] : 0.0f;
}

// =========================================================================
// True Matrix-Absorbed Compressed Latent MLA KV-Cache (`c_kv || k_pe`: -8x to -85x KV VRAM!)
// =========================================================================
struct MlaRopeArgs { uint n_heads; uint nope_d; uint rope_d; uint d_head; uint pos; float theta; uint rope_neox; uint has_freqs; };

// Rotates ONLY the positional slice `q[h, nope_d..d_head-1]` and shared MQA `k_pe[0..rope_d-1]`!
kernel void moe_mla_rope_pe(
    device float* q [[buffer(0)]], device float* k_pe [[buffer(1)]], const device float* rope_freqs [[buffer(2)]],
    constant MlaRopeArgs& p [[buffer(3)]], uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x, h = gid.y, half_r = p.rope_d >> 1;
    if (i >= half_r || h >= p.n_heads) return;
    uint idx0 = p.rope_neox ? i : (2u * i);
    uint idx1 = p.rope_neox ? (i + half_r) : (2u * i + 1u);
    float inv_f = 1.0f / pow(p.theta, float(2u * i) / float(p.rope_d));
    if (p.has_freqs) inv_f /= rope_freqs[i];
    float ang = float(p.pos) * inv_f, cs = cos(ang), sn = sin(ang);
    uint qbase = h * p.d_head + p.nope_d;
    float q0 = q[qbase + idx0], q1 = q[qbase + idx1];
    q[qbase + idx0] = q0 * cs - q1 * sn;
    q[qbase + idx1] = q0 * sn + q1 * cs;
    if (h == 0u) {
        float k0 = k_pe[idx0], k1 = k_pe[idx1];
        k_pe[idx0] = k0 * cs - k1 * sn;
        k_pe[idx1] = k0 * sn + k1 * cs;
    }
}

struct MlaStoreArgs { uint kv_r; uint rope_d; uint pos; };
kernel void moe_mla_store_latent(
    const device float* c_kv_and_pe [[buffer(0)]], device half* lat_cache [[buffer(1)]],
    constant MlaStoreArgs& p [[buffer(2)]], uint gid [[thread_position_in_grid]]
) {
    uint dim = p.kv_r + p.rope_d;
    if (gid < dim) lat_cache[(ulong)p.pos * dim + gid] = half(c_kv_and_pe[gid]);
}

struct MlaUnpackArgs { uint n_heads; uint nope_d; uint v_d; uint kv_r; uint gtype; };
// One-time load-time unpack of wkv_b[n_heads*(nope_d+v_d), kv_r] -> coalesced FP16 W_UK^T[n_heads, kv_r, nope_d] & W_UV[n_heads, v_d, kv_r]!
kernel void moe_mla_unpack_wkv_b(
    const device uchar* wkv_b [[buffer(0)]], device half* wuk_t [[buffer(1)]], device half* wuv [[buffer(2)]],
    constant MlaUnpackArgs& p [[buffer(3)]], uint3 gid [[thread_position_in_grid]]
) {
    uint r = gid.x, j = gid.y, h = gid.z;
    if (r >= p.kv_r || h >= p.n_heads) return;
    ulong h_row0 = (ulong)h * (p.nope_d + p.v_d);
    if (j < p.nope_d) {
        float v = dequant_any_elem(wkv_b, h_row0 + j, r, p.kv_r, p.gtype);
        wuk_t[((ulong)h * p.kv_r + r) * p.nope_d + j] = half(v);
    }
    if (j < p.v_d) {
        float v = dequant_any_elem(wkv_b, h_row0 + p.nope_d + j, r, p.kv_r, p.gtype);
        wuv[((ulong)h * p.v_d + j) * p.kv_r + r] = half(v);
    }
}

struct MlaAbsorbArgs { uint n_heads; uint nope_d; uint kv_r; uint d_head; };
kernel void moe_mla_absorb_q(
    const device half* wuk_t [[buffer(0)]], const device float* q [[buffer(1)]], device float* q_abs [[buffer(2)]],
    constant MlaAbsorbArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r = tgid.x, h = tgid.y;
    if (r >= p.kv_r || h >= p.n_heads) return;
    const device half* wp = wuk_t + ((ulong)h * p.kv_r + r) * p.nope_d;
    const device float* qp = q + (ulong)h * p.d_head;
    float acc = 0.0f;
    for (uint j = lane; j < p.nope_d; j += 32u) acc = fma(float(wp[j]), qp[j], acc);
    acc = simd_sum(acc);
    if (lane == 0u) q_abs[(ulong)h * p.kv_r + r] = acc;
}

struct MlaFlashArgs { uint n_heads; uint kv_r; uint nope_d; uint rope_d; uint d_head; uint kv_len; float scale; };
kernel void moe_flash_mla_latent(
    const device float* q_abs [[buffer(0)]], const device float* q [[buffer(1)]],
    const device half* lat_cache [[buffer(2)]], device float* o_lat [[buffer(3)]],
    constant MlaFlashArgs& p [[buffer(4)]],
    uint h [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    if (h >= p.n_heads) return;
    uint lat_dim = p.kv_r + p.rope_d;
    float scale = p.scale;
    threadgroup float s_qa[512];
    threadgroup float s_qp[128];
    threadgroup float s_m[4];
    threadgroup float s_l[4];
    threadgroup float s_o[4][512];
    for (uint r = lid; r < p.kv_r; r += 128u) s_qa[r] = q_abs[(ulong)h * p.kv_r + r];
    for (uint d = lid; d < p.rope_d; d += 128u) s_qp[d] = q[(ulong)h * p.d_head + p.nope_d + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_run = -1e30f, l_run = 0.0f;
    float o_acc[16];
    uint rpl = (p.kv_r + 31u) >> 5;
    for (uint k = 0; k < rpl; k++) o_acc[k] = 0.0f;
    uint dpl = (p.rope_d + 31u) >> 5;

    for (uint t = sg; t < p.kv_len; t += 4u) {
        const device half* cp = lat_cache + (ulong)t * lat_dim;
        const device half* kp = cp + p.kv_r;
        float dot = 0.0f;
        for (uint k = 0; k < rpl; k++) { uint r = lane + k * 32u; if (r < p.kv_r) dot = fma(s_qa[r], float(cp[r]), dot); }
        for (uint k = 0; k < dpl; k++) { uint d = lane + k * 32u; if (d < p.rope_d) dot = fma(s_qp[d], float(kp[d]), dot); }
        float sc = simd_sum(dot) * scale;
        float mn = max(m_run, sc), al = exp(m_run - mn), wt = exp(sc - mn);
        l_run = fma(l_run, al, wt); m_run = mn;
        for (uint k = 0; k < rpl; k++) { uint r = lane + k * 32u; if (r < p.kv_r) o_acc[k] = fma(o_acc[k], al, wt * float(cp[r])); }
    }
    if (lane == 0u) { s_m[sg] = m_run; s_l[sg] = l_run; }
    for (uint k = 0; k < rpl; k++) { uint r = lane + k * 32u; if (r < p.kv_r) s_o[sg][r] = o_acc[k]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = max(max(s_m[0], s_m[1]), max(s_m[2], s_m[3])), gl = 0.0f;
    for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * exp(s_m[s] - gm);
    float inv = 1.0f / max(gl, 1e-12f);
    for (uint r = lid; r < p.kv_r; r += 128u) {
        float ov = 0.0f;
        for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fma(s_o[s][r], exp(s_m[s] - gm), ov);
        o_lat[(ulong)h * p.kv_r + r] = ov * inv;
    }
}

struct MlaProjVArgs { uint n_heads; uint v_d; uint kv_r; uint out_stride; };
kernel void moe_mla_proj_v(
    const device half* wuv [[buffer(0)]], const device float* o_lat [[buffer(1)]], device float* ctx [[buffer(2)]],
    constant MlaProjVArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    uint j = tgid.x, h = tgid.y;
    if (j >= p.out_stride || h >= p.n_heads) return;
    if (j >= p.v_d) { if (lane == 0u) ctx[(ulong)h * p.out_stride + j] = 0.0f; return; }
    const device half* wp = wuv + ((ulong)h * p.v_d + j) * p.kv_r;
    const device float* op = o_lat + (ulong)h * p.kv_r;
    float acc = 0.0f;
    for (uint r = lane; r < p.kv_r; r += 32u) acc = fma(float(wp[r]), op[r], acc);
    acc = simd_sum(acc);
    if (lane == 0u) ctx[(ulong)h * p.out_stride + j] = acc;
}

// =========================================================================
// Tenon v6: PagedAttention 16-Token Block-Table Kernels (Continuous Batching!)
// =========================================================================
struct MoePagedRopeArgs {
    uint n_heads; uint n_kv; uint d_head; uint rope_dim;
    float theta; uint rope_neox; uint has_freqs; uint batch_b; uint max_blks;
};

kernel void moe_rope_kv_paged(
    device float* q [[buffer(0)]], device float* k [[buffer(1)]], const device float* v [[buffer(2)]],
    device half* k_cache [[buffer(3)]], device half* v_cache [[buffer(4)]],
    const device float* rope_freqs [[buffer(5)]],
    const device uint* pos_arr [[buffer(6)]], const device uint* slot_arr [[buffer(7)]],
    const device int* block_table [[buffer(8)]], constant MoePagedRopeArgs& p [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint i = gid.x, h = gid.y, b = gid.z;
    uint half_d = p.d_head >> 1, half_r = p.rope_dim >> 1;
    if (i >= half_d || h >= p.n_heads || b >= p.batch_b) return;
    uint pos = pos_arr[b], slot = slot_arr[b];
    uint pb = uint(block_table[(ulong)slot * p.max_blks + (pos >> 4)]);
    uint phys_pos = (pb << 4) | (pos & 15u);
    uint qoff = b * (p.n_heads * p.d_head) + h * p.d_head;
    uint koff = b * (p.n_kv * p.d_head) + h * p.d_head;
    if (i < half_r) {
        uint idx0 = p.rope_neox ? i : (2u * i);
        uint idx1 = p.rope_neox ? (i + half_r) : (2u * i + 1u);
        float inv_f = 1.0f / pow(p.theta, float(2u * i) / float(p.rope_dim));
        if (p.has_freqs) inv_f /= rope_freqs[i];
        float ang = float(pos) * inv_f, cs = cos(ang), sn = sin(ang);
        float q0 = q[qoff + idx0], q1 = q[qoff + idx1];
        q[qoff + idx0] = q0 * cs - q1 * sn; q[qoff + idx1] = q0 * sn + q1 * cs;
        if (h < p.n_kv) {
            float k0 = k[koff + idx0], k1 = k[koff + idx1];
            float rk0 = k0 * cs - k1 * sn, rk1 = k0 * sn + k1 * cs;
            k[koff + idx0] = rk0; k[koff + idx1] = rk1;
            ulong cbase = (ulong)phys_pos * (p.n_kv * p.d_head) + h * p.d_head;
            k_cache[cbase + idx0] = half(rk0); k_cache[cbase + idx1] = half(rk1);
            v_cache[cbase + idx0] = half(v[koff + idx0]); v_cache[cbase + idx1] = half(v[koff + idx1]);
        }
    } else if (h < p.n_kv) {
        uint idx0 = 2u * i, idx1 = 2u * i + 1u;
        ulong cbase = (ulong)phys_pos * (p.n_kv * p.d_head) + h * p.d_head;
        k_cache[cbase + idx0] = half(k[koff + idx0]); k_cache[cbase + idx1] = half(k[koff + idx1]);
        v_cache[cbase + idx0] = half(v[koff + idx0]); v_cache[cbase + idx1] = half(v[koff + idx1]);
    }
}

kernel void moe_rope_kv_q8_paged(
    device float* q [[buffer(0)]], device float* k [[buffer(1)]], const device float* v [[buffer(2)]],
    device uchar* k_cache [[buffer(3)]], device uchar* v_cache [[buffer(4)]],
    const device float* rope_freqs [[buffer(5)]],
    const device uint* pos_arr [[buffer(6)]], const device uint* slot_arr [[buffer(7)]],
    const device int* block_table [[buffer(8)]], constant MoePagedRopeArgs& p [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    uint h = tgid.x, b = tgid.y;
    if (h >= p.n_heads || b >= p.batch_b) return;
    uint pos = pos_arr[b], slot = slot_arr[b];
    uint pb = uint(block_table[(ulong)slot * p.max_blks + (pos >> 4)]);
    uint phys_pos = (pb << 4) | (pos & 15u);
    uint half_r = p.rope_dim >> 1;
    uint qbase = b * (p.n_heads * p.d_head) + h * p.d_head;
    uint kbase = b * (p.n_kv * p.d_head) + h * p.d_head;
    uint lid = sg * 32u + lane;
    for (uint i = lid; i < half_r; i += 128u) {
        uint idx0 = p.rope_neox ? i : (2u * i);
        uint idx1 = p.rope_neox ? (i + half_r) : (2u * i + 1u);
        float inv_f = 1.0f / pow(p.theta, float(2u * i) / float(p.rope_dim));
        if (p.has_freqs) inv_f /= rope_freqs[i];
        float ang = float(pos) * inv_f, cs = cos(ang), sn = sin(ang);
        float q0 = q[qbase + idx0], q1 = q[qbase + idx1];
        q[qbase + idx0] = q0 * cs - q1 * sn; q[qbase + idx1] = q0 * sn + q1 * cs;
        if (h < p.n_kv) {
            float k0 = k[kbase + idx0], k1 = k[kbase + idx1];
            k[kbase + idx0] = k0 * cs - k1 * sn; k[kbase + idx1] = k0 * sn + k1 * cs;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (h < p.n_kv) {
        uint nblk = p.d_head >> 5;
        ulong row_bytes = (ulong)p.n_kv * (ulong)nblk * 34ul;
        ulong h_off = (ulong)phys_pos * row_bytes + (ulong)h * (ulong)nblk * 34ul;
        for (uint blk = sg; blk < nblk; blk += 4u) {
            float kv = k[kbase + blk * 32u + lane], vv = v[kbase + blk * 32u + lane];
            float kam = simd_max(fabs(kv)), vam = simd_max(fabs(vv));
            float kd = kam / 127.0f, kid = (kam > 0.0f) ? (127.0f / kam) : 0.0f;
            float vd = vam / 127.0f, vid = (vam > 0.0f) ? (127.0f / vam) : 0.0f;
            device uchar* kp = k_cache + h_off + (ulong)blk * 34ul;
            device uchar* vp = v_cache + h_off + (ulong)blk * 34ul;
            if (lane == 0u) { *(device half*)kp = half(kd); *(device half*)vp = half(vd); }
            ((device char*)(kp + 2u))[lane] = char(rint(kv * kid));
            ((device char*)(vp + 2u))[lane] = char(rint(vv * vid));
        }
    }
}

struct MoePagedAttnArgs { uint n_heads; uint n_kv; uint d_head; uint batch_b; uint out_stride; float scale; uint max_blks; };

kernel void moe_flash_decode_paged(
    const device float* q [[buffer(0)]], const device half* k_cache [[buffer(1)]], const device half* v_cache [[buffer(2)]],
    device float* ctx [[buffer(3)]], const device uint* pos_arr [[buffer(4)]], const device uint* slot_arr [[buffer(5)]],
    const device int* block_table [[buffer(6)]], constant MoePagedAttnArgs& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x, b = tgid.y;
    if (h >= p.n_heads || b >= p.batch_b) return;
    uint kv_len = pos_arr[b] + 1u, slot = slot_arr[b];
    const device int* bt = block_table + (ulong)slot * p.max_blks;
    uint grp = max(p.n_heads / p.n_kv, 1u), kvh = h / grp;
    uint kv_stride = p.n_kv * p.d_head;

    threadgroup float s_q[256]; threadgroup float s_m[4]; threadgroup float s_l[4]; threadgroup float s_o[4][256];
    ulong qbase = (ulong)b * (p.n_heads * p.d_head) + (ulong)h * p.d_head;
    ulong cbase = (ulong)b * (p.n_heads * p.out_stride) + (ulong)h * p.out_stride;
    for (uint d = lid; d < p.d_head; d += 128u) s_q[d] = q[qbase + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_run = -1e30f, l_run = 0.0f;
    float o_acc[8] = {0,0,0,0,0,0,0,0};
    uint d_per_lane = (p.d_head + 31u) >> 5;
    for (uint t = sg; t < kv_len; t += 4u) {
        uint pt = (uint(bt[t >> 4]) << 4) | (t & 15u);
        const device half* kp = k_cache + (ulong)pt * kv_stride + kvh * p.d_head;
        const device half* vp = v_cache + (ulong)pt * kv_stride + kvh * p.d_head;
        float dot = 0.0f;
        for (uint k = 0; k < d_per_lane; k++) { uint d = lane + k * 32u; if (d < p.d_head) dot = fma(s_q[d], float(kp[d]), dot); }
        float sc = simd_sum(dot) * p.scale;
        float mn = max(m_run, sc), al = exp(m_run - mn), wt = exp(sc - mn);
        l_run = fma(l_run, al, wt); m_run = mn;
        for (uint k = 0; k < d_per_lane; k++) { uint d = lane + k * 32u; if (d < p.d_head) o_acc[k] = fma(o_acc[k], al, wt * float(vp[d])); }
    }
    if (lane == 0u) { s_m[sg] = m_run; s_l[sg] = l_run; }
    for (uint k = 0; k < d_per_lane; k++) { uint d = lane + k * 32u; if (d < p.d_head) s_o[sg][d] = o_acc[k]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = max(max(s_m[0], s_m[1]), max(s_m[2], s_m[3])), gl = 0.0f;
    for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * exp(s_m[s] - gm);
    float inv = 1.0f / max(gl, 1e-12f);
    for (uint d = lid; d < p.out_stride; d += 128u) {
        float ov = 0.0f;
        for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fma(s_o[s][d], exp(s_m[s] - gm), ov);
        ctx[cbase + d] = ov * inv;
    }
}

kernel void moe_flash_decode_q8_paged(
    const device float* q [[buffer(0)]], const device uchar* k_cache [[buffer(1)]], const device uchar* v_cache [[buffer(2)]],
    device float* ctx [[buffer(3)]], const device uint* pos_arr [[buffer(4)]], const device uint* slot_arr [[buffer(5)]],
    const device int* block_table [[buffer(6)]], constant MoePagedAttnArgs& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x, b = tgid.y;
    if (h >= p.n_heads || b >= p.batch_b) return;
    uint kv_len = pos_arr[b] + 1u, slot = slot_arr[b];
    const device int* bt = block_table + (ulong)slot * p.max_blks;
    uint grp = max(p.n_heads / p.n_kv, 1u), kvh = h / grp;
    uint nblk = p.d_head >> 5;
    ulong row_bytes = (ulong)p.n_kv * (ulong)nblk * 34ul;

    threadgroup float s_q[256]; threadgroup float s_m[4]; threadgroup float s_l[4]; threadgroup float s_o[4][256];
    ulong qbase = (ulong)b * (p.n_heads * p.d_head) + (ulong)h * p.d_head;
    ulong cbase = (ulong)b * (p.n_heads * p.out_stride) + (ulong)h * p.out_stride;
    for (uint d = lid; d < p.d_head; d += 128u) s_q[d] = q[qbase + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_run = -1e30f, l_run = 0.0f;
    float o_acc[8] = {0,0,0,0,0,0,0,0};
    for (uint t = sg; t < kv_len; t += 4u) {
        uint pt = (uint(bt[t >> 4]) << 4) | (t & 15u);
        ulong h_off = (ulong)pt * row_bytes + (ulong)kvh * (ulong)nblk * 34ul;
        const device uchar* kp = k_cache + h_off;
        const device uchar* vp = v_cache + h_off;
        float dot = 0.0f;
        for (uint blk = 0; blk < nblk; blk++) {
            const device uchar* kb = kp + (ulong)blk * 34ul;
            float kd = float(*(const device half*)kb);
            dot = fma(s_q[blk * 32u + lane], float(((const device char*)(kb + 2u))[lane]) * kd, dot);
        }
        float sc = simd_sum(dot) * p.scale;
        float mn = max(m_run, sc), al = exp(m_run - mn), wt = exp(sc - mn);
        l_run = fma(l_run, al, wt); m_run = mn;
        for (uint blk = 0; blk < nblk; blk++) {
            const device uchar* vb = vp + (ulong)blk * 34ul;
            float vd = float(*(const device half*)vb);
            o_acc[blk] = fma(o_acc[blk], al, wt * float(((const device char*)(vb + 2u))[lane]) * vd);
        }
    }
    if (lane == 0u) { s_m[sg] = m_run; s_l[sg] = l_run; }
    for (uint blk = 0; blk < nblk; blk++) s_o[sg][blk * 32u + lane] = o_acc[blk];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = max(max(s_m[0], s_m[1]), max(s_m[2], s_m[3])), gl = 0.0f;
    for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * exp(s_m[s] - gm);
    float inv = 1.0f / max(gl, 1e-12f);
    for (uint d = lid; d < p.out_stride; d += 128u) {
        float ov = 0.0f;
        for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fma(s_o[s][d], exp(s_m[s] - gm), ov);
        ctx[cbase + d] = ov * inv;
    }
}

// =========================================================================
// Tenon v6: Tree-Attention Speculative Verification Kernels (Bitmask Tree!)
// =========================================================================
kernel void moe_rope_kv_tree(
    device float* q [[buffer(0)]], device float* k [[buffer(1)]], const device float* v [[buffer(2)]],
    device half* k_cache [[buffer(3)]], device half* v_cache [[buffer(4)]],
    const device float* rope_freqs [[buffer(5)]], const device uint* tree_depth [[buffer(6)]],
    constant MoeRopeBatchArgs& p [[buffer(7)]], uint3 gid [[thread_position_in_grid]]
) {
    uint i = gid.x, h = gid.y, b = gid.z;
    uint half_d = p.d_head >> 1, half_r = p.rope_dim >> 1;
    if (i >= half_d || h >= p.n_heads || b >= p.batch_b) return;
    uint rot_pos = p.start_pos + tree_depth[b];
    uint wr_pos  = p.start_pos + b;
    uint qoff = b * (p.n_heads * p.d_head) + h * p.d_head;
    uint koff = b * (p.n_kv * p.d_head) + h * p.d_head;
    if (i < half_r) {
        uint idx0 = p.rope_neox ? i : (2u * i);
        uint idx1 = p.rope_neox ? (i + half_r) : (2u * i + 1u);
        float inv_f = 1.0f / pow(p.theta, float(2u * i) / float(p.rope_dim));
        if (p.has_freqs) inv_f /= rope_freqs[i];
        float ang = float(rot_pos) * inv_f, cs = cos(ang), sn = sin(ang);
        float q0 = q[qoff + idx0], q1 = q[qoff + idx1];
        q[qoff + idx0] = q0 * cs - q1 * sn; q[qoff + idx1] = q0 * sn + q1 * cs;
        if (h < p.n_kv) {
            float k0 = k[koff + idx0], k1 = k[koff + idx1];
            float rk0 = k0 * cs - k1 * sn, rk1 = k0 * sn + k1 * cs;
            k[koff + idx0] = rk0; k[koff + idx1] = rk1;
            ulong cbase = (ulong)wr_pos * (p.n_kv * p.d_head) + h * p.d_head;
            k_cache[cbase + idx0] = half(rk0); k_cache[cbase + idx1] = half(rk1);
            v_cache[cbase + idx0] = half(v[koff + idx0]); v_cache[cbase + idx1] = half(v[koff + idx1]);
        }
    } else if (h < p.n_kv) {
        uint idx0 = 2u * i, idx1 = 2u * i + 1u;
        ulong cbase = (ulong)wr_pos * (p.n_kv * p.d_head) + h * p.d_head;
        k_cache[cbase + idx0] = half(k[koff + idx0]); k_cache[cbase + idx1] = half(k[koff + idx1]);
        v_cache[cbase + idx0] = half(v[koff + idx0]); v_cache[cbase + idx1] = half(v[koff + idx1]);
    }
}

kernel void moe_rope_kv_q8_tree(
    device float* q [[buffer(0)]], device float* k [[buffer(1)]], const device float* v [[buffer(2)]],
    device uchar* k_cache [[buffer(3)]], device uchar* v_cache [[buffer(4)]],
    const device float* rope_freqs [[buffer(5)]], const device uint* tree_depth [[buffer(6)]],
    constant MoeRopeBatchArgs& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    uint h = tgid.x, b = tgid.y;
    if (h >= p.n_heads || b >= p.batch_b) return;
    uint rot_pos = p.start_pos + tree_depth[b];
    uint wr_pos  = p.start_pos + b;
    uint half_r = p.rope_dim >> 1;
    uint qbase = b * (p.n_heads * p.d_head) + h * p.d_head;
    uint kbase = b * (p.n_kv * p.d_head) + h * p.d_head;
    uint lid = sg * 32u + lane;
    for (uint i = lid; i < half_r; i += 128u) {
        uint idx0 = p.rope_neox ? i : (2u * i);
        uint idx1 = p.rope_neox ? (i + half_r) : (2u * i + 1u);
        float inv_f = 1.0f / pow(p.theta, float(2u * i) / float(p.rope_dim));
        if (p.has_freqs) inv_f /= rope_freqs[i];
        float ang = float(rot_pos) * inv_f, cs = cos(ang), sn = sin(ang);
        float q0 = q[qbase + idx0], q1 = q[qbase + idx1];
        q[qbase + idx0] = q0 * cs - q1 * sn; q[qbase + idx1] = q0 * sn + q1 * cs;
        if (h < p.n_kv) {
            float k0 = k[kbase + idx0], k1 = k[kbase + idx1];
            k[kbase + idx0] = k0 * cs - k1 * sn; k[kbase + idx1] = k0 * sn + k1 * cs;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (h < p.n_kv) {
        uint nblk = p.d_head >> 5;
        ulong row_bytes = (ulong)p.n_kv * (ulong)nblk * 34ul;
        ulong h_off = (ulong)wr_pos * row_bytes + (ulong)h * (ulong)nblk * 34ul;
        for (uint blk = sg; blk < nblk; blk += 4u) {
            float kv = k[kbase + blk * 32u + lane], vv = v[kbase + blk * 32u + lane];
            float kam = simd_max(fabs(kv)), vam = simd_max(fabs(vv));
            float kd = kam / 127.0f, kid = (kam > 0.0f) ? (127.0f / kam) : 0.0f;
            float vd = vam / 127.0f, vid = (vam > 0.0f) ? (127.0f / vam) : 0.0f;
            device uchar* kp = k_cache + h_off + (ulong)blk * 34ul;
            device uchar* vp = v_cache + h_off + (ulong)blk * 34ul;
            if (lane == 0u) { *(device half*)kp = half(kd); *(device half*)vp = half(vd); }
            ((device char*)(kp + 2u))[lane] = char(rint(kv * kid));
            ((device char*)(vp + 2u))[lane] = char(rint(vv * vid));
        }
    }
}

kernel void moe_flash_tree_verify(
    const device float* q [[buffer(0)]], const device half* k_cache [[buffer(1)]], const device half* v_cache [[buffer(2)]],
    device float* ctx [[buffer(3)]], const device ulong* tree_mask [[buffer(4)]], constant MoeAttnBatchArgs& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x, b = tgid.y;
    if (h >= p.n_heads || b >= p.batch_b) return;
    ulong mask = tree_mask[b];
    uint total_kv = p.start_pos + b + 1u;
    uint grp = max(p.n_heads / p.n_kv, 1u), kvh = h / grp;
    uint kv_stride = p.n_kv * p.d_head;

    threadgroup float s_q[256]; threadgroup float s_m[4]; threadgroup float s_l[4]; threadgroup float s_o[4][256];
    ulong qbase = (ulong)b * (p.n_heads * p.d_head) + (ulong)h * p.d_head;
    ulong cbase = (ulong)b * (p.n_heads * p.out_stride) + (ulong)h * p.out_stride;
    for (uint d = lid; d < p.d_head; d += 128u) s_q[d] = q[qbase + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_run = -1e30f, l_run = 0.0f;
    float o_acc[8] = {0,0,0,0,0,0,0,0};
    uint d_per_lane = (p.d_head + 31u) >> 5;
    for (uint t = sg; t < total_kv; t += 4u) {
        if (t >= p.start_pos) {
            uint j = t - p.start_pos;
            if (((mask >> j) & 1ul) == 0ul) continue;
        }
        const device half* kp = k_cache + (ulong)t * kv_stride + kvh * p.d_head;
        const device half* vp = v_cache + (ulong)t * kv_stride + kvh * p.d_head;
        float dot = 0.0f;
        for (uint k = 0; k < d_per_lane; k++) { uint d = lane + k * 32u; if (d < p.d_head) dot = fma(s_q[d], float(kp[d]), dot); }
        float sc = simd_sum(dot) * p.scale;
        float mn = max(m_run, sc), al = exp(m_run - mn), wt = exp(sc - mn);
        l_run = fma(l_run, al, wt); m_run = mn;
        for (uint k = 0; k < d_per_lane; k++) { uint d = lane + k * 32u; if (d < p.d_head) o_acc[k] = fma(o_acc[k], al, wt * float(vp[d])); }
    }
    if (lane == 0u) { s_m[sg] = m_run; s_l[sg] = l_run; }
    for (uint k = 0; k < d_per_lane; k++) { uint d = lane + k * 32u; if (d < p.d_head) s_o[sg][d] = o_acc[k]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = max(max(s_m[0], s_m[1]), max(s_m[2], s_m[3])), gl = 0.0f;
    for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * exp(s_m[s] - gm);
    float inv = 1.0f / max(gl, 1e-12f);
    for (uint d = lid; d < p.out_stride; d += 128u) {
        float ov = 0.0f;
        for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fma(s_o[s][d], exp(s_m[s] - gm), ov);
        ctx[cbase + d] = ov * inv;
    }
}

kernel void moe_flash_tree_verify_q8(
    const device float* q [[buffer(0)]], const device uchar* k_cache [[buffer(1)]], const device uchar* v_cache [[buffer(2)]],
    device float* ctx [[buffer(3)]], const device ulong* tree_mask [[buffer(4)]], constant MoeAttnBatchArgs& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x, b = tgid.y;
    if (h >= p.n_heads || b >= p.batch_b) return;
    ulong mask = tree_mask[b];
    uint total_kv = p.start_pos + b + 1u;
    uint grp = max(p.n_heads / p.n_kv, 1u), kvh = h / grp;
    uint nblk = p.d_head >> 5;
    ulong row_bytes = (ulong)p.n_kv * (ulong)nblk * 34ul;

    threadgroup float s_q[256]; threadgroup float s_m[4]; threadgroup float s_l[4]; threadgroup float s_o[4][256];
    ulong qbase = (ulong)b * (p.n_heads * p.d_head) + (ulong)h * p.d_head;
    ulong cbase = (ulong)b * (p.n_heads * p.out_stride) + (ulong)h * p.out_stride;
    for (uint d = lid; d < p.d_head; d += 128u) s_q[d] = q[qbase + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m_run = -1e30f, l_run = 0.0f;
    float o_acc[8] = {0,0,0,0,0,0,0,0};
    for (uint t = sg; t < total_kv; t += 4u) {
        if (t >= p.start_pos) {
            uint j = t - p.start_pos;
            if (((mask >> j) & 1ul) == 0ul) continue;
        }
        ulong h_off = (ulong)t * row_bytes + (ulong)kvh * (ulong)nblk * 34ul;
        const device uchar* kp = k_cache + h_off;
        const device uchar* vp = v_cache + h_off;
        float dot = 0.0f;
        for (uint blk = 0; blk < nblk; blk++) {
            const device uchar* kb = kp + (ulong)blk * 34ul;
            float kd = float(*(const device half*)kb);
            dot = fma(s_q[blk * 32u + lane], float(((const device char*)(kb + 2u))[lane]) * kd, dot);
        }
        float sc = simd_sum(dot) * p.scale;
        float mn = max(m_run, sc), al = exp(m_run - mn), wt = exp(sc - mn);
        l_run = fma(l_run, al, wt); m_run = mn;
        for (uint blk = 0; blk < nblk; blk++) {
            const device uchar* vb = vp + (ulong)blk * 34ul;
            float vd = float(*(const device half*)vb);
            o_acc[blk] = fma(o_acc[blk], al, wt * float(((const device char*)(vb + 2u))[lane]) * vd);
        }
    }
    if (lane == 0u) { s_m[sg] = m_run; s_l[sg] = l_run; }
    for (uint blk = 0; blk < nblk; blk++) s_o[sg][blk * 32u + lane] = o_acc[blk];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = max(max(s_m[0], s_m[1]), max(s_m[2], s_m[3])), gl = 0.0f;
    for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * exp(s_m[s] - gm);
    float inv = 1.0f / max(gl, 1e-12f);
    for (uint d = lid; d < p.out_stride; d += 128u) {
        float ov = 0.0f;
        for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fma(s_o[s][d], exp(s_m[s] - gm), ov);
        ctx[cbase + d] = ov * inv;
    }
}

// =========================================================================
// Tenon v6: Native Multimodal ViT (`mmproj` CLIP/SigLIP/SmolVLM) GPU Kernels
// =========================================================================
struct VitLnArgs { uint dim; uint n_tok; float eps; uint has_bias; };
kernel void vit_layernorm(
    const device float* x [[buffer(0)]], const device float* w [[buffer(1)]], const device float* b [[buffer(2)]],
    device float* out [[buffer(3)]], constant VitLnArgs& p [[buffer(4)]],
    uint g [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    if (g >= p.n_tok) return;
    threadgroup float sm[8]; threadgroup float sv[8];
    ulong base = (ulong)g * p.dim;
    float s = 0.0f;
    for (uint i = lid; i < p.dim; i += 256u) s += x[base + i];
    s = simd_sum(s);
    if (lane == 0u) sm[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = 0.0f; for (uint k = 0; k < 8u; k++) mean += sm[k];
    mean /= float(p.dim);
    float v = 0.0f;
    for (uint i = lid; i < p.dim; i += 256u) { float d = x[base + i] - mean; v = fma(d, d, v); }
    v = simd_sum(v);
    if (lane == 0u) sv[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = 0.0f; for (uint k = 0; k < 8u; k++) var += sv[k];
    float inv = rsqrt(var / float(p.dim) + p.eps);
    for (uint i = lid; i < p.dim; i += 256u) {
        float y = (x[base + i] - mean) * inv * w[i];
        out[base + i] = p.has_bias ? (y + b[i]) : y;
    }
}

struct VitGeluArgs { uint n; uint use_tanh; };
kernel void vit_gelu(device float* x [[buffer(0)]], constant VitGeluArgs& p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n) return;
    float v = x[gid];
    if (p.use_tanh) {
        if (v > 8.0f) { x[gid] = v; return; }
        if (v < -8.0f) { x[gid] = 0.0f; return; }
        float c = clamp(0.7978845608f * (v + 0.044715f * v * v * v), -10.0f, 10.0f);
        x[gid] = 0.5f * v * (1.0f + tanh(c));
    } else {
        x[gid] = v / (1.0f + exp(-clamp(1.702f * v, -20.0f, 20.0f)));
    }
}

kernel void vit_add(device float* y [[buffer(0)]], const device float* x [[buffer(1)]], constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) y[gid] += x[gid];
}

struct VitAttnArgs { uint n_tok; uint n_heads; uint d_head; float scale; };
kernel void vit_full_attn(
    const device float* q [[buffer(0)]], const device float* k [[buffer(1)]], const device float* v [[buffer(2)]],
    device float* out [[buffer(3)]], constant VitAttnArgs& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x, qi = tgid.y;
    if (h >= p.n_heads || qi >= p.n_tok) return;
    uint dm = p.n_heads * p.d_head;
    threadgroup float s_q[128]; threadgroup float s_m[4]; threadgroup float s_l[4]; threadgroup float s_o[4][128];
    ulong qbase = (ulong)qi * dm + (ulong)h * p.d_head;
    for (uint d = lid; d < p.d_head; d += 128u) s_q[d] = q[qbase + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m_run = -1e30f, l_run = 0.0f;
    float o_acc[4] = {0,0,0,0};
    uint dpl = (p.d_head + 31u) >> 5;
    for (uint t = sg; t < p.n_tok; t += 4u) {
        const device float* kp = k + (ulong)t * dm + (ulong)h * p.d_head;
        const device float* vp = v + (ulong)t * dm + (ulong)h * p.d_head;
        float dot = 0.0f;
        for (uint c = 0; c < dpl; c++) { uint d = lane + c * 32u; if (d < p.d_head) dot = fma(s_q[d], kp[d], dot); }
        float sc = simd_sum(dot) * p.scale;
        float mn = max(m_run, sc), al = exp(m_run - mn), wt = exp(sc - mn);
        l_run = fma(l_run, al, wt); m_run = mn;
        for (uint c = 0; c < dpl; c++) { uint d = lane + c * 32u; if (d < p.d_head) o_acc[c] = fma(o_acc[c], al, wt * vp[d]); }
    }
    if (lane == 0u) { s_m[sg] = m_run; s_l[sg] = l_run; }
    for (uint c = 0; c < dpl; c++) { uint d = lane + c * 32u; if (d < p.d_head) s_o[sg][d] = o_acc[c]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = max(max(s_m[0], s_m[1]), max(s_m[2], s_m[3])), gl = 0.0f;
    for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * exp(s_m[s] - gm);
    float inv = 1.0f / max(gl, 1e-12f);
    for (uint d = lid; d < p.d_head; d += 128u) {
        float ov = 0.0f;
        for (uint s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fma(s_o[s][d], exp(s_m[s] - gm), ov);
        out[qbase + d] = ov * inv;
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
    g_pso_gemv_q5_0         = mk(@"moe_gemv_q5_0");
    g_pso_gemv_q5_1         = mk(@"moe_gemv_q5_1");
    g_pso_gemv_q8           = mk(@"moe_gemv_q8");
    g_pso_gemv_q2k          = mk(@"moe_gemv_q2k");
    g_pso_gemv_q3k          = mk(@"moe_gemv_q3k");
    g_pso_gemv_iq4xs        = mk(@"moe_gemv_iq4xs");
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
    g_pso_moe_down_q5_0     = mk(@"moe_topk_down_accum_q5_0");
    g_pso_moe_down_q5_1     = mk(@"moe_topk_down_accum_q5_1");
    g_pso_moe_down_q8       = mk(@"moe_topk_down_accum_q8");
    g_pso_moe_down_q6k      = mk(@"moe_topk_down_accum_q6k");
    g_pso_moe_slot_gu_q4    = mk(@"moe_slot_gate_up_q4");
    g_pso_moe_slot_down_q4  = mk(@"moe_slot_down_accum_q4");
    g_pso_moe_slot_down_q4_1= mk(@"moe_slot_down_accum_q4_1");
    g_pso_moe_slot_down_q5_0= mk(@"moe_slot_down_accum_q5_0");
    g_pso_moe_slot_down_q5_1= mk(@"moe_slot_down_accum_q5_1");
    g_pso_moe_slot_down_q8  = mk(@"moe_slot_down_accum_q8");
    g_pso_moe_slot_down_q6k = mk(@"moe_slot_down_accum_q6k");
    g_pso_moe_slot_gu_q4k   = mk(@"moe_slot_gate_up_q4k");
    g_pso_moe_slot_down_q4k = mk(@"moe_slot_down_accum_q4k");
    g_pso_moe_gu_q4k        = mk(@"moe_topk_gate_up_q4k");
    g_pso_moe_down_q4k      = mk(@"moe_topk_down_accum_q4k");
    g_pso_rope_kv_q8        = mk(@"moe_rope_and_kv_q8_batch");
    g_pso_flash_dec_q8      = mk(@"moe_flash_attn_q8");
    g_pso_gemm_q4_b         = mk(@"moe_gemm_q4_batch");
    g_pso_gemm_q4_sb        = mk(@"moe_gemm_q4_small_b");
    g_pso_gemm_q4_1_b       = mk(@"moe_gemm_q4_1_batch");
    g_pso_gemm_q5_0_b       = mk(@"moe_gemm_q5_0_batch");
    g_pso_gemm_q5_1_b       = mk(@"moe_gemm_q5_1_batch");
    g_pso_gemm_q8_b         = mk(@"moe_gemm_q8_batch");
    g_pso_gemm_q2k_b        = mk(@"moe_gemm_q2k_batch");
    g_pso_gemm_q3k_b        = mk(@"moe_gemm_q3k_batch");
    g_pso_gemm_iq4xs_b      = mk(@"moe_gemm_iq4xs_batch");
    g_pso_gemm_q4k_b        = mk(@"moe_gemm_q4k_batch");
    g_pso_gemm_q5k_b        = mk(@"moe_gemm_q5k_batch");
    g_pso_gemm_q6k_b        = mk(@"moe_gemm_q6k_batch");
    g_pso_gemm_f16_b        = mk(@"moe_gemm_f16_batch");
    g_pso_gemm_f32_b        = mk(@"moe_gemm_f32_batch");
    g_pso_add_bias_b        = mk(@"moe_add_bias_batch");
    g_pso_rope_kv_b         = mk(@"moe_rope_and_kv_batch");
    g_pso_rope_kv_q8_b      = g_pso_rope_kv_q8;
    g_pso_flash_prefill     = mk(@"moe_flash_prefill_fp16");
    g_pso_flash_prefill_q8  = g_pso_flash_dec_q8;
    g_pso_mla_split_kv      = mk(@"moe_mla_split_kv");
    g_pso_mla_rope_pe       = mk(@"moe_mla_rope_pe");
    g_pso_mla_store_lat     = mk(@"moe_mla_store_latent");
    g_pso_mla_unpack_kvb    = mk(@"moe_mla_unpack_wkv_b");
    g_pso_mla_absorb_q      = mk(@"moe_mla_absorb_q");
    g_pso_mla_flash_lat     = mk(@"moe_flash_mla_latent");
    g_pso_mla_proj_v        = mk(@"moe_mla_proj_v");
    g_pso_rope_kv_paged     = mk(@"moe_rope_kv_paged");
    g_pso_rope_kv_q8_paged  = mk(@"moe_rope_kv_q8_paged");
    g_pso_flash_dec_paged   = mk(@"moe_flash_decode_paged");
    g_pso_flash_dec_q8_paged= mk(@"moe_flash_decode_q8_paged");
    g_pso_rope_kv_tree      = mk(@"moe_rope_kv_tree");
    g_pso_rope_kv_q8_tree   = mk(@"moe_rope_kv_q8_tree");
    g_pso_flash_tree        = mk(@"moe_flash_tree_verify");
    g_pso_flash_tree_q8     = mk(@"moe_flash_tree_verify_q8");
    g_pso_vit_ln            = mk(@"vit_layernorm");
    g_pso_vit_gelu          = mk(@"vit_gelu");
    g_pso_vit_add           = mk(@"vit_add");
    g_pso_vit_attn          = mk(@"vit_full_attn");
    g_pso_argmax            = mk(@"moe_argmax_f32");
    return (g_pso_gemv_q4 != nil && g_pso_gemv_q5_0 != nil && g_pso_gemv_q2k != nil && g_pso_gemv_iq4xs != nil && g_pso_mla_flash_lat != nil && g_pso_gemm_q4_b != nil && g_pso_flash_dec_q8 != nil && g_pso_flash_dec_paged != nil && g_pso_flash_tree != nil && g_pso_vit_attn != nil);
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
        } else if (kv_ends_with(key, ".expert_group_count")) {
            m->n_expert_groups = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".expert_group_used_count")) {
            m->n_group_used = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".attention.kv_lora_rank")) {
            m->kv_lora_rank = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".attention.q_lora_rank")) {
            m->q_lora_rank = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".attention.value_length")) {
            m->v_head_dim = (int)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".rope.freq_base") && vt == GGUF_T_F32) {
            m->rope_theta = gr_f32(&gr);
        } else if (kv_ends_with(key, ".rope.scaling.factor") && vt == GGUF_T_F32) {
            m->yarn_factor = gr_f32(&gr);
        } else if (kv_ends_with(key, ".rope.scaling.original_context_length")) {
            m->yarn_orig_ctx = (float)gr_read_int(&gr, vt);
        } else if (kv_ends_with(key, ".rope.scaling.yarn_log_multiplier") && vt == GGUF_T_F32) {
            m->yarn_log_m = gr_f32(&gr);
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
        } else if (strcmp(key, "tokenizer.ggml.merges") == 0 && vt == GGUF_T_ARR) {
            uint32_t et = gr_u32(&gr);
            uint64_t n = gr_u64(&gr);
            if (et == GGUF_T_STR && n > 0 && n <= 1048576) {
                m->n_bpe_merges = (int)n;
                m->bpe_merges = (char**)calloc((size_t)n, sizeof(char*));
                for (uint64_t ti = 0; ti < n; ti++) m->bpe_merges[ti] = gr_str(&gr);
            } else {
                for (uint64_t ti = 0; ti < n; ti++) gr_skip_val(&gr, et);
            }
        } else {
            gr_skip_val(&gr, vt);
        }
        free(key);
    }

    if (is_primary && !norm_wt_set) {
        m->expert_weights_norm = (strcmp(m->arch, "olmoe") == 0 || strcmp(m->arch, "deepseek2") == 0) ? 0 : 1;
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
        memset((void*)m, 0, sizeof(MoeModel));
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
        if (m->v_head_dim <= 0) m->v_head_dim = m->d_head;

        // Dynamic Context Length (`TENON_MAX_CTX`, default 2048, up to 16384) & Q8_0 KV-Cache (`TENON_KV_QUANT=q8|fp16|auto`)!
        m->max_ctx = MOE_DEFAULT_SEQ;
        const char* ctx_env = getenv("TENON_MAX_CTX");
        if (ctx_env && atoi(ctx_env) > 0) {
            int cv = atoi(ctx_env);
            if (cv < 256) cv = 256;
            if (cv > MOE_MAX_SEQ) cv = MOE_MAX_SEQ;
            m->max_ctx = cv;
        }
        const char* kvq_env = getenv("TENON_KV_QUANT");
        if (kvq_env && (strcmp(kvq_env, "q8") == 0 || strcmp(kvq_env, "q8_0") == 0 || strcmp(kvq_env, "1") == 0)) {
            m->kv_quant_q8 = 1;
        } else if (kvq_env && (strcmp(kvq_env, "fp16") == 0 || strcmp(kvq_env, "f16") == 0 || strcmp(kvq_env, "0") == 0)) {
            m->kv_quant_q8 = 0;
        } else {
            m->kv_quant_q8 = (m->max_ctx > 2048) ? 1 : 0;
        }
        const char* mla_env = getenv("TENON_MLA_LATENT_KV");
        m->mla_latent_kv = (mla_env && atoi(mla_env) == 0) ? 0 : 1;
        const char* kvs_env = getenv("TENON_KV_SLOTS");
        m->n_kv_slots = (kvs_env && atoi(kvs_env) > 1) ? atoi(kvs_env) : 1;
        if (m->n_kv_slots > 16) m->n_kv_slots = 16;
        m->cur_kv_slot = 0;
        m->slot_ctx = m->max_ctx / m->n_kv_slots;
        if (m->slot_ctx < 64) { m->n_kv_slots = 1; m->slot_ctx = m->max_ctx; }

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
                if (!dot) continue;
                const char* sub = dot + 1;
                if (L == m->n_layers && m->nextn_layers > 0) {
                    if (strcmp(sub, "enorm.weight") == 0) { load_pinned_weight(m, &m->mtp_enorm, ti); m->has_mtp = 1; }
                    else if (strcmp(sub, "hnorm.weight") == 0) { load_pinned_weight(m, &m->mtp_hnorm, ti); m->has_mtp = 1; }
                    else if (strcmp(sub, "eh_proj.weight") == 0) { load_pinned_weight(m, &m->mtp_eh_proj, ti); m->has_mtp = 1; }
                    else {
                        if (!m->mtp_layer) {
                            m->mtp_layer = (MoeLayer*)calloc(1, sizeof(MoeLayer));
                            m->mtp_layer->is_dense = m->dense_model ? 1 : 0;
                            m->mtp_layer->is_pinned_dense = 1;
                        }
                        MoeLayer* ml = m->mtp_layer;
                        if (strcmp(sub, "attn_norm.weight") == 0) { load_pinned_weight(m, &ml->attn_norm, ti); m->has_mtp_layer = 1; }
                        else if (strcmp(sub, "ffn_norm.weight") == 0 || strcmp(sub, "post_attention_norm.weight") == 0) load_pinned_weight(m, &ml->ffn_norm, ti);
                        else if (strcmp(sub, "attn_q.weight") == 0) load_pinned_weight(m, &ml->wq, ti);
                        else if (strcmp(sub, "attn_k.weight") == 0) load_pinned_weight(m, &ml->wk, ti);
                        else if (strcmp(sub, "attn_v.weight") == 0) load_pinned_weight(m, &ml->wv, ti);
                        else if (strcmp(sub, "attn_output.weight") == 0) load_pinned_weight(m, &ml->wo, ti);
                        else if (strcmp(sub, "ffn_gate.weight") == 0) { load_pinned_weight(m, &ml->ffn_gate, ti); ml->is_dense = 1; }
                        else if (strcmp(sub, "ffn_up.weight") == 0) load_pinned_weight(m, &ml->ffn_up, ti);
                        else if (strcmp(sub, "ffn_down.weight") == 0) load_pinned_weight(m, &ml->ffn_down, ti);
                        else if (strcmp(sub, "ffn_gate_inp.weight") == 0) load_pinned_weight(m, &ml->router, ti);
                        else if (strcmp(sub, "ffn_gate_shexp.weight") == 0) { load_pinned_weight(m, &ml->sh_gate, ti); ml->has_shared_exp = 1; }
                        else if (strcmp(sub, "ffn_up_shexp.weight") == 0) load_pinned_weight(m, &ml->sh_up, ti);
                        else if (strcmp(sub, "ffn_down_shexp.weight") == 0) load_pinned_weight(m, &ml->sh_down, ti);
                    }
                    continue;
                }
                if (L < 0 || L >= m->n_layers) continue;
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
                } else if (strcmp(sub, "attn_q_a.weight") == 0) {
                    load_pinned_weight(m, &ly->wq_a, ti); ly->is_mla = 1;
                } else if (strcmp(sub, "attn_q_a_norm.weight") == 0) {
                    load_pinned_weight(m, &ly->q_a_norm, ti);
                } else if (strcmp(sub, "attn_q_b.weight") == 0) {
                    load_pinned_weight(m, &ly->wq_b, ti);
                } else if (strcmp(sub, "attn_kv_a_mqa.weight") == 0) {
                    load_pinned_weight(m, &ly->wkv_a, ti); ly->is_mla = 1;
                } else if (strcmp(sub, "attn_kv_a_norm.weight") == 0) {
                    load_pinned_weight(m, &ly->kv_a_norm, ti);
                } else if (strcmp(sub, "attn_kv_b.weight") == 0) {
                    load_pinned_weight(m, &ly->wkv_b, ti);
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

        // YaRN RoPE Frequency Divisor Synthesis & `mscale^2` Softmax Scaling (exact DeepSeek-V2/V3 math!)
        m->attn_mscale = 1.0f;
        if (m->yarn_factor > 1.0f) {
            if (m->yarn_log_m > 0.0f) {
                float ms = 1.0f + m->yarn_log_m * logf(m->yarn_factor);
                m->attn_mscale = ms * ms;
            }
            if (!m->has_rope_freqs && m->rope_dim > 0) {
                int half_r = m->rope_dim / 2;
                float orig_ctx = (m->yarn_orig_ctx > 0.0f) ? m->yarn_orig_ctx : 4096.0f;
                float beta_fast = 32.0f, beta_slow = 1.0f;
                float pi2 = 6.28318530718f;
                float low  = floorf((float)m->rope_dim * logf(orig_ctx / (beta_fast * pi2)) / (2.0f * logf(m->rope_theta)));
                float high = ceilf((float)m->rope_dim  * logf(orig_ctx / (beta_slow * pi2)) / (2.0f * logf(m->rope_theta)));
                if (low < 0.0f) low = 0.0f;
                if (high > (float)(half_r - 1)) high = (float)(half_r - 1);
                if (high <= low) high = low + 0.001f;
                m->rope_freqs.buf = [g_moe_dev newBufferWithLength:(size_t)half_r * 4u options:MTLResourceStorageModeShared];
                m->rope_freqs.offset = 0;
                float* rf = (float*)[m->rope_freqs.buf contents];
                for (int i = 0; i < half_r; i++) {
                    float ramp = ((float)i - low) / (high - low);
                    if (ramp < 0.0f) ramp = 0.0f;
                    if (ramp > 1.0f) ramp = 1.0f;
                    float alpha = 1.0f - ramp; // 1 = high freq (keep), 0 = low freq (scale by 1/factor)
                    float freq_mult = alpha + (1.0f - alpha) / m->yarn_factor;
                    rf[i] = 1.0f / (freq_mult > 1e-12f ? freq_mult : 1e-12f);
                }
                m->has_rope_freqs = 1;
            }
        }

        // Allocate KV caches (MLA Compressed Latent `[c_kv||k_pe]` = -8x..-85x KV VRAM! OR Q8_0 = -47%! OR FP16) + Streaming Pools!
        size_t kv_bytes = m->kv_quant_q8
            ? ((size_t)m->max_ctx * (size_t)m->n_kv * (size_t)(m->d_head >> 5) * 34u)
            : ((size_t)m->max_ctx * (size_t)(m->n_kv * m->d_head) * 2u);
        uint64_t max_wq = 0, max_wk = 0, max_wv = 0, max_wo = 0, max_fg = 0, max_fu = 0, max_fd = 0;
        for (int L = 0; L < m->n_layers; L++) {
            MoeLayer* ly = &m->layers[L];
            if (ly->is_mla && m->mla_latent_kv && ly->wkv_b.buf != nil) {
                uint32_t kv_r = (uint32_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
                uint32_t nope_d = (uint32_t)(m->d_head - m->rope_dim);
                uint32_t v_d = (uint32_t)m->v_head_dim;
                size_t lat_bytes = (size_t)m->max_ctx * (size_t)(kv_r + (uint32_t)m->rope_dim) * 2u;
                ly->k_cache = [g_moe_dev newBufferWithLength:lat_bytes options:MTLResourceStorageModeShared];
                ly->v_cache = nil; // ZERO V-cache VRAM in Matrix-Absorbed Latent MLA!
                m->core_vram_bytes += (int64_t)lat_bytes;
                // One-time load-time unpack of wkv_b into coalesced FP16 W_UK^T and W_UV!
                size_t wuk_b = (size_t)m->n_heads * kv_r * nope_d * 2u;
                size_t wuv_b = (size_t)m->n_heads * v_d * kv_r * 2u;
                ly->mla_wuk_f16 = [g_moe_dev newBufferWithLength:(wuk_b > 64 ? wuk_b : 64) options:MTLResourceStorageModeShared];
                ly->mla_wuv_f16 = [g_moe_dev newBufferWithLength:(wuv_b > 64 ? wuv_b : 64) options:MTLResourceStorageModeShared];
                id<MTLCommandBuffer> ucmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> uenc = [ucmd computeCommandEncoder];
                struct { uint32_t nh, nd, vd, kvr, gt; } ua = { (uint32_t)m->n_heads, nope_d, v_d, kv_r, ly->wkv_b.ggml_type };
                [uenc setComputePipelineState:g_pso_mla_unpack_kvb];
                [uenc setBuffer:ly->wkv_b.buf offset:(NSUInteger)ly->wkv_b.offset atIndex:0];
                [uenc setBuffer:ly->mla_wuk_f16 offset:0 atIndex:1];
                [uenc setBuffer:ly->mla_wuv_f16 offset:0 atIndex:2];
                [uenc setBytes:&ua length:sizeof(ua) atIndex:3];
                uint32_t max_j = nope_d > v_d ? nope_d : v_d;
                [uenc dispatchThreads:MTLSizeMake(kv_r, max_j, (NSUInteger)m->n_heads) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                [uenc endEncoding]; [ucmd commit]; [ucmd waitUntilCompleted];
            } else {
                ly->k_cache = [g_moe_dev newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
                ly->v_cache = [g_moe_dev newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
                m->core_vram_bytes += (int64_t)(kv_bytes * 2);
            }
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
        for (int L = 0; L < m->n_layers; L++) {
            if (m->layers[L].has_shared_exp && m->layers[L].sh_gate.ne1 > sh_elems) sh_elems = (size_t)m->layers[L].sh_gate.ne1;
            if (m->layers[L].is_mla) {
                size_t kvb_o = (size_t)m->n_heads * (size_t)((m->d_head - m->rope_dim) + m->v_head_dim);
                if (kvb_o > sh_elems) sh_elems = kvb_o;
            }
        }
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
        m->buf_last_hidden = [g_moe_dev newBufferWithLength:(size_t)m->d_model * 4 options:MTLResourceStorageModeShared];
        if (m->has_mtp) {
            m->buf_mtp_cat = [g_moe_dev newBufferWithLength:(size_t)m->d_model * 8 options:MTLResourceStorageModeShared];
            if (m->mtp_layer && m->has_mtp_layer) {
                m->mtp_layer->k_cache = [g_moe_dev newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
                m->mtp_layer->v_cache = [g_moe_dev newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
            }
        }
        // Initialize PagedAttention 16-Token Physical Block Pool & Block Tables + Tree-Attention Buffers!
        m->pa_num_blocks = m->max_ctx / PA_BLOCK_SIZE;
        m->pa_free_blocks = m->pa_num_blocks;
        m->pa_refcnt = (int*)calloc((size_t)(m->pa_num_blocks > 0 ? m->pa_num_blocks : 1), sizeof(int));
        for (int s = 0; s < PA_MAX_SEQS; s++) {
            m->pa_seq_len[s] = 0;
            for (int b = 0; b < PA_MAX_LOGICAL_BLKS; b++) m->pa_block_table[s][b] = -1;
        }
        m->buf_pa_bt      = [g_moe_dev newBufferWithLength:(size_t)PA_MAX_SEQS * PA_MAX_LOGICAL_BLKS * 4 options:MTLResourceStorageModeShared];
        memset([m->buf_pa_bt contents], 0xFF, (size_t)PA_MAX_SEQS * PA_MAX_LOGICAL_BLKS * 4);
        m->buf_cb_pos     = [g_moe_dev newBufferWithLength:(size_t)MOE_PREFILL_MAX_B * 4 options:MTLResourceStorageModeShared];
        m->buf_cb_slot    = [g_moe_dev newBufferWithLength:(size_t)MOE_PREFILL_MAX_B * 4 options:MTLResourceStorageModeShared];
        m->buf_tree_depth = [g_moe_dev newBufferWithLength:(size_t)TREE_MAX_NODES * 4 options:MTLResourceStorageModeShared];
        m->buf_tree_mask  = [g_moe_dev newBufferWithLength:(size_t)TREE_MAX_NODES * 8 options:MTLResourceStorageModeShared];
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
    NSUInteger tptg = 128;
    switch (gtype) {
        case TN_GGML_Q4_0: pso = g_pso_gemv_q4;   rows_per_tg = 8; tptg = 64; break;
        case TN_GGML_Q4_1: pso = g_pso_gemv_q4_1; rows_per_tg = 8; tptg = 64; break;
        case TN_GGML_Q5_0: pso = g_pso_gemv_q5_0; rows_per_tg = 4; tptg = 64; break;
        case TN_GGML_Q5_1: pso = g_pso_gemv_q5_1; rows_per_tg = 4; tptg = 64; break;
        case TN_GGML_Q8_0:   pso = g_pso_gemv_q8;    rows_per_tg = 4; tptg = 64; break;
        case TN_GGML_Q2_K:   pso = g_pso_gemv_q2k;   rows_per_tg = 8; break;
        case TN_GGML_Q3_K:   pso = g_pso_gemv_q3k;   rows_per_tg = 8; break;
        case TN_GGML_IQ4_XS: pso = g_pso_gemv_iq4xs; rows_per_tg = 8; break;
        case TN_GGML_Q4_K:   pso = g_pso_gemv_q4k;   rows_per_tg = 8; break;
        case TN_GGML_Q5_K: pso = g_pso_gemv_q5k;  rows_per_tg = 8; break;
        case TN_GGML_Q6_K: pso = g_pso_gemv_q6k;  rows_per_tg = 8; break;
        case TN_GGML_F16:  pso = g_pso_gemv_f16;  rows_per_tg = 4; break;
        default:           pso = g_pso_gemv_f32;  rows_per_tg = 4; break;
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf offset:(NSUInteger)off atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1];
    [enc setBuffer:y offset:0 atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((out_dim + rows_per_tg - 1) / rows_per_tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tptg, 1, 1)];
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
        [enc dispatchThreadgroups:MTLSizeMake((d_ff + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
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

static void encode_layer_attn_body_ptr(
    MoeModel* m, id<MTLComputeCommandEncoder> enc, MoeLayer* ly, uint32_t pos,
    id<MTLBuffer> ping_buf // nil -> use ly->wq/wk/wv/wo pinned buffers; non-nil -> use ping_buf offsets!
) {
    uint32_t dm = (uint32_t)m->d_model;
    uint32_t qdim = (uint32_t)(m->n_heads * m->d_head);
    uint32_t kvdim = (uint32_t)(m->n_kv * m->d_head);
    uint32_t wo_in = (ly->wo.ne0 > 0) ? (uint32_t)ly->wo.ne0 : qdim;
    uint32_t out_stride = wo_in / (uint32_t)(m->n_heads > 0 ? m->n_heads : 1);
    float attn_scale = (m->attn_mscale > 0.0f ? m->attn_mscale : 1.0f) / sqrtf((float)m->d_head);
    encode_rmsnorm(enc, m->buf_x, &ly->attn_norm, m->buf_xn, dm, 1, m->rms_eps, 0);

    if (ly->is_mla && ly->wkv_a.buf != nil && ly->wkv_b.buf != nil) {
        // DeepSeek-V2/V3/R1 Multi-Head Latent Attention (MLA): Official RoPE-on-PE + Matrix-Absorbed Compressed Latent KV!
        uint32_t kv_r = (uint32_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
        uint32_t nope_d = (uint32_t)(m->d_head - m->rope_dim);
        uint32_t v_d = (uint32_t)m->v_head_dim;
        if (m->pbuf_mla_q == nil) {
            m->pbuf_mla_q    = [g_moe_dev newBufferWithLength:(size_t)(m->q_lora_rank > 0 ? m->q_lora_rank : m->d_model) * 4 options:MTLResourceStorageModeShared];
            m->pbuf_mla_kv   = [g_moe_dev newBufferWithLength:(size_t)(kv_r + m->rope_dim + m->n_heads * (nope_d + v_d)) * 4 options:MTLResourceStorageModeShared];
            m->pbuf_mla_qabs = [g_moe_dev newBufferWithLength:(size_t)m->n_heads * kv_r * 4 options:MTLResourceStorageModeShared];
            m->pbuf_mla_olat = [g_moe_dev newBufferWithLength:(size_t)m->n_heads * kv_r * 4 options:MTLResourceStorageModeShared];
        }
        if (ly->wq_a.buf != nil && ly->wq_b.buf != nil && m->q_lora_rank > 0) {
            encode_gemv(enc, &ly->wq_a, m->buf_xn, m->pbuf_mla_q, (uint32_t)m->q_lora_rank, dm, 0);
            if (ly->q_a_norm.buf != nil) encode_rmsnorm(enc, m->pbuf_mla_q, &ly->q_a_norm, m->pbuf_mla_q, (uint32_t)m->q_lora_rank, 1, m->rms_eps, 0);
            encode_gemv(enc, &ly->wq_b, m->pbuf_mla_q, m->buf_q, qdim, (uint32_t)m->q_lora_rank, 0);
        } else {
            encode_gemv(enc, &ly->wq, m->buf_xn, m->buf_q, qdim, dm, 0);
        }
        // kv_a -> [c_kv (kv_r) | k_pe (rope_dim)]
        encode_gemv(enc, &ly->wkv_a, m->buf_xn, m->pbuf_mla_kv, kv_r + (uint32_t)m->rope_dim, dm, 0);
        if (ly->kv_a_norm.buf != nil) encode_rmsnorm(enc, m->pbuf_mla_kv, &ly->kv_a_norm, m->pbuf_mla_kv, kv_r, 1, m->rms_eps, 0);
        // Rotate ONLY the positional slice `q[h, nope_d..d_head-1]` and MQA `k_pe[0..rope_dim-1]`!
        struct { uint32_t nh, nd, rd, dh, pos; float theta; uint32_t rn, hf; } mr = {
            (uint32_t)m->n_heads, nope_d, (uint32_t)m->rope_dim, (uint32_t)m->d_head, pos,
            m->rope_theta, (uint32_t)m->rope_neox, (uint32_t)m->has_rope_freqs
        };
        [enc setComputePipelineState:g_pso_mla_rope_pe];
        [enc setBuffer:m->buf_q offset:0 atIndex:0];
        [enc setBuffer:m->pbuf_mla_kv offset:(NSUInteger)(kv_r * 4u) atIndex:1];
        [enc setBuffer:(m->has_rope_freqs ? m->rope_freqs.buf : m->buf_argmax)
                offset:(m->has_rope_freqs ? (NSUInteger)m->rope_freqs.offset : 0) atIndex:2];
        [enc setBytes:&mr length:sizeof(mr) atIndex:3];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)(m->rope_dim / 2), (NSUInteger)m->n_heads, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        if (m->mla_latent_kv && ly->mla_wuk_f16 != nil) {
            NSUInteger lat_slot_off = (NSUInteger)m->cur_kv_slot * (NSUInteger)m->slot_ctx * (NSUInteger)(kv_r + (uint32_t)m->rope_dim) * 2u;
            struct { uint32_t kvr, rd, pos; } st = { kv_r, (uint32_t)m->rope_dim, pos };
            [enc setComputePipelineState:g_pso_mla_store_lat];
            [enc setBuffer:m->pbuf_mla_kv offset:0 atIndex:0];
            [enc setBuffer:ly->k_cache offset:lat_slot_off atIndex:1];
            [enc setBytes:&st length:sizeof(st) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(kv_r + (uint32_t)m->rope_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

            struct { uint32_t nh, nd, kvr, dh; } ab = { (uint32_t)m->n_heads, nope_d, kv_r, (uint32_t)m->d_head };
            [enc setComputePipelineState:g_pso_mla_absorb_q];
            [enc setBuffer:ly->mla_wuk_f16 offset:0 atIndex:0];
            [enc setBuffer:m->buf_q offset:0 atIndex:1];
            [enc setBuffer:m->pbuf_mla_qabs offset:0 atIndex:2];
            [enc setBytes:&ab length:sizeof(ab) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(kv_r, (NSUInteger)m->n_heads, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            struct { uint32_t nh, kvr, nd, rd, dh, kvl; float sc; } fl = {
                (uint32_t)m->n_heads, kv_r, nope_d, (uint32_t)m->rope_dim, (uint32_t)m->d_head, pos + 1u, attn_scale
            };
            [enc setComputePipelineState:g_pso_mla_flash_lat];
            [enc setBuffer:m->pbuf_mla_qabs offset:0 atIndex:0];
            [enc setBuffer:m->buf_q offset:0 atIndex:1];
            [enc setBuffer:ly->k_cache offset:lat_slot_off atIndex:2];
            [enc setBuffer:m->pbuf_mla_olat offset:0 atIndex:3];
            [enc setBytes:&fl length:sizeof(fl) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            struct { uint32_t nh, vd, kvr, os; } pv = { (uint32_t)m->n_heads, v_d, kv_r, out_stride };
            [enc setComputePipelineState:g_pso_mla_proj_v];
            [enc setBuffer:ly->mla_wuv_f16 offset:0 atIndex:0];
            [enc setBuffer:m->pbuf_mla_olat offset:0 atIndex:1];
            [enc setBuffer:m->buf_ctx offset:0 atIndex:2];
            [enc setBytes:&pv length:sizeof(pv) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)out_stride, (NSUInteger)m->n_heads, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

            encode_gemv(enc, &ly->wo, m->buf_ctx, m->buf_x, dm, wo_in, 1);
            encode_rmsnorm(enc, m->buf_x, &ly->ffn_norm, m->buf_xn, dm, 1, m->rms_eps, 0);
            return;
        }
        // Uncompressed MLA path (`TENON_MLA_LATENT_KV=0`): expand wkv_b -> split K/V (already rotated!)
        uint32_t kvb_out = (uint32_t)m->n_heads * (nope_d + v_d);
        encode_gemv(enc, &ly->wkv_b, m->pbuf_mla_kv, m->buf_sh_g, kvb_out, kv_r, 0);
        struct { uint32_t nh, nd, rd, vd, dh, bb; } ms = { (uint32_t)m->n_heads, nope_d, (uint32_t)m->rope_dim, v_d, (uint32_t)m->d_head, 1u };
        [enc setComputePipelineState:g_pso_mla_split_kv];
        [enc setBuffer:m->buf_sh_g offset:0 atIndex:0];
        [enc setBuffer:m->pbuf_mla_kv offset:(NSUInteger)(kv_r * 4u) atIndex:1];
        [enc setBuffer:m->buf_k offset:0 atIndex:2];
        [enc setBuffer:m->buf_v offset:0 atIndex:3];
        [enc setBytes:&ms length:sizeof(ms) atIndex:4];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)m->d_head, (NSUInteger)m->n_heads, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    } else if (ping_buf != nil) {
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

    uint32_t eff_rope_dim = ly->is_mla ? 0u : (uint32_t)m->rope_dim;
    NSUInteger kv_slot_off = m->kv_quant_q8
        ? ((NSUInteger)m->cur_kv_slot * (NSUInteger)m->slot_ctx * (NSUInteger)m->n_kv * (NSUInteger)(m->d_head >> 5) * 34u)
        : ((NSUInteger)m->cur_kv_slot * (NSUInteger)m->slot_ctx * (NSUInteger)(m->n_kv * m->d_head) * 2u);
    if (m->kv_quant_q8) {
        struct { uint32_t n_heads, n_kv, d_head, rope_dim, start_pos; float theta; uint32_t rope_neox, has_freqs, batch_b; } rargs = {
            (uint32_t)m->n_heads, (uint32_t)m->n_kv, (uint32_t)m->d_head, eff_rope_dim,
            pos, m->rope_theta, (uint32_t)m->rope_neox, (uint32_t)m->has_rope_freqs, 1u
        };
        [enc setComputePipelineState:g_pso_rope_kv_q8];
        [enc setBuffer:m->buf_q offset:0 atIndex:0];
        [enc setBuffer:m->buf_k offset:0 atIndex:1];
        [enc setBuffer:m->buf_v offset:0 atIndex:2];
        [enc setBuffer:ly->k_cache offset:kv_slot_off atIndex:3];
        [enc setBuffer:ly->v_cache offset:kv_slot_off atIndex:4];
        [enc setBuffer:(m->has_rope_freqs ? m->rope_freqs.buf : m->buf_argmax)
                offset:(m->has_rope_freqs ? (NSUInteger)m->rope_freqs.offset : 0) atIndex:5];
        [enc setBytes:&rargs length:sizeof(rargs) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        struct { uint32_t n_heads, n_kv, d_head, start_pos, batch_b, out_stride; float scale; } aargs = {
            (uint32_t)m->n_heads, (uint32_t)m->n_kv, (uint32_t)m->d_head, pos, 1u, out_stride, attn_scale
        };
        [enc setComputePipelineState:g_pso_flash_dec_q8];
        [enc setBuffer:m->buf_q offset:0 atIndex:0];
        [enc setBuffer:ly->k_cache offset:kv_slot_off atIndex:1];
        [enc setBuffer:ly->v_cache offset:kv_slot_off atIndex:2];
        [enc setBuffer:m->buf_ctx offset:0 atIndex:3];
        [enc setBytes:&aargs length:sizeof(aargs) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        struct { uint32_t n_heads, n_kv, d_head, rope_dim, pos; float theta; uint32_t rope_neox, has_freqs; } rargs = {
            (uint32_t)m->n_heads, (uint32_t)m->n_kv, (uint32_t)m->d_head, eff_rope_dim,
            pos, m->rope_theta, (uint32_t)m->rope_neox, (uint32_t)m->has_rope_freqs
        };
        [enc setComputePipelineState:g_pso_rope_kv];
        [enc setBuffer:m->buf_q offset:0 atIndex:0];
        [enc setBuffer:m->buf_k offset:0 atIndex:1];
        [enc setBuffer:m->buf_v offset:0 atIndex:2];
        [enc setBuffer:ly->k_cache offset:kv_slot_off atIndex:3];
        [enc setBuffer:ly->v_cache offset:kv_slot_off atIndex:4];
        [enc setBuffer:(m->has_rope_freqs ? m->rope_freqs.buf : m->buf_argmax)
                offset:(m->has_rope_freqs ? (NSUInteger)m->rope_freqs.offset : 0) atIndex:5];
        [enc setBytes:&rargs length:sizeof(rargs) atIndex:6];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)(m->d_head / 2), (NSUInteger)m->n_heads, 1)
       threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        struct { uint32_t n_heads, n_kv, d_head, kv_len, out_stride; float scale; } aargs = {
            (uint32_t)m->n_heads, (uint32_t)m->n_kv, (uint32_t)m->d_head, pos + 1u, out_stride, attn_scale
        };
        [enc setComputePipelineState:g_pso_flash_dec];
        [enc setBuffer:m->buf_q offset:0 atIndex:0];
        [enc setBuffer:ly->k_cache offset:kv_slot_off atIndex:1];
        [enc setBuffer:ly->v_cache offset:kv_slot_off atIndex:2];
        [enc setBuffer:m->buf_ctx offset:0 atIndex:3];
        [enc setBytes:&aargs length:sizeof(aargs) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }

    if (ping_buf != nil) {
        encode_gemv_raw(enc, ping_buf, m->dense_off_wo, ly->wo.ggml_type, m->buf_ctx, m->buf_x, dm, wo_in, 1);
    } else {
        encode_gemv(enc, &ly->wo, m->buf_ctx, m->buf_x, dm, wo_in, 1);
    }
    encode_rmsnorm(enc, m->buf_x, &ly->ffn_norm, m->buf_xn, dm, 1, m->rms_eps, 0);
}

static void encode_layer_attn_body(
    MoeModel* m, id<MTLComputeCommandEncoder> enc, int L, uint32_t pos, id<MTLBuffer> ping_buf
) {
    encode_layer_attn_body_ptr(m, enc, &m->layers[L], pos, ping_buf);
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

static void cpu_router_topk_from_logits_grp(
    const float* logits, const float* exp_bias, int n_exp, int top_k,
    int gating_func, int norm_wt, float w_scale, int n_groups, int n_group_used, int* out_e, float* out_w
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
    if (n_groups > 1 && n_group_used > 0 && n_group_used < n_groups) {
        int gsz = n_exp / n_groups;
        float gsc[32]; int gkeep[32];
        for (int g = 0; g < n_groups && g < 32; g++) {
            float m1 = -1e30f, m2 = -1e30f;
            for (int i = 0; i < gsz; i++) {
                float sv = scores[g * gsz + i];
                if (sv > m1) { m2 = m1; m1 = sv; } else if (sv > m2) { m2 = sv; }
            }
            gsc[g] = (gating_func == 2) ? (m1 + m2) : m1;
            gkeep[g] = 0;
        }
        for (int u = 0; u < n_group_used; u++) {
            int bg = 0; float bs = -1e30f;
            for (int g = 0; g < n_groups && g < 32; g++) if (!gkeep[g] && gsc[g] > bs) { bs = gsc[g]; bg = g; }
            gkeep[bg] = 1;
        }
        for (int g = 0; g < n_groups && g < 32; g++) {
            if (!gkeep[g]) for (int i = 0; i < gsz; i++) scores[g * gsz + i] = -1e30f;
        }
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

static void cpu_router_topk_from_logits(
    const float* logits, const float* exp_bias, int n_exp, int top_k,
    int gating_func, int norm_wt, float w_scale, int* out_e, float* out_w
) {
    cpu_router_topk_from_logits_grp(logits, exp_bias, n_exp, top_k, gating_func, norm_wt, w_scale, 0, 0, out_e, out_w);
}

static void schedule_next_layer_prefetch(MoeModel* m, int next_L) {
    if (!m->enable_overlap || next_L >= m->n_layers) return;
    // Only speculate when cache is large enough to avoid thrashing (>= top_k+4) AND small enough (< 50% experts)
    // that misses are frequent; at >=50% cache (e.g. 64-80/128), natural LRU hit rate is already ~90% so
    // speculative mispredicts only add wasted SSD traffic and cache pollution!
    if (m->cache_slots_per_layer < m->top_k + 4 || m->cache_slots_per_layer * 2 >= m->n_experts) return;
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

static inline void select_moe_down_pso(uint32_t gtype, int is_slot, int d_model,
                                       id<MTLComputePipelineState>* out_pso, NSUInteger* out_tgs, NSUInteger* out_tptg) {
    if (gtype == TN_GGML_Q4_K) {
        *out_pso = is_slot ? g_pso_moe_slot_down_q4k : g_pso_moe_down_q4k;
        *out_tgs = (NSUInteger)((d_model + 3) / 4); *out_tptg = 128;
    } else if (gtype == TN_GGML_Q5_0) {
        *out_pso = is_slot ? g_pso_moe_slot_down_q5_0 : g_pso_moe_down_q5_0;
        *out_tgs = (NSUInteger)((d_model + 3) / 4); *out_tptg = 128;
    } else if (gtype == TN_GGML_Q5_1) {
        *out_pso = is_slot ? g_pso_moe_slot_down_q5_1 : g_pso_moe_down_q5_1;
        *out_tgs = (NSUInteger)((d_model + 3) / 4); *out_tptg = 128;
    } else if (gtype == TN_GGML_Q8_0) {
        *out_pso = is_slot ? g_pso_moe_slot_down_q8 : g_pso_moe_down_q8;
        *out_tgs = (NSUInteger)((d_model + 3) / 4); *out_tptg = 128;
    } else if (gtype == TN_GGML_Q6_K) {
        *out_pso = is_slot ? g_pso_moe_slot_down_q6k : g_pso_moe_down_q6k;
        *out_tgs = (NSUInteger)((d_model + 3) / 4); *out_tptg = 128;
    } else if (gtype == TN_GGML_Q4_1) {
        *out_pso = is_slot ? g_pso_moe_slot_down_q4_1 : g_pso_moe_down_q4_1;
        *out_tgs = (NSUInteger)((d_model + 7) / 8); *out_tptg = 64;
    } else {
        *out_pso = is_slot ? g_pso_moe_slot_down_q4 : g_pso_moe_down_q4;
        *out_tgs = (NSUInteger)((d_model + 7) / 8); *out_tptg = 64;
    }
}

static int encode_tail_until_next_moe(MoeModel* m, id<MTLComputeCommandEncoder> enc, int start_L, uint32_t pos) {
    int L = start_L;
    while (L < m->n_layers && m->layers[L].is_dense) {
        MoeLayer* ly = &m->layers[L];
        encode_layer_attn_body(m, enc, L, pos, nil);
        encode_dense_ffn_raw(m, enc,
            ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
            ly->ffn_up.buf,   ly->ffn_up.offset,   ly->ffn_up.ggml_type,
            ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type,
            (uint32_t)m->d_ff_dense);
        L++;
    }
    if (L < m->n_layers) {
        MoeLayer* ly = &m->layers[L];
        encode_layer_attn_body(m, enc, L, pos, nil);
        encode_gemv(enc, &ly->router, m->buf_xn, m->buf_rlogits, (uint32_t)m->n_experts, (uint32_t)m->d_model, 0);
    } else {
        encode_final_lm_head(m, enc);
    }
    return L;
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
                    struct { uint32_t ne, tk, nw, gf, hb; float ws; uint32_t ng, ngu; } rt = {
                        (uint32_t)m->n_experts, (uint32_t)m->top_k, (uint32_t)m->expert_weights_norm,
                        (uint32_t)m->gating_func, (uint32_t)ly->has_exp_bias, m->expert_weights_scale,
                        (uint32_t)m->n_expert_groups, (uint32_t)m->n_group_used
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
                        uint32_t sh_ff = (ly->sh_gate.ne1 > 0) ? (uint32_t)ly->sh_gate.ne1 : (uint32_t)m->d_ff_exp;
                        encode_dense_ffn_raw(m, enc,
                            ly->sh_gate.buf, ly->sh_gate.offset, ly->sh_gate.ggml_type,
                            ly->sh_up.buf,   ly->sh_up.offset,   ly->sh_up.ggml_type,
                            ly->sh_down.buf, ly->sh_down.offset, ly->sh_down.ggml_type,
                            sh_ff);
                    }
                    struct { uint32_t dm, df, na; uint64_t gb, ub, db, sb; } margs = {
                        (uint32_t)m->d_model, (uint32_t)m->d_ff_exp, (uint32_t)m->top_k,
                        ly->gate_exp_bytes, ly->up_exp_bytes, ly->down_exp_bytes, 0
                    };
                    int is_q4k_gu = (ly->gate_exps.ggml_type == TN_GGML_Q4_K);
                    [enc setComputePipelineState:(is_q4k_gu ? g_pso_moe_gu_q4k : g_pso_moe_gu_q4)];
                    [enc setBuffer:ly->gate_exps.buf offset:(NSUInteger)ly->gate_exps.offset atIndex:0];
                    [enc setBuffer:ly->up_exps.buf offset:(NSUInteger)ly->up_exps.offset atIndex:1];
                    [enc setBuffer:m->buf_xn offset:0 atIndex:2];
                    [enc setBuffer:m->buf_eidx offset:0 atIndex:3];
                    [enc setBuffer:m->buf_h offset:0 atIndex:4];
                    [enc setBytes:&margs length:sizeof(margs) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((m->d_ff_exp + 3) / 4), (NSUInteger)m->top_k, 1)
                        threadsPerThreadgroup:MTLSizeMake(is_q4k_gu ? 128 : 64, 1, 1)];

                    id<MTLComputePipelineState> r_dn_pso = nil;
                    NSUInteger r_dn_tgs = 0, r_dn_tptg = 64;
                    select_moe_down_pso(ly->down_exps.ggml_type, 0, m->d_model, &r_dn_pso, &r_dn_tgs, &r_dn_tptg);
                    [enc setComputePipelineState:r_dn_pso];
                    [enc setBuffer:ly->down_exps.buf offset:(NSUInteger)ly->down_exps.offset atIndex:0];
                    [enc setBuffer:m->buf_h offset:0 atIndex:1];
                    [enc setBuffer:m->buf_eidx offset:0 atIndex:2];
                    [enc setBuffer:m->buf_ewt offset:0 atIndex:3];
                    [enc setBuffer:m->buf_x offset:0 atIndex:4];
                    [enc setBytes:&margs length:sizeof(margs) atIndex:5];
                    [enc dispatchThreadgroups:MTLSizeMake(r_dn_tgs, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(r_dn_tptg, 1, 1)];
                    m->cache_hits += m->top_k;
                }
            }
            encode_final_lm_head(m, enc);
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            if (m->buf_last_hidden != nil) memcpy([m->buf_last_hidden contents], [m->buf_x contents], (size_t)m->d_model * 4u);
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
            if (m->buf_last_hidden != nil) memcpy([m->buf_last_hidden contents], [m->buf_x contents], (size_t)m->d_model * 4u);

            double tot = moe_now_ms() - t0;
            m->last_total_ms = tot; m->last_ssd_ms = ssd_acc_ms;
            m->last_gpu_ms = (tot > ssd_acc_ms) ? (tot - ssd_acc_ms) : 0.0;
            return (int64_t)((uint32_t*)[m->buf_argmax contents])[0];
        }

        // =========================================================================
        // PATH C: SPARSE MoE ON-DEMAND STREAMING MODE (Bounded Per-Layer LRU Pool + Fused MoE(L)+Attn(L+1) CB!)
        // =========================================================================
        int L = 0;
        {
            id<MTLCommandBuffer> icmd = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> ienc = [icmd computeCommandEncoder];
            encode_embed_step(m, ienc, (uint32_t)tok_id);
            L = encode_tail_until_next_moe(m, ienc, 0, (uint32_t)pos);
            [ienc endEncoding];
            [icmd commit];
            [icmd waitUntilCompleted];
        }

        while (L < m->n_layers) {
            MoeLayer* ly = &m->layers[L];

            // Ensure any speculative prefetch targeting Layer L has finished writing cache slots!
            dispatch_group_wait(m->prefetch_grp, DISPATCH_TIME_FOREVER);

            // Step 2: Top-K Router selection (incl. DeepSeek Group Top-K) + HIT / MISS partition
            int top_e[MOE_MAX_TOPK]; float top_w[MOE_MAX_TOPK];
            cpu_router_topk_from_logits_grp((const float*)[m->buf_rlogits contents], ly->exp_bias_host,
                                            m->n_experts, m->top_k, m->gating_func,
                                            m->expert_weights_norm, m->expert_weights_scale,
                                            m->n_expert_groups, m->n_group_used, top_e, top_w);

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

            int is_q4k_gu = (ly->gate_exps.ggml_type == TN_GGML_Q4_K);
            id<MTLComputePipelineState> gu_pso = is_q4k_gu ? g_pso_moe_slot_gu_q4k : g_pso_moe_slot_gu_q4;
            id<MTLComputePipelineState> down_pso = nil;
            NSUInteger gu_tptg = is_q4k_gu ? 128 : 64;
            NSUInteger dn_tptg = 64, dn_tgs = 0;
            select_moe_down_pso(ly->down_exps.ggml_type, 1, m->d_model, &down_pso, &dn_tgs, &dn_tptg);
            uint32_t sh_ff = (ly->sh_gate.ne1 > 0) ? (uint32_t)ly->sh_gate.ne1 : (uint32_t)m->d_ff_exp;

            if (!m->enable_overlap) {
                // Naive Synchronous Serial Baseline (also benefits from 1-CB fused MoE(L)+Attn(L+1)!)
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
                        sh_ff);
                }
                struct { uint32_t dm, df, na; uint64_t gb, ub, db, sb; } margs = {
                    (uint32_t)m->d_model, (uint32_t)m->d_ff_exp, (uint32_t)tot_a,
                    ly->gate_exp_bytes, ly->up_exp_bytes, ly->down_exp_bytes, ly->slot_stride_bytes
                };
                [menc setComputePipelineState:gu_pso];
                [menc setBuffer:ly->cache_buf offset:0 atIndex:0];
                [menc setBuffer:m->buf_xn offset:0 atIndex:1];
                [menc setBuffer:m->buf_eidx offset:0 atIndex:2];
                [menc setBuffer:m->buf_h offset:0 atIndex:3];
                [menc setBytes:&margs length:sizeof(margs) atIndex:4];
                [menc dispatchThreadgroups:MTLSizeMake((NSUInteger)((m->d_ff_exp + 3) / 4), (NSUInteger)tot_a, 1) threadsPerThreadgroup:MTLSizeMake(gu_tptg, 1, 1)];
                [menc setComputePipelineState:down_pso];
                [menc setBuffer:ly->cache_buf offset:0 atIndex:0];
                [menc setBuffer:m->buf_h offset:0 atIndex:1];
                [menc setBuffer:m->buf_eidx offset:0 atIndex:2];
                [menc setBuffer:m->buf_ewt offset:0 atIndex:3];
                [menc setBuffer:m->buf_x offset:0 atIndex:4];
                [menc setBytes:&margs length:sizeof(margs) atIndex:5];
                [menc dispatchThreadgroups:MTLSizeMake(dn_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(dn_tptg, 1, 1)];
                int next_L = encode_tail_until_next_moe(m, menc, L + 1, (uint32_t)pos);
                [menc endEncoding];
                [mcmd commit];
                [mcmd waitUntilCompleted];
                L = next_L;
            } else if (n_misses == 0) {
                // ZERO-MISS FAST PATH: Fuse SharedExp + All Hits(L) + Attn+Router(L+1) in ONE single CommandBuffer!
                if (n_hits > 0) {
                    uint32_t hs[MOE_MAX_TOPK];
                    for (int hi = 0; hi < n_hits; hi++) hs[hi] = (uint32_t)hit_slots[hi];
                    memcpy([m->buf_eidx contents], hs, (size_t)n_hits * 4);
                    memcpy([m->buf_ewt contents], hit_w, (size_t)n_hits * 4);
                }
                id<MTLCommandBuffer> hcmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> henc = [hcmd computeCommandEncoder];
                if (ly->has_shared_exp) {
                    encode_dense_ffn_raw(m, henc,
                        ly->sh_gate.buf, ly->sh_gate.offset, ly->sh_gate.ggml_type,
                        ly->sh_up.buf,   ly->sh_up.offset,   ly->sh_up.ggml_type,
                        ly->sh_down.buf, ly->sh_down.offset, ly->sh_down.ggml_type,
                        sh_ff);
                }
                if (n_hits > 0) {
                    struct { uint32_t dm, df, na; uint64_t gb, ub, db, sb; } margs = {
                        (uint32_t)m->d_model, (uint32_t)m->d_ff_exp, (uint32_t)n_hits,
                        ly->gate_exp_bytes, ly->up_exp_bytes, ly->down_exp_bytes, ly->slot_stride_bytes
                    };
                    [henc setComputePipelineState:gu_pso];
                    [henc setBuffer:ly->cache_buf offset:0 atIndex:0];
                    [henc setBuffer:m->buf_xn offset:0 atIndex:1];
                    [henc setBuffer:m->buf_eidx offset:0 atIndex:2];
                    [henc setBuffer:m->buf_h offset:0 atIndex:3];
                    [henc setBytes:&margs length:sizeof(margs) atIndex:4];
                    [henc dispatchThreadgroups:MTLSizeMake((NSUInteger)((m->d_ff_exp + 3) / 4), (NSUInteger)n_hits, 1) threadsPerThreadgroup:MTLSizeMake(gu_tptg, 1, 1)];
                    [henc setComputePipelineState:down_pso];
                    [henc setBuffer:ly->cache_buf offset:0 atIndex:0];
                    [henc setBuffer:m->buf_h offset:0 atIndex:1];
                    [henc setBuffer:m->buf_eidx offset:0 atIndex:2];
                    [henc setBuffer:m->buf_ewt offset:0 atIndex:3];
                    [henc setBuffer:m->buf_x offset:0 atIndex:4];
                    [henc setBytes:&margs length:sizeof(margs) atIndex:5];
                    [henc dispatchThreadgroups:MTLSizeMake(dn_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(dn_tptg, 1, 1)];
                }
                // Snapshot pre-MoE residual for optional low-cache prefetch before GPU mutates buf_x!
                int next_L = encode_tail_until_next_moe(m, henc, L + 1, (uint32_t)pos);
                [henc endEncoding];
                schedule_next_layer_prefetch(m, next_L);
                [hcmd commit];
                [hcmd waitUntilCompleted];
                L = next_L;
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
                        [henc setComputePipelineState:gu_pso];
                        [henc setBuffer:ly->cache_buf offset:0 atIndex:0];
                        [henc setBuffer:m->buf_xn offset:0 atIndex:1];
                        [henc setBuffer:m->buf_eidx offset:0 atIndex:2];
                        [henc setBuffer:m->buf_h offset:0 atIndex:3];
                        [henc setBytes:&margs length:sizeof(margs) atIndex:4];
                        [henc dispatchThreadgroups:MTLSizeMake((NSUInteger)((m->d_ff_exp + 3) / 4), (NSUInteger)n_hits, 1) threadsPerThreadgroup:MTLSizeMake(gu_tptg, 1, 1)];
                        [henc setComputePipelineState:down_pso];
                        [henc setBuffer:ly->cache_buf offset:0 atIndex:0];
                        [henc setBuffer:m->buf_h offset:0 atIndex:1];
                        [henc setBuffer:m->buf_eidx offset:0 atIndex:2];
                        [henc setBuffer:m->buf_ewt offset:0 atIndex:3];
                        [henc setBuffer:m->buf_x offset:0 atIndex:4];
                        [henc setBytes:&margs length:sizeof(margs) atIndex:5];
                        [henc dispatchThreadgroups:MTLSizeMake(dn_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(dn_tptg, 1, 1)];
                    }
                    [henc endEncoding];
                    [hcmd commit];
                }

                {
                    double io0 = moe_now_ms();
                    int* p_eids = miss_eids;
                    int* p_slots = miss_slots;
                    dispatch_apply((size_t)n_misses, m->io_queue, ^(size_t mi) {
                        load_expert_into_slot(m, L, p_eids[mi], p_slots[mi]);
                    });
                    ssd_acc_ms += (moe_now_ms() - io0);
                }

                if (hcmd) [hcmd waitUntilCompleted];

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
                [menc setComputePipelineState:gu_pso];
                [menc setBuffer:ly->cache_buf offset:0 atIndex:0];
                [menc setBuffer:m->buf_xn offset:0 atIndex:1];
                [menc setBuffer:m->buf_eidx offset:0 atIndex:2];
                [menc setBuffer:m->buf_h offset:0 atIndex:3];
                [menc setBytes:&margs length:sizeof(margs) atIndex:4];
                [menc dispatchThreadgroups:MTLSizeMake((NSUInteger)((m->d_ff_exp + 3) / 4), (NSUInteger)n_misses, 1) threadsPerThreadgroup:MTLSizeMake(gu_tptg, 1, 1)];
                [menc setComputePipelineState:down_pso];
                [menc setBuffer:ly->cache_buf offset:0 atIndex:0];
                [menc setBuffer:m->buf_h offset:0 atIndex:1];
                [menc setBuffer:m->buf_eidx offset:0 atIndex:2];
                [menc setBuffer:m->buf_ewt offset:0 atIndex:3];
                [menc setBuffer:m->buf_x offset:0 atIndex:4];
                [menc setBytes:&margs length:sizeof(margs) atIndex:5];
                [menc dispatchThreadgroups:MTLSizeMake(dn_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(dn_tptg, 1, 1)];
                int next_L = encode_tail_until_next_moe(m, menc, L + 1, (uint32_t)pos);
                [menc endEncoding];
                // Read clean post-Hit buf_x BEFORE mcmd mutates it, then overlap prefetch with Misses(L)+Attn(L+1)!
                schedule_next_layer_prefetch(m, next_L);
                [mcmd commit];
                [mcmd waitUntilCompleted];
                L = next_L;
            }
        }

        double tot = moe_now_ms() - t0;
        if (m->buf_last_hidden != nil) memcpy([m->buf_last_hidden contents], [m->buf_x contents], (size_t)m->d_model * 4u);
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
        case 21: return m->max_ctx;
        case 22: return m->kv_quant_q8;
        case 23: return (m->n_layers > 0 && m->layers[0].is_mla) ? 1 : 0;
        case 24: return m->mla_latent_kv;
        case 25: return m->n_kv_slots;
        case 26: return m->slot_ctx;
        case 27: return m->has_mtp_layer ? 2 : m->has_mtp;
        case 28: return m->n_bpe_merges;
        case 29: return m->pa_num_blocks;
        case 30: return m->pa_free_blocks;
        default: return 0;
    }
}

// =========================================================================
// Batched Causal Prompt Prefill (`tn_moe_prefill`), Auto Planner (`tn_moe_auto_plan`),
// Full Sampler (`tn_moe_sample_next`), Architecture Query (`tn_moe_get_arch`)!
// =========================================================================
static void ensure_prefill_bufs(MoeModel* m, int B) {
    if (m->prefill_cap >= B) return;
    int cap = B < 32 ? 32 : (B > MOE_PREFILL_MAX_B ? MOE_PREFILL_MAX_B : B);
    size_t dm = (size_t)m->d_model;
    size_t qd = (size_t)(m->n_heads * m->d_head);
    size_t kvd = (size_t)(m->n_kv * m->d_head);
    size_t ff = (size_t)(m->d_ff_dense > m->d_ff_exp ? m->d_ff_dense : m->d_ff_exp);
    m->pbuf_x   = [g_moe_dev newBufferWithLength:(size_t)cap * dm * 4 options:MTLResourceStorageModeShared];
    m->pbuf_xn  = [g_moe_dev newBufferWithLength:(size_t)cap * dm * 4 options:MTLResourceStorageModeShared];
    m->pbuf_q   = [g_moe_dev newBufferWithLength:(size_t)cap * qd * 4 options:MTLResourceStorageModeShared];
    m->pbuf_k   = [g_moe_dev newBufferWithLength:(size_t)cap * kvd * 4 options:MTLResourceStorageModeShared];
    m->pbuf_v   = [g_moe_dev newBufferWithLength:(size_t)cap * kvd * 4 options:MTLResourceStorageModeShared];
    m->pbuf_ctx = [g_moe_dev newBufferWithLength:(size_t)cap * qd * 4 options:MTLResourceStorageModeShared];
    m->pbuf_g   = [g_moe_dev newBufferWithLength:(size_t)cap * ff * 4 options:MTLResourceStorageModeShared];
    m->pbuf_u   = [g_moe_dev newBufferWithLength:(size_t)cap * ff * 4 options:MTLResourceStorageModeShared];
    m->prefill_cap = cap;
}

static void encode_gemm_batch_raw(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> buf, uint64_t off, uint32_t gtype,
                                  id<MTLBuffer> x, id<MTLBuffer> y, uint32_t out_dim, uint32_t in_dim, uint32_t B, uint32_t add_res) {
    if (B == 1u) {
        encode_gemv_raw(enc, buf, off, gtype, x, y, out_dim, in_dim, add_res);
        return;
    }
    struct { uint32_t out_dim, in_dim, batch_b, add_res; } args = { out_dim, in_dim, B, add_res };
    id<MTLComputePipelineState> pso = g_pso_gemm_q4_b;
    uint32_t tile_n = 4, tile_b = 8;
    NSUInteger tptg = 128;
    switch (gtype) {
        case TN_GGML_Q4_0:
            if (B <= 8u) { pso = g_pso_gemm_q4_sb; tile_n = 16; tile_b = 8; tptg = 128; }
            else         { pso = g_pso_gemm_q4_b;  tile_n = 32; tile_b = 32; }
            break;
        case TN_GGML_Q4_1:   pso = g_pso_gemm_q4_1_b;  tile_n = 32; tile_b = 32; break;
        case TN_GGML_Q5_0:   pso = g_pso_gemm_q5_0_b;  tile_n = 4;  tile_b = 8;  break;
        case TN_GGML_Q5_1:   pso = g_pso_gemm_q5_1_b;  tile_n = 4;  tile_b = 8;  break;
        case TN_GGML_Q8_0:   pso = g_pso_gemm_q8_b;    tile_n = 4;  tile_b = 8;  break;
        case TN_GGML_Q2_K:   pso = g_pso_gemm_q2k_b;   tile_n = 4;  tile_b = 4;  break;
        case TN_GGML_Q3_K:   pso = g_pso_gemm_q3k_b;   tile_n = 4;  tile_b = 4;  break;
        case TN_GGML_IQ4_XS: pso = g_pso_gemm_iq4xs_b; tile_n = 4;  tile_b = 4;  break;
        case TN_GGML_Q4_K:   pso = g_pso_gemm_q4k_b;   tile_n = 32; tile_b = 32; break;
        case TN_GGML_Q5_K: pso = g_pso_gemm_q5k_b;  tile_n = 4;  tile_b = 4;  break;
        case TN_GGML_Q6_K: pso = g_pso_gemm_q6k_b;  tile_n = 4;  tile_b = 4;  break;
        case TN_GGML_F16:  pso = g_pso_gemm_f16_b;  tile_n = 4;  tile_b = 8;  break;
        default:           pso = g_pso_gemm_f32_b;  tile_n = 4;  tile_b = 8;  break;
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf offset:(NSUInteger)off atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1];
    [enc setBuffer:y offset:0 atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((out_dim + tile_n - 1) / tile_n, (B + tile_b - 1) / tile_b, 1)
        threadsPerThreadgroup:MTLSizeMake(tptg, 1, 1)];
}

static void encode_add_bias_batch(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, const MoeWeightRef* b, uint32_t dim, uint32_t B) {
    if (b->buf == nil) return;
    uint32_t db[2] = { dim, B };
    [enc setComputePipelineState:g_pso_add_bias_b];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:b->buf offset:(NSUInteger)b->offset atIndex:1];
    [enc setBytes:db length:8 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(dim, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encode_layer_attn_batch(MoeModel* m, id<MTLComputeCommandEncoder> enc, int L, uint32_t start_pos, uint32_t B, id<MTLBuffer> ping_buf) {
    MoeLayer* ly = &m->layers[L];
    uint32_t dm = (uint32_t)m->d_model, qdim = (uint32_t)(m->n_heads * m->d_head), kvdim = (uint32_t)(m->n_kv * m->d_head);
    encode_rmsnorm(enc, m->pbuf_x, &ly->attn_norm, m->pbuf_xn, dm, B, m->rms_eps, 1);
    if (ping_buf != nil) {
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wq, ly->wq.ggml_type, m->pbuf_xn, m->pbuf_q, qdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wk, ly->wk.ggml_type, m->pbuf_xn, m->pbuf_k, kvdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wv, ly->wv.ggml_type, m->pbuf_xn, m->pbuf_v, kvdim, dm, B, 0);
    } else {
        encode_gemm_batch_raw(enc, ly->wq.buf, ly->wq.offset, ly->wq.ggml_type, m->pbuf_xn, m->pbuf_q, qdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ly->wk.buf, ly->wk.offset, ly->wk.ggml_type, m->pbuf_xn, m->pbuf_k, kvdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ly->wv.buf, ly->wv.offset, ly->wv.ggml_type, m->pbuf_xn, m->pbuf_v, kvdim, dm, B, 0);
    }
    if (ly->has_qkv_bias) {
        encode_add_bias_batch(enc, m->pbuf_q, &ly->bq, qdim, B);
        encode_add_bias_batch(enc, m->pbuf_k, &ly->bk, kvdim, B);
        encode_add_bias_batch(enc, m->pbuf_v, &ly->bv, kvdim, B);
    }
    if (ly->has_qk_norm) {
        if (ly->qk_norm_per_head) {
            encode_rmsnorm(enc, m->pbuf_q, &ly->q_norm, m->pbuf_q, (uint32_t)m->d_head, B * (uint32_t)m->n_heads, m->rms_eps, 1);
            encode_rmsnorm(enc, m->pbuf_k, &ly->k_norm, m->pbuf_k, (uint32_t)m->d_head, B * (uint32_t)m->n_kv, m->rms_eps, 1);
        } else {
            encode_rmsnorm(enc, m->pbuf_q, &ly->q_norm, m->pbuf_q, qdim, B, m->rms_eps, 1);
            encode_rmsnorm(enc, m->pbuf_k, &ly->k_norm, m->pbuf_k, kvdim, B, m->rms_eps, 1);
        }
    }
    struct { uint32_t n_heads, n_kv, d_head, rope_dim, start_pos; float theta; uint32_t rope_neox, has_freqs, batch_b; } rargs = {
        (uint32_t)m->n_heads, (uint32_t)m->n_kv, (uint32_t)m->d_head, (uint32_t)m->rope_dim,
        start_pos, m->rope_theta, (uint32_t)m->rope_neox, (uint32_t)m->has_rope_freqs, B
    };
    struct { uint32_t n_heads, n_kv, d_head, start_pos, batch_b, out_stride; float scale; } aargs = {
        (uint32_t)m->n_heads, (uint32_t)m->n_kv, (uint32_t)m->d_head, start_pos, B,
        (uint32_t)(((ly->wo.ne0 > 0) ? ly->wo.ne0 : (uint64_t)qdim) / (uint64_t)m->n_heads),
        (m->attn_mscale > 0.0f ? m->attn_mscale : 1.0f) / sqrtf((float)m->d_head)
    };
    NSUInteger kv_slot_off = m->kv_quant_q8
        ? ((NSUInteger)m->cur_kv_slot * (NSUInteger)m->slot_ctx * (NSUInteger)m->n_kv * (NSUInteger)(m->d_head >> 5) * 34u)
        : ((NSUInteger)m->cur_kv_slot * (NSUInteger)m->slot_ctx * (NSUInteger)(m->n_kv * m->d_head) * 2u);
    if (m->kv_quant_q8) {
        [enc setComputePipelineState:g_pso_rope_kv_q8_b];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:m->pbuf_k offset:0 atIndex:1]; [enc setBuffer:m->pbuf_v offset:0 atIndex:2];
        [enc setBuffer:ly->k_cache offset:kv_slot_off atIndex:3]; [enc setBuffer:ly->v_cache offset:kv_slot_off atIndex:4];
        [enc setBuffer:(m->has_rope_freqs ? m->rope_freqs.buf : m->buf_argmax) offset:(m->has_rope_freqs ? (NSUInteger)m->rope_freqs.offset : 0) atIndex:5];
        [enc setBytes:&rargs length:sizeof(rargs) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc setComputePipelineState:g_pso_flash_prefill_q8];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:ly->k_cache offset:kv_slot_off atIndex:1]; [enc setBuffer:ly->v_cache offset:kv_slot_off atIndex:2];
        [enc setBuffer:m->pbuf_ctx offset:0 atIndex:3]; [enc setBytes:&aargs length:sizeof(aargs) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        [enc setComputePipelineState:g_pso_rope_kv_b];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:m->pbuf_k offset:0 atIndex:1]; [enc setBuffer:m->pbuf_v offset:0 atIndex:2];
        [enc setBuffer:ly->k_cache offset:kv_slot_off atIndex:3]; [enc setBuffer:ly->v_cache offset:kv_slot_off atIndex:4];
        [enc setBuffer:(m->has_rope_freqs ? m->rope_freqs.buf : m->buf_argmax) offset:(m->has_rope_freqs ? (NSUInteger)m->rope_freqs.offset : 0) atIndex:5];
        [enc setBytes:&rargs length:sizeof(rargs) atIndex:6];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)(m->d_head / 2), (NSUInteger)m->n_heads, B) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        [enc setComputePipelineState:g_pso_flash_prefill];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:ly->k_cache offset:kv_slot_off atIndex:1]; [enc setBuffer:ly->v_cache offset:kv_slot_off atIndex:2];
        [enc setBuffer:m->pbuf_ctx offset:0 atIndex:3]; [enc setBytes:&aargs length:sizeof(aargs) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    if (ping_buf != nil) encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wo, ly->wo.ggml_type, m->pbuf_ctx, m->pbuf_x, dm, qdim, B, 1);
    else encode_gemm_batch_raw(enc, ly->wo.buf, ly->wo.offset, ly->wo.ggml_type, m->pbuf_ctx, m->pbuf_x, dm, qdim, B, 1);
    encode_rmsnorm(enc, m->pbuf_x, &ly->ffn_norm, m->pbuf_xn, dm, B, m->rms_eps, 1);
}

static void encode_dense_ffn_batch(MoeModel* m, id<MTLComputeCommandEncoder> enc,
                                   id<MTLBuffer> gb, uint64_t go, uint32_t gt,
                                   id<MTLBuffer> ub, uint64_t uo, uint32_t ut,
                                   id<MTLBuffer> db, uint64_t doff, uint32_t dt, uint32_t d_ff, uint32_t B) {
    uint32_t dm = (uint32_t)m->d_model;
    encode_gemm_batch_raw(enc, gb, go, gt, m->pbuf_xn, m->pbuf_g, d_ff, dm, B, 0);
    encode_gemm_batch_raw(enc, ub, uo, ut, m->pbuf_xn, m->pbuf_u, d_ff, dm, B, 0);
    uint32_t tot = B * d_ff;
    [enc setComputePipelineState:g_pso_swiglu_mul];
    [enc setBuffer:m->pbuf_g offset:0 atIndex:0]; [enc setBuffer:m->pbuf_u offset:0 atIndex:1]; [enc setBytes:&tot length:4 atIndex:2];
    [enc dispatchThreads:MTLSizeMake(tot, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    encode_gemm_batch_raw(enc, db, doff, dt, m->pbuf_g, m->pbuf_x, dm, d_ff, B, 1);
}

// Batched Causal Prompt Prefill: reads weights ONCE per chunk of up to 32 tokens (8-30x faster TTFT!)
int64_t tn_moe_prefill(int64_t handle, int64_t tok_ids_ptr, int64_t n_toks, int64_t start_pos) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !tok_ids_ptr || n_toks <= 0) return 0;
    MoeModel* m = &g_moe[handle];
    const int64_t* ids = (const int64_t*)(uintptr_t)tok_ids_ptr;
    // For tiny prompts (<=2) or MLA layers, direct step path is optimal
    if (n_toks <= 2 || !m->dense_model || (m->n_layers > 0 && m->layers[0].is_mla)) {
        int64_t last = 0;
        for (int64_t i = 0; i < n_toks; i++) last = tn_moe_step(handle, ids[i], start_pos + i);
        return last;
    }
    @autoreleasepool {
        double t0 = moe_now_ms();
        int64_t last_tok = 0;
        int chunk_sz = 32;
        ensure_prefill_bufs(m, chunk_sz);
        int64_t off = 0;
        while (off < n_toks) {
            uint32_t B = (uint32_t)((n_toks - off > chunk_sz) ? chunk_sz : (n_toks - off));
            uint32_t cur_pos = (uint32_t)(start_pos + off);
            int is_final_chunk = (off + B == n_toks);

            // 1. Embed all B tokens into pbuf_x[0..B-1]
            float* px = (float*)[m->pbuf_x contents];
            for (uint32_t bi = 0; bi < B; bi++) {
                id<MTLCommandBuffer> ecmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> eenc = [ecmd computeCommandEncoder];
                encode_embed_step(m, eenc, (uint32_t)ids[off + bi]);
                [eenc endEncoding]; [ecmd commit]; [ecmd waitUntilCompleted];
                memcpy(px + (size_t)bi * (size_t)m->d_model, [m->buf_x contents], (size_t)m->d_model * 4u);
            }

            // 2. Run all 80 Dense layers in batched chunks (Resident = 1 CB; Streaming = 1 ring read per B tokens!!)
            if (m->resident_mode) {
                id<MTLCommandBuffer> cmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                for (int L = 0; L < m->n_layers; L++) {
                    MoeLayer* ly = &m->layers[L];
                    encode_layer_attn_batch(m, enc, L, cur_pos, B, nil);
                    encode_dense_ffn_batch(m, enc, ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                                           ly->ffn_up.buf, ly->ffn_up.offset, ly->ffn_up.ggml_type,
                                           ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
                }
                [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
            } else {
                int first_stream = m->cache_slots_per_layer;
                int n_ring = m->dense_pin_all_attn ? 4 : 2;
                if (first_stream > 0) {
                    id<MTLCommandBuffer> pcmd = [g_moe_q commandBuffer];
                    id<MTLComputeCommandEncoder> penc = [pcmd computeCommandEncoder];
                    for (int L = 0; L < first_stream && L < m->n_layers; L++) {
                        MoeLayer* ly = &m->layers[L];
                        encode_layer_attn_batch(m, penc, L, cur_pos, B, nil);
                        encode_dense_ffn_batch(m, penc, ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                                               ly->ffn_up.buf, ly->ffn_up.offset, ly->ffn_up.ggml_type,
                                               ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
                    }
                    [penc endEncoding]; [pcmd commit]; [pcmd waitUntilCompleted];
                }
                dispatch_semaphore_t sem_empty = nil, sem_full = nil;
                if (m->enable_overlap && first_stream < m->n_layers) {
                    sem_empty = dispatch_semaphore_create(n_ring);
                    sem_full  = dispatch_semaphore_create(0);
                    int nl_cap = m->n_layers;
                    dispatch_group_async(m->dense_io_grp, m->ring_prod_q, ^{
                        for (int pL = first_stream; pL < nl_cap; pL++) {
                            dispatch_semaphore_wait(sem_empty, DISPATCH_TIME_FOREVER);
                            load_dense_layer_into_ping(m, pL, (pL - first_stream) % n_ring, 1);
                            dispatch_semaphore_signal(sem_full);
                        }
                    });
                }
                for (int L = first_stream; L < m->n_layers; L++) {
                    MoeLayer* ly = &m->layers[L];
                    int cur = (L - first_stream) % n_ring;
                    if (m->enable_overlap) dispatch_semaphore_wait(sem_full, DISPATCH_TIME_FOREVER);
                    else load_dense_layer_into_ping(m, L, cur, 0);
                    id<MTLCommandBuffer> lcmd = [g_moe_q commandBuffer];
                    id<MTLComputeCommandEncoder> lenc = [lcmd computeCommandEncoder];
                    encode_layer_attn_batch(m, lenc, L, cur_pos, B, m->dense_pin_all_attn ? nil : m->dense_ping[cur]);
                    encode_dense_ffn_batch(m, lenc, m->dense_ping[cur], m->dense_off_gate, ly->ffn_gate.ggml_type,
                                           m->dense_ping[cur], m->dense_off_up, ly->ffn_up.ggml_type,
                                           m->dense_ping[cur], m->dense_off_down, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
                    [lenc endEncoding]; [lcmd commit]; [lcmd waitUntilCompleted];
                    if (m->enable_overlap) dispatch_semaphore_signal(sem_empty);
                }
                if (m->enable_overlap) dispatch_group_wait(m->dense_io_grp, DISPATCH_TIME_FOREVER);
            }

            if (is_final_chunk) {
                memcpy([m->buf_x contents], px + (size_t)(B - 1u) * (size_t)m->d_model, (size_t)m->d_model * 4u);
                id<MTLCommandBuffer> fcmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> fenc = [fcmd computeCommandEncoder];
                encode_final_lm_head(m, fenc);
                [fenc endEncoding]; [fcmd commit]; [fcmd waitUntilCompleted];
                if (m->buf_last_hidden != nil) memcpy([m->buf_last_hidden contents], [m->buf_x contents], (size_t)m->d_model * 4u);
                last_tok = (int64_t)((uint32_t*)[m->buf_argmax contents])[0];
            }
            off += B;
        }
        m->last_total_ms = moe_now_ms() - t0;
        return last_tok;
    }
}

// Auto Apple Silicon VRAM/RAM Budget Planner: inspects GGUF headers (<2ms) & computes optimal streaming parameters!
// out_plan[9]: [0]=cache_slots, [1]=use_nocache, [2]=kv_q8, [3]=prefill_chunk, [4]=est_vram_bytes, [5]=disk_bytes, [6]=n_layers, [7]=n_experts, [8]=d_model
int64_t tn_moe_auto_plan(int64_t path_cstr, double max_vram_gb, int64_t max_ctx, int64_t kv_mode, int64_t out_plan_ptr) {
    if (!moe_init_metal() || !path_cstr || !out_plan_ptr) return 0;
    const char* path0 = (const char*)(uintptr_t)path_cstr;
    MoeModel probe; memset((void*)&probe, 0, sizeof(probe));
    MoeTensorInfo* tlist = NULL; uint64_t tcount = 0; int split_count = 1;
    if (!parse_gguf_shard(path0, 0, 1, &probe, &tlist, &tcount, &split_count)) { free(tlist); return 0; }
    if (split_count > MOE_MAX_SHARDS) split_count = MOE_MAX_SHARDS;
    uint64_t disk_bytes = 0;
    for (int s = 0; s < split_count; s++) {
        char sp[1024];
        if (s == 0) strncpy(sp, path0, sizeof(sp) - 1);
        else {
            const char* mk = strstr(path0, "-00001-of-");
            if (!mk) break;
            snprintf(sp, sizeof(sp), "%.*s-%05d-of-%s", (int)(mk - path0), path0, s + 1, mk + 10);
            parse_gguf_shard(sp, s, 0, &probe, &tlist, &tcount, NULL);
        }
        struct stat st;
        if (stat(sp, &st) == 0) disk_bytes += (uint64_t)st.st_size;
    }
    if (probe.vocab_tokens) { for (int i = 0; i < probe.vocab_size; i++) free(probe.vocab_tokens[i]); free(probe.vocab_tokens); }
    if (probe.bpe_merges) { for (int i = 0; i < probe.n_bpe_merges; i++) free(probe.bpe_merges[i]); free(probe.bpe_merges); }
    if (probe.nextn_layers > 0 && probe.n_layers > probe.nextn_layers) probe.n_layers -= probe.nextn_layers;
    if (probe.n_kv <= 0) probe.n_kv = probe.n_heads;
    if (probe.d_head <= 0 && probe.n_heads > 0) probe.d_head = probe.d_model / probe.n_heads;
    int dense = (probe.n_experts <= 0);
    uint64_t core_w = 0, ffn_layer_b = 0, exp_layer_b = 0;
    int n_ffn_cnt = 0;
    for (uint64_t i = 0; i < tcount; i++) {
        const char* nm = tlist[i].name;
        if (strstr(nm, "token_embd")) continue; // zero-VRAM pread!
        if (strstr(nm, "ffn_gate_exps") || strstr(nm, "ffn_up_exps") || strstr(nm, "ffn_down_exps")) {
            exp_layer_b += tlist[i].byte_size;
        } else if (dense && (strstr(nm, "ffn_gate.weight") || strstr(nm, "ffn_up.weight") || strstr(nm, "ffn_down.weight"))) {
            ffn_layer_b += tlist[i].byte_size;
            if (strstr(nm, "ffn_gate.weight")) n_ffn_cnt++;
        } else {
            core_w += tlist[i].byte_size;
        }
    }
    free(tlist);

    uint64_t sys_ram = 0; size_t rsz = sizeof(sys_ram);
    sysctlbyname("hw.memsize", &sys_ram, &rsz, NULL, 0);
    uint64_t gpu_rec = (uint64_t)[g_moe_dev recommendedMaxWorkingSetSize];
    uint64_t budget = (max_vram_gb > 0.5) ? (uint64_t)(max_vram_gb * 1024.0 * 1024.0 * 1024.0) : (uint64_t)((double)gpu_rec * 0.78);
    if (budget > (uint64_t)((double)sys_ram * 0.80) && sys_ram > 0) budget = (uint64_t)((double)sys_ram * 0.80);

    int64_t ctx = (max_ctx > 0) ? max_ctx : MOE_DEFAULT_SEQ;
    if (ctx > MOE_MAX_SEQ) ctx = MOE_MAX_SEQ;
    int kv_q8 = (kv_mode == 1) ? 1 : (kv_mode == 0 ? 0 : (ctx > 2048 ? 1 : 0));
    uint64_t kv_b = kv_q8
        ? ((uint64_t)probe.n_layers * 2ull * (uint64_t)ctx * (uint64_t)probe.n_kv * (uint64_t)(probe.d_head >> 5) * 34ull)
        : ((uint64_t)probe.n_layers * 2ull * (uint64_t)ctx * (uint64_t)(probe.n_kv * probe.d_head) * 2ull);
    uint64_t base_vram = core_w + kv_b + (256ull * 1024 * 1024);

    int64_t slots = -1, nocache = 0;
    uint64_t est_vram = disk_bytes + kv_b;
    if (disk_bytes + kv_b + (512ull * 1024 * 1024) <= budget) {
        slots = dense ? -1 : probe.n_experts; // Full Resident Zero-Copy!
        nocache = 0;
        est_vram = disk_bytes + kv_b;
    } else if (dense) {
        uint64_t one_ffn = (n_ffn_cnt > 0) ? (ffn_layer_b / (uint64_t)n_ffn_cnt) : (370ull * 1024 * 1024);
        uint64_t ring4 = one_ffn * 4ull;
        if (budget > base_vram + ring4 + one_ffn) {
            slots = (int64_t)((budget - base_vram - ring4) / one_ffn);
            if (slots >= probe.n_layers) slots = probe.n_layers - 1;
            est_vram = base_vram + ring4 + (uint64_t)slots * one_ffn;
        } else {
            slots = 0; // Ultra-low 2-layer Ping-Pong (<2.5 GiB!)
            est_vram = (uint64_t)(2.5 * 1024 * 1024 * 1024);
        }
        nocache = (disk_bytes > (uint64_t)((double)budget * 0.90)) ? 1 : 0;
    } else {
        uint64_t per_slot_all = (probe.n_experts > 0) ? (exp_layer_b / (uint64_t)probe.n_experts) : (400ull * 1024 * 1024);
        if (budget > base_vram + per_slot_all * (uint64_t)(probe.top_k + 2)) {
            slots = (int64_t)((budget - base_vram) / per_slot_all);
            if (slots < probe.top_k + 2) slots = probe.top_k + 2;
            if (slots > probe.n_experts) slots = probe.n_experts;
        } else {
            slots = probe.top_k;
        }
        est_vram = base_vram + (uint64_t)slots * per_slot_all;
        nocache = (disk_bytes > (uint64_t)((double)budget * 0.90)) ? 1 : 0;
    }
    int64_t* out = (int64_t*)(uintptr_t)out_plan_ptr;
    out[0] = slots; out[1] = nocache; out[2] = kv_q8; out[3] = 32;
    out[4] = (int64_t)est_vram; out[5] = (int64_t)disk_bytes;
    out[6] = probe.n_layers; out[7] = probe.n_experts; out[8] = probe.d_model;
    return 1;
}

typedef struct { int id; float val; } MoeCand;
static int cmp_cand_desc(const void* a, const void* b) {
    float va = ((const MoeCand*)a)->val, vb = ((const MoeCand*)b)->val;
    return (va < vb) ? 1 : (va > vb ? -1 : 0);
}

// Full Sampler (temperature + top_k + top_p + min_p + repetition_penalty); pass tok_id < 0 to sample from existing logits (e.g. post-prefill!)
int64_t tn_moe_sample_next(
    int64_t handle, int64_t tok_id, int64_t pos,
    double temp, int64_t top_k, double top_p, double min_p, double repeat_penalty,
    int64_t recent_ptr, int64_t n_recent, int64_t seed
) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return 0;
    MoeModel* m = &g_moe[handle];
    if (tok_id >= 0) {
        int64_t argmax_id = tn_moe_step(handle, tok_id, pos);
        if (temp <= 1e-5 && (repeat_penalty <= 1.0001 || n_recent <= 0)) return argmax_id;
    }
    const float* raw_logits = (const float*)[m->buf_logits contents];
    int V = m->vocab_size;
    float* lg = (float*)malloc((size_t)V * sizeof(float));
    memcpy(lg, raw_logits, (size_t)V * sizeof(float));
    if (repeat_penalty > 1.0001 && recent_ptr && n_recent > 0) {
        const int64_t* rec = (const int64_t*)(uintptr_t)recent_ptr;
        for (int64_t i = 0; i < n_recent; i++) {
            int64_t r = rec[i];
            if (r >= 0 && r < V) lg[r] = (lg[r] > 0.0f) ? (lg[r] / (float)repeat_penalty) : (lg[r] * (float)repeat_penalty);
        }
    }
    if (temp <= 1e-5) {
        int best = 0; float bv = lg[0];
        for (int i = 1; i < V; i++) if (lg[i] > bv) { bv = lg[i]; best = i; }
        free(lg); return (int64_t)best;
    }
    int K = (top_k > 0 && top_k < V) ? (int)top_k : 64;
    if (K > V) K = V;
    MoeCand* cands = (MoeCand*)malloc((size_t)K * sizeof(MoeCand));
    int nc = 0; float min_in = -1e30f; int min_idx = 0;
    for (int i = 0; i < V; i++) {
        if (nc < K) {
            cands[nc++] = (MoeCand){ i, lg[i] };
            if (nc == K) {
                min_in = cands[0].val; min_idx = 0;
                for (int j = 1; j < K; j++) if (cands[j].val < min_in) { min_in = cands[j].val; min_idx = j; }
            }
        } else if (lg[i] > min_in) {
            cands[min_idx] = (MoeCand){ i, lg[i] };
            min_in = cands[0].val; min_idx = 0;
            for (int j = 1; j < K; j++) if (cands[j].val < min_in) { min_in = cands[j].val; min_idx = j; }
        }
    }
    free(lg);
    qsort(cands, (size_t)nc, sizeof(MoeCand), cmp_cand_desc);
    float inv_t = 1.0f / (float)temp;
    float mx = cands[0].val * inv_t;
    float psum = 0.0f;
    for (int i = 0; i < nc; i++) { cands[i].val = expf(cands[i].val * inv_t - mx); psum += cands[i].val; }
    for (int i = 0; i < nc; i++) cands[i].val /= (psum > 1e-12f ? psum : 1.0f);
    // Min-P + Top-P nucleus cutoff
    float p_thresh = (min_p > 0.0) ? (cands[0].val * (float)min_p) : 0.0f;
    float cum = 0.0f; int keep = 0;
    for (int i = 0; i < nc; i++) {
        if (i > 0 && cands[i].val < p_thresh) break;
        cum += cands[i].val; keep++;
        if (top_p > 0.0 && top_p < 1.0 && cum >= (float)top_p) break;
    }
    uint64_t x = (uint64_t)seed ^ 0x9E3779B97F4A7C15ull;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    double u = (double)((x * 2685821657736338717ull) >> 11) * (1.0 / 9007199254740992.0);
    float target = (float)(u * (double)cum), acc = 0.0f;
    int chosen = cands[0].id;
    for (int i = 0; i < keep; i++) { acc += cands[i].val; if (acc >= target) { chosen = cands[i].id; break; } }
    free(cands);
    return (int64_t)chosen;
}

int64_t tn_moe_get_arch(int64_t handle) {
    static char empty[2] = "";
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return (int64_t)(uintptr_t)empty;
    return (int64_t)(uintptr_t)g_moe[handle].arch;
}

// Switch active Paged/Partitioned KV-Cache slot (`0..n_kv_slots-1`), returns per-slot max context!
int64_t tn_moe_kv_slot_set(int64_t handle, int64_t slot_id) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return 0;
    MoeModel* m = &g_moe[handle];
    if (slot_id < 0 || slot_id >= m->n_kv_slots) slot_id = 0;
    m->cur_kv_slot = (int)slot_id;
    return (int64_t)m->slot_ctx;
}

// Copy current vocab_size FP32 logits into caller buffer (enables JSON-Schema/Tools grammar masking in `tenon llm`!)
int64_t tn_moe_copy_logits(int64_t handle, int64_t dst_f32_ptr) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !dst_f32_ptr) return 0;
    MoeModel* m = &g_moe[handle];
    memcpy((void*)(uintptr_t)dst_f32_ptr, [m->buf_logits contents], (size_t)m->vocab_size * sizeof(float));
    return (int64_t)m->vocab_size;
}

// Lossless Batched Speculative Verification: evaluates B candidate draft tokens in ONE AMX forward pass
// and writes target argmax next-token IDs for ALL B positions `[0..B-1]` into `out_argmax_ptr`!
int64_t tn_moe_verify_batch(int64_t handle, int64_t draft_ids_ptr, int64_t n_draft, int64_t start_pos, int64_t out_argmax_ptr) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !draft_ids_ptr || !out_argmax_ptr || n_draft <= 0) return 0;
    MoeModel* m = &g_moe[handle];
    const int64_t* ids = (const int64_t*)(uintptr_t)draft_ids_ptr;
    int64_t* out_ids = (int64_t*)(uintptr_t)out_argmax_ptr;
    if (n_draft <= 1 || !m->dense_model || (m->n_layers > 0 && m->layers[0].is_mla) || n_draft > 32) {
        for (int64_t i = 0; i < n_draft; i++) out_ids[i] = tn_moe_step(handle, ids[i], start_pos + i);
        return n_draft;
    }
    @autoreleasepool {
        double t0 = moe_now_ms();
        uint32_t B = (uint32_t)n_draft;
        uint32_t cur_pos = (uint32_t)start_pos;
        ensure_prefill_bufs(m, (int)B);
        if (m->batch_logits_cap < (int)B) {
            int ncap = (int)(B < 16 ? 16 : B);
            m->buf_batch_logits = [g_moe_dev newBufferWithLength:(size_t)ncap * (size_t)m->vocab_size * 4 options:MTLResourceStorageModeShared];
            m->batch_logits_cap = ncap;
        }
        float* px = (float*)[m->pbuf_x contents];
        if (m->resident_mode) {
            id<MTLCommandBuffer> cmd = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            for (uint32_t bi = 0; bi < B; bi++) {
                struct { uint32_t tok_id, d_model, ggml_type; } eargs = { (uint32_t)ids[bi], (uint32_t)m->d_model, m->tok_embd.ggml_type };
                [enc setComputePipelineState:g_pso_embed];
                [enc setBuffer:m->tok_embd.buf offset:(NSUInteger)m->tok_embd.offset atIndex:0];
                [enc setBuffer:m->pbuf_x offset:(NSUInteger)((size_t)bi * (size_t)m->d_model * 4u) atIndex:1];
                [enc setBytes:&eargs length:sizeof(eargs) atIndex:2];
                [enc dispatchThreads:MTLSizeMake((NSUInteger)m->d_model, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            for (int L = 0; L < m->n_layers; L++) {
                MoeLayer* ly = &m->layers[L];
                encode_layer_attn_batch(m, enc, L, cur_pos, B, nil);
                encode_dense_ffn_batch(m, enc, ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                                       ly->ffn_up.buf, ly->ffn_up.offset, ly->ffn_up.ggml_type,
                                       ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
            }
            encode_rmsnorm(enc, m->pbuf_x, &m->out_norm, m->pbuf_xn, (uint32_t)m->d_model, B, m->rms_eps, 1);
            encode_gemm_batch_raw(enc, m->lm_head.buf, m->lm_head.offset, m->lm_head.ggml_type,
                                  m->pbuf_xn, m->buf_batch_logits, (uint32_t)m->vocab_size, (uint32_t)m->d_model, B, 0);
            [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        } else {
            for (uint32_t bi = 0; bi < B; bi++) {
                id<MTLCommandBuffer> ecmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> eenc = [ecmd computeCommandEncoder];
                encode_embed_step(m, eenc, (uint32_t)ids[bi]);
                [eenc endEncoding]; [ecmd commit]; [ecmd waitUntilCompleted];
                memcpy(px + (size_t)bi * (size_t)m->d_model, [m->buf_x contents], (size_t)m->d_model * 4u);
            }
            // Streaming dense speculative verify: 1 NVMe ring read amortized across all B draft tokens!
            int first_stream = m->cache_slots_per_layer;
            int n_ring = m->dense_pin_all_attn ? 4 : 2;
            if (first_stream > 0) {
                id<MTLCommandBuffer> pcmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> penc = [pcmd computeCommandEncoder];
                for (int L = 0; L < first_stream && L < m->n_layers; L++) {
                    MoeLayer* ly = &m->layers[L];
                    encode_layer_attn_batch(m, penc, L, cur_pos, B, nil);
                    encode_dense_ffn_batch(m, penc, ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                                           ly->ffn_up.buf, ly->ffn_up.offset, ly->ffn_up.ggml_type,
                                           ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
                }
                [penc endEncoding]; [pcmd commit]; [pcmd waitUntilCompleted];
            }
            for (int L = first_stream; L < m->n_layers; L++) {
                MoeLayer* ly = &m->layers[L];
                int cur = (L - first_stream) % n_ring;
                load_dense_layer_into_ping(m, L, cur, m->enable_overlap);
                id<MTLCommandBuffer> lcmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> lenc = [lcmd computeCommandEncoder];
                encode_layer_attn_batch(m, lenc, L, cur_pos, B, m->dense_pin_all_attn ? nil : m->dense_ping[cur]);
                encode_dense_ffn_batch(m, lenc, m->dense_ping[cur], m->dense_off_gate, ly->ffn_gate.ggml_type,
                                       m->dense_ping[cur], m->dense_off_up, ly->ffn_up.ggml_type,
                                       m->dense_ping[cur], m->dense_off_down, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
                [lenc endEncoding]; [lcmd commit]; [lcmd waitUntilCompleted];
            }
            id<MTLCommandBuffer> fcmd = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> fenc = [fcmd computeCommandEncoder];
            encode_rmsnorm(fenc, m->pbuf_x, &m->out_norm, m->pbuf_xn, (uint32_t)m->d_model, B, m->rms_eps, 1);
            encode_gemm_batch_raw(fenc, m->lm_head.buf, m->lm_head.offset, m->lm_head.ggml_type,
                                  m->pbuf_xn, m->buf_batch_logits, (uint32_t)m->vocab_size, (uint32_t)m->d_model, B, 0);
            [fenc endEncoding]; [fcmd commit]; [fcmd waitUntilCompleted];
        }
        // Compute target argmax for ALL B positions from batched logits (1 LM-head weight read instead of B!)
        const float* bl = (const float*)[m->buf_batch_logits contents];
        for (uint32_t bi = 0; bi < B; bi++) {
            const float* row = bl + (size_t)bi * (size_t)m->vocab_size;
            int best = 0; float bv = row[0];
            for (int v = 1; v < m->vocab_size; v++) if (row[v] > bv) { bv = row[v]; best = v; }
            out_ids[bi] = (int64_t)best;
        }
        memcpy([m->buf_logits contents], bl + (size_t)(B - 1u) * (size_t)m->vocab_size, (size_t)m->vocab_size * 4u);
        m->last_total_ms = moe_now_ms() - t0;
        return n_draft;
    }
}

// =========================================================================
// Tenon v6: PagedAttention 16-Token Block-Table KV + Zero-Copy COW Fork + Continuous Batching (`step_batch`)
// =========================================================================
static int pa_alloc_block(MoeModel* m) {
    for (int pb = 0; pb < m->pa_num_blocks; pb++) {
        if (m->pa_refcnt[pb] == 0) {
            m->pa_refcnt[pb] = 1;
            m->pa_free_blocks--;
            return pb;
        }
    }
    return -1;
}

static size_t pa_kv_block_bytes(MoeModel* m) {
    return m->kv_quant_q8
        ? ((size_t)PA_BLOCK_SIZE * (size_t)m->n_kv * (size_t)(m->d_head >> 5) * 34u)
        : ((size_t)PA_BLOCK_SIZE * (size_t)(m->n_kv * m->d_head) * 2u);
}

static int pa_ensure_slot_pos(MoeModel* m, int slot, int pos) {
    if (slot < 0 || slot >= PA_MAX_SEQS || pos < 0) return 0;
    int lb = pos / PA_BLOCK_SIZE;
    if (lb >= PA_MAX_LOGICAL_BLKS) return 0;
    int32_t pb = m->pa_block_table[slot][lb];
    if (pb < 0) {
        pb = pa_alloc_block(m);
        if (pb < 0) return 0;
        m->pa_block_table[slot][lb] = pb;
    } else if (m->pa_refcnt[pb] > 1) {
        // Copy-on-Write split for shared partial tail block!
        int new_pb = pa_alloc_block(m);
        if (new_pb < 0) return 0;
        size_t blk_b = pa_kv_block_bytes(m);
        for (int L = 0; L < m->n_layers; L++) {
            MoeLayer* ly = &m->layers[L];
            if (ly->k_cache != nil) {
                uint8_t* kp = (uint8_t*)[ly->k_cache contents];
                memcpy(kp + (size_t)new_pb * blk_b, kp + (size_t)pb * blk_b, blk_b);
            }
            if (ly->v_cache != nil) {
                uint8_t* vp = (uint8_t*)[ly->v_cache contents];
                memcpy(vp + (size_t)new_pb * blk_b, vp + (size_t)pb * blk_b, blk_b);
            }
        }
        m->pa_refcnt[pb]--;
        m->pa_block_table[slot][lb] = new_pb;
        m->pa_cow_copies++;
    }
    if (pos + 1 > m->pa_seq_len[slot]) m->pa_seq_len[slot] = pos + 1;
    return 1;
}

static void pa_sync_bt(MoeModel* m) {
    if (m->buf_pa_bt != nil) {
        memcpy([m->buf_pa_bt contents], m->pa_block_table, (size_t)PA_MAX_SEQS * PA_MAX_LOGICAL_BLKS * sizeof(int32_t));
    }
}

int64_t tn_moe_paged_kv_info(int64_t handle, int64_t field_id) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return 0;
    MoeModel* m = &g_moe[handle];
    switch (field_id) {
        case 0: return PA_BLOCK_SIZE;
        case 1: return m->pa_num_blocks;
        case 2: return m->pa_free_blocks;
        case 3: return m->pa_num_blocks - m->pa_free_blocks;
        case 4: return m->pa_cow_forks;
        case 5: return m->pa_cow_copies;
        case 6: return PA_MAX_SEQS;
        default: return 0;
    }
}

int64_t tn_moe_paged_seq_free(int64_t handle, int64_t slot_id) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return 0;
    MoeModel* m = &g_moe[handle];
    if (slot_id < 0 || slot_id >= PA_MAX_SEQS) return 0;
    int freed = 0;
    for (int lb = 0; lb < PA_MAX_LOGICAL_BLKS; lb++) {
        int32_t pb = m->pa_block_table[slot_id][lb];
        if (pb >= 0 && pb < m->pa_num_blocks) {
            if (m->pa_refcnt[pb] > 0) {
                m->pa_refcnt[pb]--;
                if (m->pa_refcnt[pb] == 0) { m->pa_free_blocks++; freed++; }
            }
            m->pa_block_table[slot_id][lb] = -1;
        }
    }
    m->pa_seq_len[slot_id] = 0;
    pa_sync_bt(m);
    return (int64_t)freed;
}

// Zero-Copy Copy-On-Write Prefix Fork: shares ALL `ceil(prefix_len/16)` physical KV blocks from `src_slot` -> `dst_slot`!
int64_t tn_moe_paged_seq_fork(int64_t handle, int64_t src_slot, int64_t dst_slot, int64_t prefix_len) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return 0;
    MoeModel* m = &g_moe[handle];
    if (src_slot < 0 || src_slot >= PA_MAX_SEQS || dst_slot < 0 || dst_slot >= PA_MAX_SEQS || src_slot == dst_slot) return 0;
    if (prefix_len <= 0) { tn_moe_paged_seq_free(handle, dst_slot); return 0; }
    tn_moe_paged_seq_free(handle, dst_slot);
    int n_blks = (int)((prefix_len + PA_BLOCK_SIZE - 1) / PA_BLOCK_SIZE);
    if (n_blks > PA_MAX_LOGICAL_BLKS) n_blks = PA_MAX_LOGICAL_BLKS;
    int shared = 0;
    for (int lb = 0; lb < n_blks; lb++) {
        int32_t pb = m->pa_block_table[src_slot][lb];
        if (pb < 0 || pb >= m->pa_num_blocks) break;
        m->pa_block_table[dst_slot][lb] = pb;
        m->pa_refcnt[pb]++;
        shared++;
    }
    m->pa_seq_len[dst_slot] = (int)prefix_len;
    m->pa_cow_forks++;
    pa_sync_bt(m);
    return (int64_t)shared;
}

static void encode_layer_attn_paged_batch(MoeModel* m, id<MTLComputeCommandEncoder> enc, int L, uint32_t B, id<MTLBuffer> ping_buf) {
    MoeLayer* ly = &m->layers[L];
    uint32_t dm = (uint32_t)m->d_model, qdim = (uint32_t)(m->n_heads * m->d_head), kvdim = (uint32_t)(m->n_kv * m->d_head);
    uint32_t wo_in = (ly->wo.ne0 > 0) ? (uint32_t)ly->wo.ne0 : qdim;
    uint32_t out_stride = wo_in / (uint32_t)m->n_heads;
    float attn_scale = (m->attn_mscale > 0.0f ? m->attn_mscale : 1.0f) / sqrtf((float)m->d_head);
    encode_rmsnorm(enc, m->pbuf_x, &ly->attn_norm, m->pbuf_xn, dm, B, m->rms_eps, 1);
    if (ping_buf != nil) {
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wq, ly->wq.ggml_type, m->pbuf_xn, m->pbuf_q, qdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wk, ly->wk.ggml_type, m->pbuf_xn, m->pbuf_k, kvdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wv, ly->wv.ggml_type, m->pbuf_xn, m->pbuf_v, kvdim, dm, B, 0);
    } else {
        encode_gemm_batch_raw(enc, ly->wq.buf, ly->wq.offset, ly->wq.ggml_type, m->pbuf_xn, m->pbuf_q, qdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ly->wk.buf, ly->wk.offset, ly->wk.ggml_type, m->pbuf_xn, m->pbuf_k, kvdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ly->wv.buf, ly->wv.offset, ly->wv.ggml_type, m->pbuf_xn, m->pbuf_v, kvdim, dm, B, 0);
    }
    if (ly->has_qkv_bias) {
        encode_add_bias_batch(enc, m->pbuf_q, &ly->bq, qdim, B);
        encode_add_bias_batch(enc, m->pbuf_k, &ly->bk, kvdim, B);
        encode_add_bias_batch(enc, m->pbuf_v, &ly->bv, kvdim, B);
    }
    if (ly->has_qk_norm) {
        if (ly->qk_norm_per_head) {
            encode_rmsnorm(enc, m->pbuf_q, &ly->q_norm, m->pbuf_q, (uint32_t)m->d_head, B * (uint32_t)m->n_heads, m->rms_eps, 1);
            encode_rmsnorm(enc, m->pbuf_k, &ly->k_norm, m->pbuf_k, (uint32_t)m->d_head, B * (uint32_t)m->n_kv, m->rms_eps, 1);
        } else {
            encode_rmsnorm(enc, m->pbuf_q, &ly->q_norm, m->pbuf_q, qdim, B, m->rms_eps, 1);
            encode_rmsnorm(enc, m->pbuf_k, &ly->k_norm, m->pbuf_k, kvdim, B, m->rms_eps, 1);
        }
    }
    struct { uint32_t nh, nkv, dh, rd; float theta; uint32_t rn, hf, bb, mb; } rargs = {
        (uint32_t)m->n_heads, (uint32_t)m->n_kv, (uint32_t)m->d_head, (uint32_t)m->rope_dim,
        m->rope_theta, (uint32_t)m->rope_neox, (uint32_t)m->has_rope_freqs, B, (uint32_t)PA_MAX_LOGICAL_BLKS
    };
    struct { uint32_t nh, nkv, dh, bb, os; float sc; uint32_t mb; } aargs = {
        (uint32_t)m->n_heads, (uint32_t)m->n_kv, (uint32_t)m->d_head, B, out_stride, attn_scale, (uint32_t)PA_MAX_LOGICAL_BLKS
    };
    if (m->kv_quant_q8) {
        [enc setComputePipelineState:g_pso_rope_kv_q8_paged];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:m->pbuf_k offset:0 atIndex:1]; [enc setBuffer:m->pbuf_v offset:0 atIndex:2];
        [enc setBuffer:ly->k_cache offset:0 atIndex:3]; [enc setBuffer:ly->v_cache offset:0 atIndex:4];
        [enc setBuffer:(m->has_rope_freqs ? m->rope_freqs.buf : m->buf_argmax) offset:(m->has_rope_freqs ? (NSUInteger)m->rope_freqs.offset : 0) atIndex:5];
        [enc setBuffer:m->buf_cb_pos offset:0 atIndex:6]; [enc setBuffer:m->buf_cb_slot offset:0 atIndex:7]; [enc setBuffer:m->buf_pa_bt offset:0 atIndex:8];
        [enc setBytes:&rargs length:sizeof(rargs) atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc setComputePipelineState:g_pso_flash_dec_q8_paged];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:ly->k_cache offset:0 atIndex:1]; [enc setBuffer:ly->v_cache offset:0 atIndex:2];
        [enc setBuffer:m->pbuf_ctx offset:0 atIndex:3]; [enc setBuffer:m->buf_cb_pos offset:0 atIndex:4]; [enc setBuffer:m->buf_cb_slot offset:0 atIndex:5];
        [enc setBuffer:m->buf_pa_bt offset:0 atIndex:6]; [enc setBytes:&aargs length:sizeof(aargs) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        [enc setComputePipelineState:g_pso_rope_kv_paged];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:m->pbuf_k offset:0 atIndex:1]; [enc setBuffer:m->pbuf_v offset:0 atIndex:2];
        [enc setBuffer:ly->k_cache offset:0 atIndex:3]; [enc setBuffer:ly->v_cache offset:0 atIndex:4];
        [enc setBuffer:(m->has_rope_freqs ? m->rope_freqs.buf : m->buf_argmax) offset:(m->has_rope_freqs ? (NSUInteger)m->rope_freqs.offset : 0) atIndex:5];
        [enc setBuffer:m->buf_cb_pos offset:0 atIndex:6]; [enc setBuffer:m->buf_cb_slot offset:0 atIndex:7]; [enc setBuffer:m->buf_pa_bt offset:0 atIndex:8];
        [enc setBytes:&rargs length:sizeof(rargs) atIndex:9];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)(m->d_head / 2), (NSUInteger)m->n_heads, B) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        [enc setComputePipelineState:g_pso_flash_dec_paged];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:ly->k_cache offset:0 atIndex:1]; [enc setBuffer:ly->v_cache offset:0 atIndex:2];
        [enc setBuffer:m->pbuf_ctx offset:0 atIndex:3]; [enc setBuffer:m->buf_cb_pos offset:0 atIndex:4]; [enc setBuffer:m->buf_cb_slot offset:0 atIndex:5];
        [enc setBuffer:m->buf_pa_bt offset:0 atIndex:6]; [enc setBytes:&aargs length:sizeof(aargs) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    if (ping_buf != nil) encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wo, ly->wo.ggml_type, m->pbuf_ctx, m->pbuf_x, dm, wo_in, B, 1);
    else encode_gemm_batch_raw(enc, ly->wo.buf, ly->wo.offset, ly->wo.ggml_type, m->pbuf_ctx, m->pbuf_x, dm, wo_in, B, 1);
    encode_rmsnorm(enc, m->pbuf_x, &ly->ffn_norm, m->pbuf_xn, dm, B, m->rms_eps, 1);
}

static void run_dense_paged_layers_batch(MoeModel* m, uint32_t B) {
    if (m->resident_mode) {
        id<MTLCommandBuffer> cmd = [g_moe_q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        for (int L = 0; L < m->n_layers; L++) {
            MoeLayer* ly = &m->layers[L];
            encode_layer_attn_paged_batch(m, enc, L, B, nil);
            encode_dense_ffn_batch(m, enc, ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                                   ly->ffn_up.buf, ly->ffn_up.offset, ly->ffn_up.ggml_type,
                                   ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
        }
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    } else {
        int first_stream = m->cache_slots_per_layer;
        int n_ring = m->dense_pin_all_attn ? 4 : 2;
        if (first_stream > 0) {
            id<MTLCommandBuffer> pcmd = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> penc = [pcmd computeCommandEncoder];
            for (int L = 0; L < first_stream && L < m->n_layers; L++) {
                MoeLayer* ly = &m->layers[L];
                encode_layer_attn_paged_batch(m, penc, L, B, nil);
                encode_dense_ffn_batch(m, penc, ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                                       ly->ffn_up.buf, ly->ffn_up.offset, ly->ffn_up.ggml_type,
                                       ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
            }
            [penc endEncoding]; [pcmd commit]; [pcmd waitUntilCompleted];
        }
        for (int L = first_stream; L < m->n_layers; L++) {
            MoeLayer* ly = &m->layers[L];
            int cur = (L - first_stream) % n_ring;
            load_dense_layer_into_ping(m, L, cur, m->enable_overlap);
            id<MTLCommandBuffer> lcmd = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> lenc = [lcmd computeCommandEncoder];
            encode_layer_attn_paged_batch(m, lenc, L, B, m->dense_pin_all_attn ? nil : m->dense_ping[cur]);
            encode_dense_ffn_batch(m, lenc, m->dense_ping[cur], m->dense_off_gate, ly->ffn_gate.ggml_type,
                                   m->dense_ping[cur], m->dense_off_up, ly->ffn_up.ggml_type,
                                   m->dense_ping[cur], m->dense_off_down, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
            [lenc endEncoding]; [lcmd commit]; [lcmd waitUntilCompleted];
        }
    }
}

// Iteration-Level Continuous Batching: evaluates `n_active` independent sequences (each with own `slot_ids[b]` & `pos[b]`)
// in ONE batched AMX/register-direct forward pass, saving all `n_active` logit rows for per-client sampling!
int64_t tn_moe_step_batch(int64_t handle, int64_t tok_ids_ptr, int64_t pos_ptr, int64_t slot_ids_ptr, int64_t n_active, int64_t out_toks_ptr) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !tok_ids_ptr || !pos_ptr || !slot_ids_ptr || !out_toks_ptr || n_active <= 0) return 0;
    MoeModel* m = &g_moe[handle];
    const int64_t* ids = (const int64_t*)(uintptr_t)tok_ids_ptr;
    const int64_t* pos = (const int64_t*)(uintptr_t)pos_ptr;
    const int64_t* slt = (const int64_t*)(uintptr_t)slot_ids_ptr;
    int64_t* out = (int64_t*)(uintptr_t)out_toks_ptr;
    if (n_active > MOE_PREFILL_MAX_B) n_active = MOE_PREFILL_MAX_B;
    uint32_t B = (uint32_t)n_active;

    if (m->batch_logits_cap < (int)B) {
        int ncap = (int)(B < 16 ? 16 : B);
        m->buf_batch_logits = [g_moe_dev newBufferWithLength:(size_t)ncap * (size_t)m->vocab_size * 4 options:MTLResourceStorageModeShared];
        m->batch_logits_cap = ncap;
    }

    if (!m->dense_model || (m->n_layers > 0 && m->layers[0].is_mla)) {
        float* bl = (float*)[m->buf_batch_logits contents];
        for (uint32_t b = 0; b < B; b++) {
            tn_moe_kv_slot_set(handle, slt[b] % (m->n_kv_slots > 0 ? m->n_kv_slots : 1));
            out[b] = tn_moe_step(handle, ids[b], pos[b]);
            memcpy(bl + (size_t)b * (size_t)m->vocab_size, [m->buf_logits contents], (size_t)m->vocab_size * 4u);
        }
        return (int64_t)B;
    }

    @autoreleasepool {
        double t0 = moe_now_ms();
        ensure_prefill_bufs(m, (int)B);
        uint32_t* p_arr = (uint32_t*)[m->buf_cb_pos contents];
        uint32_t* s_arr = (uint32_t*)[m->buf_cb_slot contents];
        for (uint32_t b = 0; b < B; b++) {
            int s = (int)slt[b]; if (s < 0 || s >= PA_MAX_SEQS) s = 0;
            int p = (int)pos[b]; if (p < 0) p = 0;
            if (!pa_ensure_slot_pos(m, s, p)) return 0;
            p_arr[b] = (uint32_t)p;
            s_arr[b] = (uint32_t)s;
        }
        pa_sync_bt(m);

        // 1. Fuse Embedding(B) + All Layers + Batched Out-Norm + Batched LM-Head into ONE CommandBuffer in Resident mode!
        float* px = (float*)[m->pbuf_x contents];
        if (m->resident_mode) {
            id<MTLCommandBuffer> cmd = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            for (uint32_t bi = 0; bi < B; bi++) {
                struct { uint32_t tok_id, d_model, ggml_type; } eargs = { (uint32_t)ids[bi], (uint32_t)m->d_model, m->tok_embd.ggml_type };
                [enc setComputePipelineState:g_pso_embed];
                [enc setBuffer:m->tok_embd.buf offset:(NSUInteger)m->tok_embd.offset atIndex:0];
                [enc setBuffer:m->pbuf_x offset:(NSUInteger)((size_t)bi * (size_t)m->d_model * 4u) atIndex:1];
                [enc setBytes:&eargs length:sizeof(eargs) atIndex:2];
                [enc dispatchThreads:MTLSizeMake((NSUInteger)m->d_model, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            for (int L = 0; L < m->n_layers; L++) {
                MoeLayer* ly = &m->layers[L];
                encode_layer_attn_paged_batch(m, enc, L, B, nil);
                encode_dense_ffn_batch(m, enc, ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                                       ly->ffn_up.buf, ly->ffn_up.offset, ly->ffn_up.ggml_type,
                                       ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
            }
            encode_rmsnorm(enc, m->pbuf_x, &m->out_norm, m->pbuf_xn, (uint32_t)m->d_model, B, m->rms_eps, 1);
            encode_gemm_batch_raw(enc, m->lm_head.buf, m->lm_head.offset, m->lm_head.ggml_type,
                                  m->pbuf_xn, m->buf_batch_logits, (uint32_t)m->vocab_size, (uint32_t)m->d_model, B, 0);
            [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        } else {
            for (uint32_t bi = 0; bi < B; bi++) {
                id<MTLCommandBuffer> ecmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> eenc = [ecmd computeCommandEncoder];
                encode_embed_step(m, eenc, (uint32_t)ids[bi]);
                [eenc endEncoding]; [ecmd commit]; [ecmd waitUntilCompleted];
                memcpy(px + (size_t)bi * (size_t)m->d_model, [m->buf_x contents], (size_t)m->d_model * 4u);
            }
            run_dense_paged_layers_batch(m, B);
            id<MTLCommandBuffer> fcmd = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> fenc = [fcmd computeCommandEncoder];
            encode_rmsnorm(fenc, m->pbuf_x, &m->out_norm, m->pbuf_xn, (uint32_t)m->d_model, B, m->rms_eps, 1);
            encode_gemm_batch_raw(fenc, m->lm_head.buf, m->lm_head.offset, m->lm_head.ggml_type,
                                  m->pbuf_xn, m->buf_batch_logits, (uint32_t)m->vocab_size, (uint32_t)m->d_model, B, 0);
            [fenc endEncoding]; [fcmd commit]; [fcmd waitUntilCompleted];
        }
        const float* bl = (const float*)[m->buf_batch_logits contents];
        for (uint32_t b = 0; b < B; b++) {
            const float* row = bl + (size_t)b * (size_t)m->vocab_size;
            int best = 0; float bv = row[0];
            for (int v = 1; v < m->vocab_size; v++) if (row[v] > bv) { bv = row[v]; best = v; }
            out[b] = (int64_t)best;
        }
        memcpy([m->buf_logits contents], bl + (size_t)(B - 1u) * (size_t)m->vocab_size, (size_t)m->vocab_size * 4u);
        if (m->buf_last_hidden != nil) memcpy([m->buf_last_hidden contents], px + (size_t)(B - 1u) * (size_t)m->d_model, (size_t)m->d_model * 4u);
        m->last_total_ms = moe_now_ms() - t0;
        return (int64_t)B;
    }
}

int64_t tn_moe_copy_logits_batch(int64_t handle, int64_t batch_idx, int64_t dst_f32_ptr) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !dst_f32_ptr) return 0;
    MoeModel* m = &g_moe[handle];
    if (m->buf_batch_logits == nil || batch_idx < 0 || batch_idx >= m->batch_logits_cap) return 0;
    const float* bl = (const float*)[m->buf_batch_logits contents];
    memcpy((void*)(uintptr_t)dst_f32_ptr, bl + (size_t)batch_idx * (size_t)m->vocab_size, (size_t)m->vocab_size * sizeof(float));
    return (int64_t)m->vocab_size;
}

// Batched Paged Prefill into PagedAttention sequence slot `slot_id` (enables instant COW Prefix Fork afterwards!)
int64_t tn_moe_paged_prefill(int64_t handle, int64_t slot_id, int64_t tok_ids_ptr, int64_t n_toks, int64_t start_pos) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !tok_ids_ptr || n_toks <= 0) return 0;
    MoeModel* m = &g_moe[handle];
    const int64_t* ids = (const int64_t*)(uintptr_t)tok_ids_ptr;
    if (!m->dense_model || (m->n_layers > 0 && m->layers[0].is_mla)) {
        tn_moe_kv_slot_set(handle, slot_id % (m->n_kv_slots > 0 ? m->n_kv_slots : 1));
        return tn_moe_prefill(handle, tok_ids_ptr, n_toks, start_pos);
    }
    @autoreleasepool {
        double t0 = moe_now_ms();
        int64_t last_tok = 0;
        int s = (int)slot_id; if (s < 0 || s >= PA_MAX_SEQS) s = 0;
        int chunk_sz = 32;
        ensure_prefill_bufs(m, chunk_sz);
        int64_t off = 0;
        while (off < n_toks) {
            uint32_t B = (uint32_t)((n_toks - off > chunk_sz) ? chunk_sz : (n_toks - off));
            uint32_t* p_arr = (uint32_t*)[m->buf_cb_pos contents];
            uint32_t* s_arr = (uint32_t*)[m->buf_cb_slot contents];
            for (uint32_t b = 0; b < B; b++) {
                int p = (int)(start_pos + off + b);
                if (!pa_ensure_slot_pos(m, s, p)) return 0;
                p_arr[b] = (uint32_t)p;
                s_arr[b] = (uint32_t)s;
            }
            pa_sync_bt(m);
            float* px = (float*)[m->pbuf_x contents];
            for (uint32_t bi = 0; bi < B; bi++) {
                id<MTLCommandBuffer> ecmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> eenc = [ecmd computeCommandEncoder];
                encode_embed_step(m, eenc, (uint32_t)ids[off + bi]);
                [eenc endEncoding]; [ecmd commit]; [ecmd waitUntilCompleted];
                memcpy(px + (size_t)bi * (size_t)m->d_model, [m->buf_x contents], (size_t)m->d_model * 4u);
            }
            run_dense_paged_layers_batch(m, B);
            if (off + B == n_toks) {
                memcpy([m->buf_x contents], px + (size_t)(B - 1u) * (size_t)m->d_model, (size_t)m->d_model * 4u);
                id<MTLCommandBuffer> fcmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> fenc = [fcmd computeCommandEncoder];
                encode_final_lm_head(m, fenc);
                [fenc endEncoding]; [fcmd commit]; [fcmd waitUntilCompleted];
                if (m->buf_last_hidden != nil) memcpy([m->buf_last_hidden contents], [m->buf_x contents], (size_t)m->d_model * 4u);
                last_tok = (int64_t)((uint32_t*)[m->buf_argmax contents])[0];
            }
            off += B;
        }
        m->last_total_ms = moe_now_ms() - t0;
        return last_tok;
    }
}

// =========================================================================
// Tenon v6: Tree-Attention Speculative Verification (`verify_tree` + `commit_tree_path`)
// =========================================================================
static void encode_layer_attn_tree(MoeModel* m, id<MTLComputeCommandEncoder> enc, int L, uint32_t start_pos, uint32_t B, id<MTLBuffer> ping_buf) {
    MoeLayer* ly = &m->layers[L];
    uint32_t dm = (uint32_t)m->d_model, qdim = (uint32_t)(m->n_heads * m->d_head), kvdim = (uint32_t)(m->n_kv * m->d_head);
    uint32_t wo_in = (ly->wo.ne0 > 0) ? (uint32_t)ly->wo.ne0 : qdim;
    uint32_t out_stride = wo_in / (uint32_t)m->n_heads;
    float attn_scale = (m->attn_mscale > 0.0f ? m->attn_mscale : 1.0f) / sqrtf((float)m->d_head);
    encode_rmsnorm(enc, m->pbuf_x, &ly->attn_norm, m->pbuf_xn, dm, B, m->rms_eps, 1);
    if (ping_buf != nil) {
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wq, ly->wq.ggml_type, m->pbuf_xn, m->pbuf_q, qdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wk, ly->wk.ggml_type, m->pbuf_xn, m->pbuf_k, kvdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wv, ly->wv.ggml_type, m->pbuf_xn, m->pbuf_v, kvdim, dm, B, 0);
    } else {
        encode_gemm_batch_raw(enc, ly->wq.buf, ly->wq.offset, ly->wq.ggml_type, m->pbuf_xn, m->pbuf_q, qdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ly->wk.buf, ly->wk.offset, ly->wk.ggml_type, m->pbuf_xn, m->pbuf_k, kvdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ly->wv.buf, ly->wv.offset, ly->wv.ggml_type, m->pbuf_xn, m->pbuf_v, kvdim, dm, B, 0);
    }
    if (ly->has_qkv_bias) {
        encode_add_bias_batch(enc, m->pbuf_q, &ly->bq, qdim, B);
        encode_add_bias_batch(enc, m->pbuf_k, &ly->bk, kvdim, B);
        encode_add_bias_batch(enc, m->pbuf_v, &ly->bv, kvdim, B);
    }
    if (ly->has_qk_norm) {
        if (ly->qk_norm_per_head) {
            encode_rmsnorm(enc, m->pbuf_q, &ly->q_norm, m->pbuf_q, (uint32_t)m->d_head, B * (uint32_t)m->n_heads, m->rms_eps, 1);
            encode_rmsnorm(enc, m->pbuf_k, &ly->k_norm, m->pbuf_k, (uint32_t)m->d_head, B * (uint32_t)m->n_kv, m->rms_eps, 1);
        } else {
            encode_rmsnorm(enc, m->pbuf_q, &ly->q_norm, m->pbuf_q, qdim, B, m->rms_eps, 1);
            encode_rmsnorm(enc, m->pbuf_k, &ly->k_norm, m->pbuf_k, kvdim, B, m->rms_eps, 1);
        }
    }
    struct { uint32_t n_heads, n_kv, d_head, rope_dim, start_pos; float theta; uint32_t rope_neox, has_freqs, batch_b; } rargs = {
        (uint32_t)m->n_heads, (uint32_t)m->n_kv, (uint32_t)m->d_head, (uint32_t)m->rope_dim,
        start_pos, m->rope_theta, (uint32_t)m->rope_neox, (uint32_t)m->has_rope_freqs, B
    };
    struct { uint32_t n_heads, n_kv, d_head, start_pos, batch_b, out_stride; float scale; } aargs = {
        (uint32_t)m->n_heads, (uint32_t)m->n_kv, (uint32_t)m->d_head, start_pos, B, out_stride, attn_scale
    };
    NSUInteger kv_slot_off = m->kv_quant_q8
        ? ((NSUInteger)m->cur_kv_slot * (NSUInteger)m->slot_ctx * (NSUInteger)m->n_kv * (NSUInteger)(m->d_head >> 5) * 34u)
        : ((NSUInteger)m->cur_kv_slot * (NSUInteger)m->slot_ctx * (NSUInteger)(m->n_kv * m->d_head) * 2u);
    if (m->kv_quant_q8) {
        [enc setComputePipelineState:g_pso_rope_kv_q8_tree];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:m->pbuf_k offset:0 atIndex:1]; [enc setBuffer:m->pbuf_v offset:0 atIndex:2];
        [enc setBuffer:ly->k_cache offset:kv_slot_off atIndex:3]; [enc setBuffer:ly->v_cache offset:kv_slot_off atIndex:4];
        [enc setBuffer:(m->has_rope_freqs ? m->rope_freqs.buf : m->buf_argmax) offset:(m->has_rope_freqs ? (NSUInteger)m->rope_freqs.offset : 0) atIndex:5];
        [enc setBuffer:m->buf_tree_depth offset:0 atIndex:6]; [enc setBytes:&rargs length:sizeof(rargs) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        [enc setComputePipelineState:g_pso_flash_tree_q8];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:ly->k_cache offset:kv_slot_off atIndex:1]; [enc setBuffer:ly->v_cache offset:kv_slot_off atIndex:2];
        [enc setBuffer:m->pbuf_ctx offset:0 atIndex:3]; [enc setBuffer:m->buf_tree_mask offset:0 atIndex:4]; [enc setBytes:&aargs length:sizeof(aargs) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        [enc setComputePipelineState:g_pso_rope_kv_tree];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:m->pbuf_k offset:0 atIndex:1]; [enc setBuffer:m->pbuf_v offset:0 atIndex:2];
        [enc setBuffer:ly->k_cache offset:kv_slot_off atIndex:3]; [enc setBuffer:ly->v_cache offset:kv_slot_off atIndex:4];
        [enc setBuffer:(m->has_rope_freqs ? m->rope_freqs.buf : m->buf_argmax) offset:(m->has_rope_freqs ? (NSUInteger)m->rope_freqs.offset : 0) atIndex:5];
        [enc setBuffer:m->buf_tree_depth offset:0 atIndex:6]; [enc setBytes:&rargs length:sizeof(rargs) atIndex:7];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)(m->d_head / 2), (NSUInteger)m->n_heads, B) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        [enc setComputePipelineState:g_pso_flash_tree];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:ly->k_cache offset:kv_slot_off atIndex:1]; [enc setBuffer:ly->v_cache offset:kv_slot_off atIndex:2];
        [enc setBuffer:m->pbuf_ctx offset:0 atIndex:3]; [enc setBuffer:m->buf_tree_mask offset:0 atIndex:4]; [enc setBytes:&aargs length:sizeof(aargs) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    if (ping_buf != nil) encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wo, ly->wo.ggml_type, m->pbuf_ctx, m->pbuf_x, dm, wo_in, B, 1);
    else encode_gemm_batch_raw(enc, ly->wo.buf, ly->wo.offset, ly->wo.ggml_type, m->pbuf_ctx, m->pbuf_x, dm, wo_in, B, 1);
    encode_rmsnorm(enc, m->pbuf_x, &ly->ffn_norm, m->pbuf_xn, dm, B, m->rms_eps, 1);
}

int64_t tn_moe_verify_tree(int64_t handle, int64_t tok_ids_ptr, int64_t parents_ptr, int64_t n_nodes, int64_t start_pos, int64_t out_toks_ptr) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !tok_ids_ptr || !parents_ptr || !out_toks_ptr || n_nodes <= 0) return 0;
    MoeModel* m = &g_moe[handle];
    if (n_nodes > TREE_MAX_NODES) n_nodes = TREE_MAX_NODES;
    const int64_t* ids = (const int64_t*)(uintptr_t)tok_ids_ptr;
    const int64_t* par = (const int64_t*)(uintptr_t)parents_ptr;
    int64_t* out = (int64_t*)(uintptr_t)out_toks_ptr;
    if (n_nodes == 1 || !m->dense_model || (m->n_layers > 0 && m->layers[0].is_mla)) {
        out[0] = tn_moe_step(handle, ids[0], start_pos);
        return 1;
    }
    @autoreleasepool {
        double t0 = moe_now_ms();
        uint32_t B = (uint32_t)n_nodes;
        ensure_prefill_bufs(m, (int)B);
        uint32_t* depths = (uint32_t*)[m->buf_tree_depth contents];
        uint64_t* masks  = (uint64_t*)[m->buf_tree_mask contents];
        for (uint32_t i = 0; i < B; i++) {
            int64_t p = par[i];
            if (p < 0 || p >= (int64_t)i) {
                depths[i] = 0u;
                masks[i] = (1ull << i);
            } else {
                depths[i] = depths[p] + 1u;
                masks[i] = masks[p] | (1ull << i);
            }
        }
        if (m->batch_logits_cap < (int)B) {
            int ncap = (int)(B < 16 ? 16 : B);
            m->buf_batch_logits = [g_moe_dev newBufferWithLength:(size_t)ncap * (size_t)m->vocab_size * 4 options:MTLResourceStorageModeShared];
            m->batch_logits_cap = ncap;
        }
        float* px = (float*)[m->pbuf_x contents];
        if (m->resident_mode) {
            id<MTLCommandBuffer> cmd = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            for (uint32_t bi = 0; bi < B; bi++) {
                struct { uint32_t tok_id, d_model, ggml_type; } eargs = { (uint32_t)ids[bi], (uint32_t)m->d_model, m->tok_embd.ggml_type };
                [enc setComputePipelineState:g_pso_embed];
                [enc setBuffer:m->tok_embd.buf offset:(NSUInteger)m->tok_embd.offset atIndex:0];
                [enc setBuffer:m->pbuf_x offset:(NSUInteger)((size_t)bi * (size_t)m->d_model * 4u) atIndex:1];
                [enc setBytes:&eargs length:sizeof(eargs) atIndex:2];
                [enc dispatchThreads:MTLSizeMake((NSUInteger)m->d_model, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            for (int L = 0; L < m->n_layers; L++) {
                MoeLayer* ly = &m->layers[L];
                encode_layer_attn_tree(m, enc, L, (uint32_t)start_pos, B, nil);
                encode_dense_ffn_batch(m, enc, ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                                       ly->ffn_up.buf, ly->ffn_up.offset, ly->ffn_up.ggml_type,
                                       ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
            }
            encode_rmsnorm(enc, m->pbuf_x, &m->out_norm, m->pbuf_xn, (uint32_t)m->d_model, B, m->rms_eps, 1);
            encode_gemm_batch_raw(enc, m->lm_head.buf, m->lm_head.offset, m->lm_head.ggml_type,
                                  m->pbuf_xn, m->buf_batch_logits, (uint32_t)m->vocab_size, (uint32_t)m->d_model, B, 0);
            [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        } else {
            for (uint32_t bi = 0; bi < B; bi++) {
                id<MTLCommandBuffer> ecmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> eenc = [ecmd computeCommandEncoder];
                encode_embed_step(m, eenc, (uint32_t)ids[bi]);
                [eenc endEncoding]; [ecmd commit]; [ecmd waitUntilCompleted];
                memcpy(px + (size_t)bi * (size_t)m->d_model, [m->buf_x contents], (size_t)m->d_model * 4u);
            }
            int first_stream = m->cache_slots_per_layer;
            int n_ring = m->dense_pin_all_attn ? 4 : 2;
            if (first_stream > 0) {
                id<MTLCommandBuffer> pcmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> penc = [pcmd computeCommandEncoder];
                for (int L = 0; L < first_stream && L < m->n_layers; L++) {
                    MoeLayer* ly = &m->layers[L];
                    encode_layer_attn_tree(m, penc, L, (uint32_t)start_pos, B, nil);
                    encode_dense_ffn_batch(m, penc, ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                                           ly->ffn_up.buf, ly->ffn_up.offset, ly->ffn_up.ggml_type,
                                           ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
                }
                [penc endEncoding]; [pcmd commit]; [pcmd waitUntilCompleted];
            }
            for (int L = first_stream; L < m->n_layers; L++) {
                MoeLayer* ly = &m->layers[L];
                int cur = (L - first_stream) % n_ring;
                load_dense_layer_into_ping(m, L, cur, m->enable_overlap);
                id<MTLCommandBuffer> lcmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> lenc = [lcmd computeCommandEncoder];
                encode_layer_attn_tree(m, lenc, L, (uint32_t)start_pos, B, m->dense_pin_all_attn ? nil : m->dense_ping[cur]);
                encode_dense_ffn_batch(m, lenc, m->dense_ping[cur], m->dense_off_gate, ly->ffn_gate.ggml_type,
                                       m->dense_ping[cur], m->dense_off_up, ly->ffn_up.ggml_type,
                                       m->dense_ping[cur], m->dense_off_down, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
                [lenc endEncoding]; [lcmd commit]; [lcmd waitUntilCompleted];
            }
            id<MTLCommandBuffer> fcmd = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> fenc = [fcmd computeCommandEncoder];
            encode_rmsnorm(fenc, m->pbuf_x, &m->out_norm, m->pbuf_xn, (uint32_t)m->d_model, B, m->rms_eps, 1);
            encode_gemm_batch_raw(fenc, m->lm_head.buf, m->lm_head.offset, m->lm_head.ggml_type,
                                  m->pbuf_xn, m->buf_batch_logits, (uint32_t)m->vocab_size, (uint32_t)m->d_model, B, 0);
            [fenc endEncoding]; [fcmd commit]; [fcmd waitUntilCompleted];
        }
        const float* bl = (const float*)[m->buf_batch_logits contents];
        for (uint32_t bi = 0; bi < B; bi++) {
            const float* row = bl + (size_t)bi * (size_t)m->vocab_size;
            int best = 0; float bv = row[0];
            for (int v = 1; v < m->vocab_size; v++) if (row[v] > bv) { bv = row[v]; best = v; }
            out[bi] = (int64_t)best;
        }
        memcpy([m->buf_logits contents], bl + (size_t)(B - 1u) * (size_t)m->vocab_size, (size_t)m->vocab_size * 4u);
        m->last_total_ms = moe_now_ms() - t0;
        return (int64_t)B;
    }
}

int64_t tn_moe_commit_tree_path(int64_t handle, int64_t path_nodes_ptr, int64_t path_len, int64_t start_pos) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !path_nodes_ptr || path_len <= 0) return 0;
    MoeModel* m = &g_moe[handle];
    const int64_t* nodes = (const int64_t*)(uintptr_t)path_nodes_ptr;
    size_t row_b = m->kv_quant_q8
        ? ((size_t)m->n_kv * (size_t)(m->d_head >> 5) * 34u)
        : ((size_t)(m->n_kv * m->d_head) * 2u);
    size_t slot_off = (size_t)m->cur_kv_slot * (size_t)m->slot_ctx * row_b;
    uint8_t* tmp = (uint8_t*)malloc((size_t)path_len * row_b);
    for (int L = 0; L < m->n_layers; L++) {
        MoeLayer* ly = &m->layers[L];
        if (ly->k_cache != nil) {
            uint8_t* base = ((uint8_t*)[ly->k_cache contents]) + slot_off;
            for (int64_t d = 0; d < path_len; d++) memcpy(tmp + (size_t)d * row_b, base + (size_t)(start_pos + nodes[d]) * row_b, row_b);
            memcpy(base + (size_t)start_pos * row_b, tmp, (size_t)path_len * row_b);
        }
        if (ly->v_cache != nil) {
            uint8_t* base = ((uint8_t*)[ly->v_cache contents]) + slot_off;
            for (int64_t d = 0; d < path_len; d++) memcpy(tmp + (size_t)d * row_b, base + (size_t)(start_pos + nodes[d]) * row_b, row_b);
            memcpy(base + (size_t)start_pos * row_b, tmp, (size_t)path_len * row_b);
        }
    }
    free(tmp);
    return path_len;
}

// =========================================================================
// Tenon v6: Native MTP (`nextn_layers`) Forward Predictor (`tn_moe_mtp_predict`)
// =========================================================================
int64_t tn_moe_mtp_predict(int64_t handle, int64_t tok_id, int64_t pos) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return -1;
    MoeModel* m = &g_moe[handle];
    if (!m->has_mtp || m->mtp_enorm.buf == nil || m->mtp_hnorm.buf == nil || m->mtp_eh_proj.buf == nil) return -1;
    @autoreleasepool {
        uint32_t dm = (uint32_t)m->d_model;
        id<MTLCommandBuffer> cmd = [g_moe_q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        encode_embed_step(m, enc, (uint32_t)tok_id);
        encode_rmsnorm(enc, m->buf_x, &m->mtp_enorm, m->buf_xn, dm, 1, m->rms_eps, 0);
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];

        // Normalize saved `buf_last_hidden` and concatenate `[e_norm || h_norm]` -> `buf_mtp_cat`
        float* cat = (float*)[m->buf_mtp_cat contents];
        memcpy(cat, [m->buf_xn contents], (size_t)dm * 4u);
        {
            id<MTLCommandBuffer> c2 = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> e2 = [c2 computeCommandEncoder];
            encode_rmsnorm(e2, m->buf_last_hidden, &m->mtp_hnorm, m->buf_xn, dm, 1, m->rms_eps, 0);
            [e2 endEncoding]; [c2 commit]; [c2 waitUntilCompleted];
        }
        memcpy(cat + dm, [m->buf_xn contents], (size_t)dm * 4u);

        id<MTLCommandBuffer> c3 = [g_moe_q commandBuffer];
        id<MTLComputeCommandEncoder> e3 = [c3 computeCommandEncoder];
        encode_gemv(e3, &m->mtp_eh_proj, m->buf_mtp_cat, m->buf_x, dm, dm * 2u, 0);
        if (m->has_mtp_layer && m->mtp_layer && m->mtp_layer->attn_norm.buf != nil) {
            MoeLayer* ml = m->mtp_layer;
            encode_layer_attn_body_ptr(m, e3, ml, (uint32_t)pos, nil);
            if (ml->ffn_gate.buf != nil && ml->ffn_up.buf != nil && ml->ffn_down.buf != nil) {
                uint32_t df = (ml->ffn_gate.ne1 > 0) ? (uint32_t)ml->ffn_gate.ne1 : (uint32_t)m->d_ff_dense;
                encode_dense_ffn_raw(m, e3, ml->ffn_gate.buf, ml->ffn_gate.offset, ml->ffn_gate.ggml_type,
                                     ml->ffn_up.buf, ml->ffn_up.offset, ml->ffn_up.ggml_type,
                                     ml->ffn_down.buf, ml->ffn_down.offset, ml->ffn_down.ggml_type, df);
            } else if (ml->has_shared_exp && ml->sh_gate.buf != nil) {
                uint32_t sf = (ml->sh_gate.ne1 > 0) ? (uint32_t)ml->sh_gate.ne1 : (uint32_t)m->d_ff_exp;
                encode_dense_ffn_raw(m, e3, ml->sh_gate.buf, ml->sh_gate.offset, ml->sh_gate.ggml_type,
                                     ml->sh_up.buf, ml->sh_up.offset, ml->sh_up.ggml_type,
                                     ml->sh_down.buf, ml->sh_down.offset, ml->sh_down.ggml_type, sf);
            }
        }
        encode_final_lm_head(m, e3);
        [e3 endEncoding]; [c3 commit]; [c3 waitUntilCompleted];
        return (int64_t)((uint32_t*)[m->buf_argmax contents])[0];
    }
}

// =========================================================================
// Tenon v6: Text Embeddings (`tn_moe_embed_text`) & Soft-Prompt / Vision Embedding Injection (`tn_moe_step_embd`)
// =========================================================================
int64_t tn_moe_embed_text(int64_t handle, int64_t tok_ids_ptr, int64_t n_toks, int64_t out_f32_ptr, int64_t max_dim) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !tok_ids_ptr || !out_f32_ptr || n_toks <= 0) return 0;
    MoeModel* m = &g_moe[handle];
    int dm = m->d_model;
    if (max_dim < dm) return 0;
    const int64_t* ids = (const int64_t*)(uintptr_t)tok_ids_ptr;
    float* out = (float*)(uintptr_t)out_f32_ptr;
    memset(out, 0, (size_t)dm * sizeof(float));
    @autoreleasepool {
        // Mean-pool final post-layer hidden states across all tokens, apply `out_norm`, and L2-normalize!
        for (int64_t i = 0; i < n_toks; i++) {
            tn_moe_step(handle, ids[i], i);
            const float* hx = (const float*)[m->buf_x contents];
            for (int d = 0; d < dm; d++) out[d] += hx[d];
        }
        float inv_n = 1.0f / (float)n_toks;
        float* bx = (float*)[m->buf_x contents];
        for (int d = 0; d < dm; d++) bx[d] = out[d] * inv_n;
        id<MTLCommandBuffer> cmd = [g_moe_q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        encode_rmsnorm(enc, m->buf_x, &m->out_norm, m->buf_xn, (uint32_t)dm, 1, m->rms_eps, 0);
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
        const float* xn = (const float*)[m->buf_xn contents];
        double sq = 0.0;
        for (int d = 0; d < dm; d++) sq += (double)xn[d] * (double)xn[d];
        float inv_l2 = (sq > 1e-12) ? (float)(1.0 / sqrt(sq)) : 1.0f;
        for (int d = 0; d < dm; d++) out[d] = xn[d] * inv_l2;
        return (int64_t)dm;
    }
}

int64_t tn_moe_step_embd(int64_t handle, int64_t embd_f32_ptr, int64_t n_vecs, int64_t start_pos) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !embd_f32_ptr || n_vecs <= 0) return 0;
    MoeModel* m = &g_moe[handle];
    const float* vecs = (const float*)(uintptr_t)embd_f32_ptr;
    @autoreleasepool {
        double t0 = moe_now_ms();
        int64_t last_tok = 0;
        if (m->dense_model && m->resident_mode && !(m->n_layers > 0 && m->layers[0].is_mla)) {
            int chunk_sz = 32;
            ensure_prefill_bufs(m, chunk_sz);
            int64_t off = 0;
            while (off < n_vecs) {
                uint32_t B = (uint32_t)((n_vecs - off > chunk_sz) ? chunk_sz : (n_vecs - off));
                memcpy([m->pbuf_x contents], vecs + (size_t)off * (size_t)m->d_model, (size_t)B * (size_t)m->d_model * 4u);
                id<MTLCommandBuffer> cmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                for (int L = 0; L < m->n_layers; L++) {
                    MoeLayer* ly = &m->layers[L];
                    encode_layer_attn_batch(m, enc, L, (uint32_t)(start_pos + off), B, nil);
                    encode_dense_ffn_batch(m, enc, ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                                           ly->ffn_up.buf, ly->ffn_up.offset, ly->ffn_up.ggml_type,
                                           ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
                }
                [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
                if (off + B == n_vecs) {
                    const float* px = (const float*)[m->pbuf_x contents];
                    memcpy([m->buf_x contents], px + (size_t)(B - 1u) * (size_t)m->d_model, (size_t)m->d_model * 4u);
                    id<MTLCommandBuffer> fcmd = [g_moe_q commandBuffer];
                    id<MTLComputeCommandEncoder> fenc = [fcmd computeCommandEncoder];
                    encode_final_lm_head(m, fenc);
                    [fenc endEncoding]; [fcmd commit]; [fcmd waitUntilCompleted];
                    if (m->buf_last_hidden != nil) memcpy([m->buf_last_hidden contents], [m->buf_x contents], (size_t)m->d_model * 4u);
                    last_tok = (int64_t)((uint32_t*)[m->buf_argmax contents])[0];
                }
                off += B;
            }
        } else {
            for (int64_t i = 0; i < n_vecs; i++) {
                memcpy([m->buf_x contents], vecs + (size_t)i * (size_t)m->d_model, (size_t)m->d_model * 4u);
                id<MTLCommandBuffer> cmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                for (int L = 0; L < m->n_layers; L++) {
                    MoeLayer* ly = &m->layers[L];
                    encode_layer_attn_body(m, enc, L, (uint32_t)(start_pos + i), nil);
                    if (ly->is_dense) {
                        encode_dense_ffn_raw(m, enc, ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                                             ly->ffn_up.buf, ly->ffn_up.offset, ly->ffn_up.ggml_type,
                                             ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense);
                    }
                }
                encode_final_lm_head(m, enc);
                [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
                last_tok = (int64_t)((uint32_t*)[m->buf_argmax contents])[0];
            }
        }
        m->last_total_ms = moe_now_ms() - t0;
        return last_tok;
    }
}

// =========================================================================
// Tenon v6: Native Multimodal Vision Encoder (`mmproj` CLIP/SigLIP/SmolVLM GGUF on Metal GPU!)
// =========================================================================
#define MMPROJ_MAX_MODELS 4
#define MMPROJ_MAX_LAYERS 48

typedef struct {
    MoeWeightRef ln1_w, ln1_b, ln2_w, ln2_b;
    MoeWeightRef q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b;
    MoeWeightRef fc1_w, fc1_b, fc2_w, fc2_b;
} MmprojLayer;

typedef struct {
    int active;
    int fd;
    void* mmap_ptr;
    size_t mmap_len;
    id<MTLBuffer> mmap_buf;

    int image_size;
    int patch_size;
    int d_vis;
    int d_ff;
    int n_layers;
    int n_heads;
    int proj_dim;
    int scale_factor;
    int use_gelu;
    int has_cls;
    int is_idefics3;
    float eps;
    float mean[3];
    float std[3];

    MoeWeightRef patch_w, patch_b;
    MoeWeightRef class_embd;
    MoeWeightRef pos_embd;
    MoeWeightRef pre_ln_w, pre_ln_b;
    MoeWeightRef post_ln_w, post_ln_b;
    MoeWeightRef mm_fc_w, mm_fc_b;
    MoeWeightRef mm_0_w, mm_0_b, mm_2_w, mm_2_b;
    MmprojLayer layers[MMPROJ_MAX_LAYERS];

    id<MTLBuffer> buf_in, buf_x, buf_xn, buf_q, buf_k, buf_v, buf_ctx, buf_ff, buf_ps, buf_out, buf_zero;
} MmprojModel;

static MmprojModel g_mmproj[MMPROJ_MAX_MODELS];

static void mm_bind(MmprojModel* mm, MoeWeightRef* r, const MoeTensorInfo* ti) {
    record_weight_meta(r, ti);
    r->buf = mm->mmap_buf;
    r->offset = ti->file_off;
}

int64_t tn_moe_mmproj_load(int64_t path_cstr) {
    if (!moe_init_metal() || !path_cstr) return 0;
    @autoreleasepool {
        int slot = -1;
        for (int i = 1; i < MMPROJ_MAX_MODELS; i++) { if (!g_mmproj[i].active) { slot = i; break; } }
        if (slot < 0) return 0;
        MmprojModel* mm = &g_mmproj[slot];
        memset((void*)mm, 0, sizeof(MmprojModel));
        mm->fd = -1;
        mm->image_size = 512; mm->patch_size = 16; mm->scale_factor = 1;
        mm->eps = 1e-6f;
        mm->mean[0] = mm->mean[1] = mm->mean[2] = 0.5f;
        mm->std[0]  = mm->std[1]  = mm->std[2]  = 0.5f;

        const char* path = (const char*)(uintptr_t)path_cstr;
        FILE* fp = fopen(path, "rb");
        if (!fp) return 0;
        GgufReader gr = { fp };
        uint32_t magic = gr_u32(&gr), ver = gr_u32(&gr);
        if (magic != 0x46554747u || ver < 2 || ver > 3) { fclose(fp); return 0; }
        uint64_t tcnt = gr_u64(&gr), kvcnt = gr_u64(&gr), align = 32;
        for (uint64_t i = 0; i < kvcnt; i++) {
            char* key = gr_str(&gr); uint32_t vt = gr_u32(&gr);
            if (strcmp(key, "general.alignment") == 0) align = gr_read_int(&gr, vt);
            else if (strcmp(key, "clip.vision.image_size") == 0) mm->image_size = (int)gr_read_int(&gr, vt);
            else if (strcmp(key, "clip.vision.patch_size") == 0) mm->patch_size = (int)gr_read_int(&gr, vt);
            else if (strcmp(key, "clip.vision.embedding_length") == 0) mm->d_vis = (int)gr_read_int(&gr, vt);
            else if (strcmp(key, "clip.vision.feed_forward_length") == 0) mm->d_ff = (int)gr_read_int(&gr, vt);
            else if (strcmp(key, "clip.vision.block_count") == 0) mm->n_layers = (int)gr_read_int(&gr, vt);
            else if (strcmp(key, "clip.vision.attention.head_count") == 0) mm->n_heads = (int)gr_read_int(&gr, vt);
            else if (strcmp(key, "clip.vision.projection_dim") == 0) mm->proj_dim = (int)gr_read_int(&gr, vt);
            else if (strcmp(key, "clip.vision.projector.scale_factor") == 0) mm->scale_factor = (int)gr_read_int(&gr, vt);
            else if (strcmp(key, "clip.use_gelu") == 0) mm->use_gelu = (int)gr_read_int(&gr, vt);
            else if (strcmp(key, "clip.vision.attention.layer_norm_epsilon") == 0 && vt == GGUF_T_F32) mm->eps = gr_f32(&gr);
            else if (strcmp(key, "clip.projector_type") == 0 && vt == GGUF_T_STR) {
                char* pt = gr_str(&gr);
                if (strstr(pt, "idefics3") != NULL) mm->is_idefics3 = 1;
                free(pt);
            } else if ((strcmp(key, "clip.vision.image_mean") == 0 || strcmp(key, "clip.vision.image_std") == 0) && vt == GGUF_T_ARR) {
                uint32_t et = gr_u32(&gr); uint64_t n = gr_u64(&gr);
                float* dst = (key[18] == 'm') ? mm->mean : mm->std;
                for (uint64_t j = 0; j < n; j++) {
                    if (et == GGUF_T_F32) { float v = gr_f32(&gr); if (j < 3) dst[j] = v; }
                    else gr_skip_val(&gr, et);
                }
            } else gr_skip_val(&gr, vt);
            free(key);
        }
        MoeTensorInfo* tlist = (MoeTensorInfo*)calloc((size_t)tcnt, sizeof(MoeTensorInfo));
        for (uint64_t i = 0; i < tcnt; i++) {
            char* tn = gr_str(&gr); strncpy(tlist[i].name, tn, sizeof(tlist[i].name) - 1); free(tn);
            tlist[i].n_dims = gr_u32(&gr);
            uint64_t numel = 1;
            for (uint32_t d = 0; d < tlist[i].n_dims; d++) { uint64_t dm = gr_u64(&gr); if (d < 4) tlist[i].ne[d] = dm; numel *= dm; }
            tlist[i].ggml_type = gr_u32(&gr); tlist[i].rel_off = gr_u64(&gr);
            tlist[i].byte_size = ggml_type_bytes(tlist[i].ggml_type, numel);
        }
        uint64_t hdr_end = (uint64_t)ftello(fp);
        uint64_t data_off = (hdr_end + align - 1) & ~(align - 1);
        fclose(fp);
        for (uint64_t i = 0; i < tcnt; i++) tlist[i].file_off = data_off + tlist[i].rel_off;

        struct stat st;
        if (stat(path, &st) != 0) { free(tlist); return 0; }
        mm->fd = open(path, O_RDONLY);
        if (mm->fd < 0) { free(tlist); return 0; }
        size_t psz = (size_t)getpagesize();
        mm->mmap_len = ((size_t)st.st_size + psz - 1) & ~(psz - 1);
        mm->mmap_ptr = mmap(NULL, mm->mmap_len, PROT_READ, MAP_PRIVATE, mm->fd, 0);
        if (mm->mmap_ptr == MAP_FAILED) { close(mm->fd); free(tlist); return 0; }
        mm->mmap_buf = [g_moe_dev newBufferWithBytesNoCopy:mm->mmap_ptr length:mm->mmap_len options:MTLResourceStorageModeShared deallocator:nil];
        if (mm->n_layers > MMPROJ_MAX_LAYERS) mm->n_layers = MMPROJ_MAX_LAYERS;

        for (uint64_t i = 0; i < tcnt; i++) {
            const MoeTensorInfo* ti = &tlist[i];
            if (strcmp(ti->name, "v.patch_embd.weight") == 0) mm_bind(mm, &mm->patch_w, ti);
            else if (strcmp(ti->name, "v.patch_embd.bias") == 0) mm_bind(mm, &mm->patch_b, ti);
            else if (strcmp(ti->name, "v.class_embd") == 0) { mm_bind(mm, &mm->class_embd, ti); mm->has_cls = 1; }
            else if (strcmp(ti->name, "v.position_embd.weight") == 0) mm_bind(mm, &mm->pos_embd, ti);
            else if (strcmp(ti->name, "v.pre_ln.weight") == 0) mm_bind(mm, &mm->pre_ln_w, ti);
            else if (strcmp(ti->name, "v.pre_ln.bias") == 0) mm_bind(mm, &mm->pre_ln_b, ti);
            else if (strcmp(ti->name, "v.post_ln.weight") == 0) mm_bind(mm, &mm->post_ln_w, ti);
            else if (strcmp(ti->name, "v.post_ln.bias") == 0) mm_bind(mm, &mm->post_ln_b, ti);
            else if (strcmp(ti->name, "mm.model.fc.weight") == 0) { mm_bind(mm, &mm->mm_fc_w, ti); mm->is_idefics3 = 1; }
            else if (strcmp(ti->name, "mm.model.fc.bias") == 0) mm_bind(mm, &mm->mm_fc_b, ti);
            else if (strcmp(ti->name, "mm.0.weight") == 0) mm_bind(mm, &mm->mm_0_w, ti);
            else if (strcmp(ti->name, "mm.0.bias") == 0) mm_bind(mm, &mm->mm_0_b, ti);
            else if (strcmp(ti->name, "mm.2.weight") == 0) mm_bind(mm, &mm->mm_2_w, ti);
            else if (strcmp(ti->name, "mm.2.bias") == 0) mm_bind(mm, &mm->mm_2_b, ti);
            else if (strncmp(ti->name, "v.blk.", 6) == 0) {
                int L = atoi(ti->name + 6);
                const char* dot = strchr(ti->name + 6, '.');
                if (!dot || L < 0 || L >= mm->n_layers) continue;
                const char* sub = dot + 1;
                MmprojLayer* ly = &mm->layers[L];
                if (strcmp(sub, "ln1.weight") == 0) mm_bind(mm, &ly->ln1_w, ti);
                else if (strcmp(sub, "ln1.bias") == 0) mm_bind(mm, &ly->ln1_b, ti);
                else if (strcmp(sub, "ln2.weight") == 0) mm_bind(mm, &ly->ln2_w, ti);
                else if (strcmp(sub, "ln2.bias") == 0) mm_bind(mm, &ly->ln2_b, ti);
                else if (strcmp(sub, "attn_q.weight") == 0) mm_bind(mm, &ly->q_w, ti);
                else if (strcmp(sub, "attn_q.bias") == 0) mm_bind(mm, &ly->q_b, ti);
                else if (strcmp(sub, "attn_k.weight") == 0) mm_bind(mm, &ly->k_w, ti);
                else if (strcmp(sub, "attn_k.bias") == 0) mm_bind(mm, &ly->k_b, ti);
                else if (strcmp(sub, "attn_v.weight") == 0) mm_bind(mm, &ly->v_w, ti);
                else if (strcmp(sub, "attn_v.bias") == 0) mm_bind(mm, &ly->v_b, ti);
                else if (strcmp(sub, "attn_out.weight") == 0) mm_bind(mm, &ly->o_w, ti);
                else if (strcmp(sub, "attn_out.bias") == 0) mm_bind(mm, &ly->o_b, ti);
                else if (strcmp(sub, "ffn_down.weight") == 0) {
                    if ((int)ti->ne[1] == mm->d_ff) mm_bind(mm, &ly->fc1_w, ti); else mm_bind(mm, &ly->fc2_w, ti);
                } else if (strcmp(sub, "ffn_down.bias") == 0) {
                    if ((int)ti->ne[0] == mm->d_ff) mm_bind(mm, &ly->fc1_b, ti); else mm_bind(mm, &ly->fc2_b, ti);
                } else if (strcmp(sub, "ffn_up.weight") == 0) {
                    if ((int)ti->ne[1] == mm->d_ff) mm_bind(mm, &ly->fc1_w, ti); else mm_bind(mm, &ly->fc2_w, ti);
                } else if (strcmp(sub, "ffn_up.bias") == 0) {
                    if ((int)ti->ne[0] == mm->d_ff) mm_bind(mm, &ly->fc1_b, ti); else mm_bind(mm, &ly->fc2_b, ti);
                }
            }
        }
        free(tlist);
        if (mm->d_vis <= 0 || mm->n_layers <= 0 || mm->patch_size <= 0) { close(mm->fd); munmap(mm->mmap_ptr, mm->mmap_len); return 0; }
        int grid = mm->image_size / mm->patch_size;
        int n_tok = grid * grid + (mm->has_cls ? 1 : 0);
        int patch_dim = 3 * mm->patch_size * mm->patch_size;
        int n_out = mm->is_idefics3 ? ((grid / mm->scale_factor) * (grid / mm->scale_factor)) : (grid * grid);
        int ps_dim = mm->is_idefics3 ? (mm->scale_factor * mm->scale_factor * mm->d_vis) : mm->d_vis;
        int pdim = (mm->proj_dim > 0) ? mm->proj_dim : mm->d_vis;

        mm->buf_in   = [g_moe_dev newBufferWithLength:(size_t)n_tok * (size_t)patch_dim * 4 options:MTLResourceStorageModeShared];
        mm->buf_x    = [g_moe_dev newBufferWithLength:(size_t)n_tok * (size_t)mm->d_vis * 4 options:MTLResourceStorageModeShared];
        mm->buf_xn   = [g_moe_dev newBufferWithLength:(size_t)n_tok * (size_t)mm->d_vis * 4 options:MTLResourceStorageModeShared];
        mm->buf_q    = [g_moe_dev newBufferWithLength:(size_t)n_tok * (size_t)mm->d_vis * 4 options:MTLResourceStorageModeShared];
        mm->buf_k    = [g_moe_dev newBufferWithLength:(size_t)n_tok * (size_t)mm->d_vis * 4 options:MTLResourceStorageModeShared];
        mm->buf_v    = [g_moe_dev newBufferWithLength:(size_t)n_tok * (size_t)mm->d_vis * 4 options:MTLResourceStorageModeShared];
        mm->buf_ctx  = [g_moe_dev newBufferWithLength:(size_t)n_tok * (size_t)mm->d_vis * 4 options:MTLResourceStorageModeShared];
        mm->buf_ff   = [g_moe_dev newBufferWithLength:(size_t)n_tok * (size_t)mm->d_ff * 4 options:MTLResourceStorageModeShared];
        mm->buf_ps   = [g_moe_dev newBufferWithLength:(size_t)n_out * (size_t)ps_dim * 4 options:MTLResourceStorageModeShared];
        mm->buf_out  = [g_moe_dev newBufferWithLength:(size_t)n_out * (size_t)pdim * 4 options:MTLResourceStorageModeShared];
        mm->buf_zero = [g_moe_dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        mm->active = 1;
        return (int64_t)slot;
    }
}

int64_t tn_moe_mmproj_info(int64_t mm_h, int64_t field_id) {
    if (mm_h <= 0 || mm_h >= MMPROJ_MAX_MODELS || !g_mmproj[mm_h].active) return 0;
    MmprojModel* mm = &g_mmproj[mm_h];
    int grid = mm->image_size / (mm->patch_size > 0 ? mm->patch_size : 16);
    int n_out = mm->is_idefics3 ? ((grid / mm->scale_factor) * (grid / mm->scale_factor)) : (grid * grid);
    switch (field_id) {
        case 0: return mm->image_size;
        case 1: return mm->patch_size;
        case 2: return mm->d_vis;
        case 3: return mm->n_layers;
        case 4: return mm->n_heads;
        case 5: return mm->proj_dim > 0 ? mm->proj_dim : mm->d_vis;
        case 6: return n_out;
        case 7: return mm->scale_factor;
        case 8: return mm->has_cls;
        default: return 0;
    }
}

static void encode_vit_ln(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, const MoeWeightRef* w, const MoeWeightRef* b,
                          id<MTLBuffer> out, uint32_t dim, uint32_t n_tok, float eps, id<MTLBuffer> zero) {
    struct { uint32_t dim, n_tok; float eps; uint32_t hb; } a = { dim, n_tok, eps, (b->buf != nil ? 1u : 0u) };
    [enc setComputePipelineState:g_pso_vit_ln];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:w->buf offset:(NSUInteger)w->offset atIndex:1];
    [enc setBuffer:(b->buf != nil ? b->buf : zero) offset:(b->buf != nil ? (NSUInteger)b->offset : 0) atIndex:2];
    [enc setBuffer:out offset:0 atIndex:3];
    [enc setBytes:&a length:sizeof(a) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(n_tok, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

int64_t tn_moe_mmproj_encode_rgb(int64_t mm_h, int64_t rgb_u8_ptr, int64_t width, int64_t height, int64_t out_embd_ptr, int64_t max_vecs) {
    if (mm_h <= 0 || mm_h >= MMPROJ_MAX_MODELS || !g_mmproj[mm_h].active || !rgb_u8_ptr || !out_embd_ptr || width <= 0 || height <= 0) return 0;
    MmprojModel* mm = &g_mmproj[mm_h];
    int sz = mm->image_size, ps = mm->patch_size, grid = sz / ps;
    int n_patches = grid * grid, n_tok = n_patches + (mm->has_cls ? 1 : 0);
    int pdim = 3 * ps * ps;
    int n_out = mm->is_idefics3 ? ((grid / mm->scale_factor) * (grid / mm->scale_factor)) : n_patches;
    if (max_vecs < n_out) return 0;

    @autoreleasepool {
        // 1. Bilinear resize + ImageNet/SigLIP normalize + pack `[n_patches, 3 * ps * ps]` (`[c][ky][kx]`)!
        const uint8_t* src = (const uint8_t*)(uintptr_t)rgb_u8_ptr;
        float* img = (float*)malloc((size_t)3 * (size_t)sz * (size_t)sz * sizeof(float));
        for (int y = 0; y < sz; y++) {
            float fy = ((float)y + 0.5f) * (float)height / (float)sz - 0.5f;
            int y0 = (int)floorf(fy); float dy = fy - (float)y0;
            int y1 = y0 + 1;
            if (y0 < 0) y0 = 0; if (y0 >= height) y0 = (int)height - 1;
            if (y1 < 0) y1 = 0; if (y1 >= height) y1 = (int)height - 1;
            for (int x = 0; x < sz; x++) {
                float fx = ((float)x + 0.5f) * (float)width / (float)sz - 0.5f;
                int x0 = (int)floorf(fx); float dx = fx - (float)x0;
                int x1 = x0 + 1;
                if (x0 < 0) x0 = 0; if (x0 >= width) x0 = (int)width - 1;
                if (x1 < 0) x1 = 0; if (x1 >= width) x1 = (int)width - 1;
                for (int c = 0; c < 3; c++) {
                    float p00 = (float)src[((size_t)y0 * (size_t)width + (size_t)x0) * 3u + c];
                    float p01 = (float)src[((size_t)y0 * (size_t)width + (size_t)x1) * 3u + c];
                    float p10 = (float)src[((size_t)y1 * (size_t)width + (size_t)x0) * 3u + c];
                    float p11 = (float)src[((size_t)y1 * (size_t)width + (size_t)x1) * 3u + c];
                    float v = ((p00 * (1.0f - dx) + p01 * dx) * (1.0f - dy) + (p10 * (1.0f - dx) + p11 * dx) * dy) / 255.0f;
                    img[((size_t)c * (size_t)sz + (size_t)y) * (size_t)sz + (size_t)x] = (v - mm->mean[c]) / mm->std[c];
                }
            }
        }
        float* pin = (float*)[mm->buf_in contents];
        for (int py = 0; py < grid; py++) {
            for (int px = 0; px < grid; px++) {
                float* dst = pin + (size_t)(py * grid + px) * (size_t)pdim;
                for (int c = 0; c < 3; c++) {
                    for (int ky = 0; ky < ps; ky++) {
                        const float* row = img + ((size_t)c * (size_t)sz + (size_t)(py * ps + ky)) * (size_t)sz + (size_t)(px * ps);
                        memcpy(dst + ((size_t)c * (size_t)ps + (size_t)ky) * (size_t)ps, row, (size_t)ps * sizeof(float));
                    }
                }
            }
        }
        free(img);

        // 2. Metal GPU Patch Conv2D GEMM + Bias + Optional [CLS] + Positional Embedding + ViT Layers!
        id<MTLCommandBuffer> cmd = [g_moe_q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        id<MTLBuffer> patch_out = mm->has_cls ? mm->buf_xn : mm->buf_x;
        encode_gemm_batch_raw(enc, mm->patch_w.buf, mm->patch_w.offset, mm->patch_w.ggml_type,
                              mm->buf_in, patch_out, (uint32_t)mm->d_vis, (uint32_t)pdim, (uint32_t)n_patches, 0);
        if (mm->patch_b.buf != nil) encode_add_bias_batch(enc, patch_out, &mm->patch_b, (uint32_t)mm->d_vis, (uint32_t)n_patches);
        [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];

        if (mm->has_cls && mm->class_embd.buf != nil) {
            float* bx = (float*)[mm->buf_x contents];
            const float* ce = (const float*)(((const uint8_t*)[mm->class_embd.buf contents]) + mm->class_embd.offset);
            memcpy(bx, ce, (size_t)mm->d_vis * sizeof(float));
            memcpy(bx + mm->d_vis, [mm->buf_xn contents], (size_t)n_patches * (size_t)mm->d_vis * sizeof(float));
        }

        id<MTLCommandBuffer> vcmd = [g_moe_q commandBuffer];
        id<MTLComputeCommandEncoder> venc = [vcmd computeCommandEncoder];
        if (mm->pos_embd.buf != nil) {
            uint32_t tot = (uint32_t)(n_tok * mm->d_vis);
            [venc setComputePipelineState:g_pso_vit_add];
            [venc setBuffer:mm->buf_x offset:0 atIndex:0];
            [venc setBuffer:mm->pos_embd.buf offset:(NSUInteger)mm->pos_embd.offset atIndex:1];
            [venc setBytes:&tot length:4 atIndex:2];
            [venc dispatchThreads:MTLSizeMake(tot, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        if (mm->pre_ln_w.buf != nil) encode_vit_ln(venc, mm->buf_x, &mm->pre_ln_w, &mm->pre_ln_b, mm->buf_x, (uint32_t)mm->d_vis, (uint32_t)n_tok, mm->eps, mm->buf_zero);

        uint32_t d_head = (uint32_t)(mm->d_vis / (mm->n_heads > 0 ? mm->n_heads : 1));
        float attn_sc = 1.0f / sqrtf((float)d_head);
        for (int L = 0; L < mm->n_layers; L++) {
            MmprojLayer* ly = &mm->layers[L];
            encode_vit_ln(venc, mm->buf_x, &ly->ln1_w, &ly->ln1_b, mm->buf_xn, (uint32_t)mm->d_vis, (uint32_t)n_tok, mm->eps, mm->buf_zero);
            encode_gemm_batch_raw(venc, ly->q_w.buf, ly->q_w.offset, ly->q_w.ggml_type, mm->buf_xn, mm->buf_q, (uint32_t)mm->d_vis, (uint32_t)mm->d_vis, (uint32_t)n_tok, 0);
            encode_gemm_batch_raw(venc, ly->k_w.buf, ly->k_w.offset, ly->k_w.ggml_type, mm->buf_xn, mm->buf_k, (uint32_t)mm->d_vis, (uint32_t)mm->d_vis, (uint32_t)n_tok, 0);
            encode_gemm_batch_raw(venc, ly->v_w.buf, ly->v_w.offset, ly->v_w.ggml_type, mm->buf_xn, mm->buf_v, (uint32_t)mm->d_vis, (uint32_t)mm->d_vis, (uint32_t)n_tok, 0);
            encode_add_bias_batch(venc, mm->buf_q, &ly->q_b, (uint32_t)mm->d_vis, (uint32_t)n_tok);
            encode_add_bias_batch(venc, mm->buf_k, &ly->k_b, (uint32_t)mm->d_vis, (uint32_t)n_tok);
            encode_add_bias_batch(venc, mm->buf_v, &ly->v_b, (uint32_t)mm->d_vis, (uint32_t)n_tok);

            struct { uint32_t nt, nh, dh; float sc; } va = { (uint32_t)n_tok, (uint32_t)mm->n_heads, d_head, attn_sc };
            [venc setComputePipelineState:g_pso_vit_attn];
            [venc setBuffer:mm->buf_q offset:0 atIndex:0]; [venc setBuffer:mm->buf_k offset:0 atIndex:1];
            [venc setBuffer:mm->buf_v offset:0 atIndex:2]; [venc setBuffer:mm->buf_ctx offset:0 atIndex:3];
            [venc setBytes:&va length:sizeof(va) atIndex:4];
            [venc dispatchThreadgroups:MTLSizeMake((NSUInteger)mm->n_heads, (NSUInteger)n_tok, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            encode_gemm_batch_raw(venc, ly->o_w.buf, ly->o_w.offset, ly->o_w.ggml_type, mm->buf_ctx, mm->buf_x, (uint32_t)mm->d_vis, (uint32_t)mm->d_vis, (uint32_t)n_tok, 1);
            encode_add_bias_batch(venc, mm->buf_x, &ly->o_b, (uint32_t)mm->d_vis, (uint32_t)n_tok);

            encode_vit_ln(venc, mm->buf_x, &ly->ln2_w, &ly->ln2_b, mm->buf_xn, (uint32_t)mm->d_vis, (uint32_t)n_tok, mm->eps, mm->buf_zero);
            encode_gemm_batch_raw(venc, ly->fc1_w.buf, ly->fc1_w.offset, ly->fc1_w.ggml_type, mm->buf_xn, mm->buf_ff, (uint32_t)mm->d_ff, (uint32_t)mm->d_vis, (uint32_t)n_tok, 0);
            encode_add_bias_batch(venc, mm->buf_ff, &ly->fc1_b, (uint32_t)mm->d_ff, (uint32_t)n_tok);

            struct { uint32_t n, ut; } ga = { (uint32_t)(n_tok * mm->d_ff), (uint32_t)mm->use_gelu };
            [venc setComputePipelineState:g_pso_vit_gelu];
            [venc setBuffer:mm->buf_ff offset:0 atIndex:0]; [venc setBytes:&ga length:sizeof(ga) atIndex:1];
            [venc dispatchThreads:MTLSizeMake((NSUInteger)ga.n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            encode_gemm_batch_raw(venc, ly->fc2_w.buf, ly->fc2_w.offset, ly->fc2_w.ggml_type, mm->buf_ff, mm->buf_x, (uint32_t)mm->d_vis, (uint32_t)mm->d_ff, (uint32_t)n_tok, 1);
            encode_add_bias_batch(venc, mm->buf_x, &ly->fc2_b, (uint32_t)mm->d_vis, (uint32_t)n_tok);
        }
        if (mm->post_ln_w.buf != nil) encode_vit_ln(venc, mm->buf_x, &mm->post_ln_w, &mm->post_ln_b, mm->buf_x, (uint32_t)mm->d_vis, (uint32_t)n_tok, mm->eps, mm->buf_zero);
        [venc endEncoding]; [vcmd commit]; [vcmd waitUntilCompleted];

        // 3. Multimodal Projector: Idefics3/SmolVLM Pixel-Shuffle + FC OR LLaVA 2-Layer MLP!
        int out_dim = (mm->proj_dim > 0) ? mm->proj_dim : mm->d_vis;
        if (mm->is_idefics3 && mm->mm_fc_w.buf != nil) {
            int sf = mm->scale_factor, og = grid / sf, in_fc = sf * sf * mm->d_vis;
            const float* vx = ((const float*)[mm->buf_x contents]) + (mm->has_cls ? mm->d_vis : 0);
            float* psb = (float*)[mm->buf_ps contents];
            for (int oh = 0; oh < og; oh++) {
                for (int ow = 0; ow < og; ow++) {
                    float* dst = psb + (size_t)(oh * og + ow) * (size_t)in_fc;
                    for (int dh = 0; dh < sf; dh++) {
                        for (int dw = 0; dw < sf; dw++) {
                            const float* sp = vx + (size_t)((oh * sf + dh) * grid + (ow * sf + dw)) * (size_t)mm->d_vis;
                            memcpy(dst + (size_t)(dh * sf + dw) * (size_t)mm->d_vis, sp, (size_t)mm->d_vis * sizeof(float));
                        }
                    }
                }
            }
            id<MTLCommandBuffer> pcmd = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> penc = [pcmd computeCommandEncoder];
            encode_gemm_batch_raw(penc, mm->mm_fc_w.buf, mm->mm_fc_w.offset, mm->mm_fc_w.ggml_type,
                                  mm->buf_ps, mm->buf_out, (uint32_t)out_dim, (uint32_t)in_fc, (uint32_t)n_out, 0);
            if (mm->mm_fc_b.buf != nil) encode_add_bias_batch(penc, mm->buf_out, &mm->mm_fc_b, (uint32_t)out_dim, (uint32_t)n_out);
            [penc endEncoding]; [pcmd commit]; [pcmd waitUntilCompleted];
            memcpy((void*)(uintptr_t)out_embd_ptr, [mm->buf_out contents], (size_t)n_out * (size_t)out_dim * sizeof(float));
        } else if (mm->mm_0_w.buf != nil && mm->mm_2_w.buf != nil) {
            uint32_t mid = (uint32_t)mm->mm_0_w.ne1;
            const float* vx = ((const float*)[mm->buf_x contents]) + (mm->has_cls ? mm->d_vis : 0);
            memcpy([mm->buf_ps contents], vx, (size_t)n_out * (size_t)mm->d_vis * sizeof(float));
            id<MTLCommandBuffer> pcmd = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> penc = [pcmd computeCommandEncoder];
            encode_gemm_batch_raw(penc, mm->mm_0_w.buf, mm->mm_0_w.offset, mm->mm_0_w.ggml_type,
                                  mm->buf_ps, mm->buf_ff, mid, (uint32_t)mm->d_vis, (uint32_t)n_out, 0);
            if (mm->mm_0_b.buf != nil) encode_add_bias_batch(penc, mm->buf_ff, &mm->mm_0_b, mid, (uint32_t)n_out);
            struct { uint32_t n, ut; } ga = { (uint32_t)n_out * mid, 1u };
            [penc setComputePipelineState:g_pso_vit_gelu];
            [penc setBuffer:mm->buf_ff offset:0 atIndex:0]; [penc setBytes:&ga length:sizeof(ga) atIndex:1];
            [penc dispatchThreads:MTLSizeMake((NSUInteger)ga.n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            encode_gemm_batch_raw(penc, mm->mm_2_w.buf, mm->mm_2_w.offset, mm->mm_2_w.ggml_type,
                                  mm->buf_ff, mm->buf_out, (uint32_t)out_dim, mid, (uint32_t)n_out, 0);
            if (mm->mm_2_b.buf != nil) encode_add_bias_batch(penc, mm->buf_out, &mm->mm_2_b, (uint32_t)out_dim, (uint32_t)n_out);
            [penc endEncoding]; [pcmd commit]; [pcmd waitUntilCompleted];
            memcpy((void*)(uintptr_t)out_embd_ptr, [mm->buf_out contents], (size_t)n_out * (size_t)out_dim * sizeof(float));
        } else {
            const float* vx = ((const float*)[mm->buf_x contents]) + (mm->has_cls ? mm->d_vis : 0);
            memcpy((void*)(uintptr_t)out_embd_ptr, vx, (size_t)n_out * (size_t)mm->d_vis * sizeof(float));
        }
        return (int64_t)n_out;
    }
}

int64_t tn_moe_mmproj_free(int64_t mm_h) {
    if (mm_h <= 0 || mm_h >= MMPROJ_MAX_MODELS || !g_mmproj[mm_h].active) return 0;
    @autoreleasepool {
        MmprojModel* mm = &g_mmproj[mm_h];
        mm->buf_in = nil; mm->buf_x = nil; mm->buf_xn = nil; mm->buf_q = nil; mm->buf_k = nil; mm->buf_v = nil;
        mm->buf_ctx = nil; mm->buf_ff = nil; mm->buf_ps = nil; mm->buf_out = nil; mm->buf_zero = nil;
        mm->mmap_buf = nil;
        if (mm->mmap_ptr && mm->mmap_ptr != MAP_FAILED) { munmap(mm->mmap_ptr, mm->mmap_len); mm->mmap_ptr = NULL; }
        if (mm->fd >= 0) { close(mm->fd); mm->fd = -1; }
        memset((void*)mm, 0, sizeof(MmprojModel));
    }
    return 1;
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

// =========================================================================
// Full Standard GGUF BPE Merge Tokenizer (`tokenizer.ggml.merges` + GPT-2 Byte-Unicode Codec!)
// =========================================================================
static uint64_t moe_fnv1a(const char* s, size_t n) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 1099511628211ull; }
    return h ? h : 1ull;
}

static int g_gpt2_init = 0;
static uint16_t g_gpt2_b2u[256];
static int16_t  g_gpt2_u2b[512];

static void moe_init_gpt2_byte_tables(void) {
    if (g_gpt2_init) return;
    for (int i = 0; i < 512; i++) g_gpt2_u2b[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            g_gpt2_b2u[b] = (uint16_t)b;
            g_gpt2_u2b[b] = (int16_t)b;
        } else {
            uint16_t cp = (uint16_t)(256 + n++);
            g_gpt2_b2u[b] = cp;
            g_gpt2_u2b[cp] = (int16_t)b;
        }
    }
    g_gpt2_init = 1;
}

static size_t moe_utf8_encode_cp(uint16_t cp, char* dst) {
    if (cp < 0x80) { dst[0] = (char)cp; return 1; }
    dst[0] = (char)(0xC0 | (cp >> 6));
    dst[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
}

static int32_t moe_tok_lookup(MoeModel* m, const char* s, size_t len) {
    if (!m->tok_ht_hash) return -1;
    uint64_t h = moe_fnv1a(s, len);
    uint32_t idx = (uint32_t)h & m->tok_ht_mask;
    for (;;) {
        uint64_t kh = m->tok_ht_hash[idx];
        if (kh == 0) return -1;
        if (kh == h) {
            int32_t id = m->tok_ht_id[idx];
            const char* tv = m->vocab_tokens[id];
            if (tv && strlen(tv) == len && memcmp(tv, s, len) == 0) return id;
        }
        idx = (idx + 1u) & m->tok_ht_mask;
    }
}

static int32_t moe_merge_lookup(MoeModel* m, int32_t a, int32_t b) {
    if (!m->merge_ht_key) return -1;
    uint64_t key = (((uint64_t)(uint32_t)a) << 32) | (uint64_t)(uint32_t)b;
    uint64_t h = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ull;
    uint32_t idx = (uint32_t)h & m->merge_ht_mask;
    for (;;) {
        uint64_t kk = m->merge_ht_key[idx];
        if (kk == 0xFFFFFFFFFFFFFFFFull) return -1;
        if (kk == key) return m->merge_ht_rank[idx];
        idx = (idx + 1u) & m->merge_ht_mask;
    }
}

static void moe_bpe_ensure_tables(MoeModel* m) {
    if (m->bpe_init || !m->vocab_tokens || m->vocab_size <= 0) return;
    moe_init_gpt2_byte_tables();
    m->bpe_use_sp = 0;
    for (int v = 0; v < m->vocab_size && v < 8192; v++) {
        const unsigned char* tv = (const unsigned char*)m->vocab_tokens[v];
        if (tv && tv[0] == 0xE2 && tv[1] == 0x96 && tv[2] == 0x81) { m->bpe_use_sp = 1; break; }
        if (tv && tv[0] == 0xC4 && tv[1] == 0xA0) { m->bpe_use_sp = 0; break; }
    }
    uint32_t tcap = 1024u;
    while (tcap < (uint32_t)m->vocab_size * 2u) tcap <<= 1;
    m->tok_ht_mask = tcap - 1u;
    m->tok_ht_hash = (uint64_t*)calloc(tcap, sizeof(uint64_t));
    m->tok_ht_id   = (int32_t*)calloc(tcap, sizeof(int32_t));
    for (int v = 0; v < m->vocab_size; v++) {
        const char* tv = m->vocab_tokens[v];
        if (!tv || !tv[0]) continue;
        size_t len = strlen(tv);
        uint64_t h = moe_fnv1a(tv, len);
        uint32_t idx = (uint32_t)h & m->tok_ht_mask;
        while (m->tok_ht_hash[idx] != 0) idx = (idx + 1u) & m->tok_ht_mask;
        m->tok_ht_hash[idx] = h;
        m->tok_ht_id[idx] = v;
    }
    if (m->n_bpe_merges > 0 && m->bpe_merges) {
        uint32_t mcap = 1024u;
        while (mcap < (uint32_t)m->n_bpe_merges * 2u) mcap <<= 1;
        m->merge_ht_mask = mcap - 1u;
        m->merge_ht_key  = (uint64_t*)malloc(mcap * sizeof(uint64_t));
        m->merge_ht_rank = (int32_t*)malloc(mcap * sizeof(int32_t));
        memset(m->merge_ht_key, 0xFF, mcap * sizeof(uint64_t));
        for (int r = 0; r < m->n_bpe_merges; r++) {
            const char* ms = m->bpe_merges[r];
            if (!ms) continue;
            const char* sp = strchr(ms, ' ');
            if (!sp) continue;
            int32_t a = moe_tok_lookup(m, ms, (size_t)(sp - ms));
            int32_t b = moe_tok_lookup(m, sp + 1, strlen(sp + 1));
            if (a < 0 || b < 0) continue;
            uint64_t key = (((uint64_t)(uint32_t)a) << 32) | (uint64_t)(uint32_t)b;
            uint64_t h = (key ^ (key >> 30)) * 0xbf58476d1ce4e5b9ull;
            uint32_t idx = (uint32_t)h & m->merge_ht_mask;
            while (m->merge_ht_key[idx] != 0xFFFFFFFFFFFFFFFFull) idx = (idx + 1u) & m->merge_ht_mask;
            m->merge_ht_key[idx] = key;
            m->merge_ht_rank[idx] = r;
        }
    }
    m->bpe_init = 1;
}

int64_t tn_moe_token_cstr(int64_t handle, int64_t tok_id) {
    static char g_tok_buf[512];
    g_tok_buf[0] = 0;
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return (int64_t)(uintptr_t)g_tok_buf;
    MoeModel* m = &g_moe[handle];
    if (!m->vocab_tokens || tok_id < 0 || tok_id >= m->vocab_size || !m->vocab_tokens[tok_id]) {
        return (int64_t)(uintptr_t)g_tok_buf;
    }
    moe_init_gpt2_byte_tables();
    const unsigned char* s = (const unsigned char*)m->vocab_tokens[tok_id];
    // Preserve special tokens like `<|eot_id|>` verbatim!
    if ((s[0] == '<' && (s[1] == '|' || s[1] == 's' || s[1] == '/')) || (s[0] == '[' && s[1] == 'g')) {
        strncpy(g_tok_buf, (const char*)s, sizeof(g_tok_buf) - 1);
        return (int64_t)(uintptr_t)g_tok_buf;
    }
    // Handle `<0xXX>` byte tokens (SentencePiece fallback)
    if (s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>' && s[6] == 0) {
        unsigned int bv = 0;
        if (sscanf((const char*)s + 3, "%2x", &bv) == 1) { g_tok_buf[0] = (char)bv; g_tok_buf[1] = 0; return (int64_t)(uintptr_t)g_tok_buf; }
    }
    size_t o = 0;
    for (size_t i = 0; s[i] && o + 4 < sizeof(g_tok_buf); ) {
        if (s[i] == 0xE2 && s[i + 1] == 0x96 && s[i + 2] == 0x81) { g_tok_buf[o++] = ' '; i += 3; }
        else if ((s[i] & 0xE0) == 0xC0 && s[i + 1]) {
            uint16_t cp = (uint16_t)(((s[i] & 0x1F) << 6) | (s[i + 1] & 0x3F));
            if (cp < 512 && g_gpt2_u2b[cp] >= 0) { g_tok_buf[o++] = (char)(uint8_t)g_gpt2_u2b[cp]; i += 2; }
            else { g_tok_buf[o++] = (char)s[i++]; g_tok_buf[o++] = (char)s[i++]; }
        } else {
            g_tok_buf[o++] = (char)s[i++];
        }
    }
    g_tok_buf[o] = 0;
    return (int64_t)(uintptr_t)g_tok_buf;
}

static int64_t moe_bpe_encode_segment(MoeModel* m, const unsigned char* seg, size_t seg_len, int64_t* out, int64_t max_out) {
    if (seg_len == 0 || max_out <= 0) return 0;
    if (m->n_bpe_merges > 0 && !m->bpe_use_sp) {
        // Exact GPT-2 Byte-Level BPE Merge Loop!
        int32_t* syms = (int32_t*)malloc((seg_len + 4) * sizeof(int32_t));
        size_t n_syms = 0;
        char u8[4];
        for (size_t i = 0; i < seg_len; i++) {
            size_t ulen = moe_utf8_encode_cp(g_gpt2_b2u[seg[i]], u8);
            int32_t id = moe_tok_lookup(m, u8, ulen);
            if (id >= 0) syms[n_syms++] = id;
        }
        char comb[512];
        while (n_syms >= 2) {
            int32_t best_rank = 0x7FFFFFFF;
            size_t best_i = (size_t)-1;
            int32_t best_merged = -1;
            for (size_t i = 0; i + 1 < n_syms; i++) {
                int32_t rk = moe_merge_lookup(m, syms[i], syms[i + 1]);
                if (rk >= 0 && rk < best_rank) {
                    const char* sa = m->vocab_tokens[syms[i]];
                    const char* sb = m->vocab_tokens[syms[i + 1]];
                    size_t la = strlen(sa), lb = strlen(sb);
                    if (la + lb < sizeof(comb)) {
                        memcpy(comb, sa, la); memcpy(comb + la, sb, lb);
                        int32_t mid = moe_tok_lookup(m, comb, la + lb);
                        if (mid >= 0) { best_rank = rk; best_i = i; best_merged = mid; }
                    }
                }
            }
            if (best_merged < 0) break;
            syms[best_i] = best_merged;
            memmove(syms + best_i + 1, syms + best_i + 2, (n_syms - best_i - 2) * sizeof(int32_t));
            n_syms--;
        }
        int64_t n_emit = 0;
        for (size_t i = 0; i < n_syms && n_emit < max_out; i++) out[n_emit++] = (int64_t)syms[i];
        free(syms);
        return n_emit;
    }
    // Hash-accelerated longest-match fallback (SentencePiece or models without merges array)
    char* enc = (char*)malloc(seg_len * 3u + 16u);
    size_t elen = 0;
    for (size_t i = 0; i < seg_len; i++) {
        if (seg[i] == ' ') {
            if (m->bpe_use_sp) { enc[elen++] = (char)0xE2; enc[elen++] = (char)0x96; enc[elen++] = (char)0x81; }
            else               { enc[elen++] = (char)0xC4; enc[elen++] = (char)0xA0; }
        } else if (seg[i] == '\n' && !m->bpe_use_sp) {
            enc[elen++] = (char)0xC4; enc[elen++] = (char)0x8A;
        } else if (!m->bpe_use_sp) {
            elen += moe_utf8_encode_cp(g_gpt2_b2u[seg[i]], enc + elen);
        } else {
            enc[elen++] = (char)seg[i];
        }
    }
    int64_t n_out = 0;
    size_t pos = 0;
    while (pos < elen && n_out < max_out) {
        size_t max_l = (elen - pos < 48u) ? (elen - pos) : 48u;
        int32_t found = -1; size_t flen = 0;
        for (size_t l = max_l; l >= 1; l--) {
            int32_t id = moe_tok_lookup(m, enc + pos, l);
            if (id >= 0) { found = id; flen = l; break; }
        }
        if (found >= 0) { out[n_out++] = (int64_t)found; pos += flen; }
        else {
            char hex[8]; snprintf(hex, sizeof(hex), "<0x%02X>", (unsigned char)enc[pos]);
            int32_t bid = moe_tok_lookup(m, hex, 6);
            if (bid >= 0) out[n_out++] = (int64_t)bid;
            pos++;
        }
    }
    free(enc);
    return n_out;
}

static int64_t moe_bpe_encode_chunked(MoeModel* m, const unsigned char* p, size_t rem, int64_t* out, int64_t max_out) {
    int64_t n_out = 0; size_t off = 0;
    while (off < rem && n_out < max_out) {
        size_t end = off + 1;
        if (p[off] == '\n') {
            while (end < rem && p[end] == '\n') end++;
        } else {
            if (p[off] == ' ') {
                while (end < rem && p[end] != ' ' && p[end] != '\n' && (end - off) < 64u) end++;
            } else {
                while (end < rem && p[end] != ' ' && p[end] != '\n' && (end - off) < 64u) end++;
            }
        }
        n_out += moe_bpe_encode_segment(m, p + off, end - off, out + n_out, max_out - n_out);
        off = end;
    }
    return n_out;
}

int64_t tn_moe_tokenize(int64_t handle, int64_t text_cstr, int64_t out_ids_ptr, int64_t max_tokens) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !text_cstr || !out_ids_ptr) return 0;
    MoeModel* m = &g_moe[handle];
    if (!m->vocab_tokens) return 0;
    moe_bpe_ensure_tables(m);
    const char* raw = (const char*)(uintptr_t)text_cstr;
    size_t rlen = strlen(raw);
    int64_t* out = (int64_t*)(uintptr_t)out_ids_ptr;
    int64_t n_out = 0;
    size_t seg_start = 0, i = 0;
    while (i < rlen && n_out < max_tokens) {
        int32_t sp_id = -1; size_t sp_len = 0;
        if (raw[i] == '<' || raw[i] == '[') {
            size_t max_s = (rlen - i < 32u) ? (rlen - i) : 32u;
            for (size_t l = max_s; l >= 3; l--) {
                if ((raw[i] == '<' && raw[i + l - 1] == '>') || (raw[i] == '[' && raw[i + l - 1] == ']')) {
                    int32_t id = moe_tok_lookup(m, raw + i, l);
                    if (id >= 0) { sp_id = id; sp_len = l; break; }
                }
            }
        }
        if (sp_id >= 0) {
            if (i > seg_start) {
                n_out += moe_bpe_encode_chunked(m, (const unsigned char*)(raw + seg_start), i - seg_start, out + n_out, max_tokens - n_out);
            }
            if (n_out < max_tokens) out[n_out++] = (int64_t)sp_id;
            i += sp_len;
            seg_start = i;
        } else {
            i++;
        }
    }
    if (rlen > seg_start && n_out < max_tokens) {
        n_out += moe_bpe_encode_chunked(m, (const unsigned char*)(raw + seg_start), rlen - seg_start, out + n_out, max_tokens - n_out);
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

static inline void purge_mtl_buf(id<MTLBuffer> buf, int is_nocopy) {
    if (!is_nocopy && buf != nil) {
        [buf setPurgeableState:MTLPurgeableStateEmpty];
    }
}

int64_t tn_moe_free(int64_t handle) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return 0;
    @autoreleasepool {
        MoeModel* m = &g_moe[handle];
        if (m->prefetch_grp) dispatch_group_wait(m->prefetch_grp, DISPATCH_TIME_FOREVER);
        if (m->dense_io_grp) dispatch_group_wait(m->dense_io_grp, DISPATCH_TIME_FOREVER);
        int nc = m->resident_mode;
        for (int L = 0; L < m->n_layers; L++) {
            MoeLayer* ly = &m->layers[L];
            if (ly->router_host) { free(ly->router_host); ly->router_host = NULL; }
            if (ly->ffn_norm_host) { free(ly->ffn_norm_host); ly->ffn_norm_host = NULL; }
            if (ly->exp_bias_host) { free(ly->exp_bias_host); ly->exp_bias_host = NULL; }
            purge_mtl_buf(ly->cache_buf, 0);
            purge_mtl_buf(ly->wq.buf, nc); purge_mtl_buf(ly->wk.buf, nc);
            purge_mtl_buf(ly->wv.buf, nc); purge_mtl_buf(ly->wo.buf, nc);
            purge_mtl_buf(ly->ffn_gate.buf, nc); purge_mtl_buf(ly->ffn_up.buf, nc); purge_mtl_buf(ly->ffn_down.buf, nc);
            purge_mtl_buf(ly->sh_gate.buf, nc); purge_mtl_buf(ly->sh_up.buf, nc); purge_mtl_buf(ly->sh_down.buf, nc);
            ly->cache_buf = nil; ly->k_cache = nil; ly->v_cache = nil;
            ly->mla_wuk_f16 = nil; ly->mla_wuv_f16 = nil;
            ly->wq_a.buf = nil; ly->q_a_norm.buf = nil; ly->wq_b.buf = nil;
            ly->wkv_a.buf = nil; ly->kv_a_norm.buf = nil; ly->wkv_b.buf = nil;
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
        if (m->bpe_merges) {
            for (int i = 0; i < m->n_bpe_merges; i++) free(m->bpe_merges[i]);
            free(m->bpe_merges);
            m->bpe_merges = NULL;
            m->n_bpe_merges = 0;
        }
        free(m->tok_ht_hash); m->tok_ht_hash = NULL;
        free(m->tok_ht_id); m->tok_ht_id = NULL;
        free(m->merge_ht_key); m->merge_ht_key = NULL;
        free(m->merge_ht_rank); m->merge_ht_rank = NULL;
        m->bpe_init = 0;
        m->mtp_enorm.buf = nil; m->mtp_hnorm.buf = nil; m->mtp_eh_proj.buf = nil;
        m->buf_last_hidden = nil; m->buf_mtp_cat = nil;
        if (m->mtp_layer) {
            m->mtp_layer->k_cache = nil; m->mtp_layer->v_cache = nil;
            free(m->mtp_layer); m->mtp_layer = NULL;
        }
        free(m->pa_refcnt); m->pa_refcnt = NULL;
        m->buf_pa_bt = nil; m->buf_cb_pos = nil; m->buf_cb_slot = nil;
        m->buf_batch_logits = nil; m->batch_logits_cap = 0;
        m->buf_tree_depth = nil; m->buf_tree_mask = nil;
        m->pbuf_mla_qabs = nil; m->pbuf_mla_olat = nil; m->buf_mtp_x = nil;
        for (int s = 0; s < 4; s++) { purge_mtl_buf(m->dense_ping[s], 0); m->dense_ping[s] = nil; }
        purge_mtl_buf(m->tok_embd.buf, nc); purge_mtl_buf(m->lm_head.buf, nc);
        m->tok_embd.buf = nil; m->out_norm.buf = nil; m->lm_head.buf = nil; m->rope_freqs.buf = nil;
        m->buf_emb_row = nil; m->buf_x = nil; m->buf_xn = nil; m->buf_q = nil; m->buf_k = nil; m->buf_v = nil;
        m->buf_ctx = nil; m->buf_rlogits = nil; m->buf_eidx = nil; m->buf_ewt = nil;
        m->buf_h = nil; m->buf_sh_g = nil; m->buf_sh_u = nil; m->buf_logits = nil; m->buf_argmax = nil;
        m->pbuf_x = nil; m->pbuf_xn = nil; m->pbuf_q = nil; m->pbuf_k = nil; m->pbuf_v = nil;
        m->pbuf_ctx = nil; m->pbuf_g = nil; m->pbuf_u = nil; m->pbuf_mla_q = nil; m->pbuf_mla_kv = nil;
        m->prefill_cap = 0;
        for (int s = 0; s < m->n_shards; s++) {
            m->mmap_mtl_bufs[s] = nil;
            if (m->mmap_ptrs[s] && m->mmap_ptrs[s] != MAP_FAILED) { munmap(m->mmap_ptrs[s], m->mmap_lens[s]); m->mmap_ptrs[s] = NULL; }
            if (m->fds[s] >= 0) { close(m->fds[s]); m->fds[s] = -1; }
        }
        m->active = 0;
    }
    return 1;
}
