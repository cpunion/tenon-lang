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
#define MOE_MAX_SEQ     131072
#define MOE_DEFAULT_SEQ 2048
#define MOE_PREFILL_MAX_B 512
#define MOE_FD_MAX_SPLITS 32
#define MOE_FD_MAX_B 32
#define PA_BLOCK_SIZE   16
#define PA_MAX_SEQS     32
#define PA_MAX_LOGICAL_BLKS (MOE_MAX_SEQ / PA_BLOCK_SIZE)
#define TREE_MAX_NODES  64
#define MOE_ICB_CACHE_MAX 8

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
#define TN_GGML_Q6_K     14
#define TN_GGML_IQ4_XS   23
#define TN_GGML_FP8_E4M3 39
#define TN_GGML_MXFP4    40
#define MOE_MAX_LORAS    16

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
    int max_ctx;         // Dynamic KV context limit (default 2048, up to 131072 / 128K!)
    int kv_quant_q8;     // 0=FP16 (2.0B), 1=Q8_0 (1.0625B), 2=FP8_E4M3 (1.0625B), 3=Q4_0 (0.5625B -> 3.56x!)
    int sliding_window;  // Tenon v10: Sliding Window Attention size (0 = full causal)
    int attn_sink;       // Tenon v10: Attention Sink token count kept at pos 0..attn_sink-1
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
    id<MTLBuffer> buf_cb_tok;       // [MOE_PREFILL_MAX_B] uint32 (Tenon v10 ICB indirect tok_ids)
    id<MTLBuffer> buf_cb_pos;       // [MOE_PREFILL_MAX_B] uint32
    id<MTLBuffer> buf_cb_slot;      // [MOE_PREFILL_MAX_B] uint32
    // Tenon v10/v11: Metal MTLIndirectCommandBuffer (ICB) Dual-Queue Async Persistent Decode Graph Cache
    int icb_enabled;
    int64_t icb_replays;
    int64_t icb_captured;
    int64_t icb_pipe_overlaps;
    id<MTLSharedEvent> icb_event;
    uint64_t icb_ev_val;
    // Tenon v11/v12: Hardware-Native GQA/MLA Head-Grouped FlashDecoding Split-KV + Metal 4 MTLResidencySet
    int fd_enabled;
    int fd_min_len;
    int fd_splits;
    int fd_splits_forced;
    int cur_use_fd;
    int cur_fd_splits;
    int cur_is_prefill;
    int batch_logits_dirty_row;
    id residency_set;
    int64_t flash_decode_launches;
    id<MTLBuffer> buf_fd_o; // [min(cap,32) * n_heads * MOE_FD_MAX_SPLITS * max(d_head, kv_r)] F32
    id<MTLBuffer> buf_fd_m; // [min(cap,32) * n_heads * MOE_FD_MAX_SPLITS] F32
    id<MTLBuffer> buf_fd_l; // [min(cap,32) * n_heads * MOE_FD_MAX_SPLITS] F32
    struct {
        int valid;
        uint32_t B;
        uint64_t lora_sig;
        int use_fd;
        int fd_splits;
        uint32_t n_cmds;
        id<MTLIndirectCommandBuffer> icb;
        id<MTLBuffer> arg_buf;
        id<MTLBuffer> __strong * res_list;
        int res_count;
    } icb_cache[MOE_ICB_CACHE_MAX];
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
    // Tenon v7: 2D/3D MRoPE spatial position support (Qwen2.5-VL / Gemma-3)
    int has_mrope;
    int mrope_sections[3];
    int mrope_pos[3];
    // Tenon v7: Zero-Copy Runtime Multi-Adapter GGUF LoRA Hot-Swapping
    struct {
        int active;
        int rank;
        float alpha;
        float user_scale;
        int n_tensors;
        int64_t vram_bytes;
        id<MTLBuffer> a[MOE_MAX_LAYERS][7]; // 0=q,1=k,2=v,3=o,4=gate,5=up,6=down; layout [rank, in_dim] F32
        id<MTLBuffer> b[MOE_MAX_LAYERS][7]; // layout [out_dim, rank] F32
        int r_dim[MOE_MAX_LAYERS][7];
    } loras[MOE_MAX_LORAS];
    int active_lora;
    int batch_lora_ids[MOE_PREFILL_MAX_B];
    int use_batch_lora;
    int cur_exec_layer;
    id<MTLBuffer> buf_lora_tmp;       // [MOE_PREFILL_MAX_B * 256] F32
    id<MTLBuffer> buf_batch_lora_ids; // [MOE_PREFILL_MAX_B] uint32 (Tenon v9 S-LoRA/Punica BGMV)
    id<MTLBuffer> buf_lora_dummy;     // 64B dummy buffer for inactive BGMV slots
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
    id<MTLBuffer> pbuf_mla_qabs; // [cap * n_heads * kv_lora_rank] F32
    id<MTLBuffer> pbuf_mla_olat; // [cap * n_heads * kv_lora_rank] F32
    id<MTLBuffer> pbuf_rlogits;     // [cap * n_experts] F32 (Tenon v8/v9 MoE batched router)
    id<MTLBuffer> pbuf_moe_x;       // [cap * d_model] F32
    id<MTLBuffer> pbuf_moe_h;       // [cap * top_k * d_ff_exp] F32 (Tenon v9 Mega-Kernel hidden)
    id<MTLBuffer> pbuf_moe_y;       // [cap * top_k * d_model] F32 (Tenon v9 Mega-Kernel output)
    id<MTLBuffer> buf_moe_idx;      // [cap * MOE_MAX_TOPK] uint32
    id<MTLBuffer> buf_moe_wt;       // [cap * MOE_MAX_TOPK] F32
    id<MTLBuffer> buf_moe_active_e; // [MOE_MAX_EXPERTS] uint32 (compact active expert IDs)
    id<MTLBuffer> buf_moe_grp_off;  // [MOE_MAX_EXPERTS] uint32
    id<MTLBuffer> buf_moe_grp_cnt;  // [MOE_MAX_EXPERTS] uint32
    id<MTLBuffer> buf_moe_tok_slot; // [cap * MOE_MAX_TOPK] int32 ((b,k) -> compact slot)
    id<MTLBuffer> buf_moe_meta;     // [4] uint32: [0]=n_uniq, [1]=saved_reads_acc, [2]=mega_launches
    id<MTLBuffer> buf_moe_zero;     // 64B dummy buffer when exp_bias absent
    int64_t moe_grouped_saved_reads;
    id<MTLBuffer> buf_mtp_x;        // [d_model] F32 scratch for MTP self-speculative head

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
static id<MTLCommandQueue> g_moe_aux_q = nil;
static id<MTLLibrary> g_moe_lib = nil;
static id<MTLComputePipelineState> g_pso_embed = nil;
static id<MTLComputePipelineState> g_pso_rmsnorm = nil;
static id<MTLComputePipelineState> g_pso_rmsnorm_strided = nil;
static id<MTLComputePipelineState> g_pso_gemv_qkv_q4 = nil;
static id<MTLComputePipelineState> g_pso_flash_prefill_paged_fp16 = nil;
static id<MTLComputePipelineState> g_pso_flash_prefill_amx64 = nil;
static id<MTLComputePipelineState> g_pso_flash_prefill_amx128 = nil;
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
static id<MTLComputePipelineState> g_pso_dense_gu_q4k = nil;
static id<MTLComputePipelineState> g_pso_dense_gu_q5k = nil;
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
static id<MTLComputePipelineState> g_pso_gemm_q4_qkv_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q4_gu_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q4_amx16 = nil;
static id<MTLComputePipelineState> g_pso_gemm_q4_amx8 = nil;
static id<MTLComputePipelineState> g_pso_gemm_q4_sb = nil;
static id<MTLComputePipelineState> g_pso_gemm_q4_1_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q4_1_amx8 = nil;
static id<MTLComputePipelineState> g_pso_gemm_q4_1_sb = nil;
static id<MTLComputePipelineState> g_pso_gemm_q5_0_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q5_1_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q8_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q2k_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q3k_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_iq4xs_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q4k_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q4k_amx16 = nil;
static id<MTLComputePipelineState> g_pso_gemm_q4k_sb = nil;
static id<MTLComputePipelineState> g_pso_gemm_q5k_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q6k_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_q6k_amx16 = nil;
static id<MTLComputePipelineState> g_pso_gemm_q6k_sb = nil;
static id<MTLComputePipelineState> g_pso_gemv_fp8 = nil;
static id<MTLComputePipelineState> g_pso_gemm_fp8_b = nil;
static id<MTLComputePipelineState> g_pso_gemv_mxfp4 = nil;
static id<MTLComputePipelineState> g_pso_gemm_mxfp4_b = nil;
static id<MTLComputePipelineState> g_pso_lora_a = nil;
static id<MTLComputePipelineState> g_pso_lora_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_f16_b = nil;
static id<MTLComputePipelineState> g_pso_gemm_f16_amx = nil;
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
static id<MTLComputePipelineState> g_pso_vit_attn_amx = nil;
static id<MTLComputePipelineState> g_pso_gemm_q6k_amx = nil;
static id<MTLComputePipelineState> g_pso_gemm_fp8_amx = nil;
static id<MTLComputePipelineState> g_pso_gemm_mxfp4_amx = nil;
static id<MTLComputePipelineState> g_pso_gemm_q8_amx = nil;
static id<MTLComputePipelineState> g_pso_moe_route_sort = nil;
static id<MTLComputePipelineState> g_pso_moe_mega_gu = nil;
static id<MTLComputePipelineState> g_pso_moe_mega_down = nil;
static id<MTLComputePipelineState> g_pso_moe_mega_reduce = nil;
static id<MTLComputePipelineState> g_pso_lora_bgmv_a = nil;
static id<MTLComputePipelineState> g_pso_lora_bgmv_b = nil;
static id<MTLComputePipelineState> g_pso_mla_rope_store_paged = nil;
static id<MTLComputePipelineState> g_pso_mla_absorb_q_b = nil;
static id<MTLComputePipelineState> g_pso_mla_flash_lat_paged = nil;
static id<MTLComputePipelineState> g_pso_mla_proj_v_b = nil;
static id<MTLComputePipelineState> g_pso_gather_rows = nil;
static id<MTLComputePipelineState> g_pso_scatter_add_rows = nil;
static id<MTLComputePipelineState> g_pso_embed_batch = nil;
static id<MTLComputePipelineState> g_pso_moe_mega_gu_amx = nil;
static id<MTLComputePipelineState> g_pso_moe_mega_down_amx = nil;
static id<MTLComputePipelineState> g_pso_rope_kv_fp8_paged = nil;
static id<MTLComputePipelineState> g_pso_flash_dec_fp8_paged = nil;
static id<MTLComputePipelineState> g_pso_rope_kv_q4_paged = nil;
static id<MTLComputePipelineState> g_pso_flash_dec_q4_paged = nil;
static id<MTLComputePipelineState> g_pso_flash_dec_split_paged = nil;
static id<MTLComputePipelineState> g_pso_flash_mla_split_paged = nil;
static id<MTLComputePipelineState> g_pso_flash_mla_split_paged_512 = nil;
static id<MTLComputePipelineState> g_pso_flash_dec_reduce_paged = nil;
static id<MTLComputePipelineState> g_pso_argmax = nil;
static id<MTLComputePipelineState> g_pso_argmax_b = nil;

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
        case TN_GGML_Q6_K:     return (n_elems / 256ull) * 210ull;
        case TN_GGML_IQ4_XS:   return (n_elems / 256ull) * 136ull;
        case TN_GGML_FP8_E4M3: return (n_elems / 32ull) * 34ull;
        case TN_GGML_MXFP4:    return (n_elems / 32ull) * 17ull;
        default:               return n_elems * 2ull;
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

// Vectorized 64-bit `packed_ushort4 + float4` GGML Q5_0 dot (22 bytes / 32 elems: half d + uchar qh[4] + uchar qs[16])
static inline float dot_q5_0_fast(const device uchar* blk, const device float* x) {
    float d = float(*(const device half*)blk);
    uint qh = uint(blk[2u]) | (uint(blk[3u]) << 8) | (uint(blk[4u]) << 16) | (uint(blk[5u]) << 24);
    const device packed_ushort4* q64 = (const device packed_ushort4*)(blk + 6u);
    ushort4 p0 = q64[0], p1 = q64[1];
    uchar4 c0 = as_type<uchar4>(p0.xy), c1 = as_type<uchar4>(p0.zw);
    uchar4 c2 = as_type<uchar4>(p1.xy), c3 = as_type<uchar4>(p1.zw);
    const device float4* xf = (const device float4*)x;
    uint4 b0 = uint4(0u, 1u, 2u, 3u);
    float4 s = float4(0.0f);
    #define Q50_STEP(CV, SH, XLO, XHI) do { \
        uint4 lo_b = ((uint4(qh) >> ((SH) + b0)) & 1u) << 4; \
        uint4 hi_b = ((uint4(qh) >> ((SH) + 16u + b0)) & 1u) << 4; \
        float4 ql = float4(int4((uint4((CV) & uchar4(0x0Fu)) | lo_b)) - 16); \
        float4 qhi = float4(int4((uint4((CV) >> 4) | hi_b)) - 16); \
        s = fma(ql, (XLO), fma(qhi, (XHI), s)); \
    } while (0)
    Q50_STEP(c0,  0u, xf[0], xf[4]);
    Q50_STEP(c1,  4u, xf[1], xf[5]);
    Q50_STEP(c2,  8u, xf[2], xf[6]);
    Q50_STEP(c3, 12u, xf[3], xf[7]);
    #undef Q50_STEP
    return d * ((s.x + s.y) + (s.z + s.w));
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

// Tenon v12 Round 7: 448 GB/s half4 Noshift (0x0F/0xF0) GGML Q4_K quarter-block (4 lanes/sb, overflow-proof FP32 scale!)
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
        const device uchar4* q4 = (const device uchar4*)(qs + (j >> 1) + l0);
        const device float4* xl = (const device float4*)(x + j + l0);
        const device float4* xh = (const device float4*)(x + j + 32u + l0);
        half4 vlo0 = half4(xl[0]), vlo1 = half4(xl[1]);
        half4 vhi0 = half4(xh[0]), vhi1 = half4(xh[1]);
        half4 sx0 = vlo0 + vlo1, sx1 = vhi0 + vhi1;
        vhi0 *= 0.0625h; vhi1 *= 0.0625h;
        uchar4 b0 = q4[0], b1 = q4[1];
        half4 p0 = fma(half4(b0 & uchar4(0x0Fu)), vlo0, half4(b1 & uchar4(0x0Fu)) * vlo1);
        half4 p1 = fma(half4(b0 & uchar4(0xF0u)), vhi0, half4(b1 & uchar4(0xF0u)) * vhi1);
        float sp0 = float(p0.x + p0.y) + float(p0.z + p0.w);
        float sp1 = float(p1.x + p1.y) + float(p1.z + p1.w);
        float ss0 = float(sx0.x + sx0.y) + float(sx0.z + sx0.w);
        float ss1 = float(sx1.x + sx1.y) + float(sx1.z + sx1.w);
        sum += fma(d * float(sc0), sp0, -dmin * float(m0) * ss0)
             + fma(d * float(sc1), sp1, -dmin * float(m1) * ss1);
        is += 2u;
    }
    return sum;
}

// Tenon v13 T2.2b: Zero-Branch Hoisted-qh half4 GGML Q5_K quarter-block (4x fewer qh loads, NO select!)
static inline float dot_q5_k_quarter(const device uchar* blk, const device float* x, uint qtr) {
    float d    = float(*(const device half*)(blk));
    float dmin = float(*(const device half*)(blk + 2u));
    const device uchar* sc = blk + 4u;
    const device uchar* qs = blk + 48u;
    uint l0 = qtr * 8u;
    const device uchar4* hp = (const device uchar4*)(blk + 16u + l0);
    uchar4 h0 = hp[0], h1 = hp[1];
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
        const device uchar4* qp = (const device uchar4*)(qs + (j >> 1) + l0);
        const device float4* xl = (const device float4*)(x + j + l0);
        const device float4* xh = (const device float4*)(x + j + 32u + l0);
        half4 vlo0 = half4(xl[0]), vlo1 = half4(xl[1]);
        half4 vhi0 = half4(xh[0]), vhi1 = half4(xh[1]);
        half4 sx0 = vlo0 + vlo1, sx1 = vhi0 + vhi1;
        uchar4 q0 = qp[0], q1 = qp[1];
        half4 qlo0 = half4((q0 & uchar4(0x0Fu)) | ((h0 & uchar4(1u)) << 4));
        half4 qlo1 = half4((q1 & uchar4(0x0Fu)) | ((h1 & uchar4(1u)) << 4));
        half4 qhi0 = half4((q0 >> 4)            | ((h0 & uchar4(2u)) << 3));
        half4 qhi1 = half4((q1 >> 4)            | ((h1 & uchar4(2u)) << 3));
        h0 >>= 2; h1 >>= 2;
        half4 p0 = fma(qlo0, vlo0, qlo1 * vlo1);
        half4 p1 = fma(qhi0, vhi0, qhi1 * vhi1);
        float sp0 = float(p0.x + p0.y) + float(p0.z + p0.w);
        float sp1 = float(p1.x + p1.y) + float(p1.z + p1.w);
        float ss0 = float(sx0.x + sx0.y) + float(sx0.z + sx0.w);
        float ss1 = float(sx1.x + sx1.y) + float(sx1.z + sx1.w);
        sum += fma(d * float(sc0), sp0, -dmin * float(m0) * ss0)
             + fma(d * float(sc1), sp1, -dmin * float(m1) * ss1);
        is += 2u;
    }
    return sum;
}

// Tenon v12 Round 8: Aligned 32-bit packed_ushort2 Q6_K Quarter-Block (4x fewer LD ops, overflow-proof FP32!)
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
        const device uchar* qlp = ql + (n >> 1) + l0;
        const device uchar* qhp = qh + (n >> 2) + l0;
        const device char*  scp = sc + (n >> 4);
        const device float4* xp = (const device float4*)(x + n + l0);
        float s0 = float(scp[is + 0]), s2 = float(scp[is + 2]);
        float s4 = float(scp[is + 4]), s6 = float(scp[is + 6]);
        float4 p0 = float4(0.0f), p2 = float4(0.0f), p4 = float4(0.0f), p6 = float4(0.0f);
        #pragma unroll
        for (uint g = 0; g < 2u; g++) {
            uint o = g * 4u;
            uchar4 qa  = as_type<uchar4>(*(const device packed_ushort2*)(qlp + o));
            uchar4 qb  = as_type<uchar4>(*(const device packed_ushort2*)(qlp + o + 32u));
            uchar4 qh4 = as_type<uchar4>(*(const device packed_ushort2*)(qhp + o));
            float4 w1 = float4(int4((qa & uchar4(0x0Fu)) | (((qh4 >> 0) & uchar4(3u)) << 4)) - 32);
            float4 w2 = float4(int4((qb & uchar4(0x0Fu)) | (((qh4 >> 2) & uchar4(3u)) << 4)) - 32);
            float4 w3 = float4(int4((qa >> 4)            | (((qh4 >> 4) & uchar4(3u)) << 4)) - 32);
            float4 w4 = float4(int4((qb >> 4)            | (((qh4 >> 6) & uchar4(3u)) << 4)) - 32);
            p0 = fma(w1, xp[g],       p0);
            p2 = fma(w2, xp[g + 8u],  p2);
            p4 = fma(w3, xp[g + 16u], p4);
            p6 = fma(w4, xp[g + 24u], p6);
        }
        sum += s0 * ((p0.x + p0.y) + (p0.z + p0.w))
             + s2 * ((p2.x + p2.y) + (p2.z + p2.w))
             + s4 * ((p4.x + p4.y) + (p4.z + p4.w))
             + s6 * ((p6.x + p6.y) + (p6.z + p6.w));
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
    if (gtype == 39u) { // FP8_E4M3 (34B / 32 elems: half d + 32 x OCP E4M3FN)
        ulong rb = (ulong)(in_dim >> 5) * 34ul;
        const device uchar* bp = w + row * rb + (ulong)(col >> 5) * 34ul;
        float d = float(*(const device half*)bp);
        uchar b = bp[2u + (col & 31u)];
        uint s = b >> 7, e = (b >> 3) & 0xFu, m = b & 7u;
        float v = (e == 0u) ? (float(m) * 0.001953125f) : (ldexp(1.0f + float(m) * 0.125f, int(e) - 7));
        return d * (s ? -v : v);
    }
    if (gtype == 40u) { // OCP MXFP4 (17B / 32 elems: uchar e8m0 + 16 x E2M1 nibble pairs)
        const float kMx4[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };
        ulong rb = (ulong)(in_dim >> 5) * 17ul;
        const device uchar* bp = w + row * rb + (ulong)(col >> 5) * 17ul;
        float sc = as_type<float>(uint(bp[0]) << 23);
        uint rem = col & 31u;
        uchar bv = bp[1u + (rem & 15u)];
        uint qi = (rem < 16u) ? (bv & 0xFu) : (bv >> 4);
        return sc * kMx4[qi];
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

struct MoeEmbedBatchArgs { uint d_model; uint ggml_type; uint batch_b; };
kernel void moe_embed_lookup_batch(
    const device uchar* emb [[buffer(0)]],
    const device uint* tok_ids [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant MoeEmbedBatchArgs& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= p.d_model || gid.y >= p.batch_b) return;
    uint tid = tok_ids[gid.y];
    out[(ulong)gid.y * p.d_model + gid.x] = dequant_any_elem(emb, (ulong)tid, gid.x, p.d_model, p.ggml_type);
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

struct MoeNormStridedArgs { uint dim; uint stride; uint n_groups; float eps; };
kernel void moe_rmsnorm_strided(
    device float* x [[buffer(0)]],
    const device float* w [[buffer(1)]],
    constant MoeNormStridedArgs& p [[buffer(2)]],
    uint g [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (g >= p.n_groups) return;
    threadgroup float s_sq[8];
    ulong base = (ulong)g * p.stride;
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
    for (uint i = lid; i < p.dim; i += 256u) {
        x[base + i] = x[base + i] * inv * w[i];
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
// Tenon v12 Round 7: 465 GB/s (93% M4 Max DRAM Peak!) 4tpb half4 Noshift Q4_0 GEMV (4 rows/SG, 2 SGs/TG = 8 rows/TG)
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
    uint sub = lane & 3u, qoff = sub * 4u;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float4* xf = (const device float4*)(x + (b << 5));
        half4 xl = half4(xf[sub]), xh = half4(xf[sub + 4u]);
        half4 sm = xl + xh;
        half xs = -8.0h * ((sm.x + sm.y) + (sm.z + sm.w));
        xh *= 0.0625h;
        ulong boff = (ulong)b * 18ul;
        #define Q4_4TPB(WP, ACC) do { \
            const device uchar* bp = (WP) + boff; \
            half d = *(const device half*)bp; \
            uchar4 c = as_type<uchar4>(*(const device packed_ushort2*)(bp + 2u + qoff)); \
            half4 s = fma(half4(c & uchar4(0x0Fu)), xl, half4(c & uchar4(0xF0u)) * xh); \
            (ACC) += float(d * (((s.x + s.y) + (s.z + s.w)) + xs)); \
        } while (0)
        Q4_4TPB(wp0, a0); Q4_4TPB(wp1, a1); Q4_4TPB(wp2, a2); Q4_4TPB(wp3, a3);
        #undef Q4_4TPB
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row0 + 1u < p.out_dim) y[row0 + 1u] = p.add_res ? (y[row0 + 1u] + a1) : a1;
        if (row0 + 2u < p.out_dim) y[row0 + 2u] = p.add_res ? (y[row0 + 2u] + a2) : a2;
        if (row0 + 3u < p.out_dim) y[row0 + 3u] = p.add_res ? (y[row0 + 3u] + a3) : a3;
    }
}

// Tenon v12 Round 7: 465 GB/s 4tpb half4 Noshift Q4_1 GEMV (4 rows/SG, 2 SGs/TG = 8 rows/TG)
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
    uint sub = lane & 3u, qoff = sub * 4u;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float4* xf = (const device float4*)(x + (b << 5));
        half4 xl = half4(xf[sub]), xh = half4(xf[sub + 4u]);
        half4 sm = xl + xh;
        half sx = (sm.x + sm.y) + (sm.z + sm.w);
        xh *= 0.0625h;
        ulong boff = (ulong)b * 20ul;
        #define Q41_4TPB(WP, ACC) do { \
            const device uchar* bp = (WP) + boff; \
            half2 dm = *(const device half2*)bp; \
            uchar4 c = as_type<uchar4>(*(const device packed_ushort2*)(bp + 4u + qoff)); \
            half4 s = fma(half4(c & uchar4(0x0Fu)), xl, half4(c & uchar4(0xF0u)) * xh); \
            (ACC) += float(fma(dm.x, (s.x + s.y) + (s.z + s.w), dm.y * sx)); \
        } while (0)
        Q41_4TPB(wp0, a0); Q41_4TPB(wp1, a1); Q41_4TPB(wp2, a2); Q41_4TPB(wp3, a3);
        #undef Q41_4TPB
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

// Tenon v12 Round 7: 497 GB/s (99.4% M4 Max DRAM Peak!) 2tpb FP32-Safe Q8_0 GEMV (4 rows/SG, 2 SGs/TG = 8 rows/TG)
kernel void moe_gemv_q8(
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
    ulong rb = (ulong)nb * 34ul;
    const device uchar* wp0 = w + (ulong)row0 * rb;
    const device uchar* wp1 = (row0 + 1u < p.out_dim) ? (wp0 + rb) : wp0;
    const device uchar* wp2 = (row0 + 2u < p.out_dim) ? (wp0 + rb * 2ul) : wp0;
    const device uchar* wp3 = (row0 + 3u < p.out_dim) ? (wp0 + rb * 3ul) : wp0;
    uint sub = lane & 1u, qoff = sub * 16u;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint b = (lane >> 1); b < nb; b += 16u) {
        const device float4* xf = (const device float4*)(x + (b << 5) + qoff);
        float4 x0 = xf[0], x1 = xf[1], x2 = xf[2], x3 = xf[3];
        ulong boff = (ulong)b * 34ul;
        #define Q8_2TPB(WP, ACC) do { \
            const device uchar* blk = (WP) + boff; \
            float d = float(*(const device half*)blk); \
            const device packed_short4* q = (const device packed_short4*)(blk + 2u + qoff); \
            short4 p0 = q[0], p1 = q[1]; \
            float s = dot(float4(as_type<char4>(p0.xy)), x0) + dot(float4(as_type<char4>(p0.zw)), x1) \
                    + dot(float4(as_type<char4>(p1.xy)), x2) + dot(float4(as_type<char4>(p1.zw)), x3); \
            (ACC) = fma(d, s, (ACC)); \
        } while (0)
        Q8_2TPB(wp0, a0); Q8_2TPB(wp1, a1); Q8_2TPB(wp2, a2); Q8_2TPB(wp3, a3);
        #undef Q8_2TPB
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row0 + 1u < p.out_dim) y[row0 + 1u] = p.add_res ? (y[row0 + 1u] + a1) : a1;
        if (row0 + 2u < p.out_dim) y[row0 + 2u] = p.add_res ? (y[row0 + 2u] + a2) : a2;
        if (row0 + 3u < p.out_dim) y[row0 + 3u] = p.add_res ? (y[row0 + 3u] + a3) : a3;
    }
}

// Tenon v13 Pass 2: 8tpb half4 Noshift 4-Row/SG Q4_K GEMV (tptg=64, 8 rows/TG -> +62% on 3072x2048!)
kernel void moe_gemv_q4k(
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
    uint nb = p.in_dim >> 8;
    ulong rb = (ulong)nb * 144ul;
    const device uchar* wp0 = w + (ulong)row0 * rb;
    const device uchar* wp1 = (row0 + 1u < p.out_dim) ? (wp0 + rb) : wp0;
    const device uchar* wp2 = (row0 + 2u < p.out_dim) ? (wp0 + rb * 2ul) : wp0;
    const device uchar* wp3 = (row0 + 3u < p.out_dim) ? (wp0 + rb * 3ul) : wp0;
    uint sub = lane & 7u, qtr = sub >> 1, hi = sub & 1u;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint b = (lane >> 3); b < nb; b += 4u) {
        const device float4* xp = (const device float4*)(x + (b << 8) + sub * 32u);
        half4 x0 = half4(xp[0]), x1 = half4(xp[1]), x2 = half4(xp[2]), x3 = half4(xp[3]);
        half4 x4 = half4(xp[4]), x5 = half4(xp[5]), x6 = half4(xp[6]), x7 = half4(xp[7]);
        half4 sm = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7));
        float sx = float((sm.x + sm.y) + (sm.z + sm.w));
        if (hi) { x0 *= 0.0625h; x1 *= 0.0625h; x2 *= 0.0625h; x3 *= 0.0625h; x4 *= 0.0625h; x5 *= 0.0625h; x6 *= 0.0625h; x7 *= 0.0625h; }
        ulong boff = (ulong)b * 144ul;
        #define Q4K_8TPB(WP, ACC) do { \
            const device uchar* blk = (WP) + boff; \
            float d = float(*(const device half*)blk), dmin = float(*(const device half*)(blk + 2u)); \
            const device uchar* sc = blk + 4u; \
            uchar s0, m0; \
            if (sub < 4u) { s0 = sc[sub] & 63u; m0 = sc[sub + 4u] & 63u; } \
            else { s0 = (sc[sub + 4u] & 0xFu) | ((sc[sub - 4u] >> 6) << 4); m0 = (sc[sub + 4u] >> 4) | ((sc[sub] >> 6) << 4); } \
            const device packed_ushort4* qp = (const device packed_ushort4*)(blk + 16u + qtr * 32u); \
            ushort4 p0 = qp[0], p1 = qp[1], p2 = qp[2], p3 = qp[3]; \
            half4 acc; \
            if (hi) { \
                acc = fma(half4(as_type<uchar4>(p0.xy) & uchar4(0xF0u)), x0, half4(as_type<uchar4>(p0.zw) & uchar4(0xF0u)) * x1); \
                acc = fma(half4(as_type<uchar4>(p1.xy) & uchar4(0xF0u)), x2, fma(half4(as_type<uchar4>(p1.zw) & uchar4(0xF0u)), x3, acc)); \
                acc = fma(half4(as_type<uchar4>(p2.xy) & uchar4(0xF0u)), x4, fma(half4(as_type<uchar4>(p2.zw) & uchar4(0xF0u)), x5, acc)); \
                acc = fma(half4(as_type<uchar4>(p3.xy) & uchar4(0xF0u)), x6, fma(half4(as_type<uchar4>(p3.zw) & uchar4(0xF0u)), x7, acc)); \
            } else { \
                acc = fma(half4(as_type<uchar4>(p0.xy) & uchar4(0x0Fu)), x0, half4(as_type<uchar4>(p0.zw) & uchar4(0x0Fu)) * x1); \
                acc = fma(half4(as_type<uchar4>(p1.xy) & uchar4(0x0Fu)), x2, fma(half4(as_type<uchar4>(p1.zw) & uchar4(0x0Fu)), x3, acc)); \
                acc = fma(half4(as_type<uchar4>(p2.xy) & uchar4(0x0Fu)), x4, fma(half4(as_type<uchar4>(p2.zw) & uchar4(0x0Fu)), x5, acc)); \
                acc = fma(half4(as_type<uchar4>(p3.xy) & uchar4(0x0Fu)), x6, fma(half4(as_type<uchar4>(p3.zw) & uchar4(0x0Fu)), x7, acc)); \
            } \
            (ACC) += fma(d * float(s0), float((acc.x + acc.y) + (acc.z + acc.w)), -dmin * float(m0) * sx); \
        } while (0)
        Q4K_8TPB(wp0, a0); Q4K_8TPB(wp1, a1); Q4K_8TPB(wp2, a2); Q4K_8TPB(wp3, a3);
        #undef Q4K_8TPB
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    if (lane == 0u) {
        y[row0] = p.add_res ? (y[row0] + a0) : a0;
        if (row0 + 1u < p.out_dim) y[row0 + 1u] = p.add_res ? (y[row0 + 1u] + a1) : a1;
        if (row0 + 2u < p.out_dim) y[row0 + 2u] = p.add_res ? (y[row0 + 2u] + a2) : a2;
        if (row0 + 3u < p.out_dim) y[row0 + 3u] = p.add_res ? (y[row0 + 3u] + a3) : a3;
    }
}

// Tenon v13 T2.2b: 2-SG High-Occupancy Q5_K GEMV (tptg=64, 2 rows/SG = 4 rows/TG, zero-spill 4tpb!)
kernel void moe_gemv_q5k(
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

// Tenon v12 Round 8: 2-SG (tptg=64, 4 rows/TG) High-Occupancy Q6_K GEMV (450+ GB/s on lm_head!)
kernel void moe_gemv_q6k(
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

// Tenon v12 Round 7: 465 GB/s 4tpb half4 Noshift Fused Dense SwiGLU Gate || Up Q4_0 (NSG=2, 2 rows/SG = 4 rows/TG)
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
    uint sub = lane & 3u, qoff = sub * 4u;
    float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float4* xf = (const device float4*)(xn + (b << 5));
        half4 xl = half4(xf[sub]), xh = half4(xf[sub + 4u]);
        half4 sm = xl + xh;
        half xs = -8.0h * ((sm.x + sm.y) + (sm.z + sm.w));
        xh *= 0.0625h;
        ulong boff = (ulong)b * 18ul;
        #define DGU_4TPB(WP, ACC) do { \
            const device uchar* bp = (WP) + boff; \
            half d = *(const device half*)bp; \
            uchar4 c = as_type<uchar4>(*(const device packed_ushort2*)(bp + 2u + qoff)); \
            half4 s = fma(half4(c & uchar4(0x0Fu)), xl, half4(c & uchar4(0xF0u)) * xh); \
            (ACC) += float(d * (((s.x + s.y) + (s.z + s.w)) + xs)); \
        } while (0)
        DGU_4TPB(g0, ag0); DGU_4TPB(u0, au0); DGU_4TPB(g1, ag1); DGU_4TPB(u1, au1);
        #undef DGU_4TPB
    }
    ag0 = simd_sum(ag0); au0 = simd_sum(au0);
    ag1 = simd_sum(ag1); au1 = simd_sum(au1);
    if (lane == 0u) {
        h_out[row0] = (ag0 / (1.0f + exp(-ag0))) * au0;
        if (row1 < p.out_dim) h_out[row1] = (ag1 / (1.0f + exp(-ag1))) * au1;
    }
}

// Tenon v13 T2.1: 2-SG High-Occupancy Zero-Spill Fused Dense SwiGLU Gate || Up Q4_K (tptg=64, 4 rows/TG!)
kernel void moe_dense_gate_up_q4k(
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
    uint nb = p.in_dim >> 8, qtr = lane & 3u;
    ulong rb = (ulong)nb * 144ul;
    const device uchar* g0 = wg + (ulong)row0 * rb;
    const device uchar* g1 = (row1 < p.out_dim) ? (g0 + rb) : g0;
    const device uchar* u0 = wu + (ulong)row0 * rb;
    const device uchar* u1 = (row1 < p.out_dim) ? (u0 + rb) : u0;
    float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float* xp = xn + (b << 8);
        ulong boff = (ulong)b * 144ul;
        ag0 += dot_q4_k_quarter(g0 + boff, xp, qtr);
        au0 += dot_q4_k_quarter(u0 + boff, xp, qtr);
        if (row1 < p.out_dim) {
            ag1 += dot_q4_k_quarter(g1 + boff, xp, qtr);
            au1 += dot_q4_k_quarter(u1 + boff, xp, qtr);
        }
    }
    ag0 = simd_sum(ag0); au0 = simd_sum(au0);
    ag1 = simd_sum(ag1); au1 = simd_sum(au1);
    if (lane == 0u) {
        h_out[row0] = (ag0 / (1.0f + exp(-ag0))) * au0;
        if (row1 < p.out_dim) h_out[row1] = (ag1 / (1.0f + exp(-ag1))) * au1;
    }
}

// Tenon v13 T2.2b: 2-SG High-Occupancy Zero-Spill Fused Dense SwiGLU Gate || Up Q5_K (tptg=64, 4 rows/TG!)
kernel void moe_dense_gate_up_q5k(
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
    uint nb = p.in_dim >> 8, qtr = lane & 3u;
    ulong rb = (ulong)nb * 176ul;
    const device uchar* g0 = wg + (ulong)row0 * rb;
    const device uchar* g1 = (row1 < p.out_dim) ? (g0 + rb) : g0;
    const device uchar* u0 = wu + (ulong)row0 * rb;
    const device uchar* u1 = (row1 < p.out_dim) ? (u0 + rb) : u0;
    float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float* xp = xn + (b << 8);
        ulong boff = (ulong)b * 176ul;
        ag0 += dot_q5_k_quarter(g0 + boff, xp, qtr);
        au0 += dot_q5_k_quarter(u0 + boff, xp, qtr);
        if (row1 < p.out_dim) {
            ag1 += dot_q5_k_quarter(g1 + boff, xp, qtr);
            au1 += dot_q5_k_quarter(u1 + boff, xp, qtr);
        }
    }
    ag0 = simd_sum(ag0); au0 = simd_sum(au0);
    ag1 = simd_sum(ag1); au1 = simd_sum(au1);
    if (lane == 0u) {
        h_out[row0] = (ag0 / (1.0f + exp(-ag0))) * au0;
        if (row1 < p.out_dim) h_out[row1] = (ag1 / (1.0f + exp(-ag1))) * au1;
    }
}

struct MoeQkvArgs { uint q_dim; uint kv_dim; uint in_dim; uint has_bias; };
// Tenon v13 Pass 2: 465 GB/s 4tpb half4 Noshift Fused Q || K || V + Optional QKV Bias Q4_0 GEMV!
kernel void moe_gemv_qkv_q4(
    const device uchar* wq [[buffer(0)]],
    const device uchar* wk [[buffer(1)]],
    const device uchar* wv [[buffer(2)]],
    const device float* x  [[buffer(3)]],
    device float* yq [[buffer(4)]],
    device float* yk [[buffer(5)]],
    device float* yv [[buffer(6)]],
    constant MoeQkvArgs& p [[buffer(7)]],
    const device float* bq [[buffer(8)]],
    const device float* bk [[buffer(9)]],
    const device float* bv [[buffer(10)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 2u + sg) * 4u;
    uint tot = p.q_dim + 2u * p.kv_dim;
    if (row0 >= tot) return;
    const device uchar* wp;
    device float* yp;
    const device float* bp;
    uint lrow, ldim;
    if (row0 < p.q_dim) {
        wp = wq; yp = yq; bp = bq; lrow = row0; ldim = p.q_dim;
    } else if (row0 < p.q_dim + p.kv_dim) {
        wp = wk; yp = yk; bp = bk; lrow = row0 - p.q_dim; ldim = p.kv_dim;
    } else {
        wp = wv; yp = yv; bp = bv; lrow = row0 - p.q_dim - p.kv_dim; ldim = p.kv_dim;
    }
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 18ul;
    const device uchar* r0 = wp + (ulong)lrow * rb;
    const device uchar* r1 = (lrow + 1u < ldim) ? (r0 + rb) : r0;
    const device uchar* r2 = (lrow + 2u < ldim) ? (r0 + rb * 2ul) : r0;
    const device uchar* r3 = (lrow + 3u < ldim) ? (r0 + rb * 3ul) : r0;
    uint sub = lane & 3u, qoff = sub * 4u;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float4* xf = (const device float4*)(x + (b << 5));
        half4 xl = half4(xf[sub]), xh = half4(xf[sub + 4u]);
        half4 sm = xl + xh;
        half xs = -8.0h * ((sm.x + sm.y) + (sm.z + sm.w));
        xh *= 0.0625h;
        ulong boff = (ulong)b * 18ul;
        #define QKV_4TPB(WP, ACC) do { \
            const device uchar* bp_ = (WP) + boff; \
            half d = *(const device half*)bp_; \
            uchar4 c = as_type<uchar4>(*(const device packed_ushort2*)(bp_ + 2u + qoff)); \
            half4 s = fma(half4(c & uchar4(0x0Fu)), xl, half4(c & uchar4(0xF0u)) * xh); \
            (ACC) += float(d * (((s.x + s.y) + (s.z + s.w)) + xs)); \
        } while (0)
        QKV_4TPB(r0, a0); QKV_4TPB(r1, a1); QKV_4TPB(r2, a2); QKV_4TPB(r3, a3);
        #undef QKV_4TPB
    }
    a0 = simd_sum(a0); a1 = simd_sum(a1); a2 = simd_sum(a2); a3 = simd_sum(a3);
    if (lane == 0u) {
        if (p.has_bias) {
            yp[lrow] = a0 + bp[lrow];
            if (lrow + 1u < ldim) yp[lrow + 1u] = a1 + bp[lrow + 1u];
            if (lrow + 2u < ldim) yp[lrow + 2u] = a2 + bp[lrow + 2u];
            if (lrow + 3u < ldim) yp[lrow + 3u] = a3 + bp[lrow + 3u];
        } else {
            yp[lrow] = a0;
            if (lrow + 1u < ldim) yp[lrow + 1u] = a1;
            if (lrow + 2u < ldim) yp[lrow + 2u] = a2;
            if (lrow + 3u < ldim) yp[lrow + 3u] = a3;
        }
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
    uint mrope_en;
    uint mrope_s0;
    uint mrope_s1;
    uint mrope_t;
    uint mrope_h;
    uint mrope_w;
};

// Universal RoPE (Full/Partial, NORM/NEOX, Optional Llama-3 rope_freqs divisors, 2D/3D MRoPE) + FP16 KV-cache!
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
        uint eff_pos = p.pos;
        if (p.mrope_en) {
            eff_pos = (i < p.mrope_s0) ? p.mrope_t : ((i < p.mrope_s0 + p.mrope_s1) ? p.mrope_h : p.mrope_w);
        }
        float ang = float(eff_pos) * inv_f;
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

// Tenon v12 Round 7: 32-Lane Warp-Parallel MoE Router Top-K (supports Softmax, Sigmoid+Bias, DeepSeek Group Top-K!)
kernel void moe_router_topk(
    const device float* logits [[buffer(0)]],
    const device float* exp_bias [[buffer(1)]],
    device uint* out_idx [[buffer(2)]],
    device float* out_wt [[buffer(3)]],
    constant MoeRouterArgs& p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float s_probs[256];
    threadgroup float s_scores[256];
    uint ne = min(p.n_experts, 256u);
    if (p.gating_func == 2u) {
        for (uint e = lane; e < ne; e += 32u) {
            float s = 1.0f / (1.0f + exp(-logits[e]));
            s_probs[e] = s;
            s_scores[e] = p.has_bias ? (s + exp_bias[e]) : s;
        }
    } else {
        float lmx = -1e30f;
        for (uint e = lane; e < ne; e += 32u) lmx = max(lmx, logits[e]);
        float mx = simd_max(lmx);
        float lsum = 0.0f;
        for (uint e = lane; e < ne; e += 32u) {
            float v = exp(logits[e] - mx);
            s_probs[e] = v;
            lsum += v;
        }
        float inv = 1.0f / max(simd_sum(lsum), 1e-12f);
        for (uint e = lane; e < ne; e += 32u) {
            float pr = s_probs[e] * inv;
            s_probs[e] = pr;
            s_scores[e] = pr;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (p.n_groups > 1u && p.n_group_used > 0u && p.n_group_used < p.n_groups) {
        if (lane == 0u) {
            uint gsz = ne / p.n_groups;
            float gsc[32];
            bool gkeep[32];
            for (uint g = 0; g < p.n_groups && g < 32u; g++) {
                float m1 = -1e30f, m2 = -1e30f;
                for (uint i = 0; i < gsz; i++) {
                    float sv = s_scores[g * gsz + i];
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
                if (!gkeep[g]) for (uint i = 0; i < gsz; i++) s_scores[g * gsz + i] = -1e30f;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    float wsum = 0.0f;
    for (uint k = 0; k < p.top_k; k++) {
        float bv = -1e30f; uint bi = 0u;
        for (uint e = lane; e < ne; e += 32u) {
            float sv = s_scores[e];
            if (sv > bv) { bv = sv; bi = e; }
        }
        for (uint off = 16u; off > 0u; off >>= 1) {
            float ov = simd_shuffle_xor(bv, off);
            uint oi = simd_shuffle_xor(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0u) {
            out_idx[k] = bi;
            out_wt[k] = s_probs[bi];
            wsum += s_probs[bi];
            s_scores[bi] = -1e30f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        float mult = p.w_scale;
        if (p.norm_weights && wsum > 1e-12f) mult /= wsum;
        for (uint k = 0; k < p.top_k; k++) out_wt[k] *= mult;
    }
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

// Tenon v12 Round 7: 465 GB/s 4tpb half4 Noshift Top-K Gate || Up + SwiGLU Q4_0!
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
    uint sub = lane & 3u, qoff = sub * 4u;
    float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float4* xf = (const device float4*)(xn + (b << 5));
        half4 xl = half4(xf[sub]), xh = half4(xf[sub + 4u]);
        half4 sm = xl + xh;
        half xs = -8.0h * ((sm.x + sm.y) + (sm.z + sm.w));
        xh *= 0.0625h;
        ulong boff = (ulong)b * 18ul;
        #define TGU_4TPB(WP, ACC) do { \
            const device uchar* bp = (WP) + boff; \
            half d = *(const device half*)bp; \
            uchar4 c = as_type<uchar4>(*(const device packed_ushort2*)(bp + 2u + qoff)); \
            half4 s = fma(half4(c & uchar4(0x0Fu)), xl, half4(c & uchar4(0xF0u)) * xh); \
            (ACC) += float(d * (((s.x + s.y) + (s.z + s.w)) + xs)); \
        } while (0)
        TGU_4TPB(wg0, ag0); TGU_4TPB(wu0, au0); TGU_4TPB(wg1, ag1); TGU_4TPB(wu1, au1);
        #undef TGU_4TPB
    }
    ag0 = simd_sum(ag0); au0 = simd_sum(au0);
    ag1 = simd_sum(ag1); au1 = simd_sum(au1);
    if (lane == 0u) {
        h_out[(ulong)slot * p.d_ff + row0] = (ag0 / (1.0f + exp(-ag0))) * au0;
        if (row1 < p.d_ff) h_out[(ulong)slot * p.d_ff + row1] = (ag1 / (1.0f + exp(-ag1))) * au1;
    }
}

// Tenon v12 Round 7: 465 GB/s Half-SG (16-lane) x 4tpb half4 Noshift Q4_0 Down-Accum (100% SIMD util on nb=44!)
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
    uint sg_half = lane >> 4;
    uint hlane   = lane & 15u;
    uint sub = hlane & 3u, qoff = sub * 4u;
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    for (uint s = sg_half; s < p.n_active; s += 2u) {
        uint eid = e_idx[s];
        float w = e_wt[s];
        const device uchar* wd0 = down_exps + (ulong)eid * p.down_exp_bytes + (ulong)row0 * rb;
        const device uchar* wd1 = (row0 + 1u < p.d_model) ? (wd0 + rb) : wd0;
        const device uchar* wd2 = (row0 + 2u < p.d_model) ? (wd0 + rb * 2ul) : wd0;
        const device uchar* wd3 = (row0 + 3u < p.d_model) ? (wd0 + rb * 3ul) : wd0;
        const device float* hp = h_in + (ulong)s * p.d_ff;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint b = (hlane >> 2); b < nb; b += 4u) {
            const device float4* xf = (const device float4*)(hp + (b << 5));
            half4 xl = half4(xf[sub]), xh = half4(xf[sub + 4u]);
            half4 sm = xl + xh;
            half xs = -8.0h * ((sm.x + sm.y) + (sm.z + sm.w));
            xh *= 0.0625h;
            ulong boff = (ulong)b * 18ul;
            #define TDN_4TPB(WP, ACC) do { \
                const device uchar* bp = (WP) + boff; \
                half d = *(const device half*)bp; \
                uchar4 c = as_type<uchar4>(*(const device packed_ushort2*)(bp + 2u + qoff)); \
                half4 sv = fma(half4(c & uchar4(0x0Fu)), xl, half4(c & uchar4(0xF0u)) * xh); \
                (ACC) += float(d * (((sv.x + sv.y) + (sv.z + sv.w)) + xs)); \
            } while (0)
            TDN_4TPB(wd0, a0); TDN_4TPB(wd1, a1); TDN_4TPB(wd2, a2); TDN_4TPB(wd3, a3);
            #undef TDN_4TPB
        }
        t0 = fma(w, a0, t0); t1 = fma(w, a1, t1);
        t2 = fma(w, a2, t2); t3 = fma(w, a3, t3);
    }
    t0 = simd_sum(t0); t1 = simd_sum(t1); t2 = simd_sum(t2); t3 = simd_sum(t3);
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
    uint sg_half = lane >> 4;
    uint hlane   = lane & 15u;
    uint sub = hlane & 3u, qoff = sub * 4u;
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    for (uint s = sg_half; s < p.n_active; s += 2u) {
        uint eid = e_idx[s];
        float w = e_wt[s];
        const device uchar* wd0 = down_exps + (ulong)eid * p.down_exp_bytes + (ulong)row0 * rb;
        const device uchar* wd1 = (row0 + 1u < p.d_model) ? (wd0 + rb) : wd0;
        const device uchar* wd2 = (row0 + 2u < p.d_model) ? (wd0 + rb * 2ul) : wd0;
        const device uchar* wd3 = (row0 + 3u < p.d_model) ? (wd0 + rb * 3ul) : wd0;
        const device float* hp = h_in + (ulong)s * p.d_ff;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint b = (hlane >> 2); b < nb; b += 4u) {
            const device float4* xf = (const device float4*)(hp + (b << 5));
            half4 xl = half4(xf[sub]), xh = half4(xf[sub + 4u]);
            half4 sm = xl + xh;
            half sx = (sm.x + sm.y) + (sm.z + sm.w);
            xh *= 0.0625h;
            ulong boff = (ulong)b * 20ul;
            #define TDN41_4TPB(WP, ACC) do { \
                const device uchar* bp = (WP) + boff; \
                half2 dm = *(const device half2*)bp; \
                uchar4 c = as_type<uchar4>(*(const device packed_ushort2*)(bp + 4u + qoff)); \
                half4 sv = fma(half4(c & uchar4(0x0Fu)), xl, half4(c & uchar4(0xF0u)) * xh); \
                (ACC) += float(fma(dm.x, (sv.x + sv.y) + (sv.z + sv.w), dm.y * sx)); \
            } while (0)
            TDN41_4TPB(wd0, a0); TDN41_4TPB(wd1, a1); TDN41_4TPB(wd2, a2); TDN41_4TPB(wd3, a3);
            #undef TDN41_4TPB
        }
        t0 = fma(w, a0, t0); t1 = fma(w, a1, t1);
        t2 = fma(w, a2, t2); t3 = fma(w, a3, t3);
    }
    t0 = simd_sum(t0); t1 = simd_sum(t1); t2 = simd_sum(t2); t3 = simd_sum(t3);
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
    uint sub = lane & 3u, qoff = sub * 4u;
    float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float4* xf = (const device float4*)(xn + (b << 5));
        half4 xl = half4(xf[sub]), xh = half4(xf[sub + 4u]);
        half4 sm = xl + xh;
        half xs = -8.0h * ((sm.x + sm.y) + (sm.z + sm.w));
        xh *= 0.0625h;
        ulong boff = (ulong)b * 18ul;
        #define SGU_4TPB(WP, ACC) do { \
            const device uchar* bp = (WP) + boff; \
            half d = *(const device half*)bp; \
            uchar4 c = as_type<uchar4>(*(const device packed_ushort2*)(bp + 2u + qoff)); \
            half4 sv = fma(half4(c & uchar4(0x0Fu)), xl, half4(c & uchar4(0xF0u)) * xh); \
            (ACC) += float(d * (((sv.x + sv.y) + (sv.z + sv.w)) + xs)); \
        } while (0)
        SGU_4TPB(wg0, ag0); SGU_4TPB(wu0, au0); SGU_4TPB(wg1, ag1); SGU_4TPB(wu1, au1);
        #undef SGU_4TPB
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
    uint sg_half = lane >> 4, hlane = lane & 15u, sub = hlane & 3u, qoff = sub * 4u;
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    for (uint s = sg_half; s < p.n_active; s += 2u) {
        uint cslot = slot_ids[s];
        float w = e_wt[s];
        const device uchar* wd0 = cache_pool + (ulong)cslot * p.slot_stride_bytes + down_off + (ulong)row0 * rb;
        const device uchar* wd1 = (row0 + 1u < p.d_model) ? (wd0 + rb) : wd0;
        const device uchar* wd2 = (row0 + 2u < p.d_model) ? (wd0 + rb * 2ul) : wd0;
        const device uchar* wd3 = (row0 + 3u < p.d_model) ? (wd0 + rb * 3ul) : wd0;
        const device float* hp = h_in + (ulong)s * p.d_ff;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint b = (hlane >> 2); b < nb; b += 4u) {
            const device float4* xf = (const device float4*)(hp + (b << 5));
            half4 xl = half4(xf[sub]), xh = half4(xf[sub + 4u]);
            half4 sm = xl + xh;
            half xs = -8.0h * ((sm.x + sm.y) + (sm.z + sm.w));
            xh *= 0.0625h;
            ulong boff = (ulong)b * 18ul;
            #define SDN_4TPB(WP, ACC) do { \
                const device uchar* bp = (WP) + boff; \
                half d = *(const device half*)bp; \
                uchar4 c = as_type<uchar4>(*(const device packed_ushort2*)(bp + 2u + qoff)); \
                half4 sv = fma(half4(c & uchar4(0x0Fu)), xl, half4(c & uchar4(0xF0u)) * xh); \
                (ACC) += float(d * (((sv.x + sv.y) + (sv.z + sv.w)) + xs)); \
            } while (0)
            SDN_4TPB(wd0, a0); SDN_4TPB(wd1, a1); SDN_4TPB(wd2, a2); SDN_4TPB(wd3, a3);
            #undef SDN_4TPB
        }
        t0 = fma(w, a0, t0); t1 = fma(w, a1, t1);
        t2 = fma(w, a2, t2); t3 = fma(w, a3, t3);
    }
    t0 = simd_sum(t0); t1 = simd_sum(t1); t2 = simd_sum(t2); t3 = simd_sum(t3);
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
    uint sg_half = lane >> 4, hlane = lane & 15u, sub = hlane & 3u, qoff = sub * 4u;
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;
    for (uint s = sg_half; s < p.n_active; s += 2u) {
        uint cslot = slot_ids[s];
        float w = e_wt[s];
        const device uchar* wd0 = cache_pool + (ulong)cslot * p.slot_stride_bytes + down_off + (ulong)row0 * rb;
        const device uchar* wd1 = (row0 + 1u < p.d_model) ? (wd0 + rb) : wd0;
        const device uchar* wd2 = (row0 + 2u < p.d_model) ? (wd0 + rb * 2ul) : wd0;
        const device uchar* wd3 = (row0 + 3u < p.d_model) ? (wd0 + rb * 3ul) : wd0;
        const device float* hp = h_in + (ulong)s * p.d_ff;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint b = (hlane >> 2); b < nb; b += 4u) {
            const device float4* xf = (const device float4*)(hp + (b << 5));
            half4 xl = half4(xf[sub]), xh = half4(xf[sub + 4u]);
            half4 sm = xl + xh;
            half sx = (sm.x + sm.y) + (sm.z + sm.w);
            xh *= 0.0625h;
            ulong boff = (ulong)b * 20ul;
            #define SDN41_4TPB(WP, ACC) do { \
                const device uchar* bp = (WP) + boff; \
                half2 dm = *(const device half2*)bp; \
                uchar4 c = as_type<uchar4>(*(const device packed_ushort2*)(bp + 4u + qoff)); \
                half4 sv = fma(half4(c & uchar4(0x0Fu)), xl, half4(c & uchar4(0xF0u)) * xh); \
                (ACC) += float(fma(dm.x, (sv.x + sv.y) + (sv.z + sv.w), dm.y * sx)); \
            } while (0)
            SDN41_4TPB(wd0, a0); SDN41_4TPB(wd1, a1); SDN41_4TPB(wd2, a2); SDN41_4TPB(wd3, a3);
            #undef SDN41_4TPB
        }
        t0 = fma(w, a0, t0); t1 = fma(w, a1, t1);
        t2 = fma(w, a2, t2); t3 = fma(w, a3, t3);
    }
    t0 = simd_sum(t0); t1 = simd_sum(t1); t2 = simd_sum(t2); t3 = simd_sum(t3);
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
            if (s_v[s] > best_v || (s_v[s] == best_v && s_i[s] < best_i)) { best_v = s_v[s]; best_i = s_i[s]; }
        }
        out_tok[0] = best_i;
    }
}

// Tenon v12 Round 6: Parallel Batched GPU Argmax (eliminates B*128K serial CPU vocab scan in tn_moe_step_batch!)
kernel void moe_argmax_batch_f32(
    const device float* logits [[buffer(0)]],
    device uint* out_tok [[buffer(1)]],
    constant uint2& vb [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint b = tgid;
    if (b >= vb.y) return;
    const device float* row = logits + (ulong)b * (ulong)vb.x;
    threadgroup float s_v[8];
    threadgroup uint  s_i[8];
    float bv = -1e30f;
    uint bi = 0u;
    for (uint i = lid; i < vb.x; i += 256u) {
        float v = row[i];
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
            if (s_v[s] > best_v || (s_v[s] == best_v && s_i[s] < best_i)) { best_v = s_v[s]; best_i = s_i[s]; }
        }
        out_tok[b] = best_i;
    }
}

// =========================================================================
// Q4_K MoE Resident & Streaming-Slot Kernels (Qwen3-MoE / DeepSeek-R1 Q4_K_M!)
// =========================================================================
// Tenon v13 T2.1: 2-SG High-Occupancy Zero-Spill Q4_K MoE TopK Gate+Up (tptg=64, 4 rows/TG!)
kernel void moe_topk_gate_up_q4k(
    const device uchar* gate_exps [[buffer(0)]], const device uchar* up_exps [[buffer(1)]],
    const device float* xn [[buffer(2)]], const device uint* eidx [[buffer(3)]],
    device float* h_out [[buffer(4)]], constant MoeExpArgs& p [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid.x * 2u + sg) * 2u, row1 = row0 + 1u;
    uint k = tgid.y;
    if (row0 >= p.d_ff || k >= p.n_active) return;
    uint eid = eidx[k]; uint nb = p.d_model >> 8; ulong rb = (ulong)nb * 144ul;
    const device uchar* g0 = gate_exps + (ulong)eid * p.gate_exp_bytes + (ulong)row0 * rb;
    const device uchar* u0 = up_exps   + (ulong)eid * p.up_exp_bytes   + (ulong)row0 * rb;
    const device uchar* g1 = (row1 < p.d_ff) ? (g0 + rb) : g0;
    const device uchar* u1 = (row1 < p.d_ff) ? (u0 + rb) : u0;
    uint qtr = lane & 3u;
    float ag0 = 0.0f, au0 = 0.0f, ag1 = 0.0f, au1 = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device float* xp = xn + (b << 8); ulong boff = (ulong)b * 144ul;
        ag0 += dot_q4_k_quarter(g0 + boff, xp, qtr);
        au0 += dot_q4_k_quarter(u0 + boff, xp, qtr);
        if (row1 < p.d_ff) {
            ag1 += dot_q4_k_quarter(g1 + boff, xp, qtr);
            au1 += dot_q4_k_quarter(u1 + boff, xp, qtr);
        }
    }
    ag0 = simd_sum(ag0); au0 = simd_sum(au0);
    ag1 = simd_sum(ag1); au1 = simd_sum(au1);
    if (lane == 0u) {
        h_out[k * p.d_ff + row0] = (ag0 / (1.0f + exp(-ag0))) * au0;
        if (row1 < p.d_ff) h_out[k * p.d_ff + row1] = (ag1 / (1.0f + exp(-ag1))) * au1;
    }
}

kernel void moe_topk_down_accum_q4k(
    const device uchar* down_exps [[buffer(0)]], const device float* h_in [[buffer(1)]],
    const device uint* eidx [[buffer(2)]], const device float* ewt [[buffer(3)]],
    device float* x_io [[buffer(4)]], constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 2u + sg) * 2u, row1 = row0 + 1u;
    if (row0 >= p.d_model) return;
    uint nb = p.d_ff >> 8; ulong rb = (ulong)nb * 144ul; uint qtr = lane & 3u;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (uint k = 0; k < p.n_active; k++) {
        const device uchar* wd0 = down_exps + (ulong)eidx[k] * p.down_exp_bytes + (ulong)row0 * rb;
        const device uchar* wd1 = (row1 < p.d_model) ? (wd0 + rb) : wd0;
        const device float* hk = h_in + (ulong)k * p.d_ff;
        float s0 = 0.0f, s1 = 0.0f;
        for (uint b = (lane >> 2); b < nb; b += 8u) {
            const device float* xp = hk + (b << 8); ulong boff = (ulong)b * 144ul;
            s0 += dot_q4_k_quarter(wd0 + boff, xp, qtr);
            if (row1 < p.d_model) s1 += dot_q4_k_quarter(wd1 + boff, xp, qtr);
        }
        float w = ewt[k];
        acc0 += w * simd_sum(s0);
        acc1 += w * simd_sum(s1);
    }
    if (lane == 0u) {
        x_io[row0] += acc0;
        if (row1 < p.d_model) x_io[row1] += acc1;
    }
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

// Tenon v12 Round 8: 2-Row/Half-SG (8 rows/TG, tptg=64, ILP=2, 2x h_in reuse) FP32-Safe Q5_0 Down-Accum!
kernel void moe_topk_down_accum_q5_0(
    const device uchar* down_exps [[buffer(0)]], const device float* h_in [[buffer(1)]],
    const device uint* e_idx [[buffer(2)]], const device float* e_wt [[buffer(3)]],
    device float* x_res [[buffer(4)]], constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 4u + sg * 2u + (lane >> 4)) * 2u;
    uint row1 = row0 + 1u;
    uint l16 = lane & 15u;
    if (row0 >= p.d_model) return;
    uint nb = p.d_ff >> 5; ulong rb = (ulong)nb * 22ul;
    float acc0 = 0.0f, acc1 = 0.0f;
    uint sub = l16 & 3u, qoff = sub * 4u, sh_lo = sub * 4u, sh_hi = 16u + sub * 4u;
    for (uint k = 0; k < p.n_active; k++) {
        const device uchar* wd0 = down_exps + (ulong)e_idx[k] * p.down_exp_bytes + (ulong)row0 * rb;
        const device uchar* wd1 = (row1 < p.d_model) ? (wd0 + rb) : wd0;
        const device float* hk = h_in + (ulong)k * p.d_ff;
        float s0 = 0.0f, s1 = 0.0f;
        for (uint b = (l16 >> 2); b < nb; b += 4u) {
            const device float4* xf = (const device float4*)(hk + (b << 5));
            half4 xl = half4(xf[sub]), xh = half4(xf[sub + 4u]);
            half4 sm = xl + xh;
            half xs = -16.0h * ((sm.x + sm.y) + (sm.z + sm.w));
            ulong boff = (ulong)b * 22ul;
            #define TDN_Q5_1(WD, SC) do { \
                const device uchar* bp = (WD) + boff; \
                half d = *(const device half*)bp; \
                uint qh = uint(*(const device ushort*)(bp + 2u)) | (uint(*(const device ushort*)(bp + 4u)) << 16); \
                uchar4 c = as_type<uchar4>(*(const device packed_ushort2*)(bp + 6u + qoff)); \
                uchar4 hl = uchar4((qh >> sh_lo) & 1u, (qh >> (sh_lo + 1u)) & 1u, (qh >> (sh_lo + 2u)) & 1u, (qh >> (sh_lo + 3u)) & 1u) << 4; \
                uchar4 hh = uchar4((qh >> sh_hi) & 1u, (qh >> (sh_hi + 1u)) & 1u, (qh >> (sh_hi + 2u)) & 1u, (qh >> (sh_hi + 3u)) & 1u) << 4; \
                half4 sv = fma(half4((c & uchar4(0x0Fu)) | hl), xl, half4((c >> 4) | hh) * xh); \
                (SC) += float(d * (((sv.x + sv.y) + (sv.z + sv.w)) + xs)); \
            } while (0)
            TDN_Q5_1(wd0, s0);
            TDN_Q5_1(wd1, s1);
            #undef TDN_Q5_1
        }
        s0 += simd_shuffle_xor(s0, 8u); s0 += simd_shuffle_xor(s0, 4u); s0 += simd_shuffle_xor(s0, 2u); s0 += simd_shuffle_xor(s0, 1u);
        s1 += simd_shuffle_xor(s1, 8u); s1 += simd_shuffle_xor(s1, 4u); s1 += simd_shuffle_xor(s1, 2u); s1 += simd_shuffle_xor(s1, 1u);
        float w = e_wt[k];
        acc0 = fma(w, s0, acc0);
        acc1 = fma(w, s1, acc1);
    }
    if (l16 == 0u) {
        x_res[row0] += acc0;
        if (row1 < p.d_model) x_res[row1] += acc1;
    }
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

// Tenon v12 Round 8: 2-Row/Half-SG (8 rows/TG, tptg=64, ILP=2, 497 GB/s) FP32-Safe Q8_0 Down-Accum!
kernel void moe_topk_down_accum_q8(
    const device uchar* down_exps [[buffer(0)]], const device float* h_in [[buffer(1)]],
    const device uint* e_idx [[buffer(2)]], const device float* e_wt [[buffer(3)]],
    device float* x_res [[buffer(4)]], constant MoeExpArgs& p [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row0 = (tgid * 4u + sg * 2u + (lane >> 4)) * 2u;
    uint row1 = row0 + 1u;
    uint l16 = lane & 15u;
    if (row0 >= p.d_model) return;
    uint nb = p.d_ff >> 5; ulong rb = (ulong)nb * 34ul;
    float acc0 = 0.0f, acc1 = 0.0f;
    uint sub = l16 & 1u, qoff = sub * 16u;
    for (uint k = 0; k < p.n_active; k++) {
        const device uchar* wd0 = down_exps + (ulong)e_idx[k] * p.down_exp_bytes + (ulong)row0 * rb;
        const device uchar* wd1 = (row1 < p.d_model) ? (wd0 + rb) : wd0;
        const device float* hk = h_in + (ulong)k * p.d_ff;
        float s0 = 0.0f, s1 = 0.0f;
        for (uint b = (l16 >> 1); b < nb; b += 8u) {
            const device float4* xf = (const device float4*)(hk + (b << 5) + qoff);
            float4 x0 = xf[0], x1 = xf[1], x2 = xf[2], x3 = xf[3];
            ulong boff = (ulong)b * 34ul;
            #define TDN_Q8_1(WD, SC) do { \
                const device uchar* blk = (WD) + boff; \
                float d = float(*(const device half*)blk); \
                const device packed_short4* q = (const device packed_short4*)(blk + 2u + qoff); \
                short4 p0 = q[0], p1 = q[1]; \
                float sv = dot(float4(as_type<char4>(p0.xy)), x0) + dot(float4(as_type<char4>(p0.zw)), x1) \
                         + dot(float4(as_type<char4>(p1.xy)), x2) + dot(float4(as_type<char4>(p1.zw)), x3); \
                (SC) = fma(d, sv, (SC)); \
            } while (0)
            TDN_Q8_1(wd0, s0);
            TDN_Q8_1(wd1, s1);
            #undef TDN_Q8_1
        }
        s0 += simd_shuffle_xor(s0, 8u); s0 += simd_shuffle_xor(s0, 4u); s0 += simd_shuffle_xor(s0, 2u); s0 += simd_shuffle_xor(s0, 1u);
        s1 += simd_shuffle_xor(s1, 8u); s1 += simd_shuffle_xor(s1, 4u); s1 += simd_shuffle_xor(s1, 2u); s1 += simd_shuffle_xor(s1, 1u);
        float w = e_wt[k];
        acc0 = fma(w, s0, acc0);
        acc1 = fma(w, s1, acc1);
    }
    if (l16 == 0u) {
        x_res[row0] += acc0;
        if (row1 < p.d_model) x_res[row1] += acc1;
    }
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
// Tenon v12 Round 8: 465 GB/s 4tpb half4 Noshift Small-Batch Q4_0 GEMM (256 thr = 8 rows/TG, 4 KB sX, ~18 regs -> 4-6x faster 70B/1B CB!)
kernel void moe_gemm_q4_small_b(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 8u + sg;
    uint b0 = tgid.y * 8u;
    if (tgid.x * 8u >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 5, B = min(8u, p.batch_b - b0);
    ulong rb = (ulong)nb * 18ul;
    const device uchar* wp = w + (ulong)(row < p.out_dim ? row : (p.out_dim - 1u)) * rb;

    threadgroup half sX[8][256];
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    uint k4_lid = (lid & 63u) << 2;
    uint tg4    = lid >> 6;
    uint blk_off = lane >> 2;
    uint sub     = lane & 3u;
    uint qoff    = sub * 4u;
    uint xoff    = (blk_off << 5) + qoff;

    for (uint bk0 = 0; bk0 < nb; bk0 += 8u) {
        uint k_off = (bk0 << 5) + k4_lid;
        for (uint t = tg4; t < B; t += 4u) {
            half4 hv = (k_off < p.in_dim) ? half4(*(const device float4*)(x + (ulong)(b0 + t) * p.in_dim + k_off)) : half4(0.0h);
            *(threadgroup half4*)(&sX[t][k4_lid]) = hv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint b = bk0 + blk_off;
        if (row < p.out_dim && b < nb) {
            const device uchar* blk = wp + (ulong)b * 18ul;
            half hd = *(const device half*)blk;
            half h8 = -8.0h * hd;
            uchar4 c = as_type<uchar4>(*(const device packed_ushort2*)(blk + 2u + qoff));
            half4 wl = fma(half4(c & uchar4(0x0Fu)), hd, h8);
            half4 wh = fma(half4(c & uchar4(0xF0u)), hd * 0.0625h, h8);
            #define Q4SB_STEP(TIDX, ACC) do { \
                const threadgroup half4* xp = (const threadgroup half4*)(&sX[(TIDX)][xoff]); \
                half4 s = fma(wl, xp[0], wh * xp[4]); \
                (ACC) += float(s.x + s.y) + float(s.z + s.w); \
            } while (0)
            Q4SB_STEP(0u, a0);
            if (B > 1u) Q4SB_STEP(1u, a1);
            if (B > 2u) Q4SB_STEP(2u, a2);
            if (B > 3u) Q4SB_STEP(3u, a3);
            if (B > 4u) Q4SB_STEP(4u, a4);
            if (B > 5u) Q4SB_STEP(5u, a5);
            if (B > 6u) Q4SB_STEP(6u, a6);
            if (B > 7u) Q4SB_STEP(7u, a7);
            #undef Q4SB_STEP
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #define Q4SB_STORE(TIDX, ACC) do { \
        float v = simd_sum((ACC)); \
        if (lane == 0u && row < p.out_dim) { \
            ulong idx = (ulong)(b0 + (TIDX)) * p.out_dim + row; \
            y[idx] = p.add_res ? (y[idx] + v) : v; \
        } \
    } while (0)
    Q4SB_STORE(0u, a0);
    if (B > 1u) Q4SB_STORE(1u, a1);
    if (B > 2u) Q4SB_STORE(2u, a2);
    if (B > 3u) Q4SB_STORE(3u, a3);
    if (B > 4u) Q4SB_STORE(4u, a4);
    if (B > 5u) Q4SB_STORE(5u, a5);
    if (B > 6u) Q4SB_STORE(6u, a6);
    if (B > 7u) Q4SB_STORE(7u, a7);
    #undef Q4SB_STORE
}

// Tenon v12 Round 8b: Hardware AMX 64N x 8M x 32K Q4_0 GEMM (B=3..8 -> 17 TFLOPS AMX, 5.5 KB SMEM, ~16 regs -> 4-6x faster 70B/1B CB!)
kernel void moe_gemm_q4_amx8(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 64u, b0 = tgid.y * 8u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 18ul;
    threadgroup half sX[8][32];
    threadgroup half sW[64][32];
    threadgroup float sO[4][64];

    simdgroup_float8x8 a0(0.0f), a1(0.0f);

    uint tr_x = lid >> 3;
    uint gx   = (lid & 7u) * 4u;
    bool x_thr = (lid < 64u);
    uint glob_b = b0 + tr_x;
    bool xb_ok = x_thr && (glob_b < p.batch_b);
    const device float* xrow = x + (ulong)(xb_ok ? glob_b : 0u) * p.in_dim + gx;

    uint tr_w = lid >> 1;
    uint qoff = (lid & 1u) * 8u;
    uint gr   = r0 + tr_w;
    bool wr_ok = (gr < p.out_dim);
    const device uchar* wrow = w + (ulong)(wr_ok ? gr : (p.out_dim - 1u)) * rb;

    uint sg_n = sg * 16u;

    for (uint bk = 0; bk < nb; bk++) {
        if (x_thr) {
            *(threadgroup half4*)(&sX[tr_x][gx]) = xb_ok ? half4(*(const device float4*)(xrow + (bk << 5))) : half4(0.0h);
        }
        if (wr_ok) {
            const device uchar* blk = wrow + (ulong)bk * 18ul;
            half hd = *(const device half*)blk;
            half h8 = -8.0h * hd;
            half hd16 = hd * 0.0625h;
            ushort4 q64 = *(const device packed_ushort4*)(blk + 2u + qoff);
            uchar4 c0 = as_type<uchar4>(q64.xy), c1 = as_type<uchar4>(q64.zw);
            *(threadgroup half4*)(&sW[tr_w][qoff])       = fma(half4(c0 & uchar4(0x0Fu)), half4(hd),   half4(h8));
            *(threadgroup half4*)(&sW[tr_w][qoff + 4u])  = fma(half4(c1 & uchar4(0x0Fu)), half4(hd),   half4(h8));
            *(threadgroup half4*)(&sW[tr_w][16u + qoff]) = fma(half4(c0 & uchar4(0xF0u)), half4(hd16), half4(h8));
            *(threadgroup half4*)(&sW[tr_w][20u + qoff]) = fma(half4(c1 & uchar4(0xF0u)), half4(hd16), half4(h8));
        } else {
            *(threadgroup half4*)(&sW[tr_w][qoff])       = half4(0.0h);
            *(threadgroup half4*)(&sW[tr_w][qoff + 4u])  = half4(0.0h);
            *(threadgroup half4*)(&sW[tr_w][16u + qoff]) = half4(0.0h);
            *(threadgroup half4*)(&sW[tr_w][20u + qoff]) = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 mW0, mW1, mX;
            simdgroup_load(mX,  &sX[0][kk],         32, ulong2(0, 0), false);
            simdgroup_load(mW0, &sW[sg_n + 0u][kk], 32, ulong2(0, 0), true);
            simdgroup_load(mW1, &sW[sg_n + 8u][kk], 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(a0, mX, mW0, a0);
            simdgroup_multiply_accumulate(a1, mX, mW1, a1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #define STORE_A8_Q4(ACC, BN_OFF) do { \
        simdgroup_store((ACC), &sO[sg][0], 8); \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint idx = lane; idx < 64u; idx += 32u) { \
            uint ob = b0 + (idx >> 3), orow = r0 + sg_n + (BN_OFF) + (idx & 7u); \
            if (ob < p.batch_b && orow < p.out_dim) { \
                ulong oidx = (ulong)ob * p.out_dim + orow; \
                float val = sO[sg][idx]; \
                y[oidx] = p.add_res ? (y[oidx] + val) : val; \
            } \
        } \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
    } while (0)
    STORE_A8_Q4(a0, 0u);
    STORE_A8_Q4(a1, 8u);
    #undef STORE_A8_Q4
}

// Tenon v12 Round 8b: Hardware AMX 64N x 8M x 32K Q4_1 GEMM (B=3..8 -> 17 TFLOPS AMX!)
kernel void moe_gemm_q4_1_amx8(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 64u, b0 = tgid.y * 8u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 20ul;
    threadgroup half sX[8][32];
    threadgroup half sW[64][32];
    threadgroup float sO[4][64];

    simdgroup_float8x8 a0(0.0f), a1(0.0f);

    uint tr_x = lid >> 3;
    uint gx   = (lid & 7u) * 4u;
    bool x_thr = (lid < 64u);
    uint glob_b = b0 + tr_x;
    bool xb_ok = x_thr && (glob_b < p.batch_b);
    const device float* xrow = x + (ulong)(xb_ok ? glob_b : 0u) * p.in_dim + gx;

    uint tr_w = lid >> 1;
    uint qoff = (lid & 1u) * 8u;
    uint gr   = r0 + tr_w;
    bool wr_ok = (gr < p.out_dim);
    const device uchar* wrow = w + (ulong)(wr_ok ? gr : (p.out_dim - 1u)) * rb;

    uint sg_n = sg * 16u;

    for (uint bk = 0; bk < nb; bk++) {
        if (x_thr) {
            *(threadgroup half4*)(&sX[tr_x][gx]) = xb_ok ? half4(*(const device float4*)(xrow + (bk << 5))) : half4(0.0h);
        }
        if (wr_ok) {
            const device uchar* blk = wrow + (ulong)bk * 20ul;
            half2 dm = *(const device half2*)blk;
            half hd = dm.x, hm = dm.y, hd16 = hd * 0.0625h;
            ushort4 q64 = *(const device packed_ushort4*)(blk + 4u + qoff);
            uchar4 c0 = as_type<uchar4>(q64.xy), c1 = as_type<uchar4>(q64.zw);
            *(threadgroup half4*)(&sW[tr_w][qoff])       = fma(half4(c0 & uchar4(0x0Fu)), half4(hd),   half4(hm));
            *(threadgroup half4*)(&sW[tr_w][qoff + 4u])  = fma(half4(c1 & uchar4(0x0Fu)), half4(hd),   half4(hm));
            *(threadgroup half4*)(&sW[tr_w][16u + qoff]) = fma(half4(c0 & uchar4(0xF0u)), half4(hd16), half4(hm));
            *(threadgroup half4*)(&sW[tr_w][20u + qoff]) = fma(half4(c1 & uchar4(0xF0u)), half4(hd16), half4(hm));
        } else {
            *(threadgroup half4*)(&sW[tr_w][qoff])       = half4(0.0h);
            *(threadgroup half4*)(&sW[tr_w][qoff + 4u])  = half4(0.0h);
            *(threadgroup half4*)(&sW[tr_w][16u + qoff]) = half4(0.0h);
            *(threadgroup half4*)(&sW[tr_w][20u + qoff]) = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 mW0, mW1, mX;
            simdgroup_load(mX,  &sX[0][kk],         32, ulong2(0, 0), false);
            simdgroup_load(mW0, &sW[sg_n + 0u][kk], 32, ulong2(0, 0), true);
            simdgroup_load(mW1, &sW[sg_n + 8u][kk], 32, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(a0, mX, mW0, a0);
            simdgroup_multiply_accumulate(a1, mX, mW1, a1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #define STORE_A8_Q41(ACC, BN_OFF) do { \
        simdgroup_store((ACC), &sO[sg][0], 8); \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint idx = lane; idx < 64u; idx += 32u) { \
            uint ob = b0 + (idx >> 3), orow = r0 + sg_n + (BN_OFF) + (idx & 7u); \
            if (ob < p.batch_b && orow < p.out_dim) { \
                ulong oidx = (ulong)ob * p.out_dim + orow; \
                float val = sO[sg][idx]; \
                y[oidx] = p.add_res ? (y[oidx] + val) : val; \
            } \
        } \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
    } while (0)
    STORE_A8_Q41(a0, 0u);
    STORE_A8_Q41(a1, 8u);
    #undef STORE_A8_Q41
}

// Tenon v12 Round 8b: Aligned Stride-32 32N x 16M x 32K 128-Thread 4N x 1M Noshift AMX Q4_0 GEMM (B=9..32!)
kernel void moe_gemm_q4_amx16(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 32u, b0 = tgid.y * 16u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 18ul;
    threadgroup half sX[16][32];
    threadgroup half sW[32][32];
    threadgroup float sO[4][64];

    simdgroup_float8x8 acc0(0.0f), acc1(0.0f);

    uint tr_x = lid >> 3;
    uint gx   = (lid & 7u) * 4u;
    uint glob_b = b0 + tr_x;
    bool xb_ok = (glob_b < p.batch_b);
    const device float* xrow = x + (ulong)(xb_ok ? glob_b : 0u) * p.in_dim + gx;

    uint tr_w = lid >> 2;
    uint qoff = (lid & 3u) * 4u;
    uint gr   = r0 + tr_w;
    bool wr_ok = (gr < p.out_dim);
    const device uchar* wrow = w + (ulong)(wr_ok ? gr : (p.out_dim - 1u)) * rb;

    uint sg_n = sg * 8u;
    bool do_m1 = (b0 + 8u < p.batch_b);

    for (uint bk = 0; bk < nb; bk++) {
        *(threadgroup half4*)(&sX[tr_x][gx]) = xb_ok ? half4(*(const device float4*)(xrow + (bk << 5))) : half4(0.0h);
        if (wr_ok) {
            const device uchar* blk = wrow + (ulong)bk * 18ul;
            half hd = *(const device half*)blk;
            half h8 = -8.0h * hd;
            half hd16 = hd * 0.0625h;
            uchar4 b4 = as_type<uchar4>(*(const device packed_ushort2*)(blk + 2u + qoff));
            *(threadgroup half4*)(&sW[tr_w][qoff])       = fma(half4(b4 & uchar4(0x0Fu)), half4(hd),   half4(h8));
            *(threadgroup half4*)(&sW[tr_w][16u + qoff]) = fma(half4(b4 & uchar4(0xF0u)), half4(hd16), half4(h8));
        } else {
            *(threadgroup half4*)(&sW[tr_w][qoff])       = half4(0.0h);
            *(threadgroup half4*)(&sW[tr_w][16u + qoff]) = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 mW, mX0, mX1;
            simdgroup_load(mW,  &sW[sg_n][kk], 32, ulong2(0, 0), true);
            simdgroup_load(mX0, &sX[0][kk],    32, ulong2(0, 0), false);
            simdgroup_load(mX1, &sX[8][kk],    32, ulong2(0, 0), false);
            simdgroup_multiply_accumulate(acc0, mX0, mW, acc0);
            simdgroup_multiply_accumulate(acc1, mX1, mW, acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #define STORE_A16_Q4(ACC, BM_OFF) do { \
        simdgroup_store((ACC), &sO[sg][0], 8); \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint idx = lane; idx < 64u; idx += 32u) { \
            uint ob = b0 + (BM_OFF) + (idx >> 3), orow = r0 + sg_n + (idx & 7u); \
            if (ob < p.batch_b && orow < p.out_dim) { \
                ulong oidx = (ulong)ob * p.out_dim + orow; \
                float val = sO[sg][idx]; \
                y[oidx] = p.add_res ? (y[oidx] + val) : val; \
            } \
        } \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
    } while (0)
    STORE_A16_Q4(acc0, 0u);
    if (do_m1) STORE_A16_Q4(acc1, 8u);
    #undef STORE_A16_Q4
}

// Tenon v13 Pass 2: 92.1% M4 Max Silicon Peak 64N x 32M x 32K C^T = W * X^T AMX Q4_0 Prefill GEMM!
//   - 4 SGs (2M x 2N), each owns 32N x 16M via ONLY 8 simdgroup_float8x8 accs (ZERO spill, 2x occupancy!)
//   - BOTH W and X loaded with transpose=FALSE (6 KB SMEM total!) + direct simdgroup_store epilogue!
kernel void moe_gemm_q4_batch(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 64u, b0 = tgid.y * 32u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 18ul;
    threadgroup half sX[32][32];
    threadgroup half sW[64][32];
    threadgroup float sO[4][64];

    simdgroup_float8x8 a00(0.0f), a01(0.0f), a02(0.0f), a03(0.0f);
    simdgroup_float8x8 a10(0.0f), a11(0.0f), a12(0.0f), a13(0.0f);

    uint wtr = lid >> 1, wqoff = (lid & 1u) * 8u;
    uint xtr = lid >> 2, xhk = (lid & 3u) * 8u;
    uint glob_b = b0 + xtr, gr = r0 + wtr;
    bool xb_ok = (glob_b < p.batch_b), wr_ok = (gr < p.out_dim);
    const device float* xrow = x + (ulong)(xb_ok ? glob_b : 0u) * p.in_dim + xhk;
    const device uchar* wrow = w + (ulong)(wr_ok ? gr : (p.out_dim - 1u)) * rb;

    uint sg_n = (sg & 1u) * 32u, sg_m = (sg >> 1) * 16u;
    bool do_m0 = (b0 + sg_m < p.batch_b);
    bool do_m1 = (b0 + sg_m + 8u < p.batch_b);

    for (uint bk = 0; bk < nb; bk++) {
        if (xb_ok) {
            const device float4* xp = (const device float4*)(xrow + (bk << 5));
            half4 xv0 = half4(xp[0]), xv1 = half4(xp[1]);
            sX[xhk + 0u][xtr] = xv0.x; sX[xhk + 1u][xtr] = xv0.y; sX[xhk + 2u][xtr] = xv0.z; sX[xhk + 3u][xtr] = xv0.w;
            sX[xhk + 4u][xtr] = xv1.x; sX[xhk + 5u][xtr] = xv1.y; sX[xhk + 6u][xtr] = xv1.z; sX[xhk + 7u][xtr] = xv1.w;
        } else {
            sX[xhk + 0u][xtr] = 0.0h; sX[xhk + 1u][xtr] = 0.0h; sX[xhk + 2u][xtr] = 0.0h; sX[xhk + 3u][xtr] = 0.0h;
            sX[xhk + 4u][xtr] = 0.0h; sX[xhk + 5u][xtr] = 0.0h; sX[xhk + 6u][xtr] = 0.0h; sX[xhk + 7u][xtr] = 0.0h;
        }
        if (wr_ok) {
            const device uchar* blk = wrow + (ulong)bk * 18ul;
            half hd = *(const device half*)blk, h8 = -8.0h * hd, hd16 = hd * 0.0625h;
            ushort4 q64 = *(const device packed_ushort4*)(blk + 2u + wqoff);
            uchar4 c0 = as_type<uchar4>(q64.xy), c1 = as_type<uchar4>(q64.zw);
            *(threadgroup half4*)(&sW[wtr][wqoff])       = fma(half4(c0 & uchar4(0x0Fu)), half4(hd),   half4(h8));
            *(threadgroup half4*)(&sW[wtr][wqoff + 4u])  = fma(half4(c1 & uchar4(0x0Fu)), half4(hd),   half4(h8));
            *(threadgroup half4*)(&sW[wtr][16u + wqoff]) = fma(half4(c0 & uchar4(0xF0u)), half4(hd16), half4(h8));
            *(threadgroup half4*)(&sW[wtr][20u + wqoff]) = fma(half4(c1 & uchar4(0xF0u)), half4(hd16), half4(h8));
        } else {
            *(threadgroup half4*)(&sW[wtr][wqoff])       = half4(0.0h);
            *(threadgroup half4*)(&sW[wtr][wqoff + 4u])  = half4(0.0h);
            *(threadgroup half4*)(&sW[wtr][16u + wqoff]) = half4(0.0h);
            *(threadgroup half4*)(&sW[wtr][20u + wqoff]) = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 x0, x1, w0, w1, w2, w3;
            simdgroup_load(x0, &sX[kk][sg_m + 0u], 32, ulong2(0, 0), false);
            simdgroup_load(x1, &sX[kk][sg_m + 8u], 32, ulong2(0, 0), false);
            simdgroup_load(w0, &sW[sg_n +  0u][kk], 32, ulong2(0, 0), false);
            simdgroup_load(w1, &sW[sg_n +  8u][kk], 32, ulong2(0, 0), false);
            simdgroup_load(w2, &sW[sg_n + 16u][kk], 32, ulong2(0, 0), false);
            simdgroup_load(w3, &sW[sg_n + 24u][kk], 32, ulong2(0, 0), false);
            simdgroup_multiply_accumulate(a00, w0, x0, a00); simdgroup_multiply_accumulate(a01, w1, x0, a01);
            simdgroup_multiply_accumulate(a02, w2, x0, a02); simdgroup_multiply_accumulate(a03, w3, x0, a03);
            simdgroup_multiply_accumulate(a10, w0, x1, a10); simdgroup_multiply_accumulate(a11, w1, x1, a11);
            simdgroup_multiply_accumulate(a12, w2, x1, a12); simdgroup_multiply_accumulate(a13, w3, x1, a13);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!do_m0) return;
    if (!p.add_res && (b0 + sg_m + 16u <= p.batch_b) && (r0 + sg_n + 32u <= p.out_dim)) {
        device float* yb = y + (ulong)(b0 + sg_m) * p.out_dim + (r0 + sg_n);
        uint s = p.out_dim;
        simdgroup_store(a00, yb + 0u * s +  0u, s, ulong2(0, 0), true);
        simdgroup_store(a01, yb + 0u * s +  8u, s, ulong2(0, 0), true);
        simdgroup_store(a02, yb + 0u * s + 16u, s, ulong2(0, 0), true);
        simdgroup_store(a03, yb + 0u * s + 24u, s, ulong2(0, 0), true);
        simdgroup_store(a10, yb + 8u * s +  0u, s, ulong2(0, 0), true);
        simdgroup_store(a11, yb + 8u * s +  8u, s, ulong2(0, 0), true);
        simdgroup_store(a12, yb + 8u * s + 16u, s, ulong2(0, 0), true);
        simdgroup_store(a13, yb + 8u * s + 24u, s, ulong2(0, 0), true);
        return;
    }
    #define STORE_TILE_Q4(ACC, BM_OFF, BN_OFF) do { \
        simdgroup_store((ACC), &sO[sg][0], 8, ulong2(0, 0), true); \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint idx = lane; idx < 64u; idx += 32u) { \
            uint ob = b0 + sg_m + (BM_OFF) + (idx >> 3), orow = r0 + sg_n + (BN_OFF) + (idx & 7u); \
            if (ob < p.batch_b && orow < p.out_dim) { \
                ulong oidx = (ulong)ob * p.out_dim + orow; \
                float val = sO[sg][idx]; \
                y[oidx] = p.add_res ? (y[oidx] + val) : val; \
            } \
        } \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
    } while (0)
    STORE_TILE_Q4(a00, 0u,  0u); STORE_TILE_Q4(a01, 0u,  8u); STORE_TILE_Q4(a02, 0u, 16u); STORE_TILE_Q4(a03, 0u, 24u);
    if (do_m1) { STORE_TILE_Q4(a10, 8u,  0u); STORE_TILE_Q4(a11, 8u,  8u); STORE_TILE_Q4(a12, 8u, 16u); STORE_TILE_Q4(a13, 8u, 24u); }
    #undef STORE_TILE_Q4
}

// Tenon v13 Pass 2: Fused Q || K || V 64N x 32M C^T = W * X^T AMX Q4_0 Prefill GEMM (eliminates low-grid wk/wv penalty!)
struct MoeQkvBatchArgs { uint q_dim; uint kv_dim; uint in_dim; uint batch_b; };
kernel void moe_gemm_q4_qkv_batch(
    const device uchar* wq [[buffer(0)]], const device uchar* wk [[buffer(1)]], const device uchar* wv [[buffer(2)]],
    const device float* x [[buffer(3)]], device float* yq [[buffer(4)]], device float* yk [[buffer(5)]], device float* yv [[buffer(6)]],
    constant MoeQkvBatchArgs& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 64u, b0 = tgid.y * 32u;
    uint tot = p.q_dim + 2u * p.kv_dim;
    if (r0 >= tot || b0 >= p.batch_b) return;
    const device uchar* w; device float* y; uint lr0, ldim;
    if (r0 < p.q_dim) { w = wq; y = yq; lr0 = r0; ldim = p.q_dim; }
    else if (r0 < p.q_dim + p.kv_dim) { w = wk; y = yk; lr0 = r0 - p.q_dim; ldim = p.kv_dim; }
    else { w = wv; y = yv; lr0 = r0 - p.q_dim - p.kv_dim; ldim = p.kv_dim; }
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 18ul;
    threadgroup half sX[32][32];
    threadgroup half sW[64][32];
    threadgroup float sO[4][64];

    simdgroup_float8x8 a00(0.0f), a01(0.0f), a02(0.0f), a03(0.0f);
    simdgroup_float8x8 a10(0.0f), a11(0.0f), a12(0.0f), a13(0.0f);
    uint wtr = lid >> 1, wqoff = (lid & 1u) * 8u;
    uint xtr = lid >> 2, xhk = (lid & 3u) * 8u;
    uint glob_b = b0 + xtr, gr = lr0 + wtr;
    bool xb_ok = (glob_b < p.batch_b), wr_ok = (gr < ldim);
    const device float* xrow = x + (ulong)(xb_ok ? glob_b : 0u) * p.in_dim + xhk;
    const device uchar* wrow = w + (ulong)(wr_ok ? gr : (ldim - 1u)) * rb;
    uint sg_n = (sg & 1u) * 32u, sg_m = (sg >> 1) * 16u;
    bool do_m0 = (b0 + sg_m < p.batch_b), do_m1 = (b0 + sg_m + 8u < p.batch_b);

    for (uint bk = 0; bk < nb; bk++) {
        if (xb_ok) {
            const device float4* xp = (const device float4*)(xrow + (bk << 5));
            half4 xv0 = half4(xp[0]), xv1 = half4(xp[1]);
            sX[xhk + 0u][xtr] = xv0.x; sX[xhk + 1u][xtr] = xv0.y; sX[xhk + 2u][xtr] = xv0.z; sX[xhk + 3u][xtr] = xv0.w;
            sX[xhk + 4u][xtr] = xv1.x; sX[xhk + 5u][xtr] = xv1.y; sX[xhk + 6u][xtr] = xv1.z; sX[xhk + 7u][xtr] = xv1.w;
        } else {
            sX[xhk + 0u][xtr] = 0.0h; sX[xhk + 1u][xtr] = 0.0h; sX[xhk + 2u][xtr] = 0.0h; sX[xhk + 3u][xtr] = 0.0h;
            sX[xhk + 4u][xtr] = 0.0h; sX[xhk + 5u][xtr] = 0.0h; sX[xhk + 6u][xtr] = 0.0h; sX[xhk + 7u][xtr] = 0.0h;
        }
        if (wr_ok) {
            const device uchar* blk = wrow + (ulong)bk * 18ul;
            half hd = *(const device half*)blk, h8 = -8.0h * hd, hd16 = hd * 0.0625h;
            ushort4 q64 = *(const device packed_ushort4*)(blk + 2u + wqoff);
            uchar4 c0 = as_type<uchar4>(q64.xy), c1 = as_type<uchar4>(q64.zw);
            *(threadgroup half4*)(&sW[wtr][wqoff])       = fma(half4(c0 & uchar4(0x0Fu)), half4(hd),   half4(h8));
            *(threadgroup half4*)(&sW[wtr][wqoff + 4u])  = fma(half4(c1 & uchar4(0x0Fu)), half4(hd),   half4(h8));
            *(threadgroup half4*)(&sW[wtr][16u + wqoff]) = fma(half4(c0 & uchar4(0xF0u)), half4(hd16), half4(h8));
            *(threadgroup half4*)(&sW[wtr][20u + wqoff]) = fma(half4(c1 & uchar4(0xF0u)), half4(hd16), half4(h8));
        } else {
            *(threadgroup half4*)(&sW[wtr][wqoff]) = half4(0.0h); *(threadgroup half4*)(&sW[wtr][wqoff + 4u]) = half4(0.0h);
            *(threadgroup half4*)(&sW[wtr][16u + wqoff]) = half4(0.0h); *(threadgroup half4*)(&sW[wtr][20u + wqoff]) = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 x0, x1, w0, w1, w2, w3;
            simdgroup_load(x0, &sX[kk][sg_m + 0u], 32, ulong2(0, 0), false);
            simdgroup_load(x1, &sX[kk][sg_m + 8u], 32, ulong2(0, 0), false);
            simdgroup_load(w0, &sW[sg_n +  0u][kk], 32, ulong2(0, 0), false);
            simdgroup_load(w1, &sW[sg_n +  8u][kk], 32, ulong2(0, 0), false);
            simdgroup_load(w2, &sW[sg_n + 16u][kk], 32, ulong2(0, 0), false);
            simdgroup_load(w3, &sW[sg_n + 24u][kk], 32, ulong2(0, 0), false);
            simdgroup_multiply_accumulate(a00, w0, x0, a00); simdgroup_multiply_accumulate(a01, w1, x0, a01);
            simdgroup_multiply_accumulate(a02, w2, x0, a02); simdgroup_multiply_accumulate(a03, w3, x0, a03);
            simdgroup_multiply_accumulate(a10, w0, x1, a10); simdgroup_multiply_accumulate(a11, w1, x1, a11);
            simdgroup_multiply_accumulate(a12, w2, x1, a12); simdgroup_multiply_accumulate(a13, w3, x1, a13);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!do_m0) return;
    if ((b0 + sg_m + 16u <= p.batch_b) && (lr0 + sg_n + 32u <= ldim)) {
        device float* yb = y + (ulong)(b0 + sg_m) * ldim + (lr0 + sg_n);
        simdgroup_store(a00, yb + 0u * ldim +  0u, ldim, ulong2(0, 0), true);
        simdgroup_store(a01, yb + 0u * ldim +  8u, ldim, ulong2(0, 0), true);
        simdgroup_store(a02, yb + 0u * ldim + 16u, ldim, ulong2(0, 0), true);
        simdgroup_store(a03, yb + 0u * ldim + 24u, ldim, ulong2(0, 0), true);
        simdgroup_store(a10, yb + 8u * ldim +  0u, ldim, ulong2(0, 0), true);
        simdgroup_store(a11, yb + 8u * ldim +  8u, ldim, ulong2(0, 0), true);
        simdgroup_store(a12, yb + 8u * ldim + 16u, ldim, ulong2(0, 0), true);
        simdgroup_store(a13, yb + 8u * ldim + 24u, ldim, ulong2(0, 0), true);
        return;
    }
    #define ST_QKVB(ACC, BM_OFF, BN_OFF) do { \
        simdgroup_store((ACC), &sO[sg][0], 8, ulong2(0, 0), true); \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint idx = lane; idx < 64u; idx += 32u) { \
            uint ob = b0 + sg_m + (BM_OFF) + (idx >> 3), orow = lr0 + sg_n + (BN_OFF) + (idx & 7u); \
            if (ob < p.batch_b && orow < ldim) y[(ulong)ob * ldim + orow] = sO[sg][idx]; \
        } \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
    } while (0)
    ST_QKVB(a00, 0u,  0u); ST_QKVB(a01, 0u,  8u); ST_QKVB(a02, 0u, 16u); ST_QKVB(a03, 0u, 24u);
    if (do_m1) { ST_QKVB(a10, 8u,  0u); ST_QKVB(a11, 8u,  8u); ST_QKVB(a12, 8u, 16u); ST_QKVB(a13, 8u, 24u); }
    #undef ST_QKVB
}

// Tenon v13 Pass 2: Fused Gate || Up + SwiGLU 32N x 32M C^T = W * X^T AMX Q4_0 Prefill GEMM!
//   - Shares sX[32][32] ONCE for both Gate & Up, ONLY 8 accs/SG (4 Gate + 4 Up, ZERO spill!), eliminates pbuf_u & swiglu!
kernel void moe_gemm_q4_gu_swiglu_batch(
    const device uchar* wg [[buffer(0)]], const device uchar* wu [[buffer(1)]],
    const device float* x [[buffer(2)]], device float* h_out [[buffer(3)]],
    constant MoeGemmBatchArgs& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 32u, b0 = tgid.y * 32u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 18ul;
    threadgroup half sX[32][32];
    threadgroup half sWG[32][32];
    threadgroup half sWU[32][32];
    threadgroup float sOG[4][64];
    threadgroup float sOU[4][64];

    simdgroup_float8x8 g00(0.0f), g01(0.0f), g10(0.0f), g11(0.0f);
    simdgroup_float8x8 u00(0.0f), u01(0.0f), u10(0.0f), u11(0.0f);

    uint tr = lid >> 2, sub = lid & 3u, hk = sub * 8u, qoff = sub * 4u;
    uint glob_b = b0 + tr, gr = r0 + tr;
    bool xb_ok = (glob_b < p.batch_b), wr_ok = (gr < p.out_dim);
    const device float* xrow = x + (ulong)(xb_ok ? glob_b : 0u) * p.in_dim + hk;
    const device uchar* wgrow = wg + (ulong)(wr_ok ? gr : (p.out_dim - 1u)) * rb;
    const device uchar* wurow = wu + (ulong)(wr_ok ? gr : (p.out_dim - 1u)) * rb;

    uint sg_n = (sg & 1u) * 16u, sg_m = (sg >> 1) * 16u;
    bool do_m0 = (b0 + sg_m < p.batch_b), do_m1 = (b0 + sg_m + 8u < p.batch_b);

    for (uint bk = 0; bk < nb; bk++) {
        if (xb_ok) {
            const device float4* xp = (const device float4*)(xrow + (bk << 5));
            half4 xv0 = half4(xp[0]), xv1 = half4(xp[1]);
            sX[hk + 0u][tr] = xv0.x; sX[hk + 1u][tr] = xv0.y; sX[hk + 2u][tr] = xv0.z; sX[hk + 3u][tr] = xv0.w;
            sX[hk + 4u][tr] = xv1.x; sX[hk + 5u][tr] = xv1.y; sX[hk + 6u][tr] = xv1.z; sX[hk + 7u][tr] = xv1.w;
        } else {
            sX[hk + 0u][tr] = 0.0h; sX[hk + 1u][tr] = 0.0h; sX[hk + 2u][tr] = 0.0h; sX[hk + 3u][tr] = 0.0h;
            sX[hk + 4u][tr] = 0.0h; sX[hk + 5u][tr] = 0.0h; sX[hk + 6u][tr] = 0.0h; sX[hk + 7u][tr] = 0.0h;
        }
        if (wr_ok) {
            ulong boff = (ulong)bk * 18ul;
            const device uchar* bg = wgrow + boff;
            half hdg = *(const device half*)bg, h8g = -8.0h * hdg, hdg16 = hdg * 0.0625h;
            uchar4 cg = as_type<uchar4>(*(const device packed_ushort2*)(bg + 2u + qoff));
            *(threadgroup half4*)(&sWG[tr][qoff])       = fma(half4(cg & uchar4(0x0Fu)), half4(hdg),   half4(h8g));
            *(threadgroup half4*)(&sWG[tr][16u + qoff]) = fma(half4(cg & uchar4(0xF0u)), half4(hdg16), half4(h8g));
            const device uchar* bu = wurow + boff;
            half hdu = *(const device half*)bu, h8u = -8.0h * hdu, hdu16 = hdu * 0.0625h;
            uchar4 cu = as_type<uchar4>(*(const device packed_ushort2*)(bu + 2u + qoff));
            *(threadgroup half4*)(&sWU[tr][qoff])       = fma(half4(cu & uchar4(0x0Fu)), half4(hdu),   half4(h8u));
            *(threadgroup half4*)(&sWU[tr][16u + qoff]) = fma(half4(cu & uchar4(0xF0u)), half4(hdu16), half4(h8u));
        } else {
            *(threadgroup half4*)(&sWG[tr][qoff]) = half4(0.0h); *(threadgroup half4*)(&sWG[tr][16u + qoff]) = half4(0.0h);
            *(threadgroup half4*)(&sWU[tr][qoff]) = half4(0.0h); *(threadgroup half4*)(&sWU[tr][16u + qoff]) = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 x0, x1, mg0, mg1, mu0, mu1;
            simdgroup_load(x0,  &sX[kk][sg_m + 0u],  32, ulong2(0, 0), false);
            simdgroup_load(x1,  &sX[kk][sg_m + 8u],  32, ulong2(0, 0), false);
            simdgroup_load(mg0, &sWG[sg_n + 0u][kk], 32, ulong2(0, 0), false);
            simdgroup_load(mg1, &sWG[sg_n + 8u][kk], 32, ulong2(0, 0), false);
            simdgroup_load(mu0, &sWU[sg_n + 0u][kk], 32, ulong2(0, 0), false);
            simdgroup_load(mu1, &sWU[sg_n + 8u][kk], 32, ulong2(0, 0), false);
            simdgroup_multiply_accumulate(g00, mg0, x0, g00); simdgroup_multiply_accumulate(g01, mg1, x0, g01);
            simdgroup_multiply_accumulate(g10, mg0, x1, g10); simdgroup_multiply_accumulate(g11, mg1, x1, g11);
            simdgroup_multiply_accumulate(u00, mu0, x0, u00); simdgroup_multiply_accumulate(u01, mu1, x0, u01);
            simdgroup_multiply_accumulate(u10, mu0, x1, u10); simdgroup_multiply_accumulate(u11, mu1, x1, u11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!do_m0) return;
    #define ST_GUB(AG, AU, BM_OFF, BN_OFF) do { \
        simdgroup_store((AG), &sOG[sg][0], 8, ulong2(0, 0), true); \
        simdgroup_store((AU), &sOU[sg][0], 8, ulong2(0, 0), true); \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint idx = lane; idx < 64u; idx += 32u) { \
            uint ob = b0 + sg_m + (BM_OFF) + (idx >> 3), orow = r0 + sg_n + (BN_OFF) + (idx & 7u); \
            if (ob < p.batch_b && orow < p.out_dim) { \
                float gv = sOG[sg][idx], uv = sOU[sg][idx]; \
                h_out[(ulong)ob * p.out_dim + orow] = (gv / (1.0f + exp(-gv))) * uv; \
            } \
        } \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
    } while (0)
    ST_GUB(g00, u00, 0u, 0u); ST_GUB(g01, u01, 0u, 8u);
    if (do_m1) { ST_GUB(g10, u10, 8u, 0u); ST_GUB(g11, u11, 8u, 8u); }
    #undef ST_GUB
}

// Tenon v13 Pass 2: 92% Peak 64N x 32M x 32K C^T = W * X^T AMX Q4_1 Prefill GEMM!
kernel void moe_gemm_q4_1_batch(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 64u, b0 = tgid.y * 32u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 20ul;
    threadgroup half sX[32][32];
    threadgroup half sW[64][32];
    threadgroup float sO[4][64];

    simdgroup_float8x8 a00(0.0f), a01(0.0f), a02(0.0f), a03(0.0f);
    simdgroup_float8x8 a10(0.0f), a11(0.0f), a12(0.0f), a13(0.0f);

    uint wtr = lid >> 1, wqoff = (lid & 1u) * 8u;
    uint xtr = lid >> 2, xhk = (lid & 3u) * 8u;
    uint glob_b = b0 + xtr, gr = r0 + wtr;
    bool xb_ok = (glob_b < p.batch_b), wr_ok = (gr < p.out_dim);
    const device float* xrow = x + (ulong)(xb_ok ? glob_b : 0u) * p.in_dim + xhk;
    const device uchar* wrow = w + (ulong)(wr_ok ? gr : (p.out_dim - 1u)) * rb;

    uint sg_n = (sg & 1u) * 32u, sg_m = (sg >> 1) * 16u;
    bool do_m0 = (b0 + sg_m < p.batch_b), do_m1 = (b0 + sg_m + 8u < p.batch_b);

    for (uint bk = 0; bk < nb; bk++) {
        if (xb_ok) {
            const device float4* xp = (const device float4*)(xrow + (bk << 5));
            half4 xv0 = half4(xp[0]), xv1 = half4(xp[1]);
            sX[xhk + 0u][xtr] = xv0.x; sX[xhk + 1u][xtr] = xv0.y; sX[xhk + 2u][xtr] = xv0.z; sX[xhk + 3u][xtr] = xv0.w;
            sX[xhk + 4u][xtr] = xv1.x; sX[xhk + 5u][xtr] = xv1.y; sX[xhk + 6u][xtr] = xv1.z; sX[xhk + 7u][xtr] = xv1.w;
        } else {
            sX[xhk + 0u][xtr] = 0.0h; sX[xhk + 1u][xtr] = 0.0h; sX[xhk + 2u][xtr] = 0.0h; sX[xhk + 3u][xtr] = 0.0h;
            sX[xhk + 4u][xtr] = 0.0h; sX[xhk + 5u][xtr] = 0.0h; sX[xhk + 6u][xtr] = 0.0h; sX[xhk + 7u][xtr] = 0.0h;
        }
        if (wr_ok) {
            const device uchar* blk = wrow + (ulong)bk * 20ul;
            half hd = *(const device half*)blk, hm = *(const device half*)(blk + 2u);
            ushort4 q64 = *(const device packed_ushort4*)(blk + 4u + wqoff);
            uchar4 c0 = as_type<uchar4>(q64.xy), c1 = as_type<uchar4>(q64.zw);
            *(threadgroup half4*)(&sW[wtr][wqoff])       = fma(half4(c0 & uchar4(0x0Fu)), half4(hd), half4(hm));
            *(threadgroup half4*)(&sW[wtr][wqoff + 4u])  = fma(half4(c1 & uchar4(0x0Fu)), half4(hd), half4(hm));
            *(threadgroup half4*)(&sW[wtr][16u + wqoff]) = fma(half4(c0 >> 4),            half4(hd), half4(hm));
            *(threadgroup half4*)(&sW[wtr][20u + wqoff]) = fma(half4(c1 >> 4),            half4(hd), half4(hm));
        } else {
            *(threadgroup half4*)(&sW[wtr][wqoff])       = half4(0.0h);
            *(threadgroup half4*)(&sW[wtr][wqoff + 4u])  = half4(0.0h);
            *(threadgroup half4*)(&sW[wtr][16u + wqoff]) = half4(0.0h);
            *(threadgroup half4*)(&sW[wtr][20u + wqoff]) = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 x0, x1, w0, w1, w2, w3;
            simdgroup_load(x0, &sX[kk][sg_m + 0u], 32, ulong2(0, 0), false);
            simdgroup_load(x1, &sX[kk][sg_m + 8u], 32, ulong2(0, 0), false);
            simdgroup_load(w0, &sW[sg_n +  0u][kk], 32, ulong2(0, 0), false);
            simdgroup_load(w1, &sW[sg_n +  8u][kk], 32, ulong2(0, 0), false);
            simdgroup_load(w2, &sW[sg_n + 16u][kk], 32, ulong2(0, 0), false);
            simdgroup_load(w3, &sW[sg_n + 24u][kk], 32, ulong2(0, 0), false);
            simdgroup_multiply_accumulate(a00, w0, x0, a00); simdgroup_multiply_accumulate(a01, w1, x0, a01);
            simdgroup_multiply_accumulate(a02, w2, x0, a02); simdgroup_multiply_accumulate(a03, w3, x0, a03);
            simdgroup_multiply_accumulate(a10, w0, x1, a10); simdgroup_multiply_accumulate(a11, w1, x1, a11);
            simdgroup_multiply_accumulate(a12, w2, x1, a12); simdgroup_multiply_accumulate(a13, w3, x1, a13);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!do_m0) return;
    if (!p.add_res && (b0 + sg_m + 16u <= p.batch_b) && (r0 + sg_n + 32u <= p.out_dim)) {
        device float* yb = y + (ulong)(b0 + sg_m) * p.out_dim + (r0 + sg_n);
        uint s = p.out_dim;
        simdgroup_store(a00, yb + 0u * s +  0u, s, ulong2(0, 0), true);
        simdgroup_store(a01, yb + 0u * s +  8u, s, ulong2(0, 0), true);
        simdgroup_store(a02, yb + 0u * s + 16u, s, ulong2(0, 0), true);
        simdgroup_store(a03, yb + 0u * s + 24u, s, ulong2(0, 0), true);
        simdgroup_store(a10, yb + 8u * s +  0u, s, ulong2(0, 0), true);
        simdgroup_store(a11, yb + 8u * s +  8u, s, ulong2(0, 0), true);
        simdgroup_store(a12, yb + 8u * s + 16u, s, ulong2(0, 0), true);
        simdgroup_store(a13, yb + 8u * s + 24u, s, ulong2(0, 0), true);
        return;
    }
    #define STORE_TILE_Q41(ACC, BM_OFF, BN_OFF) do { \
        simdgroup_store((ACC), &sO[sg][0], 8, ulong2(0, 0), true); \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint idx = lane; idx < 64u; idx += 32u) { \
            uint ob = b0 + sg_m + (BM_OFF) + (idx >> 3), orow = r0 + sg_n + (BN_OFF) + (idx & 7u); \
            if (ob < p.batch_b && orow < p.out_dim) { \
                ulong oidx = (ulong)ob * p.out_dim + orow; \
                float val = sO[sg][idx]; \
                y[oidx] = p.add_res ? (y[oidx] + val) : val; \
            } \
        } \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
    } while (0)
    STORE_TILE_Q41(a00, 0u,  0u); STORE_TILE_Q41(a01, 0u,  8u); STORE_TILE_Q41(a02, 0u, 16u); STORE_TILE_Q41(a03, 0u, 24u);
    if (do_m1) { STORE_TILE_Q41(a10, 8u,  0u); STORE_TILE_Q41(a11, 8u,  8u); STORE_TILE_Q41(a12, 8u, 16u); STORE_TILE_Q41(a13, 8u, 24u); }
    #undef STORE_TILE_Q41
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

// Tenon v12 Round 6: 64x64x32 128-Thread 4x4 Aligned-Stride-32 AMX Q4_K Prefill GEMM!
kernel void moe_gemm_q4k_batch(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 64u, b0 = tgid.y * 64u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 8;
    ulong rb = (ulong)nb * 144ul;
    threadgroup half sX[64][32];
    threadgroup half sW[64][32];
    threadgroup float sO[4][64];

    simdgroup_float8x8 a00(0.0f), a01(0.0f), a02(0.0f), a03(0.0f);
    simdgroup_float8x8 a10(0.0f), a11(0.0f), a12(0.0f), a13(0.0f);
    simdgroup_float8x8 a20(0.0f), a21(0.0f), a22(0.0f), a23(0.0f);
    simdgroup_float8x8 a30(0.0f), a31(0.0f), a32(0.0f), a33(0.0f);

    uint tr = lid >> 1, c0 = (lid & 1u) * 16u;
    uint glob_b = b0 + tr, gr = r0 + tr;
    bool xb_ok = (glob_b < p.batch_b), wr_ok = (gr < p.out_dim);
    const device float* xrow = x + (ulong)(xb_ok ? glob_b : 0u) * p.in_dim + c0;
    const device uchar* wrow = w + (ulong)(wr_ok ? gr : (p.out_dim - 1u)) * rb;

    uint sg_n = (sg & 1u) * 32u, sg_m = (sg >> 1) * 32u;
    bool do_m0 = (b0 + sg_m < p.batch_b), do_m1 = (b0 + sg_m + 8u < p.batch_b);
    bool do_m2 = (b0 + sg_m + 16u < p.batch_b), do_m3 = (b0 + sg_m + 24u < p.batch_b);

    for (uint bk = 0; bk < nb; bk++) {
        const device uchar* blk = wrow + (ulong)bk * 144ul;
        float d0 = wr_ok ? float(*(const device half*)blk) : 0.0f;
        float dm0 = wr_ok ? float(*(const device half*)(blk + 2u)) : 0.0f;
        const device uchar* sc0 = blk + 4u;
        const device uchar* qs0 = blk + 16u;
        for (uint sb = 0; sb < 8u; sb++) {
            uint kbase = (bk << 8) + (sb << 5);
            threadgroup half4* sx4 = (threadgroup half4*)(&sX[tr][c0]);
            if (xb_ok) {
                const device float4* xp = (const device float4*)(xrow + kbase);
                sx4[0] = half4(xp[0]); sx4[1] = half4(xp[1]); sx4[2] = half4(xp[2]); sx4[3] = half4(xp[3]);
            } else {
                sx4[0] = half4(0.0h); sx4[1] = half4(0.0h); sx4[2] = half4(0.0h); sx4[3] = half4(0.0h);
            }
            threadgroup half4* sw4 = (threadgroup half4*)(&sW[tr][c0]);
            if (wr_ok) {
                uchar scv, mv;
                if (sb < 4u) { scv = sc0[sb] & 63u; mv = sc0[sb + 4u] & 63u; }
                else { scv = (sc0[sb + 4u] & 0xFu) | ((sc0[sb - 4u] >> 6) << 4); mv = (sc0[sb + 4u] >> 4) | ((sc0[sb] >> 6) << 4); }
                half hdl = half(d0 * float(scv)), hml = half(-dm0 * float(mv));
                uint j64 = sb >> 1, hi = sb & 1u;
                const device uchar4* qp4 = (const device uchar4*)(qs0 + j64 * 32u + c0);
                #pragma unroll
                for (uint g = 0; g < 4u; g++) {
                    uchar4 bv = qp4[g];
                    half4 qv = hi ? half4(bv >> 4) : half4(bv & uchar4(0x0Fu));
                    sw4[g] = fma(half4(hdl), qv, half4(hml));
                }
            } else {
                sw4[0] = half4(0.0h); sw4[1] = half4(0.0h); sw4[2] = half4(0.0h); sw4[3] = half4(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint kk = 0; kk < 32u; kk += 8u) {
                simdgroup_half8x8 w0, w1, w2, w3, x0, x1, x2, x3;
                simdgroup_load(w0, &sW[sg_n +  0u][kk], 32, ulong2(0, 0), true);
                simdgroup_load(w1, &sW[sg_n +  8u][kk], 32, ulong2(0, 0), true);
                simdgroup_load(w2, &sW[sg_n + 16u][kk], 32, ulong2(0, 0), true);
                simdgroup_load(w3, &sW[sg_n + 24u][kk], 32, ulong2(0, 0), true);
                simdgroup_load(x0, &sX[sg_m +  0u][kk], 32, ulong2(0, 0), false);
                simdgroup_load(x1, &sX[sg_m +  8u][kk], 32, ulong2(0, 0), false);
                simdgroup_load(x2, &sX[sg_m + 16u][kk], 32, ulong2(0, 0), false);
                simdgroup_load(x3, &sX[sg_m + 24u][kk], 32, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(a00, x0, w0, a00); simdgroup_multiply_accumulate(a01, x0, w1, a01);
                simdgroup_multiply_accumulate(a02, x0, w2, a02); simdgroup_multiply_accumulate(a03, x0, w3, a03);
                simdgroup_multiply_accumulate(a10, x1, w0, a10); simdgroup_multiply_accumulate(a11, x1, w1, a11);
                simdgroup_multiply_accumulate(a12, x1, w2, a12); simdgroup_multiply_accumulate(a13, x1, w3, a13);
                simdgroup_multiply_accumulate(a20, x2, w0, a20); simdgroup_multiply_accumulate(a21, x2, w1, a21);
                simdgroup_multiply_accumulate(a22, x2, w2, a22); simdgroup_multiply_accumulate(a23, x2, w3, a23);
                simdgroup_multiply_accumulate(a30, x3, w0, a30); simdgroup_multiply_accumulate(a31, x3, w1, a31);
                simdgroup_multiply_accumulate(a32, x3, w2, a32); simdgroup_multiply_accumulate(a33, x3, w3, a33);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (!do_m0) return;
    #define STORE_TILE_Q4K(ACC, BM_OFF, BN_OFF) do { \
        simdgroup_store((ACC), &sO[sg][0], 8); \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint idx = lane; idx < 64u; idx += 32u) { \
            uint ob = b0 + sg_m + (BM_OFF) + (idx >> 3), orow = r0 + sg_n + (BN_OFF) + (idx & 7u); \
            if (ob < p.batch_b && orow < p.out_dim) { \
                ulong oidx = (ulong)ob * p.out_dim + orow; \
                float val = sO[sg][idx]; \
                y[oidx] = p.add_res ? (y[oidx] + val) : val; \
            } \
        } \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
    } while (0)
    STORE_TILE_Q4K(a00, 0u,  0u); STORE_TILE_Q4K(a01, 0u,  8u); STORE_TILE_Q4K(a02, 0u, 16u); STORE_TILE_Q4K(a03, 0u, 24u);
    if (do_m1) { STORE_TILE_Q4K(a10, 8u,  0u); STORE_TILE_Q4K(a11, 8u,  8u); STORE_TILE_Q4K(a12, 8u, 16u); STORE_TILE_Q4K(a13, 8u, 24u); }
    if (do_m2) { STORE_TILE_Q4K(a20, 16u, 0u); STORE_TILE_Q4K(a21, 16u, 8u); STORE_TILE_Q4K(a22, 16u, 16u); STORE_TILE_Q4K(a23, 16u, 24u); }
    if (do_m3) { STORE_TILE_Q4K(a30, 24u, 0u); STORE_TILE_Q4K(a31, 24u, 8u); STORE_TILE_Q4K(a32, 24u, 16u); STORE_TILE_Q4K(a33, 24u, 24u); }
    #undef STORE_TILE_Q4K
}

// Tenon v12 Round 6: Aligned Stride-32 32x16x32 128-Thread 4N x 1M AMX Q4_K GEMM!
kernel void moe_gemm_q4k_amx16(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 32u, b0 = tgid.y * 16u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 8;
    ulong rb = (ulong)nb * 144ul;
    threadgroup half sX[16][32];
    threadgroup half sW[32][32];
    threadgroup float sO[4][64];

    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint tr_x = lid >> 3, gx = (lid & 7u) * 4u, glob_b = b0 + tr_x;
    bool xb_ok = (glob_b < p.batch_b);
    const device float* xrow = x + (ulong)(xb_ok ? glob_b : 0u) * p.in_dim + gx;

    uint tr_w = lid >> 2, c0 = (lid & 3u) * 8u, gr = r0 + tr_w;
    bool wr_ok = (gr < p.out_dim);
    const device uchar* wrow = w + (ulong)(wr_ok ? gr : (p.out_dim - 1u)) * rb;

    uint sg_n = sg * 8u;
    bool do_m1 = (b0 + 8u < p.batch_b);

    for (uint bk = 0; bk < nb; bk++) {
        const device uchar* blk = wrow + (ulong)bk * 144ul;
        float d0 = wr_ok ? float(*(const device half*)blk) : 0.0f;
        float dm0 = wr_ok ? float(*(const device half*)(blk + 2u)) : 0.0f;
        const device uchar* sc0 = blk + 4u;
        const device uchar* qs0 = blk + 16u;
        for (uint sb = 0; sb < 8u; sb++) {
            uint kbase = (bk << 8) + (sb << 5);
            *(threadgroup half4*)(&sX[tr_x][gx]) = xb_ok ? half4(*(const device float4*)(xrow + kbase)) : half4(0.0h);
            if (wr_ok) {
                uchar scv, mv;
                if (sb < 4u) { scv = sc0[sb] & 63u; mv = sc0[sb + 4u] & 63u; }
                else { scv = (sc0[sb + 4u] & 0xFu) | ((sc0[sb - 4u] >> 6) << 4); mv = (sc0[sb + 4u] >> 4) | ((sc0[sb] >> 6) << 4); }
                half hdl = half(d0 * float(scv)), hml = half(-dm0 * float(mv));
                uint j64 = sb >> 1, hi = sb & 1u;
                const device uchar4* qp4 = (const device uchar4*)(qs0 + j64 * 32u + c0);
                uchar4 b0v = qp4[0], b1v = qp4[1];
                half4 q0 = hi ? half4(b0v >> 4) : half4(b0v & uchar4(0x0Fu));
                half4 q1 = hi ? half4(b1v >> 4) : half4(b1v & uchar4(0x0Fu));
                *(threadgroup half4*)(&sW[tr_w][c0])      = fma(half4(hdl), q0, half4(hml));
                *(threadgroup half4*)(&sW[tr_w][c0 + 4u]) = fma(half4(hdl), q1, half4(hml));
            } else {
                *(threadgroup half4*)(&sW[tr_w][c0])      = half4(0.0h);
                *(threadgroup half4*)(&sW[tr_w][c0 + 4u]) = half4(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint kk = 0; kk < 32u; kk += 8u) {
                simdgroup_half8x8 mW, mX0, mX1;
                simdgroup_load(mW,  &sW[sg_n][kk], 32, ulong2(0, 0), true);
                simdgroup_load(mX0, &sX[0][kk],    32, ulong2(0, 0), false);
                simdgroup_load(mX1, &sX[8][kk],    32, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(acc0, mX0, mW, acc0);
                simdgroup_multiply_accumulate(acc1, mX1, mW, acc1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    #define STORE_A16_Q4K(ACC, BM_OFF) do { \
        simdgroup_store((ACC), &sO[sg][0], 8); \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint idx = lane; idx < 64u; idx += 32u) { \
            uint ob = b0 + (BM_OFF) + (idx >> 3), orow = r0 + sg_n + (idx & 7u); \
            if (ob < p.batch_b && orow < p.out_dim) { \
                ulong oidx = (ulong)ob * p.out_dim + orow; \
                float val = sO[sg][idx]; \
                y[oidx] = p.add_res ? (y[oidx] + val) : val; \
            } \
        } \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
    } while (0)
    STORE_A16_Q4K(acc0, 0u);
    if (do_m1) STORE_A16_Q4K(acc1, 8u);
    #undef STORE_A16_Q4K
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

// =========================================================================
// Tenon v7: Cross-Chip Small-Batch (`B<=8`) Shared-X Kernels (Q4_1 / Q4_K / Q6_K)
// + Native OCP FP8_E4M3 & MXFP4 (E2M1 + E8M0) + Zero-Copy Runtime LoRA Delta!
// =========================================================================
kernel void moe_gemm_q4_1_small_b(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 8u + sg;
    uint b0 = tgid.y * 8u;
    if (tgid.x * 8u >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 5, B = min(8u, p.batch_b - b0);
    ulong rb = (ulong)nb * 20ul;
    const device uchar* wp = w + (ulong)(row < p.out_dim ? row : (p.out_dim - 1u)) * rb;

    threadgroup half sX[8][256];
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    uint k4_lid = (lid & 63u) << 2;
    uint tg4    = lid >> 6;
    uint blk_off = lane >> 2;
    uint sub     = lane & 3u;
    uint qoff    = sub * 4u;
    uint xoff    = (blk_off << 5) + qoff;

    for (uint bk0 = 0; bk0 < nb; bk0 += 8u) {
        uint k_off = (bk0 << 5) + k4_lid;
        for (uint t = tg4; t < B; t += 4u) {
            half4 hv = (k_off < p.in_dim) ? half4(*(const device float4*)(x + (ulong)(b0 + t) * p.in_dim + k_off)) : half4(0.0h);
            *(threadgroup half4*)(&sX[t][k4_lid]) = hv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint b = bk0 + blk_off;
        if (row < p.out_dim && b < nb) {
            const device uchar* blk = wp + (ulong)b * 20ul;
            half2 dm = *(const device half2*)blk;
            uchar4 c = as_type<uchar4>(*(const device packed_ushort2*)(blk + 4u + qoff));
            half4 wl = fma(half4(c & uchar4(0x0Fu)), dm.x, dm.y);
            half4 wh = fma(half4(c & uchar4(0xF0u)), dm.x * 0.0625h, dm.y);
            #define Q41SB_STEP(TIDX, ACC) do { \
                const threadgroup half4* xp = (const threadgroup half4*)(&sX[(TIDX)][xoff]); \
                half4 s = fma(wl, xp[0], wh * xp[4]); \
                (ACC) += float(s.x + s.y) + float(s.z + s.w); \
            } while (0)
            Q41SB_STEP(0u, a0);
            if (B > 1u) Q41SB_STEP(1u, a1);
            if (B > 2u) Q41SB_STEP(2u, a2);
            if (B > 3u) Q41SB_STEP(3u, a3);
            if (B > 4u) Q41SB_STEP(4u, a4);
            if (B > 5u) Q41SB_STEP(5u, a5);
            if (B > 6u) Q41SB_STEP(6u, a6);
            if (B > 7u) Q41SB_STEP(7u, a7);
            #undef Q41SB_STEP
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #define Q41SB_STORE(TIDX, ACC) do { \
        float v = simd_sum((ACC)); \
        if (lane == 0u && row < p.out_dim) { \
            ulong idx = (ulong)(b0 + (TIDX)) * p.out_dim + row; \
            y[idx] = p.add_res ? (y[idx] + v) : v; \
        } \
    } while (0)
    Q41SB_STORE(0u, a0);
    if (B > 1u) Q41SB_STORE(1u, a1);
    if (B > 2u) Q41SB_STORE(2u, a2);
    if (B > 3u) Q41SB_STORE(3u, a3);
    if (B > 4u) Q41SB_STORE(4u, a4);
    if (B > 5u) Q41SB_STORE(5u, a5);
    if (B > 6u) Q41SB_STORE(6u, a6);
    if (B > 7u) Q41SB_STORE(7u, a7);
    #undef Q41SB_STORE
}

// Tenon v12 Round 5: Zero-Spill Explicit-Register (a0..a7) Q4_K Small-Batch GEMM (hoisted scale FMAs!)
kernel void moe_gemm_q4k_small_b(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 4u + sg;
    uint b0 = tgid.y * 8u;
    if (row >= p.out_dim || b0 >= p.batch_b) return;
    uint B = min(8u, p.batch_b - b0), nb = p.in_dim >> 8, qtr = lane & 3u;
    const device uchar* wp = w + (ulong)row * (ulong)nb * 144ul;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device uchar* blk = wp + (ulong)b * 144ul;
        float d = float(*(const device half*)blk), dmin = float(*(const device half*)(blk + 2u));
        const device uchar* sc = blk + 4u;
        const device uchar4* qs4 = (const device uchar4*)(blk + 16u + qtr * 32u);
        uint is = qtr * 2u;
        uchar s0, m0, s1, m1;
        if (is < 4u) {
            s0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u;
            s1 = sc[is + 1u] & 63u; m1 = sc[is + 5u] & 63u;
        } else {
            s0 = (sc[is + 4u] & 0xFu) | ((sc[is - 4u] >> 6) << 4);
            m0 = (sc[is + 4u] >> 4)   | ((sc[is]      >> 6) << 4);
            s1 = (sc[is + 5u] & 0xFu) | ((sc[is - 3u] >> 6) << 4);
            m1 = (sc[is + 5u] >> 4)   | ((sc[is + 1u] >> 6) << 4);
        }
        float d0 = d * float(s0), mn0 = dmin * float(m0), d1 = d * float(s1), mn1 = dmin * float(m1);
        uint xoff = (b << 8) + qtr * 64u;
        float4 pl0 = float4(0.0f), ph0 = float4(0.0f), sl0 = float4(0.0f), sh0 = float4(0.0f);
        float4 pl1 = float4(0.0f), ph1 = float4(0.0f), sl1 = float4(0.0f), sh1 = float4(0.0f);
        #pragma unroll
        for (uint g = 0; g < 8u; g++) {
            uchar4 qv = qs4[g];
            float4 ql = float4(qv & uchar4(0x0Fu)), qh = float4(qv >> 4);
            #define Q4KSB_T(TIDX, PL, PH, SL, SH) do { \
                const device float4* xp = (const device float4*)(x + (ulong)(b0 + (TIDX)) * p.in_dim + xoff); \
                float4 xl = xp[g], xh = xp[g + 8u]; \
                (PL) += dot(ql, xl); (PH) += dot(qh, xh); \
                (SL) += (xl.x + xl.y) + (xl.z + xl.w); (SH) += (xh.x + xh.y) + (xh.z + xh.w); \
            } while (0)
            Q4KSB_T(0u, pl0.x, ph0.x, sl0.x, sh0.x);
            if (B > 1u) Q4KSB_T(1u, pl0.y, ph0.y, sl0.y, sh0.y);
            if (B > 2u) Q4KSB_T(2u, pl0.z, ph0.z, sl0.z, sh0.z);
            if (B > 3u) Q4KSB_T(3u, pl0.w, ph0.w, sl0.w, sh0.w);
            if (B > 4u) Q4KSB_T(4u, pl1.x, ph1.x, sl1.x, sh1.x);
            if (B > 5u) Q4KSB_T(5u, pl1.y, ph1.y, sl1.y, sh1.y);
            if (B > 6u) Q4KSB_T(6u, pl1.z, ph1.z, sl1.z, sh1.z);
            if (B > 7u) Q4KSB_T(7u, pl1.w, ph1.w, sl1.w, sh1.w);
            #undef Q4KSB_T
        }
        float4 r0 = fma(float4(d0), pl0, -float4(mn0) * sl0) + fma(float4(d1), ph0, -float4(mn1) * sh0);
        a0 += r0.x; a1 += r0.y; a2 += r0.z; a3 += r0.w;
        if (B > 4u) {
            float4 r1 = fma(float4(d0), pl1, -float4(mn0) * sl1) + fma(float4(d1), ph1, -float4(mn1) * sh1);
            a4 += r1.x; a5 += r1.y; a6 += r1.z; a7 += r1.w;
        }
    }
    #define Q4KSB_ST(TIDX, ACC) do { \
        float res = simd_sum((ACC)); \
        if (lane == 0u) { ulong idx = (ulong)(b0 + (TIDX)) * p.out_dim + row; y[idx] = p.add_res ? (y[idx] + res) : res; } \
    } while (0)
    Q4KSB_ST(0u, a0);
    if (B > 1u) Q4KSB_ST(1u, a1);
    if (B > 2u) Q4KSB_ST(2u, a2);
    if (B > 3u) Q4KSB_ST(3u, a3);
    if (B > 4u) Q4KSB_ST(4u, a4);
    if (B > 5u) Q4KSB_ST(5u, a5);
    if (B > 6u) Q4KSB_ST(6u, a6);
    if (B > 7u) Q4KSB_ST(7u, a7);
    #undef Q4KSB_ST
}

// Tenon v12 Round 5: Zero-Spill Explicit-Register (a0..a7) Q6_K Small-Batch GEMM!
kernel void moe_gemm_q6k_small_b(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 4u + sg;
    uint b0 = tgid.y * 8u;
    if (row >= p.out_dim || b0 >= p.batch_b) return;
    uint B = min(8u, p.batch_b - b0), nb = p.in_dim >> 8, qtr = lane & 3u;
    const device uchar* wp = w + (ulong)row * (ulong)nb * 210ul;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    uint n128 = (qtr >> 1) * 128u;
    uint l0 = (qtr & 1u) * 16u;
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device uchar* blk = wp + (ulong)b * 210ul;
        float d = float(*(const device half*)(blk + 208u));
        const device uchar* ql = blk + (n128 >> 1);
        const device uchar* qh = blk + 128u + (n128 >> 2);
        const device char* sc = (const device char*)(blk + 192u) + (n128 >> 4);
        uint is = l0 >> 4;
        float ds0 = d * float(sc[is + 0u]), ds2 = d * float(sc[is + 2u]);
        float ds4 = d * float(sc[is + 4u]), ds6 = d * float(sc[is + 6u]);
        uint xbase = (b << 8) + n128 + l0;
        for (uint l = 0; l < 16u; l += 4u) {
            uint lj = l0 + l;
            uchar4 qa = uchar4(ql[lj],      ql[lj + 1u],      ql[lj + 2u],      ql[lj + 3u]);
            uchar4 qb = uchar4(ql[lj + 32u],ql[lj + 33u],     ql[lj + 34u],     ql[lj + 35u]);
            uchar4 qh4 = uchar4(qh[lj],     qh[lj + 1u],      qh[lj + 2u],      qh[lj + 3u]);
            float4 w0 = ds0 * float4(int4((qa & uchar4(0x0Fu)) | (((qh4 >> 0) & uchar4(3u)) << 4)) - 32);
            float4 w1 = ds2 * float4(int4((qb & uchar4(0x0Fu)) | (((qh4 >> 2) & uchar4(3u)) << 4)) - 32);
            float4 w2 = ds4 * float4(int4((qa >> 4)            | (((qh4 >> 4) & uchar4(3u)) << 4)) - 32);
            float4 w3 = ds6 * float4(int4((qb >> 4)            | (((qh4 >> 6) & uchar4(3u)) << 4)) - 32);
            #define Q6KSB_T(TIDX, ACC) do { \
                const device float4* xp = (const device float4*)(x + (ulong)(b0 + (TIDX)) * p.in_dim + xbase + l); \
                float4 s = fma(w0, xp[0], fma(w1, xp[8], fma(w2, xp[16], w3 * xp[24]))); \
                (ACC) += (s.x + s.y) + (s.z + s.w); \
            } while (0)
            Q6KSB_T(0u, a0);
            if (B > 1u) Q6KSB_T(1u, a1);
            if (B > 2u) Q6KSB_T(2u, a2);
            if (B > 3u) Q6KSB_T(3u, a3);
            if (B > 4u) Q6KSB_T(4u, a4);
            if (B > 5u) Q6KSB_T(5u, a5);
            if (B > 6u) Q6KSB_T(6u, a6);
            if (B > 7u) Q6KSB_T(7u, a7);
            #undef Q6KSB_T
        }
    }
    #define Q6KSB_ST(TIDX, ACC) do { \
        float res = simd_sum((ACC)); \
        if (lane == 0u) { ulong idx = (ulong)(b0 + (TIDX)) * p.out_dim + row; y[idx] = p.add_res ? (y[idx] + res) : res; } \
    } while (0)
    Q6KSB_ST(0u, a0);
    if (B > 1u) Q6KSB_ST(1u, a1);
    if (B > 2u) Q6KSB_ST(2u, a2);
    if (B > 3u) Q6KSB_ST(3u, a3);
    if (B > 4u) Q6KSB_ST(4u, a4);
    if (B > 5u) Q6KSB_ST(5u, a5);
    if (B > 6u) Q6KSB_ST(6u, a6);
    if (B > 7u) Q6KSB_ST(7u, a7);
    #undef Q6KSB_ST
}

static inline float fp8_e4m3_dec(uchar b) {
    uint s = b >> 7, e = (b >> 3) & 0xFu, m = b & 7u;
    float v = (e == 0u) ? (float(m) * 0.001953125f) : ldexp(1.0f + float(m) * 0.125f, int(e) - 7);
    return s ? -v : v;
}

static inline uchar fp8_e4m3_enc(float x) {
    uint bits = as_type<uint>(x);
    uchar sign = uchar((bits >> 24) & 0x80u);
    float ax = fabs(x);
    if (ax >= 448.0f) return sign | 0x7Eu;
    if (ax < 0.0009765625f) return sign;
    uint ab = bits & 0x7FFFFFFFu;
    int exp_unb = int((ab >> 23) & 0xFFu) - 127;
    if (exp_unb < -6) {
        int m = int(rint(ax * 512.0f));
        if (m < 0) m = 0; if (m > 7) m = 7;
        return sign | uchar(m);
    }
    int e = exp_unb + 7;
    float frac = (float(ab & 0x7FFFFFu) * (1.0f / 8388608.0f)) * 8.0f;
    int m = int(rint(frac));
    if (m >= 8) { m = 0; e++; }
    if (e >= 15 && m >= 7) { e = 15; m = 6; }
    if (e > 15) { e = 15; m = 6; }
    return sign | uchar((e << 3) | (m & 7));
}

kernel void moe_gemv_fp8_e4m3(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4u + sg;
    if (row >= p.out_dim) return;
    uint nb = p.in_dim >> 5; ulong rb = (ulong)nb * 34ul;
    const device uchar* wp = w + (ulong)row * rb;
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        const device uchar* blk = wp + (ulong)b * 34ul;
        float d = float(*(const device half*)blk);
        const device float* xp = x + (b << 5);
        float s = 0.0f;
        for (uint i = 0; i < 32u; i++) s = fma(fp8_e4m3_dec(blk[2u + i]), xp[i], s);
        acc = fma(d, s, acc);
    }
    float res = simd_sum(acc);
    if (lane == 0u) y[row] = p.add_res ? (y[row] + res) : res;
}

kernel void moe_gemm_fp8_e4m3_batch(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 4u + sg, b0 = tgid.y * 8u;
    if (row >= p.out_dim || b0 >= p.batch_b) return;
    uint bt = min(8u, p.batch_b - b0), nb = p.in_dim >> 5; ulong rb = (ulong)nb * 34ul;
    const device uchar* wp = w + (ulong)row * rb;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (uint b = lane; b < nb; b += 32u) {
        const device uchar* blk = wp + (ulong)b * 34ul;
        float d = float(*(const device half*)blk);
        for (uint i = 0; i < 32u; i += 4u) {
            float4 wv = float4(fp8_e4m3_dec(blk[2u+i]), fp8_e4m3_dec(blk[3u+i]), fp8_e4m3_dec(blk[4u+i]), fp8_e4m3_dec(blk[5u+i])) * d;
            for (uint t = 0; t < bt; t++) {
                float4 xv = *(const device float4*)(x + (ulong)(b0 + t) * p.in_dim + (b << 5) + i);
                acc[t] += dot(wv, xv);
            }
        }
    }
    for (uint t = 0; t < bt; t++) {
        float res = simd_sum(acc[t]);
        if (lane == 0u) { ulong idx = (ulong)(b0 + t) * p.out_dim + row; y[idx] = p.add_res ? (y[idx] + res) : res; }
    }
}

constant float kMxfp4Lut[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };

kernel void moe_gemv_mxfp4(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemvArgs& p [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid * 4u + sg;
    if (row >= p.out_dim) return;
    uint nb = p.in_dim >> 5; ulong rb = (ulong)nb * 17ul;
    const device uchar* wp = w + (ulong)row * rb;
    float acc = 0.0f;
    for (uint b = lane; b < nb; b += 32u) {
        const device uchar* blk = wp + (ulong)b * 17ul;
        float sc = as_type<float>(uint(blk[0]) << 23);
        const device float* xp = x + (b << 5);
        float s = 0.0f;
        for (uint i = 0; i < 16u; i++) {
            uchar bv = blk[1u + i];
            s = fma(kMxfp4Lut[bv & 0xFu], xp[i], fma(kMxfp4Lut[bv >> 4], xp[16u + i], s));
        }
        acc = fma(sc, s, acc);
    }
    float res = simd_sum(acc);
    if (lane == 0u) y[row] = p.add_res ? (y[row] + res) : res;
}

kernel void moe_gemm_mxfp4_batch(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint row = tgid.x * 4u + sg, b0 = tgid.y * 8u;
    if (row >= p.out_dim || b0 >= p.batch_b) return;
    uint bt = min(8u, p.batch_b - b0), nb = p.in_dim >> 5; ulong rb = (ulong)nb * 17ul;
    const device uchar* wp = w + (ulong)row * rb;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (uint b = lane; b < nb; b += 32u) {
        const device uchar* blk = wp + (ulong)b * 17ul;
        float sc = as_type<float>(uint(blk[0]) << 23);
        for (uint i = 0; i < 16u; i += 4u) {
            uchar b0v = blk[1u+i], b1v = blk[2u+i], b2v = blk[3u+i], b3v = blk[4u+i];
            float4 wl = float4(kMxfp4Lut[b0v&0xFu], kMxfp4Lut[b1v&0xFu], kMxfp4Lut[b2v&0xFu], kMxfp4Lut[b3v&0xFu]) * sc;
            float4 wh = float4(kMxfp4Lut[b0v>>4],   kMxfp4Lut[b1v>>4],   kMxfp4Lut[b2v>>4],   kMxfp4Lut[b3v>>4])   * sc;
            for (uint t = 0; t < bt; t++) {
                const device float4* xp = (const device float4*)(x + (ulong)(b0 + t) * p.in_dim + (b << 5) + i);
                acc[t] += dot(wl, xp[0]) + dot(wh, xp[4]);
            }
        }
    }
    for (uint t = 0; t < bt; t++) {
        float res = simd_sum(acc[t]);
        if (lane == 0u) { ulong idx = (ulong)(b0 + t) * p.out_dim + row; y[idx] = p.add_res ? (y[idx] + res) : res; }
    }
}

// =========================================================================
// Tenon v8: Hardware Apple AMX 32x32 Kernels (Q6_K / FP8_E4M3 / MXFP4)
// + Batched Paged Latent MLA + Counting-Sort Grouped MoE Gather/Scatter!
// =========================================================================
// Tenon v12 Round 3: 64x32x32 256-Thread 4N x 2M Zero-Spill Vectorized AMX Q6_K Prefill GEMM!
kernel void moe_gemm_q6k_amx(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 64u, b0 = tgid.y * 64u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 8;
    ulong rb = (ulong)nb * 210ul;
    threadgroup half sX[64][32];
    threadgroup half sW[64][32];
    threadgroup float sO[4][64];

    simdgroup_float8x8 a00(0.0f), a01(0.0f), a02(0.0f), a03(0.0f);
    simdgroup_float8x8 a10(0.0f), a11(0.0f), a12(0.0f), a13(0.0f);
    simdgroup_float8x8 a20(0.0f), a21(0.0f), a22(0.0f), a23(0.0f);
    simdgroup_float8x8 a30(0.0f), a31(0.0f), a32(0.0f), a33(0.0f);

    uint tr = lid >> 1, c0 = (lid & 1u) * 16u;
    uint glob_b = b0 + tr, gr = r0 + tr;
    bool xb_ok = (glob_b < p.batch_b), wr_ok = (gr < p.out_dim);
    const device float* xrow = x + (ulong)(xb_ok ? glob_b : 0u) * p.in_dim + c0;
    const device uchar* wrow = w + (ulong)(wr_ok ? gr : (p.out_dim - 1u)) * rb;

    uint sg_n = (sg & 1u) * 32u, sg_m = (sg >> 1) * 32u;
    bool do_m0 = (b0 + sg_m < p.batch_b), do_m1 = (b0 + sg_m + 8u < p.batch_b);
    bool do_m2 = (b0 + sg_m + 16u < p.batch_b), do_m3 = (b0 + sg_m + 24u < p.batch_b);

    for (uint bk = 0; bk < nb; bk++) {
        const device uchar* blk = wrow + (ulong)bk * 210ul;
        float d = wr_ok ? float(*(const device half*)(blk + 208u)) : 0.0f;
        for (uint sb = 0; sb < 8u; sb++) {
            uint kbase = (bk << 8) + (sb << 5);
            threadgroup half4* sx4 = (threadgroup half4*)(&sX[tr][c0]);
            if (xb_ok) {
                const device float4* xp = (const device float4*)(xrow + kbase);
                sx4[0] = half4(xp[0]); sx4[1] = half4(xp[1]); sx4[2] = half4(xp[2]); sx4[3] = half4(xp[3]);
            } else {
                sx4[0] = half4(0.0h); sx4[1] = half4(0.0h); sx4[2] = half4(0.0h); sx4[3] = half4(0.0h);
            }
            threadgroup half4* sw4 = (threadgroup half4*)(&sW[tr][c0]);
            if (wr_ok) {
                uint n128 = (sb >> 2) << 7, qp = sb & 3u;
                const device uchar* ql = blk + (n128 >> 1) + ((qp & 1u) ? 32u : 0u) + c0;
                const device uchar* qh = blk + 128u + (n128 >> 2) + c0;
                const device char*  sc = (const device char*)(blk + 192u) + (n128 >> 4);
                half hds = half(d * float(sc[(c0 >> 4) + qp * 2u]));
                half h32 = half(-32.0f) * hds;
                uint qh_sh = qp * 2u;
                bool hi_ql = (qp >= 2u);
                #pragma unroll
                for (uint g = 0; g < 4u; g++) {
                    uint o = g * 4u;
                    uchar4 lb = uchar4(ql[o], ql[o+1u], ql[o+2u], ql[o+3u]);
                    uchar4 hb = uchar4(qh[o], qh[o+1u], qh[o+2u], qh[o+3u]);
                    uchar4 qv = (hi_ql ? (lb >> 4) : (lb & uchar4(0x0Fu))) | (((hb >> qh_sh) & uchar4(3u)) << 4);
                    sw4[g] = fma(half4(qv), half4(hds), half4(h32));
                }
            } else {
                sw4[0] = half4(0.0h); sw4[1] = half4(0.0h); sw4[2] = half4(0.0h); sw4[3] = half4(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint kk = 0; kk < 32u; kk += 8u) {
                simdgroup_half8x8 w0, w1, w2, w3, x0, x1, x2, x3;
                simdgroup_load(w0, &sW[sg_n +  0u][kk], 32, ulong2(0, 0), true);
                simdgroup_load(w1, &sW[sg_n +  8u][kk], 32, ulong2(0, 0), true);
                simdgroup_load(w2, &sW[sg_n + 16u][kk], 32, ulong2(0, 0), true);
                simdgroup_load(w3, &sW[sg_n + 24u][kk], 32, ulong2(0, 0), true);
                simdgroup_load(x0, &sX[sg_m +  0u][kk], 32, ulong2(0, 0), false);
                simdgroup_load(x1, &sX[sg_m +  8u][kk], 32, ulong2(0, 0), false);
                simdgroup_load(x2, &sX[sg_m + 16u][kk], 32, ulong2(0, 0), false);
                simdgroup_load(x3, &sX[sg_m + 24u][kk], 32, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(a00, x0, w0, a00); simdgroup_multiply_accumulate(a01, x0, w1, a01);
                simdgroup_multiply_accumulate(a02, x0, w2, a02); simdgroup_multiply_accumulate(a03, x0, w3, a03);
                simdgroup_multiply_accumulate(a10, x1, w0, a10); simdgroup_multiply_accumulate(a11, x1, w1, a11);
                simdgroup_multiply_accumulate(a12, x1, w2, a12); simdgroup_multiply_accumulate(a13, x1, w3, a13);
                simdgroup_multiply_accumulate(a20, x2, w0, a20); simdgroup_multiply_accumulate(a21, x2, w1, a21);
                simdgroup_multiply_accumulate(a22, x2, w2, a22); simdgroup_multiply_accumulate(a23, x2, w3, a23);
                simdgroup_multiply_accumulate(a30, x3, w0, a30); simdgroup_multiply_accumulate(a31, x3, w1, a31);
                simdgroup_multiply_accumulate(a32, x3, w2, a32); simdgroup_multiply_accumulate(a33, x3, w3, a33);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (!do_m0) return;
    #define STORE_TILE_Q6K(ACC, BM_OFF, BN_OFF) do { \
        simdgroup_store((ACC), &sO[sg][0], 8); \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint idx = lane; idx < 64u; idx += 32u) { \
            uint ob = b0 + sg_m + (BM_OFF) + (idx >> 3), orow = r0 + sg_n + (BN_OFF) + (idx & 7u); \
            if (ob < p.batch_b && orow < p.out_dim) { \
                ulong oidx = (ulong)ob * p.out_dim + orow; \
                float val = sO[sg][idx]; \
                y[oidx] = p.add_res ? (y[oidx] + val) : val; \
            } \
        } \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
    } while (0)
    STORE_TILE_Q6K(a00, 0u,  0u); STORE_TILE_Q6K(a01, 0u,  8u); STORE_TILE_Q6K(a02, 0u, 16u); STORE_TILE_Q6K(a03, 0u, 24u);
    if (do_m1) { STORE_TILE_Q6K(a10, 8u,  0u); STORE_TILE_Q6K(a11, 8u,  8u); STORE_TILE_Q6K(a12, 8u, 16u); STORE_TILE_Q6K(a13, 8u, 24u); }
    if (do_m2) { STORE_TILE_Q6K(a20, 16u, 0u); STORE_TILE_Q6K(a21, 16u, 8u); STORE_TILE_Q6K(a22, 16u, 16u); STORE_TILE_Q6K(a23, 16u, 24u); }
    if (do_m3) { STORE_TILE_Q6K(a30, 24u, 0u); STORE_TILE_Q6K(a31, 24u, 8u); STORE_TILE_Q6K(a32, 24u, 16u); STORE_TILE_Q6K(a33, 24u, 24u); }
    #undef STORE_TILE_Q6K
}

// Tenon v12 Round 6b: 16x16x256 1-Superblock-Per-Barrier Q6_K AMX GEMM (2.32x faster LM_head at B=5..32!)
kernel void moe_gemm_q6k_amx16(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 16u, b0 = tgid.y * 16u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 8;
    ulong rb = (ulong)nb * 210ul;
    uint M = min(16u, p.batch_b - b0);
    threadgroup half sX[16][256];
    threadgroup half sW[16][256];
    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);

    uint nr = lid >> 3, seg = lid & 7u;
    uint r_a0 = lid >> 5, c_a = (lid & 31u) << 3;
    uint gr = r0 + nr;
    bool wr_ok = (gr < p.out_dim);
    const device uchar* wrow = w + (ulong)(wr_ok ? gr : (p.out_dim - 1u)) * rb;
    uint sg_k = sg * 64u;
    bool do_m1 = (M > 8u);

    for (uint bk = 0; bk < nb; bk++) {
        uint k_off = bk << 8;
        for (uint rm = 0; rm < 16u; rm += 4u) {
            uint rr = r_a0 + rm;
            if (rr < M) {
                const device float4* xp = (const device float4*)(x + (ulong)(b0 + rr) * p.in_dim + k_off + c_a);
                *(threadgroup half4*)(&sX[rr][c_a])      = half4(xp[0]);
                *(threadgroup half4*)(&sX[rr][c_a + 4u]) = half4(xp[1]);
            } else {
                *(threadgroup half4*)(&sX[rr][c_a])      = half4(0.0h);
                *(threadgroup half4*)(&sX[rr][c_a + 4u]) = half4(0.0h);
            }
        }
        threadgroup half4* sw4 = (threadgroup half4*)(&sW[nr][seg * 32u]);
        if (wr_ok) {
            const device uchar* blk = wrow + (ulong)bk * 210ul;
            float d = float(*(const device half*)(blk + 208u));
            uint h128 = seg >> 2, sub4 = seg & 3u;
            const device uchar* ql = blk + h128 * 64u + (sub4 & 1u) * 32u;
            const device uchar* qh = blk + 128u + h128 * 32u;
            const device char*  sc = (const device char*)(blk + 192u) + h128 * 8u;
            uint qh_sh = sub4 * 2u; bool hi_nib = (sub4 >= 2u);
            half ds0 = half(d * float(sc[sub4 * 2u])), ds1 = half(d * float(sc[sub4 * 2u + 1u]));
            #pragma unroll
            for (uint g = 0; g < 8u; g++) {
                uint k4 = g * 4u;
                uchar4 lv = uchar4(ql[k4], ql[k4+1u], ql[k4+2u], ql[k4+3u]);
                uchar4 hv = uchar4(qh[k4], qh[k4+1u], qh[k4+2u], qh[k4+3u]);
                uchar4 lo4 = hi_nib ? (lv >> 4) : (lv & uchar4(0x0Fu));
                int4 qv = int4(lo4 | (((hv >> qh_sh) & uchar4(3u)) << 4)) - 32;
                sw4[g] = half4(qv) * ((g < 4u) ? ds0 : ds1);
            }
        } else {
            for (uint g = 0; g < 8u; g++) sw4[g] = half4(0.0h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = sg_k; kk < sg_k + 64u; kk += 8u) {
            simdgroup_half8x8 a0, a1, b0v, b1v;
            simdgroup_load(a0,  &sX[0][kk], 256, ulong2(0, 0), false);
            simdgroup_load(b0v, &sW[0][kk], 256, ulong2(0, 0), true);
            simdgroup_load(b1v, &sW[8][kk], 256, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(c00, a0, b0v, c00);
            simdgroup_multiply_accumulate(c01, a0, b1v, c01);
            if (do_m1) {
                simdgroup_load(a1, &sX[8][kk], 256, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(c10, a1, b0v, c10);
                simdgroup_multiply_accumulate(c11, a1, b1v, c11);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float* sOut = (threadgroup float*)(&sW[0][0]);
    uint ob = sg * 256u;
    simdgroup_store(c00, sOut + ob, 16);
    simdgroup_store(c01, sOut + ob + 8u, 16);
    if (do_m1) {
        simdgroup_store(c10, sOut + ob + 128u, 16);
        simdgroup_store(c11, sOut + ob + 136u, 16);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lid; idx < M * 16u; idx += 128u) {
        uint t = idx >> 4, lc = idx & 15u, col = r0 + lc;
        if (col < p.out_dim) {
            float val = sOut[idx] + sOut[256u + idx] + sOut[512u + idx] + sOut[768u + idx];
            ulong oidx = (ulong)(b0 + t) * p.out_dim + col;
            y[oidx] = p.add_res ? (y[oidx] + val) : val;
        }
    }
}

kernel void moe_gemm_fp8_amx(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 32u, b0 = tgid.y * 32u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 34ul;
    threadgroup half sX[32][40];
    threadgroup half sW[32][40];
    threadgroup float sO[4][64];
    uint n_tb = min(4u, (p.batch_b - b0 + 7u) >> 3);
    simdgroup_float8x8 acc[4];
    for (uint tb = 0; tb < 4u; tb++) acc[tb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint tr = lid >> 2, g = lid & 3u, c0 = g * 8u;
    uint gb = b0 + tr, gr = r0 + tr;
    const device uchar* wrow = w + (ulong)min(gr, p.out_dim - 1u) * rb;
    for (uint bk = 0; bk < nb; bk++) {
        uint kbase = bk << 5;
        if (gb < p.batch_b) {
            const device float4* xp = (const device float4*)(x + (ulong)gb * p.in_dim + kbase + c0);
            float4 v0 = xp[0], v1 = xp[1];
            sX[tr][c0+0]=half(v0.x); sX[tr][c0+1]=half(v0.y); sX[tr][c0+2]=half(v0.z); sX[tr][c0+3]=half(v0.w);
            sX[tr][c0+4]=half(v1.x); sX[tr][c0+5]=half(v1.y); sX[tr][c0+6]=half(v1.z); sX[tr][c0+7]=half(v1.w);
        } else if (tr < n_tb * 8u) { for (uint j = 0; j < 8u; j++) sX[tr][c0 + j] = half(0.0f); }
        if (gr < p.out_dim) {
            const device uchar* blk = wrow + (ulong)bk * 34ul;
            float d = float(*(const device half*)blk);
            #pragma unroll
            for (uint j = 0; j < 8u; j++) sW[tr][c0 + j] = half(d * fp8_e4m3_dec(blk[2u + c0 + j]));
        } else { for (uint j = 0; j < 8u; j++) sW[tr][c0 + j] = half(0.0f); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint sg_r = sg * 8u;
        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 mW; simdgroup_load(mW, &sW[sg_r][kk], 40, ulong2(0, 0), true);
            for (uint tb = 0; tb < n_tb; tb++) {
                simdgroup_half8x8 mX; simdgroup_load(mX, &sX[tb * 8u][kk], 40, ulong2(0, 0), false);
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

kernel void moe_gemm_mxfp4_amx(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 32u, b0 = tgid.y * 32u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 17ul;
    threadgroup half sX[32][40];
    threadgroup half sW[32][40];
    threadgroup float sO[4][64];
    uint n_tb = min(4u, (p.batch_b - b0 + 7u) >> 3);
    simdgroup_float8x8 acc[4];
    for (uint tb = 0; tb < 4u; tb++) acc[tb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint tr = lid >> 2, g = lid & 3u, c0 = g * 8u;
    uint gb = b0 + tr, gr = r0 + tr;
    const device uchar* wrow = w + (ulong)min(gr, p.out_dim - 1u) * rb;
    for (uint bk = 0; bk < nb; bk++) {
        uint kbase = bk << 5;
        if (gb < p.batch_b) {
            const device float4* xp = (const device float4*)(x + (ulong)gb * p.in_dim + kbase + c0);
            float4 v0 = xp[0], v1 = xp[1];
            sX[tr][c0+0]=half(v0.x); sX[tr][c0+1]=half(v0.y); sX[tr][c0+2]=half(v0.z); sX[tr][c0+3]=half(v0.w);
            sX[tr][c0+4]=half(v1.x); sX[tr][c0+5]=half(v1.y); sX[tr][c0+6]=half(v1.z); sX[tr][c0+7]=half(v1.w);
        } else if (tr < n_tb * 8u) { for (uint j = 0; j < 8u; j++) sX[tr][c0 + j] = half(0.0f); }
        if (gr < p.out_dim) {
            const device uchar* blk = wrow + (ulong)bk * 17ul;
            float sc = as_type<float>(uint(blk[0]) << 23);
            uint hi = (c0 >= 16u) ? 1u : 0u;
            uint boff = hi ? (c0 - 16u) : c0;
            #pragma unroll
            for (uint j = 0; j < 8u; j++) {
                uchar bv = blk[1u + boff + j];
                sW[tr][c0 + j] = half(kMxfp4Lut[hi ? (bv >> 4) : (bv & 0xFu)] * sc);
            }
        } else { for (uint j = 0; j < 8u; j++) sW[tr][c0 + j] = half(0.0f); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint sg_r = sg * 8u;
        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 mW; simdgroup_load(mW, &sW[sg_r][kk], 40, ulong2(0, 0), true);
            for (uint tb = 0; tb < n_tb; tb++) {
                simdgroup_half8x8 mX; simdgroup_load(mX, &sX[tb * 8u][kk], 40, ulong2(0, 0), false);
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

kernel void moe_gemm_q8_amx_f16(
    const device uchar* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 32u, b0 = tgid.y * 32u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    uint nb = p.in_dim >> 5;
    ulong rb = (ulong)nb * 34ul;
    threadgroup half sX[32][40];
    threadgroup half sW[32][40];
    threadgroup float sO[4][64];
    uint n_tb = min(4u, (p.batch_b - b0 + 7u) >> 3);
    simdgroup_float8x8 acc[4];
    for (uint tb = 0; tb < 4u; tb++) acc[tb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint tr = lid >> 2, g = lid & 3u, c0 = g * 8u;
    uint gb = b0 + tr, gr = r0 + tr;
    const device uchar* wrow = w + (ulong)min(gr, p.out_dim - 1u) * rb;
    for (uint bk = 0; bk < nb; bk++) {
        uint kbase = bk << 5;
        if (gb < p.batch_b) {
            const device float4* xp = (const device float4*)(x + (ulong)gb * p.in_dim + kbase + c0);
            float4 v0 = xp[0], v1 = xp[1];
            sX[tr][c0+0]=half(v0.x); sX[tr][c0+1]=half(v0.y); sX[tr][c0+2]=half(v0.z); sX[tr][c0+3]=half(v0.w);
            sX[tr][c0+4]=half(v1.x); sX[tr][c0+5]=half(v1.y); sX[tr][c0+6]=half(v1.z); sX[tr][c0+7]=half(v1.w);
        } else if (tr < n_tb * 8u) { for (uint j = 0; j < 8u; j++) sX[tr][c0 + j] = half(0.0f); }
        if (gr < p.out_dim) {
            const device uchar* blk = wrow + (ulong)bk * 34ul;
            float d = float(*(const device half*)blk);
            const device char* qs = (const device char*)(blk + 2u);
            #pragma unroll
            for (uint j = 0; j < 8u; j++) sW[tr][c0 + j] = half(d * float(qs[c0 + j]));
        } else { for (uint j = 0; j < 8u; j++) sW[tr][c0 + j] = half(0.0f); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint sg_r = sg * 8u;
        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 mW; simdgroup_load(mW, &sW[sg_r][kk], 40, ulong2(0, 0), true);
            for (uint tb = 0; tb < n_tb; tb++) {
                simdgroup_half8x8 mX; simdgroup_load(mX, &sX[tb * 8u][kk], 40, ulong2(0, 0), false);
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

struct MlaPagedRopeArgs { uint n_heads; uint nope_d; uint rope_d; uint d_head; uint kv_r; float theta; uint rope_neox; uint has_freqs; uint batch_b; uint max_blks; };
kernel void moe_mla_rope_store_paged(
    device float* q [[buffer(0)]], device float* c_kv_pe [[buffer(1)]], device half* lat_cache [[buffer(2)]],
    const device float* rope_freqs [[buffer(3)]], const device uint* pos_arr [[buffer(4)]],
    const device uint* slot_arr [[buffer(5)]], const device int* block_table [[buffer(6)]],
    constant MlaPagedRopeArgs& p [[buffer(7)]],
    uint b [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]
) {
    if (b >= p.batch_b) return;
    uint pos = pos_arr[b], slot = slot_arr[b];
    uint pb = uint(block_table[(ulong)slot * p.max_blks + (pos >> 4)]);
    uint phys_pos = (pb << 4) | (pos & 15u);
    uint half_r = p.rope_d >> 1;
    uint lat_dim = p.kv_r + p.rope_d;
    ulong qb0 = (ulong)b * p.n_heads * p.d_head;
    device float* kvb = c_kv_pe + (ulong)b * lat_dim;
    device float* kpe = kvb + p.kv_r;
    uint tot_qh = p.n_heads * half_r;
    for (uint idx = lid; idx < tot_qh; idx += 128u) {
        uint h = idx / half_r, i = idx - h * half_r;
        uint idx0 = p.rope_neox ? i : (2u * i);
        uint idx1 = p.rope_neox ? (i + half_r) : (2u * i + 1u);
        float inv_f = 1.0f / pow(p.theta, float(2u * i) / float(p.rope_d));
        if (p.has_freqs) inv_f /= rope_freqs[i];
        float ang = float(pos) * inv_f, cs = cos(ang), sn = sin(ang);
        ulong qbase = qb0 + (ulong)h * p.d_head + p.nope_d;
        float q0 = q[qbase + idx0], q1 = q[qbase + idx1];
        q[qbase + idx0] = q0 * cs - q1 * sn;
        q[qbase + idx1] = q0 * sn + q1 * cs;
    }
    for (uint i = lid; i < half_r; i += 128u) {
        uint idx0 = p.rope_neox ? i : (2u * i);
        uint idx1 = p.rope_neox ? (i + half_r) : (2u * i + 1u);
        float inv_f = 1.0f / pow(p.theta, float(2u * i) / float(p.rope_d));
        if (p.has_freqs) inv_f /= rope_freqs[i];
        float ang = float(pos) * inv_f, cs = cos(ang), sn = sin(ang);
        float k0 = kpe[idx0], k1 = kpe[idx1];
        kpe[idx0] = k0 * cs - k1 * sn;
        kpe[idx1] = k0 * sn + k1 * cs;
    }
    threadgroup_barrier(mem_flags::mem_device);
    device half* dst = lat_cache + (ulong)phys_pos * lat_dim;
    for (uint d = lid; d < lat_dim; d += 128u) dst[d] = half(kvb[d]);
}

struct MlaAbsorbBatchArgs { uint n_heads; uint nope_d; uint kv_r; uint d_head; uint batch_b; };
// Tenon v12 Round 6b: Batched AMX/Vectorized MLA Q-Absorption (16N x 16M x 32K -> 1000x fewer TGs, 16x less wuk DRAM!)
kernel void moe_mla_absorb_q_batch(
    const device half* wuk_t [[buffer(0)]], const device float* q [[buffer(1)]], device float* q_abs [[buffer(2)]],
    constant MlaAbsorbBatchArgs& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 16u, h = tgid.y, b0 = tgid.z * 16u;
    if (r0 >= p.kv_r || h >= p.n_heads || b0 >= p.batch_b) return;
    uint bt = min(16u, p.batch_b - b0);
    if (bt <= 4u && (p.nope_d & 3u) == 0u) {
        // Fast low-batch half4 x float4 path: each SG computes 4 rows of r (reads wuk ONCE for all bt tokens!)
        uint nd4 = p.nope_d >> 2;
        for (uint rr = sg; rr < 16u && (r0 + rr) < p.kv_r; rr += 4u) {
            uint r = r0 + rr;
            const device half4* wp = (const device half4*)(wuk_t + ((ulong)h * p.kv_r + r) * p.nope_d);
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            const device float4* qp0 = (const device float4*)(q + ((ulong)(b0 + 0u) * p.n_heads + h) * p.d_head);
            const device float4* qp1 = (bt > 1u) ? (const device float4*)(q + ((ulong)(b0 + 1u) * p.n_heads + h) * p.d_head) : qp0;
            const device float4* qp2 = (bt > 2u) ? (const device float4*)(q + ((ulong)(b0 + 2u) * p.n_heads + h) * p.d_head) : qp0;
            const device float4* qp3 = (bt > 3u) ? (const device float4*)(q + ((ulong)(b0 + 3u) * p.n_heads + h) * p.d_head) : qp0;
            for (uint j = lane; j < nd4; j += 32u) {
                float4 wv = float4(wp[j]);
                a0 += dot(wv, qp0[j]);
                if (bt > 1u) a1 += dot(wv, qp1[j]);
                if (bt > 2u) a2 += dot(wv, qp2[j]);
                if (bt > 3u) a3 += dot(wv, qp3[j]);
            }
            a0 = simd_sum(a0);
            if (lane == 0u) q_abs[((ulong)(b0 + 0u) * p.n_heads + h) * p.kv_r + r] = a0;
            if (bt > 1u) { a1 = simd_sum(a1); if (lane == 0u) q_abs[((ulong)(b0 + 1u) * p.n_heads + h) * p.kv_r + r] = a1; }
            if (bt > 2u) { a2 = simd_sum(a2); if (lane == 0u) q_abs[((ulong)(b0 + 2u) * p.n_heads + h) * p.kv_r + r] = a2; }
            if (bt > 3u) { a3 = simd_sum(a3); if (lane == 0u) q_abs[((ulong)(b0 + 3u) * p.n_heads + h) * p.kv_r + r] = a3; }
        }
        return;
    }
    // Batched AMX 16N x 16M x 32K (2 KB SMEM! 4 SGs split M=2 x N=2 -> 1 acc/SG!)
    threadgroup half sX[16][32];
    threadgroup half sW[16][32];
    threadgroup float sO[4][64];
    simdgroup_float8x8 acc(0.0f);
    uint tr = lid >> 3, c4 = (lid & 7u) * 4u;
    uint sg_n = (sg & 1u) * 8u, sg_m = (sg >> 1) * 8u;
    bool xb_ok = (tr < bt), wr_ok = (r0 + tr < p.kv_r);
    const device float* qrow = q + ((ulong)(xb_ok ? (b0 + tr) : 0u) * p.n_heads + h) * p.d_head;
    const device half*  wrow = wuk_t + ((ulong)h * p.kv_r + (wr_ok ? (r0 + tr) : 0u)) * p.nope_d;
    for (uint k0 = 0; k0 < p.nope_d; k0 += 32u) {
        bool k_ok = (k0 + c4 + 3u < p.nope_d);
        if (xb_ok && k_ok) *(threadgroup half4*)(&sX[tr][c4]) = half4(*(const device float4*)(qrow + k0 + c4));
        else { for (uint j = 0; j < 4u; j++) sX[tr][c4 + j] = (xb_ok && k0 + c4 + j < p.nope_d) ? half(qrow[k0 + c4 + j]) : half(0.0h); }
        if (wr_ok && k_ok) *(threadgroup half4*)(&sW[tr][c4]) = *(const device half4*)(wrow + k0 + c4);
        else { for (uint j = 0; j < 4u; j++) sW[tr][c4 + j] = (wr_ok && k0 + c4 + j < p.nope_d) ? wrow[k0 + c4 + j] : half(0.0h); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 mW, mX;
            simdgroup_load(mW, &sW[sg_n][kk], 32, ulong2(0, 0), true);
            simdgroup_load(mX, &sX[sg_m][kk], 32, ulong2(0, 0), false);
            simdgroup_multiply_accumulate(acc, mX, mW, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc, &sO[sg][0], 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lane; idx < 64u; idx += 32u) {
        uint lm = sg_m + (idx >> 3), ln = sg_n + (idx & 7u);
        if (lm < bt && r0 + ln < p.kv_r) q_abs[((ulong)(b0 + lm) * p.n_heads + h) * p.kv_r + r0 + ln] = sO[sg][idx];
    }
}

struct MlaFlashPagedArgs { uint n_heads; uint kv_r; uint nope_d; uint rope_d; uint d_head; uint batch_b; float scale; uint max_blks; uint swa_window; uint attn_sink; };
// Tenon v12 Round 6b: Zero-Spill 4-Head-Grouped half4 Latent MLA FlashAttention (4x less KV DRAM + ZERO stack spill!)
kernel void moe_flash_mla_latent_paged(
    const device float* q_abs [[buffer(0)]], const device float* q [[buffer(1)]],
    const device half* lat_cache [[buffer(2)]], device float* o_lat [[buffer(3)]],
    const device uint* pos_arr [[buffer(4)]], const device uint* slot_arr [[buffer(5)]],
    const device int* block_table [[buffer(6)]], constant MlaFlashPagedArgs& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint h0 = tgid.x * 4u, b = tgid.y;
    if (h0 >= p.n_heads || b >= p.batch_b) return;
    uint h = h0 + sg;
    bool h_ok = (h < p.n_heads);
    uint kv_len = pos_arr[b] + 1u, slot = slot_arr[b];
    const device int* bt = block_table + (ulong)slot * p.max_blks;
    uint lat_dim = p.kv_r + p.rope_d;
    uint win_start = (p.swa_window > 0u && kv_len > p.swa_window) ? (kv_len - p.swa_window) : 0u;
    uint sink_end = min(p.attn_sink, win_start);
    uint total_vis = sink_end + (kv_len - win_start);
    float scale_log2 = p.scale * 1.4426950408889634f;

    if (p.kv_r == 512u && p.rope_d == 64u) {
        // Zero-spill fast path for DeepSeek-V2/V3/R1 (kv_r=512, rope_d=64): 4 heads share 4 staged KV tokens in SMEM!
        ulong bh = (ulong)b * p.n_heads + (h_ok ? h : 0u);
        uint r16 = lane * 16u, d2 = lane * 2u;
        const device float4* qap = (const device float4*)(q_abs + bh * 512u + r16);
        float4 qa0 = h_ok ? qap[0] : float4(0.0f), qa1 = h_ok ? qap[1] : float4(0.0f);
        float4 qa2 = h_ok ? qap[2] : float4(0.0f), qa3 = h_ok ? qap[3] : float4(0.0f);
        const device float* qpp = q + bh * p.d_head + p.nope_d + d2;
        float2 qp = h_ok ? float2(qpp[0], qpp[1]) : float2(0.0f);
        float4 o0 = float4(0.0f), o1 = float4(0.0f), o2 = float4(0.0f), o3 = float4(0.0f);
        float m_run = -1e30f, l_run = 0.0f;

        threadgroup half sL[4][576];
        for (uint ib = 0; ib < total_vis; ib += 4u) {
            uint nt = min(4u, total_vis - ib);
            uint tot4 = nt * 144u; // 576 / 4 = 144 half4s per token
            for (uint vi = lid; vi < tot4; vi += 128u) {
                uint ti = vi / 144u, c4 = (vi % 144u) * 4u;
                uint idx = ib + ti;
                uint t = (idx < sink_end) ? idx : (win_start + (idx - sink_end));
                uint pt = (uint(bt[t >> 4]) << 4) | (t & 15u);
                *(threadgroup half4*)(&sL[ti][c4]) = *(const device half4*)(lat_cache + (ulong)pt * 576ul + c4);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (h_ok) {
                for (uint ti = 0; ti < nt; ti++) {
                    const threadgroup half4* cp = (const threadgroup half4*)(&sL[ti][r16]);
                    float4 c0 = float4(cp[0]), c1 = float4(cp[1]), c2 = float4(cp[2]), c3 = float4(cp[3]);
                    float2 kp = float2(*(const threadgroup half2*)(&sL[ti][512u + d2]));
                    float dt = dot(qa0, c0) + dot(qa1, c1) + dot(qa2, c2) + dot(qa3, c3) + dot(qp, kp);
                    float sc = simd_sum(dt) * scale_log2;
                    float mn = max(m_run, sc), al = exp2(m_run - mn), wt = exp2(sc - mn);
                    l_run = fma(l_run, al, wt); m_run = mn;
                    o0 = fma(o0, al, wt * c0); o1 = fma(o1, al, wt * c1);
                    o2 = fma(o2, al, wt * c2); o3 = fma(o3, al, wt * c3);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (h_ok) {
            float inv = 1.0f / max(l_run, 1e-12f);
            device float4* op = (device float4*)(o_lat + bh * 512u + r16);
            op[0] = o0 * inv; op[1] = o1 * inv; op[2] = o2 * inv; op[3] = o3 * inv;
        }
        return;
    }

    // Generic fallback for non-512 kv_r (still 1 SG per head!)
    if (!h_ok) return;
    ulong bh = (ulong)b * p.n_heads + h;
    float m_run = -1e30f, l_run = 0.0f;
    float o_acc[16];
    uint rpl = (p.kv_r + 31u) >> 5, dpl = (p.rope_d + 31u) >> 5;
    for (uint k = 0; k < rpl && k < 16u; k++) o_acc[k] = 0.0f;
    for (uint idx = 0; idx < total_vis; idx++) {
        uint t = (idx < sink_end) ? idx : (win_start + (idx - sink_end));
        uint pt = (uint(bt[t >> 4]) << 4) | (t & 15u);
        const device half* cp = lat_cache + (ulong)pt * lat_dim;
        const device half* kp = cp + p.kv_r;
        float dt = 0.0f;
        for (uint k = 0; k < rpl && k < 16u; k++) { uint r = lane + k * 32u; if (r < p.kv_r) dt = fma(q_abs[bh * p.kv_r + r], float(cp[r]), dt); }
        for (uint k = 0; k < dpl; k++) { uint d = lane + k * 32u; if (d < p.rope_d) dt = fma(q[bh * p.d_head + p.nope_d + d], float(kp[d]), dt); }
        float sc = simd_sum(dt) * scale_log2;
        float mn = max(m_run, sc), al = exp2(m_run - mn), wt = exp2(sc - mn);
        l_run = fma(l_run, al, wt); m_run = mn;
        for (uint k = 0; k < rpl && k < 16u; k++) { uint r = lane + k * 32u; if (r < p.kv_r) o_acc[k] = fma(o_acc[k], al, wt * float(cp[r])); }
    }
    float inv = 1.0f / max(l_run, 1e-12f);
    for (uint k = 0; k < rpl && k < 16u; k++) { uint r = lane + k * 32u; if (r < p.kv_r) o_lat[bh * p.kv_r + r] = o_acc[k] * inv; }
}

struct MlaProjVBatchArgs { uint n_heads; uint v_d; uint kv_r; uint out_stride; uint batch_b; };
// Tenon v12 Round 6b: Batched AMX/Vectorized MLA V-Projection (16N x 16M x 32K -> 256x fewer TGs, 16x less wuv DRAM!)
kernel void moe_mla_proj_v_batch(
    const device half* wuv [[buffer(0)]], const device float* o_lat [[buffer(1)]], device float* ctx [[buffer(2)]],
    constant MlaProjVBatchArgs& p [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint j0 = tgid.x * 16u, h = tgid.y, b0 = tgid.z * 16u;
    if (j0 >= p.out_stride || h >= p.n_heads || b0 >= p.batch_b) return;
    uint bt = min(16u, p.batch_b - b0);
    if (bt <= 4u && (p.kv_r & 3u) == 0u) {
        uint kvr4 = p.kv_r >> 2;
        for (uint jj = sg; jj < 16u && (j0 + jj) < p.out_stride; jj += 4u) {
            uint j = j0 + jj;
            if (j >= p.v_d) {
                if (lane == 0u) { for (uint t = 0; t < bt; t++) ctx[((ulong)(b0 + t) * p.n_heads + h) * p.out_stride + j] = 0.0f; }
                continue;
            }
            const device half4* wp = (const device half4*)(wuv + ((ulong)h * p.v_d + j) * p.kv_r);
            const device float4* op0 = (const device float4*)(o_lat + ((ulong)(b0 + 0u) * p.n_heads + h) * p.kv_r);
            const device float4* op1 = (bt > 1u) ? (const device float4*)(o_lat + ((ulong)(b0 + 1u) * p.n_heads + h) * p.kv_r) : op0;
            const device float4* op2 = (bt > 2u) ? (const device float4*)(o_lat + ((ulong)(b0 + 2u) * p.n_heads + h) * p.kv_r) : op0;
            const device float4* op3 = (bt > 3u) ? (const device float4*)(o_lat + ((ulong)(b0 + 3u) * p.n_heads + h) * p.kv_r) : op0;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            for (uint r = lane; r < kvr4; r += 32u) {
                float4 wv = float4(wp[r]);
                a0 += dot(wv, op0[r]);
                if (bt > 1u) a1 += dot(wv, op1[r]);
                if (bt > 2u) a2 += dot(wv, op2[r]);
                if (bt > 3u) a3 += dot(wv, op3[r]);
            }
            a0 = simd_sum(a0);
            if (lane == 0u) ctx[((ulong)(b0 + 0u) * p.n_heads + h) * p.out_stride + j] = a0;
            if (bt > 1u) { a1 = simd_sum(a1); if (lane == 0u) ctx[((ulong)(b0 + 1u) * p.n_heads + h) * p.out_stride + j] = a1; }
            if (bt > 2u) { a2 = simd_sum(a2); if (lane == 0u) ctx[((ulong)(b0 + 2u) * p.n_heads + h) * p.out_stride + j] = a2; }
            if (bt > 3u) { a3 = simd_sum(a3); if (lane == 0u) ctx[((ulong)(b0 + 3u) * p.n_heads + h) * p.out_stride + j] = a3; }
        }
        return;
    }
    threadgroup half sX[16][32];
    threadgroup half sW[16][32];
    threadgroup float sO[4][64];
    simdgroup_float8x8 acc(0.0f);
    uint tr = lid >> 3, c4 = (lid & 7u) * 4u;
    uint sg_n = (sg & 1u) * 8u, sg_m = (sg >> 1) * 8u;
    bool xb_ok = (tr < bt), wr_ok = (j0 + tr < p.v_d);
    const device float* orow = o_lat + ((ulong)(xb_ok ? (b0 + tr) : 0u) * p.n_heads + h) * p.kv_r;
    const device half*  wrow = wuv   + ((ulong)h * p.v_d + (wr_ok ? (j0 + tr) : 0u)) * p.kv_r;
    for (uint k0 = 0; k0 < p.kv_r; k0 += 32u) {
        bool k_ok = (k0 + c4 + 3u < p.kv_r);
        if (xb_ok && k_ok) *(threadgroup half4*)(&sX[tr][c4]) = half4(*(const device float4*)(orow + k0 + c4));
        else { for (uint j = 0; j < 4u; j++) sX[tr][c4 + j] = (xb_ok && k0 + c4 + j < p.kv_r) ? half(orow[k0 + c4 + j]) : half(0.0h); }
        if (wr_ok && k_ok) *(threadgroup half4*)(&sW[tr][c4]) = *(const device half4*)(wrow + k0 + c4);
        else { for (uint j = 0; j < 4u; j++) sW[tr][c4 + j] = (wr_ok && k0 + c4 + j < p.kv_r) ? wrow[k0 + c4 + j] : half(0.0h); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 mW, mX;
            simdgroup_load(mW, &sW[sg_n][kk], 32, ulong2(0, 0), true);
            simdgroup_load(mX, &sX[sg_m][kk], 32, ulong2(0, 0), false);
            simdgroup_multiply_accumulate(acc, mX, mW, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc, &sO[sg][0], 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lane; idx < 64u; idx += 32u) {
        uint lm = sg_m + (idx >> 3), ln = sg_n + (idx & 7u);
        if (lm < bt && j0 + ln < p.out_stride) {
            ctx[((ulong)(b0 + lm) * p.n_heads + h) * p.out_stride + j0 + ln] = (j0 + ln < p.v_d) ? sO[sg][idx] : 0.0f;
        }
    }
}

kernel void moe_gather_rows(
    const device float* src [[buffer(0)]], device float* dst [[buffer(1)]],
    const device uint* tok_idx [[buffer(2)]], constant uint2& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x, r = gid.y;
    if (d >= p.x || r >= p.y) return;
    dst[(ulong)r * p.x + d] = src[(ulong)tok_idx[r] * p.x + d];
}

kernel void moe_scatter_add_rows(
    const device float* src [[buffer(0)]], device float* dst [[buffer(1)]],
    const device uint* tok_idx [[buffer(2)]], const device float* wts [[buffer(3)]],
    constant uint2& p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x, r = gid.y;
    if (d >= p.x || r >= p.y) return;
    dst[(ulong)tok_idx[r] * p.x + d] += wts[r] * src[(ulong)r * p.x + d];
}

// =========================================================================
// Tenon v9: Zero-Host-Sync GPU-Resident Grouped MoE Mega-Kernels (4 launches/layer!)
// =========================================================================
struct MoeRouteSortArgs {
    uint batch_b; uint n_exp; uint top_k; uint gating_func;
    uint norm_wt; uint has_bias; uint n_groups; uint n_group_used;
    float w_scale;
};

kernel void moe_route_count_sort_b(
    const device float* rlogits [[buffer(0)]], const device float* exp_bias [[buffer(1)]],
    device uint* active_e [[buffer(2)]], device uint* grp_off [[buffer(3)]], device uint* grp_cnt [[buffer(4)]],
    device uint* tok_idx [[buffer(5)]], device float* tok_wt [[buffer(6)]],
    device int* tok_slot [[buffer(7)]], device uint* meta [[buffer(8)]],
    constant MoeRouteSortArgs& p [[buffer(9)]], uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup uchar te_all[256 * 16];
    threadgroup float tw_all[256 * 16];
    uint b_lim = min(p.batch_b, 256u);
    for (uint b = tid; b < b_lim; b += 64u) {
        float probs[256];
        float scores[256];
        const device float* rl = rlogits + (ulong)b * p.n_exp;
        if (p.gating_func == 2u) {
            for (uint e = 0; e < p.n_exp; e++) {
                float s = 1.0f / (1.0f + exp(-rl[e]));
                probs[e] = s;
                scores[e] = p.has_bias ? (s + exp_bias[e]) : s;
            }
        } else {
            float mx = -1e30f;
            for (uint e = 0; e < p.n_exp; e++) mx = max(mx, rl[e]);
            float sm = 0.0f;
            for (uint e = 0; e < p.n_exp; e++) { probs[e] = exp(rl[e] - mx); sm += probs[e]; }
            float inv = 1.0f / max(sm, 1e-12f);
            for (uint e = 0; e < p.n_exp; e++) { probs[e] *= inv; scores[e] = probs[e]; }
        }
        if (p.n_groups > 1u && p.n_group_used > 0u && p.n_group_used < p.n_groups) {
            uint gsz = p.n_exp / p.n_groups;
            float gsc[32]; uchar gkeep[32];
            for (uint g = 0; g < p.n_groups && g < 32u; g++) {
                float m1 = -1e30f, m2 = -1e30f;
                for (uint i = 0; i < gsz; i++) {
                    float sv = scores[g * gsz + i];
                    if (sv > m1) { m2 = m1; m1 = sv; } else if (sv > m2) m2 = sv;
                }
                gsc[g] = (p.gating_func == 2u) ? (m1 + m2) : m1;
                gkeep[g] = 0;
            }
            for (uint u = 0; u < p.n_group_used; u++) {
                uint bg = 0; float bs = -1e30f;
                for (uint g = 0; g < p.n_groups && g < 32u; g++) if (!gkeep[g] && gsc[g] > bs) { bs = gsc[g]; bg = g; }
                gkeep[bg] = 1;
            }
            for (uint g = 0; g < p.n_groups && g < 32u; g++) if (!gkeep[g]) for (uint i = 0; i < gsz; i++) scores[g * gsz + i] = -1e30f;
        }
        float wsum = 0.0f;
        for (uint k = 0; k < p.top_k; k++) {
            uint be = 0; float bs = -1e30f;
            for (uint e = 0; e < p.n_exp; e++) if (scores[e] > bs) { bs = scores[e]; be = e; }
            te_all[b * 16u + k] = uchar(be);
            tw_all[b * 16u + k] = probs[be];
            wsum += probs[be];
            scores[be] = -1e30f;
        }
        float mult = p.w_scale;
        if (p.norm_wt && wsum > 1e-12f) mult /= wsum;
        for (uint k = 0; k < p.top_k; k++) tw_all[b * 16u + k] *= mult;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid != 0u) return;
    uint cnt[256];
    for (uint e = 0; e < p.n_exp && e < 256u; e++) cnt[e] = 0u;
    for (uint b = 0; b < b_lim; b++) {
        for (uint k = 0; k < p.top_k; k++) cnt[te_all[b * 16u + k]]++;
    }
    uint exp_cur[256];
    uint n_uniq = 0u, cur_off = 0u;
    for (uint e = 0; e < p.n_exp; e++) {
        if (cnt[e] > 0u) {
            active_e[n_uniq] = e;
            grp_off[n_uniq] = cur_off;
            grp_cnt[n_uniq] = cnt[e];
            exp_cur[e] = cur_off;
            cur_off += cnt[e];
            n_uniq++;
        } else {
            exp_cur[e] = 0u;
        }
    }
    for (uint b = 0; b < b_lim; b++) {
        for (uint k = 0; k < p.top_k; k++) {
            uint e = uint(te_all[b * 16u + k]);
            uint sl = exp_cur[e]++;
            tok_idx[sl] = b;
            tok_wt[sl] = tw_all[b * 16u + k];
            tok_slot[b * p.top_k + k] = int(sl);
        }
    }
    meta[0] = n_uniq;
    uint tot_assign = p.batch_b * p.top_k;
    if (tot_assign > n_uniq) meta[1] += (tot_assign - n_uniq);
    meta[2] += 1u;
}

struct MoeMegaGemmArgs {
    uint out_dim; uint in_dim; uint gtype; uint gtype2;
    ulong exp_bytes; ulong exp_bytes2;
};

// Tenon v12 Round 5: Ultra-Lean Zero-Spill `bt<=4` Mega-Dot + Fused Q4_K Gate+Up!
// Replaces 8-pointer spill-prone mega_dot8 with 4-reg `float4` accumulators (2x higher occupancy on DSV2!)
static inline float4 mega_dot4_fast(
    const device uchar* w, uint row, uint in_dim, uint gtype,
    const device float* x, uint4 tids, uint bt, uint lane
) {
    float4 acc = float4(0.0f);
    if (gtype == 12u) {
        uint nb = in_dim >> 8, qtr = lane & 3u;
        const device uchar* wp = w + (ulong)row * (ulong)nb * 144ul;
        for (uint b = (lane >> 2); b < nb; b += 8u) {
            const device uchar* blk = wp + (ulong)b * 144ul;
            float d = float(*(const device half*)blk), dmin = float(*(const device half*)(blk + 2u));
            const device uchar* sc = blk + 4u;
            const device uchar4* qs4 = (const device uchar4*)(blk + 16u + qtr * 32u);
            uint is = qtr * 2u;
            uchar s0, m0, s1, m1;
            if (is < 4u) {
                s0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u;
                s1 = sc[is + 1u] & 63u; m1 = sc[is + 5u] & 63u;
            } else {
                s0 = (sc[is + 4u] & 0xFu) | ((sc[is - 4u] >> 6) << 4);
                m0 = (sc[is + 4u] >> 4)   | ((sc[is]      >> 6) << 4);
                s1 = (sc[is + 5u] & 0xFu) | ((sc[is - 3u] >> 6) << 4);
                m1 = (sc[is + 5u] >> 4)   | ((sc[is + 1u] >> 6) << 4);
            }
            float d0 = d * float(s0), mn0 = dmin * float(m0), d1 = d * float(s1), mn1 = dmin * float(m1);
            uint xoff = (b << 8) + qtr * 64u;
            float4 pl = float4(0.0f), ph = float4(0.0f), sl = float4(0.0f), sh = float4(0.0f);
            const device float4* xp0 = (const device float4*)(x + (ulong)tids.x * in_dim + xoff);
            const device float4* xp1 = (const device float4*)(x + (ulong)tids.y * in_dim + xoff);
            const device float4* xp2 = (const device float4*)(x + (ulong)tids.z * in_dim + xoff);
            const device float4* xp3 = (const device float4*)(x + (ulong)tids.w * in_dim + xoff);
            #pragma unroll
            for (uint g = 0; g < 8u; g++) {
                uchar4 qv = qs4[g];
                float4 ql = float4(qv & uchar4(0x0Fu)), qh = float4(qv >> 4);
                float4 x0l = xp0[g], x0h = xp0[g+8u]; pl.x += dot(ql, x0l); ph.x += dot(qh, x0h); sl.x += (x0l.x+x0l.y)+(x0l.z+x0l.w); sh.x += (x0h.x+x0h.y)+(x0h.z+x0h.w);
                if (bt > 1u) { float4 x1l = xp1[g], x1h = xp1[g+8u]; pl.y += dot(ql, x1l); ph.y += dot(qh, x1h); sl.y += (x1l.x+x1l.y)+(x1l.z+x1l.w); sh.y += (x1h.x+x1h.y)+(x1h.z+x1h.w); }
                if (bt > 2u) { float4 x2l = xp2[g], x2h = xp2[g+8u]; pl.z += dot(ql, x2l); ph.z += dot(qh, x2h); sl.z += (x2l.x+x2l.y)+(x2l.z+x2l.w); sh.z += (x2h.x+x2h.y)+(x2h.z+x2h.w); }
                if (bt > 3u) { float4 x3l = xp3[g], x3h = xp3[g+8u]; pl.w += dot(ql, x3l); ph.w += dot(qh, x3h); sl.w += (x3l.x+x3l.y)+(x3l.z+x3l.w); sh.w += (x3h.x+x3h.y)+(x3h.z+x3h.w); }
            }
            acc += fma(float4(d0), pl, -float4(mn0) * sl) + fma(float4(d1), ph, -float4(mn1) * sh);
        }
    } else if (gtype == 2u) {
        uint nb = in_dim >> 5;
        const device uchar* wp = w + (ulong)row * (ulong)nb * 18ul;
        for (uint b = lane; b < nb; b += 32u) {
            const device uchar* blk = wp + (ulong)b * 18ul;
            float d = float(*(const device half*)blk);
            const device uchar* qs = blk + 2u;
            uint xoff = b << 5;
            const device float4* xp0 = (const device float4*)(x + (ulong)tids.x * in_dim + xoff);
            const device float4* xp1 = (const device float4*)(x + (ulong)tids.y * in_dim + xoff);
            const device float4* xp2 = (const device float4*)(x + (ulong)tids.z * in_dim + xoff);
            const device float4* xp3 = (const device float4*)(x + (ulong)tids.w * in_dim + xoff);
            float4 s = float4(0.0f);
            #pragma unroll
            for (uint g = 0; g < 4u; g++) {
                uchar4 qv = uchar4(qs[g*4u], qs[g*4u+1u], qs[g*4u+2u], qs[g*4u+3u]);
                float4 wl = float4(qv & uchar4(0x0Fu)) - 8.0f, wh = float4(qv >> 4) - 8.0f;
                s.x += dot(wl, xp0[g]) + dot(wh, xp0[g+4u]);
                if (bt > 1u) s.y += dot(wl, xp1[g]) + dot(wh, xp1[g+4u]);
                if (bt > 2u) s.z += dot(wl, xp2[g]) + dot(wh, xp2[g+4u]);
                if (bt > 3u) s.w += dot(wl, xp3[g]) + dot(wh, xp3[g+4u]);
            }
            acc = fma(float4(d), s, acc);
        }
    } else if (gtype == 3u) {
        uint nb = in_dim >> 5;
        const device uchar* wp = w + (ulong)row * (ulong)nb * 20ul;
        for (uint b = lane; b < nb; b += 32u) {
            const device uchar* blk = wp + (ulong)b * 20ul;
            float d = float(*(const device half*)blk), mn = float(*(const device half*)(blk + 2u));
            const device uchar* qs = blk + 4u;
            uint xoff = b << 5;
            const device float4* xp0 = (const device float4*)(x + (ulong)tids.x * in_dim + xoff);
            const device float4* xp1 = (const device float4*)(x + (ulong)tids.y * in_dim + xoff);
            const device float4* xp2 = (const device float4*)(x + (ulong)tids.z * in_dim + xoff);
            const device float4* xp3 = (const device float4*)(x + (ulong)tids.w * in_dim + xoff);
            #pragma unroll
            for (uint g = 0; g < 4u; g++) {
                uchar4 qv = uchar4(qs[g*4u], qs[g*4u+1u], qs[g*4u+2u], qs[g*4u+3u]);
                float4 wl = fma(float4(qv & uchar4(0x0Fu)), d, mn), wh = fma(float4(qv >> 4), d, mn);
                acc.x += dot(wl, xp0[g]) + dot(wh, xp0[g+4u]);
                if (bt > 1u) acc.y += dot(wl, xp1[g]) + dot(wh, xp1[g+4u]);
                if (bt > 2u) acc.z += dot(wl, xp2[g]) + dot(wh, xp2[g+4u]);
                if (bt > 3u) acc.w += dot(wl, xp3[g]) + dot(wh, xp3[g+4u]);
            }
        }
    } else if (gtype == 6u) {
        // Tenon v12 Round 7: Vectorized packed_ushort4 Q5_0 (fixes 14/26 DSV2 down_exps layers from scalar fallback!)
        uint nb = in_dim >> 5;
        const device uchar* wp = w + (ulong)row * (ulong)nb * 22ul;
        for (uint b = lane; b < nb; b += 32u) {
            const device uchar* blk = wp + (ulong)b * 22ul;
            float d = float(*(const device half*)blk);
            uint qh = uint(*(const device ushort*)(blk + 2u)) | (uint(*(const device ushort*)(blk + 4u)) << 16);
            const device packed_ushort4* q64 = (const device packed_ushort4*)(blk + 6u);
            ushort4 p0 = q64[0], p1 = q64[1];
            uchar4 cs[4] = { as_type<uchar4>(p0.xy), as_type<uchar4>(p0.zw), as_type<uchar4>(p1.xy), as_type<uchar4>(p1.zw) };
            uint xoff = b << 5;
            const device float4* xp0 = (const device float4*)(x + (ulong)tids.x * in_dim + xoff);
            const device float4* xp1 = (const device float4*)(x + (ulong)tids.y * in_dim + xoff);
            const device float4* xp2 = (const device float4*)(x + (ulong)tids.z * in_dim + xoff);
            const device float4* xp3 = (const device float4*)(x + (ulong)tids.w * in_dim + xoff);
            float4 s = float4(0.0f);
            #pragma unroll
            for (uint g = 0; g < 4u; g++) {
                uchar4 c = cs[g];
                uint sl = g * 4u, sh = 16u + g * 4u;
                uint4 hl = uint4((qh >> sl) & 1u, (qh >> (sl + 1u)) & 1u, (qh >> (sl + 2u)) & 1u, (qh >> (sl + 3u)) & 1u) << 4;
                uint4 hh = uint4((qh >> sh) & 1u, (qh >> (sh + 1u)) & 1u, (qh >> (sh + 2u)) & 1u, (qh >> (sh + 3u)) & 1u) << 4;
                float4 wl = float4(int4((uint4(c) & 0x0Fu) | hl) - 16);
                float4 wh = float4(int4((uint4(c) >> 4)   | hh) - 16);
                s.x += dot(wl, xp0[g]) + dot(wh, xp0[g + 4u]);
                if (bt > 1u) s.y += dot(wl, xp1[g]) + dot(wh, xp1[g + 4u]);
                if (bt > 2u) s.z += dot(wl, xp2[g]) + dot(wh, xp2[g + 4u]);
                if (bt > 3u) s.w += dot(wl, xp3[g]) + dot(wh, xp3[g + 4u]);
            }
            acc = fma(float4(d), s, acc);
        }
    } else if (gtype == 8u) {
        uint nb = in_dim >> 5;
        const device uchar* wp = w + (ulong)row * (ulong)nb * 34ul;
        for (uint b = lane; b < nb; b += 32u) {
            const device uchar* blk = wp + (ulong)b * 34ul;
            float d = float(*(const device half*)blk);
            const device packed_short4* q64 = (const device packed_short4*)(blk + 2u);
            uint xoff = b << 5;
            const device float4* xp0 = (const device float4*)(x + (ulong)tids.x * in_dim + xoff);
            const device float4* xp1 = (const device float4*)(x + (ulong)tids.y * in_dim + xoff);
            const device float4* xp2 = (const device float4*)(x + (ulong)tids.z * in_dim + xoff);
            const device float4* xp3 = (const device float4*)(x + (ulong)tids.w * in_dim + xoff);
            float4 s = float4(0.0f);
            #pragma unroll
            for (uint g = 0; g < 4u; g++) {
                short4 pk = q64[g];
                float4 w0 = float4(as_type<char4>(pk.xy)), w1 = float4(as_type<char4>(pk.zw));
                s.x += dot(w0, xp0[g * 2u]) + dot(w1, xp0[g * 2u + 1u]);
                if (bt > 1u) s.y += dot(w0, xp1[g * 2u]) + dot(w1, xp1[g * 2u + 1u]);
                if (bt > 2u) s.z += dot(w0, xp2[g * 2u]) + dot(w1, xp2[g * 2u + 1u]);
                if (bt > 3u) s.w += dot(w0, xp3[g * 2u]) + dot(w1, xp3[g * 2u + 1u]);
            }
            acc = fma(float4(d), s, acc);
        }
    } else if (gtype == 14u) {
        uint nb = in_dim >> 8, qtr = lane & 3u;
        const device uchar* wp = w + (ulong)row * (ulong)nb * 210ul;
        uint n128 = (qtr >> 1) * 128u, l0 = (qtr & 1u) * 16u;
        for (uint b = (lane >> 2); b < nb; b += 8u) {
            const device uchar* blk = wp + (ulong)b * 210ul;
            float d = float(*(const device half*)(blk + 208u));
            const device uchar* ql = blk + (n128 >> 1);
            const device uchar* qh = blk + 128u + (n128 >> 2);
            const device char* sc = (const device char*)(blk + 192u) + (n128 >> 4);
            uint is = l0 >> 4;
            float s0 = d * float(sc[is + 0u]), s1 = d * float(sc[is + 2u]), s2 = d * float(sc[is + 4u]), s3 = d * float(sc[is + 6u]);
            uint xoff = (b << 8) + n128 + l0;
            for (uint l = 0; l < 16u; l += 4u) {
                uint lj = l0 + l;
                uchar4 qa = uchar4(ql[lj], ql[lj+1u], ql[lj+2u], ql[lj+3u]);
                uchar4 qb = uchar4(ql[lj+32u], ql[lj+33u], ql[lj+34u], ql[lj+35u]);
                uchar4 qh4 = uchar4(qh[lj], qh[lj+1u], qh[lj+2u], qh[lj+3u]);
                float4 w0 = s0 * float4(int4((qa & uchar4(0x0Fu)) | (((qh4 >> 0) & uchar4(3u)) << 4)) - 32);
                float4 w1 = s1 * float4(int4((qb & uchar4(0x0Fu)) | (((qh4 >> 2) & uchar4(3u)) << 4)) - 32);
                float4 w2 = s2 * float4(int4((qa >> 4)            | (((qh4 >> 4) & uchar4(3u)) << 4)) - 32);
                float4 w3 = s3 * float4(int4((qb >> 4)            | (((qh4 >> 6) & uchar4(3u)) << 4)) - 32);
                const device float4* p0 = (const device float4*)(x + (ulong)tids.x * in_dim + xoff + l);
                acc.x += dot(w0, p0[0]) + dot(w1, p0[8]) + dot(w2, p0[16]) + dot(w3, p0[24]);
                if (bt > 1u) { const device float4* p1 = (const device float4*)(x + (ulong)tids.y * in_dim + xoff + l); acc.y += dot(w0, p1[0]) + dot(w1, p1[8]) + dot(w2, p1[16]) + dot(w3, p1[24]); }
                if (bt > 2u) { const device float4* p2 = (const device float4*)(x + (ulong)tids.z * in_dim + xoff + l); acc.z += dot(w0, p2[0]) + dot(w1, p2[8]) + dot(w2, p2[16]) + dot(w3, p2[24]); }
                if (bt > 3u) { const device float4* p3 = (const device float4*)(x + (ulong)tids.w * in_dim + xoff + l); acc.w += dot(w0, p3[0]) + dot(w1, p3[8]) + dot(w2, p3[16]) + dot(w3, p3[24]); }
            }
        }
    } else if (gtype == 1u) {
        const device half* wp = (const device half*)w + (ulong)row * in_dim;
        for (uint c = lane; c < in_dim; c += 32u) {
            float wv = float(wp[c]);
            acc.x = fma(wv, x[(ulong)tids.x * in_dim + c], acc.x);
            if (bt > 1u) acc.y = fma(wv, x[(ulong)tids.y * in_dim + c], acc.y);
            if (bt > 2u) acc.z = fma(wv, x[(ulong)tids.z * in_dim + c], acc.z);
            if (bt > 3u) acc.w = fma(wv, x[(ulong)tids.w * in_dim + c], acc.w);
        }
    } else {
        for (uint c = lane; c < in_dim; c += 32u) {
            float wv = dequant_any_elem(w, (ulong)row, c, in_dim, gtype);
            acc.x = fma(wv, x[(ulong)tids.x * in_dim + c], acc.x);
            if (bt > 1u) acc.y = fma(wv, x[(ulong)tids.y * in_dim + c], acc.y);
            if (bt > 2u) acc.z = fma(wv, x[(ulong)tids.z * in_dim + c], acc.z);
            if (bt > 3u) acc.w = fma(wv, x[(ulong)tids.w * in_dim + c], acc.w);
        }
    }
    return acc;
}

// Fused Q4_K Gate+Up `bt<=4` helper: reads activation `x` ONCE for both `gw` and `uw` (halves L1/DRAM X traffic on DSV2!)
static inline void mega_dot4_q4k_gu(
    const device uchar* gw, const device uchar* uw, uint row, uint in_dim,
    const device float* x, uint4 tids, uint bt, uint lane, thread float4& gacc, thread float4& uacc
) {
    uint nb = in_dim >> 8, qtr = lane & 3u, is = qtr * 2u;
    ulong rb = (ulong)nb * 144ul;
    const device uchar* gwp = gw + (ulong)row * rb;
    const device uchar* uwp = uw + (ulong)row * rb;
    float4 ga = float4(0.0f), ua = float4(0.0f);
    for (uint b = (lane >> 2); b < nb; b += 8u) {
        const device uchar* gb = gwp + (ulong)b * 144ul;
        const device uchar* ub = uwp + (ulong)b * 144ul;
        float gd = float(*(const device half*)gb), gdm = float(*(const device half*)(gb + 2u));
        float ud = float(*(const device half*)ub), udm = float(*(const device half*)(ub + 2u));
        const device uchar* gsc = gb + 4u; const device uchar* usc = ub + 4u;
        uchar gs0, gm0, gs1, gm1, us0, um0, us1, um1;
        if (is < 4u) {
            gs0 = gsc[is] & 63u; gm0 = gsc[is+4u] & 63u; gs1 = gsc[is+1u] & 63u; gm1 = gsc[is+5u] & 63u;
            us0 = usc[is] & 63u; um0 = usc[is+4u] & 63u; us1 = usc[is+1u] & 63u; um1 = usc[is+5u] & 63u;
        } else {
            gs0 = (gsc[is+4u]&0xFu)|((gsc[is-4u]>>6)<<4); gm0 = (gsc[is+4u]>>4)|((gsc[is]>>6)<<4);
            gs1 = (gsc[is+5u]&0xFu)|((gsc[is-3u]>>6)<<4); gm1 = (gsc[is+5u]>>4)|((gsc[is+1u]>>6)<<4);
            us0 = (usc[is+4u]&0xFu)|((usc[is-4u]>>6)<<4); um0 = (usc[is+4u]>>4)|((usc[is]>>6)<<4);
            us1 = (usc[is+5u]&0xFu)|((usc[is-3u]>>6)<<4); um1 = (usc[is+5u]>>4)|((usc[is+1u]>>6)<<4);
        }
        const device uchar4* gqs = (const device uchar4*)(gb + 16u + qtr * 32u);
        const device uchar4* uqs = (const device uchar4*)(ub + 16u + qtr * 32u);
        uint xoff = (b << 8) + qtr * 64u;
        const device float4* xp0 = (const device float4*)(x + (ulong)tids.x * in_dim + xoff);
        const device float4* xp1 = (const device float4*)(x + (ulong)tids.y * in_dim + xoff);
        const device float4* xp2 = (const device float4*)(x + (ulong)tids.z * in_dim + xoff);
        const device float4* xp3 = (const device float4*)(x + (ulong)tids.w * in_dim + xoff);
        float4 gpl = float4(0.0f), gph = float4(0.0f), upl = float4(0.0f), uph = float4(0.0f);
        float4 sl = float4(0.0f), sh = float4(0.0f);
        #pragma unroll
        for (uint g = 0; g < 8u; g++) {
            uchar4 gq = gqs[g], uq = uqs[g];
            float4 gql = float4(gq & uchar4(0x0Fu)), gqh = float4(gq >> 4);
            float4 uql = float4(uq & uchar4(0x0Fu)), uqh = float4(uq >> 4);
            #define GU4K_T(XP, C) do { \
                float4 xl = (XP)[g], xh = (XP)[g+8u]; \
                gpl.C += dot(gql, xl); gph.C += dot(gqh, xh); \
                upl.C += dot(uql, xl); uph.C += dot(uqh, xh); \
                sl.C += (xl.x+xl.y)+(xl.z+xl.w); sh.C += (xh.x+xh.y)+(xh.z+xh.w); \
            } while (0)
            GU4K_T(xp0, x);
            if (bt > 1u) GU4K_T(xp1, y);
            if (bt > 2u) GU4K_T(xp2, z);
            if (bt > 3u) GU4K_T(xp3, w);
            #undef GU4K_T
        }
        ga += fma(float4(gd*float(gs0)), gpl, -float4(gdm*float(gm0))*sl) + fma(float4(gd*float(gs1)), gph, -float4(gdm*float(gm1))*sh);
        ua += fma(float4(ud*float(us0)), upl, -float4(udm*float(um0))*sl) + fma(float4(ud*float(us1)), uph, -float4(udm*float(um1))*sh);
    }
    gacc = ga; uacc = ua;
}

kernel void moe_grouped_gu_swiglu_mega(
    const device uchar* gate_exps [[buffer(0)]], const device uchar* up_exps [[buffer(1)]],
    const device float* x [[buffer(2)]], device float* h_grp [[buffer(3)]],
    const device uint* active_e [[buffer(4)]], const device uint* grp_off [[buffer(5)]],
    const device uint* grp_cnt [[buffer(6)]], const device uint* tok_idx [[buffer(7)]],
    const device uint* meta [[buffer(8)]], constant MoeMegaGemmArgs& p [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint ei = tgid.y;
    if (ei >= meta[0]) return;
    uint row = tgid.x * 4u + sg;
    if (row >= p.out_dim) return;
    uint e = active_e[ei], off = grp_off[ei], cnt = grp_cnt[ei];
    const device uchar* gw = gate_exps + (ulong)e * p.exp_bytes;
    const device uchar* uw = up_exps   + (ulong)e * p.exp_bytes2;
    for (uint m0 = 0; m0 < cnt; m0 += 4u) {
        uint bt = min(4u, cnt - m0);
        uint t0 = tok_idx[off + m0];
        uint4 tids = uint4(t0, (bt > 1u) ? tok_idx[off + m0 + 1u] : t0,
                           (bt > 2u) ? tok_idx[off + m0 + 2u] : t0,
                           (bt > 3u) ? tok_idx[off + m0 + 3u] : t0);
        float4 ga, ua;
        if (p.gtype == 12u && p.gtype2 == 12u) {
            mega_dot4_q4k_gu(gw, uw, row, p.in_dim, x, tids, bt, lane, ga, ua);
        } else {
            ga = mega_dot4_fast(gw, row, p.in_dim, p.gtype,  x, tids, bt, lane);
            ua = mega_dot4_fast(uw, row, p.in_dim, p.gtype2, x, tids, bt, lane);
        }
        #define MEGA_GU_ST(K, C) do { \
            float gv = simd_sum(ga.C), uv = simd_sum(ua.C); \
            if (lane == 0u) h_grp[(ulong)(off + m0 + (K)) * p.out_dim + row] = (gv / (1.0f + exp(-gv))) * uv; \
        } while (0)
        MEGA_GU_ST(0u, x);
        if (bt > 1u) MEGA_GU_ST(1u, y);
        if (bt > 2u) MEGA_GU_ST(2u, z);
        if (bt > 3u) MEGA_GU_ST(3u, w);
        #undef MEGA_GU_ST
    }
}

kernel void moe_grouped_down_mega(
    const device uchar* down_exps [[buffer(0)]], const device float* h_grp [[buffer(1)]], device float* y_grp [[buffer(2)]],
    const device uint* active_e [[buffer(3)]], const device uint* grp_off [[buffer(4)]],
    const device uint* grp_cnt [[buffer(5)]], const device uint* meta [[buffer(6)]],
    constant MoeMegaGemmArgs& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint ei = tgid.y;
    if (ei >= meta[0]) return;
    uint row = tgid.x * 4u + sg;
    if (row >= p.out_dim) return;
    uint e = active_e[ei], off = grp_off[ei], cnt = grp_cnt[ei];
    const device uchar* dw = down_exps + (ulong)e * p.exp_bytes;
    for (uint m0 = 0; m0 < cnt; m0 += 4u) {
        uint bt = min(4u, cnt - m0);
        uint s0 = off + m0;
        uint4 sids = uint4(s0, (bt > 1u) ? (s0 + 1u) : s0, (bt > 2u) ? (s0 + 2u) : s0, (bt > 3u) ? (s0 + 3u) : s0);
        float4 acc = mega_dot4_fast(dw, row, p.in_dim, p.gtype, h_grp, sids, bt, lane);
        #define MEGA_DN_ST(K, C) do { \
            float val = simd_sum(acc.C); \
            if (lane == 0u) y_grp[(ulong)(s0 + (K)) * p.out_dim + row] = val; \
        } while (0)
        MEGA_DN_ST(0u, x);
        if (bt > 1u) MEGA_DN_ST(1u, y);
        if (bt > 2u) MEGA_DN_ST(2u, z);
        if (bt > 3u) MEGA_DN_ST(3u, w);
        #undef MEGA_DN_ST
    }
}

// Tenon v11: Register-Direct Vectorized half4 FP16-AMX Grouped MoE Mega-Kernel v3
// Eliminates scalar half8 temp arrays & writes aligned half4 vectors directly into threadgroup SMEM!
static inline void mega_dequant_h4_direct(
    const device uchar* w, uint row, uint bk, uint in_dim, uint gtype, uint c0, threadgroup half* dst
) {
    threadgroup half4* dst4 = (threadgroup half4*)dst;
    if (gtype == 12u) {
        uint nb256 = in_dim >> 8, sb = bk >> 3, sub = bk & 7u, qtr = sub >> 1, hi = sub & 1u;
        const device uchar* blk = w + ((ulong)row * nb256 + sb) * 144ul;
        float d = float(*(const device half*)blk), dmin = float(*(const device half*)(blk + 2u));
        const device uchar* sc = blk + 4u;
        uchar s0, m0;
        if (sub < 4u) { s0 = sc[sub] & 63u; m0 = sc[sub + 4u] & 63u; }
        else { s0 = (sc[sub + 4u] & 0xFu) | ((sc[sub - 4u] >> 6) << 4); m0 = (sc[sub + 4u] >> 4) | ((sc[sub] >> 6) << 4); }
        half hsc = half(hi ? (d * float(s0) * 0.0625f) : (d * float(s0)));
        half hmn = half(-dmin * float(m0));
        const device uchar4* qs4 = (const device uchar4*)(blk + 16u + qtr * 32u + c0);
        uchar4 b0 = qs4[0], b1 = qs4[1];
        half4 q0 = hi ? half4(b0 & uchar4(0xF0u)) : half4(b0 & uchar4(0x0Fu));
        half4 q1 = hi ? half4(b1 & uchar4(0xF0u)) : half4(b1 & uchar4(0x0Fu));
        dst4[0] = fma(half4(hsc), q0, half4(hmn));
        dst4[1] = fma(half4(hsc), q1, half4(hmn));
    } else if (gtype == 6u) {
        // Tenon v12 Round 7: Vectorized packed_ushort4 Q5_0 AMX staging (fixes 14/26 DSV2 down_exps layers!)
        uint nb = in_dim >> 5;
        const device uchar* blk = w + ((ulong)row * nb + bk) * 22ul;
        half hd = *(const device half*)blk;
        half h16 = half(-16.0f) * hd;
        uint qh = uint(*(const device ushort*)(blk + 2u)) | (uint(*(const device ushort*)(blk + 4u)) << 16);
        bool hi = (c0 >= 16u);
        uint qoff = hi ? (c0 - 16u) : c0;
        ushort4 q64 = *(const device packed_ushort4*)(blk + 6u + qoff);
        uchar4 b0 = as_type<uchar4>(q64.xy), b1 = as_type<uchar4>(q64.zw);
        uint4 h0 = uint4((qh >> c0) & 1u, (qh >> (c0 + 1u)) & 1u, (qh >> (c0 + 2u)) & 1u, (qh >> (c0 + 3u)) & 1u) << 4;
        uint4 h1 = uint4((qh >> (c0 + 4u)) & 1u, (qh >> (c0 + 5u)) & 1u, (qh >> (c0 + 6u)) & 1u, (qh >> (c0 + 7u)) & 1u) << 4;
        uint4 l0 = hi ? (uint4(b0) >> 4) : (uint4(b0) & 0x0Fu);
        uint4 l1 = hi ? (uint4(b1) >> 4) : (uint4(b1) & 0x0Fu);
        dst4[0] = fma(half4(l0 | h0), half4(hd), half4(h16));
        dst4[1] = fma(half4(l1 | h1), half4(hd), half4(h16));
    } else if (gtype == 8u) {
        uint nb = in_dim >> 5;
        const device uchar* blk = w + ((ulong)row * nb + bk) * 34ul;
        half hd = *(const device half*)blk;
        short4 pk = *(const device packed_short4*)(blk + 2u + c0);
        dst4[0] = half4(as_type<char4>(pk.xy)) * hd;
        dst4[1] = half4(as_type<char4>(pk.zw)) * hd;
    } else if (gtype == 2u) {
        uint nb = in_dim >> 5;
        const device uchar* blk = w + ((ulong)row * nb + bk) * 18ul;
        half hd = *(const device half*)blk;
        half h8 = half(-8.0f) * hd;
        bool hi = (c0 >= 16u);
        ushort4 q64 = *(const device packed_ushort4*)(blk + 2u + (hi ? (c0 - 16u) : c0));
        uchar4 b0 = as_type<uchar4>(q64.xy), b1 = as_type<uchar4>(q64.zw);
        dst4[0] = fma(hi ? half4(b0 >> 4) : half4(b0 & uchar4(0x0Fu)), half4(hd), half4(h8));
        dst4[1] = fma(hi ? half4(b1 >> 4) : half4(b1 & uchar4(0x0Fu)), half4(hd), half4(h8));
    } else if (gtype == 3u) {
        uint nb = in_dim >> 5;
        const device uchar* blk = w + ((ulong)row * nb + bk) * 20ul;
        float d = float(*(const device half*)blk), mn = float(*(const device half*)(blk + 2u));
        const device uchar* qs = blk + 4u + ((c0 < 16u) ? c0 : (c0 - 16u));
        float4 v0 = (c0 < 16u)
            ? float4(float(qs[0] & 15u), float(qs[1] & 15u), float(qs[2] & 15u), float(qs[3] & 15u))
            : float4(float(qs[0] >> 4),  float(qs[1] >> 4),  float(qs[2] >> 4),  float(qs[3] >> 4));
        float4 v1 = (c0 < 16u)
            ? float4(float(qs[4] & 15u), float(qs[5] & 15u), float(qs[6] & 15u), float(qs[7] & 15u))
            : float4(float(qs[4] >> 4),  float(qs[5] >> 4),  float(qs[6] >> 4),  float(qs[7] >> 4));
        dst4[0] = half4(fma(float4(d), v0, float4(mn)));
        dst4[1] = half4(fma(float4(d), v1, float4(mn)));
    } else if (gtype == 14u) {
        uint nb256 = in_dim >> 8, sb = bk >> 3, sub = bk & 7u;
        uint n128 = (sub >> 2) * 128u, qp = sub & 3u;
        const device uchar* blk = w + ((ulong)row * nb256 + sb) * 210ul;
        float d = float(*(const device half*)(blk + 208u));
        const device uchar* ql = blk + (n128 >> 1) + ((qp & 1u) ? 32u : 0u);
        const device uchar* qh = blk + 128u + (n128 >> 2);
        const device char* sc = (const device char*)(blk + 192u) + (n128 >> 4);
        float sv = d * float(sc[qp * 2u + (c0 >> 4)]);
        uint sh = qp * 2u;
        bool hi_n = (qp >= 2u);
        half4 h0, h1;
        #pragma unroll
        for (uint j = 0; j < 4u; j++) {
            uint l0 = c0 + j, l1 = c0 + 4u + j;
            uint lo0 = hi_n ? (ql[l0] >> 4) : (ql[l0] & 15u);
            uint lo1 = hi_n ? (ql[l1] >> 4) : (ql[l1] & 15u);
            h0[j] = half(sv * float(int(lo0 | (((qh[l0] >> sh) & 3u) << 4)) - 32));
            h1[j] = half(sv * float(int(lo1 | (((qh[l1] >> sh) & 3u) << 4)) - 32));
        }
        dst4[0] = h0; dst4[1] = h1;
    } else if (gtype == 39u) {
        uint nb = in_dim >> 5;
        const device uchar* blk = w + ((ulong)row * nb + bk) * 34ul;
        float d = float(*(const device half*)blk);
        const device uchar* qs = blk + 2u + c0;
        dst4[0] = half4(d * float4(fp8_e4m3_dec(qs[0]), fp8_e4m3_dec(qs[1]), fp8_e4m3_dec(qs[2]), fp8_e4m3_dec(qs[3])));
        dst4[1] = half4(d * float4(fp8_e4m3_dec(qs[4]), fp8_e4m3_dec(qs[5]), fp8_e4m3_dec(qs[6]), fp8_e4m3_dec(qs[7])));
    } else {
        uint k0 = (bk << 5) + c0;
        half4 h0, h1;
        #pragma unroll
        for (uint j = 0; j < 4u; j++) {
            h0[j] = half(dequant_any_elem(w, (ulong)row, k0 + j, in_dim, gtype));
            h1[j] = half(dequant_any_elem(w, (ulong)row, k0 + 4u + j, in_dim, gtype));
        }
        dst4[0] = h0; dst4[1] = h1;
    }
}

// Tenon v13 Pass 2: Zero-Spill 32N x 32M x 32K Fused Grouped MoE Mega-Kernel (8 accs/SG ZERO spill, 2 barriers/K, 6 KB SMEM!)
kernel void moe_grouped_gu_swiglu_mega_amx(
    const device uchar* gate_exps [[buffer(0)]], const device uchar* up_exps [[buffer(1)]],
    const device float* x [[buffer(2)]], device float* h_grp [[buffer(3)]],
    const device uint* active_e [[buffer(4)]], const device uint* grp_off [[buffer(5)]],
    const device uint* grp_cnt [[buffer(6)]], const device uint* tok_idx [[buffer(7)]],
    device atomic_uint* meta [[buffer(8)]], constant MoeMegaGemmArgs& p [[buffer(9)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint ei = tgid.y;
    if (ei >= atomic_load_explicit(&meta[0], memory_order_relaxed)) return;
    uint r0 = tgid.x * 32u;
    if (r0 >= p.out_dim) return;
    uint e = active_e[ei], off = grp_off[ei], cnt = grp_cnt[ei];
    const device uchar* gw = gate_exps + (ulong)e * p.exp_bytes;
    const device uchar* uw = up_exps   + (ulong)e * p.exp_bytes2;

    if (cnt < 4u) {
        uint t0 = tok_idx[off];
        uint4 tids = uint4(t0, (cnt > 1u) ? tok_idx[off + 1u] : t0,
                           (cnt > 2u) ? tok_idx[off + 2u] : t0, t0);
        for (uint rr = sg; rr < 32u && (r0 + rr) < p.out_dim; rr += 4u) {
            uint row = r0 + rr;
            float4 ga, ua;
            if (p.gtype == 12u && p.gtype2 == 12u) {
                mega_dot4_q4k_gu(gw, uw, row, p.in_dim, x, tids, cnt, lane, ga, ua);
            } else {
                ga = mega_dot4_fast(gw, row, p.in_dim, p.gtype,  x, tids, cnt, lane);
                ua = mega_dot4_fast(uw, row, p.in_dim, p.gtype2, x, tids, cnt, lane);
            }
            #define AMX_GU_FB(K, C) do { \
                float gv = simd_sum(ga.C), uv = simd_sum(ua.C); \
                if (lane == 0u) h_grp[(ulong)(off + (K)) * p.out_dim + row] = (gv / (1.0f + exp(-gv))) * uv; \
            } while (0)
            AMX_GU_FB(0u, x);
            if (cnt > 1u) AMX_GU_FB(1u, y);
            if (cnt > 2u) AMX_GU_FB(2u, z);
            #undef AMX_GU_FB
        }
        return;
    }

    if (lid == 0u && tgid.x == 0u) atomic_fetch_add_explicit(&meta[3], 1u, memory_order_relaxed);
    threadgroup half sX[32][32];
    threadgroup half sWG[32][32];
    threadgroup half sWU[32][32];
    threadgroup float sOG[4][64];
    threadgroup float sOU[4][64];
    uint tr = lid >> 2, g = lid & 3u, c0 = g * 8u;
    uint gr = r0 + tr, sg_r = sg * 8u, nb = p.in_dim >> 5;

    for (uint m0 = 0; m0 < cnt; m0 += 32u) {
        uint bt = min(32u, cnt - m0);
        bool do_m1 = (bt > 8u), do_m2 = (bt > 16u), do_m3 = (bt > 24u);
        uint my_tid = (tr < bt) ? tok_idx[off + m0 + tr] : 0u;
        simdgroup_float8x8 accG0(0.0f), accG1(0.0f), accG2(0.0f), accG3(0.0f);
        simdgroup_float8x8 accU0(0.0f), accU1(0.0f), accU2(0.0f), accU3(0.0f);

        for (uint bk = 0; bk < nb; bk++) {
            uint kbase = bk << 5;
            threadgroup half4* sx4 = (threadgroup half4*)&sX[tr][c0];
            if (tr < bt) {
                const device float4* xp = (const device float4*)(x + (ulong)my_tid * p.in_dim + kbase + c0);
                sx4[0] = half4(xp[0]); sx4[1] = half4(xp[1]);
            } else {
                sx4[0] = half4(0.0h); sx4[1] = half4(0.0h);
            }
            if (gr < p.out_dim) {
                mega_dequant_h4_direct(gw, gr, bk, p.in_dim, p.gtype,  c0, &sWG[tr][c0]);
                mega_dequant_h4_direct(uw, gr, bk, p.in_dim, p.gtype2, c0, &sWU[tr][c0]);
            } else {
                threadgroup half4* g4 = (threadgroup half4*)&sWG[tr][c0]; g4[0] = half4(0.0h); g4[1] = half4(0.0h);
                threadgroup half4* u4 = (threadgroup half4*)&sWU[tr][c0]; u4[0] = half4(0.0h); u4[1] = half4(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint kk = 0; kk < 32u; kk += 8u) {
                simdgroup_half8x8 mX0, mX1, mX2, mX3, mWG, mWU;
                simdgroup_load(mX0, &sX[0][kk],   32, ulong2(0, 0), false);
                simdgroup_load(mX1, &sX[8][kk],   32, ulong2(0, 0), false);
                simdgroup_load(mX2, &sX[16][kk],  32, ulong2(0, 0), false);
                simdgroup_load(mX3, &sX[24][kk],  32, ulong2(0, 0), false);
                simdgroup_load(mWG, &sWG[sg_r][kk], 32, ulong2(0, 0), true);
                simdgroup_load(mWU, &sWU[sg_r][kk], 32, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(accG0, mX0, mWG, accG0);
                simdgroup_multiply_accumulate(accG1, mX1, mWG, accG1);
                simdgroup_multiply_accumulate(accG2, mX2, mWG, accG2);
                simdgroup_multiply_accumulate(accG3, mX3, mWG, accG3);
                simdgroup_multiply_accumulate(accU0, mX0, mWU, accU0);
                simdgroup_multiply_accumulate(accU1, mX1, mWU, accU1);
                simdgroup_multiply_accumulate(accU2, mX2, mWU, accU2);
                simdgroup_multiply_accumulate(accU3, mX3, mWU, accU3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #define STORE_MGU(AG, AU, M_OFF) do { \
            simdgroup_store((AG), &sOG[sg][0], 8); \
            simdgroup_store((AU), &sOU[sg][0], 8); \
            simdgroup_barrier(mem_flags::mem_threadgroup); \
            for (uint idx = lane; idx < 64u; idx += 32u) { \
                uint om = (M_OFF) + (idx >> 3), orow = r0 + sg_r + (idx & 7u); \
                if (om < bt && orow < p.out_dim) { \
                    float gv = sOG[sg][idx], uv = sOU[sg][idx]; \
                    h_grp[(ulong)(off + m0 + om) * p.out_dim + orow] = (gv / (1.0f + exp(-gv))) * uv; \
                } \
            } \
            simdgroup_barrier(mem_flags::mem_threadgroup); \
        } while (0)
        STORE_MGU(accG0, accU0, 0u);
        if (do_m1) STORE_MGU(accG1, accU1, 8u);
        if (do_m2) STORE_MGU(accG2, accU2, 16u);
        if (do_m3) STORE_MGU(accG3, accU3, 24u);
        #undef STORE_MGU
    }
}

kernel void moe_grouped_down_mega_amx(
    const device uchar* down_exps [[buffer(0)]], const device float* h_grp [[buffer(1)]], device float* y_grp [[buffer(2)]],
    const device uint* active_e [[buffer(3)]], const device uint* grp_off [[buffer(4)]],
    const device uint* grp_cnt [[buffer(5)]], device atomic_uint* meta [[buffer(6)]],
    constant MoeMegaGemmArgs& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint ei = tgid.y;
    if (ei >= atomic_load_explicit(&meta[0], memory_order_relaxed)) return;
    uint r0 = tgid.x * 64u;
    if (r0 >= p.out_dim) return;
    uint e = active_e[ei], off = grp_off[ei], cnt = grp_cnt[ei];
    const device uchar* dw = down_exps + (ulong)e * p.exp_bytes;

    if (cnt < 4u) {
        uint4 sids = uint4(off, (cnt > 1u) ? (off + 1u) : off, (cnt > 2u) ? (off + 2u) : off, off);
        for (uint rr = sg; rr < 64u && (r0 + rr) < p.out_dim; rr += 4u) {
            uint row = r0 + rr;
            float4 acc = mega_dot4_fast(dw, row, p.in_dim, p.gtype, h_grp, sids, cnt, lane);
            #define AMX_DN_FB(K, C) do { \
                float val = simd_sum(acc.C); \
                if (lane == 0u) y_grp[(ulong)(off + (K)) * p.out_dim + row] = val; \
            } while (0)
            AMX_DN_FB(0u, x);
            if (cnt > 1u) AMX_DN_FB(1u, y);
            if (cnt > 2u) AMX_DN_FB(2u, z);
            #undef AMX_DN_FB
        }
        return;
    }

    if (lid == 0u && tgid.x == 0u) atomic_fetch_add_explicit(&meta[3], 1u, memory_order_relaxed);
    threadgroup half sX[32][32];
    threadgroup half sW[64][32];
    threadgroup float sO[4][64];
    uint tr = lid >> 2, g = lid & 3u, c0 = g * 8u;
    uint gr0 = r0 + tr, gr1 = r0 + tr + 32u, sg_r = sg * 16u, nb = p.in_dim >> 5;

    for (uint m0 = 0; m0 < cnt; m0 += 32u) {
        uint bt = min(32u, cnt - m0);
        bool do_m1 = (bt > 8u), do_m2 = (bt > 16u), do_m3 = (bt > 24u);
        simdgroup_float8x8 acc00(0.0f), acc01(0.0f), acc02(0.0f), acc03(0.0f);
        simdgroup_float8x8 acc10(0.0f), acc11(0.0f), acc12(0.0f), acc13(0.0f);
        for (uint bk = 0; bk < nb; bk++) {
            uint kbase = bk << 5;
            threadgroup half4* sx4 = (threadgroup half4*)&sX[tr][c0];
            if (tr < bt) {
                const device float4* xp = (const device float4*)(h_grp + (ulong)(off + m0 + tr) * p.in_dim + kbase + c0);
                sx4[0] = half4(xp[0]); sx4[1] = half4(xp[1]);
            } else {
                sx4[0] = half4(0.0h); sx4[1] = half4(0.0h);
            }
            if (gr0 < p.out_dim) {
                mega_dequant_h4_direct(dw, gr0, bk, p.in_dim, p.gtype, c0, &sW[tr][c0]);
            } else {
                threadgroup half4* sw4 = (threadgroup half4*)&sW[tr][c0]; sw4[0] = half4(0.0h); sw4[1] = half4(0.0h);
            }
            if (gr1 < p.out_dim) {
                mega_dequant_h4_direct(dw, gr1, bk, p.in_dim, p.gtype, c0, &sW[tr + 32u][c0]);
            } else {
                threadgroup half4* sw4 = (threadgroup half4*)&sW[tr + 32u][c0]; sw4[0] = half4(0.0h); sw4[1] = half4(0.0h);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            #pragma unroll
            for (uint kk = 0; kk < 32u; kk += 8u) {
                simdgroup_half8x8 mX0, mX1, mX2, mX3, mW0, mW1;
                simdgroup_load(mX0, &sX[0][kk],         32, ulong2(0, 0), false);
                simdgroup_load(mX1, &sX[8][kk],         32, ulong2(0, 0), false);
                simdgroup_load(mX2, &sX[16][kk],        32, ulong2(0, 0), false);
                simdgroup_load(mX3, &sX[24][kk],        32, ulong2(0, 0), false);
                simdgroup_load(mW0, &sW[sg_r][kk],      32, ulong2(0, 0), true);
                simdgroup_load(mW1, &sW[sg_r + 8u][kk], 32, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc00, mX0, mW0, acc00);
                simdgroup_multiply_accumulate(acc01, mX1, mW0, acc01);
                simdgroup_multiply_accumulate(acc02, mX2, mW0, acc02);
                simdgroup_multiply_accumulate(acc03, mX3, mW0, acc03);
                simdgroup_multiply_accumulate(acc10, mX0, mW1, acc10);
                simdgroup_multiply_accumulate(acc11, mX1, mW1, acc11);
                simdgroup_multiply_accumulate(acc12, mX2, mW1, acc12);
                simdgroup_multiply_accumulate(acc13, mX3, mW1, acc13);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        #define STORE_MDN(ACC, M_OFF, N_OFF) do { \
            simdgroup_store((ACC), &sO[sg][0], 8); \
            simdgroup_barrier(mem_flags::mem_threadgroup); \
            for (uint idx = lane; idx < 64u; idx += 32u) { \
                uint om = (M_OFF) + (idx >> 3), orow = r0 + sg_r + (N_OFF) + (idx & 7u); \
                if (om < bt && orow < p.out_dim) { \
                    y_grp[(ulong)(off + m0 + om) * p.out_dim + orow] = sO[sg][idx]; \
                } \
            } \
            simdgroup_barrier(mem_flags::mem_threadgroup); \
        } while (0)
        STORE_MDN(acc00, 0u,  0u); STORE_MDN(acc10, 0u,  8u);
        if (do_m1) { STORE_MDN(acc01, 8u,  0u); STORE_MDN(acc11, 8u,  8u); }
        if (do_m2) { STORE_MDN(acc02, 16u, 0u); STORE_MDN(acc12, 16u, 8u); }
        if (do_m3) { STORE_MDN(acc03, 24u, 0u); STORE_MDN(acc13, 24u, 8u); }
        #undef STORE_MDN
    }
}

struct MoeReduceArgs { uint dm; uint batch_b; uint top_k; };
kernel void moe_reduce_tokens_mega(
    const device float* y_grp [[buffer(0)]], device float* dst_x [[buffer(1)]],
    const device int* tok_slot [[buffer(2)]], const device float* tok_wt [[buffer(3)]],
    constant MoeReduceArgs& p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]
) {
    uint d = gid.x, b = gid.y;
    if (d >= p.dm || b >= p.batch_b) return;
    float sum = 0.0f;
    for (uint k = 0; k < p.top_k; k++) {
        int sl = tok_slot[b * p.top_k + k];
        if (sl >= 0) sum = fma(tok_wt[sl], y_grp[(ulong)sl * p.dm + d], sum);
    }
    dst_x[(ulong)b * p.dm + d] += sum;
}

struct MoeLoraArgs { uint out_dim; uint in_dim; uint rank; uint batch_b; float scale; };

kernel void moe_lora_proj_a(
    const device float* A [[buffer(0)]], const device float* x [[buffer(1)]], device float* tmp [[buffer(2)]],
    constant MoeLoraArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r = tgid.x, b = tgid.y;
    if (r >= p.rank || b >= p.batch_b) return;
    const device float* ar = A + (ulong)r * p.in_dim;
    const device float* xb = x + (ulong)b * p.in_dim;
    float acc = 0.0f;
    for (uint k = lane; k < p.in_dim; k += 32u) acc = fma(ar[k], xb[k], acc);
    float s = simd_sum(acc);
    if (lane == 0u) tmp[(ulong)b * p.rank + r] = s;
}

kernel void moe_lora_proj_b_add(
    const device float* B [[buffer(0)]], const device float* tmp [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeLoraArgs& p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint o = gid.x, b = gid.y;
    if (o >= p.out_dim || b >= p.batch_b) return;
    const device float* bo = B + (ulong)o * p.rank;
    const device float* tb = tmp + (ulong)b * p.rank;
    float acc = 0.0f;
    for (uint r = 0; r < p.rank; r++) acc = fma(bo[r], tb[r], acc);
    y[(ulong)b * p.out_dim + o] += p.scale * acc;
}

// Tenon v9: Mixed-Batch Heterogeneous Multi-Tenant S-LoRA / Punica BGMV Fused Kernels (1 launch for ALL tenants!)
struct MoeBgmvArgs {
    uint out_dim; uint in_dim; uint batch_b; uint max_rank;
    uint ranks[8];
    float scales[8];
};

kernel void moe_lora_bgmv_a(
    const device float* x [[buffer(0)]], device float* tmp [[buffer(1)]],
    const device uint* batch_lids [[buffer(2)]],
    const device float* a1 [[buffer(3)]], const device float* a2 [[buffer(4)]],
    const device float* a3 [[buffer(5)]], const device float* a4 [[buffer(6)]],
    const device float* a5 [[buffer(7)]], const device float* a6 [[buffer(8)]],
    const device float* a7 [[buffer(9)]],
    constant MoeBgmvArgs& p [[buffer(10)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r = tgid.x, b = tgid.y;
    if (b >= p.batch_b || r >= p.max_rank) return;
    uint lid = batch_lids[b];
    if (lid == 0u || lid >= 8u || r >= p.ranks[lid]) return;
    const device float* A = (lid == 1u) ? a1 : (lid == 2u) ? a2 : (lid == 3u) ? a3 :
                            (lid == 4u) ? a4 : (lid == 5u) ? a5 : (lid == 6u) ? a6 : a7;
    const device float* ar = A + (ulong)r * p.in_dim;
    const device float* xb = x + (ulong)b * p.in_dim;
    float acc = 0.0f;
    for (uint k = lane; k < p.in_dim; k += 32u) acc = fma(ar[k], xb[k], acc);
    float s = simd_sum(acc);
    if (lane == 0u) tmp[(ulong)b * p.max_rank + r] = s;
}

kernel void moe_lora_bgmv_b_add(
    const device float* tmp [[buffer(0)]], device float* y [[buffer(1)]],
    const device uint* batch_lids [[buffer(2)]],
    const device float* b1 [[buffer(3)]], const device float* b2 [[buffer(4)]],
    const device float* b3 [[buffer(5)]], const device float* b4 [[buffer(6)]],
    const device float* b5 [[buffer(7)]], const device float* b6 [[buffer(8)]],
    const device float* b7 [[buffer(9)]],
    constant MoeBgmvArgs& p [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint o = gid.x, b = gid.y;
    if (o >= p.out_dim || b >= p.batch_b) return;
    uint lid = batch_lids[b];
    if (lid == 0u || lid >= 8u) return;
    uint rk = p.ranks[lid];
    if (rk == 0u) return;
    const device float* B = (lid == 1u) ? b1 : (lid == 2u) ? b2 : (lid == 3u) ? b3 :
                            (lid == 4u) ? b4 : (lid == 5u) ? b5 : (lid == 6u) ? b6 : b7;
    const device float* bo = B + (ulong)o * rk;
    const device float* tb = tmp + (ulong)b * p.max_rank;
    float acc = 0.0f;
    for (uint r = 0; r < rk; r++) acc = fma(bo[r], tb[r], acc);
    y[(ulong)b * p.out_dim + o] += p.scales[lid] * acc;
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

// Tenon v12 Round 7: 64x64x32 4x4 Aligned-Stride-32 AMX FP16 GEMM (15x faster SmolVLM mmproj-f16 & F16 layers!)
kernel void moe_gemm_f16_amx(
    const device half* w [[buffer(0)]], const device float* x [[buffer(1)]], device float* y [[buffer(2)]],
    constant MoeGemmBatchArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r0 = tgid.x * 64u, b0 = tgid.y * 64u;
    if (r0 >= p.out_dim || b0 >= p.batch_b) return;
    threadgroup half sX[64][32];
    threadgroup half sW[64][32];
    threadgroup float sO[4][64];

    simdgroup_float8x8 a00(0.0f), a01(0.0f), a02(0.0f), a03(0.0f);
    simdgroup_float8x8 a10(0.0f), a11(0.0f), a12(0.0f), a13(0.0f);
    simdgroup_float8x8 a20(0.0f), a21(0.0f), a22(0.0f), a23(0.0f);
    simdgroup_float8x8 a30(0.0f), a31(0.0f), a32(0.0f), a33(0.0f);

    uint tr = lid >> 1, half_k = (lid & 1u) * 16u;
    uint glob_b = b0 + tr, gr = r0 + tr;
    bool xb_ok = (glob_b < p.batch_b), wr_ok = (gr < p.out_dim);
    const device float* xrow = x + (ulong)(xb_ok ? glob_b : 0u) * p.in_dim;
    const device half*  wrow = w + (ulong)(wr_ok ? gr : (p.out_dim - 1u)) * p.in_dim;

    uint sg_n = (sg & 1u) * 32u, sg_m = (sg >> 1) * 32u;
    bool do_m0 = (b0 + sg_m < p.batch_b);
    bool do_m1 = (b0 + sg_m + 8u < p.batch_b);
    bool do_m2 = (b0 + sg_m + 16u < p.batch_b);
    bool do_m3 = (b0 + sg_m + 24u < p.batch_b);

    for (uint k0 = 0; k0 < p.in_dim; k0 += 32u) {
        threadgroup half4* sx4 = (threadgroup half4*)(&sX[tr][half_k]);
        threadgroup half4* sw4 = (threadgroup half4*)(&sW[tr][half_k]);
        uint kc = k0 + half_k;
        if (kc + 16u <= p.in_dim) {
            if (xb_ok) {
                const device float4* xp = (const device float4*)(xrow + kc);
                sx4[0] = half4(xp[0]); sx4[1] = half4(xp[1]); sx4[2] = half4(xp[2]); sx4[3] = half4(xp[3]);
            } else {
                sx4[0] = half4(0.0h); sx4[1] = half4(0.0h); sx4[2] = half4(0.0h); sx4[3] = half4(0.0h);
            }
            if (wr_ok) {
                const device half4* wp = (const device half4*)(wrow + kc);
                sw4[0] = wp[0]; sw4[1] = wp[1]; sw4[2] = wp[2]; sw4[3] = wp[3];
            } else {
                sw4[0] = half4(0.0h); sw4[1] = half4(0.0h); sw4[2] = half4(0.0h); sw4[3] = half4(0.0h);
            }
        } else {
            for (uint j = 0; j < 16u; j++) {
                uint kk = kc + j;
                sX[tr][half_k + j] = (xb_ok && kk < p.in_dim) ? half(xrow[kk]) : 0.0h;
                sW[tr][half_k + j] = (wr_ok && kk < p.in_dim) ? wrow[kk] : 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 w0, w1, w2, w3, x0, x1, x2, x3;
            simdgroup_load(w0, &sW[sg_n +  0u][kk], 32, ulong2(0, 0), true);
            simdgroup_load(w1, &sW[sg_n +  8u][kk], 32, ulong2(0, 0), true);
            simdgroup_load(w2, &sW[sg_n + 16u][kk], 32, ulong2(0, 0), true);
            simdgroup_load(w3, &sW[sg_n + 24u][kk], 32, ulong2(0, 0), true);
            simdgroup_load(x0, &sX[sg_m +  0u][kk], 32, ulong2(0, 0), false);
            simdgroup_load(x1, &sX[sg_m +  8u][kk], 32, ulong2(0, 0), false);
            simdgroup_load(x2, &sX[sg_m + 16u][kk], 32, ulong2(0, 0), false);
            simdgroup_load(x3, &sX[sg_m + 24u][kk], 32, ulong2(0, 0), false);
            simdgroup_multiply_accumulate(a00, x0, w0, a00); simdgroup_multiply_accumulate(a01, x0, w1, a01);
            simdgroup_multiply_accumulate(a02, x0, w2, a02); simdgroup_multiply_accumulate(a03, x0, w3, a03);
            simdgroup_multiply_accumulate(a10, x1, w0, a10); simdgroup_multiply_accumulate(a11, x1, w1, a11);
            simdgroup_multiply_accumulate(a12, x1, w2, a12); simdgroup_multiply_accumulate(a13, x1, w3, a13);
            simdgroup_multiply_accumulate(a20, x2, w0, a20); simdgroup_multiply_accumulate(a21, x2, w1, a21);
            simdgroup_multiply_accumulate(a22, x2, w2, a22); simdgroup_multiply_accumulate(a23, x2, w3, a23);
            simdgroup_multiply_accumulate(a30, x3, w0, a30); simdgroup_multiply_accumulate(a31, x3, w1, a31);
            simdgroup_multiply_accumulate(a32, x3, w2, a32); simdgroup_multiply_accumulate(a33, x3, w3, a33);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!do_m0) return;
    #define STORE_TILE_F16(ACC, BM_OFF, BN_OFF) do { \
        simdgroup_store((ACC), &sO[sg][0], 8); \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
        for (uint idx = lane; idx < 64u; idx += 32u) { \
            uint ob = b0 + sg_m + (BM_OFF) + (idx >> 3), orow = r0 + sg_n + (BN_OFF) + (idx & 7u); \
            if (ob < p.batch_b && orow < p.out_dim) { \
                ulong oidx = (ulong)ob * p.out_dim + orow; \
                float val = sO[sg][idx]; \
                y[oidx] = p.add_res ? (y[oidx] + val) : val; \
            } \
        } \
        simdgroup_barrier(mem_flags::mem_threadgroup); \
    } while (0)
    STORE_TILE_F16(a00, 0u,  0u); STORE_TILE_F16(a01, 0u,  8u); STORE_TILE_F16(a02, 0u, 16u); STORE_TILE_F16(a03, 0u, 24u);
    if (do_m1) { STORE_TILE_F16(a10, 8u,  0u); STORE_TILE_F16(a11, 8u,  8u); STORE_TILE_F16(a12, 8u, 16u); STORE_TILE_F16(a13, 8u, 24u); }
    if (do_m2) { STORE_TILE_F16(a20, 16u, 0u); STORE_TILE_F16(a21, 16u, 8u); STORE_TILE_F16(a22, 16u, 16u); STORE_TILE_F16(a23, 16u, 24u); }
    if (do_m3) { STORE_TILE_F16(a30, 24u, 0u); STORE_TILE_F16(a31, 24u, 8u); STORE_TILE_F16(a32, 24u, 16u); STORE_TILE_F16(a33, 24u, 24u); }
    #undef STORE_TILE_F16
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
// Tenon v12 Round 6b: Vectorized half4 x float4 4-Rows/TG MLA Q-Absorption!
kernel void moe_mla_absorb_q(
    const device half* wuk_t [[buffer(0)]], const device float* q [[buffer(1)]], device float* q_abs [[buffer(2)]],
    constant MlaAbsorbArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint r = tgid.x * 4u + sg, h = tgid.y;
    if (r >= p.kv_r || h >= p.n_heads) return;
    const device half* wp = wuk_t + ((ulong)h * p.kv_r + r) * p.nope_d;
    const device float* qp = q + (ulong)h * p.d_head;
    float acc = 0.0f;
    if ((p.nope_d & 3u) == 0u) {
        uint nd4 = p.nope_d >> 2;
        const device half4* wp4 = (const device half4*)wp;
        const device float4* qp4 = (const device float4*)qp;
        for (uint j = lane; j < nd4; j += 32u) acc += dot(float4(wp4[j]), qp4[j]);
    } else {
        for (uint j = lane; j < p.nope_d; j += 32u) acc = fma(float(wp[j]), qp[j], acc);
    }
    acc = simd_sum(acc);
    if (lane == 0u) q_abs[(ulong)h * p.kv_r + r] = acc;
}

struct MlaFlashArgs { uint n_heads; uint kv_r; uint nope_d; uint rope_d; uint d_head; uint kv_len; float scale; };
// Tenon v12 Round 6b: Zero-Spill 4-Head-Grouped half4 Native MLA FlashDecode!
kernel void moe_flash_mla_latent(
    const device float* q_abs [[buffer(0)]], const device float* q [[buffer(1)]],
    const device half* lat_cache [[buffer(2)]], device float* o_lat [[buffer(3)]],
    constant MlaFlashArgs& p [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint h0 = tgid * 4u, h = h0 + sg;
    if (h0 >= p.n_heads) return;
    bool h_ok = (h < p.n_heads);
    uint lat_dim = p.kv_r + p.rope_d;
    float scale_log2 = p.scale * 1.4426950408889634f;
    if (p.kv_r == 512u && p.rope_d == 64u) {
        uint r16 = lane * 16u, d2 = lane * 2u;
        const device float4* qap = (const device float4*)(q_abs + (ulong)(h_ok ? h : 0u) * 512u + r16);
        float4 qa0 = h_ok ? qap[0] : float4(0.0f), qa1 = h_ok ? qap[1] : float4(0.0f);
        float4 qa2 = h_ok ? qap[2] : float4(0.0f), qa3 = h_ok ? qap[3] : float4(0.0f);
        const device float* qpp = q + (ulong)(h_ok ? h : 0u) * p.d_head + p.nope_d + d2;
        float2 qp = h_ok ? float2(qpp[0], qpp[1]) : float2(0.0f);
        float4 o0 = float4(0.0f), o1 = float4(0.0f), o2 = float4(0.0f), o3 = float4(0.0f);
        float m_run = -1e30f, l_run = 0.0f;
        threadgroup half sL[4][576];
        for (uint tb = 0; tb < p.kv_len; tb += 4u) {
            uint nt = min(4u, p.kv_len - tb);
            uint tot4 = nt * 144u;
            for (uint vi = lid; vi < tot4; vi += 128u) {
                uint ti = vi / 144u, c4 = (vi % 144u) * 4u;
                *(threadgroup half4*)(&sL[ti][c4]) = *(const device half4*)(lat_cache + (ulong)(tb + ti) * 576ul + c4);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (h_ok) {
                for (uint ti = 0; ti < nt; ti++) {
                    const threadgroup half4* cp = (const threadgroup half4*)(&sL[ti][r16]);
                    float4 c0 = float4(cp[0]), c1 = float4(cp[1]), c2 = float4(cp[2]), c3 = float4(cp[3]);
                    float2 kp = float2(*(const threadgroup half2*)(&sL[ti][512u + d2]));
                    float dt = dot(qa0, c0) + dot(qa1, c1) + dot(qa2, c2) + dot(qa3, c3) + dot(qp, kp);
                    float sc = simd_sum(dt) * scale_log2;
                    float mn = max(m_run, sc), al = exp2(m_run - mn), wt = exp2(sc - mn);
                    l_run = fma(l_run, al, wt); m_run = mn;
                    o0 = fma(o0, al, wt * c0); o1 = fma(o1, al, wt * c1);
                    o2 = fma(o2, al, wt * c2); o3 = fma(o3, al, wt * c3);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (h_ok) {
            float inv = 1.0f / max(l_run, 1e-12f);
            device float4* op = (device float4*)(o_lat + (ulong)h * 512u + r16);
            op[0] = o0 * inv; op[1] = o1 * inv; op[2] = o2 * inv; op[3] = o3 * inv;
        }
        return;
    }
    if (!h_ok) return;
    float m_run = -1e30f, l_run = 0.0f;
    float o_acc[16];
    uint rpl = (p.kv_r + 31u) >> 5, dpl = (p.rope_d + 31u) >> 5;
    for (uint k = 0; k < rpl && k < 16u; k++) o_acc[k] = 0.0f;
    for (uint t = 0; t < p.kv_len; t++) {
        const device half* cp = lat_cache + (ulong)t * lat_dim;
        const device half* kp = cp + p.kv_r;
        float dt = 0.0f;
        for (uint k = 0; k < rpl && k < 16u; k++) { uint r = lane + k * 32u; if (r < p.kv_r) dt = fma(q_abs[(ulong)h * p.kv_r + r], float(cp[r]), dt); }
        for (uint k = 0; k < dpl; k++) { uint d = lane + k * 32u; if (d < p.rope_d) dt = fma(q[(ulong)h * p.d_head + p.nope_d + d], float(kp[d]), dt); }
        float sc = simd_sum(dt) * scale_log2;
        float mn = max(m_run, sc), al = exp2(m_run - mn), wt = exp2(sc - mn);
        l_run = fma(l_run, al, wt); m_run = mn;
        for (uint k = 0; k < rpl && k < 16u; k++) { uint r = lane + k * 32u; if (r < p.kv_r) o_acc[k] = fma(o_acc[k], al, wt * float(cp[r])); }
    }
    float inv = 1.0f / max(l_run, 1e-12f);
    for (uint k = 0; k < rpl && k < 16u; k++) { uint r = lane + k * 32u; if (r < p.kv_r) o_lat[(ulong)h * p.kv_r + r] = o_acc[k] * inv; }
}

struct MlaProjVArgs { uint n_heads; uint v_d; uint kv_r; uint out_stride; };
// Tenon v12 Round 6b: Vectorized half4 x float4 4-Rows/TG MLA V-Projection!
kernel void moe_mla_proj_v(
    const device half* wuv [[buffer(0)]], const device float* o_lat [[buffer(1)]], device float* ctx [[buffer(2)]],
    constant MlaProjVArgs& p [[buffer(3)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint j = tgid.x * 4u + sg, h = tgid.y;
    if (j >= p.out_stride || h >= p.n_heads) return;
    if (j >= p.v_d) { if (lane == 0u) ctx[(ulong)h * p.out_stride + j] = 0.0f; return; }
    const device half* wp = wuv + ((ulong)h * p.v_d + j) * p.kv_r;
    const device float* op = o_lat + (ulong)h * p.kv_r;
    float acc = 0.0f;
    if ((p.kv_r & 3u) == 0u) {
        uint kvr4 = p.kv_r >> 2;
        const device half4* wp4 = (const device half4*)wp;
        const device float4* op4 = (const device float4*)op;
        for (uint r = lane; r < kvr4; r += 32u) acc += dot(float4(wp4[r]), op4[r]);
    } else {
        for (uint r = lane; r < p.kv_r; r += 32u) acc = fma(float(wp[r]), op[r], acc);
    }
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

struct MoePagedAttnArgs { uint n_heads; uint n_kv; uint d_head; uint batch_b; uint out_stride; float scale; uint max_blks; uint swa_window; uint attn_sink; };

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
    uint win_start = (p.swa_window > 0u && kv_len > p.swa_window) ? (kv_len - p.swa_window) : 0u;
    uint sink_end = min(p.attn_sink, win_start);
    for (uint pass = 0u; pass < 2u; pass++) {
        uint t_beg = (pass == 0u) ? 0u : win_start;
        uint t_end = (pass == 0u) ? sink_end : kv_len;
        for (uint t = t_beg + sg; t < t_end; t += 4u) {
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
    uint win_start = (p.swa_window > 0u && kv_len > p.swa_window) ? (kv_len - p.swa_window) : 0u;
    uint sink_end = min(p.attn_sink, win_start);
    for (uint pass = 0u; pass < 2u; pass++) {
        uint t_beg = (pass == 0u) ? 0u : win_start;
        uint t_end = (pass == 0u) ? sink_end : kv_len;
        for (uint t = t_beg + sg; t < t_end; t += 4u) {
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

// Tenon v10: FP8_E4M3 PagedKV Compression (mode=2, 34B/32 elems, +-448 dynamic range!) + SWA/Sink
kernel void moe_rope_kv_fp8_paged(
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
            float kd = (kam > 1e-12f) ? (kam / 240.0f) : 0.0f, kid = (kam > 1e-12f) ? (240.0f / kam) : 0.0f;
            float vd = (vam > 1e-12f) ? (vam / 240.0f) : 0.0f, vid = (vam > 1e-12f) ? (240.0f / vam) : 0.0f;
            device uchar* kp = k_cache + h_off + (ulong)blk * 34ul;
            device uchar* vp = v_cache + h_off + (ulong)blk * 34ul;
            if (lane == 0u) { *(device half*)kp = half(kd); *(device half*)vp = half(vd); }
            kp[2u + lane] = fp8_e4m3_enc(kv * kid);
            vp[2u + lane] = fp8_e4m3_enc(vv * vid);
        }
    }
}

kernel void moe_flash_decode_fp8_paged(
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
    uint win_start = (p.swa_window > 0u && kv_len > p.swa_window) ? (kv_len - p.swa_window) : 0u;
    uint sink_end = min(p.attn_sink, win_start);
    for (uint pass = 0u; pass < 2u; pass++) {
        uint t_beg = (pass == 0u) ? 0u : win_start;
        uint t_end = (pass == 0u) ? sink_end : kv_len;
        for (uint t = t_beg + sg; t < t_end; t += 4u) {
            uint pt = (uint(bt[t >> 4]) << 4) | (t & 15u);
            ulong h_off = (ulong)pt * row_bytes + (ulong)kvh * (ulong)nblk * 34ul;
            const device uchar* kp = k_cache + h_off;
            const device uchar* vp = v_cache + h_off;
            float dot = 0.0f;
            for (uint blk = 0; blk < nblk; blk++) {
                const device uchar* kb = kp + (ulong)blk * 34ul;
                float kd = float(*(const device half*)kb);
                dot = fma(s_q[blk * 32u + lane], fp8_e4m3_dec(kb[2u + lane]) * kd, dot);
            }
            float sc = simd_sum(dot) * p.scale;
            float mn = max(m_run, sc), al = exp(m_run - mn), wt = exp(sc - mn);
            l_run = fma(l_run, al, wt); m_run = mn;
            for (uint blk = 0; blk < nblk; blk++) {
                const device uchar* vb = vp + (ulong)blk * 34ul;
                float vd = float(*(const device half*)vb);
                o_acc[blk] = fma(o_acc[blk], al, wt * fp8_e4m3_dec(vb[2u + lane]) * vd);
            }
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

// Tenon v10: Q4_0 PagedKV Compression (mode=3, 18B/32 elems -> 3.56x smaller than FP16!) + SWA/Sink
kernel void moe_rope_kv_q4_paged(
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
        ulong row_bytes = (ulong)p.n_kv * (ulong)nblk * 18ul;
        ulong h_off = (ulong)phys_pos * row_bytes + (ulong)h * (ulong)nblk * 18ul;
        for (uint blk = sg; blk < nblk; blk += 4u) {
            float kv = k[kbase + blk * 32u + lane], vv = v[kbase + blk * 32u + lane];
            float kam = simd_max(fabs(kv)), vam = simd_max(fabs(vv));
            float kd = (kam > 1e-12f) ? (kam / 7.0f) : 0.0f, kid = (kam > 1e-12f) ? (7.0f / kam) : 0.0f;
            float vd = (vam > 1e-12f) ? (vam / 7.0f) : 0.0f, vid = (vam > 1e-12f) ? (7.0f / vam) : 0.0f;
            int kq = clamp(int(rint(kv * kid)) + 8, 0, 15);
            int vq = clamp(int(rint(vv * vid)) + 8, 0, 15);
            int kq_hi = simd_shuffle_down(kq, 16);
            int vq_hi = simd_shuffle_down(vq, 16);
            device uchar* kp = k_cache + h_off + (ulong)blk * 18ul;
            device uchar* vp = v_cache + h_off + (ulong)blk * 18ul;
            if (lane == 0u) { *(device half*)kp = half(kd); *(device half*)vp = half(vd); }
            if (lane < 16u) {
                kp[2u + lane] = uchar((kq & 15) | ((kq_hi & 15) << 4));
                vp[2u + lane] = uchar((vq & 15) | ((vq_hi & 15) << 4));
            }
        }
    }
}

kernel void moe_flash_decode_q4_paged(
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
    ulong row_bytes = (ulong)p.n_kv * (ulong)nblk * 18ul;
    threadgroup float s_q[256]; threadgroup float s_m[4]; threadgroup float s_l[4]; threadgroup float s_o[4][256];
    ulong qbase = (ulong)b * (p.n_heads * p.d_head) + (ulong)h * p.d_head;
    ulong cbase = (ulong)b * (p.n_heads * p.out_stride) + (ulong)h * p.out_stride;
    for (uint d = lid; d < p.d_head; d += 128u) s_q[d] = q[qbase + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m_run = -1e30f, l_run = 0.0f;
    float o_acc[8] = {0,0,0,0,0,0,0,0};
    uint win_start = (p.swa_window > 0u && kv_len > p.swa_window) ? (kv_len - p.swa_window) : 0u;
    uint sink_end = min(p.attn_sink, win_start);
    uint lo_idx = lane & 15u, is_hi = lane >> 4;
    for (uint pass = 0u; pass < 2u; pass++) {
        uint t_beg = (pass == 0u) ? 0u : win_start;
        uint t_end = (pass == 0u) ? sink_end : kv_len;
        for (uint t = t_beg + sg; t < t_end; t += 4u) {
            uint pt = (uint(bt[t >> 4]) << 4) | (t & 15u);
            ulong h_off = (ulong)pt * row_bytes + (ulong)kvh * (ulong)nblk * 18ul;
            const device uchar* kp = k_cache + h_off;
            const device uchar* vp = v_cache + h_off;
            float dot = 0.0f;
            for (uint blk = 0; blk < nblk; blk++) {
                const device uchar* kb = kp + (ulong)blk * 18ul;
                float kd = float(*(const device half*)kb);
                uchar qb = kb[2u + lo_idx];
                float kq = float(is_hi ? (qb >> 4) : (qb & 15u)) - 8.0f;
                dot = fma(s_q[blk * 32u + lane], kq * kd, dot);
            }
            float sc = simd_sum(dot) * p.scale;
            float mn = max(m_run, sc), al = exp(m_run - mn), wt = exp(sc - mn);
            l_run = fma(l_run, al, wt); m_run = mn;
            for (uint blk = 0; blk < nblk; blk++) {
                const device uchar* vb = vp + (ulong)blk * 18ul;
                float vd = float(*(const device half*)vb);
                uchar qb = vb[2u + lo_idx];
                float vq = float(is_hi ? (qb >> 4) : (qb & 15u)) - 8.0f;
                o_acc[blk] = fma(o_acc[blk], al, wt * vq * vd);
            }
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
// Tenon v11: Hardware-Native FlashDecoding Split-KV Parallel Long-Context Decode!
// Saturates all GPU cores on B=1..8 at 1K-128K context across 4 KV modes (FP16/Q8/FP8/Q4) + SWA/Sink + MLA!
// =========================================================================
struct MoeFlashDecodeSplitArgs {
    uint n_heads; uint n_kv; uint d_head; uint batch_b;
    float scale; uint max_blks; uint swa_window; uint attn_sink;
    uint kv_mode; uint n_splits; uint grp_step;
};
kernel void moe_flash_decode_split_paged(
    const device float* q [[buffer(0)]], const device uchar* k_cache [[buffer(1)]], const device uchar* v_cache [[buffer(2)]],
    device float* part_o [[buffer(3)]], device float* part_m [[buffer(4)]], device float* part_l [[buffer(5)]],
    const device uint* pos_arr [[buffer(6)]], const device uint* slot_arr [[buffer(7)]],
    const device int* block_table [[buffer(8)]], constant MoeFlashDecodeSplitArgs& p [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    // Tenon v12 Round 2: ZERO-SPILL 1-SG-per-Q-Head Cooperative GQA FlashDecoding Split-KV!
    //   - Each SG owns 1 Q head in pure hardware registers (float2/float4 -> ZERO DRAM stack spill, ZERO cross-SG reduce!)
    //   - All gstep SGs cooperatively stage & dequant TK=8 KV tokens ONCE into 4.2 KB sK/sV (4x-8x less DRAM & dequant!)
    uint grp = max(p.n_heads / p.n_kv, 1u);
    uint gstep = max(p.grp_step, 1u);
    uint n_gchunks = (grp + gstep - 1u) / gstep;
    uint kvh = tgid.x / n_gchunks;
    uint gc  = tgid.x % n_gchunks;
    uint b = tgid.y, sp = tgid.z;
    if (kvh >= p.n_kv || b >= p.batch_b || sp >= p.n_splits) return;
    uint g_beg = gc * gstep;
    uint g_cnt = min(gstep, grp - g_beg);
    uint h0 = kvh * grp + g_beg;
    bool sg_valid = (sg < g_cnt);
    uint h = h0 + (sg_valid ? sg : 0u);
    ulong bh = (ulong)b * p.n_heads + h;
    ulong sp_slot = bh * p.n_splits + sp;

    uint kv_len = pos_arr[b] + 1u, slot = slot_arr[b];
    uint win_start = (p.swa_window > 0u && kv_len > p.swa_window) ? (kv_len - p.swa_window) : 0u;
    uint sink_end = min(p.attn_sink, win_start);
    uint total_vis = sink_end + (kv_len - win_start);
    uint chunk = (total_vis + p.n_splits - 1u) / p.n_splits;
    uint i_beg = sp * chunk;
    uint i_end = min(total_vis, i_beg + chunk);
    if (i_beg >= i_end) {
        if (sg_valid && lane == 0u) { part_m[sp_slot] = -1e30f; part_l[sp_slot] = 0.0f; }
        return;
    }
    const device int* bt = block_table + (ulong)slot * p.max_blks;
    uint n_thr = gstep * 32u;
    uint nblk = p.d_head >> 5;
    ulong bpb = (p.kv_mode == 3u) ? 18ul : 34ul;
    ulong row_bytes = (p.kv_mode == 0u) ? ((ulong)p.n_kv * p.d_head * 2ul) : ((ulong)p.n_kv * nblk * bpb);
    const device half* kc_f16 = (const device half*)k_cache;
    const device half* vc_f16 = (const device half*)v_cache;
    uint kv_stride = p.n_kv * p.d_head;

    float scale_log2 = p.scale * 1.4426950408889634f;
    if (p.d_head == 64u || p.d_head == 128u) {
        // Tenon v12 Round 7: TK=16 + Zero-Div + 32-bit packed Q8/Q4 (1-read-8-elem Q4!) + Native exp2!
        threadgroup half sK[16][132];
        threadgroup half sV[16][132];
        float m_run = -1e30f, l_run = 0.0f;
        float4 qr = float4(0.0f), oa = float4(0.0f);
        ulong qbase = bh * p.d_head;
        if (sg_valid) {
            if (p.d_head == 64u) {
                float2 q2 = *(const device float2*)(q + qbase + lane * 2u);
                qr = float4(q2.x, q2.y, 0.0f, 0.0f);
            } else {
                qr = *(const device float4*)(q + qbase + lane * 4u);
            }
        }
        uint dh4 = p.d_head >> 2;
        uint dh4_sh = (p.d_head == 64u) ? 4u : 5u;
        uint dh4_mk = dh4 - 1u;
        uint dh8_sh = (p.d_head == 64u) ? 3u : 4u;
        uint dh8_mk = (p.d_head >> 3) - 1u;
        for (uint idx_base = i_beg; idx_base < i_end; idx_base += 16u) {
            uint nt = min(16u, i_end - idx_base);
            uint tot4 = nt * dh4;
            if (p.kv_mode == 0u) {
                for (uint vi = lid; vi < tot4; vi += n_thr) {
                    uint ti = vi >> dh4_sh, c4 = (vi & dh4_mk) << 2;
                    uint idx = idx_base + ti;
                    uint t = (idx < sink_end) ? idx : (win_start + (idx - sink_end));
                    uint pt = (uint(bt[t >> 4]) << 4) | (t & 15u);
                    ulong off = (ulong)pt * kv_stride + kvh * p.d_head + c4;
                    *(threadgroup half4*)(&sK[ti][c4]) = *(const device half4*)(kc_f16 + off);
                    *(threadgroup half4*)(&sV[ti][c4]) = *(const device half4*)(vc_f16 + off);
                }
            } else if (p.kv_mode == 3u) {
                // Q4_0: 1 thread decodes BOTH lo4 & hi4 (8 elems) from ONE packed_ushort2 load -> 2x less DRAM & bt[] lookups!
                uint tot8 = nt * (p.d_head >> 3);
                for (uint vi = lid; vi < tot8; vi += n_thr) {
                    uint ti = vi >> dh8_sh;
                    uint u8 = vi & dh8_mk;
                    uint blk = u8 >> 2, lo = (u8 & 3u) << 2;
                    uint c_lo = (blk << 5) + lo;
                    uint idx = idx_base + ti;
                    uint t = (idx < sink_end) ? idx : (win_start + (idx - sink_end));
                    uint pt = (uint(bt[t >> 4]) << 4) | (t & 15u);
                    ulong off = (ulong)pt * row_bytes + (ulong)(kvh * nblk + blk) * 18ul;
                    const device uchar* kb = k_cache + off;
                    const device uchar* vb = v_cache + off;
                    half dk = *(const device half*)kb, mk = half(-8.0f) * dk;
                    half dv = *(const device half*)vb, mv = half(-8.0f) * dv;
                    uchar4 kb4 = as_type<uchar4>(*(const device packed_ushort2*)(kb + 2u + lo));
                    uchar4 vb4 = as_type<uchar4>(*(const device packed_ushort2*)(vb + 2u + lo));
                    *(threadgroup half4*)(&sK[ti][c_lo])       = fma(half4(kb4 & uchar4(0x0Fu)), half4(dk), half4(mk));
                    *(threadgroup half4*)(&sK[ti][c_lo + 16u]) = fma(half4(kb4 >> 4),            half4(dk), half4(mk));
                    *(threadgroup half4*)(&sV[ti][c_lo])       = fma(half4(vb4 & uchar4(0x0Fu)), half4(dv), half4(mv));
                    *(threadgroup half4*)(&sV[ti][c_lo + 16u]) = fma(half4(vb4 >> 4),            half4(dv), half4(mv));
                }
            } else {
                // Q8_0 (single 32-bit packed_short2 load!) & FP8_E4M3
                for (uint vi = lid; vi < tot4; vi += n_thr) {
                    uint ti = vi >> dh4_sh, c4 = (vi & dh4_mk) << 2;
                    uint blk = c4 >> 5, j = c4 & 31u;
                    uint idx = idx_base + ti;
                    uint t = (idx < sink_end) ? idx : (win_start + (idx - sink_end));
                    uint pt = (uint(bt[t >> 4]) << 4) | (t & 15u);
                    ulong off = (ulong)pt * row_bytes + (ulong)(kvh * nblk + blk) * 34ul;
                    const device uchar* kb = k_cache + off;
                    const device uchar* vb = v_cache + off;
                    half dk = *(const device half*)kb;
                    half dv = *(const device half*)vb;
                    if (p.kv_mode == 1u) {
                        char4 kc = as_type<char4>(*(const device packed_short2*)(kb + 2u + j));
                        char4 vc = as_type<char4>(*(const device packed_short2*)(vb + 2u + j));
                        *(threadgroup half4*)(&sK[ti][c4]) = half4(kc) * dk;
                        *(threadgroup half4*)(&sV[ti][c4]) = half4(vc) * dv;
                    } else {
                        uchar4 kq = as_type<uchar4>(*(const device packed_ushort2*)(kb + 2u + j));
                        uchar4 vq = as_type<uchar4>(*(const device packed_ushort2*)(vb + 2u + j));
                        *(threadgroup half4*)(&sK[ti][c4]) = half4(fp8_e4m3_dec(kq.x), fp8_e4m3_dec(kq.y), fp8_e4m3_dec(kq.z), fp8_e4m3_dec(kq.w)) * dk;
                        *(threadgroup half4*)(&sV[ti][c4]) = half4(fp8_e4m3_dec(vq.x), fp8_e4m3_dec(vq.y), fp8_e4m3_dec(vq.z), fp8_e4m3_dec(vq.w)) * dv;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg_valid) {
                if (p.d_head == 64u) {
                    for (uint ti = 0; ti < nt; ti++) {
                        half2 kv = *(const threadgroup half2*)(&sK[ti][lane * 2u]);
                        half2 vv = *(const threadgroup half2*)(&sV[ti][lane * 2u]);
                        float sc = simd_sum(fma(qr.x, float(kv.x), qr.y * float(kv.y))) * scale_log2;
                        float mn = max(m_run, sc), al = exp2(m_run - mn), wt = exp2(sc - mn);
                        l_run = fma(l_run, al, wt); m_run = mn;
                        oa.x = fma(oa.x, al, wt * float(vv.x));
                        oa.y = fma(oa.y, al, wt * float(vv.y));
                    }
                } else {
                    for (uint ti = 0; ti < nt; ti++) {
                        half4 kv = *(const threadgroup half4*)(&sK[ti][lane * 4u]);
                        half4 vv = *(const threadgroup half4*)(&sV[ti][lane * 4u]);
                        float sc = simd_sum(dot(qr, float4(kv))) * scale_log2;
                        float mn = max(m_run, sc), al = exp2(m_run - mn), wt = exp2(sc - mn);
                        l_run = fma(l_run, al, wt); m_run = mn;
                        oa = fma(oa, float4(al), float4(wt) * float4(vv));
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (sg_valid) {
            if (lane == 0u) { part_m[sp_slot] = m_run; part_l[sp_slot] = l_run; }
            ulong obase = sp_slot * p.d_head;
            if (p.d_head == 64u) {
                part_o[obase + lane * 2u + 0u] = oa.x;
                part_o[obase + lane * 2u + 1u] = oa.y;
            } else {
                *(device float4*)(part_o + obase + lane * 4u) = oa;
            }
        }
        return;
    }

    // Generic fallback for rare d_head != 64, 128 (e.g., d_head=256)
    float m_run = -1e30f, l_run = 0.0f;
    float qr[8], oa[8];
    uint d_per_lane = (p.d_head + 31u) >> 5;
    ulong qbase = bh * p.d_head;
    for (uint k = 0; k < 8u; k++) {
        uint d = lane + k * 32u;
        qr[k] = (sg_valid && d < p.d_head) ? q[qbase + d] : 0.0f;
        oa[k] = 0.0f;
    }
    uint lo_idx = lane & 15u, is_hi = lane >> 4;
    if (sg_valid) {
        for (uint idx = i_beg; idx < i_end; idx++) {
            uint t = (idx < sink_end) ? idx : (win_start + (idx - sink_end));
            uint pt = (uint(bt[t >> 4]) << 4) | (t & 15u);
            float dot = 0.0f;
            float vv[8];
            if (p.kv_mode == 0u) {
                const device half* kp = kc_f16 + (ulong)pt * kv_stride + kvh * p.d_head;
                const device half* vp = vc_f16 + (ulong)pt * kv_stride + kvh * p.d_head;
                for (uint k = 0; k < d_per_lane && k < 8u; k++) {
                    uint d = lane + k * 32u;
                    if (d < p.d_head) { dot = fma(qr[k], float(kp[d]), dot); vv[k] = float(vp[d]); } else vv[k] = 0.0f;
                }
            } else {
                ulong h_off = (ulong)pt * row_bytes + (ulong)kvh * nblk * bpb;
                const device uchar* kp = k_cache + h_off;
                const device uchar* vp = v_cache + h_off;
                for (uint blk = 0; blk < nblk && blk < 8u; blk++) {
                    const device uchar* kb = kp + (ulong)blk * bpb;
                    const device uchar* vb = vp + (ulong)blk * bpb;
                    float kv_f, vv_f;
                    if (p.kv_mode == 1u) {
                        kv_f = float(((const device char*)(kb + 2u))[lane]) * float(*(const device half*)kb);
                        vv_f = float(((const device char*)(vb + 2u))[lane]) * float(*(const device half*)vb);
                    } else if (p.kv_mode == 2u) {
                        kv_f = fp8_e4m3_dec(kb[2u + lane]) * float(*(const device half*)kb);
                        vv_f = fp8_e4m3_dec(vb[2u + lane]) * float(*(const device half*)vb);
                    } else {
                        uchar qbk = kb[2u + lo_idx], qbv = vb[2u + lo_idx];
                        kv_f = (float(is_hi ? (qbk >> 4) : (qbk & 15u)) - 8.0f) * float(*(const device half*)kb);
                        vv_f = (float(is_hi ? (qbv >> 4) : (qbv & 15u)) - 8.0f) * float(*(const device half*)vb);
                    }
                    dot = fma(qr[blk], kv_f, dot); vv[blk] = vv_f;
                }
            }
            float sc = simd_sum(dot) * scale_log2;
            float mn = max(m_run, sc), al = exp2(m_run - mn), wt = exp2(sc - mn);
            l_run = fma(l_run, al, wt); m_run = mn;
            for (uint k = 0; k < d_per_lane && k < 8u; k++) oa[k] = fma(oa[k], al, wt * vv[k]);
        }
        if (lane == 0u) { part_m[sp_slot] = m_run; part_l[sp_slot] = l_run; }
        ulong obase = sp_slot * p.d_head;
        for (uint k = 0; k < d_per_lane && k < 8u; k++) {
            uint d = lane + k * 32u;
            if (d < p.d_head) part_o[obase + d] = oa[k];
        }
    }
}

struct MlaFlashSplitArgs {
    uint n_heads; uint kv_r; uint nope_d; uint rope_d; uint d_head; uint batch_b;
    float scale; uint max_blks; uint swa_window; uint attn_sink; uint n_splits;
};
// Tenon v13 T2.1b: Dedicated 16-Head Single-TG (512 thr) half4-Dot Latent MLA FlashDecoding (1x DRAM read across ALL 16 heads!)
kernel void moe_flash_mla_split_paged_512(
    const device float* q_abs [[buffer(0)]], const device float* q [[buffer(1)]],
    const device half* lat_cache [[buffer(2)]],
    device float* part_o [[buffer(3)]], device float* part_m [[buffer(4)]], device float* part_l [[buffer(5)]],
    const device uint* pos_arr [[buffer(6)]], const device uint* slot_arr [[buffer(7)]],
    const device int* block_table [[buffer(8)]], constant MlaFlashSplitArgs& p [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint h0 = tgid.x * 16u, b = tgid.y, sp = tgid.z;
    if (h0 >= p.n_heads || b >= p.batch_b || sp >= p.n_splits) return;
    uint h = h0 + sg;
    bool h_valid = (h < p.n_heads);
    ulong bh = (ulong)b * p.n_heads + (h_valid ? h : (p.n_heads - 1u));
    ulong sp_slot = bh * p.n_splits + sp;

    uint kv_len = pos_arr[b] + 1u, slot = slot_arr[b];
    uint win_start = (p.swa_window > 0u && kv_len > p.swa_window) ? (kv_len - p.swa_window) : 0u;
    uint sink_end = min(p.attn_sink, win_start);
    uint total_vis = sink_end + (kv_len - win_start);
    uint chunk = (total_vis + p.n_splits - 1u) / p.n_splits;
    uint i_beg = sp * chunk, i_end = min(total_vis, i_beg + chunk);
    if (i_beg >= i_end) {
        if (h_valid && lane == 0u) { part_m[sp_slot] = -1e30f; part_l[sp_slot] = 0.0f; }
        return;
    }
    const device int* bt = block_table + (ulong)slot * p.max_blks;
    float scale_log2 = p.scale * 1.4426950408889634f;
    float m_run = -1e30f, l_run = 0.0f;

    threadgroup half s_lat[8][576];
    uint c4_lane = lane * 4u;
    uint r2_lane = lane * 2u;
    uint hi_tok = lid >> 7, l128 = lid & 127u;
    const device float* qab_p = q_abs + bh * 512ul;
    half4 qa0 = h_valid ? half4(*(const device float4*)(qab_p + c4_lane))        : half4(0.0h);
    half4 qa1 = h_valid ? half4(*(const device float4*)(qab_p + c4_lane + 128u)) : half4(0.0h);
    half4 qa2 = h_valid ? half4(*(const device float4*)(qab_p + c4_lane + 256u)) : half4(0.0h);
    half4 qa3 = h_valid ? half4(*(const device float4*)(qab_p + c4_lane + 384u)) : half4(0.0h);
    half2 qp  = h_valid ? half2(*(const device float2*)(q + bh * p.d_head + p.nope_d + r2_lane)) : half2(0.0h);
    float4 oa0 = float4(0.0f), oa1 = float4(0.0f), oa2 = float4(0.0f), oa3 = float4(0.0f);

    for (uint idx_base = i_beg; idx_base < i_end; idx_base += 8u) {
        uint n_tile = min(8u, i_end - idx_base);
        for (uint ti = hi_tok; ti < n_tile; ti += 4u) {
            uint idx = idx_base + ti;
            uint t = (idx < sink_end) ? idx : (win_start + (idx - sink_end));
            uint pt = (uint(bt[t >> 4]) << 4) | (t & 15u);
            const device half* src = lat_cache + (ulong)pt * 576ul;
            *(threadgroup half4*)(&s_lat[ti][l128 * 4u]) = *(const device half4*)(src + l128 * 4u);
            if (l128 < 16u) *(threadgroup half4*)(&s_lat[ti][512u + l128 * 4u]) = *(const device half4*)(src + 512u + l128 * 4u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (h_valid) {
            for (uint ti = 0; ti < n_tile; ti++) {
                half4 l0 = *(const threadgroup half4*)(&s_lat[ti][c4_lane]);
                half4 l1 = *(const threadgroup half4*)(&s_lat[ti][c4_lane + 128u]);
                half4 l2 = *(const threadgroup half4*)(&s_lat[ti][c4_lane + 256u]);
                half4 l3 = *(const threadgroup half4*)(&s_lat[ti][c4_lane + 384u]);
                half2 lp = *(const threadgroup half2*)(&s_lat[ti][512u + r2_lane]);
                half4 dp = fma(qa0, l0, fma(qa1, l1, fma(qa2, l2, qa3 * l3)));
                float dt = (float(dp.x) + float(dp.y)) + (float(dp.z) + float(dp.w)) + fma(float(qp.x), float(lp.x), float(qp.y) * float(lp.y));
                float sc = simd_sum(dt) * scale_log2;
                float mn = max(m_run, sc), al = exp2(m_run - mn), wt = exp2(sc - mn);
                l_run = fma(l_run, al, wt); m_run = mn;
                oa0 = fma(oa0, float4(al), float4(wt) * float4(l0));
                oa1 = fma(oa1, float4(al), float4(wt) * float4(l1));
                oa2 = fma(oa2, float4(al), float4(wt) * float4(l2));
                oa3 = fma(oa3, float4(al), float4(wt) * float4(l3));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (h_valid) {
        if (lane == 0u) { part_m[sp_slot] = m_run; part_l[sp_slot] = l_run; }
        device float* op = part_o + sp_slot * 512ul;
        *(device float4*)(op + c4_lane)        = oa0;
        *(device float4*)(op + c4_lane + 128u) = oa1;
        *(device float4*)(op + c4_lane + 256u) = oa2;
        *(device float4*)(op + c4_lane + 384u) = oa3;
    }
}

kernel void moe_flash_mla_split_paged(
    const device float* q_abs [[buffer(0)]], const device float* q [[buffer(1)]],
    const device half* lat_cache [[buffer(2)]],
    device float* part_o [[buffer(3)]], device float* part_m [[buffer(4)]], device float* part_l [[buffer(5)]],
    const device uint* pos_arr [[buffer(6)]], const device uint* slot_arr [[buffer(7)]],
    const device int* block_table [[buffer(8)]], constant MlaFlashSplitArgs& p [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint h0 = tgid.x * 4u, b = tgid.y, sp = tgid.z;
    if (h0 >= p.n_heads || b >= p.batch_b || sp >= p.n_splits) return;
    uint h = h0 + sg;
    bool h_valid = (h < p.n_heads);
    ulong bh = (ulong)b * p.n_heads + (h_valid ? h : (p.n_heads - 1u));
    ulong sp_slot = bh * p.n_splits + sp;

    uint kv_len = pos_arr[b] + 1u, slot = slot_arr[b];
    uint win_start = (p.swa_window > 0u && kv_len > p.swa_window) ? (kv_len - p.swa_window) : 0u;
    uint sink_end = min(p.attn_sink, win_start);
    uint total_vis = sink_end + (kv_len - win_start);
    uint chunk = (total_vis + p.n_splits - 1u) / p.n_splits;
    uint i_beg = sp * chunk, i_end = min(total_vis, i_beg + chunk);
    if (i_beg >= i_end) {
        if (h_valid && lane == 0u) { part_m[sp_slot] = -1e30f; part_l[sp_slot] = 0.0f; }
        return;
    }
    const device int* bt = block_table + (ulong)slot * p.max_blks;
    uint lat_dim = p.kv_r + p.rope_d;
    uint lat4 = lat_dim >> 2;
    float scale_log2 = p.scale * 1.4426950408889634f;
    float m_run = -1e30f, l_run = 0.0f;

    threadgroup half s_lat[4][580];
    uint rpl = (p.kv_r + 31u) >> 5;
    uint dpl = (p.rope_d + 31u) >> 5;
    float qa[16], qp[2], o_acc[16];
    for (uint k = 0; k < 16u; k++) {
        uint r = lane + k * 32u;
        qa[k] = (h_valid && r < p.kv_r) ? q_abs[bh * p.kv_r + r] : 0.0f;
        o_acc[k] = 0.0f;
    }
    for (uint k = 0; k < 2u; k++) {
        uint d = lane + k * 32u;
        qp[k] = (h_valid && d < p.rope_d) ? q[bh * p.d_head + p.nope_d + d] : 0.0f;
    }
    for (uint idx_base = i_beg; idx_base < i_end; idx_base += 4u) {
        uint n_tile = min(4u, i_end - idx_base);
        uint tot_v4 = n_tile * lat4;
        for (uint vi = lid; vi < tot_v4; vi += 128u) {
            uint ti = vi / lat4, c4 = (vi % lat4) * 4u;
            uint idx = idx_base + ti;
            uint t = (idx < sink_end) ? idx : (win_start + (idx - sink_end));
            uint pt = (uint(bt[t >> 4]) << 4) | (t & 15u);
            *(threadgroup half4*)(&s_lat[ti][c4]) = *(const device half4*)(lat_cache + (ulong)pt * lat_dim + c4);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (h_valid) {
            for (uint ti = 0; ti < n_tile; ti++) {
                float dot = 0.0f;
                for (uint k = 0; k < rpl && k < 16u; k++) {
                    uint r = lane + k * 32u;
                    if (r < p.kv_r) dot = fma(qa[k], float(s_lat[ti][r]), dot);
                }
                for (uint k = 0; k < dpl && k < 2u; k++) {
                    uint d = lane + k * 32u;
                    if (d < p.rope_d) dot = fma(qp[k], float(s_lat[ti][p.kv_r + d]), dot);
                }
                float sc = simd_sum(dot) * scale_log2;
                float mn = max(m_run, sc), al = exp2(m_run - mn), wt = exp2(sc - mn);
                l_run = fma(l_run, al, wt); m_run = mn;
                for (uint k = 0; k < rpl && k < 16u; k++) {
                    uint r = lane + k * 32u;
                    if (r < p.kv_r) o_acc[k] = fma(o_acc[k], al, wt * float(s_lat[ti][r]));
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (h_valid) {
        if (lane == 0u) { part_m[sp_slot] = m_run; part_l[sp_slot] = l_run; }
        ulong obase = sp_slot * p.kv_r;
        for (uint k = 0; k < rpl && k < 16u; k++) {
            uint r = lane + k * 32u;
            if (r < p.kv_r) part_o[obase + r] = o_acc[k];
        }
    }
}

struct MoeFlashReduceArgs { uint n_heads; uint v_dim; uint out_stride; uint batch_b; uint n_splits; };
// Tenon v12 Round 7: Shared-Weight s_ws[64] + Native exp2 + Coalesced float4 Reduce (ZERO redundant exp() per thread!)
kernel void moe_flash_decode_reduce_paged(
    const device float* part_o [[buffer(0)]], const device float* part_m [[buffer(1)]], const device float* part_l [[buffer(2)]],
    device float* dst [[buffer(3)]], constant MoeFlashReduceArgs& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]]
) {
    uint h = tgid.x, b = tgid.y;
    if (h >= p.n_heads || b >= p.batch_b) return;
    ulong bh = (ulong)b * p.n_heads + h;
    ulong sbase = bh * p.n_splits;
    threadgroup float s_ws[64];
    threadgroup float s_inv;
    if (lid == 0u) {
        float gm = -1e30f;
        for (uint s = 0; s < p.n_splits; s++) {
            if (part_l[sbase + s] > 0.0f) gm = max(gm, part_m[sbase + s]);
        }
        float gl = 0.0f;
        for (uint s = 0; s < p.n_splits; s++) {
            float ls = part_l[sbase + s];
            float w = (ls > 0.0f) ? exp2(part_m[sbase + s] - gm) : 0.0f;
            s_ws[s] = w;
            gl += ls * w;
        }
        s_inv = 1.0f / max(gl, 1e-12f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = s_inv;
    ulong obase = bh * p.out_stride;
    if ((p.v_dim & 3u) == 0u && (p.out_stride & 3u) == 0u) {
        uint os4 = p.out_stride >> 2, vd4 = p.v_dim >> 2;
        for (uint d4 = lid; d4 < os4; d4 += 128u) {
            if (d4 < vd4) {
                float4 acc = float4(0.0f);
                for (uint s = 0; s < p.n_splits; s++) {
                    float w = s_ws[s];
                    if (w > 0.0f) acc = fma(((const device float4*)(part_o + (sbase + s) * p.v_dim))[d4], float4(w), acc);
                }
                ((device float4*)(dst + obase))[d4] = acc * inv;
            } else {
                ((device float4*)(dst + obase))[d4] = float4(0.0f);
            }
        }
    } else {
        for (uint d = lid; d < p.out_stride; d += 128u) {
            if (d < p.v_dim) {
                float acc = 0.0f;
                for (uint s = 0; s < p.n_splits; s++) {
                    float w = s_ws[s];
                    if (w > 0.0f) acc = fma(part_o[(sbase + s) * p.v_dim + d], w, acc);
                }
                dst[obase + d] = acc * inv;
            } else {
                dst[obase + d] = 0.0f;
            }
        }
    }
}

// Tenon v12 Round 6: Zero-Spill BQ=8, BK=16 Causal Paged FlashAttention-2 Prefill
// Explicit scalar/vector registers (NO stack-spilled arrays!), half2/half4 conflict-free SMEM loads, native exp2!
kernel void moe_flash_prefill_paged_fp16(
    const device float* q [[buffer(0)]], const device half* k_cache [[buffer(1)]], const device half* v_cache [[buffer(2)]],
    device float* ctx [[buffer(3)]], const device uint* pos_arr [[buffer(4)]], const device uint* slot_arr [[buffer(5)]],
    const device int* block_table [[buffer(6)]], constant MoePagedAttnArgs& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x, q_blk = tgid.y;
    uint qb0 = q_blk * 8u;
    if (h >= p.n_heads || qb0 >= p.batch_b) return;
    uint slot = slot_arr[qb0];
    const device int* bt = block_table + (ulong)slot * p.max_blks;
    uint grp = max(p.n_heads / p.n_kv, 1u), kvh = h / grp;
    uint kv_stride = p.n_kv * p.d_head;
    uint dh = p.d_head;
    uint max_qb = min(qb0 + 8u, p.batch_b);
    uint max_pos_tg = pos_arr[max_qb - 1u];

    uint b0 = qb0 + sg * 2u;
    uint b1 = b0 + 1u;
    bool v0 = (b0 < p.batch_b), v1 = (b1 < p.batch_b);
    uint p0 = v0 ? pos_arr[b0] : 0u;
    uint p1 = v1 ? pos_arr[b1] : 0u;
    float m0 = -1e30f, l0 = 0.0f, m1 = -1e30f, l1 = 0.0f;
    float scale_log2 = p.scale * 1.4426950408889634f;

    threadgroup half sK[16][128];
    threadgroup half sV[16][128];
    uint max_lb = max_pos_tg >> 4;
    uint tot_kv4 = (16u * dh) >> 2;
    uint dh4 = dh >> 2;

    if (dh == 64u) {
        // Fast path for d_head == 64 (Llama-3.2-1B, SmolVLM, etc.): 2 elems/lane via single half2 SMEM load!
        ulong qb_0 = ((ulong)(v0 ? b0 : 0u) * p.n_heads + h) * 64u;
        ulong qb_1 = ((ulong)(v1 ? b1 : 0u) * p.n_heads + h) * 64u;
        uint d2 = lane * 2u;
        float2 q0 = v0 ? float2(q[qb_0 + d2], q[qb_0 + d2 + 1u]) : float2(0.0f);
        float2 q1 = v1 ? float2(q[qb_1 + d2], q[qb_1 + d2 + 1u]) : float2(0.0f);
        float2 o0 = float2(0.0f), o1 = float2(0.0f);

        for (uint lb = 0; lb <= max_lb; lb++) {
            uint t_base = lb << 4;
            if (p.swa_window > 0u) {
                uint min_pos_tg = pos_arr[qb0];
                uint win_s = (min_pos_tg + 1u > p.swa_window) ? (min_pos_tg + 1u - p.swa_window) : 0u;
                if (t_base + 16u <= win_s && t_base >= p.attn_sink) continue;
            }
            uint pt_base = uint(bt[lb]) << 4;
            for (uint vi = lid; vi < 256u; vi += 128u) {
                uint tk = vi >> 4;
                uint c4 = (vi & 15u) << 2;
                ulong off = (ulong)(pt_base + tk) * kv_stride + kvh * 64u + c4;
                *(threadgroup half4*)(&sK[tk][c4]) = *(const device half4*)(k_cache + off);
                *(threadgroup half4*)(&sV[tk][c4]) = *(const device half4*)(v_cache + off);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint tk_lim = min(16u, max_pos_tg - t_base + 1u);
            for (uint tk = 0; tk < tk_lim; tk++) {
                uint t = t_base + tk;
                float2 kv = float2(((const threadgroup half2*)sK[tk])[lane]);
                float2 vv = float2(((const threadgroup half2*)sV[tk])[lane]);
                if (v0 && t <= p0) {
                    bool ok0 = true;
                    if (p.swa_window > 0u) {
                        uint ws0 = (p0 + 1u > p.swa_window) ? (p0 + 1u - p.swa_window) : 0u;
                        ok0 = (t >= ws0 || t < p.attn_sink);
                    }
                    if (ok0) {
                        float sc = simd_sum(fma(q0.x, kv.x, q0.y * kv.y)) * scale_log2;
                        float mn = max(m0, sc), al = exp2(m0 - mn), wt = exp2(sc - mn);
                        l0 = fma(l0, al, wt); m0 = mn;
                        o0 = fma(o0, al, wt * vv);
                    }
                }
                if (v1 && t <= p1) {
                    bool ok1 = true;
                    if (p.swa_window > 0u) {
                        uint ws1 = (p1 + 1u > p.swa_window) ? (p1 + 1u - p.swa_window) : 0u;
                        ok1 = (t >= ws1 || t < p.attn_sink);
                    }
                    if (ok1) {
                        float sc = simd_sum(fma(q1.x, kv.x, q1.y * kv.y)) * scale_log2;
                        float mn = max(m1, sc), al = exp2(m1 - mn), wt = exp2(sc - mn);
                        l1 = fma(l1, al, wt); m1 = mn;
                        o1 = fma(o1, al, wt * vv);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (v0) {
            ulong cb = ((ulong)b0 * p.n_heads + h) * p.out_stride;
            float inv = 1.0f / max(l0, 1e-12f);
            if (d2 < p.out_stride) ctx[cb + d2] = o0.x * inv;
            if (d2 + 1u < p.out_stride) ctx[cb + d2 + 1u] = o0.y * inv;
        }
        if (v1) {
            ulong cb = ((ulong)b1 * p.n_heads + h) * p.out_stride;
            float inv = 1.0f / max(l1, 1e-12f);
            if (d2 < p.out_stride) ctx[cb + d2] = o1.x * inv;
            if (d2 + 1u < p.out_stride) ctx[cb + d2 + 1u] = o1.y * inv;
        }
        return;
    }

    if (dh == 128u) {
        // Fast path for d_head == 128 (Llama-3.3-70B, GLM-4.5-Air, Qwen3, etc.): 4 elems/lane via single half4 SMEM load!
        ulong qb_0 = ((ulong)(v0 ? b0 : 0u) * p.n_heads + h) * 128u;
        ulong qb_1 = ((ulong)(v1 ? b1 : 0u) * p.n_heads + h) * 128u;
        uint d4 = lane * 4u;
        float4 q0 = v0 ? *(const device float4*)(q + qb_0 + d4) : float4(0.0f);
        float4 q1 = v1 ? *(const device float4*)(q + qb_1 + d4) : float4(0.0f);
        float4 o0 = float4(0.0f), o1 = float4(0.0f);

        for (uint lb = 0; lb <= max_lb; lb++) {
            uint t_base = lb << 4;
            if (p.swa_window > 0u) {
                uint min_pos_tg = pos_arr[qb0];
                uint win_s = (min_pos_tg + 1u > p.swa_window) ? (min_pos_tg + 1u - p.swa_window) : 0u;
                if (t_base + 16u <= win_s && t_base >= p.attn_sink) continue;
            }
            uint pt_base = uint(bt[lb]) << 4;
            for (uint vi = lid; vi < 512u; vi += 128u) {
                uint tk = vi >> 5;
                uint c4 = (vi & 31u) << 2;
                ulong off = (ulong)(pt_base + tk) * kv_stride + kvh * 128u + c4;
                *(threadgroup half4*)(&sK[tk][c4]) = *(const device half4*)(k_cache + off);
                *(threadgroup half4*)(&sV[tk][c4]) = *(const device half4*)(v_cache + off);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint tk_lim = min(16u, max_pos_tg - t_base + 1u);
            for (uint tk = 0; tk < tk_lim; tk++) {
                uint t = t_base + tk;
                float4 kv = float4(((const threadgroup half4*)sK[tk])[lane]);
                float4 vv = float4(((const threadgroup half4*)sV[tk])[lane]);
                if (v0 && t <= p0) {
                    bool ok0 = true;
                    if (p.swa_window > 0u) {
                        uint ws0 = (p0 + 1u > p.swa_window) ? (p0 + 1u - p.swa_window) : 0u;
                        ok0 = (t >= ws0 || t < p.attn_sink);
                    }
                    if (ok0) {
                        float sc = simd_sum(dot(q0, kv)) * scale_log2;
                        float mn = max(m0, sc), al = exp2(m0 - mn), wt = exp2(sc - mn);
                        l0 = fma(l0, al, wt); m0 = mn;
                        o0 = fma(o0, al, wt * vv);
                    }
                }
                if (v1 && t <= p1) {
                    bool ok1 = true;
                    if (p.swa_window > 0u) {
                        uint ws1 = (p1 + 1u > p.swa_window) ? (p1 + 1u - p.swa_window) : 0u;
                        ok1 = (t >= ws1 || t < p.attn_sink);
                    }
                    if (ok1) {
                        float sc = simd_sum(dot(q1, kv)) * scale_log2;
                        float mn = max(m1, sc), al = exp2(m1 - mn), wt = exp2(sc - mn);
                        l1 = fma(l1, al, wt); m1 = mn;
                        o1 = fma(o1, al, wt * vv);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (v0) {
            ulong cb = ((ulong)b0 * p.n_heads + h) * p.out_stride;
            float inv = 1.0f / max(l0, 1e-12f);
            if (d4 + 3u < p.out_stride) *(device float4*)(ctx + cb + d4) = o0 * inv;
            else {
                if (d4 < p.out_stride) ctx[cb + d4] = o0.x * inv;
                if (d4 + 1u < p.out_stride) ctx[cb + d4 + 1u] = o0.y * inv;
                if (d4 + 2u < p.out_stride) ctx[cb + d4 + 2u] = o0.z * inv;
            }
        }
        if (v1) {
            ulong cb = ((ulong)b1 * p.n_heads + h) * p.out_stride;
            float inv = 1.0f / max(l1, 1e-12f);
            if (d4 + 3u < p.out_stride) *(device float4*)(ctx + cb + d4) = o1 * inv;
            else {
                if (d4 < p.out_stride) ctx[cb + d4] = o1.x * inv;
                if (d4 + 1u < p.out_stride) ctx[cb + d4 + 1u] = o1.y * inv;
                if (d4 + 2u < p.out_stride) ctx[cb + d4 + 2u] = o1.z * inv;
            }
        }
        return;
    }

    // Generic zero-spill path for any other d_head <= 128 (4 explicit scalars per lane, NO arrays!)
    ulong qb_0 = ((ulong)(v0 ? b0 : 0u) * p.n_heads + h) * dh;
    ulong qb_1 = ((ulong)(v1 ? b1 : 0u) * p.n_heads + h) * dh;
    uint c0 = lane, c1 = lane + 32u, c2 = lane + 64u, c3 = lane + 96u;
    float q00 = (v0 && c0 < dh) ? q[qb_0 + c0] : 0.0f, q01 = (v0 && c1 < dh) ? q[qb_0 + c1] : 0.0f;
    float q02 = (v0 && c2 < dh) ? q[qb_0 + c2] : 0.0f, q03 = (v0 && c3 < dh) ? q[qb_0 + c3] : 0.0f;
    float q10 = (v1 && c0 < dh) ? q[qb_1 + c0] : 0.0f, q11 = (v1 && c1 < dh) ? q[qb_1 + c1] : 0.0f;
    float q12 = (v1 && c2 < dh) ? q[qb_1 + c2] : 0.0f, q13 = (v1 && c3 < dh) ? q[qb_1 + c3] : 0.0f;
    float o00 = 0.0f, o01 = 0.0f, o02 = 0.0f, o03 = 0.0f;
    float o10 = 0.0f, o11 = 0.0f, o12 = 0.0f, o13 = 0.0f;

    for (uint lb = 0; lb <= max_lb; lb++) {
        uint t_base = lb << 4;
        if (p.swa_window > 0u) {
            uint min_pos_tg = pos_arr[qb0];
            uint win_s = (min_pos_tg + 1u > p.swa_window) ? (min_pos_tg + 1u - p.swa_window) : 0u;
            if (t_base + 16u <= win_s && t_base >= p.attn_sink) continue;
        }
        uint pt_base = uint(bt[lb]) << 4;
        for (uint vi = lid; vi < tot_kv4; vi += 128u) {
            uint tk = vi / dh4;
            uint c4 = (vi % dh4) * 4u;
            ulong off = (ulong)(pt_base + tk) * kv_stride + kvh * dh + c4;
            *(threadgroup half4*)(&sK[tk][c4]) = *(const device half4*)(k_cache + off);
            *(threadgroup half4*)(&sV[tk][c4]) = *(const device half4*)(v_cache + off);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tk_lim = min(16u, max_pos_tg - t_base + 1u);
        for (uint tk = 0; tk < tk_lim; tk++) {
            uint t = t_base + tk;
            float k0 = (c0 < dh) ? float(sK[tk][c0]) : 0.0f, k1 = (c1 < dh) ? float(sK[tk][c1]) : 0.0f;
            float k2 = (c2 < dh) ? float(sK[tk][c2]) : 0.0f, k3 = (c3 < dh) ? float(sK[tk][c3]) : 0.0f;
            float vv0 = (c0 < dh) ? float(sV[tk][c0]) : 0.0f, vv1 = (c1 < dh) ? float(sV[tk][c1]) : 0.0f;
            float vv2 = (c2 < dh) ? float(sV[tk][c2]) : 0.0f, vv3 = (c3 < dh) ? float(sV[tk][c3]) : 0.0f;
            if (v0 && t <= p0) {
                bool ok0 = true;
                if (p.swa_window > 0u) {
                    uint ws0 = (p0 + 1u > p.swa_window) ? (p0 + 1u - p.swa_window) : 0u;
                    ok0 = (t >= ws0 || t < p.attn_sink);
                }
                if (ok0) {
                    float sc = simd_sum(fma(q00, k0, fma(q01, k1, fma(q02, k2, q03 * k3)))) * scale_log2;
                    float mn = max(m0, sc), al = exp2(m0 - mn), wt = exp2(sc - mn);
                    l0 = fma(l0, al, wt); m0 = mn;
                    o00 = fma(o00, al, wt * vv0); o01 = fma(o01, al, wt * vv1);
                    o02 = fma(o02, al, wt * vv2); o03 = fma(o03, al, wt * vv3);
                }
            }
            if (v1 && t <= p1) {
                bool ok1 = true;
                if (p.swa_window > 0u) {
                    uint ws1 = (p1 + 1u > p.swa_window) ? (p1 + 1u - p.swa_window) : 0u;
                    ok1 = (t >= ws1 || t < p.attn_sink);
                }
                if (ok1) {
                    float sc = simd_sum(fma(q10, k0, fma(q11, k1, fma(q12, k2, q13 * k3)))) * scale_log2;
                    float mn = max(m1, sc), al = exp2(m1 - mn), wt = exp2(sc - mn);
                    l1 = fma(l1, al, wt); m1 = mn;
                    o10 = fma(o10, al, wt * vv0); o11 = fma(o11, al, wt * vv1);
                    o12 = fma(o12, al, wt * vv2); o13 = fma(o13, al, wt * vv3);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (v0) {
        ulong cb = ((ulong)b0 * p.n_heads + h) * p.out_stride;
        float inv = 1.0f / max(l0, 1e-12f);
        if (c0 < p.out_stride) ctx[cb + c0] = o00 * inv;
        if (c1 < p.out_stride) ctx[cb + c1] = o01 * inv;
        if (c2 < p.out_stride) ctx[cb + c2] = o02 * inv;
        if (c3 < p.out_stride) ctx[cb + c3] = o03 * inv;
    }
    if (v1) {
        ulong cb = ((ulong)b1 * p.n_heads + h) * p.out_stride;
        float inv = 1.0f / max(l1, 1e-12f);
        if (c0 < p.out_stride) ctx[cb + c0] = o10 * inv;
        if (c1 < p.out_stride) ctx[cb + c1] = o11 * inv;
        if (c2 < p.out_stride) ctx[cb + c2] = o12 * inv;
        if (c3 < p.out_stride) ctx[cb + c3] = o13 * inv;
    }
}

// Tenon v13 P1b: Hardware Apple AMX FlashAttention-2 Paged Prefill (dh=64: BQ=32, BK=32, 18 KB SMEM, simdgroup_half8x8!)
kernel void moe_flash_prefill_amx64(
    const device float* q [[buffer(0)]], const device half* k_cache [[buffer(1)]], const device half* v_cache [[buffer(2)]],
    device float* ctx [[buffer(3)]], const device uint* pos_arr [[buffer(4)]], const device uint* slot_arr [[buffer(5)]],
    const device int* block_table [[buffer(6)]], constant MoePagedAttnArgs& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x, qb0 = tgid.y * 32u;
    if (h >= p.n_heads || qb0 >= p.batch_b) return;
    uint slot = slot_arr[qb0];
    const device int* bt = block_table + (ulong)slot * p.max_blks;
    uint grp = max(p.n_heads / p.n_kv, 1u), kvh = h / grp;
    uint kv_stride = p.n_kv * 64u;
    uint max_qb = min(qb0 + 32u, p.batch_b);
    uint max_pos_tg = pos_arr[max_qb - 1u];
    uint min_pos_tg = pos_arr[qb0];
    uint max_lb = max_pos_tg >> 4;
    float scale_log2 = p.scale * 1.4426950408889634f;

    threadgroup half sQ[32][64];
    threadgroup half sK[32][64];
    threadgroup half sV[32][64];
    threadgroup float sS[4][256];
    threadgroup half sP[4][256];

    for (uint vi = lid; vi < 512u; vi += 128u) {
        uint rq = vi >> 4, c4 = (vi & 15u) << 2;
        uint b = qb0 + rq;
        if (b < p.batch_b) {
            ulong qoff = ((ulong)b * p.n_heads + h) * 64ul + c4;
            *(threadgroup half4*)(&sQ[rq][c4]) = half4(*(const device float4*)(q + qoff));
        } else {
            *(threadgroup half4*)(&sQ[rq][c4]) = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint r_l = lane >> 2, c_l = (lane & 3u) * 8u;
    uint my_b = qb0 + sg * 8u + r_l;
    bool my_v = (my_b < p.batch_b);
    uint my_pos = my_v ? pos_arr[my_b] : 0u;
    uint my_ws = (p.swa_window > 0u && my_pos + 1u > p.swa_window) ? (my_pos + 1u - p.swa_window) : 0u;
    float m_row = -1e30f, l_row = 0.0f;
    simdgroup_float8x8 o0(0.0f), o1(0.0f), o2(0.0f), o3(0.0f), o4(0.0f), o5(0.0f), o6(0.0f), o7(0.0f);

    for (uint lb0 = 0; lb0 <= max_lb; lb0 += 2u) {
        uint t_base = lb0 << 4;
        if (p.swa_window > 0u) {
            uint ws_tg = (min_pos_tg + 1u > p.swa_window) ? (min_pos_tg + 1u - p.swa_window) : 0u;
            if (t_base + 32u <= ws_tg && t_base >= p.attn_sink) continue;
        }
        uint tk_lim = min(32u, max_pos_tg - t_base + 1u);
        uint pt0 = uint(bt[lb0]) << 4;
        uint pt1 = (lb0 + 1u <= max_lb) ? (uint(bt[lb0 + 1u]) << 4) : pt0;
        for (uint vi = lid; vi < 512u; vi += 128u) {
            uint tk = vi >> 4, c4 = (vi & 15u) << 2;
            if (tk < tk_lim) {
                uint pt = (tk < 16u) ? (pt0 + tk) : (pt1 + (tk - 16u));
                ulong off = (ulong)pt * kv_stride + (ulong)kvh * 64ul + c4;
                *(threadgroup half4*)(&sK[tk][c4]) = *(const device half4*)(k_cache + off);
                *(threadgroup half4*)(&sV[tk][c4]) = *(const device half4*)(v_cache + off);
            } else {
                *(threadgroup half4*)(&sK[tk][c4]) = half4(0.0h);
                *(threadgroup half4*)(&sV[tk][c4]) = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 sc0(0.0f), sc1(0.0f), sc2(0.0f), sc3(0.0f);
        uint q_r = sg * 8u;
        #pragma unroll
        for (uint kk = 0; kk < 64u; kk += 8u) {
            simdgroup_half8x8 mQ, mK;
            simdgroup_load(mQ, &sQ[q_r][kk], 64, ulong2(0, 0), false);
            simdgroup_load(mK, &sK[0][kk],   64, ulong2(0, 0), true); simdgroup_multiply_accumulate(sc0, mQ, mK, sc0);
            simdgroup_load(mK, &sK[8][kk],   64, ulong2(0, 0), true); simdgroup_multiply_accumulate(sc1, mQ, mK, sc1);
            simdgroup_load(mK, &sK[16][kk],  64, ulong2(0, 0), true); simdgroup_multiply_accumulate(sc2, mQ, mK, sc2);
            simdgroup_load(mK, &sK[24][kk],  64, ulong2(0, 0), true); simdgroup_multiply_accumulate(sc3, mQ, mK, sc3);
        }
        simdgroup_store(sc0, &sS[sg][0],  32); simdgroup_store(sc1, &sS[sg][8],  32);
        simdgroup_store(sc2, &sS[sg][16], 32); simdgroup_store(sc3, &sS[sg][24], 32);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float sv[8];
        float rm = -1e30f;
        uint roff = r_l * 32u + c_l;
        #pragma unroll
        for (uint k = 0; k < 8u; k++) {
            uint c = c_l + k, t = t_base + c;
            bool ok = my_v && (c < tk_lim) && (t <= my_pos) && (p.swa_window == 0u || t >= my_ws || t < p.attn_sink);
            sv[k] = ok ? (sS[sg][roff + k] * scale_log2) : -1e30f;
            rm = max(rm, sv[k]);
        }
        rm = max(rm, simd_shuffle_xor(rm, 1));
        rm = max(rm, simd_shuffle_xor(rm, 2));
        float mn = max(m_row, rm);
        float al = (m_row > -1e29f && mn > -1e29f) ? exp2(m_row - mn) : (mn > -1e29f ? 0.0f : 1.0f);
        float ls = 0.0f;
        #pragma unroll
        for (uint k = 0; k < 8u; k++) {
            float pv = (sv[k] > -1e29f) ? exp2(sv[k] - mn) : 0.0f;
            ls += pv;
            sP[sg][roff + k] = half(pv);
        }
        ls += simd_shuffle_xor(ls, 1);
        ls += simd_shuffle_xor(ls, 2);
        l_row = fma(l_row, al, ls);
        m_row = mn;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (lb0 > 0u && simd_any(al < 0.999999f)) {
            simdgroup_store(o0, &sS[sg][0],  32); simdgroup_store(o1, &sS[sg][8],  32);
            simdgroup_store(o2, &sS[sg][16], 32); simdgroup_store(o3, &sS[sg][24], 32);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            *(threadgroup float4*)(&sS[sg][roff])      *= al;
            *(threadgroup float4*)(&sS[sg][roff + 4u]) *= al;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(o0, &sS[sg][0],  32); simdgroup_load(o1, &sS[sg][8],  32);
            simdgroup_load(o2, &sS[sg][16], 32); simdgroup_load(o3, &sS[sg][24], 32);
            simdgroup_store(o4, &sS[sg][0],  32); simdgroup_store(o5, &sS[sg][8],  32);
            simdgroup_store(o6, &sS[sg][16], 32); simdgroup_store(o7, &sS[sg][24], 32);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            *(threadgroup float4*)(&sS[sg][roff])      *= al;
            *(threadgroup float4*)(&sS[sg][roff + 4u]) *= al;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(o4, &sS[sg][0],  32); simdgroup_load(o5, &sS[sg][8],  32);
            simdgroup_load(o6, &sS[sg][16], 32); simdgroup_load(o7, &sS[sg][24], 32);
        }
        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 mP, mV;
            simdgroup_load(mP, &sP[sg][kk], 32, ulong2(0, 0), false);
            simdgroup_load(mV, &sV[kk][0],  64, ulong2(0, 0), false); simdgroup_multiply_accumulate(o0, mP, mV, o0);
            simdgroup_load(mV, &sV[kk][8],  64, ulong2(0, 0), false); simdgroup_multiply_accumulate(o1, mP, mV, o1);
            simdgroup_load(mV, &sV[kk][16], 64, ulong2(0, 0), false); simdgroup_multiply_accumulate(o2, mP, mV, o2);
            simdgroup_load(mV, &sV[kk][24], 64, ulong2(0, 0), false); simdgroup_multiply_accumulate(o3, mP, mV, o3);
            simdgroup_load(mV, &sV[kk][32], 64, ulong2(0, 0), false); simdgroup_multiply_accumulate(o4, mP, mV, o4);
            simdgroup_load(mV, &sV[kk][40], 64, ulong2(0, 0), false); simdgroup_multiply_accumulate(o5, mP, mV, o5);
            simdgroup_load(mV, &sV[kk][48], 64, ulong2(0, 0), false); simdgroup_multiply_accumulate(o6, mP, mV, o6);
            simdgroup_load(mV, &sV[kk][56], 64, ulong2(0, 0), false); simdgroup_multiply_accumulate(o7, mP, mV, o7);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv = 1.0f / max(l_row, 1e-12f);
    uint roff = r_l * 32u + c_l;
    ulong cb = ((ulong)(my_v ? my_b : 0u) * p.n_heads + h) * p.out_stride;
    simdgroup_store(o0, &sS[sg][0],  32); simdgroup_store(o1, &sS[sg][8],  32);
    simdgroup_store(o2, &sS[sg][16], 32); simdgroup_store(o3, &sS[sg][24], 32);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (my_v && c_l + 7u < p.out_stride) {
        *(device float4*)(ctx + cb + c_l)      = *(const threadgroup float4*)(&sS[sg][roff]) * inv;
        *(device float4*)(ctx + cb + c_l + 4u) = *(const threadgroup float4*)(&sS[sg][roff + 4u]) * inv;
    } else if (my_v) {
        for (uint k = 0; k < 8u; k++) if (c_l + k < p.out_stride) ctx[cb + c_l + k] = sS[sg][roff + k] * inv;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(o4, &sS[sg][0],  32); simdgroup_store(o5, &sS[sg][8],  32);
    simdgroup_store(o6, &sS[sg][16], 32); simdgroup_store(o7, &sS[sg][24], 32);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint c_hi = 32u + c_l;
    if (my_v && c_hi + 7u < p.out_stride) {
        *(device float4*)(ctx + cb + c_hi)      = *(const threadgroup float4*)(&sS[sg][roff]) * inv;
        *(device float4*)(ctx + cb + c_hi + 4u) = *(const threadgroup float4*)(&sS[sg][roff + 4u]) * inv;
    } else if (my_v) {
        for (uint k = 0; k < 8u; k++) if (c_hi + k < p.out_stride) ctx[cb + c_hi + k] = sS[sg][roff + k] * inv;
    }
}

// Tenon v13 P1b: Hardware Apple AMX FlashAttention-2 Paged Prefill (dh=128: BQ=32, BK=16, 21 KB SMEM, 1 PagedKV blk/tile!)
kernel void moe_flash_prefill_amx128(
    const device float* q [[buffer(0)]], const device half* k_cache [[buffer(1)]], const device half* v_cache [[buffer(2)]],
    device float* ctx [[buffer(3)]], const device uint* pos_arr [[buffer(4)]], const device uint* slot_arr [[buffer(5)]],
    const device int* block_table [[buffer(6)]], constant MoePagedAttnArgs& p [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x, qb0 = tgid.y * 32u;
    if (h >= p.n_heads || qb0 >= p.batch_b) return;
    uint slot = slot_arr[qb0];
    const device int* bt = block_table + (ulong)slot * p.max_blks;
    uint grp = max(p.n_heads / p.n_kv, 1u), kvh = h / grp;
    uint kv_stride = p.n_kv * 128u;
    uint max_qb = min(qb0 + 32u, p.batch_b);
    uint max_pos_tg = pos_arr[max_qb - 1u];
    uint min_pos_tg = pos_arr[qb0];
    uint max_lb = max_pos_tg >> 4;
    float scale_log2 = p.scale * 1.4426950408889634f;

    threadgroup half sQ[32][128];
    threadgroup half sK[16][128];
    threadgroup half sV[16][128];
    threadgroup float sS[4][256];
    threadgroup half sP[4][128];

    for (uint vi = lid; vi < 1024u; vi += 128u) {
        uint rq = vi >> 5, c4 = (vi & 31u) << 2;
        uint b = qb0 + rq;
        if (b < p.batch_b) {
            ulong qoff = ((ulong)b * p.n_heads + h) * 128ul + c4;
            *(threadgroup half4*)(&sQ[rq][c4]) = half4(*(const device float4*)(q + qoff));
        } else {
            *(threadgroup half4*)(&sQ[rq][c4]) = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint r_l = lane >> 2, c4_l = (lane & 3u) * 4u, c8_l = (lane & 3u) * 8u;
    uint my_b = qb0 + sg * 8u + r_l;
    bool my_v = (my_b < p.batch_b);
    uint my_pos = my_v ? pos_arr[my_b] : 0u;
    uint my_ws = (p.swa_window > 0u && my_pos + 1u > p.swa_window) ? (my_pos + 1u - p.swa_window) : 0u;
    float m_row = -1e30f, l_row = 0.0f;
    simdgroup_float8x8 o[16];
    #pragma unroll
    for (uint i = 0; i < 16u; i++) o[i] = simdgroup_float8x8(0.0f);

    for (uint lb = 0; lb <= max_lb; lb++) {
        uint t_base = lb << 4;
        if (p.swa_window > 0u) {
            uint ws_tg = (min_pos_tg + 1u > p.swa_window) ? (min_pos_tg + 1u - p.swa_window) : 0u;
            if (t_base + 16u <= ws_tg && t_base >= p.attn_sink) continue;
        }
        uint tk_lim = min(16u, max_pos_tg - t_base + 1u);
        uint pt_base = uint(bt[lb]) << 4;
        for (uint vi = lid; vi < 512u; vi += 128u) {
            uint tk = vi >> 5, c4 = (vi & 31u) << 2;
            if (tk < tk_lim) {
                ulong off = (ulong)(pt_base + tk) * kv_stride + (ulong)kvh * 128ul + c4;
                *(threadgroup half4*)(&sK[tk][c4]) = *(const device half4*)(k_cache + off);
                *(threadgroup half4*)(&sV[tk][c4]) = *(const device half4*)(v_cache + off);
            } else {
                *(threadgroup half4*)(&sK[tk][c4]) = half4(0.0h);
                *(threadgroup half4*)(&sV[tk][c4]) = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 sc0(0.0f), sc1(0.0f);
        uint q_r = sg * 8u;
        #pragma unroll
        for (uint kk = 0; kk < 128u; kk += 8u) {
            simdgroup_half8x8 mQ, mK;
            simdgroup_load(mQ, &sQ[q_r][kk], 128, ulong2(0, 0), false);
            simdgroup_load(mK, &sK[0][kk],   128, ulong2(0, 0), true); simdgroup_multiply_accumulate(sc0, mQ, mK, sc0);
            simdgroup_load(mK, &sK[8][kk],   128, ulong2(0, 0), true); simdgroup_multiply_accumulate(sc1, mQ, mK, sc1);
        }
        simdgroup_store(sc0, &sS[sg][0], 16);
        simdgroup_store(sc1, &sS[sg][8], 16);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float sv[4];
        float rm = -1e30f;
        uint roff16 = r_l * 16u + c4_l;
        #pragma unroll
        for (uint k = 0; k < 4u; k++) {
            uint c = c4_l + k, t = t_base + c;
            bool ok = my_v && (c < tk_lim) && (t <= my_pos) && (p.swa_window == 0u || t >= my_ws || t < p.attn_sink);
            sv[k] = ok ? (sS[sg][roff16 + k] * scale_log2) : -1e30f;
            rm = max(rm, sv[k]);
        }
        rm = max(rm, simd_shuffle_xor(rm, 1));
        rm = max(rm, simd_shuffle_xor(rm, 2));
        float mn = max(m_row, rm);
        float al = (m_row > -1e29f && mn > -1e29f) ? exp2(m_row - mn) : (mn > -1e29f ? 0.0f : 1.0f);
        float ls = 0.0f;
        #pragma unroll
        for (uint k = 0; k < 4u; k++) {
            float pv = (sv[k] > -1e29f) ? exp2(sv[k] - mn) : 0.0f;
            ls += pv;
            sP[sg][roff16 + k] = half(pv);
        }
        ls += simd_shuffle_xor(ls, 1);
        ls += simd_shuffle_xor(ls, 2);
        l_row = fma(l_row, al, ls);
        m_row = mn;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        uint roff32 = r_l * 32u + c8_l;
        if (lb > 0u && simd_any(al < 0.999999f)) {
            #pragma unroll
            for (uint blk = 0; blk < 16u; blk += 4u) {
                simdgroup_store(o[blk + 0u], &sS[sg][0],  32); simdgroup_store(o[blk + 1u], &sS[sg][8],  32);
                simdgroup_store(o[blk + 2u], &sS[sg][16], 32); simdgroup_store(o[blk + 3u], &sS[sg][24], 32);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                *(threadgroup float4*)(&sS[sg][roff32])      *= al;
                *(threadgroup float4*)(&sS[sg][roff32 + 4u]) *= al;
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_load(o[blk + 0u], &sS[sg][0],  32); simdgroup_load(o[blk + 1u], &sS[sg][8],  32);
                simdgroup_load(o[blk + 2u], &sS[sg][16], 32); simdgroup_load(o[blk + 3u], &sS[sg][24], 32);
            }
        }
        #pragma unroll
        for (uint kk = 0; kk < 16u; kk += 8u) {
            simdgroup_half8x8 mP, mV;
            simdgroup_load(mP, &sP[sg][kk], 16, ulong2(0, 0), false);
            #pragma unroll
            for (uint d = 0; d < 16u; d++) {
                simdgroup_load(mV, &sV[kk][d * 8u], 128, ulong2(0, 0), false);
                simdgroup_multiply_accumulate(o[d], mP, mV, o[d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv = 1.0f / max(l_row, 1e-12f);
    uint roff32 = r_l * 32u + c8_l;
    ulong cb = ((ulong)(my_v ? my_b : 0u) * p.n_heads + h) * p.out_stride;
    #pragma unroll
    for (uint blk = 0; blk < 16u; blk += 4u) {
        simdgroup_store(o[blk + 0u], &sS[sg][0],  32); simdgroup_store(o[blk + 1u], &sS[sg][8],  32);
        simdgroup_store(o[blk + 2u], &sS[sg][16], 32); simdgroup_store(o[blk + 3u], &sS[sg][24], 32);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        uint cd = blk * 8u + c8_l;
        if (my_v && cd + 7u < p.out_stride) {
            *(device float4*)(ctx + cb + cd)      = *(const threadgroup float4*)(&sS[sg][roff32]) * inv;
            *(device float4*)(ctx + cb + cd + 4u) = *(const threadgroup float4*)(&sS[sg][roff32 + 4u]) * inv;
        } else if (my_v) {
            for (uint k = 0; k < 8u; k++) if (cd + k < p.out_stride) ctx[cb + cd + k] = sS[sg][roff32 + k] * inv;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
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
// Tenon v12 Round 7: Zero-Spill BQ=8, BK=16 Cooperative FlashAttention for ViT / SmolVLM (8x less K/V DRAM, native exp2!)
kernel void vit_full_attn(
    const device float* q [[buffer(0)]], const device float* k [[buffer(1)]], const device float* v [[buffer(2)]],
    device float* out [[buffer(3)]], constant VitAttnArgs& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x, qb0 = tgid.y * 8u;
    if (h >= p.n_heads || qb0 >= p.n_tok) return;
    uint dm = p.n_heads * p.d_head;
    uint dh = p.d_head;
    uint b0 = qb0 + sg * 2u, b1 = b0 + 1u;
    bool v0 = (b0 < p.n_tok), v1 = (b1 < p.n_tok);
    float scale_log2 = p.scale * 1.4426950408889634f;
    float m0 = -1e30f, l0 = 0.0f, m1 = -1e30f, l1 = 0.0f;

    if (dh == 64u) {
        threadgroup half sK[16][64];
        threadgroup half sV[16][64];
        ulong qb_0 = (ulong)(v0 ? b0 : 0u) * dm + (ulong)h * 64u;
        ulong qb_1 = (ulong)(v1 ? b1 : 0u) * dm + (ulong)h * 64u;
        uint d2 = lane * 2u;
        float2 q0 = v0 ? *(const device float2*)(q + qb_0 + d2) : float2(0.0f);
        float2 q1 = v1 ? *(const device float2*)(q + qb_1 + d2) : float2(0.0f);
        float2 o0 = float2(0.0f), o1 = float2(0.0f);
        for (uint tb = 0; tb < p.n_tok; tb += 16u) {
            uint nt = min(16u, p.n_tok - tb);
            uint tot4 = nt * 16u;
            for (uint vi = lid; vi < tot4; vi += 128u) {
                uint ti = vi >> 4, c4 = (vi & 15u) << 2;
                ulong ko = (ulong)(tb + ti) * dm + (ulong)h * 64u + c4;
                *(threadgroup half4*)(&sK[ti][c4]) = half4(*(const device float4*)(k + ko));
                *(threadgroup half4*)(&sV[ti][c4]) = half4(*(const device float4*)(v + ko));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint ti = 0; ti < nt; ti++) {
                float2 kv = float2(*(const threadgroup half2*)(&sK[ti][d2]));
                float2 vv = float2(*(const threadgroup half2*)(&sV[ti][d2]));
                if (v0) {
                    float sc = simd_sum(fma(q0.x, kv.x, q0.y * kv.y)) * scale_log2;
                    float mn = max(m0, sc), al = exp2(m0 - mn), wt = exp2(sc - mn);
                    l0 = fma(l0, al, wt); m0 = mn;
                    o0 = fma(o0, al, wt * vv);
                }
                if (v1) {
                    float sc = simd_sum(fma(q1.x, kv.x, q1.y * kv.y)) * scale_log2;
                    float mn = max(m1, sc), al = exp2(m1 - mn), wt = exp2(sc - mn);
                    l1 = fma(l1, al, wt); m1 = mn;
                    o1 = fma(o1, al, wt * vv);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (v0) *(device float2*)(out + qb_0 + d2) = o0 / max(l0, 1e-12f);
        if (v1) *(device float2*)(out + qb_1 + d2) = o1 / max(l1, 1e-12f);
        return;
    }

    // Generic fallback for dh != 64 (e.g. dh=72, 80, 128)
    threadgroup float sK[16][128];
    threadgroup float sV[16][128];
    uint dpl = (dh + 31u) >> 5;
    float q0a[4] = {0,0,0,0}, q1a[4] = {0,0,0,0}, o0a[4] = {0,0,0,0}, o1a[4] = {0,0,0,0};
    ulong qb_0 = (ulong)(v0 ? b0 : 0u) * dm + (ulong)h * dh;
    ulong qb_1 = (ulong)(v1 ? b1 : 0u) * dm + (ulong)h * dh;
    for (uint c = 0; c < dpl && c < 4u; c++) {
        uint d = lane + c * 32u;
        if (d < dh) { q0a[c] = v0 ? q[qb_0 + d] : 0.0f; q1a[c] = v1 ? q[qb_1 + d] : 0.0f; }
    }
    for (uint tb = 0; tb < p.n_tok; tb += 16u) {
        uint nt = min(16u, p.n_tok - tb);
        for (uint vi = lid; vi < nt * dh; vi += 128u) {
            uint ti = vi / dh, d = vi % dh;
            ulong ko = (ulong)(tb + ti) * dm + (ulong)h * dh + d;
            sK[ti][d] = k[ko]; sV[ti][d] = v[ko];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ti = 0; ti < nt; ti++) {
            float dt0 = 0.0f, dt1 = 0.0f;
            for (uint c = 0; c < dpl && c < 4u; c++) {
                uint d = lane + c * 32u;
                if (d < dh) { float kv = sK[ti][d]; dt0 = fma(q0a[c], kv, dt0); dt1 = fma(q1a[c], kv, dt1); }
            }
            if (v0) {
                float sc = simd_sum(dt0) * scale_log2;
                float mn = max(m0, sc), al = exp2(m0 - mn), wt = exp2(sc - mn);
                l0 = fma(l0, al, wt); m0 = mn;
                for (uint c = 0; c < dpl && c < 4u; c++) { uint d = lane + c * 32u; if (d < dh) o0a[c] = fma(o0a[c], al, wt * sV[ti][d]); }
            }
            if (v1) {
                float sc = simd_sum(dt1) * scale_log2;
                float mn = max(m1, sc), al = exp2(m1 - mn), wt = exp2(sc - mn);
                l1 = fma(l1, al, wt); m1 = mn;
                for (uint c = 0; c < dpl && c < 4u; c++) { uint d = lane + c * 32u; if (d < dh) o1a[c] = fma(o1a[c], al, wt * sV[ti][d]); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (v0) { float inv = 1.0f / max(l0, 1e-12f); for (uint c = 0; c < dpl && c < 4u; c++) { uint d = lane + c * 32u; if (d < dh) out[qb_0 + d] = o0a[c] * inv; } }
    if (v1) { float inv = 1.0f / max(l1, 1e-12f); for (uint c = 0; c < dpl && c < 4u; c++) { uint d = lane + c * 32u; if (d < dh) out[qb_1 + d] = o1a[c] * inv; } }
}

// Tenon v13 P1a: Hardware Apple AMX FlashAttention-2 for Multimodal ViT / SmolVLM (dh=64: BQ=32, BK=32, 18 KB SMEM!)
kernel void vit_full_attn_amx(
    const device float* q [[buffer(0)]], const device float* k [[buffer(1)]], const device float* v [[buffer(2)]],
    device float* out [[buffer(3)]], constant VitAttnArgs& p [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]
) {
    uint h = tgid.x, qb0 = tgid.y * 32u;
    if (h >= p.n_heads || qb0 >= p.n_tok) return;
    uint dm = p.n_heads * 64u;
    float scale_log2 = p.scale * 1.4426950408889634f;

    threadgroup half sQ[32][64];
    threadgroup half sK[32][64];
    threadgroup half sV[32][64];
    threadgroup float sS[4][256];
    threadgroup half sP[4][256];

    for (uint vi = lid; vi < 512u; vi += 128u) {
        uint rq = vi >> 4, c4 = (vi & 15u) << 2;
        uint b = qb0 + rq;
        if (b < p.n_tok) {
            ulong qoff = (ulong)b * dm + (ulong)h * 64ul + c4;
            *(threadgroup half4*)(&sQ[rq][c4]) = half4(*(const device float4*)(q + qoff));
        } else {
            *(threadgroup half4*)(&sQ[rq][c4]) = half4(0.0h);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint r_l = lane >> 2, c_l = (lane & 3u) * 8u;
    uint my_b = qb0 + sg * 8u + r_l;
    bool my_v = (my_b < p.n_tok);
    float m_row = -1e30f, l_row = 0.0f;
    simdgroup_float8x8 o0(0.0f), o1(0.0f), o2(0.0f), o3(0.0f), o4(0.0f), o5(0.0f), o6(0.0f), o7(0.0f);

    for (uint tb = 0; tb < p.n_tok; tb += 32u) {
        uint nt = min(32u, p.n_tok - tb);
        for (uint vi = lid; vi < 512u; vi += 128u) {
            uint tk = vi >> 4, c4 = (vi & 15u) << 2;
            if (tk < nt) {
                ulong ko = (ulong)(tb + tk) * dm + (ulong)h * 64ul + c4;
                *(threadgroup half4*)(&sK[tk][c4]) = half4(*(const device float4*)(k + ko));
                *(threadgroup half4*)(&sV[tk][c4]) = half4(*(const device float4*)(v + ko));
            } else {
                *(threadgroup half4*)(&sK[tk][c4]) = half4(0.0h);
                *(threadgroup half4*)(&sV[tk][c4]) = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 sc0(0.0f), sc1(0.0f), sc2(0.0f), sc3(0.0f);
        uint q_r = sg * 8u;
        #pragma unroll
        for (uint kk = 0; kk < 64u; kk += 8u) {
            simdgroup_half8x8 mQ, mK;
            simdgroup_load(mQ, &sQ[q_r][kk], 64, ulong2(0, 0), false);
            simdgroup_load(mK, &sK[0][kk],   64, ulong2(0, 0), true); simdgroup_multiply_accumulate(sc0, mQ, mK, sc0);
            simdgroup_load(mK, &sK[8][kk],   64, ulong2(0, 0), true); simdgroup_multiply_accumulate(sc1, mQ, mK, sc1);
            simdgroup_load(mK, &sK[16][kk],  64, ulong2(0, 0), true); simdgroup_multiply_accumulate(sc2, mQ, mK, sc2);
            simdgroup_load(mK, &sK[24][kk],  64, ulong2(0, 0), true); simdgroup_multiply_accumulate(sc3, mQ, mK, sc3);
        }
        simdgroup_store(sc0, &sS[sg][0],  32); simdgroup_store(sc1, &sS[sg][8],  32);
        simdgroup_store(sc2, &sS[sg][16], 32); simdgroup_store(sc3, &sS[sg][24], 32);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float sv[8];
        float rm = -1e30f;
        uint roff = r_l * 32u + c_l;
        #pragma unroll
        for (uint j = 0; j < 8u; j++) {
            uint c = c_l + j;
            sv[j] = (my_v && c < nt) ? (sS[sg][roff + j] * scale_log2) : -1e30f;
            rm = max(rm, sv[j]);
        }
        rm = max(rm, simd_shuffle_xor(rm, 1));
        rm = max(rm, simd_shuffle_xor(rm, 2));
        float mn = max(m_row, rm);
        float al = (m_row > -1e29f && mn > -1e29f) ? exp2(m_row - mn) : (mn > -1e29f ? 0.0f : 1.0f);
        float ls = 0.0f;
        #pragma unroll
        for (uint j = 0; j < 8u; j++) {
            float pv = (sv[j] > -1e29f) ? exp2(sv[j] - mn) : 0.0f;
            ls += pv;
            sP[sg][roff + j] = half(pv);
        }
        ls += simd_shuffle_xor(ls, 1);
        ls += simd_shuffle_xor(ls, 2);
        l_row = fma(l_row, al, ls);
        m_row = mn;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        if (tb > 0u && simd_any(al < 0.999999f)) {
            simdgroup_store(o0, &sS[sg][0],  32); simdgroup_store(o1, &sS[sg][8],  32);
            simdgroup_store(o2, &sS[sg][16], 32); simdgroup_store(o3, &sS[sg][24], 32);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            *(threadgroup float4*)(&sS[sg][roff])      *= al;
            *(threadgroup float4*)(&sS[sg][roff + 4u]) *= al;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(o0, &sS[sg][0],  32); simdgroup_load(o1, &sS[sg][8],  32);
            simdgroup_load(o2, &sS[sg][16], 32); simdgroup_load(o3, &sS[sg][24], 32);
            simdgroup_store(o4, &sS[sg][0],  32); simdgroup_store(o5, &sS[sg][8],  32);
            simdgroup_store(o6, &sS[sg][16], 32); simdgroup_store(o7, &sS[sg][24], 32);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            *(threadgroup float4*)(&sS[sg][roff])      *= al;
            *(threadgroup float4*)(&sS[sg][roff + 4u]) *= al;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(o4, &sS[sg][0],  32); simdgroup_load(o5, &sS[sg][8],  32);
            simdgroup_load(o6, &sS[sg][16], 32); simdgroup_load(o7, &sS[sg][24], 32);
        }
        #pragma unroll
        for (uint kk = 0; kk < 32u; kk += 8u) {
            simdgroup_half8x8 mP, mV;
            simdgroup_load(mP, &sP[sg][kk], 32, ulong2(0, 0), false);
            simdgroup_load(mV, &sV[kk][0],  64, ulong2(0, 0), false); simdgroup_multiply_accumulate(o0, mP, mV, o0);
            simdgroup_load(mV, &sV[kk][8],  64, ulong2(0, 0), false); simdgroup_multiply_accumulate(o1, mP, mV, o1);
            simdgroup_load(mV, &sV[kk][16], 64, ulong2(0, 0), false); simdgroup_multiply_accumulate(o2, mP, mV, o2);
            simdgroup_load(mV, &sV[kk][24], 64, ulong2(0, 0), false); simdgroup_multiply_accumulate(o3, mP, mV, o3);
            simdgroup_load(mV, &sV[kk][32], 64, ulong2(0, 0), false); simdgroup_multiply_accumulate(o4, mP, mV, o4);
            simdgroup_load(mV, &sV[kk][40], 64, ulong2(0, 0), false); simdgroup_multiply_accumulate(o5, mP, mV, o5);
            simdgroup_load(mV, &sV[kk][48], 64, ulong2(0, 0), false); simdgroup_multiply_accumulate(o6, mP, mV, o6);
            simdgroup_load(mV, &sV[kk][56], 64, ulong2(0, 0), false); simdgroup_multiply_accumulate(o7, mP, mV, o7);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv = 1.0f / max(l_row, 1e-12f);
    uint roff = r_l * 32u + c_l;
    ulong ob = (ulong)(my_v ? my_b : 0u) * dm + (ulong)h * 64ul;
    simdgroup_store(o0, &sS[sg][0],  32); simdgroup_store(o1, &sS[sg][8],  32);
    simdgroup_store(o2, &sS[sg][16], 32); simdgroup_store(o3, &sS[sg][24], 32);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (my_v) {
        *(device float4*)(out + ob + c_l)      = *(const threadgroup float4*)(&sS[sg][roff]) * inv;
        *(device float4*)(out + ob + c_l + 4u) = *(const threadgroup float4*)(&sS[sg][roff + 4u]) * inv;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(o4, &sS[sg][0],  32); simdgroup_store(o5, &sS[sg][8],  32);
    simdgroup_store(o6, &sS[sg][16], 32); simdgroup_store(o7, &sS[sg][24], 32);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (my_v) {
        *(device float4*)(out + ob + 32u + c_l)      = *(const threadgroup float4*)(&sS[sg][roff]) * inv;
        *(device float4*)(out + ob + 32u + c_l + 4u) = *(const threadgroup float4*)(&sS[sg][roff + 4u]) * inv;
    }
}
)";

static int moe_init_metal(void) {
    if (g_moe_dev != nil) return 1;
    g_moe_dev = MTLCreateSystemDefaultDevice();
    if (!g_moe_dev) return 0;
    g_moe_q = [g_moe_dev newCommandQueue];
    g_moe_aux_q = [g_moe_dev newCommandQueue];
    NSError* err = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
    g_moe_lib = [g_moe_dev newLibraryWithSource:kMoeMslSource options:opts error:&err];
    if (!g_moe_lib) {
        fprintf(stderr, "[tenon-stream] MSL compile error: %s\n", [[err localizedDescription] UTF8String]);
        return 0;
    }
    id<MTLComputePipelineState> (^mk)(NSString*) = ^id<MTLComputePipelineState>(NSString* nm) {
        NSError* pe = nil;
        id<MTLFunction> fn = [g_moe_lib newFunctionWithName:nm];
        if (!fn) { fprintf(stderr, "[tenon-stream] missing kernel %s\n", [nm UTF8String]); return nil; }
        MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
        desc.computeFunction = fn;
        desc.supportIndirectCommandBuffers = YES;
        id<MTLComputePipelineState> pso = [g_moe_dev newComputePipelineStateWithDescriptor:desc options:0 reflection:nil error:&pe];
        if (!pso) {
            pe = nil;
            pso = [g_moe_dev newComputePipelineStateWithFunction:fn error:&pe];
        }
        if (!pso) fprintf(stderr, "[tenon-stream] PSO error %s: %s\n", [nm UTF8String], [[pe localizedDescription] UTF8String]);
        return pso;
    };
    g_pso_embed             = mk(@"moe_embed_lookup");
    g_pso_rmsnorm           = mk(@"moe_rmsnorm");
    g_pso_rmsnorm_strided   = mk(@"moe_rmsnorm_strided");
    g_pso_gemv_qkv_q4       = mk(@"moe_gemv_qkv_q4");
    g_pso_flash_prefill_paged_fp16 = mk(@"moe_flash_prefill_paged_fp16");
    g_pso_flash_prefill_amx64      = mk(@"moe_flash_prefill_amx64");
    g_pso_flash_prefill_amx128     = mk(@"moe_flash_prefill_amx128");
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
    g_pso_dense_gu_q4k      = mk(@"moe_dense_gate_up_q4k");
    g_pso_dense_gu_q5k      = mk(@"moe_dense_gate_up_q5k");
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
    g_pso_gemm_q4_qkv_b     = mk(@"moe_gemm_q4_qkv_batch");
    g_pso_gemm_q4_gu_b      = mk(@"moe_gemm_q4_gu_swiglu_batch");
    g_pso_gemm_q4_amx16     = mk(@"moe_gemm_q4_amx16");
    g_pso_gemm_q4_amx8      = mk(@"moe_gemm_q4_amx8");
    g_pso_gemm_q4_sb        = mk(@"moe_gemm_q4_small_b");
    g_pso_gemm_q4_1_b       = mk(@"moe_gemm_q4_1_batch");
    g_pso_gemm_q4_1_amx8    = mk(@"moe_gemm_q4_1_amx8");
    g_pso_gemm_q4_1_sb      = mk(@"moe_gemm_q4_1_small_b");
    g_pso_gemm_q5_0_b       = mk(@"moe_gemm_q5_0_batch");
    g_pso_gemm_q5_1_b       = mk(@"moe_gemm_q5_1_batch");
    g_pso_gemm_q8_b         = mk(@"moe_gemm_q8_batch");
    g_pso_gemm_q2k_b        = mk(@"moe_gemm_q2k_batch");
    g_pso_gemm_q3k_b        = mk(@"moe_gemm_q3k_batch");
    g_pso_gemm_iq4xs_b      = mk(@"moe_gemm_iq4xs_batch");
    g_pso_gemm_q4k_b        = mk(@"moe_gemm_q4k_batch");
    g_pso_gemm_q4k_amx16    = mk(@"moe_gemm_q4k_amx16");
    g_pso_gemm_q4k_sb       = mk(@"moe_gemm_q4k_small_b");
    g_pso_gemm_q5k_b        = mk(@"moe_gemm_q5k_batch");
    g_pso_gemm_q6k_b        = mk(@"moe_gemm_q6k_batch");
    g_pso_gemm_q6k_amx16    = mk(@"moe_gemm_q6k_amx16");
    g_pso_gemm_q6k_sb       = mk(@"moe_gemm_q6k_small_b");
    g_pso_gemv_fp8          = mk(@"moe_gemv_fp8_e4m3");
    g_pso_gemm_fp8_b        = mk(@"moe_gemm_fp8_e4m3_batch");
    g_pso_gemv_mxfp4        = mk(@"moe_gemv_mxfp4");
    g_pso_gemm_mxfp4_b      = mk(@"moe_gemm_mxfp4_batch");
    g_pso_lora_a            = mk(@"moe_lora_proj_a");
    g_pso_lora_b            = mk(@"moe_lora_proj_b_add");
    g_pso_gemm_f16_b        = mk(@"moe_gemm_f16_batch");
    g_pso_gemm_f16_amx      = mk(@"moe_gemm_f16_amx");
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
    g_pso_vit_attn_amx      = mk(@"vit_full_attn_amx");
    g_pso_gemm_q6k_amx      = mk(@"moe_gemm_q6k_amx");
    g_pso_gemm_fp8_amx      = mk(@"moe_gemm_fp8_amx");
    g_pso_gemm_mxfp4_amx    = mk(@"moe_gemm_mxfp4_amx");
    g_pso_gemm_q8_amx       = mk(@"moe_gemm_q8_amx_f16");
    g_pso_moe_route_sort    = mk(@"moe_route_count_sort_b");
    g_pso_moe_mega_gu       = mk(@"moe_grouped_gu_swiglu_mega");
    g_pso_moe_mega_down     = mk(@"moe_grouped_down_mega");
    g_pso_moe_mega_reduce   = mk(@"moe_reduce_tokens_mega");
    g_pso_lora_bgmv_a       = mk(@"moe_lora_bgmv_a");
    g_pso_lora_bgmv_b       = mk(@"moe_lora_bgmv_b_add");
    g_pso_mla_rope_store_paged = mk(@"moe_mla_rope_store_paged");
    g_pso_mla_absorb_q_b    = mk(@"moe_mla_absorb_q_batch");
    g_pso_mla_flash_lat_paged = mk(@"moe_flash_mla_latent_paged");
    g_pso_mla_proj_v_b      = mk(@"moe_mla_proj_v_batch");
    g_pso_gather_rows       = mk(@"moe_gather_rows");
    g_pso_scatter_add_rows  = mk(@"moe_scatter_add_rows");
    g_pso_embed_batch       = mk(@"moe_embed_lookup_batch");
    g_pso_moe_mega_gu_amx   = mk(@"moe_grouped_gu_swiglu_mega_amx");
    g_pso_moe_mega_down_amx = mk(@"moe_grouped_down_mega_amx");
    g_pso_rope_kv_fp8_paged = mk(@"moe_rope_kv_fp8_paged");
    g_pso_flash_dec_fp8_paged = mk(@"moe_flash_decode_fp8_paged");
    g_pso_rope_kv_q4_paged  = mk(@"moe_rope_kv_q4_paged");
    g_pso_flash_dec_q4_paged = mk(@"moe_flash_decode_q4_paged");
    g_pso_flash_dec_split_paged = mk(@"moe_flash_decode_split_paged");
    g_pso_flash_mla_split_paged = mk(@"moe_flash_mla_split_paged");
    g_pso_flash_mla_split_paged_512 = mk(@"moe_flash_mla_split_paged_512");
    g_pso_flash_dec_reduce_paged = mk(@"moe_flash_decode_reduce_paged");
    g_pso_argmax            = mk(@"moe_argmax_f32");
    g_pso_argmax_b          = mk(@"moe_argmax_batch_f32");
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
        // Tenon v10: Apple Silicon Metal ICB `setKernelBuffer:offset:` has a 32-bit offset/range limit on >2 GiB buffers.
        // Always create a zero-copy page-aligned `newBufferWithBytesNoCopy` view for shards >2 GiB (0 extra RAM, offset <16 KiB)!
        if (m->mmap_ptrs[s] != NULL && m->mmap_lens[s] > 0x80000000ULL) {
            size_t psz = (size_t)getpagesize();
            uint64_t pg_off = ti->file_off & ~(uint64_t)(psz - 1);
            uint64_t sub_off = ti->file_off - pg_off;
            size_t view_len = (size_t)((sub_off + ti->byte_size + psz - 1) & ~(uint64_t)(psz - 1));
            if (pg_off + view_len > m->mmap_lens[s]) view_len = m->mmap_lens[s] - (size_t)pg_off;
            id<MTLBuffer> sub = [g_moe_dev newBufferWithBytesNoCopy:((uint8_t*)m->mmap_ptrs[s] + pg_off)
                                                             length:view_len
                                                            options:MTLResourceStorageModeShared
                                                        deallocator:nil];
            if (sub != nil) {
                ref->buf = sub;
                ref->offset = sub_off;
                return;
            }
        }
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
        } else if (kv_ends_with(key, ".rope.dimension_sections") && vt == GGUF_T_ARR) {
            uint32_t et = gr_u32(&gr);
            uint64_t n = gr_u64(&gr);
            int sum_sec = 0;
            for (uint64_t si = 0; si < n; si++) {
                int64_t sv = gr_read_int(&gr, et);
                if (si < 3) { m->mrope_sections[si] = (int)sv; sum_sec += (int)sv; }
            }
            if (sum_sec > 0) m->has_mrope = 1;
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
        } else if (kv_ends_with(key, ".attention.sliding_window")) {
            m->sliding_window = (int)gr_read_int(&gr, vt);
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
        if (kvq_env && (strcmp(kvq_env, "q4") == 0 || strcmp(kvq_env, "q4_0") == 0 || strcmp(kvq_env, "3") == 0)) {
            m->kv_quant_q8 = 3;
        } else if (kvq_env && (strcmp(kvq_env, "fp8") == 0 || strcmp(kvq_env, "fp8_e4m3") == 0 || strcmp(kvq_env, "2") == 0)) {
            m->kv_quant_q8 = 2;
        } else if (kvq_env && (strcmp(kvq_env, "q8") == 0 || strcmp(kvq_env, "q8_0") == 0 || strcmp(kvq_env, "1") == 0)) {
            m->kv_quant_q8 = 1;
        } else if (kvq_env && (strcmp(kvq_env, "fp16") == 0 || strcmp(kvq_env, "f16") == 0 || strcmp(kvq_env, "0") == 0)) {
            m->kv_quant_q8 = 0;
        } else {
            m->kv_quant_q8 = (m->max_ctx > 2048) ? 1 : 0;
        }
        const char* swa_env = getenv("TENON_SLIDING_WINDOW");
        if (swa_env && atoi(swa_env) >= 0) m->sliding_window = atoi(swa_env);
        const char* sink_env = getenv("TENON_ATTN_SINK");
        m->attn_sink = (sink_env && atoi(sink_env) > 0) ? atoi(sink_env) : 0;
        const char* icb_env = getenv("TENON_METAL_ICB");
        m->icb_enabled = (icb_env && atoi(icb_env) == 0) ? 0 : 1;
        m->icb_event = [g_moe_dev newSharedEvent];
        m->icb_ev_val = 0;
        const char* fd_env = getenv("TENON_FLASH_DECODE");
        m->fd_enabled = (fd_env && atoi(fd_env) == 0) ? 0 : 1;
        const char* fdm_env = getenv("TENON_FLASH_DECODE_MIN_LEN");
        m->fd_min_len = (fdm_env && atoi(fdm_env) > 0) ? atoi(fdm_env) : 512;
        const char* fds_env = getenv("TENON_FLASH_DECODE_SPLITS");
        if (fds_env && atoi(fds_env) >= 2 && atoi(fds_env) <= MOE_FD_MAX_SPLITS) {
            m->fd_splits = atoi(fds_env);
            m->fd_splits_forced = 1;
        } else {
            m->fd_splits = 8;
            m->fd_splits_forced = 0;
        }
        m->cur_fd_splits = m->fd_splits;
        m->cur_is_prefill = 0;
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
            } else if (strncmp(ti->name, "nextn.", 6) == 0 || strncmp(ti->name, "blk.", 4) == 0) {
                int L = -1;
                const char* sub = NULL;
                if (strncmp(ti->name, "nextn.", 6) == 0) {
                    L = m->n_layers;
                    sub = ti->name + 6;
                } else {
                    L = atoi(ti->name + 4);
                    const char* dot = strchr(ti->name + 4, '.');
                    if (!dot) continue;
                    sub = dot + 1;
                }
                if (L == m->n_layers && (m->nextn_layers > 0 || strncmp(ti->name, "nextn.", 6) == 0)) {
                    if (strncmp(sub, "nextn.", 6) == 0) sub += 6;
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
                        else if (strcmp(sub, "attn_q.bias") == 0) { load_pinned_weight(m, &ml->bq, ti); ml->has_qkv_bias = 1; }
                        else if (strcmp(sub, "attn_k.bias") == 0) { load_pinned_weight(m, &ml->bk, ti); ml->has_qkv_bias = 1; }
                        else if (strcmp(sub, "attn_v.bias") == 0) { load_pinned_weight(m, &ml->bv, ti); ml->has_qkv_bias = 1; }
                        else if (strcmp(sub, "attn_q_norm.weight") == 0) { load_pinned_weight(m, &ml->q_norm, ti); ml->has_qk_norm = 1; ml->qk_norm_per_head = ((int)ti->ne[0] == m->d_head) ? 1 : 0; }
                        else if (strcmp(sub, "attn_k_norm.weight") == 0) load_pinned_weight(m, &ml->k_norm, ti);
                        else if (strcmp(sub, "attn_q_a.weight") == 0) { load_pinned_weight(m, &ml->wq_a, ti); ml->is_mla = 1; }
                        else if (strcmp(sub, "attn_q_a_norm.weight") == 0) load_pinned_weight(m, &ml->q_a_norm, ti);
                        else if (strcmp(sub, "attn_q_b.weight") == 0) load_pinned_weight(m, &ml->wq_b, ti);
                        else if (strcmp(sub, "attn_kv_a_mqa.weight") == 0) { load_pinned_weight(m, &ml->wkv_a, ti); ml->is_mla = 1; }
                        else if (strcmp(sub, "attn_kv_a_norm.weight") == 0) load_pinned_weight(m, &ml->kv_a_norm, ti);
                        else if (strcmp(sub, "attn_kv_b.weight") == 0) load_pinned_weight(m, &ml->wkv_b, ti);
                        else if (strcmp(sub, "attn_output.weight") == 0) load_pinned_weight(m, &ml->wo, ti);
                        else if (strcmp(sub, "ffn_gate.weight") == 0) { load_pinned_weight(m, &ml->ffn_gate, ti); ml->is_dense = 1; }
                        else if (strcmp(sub, "ffn_up.weight") == 0) load_pinned_weight(m, &ml->ffn_up, ti);
                        else if (strcmp(sub, "ffn_down.weight") == 0) load_pinned_weight(m, &ml->ffn_down, ti);
                        else if (strcmp(sub, "ffn_gate_inp.weight") == 0) load_pinned_weight(m, &ml->router, ti);
                        else if (strcmp(sub, "exp_probs_b.bias") == 0) { load_pinned_weight(m, &ml->exp_bias, ti); ml->has_exp_bias = 1; }
                        else if (strcmp(sub, "ffn_gate_exps.weight") == 0) {
                            load_pinned_weight(m, &ml->gate_exps, ti);
                            ml->gate_exp_bytes = (m->n_experts > 0) ? (ti->byte_size / (uint64_t)m->n_experts) : ti->byte_size;
                            ml->is_dense = 0;
                        } else if (strcmp(sub, "ffn_up_exps.weight") == 0) {
                            load_pinned_weight(m, &ml->up_exps, ti);
                            ml->up_exp_bytes = (m->n_experts > 0) ? (ti->byte_size / (uint64_t)m->n_experts) : ti->byte_size;
                        } else if (strcmp(sub, "ffn_down_exps.weight") == 0) {
                            load_pinned_weight(m, &ml->down_exps, ti);
                            ml->down_exp_bytes = (m->n_experts > 0) ? (ti->byte_size / (uint64_t)m->n_experts) : ti->byte_size;
                        }
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
                    if (m->resident_mode) load_pinned_weight(m, &ly->gate_exps, ti);
                    m->total_streamable_bytes += (int64_t)ti->byte_size;
                } else if (strcmp(sub, "ffn_up_exps.weight") == 0) {
                    record_weight_meta(&ly->up_exps, ti);
                    ly->up_exp_bytes = ti->byte_size / (uint64_t)m->n_experts;
                    if (m->resident_mode) load_pinned_weight(m, &ly->up_exps, ti);
                    m->total_streamable_bytes += (int64_t)ti->byte_size;
                } else if (strcmp(sub, "ffn_down_exps.weight") == 0) {
                    record_weight_meta(&ly->down_exps, ti);
                    ly->down_exp_bytes = ti->byte_size / (uint64_t)m->n_experts;
                    if (m->resident_mode) load_pinned_weight(m, &ly->down_exps, ti);
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

        // Allocate KV caches (MLA Compressed Latent `[c_kv||k_pe]` = -8x..-85x! OR Q4_0 = 3.56x! OR Q8_0/FP8_E4M3 = 1.88x! OR FP16)
        size_t kv_row_b = (m->kv_quant_q8 == 3) ? ((size_t)m->n_kv * (size_t)(m->d_head >> 5) * 18u)
                        : (m->kv_quant_q8 >= 1) ? ((size_t)m->n_kv * (size_t)(m->d_head >> 5) * 34u)
                        : ((size_t)(m->n_kv * m->d_head) * 2u);
        size_t kv_bytes = (size_t)m->max_ctx * kv_row_b;
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
        m->buf_argmax  = [g_moe_dev newBufferWithLength:(size_t)MOE_PREFILL_MAX_B * 4 options:MTLResourceStorageModeShared];
        m->batch_logits_dirty_row = -1;
        m->buf_last_hidden = [g_moe_dev newBufferWithLength:(size_t)m->d_model * 4 options:MTLResourceStorageModeShared];
        if (m->has_mtp) {
            m->buf_mtp_cat = [g_moe_dev newBufferWithLength:(size_t)m->d_model * 8 options:MTLResourceStorageModeShared];
            if (m->mtp_layer && m->has_mtp_layer) {
                MoeLayer* ml = m->mtp_layer;
                if (ml->is_mla && m->mla_latent_kv && ml->wkv_b.buf != nil) {
                    uint32_t kv_r = (uint32_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
                    uint32_t nope_d = (uint32_t)(m->d_head - m->rope_dim);
                    uint32_t v_d = (uint32_t)m->v_head_dim;
                    size_t lat_bytes = (size_t)m->max_ctx * (size_t)(kv_r + (uint32_t)m->rope_dim) * 2u;
                    ml->k_cache = [g_moe_dev newBufferWithLength:lat_bytes options:MTLResourceStorageModeShared];
                    ml->v_cache = nil;
                    size_t wuk_b = (size_t)m->n_heads * kv_r * nope_d * 2u;
                    size_t wuv_b = (size_t)m->n_heads * v_d * kv_r * 2u;
                    ml->mla_wuk_f16 = [g_moe_dev newBufferWithLength:(wuk_b > 64 ? wuk_b : 64) options:MTLResourceStorageModeShared];
                    ml->mla_wuv_f16 = [g_moe_dev newBufferWithLength:(wuv_b > 64 ? wuv_b : 64) options:MTLResourceStorageModeShared];
                    id<MTLCommandBuffer> ucmd = [g_moe_q commandBuffer];
                    id<MTLComputeCommandEncoder> uenc = [ucmd computeCommandEncoder];
                    struct { uint32_t nh, nd, vd, kvr, gt; } ua = { (uint32_t)m->n_heads, nope_d, v_d, kv_r, ml->wkv_b.ggml_type };
                    [uenc setComputePipelineState:g_pso_mla_unpack_kvb];
                    [uenc setBuffer:ml->wkv_b.buf offset:(NSUInteger)ml->wkv_b.offset atIndex:0];
                    [uenc setBuffer:ml->mla_wuk_f16 offset:0 atIndex:1];
                    [uenc setBuffer:ml->mla_wuv_f16 offset:0 atIndex:2];
                    [uenc setBytes:&ua length:sizeof(ua) atIndex:3];
                    uint32_t max_j = nope_d > v_d ? nope_d : v_d;
                    [uenc dispatchThreads:MTLSizeMake(kv_r, max_j, (NSUInteger)m->n_heads) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    [uenc endEncoding]; [ucmd commit]; [ucmd waitUntilCompleted];
                } else {
                    ml->k_cache = [g_moe_dev newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
                    ml->v_cache = [g_moe_dev newBufferWithLength:kv_bytes options:MTLResourceStorageModeShared];
                }
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
        m->buf_cb_tok     = [g_moe_dev newBufferWithLength:(size_t)MOE_PREFILL_MAX_B * 4 options:MTLResourceStorageModeShared];
        m->buf_cb_pos     = [g_moe_dev newBufferWithLength:(size_t)MOE_PREFILL_MAX_B * 4 options:MTLResourceStorageModeShared];
        m->buf_cb_slot    = [g_moe_dev newBufferWithLength:(size_t)MOE_PREFILL_MAX_B * 4 options:MTLResourceStorageModeShared];
        m->buf_tree_depth = [g_moe_dev newBufferWithLength:(size_t)TREE_MAX_NODES * 4 options:MTLResourceStorageModeShared];
        m->buf_tree_mask  = [g_moe_dev newBufferWithLength:(size_t)TREE_MAX_NODES * 8 options:MTLResourceStorageModeShared];
        m->prefetch_grp = dispatch_group_create();
        m->dense_io_grp = dispatch_group_create();
        m->io_queue = dispatch_queue_create("tenon.stream.nvme", DISPATCH_QUEUE_CONCURRENT);
        m->ring_prod_q = dispatch_queue_create("tenon.stream.ring_prod", DISPATCH_QUEUE_SERIAL);
        // Tenon v12 Round 2: Safe Metal 4 / macOS 15+ MTLResidencySet (deduplicated; guards >24GB mmap against macOS wired-vnode stall!)
        if (@available(macOS 15.0, *)) {
            const char* res_env = getenv("TENON_METAL_RESIDENCY");
            int res_en = (res_env && atoi(res_env) == 0) ? 0 : 1;
            if (res_en) {
                MTLResidencySetDescriptor* rdesc = [[MTLResidencySetDescriptor alloc] init];
                rdesc.initialCapacity = (NSUInteger)(m->n_layers * 6 + m->n_shards + 32);
                NSError* rerr = nil;
                id<MTLResidencySet> rset = [g_moe_dev newResidencySetWithDescriptor:rdesc error:&rerr];
                if (rset != nil) {
                    int pin_mmap = (m->total_disk_bytes <= (24ull << 30)) ? 1 : 0;
                    #define ADD_RES(b) do { \
                        id<MTLBuffer> _b = (b); \
                        if (_b != nil) { \
                            int _is_mm = 0; \
                            for (int _s = 0; _s < m->n_shards; _s++) if (_b == m->mmap_mtl_bufs[_s]) { _is_mm = 1; break; } \
                            if (!_is_mm || pin_mmap) [rset addAllocation:_b]; \
                        } \
                    } while (0)
                    if (pin_mmap) {
                        for (int s = 0; s < m->n_shards; s++) ADD_RES(m->mmap_mtl_bufs[s]);
                    }
                    ADD_RES(m->tok_embd.buf); ADD_RES(m->out_norm.buf); ADD_RES(m->lm_head.buf);
                    ADD_RES(m->rope_freqs.buf);
                    for (int L = 0; L < m->n_layers; L++) {
                        MoeLayer* ly = &m->layers[L];
                        ADD_RES(ly->mla_wuk_f16); ADD_RES(ly->mla_wuv_f16);
                        ADD_RES(ly->cache_buf); ADD_RES(ly->k_cache); ADD_RES(ly->v_cache);
                    }
                    #undef ADD_RES
                    [rset commit];
                    [rset requestResidency];
                    [g_moe_q addResidencySet:rset];
                    m->residency_set = rset;
                }
            }
        }
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
        case TN_GGML_Q8_0:   pso = g_pso_gemv_q8;    rows_per_tg = 8; tptg = 64; break;
        case TN_GGML_Q2_K:   pso = g_pso_gemv_q2k;   rows_per_tg = 8; break;
        case TN_GGML_Q3_K:   pso = g_pso_gemv_q3k;   rows_per_tg = 8; break;
        case TN_GGML_IQ4_XS: pso = g_pso_gemv_iq4xs; rows_per_tg = 8; break;
        case TN_GGML_Q4_K:   pso = g_pso_gemv_q4k;   rows_per_tg = 8; tptg = 64; break;
        case TN_GGML_Q5_K:   pso = g_pso_gemv_q5k;   rows_per_tg = 4; tptg = 64; break;
        case TN_GGML_Q6_K:     pso = g_pso_gemv_q6k;   rows_per_tg = 4; tptg = 64; break;
        case TN_GGML_FP8_E4M3: pso = g_pso_gemv_fp8;   rows_per_tg = 4; break;
        case TN_GGML_MXFP4:    pso = g_pso_gemv_mxfp4; rows_per_tg = 4; break;
        case TN_GGML_F16:      pso = g_pso_gemv_f16;   rows_per_tg = 4; break;
        default:               pso = g_pso_gemv_f32;   rows_per_tg = 4; break;
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

static void encode_lora_delta(
    MoeModel* m, id<MTLComputeCommandEncoder> enc, int L, int proj,
    id<MTLBuffer> x, id<MTLBuffer> y, uint32_t out_dim, uint32_t in_dim, uint32_t B
) {
    if (L < 0 || L >= MOE_MAX_LAYERS) return;
    if (m->use_batch_lora) {
        if (B == 1u) {
            int lid1 = m->batch_lora_ids[0];
            if (lid1 <= 0 || lid1 >= MOE_MAX_LORAS || !m->loras[lid1].active) return;
            id<MTLBuffer> ba = m->loras[lid1].a[L][proj];
            id<MTLBuffer> bb = m->loras[lid1].b[L][proj];
            int rk = m->loras[lid1].r_dim[L][proj];
            if (ba == nil || bb == nil || rk <= 0) return;
            if (m->buf_lora_tmp == nil) m->buf_lora_tmp = [g_moe_dev newBufferWithLength:(size_t)MOE_PREFILL_MAX_B * 256u * 4u options:MTLResourceStorageModeShared];
            float alpha = (m->loras[lid1].alpha > 0.0f) ? m->loras[lid1].alpha : (float)rk;
            float sc = m->loras[lid1].user_scale * (alpha / (float)rk);
            struct { uint32_t out_dim, in_dim, rank, batch_b; float scale; } args = { out_dim, in_dim, (uint32_t)rk, 1u, sc };
            [enc setComputePipelineState:g_pso_lora_a];
            [enc setBuffer:ba offset:0 atIndex:0]; [enc setBuffer:x offset:0 atIndex:1];
            [enc setBuffer:m->buf_lora_tmp offset:0 atIndex:2]; [enc setBytes:&args length:sizeof(args) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rk, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc setComputePipelineState:g_pso_lora_b];
            [enc setBuffer:bb offset:0 atIndex:0]; [enc setBuffer:m->buf_lora_tmp offset:0 atIndex:1];
            [enc setBuffer:y offset:0 atIndex:2]; [enc setBytes:&args length:sizeof(args) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(out_dim, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            return;
        }
        // Tenon v9: Fused S-LoRA / Punica Heterogeneous BGMV across ALL active tenants in ONE single launch!
        struct { uint32_t out_dim, in_dim, batch_b, max_rank; uint32_t ranks[8]; float scales[8]; } bg;
        memset(&bg, 0, sizeof(bg));
        bg.out_dim = out_dim; bg.in_dim = in_dim; bg.batch_b = B;
        int any_hit = 0;
        for (int lid = 1; lid < MOE_MAX_LORAS && lid < 8; lid++) {
            if (!m->loras[lid].active) continue;
            int rk = m->loras[lid].r_dim[L][proj];
            if (m->loras[lid].a[L][proj] != nil && m->loras[lid].b[L][proj] != nil && rk > 0) {
                bg.ranks[lid] = (uint32_t)rk;
                float alpha = (m->loras[lid].alpha > 0.0f) ? m->loras[lid].alpha : (float)rk;
                bg.scales[lid] = m->loras[lid].user_scale * (alpha / (float)rk);
                if ((uint32_t)rk > bg.max_rank) bg.max_rank = (uint32_t)rk;
                any_hit = 1;
            }
        }
        if (!any_hit || bg.max_rank == 0u) return;
        if (m->buf_lora_tmp == nil) m->buf_lora_tmp = [g_moe_dev newBufferWithLength:(size_t)MOE_PREFILL_MAX_B * 256u * 4u options:MTLResourceStorageModeShared];
        if (m->buf_lora_dummy == nil) m->buf_lora_dummy = [g_moe_dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        [enc setComputePipelineState:g_pso_lora_bgmv_a];
        [enc setBuffer:x offset:0 atIndex:0];
        [enc setBuffer:m->buf_lora_tmp offset:0 atIndex:1];
        [enc setBuffer:m->buf_batch_lora_ids offset:0 atIndex:2];
        for (int lid = 1; lid < 8; lid++) {
            id<MTLBuffer> ba = (lid < MOE_MAX_LORAS && m->loras[lid].a[L][proj] != nil) ? m->loras[lid].a[L][proj] : m->buf_lora_dummy;
            [enc setBuffer:ba offset:0 atIndex:(NSUInteger)(2 + lid)];
        }
        [enc setBytes:&bg length:sizeof(bg) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake(bg.max_rank, B, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc setComputePipelineState:g_pso_lora_bgmv_b];
        [enc setBuffer:m->buf_lora_tmp offset:0 atIndex:0];
        [enc setBuffer:y offset:0 atIndex:1];
        [enc setBuffer:m->buf_batch_lora_ids offset:0 atIndex:2];
        for (int lid = 1; lid < 8; lid++) {
            id<MTLBuffer> bb = (lid < MOE_MAX_LORAS && m->loras[lid].b[L][proj] != nil) ? m->loras[lid].b[L][proj] : m->buf_lora_dummy;
            [enc setBuffer:bb offset:0 atIndex:(NSUInteger)(2 + lid)];
        }
        [enc setBytes:&bg length:sizeof(bg) atIndex:10];
        [enc dispatchThreads:MTLSizeMake(out_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        return;
    }
    int lid = m->active_lora;
    if (lid <= 0 || lid >= MOE_MAX_LORAS || !m->loras[lid].active) return;
    id<MTLBuffer> buf_a = m->loras[lid].a[L][proj];
    id<MTLBuffer> buf_b = m->loras[lid].b[L][proj];
    int rank = m->loras[lid].r_dim[L][proj];
    if (buf_a == nil || buf_b == nil || rank <= 0) return;
    if (m->buf_lora_tmp == nil) {
        m->buf_lora_tmp = [g_moe_dev newBufferWithLength:(size_t)MOE_PREFILL_MAX_B * 256u * 4u options:MTLResourceStorageModeShared];
    }
    float alpha = (m->loras[lid].alpha > 0.0f) ? m->loras[lid].alpha : (float)rank;
    float scale = m->loras[lid].user_scale * (alpha / (float)rank);
    struct { uint32_t out_dim, in_dim, rank, batch_b; float scale; } args = { out_dim, in_dim, (uint32_t)rank, B, scale };
    [enc setComputePipelineState:g_pso_lora_a];
    [enc setBuffer:buf_a offset:0 atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1];
    [enc setBuffer:m->buf_lora_tmp offset:0 atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rank, B, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    [enc setComputePipelineState:g_pso_lora_b];
    [enc setBuffer:buf_b offset:0 atIndex:0];
    [enc setBuffer:m->buf_lora_tmp offset:0 atIndex:1];
    [enc setBuffer:y offset:0 atIndex:2];
    [enc setBytes:&args length:sizeof(args) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(out_dim, B, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

static void encode_dense_ffn_raw(
    MoeModel* m, id<MTLComputeCommandEncoder> enc,
    id<MTLBuffer> g_buf, uint64_t g_off, uint32_t g_type,
    id<MTLBuffer> u_buf, uint64_t u_off, uint32_t u_type,
    id<MTLBuffer> d_buf, uint64_t d_off, uint32_t d_type,
    uint32_t d_ff
) {
    uint32_t dm = (uint32_t)m->d_model;
    int L = m->cur_exec_layer;
    int has_gu_lora = (m->active_lora > 0 && L >= 0 && L < MOE_MAX_LAYERS &&
                       (m->loras[m->active_lora].a[L][4] != nil || m->loras[m->active_lora].a[L][5] != nil));
    if (g_type == TN_GGML_Q4_0 && u_type == TN_GGML_Q4_0 && !has_gu_lora) {
        struct { uint32_t out_dim, in_dim, add_res; } args = { d_ff, dm, 0 };
        [enc setComputePipelineState:g_pso_dense_gu_q4];
        [enc setBuffer:g_buf offset:(NSUInteger)g_off atIndex:0];
        [enc setBuffer:u_buf offset:(NSUInteger)u_off atIndex:1];
        [enc setBuffer:m->buf_xn offset:0 atIndex:2];
        [enc setBuffer:m->buf_sh_g offset:0 atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((d_ff + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else if (g_type == TN_GGML_Q4_K && u_type == TN_GGML_Q4_K && !has_gu_lora && g_pso_dense_gu_q4k != nil) {
        struct { uint32_t out_dim, in_dim, add_res; } args = { d_ff, dm, 0 };
        [enc setComputePipelineState:g_pso_dense_gu_q4k];
        [enc setBuffer:g_buf offset:(NSUInteger)g_off atIndex:0];
        [enc setBuffer:u_buf offset:(NSUInteger)u_off atIndex:1];
        [enc setBuffer:m->buf_xn offset:0 atIndex:2];
        [enc setBuffer:m->buf_sh_g offset:0 atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((d_ff + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else if (g_type == TN_GGML_Q5_K && u_type == TN_GGML_Q5_K && !has_gu_lora && g_pso_dense_gu_q5k != nil) {
        struct { uint32_t out_dim, in_dim, add_res; } args = { d_ff, dm, 0 };
        [enc setComputePipelineState:g_pso_dense_gu_q5k];
        [enc setBuffer:g_buf offset:(NSUInteger)g_off atIndex:0];
        [enc setBuffer:u_buf offset:(NSUInteger)u_off atIndex:1];
        [enc setBuffer:m->buf_xn offset:0 atIndex:2];
        [enc setBuffer:m->buf_sh_g offset:0 atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((d_ff + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else {
        encode_gemv_raw(enc, g_buf, g_off, g_type, m->buf_xn, m->buf_sh_g, d_ff, dm, 0);
        if (has_gu_lora) encode_lora_delta(m, enc, L, 4, m->buf_xn, m->buf_sh_g, d_ff, dm, 1);
        encode_gemv_raw(enc, u_buf, u_off, u_type, m->buf_xn, m->buf_sh_u, d_ff, dm, 0);
        if (has_gu_lora) encode_lora_delta(m, enc, L, 5, m->buf_xn, m->buf_sh_u, d_ff, dm, 1);
        [enc setComputePipelineState:g_pso_swiglu_mul];
        [enc setBuffer:m->buf_sh_g offset:0 atIndex:0];
        [enc setBuffer:m->buf_sh_u offset:0 atIndex:1];
        [enc setBytes:&d_ff length:4 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(d_ff, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    encode_gemv_raw(enc, d_buf, d_off, d_type, m->buf_sh_g, m->buf_x, dm, d_ff, 1);
    if (m->active_lora > 0 && L >= 0) encode_lora_delta(m, enc, L, 6, m->buf_sh_g, m->buf_x, dm, d_ff, 1);
}

static void encode_layer_attn_body_ptr(
    MoeModel* m, id<MTLComputeCommandEncoder> enc, MoeLayer* ly, uint32_t pos,
    id<MTLBuffer> ping_buf // nil -> use ly->wq/wk/wv/wo pinned buffers; non-nil -> use ping_buf offsets!
) {
    int L = (ly >= &m->layers[0] && ly < &m->layers[MOE_MAX_LAYERS]) ? (int)(ly - &m->layers[0]) : -1;
    m->cur_exec_layer = L;
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
            [enc dispatchThreadgroups:MTLSizeMake((kv_r + 3u) / 4u, (NSUInteger)m->n_heads, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            struct { uint32_t nh, kvr, nd, rd, dh, kvl; float sc; } fl = {
                (uint32_t)m->n_heads, kv_r, nope_d, (uint32_t)m->rope_dim, (uint32_t)m->d_head, pos + 1u, attn_scale
            };
            [enc setComputePipelineState:g_pso_mla_flash_lat];
            [enc setBuffer:m->pbuf_mla_qabs offset:0 atIndex:0];
            [enc setBuffer:m->buf_q offset:0 atIndex:1];
            [enc setBuffer:ly->k_cache offset:lat_slot_off atIndex:2];
            [enc setBuffer:m->pbuf_mla_olat offset:0 atIndex:3];
            [enc setBytes:&fl length:sizeof(fl) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((m->n_heads + 3) / 4), 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

            struct { uint32_t nh, vd, kvr, os; } pv = { (uint32_t)m->n_heads, v_d, kv_r, out_stride };
            [enc setComputePipelineState:g_pso_mla_proj_v];
            [enc setBuffer:ly->mla_wuv_f16 offset:0 atIndex:0];
            [enc setBuffer:m->pbuf_mla_olat offset:0 atIndex:1];
            [enc setBuffer:m->buf_ctx offset:0 atIndex:2];
            [enc setBytes:&pv length:sizeof(pv) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((out_stride + 3u) / 4u), (NSUInteger)m->n_heads, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

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
    } else if (m->active_lora == 0 && g_pso_gemv_qkv_q4 != nil &&
               ly->wq.ggml_type == TN_GGML_Q4_0 && ly->wk.ggml_type == TN_GGML_Q4_0 && ly->wv.ggml_type == TN_GGML_Q4_0) {
        int fuse_b = (ly->has_qkv_bias && ly->bq.buf != nil && ly->bk.buf != nil && ly->bv.buf != nil);
        struct { uint32_t q_dim, kv_dim, in_dim, has_bias; } qa = { qdim, kvdim, dm, (uint32_t)fuse_b };
        id<MTLBuffer> qb = (ping_buf != nil) ? ping_buf : ly->wq.buf;
        id<MTLBuffer> kb = (ping_buf != nil) ? ping_buf : ly->wk.buf;
        id<MTLBuffer> vb = (ping_buf != nil) ? ping_buf : ly->wv.buf;
        NSUInteger qo = (NSUInteger)((ping_buf != nil) ? m->dense_off_wq : ly->wq.offset);
        NSUInteger ko = (NSUInteger)((ping_buf != nil) ? m->dense_off_wk : ly->wk.offset);
        NSUInteger vo = (NSUInteger)((ping_buf != nil) ? m->dense_off_wv : ly->wv.offset);
        [enc setComputePipelineState:g_pso_gemv_qkv_q4];
        [enc setBuffer:qb offset:qo atIndex:0];
        [enc setBuffer:kb offset:ko atIndex:1];
        [enc setBuffer:vb offset:vo atIndex:2];
        [enc setBuffer:m->buf_xn offset:0 atIndex:3];
        [enc setBuffer:m->buf_q offset:0 atIndex:4];
        [enc setBuffer:m->buf_k offset:0 atIndex:5];
        [enc setBuffer:m->buf_v offset:0 atIndex:6];
        [enc setBytes:&qa length:sizeof(qa) atIndex:7];
        [enc setBuffer:(fuse_b ? ly->bq.buf : m->buf_argmax) offset:(fuse_b ? (NSUInteger)ly->bq.offset : 0) atIndex:8];
        [enc setBuffer:(fuse_b ? ly->bk.buf : m->buf_argmax) offset:(fuse_b ? (NSUInteger)ly->bk.offset : 0) atIndex:9];
        [enc setBuffer:(fuse_b ? ly->bv.buf : m->buf_argmax) offset:(fuse_b ? (NSUInteger)ly->bv.offset : 0) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((qdim + 2u * kvdim + 7u) / 8u), 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (fuse_b) goto skip_qkv_bias_step;
    } else if (ping_buf != nil) {
        encode_gemv_raw(enc, ping_buf, m->dense_off_wq, ly->wq.ggml_type, m->buf_xn, m->buf_q, qdim, dm, 0);
        encode_gemv_raw(enc, ping_buf, m->dense_off_wk, ly->wk.ggml_type, m->buf_xn, m->buf_k, kvdim, dm, 0);
        encode_gemv_raw(enc, ping_buf, m->dense_off_wv, ly->wv.ggml_type, m->buf_xn, m->buf_v, kvdim, dm, 0);
    } else {
        encode_gemv(enc, &ly->wq, m->buf_xn, m->buf_q, qdim, dm, 0);
        encode_gemv(enc, &ly->wk, m->buf_xn, m->buf_k, kvdim, dm, 0);
        encode_gemv(enc, &ly->wv, m->buf_xn, m->buf_v, kvdim, dm, 0);
    }
    if (m->active_lora > 0 && L >= 0) {
        encode_lora_delta(m, enc, L, 0, m->buf_xn, m->buf_q, qdim, dm, 1);
        encode_lora_delta(m, enc, L, 1, m->buf_xn, m->buf_k, kvdim, dm, 1);
        encode_lora_delta(m, enc, L, 2, m->buf_xn, m->buf_v, kvdim, dm, 1);
    }
    if (ly->has_qkv_bias) {
        encode_add_bias(enc, m->buf_q, &ly->bq, qdim);
        encode_add_bias(enc, m->buf_k, &ly->bk, kvdim);
        encode_add_bias(enc, m->buf_v, &ly->bv, kvdim);
    }
skip_qkv_bias_step:;
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
        uint32_t half_r_u = eff_rope_dim >> 1;
        uint32_t ms0 = (m->mrope_sections[0] > 0) ? (uint32_t)m->mrope_sections[0] : (half_r_u / 3u);
        uint32_t ms1 = (m->mrope_sections[1] > 0) ? (uint32_t)m->mrope_sections[1] : (half_r_u / 3u);
        struct { uint32_t n_heads, n_kv, d_head, rope_dim, pos; float theta; uint32_t rope_neox, has_freqs, mrope_en, mrope_s0, mrope_s1, mrope_t, mrope_h, mrope_w; } rargs = {
            (uint32_t)m->n_heads, (uint32_t)m->n_kv, (uint32_t)m->d_head, eff_rope_dim,
            pos, m->rope_theta, (uint32_t)m->rope_neox, (uint32_t)m->has_rope_freqs,
            (uint32_t)(m->has_mrope == 2 ? 1 : 0), ms0, ms1,
            (uint32_t)(m->mrope_pos[0] >= 0 ? m->mrope_pos[0] : (int)pos),
            (uint32_t)(m->mrope_pos[1] >= 0 ? m->mrope_pos[1] : (int)pos),
            (uint32_t)(m->mrope_pos[2] >= 0 ? m->mrope_pos[2] : (int)pos)
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
    if (m->active_lora > 0 && L >= 0) {
        encode_lora_delta(m, enc, L, 3, m->buf_ctx, m->buf_x, dm, wo_in, 1);
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
    m->batch_logits_dirty_row = -1;
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
        *out_tgs = (NSUInteger)((d_model + 3) / 4); *out_tptg = is_slot ? 128 : 64;
    } else if (gtype == TN_GGML_Q5_0) {
        *out_pso = is_slot ? g_pso_moe_slot_down_q5_0 : g_pso_moe_down_q5_0;
        *out_tgs = (NSUInteger)(is_slot ? ((d_model + 3) / 4) : ((d_model + 7) / 8)); *out_tptg = is_slot ? 128 : 64;
    } else if (gtype == TN_GGML_Q5_1) {
        *out_pso = is_slot ? g_pso_moe_slot_down_q5_1 : g_pso_moe_down_q5_1;
        *out_tgs = (NSUInteger)((d_model + 3) / 4); *out_tptg = 128;
    } else if (gtype == TN_GGML_Q8_0) {
        *out_pso = is_slot ? g_pso_moe_slot_down_q8 : g_pso_moe_down_q8;
        *out_tgs = (NSUInteger)(is_slot ? ((d_model + 3) / 4) : ((d_model + 7) / 8)); *out_tptg = is_slot ? 128 : 64;
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
                        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

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
        case 31: return m->active_lora;
        case 32: { int cnt = 0; for (int i = 1; i < MOE_MAX_LORAS; i++) if (m->loras[i].active) cnt++; return cnt; }
        case 33: return m->has_mrope;
        case 34: {
            int64_t gpu_sv = (m->buf_moe_meta != nil) ? (int64_t)((const uint32_t*)[m->buf_moe_meta contents])[1] : 0;
            return m->moe_grouped_saved_reads + gpu_sv;
        }
        case 35: return (m->buf_moe_meta != nil) ? (int64_t)((const uint32_t*)[m->buf_moe_meta contents])[2] : 0;
        case 36: return 1; // Apple AMX simdgroup_half8x8 -> float8x8 active
        case 37: return m->kv_quant_q8; // 0=FP16, 1=Q8_0, 2=FP8_E4M3, 3=Q4_0
        case 38: return m->sliding_window;
        case 39: return (m->buf_moe_meta != nil) ? (int64_t)((const uint32_t*)[m->buf_moe_meta contents])[3] : 0;
        case 40: return m->icb_replays;
        case 41: return m->icb_captured;
        case 42: return m->attn_sink;
        case 43: return m->flash_decode_launches;
        case 44: return m->icb_pipe_overlaps;
        case 45: return (m->buf_moe_meta != nil) ? (int64_t)((const uint32_t*)[m->buf_moe_meta contents])[3] : 0;
        default: return 0;
    }
}

// =========================================================================
// Batched Causal Prompt Prefill (`tn_moe_prefill`), Auto Planner (`tn_moe_auto_plan`),
// Full Sampler (`tn_moe_sample_next`), Architecture Query (`tn_moe_get_arch`)!
// =========================================================================
static void icb_cache_clear(MoeModel* m);
static void ensure_prefill_bufs(MoeModel* m, int B) {
    if (m->prefill_cap >= B) return;
    icb_cache_clear(m);
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
    if (m->n_layers > 0 && m->layers[0].is_mla) {
        size_t kv_r = (size_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
        size_t qr = (size_t)(m->q_lora_rank > 0 ? m->q_lora_rank : m->d_model);
        size_t lat_d = kv_r + (size_t)m->rope_dim;
        size_t nope_d = (size_t)(m->d_head - m->rope_dim);
        size_t v_d = (size_t)m->v_head_dim;
        size_t kv_sz = lat_d > (size_t)m->n_heads * (nope_d + v_d) ? lat_d : (size_t)m->n_heads * (nope_d + v_d);
        m->pbuf_mla_q    = [g_moe_dev newBufferWithLength:(size_t)cap * qr * 4 options:MTLResourceStorageModeShared];
        m->pbuf_mla_kv   = [g_moe_dev newBufferWithLength:(size_t)cap * (lat_d + kv_sz) * 4 options:MTLResourceStorageModeShared];
        m->pbuf_mla_qabs = [g_moe_dev newBufferWithLength:(size_t)cap * (size_t)m->n_heads * kv_r * 4 options:MTLResourceStorageModeShared];
        m->pbuf_mla_olat = [g_moe_dev newBufferWithLength:(size_t)cap * (size_t)m->n_heads * kv_r * 4 options:MTLResourceStorageModeShared];
    }
    {
        size_t fd_cap = (size_t)(cap < MOE_FD_MAX_B ? cap : MOE_FD_MAX_B);
        size_t fd_sp = (size_t)MOE_FD_MAX_SPLITS;
        size_t fd_v = (size_t)m->d_head;
        if (m->n_layers > 0 && m->layers[0].is_mla && (size_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512) > fd_v) {
            fd_v = (size_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
        }
        size_t n_sp_slots = fd_cap * (size_t)m->n_heads * fd_sp;
        m->buf_fd_o = [g_moe_dev newBufferWithLength:n_sp_slots * fd_v * 4 options:MTLResourceStorageModeShared];
        m->buf_fd_m = [g_moe_dev newBufferWithLength:n_sp_slots * 4 options:MTLResourceStorageModeShared];
        m->buf_fd_l = [g_moe_dev newBufferWithLength:n_sp_slots * 4 options:MTLResourceStorageModeShared];
    }
    if (m->n_experts > 0) {
        size_t tk = (size_t)(m->top_k > 0 ? m->top_k : 8);
        size_t dfe = (size_t)(m->d_ff_exp > 0 ? m->d_ff_exp : m->d_ff_dense);
        m->pbuf_rlogits     = [g_moe_dev newBufferWithLength:(size_t)cap * (size_t)m->n_experts * 4 options:MTLResourceStorageModeShared];
        m->pbuf_moe_x       = [g_moe_dev newBufferWithLength:(size_t)cap * dm * 4 options:MTLResourceStorageModeShared];
        m->pbuf_moe_h       = [g_moe_dev newBufferWithLength:(size_t)cap * tk * dfe * 4 options:MTLResourceStorageModeShared];
        m->pbuf_moe_y       = [g_moe_dev newBufferWithLength:(size_t)cap * tk * dm * 4 options:MTLResourceStorageModeShared];
        m->buf_moe_idx      = [g_moe_dev newBufferWithLength:(size_t)cap * MOE_MAX_TOPK * 4 options:MTLResourceStorageModeShared];
        m->buf_moe_wt       = [g_moe_dev newBufferWithLength:(size_t)cap * MOE_MAX_TOPK * 4 options:MTLResourceStorageModeShared];
        m->buf_moe_active_e = [g_moe_dev newBufferWithLength:(size_t)MOE_MAX_EXPERTS * 4 options:MTLResourceStorageModeShared];
        m->buf_moe_grp_off  = [g_moe_dev newBufferWithLength:(size_t)MOE_MAX_EXPERTS * 4 options:MTLResourceStorageModeShared];
        m->buf_moe_grp_cnt  = [g_moe_dev newBufferWithLength:(size_t)MOE_MAX_EXPERTS * 4 options:MTLResourceStorageModeShared];
        m->buf_moe_tok_slot = [g_moe_dev newBufferWithLength:(size_t)cap * MOE_MAX_TOPK * 4 options:MTLResourceStorageModeShared];
        if (m->buf_moe_meta == nil) {
            m->buf_moe_meta = [g_moe_dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
            memset([m->buf_moe_meta contents], 0, 16);
        }
        if (m->buf_moe_zero == nil) {
            m->buf_moe_zero = [g_moe_dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
            memset([m->buf_moe_zero contents], 0, 64);
        }
    }
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
            if (B <= 4u || (B <= 8u && out_dim < 4096u)) { pso = g_pso_gemm_q4_sb;    tile_n = 8;  tile_b = 8;  tptg = 256; }
            else if (B <= 8u)                            { pso = g_pso_gemm_q4_amx8;  tile_n = 64; tile_b = 8;  tptg = 128; }
            else if (B <= 32u)                           { pso = g_pso_gemm_q4_amx16; tile_n = 32; tile_b = 16; tptg = 128; }
            else                                         { pso = g_pso_gemm_q4_b;     tile_n = 64; tile_b = 32; tptg = 128; }
            break;
        case TN_GGML_Q4_1:
            if (B <= 4u || (B <= 8u && out_dim < 4096u)) { pso = g_pso_gemm_q4_1_sb;   tile_n = 8;  tile_b = 8;  tptg = 256; }
            else if (B <= 8u)                            { pso = g_pso_gemm_q4_1_amx8; tile_n = 64; tile_b = 8;  tptg = 128; }
            else                                         { pso = g_pso_gemm_q4_1_b;    tile_n = 64; tile_b = 32; tptg = 128; }
            break;
        case TN_GGML_Q5_0:   pso = g_pso_gemm_q5_0_b;  tile_n = 4;  tile_b = 8;  break;
        case TN_GGML_Q5_1:   pso = g_pso_gemm_q5_1_b;  tile_n = 4;  tile_b = 8;  break;
        case TN_GGML_Q8_0:
            if (B <= 8u) { pso = g_pso_gemm_q8_b;   tile_n = 4;  tile_b = 8;  tptg = 128; }
            else         { pso = g_pso_gemm_q8_amx; tile_n = 32; tile_b = 32; tptg = 128; }
            break;
        case TN_GGML_Q2_K:   pso = g_pso_gemm_q2k_b;   tile_n = 4;  tile_b = 4;  break;
        case TN_GGML_Q3_K:   pso = g_pso_gemm_q3k_b;   tile_n = 4;  tile_b = 4;  break;
        case TN_GGML_IQ4_XS: pso = g_pso_gemm_iq4xs_b; tile_n = 4;  tile_b = 4;  break;
        case TN_GGML_Q4_K:
            if (B <= 4u)       { pso = g_pso_gemm_q4k_sb;    tile_n = 4;  tile_b = 8;  tptg = 128; }
            else if (B <= 32u) { pso = g_pso_gemm_q4k_amx16; tile_n = 32; tile_b = 16; tptg = 128; }
            else               { pso = g_pso_gemm_q4k_b;     tile_n = 64; tile_b = 64; tptg = 128; }
            break;
        case TN_GGML_Q5_K:   pso = g_pso_gemm_q5k_b;   tile_n = 4;  tile_b = 4;  break;
        case TN_GGML_Q6_K:
            if (B <= 4u)       { pso = g_pso_gemm_q6k_sb;    tile_n = 4;  tile_b = 8;  tptg = 128; }
            else if (B <= 32u) { pso = g_pso_gemm_q6k_amx16; tile_n = 16; tile_b = 16; tptg = 128; }
            else               { pso = g_pso_gemm_q6k_amx;   tile_n = 64; tile_b = 64; tptg = 128; }
            break;
        case TN_GGML_FP8_E4M3:
            if (B <= 8u) { pso = g_pso_gemm_fp8_b;   tile_n = 4;  tile_b = 8;  tptg = 128; }
            else         { pso = g_pso_gemm_fp8_amx; tile_n = 32; tile_b = 32; tptg = 128; }
            break;
        case TN_GGML_MXFP4:
            if (B <= 8u) { pso = g_pso_gemm_mxfp4_b;   tile_n = 4;  tile_b = 8;  tptg = 128; }
            else         { pso = g_pso_gemm_mxfp4_amx; tile_n = 32; tile_b = 32; tptg = 128; }
            break;
        case TN_GGML_F16:
            if (B <= 8u || g_pso_gemm_f16_amx == nil) { pso = g_pso_gemm_f16_b;   tile_n = 4;  tile_b = 8;  tptg = 128; }
            else                                      { pso = g_pso_gemm_f16_amx; tile_n = 64; tile_b = 64; tptg = 128; }
            break;
        default:               pso = g_pso_gemm_f32_b;   tile_n = 4; tile_b = 8; break;
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
    m->cur_exec_layer = L;
    MoeLayer* ly = &m->layers[L];
    uint32_t dm = (uint32_t)m->d_model, qdim = (uint32_t)(m->n_heads * m->d_head), kvdim = (uint32_t)(m->n_kv * m->d_head);
    encode_rmsnorm(enc, m->pbuf_x, &ly->attn_norm, m->pbuf_xn, dm, B, m->rms_eps, 1);
    int has_qkv_lora = (m->active_lora > 0 || m->use_batch_lora);
    int qkv_bias_fused = 0;
    if (B == 1u && !has_qkv_lora && g_pso_gemv_qkv_q4 != nil &&
        ly->wq.ggml_type == TN_GGML_Q4_0 && ly->wk.ggml_type == TN_GGML_Q4_0 && ly->wv.ggml_type == TN_GGML_Q4_0) {
        // Tenon v13: Single-dispatch fused Q||K||V + Optional QKV Bias for B=1 Paged Decode (saves 2-5 dispatches/layer!)
        qkv_bias_fused = (ly->has_qkv_bias && ly->bq.buf != nil && ly->bk.buf != nil && ly->bv.buf != nil);
        struct { uint32_t q_dim, kv_dim, in_dim, has_bias; } qa = { qdim, kvdim, dm, (uint32_t)qkv_bias_fused };
        id<MTLBuffer> qb = (ping_buf != nil) ? ping_buf : ly->wq.buf;
        id<MTLBuffer> kb = (ping_buf != nil) ? ping_buf : ly->wk.buf;
        id<MTLBuffer> vb = (ping_buf != nil) ? ping_buf : ly->wv.buf;
        NSUInteger qo = (NSUInteger)((ping_buf != nil) ? m->dense_off_wq : ly->wq.offset);
        NSUInteger ko = (NSUInteger)((ping_buf != nil) ? m->dense_off_wk : ly->wk.offset);
        NSUInteger vo = (NSUInteger)((ping_buf != nil) ? m->dense_off_wv : ly->wv.offset);
        [enc setComputePipelineState:g_pso_gemv_qkv_q4];
        [enc setBuffer:qb offset:qo atIndex:0];
        [enc setBuffer:kb offset:ko atIndex:1];
        [enc setBuffer:vb offset:vo atIndex:2];
        [enc setBuffer:m->pbuf_xn offset:0 atIndex:3];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:4];
        [enc setBuffer:m->pbuf_k offset:0 atIndex:5];
        [enc setBuffer:m->pbuf_v offset:0 atIndex:6];
        [enc setBytes:&qa length:sizeof(qa) atIndex:7];
        [enc setBuffer:(qkv_bias_fused ? ly->bq.buf : m->buf_argmax) offset:(qkv_bias_fused ? (NSUInteger)ly->bq.offset : 0) atIndex:8];
        [enc setBuffer:(qkv_bias_fused ? ly->bk.buf : m->buf_argmax) offset:(qkv_bias_fused ? (NSUInteger)ly->bk.offset : 0) atIndex:9];
        [enc setBuffer:(qkv_bias_fused ? ly->bv.buf : m->buf_argmax) offset:(qkv_bias_fused ? (NSUInteger)ly->bv.offset : 0) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((qdim + 2u * kvdim + 7u) / 8u), 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else if (B >= 33u && !has_qkv_lora && ping_buf == nil && g_pso_gemm_q4_qkv_b != nil &&
               ly->wq.ggml_type == TN_GGML_Q4_0 && ly->wk.ggml_type == TN_GGML_Q4_0 && ly->wv.ggml_type == TN_GGML_Q4_0 &&
               (qdim & 63u) == 0u && (kvdim & 63u) == 0u) {
        struct { uint32_t q_dim, kv_dim, in_dim, batch_b; } qa = { qdim, kvdim, dm, B };
        [enc setComputePipelineState:g_pso_gemm_q4_qkv_b];
        [enc setBuffer:ly->wq.buf offset:(NSUInteger)ly->wq.offset atIndex:0];
        [enc setBuffer:ly->wk.buf offset:(NSUInteger)ly->wk.offset atIndex:1];
        [enc setBuffer:ly->wv.buf offset:(NSUInteger)ly->wv.offset atIndex:2];
        [enc setBuffer:m->pbuf_xn offset:0 atIndex:3];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:4];
        [enc setBuffer:m->pbuf_k offset:0 atIndex:5];
        [enc setBuffer:m->pbuf_v offset:0 atIndex:6];
        [enc setBytes:&qa length:sizeof(qa) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((qdim + 2u * kvdim + 63u) / 64u), (NSUInteger)((B + 31u) / 32u), 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (ping_buf != nil) {
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wq, ly->wq.ggml_type, m->pbuf_xn, m->pbuf_q, qdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wk, ly->wk.ggml_type, m->pbuf_xn, m->pbuf_k, kvdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wv, ly->wv.ggml_type, m->pbuf_xn, m->pbuf_v, kvdim, dm, B, 0);
    } else {
        encode_gemm_batch_raw(enc, ly->wq.buf, ly->wq.offset, ly->wq.ggml_type, m->pbuf_xn, m->pbuf_q, qdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ly->wk.buf, ly->wk.offset, ly->wk.ggml_type, m->pbuf_xn, m->pbuf_k, kvdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ly->wv.buf, ly->wv.offset, ly->wv.ggml_type, m->pbuf_xn, m->pbuf_v, kvdim, dm, B, 0);
    }
    if (m->active_lora > 0 || m->use_batch_lora) {
        encode_lora_delta(m, enc, L, 0, m->pbuf_xn, m->pbuf_q, qdim, dm, B);
        encode_lora_delta(m, enc, L, 1, m->pbuf_xn, m->pbuf_k, kvdim, dm, B);
        encode_lora_delta(m, enc, L, 2, m->pbuf_xn, m->pbuf_v, kvdim, dm, B);
    }
    if (ly->has_qkv_bias && !qkv_bias_fused) {
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
    if (m->active_lora > 0 || m->use_batch_lora) encode_lora_delta(m, enc, L, 3, m->pbuf_ctx, m->pbuf_x, dm, qdim, B);
    encode_rmsnorm(enc, m->pbuf_x, &ly->ffn_norm, m->pbuf_xn, dm, B, m->rms_eps, 1);
}

static void encode_dense_ffn_batch(MoeModel* m, id<MTLComputeCommandEncoder> enc,
                                   id<MTLBuffer> gb, uint64_t go, uint32_t gt,
                                   id<MTLBuffer> ub, uint64_t uo, uint32_t ut,
                                   id<MTLBuffer> db, uint64_t doff, uint32_t dt, uint32_t d_ff, uint32_t B) {
    uint32_t dm = (uint32_t)m->d_model;
    int L = m->cur_exec_layer;
    int has_gu_lora = ((m->active_lora > 0 || m->use_batch_lora) && L >= 0);
    if (B == 1u && gt == TN_GGML_Q4_0 && ut == TN_GGML_Q4_0 && !has_gu_lora) {
        // Tenon v12: Single-dispatch fused Gate+Up+SwiGLU for B=1 Paged Decode (saves 2 dispatches & pbuf_u write/read!)
        struct { uint32_t out_dim, in_dim, add_res; } args = { d_ff, dm, 0 };
        [enc setComputePipelineState:g_pso_dense_gu_q4];
        [enc setBuffer:gb offset:(NSUInteger)go atIndex:0];
        [enc setBuffer:ub offset:(NSUInteger)uo atIndex:1];
        [enc setBuffer:m->pbuf_xn offset:0 atIndex:2];
        [enc setBuffer:m->pbuf_g offset:0 atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((d_ff + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else if (B == 1u && gt == TN_GGML_Q4_K && ut == TN_GGML_Q4_K && !has_gu_lora && g_pso_dense_gu_q4k != nil) {
        struct { uint32_t out_dim, in_dim, add_res; } args = { d_ff, dm, 0 };
        [enc setComputePipelineState:g_pso_dense_gu_q4k];
        [enc setBuffer:gb offset:(NSUInteger)go atIndex:0];
        [enc setBuffer:ub offset:(NSUInteger)uo atIndex:1];
        [enc setBuffer:m->pbuf_xn offset:0 atIndex:2];
        [enc setBuffer:m->pbuf_g offset:0 atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((d_ff + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else if (B == 1u && gt == TN_GGML_Q5_K && ut == TN_GGML_Q5_K && !has_gu_lora && g_pso_dense_gu_q5k != nil) {
        struct { uint32_t out_dim, in_dim, add_res; } args = { d_ff, dm, 0 };
        [enc setComputePipelineState:g_pso_dense_gu_q5k];
        [enc setBuffer:gb offset:(NSUInteger)go atIndex:0];
        [enc setBuffer:ub offset:(NSUInteger)uo atIndex:1];
        [enc setBuffer:m->pbuf_xn offset:0 atIndex:2];
        [enc setBuffer:m->pbuf_g offset:0 atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((d_ff + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else if (B >= 33u && gt == TN_GGML_Q4_0 && ut == TN_GGML_Q4_0 && !has_gu_lora && g_pso_gemm_q4_gu_b != nil) {
        struct { uint32_t out_dim, in_dim, batch_b, add_res; } args = { d_ff, dm, B, 0 };
        [enc setComputePipelineState:g_pso_gemm_q4_gu_b];
        [enc setBuffer:gb offset:(NSUInteger)go atIndex:0];
        [enc setBuffer:ub offset:(NSUInteger)uo atIndex:1];
        [enc setBuffer:m->pbuf_xn offset:0 atIndex:2];
        [enc setBuffer:m->pbuf_g offset:0 atIndex:3];
        [enc setBytes:&args length:sizeof(args) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((d_ff + 31u) / 32u, (B + 31u) / 32u, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        encode_gemm_batch_raw(enc, gb, go, gt, m->pbuf_xn, m->pbuf_g, d_ff, dm, B, 0);
        if (has_gu_lora) encode_lora_delta(m, enc, L, 4, m->pbuf_xn, m->pbuf_g, d_ff, dm, B);
        encode_gemm_batch_raw(enc, ub, uo, ut, m->pbuf_xn, m->pbuf_u, d_ff, dm, B, 0);
        if (has_gu_lora) encode_lora_delta(m, enc, L, 5, m->pbuf_xn, m->pbuf_u, d_ff, dm, B);
        uint32_t tot = B * d_ff;
        [enc setComputePipelineState:g_pso_swiglu_mul];
        [enc setBuffer:m->pbuf_g offset:0 atIndex:0]; [enc setBuffer:m->pbuf_u offset:0 atIndex:1]; [enc setBytes:&tot length:4 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(tot, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    encode_gemm_batch_raw(enc, db, doff, dt, m->pbuf_g, m->pbuf_x, dm, d_ff, B, 1);
    if (has_gu_lora) encode_lora_delta(m, enc, L, 6, m->pbuf_g, m->pbuf_x, dm, d_ff, B);
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
    if (m->batch_logits_dirty_row >= 0 && m->buf_batch_logits != nil && m->batch_logits_dirty_row < m->batch_logits_cap) {
        const float* bl = (const float*)[m->buf_batch_logits contents];
        memcpy([m->buf_logits contents], bl + (size_t)m->batch_logits_dirty_row * (size_t)m->vocab_size, (size_t)m->vocab_size * 4u);
        m->batch_logits_dirty_row = -1;
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
    if (m->batch_logits_dirty_row >= 0 && m->buf_batch_logits != nil && m->batch_logits_dirty_row < m->batch_logits_cap) {
        const float* bl = (const float*)[m->buf_batch_logits contents];
        memcpy([m->buf_logits contents], bl + (size_t)m->batch_logits_dirty_row * (size_t)m->vocab_size, (size_t)m->vocab_size * 4u);
        m->batch_logits_dirty_row = -1;
    }
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

// Tenon v10: Zero-Duplication Metal MTLIndirectCommandBuffer (ICB) Persistent Decode Graph Recorder!
// Implements the exact 5 compute-encoder methods so ALL existing encode_* functions capture straight into hardware ICB!
@interface TenonIcbRecorder : NSObject {
@public
    id<MTLIndirectCommandBuffer> icb;
    id<MTLBuffer> arg_buf;
    uint8_t* arg_ptr;
    size_t arg_off;
    size_t arg_cap;
    uint32_t n_cmds;
    uint32_t cmd_cap;
    id<MTLComputePipelineState> cur_pso;
    id<MTLBuffer> cur_buf[16];
    NSUInteger cur_off[16];
    uint32_t cur_mask;
    id<MTLBuffer> __strong * res_list;
    int res_count;
    int res_cap;
    int overflow;
}
- (instancetype)initWithDevice:(id<MTLDevice>)dev maxCommands:(uint32_t)mc;
- (void)addRes:(id<MTLBuffer>)buf;
- (void)setComputePipelineState:(id<MTLComputePipelineState>)pso;
- (void)setBuffer:(id<MTLBuffer>)buf offset:(NSUInteger)off atIndex:(NSUInteger)idx;
- (void)setBytes:(const void*)bytes length:(NSUInteger)len atIndex:(NSUInteger)idx;
- (void)dispatchThreadgroups:(MTLSize)tg threadsPerThreadgroup:(MTLSize)tptg;
- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)tptg;
- (void)endEncoding;
@end

@implementation TenonIcbRecorder
- (instancetype)initWithDevice:(id<MTLDevice>)dev maxCommands:(uint32_t)mc {
    self = [super init];
    if (!self) return nil;
    MTLIndirectCommandBufferDescriptor* d = [[MTLIndirectCommandBufferDescriptor alloc] init];
    d.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    d.inheritBuffers = NO;
    d.inheritPipelineState = NO;
    d.maxKernelBufferBindCount = 16;
    icb = [dev newIndirectCommandBufferWithDescriptor:d maxCommandCount:mc options:MTLResourceStorageModeShared];
    if (!icb) return nil;
    cmd_cap = mc;
    arg_cap = (size_t)mc * 256u;
    arg_buf = [dev newBufferWithLength:arg_cap options:MTLResourceStorageModeShared];
    arg_ptr = (uint8_t*)[arg_buf contents];
    arg_off = 0;
    n_cmds = 0;
    cur_mask = 0;
    res_cap = 256;
    res_count = 0;
    res_list = (id<MTLBuffer> __strong *)calloc((size_t)res_cap, sizeof(id<MTLBuffer>));
    overflow = 0;
    [self addRes:arg_buf];
    return self;
}
- (void)addRes:(id<MTLBuffer>)buf {
    if (!buf) return;
    for (int i = 0; i < res_count; i++) if (res_list[i] == buf) return;
    if (res_count >= res_cap) {
        int ncap = res_cap * 2;
        id<MTLBuffer> __strong * nl = (id<MTLBuffer> __strong *)calloc((size_t)ncap, sizeof(id<MTLBuffer>));
        for (int i = 0; i < res_count; i++) { nl[i] = res_list[i]; res_list[i] = nil; }
        free(res_list);
        res_list = nl; res_cap = ncap;
    }
    res_list[res_count++] = buf;
}
- (void)setComputePipelineState:(id<MTLComputePipelineState>)pso { cur_pso = pso; }
- (void)setBuffer:(id<MTLBuffer>)buf offset:(NSUInteger)off atIndex:(NSUInteger)idx {
    if (idx < 16u) { cur_buf[idx] = buf; cur_off[idx] = off; cur_mask |= (1u << idx); [self addRes:buf]; }
}
- (void)setBytes:(const void*)bytes length:(NSUInteger)len atIndex:(NSUInteger)idx {
    size_t aligned = (len + 63u) & ~(size_t)63u;
    if (arg_off + aligned > arg_cap) { overflow = 1; return; }
    memcpy(arg_ptr + arg_off, bytes, len);
    if (idx < 16u) { cur_buf[idx] = arg_buf; cur_off[idx] = (NSUInteger)arg_off; cur_mask |= (1u << idx); }
    arg_off += aligned;
}
- (void)dispatchThreadgroups:(MTLSize)tg threadsPerThreadgroup:(MTLSize)tptg {
    if (n_cmds >= cmd_cap || !cur_pso || !cur_pso.supportIndirectCommandBuffers) { overflow = 1; return; }
    id<MTLIndirectComputeCommand> ic = [icb indirectComputeCommandAtIndex:n_cmds];
    [ic setBarrier];
    [ic setComputePipelineState:cur_pso];
    for (uint32_t i = 0; i < 16u; i++) {
        if ((cur_mask >> i) & 1u) [ic setKernelBuffer:cur_buf[i] offset:cur_off[i] atIndex:i];
    }
    [ic concurrentDispatchThreadgroups:tg threadsPerThreadgroup:tptg];
    n_cmds++;
    cur_mask = 0u;
}
- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)tptg {
    MTLSize tg = MTLSizeMake(
        (threads.width  + tptg.width  - 1u) / tptg.width,
        (threads.height + tptg.height - 1u) / tptg.height,
        (threads.depth  + tptg.depth  - 1u) / tptg.depth
    );
    [self dispatchThreadgroups:tg threadsPerThreadgroup:tptg];
}
- (void)endEncoding {}
@end

static void icb_cache_clear(MoeModel* m) {
    for (int i = 0; i < MOE_ICB_CACHE_MAX; i++) {
        if (m->icb_cache[i].res_list) {
            for (int r = 0; r < m->icb_cache[i].res_count; r++) m->icb_cache[i].res_list[r] = nil;
            free(m->icb_cache[i].res_list);
            m->icb_cache[i].res_list = NULL;
        }
        m->icb_cache[i].icb = nil;
        m->icb_cache[i].arg_buf = nil;
        m->icb_cache[i].valid = 0;
        m->icb_cache[i].res_count = 0;
    }
}

static size_t pa_kv_block_bytes(MoeModel* m) {
    if (m->n_layers > 0 && m->layers[0].is_mla && m->mla_latent_kv) {
        size_t kv_r = (size_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
        return (size_t)PA_BLOCK_SIZE * (kv_r + (size_t)m->rope_dim) * 2u;
    }
    if (m->kv_quant_q8 == 3) {
        return (size_t)PA_BLOCK_SIZE * (size_t)m->n_kv * (size_t)(m->d_head >> 5) * 18u;
    }
    return (m->kv_quant_q8 >= 1)
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
    m->cur_exec_layer = L;
    MoeLayer* ly = &m->layers[L];
    uint32_t dm = (uint32_t)m->d_model, qdim = (uint32_t)(m->n_heads * m->d_head), kvdim = (uint32_t)(m->n_kv * m->d_head);
    uint32_t wo_in = (ly->wo.ne0 > 0) ? (uint32_t)ly->wo.ne0 : qdim;
    uint32_t out_stride = wo_in / (uint32_t)m->n_heads;
    float attn_scale = (m->attn_mscale > 0.0f ? m->attn_mscale : 1.0f) / sqrtf((float)m->d_head);
    encode_rmsnorm(enc, m->pbuf_x, &ly->attn_norm, m->pbuf_xn, dm, B, m->rms_eps, 1);

    // Tenon v8: Native Batched PagedAttention Compressed Latent MLA (DeepSeek-V2/V3/R1)!
    if (ly->is_mla && m->mla_latent_kv && ly->mla_wuk_f16 != nil) {
        uint32_t kv_r = (uint32_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
        uint32_t nope_d = (uint32_t)(m->d_head - m->rope_dim);
        uint32_t v_d = (uint32_t)m->v_head_dim;
        uint32_t lat_d = kv_r + (uint32_t)m->rope_dim;
        if (ly->wq_a.buf != nil && ly->wq_b.buf != nil && m->q_lora_rank > 0) {
            encode_gemm_batch_raw(enc, ly->wq_a.buf, ly->wq_a.offset, ly->wq_a.ggml_type, m->pbuf_xn, m->pbuf_mla_q, (uint32_t)m->q_lora_rank, dm, B, 0);
            if (ly->q_a_norm.buf != nil) encode_rmsnorm(enc, m->pbuf_mla_q, &ly->q_a_norm, m->pbuf_mla_q, (uint32_t)m->q_lora_rank, B, m->rms_eps, 1);
            encode_gemm_batch_raw(enc, ly->wq_b.buf, ly->wq_b.offset, ly->wq_b.ggml_type, m->pbuf_mla_q, m->pbuf_q, qdim, (uint32_t)m->q_lora_rank, B, 0);
        } else {
            encode_gemm_batch_raw(enc, ly->wq.buf, ly->wq.offset, ly->wq.ggml_type, m->pbuf_xn, m->pbuf_q, qdim, dm, B, 0);
        }
        encode_gemm_batch_raw(enc, ly->wkv_a.buf, ly->wkv_a.offset, ly->wkv_a.ggml_type, m->pbuf_xn, m->pbuf_mla_kv, lat_d, dm, B, 0);
        if (ly->kv_a_norm.buf != nil) {
            if (g_pso_rmsnorm_strided != nil && B > 1u) {
                struct { uint32_t dim, stride, n_groups; float eps; } sn = { kv_r, lat_d, B, m->rms_eps };
                [enc setComputePipelineState:g_pso_rmsnorm_strided];
                [enc setBuffer:m->pbuf_mla_kv offset:0 atIndex:0];
                [enc setBuffer:ly->kv_a_norm.buf offset:(NSUInteger)ly->kv_a_norm.offset atIndex:1];
                [enc setBytes:&sn length:sizeof(sn) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                for (uint32_t bi = 0; bi < B; bi++) {
                    struct { uint32_t dim, n_groups; float eps; uint32_t shared_w; } nargs = { kv_r, 1u, m->rms_eps, 0u };
                    [enc setComputePipelineState:g_pso_rmsnorm];
                    [enc setBuffer:m->pbuf_mla_kv offset:(NSUInteger)((size_t)bi * lat_d * 4u) atIndex:0];
                    [enc setBuffer:ly->kv_a_norm.buf offset:(NSUInteger)ly->kv_a_norm.offset atIndex:1];
                    [enc setBuffer:m->pbuf_mla_kv offset:(NSUInteger)((size_t)bi * lat_d * 4u) atIndex:2];
                    [enc setBytes:&nargs length:sizeof(nargs) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
            }
        }
        struct { uint32_t nh, nd, rd, dh, kvr; float theta; uint32_t rn, hf, bb, mb; } mr = {
            (uint32_t)m->n_heads, nope_d, (uint32_t)m->rope_dim, (uint32_t)m->d_head, kv_r,
            m->rope_theta, (uint32_t)m->rope_neox, (uint32_t)m->has_rope_freqs, B, (uint32_t)PA_MAX_LOGICAL_BLKS
        };
        [enc setComputePipelineState:g_pso_mla_rope_store_paged];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:0];
        [enc setBuffer:m->pbuf_mla_kv offset:0 atIndex:1];
        [enc setBuffer:ly->k_cache offset:0 atIndex:2];
        [enc setBuffer:(m->has_rope_freqs ? m->rope_freqs.buf : m->buf_argmax)
                offset:(m->has_rope_freqs ? (NSUInteger)m->rope_freqs.offset : 0) atIndex:3];
        [enc setBuffer:m->buf_cb_pos offset:0 atIndex:4];
        [enc setBuffer:m->buf_cb_slot offset:0 atIndex:5];
        [enc setBuffer:m->buf_pa_bt offset:0 atIndex:6];
        [enc setBytes:&mr length:sizeof(mr) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        if (B == 1u && g_pso_mla_absorb_q != nil) {
            struct { uint32_t nh, nd, kvr, dh; } ab1 = { (uint32_t)m->n_heads, nope_d, kv_r, (uint32_t)m->d_head };
            [enc setComputePipelineState:g_pso_mla_absorb_q];
            [enc setBuffer:ly->mla_wuk_f16 offset:0 atIndex:0];
            [enc setBuffer:m->pbuf_q offset:0 atIndex:1];
            [enc setBuffer:m->pbuf_mla_qabs offset:0 atIndex:2];
            [enc setBytes:&ab1 length:sizeof(ab1) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((kv_r + 3u) / 4u, (NSUInteger)m->n_heads, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            struct { uint32_t nh, nd, kvr, dh, bb; } ab = { (uint32_t)m->n_heads, nope_d, kv_r, (uint32_t)m->d_head, B };
            [enc setComputePipelineState:g_pso_mla_absorb_q_b];
            [enc setBuffer:ly->mla_wuk_f16 offset:0 atIndex:0];
            [enc setBuffer:m->pbuf_q offset:0 atIndex:1];
            [enc setBuffer:m->pbuf_mla_qabs offset:0 atIndex:2];
            [enc setBytes:&ab length:sizeof(ab) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((kv_r + 15u) / 16u, (NSUInteger)m->n_heads, (B + 15u) / 16u) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }

        if (m->cur_use_fd && g_pso_flash_mla_split_paged != nil && g_pso_flash_dec_reduce_paged != nil && m->buf_fd_o != nil) {
            uint32_t ns = (uint32_t)(m->cur_fd_splits > 0 ? m->cur_fd_splits : 8);
            struct { uint32_t nh, kvr, nd, rd, dh, bb; float sc; uint32_t mb, swa, sink, ns; } fls = {
                (uint32_t)m->n_heads, kv_r, nope_d, (uint32_t)m->rope_dim, (uint32_t)m->d_head, B, attn_scale,
                (uint32_t)PA_MAX_LOGICAL_BLKS, (uint32_t)(m->sliding_window > 0 ? m->sliding_window : 0),
                (uint32_t)(m->attn_sink > 0 ? m->attn_sink : 0), ns
            };
            int use_mla512 = (kv_r == 512u && m->rope_dim == 64 && g_pso_flash_mla_split_paged_512 != nil &&
                              g_pso_flash_mla_split_paged_512.maxTotalThreadsPerThreadgroup >= 512);
            [enc setComputePipelineState:(use_mla512 ? g_pso_flash_mla_split_paged_512 : g_pso_flash_mla_split_paged)];
            [enc setBuffer:m->pbuf_mla_qabs offset:0 atIndex:0];
            [enc setBuffer:m->pbuf_q offset:0 atIndex:1];
            [enc setBuffer:ly->k_cache offset:0 atIndex:2];
            [enc setBuffer:m->buf_fd_o offset:0 atIndex:3];
            [enc setBuffer:m->buf_fd_m offset:0 atIndex:4];
            [enc setBuffer:m->buf_fd_l offset:0 atIndex:5];
            [enc setBuffer:m->buf_cb_pos offset:0 atIndex:6];
            [enc setBuffer:m->buf_cb_slot offset:0 atIndex:7];
            [enc setBuffer:m->buf_pa_bt offset:0 atIndex:8];
            [enc setBytes:&fls length:sizeof(fls) atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(use_mla512 ? ((m->n_heads + 15) / 16) : ((m->n_heads + 3) / 4)), B, ns)
                threadsPerThreadgroup:MTLSizeMake(use_mla512 ? 512 : 128, 1, 1)];

            struct { uint32_t nh, vd, os, bb, ns; } red = { (uint32_t)m->n_heads, kv_r, kv_r, B, ns };
            [enc setComputePipelineState:g_pso_flash_dec_reduce_paged];
            [enc setBuffer:m->buf_fd_o offset:0 atIndex:0];
            [enc setBuffer:m->buf_fd_m offset:0 atIndex:1];
            [enc setBuffer:m->buf_fd_l offset:0 atIndex:2];
            [enc setBuffer:m->pbuf_mla_olat offset:0 atIndex:3];
            [enc setBytes:&red length:sizeof(red) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            m->flash_decode_launches++;
        } else {
            struct { uint32_t nh, kvr, nd, rd, dh, bb; float sc; uint32_t mb, swa, sink; } fl = {
                (uint32_t)m->n_heads, kv_r, nope_d, (uint32_t)m->rope_dim, (uint32_t)m->d_head, B, attn_scale,
                (uint32_t)PA_MAX_LOGICAL_BLKS, (uint32_t)(m->sliding_window > 0 ? m->sliding_window : 0),
                (uint32_t)(m->attn_sink > 0 ? m->attn_sink : 0)
            };
            [enc setComputePipelineState:g_pso_mla_flash_lat_paged];
            [enc setBuffer:m->pbuf_mla_qabs offset:0 atIndex:0];
            [enc setBuffer:m->pbuf_q offset:0 atIndex:1];
            [enc setBuffer:ly->k_cache offset:0 atIndex:2];
            [enc setBuffer:m->pbuf_mla_olat offset:0 atIndex:3];
            [enc setBuffer:m->buf_cb_pos offset:0 atIndex:4];
            [enc setBuffer:m->buf_cb_slot offset:0 atIndex:5];
            [enc setBuffer:m->buf_pa_bt offset:0 atIndex:6];
            [enc setBytes:&fl length:sizeof(fl) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((m->n_heads + 3) / 4), B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }

        if (B == 1u && g_pso_mla_proj_v != nil) {
            struct { uint32_t nh, vd, kvr, os; } pv1 = { (uint32_t)m->n_heads, v_d, kv_r, out_stride };
            [enc setComputePipelineState:g_pso_mla_proj_v];
            [enc setBuffer:ly->mla_wuv_f16 offset:0 atIndex:0];
            [enc setBuffer:m->pbuf_mla_olat offset:0 atIndex:1];
            [enc setBuffer:m->pbuf_ctx offset:0 atIndex:2];
            [enc setBytes:&pv1 length:sizeof(pv1) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((out_stride + 3u) / 4u), (NSUInteger)m->n_heads, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            struct { uint32_t nh, vd, kvr, os, bb; } pv = { (uint32_t)m->n_heads, v_d, kv_r, out_stride, B };
            [enc setComputePipelineState:g_pso_mla_proj_v_b];
            [enc setBuffer:ly->mla_wuv_f16 offset:0 atIndex:0];
            [enc setBuffer:m->pbuf_mla_olat offset:0 atIndex:1];
            [enc setBuffer:m->pbuf_ctx offset:0 atIndex:2];
            [enc setBytes:&pv length:sizeof(pv) atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((out_stride + 15u) / 16u), (NSUInteger)m->n_heads, (B + 15u) / 16u) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }

        encode_gemm_batch_raw(enc, ly->wo.buf, ly->wo.offset, ly->wo.ggml_type, m->pbuf_ctx, m->pbuf_x, dm, wo_in, B, 1);
        if (m->active_lora > 0 || m->use_batch_lora) encode_lora_delta(m, enc, L, 3, m->pbuf_ctx, m->pbuf_x, dm, wo_in, B);
        encode_rmsnorm(enc, m->pbuf_x, &ly->ffn_norm, m->pbuf_xn, dm, B, m->rms_eps, 1);
        return;
    }

    int has_qkv_lora = (m->active_lora > 0 || m->use_batch_lora);
    int qkv_bias_fused = 0;
    if (B == 1u && !has_qkv_lora && g_pso_gemv_qkv_q4 != nil &&
        ly->wq.ggml_type == TN_GGML_Q4_0 && ly->wk.ggml_type == TN_GGML_Q4_0 && ly->wv.ggml_type == TN_GGML_Q4_0) {
        // Tenon v13: Single-dispatch fused Q||K||V + Optional QKV Bias for B=1 Paged Decode (saves 2-5 dispatches/layer!)
        qkv_bias_fused = (ly->has_qkv_bias && ly->bq.buf != nil && ly->bk.buf != nil && ly->bv.buf != nil);
        struct { uint32_t q_dim, kv_dim, in_dim, has_bias; } qa = { qdim, kvdim, dm, (uint32_t)qkv_bias_fused };
        id<MTLBuffer> qb = (ping_buf != nil) ? ping_buf : ly->wq.buf;
        id<MTLBuffer> kb = (ping_buf != nil) ? ping_buf : ly->wk.buf;
        id<MTLBuffer> vb = (ping_buf != nil) ? ping_buf : ly->wv.buf;
        NSUInteger qo = (NSUInteger)((ping_buf != nil) ? m->dense_off_wq : ly->wq.offset);
        NSUInteger ko = (NSUInteger)((ping_buf != nil) ? m->dense_off_wk : ly->wk.offset);
        NSUInteger vo = (NSUInteger)((ping_buf != nil) ? m->dense_off_wv : ly->wv.offset);
        [enc setComputePipelineState:g_pso_gemv_qkv_q4];
        [enc setBuffer:qb offset:qo atIndex:0];
        [enc setBuffer:kb offset:ko atIndex:1];
        [enc setBuffer:vb offset:vo atIndex:2];
        [enc setBuffer:m->pbuf_xn offset:0 atIndex:3];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:4];
        [enc setBuffer:m->pbuf_k offset:0 atIndex:5];
        [enc setBuffer:m->pbuf_v offset:0 atIndex:6];
        [enc setBytes:&qa length:sizeof(qa) atIndex:7];
        [enc setBuffer:(qkv_bias_fused ? ly->bq.buf : m->buf_argmax) offset:(qkv_bias_fused ? (NSUInteger)ly->bq.offset : 0) atIndex:8];
        [enc setBuffer:(qkv_bias_fused ? ly->bk.buf : m->buf_argmax) offset:(qkv_bias_fused ? (NSUInteger)ly->bk.offset : 0) atIndex:9];
        [enc setBuffer:(qkv_bias_fused ? ly->bv.buf : m->buf_argmax) offset:(qkv_bias_fused ? (NSUInteger)ly->bv.offset : 0) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((qdim + 2u * kvdim + 7u) / 8u), 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else if (B >= 33u && !has_qkv_lora && ping_buf == nil && g_pso_gemm_q4_qkv_b != nil &&
               ly->wq.ggml_type == TN_GGML_Q4_0 && ly->wk.ggml_type == TN_GGML_Q4_0 && ly->wv.ggml_type == TN_GGML_Q4_0 &&
               (qdim & 63u) == 0u && (kvdim & 63u) == 0u) {
        struct { uint32_t q_dim, kv_dim, in_dim, batch_b; } qa = { qdim, kvdim, dm, B };
        [enc setComputePipelineState:g_pso_gemm_q4_qkv_b];
        [enc setBuffer:ly->wq.buf offset:(NSUInteger)ly->wq.offset atIndex:0];
        [enc setBuffer:ly->wk.buf offset:(NSUInteger)ly->wk.offset atIndex:1];
        [enc setBuffer:ly->wv.buf offset:(NSUInteger)ly->wv.offset atIndex:2];
        [enc setBuffer:m->pbuf_xn offset:0 atIndex:3];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:4];
        [enc setBuffer:m->pbuf_k offset:0 atIndex:5];
        [enc setBuffer:m->pbuf_v offset:0 atIndex:6];
        [enc setBytes:&qa length:sizeof(qa) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((qdim + 2u * kvdim + 63u) / 64u), (NSUInteger)((B + 31u) / 32u), 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (ping_buf != nil) {
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wq, ly->wq.ggml_type, m->pbuf_xn, m->pbuf_q, qdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wk, ly->wk.ggml_type, m->pbuf_xn, m->pbuf_k, kvdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wv, ly->wv.ggml_type, m->pbuf_xn, m->pbuf_v, kvdim, dm, B, 0);
    } else {
        encode_gemm_batch_raw(enc, ly->wq.buf, ly->wq.offset, ly->wq.ggml_type, m->pbuf_xn, m->pbuf_q, qdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ly->wk.buf, ly->wk.offset, ly->wk.ggml_type, m->pbuf_xn, m->pbuf_k, kvdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ly->wv.buf, ly->wv.offset, ly->wv.ggml_type, m->pbuf_xn, m->pbuf_v, kvdim, dm, B, 0);
    }
    if (m->active_lora > 0 || m->use_batch_lora) {
        encode_lora_delta(m, enc, L, 0, m->pbuf_xn, m->pbuf_q, qdim, dm, B);
        encode_lora_delta(m, enc, L, 1, m->pbuf_xn, m->pbuf_k, kvdim, dm, B);
        encode_lora_delta(m, enc, L, 2, m->pbuf_xn, m->pbuf_v, kvdim, dm, B);
    }
    if (ly->has_qkv_bias && !qkv_bias_fused) {
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
    struct { uint32_t nh, nkv, dh, bb, os; float sc; uint32_t mb, swa, sink; } aargs = {
        (uint32_t)m->n_heads, (uint32_t)m->n_kv, (uint32_t)m->d_head, B, out_stride, attn_scale,
        (uint32_t)PA_MAX_LOGICAL_BLKS, (uint32_t)(m->sliding_window > 0 ? m->sliding_window : 0),
        (uint32_t)(m->attn_sink > 0 ? m->attn_sink : 0)
    };
    if (m->kv_quant_q8 >= 1) {
        id<MTLComputePipelineState> rpso = (m->kv_quant_q8 == 3) ? g_pso_rope_kv_q4_paged
                                         : (m->kv_quant_q8 == 2) ? g_pso_rope_kv_fp8_paged
                                         : g_pso_rope_kv_q8_paged;
        id<MTLComputePipelineState> apso = (m->kv_quant_q8 == 3) ? g_pso_flash_dec_q4_paged
                                         : (m->kv_quant_q8 == 2) ? g_pso_flash_dec_fp8_paged
                                         : g_pso_flash_dec_q8_paged;
        [enc setComputePipelineState:rpso];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:m->pbuf_k offset:0 atIndex:1]; [enc setBuffer:m->pbuf_v offset:0 atIndex:2];
        [enc setBuffer:ly->k_cache offset:0 atIndex:3]; [enc setBuffer:ly->v_cache offset:0 atIndex:4];
        [enc setBuffer:(m->has_rope_freqs ? m->rope_freqs.buf : m->buf_argmax) offset:(m->has_rope_freqs ? (NSUInteger)m->rope_freqs.offset : 0) atIndex:5];
        [enc setBuffer:m->buf_cb_pos offset:0 atIndex:6]; [enc setBuffer:m->buf_cb_slot offset:0 atIndex:7]; [enc setBuffer:m->buf_pa_bt offset:0 atIndex:8];
        [enc setBytes:&rargs length:sizeof(rargs) atIndex:9];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

        if (m->cur_use_fd && g_pso_flash_dec_split_paged != nil && g_pso_flash_dec_reduce_paged != nil && m->buf_fd_o != nil) {
            uint32_t ns = (uint32_t)(m->cur_fd_splits > 0 ? m->cur_fd_splits : 8);
            uint32_t grp = (uint32_t)((m->n_heads / m->n_kv) > 0 ? (m->n_heads / m->n_kv) : 1);
            uint32_t gstep = (grp < 8u ? grp : 8u);
            uint32_t n_gchunks = (grp + gstep - 1u) / gstep;
            struct { uint32_t nh, nkv, dh, bb; float sc; uint32_t mb, swa, sink, kvm, ns, gs; } sargs = {
                (uint32_t)m->n_heads, (uint32_t)m->n_kv, (uint32_t)m->d_head, B, attn_scale,
                (uint32_t)PA_MAX_LOGICAL_BLKS, (uint32_t)(m->sliding_window > 0 ? m->sliding_window : 0),
                (uint32_t)(m->attn_sink > 0 ? m->attn_sink : 0), (uint32_t)m->kv_quant_q8, ns, gstep
            };
            [enc setComputePipelineState:g_pso_flash_dec_split_paged];
            [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:ly->k_cache offset:0 atIndex:1]; [enc setBuffer:ly->v_cache offset:0 atIndex:2];
            [enc setBuffer:m->buf_fd_o offset:0 atIndex:3]; [enc setBuffer:m->buf_fd_m offset:0 atIndex:4]; [enc setBuffer:m->buf_fd_l offset:0 atIndex:5];
            [enc setBuffer:m->buf_cb_pos offset:0 atIndex:6]; [enc setBuffer:m->buf_cb_slot offset:0 atIndex:7]; [enc setBuffer:m->buf_pa_bt offset:0 atIndex:8];
            [enc setBytes:&sargs length:sizeof(sargs) atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_kv * (NSUInteger)n_gchunks, B, ns) threadsPerThreadgroup:MTLSizeMake((NSUInteger)gstep * 32u, 1, 1)];

            struct { uint32_t nh, vd, os, bb, ns; } red = { (uint32_t)m->n_heads, (uint32_t)m->d_head, out_stride, B, ns };
            [enc setComputePipelineState:g_pso_flash_dec_reduce_paged];
            [enc setBuffer:m->buf_fd_o offset:0 atIndex:0]; [enc setBuffer:m->buf_fd_m offset:0 atIndex:1]; [enc setBuffer:m->buf_fd_l offset:0 atIndex:2];
            [enc setBuffer:m->pbuf_ctx offset:0 atIndex:3]; [enc setBytes:&red length:sizeof(red) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            m->flash_decode_launches++;
        } else {
            [enc setComputePipelineState:apso];
            [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:ly->k_cache offset:0 atIndex:1]; [enc setBuffer:ly->v_cache offset:0 atIndex:2];
            [enc setBuffer:m->pbuf_ctx offset:0 atIndex:3]; [enc setBuffer:m->buf_cb_pos offset:0 atIndex:4]; [enc setBuffer:m->buf_cb_slot offset:0 atIndex:5];
            [enc setBuffer:m->buf_pa_bt offset:0 atIndex:6]; [enc setBytes:&aargs length:sizeof(aargs) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
    } else {
        [enc setComputePipelineState:g_pso_rope_kv_paged];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:m->pbuf_k offset:0 atIndex:1]; [enc setBuffer:m->pbuf_v offset:0 atIndex:2];
        [enc setBuffer:ly->k_cache offset:0 atIndex:3]; [enc setBuffer:ly->v_cache offset:0 atIndex:4];
        [enc setBuffer:(m->has_rope_freqs ? m->rope_freqs.buf : m->buf_argmax) offset:(m->has_rope_freqs ? (NSUInteger)m->rope_freqs.offset : 0) atIndex:5];
        [enc setBuffer:m->buf_cb_pos offset:0 atIndex:6]; [enc setBuffer:m->buf_cb_slot offset:0 atIndex:7]; [enc setBuffer:m->buf_pa_bt offset:0 atIndex:8];
        [enc setBytes:&rargs length:sizeof(rargs) atIndex:9];
        [enc dispatchThreads:MTLSizeMake((NSUInteger)(m->d_head / 2), (NSUInteger)m->n_heads, B) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        if (m->cur_use_fd && g_pso_flash_dec_split_paged != nil && g_pso_flash_dec_reduce_paged != nil && m->buf_fd_o != nil) {
            uint32_t ns = (uint32_t)(m->cur_fd_splits > 0 ? m->cur_fd_splits : 8);
            uint32_t grp = (uint32_t)((m->n_heads / m->n_kv) > 0 ? (m->n_heads / m->n_kv) : 1);
            uint32_t gstep = (grp < 8u ? grp : 8u);
            uint32_t n_gchunks = (grp + gstep - 1u) / gstep;
            struct { uint32_t nh, nkv, dh, bb; float sc; uint32_t mb, swa, sink, kvm, ns, gs; } sargs = {
                (uint32_t)m->n_heads, (uint32_t)m->n_kv, (uint32_t)m->d_head, B, attn_scale,
                (uint32_t)PA_MAX_LOGICAL_BLKS, (uint32_t)(m->sliding_window > 0 ? m->sliding_window : 0),
                (uint32_t)(m->attn_sink > 0 ? m->attn_sink : 0), 0u, ns, gstep
            };
            [enc setComputePipelineState:g_pso_flash_dec_split_paged];
            [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:ly->k_cache offset:0 atIndex:1]; [enc setBuffer:ly->v_cache offset:0 atIndex:2];
            [enc setBuffer:m->buf_fd_o offset:0 atIndex:3]; [enc setBuffer:m->buf_fd_m offset:0 atIndex:4]; [enc setBuffer:m->buf_fd_l offset:0 atIndex:5];
            [enc setBuffer:m->buf_cb_pos offset:0 atIndex:6]; [enc setBuffer:m->buf_cb_slot offset:0 atIndex:7]; [enc setBuffer:m->buf_pa_bt offset:0 atIndex:8];
            [enc setBytes:&sargs length:sizeof(sargs) atIndex:9];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_kv * (NSUInteger)n_gchunks, B, ns) threadsPerThreadgroup:MTLSizeMake((NSUInteger)gstep * 32u, 1, 1)];

            struct { uint32_t nh, vd, os, bb, ns; } red = { (uint32_t)m->n_heads, (uint32_t)m->d_head, out_stride, B, ns };
            [enc setComputePipelineState:g_pso_flash_dec_reduce_paged];
            [enc setBuffer:m->buf_fd_o offset:0 atIndex:0]; [enc setBuffer:m->buf_fd_m offset:0 atIndex:1]; [enc setBuffer:m->buf_fd_l offset:0 atIndex:2];
            [enc setBuffer:m->pbuf_ctx offset:0 atIndex:3]; [enc setBytes:&red length:sizeof(red) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            m->flash_decode_launches++;
        } else if (m->cur_is_prefill && B >= 32u && m->d_head == 64 && g_pso_flash_prefill_amx64 != nil) {
            [enc setComputePipelineState:g_pso_flash_prefill_amx64];
            [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:ly->k_cache offset:0 atIndex:1]; [enc setBuffer:ly->v_cache offset:0 atIndex:2];
            [enc setBuffer:m->pbuf_ctx offset:0 atIndex:3]; [enc setBuffer:m->buf_cb_pos offset:0 atIndex:4]; [enc setBuffer:m->buf_cb_slot offset:0 atIndex:5];
            [enc setBuffer:m->buf_pa_bt offset:0 atIndex:6]; [enc setBytes:&aargs length:sizeof(aargs) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, (NSUInteger)((B + 31u) / 32u), 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (m->cur_is_prefill && B >= 32u && m->d_head == 128 && g_pso_flash_prefill_amx128 != nil) {
            [enc setComputePipelineState:g_pso_flash_prefill_amx128];
            [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:ly->k_cache offset:0 atIndex:1]; [enc setBuffer:ly->v_cache offset:0 atIndex:2];
            [enc setBuffer:m->pbuf_ctx offset:0 atIndex:3]; [enc setBuffer:m->buf_cb_pos offset:0 atIndex:4]; [enc setBuffer:m->buf_cb_slot offset:0 atIndex:5];
            [enc setBuffer:m->buf_pa_bt offset:0 atIndex:6]; [enc setBytes:&aargs length:sizeof(aargs) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, (NSUInteger)((B + 31u) / 32u), 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (m->cur_is_prefill && B >= 16u && m->d_head <= 128 && (m->d_head & 3) == 0 && g_pso_flash_prefill_paged_fp16 != nil) {
            [enc setComputePipelineState:g_pso_flash_prefill_paged_fp16];
            [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:ly->k_cache offset:0 atIndex:1]; [enc setBuffer:ly->v_cache offset:0 atIndex:2];
            [enc setBuffer:m->pbuf_ctx offset:0 atIndex:3]; [enc setBuffer:m->buf_cb_pos offset:0 atIndex:4]; [enc setBuffer:m->buf_cb_slot offset:0 atIndex:5];
            [enc setBuffer:m->buf_pa_bt offset:0 atIndex:6]; [enc setBytes:&aargs length:sizeof(aargs) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, (NSUInteger)((B + 7u) / 8u), 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [enc setComputePipelineState:g_pso_flash_dec_paged];
            [enc setBuffer:m->pbuf_q offset:0 atIndex:0]; [enc setBuffer:ly->k_cache offset:0 atIndex:1]; [enc setBuffer:ly->v_cache offset:0 atIndex:2];
            [enc setBuffer:m->pbuf_ctx offset:0 atIndex:3]; [enc setBuffer:m->buf_cb_pos offset:0 atIndex:4]; [enc setBuffer:m->buf_cb_slot offset:0 atIndex:5];
            [enc setBuffer:m->buf_pa_bt offset:0 atIndex:6]; [enc setBytes:&aargs length:sizeof(aargs) atIndex:7];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)m->n_heads, B, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
    }
    if (ping_buf != nil) encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wo, ly->wo.ggml_type, m->pbuf_ctx, m->pbuf_x, dm, wo_in, B, 1);
    else encode_gemm_batch_raw(enc, ly->wo.buf, ly->wo.offset, ly->wo.ggml_type, m->pbuf_ctx, m->pbuf_x, dm, wo_in, B, 1);
    if (m->active_lora > 0 || m->use_batch_lora) encode_lora_delta(m, enc, L, 3, m->pbuf_ctx, m->pbuf_x, dm, wo_in, B);
    encode_rmsnorm(enc, m->pbuf_x, &ly->ffn_norm, m->pbuf_xn, dm, B, m->rms_eps, 1);
}

// Tenon v9: Zero-Host-Sync GPU-Resident Grouped MoE Mega-Kernel (ALL 27 layers in ONE SINGLE MTLCommandBuffer!)
// Replaces CPU waitUntilCompleted + ~180 launches/layer with 4 deterministic GPU dispatches/layer!
static void encode_grouped_moe_ffn_layer(MoeModel* m, id<MTLCommandBuffer>* io_cmd, id<MTLComputeCommandEncoder>* io_enc, int L, uint32_t B) {
    (void)io_cmd;
    id<MTLComputeCommandEncoder> enc = *io_enc;
    MoeLayer* ly = &m->layers[L];
    uint32_t dm = (uint32_t)m->d_model, df = (uint32_t)m->d_ff_exp, ne = (uint32_t)m->n_experts;
    uint32_t tk = (uint32_t)m->top_k;
    encode_gemm_batch_raw(enc, ly->router.buf, ly->router.offset, ly->router.ggml_type, m->pbuf_xn, m->pbuf_rlogits, ne, dm, B, 0);
    if (ly->has_shared_exp) {
        uint32_t sh_ff = (ly->sh_gate.ne1 > 0) ? (uint32_t)ly->sh_gate.ne1 : df;
        encode_dense_ffn_batch(m, enc, ly->sh_gate.buf, ly->sh_gate.offset, ly->sh_gate.ggml_type,
                               ly->sh_up.buf, ly->sh_up.offset, ly->sh_up.ggml_type,
                               ly->sh_down.buf, ly->sh_down.offset, ly->sh_down.ggml_type, sh_ff, B);
    }
    // Tenon v12: Direct 3-Dispatch Resident B=1 MoE Fast Path (Fused TopK Gate+Up+SwiGLU + Vectorized Down-Accum, 100% ICB-compatible!)
    if (B == 1u && m->resident_mode &&
        (ly->gate_exps.ggml_type == TN_GGML_Q4_0 || ly->gate_exps.ggml_type == TN_GGML_Q4_K)) {
        struct { uint32_t ne, tk, nw, gf, hb; float ws; uint32_t ng, ngu; } rt = {
            ne, tk, (uint32_t)m->expert_weights_norm, (uint32_t)m->gating_func,
            (uint32_t)(ly->has_exp_bias && ly->exp_bias.buf != nil), m->expert_weights_scale,
            (uint32_t)m->n_expert_groups, (uint32_t)m->n_group_used
        };
        [enc setComputePipelineState:g_pso_router];
        [enc setBuffer:m->pbuf_rlogits offset:0 atIndex:0];
        [enc setBuffer:(rt.hb ? ly->exp_bias.buf : m->buf_moe_zero)
                offset:(rt.hb ? (NSUInteger)ly->exp_bias.offset : 0) atIndex:1];
        [enc setBuffer:m->buf_eidx offset:0 atIndex:2];
        [enc setBuffer:m->buf_ewt offset:0 atIndex:3];
        [enc setBytes:&rt length:sizeof(rt) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

        struct { uint32_t dm, df, na; uint64_t gb, ub, db, sb; } margs = {
            dm, df, tk, ly->gate_exp_bytes, ly->up_exp_bytes, ly->down_exp_bytes, 0
        };
        int is_q4k_gu = (ly->gate_exps.ggml_type == TN_GGML_Q4_K);
        [enc setComputePipelineState:(is_q4k_gu ? g_pso_moe_gu_q4k : g_pso_moe_gu_q4)];
        [enc setBuffer:ly->gate_exps.buf offset:(NSUInteger)ly->gate_exps.offset atIndex:0];
        [enc setBuffer:ly->up_exps.buf offset:(NSUInteger)ly->up_exps.offset atIndex:1];
        [enc setBuffer:m->pbuf_xn offset:0 atIndex:2];
        [enc setBuffer:m->buf_eidx offset:0 atIndex:3];
        [enc setBuffer:m->buf_h offset:0 atIndex:4];
        [enc setBytes:&margs length:sizeof(margs) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((df + 3u) / 4u), (NSUInteger)tk, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

        id<MTLComputePipelineState> r_dn_pso = nil;
        NSUInteger r_dn_tgs = 0, r_dn_tptg = 64;
        select_moe_down_pso(ly->down_exps.ggml_type, 0, m->d_model, &r_dn_pso, &r_dn_tgs, &r_dn_tptg);
        [enc setComputePipelineState:r_dn_pso];
        [enc setBuffer:ly->down_exps.buf offset:(NSUInteger)ly->down_exps.offset atIndex:0];
        [enc setBuffer:m->buf_h offset:0 atIndex:1];
        [enc setBuffer:m->buf_eidx offset:0 atIndex:2];
        [enc setBuffer:m->buf_ewt offset:0 atIndex:3];
        [enc setBuffer:m->pbuf_x offset:0 atIndex:4];
        [enc setBytes:&margs length:sizeof(margs) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(r_dn_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(r_dn_tptg, 1, 1)];
        m->cache_hits += (int64_t)tk;
        return;
    }
    // 1. GPU-side Batched TopK + Counting-Sort (ZERO CPU sync!)
    struct {
        uint32_t batch_b, n_exp, top_k, gating_func;
        uint32_t norm_wt, has_bias, n_groups, n_group_used;
        float w_scale;
    } rs = {
        B, ne, tk, (uint32_t)m->gating_func,
        (uint32_t)m->expert_weights_norm, (uint32_t)(ly->has_exp_bias && ly->exp_bias.buf != nil),
        (uint32_t)m->n_expert_groups, (uint32_t)m->n_group_used, m->expert_weights_scale
    };
    id<MTLBuffer> eb_buf = (rs.has_bias) ? ly->exp_bias.buf : m->buf_moe_zero;
    NSUInteger eb_off = (rs.has_bias) ? (NSUInteger)ly->exp_bias.offset : 0;
    [enc setComputePipelineState:g_pso_moe_route_sort];
    [enc setBuffer:m->pbuf_rlogits offset:0 atIndex:0];
    [enc setBuffer:eb_buf offset:eb_off atIndex:1];
    [enc setBuffer:m->buf_moe_active_e offset:0 atIndex:2];
    [enc setBuffer:m->buf_moe_grp_off offset:0 atIndex:3];
    [enc setBuffer:m->buf_moe_grp_cnt offset:0 atIndex:4];
    [enc setBuffer:m->buf_moe_idx offset:0 atIndex:5];
    [enc setBuffer:m->buf_moe_wt offset:0 atIndex:6];
    [enc setBuffer:m->buf_moe_tok_slot offset:0 atIndex:7];
    [enc setBuffer:m->buf_moe_meta offset:0 atIndex:8];
    [enc setBytes:&rs length:sizeof(rs) atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];

    uint32_t max_slots = B * tk;
    if (max_slots > ne) max_slots = ne;
    // Tenon v10: Adaptive Roofline Dispatch -> FP16-AMX Mega-Kernel v2 for B>=8 (hot experts cnt>=8 use simdgroup_half8x8 AMX,
    // cold experts cnt<8 stay on zero-waste warp-vectorized fast path!), warp-vectorized Mega-Kernel for B<8!
    int use_amx_mega = (B >= 8u && g_pso_moe_mega_gu_amx != nil && g_pso_moe_mega_down_amx != nil) ? 1 : 0;
    // 2. Single-Launch Fused Grouped Gate + Up + SwiGLU Mega-Kernel (reads each active expert's weights ONCE!)
    struct { uint32_t out_dim, in_dim, gtype, gtype2; uint64_t exp_bytes, exp_bytes2; } gu_a = {
        df, dm, ly->gate_exps.ggml_type, ly->up_exps.ggml_type, ly->gate_exp_bytes, ly->up_exp_bytes
    };
    [enc setComputePipelineState:(use_amx_mega ? g_pso_moe_mega_gu_amx : g_pso_moe_mega_gu)];
    [enc setBuffer:ly->gate_exps.buf offset:(NSUInteger)ly->gate_exps.offset atIndex:0];
    [enc setBuffer:ly->up_exps.buf offset:(NSUInteger)ly->up_exps.offset atIndex:1];
    [enc setBuffer:m->pbuf_xn offset:0 atIndex:2];
    [enc setBuffer:m->pbuf_moe_h offset:0 atIndex:3];
    [enc setBuffer:m->buf_moe_active_e offset:0 atIndex:4];
    [enc setBuffer:m->buf_moe_grp_off offset:0 atIndex:5];
    [enc setBuffer:m->buf_moe_grp_cnt offset:0 atIndex:6];
    [enc setBuffer:m->buf_moe_idx offset:0 atIndex:7];
    [enc setBuffer:m->buf_moe_meta offset:0 atIndex:8];
    [enc setBytes:&gu_a length:sizeof(gu_a) atIndex:9];
    [enc dispatchThreadgroups:MTLSizeMake(use_amx_mega ? ((df + 31u) / 32u) : ((df + 3u) / 4u), max_slots, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    // 3. Single-Launch Grouped Down-Proj Mega-Kernel
    struct { uint32_t out_dim, in_dim, gtype, gtype2; uint64_t exp_bytes, exp_bytes2; } dn_a = {
        dm, df, ly->down_exps.ggml_type, 0u, ly->down_exp_bytes, 0ull
    };
    [enc setComputePipelineState:(use_amx_mega ? g_pso_moe_mega_down_amx : g_pso_moe_mega_down)];
    [enc setBuffer:ly->down_exps.buf offset:(NSUInteger)ly->down_exps.offset atIndex:0];
    [enc setBuffer:m->pbuf_moe_h offset:0 atIndex:1];
    [enc setBuffer:m->pbuf_moe_y offset:0 atIndex:2];
    [enc setBuffer:m->buf_moe_active_e offset:0 atIndex:3];
    [enc setBuffer:m->buf_moe_grp_off offset:0 atIndex:4];
    [enc setBuffer:m->buf_moe_grp_cnt offset:0 atIndex:5];
    [enc setBuffer:m->buf_moe_meta offset:0 atIndex:6];
    [enc setBytes:&dn_a length:sizeof(dn_a) atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(use_amx_mega ? ((dm + 63u) / 64u) : ((dm + 3u) / 4u), max_slots, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    // 4. Deterministic TopK-Ordered Weighted Residual Reduction (100% BIT-EXACT IEEE float order!)
    struct { uint32_t dm, batch_b, top_k; } rd_a = { dm, B, tk };
    [enc setComputePipelineState:g_pso_moe_mega_reduce];
    [enc setBuffer:m->pbuf_moe_y offset:0 atIndex:0];
    [enc setBuffer:m->pbuf_x offset:0 atIndex:1];
    [enc setBuffer:m->buf_moe_tok_slot offset:0 atIndex:2];
    [enc setBuffer:m->buf_moe_wt offset:0 atIndex:3];
    [enc setBytes:&rd_a length:sizeof(rd_a) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(dm, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    m->cache_hits += (int64_t)B * (int64_t)tk;
}

static void run_dense_paged_layers_batch(MoeModel* m, uint32_t B) {
    if (m->resident_mode) {
        id<MTLCommandBuffer> cmd = [g_moe_q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        for (int L = 0; L < m->n_layers; L++) {
            MoeLayer* ly = &m->layers[L];
            encode_layer_attn_paged_batch(m, enc, L, B, nil);
            if (ly->is_dense) {
                encode_dense_ffn_batch(m, enc, ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                                       ly->ffn_up.buf, ly->ffn_up.offset, ly->ffn_up.ggml_type,
                                       ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
            } else {
                encode_grouped_moe_ffn_layer(m, &cmd, &enc, L, B);
            }
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
// in ONE batched AMX/register-direct forward pass (Dense, Latent MLA, Resident Grouped MoE!), saving all `n_active` logit rows!
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
        icb_cache_clear(m);
        int ncap = (int)(B < 16 ? 16 : B);
        m->buf_batch_logits = [g_moe_dev newBufferWithLength:(size_t)ncap * (size_t)m->vocab_size * 4 options:MTLResourceStorageModeShared];
        m->batch_logits_cap = ncap;
    }

    int is_mla_m = (m->n_layers > 0 && m->layers[0].is_mla);
    int can_paged = m->dense_model
        ? (!is_mla_m || (m->mla_latent_kv && m->layers[0].mla_wuk_f16 != nil))
        : (m->resident_mode && (!is_mla_m || (m->mla_latent_kv && m->layers[0].mla_wuk_f16 != nil)));
    if (!can_paged) {
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
        int max_pos = 0;
        for (uint32_t b = 0; b < B; b++) {
            int s = (int)slt[b]; if (s < 0 || s >= PA_MAX_SEQS) s = 0;
            int p = (int)pos[b]; if (p < 0) p = 0;
            if (!pa_ensure_slot_pos(m, s, p)) return 0;
            p_arr[b] = (uint32_t)p;
            s_arr[b] = (uint32_t)s;
            if (p > max_pos) max_pos = p;
        }
        pa_sync_bt(m);
        int use_fd = (m->fd_enabled && (max_pos + 1 >= m->fd_min_len) && B <= 8u) ? 1 : 0;
        int cur_sp = m->fd_splits;
        if (use_fd && !m->fd_splits_forced) {
            int kv_l = max_pos + 1;
            int is_mla_l = (m->n_layers > 0 && m->layers[0].is_mla) ? 1 : 0;
            if (is_mla_l) cur_sp = (kv_l >= 1024 && B <= 2u) ? 32 : 16;
            else if (kv_l >= 1536) cur_sp = (B <= 2u && m->n_kv <= 8) ? 32 : 16;
            else if (kv_l >= 512) cur_sp = 16;
            else cur_sp = 8;
        }
        m->cur_use_fd = use_fd;
        m->cur_fd_splits = cur_sp;
        m->cur_is_prefill = 0;

        // 1. Fuse Embedding(B) + All Layers (Dense OR Resident MLA + GPU MoE Mega-Kernel!) + Out-Norm + LM-Head!
        // Tenon v11: Dual-Queue Async MTLIndirectCommandBuffer (ICB) Persistent Decode Graph Capture & Zero-Retain Replay!
        float* px = (float*)[m->pbuf_x contents];
        if (m->resident_mode) {
            uint32_t* t_arr = (uint32_t*)[m->buf_cb_tok contents];
            for (uint32_t bi = 0; bi < B; bi++) t_arr[bi] = (uint32_t)ids[bi];
            uint64_t lsig = ((uint64_t)(m->use_batch_lora ? 1u : 0u) << 63) | ((uint64_t)(uint32_t)m->active_lora << 48);
            for (int lid = 1; lid < MOE_MAX_LORAS; lid++) {
                if (m->loras[lid].active) lsig ^= ((uint64_t)m->loras[lid].vram_bytes + (uint64_t)lid * 1315423911ull) * 2654435761ull;
            }
            if (m->use_batch_lora) {
                for (uint32_t bi = 0; bi < B; bi++) lsig = (lsig * 1315423911ull) ^ (uint64_t)(uint32_t)m->batch_lora_ids[bi];
            }
            int icb_slot = -1;
            if (m->icb_enabled && g_pso_embed_batch != nil) {
                for (int ci = 0; ci < MOE_ICB_CACHE_MAX; ci++) {
                    if (m->icb_cache[ci].valid && m->icb_cache[ci].B == B && m->icb_cache[ci].lora_sig == lsig &&
                        m->icb_cache[ci].use_fd == use_fd && (!use_fd || m->icb_cache[ci].fd_splits == cur_sp)) {
                        icb_slot = ci; break;
                    }
                }
                if (icb_slot < 0) {
                    int free_i = -1;
                    for (int ci = 0; ci < MOE_ICB_CACHE_MAX; ci++) if (!m->icb_cache[ci].valid) { free_i = ci; break; }
                    if (free_i < 0) {
                        free_i = (int)(m->icb_captured % MOE_ICB_CACHE_MAX);
                        if (m->icb_cache[free_i].res_list) {
                            for (int r = 0; r < m->icb_cache[free_i].res_count; r++) m->icb_cache[free_i].res_list[r] = nil;
                            free(m->icb_cache[free_i].res_list);
                            m->icb_cache[free_i].res_list = NULL;
                        }
                        m->icb_cache[free_i].icb = nil; m->icb_cache[free_i].arg_buf = nil; m->icb_cache[free_i].valid = 0;
                    }
                    uint32_t max_cmds = (uint32_t)m->n_layers * 40u + 128u;
                    TenonIcbRecorder* rec = [[TenonIcbRecorder alloc] initWithDevice:g_moe_dev maxCommands:max_cmds];
                    if (rec) {
                        id<MTLComputeCommandEncoder> renc = (id<MTLComputeCommandEncoder>)rec;
                        id<MTLCommandBuffer> dummy_cmd = nil;
                        int64_t saved_fd_cnt = m->flash_decode_launches;
                        struct { uint32_t dm, gt, bb; } eb = { (uint32_t)m->d_model, m->tok_embd.ggml_type, B };
                        [renc setComputePipelineState:g_pso_embed_batch];
                        [renc setBuffer:m->tok_embd.buf offset:(NSUInteger)m->tok_embd.offset atIndex:0];
                        [renc setBuffer:m->buf_cb_tok offset:0 atIndex:1];
                        [renc setBuffer:m->pbuf_x offset:0 atIndex:2];
                        [renc setBytes:&eb length:sizeof(eb) atIndex:3];
                        [renc dispatchThreads:MTLSizeMake((NSUInteger)m->d_model, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        for (int L = 0; L < m->n_layers; L++) {
                            MoeLayer* ly = &m->layers[L];
                            encode_layer_attn_paged_batch(m, renc, L, B, nil);
                            if (ly->is_dense) {
                                encode_dense_ffn_batch(m, renc, ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                                                       ly->ffn_up.buf, ly->ffn_up.offset, ly->ffn_up.ggml_type,
                                                       ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
                            } else {
                                encode_grouped_moe_ffn_layer(m, &dummy_cmd, &renc, L, B);
                            }
                        }
                        encode_rmsnorm(renc, m->pbuf_x, &m->out_norm, m->pbuf_xn, (uint32_t)m->d_model, B, m->rms_eps, 1);
                        encode_gemm_batch_raw(renc, m->lm_head.buf, m->lm_head.offset, m->lm_head.ggml_type,
                                              m->pbuf_xn, m->buf_batch_logits, (uint32_t)m->vocab_size, (uint32_t)m->d_model, B, 0);
                        if (g_pso_argmax_b != nil) {
                            uint32_t vb[2] = { (uint32_t)m->vocab_size, B };
                            [renc setComputePipelineState:g_pso_argmax_b];
                            [renc setBuffer:m->buf_batch_logits offset:0 atIndex:0];
                            [renc setBuffer:m->buf_argmax offset:0 atIndex:1];
                            [renc setBytes:vb length:sizeof(vb) atIndex:2];
                            [renc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                        }
                        m->flash_decode_launches = saved_fd_cnt;
                        if (!rec->overflow && rec->n_cmds > 0u) {
                            m->icb_cache[free_i].valid = 1;
                            m->icb_cache[free_i].B = B;
                            m->icb_cache[free_i].lora_sig = lsig;
                            m->icb_cache[free_i].use_fd = use_fd;
                            m->icb_cache[free_i].fd_splits = cur_sp;
                            m->icb_cache[free_i].n_cmds = rec->n_cmds;
                            m->icb_cache[free_i].icb = rec->icb;
                            m->icb_cache[free_i].arg_buf = rec->arg_buf;
                            m->icb_cache[free_i].res_list = rec->res_list;
                            m->icb_cache[free_i].res_count = rec->res_count;
                            rec->res_list = NULL; rec->res_count = 0;
                            m->icb_captured++;
                            icb_slot = free_i;
                        } else if (rec->res_list) {
                            for (int r = 0; r < rec->res_count; r++) rec->res_list[r] = nil;
                            free(rec->res_list); rec->res_list = NULL;
                        }
                    }
                }
            }
            id<MTLCommandBuffer> cmd = (icb_slot >= 0) ? [g_moe_q commandBufferWithUnretainedReferences] : [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            if (icb_slot >= 0) {
                [enc useResources:m->icb_cache[icb_slot].res_list count:(NSUInteger)m->icb_cache[icb_slot].res_count
                            usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
                [enc executeCommandsInBuffer:m->icb_cache[icb_slot].icb withRange:NSMakeRange(0, m->icb_cache[icb_slot].n_cmds)];
                m->icb_replays++;
                if (use_fd) m->flash_decode_launches += m->n_layers;
            } else {
                struct { uint32_t dm, gt, bb; } eb = { (uint32_t)m->d_model, m->tok_embd.ggml_type, B };
                [enc setComputePipelineState:g_pso_embed_batch];
                [enc setBuffer:m->tok_embd.buf offset:(NSUInteger)m->tok_embd.offset atIndex:0];
                [enc setBuffer:m->buf_cb_tok offset:0 atIndex:1];
                [enc setBuffer:m->pbuf_x offset:0 atIndex:2];
                [enc setBytes:&eb length:sizeof(eb) atIndex:3];
                [enc dispatchThreads:MTLSizeMake((NSUInteger)m->d_model, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                for (int L = 0; L < m->n_layers; L++) {
                    MoeLayer* ly = &m->layers[L];
                    encode_layer_attn_paged_batch(m, enc, L, B, nil);
                    if (ly->is_dense) {
                        encode_dense_ffn_batch(m, enc, ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                                               ly->ffn_up.buf, ly->ffn_up.offset, ly->ffn_up.ggml_type,
                                               ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
                    } else {
                        encode_grouped_moe_ffn_layer(m, &cmd, &enc, L, B);
                    }
                }
                encode_rmsnorm(enc, m->pbuf_x, &m->out_norm, m->pbuf_xn, (uint32_t)m->d_model, B, m->rms_eps, 1);
                encode_gemm_batch_raw(enc, m->lm_head.buf, m->lm_head.offset, m->lm_head.ggml_type,
                                      m->pbuf_xn, m->buf_batch_logits, (uint32_t)m->vocab_size, (uint32_t)m->d_model, B, 0);
                if (g_pso_argmax_b != nil) {
                    uint32_t vb[2] = { (uint32_t)m->vocab_size, B };
                    [enc setComputePipelineState:g_pso_argmax_b];
                    [enc setBuffer:m->buf_batch_logits offset:0 atIndex:0];
                    [enc setBuffer:m->buf_argmax offset:0 atIndex:1];
                    [enc setBytes:vb length:sizeof(vb) atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
            }
            [enc endEncoding];
            if (icb_slot >= 0) {
                [cmd enqueue];
                [cmd commit];
                [cmd waitUntilCompleted];
                m->icb_pipe_overlaps++;
            } else {
                [cmd commit]; [cmd waitUntilCompleted];
            }
            m->cur_use_fd = 0;
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
            if (g_pso_argmax_b != nil) {
                uint32_t vb[2] = { (uint32_t)m->vocab_size, B };
                [fenc setComputePipelineState:g_pso_argmax_b];
                [fenc setBuffer:m->buf_batch_logits offset:0 atIndex:0];
                [fenc setBuffer:m->buf_argmax offset:0 atIndex:1];
                [fenc setBytes:vb length:sizeof(vb) atIndex:2];
                [fenc dispatchThreadgroups:MTLSizeMake(B, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [fenc endEncoding]; [fcmd commit]; [fcmd waitUntilCompleted];
        }
        if (g_pso_argmax_b != nil) {
            const uint32_t* am = (const uint32_t*)[m->buf_argmax contents];
            for (uint32_t b = 0; b < B; b++) out[b] = (int64_t)am[b];
            m->batch_logits_dirty_row = (int)(B - 1u);
        } else {
            const float* bl = (const float*)[m->buf_batch_logits contents];
            for (uint32_t b = 0; b < B; b++) {
                const float* row = bl + (size_t)b * (size_t)m->vocab_size;
                int best = 0; float bv = row[0];
                for (int v = 1; v < m->vocab_size; v++) if (row[v] > bv) { bv = row[v]; best = v; }
                out[b] = (int64_t)best;
            }
            memcpy([m->buf_logits contents], bl + (size_t)(B - 1u) * (size_t)m->vocab_size, (size_t)m->vocab_size * 4u);
            m->batch_logits_dirty_row = -1;
        }
        if (m->buf_last_hidden != nil) memcpy([m->buf_last_hidden contents], px + (size_t)(B - 1u) * (size_t)m->d_model, (size_t)m->d_model * 4u);
        m->last_total_ms = moe_now_ms() - t0;
        return (int64_t)B;
    }
}

// Tenon v8: Native Zero-Migration In-Slot Speculative Verification directly on PagedAttention `slot_id`!
// Because `moe_flash_decode_paged` & `moe_flash_mla_latent_paged` causally attend `0..pos[b]`,
// setting `pos[b] = start_pos + b` and `slot[b] = slot_id` verifies K+1 draft tokens IN-PLACE with ZERO KV migration!
int64_t tn_moe_paged_verify_batch(int64_t handle, int64_t slot_id, int64_t tok_ids_ptr, int64_t n_toks, int64_t start_pos, int64_t out_argmax_ptr) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !tok_ids_ptr || !out_argmax_ptr || n_toks <= 0) return 0;
    if (n_toks > MOE_PREFILL_MAX_B) n_toks = MOE_PREFILL_MAX_B;
    int64_t pos_arr[MOE_PREFILL_MAX_B];
    int64_t slot_arr[MOE_PREFILL_MAX_B];
    for (int64_t i = 0; i < n_toks; i++) {
        pos_arr[i] = start_pos + i;
        slot_arr[i] = slot_id;
    }
    return tn_moe_step_batch(handle, tok_ids_ptr, (int64_t)(uintptr_t)pos_arr, (int64_t)(uintptr_t)slot_arr, n_toks, out_argmax_ptr);
}

int64_t tn_moe_copy_logits_batch(int64_t handle, int64_t batch_idx, int64_t dst_f32_ptr) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !dst_f32_ptr) return 0;
    MoeModel* m = &g_moe[handle];
    if (m->buf_batch_logits == nil || batch_idx < 0 || batch_idx >= m->batch_logits_cap) return 0;
    const float* bl = (const float*)[m->buf_batch_logits contents];
    memcpy((void*)(uintptr_t)dst_f32_ptr, bl + (size_t)batch_idx * (size_t)m->vocab_size, (size_t)m->vocab_size * sizeof(float));
    return (int64_t)m->vocab_size;
}

// Batched Paged Prefill into PagedAttention sequence slot `slot_id` (enables instant COW Prefix Fork for Dense, MLA, and Resident MoE!)
int64_t tn_moe_paged_prefill(int64_t handle, int64_t slot_id, int64_t tok_ids_ptr, int64_t n_toks, int64_t start_pos) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !tok_ids_ptr || n_toks <= 0) return 0;
    MoeModel* m = &g_moe[handle];
    const int64_t* ids = (const int64_t*)(uintptr_t)tok_ids_ptr;
    int is_mla_m = (m->n_layers > 0 && m->layers[0].is_mla);
    int can_paged = m->dense_model
        ? (!is_mla_m || (m->mla_latent_kv && m->layers[0].mla_wuk_f16 != nil))
        : (m->resident_mode && (!is_mla_m || (m->mla_latent_kv && m->layers[0].mla_wuk_f16 != nil)));
    if (!can_paged) {
        tn_moe_kv_slot_set(handle, slot_id % (m->n_kv_slots > 0 ? m->n_kv_slots : 1));
        return tn_moe_prefill(handle, tok_ids_ptr, n_toks, start_pos);
    }
    @autoreleasepool {
        double t0 = moe_now_ms();
        int64_t last_tok = 0;
        int s = (int)slot_id; if (s < 0 || s >= PA_MAX_SEQS) s = 0;
        int chunk_sz = (m->d_ff_dense >= 20000) ? 128 : ((m->n_experts > 0) ? 256 : 512);
        if (chunk_sz > MOE_PREFILL_MAX_B) chunk_sz = MOE_PREFILL_MAX_B;
        if ((int64_t)chunk_sz > n_toks) chunk_sz = (int)(n_toks < 32 ? 32 : n_toks);
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
            m->cur_use_fd = 0;
            m->cur_is_prefill = 1;
            float* px = (float*)[m->pbuf_x contents];
            if (m->resident_mode && g_pso_embed_batch != nil) {
                uint32_t* t_arr = (uint32_t*)[m->buf_cb_tok contents];
                for (uint32_t bi = 0; bi < B; bi++) t_arr[bi] = (uint32_t)ids[off + bi];
                id<MTLCommandBuffer> cmd = [g_moe_q commandBufferWithUnretainedReferences];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                struct { uint32_t dm, gt, bb; } eb = { (uint32_t)m->d_model, m->tok_embd.ggml_type, B };
                [enc setComputePipelineState:g_pso_embed_batch];
                [enc setBuffer:m->tok_embd.buf offset:(NSUInteger)m->tok_embd.offset atIndex:0];
                [enc setBuffer:m->buf_cb_tok offset:0 atIndex:1];
                [enc setBuffer:m->pbuf_x offset:0 atIndex:2];
                [enc setBytes:&eb length:sizeof(eb) atIndex:3];
                [enc dispatchThreads:MTLSizeMake((NSUInteger)m->d_model, B, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                for (int L = 0; L < m->n_layers; L++) {
                    MoeLayer* ly = &m->layers[L];
                    encode_layer_attn_paged_batch(m, enc, L, B, nil);
                    if (ly->is_dense) {
                        encode_dense_ffn_batch(m, enc, ly->ffn_gate.buf, ly->ffn_gate.offset, ly->ffn_gate.ggml_type,
                                               ly->ffn_up.buf, ly->ffn_up.offset, ly->ffn_up.ggml_type,
                                               ly->ffn_down.buf, ly->ffn_down.offset, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
                    } else {
                        encode_grouped_moe_ffn_layer(m, &cmd, &enc, L, B);
                    }
                }
                if (off + B == n_toks) {
                    NSUInteger last_off = (NSUInteger)((size_t)(B - 1u) * (size_t)m->d_model * 4u);
                    struct { uint32_t dim, n_groups; float eps; uint32_t shared_w; } nargs = { (uint32_t)m->d_model, 1u, m->rms_eps, 0u };
                    [enc setComputePipelineState:g_pso_rmsnorm];
                    [enc setBuffer:m->pbuf_x offset:last_off atIndex:0];
                    [enc setBuffer:m->out_norm.buf offset:(NSUInteger)m->out_norm.offset atIndex:1];
                    [enc setBuffer:m->buf_xn offset:0 atIndex:2];
                    [enc setBytes:&nargs length:sizeof(nargs) atIndex:3];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    encode_gemv(enc, &m->lm_head, m->buf_xn, m->buf_logits, (uint32_t)m->vocab_size, (uint32_t)m->d_model, 0);
                    uint32_t v_u = (uint32_t)m->vocab_size;
                    [enc setComputePipelineState:g_pso_argmax];
                    [enc setBuffer:m->buf_logits offset:0 atIndex:0];
                    [enc setBuffer:m->buf_argmax offset:0 atIndex:1];
                    [enc setBytes:&v_u length:4 atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    m->batch_logits_dirty_row = -1;
                }
                [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
                m->cur_is_prefill = 0;
                if (off + B == n_toks) {
                    if (m->buf_last_hidden != nil) memcpy([m->buf_last_hidden contents], px + (size_t)(B - 1u) * (size_t)m->d_model, (size_t)m->d_model * 4u);
                    last_tok = (int64_t)((uint32_t*)[m->buf_argmax contents])[0];
                }
            } else {
                for (uint32_t bi = 0; bi < B; bi++) {
                    id<MTLCommandBuffer> ecmd = [g_moe_q commandBuffer];
                    id<MTLComputeCommandEncoder> eenc = [ecmd computeCommandEncoder];
                    encode_embed_step(m, eenc, (uint32_t)ids[off + bi]);
                    [eenc endEncoding]; [ecmd commit]; [ecmd waitUntilCompleted];
                    memcpy(px + (size_t)bi * (size_t)m->d_model, [m->buf_x contents], (size_t)m->d_model * 4u);
                }
                run_dense_paged_layers_batch(m, B);
                m->cur_is_prefill = 0;
                if (off + B == n_toks) {
                    memcpy([m->buf_x contents], px + (size_t)(B - 1u) * (size_t)m->d_model, (size_t)m->d_model * 4u);
                    id<MTLCommandBuffer> fcmd = [g_moe_q commandBuffer];
                    id<MTLComputeCommandEncoder> fenc = [fcmd computeCommandEncoder];
                    encode_final_lm_head(m, fenc);
                    [fenc endEncoding]; [fcmd commit]; [fcmd waitUntilCompleted];
                    if (m->buf_last_hidden != nil) memcpy([m->buf_last_hidden contents], [m->buf_x contents], (size_t)m->d_model * 4u);
                    last_tok = (int64_t)((uint32_t*)[m->buf_argmax contents])[0];
                }
            }
            off += B;
        }
        m->last_total_ms = moe_now_ms() - t0;
        return last_tok;
    }
}

// Tenon v9: PagedKV-Block-Level Multimodal Embedding Prefill (`tn_moe_paged_step_embd`)!
// Writes vision/audio embeddings directly into PagedAttention blocks of `slot_id` so repeated
// multimodal prefixes share GPU KV blocks via ZERO-COPY `tn_moe_paged_seq_fork` (0.0ms ViT + 0.0ms LLM prefill!)
int64_t tn_moe_step_embd(int64_t handle, int64_t embd_f32_ptr, int64_t n_vecs, int64_t start_pos);
int64_t tn_moe_step_embd_mrope(int64_t handle, int64_t embd_f32_ptr, int64_t n_vecs, int64_t start_pos, int64_t grid_t, int64_t grid_h, int64_t grid_w);
int64_t tn_moe_paged_step_embd(
    int64_t handle, int64_t slot_id, int64_t embd_f32_ptr, int64_t n_vecs, int64_t start_pos,
    int64_t grid_t, int64_t grid_h, int64_t grid_w
) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !embd_f32_ptr || n_vecs <= 0) return 0;
    MoeModel* m = &g_moe[handle];
    int is_mla_m = (m->n_layers > 0 && m->layers[0].is_mla);
    int can_paged = m->dense_model
        ? (!is_mla_m || (m->mla_latent_kv && m->layers[0].mla_wuk_f16 != nil))
        : (m->resident_mode && (!is_mla_m || (m->mla_latent_kv && m->layers[0].mla_wuk_f16 != nil)));
    if (!can_paged) {
        tn_moe_kv_slot_set(handle, slot_id % (m->n_kv_slots > 0 ? m->n_kv_slots : 1));
        if (grid_t > 1 || grid_h > 1 || grid_w > 1) {
            return tn_moe_step_embd_mrope(handle, embd_f32_ptr, n_vecs, start_pos, grid_t, grid_h, grid_w);
        }
        return tn_moe_step_embd(handle, embd_f32_ptr, n_vecs, start_pos);
    }
    @autoreleasepool {
        double t0 = moe_now_ms();
        int s = (int)slot_id; if (s < 0 || s >= PA_MAX_SEQS) s = 0;
        const float* src = (const float*)(uintptr_t)embd_f32_ptr;
        int chunk_sz = 32;
        ensure_prefill_bufs(m, chunk_sz);
        int64_t last_tok = 0, off = 0;
        while (off < n_vecs) {
            uint32_t B = (uint32_t)((n_vecs - off > chunk_sz) ? chunk_sz : (n_vecs - off));
            uint32_t* p_arr = (uint32_t*)[m->buf_cb_pos contents];
            uint32_t* s_arr = (uint32_t*)[m->buf_cb_slot contents];
            for (uint32_t b = 0; b < B; b++) {
                int p = (int)(start_pos + off + b);
                if (!pa_ensure_slot_pos(m, s, p)) return 0;
                p_arr[b] = (uint32_t)p;
                s_arr[b] = (uint32_t)s;
            }
            pa_sync_bt(m);
            memcpy([m->pbuf_x contents], src + (size_t)off * (size_t)m->d_model, (size_t)B * (size_t)m->d_model * sizeof(float));
            run_dense_paged_layers_batch(m, B);
            if (off + B == n_vecs) {
                float* px = (float*)[m->pbuf_x contents];
                memcpy([m->buf_x contents], px + (size_t)(B - 1u) * (size_t)m->d_model, (size_t)m->d_model * sizeof(float));
                id<MTLCommandBuffer> fcmd = [g_moe_q commandBuffer];
                id<MTLComputeCommandEncoder> fenc = [fcmd computeCommandEncoder];
                encode_final_lm_head(m, fenc);
                [fenc endEncoding]; [fcmd commit]; [fcmd waitUntilCompleted];
                if (m->buf_last_hidden != nil) memcpy([m->buf_last_hidden contents], [m->buf_x contents], (size_t)m->d_model * sizeof(float));
                last_tok = (int64_t)((uint32_t*)[m->buf_argmax contents])[0];
            }
            off += B;
        }
        m->last_total_ms = moe_now_ms() - t0;
        return last_tok;
    }
}

// =========================================================================
// Tenon v9: Disaggregated Prefill-Decode (PD) PagedKV Block Serialization & Transfer!
// =========================================================================
typedef struct {
    uint32_t magic;       // 0x544E4B56u ("TNKV")
    uint32_t version;     // 1
    uint32_t n_layers;
    uint32_t n_tokens;
    uint32_t n_blocks;
    uint32_t kv_flags;    // bit0=q8, bit1=mla_latent, bit2=tp_sharded
    uint32_t tp_size;
    uint32_t blk_bytes;
    uint64_t payload_len;
    uint64_t fnv1a_hash;
} TenonKvBlobHeader;

static uint64_t tn_kv_fnv1a64(const uint8_t* p, size_t n) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; i++) { h ^= (uint64_t)p[i]; h *= 1099511628211ull; }
    return h;
}

int64_t tn_moe_kv_slot_bytes(int64_t handle, int64_t slot_id, int64_t n_tokens) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || slot_id < 0 || slot_id >= PA_MAX_SEQS) return 0;
    MoeModel* m = &g_moe[handle];
    int nt = (n_tokens > 0) ? (int)n_tokens : m->pa_seq_len[slot_id];
    if (nt <= 0) return 0;
    int nb = (nt + PA_BLOCK_SIZE - 1) / PA_BLOCK_SIZE;
    int is_mla_lat = (m->n_layers > 0 && m->layers[0].is_mla && m->mla_latent_kv) ? 1 : 0;
    size_t blk_b = pa_kv_block_bytes(m);
    uint64_t payload = (uint64_t)m->n_layers * (uint64_t)nb * (uint64_t)blk_b * (uint64_t)(is_mla_lat ? 1u : 2u);
    return (int64_t)(sizeof(TenonKvBlobHeader) + payload);
}

int64_t tn_moe_kv_export_slot(int64_t handle, int64_t slot_id, int64_t n_tokens, int64_t dst_ptr, int64_t dst_cap) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !dst_ptr || slot_id < 0 || slot_id >= PA_MAX_SEQS) return 0;
    MoeModel* m = &g_moe[handle];
    int nt = (n_tokens > 0) ? (int)n_tokens : m->pa_seq_len[slot_id];
    if (nt <= 0) return 0;
    int64_t need = tn_moe_kv_slot_bytes(handle, slot_id, nt);
    if (need <= 0 || dst_cap < need) return 0;
    int nb = (nt + PA_BLOCK_SIZE - 1) / PA_BLOCK_SIZE;
    int is_mla_lat = (m->n_layers > 0 && m->layers[0].is_mla && m->mla_latent_kv) ? 1 : 0;
    size_t blk_b = pa_kv_block_bytes(m);
    uint8_t* dst = (uint8_t*)(uintptr_t)dst_ptr;
    uint8_t* pay = dst + sizeof(TenonKvBlobHeader);
    size_t cur = 0;
    for (int L = 0; L < m->n_layers; L++) {
        MoeLayer* ly = &m->layers[L];
        const uint8_t* kp = ly->k_cache != nil ? (const uint8_t*)[ly->k_cache contents] : NULL;
        const uint8_t* vp = (!is_mla_lat && ly->v_cache != nil) ? (const uint8_t*)[ly->v_cache contents] : NULL;
        for (int lb = 0; lb < nb; lb++) {
            int32_t pb = m->pa_block_table[slot_id][lb];
            if (pb < 0 || pb >= m->pa_num_blocks || !kp) return 0;
            memcpy(pay + cur, kp + (size_t)pb * blk_b, blk_b); cur += blk_b;
            if (!is_mla_lat) {
                if (!vp) return 0;
                memcpy(pay + cur, vp + (size_t)pb * blk_b, blk_b); cur += blk_b;
            }
        }
    }
    TenonKvBlobHeader hdr;
    hdr.magic = 0x544E4B56u;
    hdr.version = 1u;
    hdr.n_layers = (uint32_t)m->n_layers;
    hdr.n_tokens = (uint32_t)nt;
    hdr.n_blocks = (uint32_t)nb;
    hdr.kv_flags = (uint32_t)((m->kv_quant_q8 == 1 ? 1u : 0u) | (is_mla_lat ? 2u : 0u) | (((uint32_t)(m->kv_quant_q8 & 3)) << 3));
    hdr.tp_size = 1u;
    hdr.blk_bytes = (uint32_t)blk_b;
    hdr.payload_len = (uint64_t)cur;
    hdr.fnv1a_hash = tn_kv_fnv1a64(pay, cur);
    memcpy(dst, &hdr, sizeof(hdr));
    return (int64_t)(sizeof(hdr) + cur);
}

int64_t tn_moe_kv_import_slot(int64_t handle, int64_t slot_id, int64_t src_ptr, int64_t src_len) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !src_ptr || slot_id < 0 || slot_id >= PA_MAX_SEQS) return 0;
    if ((size_t)src_len < sizeof(TenonKvBlobHeader)) return 0;
    MoeModel* m = &g_moe[handle];
    const uint8_t* src = (const uint8_t*)(uintptr_t)src_ptr;
    TenonKvBlobHeader hdr;
    memcpy(&hdr, src, sizeof(hdr));
    int is_mla_lat = (m->n_layers > 0 && m->layers[0].is_mla && m->mla_latent_kv) ? 1 : 0;
    uint32_t exp_flags = (uint32_t)((m->kv_quant_q8 == 1 ? 1u : 0u) | (is_mla_lat ? 2u : 0u) | (((uint32_t)(m->kv_quant_q8 & 3)) << 3));
    size_t blk_b = pa_kv_block_bytes(m);
    if (hdr.magic != 0x544E4B56u || hdr.version != 1u || hdr.n_layers != (uint32_t)m->n_layers ||
        hdr.kv_flags != exp_flags || hdr.tp_size != 1u || hdr.blk_bytes != (uint32_t)blk_b) return 0;
    if (sizeof(hdr) + hdr.payload_len > (uint64_t)src_len) return 0;
    const uint8_t* pay = src + sizeof(hdr);
    if (tn_kv_fnv1a64(pay, (size_t)hdr.payload_len) != hdr.fnv1a_hash) return 0;
    tn_moe_paged_seq_free(handle, slot_id);
    for (uint32_t lb = 0; lb < hdr.n_blocks; lb++) {
        int pb = pa_alloc_block(m);
        if (pb < 0) { tn_moe_paged_seq_free(handle, slot_id); return 0; }
        m->pa_block_table[slot_id][lb] = pb;
    }
    size_t cur = 0;
    for (int L = 0; L < m->n_layers; L++) {
        MoeLayer* ly = &m->layers[L];
        uint8_t* kp = ly->k_cache != nil ? (uint8_t*)[ly->k_cache contents] : NULL;
        uint8_t* vp = (!is_mla_lat && ly->v_cache != nil) ? (uint8_t*)[ly->v_cache contents] : NULL;
        for (uint32_t lb = 0; lb < hdr.n_blocks; lb++) {
            int32_t pb = m->pa_block_table[slot_id][lb];
            if (!kp) return 0;
            memcpy(kp + (size_t)pb * blk_b, pay + cur, blk_b); cur += blk_b;
            if (!is_mla_lat) {
                if (!vp) return 0;
                memcpy(vp + (size_t)pb * blk_b, pay + cur, blk_b); cur += blk_b;
            }
        }
    }
    m->pa_seq_len[slot_id] = (int)hdr.n_tokens;
    pa_sync_bt(m);
    return (int64_t)hdr.n_tokens;
}

// Tenon v10: Zero-Copy POSIX Shared-Memory (`shm_open` / `mmap`) Disaggregated PD Transport!
int64_t tn_moe_kv_export_shm(int64_t handle, int64_t slot_id, int64_t n_tokens, int64_t shm_name_ptr) {
    if (!shm_name_ptr) return 0;
    const char* name = (const char*)(uintptr_t)shm_name_ptr;
    int64_t need = tn_moe_kv_slot_bytes(handle, slot_id, n_tokens);
    if (need <= 0) return 0;
    shm_unlink(name);
    int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
    if (fd < 0) return 0;
    if (ftruncate(fd, (off_t)need) != 0) { close(fd); shm_unlink(name); return 0; }
    void* ptr = mmap(NULL, (size_t)need, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) { shm_unlink(name); return 0; }
    int64_t written = tn_moe_kv_export_slot(handle, slot_id, n_tokens, (int64_t)(uintptr_t)ptr, need);
    munmap(ptr, (size_t)need);
    if (written <= 0) { shm_unlink(name); return 0; }
    return written;
}

int64_t tn_moe_kv_import_shm(int64_t handle, int64_t slot_id, int64_t shm_name_ptr, int64_t unlink_after) {
    if (!shm_name_ptr) return 0;
    const char* name = (const char*)(uintptr_t)shm_name_ptr;
    int fd = shm_open(name, O_RDONLY, 0);
    if (fd < 0) return 0;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { close(fd); if (unlink_after) shm_unlink(name); return 0; }
    void* ptr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) { if (unlink_after) shm_unlink(name); return 0; }
    int64_t nt = tn_moe_kv_import_slot(handle, slot_id, (int64_t)(uintptr_t)ptr, (int64_t)st.st_size);
    munmap(ptr, (size_t)st.st_size);
    if (unlink_after) shm_unlink(name);
    return nt;
}

// Tenon v11: Layer-Wise Streaming Pipelined Disaggregated PD (`TNKL` header + per-layer-group chunks!)
typedef struct {
    uint32_t magic;        // 0x544E4B4C ("TNKL")
    uint32_t version;      // 1
    uint32_t layer_start;
    uint32_t layer_count;
    uint32_t total_layers;
    uint32_t n_tokens;
    uint32_t n_blocks;
    uint32_t kv_flags;
    uint32_t tp_size;
    uint32_t blk_bytes;
    uint64_t payload_len;
    uint64_t fnv1a_hash;
} TenonKvLayerHeader;

int64_t tn_moe_kv_layer_bytes(int64_t handle, int64_t slot_id, int64_t n_tokens, int64_t layer_start, int64_t layer_count) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || slot_id < 0 || slot_id >= PA_MAX_SEQS) return 0;
    MoeModel* m = &g_moe[handle];
    int nt = (n_tokens > 0) ? (int)n_tokens : m->pa_seq_len[slot_id];
    if (nt <= 0 || layer_start < 0 || layer_count <= 0 || layer_start + layer_count > m->n_layers) return 0;
    int nb = (nt + PA_BLOCK_SIZE - 1) / PA_BLOCK_SIZE;
    int is_mla_lat = (m->n_layers > 0 && m->layers[0].is_mla && m->mla_latent_kv) ? 1 : 0;
    size_t blk_b = pa_kv_block_bytes(m);
    size_t pay = (size_t)layer_count * (size_t)nb * blk_b * (is_mla_lat ? 1u : 2u);
    return (int64_t)(sizeof(TenonKvLayerHeader) + pay);
}

int64_t tn_moe_kv_export_layer(int64_t handle, int64_t slot_id, int64_t n_tokens, int64_t layer_start, int64_t layer_count, int64_t dst_ptr, int64_t dst_cap) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !dst_ptr || slot_id < 0 || slot_id >= PA_MAX_SEQS) return 0;
    MoeModel* m = &g_moe[handle];
    int nt = (n_tokens > 0) ? (int)n_tokens : m->pa_seq_len[slot_id];
    int64_t need = tn_moe_kv_layer_bytes(handle, slot_id, nt, layer_start, layer_count);
    if (need <= 0 || dst_cap < need) return 0;
    int nb = (nt + PA_BLOCK_SIZE - 1) / PA_BLOCK_SIZE;
    int is_mla_lat = (m->n_layers > 0 && m->layers[0].is_mla && m->mla_latent_kv) ? 1 : 0;
    size_t blk_b = pa_kv_block_bytes(m);
    uint8_t* dst = (uint8_t*)(uintptr_t)dst_ptr;
    uint8_t* pay = dst + sizeof(TenonKvLayerHeader);
    size_t cur = 0;
    for (int L = (int)layer_start; L < (int)(layer_start + layer_count); L++) {
        MoeLayer* ly = &m->layers[L];
        const uint8_t* kp = ly->k_cache != nil ? (const uint8_t*)[ly->k_cache contents] : NULL;
        const uint8_t* vp = (!is_mla_lat && ly->v_cache != nil) ? (const uint8_t*)[ly->v_cache contents] : NULL;
        for (int lb = 0; lb < nb; lb++) {
            int32_t pb = m->pa_block_table[slot_id][lb];
            if (pb < 0 || pb >= m->pa_num_blocks || !kp) return 0;
            memcpy(pay + cur, kp + (size_t)pb * blk_b, blk_b); cur += blk_b;
            if (!is_mla_lat) {
                if (!vp) return 0;
                memcpy(pay + cur, vp + (size_t)pb * blk_b, blk_b); cur += blk_b;
            }
        }
    }
    TenonKvLayerHeader hdr;
    hdr.magic = 0x544E4B4Cu; hdr.version = 1u;
    hdr.layer_start = (uint32_t)layer_start; hdr.layer_count = (uint32_t)layer_count;
    hdr.total_layers = (uint32_t)m->n_layers; hdr.n_tokens = (uint32_t)nt; hdr.n_blocks = (uint32_t)nb;
    hdr.kv_flags = (uint32_t)((m->kv_quant_q8 == 1 ? 1u : 0u) | (is_mla_lat ? 2u : 0u) | (((uint32_t)(m->kv_quant_q8 & 3)) << 3));
    hdr.tp_size = 1u; hdr.blk_bytes = (uint32_t)blk_b;
    hdr.payload_len = (uint64_t)cur; hdr.fnv1a_hash = tn_kv_fnv1a64(pay, cur);
    memcpy(dst, &hdr, sizeof(hdr));
    return (int64_t)(sizeof(hdr) + cur);
}

int64_t tn_moe_kv_import_layer(int64_t handle, int64_t slot_id, int64_t src_ptr, int64_t src_len) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !src_ptr || slot_id < 0 || slot_id >= PA_MAX_SEQS) return 0;
    if ((size_t)src_len < sizeof(TenonKvLayerHeader)) return 0;
    MoeModel* m = &g_moe[handle];
    const uint8_t* src = (const uint8_t*)(uintptr_t)src_ptr;
    TenonKvLayerHeader hdr;
    memcpy(&hdr, src, sizeof(hdr));
    int is_mla_lat = (m->n_layers > 0 && m->layers[0].is_mla && m->mla_latent_kv) ? 1 : 0;
    uint32_t exp_flags = (uint32_t)((m->kv_quant_q8 == 1 ? 1u : 0u) | (is_mla_lat ? 2u : 0u) | (((uint32_t)(m->kv_quant_q8 & 3)) << 3));
    size_t blk_b = pa_kv_block_bytes(m);
    if (hdr.magic != 0x544E4B4Cu || hdr.version != 1u || hdr.total_layers != (uint32_t)m->n_layers ||
        hdr.layer_start + hdr.layer_count > hdr.total_layers || hdr.layer_count == 0u ||
        hdr.kv_flags != exp_flags || hdr.tp_size != 1u || hdr.blk_bytes != (uint32_t)blk_b) return 0;
    if (sizeof(hdr) + hdr.payload_len > (uint64_t)src_len) return 0;
    const uint8_t* pay = src + sizeof(hdr);
    if (tn_kv_fnv1a64(pay, (size_t)hdr.payload_len) != hdr.fnv1a_hash) return 0;
    if (hdr.layer_start == 0u) {
        tn_moe_paged_seq_free(handle, slot_id);
        for (uint32_t lb = 0; lb < hdr.n_blocks; lb++) {
            int pb = pa_alloc_block(m);
            if (pb < 0) { tn_moe_paged_seq_free(handle, slot_id); return 0; }
            m->pa_block_table[slot_id][lb] = pb;
        }
    }
    size_t cur = 0;
    for (uint32_t L = hdr.layer_start; L < hdr.layer_start + hdr.layer_count; L++) {
        MoeLayer* ly = &m->layers[L];
        uint8_t* kp = ly->k_cache != nil ? (uint8_t*)[ly->k_cache contents] : NULL;
        uint8_t* vp = (!is_mla_lat && ly->v_cache != nil) ? (uint8_t*)[ly->v_cache contents] : NULL;
        for (uint32_t lb = 0; lb < hdr.n_blocks; lb++) {
            int32_t pb = m->pa_block_table[slot_id][lb];
            if (pb < 0 || pb >= m->pa_num_blocks || !kp) return 0;
            memcpy(kp + (size_t)pb * blk_b, pay + cur, blk_b); cur += blk_b;
            if (!is_mla_lat) {
                if (!vp) return 0;
                memcpy(vp + (size_t)pb * blk_b, pay + cur, blk_b); cur += blk_b;
            }
        }
    }
    if (hdr.layer_start + hdr.layer_count == hdr.total_layers) {
        m->pa_seq_len[slot_id] = (int)hdr.n_tokens;
        pa_sync_bt(m);
    }
    return (int64_t)hdr.n_tokens;
}

// =========================================================================
// Tenon v6: Tree-Attention Speculative Verification (`verify_tree` + `commit_tree_path`)
// =========================================================================
static void encode_layer_attn_tree(MoeModel* m, id<MTLComputeCommandEncoder> enc, int L, uint32_t start_pos, uint32_t B, id<MTLBuffer> ping_buf) {
    m->cur_exec_layer = L;
    MoeLayer* ly = &m->layers[L];
    uint32_t dm = (uint32_t)m->d_model, qdim = (uint32_t)(m->n_heads * m->d_head), kvdim = (uint32_t)(m->n_kv * m->d_head);
    uint32_t wo_in = (ly->wo.ne0 > 0) ? (uint32_t)ly->wo.ne0 : qdim;
    uint32_t out_stride = wo_in / (uint32_t)m->n_heads;
    float attn_scale = (m->attn_mscale > 0.0f ? m->attn_mscale : 1.0f) / sqrtf((float)m->d_head);
    encode_rmsnorm(enc, m->pbuf_x, &ly->attn_norm, m->pbuf_xn, dm, B, m->rms_eps, 1);
    int has_qkv_lora = (m->active_lora > 0 || m->use_batch_lora);
    int qkv_bias_fused = 0;
    if (B == 1u && !has_qkv_lora && g_pso_gemv_qkv_q4 != nil &&
        ly->wq.ggml_type == TN_GGML_Q4_0 && ly->wk.ggml_type == TN_GGML_Q4_0 && ly->wv.ggml_type == TN_GGML_Q4_0) {
        // Tenon v13: Single-dispatch fused Q||K||V + Optional QKV Bias for B=1 Paged Decode (saves 2-5 dispatches/layer!)
        qkv_bias_fused = (ly->has_qkv_bias && ly->bq.buf != nil && ly->bk.buf != nil && ly->bv.buf != nil);
        struct { uint32_t q_dim, kv_dim, in_dim, has_bias; } qa = { qdim, kvdim, dm, (uint32_t)qkv_bias_fused };
        id<MTLBuffer> qb = (ping_buf != nil) ? ping_buf : ly->wq.buf;
        id<MTLBuffer> kb = (ping_buf != nil) ? ping_buf : ly->wk.buf;
        id<MTLBuffer> vb = (ping_buf != nil) ? ping_buf : ly->wv.buf;
        NSUInteger qo = (NSUInteger)((ping_buf != nil) ? m->dense_off_wq : ly->wq.offset);
        NSUInteger ko = (NSUInteger)((ping_buf != nil) ? m->dense_off_wk : ly->wk.offset);
        NSUInteger vo = (NSUInteger)((ping_buf != nil) ? m->dense_off_wv : ly->wv.offset);
        [enc setComputePipelineState:g_pso_gemv_qkv_q4];
        [enc setBuffer:qb offset:qo atIndex:0];
        [enc setBuffer:kb offset:ko atIndex:1];
        [enc setBuffer:vb offset:vo atIndex:2];
        [enc setBuffer:m->pbuf_xn offset:0 atIndex:3];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:4];
        [enc setBuffer:m->pbuf_k offset:0 atIndex:5];
        [enc setBuffer:m->pbuf_v offset:0 atIndex:6];
        [enc setBytes:&qa length:sizeof(qa) atIndex:7];
        [enc setBuffer:(qkv_bias_fused ? ly->bq.buf : m->buf_argmax) offset:(qkv_bias_fused ? (NSUInteger)ly->bq.offset : 0) atIndex:8];
        [enc setBuffer:(qkv_bias_fused ? ly->bk.buf : m->buf_argmax) offset:(qkv_bias_fused ? (NSUInteger)ly->bk.offset : 0) atIndex:9];
        [enc setBuffer:(qkv_bias_fused ? ly->bv.buf : m->buf_argmax) offset:(qkv_bias_fused ? (NSUInteger)ly->bv.offset : 0) atIndex:10];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((qdim + 2u * kvdim + 7u) / 8u), 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else if (B >= 33u && !has_qkv_lora && ping_buf == nil && g_pso_gemm_q4_qkv_b != nil &&
               ly->wq.ggml_type == TN_GGML_Q4_0 && ly->wk.ggml_type == TN_GGML_Q4_0 && ly->wv.ggml_type == TN_GGML_Q4_0 &&
               (qdim & 63u) == 0u && (kvdim & 63u) == 0u) {
        struct { uint32_t q_dim, kv_dim, in_dim, batch_b; } qa = { qdim, kvdim, dm, B };
        [enc setComputePipelineState:g_pso_gemm_q4_qkv_b];
        [enc setBuffer:ly->wq.buf offset:(NSUInteger)ly->wq.offset atIndex:0];
        [enc setBuffer:ly->wk.buf offset:(NSUInteger)ly->wk.offset atIndex:1];
        [enc setBuffer:ly->wv.buf offset:(NSUInteger)ly->wv.offset atIndex:2];
        [enc setBuffer:m->pbuf_xn offset:0 atIndex:3];
        [enc setBuffer:m->pbuf_q offset:0 atIndex:4];
        [enc setBuffer:m->pbuf_k offset:0 atIndex:5];
        [enc setBuffer:m->pbuf_v offset:0 atIndex:6];
        [enc setBytes:&qa length:sizeof(qa) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)((qdim + 2u * kvdim + 63u) / 64u), (NSUInteger)((B + 31u) / 32u), 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else if (ping_buf != nil) {
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wq, ly->wq.ggml_type, m->pbuf_xn, m->pbuf_q, qdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wk, ly->wk.ggml_type, m->pbuf_xn, m->pbuf_k, kvdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ping_buf, m->dense_off_wv, ly->wv.ggml_type, m->pbuf_xn, m->pbuf_v, kvdim, dm, B, 0);
    } else {
        encode_gemm_batch_raw(enc, ly->wq.buf, ly->wq.offset, ly->wq.ggml_type, m->pbuf_xn, m->pbuf_q, qdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ly->wk.buf, ly->wk.offset, ly->wk.ggml_type, m->pbuf_xn, m->pbuf_k, kvdim, dm, B, 0);
        encode_gemm_batch_raw(enc, ly->wv.buf, ly->wv.offset, ly->wv.ggml_type, m->pbuf_xn, m->pbuf_v, kvdim, dm, B, 0);
    }
    if (m->active_lora > 0 || m->use_batch_lora) {
        encode_lora_delta(m, enc, L, 0, m->pbuf_xn, m->pbuf_q, qdim, dm, B);
        encode_lora_delta(m, enc, L, 1, m->pbuf_xn, m->pbuf_k, kvdim, dm, B);
        encode_lora_delta(m, enc, L, 2, m->pbuf_xn, m->pbuf_v, kvdim, dm, B);
    }
    if (ly->has_qkv_bias && !qkv_bias_fused) {
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
    if (m->active_lora > 0 || m->use_batch_lora) encode_lora_delta(m, enc, L, 3, m->pbuf_ctx, m->pbuf_x, dm, wo_in, B);
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
            if (ml->gate_exps.buf != nil && ml->up_exps.buf != nil && ml->down_exps.buf != nil && ml->router.buf != nil && m->n_experts > 0) {
                // Full DeepSeek-V3/R1 Routed MoE + Optional SharedExp MTP Block!
                encode_gemv(e3, &ml->router, m->buf_xn, m->buf_rlogits, (uint32_t)m->n_experts, dm, 0);
                struct { uint32_t ne, tk, nw, gf, hb; float ws; uint32_t ng, ngu; } rt = {
                    (uint32_t)m->n_experts, (uint32_t)m->top_k, (uint32_t)m->expert_weights_norm,
                    (uint32_t)m->gating_func, (uint32_t)ml->has_exp_bias, m->expert_weights_scale,
                    (uint32_t)m->n_expert_groups, (uint32_t)m->n_group_used
                };
                [e3 setComputePipelineState:g_pso_router];
                [e3 setBuffer:m->buf_rlogits offset:0 atIndex:0];
                [e3 setBuffer:(ml->has_exp_bias ? ml->exp_bias.buf : m->buf_argmax)
                        offset:(ml->has_exp_bias ? (NSUInteger)ml->exp_bias.offset : 0) atIndex:1];
                [e3 setBuffer:m->buf_eidx offset:0 atIndex:2];
                [e3 setBuffer:m->buf_ewt offset:0 atIndex:3];
                [e3 setBytes:&rt length:sizeof(rt) atIndex:4];
                [e3 dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                if (ml->has_shared_exp && ml->sh_gate.buf != nil) {
                    uint32_t sf = (ml->sh_gate.ne1 > 0) ? (uint32_t)ml->sh_gate.ne1 : (uint32_t)m->d_ff_exp;
                    encode_dense_ffn_raw(m, e3, ml->sh_gate.buf, ml->sh_gate.offset, ml->sh_gate.ggml_type,
                                         ml->sh_up.buf, ml->sh_up.offset, ml->sh_up.ggml_type,
                                         ml->sh_down.buf, ml->sh_down.offset, ml->sh_down.ggml_type, sf);
                }
                struct { uint32_t dm, df, na; uint64_t gb, ub, db, sb; } margs = {
                    dm, (uint32_t)m->d_ff_exp, (uint32_t)m->top_k,
                    ml->gate_exp_bytes, ml->up_exp_bytes, ml->down_exp_bytes, 0
                };
                int is_q4k_gu = (ml->gate_exps.ggml_type == TN_GGML_Q4_K);
                [e3 setComputePipelineState:(is_q4k_gu ? g_pso_moe_gu_q4k : g_pso_moe_gu_q4)];
                [e3 setBuffer:ml->gate_exps.buf offset:(NSUInteger)ml->gate_exps.offset atIndex:0];
                [e3 setBuffer:ml->up_exps.buf offset:(NSUInteger)ml->up_exps.offset atIndex:1];
                [e3 setBuffer:m->buf_xn offset:0 atIndex:2];
                [e3 setBuffer:m->buf_eidx offset:0 atIndex:3];
                [e3 setBuffer:m->buf_h offset:0 atIndex:4];
                [e3 setBytes:&margs length:sizeof(margs) atIndex:5];
                [e3 dispatchThreadgroups:MTLSizeMake((NSUInteger)((m->d_ff_exp + 3) / 4), (NSUInteger)m->top_k, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                id<MTLComputePipelineState> r_dn_pso = nil;
                NSUInteger r_dn_tgs = 0, r_dn_tptg = 64;
                select_moe_down_pso(ml->down_exps.ggml_type, 0, m->d_model, &r_dn_pso, &r_dn_tgs, &r_dn_tptg);
                [e3 setComputePipelineState:r_dn_pso];
                [e3 setBuffer:ml->down_exps.buf offset:(NSUInteger)ml->down_exps.offset atIndex:0];
                [e3 setBuffer:m->buf_h offset:0 atIndex:1];
                [e3 setBuffer:m->buf_eidx offset:0 atIndex:2];
                [e3 setBuffer:m->buf_ewt offset:0 atIndex:3];
                [e3 setBuffer:m->buf_x offset:0 atIndex:4];
                [e3 setBytes:&margs length:sizeof(margs) atIndex:5];
                [e3 dispatchThreadgroups:MTLSizeMake(r_dn_tgs, 1, 1) threadsPerThreadgroup:MTLSizeMake(r_dn_tptg, 1, 1)];
            } else if (ml->ffn_gate.buf != nil && ml->ffn_up.buf != nil && ml->ffn_down.buf != nil) {
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
        if (n_vecs > 1 && m->has_mrope != 2 && m->dense_model && m->resident_mode && !(m->n_layers > 0 && m->layers[0].is_mla)) {
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
    int is_qwen2vl;
    int spatial_merge_size;
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
            else if (strcmp(key, "clip.vision.spatial_merge_size") == 0) { mm->spatial_merge_size = (int)gr_read_int(&gr, vt); mm->is_qwen2vl = 1; }
            else if (strcmp(key, "clip.use_gelu") == 0) mm->use_gelu = (int)gr_read_int(&gr, vt);
            else if (strcmp(key, "clip.vision.attention.layer_norm_epsilon") == 0 && vt == GGUF_T_F32) mm->eps = gr_f32(&gr);
            else if (strcmp(key, "clip.projector_type") == 0 && vt == GGUF_T_STR) {
                char* pt = gr_str(&gr);
                if (strstr(pt, "idefics3") != NULL) mm->is_idefics3 = 1;
                if (strstr(pt, "qwen2") != NULL) { mm->is_qwen2vl = 1; if (mm->spatial_merge_size <= 0) mm->spatial_merge_size = 2; }
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
        int merge_s = (mm->is_qwen2vl && mm->spatial_merge_size > 1) ? mm->spatial_merge_size : (mm->is_idefics3 ? mm->scale_factor : 1);
        int n_out = (merge_s > 1) ? ((grid / merge_s) * (grid / merge_s)) : (grid * grid);
        int ps_dim = (merge_s > 1) ? (merge_s * merge_s * mm->d_vis) : mm->d_vis;
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
    int merge_s = (mm->is_qwen2vl && mm->spatial_merge_size > 1) ? mm->spatial_merge_size : (mm->is_idefics3 ? mm->scale_factor : 1);
    int og = (merge_s > 1) ? (grid / merge_s) : grid;
    int n_out = og * og;
    switch (field_id) {
        case 0: return mm->image_size;
        case 1: return mm->patch_size;
        case 2: return mm->d_vis;
        case 3: return mm->n_layers;
        case 4: return mm->n_heads;
        case 5: return mm->proj_dim > 0 ? mm->proj_dim : mm->d_vis;
        case 6: return n_out;
        case 7: return merge_s;
        case 8: return mm->has_cls;
        case 9: return og;  // grid_h for automatic 2D/3D MRoPE!
        case 10: return og; // grid_w for automatic 2D/3D MRoPE!
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
    int merge_s = (mm->is_qwen2vl && mm->spatial_merge_size > 1) ? mm->spatial_merge_size : (mm->is_idefics3 ? mm->scale_factor : 1);
    int n_out = (merge_s > 1) ? ((grid / merge_s) * (grid / merge_s)) : n_patches;
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
            int use_vit_amx = (d_head == 64u && n_tok >= 32 && g_pso_vit_attn_amx != nil);
            [venc setComputePipelineState:(use_vit_amx ? g_pso_vit_attn_amx : g_pso_vit_attn)];
            [venc setBuffer:mm->buf_q offset:0 atIndex:0]; [venc setBuffer:mm->buf_k offset:0 atIndex:1];
            [venc setBuffer:mm->buf_v offset:0 atIndex:2]; [venc setBuffer:mm->buf_ctx offset:0 atIndex:3];
            [venc setBytes:&va length:sizeof(va) atIndex:4];
            [venc dispatchThreadgroups:MTLSizeMake((NSUInteger)mm->n_heads, (NSUInteger)(use_vit_amx ? ((n_tok + 31) / 32) : ((n_tok + 7) / 8)), 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

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
            int in_m = (merge_s > 1) ? (merge_s * merge_s * mm->d_vis) : mm->d_vis;
            const float* vx = ((const float*)[mm->buf_x contents]) + (mm->has_cls ? mm->d_vis : 0);
            float* psb = (float*)[mm->buf_ps contents];
            if (merge_s > 1) {
                // Qwen2.5-VL 2x2 Spatial Patch Merger before 2-layer MLP (`mm.0 -> GELU -> mm.2`)!
                int og = grid / merge_s;
                for (int oh = 0; oh < og; oh++) {
                    for (int ow = 0; ow < og; ow++) {
                        float* dst = psb + (size_t)(oh * og + ow) * (size_t)in_m;
                        for (int dh = 0; dh < merge_s; dh++) {
                            for (int dw = 0; dw < merge_s; dw++) {
                                const float* sp = vx + (size_t)((oh * merge_s + dh) * grid + (ow * merge_s + dw)) * (size_t)mm->d_vis;
                                memcpy(dst + (size_t)(dh * merge_s + dw) * (size_t)mm->d_vis, sp, (size_t)mm->d_vis * sizeof(float));
                            }
                        }
                    }
                }
            } else {
                memcpy(psb, vx, (size_t)n_out * (size_t)mm->d_vis * sizeof(float));
            }
            id<MTLCommandBuffer> pcmd = [g_moe_q commandBuffer];
            id<MTLComputeCommandEncoder> penc = [pcmd computeCommandEncoder];
            encode_gemm_batch_raw(penc, mm->mm_0_w.buf, mm->mm_0_w.offset, mm->mm_0_w.ggml_type,
                                  mm->buf_ps, mm->buf_ff, mid, (uint32_t)in_m, (uint32_t)n_out, 0);
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

// =========================================================================
// Tenon v7: AnyRes High-Res Multi-Tile Vision Encoder + 2D/3D MRoPE + Zero-Copy GGUF LoRA Hot-Swapping
// =========================================================================
int64_t tn_moe_mmproj_encode_anyres(
    int64_t mm_h, int64_t rgb_u8_ptr, int64_t width, int64_t height,
    int64_t max_tiles, int64_t out_embd_ptr, int64_t max_vecs
) {
    if (mm_h <= 0 || mm_h >= MMPROJ_MAX_MODELS || !g_mmproj[mm_h].active || !rgb_u8_ptr || !out_embd_ptr || width <= 0 || height <= 0) return 0;
    MmprojModel* mm = &g_mmproj[mm_h];
    int64_t n_out = tn_moe_mmproj_info(mm_h, 6);
    int64_t pdim  = tn_moe_mmproj_info(mm_h, 5);
    if (n_out <= 0 || pdim <= 0 || max_vecs < n_out) return 0;
    // 1. Always encode full global overview tile first
    int64_t written = tn_moe_mmproj_encode_rgb(mm_h, rgb_u8_ptr, width, height, out_embd_ptr, max_vecs);
    if (written <= 0 || max_tiles <= 1) return written;
    if (width <= mm->image_size && height <= mm->image_size) return written;
    // 2. Choose sub-tile grid (2x2 for large images, 1x2 / 2x1 for wide/tall)
    int gx = 2, gy = 2;
    if (width >= height * 2 && max_tiles < 5) { gx = 2; gy = 1; }
    else if (height >= width * 2 && max_tiles < 5) { gx = 1; gy = 2; }
    int n_sub = gx * gy;
    if (1 + n_sub > (int)max_tiles) n_sub = (int)max_tiles - 1;
    if (written + (int64_t)n_sub * n_out > max_vecs) n_sub = (int)((max_vecs - written) / n_out);
    if (n_sub <= 0) return written;

    const uint8_t* src = (const uint8_t*)(uintptr_t)rgb_u8_ptr;
    int tw = (int)((width + gx - 1) / gx);
    int th = (int)((height + gy - 1) / gy);
    uint8_t* crop = (uint8_t*)malloc((size_t)tw * (size_t)th * 3u);
    if (!crop) return written;
    int done_sub = 0;
    for (int ty = 0; ty < gy && done_sub < n_sub; ty++) {
        for (int tx = 0; tx < gx && done_sub < n_sub; tx++) {
            int x0 = tx * (int)width / gx;
            int y0 = ty * (int)height / gy;
            int cw = ((tx + 1) * (int)width / gx) - x0;
            int ch = ((ty + 1) * (int)height / gy) - y0;
            if (cw <= 0 || ch <= 0) continue;
            for (int row = 0; row < ch; row++) {
                memcpy(crop + (size_t)row * (size_t)cw * 3u,
                       src + ((size_t)(y0 + row) * (size_t)width + (size_t)x0) * 3u,
                       (size_t)cw * 3u);
            }
            float* dst_f = ((float*)(uintptr_t)out_embd_ptr) + (size_t)written * (size_t)pdim;
            int64_t nv = tn_moe_mmproj_encode_rgb(mm_h, (int64_t)(uintptr_t)crop, (int64_t)cw, (int64_t)ch,
                                                  (int64_t)(uintptr_t)dst_f, max_vecs - written);
            if (nv > 0) written += nv;
            done_sub++;
        }
    }
    free(crop);
    return written;
}

int64_t tn_moe_set_mrope_pos(int64_t handle, int64_t pos_t, int64_t pos_h, int64_t pos_w) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return 0;
    MoeModel* m = &g_moe[handle];
    if (pos_t < 0) {
        m->has_mrope = (m->mrope_sections[0] > 0 || m->mrope_sections[1] > 0) ? 1 : 0;
        m->mrope_pos[0] = m->mrope_pos[1] = m->mrope_pos[2] = -1;
        return 1;
    }
    m->has_mrope = 2;
    m->mrope_pos[0] = (int)pos_t;
    m->mrope_pos[1] = (int)pos_h;
    m->mrope_pos[2] = (int)pos_w;
    return 1;
}

static inline float moe_f16_to_f32_u16(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31;
    uint32_t e = (uint32_t)((h >> 10) & 0x1Fu);
    uint32_t m = (uint32_t)(h & 0x3FFu);
    uint32_t bits = 0;
    if (e == 0u) {
        if (m == 0u) bits = s;
        else {
            float v = ldexpf((float)m, -24);
            return s ? -v : v;
        }
    } else if (e == 31u) {
        bits = s | 0x7F800000u | (m << 13);
    } else {
        bits = s | ((e + 112u) << 23) | (m << 13);
    }
    float out; memcpy(&out, &bits, 4); return out;
}

static void moe_dequant_f32_cpu(const uint8_t* src, uint32_t gtype, size_t n_elems, float* dst) {
    if (gtype == TN_GGML_F32) {
        memcpy(dst, src, n_elems * sizeof(float));
    } else if (gtype == TN_GGML_F16) {
        const uint16_t* p = (const uint16_t*)src;
        for (size_t i = 0; i < n_elems; i++) dst[i] = moe_f16_to_f32_u16(p[i]);
    } else if (gtype == TN_GGML_Q8_0) {
        size_t nb = n_elems / 32u;
        for (size_t b = 0; b < nb; b++) {
            const uint8_t* bp = src + b * 34u;
            uint16_t dh; memcpy(&dh, bp, 2);
            float d = moe_f16_to_f32_u16(dh);
            const int8_t* qs = (const int8_t*)(bp + 2u);
            for (size_t j = 0; j < 32u; j++) dst[b * 32u + j] = d * (float)qs[j];
        }
    } else if (gtype == TN_GGML_Q4_0) {
        size_t nb = n_elems / 32u;
        for (size_t b = 0; b < nb; b++) {
            const uint8_t* bp = src + b * 18u;
            uint16_t dh; memcpy(&dh, bp, 2);
            float d = moe_f16_to_f32_u16(dh);
            for (size_t j = 0; j < 16u; j++) {
                uint8_t byte = bp[2u + j];
                dst[b * 32u + j]      = d * ((float)(byte & 0x0Fu) - 8.0f);
                dst[b * 32u + 16u + j] = d * ((float)(byte >> 4) - 8.0f);
            }
        }
    } else {
        memset(dst, 0, n_elems * sizeof(float));
    }
}

int64_t tn_moe_lora_load(int64_t handle, int64_t path_cstr, double scale) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !path_cstr) return 0;
    MoeModel* m = &g_moe[handle];
    int lslot = -1;
    for (int i = 1; i < MOE_MAX_LORAS; i++) { if (!m->loras[i].active) { lslot = i; break; } }
    if (lslot < 0) return 0;

    const char* path = (const char*)(uintptr_t)path_cstr;
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    GgufReader gr = { fp };
    uint32_t magic = gr_u32(&gr), ver = gr_u32(&gr);
    if (magic != 0x46554747u || ver < 2 || ver > 3) { fclose(fp); return 0; }
    uint64_t tcnt = gr_u64(&gr), kvcnt = gr_u64(&gr), align = 32;
    float alpha = 0.0f;
    for (uint64_t i = 0; i < kvcnt; i++) {
        char* key = gr_str(&gr); uint32_t vt = gr_u32(&gr);
        if (strcmp(key, "general.alignment") == 0) align = gr_read_int(&gr, vt);
        else if (strcmp(key, "adapter.lora.alpha") == 0 && vt == GGUF_T_F32) alpha = gr_f32(&gr);
        else if (strcmp(key, "adapter.lora.alpha") == 0) alpha = (float)gr_read_int(&gr, vt);
        else gr_skip_val(&gr, vt);
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
    int fd = open(path, O_RDONLY);
    if (fd < 0) { free(tlist); return 0; }

    @autoreleasepool {
        memset((void*)&m->loras[lslot], 0, sizeof(m->loras[lslot]));
        m->loras[lslot].alpha = alpha;
        m->loras[lslot].user_scale = (scale > 0.0) ? (float)scale : 1.0f;
        int loaded = 0;
        for (uint64_t i = 0; i < tcnt; i++) {
            const MoeTensorInfo* ti = &tlist[i];
            if (strncmp(ti->name, "blk.", 4) != 0) continue;
            int L = atoi(ti->name + 4);
            const char* dot = strchr(ti->name + 4, '.');
            if (!dot || L < 0 || L >= m->n_layers) continue;
            const char* sub = dot + 1;
            int proj = -1, is_b = 0;
            if      (strncmp(sub, "attn_q.", 7) == 0) proj = 0;
            else if (strncmp(sub, "attn_k.", 7) == 0) proj = 1;
            else if (strncmp(sub, "attn_v.", 7) == 0) proj = 2;
            else if (strncmp(sub, "attn_output.", 12) == 0) proj = 3;
            else if (strncmp(sub, "ffn_gate.", 9) == 0) proj = 4;
            else if (strncmp(sub, "ffn_up.", 7) == 0) proj = 5;
            else if (strncmp(sub, "ffn_down.", 9) == 0) proj = 6;
            else continue;
            if (strstr(sub, ".lora_a") != NULL) is_b = 0;
            else if (strstr(sub, ".lora_b") != NULL) is_b = 1;
            else continue;
            if (ti->n_dims != 2) continue;

            size_t d0 = (size_t)ti->ne[0], d1 = (size_t)ti->ne[1];
            size_t numel = d0 * d1;
            uint8_t* raw = (uint8_t*)malloc((size_t)ti->byte_size);
            float* fbuf = (float*)malloc(numel * sizeof(float));
            if (!raw || !fbuf) { free(raw); free(fbuf); continue; }
            pread_exact(fd, raw, (size_t)ti->byte_size, data_off + ti->rel_off);
            moe_dequant_f32_cpu(raw, ti->ggml_type, numel, fbuf);
            free(raw);

            id<MTLBuffer> mb = [g_moe_dev newBufferWithLength:numel * sizeof(float) options:MTLResourceStorageModeShared];
            float* dst = (float*)[mb contents];
            int rank = 0;
            if (!is_b) {
                // Target layout: A[rank][in_dim]. GGML standard: ne0=in_dim, ne1=rank (already [rank][in_dim]).
                if (d1 <= d0 && d1 <= 256u) {
                    rank = (int)d1;
                    memcpy(dst, fbuf, numel * sizeof(float));
                } else {
                    rank = (int)d0;
                    for (size_t r = 0; r < d0; r++)
                        for (size_t k = 0; k < d1; k++)
                            dst[r * d1 + k] = fbuf[k * d0 + r];
                }
                m->loras[lslot].a[L][proj] = mb;
                m->loras[lslot].r_dim[L][proj] = rank;
            } else {
                // Target layout: B[out_dim][rank]. GGML standard: ne0=rank, ne1=out_dim (already [out_dim][rank]).
                if (d0 <= d1 && d0 <= 256u) {
                    rank = (int)d0;
                    memcpy(dst, fbuf, numel * sizeof(float));
                } else {
                    rank = (int)d1;
                    for (size_t o = 0; o < d0; o++)
                        for (size_t r = 0; r < d1; r++)
                            dst[o * d1 + r] = fbuf[r * d0 + o];
                }
                m->loras[lslot].b[L][proj] = mb;
                if (m->loras[lslot].r_dim[L][proj] <= 0) m->loras[lslot].r_dim[L][proj] = rank;
            }
            free(fbuf);
            if (m->loras[lslot].rank == 0 && rank > 0) m->loras[lslot].rank = rank;
            m->loras[lslot].vram_bytes += (int64_t)(numel * sizeof(float));
            loaded++;
        }
        close(fd);
        free(tlist);
        if (loaded == 0) return 0;
        m->loras[lslot].n_tensors = loaded;
        m->loras[lslot].active = 1;
        return (int64_t)lslot;
    }
}

int64_t tn_moe_lora_set(int64_t handle, int64_t lora_id, double scale) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return 0;
    MoeModel* m = &g_moe[handle];
    m->use_batch_lora = 0;
    if (lora_id <= 0) { m->active_lora = 0; return 1; }
    if (lora_id >= MOE_MAX_LORAS || !m->loras[lora_id].active) return 0;
    if (scale >= 0.0) m->loras[lora_id].user_scale = (float)scale;
    m->active_lora = (int)lora_id;
    return 1;
}

// Tenon v9: Per-Batch-Slot Heterogeneous Multi-Tenant LoRA Assignment (`S-LoRA / Punica BGMV`)!
int64_t tn_moe_lora_set_batch(int64_t handle, int64_t lora_ids_ptr, int64_t n_active) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return 0;
    MoeModel* m = &g_moe[handle];
    if (!lora_ids_ptr || n_active <= 0) { m->use_batch_lora = 0; return 1; }
    if (n_active > MOE_PREFILL_MAX_B) n_active = MOE_PREFILL_MAX_B;
    const int64_t* ids = (const int64_t*)(uintptr_t)lora_ids_ptr;
    if (m->buf_batch_lora_ids == nil) {
        m->buf_batch_lora_ids = [g_moe_dev newBufferWithLength:(size_t)MOE_PREFILL_MAX_B * 4u options:MTLResourceStorageModeShared];
    }
    uint32_t* dst = (uint32_t*)[m->buf_batch_lora_ids contents];
    int any_nonzero = 0;
    for (int64_t i = 0; i < n_active; i++) {
        int lid = (int)ids[i];
        if (lid < 0 || lid >= MOE_MAX_LORAS || (lid > 0 && !m->loras[lid].active)) lid = 0;
        m->batch_lora_ids[i] = lid;
        dst[i] = (uint32_t)lid;
        if (lid > 0) any_nonzero = 1;
    }
    m->use_batch_lora = any_nonzero;
    return n_active;
}

int64_t tn_moe_lora_free(int64_t handle, int64_t lora_id) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return 0;
    MoeModel* m = &g_moe[handle];
    if (lora_id <= 0 || lora_id >= MOE_MAX_LORAS || !m->loras[lora_id].active) return 0;
    if (m->active_lora == (int)lora_id) m->active_lora = 0;
    @autoreleasepool {
        for (int L = 0; L < MOE_MAX_LAYERS; L++) {
            for (int p = 0; p < 7; p++) {
                m->loras[lora_id].a[L][p] = nil;
                m->loras[lora_id].b[L][p] = nil;
                m->loras[lora_id].r_dim[L][p] = 0;
            }
        }
        memset((void*)&m->loras[lora_id], 0, sizeof(m->loras[lora_id]));
    }
    return 1;
}

int64_t tn_moe_lora_info(int64_t handle, int64_t lora_id, int64_t field_id) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active) return 0;
    MoeModel* m = &g_moe[handle];
    if (field_id == 4) return m->active_lora;
    if (lora_id <= 0 || lora_id >= MOE_MAX_LORAS || !m->loras[lora_id].active) return 0;
    switch (field_id) {
        case 0: return m->loras[lora_id].active;
        case 1: return m->loras[lora_id].rank;
        case 2: return m->loras[lora_id].n_tensors;
        case 3: return m->loras[lora_id].vram_bytes;
        default: return 0;
    }
}

// =========================================================================
// Tenon v8: Auto 2D/3D MRoPE Vision/Video Grid Injection (`tn_moe_step_embd_mrope`)
// + Native Whisper 80-Bin Log-Mel Spectrogram & Audio GGUF Encoder (`tn_moe_audio_*`)!
// =========================================================================
int64_t tn_moe_step_embd_mrope(int64_t handle, int64_t embd_f32_ptr, int64_t n_vecs, int64_t start_pos,
                               int64_t grid_t, int64_t grid_h, int64_t grid_w) {
    if (handle <= 0 || handle >= MOE_MAX_MODELS || !g_moe[handle].active || !embd_f32_ptr || n_vecs <= 0) return 0;
    MoeModel* m = &g_moe[handle];
    int64_t gt = grid_t > 0 ? grid_t : 1;
    int64_t gh = grid_h > 0 ? grid_h : 1;
    int64_t gw = grid_w > 0 ? grid_w : n_vecs;
    int64_t hw = gh * gw;
    int64_t last_tok = 0;
    for (int64_t i = 0; i < n_vecs; i++) {
        int64_t ti = (hw > 0) ? ((i / hw) % gt) : 0;
        int64_t rem = (hw > 0) ? (i % hw) : i;
        int64_t hi = (gw > 0) ? (rem / gw) : 0;
        int64_t wi = (gw > 0) ? (rem % gw) : 0;
        tn_moe_set_mrope_pos(handle, start_pos + ti, start_pos + hi, start_pos + wi);
        last_tok = tn_moe_step_embd(handle, embd_f32_ptr + i * (int64_t)m->d_model * 4, 1, start_pos + i);
    }
    tn_moe_set_mrope_pos(handle, -1, -1, -1); // Restore 1D RoPE cleanly!
    return last_tok;
}

#define TN_AUDIO_MAX_MODELS 4
#define TN_AUDIO_MAX_LAYERS 32
#define TN_AUDIO_N_MELS     80
#define TN_AUDIO_N_FFT      400
#define TN_AUDIO_HOP        160
#define TN_AUDIO_SR         16000
#define TN_AUDIO_N_BINS     (TN_AUDIO_N_FFT / 2 + 1)

typedef struct {
    float *ln1_w, *ln1_b, *ln2_w, *ln2_b;
    float *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
    float *fc1_w, *fc1_b, *fc2_w, *fc2_b;
} TnAudioLayer;

typedef struct {
    int active;
    int n_mels;
    int sample_rate;
    int n_fft;
    int hop_length;
    int d_audio;
    int d_ff;
    int n_heads;
    int proj_dim;
    int proj_mid;
    int n_layers;
    int max_pos;
    int has_gguf; // 0=mel fallback, 1=gguf header, 2=full Whisper/Qwen2-Audio Transformer weights loaded!
    float hann[TN_AUDIO_N_FFT];
    float mel_fb[TN_AUDIO_N_MELS][TN_AUDIO_N_BINS];
    float* proj_w; // fallback [proj_dim * n_mels]
    // Real Whisper / Qwen2-Audio GGUF weights
    float *conv1_w, *conv1_b; // [d_audio, n_mels, 3]
    float *conv2_w, *conv2_b; // [d_audio, d_audio, 3]
    float *pos_embd;          // [max_pos, d_audio]
    float *post_ln_w, *post_ln_b;
    float *mm0_w, *mm0_b;     // [proj_mid, d_audio]
    float *mm2_w, *mm2_b;     // [proj_dim, proj_mid]
    TnAudioLayer layers[TN_AUDIO_MAX_LAYERS];
} TnAudioEncoder;

static TnAudioEncoder g_audio[TN_AUDIO_MAX_MODELS];

static float* tn_audio_read_f32_tensor(int fd, const MoeTensorInfo* ti, size_t* out_numel) {
    size_t numel = 1;
    for (uint32_t d = 0; d < ti->n_dims && d < 4; d++) numel *= (size_t)ti->ne[d];
    if (out_numel) *out_numel = numel;
    uint8_t* raw = (uint8_t*)malloc((size_t)ti->byte_size);
    float* dst = (float*)malloc(numel * sizeof(float));
    if (!raw || !dst) { free(raw); free(dst); return NULL; }
    pread_exact(fd, raw, (size_t)ti->byte_size, ti->file_off);
    moe_dequant_f32_cpu(raw, ti->ggml_type, numel, dst);
    free(raw);
    return dst;
}

static void tn_audio_layernorm(const float* x, const float* w, const float* b, float* out, int T, int D) {
    for (int t = 0; t < T; t++) {
        const float* xp = x + (size_t)t * D;
        float* op = out + (size_t)t * D;
        float mean = 0.0f;
        for (int d = 0; d < D; d++) mean += xp[d];
        mean /= (float)D;
        float var = 0.0f;
        for (int d = 0; d < D; d++) { float v = xp[d] - mean; var += v * v; }
        float inv = 1.0f / sqrtf(var / (float)D + 1e-5f);
        for (int d = 0; d < D; d++) {
            float y = (xp[d] - mean) * inv * (w ? w[d] : 1.0f);
            op[d] = b ? (y + b[d]) : y;
        }
    }
}

static void tn_audio_linear(const float* x, const float* w, const float* b, float* y, int T, int in_d, int out_d, int add_res) {
    for (int t = 0; t < T; t++) {
        const float* xp = x + (size_t)t * in_d;
        float* yp = y + (size_t)t * out_d;
        for (int o = 0; o < out_d; o++) {
            const float* wp = w + (size_t)o * in_d;
            float acc = b ? b[o] : 0.0f;
            for (int i = 0; i < in_d; i++) acc += wp[i] * xp[i];
            yp[o] = add_res ? (yp[o] + acc) : acc;
        }
    }
}

static inline float tn_gelu_f32(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static float tn_hz_to_mel_slaney(float hz) {
    float f_sp = 200.0f / 3.0f;
    if (hz < 1000.0f) return hz / f_sp;
    return 15.0f + logf(hz / 1000.0f) / (logf(6.4f) / 27.0f);
}

static float tn_mel_to_hz_slaney(float mel) {
    float f_sp = 200.0f / 3.0f;
    if (mel < 15.0f) return mel * f_sp;
    return 1000.0f * expf((mel - 15.0f) * (logf(6.4f) / 27.0f));
}

static void tn_audio_init_mel_fb(TnAudioEncoder* ae) {
    for (int i = 0; i < TN_AUDIO_N_FFT; i++) {
        ae->hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)TN_AUDIO_N_FFT));
    }
    float mel_min = tn_hz_to_mel_slaney(0.0f);
    float mel_max = tn_hz_to_mel_slaney((float)TN_AUDIO_SR * 0.5f);
    float pts[TN_AUDIO_N_MELS + 2];
    for (int m = 0; m < TN_AUDIO_N_MELS + 2; m++) {
        float mel = mel_min + (mel_max - mel_min) * (float)m / (float)(TN_AUDIO_N_MELS + 1);
        pts[m] = tn_mel_to_hz_slaney(mel);
    }
    memset(ae->mel_fb, 0, sizeof(ae->mel_fb));
    for (int m = 0; m < TN_AUDIO_N_MELS; m++) {
        float f_l = pts[m], f_c = pts[m + 1], f_r = pts[m + 2];
        float enorm = 2.0f / fmaxf(f_r - f_l, 1e-6f);
        for (int b = 0; b < TN_AUDIO_N_BINS; b++) {
            float f = (float)b * (float)TN_AUDIO_SR / (float)TN_AUDIO_N_FFT;
            float w = 0.0f;
            if (f >= f_l && f <= f_c) w = (f - f_l) / fmaxf(f_c - f_l, 1e-6f);
            else if (f > f_c && f <= f_r) w = (f_r - f) / fmaxf(f_r - f_c, 1e-6f);
            ae->mel_fb[m][b] = w * enorm;
        }
    }
}

int64_t tn_moe_audio_load(int64_t path_cstr) {
    if (!path_cstr) return 0;
    const char* path = (const char*)(uintptr_t)path_cstr;
    int slot = -1;
    for (int i = 1; i < TN_AUDIO_MAX_MODELS; i++) if (!g_audio[i].active) { slot = i; break; }
    if (slot < 0) return 0;
    TnAudioEncoder* ae = &g_audio[slot];
    memset(ae, 0, sizeof(*ae));
    ae->n_mels = TN_AUDIO_N_MELS;
    ae->sample_rate = TN_AUDIO_SR;
    ae->n_fft = TN_AUDIO_N_FFT;
    ae->hop_length = TN_AUDIO_HOP;
    ae->d_audio = TN_AUDIO_N_MELS;
    ae->proj_dim = 2048;
    ae->n_layers = 2;

    if (strncmp(path, "mel:", 4) == 0) {
        int pd = atoi(path + 4);
        if (pd >= 64 && pd <= 16384) ae->proj_dim = pd;
    } else {
        int fd = open(path, O_RDONLY);
        if (fd < 0) return 0;
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 32) { close(fd); return 0; }
        uint8_t hdr[4];
        if (pread(fd, hdr, 4, 0) == 4 && memcmp(hdr, "GGUF", 4) == 0) {
            ae->has_gguf = 1;
            MoeModel probe; memset((void*)&probe, 0, sizeof(probe));
            MoeTensorInfo* tlist = NULL; uint64_t tcount = 0;
            if (parse_gguf_shard(path, 0, 1, &probe, &tlist, &tcount, NULL)) {
                if (probe.d_model >= 64 && probe.d_model <= 16384) ae->proj_dim = probe.d_model;
                if (probe.n_layers > 0 && probe.n_layers <= TN_AUDIO_MAX_LAYERS) ae->n_layers = probe.n_layers;
                if (probe.n_heads > 0) ae->n_heads = probe.n_heads;
                int max_L = -1;
                for (uint64_t ti = 0; ti < tcount; ti++) {
                    const MoeTensorInfo* t = &tlist[ti];
                    size_t numel = 0;
                    if (strcmp(t->name, "a.conv1.weight") == 0) {
                        ae->conv1_w = tn_audio_read_f32_tensor(fd, t, &numel);
                        if (t->n_dims >= 3 && t->ne[2] > 0) ae->d_audio = (int)t->ne[2];
                        else if (t->n_dims == 2 && t->ne[1] > 0) ae->d_audio = (int)t->ne[1];
                    } else if (strcmp(t->name, "a.conv1.bias") == 0) ae->conv1_b = tn_audio_read_f32_tensor(fd, t, &numel);
                    else if (strcmp(t->name, "a.conv2.weight") == 0) ae->conv2_w = tn_audio_read_f32_tensor(fd, t, &numel);
                    else if (strcmp(t->name, "a.conv2.bias") == 0) ae->conv2_b = tn_audio_read_f32_tensor(fd, t, &numel);
                    else if (strcmp(t->name, "a.position_embd.weight") == 0) {
                        ae->pos_embd = tn_audio_read_f32_tensor(fd, t, &numel);
                        if (t->n_dims >= 2) ae->max_pos = (int)t->ne[1];
                    } else if (strcmp(t->name, "a.post_ln.weight") == 0) ae->post_ln_w = tn_audio_read_f32_tensor(fd, t, &numel);
                    else if (strcmp(t->name, "a.post_ln.bias") == 0) ae->post_ln_b = tn_audio_read_f32_tensor(fd, t, &numel);
                    else if (strcmp(t->name, "mm.a.0.weight") == 0 || strcmp(t->name, "audio.proj.weight") == 0) {
                        ae->mm0_w = tn_audio_read_f32_tensor(fd, t, &numel);
                        if (t->n_dims >= 2) { ae->proj_mid = (int)t->ne[1]; if (!ae->mm2_w) ae->proj_dim = (int)t->ne[1]; }
                    } else if (strcmp(t->name, "mm.a.0.bias") == 0 || strcmp(t->name, "audio.proj.bias") == 0) {
                        ae->mm0_b = tn_audio_read_f32_tensor(fd, t, &numel);
                    } else if (strcmp(t->name, "mm.a.2.weight") == 0) {
                        ae->mm2_w = tn_audio_read_f32_tensor(fd, t, &numel);
                        if (t->n_dims >= 2) ae->proj_dim = (int)t->ne[1];
                    } else if (strcmp(t->name, "mm.a.2.bias") == 0) {
                        ae->mm2_b = tn_audio_read_f32_tensor(fd, t, &numel);
                    } else if (strncmp(t->name, "a.blk.", 6) == 0) {
                        int L = atoi(t->name + 6);
                        const char* dot = strchr(t->name + 6, '.');
                        if (!dot || L < 0 || L >= TN_AUDIO_MAX_LAYERS) continue;
                        if (L > max_L) max_L = L;
                        const char* sub = dot + 1;
                        TnAudioLayer* ly = &ae->layers[L];
                        if      (strcmp(sub, "ln1.weight") == 0 || strcmp(sub, "attn_norm.weight") == 0) ly->ln1_w = tn_audio_read_f32_tensor(fd, t, &numel);
                        else if (strcmp(sub, "ln1.bias") == 0   || strcmp(sub, "attn_norm.bias") == 0)   ly->ln1_b = tn_audio_read_f32_tensor(fd, t, &numel);
                        else if (strcmp(sub, "ln2.weight") == 0 || strcmp(sub, "ffn_norm.weight") == 0)  ly->ln2_w = tn_audio_read_f32_tensor(fd, t, &numel);
                        else if (strcmp(sub, "ln2.bias") == 0   || strcmp(sub, "ffn_norm.bias") == 0)    ly->ln2_b = tn_audio_read_f32_tensor(fd, t, &numel);
                        else if (strcmp(sub, "attn_q.weight") == 0) ly->q_w = tn_audio_read_f32_tensor(fd, t, &numel);
                        else if (strcmp(sub, "attn_q.bias") == 0)   ly->q_b = tn_audio_read_f32_tensor(fd, t, &numel);
                        else if (strcmp(sub, "attn_k.weight") == 0) ly->k_w = tn_audio_read_f32_tensor(fd, t, &numel);
                        else if (strcmp(sub, "attn_k.bias") == 0)   ly->k_b = tn_audio_read_f32_tensor(fd, t, &numel);
                        else if (strcmp(sub, "attn_v.weight") == 0) ly->v_w = tn_audio_read_f32_tensor(fd, t, &numel);
                        else if (strcmp(sub, "attn_v.bias") == 0)   ly->v_b = tn_audio_read_f32_tensor(fd, t, &numel);
                        else if (strcmp(sub, "attn_out.weight") == 0) ly->o_w = tn_audio_read_f32_tensor(fd, t, &numel);
                        else if (strcmp(sub, "attn_out.bias") == 0)   ly->o_b = tn_audio_read_f32_tensor(fd, t, &numel);
                        else if (strcmp(sub, "ffn_up.weight") == 0) { ly->fc1_w = tn_audio_read_f32_tensor(fd, t, &numel); if (t->n_dims >= 2) ae->d_ff = (int)t->ne[1]; }
                        else if (strcmp(sub, "ffn_up.bias") == 0)   ly->fc1_b = tn_audio_read_f32_tensor(fd, t, &numel);
                        else if (strcmp(sub, "ffn_down.weight") == 0) ly->fc2_w = tn_audio_read_f32_tensor(fd, t, &numel);
                        else if (strcmp(sub, "ffn_down.bias") == 0)   ly->fc2_b = tn_audio_read_f32_tensor(fd, t, &numel);
                    }
                }
                if (max_L >= 0) ae->n_layers = max_L + 1;
                if (ae->d_ff <= 0) ae->d_ff = ae->d_audio * 4;
                if (ae->n_heads <= 0) ae->n_heads = (ae->d_audio >= 64) ? (ae->d_audio / 64) : 1;
                if (ae->conv1_w && ae->conv2_w) ae->has_gguf = 2; // Full Whisper/Qwen2-Audio Transformer Encoder!
                if (probe.vocab_tokens) { for (int vi = 0; vi < probe.vocab_size; vi++) free(probe.vocab_tokens[vi]); free(probe.vocab_tokens); }
                if (probe.bpe_merges) { for (int bi = 0; bi < probe.n_bpe_merges; bi++) free(probe.bpe_merges[bi]); free(probe.bpe_merges); }
            }
            free(tlist);
        }
        close(fd);
    }
    tn_audio_init_mel_fb(ae);
    ae->proj_w = (float*)malloc((size_t)ae->proj_dim * (size_t)ae->n_mels * sizeof(float));
    if (!ae->proj_w) return 0;
    float inv_s = 1.0f / sqrtf((float)ae->n_mels);
    for (int d = 0; d < ae->proj_dim; d++) {
        for (int m = 0; m < ae->n_mels; m++) {
            uint64_t h = ((uint64_t)(d + 1) * 6364136223846793005ull) ^ ((uint64_t)(m + 1) * 1442695040888963407ull);
            h ^= h >> 33; h *= 0xff51afd7ed558ccdull; h ^= h >> 33;
            float u = ((float)(h & 0xFFFFFFu) / 16777216.0f) * 2.0f - 1.0f;
            ae->proj_w[(size_t)d * (size_t)ae->n_mels + (size_t)m] = u * inv_s * 0.1f;
        }
    }
    ae->active = 1;
    return (int64_t)slot;
}

int64_t tn_moe_audio_info(int64_t audio_h, int64_t field_id) {
    if (audio_h <= 0 || audio_h >= TN_AUDIO_MAX_MODELS || !g_audio[audio_h].active) return 0;
    TnAudioEncoder* ae = &g_audio[audio_h];
    switch (field_id) {
        case 0: return ae->n_mels;
        case 1: return ae->sample_rate;
        case 2: return ae->n_fft;
        case 3: return ae->hop_length;
        case 4: return ae->d_audio;
        case 5: return ae->proj_dim;
        case 6: return ae->n_layers;
        case 7: return ae->has_gguf;
        default: return 0;
    }
}

int64_t tn_moe_audio_encode_pcm16(int64_t audio_h, int64_t pcm16_ptr, int64_t n_samples, int64_t sample_rate,
                                  int64_t out_embd_ptr, int64_t max_vecs) {
    if (audio_h <= 0 || audio_h >= TN_AUDIO_MAX_MODELS || !g_audio[audio_h].active) return 0;
    if (!pcm16_ptr || !out_embd_ptr || n_samples <= 0 || max_vecs <= 0) return 0;
    TnAudioEncoder* ae = &g_audio[audio_h];
    const int16_t* pcm = (const int16_t*)(uintptr_t)pcm16_ptr;
    int sr = (sample_rate > 0) ? (int)sample_rate : TN_AUDIO_SR;
    int64_t n16 = (sr == TN_AUDIO_SR) ? n_samples : (int64_t)((double)n_samples * (double)TN_AUDIO_SR / (double)sr);
    if (n16 < TN_AUDIO_N_FFT) n16 = TN_AUDIO_N_FFT;
    if (n16 > TN_AUDIO_SR * 30) n16 = TN_AUDIO_SR * 30; // Cap at Whisper 30s chunk
    float* wav = (float*)calloc((size_t)n16, sizeof(float));
    if (!wav) return 0;
    if (sr == TN_AUDIO_SR) {
        int64_t copy_n = n_samples < n16 ? n_samples : n16;
        for (int64_t i = 0; i < copy_n; i++) wav[i] = (float)pcm[i] / 32768.0f;
    } else {
        double ratio = (double)sr / (double)TN_AUDIO_SR;
        for (int64_t i = 0; i < n16; i++) {
            double src = (double)i * ratio;
            int64_t i0 = (int64_t)src;
            if (i0 >= n_samples) break;
            int64_t i1 = (i0 + 1 < n_samples) ? (i0 + 1) : i0;
            float frac = (float)(src - (double)i0);
            wav[i] = ((1.0f - frac) * (float)pcm[i0] + frac * (float)pcm[i1]) / 32768.0f;
        }
    }
    int n_frames = (int)(1 + (n16 - TN_AUDIO_N_FFT) / TN_AUDIO_HOP);
    if (n_frames < 1) n_frames = 1;
    float* mel = (float*)malloc((size_t)n_frames * TN_AUDIO_N_MELS * sizeof(float));
    if (!mel) { free(wav); return 0; }

    // Precompute DFT twiddles for N_FFT=400 (201 bins)
    float* cos_t = (float*)malloc((size_t)TN_AUDIO_N_BINS * TN_AUDIO_N_FFT * sizeof(float));
    float* sin_t = (float*)malloc((size_t)TN_AUDIO_N_BINS * TN_AUDIO_N_FFT * sizeof(float));
    if (!cos_t || !sin_t) { free(cos_t); free(sin_t); free(mel); free(wav); return 0; }
    for (int k = 0; k < TN_AUDIO_N_BINS; k++) {
        float w0 = -2.0f * (float)M_PI * (float)k / (float)TN_AUDIO_N_FFT;
        for (int n = 0; n < TN_AUDIO_N_FFT; n++) {
            float a = w0 * (float)n;
            cos_t[k * TN_AUDIO_N_FFT + n] = cosf(a);
            sin_t[k * TN_AUDIO_N_FFT + n] = sinf(a);
        }
    }
    float max_log = -1e30f;
    float win[TN_AUDIO_N_FFT];
    float pwr[TN_AUDIO_N_BINS];
    for (int f = 0; f < n_frames; f++) {
        int off = f * TN_AUDIO_HOP;
        for (int n = 0; n < TN_AUDIO_N_FFT; n++) win[n] = wav[off + n] * ae->hann[n];
        for (int k = 0; k < TN_AUDIO_N_BINS; k++) {
            const float* ct = cos_t + k * TN_AUDIO_N_FFT;
            const float* st = sin_t + k * TN_AUDIO_N_FFT;
            float re = 0.0f, im = 0.0f;
            for (int n = 0; n < TN_AUDIO_N_FFT; n++) { re += win[n] * ct[n]; im += win[n] * st[n]; }
            pwr[k] = re * re + im * im;
        }
        float* mf = mel + (size_t)f * TN_AUDIO_N_MELS;
        for (int m = 0; m < TN_AUDIO_N_MELS; m++) {
            float s = 0.0f;
            for (int k = 0; k < TN_AUDIO_N_BINS; k++) s += ae->mel_fb[m][k] * pwr[k];
            float lg = log10f(fmaxf(s, 1e-10f));
            mf[m] = lg;
            if (lg > max_log) max_log = lg;
        }
    }
    free(cos_t); free(sin_t); free(wav);
    // Whisper dynamic-range clamp [max-8, max] -> (x + 4) / 4
    for (int i = 0; i < n_frames * TN_AUDIO_N_MELS; i++) {
        float v = mel[i];
        if (v < max_log - 8.0f) v = max_log - 8.0f;
        mel[i] = (v + 4.0f) * 0.25f;
    }
    int n_out = (n_frames + 1) / 2;
    if (n_out > (int)max_vecs) n_out = (int)max_vecs;
    float* out = (float*)(uintptr_t)out_embd_ptr;

    if (ae->has_gguf == 2 && ae->conv1_w && ae->conv2_w) {
        // Tenon v9: FULL Real Whisper / Qwen2-Audio GGUF Transformer Encoder!
        // Conv1d(80->D, k=3, s=1) + GELU -> Conv1d(D->D, k=3, s=2) + GELU -> PosEmbd -> N-Layer MHA + FFN -> PostLN -> Projector!
        int D = ae->d_audio, T1 = n_frames, T2 = n_out, H = ae->n_heads > 0 ? ae->n_heads : 1, Dh = D / H;
        float* c1 = (float*)calloc((size_t)T1 * D, sizeof(float));
        float* x  = (float*)calloc((size_t)T2 * D, sizeof(float));
        float* xn = (float*)malloc((size_t)T2 * D * sizeof(float));
        float* q  = (float*)malloc((size_t)T2 * D * sizeof(float));
        float* k  = (float*)malloc((size_t)T2 * D * sizeof(float));
        float* v  = (float*)malloc((size_t)T2 * D * sizeof(float));
        float* ct = (float*)calloc((size_t)T2 * D, sizeof(float));
        float* ff = (float*)malloc((size_t)T2 * (size_t)ae->d_ff * sizeof(float));
        // Conv1: pad=1, k=3, stride=1, weight layout [D, 80, 3]
        for (int t = 0; t < T1; t++) {
            for (int oc = 0; oc < D; oc++) {
                float acc = ae->conv1_b ? ae->conv1_b[oc] : 0.0f;
                const float* wo = ae->conv1_w + (size_t)oc * TN_AUDIO_N_MELS * 3u;
                for (int ic = 0; ic < TN_AUDIO_N_MELS; ic++) {
                    const float* wk = wo + (size_t)ic * 3u;
                    for (int ki = 0; ki < 3; ki++) {
                        int src_t = t + ki - 1;
                        if (src_t >= 0 && src_t < T1) acc += wk[ki] * mel[(size_t)src_t * TN_AUDIO_N_MELS + ic];
                    }
                }
                c1[(size_t)t * D + oc] = tn_gelu_f32(acc);
            }
        }
        // Conv2: pad=1, k=3, stride=2, weight layout [D, D, 3]
        for (int t = 0; t < T2; t++) {
            int t_in = t * 2;
            for (int oc = 0; oc < D; oc++) {
                float acc = ae->conv2_b ? ae->conv2_b[oc] : 0.0f;
                const float* wo = ae->conv2_w + (size_t)oc * (size_t)D * 3u;
                for (int ic = 0; ic < D; ic++) {
                    const float* wk = wo + (size_t)ic * 3u;
                    for (int ki = 0; ki < 3; ki++) {
                        int src_t = t_in + ki - 1;
                        if (src_t >= 0 && src_t < T1) acc += wk[ki] * c1[(size_t)src_t * D + ic];
                    }
                }
                float val = tn_gelu_f32(acc);
                if (ae->pos_embd && t < ae->max_pos) val += ae->pos_embd[(size_t)t * D + oc];
                x[(size_t)t * D + oc] = val;
            }
        }
        float inv_sq = 1.0f / sqrtf((float)Dh);
        for (int L = 0; L < ae->n_layers; L++) {
            TnAudioLayer* ly = &ae->layers[L];
            if (!ly->q_w || !ly->k_w || !ly->v_w || !ly->o_w) continue;
            tn_audio_layernorm(x, ly->ln1_w, ly->ln1_b, xn, T2, D);
            tn_audio_linear(xn, ly->q_w, ly->q_b, q, T2, D, D, 0);
            tn_audio_linear(xn, ly->k_w, ly->k_b, k, T2, D, D, 0);
            tn_audio_linear(xn, ly->v_w, ly->v_b, v, T2, D, D, 0);
            for (int h = 0; h < H; h++) {
                for (int qi = 0; qi < T2; qi++) {
                    const float* qp = q + (size_t)qi * D + h * Dh;
                    float mx = -1e30f, wts[1536];
                    for (int ki = 0; ki < T2 && ki < 1536; ki++) {
                        const float* kp = k + (size_t)ki * D + h * Dh;
                        float dot = 0.0f;
                        for (int d = 0; d < Dh; d++) dot += qp[d] * kp[d];
                        wts[ki] = dot * inv_sq;
                        if (wts[ki] > mx) mx = wts[ki];
                    }
                    float sm = 0.0f;
                    for (int ki = 0; ki < T2 && ki < 1536; ki++) { wts[ki] = expf(wts[ki] - mx); sm += wts[ki]; }
                    float inv_s = 1.0f / fmaxf(sm, 1e-12f);
                    float* cp = ct + (size_t)qi * D + h * Dh;
                    for (int d = 0; d < Dh; d++) {
                        float acc = 0.0f;
                        for (int ki = 0; ki < T2 && ki < 1536; ki++) acc += (wts[ki] * inv_s) * v[(size_t)ki * D + h * Dh + d];
                        cp[d] = acc;
                    }
                }
            }
            tn_audio_linear(ct, ly->o_w, ly->o_b, x, T2, D, D, 1);
            if (ly->fc1_w && ly->fc2_w) {
                tn_audio_layernorm(x, ly->ln2_w, ly->ln2_b, xn, T2, D);
                tn_audio_linear(xn, ly->fc1_w, ly->fc1_b, ff, T2, D, ae->d_ff, 0);
                for (int i = 0; i < T2 * ae->d_ff; i++) ff[i] = tn_gelu_f32(ff[i]);
                tn_audio_linear(ff, ly->fc2_w, ly->fc2_b, x, T2, ae->d_ff, D, 1);
            }
        }
        if (ae->post_ln_w) tn_audio_layernorm(x, ae->post_ln_w, ae->post_ln_b, x, T2, D);
        if (ae->mm0_w && ae->mm2_w) {
            float* mid = (float*)malloc((size_t)T2 * (size_t)ae->proj_mid * sizeof(float));
            tn_audio_linear(x, ae->mm0_w, ae->mm0_b, mid, T2, D, ae->proj_mid, 0);
            for (int i = 0; i < T2 * ae->proj_mid; i++) mid[i] = tn_gelu_f32(mid[i]);
            tn_audio_linear(mid, ae->mm2_w, ae->mm2_b, out, T2, ae->proj_mid, ae->proj_dim, 0);
            free(mid);
        } else if (ae->mm0_w) {
            tn_audio_linear(x, ae->mm0_w, ae->mm0_b, out, T2, D, ae->proj_dim, 0);
        } else {
            for (int t = 0; t < T2; t++) {
                float* dst = out + (size_t)t * ae->proj_dim;
                int cpy = D < ae->proj_dim ? D : ae->proj_dim;
                memcpy(dst, x + (size_t)t * D, (size_t)cpy * sizeof(float));
                if (ae->proj_dim > cpy) memset(dst + cpy, 0, (size_t)(ae->proj_dim - cpy) * sizeof(float));
            }
        }
        free(c1); free(x); free(xn); free(q); free(k); free(v); free(ct); free(ff);
        free(mel);
        return (int64_t)n_out;
    }

    // Fallback: Stride-2 temporal pooling (100 Hz -> 50 Hz) + GELU + Linear Projection -> [n_out, proj_dim]
    for (int t = 0; t < n_out; t++) {
        int f0 = t * 2, f1 = (f0 + 1 < n_frames) ? (f0 + 1) : f0;
        float feat[TN_AUDIO_N_MELS];
        for (int m = 0; m < TN_AUDIO_N_MELS; m++) {
            float x = 0.5f * (mel[(size_t)f0 * TN_AUDIO_N_MELS + m] + mel[(size_t)f1 * TN_AUDIO_N_MELS + m]);
            feat[m] = tn_gelu_f32(x);
        }
        float* dst = out + (size_t)t * (size_t)ae->proj_dim;
        for (int d = 0; d < ae->proj_dim; d++) {
            const float* wp = ae->proj_w + (size_t)d * TN_AUDIO_N_MELS;
            float acc = 0.0f;
            for (int m = 0; m < TN_AUDIO_N_MELS; m++) acc += wp[m] * feat[m];
            dst[d] = acc;
        }
    }
    free(mel);
    return (int64_t)n_out;
}

int64_t tn_moe_audio_free(int64_t audio_h) {
    if (audio_h <= 0 || audio_h >= TN_AUDIO_MAX_MODELS || !g_audio[audio_h].active) return 0;
    TnAudioEncoder* ae = &g_audio[audio_h];
    free(ae->proj_w); free(ae->conv1_w); free(ae->conv1_b); free(ae->conv2_w); free(ae->conv2_b);
    free(ae->pos_embd); free(ae->post_ln_w); free(ae->post_ln_b);
    free(ae->mm0_w); free(ae->mm0_b); free(ae->mm2_w); free(ae->mm2_b);
    for (int L = 0; L < TN_AUDIO_MAX_LAYERS; L++) {
        TnAudioLayer* ly = &ae->layers[L];
        free(ly->ln1_w); free(ly->ln1_b); free(ly->ln2_w); free(ly->ln2_b);
        free(ly->q_w); free(ly->q_b); free(ly->k_w); free(ly->k_b);
        free(ly->v_w); free(ly->v_b); free(ly->o_w); free(ly->o_b);
        free(ly->fc1_w); free(ly->fc1_b); free(ly->fc2_w); free(ly->fc2_b);
    }
    memset(ae, 0, sizeof(*ae));
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
            m->mtp_layer->mla_wuk_f16 = nil; m->mtp_layer->mla_wuv_f16 = nil;
            free(m->mtp_layer); m->mtp_layer = NULL;
        }
        for (int lid = 1; lid < MOE_MAX_LORAS; lid++) {
            if (m->loras[lid].active) {
                for (int L = 0; L < MOE_MAX_LAYERS; L++) {
                    for (int p = 0; p < 7; p++) {
                        m->loras[lid].a[L][p] = nil;
                        m->loras[lid].b[L][p] = nil;
                    }
                }
                memset((void*)&m->loras[lid], 0, sizeof(m->loras[lid]));
            }
        }
        m->active_lora = 0;
        m->use_batch_lora = 0;
        m->buf_lora_tmp = nil; m->buf_batch_lora_ids = nil; m->buf_lora_dummy = nil;
        icb_cache_clear(m);
        if (@available(macOS 15.0, *)) {
            if (m->residency_set != nil) {
                id<MTLResidencySet> rset = (id<MTLResidencySet>)m->residency_set;
                if (g_moe_q != nil) [g_moe_q removeResidencySet:rset];
                [rset endResidency];
                m->residency_set = nil;
            }
        }
        m->icb_event = nil;
        m->buf_fd_o = nil; m->buf_fd_m = nil; m->buf_fd_l = nil;
        free(m->pa_refcnt); m->pa_refcnt = NULL;
        m->buf_pa_bt = nil; m->buf_cb_tok = nil; m->buf_cb_pos = nil; m->buf_cb_slot = nil;
        m->buf_batch_logits = nil; m->batch_logits_cap = 0;
        m->buf_tree_depth = nil; m->buf_tree_mask = nil;
        m->pbuf_mla_qabs = nil; m->pbuf_mla_olat = nil; m->buf_mtp_x = nil;
        m->pbuf_rlogits = nil; m->pbuf_moe_x = nil; m->pbuf_moe_h = nil; m->pbuf_moe_y = nil;
        m->buf_moe_idx = nil; m->buf_moe_wt = nil; m->buf_moe_active_e = nil;
        m->buf_moe_grp_off = nil; m->buf_moe_grp_cnt = nil; m->buf_moe_tok_slot = nil;
        m->buf_moe_meta = nil; m->buf_moe_zero = nil;
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
