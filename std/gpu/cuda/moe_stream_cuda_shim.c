// std/gpu/cuda/moe_stream_cuda_shim.c — Universal Hard-VRAM-Capped NVIDIA CUDA Multi-Shard GGUF Streaming Engine
//
// Zero-Link-Flag Dynamic CUDA Driver (`dlopen("libcuda.so.1")` + `nvcc --ptx` / `libnvrtc.so`) Engine:
//   - Supports Oversized Sparse MoE (GLM-4.5-Air 106B-A12B, Qwen3-MoE, OLMoE, Mixtral) AND Oversized Dense (Llama-3.3-70B)
//     on a SINGLE consumer GPU (e.g. RTX 4090 24GB VRAM, RTX 5090 32GB, A100 40/80GB)!
//   - Multi-Shard GGUF v2/v3 (`-00001-of-0000N.gguf` auto-discovery) + All 8 GGML quant types:
//     F32, F16, Q4_0, Q4_1, Q8_0, Q4_K, Q5_K, Q6_K
//   - 3-Tier Memory Hierarchy:
//       * Tier 1 (VRAM, 1,008 GB/s GDDR6X): Core trunk + Per-Layer VRAM LRU Expert Pool (or All-80-Attn + Prefix FFNs + 4-Slot Ring)
//       * Tier 2 (Pinned Host Staging, `cuMemHostAlloc`, 24-26 GB/s PCIe 4.0 H2D DMA): 16-slot MoE / 4-slot Dense staging ring
//       * Tier 3 (Backing Store): Host RAM `mmap` (`use_nocache=0`, zero syscall memcpy -> pinned -> DMA) OR
//         Direct NVMe `pread` + `POSIX_FADV_DONTNEED` (`use_nocache=1`, bounded host RAM!)
//   - Dual CUDA Stream Hardware Overlap (`compute_stream` || `dma_stream` via `cuStreamWaitEvent`)
//     + Fused `MoE(L) + Attn/Router(L+1)` pipeline (1 sync per layer, zero CPU stall between Hits and Misses!)

#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>
#endif

#define CM_MAX_MODELS   8
#define CM_MAX_SHARDS   16
#define CM_MAX_LAYERS   128
#define CM_MAX_EXPERTS  256
#define CM_MAX_TOPK     16
#define CM_MAX_SEQ      16384
#define CM_DEFAULT_SEQ  2048
#define CM_PREFILL_MAX_B 64
#define CM_STAGE_SLOTS  16
#define CM_IO_WORKERS   8
#define CM_MAX_CPU_THR  64
#define CM_MAX_TP       8
#define CM_CPU_MAX_B    8
#define CM_PA_BLOCK_SIZE       16
#define CM_PA_MAX_SEQS         32
#define CM_PA_MAX_LOGICAL_BLKS (CM_MAX_SEQ / CM_PA_BLOCK_SIZE)
#define CM_TREE_MAX_NODES      64
#define CM_MAX_LORAS           16

#define CM_GGML_F32    0
#define CM_GGML_F16    1
#define CM_GGML_Q4_0   2
#define CM_GGML_Q4_1   3
#define CM_GGML_Q5_0   6
#define CM_GGML_Q5_1   7
#define CM_GGML_Q8_0   8
#define CM_GGML_Q2_K   10
#define CM_GGML_Q3_K   11
#define CM_GGML_Q4_K   12
#define CM_GGML_Q5_K   13
#define CM_GGML_Q6_K   14
#define CM_GGML_IQ4_XS 23
#define CM_GGML_FP8_E4M3 39
#define CM_GGML_MXFP4    40

enum {
    CM_GGUF_U8 = 0, CM_GGUF_I8 = 1, CM_GGUF_U16 = 2, CM_GGUF_I16 = 3,
    CM_GGUF_U32 = 4, CM_GGUF_I32 = 5, CM_GGUF_F32 = 6, CM_GGUF_BOOL = 7,
    CM_GGUF_STR = 8, CM_GGUF_ARR = 9, CM_GGUF_U64 = 10, CM_GGUF_I64 = 11, CM_GGUF_F64 = 12
};

// ============================================================================
// Dynamic CUDA Driver + NVRTC Types & Loader (Zero Static CUDA Links!)
// ============================================================================
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef void* CUevent;
typedef void* CUgraph;
typedef void* CUgraphExec;
typedef uint64_t CUdeviceptr;

typedef int nvrtcResult;
typedef void* nvrtcProgram;

#define CUDA_SUCCESS 0
#define NVRTC_SUCCESS 0
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
#define CU_STREAM_NON_BLOCKING 1
#define CU_EVENT_DISABLE_TIMING 2
#define CU_MEMHOSTALLOC_PORTABLE 1

typedef CUresult (*fn_cuInit)(unsigned int);
typedef CUresult (*fn_cuDeviceGetCount)(int*);
typedef CUresult (*fn_cuDeviceGet)(CUdevice*, int);
typedef CUresult (*fn_cuDeviceGetName)(char*, int, CUdevice);
typedef CUresult (*fn_cuDeviceTotalMem_v2)(size_t*, CUdevice);
typedef CUresult (*fn_cuDeviceGetAttribute)(int*, int, CUdevice);
typedef CUresult (*fn_cuCtxCreate_v2)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*fn_cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
typedef CUresult (*fn_cuCtxSetCurrent)(CUcontext);
typedef CUresult (*fn_cuMemAlloc_v2)(CUdeviceptr*, size_t);
typedef CUresult (*fn_cuMemFree_v2)(CUdeviceptr);
typedef CUresult (*fn_cuMemGetInfo_v2)(size_t*, size_t*);
typedef CUresult (*fn_cuMemHostAlloc)(void**, size_t, unsigned int);
typedef CUresult (*fn_cuMemFreeHost)(void*);
typedef CUresult (*fn_cuMemcpyHtoD_v2)(CUdeviceptr, const void*, size_t);
typedef CUresult (*fn_cuMemcpyDtoH_v2)(void*, CUdeviceptr, size_t);
typedef CUresult (*fn_cuMemcpyHtoDAsync_v2)(CUdeviceptr, const void*, size_t, CUstream);
typedef CUresult (*fn_cuMemcpyDtoHAsync_v2)(void*, CUdeviceptr, size_t, CUstream);
typedef CUresult (*fn_cuMemcpyDtoDAsync_v2)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
typedef CUresult (*fn_cuDeviceCanAccessPeer)(int*, CUdevice, CUdevice);
typedef CUresult (*fn_cuCtxEnablePeerAccess)(CUcontext, unsigned int);
typedef CUresult (*fn_cuMemcpyPeerAsync)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream);
typedef CUresult (*fn_cuMemsetD8_v2)(CUdeviceptr, unsigned char, size_t);
typedef CUresult (*fn_cuMemsetD8Async)(CUdeviceptr, unsigned char, size_t, CUstream);
typedef CUresult (*fn_cuStreamCreate)(CUstream*, unsigned int);
typedef CUresult (*fn_cuStreamDestroy_v2)(CUstream);
typedef CUresult (*fn_cuStreamSynchronize)(CUstream);
typedef CUresult (*fn_cuStreamWaitEvent)(CUstream, CUevent, unsigned int);
typedef CUresult (*fn_cuEventCreate)(CUevent*, unsigned int);
typedef CUresult (*fn_cuEventDestroy_v2)(CUevent);
typedef CUresult (*fn_cuEventRecord)(CUevent, CUstream);
typedef CUresult (*fn_cuEventSynchronize)(CUevent);
typedef CUresult (*fn_cuEventQuery)(CUevent);
typedef CUresult (*fn_cuStreamBeginCapture)(CUstream, int);
typedef CUresult (*fn_cuStreamEndCapture)(CUstream, CUgraph*);
typedef CUresult (*fn_cuGraphInstantiateWithFlags)(CUgraphExec*, CUgraph, uint64_t);
typedef CUresult (*fn_cuGraphInstantiate)(CUgraphExec*, CUgraph, void**, char*, size_t);
typedef CUresult (*fn_cuGraphLaunch)(CUgraphExec, CUstream);
typedef CUresult (*fn_cuGraphExecDestroy)(CUgraphExec);
typedef CUresult (*fn_cuGraphDestroy)(CUgraph);
typedef CUresult (*fn_cuModuleLoadData)(CUmodule*, const void*);
typedef CUresult (*fn_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef CUresult (*fn_cuLaunchKernel)(
    CUfunction, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, CUstream, void**, void**);

typedef nvrtcResult (*fn_nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
typedef nvrtcResult (*fn_nvrtcCompileProgram)(nvrtcProgram, int, const char* const*);
typedef nvrtcResult (*fn_nvrtcGetPTXSize)(nvrtcProgram, size_t*);
typedef nvrtcResult (*fn_nvrtcGetPTX)(nvrtcProgram, char*);
typedef nvrtcResult (*fn_nvrtcGetProgramLogSize)(nvrtcProgram, size_t*);
typedef nvrtcResult (*fn_nvrtcGetProgramLog)(nvrtcProgram, char*);
typedef nvrtcResult (*fn_nvrtcDestroyProgram)(nvrtcProgram*);

// Zero-Link Dynamic NCCL Collective Auto-Dispatch (`dlopen("libnccl.so.2")`)
typedef void* ncclComm_t;
typedef int ncclResult_t;
#define NCCL_SUCCESS 0
#define NCCL_FLOAT32 7
#define NCCL_SUM     0
typedef ncclResult_t (*fn_ncclCommInitAll)(ncclComm_t*, int, const int*);
typedef ncclResult_t (*fn_ncclCommDestroy)(ncclComm_t);
typedef ncclResult_t (*fn_ncclAllReduce)(const void*, void*, size_t, int, int, ncclComm_t, CUstream);
typedef ncclResult_t (*fn_ncclBroadcast)(const void*, void*, size_t, int, int, ncclComm_t, CUstream);
typedef ncclResult_t (*fn_ncclGroupStart)(void);
typedef ncclResult_t (*fn_ncclGroupEnd)(void);

typedef struct {
    int ready;
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    size_t vram_bytes;
    char dev_name[128];
    CUfunction k_embed, k_embed_ind, k_rmsnorm, k_add_bias, k_swiglu_mul, k_quant_q8_1;
    CUfunction k_gemv_q4, k_gemv_qkv_q4, k_gemv_q4_1, k_gemv_q5_0, k_gemv_q5_1, k_gemv_q8, k_gemv_q2k, k_gemv_q3k, k_gemv_iq4xs;
    CUfunction k_gemv_q4k, k_gemv_q5k, k_gemv_q6k, k_gemv_f16, k_gemv_f32;
    CUfunction k_gemv_q4_dp4a, k_gemv_qkv_q4_dp4a, k_gemv_q4_1_dp4a, k_dense_gu_q4_dp4a;
    CUfunction k_dense_gu_q4, k_rope_kv, k_rope_kv_ind, k_flash_dec, k_flash_dec_ind;
    CUfunction k_rope_kv_q8_ind, k_flash_dec_q8_ind, k_rope_kv_b, k_rope_kv_q8_b, k_flash_prefill, k_flash_prefill_q8;
    CUfunction k_gemm_q4_b, k_gemm_q4_1_b, k_gemm_q5_0_b, k_gemm_q5_1_b, k_gemm_q8_b, k_gemm_q2k_b, k_gemm_q3k_b, k_gemm_iq4xs_b;
    CUfunction k_gemm_q4k_b, k_gemm_q5k_b, k_gemm_q6k_b, k_gemm_f16_b, k_gemm_f32_b;
    CUfunction k_add_bias_b, k_mla_split_kv;
    CUfunction k_mla_rope_pe, k_mla_store_latent, k_mla_unpack_wkv_b, k_mla_unpack_kb_vb;
    CUfunction k_mla_absorb_q, k_flash_mla_latent, k_mla_proj_v, k_tp_accum_add;
    CUfunction k_moe_slot_gu_q4, k_moe_slot_down_q4, k_moe_slot_down_q4_1, k_moe_slot_down_q5_0, k_moe_slot_down_q5_1, k_moe_slot_down_q8;
    CUfunction k_moe_slot_gu_q4k, k_moe_slot_down_q4k, k_moe_slot_down_q6k;
    CUfunction k_rope_kv_paged, k_rope_kv_q8_paged, k_flash_dec_paged, k_flash_dec_q8_paged;
    CUfunction k_rope_kv_tree, k_rope_kv_q8_tree, k_flash_tree, k_flash_tree_q8;
    CUfunction k_vit_ln, k_vit_gelu, k_vit_add, k_vit_attn;
    CUfunction k_gemm_q4_sb, k_gemm_q4_1_sb, k_gemm_q4k_sb, k_gemm_q6k_sb;
    CUfunction k_gemv_fp8, k_gemm_fp8_b, k_gemv_mxfp4, k_gemm_mxfp4_b;
    CUfunction k_lora_a, k_lora_b, k_argmax_b, k_rope_kv_mrope, k_router;
    CUfunction k_gemm_q4k_marlin, k_gemm_q6k_marlin, k_gemm_fp8_tc, k_gemm_mxfp4_tc;
    CUfunction k_mla_rope_store_paged, k_flash_mla_latent_paged, k_gather_rows, k_scatter_add_rows;
    CUfunction k_argmax;
    CUdeviceptr d_xq8;
} CmDevCtx;

typedef struct {
    int ready;
    void* libcuda;
    void* libnvrtc;
    void* libnccl;
    int nccl_avail;
    fn_ncclCommInitAll ncclCommInitAll;
    fn_ncclCommDestroy ncclCommDestroy;
    fn_ncclAllReduce ncclAllReduce;
    fn_ncclBroadcast ncclBroadcast;
    fn_ncclGroupStart ncclGroupStart;
    fn_ncclGroupEnd ncclGroupEnd;
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    char dev_name[128];
    int cc_major, cc_minor;
    size_t vram_bytes;
    int n_gpus;
    CmDevCtx gpus[CM_MAX_TP];
    int p2p_matrix[CM_MAX_TP][CM_MAX_TP];
    char* compiled_bin;
    size_t compiled_bin_sz;

    fn_cuInit cuInit;
    fn_cuDeviceGetCount cuDeviceGetCount;
    fn_cuDeviceGet cuDeviceGet;
    fn_cuDeviceGetName cuDeviceGetName;
    fn_cuDeviceTotalMem_v2 cuDeviceTotalMem;
    fn_cuDeviceGetAttribute cuDeviceGetAttribute;
    fn_cuCtxCreate_v2 cuCtxCreate;
    fn_cuDevicePrimaryCtxRetain cuDevicePrimaryCtxRetain;
    fn_cuCtxSetCurrent cuCtxSetCurrent;
    fn_cuMemAlloc_v2 cuMemAlloc;
    fn_cuMemFree_v2 cuMemFree;
    fn_cuMemGetInfo_v2 cuMemGetInfo;
    fn_cuMemHostAlloc cuMemHostAlloc;
    fn_cuMemFreeHost cuMemFreeHost;
    fn_cuMemcpyHtoD_v2 cuMemcpyHtoD;
    fn_cuMemcpyDtoH_v2 cuMemcpyDtoH;
    fn_cuMemcpyHtoDAsync_v2 cuMemcpyHtoDAsync;
    fn_cuMemcpyDtoHAsync_v2 cuMemcpyDtoHAsync;
    fn_cuMemcpyDtoDAsync_v2 cuMemcpyDtoDAsync;
    fn_cuDeviceCanAccessPeer cuDeviceCanAccessPeer;
    fn_cuCtxEnablePeerAccess cuCtxEnablePeerAccess;
    fn_cuMemcpyPeerAsync cuMemcpyPeerAsync;
    fn_cuMemsetD8_v2 cuMemsetD8;
    fn_cuMemsetD8Async cuMemsetD8Async;
    fn_cuStreamCreate cuStreamCreate;
    fn_cuStreamDestroy_v2 cuStreamDestroy;
    fn_cuStreamSynchronize cuStreamSynchronize;
    fn_cuStreamWaitEvent cuStreamWaitEvent;
    fn_cuEventCreate cuEventCreate;
    fn_cuEventDestroy_v2 cuEventDestroy;
    fn_cuEventRecord cuEventRecord;
    fn_cuEventSynchronize cuEventSynchronize;
    fn_cuEventQuery cuEventQuery;
    fn_cuStreamBeginCapture cuStreamBeginCapture;
    fn_cuStreamEndCapture cuStreamEndCapture;
    fn_cuGraphInstantiateWithFlags cuGraphInstantiateWithFlags;
    fn_cuGraphInstantiate cuGraphInstantiate;
    fn_cuGraphLaunch cuGraphLaunch;
    fn_cuGraphExecDestroy cuGraphExecDestroy;
    fn_cuGraphDestroy cuGraphDestroy;
    fn_cuModuleLoadData cuModuleLoadData;
    fn_cuModuleGetFunction cuModuleGetFunction;
    fn_cuLaunchKernel cuLaunchKernel;

    fn_nvrtcCreateProgram nvrtcCreateProgram;
    fn_nvrtcCompileProgram nvrtcCompileProgram;
    fn_nvrtcGetPTXSize nvrtcGetPTXSize;
    fn_nvrtcGetPTX nvrtcGetPTX;
    fn_nvrtcGetProgramLogSize nvrtcGetProgramLogSize;
    fn_nvrtcGetProgramLog nvrtcGetProgramLog;
    fn_nvrtcDestroyProgram nvrtcDestroyProgram;

    CUfunction k_embed, k_embed_ind, k_rmsnorm, k_add_bias, k_swiglu_mul, k_quant_q8_1;
    CUfunction k_gemv_q4, k_gemv_qkv_q4, k_gemv_q4_1, k_gemv_q5_0, k_gemv_q5_1, k_gemv_q8, k_gemv_q2k, k_gemv_q3k, k_gemv_iq4xs;
    CUfunction k_gemv_q4k, k_gemv_q5k, k_gemv_q6k, k_gemv_f16, k_gemv_f32;
    CUfunction k_gemv_q4_dp4a, k_gemv_qkv_q4_dp4a, k_gemv_q4_1_dp4a, k_dense_gu_q4_dp4a;
    CUfunction k_dense_gu_q4, k_rope_kv, k_rope_kv_ind, k_flash_dec, k_flash_dec_ind, k_router;
    CUfunction k_rope_kv_q8_ind, k_flash_dec_q8_ind, k_rope_kv_b, k_rope_kv_q8_b, k_flash_prefill, k_flash_prefill_q8;
    CUfunction k_gemm_q4_b, k_gemm_q4_1_b, k_gemm_q5_0_b, k_gemm_q5_1_b, k_gemm_q8_b, k_gemm_q2k_b, k_gemm_q3k_b, k_gemm_iq4xs_b;
    CUfunction k_gemm_q4k_b, k_gemm_q5k_b, k_gemm_q6k_b, k_gemm_f16_b, k_gemm_f32_b;
    CUfunction k_add_bias_b, k_mla_split_kv;
    CUfunction k_mla_rope_pe, k_mla_store_latent, k_mla_unpack_wkv_b, k_mla_unpack_kb_vb;
    CUfunction k_mla_absorb_q, k_flash_mla_latent, k_mla_proj_v, k_tp_accum_add;
    CUfunction k_moe_gu_q4, k_moe_down_q4, k_moe_down_q4_1;
    CUfunction k_moe_slot_gu_q4, k_moe_slot_down_q4, k_moe_slot_down_q4_1, k_moe_slot_down_q5_0, k_moe_slot_down_q5_1, k_moe_slot_down_q8;
    CUfunction k_moe_slot_gu_q4k, k_moe_slot_down_q4k, k_moe_slot_down_q6k;
    CUfunction k_rope_kv_paged, k_rope_kv_q8_paged, k_flash_dec_paged, k_flash_dec_q8_paged;
    CUfunction k_rope_kv_tree, k_rope_kv_q8_tree, k_flash_tree, k_flash_tree_q8;
    CUfunction k_vit_ln, k_vit_gelu, k_vit_add, k_vit_attn;
    CUfunction k_gemm_q4_sb, k_gemm_q4_1_sb, k_gemm_q4k_sb, k_gemm_q6k_sb;
    CUfunction k_gemv_fp8, k_gemm_fp8_b, k_gemv_mxfp4, k_gemm_mxfp4_b;
    CUfunction k_lora_a, k_lora_b, k_argmax_b, k_rope_kv_mrope;
    CUfunction k_gemm_q4k_marlin, k_gemm_q6k_marlin, k_gemm_fp8_tc, k_gemm_mxfp4_tc;
    CUfunction k_mla_rope_store_paged, k_flash_mla_latent_paged, k_gather_rows, k_scatter_add_rows;
    CUfunction k_argmax;
    CUdeviceptr d_xq8;
    int fast_dp4a;
} CmCuda;

static CmCuda g_cu = {0};

static double cm_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static int64_t cm_os_rss_bytes(void) {
#if defined(__linux__)
    FILE* f = fopen("/proc/self/statm", "r");
    if (f) {
        long sz = 0, res = 0;
        if (fscanf(f, "%ld %ld", &sz, &res) == 2) {
            fclose(f);
            return (int64_t)res * (int64_t)getpagesize();
        }
        fclose(f);
    }
#endif
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
#if defined(__APPLE__)
        return (int64_t)ru.ru_maxrss;
#else
        return (int64_t)ru.ru_maxrss * 1024LL;
#endif
    }
    return 0;
}

static int64_t cm_os_peak_rss_bytes(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
#if defined(__APPLE__)
        return (int64_t)ru.ru_maxrss;
#else
        return (int64_t)ru.ru_maxrss * 1024LL;
#endif
    }
    return 0;
}

static uint64_t cm_ggml_type_bytes(uint32_t t, uint64_t n_elems) {
    switch (t) {
        case CM_GGML_F32:    return n_elems * 4ull;
        case CM_GGML_F16:    return n_elems * 2ull;
        case CM_GGML_Q4_0:   return (n_elems / 32ull) * 18ull;
        case CM_GGML_Q4_1:   return (n_elems / 32ull) * 20ull;
        case CM_GGML_Q5_0:   return (n_elems / 32ull) * 22ull;
        case CM_GGML_Q5_1:   return (n_elems / 32ull) * 24ull;
        case CM_GGML_Q8_0:   return (n_elems / 32ull) * 34ull;
        case CM_GGML_Q2_K:   return (n_elems / 256ull) * 84ull;
        case CM_GGML_Q3_K:   return (n_elems / 256ull) * 110ull;
        case CM_GGML_Q4_K:   return (n_elems / 256ull) * 144ull;
        case CM_GGML_Q5_K:   return (n_elems / 256ull) * 176ull;
        case CM_GGML_Q6_K:   return (n_elems / 256ull) * 210ull;
        case CM_GGML_IQ4_XS: return (n_elems / 256ull) * 136ull;
        case CM_GGML_FP8_E4M3: return (n_elems / 32ull) * 34ull;
        case CM_GGML_MXFP4:    return (n_elems / 32ull) * 17ull;
        default:             return n_elems * 2ull;
    }
}

// ============================================================================
// Embedded Universal Quantized CUDA Kernels (100% Alignment-Safe & Warp-Vectorized!)
// ============================================================================
static const char* kCuMoeSrc =
"#include <cuda_fp16.h>\n"
"#if !defined(__CUDACC_RTC__)\n"
"#include <stdint.h>\n"
"#include <math.h>\n"
"#else\n"
"typedef unsigned char uint8_t;\n"
"typedef signed char int8_t;\n"
"typedef unsigned short uint16_t;\n"
"typedef unsigned int uint32_t;\n"
"typedef int int32_t;\n"
"typedef unsigned long long uint64_t;\n"
"typedef long long int64_t;\n"
"#endif\n"
"\n"
"__device__ __forceinline__ float warp_sum(float v) {\n"
"    #pragma unroll\n"
"    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);\n"
"    return v;\n"
"}\n"
"\n"
"__device__ __forceinline__ float load_f16_le(const uint8_t* p) {\n"
"    uint16_t bits = *(const uint16_t*)p;\n"
"    return __half2float(__ushort_as_half(bits));\n"
"}\n"
"\n"
"__device__ __forceinline__ float load_x32_q4(const float* __restrict__ x, float4 xv_lo[4], float4 xv_hi[4]) {\n"
"    const float4* x4 = (const float4*)x;\n"
"    float4 xs = make_float4(0.0f, 0.0f, 0.0f, 0.0f);\n"
"    #pragma unroll\n"
"    for (uint32_t g = 0; g < 4u; g++) {\n"
"        float4 lo = __ldg(&x4[g]);\n"
"        float4 hi = __ldg(&x4[g + 4u]);\n"
"        xs.x += lo.x + hi.x; xs.y += lo.y + hi.y; xs.z += lo.z + hi.z; xs.w += lo.w + hi.w;\n"
"        xv_lo[g] = lo;\n"
"        xv_hi[g] = hi;\n"
"    }\n"
"    return (xs.x + xs.y) + (xs.z + xs.w);\n"
"}\n"
"\n"
"// Helper: dot 4 packed bytes (4 lo nibbles + 4 hi nibbles) against float4 lo/hi\n"
"__device__ __forceinline__ void dot_q4_word4(uint32_t w, float4 lo, float4 hi, float& s0, float& s1) {\n"
"    uint32_t b0 = w & 0xFFu, b1 = (w >> 8) & 0xFFu, b2 = (w >> 16) & 0xFFu, b3 = (w >> 24) & 0xFFu;\n"
"    s0 = fmaf((float)(b0 & 0x0Fu), lo.x, fmaf((float)(b1 & 0x0Fu), lo.y, fmaf((float)(b2 & 0x0Fu), lo.z, fmaf((float)(b3 & 0x0Fu), lo.w, s0))));\n"
"    s1 = fmaf((float)(b0 & 0xF0u), hi.x, fmaf((float)(b1 & 0xF0u), hi.y, fmaf((float)(b2 & 0xF0u), hi.z, fmaf((float)(b3 & 0xF0u), hi.w, s1))));\n"
"}\n"
"\n"
"// 2-byte alignment-safe 16-bit vectorized FP32 Q4_0 dot (854+ GB/s GDDR6X + 100% BIT-EXACT!)\n"
"\n"
"// 2-byte alignment-safe 16-bit vectorized Q4_0 dot (18B/blk: half d + 16B qs, ZERO bit shifts!)\n"
"__device__ __forceinline__ float dot_q4_0_fast(const uint8_t* __restrict__ blk, const float4 xv_lo[4], const float4 xv_hi[4], float x_sum) {\n"
"    float d = load_f16_le(blk);\n"
"    const uint16_t* q16 = (const uint16_t*)(blk + 2u);\n"
"    float s0 = 0.0f, s1 = 0.0f;\n"
"    #pragma unroll\n"
"    for (uint32_t g = 0; g < 4u; g++) {\n"
"        uint16_t w0 = __ldg(&q16[g * 2u]);\n"
"        uint16_t w1 = __ldg(&q16[g * 2u + 1u]);\n"
"        uint32_t b0 = w0 & 0xFFu, b1 = (w0 >> 8) & 0xFFu, b2 = w1 & 0xFFu, b3 = (w1 >> 8) & 0xFFu;\n"
"        float4 lo = xv_lo[g], hi = xv_hi[g];\n"
"        s0 = fmaf((float)(b0 & 0x0Fu), lo.x, fmaf((float)(b1 & 0x0Fu), lo.y, fmaf((float)(b2 & 0x0Fu), lo.z, fmaf((float)(b3 & 0x0Fu), lo.w, s0))));\n"
"        s1 = fmaf((float)(b0 & 0xF0u), hi.x, fmaf((float)(b1 & 0xF0u), hi.y, fmaf((float)(b2 & 0xF0u), hi.z, fmaf((float)(b3 & 0xF0u), hi.w, s1))));\n"
"    }\n"
"    return d * (fmaf(s1, 0.0625f, s0) - 8.0f * x_sum);\n"
"}\n"
"\n"
"// 4-byte alignment-safe 32-bit vectorized Q4_1 dot (20B/blk: half d + half m + 16B qs, 4x fewer loads!)\n"
"__device__ __forceinline__ float dot_q4_1_fast(const uint8_t* __restrict__ blk, const float4 xv_lo[4], const float4 xv_hi[4], float x_sum) {\n"
"    float d = load_f16_le(blk);\n"
"    float m = load_f16_le(blk + 2u);\n"
"    const uint32_t* q32 = (const uint32_t*)(blk + 4u);\n"
"    float s0 = 0.0f, s1 = 0.0f;\n"
"    #pragma unroll\n"
"    for (uint32_t g = 0; g < 4u; g++) {\n"
"        uint32_t w = __ldg(&q32[g]);\n"
"        uint32_t b0 = w & 0xFFu, b1 = (w >> 8) & 0xFFu, b2 = (w >> 16) & 0xFFu, b3 = (w >> 24) & 0xFFu;\n"
"        float4 lo = xv_lo[g], hi = xv_hi[g];\n"
"        s0 = fmaf((float)(b0 & 0x0Fu), lo.x, fmaf((float)(b1 & 0x0Fu), lo.y, fmaf((float)(b2 & 0x0Fu), lo.z, fmaf((float)(b3 & 0x0Fu), lo.w, s0))));\n"
"        s1 = fmaf((float)(b0 & 0xF0u), hi.x, fmaf((float)(b1 & 0xF0u), hi.y, fmaf((float)(b2 & 0xF0u), hi.z, fmaf((float)(b3 & 0xF0u), hi.w, s1))));\n"
"    }\n"
"    return fmaf(d, fmaf(s1, 0.0625f, s0), m * x_sum);\n"
"}\n"
"\n"
"// 2-byte alignment-safe Q5_0 dot (22B/blk: half d + uint32 qh + 16B qs)\n"
"__device__ __forceinline__ float dot_q5_0_fast(const uint8_t* __restrict__ blk, const float4 xv_lo[4], const float4 xv_hi[4], float x_sum) {\n"
"    float d = load_f16_le(blk);\n"
"    const uint16_t* h16 = (const uint16_t*)(blk + 2u);\n"
"    uint32_t qh = (uint32_t)__ldg(&h16[0]) | ((uint32_t)__ldg(&h16[1]) << 16);\n"
"    const uint16_t* q16 = (const uint16_t*)(blk + 6u);\n"
"    float s = 0.0f;\n"
"    #pragma unroll\n"
"    for (uint32_t g = 0; g < 4u; g++) {\n"
"        uint16_t w0 = __ldg(&q16[g * 2u]), w1 = __ldg(&q16[g * 2u + 1u]);\n"
"        uint32_t b0 = w0 & 0xFFu, b1 = w0 >> 8, b2 = w1 & 0xFFu, b3 = w1 >> 8;\n"
"        uint32_t j = g * 4u;\n"
"        float4 lo = xv_lo[g], hi = xv_hi[g];\n"
"        s = fmaf((float)((b0 & 0xFu) | (((qh >> (j+0u)) & 1u) << 4)), lo.x, s);\n"
"        s = fmaf((float)((b1 & 0xFu) | (((qh >> (j+1u)) & 1u) << 4)), lo.y, s);\n"
"        s = fmaf((float)((b2 & 0xFu) | (((qh >> (j+2u)) & 1u) << 4)), lo.z, s);\n"
"        s = fmaf((float)((b3 & 0xFu) | (((qh >> (j+3u)) & 1u) << 4)), lo.w, s);\n"
"        s = fmaf((float)((b0 >> 4)   | (((qh >> (j+16u)) & 1u) << 4)), hi.x, s);\n"
"        s = fmaf((float)((b1 >> 4)   | (((qh >> (j+17u)) & 1u) << 4)), hi.y, s);\n"
"        s = fmaf((float)((b2 >> 4)   | (((qh >> (j+18u)) & 1u) << 4)), hi.z, s);\n"
"        s = fmaf((float)((b3 >> 4)   | (((qh >> (j+19u)) & 1u) << 4)), hi.w, s);\n"
"    }\n"
"    return d * (s - 16.0f * x_sum);\n"
"}\n"
"\n"
"// 4-byte alignment-safe Q5_1 dot (24B/blk: half d + half m + uint32 qh + 16B qs)\n"
"__device__ __forceinline__ float dot_q5_1_fast(const uint8_t* __restrict__ blk, const float4 xv_lo[4], const float4 xv_hi[4], float x_sum) {\n"
"    float d = load_f16_le(blk), m = load_f16_le(blk + 2u);\n"
"    uint32_t qh = __ldg((const uint32_t*)(blk + 4u));\n"
"    const uint32_t* q32 = (const uint32_t*)(blk + 8u);\n"
"    float s = 0.0f;\n"
"    #pragma unroll\n"
"    for (uint32_t g = 0; g < 4u; g++) {\n"
"        uint32_t w = __ldg(&q32[g]);\n"
"        uint32_t b0 = w & 0xFFu, b1 = (w >> 8) & 0xFFu, b2 = (w >> 16) & 0xFFu, b3 = (w >> 24) & 0xFFu;\n"
"        uint32_t j = g * 4u;\n"
"        float4 lo = xv_lo[g], hi = xv_hi[g];\n"
"        s = fmaf((float)((b0 & 0xFu) | (((qh >> (j+0u)) & 1u) << 4)), lo.x, s);\n"
"        s = fmaf((float)((b1 & 0xFu) | (((qh >> (j+1u)) & 1u) << 4)), lo.y, s);\n"
"        s = fmaf((float)((b2 & 0xFu) | (((qh >> (j+2u)) & 1u) << 4)), lo.z, s);\n"
"        s = fmaf((float)((b3 & 0xFu) | (((qh >> (j+3u)) & 1u) << 4)), lo.w, s);\n"
"        s = fmaf((float)((b0 >> 4)   | (((qh >> (j+16u)) & 1u) << 4)), hi.x, s);\n"
"        s = fmaf((float)((b1 >> 4)   | (((qh >> (j+17u)) & 1u) << 4)), hi.y, s);\n"
"        s = fmaf((float)((b2 >> 4)   | (((qh >> (j+18u)) & 1u) << 4)), hi.z, s);\n"
"        s = fmaf((float)((b3 >> 4)   | (((qh >> (j+19u)) & 1u) << 4)), hi.w, s);\n"
"    }\n"
"    return fmaf(d, s, m * x_sum);\n"
"}\n"
"\n"
"// 2-byte alignment-safe Q8_0 dot (34B/blk: half d + 32 int8)\n"
"__device__ __forceinline__ float dot_q8_0_fast(const uint8_t* __restrict__ blk, const float4 xv[8]) {\n"
"    float d = load_f16_le(blk);\n"
"    const uint16_t* q16 = (const uint16_t*)(blk + 2u);\n"
"    float acc = 0.0f;\n"
"    #pragma unroll\n"
"    for (uint32_t g = 0; g < 8u; g++) {\n"
"        uint16_t w0 = __ldg(&q16[g * 2u]);\n"
"        uint16_t w1 = __ldg(&q16[g * 2u + 1u]);\n"
"        int8_t c0 = (int8_t)(w0 & 0xFFu), c1 = (int8_t)(w0 >> 8), c2 = (int8_t)(w1 & 0xFFu), c3 = (int8_t)(w1 >> 8);\n"
"        float4 v = xv[g];\n"
"        acc = fmaf((float)c0, v.x, fmaf((float)c1, v.y, fmaf((float)c2, v.z, fmaf((float)c3, v.w, acc))));\n"
"    }\n"
"    return d * acc;\n"
"}\n"
"\n"
"// Vectorized 4-lane/superblock Q4_K quarter-block (float4 x loads + 4x higher warp occupancy!)\n"
"__device__ __forceinline__ float dot_q4_k_quarter(const uint8_t* __restrict__ blk, const float* __restrict__ x, uint32_t qtr) {\n"
"    float d = load_f16_le(blk), dmin = load_f16_le(blk + 2u);\n"
"    const uint8_t* sc = blk + 4u; const uint8_t* qs = blk + 16u;\n"
"    uint32_t l0 = qtr * 8u; float sum = 0.0f; uint32_t is = 0u;\n"
"    #pragma unroll\n"
"    for (uint32_t j = 0; j < 256u; j += 64u) {\n"
"        uint8_t sc0, m0, sc1, m1;\n"
"        if (is < 4u) { sc0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u; sc1 = sc[is + 1u] & 63u; m1 = sc[is + 5u] & 63u; }\n"
"        else { sc0 = (sc[is + 4u] & 0xFu) | ((sc[is - 4u] >> 6) << 4); m0 = (sc[is + 4u] >> 4) | ((sc[is] >> 6) << 4);\n"
"               sc1 = (sc[is + 5u] & 0xFu) | ((sc[is - 3u] >> 6) << 4); m1 = (sc[is + 5u] >> 4) | ((sc[is + 1u] >> 6) << 4); }\n"
"        float d1 = d * (float)sc0, dm1 = dmin * (float)m0, d2 = d * (float)sc1, dm2 = dmin * (float)m1;\n"
"        const uint8_t* q = qs + (j >> 1) + l0;\n"
"        const float4* xl = (const float4*)(x + j + l0);\n"
"        const float4* xh = (const float4*)(x + j + 32u + l0);\n"
"        float p0 = 0.0f, p1 = 0.0f, sx0 = 0.0f, sx1 = 0.0f;\n"
"        #pragma unroll\n"
"        for (uint32_t g = 0; g < 2u; g++) {\n"
"            float4 vlo = __ldg(&xl[g]), vhi = __ldg(&xh[g]);\n"
"            sx0 += (vlo.x + vlo.y) + (vlo.z + vlo.w); sx1 += (vhi.x + vhi.y) + (vhi.z + vhi.w);\n"
"            uint8_t b0 = __ldg(&q[g*4u+0u]), b1 = __ldg(&q[g*4u+1u]), b2 = __ldg(&q[g*4u+2u]), b3 = __ldg(&q[g*4u+3u]);\n"
"            p0 = fmaf((float)(b0 & 0xFu), vlo.x, fmaf((float)(b1 & 0xFu), vlo.y, fmaf((float)(b2 & 0xFu), vlo.z, fmaf((float)(b3 & 0xFu), vlo.w, p0))));\n"
"            p1 = fmaf((float)(b0 >> 4),   vhi.x, fmaf((float)(b1 >> 4),   vhi.y, fmaf((float)(b2 >> 4),   vhi.z, fmaf((float)(b3 >> 4),   vhi.w, p1))));\n"
"        }\n"
"        sum += fmaf(d1, p0, -dm1 * sx0) + fmaf(d2, p1, -dm2 * sx1);\n"
"        is += 2u;\n"
"    }\n"
"    return sum;\n"
"}\n"
"\n"
"__device__ __forceinline__ float dot_q5_k_quarter(const uint8_t* __restrict__ blk, const float* __restrict__ x, uint32_t qtr) {\n"
"    float d = load_f16_le(blk), dmin = load_f16_le(blk + 2u);\n"
"    const uint8_t* sc = blk + 4u; const uint8_t* qh = blk + 16u; const uint8_t* qs = blk + 48u;\n"
"    uint32_t l0 = qtr * 8u; float sum = 0.0f; uint32_t is = 0u, u1 = 1u, u2 = 2u;\n"
"    for (uint32_t j = 0; j < 256u; j += 64u) {\n"
"        uint8_t sc0, m0, sc1, m1;\n"
"        if (is < 4u) { sc0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u; sc1 = sc[is + 1u] & 63u; m1 = sc[is + 5u] & 63u; }\n"
"        else { sc0 = (sc[is + 4u] & 0xFu) | ((sc[is - 4u] >> 6) << 4); m0 = (sc[is + 4u] >> 4) | ((sc[is] >> 6) << 4);\n"
"               sc1 = (sc[is + 5u] & 0xFu) | ((sc[is - 3u] >> 6) << 4); m1 = (sc[is + 5u] >> 4) | ((sc[is + 1u] >> 6) << 4); }\n"
"        float d1 = d * (float)sc0, dm1 = dmin * (float)m0, d2 = d * (float)sc1, dm2 = dmin * (float)m1;\n"
"        const uint8_t* q = qs + (j >> 1);\n"
"        float p0 = 0.0f, p1 = 0.0f, sx0 = 0.0f, sx1 = 0.0f;\n"
"        for (uint32_t l = l0; l < l0 + 8u; l++) {\n"
"            uint8_t qv = q[l], hv = qh[l];\n"
"            float qlo = (float)((qv & 0xFu) + ((hv & u1) ? 16u : 0u));\n"
"            float qhi = (float)((qv >> 4)   + ((hv & u2) ? 16u : 0u));\n"
"            float x0 = x[j + l], x1 = x[j + 32u + l];\n"
"            p0 = fmaf(qlo, x0, p0); sx0 += x0;\n"
"            p1 = fmaf(qhi, x1, p1); sx1 += x1;\n"
"        }\n"
"        sum += fmaf(d1, p0, -dm1 * sx0) + fmaf(d2, p1, -dm2 * sx1);\n"
"        is += 2u; u1 <<= 2; u2 <<= 2;\n"
"    }\n"
"    return sum;\n"
"}\n"
"\n"
"__device__ __forceinline__ float dot_q6_k_quarter(const uint8_t* __restrict__ blk, const float* __restrict__ x, uint32_t qtr) {\n"
"    const uint8_t* ql = blk; const uint8_t* qh = blk + 128u; const int8_t* sc = (const int8_t*)(blk + 192u);\n"
"    float d = load_f16_le(blk + 208u); uint32_t l0 = qtr * 8u; uint32_t is = l0 >> 4; float sum = 0.0f;\n"
"    #pragma unroll\n"
"    for (uint32_t n = 0; n < 256u; n += 128u) {\n"
"        const uint8_t* qlp = ql + (n >> 1) + l0; const uint8_t* qhp = qh + (n >> 2) + l0;\n"
"        const int8_t* scp = sc + (n >> 4); const float* xp = x + n + l0;\n"
"        float s0 = (float)__ldg(&scp[is + 0]), s2 = (float)__ldg(&scp[is + 2]), s4 = (float)__ldg(&scp[is + 4]), s6 = (float)__ldg(&scp[is + 6]);\n"
"        float p0 = 0.0f, p2 = 0.0f, p4 = 0.0f, p6 = 0.0f;\n"
"        const float4* x0 = (const float4*)(xp + 0u);  const float4* x2 = (const float4*)(xp + 32u);\n"
"        const float4* x4 = (const float4*)(xp + 64u); const float4* x6 = (const float4*)(xp + 96u);\n"
"        #pragma unroll\n"
"        for (uint32_t g = 0; g < 2u; g++) {\n"
"            float4 v0 = __ldg(&x0[g]), v2 = __ldg(&x2[g]), v4 = __ldg(&x4[g]), v6 = __ldg(&x6[g]);\n"
"            uint32_t o = g * 4u;\n"
"            #define Q6_TAP(idx, f0, f2, f4, f6) do { \\\n"
"                uint8_t qa = __ldg(&qlp[o + (idx)]), qb = __ldg(&qlp[32u + o + (idx)]), qhv = __ldg(&qhp[o + (idx)]); \\\n"
"                p0 = fmaf((float)((int)((qa & 0xFu) | (((qhv >> 0) & 3u) << 4)) - 32), (f0), p0); \\\n"
"                p2 = fmaf((float)((int)((qb & 0xFu) | (((qhv >> 2) & 3u) << 4)) - 32), (f2), p2); \\\n"
"                p4 = fmaf((float)((int)((qa >> 4)   | (((qhv >> 4) & 3u) << 4)) - 32), (f4), p4); \\\n"
"                p6 = fmaf((float)((int)((qb >> 4)   | (((qhv >> 6) & 3u) << 4)) - 32), (f6), p6); \\\n"
"            } while (0)\n"
"            Q6_TAP(0, v0.x, v2.x, v4.x, v6.x); Q6_TAP(1, v0.y, v2.y, v4.y, v6.y);\n"
"            Q6_TAP(2, v0.z, v2.z, v4.z, v6.z); Q6_TAP(3, v0.w, v2.w, v4.w, v6.w);\n"
"            #undef Q6_TAP\n"
"        }\n"
"        sum += fmaf(s0, p0, fmaf(s2, p2, fmaf(s4, p4, s6 * p6)));\n"
"    }\n"
"    return d * sum;\n"
"}\n"
"\n"
"__device__ __constant__ int8_t kIq4Nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};\n"
"\n"
"__device__ __forceinline__ float dot_q2_k_quarter(const uint8_t* __restrict__ blk, const float* __restrict__ x, uint32_t qtr) {\n"
"    const uint8_t* sc = blk; const uint8_t* qs = blk + 16u;\n"
"    float d = load_f16_le(blk + 80u), dmin = load_f16_le(blk + 82u);\n"
"    uint32_t l0 = qtr * 8u, sb = qtr >> 1; float sum = 0.0f;\n"
"    #pragma unroll\n"
"    for (uint32_t n = 0; n < 256u; n += 128u) {\n"
"        const uint8_t* q = qs + (n >> 2);\n"
"        uint32_t is_b = n >> 4;\n"
"        #pragma unroll\n"
"        for (uint32_t sg = 0; sg < 4u; sg++) {\n"
"            uint32_t sh = sg * 2u;\n"
"            uint8_t scv = __ldg(&sc[is_b + sg * 2u + sb]);\n"
"            float dl = d * (float)(scv & 0xFu), ml = dmin * (float)(scv >> 4);\n"
"            const float* xp = x + n + sg * 32u + l0;\n"
"            float p = 0.0f, sx = 0.0f;\n"
"            #pragma unroll\n"
"            for (uint32_t i = 0; i < 8u; i++) {\n"
"                float xv = __ldg(&xp[i]);\n"
"                uint32_t qv = (__ldg(&q[l0 + i]) >> sh) & 3u;\n"
"                p = fmaf((float)qv, xv, p); sx += xv;\n"
"            }\n"
"            sum += fmaf(dl, p, -ml * sx);\n"
"        }\n"
"    }\n"
"    return sum;\n"
"}\n"
"\n"
"__device__ __forceinline__ float dot_q3_k_quarter(const uint8_t* __restrict__ blk, const float* __restrict__ x, uint32_t qtr) {\n"
"    const uint8_t* hm = blk; const uint8_t* qs = blk + 32u; const uint8_t* scb = blk + 96u;\n"
"    float d = load_f16_le(blk + 108u);\n"
"    uint32_t a0 = (uint32_t)scb[0] | ((uint32_t)scb[1]<<8) | ((uint32_t)scb[2]<<16) | ((uint32_t)scb[3]<<24);\n"
"    uint32_t a1 = (uint32_t)scb[4] | ((uint32_t)scb[5]<<8) | ((uint32_t)scb[6]<<16) | ((uint32_t)scb[7]<<24);\n"
"    uint32_t a2 = (uint32_t)scb[8] | ((uint32_t)scb[9]<<8) | ((uint32_t)scb[10]<<16) | ((uint32_t)scb[11]<<24);\n"
"    uint32_t scu[4];\n"
"    scu[0] = (a0 & 0x0F0F0F0Fu) | (((a2 >> 0) & 0x03030303u) << 4);\n"
"    scu[1] = (a1 & 0x0F0F0F0Fu) | (((a2 >> 2) & 0x03030303u) << 4);\n"
"    scu[2] = ((a0 >> 4) & 0x0F0F0F0Fu) | (((a2 >> 4) & 0x03030303u) << 4);\n"
"    scu[3] = ((a1 >> 4) & 0x0F0F0F0Fu) | (((a2 >> 6) & 0x03030303u) << 4);\n"
"    uint32_t l0 = qtr * 8u, sb = qtr >> 1; float sum = 0.0f;\n"
"    #pragma unroll\n"
"    for (uint32_t n = 0; n < 256u; n += 128u) {\n"
"        const uint8_t* q = qs + (n >> 2);\n"
"        uint32_t is_b = n >> 4;\n"
"        #pragma unroll\n"
"        for (uint32_t sg = 0; sg < 4u; sg++) {\n"
"            uint32_t sh = sg * 2u, m = 1u << ((n >> 7) * 4u + sg);\n"
"            uint32_t is = is_b + sg * 2u + sb;\n"
"            int scv = (int)((scu[is >> 2] >> ((is & 3u) * 8u)) & 0xFFu) - 32;\n"
"            float dl = d * (float)scv;\n"
"            const float* xp = x + n + sg * 32u + l0;\n"
"            float p = 0.0f;\n"
"            #pragma unroll\n"
"            for (uint32_t i = 0; i < 8u; i++) {\n"
"                int q3 = (int)((__ldg(&q[l0 + i]) >> sh) & 3u) - ((__ldg(&hm[l0 + i]) & m) ? 0 : 4);\n"
"                p = fmaf((float)q3, __ldg(&xp[i]), p);\n"
"            }\n"
"            sum = fmaf(dl, p, sum);\n"
"        }\n"
"    }\n"
"    return sum;\n"
"}\n"
"\n"
"__device__ __forceinline__ float dot_iq4_xs_quarter(const uint8_t* __restrict__ blk, const float* __restrict__ x, uint32_t qtr) {\n"
"    float d = load_f16_le(blk);\n"
"    uint16_t sch = *(const uint16_t*)(blk + 2u);\n"
"    const uint8_t* scl = blk + 4u; const uint8_t* qs = blk + 8u;\n"
"    uint32_t l0 = qtr * 4u; float sum = 0.0f;\n"
"    #pragma unroll\n"
"    for (uint32_t ib = 0; ib < 8u; ib++) {\n"
"        uint32_t ls = ((scl[ib >> 1] >> ((ib & 1u) * 4u)) & 0xFu) | (((uint32_t)(sch >> (ib * 2u)) & 3u) << 4);\n"
"        float dl = d * (float)((int)ls - 32);\n"
"        const uint8_t* qp = qs + ib * 16u + l0;\n"
"        const float* xlo = x + ib * 32u + l0; const float* xhi = x + ib * 32u + 16u + l0;\n"
"        float p = 0.0f;\n"
"        #pragma unroll\n"
"        for (uint32_t i = 0; i < 4u; i++) {\n"
"            uint8_t bv = __ldg(&qp[i]);\n"
"            p = fmaf((float)kIq4Nl[bv & 0xFu], __ldg(&xlo[i]), fmaf((float)kIq4Nl[bv >> 4], __ldg(&xhi[i]), p));\n"
"        }\n"
"        sum = fmaf(dl, p, sum);\n"
"    }\n"
"    return sum;\n"
"}\n"
"\n"
"__device__ __forceinline__ float dequant_any_elem(const uint8_t* w, uint32_t gtype, uint32_t row, uint32_t col, uint32_t in_dim) {\n"
"    if (gtype == 0u) return ((const float*)w)[(uint64_t)row * in_dim + col];\n"
"    if (gtype == 1u) return __half2float(((const half*)w)[(uint64_t)row * in_dim + col]);\n"
"    if (gtype == 2u) {\n"
"        uint64_t rb = (uint64_t)(in_dim >> 5) * 18ull;\n"
"        const uint8_t* bp = w + (uint64_t)row * rb + (uint64_t)(col >> 5) * 18ull;\n"
"        uint32_t sub = col & 31u; uint8_t bv = bp[2u + (sub & 15u)];\n"
"        return load_f16_le(bp) * (float)((sub < 16u) ? ((int)(bv & 0xFu) - 8) : ((int)(bv >> 4) - 8));\n"
"    }\n"
"    if (gtype == 3u) {\n"
"        uint64_t rb = (uint64_t)(in_dim >> 5) * 20ull;\n"
"        const uint8_t* bp = w + (uint64_t)row * rb + (uint64_t)(col >> 5) * 20ull;\n"
"        uint32_t sub = col & 31u; uint8_t bv = bp[4u + (sub & 15u)];\n"
"        return fmaf(load_f16_le(bp), (float)((sub < 16u) ? (bv & 0xFu) : (bv >> 4)), load_f16_le(bp + 2u));\n"
"    }\n"
"    if (gtype == 6u) {\n"
"        uint64_t rb = (uint64_t)(in_dim >> 5) * 22ull;\n"
"        const uint8_t* bp = w + (uint64_t)row * rb + (uint64_t)(col >> 5) * 22ull;\n"
"        uint32_t sub = col & 31u; uint8_t bv = bp[6u + (sub & 15u)];\n"
"        uint32_t qh = (uint32_t)(*(const uint16_t*)(bp + 2u)) | ((uint32_t)(*(const uint16_t*)(bp + 4u)) << 16);\n"
"        uint32_t qv = ((sub < 16u) ? (bv & 0xFu) : (bv >> 4)) | (((qh >> sub) & 1u) << 4);\n"
"        return load_f16_le(bp) * (float)((int)qv - 16);\n"
"    }\n"
"    if (gtype == 7u) {\n"
"        uint64_t rb = (uint64_t)(in_dim >> 5) * 24ull;\n"
"        const uint8_t* bp = w + (uint64_t)row * rb + (uint64_t)(col >> 5) * 24ull;\n"
"        uint32_t sub = col & 31u; uint8_t bv = bp[8u + (sub & 15u)];\n"
"        uint32_t qh = *(const uint32_t*)(bp + 4u);\n"
"        uint32_t qv = ((sub < 16u) ? (bv & 0xFu) : (bv >> 4)) | (((qh >> sub) & 1u) << 4);\n"
"        return fmaf(load_f16_le(bp), (float)qv, load_f16_le(bp + 2u));\n"
"    }\n"
"    if (gtype == 8u) {\n"
"        uint64_t rb = (uint64_t)(in_dim >> 5) * 34ull;\n"
"        const uint8_t* bp = w + (uint64_t)row * rb + (uint64_t)(col >> 5) * 34ull;\n"
"        return load_f16_le(bp) * (float)((const int8_t*)(bp + 2u))[col & 31u];\n"
"    }\n"
"    if (gtype == 10u) {\n"
"        uint64_t rb = (uint64_t)(in_dim >> 8) * 84ull;\n"
"        const uint8_t* bp = w + (uint64_t)row * rb + (uint64_t)(col >> 8) * 84ull;\n"
"        uint32_t rem = col & 255u, n = rem & 128u, r = rem & 127u, sg = r >> 5, l = r & 31u;\n"
"        uint8_t scv = bp[(n >> 4) + sg * 2u + (l >> 4)];\n"
"        uint32_t qv = (bp[16u + (n >> 2) + l] >> (sg * 2u)) & 3u;\n"
"        return fmaf(load_f16_le(bp + 80u) * (float)(scv & 0xFu), (float)qv, -load_f16_le(bp + 82u) * (float)(scv >> 4));\n"
"    }\n"
"    if (gtype == 11u) {\n"
"        uint64_t rb = (uint64_t)(in_dim >> 8) * 110ull;\n"
"        const uint8_t* bp = w + (uint64_t)row * rb + (uint64_t)(col >> 8) * 110ull;\n"
"        const uint8_t* scb = bp + 96u;\n"
"        uint32_t a0 = (uint32_t)scb[0] | ((uint32_t)scb[1]<<8) | ((uint32_t)scb[2]<<16) | ((uint32_t)scb[3]<<24);\n"
"        uint32_t a1 = (uint32_t)scb[4] | ((uint32_t)scb[5]<<8) | ((uint32_t)scb[6]<<16) | ((uint32_t)scb[7]<<24);\n"
"        uint32_t a2 = (uint32_t)scb[8] | ((uint32_t)scb[9]<<8) | ((uint32_t)scb[10]<<16) | ((uint32_t)scb[11]<<24);\n"
"        uint32_t scu[4] = {(a0&0x0F0F0F0Fu)|(((a2>>0)&0x03030303u)<<4), (a1&0x0F0F0F0Fu)|(((a2>>2)&0x03030303u)<<4),\n"
"                           ((a0>>4)&0x0F0F0F0Fu)|(((a2>>4)&0x03030303u)<<4), ((a1>>4)&0x0F0F0F0Fu)|(((a2>>6)&0x03030303u)<<4)};\n"
"        uint32_t rem = col & 255u, n = rem & 128u, r = rem & 127u, sg = r >> 5, l = r & 31u;\n"
"        uint32_t is = (n >> 4) + sg * 2u + (l >> 4), m = 1u << ((n >> 7) * 4u + sg);\n"
"        int scv = (int)((scu[is >> 2] >> ((is & 3u) * 8u)) & 0xFFu) - 32;\n"
"        int q3 = (int)((bp[32u + (n >> 2) + l] >> (sg * 2u)) & 3u) - ((bp[l] & m) ? 0 : 4);\n"
"        return load_f16_le(bp + 108u) * (float)scv * (float)q3;\n"
"    }\n"
"    if (gtype == 12u) {\n"
"        uint64_t rb = (uint64_t)(in_dim >> 8) * 144ull;\n"
"        const uint8_t* bp = w + (uint64_t)row * rb + (uint64_t)(col >> 8) * 144ull;\n"
"        uint32_t rem = col & 255u, sb = rem >> 5, l = rem & 31u; const uint8_t* sc = bp + 4u;\n"
"        uint8_t scv, mv;\n"
"        if (sb < 4u) { scv = sc[sb] & 63u; mv = sc[sb + 4u] & 63u; }\n"
"        else { scv = (sc[sb + 4u] & 0xFu) | ((sc[sb - 4u] >> 6) << 4); mv = (sc[sb + 4u] >> 4) | ((sc[sb] >> 6) << 4); }\n"
"        uint8_t bv = bp[16u + (sb >> 1) * 32u + l];\n"
"        uint32_t qv = (sb & 1u) ? (bv >> 4) : (bv & 0xFu);\n"
"        return fmaf(load_f16_le(bp) * (float)scv, (float)qv, -load_f16_le(bp + 2u) * (float)mv);\n"
"    }\n"
"    if (gtype == 13u) {\n"
"        uint64_t rb = (uint64_t)(in_dim >> 8) * 176ull;\n"
"        const uint8_t* bp = w + (uint64_t)row * rb + (uint64_t)(col >> 8) * 176ull;\n"
"        uint32_t rem = col & 255u, sb = rem >> 5, l = rem & 31u; const uint8_t* sc = bp + 4u;\n"
"        uint8_t scv, mv;\n"
"        if (sb < 4u) { scv = sc[sb] & 63u; mv = sc[sb + 4u] & 63u; }\n"
"        else { scv = (sc[sb + 4u] & 0xFu) | ((sc[sb - 4u] >> 6) << 4); mv = (sc[sb + 4u] >> 4) | ((sc[sb] >> 6) << 4); }\n"
"        uint8_t bv = bp[48u + (sb >> 1) * 32u + l];\n"
"        uint32_t qv = ((sb & 1u) ? (bv >> 4) : (bv & 0xFu)) + ((bp[16u + l] & (1u << sb)) ? 16u : 0u);\n"
"        return fmaf(load_f16_le(bp) * (float)scv, (float)qv, -load_f16_le(bp + 2u) * (float)mv);\n"
"    }\n"
"    if (gtype == 14u) {\n"
"        uint64_t rb = (uint64_t)(in_dim >> 8) * 210ull;\n"
"        const uint8_t* bp = w + (uint64_t)row * rb + (uint64_t)(col >> 8) * 210ull;\n"
"        uint32_t rem = col & 255u, n = (rem >= 128u) ? 128u : 0u, r = rem - n, qpart = r >> 5, l = r & 31u;\n"
"        const uint8_t* qlp = bp + (n >> 1); const uint8_t* qhp = bp + 128u + (n >> 2);\n"
"        const int8_t* scp = (const int8_t*)(bp + 192u) + (n >> 4);\n"
"        uint8_t lo = (qpart == 0u) ? (qlp[l] & 0xFu) : (qpart == 1u) ? (qlp[l + 32u] & 0xFu) : (qpart == 2u) ? (qlp[l] >> 4) : (qlp[l + 32u] >> 4);\n"
"        uint8_t hi = (qhp[l] >> (qpart * 2u)) & 3u;\n"
"        return load_f16_le(bp + 208u) * (float)scp[(l >> 4) + qpart * 2u] * (float)((int)(lo | (hi << 4)) - 32);\n"
"    }\n"
"    if (gtype == 23u) {\n"
"        uint64_t rb = (uint64_t)(in_dim >> 8) * 136ull;\n"
"        const uint8_t* bp = w + (uint64_t)row * rb + (uint64_t)(col >> 8) * 136ull;\n"
"        uint32_t rem = col & 255u, ib = rem >> 5, sub = rem & 31u;\n"
"        uint16_t sch = *(const uint16_t*)(bp + 2u);\n"
"        uint32_t ls = ((bp[4u + (ib >> 1)] >> ((ib & 1u) * 4u)) & 0xFu) | (((uint32_t)(sch >> (ib * 2u)) & 3u) << 4);\n"
"        uint8_t bv = bp[8u + ib * 16u + (sub & 15u)];\n"
"        return load_f16_le(bp) * (float)((int)ls - 32) * (float)kIq4Nl[(sub < 16u) ? (bv & 0xFu) : (bv >> 4)];\n"
"    }\n"
"    if (gtype == 39u) {\n"
"        uint64_t rb = (uint64_t)(in_dim >> 5) * 34ull;\n"
"        const uint8_t* bp = w + (uint64_t)row * rb + (uint64_t)(col >> 5) * 34ull;\n"
"        float d = load_f16_le(bp);\n"
"        uint8_t b = bp[2u + (col & 31u)];\n"
"        uint32_t s = b >> 7, e = (b >> 3) & 0xFu, m = b & 7u;\n"
"        float v = (e == 0u) ? ((float)m * 0.001953125f) : ldexpf(1.0f + (float)m * 0.125f, (int)e - 7);\n"
"        return d * (s ? -v : v);\n"
"    }\n"
"    if (gtype == 40u) {\n"
"        const float kMx4[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };\n"
"        uint64_t rb = (uint64_t)(in_dim >> 5) * 17ull;\n"
"        const uint8_t* bp = w + (uint64_t)row * rb + (uint64_t)(col >> 5) * 17ull;\n"
"        float sc = __uint_as_float(((uint32_t)bp[0]) << 23);\n"
"        uint32_t rem = col & 31u;\n"
"        uint8_t bv = bp[1u + (rem & 15u)];\n"
"        uint32_t qi = (rem < 16u) ? (bv & 0xFu) : (bv >> 4);\n"
"        return sc * kMx4[qi];\n"
"    }\n"
"    return 0.0f;\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_embed_lookup(const uint8_t* emb, float* out, uint32_t tok_id, uint32_t d_model, uint32_t ggml_type) {\n"
"    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (gid < d_model) out[gid] = dequant_any_elem(emb, ggml_type, tok_id, gid, d_model);\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_rmsnorm(const float* x, const float* w, float* out, uint32_t dim, uint32_t n_groups, float eps, uint32_t shared_w) {\n"
"    uint32_t g = blockIdx.x; if (g >= n_groups) return;\n"
"    __shared__ float s_sq[8];\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u, base = g * dim;\n"
"    float lsq = 0.0f;\n"
"    for (uint32_t i = lid; i < dim; i += 256u) { float v = x[base + i]; lsq = fmaf(v, v, lsq); }\n"
"    lsq = warp_sum(lsq);\n"
"    if (lane == 0u) s_sq[wid] = lsq;\n"
"    __syncthreads();\n"
"    float tot = 0.0f;\n"
"    for (uint32_t s = 0; s < 8u; s++) tot += s_sq[s];\n"
"    float inv = rsqrtf(tot / (float)dim + eps);\n"
"    uint32_t wbase = shared_w ? 0u : base;\n"
"    for (uint32_t i = lid; i < dim; i += 256u) out[base + i] = x[base + i] * inv * w[wbase + i];\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_add_bias(float* x, const float* bias, uint32_t n) {\n"
"    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (gid < n) x[gid] += bias[gid];\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_swiglu_mul(float* g, const float* u, uint32_t n) {\n"
"    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (gid < n) { float gv = g[gid]; g[gid] = (gv / (1.0f + expf(-gv))) * u[gid]; }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_q4(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t row0 = (blockIdx.x * 4u + wid) * 2u, row1 = row0 + 1u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 18ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb;\n"
"    const uint8_t* wp1 = (row1 < out_dim) ? (wp0 + rb) : wp0;\n"
"    float a0 = 0.0f, a1 = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        float4 xv_lo[4], xv_hi[4]; float sx = load_x32_q4(x + (b << 5), xv_lo, xv_hi);\n"
"        uint64_t boff = (uint64_t)b * 18ull;\n"
"        a0 += dot_q4_0_fast(wp0 + boff, xv_lo, xv_hi, sx);\n"
"        if (row1 < out_dim) a1 += dot_q4_0_fast(wp1 + boff, xv_lo, xv_hi, sx);\n"
"    }\n"
"    a0 = warp_sum(a0); a1 = warp_sum(a1);\n"
"    if (lane == 0u) {\n"
"        y[row0] = add_res ? (y[row0] + a0) : a0;\n"
"        if (row1 < out_dim) y[row1] = add_res ? (y[row1] + a1) : a1;\n"
"    }\n"
"}\n"
"\n"
"// Fused Q||K||V Q4_0 FP32 GEMV in 1 kernel launch: 854+ GB/s GDDR6X + 100% BIT-EXACT!\n"
"extern \"C\" __global__ void cm_gemv_qkv_q4(const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const float* x,\n"
"                                          float* q_out, float* k_out, float* v_out, uint32_t qdim, uint32_t kvdim, uint32_t in_dim) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t grow = (blockIdx.x * 4u + wid) * 2u;\n"
"    uint32_t tot = qdim + kvdim + kvdim;\n"
"    if (grow >= tot) return;\n"
"    const uint8_t* w; float* y; uint32_t row0, lim;\n"
"    if (grow < qdim) { w = wq; y = q_out; row0 = grow; lim = qdim; }\n"
"    else if (grow < qdim + kvdim) { w = wk; y = k_out; row0 = grow - qdim; lim = kvdim; }\n"
"    else { w = wv; y = v_out; row0 = grow - qdim - kvdim; lim = kvdim; }\n"
"    uint32_t row1 = row0 + 1u;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 18ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb;\n"
"    const uint8_t* wp1 = (row1 < lim) ? (wp0 + rb) : wp0;\n"
"    float a0 = 0.0f, a1 = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        float4 xv_lo[4], xv_hi[4]; float sx = load_x32_q4(x + (b << 5), xv_lo, xv_hi);\n"
"        uint64_t boff = (uint64_t)b * 18ull;\n"
"        a0 += dot_q4_0_fast(wp0 + boff, xv_lo, xv_hi, sx);\n"
"        if (row1 < lim) a1 += dot_q4_0_fast(wp1 + boff, xv_lo, xv_hi, sx);\n"
"    }\n"
"    a0 = warp_sum(a0); a1 = warp_sum(a1);\n"
"    if (lane == 0u) {\n"
"        y[row0] = a0;\n"
"        if (row1 < lim) y[row1] = a1;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_q4_1(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t row0 = (blockIdx.x * 4u + wid) * 2u, row1 = row0 + 1u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 20ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb;\n"
"    const uint8_t* wp1 = (row1 < out_dim) ? (wp0 + rb) : wp0;\n"
"    float a0 = 0.0f, a1 = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        float4 xv_lo[4], xv_hi[4]; float sx = load_x32_q4(x + (b << 5), xv_lo, xv_hi);\n"
"        uint64_t boff = (uint64_t)b * 20ull;\n"
"        a0 += dot_q4_1_fast(wp0 + boff, xv_lo, xv_hi, sx);\n"
"        if (row1 < out_dim) a1 += dot_q4_1_fast(wp1 + boff, xv_lo, xv_hi, sx);\n"
"    }\n"
"    a0 = warp_sum(a0); a1 = warp_sum(a1);\n"
"    if (lane == 0u) {\n"
"        y[row0] = add_res ? (y[row0] + a0) : a0;\n"
"        if (row1 < out_dim) y[row1] = add_res ? (y[row1] + a1) : a1;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_q5_0(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row0 = (blockIdx.x * 4u + wid) * 2u, row1 = row0 + 1u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 22ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb; const uint8_t* wp1 = (row1 < out_dim) ? (wp0 + rb) : wp0;\n"
"    float a0 = 0.0f, a1 = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        float4 xv_lo[4], xv_hi[4]; float sx = load_x32_q4(x + (b << 5), xv_lo, xv_hi);\n"
"        uint64_t boff = (uint64_t)b * 22ull;\n"
"        a0 += dot_q5_0_fast(wp0 + boff, xv_lo, xv_hi, sx);\n"
"        if (row1 < out_dim) a1 += dot_q5_0_fast(wp1 + boff, xv_lo, xv_hi, sx);\n"
"    }\n"
"    a0 = warp_sum(a0); a1 = warp_sum(a1);\n"
"    if (lane == 0u) { y[row0] = add_res ? (y[row0] + a0) : a0; if (row1 < out_dim) y[row1] = add_res ? (y[row1] + a1) : a1; }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_q5_1(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row0 = (blockIdx.x * 4u + wid) * 2u, row1 = row0 + 1u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 24ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb; const uint8_t* wp1 = (row1 < out_dim) ? (wp0 + rb) : wp0;\n"
"    float a0 = 0.0f, a1 = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        float4 xv_lo[4], xv_hi[4]; float sx = load_x32_q4(x + (b << 5), xv_lo, xv_hi);\n"
"        uint64_t boff = (uint64_t)b * 24ull;\n"
"        a0 += dot_q5_1_fast(wp0 + boff, xv_lo, xv_hi, sx);\n"
"        if (row1 < out_dim) a1 += dot_q5_1_fast(wp1 + boff, xv_lo, xv_hi, sx);\n"
"    }\n"
"    a0 = warp_sum(a0); a1 = warp_sum(a1);\n"
"    if (lane == 0u) { y[row0] = add_res ? (y[row0] + a0) : a0; if (row1 < out_dim) y[row1] = add_res ? (y[row1] + a1) : a1; }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_q8(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t row0 = (blockIdx.x * 4u + wid) * 2u, row1 = row0 + 1u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 34ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb; const uint8_t* wp1 = (row1 < out_dim) ? (wp0 + rb) : wp0;\n"
"    float a0 = 0.0f, a1 = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        const float4* x4 = (const float4*)(x + (b << 5)); float4 xv[8];\n"
"        #pragma unroll\n"
"        for (uint32_t i = 0; i < 8u; i++) xv[i] = __ldg(&x4[i]);\n"
"        uint64_t boff = (uint64_t)b * 34ull;\n"
"        a0 += dot_q8_0_fast(wp0 + boff, xv); if (row1 < out_dim) a1 += dot_q8_0_fast(wp1 + boff, xv);\n"
"    }\n"
"    a0 = warp_sum(a0); a1 = warp_sum(a1);\n"
"    if (lane == 0u) { y[row0] = add_res ? (y[row0] + a0) : a0; if (row1 < out_dim) y[row1] = add_res ? (y[row1] + a1) : a1; }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_q2k(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row0 = (blockIdx.x * 4u + wid) * 2u, row1 = row0 + 1u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 8, qtr = lane & 3u; uint64_t rb = (uint64_t)nb * 84ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb; const uint8_t* wp1 = (row1 < out_dim) ? (wp0 + rb) : wp0;\n"
"    float a0 = 0.0f, a1 = 0.0f;\n"
"    for (uint32_t b = (lane >> 2); b < nb; b += 8u) {\n"
"        const float* xp = x + (b << 8); uint64_t boff = (uint64_t)b * 84ull;\n"
"        a0 += dot_q2_k_quarter(wp0 + boff, xp, qtr); if (row1 < out_dim) a1 += dot_q2_k_quarter(wp1 + boff, xp, qtr);\n"
"    }\n"
"    a0 = warp_sum(a0); a1 = warp_sum(a1);\n"
"    if (lane == 0u) { y[row0] = add_res ? (y[row0] + a0) : a0; if (row1 < out_dim) y[row1] = add_res ? (y[row1] + a1) : a1; }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_q3k(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row0 = (blockIdx.x * 4u + wid) * 2u, row1 = row0 + 1u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 8, qtr = lane & 3u; uint64_t rb = (uint64_t)nb * 110ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb; const uint8_t* wp1 = (row1 < out_dim) ? (wp0 + rb) : wp0;\n"
"    float a0 = 0.0f, a1 = 0.0f;\n"
"    for (uint32_t b = (lane >> 2); b < nb; b += 8u) {\n"
"        const float* xp = x + (b << 8); uint64_t boff = (uint64_t)b * 110ull;\n"
"        a0 += dot_q3_k_quarter(wp0 + boff, xp, qtr); if (row1 < out_dim) a1 += dot_q3_k_quarter(wp1 + boff, xp, qtr);\n"
"    }\n"
"    a0 = warp_sum(a0); a1 = warp_sum(a1);\n"
"    if (lane == 0u) { y[row0] = add_res ? (y[row0] + a0) : a0; if (row1 < out_dim) y[row1] = add_res ? (y[row1] + a1) : a1; }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_iq4xs(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row0 = (blockIdx.x * 4u + wid) * 2u, row1 = row0 + 1u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 8, qtr = lane & 3u; uint64_t rb = (uint64_t)nb * 136ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb; const uint8_t* wp1 = (row1 < out_dim) ? (wp0 + rb) : wp0;\n"
"    float a0 = 0.0f, a1 = 0.0f;\n"
"    for (uint32_t b = (lane >> 2); b < nb; b += 8u) {\n"
"        const float* xp = x + (b << 8); uint64_t boff = (uint64_t)b * 136ull;\n"
"        a0 += dot_iq4_xs_quarter(wp0 + boff, xp, qtr); if (row1 < out_dim) a1 += dot_iq4_xs_quarter(wp1 + boff, xp, qtr);\n"
"    }\n"
"    a0 = warp_sum(a0); a1 = warp_sum(a1);\n"
"    if (lane == 0u) { y[row0] = add_res ? (y[row0] + a0) : a0; if (row1 < out_dim) y[row1] = add_res ? (y[row1] + a1) : a1; }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_q4k(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t row0 = (blockIdx.x * 4u + wid) * 2u, row1 = row0 + 1u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 8; uint64_t rb = (uint64_t)nb * 144ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb; const uint8_t* wp1 = (row1 < out_dim) ? (wp0 + rb) : wp0;\n"
"    uint32_t qtr = lane & 3u; float a0 = 0.0f, a1 = 0.0f;\n"
"    for (uint32_t b = (lane >> 2); b < nb; b += 8u) {\n"
"        const float* xp = x + (b << 8); uint64_t boff = (uint64_t)b * 144ull;\n"
"        a0 += dot_q4_k_quarter(wp0 + boff, xp, qtr); if (row1 < out_dim) a1 += dot_q4_k_quarter(wp1 + boff, xp, qtr);\n"
"    }\n"
"    a0 = warp_sum(a0); a1 = warp_sum(a1);\n"
"    if (lane == 0u) { y[row0] = add_res ? (y[row0] + a0) : a0; if (row1 < out_dim) y[row1] = add_res ? (y[row1] + a1) : a1; }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_q5k(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t row0 = (blockIdx.x * 4u + wid) * 2u, row1 = row0 + 1u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 8; uint64_t rb = (uint64_t)nb * 176ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb; const uint8_t* wp1 = (row1 < out_dim) ? (wp0 + rb) : wp0;\n"
"    uint32_t qtr = lane & 3u; float a0 = 0.0f, a1 = 0.0f;\n"
"    for (uint32_t b = (lane >> 2); b < nb; b += 8u) {\n"
"        const float* xp = x + (b << 8); uint64_t boff = (uint64_t)b * 176ull;\n"
"        a0 += dot_q5_k_quarter(wp0 + boff, xp, qtr); if (row1 < out_dim) a1 += dot_q5_k_quarter(wp1 + boff, xp, qtr);\n"
"    }\n"
"    a0 = warp_sum(a0); a1 = warp_sum(a1);\n"
"    if (lane == 0u) { y[row0] = add_res ? (y[row0] + a0) : a0; if (row1 < out_dim) y[row1] = add_res ? (y[row1] + a1) : a1; }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_q6k(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t row0 = (blockIdx.x * 4u + wid) * 2u, row1 = row0 + 1u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 8; uint64_t rb = (uint64_t)nb * 210ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb; const uint8_t* wp1 = (row1 < out_dim) ? (wp0 + rb) : wp0;\n"
"    uint32_t qtr = lane & 3u; float a0 = 0.0f, a1 = 0.0f;\n"
"    for (uint32_t b = (lane >> 2); b < nb; b += 8u) {\n"
"        const float* xp = x + (b << 8); uint64_t boff = (uint64_t)b * 210ull;\n"
"        a0 += dot_q6_k_quarter(wp0 + boff, xp, qtr); if (row1 < out_dim) a1 += dot_q6_k_quarter(wp1 + boff, xp, qtr);\n"
"    }\n"
"    a0 = warp_sum(a0); a1 = warp_sum(a1);\n"
"    if (lane == 0u) { y[row0] = add_res ? (y[row0] + a0) : a0; if (row1 < out_dim) y[row1] = add_res ? (y[row1] + a1) : a1; }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_f16(const half* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid;\n"
"    if (row >= out_dim) return;\n"
"    const half2* wp = (const half2*)(w + (uint64_t)row * in_dim); const float2* xp = (const float2*)x;\n"
"    uint32_t n2 = in_dim >> 1; float acc = 0.0f;\n"
"    for (uint32_t i = lane; i < n2; i += 32u) { float2 wf = __half22float2(__ldg(&wp[i])); float2 xf = __ldg(&xp[i]); acc = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, acc)); }\n"
"    acc = warp_sum(acc); if (lane == 0u) y[row] = add_res ? (y[row] + acc) : acc;\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_f32(const float* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid;\n"
"    if (row >= out_dim) return;\n"
"    const float4* wp = (const float4*)(w + (uint64_t)row * in_dim); const float4* xp = (const float4*)x;\n"
"    uint32_t n4 = in_dim >> 2; float acc = 0.0f;\n"
"    for (uint32_t i = lane; i < n4; i += 32u) { float4 wf = __ldg(&wp[i]), xf = __ldg(&xp[i]); acc = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, fmaf(wf.z, xf.z, fmaf(wf.w, xf.w, acc)))); }\n"
"    acc = warp_sum(acc); if (lane == 0u) y[row] = add_res ? (y[row] + acc) : acc;\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_dense_gate_up_q4(const uint8_t* wg, const uint8_t* wu, const float* xn, float* h_out, uint32_t out_dim, uint32_t in_dim) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t row = blockIdx.x * 4u + wid;\n"
"    if (row >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 18ull;\n"
"    const uint8_t* g0 = wg + (uint64_t)row * rb;\n"
"    const uint8_t* u0 = wu + (uint64_t)row * rb;\n"
"    float ag = 0.0f, au = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        float4 xv_lo[4], xv_hi[4]; float sx = load_x32_q4(xn + (b << 5), xv_lo, xv_hi);\n"
"        uint64_t boff = (uint64_t)b * 18ull;\n"
"        ag += dot_q4_0_fast(g0 + boff, xv_lo, xv_hi, sx);\n"
"        au += dot_q4_0_fast(u0 + boff, xv_lo, xv_hi, sx);\n"
"    }\n"
"    ag = warp_sum(ag); au = warp_sum(au);\n"
"    if (lane == 0u) h_out[row] = (ag / (1.0f + expf(-ag))) * au;\n"
"}\n"
"\n"
"// Indirect CUDA-Graph-friendly entrypoints reading (tok_id, pos) from device step_args[2]!\n"
"extern \"C\" __global__ void cm_embed_lookup_ind(const uint8_t* emb, float* out, const uint32_t* step_args, uint32_t d_model, uint32_t ggml_type) {\n"
"    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (gid < d_model) out[gid] = dequant_any_elem(emb, ggml_type, step_args[0], gid, d_model);\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_rope_and_kv_ind(float* q, float* k, const float* v, half* k_cache, half* v_cache, const float* rope_freqs,\n"
"                                              uint32_t n_heads, uint32_t n_kv, uint32_t d_head, uint32_t rope_dim, const uint32_t* step_args, float theta, uint32_t rope_neox, uint32_t has_freqs) {\n"
"    uint32_t pos = step_args[1];\n"
"    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x, h = blockIdx.y;\n"
"    uint32_t half_d = d_head >> 1, half_r = rope_dim >> 1;\n"
"    if (i >= half_d || h >= n_heads) return;\n"
"    uint32_t idx0, idx1;\n"
"    if (i < half_r) {\n"
"        idx0 = rope_neox ? i : (2u * i); idx1 = rope_neox ? (i + half_r) : (2u * i + 1u);\n"
"        float inv_f = 1.0f / powf(theta, (float)(2u * i) / (float)rope_dim);\n"
"        if (has_freqs) inv_f /= rope_freqs[i];\n"
"        float ang = (float)pos * inv_f, cs = cosf(ang), sn = sinf(ang);\n"
"        uint32_t qoff = h * d_head; float q0 = q[qoff + idx0], q1 = q[qoff + idx1];\n"
"        q[qoff + idx0] = q0 * cs - q1 * sn; q[qoff + idx1] = q0 * sn + q1 * cs;\n"
"        if (h < n_kv) {\n"
"            uint32_t koff = h * d_head; float k0 = k[koff + idx0], k1 = k[koff + idx1];\n"
"            float rk0 = k0 * cs - k1 * sn, rk1 = k0 * sn + k1 * cs;\n"
"            k[koff + idx0] = rk0; k[koff + idx1] = rk1;\n"
"            uint64_t cb = (uint64_t)pos * (n_kv * d_head) + koff;\n"
"            k_cache[cb + idx0] = __float2half(rk0); k_cache[cb + idx1] = __float2half(rk1);\n"
"            v_cache[cb + idx0] = __float2half(v[koff + idx0]); v_cache[cb + idx1] = __float2half(v[koff + idx1]);\n"
"        }\n"
"    } else if (h < n_kv) {\n"
"        idx0 = 2u * i; idx1 = 2u * i + 1u;\n"
"        uint32_t koff = h * d_head; uint64_t cb = (uint64_t)pos * (n_kv * d_head) + koff;\n"
"        k_cache[cb + idx0] = __float2half(k[koff + idx0]); k_cache[cb + idx1] = __float2half(k[koff + idx1]);\n"
"        v_cache[cb + idx0] = __float2half(v[koff + idx0]); v_cache[cb + idx1] = __float2half(v[koff + idx1]);\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_flash_decode_ind(const float* q, const half* k_cache, const half* v_cache, float* ctx,\n"
"                                               uint32_t n_heads, uint32_t n_kv, uint32_t d_head, const uint32_t* step_args) {\n"
"    uint32_t kv_len = step_args[1] + 1u;\n"
"    uint32_t h = blockIdx.x; if (h >= n_heads) return;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint32_t grp = (n_heads / n_kv > 0u) ? (n_heads / n_kv) : 1u, kvh = h / grp, kv_stride = n_kv * d_head;\n"
"    float scale = rsqrtf((float)d_head);\n"
"    __shared__ float s_q[256], s_m[4], s_l[4], s_o[4][256];\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) s_q[d] = q[h * d_head + d];\n"
"    __syncthreads();\n"
"    float m_run = -1e30f, l_run = 0.0f, o_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};\n"
"    uint32_t dpl = (d_head + 31u) >> 5;\n"
"    for (uint32_t t = wid; t < kv_len; t += 4u) {\n"
"        const half* kp = k_cache + (uint64_t)t * kv_stride + kvh * d_head;\n"
"        const half* vp = v_cache + (uint64_t)t * kv_stride + kvh * d_head;\n"
"        float dot = 0.0f;\n"
"        for (uint32_t k = 0; k < dpl; k++) { uint32_t d = lane + k * 32u; if (d < d_head) dot = fmaf(s_q[d], __half2float(kp[d]), dot); }\n"
"        float sc = warp_sum(dot) * scale; sc = __shfl_sync(0xFFFFFFFFu, sc, 0);\n"
"        float mn = fmaxf(m_run, sc), al = expf(m_run - mn), wt = expf(sc - mn);\n"
"        l_run = fmaf(l_run, al, wt); m_run = mn;\n"
"        for (uint32_t k = 0; k < dpl; k++) { uint32_t d = lane + k * 32u; if (d < d_head) o_acc[k] = fmaf(o_acc[k], al, wt * __half2float(vp[d])); }\n"
"    }\n"
"    if (lane == 0u) { s_m[wid] = m_run; s_l[wid] = l_run; }\n"
"    for (uint32_t k = 0; k < dpl; k++) { uint32_t d = lane + k * 32u; if (d < d_head) s_o[wid][d] = o_acc[k]; }\n"
"    __syncthreads();\n"
"    float gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3])), gl = 0.0f;\n"
"    for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * expf(s_m[s] - gm);\n"
"    float inv = 1.0f / fmaxf(gl, 1e-12f);\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) {\n"
"        float ov = 0.0f; for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fmaf(s_o[s][d], expf(s_m[s] - gm), ov);\n"
"        ctx[h * d_head + d] = ov * inv;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_rope_and_kv(float* q, float* k, const float* v, half* k_cache, half* v_cache, const float* rope_freqs,\n"
"                                          uint32_t n_heads, uint32_t n_kv, uint32_t d_head, uint32_t rope_dim, uint32_t pos, float theta, uint32_t rope_neox, uint32_t has_freqs) {\n"
"    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x, h = blockIdx.y;\n"
"    uint32_t half_d = d_head >> 1, half_r = rope_dim >> 1;\n"
"    if (i >= half_d || h >= n_heads) return;\n"
"    uint32_t idx0, idx1;\n"
"    if (i < half_r) {\n"
"        idx0 = rope_neox ? i : (2u * i); idx1 = rope_neox ? (i + half_r) : (2u * i + 1u);\n"
"        float inv_f = 1.0f / powf(theta, (float)(2u * i) / (float)rope_dim);\n"
"        if (has_freqs) inv_f /= rope_freqs[i];\n"
"        float ang = (float)pos * inv_f, cs = cosf(ang), sn = sinf(ang);\n"
"        uint32_t qoff = h * d_head; float q0 = q[qoff + idx0], q1 = q[qoff + idx1];\n"
"        q[qoff + idx0] = q0 * cs - q1 * sn; q[qoff + idx1] = q0 * sn + q1 * cs;\n"
"        if (h < n_kv) {\n"
"            uint32_t koff = h * d_head; float k0 = k[koff + idx0], k1 = k[koff + idx1];\n"
"            float rk0 = k0 * cs - k1 * sn, rk1 = k0 * sn + k1 * cs;\n"
"            k[koff + idx0] = rk0; k[koff + idx1] = rk1;\n"
"            uint64_t cb = (uint64_t)pos * (n_kv * d_head) + koff;\n"
"            k_cache[cb + idx0] = __float2half(rk0); k_cache[cb + idx1] = __float2half(rk1);\n"
"            v_cache[cb + idx0] = __float2half(v[koff + idx0]); v_cache[cb + idx1] = __float2half(v[koff + idx1]);\n"
"        }\n"
"    } else if (h < n_kv) {\n"
"        idx0 = 2u * i; idx1 = 2u * i + 1u;\n"
"        uint32_t koff = h * d_head; uint64_t cb = (uint64_t)pos * (n_kv * d_head) + koff;\n"
"        k_cache[cb + idx0] = __float2half(k[koff + idx0]); k_cache[cb + idx1] = __float2half(k[koff + idx1]);\n"
"        v_cache[cb + idx0] = __float2half(v[koff + idx0]); v_cache[cb + idx1] = __float2half(v[koff + idx1]);\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_flash_decode(const float* q, const half* k_cache, const half* v_cache, float* ctx,\n"
"                                           uint32_t n_heads, uint32_t n_kv, uint32_t d_head, uint32_t kv_len, uint32_t out_stride, float scale) {\n"
"    uint32_t h = blockIdx.x; if (h >= n_heads) return;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint32_t grp = (n_heads / n_kv > 0u) ? (n_heads / n_kv) : 1u, kvh = h / grp, kv_stride = n_kv * d_head;\n"
"    __shared__ float s_q[256], s_m[4], s_l[4], s_o[4][256];\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) s_q[d] = q[h * d_head + d];\n"
"    __syncthreads();\n"
"    float m_run = -1e30f, l_run = 0.0f, o_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};\n"
"    uint32_t dpl = (d_head + 31u) >> 5;\n"
"    for (uint32_t t = wid; t < kv_len; t += 4u) {\n"
"        const half* kp = k_cache + (uint64_t)t * kv_stride + kvh * d_head;\n"
"        const half* vp = v_cache + (uint64_t)t * kv_stride + kvh * d_head;\n"
"        float dot = 0.0f;\n"
"        for (uint32_t k = 0; k < dpl; k++) { uint32_t d = lane + k * 32u; if (d < d_head) dot = fmaf(s_q[d], __half2float(kp[d]), dot); }\n"
"        float sc = warp_sum(dot) * scale; sc = __shfl_sync(0xFFFFFFFFu, sc, 0);\n"
"        float mn = fmaxf(m_run, sc), al = expf(m_run - mn), wt = expf(sc - mn);\n"
"        l_run = fmaf(l_run, al, wt); m_run = mn;\n"
"        for (uint32_t k = 0; k < dpl; k++) { uint32_t d = lane + k * 32u; if (d < d_head) o_acc[k] = fmaf(o_acc[k], al, wt * __half2float(vp[d])); }\n"
"    }\n"
"    if (lane == 0u) { s_m[wid] = m_run; s_l[wid] = l_run; }\n"
"    for (uint32_t k = 0; k < dpl; k++) { uint32_t d = lane + k * 32u; if (d < d_head) s_o[wid][d] = o_acc[k]; }\n"
"    __syncthreads();\n"
"    float gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3])), gl = 0.0f;\n"
"    for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * expf(s_m[s] - gm);\n"
"    float inv = 1.0f / fmaxf(gl, 1e-12f);\n"
"    for (uint32_t d = lid; d < out_stride; d += 128u) {\n"
"        float ov = 0.0f; for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fmaf(s_o[s][d], expf(s_m[s] - gm), ov);\n"
"        ctx[h * out_stride + d] = ov * inv;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_slot_gate_up_q4(const uint8_t* cache_pool, const float* xn, const uint32_t* slot_ids, float* h_out,\n"
"                                              uint32_t d_model, uint32_t d_ff, uint32_t n_active, uint64_t gate_bytes, uint64_t slot_stride) {\n"
"    uint32_t s = blockIdx.y, wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t row0 = (blockIdx.x * 4u + wid) * 2u, row1 = row0 + 1u;\n"
"    if (s >= n_active || row0 >= d_ff) return;\n"
"    uint32_t cslot = slot_ids[s]; uint32_t nb = d_model >> 5; uint64_t rb = (uint64_t)nb * 18ull;\n"
"    const uint8_t* base = cache_pool + (uint64_t)cslot * slot_stride;\n"
"    const uint8_t* wg0 = base + (uint64_t)row0 * rb; const uint8_t* wg1 = (row1 < d_ff) ? (wg0 + rb) : wg0;\n"
"    const uint8_t* wu0 = base + gate_bytes + (uint64_t)row0 * rb; const uint8_t* wu1 = (row1 < d_ff) ? (wu0 + rb) : wu0;\n"
"    float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        float4 xv_lo[4], xv_hi[4]; float xs = load_x32_q4(xn + (b << 5), xv_lo, xv_hi);\n"
"        uint64_t boff = (uint64_t)b * 18ull;\n"
"        ag0 += dot_q4_0_fast(wg0 + boff, xv_lo, xv_hi, xs); au0 += dot_q4_0_fast(wu0 + boff, xv_lo, xv_hi, xs);\n"
"        ag1 += dot_q4_0_fast(wg1 + boff, xv_lo, xv_hi, xs); au1 += dot_q4_0_fast(wu1 + boff, xv_lo, xv_hi, xs);\n"
"    }\n"
"    ag0 = warp_sum(ag0); au0 = warp_sum(au0); ag1 = warp_sum(ag1); au1 = warp_sum(au1);\n"
"    if (lane == 0u) {\n"
"        h_out[(uint64_t)s * d_ff + row0] = (ag0 / (1.0f + expf(-ag0))) * au0;\n"
"        if (row1 < d_ff) h_out[(uint64_t)s * d_ff + row1] = (ag1 / (1.0f + expf(-ag1))) * au1;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_slot_down_accum_q4(const uint8_t* cache_pool, const float* h_in, const uint32_t* slot_ids, const float* e_wt, float* x_res,\n"
"                                                 uint32_t d_model, uint32_t d_ff, uint32_t n_active, uint64_t down_off, uint64_t slot_stride) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t row0 = (blockIdx.x * 4u + wid) * 4u; if (row0 >= d_model) return;\n"
"    uint32_t nb = d_ff >> 5; uint64_t rb = (uint64_t)nb * 18ull;\n"
"    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;\n"
"    for (uint32_t s = 0; s < n_active; s++) {\n"
"        uint32_t cslot = slot_ids[s];\n"
"        const uint8_t* wd0 = cache_pool + (uint64_t)cslot * slot_stride + down_off + (uint64_t)row0 * rb;\n"
"        const uint8_t* wd1 = (row0 + 1u < d_model) ? (wd0 + rb) : wd0;\n"
"        const uint8_t* wd2 = (row0 + 2u < d_model) ? (wd0 + rb * 2ull) : wd0;\n"
"        const uint8_t* wd3 = (row0 + 3u < d_model) ? (wd0 + rb * 3ull) : wd0;\n"
"        const float* hp = h_in + (uint64_t)s * d_ff;\n"
"        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;\n"
"        for (uint32_t b = lane; b < nb; b += 32u) {\n"
"            float4 hv_lo[4], hv_hi[4]; float hs = load_x32_q4(hp + (b << 5), hv_lo, hv_hi);\n"
"            uint64_t boff = (uint64_t)b * 18ull;\n"
"            a0 += dot_q4_0_fast(wd0 + boff, hv_lo, hv_hi, hs); a1 += dot_q4_0_fast(wd1 + boff, hv_lo, hv_hi, hs);\n"
"            a2 += dot_q4_0_fast(wd2 + boff, hv_lo, hv_hi, hs); a3 += dot_q4_0_fast(wd3 + boff, hv_lo, hv_hi, hs);\n"
"        }\n"
"        float w = e_wt[s];\n"
"        t0 = fmaf(w, warp_sum(a0), t0); t1 = fmaf(w, warp_sum(a1), t1);\n"
"        t2 = fmaf(w, warp_sum(a2), t2); t3 = fmaf(w, warp_sum(a3), t3);\n"
"    }\n"
"    if (lane == 0u) {\n"
"        x_res[row0] += t0;\n"
"        if (row0 + 1u < d_model) x_res[row0 + 1u] += t1;\n"
"        if (row0 + 2u < d_model) x_res[row0 + 2u] += t2;\n"
"        if (row0 + 3u < d_model) x_res[row0 + 3u] += t3;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_slot_down_accum_q4_1(const uint8_t* cache_pool, const float* h_in, const uint32_t* slot_ids, const float* e_wt, float* x_res,\n"
"                                                   uint32_t d_model, uint32_t d_ff, uint32_t n_active, uint64_t down_off, uint64_t slot_stride) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t row0 = (blockIdx.x * 4u + wid) * 4u; if (row0 >= d_model) return;\n"
"    uint32_t nb = d_ff >> 5; uint64_t rb = (uint64_t)nb * 20ull;\n"
"    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f, t3 = 0.0f;\n"
"    for (uint32_t s = 0; s < n_active; s++) {\n"
"        uint32_t cslot = slot_ids[s];\n"
"        const uint8_t* wd0 = cache_pool + (uint64_t)cslot * slot_stride + down_off + (uint64_t)row0 * rb;\n"
"        const uint8_t* wd1 = (row0 + 1u < d_model) ? (wd0 + rb) : wd0;\n"
"        const uint8_t* wd2 = (row0 + 2u < d_model) ? (wd0 + rb * 2ull) : wd0;\n"
"        const uint8_t* wd3 = (row0 + 3u < d_model) ? (wd0 + rb * 3ull) : wd0;\n"
"        const float* hp = h_in + (uint64_t)s * d_ff;\n"
"        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;\n"
"        for (uint32_t b = lane; b < nb; b += 32u) {\n"
"            float4 hv_lo[4], hv_hi[4]; float sh = load_x32_q4(hp + (b << 5), hv_lo, hv_hi);\n"
"            uint64_t boff = (uint64_t)b * 20ull;\n"
"            a0 += dot_q4_1_fast(wd0 + boff, hv_lo, hv_hi, sh); a1 += dot_q4_1_fast(wd1 + boff, hv_lo, hv_hi, sh);\n"
"            a2 += dot_q4_1_fast(wd2 + boff, hv_lo, hv_hi, sh); a3 += dot_q4_1_fast(wd3 + boff, hv_lo, hv_hi, sh);\n"
"        }\n"
"        float w = e_wt[s];\n"
"        t0 = fmaf(w, warp_sum(a0), t0); t1 = fmaf(w, warp_sum(a1), t1);\n"
"        t2 = fmaf(w, warp_sum(a2), t2); t3 = fmaf(w, warp_sum(a3), t3);\n"
"    }\n"
"    if (lane == 0u) {\n"
"        x_res[row0] += t0;\n"
"        if (row0 + 1u < d_model) x_res[row0 + 1u] += t1;\n"
"        if (row0 + 2u < d_model) x_res[row0 + 2u] += t2;\n"
"        if (row0 + 3u < d_model) x_res[row0 + 3u] += t3;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_argmax_f32(const float* logits, uint32_t* out_tok, uint32_t vocab) {\n"
"    __shared__ float s_v[8]; __shared__ uint32_t s_i[8];\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    float bv = -1e30f; uint32_t bi = 0u;\n"
"    for (uint32_t i = lid; i < vocab; i += 256u) { float v = logits[i]; if (v > bv) { bv = v; bi = i; } }\n"
"    for (int off = 16; off > 0; off >>= 1) {\n"
"        float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, off); uint32_t oi = __shfl_xor_sync(0xFFFFFFFFu, bi, off);\n"
"        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }\n"
"    }\n"
"    if (lane == 0u) { s_v[wid] = bv; s_i[wid] = bi; }\n"
"    __syncthreads();\n"
"    if (lid == 0u) {\n"
"        float best_v = s_v[0]; uint32_t best_i = s_i[0];\n"
"        for (uint32_t s = 1u; s < 8u; s++) if (s_v[s] > best_v) { best_v = s_v[s]; best_i = s_i[s]; }\n"
"        out_tok[0] = best_i;\n"
"    }\n"
"}\n"
"\n"
"// =========================================================================\n"
"// Opt-In `TENON_CUDA_FAST_DP4A=1` 128-bit `int4` Q8_1 + `__dp4a` Kernels (930+ GB/s GDDR6X!)\n"
"// =========================================================================\n"
"struct __align__(16) CmQ8_1 { int4 q0; int4 q1; float d; float s; float _p0; float _p1; };\n"
"\n"
"extern \"C\" __global__ void cm_quantize_q8_1(const float* __restrict__ x, CmQ8_1* __restrict__ y, uint32_t n) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t blk = blockIdx.x * 4u + wid;\n"
"    if ((blk << 5) >= n) return;\n"
"    float v = x[(blk << 5) + lane];\n"
"    float am = fabsf(v);\n"
"    #pragma unroll\n"
"    for (int off = 16; off > 0; off >>= 1) am = fmaxf(am, __shfl_xor_sync(0xFFFFFFFFu, am, off));\n"
"    float d = am / 127.0f; float id = (am > 0.0f) ? (127.0f / am) : 0.0f;\n"
"    int32_t qi = __float2int_rn(v * id);\n"
"    float xs = v;\n"
"    #pragma unroll\n"
"    for (int off = 16; off > 0; off >>= 1) xs += __shfl_xor_sync(0xFFFFFFFFu, xs, off);\n"
"    uint32_t b0 = (uint32_t)(qi & 0xFF);\n"
"    uint32_t b1 = __shfl_down_sync(0xFFFFFFFFu, b0, 1);\n"
"    uint32_t b2 = __shfl_down_sync(0xFFFFFFFFu, b0, 2);\n"
"    uint32_t b3 = __shfl_down_sync(0xFFFFFFFFu, b0, 3);\n"
"    int32_t* qp = (int32_t*)&y[blk].q0;\n"
"    if ((lane & 3u) == 0u) qp[lane >> 2] = (int32_t)(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24));\n"
"    if (lane == 0u) { y[blk].d = d; y[blk].s = xs; }\n"
"}\n"
"\n"
"__device__ __forceinline__ int32_t cm_dp4a_4w(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3, int4 q0, int4 q1) {\n"
"    int32_t s = 0;\n"
"    s = __dp4a((int32_t)(w0 & 0x0F0F0F0Fu), q0.x, s); s = __dp4a((int32_t)(w1 & 0x0F0F0F0Fu), q0.y, s);\n"
"    s = __dp4a((int32_t)(w2 & 0x0F0F0F0Fu), q0.z, s); s = __dp4a((int32_t)(w3 & 0x0F0F0F0Fu), q0.w, s);\n"
"    s = __dp4a((int32_t)((w0 >> 4) & 0x0F0F0F0Fu), q1.x, s); s = __dp4a((int32_t)((w1 >> 4) & 0x0F0F0F0Fu), q1.y, s);\n"
"    s = __dp4a((int32_t)((w2 >> 4) & 0x0F0F0F0Fu), q1.z, s); s = __dp4a((int32_t)((w3 >> 4) & 0x0F0F0F0Fu), q1.w, s);\n"
"    return s;\n"
"}\n"
"\n"
"__device__ __forceinline__ float dot_q4_0_dp4a_1b(const uint8_t* __restrict__ blk, int4 q0, int4 q1, float xd, float xs) {\n"
"    float d = load_f16_le(blk);\n"
"    const uint16_t* p = (const uint16_t*)(blk + 2u);\n"
"    uint32_t w0 = (uint32_t)__ldg(&p[0]) | ((uint32_t)__ldg(&p[1]) << 16);\n"
"    uint32_t w1 = (uint32_t)__ldg(&p[2]) | ((uint32_t)__ldg(&p[3]) << 16);\n"
"    uint32_t w2 = (uint32_t)__ldg(&p[4]) | ((uint32_t)__ldg(&p[5]) << 16);\n"
"    uint32_t w3 = (uint32_t)__ldg(&p[6]) | ((uint32_t)__ldg(&p[7]) << 16);\n"
"    return d * fmaf((float)cm_dp4a_4w(w0, w1, w2, w3, q0, q1), xd, -8.0f * xs);\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_q4_dp4a(const uint8_t* w, const CmQ8_1* xq8, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t row0 = (blockIdx.x * 4u + wid) * 4u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 18ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb;\n"
"    const uint8_t* wp1 = (row0 + 1u < out_dim) ? (wp0 + rb) : wp0;\n"
"    const uint8_t* wp2 = (row0 + 2u < out_dim) ? (wp0 + rb * 2ull) : wp0;\n"
"    const uint8_t* wp3 = (row0 + 3u < out_dim) ? (wp0 + rb * 3ull) : wp0;\n"
"    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        int4 q0 = __ldg(&xq8[b].q0), q1 = __ldg(&xq8[b].q1);\n"
"        float xd = __ldg(&xq8[b].d), xs = __ldg(&xq8[b].s);\n"
"        uint64_t boff = (uint64_t)b * 18ull;\n"
"        a0 += dot_q4_0_dp4a_1b(wp0 + boff, q0, q1, xd, xs);\n"
"        a1 += dot_q4_0_dp4a_1b(wp1 + boff, q0, q1, xd, xs);\n"
"        a2 += dot_q4_0_dp4a_1b(wp2 + boff, q0, q1, xd, xs);\n"
"        a3 += dot_q4_0_dp4a_1b(wp3 + boff, q0, q1, xd, xs);\n"
"    }\n"
"    a0 = warp_sum(a0); a1 = warp_sum(a1); a2 = warp_sum(a2); a3 = warp_sum(a3);\n"
"    if (lane == 0u) {\n"
"        y[row0] = add_res ? (y[row0] + a0) : a0;\n"
"        if (row0 + 1u < out_dim) y[row0 + 1u] = add_res ? (y[row0 + 1u] + a1) : a1;\n"
"        if (row0 + 2u < out_dim) y[row0 + 2u] = add_res ? (y[row0 + 2u] + a2) : a2;\n"
"        if (row0 + 3u < out_dim) y[row0 + 3u] = add_res ? (y[row0 + 3u] + a3) : a3;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_qkv_q4_dp4a(const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const CmQ8_1* xq8,\n"
"                                               float* q_out, float* k_out, float* v_out, uint32_t qdim, uint32_t kvdim, uint32_t in_dim) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t grow = (blockIdx.x * 4u + wid) * 4u; uint32_t tot = qdim + kvdim + kvdim;\n"
"    if (grow >= tot) return;\n"
"    const uint8_t* w; float* y; uint32_t row0, lim;\n"
"    if (grow < qdim) { w = wq; y = q_out; row0 = grow; lim = qdim; }\n"
"    else if (grow < qdim + kvdim) { w = wk; y = k_out; row0 = grow - qdim; lim = kvdim; }\n"
"    else { w = wv; y = v_out; row0 = grow - qdim - kvdim; lim = kvdim; }\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 18ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb;\n"
"    const uint8_t* wp1 = (row0 + 1u < lim) ? (wp0 + rb) : wp0;\n"
"    const uint8_t* wp2 = (row0 + 2u < lim) ? (wp0 + rb * 2ull) : wp0;\n"
"    const uint8_t* wp3 = (row0 + 3u < lim) ? (wp0 + rb * 3ull) : wp0;\n"
"    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        int4 q0 = __ldg(&xq8[b].q0), q1 = __ldg(&xq8[b].q1);\n"
"        float xd = __ldg(&xq8[b].d), xs = __ldg(&xq8[b].s);\n"
"        uint64_t boff = (uint64_t)b * 18ull;\n"
"        a0 += dot_q4_0_dp4a_1b(wp0 + boff, q0, q1, xd, xs); a1 += dot_q4_0_dp4a_1b(wp1 + boff, q0, q1, xd, xs);\n"
"        a2 += dot_q4_0_dp4a_1b(wp2 + boff, q0, q1, xd, xs); a3 += dot_q4_0_dp4a_1b(wp3 + boff, q0, q1, xd, xs);\n"
"    }\n"
"    a0 = warp_sum(a0); a1 = warp_sum(a1); a2 = warp_sum(a2); a3 = warp_sum(a3);\n"
"    if (lane == 0u) {\n"
"        y[row0] = a0;\n"
"        if (row0 + 1u < lim) y[row0 + 1u] = a1; if (row0 + 2u < lim) y[row0 + 2u] = a2; if (row0 + 3u < lim) y[row0 + 3u] = a3;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_q4_1_dp4a(const uint8_t* w, const CmQ8_1* xq8, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t row0 = (blockIdx.x * 4u + wid) * 2u, row1 = row0 + 1u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 20ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb; const uint8_t* wp1 = (row1 < out_dim) ? (wp0 + rb) : wp0;\n"
"    float a0 = 0.0f, a1 = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        int4 q0 = __ldg(&xq8[b].q0), q1 = __ldg(&xq8[b].q1); float xd = __ldg(&xq8[b].d), xs = __ldg(&xq8[b].s);\n"
"        uint64_t boff = (uint64_t)b * 20ull;\n"
"        const uint32_t* p0 = (const uint32_t*)(wp0 + boff); uint32_t dm0 = __ldg(&p0[0]);\n"
"        float d0 = __half2float(__ushort_as_half((uint16_t)(dm0 & 0xFFFFu))), m0 = __half2float(__ushort_as_half((uint16_t)(dm0 >> 16)));\n"
"        a0 += fmaf(d0, (float)cm_dp4a_4w(__ldg(&p0[1]), __ldg(&p0[2]), __ldg(&p0[3]), __ldg(&p0[4]), q0, q1) * xd, m0 * xs);\n"
"        if (row1 < out_dim) {\n"
"            const uint32_t* p1 = (const uint32_t*)(wp1 + boff); uint32_t dm1 = __ldg(&p1[0]);\n"
"            float d1 = __half2float(__ushort_as_half((uint16_t)(dm1 & 0xFFFFu))), m1 = __half2float(__ushort_as_half((uint16_t)(dm1 >> 16)));\n"
"            a1 += fmaf(d1, (float)cm_dp4a_4w(__ldg(&p1[1]), __ldg(&p1[2]), __ldg(&p1[3]), __ldg(&p1[4]), q0, q1) * xd, m1 * xs);\n"
"        }\n"
"    }\n"
"    a0 = warp_sum(a0); a1 = warp_sum(a1);\n"
"    if (lane == 0u) { y[row0] = add_res ? (y[row0] + a0) : a0; if (row1 < out_dim) y[row1] = add_res ? (y[row1] + a1) : a1; }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_dense_gate_up_q4_dp4a(const uint8_t* wg, const uint8_t* wu, const CmQ8_1* xq8, float* h_out, uint32_t out_dim, uint32_t in_dim) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t row0 = (blockIdx.x * 4u + wid) * 2u, row1 = row0 + 1u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 18ull;\n"
"    const uint8_t* g0 = wg + (uint64_t)row0 * rb; const uint8_t* g1 = (row1 < out_dim) ? (g0 + rb) : g0;\n"
"    const uint8_t* u0 = wu + (uint64_t)row0 * rb; const uint8_t* u1 = (row1 < out_dim) ? (u0 + rb) : u0;\n"
"    float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        int4 q0 = __ldg(&xq8[b].q0), q1 = __ldg(&xq8[b].q1); float xd = __ldg(&xq8[b].d), xs = __ldg(&xq8[b].s);\n"
"        uint64_t boff = (uint64_t)b * 18ull;\n"
"        ag0 += dot_q4_0_dp4a_1b(g0 + boff, q0, q1, xd, xs); au0 += dot_q4_0_dp4a_1b(u0 + boff, q0, q1, xd, xs);\n"
"        ag1 += dot_q4_0_dp4a_1b(g1 + boff, q0, q1, xd, xs); au1 += dot_q4_0_dp4a_1b(u1 + boff, q0, q1, xd, xs);\n"
"    }\n"
"    ag0 = warp_sum(ag0); au0 = warp_sum(au0); ag1 = warp_sum(ag1); au1 = warp_sum(au1);\n"
"    if (lane == 0u) {\n"
"        h_out[row0] = (ag0 / (1.0f + expf(-ag0))) * au0;\n"
"        if (row1 < out_dim) h_out[row1] = (ag1 / (1.0f + expf(-ag1))) * au1;\n"
"    }\n"
"}\n"
"\n"
"// Q4_K MoE Slot Kernels (Qwen3-MoE / DeepSeek-R1 Q4_K_M!)\n"
"extern \"C\" __global__ void cm_slot_gate_up_q4k(const uint8_t* pool, const float* xn, const uint32_t* slot_ids, float* h_out,\n"
"                                               uint32_t d_model, uint32_t d_ff, uint32_t n_active, uint64_t gate_b, uint64_t slot_stride) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, s = blockIdx.y;\n"
"    uint32_t row = blockIdx.x * 4u + wid;\n"
"    if (s >= n_active || row >= d_ff) return;\n"
"    const uint8_t* base = pool + (uint64_t)slot_ids[s] * slot_stride;\n"
"    uint32_t nb = d_model >> 8; uint64_t rb = (uint64_t)nb * 144ull; uint32_t qtr = lane & 3u;\n"
"    const uint8_t* g0 = base + (uint64_t)row * rb; const uint8_t* u0 = base + gate_b + (uint64_t)row * rb;\n"
"    float ag = 0.0f, au = 0.0f;\n"
"    for (uint32_t b = (lane >> 2); b < nb; b += 8u) {\n"
"        const float* xp = xn + (b << 8); uint64_t boff = (uint64_t)b * 144ull;\n"
"        ag += dot_q4_k_quarter(g0 + boff, xp, qtr); au += dot_q4_k_quarter(u0 + boff, xp, qtr);\n"
"    }\n"
"    ag = warp_sum(ag); au = warp_sum(au);\n"
"    if (lane == 0u) h_out[(uint64_t)s * d_ff + row] = (ag / (1.0f + expf(-ag))) * au;\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_slot_down_accum_q4k(const uint8_t* pool, const float* h_in, const uint32_t* slot_ids, const float* e_wt, float* x_res,\n"
"                                                  uint32_t d_model, uint32_t d_ff, uint32_t n_active, uint64_t down_off, uint64_t slot_stride) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid;\n"
"    if (row >= d_model) return;\n"
"    uint32_t nb = d_ff >> 8; uint64_t rb = (uint64_t)nb * 144ull; uint32_t qtr = lane & 3u;\n"
"    float tot = 0.0f;\n"
"    for (uint32_t s = 0; s < n_active; s++) {\n"
"        const uint8_t* wd = pool + (uint64_t)slot_ids[s] * slot_stride + down_off + (uint64_t)row * rb;\n"
"        const float* hp = h_in + (uint64_t)s * d_ff; float acc = 0.0f;\n"
"        for (uint32_t b = (lane >> 2); b < nb; b += 8u) acc += dot_q4_k_quarter(wd + (uint64_t)b * 144ull, hp + (b << 8), qtr);\n"
"        tot = fmaf(e_wt[s], warp_sum(acc), tot);\n"
"    }\n"
"    if (lane == 0u) x_res[row] += tot;\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_slot_down_accum_q6k(const uint8_t* pool, const float* h_in, const uint32_t* slot_ids, const float* e_wt, float* x_res,\n"
"                                                  uint32_t d_model, uint32_t d_ff, uint32_t n_active, uint64_t down_off, uint64_t slot_stride) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid;\n"
"    if (row >= d_model) return;\n"
"    uint32_t nb = d_ff >> 8; uint64_t rb = (uint64_t)nb * 210ull; uint32_t qtr = lane & 3u;\n"
"    float tot = 0.0f;\n"
"    for (uint32_t s = 0; s < n_active; s++) {\n"
"        const uint8_t* wd = pool + (uint64_t)slot_ids[s] * slot_stride + down_off + (uint64_t)row * rb;\n"
"        const float* hp = h_in + (uint64_t)s * d_ff; float acc = 0.0f;\n"
"        for (uint32_t b = (lane >> 2); b < nb; b += 8u) acc += dot_q6_k_quarter(wd + (uint64_t)b * 210ull, hp + (b << 8), qtr);\n"
"        tot = fmaf(e_wt[s], warp_sum(acc), tot);\n"
"    }\n"
"    if (lane == 0u) x_res[row] += tot;\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_slot_down_accum_q5_0(const uint8_t* pool, const float* h_in, const uint32_t* slot_ids, const float* e_wt, float* x_res,\n"
"                                                   uint32_t d_model, uint32_t d_ff, uint32_t n_active, uint64_t down_off, uint64_t slot_stride) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid;\n"
"    if (row >= d_model) return;\n"
"    uint32_t nb = d_ff >> 5; uint64_t rb = (uint64_t)nb * 22ull;\n"
"    float tot = 0.0f;\n"
"    for (uint32_t s = 0; s < n_active; s++) {\n"
"        const uint8_t* wd = pool + (uint64_t)slot_ids[s] * slot_stride + down_off + (uint64_t)row * rb;\n"
"        const float* hp = h_in + (uint64_t)s * d_ff; float acc = 0.0f;\n"
"        for (uint32_t b = lane; b < nb; b += 32u) {\n"
"            float4 xv_lo[4], xv_hi[4]; float sx = load_x32_q4(hp + (b << 5), xv_lo, xv_hi);\n"
"            acc += dot_q5_0_fast(wd + (uint64_t)b * 22ull, xv_lo, xv_hi, sx);\n"
"        }\n"
"        tot = fmaf(e_wt[s], warp_sum(acc), tot);\n"
"    }\n"
"    if (lane == 0u) x_res[row] += tot;\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_slot_down_accum_q5_1(const uint8_t* pool, const float* h_in, const uint32_t* slot_ids, const float* e_wt, float* x_res,\n"
"                                                   uint32_t d_model, uint32_t d_ff, uint32_t n_active, uint64_t down_off, uint64_t slot_stride) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid;\n"
"    if (row >= d_model) return;\n"
"    uint32_t nb = d_ff >> 5; uint64_t rb = (uint64_t)nb * 24ull;\n"
"    float tot = 0.0f;\n"
"    for (uint32_t s = 0; s < n_active; s++) {\n"
"        const uint8_t* wd = pool + (uint64_t)slot_ids[s] * slot_stride + down_off + (uint64_t)row * rb;\n"
"        const float* hp = h_in + (uint64_t)s * d_ff; float acc = 0.0f;\n"
"        for (uint32_t b = lane; b < nb; b += 32u) {\n"
"            float4 xv_lo[4], xv_hi[4]; float sx = load_x32_q4(hp + (b << 5), xv_lo, xv_hi);\n"
"            acc += dot_q5_1_fast(wd + (uint64_t)b * 24ull, xv_lo, xv_hi, sx);\n"
"        }\n"
"        tot = fmaf(e_wt[s], warp_sum(acc), tot);\n"
"    }\n"
"    if (lane == 0u) x_res[row] += tot;\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_slot_down_accum_q8(const uint8_t* pool, const float* h_in, const uint32_t* slot_ids, const float* e_wt, float* x_res,\n"
"                                                 uint32_t d_model, uint32_t d_ff, uint32_t n_active, uint64_t down_off, uint64_t slot_stride) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid;\n"
"    if (row >= d_model) return;\n"
"    uint32_t nb = d_ff >> 5; uint64_t rb = (uint64_t)nb * 34ull;\n"
"    float tot = 0.0f;\n"
"    for (uint32_t s = 0; s < n_active; s++) {\n"
"        const uint8_t* wd = pool + (uint64_t)slot_ids[s] * slot_stride + down_off + (uint64_t)row * rb;\n"
"        const float* hp = h_in + (uint64_t)s * d_ff; float acc = 0.0f;\n"
"        for (uint32_t b = lane; b < nb; b += 32u) {\n"
"            const float4* x4 = (const float4*)(hp + (b << 5)); float4 xv[8];\n"
"            #pragma unroll\n"
"            for (uint32_t i = 0; i < 8u; i++) xv[i] = __ldg(&x4[i]);\n"
"            acc += dot_q8_0_fast(wd + (uint64_t)b * 34ull, xv);\n"
"        }\n"
"        tot = fmaf(e_wt[s], warp_sum(acc), tot);\n"
"    }\n"
"    if (lane == 0u) x_res[row] += tot;\n"
"}\n"
"\n"
"// =========================================================================\n"
"// Q8_0 Quantized KV-Cache (CUDA-Graph Indirect + Batched Prefill, -47% KV VRAM!)\n"
"// =========================================================================\n"
"extern \"C\" __global__ void cm_rope_and_kv_q8_batch(float* q, float* k, const float* v, uint8_t* k_cache, uint8_t* v_cache, const float* rope_freqs,\n"
"                                                   uint32_t n_heads, uint32_t n_kv, uint32_t d_head, uint32_t rope_dim,\n"
"                                                   uint32_t start_pos, float theta, uint32_t rope_neox, uint32_t has_freqs, uint32_t batch_b) {\n"
"    uint32_t h = blockIdx.x, b = blockIdx.y; if (h >= n_heads || b >= batch_b) return;\n"
"    uint32_t pos = start_pos + b, half_r = rope_dim >> 1;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint64_t qbase = (uint64_t)b * (n_heads * d_head) + h * d_head;\n"
"    uint64_t kbase = (uint64_t)b * (n_kv * d_head) + h * d_head;\n"
"    for (uint32_t i = lid; i < half_r; i += 128u) {\n"
"        uint32_t idx0 = rope_neox ? i : (2u * i), idx1 = rope_neox ? (i + half_r) : (2u * i + 1u);\n"
"        float inv_f = 1.0f / powf(theta, (float)(2u * i) / (float)rope_dim);\n"
"        if (has_freqs) inv_f /= rope_freqs[i];\n"
"        float ang = (float)pos * inv_f, cs = cosf(ang), sn = sinf(ang);\n"
"        float q0 = q[qbase + idx0], q1 = q[qbase + idx1];\n"
"        q[qbase + idx0] = q0 * cs - q1 * sn; q[qbase + idx1] = q0 * sn + q1 * cs;\n"
"        if (h < n_kv) {\n"
"            float k0 = k[kbase + idx0], k1 = k[kbase + idx1];\n"
"            k[kbase + idx0] = k0 * cs - k1 * sn; k[kbase + idx1] = k0 * sn + k1 * cs;\n"
"        }\n"
"    }\n"
"    __syncthreads();\n"
"    if (h < n_kv) {\n"
"        uint32_t nblk = d_head >> 5;\n"
"        uint64_t row_bytes = (uint64_t)n_kv * (uint64_t)nblk * 34ull;\n"
"        uint64_t hoff = (uint64_t)pos * row_bytes + (uint64_t)h * (uint64_t)nblk * 34ull;\n"
"        for (uint32_t blk = wid; blk < nblk; blk += 4u) {\n"
"            float kv = k[kbase + blk * 32u + lane], vv = v[kbase + blk * 32u + lane];\n"
"            float kam = fabsf(kv), vam = fabsf(vv);\n"
"            #pragma unroll\n"
"            for (int off = 16; off > 0; off >>= 1) {\n"
"                kam = fmaxf(kam, __shfl_xor_sync(0xFFFFFFFFu, kam, off));\n"
"                vam = fmaxf(vam, __shfl_xor_sync(0xFFFFFFFFu, vam, off));\n"
"            }\n"
"            float kd = kam / 127.0f, kid = (kam > 0.0f) ? (127.0f / kam) : 0.0f;\n"
"            float vd = vam / 127.0f, vid = (vam > 0.0f) ? (127.0f / vam) : 0.0f;\n"
"            uint8_t* kp = k_cache + hoff + (uint64_t)blk * 34ull;\n"
"            uint8_t* vp = v_cache + hoff + (uint64_t)blk * 34ull;\n"
"            if (lane == 0u) { *(uint16_t*)kp = __half_as_ushort(__float2half(kd)); *(uint16_t*)vp = __half_as_ushort(__float2half(vd)); }\n"
"            ((int8_t*)(kp + 2u))[lane] = (int8_t)__float2int_rn(kv * kid);\n"
"            ((int8_t*)(vp + 2u))[lane] = (int8_t)__float2int_rn(vv * vid);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_rope_and_kv_q8_ind(float* q, float* k, const float* v, uint8_t* k_cache, uint8_t* v_cache, const float* rope_freqs,\n"
"                                                 uint32_t n_heads, uint32_t n_kv, uint32_t d_head, uint32_t rope_dim,\n"
"                                                 const uint32_t* step_args, float theta, uint32_t rope_neox, uint32_t has_freqs) {\n"
"    uint32_t h = blockIdx.x; if (h >= n_heads) return;\n"
"    uint32_t pos = step_args[1], half_r = rope_dim >> 1;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint64_t qbase = (uint64_t)h * d_head, kbase = (uint64_t)h * d_head;\n"
"    for (uint32_t i = lid; i < half_r; i += 128u) {\n"
"        uint32_t idx0 = rope_neox ? i : (2u * i), idx1 = rope_neox ? (i + half_r) : (2u * i + 1u);\n"
"        float inv_f = 1.0f / powf(theta, (float)(2u * i) / (float)rope_dim);\n"
"        if (has_freqs) inv_f /= rope_freqs[i];\n"
"        float ang = (float)pos * inv_f, cs = cosf(ang), sn = sinf(ang);\n"
"        float q0 = q[qbase + idx0], q1 = q[qbase + idx1];\n"
"        q[qbase + idx0] = q0 * cs - q1 * sn; q[qbase + idx1] = q0 * sn + q1 * cs;\n"
"        if (h < n_kv) {\n"
"            float k0 = k[kbase + idx0], k1 = k[kbase + idx1];\n"
"            k[kbase + idx0] = k0 * cs - k1 * sn; k[kbase + idx1] = k0 * sn + k1 * cs;\n"
"        }\n"
"    }\n"
"    __syncthreads();\n"
"    if (h < n_kv) {\n"
"        uint32_t nblk = d_head >> 5;\n"
"        uint64_t row_bytes = (uint64_t)n_kv * (uint64_t)nblk * 34ull;\n"
"        uint64_t hoff = (uint64_t)pos * row_bytes + (uint64_t)h * (uint64_t)nblk * 34ull;\n"
"        for (uint32_t blk = wid; blk < nblk; blk += 4u) {\n"
"            float kv = k[kbase + blk * 32u + lane], vv = v[kbase + blk * 32u + lane];\n"
"            float kam = fabsf(kv), vam = fabsf(vv);\n"
"            #pragma unroll\n"
"            for (int off = 16; off > 0; off >>= 1) {\n"
"                kam = fmaxf(kam, __shfl_xor_sync(0xFFFFFFFFu, kam, off));\n"
"                vam = fmaxf(vam, __shfl_xor_sync(0xFFFFFFFFu, vam, off));\n"
"            }\n"
"            float kd = kam / 127.0f, kid = (kam > 0.0f) ? (127.0f / kam) : 0.0f;\n"
"            float vd = vam / 127.0f, vid = (vam > 0.0f) ? (127.0f / vam) : 0.0f;\n"
"            uint8_t* kp = k_cache + hoff + (uint64_t)blk * 34ull;\n"
"            uint8_t* vp = v_cache + hoff + (uint64_t)blk * 34ull;\n"
"            if (lane == 0u) { *(uint16_t*)kp = __half_as_ushort(__float2half(kd)); *(uint16_t*)vp = __half_as_ushort(__float2half(vd)); }\n"
"            ((int8_t*)(kp + 2u))[lane] = (int8_t)__float2int_rn(kv * kid);\n"
"            ((int8_t*)(vp + 2u))[lane] = (int8_t)__float2int_rn(vv * vid);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_flash_attn_q8_batch(const float* q, const uint8_t* k_cache, const uint8_t* v_cache, float* ctx,\n"
"                                                  uint32_t n_heads, uint32_t n_kv, uint32_t d_head, uint32_t start_pos, uint32_t batch_b) {\n"
"    uint32_t h = blockIdx.x, b = blockIdx.y; if (h >= n_heads || b >= batch_b) return;\n"
"    uint32_t kv_len = start_pos + b + 1u;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint32_t grp = (n_heads / n_kv) > 0u ? (n_heads / n_kv) : 1u, kvh = h / grp;\n"
"    uint32_t nblk = d_head >> 5; uint64_t row_bytes = (uint64_t)n_kv * (uint64_t)nblk * 34ull;\n"
"    float scale = rsqrtf((float)d_head);\n"
"    __shared__ float s_q[256], s_m[4], s_l[4], s_o[4][256];\n"
"    uint64_t qbase = (uint64_t)b * (n_heads * d_head) + h * d_head;\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) s_q[d] = q[qbase + d];\n"
"    __syncthreads();\n"
"    float m_run = -1e30f, l_run = 0.0f, o_acc[8] = {0,0,0,0,0,0,0,0};\n"
"    for (uint32_t t = wid; t < kv_len; t += 4u) {\n"
"        uint64_t hoff = (uint64_t)t * row_bytes + (uint64_t)kvh * (uint64_t)nblk * 34ull;\n"
"        const uint8_t* kp = k_cache + hoff; const uint8_t* vp = v_cache + hoff;\n"
"        float dot = 0.0f;\n"
"        for (uint32_t blk = 0; blk < nblk; blk++) {\n"
"            const uint8_t* kb = kp + (uint64_t)blk * 34ull;\n"
"            dot = fmaf(s_q[blk * 32u + lane], (float)((const int8_t*)(kb + 2u))[lane] * load_f16_le(kb), dot);\n"
"        }\n"
"        float sc = warp_sum(dot) * scale; sc = __shfl_sync(0xFFFFFFFFu, sc, 0);\n"
"        float mn = fmaxf(m_run, sc), al = expf(m_run - mn), wt = expf(sc - mn);\n"
"        l_run = fmaf(l_run, al, wt); m_run = mn;\n"
"        for (uint32_t blk = 0; blk < nblk; blk++) {\n"
"            const uint8_t* vb = vp + (uint64_t)blk * 34ull;\n"
"            o_acc[blk] = fmaf(o_acc[blk], al, wt * (float)((const int8_t*)(vb + 2u))[lane] * load_f16_le(vb));\n"
"        }\n"
"    }\n"
"    if (lane == 0u) { s_m[wid] = m_run; s_l[wid] = l_run; }\n"
"    for (uint32_t blk = 0; blk < nblk; blk++) s_o[wid][blk * 32u + lane] = o_acc[blk];\n"
"    __syncthreads();\n"
"    float gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3])), gl = 0.0f;\n"
"    for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * expf(s_m[s] - gm);\n"
"    float inv = 1.0f / fmaxf(gl, 1e-12f);\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) {\n"
"        float ov = 0.0f; for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fmaf(s_o[s][d], expf(s_m[s] - gm), ov);\n"
"        ctx[qbase + d] = ov * inv;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_flash_decode_q8_ind(const float* q, const uint8_t* k_cache, const uint8_t* v_cache, float* ctx,\n"
"                                                  uint32_t n_heads, uint32_t n_kv, uint32_t d_head, const uint32_t* step_args) {\n"
"    uint32_t h = blockIdx.x; if (h >= n_heads) return;\n"
"    uint32_t kv_len = step_args[1] + 1u;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint32_t grp = (n_heads / n_kv) > 0u ? (n_heads / n_kv) : 1u, kvh = h / grp;\n"
"    uint32_t nblk = d_head >> 5; uint64_t row_bytes = (uint64_t)n_kv * (uint64_t)nblk * 34ull;\n"
"    float scale = rsqrtf((float)d_head);\n"
"    __shared__ float s_q[256], s_m[4], s_l[4], s_o[4][256];\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) s_q[d] = q[h * d_head + d];\n"
"    __syncthreads();\n"
"    float m_run = -1e30f, l_run = 0.0f, o_acc[8] = {0,0,0,0,0,0,0,0};\n"
"    for (uint32_t t = wid; t < kv_len; t += 4u) {\n"
"        uint64_t hoff = (uint64_t)t * row_bytes + (uint64_t)kvh * (uint64_t)nblk * 34ull;\n"
"        const uint8_t* kp = k_cache + hoff; const uint8_t* vp = v_cache + hoff;\n"
"        float dot = 0.0f;\n"
"        for (uint32_t blk = 0; blk < nblk; blk++) {\n"
"            const uint8_t* kb = kp + (uint64_t)blk * 34ull;\n"
"            dot = fmaf(s_q[blk * 32u + lane], (float)((const int8_t*)(kb + 2u))[lane] * load_f16_le(kb), dot);\n"
"        }\n"
"        float sc = warp_sum(dot) * scale; sc = __shfl_sync(0xFFFFFFFFu, sc, 0);\n"
"        float mn = fmaxf(m_run, sc), al = expf(m_run - mn), wt = expf(sc - mn);\n"
"        l_run = fmaf(l_run, al, wt); m_run = mn;\n"
"        for (uint32_t blk = 0; blk < nblk; blk++) {\n"
"            const uint8_t* vb = vp + (uint64_t)blk * 34ull;\n"
"            o_acc[blk] = fmaf(o_acc[blk], al, wt * (float)((const int8_t*)(vb + 2u))[lane] * load_f16_le(vb));\n"
"        }\n"
"    }\n"
"    if (lane == 0u) { s_m[wid] = m_run; s_l[wid] = l_run; }\n"
"    for (uint32_t blk = 0; blk < nblk; blk++) s_o[wid][blk * 32u + lane] = o_acc[blk];\n"
"    __syncthreads();\n"
"    float gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3])), gl = 0.0f;\n"
"    for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * expf(s_m[s] - gm);\n"
"    float inv = 1.0f / fmaxf(gl, 1e-12f);\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) {\n"
"        float ov = 0.0f; for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fmaf(s_o[s][d], expf(s_m[s] - gm), ov);\n"
"        ctx[h * d_head + d] = ov * inv;\n"
"    }\n"
"}\n"
"\n"
"// =========================================================================\n"
"// Batched Causal Prefill Kernels (B=1..64: 1 Weight Pass -> 10-25x Faster TTFT!)\n"
"// =========================================================================\n"
"extern \"C\" __global__ void cm_add_bias_batch(float* x, const float* bias, uint32_t dim, uint32_t batch_b) {\n"
"    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y;\n"
"    if (d < dim && b < batch_b) x[(uint64_t)b * dim + d] += bias[d];\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_rope_and_kv_batch(float* q, float* k, const float* v, half* k_cache, half* v_cache, const float* rope_freqs,\n"
"                                                uint32_t n_heads, uint32_t n_kv, uint32_t d_head, uint32_t rope_dim,\n"
"                                                uint32_t start_pos, float theta, uint32_t rope_neox, uint32_t has_freqs, uint32_t batch_b) {\n"
"    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x, h = blockIdx.y, b = blockIdx.z;\n"
"    uint32_t half_d = d_head >> 1, half_r = rope_dim >> 1;\n"
"    if (i >= half_d || h >= n_heads || b >= batch_b) return;\n"
"    uint32_t pos = start_pos + b;\n"
"    uint64_t qoff = (uint64_t)b * (n_heads * d_head) + h * d_head;\n"
"    uint64_t koff = (uint64_t)b * (n_kv * d_head) + h * d_head;\n"
"    if (i < half_r) {\n"
"        uint32_t idx0 = rope_neox ? i : (2u * i), idx1 = rope_neox ? (i + half_r) : (2u * i + 1u);\n"
"        float inv_f = 1.0f / powf(theta, (float)(2u * i) / (float)rope_dim);\n"
"        if (has_freqs) inv_f /= rope_freqs[i];\n"
"        float ang = (float)pos * inv_f, cs = cosf(ang), sn = sinf(ang);\n"
"        float q0 = q[qoff + idx0], q1 = q[qoff + idx1];\n"
"        q[qoff + idx0] = q0 * cs - q1 * sn; q[qoff + idx1] = q0 * sn + q1 * cs;\n"
"        if (h < n_kv) {\n"
"            float k0 = k[koff + idx0], k1 = k[koff + idx1];\n"
"            float rk0 = k0 * cs - k1 * sn, rk1 = k0 * sn + k1 * cs;\n"
"            k[koff + idx0] = rk0; k[koff + idx1] = rk1;\n"
"            uint64_t cb = (uint64_t)pos * (n_kv * d_head) + h * d_head;\n"
"            k_cache[cb + idx0] = __float2half(rk0); k_cache[cb + idx1] = __float2half(rk1);\n"
"            v_cache[cb + idx0] = __float2half(v[koff + idx0]); v_cache[cb + idx1] = __float2half(v[koff + idx1]);\n"
"        }\n"
"    } else if (h < n_kv) {\n"
"        uint32_t idx0 = 2u * i, idx1 = 2u * i + 1u;\n"
"        uint64_t cb = (uint64_t)pos * (n_kv * d_head) + h * d_head;\n"
"        k_cache[cb + idx0] = __float2half(k[koff + idx0]); k_cache[cb + idx1] = __float2half(k[koff + idx1]);\n"
"        v_cache[cb + idx0] = __float2half(v[koff + idx0]); v_cache[cb + idx1] = __float2half(v[koff + idx1]);\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_flash_prefill_fp16(const float* q, const half* k_cache, const half* v_cache, float* ctx,\n"
"                                                 uint32_t n_heads, uint32_t n_kv, uint32_t d_head, uint32_t start_pos, uint32_t batch_b) {\n"
"    uint32_t h = blockIdx.x, b = blockIdx.y; if (h >= n_heads || b >= batch_b) return;\n"
"    uint32_t kv_len = start_pos + b + 1u;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint32_t grp = (n_heads / n_kv) > 0u ? (n_heads / n_kv) : 1u, kvh = h / grp;\n"
"    uint32_t kv_stride = n_kv * d_head; float scale = rsqrtf((float)d_head);\n"
"    __shared__ float s_q[256], s_m[4], s_l[4], s_o[4][256];\n"
"    uint64_t qbase = (uint64_t)b * (n_heads * d_head) + h * d_head;\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) s_q[d] = q[qbase + d];\n"
"    __syncthreads();\n"
"    float m_run = -1e30f, l_run = 0.0f, o_acc[8] = {0,0,0,0,0,0,0,0}; uint32_t dpl = (d_head + 31u) >> 5;\n"
"    for (uint32_t t = wid; t < kv_len; t += 4u) {\n"
"        const half* kp = k_cache + (uint64_t)t * kv_stride + kvh * d_head;\n"
"        const half* vp = v_cache + (uint64_t)t * kv_stride + kvh * d_head;\n"
"        float dot = 0.0f;\n"
"        for (uint32_t k = 0; k < dpl; k++) { uint32_t d = lane + k * 32u; if (d < d_head) dot = fmaf(s_q[d], __half2float(kp[d]), dot); }\n"
"        float sc = warp_sum(dot) * scale; sc = __shfl_sync(0xFFFFFFFFu, sc, 0);\n"
"        float mn = fmaxf(m_run, sc), al = expf(m_run - mn), wt = expf(sc - mn);\n"
"        l_run = fmaf(l_run, al, wt); m_run = mn;\n"
"        for (uint32_t k = 0; k < dpl; k++) { uint32_t d = lane + k * 32u; if (d < d_head) o_acc[k] = fmaf(o_acc[k], al, wt * __half2float(vp[d])); }\n"
"    }\n"
"    if (lane == 0u) { s_m[wid] = m_run; s_l[wid] = l_run; }\n"
"    for (uint32_t k = 0; k < dpl; k++) { uint32_t d = lane + k * 32u; if (d < d_head) s_o[wid][d] = o_acc[k]; }\n"
"    __syncthreads();\n"
"    float gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3])), gl = 0.0f;\n"
"    for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * expf(s_m[s] - gm);\n"
"    float inv = 1.0f / fmaxf(gl, 1e-12f);\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) {\n"
"        float ov = 0.0f; for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fmaf(s_o[s][d], expf(s_m[s] - gm), ov);\n"
"        ctx[qbase + d] = ov * inv;\n"
"    }\n"
"}\n"
"\n"
"// =========================================================================\n"
"// Tenon v6 Marlin-Style W4A16 Tensor Core Prefill (`BM=32 x BN=64 x BK=32`, 8 warps = 256 thr):\n"
"//   - 2-Stage `cp.async.ca.shared.global` Hardware Async Double-Buffering of FP32 Activations\n"
"//   - IEEE-754 FP16x2 Magic-Bit (`0x64006400`) In-Register INT4 -> FP16x2 Dequant (ZERO int->float!)\n"
"//   - `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` Tensor Core Mainloop (2x BN reuse!)\n"
"// =========================================================================\n"
"#if __CUDA_ARCH__ >= 800\n"
"__device__ __forceinline__ void cm_cp_async16(void* smem_ptr, const void* gmem_ptr, int src_bytes) {\n"
"    uint32_t saddr = (uint32_t)__cvta_generic_to_shared(smem_ptr);\n"
"    asm volatile(\"cp.async.ca.shared.global [%0], [%1], 16, %2;\\n\" :: \"r\"(saddr), \"l\"(gmem_ptr), \"r\"(src_bytes));\n"
"}\n"
"__device__ __forceinline__ uint32_t cm_deq_q4_0_pair(uint32_t n0, uint32_t n1, __half2 hd) {\n"
"    uint32_t raw = 0x64006400u | (n0 & 0xFu) | ((n1 & 0xFu) << 16);\n"
"    uint32_t sub = 0x64086408u;\n"
"    __half2 v = __hmul2(__hsub2(*reinterpret_cast<const __half2*>(&raw), *reinterpret_cast<const __half2*>(&sub)), hd);\n"
"    return *reinterpret_cast<const uint32_t*>(&v);\n"
"}\n"
"__device__ __forceinline__ uint32_t cm_deq_q4_1_pair(uint32_t n0, uint32_t n1, __half2 hd, __half2 hm) {\n"
"    uint32_t raw = 0x64006400u | (n0 & 0xFu) | ((n1 & 0xFu) << 16);\n"
"    uint32_t sub = 0x64006400u;\n"
"    __half2 v = __hfma2(__hsub2(*reinterpret_cast<const __half2*>(&raw), *reinterpret_cast<const __half2*>(&sub)), hd, hm);\n"
"    return *reinterpret_cast<const uint32_t*>(&v);\n"
"}\n"
"#endif\n"
"\n"
"extern \"C\" __global__ void cm_gemm_q4_batch(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t r0 = blockIdx.x * 64u, b0 = blockIdx.y * 32u;\n"
"    if (r0 >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t lid = threadIdx.x, nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 18ull;\n"
"#if __CUDA_ARCH__ >= 800\n"
"    __shared__ float sXf[2][32][32];\n"
"    __shared__ uint16_t sX[32][34];\n"
"    __shared__ uint16_t sW[64][34];\n"
"    // Activation cp.async mapping: 256 threads = 32 rows x 8 float4 chunks (16 bytes each = 32 floats/row)!\n"
"    uint32_t tr_x = lid >> 3, cx4 = (lid & 7u) * 4u;\n"
"    uint32_t gb_clamped = (b0 + tr_x < batch_b) ? (b0 + tr_x) : (batch_b - 1u);\n"
"    int x_bytes = (b0 + tr_x < batch_b) ? 16 : 0;\n"
"    // Weight magic-bit FP16x2 mapping: 256 threads = 64 rows x 4 chunks (4 Q4 bytes = 8 weights = 4 half2s/thr)!\n"
"    uint32_t tr_w = lid >> 2, gw = lid & 3u, qoff = gw * 4u;\n"
"    uint32_t gr = r0 + tr_w;\n"
"    const uint8_t* wrow = w + (uint64_t)(gr < out_dim ? gr : (out_dim - 1u)) * rb;\n"
"    // 8 warps = 2 M-tiles (wb = 0,16) x 4 N-tiles (wr = 0,16,32,48)\n"
"    uint32_t wid = lid >> 5, lane = lid & 31u, gid = lane >> 2, tid = lane & 3u;\n"
"    uint32_t wb = (wid >> 2) * 16u, wr = (wid & 3u) * 16u;\n"
"    int warp_act = (b0 + wb < batch_b) && (r0 + wr < out_dim);\n"
"    float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;\n"
"    float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;\n"
"\n"
"    // Stage 0 async prefetch of bk=0 FP32 activations via hardware cp.async!\n"
"    cm_cp_async16(&sXf[0][tr_x][cx4], x + (uint64_t)gb_clamped * in_dim + cx4, x_bytes);\n"
"    asm volatile(\"cp.async.commit_group;\\n\");\n"
"\n"
"    for (uint32_t bk = 0; bk < nb; bk++) {\n"
"        uint32_t st = bk & 1u, nxt = st ^ 1u;\n"
"        if (bk + 1u < nb) {\n"
"            cm_cp_async16(&sXf[nxt][tr_x][cx4], x + (uint64_t)gb_clamped * in_dim + ((bk + 1u) << 5) + cx4, x_bytes);\n"
"        }\n"
"        asm volatile(\"cp.async.commit_group;\\n\");\n"
"        // Dequantize 64 rows x 32 Q4_0 weights using FP16x2 magic exponent 0x64006400 OVERLAPPED with cp.async DMA!\n"
"        if (gr < out_dim) {\n"
"            const uint8_t* blk = wrow + (uint64_t)bk * 18ull;\n"
"            __half2 hd = __half2half2(__ushort_as_half(*(const uint16_t*)blk));\n"
"            uint8_t b0v = __ldg(&blk[2u + qoff + 0u]), b1v = __ldg(&blk[2u + qoff + 1u]);\n"
"            uint8_t b2v = __ldg(&blk[2u + qoff + 2u]), b3v = __ldg(&blk[2u + qoff + 3u]);\n"
"            *(uint32_t*)&sW[tr_w][qoff + 0u]       = cm_deq_q4_0_pair(b0v, b1v, hd);\n"
"            *(uint32_t*)&sW[tr_w][qoff + 2u]       = cm_deq_q4_0_pair(b2v, b3v, hd);\n"
"            *(uint32_t*)&sW[tr_w][16u + qoff + 0u] = cm_deq_q4_0_pair(b0v >> 4, b1v >> 4, hd);\n"
"            *(uint32_t*)&sW[tr_w][16u + qoff + 2u] = cm_deq_q4_0_pair(b2v >> 4, b3v >> 4, hd);\n"
"        } else {\n"
"            *(uint32_t*)&sW[tr_w][qoff + 0u] = 0u; *(uint32_t*)&sW[tr_w][qoff + 2u] = 0u;\n"
"            *(uint32_t*)&sW[tr_w][16u + qoff + 0u] = 0u; *(uint32_t*)&sW[tr_w][16u + qoff + 2u] = 0u;\n"
"        }\n"
"        asm volatile(\"cp.async.wait_group 1;\\n\");\n"
"        __syncthreads();\n"
"        // Convert staged FP32 activations -> packed half2 in sX[32][34] (2 half2/thread!)\n"
"        __half2 xh0 = __float22half2_rn(make_float2(sXf[st][tr_x][cx4 + 0u], sXf[st][tr_x][cx4 + 1u]));\n"
"        __half2 xh1 = __float22half2_rn(make_float2(sXf[st][tr_x][cx4 + 2u], sXf[st][tr_x][cx4 + 3u]));\n"
"        *(uint32_t*)&sX[tr_x][cx4 + 0u] = *reinterpret_cast<const uint32_t*>(&xh0);\n"
"        *(uint32_t*)&sX[tr_x][cx4 + 2u] = *reinterpret_cast<const uint32_t*>(&xh1);\n"
"        __syncthreads();\n"
"        if (warp_act) {\n"
"            #pragma unroll\n"
"            for (uint32_t kk = 0; kk < 32u; kk += 16u) {\n"
"                uint32_t a0 = *(const uint32_t*)&sX[wb + gid][kk + tid * 2u];\n"
"                uint32_t a1 = *(const uint32_t*)&sX[wb + gid + 8u][kk + tid * 2u];\n"
"                uint32_t a2 = *(const uint32_t*)&sX[wb + gid][kk + 8u + tid * 2u];\n"
"                uint32_t a3 = *(const uint32_t*)&sX[wb + gid + 8u][kk + 8u + tid * 2u];\n"
"                uint32_t b00 = *(const uint32_t*)&sW[wr + gid][kk + tid * 2u];\n"
"                uint32_t b01 = *(const uint32_t*)&sW[wr + gid][kk + 8u + tid * 2u];\n"
"                asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
"                             : \"+f\"(c00), \"+f\"(c01), \"+f\"(c02), \"+f\"(c03) : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b00), \"r\"(b01));\n"
"                uint32_t b10 = *(const uint32_t*)&sW[wr + 8u + gid][kk + tid * 2u];\n"
"                uint32_t b11 = *(const uint32_t*)&sW[wr + 8u + gid][kk + 8u + tid * 2u];\n"
"                asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
"                             : \"+f\"(c10), \"+f\"(c11), \"+f\"(c12), \"+f\"(c13) : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b10), \"r\"(b11));\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    asm volatile(\"cp.async.wait_all;\\n\");\n"
"    if (warp_act) {\n"
"        uint32_t ob0 = b0 + wb + gid, ob1 = ob0 + 8u, or0 = r0 + wr + tid * 2u, or1 = or0 + 8u;\n"
"        #define ST_TC(ob, orw, val) if ((ob) < batch_b && (orw) < out_dim) { uint64_t idx = (uint64_t)(ob) * out_dim + (orw); y[idx] = add_res ? (y[idx] + (val)) : (val); }\n"
"        ST_TC(ob0, or0,      c00); ST_TC(ob0, or0 + 1u, c01); ST_TC(ob1, or0,      c02); ST_TC(ob1, or0 + 1u, c03);\n"
"        ST_TC(ob0, or1,      c10); ST_TC(ob0, or1 + 1u, c11); ST_TC(ob1, or1,      c12); ST_TC(ob1, or1 + 1u, c13);\n"
"        #undef ST_TC\n"
"    }\n"
"#else\n"
"    uint32_t tr_w = lid >> 2, gw = lid & 3u, qoff = gw * 4u, gr = r0 + tr_w;\n"
"    uint32_t tr_x = lid >> 3, cx4 = (lid & 7u) * 4u, gb = b0 + tr_x;\n"
"    const uint8_t* wrow = w + (uint64_t)(gr < out_dim ? gr : (out_dim - 1u)) * rb;\n"
"    __shared__ float sX[32][33]; __shared__ float sW[64][33];\n"
"    uint32_t th_b = (lid >> 4) * 2u, th_r = (lid & 15u) * 4u;\n"
"    float acc[2][4] = {{0,0,0,0},{0,0,0,0}};\n"
"    for (uint32_t bk = 0; bk < nb; bk++) {\n"
"        if (gb < batch_b) {\n"
"            float4 v0 = __ldg((const float4*)(x + (uint64_t)gb * in_dim + (bk << 5) + cx4));\n"
"            sX[tr_x][cx4+0]=v0.x; sX[tr_x][cx4+1]=v0.y; sX[tr_x][cx4+2]=v0.z; sX[tr_x][cx4+3]=v0.w;\n"
"        } else { for (uint32_t j = 0; j < 4u; j++) sX[tr_x][cx4 + j] = 0.0f; }\n"
"        if (gr < out_dim) {\n"
"            const uint8_t* blk = wrow + (uint64_t)bk * 18ull;\n"
"            float d = load_f16_le(blk);\n"
"            for (uint32_t j = 0; j < 4u; j++) {\n"
"                uint8_t bv = __ldg(&blk[2u + qoff + j]);\n"
"                sW[tr_w][qoff + j] = (float)((int)(bv & 0xFu) - 8) * d;\n"
"                sW[tr_w][16u + qoff + j] = (float)((int)(bv >> 4) - 8) * d;\n"
"            }\n"
"        } else { for (uint32_t j = 0; j < 4u; j++) { sW[tr_w][qoff+j]=0.0f; sW[tr_w][16u+qoff+j]=0.0f; } }\n"
"        __syncthreads();\n"
"        for (uint32_t k = 0; k < 32u; k++) {\n"
"            float x0 = sX[th_b + 0u][k], x1 = sX[th_b + 1u][k];\n"
"            float w0 = sW[th_r + 0u][k], w1 = sW[th_r + 1u][k], w2 = sW[th_r + 2u][k], w3 = sW[th_r + 3u][k];\n"
"            acc[0][0] = fmaf(x0, w0, acc[0][0]); acc[0][1] = fmaf(x0, w1, acc[0][1]); acc[0][2] = fmaf(x0, w2, acc[0][2]); acc[0][3] = fmaf(x0, w3, acc[0][3]);\n"
"            acc[1][0] = fmaf(x1, w0, acc[1][0]); acc[1][1] = fmaf(x1, w1, acc[1][1]); acc[1][2] = fmaf(x1, w2, acc[1][2]); acc[1][3] = fmaf(x1, w3, acc[1][3]);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (uint32_t tb = 0; tb < 2u; tb++) for (uint32_t rr = 0; rr < 4u; rr++) {\n"
"        uint32_t ob = b0 + th_b + tb, orow = r0 + th_r + rr;\n"
"        if (ob < batch_b && orow < out_dim) { uint64_t idx = (uint64_t)ob * out_dim + orow; y[idx] = add_res ? (y[idx] + acc[tb][rr]) : acc[tb][rr]; }\n"
"    }\n"
"#endif\n"
"}\n"
"\n"
"// Marlin-Style W4A16 Tensor Core Batched Q4_1 GEMM (`BM=32 x BN=64 x BK=32`, 2-stage cp.async + FP16x2 magic-bit!)\n"
"extern \"C\" __global__ void cm_gemm_q4_1_batch(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t r0 = blockIdx.x * 64u, b0 = blockIdx.y * 32u;\n"
"    if (r0 >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t lid = threadIdx.x, nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 20ull;\n"
"#if __CUDA_ARCH__ >= 800\n"
"    __shared__ float sXf[2][32][32];\n"
"    __shared__ uint16_t sX[32][34];\n"
"    __shared__ uint16_t sW[64][34];\n"
"    uint32_t tr_x = lid >> 3, cx4 = (lid & 7u) * 4u;\n"
"    uint32_t gb_clamped = (b0 + tr_x < batch_b) ? (b0 + tr_x) : (batch_b - 1u);\n"
"    int x_bytes = (b0 + tr_x < batch_b) ? 16 : 0;\n"
"    uint32_t tr_w = lid >> 2, gw = lid & 3u, qoff = gw * 4u;\n"
"    uint32_t gr = r0 + tr_w;\n"
"    const uint8_t* wrow = w + (uint64_t)(gr < out_dim ? gr : (out_dim - 1u)) * rb;\n"
"    uint32_t wid = lid >> 5, lane = lid & 31u, gid = lane >> 2, tid = lane & 3u;\n"
"    uint32_t wb = (wid >> 2) * 16u, wr = (wid & 3u) * 16u;\n"
"    int warp_act = (b0 + wb < batch_b) && (r0 + wr < out_dim);\n"
"    float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;\n"
"    float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;\n"
"\n"
"    cm_cp_async16(&sXf[0][tr_x][cx4], x + (uint64_t)gb_clamped * in_dim + cx4, x_bytes);\n"
"    asm volatile(\"cp.async.commit_group;\\n\");\n"
"\n"
"    for (uint32_t bk = 0; bk < nb; bk++) {\n"
"        uint32_t st = bk & 1u, nxt = st ^ 1u;\n"
"        if (bk + 1u < nb) {\n"
"            cm_cp_async16(&sXf[nxt][tr_x][cx4], x + (uint64_t)gb_clamped * in_dim + ((bk + 1u) << 5) + cx4, x_bytes);\n"
"        }\n"
"        asm volatile(\"cp.async.commit_group;\\n\");\n"
"        if (gr < out_dim) {\n"
"            const uint8_t* blk = wrow + (uint64_t)bk * 20ull;\n"
"            __half2 hd = __half2half2(__ushort_as_half(*(const uint16_t*)blk));\n"
"            __half2 hm = __half2half2(__ushort_as_half(*(const uint16_t*)(blk + 2u)));\n"
"            uint8_t b0v = __ldg(&blk[4u + qoff + 0u]), b1v = __ldg(&blk[4u + qoff + 1u]);\n"
"            uint8_t b2v = __ldg(&blk[4u + qoff + 2u]), b3v = __ldg(&blk[4u + qoff + 3u]);\n"
"            *(uint32_t*)&sW[tr_w][qoff + 0u]       = cm_deq_q4_1_pair(b0v, b1v, hd, hm);\n"
"            *(uint32_t*)&sW[tr_w][qoff + 2u]       = cm_deq_q4_1_pair(b2v, b3v, hd, hm);\n"
"            *(uint32_t*)&sW[tr_w][16u + qoff + 0u] = cm_deq_q4_1_pair(b0v >> 4, b1v >> 4, hd, hm);\n"
"            *(uint32_t*)&sW[tr_w][16u + qoff + 2u] = cm_deq_q4_1_pair(b2v >> 4, b3v >> 4, hd, hm);\n"
"        } else {\n"
"            *(uint32_t*)&sW[tr_w][qoff + 0u] = 0u; *(uint32_t*)&sW[tr_w][qoff + 2u] = 0u;\n"
"            *(uint32_t*)&sW[tr_w][16u + qoff + 0u] = 0u; *(uint32_t*)&sW[tr_w][16u + qoff + 2u] = 0u;\n"
"        }\n"
"        asm volatile(\"cp.async.wait_group 1;\\n\");\n"
"        __syncthreads();\n"
"        __half2 xh0 = __float22half2_rn(make_float2(sXf[st][tr_x][cx4 + 0u], sXf[st][tr_x][cx4 + 1u]));\n"
"        __half2 xh1 = __float22half2_rn(make_float2(sXf[st][tr_x][cx4 + 2u], sXf[st][tr_x][cx4 + 3u]));\n"
"        *(uint32_t*)&sX[tr_x][cx4 + 0u] = *reinterpret_cast<const uint32_t*>(&xh0);\n"
"        *(uint32_t*)&sX[tr_x][cx4 + 2u] = *reinterpret_cast<const uint32_t*>(&xh1);\n"
"        __syncthreads();\n"
"        if (warp_act) {\n"
"            #pragma unroll\n"
"            for (uint32_t kk = 0; kk < 32u; kk += 16u) {\n"
"                uint32_t a0 = *(const uint32_t*)&sX[wb + gid][kk + tid * 2u];\n"
"                uint32_t a1 = *(const uint32_t*)&sX[wb + gid + 8u][kk + tid * 2u];\n"
"                uint32_t a2 = *(const uint32_t*)&sX[wb + gid][kk + 8u + tid * 2u];\n"
"                uint32_t a3 = *(const uint32_t*)&sX[wb + gid + 8u][kk + 8u + tid * 2u];\n"
"                uint32_t b00 = *(const uint32_t*)&sW[wr + gid][kk + tid * 2u];\n"
"                uint32_t b01 = *(const uint32_t*)&sW[wr + gid][kk + 8u + tid * 2u];\n"
"                asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
"                             : \"+f\"(c00), \"+f\"(c01), \"+f\"(c02), \"+f\"(c03) : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b00), \"r\"(b01));\n"
"                uint32_t b10 = *(const uint32_t*)&sW[wr + 8u + gid][kk + tid * 2u];\n"
"                uint32_t b11 = *(const uint32_t*)&sW[wr + 8u + gid][kk + 8u + tid * 2u];\n"
"                asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
"                             : \"+f\"(c10), \"+f\"(c11), \"+f\"(c12), \"+f\"(c13) : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b10), \"r\"(b11));\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    asm volatile(\"cp.async.wait_all;\\n\");\n"
"    if (warp_act) {\n"
"        uint32_t ob0 = b0 + wb + gid, ob1 = ob0 + 8u, or0 = r0 + wr + tid * 2u, or1 = or0 + 8u;\n"
"        #define ST_TC(ob, orw, val) if ((ob) < batch_b && (orw) < out_dim) { uint64_t idx = (uint64_t)(ob) * out_dim + (orw); y[idx] = add_res ? (y[idx] + (val)) : (val); }\n"
"        ST_TC(ob0, or0,      c00); ST_TC(ob0, or0 + 1u, c01); ST_TC(ob1, or0,      c02); ST_TC(ob1, or0 + 1u, c03);\n"
"        ST_TC(ob0, or1,      c10); ST_TC(ob0, or1 + 1u, c11); ST_TC(ob1, or1,      c12); ST_TC(ob1, or1 + 1u, c13);\n"
"        #undef ST_TC\n"
"    }\n"
"#else\n"
"    uint32_t tr_w = lid >> 2, gw = lid & 3u, qoff = gw * 4u, gr = r0 + tr_w;\n"
"    uint32_t tr_x = lid >> 3, cx4 = (lid & 7u) * 4u, gb = b0 + tr_x;\n"
"    const uint8_t* wrow = w + (uint64_t)(gr < out_dim ? gr : (out_dim - 1u)) * rb;\n"
"    __shared__ float sX[32][33]; __shared__ float sW[64][33];\n"
"    uint32_t th_b = (lid >> 4) * 2u, th_r = (lid & 15u) * 4u;\n"
"    float acc[2][4] = {{0,0,0,0},{0,0,0,0}};\n"
"    for (uint32_t bk = 0; bk < nb; bk++) {\n"
"        if (gb < batch_b) {\n"
"            float4 v0 = __ldg((const float4*)(x + (uint64_t)gb * in_dim + (bk << 5) + cx4));\n"
"            sX[tr_x][cx4+0]=v0.x; sX[tr_x][cx4+1]=v0.y; sX[tr_x][cx4+2]=v0.z; sX[tr_x][cx4+3]=v0.w;\n"
"        } else { for (uint32_t j = 0; j < 4u; j++) sX[tr_x][cx4 + j] = 0.0f; }\n"
"        if (gr < out_dim) {\n"
"            const uint8_t* blk = wrow + (uint64_t)bk * 20ull;\n"
"            float d = load_f16_le(blk), mv = load_f16_le(blk + 2u);\n"
"            for (uint32_t j = 0; j < 4u; j++) {\n"
"                uint8_t bv = __ldg(&blk[4u + qoff + j]);\n"
"                sW[tr_w][qoff + j] = fmaf((float)(bv & 0xFu), d, mv);\n"
"                sW[tr_w][16u + qoff + j] = fmaf((float)(bv >> 4), d, mv);\n"
"            }\n"
"        } else { for (uint32_t j = 0; j < 4u; j++) { sW[tr_w][qoff+j]=0.0f; sW[tr_w][16u+qoff+j]=0.0f; } }\n"
"        __syncthreads();\n"
"        for (uint32_t k = 0; k < 32u; k++) {\n"
"            float x0 = sX[th_b + 0u][k], x1 = sX[th_b + 1u][k];\n"
"            float w0 = sW[th_r + 0u][k], w1 = sW[th_r + 1u][k], w2 = sW[th_r + 2u][k], w3 = sW[th_r + 3u][k];\n"
"            acc[0][0] = fmaf(x0, w0, acc[0][0]); acc[0][1] = fmaf(x0, w1, acc[0][1]); acc[0][2] = fmaf(x0, w2, acc[0][2]); acc[0][3] = fmaf(x0, w3, acc[0][3]);\n"
"            acc[1][0] = fmaf(x1, w0, acc[1][0]); acc[1][1] = fmaf(x1, w1, acc[1][1]); acc[1][2] = fmaf(x1, w2, acc[1][2]); acc[1][3] = fmaf(x1, w3, acc[1][3]);\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (uint32_t tb = 0; tb < 2u; tb++) for (uint32_t rr = 0; rr < 4u; rr++) {\n"
"        uint32_t ob = b0 + th_b + tb, orow = r0 + th_r + rr;\n"
"        if (ob < batch_b && orow < out_dim) { uint64_t idx = (uint64_t)ob * out_dim + orow; y[idx] = add_res ? (y[idx] + acc[tb][rr]) : acc[tb][rr]; }\n"
"    }\n"
"#endif\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_q5_0_batch(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid, b0 = blockIdx.y * 8u;\n"
"    if (row >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t bt = (batch_b - b0 < 8u) ? (batch_b - b0) : 8u; uint32_t nb = in_dim >> 5;\n"
"    const uint8_t* wp = w + (uint64_t)row * (uint64_t)nb * 22ull; float acc[8] = {0,0,0,0,0,0,0,0};\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        const uint8_t* blk = wp + (uint64_t)b * 22ull;\n"
"        for (uint32_t t = 0; t < bt; t++) {\n"
"            float4 xv_lo[4], xv_hi[4]; float sx = load_x32_q4(x + (uint64_t)(b0 + t) * in_dim + (b << 5), xv_lo, xv_hi);\n"
"            acc[t] += dot_q5_0_fast(blk, xv_lo, xv_hi, sx);\n"
"        }\n"
"    }\n"
"    for (uint32_t t = 0; t < bt; t++) { float r = warp_sum(acc[t]); if (lane == 0u) { uint64_t idx = (uint64_t)(b0 + t) * out_dim + row; y[idx] = add_res ? (y[idx] + r) : r; } }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_q5_1_batch(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid, b0 = blockIdx.y * 8u;\n"
"    if (row >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t bt = (batch_b - b0 < 8u) ? (batch_b - b0) : 8u; uint32_t nb = in_dim >> 5;\n"
"    const uint8_t* wp = w + (uint64_t)row * (uint64_t)nb * 24ull; float acc[8] = {0,0,0,0,0,0,0,0};\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        const uint8_t* blk = wp + (uint64_t)b * 24ull;\n"
"        for (uint32_t t = 0; t < bt; t++) {\n"
"            float4 xv_lo[4], xv_hi[4]; float sx = load_x32_q4(x + (uint64_t)(b0 + t) * in_dim + (b << 5), xv_lo, xv_hi);\n"
"            acc[t] += dot_q5_1_fast(blk, xv_lo, xv_hi, sx);\n"
"        }\n"
"    }\n"
"    for (uint32_t t = 0; t < bt; t++) { float r = warp_sum(acc[t]); if (lane == 0u) { uint64_t idx = (uint64_t)(b0 + t) * out_dim + row; y[idx] = add_res ? (y[idx] + r) : r; } }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_q8_batch(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid, b0 = blockIdx.y * 8u;\n"
"    if (row >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t bt = (batch_b - b0 < 8u) ? (batch_b - b0) : 8u; uint32_t nb = in_dim >> 5;\n"
"    const uint8_t* wp = w + (uint64_t)row * (uint64_t)nb * 34ull; float acc[8] = {0,0,0,0,0,0,0,0};\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        const uint8_t* blk = wp + (uint64_t)b * 34ull; float d = load_f16_le(blk); const int8_t* qs = (const int8_t*)(blk + 2u);\n"
"        for (uint32_t t = 0; t < bt; t++) {\n"
"            const float* xt = x + (uint64_t)(b0 + t) * in_dim + (b << 5); float s = 0.0f;\n"
"            for (uint32_t i = 0; i < 32u; i++) s = fmaf((float)qs[i], xt[i], s);\n"
"            acc[t] += d * s;\n"
"        }\n"
"    }\n"
"    for (uint32_t t = 0; t < bt; t++) { float r = warp_sum(acc[t]); if (lane == 0u) { uint64_t idx = (uint64_t)(b0 + t) * out_dim + row; y[idx] = add_res ? (y[idx] + r) : r; } }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_q2k_batch(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid, b0 = blockIdx.y * 4u;\n"
"    if (row >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t bt = (batch_b - b0 < 4u) ? (batch_b - b0) : 4u; uint32_t nb = in_dim >> 8, qtr = lane & 3u;\n"
"    const uint8_t* wp = w + (uint64_t)row * (uint64_t)nb * 84ull; float acc[4] = {0,0,0,0};\n"
"    for (uint32_t b = (lane >> 2); b < nb; b += 8u) {\n"
"        const uint8_t* blk = wp + (uint64_t)b * 84ull;\n"
"        for (uint32_t t = 0; t < bt; t++) acc[t] += dot_q2_k_quarter(blk, x + (uint64_t)(b0 + t) * in_dim + (b << 8), qtr);\n"
"    }\n"
"    for (uint32_t t = 0; t < bt; t++) { float r = warp_sum(acc[t]); if (lane == 0u) { uint64_t idx = (uint64_t)(b0 + t) * out_dim + row; y[idx] = add_res ? (y[idx] + r) : r; } }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_q3k_batch(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid, b0 = blockIdx.y * 4u;\n"
"    if (row >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t bt = (batch_b - b0 < 4u) ? (batch_b - b0) : 4u; uint32_t nb = in_dim >> 8, qtr = lane & 3u;\n"
"    const uint8_t* wp = w + (uint64_t)row * (uint64_t)nb * 110ull; float acc[4] = {0,0,0,0};\n"
"    for (uint32_t b = (lane >> 2); b < nb; b += 8u) {\n"
"        const uint8_t* blk = wp + (uint64_t)b * 110ull;\n"
"        for (uint32_t t = 0; t < bt; t++) acc[t] += dot_q3_k_quarter(blk, x + (uint64_t)(b0 + t) * in_dim + (b << 8), qtr);\n"
"    }\n"
"    for (uint32_t t = 0; t < bt; t++) { float r = warp_sum(acc[t]); if (lane == 0u) { uint64_t idx = (uint64_t)(b0 + t) * out_dim + row; y[idx] = add_res ? (y[idx] + r) : r; } }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_iq4xs_batch(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid, b0 = blockIdx.y * 4u;\n"
"    if (row >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t bt = (batch_b - b0 < 4u) ? (batch_b - b0) : 4u; uint32_t nb = in_dim >> 8, qtr = lane & 3u;\n"
"    const uint8_t* wp = w + (uint64_t)row * (uint64_t)nb * 136ull; float acc[4] = {0,0,0,0};\n"
"    for (uint32_t b = (lane >> 2); b < nb; b += 8u) {\n"
"        const uint8_t* blk = wp + (uint64_t)b * 136ull;\n"
"        for (uint32_t t = 0; t < bt; t++) acc[t] += dot_iq4_xs_quarter(blk, x + (uint64_t)(b0 + t) * in_dim + (b << 8), qtr);\n"
"    }\n"
"    for (uint32_t t = 0; t < bt; t++) { float r = warp_sum(acc[t]); if (lane == 0u) { uint64_t idx = (uint64_t)(b0 + t) * out_dim + row; y[idx] = add_res ? (y[idx] + r) : r; } }\n"
"}\n"
"\n"
"// Hardware TF32 Tensor Core (`mma.sync.aligned.m16n8k8`) Batched Q4_K GEMM (32x32 tile, full FP32 exponent range!)\n"
"extern \"C\" __global__ void cm_gemm_q4k_batch(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t r0 = blockIdx.x * 32u, b0 = blockIdx.y * 32u;\n"
"    if (r0 >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t lid = threadIdx.x, nb = in_dim >> 8; uint64_t rb = (uint64_t)nb * 144ull;\n"
"    __shared__ float sX[32][33]; __shared__ float sW[32][33];\n"
"    uint32_t tr = lid >> 2, g = lid & 3u, c0 = g * 8u;\n"
"    uint32_t gb = b0 + tr, gr = r0 + tr;\n"
"    const uint8_t* wrow = w + (uint64_t)(gr < out_dim ? gr : (out_dim - 1u)) * rb;\n"
"#if __CUDA_ARCH__ >= 800\n"
"    uint32_t wid = lid >> 5, lane = lid & 31u, gid = lane >> 2, tid = lane & 3u;\n"
"    uint32_t wb = (wid >> 1) * 16u, wr = (wid & 1u) * 16u;\n"
"    int warp_act = (b0 + wb < batch_b);\n"
"    float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f, c03 = 0.0f;\n"
"    float c10 = 0.0f, c11 = 0.0f, c12 = 0.0f, c13 = 0.0f;\n"
"#else\n"
"    uint32_t th_b = (lid >> 3) * 2u, th_r = (lid & 7u) * 4u;\n"
"    float acc[2][4] = {{0,0,0,0},{0,0,0,0}};\n"
"#endif\n"
"    for (uint32_t bk = 0; bk < nb; bk++) {\n"
"        const uint8_t* blk = wrow + (uint64_t)bk * 144ull;\n"
"        float d = load_f16_le(blk), dm = load_f16_le(blk + 2u);\n"
"        const uint8_t* sc = blk + 4u; const uint8_t* qs = blk + 16u;\n"
"        for (uint32_t sb = 0; sb < 8u; sb++) {\n"
"            uint32_t kbase = (bk << 8) + (sb << 5);\n"
"            if (gb < batch_b) {\n"
"                const float4* xp = (const float4*)(x + (uint64_t)gb * in_dim + kbase + c0);\n"
"                float4 v0 = __ldg(&xp[0]), v1 = __ldg(&xp[1]);\n"
"                sX[tr][c0+0]=v0.x; sX[tr][c0+1]=v0.y; sX[tr][c0+2]=v0.z; sX[tr][c0+3]=v0.w;\n"
"                sX[tr][c0+4]=v1.x; sX[tr][c0+5]=v1.y; sX[tr][c0+6]=v1.z; sX[tr][c0+7]=v1.w;\n"
"            } else { for (uint32_t j = 0; j < 8u; j++) sX[tr][c0 + j] = 0.0f; }\n"
"            if (gr < out_dim) {\n"
"                uint8_t scv, mv;\n"
"                if (sb < 4u) { scv = sc[sb] & 63u; mv = sc[sb + 4u] & 63u; }\n"
"                else { scv = (sc[sb + 4u] & 0xFu) | ((sc[sb - 4u] >> 6) << 4); mv = (sc[sb + 4u] >> 4) | ((sc[sb] >> 6) << 4); }\n"
"                float dl = d * (float)scv, ml = dm * (float)mv;\n"
"                const uint8_t* qp = qs + (sb >> 1) * 32u + c0; uint32_t hi = sb & 1u;\n"
"                #pragma unroll\n"
"                for (uint32_t j = 0; j < 8u; j++) {\n"
"                    uint8_t bv = __ldg(&qp[j]);\n"
"                    sW[tr][c0 + j] = fmaf(dl, (float)(hi ? (bv >> 4) : (bv & 0xFu)), -ml);\n"
"                }\n"
"            } else { for (uint32_t j = 0; j < 8u; j++) sW[tr][c0 + j] = 0.0f; }\n"
"            __syncthreads();\n"
"#if __CUDA_ARCH__ >= 800\n"
"            if (warp_act) {\n"
"                #pragma unroll\n"
"                for (uint32_t kk = 0; kk < 32u; kk += 8u) {\n"
"                    uint32_t a0, a1, a2, a3, b00, b01, b10, b11;\n"
"                    asm(\"cvt.rna.tf32.f32 %0, %1;\" : \"=r\"(a0) : \"f\"(sX[wb + gid][kk + tid]));\n"
"                    asm(\"cvt.rna.tf32.f32 %0, %1;\" : \"=r\"(a1) : \"f\"(sX[wb + gid + 8u][kk + tid]));\n"
"                    asm(\"cvt.rna.tf32.f32 %0, %1;\" : \"=r\"(a2) : \"f\"(sX[wb + gid][kk + tid + 4u]));\n"
"                    asm(\"cvt.rna.tf32.f32 %0, %1;\" : \"=r\"(a3) : \"f\"(sX[wb + gid + 8u][kk + tid + 4u]));\n"
"                    asm(\"cvt.rna.tf32.f32 %0, %1;\" : \"=r\"(b00) : \"f\"(sW[wr + gid][kk + tid]));\n"
"                    asm(\"cvt.rna.tf32.f32 %0, %1;\" : \"=r\"(b01) : \"f\"(sW[wr + gid][kk + tid + 4u]));\n"
"                    asm volatile(\"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
"                                 : \"+f\"(c00), \"+f\"(c01), \"+f\"(c02), \"+f\"(c03) : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b00), \"r\"(b01));\n"
"                    asm(\"cvt.rna.tf32.f32 %0, %1;\" : \"=r\"(b10) : \"f\"(sW[wr + 8u + gid][kk + tid]));\n"
"                    asm(\"cvt.rna.tf32.f32 %0, %1;\" : \"=r\"(b11) : \"f\"(sW[wr + 8u + gid][kk + tid + 4u]));\n"
"                    asm volatile(\"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
"                                 : \"+f\"(c10), \"+f\"(c11), \"+f\"(c12), \"+f\"(c13) : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b10), \"r\"(b11));\n"
"                }\n"
"            }\n"
"#else\n"
"            for (uint32_t k = 0; k < 32u; k++) {\n"
"                float x0 = sX[th_b + 0u][k], x1 = sX[th_b + 1u][k];\n"
"                float w0 = sW[th_r + 0u][k], w1 = sW[th_r + 1u][k], w2 = sW[th_r + 2u][k], w3 = sW[th_r + 3u][k];\n"
"                acc[0][0]=fmaf(x0,w0,acc[0][0]); acc[0][1]=fmaf(x0,w1,acc[0][1]); acc[0][2]=fmaf(x0,w2,acc[0][2]); acc[0][3]=fmaf(x0,w3,acc[0][3]);\n"
"                acc[1][0]=fmaf(x1,w0,acc[1][0]); acc[1][1]=fmaf(x1,w1,acc[1][1]); acc[1][2]=fmaf(x1,w2,acc[1][2]); acc[1][3]=fmaf(x1,w3,acc[1][3]);\n"
"            }\n"
"#endif\n"
"            __syncthreads();\n"
"        }\n"
"    }\n"
"#if __CUDA_ARCH__ >= 800\n"
"    if (warp_act) {\n"
"        uint32_t ob0 = b0 + wb + gid, ob1 = ob0 + 8u, or0 = r0 + wr + tid * 2u, or1 = or0 + 8u;\n"
"        #define ST_TC(ob, orw, val) if ((ob) < batch_b && (orw) < out_dim) { uint64_t idx = (uint64_t)(ob) * out_dim + (orw); y[idx] = add_res ? (y[idx] + (val)) : (val); }\n"
"        ST_TC(ob0, or0, c00); ST_TC(ob0, or0 + 1u, c01); ST_TC(ob1, or0, c02); ST_TC(ob1, or0 + 1u, c03);\n"
"        ST_TC(ob0, or1, c10); ST_TC(ob0, or1 + 1u, c11); ST_TC(ob1, or1, c12); ST_TC(ob1, or1 + 1u, c13);\n"
"        #undef ST_TC\n"
"    }\n"
"#else\n"
"    for (uint32_t tb = 0; tb < 2u; tb++) for (uint32_t rr = 0; rr < 4u; rr++) {\n"
"        uint32_t ob = b0 + th_b + tb, orow = r0 + th_r + rr;\n"
"        if (ob < batch_b && orow < out_dim) { uint64_t idx = (uint64_t)ob * out_dim + orow; y[idx] = add_res ? (y[idx] + acc[tb][rr]) : acc[tb][rr]; }\n"
"    }\n"
"#endif\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_q5k_batch(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid, b0 = blockIdx.y * 4u;\n"
"    if (row >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t bt = (batch_b - b0 < 4u) ? (batch_b - b0) : 4u; uint32_t nb = in_dim >> 8, qtr = lane & 3u;\n"
"    const uint8_t* wp = w + (uint64_t)row * (uint64_t)nb * 176ull; float acc[4] = {0,0,0,0};\n"
"    for (uint32_t b = (lane >> 2); b < nb; b += 8u) {\n"
"        const uint8_t* blk = wp + (uint64_t)b * 176ull;\n"
"        for (uint32_t t = 0; t < bt; t++) acc[t] += dot_q5_k_quarter(blk, x + (uint64_t)(b0 + t) * in_dim + (b << 8), qtr);\n"
"    }\n"
"    for (uint32_t t = 0; t < bt; t++) { float r = warp_sum(acc[t]); if (lane == 0u) { uint64_t idx = (uint64_t)(b0 + t) * out_dim + row; y[idx] = add_res ? (y[idx] + r) : r; } }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_q6k_batch(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid, b0 = blockIdx.y * 4u;\n"
"    if (row >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t bt = (batch_b - b0 < 4u) ? (batch_b - b0) : 4u; uint32_t nb = in_dim >> 8, qtr = lane & 3u;\n"
"    const uint8_t* wp = w + (uint64_t)row * (uint64_t)nb * 210ull; float acc[4] = {0,0,0,0};\n"
"    for (uint32_t b = (lane >> 2); b < nb; b += 8u) {\n"
"        const uint8_t* blk = wp + (uint64_t)b * 210ull;\n"
"        for (uint32_t t = 0; t < bt; t++) acc[t] += dot_q6_k_quarter(blk, x + (uint64_t)(b0 + t) * in_dim + (b << 8), qtr);\n"
"    }\n"
"    for (uint32_t t = 0; t < bt; t++) { float r = warp_sum(acc[t]); if (lane == 0u) { uint64_t idx = (uint64_t)(b0 + t) * out_dim + row; y[idx] = add_res ? (y[idx] + r) : r; } }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_f16_batch(const half* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid, b0 = blockIdx.y * 8u;\n"
"    if (row >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t bt = (batch_b - b0 < 8u) ? (batch_b - b0) : 8u; uint32_t n2 = in_dim >> 1;\n"
"    const half2* wp = (const half2*)(w + (uint64_t)row * in_dim); float acc[8] = {0,0,0,0,0,0,0,0};\n"
"    for (uint32_t i = lane; i < n2; i += 32u) {\n"
"        float2 wf = __half22float2(__ldg(&wp[i]));\n"
"        for (uint32_t t = 0; t < bt; t++) { float2 xf = __ldg(&((const float2*)(x + (uint64_t)(b0 + t) * in_dim))[i]); acc[t] = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, acc[t])); }\n"
"    }\n"
"    for (uint32_t t = 0; t < bt; t++) { float r = warp_sum(acc[t]); if (lane == 0u) { uint64_t idx = (uint64_t)(b0 + t) * out_dim + row; y[idx] = add_res ? (y[idx] + r) : r; } }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_f32_batch(const float* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid, b0 = blockIdx.y * 8u;\n"
"    if (row >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t bt = (batch_b - b0 < 8u) ? (batch_b - b0) : 8u; uint32_t n4 = in_dim >> 2;\n"
"    const float4* wp = (const float4*)(w + (uint64_t)row * in_dim); float acc[8] = {0,0,0,0,0,0,0,0};\n"
"    for (uint32_t i = lane; i < n4; i += 32u) {\n"
"        float4 wf = __ldg(&wp[i]);\n"
"        for (uint32_t t = 0; t < bt; t++) { float4 xf = __ldg(&((const float4*)(x + (uint64_t)(b0 + t) * in_dim))[i]); acc[t] = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, fmaf(wf.z, xf.z, fmaf(wf.w, xf.w, acc[t])))); }\n"
"    }\n"
"    for (uint32_t t = 0; t < bt; t++) { float r = warp_sum(acc[t]); if (lane == 0u) { uint64_t idx = (uint64_t)(b0 + t) * out_dim + row; y[idx] = add_res ? (y[idx] + r) : r; } }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_mla_split_kv(const float* kv_b_out, const float* k_pe, float* k_out, float* v_out,\n"
"                                           uint32_t n_heads, uint32_t nope_d, uint32_t rope_d, uint32_t v_d, uint32_t d_head, uint32_t batch_b) {\n"
"    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x, h = blockIdx.y, b = blockIdx.z;\n"
"    if (d >= d_head || h >= n_heads || b >= batch_b) return;\n"
"    uint64_t kb_base = ((uint64_t)b * n_heads + h) * (nope_d + v_d);\n"
"    uint64_t out_base = ((uint64_t)b * n_heads + h) * d_head;\n"
"    if (d < nope_d) k_out[out_base + d] = kv_b_out[kb_base + d];\n"
"    else if (d < nope_d + rope_d) k_out[out_base + d] = k_pe[(uint64_t)b * rope_d + (d - nope_d)];\n"
"    else k_out[out_base + d] = 0.0f;\n"
"    v_out[out_base + d] = (d < v_d) ? kv_b_out[kb_base + nope_d + d] : 0.0f;\n"
"}\n"
"\n"
"// =========================================================================\n"
"// True Matrix-Absorbed Compressed Latent MLA KV-Cache (c_kv || k_pe, ZERO V-Cache!)\n"
"// =========================================================================\n"
"extern \"C\" __global__ void cm_mla_rope_pe(float* q, float* k_pe, const float* rope_freqs,\n"
"                                          uint32_t n_heads, uint32_t d_head, uint32_t nope_d, uint32_t rope_d,\n"
"                                          uint32_t start_pos, float theta, uint32_t rope_neox, uint32_t has_freqs, uint32_t batch_b) {\n"
"    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x, h = blockIdx.y, b = blockIdx.z;\n"
"    uint32_t half_r = rope_d >> 1;\n"
"    if (i >= half_r || h > n_heads || b >= batch_b) return;\n"
"    uint32_t pos = start_pos + b;\n"
"    uint32_t idx0 = rope_neox ? i : (2u * i), idx1 = rope_neox ? (i + half_r) : (2u * i + 1u);\n"
"    float inv_f = 1.0f / powf(theta, (float)(2u * i) / (float)rope_d);\n"
"    if (has_freqs) inv_f /= rope_freqs[i];\n"
"    float ang = (float)pos * inv_f, cs = cosf(ang), sn = sinf(ang);\n"
"    if (h < n_heads) {\n"
"        uint64_t qoff = ((uint64_t)b * n_heads + h) * d_head + nope_d;\n"
"        float q0 = q[qoff + idx0], q1 = q[qoff + idx1];\n"
"        q[qoff + idx0] = q0 * cs - q1 * sn; q[qoff + idx1] = q0 * sn + q1 * cs;\n"
"    } else {\n"
"        uint64_t koff = (uint64_t)b * rope_d;\n"
"        float k0 = k_pe[koff + idx0], k1 = k_pe[koff + idx1];\n"
"        k_pe[koff + idx0] = k0 * cs - k1 * sn; k_pe[koff + idx1] = k0 * sn + k1 * cs;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_mla_store_latent(const float* c_kv, const float* k_pe, half* latent_cache,\n"
"                                               uint32_t kv_r, uint32_t rope_d, uint32_t start_pos, uint32_t batch_b) {\n"
"    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y;\n"
"    uint32_t ldim = kv_r + rope_d;\n"
"    if (d >= ldim || b >= batch_b) return;\n"
"    uint64_t dst = (uint64_t)(start_pos + b) * ldim + d;\n"
"    float v = (d < kv_r) ? c_kv[(uint64_t)b * ldim + d] : k_pe[(uint64_t)b * rope_d + (d - kv_r)];\n"
"    latent_cache[dst] = __float2half(v);\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_mla_unpack_wkv_b(const uint8_t* wkv_b, half* wuk_f16, half* wuv_f16,\n"
"                                               uint32_t gtype, uint32_t n_heads, uint32_t nope_d, uint32_t v_d, uint32_t kv_r) {\n"
"    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y, h = blockIdx.z;\n"
"    if (r >= kv_r || h >= n_heads) return;\n"
"    if (j < nope_d) {\n"
"        uint32_t row = h * (nope_d + v_d) + j;\n"
"        wuk_f16[((uint64_t)h * kv_r + r) * nope_d + j] = __float2half(dequant_any_elem(wkv_b, gtype, row, r, kv_r));\n"
"    }\n"
"    if (j < v_d) {\n"
"        uint32_t row = h * (nope_d + v_d) + nope_d + j;\n"
"        wuv_f16[((uint64_t)h * v_d + j) * kv_r + r] = __float2half(dequant_any_elem(wkv_b, gtype, row, r, kv_r));\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_mla_unpack_kb_vb(const uint8_t* wk_b, const uint8_t* wv_b, half* wuk_f16, half* wuv_f16,\n"
"                                               uint32_t gt_k, uint32_t gt_v, uint32_t k_trans, uint32_t v_trans,\n"
"                                               uint32_t n_heads, uint32_t nope_d, uint32_t v_d, uint32_t kv_r) {\n"
"    uint32_t r = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y, h = blockIdx.z;\n"
"    if (r >= kv_r || h >= n_heads) return;\n"
"    if (j < nope_d) {\n"
"        float val = k_trans ? dequant_any_elem(wk_b, gt_k, h * nope_d + j, r, kv_r)\n"
"                            : dequant_any_elem(wk_b, gt_k, h * kv_r + r, j, nope_d);\n"
"        wuk_f16[((uint64_t)h * kv_r + r) * nope_d + j] = __float2half(val);\n"
"    }\n"
"    if (j < v_d) {\n"
"        float val = v_trans ? dequant_any_elem(wv_b, gt_v, h * kv_r + r, j, v_d)\n"
"                            : dequant_any_elem(wv_b, gt_v, h * v_d + j, r, kv_r);\n"
"        wuv_f16[((uint64_t)h * v_d + j) * kv_r + r] = __float2half(val);\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_mla_absorb_q(const float* q, const half* wuk_f16, float* q_abs,\n"
"                                           uint32_t n_heads, uint32_t d_head, uint32_t nope_d, uint32_t rope_d, uint32_t kv_r, uint32_t batch_b) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t r = blockIdx.x * 4u + wid, h = blockIdx.y, b = blockIdx.z;\n"
"    uint32_t ldim = kv_r + rope_d;\n"
"    if (r >= ldim || h >= n_heads || b >= batch_b) return;\n"
"    uint64_t qbase = ((uint64_t)b * n_heads + h) * d_head;\n"
"    uint64_t obase = ((uint64_t)b * n_heads + h) * ldim;\n"
"    if (r >= kv_r) { if (lane == 0u) q_abs[obase + r] = q[qbase + nope_d + (r - kv_r)]; return; }\n"
"    const half* wp = wuk_f16 + ((uint64_t)h * kv_r + r) * nope_d;\n"
"    float acc = 0.0f;\n"
"    for (uint32_t j = lane; j < nope_d; j += 32u) acc = fmaf(q[qbase + j], __half2float(wp[j]), acc);\n"
"    acc = warp_sum(acc);\n"
"    if (lane == 0u) q_abs[obase + r] = acc;\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_flash_mla_latent(const float* q_abs, const half* latent_cache, float* o_lat,\n"
"                                               uint32_t n_heads, uint32_t kv_r, uint32_t rope_d, float scale, uint32_t start_pos, uint32_t batch_b) {\n"
"    uint32_t h = blockIdx.x, b = blockIdx.y;\n"
"    if (h >= n_heads || b >= batch_b) return;\n"
"    uint32_t kv_len = start_pos + b + 1u, ldim = kv_r + rope_d;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    __shared__ float s_q[768], s_m[4], s_l[4], s_o[4][576];\n"
"    uint64_t qbase = ((uint64_t)b * n_heads + h) * ldim;\n"
"    for (uint32_t d = lid; d < ldim && d < 768u; d += 128u) s_q[d] = q_abs[qbase + d];\n"
"    __syncthreads();\n"
"    float m_run = -1e30f, l_run = 0.0f, o_acc[18];\n"
"    uint32_t dpl = (kv_r + 31u) >> 5;\n"
"    for (uint32_t k = 0; k < dpl && k < 18u; k++) o_acc[k] = 0.0f;\n"
"    uint32_t lpl = (ldim + 31u) >> 5;\n"
"    for (uint32_t t = wid; t < kv_len; t += 4u) {\n"
"        const half* lp = latent_cache + (uint64_t)t * ldim;\n"
"        float dot = 0.0f;\n"
"        for (uint32_t k = 0; k < lpl; k++) { uint32_t d = lane + k * 32u; if (d < ldim) dot = fmaf(s_q[d], __half2float(lp[d]), dot); }\n"
"        float sc = warp_sum(dot) * scale; sc = __shfl_sync(0xFFFFFFFFu, sc, 0);\n"
"        float mn = fmaxf(m_run, sc), al = expf(m_run - mn), wt = expf(sc - mn);\n"
"        l_run = fmaf(l_run, al, wt); m_run = mn;\n"
"        for (uint32_t k = 0; k < dpl && k < 18u; k++) { uint32_t d = lane + k * 32u; if (d < kv_r) o_acc[k] = fmaf(o_acc[k], al, wt * __half2float(lp[d])); }\n"
"    }\n"
"    if (lane == 0u) { s_m[wid] = m_run; s_l[wid] = l_run; }\n"
"    for (uint32_t k = 0; k < dpl && k < 18u; k++) { uint32_t d = lane + k * 32u; if (d < kv_r && d < 576u) s_o[wid][d] = o_acc[k]; }\n"
"    __syncthreads();\n"
"    float gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3])), gl = 0.0f;\n"
"    for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * expf(s_m[s] - gm);\n"
"    float inv = 1.0f / fmaxf(gl, 1e-12f);\n"
"    uint64_t obase = ((uint64_t)b * n_heads + h) * kv_r;\n"
"    for (uint32_t d = lid; d < kv_r && d < 576u; d += 128u) {\n"
"        float ov = 0.0f; for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fmaf(s_o[s][d], expf(s_m[s] - gm), ov);\n"
"        o_lat[obase + d] = ov * inv;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_mla_proj_v(const float* o_lat, const half* wuv_f16, float* ctx,\n"
"                                         uint32_t n_heads, uint32_t kv_r, uint32_t v_d, uint32_t out_stride, uint32_t batch_b) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t j = blockIdx.x * 4u + wid, h = blockIdx.y, b = blockIdx.z;\n"
"    if (j >= out_stride || h >= n_heads || b >= batch_b) return;\n"
"    uint64_t obase = ((uint64_t)b * n_heads + h) * out_stride;\n"
"    if (j >= v_d) { if (lane == 0u) ctx[obase + j] = 0.0f; return; }\n"
"    const float* op = o_lat + ((uint64_t)b * n_heads + h) * kv_r;\n"
"    const half* wp = wuv_f16 + ((uint64_t)h * v_d + j) * kv_r;\n"
"    float acc = 0.0f;\n"
"    for (uint32_t r = lane; r < kv_r; r += 32u) acc = fmaf(op[r], __half2float(wp[r]), acc);\n"
"    acc = warp_sum(acc);\n"
"    if (lane == 0u) ctx[obase + j] = acc;\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_tp_accum_add(float* dst, const float* a, const float* b, uint32_t n) {\n"
"    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) dst[i] += a[i] + b[i];\n"
"}\n"
"\n"
"// =========================================================================\n"
"// Tenon v6: PagedAttention 16-Token Block-Table Kernels + Tree-Attention + ViT\n"
"// =========================================================================\n"
"extern \"C\" __global__ void cm_rope_kv_paged(float* q, float* k, const float* v, half* k_cache, half* v_cache, const float* rope_freqs,\n"
"                                            const uint32_t* pos_arr, const uint32_t* slot_arr, const int32_t* block_table,\n"
"                                            uint32_t n_heads, uint32_t n_kv, uint32_t d_head, uint32_t rope_dim,\n"
"                                            float theta, uint32_t rope_neox, uint32_t has_freqs, uint32_t batch_b, uint32_t max_blks) {\n"
"    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x, h = blockIdx.y, b = blockIdx.z;\n"
"    uint32_t half_d = d_head >> 1, half_r = rope_dim >> 1;\n"
"    if (i >= half_d || h >= n_heads || b >= batch_b) return;\n"
"    uint32_t pos = pos_arr[b], slot = slot_arr[b];\n"
"    uint32_t pb = (uint32_t)block_table[(uint64_t)slot * max_blks + (pos >> 4)];\n"
"    uint32_t phys_pos = (pb << 4) | (pos & 15u);\n"
"    uint64_t qoff = (uint64_t)b * (n_heads * d_head) + h * d_head;\n"
"    uint64_t koff = (uint64_t)b * (n_kv * d_head) + h * d_head;\n"
"    if (i < half_r) {\n"
"        uint32_t idx0 = rope_neox ? i : (2u * i), idx1 = rope_neox ? (i + half_r) : (2u * i + 1u);\n"
"        float inv_f = 1.0f / powf(theta, (float)(2u * i) / (float)rope_dim);\n"
"        if (has_freqs) inv_f /= rope_freqs[i];\n"
"        float ang = (float)pos * inv_f, cs = cosf(ang), sn = sinf(ang);\n"
"        float q0 = q[qoff + idx0], q1 = q[qoff + idx1];\n"
"        q[qoff + idx0] = q0 * cs - q1 * sn; q[qoff + idx1] = q0 * sn + q1 * cs;\n"
"        if (h < n_kv) {\n"
"            float k0 = k[koff + idx0], k1 = k[koff + idx1];\n"
"            float rk0 = k0 * cs - k1 * sn, rk1 = k0 * sn + k1 * cs;\n"
"            k[koff + idx0] = rk0; k[koff + idx1] = rk1;\n"
"            uint64_t cb = (uint64_t)phys_pos * (n_kv * d_head) + h * d_head;\n"
"            k_cache[cb + idx0] = __float2half(rk0); k_cache[cb + idx1] = __float2half(rk1);\n"
"            v_cache[cb + idx0] = __float2half(v[koff + idx0]); v_cache[cb + idx1] = __float2half(v[koff + idx1]);\n"
"        }\n"
"    } else if (h < n_kv) {\n"
"        uint32_t idx0 = 2u * i, idx1 = 2u * i + 1u;\n"
"        uint64_t cb = (uint64_t)phys_pos * (n_kv * d_head) + h * d_head;\n"
"        k_cache[cb + idx0] = __float2half(k[koff + idx0]); k_cache[cb + idx1] = __float2half(k[koff + idx1]);\n"
"        v_cache[cb + idx0] = __float2half(v[koff + idx0]); v_cache[cb + idx1] = __float2half(v[koff + idx1]);\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_rope_kv_q8_paged(float* q, float* k, const float* v, uint8_t* k_cache, uint8_t* v_cache, const float* rope_freqs,\n"
"                                               const uint32_t* pos_arr, const uint32_t* slot_arr, const int32_t* block_table,\n"
"                                               uint32_t n_heads, uint32_t n_kv, uint32_t d_head, uint32_t rope_dim,\n"
"                                               float theta, uint32_t rope_neox, uint32_t has_freqs, uint32_t batch_b, uint32_t max_blks) {\n"
"    uint32_t h = blockIdx.x, b = blockIdx.y; if (h >= n_heads || b >= batch_b) return;\n"
"    uint32_t pos = pos_arr[b], slot = slot_arr[b];\n"
"    uint32_t pb = (uint32_t)block_table[(uint64_t)slot * max_blks + (pos >> 4)];\n"
"    uint32_t phys_pos = (pb << 4) | (pos & 15u), half_r = rope_dim >> 1;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint64_t qbase = (uint64_t)b * (n_heads * d_head) + h * d_head;\n"
"    uint64_t kbase = (uint64_t)b * (n_kv * d_head) + h * d_head;\n"
"    for (uint32_t i = lid; i < half_r; i += 128u) {\n"
"        uint32_t idx0 = rope_neox ? i : (2u * i), idx1 = rope_neox ? (i + half_r) : (2u * i + 1u);\n"
"        float inv_f = 1.0f / powf(theta, (float)(2u * i) / (float)rope_dim);\n"
"        if (has_freqs) inv_f /= rope_freqs[i];\n"
"        float ang = (float)pos * inv_f, cs = cosf(ang), sn = sinf(ang);\n"
"        float q0 = q[qbase + idx0], q1 = q[qbase + idx1];\n"
"        q[qbase + idx0] = q0 * cs - q1 * sn; q[qbase + idx1] = q0 * sn + q1 * cs;\n"
"        if (h < n_kv) {\n"
"            float k0 = k[kbase + idx0], k1 = k[kbase + idx1];\n"
"            k[kbase + idx0] = k0 * cs - k1 * sn; k[kbase + idx1] = k0 * sn + k1 * cs;\n"
"        }\n"
"    }\n"
"    __syncthreads();\n"
"    if (h < n_kv) {\n"
"        uint32_t nblk = d_head >> 5;\n"
"        uint64_t row_bytes = (uint64_t)n_kv * (uint64_t)nblk * 34ull;\n"
"        uint64_t hoff = (uint64_t)phys_pos * row_bytes + (uint64_t)h * (uint64_t)nblk * 34ull;\n"
"        for (uint32_t blk = wid; blk < nblk; blk += 4u) {\n"
"            float kv = k[kbase + blk * 32u + lane], vv = v[kbase + blk * 32u + lane];\n"
"            float kam = fabsf(kv), vam = fabsf(vv);\n"
"            #pragma unroll\n"
"            for (int off = 16; off > 0; off >>= 1) { kam = fmaxf(kam, __shfl_xor_sync(0xFFFFFFFFu, kam, off)); vam = fmaxf(vam, __shfl_xor_sync(0xFFFFFFFFu, vam, off)); }\n"
"            float kd = kam / 127.0f, kid = (kam > 0.0f) ? (127.0f / kam) : 0.0f;\n"
"            float vd = vam / 127.0f, vid = (vam > 0.0f) ? (127.0f / vam) : 0.0f;\n"
"            uint8_t* kp = k_cache + hoff + (uint64_t)blk * 34ull;\n"
"            uint8_t* vp = v_cache + hoff + (uint64_t)blk * 34ull;\n"
"            if (lane == 0u) { *(uint16_t*)kp = __half_as_ushort(__float2half(kd)); *(uint16_t*)vp = __half_as_ushort(__float2half(vd)); }\n"
"            ((int8_t*)(kp + 2u))[lane] = (int8_t)__float2int_rn(kv * kid);\n"
"            ((int8_t*)(vp + 2u))[lane] = (int8_t)__float2int_rn(vv * vid);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_flash_decode_paged(const float* q, const half* k_cache, const half* v_cache, float* ctx,\n"
"                                                 const uint32_t* pos_arr, const uint32_t* slot_arr, const int32_t* block_table,\n"
"                                                 uint32_t n_heads, uint32_t n_kv, uint32_t d_head, uint32_t batch_b, float scale, uint32_t max_blks) {\n"
"    uint32_t h = blockIdx.x, b = blockIdx.y; if (h >= n_heads || b >= batch_b) return;\n"
"    uint32_t kv_len = pos_arr[b] + 1u, slot = slot_arr[b];\n"
"    const int32_t* bt = block_table + (uint64_t)slot * max_blks;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint32_t grp = (n_heads / n_kv) > 0u ? (n_heads / n_kv) : 1u, kvh = h / grp;\n"
"    uint32_t kv_stride = n_kv * d_head;\n"
"    __shared__ float s_q[256], s_m[4], s_l[4], s_o[4][256];\n"
"    uint64_t qbase = (uint64_t)b * (n_heads * d_head) + h * d_head;\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) s_q[d] = q[qbase + d];\n"
"    __syncthreads();\n"
"    float m_run = -1e30f, l_run = 0.0f, o_acc[8] = {0,0,0,0,0,0,0,0}; uint32_t dpl = (d_head + 31u) >> 5;\n"
"    for (uint32_t t = wid; t < kv_len; t += 4u) {\n"
"        uint32_t pt = ((uint32_t)bt[t >> 4] << 4) | (t & 15u);\n"
"        const half* kp = k_cache + (uint64_t)pt * kv_stride + kvh * d_head;\n"
"        const half* vp = v_cache + (uint64_t)pt * kv_stride + kvh * d_head;\n"
"        float dot = 0.0f;\n"
"        for (uint32_t k = 0; k < dpl; k++) { uint32_t d = lane + k * 32u; if (d < d_head) dot = fmaf(s_q[d], __half2float(kp[d]), dot); }\n"
"        float sc = warp_sum(dot) * scale; sc = __shfl_sync(0xFFFFFFFFu, sc, 0);\n"
"        float mn = fmaxf(m_run, sc), al = expf(m_run - mn), wt = expf(sc - mn);\n"
"        l_run = fmaf(l_run, al, wt); m_run = mn;\n"
"        for (uint32_t k = 0; k < dpl; k++) { uint32_t d = lane + k * 32u; if (d < d_head) o_acc[k] = fmaf(o_acc[k], al, wt * __half2float(vp[d])); }\n"
"    }\n"
"    if (lane == 0u) { s_m[wid] = m_run; s_l[wid] = l_run; }\n"
"    for (uint32_t k = 0; k < dpl; k++) { uint32_t d = lane + k * 32u; if (d < d_head) s_o[wid][d] = o_acc[k]; }\n"
"    __syncthreads();\n"
"    float gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3])), gl = 0.0f;\n"
"    for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * expf(s_m[s] - gm);\n"
"    float inv = 1.0f / fmaxf(gl, 1e-12f);\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) {\n"
"        float ov = 0.0f; for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fmaf(s_o[s][d], expf(s_m[s] - gm), ov);\n"
"        ctx[qbase + d] = ov * inv;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_flash_decode_q8_paged(const float* q, const uint8_t* k_cache, const uint8_t* v_cache, float* ctx,\n"
"                                                    const uint32_t* pos_arr, const uint32_t* slot_arr, const int32_t* block_table,\n"
"                                                    uint32_t n_heads, uint32_t n_kv, uint32_t d_head, uint32_t batch_b, float scale, uint32_t max_blks) {\n"
"    uint32_t h = blockIdx.x, b = blockIdx.y; if (h >= n_heads || b >= batch_b) return;\n"
"    uint32_t kv_len = pos_arr[b] + 1u, slot = slot_arr[b];\n"
"    const int32_t* bt = block_table + (uint64_t)slot * max_blks;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint32_t grp = (n_heads / n_kv) > 0u ? (n_heads / n_kv) : 1u, kvh = h / grp;\n"
"    uint32_t nblk = d_head >> 5; uint64_t row_bytes = (uint64_t)n_kv * (uint64_t)nblk * 34ull;\n"
"    __shared__ float s_q[256], s_m[4], s_l[4], s_o[4][256];\n"
"    uint64_t qbase = (uint64_t)b * (n_heads * d_head) + h * d_head;\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) s_q[d] = q[qbase + d];\n"
"    __syncthreads();\n"
"    float m_run = -1e30f, l_run = 0.0f, o_acc[8] = {0,0,0,0,0,0,0,0};\n"
"    for (uint32_t t = wid; t < kv_len; t += 4u) {\n"
"        uint32_t pt = ((uint32_t)bt[t >> 4] << 4) | (t & 15u);\n"
"        uint64_t hoff = (uint64_t)pt * row_bytes + (uint64_t)kvh * (uint64_t)nblk * 34ull;\n"
"        const uint8_t* kp = k_cache + hoff; const uint8_t* vp = v_cache + hoff;\n"
"        float dot = 0.0f;\n"
"        for (uint32_t blk = 0; blk < nblk; blk++) {\n"
"            const uint8_t* kb = kp + (uint64_t)blk * 34ull;\n"
"            dot = fmaf(s_q[blk * 32u + lane], (float)((const int8_t*)(kb + 2u))[lane] * load_f16_le(kb), dot);\n"
"        }\n"
"        float sc = warp_sum(dot) * scale; sc = __shfl_sync(0xFFFFFFFFu, sc, 0);\n"
"        float mn = fmaxf(m_run, sc), al = expf(m_run - mn), wt = expf(sc - mn);\n"
"        l_run = fmaf(l_run, al, wt); m_run = mn;\n"
"        for (uint32_t blk = 0; blk < nblk; blk++) {\n"
"            const uint8_t* vb = vp + (uint64_t)blk * 34ull;\n"
"            o_acc[blk] = fmaf(o_acc[blk], al, wt * (float)((const int8_t*)(vb + 2u))[lane] * load_f16_le(vb));\n"
"        }\n"
"    }\n"
"    if (lane == 0u) { s_m[wid] = m_run; s_l[wid] = l_run; }\n"
"    for (uint32_t blk = 0; blk < nblk; blk++) s_o[wid][blk * 32u + lane] = o_acc[blk];\n"
"    __syncthreads();\n"
"    float gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3])), gl = 0.0f;\n"
"    for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * expf(s_m[s] - gm);\n"
"    float inv = 1.0f / fmaxf(gl, 1e-12f);\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) {\n"
"        float ov = 0.0f; for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fmaf(s_o[s][d], expf(s_m[s] - gm), ov);\n"
"        ctx[qbase + d] = ov * inv;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_rope_kv_tree(float* q, float* k, const float* v, half* k_cache, half* v_cache, const float* rope_freqs,\n"
"                                           const uint32_t* tree_depth, uint32_t n_heads, uint32_t n_kv, uint32_t d_head, uint32_t rope_dim,\n"
"                                           uint32_t start_pos, float theta, uint32_t rope_neox, uint32_t has_freqs, uint32_t n_nodes) {\n"
"    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x, h = blockIdx.y, b = blockIdx.z;\n"
"    uint32_t half_d = d_head >> 1, half_r = rope_dim >> 1;\n"
"    if (i >= half_d || h >= n_heads || b >= n_nodes) return;\n"
"    uint32_t rot_pos = start_pos + tree_depth[b], wr_pos = start_pos + b;\n"
"    uint64_t qoff = (uint64_t)b * (n_heads * d_head) + h * d_head;\n"
"    uint64_t koff = (uint64_t)b * (n_kv * d_head) + h * d_head;\n"
"    if (i < half_r) {\n"
"        uint32_t idx0 = rope_neox ? i : (2u * i), idx1 = rope_neox ? (i + half_r) : (2u * i + 1u);\n"
"        float inv_f = 1.0f / powf(theta, (float)(2u * i) / (float)rope_dim);\n"
"        if (has_freqs) inv_f /= rope_freqs[i];\n"
"        float ang = (float)rot_pos * inv_f, cs = cosf(ang), sn = sinf(ang);\n"
"        float q0 = q[qoff + idx0], q1 = q[qoff + idx1];\n"
"        q[qoff + idx0] = q0 * cs - q1 * sn; q[qoff + idx1] = q0 * sn + q1 * cs;\n"
"        if (h < n_kv) {\n"
"            float k0 = k[koff + idx0], k1 = k[koff + idx1];\n"
"            float rk0 = k0 * cs - k1 * sn, rk1 = k0 * sn + k1 * cs;\n"
"            k[koff + idx0] = rk0; k[koff + idx1] = rk1;\n"
"            uint64_t cb = (uint64_t)wr_pos * (n_kv * d_head) + h * d_head;\n"
"            k_cache[cb + idx0] = __float2half(rk0); k_cache[cb + idx1] = __float2half(rk1);\n"
"            v_cache[cb + idx0] = __float2half(v[koff + idx0]); v_cache[cb + idx1] = __float2half(v[koff + idx1]);\n"
"        }\n"
"    } else if (h < n_kv) {\n"
"        uint32_t idx0 = 2u * i, idx1 = 2u * i + 1u;\n"
"        uint64_t cb = (uint64_t)wr_pos * (n_kv * d_head) + h * d_head;\n"
"        k_cache[cb + idx0] = __float2half(k[koff + idx0]); k_cache[cb + idx1] = __float2half(k[koff + idx1]);\n"
"        v_cache[cb + idx0] = __float2half(v[koff + idx0]); v_cache[cb + idx1] = __float2half(v[koff + idx1]);\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_rope_kv_q8_tree(float* q, float* k, const float* v, uint8_t* k_cache, uint8_t* v_cache, const float* rope_freqs,\n"
"                                              const uint32_t* tree_depth, uint32_t n_heads, uint32_t n_kv, uint32_t d_head, uint32_t rope_dim,\n"
"                                              uint32_t start_pos, float theta, uint32_t rope_neox, uint32_t has_freqs, uint32_t n_nodes) {\n"
"    uint32_t h = blockIdx.x, b = blockIdx.y; if (h >= n_heads || b >= n_nodes) return;\n"
"    uint32_t rot_pos = start_pos + tree_depth[b], wr_pos = start_pos + b, half_r = rope_dim >> 1;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint64_t qbase = (uint64_t)b * (n_heads * d_head) + h * d_head;\n"
"    uint64_t kbase = (uint64_t)b * (n_kv * d_head) + h * d_head;\n"
"    for (uint32_t i = lid; i < half_r; i += 128u) {\n"
"        uint32_t idx0 = rope_neox ? i : (2u * i), idx1 = rope_neox ? (i + half_r) : (2u * i + 1u);\n"
"        float inv_f = 1.0f / powf(theta, (float)(2u * i) / (float)rope_dim);\n"
"        if (has_freqs) inv_f /= rope_freqs[i];\n"
"        float ang = (float)rot_pos * inv_f, cs = cosf(ang), sn = sinf(ang);\n"
"        float q0 = q[qbase + idx0], q1 = q[qbase + idx1];\n"
"        q[qbase + idx0] = q0 * cs - q1 * sn; q[qbase + idx1] = q0 * sn + q1 * cs;\n"
"        if (h < n_kv) {\n"
"            float k0 = k[kbase + idx0], k1 = k[kbase + idx1];\n"
"            k[kbase + idx0] = k0 * cs - k1 * sn; k[kbase + idx1] = k0 * sn + k1 * cs;\n"
"        }\n"
"    }\n"
"    __syncthreads();\n"
"    if (h < n_kv) {\n"
"        uint32_t nblk = d_head >> 5;\n"
"        uint64_t row_bytes = (uint64_t)n_kv * (uint64_t)nblk * 34ull;\n"
"        uint64_t hoff = (uint64_t)wr_pos * row_bytes + (uint64_t)h * (uint64_t)nblk * 34ull;\n"
"        for (uint32_t blk = wid; blk < nblk; blk += 4u) {\n"
"            float kv = k[kbase + blk * 32u + lane], vv = v[kbase + blk * 32u + lane];\n"
"            float kam = fabsf(kv), vam = fabsf(vv);\n"
"            #pragma unroll\n"
"            for (int off = 16; off > 0; off >>= 1) { kam = fmaxf(kam, __shfl_xor_sync(0xFFFFFFFFu, kam, off)); vam = fmaxf(vam, __shfl_xor_sync(0xFFFFFFFFu, vam, off)); }\n"
"            float kd = kam / 127.0f, kid = (kam > 0.0f) ? (127.0f / kam) : 0.0f;\n"
"            float vd = vam / 127.0f, vid = (vam > 0.0f) ? (127.0f / vam) : 0.0f;\n"
"            uint8_t* kp = k_cache + hoff + (uint64_t)blk * 34ull;\n"
"            uint8_t* vp = v_cache + hoff + (uint64_t)blk * 34ull;\n"
"            if (lane == 0u) { *(uint16_t*)kp = __half_as_ushort(__float2half(kd)); *(uint16_t*)vp = __half_as_ushort(__float2half(vd)); }\n"
"            ((int8_t*)(kp + 2u))[lane] = (int8_t)__float2int_rn(kv * kid);\n"
"            ((int8_t*)(vp + 2u))[lane] = (int8_t)__float2int_rn(vv * vid);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_flash_tree_verify(const float* q, const half* k_cache, const half* v_cache, float* ctx,\n"
"                                                const uint64_t* tree_mask, uint32_t n_heads, uint32_t n_kv, uint32_t d_head,\n"
"                                                uint32_t start_pos, uint32_t n_nodes, float scale) {\n"
"    uint32_t h = blockIdx.x, b = blockIdx.y; if (h >= n_heads || b >= n_nodes) return;\n"
"    uint64_t mask = tree_mask[b]; uint32_t total_kv = start_pos + b + 1u;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint32_t grp = (n_heads / n_kv) > 0u ? (n_heads / n_kv) : 1u, kvh = h / grp;\n"
"    uint32_t kv_stride = n_kv * d_head;\n"
"    __shared__ float s_q[256], s_m[4], s_l[4], s_o[4][256];\n"
"    uint64_t qbase = (uint64_t)b * (n_heads * d_head) + h * d_head;\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) s_q[d] = q[qbase + d];\n"
"    __syncthreads();\n"
"    float m_run = -1e30f, l_run = 0.0f, o_acc[8] = {0,0,0,0,0,0,0,0}; uint32_t dpl = (d_head + 31u) >> 5;\n"
"    for (uint32_t t = wid; t < total_kv; t += 4u) {\n"
"        if (t >= start_pos && (((mask >> (t - start_pos)) & 1ull) == 0ull)) continue;\n"
"        const half* kp = k_cache + (uint64_t)t * kv_stride + kvh * d_head;\n"
"        const half* vp = v_cache + (uint64_t)t * kv_stride + kvh * d_head;\n"
"        float dot = 0.0f;\n"
"        for (uint32_t k = 0; k < dpl; k++) { uint32_t d = lane + k * 32u; if (d < d_head) dot = fmaf(s_q[d], __half2float(kp[d]), dot); }\n"
"        float sc = warp_sum(dot) * scale; sc = __shfl_sync(0xFFFFFFFFu, sc, 0);\n"
"        float mn = fmaxf(m_run, sc), al = expf(m_run - mn), wt = expf(sc - mn);\n"
"        l_run = fmaf(l_run, al, wt); m_run = mn;\n"
"        for (uint32_t k = 0; k < dpl; k++) { uint32_t d = lane + k * 32u; if (d < d_head) o_acc[k] = fmaf(o_acc[k], al, wt * __half2float(vp[d])); }\n"
"    }\n"
"    if (lane == 0u) { s_m[wid] = m_run; s_l[wid] = l_run; }\n"
"    for (uint32_t k = 0; k < dpl; k++) { uint32_t d = lane + k * 32u; if (d < d_head) s_o[wid][d] = o_acc[k]; }\n"
"    __syncthreads();\n"
"    float gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3])), gl = 0.0f;\n"
"    for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * expf(s_m[s] - gm);\n"
"    float inv = 1.0f / fmaxf(gl, 1e-12f);\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) {\n"
"        float ov = 0.0f; for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fmaf(s_o[s][d], expf(s_m[s] - gm), ov);\n"
"        ctx[qbase + d] = ov * inv;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_flash_tree_verify_q8(const float* q, const uint8_t* k_cache, const uint8_t* v_cache, float* ctx,\n"
"                                                   const uint64_t* tree_mask, uint32_t n_heads, uint32_t n_kv, uint32_t d_head,\n"
"                                                   uint32_t start_pos, uint32_t n_nodes, float scale) {\n"
"    uint32_t h = blockIdx.x, b = blockIdx.y; if (h >= n_heads || b >= n_nodes) return;\n"
"    uint64_t mask = tree_mask[b]; uint32_t total_kv = start_pos + b + 1u;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint32_t grp = (n_heads / n_kv) > 0u ? (n_heads / n_kv) : 1u, kvh = h / grp;\n"
"    uint32_t nblk = d_head >> 5; uint64_t row_bytes = (uint64_t)n_kv * (uint64_t)nblk * 34ull;\n"
"    __shared__ float s_q[256], s_m[4], s_l[4], s_o[4][256];\n"
"    uint64_t qbase = (uint64_t)b * (n_heads * d_head) + h * d_head;\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) s_q[d] = q[qbase + d];\n"
"    __syncthreads();\n"
"    float m_run = -1e30f, l_run = 0.0f, o_acc[8] = {0,0,0,0,0,0,0,0};\n"
"    for (uint32_t t = wid; t < total_kv; t += 4u) {\n"
"        if (t >= start_pos && (((mask >> (t - start_pos)) & 1ull) == 0ull)) continue;\n"
"        uint64_t hoff = (uint64_t)t * row_bytes + (uint64_t)kvh * (uint64_t)nblk * 34ull;\n"
"        const uint8_t* kp = k_cache + hoff; const uint8_t* vp = v_cache + hoff;\n"
"        float dot = 0.0f;\n"
"        for (uint32_t blk = 0; blk < nblk; blk++) {\n"
"            const uint8_t* kb = kp + (uint64_t)blk * 34ull;\n"
"            dot = fmaf(s_q[blk * 32u + lane], (float)((const int8_t*)(kb + 2u))[lane] * load_f16_le(kb), dot);\n"
"        }\n"
"        float sc = warp_sum(dot) * scale; sc = __shfl_sync(0xFFFFFFFFu, sc, 0);\n"
"        float mn = fmaxf(m_run, sc), al = expf(m_run - mn), wt = expf(sc - mn);\n"
"        l_run = fmaf(l_run, al, wt); m_run = mn;\n"
"        for (uint32_t blk = 0; blk < nblk; blk++) {\n"
"            const uint8_t* vb = vp + (uint64_t)blk * 34ull;\n"
"            o_acc[blk] = fmaf(o_acc[blk], al, wt * (float)((const int8_t*)(vb + 2u))[lane] * load_f16_le(vb));\n"
"        }\n"
"    }\n"
"    if (lane == 0u) { s_m[wid] = m_run; s_l[wid] = l_run; }\n"
"    for (uint32_t blk = 0; blk < nblk; blk++) s_o[wid][blk * 32u + lane] = o_acc[blk];\n"
"    __syncthreads();\n"
"    float gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3])), gl = 0.0f;\n"
"    for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * expf(s_m[s] - gm);\n"
"    float inv = 1.0f / fmaxf(gl, 1e-12f);\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) {\n"
"        float ov = 0.0f; for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fmaf(s_o[s][d], expf(s_m[s] - gm), ov);\n"
"        ctx[qbase + d] = ov * inv;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_vit_layernorm(const float* x, const float* w, const float* b, float* out, uint32_t dim, uint32_t n_tok, float eps, uint32_t has_bias) {\n"
"    uint32_t g = blockIdx.x; if (g >= n_tok) return;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    __shared__ float sm[8], sv[8];\n"
"    uint64_t base = (uint64_t)g * dim;\n"
"    float s = 0.0f; for (uint32_t i = lid; i < dim; i += 256u) s += x[base + i];\n"
"    s = warp_sum(s); if (lane == 0u) sm[wid] = s; __syncthreads();\n"
"    float mean = 0.0f; for (uint32_t k = 0; k < 8u; k++) mean += sm[k]; mean /= (float)dim;\n"
"    float v = 0.0f; for (uint32_t i = lid; i < dim; i += 256u) { float d = x[base + i] - mean; v = fmaf(d, d, v); }\n"
"    v = warp_sum(v); if (lane == 0u) sv[wid] = v; __syncthreads();\n"
"    float var = 0.0f; for (uint32_t k = 0; k < 8u; k++) var += sv[k];\n"
"    float inv = rsqrtf(var / (float)dim + eps);\n"
"    for (uint32_t i = lid; i < dim; i += 256u) { float y = (x[base + i] - mean) * inv * w[i]; out[base + i] = has_bias ? (y + b[i]) : y; }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_vit_gelu(float* x, uint32_t n, uint32_t use_tanh) {\n"
"    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;\n"
"    float v = x[i];\n"
"    if (use_tanh) {\n"
"        if (v > 8.0f) return; if (v < -8.0f) { x[i] = 0.0f; return; }\n"
"        float c = fminf(fmaxf(0.7978845608f * (v + 0.044715f * v * v * v), -10.0f), 10.0f);\n"
"        x[i] = 0.5f * v * (1.0f + tanhf(c));\n"
"    } else {\n"
"        x[i] = v / (1.0f + expf(-fminf(fmaxf(1.702f * v, -20.0f), 20.0f)));\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_vit_add(float* y, const float* x, uint32_t n) {\n"
"    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] += x[i];\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_vit_full_attn(const float* q, const float* k, const float* v, float* out, uint32_t n_tok, uint32_t n_heads, uint32_t d_head, float scale) {\n"
"    uint32_t h = blockIdx.x, qi = blockIdx.y; if (h >= n_heads || qi >= n_tok) return;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u, dm = n_heads * d_head;\n"
"    __shared__ float s_q[128], s_m[4], s_l[4], s_o[4][128];\n"
"    uint64_t qbase = (uint64_t)qi * dm + h * d_head;\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) s_q[d] = q[qbase + d];\n"
"    __syncthreads();\n"
"    float m_run = -1e30f, l_run = 0.0f, o_acc[4] = {0,0,0,0}; uint32_t dpl = (d_head + 31u) >> 5;\n"
"    for (uint32_t t = wid; t < n_tok; t += 4u) {\n"
"        const float* kp = k + (uint64_t)t * dm + h * d_head;\n"
"        const float* vp = v + (uint64_t)t * dm + h * d_head;\n"
"        float dot = 0.0f;\n"
"        for (uint32_t c = 0; c < dpl; c++) { uint32_t d = lane + c * 32u; if (d < d_head) dot = fmaf(s_q[d], kp[d], dot); }\n"
"        float sc = warp_sum(dot) * scale; sc = __shfl_sync(0xFFFFFFFFu, sc, 0);\n"
"        float mn = fmaxf(m_run, sc), al = expf(m_run - mn), wt = expf(sc - mn);\n"
"        l_run = fmaf(l_run, al, wt); m_run = mn;\n"
"        for (uint32_t c = 0; c < dpl; c++) { uint32_t d = lane + c * 32u; if (d < d_head) o_acc[c] = fmaf(o_acc[c], al, wt * vp[d]); }\n"
"    }\n"
"    if (lane == 0u) { s_m[wid] = m_run; s_l[wid] = l_run; }\n"
"    for (uint32_t c = 0; c < dpl; c++) { uint32_t d = lane + c * 32u; if (d < d_head) s_o[wid][d] = o_acc[c]; }\n"
"    __syncthreads();\n"
"    float gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3])), gl = 0.0f;\n"
"    for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * expf(s_m[s] - gm);\n"
"    float inv = 1.0f / fmaxf(gl, 1e-12f);\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) {\n"
"        float ov = 0.0f; for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fmaf(s_o[s][d], expf(s_m[s] - gm), ov);\n"
"        out[qbase + d] = ov * inv;\n"
"    }\n"
"}\n"
"\n"
"// =========================================================================\n"
"// Tenon v7: High-Occupancy Small-Batch Shared-X Kernels (B=2..8, Q4_0/Q4_1/Q4_K/Q6_K),\n"
"// Native FP8_E4M3 & OCP MXFP4, GPU Router Top-K, Batched Argmax, Zero-Copy LoRA, 2D/3D MRoPE!\n"
"// =========================================================================\n"
"extern \"C\" __global__ void cm_gemm_q4_small_b(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,\n"
"                                              uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint32_t r0 = blockIdx.x * 8u + wid * 2u;\n"
"    uint32_t B = (batch_b < 8u) ? batch_b : 8u;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 18ull;\n"
"    __shared__ float sX[8][256];\n"
"    const uint8_t* wp0 = (r0 < out_dim)      ? (w + (uint64_t)r0 * rb) : NULL;\n"
"    const uint8_t* wp1 = (r0 + 1u < out_dim) ? (w + (uint64_t)(r0 + 1u) * rb) : NULL;\n"
"    float acc0[8] = {0,0,0,0,0,0,0,0}, acc1[8] = {0,0,0,0,0,0,0,0};\n"
"    uint32_t bi = lane >> 2, g = lane & 3u, goff = g * 4u;\n"
"    for (uint32_t bk0 = 0; bk0 < nb; bk0 += 8u) {\n"
"        uint32_t kbase = bk0 << 5, idx2 = lid * 2u;\n"
"        for (uint32_t t = 0; t < B; t++) {\n"
"            if (kbase + idx2 < in_dim) {\n"
"                float2 v = __ldg((const float2*)(x + (uint64_t)t * in_dim + kbase + idx2));\n"
"                sX[t][idx2] = v.x; sX[t][idx2 + 1u] = v.y;\n"
"            } else { sX[t][idx2] = 0.0f; sX[t][idx2 + 1u] = 0.0f; }\n"
"        }\n"
"        __syncthreads();\n"
"        if (bk0 + bi < nb) {\n"
"            uint32_t sx_lo = (bi << 5) + goff, sx_hi = sx_lo + 16u;\n"
"            float d0 = 0.0f, d1 = 0.0f;\n"
"            uint32_t b00=0,b01=0,b02=0,b03=0, b10=0,b11=0,b12=0,b13=0;\n"
"            if (wp0) {\n"
"                const uint8_t* blk = wp0 + (uint64_t)(bk0 + bi) * 18ull;\n"
"                d0 = load_f16_le(blk);\n"
"                const uint16_t* q16 = (const uint16_t*)(blk + 2u + goff);\n"
"                uint16_t w0 = __ldg(&q16[0]), w1 = __ldg(&q16[1]);\n"
"                b00 = w0 & 0xFFu; b01 = w0 >> 8; b02 = w1 & 0xFFu; b03 = w1 >> 8;\n"
"            }\n"
"            if (wp1) {\n"
"                const uint8_t* blk = wp1 + (uint64_t)(bk0 + bi) * 18ull;\n"
"                d1 = load_f16_le(blk);\n"
"                const uint16_t* q16 = (const uint16_t*)(blk + 2u + goff);\n"
"                uint16_t w0 = __ldg(&q16[0]), w1 = __ldg(&q16[1]);\n"
"                b10 = w0 & 0xFFu; b11 = w0 >> 8; b12 = w1 & 0xFFu; b13 = w1 >> 8;\n"
"            }\n"
"            for (uint32_t t = 0; t < B; t++) {\n"
"                float l0 = sX[t][sx_lo+0u], l1 = sX[t][sx_lo+1u], l2 = sX[t][sx_lo+2u], l3 = sX[t][sx_lo+3u];\n"
"                float h0 = sX[t][sx_hi+0u], h1 = sX[t][sx_hi+1u], h2 = sX[t][sx_hi+2u], h3 = sX[t][sx_hi+3u];\n"
"                float xs = ((l0 + l1) + (l2 + l3)) + ((h0 + h1) + (h2 + h3));\n"
"                if (wp0) {\n"
"                    float s0 = fmaf((float)(b00&0xFu),l0, fmaf((float)(b01&0xFu),l1, fmaf((float)(b02&0xFu),l2, (float)(b03&0xFu)*l3)));\n"
"                    float s1 = fmaf((float)(b00>>4),  h0, fmaf((float)(b01>>4),  h1, fmaf((float)(b02>>4),  h2, (float)(b03>>4)*h3)));\n"
"                    acc0[t] = fmaf(d0, (s0 + s1) - 8.0f * xs, acc0[t]);\n"
"                }\n"
"                if (wp1) {\n"
"                    float s0 = fmaf((float)(b10&0xFu),l0, fmaf((float)(b11&0xFu),l1, fmaf((float)(b12&0xFu),l2, (float)(b13&0xFu)*l3)));\n"
"                    float s1 = fmaf((float)(b10>>4),  h0, fmaf((float)(b11>>4),  h1, fmaf((float)(b12>>4),  h2, (float)(b13>>4)*h3)));\n"
"                    acc1[t] = fmaf(d1, (s0 + s1) - 8.0f * xs, acc1[t]);\n"
"                }\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (uint32_t t = 0; t < B; t++) {\n"
"        float v0 = warp_sum(acc0[t]), v1 = warp_sum(acc1[t]);\n"
"        if (lane == 0u) {\n"
"            if (wp0) { uint64_t idx = (uint64_t)t * out_dim + r0;      y[idx] = add_res ? (y[idx] + v0) : v0; }\n"
"            if (wp1) { uint64_t idx = (uint64_t)t * out_dim + r0 + 1u; y[idx] = add_res ? (y[idx] + v1) : v1; }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_q4_1_small_b(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,\n"
"                                                uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint32_t r0 = blockIdx.x * 8u + wid * 2u;\n"
"    uint32_t B = (batch_b < 8u) ? batch_b : 8u;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 20ull;\n"
"    __shared__ float sX[8][256];\n"
"    const uint8_t* wp0 = (r0 < out_dim)      ? (w + (uint64_t)r0 * rb) : NULL;\n"
"    const uint8_t* wp1 = (r0 + 1u < out_dim) ? (w + (uint64_t)(r0 + 1u) * rb) : NULL;\n"
"    float acc0[8] = {0,0,0,0,0,0,0,0}, acc1[8] = {0,0,0,0,0,0,0,0};\n"
"    uint32_t bi = lane >> 2, g = lane & 3u, goff = g * 4u;\n"
"    for (uint32_t bk0 = 0; bk0 < nb; bk0 += 8u) {\n"
"        uint32_t kbase = bk0 << 5, idx2 = lid * 2u;\n"
"        for (uint32_t t = 0; t < B; t++) {\n"
"            if (kbase + idx2 < in_dim) {\n"
"                float2 v = __ldg((const float2*)(x + (uint64_t)t * in_dim + kbase + idx2));\n"
"                sX[t][idx2] = v.x; sX[t][idx2 + 1u] = v.y;\n"
"            } else { sX[t][idx2] = 0.0f; sX[t][idx2 + 1u] = 0.0f; }\n"
"        }\n"
"        __syncthreads();\n"
"        if (bk0 + bi < nb) {\n"
"            uint32_t sx_lo = (bi << 5) + goff, sx_hi = sx_lo + 16u;\n"
"            float d0 = 0.0f, m0 = 0.0f, d1 = 0.0f, m1 = 0.0f;\n"
"            uint32_t w0v = 0u, w1v = 0u;\n"
"            if (wp0) {\n"
"                const uint8_t* blk = wp0 + (uint64_t)(bk0 + bi) * 20ull;\n"
"                d0 = load_f16_le(blk); m0 = load_f16_le(blk + 2u);\n"
"                w0v = __ldg((const uint32_t*)(blk + 4u + goff));\n"
"            }\n"
"            if (wp1) {\n"
"                const uint8_t* blk = wp1 + (uint64_t)(bk0 + bi) * 20ull;\n"
"                d1 = load_f16_le(blk); m1 = load_f16_le(blk + 2u);\n"
"                w1v = __ldg((const uint32_t*)(blk + 4u + goff));\n"
"            }\n"
"            uint32_t b00=w0v&0xFFu, b01=(w0v>>8)&0xFFu, b02=(w0v>>16)&0xFFu, b03=w0v>>24;\n"
"            uint32_t b10=w1v&0xFFu, b11=(w1v>>8)&0xFFu, b12=(w1v>>16)&0xFFu, b13=w1v>>24;\n"
"            for (uint32_t t = 0; t < B; t++) {\n"
"                float l0 = sX[t][sx_lo+0u], l1 = sX[t][sx_lo+1u], l2 = sX[t][sx_lo+2u], l3 = sX[t][sx_lo+3u];\n"
"                float h0 = sX[t][sx_hi+0u], h1 = sX[t][sx_hi+1u], h2 = sX[t][sx_hi+2u], h3 = sX[t][sx_hi+3u];\n"
"                float xs = ((l0 + l1) + (l2 + l3)) + ((h0 + h1) + (h2 + h3));\n"
"                if (wp0) {\n"
"                    float s0 = fmaf((float)(b00&0xFu),l0, fmaf((float)(b01&0xFu),l1, fmaf((float)(b02&0xFu),l2, (float)(b03&0xFu)*l3)));\n"
"                    float s1 = fmaf((float)(b00>>4),  h0, fmaf((float)(b01>>4),  h1, fmaf((float)(b02>>4),  h2, (float)(b03>>4)*h3)));\n"
"                    acc0[t] += fmaf(d0, s0 + s1, m0 * xs);\n"
"                }\n"
"                if (wp1) {\n"
"                    float s0 = fmaf((float)(b10&0xFu),l0, fmaf((float)(b11&0xFu),l1, fmaf((float)(b12&0xFu),l2, (float)(b13&0xFu)*l3)));\n"
"                    float s1 = fmaf((float)(b10>>4),  h0, fmaf((float)(b11>>4),  h1, fmaf((float)(b12>>4),  h2, (float)(b13>>4)*h3)));\n"
"                    acc1[t] += fmaf(d1, s0 + s1, m1 * xs);\n"
"                }\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    for (uint32_t t = 0; t < B; t++) {\n"
"        float v0 = warp_sum(acc0[t]), v1 = warp_sum(acc1[t]);\n"
"        if (lane == 0u) {\n"
"            if (wp0) { uint64_t idx = (uint64_t)t * out_dim + r0;      y[idx] = add_res ? (y[idx] + v0) : v0; }\n"
"            if (wp1) { uint64_t idx = (uint64_t)t * out_dim + r0 + 1u; y[idx] = add_res ? (y[idx] + v1) : v1; }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_q4k_small_b(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,\n"
"                                               uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint32_t row = blockIdx.x * 4u + wid;\n"
"    uint32_t B = (batch_b < 8u) ? batch_b : 8u;\n"
"    uint32_t nb = in_dim >> 8; uint64_t rb = (uint64_t)nb * 144ull;\n"
"    __shared__ float sX[8][256];\n"
"    const uint8_t* wp = (row < out_dim) ? (w + (uint64_t)row * rb) : NULL;\n"
"    float acc[8] = {0,0,0,0,0,0,0,0};\n"
"    uint32_t sb = lane >> 2, g = lane & 3u, l0 = g * 8u, sx0 = sb * 32u + l0, hi = sb & 1u;\n"
"    for (uint32_t bk = 0; bk < nb; bk++) {\n"
"        uint32_t kbase = bk << 8, idx2 = lid * 2u;\n"
"        for (uint32_t t = 0; t < B; t++) {\n"
"            float2 v = __ldg((const float2*)(x + (uint64_t)t * in_dim + kbase + idx2));\n"
"            sX[t][idx2] = v.x; sX[t][idx2 + 1u] = v.y;\n"
"        }\n"
"        __syncthreads();\n"
"        if (wp) {\n"
"            const uint8_t* blk = wp + (uint64_t)bk * 144ull;\n"
"            float d = load_f16_le(blk), dm = load_f16_le(blk + 2u);\n"
"            const uint8_t* sc = blk + 4u;\n"
"            uint8_t scv, mv;\n"
"            if (sb < 4u) { scv = sc[sb] & 63u; mv = sc[sb + 4u] & 63u; }\n"
"            else { scv = (sc[sb + 4u] & 0xFu) | ((sc[sb - 4u] >> 6) << 4); mv = (sc[sb + 4u] >> 4) | ((sc[sb] >> 6) << 4); }\n"
"            float dl = d * (float)scv, ml = dm * (float)mv;\n"
"            const uint8_t* qp = blk + 16u + (sb >> 1) * 32u + l0;\n"
"            float w8[8];\n"
"            #pragma unroll\n"
"            for (uint32_t j = 0; j < 8u; j++) { uint8_t bv = __ldg(&qp[j]); w8[j] = fmaf(dl, (float)(hi ? (bv >> 4) : (bv & 0xFu)), -ml); }\n"
"            for (uint32_t t = 0; t < B; t++) {\n"
"                float s = 0.0f;\n"
"                #pragma unroll\n"
"                for (uint32_t j = 0; j < 8u; j++) s = fmaf(w8[j], sX[t][sx0 + j], s);\n"
"                acc[t] += s;\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    if (wp) {\n"
"        for (uint32_t t = 0; t < B; t++) {\n"
"            float r = warp_sum(acc[t]);\n"
"            if (lane == 0u) { uint64_t idx = (uint64_t)t * out_dim + row; y[idx] = add_res ? (y[idx] + r) : r; }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_q6k_small_b(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,\n"
"                                               uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint32_t row = blockIdx.x * 4u + wid;\n"
"    uint32_t B = (batch_b < 8u) ? batch_b : 8u;\n"
"    uint32_t nb = in_dim >> 8; uint64_t rb = (uint64_t)nb * 210ull;\n"
"    __shared__ float sX[8][256];\n"
"    const uint8_t* wp = (row < out_dim) ? (w + (uint64_t)row * rb) : NULL;\n"
"    float acc[8] = {0,0,0,0,0,0,0,0};\n"
"    uint32_t n128 = (lane >> 4) * 128u, qpart = (lane >> 2) & 3u, g = lane & 3u, l0 = g * 8u;\n"
"    uint32_t sx0 = n128 + qpart * 32u + l0;\n"
"    for (uint32_t bk = 0; bk < nb; bk++) {\n"
"        uint32_t kbase = bk << 8, idx2 = lid * 2u;\n"
"        for (uint32_t t = 0; t < B; t++) {\n"
"            float2 v = __ldg((const float2*)(x + (uint64_t)t * in_dim + kbase + idx2));\n"
"            sX[t][idx2] = v.x; sX[t][idx2 + 1u] = v.y;\n"
"        }\n"
"        __syncthreads();\n"
"        if (wp) {\n"
"            const uint8_t* blk = wp + (uint64_t)bk * 210ull;\n"
"            float d = load_f16_le(blk + 208u);\n"
"            const uint8_t* qlp = blk + (n128 >> 1);\n"
"            const uint8_t* qhp = blk + 128u + (n128 >> 2);\n"
"            const int8_t* scp = (const int8_t*)(blk + 192u) + (n128 >> 4);\n"
"            float ds = d * (float)scp[(l0 >> 4) + qpart * 2u];\n"
"            float w8[8];\n"
"            #pragma unroll\n"
"            for (uint32_t j = 0; j < 8u; j++) {\n"
"                uint32_t l = l0 + j;\n"
"                uint8_t lo = (qpart == 0u) ? (__ldg(&qlp[l]) & 0xFu) : (qpart == 1u) ? (__ldg(&qlp[l + 32u]) & 0xFu)\n"
"                           : (qpart == 2u) ? (__ldg(&qlp[l]) >> 4)   : (__ldg(&qlp[l + 32u]) >> 4);\n"
"                uint8_t hi = (__ldg(&qhp[l]) >> (qpart * 2u)) & 3u;\n"
"                w8[j] = ds * (float)((int)(lo | (hi << 4)) - 32);\n"
"            }\n"
"            for (uint32_t t = 0; t < B; t++) {\n"
"                float s = 0.0f;\n"
"                #pragma unroll\n"
"                for (uint32_t j = 0; j < 8u; j++) s = fmaf(w8[j], sX[t][sx0 + j], s);\n"
"                acc[t] += s;\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    if (wp) {\n"
"        for (uint32_t t = 0; t < B; t++) {\n"
"            float r = warp_sum(acc[t]);\n"
"            if (lane == 0u) { uint64_t idx = (uint64_t)t * out_dim + row; y[idx] = add_res ? (y[idx] + r) : r; }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"__device__ __forceinline__ float cm_fp8_e4m3_dec(uint8_t b) {\n"
"    uint32_t s = b >> 7, e = (b >> 3) & 0xFu, m = b & 7u;\n"
"    float v = (e == 0u) ? ((float)m * 0.001953125f) : ldexpf(1.0f + (float)m * 0.125f, (int)e - 7);\n"
"    return s ? -v : v;\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_fp8_e4m3(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid;\n"
"    if (row >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 34ull;\n"
"    const uint8_t* wp = w + (uint64_t)row * rb;\n"
"    float acc = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        const uint8_t* blk = wp + (uint64_t)b * 34ull;\n"
"        float d = load_f16_le(blk); const float* xp = x + (b << 5); float s = 0.0f;\n"
"        for (uint32_t i = 0; i < 32u; i++) s = fmaf(cm_fp8_e4m3_dec(blk[2u + i]), xp[i], s);\n"
"        acc = fmaf(d, s, acc);\n"
"    }\n"
"    float r = warp_sum(acc);\n"
"    if (lane == 0u) y[row] = add_res ? (y[row] + r) : r;\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_fp8_e4m3_batch(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid, b0 = blockIdx.y * 8u;\n"
"    if (row >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t bt = (batch_b - b0 < 8u) ? (batch_b - b0) : 8u, nb = in_dim >> 5;\n"
"    const uint8_t* wp = w + (uint64_t)row * (uint64_t)nb * 34ull;\n"
"    float acc[8] = {0,0,0,0,0,0,0,0};\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        const uint8_t* blk = wp + (uint64_t)b * 34ull; float d = load_f16_le(blk);\n"
"        for (uint32_t i = 0; i < 32u; i++) {\n"
"            float wv = d * cm_fp8_e4m3_dec(blk[2u + i]);\n"
"            for (uint32_t t = 0; t < bt; t++) acc[t] = fmaf(wv, x[(uint64_t)(b0 + t) * in_dim + (b << 5) + i], acc[t]);\n"
"        }\n"
"    }\n"
"    for (uint32_t t = 0; t < bt; t++) { float r = warp_sum(acc[t]); if (lane == 0u) { uint64_t idx = (uint64_t)(b0 + t) * out_dim + row; y[idx] = add_res ? (y[idx] + r) : r; } }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemv_mxfp4(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    const float kMx4[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid;\n"
"    if (row >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 17ull;\n"
"    const uint8_t* wp = w + (uint64_t)row * rb;\n"
"    float acc = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        const uint8_t* blk = wp + (uint64_t)b * 17ull;\n"
"        float sc = __uint_as_float(((uint32_t)blk[0]) << 23); const float* xp = x + (b << 5); float s = 0.0f;\n"
"        for (uint32_t i = 0; i < 16u; i++) { uint8_t bv = blk[1u + i]; s = fmaf(kMx4[bv & 0xFu], xp[i], fmaf(kMx4[bv >> 4], xp[16u + i], s)); }\n"
"        acc = fmaf(sc, s, acc);\n"
"    }\n"
"    float r = warp_sum(acc);\n"
"    if (lane == 0u) y[row] = add_res ? (y[row] + r) : r;\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_mxfp4_batch(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    const float kMx4[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid, b0 = blockIdx.y * 8u;\n"
"    if (row >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t bt = (batch_b - b0 < 8u) ? (batch_b - b0) : 8u, nb = in_dim >> 5;\n"
"    const uint8_t* wp = w + (uint64_t)row * (uint64_t)nb * 17ull;\n"
"    float acc[8] = {0,0,0,0,0,0,0,0};\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        const uint8_t* blk = wp + (uint64_t)b * 17ull;\n"
"        float sc = __uint_as_float(((uint32_t)blk[0]) << 23);\n"
"        for (uint32_t i = 0; i < 16u; i++) {\n"
"            uint8_t bv = blk[1u + i]; float wl = sc * kMx4[bv & 0xFu], wh = sc * kMx4[bv >> 4];\n"
"            for (uint32_t t = 0; t < bt; t++) {\n"
"                const float* xp = x + (uint64_t)(b0 + t) * in_dim + (b << 5);\n"
"                acc[t] = fmaf(wl, xp[i], fmaf(wh, xp[16u + i], acc[t]));\n"
"            }\n"
"        }\n"
"    }\n"
"    for (uint32_t t = 0; t < bt; t++) { float r = warp_sum(acc[t]); if (lane == 0u) { uint64_t idx = (uint64_t)(b0 + t) * out_dim + row; y[idx] = add_res ? (y[idx] + r) : r; } }\n"
"}\n"
"\n"
"// =========================================================================\n"
"// Tenon v8: Hardware FP16 Tensor Core 2-Stage cp.async Marlin Kernels\n"
"//   (Q4_K, Q6_K, FP8_E4M3, MXFP4) + Batched Paged Latent MLA + Grouped MoE!\n"
"// =========================================================================\n"
"extern \"C\" __global__ void cm_gemm_q4k_marlin(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,\n"
"                                              uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t r0 = blockIdx.x * 32u, b0 = blockIdx.y * 32u;\n"
"    if (r0 >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t lid = threadIdx.x, nb = in_dim >> 8, n_sub = in_dim >> 5;\n"
"    uint64_t rb = (uint64_t)nb * 144ull;\n"
"    uint32_t tr = lid >> 2, g = lid & 3u, c0 = g * 8u, gr = r0 + tr;\n"
"    const uint8_t* wrow = w + (uint64_t)(gr < out_dim ? gr : (out_dim - 1u)) * rb;\n"
"#if __CUDA_ARCH__ >= 800\n"
"    __shared__ float sXf[2][32][32];\n"
"    __shared__ uint16_t sX[32][34];\n"
"    __shared__ uint16_t sW[32][34];\n"
"    uint32_t tr_x0 = lid >> 3, tr_x1 = tr_x0 + 16u, cx4 = (lid & 7u) * 4u;\n"
"    uint32_t gb0 = (b0 + tr_x0 < batch_b) ? (b0 + tr_x0) : (batch_b - 1u);\n"
"    uint32_t gb1 = (b0 + tr_x1 < batch_b) ? (b0 + tr_x1) : (batch_b - 1u);\n"
"    int xb0 = (b0 + tr_x0 < batch_b) ? 16 : 0, xb1 = (b0 + tr_x1 < batch_b) ? 16 : 0;\n"
"    uint32_t wid = lid >> 5, lane = lid & 31u, gid = lane >> 2, tid = lane & 3u;\n"
"    uint32_t wb = (wid >> 1) * 16u, wr = (wid & 1u) * 16u;\n"
"    int warp_act = (b0 + wb < batch_b) && (r0 + wr < out_dim);\n"
"    float c00=0,c01=0,c02=0,c03=0, c10=0,c11=0,c12=0,c13=0;\n"
"    cm_cp_async16(&sXf[0][tr_x0][cx4], x + (uint64_t)gb0 * in_dim + cx4, xb0);\n"
"    cm_cp_async16(&sXf[0][tr_x1][cx4], x + (uint64_t)gb1 * in_dim + cx4, xb1);\n"
"    asm volatile(\"cp.async.commit_group;\\n\");\n"
"    for (uint32_t sub = 0; sub < n_sub; sub++) {\n"
"        uint32_t st = sub & 1u, nxt = st ^ 1u;\n"
"        if (sub + 1u < n_sub) {\n"
"            uint32_t nk = ((sub + 1u) << 5) + cx4;\n"
"            cm_cp_async16(&sXf[nxt][tr_x0][cx4], x + (uint64_t)gb0 * in_dim + nk, xb0);\n"
"            cm_cp_async16(&sXf[nxt][tr_x1][cx4], x + (uint64_t)gb1 * in_dim + nk, xb1);\n"
"        }\n"
"        asm volatile(\"cp.async.commit_group;\\n\");\n"
"        if (gr < out_dim) {\n"
"            uint32_t bk = sub >> 3, sb = sub & 7u;\n"
"            const uint8_t* blk = wrow + (uint64_t)bk * 144ull;\n"
"            float d = load_f16_le(blk), dm = load_f16_le(blk + 2u);\n"
"            const uint8_t* sc = blk + 4u; const uint8_t* qp = blk + 16u + (sb >> 1) * 32u + c0;\n"
"            uint8_t scv, mv;\n"
"            if (sb < 4u) { scv = sc[sb] & 63u; mv = sc[sb + 4u] & 63u; }\n"
"            else { scv = (sc[sb + 4u] & 0xFu) | ((sc[sb - 4u] >> 6) << 4); mv = (sc[sb + 4u] >> 4) | ((sc[sb] >> 6) << 4); }\n"
"            float dl = d * (float)scv, ml = dm * (float)mv; uint32_t hi = sb & 1u;\n"
"            #pragma unroll\n"
"            for (uint32_t j = 0; j < 8u; j += 2u) {\n"
"                uint8_t b0v = __ldg(&qp[j]), b1v = __ldg(&qp[j + 1u]);\n"
"                float w0 = fmaf(dl, (float)(hi ? (b0v >> 4) : (b0v & 0xFu)), -ml);\n"
"                float w1 = fmaf(dl, (float)(hi ? (b1v >> 4) : (b1v & 0xFu)), -ml);\n"
"                __half2 wh = __float22half2_rn(make_float2(w0, w1));\n"
"                *(uint32_t*)&sW[tr][c0 + j] = *reinterpret_cast<const uint32_t*>(&wh);\n"
"            }\n"
"        } else {\n"
"            #pragma unroll\n"
"            for (uint32_t j = 0; j < 8u; j += 2u) *(uint32_t*)&sW[tr][c0 + j] = 0u;\n"
"        }\n"
"        asm volatile(\"cp.async.wait_group 1;\\n\");\n"
"        __syncthreads();\n"
"        __half2 h00 = __float22half2_rn(make_float2(sXf[st][tr_x0][cx4+0u], sXf[st][tr_x0][cx4+1u]));\n"
"        __half2 h01 = __float22half2_rn(make_float2(sXf[st][tr_x0][cx4+2u], sXf[st][tr_x0][cx4+3u]));\n"
"        __half2 h10 = __float22half2_rn(make_float2(sXf[st][tr_x1][cx4+0u], sXf[st][tr_x1][cx4+1u]));\n"
"        __half2 h11 = __float22half2_rn(make_float2(sXf[st][tr_x1][cx4+2u], sXf[st][tr_x1][cx4+3u]));\n"
"        *(uint32_t*)&sX[tr_x0][cx4+0u] = *reinterpret_cast<const uint32_t*>(&h00);\n"
"        *(uint32_t*)&sX[tr_x0][cx4+2u] = *reinterpret_cast<const uint32_t*>(&h01);\n"
"        *(uint32_t*)&sX[tr_x1][cx4+0u] = *reinterpret_cast<const uint32_t*>(&h10);\n"
"        *(uint32_t*)&sX[tr_x1][cx4+2u] = *reinterpret_cast<const uint32_t*>(&h11);\n"
"        __syncthreads();\n"
"        if (warp_act) {\n"
"            #pragma unroll\n"
"            for (uint32_t kk = 0; kk < 32u; kk += 16u) {\n"
"                uint32_t a0 = *(const uint32_t*)&sX[wb + gid][kk + tid * 2u];\n"
"                uint32_t a1 = *(const uint32_t*)&sX[wb + gid + 8u][kk + tid * 2u];\n"
"                uint32_t a2 = *(const uint32_t*)&sX[wb + gid][kk + 8u + tid * 2u];\n"
"                uint32_t a3 = *(const uint32_t*)&sX[wb + gid + 8u][kk + 8u + tid * 2u];\n"
"                uint32_t b00 = *(const uint32_t*)&sW[wr + gid][kk + tid * 2u];\n"
"                uint32_t b01 = *(const uint32_t*)&sW[wr + gid][kk + 8u + tid * 2u];\n"
"                asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
"                             : \"+f\"(c00), \"+f\"(c01), \"+f\"(c02), \"+f\"(c03) : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b00), \"r\"(b01));\n"
"                uint32_t b10 = *(const uint32_t*)&sW[wr + 8u + gid][kk + tid * 2u];\n"
"                uint32_t b11 = *(const uint32_t*)&sW[wr + 8u + gid][kk + 8u + tid * 2u];\n"
"                asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
"                             : \"+f\"(c10), \"+f\"(c11), \"+f\"(c12), \"+f\"(c13) : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b10), \"r\"(b11));\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    asm volatile(\"cp.async.wait_all;\\n\");\n"
"    if (warp_act) {\n"
"        uint32_t ob0 = b0 + wb + gid, ob1 = ob0 + 8u, or0 = r0 + wr + tid * 2u, or1 = or0 + 8u;\n"
"        #define ST_M(ob, orw, val) if ((ob) < batch_b && (orw) < out_dim) { uint64_t idx = (uint64_t)(ob) * out_dim + (orw); y[idx] = add_res ? (y[idx] + (val)) : (val); }\n"
"        ST_M(ob0, or0, c00); ST_M(ob0, or0+1u, c01); ST_M(ob1, or0, c02); ST_M(ob1, or0+1u, c03);\n"
"        ST_M(ob0, or1, c10); ST_M(ob0, or1+1u, c11); ST_M(ob1, or1, c12); ST_M(ob1, or1+1u, c13);\n"
"        #undef ST_M\n"
"    }\n"
"#else\n"
"    __shared__ float sX[32][33]; __shared__ float sW[32][33];\n"
"    uint32_t gb = b0 + tr, th_b = (lid >> 3) * 2u, th_r = (lid & 7u) * 4u;\n"
"    float acc[2][4] = {{0,0,0,0},{0,0,0,0}};\n"
"    for (uint32_t sub = 0; sub < n_sub; sub++) {\n"
"        uint32_t bk = sub >> 3, sb = sub & 7u, kbase = sub << 5;\n"
"        if (gb < batch_b) { for (uint32_t j = 0; j < 8u; j++) sX[tr][c0+j] = x[(uint64_t)gb*in_dim + kbase + c0 + j]; }\n"
"        else { for (uint32_t j = 0; j < 8u; j++) sX[tr][c0+j] = 0.0f; }\n"
"        if (gr < out_dim) {\n"
"            const uint8_t* blk = wrow + (uint64_t)bk * 144ull;\n"
"            float d = load_f16_le(blk), dm = load_f16_le(blk + 2u); const uint8_t* sc = blk + 4u; const uint8_t* qp = blk + 16u + (sb >> 1) * 32u + c0;\n"
"            uint8_t scv = (sb < 4u) ? (sc[sb]&63u) : ((sc[sb+4u]&0xFu)|((sc[sb-4u]>>6)<<4));\n"
"            uint8_t mv  = (sb < 4u) ? (sc[sb+4u]&63u) : ((sc[sb+4u]>>4)|((sc[sb]>>6)<<4));\n"
"            float dl = d * (float)scv, ml = dm * (float)mv; uint32_t hi = sb & 1u;\n"
"            for (uint32_t j = 0; j < 8u; j++) { uint8_t bv = qp[j]; sW[tr][c0+j] = fmaf(dl, (float)(hi ? (bv>>4) : (bv&0xFu)), -ml); }\n"
"        } else { for (uint32_t j = 0; j < 8u; j++) sW[tr][c0+j] = 0.0f; }\n"
"        __syncthreads();\n"
"        for (uint32_t k = 0; k < 32u; k++) for (uint32_t tb = 0; tb < 2u; tb++) for (uint32_t rr = 0; rr < 4u; rr++) acc[tb][rr] = fmaf(sX[th_b+tb][k], sW[th_r+rr][k], acc[tb][rr]);\n"
"        __syncthreads();\n"
"    }\n"
"    for (uint32_t tb = 0; tb < 2u; tb++) for (uint32_t rr = 0; rr < 4u; rr++) { uint32_t ob = b0+th_b+tb, orw = r0+th_r+rr; if (ob < batch_b && orw < out_dim) { uint64_t idx = (uint64_t)ob*out_dim + orw; y[idx] = add_res ? (y[idx] + acc[tb][rr]) : acc[tb][rr]; } }\n"
"#endif\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_q6k_marlin(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,\n"
"                                              uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t r0 = blockIdx.x * 32u, b0 = blockIdx.y * 32u;\n"
"    if (r0 >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t lid = threadIdx.x, nb = in_dim >> 8, n_sub = in_dim >> 5;\n"
"    uint64_t rb = (uint64_t)nb * 210ull;\n"
"    uint32_t tr = lid >> 2, g = lid & 3u, c0 = g * 8u, gr = r0 + tr;\n"
"    const uint8_t* wrow = w + (uint64_t)(gr < out_dim ? gr : (out_dim - 1u)) * rb;\n"
"#if __CUDA_ARCH__ >= 800\n"
"    __shared__ float sXf[2][32][32];\n"
"    __shared__ uint16_t sX[32][34];\n"
"    __shared__ uint16_t sW[32][34];\n"
"    uint32_t tr_x0 = lid >> 3, tr_x1 = tr_x0 + 16u, cx4 = (lid & 7u) * 4u;\n"
"    uint32_t gb0 = (b0 + tr_x0 < batch_b) ? (b0 + tr_x0) : (batch_b - 1u);\n"
"    uint32_t gb1 = (b0 + tr_x1 < batch_b) ? (b0 + tr_x1) : (batch_b - 1u);\n"
"    int xb0 = (b0 + tr_x0 < batch_b) ? 16 : 0, xb1 = (b0 + tr_x1 < batch_b) ? 16 : 0;\n"
"    uint32_t wid = lid >> 5, lane = lid & 31u, gid = lane >> 2, tid = lane & 3u;\n"
"    uint32_t wb = (wid >> 1) * 16u, wr = (wid & 1u) * 16u;\n"
"    int warp_act = (b0 + wb < batch_b) && (r0 + wr < out_dim);\n"
"    float c00=0,c01=0,c02=0,c03=0, c10=0,c11=0,c12=0,c13=0;\n"
"    cm_cp_async16(&sXf[0][tr_x0][cx4], x + (uint64_t)gb0 * in_dim + cx4, xb0);\n"
"    cm_cp_async16(&sXf[0][tr_x1][cx4], x + (uint64_t)gb1 * in_dim + cx4, xb1);\n"
"    asm volatile(\"cp.async.commit_group;\\n\");\n"
"    for (uint32_t sub = 0; sub < n_sub; sub++) {\n"
"        uint32_t st = sub & 1u, nxt = st ^ 1u;\n"
"        if (sub + 1u < n_sub) {\n"
"            uint32_t nk = ((sub + 1u) << 5) + cx4;\n"
"            cm_cp_async16(&sXf[nxt][tr_x0][cx4], x + (uint64_t)gb0 * in_dim + nk, xb0);\n"
"            cm_cp_async16(&sXf[nxt][tr_x1][cx4], x + (uint64_t)gb1 * in_dim + nk, xb1);\n"
"        }\n"
"        asm volatile(\"cp.async.commit_group;\\n\");\n"
"        if (gr < out_dim) {\n"
"            uint32_t bk = sub >> 3, sb = sub & 7u, n128 = (sb >> 2) << 7, qp = sb & 3u;\n"
"            const uint8_t* blk = wrow + (uint64_t)bk * 210ull;\n"
"            float d = load_f16_le(blk + 208u);\n"
"            const uint8_t* ql = blk + (n128 >> 1); const uint8_t* qh = blk + 128u + (n128 >> 2);\n"
"            const int8_t* sc = (const int8_t*)(blk + 192u) + (n128 >> 4);\n"
"            float ds = d * (float)sc[(c0 >> 4) + qp * 2u];\n"
"            #pragma unroll\n"
"            for (uint32_t j = 0; j < 8u; j += 2u) {\n"
"                float w2[2];\n"
"                #pragma unroll\n"
"                for (uint32_t u = 0; u < 2u; u++) {\n"
"                    uint32_t lj = c0 + j + u;\n"
"                    uint8_t lo = (qp == 0u) ? (__ldg(&ql[lj]) & 0xFu) : (qp == 1u) ? (__ldg(&ql[lj + 32u]) & 0xFu)\n"
"                               : (qp == 2u) ? (__ldg(&ql[lj]) >> 4)   : (__ldg(&ql[lj + 32u]) >> 4);\n"
"                    uint8_t hi = (__ldg(&qh[lj]) >> (qp * 2u)) & 3u;\n"
"                    w2[u] = ds * (float)((int)(lo | (hi << 4)) - 32);\n"
"                }\n"
"                __half2 wh = __float22half2_rn(make_float2(w2[0], w2[1]));\n"
"                *(uint32_t*)&sW[tr][c0 + j] = *reinterpret_cast<const uint32_t*>(&wh);\n"
"            }\n"
"        } else {\n"
"            #pragma unroll\n"
"            for (uint32_t j = 0; j < 8u; j += 2u) *(uint32_t*)&sW[tr][c0 + j] = 0u;\n"
"        }\n"
"        asm volatile(\"cp.async.wait_group 1;\\n\");\n"
"        __syncthreads();\n"
"        __half2 h00 = __float22half2_rn(make_float2(sXf[st][tr_x0][cx4+0u], sXf[st][tr_x0][cx4+1u]));\n"
"        __half2 h01 = __float22half2_rn(make_float2(sXf[st][tr_x0][cx4+2u], sXf[st][tr_x0][cx4+3u]));\n"
"        __half2 h10 = __float22half2_rn(make_float2(sXf[st][tr_x1][cx4+0u], sXf[st][tr_x1][cx4+1u]));\n"
"        __half2 h11 = __float22half2_rn(make_float2(sXf[st][tr_x1][cx4+2u], sXf[st][tr_x1][cx4+3u]));\n"
"        *(uint32_t*)&sX[tr_x0][cx4+0u] = *reinterpret_cast<const uint32_t*>(&h00);\n"
"        *(uint32_t*)&sX[tr_x0][cx4+2u] = *reinterpret_cast<const uint32_t*>(&h01);\n"
"        *(uint32_t*)&sX[tr_x1][cx4+0u] = *reinterpret_cast<const uint32_t*>(&h10);\n"
"        *(uint32_t*)&sX[tr_x1][cx4+2u] = *reinterpret_cast<const uint32_t*>(&h11);\n"
"        __syncthreads();\n"
"        if (warp_act) {\n"
"            #pragma unroll\n"
"            for (uint32_t kk = 0; kk < 32u; kk += 16u) {\n"
"                uint32_t a0 = *(const uint32_t*)&sX[wb + gid][kk + tid * 2u];\n"
"                uint32_t a1 = *(const uint32_t*)&sX[wb + gid + 8u][kk + tid * 2u];\n"
"                uint32_t a2 = *(const uint32_t*)&sX[wb + gid][kk + 8u + tid * 2u];\n"
"                uint32_t a3 = *(const uint32_t*)&sX[wb + gid + 8u][kk + 8u + tid * 2u];\n"
"                uint32_t b00 = *(const uint32_t*)&sW[wr + gid][kk + tid * 2u];\n"
"                uint32_t b01 = *(const uint32_t*)&sW[wr + gid][kk + 8u + tid * 2u];\n"
"                asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
"                             : \"+f\"(c00), \"+f\"(c01), \"+f\"(c02), \"+f\"(c03) : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b00), \"r\"(b01));\n"
"                uint32_t b10 = *(const uint32_t*)&sW[wr + 8u + gid][kk + tid * 2u];\n"
"                uint32_t b11 = *(const uint32_t*)&sW[wr + 8u + gid][kk + 8u + tid * 2u];\n"
"                asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
"                             : \"+f\"(c10), \"+f\"(c11), \"+f\"(c12), \"+f\"(c13) : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b10), \"r\"(b11));\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    asm volatile(\"cp.async.wait_all;\\n\");\n"
"    if (warp_act) {\n"
"        uint32_t ob0 = b0 + wb + gid, ob1 = ob0 + 8u, or0 = r0 + wr + tid * 2u, or1 = or0 + 8u;\n"
"        #define ST_M(ob, orw, val) if ((ob) < batch_b && (orw) < out_dim) { uint64_t idx = (uint64_t)(ob) * out_dim + (orw); y[idx] = add_res ? (y[idx] + (val)) : (val); }\n"
"        ST_M(ob0, or0, c00); ST_M(ob0, or0+1u, c01); ST_M(ob1, or0, c02); ST_M(ob1, or0+1u, c03);\n"
"        ST_M(ob0, or1, c10); ST_M(ob0, or1+1u, c11); ST_M(ob1, or1, c12); ST_M(ob1, or1+1u, c13);\n"
"        #undef ST_M\n"
"    }\n"
"#else\n"
"    __shared__ float sX[32][33]; __shared__ float sW[32][33];\n"
"    uint32_t gb = b0 + tr, th_b = (lid >> 3) * 2u, th_r = (lid & 7u) * 4u;\n"
"    float acc[2][4] = {{0,0,0,0},{0,0,0,0}};\n"
"    for (uint32_t sub = 0; sub < n_sub; sub++) {\n"
"        uint32_t bk = sub >> 3, sb = sub & 7u, kbase = sub << 5;\n"
"        if (gb < batch_b) { for (uint32_t j = 0; j < 8u; j++) sX[tr][c0+j] = x[(uint64_t)gb*in_dim + kbase + c0 + j]; }\n"
"        else { for (uint32_t j = 0; j < 8u; j++) sX[tr][c0+j] = 0.0f; }\n"
"        if (gr < out_dim) {\n"
"            uint32_t n128 = (sb >> 2) << 7, qp = sb & 3u;\n"
"            const uint8_t* blk = wrow + (uint64_t)bk * 210ull; float d = load_f16_le(blk + 208u);\n"
"            const uint8_t* ql = blk + (n128 >> 1); const uint8_t* qh = blk + 128u + (n128 >> 2); const int8_t* sc = (const int8_t*)(blk + 192u) + (n128 >> 4);\n"
"            float ds = d * (float)sc[(c0 >> 4) + qp * 2u];\n"
"            for (uint32_t j = 0; j < 8u; j++) {\n"
"                uint32_t lj = c0 + j;\n"
"                uint8_t lo = (qp == 0u) ? (ql[lj]&0xFu) : (qp == 1u) ? (ql[lj+32u]&0xFu) : (qp == 2u) ? (ql[lj]>>4) : (ql[lj+32u]>>4);\n"
"                uint8_t hi = (qh[lj] >> (qp * 2u)) & 3u;\n"
"                sW[tr][c0+j] = ds * (float)((int)(lo | (hi << 4)) - 32);\n"
"            }\n"
"        } else { for (uint32_t j = 0; j < 8u; j++) sW[tr][c0+j] = 0.0f; }\n"
"        __syncthreads();\n"
"        for (uint32_t k = 0; k < 32u; k++) for (uint32_t tb = 0; tb < 2u; tb++) for (uint32_t rr = 0; rr < 4u; rr++) acc[tb][rr] = fmaf(sX[th_b+tb][k], sW[th_r+rr][k], acc[tb][rr]);\n"
"        __syncthreads();\n"
"    }\n"
"    for (uint32_t tb = 0; tb < 2u; tb++) for (uint32_t rr = 0; rr < 4u; rr++) { uint32_t ob = b0+th_b+tb, orw = r0+th_r+rr; if (ob < batch_b && orw < out_dim) { uint64_t idx = (uint64_t)ob*out_dim + orw; y[idx] = add_res ? (y[idx] + acc[tb][rr]) : acc[tb][rr]; } }\n"
"#endif\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_fp8_e4m3_tc(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,\n"
"                                               uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    uint32_t r0 = blockIdx.x * 32u, b0 = blockIdx.y * 32u;\n"
"    if (r0 >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t lid = threadIdx.x, nb = in_dim >> 5;\n"
"    uint64_t rb = (uint64_t)nb * 34ull;\n"
"    uint32_t tr = lid >> 2, g = lid & 3u, c0 = g * 8u, gr = r0 + tr;\n"
"    const uint8_t* wrow = w + (uint64_t)(gr < out_dim ? gr : (out_dim - 1u)) * rb;\n"
"#if __CUDA_ARCH__ >= 800\n"
"    __shared__ float sXf[2][32][32];\n"
"    __shared__ uint16_t sX[32][34];\n"
"    __shared__ uint16_t sW[32][34];\n"
"    uint32_t tr_x0 = lid >> 3, tr_x1 = tr_x0 + 16u, cx4 = (lid & 7u) * 4u;\n"
"    uint32_t gb0 = (b0 + tr_x0 < batch_b) ? (b0 + tr_x0) : (batch_b - 1u);\n"
"    uint32_t gb1 = (b0 + tr_x1 < batch_b) ? (b0 + tr_x1) : (batch_b - 1u);\n"
"    int xb0 = (b0 + tr_x0 < batch_b) ? 16 : 0, xb1 = (b0 + tr_x1 < batch_b) ? 16 : 0;\n"
"    uint32_t wid = lid >> 5, lane = lid & 31u, gid = lane >> 2, tid = lane & 3u;\n"
"    uint32_t wb = (wid >> 1) * 16u, wr = (wid & 1u) * 16u;\n"
"    int warp_act = (b0 + wb < batch_b) && (r0 + wr < out_dim);\n"
"    float c00=0,c01=0,c02=0,c03=0, c10=0,c11=0,c12=0,c13=0;\n"
"    cm_cp_async16(&sXf[0][tr_x0][cx4], x + (uint64_t)gb0 * in_dim + cx4, xb0);\n"
"    cm_cp_async16(&sXf[0][tr_x1][cx4], x + (uint64_t)gb1 * in_dim + cx4, xb1);\n"
"    asm volatile(\"cp.async.commit_group;\\n\");\n"
"    for (uint32_t bk = 0; bk < nb; bk++) {\n"
"        uint32_t st = bk & 1u, nxt = st ^ 1u;\n"
"        if (bk + 1u < nb) {\n"
"            uint32_t nk = ((bk + 1u) << 5) + cx4;\n"
"            cm_cp_async16(&sXf[nxt][tr_x0][cx4], x + (uint64_t)gb0 * in_dim + nk, xb0);\n"
"            cm_cp_async16(&sXf[nxt][tr_x1][cx4], x + (uint64_t)gb1 * in_dim + nk, xb1);\n"
"        }\n"
"        asm volatile(\"cp.async.commit_group;\\n\");\n"
"        if (gr < out_dim) {\n"
"            const uint8_t* blk = wrow + (uint64_t)bk * 34ull;\n"
"            float d = load_f16_le(blk);\n"
"            #pragma unroll\n"
"            for (uint32_t j = 0; j < 8u; j += 2u) {\n"
"                float w0 = d * cm_fp8_e4m3_dec(__ldg(&blk[2u + c0 + j]));\n"
"                float w1 = d * cm_fp8_e4m3_dec(__ldg(&blk[2u + c0 + j + 1u]));\n"
"                __half2 wh = __float22half2_rn(make_float2(w0, w1));\n"
"                *(uint32_t*)&sW[tr][c0 + j] = *reinterpret_cast<const uint32_t*>(&wh);\n"
"            }\n"
"        } else {\n"
"            #pragma unroll\n"
"            for (uint32_t j = 0; j < 8u; j += 2u) *(uint32_t*)&sW[tr][c0 + j] = 0u;\n"
"        }\n"
"        asm volatile(\"cp.async.wait_group 1;\\n\");\n"
"        __syncthreads();\n"
"        __half2 h00 = __float22half2_rn(make_float2(sXf[st][tr_x0][cx4+0u], sXf[st][tr_x0][cx4+1u]));\n"
"        __half2 h01 = __float22half2_rn(make_float2(sXf[st][tr_x0][cx4+2u], sXf[st][tr_x0][cx4+3u]));\n"
"        __half2 h10 = __float22half2_rn(make_float2(sXf[st][tr_x1][cx4+0u], sXf[st][tr_x1][cx4+1u]));\n"
"        __half2 h11 = __float22half2_rn(make_float2(sXf[st][tr_x1][cx4+2u], sXf[st][tr_x1][cx4+3u]));\n"
"        *(uint32_t*)&sX[tr_x0][cx4+0u] = *reinterpret_cast<const uint32_t*>(&h00);\n"
"        *(uint32_t*)&sX[tr_x0][cx4+2u] = *reinterpret_cast<const uint32_t*>(&h01);\n"
"        *(uint32_t*)&sX[tr_x1][cx4+0u] = *reinterpret_cast<const uint32_t*>(&h10);\n"
"        *(uint32_t*)&sX[tr_x1][cx4+2u] = *reinterpret_cast<const uint32_t*>(&h11);\n"
"        __syncthreads();\n"
"        if (warp_act) {\n"
"            #pragma unroll\n"
"            for (uint32_t kk = 0; kk < 32u; kk += 16u) {\n"
"                uint32_t a0 = *(const uint32_t*)&sX[wb + gid][kk + tid * 2u];\n"
"                uint32_t a1 = *(const uint32_t*)&sX[wb + gid + 8u][kk + tid * 2u];\n"
"                uint32_t a2 = *(const uint32_t*)&sX[wb + gid][kk + 8u + tid * 2u];\n"
"                uint32_t a3 = *(const uint32_t*)&sX[wb + gid + 8u][kk + 8u + tid * 2u];\n"
"                uint32_t b00 = *(const uint32_t*)&sW[wr + gid][kk + tid * 2u];\n"
"                uint32_t b01 = *(const uint32_t*)&sW[wr + gid][kk + 8u + tid * 2u];\n"
"                asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
"                             : \"+f\"(c00), \"+f\"(c01), \"+f\"(c02), \"+f\"(c03) : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b00), \"r\"(b01));\n"
"                uint32_t b10 = *(const uint32_t*)&sW[wr + 8u + gid][kk + tid * 2u];\n"
"                uint32_t b11 = *(const uint32_t*)&sW[wr + 8u + gid][kk + 8u + tid * 2u];\n"
"                asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
"                             : \"+f\"(c10), \"+f\"(c11), \"+f\"(c12), \"+f\"(c13) : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b10), \"r\"(b11));\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    asm volatile(\"cp.async.wait_all;\\n\");\n"
"    if (warp_act) {\n"
"        uint32_t ob0 = b0 + wb + gid, ob1 = ob0 + 8u, or0 = r0 + wr + tid * 2u, or1 = or0 + 8u;\n"
"        #define ST_M(ob, orw, val) if ((ob) < batch_b && (orw) < out_dim) { uint64_t idx = (uint64_t)(ob) * out_dim + (orw); y[idx] = add_res ? (y[idx] + (val)) : (val); }\n"
"        ST_M(ob0, or0, c00); ST_M(ob0, or0+1u, c01); ST_M(ob1, or0, c02); ST_M(ob1, or0+1u, c03);\n"
"        ST_M(ob0, or1, c10); ST_M(ob0, or1+1u, c11); ST_M(ob1, or1, c12); ST_M(ob1, or1+1u, c13);\n"
"        #undef ST_M\n"
"    }\n"
"#else\n"
"    __shared__ float sX[32][33]; __shared__ float sW[32][33];\n"
"    uint32_t gb = b0 + tr, th_b = (lid >> 3) * 2u, th_r = (lid & 7u) * 4u;\n"
"    float acc[2][4] = {{0,0,0,0},{0,0,0,0}};\n"
"    for (uint32_t bk = 0; bk < nb; bk++) {\n"
"        if (gb < batch_b) { for (uint32_t j = 0; j < 8u; j++) sX[tr][c0+j] = x[(uint64_t)gb*in_dim + (bk<<5) + c0 + j]; }\n"
"        else { for (uint32_t j = 0; j < 8u; j++) sX[tr][c0+j] = 0.0f; }\n"
"        if (gr < out_dim) { const uint8_t* blk = wrow + (uint64_t)bk * 34ull; float d = load_f16_le(blk); for (uint32_t j = 0; j < 8u; j++) sW[tr][c0+j] = d * cm_fp8_e4m3_dec(blk[2u+c0+j]); }\n"
"        else { for (uint32_t j = 0; j < 8u; j++) sW[tr][c0+j] = 0.0f; }\n"
"        __syncthreads();\n"
"        for (uint32_t k = 0; k < 32u; k++) for (uint32_t tb = 0; tb < 2u; tb++) for (uint32_t rr = 0; rr < 4u; rr++) acc[tb][rr] = fmaf(sX[th_b+tb][k], sW[th_r+rr][k], acc[tb][rr]);\n"
"        __syncthreads();\n"
"    }\n"
"    for (uint32_t tb = 0; tb < 2u; tb++) for (uint32_t rr = 0; rr < 4u; rr++) { uint32_t ob = b0+th_b+tb, orw = r0+th_r+rr; if (ob < batch_b && orw < out_dim) { uint64_t idx = (uint64_t)ob*out_dim + orw; y[idx] = add_res ? (y[idx] + acc[tb][rr]) : acc[tb][rr]; } }\n"
"#endif\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gemm_mxfp4_tc(const uint8_t* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,\n"
"                                            uint32_t out_dim, uint32_t in_dim, uint32_t batch_b, uint32_t add_res) {\n"
"    const float kMx4[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };\n"
"    uint32_t r0 = blockIdx.x * 32u, b0 = blockIdx.y * 32u;\n"
"    if (r0 >= out_dim || b0 >= batch_b) return;\n"
"    uint32_t lid = threadIdx.x, nb = in_dim >> 5;\n"
"    uint64_t rb = (uint64_t)nb * 17ull;\n"
"    uint32_t tr = lid >> 2, g = lid & 3u, c0 = g * 8u, gr = r0 + tr;\n"
"    const uint8_t* wrow = w + (uint64_t)(gr < out_dim ? gr : (out_dim - 1u)) * rb;\n"
"#if __CUDA_ARCH__ >= 800\n"
"    __shared__ float sXf[2][32][32];\n"
"    __shared__ uint16_t sX[32][34];\n"
"    __shared__ uint16_t sW[32][34];\n"
"    uint32_t tr_x0 = lid >> 3, tr_x1 = tr_x0 + 16u, cx4 = (lid & 7u) * 4u;\n"
"    uint32_t gb0 = (b0 + tr_x0 < batch_b) ? (b0 + tr_x0) : (batch_b - 1u);\n"
"    uint32_t gb1 = (b0 + tr_x1 < batch_b) ? (b0 + tr_x1) : (batch_b - 1u);\n"
"    int xb0 = (b0 + tr_x0 < batch_b) ? 16 : 0, xb1 = (b0 + tr_x1 < batch_b) ? 16 : 0;\n"
"    uint32_t wid = lid >> 5, lane = lid & 31u, gid = lane >> 2, tid = lane & 3u;\n"
"    uint32_t wb = (wid >> 1) * 16u, wr = (wid & 1u) * 16u;\n"
"    int warp_act = (b0 + wb < batch_b) && (r0 + wr < out_dim);\n"
"    float c00=0,c01=0,c02=0,c03=0, c10=0,c11=0,c12=0,c13=0;\n"
"    cm_cp_async16(&sXf[0][tr_x0][cx4], x + (uint64_t)gb0 * in_dim + cx4, xb0);\n"
"    cm_cp_async16(&sXf[0][tr_x1][cx4], x + (uint64_t)gb1 * in_dim + cx4, xb1);\n"
"    asm volatile(\"cp.async.commit_group;\\n\");\n"
"    for (uint32_t bk = 0; bk < nb; bk++) {\n"
"        uint32_t st = bk & 1u, nxt = st ^ 1u;\n"
"        if (bk + 1u < nb) {\n"
"            uint32_t nk = ((bk + 1u) << 5) + cx4;\n"
"            cm_cp_async16(&sXf[nxt][tr_x0][cx4], x + (uint64_t)gb0 * in_dim + nk, xb0);\n"
"            cm_cp_async16(&sXf[nxt][tr_x1][cx4], x + (uint64_t)gb1 * in_dim + nk, xb1);\n"
"        }\n"
"        asm volatile(\"cp.async.commit_group;\\n\");\n"
"        if (gr < out_dim) {\n"
"            const uint8_t* blk = wrow + (uint64_t)bk * 17ull;\n"
"            float sc = __uint_as_float(((uint32_t)__ldg(&blk[0])) << 23);\n"
"            uint32_t hi = (c0 >= 16u) ? 1u : 0u, boff = hi ? (c0 - 16u) : c0;\n"
"            #pragma unroll\n"
"            for (uint32_t j = 0; j < 8u; j += 2u) {\n"
"                uint8_t bv0 = __ldg(&blk[1u + boff + j]), bv1 = __ldg(&blk[1u + boff + j + 1u]);\n"
"                float w0 = sc * kMx4[hi ? (bv0 >> 4) : (bv0 & 0xFu)];\n"
"                float w1 = sc * kMx4[hi ? (bv1 >> 4) : (bv1 & 0xFu)];\n"
"                __half2 wh = __float22half2_rn(make_float2(w0, w1));\n"
"                *(uint32_t*)&sW[tr][c0 + j] = *reinterpret_cast<const uint32_t*>(&wh);\n"
"            }\n"
"        } else {\n"
"            #pragma unroll\n"
"            for (uint32_t j = 0; j < 8u; j += 2u) *(uint32_t*)&sW[tr][c0 + j] = 0u;\n"
"        }\n"
"        asm volatile(\"cp.async.wait_group 1;\\n\");\n"
"        __syncthreads();\n"
"        __half2 h00 = __float22half2_rn(make_float2(sXf[st][tr_x0][cx4+0u], sXf[st][tr_x0][cx4+1u]));\n"
"        __half2 h01 = __float22half2_rn(make_float2(sXf[st][tr_x0][cx4+2u], sXf[st][tr_x0][cx4+3u]));\n"
"        __half2 h10 = __float22half2_rn(make_float2(sXf[st][tr_x1][cx4+0u], sXf[st][tr_x1][cx4+1u]));\n"
"        __half2 h11 = __float22half2_rn(make_float2(sXf[st][tr_x1][cx4+2u], sXf[st][tr_x1][cx4+3u]));\n"
"        *(uint32_t*)&sX[tr_x0][cx4+0u] = *reinterpret_cast<const uint32_t*>(&h00);\n"
"        *(uint32_t*)&sX[tr_x0][cx4+2u] = *reinterpret_cast<const uint32_t*>(&h01);\n"
"        *(uint32_t*)&sX[tr_x1][cx4+0u] = *reinterpret_cast<const uint32_t*>(&h10);\n"
"        *(uint32_t*)&sX[tr_x1][cx4+2u] = *reinterpret_cast<const uint32_t*>(&h11);\n"
"        __syncthreads();\n"
"        if (warp_act) {\n"
"            #pragma unroll\n"
"            for (uint32_t kk = 0; kk < 32u; kk += 16u) {\n"
"                uint32_t a0 = *(const uint32_t*)&sX[wb + gid][kk + tid * 2u];\n"
"                uint32_t a1 = *(const uint32_t*)&sX[wb + gid + 8u][kk + tid * 2u];\n"
"                uint32_t a2 = *(const uint32_t*)&sX[wb + gid][kk + 8u + tid * 2u];\n"
"                uint32_t a3 = *(const uint32_t*)&sX[wb + gid + 8u][kk + 8u + tid * 2u];\n"
"                uint32_t b00 = *(const uint32_t*)&sW[wr + gid][kk + tid * 2u];\n"
"                uint32_t b01 = *(const uint32_t*)&sW[wr + gid][kk + 8u + tid * 2u];\n"
"                asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
"                             : \"+f\"(c00), \"+f\"(c01), \"+f\"(c02), \"+f\"(c03) : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b00), \"r\"(b01));\n"
"                uint32_t b10 = *(const uint32_t*)&sW[wr + 8u + gid][kk + tid * 2u];\n"
"                uint32_t b11 = *(const uint32_t*)&sW[wr + 8u + gid][kk + 8u + tid * 2u];\n"
"                asm volatile(\"mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
"                             : \"+f\"(c10), \"+f\"(c11), \"+f\"(c12), \"+f\"(c13) : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"r\"(b10), \"r\"(b11));\n"
"            }\n"
"        }\n"
"        __syncthreads();\n"
"    }\n"
"    asm volatile(\"cp.async.wait_all;\\n\");\n"
"    if (warp_act) {\n"
"        uint32_t ob0 = b0 + wb + gid, ob1 = ob0 + 8u, or0 = r0 + wr + tid * 2u, or1 = or0 + 8u;\n"
"        #define ST_M(ob, orw, val) if ((ob) < batch_b && (orw) < out_dim) { uint64_t idx = (uint64_t)(ob) * out_dim + (orw); y[idx] = add_res ? (y[idx] + (val)) : (val); }\n"
"        ST_M(ob0, or0, c00); ST_M(ob0, or0+1u, c01); ST_M(ob1, or0, c02); ST_M(ob1, or0+1u, c03);\n"
"        ST_M(ob0, or1, c10); ST_M(ob0, or1+1u, c11); ST_M(ob1, or1, c12); ST_M(ob1, or1+1u, c13);\n"
"        #undef ST_M\n"
"    }\n"
"#else\n"
"    __shared__ float sX[32][33]; __shared__ float sW[32][33];\n"
"    uint32_t gb = b0 + tr, th_b = (lid >> 3) * 2u, th_r = (lid & 7u) * 4u;\n"
"    float acc[2][4] = {{0,0,0,0},{0,0,0,0}};\n"
"    for (uint32_t bk = 0; bk < nb; bk++) {\n"
"        if (gb < batch_b) { for (uint32_t j = 0; j < 8u; j++) sX[tr][c0+j] = x[(uint64_t)gb*in_dim + (bk<<5) + c0 + j]; }\n"
"        else { for (uint32_t j = 0; j < 8u; j++) sX[tr][c0+j] = 0.0f; }\n"
"        if (gr < out_dim) {\n"
"            const uint8_t* blk = wrow + (uint64_t)bk * 17ull; float sc = __uint_as_float(((uint32_t)blk[0]) << 23);\n"
"            uint32_t hi = (c0 >= 16u) ? 1u : 0u, boff = hi ? (c0 - 16u) : c0;\n"
"            for (uint32_t j = 0; j < 8u; j++) { uint8_t bv = blk[1u+boff+j]; sW[tr][c0+j] = sc * kMx4[hi ? (bv>>4) : (bv&0xFu)]; }\n"
"        } else { for (uint32_t j = 0; j < 8u; j++) sW[tr][c0+j] = 0.0f; }\n"
"        __syncthreads();\n"
"        for (uint32_t k = 0; k < 32u; k++) for (uint32_t tb = 0; tb < 2u; tb++) for (uint32_t rr = 0; rr < 4u; rr++) acc[tb][rr] = fmaf(sX[th_b+tb][k], sW[th_r+rr][k], acc[tb][rr]);\n"
"        __syncthreads();\n"
"    }\n"
"    for (uint32_t tb = 0; tb < 2u; tb++) for (uint32_t rr = 0; rr < 4u; rr++) { uint32_t ob = b0+th_b+tb, orw = r0+th_r+rr; if (ob < batch_b && orw < out_dim) { uint64_t idx = (uint64_t)ob*out_dim + orw; y[idx] = add_res ? (y[idx] + acc[tb][rr]) : acc[tb][rr]; } }\n"
"#endif\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_mla_rope_store_paged(float* q, float* c_kv_pe, half* lat_cache, const float* rope_freqs,\n"
"                                                   const uint32_t* pos_arr, const uint32_t* slot_arr, const int32_t* block_table,\n"
"                                                   uint32_t n_heads, uint32_t nope_d, uint32_t rope_d, uint32_t d_head, uint32_t kv_r,\n"
"                                                   float theta, uint32_t rope_neox, uint32_t has_freqs, uint32_t batch_b, uint32_t max_blks) {\n"
"    uint32_t b = blockIdx.x, lid = threadIdx.x;\n"
"    if (b >= batch_b) return;\n"
"    uint32_t pos = pos_arr[b], slot = slot_arr[b];\n"
"    uint32_t pb = (uint32_t)block_table[(uint64_t)slot * max_blks + (pos >> 4)];\n"
"    uint32_t phys_pos = (pb << 4) | (pos & 15u);\n"
"    uint32_t half_r = rope_d >> 1, ldim = kv_r + rope_d;\n"
"    uint64_t qb0 = (uint64_t)b * n_heads * d_head;\n"
"    float* kvb = c_kv_pe + (uint64_t)b * ldim;\n"
"    float* kpe = kvb + kv_r;\n"
"    uint32_t tot_qh = n_heads * half_r;\n"
"    for (uint32_t idx = lid; idx < tot_qh; idx += 128u) {\n"
"        uint32_t h = idx / half_r, i = idx - h * half_r;\n"
"        uint32_t idx0 = rope_neox ? i : (2u * i), idx1 = rope_neox ? (i + half_r) : (2u * i + 1u);\n"
"        float inv_f = 1.0f / powf(theta, (float)(2u * i) / (float)rope_d);\n"
"        if (has_freqs) inv_f /= rope_freqs[i];\n"
"        float ang = (float)pos * inv_f, cs = cosf(ang), sn = sinf(ang);\n"
"        uint64_t qbase = qb0 + (uint64_t)h * d_head + nope_d;\n"
"        float q0 = q[qbase + idx0], q1 = q[qbase + idx1];\n"
"        q[qbase + idx0] = q0 * cs - q1 * sn; q[qbase + idx1] = q0 * sn + q1 * cs;\n"
"    }\n"
"    for (uint32_t i = lid; i < half_r; i += 128u) {\n"
"        uint32_t idx0 = rope_neox ? i : (2u * i), idx1 = rope_neox ? (i + half_r) : (2u * i + 1u);\n"
"        float inv_f = 1.0f / powf(theta, (float)(2u * i) / (float)rope_d);\n"
"        if (has_freqs) inv_f /= rope_freqs[i];\n"
"        float ang = (float)pos * inv_f, cs = cosf(ang), sn = sinf(ang);\n"
"        float k0 = kpe[idx0], k1 = kpe[idx1];\n"
"        kpe[idx0] = k0 * cs - k1 * sn; kpe[idx1] = k0 * sn + k1 * cs;\n"
"    }\n"
"    __syncthreads();\n"
"    half* dst = lat_cache + (uint64_t)phys_pos * ldim;\n"
"    for (uint32_t d = lid; d < ldim; d += 128u) dst[d] = __float2half(kvb[d]);\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_flash_mla_latent_paged(const float* q_abs, const half* lat_cache, float* o_lat,\n"
"                                                     const uint32_t* pos_arr, const uint32_t* slot_arr, const int32_t* block_table,\n"
"                                                     uint32_t n_heads, uint32_t kv_r, uint32_t rope_d, float scale, uint32_t batch_b, uint32_t max_blks) {\n"
"    uint32_t h = blockIdx.x, b = blockIdx.y;\n"
"    if (h >= n_heads || b >= batch_b) return;\n"
"    uint32_t kv_len = pos_arr[b] + 1u, slot = slot_arr[b];\n"
"    const int32_t* bt = block_table + (uint64_t)slot * max_blks;\n"
"    uint32_t ldim = kv_r + rope_d, lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    __shared__ float s_q[768], s_m[4], s_l[4], s_o[4][576];\n"
"    uint64_t qbase = ((uint64_t)b * n_heads + h) * ldim;\n"
"    for (uint32_t d = lid; d < ldim && d < 768u; d += 128u) s_q[d] = q_abs[qbase + d];\n"
"    __syncthreads();\n"
"    float m_run = -1e30f, l_run = 0.0f, o_acc[18];\n"
"    uint32_t dpl = (kv_r + 31u) >> 5, lpl = (ldim + 31u) >> 5;\n"
"    for (uint32_t k = 0; k < dpl && k < 18u; k++) o_acc[k] = 0.0f;\n"
"    for (uint32_t t = wid; t < kv_len; t += 4u) {\n"
"        uint32_t pt = ((uint32_t)bt[t >> 4] << 4) | (t & 15u);\n"
"        const half* lp = lat_cache + (uint64_t)pt * ldim;\n"
"        float dot = 0.0f;\n"
"        for (uint32_t k = 0; k < lpl; k++) { uint32_t d = lane + k * 32u; if (d < ldim) dot = fmaf(s_q[d], __half2float(lp[d]), dot); }\n"
"        float sc = warp_sum(dot) * scale; sc = __shfl_sync(0xFFFFFFFFu, sc, 0);\n"
"        float mn = fmaxf(m_run, sc), al = expf(m_run - mn), wt = expf(sc - mn);\n"
"        l_run = fmaf(l_run, al, wt); m_run = mn;\n"
"        for (uint32_t k = 0; k < dpl && k < 18u; k++) { uint32_t d = lane + k * 32u; if (d < kv_r) o_acc[k] = fmaf(o_acc[k], al, wt * __half2float(lp[d])); }\n"
"    }\n"
"    if (lane == 0u) { s_m[wid] = m_run; s_l[wid] = l_run; }\n"
"    for (uint32_t k = 0; k < dpl && k < 18u; k++) { uint32_t d = lane + k * 32u; if (d < kv_r && d < 576u) s_o[wid][d] = o_acc[k]; }\n"
"    __syncthreads();\n"
"    float gm = fmaxf(fmaxf(s_m[0], s_m[1]), fmaxf(s_m[2], s_m[3])), gl = 0.0f;\n"
"    for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) gl += s_l[s] * expf(s_m[s] - gm);\n"
"    float inv = 1.0f / fmaxf(gl, 1e-12f);\n"
"    uint64_t obase = ((uint64_t)b * n_heads + h) * kv_r;\n"
"    for (uint32_t d = lid; d < kv_r && d < 576u; d += 128u) {\n"
"        float ov = 0.0f; for (uint32_t s = 0; s < 4u; s++) if (s_l[s] > 0.0f) ov = fmaf(s_o[s][d], expf(s_m[s] - gm), ov);\n"
"        o_lat[obase + d] = ov * inv;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_gather_rows(const float* src, float* dst, const uint32_t* tok_idx, uint32_t dim, uint32_t m_rows) {\n"
"    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;\n"
"    if (d < dim && r < m_rows) dst[(uint64_t)r * dim + d] = src[(uint64_t)tok_idx[r] * dim + d];\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_scatter_add_rows(const float* src, float* dst, const uint32_t* tok_idx, const float* wts, uint32_t dim, uint32_t m_rows) {\n"
"    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;\n"
"    if (d < dim && r < m_rows) dst[(uint64_t)tok_idx[r] * dim + d] += wts[r] * src[(uint64_t)r * dim + d];\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_lora_proj_a(const float* A, const float* x, float* tmp, uint32_t in_dim, uint32_t rank, uint32_t batch_b) {\n"
"    uint32_t r = blockIdx.x, b = blockIdx.y, lane = threadIdx.x & 31u;\n"
"    if (r >= rank || b >= batch_b) return;\n"
"    const float* ar = A + (uint64_t)r * in_dim; const float* xb = x + (uint64_t)b * in_dim;\n"
"    float acc = 0.0f;\n"
"    for (uint32_t k = lane; k < in_dim; k += 32u) acc = fmaf(ar[k], xb[k], acc);\n"
"    float s = warp_sum(acc);\n"
"    if (lane == 0u) tmp[(uint64_t)b * rank + r] = s;\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_lora_proj_b_add(const float* B, const float* tmp, float* y, uint32_t out_dim, uint32_t rank, uint32_t batch_b, float scale) {\n"
"    uint32_t o = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y;\n"
"    if (o >= out_dim || b >= batch_b) return;\n"
"    const float* bo = B + (uint64_t)o * rank; const float* tb = tmp + (uint64_t)b * rank;\n"
"    float acc = 0.0f;\n"
"    for (uint32_t r = 0; r < rank; r++) acc = fmaf(bo[r], tb[r], acc);\n"
"    y[(uint64_t)b * out_dim + o] += scale * acc;\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_argmax_batch(const float* logits, uint32_t* out_toks, uint32_t vocab, uint32_t batch_b) {\n"
"    uint32_t b = blockIdx.x; if (b >= batch_b) return;\n"
"    const float* row = logits + (uint64_t)b * vocab;\n"
"    __shared__ float s_v[8]; __shared__ uint32_t s_i[8];\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    float bv = -1e30f; uint32_t bi = 0u;\n"
"    for (uint32_t i = lid; i < vocab; i += 256u) { float v = row[i]; if (v > bv) { bv = v; bi = i; } }\n"
"    for (int off = 16; off > 0; off >>= 1) {\n"
"        float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, off); uint32_t oi = __shfl_xor_sync(0xFFFFFFFFu, bi, off);\n"
"        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }\n"
"    }\n"
"    if (lane == 0u) { s_v[wid] = bv; s_i[wid] = bi; }\n"
"    __syncthreads();\n"
"    if (lid == 0u) {\n"
"        float best_v = s_v[0]; uint32_t best_i = s_i[0];\n"
"        for (uint32_t s = 1u; s < 8u; s++) if (s_v[s] > best_v) { best_v = s_v[s]; best_i = s_i[s]; }\n"
"        out_toks[b] = best_i;\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_rope_and_kv_mrope(float* q, float* k, const float* v, half* k_cache, half* v_cache, const float* rope_freqs,\n"
"                                                uint32_t n_heads, uint32_t n_kv, uint32_t d_head, uint32_t rope_dim, uint32_t pos,\n"
"                                                float theta, uint32_t rope_neox, uint32_t has_freqs,\n"
"                                                uint32_t s0, uint32_t s1, uint32_t pt, uint32_t ph, uint32_t pw) {\n"
"    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x, h = blockIdx.y;\n"
"    uint32_t half_d = d_head >> 1, half_r = rope_dim >> 1;\n"
"    if (i >= half_d || h >= n_heads) return;\n"
"    if (i < half_r) {\n"
"        uint32_t idx0 = rope_neox ? i : (2u * i), idx1 = rope_neox ? (i + half_r) : (2u * i + 1u);\n"
"        float inv_f = 1.0f / powf(theta, (float)(2u * i) / (float)rope_dim);\n"
"        if (has_freqs) inv_f /= rope_freqs[i];\n"
"        uint32_t eff_pos = (i < s0) ? pt : ((i < s0 + s1) ? ph : pw);\n"
"        float ang = (float)eff_pos * inv_f, cs = cosf(ang), sn = sinf(ang);\n"
"        uint32_t qoff = h * d_head; float q0 = q[qoff + idx0], q1 = q[qoff + idx1];\n"
"        q[qoff + idx0] = q0 * cs - q1 * sn; q[qoff + idx1] = q0 * sn + q1 * cs;\n"
"        if (h < n_kv) {\n"
"            uint32_t koff = h * d_head; float k0 = k[koff + idx0], k1 = k[koff + idx1];\n"
"            float rk0 = k0 * cs - k1 * sn, rk1 = k0 * sn + k1 * cs;\n"
"            k[koff + idx0] = rk0; k[koff + idx1] = rk1;\n"
"            uint64_t cb = (uint64_t)pos * (n_kv * d_head) + koff;\n"
"            k_cache[cb + idx0] = __float2half(rk0); k_cache[cb + idx1] = __float2half(rk1);\n"
"            v_cache[cb + idx0] = __float2half(v[koff + idx0]); v_cache[cb + idx1] = __float2half(v[koff + idx1]);\n"
"        }\n"
"    } else if (h < n_kv) {\n"
"        uint32_t idx0 = 2u * i, idx1 = 2u * i + 1u;\n"
"        uint32_t koff = h * d_head; uint64_t cb = (uint64_t)pos * (n_kv * d_head) + koff;\n"
"        k_cache[cb + idx0] = __float2half(k[koff + idx0]); k_cache[cb + idx1] = __float2half(k[koff + idx1]);\n"
"        v_cache[cb + idx0] = __float2half(v[koff + idx0]); v_cache[cb + idx1] = __float2half(v[koff + idx1]);\n"
"    }\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_router_topk(const float* logits, const float* exp_bias, uint32_t* out_e, float* out_w,\n"
"                                          uint32_t n_exp, uint32_t top_k, uint32_t norm_wt, uint32_t gating_func,\n"
"                                          uint32_t has_bias, float w_scale, uint32_t n_groups, uint32_t n_group_used) {\n"
"    if (threadIdx.x != 0u) return;\n"
"    float probs[256], scores[256];\n"
"    if (gating_func == 2u) {\n"
"        for (uint32_t e = 0; e < n_exp; e++) { float s = 1.0f / (1.0f + expf(-logits[e])); probs[e] = s; scores[e] = has_bias ? (s + exp_bias[e]) : s; }\n"
"    } else {\n"
"        float mx = -1e30f; for (uint32_t e = 0; e < n_exp; e++) if (logits[e] > mx) mx = logits[e];\n"
"        float sum = 0.0f; for (uint32_t e = 0; e < n_exp; e++) { probs[e] = expf(logits[e] - mx); sum += probs[e]; }\n"
"        float inv = 1.0f / fmaxf(sum, 1e-12f);\n"
"        for (uint32_t e = 0; e < n_exp; e++) { probs[e] *= inv; scores[e] = probs[e]; }\n"
"    }\n"
"    if (n_groups > 1u && n_group_used > 0u && n_group_used < n_groups) {\n"
"        uint32_t gsz = n_exp / n_groups; float gsc[32]; int gkeep[32];\n"
"        for (uint32_t g = 0; g < n_groups && g < 32u; g++) {\n"
"            float m1 = -1e30f, m2 = -1e30f;\n"
"            for (uint32_t i = 0; i < gsz; i++) { float sv = scores[g*gsz + i]; if (sv > m1) { m2 = m1; m1 = sv; } else if (sv > m2) m2 = sv; }\n"
"            gsc[g] = (gating_func == 2u) ? (m1 + m2) : m1; gkeep[g] = 0;\n"
"        }\n"
"        for (uint32_t u = 0; u < n_group_used; u++) {\n"
"            uint32_t bg = 0; float bs = -1e30f;\n"
"            for (uint32_t g = 0; g < n_groups && g < 32u; g++) if (!gkeep[g] && gsc[g] > bs) { bs = gsc[g]; bg = g; }\n"
"            gkeep[bg] = 1;\n"
"        }\n"
"        for (uint32_t g = 0; g < n_groups && g < 32u; g++) if (!gkeep[g]) for (uint32_t i = 0; i < gsz; i++) scores[g*gsz + i] = -1e30f;\n"
"    }\n"
"    float wsum = 0.0f;\n"
"    for (uint32_t k = 0; k < top_k; k++) {\n"
"        uint32_t be = 0; float bs = -1e30f;\n"
"        for (uint32_t e = 0; e < n_exp; e++) if (scores[e] > bs) { bs = scores[e]; be = e; }\n"
"        out_e[k] = be; out_w[k] = probs[be]; wsum += probs[be]; scores[be] = -1e30f;\n"
"    }\n"
"    float mult = w_scale; if (norm_wt && wsum > 1e-12f) mult /= wsum;\n"
"    for (uint32_t k = 0; k < top_k; k++) out_w[k] *= mult;\n"
"}\n";

static void* cm_sym(void* lib, const char* a, const char* b) {
    if (!lib) return NULL;
    void* p = dlsym(lib, a);
    if (!p && b) p = dlsym(lib, b);
    return p;
}

static char* cm_compile_ptx(const char* src) {
    // Prefer `nvcc --cubin -arch=sm_XX`: emits native SASS ELF directly (avoids CUDA_ERROR_UNSUPPORTED_PTX_VERSION
    // when newer CUDA toolkits run inside containers on older host drivers, AND eliminates driver JIT latency!).
    char cu_path[128], bin_path[128], cmd[512];
    snprintf(cu_path, sizeof(cu_path), "/tmp/tenon_cumoe_%d.cu", (int)getpid());
    snprintf(bin_path, sizeof(bin_path), "/tmp/tenon_cumoe_%d.cubin", (int)getpid());
    FILE* f = fopen(cu_path, "wb");
    if (f) {
        fwrite(src, 1, strlen(src), f);
        fclose(f);
        snprintf(cmd, sizeof(cmd), "PATH=\"/usr/local/cuda/bin:$PATH\" nvcc --cubin -O3 -use_fast_math -arch=sm_%d%d \"%s\" -o \"%s\" 2>/tmp/tenon_cumoe_nvcc.err",
                 g_cu.cc_major, g_cu.cc_minor, cu_path, bin_path);
        if (system(cmd) == 0) {
            FILE* pf = fopen(bin_path, "rb");
            if (pf) {
                fseek(pf, 0, SEEK_END);
                long sz = ftell(pf);
                fseek(pf, 0, SEEK_SET);
                char* bin = (char*)malloc((size_t)sz + 1);
                if (bin && sz > 0) {
                    size_t nr = fread(bin, 1, (size_t)sz, pf);
                    bin[nr] = 0;
                    fclose(pf);
                    unlink(cu_path); unlink(bin_path);
                    return bin;
                }
                free(bin);
                fclose(pf);
            }
        }
        unlink(cu_path); unlink(bin_path);
    }
    if (!g_cu.nvrtcCreateProgram) return NULL;
    nvrtcProgram prog = NULL;
    if (g_cu.nvrtcCreateProgram(&prog, src, "tenon_cumoe.cu", 0, NULL, NULL) != NVRTC_SUCCESS) return NULL;
    char arch_opt[32];
    snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d", g_cu.cc_major, g_cu.cc_minor);
    const char* opts[3] = { arch_opt, "--use_fast_math", "-I/usr/local/cuda/include" };
    if (g_cu.nvrtcCompileProgram(prog, 3, opts) != NVRTC_SUCCESS) {
        size_t lsz = 0;
        g_cu.nvrtcGetProgramLogSize(prog, &lsz);
        if (lsz > 1) {
            char* log = (char*)malloc(lsz);
            if (log) { g_cu.nvrtcGetProgramLog(prog, log); fprintf(stderr, "[tenon-cuda-stream] NVRTC err:\n%s\n", log); free(log); }
        }
        g_cu.nvrtcDestroyProgram(&prog);
        return NULL;
    }
    size_t psz = 0;
    g_cu.nvrtcGetPTXSize(prog, &psz);
    char* ptx = (char*)malloc(psz + 1);
    if (ptx) { g_cu.nvrtcGetPTX(prog, ptx); ptx[psz] = 0; }
    g_cu.nvrtcDestroyProgram(&prog);
    return ptx;
}

static int cm_load_dev_kernels(CmDevCtx* dc, const void* bin) {
    if (g_cu.cuModuleLoadData(&dc->mod, bin) != CUDA_SUCCESS) return 0;
    #define LK(dst, nm) if (g_cu.cuModuleGetFunction(&dc->dst, dc->mod, nm) != CUDA_SUCCESS) { fprintf(stderr, "[tenon-cuda-stream] missing %s\n", nm); return 0; }
    LK(k_embed, "cm_embed_lookup");
    LK(k_embed_ind, "cm_embed_lookup_ind");
    LK(k_rmsnorm, "cm_rmsnorm");
    LK(k_add_bias, "cm_add_bias");
    LK(k_swiglu_mul, "cm_swiglu_mul");
    LK(k_quant_q8_1, "cm_quantize_q8_1");
    LK(k_gemv_q4, "cm_gemv_q4");
    LK(k_gemv_qkv_q4, "cm_gemv_qkv_q4");
    LK(k_gemv_q4_1, "cm_gemv_q4_1");
    LK(k_gemv_q4_dp4a, "cm_gemv_q4_dp4a");
    LK(k_gemv_qkv_q4_dp4a, "cm_gemv_qkv_q4_dp4a");
    LK(k_gemv_q4_1_dp4a, "cm_gemv_q4_1_dp4a");
    LK(k_dense_gu_q4_dp4a, "cm_dense_gate_up_q4_dp4a");
    LK(k_gemv_q5_0, "cm_gemv_q5_0");
    LK(k_gemv_q5_1, "cm_gemv_q5_1");
    LK(k_gemv_q8, "cm_gemv_q8");
    LK(k_gemv_q2k, "cm_gemv_q2k");
    LK(k_gemv_q3k, "cm_gemv_q3k");
    LK(k_gemv_iq4xs, "cm_gemv_iq4xs");
    LK(k_gemv_q4k, "cm_gemv_q4k");
    LK(k_gemv_q5k, "cm_gemv_q5k");
    LK(k_gemv_q6k, "cm_gemv_q6k");
    LK(k_gemv_f16, "cm_gemv_f16");
    LK(k_gemv_f32, "cm_gemv_f32");
    LK(k_dense_gu_q4, "cm_dense_gate_up_q4");
    LK(k_rope_kv, "cm_rope_and_kv");
    LK(k_rope_kv_ind, "cm_rope_and_kv_ind");
    LK(k_flash_dec, "cm_flash_decode");
    LK(k_flash_dec_ind, "cm_flash_decode_ind");
    LK(k_rope_kv_q8_ind, "cm_rope_and_kv_q8_ind");
    LK(k_flash_dec_q8_ind, "cm_flash_decode_q8_ind");
    LK(k_rope_kv_b, "cm_rope_and_kv_batch");
    LK(k_rope_kv_q8_b, "cm_rope_and_kv_q8_batch");
    LK(k_flash_prefill, "cm_flash_prefill_fp16");
    LK(k_flash_prefill_q8, "cm_flash_attn_q8_batch");
    LK(k_gemm_q4_b, "cm_gemm_q4_batch");
    LK(k_gemm_q4_1_b, "cm_gemm_q4_1_batch");
    LK(k_gemm_q5_0_b, "cm_gemm_q5_0_batch");
    LK(k_gemm_q5_1_b, "cm_gemm_q5_1_batch");
    LK(k_gemm_q8_b, "cm_gemm_q8_batch");
    LK(k_gemm_q2k_b, "cm_gemm_q2k_batch");
    LK(k_gemm_q3k_b, "cm_gemm_q3k_batch");
    LK(k_gemm_iq4xs_b, "cm_gemm_iq4xs_batch");
    LK(k_gemm_q4k_b, "cm_gemm_q4k_batch");
    LK(k_gemm_q5k_b, "cm_gemm_q5k_batch");
    LK(k_gemm_q6k_b, "cm_gemm_q6k_batch");
    LK(k_gemm_f16_b, "cm_gemm_f16_batch");
    LK(k_gemm_f32_b, "cm_gemm_f32_batch");
    LK(k_add_bias_b, "cm_add_bias_batch");
    LK(k_mla_split_kv, "cm_mla_split_kv");
    LK(k_mla_rope_pe, "cm_mla_rope_pe");
    LK(k_mla_store_latent, "cm_mla_store_latent");
    LK(k_mla_unpack_wkv_b, "cm_mla_unpack_wkv_b");
    LK(k_mla_unpack_kb_vb, "cm_mla_unpack_kb_vb");
    LK(k_mla_absorb_q, "cm_mla_absorb_q");
    LK(k_flash_mla_latent, "cm_flash_mla_latent");
    LK(k_mla_proj_v, "cm_mla_proj_v");
    LK(k_tp_accum_add, "cm_tp_accum_add");
    LK(k_moe_slot_gu_q4, "cm_slot_gate_up_q4");
    LK(k_moe_slot_down_q4, "cm_slot_down_accum_q4");
    LK(k_moe_slot_down_q4_1, "cm_slot_down_accum_q4_1");
    LK(k_moe_slot_down_q5_0, "cm_slot_down_accum_q5_0");
    LK(k_moe_slot_down_q5_1, "cm_slot_down_accum_q5_1");
    LK(k_moe_slot_down_q8, "cm_slot_down_accum_q8");
    LK(k_moe_slot_gu_q4k, "cm_slot_gate_up_q4k");
    LK(k_moe_slot_down_q4k, "cm_slot_down_accum_q4k");
    LK(k_moe_slot_down_q6k, "cm_slot_down_accum_q6k");
    LK(k_rope_kv_paged, "cm_rope_kv_paged");
    LK(k_rope_kv_q8_paged, "cm_rope_kv_q8_paged");
    LK(k_flash_dec_paged, "cm_flash_decode_paged");
    LK(k_flash_dec_q8_paged, "cm_flash_decode_q8_paged");
    LK(k_rope_kv_tree, "cm_rope_kv_tree");
    LK(k_rope_kv_q8_tree, "cm_rope_kv_q8_tree");
    LK(k_flash_tree, "cm_flash_tree_verify");
    LK(k_flash_tree_q8, "cm_flash_tree_verify_q8");
    LK(k_vit_ln, "cm_vit_layernorm");
    LK(k_vit_gelu, "cm_vit_gelu");
    LK(k_vit_add, "cm_vit_add");
    LK(k_vit_attn, "cm_vit_full_attn");
    LK(k_gemm_q4_sb, "cm_gemm_q4_small_b");
    LK(k_gemm_q4_1_sb, "cm_gemm_q4_1_small_b");
    LK(k_gemm_q4k_sb, "cm_gemm_q4k_small_b");
    LK(k_gemm_q6k_sb, "cm_gemm_q6k_small_b");
    LK(k_gemv_fp8, "cm_gemv_fp8_e4m3");
    LK(k_gemm_fp8_b, "cm_gemm_fp8_e4m3_batch");
    LK(k_gemv_mxfp4, "cm_gemv_mxfp4");
    LK(k_gemm_mxfp4_b, "cm_gemm_mxfp4_batch");
    LK(k_lora_a, "cm_lora_proj_a");
    LK(k_lora_b, "cm_lora_proj_b_add");
    LK(k_argmax_b, "cm_argmax_batch");
    LK(k_rope_kv_mrope, "cm_rope_and_kv_mrope");
    LK(k_gemm_q4k_marlin, "cm_gemm_q4k_marlin");
    LK(k_gemm_q6k_marlin, "cm_gemm_q6k_marlin");
    LK(k_gemm_fp8_tc, "cm_gemm_fp8_e4m3_tc");
    LK(k_gemm_mxfp4_tc, "cm_gemm_mxfp4_tc");
    LK(k_mla_rope_store_paged, "cm_mla_rope_store_paged");
    LK(k_flash_mla_latent_paged, "cm_flash_mla_latent_paged");
    LK(k_gather_rows, "cm_gather_rows");
    LK(k_scatter_add_rows, "cm_scatter_add_rows");
    LK(k_router, "cm_router_topk");
    LK(k_argmax, "cm_argmax_f32");
    #undef LK
    g_cu.cuMemAlloc(&dc->d_xq8, 512u * 1024u);
    dc->ready = 1;
    return 1;
}

int64_t tn_cumoe_init(void) {
    if (g_cu.ready) return 1;
    g_cu.libcuda = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!g_cu.libcuda) g_cu.libcuda = dlopen("libcuda.so", RTLD_LAZY);
    if (!g_cu.libcuda) { fprintf(stderr, "[tenon-cuda-stream] libcuda.so.1 not found\n"); return 0; }

    g_cu.cuInit               = (fn_cuInit)cm_sym(g_cu.libcuda, "cuInit", NULL);
    g_cu.cuDeviceGetCount     = (fn_cuDeviceGetCount)cm_sym(g_cu.libcuda, "cuDeviceGetCount", NULL);
    g_cu.cuDeviceGet          = (fn_cuDeviceGet)cm_sym(g_cu.libcuda, "cuDeviceGet", NULL);
    g_cu.cuDeviceGetName      = (fn_cuDeviceGetName)cm_sym(g_cu.libcuda, "cuDeviceGetName", NULL);
    g_cu.cuDeviceTotalMem     = (fn_cuDeviceTotalMem_v2)cm_sym(g_cu.libcuda, "cuDeviceTotalMem_v2", "cuDeviceTotalMem");
    g_cu.cuDeviceGetAttribute = (fn_cuDeviceGetAttribute)cm_sym(g_cu.libcuda, "cuDeviceGetAttribute", NULL);
    g_cu.cuCtxCreate          = (fn_cuCtxCreate_v2)cm_sym(g_cu.libcuda, "cuCtxCreate_v2", "cuCtxCreate");
    g_cu.cuDevicePrimaryCtxRetain = (fn_cuDevicePrimaryCtxRetain)cm_sym(g_cu.libcuda, "cuDevicePrimaryCtxRetain", NULL);
    g_cu.cuCtxSetCurrent      = (fn_cuCtxSetCurrent)cm_sym(g_cu.libcuda, "cuCtxSetCurrent", NULL);
    g_cu.cuMemAlloc           = (fn_cuMemAlloc_v2)cm_sym(g_cu.libcuda, "cuMemAlloc_v2", "cuMemAlloc");
    g_cu.cuMemFree            = (fn_cuMemFree_v2)cm_sym(g_cu.libcuda, "cuMemFree_v2", "cuMemFree");
    g_cu.cuMemGetInfo         = (fn_cuMemGetInfo_v2)cm_sym(g_cu.libcuda, "cuMemGetInfo_v2", "cuMemGetInfo");
    g_cu.cuMemHostAlloc       = (fn_cuMemHostAlloc)cm_sym(g_cu.libcuda, "cuMemHostAlloc", NULL);
    g_cu.cuMemFreeHost        = (fn_cuMemFreeHost)cm_sym(g_cu.libcuda, "cuMemFreeHost", NULL);
    g_cu.cuMemcpyHtoD         = (fn_cuMemcpyHtoD_v2)cm_sym(g_cu.libcuda, "cuMemcpyHtoD_v2", "cuMemcpyHtoD");
    g_cu.cuMemcpyDtoH         = (fn_cuMemcpyDtoH_v2)cm_sym(g_cu.libcuda, "cuMemcpyDtoH_v2", "cuMemcpyDtoH");
    g_cu.cuMemcpyHtoDAsync    = (fn_cuMemcpyHtoDAsync_v2)cm_sym(g_cu.libcuda, "cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync");
    g_cu.cuMemcpyDtoHAsync    = (fn_cuMemcpyDtoHAsync_v2)cm_sym(g_cu.libcuda, "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync");
    g_cu.cuMemcpyDtoDAsync    = (fn_cuMemcpyDtoDAsync_v2)cm_sym(g_cu.libcuda, "cuMemcpyDtoDAsync_v2", "cuMemcpyDtoDAsync");
    g_cu.cuDeviceCanAccessPeer= (fn_cuDeviceCanAccessPeer)cm_sym(g_cu.libcuda, "cuDeviceCanAccessPeer", NULL);
    g_cu.cuCtxEnablePeerAccess= (fn_cuCtxEnablePeerAccess)cm_sym(g_cu.libcuda, "cuCtxEnablePeerAccess", NULL);
    g_cu.cuMemcpyPeerAsync    = (fn_cuMemcpyPeerAsync)cm_sym(g_cu.libcuda, "cuMemcpyPeerAsync", NULL);
    g_cu.cuMemsetD8           = (fn_cuMemsetD8_v2)cm_sym(g_cu.libcuda, "cuMemsetD8_v2", "cuMemsetD8");
    g_cu.cuMemsetD8Async      = (fn_cuMemsetD8Async)cm_sym(g_cu.libcuda, "cuMemsetD8Async", NULL);
    g_cu.cuStreamCreate       = (fn_cuStreamCreate)cm_sym(g_cu.libcuda, "cuStreamCreate", NULL);
    g_cu.cuStreamDestroy      = (fn_cuStreamDestroy_v2)cm_sym(g_cu.libcuda, "cuStreamDestroy_v2", "cuStreamDestroy");
    g_cu.cuStreamSynchronize  = (fn_cuStreamSynchronize)cm_sym(g_cu.libcuda, "cuStreamSynchronize", NULL);
    g_cu.cuStreamWaitEvent    = (fn_cuStreamWaitEvent)cm_sym(g_cu.libcuda, "cuStreamWaitEvent", NULL);
    g_cu.cuEventCreate        = (fn_cuEventCreate)cm_sym(g_cu.libcuda, "cuEventCreate", NULL);
    g_cu.cuEventDestroy       = (fn_cuEventDestroy_v2)cm_sym(g_cu.libcuda, "cuEventDestroy_v2", "cuEventDestroy");
    g_cu.cuEventRecord        = (fn_cuEventRecord)cm_sym(g_cu.libcuda, "cuEventRecord", NULL);
    g_cu.cuEventSynchronize   = (fn_cuEventSynchronize)cm_sym(g_cu.libcuda, "cuEventSynchronize", NULL);
    g_cu.cuEventQuery         = (fn_cuEventQuery)cm_sym(g_cu.libcuda, "cuEventQuery", NULL);
    g_cu.cuStreamBeginCapture = (fn_cuStreamBeginCapture)cm_sym(g_cu.libcuda, "cuStreamBeginCapture_v2", "cuStreamBeginCapture");
    g_cu.cuStreamEndCapture   = (fn_cuStreamEndCapture)cm_sym(g_cu.libcuda, "cuStreamEndCapture", NULL);
    g_cu.cuGraphInstantiateWithFlags = (fn_cuGraphInstantiateWithFlags)cm_sym(g_cu.libcuda, "cuGraphInstantiateWithFlags", NULL);
    g_cu.cuGraphInstantiate   = (fn_cuGraphInstantiate)cm_sym(g_cu.libcuda, "cuGraphInstantiate_v2", "cuGraphInstantiate");
    g_cu.cuGraphLaunch        = (fn_cuGraphLaunch)cm_sym(g_cu.libcuda, "cuGraphLaunch", NULL);
    g_cu.cuGraphExecDestroy   = (fn_cuGraphExecDestroy)cm_sym(g_cu.libcuda, "cuGraphExecDestroy", NULL);
    g_cu.cuGraphDestroy       = (fn_cuGraphDestroy)cm_sym(g_cu.libcuda, "cuGraphDestroy", NULL);
    g_cu.cuModuleLoadData     = (fn_cuModuleLoadData)cm_sym(g_cu.libcuda, "cuModuleLoadData", NULL);
    g_cu.cuModuleGetFunction  = (fn_cuModuleGetFunction)cm_sym(g_cu.libcuda, "cuModuleGetFunction", NULL);
    g_cu.cuLaunchKernel       = (fn_cuLaunchKernel)cm_sym(g_cu.libcuda, "cuLaunchKernel", NULL);

    g_cu.libnvrtc = dlopen("libnvrtc.so", RTLD_LAZY);
    if (!g_cu.libnvrtc) g_cu.libnvrtc = dlopen("libnvrtc.so.12", RTLD_LAZY);
    if (g_cu.libnvrtc) {
        g_cu.nvrtcCreateProgram     = (fn_nvrtcCreateProgram)cm_sym(g_cu.libnvrtc, "nvrtcCreateProgram", NULL);
        g_cu.nvrtcCompileProgram    = (fn_nvrtcCompileProgram)cm_sym(g_cu.libnvrtc, "nvrtcCompileProgram", NULL);
        g_cu.nvrtcGetPTXSize        = (fn_nvrtcGetPTXSize)cm_sym(g_cu.libnvrtc, "nvrtcGetPTXSize", NULL);
        g_cu.nvrtcGetPTX            = (fn_nvrtcGetPTX)cm_sym(g_cu.libnvrtc, "nvrtcGetPTX", NULL);
        g_cu.nvrtcGetProgramLogSize = (fn_nvrtcGetProgramLogSize)cm_sym(g_cu.libnvrtc, "nvrtcGetProgramLogSize", NULL);
        g_cu.nvrtcGetProgramLog     = (fn_nvrtcGetProgramLog)cm_sym(g_cu.libnvrtc, "nvrtcGetProgramLog", NULL);
        g_cu.nvrtcDestroyProgram    = (fn_nvrtcDestroyProgram)cm_sym(g_cu.libnvrtc, "nvrtcDestroyProgram", NULL);
    }

    if (!g_cu.cuInit || g_cu.cuInit(0) != CUDA_SUCCESS) return 0;
    int cnt = 0;
    if (g_cu.cuDeviceGetCount(&cnt) != CUDA_SUCCESS || cnt <= 0) return 0;
    if (cnt > CM_MAX_TP) cnt = CM_MAX_TP;
    g_cu.n_gpus = cnt;
    if (g_cu.cuDeviceGet(&g_cu.dev, 0) != CUDA_SUCCESS) return 0;
    g_cu.cuDeviceGetName(g_cu.dev_name, sizeof(g_cu.dev_name), g_cu.dev);
    g_cu.cuDeviceTotalMem(&g_cu.vram_bytes, g_cu.dev);
    g_cu.cuDeviceGetAttribute(&g_cu.cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, g_cu.dev);
    g_cu.cuDeviceGetAttribute(&g_cu.cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, g_cu.dev);
    if (g_cu.cuDevicePrimaryCtxRetain) {
        if (g_cu.cuDevicePrimaryCtxRetain(&g_cu.ctx, g_cu.dev) != CUDA_SUCCESS) return 0;
        if (g_cu.cuCtxSetCurrent(g_cu.ctx) != CUDA_SUCCESS) return 0;
    } else {
        if (g_cu.cuCtxCreate(&g_cu.ctx, 0, g_cu.dev) != CUDA_SUCCESS) return 0;
    }

    char* ptx = cm_compile_ptx(kCuMoeSrc);
    if (!ptx) { fprintf(stderr, "[tenon-cuda-stream] failed to compile CUDA kernels\n"); return 0; }
    g_cu.compiled_bin = ptx;

    g_cu.gpus[0].dev = g_cu.dev; g_cu.gpus[0].ctx = g_cu.ctx; g_cu.gpus[0].vram_bytes = g_cu.vram_bytes;
    snprintf(g_cu.gpus[0].dev_name, sizeof(g_cu.gpus[0].dev_name), "%s", g_cu.dev_name);
    if (!cm_load_dev_kernels(&g_cu.gpus[0], ptx)) { fprintf(stderr, "[tenon-cuda-stream] GPU 0 kernel load failed\n"); return 0; }
    g_cu.mod = g_cu.gpus[0].mod;
    g_cu.d_xq8 = g_cu.gpus[0].d_xq8;

    #define COPY_K(nm) g_cu.nm = g_cu.gpus[0].nm
    COPY_K(k_embed); COPY_K(k_embed_ind); COPY_K(k_rmsnorm); COPY_K(k_add_bias); COPY_K(k_swiglu_mul); COPY_K(k_quant_q8_1);
    COPY_K(k_gemv_q4); COPY_K(k_gemv_qkv_q4); COPY_K(k_gemv_q4_1); COPY_K(k_gemv_q5_0); COPY_K(k_gemv_q5_1); COPY_K(k_gemv_q8);
    COPY_K(k_gemv_q2k); COPY_K(k_gemv_q3k); COPY_K(k_gemv_iq4xs);
    COPY_K(k_gemv_q4k); COPY_K(k_gemv_q5k); COPY_K(k_gemv_q6k); COPY_K(k_gemv_f16); COPY_K(k_gemv_f32);
    COPY_K(k_gemv_q4_dp4a); COPY_K(k_gemv_qkv_q4_dp4a); COPY_K(k_gemv_q4_1_dp4a); COPY_K(k_dense_gu_q4_dp4a);
    COPY_K(k_dense_gu_q4); COPY_K(k_rope_kv); COPY_K(k_rope_kv_ind); COPY_K(k_flash_dec); COPY_K(k_flash_dec_ind);
    COPY_K(k_rope_kv_q8_ind); COPY_K(k_flash_dec_q8_ind); COPY_K(k_rope_kv_b); COPY_K(k_rope_kv_q8_b);
    COPY_K(k_flash_prefill); COPY_K(k_flash_prefill_q8);
    COPY_K(k_gemm_q4_b); COPY_K(k_gemm_q4_1_b); COPY_K(k_gemm_q5_0_b); COPY_K(k_gemm_q5_1_b); COPY_K(k_gemm_q8_b); COPY_K(k_gemm_q2k_b); COPY_K(k_gemm_q3k_b); COPY_K(k_gemm_iq4xs_b);
    COPY_K(k_gemm_q4k_b); COPY_K(k_gemm_q5k_b); COPY_K(k_gemm_q6k_b); COPY_K(k_gemm_f16_b); COPY_K(k_gemm_f32_b);
    COPY_K(k_add_bias_b); COPY_K(k_mla_split_kv);
    COPY_K(k_mla_rope_pe); COPY_K(k_mla_store_latent); COPY_K(k_mla_unpack_wkv_b); COPY_K(k_mla_unpack_kb_vb);
    COPY_K(k_mla_absorb_q); COPY_K(k_flash_mla_latent); COPY_K(k_mla_proj_v); COPY_K(k_tp_accum_add);
    COPY_K(k_moe_slot_gu_q4); COPY_K(k_moe_slot_down_q4); COPY_K(k_moe_slot_down_q4_1); COPY_K(k_moe_slot_down_q5_0); COPY_K(k_moe_slot_down_q5_1); COPY_K(k_moe_slot_down_q8);
    COPY_K(k_moe_slot_gu_q4k); COPY_K(k_moe_slot_down_q4k); COPY_K(k_moe_slot_down_q6k);
    COPY_K(k_rope_kv_paged); COPY_K(k_rope_kv_q8_paged); COPY_K(k_flash_dec_paged); COPY_K(k_flash_dec_q8_paged);
    COPY_K(k_rope_kv_tree); COPY_K(k_rope_kv_q8_tree); COPY_K(k_flash_tree); COPY_K(k_flash_tree_q8);
    COPY_K(k_vit_ln); COPY_K(k_vit_gelu); COPY_K(k_vit_add); COPY_K(k_vit_attn);
    COPY_K(k_gemm_q4_sb); COPY_K(k_gemm_q4_1_sb); COPY_K(k_gemm_q4k_sb); COPY_K(k_gemm_q6k_sb);
    COPY_K(k_gemv_fp8); COPY_K(k_gemm_fp8_b); COPY_K(k_gemv_mxfp4); COPY_K(k_gemm_mxfp4_b);
    COPY_K(k_lora_a); COPY_K(k_lora_b); COPY_K(k_argmax_b); COPY_K(k_rope_kv_mrope); COPY_K(k_router);
    COPY_K(k_gemm_q4k_marlin); COPY_K(k_gemm_q6k_marlin); COPY_K(k_gemm_fp8_tc); COPY_K(k_gemm_mxfp4_tc);
    COPY_K(k_mla_rope_store_paged); COPY_K(k_flash_mla_latent_paged); COPY_K(k_gather_rows); COPY_K(k_scatter_add_rows);
    COPY_K(k_argmax);
    #undef COPY_K

    // Zero-Link Dynamic NCCL Auto-Discovery (`dlopen("libnccl.so.2")`) for Multi-GPU TP & EP!
    const char* nccl_env = getenv("TENON_NCCL");
    if (!(nccl_env && atoi(nccl_env) == 0)) {
        const char* nccl_paths[] = {
            "libnccl.so.2", "libnccl.so",
            "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
            "/usr/local/lib/python3.11/dist-packages/nvidia/nccl/lib/libnccl.so.2",
            "/usr/local/lib/python3.10/dist-packages/nvidia/nccl/lib/libnccl.so.2",
            "/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2",
            NULL
        };
        for (int i = 0; nccl_paths[i] && !g_cu.libnccl; i++) g_cu.libnccl = dlopen(nccl_paths[i], RTLD_LAZY);
        if (g_cu.libnccl) {
            g_cu.ncclCommInitAll = (fn_ncclCommInitAll)cm_sym(g_cu.libnccl, "ncclCommInitAll", NULL);
            g_cu.ncclCommDestroy = (fn_ncclCommDestroy)cm_sym(g_cu.libnccl, "ncclCommDestroy", NULL);
            g_cu.ncclAllReduce   = (fn_ncclAllReduce)cm_sym(g_cu.libnccl, "ncclAllReduce", NULL);
            g_cu.ncclBroadcast   = (fn_ncclBroadcast)cm_sym(g_cu.libnccl, "ncclBroadcast", NULL);
            g_cu.ncclGroupStart  = (fn_ncclGroupStart)cm_sym(g_cu.libnccl, "ncclGroupStart", NULL);
            g_cu.ncclGroupEnd    = (fn_ncclGroupEnd)cm_sym(g_cu.libnccl, "ncclGroupEnd", NULL);
            if (g_cu.ncclCommInitAll && g_cu.ncclAllReduce && g_cu.ncclBroadcast && g_cu.ncclGroupStart && g_cu.ncclGroupEnd) {
                g_cu.nccl_avail = 1;
            }
        }
    }

    // Initialize peer GPUs 1..n_gpus-1 + enable P2P when available (for true Multi-GPU TP=2..8!)
    for (int g = 1; g < g_cu.n_gpus; g++) {
        CmDevCtx* dc = &g_cu.gpus[g];
        if (g_cu.cuDeviceGet(&dc->dev, g) == CUDA_SUCCESS) {
            CUresult rc = g_cu.cuDevicePrimaryCtxRetain
                ? g_cu.cuDevicePrimaryCtxRetain(&dc->ctx, dc->dev)
                : g_cu.cuCtxCreate(&dc->ctx, 0, dc->dev);
            if (rc == CUDA_SUCCESS && g_cu.cuCtxSetCurrent(dc->ctx) == CUDA_SUCCESS) {
                g_cu.cuDeviceGetName(dc->dev_name, sizeof(dc->dev_name), dc->dev);
                g_cu.cuDeviceTotalMem(&dc->vram_bytes, dc->dev);
                cm_load_dev_kernels(dc, ptx);
            }
        }
    }
    if (g_cu.cuDeviceCanAccessPeer && g_cu.cuCtxEnablePeerAccess) {
        const char* p2p_env = getenv("TENON_TP_P2P");
        int force_p2p = p2p_env ? atoi(p2p_env) : -1;
        for (int a = 0; a < g_cu.n_gpus; a++) {
            if (!g_cu.gpus[a].ready) continue;
            g_cu.cuCtxSetCurrent(g_cu.gpus[a].ctx);
            for (int b = 0; b < g_cu.n_gpus; b++) {
                if (a == b || !g_cu.gpus[b].ready || force_p2p == 0) continue;
                int can = 0;
                if (g_cu.cuDeviceCanAccessPeer(&can, g_cu.gpus[a].dev, g_cu.gpus[b].dev) == CUDA_SUCCESS && can) {
                    if (g_cu.cuCtxEnablePeerAccess(g_cu.gpus[b].ctx, 0) == CUDA_SUCCESS) g_cu.p2p_matrix[a][b] = 1;
                }
            }
        }
        // Quick 1ms real P2P vs Pinned-Host latency probe: auto-demote slow cross-socket SYS PCIe P2P!
        if (force_p2p < 0 && g_cu.n_gpus >= 2 && g_cu.p2p_matrix[0][1] && g_cu.cuMemcpyPeerAsync) {
            CUdeviceptr p0 = 0, p1 = 0; void* ph = NULL; CUstream s0 = NULL;
            g_cu.cuCtxSetCurrent(g_cu.gpus[0].ctx);
            g_cu.cuMemAlloc(&p0, 32768); g_cu.cuStreamCreate(&s0, CU_STREAM_NON_BLOCKING);
            g_cu.cuMemHostAlloc(&ph, 32768, CU_MEMHOSTALLOC_PORTABLE);
            g_cu.cuCtxSetCurrent(g_cu.gpus[1].ctx);
            g_cu.cuMemAlloc(&p1, 32768);
            g_cu.cuCtxSetCurrent(g_cu.gpus[0].ctx);
            CUresult pr = CUDA_SUCCESS;
            for (int w = 0; w < 4 && pr == CUDA_SUCCESS; w++) pr = g_cu.cuMemcpyPeerAsync(p1, g_cu.gpus[1].ctx, p0, g_cu.gpus[0].ctx, 32768, s0);
            g_cu.cuStreamSynchronize(s0);
            double t0 = cm_now_ms();
            for (int it = 0; it < 16 && pr == CUDA_SUCCESS; it++) pr = g_cu.cuMemcpyPeerAsync(p1, g_cu.gpus[1].ctx, p0, g_cu.gpus[0].ctx, 32768, s0);
            g_cu.cuStreamSynchronize(s0);
            double p2p_ms = (cm_now_ms() - t0) / 16.0;
            double t1 = cm_now_ms();
            for (int it = 0; it < 16; it++) {
                g_cu.cuMemcpyDtoHAsync(ph, p0, 32768, s0);
                g_cu.cuStreamSynchronize(s0);
                g_cu.cuCtxSetCurrent(g_cu.gpus[1].ctx);
                g_cu.cuMemcpyHtoD(p1, ph, 32768);
                g_cu.cuCtxSetCurrent(g_cu.gpus[0].ctx);
            }
            double host_ms = (cm_now_ms() - t1) / 16.0;
            if (pr != CUDA_SUCCESS || p2p_ms > host_ms * 1.5 || p2p_ms > 0.25) {
                for (int a = 0; a < g_cu.n_gpus; a++) for (int b = 0; b < g_cu.n_gpus; b++) g_cu.p2p_matrix[a][b] = 0;
            }
            if (s0) g_cu.cuStreamDestroy(s0);
            if (p0) g_cu.cuMemFree(p0);
            if (ph) g_cu.cuMemFreeHost(ph);
            g_cu.cuCtxSetCurrent(g_cu.gpus[1].ctx);
            if (p1) g_cu.cuMemFree(p1);
        }
    }
    g_cu.cuCtxSetCurrent(g_cu.ctx);
    // Single-node multi-GPU NCCL safety: disable unconfigured IB/RoCE and broken cross-socket SYS PCIe P2P
    // when the 1ms P2P probe found no fast direct GPU-to-GPU link (respects explicit user env overrides).
    if (g_cu.n_gpus >= 2) {
        setenv("NCCL_IB_DISABLE", "1", 0);
        if (!g_cu.p2p_matrix[0][1]) setenv("NCCL_P2P_DISABLE", "1", 0);
    }

    const char* fdp = getenv("TENON_CUDA_FAST_DP4A");
    g_cu.fast_dp4a = (fdp && atoi(fdp) > 0) ? 1 : 0;
    g_cu.ready = 1;
    return 1;
}

int64_t tn_cumoe_device_name(void) { return (int64_t)(uintptr_t)g_cu.dev_name; }
int64_t tn_cumoe_vram_bytes(void)  { return (int64_t)g_cu.vram_bytes; }

// ============================================================================
// Multi-Shard GGUF v3 Parser + Model State
// ============================================================================
typedef struct {
    char name[128];
    uint32_t n_dims;
    uint64_t ne[4];
    uint32_t ggml_type;
    uint64_t rel_off;
    uint64_t file_off;
    uint64_t byte_size;
    int shard_idx;
} CmTensorInfo;

typedef struct {
    CUdeviceptr dptr;
    uint32_t ggml_type;
    uint64_t ne0, ne1, ne2;
    uint64_t file_off;
    uint64_t byte_size;
    int shard_idx;
} CmWeightRef;

typedef struct {
    int is_dense;
    int is_pinned_dense;
    CmWeightRef attn_norm, wq, wk, wv, wo;
    int is_mla;
    CmWeightRef wq_a, q_a_norm, wq_b, wkv_a, kv_a_norm, wkv_b, wk_b, wv_b;
    CUdeviceptr mla_wuk_f16, mla_wuv_f16;
    CmWeightRef bq, bk, bv; int has_qkv_bias;
    CmWeightRef q_norm, k_norm; int has_qk_norm, qk_norm_per_head;
    CmWeightRef ffn_norm;
    CmWeightRef ffn_gate, ffn_up, ffn_down;
    CmWeightRef router;
    CmWeightRef exp_bias; float* exp_bias_host; int has_exp_bias;
    CmWeightRef gate_exps, up_exps, down_exps;
    int has_shared_exp; CmWeightRef sh_gate, sh_up, sh_down;
    uint64_t gate_exp_bytes, up_exp_bytes, down_exp_bytes, slot_stride_bytes;
    CUdeviceptr cache_dptr;
    int exp_to_slot[CM_MAX_EXPERTS];
    int slot_to_exp[CM_MAX_EXPERTS];
    uint64_t slot_lru[CM_MAX_EXPERTS];
    uint32_t exp_freq[CM_MAX_EXPERTS];
    CUdeviceptr k_cache, v_cache, k_cache_base, v_cache_base;
    size_t kv_slot_bytes;
    // NUMA-local first-touch contiguous 4-row repacked CPU weights for unpinned Dense FFNs!
    uint8_t* repack_gu;    size_t repack_gu_bytes;
    uint8_t* repack_down;  size_t repack_down_bytes;
} CmLayer;

typedef struct { const char* key; int id; } CmBpeTokEntry;
typedef struct { uint64_t pair; int rank; } CmBpeMergeEntry;

typedef struct {
    CmWeightRef attn_norm, wq, wk, wv, wo, bq, bk, bv, q_norm, k_norm, ffn_norm, ffn_gate, ffn_up, ffn_down;
    int is_mla;
    CmWeightRef wq_a, q_a_norm, wq_b, wkv_a, kv_a_norm, wkv_b, wk_b, wv_b;
    CUdeviceptr mla_wuk_f16, mla_wuv_f16;
    CUdeviceptr k_cache, v_cache, k_cache_base, v_cache_base;
    size_t kv_slot_bytes;
} CmTpLayer;

typedef struct {
    int dev_idx;
    CUstream stream;
    CUevent ev_sent[CM_MAX_LAYERS * 2 + 4];
    CUevent ev_done[CM_MAX_LAYERS * 2 + 4];
    CmWeightRef rope_freqs;
    CmTpLayer layers[CM_MAX_LAYERS];
    CUdeviceptr d_x, d_xn, d_q, d_k, d_v, d_ctx, d_g, d_u, d_tp_part, d_tp_peer[2][CM_MAX_TP];
    CUdeviceptr d_mla_q, d_mla_kv, d_mla_qabs, d_mla_olat;
    CUdeviceptr d_pa_bt, d_cb_pos, d_cb_slot;
    CUdeviceptr d_moe_x, d_moe_y, d_moe_idx, d_moe_wt;
    int buf_cap;
} CmTpRank;

typedef struct {
    int dev_idx;
    int exp_start;
    int exp_count;
    CUstream stream;
    CUevent ev_done;
    CUdeviceptr cache_dptr[CM_MAX_LAYERS];
    CUdeviceptr d_xn, d_eidx, d_ewt, d_h, d_part;
    uint32_t* h_eidx;
    float* h_ewt;
    float* h_part;
} CmEpRank;

typedef struct {
    int active;
    int n_shards;
    int fds[CM_MAX_SHARDS];
    size_t shard_sizes[CM_MAX_SHARDS];
    uint8_t* mmap_ptrs[CM_MAX_SHARDS];
    uint64_t total_disk_bytes;
    int use_nocache;
    int enable_overlap;
    int resident_mode;
    int dense_model;

    char arch[64];
    int n_layers, nextn_layers, d_model, d_ff_dense, d_ff_exp;
    int n_heads, n_kv, d_head, rope_dim, rope_neox;
    int n_experts, top_k, vocab_size, bos_id, eos_id;
    int gating_func, expert_weights_norm;
    int n_expert_groups, n_group_used;
    int kv_lora_rank, q_lora_rank, v_head_dim;
    int max_ctx, kv_quant_q8, fast_dp4a;
    int mla_latent_kv, n_kv_slots, cur_kv_slot, slot_ctx, has_mtp;
    float expert_weights_scale, rope_theta, rms_eps;
    float yarn_mscale_k, yarn_factor;
    int yarn_orig_ctx;
    int cache_slots_per_layer;
    char** vocab_tokens;
    char** bpe_merges;
    int n_bpe_merges;
    CmBpeTokEntry* bpe_tok_ht;
    uint32_t bpe_tok_cap;
    CmBpeMergeEntry* bpe_merge_ht;
    uint32_t bpe_merge_cap;
    int* bpe_special_ids;
    int n_bpe_special;
    int bpe_use_sp, bpe_is_gpt2;

    CmWeightRef tok_embd, out_norm, lm_head, rope_freqs;
    CmWeightRef mtp_enorm, mtp_hnorm, mtp_eh_proj;
    int has_rope_freqs;
    CmLayer layers[CM_MAX_LAYERS];

    // True Multi-GPU Tensor Parallelism (TP=2..8, Megatron Row/Col + Dynamic NCCL / P2P AllReduce)
    int tp_size;
    int tp_p2p_ok;
    int tp_nccl_ok;
    ncclComm_t tp_nccl[CM_MAX_TP];
    CmTpRank* tp;
    float* h_tp_buf[CM_MAX_TP];
    float* h_tp_sum;
    int tp_host_cap;

    // True Multi-GPU MoE Expert Parallelism (EP=2..8, Sharded Experts + Dynamic NCCL AllReduce)
    int ep_size;
    int ep_exp_per_rank;
    int ep_nccl_ok;
    ncclComm_t ep_nccl[CM_MAX_TP];
    CmEpRank* ep;
    int hybrid_2d_mode; // 1 = True 2D Hybrid TP x EP on MoE (Attention sharded across TP + Experts sharded across EP!)

    // Tenon v7: 2D/3D MRoPE spatial position + Zero-Copy Runtime Multi-Adapter GGUF LoRA + Batched Argmax
    int has_mrope;
    int mrope_sections[3];
    int mrope_pos[3];
    struct {
        int active;
        int rank;
        float alpha;
        float user_scale;
        int n_tensors;
        int64_t vram_bytes;
        CUdeviceptr a[CM_MAX_LAYERS][7]; // 0=q,1=k,2=v,3=o,4=gate,5=up,6=down; [rank, in_dim] F32
        CUdeviceptr b[CM_MAX_LAYERS][7]; // [out_dim, rank] F32
        int r_dim[CM_MAX_LAYERS][7];
    } loras[CM_MAX_LORAS];
    int active_lora;
    int cur_exec_layer;
    CUdeviceptr d_lora_tmp;
    CUdeviceptr d_argmax_batch;
    uint32_t* h_argmax_batch;

    // Tenon v6: Native MTP Full-Block Predictor + PagedAttention 16-Token Block-Table + Tree-Verify
    CmLayer* mtp_layer;
    int has_mtp_layer;
    CUdeviceptr d_last_hidden;
    CUdeviceptr d_mtp_cat;
    int pa_num_blocks, pa_free_blocks;
    int32_t* pa_refcnt;
    int32_t pa_block_table[CM_PA_MAX_SEQS][CM_PA_MAX_LOGICAL_BLKS];
    int pa_seq_len[CM_PA_MAX_SEQS];
    int64_t pa_cow_forks, pa_cow_copies;
    CUdeviceptr d_pa_bt, d_cb_pos, d_cb_slot, d_batch_logits;
    int batch_logits_cap;
    CUdeviceptr d_tree_depth, d_tree_mask;

    // CUDA streams, events & Resident CUDA Graph (eliminates 880 launches/tok CPU bubble!)
    CUstream compute_stream;
    CUstream dma_stream;
    CUevent ev_dma_done;
    CUgraphExec res_graph_exec;
    int res_graph_ready;
    CUdeviceptr d_step_args;
    uint32_t* h_step_args;

    // Pinned host staging ring for 25 GB/s PCIe 4.0 H2D DMA!
    uint8_t* stage_host;
    size_t stage_slot_bytes;
    int stage_n_slots;
    float* h_rlogits;
    uint32_t* h_argmax;
    uint32_t* h_eidx;
    float* h_ewt;
    uint8_t* h_emb_row;
    float* h_xn;
    float* h_cpu_accum;
    float* cpu_h;
    CUdeviceptr d_cpu_accum;

    // Zero-stall background VRAM promotion state (dedicated promoter thread + dma_stream!)
    CUevent ev_promo_done;
    uint8_t* promo_stage;
    pthread_t promo_thread;
    pthread_mutex_t promo_mu;
    pthread_cond_t promo_cv;
    int promo_thread_init;
    int promo_stop;
    volatile int promo_state; // 0 = idle, 1 = staging+submitting, 2 = dma_submitted
    int promo_layer;
    int promo_slot;
    int promo_eid;
    int decode_phase;

    // Spin-then-wait 32-thread AVX2+FMA CPU compute pool for in-place DDR4 miss/FFN execution
    pthread_t cpu_threads[CM_MAX_CPU_THR];
    pthread_mutex_t cpu_mu;
    pthread_cond_t cpu_cv_go;
    int cpu_n_thr;
    int cpu_pool_init;
    volatile int cpu_stop;
    volatile int cpu_sleeping;
    volatile uint64_t cpu_gen;
    volatile int cpu_mode; // 1 = MoE missed experts, 2 = Dense unpinned FFN
    volatile int cpu_phase1_done;
    volatile int cpu_phase2_done;
    int cpu_layer;
    int cpu_n_misses;
    int cpu_miss_eids[CM_MAX_TOPK];
    float cpu_miss_w[CM_MAX_TOPK];
    // Pre-quantized Q8_0 (MoE) + Q16_0 (exact BIT-EXACT Dense gate/up/down_proj) activation buffers!
    int8_t*  cpu_xq8;
    int16_t* cpu_xq16;
    float*   cpu_xd;
    float*   cpu_xsum;
    int8_t*  cpu_hq8;
    int16_t* cpu_hq16;
    float*   cpu_hd;
    float*   cpu_hsum;
    int      cpu_repack_layer;

    // Dense streaming 4-slot VRAM ring + 4-slot pinned host ring
    int dense_pin_all_attn;
    CUdeviceptr dense_ring_dptr[4];
    uint8_t* dense_ring_host[4];
    CUevent dense_slot_ready[4];
    CUevent dense_slot_consumed[4];
    uint64_t dense_ring_stride;
    uint64_t dense_off_wq, dense_off_wk, dense_off_wv, dense_off_wo;
    uint64_t dense_off_gate, dense_off_up, dense_off_down;

    // GPU activations
    CUdeviceptr d_emb_row, d_x, d_xn, d_q, d_k, d_v, d_ctx;
    CUdeviceptr d_rlogits, d_eidx, d_ewt, d_h, d_sh_g, d_sh_u, d_logits, d_argmax;
    // Lazy batched prefill & MLA GPU buffers + host logits for sampling
    int prefill_cap;
    CUdeviceptr d_px, d_pxn, d_pq, d_pk, d_pv, d_pctx, d_pg, d_pu, d_mla_q, d_mla_kv, d_mla_qabs, d_mla_olat;
    CUdeviceptr d_prlogits, d_moe_x, d_moe_y, d_moe_idx, d_moe_wt;
    float* h_prlogits;
    uint32_t* h_moe_idx;
    float* h_moe_wt;
    int64_t moe_grouped_saved_reads;
    size_t mla_buf_cap;
    float* h_logits;
    int cpu_batch_b;
    size_t cpu_batch_off;
    float* h_pxn;
    float* h_pcpu_accum;

    // Persistent host staging worker pool (zero pthread_create/join overhead per layer!)
    pthread_t io_threads[CM_IO_WORKERS];
    pthread_mutex_t io_mu;
    pthread_cond_t io_cv_go;
    pthread_cond_t io_cv_done;
    int io_pool_init;
    int io_stop;
    uint64_t io_gen;
    int io_mode; // 1 = MoE experts, 2 = Dense slice copy
    int io_n_tasks;
    int io_done_cnt;
    int io_layer;
    int io_eids[CM_MAX_TOPK];
    uint8_t* io_dsts[CM_MAX_TOPK];
    volatile int io_task_ready[CM_MAX_TOPK];
    // Dense slice copy args
    int io_dense_shard;
    uint8_t* io_dense_dst;
    uint64_t io_dense_off;
    size_t io_dense_len;

    // Counters
    uint64_t lru_clock;
    int64_t core_vram_bytes, cache_vram_bytes, total_streamable_bytes;
    int64_t cache_hits, cache_misses, prefetch_hits;
    int64_t io_bytes_read, pcie_h2d_bytes;
    double last_total_ms, last_io_ms, last_gpu_ms;
} CmModel;

static CmModel g_cm[CM_MAX_MODELS];

static uint64_t cm_align64(uint64_t n) { return (n + 63ull) & ~63ull; }

static void cm_pread_exact(int fd, void* dst, size_t n, uint64_t off) {
    uint8_t* p = (uint8_t*)dst;
    size_t got = 0;
    while (got < n) {
        ssize_t r = pread(fd, p + got, n - got, (off_t)(off + got));
        if (r <= 0) { if (r < 0 && errno == EINTR) continue; break; }
        got += (size_t)r;
    }
}

static void cm_read_slice(CmModel* m, int shard, void* dst, size_t n, uint64_t off) {
    if (!m->use_nocache && m->mmap_ptrs[shard] != NULL) {
        memcpy(dst, m->mmap_ptrs[shard] + off, n);
    } else {
        cm_pread_exact(m->fds[shard], dst, n, off);
#if defined(__linux__)
        if (m->use_nocache) {
            posix_fadvise(m->fds[shard], (off_t)off, (off_t)n, POSIX_FADV_DONTNEED);
        }
#endif
    }
    __sync_fetch_and_add(&m->io_bytes_read, (int64_t)n);
}

typedef struct { FILE* f; } CmGr;
static uint32_t gr_u32(CmGr* r) { uint32_t v = 0; if (fread(&v, 4, 1, r->f) != 1) {} return v; }
static uint64_t gr_u64(CmGr* r) { uint64_t v = 0; if (fread(&v, 8, 1, r->f) != 1) {} return v; }
static float    gr_f32(CmGr* r) { float    v = 0; if (fread(&v, 4, 1, r->f) != 1) {} return v; }

static char* gr_str(CmGr* r) {
    uint64_t len = gr_u64(r);
    char* s = (char*)malloc((size_t)len + 1);
    if (len > 0 && fread(s, 1, (size_t)len, r->f) != len) {}
    s[len] = 0;
    return s;
}

static void gr_skip(CmGr* r, uint32_t t) {
    switch (t) {
        case CM_GGUF_U8: case CM_GGUF_I8: case CM_GGUF_BOOL: fseeko(r->f, 1, SEEK_CUR); break;
        case CM_GGUF_U16: case CM_GGUF_I16: fseeko(r->f, 2, SEEK_CUR); break;
        case CM_GGUF_U32: case CM_GGUF_I32: case CM_GGUF_F32: fseeko(r->f, 4, SEEK_CUR); break;
        case CM_GGUF_U64: case CM_GGUF_I64: case CM_GGUF_F64: fseeko(r->f, 8, SEEK_CUR); break;
        case CM_GGUF_STR: { uint64_t l = gr_u64(r); fseeko(r->f, (off_t)l, SEEK_CUR); break; }
        case CM_GGUF_ARR: { uint32_t et = gr_u32(r); uint64_t n = gr_u64(r); for (uint64_t i = 0; i < n; i++) gr_skip(r, et); break; }
        default: break;
    }
}

static uint64_t gr_int(CmGr* r, uint32_t t) {
    switch (t) {
        case CM_GGUF_U8:  { uint8_t v = 0; if (fread(&v, 1, 1, r->f) != 1) {} return v; }
        case CM_GGUF_I8:  { int8_t v = 0; if (fread(&v, 1, 1, r->f) != 1) {} return (uint64_t)(int64_t)v; }
        case CM_GGUF_U16: { uint16_t v = 0; if (fread(&v, 2, 1, r->f) != 1) {} return v; }
        case CM_GGUF_I16: { int16_t v = 0; if (fread(&v, 2, 1, r->f) != 1) {} return (uint64_t)(int64_t)v; }
        case CM_GGUF_U32: return gr_u32(r);
        case CM_GGUF_I32: { int32_t v = 0; if (fread(&v, 4, 1, r->f) != 1) {} return (uint64_t)(int64_t)v; }
        case CM_GGUF_U64: return gr_u64(r);
        case CM_GGUF_I64: { int64_t v = 0; if (fread(&v, 8, 1, r->f) != 1) {} return (uint64_t)v; }
        case CM_GGUF_BOOL:{ uint8_t v = 0; if (fread(&v, 1, 1, r->f) != 1) {} return v; }
        default: gr_skip(r, t); return 0;
    }
}

static int cm_parse_shard(const char* path, int shard_idx, int read_meta, CmModel* m,
                          CmTensorInfo** tlist_io, uint64_t* tcount_io, int* split_count_out) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    CmGr gr = { fp };
    char magic[4];
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "GGUF", 4) != 0) { fclose(fp); return 0; }
    (void)gr_u32(&gr);
    uint64_t tensor_count = gr_u64(&gr), kv_count = gr_u64(&gr), alignment = 32;
    if (split_count_out) *split_count_out = 1;

    for (uint64_t i = 0; i < kv_count; i++) {
        char* k = gr_str(&gr);
        uint32_t vt = gr_u32(&gr);
        if (strcmp(k, "general.alignment") == 0) alignment = gr_int(&gr, vt);
        else if (split_count_out && strcmp(k, "split.count") == 0) *split_count_out = (int)gr_int(&gr, vt);
        else if (read_meta) {
            if (strcmp(k, "general.architecture") == 0 && vt == CM_GGUF_STR) {
                char* av = gr_str(&gr); strncpy(m->arch, av, sizeof(m->arch) - 1); free(av);
            } else if (strstr(k, ".block_count")) m->n_layers = (int)gr_int(&gr, vt);
            else if (strstr(k, ".nextn_predict_layers")) m->nextn_layers = (int)gr_int(&gr, vt);
            else if (strstr(k, ".embedding_length")) m->d_model = (int)gr_int(&gr, vt);
            else if (strstr(k, ".expert_feed_forward_length")) m->d_ff_exp = (int)gr_int(&gr, vt);
            else if (strstr(k, ".feed_forward_length") && !strstr(k, "shared")) m->d_ff_dense = (int)gr_int(&gr, vt);
            else if (strstr(k, ".attention.head_count_kv")) m->n_kv = (int)gr_int(&gr, vt);
            else if (strstr(k, ".attention.head_count")) m->n_heads = (int)gr_int(&gr, vt);
            else if (strstr(k, ".attention.key_length")) m->d_head = (int)gr_int(&gr, vt);
            else if (strstr(k, ".rope.dimension_count")) m->rope_dim = (int)gr_int(&gr, vt);
            else if (strstr(k, ".rope.dimension_sections") && vt == CM_GGUF_ARR) {
                uint32_t et = gr_u32(&gr); uint64_t n = gr_u64(&gr); int sum_s = 0;
                for (uint64_t si = 0; si < n; si++) { int sv = (int)gr_int(&gr, et); if (si < 3) { m->mrope_sections[si] = sv; sum_s += sv; } }
                if (sum_s > 0) m->has_mrope = 1;
            }
            else if (strstr(k, ".expert_count")) m->n_experts = (int)gr_int(&gr, vt);
            else if (strstr(k, ".expert_used_count")) m->top_k = (int)gr_int(&gr, vt);
            else if (strstr(k, ".expert_gating_func")) m->gating_func = (int)gr_int(&gr, vt);
            else if (strstr(k, ".expert_weights_norm")) m->expert_weights_norm = (int)gr_int(&gr, vt);
            else if (strstr(k, ".expert_weights_scale") && vt == CM_GGUF_F32) m->expert_weights_scale = gr_f32(&gr);
            else if (strstr(k, ".expert_group_count")) m->n_expert_groups = (int)gr_int(&gr, vt);
            else if (strstr(k, ".expert_group_used_count")) m->n_group_used = (int)gr_int(&gr, vt);
            else if (strstr(k, ".attention.kv_lora_rank")) m->kv_lora_rank = (int)gr_int(&gr, vt);
            else if (strstr(k, ".attention.q_lora_rank")) m->q_lora_rank = (int)gr_int(&gr, vt);
            else if (strstr(k, ".attention.value_length")) m->v_head_dim = (int)gr_int(&gr, vt);
            else if (strstr(k, ".rope.freq_base") && vt == CM_GGUF_F32) m->rope_theta = gr_f32(&gr);
            else if (strstr(k, ".rope.scaling.yarn_log_multiplier") && vt == CM_GGUF_F32) m->yarn_mscale_k = gr_f32(&gr);
            else if (strstr(k, ".rope.scaling.factor") && vt == CM_GGUF_F32) m->yarn_factor = gr_f32(&gr);
            else if (strstr(k, ".rope.scaling.original_context_length")) m->yarn_orig_ctx = (int)gr_int(&gr, vt);
            else if (strstr(k, ".layer_norm_rms_epsilon") && vt == CM_GGUF_F32) m->rms_eps = gr_f32(&gr);
            else if (strcmp(k, "tokenizer.ggml.bos_token_id") == 0) m->bos_id = (int)gr_int(&gr, vt);
            else if (strcmp(k, "tokenizer.ggml.eos_token_id") == 0) m->eos_id = (int)gr_int(&gr, vt);
            else if (strcmp(k, "tokenizer.ggml.tokens") == 0 && vt == CM_GGUF_ARR) {
                uint32_t et = gr_u32(&gr); uint64_t n = gr_u64(&gr);
                if (et == CM_GGUF_STR) {
                    m->vocab_size = (int)n;
                    m->vocab_tokens = (char**)calloc((size_t)n, sizeof(char*));
                    for (uint64_t ti = 0; ti < n; ti++) m->vocab_tokens[ti] = gr_str(&gr);
                } else { for (uint64_t ti = 0; ti < n; ti++) gr_skip(&gr, et); }
            } else if (strcmp(k, "tokenizer.ggml.merges") == 0 && vt == CM_GGUF_ARR) {
                uint32_t et = gr_u32(&gr); uint64_t n = gr_u64(&gr);
                if (et == CM_GGUF_STR) {
                    m->n_bpe_merges = (int)n;
                    m->bpe_merges = (char**)calloc((size_t)n, sizeof(char*));
                    for (uint64_t ti = 0; ti < n; ti++) m->bpe_merges[ti] = gr_str(&gr);
                } else { for (uint64_t ti = 0; ti < n; ti++) gr_skip(&gr, et); }
            } else gr_skip(&gr, vt);
        } else gr_skip(&gr, vt);
        free(k);
    }

    uint64_t old_n = *tcount_io;
    CmTensorInfo* grown = (CmTensorInfo*)realloc(*tlist_io, (size_t)(old_n + tensor_count) * sizeof(CmTensorInfo));
    if (!grown) { fclose(fp); return 0; }
    *tlist_io = grown;
    CmTensorInfo* dst = grown + old_n;
    memset(dst, 0, (size_t)tensor_count * sizeof(CmTensorInfo));
    for (uint64_t i = 0; i < tensor_count; i++) {
        char* tn = gr_str(&gr); strncpy(dst[i].name, tn, sizeof(dst[i].name) - 1); free(tn);
        dst[i].n_dims = gr_u32(&gr);
        uint64_t numel = 1;
        for (uint32_t d = 0; d < dst[i].n_dims; d++) { uint64_t dm = gr_u64(&gr); if (d < 4) dst[i].ne[d] = dm; numel *= dm; }
        dst[i].ggml_type = gr_u32(&gr);
        dst[i].rel_off = gr_u64(&gr);
        dst[i].byte_size = cm_ggml_type_bytes(dst[i].ggml_type, numel);
        dst[i].shard_idx = shard_idx;
    }
    uint64_t hdr_end = (uint64_t)ftello(fp);
    uint64_t data_off = (hdr_end + alignment - 1) & ~(alignment - 1);
    fclose(fp);
    for (uint64_t i = 0; i < tensor_count; i++) dst[i].file_off = data_off + dst[i].rel_off;
    *tcount_io = old_n + tensor_count;
    return 1;
}

static void cm_record_meta(CmWeightRef* w, const CmTensorInfo* ti) {
    w->dptr = 0; w->ggml_type = ti->ggml_type;
    w->ne0 = ti->ne[0]; w->ne1 = ti->ne[1]; w->ne2 = ti->ne[2];
    w->file_off = ti->file_off; w->byte_size = ti->byte_size; w->shard_idx = ti->shard_idx;
}

static void cm_load_pinned_vram(CmModel* m, CmWeightRef* w, const CmTensorInfo* ti) {
    cm_record_meta(w, ti);
    if (ti->byte_size == 0) return;
    if (g_cu.cuMemAlloc(&w->dptr, (size_t)ti->byte_size) != CUDA_SUCCESS) return;
    void* tmp = malloc((size_t)ti->byte_size);
    if (tmp) {
        cm_pread_exact(m->fds[ti->shard_idx], tmp, (size_t)ti->byte_size, ti->file_off);
        g_cu.cuMemcpyHtoD(w->dptr, tmp, (size_t)ti->byte_size);
        free(tmp);
    }
    m->core_vram_bytes += (int64_t)ti->byte_size;
}

// ============================================================================
// CUDA Kernel Launch Helpers
// ============================================================================
static void launch_quant_q8_1(CUstream st, CUdeviceptr x, uint32_t in_dim) {
    uint32_t nb = in_dim >> 5;
    void* args[] = { &x, &g_cu.d_xq8, &in_dim };
    g_cu.cuLaunchKernel(g_cu.k_quant_q8_1, (nb + 3u) / 4u, 1, 1, 128, 1, 1, 0, st, args, NULL);
}

static void launch_gemv_raw(CUstream st, CUdeviceptr w, uint32_t gtype, CUdeviceptr x, CUdeviceptr y,
                            uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {
    CUfunction fn = g_cu.k_gemv_q4;
    uint32_t rptg = 4;
    CUdeviceptr xin = x;
    switch (gtype) {
        case CM_GGML_Q4_0:
            if (g_cu.fast_dp4a) { launch_quant_q8_1(st, x, in_dim); xin = g_cu.d_xq8; fn = g_cu.k_gemv_q4_dp4a; rptg = 16; }
            else { fn = g_cu.k_gemv_q4; rptg = 8; }
            break;
        case CM_GGML_Q4_1:
            if (g_cu.fast_dp4a) { launch_quant_q8_1(st, x, in_dim); xin = g_cu.d_xq8; fn = g_cu.k_gemv_q4_1_dp4a; rptg = 8; }
            else { fn = g_cu.k_gemv_q4_1; rptg = 8; }
            break;
        case CM_GGML_Q5_0:   fn = g_cu.k_gemv_q5_0;  rptg = 8; break;
        case CM_GGML_Q5_1:   fn = g_cu.k_gemv_q5_1;  rptg = 8; break;
        case CM_GGML_Q8_0:   fn = g_cu.k_gemv_q8;    rptg = 8; break;
        case CM_GGML_Q2_K:   fn = g_cu.k_gemv_q2k;   rptg = 8; break;
        case CM_GGML_Q3_K:   fn = g_cu.k_gemv_q3k;   rptg = 8; break;
        case CM_GGML_IQ4_XS: fn = g_cu.k_gemv_iq4xs; rptg = 8; break;
        case CM_GGML_Q4_K:   fn = g_cu.k_gemv_q4k;   rptg = 8; break;
        case CM_GGML_Q5_K:   fn = g_cu.k_gemv_q5k;   rptg = 8; break;
        case CM_GGML_Q6_K:   fn = g_cu.k_gemv_q6k;   rptg = 8; break;
        case CM_GGML_FP8_E4M3: fn = g_cu.k_gemv_fp8;   rptg = 4; break;
        case CM_GGML_MXFP4:    fn = g_cu.k_gemv_mxfp4; rptg = 4; break;
        case CM_GGML_F16:    fn = g_cu.k_gemv_f16;   rptg = 4; break;
        default:             fn = g_cu.k_gemv_f32;   rptg = 4; break;
    }
    void* args[] = { &w, &xin, &y, &out_dim, &in_dim, &add_res };
    g_cu.cuLaunchKernel(fn, (out_dim + rptg - 1) / rptg, 1, 1, 128, 1, 1, 0, st, args, NULL);
}

static void launch_gemv(CUstream st, const CmWeightRef* w, CUdeviceptr x, CUdeviceptr y,
                        uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {
    launch_gemv_raw(st, w->dptr, w->ggml_type, x, y, out_dim, in_dim, add_res);
}

static void launch_rmsnorm(CUstream st, CUdeviceptr x, const CmWeightRef* w, CUdeviceptr out,
                           uint32_t dim, uint32_t n_groups, float eps, uint32_t shared_w) {
    CUdeviceptr wp = w->dptr;
    void* args[] = { &x, &wp, &out, &dim, &n_groups, &eps, &shared_w };
    g_cu.cuLaunchKernel(g_cu.k_rmsnorm, n_groups, 1, 1, 256, 1, 1, 0, st, args, NULL);
}

static void launch_add_bias(CUstream st, CUdeviceptr x, const CmWeightRef* b, uint32_t n) {
    if (!b->dptr) return;
    CUdeviceptr bp = b->dptr;
    void* args[] = { &x, &bp, &n };
    g_cu.cuLaunchKernel(g_cu.k_add_bias, (n + 255) / 256, 1, 1, 256, 1, 1, 0, st, args, NULL);
}

static void launch_lora_delta(CmModel* m, CUstream st, int L, int proj, CUdeviceptr x, CUdeviceptr y,
                              uint32_t out_dim, uint32_t in_dim, uint32_t B) {
    int lid = m->active_lora;
    if (lid <= 0 || lid >= CM_MAX_LORAS || !m->loras[lid].active || L < 0 || L >= CM_MAX_LAYERS) return;
    CUdeviceptr da = m->loras[lid].a[L][proj], db = m->loras[lid].b[L][proj];
    int rank = m->loras[lid].r_dim[L][proj];
    if (!da || !db || rank <= 0) return;
    if (!m->d_lora_tmp) g_cu.cuMemAlloc(&m->d_lora_tmp, (size_t)CM_PREFILL_MAX_B * 256u * 4u);
    float alpha = (m->loras[lid].alpha > 0.0f) ? m->loras[lid].alpha : (float)rank;
    float scale = m->loras[lid].user_scale * (alpha / (float)rank);
    uint32_t rk = (uint32_t)rank;
    void* aa[] = { &da, &x, &m->d_lora_tmp, &in_dim, &rk, &B };
    g_cu.cuLaunchKernel(g_cu.k_lora_a, rk, B, 1, 32, 1, 1, 0, st, aa, NULL);
    void* ba[] = { &db, &m->d_lora_tmp, &y, &out_dim, &rk, &B, &scale };
    g_cu.cuLaunchKernel(g_cu.k_lora_b, (out_dim + 63u) / 64u, B, 1, 64, 1, 1, 0, st, ba, NULL);
}

static void launch_dense_ffn_raw(CmModel* m, CUstream st,
                                 CUdeviceptr wg, uint32_t gt, CUdeviceptr wu, uint32_t ut,
                                 CUdeviceptr wd, uint32_t dt, uint32_t d_ff) {
    uint32_t dm = (uint32_t)m->d_model;
    int L = m->cur_exec_layer;
    int has_gu_lora = (m->active_lora > 0 && L >= 0 && L < CM_MAX_LAYERS &&
                       (m->loras[m->active_lora].a[L][4] || m->loras[m->active_lora].a[L][5]));
    if (gt == CM_GGML_Q4_0 && ut == CM_GGML_Q4_0 && !has_gu_lora) {
        if (g_cu.fast_dp4a) {
            launch_quant_q8_1(st, m->d_xn, dm);
            void* args[] = { &wg, &wu, &g_cu.d_xq8, &m->d_sh_g, &d_ff, &dm };
            g_cu.cuLaunchKernel(g_cu.k_dense_gu_q4_dp4a, (d_ff + 7) / 8, 1, 1, 128, 1, 1, 0, st, args, NULL);
        } else {
            void* args[] = { &wg, &wu, &m->d_xn, &m->d_sh_g, &d_ff, &dm };
            g_cu.cuLaunchKernel(g_cu.k_dense_gu_q4, (d_ff + 3) / 4, 1, 1, 128, 1, 1, 0, st, args, NULL);
        }
    } else {
        launch_gemv_raw(st, wg, gt, m->d_xn, m->d_sh_g, d_ff, dm, 0);
        if (has_gu_lora) launch_lora_delta(m, st, L, 4, m->d_xn, m->d_sh_g, d_ff, dm, 1u);
        launch_gemv_raw(st, wu, ut, m->d_xn, m->d_sh_u, d_ff, dm, 0);
        if (has_gu_lora) launch_lora_delta(m, st, L, 5, m->d_xn, m->d_sh_u, d_ff, dm, 1u);
        void* args[] = { &m->d_sh_g, &m->d_sh_u, &d_ff };
        g_cu.cuLaunchKernel(g_cu.k_swiglu_mul, (d_ff + 255) / 256, 1, 1, 256, 1, 1, 0, st, args, NULL);
    }
    launch_gemv_raw(st, wd, dt, m->d_sh_g, m->d_x, dm, d_ff, 1);
    if (m->active_lora > 0 && L >= 0) launch_lora_delta(m, st, L, 6, m->d_sh_g, m->d_x, dm, d_ff, 1u);
}

static void cm_ensure_mla_bufs(CmModel* m, uint32_t B) {
    uint32_t kv_r = (uint32_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
    uint32_t ldim = kv_r + (uint32_t)m->rope_dim;
    size_t need = (size_t)B;
    if (m->mla_buf_cap >= need && m->d_mla_q && m->d_mla_kv && m->d_mla_qabs && m->d_mla_olat) return;
    if (need < 32u) need = 32u;
    if (m->d_mla_q)    g_cu.cuMemFree(m->d_mla_q);
    if (m->d_mla_kv)   g_cu.cuMemFree(m->d_mla_kv);
    if (m->d_mla_qabs) g_cu.cuMemFree(m->d_mla_qabs);
    if (m->d_mla_olat) g_cu.cuMemFree(m->d_mla_olat);
    size_t qr = (size_t)(m->q_lora_rank > 0 ? m->q_lora_rank : m->d_model);
    g_cu.cuMemAlloc(&m->d_mla_q,    need * qr * 4u);
    g_cu.cuMemAlloc(&m->d_mla_kv,   need * (size_t)ldim * 4u);
    g_cu.cuMemAlloc(&m->d_mla_qabs, need * (size_t)m->n_heads * (size_t)ldim * 4u);
    g_cu.cuMemAlloc(&m->d_mla_olat, need * (size_t)m->n_heads * (size_t)kv_r * 4u);
    m->mla_buf_cap = need;
}

static void launch_layer_attn_ptr(CmModel* m, CUstream st, CmLayer* ly, uint32_t pos, CUdeviceptr ring_base, int indirect) {
    int L = (ly >= &m->layers[0] && ly < &m->layers[CM_MAX_LAYERS]) ? (int)(ly - &m->layers[0]) : -1;
    m->cur_exec_layer = L;
    int has_qkv_lora = (m->active_lora > 0 && L >= 0 && L < CM_MAX_LAYERS &&
                        (m->loras[m->active_lora].a[L][0] || m->loras[m->active_lora].a[L][1] || m->loras[m->active_lora].a[L][2]));
    uint32_t dm = (uint32_t)m->d_model, qdim = (uint32_t)(m->n_heads * m->d_head), kvdim = (uint32_t)(m->n_kv * m->d_head);
    uint32_t wo_in = (ly->wo.ne0 > 0) ? (uint32_t)ly->wo.ne0 : qdim;
    launch_rmsnorm(st, m->d_x, &ly->attn_norm, m->d_xn, dm, 1, m->rms_eps, 0);
    CUdeviceptr pq = ring_base ? (ring_base + m->dense_off_wq) : ly->wq.dptr;
    CUdeviceptr pk = ring_base ? (ring_base + m->dense_off_wk) : ly->wk.dptr;
    CUdeviceptr pv = ring_base ? (ring_base + m->dense_off_wv) : ly->wv.dptr;
    int mla_done = 0;
    if (ly->is_mla && ly->wkv_a.dptr && (ly->wkv_b.dptr || (ly->wk_b.dptr && ly->wv_b.dptr))) {
        uint32_t kv_r = (uint32_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
        uint32_t nope_d = (uint32_t)(m->d_head - m->rope_dim), v_d = (uint32_t)m->v_head_dim;
        uint32_t nh_u = (uint32_t)m->n_heads, rd_u = (uint32_t)m->rope_dim, dh_u = (uint32_t)m->d_head;
        uint32_t rn = (uint32_t)m->rope_neox, hf = (uint32_t)m->has_rope_freqs, one = 1u;
        float theta = m->rope_theta;
        CUdeviceptr rf = hf ? m->rope_freqs.dptr : m->d_argmax;
        cm_ensure_mla_bufs(m, 1u);
        if (ly->wq_a.dptr && ly->wq_b.dptr && m->q_lora_rank > 0) {
            launch_gemv(st, &ly->wq_a, m->d_xn, m->d_mla_q, (uint32_t)m->q_lora_rank, dm, 0);
            if (ly->q_a_norm.dptr) launch_rmsnorm(st, m->d_mla_q, &ly->q_a_norm, m->d_mla_q, (uint32_t)m->q_lora_rank, 1, m->rms_eps, 0);
            launch_gemv(st, &ly->wq_b, m->d_mla_q, m->d_q, qdim, (uint32_t)m->q_lora_rank, 0);
        } else {
            launch_gemv(st, &ly->wq, m->d_xn, m->d_q, qdim, dm, 0);
        }
        launch_gemv(st, &ly->wkv_a, m->d_xn, m->d_mla_kv, kv_r + rd_u, dm, 0);
        if (ly->kv_a_norm.dptr) launch_rmsnorm(st, m->d_mla_kv, &ly->kv_a_norm, m->d_mla_kv, kv_r, 1, m->rms_eps, 0);
        CUdeviceptr kpe_ptr = m->d_mla_kv + (uint64_t)kv_r * 4ull;
        // Official DeepSeek MLA RoPE strictly on PE slice `nope_d..d_head-1` of Q and `k_pe[0..rope_dim-1]`!
        uint32_t half_r = rd_u >> 1;
        void* rpe_args[] = { &m->d_q, &kpe_ptr, &rf, &nh_u, &dh_u, &nope_d, &rd_u, &pos, &theta, &rn, &hf, &one };
        g_cu.cuLaunchKernel(g_cu.k_mla_rope_pe, (half_r + 31u) / 32u, nh_u + 1u, 1, 32, 1, 1, 0, st, rpe_args, NULL);

        float mla_scale = 1.0f / sqrtf((float)dh_u);
        if (m->yarn_mscale_k > 0.0f && m->yarn_factor > 1.0f) {
            float ms = 1.0f + m->yarn_mscale_k * logf(m->yarn_factor);
            mla_scale *= ms * ms;
        }
        uint32_t out_st = wo_in / nh_u;
        if (m->mla_latent_kv && ly->mla_wuk_f16 && ly->mla_wuv_f16) {
            // TRUE MATRIX-ABSORBED COMPRESSED LATENT MLA (ZERO V-cache, -8x..-85x KV VRAM!)
            uint32_t ldim = kv_r + rd_u;
            void* sargs[] = { &m->d_mla_kv, &kpe_ptr, &ly->k_cache, &kv_r, &rd_u, &pos, &one };
            g_cu.cuLaunchKernel(g_cu.k_mla_store_latent, (ldim + 255u) / 256u, 1, 1, 256, 1, 1, 0, st, sargs, NULL);
            void* qargs[] = { &m->d_q, &ly->mla_wuk_f16, &m->d_mla_qabs, &nh_u, &dh_u, &nope_d, &rd_u, &kv_r, &one };
            g_cu.cuLaunchKernel(g_cu.k_mla_absorb_q, (ldim + 3u) / 4u, nh_u, 1, 128, 1, 1, 0, st, qargs, NULL);
            void* fargs[] = { &m->d_mla_qabs, &ly->k_cache, &m->d_mla_olat, &nh_u, &kv_r, &rd_u, &mla_scale, &pos, &one };
            g_cu.cuLaunchKernel(g_cu.k_flash_mla_latent, nh_u, 1, 1, 128, 1, 1, 0, st, fargs, NULL);
            void* vargs[] = { &m->d_mla_olat, &ly->mla_wuv_f16, &m->d_ctx, &nh_u, &kv_r, &v_d, &out_st, &one };
            g_cu.cuLaunchKernel(g_cu.k_mla_proj_v, (out_st + 3u) / 4u, nh_u, 1, 128, 1, 1, 0, st, vargs, NULL);
            mla_done = 1;
        } else if (ly->wkv_b.dptr) {
            uint32_t kvb_out = nh_u * (nope_d + v_d);
            launch_gemv(st, &ly->wkv_b, m->d_mla_kv, m->d_sh_g, kvb_out, kv_r, 0);
            void* margs[] = { &m->d_sh_g, &kpe_ptr, &m->d_k, &m->d_v, &nh_u, &nope_d, &rd_u, &v_d, &dh_u, &one };
            g_cu.cuLaunchKernel(g_cu.k_mla_split_kv, (dh_u + 31u) / 32u, nh_u, 1, 32, 1, 1, 0, st, margs, NULL);
            // Store K/V with rope_dim=0 since RoPE was already applied to Q_pe & k_pe!
            uint32_t zrd = 0u, nkv = nh_u, hd2 = dh_u >> 1;
            void* rargs[] = { &m->d_q, &m->d_k, &m->d_v, &ly->k_cache, &ly->v_cache, &rf, &nh_u, &nkv, &dh_u, &zrd, &pos, &theta, &rn, &hf };
            g_cu.cuLaunchKernel(g_cu.k_rope_kv, (hd2 + 31u) / 32u, nh_u, 1, 32, 1, 1, 0, st, rargs, NULL);
            uint32_t kv_len = pos + 1u;
            void* aargs[] = { &m->d_q, &ly->k_cache, &ly->v_cache, &m->d_ctx, &nh_u, &nkv, &dh_u, &kv_len, &out_st, &mla_scale };
            g_cu.cuLaunchKernel(g_cu.k_flash_dec, nh_u, 1, 1, 128, 1, 1, 0, st, aargs, NULL);
            mla_done = 1;
        }
    } else if (ly->wq.ggml_type == CM_GGML_Q4_0 && ly->wk.ggml_type == CM_GGML_Q4_0 && ly->wv.ggml_type == CM_GGML_Q4_0 && !has_qkv_lora) {
        uint32_t tot = qdim + kvdim + kvdim;
        if (g_cu.fast_dp4a) {
            launch_quant_q8_1(st, m->d_xn, dm);
            void* qargs[] = { &pq, &pk, &pv, &g_cu.d_xq8, &m->d_q, &m->d_k, &m->d_v, &qdim, &kvdim, &dm };
            g_cu.cuLaunchKernel(g_cu.k_gemv_qkv_q4_dp4a, (tot + 15) / 16, 1, 1, 128, 1, 1, 0, st, qargs, NULL);
        } else {
            void* qargs[] = { &pq, &pk, &pv, &m->d_xn, &m->d_q, &m->d_k, &m->d_v, &qdim, &kvdim, &dm };
            g_cu.cuLaunchKernel(g_cu.k_gemv_qkv_q4, (tot + 7) / 8, 1, 1, 128, 1, 1, 0, st, qargs, NULL);
        }
    } else {
        launch_gemv_raw(st, pq, ly->wq.ggml_type, m->d_xn, m->d_q, qdim, dm, 0);
        launch_gemv_raw(st, pk, ly->wk.ggml_type, m->d_xn, m->d_k, kvdim, dm, 0);
        launch_gemv_raw(st, pv, ly->wv.ggml_type, m->d_xn, m->d_v, kvdim, dm, 0);
    }
    if (m->active_lora > 0 && L >= 0) {
        launch_lora_delta(m, st, L, 0, m->d_xn, m->d_q, qdim, dm, 1u);
        launch_lora_delta(m, st, L, 1, m->d_xn, m->d_k, kvdim, dm, 1u);
        launch_lora_delta(m, st, L, 2, m->d_xn, m->d_v, kvdim, dm, 1u);
    }
    if (!mla_done) {
        if (ly->has_qkv_bias) {
            launch_add_bias(st, m->d_q, &ly->bq, qdim);
            launch_add_bias(st, m->d_k, &ly->bk, kvdim);
            launch_add_bias(st, m->d_v, &ly->bv, kvdim);
        }
        if (ly->has_qk_norm) {
            if (ly->qk_norm_per_head) {
                launch_rmsnorm(st, m->d_q, &ly->q_norm, m->d_q, (uint32_t)m->d_head, (uint32_t)m->n_heads, m->rms_eps, 1);
                launch_rmsnorm(st, m->d_k, &ly->k_norm, m->d_k, (uint32_t)m->d_head, (uint32_t)m->n_kv, m->rms_eps, 1);
            } else {
                launch_rmsnorm(st, m->d_q, &ly->q_norm, m->d_q, qdim, 1, m->rms_eps, 0);
                launch_rmsnorm(st, m->d_k, &ly->k_norm, m->d_k, kvdim, 1, m->rms_eps, 0);
            }
        }
        uint32_t nh = (uint32_t)m->n_heads, nkv = (uint32_t)m->n_kv, dh = (uint32_t)m->d_head, rd = (uint32_t)m->rope_dim;
        uint32_t rn = (uint32_t)m->rope_neox, hf = (uint32_t)m->has_rope_freqs;
        float theta = m->rope_theta;
        CUdeviceptr rf = hf ? m->rope_freqs.dptr : m->d_argmax;
        uint32_t hd2 = dh >> 1;
        if (m->kv_quant_q8) {
            if (indirect) {
                void* rargs[] = { &m->d_q, &m->d_k, &m->d_v, &ly->k_cache, &ly->v_cache, &rf, &nh, &nkv, &dh, &rd, &m->d_step_args, &theta, &rn, &hf };
                g_cu.cuLaunchKernel(g_cu.k_rope_kv_q8_ind, nh, 1, 1, 128, 1, 1, 0, st, rargs, NULL);
                void* aargs[] = { &m->d_q, &ly->k_cache, &ly->v_cache, &m->d_ctx, &nh, &nkv, &dh, &m->d_step_args };
                g_cu.cuLaunchKernel(g_cu.k_flash_dec_q8_ind, nh, 1, 1, 128, 1, 1, 0, st, aargs, NULL);
            } else {
                uint32_t one = 1u;
                void* rargs[] = { &m->d_q, &m->d_k, &m->d_v, &ly->k_cache, &ly->v_cache, &rf, &nh, &nkv, &dh, &rd, &pos, &theta, &rn, &hf, &one };
                g_cu.cuLaunchKernel(g_cu.k_rope_kv_q8_b, nh, 1, 1, 128, 1, 1, 0, st, rargs, NULL);
                void* aargs[] = { &m->d_q, &ly->k_cache, &ly->v_cache, &m->d_ctx, &nh, &nkv, &dh, &pos, &one };
                g_cu.cuLaunchKernel(g_cu.k_flash_prefill_q8, nh, 1, 1, 128, 1, 1, 0, st, aargs, NULL);
            }
        } else if (indirect) {
            void* rargs[] = { &m->d_q, &m->d_k, &m->d_v, &ly->k_cache, &ly->v_cache, &rf, &nh, &nkv, &dh, &rd, &m->d_step_args, &theta, &rn, &hf };
            g_cu.cuLaunchKernel(g_cu.k_rope_kv_ind, (hd2 + 31) / 32, nh, 1, 32, 1, 1, 0, st, rargs, NULL);
            void* aargs[] = { &m->d_q, &ly->k_cache, &ly->v_cache, &m->d_ctx, &nh, &nkv, &dh, &m->d_step_args };
            g_cu.cuLaunchKernel(g_cu.k_flash_dec_ind, nh, 1, 1, 128, 1, 1, 0, st, aargs, NULL);
        } else {
            if (m->has_mrope == 2) {
                uint32_t hr = rd >> 1;
                uint32_t ms0 = (m->mrope_sections[0] > 0) ? (uint32_t)m->mrope_sections[0] : (hr / 3u);
                uint32_t ms1 = (m->mrope_sections[1] > 0) ? (uint32_t)m->mrope_sections[1] : (hr / 3u);
                uint32_t pt = (uint32_t)(m->mrope_pos[0] >= 0 ? m->mrope_pos[0] : (int)pos);
                uint32_t ph = (uint32_t)(m->mrope_pos[1] >= 0 ? m->mrope_pos[1] : (int)pos);
                uint32_t pw = (uint32_t)(m->mrope_pos[2] >= 0 ? m->mrope_pos[2] : (int)pos);
                void* rargs[] = { &m->d_q, &m->d_k, &m->d_v, &ly->k_cache, &ly->v_cache, &rf, &nh, &nkv, &dh, &rd, &pos, &theta, &rn, &hf, &ms0, &ms1, &pt, &ph, &pw };
                g_cu.cuLaunchKernel(g_cu.k_rope_kv_mrope, (hd2 + 31) / 32, nh, 1, 32, 1, 1, 0, st, rargs, NULL);
            } else {
                void* rargs[] = { &m->d_q, &m->d_k, &m->d_v, &ly->k_cache, &ly->v_cache, &rf, &nh, &nkv, &dh, &rd, &pos, &theta, &rn, &hf };
                g_cu.cuLaunchKernel(g_cu.k_rope_kv, (hd2 + 31) / 32, nh, 1, 32, 1, 1, 0, st, rargs, NULL);
            }
            uint32_t kv_len = pos + 1u;
            float std_scale = 1.0f / sqrtf((float)dh);
            void* aargs[] = { &m->d_q, &ly->k_cache, &ly->v_cache, &m->d_ctx, &nh, &nkv, &dh, &kv_len, &dh, &std_scale };
            g_cu.cuLaunchKernel(g_cu.k_flash_dec, nh, 1, 1, 128, 1, 1, 0, st, aargs, NULL);
        }
    }

    if (ring_base) launch_gemv_raw(st, ring_base + m->dense_off_wo, ly->wo.ggml_type, m->d_ctx, m->d_x, dm, wo_in, 1);
    else launch_gemv(st, &ly->wo, m->d_ctx, m->d_x, dm, wo_in, 1);
    if (m->active_lora > 0 && L >= 0) launch_lora_delta(m, st, L, 3, m->d_ctx, m->d_x, dm, wo_in, 1u);
    launch_rmsnorm(st, m->d_x, &ly->ffn_norm, m->d_xn, dm, 1, m->rms_eps, 0);
}

static void launch_layer_attn_ex(CmModel* m, CUstream st, int L, uint32_t pos, CUdeviceptr ring_base, int indirect) {
    launch_layer_attn_ptr(m, st, &m->layers[L], pos, ring_base, indirect);
}

static void launch_layer_attn(CmModel* m, CUstream st, int L, uint32_t pos, CUdeviceptr ring_base) {
    launch_layer_attn_ptr(m, st, &m->layers[L], pos, ring_base, 0);
}

static void launch_embed(CmModel* m, CUstream st, uint32_t tok_id) {
    uint32_t dm = (uint32_t)m->d_model, gt = m->tok_embd.ggml_type;
    if (m->resident_mode && m->tok_embd.dptr) {
        CUdeviceptr ep = m->tok_embd.dptr;
        void* args[] = { &ep, &m->d_x, &tok_id, &dm, &gt };
        g_cu.cuLaunchKernel(g_cu.k_embed, (dm + 255) / 256, 1, 1, 256, 1, 1, 0, st, args, NULL);
    } else {
        uint64_t rb = cm_ggml_type_bytes(gt, (uint64_t)dm);
        cm_read_slice(m, m->tok_embd.shard_idx, m->h_emb_row, (size_t)rb, m->tok_embd.file_off + (uint64_t)tok_id * rb);
        g_cu.cuMemcpyHtoDAsync(m->d_emb_row, m->h_emb_row, (size_t)rb, st);
        uint32_t z = 0;
        void* args[] = { &m->d_emb_row, &m->d_x, &z, &dm, &gt };
        g_cu.cuLaunchKernel(g_cu.k_embed, (dm + 255) / 256, 1, 1, 256, 1, 1, 0, st, args, NULL);
    }
}

static void launch_final_lm_head(CmModel* m, CUstream st) {
    launch_rmsnorm(st, m->d_x, &m->out_norm, m->d_xn, (uint32_t)m->d_model, 1, m->rms_eps, 0);
    launch_gemv(st, &m->lm_head, m->d_xn, m->d_logits, (uint32_t)m->vocab_size, (uint32_t)m->d_model, 0);
    uint32_t v = (uint32_t)m->vocab_size;
    void* args[] = { &m->d_logits, &m->d_argmax, &v };
    g_cu.cuLaunchKernel(g_cu.k_argmax, 1, 1, 1, 256, 1, 1, 0, st, args, NULL);
}

static int launch_tail_until_next_moe(CmModel* m, CUstream st, int start_L, uint32_t pos) {
    int L = start_L;
    while (L < m->n_layers && m->layers[L].is_dense) {
        CmLayer* ly = &m->layers[L];
        launch_layer_attn(m, st, L, pos, 0);
        launch_dense_ffn_raw(m, st, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                             ly->ffn_up.dptr, ly->ffn_up.ggml_type,
                             ly->ffn_down.dptr, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense);
        L++;
    }
    if (L < m->n_layers) {
        CmLayer* ly = &m->layers[L];
        launch_layer_attn(m, st, L, pos, 0);
        launch_gemv(st, &ly->router, m->d_xn, m->d_rlogits, (uint32_t)m->n_experts, (uint32_t)m->d_model, 0);
        g_cu.cuMemcpyDtoHAsync(m->h_rlogits, m->d_rlogits, (size_t)m->n_experts * 4, st);
        if (m->enable_overlap && !m->use_nocache && m->h_xn) {
            g_cu.cuMemcpyDtoHAsync(m->h_xn, m->d_xn, (size_t)m->d_model * 4, st);
        }
    } else {
        launch_final_lm_head(m, st);
        g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, st);
    }
    return L;
}

static void launch_moe_slots(CmModel* m, CUstream st, int L, uint32_t n_act) {
    CmLayer* ly = &m->layers[L];
    uint32_t dm = (uint32_t)m->d_model, df = (uint32_t)m->d_ff_exp;
    uint64_t gb = ly->gate_exp_bytes, sb = ly->slot_stride_bytes;
    uint64_t doff = ly->gate_exp_bytes + ly->up_exp_bytes;
    int is_q4k_gu = (ly->gate_exps.ggml_type == CM_GGML_Q4_K);
    int is_q4k_dn = (ly->down_exps.ggml_type == CM_GGML_Q4_K);
    int is_q6k_dn = (ly->down_exps.ggml_type == CM_GGML_Q6_K);
    int is_q8_dn  = (ly->down_exps.ggml_type == CM_GGML_Q8_0);
    int is_q5_0_dn = (ly->down_exps.ggml_type == CM_GGML_Q5_0);
    int is_q5_1_dn = (ly->down_exps.ggml_type == CM_GGML_Q5_1);
    void* gu_args[] = { &ly->cache_dptr, &m->d_xn, &m->d_eidx, &m->d_h, &dm, &df, &n_act, &gb, &sb };
    g_cu.cuLaunchKernel(is_q4k_gu ? g_cu.k_moe_slot_gu_q4k : g_cu.k_moe_slot_gu_q4,
                        is_q4k_gu ? ((df + 3) / 4) : ((df + 7) / 8), n_act, 1, 128, 1, 1, 0, st, gu_args, NULL);
    CUfunction dfn = is_q6k_dn ? g_cu.k_moe_slot_down_q6k
        : (is_q4k_dn ? g_cu.k_moe_slot_down_q4k
           : (is_q8_dn ? g_cu.k_moe_slot_down_q8
              : (is_q5_0_dn ? g_cu.k_moe_slot_down_q5_0
                 : (is_q5_1_dn ? g_cu.k_moe_slot_down_q5_1
                    : ((ly->down_exps.ggml_type == CM_GGML_Q4_1) ? g_cu.k_moe_slot_down_q4_1 : g_cu.k_moe_slot_down_q4)))));
    void* dn_args[] = { &ly->cache_dptr, &m->d_h, &m->d_eidx, &m->d_ewt, &m->d_x, &dm, &df, &n_act, &doff, &sb };
    g_cu.cuLaunchKernel(dfn, (is_q4k_dn || is_q6k_dn || is_q8_dn || is_q5_0_dn || is_q5_1_dn) ? ((dm + 3) / 4) : ((dm + 15) / 16), 1, 1, 128, 1, 1, 0, st, dn_args, NULL);
}

// ============================================================================
// Persistent Host Staging Worker Pool + Pipelined Async PCIe DMA
// ============================================================================
static void cm_stage_expert(CmModel* m, int L, int eid, uint8_t* dst) {
    CmLayer* ly = &m->layers[L];
    cm_read_slice(m, ly->gate_exps.shard_idx, dst, (size_t)ly->gate_exp_bytes,
                  ly->gate_exps.file_off + (uint64_t)eid * ly->gate_exp_bytes);
    cm_read_slice(m, ly->up_exps.shard_idx, dst + ly->gate_exp_bytes, (size_t)ly->up_exp_bytes,
                  ly->up_exps.file_off + (uint64_t)eid * ly->up_exp_bytes);
    cm_read_slice(m, ly->down_exps.shard_idx, dst + ly->gate_exp_bytes + ly->up_exp_bytes, (size_t)ly->down_exp_bytes,
                  ly->down_exps.file_off + (uint64_t)eid * ly->down_exp_bytes);
}

typedef struct { CmModel* m; int tid; } IoPoolArg;

static void* cm_io_pool_worker(void* arg) {
    IoPoolArg* a = (IoPoolArg*)arg;
    CmModel* m = a->m;
    int tid = a->tid;
    free(a);
    uint64_t seen_gen = 0;
    for (;;) {
        pthread_mutex_lock(&m->io_mu);
        while (!m->io_stop && m->io_gen == seen_gen) pthread_cond_wait(&m->io_cv_go, &m->io_mu);
        if (m->io_stop) { pthread_mutex_unlock(&m->io_mu); break; }
        seen_gen = m->io_gen;
        int mode = m->io_mode;
        int n_tasks = m->io_n_tasks;
        int L = m->io_layer;
        pthread_mutex_unlock(&m->io_mu);

        if (mode == 1) {
            for (int i = tid; i < n_tasks; i += CM_IO_WORKERS) {
                cm_stage_expert(m, L, m->io_eids[i], m->io_dsts[i]);
                __sync_synchronize();
                m->io_task_ready[i] = 1;
                pthread_mutex_lock(&m->io_mu);
                m->io_done_cnt++;
                pthread_cond_signal(&m->io_cv_done);
                pthread_mutex_unlock(&m->io_mu);
            }
        } else if (mode == 2) {
            if (tid < n_tasks) {
                size_t tot = m->io_dense_len;
                size_t chunk = (tot + (size_t)n_tasks - 1) / (size_t)n_tasks;
                size_t s0 = (size_t)tid * chunk;
                if (s0 < tot) {
                    size_t len = (s0 + chunk <= tot) ? chunk : (tot - s0);
                    cm_read_slice(m, m->io_dense_shard, m->io_dense_dst + s0, len, m->io_dense_off + (uint64_t)s0);
                }
                pthread_mutex_lock(&m->io_mu);
                m->io_done_cnt++;
                if (m->io_done_cnt >= n_tasks) pthread_cond_signal(&m->io_cv_done);
                pthread_mutex_unlock(&m->io_mu);
            }
        }
    }
    return NULL;
}

static void cm_io_pool_start(CmModel* m) {
    if (m->io_pool_init) return;
    pthread_mutex_init(&m->io_mu, NULL);
    pthread_cond_init(&m->io_cv_go, NULL);
    pthread_cond_init(&m->io_cv_done, NULL);
    m->io_stop = 0; m->io_gen = 0; m->io_pool_init = 1;
    for (int t = 0; t < CM_IO_WORKERS; t++) {
        IoPoolArg* a = (IoPoolArg*)malloc(sizeof(IoPoolArg));
        a->m = m; a->tid = t;
        pthread_create(&m->io_threads[t], NULL, cm_io_pool_worker, a);
    }
}

static void cm_io_pool_stop(CmModel* m) {
    if (!m->io_pool_init) return;
    pthread_mutex_lock(&m->io_mu);
    m->io_stop = 1;
    pthread_cond_broadcast(&m->io_cv_go);
    pthread_mutex_unlock(&m->io_mu);
    for (int t = 0; t < CM_IO_WORKERS; t++) pthread_join(m->io_threads[t], NULL);
    pthread_mutex_destroy(&m->io_mu);
    pthread_cond_destroy(&m->io_cv_go);
    pthread_cond_destroy(&m->io_cv_done);
    m->io_pool_init = 0;
}

static void cm_read_slice_parallel(CmModel* m, int shard, uint8_t* dst, size_t n, uint64_t off) {
    if (!m->io_pool_init || n < (4u << 20)) {
        cm_read_slice(m, shard, dst, n, off);
        return;
    }
    pthread_mutex_lock(&m->io_mu);
    m->io_mode = 2;
    m->io_n_tasks = CM_IO_WORKERS;
    m->io_done_cnt = 0;
    m->io_dense_shard = shard;
    m->io_dense_dst = dst;
    m->io_dense_off = off;
    m->io_dense_len = n;
    m->io_gen++;
    pthread_cond_broadcast(&m->io_cv_go);
    while (m->io_done_cnt < CM_IO_WORKERS) pthread_cond_wait(&m->io_cv_done, &m->io_mu);
    pthread_mutex_unlock(&m->io_mu);
}

static int cm_pick_lru(CmModel* m, CmLayer* ly, const int* pinned, int npin) {
    int victim = -1; uint64_t best_ts = UINT64_MAX;
    for (int s = 0; s < m->cache_slots_per_layer; s++) {
        int is_p = 0;
        for (int p = 0; p < npin; p++) if (pinned[p] == s) { is_p = 1; break; }
        if (is_p) continue;
        if (ly->slot_to_exp[s] == -2) continue; // reserved for in-flight async promotion
        if (ly->slot_to_exp[s] == -1) return s;
        if (ly->slot_lru[s] < best_ts) { best_ts = ly->slot_lru[s]; victim = s; }
    }
    return (victim >= 0) ? victim : 0;
}

// ============================================================================
// Exact FP32 AVX2+FMA+F16C In-Place DDR4 CPU Kernels + 32-Thread Spin-Wait Pool
// ============================================================================
static inline float cm_f16_bits_to_f32(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    uint32_t s = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x03FFu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else { e = 127 - 14; while (!(m & 0x0400u)) { m <<= 1; e--; } m &= 0x03FFu; bits = s | (e << 23) | (m << 13); }
    } else if (e == 31) bits = s | 0x7F800000u | (m << 13);
    else bits = s | ((e + 112u) << 23) | (m << 13);
    float f; memcpy(&f, &bits, 4); return f;
#endif
}

static inline void cm_cpu_pause(void) {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
static inline float cm_hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

// Quantize FP32 activation vector into Q8_0 blocks (int8[32] + float scale + exact FP32 block sum)
static void cm_quantize_q8_0(const float* __restrict__ x, int8_t* __restrict__ q8,
                             float* __restrict__ ds, float* __restrict__ xsum, uint32_t dim) {
    uint32_t nb = dim >> 5;
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (uint32_t b = 0; b < nb; b++) {
        const float* xp = x + ((size_t)b << 5);
        __m256 v0 = _mm256_loadu_ps(xp),      v1 = _mm256_loadu_ps(xp + 8);
        __m256 v2 = _mm256_loadu_ps(xp + 16), v3 = _mm256_loadu_ps(xp + 24);
        __m256 sv = _mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3));
        xsum[b] = cm_hsum256(sv);
        __m256 am = _mm256_max_ps(
            _mm256_max_ps(_mm256_andnot_ps(sign_mask, v0), _mm256_andnot_ps(sign_mask, v1)),
            _mm256_max_ps(_mm256_andnot_ps(sign_mask, v2), _mm256_andnot_ps(sign_mask, v3)));
        __m128 am4 = _mm_max_ps(_mm256_castps256_ps128(am), _mm256_extractf128_ps(am, 1));
        am4 = _mm_max_ps(am4, _mm_movehl_ps(am4, am4));
        am4 = _mm_max_ss(am4, _mm_shuffle_ps(am4, am4, 1));
        float amax = _mm_cvtss_f32(am4);
        float d = amax / 127.0f;
        float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        ds[b] = d;
        __m256 vid = _mm256_set1_ps(id);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, vid));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, vid));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, vid));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, vid));
        // Pack 32x int32 -> 32x int8 preserving exact lane order [0..31]!
        __m128i p01_lo = _mm_packs_epi32(_mm256_castsi256_si128(i0), _mm256_extracti128_si256(i0, 1));
        __m128i p01_hi = _mm_packs_epi32(_mm256_castsi256_si128(i1), _mm256_extracti128_si256(i1, 1));
        __m128i p23_lo = _mm_packs_epi32(_mm256_castsi256_si128(i2), _mm256_extracti128_si256(i2, 1));
        __m128i p23_hi = _mm_packs_epi32(_mm256_castsi256_si128(i3), _mm256_extracti128_si256(i3, 1));
        _mm_storeu_si128((__m128i*)(q8 + ((size_t)b << 5)),      _mm_packs_epi16(p01_lo, p01_hi));
        _mm_storeu_si128((__m128i*)(q8 + ((size_t)b << 5) + 16), _mm_packs_epi16(p23_lo, p23_hi));
    }
}

// Ultra-fast AVX2 INT8 `vpmaddubsw` Q4_0 x Q8_0 2-row dot product (only 8 SIMD ops per 32 weights -> 180+ GB/s!)
static void cm_cpu_dot_q4_0_q8_2rows(const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1,
                                     const int8_t* __restrict__ xq8, const float* __restrict__ xd,
                                     const float* __restrict__ xsum, uint32_t nb, float* o0, float* o1) {
    const __m128i m0f    = _mm_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    float b0 = 0.0f, b1 = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        __m256i xq = _mm256_loadu_si256((const __m256i*)(xq8 + ((size_t)b << 5)));
        float d_x = xd[b], xs = xsum[b];

        const uint8_t* p0 = w0 + (size_t)b * 18u;
        const uint8_t* p1 = w1 + (size_t)b * 18u;
        _mm_prefetch((const char*)(p0 + 18u * 8u), _MM_HINT_T0);
        _mm_prefetch((const char*)(p1 + 18u * 8u), _MM_HINT_T0);

        float dw0 = _cvtsh_ss(*(const uint16_t*)p0);
        __m128i q0 = _mm_loadu_si128((const __m128i*)(p0 + 2));
        __m256i u0 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q0, 4), m0f), _mm_and_si128(q0, m0f));
        __m256i s0 = _mm256_madd_epi16(_mm256_maddubs_epi16(u0, xq), ones16);
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(dw0 * d_x), _mm256_cvtepi32_ps(s0), acc0);
        b0 += dw0 * xs;

        float dw1 = _cvtsh_ss(*(const uint16_t*)p1);
        __m128i q1 = _mm_loadu_si128((const __m128i*)(p1 + 2));
        __m256i u1 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q1, 4), m0f), _mm_and_si128(q1, m0f));
        __m256i s1 = _mm256_madd_epi16(_mm256_maddubs_epi16(u1, xq), ones16);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(dw1 * d_x), _mm256_cvtepi32_ps(s1), acc1);
        b1 += dw1 * xs;
    }
    *o0 = cm_hsum256(acc0) - 8.0f * b0;
    *o1 = cm_hsum256(acc1) - 8.0f * b1;
}

// 4-row unrolled AVX2 INT8 `vpmaddubsw` Q4_0 x Q8_0 (4x xq reuse, saturates 8-channel EPYC DDR4!)
static void cm_cpu_dot_q4_0_q8_4rows(const uint8_t* __restrict__ w0, uint64_t rb,
                                     const int8_t* __restrict__ xq8, const float* __restrict__ xd,
                                     const float* __restrict__ xsum, uint32_t nb, float out[4]) {
    const uint8_t* w1 = w0 + rb;
    const uint8_t* w2 = w0 + rb * 2u;
    const uint8_t* w3 = w0 + rb * 3u;
    const __m128i m0f    = _mm_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        __m256i xq = _mm256_loadu_si256((const __m256i*)(xq8 + ((size_t)b << 5)));
        float d_x = xd[b], xs = xsum[b];
        size_t boff = (size_t)b * 18u;
        const uint8_t *p0 = w0 + boff, *p1 = w1 + boff, *p2 = w2 + boff, *p3 = w3 + boff;
        _mm_prefetch((const char*)(p0 + 18u * 8u), _MM_HINT_T0);
        _mm_prefetch((const char*)(p1 + 18u * 8u), _MM_HINT_T0);
        _mm_prefetch((const char*)(p2 + 18u * 8u), _MM_HINT_T0);
        _mm_prefetch((const char*)(p3 + 18u * 8u), _MM_HINT_T0);

        float dw0 = _cvtsh_ss(*(const uint16_t*)p0);
        __m128i q0 = _mm_loadu_si128((const __m128i*)(p0 + 2));
        __m256i u0 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q0, 4), m0f), _mm_and_si128(q0, m0f));
        a0 = _mm256_fmadd_ps(_mm256_set1_ps(dw0 * d_x), _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u0, xq), ones16)), a0);
        b0 += dw0 * xs;

        float dw1 = _cvtsh_ss(*(const uint16_t*)p1);
        __m128i q1 = _mm_loadu_si128((const __m128i*)(p1 + 2));
        __m256i u1 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q1, 4), m0f), _mm_and_si128(q1, m0f));
        a1 = _mm256_fmadd_ps(_mm256_set1_ps(dw1 * d_x), _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u1, xq), ones16)), a1);
        b1 += dw1 * xs;

        float dw2 = _cvtsh_ss(*(const uint16_t*)p2);
        __m128i q2 = _mm_loadu_si128((const __m128i*)(p2 + 2));
        __m256i u2 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q2, 4), m0f), _mm_and_si128(q2, m0f));
        a2 = _mm256_fmadd_ps(_mm256_set1_ps(dw2 * d_x), _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u2, xq), ones16)), a2);
        b2 += dw2 * xs;

        float dw3 = _cvtsh_ss(*(const uint16_t*)p3);
        __m128i q3 = _mm_loadu_si128((const __m128i*)(p3 + 2));
        __m256i u3 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q3, 4), m0f), _mm_and_si128(q3, m0f));
        a3 = _mm256_fmadd_ps(_mm256_set1_ps(dw3 * d_x), _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u3, xq), ones16)), a3);
        b3 += dw3 * xs;
    }
    out[0] = cm_hsum256(a0) - 8.0f * b0;
    out[1] = cm_hsum256(a1) - 8.0f * b1;
    out[2] = cm_hsum256(a2) - 8.0f * b2;
    out[3] = cm_hsum256(a3) - 8.0f * b3;
}

static void cm_cpu_dot_q4_1_q8_2rows(const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1,
                                     const int8_t* __restrict__ xq8, const float* __restrict__ xd,
                                     const float* __restrict__ xsum, uint32_t nb, float* o0, float* o1) {
    const __m128i m0f    = _mm_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    float m0 = 0.0f, m1 = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        __m256i xq = _mm256_loadu_si256((const __m256i*)(xq8 + ((size_t)b << 5)));
        float d_x = xd[b], xs = xsum[b];
        const uint8_t* p0 = w0 + (size_t)b * 20u;
        const uint8_t* p1 = w1 + (size_t)b * 20u;

        float dw0 = _cvtsh_ss(*(const uint16_t*)p0), mw0 = _cvtsh_ss(*(const uint16_t*)(p0 + 2));
        __m128i q0 = _mm_loadu_si128((const __m128i*)(p0 + 4));
        __m256i u0 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q0, 4), m0f), _mm_and_si128(q0, m0f));
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(dw0 * d_x), _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u0, xq), ones16)), acc0);
        m0 += mw0 * xs;

        float dw1 = _cvtsh_ss(*(const uint16_t*)p1), mw1 = _cvtsh_ss(*(const uint16_t*)(p1 + 2));
        __m128i q1 = _mm_loadu_si128((const __m128i*)(p1 + 4));
        __m256i u1 = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q1, 4), m0f), _mm_and_si128(q1, m0f));
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(dw1 * d_x), _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16(u1, xq), ones16)), acc1);
        m1 += mw1 * xs;
    }
    *o0 = cm_hsum256(acc0) + m0;
    *o1 = cm_hsum256(acc1) + m1;
}

static void cm_cpu_dot_q4_0_fp32_2rows(const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1,
                                       const float* __restrict__ x, uint32_t nb, float* o0, float* o1) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m256i b8  = _mm256_set1_epi32(8);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (uint32_t b = 0; b < nb; b++) {
        const float* xp = x + ((size_t)b << 5);
        __m256 x0 = _mm256_loadu_ps(xp),      x1 = _mm256_loadu_ps(xp + 8);
        __m256 x2 = _mm256_loadu_ps(xp + 16), x3 = _mm256_loadu_ps(xp + 24);
        const uint8_t *p0 = w0 + (size_t)b * 18u, *p1 = w1 + (size_t)b * 18u;
        __m256 d0 = _mm256_set1_ps(_cvtsh_ss(*(const uint16_t*)p0));
        __m128i q0 = _mm_loadu_si128((const __m128i*)(p0 + 2));
        __m128i lo0 = _mm_and_si128(q0, m0f), hi0 = _mm_and_si128(_mm_srli_epi16(q0, 4), m0f);
        __m256 s0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(lo0), b8)), x0,
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(lo0, 8)), b8)), x1,
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(hi0), b8)), x2,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(hi0, 8)), b8)), x3))));
        acc0 = _mm256_fmadd_ps(d0, s0, acc0);

        __m256 d1 = _mm256_set1_ps(_cvtsh_ss(*(const uint16_t*)p1));
        __m128i q1 = _mm_loadu_si128((const __m128i*)(p1 + 2));
        __m128i lo1 = _mm_and_si128(q1, m0f), hi1 = _mm_and_si128(_mm_srli_epi16(q1, 4), m0f);
        __m256 s1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(lo1), b8)), x0,
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(lo1, 8)), b8)), x1,
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(hi1), b8)), x2,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(hi1, 8)), b8)), x3))));
        acc1 = _mm256_fmadd_ps(d1, s1, acc1);
    }
    *o0 = cm_hsum256(acc0); *o1 = cm_hsum256(acc1);
}

static void cm_cpu_dot_q4_1_fp32_2rows(const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1,
                                       const float* __restrict__ x, uint32_t nb, float* o0, float* o1) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (uint32_t b = 0; b < nb; b++) {
        const float* xp = x + ((size_t)b << 5);
        __m256 x0 = _mm256_loadu_ps(xp),      x1 = _mm256_loadu_ps(xp + 8);
        __m256 x2 = _mm256_loadu_ps(xp + 16), x3 = _mm256_loadu_ps(xp + 24);
        __m256 xs = _mm256_add_ps(_mm256_add_ps(x0, x1), _mm256_add_ps(x2, x3));
        const uint8_t *p0 = w0 + (size_t)b * 20u, *p1 = w1 + (size_t)b * 20u;
        __m256 d0 = _mm256_set1_ps(_cvtsh_ss(*(const uint16_t*)p0)), mv0 = _mm256_set1_ps(_cvtsh_ss(*(const uint16_t*)(p0 + 2)));
        __m128i q0 = _mm_loadu_si128((const __m128i*)(p0 + 4));
        __m128i lo0 = _mm_and_si128(q0, m0f), hi0 = _mm_and_si128(_mm_srli_epi16(q0, 4), m0f);
        __m256 s0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo0)), x0,
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo0, 8))), x1,
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi0)), x2,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi0, 8))), x3))));
        acc0 = _mm256_fmadd_ps(d0, s0, _mm256_fmadd_ps(mv0, xs, acc0));

        __m256 d1 = _mm256_set1_ps(_cvtsh_ss(*(const uint16_t*)p1)), mv1 = _mm256_set1_ps(_cvtsh_ss(*(const uint16_t*)(p1 + 2)));
        __m128i q1 = _mm_loadu_si128((const __m128i*)(p1 + 4));
        __m128i lo1 = _mm_and_si128(q1, m0f), hi1 = _mm_and_si128(_mm_srli_epi16(q1, 4), m0f);
        __m256 s1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo1)), x0,
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo1, 8))), x1,
                    _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi1)), x2,
                    _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi1, 8))), x3))));
        acc1 = _mm256_fmadd_ps(d1, s1, _mm256_fmadd_ps(mv1, xs, acc1));
    }
    *o0 = cm_hsum256(acc0); *o1 = cm_hsum256(acc1);
}

// High-precision INT16 activation quantization (32,767 levels = 256x finer than Q8_0 -> 100% BIT-EXACT on 28672-dim SwiGLU!)
static void cm_quantize_q16_0(const float* __restrict__ x, int16_t* __restrict__ q16,
                              float* __restrict__ ds, float* __restrict__ xsum, uint32_t dim) {
    uint32_t nb = dim >> 5;
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (uint32_t b = 0; b < nb; b++) {
        const float* xp = x + ((size_t)b << 5);
        __m256 v0 = _mm256_loadu_ps(xp),      v1 = _mm256_loadu_ps(xp + 8);
        __m256 v2 = _mm256_loadu_ps(xp + 16), v3 = _mm256_loadu_ps(xp + 24);
        xsum[b] = cm_hsum256(_mm256_add_ps(_mm256_add_ps(v0, v1), _mm256_add_ps(v2, v3)));
        __m256 am = _mm256_max_ps(
            _mm256_max_ps(_mm256_andnot_ps(sign_mask, v0), _mm256_andnot_ps(sign_mask, v1)),
            _mm256_max_ps(_mm256_andnot_ps(sign_mask, v2), _mm256_andnot_ps(sign_mask, v3)));
        __m128 am4 = _mm_max_ps(_mm256_castps256_ps128(am), _mm256_extractf128_ps(am, 1));
        am4 = _mm_max_ps(am4, _mm_movehl_ps(am4, am4));
        am4 = _mm_max_ss(am4, _mm_shuffle_ps(am4, am4, 1));
        float amax = _mm_cvtss_f32(am4);
        ds[b] = amax / 32767.0f;
        __m256 vid = _mm256_set1_ps((amax > 0.0f) ? (32767.0f / amax) : 0.0f);
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, vid));
        __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, vid));
        __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, vid));
        __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, vid));
        int16_t* dst = q16 + ((size_t)b << 5);
        _mm_storeu_si128((__m128i*)(dst +  0), _mm_packs_epi32(_mm256_castsi256_si128(i0), _mm256_extracti128_si256(i0, 1)));
        _mm_storeu_si128((__m128i*)(dst +  8), _mm_packs_epi32(_mm256_castsi256_si128(i1), _mm256_extracti128_si256(i1, 1)));
        _mm_storeu_si128((__m128i*)(dst + 16), _mm_packs_epi32(_mm256_castsi256_si128(i2), _mm256_extracti128_si256(i2, 1)));
        _mm_storeu_si128((__m128i*)(dst + 24), _mm_packs_epi32(_mm256_castsi256_si128(i3), _mm256_extracti128_si256(i3, 1)));
    }
}

// Contiguous NUMA-local Fused Gate+Up Q4_0_4R x INT16 (`_mm256_madd_epi16`): zero YMM spill via packed vbg/vbu + vector F16C!
static void cm_cpu_dot_q4_0_4r_gu_q16(const uint8_t* __restrict__ p, uint32_t nb,
                                      const int16_t* __restrict__ xq16, const float* __restrict__ xd,
                                      const float* __restrict__ xsum, float gout[4], float uout[4]) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    __m256 ag0 = _mm256_setzero_ps(), ag1 = _mm256_setzero_ps(), ag2 = _mm256_setzero_ps(), ag3 = _mm256_setzero_ps();
    __m256 au0 = _mm256_setzero_ps(), au1 = _mm256_setzero_ps(), au2 = _mm256_setzero_ps(), au3 = _mm256_setzero_ps();
    __m128 vbg = _mm_setzero_ps(), vbu = _mm_setzero_ps();
    for (uint32_t b = 0; b < nb; b++, p += 144u) {
        _mm_prefetch((const char*)(p + 144u * 4u), _MM_HINT_T0);
        _mm_prefetch((const char*)(p + 144u * 4u + 64u), _MM_HINT_T0);
        __m256i xlo = _mm256_loadu_si256((const __m256i*)(xq16 + ((size_t)b << 5)));
        __m256i xhi = _mm256_loadu_si256((const __m256i*)(xq16 + ((size_t)b << 5) + 16u));
        __m128 vxs = _mm_set1_ps(xsum[b]), vxd = _mm_set1_ps(xd[b]);
        __m128 vdg = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)p));
        __m128 vdu = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)(p + 72u)));
        vbg = _mm_fmadd_ps(vdg, vxs, vbg);
        vbu = _mm_fmadd_ps(vdu, vxs, vbu);
        float dg[4], du[4];
        _mm_storeu_ps(dg, _mm_mul_ps(vdg, vxd));
        _mm_storeu_ps(du, _mm_mul_ps(vdu, vxd));
        const uint8_t* qg = p + 8u;
        const uint8_t* qu = p + 80u;
        #define DO_GU16(idx, sc, qarr, acc) do { \
            __m128i qb = _mm_loadu_si128((const __m128i*)(qarr + (idx) * 16u)); \
            __m256i l16 = _mm256_cvtepu8_epi16(_mm_and_si128(qb, m0f)); \
            __m256i h16 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(qb, 4), m0f)); \
            __m256i s32 = _mm256_add_epi32(_mm256_madd_epi16(l16, xlo), _mm256_madd_epi16(h16, xhi)); \
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&(sc)[idx]), _mm256_cvtepi32_ps(s32), acc); \
        } while (0)
        DO_GU16(0, dg, qg, ag0); DO_GU16(1, dg, qg, ag1);
        DO_GU16(2, dg, qg, ag2); DO_GU16(3, dg, qg, ag3);
        DO_GU16(0, du, qu, au0); DO_GU16(1, du, qu, au1);
        DO_GU16(2, du, qu, au2); DO_GU16(3, du, qu, au3);
        #undef DO_GU16
    }
    float bg[4], bu[4];
    _mm_storeu_ps(bg, vbg); _mm_storeu_ps(bu, vbu);
    gout[0] = cm_hsum256(ag0) - 8.0f * bg[0]; gout[1] = cm_hsum256(ag1) - 8.0f * bg[1];
    gout[2] = cm_hsum256(ag2) - 8.0f * bg[2]; gout[3] = cm_hsum256(ag3) - 8.0f * bg[3];
    uout[0] = cm_hsum256(au0) - 8.0f * bu[0]; uout[1] = cm_hsum256(au1) - 8.0f * bu[1];
    uout[2] = cm_hsum256(au2) - 8.0f * bu[2]; uout[3] = cm_hsum256(au3) - 8.0f * bu[3];
}

// Contiguous NUMA-local Q4_0_4R x INT16 (`_mm256_madd_epi16`): vector F16C + 72B/blk sequential -> 200+ GB/s & BIT-EXACT!
static void cm_cpu_dot_q4_0_4r_q16(const uint8_t* __restrict__ p, uint32_t nb,
                                   const int16_t* __restrict__ hq16, const float* __restrict__ hd,
                                   const float* __restrict__ hsum, float out[4]) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m128 vb = _mm_setzero_ps();
    for (uint32_t b = 0; b < nb; b++, p += 72u) {
        _mm_prefetch((const char*)(p + 72u * 8u), _MM_HINT_T0);
        __m256i xlo = _mm256_loadu_si256((const __m256i*)(hq16 + ((size_t)b << 5)));
        __m256i xhi = _mm256_loadu_si256((const __m256i*)(hq16 + ((size_t)b << 5) + 16u));
        __m128 vd = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)p));
        vb = _mm_fmadd_ps(vd, _mm_set1_ps(hsum[b]), vb);
        float sc[4];
        _mm_storeu_ps(sc, _mm_mul_ps(vd, _mm_set1_ps(hd[b])));
        const uint8_t* q = p + 8u;
        #define DO_Q16(idx, acc) do { \
            __m128i qb = _mm_loadu_si128((const __m128i*)(q + (idx) * 16u)); \
            __m256i l16 = _mm256_cvtepu8_epi16(_mm_and_si128(qb, m0f)); \
            __m256i h16 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(qb, 4), m0f)); \
            __m256i s32 = _mm256_add_epi32(_mm256_madd_epi16(l16, xlo), _mm256_madd_epi16(h16, xhi)); \
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&sc[idx]), _mm256_cvtepi32_ps(s32), acc); \
        } while (0)
        DO_Q16(0, a0); DO_Q16(1, a1); DO_Q16(2, a2); DO_Q16(3, a3);
        #undef DO_Q16
    }
    float bv[4]; _mm_storeu_ps(bv, vb);
    out[0] = cm_hsum256(a0) - 8.0f * bv[0]; out[1] = cm_hsum256(a1) - 8.0f * bv[1];
    out[2] = cm_hsum256(a2) - 8.0f * bv[2]; out[3] = cm_hsum256(a3) - 8.0f * bv[3];
}

// Contiguous NUMA-local Q4_1_4R x INT16 (`_mm256_madd_epi16`): vector F16C + 80B/blk sequential -> BIT-EXACT!
static void cm_cpu_dot_q4_1_4r_q16(const uint8_t* __restrict__ p, uint32_t nb,
                                   const int16_t* __restrict__ hq16, const float* __restrict__ hd,
                                   const float* __restrict__ hsum, float out[4]) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
    __m128 vm = _mm_setzero_ps();
    for (uint32_t b = 0; b < nb; b++, p += 80u) {
        _mm_prefetch((const char*)(p + 80u * 8u), _MM_HINT_T0);
        __m256i xlo = _mm256_loadu_si256((const __m256i*)(hq16 + ((size_t)b << 5)));
        __m256i xhi = _mm256_loadu_si256((const __m256i*)(hq16 + ((size_t)b << 5) + 16u));
        __m128 vd = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)p));
        __m128 vw = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)(p + 8u)));
        vm = _mm_fmadd_ps(vw, _mm_set1_ps(hsum[b]), vm);
        float sc[4];
        _mm_storeu_ps(sc, _mm_mul_ps(vd, _mm_set1_ps(hd[b])));
        const uint8_t* q = p + 16u;
        #define DO_Q16_1(idx, acc) do { \
            __m128i qb = _mm_loadu_si128((const __m128i*)(q + (idx) * 16u)); \
            __m256i l16 = _mm256_cvtepu8_epi16(_mm_and_si128(qb, m0f)); \
            __m256i h16 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(qb, 4), m0f)); \
            __m256i s32 = _mm256_add_epi32(_mm256_madd_epi16(l16, xlo), _mm256_madd_epi16(h16, xhi)); \
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&sc[idx]), _mm256_cvtepi32_ps(s32), acc); \
        } while (0)
        DO_Q16_1(0, a0); DO_Q16_1(1, a1); DO_Q16_1(2, a2); DO_Q16_1(3, a3);
        #undef DO_Q16_1
    }
    float mv[4]; _mm_storeu_ps(mv, vm);
    out[0] = cm_hsum256(a0) + mv[0]; out[1] = cm_hsum256(a1) + mv[1];
    out[2] = cm_hsum256(a2) + mv[2]; out[3] = cm_hsum256(a3) + mv[3];
}

// 8-Token Shared-Weight Batched AVX2+FMA+F16C Q4_0_4R Gate+Up (1 DDR4 load -> up to 8 tokens! 4-6x faster 24GB Hybrid Prefill!)
static void cm_cpu_dot_q4_0_4r_gu_q16_bn(const uint8_t* __restrict__ p, uint32_t nb, uint32_t dim,
                                         const int16_t* __restrict__ xq16, const float* __restrict__ xd,
                                         const float* __restrict__ xsum, int bt, float gout[CM_CPU_MAX_B][4], float uout[CM_CPU_MAX_B][4]) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    __m256 ag[CM_CPU_MAX_B][4], au[CM_CPU_MAX_B][4];
    __m128 vbg[CM_CPU_MAX_B], vbu[CM_CPU_MAX_B];
    for (int t = 0; t < bt; t++) {
        vbg[t] = _mm_setzero_ps(); vbu[t] = _mm_setzero_ps();
        for (int r = 0; r < 4; r++) { ag[t][r] = _mm256_setzero_ps(); au[t][r] = _mm256_setzero_ps(); }
    }
    for (uint32_t b = 0; b < nb; b++, p += 144u) {
        _mm_prefetch((const char*)(p + 144u * 4u), _MM_HINT_T0);
        _mm_prefetch((const char*)(p + 144u * 4u + 64u), _MM_HINT_T0);
        __m128 vdg = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)p));
        __m128 vdu = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)(p + 72u)));
        float dg[4], du[4]; _mm_storeu_ps(dg, vdg); _mm_storeu_ps(du, vdu);
        for (int t = 0; t < bt; t++) {
            __m128 vxs = _mm_set1_ps(xsum[(size_t)t * nb + b]);
            vbg[t] = _mm_fmadd_ps(vdg, vxs, vbg[t]);
            vbu[t] = _mm_fmadd_ps(vdu, vxs, vbu[t]);
        }
        const uint8_t* qg = p + 8u;
        const uint8_t* qu = p + 80u;
        for (int r = 0; r < 4; r++) {
            __m128i qb = _mm_loadu_si128((const __m128i*)(qg + r * 16u));
            __m256i l16 = _mm256_cvtepu8_epi16(_mm_and_si128(qb, m0f));
            __m256i h16 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(qb, 4), m0f));
            for (int t = 0; t < bt; t++) {
                const int16_t* xp = xq16 + (size_t)t * dim + ((size_t)b << 5);
                __m256i s32 = _mm256_add_epi32(_mm256_madd_epi16(l16, _mm256_loadu_si256((const __m256i*)xp)),
                                               _mm256_madd_epi16(h16, _mm256_loadu_si256((const __m256i*)(xp + 16u))));
                float sc = dg[r] * xd[(size_t)t * nb + b];
                ag[t][r] = _mm256_fmadd_ps(_mm256_set1_ps(sc), _mm256_cvtepi32_ps(s32), ag[t][r]);
            }
        }
        for (int r = 0; r < 4; r++) {
            __m128i qb = _mm_loadu_si128((const __m128i*)(qu + r * 16u));
            __m256i l16 = _mm256_cvtepu8_epi16(_mm_and_si128(qb, m0f));
            __m256i h16 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(qb, 4), m0f));
            for (int t = 0; t < bt; t++) {
                const int16_t* xp = xq16 + (size_t)t * dim + ((size_t)b << 5);
                __m256i s32 = _mm256_add_epi32(_mm256_madd_epi16(l16, _mm256_loadu_si256((const __m256i*)xp)),
                                               _mm256_madd_epi16(h16, _mm256_loadu_si256((const __m256i*)(xp + 16u))));
                float sc = du[r] * xd[(size_t)t * nb + b];
                au[t][r] = _mm256_fmadd_ps(_mm256_set1_ps(sc), _mm256_cvtepi32_ps(s32), au[t][r]);
            }
        }
    }
    for (int t = 0; t < bt; t++) {
        float bg[4], bu[4]; _mm_storeu_ps(bg, vbg[t]); _mm_storeu_ps(bu, vbu[t]);
        for (int r = 0; r < 4; r++) {
            gout[t][r] = cm_hsum256(ag[t][r]) - 8.0f * bg[r];
            uout[t][r] = cm_hsum256(au[t][r]) - 8.0f * bu[r];
        }
    }
}

static void cm_cpu_dot_q4_0_4r_q16_bn(const uint8_t* __restrict__ p, uint32_t nb, uint32_t dim,
                                      const int16_t* __restrict__ hq16, const float* __restrict__ hd,
                                      const float* __restrict__ hsum, int bt, float out[CM_CPU_MAX_B][4]) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    __m256 acc[CM_CPU_MAX_B][4]; __m128 vb[CM_CPU_MAX_B];
    for (int t = 0; t < bt; t++) {
        vb[t] = _mm_setzero_ps();
        for (int r = 0; r < 4; r++) acc[t][r] = _mm256_setzero_ps();
    }
    for (uint32_t b = 0; b < nb; b++, p += 72u) {
        _mm_prefetch((const char*)(p + 72u * 8u), _MM_HINT_T0);
        __m128 vd = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)p));
        float d[4]; _mm_storeu_ps(d, vd);
        for (int t = 0; t < bt; t++) vb[t] = _mm_fmadd_ps(vd, _mm_set1_ps(hsum[(size_t)t * nb + b]), vb[t]);
        const uint8_t* q = p + 8u;
        for (int r = 0; r < 4; r++) {
            __m128i qb = _mm_loadu_si128((const __m128i*)(q + r * 16u));
            __m256i l16 = _mm256_cvtepu8_epi16(_mm_and_si128(qb, m0f));
            __m256i h16 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(qb, 4), m0f));
            for (int t = 0; t < bt; t++) {
                const int16_t* hp = hq16 + (size_t)t * dim + ((size_t)b << 5);
                __m256i s32 = _mm256_add_epi32(_mm256_madd_epi16(l16, _mm256_loadu_si256((const __m256i*)hp)),
                                               _mm256_madd_epi16(h16, _mm256_loadu_si256((const __m256i*)(hp + 16u))));
                float sc = d[r] * hd[(size_t)t * nb + b];
                acc[t][r] = _mm256_fmadd_ps(_mm256_set1_ps(sc), _mm256_cvtepi32_ps(s32), acc[t][r]);
            }
        }
    }
    for (int t = 0; t < bt; t++) {
        float bv[4]; _mm_storeu_ps(bv, vb[t]);
        for (int r = 0; r < 4; r++) out[t][r] = cm_hsum256(acc[t][r]) - 8.0f * bv[r];
    }
}

static void cm_cpu_dot_q4_1_4r_q16_bn(const uint8_t* __restrict__ p, uint32_t nb, uint32_t dim,
                                      const int16_t* __restrict__ hq16, const float* __restrict__ hd,
                                      const float* __restrict__ hsum, int bt, float out[CM_CPU_MAX_B][4]) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    __m256 acc[CM_CPU_MAX_B][4]; __m128 vm[CM_CPU_MAX_B];
    for (int t = 0; t < bt; t++) {
        vm[t] = _mm_setzero_ps();
        for (int r = 0; r < 4; r++) acc[t][r] = _mm256_setzero_ps();
    }
    for (uint32_t b = 0; b < nb; b++, p += 80u) {
        _mm_prefetch((const char*)(p + 80u * 8u), _MM_HINT_T0);
        __m128 vd = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)p));
        __m128 vw = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i*)(p + 8u)));
        float d[4]; _mm_storeu_ps(d, vd);
        for (int t = 0; t < bt; t++) vm[t] = _mm_fmadd_ps(vw, _mm_set1_ps(hsum[(size_t)t * nb + b]), vm[t]);
        const uint8_t* q = p + 16u;
        for (int r = 0; r < 4; r++) {
            __m128i qb = _mm_loadu_si128((const __m128i*)(q + r * 16u));
            __m256i l16 = _mm256_cvtepu8_epi16(_mm_and_si128(qb, m0f));
            __m256i h16 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(qb, 4), m0f));
            for (int t = 0; t < bt; t++) {
                const int16_t* hp = hq16 + (size_t)t * dim + ((size_t)b << 5);
                __m256i s32 = _mm256_add_epi32(_mm256_madd_epi16(l16, _mm256_loadu_si256((const __m256i*)hp)),
                                               _mm256_madd_epi16(h16, _mm256_loadu_si256((const __m256i*)(hp + 16u))));
                float sc = d[r] * hd[(size_t)t * nb + b];
                acc[t][r] = _mm256_fmadd_ps(_mm256_set1_ps(sc), _mm256_cvtepi32_ps(s32), acc[t][r]);
            }
        }
    }
    for (int t = 0; t < bt; t++) {
        float mv[4]; _mm_storeu_ps(mv, vm[t]);
        for (int r = 0; r < 4; r++) out[t][r] = cm_hsum256(acc[t][r]) + mv[r];
    }
}
#endif

static const int8_t cm_k_iq4_nl[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};

static float cm_cpu_dot_1row_generic(const uint8_t* w, uint32_t gtype, const float* x, uint32_t in_dim) {
    if (gtype == CM_GGML_Q4_0) {
        uint32_t nb = in_dim >> 5; float sum = 0.0f;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t* p = w + (size_t)b * 18u; const float* xp = x + ((size_t)b << 5);
            float d = cm_f16_bits_to_f32(*(const uint16_t*)p); float s = 0.0f;
            for (int i = 0; i < 16; i++) {
                uint8_t q = p[2 + i];
                s += (float)((int)(q & 0x0Fu) - 8) * xp[i] + (float)((int)(q >> 4) - 8) * xp[16 + i];
            }
            sum += d * s;
        }
        return sum;
    } else if (gtype == CM_GGML_Q4_1) {
        uint32_t nb = in_dim >> 5; float sum = 0.0f;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t* p = w + (size_t)b * 20u; const float* xp = x + ((size_t)b << 5);
            float d = cm_f16_bits_to_f32(*(const uint16_t*)p), mv = cm_f16_bits_to_f32(*(const uint16_t*)(p + 2));
            float s = 0.0f, sx = 0.0f;
            for (int i = 0; i < 16; i++) {
                uint8_t q = p[4 + i];
                s += (float)(q & 0x0Fu) * xp[i] + (float)(q >> 4) * xp[16 + i];
                sx += xp[i] + xp[16 + i];
            }
            sum += d * s + mv * sx;
        }
        return sum;
    } else if (gtype == CM_GGML_Q5_0) {
        uint32_t nb = in_dim >> 5; float sum = 0.0f;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t* p = w + (size_t)b * 22u; const float* xp = x + ((size_t)b << 5);
            float d = cm_f16_bits_to_f32(*(const uint16_t*)p);
            uint32_t qh; memcpy(&qh, p + 2, 4); float s = 0.0f;
            for (int i = 0; i < 16; i++) {
                uint8_t q = p[6 + i];
                int q0 = (int)((q & 0x0Fu) | (((qh >> i) & 1u) << 4)) - 16;
                int q1 = (int)((q >> 4) | (((qh >> (i + 16)) & 1u) << 4)) - 16;
                s += (float)q0 * xp[i] + (float)q1 * xp[16 + i];
            }
            sum += d * s;
        }
        return sum;
    } else if (gtype == CM_GGML_Q5_1) {
        uint32_t nb = in_dim >> 5; float sum = 0.0f;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t* p = w + (size_t)b * 24u; const float* xp = x + ((size_t)b << 5);
            float d = cm_f16_bits_to_f32(*(const uint16_t*)p), mv = cm_f16_bits_to_f32(*(const uint16_t*)(p + 2));
            uint32_t qh; memcpy(&qh, p + 4, 4); float s = 0.0f, sx = 0.0f;
            for (int i = 0; i < 16; i++) {
                uint8_t q = p[8 + i];
                int q0 = (int)((q & 0x0Fu) | (((qh >> i) & 1u) << 4));
                int q1 = (int)((q >> 4) | (((qh >> (i + 16)) & 1u) << 4));
                s += (float)q0 * xp[i] + (float)q1 * xp[16 + i];
                sx += xp[i] + xp[16 + i];
            }
            sum += d * s + mv * sx;
        }
        return sum;
    } else if (gtype == CM_GGML_Q8_0) {
        uint32_t nb = in_dim >> 5; float sum = 0.0f;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t* p = w + (size_t)b * 34u; const float* xp = x + ((size_t)b << 5);
            float d = cm_f16_bits_to_f32(*(const uint16_t*)p); float s = 0.0f;
            const int8_t* qs = (const int8_t*)(p + 2);
            for (int i = 0; i < 32; i++) s += (float)qs[i] * xp[i];
            sum += d * s;
        }
        return sum;
    } else if (gtype == CM_GGML_Q4_K) {
        uint32_t nb = in_dim >> 8; float sum = 0.0f;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t* blk = w + (size_t)b * 144u; const float* xp = x + ((size_t)b << 8);
            float d = cm_f16_bits_to_f32(*(const uint16_t*)blk), dmin = cm_f16_bits_to_f32(*(const uint16_t*)(blk + 2u));
            const uint8_t* sc = blk + 4u; const uint8_t* qs = blk + 16u; uint32_t is = 0u;
            for (uint32_t j = 0; j < 256u; j += 64u) {
                uint8_t sc0, m0, sc1, m1;
                if (is < 4u) { sc0 = sc[is] & 63u; m0 = sc[is+4u] & 63u; sc1 = sc[is+1u] & 63u; m1 = sc[is+5u] & 63u; }
                else { sc0 = (sc[is+4u]&0xFu)|((sc[is-4u]>>6)<<4); m0 = (sc[is+4u]>>4)|((sc[is]>>6)<<4);
                       sc1 = (sc[is+5u]&0xFu)|((sc[is-3u]>>6)<<4); m1 = (sc[is+5u]>>4)|((sc[is+1u]>>6)<<4); }
                float d1 = d*(float)sc0, dm1 = dmin*(float)m0, d2 = d*(float)sc1, dm2 = dmin*(float)m1;
                const uint8_t* q = qs + (j >> 1);
                for (uint32_t l = 0; l < 32u; l++) {
                    sum += (d1 * (float)(q[l] & 0xFu) - dm1) * xp[j + l] + (d2 * (float)(q[l] >> 4) - dm2) * xp[j + 32u + l];
                }
                is += 2u;
            }
        }
        return sum;
    } else if (gtype == CM_GGML_Q5_K) {
        uint32_t nb = in_dim >> 8; float sum = 0.0f;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t* blk = w + (size_t)b * 176u; const float* xp = x + ((size_t)b << 8);
            float d = cm_f16_bits_to_f32(*(const uint16_t*)blk), dmin = cm_f16_bits_to_f32(*(const uint16_t*)(blk + 2u));
            const uint8_t* sc = blk + 4u; const uint8_t* qh = blk + 16u; const uint8_t* qs = blk + 48u;
            uint32_t is = 0u, u1 = 1u, u2 = 2u;
            for (uint32_t j = 0; j < 256u; j += 64u) {
                uint8_t sc0, m0, sc1, m1;
                if (is < 4u) { sc0 = sc[is] & 63u; m0 = sc[is+4u] & 63u; sc1 = sc[is+1u] & 63u; m1 = sc[is+5u] & 63u; }
                else { sc0 = (sc[is+4u]&0xFu)|((sc[is-4u]>>6)<<4); m0 = (sc[is+4u]>>4)|((sc[is]>>6)<<4);
                       sc1 = (sc[is+5u]&0xFu)|((sc[is-3u]>>6)<<4); m1 = (sc[is+5u]>>4)|((sc[is+1u]>>6)<<4); }
                float d1 = d*(float)sc0, dm1 = dmin*(float)m0, d2 = d*(float)sc1, dm2 = dmin*(float)m1;
                const uint8_t* q = qs + (j >> 1);
                for (uint32_t l = 0; l < 32u; l++) {
                    float qlo = (float)((q[l] & 0xFu) + ((qh[l] & u1) ? 16u : 0u));
                    float qhi = (float)((q[l] >> 4)   + ((qh[l] & u2) ? 16u : 0u));
                    sum += (d1 * qlo - dm1) * xp[j + l] + (d2 * qhi - dm2) * xp[j + 32u + l];
                }
                is += 2u; u1 <<= 2; u2 <<= 2;
            }
        }
        return sum;
    } else if (gtype == CM_GGML_Q6_K) {
        uint32_t nb = in_dim >> 8; float sum = 0.0f;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t* blk = w + (size_t)b * 210u; const float* xp = x + ((size_t)b << 8);
            const uint8_t* ql = blk; const uint8_t* qh = blk + 128u; const int8_t* sc = (const int8_t*)(blk + 192u);
            float d = cm_f16_bits_to_f32(*(const uint16_t*)(blk + 208u)); float bsum = 0.0f;
            for (uint32_t n = 0; n < 256u; n += 128u) {
                const uint8_t* qlp = ql + (n >> 1); const uint8_t* qhp = qh + (n >> 2);
                const int8_t* scp = sc + (n >> 4); const float* xpp = xp + n;
                for (uint32_t l = 0; l < 32u; l++) {
                    uint32_t is = l >> 4; uint8_t qla = qlp[l], qlb = qlp[l + 32u], qhv = qhp[l];
                    int q1 = (int)((qla & 0xFu) | (((qhv >> 0) & 3u) << 4)) - 32;
                    int q2 = (int)((qlb & 0xFu) | (((qhv >> 2) & 3u) << 4)) - 32;
                    int q3 = (int)((qla >> 4)   | (((qhv >> 4) & 3u) << 4)) - 32;
                    int q4 = (int)((qlb >> 4)   | (((qhv >> 6) & 3u) << 4)) - 32;
                    bsum += (float)scp[is + 0] * (float)q1 * xpp[l] + (float)scp[is + 2] * (float)q2 * xpp[l + 32u]
                          + (float)scp[is + 4] * (float)q3 * xpp[l + 64u] + (float)scp[is + 6] * (float)q4 * xpp[l + 96u];
                }
            }
            sum += d * bsum;
        }
        return sum;
    } else if (gtype == CM_GGML_Q2_K) {
        uint32_t nb = in_dim >> 8; float sum = 0.0f;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t* blk = w + (size_t)b * 84u; const float* xp = x + ((size_t)b << 8);
            const uint8_t* sc = blk; const uint8_t* qs = blk + 16u;
            float d = cm_f16_bits_to_f32(*(const uint16_t*)(blk + 80u)), dm = cm_f16_bits_to_f32(*(const uint16_t*)(blk + 82u));
            for (uint32_t n = 0; n < 256u; n += 128u) {
                const uint8_t* q = qs + (n >> 2); uint32_t is_b = n >> 4;
                for (uint32_t sg = 0; sg < 4u; sg++) {
                    for (uint32_t l = 0; l < 32u; l++) {
                        uint8_t scv = sc[is_b + sg * 2u + (l >> 4)];
                        float dl = d * (float)(scv & 0xFu), ml = dm * (float)(scv >> 4);
                        sum += (dl * (float)((q[l] >> (sg * 2u)) & 3u) - ml) * xp[n + sg * 32u + l];
                    }
                }
            }
        }
        return sum;
    } else if (gtype == CM_GGML_Q3_K) {
        uint32_t nb = in_dim >> 8; float sum = 0.0f;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t* blk = w + (size_t)b * 110u; const float* xp = x + ((size_t)b << 8);
            const uint8_t* hm = blk; const uint8_t* qs = blk + 32u; const uint8_t* scb = blk + 96u;
            float d = cm_f16_bits_to_f32(*(const uint16_t*)(blk + 108u));
            uint32_t a0 = (uint32_t)scb[0] | ((uint32_t)scb[1]<<8) | ((uint32_t)scb[2]<<16) | ((uint32_t)scb[3]<<24);
            uint32_t a1 = (uint32_t)scb[4] | ((uint32_t)scb[5]<<8) | ((uint32_t)scb[6]<<16) | ((uint32_t)scb[7]<<24);
            uint32_t a2 = (uint32_t)scb[8] | ((uint32_t)scb[9]<<8) | ((uint32_t)scb[10]<<16) | ((uint32_t)scb[11]<<24);
            uint32_t scu[4] = {(a0&0x0F0F0F0Fu)|(((a2>>0)&0x03030303u)<<4), (a1&0x0F0F0F0Fu)|(((a2>>2)&0x03030303u)<<4),
                               ((a0>>4)&0x0F0F0F0Fu)|(((a2>>4)&0x03030303u)<<4), ((a1>>4)&0x0F0F0F0Fu)|(((a2>>6)&0x03030303u)<<4)};
            for (uint32_t n = 0; n < 256u; n += 128u) {
                const uint8_t* q = qs + (n >> 2); uint32_t is_b = n >> 4;
                for (uint32_t sg = 0; sg < 4u; sg++) {
                    uint32_t m = 1u << ((n >> 7) * 4u + sg);
                    for (uint32_t l = 0; l < 32u; l++) {
                        uint32_t is = is_b + sg * 2u + (l >> 4);
                        int scv = (int)((scu[is >> 2] >> ((is & 3u) * 8u)) & 0xFFu) - 32;
                        int q3 = (int)((q[l] >> (sg * 2u)) & 3u) - ((hm[l] & m) ? 0 : 4);
                        sum += d * (float)scv * (float)q3 * xp[n + sg * 32u + l];
                    }
                }
            }
        }
        return sum;
    } else if (gtype == CM_GGML_IQ4_XS) {
        uint32_t nb = in_dim >> 8; float sum = 0.0f;
        for (uint32_t b = 0; b < nb; b++) {
            const uint8_t* blk = w + (size_t)b * 136u; const float* xp = x + ((size_t)b << 8);
            float d = cm_f16_bits_to_f32(*(const uint16_t*)blk);
            uint16_t sch = *(const uint16_t*)(blk + 2u);
            const uint8_t* scl = blk + 4u; const uint8_t* qs = blk + 8u;
            for (uint32_t ib = 0; ib < 8u; ib++) {
                uint32_t ls = ((scl[ib >> 1] >> ((ib & 1u) * 4u)) & 0xFu) | (((uint32_t)(sch >> (ib * 2u)) & 3u) << 4);
                float dl = d * (float)((int)ls - 32);
                for (uint32_t i = 0; i < 16u; i++) {
                    uint8_t bv = qs[ib * 16u + i];
                    sum += dl * ((float)cm_k_iq4_nl[bv & 0xFu] * xp[ib * 32u + i] + (float)cm_k_iq4_nl[bv >> 4] * xp[ib * 32u + 16u + i]);
                }
            }
        }
        return sum;
    } else if (gtype == CM_GGML_F16) {
        const uint16_t* p = (const uint16_t*)w; float s = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) s += cm_f16_bits_to_f32(p[i]) * x[i];
        return s;
    } else {
        const float* p = (const float*)w; float s = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) s += p[i] * x[i];
        return s;
    }
}

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
// AVX2 FMA+F16C Q4_K 2-row dot kernel (144B/256w -> ~4x faster than scalar on Q4_K_M CPU offload!)
static void cm_cpu_dot_q4_k_avx2_2rows(const uint8_t* __restrict__ w0, const uint8_t* __restrict__ w1,
                                       const float* __restrict__ x, uint32_t nb, float* o0, float* o1) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (uint32_t b = 0; b < nb; b++) {
        const float* xp = x + ((size_t)b << 8);
        const uint8_t* blks[2] = { w0 + (size_t)b * 144u, w1 + (size_t)b * 144u };
        __m256* accs[2] = { &acc0, &acc1 };
        for (int r = 0; r < 2; r++) {
            const uint8_t* blk = blks[r];
            float d = _cvtsh_ss(*(const uint16_t*)blk), dmin = _cvtsh_ss(*(const uint16_t*)(blk + 2u));
            const uint8_t* sc = blk + 4u; const uint8_t* qs = blk + 16u; uint32_t is = 0u;
            for (uint32_t j = 0; j < 256u; j += 64u) {
                uint8_t sc0, m0, sc1, m1;
                if (is < 4u) { sc0 = sc[is] & 63u; m0 = sc[is+4u] & 63u; sc1 = sc[is+1u] & 63u; m1 = sc[is+5u] & 63u; }
                else { sc0 = (sc[is+4u]&0xFu)|((sc[is-4u]>>6)<<4); m0 = (sc[is+4u]>>4)|((sc[is]>>6)<<4);
                       sc1 = (sc[is+5u]&0xFu)|((sc[is-3u]>>6)<<4); m1 = (sc[is+5u]>>4)|((sc[is+1u]>>6)<<4); }
                __m256 vd1 = _mm256_set1_ps(d * (float)sc0), vm1 = _mm256_set1_ps(-dmin * (float)m0);
                __m256 vd2 = _mm256_set1_ps(d * (float)sc1), vm2 = _mm256_set1_ps(-dmin * (float)m1);
                const uint8_t* q = qs + (j >> 1);
                for (uint32_t off = 0; off < 32u; off += 16u) {
                    __m128i qb = _mm_loadu_si128((const __m128i*)(q + off));
                    __m128i ql = _mm_and_si128(qb, m0f);
                    __m128i qh = _mm_and_si128(_mm_srli_epi16(qb, 4), m0f);
                    __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(ql));
                    __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(ql, 8)));
                    __m256 g0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(qh));
                    __m256 g1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(qh, 8)));
                    *accs[r] = _mm256_fmadd_ps(_mm256_fmadd_ps(vd1, f0, vm1), _mm256_loadu_ps(xp + j + off), *accs[r]);
                    *accs[r] = _mm256_fmadd_ps(_mm256_fmadd_ps(vd1, f1, vm1), _mm256_loadu_ps(xp + j + off + 8u), *accs[r]);
                    *accs[r] = _mm256_fmadd_ps(_mm256_fmadd_ps(vd2, g0, vm2), _mm256_loadu_ps(xp + j + 32u + off), *accs[r]);
                    *accs[r] = _mm256_fmadd_ps(_mm256_fmadd_ps(vd2, g1, vm2), _mm256_loadu_ps(xp + j + 32u + off + 8u), *accs[r]);
                }
                is += 2u;
            }
        }
    }
    *o0 = cm_hsum256(acc0); *o1 = cm_hsum256(acc1);
}
#endif

static inline void cm_cpu_dot_2rows(const uint8_t* w0, const uint8_t* w1, uint32_t gtype,
                                    const float* x, const int8_t* xq8, const float* xd, const float* xsum,
                                    uint32_t in_dim, float* o0, float* o1) {
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    if (gtype == CM_GGML_Q4_K && (in_dim & 255u) == 0u) { cm_cpu_dot_q4_k_avx2_2rows(w0, w1, x, in_dim >> 8, o0, o1); return; }
    if (in_dim > 8192u) {
        if (gtype == CM_GGML_Q4_0) { cm_cpu_dot_q4_0_fp32_2rows(w0, w1, x, in_dim >> 5, o0, o1); return; }
        if (gtype == CM_GGML_Q4_1) { cm_cpu_dot_q4_1_fp32_2rows(w0, w1, x, in_dim >> 5, o0, o1); return; }
    } else {
        if (gtype == CM_GGML_Q4_0) { cm_cpu_dot_q4_0_q8_2rows(w0, w1, xq8, xd, xsum, in_dim >> 5, o0, o1); return; }
        if (gtype == CM_GGML_Q4_1) { cm_cpu_dot_q4_1_q8_2rows(w0, w1, xq8, xd, xsum, in_dim >> 5, o0, o1); return; }
    }
#endif
    (void)xq8; (void)xd; (void)xsum;
    *o0 = cm_cpu_dot_1row_generic(w0, gtype, x, in_dim);
    *o1 = cm_cpu_dot_1row_generic(w1, gtype, x, in_dim);
}

static inline void cm_cpu_dot_4rows(const uint8_t* w0, uint64_t rb, uint32_t gtype,
                                    const float* x, const int8_t* xq8, const float* xd, const float* xsum,
                                    uint32_t in_dim, float out[4]) {
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    if (gtype == CM_GGML_Q4_0 && in_dim <= 8192u) { cm_cpu_dot_q4_0_q8_4rows(w0, rb, xq8, xd, xsum, in_dim >> 5, out); return; }
#endif
    cm_cpu_dot_2rows(w0, w0 + rb, gtype, x, xq8, xd, xsum, in_dim, &out[0], &out[1]);
    cm_cpu_dot_2rows(w0 + rb * 2u, w0 + rb * 3u, gtype, x, xq8, xd, xsum, in_dim, &out[2], &out[3]);
}

typedef struct { CmModel* m; int tid; } CpuPoolArg;

static void cm_cpu_exec_task(CmModel* m, int tid) {
    int n_thr = m->cpu_n_thr;
    int mode = m->cpu_mode;
    int L = m->cpu_layer;
    CmLayer* ly = &m->layers[L];
    const float* xn = m->h_xn;
    const int8_t* xq8 = m->cpu_xq8;
    const float* xd = m->cpu_xd;
    const float* xsum = m->cpu_xsum;
    uint32_t dm = (uint32_t)m->d_model;

    if (mode == 1) {
            // SPARSE MoE IN-PLACE AVX2 INT8 `vpmaddubsw` COMPUTE FOR `n_misses` MISSED EXPERTS!
            int nm = m->cpu_n_misses;
            uint32_t df = (uint32_t)m->d_ff_exp;
            uint32_t gt = ly->gate_exps.ggml_type, ut = ly->up_exps.ggml_type, dt = ly->down_exps.ggml_type;
            uint64_t grb = cm_ggml_type_bytes(gt, dm), urb = cm_ggml_type_bytes(ut, dm), drb = cm_ggml_type_bytes(dt, df);
            const uint8_t* g_base = m->mmap_ptrs[ly->gate_exps.shard_idx] + ly->gate_exps.file_off;
            const uint8_t* u_base = m->mmap_ptrs[ly->up_exps.shard_idx]   + ly->up_exps.file_off;
            const uint8_t* d_base = m->mmap_ptrs[ly->down_exps.shard_idx] + ly->down_exps.file_off;

            // Phase 1: parallel over 32-row chunks (so each thread quantizes its own Q8_0 blocks of cpu_h!)
            uint32_t blks_per_exp = df >> 5;
            uint32_t tot_blks = (uint32_t)nm * blks_per_exp;
            uint32_t b0 = (uint32_t)(((uint64_t)tot_blks * (uint64_t)tid) / (uint64_t)n_thr);
            uint32_t b1 = (uint32_t)(((uint64_t)tot_blks * (uint64_t)(tid + 1)) / (uint64_t)n_thr);
            for (uint32_t bi = b0; bi < b1; bi++) {
                int mi = (int)(bi / blks_per_exp);
                uint32_t rb_idx = bi % blks_per_exp;
                uint32_t r0 = rb_idx << 5;
                int eid = m->cpu_miss_eids[mi];
                float* hp = m->cpu_h + (size_t)mi * df + r0;
                const uint8_t* gp = g_base + (uint64_t)eid * ly->gate_exp_bytes + (uint64_t)r0 * grb;
                const uint8_t* up = u_base + (uint64_t)eid * ly->up_exp_bytes   + (uint64_t)r0 * urb;
                for (uint32_t k = 0; k < 32u; k += 4u) {
                    float gv[4], uv[4];
                    cm_cpu_dot_4rows(gp + (uint64_t)k * grb, grb, gt, xn, xq8, xd, xsum, dm, gv);
                    cm_cpu_dot_4rows(up + (uint64_t)k * urb, urb, ut, xn, xq8, xd, xsum, dm, uv);
                    for (int j = 0; j < 4; j++) hp[k + (uint32_t)j] = (gv[j] / (1.0f + expf(-gv[j]))) * uv[j];
                }
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
                cm_quantize_q8_0(hp, m->cpu_hq8 + (size_t)bi * 32u, &m->cpu_hd[bi], &m->cpu_hsum[bi], 32u);
#endif
            }

            // Lock-free spin barrier between Phase 1 and Phase 2
            __atomic_fetch_add(&m->cpu_phase1_done, 1, __ATOMIC_ACQ_REL);
            while (__atomic_load_n(&m->cpu_phase1_done, __ATOMIC_ACQUIRE) < n_thr) cm_cpu_pause();

            // Phase 2: parallel over 4-row groups of output dimension d_model!
            uint32_t dm_quads = dm >> 2;
            uint32_t d0 = (uint32_t)(((uint64_t)dm_quads * (uint64_t)tid) / (uint64_t)n_thr);
            uint32_t d1 = (uint32_t)(((uint64_t)dm_quads * (uint64_t)(tid + 1)) / (uint64_t)n_thr);
            for (uint32_t dq = d0; dq < d1; dq++) {
                uint32_t r = dq << 2;
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (int mi = 0; mi < nm; mi++) {
                    int eid = m->cpu_miss_eids[mi];
                    const uint8_t* dwp = d_base + (uint64_t)eid * ly->down_exp_bytes + (uint64_t)r * drb;
                    float ov[4];
                    size_t hb = (size_t)mi * blks_per_exp;
                    cm_cpu_dot_4rows(dwp, drb, dt, m->cpu_h + (size_t)mi * df,
                                     m->cpu_hq8 + hb * 32u, m->cpu_hd + hb, m->cpu_hsum + hb, df, ov);
                    float w = m->cpu_miss_w[mi];
                    s0 += w * ov[0]; s1 += w * ov[1]; s2 += w * ov[2]; s3 += w * ov[3];
                }
                m->h_cpu_accum[r + 0] = s0; m->h_cpu_accum[r + 1] = s1;
                m->h_cpu_accum[r + 2] = s2; m->h_cpu_accum[r + 3] = s3;
            }
        } else if (mode == 2) {
            // DENSE UNPINNED FFN: NUMA-LOCAL CONTIGUOUS `Q4_0_4R` + INT16 `_mm256_madd_epi16` (200+ GB/s & BIT-EXACT!)
            uint32_t df = (uint32_t)m->d_ff_dense;
            uint32_t gt = ly->ffn_gate.ggml_type, ut = ly->ffn_up.ggml_type, dt = ly->ffn_down.ggml_type;
            uint32_t df_blks = df >> 5;
            uint32_t b0 = (uint32_t)(((uint64_t)df_blks * (uint64_t)tid) / (uint64_t)n_thr);
            uint32_t b1 = (uint32_t)(((uint64_t)df_blks * (uint64_t)(tid + 1)) / (uint64_t)n_thr);

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
            if (ly->repack_gu != NULL) {
                uint32_t nb_dm = dm >> 5;
                size_t grp_stride = (size_t)nb_dm * 144u;
                for (uint32_t bi = b0; bi < b1; bi++) {
                    uint32_t r0 = bi << 5;
                    float* hp = m->cpu_h + r0;
                    for (uint32_t k = 0; k < 32u; k += 4u) {
                        uint32_t rg = (r0 + k) >> 2;
                        float gv[4], uv[4];
                        cm_cpu_dot_q4_0_4r_gu_q16(ly->repack_gu + (size_t)rg * grp_stride, nb_dm, m->cpu_xq16, xd, xsum, gv, uv);
                        for (int j = 0; j < 4; j++) hp[k + (uint32_t)j] = (gv[j] / (1.0f + expf(-gv[j]))) * uv[j];
                    }
                    cm_quantize_q16_0(hp, m->cpu_hq16 + (size_t)bi * 32u, &m->cpu_hd[bi], &m->cpu_hsum[bi], 32u);
                }
            } else
#endif
            {
                uint64_t grb = cm_ggml_type_bytes(gt, dm), urb = cm_ggml_type_bytes(ut, dm);
                const uint8_t* g_base = m->mmap_ptrs[ly->ffn_gate.shard_idx] + ly->ffn_gate.file_off;
                const uint8_t* u_base = m->mmap_ptrs[ly->ffn_up.shard_idx]   + ly->ffn_up.file_off;
                for (uint32_t bi = b0; bi < b1; bi++) {
                    uint32_t r0 = bi << 5;
                    float* hp = m->cpu_h + r0;
                    const uint8_t* gp = g_base + (uint64_t)r0 * grb;
                    const uint8_t* up = u_base + (uint64_t)r0 * urb;
                    for (uint32_t k = 0; k < 32u; k += 4u) {
                        float gv[4], uv[4];
                        cm_cpu_dot_4rows(gp + (uint64_t)k * grb, grb, gt, xn, xq8, xd, xsum, dm, gv);
                        cm_cpu_dot_4rows(up + (uint64_t)k * urb, urb, ut, xn, xq8, xd, xsum, dm, uv);
                        for (int j = 0; j < 4; j++) hp[k + (uint32_t)j] = (gv[j] / (1.0f + expf(-gv[j]))) * uv[j];
                    }
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
                    cm_quantize_q16_0(hp, m->cpu_hq16 + (size_t)bi * 32u, &m->cpu_hd[bi], &m->cpu_hsum[bi], 32u);
#endif
                }
            }

            __atomic_fetch_add(&m->cpu_phase1_done, 1, __ATOMIC_ACQ_REL);
            while (__atomic_load_n(&m->cpu_phase1_done, __ATOMIC_ACQUIRE) < n_thr) cm_cpu_pause();

            uint32_t dm_quads = dm >> 2;
            uint32_t d0 = (uint32_t)(((uint64_t)dm_quads * (uint64_t)tid) / (uint64_t)n_thr);
            uint32_t d1 = (uint32_t)(((uint64_t)dm_quads * (uint64_t)(tid + 1)) / (uint64_t)n_thr);
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
            if (ly->repack_down != NULL) {
                uint32_t nb_df = df >> 5;
                size_t blk_b = (dt == CM_GGML_Q4_1) ? 80u : 72u;
                size_t grp_stride = (size_t)nb_df * blk_b;
                for (uint32_t dq = d0; dq < d1; dq++) {
                    float ov[4];
                    if (dt == CM_GGML_Q4_1) cm_cpu_dot_q4_1_4r_q16(ly->repack_down + (size_t)dq * grp_stride, nb_df, m->cpu_hq16, m->cpu_hd, m->cpu_hsum, ov);
                    else                    cm_cpu_dot_q4_0_4r_q16(ly->repack_down + (size_t)dq * grp_stride, nb_df, m->cpu_hq16, m->cpu_hd, m->cpu_hsum, ov);
                    uint32_t r = dq << 2;
                    m->h_cpu_accum[r + 0] = ov[0]; m->h_cpu_accum[r + 1] = ov[1];
                    m->h_cpu_accum[r + 2] = ov[2]; m->h_cpu_accum[r + 3] = ov[3];
                }
            } else
#endif
            {
                uint64_t drb = cm_ggml_type_bytes(dt, df);
                const uint8_t* d_base = m->mmap_ptrs[ly->ffn_down.shard_idx] + ly->ffn_down.file_off;
                for (uint32_t dq = d0; dq < d1; dq++) {
                    uint32_t r = dq << 2;
                    float ov[4];
                    cm_cpu_dot_4rows(d_base + (uint64_t)r * drb, drb, dt, m->cpu_h,
                                     m->cpu_hq8, m->cpu_hd, m->cpu_hsum, df, ov);
                    m->h_cpu_accum[r + 0] = ov[0]; m->h_cpu_accum[r + 1] = ov[1];
                    m->h_cpu_accum[r + 2] = ov[2]; m->h_cpu_accum[r + 3] = ov[3];
                }
            }
        } else if (mode == 3) {
            // PARALLEL NUMA-LOCAL FIRST-TOUCH REPACKING OF UNPINNED DENSE FFN LAYER `L`!
            uint32_t df = (uint32_t)m->d_ff_dense;
            uint32_t nb_dm = dm >> 5, nb_df = df >> 5;
            if (ly->repack_gu != NULL) {
                uint64_t grb = (uint64_t)nb_dm * 18ull;
                const uint8_t* g_base = m->mmap_ptrs[ly->ffn_gate.shard_idx] + ly->ffn_gate.file_off;
                const uint8_t* u_base = m->mmap_ptrs[ly->ffn_up.shard_idx]   + ly->ffn_up.file_off;
                uint32_t df_blks = df >> 5;
                uint32_t b0 = (uint32_t)(((uint64_t)df_blks * (uint64_t)tid) / (uint64_t)n_thr);
                uint32_t b1 = (uint32_t)(((uint64_t)df_blks * (uint64_t)(tid + 1)) / (uint64_t)n_thr);
                size_t grp_stride = (size_t)nb_dm * 144u;
                for (uint32_t bi = b0; bi < b1; bi++) {
                    uint32_t r0 = bi << 5;
                    for (uint32_t k = 0; k < 32u; k += 4u) {
                        uint32_t rg = (r0 + k) >> 2;
                        uint8_t* dst = ly->repack_gu + (size_t)rg * grp_stride;
                        for (uint32_t b = 0; b < nb_dm; b++, dst += 144u) {
                            uint16_t* dg = (uint16_t*)dst;
                            uint8_t*  qg = dst + 8u;
                            uint16_t* du = (uint16_t*)(dst + 72u);
                            uint8_t*  qu = dst + 80u;
                            for (uint32_t j = 0; j < 4u; j++) {
                                const uint8_t* gp = g_base + (uint64_t)(r0 + k + j) * grb + (uint64_t)b * 18ull;
                                const uint8_t* up = u_base + (uint64_t)(r0 + k + j) * grb + (uint64_t)b * 18ull;
                                dg[j] = *(const uint16_t*)gp;
                                memcpy(qg + j * 16u, gp + 2u, 16u);
                                du[j] = *(const uint16_t*)up;
                                memcpy(qu + j * 16u, up + 2u, 16u);
                            }
                        }
                    }
                }
            }
            if (ly->repack_down != NULL) {
                uint32_t dt = ly->ffn_down.ggml_type;
                uint64_t drb = (uint64_t)nb_df * (dt == CM_GGML_Q4_1 ? 20ull : 18ull);
                const uint8_t* d_base = m->mmap_ptrs[ly->ffn_down.shard_idx] + ly->ffn_down.file_off;
                uint32_t dm_quads = dm >> 2;
                uint32_t d0 = (uint32_t)(((uint64_t)dm_quads * (uint64_t)tid) / (uint64_t)n_thr);
                uint32_t d1 = (uint32_t)(((uint64_t)dm_quads * (uint64_t)(tid + 1)) / (uint64_t)n_thr);
                if (dt == CM_GGML_Q4_1) {
                    size_t grp_stride = (size_t)nb_df * 80u;
                    for (uint32_t dq = d0; dq < d1; dq++) {
                        uint32_t r = dq << 2;
                        uint8_t* dst = ly->repack_down + (size_t)dq * grp_stride;
                        for (uint32_t b = 0; b < nb_df; b++, dst += 80u) {
                            uint16_t* dd = (uint16_t*)dst;
                            uint16_t* dm_arr = (uint16_t*)(dst + 8u);
                            uint8_t*  qq = dst + 16u;
                            for (uint32_t j = 0; j < 4u; j++) {
                                const uint8_t* dp = d_base + (uint64_t)(r + j) * drb + (uint64_t)b * 20ull;
                                dd[j] = *(const uint16_t*)dp;
                                dm_arr[j] = *(const uint16_t*)(dp + 2u);
                                memcpy(qq + j * 16u, dp + 4u, 16u);
                            }
                        }
                    }
                } else {
                    size_t grp_stride = (size_t)nb_df * 72u;
                    for (uint32_t dq = d0; dq < d1; dq++) {
                        uint32_t r = dq << 2;
                        uint8_t* dst = ly->repack_down + (size_t)dq * grp_stride;
                        for (uint32_t b = 0; b < nb_df; b++, dst += 72u) {
                            uint16_t* dd = (uint16_t*)dst;
                            uint8_t*  qq = dst + 8u;
                            for (uint32_t j = 0; j < 4u; j++) {
                                const uint8_t* dp = d_base + (uint64_t)(r + j) * drb + (uint64_t)b * 18ull;
                                dd[j] = *(const uint16_t*)dp;
                                memcpy(qq + j * 16u, dp + 2u, 16u);
                            }
                        }
                    }
                }
            }
        } else if (mode == 4) {
            // 24GB HYBRID 8-TOKEN SHARED-WEIGHT BATCHED CPU FFN (1 DDR4 pass -> up to 8 tokens -> 4-6x faster Prefill!)
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
            int bt = m->cpu_batch_b;
            size_t boff = m->cpu_batch_off;
            uint32_t df = (uint32_t)m->d_ff_dense, dt = ly->ffn_down.ggml_type;
            uint32_t df_blks = df >> 5, nb_dm = dm >> 5, nb_df = df >> 5;
            uint32_t b0 = (uint32_t)(((uint64_t)df_blks * (uint64_t)tid) / (uint64_t)n_thr);
            uint32_t b1 = (uint32_t)(((uint64_t)df_blks * (uint64_t)(tid + 1)) / (uint64_t)n_thr);
            size_t gu_stride = (size_t)nb_dm * 144u;
            for (uint32_t bi = b0; bi < b1; bi++) {
                uint32_t r0 = bi << 5;
                for (uint32_t k = 0; k < 32u; k += 4u) {
                    uint32_t rg = (r0 + k) >> 2;
                    float gv[CM_CPU_MAX_B][4], uv[CM_CPU_MAX_B][4];
                    cm_cpu_dot_q4_0_4r_gu_q16_bn(ly->repack_gu + (size_t)rg * gu_stride, nb_dm, dm,
                                                 m->cpu_xq16, m->cpu_xd, m->cpu_xsum, bt, gv, uv);
                    for (int t = 0; t < bt; t++) {
                        float* hp = m->cpu_h + (size_t)t * df + r0 + k;
                        for (int j = 0; j < 4; j++) hp[j] = (gv[t][j] / (1.0f + expf(-gv[t][j]))) * uv[t][j];
                    }
                }
                for (int t = 0; t < bt; t++) {
                    cm_quantize_q16_0(m->cpu_h + (size_t)t * df + r0,
                                      m->cpu_hq16 + (size_t)t * df + r0,
                                      &m->cpu_hd[(size_t)t * nb_df + bi],
                                      &m->cpu_hsum[(size_t)t * nb_df + bi], 32u);
                }
            }
            __atomic_fetch_add(&m->cpu_phase1_done, 1, __ATOMIC_ACQ_REL);
            while (__atomic_load_n(&m->cpu_phase1_done, __ATOMIC_ACQUIRE) < n_thr) cm_cpu_pause();

            uint32_t dm_quads = dm >> 2;
            uint32_t d0 = (uint32_t)(((uint64_t)dm_quads * (uint64_t)tid) / (uint64_t)n_thr);
            uint32_t d1 = (uint32_t)(((uint64_t)dm_quads * (uint64_t)(tid + 1)) / (uint64_t)n_thr);
            size_t blk_b = (dt == CM_GGML_Q4_1) ? 80u : 72u;
            size_t dn_stride = (size_t)nb_df * blk_b;
            for (uint32_t dq = d0; dq < d1; dq++) {
                float ov[CM_CPU_MAX_B][4];
                if (dt == CM_GGML_Q4_1) cm_cpu_dot_q4_1_4r_q16_bn(ly->repack_down + (size_t)dq * dn_stride, nb_df, df, m->cpu_hq16, m->cpu_hd, m->cpu_hsum, bt, ov);
                else                    cm_cpu_dot_q4_0_4r_q16_bn(ly->repack_down + (size_t)dq * dn_stride, nb_df, df, m->cpu_hq16, m->cpu_hd, m->cpu_hsum, bt, ov);
                uint32_t r = dq << 2;
                for (int t = 0; t < bt; t++) {
                    float* dst = m->h_pcpu_accum + (boff + (size_t)t) * dm + r;
                    dst[0] = ov[t][0]; dst[1] = ov[t][1]; dst[2] = ov[t][2]; dst[3] = ov[t][3];
                }
            }
#endif
        }

        __atomic_fetch_add(&m->cpu_phase2_done, 1, __ATOMIC_ACQ_REL);
}

static void cm_pin_thread_numa(int tid, int n_thr) {
#if defined(__linux__)
    cpu_set_t avail;
    CPU_ZERO(&avail);
    if (sched_getaffinity(0, sizeof(avail), &avail) == 0) {
        int cpus[1024], nc = 0;
        for (int c = 0; c < CPU_SETSIZE && nc < 1024; c++) {
            if (CPU_ISSET(c, &avail)) cpus[nc++] = c;
        }
        if (nc >= n_thr && n_thr > 0) {
            int pick = cpus[(tid * nc) / n_thr];
            cpu_set_t one; CPU_ZERO(&one); CPU_SET(pick, &one);
            pthread_setaffinity_np(pthread_self(), sizeof(one), &one);
        }
    }
#else
    (void)tid; (void)n_thr;
#endif
}

static void* cm_cpu_pool_worker(void* arg) {
    CpuPoolArg* a = (CpuPoolArg*)arg;
    CmModel* m = a->m;
    int tid = a->tid;
    free(a);
    cm_pin_thread_numa(tid, m->cpu_n_thr);
    uint64_t seen_gen = 0;
    for (;;) {
        uint64_t g = __atomic_load_n(&m->cpu_gen, __ATOMIC_ACQUIRE);
        if (g == seen_gen && !__atomic_load_n(&m->cpu_stop, __ATOMIC_ACQUIRE)) {
            for (int sp = 0; sp < 15000; sp++) {
                cm_cpu_pause();
                g = __atomic_load_n(&m->cpu_gen, __ATOMIC_ACQUIRE);
                if (g != seen_gen || __atomic_load_n(&m->cpu_stop, __ATOMIC_ACQUIRE)) break;
            }
        }
        if (g == seen_gen && !__atomic_load_n(&m->cpu_stop, __ATOMIC_ACQUIRE)) {
            pthread_mutex_lock(&m->cpu_mu);
            __atomic_fetch_add(&m->cpu_sleeping, 1, __ATOMIC_ACQ_REL);
            while (!__atomic_load_n(&m->cpu_stop, __ATOMIC_ACQUIRE) &&
                   __atomic_load_n(&m->cpu_gen, __ATOMIC_ACQUIRE) == seen_gen) {
                pthread_cond_wait(&m->cpu_cv_go, &m->cpu_mu);
            }
            __atomic_fetch_sub(&m->cpu_sleeping, 1, __ATOMIC_ACQ_REL);
            pthread_mutex_unlock(&m->cpu_mu);
            g = __atomic_load_n(&m->cpu_gen, __ATOMIC_ACQUIRE);
        }
        if (__atomic_load_n(&m->cpu_stop, __ATOMIC_ACQUIRE)) break;
        seen_gen = g;
        cm_cpu_exec_task(m, tid);
    }
    return NULL;
}

// Dedicated background promoter thread: stages hot expert + fires cuMemcpyHtoDAsync completely OFF main thread!
static void* cm_promo_worker(void* arg) {
    CmModel* m = (CmModel*)arg;
    g_cu.cuCtxSetCurrent(g_cu.ctx);
    for (;;) {
        pthread_mutex_lock(&m->promo_mu);
        while (!m->promo_stop && __atomic_load_n(&m->promo_state, __ATOMIC_ACQUIRE) != 1) {
            pthread_cond_wait(&m->promo_cv, &m->promo_mu);
        }
        int stop = m->promo_stop;
        int L = m->promo_layer, vs = m->promo_slot, eid = m->promo_eid;
        pthread_mutex_unlock(&m->promo_mu);
        if (stop) break;

        CmLayer* ly = &m->layers[L];
        cm_stage_expert(m, L, eid, m->promo_stage);
        g_cu.cuMemcpyHtoDAsync(ly->cache_dptr + (uint64_t)vs * ly->slot_stride_bytes,
                               m->promo_stage, (size_t)ly->slot_stride_bytes, m->dma_stream);
        g_cu.cuEventRecord(m->ev_promo_done, m->dma_stream);
        __sync_fetch_and_add(&m->pcie_h2d_bytes, (int64_t)ly->slot_stride_bytes);
        __atomic_store_n(&m->promo_state, 2, __ATOMIC_RELEASE);
    }
    return NULL;
}

static void cm_promo_start(CmModel* m) {
    if (m->promo_thread_init) return;
    pthread_mutex_init(&m->promo_mu, NULL);
    pthread_cond_init(&m->promo_cv, NULL);
    m->promo_stop = 0; m->promo_state = 0; m->promo_thread_init = 1;
    pthread_create(&m->promo_thread, NULL, cm_promo_worker, m);
}

static void cm_promo_stop(CmModel* m) {
    if (!m->promo_thread_init) return;
    pthread_mutex_lock(&m->promo_mu);
    m->promo_stop = 1;
    pthread_cond_signal(&m->promo_cv);
    pthread_mutex_unlock(&m->promo_mu);
    pthread_join(m->promo_thread, NULL);
    pthread_mutex_destroy(&m->promo_mu);
    pthread_cond_destroy(&m->promo_cv);
    m->promo_thread_init = 0;
}

static int cm_detect_cgroup_vcpus(void) {
#if defined(__linux__)
    FILE* f = fopen("/sys/fs/cgroup/cpu.max", "r");
    if (f) {
        char qbuf[64] = {0}; long per = 0;
        if (fscanf(f, "%63s %ld", qbuf, &per) == 2 && strcmp(qbuf, "max") != 0 && per > 0) {
            long q = atol(qbuf);
            fclose(f);
            if (q > 0) return (int)((q + per - 1) / per);
        } else fclose(f);
    }
    FILE* fq = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r");
    FILE* fp = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r");
    if (fq && fp) {
        long q = -1, per = 0;
        if (fscanf(fq, "%ld", &q) == 1 && fscanf(fp, "%ld", &per) == 1 && q > 0 && per > 0) {
            fclose(fq); fclose(fp);
            return (int)((q + per - 1) / per);
        }
    }
    if (fq) fclose(fq);
    if (fp) fclose(fp);
#endif
    return 0;
}

static void cm_cpu_pool_start(CmModel* m) {
    if (m->cpu_pool_init) return;
    int nt = 32;
    int cg = cm_detect_cgroup_vcpus();
    if (cg > 0 && cg < nt) nt = cg;
    const char* env_t = getenv("TENON_CPU_THREADS");
    if (env_t && atoi(env_t) > 0) nt = atoi(env_t);
    long nproc = sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc > 0 && nt > (int)nproc) nt = (int)nproc;
    if (nt > CM_MAX_CPU_THR) nt = CM_MAX_CPU_THR;
    if (nt < 1) nt = 1;
    m->cpu_n_thr = nt;
    pthread_mutex_init(&m->cpu_mu, NULL);
    pthread_cond_init(&m->cpu_cv_go, NULL);
    m->cpu_stop = 0; m->cpu_sleeping = 0; m->cpu_gen = 0; m->cpu_pool_init = 1;
    // Main thread executes tid=0 (exact N active threads for N vCPUs -> ZERO CFS quota oversubscription!)
    for (int t = 1; t < nt; t++) {
        CpuPoolArg* a = (CpuPoolArg*)malloc(sizeof(CpuPoolArg));
        a->m = m; a->tid = t;
        pthread_create(&m->cpu_threads[t], NULL, cm_cpu_pool_worker, a);
    }
}

static void cm_cpu_pool_stop(CmModel* m) {
    if (!m->cpu_pool_init) return;
    pthread_mutex_lock(&m->cpu_mu);
    __atomic_store_n(&m->cpu_stop, 1, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&m->cpu_cv_go);
    pthread_mutex_unlock(&m->cpu_mu);
    for (int t = 1; t < m->cpu_n_thr; t++) pthread_join(m->cpu_threads[t], NULL);
    pthread_mutex_destroy(&m->cpu_mu);
    pthread_cond_destroy(&m->cpu_cv_go);
    m->cpu_pool_init = 0;
}

static void cm_cpu_pool_run(CmModel* m, int mode, int L, int n_misses, const int* eids, const float* ws) {
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    if (mode == 2 && m->layers[L].repack_gu != NULL) {
        cm_quantize_q16_0(m->h_xn, m->cpu_xq16, m->cpu_xd, m->cpu_xsum, (uint32_t)m->d_model);
    } else if (mode == 4) {
        uint32_t dm = (uint32_t)m->d_model, nb = dm >> 5;
        for (int t = 0; t < m->cpu_batch_b; t++) {
            cm_quantize_q16_0(m->h_pxn + (m->cpu_batch_off + (size_t)t) * dm,
                              m->cpu_xq16 + (size_t)t * dm,
                              m->cpu_xd + (size_t)t * nb,
                              m->cpu_xsum + (size_t)t * nb, dm);
        }
    } else if (mode != 3) {
        cm_quantize_q8_0(m->h_xn, m->cpu_xq8, m->cpu_xd, m->cpu_xsum, (uint32_t)m->d_model);
    }
#endif
    m->cpu_mode = mode;
    m->cpu_layer = L;
    m->cpu_n_misses = n_misses;
    for (int i = 0; i < n_misses; i++) { m->cpu_miss_eids[i] = eids[i]; m->cpu_miss_w[i] = ws[i]; }
    __atomic_store_n(&m->cpu_phase1_done, 0, __ATOMIC_RELAXED);
    __atomic_store_n(&m->cpu_phase2_done, 0, __ATOMIC_RELAXED);
    __atomic_fetch_add(&m->cpu_gen, 1, __ATOMIC_RELEASE);
    if (__atomic_load_n(&m->cpu_sleeping, __ATOMIC_ACQUIRE) > 0) {
        pthread_mutex_lock(&m->cpu_mu);
        pthread_cond_broadcast(&m->cpu_cv_go);
        pthread_mutex_unlock(&m->cpu_mu);
    }
}

static inline void cm_cpu_pool_wait(CmModel* m) {
    // Main thread executes tid=0 directly (no wasted spinning core!)
    cm_cpu_exec_task(m, 0);
    int nt = m->cpu_n_thr;
    while (__atomic_load_n(&m->cpu_phase2_done, __ATOMIC_ACQUIRE) < nt) cm_cpu_pause();
}

static void cm_cpu_router_topk_grp(const float* logits, const float* exp_bias, int n_exp, int top_k,
                                   int gating_func, int norm_wt, float w_scale, int n_groups, int n_group_used, int* out_e, float* out_w) {
    float probs[CM_MAX_EXPERTS], scores[CM_MAX_EXPERTS];
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
        int gsz = n_exp / n_groups; float gsc[32]; int gkeep[32];
        for (int g = 0; g < n_groups && g < 32; g++) {
            float m1 = -1e30f, m2 = -1e30f;
            for (int i = 0; i < gsz; i++) { float sv = scores[g*gsz + i]; if (sv > m1) { m2 = m1; m1 = sv; } else if (sv > m2) m2 = sv; }
            gsc[g] = (gating_func == 2) ? (m1 + m2) : m1; gkeep[g] = 0;
        }
        for (int u = 0; u < n_group_used; u++) {
            int bg = 0; float bs = -1e30f;
            for (int g = 0; g < n_groups && g < 32; g++) if (!gkeep[g] && gsc[g] > bs) { bs = gsc[g]; bg = g; }
            gkeep[bg] = 1;
        }
        for (int g = 0; g < n_groups && g < 32; g++) if (!gkeep[g]) for (int i = 0; i < gsz; i++) scores[g*gsz + i] = -1e30f;
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

static void cm_stage_dense_layer(CmModel* m, int L, uint8_t* dst) {
    CmLayer* ly = &m->layers[L];
    if (!m->dense_pin_all_attn) {
        cm_read_slice_parallel(m, ly->wq.shard_idx, dst + m->dense_off_wq, (size_t)ly->wq.byte_size, ly->wq.file_off);
        cm_read_slice_parallel(m, ly->wk.shard_idx, dst + m->dense_off_wk, (size_t)ly->wk.byte_size, ly->wk.file_off);
        cm_read_slice_parallel(m, ly->wv.shard_idx, dst + m->dense_off_wv, (size_t)ly->wv.byte_size, ly->wv.file_off);
        cm_read_slice_parallel(m, ly->wo.shard_idx, dst + m->dense_off_wo, (size_t)ly->wo.byte_size, ly->wo.file_off);
    }
    cm_read_slice_parallel(m, ly->ffn_gate.shard_idx, dst + m->dense_off_gate, (size_t)ly->ffn_gate.byte_size, ly->ffn_gate.file_off);
    cm_read_slice_parallel(m, ly->ffn_up.shard_idx,   dst + m->dense_off_up,   (size_t)ly->ffn_up.byte_size,   ly->ffn_up.file_off);
    cm_read_slice_parallel(m, ly->ffn_down.shard_idx, dst + m->dense_off_down, (size_t)ly->ffn_down.byte_size, ly->ffn_down.file_off);
}

typedef struct {
    CmModel* m;
    int first_stream;
    int n_ring;
    pthread_mutex_t mu;
    pthread_cond_t cv_empty;
    pthread_cond_t cv_full;
    int empty_cnt;
    int full_cnt;
} DenseRingProd;

static void* cm_dense_ring_producer(void* arg) {
    DenseRingProd* p = (DenseRingProd*)arg;
    CmModel* m = p->m;
    g_cu.cuCtxSetCurrent(g_cu.ctx);
    for (int L = p->first_stream; L < m->n_layers; L++) {
        int slot = (L - p->first_stream) % p->n_ring;
        pthread_mutex_lock(&p->mu);
        while (p->empty_cnt == 0) pthread_cond_wait(&p->cv_empty, &p->mu);
        p->empty_cnt--;
        pthread_mutex_unlock(&p->mu);

        // Wait until GPU has finished reading this ring slot in previous cycle
        if (L >= p->first_stream + p->n_ring) {
            g_cu.cuEventSynchronize(m->dense_slot_consumed[slot]);
        }
        cm_stage_dense_layer(m, L, m->dense_ring_host[slot]);
        g_cu.cuMemcpyHtoDAsync(m->dense_ring_dptr[slot], m->dense_ring_host[slot], (size_t)m->dense_ring_stride, m->dma_stream);
        g_cu.cuEventRecord(m->dense_slot_ready[slot], m->dma_stream);
        __sync_fetch_and_add(&m->pcie_h2d_bytes, (int64_t)m->dense_ring_stride);

        pthread_mutex_lock(&p->mu);
        p->full_cnt++;
        pthread_cond_signal(&p->cv_full);
        pthread_mutex_unlock(&p->mu);
    }
    return NULL;
}

// ============================================================================
// Standard GGUF BPE Merge Tokenizer + GPT-2 UTF-8 Byte Codec (O(1) Hash Tables)
// ============================================================================
static uint16_t g_cm_gpt2_b2u[256];
static int16_t  g_cm_gpt2_u2b[512];
static int      g_cm_gpt2_init = 0;

static void cm_init_gpt2_tables(void) {
    if (g_cm_gpt2_init) return;
    for (int i = 0; i < 512; i++) g_cm_gpt2_u2b[i] = -1;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            g_cm_gpt2_b2u[b] = (uint16_t)b; g_cm_gpt2_u2b[b] = (int16_t)b;
        } else {
            uint16_t cp = (uint16_t)(256 + n++);
            g_cm_gpt2_b2u[b] = cp; g_cm_gpt2_u2b[cp] = (int16_t)b;
        }
    }
    g_cm_gpt2_init = 1;
}

static size_t cm_utf8_encode_cp(uint16_t cp, char* dst) {
    if (cp < 0x80) { dst[0] = (char)cp; return 1; }
    dst[0] = (char)(0xC0 | (cp >> 6)); dst[1] = (char)(0x80 | (cp & 0x3F)); return 2;
}

static uint64_t cm_fnv1a(const char* s, size_t len) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < len; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ull; }
    return h ? h : 1ull;
}

static int32_t cm_tok_lookup(CmModel* m, const char* s, size_t len) {
    if (!m->bpe_tok_ht || m->bpe_tok_cap == 0) return -1;
    uint32_t mask = m->bpe_tok_cap - 1u;
    uint32_t idx = (uint32_t)cm_fnv1a(s, len) & mask;
    for (;;) {
        const char* k = m->bpe_tok_ht[idx].key;
        if (!k) return -1;
        if (strlen(k) == len && memcmp(k, s, len) == 0) return m->bpe_tok_ht[idx].id;
        idx = (idx + 1u) & mask;
    }
}

static int32_t cm_merge_lookup(CmModel* m, int32_t a, int32_t b) {
    if (!m->bpe_merge_ht || m->bpe_merge_cap == 0) return -1;
    uint64_t pair = (((uint64_t)(uint32_t)a) << 32) | (uint64_t)(uint32_t)b;
    uint32_t mask = m->bpe_merge_cap - 1u;
    uint32_t idx = (uint32_t)((pair ^ (pair >> 30)) * 0xbf58476d1ce4e5b9ull) & mask;
    for (;;) {
        uint64_t kk = m->bpe_merge_ht[idx].pair;
        if (kk == 0xFFFFFFFFFFFFFFFFull) return -1;
        if (kk == pair) return m->bpe_merge_ht[idx].rank;
        idx = (idx + 1u) & mask;
    }
}

static void cm_bpe_ensure_tables(CmModel* m) {
    if (m->bpe_tok_ht || !m->vocab_tokens || m->vocab_size <= 0) return;
    cm_init_gpt2_tables();
    m->bpe_use_sp = 0;
    for (int v = 0; v < m->vocab_size && v < 8192; v++) {
        const unsigned char* tv = (const unsigned char*)m->vocab_tokens[v];
        if (tv && tv[0] == 0xE2 && tv[1] == 0x96 && tv[2] == 0x81) { m->bpe_use_sp = 1; break; }
        if (tv && tv[0] == 0xC4 && tv[1] == 0xA0) { m->bpe_use_sp = 0; break; }
    }
    uint32_t tcap = 1024u;
    while (tcap < (uint32_t)m->vocab_size * 2u) tcap <<= 1;
    m->bpe_tok_cap = tcap;
    m->bpe_tok_ht = (CmBpeTokEntry*)calloc(tcap, sizeof(CmBpeTokEntry));
    uint32_t tmask = tcap - 1u;
    for (int v = 0; v < m->vocab_size; v++) {
        const char* tv = m->vocab_tokens[v];
        if (!tv || !tv[0]) continue;
        uint32_t idx = (uint32_t)cm_fnv1a(tv, strlen(tv)) & tmask;
        while (m->bpe_tok_ht[idx].key) idx = (idx + 1u) & tmask;
        m->bpe_tok_ht[idx].key = tv; m->bpe_tok_ht[idx].id = v;
    }
    if (m->n_bpe_merges > 0 && m->bpe_merges) {
        uint32_t mcap = 1024u;
        while (mcap < (uint32_t)m->n_bpe_merges * 2u) mcap <<= 1;
        m->bpe_merge_cap = mcap;
        m->bpe_merge_ht = (CmBpeMergeEntry*)malloc(mcap * sizeof(CmBpeMergeEntry));
        for (uint32_t i = 0; i < mcap; i++) { m->bpe_merge_ht[i].pair = 0xFFFFFFFFFFFFFFFFull; m->bpe_merge_ht[i].rank = -1; }
        uint32_t mmask = mcap - 1u;
        for (int r = 0; r < m->n_bpe_merges; r++) {
            const char* ms = m->bpe_merges[r]; if (!ms) continue;
            const char* sp = strchr(ms, ' '); if (!sp) continue;
            int32_t a = cm_tok_lookup(m, ms, (size_t)(sp - ms));
            int32_t b = cm_tok_lookup(m, sp + 1, strlen(sp + 1));
            if (a < 0 || b < 0) continue;
            uint64_t pair = (((uint64_t)(uint32_t)a) << 32) | (uint64_t)(uint32_t)b;
            uint32_t idx = (uint32_t)((pair ^ (pair >> 30)) * 0xbf58476d1ce4e5b9ull) & mmask;
            while (m->bpe_merge_ht[idx].pair != 0xFFFFFFFFFFFFFFFFull) idx = (idx + 1u) & mmask;
            m->bpe_merge_ht[idx].pair = pair; m->bpe_merge_ht[idx].rank = r;
        }
    }
}

// ============================================================================
// True Multi-GPU Tensor Parallelism (`TP=2..8`, Megatron Row/Col + P2P AllReduce)
// ============================================================================
static void cm_tp_load_replicated(CmModel* m, CmWeightRef* dst_w, int r, const CmTensorInfo* ti) {
    CmDevCtx* dc = &g_cu.gpus[m->tp[r].dev_idx];
    g_cu.cuCtxSetCurrent(dc->ctx);
    cm_record_meta(dst_w, ti);
    if (ti->byte_size == 0) return;
    if (g_cu.cuMemAlloc(&dst_w->dptr, (size_t)ti->byte_size) != CUDA_SUCCESS) return;
    void* tmp = malloc((size_t)ti->byte_size);
    if (tmp) {
        cm_pread_exact(m->fds[ti->shard_idx], tmp, (size_t)ti->byte_size, ti->file_off);
        g_cu.cuMemcpyHtoD(dst_w->dptr, tmp, (size_t)ti->byte_size);
        free(tmp);
    }
    m->core_vram_bytes += (int64_t)ti->byte_size;
}

static void cm_tp_load_row_shard(CmModel* m, CmWeightRef* dst_w, int r, const CmTensorInfo* ti) {
    int P = m->tp_size;
    CmDevCtx* dc = &g_cu.gpus[m->tp[r].dev_idx];
    g_cu.cuCtxSetCurrent(dc->ctx);
    cm_record_meta(dst_w, ti);
    uint64_t sb = ti->byte_size / (uint64_t)P;
    dst_w->ne1 = ti->ne[1] / (uint64_t)P;
    dst_w->byte_size = sb;
    if (sb == 0) return;
    if (g_cu.cuMemAlloc(&dst_w->dptr, (size_t)sb) != CUDA_SUCCESS) return;
    void* tmp = malloc((size_t)sb);
    if (tmp) {
        cm_pread_exact(m->fds[ti->shard_idx], tmp, (size_t)sb, ti->file_off + (uint64_t)r * sb);
        g_cu.cuMemcpyHtoD(dst_w->dptr, tmp, (size_t)sb);
        free(tmp);
    }
    m->core_vram_bytes += (int64_t)sb;
}

static void cm_tp_load_col_shard_all(CmModel* m, const CmTensorInfo* ti, int L, int which) {
    int P = m->tp_size;
    uint64_t full_rb = cm_ggml_type_bytes(ti->ggml_type, ti->ne[0]);
    uint64_t part_rb = cm_ggml_type_bytes(ti->ggml_type, ti->ne[0] / (uint64_t)P);
    uint64_t rows = ti->ne[1];
    uint64_t sb = rows * part_rb;
    uint8_t* raw = (uint8_t*)malloc((size_t)ti->byte_size);
    uint8_t* pack = (uint8_t*)malloc((size_t)sb);
    if (!raw || !pack) { free(raw); free(pack); return; }
    cm_pread_exact(m->fds[ti->shard_idx], raw, (size_t)ti->byte_size, ti->file_off);
    for (int r = 0; r < P; r++) {
        CmDevCtx* dc = &g_cu.gpus[m->tp[r].dev_idx];
        g_cu.cuCtxSetCurrent(dc->ctx);
        CmWeightRef* w = (which == 0) ? &m->tp[r].layers[L].wo : &m->tp[r].layers[L].ffn_down;
        cm_record_meta(w, ti);
        w->ne0 = ti->ne[0] / (uint64_t)P; w->ne1 = rows; w->byte_size = sb;
        for (uint64_t row = 0; row < rows; row++) {
            memcpy(pack + row * part_rb, raw + row * full_rb + (uint64_t)r * part_rb, (size_t)part_rb);
        }
        if (g_cu.cuMemAlloc(&w->dptr, (size_t)sb) == CUDA_SUCCESS) {
            g_cu.cuMemcpyHtoD(w->dptr, pack, (size_t)sb);
            m->core_vram_bytes += (int64_t)sb;
        }
    }
    free(pack); free(raw);
    g_cu.cuCtxSetCurrent(g_cu.ctx);
}

static void cm_tp_ensure_bufs(CmModel* m, int B) {
    int P = m->tp_size;
    if (m->tp_host_cap >= B) return;
    int cap = B < 32 ? 32 : (B > CM_PREFILL_MAX_B ? CM_PREFILL_MAX_B : B);
    size_t dm = (size_t)m->d_model;
    size_t qd_l = (size_t)((m->n_heads / P) * m->d_head);
    size_t kvd_l = (size_t)((m->n_kv / P) * m->d_head);
    size_t ff_l = (size_t)(m->d_ff_dense / P);
    if ((size_t)m->d_ff_exp > ff_l) ff_l = (size_t)m->d_ff_exp;
    if (ff_l < 64u) ff_l = 64u;
    size_t kv_r = (size_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
    size_t ldim = kv_r + (size_t)m->rope_dim;
    size_t qr_l = (size_t)(m->q_lora_rank > 0 ? m->q_lora_rank : m->d_model);
    size_t nh_l = (size_t)(m->n_heads / P);
    for (int r = 0; r < P; r++) {
        CmTpRank* tr = &m->tp[r];
        CmDevCtx* dc = &g_cu.gpus[tr->dev_idx];
        g_cu.cuCtxSetCurrent(dc->ctx);
        if (tr->d_x)       g_cu.cuMemFree(tr->d_x);
        if (tr->d_xn)      g_cu.cuMemFree(tr->d_xn);
        if (tr->d_q)       g_cu.cuMemFree(tr->d_q);
        if (tr->d_k)       g_cu.cuMemFree(tr->d_k);
        if (tr->d_v)       g_cu.cuMemFree(tr->d_v);
        if (tr->d_ctx)     g_cu.cuMemFree(tr->d_ctx);
        if (tr->d_g)       g_cu.cuMemFree(tr->d_g);
        if (tr->d_u)       g_cu.cuMemFree(tr->d_u);
        if (tr->d_tp_part) g_cu.cuMemFree(tr->d_tp_part);
        if (tr->d_mla_q)    g_cu.cuMemFree(tr->d_mla_q);
        if (tr->d_mla_kv)   g_cu.cuMemFree(tr->d_mla_kv);
        if (tr->d_mla_qabs) g_cu.cuMemFree(tr->d_mla_qabs);
        if (tr->d_mla_olat) g_cu.cuMemFree(tr->d_mla_olat);
        if (tr->d_moe_x)    g_cu.cuMemFree(tr->d_moe_x);
        if (tr->d_moe_y)    g_cu.cuMemFree(tr->d_moe_y);
        if (tr->d_moe_idx)  g_cu.cuMemFree(tr->d_moe_idx);
        if (tr->d_moe_wt)   g_cu.cuMemFree(tr->d_moe_wt);
        for (int b = 0; b < 2; b++) for (int s = 0; s < P; s++) if (tr->d_tp_peer[b][s]) { g_cu.cuMemFree(tr->d_tp_peer[b][s]); tr->d_tp_peer[b][s] = 0; }
        g_cu.cuMemAlloc(&tr->d_x,       (size_t)cap * dm * 4u);
        g_cu.cuMemAlloc(&tr->d_xn,      (size_t)cap * dm * 4u);
        g_cu.cuMemAlloc(&tr->d_q,       (size_t)cap * qd_l * 4u);
        g_cu.cuMemAlloc(&tr->d_k,       (size_t)cap * kvd_l * 4u);
        g_cu.cuMemAlloc(&tr->d_v,       (size_t)cap * kvd_l * 4u);
        g_cu.cuMemAlloc(&tr->d_ctx,     (size_t)cap * qd_l * 4u);
        g_cu.cuMemAlloc(&tr->d_g,       (size_t)cap * ff_l * 4u);
        g_cu.cuMemAlloc(&tr->d_u,       (size_t)cap * ff_l * 4u);
        g_cu.cuMemAlloc(&tr->d_tp_part, (size_t)cap * dm * 4u);
        if (!tr->d_pa_bt) {
            g_cu.cuMemAlloc(&tr->d_pa_bt,   (size_t)CM_PA_MAX_SEQS * CM_PA_MAX_LOGICAL_BLKS * 4u);
            g_cu.cuMemAlloc(&tr->d_cb_pos,  (size_t)CM_PREFILL_MAX_B * 4u);
            g_cu.cuMemAlloc(&tr->d_cb_slot, (size_t)CM_PREFILL_MAX_B * 4u);
            g_cu.cuMemcpyHtoD(tr->d_pa_bt, m->pa_block_table, (size_t)CM_PA_MAX_SEQS * CM_PA_MAX_LOGICAL_BLKS * 4u);
        }
        if (m->hybrid_2d_mode) {
            g_cu.cuMemAlloc(&tr->d_mla_q,    (size_t)cap * qr_l * 4u);
            g_cu.cuMemAlloc(&tr->d_mla_kv,   (size_t)cap * ldim * 4u);
            g_cu.cuMemAlloc(&tr->d_mla_qabs, (size_t)cap * nh_l * ldim * 4u);
            g_cu.cuMemAlloc(&tr->d_mla_olat, (size_t)cap * nh_l * kv_r * 4u);
        }
        if (m->n_experts > 0) {
            g_cu.cuMemAlloc(&tr->d_moe_x,   (size_t)cap * dm * 4u);
            g_cu.cuMemAlloc(&tr->d_moe_y,   (size_t)cap * dm * 4u);
            g_cu.cuMemAlloc(&tr->d_moe_idx, (size_t)cap * CM_MAX_TOPK * 4u);
            g_cu.cuMemAlloc(&tr->d_moe_wt,  (size_t)cap * CM_MAX_TOPK * 4u);
        }
        for (int b = 0; b < 2; b++) for (int s = 0; s < P; s++) {
            if (s != r) g_cu.cuMemAlloc(&tr->d_tp_peer[b][s], (size_t)cap * dm * 4u);
        }
        tr->buf_cap = cap;
    }
    g_cu.cuCtxSetCurrent(g_cu.ctx);
    for (int r = 0; r < P; r++) {
        if (m->h_tp_buf[r]) g_cu.cuMemFreeHost(m->h_tp_buf[r]);
        g_cu.cuMemHostAlloc((void**)&m->h_tp_buf[r], (size_t)cap * dm * 4u, CU_MEMHOSTALLOC_PORTABLE);
    }
    if (m->h_tp_sum) g_cu.cuMemFreeHost(m->h_tp_sum);
    g_cu.cuMemHostAlloc((void**)&m->h_tp_sum, (size_t)cap * dm * 4u, CU_MEMHOSTALLOC_PORTABLE);
    m->tp_host_cap = cap;
}

static void cm_tp_launch_gemv_dc(CmDevCtx* dc, CUstream st, CUdeviceptr w, uint32_t gtype, CUdeviceptr x, CUdeviceptr y,
                                 uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {
    CUfunction fn = dc->k_gemv_q4; uint32_t rptg = 8; CUdeviceptr xin = x;
    if (gtype == CM_GGML_Q4_0 && g_cu.fast_dp4a) {
        uint32_t nb = in_dim >> 5; void* qa[] = { &x, &dc->d_xq8, &in_dim };
        g_cu.cuLaunchKernel(dc->k_quant_q8_1, (nb + 3u) / 4u, 1, 1, 128, 1, 1, 0, st, qa, NULL);
        xin = dc->d_xq8; fn = dc->k_gemv_q4_dp4a; rptg = 16;
    } else if (gtype == CM_GGML_Q4_1 && g_cu.fast_dp4a) {
        uint32_t nb = in_dim >> 5; void* qa[] = { &x, &dc->d_xq8, &in_dim };
        g_cu.cuLaunchKernel(dc->k_quant_q8_1, (nb + 3u) / 4u, 1, 1, 128, 1, 1, 0, st, qa, NULL);
        xin = dc->d_xq8; fn = dc->k_gemv_q4_1_dp4a; rptg = 8;
    } else {
        switch (gtype) {
            case CM_GGML_Q4_0:   fn = dc->k_gemv_q4;    rptg = 8; break;
            case CM_GGML_Q4_1:   fn = dc->k_gemv_q4_1;  rptg = 8; break;
            case CM_GGML_Q5_0:   fn = dc->k_gemv_q5_0;  rptg = 8; break;
            case CM_GGML_Q5_1:   fn = dc->k_gemv_q5_1;  rptg = 8; break;
            case CM_GGML_Q8_0:   fn = dc->k_gemv_q8;    rptg = 8; break;
            case CM_GGML_Q2_K:   fn = dc->k_gemv_q2k;   rptg = 8; break;
            case CM_GGML_Q3_K:   fn = dc->k_gemv_q3k;   rptg = 8; break;
            case CM_GGML_IQ4_XS: fn = dc->k_gemv_iq4xs; rptg = 8; break;
            case CM_GGML_Q4_K:   fn = dc->k_gemv_q4k;   rptg = 8; break;
            case CM_GGML_Q5_K:   fn = dc->k_gemv_q5k;   rptg = 8; break;
            case CM_GGML_Q6_K:   fn = dc->k_gemv_q6k;   rptg = 8; break;
            case CM_GGML_FP8_E4M3: fn = dc->k_gemv_fp8;   rptg = 4; break;
            case CM_GGML_MXFP4:    fn = dc->k_gemv_mxfp4; rptg = 4; break;
            case CM_GGML_F16:    fn = dc->k_gemv_f16;   rptg = 4; break;
            default:             fn = dc->k_gemv_f32;   rptg = 4; break;
        }
    }
    void* args[] = { &w, &xin, &y, &out_dim, &in_dim, &add_res };
    g_cu.cuLaunchKernel(fn, (out_dim + rptg - 1u) / rptg, 1, 1, 128, 1, 1, 0, st, args, NULL);
}

static void cm_tp_launch_gemm_dc(CmDevCtx* dc, CUstream st, CUdeviceptr w, uint32_t gtype, CUdeviceptr x, CUdeviceptr y,
                                 uint32_t out_dim, uint32_t in_dim, uint32_t B, uint32_t add_res) {
    if (B == 1u) { cm_tp_launch_gemv_dc(dc, st, w, gtype, x, y, out_dim, in_dim, add_res); return; }
    CUfunction fn = dc->k_gemm_q4_b; uint32_t tn = 64u, tb = 32u, bdim = 128u;
    if (B <= 8u) {
        if (gtype == CM_GGML_Q4_0) {
            void* args[] = { &w, &x, &y, &out_dim, &in_dim, &B, &add_res };
            g_cu.cuLaunchKernel(dc->k_gemm_q4_sb, (out_dim + 7u) / 8u, 1, 1, 128, 1, 1, 0, st, args, NULL);
            return;
        }
        if (gtype == CM_GGML_Q4_1) {
            void* args[] = { &w, &x, &y, &out_dim, &in_dim, &B, &add_res };
            g_cu.cuLaunchKernel(dc->k_gemm_q4_1_sb, (out_dim + 7u) / 8u, 1, 1, 128, 1, 1, 0, st, args, NULL);
            return;
        }
        if (gtype == CM_GGML_Q4_K) {
            void* args[] = { &w, &x, &y, &out_dim, &in_dim, &B, &add_res };
            g_cu.cuLaunchKernel(dc->k_gemm_q4k_sb, (out_dim + 3u) / 4u, 1, 1, 128, 1, 1, 0, st, args, NULL);
            return;
        }
        if (gtype == CM_GGML_Q6_K) {
            void* args[] = { &w, &x, &y, &out_dim, &in_dim, &B, &add_res };
            g_cu.cuLaunchKernel(dc->k_gemm_q6k_sb, (out_dim + 3u) / 4u, 1, 1, 128, 1, 1, 0, st, args, NULL);
            return;
        }
    }
    switch (gtype) {
        case CM_GGML_Q4_0:   fn = dc->k_gemm_q4_b;    tn = 64u; tb = 32u; bdim = 256u; break;
        case CM_GGML_Q4_1:   fn = dc->k_gemm_q4_1_b;  tn = 64u; tb = 32u; bdim = 256u; break;
        case CM_GGML_Q5_0:   fn = dc->k_gemm_q5_0_b;  tn = 4u;  tb = 8u;  break;
        case CM_GGML_Q5_1:   fn = dc->k_gemm_q5_1_b;  tn = 4u;  tb = 8u;  break;
        case CM_GGML_Q8_0:   fn = dc->k_gemm_q8_b;    tn = 4u;  tb = 8u;  break;
        case CM_GGML_Q2_K:   fn = dc->k_gemm_q2k_b;   tn = 4u;  tb = 4u;  break;
        case CM_GGML_Q3_K:   fn = dc->k_gemm_q3k_b;   tn = 4u;  tb = 4u;  break;
        case CM_GGML_IQ4_XS: fn = dc->k_gemm_iq4xs_b; tn = 4u;  tb = 4u;  break;
        case CM_GGML_Q4_K:   fn = dc->k_gemm_q4k_marlin; tn = 32u; tb = 32u; bdim = 128u; break;
        case CM_GGML_Q5_K:   fn = dc->k_gemm_q5k_b;      tn = 4u;  tb = 4u;  break;
        case CM_GGML_Q6_K:   fn = dc->k_gemm_q6k_marlin; tn = 32u; tb = 32u; bdim = 128u; break;
        case CM_GGML_FP8_E4M3: fn = (B <= 8u) ? dc->k_gemm_fp8_b   : dc->k_gemm_fp8_tc;   tn = (B <= 8u) ? 4u : 32u; tb = (B <= 8u) ? 8u : 32u; break;
        case CM_GGML_MXFP4:    fn = (B <= 8u) ? dc->k_gemm_mxfp4_b : dc->k_gemm_mxfp4_tc; tn = (B <= 8u) ? 4u : 32u; tb = (B <= 8u) ? 8u : 32u; break;
        case CM_GGML_F16:    fn = dc->k_gemm_f16_b;   tn = 4u;  tb = 8u;  break;
        default:             fn = dc->k_gemm_f32_b;   tn = 4u;  tb = 8u;  break;
    }
    void* args[] = { &w, &x, &y, &out_dim, &in_dim, &B, &add_res };
    g_cu.cuLaunchKernel(fn, (out_dim + tn - 1u) / tn, (B + tb - 1u) / tb, 1, bdim, 1, 1, 0, st, args, NULL);
}

static int cm_use_nccl_for_payload(int nccl_ok, int p2p_ok, uint32_t n_elems) {
    if (!nccl_ok) return 0;
    const char* f = getenv("TENON_FORCE_NCCL");
    if (f && atoi(f) > 0) return 1;
    // Use NCCL automatically on NVLink/P2P fabrics OR large batched payloads (>64KB);
    // on non-P2P dual-socket SYS PCIe, small <=64KB decode vectors avoid NCCL SHM spin latency via Pinned-Host!
    return (p2p_ok || n_elems > 16384u) ? 1 : 0;
}

static void cm_tp_bcast_x(CmModel* m, uint32_t n_elems) {
    int P = m->tp_size;
    size_t bytes = (size_t)n_elems * 4u;
    CmDevCtx* dc0 = &g_cu.gpus[m->tp[0].dev_idx];
    if (cm_use_nccl_for_payload(m->tp_nccl_ok, m->tp_p2p_ok, n_elems) && g_cu.ncclBroadcast) {
        g_cu.ncclGroupStart();
        for (int r = 0; r < P; r++) {
            g_cu.cuCtxSetCurrent(g_cu.gpus[m->tp[r].dev_idx].ctx);
            g_cu.ncclBroadcast((const void*)(uintptr_t)m->tp[r].d_x, (void*)(uintptr_t)m->tp[r].d_x,
                               (size_t)n_elems, NCCL_FLOAT32, 0, m->tp_nccl[r], m->tp[r].stream);
        }
        g_cu.ncclGroupEnd();
        g_cu.cuCtxSetCurrent(g_cu.ctx);
        return;
    }
    if (m->tp_p2p_ok && g_cu.cuMemcpyPeerAsync) {
        g_cu.cuCtxSetCurrent(dc0->ctx);
        for (int r = 1; r < P; r++) {
            CmDevCtx* dcr = &g_cu.gpus[m->tp[r].dev_idx];
            g_cu.cuMemcpyPeerAsync(m->tp[r].d_x, dcr->ctx, m->tp[0].d_x, dc0->ctx, bytes, m->tp[0].stream);
        }
        g_cu.cuEventRecord(m->tp[0].ev_sent[0], m->tp[0].stream);
        for (int r = 1; r < P; r++) {
            g_cu.cuCtxSetCurrent(g_cu.gpus[m->tp[r].dev_idx].ctx);
            g_cu.cuStreamWaitEvent(m->tp[r].stream, m->tp[0].ev_sent[0], 0);
        }
    } else {
        g_cu.cuCtxSetCurrent(dc0->ctx);
        g_cu.cuMemcpyDtoHAsync(m->h_tp_sum, m->tp[0].d_x, bytes, m->tp[0].stream);
        g_cu.cuStreamSynchronize(m->tp[0].stream);
        for (int r = 1; r < P; r++) {
            g_cu.cuCtxSetCurrent(g_cu.gpus[m->tp[r].dev_idx].ctx);
            g_cu.cuMemcpyHtoDAsync(m->tp[r].d_x, m->h_tp_sum, bytes, m->tp[r].stream);
        }
    }
}

static void cm_tp_allreduce_add(CmModel* m, uint32_t n_elems, int ar_id) {
    int P = m->tp_size;
    size_t bytes = (size_t)n_elems * 4u;
    if (cm_use_nccl_for_payload(m->tp_nccl_ok, m->tp_p2p_ok, n_elems) && g_cu.ncclAllReduce) {
        g_cu.ncclGroupStart();
        for (int r = 0; r < P; r++) {
            g_cu.cuCtxSetCurrent(g_cu.gpus[m->tp[r].dev_idx].ctx);
            g_cu.ncclAllReduce((const void*)(uintptr_t)m->tp[r].d_tp_part, (void*)(uintptr_t)m->tp[r].d_tp_part,
                               (size_t)n_elems, NCCL_FLOAT32, NCCL_SUM, m->tp_nccl[r], m->tp[r].stream);
        }
        g_cu.ncclGroupEnd();
        g_cu.cuCtxSetCurrent(g_cu.ctx);
        for (int r = 0; r < P; r++) {
            CmDevCtx* dcr = &g_cu.gpus[m->tp[r].dev_idx];
            g_cu.cuCtxSetCurrent(dcr->ctx);
            void* aa[] = { &m->tp[r].d_x, &m->tp[r].d_tp_part, &n_elems };
            g_cu.cuLaunchKernel(dcr->k_add_bias, (n_elems + 255u) / 256u, 1, 1, 256, 1, 1, 0, m->tp[r].stream, aa, NULL);
        }
        return;
    }
    if (m->tp_p2p_ok && g_cu.cuMemcpyPeerAsync) {
        // Strict DAG-Ordered Double-Buffered GPU-to-GPU P2P AllReduce (ZERO event reuse, ZERO data race!)
        int buf = ar_id & 1;
        for (int r = 0; r < P; r++) {
            CmDevCtx* dcr = &g_cu.gpus[m->tp[r].dev_idx];
            g_cu.cuCtxSetCurrent(dcr->ctx);
            for (int s = 0; s < P; s++) {
                if (s == r) continue;
                if (ar_id >= 4) g_cu.cuStreamWaitEvent(m->tp[r].stream, m->tp[s].ev_done[ar_id - 2], 0);
                CmDevCtx* dcs = &g_cu.gpus[m->tp[s].dev_idx];
                g_cu.cuMemcpyPeerAsync(m->tp[s].d_tp_peer[buf][r], dcs->ctx, m->tp[r].d_tp_part, dcr->ctx, bytes, m->tp[r].stream);
            }
            g_cu.cuEventRecord(m->tp[r].ev_sent[ar_id], m->tp[r].stream);
        }
        for (int r = 0; r < P; r++) {
            CmDevCtx* dcr = &g_cu.gpus[m->tp[r].dev_idx];
            g_cu.cuCtxSetCurrent(dcr->ctx);
            for (int s = 0; s < P; s++) {
                if (s != r) g_cu.cuStreamWaitEvent(m->tp[r].stream, m->tp[s].ev_sent[ar_id], 0);
            }
            if (P == 2) {
                CUdeviceptr peer = m->tp[r].d_tp_peer[buf][1 - r];
                void* aa[] = { &m->tp[r].d_x, &m->tp[r].d_tp_part, &peer, &n_elems };
                g_cu.cuLaunchKernel(dcr->k_tp_accum_add, (n_elems + 255u) / 256u, 1, 1, 256, 1, 1, 0, m->tp[r].stream, aa, NULL);
            } else {
                void* a0[] = { &m->tp[r].d_x, &m->tp[r].d_tp_part, &n_elems };
                g_cu.cuLaunchKernel(dcr->k_add_bias, (n_elems + 255u) / 256u, 1, 1, 256, 1, 1, 0, m->tp[r].stream, a0, NULL);
                for (int s = 0; s < P; s++) {
                    if (s == r) continue;
                    CUdeviceptr peer = m->tp[r].d_tp_peer[buf][s];
                    void* as[] = { &m->tp[r].d_x, &peer, &n_elems };
                    g_cu.cuLaunchKernel(dcr->k_add_bias, (n_elems + 255u) / 256u, 1, 1, 256, 1, 1, 0, m->tp[r].stream, as, NULL);
                }
            }
            g_cu.cuEventRecord(m->tp[r].ev_done[ar_id], m->tp[r].stream);
        }
    } else {
        // Ultra-Fast Pinned-Host AllReduce (~15-25 us per 32KB barrier on dual-socket SYS PCIe!)
        for (int r = 0; r < P; r++) {
            g_cu.cuCtxSetCurrent(g_cu.gpus[m->tp[r].dev_idx].ctx);
            g_cu.cuMemcpyDtoHAsync(m->h_tp_buf[r], m->tp[r].d_tp_part, bytes, m->tp[r].stream);
        }
        for (int r = 0; r < P; r++) {
            g_cu.cuCtxSetCurrent(g_cu.gpus[m->tp[r].dev_idx].ctx);
            g_cu.cuStreamSynchronize(m->tp[r].stream);
        }
        float* sum = m->h_tp_sum;
        if (P == 2) {
            const float* a = m->h_tp_buf[0];
            const float* b = m->h_tp_buf[1];
            for (uint32_t i = 0; i < n_elems; i++) sum[i] = a[i] + b[i];
        } else {
            memcpy(sum, m->h_tp_buf[0], bytes);
            for (int r = 1; r < P; r++) {
                const float* src = m->h_tp_buf[r];
                for (uint32_t i = 0; i < n_elems; i++) sum[i] += src[i];
            }
        }
        for (int r = 0; r < P; r++) {
            CmDevCtx* dcr = &g_cu.gpus[m->tp[r].dev_idx];
            g_cu.cuCtxSetCurrent(dcr->ctx);
            g_cu.cuMemcpyHtoDAsync(m->tp[r].d_tp_part, sum, bytes, m->tp[r].stream);
            void* aa[] = { &m->tp[r].d_x, &m->tp[r].d_tp_part, &n_elems };
            g_cu.cuLaunchKernel(dcr->k_add_bias, (n_elems + 255u) / 256u, 1, 1, 256, 1, 1, 0, m->tp[r].stream, aa, NULL);
        }
    }
}

static void launch_gemm_batch_raw(CUstream st, CUdeviceptr w, uint32_t gtype, CUdeviceptr x, CUdeviceptr y,
                                  uint32_t out_dim, uint32_t in_dim, uint32_t B, uint32_t add_res);

static int64_t cm_tp_forward_batch(CmModel* m, const int64_t* ids, uint32_t B, uint32_t start_pos, int64_t* out_argmax_all) {
    int P = m->tp_size;
    cm_tp_ensure_bufs(m, (int)B);
    uint32_t dm = (uint32_t)m->d_model;
    uint32_t nh_l = (uint32_t)(m->n_heads / P), nkv_l = (uint32_t)(m->n_kv / P);
    uint32_t dh = (uint32_t)m->d_head, rd = (uint32_t)m->rope_dim;
    uint32_t qdim_l = nh_l * dh, kvdim_l = nkv_l * dh, dff_l = (uint32_t)(m->d_ff_dense / P);
    uint32_t rn = (uint32_t)m->rope_neox, hf = (uint32_t)m->has_rope_freqs;
    float theta = m->rope_theta;

    // 1. Embed on Rank 0 & broadcast d_x[B, dm] to Ranks 1..P-1
    CmDevCtx* dc0 = &g_cu.gpus[m->tp[0].dev_idx];
    g_cu.cuCtxSetCurrent(dc0->ctx);
    uint32_t gt_emb = m->tok_embd.ggml_type;
    for (uint32_t bi = 0; bi < B; bi++) {
        CUdeviceptr dst = m->tp[0].d_x + (uint64_t)bi * (uint64_t)dm * 4ull;
        uint32_t tid = (uint32_t)ids[bi];
        CUdeviceptr ep = m->tok_embd.dptr;
        void* eargs[] = { &ep, &dst, &tid, &dm, &gt_emb };
        g_cu.cuLaunchKernel(dc0->k_embed, (dm + 255u) / 256u, 1, 1, 256, 1, 1, 0, m->tp[0].stream, eargs, NULL);
    }
    cm_tp_bcast_x(m, B * dm);

    // 2. All layers in lockstep across all P GPUs (Megatron Row/Col TP + 2 AllReduces/layer!)
    for (int L = 0; L < m->n_layers; L++) {
        // --- Attention (Column-Parallel QKV -> Local RoPE+KV+FlashAttn -> Row-Parallel Wo) ---
        for (int r = 0; r < P; r++) {
            CmTpRank* tr = &m->tp[r];
            CmTpLayer* tl = &tr->layers[L];
            CmDevCtx* dc = &g_cu.gpus[tr->dev_idx];
            g_cu.cuCtxSetCurrent(dc->ctx);
            CUstream st = tr->stream;
            uint32_t shw = (B > 1u) ? 1u : 0u;
            CUdeviceptr anw = tl->attn_norm.dptr;
            void* rn_a[] = { &tr->d_x, &anw, &tr->d_xn, &dm, &B, &m->rms_eps, &shw };
            g_cu.cuLaunchKernel(dc->k_rmsnorm, B, 1, 1, 256, 1, 1, 0, st, rn_a, NULL);

            if (B == 1u) {
                if (tl->wq.ggml_type == CM_GGML_Q4_0 && tl->wk.ggml_type == CM_GGML_Q4_0 && tl->wv.ggml_type == CM_GGML_Q4_0) {
                    uint32_t tot = qdim_l + kvdim_l + kvdim_l;
                    if (g_cu.fast_dp4a) {
                        uint32_t nb = dm >> 5; void* q8a[] = { &tr->d_xn, &dc->d_xq8, &dm };
                        g_cu.cuLaunchKernel(dc->k_quant_q8_1, (nb + 3u) / 4u, 1, 1, 128, 1, 1, 0, st, q8a, NULL);
                        void* qa[] = { &tl->wq.dptr, &tl->wk.dptr, &tl->wv.dptr, &dc->d_xq8, &tr->d_q, &tr->d_k, &tr->d_v, &qdim_l, &kvdim_l, &dm };
                        g_cu.cuLaunchKernel(dc->k_gemv_qkv_q4_dp4a, (tot + 15u) / 16u, 1, 1, 128, 1, 1, 0, st, qa, NULL);
                    } else {
                        void* qa[] = { &tl->wq.dptr, &tl->wk.dptr, &tl->wv.dptr, &tr->d_xn, &tr->d_q, &tr->d_k, &tr->d_v, &qdim_l, &kvdim_l, &dm };
                        g_cu.cuLaunchKernel(dc->k_gemv_qkv_q4, (tot + 7u) / 8u, 1, 1, 128, 1, 1, 0, st, qa, NULL);
                    }
                } else {
                    cm_tp_launch_gemv_dc(dc, st, tl->wq.dptr, tl->wq.ggml_type, tr->d_xn, tr->d_q, qdim_l, dm, 0);
                    cm_tp_launch_gemv_dc(dc, st, tl->wk.dptr, tl->wk.ggml_type, tr->d_xn, tr->d_k, kvdim_l, dm, 0);
                    cm_tp_launch_gemv_dc(dc, st, tl->wv.dptr, tl->wv.ggml_type, tr->d_xn, tr->d_v, kvdim_l, dm, 0);
                }
            } else {
                cm_tp_launch_gemm_dc(dc, st, tl->wq.dptr, tl->wq.ggml_type, tr->d_xn, tr->d_q, qdim_l, dm, B, 0);
                cm_tp_launch_gemm_dc(dc, st, tl->wk.dptr, tl->wk.ggml_type, tr->d_xn, tr->d_k, kvdim_l, dm, B, 0);
                cm_tp_launch_gemm_dc(dc, st, tl->wv.dptr, tl->wv.ggml_type, tr->d_xn, tr->d_v, kvdim_l, dm, B, 0);
            }
            if (tl->bq.dptr) {
                void* ba[] = { &tr->d_q, &tl->bq.dptr, &qdim_l, &B };
                g_cu.cuLaunchKernel(dc->k_add_bias_b, (qdim_l + 255u) / 256u, B, 1, 256, 1, 1, 0, st, ba, NULL);
                void* bb[] = { &tr->d_k, &tl->bk.dptr, &kvdim_l, &B };
                g_cu.cuLaunchKernel(dc->k_add_bias_b, (kvdim_l + 255u) / 256u, B, 1, 256, 1, 1, 0, st, bb, NULL);
                void* bc[] = { &tr->d_v, &tl->bv.dptr, &kvdim_l, &B };
                g_cu.cuLaunchKernel(dc->k_add_bias_b, (kvdim_l + 255u) / 256u, B, 1, 256, 1, 1, 0, st, bc, NULL);
            }
            CUdeviceptr rf = hf ? tr->rope_freqs.dptr : tr->d_x;
            if (m->kv_quant_q8) {
                void* rargs[] = { &tr->d_q, &tr->d_k, &tr->d_v, &tl->k_cache, &tl->v_cache, &rf, &nh_l, &nkv_l, &dh, &rd, &start_pos, &theta, &rn, &hf, &B };
                g_cu.cuLaunchKernel(dc->k_rope_kv_q8_b, nh_l, B, 1, 128, 1, 1, 0, st, rargs, NULL);
                void* aargs[] = { &tr->d_q, &tl->k_cache, &tl->v_cache, &tr->d_ctx, &nh_l, &nkv_l, &dh, &start_pos, &B };
                g_cu.cuLaunchKernel(dc->k_flash_prefill_q8, nh_l, B, 1, 128, 1, 1, 0, st, aargs, NULL);
            } else if (B == 1u) {
                uint32_t hd2 = dh >> 1, kv_len = start_pos + 1u;
                float std_scale = 1.0f / sqrtf((float)dh);
                void* rargs[] = { &tr->d_q, &tr->d_k, &tr->d_v, &tl->k_cache, &tl->v_cache, &rf, &nh_l, &nkv_l, &dh, &rd, &start_pos, &theta, &rn, &hf };
                g_cu.cuLaunchKernel(dc->k_rope_kv, (hd2 + 31u) / 32u, nh_l, 1, 32, 1, 1, 0, st, rargs, NULL);
                void* aargs[] = { &tr->d_q, &tl->k_cache, &tl->v_cache, &tr->d_ctx, &nh_l, &nkv_l, &dh, &kv_len, &dh, &std_scale };
                g_cu.cuLaunchKernel(dc->k_flash_dec, nh_l, 1, 1, 128, 1, 1, 0, st, aargs, NULL);
            } else {
                uint32_t hd2 = dh >> 1;
                void* rargs[] = { &tr->d_q, &tr->d_k, &tr->d_v, &tl->k_cache, &tl->v_cache, &rf, &nh_l, &nkv_l, &dh, &rd, &start_pos, &theta, &rn, &hf, &B };
                g_cu.cuLaunchKernel(dc->k_rope_kv_b, (hd2 + 31u) / 32u, nh_l, B, 32, 1, 1, 0, st, rargs, NULL);
                void* aargs[] = { &tr->d_q, &tl->k_cache, &tl->v_cache, &tr->d_ctx, &nh_l, &nkv_l, &dh, &start_pos, &B };
                g_cu.cuLaunchKernel(dc->k_flash_prefill, nh_l, B, 1, 128, 1, 1, 0, st, aargs, NULL);
            }
            if (B == 1u) cm_tp_launch_gemv_dc(dc, st, tl->wo.dptr, tl->wo.ggml_type, tr->d_ctx, tr->d_tp_part, dm, qdim_l, 0);
            else         cm_tp_launch_gemm_dc(dc, st, tl->wo.dptr, tl->wo.ggml_type, tr->d_ctx, tr->d_tp_part, dm, qdim_l, B, 0);
        }
        cm_tp_allreduce_add(m, B * dm, 2 + L * 2);

        // --- FFN (Column-Parallel Gate/Up -> Local SwiGLU -> Row-Parallel Down) ---
        for (int r = 0; r < P; r++) {
            CmTpRank* tr = &m->tp[r];
            CmTpLayer* tl = &tr->layers[L];
            CmDevCtx* dc = &g_cu.gpus[tr->dev_idx];
            g_cu.cuCtxSetCurrent(dc->ctx);
            CUstream st = tr->stream;
            uint32_t shw = (B > 1u) ? 1u : 0u;
            CUdeviceptr fnw = tl->ffn_norm.dptr;
            void* rn_f[] = { &tr->d_x, &fnw, &tr->d_xn, &dm, &B, &m->rms_eps, &shw };
            g_cu.cuLaunchKernel(dc->k_rmsnorm, B, 1, 1, 256, 1, 1, 0, st, rn_f, NULL);

            if (B == 1u) {
                if (tl->ffn_gate.ggml_type == CM_GGML_Q4_0 && tl->ffn_up.ggml_type == CM_GGML_Q4_0) {
                    if (g_cu.fast_dp4a) {
                        uint32_t nb = dm >> 5; void* q8a[] = { &tr->d_xn, &dc->d_xq8, &dm };
                        g_cu.cuLaunchKernel(dc->k_quant_q8_1, (nb + 3u) / 4u, 1, 1, 128, 1, 1, 0, st, q8a, NULL);
                        void* gua[] = { &tl->ffn_gate.dptr, &tl->ffn_up.dptr, &dc->d_xq8, &tr->d_g, &dff_l, &dm };
                        g_cu.cuLaunchKernel(dc->k_dense_gu_q4_dp4a, (dff_l + 7u) / 8u, 1, 1, 128, 1, 1, 0, st, gua, NULL);
                    } else {
                        void* gua[] = { &tl->ffn_gate.dptr, &tl->ffn_up.dptr, &tr->d_xn, &tr->d_g, &dff_l, &dm };
                        g_cu.cuLaunchKernel(dc->k_dense_gu_q4, (dff_l + 3u) / 4u, 1, 1, 128, 1, 1, 0, st, gua, NULL);
                    }
                } else {
                    cm_tp_launch_gemv_dc(dc, st, tl->ffn_gate.dptr, tl->ffn_gate.ggml_type, tr->d_xn, tr->d_g, dff_l, dm, 0);
                    cm_tp_launch_gemv_dc(dc, st, tl->ffn_up.dptr,   tl->ffn_up.ggml_type,   tr->d_xn, tr->d_u, dff_l, dm, 0);
                    void* swa[] = { &tr->d_g, &tr->d_u, &dff_l };
                    g_cu.cuLaunchKernel(dc->k_swiglu_mul, (dff_l + 255u) / 256u, 1, 1, 256, 1, 1, 0, st, swa, NULL);
                }
                cm_tp_launch_gemv_dc(dc, st, tl->ffn_down.dptr, tl->ffn_down.ggml_type, tr->d_g, tr->d_tp_part, dm, dff_l, 0);
            } else {
                cm_tp_launch_gemm_dc(dc, st, tl->ffn_gate.dptr, tl->ffn_gate.ggml_type, tr->d_xn, tr->d_g, dff_l, dm, B, 0);
                cm_tp_launch_gemm_dc(dc, st, tl->ffn_up.dptr,   tl->ffn_up.ggml_type,   tr->d_xn, tr->d_u, dff_l, dm, B, 0);
                uint32_t tot = B * dff_l;
                void* swa[] = { &tr->d_g, &tr->d_u, &tot };
                g_cu.cuLaunchKernel(dc->k_swiglu_mul, (tot + 255u) / 256u, 1, 1, 256, 1, 1, 0, st, swa, NULL);
                cm_tp_launch_gemm_dc(dc, st, tl->ffn_down.dptr, tl->ffn_down.ggml_type, tr->d_g, tr->d_tp_part, dm, dff_l, B, 0);
            }
        }
        cm_tp_allreduce_add(m, B * dm, 2 + L * 2 + 1);
    }

    // 3. Final out_norm + BATCHED lm_head + BATCHED argmax on GPU 0 (1 weight pass + 1 sync instead of B serial reads!)
    g_cu.cuCtxSetCurrent(dc0->ctx);
    CUstream st0 = m->tp[0].stream;
    uint32_t v = (uint32_t)m->vocab_size;
    int64_t last_tok = 0;
    if (out_argmax_all != NULL && B > 1u) {
        if (m->batch_logits_cap < (int)B) {
            int ncap = (int)(B < 16 ? 16 : B);
            if (m->d_batch_logits) g_cu.cuMemFree(m->d_batch_logits);
            g_cu.cuMemAlloc(&m->d_batch_logits, (size_t)ncap * (size_t)v * 4u);
            m->batch_logits_cap = ncap;
        }
        if (!m->d_argmax_batch) {
            g_cu.cuMemAlloc(&m->d_argmax_batch, (size_t)CM_PREFILL_MAX_B * 4u);
            g_cu.cuMemHostAlloc((void**)&m->h_argmax_batch, (size_t)CM_PREFILL_MAX_B * 4u, CU_MEMHOSTALLOC_PORTABLE);
        }
        launch_rmsnorm(st0, m->tp[0].d_x, &m->out_norm, m->tp[0].d_xn, dm, B, m->rms_eps, 1);
        launch_gemm_batch_raw(st0, m->lm_head.dptr, m->lm_head.ggml_type, m->tp[0].d_xn, m->d_batch_logits, v, dm, B, 0);
        void* aa[] = { &m->d_batch_logits, &m->d_argmax_batch, &v, &B };
        g_cu.cuLaunchKernel(g_cu.k_argmax_b, B, 1, 1, 256, 1, 1, 0, st0, aa, NULL);
        g_cu.cuMemcpyDtoHAsync(m->h_argmax_batch, m->d_argmax_batch, (size_t)B * 4u, st0);
        g_cu.cuMemcpyDtoDAsync(m->d_logits, m->d_batch_logits + (uint64_t)(B - 1u) * (uint64_t)v * 4ull, (size_t)v * 4u, st0);
        g_cu.cuStreamSynchronize(st0);
        for (uint32_t bi = 0; bi < B; bi++) {
            out_argmax_all[bi] = (int64_t)m->h_argmax_batch[bi];
            last_tok = out_argmax_all[bi];
        }
    } else if (out_argmax_all != NULL) {
        CUdeviceptr bx = m->tp[0].d_x;
        launch_rmsnorm(st0, bx, &m->out_norm, m->d_xn, dm, 1, m->rms_eps, 0);
        launch_gemv(st0, &m->lm_head, m->d_xn, m->d_logits, v, dm, 0);
        void* aargs[] = { &m->d_logits, &m->d_argmax, &v };
        g_cu.cuLaunchKernel(g_cu.k_argmax, 1, 1, 1, 256, 1, 1, 0, st0, aargs, NULL);
        g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, st0);
        g_cu.cuStreamSynchronize(st0);
        out_argmax_all[0] = (int64_t)m->h_argmax[0];
        last_tok = out_argmax_all[0];
    } else {
        CUdeviceptr lx = m->tp[0].d_x + (uint64_t)(B - 1u) * (uint64_t)dm * 4ull;
        launch_rmsnorm(st0, lx, &m->out_norm, m->d_xn, dm, 1, m->rms_eps, 0);
        launch_gemv(st0, &m->lm_head, m->d_xn, m->d_logits, v, dm, 0);
        void* aargs[] = { &m->d_logits, &m->d_argmax, &v };
        g_cu.cuLaunchKernel(g_cu.k_argmax, 1, 1, 1, 256, 1, 1, 0, st0, aargs, NULL);
        g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, st0);
        g_cu.cuStreamSynchronize(st0);
        if (m->d_last_hidden) g_cu.cuMemcpyDtoDAsync(m->d_last_hidden, lx, (size_t)dm * 4u, st0);
        last_tok = (int64_t)m->h_argmax[0];
    }
    for (int r = 1; r < P; r++) {
        g_cu.cuCtxSetCurrent(g_cu.gpus[m->tp[r].dev_idx].ctx);
        g_cu.cuStreamSynchronize(m->tp[r].stream);
    }
    g_cu.cuCtxSetCurrent(g_cu.ctx);
    return last_tok;
}

// ============================================================================
// Tenon v7: True 2D Hybrid Parallelism (`TP Attention x EP Experts` on Multi-GPU MoE!)
// Shards BOTH Attention heads/MLA/KV across TP AND MoE Experts across EP — ZERO weight duplication!
// ============================================================================
static int64_t cm_hybrid_2d_step(CmModel* m, int64_t tok_id, uint32_t pos) {
    int P = m->tp_size;
    cm_tp_ensure_bufs(m, 1);
    uint32_t dm = (uint32_t)m->d_model, nh_l = (uint32_t)(m->n_heads / P), nkv_l = (uint32_t)(m->n_kv / P);
    uint32_t dh = (uint32_t)m->d_head, rd = (uint32_t)m->rope_dim, vd = (uint32_t)m->v_head_dim;
    uint32_t qdim_l = nh_l * dh, kvdim_l = nkv_l * dh, dff_l = (uint32_t)(m->d_ff_dense / P);
    uint32_t rn = (uint32_t)m->rope_neox, hf = (uint32_t)m->has_rope_freqs, one = 1u, zero_u = 0u;
    float theta = m->rope_theta;

    // 1. Embed on Rank 0 & broadcast d_x[dm] to all P ranks
    CmDevCtx* dc0 = &g_cu.gpus[m->tp[0].dev_idx];
    g_cu.cuCtxSetCurrent(dc0->ctx);
    uint32_t gt_emb = m->tok_embd.ggml_type, tid = (uint32_t)tok_id;
    CUdeviceptr ep = m->tok_embd.dptr;
    void* eargs[] = { &ep, &m->tp[0].d_x, &tid, &dm, &gt_emb };
    g_cu.cuLaunchKernel(dc0->k_embed, (dm + 255u) / 256u, 1, 1, 256, 1, 1, 0, m->tp[0].stream, eargs, NULL);
    cm_tp_bcast_x(m, dm);

    for (int L = 0; L < m->n_layers; L++) {
        CmLayer* ly0 = &m->layers[L];
        // --- Stage 1: Tensor-Parallel Sharded Attention (MLA OR GQA) across all P GPUs ---
        for (int r = 0; r < P; r++) {
            CmTpRank* tr = &m->tp[r];
            CmTpLayer* tl = &tr->layers[L];
            CmDevCtx* dc = &g_cu.gpus[tr->dev_idx];
            g_cu.cuCtxSetCurrent(dc->ctx);
            CUstream st = tr->stream;
            CUdeviceptr anw = tl->attn_norm.dptr;
            void* rn_a[] = { &tr->d_x, &anw, &tr->d_xn, &dm, &one, &m->rms_eps, &zero_u };
            g_cu.cuLaunchKernel(dc->k_rmsnorm, 1, 1, 1, 256, 1, 1, 0, st, rn_a, NULL);
            CUdeviceptr rf = hf ? tr->rope_freqs.dptr : tr->d_x;
            uint32_t wo_in_l = (tl->wo.ne0 > 0) ? (uint32_t)tl->wo.ne0 : qdim_l;

            if (tl->is_mla && tl->wkv_a.dptr && tl->mla_wuk_f16 && tl->mla_wuv_f16) {
                uint32_t kv_r = (uint32_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
                uint32_t nope_d = dh - rd, ldim = kv_r + rd, half_r = rd >> 1;
                if (tl->wq_a.dptr && tl->wq_b.dptr && m->q_lora_rank > 0) {
                    uint32_t qr = (uint32_t)m->q_lora_rank;
                    cm_tp_launch_gemv_dc(dc, st, tl->wq_a.dptr, tl->wq_a.ggml_type, tr->d_xn, tr->d_mla_q, qr, dm, 0);
                    if (tl->q_a_norm.dptr) {
                        CUdeviceptr qnw = tl->q_a_norm.dptr;
                        void* qna[] = { &tr->d_mla_q, &qnw, &tr->d_mla_q, &qr, &one, &m->rms_eps, &zero_u };
                        g_cu.cuLaunchKernel(dc->k_rmsnorm, 1, 1, 1, 256, 1, 1, 0, st, qna, NULL);
                    }
                    cm_tp_launch_gemv_dc(dc, st, tl->wq_b.dptr, tl->wq_b.ggml_type, tr->d_mla_q, tr->d_q, qdim_l, qr, 0);
                } else {
                    cm_tp_launch_gemv_dc(dc, st, tl->wq.dptr, tl->wq.ggml_type, tr->d_xn, tr->d_q, qdim_l, dm, 0);
                }
                cm_tp_launch_gemv_dc(dc, st, tl->wkv_a.dptr, tl->wkv_a.ggml_type, tr->d_xn, tr->d_mla_kv, ldim, dm, 0);
                if (tl->kv_a_norm.dptr) {
                    CUdeviceptr knw = tl->kv_a_norm.dptr;
                    void* kna[] = { &tr->d_mla_kv, &knw, &tr->d_mla_kv, &kv_r, &one, &m->rms_eps, &zero_u };
                    g_cu.cuLaunchKernel(dc->k_rmsnorm, 1, 1, 1, 256, 1, 1, 0, st, kna, NULL);
                }
                CUdeviceptr kpe_ptr = tr->d_mla_kv + (uint64_t)kv_r * 4ull;
                void* rpe_a[] = { &tr->d_q, &kpe_ptr, &rf, &nh_l, &dh, &nope_d, &rd, &pos, &theta, &rn, &hf, &one };
                g_cu.cuLaunchKernel(dc->k_mla_rope_pe, (half_r + 31u) / 32u, nh_l + 1u, 1, 32, 1, 1, 0, st, rpe_a, NULL);
                float mla_sc = 1.0f / sqrtf((float)dh);
                if (m->yarn_mscale_k > 0.0f && m->yarn_factor > 1.0f) { float ms = 1.0f + m->yarn_mscale_k * logf(m->yarn_factor); mla_sc *= ms * ms; }
                uint32_t out_st = wo_in_l / nh_l;
                void* sa[] = { &tr->d_mla_kv, &kpe_ptr, &tl->k_cache, &kv_r, &rd, &pos, &one };
                g_cu.cuLaunchKernel(dc->k_mla_store_latent, (ldim + 255u) / 256u, 1, 1, 256, 1, 1, 0, st, sa, NULL);
                void* qa[] = { &tr->d_q, &tl->mla_wuk_f16, &tr->d_mla_qabs, &nh_l, &dh, &nope_d, &rd, &kv_r, &one };
                g_cu.cuLaunchKernel(dc->k_mla_absorb_q, (ldim + 3u) / 4u, nh_l, 1, 128, 1, 1, 0, st, qa, NULL);
                void* fa[] = { &tr->d_mla_qabs, &tl->k_cache, &tr->d_mla_olat, &nh_l, &kv_r, &rd, &mla_sc, &pos, &one };
                g_cu.cuLaunchKernel(dc->k_flash_mla_latent, nh_l, 1, 1, 128, 1, 1, 0, st, fa, NULL);
                void* va[] = { &tr->d_mla_olat, &tl->mla_wuv_f16, &tr->d_ctx, &nh_l, &kv_r, &vd, &out_st, &one };
                g_cu.cuLaunchKernel(dc->k_mla_proj_v, (out_st + 3u) / 4u, nh_l, 1, 128, 1, 1, 0, st, va, NULL);
            } else {
                cm_tp_launch_gemv_dc(dc, st, tl->wq.dptr, tl->wq.ggml_type, tr->d_xn, tr->d_q, qdim_l, dm, 0);
                cm_tp_launch_gemv_dc(dc, st, tl->wk.dptr, tl->wk.ggml_type, tr->d_xn, tr->d_k, kvdim_l, dm, 0);
                cm_tp_launch_gemv_dc(dc, st, tl->wv.dptr, tl->wv.ggml_type, tr->d_xn, tr->d_v, kvdim_l, dm, 0);
                if (tl->bq.dptr) {
                    void* ba[] = { &tr->d_q, &tl->bq.dptr, &qdim_l }; g_cu.cuLaunchKernel(dc->k_add_bias, (qdim_l+255u)/256u, 1, 1, 256, 1, 1, 0, st, ba, NULL);
                    void* bb[] = { &tr->d_k, &tl->bk.dptr, &kvdim_l }; g_cu.cuLaunchKernel(dc->k_add_bias, (kvdim_l+255u)/256u, 1, 1, 256, 1, 1, 0, st, bb, NULL);
                    void* bc[] = { &tr->d_v, &tl->bv.dptr, &kvdim_l }; g_cu.cuLaunchKernel(dc->k_add_bias, (kvdim_l+255u)/256u, 1, 1, 256, 1, 1, 0, st, bc, NULL);
                }
                uint32_t hd2 = dh >> 1, kv_len = pos + 1u;
                float std_sc = 1.0f / sqrtf((float)dh);
                if (m->kv_quant_q8) {
                    void* ra[] = { &tr->d_q, &tr->d_k, &tr->d_v, &tl->k_cache, &tl->v_cache, &rf, &nh_l, &nkv_l, &dh, &rd, &pos, &theta, &rn, &hf, &one };
                    g_cu.cuLaunchKernel(dc->k_rope_kv_q8_b, nh_l, 1, 1, 128, 1, 1, 0, st, ra, NULL);
                    void* aa[] = { &tr->d_q, &tl->k_cache, &tl->v_cache, &tr->d_ctx, &nh_l, &nkv_l, &dh, &pos, &one };
                    g_cu.cuLaunchKernel(dc->k_flash_prefill_q8, nh_l, 1, 1, 128, 1, 1, 0, st, aa, NULL);
                } else {
                    void* ra[] = { &tr->d_q, &tr->d_k, &tr->d_v, &tl->k_cache, &tl->v_cache, &rf, &nh_l, &nkv_l, &dh, &rd, &pos, &theta, &rn, &hf };
                    g_cu.cuLaunchKernel(dc->k_rope_kv, (hd2 + 31u) / 32u, nh_l, 1, 32, 1, 1, 0, st, ra, NULL);
                    void* aa[] = { &tr->d_q, &tl->k_cache, &tl->v_cache, &tr->d_ctx, &nh_l, &nkv_l, &dh, &kv_len, &dh, &std_sc };
                    g_cu.cuLaunchKernel(dc->k_flash_dec, nh_l, 1, 1, 128, 1, 1, 0, st, aa, NULL);
                }
            }
            cm_tp_launch_gemv_dc(dc, st, tl->wo.dptr, tl->wo.ggml_type, tr->d_ctx, tr->d_tp_part, dm, wo_in_l, 0);
        }
        cm_tp_allreduce_add(m, dm, 2 + L * 2);

        // --- Stage 2: Tensor-Parallel Dense FFN OR Expert-Parallel Sharded MoE (ZERO d_xn Broadcast!) ---
        if (ly0->is_dense) {
            for (int r = 0; r < P; r++) {
                CmTpRank* tr = &m->tp[r];
                CmTpLayer* tl = &tr->layers[L];
                CmDevCtx* dc = &g_cu.gpus[tr->dev_idx];
                g_cu.cuCtxSetCurrent(dc->ctx);
                CUstream st = tr->stream;
                CUdeviceptr fnw = tl->ffn_norm.dptr;
                void* rn_f[] = { &tr->d_x, &fnw, &tr->d_xn, &dm, &one, &m->rms_eps, &zero_u };
                g_cu.cuLaunchKernel(dc->k_rmsnorm, 1, 1, 1, 256, 1, 1, 0, st, rn_f, NULL);
                cm_tp_launch_gemv_dc(dc, st, tl->ffn_gate.dptr, tl->ffn_gate.ggml_type, tr->d_xn, tr->d_g, dff_l, dm, 0);
                cm_tp_launch_gemv_dc(dc, st, tl->ffn_up.dptr,   tl->ffn_up.ggml_type,   tr->d_xn, tr->d_u, dff_l, dm, 0);
                void* swa[] = { &tr->d_g, &tr->d_u, &dff_l };
                g_cu.cuLaunchKernel(dc->k_swiglu_mul, (dff_l + 255u) / 256u, 1, 1, 256, 1, 1, 0, st, swa, NULL);
                cm_tp_launch_gemv_dc(dc, st, tl->ffn_down.dptr, tl->ffn_down.ggml_type, tr->d_g, tr->d_tp_part, dm, dff_l, 0);
            }
            cm_tp_allreduce_add(m, dm, 2 + L * 2 + 1);
        } else {
            for (int r = 0; r < P; r++) {
                CmTpRank* tr = &m->tp[r];
                CmTpLayer* tl = &tr->layers[L];
                CmDevCtx* dc = &g_cu.gpus[tr->dev_idx];
                g_cu.cuCtxSetCurrent(dc->ctx);
                CUdeviceptr fnw = tl->ffn_norm.dptr;
                void* rn_f[] = { &tr->d_x, &fnw, &tr->d_xn, &dm, &one, &m->rms_eps, &zero_u };
                g_cu.cuLaunchKernel(dc->k_rmsnorm, 1, 1, 1, 256, 1, 1, 0, tr->stream, rn_f, NULL);
            }
            // Route on GPU 0 (d_xn is ALREADY identical on all P GPUs -> ZERO d_xn broadcast!)
            g_cu.cuCtxSetCurrent(dc0->ctx);
            CUstream st0 = m->tp[0].stream;
            cm_tp_launch_gemv_dc(dc0, st0, ly0->router.dptr, ly0->router.ggml_type, m->tp[0].d_xn, m->d_rlogits, (uint32_t)m->n_experts, dm, 0);
            g_cu.cuMemcpyDtoHAsync(m->h_rlogits, m->d_rlogits, (size_t)m->n_experts * 4u, st0);
            g_cu.cuStreamSynchronize(st0);
            int top_e[CM_MAX_TOPK]; float top_w[CM_MAX_TOPK];
            cm_cpu_router_topk_grp(m->h_rlogits, ly0->exp_bias_host, m->n_experts, m->top_k,
                                   m->gating_func, m->expert_weights_norm, m->expert_weights_scale,
                                   m->n_expert_groups, m->n_group_used, top_e, top_w);
            int Er = m->ep_exp_per_rank, cnt[CM_MAX_TP] = {0};
            for (int k = 0; k < m->top_k; k++) {
                int eid = top_e[k], r = eid / Er; if (r >= P) r = P - 1;
                int ls = eid - r * Er, idx = cnt[r]++;
                m->ep[r].h_eidx[idx] = (uint32_t)ls; m->ep[r].h_ewt[idx] = top_w[k];
                m->cache_hits++;
            }
            uint32_t df = (uint32_t)m->d_ff_exp;
            uint64_t gb = ly0->gate_exp_bytes, sb = ly0->slot_stride_bytes, doff = ly0->gate_exp_bytes + ly0->up_exp_bytes;
            int is_q4k_gu = (ly0->gate_exps.ggml_type == CM_GGML_Q4_K);
            int is_q4k_dn = (ly0->down_exps.ggml_type == CM_GGML_Q4_K);
            int is_q6k_dn = (ly0->down_exps.ggml_type == CM_GGML_Q6_K);
            int is_q8_dn  = (ly0->down_exps.ggml_type == CM_GGML_Q8_0);
            int is_q5_0_dn = (ly0->down_exps.ggml_type == CM_GGML_Q5_0);
            int is_q5_1_dn = (ly0->down_exps.ggml_type == CM_GGML_Q5_1);
            for (int r = 0; r < P; r++) {
                CmTpRank* tr = &m->tp[r];
                CmEpRank* er = &m->ep[r];
                CmDevCtx* dc = &g_cu.gpus[tr->dev_idx];
                g_cu.cuCtxSetCurrent(dc->ctx);
                CUstream st = tr->stream;
                if (r == 0 && ly0->has_shared_exp) {
                    uint32_t sh_ff = (ly0->sh_gate.ne1 > 0) ? (uint32_t)ly0->sh_gate.ne1 : df;
                    cm_tp_launch_gemv_dc(dc, st, ly0->sh_gate.dptr, ly0->sh_gate.ggml_type, tr->d_xn, m->d_sh_g, sh_ff, dm, 0);
                    cm_tp_launch_gemv_dc(dc, st, ly0->sh_up.dptr,   ly0->sh_up.ggml_type,   tr->d_xn, m->d_sh_u, sh_ff, dm, 0);
                    void* swa[] = { &m->d_sh_g, &m->d_sh_u, &sh_ff };
                    g_cu.cuLaunchKernel(dc->k_swiglu_mul, (sh_ff + 255u) / 256u, 1, 1, 256, 1, 1, 0, st, swa, NULL);
                    cm_tp_launch_gemv_dc(dc, st, ly0->sh_down.dptr, ly0->sh_down.ggml_type, m->d_sh_g, tr->d_tp_part, dm, sh_ff, 0);
                } else {
                    if (g_cu.cuMemsetD8Async) g_cu.cuMemsetD8Async(tr->d_tp_part, 0, (size_t)dm * 4u, st);
                    else g_cu.cuMemsetD8(tr->d_tp_part, 0, (size_t)dm * 4u);
                }
                if (cnt[r] > 0) {
                    uint32_t na = (uint32_t)cnt[r];
                    g_cu.cuMemcpyHtoDAsync(er->d_eidx, er->h_eidx, (size_t)na * 4u, st);
                    g_cu.cuMemcpyHtoDAsync(er->d_ewt,  er->h_ewt,  (size_t)na * 4u, st);
                    void* gu_a[] = { &er->cache_dptr[L], &tr->d_xn, &er->d_eidx, &er->d_h, &dm, &df, &na, &gb, &sb };
                    g_cu.cuLaunchKernel(is_q4k_gu ? dc->k_moe_slot_gu_q4k : dc->k_moe_slot_gu_q4,
                                        is_q4k_gu ? ((df + 3u) / 4u) : ((df + 7u) / 8u), na, 1, 128, 1, 1, 0, st, gu_a, NULL);
                    CUfunction dfn = is_q6k_dn ? dc->k_moe_slot_down_q6k
                        : (is_q4k_dn ? dc->k_moe_slot_down_q4k
                           : (is_q8_dn ? dc->k_moe_slot_down_q8
                              : (is_q5_0_dn ? dc->k_moe_slot_down_q5_0
                                 : (is_q5_1_dn ? dc->k_moe_slot_down_q5_1
                                    : ((ly0->down_exps.ggml_type == CM_GGML_Q4_1) ? dc->k_moe_slot_down_q4_1 : dc->k_moe_slot_down_q4)))));
                    void* dn_a[] = { &er->cache_dptr[L], &er->d_h, &er->d_eidx, &er->d_ewt, &tr->d_tp_part, &dm, &df, &na, &doff, &sb };
                    g_cu.cuLaunchKernel(dfn, (is_q4k_dn || is_q6k_dn || is_q8_dn || is_q5_0_dn || is_q5_1_dn) ? ((dm + 3u) / 4u) : ((dm + 15u) / 16u),
                                        1, 1, 128, 1, 1, 0, st, dn_a, NULL);
                }
            }
            cm_tp_allreduce_add(m, dm, 2 + L * 2 + 1);
        }
    }

    g_cu.cuCtxSetCurrent(dc0->ctx);
    CUstream st0 = m->tp[0].stream;
    uint32_t v = (uint32_t)m->vocab_size;
    if (m->d_last_hidden) g_cu.cuMemcpyDtoDAsync(m->d_last_hidden, m->tp[0].d_x, (size_t)dm * 4u, st0);
    launch_rmsnorm(st0, m->tp[0].d_x, &m->out_norm, m->d_xn, dm, 1, m->rms_eps, 0);
    launch_gemv(st0, &m->lm_head, m->d_xn, m->d_logits, v, dm, 0);
    void* aargs[] = { &m->d_logits, &m->d_argmax, &v };
    g_cu.cuLaunchKernel(g_cu.k_argmax, 1, 1, 1, 256, 1, 1, 0, st0, aargs, NULL);
    g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, st0);
    g_cu.cuStreamSynchronize(st0);
    for (int r = 1; r < P; r++) {
        g_cu.cuCtxSetCurrent(g_cu.gpus[m->tp[r].dev_idx].ctx);
        g_cu.cuStreamSynchronize(m->tp[r].stream);
    }
    g_cu.cuCtxSetCurrent(g_cu.ctx);
    return (int64_t)m->h_argmax[0];
}

// ============================================================================
// Public Exported C ABI for Tenon (`std/gpu/cuda/moe_stream_cuda.tn`)
// ============================================================================
int64_t tn_cumoe_load_gguf(int64_t path_cstr, int64_t cache_slots_per_layer, int64_t use_nocache, int64_t enable_overlap) {
    if (!tn_cumoe_init() || !path_cstr) return 0;
    int slot = -1;
    for (int i = 1; i < CM_MAX_MODELS; i++) if (!g_cm[i].active) { slot = i; break; }
    if (slot < 0) return 0;
    CmModel* m = &g_cm[slot];
    memset(m, 0, sizeof(CmModel));
    for (int s = 0; s < CM_MAX_SHARDS; s++) m->fds[s] = -1;
    m->rope_theta = 10000.0f; m->rms_eps = 1e-5f; m->bos_id = 1; m->eos_id = 2; m->expert_weights_scale = 1.0f;
    m->use_nocache = (int)use_nocache;
    m->enable_overlap = (int)enable_overlap;

    const char* path0 = (const char*)(uintptr_t)path_cstr;
    char shard_paths[CM_MAX_SHARDS][1024];
    strncpy(shard_paths[0], path0, sizeof(shard_paths[0]) - 1);

    CmTensorInfo* tlist = NULL; uint64_t tcount = 0; int split_count = 1;
    if (!cm_parse_shard(shard_paths[0], 0, 1, m, &tlist, &tcount, &split_count)) { free(tlist); return 0; }
    if (split_count > CM_MAX_SHARDS) split_count = CM_MAX_SHARDS;
    m->n_shards = split_count;
    for (int s = 1; s < split_count; s++) {
        const char* mk = strstr(path0, "-00001-of-");
        if (!mk) { free(tlist); return 0; }
        size_t pref = (size_t)(mk - path0);
        snprintf(shard_paths[s], sizeof(shard_paths[s]), "%.*s-%05d-of-%s", (int)pref, path0, s + 1, mk + 10);
        if (!cm_parse_shard(shard_paths[s], s, 0, m, &tlist, &tcount, NULL)) { free(tlist); return 0; }
    }

    if (m->nextn_layers > 0 && m->n_layers > m->nextn_layers) m->n_layers -= m->nextn_layers;
    if (m->n_layers > CM_MAX_LAYERS) m->n_layers = CM_MAX_LAYERS;
    if (m->n_kv <= 0) m->n_kv = m->n_heads;
    if (m->d_head <= 0 && m->n_heads > 0) m->d_head = m->d_model / m->n_heads;
    if (m->v_head_dim <= 0) m->v_head_dim = m->d_head;
    if (m->rope_dim <= 0) m->rope_dim = m->d_head;
    if (m->d_ff_exp <= 0) m->d_ff_exp = m->d_ff_dense;
    if (m->d_ff_dense <= 0) m->d_ff_dense = m->d_ff_exp;
    m->rope_neox = (strncmp(m->arch, "qwen", 4) == 0 || strstr(m->arch, "neox") != NULL) ? 1 : 0;
    m->dense_model = (m->n_experts <= 0) ? 1 : 0;
    {
        const char* fdp = getenv("TENON_CUDA_FAST_DP4A");
        g_cu.fast_dp4a = (fdp && atoi(fdp) > 0) ? 1 : 0;
        m->fast_dp4a = g_cu.fast_dp4a;
        const char* ctx_e = getenv("TENON_MAX_CTX");
        int ctx = (ctx_e && atoi(ctx_e) > 0) ? atoi(ctx_e) : CM_DEFAULT_SEQ;
        if (ctx > CM_MAX_SEQ) ctx = CM_MAX_SEQ;
        if (ctx < 128) ctx = 128;
        m->max_ctx = ctx;
        const char* kvq_e = getenv("TENON_KV_QUANT");
        if (kvq_e && (strcmp(kvq_e, "q8") == 0 || strcmp(kvq_e, "q8_0") == 0 || strcmp(kvq_e, "1") == 0)) m->kv_quant_q8 = 1;
        else if (kvq_e && (strcmp(kvq_e, "fp16") == 0 || strcmp(kvq_e, "0") == 0)) m->kv_quant_q8 = 0;
        else m->kv_quant_q8 = (m->max_ctx > 2048) ? 1 : 0;
        const char* mla_e = getenv("TENON_MLA_LATENT_KV");
        m->mla_latent_kv = (mla_e && atoi(mla_e) == 0) ? 0 : 1;
        const char* kvs_e = getenv("TENON_KV_SLOTS");
        m->n_kv_slots = (kvs_e && atoi(kvs_e) > 1) ? atoi(kvs_e) : 1;
        if (m->n_kv_slots > 16) m->n_kv_slots = 16;
        m->cur_kv_slot = 0;
        m->slot_ctx = m->max_ctx / m->n_kv_slots;
        if (m->slot_ctx < 64) { m->n_kv_slots = 1; m->slot_ctx = m->max_ctx; }
        int req_tp = 1;
        const char* tp_e = getenv("TENON_TP_SIZE");
        if (tp_e && atoi(tp_e) > 1) req_tp = atoi(tp_e);
        if (req_tp > g_cu.n_gpus) req_tp = g_cu.n_gpus;
        int req_ep = 1;
        const char* ep_e = getenv("TENON_EP_SIZE");
        if (ep_e && atoi(ep_e) > 1) req_ep = atoi(ep_e);
        if (req_ep > g_cu.n_gpus) req_ep = g_cu.n_gpus;
        int all_rdy_tp = 1;
        for (int r = 0; r < req_tp; r++) if (!g_cu.gpus[r].ready) all_rdy_tp = 0;
        int all_rdy_ep = 1;
        for (int r = 0; r < req_ep; r++) if (!g_cu.gpus[r].ready) all_rdy_ep = 0;

        if (req_tp > 1 && m->dense_model && all_rdy_tp &&
            (m->n_heads % req_tp == 0) && (m->n_kv % req_tp == 0) && (m->d_ff_dense % req_tp == 0)) {
            m->tp_size = req_tp;
            m->ep_size = 1;
            m->ep_exp_per_rank = 0;
        } else if (req_tp > 1 && req_ep == req_tp && !m->dense_model && all_rdy_tp &&
                   m->n_experts > 0 && (m->n_experts % req_ep == 0) &&
                   (m->n_heads % req_tp == 0) && (m->n_kv % req_tp == 0) &&
                   (m->d_ff_dense <= 0 || m->d_ff_dense % req_tp == 0)) {
            // Tenon v7: True Co-Located 2D Hybrid Parallelism (TP Attention x EP Experts!)
            m->tp_size = req_tp;
            m->ep_size = req_ep;
            m->ep_exp_per_rank = m->n_experts / req_ep;
            m->hybrid_2d_mode = 1;
        } else if (req_ep > 1 && !m->dense_model && all_rdy_ep && m->n_experts > 0 && (m->n_experts % req_ep == 0)) {
            m->tp_size = 1;
            m->ep_size = req_ep;
            m->ep_exp_per_rank = m->n_experts / req_ep;
        } else {
            m->tp_size = 1;
            m->ep_size = 1;
            m->ep_exp_per_rank = m->n_experts;
        }
    }

    if (m->hybrid_2d_mode) {
        m->resident_mode = 1;
        m->cache_slots_per_layer = m->ep_exp_per_rank;
        m->dense_pin_all_attn = 0;
    } else if (m->tp_size > 1) {
        m->resident_mode = 1;
        m->cache_slots_per_layer = m->n_layers;
        m->dense_pin_all_attn = 0;
    } else if (m->ep_size > 1) {
        m->resident_mode = 1;
        m->cache_slots_per_layer = m->ep_exp_per_rank;
    } else if (m->dense_model) {
        m->resident_mode = (cache_slots_per_layer < 0 || cache_slots_per_layer >= m->n_layers) ? 1 : 0;
        m->cache_slots_per_layer = m->resident_mode ? m->n_layers : (int)cache_slots_per_layer;
        m->dense_pin_all_attn = (!m->resident_mode && m->cache_slots_per_layer > 0) ? 1 : 0;
    } else {
        m->resident_mode = (cache_slots_per_layer <= 0 || cache_slots_per_layer >= m->n_experts) ? 1 : 0;
        m->cache_slots_per_layer = m->resident_mode ? m->n_experts : (int)cache_slots_per_layer;
    }

    for (int s = 0; s < m->n_shards; s++) {
        struct stat st;
        if (stat(shard_paths[s], &st) != 0) { free(tlist); return 0; }
        m->shard_sizes[s] = (size_t)st.st_size;
        m->total_disk_bytes += (uint64_t)st.st_size;
        m->fds[s] = open(shard_paths[s], O_RDONLY);
        if (m->fds[s] < 0) { free(tlist); return 0; }
        if (!m->use_nocache) {
            void* mp = mmap(NULL, m->shard_sizes[s], PROT_READ, MAP_PRIVATE, m->fds[s], 0);
            if (mp != MAP_FAILED) m->mmap_ptrs[s] = (uint8_t*)mp;
        }
    }

    for (int L = 0; L < m->n_layers; L++) {
        m->layers[L].is_dense = m->dense_model ? 1 : 0;
        m->layers[L].is_pinned_dense = (m->dense_model && (m->resident_mode || L < m->cache_slots_per_layer)) ? 1 : 0;
    }
    for (uint64_t i = 0; i < tcount; i++) {
        if (strncmp(tlist[i].name, "blk.", 4) == 0 && strstr(tlist[i].name, ".ffn_gate.weight") != NULL) {
            int L = atoi(tlist[i].name + 4);
            if (L >= 0 && L < m->n_layers) { m->layers[L].is_dense = 1; if (!m->dense_model) m->layers[L].is_pinned_dense = 1; }
        }
    }

    if (m->tp_size > 1) {
        int P = m->tp_size;
        m->tp = (CmTpRank*)calloc((size_t)P, sizeof(CmTpRank));
        m->tp_p2p_ok = (g_cu.cuMemcpyPeerAsync != NULL) ? 1 : 0;
        for (int r = 0; r < P; r++) {
            m->tp[r].dev_idx = r;
            g_cu.cuCtxSetCurrent(g_cu.gpus[r].ctx);
            g_cu.cuStreamCreate(&m->tp[r].stream, CU_STREAM_NON_BLOCKING);
            int n_ev = m->n_layers * 2 + 4;
            for (int e = 0; e < n_ev; e++) {
                g_cu.cuEventCreate(&m->tp[r].ev_sent[e], CU_EVENT_DISABLE_TIMING);
                g_cu.cuEventCreate(&m->tp[r].ev_done[e], CU_EVENT_DISABLE_TIMING);
            }
            for (int s = 0; s < P; s++) if (s != r && !g_cu.p2p_matrix[r][s]) m->tp_p2p_ok = 0;
        }
        if (g_cu.nccl_avail && g_cu.ncclCommInitAll) {
            int devs[CM_MAX_TP];
            for (int r = 0; r < P; r++) devs[r] = r;
            if (g_cu.ncclCommInitAll(m->tp_nccl, P, devs) == NCCL_SUCCESS) m->tp_nccl_ok = 1;
        }
        g_cu.cuCtxSetCurrent(g_cu.ctx);
    }
    if (m->ep_size > 1) {
        int EP = m->ep_size;
        m->ep = (CmEpRank*)calloc((size_t)EP, sizeof(CmEpRank));
        for (int r = 0; r < EP; r++) {
            m->ep[r].dev_idx = r;
            m->ep[r].exp_start = r * m->ep_exp_per_rank;
            m->ep[r].exp_count = m->ep_exp_per_rank;
            g_cu.cuCtxSetCurrent(g_cu.gpus[r].ctx);
            g_cu.cuStreamCreate(&m->ep[r].stream, CU_STREAM_NON_BLOCKING);
            g_cu.cuEventCreate(&m->ep[r].ev_done, CU_EVENT_DISABLE_TIMING);
            g_cu.cuMemAlloc(&m->ep[r].d_xn,   (size_t)m->d_model * 4u);
            g_cu.cuMemAlloc(&m->ep[r].d_eidx, (size_t)(CM_MAX_TOPK * 2) * 4u);
            g_cu.cuMemAlloc(&m->ep[r].d_ewt,  (size_t)(CM_MAX_TOPK * 2) * 4u);
            g_cu.cuMemAlloc(&m->ep[r].d_h,    (size_t)(m->top_k * m->d_ff_exp > 64 ? m->top_k * m->d_ff_exp : 64) * 4u);
            g_cu.cuMemAlloc(&m->ep[r].d_part, (size_t)m->d_model * 4u);
        }
        if (g_cu.nccl_avail && g_cu.ncclCommInitAll) {
            int devs[CM_MAX_TP];
            for (int r = 0; r < EP; r++) devs[r] = r;
            if (g_cu.ncclCommInitAll(m->ep_nccl, EP, devs) == NCCL_SUCCESS) m->ep_nccl_ok = 1;
        }
        g_cu.cuCtxSetCurrent(g_cu.ctx);
        for (int r = 0; r < EP; r++) {
            g_cu.cuMemHostAlloc((void**)&m->ep[r].h_eidx, (size_t)(CM_MAX_TOPK * 2) * 4u, CU_MEMHOSTALLOC_PORTABLE);
            g_cu.cuMemHostAlloc((void**)&m->ep[r].h_ewt,  (size_t)(CM_MAX_TOPK * 2) * 4u, CU_MEMHOSTALLOC_PORTABLE);
            g_cu.cuMemHostAlloc((void**)&m->ep[r].h_part, (size_t)m->d_model * 4u,        CU_MEMHOSTALLOC_PORTABLE);
        }
    }

    for (uint64_t i = 0; i < tcount; i++) {
        const CmTensorInfo* ti = &tlist[i];
        if (strcmp(ti->name, "token_embd.weight") == 0) {
            if (m->vocab_size <= 0) m->vocab_size = (int)ti->ne[1];
            if (m->resident_mode) cm_load_pinned_vram(m, &m->tok_embd, ti);
            else cm_record_meta(&m->tok_embd, ti);
        } else if (strcmp(ti->name, "output_norm.weight") == 0) cm_load_pinned_vram(m, &m->out_norm, ti);
        else if (strcmp(ti->name, "output.weight") == 0) cm_load_pinned_vram(m, &m->lm_head, ti);
        else if (strcmp(ti->name, "rope_freqs.weight") == 0) {
            cm_load_pinned_vram(m, &m->rope_freqs, ti); m->has_rope_freqs = 1;
            if (m->tp_size > 1) {
                for (int r = 0; r < m->tp_size; r++) cm_tp_load_replicated(m, &m->tp[r].rope_freqs, r, ti);
                g_cu.cuCtxSetCurrent(g_cu.ctx);
            }
        } else if (strncmp(ti->name, "nextn.", 6) == 0 || strncmp(ti->name, "blk.", 4) == 0) {
            int L = -1; const char* sub = NULL;
            if (strncmp(ti->name, "nextn.", 6) == 0) {
                L = m->n_layers; sub = ti->name + 6;
            } else {
                L = atoi(ti->name + 4);
                const char* dot = strchr(ti->name + 4, '.');
                if (!dot) continue;
                sub = dot + 1;
            }
            if (L == m->n_layers && (m->nextn_layers > 0 || strncmp(ti->name, "nextn.", 6) == 0)) {
                if (strncmp(sub, "nextn.", 6) == 0) sub += 6;
                if (!m->mtp_layer) m->mtp_layer = (CmLayer*)calloc(1, sizeof(CmLayer));
                CmLayer* ml = m->mtp_layer;
                if (strcmp(sub, "enorm.weight") == 0) { cm_load_pinned_vram(m, &m->mtp_enorm, ti); m->has_mtp = 1; }
                else if (strcmp(sub, "hnorm.weight") == 0) { cm_load_pinned_vram(m, &m->mtp_hnorm, ti); m->has_mtp = 1; }
                else if (strcmp(sub, "eh_proj.weight") == 0) { cm_load_pinned_vram(m, &m->mtp_eh_proj, ti); m->has_mtp = 1; }
                else if (strcmp(sub, "shared_head_norm.weight") == 0) { /* uses model out_norm */ }
                else if (strcmp(sub, "attn_norm.weight") == 0) { cm_load_pinned_vram(m, &ml->attn_norm, ti); m->has_mtp_layer = 1; }
                else if (strcmp(sub, "ffn_norm.weight") == 0 || strcmp(sub, "post_attention_norm.weight") == 0) cm_load_pinned_vram(m, &ml->ffn_norm, ti);
                else if (strcmp(sub, "attn_q.weight") == 0) cm_load_pinned_vram(m, &ml->wq, ti);
                else if (strcmp(sub, "attn_k.weight") == 0) cm_load_pinned_vram(m, &ml->wk, ti);
                else if (strcmp(sub, "attn_v.weight") == 0) cm_load_pinned_vram(m, &ml->wv, ti);
                else if (strcmp(sub, "attn_output.weight") == 0) cm_load_pinned_vram(m, &ml->wo, ti);
                else if (strcmp(sub, "attn_q_a.weight") == 0) { cm_load_pinned_vram(m, &ml->wq_a, ti); ml->is_mla = 1; }
                else if (strcmp(sub, "attn_q_a_norm.weight") == 0) cm_load_pinned_vram(m, &ml->q_a_norm, ti);
                else if (strcmp(sub, "attn_q_b.weight") == 0) { cm_load_pinned_vram(m, &ml->wq_b, ti); ml->is_mla = 1; }
                else if (strcmp(sub, "attn_kv_a_mqa.weight") == 0) { cm_load_pinned_vram(m, &ml->wkv_a, ti); ml->is_mla = 1; }
                else if (strcmp(sub, "attn_kv_a_norm.weight") == 0) cm_load_pinned_vram(m, &ml->kv_a_norm, ti);
                else if (strcmp(sub, "attn_kv_b.weight") == 0) { cm_load_pinned_vram(m, &ml->wkv_b, ti); ml->is_mla = 1; }
                else if (strcmp(sub, "attn_k_b.weight") == 0) { cm_load_pinned_vram(m, &ml->wk_b, ti); ml->is_mla = 1; }
                else if (strcmp(sub, "attn_v_b.weight") == 0) { cm_load_pinned_vram(m, &ml->wv_b, ti); ml->is_mla = 1; }
                else if (strcmp(sub, "ffn_gate.weight") == 0) { cm_load_pinned_vram(m, &ml->ffn_gate, ti); ml->is_dense = 1; }
                else if (strcmp(sub, "ffn_up.weight") == 0) cm_load_pinned_vram(m, &ml->ffn_up, ti);
                else if (strcmp(sub, "ffn_down.weight") == 0) cm_load_pinned_vram(m, &ml->ffn_down, ti);
                else if (strcmp(sub, "ffn_gate_inp.weight") == 0) cm_load_pinned_vram(m, &ml->router, ti);
                else if (strcmp(sub, "exp_probs_b.bias") == 0) { cm_load_pinned_vram(m, &ml->exp_bias, ti); ml->has_exp_bias = 1; }
                else if (strcmp(sub, "ffn_gate_exps.weight") == 0) { cm_record_meta(&ml->gate_exps, ti); ml->gate_exp_bytes = (m->n_experts > 0) ? (ti->byte_size / (uint64_t)m->n_experts) : ti->byte_size; ml->is_dense = 0; }
                else if (strcmp(sub, "ffn_up_exps.weight") == 0) { cm_record_meta(&ml->up_exps, ti); ml->up_exp_bytes = (m->n_experts > 0) ? (ti->byte_size / (uint64_t)m->n_experts) : ti->byte_size; }
                else if (strcmp(sub, "ffn_down_exps.weight") == 0) { cm_record_meta(&ml->down_exps, ti); ml->down_exp_bytes = (m->n_experts > 0) ? (ti->byte_size / (uint64_t)m->n_experts) : ti->byte_size; }
                else if (strcmp(sub, "ffn_gate_shexp.weight") == 0) { cm_load_pinned_vram(m, &ml->sh_gate, ti); ml->has_shared_exp = 1; }
                else if (strcmp(sub, "ffn_up_shexp.weight") == 0) cm_load_pinned_vram(m, &ml->sh_up, ti);
                else if (strcmp(sub, "ffn_down_shexp.weight") == 0) cm_load_pinned_vram(m, &ml->sh_down, ti);
                continue;
            }
            if (L < 0 || L >= m->n_layers) continue;
            if (m->tp_size > 1 && !m->hybrid_2d_mode) {
                int P = m->tp_size;
                if (strcmp(sub, "attn_norm.weight") == 0) { for (int r = 0; r < P; r++) cm_tp_load_replicated(m, &m->tp[r].layers[L].attn_norm, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); }
                else if (strcmp(sub, "ffn_norm.weight") == 0 || strcmp(sub, "post_attention_norm.weight") == 0) { for (int r = 0; r < P; r++) cm_tp_load_replicated(m, &m->tp[r].layers[L].ffn_norm, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); }
                else if (strcmp(sub, "attn_q.weight") == 0) { for (int r = 0; r < P; r++) cm_tp_load_row_shard(m, &m->tp[r].layers[L].wq, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); }
                else if (strcmp(sub, "attn_k.weight") == 0) { for (int r = 0; r < P; r++) cm_tp_load_row_shard(m, &m->tp[r].layers[L].wk, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); }
                else if (strcmp(sub, "attn_v.weight") == 0) { for (int r = 0; r < P; r++) cm_tp_load_row_shard(m, &m->tp[r].layers[L].wv, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); }
                else if (strcmp(sub, "attn_q.bias") == 0)   { for (int r = 0; r < P; r++) cm_tp_load_row_shard(m, &m->tp[r].layers[L].bq, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); }
                else if (strcmp(sub, "attn_k.bias") == 0)   { for (int r = 0; r < P; r++) cm_tp_load_row_shard(m, &m->tp[r].layers[L].bk, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); }
                else if (strcmp(sub, "attn_v.bias") == 0)   { for (int r = 0; r < P; r++) cm_tp_load_row_shard(m, &m->tp[r].layers[L].bv, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); }
                else if (strcmp(sub, "attn_output.weight") == 0) cm_tp_load_col_shard_all(m, ti, L, 0);
                else if (strcmp(sub, "ffn_gate.weight") == 0) { for (int r = 0; r < P; r++) cm_tp_load_row_shard(m, &m->tp[r].layers[L].ffn_gate, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); }
                else if (strcmp(sub, "ffn_up.weight") == 0)   { for (int r = 0; r < P; r++) cm_tp_load_row_shard(m, &m->tp[r].layers[L].ffn_up, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); }
                else if (strcmp(sub, "ffn_down.weight") == 0) cm_tp_load_col_shard_all(m, ti, L, 1);
                continue;
            }
            if (m->hybrid_2d_mode) {
                int P = m->tp_size;
                CmLayer* ly = &m->layers[L];
                if (strcmp(sub, "attn_norm.weight") == 0) { for (int r = 0; r < P; r++) cm_tp_load_replicated(m, &m->tp[r].layers[L].attn_norm, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); continue; }
                else if (strcmp(sub, "ffn_norm.weight") == 0 || strcmp(sub, "post_attention_norm.weight") == 0) { for (int r = 0; r < P; r++) cm_tp_load_replicated(m, &m->tp[r].layers[L].ffn_norm, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); continue; }
                else if (strcmp(sub, "attn_q.weight") == 0) { for (int r = 0; r < P; r++) cm_tp_load_row_shard(m, &m->tp[r].layers[L].wq, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); continue; }
                else if (strcmp(sub, "attn_k.weight") == 0) { for (int r = 0; r < P; r++) cm_tp_load_row_shard(m, &m->tp[r].layers[L].wk, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); continue; }
                else if (strcmp(sub, "attn_v.weight") == 0) { for (int r = 0; r < P; r++) cm_tp_load_row_shard(m, &m->tp[r].layers[L].wv, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); continue; }
                else if (strcmp(sub, "attn_q.bias") == 0)   { for (int r = 0; r < P; r++) cm_tp_load_row_shard(m, &m->tp[r].layers[L].bq, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); continue; }
                else if (strcmp(sub, "attn_k.bias") == 0)   { for (int r = 0; r < P; r++) cm_tp_load_row_shard(m, &m->tp[r].layers[L].bk, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); continue; }
                else if (strcmp(sub, "attn_v.bias") == 0)   { for (int r = 0; r < P; r++) cm_tp_load_row_shard(m, &m->tp[r].layers[L].bv, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); continue; }
                else if (strcmp(sub, "attn_q_a.weight") == 0) { for (int r = 0; r < P; r++) { cm_tp_load_replicated(m, &m->tp[r].layers[L].wq_a, r, ti); m->tp[r].layers[L].is_mla = 1; } ly->is_mla = 1; g_cu.cuCtxSetCurrent(g_cu.ctx); continue; }
                else if (strcmp(sub, "attn_q_a_norm.weight") == 0) { for (int r = 0; r < P; r++) cm_tp_load_replicated(m, &m->tp[r].layers[L].q_a_norm, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); continue; }
                else if (strcmp(sub, "attn_q_b.weight") == 0) { for (int r = 0; r < P; r++) { cm_tp_load_row_shard(m, &m->tp[r].layers[L].wq_b, r, ti); m->tp[r].layers[L].is_mla = 1; } ly->is_mla = 1; g_cu.cuCtxSetCurrent(g_cu.ctx); continue; }
                else if (strcmp(sub, "attn_kv_a_mqa.weight") == 0) { for (int r = 0; r < P; r++) { cm_tp_load_replicated(m, &m->tp[r].layers[L].wkv_a, r, ti); m->tp[r].layers[L].is_mla = 1; } ly->is_mla = 1; g_cu.cuCtxSetCurrent(g_cu.ctx); continue; }
                else if (strcmp(sub, "attn_kv_a_norm.weight") == 0) { for (int r = 0; r < P; r++) cm_tp_load_replicated(m, &m->tp[r].layers[L].kv_a_norm, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); continue; }
                else if (strcmp(sub, "attn_kv_b.weight") == 0) { cm_load_pinned_vram(m, &ly->wkv_b, ti); ly->is_mla = 1; for (int r = 0; r < P; r++) m->tp[r].layers[L].is_mla = 1; continue; }
                else if (strcmp(sub, "attn_k_b.weight") == 0)  { cm_load_pinned_vram(m, &ly->wk_b, ti); ly->is_mla = 1; for (int r = 0; r < P; r++) m->tp[r].layers[L].is_mla = 1; continue; }
                else if (strcmp(sub, "attn_v_b.weight") == 0)  { cm_load_pinned_vram(m, &ly->wv_b, ti); ly->is_mla = 1; for (int r = 0; r < P; r++) m->tp[r].layers[L].is_mla = 1; continue; }
                else if (strcmp(sub, "attn_output.weight") == 0) { cm_tp_load_col_shard_all(m, ti, L, 0); continue; }
                else if (strcmp(sub, "ffn_gate.weight") == 0) { for (int r = 0; r < P; r++) cm_tp_load_row_shard(m, &m->tp[r].layers[L].ffn_gate, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); continue; }
                else if (strcmp(sub, "ffn_up.weight") == 0)   { for (int r = 0; r < P; r++) cm_tp_load_row_shard(m, &m->tp[r].layers[L].ffn_up, r, ti); g_cu.cuCtxSetCurrent(g_cu.ctx); continue; }
                else if (strcmp(sub, "ffn_down.weight") == 0) { cm_tp_load_col_shard_all(m, ti, L, 1); continue; }
                // Remaining MoE router/exp_bias/exps/shexp fall through to standard `ly` binding below!
            }
            CmLayer* ly = &m->layers[L];
            int stream_dense_w = (m->dense_model && !ly->is_pinned_dense);
            if (strcmp(sub, "attn_norm.weight") == 0) cm_load_pinned_vram(m, &ly->attn_norm, ti);
            else if (strcmp(sub, "ffn_norm.weight") == 0 || strcmp(sub, "post_attention_norm.weight") == 0) cm_load_pinned_vram(m, &ly->ffn_norm, ti);
            else if (strcmp(sub, "attn_q.bias") == 0) { cm_load_pinned_vram(m, &ly->bq, ti); ly->has_qkv_bias = 1; }
            else if (strcmp(sub, "attn_k.bias") == 0) { cm_load_pinned_vram(m, &ly->bk, ti); ly->has_qkv_bias = 1; }
            else if (strcmp(sub, "attn_v.bias") == 0) { cm_load_pinned_vram(m, &ly->bv, ti); ly->has_qkv_bias = 1; }
            else if (strcmp(sub, "attn_q_norm.weight") == 0) { cm_load_pinned_vram(m, &ly->q_norm, ti); ly->has_qk_norm = 1; ly->qk_norm_per_head = ((int)ti->ne[0] == m->d_head) ? 1 : 0; }
            else if (strcmp(sub, "attn_k_norm.weight") == 0) cm_load_pinned_vram(m, &ly->k_norm, ti);
            else if (strcmp(sub, "attn_q.weight") == 0) { if (stream_dense_w && !m->dense_pin_all_attn) { cm_record_meta(&ly->wq, ti); m->total_streamable_bytes += (int64_t)ti->byte_size; } else cm_load_pinned_vram(m, &ly->wq, ti); }
            else if (strcmp(sub, "attn_k.weight") == 0) { if (stream_dense_w && !m->dense_pin_all_attn) { cm_record_meta(&ly->wk, ti); m->total_streamable_bytes += (int64_t)ti->byte_size; } else cm_load_pinned_vram(m, &ly->wk, ti); }
            else if (strcmp(sub, "attn_v.weight") == 0) { if (stream_dense_w && !m->dense_pin_all_attn) { cm_record_meta(&ly->wv, ti); m->total_streamable_bytes += (int64_t)ti->byte_size; } else cm_load_pinned_vram(m, &ly->wv, ti); }
            else if (strcmp(sub, "attn_q_a.weight") == 0) { cm_load_pinned_vram(m, &ly->wq_a, ti); ly->is_mla = 1; }
            else if (strcmp(sub, "attn_q_a_norm.weight") == 0) { cm_load_pinned_vram(m, &ly->q_a_norm, ti); }
            else if (strcmp(sub, "attn_q_b.weight") == 0) { cm_load_pinned_vram(m, &ly->wq_b, ti); ly->is_mla = 1; }
            else if (strcmp(sub, "attn_kv_a_mqa.weight") == 0) { cm_load_pinned_vram(m, &ly->wkv_a, ti); ly->is_mla = 1; }
            else if (strcmp(sub, "attn_kv_a_norm.weight") == 0) { cm_load_pinned_vram(m, &ly->kv_a_norm, ti); }
            else if (strcmp(sub, "attn_kv_b.weight") == 0) { cm_load_pinned_vram(m, &ly->wkv_b, ti); ly->is_mla = 1; }
            else if (strcmp(sub, "attn_k_b.weight") == 0)  { cm_load_pinned_vram(m, &ly->wk_b, ti); ly->is_mla = 1; }
            else if (strcmp(sub, "attn_v_b.weight") == 0)  { cm_load_pinned_vram(m, &ly->wv_b, ti); ly->is_mla = 1; }
            else if (strcmp(sub, "attn_output.weight") == 0) { if (stream_dense_w && !m->dense_pin_all_attn) { cm_record_meta(&ly->wo, ti); m->total_streamable_bytes += (int64_t)ti->byte_size; } else cm_load_pinned_vram(m, &ly->wo, ti); }
            else if (strcmp(sub, "ffn_gate.weight") == 0) { if (stream_dense_w) { cm_record_meta(&ly->ffn_gate, ti); m->total_streamable_bytes += (int64_t)ti->byte_size; } else cm_load_pinned_vram(m, &ly->ffn_gate, ti); }
            else if (strcmp(sub, "ffn_up.weight") == 0)   { if (stream_dense_w) { cm_record_meta(&ly->ffn_up, ti);   m->total_streamable_bytes += (int64_t)ti->byte_size; } else cm_load_pinned_vram(m, &ly->ffn_up, ti); }
            else if (strcmp(sub, "ffn_down.weight") == 0) { if (stream_dense_w) { cm_record_meta(&ly->ffn_down, ti); m->total_streamable_bytes += (int64_t)ti->byte_size; } else cm_load_pinned_vram(m, &ly->ffn_down, ti); }
            else if (strcmp(sub, "ffn_gate_inp.weight") == 0) cm_load_pinned_vram(m, &ly->router, ti);
            else if (strcmp(sub, "exp_probs_b.bias") == 0) {
                cm_load_pinned_vram(m, &ly->exp_bias, ti);
                ly->exp_bias_host = (float*)malloc((size_t)ti->byte_size);
                cm_pread_exact(m->fds[ti->shard_idx], ly->exp_bias_host, (size_t)ti->byte_size, ti->file_off);
                ly->has_exp_bias = 1;
            }
            else if (strcmp(sub, "ffn_gate_exps.weight") == 0) { cm_record_meta(&ly->gate_exps, ti); ly->gate_exp_bytes = ti->byte_size / (uint64_t)m->n_experts; m->total_streamable_bytes += (int64_t)ti->byte_size; }
            else if (strcmp(sub, "ffn_up_exps.weight") == 0)   { cm_record_meta(&ly->up_exps, ti);   ly->up_exp_bytes = ti->byte_size / (uint64_t)m->n_experts;   m->total_streamable_bytes += (int64_t)ti->byte_size; }
            else if (strcmp(sub, "ffn_down_exps.weight") == 0) { cm_record_meta(&ly->down_exps, ti); ly->down_exp_bytes = ti->byte_size / (uint64_t)m->n_experts; m->total_streamable_bytes += (int64_t)ti->byte_size; }
            else if (strcmp(sub, "ffn_gate_shexp.weight") == 0) { cm_load_pinned_vram(m, &ly->sh_gate, ti); ly->has_shared_exp = 1; }
            else if (strcmp(sub, "ffn_up_shexp.weight") == 0)   cm_load_pinned_vram(m, &ly->sh_up, ti);
            else if (strcmp(sub, "ffn_down_shexp.weight") == 0) cm_load_pinned_vram(m, &ly->sh_down, ti);
        }
    }
    free(tlist);
    if (!m->lm_head.dptr && m->tok_embd.dptr) m->lm_head = m->tok_embd;
    if (m->yarn_factor > 1.0f && !m->has_rope_freqs && m->rope_dim > 0) {
        int half_r = m->rope_dim >> 1;
        float orig = (m->yarn_orig_ctx > 0) ? (float)m->yarn_orig_ctx : 4096.0f;
        float lb = 2.0f * logf(m->rope_theta > 1.0f ? m->rope_theta : 10000.0f);
        float low  = floorf((float)m->rope_dim * logf(orig / (32.0f * 2.0f * 3.1415926535f)) / lb);
        float high = ceilf((float)m->rope_dim  * logf(orig / (1.0f  * 2.0f * 3.1415926535f)) / lb);
        if (low < 0.0f) low = 0.0f;
        if (high > (float)(m->rope_dim - 1)) high = (float)(m->rope_dim - 1);
        float* rf_h = (float*)malloc((size_t)half_r * sizeof(float));
        for (int i = 0; i < half_r; i++) {
            float g = (high > low) ? (((float)i - low) / (high - low)) : 0.0f;
            if (g < 0.0f) g = 0.0f;
            if (g > 1.0f) g = 1.0f;
            float mult = (1.0f - g) + g / m->yarn_factor;
            rf_h[i] = 1.0f / (mult > 1e-6f ? mult : 1e-6f);
        }
        if (g_cu.cuMemAlloc(&m->rope_freqs.dptr, (size_t)half_r * sizeof(float)) == CUDA_SUCCESS) {
            g_cu.cuMemcpyHtoD(m->rope_freqs.dptr, rf_h, (size_t)half_r * sizeof(float));
            m->has_rope_freqs = 1;
            if (m->tp_size > 1 && m->tp) {
                for (int r = 0; r < m->tp_size; r++) {
                    g_cu.cuCtxSetCurrent(g_cu.gpus[r].ctx);
                    if (g_cu.cuMemAlloc(&m->tp[r].rope_freqs.dptr, (size_t)half_r * sizeof(float)) == CUDA_SUCCESS) {
                        g_cu.cuMemcpyHtoD(m->tp[r].rope_freqs.dptr, rf_h, (size_t)half_r * sizeof(float));
                    }
                }
                g_cu.cuCtxSetCurrent(g_cu.ctx);
            }
        }
        free(rf_h);
    }

    g_cu.cuStreamCreate(&m->compute_stream, CU_STREAM_NON_BLOCKING);
    g_cu.cuStreamCreate(&m->dma_stream, CU_STREAM_NON_BLOCKING);
    g_cu.cuEventCreate(&m->ev_dma_done, CU_EVENT_DISABLE_TIMING);

    if (m->tp_size > 1) {
        int P = m->tp_size;
        int nkv_l = m->n_kv / P;
        size_t kv_slot_b = m->kv_quant_q8
            ? ((size_t)m->slot_ctx * (size_t)nkv_l * (size_t)(m->d_head >> 5) * 34u)
            : ((size_t)m->slot_ctx * (size_t)(nkv_l * m->d_head) * 2u);
        size_t kv_tot_b = kv_slot_b * (size_t)m->n_kv_slots;
        for (int L = 0; L < m->n_layers; L++) {
            CmLayer* ly0 = &m->layers[L];
            if (m->hybrid_2d_mode && ly0->is_mla && (ly0->wkv_b.dptr || (ly0->wk_b.dptr && ly0->wv_b.dptr))) {
                // Unpack full MLA W_UK/W_UV once on GPU 0, then shard 1/P of heads onto each GPU r (cuts MLA VRAM & compute by P!)
                uint32_t kv_r = (uint32_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
                uint32_t nope_d = (uint32_t)(m->d_head - m->rope_dim), v_d = (uint32_t)m->v_head_dim;
                uint32_t nh_u = (uint32_t)m->n_heads, nh_l = nh_u / (uint32_t)P;
                size_t full_uk_b = (size_t)nh_u * kv_r * nope_d * 2u;
                size_t full_uv_b = (size_t)nh_u * v_d * kv_r * 2u;
                size_t part_uk_b = (size_t)nh_l * kv_r * nope_d * 2u;
                size_t part_uv_b = (size_t)nh_l * v_d * kv_r * 2u;
                CUdeviceptr d_tmp_uk = 0, d_tmp_uv = 0;
                g_cu.cuCtxSetCurrent(g_cu.ctx);
                g_cu.cuMemAlloc(&d_tmp_uk, full_uk_b);
                g_cu.cuMemAlloc(&d_tmp_uv, full_uv_b);
                uint32_t max_j = nope_d > v_d ? nope_d : v_d;
                if (ly0->wk_b.dptr && ly0->wv_b.dptr) {
                    uint32_t k_tr = ((uint32_t)ly0->wk_b.ne0 == kv_r && kv_r != nope_d) ? 1u : 0u;
                    uint32_t v_tr = ((uint32_t)ly0->wv_b.ne0 == v_d  && kv_r != v_d)    ? 1u : 0u;
                    uint32_t gtk = ly0->wk_b.ggml_type, gtv = ly0->wv_b.ggml_type;
                    void* uargs[] = { &ly0->wk_b.dptr, &ly0->wv_b.dptr, &d_tmp_uk, &d_tmp_uv, &gtk, &gtv, &k_tr, &v_tr, &nh_u, &nope_d, &v_d, &kv_r };
                    g_cu.cuLaunchKernel(g_cu.k_mla_unpack_kb_vb, (kv_r + 31u) / 32u, max_j, nh_u, 32, 1, 1, 0, m->compute_stream, uargs, NULL);
                } else {
                    uint32_t gtb = ly0->wkv_b.ggml_type;
                    void* uargs[] = { &ly0->wkv_b.dptr, &d_tmp_uk, &d_tmp_uv, &gtb, &nh_u, &nope_d, &v_d, &kv_r };
                    g_cu.cuLaunchKernel(g_cu.k_mla_unpack_wkv_b, (kv_r + 31u) / 32u, max_j, nh_u, 32, 1, 1, 0, m->compute_stream, uargs, NULL);
                }
                g_cu.cuStreamSynchronize(m->compute_stream);
                uint8_t* h_uk = (uint8_t*)malloc(full_uk_b);
                uint8_t* h_uv = (uint8_t*)malloc(full_uv_b);
                g_cu.cuMemcpyDtoH(h_uk, d_tmp_uk, full_uk_b);
                g_cu.cuMemcpyDtoH(h_uv, d_tmp_uv, full_uv_b);
                g_cu.cuMemFree(d_tmp_uk); g_cu.cuMemFree(d_tmp_uv);
                if (ly0->wkv_b.dptr) { g_cu.cuMemFree(ly0->wkv_b.dptr); ly0->wkv_b.dptr = 0; }
                if (ly0->wk_b.dptr)  { g_cu.cuMemFree(ly0->wk_b.dptr);  ly0->wk_b.dptr = 0; }
                if (ly0->wv_b.dptr)  { g_cu.cuMemFree(ly0->wv_b.dptr);  ly0->wv_b.dptr = 0; }
                size_t lslot_b = (size_t)m->slot_ctx * (size_t)(kv_r + (uint32_t)m->rope_dim) * 2u;
                size_t ltot_b = lslot_b * (size_t)m->n_kv_slots;
                for (int r = 0; r < P; r++) {
                    g_cu.cuCtxSetCurrent(g_cu.gpus[r].ctx);
                    CmTpLayer* tl = &m->tp[r].layers[L];
                    g_cu.cuMemAlloc(&tl->k_cache_base, ltot_b);
                    tl->k_cache = tl->k_cache_base; tl->v_cache = 0; tl->v_cache_base = 0; tl->kv_slot_bytes = lslot_b;
                    g_cu.cuMemAlloc(&tl->mla_wuk_f16, part_uk_b);
                    g_cu.cuMemAlloc(&tl->mla_wuv_f16, part_uv_b);
                    g_cu.cuMemcpyHtoD(tl->mla_wuk_f16, h_uk + (size_t)r * part_uk_b, part_uk_b);
                    g_cu.cuMemcpyHtoD(tl->mla_wuv_f16, h_uv + (size_t)r * part_uv_b, part_uv_b);
                    m->core_vram_bytes += (int64_t)(ltot_b + part_uk_b + part_uv_b);
                }
                free(h_uk); free(h_uv);
            } else {
                for (int r = 0; r < P; r++) {
                    g_cu.cuCtxSetCurrent(g_cu.gpus[r].ctx);
                    CmTpLayer* tl = &m->tp[r].layers[L];
                    g_cu.cuMemAlloc(&tl->k_cache_base, kv_tot_b);
                    g_cu.cuMemAlloc(&tl->v_cache_base, kv_tot_b);
                    tl->k_cache = tl->k_cache_base; tl->v_cache = tl->v_cache_base; tl->kv_slot_bytes = kv_slot_b;
                    m->core_vram_bytes += (int64_t)(kv_tot_b * 2u);
                }
            }
        }
        g_cu.cuCtxSetCurrent(g_cu.ctx);
    }

    size_t kv_slot_std = m->kv_quant_q8
        ? ((size_t)m->slot_ctx * (size_t)m->n_kv * (size_t)(m->d_head >> 5) * 34u)
        : ((size_t)m->slot_ctx * (size_t)(m->n_kv * m->d_head) * 2u);
    size_t kv_bytes_std = kv_slot_std * (size_t)m->n_kv_slots;
    uint64_t max_wq = 0, max_wk = 0, max_wv = 0, max_wo = 0, max_fg = 0, max_fu = 0, max_fd = 0, max_exp_stride = 0;
    uint64_t max_sh_ff = 0;
    for (int L = 0; L < m->n_layers; L++) {
        CmLayer* ly = &m->layers[L];
        if (ly->has_shared_exp && ly->sh_gate.ne1 > max_sh_ff) max_sh_ff = ly->sh_gate.ne1;
        if (m->tp_size == 1) {
            if (ly->is_mla && m->mla_latent_kv && (ly->wkv_b.dptr || (ly->wk_b.dptr && ly->wv_b.dptr))) {
                uint32_t kv_r = (uint32_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
                uint32_t nope_d = (uint32_t)(m->d_head - m->rope_dim), v_d = (uint32_t)m->v_head_dim;
                uint32_t nh_u = (uint32_t)m->n_heads;
                size_t lslot_b = (size_t)m->slot_ctx * (size_t)(kv_r + (uint32_t)m->rope_dim) * 2u;
                size_t ltot_b = lslot_b * (size_t)m->n_kv_slots;
                g_cu.cuMemAlloc(&ly->k_cache_base, ltot_b);
                ly->k_cache = ly->k_cache_base; ly->v_cache = 0; ly->v_cache_base = 0; ly->kv_slot_bytes = lslot_b;
                m->core_vram_bytes += (int64_t)ltot_b;
                size_t wuk_b = (size_t)nh_u * kv_r * nope_d * 2u;
                size_t wuv_b = (size_t)nh_u * v_d * kv_r * 2u;
                g_cu.cuMemAlloc(&ly->mla_wuk_f16, wuk_b);
                g_cu.cuMemAlloc(&ly->mla_wuv_f16, wuv_b);
                uint32_t max_j = nope_d > v_d ? nope_d : v_d;
                if (ly->wk_b.dptr && ly->wv_b.dptr) {
                    uint32_t k_tr = ((uint32_t)ly->wk_b.ne0 == kv_r && kv_r != nope_d) ? 1u : 0u;
                    uint32_t v_tr = ((uint32_t)ly->wv_b.ne0 == v_d  && kv_r != v_d)    ? 1u : 0u;
                    uint32_t gtk = ly->wk_b.ggml_type, gtv = ly->wv_b.ggml_type;
                    void* uargs[] = { &ly->wk_b.dptr, &ly->wv_b.dptr, &ly->mla_wuk_f16, &ly->mla_wuv_f16, &gtk, &gtv, &k_tr, &v_tr, &nh_u, &nope_d, &v_d, &kv_r };
                    g_cu.cuLaunchKernel(g_cu.k_mla_unpack_kb_vb, (kv_r + 31u) / 32u, max_j, nh_u, 32, 1, 1, 0, m->compute_stream, uargs, NULL);
                } else {
                    uint32_t gtb = ly->wkv_b.ggml_type;
                    void* uargs[] = { &ly->wkv_b.dptr, &ly->mla_wuk_f16, &ly->mla_wuv_f16, &gtb, &nh_u, &nope_d, &v_d, &kv_r };
                    g_cu.cuLaunchKernel(g_cu.k_mla_unpack_wkv_b, (kv_r + 31u) / 32u, max_j, nh_u, 32, 1, 1, 0, m->compute_stream, uargs, NULL);
                }
            } else {
                g_cu.cuMemAlloc(&ly->k_cache_base, kv_bytes_std);
                g_cu.cuMemAlloc(&ly->v_cache_base, kv_bytes_std);
                ly->k_cache = ly->k_cache_base; ly->v_cache = ly->v_cache_base; ly->kv_slot_bytes = kv_slot_std;
                m->core_vram_bytes += (int64_t)(kv_bytes_std * 2u);
            }
        }
        for (int e = 0; e < CM_MAX_EXPERTS; e++) ly->exp_to_slot[e] = -1;
        for (int s = 0; s < CM_MAX_EXPERTS; s++) { ly->slot_to_exp[s] = -1; ly->slot_lru[s] = 0; }

        if (m->dense_model && !ly->is_pinned_dense && m->tp_size == 1) {
            if (ly->wq.byte_size > max_wq) max_wq = ly->wq.byte_size;
            if (ly->wk.byte_size > max_wk) max_wk = ly->wk.byte_size;
            if (ly->wv.byte_size > max_wv) max_wv = ly->wv.byte_size;
            if (ly->wo.byte_size > max_wo) max_wo = ly->wo.byte_size;
            if (ly->ffn_gate.byte_size > max_fg) max_fg = ly->ffn_gate.byte_size;
            if (ly->ffn_up.byte_size > max_fu) max_fu = ly->ffn_up.byte_size;
            if (ly->ffn_down.byte_size > max_fd) max_fd = ly->ffn_down.byte_size;
        } else if (!m->dense_model && !ly->is_dense) {
            ly->slot_stride_bytes = cm_align64(ly->gate_exp_bytes + ly->up_exp_bytes + ly->down_exp_bytes);
            if (ly->slot_stride_bytes > max_exp_stride) max_exp_stride = ly->slot_stride_bytes;
            if (m->ep_size > 1 && m->ep) {
                // True Multi-GPU MoE Expert Parallelism: shard `n_experts / EP` experts directly onto each GPU `r`!
                size_t cbytes_r = (size_t)m->ep_exp_per_rank * (size_t)ly->slot_stride_bytes;
                uint8_t* tmp_exp = (uint8_t*)malloc((size_t)ly->slot_stride_bytes);
                for (int r = 0; r < m->ep_size; r++) {
                    g_cu.cuCtxSetCurrent(g_cu.gpus[r].ctx);
                    if (g_cu.cuMemAlloc(&m->ep[r].cache_dptr[L], cbytes_r) != CUDA_SUCCESS) {
                        free(tmp_exp); g_cu.cuCtxSetCurrent(g_cu.ctx);
                        fprintf(stderr, "[tenon-cuda-stream] EP GPU %d OOM allocating layer %d (%zu MiB)\n", r, L, cbytes_r >> 20);
                        return 0;
                    }
                    if (tmp_exp) {
                        for (int ls = 0; ls < m->ep_exp_per_rank; ls++) {
                            int eid = r * m->ep_exp_per_rank + ls;
                            cm_stage_expert(m, L, eid, tmp_exp);
                            g_cu.cuMemcpyHtoD(m->ep[r].cache_dptr[L] + (uint64_t)ls * ly->slot_stride_bytes,
                                              tmp_exp, (size_t)ly->slot_stride_bytes);
                        }
                    }
                    m->cache_vram_bytes += (int64_t)cbytes_r;
                }
                free(tmp_exp);
                g_cu.cuCtxSetCurrent(g_cu.ctx);
            } else {
                size_t cbytes = (size_t)m->cache_slots_per_layer * (size_t)ly->slot_stride_bytes;
                if (g_cu.cuMemAlloc(&ly->cache_dptr, cbytes) != CUDA_SUCCESS) {
                    fprintf(stderr, "[tenon-cuda-stream] VRAM OOM allocating layer %d cache (%zu MiB)\n", L, cbytes >> 20);
                    return 0;
                }
                if (m->resident_mode) {
                    // Tenon v7: Preload ALL resident experts directly into VRAM (`slot == eid`) -> ZERO-sync GPU fast-path!
                    uint8_t* tmp_exp = (uint8_t*)malloc((size_t)ly->slot_stride_bytes);
                    if (tmp_exp) {
                        for (int eid = 0; eid < m->n_experts; eid++) {
                            cm_stage_expert(m, L, eid, tmp_exp);
                            g_cu.cuMemcpyHtoD(ly->cache_dptr + (uint64_t)eid * ly->slot_stride_bytes,
                                              tmp_exp, (size_t)ly->slot_stride_bytes);
                            ly->exp_to_slot[eid] = eid;
                            ly->slot_to_exp[eid] = eid;
                        }
                        free(tmp_exp);
                    }
                }
                m->cache_vram_bytes += (int64_t)cbytes;
            }
        }
    }
    if (m->has_mtp_layer && m->mtp_layer) {
        CmLayer* ml = m->mtp_layer;
        if (ml->is_mla && m->mla_latent_kv && (ml->wkv_b.dptr || (ml->wk_b.dptr && ml->wv_b.dptr))) {
            uint32_t kv_r = (uint32_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
            uint32_t nope_d = (uint32_t)(m->d_head - m->rope_dim), v_d = (uint32_t)m->v_head_dim;
            uint32_t nh_u = (uint32_t)m->n_heads;
            size_t lslot_b = (size_t)m->slot_ctx * (size_t)(kv_r + (uint32_t)m->rope_dim) * 2u;
            g_cu.cuMemAlloc(&ml->k_cache_base, lslot_b * (size_t)m->n_kv_slots);
            ml->k_cache = ml->k_cache_base; ml->v_cache = 0; ml->v_cache_base = 0; ml->kv_slot_bytes = lslot_b;
            g_cu.cuMemAlloc(&ml->mla_wuk_f16, (size_t)nh_u * kv_r * nope_d * 2u);
            g_cu.cuMemAlloc(&ml->mla_wuv_f16, (size_t)nh_u * v_d * kv_r * 2u);
            uint32_t max_j = nope_d > v_d ? nope_d : v_d;
            if (ml->wk_b.dptr && ml->wv_b.dptr) {
                uint32_t k_tr = ((uint32_t)ml->wk_b.ne0 == kv_r && kv_r != nope_d) ? 1u : 0u;
                uint32_t v_tr = ((uint32_t)ml->wv_b.ne0 == v_d  && kv_r != v_d)    ? 1u : 0u;
                uint32_t gtk = ml->wk_b.ggml_type, gtv = ml->wv_b.ggml_type;
                void* uargs[] = { &ml->wk_b.dptr, &ml->wv_b.dptr, &ml->mla_wuk_f16, &ml->mla_wuv_f16, &gtk, &gtv, &k_tr, &v_tr, &nh_u, &nope_d, &v_d, &kv_r };
                g_cu.cuLaunchKernel(g_cu.k_mla_unpack_kb_vb, (kv_r + 31u) / 32u, max_j, nh_u, 32, 1, 1, 0, m->compute_stream, uargs, NULL);
            } else {
                uint32_t gtb = ml->wkv_b.ggml_type;
                void* uargs[] = { &ml->wkv_b.dptr, &ml->mla_wuk_f16, &ml->mla_wuv_f16, &gtb, &nh_u, &nope_d, &v_d, &kv_r };
                g_cu.cuLaunchKernel(g_cu.k_mla_unpack_wkv_b, (kv_r + 31u) / 32u, max_j, nh_u, 32, 1, 1, 0, m->compute_stream, uargs, NULL);
            }
        } else {
            g_cu.cuMemAlloc(&ml->k_cache_base, kv_bytes_std);
            g_cu.cuMemAlloc(&ml->v_cache_base, kv_bytes_std);
            ml->k_cache = ml->k_cache_base; ml->v_cache = ml->v_cache_base; ml->kv_slot_bytes = kv_slot_std;
        }
        if (ml->gate_exps.byte_size > 0 && ml->up_exps.byte_size > 0 && ml->down_exps.byte_size > 0 && m->n_experts > 0) {
            ml->slot_stride_bytes = cm_align64(ml->gate_exp_bytes + ml->up_exp_bytes + ml->down_exp_bytes);
            if (g_cu.cuMemAlloc(&ml->cache_dptr, (size_t)m->n_experts * (size_t)ml->slot_stride_bytes) == CUDA_SUCCESS) {
                uint8_t* tmp_exp = (uint8_t*)malloc((size_t)ml->slot_stride_bytes);
                if (tmp_exp) {
                    for (int eid = 0; eid < m->n_experts; eid++) {
                        cm_read_slice(m, ml->gate_exps.shard_idx, tmp_exp, (size_t)ml->gate_exp_bytes, ml->gate_exps.file_off + (uint64_t)eid * ml->gate_exp_bytes);
                        cm_read_slice(m, ml->up_exps.shard_idx, tmp_exp + ml->gate_exp_bytes, (size_t)ml->up_exp_bytes, ml->up_exps.file_off + (uint64_t)eid * ml->up_exp_bytes);
                        cm_read_slice(m, ml->down_exps.shard_idx, tmp_exp + ml->gate_exp_bytes + ml->up_exp_bytes, (size_t)ml->down_exp_bytes, ml->down_exps.file_off + (uint64_t)eid * ml->down_exp_bytes);
                        g_cu.cuMemcpyHtoD(ml->cache_dptr + (uint64_t)eid * ml->slot_stride_bytes, tmp_exp, (size_t)ml->slot_stride_bytes);
                    }
                    free(tmp_exp);
                }
            }
        }
    }
    g_cu.cuStreamSynchronize(m->compute_stream);

    if (m->dense_model && !m->resident_mode) {
        int use_cpu_ffn = (m->enable_overlap && !m->use_nocache && m->mmap_ptrs[0] != NULL);
        if (!use_cpu_ffn) {
            uint64_t off = 0;
            if (!m->dense_pin_all_attn) {
                m->dense_off_wq = off; off += cm_align64(max_wq);
                m->dense_off_wk = off; off += cm_align64(max_wk);
                m->dense_off_wv = off; off += cm_align64(max_wv);
                m->dense_off_wo = off; off += cm_align64(max_wo);
            }
            m->dense_off_gate = off; off += cm_align64(max_fg);
            m->dense_off_up   = off; off += cm_align64(max_fu);
            m->dense_off_down = off; off += cm_align64(max_fd);
            m->dense_ring_stride = off;
            int n_slots = m->dense_pin_all_attn ? 4 : 2;
            for (int s = 0; s < n_slots; s++) {
                g_cu.cuMemAlloc(&m->dense_ring_dptr[s], (size_t)off);
                g_cu.cuMemHostAlloc((void**)&m->dense_ring_host[s], (size_t)off, CU_MEMHOSTALLOC_PORTABLE);
                g_cu.cuEventCreate(&m->dense_slot_ready[s], CU_EVENT_DISABLE_TIMING);
                g_cu.cuEventCreate(&m->dense_slot_consumed[s], CU_EVENT_DISABLE_TIMING);
            }
            m->cache_vram_bytes = (int64_t)(off * (uint64_t)n_slots);
        }
    } else if (!m->dense_model && max_exp_stride > 0) {
        m->stage_slot_bytes = (size_t)max_exp_stride;
        m->stage_n_slots = CM_STAGE_SLOTS;
        g_cu.cuMemHostAlloc((void**)&m->stage_host, m->stage_slot_bytes * (size_t)m->stage_n_slots, CU_MEMHOSTALLOC_PORTABLE);
        g_cu.cuMemHostAlloc((void**)&m->promo_stage, m->stage_slot_bytes, CU_MEMHOSTALLOC_PORTABLE);
        g_cu.cuEventCreate(&m->ev_promo_done, CU_EVENT_DISABLE_TIMING);
    }

    size_t h_elems = (size_t)m->d_ff_dense;
    if ((size_t)(m->top_k * m->d_ff_exp) > h_elems) h_elems = (size_t)(m->top_k * m->d_ff_exp);
    size_t sh_elems = (size_t)(m->d_ff_dense > m->d_ff_exp ? m->d_ff_dense : m->d_ff_exp);
    if ((size_t)max_sh_ff > sh_elems) sh_elems = (size_t)max_sh_ff;
    size_t mla_elems = (size_t)m->n_heads * (size_t)(m->d_head + m->v_head_dim);
    if (mla_elems > sh_elems) sh_elems = mla_elems;
    g_cu.cuMemAlloc(&m->d_emb_row, 65536);
    g_cu.cuMemAlloc(&m->d_x, (size_t)m->d_model * 4);
    g_cu.cuMemAlloc(&m->d_xn, (size_t)m->d_model * 4);
    g_cu.cuMemAlloc(&m->d_q, (size_t)(m->n_heads * m->d_head) * 4);
    g_cu.cuMemAlloc(&m->d_k, (size_t)(m->n_kv * m->d_head) * 4);
    g_cu.cuMemAlloc(&m->d_v, (size_t)(m->n_kv * m->d_head) * 4);
    g_cu.cuMemAlloc(&m->d_ctx, (size_t)(m->n_heads * m->d_head) * 4);
    g_cu.cuMemAlloc(&m->d_rlogits, (size_t)(m->n_experts > 16 ? m->n_experts : 16) * 4);
    g_cu.cuMemAlloc(&m->d_eidx, (size_t)(m->top_k > 8 ? m->top_k : 8) * 4);
    g_cu.cuMemAlloc(&m->d_ewt, (size_t)(m->top_k > 8 ? m->top_k : 8) * 4);
    g_cu.cuMemAlloc(&m->d_h, h_elems * 4);
    g_cu.cuMemAlloc(&m->d_sh_g, sh_elems * 4);
    g_cu.cuMemAlloc(&m->d_sh_u, sh_elems * 4);
    g_cu.cuMemAlloc(&m->d_logits, (size_t)m->vocab_size * 4);
    g_cu.cuMemAlloc(&m->d_argmax, 16);
    g_cu.cuMemAlloc(&m->d_cpu_accum, (size_t)m->d_model * 4);
    g_cu.cuMemAlloc(&m->d_step_args, 16);

    g_cu.cuMemHostAlloc((void**)&m->h_rlogits, (size_t)(m->n_experts > 16 ? m->n_experts : 16) * 4, CU_MEMHOSTALLOC_PORTABLE);
    g_cu.cuMemHostAlloc((void**)&m->h_argmax, 16, CU_MEMHOSTALLOC_PORTABLE);
    g_cu.cuMemHostAlloc((void**)&m->h_step_args, 16, CU_MEMHOSTALLOC_PORTABLE);
    g_cu.cuMemHostAlloc((void**)&m->h_eidx, (size_t)(CM_MAX_TOPK * 2) * sizeof(uint32_t), CU_MEMHOSTALLOC_PORTABLE);
    g_cu.cuMemHostAlloc((void**)&m->h_ewt,  (size_t)(CM_MAX_TOPK * 2) * sizeof(float),    CU_MEMHOSTALLOC_PORTABLE);
    g_cu.cuMemHostAlloc((void**)&m->h_emb_row, 65536, CU_MEMHOSTALLOC_PORTABLE);
    g_cu.cuMemHostAlloc((void**)&m->h_xn, (size_t)m->d_model * 4, CU_MEMHOSTALLOC_PORTABLE);
    g_cu.cuMemHostAlloc((void**)&m->h_cpu_accum, (size_t)m->d_model * 4, CU_MEMHOSTALLOC_PORTABLE);
    m->cpu_h = (float*)malloc(h_elems * (size_t)CM_CPU_MAX_B * sizeof(float));
    size_t dm_blks = ((size_t)m->d_model + 31u) >> 5;
    size_t h_blks  = (h_elems + 31u) >> 5;
    m->cpu_xq8  = (int8_t*)malloc(dm_blks * 32u);
    m->cpu_xq16 = (int16_t*)malloc(dm_blks * 32u * (size_t)CM_CPU_MAX_B * sizeof(int16_t));
    m->cpu_xd   = (float*)malloc(dm_blks * (size_t)CM_CPU_MAX_B * sizeof(float));
    m->cpu_xsum = (float*)malloc(dm_blks * (size_t)CM_CPU_MAX_B * sizeof(float));
    m->cpu_hq8  = (int8_t*)malloc(h_blks * 32u);
    m->cpu_hq16 = (int16_t*)malloc(h_blks * 32u * (size_t)CM_CPU_MAX_B * sizeof(int16_t));
    m->cpu_hd   = (float*)malloc(h_blks * (size_t)CM_CPU_MAX_B * sizeof(float));
    m->cpu_hsum = (float*)malloc(h_blks * (size_t)CM_CPU_MAX_B * sizeof(float));

    if (!m->resident_mode) {
        cm_io_pool_start(m);
        if (m->enable_overlap && !m->use_nocache && m->mmap_ptrs[0] != NULL) {
            cm_cpu_pool_start(m);
            if (!m->dense_model && m->promo_stage) cm_promo_start(m);
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
            // Parallel NUMA-local first-touch repacking of unpinned Dense FFNs (~0.4s -> 200+ GB/s DDR4!)
            if (m->dense_model && m->dense_pin_all_attn && (m->d_ff_dense % 32 == 0) && (m->d_model % 4 == 0)) {
                for (int L = m->cache_slots_per_layer; L < m->n_layers; L++) {
                    CmLayer* ly = &m->layers[L];
                    if (ly->ffn_gate.ggml_type == CM_GGML_Q4_0 && ly->ffn_up.ggml_type == CM_GGML_Q4_0) {
                        size_t gu_sz = (size_t)ly->ffn_gate.byte_size + (size_t)ly->ffn_up.byte_size;
                        void* p = mmap(NULL, gu_sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                        if (p != MAP_FAILED) {
#if defined(MADV_HUGEPAGE)
                            madvise(p, gu_sz, MADV_HUGEPAGE);
#endif
                            ly->repack_gu = (uint8_t*)p; ly->repack_gu_bytes = gu_sz;
                        }
                    }
                    if (ly->ffn_down.ggml_type == CM_GGML_Q4_0 || ly->ffn_down.ggml_type == CM_GGML_Q4_1) {
                        size_t dn_sz = (size_t)ly->ffn_down.byte_size;
                        void* p = mmap(NULL, dn_sz, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                        if (p != MAP_FAILED) {
#if defined(MADV_HUGEPAGE)
                            madvise(p, dn_sz, MADV_HUGEPAGE);
#endif
                            ly->repack_down = (uint8_t*)p; ly->repack_down_bytes = dn_sz;
                        }
                    }
                    if (ly->repack_gu || ly->repack_down) {
                        cm_cpu_pool_run(m, 3, L, 0, NULL, NULL);
                        cm_cpu_pool_wait(m);
                    }
                }
                // Release raw file pages of repacked layers so host RSS does not double!
                for (int s = 0; s < m->n_shards; s++) {
                    if (m->mmap_ptrs[s]) madvise(m->mmap_ptrs[s], m->shard_sizes[s], MADV_DONTNEED);
                }
            }
#endif
        }
    }

    // Tenon v6: Initialize MTP hidden state + PagedAttention 16-Token Block-Table + Tree-Verify GPU buffers!
    if (m->has_mtp || m->has_mtp_layer) {
        g_cu.cuMemAlloc(&m->d_last_hidden, (size_t)m->d_model * 4u);
        g_cu.cuMemAlloc(&m->d_mtp_cat,     (size_t)m->d_model * 8u);
    }
    m->pa_num_blocks = m->max_ctx / CM_PA_BLOCK_SIZE;
    if (m->pa_num_blocks < 1) m->pa_num_blocks = 1;
    m->pa_free_blocks = m->pa_num_blocks;
    m->pa_refcnt = (int32_t*)calloc((size_t)m->pa_num_blocks, sizeof(int32_t));
    memset(m->pa_block_table, 0xFF, sizeof(m->pa_block_table));
    g_cu.cuMemAlloc(&m->d_pa_bt,      (size_t)CM_PA_MAX_SEQS * CM_PA_MAX_LOGICAL_BLKS * 4u);
    g_cu.cuMemcpyHtoD(m->d_pa_bt, m->pa_block_table, (size_t)CM_PA_MAX_SEQS * CM_PA_MAX_LOGICAL_BLKS * 4u);
    g_cu.cuMemAlloc(&m->d_cb_pos,     (size_t)CM_PREFILL_MAX_B * 4u);
    g_cu.cuMemAlloc(&m->d_cb_slot,    (size_t)CM_PREFILL_MAX_B * 4u);
    g_cu.cuMemAlloc(&m->d_tree_depth, (size_t)CM_TREE_MAX_NODES * 4u);
    g_cu.cuMemAlloc(&m->d_tree_mask,  (size_t)CM_TREE_MAX_NODES * 8u);

    m->active = 1;
    return (int64_t)slot;
}

int64_t tn_cumoe_step(int64_t handle, int64_t tok_id, int64_t pos) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return 0;
    CmModel* m = &g_cm[handle];
    double t0 = cm_now_ms(), io_acc = 0.0;

    // PATH 0: TRUE 2D HYBRID (TP x EP) OR MULTI-GPU TENSOR PARALLELISM (TP=2..8)!
    if (m->hybrid_2d_mode) {
        int64_t out_tok = cm_hybrid_2d_step(m, tok_id, (uint32_t)pos);
        m->last_total_ms = cm_now_ms() - t0;
        return out_tok;
    }
    if (m->tp_size > 1) {
        int64_t out_tok = cm_tp_forward_batch(m, &tok_id, 1u, (uint32_t)pos, NULL);
        m->cache_hits += m->n_layers;
        m->last_total_ms = cm_now_ms() - t0;
        return out_tok;
    }

    // PATH A: FULL RESIDENT DENSE (100% GPU VRAM + NATIVE CUDA GRAPH -> 1 launch for all 80 layers!)
    if (m->dense_model && m->resident_mode) {
        int can_graph = (m->tok_embd.dptr != 0 && m->active_lora == 0 && m->has_mrope != 2 &&
                         !(m->n_layers > 0 && m->layers[0].is_mla) &&
                         g_cu.cuStreamBeginCapture && g_cu.cuStreamEndCapture &&
                         g_cu.cuGraphLaunch && (g_cu.cuGraphInstantiateWithFlags || g_cu.cuGraphInstantiate));
        if (can_graph && !m->res_graph_ready) {
            CUgraph gr = NULL;
            // CaptureMode = 2 (CU_STREAM_CAPTURE_MODE_RELAXED)
            if (g_cu.cuStreamBeginCapture(m->compute_stream, 2) == CUDA_SUCCESS) {
                uint32_t dm = (uint32_t)m->d_model, gt = m->tok_embd.ggml_type;
                CUdeviceptr ep = m->tok_embd.dptr;
                void* eargs[] = { &ep, &m->d_x, &m->d_step_args, &dm, &gt };
                g_cu.cuLaunchKernel(g_cu.k_embed_ind, (dm + 255) / 256, 1, 1, 256, 1, 1, 0, m->compute_stream, eargs, NULL);
                for (int L = 0; L < m->n_layers; L++) {
                    CmLayer* ly = &m->layers[L];
                    launch_layer_attn_ex(m, m->compute_stream, L, 0, 0, 1);
                    launch_dense_ffn_raw(m, m->compute_stream, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                                         ly->ffn_up.dptr, ly->ffn_up.ggml_type,
                                         ly->ffn_down.dptr, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense);
                }
                launch_final_lm_head(m, m->compute_stream);
                g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, m->compute_stream);
                if (g_cu.cuStreamEndCapture(m->compute_stream, &gr) == CUDA_SUCCESS && gr) {
                    CUresult ir = g_cu.cuGraphInstantiateWithFlags
                        ? g_cu.cuGraphInstantiateWithFlags(&m->res_graph_exec, gr, 0)
                        : g_cu.cuGraphInstantiate(&m->res_graph_exec, gr, NULL, NULL, 0);
                    if (g_cu.cuGraphDestroy) g_cu.cuGraphDestroy(gr);
                    if (ir == CUDA_SUCCESS && m->res_graph_exec) m->res_graph_ready = 1;
                }
            }
        }
        if (can_graph && m->res_graph_ready) {
            m->h_step_args[0] = (uint32_t)tok_id;
            m->h_step_args[1] = (uint32_t)pos;
            g_cu.cuMemcpyHtoDAsync(m->d_step_args, m->h_step_args, 8, m->compute_stream);
            g_cu.cuGraphLaunch(m->res_graph_exec, m->compute_stream);
            if (m->d_last_hidden) g_cu.cuMemcpyDtoDAsync(m->d_last_hidden, m->d_x, (size_t)m->d_model * 4u, m->compute_stream);
            g_cu.cuStreamSynchronize(m->compute_stream);
            m->cache_hits += m->n_layers;
            m->last_total_ms = cm_now_ms() - t0;
            return (int64_t)m->h_argmax[0];
        }
        launch_embed(m, m->compute_stream, (uint32_t)tok_id);
        for (int L = 0; L < m->n_layers; L++) {
            CmLayer* ly = &m->layers[L];
            launch_layer_attn(m, m->compute_stream, L, (uint32_t)pos, 0);
            launch_dense_ffn_raw(m, m->compute_stream, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                                 ly->ffn_up.dptr, ly->ffn_up.ggml_type,
                                 ly->ffn_down.dptr, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense);
            m->cache_hits++;
        }
        if (m->d_last_hidden) g_cu.cuMemcpyDtoDAsync(m->d_last_hidden, m->d_x, (size_t)m->d_model * 4u, m->compute_stream);
        launch_final_lm_head(m, m->compute_stream);
        g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, m->compute_stream);
        g_cu.cuStreamSynchronize(m->compute_stream);
        m->last_total_ms = cm_now_ms() - t0;
        return (int64_t)m->h_argmax[0];
    }

    // PATH A-MOE: FULL RESIDENT SPARSE MoE (EP=1, Single CUDA Stream, GPU Router Top-K, ZERO per-layer CPU sync!)
    if (!m->dense_model && m->resident_mode && m->ep_size == 1 && !m->hybrid_2d_mode) {
        CUstream st = m->compute_stream;
        uint32_t dm = (uint32_t)m->d_model, ne = (uint32_t)m->n_experts, tk = (uint32_t)m->top_k;
        uint32_t nw = (uint32_t)m->expert_weights_norm, gf = (uint32_t)m->gating_func;
        uint32_t ng = (uint32_t)m->n_expert_groups, ngu = (uint32_t)m->n_group_used;
        float ws = m->expert_weights_scale;
        launch_embed(m, st, (uint32_t)tok_id);
        for (int L = 0; L < m->n_layers; L++) {
            CmLayer* ly = &m->layers[L];
            launch_layer_attn(m, st, L, (uint32_t)pos, 0);
            if (ly->is_dense) {
                launch_dense_ffn_raw(m, st, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                                     ly->ffn_up.dptr, ly->ffn_up.ggml_type,
                                     ly->ffn_down.dptr, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense);
            } else {
                launch_gemv(st, &ly->router, m->d_xn, m->d_rlogits, ne, dm, 0);
                uint32_t hb = (uint32_t)ly->has_exp_bias;
                CUdeviceptr bp = hb ? ly->exp_bias.dptr : m->d_argmax;
                void* ra[] = { &m->d_rlogits, &bp, &m->d_eidx, &m->d_ewt, &ne, &tk, &nw, &gf, &hb, &ws, &ng, &ngu };
                g_cu.cuLaunchKernel(g_cu.k_router, 1, 1, 1, 32, 1, 1, 0, st, ra, NULL);
                if (ly->has_shared_exp) {
                    uint32_t sh_ff = (ly->sh_gate.ne1 > 0) ? (uint32_t)ly->sh_gate.ne1 : (uint32_t)m->d_ff_exp;
                    launch_dense_ffn_raw(m, st, ly->sh_gate.dptr, ly->sh_gate.ggml_type,
                                         ly->sh_up.dptr, ly->sh_up.ggml_type, ly->sh_down.dptr, ly->sh_down.ggml_type, sh_ff);
                }
                launch_moe_slots(m, st, L, tk);
                m->cache_hits += m->top_k;
            }
        }
        if (m->d_last_hidden) g_cu.cuMemcpyDtoDAsync(m->d_last_hidden, m->d_x, (size_t)dm * 4u, st);
        launch_final_lm_head(m, st);
        g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, st);
        g_cu.cuStreamSynchronize(st);
        m->last_total_ms = cm_now_ms() - t0;
        m->last_io_ms = 0.0; m->last_gpu_ms = m->last_total_ms;
        return (int64_t)m->h_argmax[0];
    }

    // PATH B: DENSE STREAMING (All-80-Attn + Prefix FFNs + In-Place AVX2/FMA DDR4 OR 4-Slot PCIe Ring!)
    if (m->dense_model) {
        int first_stream = m->cache_slots_per_layer;
        if (m->enable_overlap && !m->use_nocache && m->cpu_pool_init && m->dense_pin_all_attn) {
            // HETEROGENEOUS HYBRID DENSE:
            //   1. Prefix layers 0..first_stream-1 captured in a single CUDA Graph on 1,008 GB/s GDDR6X!
            //   2. Remaining layers: GPU Attention + NUMA-local Q4_0_4R/INT16 CPU FFN (ZERO PCIe weight copy!)
            launch_embed(m, m->compute_stream, (uint32_t)tok_id);
            int can_pgraph = (first_stream > 0 && g_cu.cuStreamBeginCapture && g_cu.cuStreamEndCapture &&
                              g_cu.cuGraphLaunch && (g_cu.cuGraphInstantiateWithFlags || g_cu.cuGraphInstantiate));
            if (can_pgraph && !m->res_graph_ready) {
                CUgraph gr = NULL;
                if (g_cu.cuStreamBeginCapture(m->compute_stream, 2) == CUDA_SUCCESS) {
                    for (int L = 0; L < first_stream && L < m->n_layers; L++) {
                        CmLayer* ly = &m->layers[L];
                        launch_layer_attn_ex(m, m->compute_stream, L, 0, 0, 1);
                        launch_dense_ffn_raw(m, m->compute_stream, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                                             ly->ffn_up.dptr, ly->ffn_up.ggml_type,
                                             ly->ffn_down.dptr, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense);
                    }
                    if (g_cu.cuStreamEndCapture(m->compute_stream, &gr) == CUDA_SUCCESS && gr) {
                        CUresult ir = g_cu.cuGraphInstantiateWithFlags
                            ? g_cu.cuGraphInstantiateWithFlags(&m->res_graph_exec, gr, 0)
                            : g_cu.cuGraphInstantiate(&m->res_graph_exec, gr, NULL, NULL, 0);
                        if (g_cu.cuGraphDestroy) g_cu.cuGraphDestroy(gr);
                        if (ir == CUDA_SUCCESS && m->res_graph_exec) m->res_graph_ready = 1;
                    }
                }
            }
            if (m->res_graph_ready) {
                m->h_step_args[0] = (uint32_t)tok_id;
                m->h_step_args[1] = (uint32_t)pos;
                g_cu.cuMemcpyHtoDAsync(m->d_step_args, m->h_step_args, 8, m->compute_stream);
                g_cu.cuGraphLaunch(m->res_graph_exec, m->compute_stream);
                m->cache_hits += (first_stream < m->n_layers ? first_stream : m->n_layers);
            } else {
                for (int L = 0; L < first_stream && L < m->n_layers; L++) {
                    CmLayer* ly = &m->layers[L];
                    launch_layer_attn(m, m->compute_stream, L, (uint32_t)pos, 0);
                    launch_dense_ffn_raw(m, m->compute_stream, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                                         ly->ffn_up.dptr, ly->ffn_up.ggml_type,
                                         ly->ffn_down.dptr, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense);
                    m->cache_hits++;
                }
            }
            CmWeightRef acc_ref = {0};
            acc_ref.dptr = m->d_cpu_accum;
            for (int L = first_stream; L < m->n_layers; L++) {
                m->cache_misses++;
                launch_layer_attn(m, m->compute_stream, L, (uint32_t)pos, 0);
                g_cu.cuMemcpyDtoHAsync(m->h_xn, m->d_xn, (size_t)m->d_model * 4, m->compute_stream);
                g_cu.cuStreamSynchronize(m->compute_stream);
                double i0 = cm_now_ms();
                cm_cpu_pool_run(m, 2, L, 0, NULL, NULL);
                cm_cpu_pool_wait(m);
                io_acc += (cm_now_ms() - i0);
                g_cu.cuMemcpyHtoDAsync(m->d_cpu_accum, m->h_cpu_accum, (size_t)m->d_model * 4, m->compute_stream);
                m->pcie_h2d_bytes += (int64_t)m->d_model * 4;
                launch_add_bias(m->compute_stream, m->d_x, &acc_ref, (uint32_t)m->d_model);
            }
            launch_final_lm_head(m, m->compute_stream);
            g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, m->compute_stream);
            g_cu.cuStreamSynchronize(m->compute_stream);
            double tot = cm_now_ms() - t0;
            m->last_total_ms = tot; m->last_io_ms = io_acc; m->last_gpu_ms = tot - io_acc;
            return (int64_t)m->h_argmax[0];
        }

        int n_ring = m->dense_pin_all_attn ? 4 : 2;
        DenseRingProd prod;
        pthread_t prod_th;
        int started_prod = 0;
        if (m->enable_overlap && first_stream < m->n_layers) {
            prod.m = m; prod.first_stream = first_stream; prod.n_ring = n_ring;
            pthread_mutex_init(&prod.mu, NULL);
            pthread_cond_init(&prod.cv_empty, NULL);
            pthread_cond_init(&prod.cv_full, NULL);
            prod.empty_cnt = n_ring; prod.full_cnt = 0;
            pthread_create(&prod_th, NULL, cm_dense_ring_producer, &prod);
            started_prod = 1;
        }

        launch_embed(m, m->compute_stream, (uint32_t)tok_id);
        for (int L = 0; L < first_stream && L < m->n_layers; L++) {
            CmLayer* ly = &m->layers[L];
            launch_layer_attn(m, m->compute_stream, L, (uint32_t)pos, 0);
            launch_dense_ffn_raw(m, m->compute_stream, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                                 ly->ffn_up.dptr, ly->ffn_up.ggml_type,
                                 ly->ffn_down.dptr, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense);
            m->cache_hits++;
        }

        for (int L = first_stream; L < m->n_layers; L++) {
            CmLayer* ly = &m->layers[L];
            int cur = (L - first_stream) % n_ring;
            m->cache_misses++;
            if (!m->enable_overlap) {
                double i0 = cm_now_ms();
                cm_stage_dense_layer(m, L, m->dense_ring_host[cur]);
                g_cu.cuMemcpyHtoD(m->dense_ring_dptr[cur], m->dense_ring_host[cur], (size_t)m->dense_ring_stride);
                m->pcie_h2d_bytes += (int64_t)m->dense_ring_stride;
                io_acc += (cm_now_ms() - i0);
            } else {
                pthread_mutex_lock(&prod.mu);
                while (prod.full_cnt == 0) pthread_cond_wait(&prod.cv_full, &prod.mu);
                prod.full_cnt--;
                pthread_mutex_unlock(&prod.mu);
                g_cu.cuStreamWaitEvent(m->compute_stream, m->dense_slot_ready[cur], 0);
            }
            launch_layer_attn(m, m->compute_stream, L, (uint32_t)pos, m->dense_pin_all_attn ? 0 : m->dense_ring_dptr[cur]);
            launch_dense_ffn_raw(m, m->compute_stream,
                                 m->dense_ring_dptr[cur] + m->dense_off_gate, ly->ffn_gate.ggml_type,
                                 m->dense_ring_dptr[cur] + m->dense_off_up,   ly->ffn_up.ggml_type,
                                 m->dense_ring_dptr[cur] + m->dense_off_down, ly->ffn_down.ggml_type,
                                 (uint32_t)m->d_ff_dense);
            if (m->enable_overlap) {
                g_cu.cuEventRecord(m->dense_slot_consumed[cur], m->compute_stream);
                pthread_mutex_lock(&prod.mu);
                prod.empty_cnt++;
                pthread_cond_signal(&prod.cv_empty);
                pthread_mutex_unlock(&prod.mu);
            }
        }
        launch_final_lm_head(m, m->compute_stream);
        g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, m->compute_stream);
        g_cu.cuStreamSynchronize(m->compute_stream);
        if (started_prod) {
            pthread_join(prod_th, NULL);
            pthread_mutex_destroy(&prod.mu);
            pthread_cond_destroy(&prod.cv_empty);
            pthread_cond_destroy(&prod.cv_full);
        }
        double tot = cm_now_ms() - t0;
        m->last_total_ms = tot; m->last_io_ms = io_acc; m->last_gpu_ms = tot - io_acc;
        return (int64_t)m->h_argmax[0];
    }

    // PATH C: SPARSE MoE VRAM-LRU + HYBRID AVX2/FMA MISS-COMPUTE (RAM) OR DUAL-STREAM PCIe DMA (NVMe)!
    int hybrid_ram = (m->enable_overlap && !m->use_nocache && m->cpu_pool_init && m->mmap_ptrs[0] != NULL);
    launch_embed(m, m->compute_stream, (uint32_t)tok_id);
    int L = launch_tail_until_next_moe(m, m->compute_stream, 0, (uint32_t)pos);
    g_cu.cuStreamSynchronize(m->compute_stream);

    while (L < m->n_layers) {
        // Non-blocking poll of any in-flight background hot-expert VRAM promotion!
        if (__atomic_load_n(&m->promo_state, __ATOMIC_ACQUIRE) == 2 &&
            g_cu.cuEventQuery && g_cu.cuEventQuery(m->ev_promo_done) == CUDA_SUCCESS) {
            CmLayer* ply = &m->layers[m->promo_layer];
            ply->slot_to_exp[m->promo_slot] = m->promo_eid;
            ply->exp_to_slot[m->promo_eid] = m->promo_slot;
            ply->slot_lru[m->promo_slot] = ++m->lru_clock;
            __atomic_store_n(&m->promo_state, 0, __ATOMIC_RELEASE);
            m->prefetch_hits++;
        }

        CmLayer* ly = &m->layers[L];
        int top_e[CM_MAX_TOPK]; float top_w[CM_MAX_TOPK];
        cm_cpu_router_topk_grp(m->h_rlogits, ly->exp_bias_host, m->n_experts, m->top_k,
                               m->gating_func, m->expert_weights_norm, m->expert_weights_scale,
                               m->n_expert_groups, m->n_group_used, top_e, top_w);

        // PATH C-EP: TRUE MULTI-GPU MoE EXPERT PARALLELISM (`EP=2..8` + Dynamic NCCL AllReduce!)
        if (m->ep_size > 1 && m->ep) {
            int EP = m->ep_size, Er = m->ep_exp_per_rank;
            uint32_t dm = (uint32_t)m->d_model, df = (uint32_t)m->d_ff_exp;
            uint64_t gb = ly->gate_exp_bytes, sb = ly->slot_stride_bytes, doff = ly->gate_exp_bytes + ly->up_exp_bytes;
            int cnt[CM_MAX_TP] = {0};
            for (int k = 0; k < m->top_k; k++) {
                int eid = top_e[k];
                int r = eid / Er; if (r >= EP) r = EP - 1;
                int ls = eid - r * Er;
                int idx = cnt[r]++;
                m->ep[r].h_eidx[idx] = (uint32_t)ls;
                m->ep[r].h_ewt[idx]  = top_w[k];
                m->cache_hits++;
            }
            // Overlap SharedExp on GPU 0 `compute_stream` while EP GPUs execute routed Top-K experts in parallel!
            if (ly->has_shared_exp) {
                uint32_t sh_ff = (ly->sh_gate.ne1 > 0) ? (uint32_t)ly->sh_gate.ne1 : (uint32_t)m->d_ff_exp;
                launch_dense_ffn_raw(m, m->compute_stream, ly->sh_gate.dptr, ly->sh_gate.ggml_type,
                                     ly->sh_up.dptr, ly->sh_up.ggml_type, ly->sh_down.dptr, ly->sh_down.ggml_type, sh_ff);
            }
            int ep_use_nccl = cm_use_nccl_for_payload(m->ep_nccl_ok, g_cu.p2p_matrix[0][1], dm);
            if (ep_use_nccl && g_cu.ncclBroadcast && g_cu.ncclAllReduce) {
                g_cu.ncclGroupStart();
                for (int r = 0; r < EP; r++) {
                    g_cu.cuCtxSetCurrent(g_cu.gpus[r].ctx);
                    const void* sbuf = (r == 0) ? (const void*)(uintptr_t)m->d_xn : (const void*)(uintptr_t)m->ep[r].d_xn;
                    g_cu.ncclBroadcast(sbuf, (void*)(uintptr_t)m->ep[r].d_xn, (size_t)dm, NCCL_FLOAT32, 0, m->ep_nccl[r], m->ep[r].stream);
                }
                g_cu.ncclGroupEnd();
                g_cu.cuCtxSetCurrent(g_cu.ctx);
            } else {
                g_cu.cuMemcpyDtoH(m->h_xn, m->d_xn, (size_t)dm * 4u);
                for (int r = 0; r < EP; r++) {
                    if (cnt[r] == 0) continue;
                    g_cu.cuCtxSetCurrent(g_cu.gpus[r].ctx);
                    g_cu.cuMemcpyHtoDAsync(m->ep[r].d_xn, m->h_xn, (size_t)dm * 4u, m->ep[r].stream);
                }
            }
            int is_q4k_gu  = (ly->gate_exps.ggml_type == CM_GGML_Q4_K);
            int is_q4k_dn  = (ly->down_exps.ggml_type == CM_GGML_Q4_K);
            int is_q6k_dn  = (ly->down_exps.ggml_type == CM_GGML_Q6_K);
            int is_q8_dn   = (ly->down_exps.ggml_type == CM_GGML_Q8_0);
            int is_q5_0_dn = (ly->down_exps.ggml_type == CM_GGML_Q5_0);
            int is_q5_1_dn = (ly->down_exps.ggml_type == CM_GGML_Q5_1);
            for (int r = 0; r < EP; r++) {
                CmDevCtx* dc = &g_cu.gpus[r];
                CmEpRank* er = &m->ep[r];
                g_cu.cuCtxSetCurrent(dc->ctx);
                if (g_cu.cuMemsetD8Async) g_cu.cuMemsetD8Async(er->d_part, 0, (size_t)dm * 4u, er->stream);
                else g_cu.cuMemsetD8(er->d_part, 0, (size_t)dm * 4u);
                if (cnt[r] > 0) {
                    uint32_t na = (uint32_t)cnt[r];
                    g_cu.cuMemcpyHtoDAsync(er->d_eidx, er->h_eidx, (size_t)na * 4u, er->stream);
                    g_cu.cuMemcpyHtoDAsync(er->d_ewt,  er->h_ewt,  (size_t)na * 4u, er->stream);
                    void* gu_a[] = { &er->cache_dptr[L], &er->d_xn, &er->d_eidx, &er->d_h, &dm, &df, &na, &gb, &sb };
                    g_cu.cuLaunchKernel(is_q4k_gu ? dc->k_moe_slot_gu_q4k : dc->k_moe_slot_gu_q4,
                                        is_q4k_gu ? ((df + 3u) / 4u) : ((df + 7u) / 8u), na, 1, 128, 1, 1, 0, er->stream, gu_a, NULL);
                    CUfunction dfn = is_q6k_dn ? dc->k_moe_slot_down_q6k
                        : (is_q4k_dn ? dc->k_moe_slot_down_q4k
                           : (is_q8_dn ? dc->k_moe_slot_down_q8
                              : (is_q5_0_dn ? dc->k_moe_slot_down_q5_0
                                 : (is_q5_1_dn ? dc->k_moe_slot_down_q5_1
                                    : ((ly->down_exps.ggml_type == CM_GGML_Q4_1) ? dc->k_moe_slot_down_q4_1 : dc->k_moe_slot_down_q4)))));
                    void* dn_a[] = { &er->cache_dptr[L], &er->d_h, &er->d_eidx, &er->d_ewt, &er->d_part, &dm, &df, &na, &doff, &sb };
                    g_cu.cuLaunchKernel(dfn, (is_q4k_dn || is_q6k_dn || is_q8_dn || is_q5_0_dn || is_q5_1_dn) ? ((dm + 3u) / 4u) : ((dm + 15u) / 16u),
                                        1, 1, 128, 1, 1, 0, er->stream, dn_a, NULL);
                }
            }
            if (ep_use_nccl && g_cu.ncclAllReduce) {
                g_cu.ncclGroupStart();
                for (int r = 0; r < EP; r++) {
                    g_cu.cuCtxSetCurrent(g_cu.gpus[r].ctx);
                    g_cu.ncclAllReduce((const void*)(uintptr_t)m->ep[r].d_part, (void*)(uintptr_t)m->ep[r].d_part,
                                       (size_t)dm, NCCL_FLOAT32, NCCL_SUM, m->ep_nccl[r], m->ep[r].stream);
                }
                g_cu.ncclGroupEnd();
                g_cu.cuCtxSetCurrent(g_cu.ctx);
                g_cu.cuEventRecord(m->ep[0].ev_done, m->ep[0].stream);
                g_cu.cuStreamWaitEvent(m->compute_stream, m->ep[0].ev_done, 0);
                CmWeightRef ep_ref = {0}; ep_ref.dptr = m->ep[0].d_part;
                launch_add_bias(m->compute_stream, m->d_x, &ep_ref, dm);
            } else {
                memset(m->h_cpu_accum, 0, (size_t)dm * 4u);
                for (int r = 0; r < EP; r++) {
                    if (cnt[r] == 0) continue;
                    g_cu.cuCtxSetCurrent(g_cu.gpus[r].ctx);
                    g_cu.cuMemcpyDtoHAsync(m->ep[r].h_part, m->ep[r].d_part, (size_t)dm * 4u, m->ep[r].stream);
                    g_cu.cuStreamSynchronize(m->ep[r].stream);
                    for (uint32_t i = 0; i < dm; i++) m->h_cpu_accum[i] += m->ep[r].h_part[i];
                }
                g_cu.cuCtxSetCurrent(g_cu.ctx);
                g_cu.cuMemcpyHtoDAsync(m->d_cpu_accum, m->h_cpu_accum, (size_t)dm * 4u, m->compute_stream);
                CmWeightRef ep_ref = {0}; ep_ref.dptr = m->d_cpu_accum;
                launch_add_bias(m->compute_stream, m->d_x, &ep_ref, dm);
            }
            int next_L = launch_tail_until_next_moe(m, m->compute_stream, L + 1, (uint32_t)pos);
            g_cu.cuStreamSynchronize(m->compute_stream);
            L = next_L;
            continue;
        }

        int hit_slots[CM_MAX_TOPK], miss_eids[CM_MAX_TOPK], miss_slots[CM_MAX_TOPK], pinned[CM_MAX_TOPK];
        float hit_w[CM_MAX_TOPK], miss_w[CM_MAX_TOPK];
        int n_hits = 0, n_misses = 0, npin = 0;

        for (int k = 0; k < m->top_k; k++) {
            int eid = top_e[k], cs = ly->exp_to_slot[eid];
            ly->exp_freq[eid]++;
            if (cs >= 0) {
                m->cache_hits++;
                ly->slot_lru[cs] = ++m->lru_clock;
                hit_slots[n_hits] = cs; hit_w[n_hits] = top_w[k]; n_hits++;
                pinned[npin++] = cs;
            } else {
                m->cache_misses++;
                miss_eids[n_misses] = eid; miss_w[n_misses] = top_w[k]; n_misses++;
            }
        }

        // Check if layer L still has empty VRAM slots
        int has_empty_slot = 0;
        if (hybrid_ram && n_misses > 0) {
            for (int s = 0; s < m->cache_slots_per_layer; s++) {
                if (ly->slot_to_exp[s] == -1) { has_empty_slot = 1; break; }
            }
        }

        // During prefill (before reset), populate empty VRAM slots; during decode, NEVER stall on PCIe!
        if (hybrid_ram && n_misses > 0 && (m->decode_phase || !has_empty_slot)) {
            // =================================================================
            // TENON v3 HETEROGENEOUS ZERO-STALL HYBRID PATH:
            //   1. GPU (1,008 GB/s GDDR6X): SharedExp + ALL VRAM Hits
            //   2. CPU (32-Thread AVX2 INT8 vpmaddubsw, 180+ GB/s DDR4): ALL Misses IN PLACE!
            //   3. Merge 16 KB FP32 partial sum -> GPU + launch Attn(L+1)
            //   4. Off-thread background promoter silently fills empty slots or hot misses!
            // =================================================================
            if (ly->has_shared_exp) {
                uint32_t sh_ff = (ly->sh_gate.ne1 > 0) ? (uint32_t)ly->sh_gate.ne1 : (uint32_t)m->d_ff_exp;
                launch_dense_ffn_raw(m, m->compute_stream, ly->sh_gate.dptr, ly->sh_gate.ggml_type,
                                     ly->sh_up.dptr, ly->sh_up.ggml_type, ly->sh_down.dptr, ly->sh_down.ggml_type, sh_ff);
            }
            if (n_hits > 0) {
                for (int hi = 0; hi < n_hits; hi++) { m->h_eidx[hi] = (uint32_t)hit_slots[hi]; m->h_ewt[hi] = hit_w[hi]; }
                g_cu.cuMemcpyHtoDAsync(m->d_eidx, m->h_eidx, (size_t)n_hits * 4, m->compute_stream);
                g_cu.cuMemcpyHtoDAsync(m->d_ewt,  m->h_ewt,  (size_t)n_hits * 4, m->compute_stream);
                launch_moe_slots(m, m->compute_stream, L, (uint32_t)n_hits);
            }

            // SIMULTANEOUS 32-thread AVX2 INT8 `vpmaddubsw` CPU compute directly from mmap (ZERO memcpy!)
            double i0 = cm_now_ms();
            cm_cpu_pool_run(m, 1, L, n_misses, miss_eids, miss_w);
            cm_cpu_pool_wait(m);
            io_acc += (cm_now_ms() - i0);

            // Merge 16 KB CPU partial sum into GPU residual d_x and immediately fire Attn(L+1)
            g_cu.cuMemcpyHtoDAsync(m->d_cpu_accum, m->h_cpu_accum, (size_t)m->d_model * 4, m->compute_stream);
            m->pcie_h2d_bytes += (int64_t)m->d_model * 4;
            CmWeightRef acc_ref = {0}; acc_ref.dptr = m->d_cpu_accum;
            launch_add_bias(m->compute_stream, m->d_x, &acc_ref, (uint32_t)m->d_model);
            int next_L = launch_tail_until_next_moe(m, m->compute_stream, L + 1, (uint32_t)pos);

            // Off-thread Zero-Stall Background Promotion (0.00 ms on main thread!)
            if (m->promo_thread_init && __atomic_load_n(&m->promo_state, __ATOMIC_ACQUIRE) == 0 && g_cu.cuEventQuery) {
                int best_eid = -1; uint32_t min_f = has_empty_slot ? 0u : 1u; uint32_t best_f = min_f;
                for (int mi = 0; mi < n_misses; mi++) {
                    int e = miss_eids[mi];
                    if (ly->exp_freq[e] > best_f) { best_f = ly->exp_freq[e]; best_eid = e; }
                }
                if (best_eid >= 0) {
                    int vs = cm_pick_lru(m, ly, pinned, npin);
                    if (ly->slot_to_exp[vs] != -2) {
                        int old_e = ly->slot_to_exp[vs];
                        if (old_e >= 0) ly->exp_to_slot[old_e] = -1;
                        ly->slot_to_exp[vs] = -2;
                        pthread_mutex_lock(&m->promo_mu);
                        m->promo_layer = L; m->promo_slot = vs; m->promo_eid = best_eid;
                        __atomic_store_n(&m->promo_state, 1, __ATOMIC_RELEASE);
                        pthread_cond_signal(&m->promo_cv);
                        pthread_mutex_unlock(&m->promo_mu);
                    }
                }
            }
            g_cu.cuStreamSynchronize(m->compute_stream);
            L = next_L;
            continue;
        }

        for (int mi = 0; mi < n_misses; mi++) {
            int vs = cm_pick_lru(m, ly, pinned, npin);
            pinned[npin++] = vs;
            int old_e = ly->slot_to_exp[vs];
            if (old_e >= 0) ly->exp_to_slot[old_e] = -1;
            ly->slot_to_exp[vs] = miss_eids[mi];
            ly->exp_to_slot[miss_eids[mi]] = vs;
            ly->slot_lru[vs] = ++m->lru_clock;
            miss_slots[mi] = vs;
        }

        if (!m->enable_overlap) {
            // Serial Baseline: Stage + Sync H2D -> All Experts + Attn(L+1)
            double i0 = cm_now_ms();
            for (int mi = 0; mi < n_misses; mi++) {
                cm_stage_expert(m, L, miss_eids[mi], m->stage_host);
                g_cu.cuMemcpyHtoD(ly->cache_dptr + (uint64_t)miss_slots[mi] * ly->slot_stride_bytes,
                                  m->stage_host, (size_t)ly->slot_stride_bytes);
                m->pcie_h2d_bytes += (int64_t)ly->slot_stride_bytes;
            }
            io_acc += (cm_now_ms() - i0);
            int tot_a = 0;
            for (int hi = 0; hi < n_hits; hi++)   { m->h_eidx[tot_a] = (uint32_t)hit_slots[hi]; m->h_ewt[tot_a] = hit_w[hi]; tot_a++; }
            for (int mi = 0; mi < n_misses; mi++) { m->h_eidx[tot_a] = (uint32_t)miss_slots[mi]; m->h_ewt[tot_a] = miss_w[mi]; tot_a++; }
            g_cu.cuMemcpyHtoDAsync(m->d_eidx, m->h_eidx, (size_t)tot_a * 4, m->compute_stream);
            g_cu.cuMemcpyHtoDAsync(m->d_ewt,  m->h_ewt,  (size_t)tot_a * 4, m->compute_stream);
            if (ly->has_shared_exp) {
                uint32_t sh_ff = (ly->sh_gate.ne1 > 0) ? (uint32_t)ly->sh_gate.ne1 : (uint32_t)m->d_ff_exp;
                launch_dense_ffn_raw(m, m->compute_stream, ly->sh_gate.dptr, ly->sh_gate.ggml_type,
                                     ly->sh_up.dptr, ly->sh_up.ggml_type, ly->sh_down.dptr, ly->sh_down.ggml_type, sh_ff);
            }
            launch_moe_slots(m, m->compute_stream, L, (uint32_t)tot_a);
            int next_L = launch_tail_until_next_moe(m, m->compute_stream, L + 1, (uint32_t)pos);
            g_cu.cuStreamSynchronize(m->compute_stream);
            L = next_L;
        } else if (n_misses == 0) {
            // ZERO-MISS FAST PATH: 100% VRAM GDDR6X, ZERO PCIe traffic!
            for (int hi = 0; hi < n_hits; hi++) { m->h_eidx[hi] = (uint32_t)hit_slots[hi]; m->h_ewt[hi] = hit_w[hi]; }
            if (n_hits > 0) {
                g_cu.cuMemcpyHtoDAsync(m->d_eidx, m->h_eidx, (size_t)n_hits * 4, m->compute_stream);
                g_cu.cuMemcpyHtoDAsync(m->d_ewt,  m->h_ewt,  (size_t)n_hits * 4, m->compute_stream);
            }
            if (ly->has_shared_exp) {
                uint32_t sh_ff = (ly->sh_gate.ne1 > 0) ? (uint32_t)ly->sh_gate.ne1 : (uint32_t)m->d_ff_exp;
                launch_dense_ffn_raw(m, m->compute_stream, ly->sh_gate.dptr, ly->sh_gate.ggml_type,
                                     ly->sh_up.dptr, ly->sh_up.ggml_type, ly->sh_down.dptr, ly->sh_down.ggml_type, sh_ff);
            }
            if (n_hits > 0) launch_moe_slots(m, m->compute_stream, L, (uint32_t)n_hits);
            int next_L = launch_tail_until_next_moe(m, m->compute_stream, L + 1, (uint32_t)pos);
            g_cu.cuStreamSynchronize(m->compute_stream);
            L = next_L;
        } else {
            // DUAL-STREAM HARDWARE OVERLAP:
            //   compute_stream: SharedExp + VRAM Hits (1,008 GB/s)
            //   dma_stream:     Parallel Pinned Staging + cuMemcpyHtoDAsync (25 GB/s PCIe 4.0)
            //   GPU-side sync via `cuStreamWaitEvent(compute_stream, ev_dma_done)` -> Misses + Attn(L+1)!
            if (ly->has_shared_exp) {
                uint32_t sh_ff = (ly->sh_gate.ne1 > 0) ? (uint32_t)ly->sh_gate.ne1 : (uint32_t)m->d_ff_exp;
                launch_dense_ffn_raw(m, m->compute_stream, ly->sh_gate.dptr, ly->sh_gate.ggml_type,
                                     ly->sh_up.dptr, ly->sh_up.ggml_type, ly->sh_down.dptr, ly->sh_down.ggml_type, sh_ff);
            }
            if (n_hits > 0) {
                for (int hi = 0; hi < n_hits; hi++) { m->h_eidx[hi] = (uint32_t)hit_slots[hi]; m->h_ewt[hi] = hit_w[hi]; }
                g_cu.cuMemcpyHtoDAsync(m->d_eidx, m->h_eidx, (size_t)n_hits * 4, m->compute_stream);
                g_cu.cuMemcpyHtoDAsync(m->d_ewt,  m->h_ewt,  (size_t)n_hits * 4, m->compute_stream);
                launch_moe_slots(m, m->compute_stream, L, (uint32_t)n_hits);
            }

            // Stage missed experts in parallel via persistent pool & pipeline cuMemcpyHtoDAsync as each completes!
            double i0 = cm_now_ms();
            if (n_misses == 1 || !m->io_pool_init) {
                for (int mi = 0; mi < n_misses; mi++) {
                    uint8_t* sp = m->stage_host + (size_t)mi * m->stage_slot_bytes;
                    cm_stage_expert(m, L, miss_eids[mi], sp);
                    g_cu.cuMemcpyHtoDAsync(ly->cache_dptr + (uint64_t)miss_slots[mi] * ly->slot_stride_bytes,
                                           sp, (size_t)ly->slot_stride_bytes, m->dma_stream);
                    m->pcie_h2d_bytes += (int64_t)ly->slot_stride_bytes;
                }
            } else {
                pthread_mutex_lock(&m->io_mu);
                m->io_mode = 1;
                m->io_layer = L;
                m->io_n_tasks = n_misses;
                m->io_done_cnt = 0;
                for (int mi = 0; mi < n_misses; mi++) {
                    m->io_eids[mi] = miss_eids[mi];
                    m->io_dsts[mi] = m->stage_host + (size_t)mi * m->stage_slot_bytes;
                    m->io_task_ready[mi] = 0;
                }
                m->io_gen++;
                pthread_cond_broadcast(&m->io_cv_go);
                pthread_mutex_unlock(&m->io_mu);

                int sent_flag[CM_MAX_TOPK] = {0};
                int sent = 0;
                while (sent < n_misses) {
                    int progressed = 0;
                    for (int mi = 0; mi < n_misses; mi++) {
                        if (!sent_flag[mi] && m->io_task_ready[mi]) {
                            __sync_synchronize();
                            g_cu.cuMemcpyHtoDAsync(ly->cache_dptr + (uint64_t)miss_slots[mi] * ly->slot_stride_bytes,
                                                   m->io_dsts[mi], (size_t)ly->slot_stride_bytes, m->dma_stream);
                            m->pcie_h2d_bytes += (int64_t)ly->slot_stride_bytes;
                            sent_flag[mi] = 1;
                            sent++;
                            progressed = 1;
                        }
                    }
                    if (!progressed && sent < n_misses) {
                        pthread_mutex_lock(&m->io_mu);
                        int any_rdy = 0;
                        for (int mi = 0; mi < n_misses; mi++) if (!sent_flag[mi] && m->io_task_ready[mi]) { any_rdy = 1; break; }
                        if (!any_rdy && m->io_done_cnt < n_misses) pthread_cond_wait(&m->io_cv_done, &m->io_mu);
                        pthread_mutex_unlock(&m->io_mu);
                    }
                }
                pthread_mutex_lock(&m->io_mu);
                while (m->io_done_cnt < n_misses) pthread_cond_wait(&m->io_cv_done, &m->io_mu);
                pthread_mutex_unlock(&m->io_mu);
            }
            g_cu.cuEventRecord(m->ev_dma_done, m->dma_stream);
            io_acc += (cm_now_ms() - i0);

            // Wait for DMA completion directly on GPU command processor, then run Misses + Attn(L+1)!
            g_cu.cuStreamWaitEvent(m->compute_stream, m->ev_dma_done, 0);
            uint32_t ms_u[CM_MAX_TOPK];
            for (int mi = 0; mi < n_misses; mi++) ms_u[mi] = (uint32_t)miss_slots[mi];
            // Offset into second half of pinned h_eidx/h_ewt so zero overlap with in-flight hit copy!
            memcpy(m->h_eidx + CM_MAX_TOPK, ms_u, (size_t)n_misses * 4);
            memcpy(m->h_ewt  + CM_MAX_TOPK, miss_w, (size_t)n_misses * 4);
            g_cu.cuMemcpyHtoDAsync(m->d_eidx, m->h_eidx + CM_MAX_TOPK, (size_t)n_misses * 4, m->compute_stream);
            g_cu.cuMemcpyHtoDAsync(m->d_ewt,  m->h_ewt  + CM_MAX_TOPK, (size_t)n_misses * 4, m->compute_stream);
            launch_moe_slots(m, m->compute_stream, L, (uint32_t)n_misses);
            int next_L = launch_tail_until_next_moe(m, m->compute_stream, L + 1, (uint32_t)pos);
            g_cu.cuStreamSynchronize(m->compute_stream);
            L = next_L;
        }
    }

    if (m->ep_size > 1 && m->ep) {
        for (int r = 0; r < m->ep_size; r++) {
            g_cu.cuCtxSetCurrent(g_cu.gpus[r].ctx);
            g_cu.cuStreamSynchronize(m->ep[r].stream);
        }
        g_cu.cuCtxSetCurrent(g_cu.ctx);
    }
    if (m->d_last_hidden) {
        g_cu.cuMemcpyDtoDAsync(m->d_last_hidden, m->d_x, (size_t)m->d_model * 4u, m->compute_stream);
        g_cu.cuStreamSynchronize(m->compute_stream);
    }
    double tot = cm_now_ms() - t0;
    m->last_total_ms = tot; m->last_io_ms = io_acc; m->last_gpu_ms = (tot > io_acc) ? (tot - io_acc) : 0.0;
    return (int64_t)m->h_argmax[0];
}

int64_t tn_cumoe_info(int64_t handle, int64_t field_id) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return 0;
    CmModel* m = &g_cm[handle];
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
        case 16: return m->io_bytes_read;
        case 17: return cm_os_rss_bytes();
        case 18: return cm_os_peak_rss_bytes();
        case 19: return (int64_t)m->total_disk_bytes;
        case 20: return m->n_shards;
        case 21: return m->pcie_h2d_bytes;
        case 22: return m->max_ctx;
        case 23: return m->kv_quant_q8;
        case 24: return (m->n_layers > 0 && m->layers[0].is_mla) ? 1 : 0;
        case 25: return m->fast_dp4a;
        case 26: return m->mla_latent_kv;
        case 27: return m->n_kv_slots;
        case 28: return m->slot_ctx;
        case 29: return m->n_bpe_merges;
        case 30: return m->has_mtp_layer ? 2 : m->has_mtp;
        case 31: return m->tp_size;
        case 32: return m->tp_p2p_ok;
        case 33: return (m->tp_nccl_ok || m->ep_nccl_ok) ? 1 : 0;
        case 34: return m->ep_size > 0 ? m->ep_size : 1;
        case 35: return m->pa_num_blocks;
        case 36: return m->pa_free_blocks;
        case 37: return g_cu.nccl_avail;
        case 38: return m->hybrid_2d_mode;
        case 39: return m->active_lora;
        case 40: { int c = 0; for (int i = 1; i < CM_MAX_LORAS; i++) if (m->loras[i].active) c++; return c; }
        case 41: return m->has_mrope;
        case 42: return m->moe_grouped_saved_reads;
        default: return 0;
    }
}

// =========================================================================
// Batched Causal Prompt Prefill (`tn_cumoe_prefill`), Auto Topology Planner
// (`tn_cumoe_auto_plan`), Full Sampler (`tn_cumoe_sample_next`), Tokenizer!
// =========================================================================
static void cm_ensure_prefill_bufs(CmModel* m, int B) {
    if (m->prefill_cap >= B) return;
    int cap = B < 32 ? 32 : (B > CM_PREFILL_MAX_B ? CM_PREFILL_MAX_B : B);
    size_t dm = (size_t)m->d_model;
    size_t qd = (size_t)(m->n_heads * m->d_head);
    size_t kvd = (size_t)(m->n_kv * m->d_head);
    size_t ff = (size_t)(m->d_ff_dense > m->d_ff_exp ? m->d_ff_dense : m->d_ff_exp);
    if (m->d_px)   g_cu.cuMemFree(m->d_px);
    if (m->d_pxn)  g_cu.cuMemFree(m->d_pxn);
    if (m->d_pq)   g_cu.cuMemFree(m->d_pq);
    if (m->d_pk)   g_cu.cuMemFree(m->d_pk);
    if (m->d_pv)   g_cu.cuMemFree(m->d_pv);
    if (m->d_pctx) g_cu.cuMemFree(m->d_pctx);
    if (m->d_pg)   g_cu.cuMemFree(m->d_pg);
    if (m->d_pu)   g_cu.cuMemFree(m->d_pu);
    g_cu.cuMemAlloc(&m->d_px,   (size_t)cap * dm * 4u);
    g_cu.cuMemAlloc(&m->d_pxn,  (size_t)cap * dm * 4u);
    g_cu.cuMemAlloc(&m->d_pq,   (size_t)cap * qd * 4u);
    g_cu.cuMemAlloc(&m->d_pk,   (size_t)cap * kvd * 4u);
    g_cu.cuMemAlloc(&m->d_pv,   (size_t)cap * kvd * 4u);
    g_cu.cuMemAlloc(&m->d_pctx, (size_t)cap * qd * 4u);
    g_cu.cuMemAlloc(&m->d_pg,   (size_t)cap * ff * 4u);
    g_cu.cuMemAlloc(&m->d_pu,   (size_t)cap * ff * 4u);
    if (m->n_layers > 0 && m->layers[0].is_mla && m->mla_buf_cap < (size_t)cap) {
        size_t kv_r = (size_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
        size_t ldim = kv_r + (size_t)m->rope_dim;
        size_t qr = (size_t)(m->q_lora_rank > 0 ? m->q_lora_rank : m->d_model);
        size_t nope_d = (size_t)(m->d_head - m->rope_dim);
        size_t v_d = (size_t)m->v_head_dim;
        size_t kv_sz = ldim > (size_t)m->n_heads * (nope_d + v_d) ? ldim : (size_t)m->n_heads * (nope_d + v_d);
        if (m->d_mla_q)    g_cu.cuMemFree(m->d_mla_q);
        if (m->d_mla_kv)   g_cu.cuMemFree(m->d_mla_kv);
        if (m->d_mla_qabs) g_cu.cuMemFree(m->d_mla_qabs);
        if (m->d_mla_olat) g_cu.cuMemFree(m->d_mla_olat);
        g_cu.cuMemAlloc(&m->d_mla_q,    (size_t)cap * qr * 4u);
        g_cu.cuMemAlloc(&m->d_mla_kv,   (size_t)cap * (ldim + kv_sz) * 4u);
        g_cu.cuMemAlloc(&m->d_mla_qabs, (size_t)cap * (size_t)m->n_heads * ldim * 4u);
        g_cu.cuMemAlloc(&m->d_mla_olat, (size_t)cap * (size_t)m->n_heads * kv_r * 4u);
        m->mla_buf_cap = (size_t)cap;
    }
    if (m->n_experts > 0) {
        if (m->d_prlogits) g_cu.cuMemFree(m->d_prlogits);
        if (m->d_moe_x)    g_cu.cuMemFree(m->d_moe_x);
        if (m->d_moe_y)    g_cu.cuMemFree(m->d_moe_y);
        if (m->d_moe_idx)  g_cu.cuMemFree(m->d_moe_idx);
        if (m->d_moe_wt)   g_cu.cuMemFree(m->d_moe_wt);
        if (m->h_prlogits) g_cu.cuMemFreeHost(m->h_prlogits);
        if (m->h_moe_idx)  g_cu.cuMemFreeHost(m->h_moe_idx);
        if (m->h_moe_wt)   g_cu.cuMemFreeHost(m->h_moe_wt);
        g_cu.cuMemAlloc(&m->d_prlogits, (size_t)cap * (size_t)m->n_experts * 4u);
        g_cu.cuMemAlloc(&m->d_moe_x,    (size_t)cap * dm * 4u);
        g_cu.cuMemAlloc(&m->d_moe_y,    (size_t)cap * dm * 4u);
        g_cu.cuMemAlloc(&m->d_moe_idx,  (size_t)cap * CM_MAX_TOPK * 4u);
        g_cu.cuMemAlloc(&m->d_moe_wt,   (size_t)cap * CM_MAX_TOPK * 4u);
        g_cu.cuMemHostAlloc((void**)&m->h_prlogits, (size_t)cap * (size_t)m->n_experts * 4u, CU_MEMHOSTALLOC_PORTABLE);
        g_cu.cuMemHostAlloc((void**)&m->h_moe_idx,  (size_t)cap * CM_MAX_TOPK * 4u, CU_MEMHOSTALLOC_PORTABLE);
        g_cu.cuMemHostAlloc((void**)&m->h_moe_wt,   (size_t)cap * CM_MAX_TOPK * 4u, CU_MEMHOSTALLOC_PORTABLE);
    }
    if (m->h_pxn) g_cu.cuMemFreeHost(m->h_pxn);
    if (m->h_pcpu_accum) g_cu.cuMemFreeHost(m->h_pcpu_accum);
    g_cu.cuMemHostAlloc((void**)&m->h_pxn,        (size_t)cap * dm * 4u, CU_MEMHOSTALLOC_PORTABLE);
    g_cu.cuMemHostAlloc((void**)&m->h_pcpu_accum, (size_t)cap * dm * 4u, CU_MEMHOSTALLOC_PORTABLE);
    m->prefill_cap = cap;
}

static void launch_gemm_batch_raw(CUstream st, CUdeviceptr w, uint32_t gtype, CUdeviceptr x, CUdeviceptr y,
                                  uint32_t out_dim, uint32_t in_dim, uint32_t B, uint32_t add_res) {
    if (B == 1u) {
        launch_gemv_raw(st, w, gtype, x, y, out_dim, in_dim, add_res);
        return;
    }
    if (B <= 8u) {
        if (gtype == CM_GGML_Q4_0) {
            void* args[] = { &w, &x, &y, &out_dim, &in_dim, &B, &add_res };
            g_cu.cuLaunchKernel(g_cu.k_gemm_q4_sb, (out_dim + 7u) / 8u, 1, 1, 128, 1, 1, 0, st, args, NULL);
            return;
        }
        if (gtype == CM_GGML_Q4_1) {
            void* args[] = { &w, &x, &y, &out_dim, &in_dim, &B, &add_res };
            g_cu.cuLaunchKernel(g_cu.k_gemm_q4_1_sb, (out_dim + 7u) / 8u, 1, 1, 128, 1, 1, 0, st, args, NULL);
            return;
        }
        if (gtype == CM_GGML_Q4_K) {
            void* args[] = { &w, &x, &y, &out_dim, &in_dim, &B, &add_res };
            g_cu.cuLaunchKernel(g_cu.k_gemm_q4k_sb, (out_dim + 3u) / 4u, 1, 1, 128, 1, 1, 0, st, args, NULL);
            return;
        }
        if (gtype == CM_GGML_Q6_K) {
            void* args[] = { &w, &x, &y, &out_dim, &in_dim, &B, &add_res };
            g_cu.cuLaunchKernel(g_cu.k_gemm_q6k_sb, (out_dim + 3u) / 4u, 1, 1, 128, 1, 1, 0, st, args, NULL);
            return;
        }
    }
    CUfunction fn = g_cu.k_gemm_q4_b;
    uint32_t tile_n = 64u, tile_b = 32u, bdim = 128u;
    switch (gtype) {
        case CM_GGML_Q4_0:   fn = g_cu.k_gemm_q4_b;    tile_n = 64u; tile_b = 32u; bdim = 256u; break;
        case CM_GGML_Q4_1:   fn = g_cu.k_gemm_q4_1_b;  tile_n = 64u; tile_b = 32u; bdim = 256u; break;
        case CM_GGML_Q5_0:   fn = g_cu.k_gemm_q5_0_b;  tile_n = 4u;  tile_b = 8u;  break;
        case CM_GGML_Q5_1:   fn = g_cu.k_gemm_q5_1_b;  tile_n = 4u;  tile_b = 8u;  break;
        case CM_GGML_Q8_0:   fn = g_cu.k_gemm_q8_b;    tile_n = 4u;  tile_b = 8u;  break;
        case CM_GGML_Q2_K:   fn = g_cu.k_gemm_q2k_b;   tile_n = 4u;  tile_b = 4u;  break;
        case CM_GGML_Q3_K:   fn = g_cu.k_gemm_q3k_b;   tile_n = 4u;  tile_b = 4u;  break;
        case CM_GGML_IQ4_XS: fn = g_cu.k_gemm_iq4xs_b; tile_n = 4u;  tile_b = 4u;  break;
        case CM_GGML_Q4_K:   fn = g_cu.k_gemm_q4k_marlin; tile_n = 32u; tile_b = 32u; bdim = 128u; break;
        case CM_GGML_Q5_K:   fn = g_cu.k_gemm_q5k_b;      tile_n = 4u;  tile_b = 4u;  break;
        case CM_GGML_Q6_K:   fn = g_cu.k_gemm_q6k_marlin; tile_n = 32u; tile_b = 32u; bdim = 128u; break;
        case CM_GGML_FP8_E4M3: fn = (B <= 8u) ? g_cu.k_gemm_fp8_b   : g_cu.k_gemm_fp8_tc;   tile_n = (B <= 8u) ? 4u : 32u; tile_b = (B <= 8u) ? 8u : 32u; break;
        case CM_GGML_MXFP4:    fn = (B <= 8u) ? g_cu.k_gemm_mxfp4_b : g_cu.k_gemm_mxfp4_tc; tile_n = (B <= 8u) ? 4u : 32u; tile_b = (B <= 8u) ? 8u : 32u; break;
        case CM_GGML_F16:    fn = g_cu.k_gemm_f16_b;   tile_n = 4u;  tile_b = 8u;  break;
        default:             fn = g_cu.k_gemm_f32_b;   tile_n = 4u;  tile_b = 8u;  break;
    }
    void* args[] = { &w, &x, &y, &out_dim, &in_dim, &B, &add_res };
    g_cu.cuLaunchKernel(fn, (out_dim + tile_n - 1u) / tile_n, (B + tile_b - 1u) / tile_b, 1, bdim, 1, 1, 0, st, args, NULL);
}

static void launch_add_bias_batch(CUstream st, CUdeviceptr x, const CmWeightRef* b, uint32_t dim, uint32_t B) {
    if (!b->dptr) return;
    CUdeviceptr bp = b->dptr;
    void* args[] = { &x, &bp, &dim, &B };
    g_cu.cuLaunchKernel(g_cu.k_add_bias_b, (dim + 255u) / 256u, B, 1, 256, 1, 1, 0, st, args, NULL);
}

static void launch_layer_attn_batch(CmModel* m, CUstream st, int L, uint32_t start_pos, uint32_t B, CUdeviceptr ring_base) {
    m->cur_exec_layer = L;
    CmLayer* ly = &m->layers[L];
    uint32_t dm = (uint32_t)m->d_model, qdim = (uint32_t)(m->n_heads * m->d_head), kvdim = (uint32_t)(m->n_kv * m->d_head);
    uint32_t wo_in = (ly->wo.ne0 > 0) ? (uint32_t)ly->wo.ne0 : qdim;
    launch_rmsnorm(st, m->d_px, &ly->attn_norm, m->d_pxn, dm, B, m->rms_eps, 1);

    if (ly->is_mla && ly->wkv_a.dptr && m->mla_latent_kv && ly->mla_wuk_f16 && ly->mla_wuv_f16) {
        uint32_t kv_r = (uint32_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
        uint32_t nope_d = (uint32_t)(m->d_head - m->rope_dim), v_d = (uint32_t)m->v_head_dim;
        uint32_t nh_u = (uint32_t)m->n_heads, rd_u = (uint32_t)m->rope_dim, dh_u = (uint32_t)m->d_head;
        uint32_t ldim = kv_r + rd_u, rn = (uint32_t)m->rope_neox, hf = (uint32_t)m->has_rope_freqs;
        float theta = m->rope_theta;
        CUdeviceptr rf = hf ? m->rope_freqs.dptr : m->d_argmax;
        cm_ensure_mla_bufs(m, B);
        if (ly->wq_a.dptr && ly->wq_b.dptr && m->q_lora_rank > 0) {
            launch_gemm_batch_raw(st, ly->wq_a.dptr, ly->wq_a.ggml_type, m->d_pxn, m->d_mla_q, (uint32_t)m->q_lora_rank, dm, B, 0);
            if (ly->q_a_norm.dptr) launch_rmsnorm(st, m->d_mla_q, &ly->q_a_norm, m->d_mla_q, (uint32_t)m->q_lora_rank, B, m->rms_eps, 1);
            launch_gemm_batch_raw(st, ly->wq_b.dptr, ly->wq_b.ggml_type, m->d_mla_q, m->d_pq, qdim, (uint32_t)m->q_lora_rank, B, 0);
        } else {
            launch_gemm_batch_raw(st, ly->wq.dptr, ly->wq.ggml_type, m->d_pxn, m->d_pq, qdim, dm, B, 0);
        }
        launch_gemm_batch_raw(st, ly->wkv_a.dptr, ly->wkv_a.ggml_type, m->d_pxn, m->d_mla_kv, ldim, dm, B, 0);
        for (uint32_t bi = 0; bi < B; bi++) {
            CUdeviceptr ckv_b = m->d_mla_kv + (uint64_t)bi * ldim * 4ull;
            if (ly->kv_a_norm.dptr) launch_rmsnorm(st, ckv_b, &ly->kv_a_norm, ckv_b, kv_r, 1, m->rms_eps, 0);
            CUdeviceptr kpe_b = ckv_b + (uint64_t)kv_r * 4ull;
            CUdeviceptr qb = m->d_pq + (uint64_t)bi * qdim * 4ull;
            uint32_t cur_p = start_pos + bi, one = 1u, half_r = rd_u >> 1;
            void* rpe_a[] = { &qb, &kpe_b, &rf, &nh_u, &dh_u, &nope_d, &rd_u, &cur_p, &theta, &rn, &hf, &one };
            g_cu.cuLaunchKernel(g_cu.k_mla_rope_pe, (half_r + 31u) / 32u, nh_u + 1u, 1, 32, 1, 1, 0, st, rpe_a, NULL);
            void* st_a[] = { &ckv_b, &kpe_b, &ly->k_cache, &kv_r, &rd_u, &cur_p, &one };
            g_cu.cuLaunchKernel(g_cu.k_mla_store_latent, (ldim + 255u) / 256u, 1, 1, 256, 1, 1, 0, st, st_a, NULL);
        }
        float mla_scale = 1.0f / sqrtf((float)dh_u);
        if (m->yarn_mscale_k > 0.0f && m->yarn_factor > 1.0f) {
            float ms = 1.0f + m->yarn_mscale_k * logf(m->yarn_factor);
            mla_scale *= ms * ms;
        }
        void* qa[] = { &m->d_pq, &ly->mla_wuk_f16, &m->d_mla_qabs, &nh_u, &dh_u, &nope_d, &rd_u, &kv_r, &B };
        g_cu.cuLaunchKernel(g_cu.k_mla_absorb_q, (ldim + 3u) / 4u, nh_u, B, 128, 1, 1, 0, st, qa, NULL);
        void* fa[] = { &m->d_mla_qabs, &ly->k_cache, &m->d_mla_olat, &nh_u, &kv_r, &rd_u, &mla_scale, &start_pos, &B };
        g_cu.cuLaunchKernel(g_cu.k_flash_mla_latent, nh_u, B, 1, 128, 1, 1, 0, st, fa, NULL);
        uint32_t out_st = wo_in / nh_u;
        void* va[] = { &m->d_mla_olat, &ly->mla_wuv_f16, &m->d_pctx, &nh_u, &kv_r, &v_d, &out_st, &B };
        g_cu.cuLaunchKernel(g_cu.k_mla_proj_v, (out_st + 3u) / 4u, nh_u, B, 128, 1, 1, 0, st, va, NULL);
        launch_gemm_batch_raw(st, ly->wo.dptr, ly->wo.ggml_type, m->d_pctx, m->d_px, dm, wo_in, B, 1);
        launch_rmsnorm(st, m->d_px, &ly->ffn_norm, m->d_pxn, dm, B, m->rms_eps, 1);
        return;
    }

    CUdeviceptr pq = ring_base ? (ring_base + m->dense_off_wq) : ly->wq.dptr;
    CUdeviceptr pk = ring_base ? (ring_base + m->dense_off_wk) : ly->wk.dptr;
    CUdeviceptr pv = ring_base ? (ring_base + m->dense_off_wv) : ly->wv.dptr;
    launch_gemm_batch_raw(st, pq, ly->wq.ggml_type, m->d_pxn, m->d_pq, qdim, dm, B, 0);
    launch_gemm_batch_raw(st, pk, ly->wk.ggml_type, m->d_pxn, m->d_pk, kvdim, dm, B, 0);
    launch_gemm_batch_raw(st, pv, ly->wv.ggml_type, m->d_pxn, m->d_pv, kvdim, dm, B, 0);
    if (m->active_lora > 0) {
        launch_lora_delta(m, st, L, 0, m->d_pxn, m->d_pq, qdim, dm, B);
        launch_lora_delta(m, st, L, 1, m->d_pxn, m->d_pk, kvdim, dm, B);
        launch_lora_delta(m, st, L, 2, m->d_pxn, m->d_pv, kvdim, dm, B);
    }
    if (ly->has_qkv_bias) {
        launch_add_bias_batch(st, m->d_pq, &ly->bq, qdim, B);
        launch_add_bias_batch(st, m->d_pk, &ly->bk, kvdim, B);
        launch_add_bias_batch(st, m->d_pv, &ly->bv, kvdim, B);
    }
    if (ly->has_qk_norm) {
        if (ly->qk_norm_per_head) {
            launch_rmsnorm(st, m->d_pq, &ly->q_norm, m->d_pq, (uint32_t)m->d_head, B * (uint32_t)m->n_heads, m->rms_eps, 1);
            launch_rmsnorm(st, m->d_pk, &ly->k_norm, m->d_pk, (uint32_t)m->d_head, B * (uint32_t)m->n_kv, m->rms_eps, 1);
        } else {
            launch_rmsnorm(st, m->d_pq, &ly->q_norm, m->d_pq, qdim, B, m->rms_eps, 1);
            launch_rmsnorm(st, m->d_pk, &ly->k_norm, m->d_pk, kvdim, B, m->rms_eps, 1);
        }
    }
    uint32_t nh = (uint32_t)m->n_heads, nkv = (uint32_t)m->n_kv, dh = (uint32_t)m->d_head, rd = (uint32_t)m->rope_dim;
    uint32_t rn = (uint32_t)m->rope_neox, hf = (uint32_t)m->has_rope_freqs;
    float theta = m->rope_theta;
    CUdeviceptr rf = hf ? m->rope_freqs.dptr : m->d_argmax;
    if (m->kv_quant_q8) {
        void* rargs[] = { &m->d_pq, &m->d_pk, &m->d_pv, &ly->k_cache, &ly->v_cache, &rf, &nh, &nkv, &dh, &rd, &start_pos, &theta, &rn, &hf, &B };
        g_cu.cuLaunchKernel(g_cu.k_rope_kv_q8_b, nh, B, 1, 128, 1, 1, 0, st, rargs, NULL);
        void* aargs[] = { &m->d_pq, &ly->k_cache, &ly->v_cache, &m->d_pctx, &nh, &nkv, &dh, &start_pos, &B };
        g_cu.cuLaunchKernel(g_cu.k_flash_prefill_q8, nh, B, 1, 128, 1, 1, 0, st, aargs, NULL);
    } else {
        uint32_t hd2 = dh >> 1;
        void* rargs[] = { &m->d_pq, &m->d_pk, &m->d_pv, &ly->k_cache, &ly->v_cache, &rf, &nh, &nkv, &dh, &rd, &start_pos, &theta, &rn, &hf, &B };
        g_cu.cuLaunchKernel(g_cu.k_rope_kv_b, (hd2 + 31u) / 32u, nh, B, 32, 1, 1, 0, st, rargs, NULL);
        void* aargs[] = { &m->d_pq, &ly->k_cache, &ly->v_cache, &m->d_pctx, &nh, &nkv, &dh, &start_pos, &B };
        g_cu.cuLaunchKernel(g_cu.k_flash_prefill, nh, B, 1, 128, 1, 1, 0, st, aargs, NULL);
    }
    CUdeviceptr po = ring_base ? (ring_base + m->dense_off_wo) : ly->wo.dptr;
    launch_gemm_batch_raw(st, po, ly->wo.ggml_type, m->d_pctx, m->d_px, dm, wo_in, B, 1);
    if (m->active_lora > 0) launch_lora_delta(m, st, L, 3, m->d_pctx, m->d_px, dm, wo_in, B);
    launch_rmsnorm(st, m->d_px, &ly->ffn_norm, m->d_pxn, dm, B, m->rms_eps, 1);
}

static void launch_dense_ffn_batch(CmModel* m, CUstream st,
                                   CUdeviceptr wg, uint32_t gt, CUdeviceptr wu, uint32_t ut,
                                   CUdeviceptr wd, uint32_t dt, uint32_t d_ff, uint32_t B) {
    uint32_t dm = (uint32_t)m->d_model;
    int L = m->cur_exec_layer;
    launch_gemm_batch_raw(st, wg, gt, m->d_pxn, m->d_pg, d_ff, dm, B, 0);
    if (m->active_lora > 0 && L >= 0) launch_lora_delta(m, st, L, 4, m->d_pxn, m->d_pg, d_ff, dm, B);
    launch_gemm_batch_raw(st, wu, ut, m->d_pxn, m->d_pu, d_ff, dm, B, 0);
    if (m->active_lora > 0 && L >= 0) launch_lora_delta(m, st, L, 5, m->d_pxn, m->d_pu, d_ff, dm, B);
    uint32_t tot = B * d_ff;
    void* sargs[] = { &m->d_pg, &m->d_pu, &tot };
    g_cu.cuLaunchKernel(g_cu.k_swiglu_mul, (tot + 255u) / 256u, 1, 1, 256, 1, 1, 0, st, sargs, NULL);
    launch_gemm_batch_raw(st, wd, dt, m->d_pg, m->d_px, dm, d_ff, B, 1);
    if (m->active_lora > 0 && L >= 0) launch_lora_delta(m, st, L, 6, m->d_pg, m->d_px, dm, d_ff, B);
}

static int64_t cm_prefill_or_verify(CmModel* m, int64_t handle, const int64_t* ids, int64_t n_toks, int64_t start_pos, int64_t* out_argmax_all) {
    if (m->tp_size > 1 && !m->hybrid_2d_mode) {
        double t0 = cm_now_ms();
        int64_t last_tok = 0, off = 0;
        while (off < n_toks) {
            uint32_t B = (uint32_t)((n_toks - off > 64) ? 64 : (n_toks - off));
            last_tok = cm_tp_forward_batch(m, ids + off, B, (uint32_t)(start_pos + off),
                                           out_argmax_all ? (out_argmax_all + off) : NULL);
            off += B;
        }
        m->last_total_ms = cm_now_ms() - t0;
        return last_tok;
    }
    int is_mla_uncomp = (m->n_layers > 0 && m->layers[0].is_mla && !m->mla_latent_kv);
    if ((n_toks <= 2 && !out_argmax_all) || !m->dense_model || is_mla_uncomp) {
        int64_t last = 0;
        for (int64_t i = 0; i < n_toks; i++) {
            last = tn_cumoe_step(handle, ids[i], start_pos + i);
            if (out_argmax_all) out_argmax_all[i] = last;
        }
        return last;
    }
    double t0 = cm_now_ms();
    int64_t last_tok = 0;
    int chunk_sz = 64;
    cm_ensure_prefill_bufs(m, chunk_sz);
    uint32_t dm = (uint32_t)m->d_model, gt = m->tok_embd.ggml_type;
    int64_t off = 0;
    while (off < n_toks) {
        uint32_t B = (uint32_t)((n_toks - off > chunk_sz) ? chunk_sz : (n_toks - off));
        uint32_t cur_pos = (uint32_t)(start_pos + off);
        int is_final_chunk = (off + B == n_toks);

        for (uint32_t bi = 0; bi < B; bi++) {
            CUdeviceptr dst = m->d_px + (uint64_t)bi * (uint64_t)dm * 4ull;
            uint32_t tid = (uint32_t)ids[off + bi];
            if (m->resident_mode && m->tok_embd.dptr) {
                CUdeviceptr ep = m->tok_embd.dptr;
                void* args[] = { &ep, &dst, &tid, &dm, &gt };
                g_cu.cuLaunchKernel(g_cu.k_embed, (dm + 255u) / 256u, 1, 1, 256, 1, 1, 0, m->compute_stream, args, NULL);
            } else {
                uint64_t rb = cm_ggml_type_bytes(gt, (uint64_t)dm);
                cm_read_slice(m, m->tok_embd.shard_idx, m->h_emb_row, (size_t)rb, m->tok_embd.file_off + (uint64_t)tid * rb);
                g_cu.cuMemcpyHtoD(m->d_emb_row, m->h_emb_row, (size_t)rb);
                uint32_t z = 0;
                void* args[] = { &m->d_emb_row, &dst, &z, &dm, &gt };
                g_cu.cuLaunchKernel(g_cu.k_embed, (dm + 255u) / 256u, 1, 1, 256, 1, 1, 0, m->compute_stream, args, NULL);
                g_cu.cuStreamSynchronize(m->compute_stream);
            }
        }

        if (m->resident_mode) {
            for (int L = 0; L < m->n_layers; L++) {
                CmLayer* ly = &m->layers[L];
                launch_layer_attn_batch(m, m->compute_stream, L, cur_pos, B, 0);
                launch_dense_ffn_batch(m, m->compute_stream, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                                       ly->ffn_up.dptr, ly->ffn_up.ggml_type,
                                       ly->ffn_down.dptr, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
            }
        } else {
            int first_stream = m->cache_slots_per_layer;
            for (int L = 0; L < first_stream && L < m->n_layers; L++) {
                CmLayer* ly = &m->layers[L];
                launch_layer_attn_batch(m, m->compute_stream, L, cur_pos, B, 0);
                launch_dense_ffn_batch(m, m->compute_stream, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                                       ly->ffn_up.dptr, ly->ffn_up.ggml_type,
                                       ly->ffn_down.dptr, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense, B);
            }
            if (m->enable_overlap && !m->use_nocache && m->cpu_pool_init && m->dense_pin_all_attn) {
                // 24GB Hybrid Batched Prefill: Batched Tensor Core Attention + 8-Token Shared-Weight AVX2 CPU FFN (`mode == 4`)!
                CmWeightRef pacc_ref = {0}; pacc_ref.dptr = m->d_pxn;
                for (int L = first_stream; L < m->n_layers; L++) {
                    launch_layer_attn_batch(m, m->compute_stream, L, cur_pos, B, 0);
                    g_cu.cuMemcpyDtoHAsync(m->h_pxn, m->d_pxn, (size_t)B * dm * 4u, m->compute_stream);
                    g_cu.cuStreamSynchronize(m->compute_stream);
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
                    CmLayer* ly = &m->layers[L];
                    if (ly->repack_gu != NULL && ly->repack_down != NULL) {
                        for (uint32_t bi = 0; bi < B; bi += (uint32_t)CM_CPU_MAX_B) {
                            uint32_t bt = (B - bi > (uint32_t)CM_CPU_MAX_B) ? (uint32_t)CM_CPU_MAX_B : (B - bi);
                            m->cpu_batch_b = (int)bt;
                            m->cpu_batch_off = (size_t)bi;
                            cm_cpu_pool_run(m, 4, L, 0, NULL, NULL);
                            cm_cpu_pool_wait(m);
                        }
                    } else
#endif
                    {
                        for (uint32_t bi = 0; bi < B; bi++) {
                            memcpy(m->h_xn, m->h_pxn + (size_t)bi * dm, (size_t)dm * 4u);
                            cm_cpu_pool_run(m, 2, L, 0, NULL, NULL);
                            cm_cpu_pool_wait(m);
                            memcpy(m->h_pcpu_accum + (size_t)bi * dm, m->h_cpu_accum, (size_t)dm * 4u);
                        }
                    }
                    g_cu.cuMemcpyHtoDAsync(m->d_pxn, m->h_pcpu_accum, (size_t)B * dm * 4u, m->compute_stream);
                    m->pcie_h2d_bytes += (int64_t)((size_t)B * dm * 4u);
                    launch_add_bias(m->compute_stream, m->d_px, &pacc_ref, B * dm);
                }
            } else {
                int n_ring = m->dense_pin_all_attn ? 4 : 2;
                DenseRingProd prod; pthread_t prod_th; int started_prod = 0;
                if (m->enable_overlap && first_stream < m->n_layers) {
                    prod.m = m; prod.first_stream = first_stream; prod.n_ring = n_ring;
                    pthread_mutex_init(&prod.mu, NULL); pthread_cond_init(&prod.cv_empty, NULL); pthread_cond_init(&prod.cv_full, NULL);
                    prod.empty_cnt = n_ring; prod.full_cnt = 0;
                    pthread_create(&prod_th, NULL, cm_dense_ring_producer, &prod);
                    started_prod = 1;
                }
                for (int L = first_stream; L < m->n_layers; L++) {
                    CmLayer* ly = &m->layers[L];
                    int cur = (L - first_stream) % n_ring;
                    if (!m->enable_overlap) {
                        cm_stage_dense_layer(m, L, m->dense_ring_host[cur]);
                        g_cu.cuMemcpyHtoD(m->dense_ring_dptr[cur], m->dense_ring_host[cur], (size_t)m->dense_ring_stride);
                        m->pcie_h2d_bytes += (int64_t)m->dense_ring_stride;
                    } else {
                        pthread_mutex_lock(&prod.mu);
                        while (prod.full_cnt == 0) pthread_cond_wait(&prod.cv_full, &prod.mu);
                        prod.full_cnt--;
                        pthread_mutex_unlock(&prod.mu);
                        g_cu.cuStreamWaitEvent(m->compute_stream, m->dense_slot_ready[cur], 0);
                    }
                    launch_layer_attn_batch(m, m->compute_stream, L, cur_pos, B, m->dense_pin_all_attn ? 0 : m->dense_ring_dptr[cur]);
                    launch_dense_ffn_batch(m, m->compute_stream,
                                           m->dense_ring_dptr[cur] + m->dense_off_gate, ly->ffn_gate.ggml_type,
                                           m->dense_ring_dptr[cur] + m->dense_off_up,   ly->ffn_up.ggml_type,
                                           m->dense_ring_dptr[cur] + m->dense_off_down, ly->ffn_down.ggml_type,
                                           (uint32_t)m->d_ff_dense, B);
                    if (m->enable_overlap) {
                        g_cu.cuEventRecord(m->dense_slot_consumed[cur], m->compute_stream);
                        pthread_mutex_lock(&prod.mu); prod.empty_cnt++; pthread_cond_signal(&prod.cv_empty); pthread_mutex_unlock(&prod.mu);
                    }
                }
                g_cu.cuStreamSynchronize(m->compute_stream);
                if (started_prod) {
                    pthread_join(prod_th, NULL);
                    pthread_mutex_destroy(&prod.mu); pthread_cond_destroy(&prod.cv_empty); pthread_cond_destroy(&prod.cv_full);
                }
            }
        }

        uint32_t v = (uint32_t)m->vocab_size;
        if (out_argmax_all != NULL && B > 1u) {
            if (m->batch_logits_cap < (int)B) {
                int ncap = (int)(B < 16 ? 16 : B);
                if (m->d_batch_logits) g_cu.cuMemFree(m->d_batch_logits);
                g_cu.cuMemAlloc(&m->d_batch_logits, (size_t)ncap * (size_t)v * 4u);
                m->batch_logits_cap = ncap;
            }
            if (!m->d_argmax_batch) {
                g_cu.cuMemAlloc(&m->d_argmax_batch, (size_t)CM_PREFILL_MAX_B * 4u);
                g_cu.cuMemHostAlloc((void**)&m->h_argmax_batch, (size_t)CM_PREFILL_MAX_B * 4u, CU_MEMHOSTALLOC_PORTABLE);
            }
            launch_rmsnorm(m->compute_stream, m->d_px, &m->out_norm, m->d_pxn, dm, B, m->rms_eps, 1);
            launch_gemm_batch_raw(m->compute_stream, m->lm_head.dptr, m->lm_head.ggml_type, m->d_pxn, m->d_batch_logits, v, dm, B, 0);
            void* aa[] = { &m->d_batch_logits, &m->d_argmax_batch, &v, &B };
            g_cu.cuLaunchKernel(g_cu.k_argmax_b, B, 1, 1, 256, 1, 1, 0, m->compute_stream, aa, NULL);
            g_cu.cuMemcpyDtoHAsync(m->h_argmax_batch, m->d_argmax_batch, (size_t)B * 4u, m->compute_stream);
            g_cu.cuMemcpyDtoDAsync(m->d_logits, m->d_batch_logits + (uint64_t)(B - 1u) * (uint64_t)v * 4ull, (size_t)v * 4u, m->compute_stream);
            g_cu.cuStreamSynchronize(m->compute_stream);
            for (uint32_t bi = 0; bi < B; bi++) {
                out_argmax_all[off + bi] = (int64_t)m->h_argmax_batch[bi];
                last_tok = out_argmax_all[off + bi];
            }
        } else if (out_argmax_all != NULL) {
            CUdeviceptr bx = m->d_px;
            launch_rmsnorm(m->compute_stream, bx, &m->out_norm, m->d_xn, dm, 1, m->rms_eps, 0);
            launch_gemv(m->compute_stream, &m->lm_head, m->d_xn, m->d_logits, v, dm, 0);
            void* aargs[] = { &m->d_logits, &m->d_argmax, &v };
            g_cu.cuLaunchKernel(g_cu.k_argmax, 1, 1, 1, 256, 1, 1, 0, m->compute_stream, aargs, NULL);
            g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, m->compute_stream);
            g_cu.cuStreamSynchronize(m->compute_stream);
            out_argmax_all[off] = (int64_t)m->h_argmax[0];
            last_tok = out_argmax_all[off];
        } else if (is_final_chunk) {
            CUdeviceptr last_x = m->d_px + (uint64_t)(B - 1u) * (uint64_t)dm * 4ull;
            if (m->d_last_hidden) g_cu.cuMemcpyDtoDAsync(m->d_last_hidden, last_x, (size_t)dm * 4u, m->compute_stream);
            launch_rmsnorm(m->compute_stream, last_x, &m->out_norm, m->d_xn, dm, 1, m->rms_eps, 0);
            launch_gemv(m->compute_stream, &m->lm_head, m->d_xn, m->d_logits, v, dm, 0);
            void* aargs[] = { &m->d_logits, &m->d_argmax, &v };
            g_cu.cuLaunchKernel(g_cu.k_argmax, 1, 1, 1, 256, 1, 1, 0, m->compute_stream, aargs, NULL);
            g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, m->compute_stream);
            g_cu.cuStreamSynchronize(m->compute_stream);
            last_tok = (int64_t)m->h_argmax[0];
        } else {
            g_cu.cuStreamSynchronize(m->compute_stream);
        }
        off += B;
    }
    m->last_total_ms = cm_now_ms() - t0;
    return last_tok;
}

int64_t tn_cumoe_prefill(int64_t handle, int64_t tok_ids_ptr, int64_t n_toks, int64_t start_pos) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active || !tok_ids_ptr || n_toks <= 0) return 0;
    return cm_prefill_or_verify(&g_cm[handle], handle, (const int64_t*)(uintptr_t)tok_ids_ptr, n_toks, start_pos, NULL);
}

int64_t tn_cumoe_verify_batch(int64_t handle, int64_t tok_ids_ptr, int64_t n_toks, int64_t start_pos, int64_t out_argmax_ptr) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active || !tok_ids_ptr || !out_argmax_ptr || n_toks <= 0) return 0;
    (void)cm_prefill_or_verify(&g_cm[handle], handle, (const int64_t*)(uintptr_t)tok_ids_ptr, n_toks, start_pos, (int64_t*)(uintptr_t)out_argmax_ptr);
    return n_toks;
}

int64_t tn_cumoe_kv_slot_set(int64_t handle, int64_t slot_id) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return 0;
    CmModel* m = &g_cm[handle];
    if (slot_id < 0 || slot_id >= m->n_kv_slots) return 0;
    if (m->cur_kv_slot == (int)slot_id) return 1;
    m->cur_kv_slot = (int)slot_id;
    if (m->res_graph_exec && g_cu.cuGraphExecDestroy) {
        g_cu.cuGraphExecDestroy(m->res_graph_exec);
        m->res_graph_exec = NULL;
    }
    m->res_graph_ready = 0;
    if (m->tp_size > 1) {
        for (int r = 0; r < m->tp_size; r++) {
            for (int L = 0; L < m->n_layers; L++) {
                CmTpLayer* tl = &m->tp[r].layers[L];
                tl->k_cache = tl->k_cache_base + (uint64_t)slot_id * (uint64_t)tl->kv_slot_bytes;
                tl->v_cache = tl->v_cache_base + (uint64_t)slot_id * (uint64_t)tl->kv_slot_bytes;
            }
        }
    } else {
        for (int L = 0; L < m->n_layers; L++) {
            CmLayer* ly = &m->layers[L];
            ly->k_cache = ly->k_cache_base + (uint64_t)slot_id * (uint64_t)ly->kv_slot_bytes;
            if (ly->v_cache_base) ly->v_cache = ly->v_cache_base + (uint64_t)slot_id * (uint64_t)ly->kv_slot_bytes;
        }
    }
    return 1;
}

int64_t tn_cumoe_copy_logits(int64_t handle, int64_t out_f32_ptr, int64_t max_vocab) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active || !out_f32_ptr || max_vocab <= 0) return 0;
    CmModel* m = &g_cm[handle];
    int64_t n = (m->vocab_size < max_vocab) ? m->vocab_size : max_vocab;
    if (n <= 0 || !m->d_logits) return 0;
    g_cu.cuCtxSetCurrent(g_cu.ctx);
    g_cu.cuMemcpyDtoH((void*)(uintptr_t)out_f32_ptr, m->d_logits, (size_t)n * sizeof(float));
    return n;
}

static uint64_t cm_linux_avail_ram_bytes(void) {
#if defined(__linux__)
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256]; uint64_t avail_kb = 0, total_kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "MemAvailable:", 13) == 0) sscanf(line + 13, "%llu", (unsigned long long*)&avail_kb);
        else if (strncmp(line, "MemTotal:", 9) == 0) sscanf(line + 9, "%llu", (unsigned long long*)&total_kb);
    }
    fclose(f);
    return (avail_kb > 0 ? avail_kb : total_kb) * 1024ull;
#else
    return 0;
#endif
}

// Auto CUDA VRAM / Host-RAM / cgroup-vCPU Topology Planner!
// out_plan[9]: [0]=cache_slots, [1]=use_nocache, [2]=kv_q8, [3]=prefill_chunk, [4]=est_vram_bytes, [5]=disk_bytes, [6]=n_layers, [7]=n_experts, [8]=d_model
int64_t tn_cumoe_auto_plan(int64_t path_cstr, double max_vram_gb, int64_t max_ctx, int64_t kv_mode, int64_t out_plan_ptr) {
    if (!tn_cumoe_init() || !path_cstr || !out_plan_ptr) return 0;
    const char* path0 = (const char*)(uintptr_t)path_cstr;
    CmModel probe; memset(&probe, 0, sizeof(probe));
    CmTensorInfo* tlist = NULL; uint64_t tcount = 0; int split_count = 1;
    if (!cm_parse_shard(path0, 0, 1, &probe, &tlist, &tcount, &split_count)) { free(tlist); return 0; }
    if (split_count > CM_MAX_SHARDS) split_count = CM_MAX_SHARDS;
    uint64_t disk_bytes = 0;
    for (int s = 0; s < split_count; s++) {
        char sp[1024];
        if (s == 0) strncpy(sp, path0, sizeof(sp) - 1);
        else {
            const char* mk = strstr(path0, "-00001-of-");
            if (!mk) break;
            snprintf(sp, sizeof(sp), "%.*s-%05d-of-%s", (int)(mk - path0), path0, s + 1, mk + 10);
            cm_parse_shard(sp, s, 0, &probe, &tlist, &tcount, NULL);
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
        if (strstr(nm, "token_embd")) continue;
        if (strstr(nm, "ffn_gate_exps") || strstr(nm, "ffn_up_exps") || strstr(nm, "ffn_down_exps")) exp_layer_b += tlist[i].byte_size;
        else if (dense && (strstr(nm, "ffn_gate.weight") || strstr(nm, "ffn_up.weight") || strstr(nm, "ffn_down.weight"))) {
            ffn_layer_b += tlist[i].byte_size;
            if (strstr(nm, "ffn_gate.weight")) n_ffn_cnt++;
        } else core_w += tlist[i].byte_size;
    }
    free(tlist);

    size_t free_v = 0, tot_v = g_cu.vram_bytes;
    if (g_cu.cuMemGetInfo) g_cu.cuMemGetInfo(&free_v, &tot_v);
    uint64_t hw_vram = (free_v > (1024ull * 1024 * 1024)) ? (uint64_t)free_v : (uint64_t)tot_v;
    const char* tp_e = getenv("TENON_TP_SIZE");
    int tp_sz = (tp_e && atoi(tp_e) > 1 && dense) ? atoi(tp_e) : 1;
    if (tp_sz > g_cu.n_gpus) tp_sz = g_cu.n_gpus;
    const char* ep_e = getenv("TENON_EP_SIZE");
    int ep_sz = (ep_e && atoi(ep_e) > 1 && !dense) ? atoi(ep_e) : 1;
    if (ep_sz > g_cu.n_gpus) ep_sz = g_cu.n_gpus;
    int mult_g = (tp_sz > ep_sz) ? tp_sz : ep_sz;
    if (mult_g > 1) hw_vram *= (uint64_t)mult_g;
    uint64_t budget = (max_vram_gb > 0.5) ? (uint64_t)(max_vram_gb * 1024.0 * 1024.0 * 1024.0) : (uint64_t)((double)hw_vram * 0.94);
    uint64_t host_ram = cm_linux_avail_ram_bytes();
    int64_t ctx = (max_ctx > 0) ? max_ctx : CM_DEFAULT_SEQ;
    if (ctx > CM_MAX_SEQ) ctx = CM_MAX_SEQ;
    int kv_q8 = (kv_mode == 1) ? 1 : (kv_mode == 0 ? 0 : (ctx > 2048 ? 1 : 0));
    const char* mla_e = getenv("TENON_MLA_LATENT_KV");
    int mla_lat = (probe.kv_lora_rank > 0 && !(mla_e && atoi(mla_e) == 0)) ? 1 : 0;
    uint64_t kv_b = mla_lat
        ? ((uint64_t)probe.n_layers * (uint64_t)ctx * (uint64_t)(probe.kv_lora_rank + (probe.rope_dim > 0 ? probe.rope_dim : 64)) * 2ull)
        : (kv_q8
            ? ((uint64_t)probe.n_layers * 2ull * (uint64_t)ctx * (uint64_t)probe.n_kv * (uint64_t)(probe.d_head >> 5) * 34ull)
            : ((uint64_t)probe.n_layers * 2ull * (uint64_t)ctx * (uint64_t)(probe.n_kv * probe.d_head) * 2ull));
    uint64_t base_vram = core_w + kv_b + (512ull * 1024 * 1024);

    int64_t slots = -1, nocache = 0;
    uint64_t est_vram = disk_bytes + kv_b;
    if (disk_bytes + kv_b + (768ull * 1024 * 1024) <= budget) {
        slots = dense ? -1 : probe.n_experts;
        nocache = 0;
        est_vram = disk_bytes + kv_b;
    } else if (dense) {
        uint64_t one_ffn = (n_ffn_cnt > 0) ? (ffn_layer_b / (uint64_t)n_ffn_cnt) : (370ull * 1024 * 1024);
        nocache = (host_ram > 0 && disk_bytes > (uint64_t)((double)host_ram * 0.85)) ? 1 : 0;
        uint64_t ring_ovh = nocache ? (one_ffn * 4ull) : 0ull;
        if (budget > base_vram + ring_ovh + one_ffn) {
            slots = (int64_t)((budget - base_vram - ring_ovh) / one_ffn);
            if (slots >= probe.n_layers) slots = probe.n_layers - 1;
            est_vram = base_vram + ring_ovh + (uint64_t)slots * one_ffn;
        } else {
            slots = 0;
            est_vram = base_vram + ring_ovh;
        }
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
        nocache = (host_ram > 0 && disk_bytes > (uint64_t)((double)host_ram * 0.85)) ? 1 : 0;
    }
    int64_t* out = (int64_t*)(uintptr_t)out_plan_ptr;
    out[0] = slots; out[1] = nocache; out[2] = kv_q8; out[3] = 32;
    out[4] = (int64_t)est_vram; out[5] = (int64_t)disk_bytes;
    out[6] = probe.n_layers; out[7] = probe.n_experts; out[8] = probe.d_model;
    return 1;
}

typedef struct { int id; float val; } CmCand;
static int cm_cmp_cand_desc(const void* a, const void* b) {
    float va = ((const CmCand*)a)->val, vb = ((const CmCand*)b)->val;
    return (va < vb) ? 1 : (va > vb ? -1 : 0);
}

int64_t tn_cumoe_sample_next(
    int64_t handle, int64_t tok_id, int64_t pos,
    double temp, int64_t top_k, double top_p, double min_p, double repeat_penalty,
    int64_t recent_ptr, int64_t n_recent, int64_t seed
) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return 0;
    CmModel* m = &g_cm[handle];
    if (tok_id >= 0) {
        int64_t argmax_id = tn_cumoe_step(handle, tok_id, pos);
        if (temp <= 1e-5 && (repeat_penalty <= 1.0001 || n_recent <= 0)) return argmax_id;
    }
    int V = m->vocab_size;
    if (!m->h_logits) m->h_logits = (float*)malloc((size_t)V * sizeof(float));
    g_cu.cuMemcpyDtoH(m->h_logits, m->d_logits, (size_t)V * sizeof(float));
    float* lg = (float*)malloc((size_t)V * sizeof(float));
    memcpy(lg, m->h_logits, (size_t)V * sizeof(float));
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
    CmCand* cands = (CmCand*)malloc((size_t)K * sizeof(CmCand));
    int nc = 0; float min_in = -1e30f; int min_idx = 0;
    for (int i = 0; i < V; i++) {
        if (nc < K) {
            cands[nc++] = (CmCand){ i, lg[i] };
            if (nc == K) {
                min_in = cands[0].val; min_idx = 0;
                for (int j = 1; j < K; j++) if (cands[j].val < min_in) { min_in = cands[j].val; min_idx = j; }
            }
        } else if (lg[i] > min_in) {
            cands[min_idx] = (CmCand){ i, lg[i] };
            min_in = cands[0].val; min_idx = 0;
            for (int j = 1; j < K; j++) if (cands[j].val < min_in) { min_in = cands[j].val; min_idx = j; }
        }
    }
    free(lg);
    qsort(cands, (size_t)nc, sizeof(CmCand), cm_cmp_cand_desc);
    float inv_t = 1.0f / (float)temp;
    float mx = cands[0].val * inv_t, psum = 0.0f;
    for (int i = 0; i < nc; i++) { cands[i].val = expf(cands[i].val * inv_t - mx); psum += cands[i].val; }
    for (int i = 0; i < nc; i++) cands[i].val /= (psum > 1e-12f ? psum : 1.0f);
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

int64_t tn_cumoe_get_arch(int64_t handle) {
    static char empty[2] = "";
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return (int64_t)(uintptr_t)empty;
    return (int64_t)(uintptr_t)g_cm[handle].arch;
}

static int64_t cm_bpe_encode_segment(CmModel* m, const unsigned char* seg, size_t seg_len, int64_t* out, int64_t max_out) {
    if (seg_len == 0 || max_out <= 0) return 0;
    if (m->n_bpe_merges > 0 && !m->bpe_use_sp) {
        int32_t* syms = (int32_t*)malloc((seg_len + 4) * sizeof(int32_t));
        size_t n_syms = 0; char u8[4];
        for (size_t i = 0; i < seg_len; i++) {
            size_t ulen = cm_utf8_encode_cp(g_cm_gpt2_b2u[seg[i]], u8);
            int32_t id = cm_tok_lookup(m, u8, ulen);
            if (id >= 0) syms[n_syms++] = id;
        }
        char comb[512];
        while (n_syms >= 2) {
            int32_t best_rank = 0x7FFFFFFF; size_t best_i = (size_t)-1; int32_t best_merged = -1;
            for (size_t i = 0; i + 1 < n_syms; i++) {
                int32_t rk = cm_merge_lookup(m, syms[i], syms[i + 1]);
                if (rk >= 0 && rk < best_rank) {
                    const char* sa = m->vocab_tokens[syms[i]]; const char* sb = m->vocab_tokens[syms[i + 1]];
                    size_t la = strlen(sa), lb = strlen(sb);
                    if (la + lb < sizeof(comb)) {
                        memcpy(comb, sa, la); memcpy(comb + la, sb, lb);
                        int32_t mid = cm_tok_lookup(m, comb, la + lb);
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
    char* enc = (char*)malloc(seg_len * 3u + 16u);
    size_t elen = 0;
    for (size_t i = 0; i < seg_len; i++) {
        if (seg[i] == ' ') {
            if (m->bpe_use_sp) { enc[elen++] = (char)0xE2; enc[elen++] = (char)0x96; enc[elen++] = (char)0x81; }
            else               { enc[elen++] = (char)0xC4; enc[elen++] = (char)0xA0; }
        } else if (seg[i] == '\n' && !m->bpe_use_sp) {
            enc[elen++] = (char)0xC4; enc[elen++] = (char)0x8A;
        } else if (!m->bpe_use_sp) {
            elen += cm_utf8_encode_cp(g_cm_gpt2_b2u[seg[i]], enc + elen);
        } else {
            enc[elen++] = (char)seg[i];
        }
    }
    int64_t n_out = 0; size_t pos = 0;
    while (pos < elen && n_out < max_out) {
        size_t max_l = (elen - pos < 48u) ? (elen - pos) : 48u;
        int32_t found = -1; size_t flen = 0;
        for (size_t l = max_l; l >= 1; l--) {
            int32_t id = cm_tok_lookup(m, enc + pos, l);
            if (id >= 0) { found = id; flen = l; break; }
        }
        if (found >= 0) { out[n_out++] = (int64_t)found; pos += flen; }
        else {
            char hex[8]; snprintf(hex, sizeof(hex), "<0x%02X>", (unsigned char)enc[pos]);
            int32_t bid = cm_tok_lookup(m, hex, 6);
            if (bid >= 0) out[n_out++] = (int64_t)bid;
            pos++;
        }
    }
    free(enc);
    return n_out;
}

static int64_t cm_bpe_encode_chunked(CmModel* m, const unsigned char* p, size_t rem, int64_t* out, int64_t max_out) {
    int64_t n_out = 0; size_t off = 0;
    while (off < rem && n_out < max_out) {
        size_t end = off + 1;
        if (p[off] == '\n') { while (end < rem && p[end] == '\n') end++; }
        else { while (end < rem && p[end] != ' ' && p[end] != '\n' && (end - off) < 64u) end++; }
        n_out += cm_bpe_encode_segment(m, p + off, end - off, out + n_out, max_out - n_out);
        off = end;
    }
    return n_out;
}

int64_t tn_cumoe_tokenize(int64_t handle, int64_t text_cstr, int64_t out_ids_ptr, int64_t max_tokens) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active || !text_cstr || !out_ids_ptr) return 0;
    CmModel* m = &g_cm[handle];
    if (!m->vocab_tokens) return 0;
    cm_bpe_ensure_tables(m);
    const char* raw = (const char*)(uintptr_t)text_cstr;
    size_t rlen = strlen(raw);
    int64_t* out = (int64_t*)(uintptr_t)out_ids_ptr;
    int64_t n_out = 0; size_t seg_start = 0, i = 0;
    while (i < rlen && n_out < max_tokens) {
        int32_t sp_id = -1; size_t sp_len = 0;
        if (raw[i] == '<' || raw[i] == '[') {
            size_t max_s = (rlen - i < 32u) ? (rlen - i) : 32u;
            for (size_t l = max_s; l >= 3; l--) {
                if ((raw[i] == '<' && raw[i + l - 1] == '>') || (raw[i] == '[' && raw[i + l - 1] == ']')) {
                    int32_t id = cm_tok_lookup(m, raw + i, l);
                    if (id >= 0) { sp_id = id; sp_len = l; break; }
                }
            }
        }
        if (sp_id >= 0) {
            if (i > seg_start) n_out += cm_bpe_encode_chunked(m, (const unsigned char*)(raw + seg_start), i - seg_start, out + n_out, max_tokens - n_out);
            if (n_out < max_tokens) out[n_out++] = (int64_t)sp_id;
            i += sp_len; seg_start = i;
        } else i++;
    }
    if (rlen > seg_start && n_out < max_tokens) {
        n_out += cm_bpe_encode_chunked(m, (const unsigned char*)(raw + seg_start), rlen - seg_start, out + n_out, max_tokens - n_out);
    }
    return n_out;
}

int64_t tn_cumoe_reset(int64_t handle) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return 0;
    CmModel* m = &g_cm[handle];
    m->cache_hits = 0; m->cache_misses = 0; m->prefetch_hits = 0;
    m->io_bytes_read = 0; m->pcie_h2d_bytes = 0;
    m->decode_phase = 1;
    return 1;
}

int64_t tn_cumoe_token_cstr(int64_t handle, int64_t tok_id) {
    static char g_tok[512]; g_tok[0] = 0;
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return (int64_t)(uintptr_t)g_tok;
    CmModel* m = &g_cm[handle];
    if (!m->vocab_tokens || tok_id < 0 || tok_id >= m->vocab_size || !m->vocab_tokens[tok_id]) return (int64_t)(uintptr_t)g_tok;
    cm_init_gpt2_tables();
    const unsigned char* s = (const unsigned char*)m->vocab_tokens[tok_id];
    if ((s[0] == '<' && (s[1] == '|' || s[1] == 's' || s[1] == '/')) || (s[0] == '[' && s[1] == 'g')) {
        strncpy(g_tok, (const char*)s, sizeof(g_tok) - 1);
        return (int64_t)(uintptr_t)g_tok;
    }
    if (s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>' && s[6] == 0) {
        unsigned int bv = 0;
        if (sscanf((const char*)s + 3, "%2x", &bv) == 1) { g_tok[0] = (char)bv; g_tok[1] = 0; return (int64_t)(uintptr_t)g_tok; }
    }
    size_t o = 0;
    for (size_t i = 0; s[i] && o + 4 < sizeof(g_tok); ) {
        if (s[i] == 0xE2 && s[i + 1] == 0x96 && s[i + 2] == 0x81) { g_tok[o++] = ' '; i += 3; }
        else if ((s[i] & 0xE0) == 0xC0 && s[i + 1]) {
            uint16_t cp = (uint16_t)(((s[i] & 0x1F) << 6) | (s[i + 1] & 0x3F));
            if (cp < 512 && g_cm_gpt2_u2b[cp] >= 0) { g_tok[o++] = (char)(uint8_t)g_cm_gpt2_u2b[cp]; i += 2; }
            else { g_tok[o++] = (char)s[i++]; g_tok[o++] = (char)s[i++]; }
        } else {
            g_tok[o++] = (char)s[i++];
        }
    }
    g_tok[o] = 0;
    return (int64_t)(uintptr_t)g_tok;
}

double tn_cumoe_now_ms(void) { return cm_now_ms(); }

double tn_cumoe_last_timing_ms(int64_t handle, int64_t which) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return 0.0;
    CmModel* m = &g_cm[handle];
    return (which == 0) ? m->last_total_ms : (which == 1) ? m->last_io_ms : m->last_gpu_ms;
}


// =========================================================================
// Tenon v6: PagedAttention 16-Token Block-Table KV + Zero-Copy COW Fork + Continuous Batching (`step_batch`)
// =========================================================================
static int cm_pa_alloc_block(CmModel* m) {
    for (int pb = 0; pb < m->pa_num_blocks; pb++) {
        if (m->pa_refcnt[pb] == 0) {
            m->pa_refcnt[pb] = 1;
            m->pa_free_blocks--;
            return pb;
        }
    }
    return -1;
}

static size_t cm_pa_kv_block_bytes(CmModel* m) {
    if (m->n_layers > 0 && m->layers[0].is_mla && m->mla_latent_kv) {
        size_t kv_r = (size_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
        return (size_t)CM_PA_BLOCK_SIZE * (kv_r + (size_t)m->rope_dim) * 2u;
    }
    size_t nkv_eff = (m->tp_size > 1) ? (size_t)(m->n_kv / m->tp_size) : (size_t)m->n_kv;
    if (nkv_eff < 1u) nkv_eff = 1u;
    return m->kv_quant_q8
        ? ((size_t)CM_PA_BLOCK_SIZE * nkv_eff * (size_t)(m->d_head >> 5) * 34u)
        : ((size_t)CM_PA_BLOCK_SIZE * (nkv_eff * (size_t)m->d_head) * 2u);
}

static int cm_pa_ensure_slot_pos(CmModel* m, int slot, int pos) {
    if (slot < 0 || slot >= CM_PA_MAX_SEQS || pos < 0) return 0;
    int lb = pos / CM_PA_BLOCK_SIZE;
    if (lb >= CM_PA_MAX_LOGICAL_BLKS) return 0;
    int32_t pb = m->pa_block_table[slot][lb];
    if (pb < 0) {
        pb = cm_pa_alloc_block(m);
        if (pb < 0) return 0;
        m->pa_block_table[slot][lb] = pb;
    } else if (m->pa_refcnt[pb] > 1) {
        // GPU-Direct VRAM Copy-on-Write split for shared partial tail block (1x GPU OR Multi-GPU TP=2..8!)
        int new_pb = cm_pa_alloc_block(m);
        if (new_pb < 0) return 0;
        size_t blk_b = cm_pa_kv_block_bytes(m);
        if (m->tp_size > 1 && m->tp) {
            for (int r = 0; r < m->tp_size; r++) {
                CmTpRank* tr = &m->tp[r];
                g_cu.cuCtxSetCurrent(g_cu.gpus[tr->dev_idx].ctx);
                for (int L = 0; L < m->n_layers; L++) {
                    CmTpLayer* tl = &tr->layers[L];
                    if (tl->k_cache_base) {
                        g_cu.cuMemcpyDtoDAsync(tl->k_cache_base + (uint64_t)new_pb * blk_b,
                                               tl->k_cache_base + (uint64_t)pb * blk_b, blk_b, tr->stream);
                    }
                    if (tl->v_cache_base) {
                        g_cu.cuMemcpyDtoDAsync(tl->v_cache_base + (uint64_t)new_pb * blk_b,
                                               tl->v_cache_base + (uint64_t)pb * blk_b, blk_b, tr->stream);
                    }
                }
            }
            g_cu.cuCtxSetCurrent(g_cu.ctx);
        } else {
            for (int L = 0; L < m->n_layers; L++) {
                CmLayer* ly = &m->layers[L];
                if (ly->k_cache_base) {
                    g_cu.cuMemcpyDtoDAsync(ly->k_cache_base + (uint64_t)new_pb * blk_b,
                                           ly->k_cache_base + (uint64_t)pb * blk_b, blk_b, m->compute_stream);
                }
                if (ly->v_cache_base) {
                    g_cu.cuMemcpyDtoDAsync(ly->v_cache_base + (uint64_t)new_pb * blk_b,
                                           ly->v_cache_base + (uint64_t)pb * blk_b, blk_b, m->compute_stream);
                }
            }
        }
        m->pa_refcnt[pb]--;
        m->pa_block_table[slot][lb] = new_pb;
        m->pa_cow_copies++;
    }
    if (pos + 1 > m->pa_seq_len[slot]) m->pa_seq_len[slot] = pos + 1;
    return 1;
}

static void cm_pa_sync_bt(CmModel* m) {
    size_t bt_bytes = (size_t)CM_PA_MAX_SEQS * CM_PA_MAX_LOGICAL_BLKS * sizeof(int32_t);
    if (m->d_pa_bt) {
        g_cu.cuMemcpyHtoDAsync(m->d_pa_bt, m->pa_block_table, bt_bytes, m->compute_stream);
    }
    if (m->tp_size > 1 && m->tp) {
        for (int r = 0; r < m->tp_size; r++) {
            if (m->tp[r].d_pa_bt) {
                g_cu.cuCtxSetCurrent(g_cu.gpus[m->tp[r].dev_idx].ctx);
                g_cu.cuMemcpyHtoDAsync(m->tp[r].d_pa_bt, m->pa_block_table, bt_bytes, m->tp[r].stream);
            }
        }
        g_cu.cuCtxSetCurrent(g_cu.ctx);
    }
}

int64_t tn_cumoe_paged_kv_info(int64_t handle, int64_t field_id) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return 0;
    CmModel* m = &g_cm[handle];
    switch (field_id) {
        case 0: return CM_PA_BLOCK_SIZE;
        case 1: return m->pa_num_blocks;
        case 2: return m->pa_free_blocks;
        case 3: return m->pa_num_blocks - m->pa_free_blocks;
        case 4: return m->pa_cow_forks;
        case 5: return m->pa_cow_copies;
        case 6: return CM_PA_MAX_SEQS;
        default: return 0;
    }
}

int64_t tn_cumoe_paged_seq_free(int64_t handle, int64_t slot_id) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return 0;
    CmModel* m = &g_cm[handle];
    if (slot_id < 0 || slot_id >= CM_PA_MAX_SEQS) return 0;
    int freed = 0;
    for (int lb = 0; lb < CM_PA_MAX_LOGICAL_BLKS; lb++) {
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
    cm_pa_sync_bt(m);
    return (int64_t)freed;
}

int64_t tn_cumoe_paged_seq_fork(int64_t handle, int64_t src_slot, int64_t dst_slot, int64_t prefix_len) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return 0;
    CmModel* m = &g_cm[handle];
    if (src_slot < 0 || src_slot >= CM_PA_MAX_SEQS || dst_slot < 0 || dst_slot >= CM_PA_MAX_SEQS || src_slot == dst_slot) return 0;
    if (prefix_len <= 0) { tn_cumoe_paged_seq_free(handle, dst_slot); return 0; }
    tn_cumoe_paged_seq_free(handle, dst_slot);
    int n_blks = (int)((prefix_len + CM_PA_BLOCK_SIZE - 1) / CM_PA_BLOCK_SIZE);
    if (n_blks > CM_PA_MAX_LOGICAL_BLKS) n_blks = CM_PA_MAX_LOGICAL_BLKS;
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
    cm_pa_sync_bt(m);
    return (int64_t)shared;
}

static void launch_layer_attn_paged_batch(CmModel* m, CUstream st, int L, uint32_t B, CUdeviceptr ring_base) {
    m->cur_exec_layer = L;
    CmLayer* ly = &m->layers[L];
    uint32_t dm = (uint32_t)m->d_model, qdim = (uint32_t)(m->n_heads * m->d_head), kvdim = (uint32_t)(m->n_kv * m->d_head);
    uint32_t wo_in = (ly->wo.ne0 > 0) ? (uint32_t)ly->wo.ne0 : qdim;
    launch_rmsnorm(st, m->d_px, &ly->attn_norm, m->d_pxn, dm, B, m->rms_eps, 1);

    // Tenon v8: Native Batched PagedAttention Compressed Latent MLA (DeepSeek-V2/V3/R1)!
    if (ly->is_mla && m->mla_latent_kv && ly->mla_wuk_f16 && ly->mla_wuv_f16) {
        uint32_t nh = (uint32_t)m->n_heads, dh = (uint32_t)m->d_head, rd = (uint32_t)m->rope_dim;
        uint32_t vd = (uint32_t)m->v_head_dim, kv_r = (uint32_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
        uint32_t nope_d = dh - rd, ldim = kv_r + rd, rn = (uint32_t)m->rope_neox, hf = (uint32_t)m->has_rope_freqs;
        uint32_t mb = (uint32_t)CM_PA_MAX_LOGICAL_BLKS, out_st = wo_in / nh;
        float theta = m->rope_theta, mla_sc = 1.0f / sqrtf((float)dh);
        if (m->yarn_mscale_k > 0.0f && m->yarn_factor > 1.0f) { float ms = 1.0f + m->yarn_mscale_k * logf(m->yarn_factor); mla_sc *= ms * ms; }
        CUdeviceptr rf = hf ? m->rope_freqs.dptr : m->d_argmax;
        if (ly->wq_a.dptr && ly->wq_b.dptr && m->q_lora_rank > 0) {
            uint32_t qr = (uint32_t)m->q_lora_rank;
            launch_gemm_batch_raw(st, ly->wq_a.dptr, ly->wq_a.ggml_type, m->d_pxn, m->d_mla_q, qr, dm, B, 0);
            if (ly->q_a_norm.dptr) launch_rmsnorm(st, m->d_mla_q, &ly->q_a_norm, m->d_mla_q, qr, B, m->rms_eps, 1);
            launch_gemm_batch_raw(st, ly->wq_b.dptr, ly->wq_b.ggml_type, m->d_mla_q, m->d_pq, qdim, qr, B, 0);
        } else {
            launch_gemm_batch_raw(st, ly->wq.dptr, ly->wq.ggml_type, m->d_pxn, m->d_pq, qdim, dm, B, 0);
        }
        launch_gemm_batch_raw(st, ly->wkv_a.dptr, ly->wkv_a.ggml_type, m->d_pxn, m->d_mla_kv, ldim, dm, B, 0);
        if (ly->kv_a_norm.dptr) {
            for (uint32_t bi = 0; bi < B; bi++) {
                CUdeviceptr cptr = m->d_mla_kv + (uint64_t)bi * ldim * 4ull;
                launch_rmsnorm(st, cptr, &ly->kv_a_norm, cptr, kv_r, 1, m->rms_eps, 0);
            }
        }
        void* rpe_a[] = { &m->d_pq, &m->d_mla_kv, &ly->k_cache_base, &rf, &m->d_cb_pos, &m->d_cb_slot, &m->d_pa_bt,
                          &nh, &nope_d, &rd, &dh, &kv_r, &theta, &rn, &hf, &B, &mb };
        g_cu.cuLaunchKernel(g_cu.k_mla_rope_store_paged, B, 1, 1, 128, 1, 1, 0, st, rpe_a, NULL);
        void* qa[] = { &m->d_pq, &ly->mla_wuk_f16, &m->d_mla_qabs, &nh, &dh, &nope_d, &rd, &kv_r, &B };
        g_cu.cuLaunchKernel(g_cu.k_mla_absorb_q, (ldim + 3u) / 4u, nh, B, 128, 1, 1, 0, st, qa, NULL);
        void* fa[] = { &m->d_mla_qabs, &ly->k_cache_base, &m->d_mla_olat, &m->d_cb_pos, &m->d_cb_slot, &m->d_pa_bt,
                       &nh, &kv_r, &rd, &mla_sc, &B, &mb };
        g_cu.cuLaunchKernel(g_cu.k_flash_mla_latent_paged, nh, B, 1, 128, 1, 1, 0, st, fa, NULL);
        void* va[] = { &m->d_mla_olat, &ly->mla_wuv_f16, &m->d_pctx, &nh, &kv_r, &vd, &out_st, &B };
        g_cu.cuLaunchKernel(g_cu.k_mla_proj_v, (out_st + 3u) / 4u, nh, B, 128, 1, 1, 0, st, va, NULL);
        launch_gemm_batch_raw(st, ly->wo.dptr, ly->wo.ggml_type, m->d_pctx, m->d_px, dm, wo_in, B, 1);
        if (m->active_lora > 0) launch_lora_delta(m, st, L, 3, m->d_pctx, m->d_px, dm, wo_in, B);
        launch_rmsnorm(st, m->d_px, &ly->ffn_norm, m->d_pxn, dm, B, m->rms_eps, 1);
        return;
    }

    CUdeviceptr pq = ring_base ? (ring_base + m->dense_off_wq) : ly->wq.dptr;
    CUdeviceptr pk = ring_base ? (ring_base + m->dense_off_wk) : ly->wk.dptr;
    CUdeviceptr pv = ring_base ? (ring_base + m->dense_off_wv) : ly->wv.dptr;
    launch_gemm_batch_raw(st, pq, ly->wq.ggml_type, m->d_pxn, m->d_pq, qdim, dm, B, 0);
    launch_gemm_batch_raw(st, pk, ly->wk.ggml_type, m->d_pxn, m->d_pk, kvdim, dm, B, 0);
    launch_gemm_batch_raw(st, pv, ly->wv.ggml_type, m->d_pxn, m->d_pv, kvdim, dm, B, 0);
    if (m->active_lora > 0) {
        launch_lora_delta(m, st, L, 0, m->d_pxn, m->d_pq, qdim, dm, B);
        launch_lora_delta(m, st, L, 1, m->d_pxn, m->d_pk, kvdim, dm, B);
        launch_lora_delta(m, st, L, 2, m->d_pxn, m->d_pv, kvdim, dm, B);
    }
    if (ly->has_qkv_bias) {
        launch_add_bias_batch(st, m->d_pq, &ly->bq, qdim, B);
        launch_add_bias_batch(st, m->d_pk, &ly->bk, kvdim, B);
        launch_add_bias_batch(st, m->d_pv, &ly->bv, kvdim, B);
    }
    if (ly->has_qk_norm) {
        if (ly->qk_norm_per_head) {
            launch_rmsnorm(st, m->d_pq, &ly->q_norm, m->d_pq, (uint32_t)m->d_head, B * (uint32_t)m->n_heads, m->rms_eps, 1);
            launch_rmsnorm(st, m->d_pk, &ly->k_norm, m->d_pk, (uint32_t)m->d_head, B * (uint32_t)m->n_kv, m->rms_eps, 1);
        } else {
            launch_rmsnorm(st, m->d_pq, &ly->q_norm, m->d_pq, qdim, B, m->rms_eps, 1);
            launch_rmsnorm(st, m->d_pk, &ly->k_norm, m->d_pk, kvdim, B, m->rms_eps, 1);
        }
    }
    uint32_t nh = (uint32_t)m->n_heads, nkv = (uint32_t)m->n_kv, dh = (uint32_t)m->d_head, rd = (uint32_t)m->rope_dim;
    uint32_t rn = (uint32_t)m->rope_neox, hf = (uint32_t)m->has_rope_freqs, mb = (uint32_t)CM_PA_MAX_LOGICAL_BLKS;
    float theta = m->rope_theta, sc = 1.0f / sqrtf((float)dh);
    CUdeviceptr rf = hf ? m->rope_freqs.dptr : m->d_argmax;
    if (m->kv_quant_q8) {
        void* ra[] = { &m->d_pq, &m->d_pk, &m->d_pv, &ly->k_cache_base, &ly->v_cache_base, &rf,
                       &m->d_cb_pos, &m->d_cb_slot, &m->d_pa_bt, &nh, &nkv, &dh, &rd, &theta, &rn, &hf, &B, &mb };
        g_cu.cuLaunchKernel(g_cu.k_rope_kv_q8_paged, nh, B, 1, 128, 1, 1, 0, st, ra, NULL);
        void* aa[] = { &m->d_pq, &ly->k_cache_base, &ly->v_cache_base, &m->d_pctx,
                       &m->d_cb_pos, &m->d_cb_slot, &m->d_pa_bt, &nh, &nkv, &dh, &B, &sc, &mb };
        g_cu.cuLaunchKernel(g_cu.k_flash_dec_q8_paged, nh, B, 1, 128, 1, 1, 0, st, aa, NULL);
    } else {
        uint32_t hd2 = dh >> 1;
        void* ra[] = { &m->d_pq, &m->d_pk, &m->d_pv, &ly->k_cache_base, &ly->v_cache_base, &rf,
                       &m->d_cb_pos, &m->d_cb_slot, &m->d_pa_bt, &nh, &nkv, &dh, &rd, &theta, &rn, &hf, &B, &mb };
        g_cu.cuLaunchKernel(g_cu.k_rope_kv_paged, (hd2 + 31u) / 32u, nh, B, 32, 1, 1, 0, st, ra, NULL);
        void* aa[] = { &m->d_pq, &ly->k_cache_base, &ly->v_cache_base, &m->d_pctx,
                       &m->d_cb_pos, &m->d_cb_slot, &m->d_pa_bt, &nh, &nkv, &dh, &B, &sc, &mb };
        g_cu.cuLaunchKernel(g_cu.k_flash_dec_paged, nh, B, 1, 128, 1, 1, 0, st, aa, NULL);
    }
    CUdeviceptr po = ring_base ? (ring_base + m->dense_off_wo) : ly->wo.dptr;
    launch_gemm_batch_raw(st, po, ly->wo.ggml_type, m->d_pctx, m->d_px, dm, wo_in, B, 1);
    if (m->active_lora > 0) launch_lora_delta(m, st, L, 3, m->d_pctx, m->d_px, dm, wo_in, B);
    launch_rmsnorm(st, m->d_px, &ly->ffn_norm, m->d_pxn, dm, B, m->rms_eps, 1);
}

// Tenon v8: Counting-Sort Grouped MoE FFN Batch on Single GPU (reads each unique active expert ONCE per batch!)
static void cm_run_grouped_moe_ffn_layer(CmModel* m, CUstream st, int L, uint32_t B) {
    CmLayer* ly = &m->layers[L];
    uint32_t dm = (uint32_t)m->d_model, df = (uint32_t)m->d_ff_exp, ne = (uint32_t)m->n_experts;
    launch_gemm_batch_raw(st, ly->router.dptr, ly->router.ggml_type, m->d_pxn, m->d_prlogits, ne, dm, B, 0);
    if (ly->has_shared_exp) {
        uint32_t sh_ff = (ly->sh_gate.ne1 > 0) ? (uint32_t)ly->sh_gate.ne1 : df;
        launch_dense_ffn_batch(m, st, ly->sh_gate.dptr, ly->sh_gate.ggml_type,
                               ly->sh_up.dptr, ly->sh_up.ggml_type, ly->sh_down.dptr, ly->sh_down.ggml_type, sh_ff, B);
    }
    g_cu.cuMemcpyDtoHAsync(m->h_prlogits, m->d_prlogits, (size_t)B * ne * 4u, st);
    g_cu.cuStreamSynchronize(st);

    uint32_t exp_tok[CM_MAX_EXPERTS][CM_PREFILL_MAX_B];
    float exp_wt[CM_MAX_EXPERTS][CM_PREFILL_MAX_B];
    uint32_t exp_cnt[CM_MAX_EXPERTS];
    memset(exp_cnt, 0, sizeof(exp_cnt));
    for (uint32_t b = 0; b < B; b++) {
        int te[CM_MAX_TOPK]; float tw[CM_MAX_TOPK];
        cm_cpu_router_topk_grp(m->h_prlogits + (size_t)b * ne, ly->exp_bias_host, (int)ne, m->top_k,
                               m->gating_func, m->expert_weights_norm, m->expert_weights_scale,
                               m->n_expert_groups, m->n_group_used, te, tw);
        for (int k = 0; k < m->top_k; k++) {
            int e = te[k];
            if (e >= 0 && e < (int)ne && exp_cnt[e] < CM_PREFILL_MAX_B) {
                uint32_t c = exp_cnt[e]++;
                exp_tok[e][c] = b;
                exp_wt[e][c] = tw[k];
            }
        }
    }
    uint32_t grp_off[CM_MAX_EXPERTS];
    uint32_t cur_off = 0, n_uniq = 0;
    for (uint32_t e = 0; e < ne; e++) {
        grp_off[e] = cur_off;
        uint32_t m_e = exp_cnt[e];
        if (m_e > 0) {
            n_uniq++;
            for (uint32_t r = 0; r < m_e; r++) {
                m->h_moe_idx[cur_off + r] = exp_tok[e][r];
                m->h_moe_wt[cur_off + r] = exp_wt[e][r];
            }
            cur_off += m_e;
        }
    }
    int64_t total_assign = (int64_t)B * (int64_t)m->top_k;
    if (total_assign > (int64_t)n_uniq) m->moe_grouped_saved_reads += (total_assign - (int64_t)n_uniq);
    m->cache_hits += total_assign;
    if (cur_off == 0) return;

    g_cu.cuMemcpyHtoDAsync(m->d_moe_idx, m->h_moe_idx, (size_t)cur_off * 4u, st);
    g_cu.cuMemcpyHtoDAsync(m->d_moe_wt,  m->h_moe_wt,  (size_t)cur_off * 4u, st);
    for (uint32_t e = 0; e < ne; e++) {
        uint32_t m_e = exp_cnt[e];
        if (m_e == 0) continue;
        int sl = ly->exp_to_slot[e];
        if (sl < 0) continue;
        CUdeviceptr ebase = ly->cache_dptr + (uint64_t)sl * ly->slot_stride_bytes;
        CUdeviceptr iptr = m->d_moe_idx + (uint64_t)grp_off[e] * 4ull;
        CUdeviceptr wptr = m->d_moe_wt  + (uint64_t)grp_off[e] * 4ull;
        void* ga[] = { &m->d_pxn, &m->d_moe_x, &iptr, &dm, &m_e };
        g_cu.cuLaunchKernel(g_cu.k_gather_rows, (dm + 255u) / 256u, m_e, 1, 256, 1, 1, 0, st, ga, NULL);
        launch_gemm_batch_raw(st, ebase, ly->gate_exps.ggml_type, m->d_moe_x, m->d_pg, df, dm, m_e, 0);
        launch_gemm_batch_raw(st, ebase + ly->gate_exp_bytes, ly->up_exps.ggml_type, m->d_moe_x, m->d_pu, df, dm, m_e, 0);
        uint32_t tot = m_e * df;
        void* swa[] = { &m->d_pg, &m->d_pu, &tot };
        g_cu.cuLaunchKernel(g_cu.k_swiglu_mul, (tot + 255u) / 256u, 1, 1, 256, 1, 1, 0, st, swa, NULL);
        launch_gemm_batch_raw(st, ebase + ly->gate_exp_bytes + ly->up_exp_bytes, ly->down_exps.ggml_type, m->d_pg, m->d_moe_y, dm, df, m_e, 0);
        void* sa[] = { &m->d_moe_y, &m->d_px, &iptr, &wptr, &dm, &m_e };
        g_cu.cuLaunchKernel(g_cu.k_scatter_add_rows, (dm + 255u) / 256u, m_e, 1, 256, 1, 1, 0, st, sa, NULL);
    }
}

// Tenon v8: Multi-GPU TP=2..8 & 2D Hybrid TP x EP PagedAttention Continuous Batching + Grouped MoE!
static int64_t cm_tp_paged_forward_batch(CmModel* m, const int64_t* ids, const uint32_t* p_arr, const uint32_t* s_arr,
                                         uint32_t B, int64_t* out_argmax_all) {
    int P = m->tp_size;
    cm_ensure_prefill_bufs(m, (int)B);
    cm_tp_ensure_bufs(m, (int)B);
    uint32_t dm = (uint32_t)m->d_model, nh_l = (uint32_t)(m->n_heads / P), nkv_l = (uint32_t)(m->n_kv / P);
    uint32_t dh = (uint32_t)m->d_head, rd = (uint32_t)m->rope_dim, vd = (uint32_t)m->v_head_dim;
    uint32_t qdim_l = nh_l * dh, kvdim_l = nkv_l * dh, dff_l = (uint32_t)(m->d_ff_dense / P);
    uint32_t rn = (uint32_t)m->rope_neox, hf = (uint32_t)m->has_rope_freqs, mb = (uint32_t)CM_PA_MAX_LOGICAL_BLKS;
    float theta = m->rope_theta;

    // Upload per-token pos/slot arrays to ALL TP ranks
    for (int r = 0; r < P; r++) {
        CmTpRank* tr = &m->tp[r];
        g_cu.cuCtxSetCurrent(g_cu.gpus[tr->dev_idx].ctx);
        g_cu.cuMemcpyHtoDAsync(tr->d_cb_pos,  p_arr, (size_t)B * 4u, tr->stream);
        g_cu.cuMemcpyHtoDAsync(tr->d_cb_slot, s_arr, (size_t)B * 4u, tr->stream);
    }

    // 1. Embed on Rank 0 & broadcast d_x[B, dm] to Ranks 1..P-1
    CmDevCtx* dc0 = &g_cu.gpus[m->tp[0].dev_idx];
    g_cu.cuCtxSetCurrent(dc0->ctx);
    uint32_t gt_emb = m->tok_embd.ggml_type;
    for (uint32_t bi = 0; bi < B; bi++) {
        CUdeviceptr dst = m->tp[0].d_x + (uint64_t)bi * (uint64_t)dm * 4ull;
        uint32_t tid = (uint32_t)ids[bi];
        CUdeviceptr ep = m->tok_embd.dptr;
        void* eargs[] = { &ep, &dst, &tid, &dm, &gt_emb };
        g_cu.cuLaunchKernel(dc0->k_embed, (dm + 255u) / 256u, 1, 1, 256, 1, 1, 0, m->tp[0].stream, eargs, NULL);
    }
    cm_tp_bcast_x(m, B * dm);

    // 2. All layers in lockstep across all P GPUs (Dense TP OR 2D TP x EP MLA + Grouped MoE!)
    for (int L = 0; L < m->n_layers; L++) {
        CmLayer* ly0 = &m->layers[L];
        for (int r = 0; r < P; r++) {
            CmTpRank* tr = &m->tp[r];
            CmTpLayer* tl = &tr->layers[L];
            CmDevCtx* dc = &g_cu.gpus[tr->dev_idx];
            g_cu.cuCtxSetCurrent(dc->ctx);
            CUstream st = tr->stream;
            uint32_t shw = (B > 1u) ? 1u : 0u;
            CUdeviceptr anw = tl->attn_norm.dptr;
            void* rn_a[] = { &tr->d_x, &anw, &tr->d_xn, &dm, &B, &m->rms_eps, &shw };
            g_cu.cuLaunchKernel(dc->k_rmsnorm, B, 1, 1, 256, 1, 1, 0, st, rn_a, NULL);
            CUdeviceptr rf = hf ? tr->rope_freqs.dptr : tr->d_x;
            uint32_t wo_in_l = (tl->wo.ne0 > 0) ? (uint32_t)tl->wo.ne0 : qdim_l;

            if (tl->is_mla && tl->wkv_a.dptr && tl->mla_wuk_f16 && tl->mla_wuv_f16) {
                uint32_t kv_r = (uint32_t)(m->kv_lora_rank > 0 ? m->kv_lora_rank : 512);
                uint32_t nope_d = dh - rd, ldim = kv_r + rd, out_st = wo_in_l / nh_l;
                float mla_sc = 1.0f / sqrtf((float)dh);
                if (m->yarn_mscale_k > 0.0f && m->yarn_factor > 1.0f) { float ms = 1.0f + m->yarn_mscale_k * logf(m->yarn_factor); mla_sc *= ms * ms; }
                if (tl->wq_a.dptr && tl->wq_b.dptr && m->q_lora_rank > 0) {
                    uint32_t qr = (uint32_t)m->q_lora_rank;
                    cm_tp_launch_gemm_dc(dc, st, tl->wq_a.dptr, tl->wq_a.ggml_type, tr->d_xn, tr->d_mla_q, qr, dm, B, 0);
                    if (tl->q_a_norm.dptr) {
                        CUdeviceptr qnw = tl->q_a_norm.dptr;
                        void* qna[] = { &tr->d_mla_q, &qnw, &tr->d_mla_q, &qr, &B, &m->rms_eps, &shw };
                        g_cu.cuLaunchKernel(dc->k_rmsnorm, B, 1, 1, 256, 1, 1, 0, st, qna, NULL);
                    }
                    cm_tp_launch_gemm_dc(dc, st, tl->wq_b.dptr, tl->wq_b.ggml_type, tr->d_mla_q, tr->d_q, qdim_l, qr, B, 0);
                } else {
                    cm_tp_launch_gemm_dc(dc, st, tl->wq.dptr, tl->wq.ggml_type, tr->d_xn, tr->d_q, qdim_l, dm, B, 0);
                }
                cm_tp_launch_gemm_dc(dc, st, tl->wkv_a.dptr, tl->wkv_a.ggml_type, tr->d_xn, tr->d_mla_kv, ldim, dm, B, 0);
                if (tl->kv_a_norm.dptr) {
                    uint32_t one = 1u, zero_u = 0u; CUdeviceptr knw = tl->kv_a_norm.dptr;
                    for (uint32_t bi = 0; bi < B; bi++) {
                        CUdeviceptr cptr = tr->d_mla_kv + (uint64_t)bi * ldim * 4ull;
                        void* kna[] = { &cptr, &knw, &cptr, &kv_r, &one, &m->rms_eps, &zero_u };
                        g_cu.cuLaunchKernel(dc->k_rmsnorm, 1, 1, 1, 256, 1, 1, 0, st, kna, NULL);
                    }
                }
                void* rpe_a[] = { &tr->d_q, &tr->d_mla_kv, &tl->k_cache_base, &rf, &tr->d_cb_pos, &tr->d_cb_slot, &tr->d_pa_bt,
                                  &nh_l, &nope_d, &rd, &dh, &kv_r, &theta, &rn, &hf, &B, &mb };
                g_cu.cuLaunchKernel(dc->k_mla_rope_store_paged, B, 1, 1, 128, 1, 1, 0, st, rpe_a, NULL);
                void* qa[] = { &tr->d_q, &tl->mla_wuk_f16, &tr->d_mla_qabs, &nh_l, &dh, &nope_d, &rd, &kv_r, &B };
                g_cu.cuLaunchKernel(dc->k_mla_absorb_q, (ldim + 3u) / 4u, nh_l, B, 128, 1, 1, 0, st, qa, NULL);
                void* fa[] = { &tr->d_mla_qabs, &tl->k_cache_base, &tr->d_mla_olat, &tr->d_cb_pos, &tr->d_cb_slot, &tr->d_pa_bt,
                               &nh_l, &kv_r, &rd, &mla_sc, &B, &mb };
                g_cu.cuLaunchKernel(dc->k_flash_mla_latent_paged, nh_l, B, 1, 128, 1, 1, 0, st, fa, NULL);
                void* va[] = { &tr->d_mla_olat, &tl->mla_wuv_f16, &tr->d_ctx, &nh_l, &kv_r, &vd, &out_st, &B };
                g_cu.cuLaunchKernel(dc->k_mla_proj_v, (out_st + 3u) / 4u, nh_l, B, 128, 1, 1, 0, st, va, NULL);
            } else {
                cm_tp_launch_gemm_dc(dc, st, tl->wq.dptr, tl->wq.ggml_type, tr->d_xn, tr->d_q, qdim_l, dm, B, 0);
                cm_tp_launch_gemm_dc(dc, st, tl->wk.dptr, tl->wk.ggml_type, tr->d_xn, tr->d_k, kvdim_l, dm, B, 0);
                cm_tp_launch_gemm_dc(dc, st, tl->wv.dptr, tl->wv.ggml_type, tr->d_xn, tr->d_v, kvdim_l, dm, B, 0);
                if (tl->bq.dptr) {
                    void* ba[] = { &tr->d_q, &tl->bq.dptr, &qdim_l, &B };  g_cu.cuLaunchKernel(dc->k_add_bias_b, (qdim_l+255u)/256u, B, 1, 256, 1, 1, 0, st, ba, NULL);
                    void* bb[] = { &tr->d_k, &tl->bk.dptr, &kvdim_l, &B }; g_cu.cuLaunchKernel(dc->k_add_bias_b, (kvdim_l+255u)/256u, B, 1, 256, 1, 1, 0, st, bb, NULL);
                    void* bc[] = { &tr->d_v, &tl->bv.dptr, &kvdim_l, &B }; g_cu.cuLaunchKernel(dc->k_add_bias_b, (kvdim_l+255u)/256u, B, 1, 256, 1, 1, 0, st, bc, NULL);
                }
                float sc = 1.0f / sqrtf((float)dh);
                if (m->kv_quant_q8) {
                    void* ra[] = { &tr->d_q, &tr->d_k, &tr->d_v, &tl->k_cache_base, &tl->v_cache_base, &rf,
                                   &tr->d_cb_pos, &tr->d_cb_slot, &tr->d_pa_bt, &nh_l, &nkv_l, &dh, &rd, &theta, &rn, &hf, &B, &mb };
                    g_cu.cuLaunchKernel(dc->k_rope_kv_q8_paged, nh_l, B, 1, 128, 1, 1, 0, st, ra, NULL);
                    void* aa[] = { &tr->d_q, &tl->k_cache_base, &tl->v_cache_base, &tr->d_ctx,
                                   &tr->d_cb_pos, &tr->d_cb_slot, &tr->d_pa_bt, &nh_l, &nkv_l, &dh, &B, &sc, &mb };
                    g_cu.cuLaunchKernel(dc->k_flash_dec_q8_paged, nh_l, B, 1, 128, 1, 1, 0, st, aa, NULL);
                } else {
                    uint32_t hd2 = dh >> 1;
                    void* ra[] = { &tr->d_q, &tr->d_k, &tr->d_v, &tl->k_cache_base, &tl->v_cache_base, &rf,
                                   &tr->d_cb_pos, &tr->d_cb_slot, &tr->d_pa_bt, &nh_l, &nkv_l, &dh, &rd, &theta, &rn, &hf, &B, &mb };
                    g_cu.cuLaunchKernel(dc->k_rope_kv_paged, (hd2 + 31u) / 32u, nh_l, B, 32, 1, 1, 0, st, ra, NULL);
                    void* aa[] = { &tr->d_q, &tl->k_cache_base, &tl->v_cache_base, &tr->d_ctx,
                                   &tr->d_cb_pos, &tr->d_cb_slot, &tr->d_pa_bt, &nh_l, &nkv_l, &dh, &B, &sc, &mb };
                    g_cu.cuLaunchKernel(dc->k_flash_dec_paged, nh_l, B, 1, 128, 1, 1, 0, st, aa, NULL);
                }
            }
            cm_tp_launch_gemm_dc(dc, st, tl->wo.dptr, tl->wo.ggml_type, tr->d_ctx, tr->d_tp_part, dm, wo_in_l, B, 0);
        }
        cm_tp_allreduce_add(m, B * dm, 2 + L * 2);

        if (m->dense_model || ly0->is_dense) {
            for (int r = 0; r < P; r++) {
                CmTpRank* tr = &m->tp[r];
                CmTpLayer* tl = &tr->layers[L];
                CmDevCtx* dc = &g_cu.gpus[tr->dev_idx];
                g_cu.cuCtxSetCurrent(dc->ctx);
                CUstream st = tr->stream;
                uint32_t shw = (B > 1u) ? 1u : 0u;
                CUdeviceptr fnw = tl->ffn_norm.dptr;
                void* rn_f[] = { &tr->d_x, &fnw, &tr->d_xn, &dm, &B, &m->rms_eps, &shw };
                g_cu.cuLaunchKernel(dc->k_rmsnorm, B, 1, 1, 256, 1, 1, 0, st, rn_f, NULL);
                cm_tp_launch_gemm_dc(dc, st, tl->ffn_gate.dptr, tl->ffn_gate.ggml_type, tr->d_xn, tr->d_g, dff_l, dm, B, 0);
                cm_tp_launch_gemm_dc(dc, st, tl->ffn_up.dptr,   tl->ffn_up.ggml_type,   tr->d_xn, tr->d_u, dff_l, dm, B, 0);
                uint32_t tot = B * dff_l;
                void* swa[] = { &tr->d_g, &tr->d_u, &tot };
                g_cu.cuLaunchKernel(dc->k_swiglu_mul, (tot + 255u) / 256u, 1, 1, 256, 1, 1, 0, st, swa, NULL);
                cm_tp_launch_gemm_dc(dc, st, tl->ffn_down.dptr, tl->ffn_down.ggml_type, tr->d_g, tr->d_tp_part, dm, dff_l, B, 0);
            }
            cm_tp_allreduce_add(m, B * dm, 2 + L * 2 + 1);
        } else {
            // 2D Hybrid TP x EP Batched Counting-Sort Grouped MoE GEMM!
            for (int r = 0; r < P; r++) {
                CmTpRank* tr = &m->tp[r];
                CmTpLayer* tl = &tr->layers[L];
                CmDevCtx* dc = &g_cu.gpus[tr->dev_idx];
                g_cu.cuCtxSetCurrent(dc->ctx);
                uint32_t shw = (B > 1u) ? 1u : 0u;
                CUdeviceptr fnw = tl->ffn_norm.dptr;
                void* rn_f[] = { &tr->d_x, &fnw, &tr->d_xn, &dm, &B, &m->rms_eps, &shw };
                g_cu.cuLaunchKernel(dc->k_rmsnorm, B, 1, 1, 256, 1, 1, 0, tr->stream, rn_f, NULL);
            }
            g_cu.cuCtxSetCurrent(dc0->ctx);
            CUstream st0 = m->tp[0].stream;
            uint32_t ne = (uint32_t)m->n_experts, df = (uint32_t)m->d_ff_exp;
            cm_tp_launch_gemm_dc(dc0, st0, ly0->router.dptr, ly0->router.ggml_type, m->tp[0].d_xn, m->d_prlogits, ne, dm, B, 0);
            g_cu.cuMemcpyDtoHAsync(m->h_prlogits, m->d_prlogits, (size_t)B * ne * 4u, st0);
            g_cu.cuStreamSynchronize(st0);

            uint32_t exp_tok[CM_MAX_EXPERTS][CM_PREFILL_MAX_B];
            float exp_wt[CM_MAX_EXPERTS][CM_PREFILL_MAX_B];
            uint32_t exp_cnt[CM_MAX_EXPERTS];
            memset(exp_cnt, 0, sizeof(exp_cnt));
            for (uint32_t b = 0; b < B; b++) {
                int te[CM_MAX_TOPK]; float tw[CM_MAX_TOPK];
                cm_cpu_router_topk_grp(m->h_prlogits + (size_t)b * ne, ly0->exp_bias_host, (int)ne, m->top_k,
                                       m->gating_func, m->expert_weights_norm, m->expert_weights_scale,
                                       m->n_expert_groups, m->n_group_used, te, tw);
                for (int k = 0; k < m->top_k; k++) {
                    int e = te[k];
                    if (e >= 0 && e < (int)ne && exp_cnt[e] < CM_PREFILL_MAX_B) {
                        uint32_t c = exp_cnt[e]++;
                        exp_tok[e][c] = b;
                        exp_wt[e][c] = tw[k];
                    }
                }
            }
            uint32_t n_uniq = 0;
            for (uint32_t e = 0; e < ne; e++) if (exp_cnt[e] > 0) n_uniq++;
            int64_t total_assign = (int64_t)B * (int64_t)m->top_k;
            if (total_assign > (int64_t)n_uniq) m->moe_grouped_saved_reads += (total_assign - (int64_t)n_uniq);
            m->cache_hits += total_assign;

            int Er = m->ep_exp_per_rank;
            uint32_t h_idx_r[CM_PREFILL_MAX_B * CM_MAX_TOPK];
            float h_wt_r[CM_PREFILL_MAX_B * CM_MAX_TOPK];
            uint32_t grp_off[CM_MAX_EXPERTS];
            for (int r = 0; r < P; r++) {
                CmTpRank* tr = &m->tp[r];
                CmEpRank* er = &m->ep[r];
                CmDevCtx* dc = &g_cu.gpus[tr->dev_idx];
                g_cu.cuCtxSetCurrent(dc->ctx);
                CUstream st = tr->stream;
                if (r == 0 && ly0->has_shared_exp) {
                    uint32_t sh_ff = (ly0->sh_gate.ne1 > 0) ? (uint32_t)ly0->sh_gate.ne1 : df;
                    cm_tp_launch_gemm_dc(dc, st, ly0->sh_gate.dptr, ly0->sh_gate.ggml_type, tr->d_xn, m->d_pg, sh_ff, dm, B, 0);
                    cm_tp_launch_gemm_dc(dc, st, ly0->sh_up.dptr,   ly0->sh_up.ggml_type,   tr->d_xn, m->d_pu, sh_ff, dm, B, 0);
                    uint32_t stot = B * sh_ff;
                    void* swa[] = { &m->d_pg, &m->d_pu, &stot };
                    g_cu.cuLaunchKernel(dc->k_swiglu_mul, (stot + 255u) / 256u, 1, 1, 256, 1, 1, 0, st, swa, NULL);
                    cm_tp_launch_gemm_dc(dc, st, ly0->sh_down.dptr, ly0->sh_down.ggml_type, m->d_pg, tr->d_tp_part, dm, sh_ff, B, 0);
                } else {
                    if (g_cu.cuMemsetD8Async) g_cu.cuMemsetD8Async(tr->d_tp_part, 0, (size_t)B * dm * 4u, st);
                    else g_cu.cuMemsetD8(tr->d_tp_part, 0, (size_t)B * dm * 4u);
                }
                int e0 = r * Er, e1 = (r == P - 1) ? (int)ne : (e0 + Er);
                uint32_t cur_r = 0;
                for (int e = e0; e < e1; e++) {
                    grp_off[e] = cur_r;
                    for (uint32_t j = 0; j < exp_cnt[e]; j++) {
                        h_idx_r[cur_r + j] = exp_tok[e][j];
                        h_wt_r[cur_r + j] = exp_wt[e][j];
                    }
                    cur_r += exp_cnt[e];
                }
                if (cur_r > 0) {
                    g_cu.cuMemcpyHtoD(tr->d_moe_idx, h_idx_r, (size_t)cur_r * 4u);
                    g_cu.cuMemcpyHtoD(tr->d_moe_wt,  h_wt_r,  (size_t)cur_r * 4u);
                    for (int e = e0; e < e1; e++) {
                        uint32_t m_e = exp_cnt[e];
                        if (m_e == 0) continue;
                        int ls = e - e0;
                        CUdeviceptr ebase = er->cache_dptr[L] + (uint64_t)ls * ly0->slot_stride_bytes;
                        CUdeviceptr iptr = tr->d_moe_idx + (uint64_t)grp_off[e] * 4ull;
                        CUdeviceptr wptr = tr->d_moe_wt  + (uint64_t)grp_off[e] * 4ull;
                        void* ga[] = { &tr->d_xn, &tr->d_moe_x, &iptr, &dm, &m_e };
                        g_cu.cuLaunchKernel(dc->k_gather_rows, (dm + 255u) / 256u, m_e, 1, 256, 1, 1, 0, st, ga, NULL);
                        cm_tp_launch_gemm_dc(dc, st, ebase, ly0->gate_exps.ggml_type, tr->d_moe_x, tr->d_g, df, dm, m_e, 0);
                        cm_tp_launch_gemm_dc(dc, st, ebase + ly0->gate_exp_bytes, ly0->up_exps.ggml_type, tr->d_moe_x, tr->d_u, df, dm, m_e, 0);
                        uint32_t tot = m_e * df;
                        void* swa[] = { &tr->d_g, &tr->d_u, &tot };
                        g_cu.cuLaunchKernel(dc->k_swiglu_mul, (tot + 255u) / 256u, 1, 1, 256, 1, 1, 0, st, swa, NULL);
                        cm_tp_launch_gemm_dc(dc, st, ebase + ly0->gate_exp_bytes + ly0->up_exp_bytes, ly0->down_exps.ggml_type, tr->d_g, tr->d_moe_y, dm, df, m_e, 0);
                        void* sa[] = { &tr->d_moe_y, &tr->d_tp_part, &iptr, &wptr, &dm, &m_e };
                        g_cu.cuLaunchKernel(dc->k_scatter_add_rows, (dm + 255u) / 256u, m_e, 1, 256, 1, 1, 0, st, sa, NULL);
                    }
                }
            }
            cm_tp_allreduce_add(m, B * dm, 2 + L * 2 + 1);
        }
    }

    // 3. Final Out-Norm + LM-Head + Argmax on GPU 0
    g_cu.cuCtxSetCurrent(dc0->ctx);
    CUstream st0 = m->tp[0].stream;
    uint32_t v = (uint32_t)m->vocab_size;
    int64_t last_tok = 0;
    if (out_argmax_all != NULL) {
        if (!m->d_argmax_batch) {
            g_cu.cuMemAlloc(&m->d_argmax_batch, (size_t)CM_PREFILL_MAX_B * 4u);
            g_cu.cuMemHostAlloc((void**)&m->h_argmax_batch, (size_t)CM_PREFILL_MAX_B * 4u, CU_MEMHOSTALLOC_PORTABLE);
        }
        launch_rmsnorm(st0, m->tp[0].d_x, &m->out_norm, m->tp[0].d_xn, dm, B, m->rms_eps, 1);
        launch_gemm_batch_raw(st0, m->lm_head.dptr, m->lm_head.ggml_type, m->tp[0].d_xn, m->d_batch_logits, v, dm, B, 0);
        void* aa[] = { &m->d_batch_logits, &m->d_argmax_batch, &v, &B };
        g_cu.cuLaunchKernel(g_cu.k_argmax_b, B, 1, 1, 256, 1, 1, 0, st0, aa, NULL);
        g_cu.cuMemcpyDtoHAsync(m->h_argmax_batch, m->d_argmax_batch, (size_t)B * 4u, st0);
        g_cu.cuMemcpyDtoDAsync(m->d_logits, m->d_batch_logits + (uint64_t)(B - 1u) * (uint64_t)v * 4ull, (size_t)v * 4u, st0);
        if (m->d_last_hidden) {
            g_cu.cuMemcpyDtoDAsync(m->d_last_hidden, m->tp[0].d_x + (uint64_t)(B - 1u) * (uint64_t)dm * 4ull, (size_t)dm * 4u, st0);
        }
        g_cu.cuStreamSynchronize(st0);
        for (uint32_t bi = 0; bi < B; bi++) { out_argmax_all[bi] = (int64_t)m->h_argmax_batch[bi]; last_tok = out_argmax_all[bi]; }
    } else {
        CUdeviceptr lx = m->tp[0].d_x + (uint64_t)(B - 1u) * (uint64_t)dm * 4ull;
        if (m->d_last_hidden) g_cu.cuMemcpyDtoDAsync(m->d_last_hidden, lx, (size_t)dm * 4u, st0);
        launch_rmsnorm(st0, lx, &m->out_norm, m->d_xn, dm, 1, m->rms_eps, 0);
        launch_gemv(st0, &m->lm_head, m->d_xn, m->d_logits, v, dm, 0);
        void* aa[] = { &m->d_logits, &m->d_argmax, &v };
        g_cu.cuLaunchKernel(g_cu.k_argmax, 1, 1, 1, 256, 1, 1, 0, st0, aa, NULL);
        g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, st0);
        g_cu.cuStreamSynchronize(st0);
        last_tok = (int64_t)m->h_argmax[0];
    }
    for (int r = 1; r < P; r++) {
        g_cu.cuCtxSetCurrent(g_cu.gpus[m->tp[r].dev_idx].ctx);
        g_cu.cuStreamSynchronize(m->tp[r].stream);
    }
    g_cu.cuCtxSetCurrent(g_cu.ctx);
    return last_tok;
}

static void cm_run_dense_paged_layers_batch(CmModel* m, uint32_t B) {
    uint32_t dm = (uint32_t)m->d_model;
    if (m->resident_mode) {
        for (int L = 0; L < m->n_layers; L++) {
            CmLayer* ly = &m->layers[L];
            launch_layer_attn_paged_batch(m, m->compute_stream, L, B, 0);
            if (ly->is_dense) {
                launch_dense_ffn_batch(m, m->compute_stream, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                                       ly->ffn_up.dptr, ly->ffn_up.ggml_type, ly->ffn_down.dptr, ly->ffn_down.ggml_type,
                                       (uint32_t)m->d_ff_dense, B);
            } else {
                cm_run_grouped_moe_ffn_layer(m, m->compute_stream, L, B);
            }
        }
    } else {
        int first_stream = m->cache_slots_per_layer;
        for (int L = 0; L < first_stream && L < m->n_layers; L++) {
            CmLayer* ly = &m->layers[L];
            launch_layer_attn_paged_batch(m, m->compute_stream, L, B, 0);
            launch_dense_ffn_batch(m, m->compute_stream, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                                   ly->ffn_up.dptr, ly->ffn_up.ggml_type, ly->ffn_down.dptr, ly->ffn_down.ggml_type,
                                   (uint32_t)m->d_ff_dense, B);
        }
        if (m->enable_overlap && !m->use_nocache && m->cpu_pool_init && m->dense_pin_all_attn) {
            CmWeightRef pacc_ref = {0}; pacc_ref.dptr = m->d_pxn;
            for (int L = first_stream; L < m->n_layers; L++) {
                launch_layer_attn_paged_batch(m, m->compute_stream, L, B, 0);
                g_cu.cuMemcpyDtoHAsync(m->h_pxn, m->d_pxn, (size_t)B * dm * 4u, m->compute_stream);
                g_cu.cuStreamSynchronize(m->compute_stream);
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
                CmLayer* ly = &m->layers[L];
                if (ly->repack_gu != NULL && ly->repack_down != NULL) {
                    for (uint32_t bi = 0; bi < B; bi += (uint32_t)CM_CPU_MAX_B) {
                        uint32_t bt = (B - bi > (uint32_t)CM_CPU_MAX_B) ? (uint32_t)CM_CPU_MAX_B : (B - bi);
                        m->cpu_batch_b = (int)bt; m->cpu_batch_off = (size_t)bi;
                        cm_cpu_pool_run(m, 4, L, 0, NULL, NULL); cm_cpu_pool_wait(m);
                    }
                } else
#endif
                {
                    for (uint32_t bi = 0; bi < B; bi++) {
                        memcpy(m->h_xn, m->h_pxn + (size_t)bi * dm, (size_t)dm * 4u);
                        cm_cpu_pool_run(m, 2, L, 0, NULL, NULL); cm_cpu_pool_wait(m);
                        memcpy(m->h_pcpu_accum + (size_t)bi * dm, m->h_cpu_accum, (size_t)dm * 4u);
                    }
                }
                g_cu.cuMemcpyHtoDAsync(m->d_pxn, m->h_pcpu_accum, (size_t)B * dm * 4u, m->compute_stream);
                launch_add_bias(m->compute_stream, m->d_px, &pacc_ref, B * dm);
            }
        } else {
            int n_ring = m->dense_pin_all_attn ? 4 : 2;
            for (int L = first_stream; L < m->n_layers; L++) {
                CmLayer* ly = &m->layers[L];
                int cur = (L - first_stream) % n_ring;
                cm_stage_dense_layer(m, L, m->dense_ring_host[cur]);
                g_cu.cuMemcpyHtoD(m->dense_ring_dptr[cur], m->dense_ring_host[cur], (size_t)m->dense_ring_stride);
                launch_layer_attn_paged_batch(m, m->compute_stream, L, B, m->dense_pin_all_attn ? 0 : m->dense_ring_dptr[cur]);
                launch_dense_ffn_batch(m, m->compute_stream,
                                       m->dense_ring_dptr[cur] + m->dense_off_gate, ly->ffn_gate.ggml_type,
                                       m->dense_ring_dptr[cur] + m->dense_off_up,   ly->ffn_up.ggml_type,
                                       m->dense_ring_dptr[cur] + m->dense_off_down, ly->ffn_down.ggml_type,
                                       (uint32_t)m->d_ff_dense, B);
                g_cu.cuStreamSynchronize(m->compute_stream);
            }
        }
    }
}

static int cm_supports_paged_cb(CmModel* m) {
    int is_mla_m = (m->n_layers > 0 && m->layers[0].is_mla);
    if (m->tp_size > 1 && m->tp) {
        if (m->dense_model) return !is_mla_m;
        return (m->hybrid_2d_mode && m->ep != NULL && (!is_mla_m || (m->mla_latent_kv && m->tp[0].layers[0].mla_wuk_f16 != 0)));
    }
    if (m->dense_model) return !is_mla_m || (m->mla_latent_kv && m->layers[0].mla_wuk_f16 != 0);
    return m->resident_mode && (!is_mla_m || (m->mla_latent_kv && m->layers[0].mla_wuk_f16 != 0));
}

int64_t tn_cumoe_step_batch(int64_t handle, int64_t tok_ids_ptr, int64_t pos_ptr, int64_t slot_ids_ptr, int64_t n_active, int64_t out_toks_ptr) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active || !tok_ids_ptr || !pos_ptr || !slot_ids_ptr || !out_toks_ptr || n_active <= 0) return 0;
    CmModel* m = &g_cm[handle];
    const int64_t* ids = (const int64_t*)(uintptr_t)tok_ids_ptr;
    const int64_t* pos = (const int64_t*)(uintptr_t)pos_ptr;
    const int64_t* slt = (const int64_t*)(uintptr_t)slot_ids_ptr;
    int64_t* out = (int64_t*)(uintptr_t)out_toks_ptr;
    if (n_active > CM_PREFILL_MAX_B) n_active = CM_PREFILL_MAX_B;
    uint32_t B = (uint32_t)n_active;

    if (m->batch_logits_cap < (int)B) {
        int ncap = (int)(B < 16 ? 16 : B);
        if (m->d_batch_logits) g_cu.cuMemFree(m->d_batch_logits);
        g_cu.cuMemAlloc(&m->d_batch_logits, (size_t)ncap * (size_t)m->vocab_size * 4u);
        m->batch_logits_cap = ncap;
    }

    if (!cm_supports_paged_cb(m)) {
        for (uint32_t b = 0; b < B; b++) {
            tn_cumoe_kv_slot_set(handle, slt[b] % (m->n_kv_slots > 0 ? m->n_kv_slots : 1));
            out[b] = tn_cumoe_step(handle, ids[b], pos[b]);
            g_cu.cuMemcpyDtoDAsync(m->d_batch_logits + (uint64_t)b * (uint64_t)m->vocab_size * 4ull,
                                   m->d_logits, (size_t)m->vocab_size * 4u, m->compute_stream);
        }
        g_cu.cuStreamSynchronize(m->compute_stream);
        return (int64_t)B;
    }

    double t0 = cm_now_ms();
    uint32_t p_arr[CM_PREFILL_MAX_B], s_arr[CM_PREFILL_MAX_B];
    for (uint32_t b = 0; b < B; b++) {
        int s = (int)slt[b]; if (s < 0 || s >= CM_PA_MAX_SEQS) s = 0;
        int p = (int)pos[b]; if (p < 0) p = 0;
        if (!cm_pa_ensure_slot_pos(m, s, p)) return 0;
        p_arr[b] = (uint32_t)p; s_arr[b] = (uint32_t)s;
    }
    if (m->tp_size > 1 && m->tp) {
        cm_tp_ensure_bufs(m, (int)B);
        cm_pa_sync_bt(m);
        cm_tp_paged_forward_batch(m, ids, p_arr, s_arr, B, out);
        m->last_total_ms = cm_now_ms() - t0;
        return (int64_t)B;
    }

    cm_ensure_prefill_bufs(m, (int)B);
    cm_pa_sync_bt(m);
    g_cu.cuMemcpyHtoDAsync(m->d_cb_pos,  p_arr, (size_t)B * 4u, m->compute_stream);
    g_cu.cuMemcpyHtoDAsync(m->d_cb_slot, s_arr, (size_t)B * 4u, m->compute_stream);

    uint32_t dm = (uint32_t)m->d_model, gt = m->tok_embd.ggml_type;
    for (uint32_t bi = 0; bi < B; bi++) {
        CUdeviceptr dst = m->d_px + (uint64_t)bi * (uint64_t)dm * 4ull;
        uint32_t tid = (uint32_t)ids[bi];
        if (m->resident_mode && m->tok_embd.dptr) {
            CUdeviceptr ep = m->tok_embd.dptr;
            void* args[] = { &ep, &dst, &tid, &dm, &gt };
            g_cu.cuLaunchKernel(g_cu.k_embed, (dm + 255u) / 256u, 1, 1, 256, 1, 1, 0, m->compute_stream, args, NULL);
        } else {
            uint64_t rb = cm_ggml_type_bytes(gt, (uint64_t)dm);
            cm_read_slice(m, m->tok_embd.shard_idx, m->h_emb_row, (size_t)rb, m->tok_embd.file_off + (uint64_t)tid * rb);
            g_cu.cuMemcpyHtoD(m->d_emb_row, m->h_emb_row, (size_t)rb);
            uint32_t z = 0; void* args[] = { &m->d_emb_row, &dst, &z, &dm, &gt };
            g_cu.cuLaunchKernel(g_cu.k_embed, (dm + 255u) / 256u, 1, 1, 256, 1, 1, 0, m->compute_stream, args, NULL);
            g_cu.cuStreamSynchronize(m->compute_stream);
        }
    }
    cm_run_dense_paged_layers_batch(m, B);

    // Batched Final Norm + Batched LM Head + Single-Launch Parallel Argmax directly on GPU!
    uint32_t v = (uint32_t)m->vocab_size;
    if (!m->d_argmax_batch) {
        g_cu.cuMemAlloc(&m->d_argmax_batch, (size_t)CM_PREFILL_MAX_B * 4u);
        g_cu.cuMemHostAlloc((void**)&m->h_argmax_batch, (size_t)CM_PREFILL_MAX_B * 4u, CU_MEMHOSTALLOC_PORTABLE);
    }
    launch_rmsnorm(m->compute_stream, m->d_px, &m->out_norm, m->d_pxn, dm, B, m->rms_eps, 1);
    launch_gemm_batch_raw(m->compute_stream, m->lm_head.dptr, m->lm_head.ggml_type, m->d_pxn, m->d_batch_logits, v, dm, B, 0);
    void* aa[] = { &m->d_batch_logits, &m->d_argmax_batch, &v, &B };
    g_cu.cuLaunchKernel(g_cu.k_argmax_b, B, 1, 1, 256, 1, 1, 0, m->compute_stream, aa, NULL);
    g_cu.cuMemcpyDtoHAsync(m->h_argmax_batch, m->d_argmax_batch, (size_t)B * 4u, m->compute_stream);
    g_cu.cuMemcpyDtoDAsync(m->d_logits, m->d_batch_logits + (uint64_t)(B - 1u) * (uint64_t)v * 4ull, (size_t)v * 4u, m->compute_stream);
    if (m->d_last_hidden) {
        g_cu.cuMemcpyDtoDAsync(m->d_last_hidden, m->d_px + (uint64_t)(B - 1u) * (uint64_t)dm * 4ull, (size_t)dm * 4u, m->compute_stream);
    }
    g_cu.cuStreamSynchronize(m->compute_stream);
    for (uint32_t b = 0; b < B; b++) out[b] = (int64_t)m->h_argmax_batch[b];
    m->last_total_ms = cm_now_ms() - t0;
    return (int64_t)B;
}

// Tenon v8: Native Zero-Migration In-Slot Speculative Verification directly on PagedAttention `slot_id`
// (Works across 1x Dense, Multi-GPU TP=2..8, Latent MLA, EP=1 MoE, and 2D TP x EP with ZERO KV migration!)
int64_t tn_cumoe_paged_verify_batch(int64_t handle, int64_t slot_id, int64_t tok_ids_ptr, int64_t n_toks, int64_t start_pos, int64_t out_argmax_ptr) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active || !tok_ids_ptr || !out_argmax_ptr || n_toks <= 0) return 0;
    if (n_toks > CM_PREFILL_MAX_B) n_toks = CM_PREFILL_MAX_B;
    int64_t pos_arr[CM_PREFILL_MAX_B], slot_arr[CM_PREFILL_MAX_B];
    for (int64_t i = 0; i < n_toks; i++) { pos_arr[i] = start_pos + i; slot_arr[i] = slot_id; }
    return tn_cumoe_step_batch(handle, tok_ids_ptr, (int64_t)(uintptr_t)pos_arr, (int64_t)(uintptr_t)slot_arr, n_toks, out_argmax_ptr);
}

int64_t tn_cumoe_copy_logits_batch(int64_t handle, int64_t batch_idx, int64_t dst_f32_ptr) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active || !dst_f32_ptr) return 0;
    CmModel* m = &g_cm[handle];
    if (!m->d_batch_logits || batch_idx < 0 || batch_idx >= m->batch_logits_cap) return 0;
    g_cu.cuCtxSetCurrent(g_cu.ctx);
    g_cu.cuMemcpyDtoH((void*)(uintptr_t)dst_f32_ptr,
                      m->d_batch_logits + (uint64_t)batch_idx * (uint64_t)m->vocab_size * 4ull,
                      (size_t)m->vocab_size * sizeof(float));
    return (int64_t)m->vocab_size;
}

int64_t tn_cumoe_paged_prefill(int64_t handle, int64_t slot_id, int64_t tok_ids_ptr, int64_t n_toks, int64_t start_pos) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active || !tok_ids_ptr || n_toks <= 0) return 0;
    CmModel* m = &g_cm[handle];
    const int64_t* ids = (const int64_t*)(uintptr_t)tok_ids_ptr;
    if (!cm_supports_paged_cb(m)) {
        tn_cumoe_kv_slot_set(handle, slot_id % (m->n_kv_slots > 0 ? m->n_kv_slots : 1));
        return tn_cumoe_prefill(handle, tok_ids_ptr, n_toks, start_pos);
    }
    double t0 = cm_now_ms();
    int64_t last_tok = 0;
    int s = (int)slot_id; if (s < 0 || s >= CM_PA_MAX_SEQS) s = 0;
    int chunk_sz = 64;
    if (m->tp_size > 1 && m->tp) {
        int64_t off = 0;
        while (off < n_toks) {
            uint32_t B = (uint32_t)((n_toks - off > chunk_sz) ? chunk_sz : (n_toks - off));
            uint32_t p_arr[CM_PREFILL_MAX_B], s_arr[CM_PREFILL_MAX_B];
            for (uint32_t b = 0; b < B; b++) {
                int p = (int)(start_pos + off + b);
                if (!cm_pa_ensure_slot_pos(m, s, p)) return 0;
                p_arr[b] = (uint32_t)p; s_arr[b] = (uint32_t)s;
            }
            cm_tp_ensure_bufs(m, (int)B);
            cm_pa_sync_bt(m);
            last_tok = cm_tp_paged_forward_batch(m, ids + off, p_arr, s_arr, B, NULL);
            off += B;
        }
        m->last_total_ms = cm_now_ms() - t0;
        return last_tok;
    }
    cm_ensure_prefill_bufs(m, chunk_sz);
    uint32_t dm = (uint32_t)m->d_model, gt = m->tok_embd.ggml_type;
    int64_t off = 0;
    while (off < n_toks) {
        uint32_t B = (uint32_t)((n_toks - off > chunk_sz) ? chunk_sz : (n_toks - off));
        uint32_t p_arr[CM_PREFILL_MAX_B], s_arr[CM_PREFILL_MAX_B];
        for (uint32_t b = 0; b < B; b++) {
            int p = (int)(start_pos + off + b);
            if (!cm_pa_ensure_slot_pos(m, s, p)) return 0;
            p_arr[b] = (uint32_t)p; s_arr[b] = (uint32_t)s;
        }
        cm_pa_sync_bt(m);
        g_cu.cuMemcpyHtoDAsync(m->d_cb_pos,  p_arr, (size_t)B * 4u, m->compute_stream);
        g_cu.cuMemcpyHtoDAsync(m->d_cb_slot, s_arr, (size_t)B * 4u, m->compute_stream);
        for (uint32_t bi = 0; bi < B; bi++) {
            CUdeviceptr dst = m->d_px + (uint64_t)bi * (uint64_t)dm * 4ull;
            uint32_t tid = (uint32_t)ids[off + bi];
            if (m->resident_mode && m->tok_embd.dptr) {
                CUdeviceptr ep = m->tok_embd.dptr;
                void* args[] = { &ep, &dst, &tid, &dm, &gt };
                g_cu.cuLaunchKernel(g_cu.k_embed, (dm + 255u) / 256u, 1, 1, 256, 1, 1, 0, m->compute_stream, args, NULL);
            } else {
                uint64_t rb = cm_ggml_type_bytes(gt, (uint64_t)dm);
                cm_read_slice(m, m->tok_embd.shard_idx, m->h_emb_row, (size_t)rb, m->tok_embd.file_off + (uint64_t)tid * rb);
                g_cu.cuMemcpyHtoD(m->d_emb_row, m->h_emb_row, (size_t)rb);
                uint32_t z = 0; void* args[] = { &m->d_emb_row, &dst, &z, &dm, &gt };
                g_cu.cuLaunchKernel(g_cu.k_embed, (dm + 255u) / 256u, 1, 1, 256, 1, 1, 0, m->compute_stream, args, NULL);
                g_cu.cuStreamSynchronize(m->compute_stream);
            }
        }
        cm_run_dense_paged_layers_batch(m, B);
        if (off + B == n_toks) {
            uint32_t v = (uint32_t)m->vocab_size;
            CUdeviceptr lx = m->d_px + (uint64_t)(B - 1u) * (uint64_t)dm * 4ull;
            if (m->d_last_hidden) g_cu.cuMemcpyDtoDAsync(m->d_last_hidden, lx, (size_t)dm * 4u, m->compute_stream);
            launch_rmsnorm(m->compute_stream, lx, &m->out_norm, m->d_xn, dm, 1, m->rms_eps, 0);
            launch_gemv(m->compute_stream, &m->lm_head, m->d_xn, m->d_logits, v, dm, 0);
            void* aa[] = { &m->d_logits, &m->d_argmax, &v };
            g_cu.cuLaunchKernel(g_cu.k_argmax, 1, 1, 1, 256, 1, 1, 0, m->compute_stream, aa, NULL);
            g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, m->compute_stream);
            g_cu.cuStreamSynchronize(m->compute_stream);
            last_tok = (int64_t)m->h_argmax[0];
        } else {
            g_cu.cuStreamSynchronize(m->compute_stream);
        }
        off += B;
    }
    m->last_total_ms = cm_now_ms() - t0;
    return last_tok;
}

// =========================================================================
// Tenon v6: Tree-Attention Speculative Verification (`verify_tree` + `commit_tree_path`)
// =========================================================================
static void launch_layer_attn_tree(CmModel* m, CUstream st, int L, uint32_t start_pos, uint32_t B, CUdeviceptr ring_base) {
    m->cur_exec_layer = L;
    CmLayer* ly = &m->layers[L];
    uint32_t dm = (uint32_t)m->d_model, qdim = (uint32_t)(m->n_heads * m->d_head), kvdim = (uint32_t)(m->n_kv * m->d_head);
    uint32_t wo_in = (ly->wo.ne0 > 0) ? (uint32_t)ly->wo.ne0 : qdim;
    launch_rmsnorm(st, m->d_px, &ly->attn_norm, m->d_pxn, dm, B, m->rms_eps, 1);
    CUdeviceptr pq = ring_base ? (ring_base + m->dense_off_wq) : ly->wq.dptr;
    CUdeviceptr pk = ring_base ? (ring_base + m->dense_off_wk) : ly->wk.dptr;
    CUdeviceptr pv = ring_base ? (ring_base + m->dense_off_wv) : ly->wv.dptr;
    launch_gemm_batch_raw(st, pq, ly->wq.ggml_type, m->d_pxn, m->d_pq, qdim, dm, B, 0);
    launch_gemm_batch_raw(st, pk, ly->wk.ggml_type, m->d_pxn, m->d_pk, kvdim, dm, B, 0);
    launch_gemm_batch_raw(st, pv, ly->wv.ggml_type, m->d_pxn, m->d_pv, kvdim, dm, B, 0);
    if (m->active_lora > 0) {
        launch_lora_delta(m, st, L, 0, m->d_pxn, m->d_pq, qdim, dm, B);
        launch_lora_delta(m, st, L, 1, m->d_pxn, m->d_pk, kvdim, dm, B);
        launch_lora_delta(m, st, L, 2, m->d_pxn, m->d_pv, kvdim, dm, B);
    }
    if (ly->has_qkv_bias) {
        launch_add_bias_batch(st, m->d_pq, &ly->bq, qdim, B);
        launch_add_bias_batch(st, m->d_pk, &ly->bk, kvdim, B);
        launch_add_bias_batch(st, m->d_pv, &ly->bv, kvdim, B);
    }
    if (ly->has_qk_norm) {
        if (ly->qk_norm_per_head) {
            launch_rmsnorm(st, m->d_pq, &ly->q_norm, m->d_pq, (uint32_t)m->d_head, B * (uint32_t)m->n_heads, m->rms_eps, 1);
            launch_rmsnorm(st, m->d_pk, &ly->k_norm, m->d_pk, (uint32_t)m->d_head, B * (uint32_t)m->n_kv, m->rms_eps, 1);
        } else {
            launch_rmsnorm(st, m->d_pq, &ly->q_norm, m->d_pq, qdim, B, m->rms_eps, 1);
            launch_rmsnorm(st, m->d_pk, &ly->k_norm, m->d_pk, kvdim, B, m->rms_eps, 1);
        }
    }
    uint32_t nh = (uint32_t)m->n_heads, nkv = (uint32_t)m->n_kv, dh = (uint32_t)m->d_head, rd = (uint32_t)m->rope_dim;
    uint32_t rn = (uint32_t)m->rope_neox, hf = (uint32_t)m->has_rope_freqs;
    float theta = m->rope_theta, sc = 1.0f / sqrtf((float)dh);
    CUdeviceptr rf = hf ? m->rope_freqs.dptr : m->d_argmax;
    if (m->kv_quant_q8) {
        void* ra[] = { &m->d_pq, &m->d_pk, &m->d_pv, &ly->k_cache, &ly->v_cache, &rf,
                       &m->d_tree_depth, &nh, &nkv, &dh, &rd, &start_pos, &theta, &rn, &hf, &B };
        g_cu.cuLaunchKernel(g_cu.k_rope_kv_q8_tree, nh, B, 1, 128, 1, 1, 0, st, ra, NULL);
        void* aa[] = { &m->d_pq, &ly->k_cache, &ly->v_cache, &m->d_pctx,
                       &m->d_tree_mask, &nh, &nkv, &dh, &start_pos, &B, &sc };
        g_cu.cuLaunchKernel(g_cu.k_flash_tree_q8, nh, B, 1, 128, 1, 1, 0, st, aa, NULL);
    } else {
        uint32_t hd2 = dh >> 1;
        void* ra[] = { &m->d_pq, &m->d_pk, &m->d_pv, &ly->k_cache, &ly->v_cache, &rf,
                       &m->d_tree_depth, &nh, &nkv, &dh, &rd, &start_pos, &theta, &rn, &hf, &B };
        g_cu.cuLaunchKernel(g_cu.k_rope_kv_tree, (hd2 + 31u) / 32u, nh, B, 32, 1, 1, 0, st, ra, NULL);
        void* aa[] = { &m->d_pq, &ly->k_cache, &ly->v_cache, &m->d_pctx,
                       &m->d_tree_mask, &nh, &nkv, &dh, &start_pos, &B, &sc };
        g_cu.cuLaunchKernel(g_cu.k_flash_tree, nh, B, 1, 128, 1, 1, 0, st, aa, NULL);
    }
    CUdeviceptr po = ring_base ? (ring_base + m->dense_off_wo) : ly->wo.dptr;
    launch_gemm_batch_raw(st, po, ly->wo.ggml_type, m->d_pctx, m->d_px, dm, wo_in, B, 1);
    if (m->active_lora > 0) launch_lora_delta(m, st, L, 3, m->d_pctx, m->d_px, dm, wo_in, B);
    launch_rmsnorm(st, m->d_px, &ly->ffn_norm, m->d_pxn, dm, B, m->rms_eps, 1);
}

int64_t tn_cumoe_verify_tree(int64_t handle, int64_t tok_ids_ptr, int64_t parents_ptr, int64_t n_nodes, int64_t start_pos, int64_t out_toks_ptr) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active || !tok_ids_ptr || !parents_ptr || !out_toks_ptr || n_nodes <= 0) return 0;
    CmModel* m = &g_cm[handle];
    if (n_nodes > CM_TREE_MAX_NODES) n_nodes = CM_TREE_MAX_NODES;
    const int64_t* ids = (const int64_t*)(uintptr_t)tok_ids_ptr;
    const int64_t* par = (const int64_t*)(uintptr_t)parents_ptr;
    int64_t* out = (int64_t*)(uintptr_t)out_toks_ptr;
    if (n_nodes == 1 || !m->dense_model || m->tp_size > 1 || (m->n_layers > 0 && m->layers[0].is_mla)) {
        out[0] = tn_cumoe_step(handle, ids[0], start_pos);
        return 1;
    }
    double t0 = cm_now_ms();
    uint32_t B = (uint32_t)n_nodes;
    cm_ensure_prefill_bufs(m, (int)B);
    uint32_t depths[CM_TREE_MAX_NODES];
    uint64_t masks[CM_TREE_MAX_NODES];
    for (uint32_t i = 0; i < B; i++) {
        int64_t p = par[i];
        if (p < 0 || p >= (int64_t)i) { depths[i] = 0u; masks[i] = (1ull << i); }
        else { depths[i] = depths[p] + 1u; masks[i] = masks[p] | (1ull << i); }
    }
    g_cu.cuMemcpyHtoDAsync(m->d_tree_depth, depths, (size_t)B * 4u, m->compute_stream);
    g_cu.cuMemcpyHtoDAsync(m->d_tree_mask,  masks,  (size_t)B * 8u, m->compute_stream);

    uint32_t dm = (uint32_t)m->d_model, gt = m->tok_embd.ggml_type;
    for (uint32_t bi = 0; bi < B; bi++) {
        CUdeviceptr dst = m->d_px + (uint64_t)bi * (uint64_t)dm * 4ull;
        uint32_t tid = (uint32_t)ids[bi];
        if (m->resident_mode && m->tok_embd.dptr) {
            CUdeviceptr ep = m->tok_embd.dptr;
            void* args[] = { &ep, &dst, &tid, &dm, &gt };
            g_cu.cuLaunchKernel(g_cu.k_embed, (dm + 255u) / 256u, 1, 1, 256, 1, 1, 0, m->compute_stream, args, NULL);
        } else {
            uint64_t rb = cm_ggml_type_bytes(gt, (uint64_t)dm);
            cm_read_slice(m, m->tok_embd.shard_idx, m->h_emb_row, (size_t)rb, m->tok_embd.file_off + (uint64_t)tid * rb);
            g_cu.cuMemcpyHtoD(m->d_emb_row, m->h_emb_row, (size_t)rb);
            uint32_t z = 0; void* args[] = { &m->d_emb_row, &dst, &z, &dm, &gt };
            g_cu.cuLaunchKernel(g_cu.k_embed, (dm + 255u) / 256u, 1, 1, 256, 1, 1, 0, m->compute_stream, args, NULL);
            g_cu.cuStreamSynchronize(m->compute_stream);
        }
    }
    if (m->resident_mode) {
        for (int L = 0; L < m->n_layers; L++) {
            CmLayer* ly = &m->layers[L];
            launch_layer_attn_tree(m, m->compute_stream, L, (uint32_t)start_pos, B, 0);
            launch_dense_ffn_batch(m, m->compute_stream, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                                   ly->ffn_up.dptr, ly->ffn_up.ggml_type, ly->ffn_down.dptr, ly->ffn_down.ggml_type,
                                   (uint32_t)m->d_ff_dense, B);
        }
    } else {
        int first_stream = m->cache_slots_per_layer;
        for (int L = 0; L < first_stream && L < m->n_layers; L++) {
            CmLayer* ly = &m->layers[L];
            launch_layer_attn_tree(m, m->compute_stream, L, (uint32_t)start_pos, B, 0);
            launch_dense_ffn_batch(m, m->compute_stream, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                                   ly->ffn_up.dptr, ly->ffn_up.ggml_type, ly->ffn_down.dptr, ly->ffn_down.ggml_type,
                                   (uint32_t)m->d_ff_dense, B);
        }
        if (m->enable_overlap && !m->use_nocache && m->cpu_pool_init && m->dense_pin_all_attn) {
            CmWeightRef pacc_ref = {0}; pacc_ref.dptr = m->d_pxn;
            for (int L = first_stream; L < m->n_layers; L++) {
                launch_layer_attn_tree(m, m->compute_stream, L, (uint32_t)start_pos, B, 0);
                g_cu.cuMemcpyDtoHAsync(m->h_pxn, m->d_pxn, (size_t)B * dm * 4u, m->compute_stream);
                g_cu.cuStreamSynchronize(m->compute_stream);
                for (uint32_t bi = 0; bi < B; bi++) {
                    memcpy(m->h_xn, m->h_pxn + (size_t)bi * dm, (size_t)dm * 4u);
                    cm_cpu_pool_run(m, 2, L, 0, NULL, NULL); cm_cpu_pool_wait(m);
                    memcpy(m->h_pcpu_accum + (size_t)bi * dm, m->h_cpu_accum, (size_t)dm * 4u);
                }
                g_cu.cuMemcpyHtoDAsync(m->d_pxn, m->h_pcpu_accum, (size_t)B * dm * 4u, m->compute_stream);
                launch_add_bias(m->compute_stream, m->d_px, &pacc_ref, B * dm);
            }
        } else {
            int n_ring = m->dense_pin_all_attn ? 4 : 2;
            for (int L = first_stream; L < m->n_layers; L++) {
                CmLayer* ly = &m->layers[L];
                int cur = (L - first_stream) % n_ring;
                cm_stage_dense_layer(m, L, m->dense_ring_host[cur]);
                g_cu.cuMemcpyHtoD(m->dense_ring_dptr[cur], m->dense_ring_host[cur], (size_t)m->dense_ring_stride);
                launch_layer_attn_tree(m, m->compute_stream, L, (uint32_t)start_pos, B, m->dense_pin_all_attn ? 0 : m->dense_ring_dptr[cur]);
                launch_dense_ffn_batch(m, m->compute_stream,
                                       m->dense_ring_dptr[cur] + m->dense_off_gate, ly->ffn_gate.ggml_type,
                                       m->dense_ring_dptr[cur] + m->dense_off_up,   ly->ffn_up.ggml_type,
                                       m->dense_ring_dptr[cur] + m->dense_off_down, ly->ffn_down.ggml_type,
                                       (uint32_t)m->d_ff_dense, B);
                g_cu.cuStreamSynchronize(m->compute_stream);
            }
        }
    }
    uint32_t v = (uint32_t)m->vocab_size;
    if (m->batch_logits_cap < (int)B) {
        int ncap = (int)(B < 16 ? 16 : B);
        if (m->d_batch_logits) g_cu.cuMemFree(m->d_batch_logits);
        g_cu.cuMemAlloc(&m->d_batch_logits, (size_t)ncap * (size_t)v * 4u);
        m->batch_logits_cap = ncap;
    }
    if (!m->d_argmax_batch) {
        g_cu.cuMemAlloc(&m->d_argmax_batch, (size_t)CM_PREFILL_MAX_B * 4u);
        g_cu.cuMemHostAlloc((void**)&m->h_argmax_batch, (size_t)CM_PREFILL_MAX_B * 4u, CU_MEMHOSTALLOC_PORTABLE);
    }
    launch_rmsnorm(m->compute_stream, m->d_px, &m->out_norm, m->d_pxn, dm, B, m->rms_eps, 1);
    launch_gemm_batch_raw(m->compute_stream, m->lm_head.dptr, m->lm_head.ggml_type, m->d_pxn, m->d_batch_logits, v, dm, B, 0);
    void* aa[] = { &m->d_batch_logits, &m->d_argmax_batch, &v, &B };
    g_cu.cuLaunchKernel(g_cu.k_argmax_b, B, 1, 1, 256, 1, 1, 0, m->compute_stream, aa, NULL);
    g_cu.cuMemcpyDtoHAsync(m->h_argmax_batch, m->d_argmax_batch, (size_t)B * 4u, m->compute_stream);
    g_cu.cuMemcpyDtoDAsync(m->d_logits, m->d_batch_logits + (uint64_t)(B - 1u) * (uint64_t)v * 4ull, (size_t)v * 4u, m->compute_stream);
    g_cu.cuStreamSynchronize(m->compute_stream);
    for (uint32_t bi = 0; bi < B; bi++) out[bi] = (int64_t)m->h_argmax_batch[bi];
    m->last_total_ms = cm_now_ms() - t0;
    return (int64_t)B;
}

int64_t tn_cumoe_commit_tree_path(int64_t handle, int64_t path_nodes_ptr, int64_t path_len, int64_t start_pos) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active || !path_nodes_ptr || path_len <= 0) return 0;
    CmModel* m = &g_cm[handle];
    const int64_t* nodes = (const int64_t*)(uintptr_t)path_nodes_ptr;
    size_t row_b = m->kv_quant_q8
        ? ((size_t)m->n_kv * (size_t)(m->d_head >> 5) * 34u)
        : ((size_t)(m->n_kv * m->d_head) * 2u);
    CUdeviceptr d_tmp = 0;
    if (g_cu.cuMemAlloc(&d_tmp, (size_t)path_len * row_b) != CUDA_SUCCESS) return 0;
    for (int L = 0; L < m->n_layers; L++) {
        CmLayer* ly = &m->layers[L];
        if (ly->k_cache) {
            for (int64_t d = 0; d < path_len; d++) {
                g_cu.cuMemcpyDtoDAsync(d_tmp + (uint64_t)d * row_b, ly->k_cache + (uint64_t)(start_pos + nodes[d]) * row_b, row_b, m->compute_stream);
            }
            g_cu.cuMemcpyDtoDAsync(ly->k_cache + (uint64_t)start_pos * row_b, d_tmp, (size_t)path_len * row_b, m->compute_stream);
        }
        if (ly->v_cache) {
            for (int64_t d = 0; d < path_len; d++) {
                g_cu.cuMemcpyDtoDAsync(d_tmp + (uint64_t)d * row_b, ly->v_cache + (uint64_t)(start_pos + nodes[d]) * row_b, row_b, m->compute_stream);
            }
            g_cu.cuMemcpyDtoDAsync(ly->v_cache + (uint64_t)start_pos * row_b, d_tmp, (size_t)path_len * row_b, m->compute_stream);
        }
    }
    g_cu.cuStreamSynchronize(m->compute_stream);
    g_cu.cuMemFree(d_tmp);
    return path_len;
}

// =========================================================================
// Tenon v6: Native MTP (`nextn_layers`) Full-Block Forward Predictor (`tn_cumoe_mtp_predict`)
// =========================================================================
int64_t tn_cumoe_mtp_predict(int64_t handle, int64_t tok_id, int64_t pos) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return -1;
    CmModel* m = &g_cm[handle];
    if (!m->has_mtp || !m->mtp_enorm.dptr || !m->mtp_hnorm.dptr || !m->mtp_eh_proj.dptr || !m->d_last_hidden || !m->d_mtp_cat) return -1;
    uint32_t dm = (uint32_t)m->d_model;
    launch_embed(m, m->compute_stream, (uint32_t)tok_id);
    launch_rmsnorm(m->compute_stream, m->d_x, &m->mtp_enorm, m->d_mtp_cat, dm, 1, m->rms_eps, 0);
    launch_rmsnorm(m->compute_stream, m->d_last_hidden, &m->mtp_hnorm, m->d_mtp_cat + (uint64_t)dm * 4ull, dm, 1, m->rms_eps, 0);
    launch_gemv(m->compute_stream, &m->mtp_eh_proj, m->d_mtp_cat, m->d_x, dm, dm * 2u, 0);
    if (m->has_mtp_layer && m->mtp_layer && m->mtp_layer->attn_norm.dptr) {
        CmLayer* ml = m->mtp_layer;
        launch_layer_attn_ptr(m, m->compute_stream, ml, (uint32_t)pos, 0, 0);
        if (ml->cache_dptr && ml->router.dptr && m->n_experts > 0) {
            uint32_t ne = (uint32_t)m->n_experts, tk = (uint32_t)m->top_k, df = (uint32_t)m->d_ff_exp;
            uint32_t nw = (uint32_t)m->expert_weights_norm, gf = (uint32_t)m->gating_func;
            uint32_t ng = (uint32_t)m->n_expert_groups, ngu = (uint32_t)m->n_group_used, hb = (uint32_t)ml->has_exp_bias;
            float ws = m->expert_weights_scale;
            launch_gemv(m->compute_stream, &ml->router, m->d_xn, m->d_rlogits, ne, dm, 0);
            CUdeviceptr bp = hb ? ml->exp_bias.dptr : m->d_argmax;
            void* ra[] = { &m->d_rlogits, &bp, &m->d_eidx, &m->d_ewt, &ne, &tk, &nw, &gf, &hb, &ws, &ng, &ngu };
            g_cu.cuLaunchKernel(g_cu.k_router, 1, 1, 1, 32, 1, 1, 0, m->compute_stream, ra, NULL);
            if (ml->has_shared_exp && ml->sh_gate.dptr) {
                uint32_t sf = (ml->sh_gate.ne1 > 0) ? (uint32_t)ml->sh_gate.ne1 : df;
                launch_dense_ffn_raw(m, m->compute_stream, ml->sh_gate.dptr, ml->sh_gate.ggml_type,
                                     ml->sh_up.dptr, ml->sh_up.ggml_type, ml->sh_down.dptr, ml->sh_down.ggml_type, sf);
            }
            uint64_t gb = ml->gate_exp_bytes, sb = ml->slot_stride_bytes, doff = ml->gate_exp_bytes + ml->up_exp_bytes;
            int is_q4k_gu = (ml->gate_exps.ggml_type == CM_GGML_Q4_K);
            int is_q4k_dn = (ml->down_exps.ggml_type == CM_GGML_Q4_K);
            int is_q6k_dn = (ml->down_exps.ggml_type == CM_GGML_Q6_K);
            void* gu_a[] = { &ml->cache_dptr, &m->d_xn, &m->d_eidx, &m->d_h, &dm, &df, &tk, &gb, &sb };
            g_cu.cuLaunchKernel(is_q4k_gu ? g_cu.k_moe_slot_gu_q4k : g_cu.k_moe_slot_gu_q4,
                                is_q4k_gu ? ((df + 3u) / 4u) : ((df + 7u) / 8u), tk, 1, 128, 1, 1, 0, m->compute_stream, gu_a, NULL);
            CUfunction dfn = is_q6k_dn ? g_cu.k_moe_slot_down_q6k : (is_q4k_dn ? g_cu.k_moe_slot_down_q4k : ((ml->down_exps.ggml_type == CM_GGML_Q4_1) ? g_cu.k_moe_slot_down_q4_1 : g_cu.k_moe_slot_down_q4));
            void* dn_a[] = { &ml->cache_dptr, &m->d_h, &m->d_eidx, &m->d_ewt, &m->d_x, &dm, &df, &tk, &doff, &sb };
            g_cu.cuLaunchKernel(dfn, (is_q4k_dn || is_q6k_dn) ? ((dm + 3u) / 4u) : ((dm + 15u) / 16u), 1, 1, 128, 1, 1, 0, m->compute_stream, dn_a, NULL);
        } else if (ml->ffn_gate.dptr && ml->ffn_up.dptr && ml->ffn_down.dptr) {
            uint32_t df = (ml->ffn_gate.ne1 > 0) ? (uint32_t)ml->ffn_gate.ne1 : (uint32_t)m->d_ff_dense;
            launch_dense_ffn_raw(m, m->compute_stream, ml->ffn_gate.dptr, ml->ffn_gate.ggml_type,
                                 ml->ffn_up.dptr, ml->ffn_up.ggml_type, ml->ffn_down.dptr, ml->ffn_down.ggml_type, df);
        } else if (ml->has_shared_exp && ml->sh_gate.dptr) {
            uint32_t sf = (ml->sh_gate.ne1 > 0) ? (uint32_t)ml->sh_gate.ne1 : (uint32_t)m->d_ff_exp;
            launch_dense_ffn_raw(m, m->compute_stream, ml->sh_gate.dptr, ml->sh_gate.ggml_type,
                                 ml->sh_up.dptr, ml->sh_up.ggml_type, ml->sh_down.dptr, ml->sh_down.ggml_type, sf);
        }
    }
    launch_final_lm_head(m, m->compute_stream);
    g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, m->compute_stream);
    g_cu.cuStreamSynchronize(m->compute_stream);
    return (int64_t)m->h_argmax[0];
}

// =========================================================================
// Tenon v6: Text Embeddings (`tn_cumoe_embed_text`) & Vision/Soft-Prompt Injection (`tn_cumoe_step_embd`)
// =========================================================================
int64_t tn_cumoe_embed_text(int64_t handle, int64_t tok_ids_ptr, int64_t n_toks, int64_t out_f32_ptr, int64_t max_dim) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active || !tok_ids_ptr || !out_f32_ptr || n_toks <= 0) return 0;
    CmModel* m = &g_cm[handle];
    int dm = m->d_model;
    if (max_dim < dm) return 0;
    const int64_t* ids = (const int64_t*)(uintptr_t)tok_ids_ptr;
    float* out = (float*)(uintptr_t)out_f32_ptr;
    float* tmp = (float*)malloc((size_t)dm * sizeof(float));
    memset(out, 0, (size_t)dm * sizeof(float));
    for (int64_t i = 0; i < n_toks; i++) {
        tn_cumoe_step(handle, ids[i], i);
        CUdeviceptr src_x = (m->tp_size > 1 && m->tp) ? m->tp[0].d_x : m->d_x;
        g_cu.cuMemcpyDtoH(tmp, src_x, (size_t)dm * sizeof(float));
        for (int d = 0; d < dm; d++) out[d] += tmp[d];
    }
    float inv_n = 1.0f / (float)n_toks;
    for (int d = 0; d < dm; d++) tmp[d] = out[d] * inv_n;
    g_cu.cuMemcpyHtoD(m->d_x, tmp, (size_t)dm * sizeof(float));
    launch_rmsnorm(m->compute_stream, m->d_x, &m->out_norm, m->d_xn, (uint32_t)dm, 1, m->rms_eps, 0);
    g_cu.cuMemcpyDtoHAsync(tmp, m->d_xn, (size_t)dm * sizeof(float), m->compute_stream);
    g_cu.cuStreamSynchronize(m->compute_stream);
    double sq = 0.0;
    for (int d = 0; d < dm; d++) sq += (double)tmp[d] * (double)tmp[d];
    float inv_l2 = (sq > 1e-12) ? (float)(1.0 / sqrt(sq)) : 1.0f;
    for (int d = 0; d < dm; d++) out[d] = tmp[d] * inv_l2;
    free(tmp);
    return (int64_t)dm;
}

int64_t tn_cumoe_step_embd(int64_t handle, int64_t embd_f32_ptr, int64_t n_vecs, int64_t start_pos) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active || !embd_f32_ptr || n_vecs <= 0) return 0;
    CmModel* m = &g_cm[handle];
    const float* vecs = (const float*)(uintptr_t)embd_f32_ptr;
    double t0 = cm_now_ms();
    int64_t last_tok = 0;
    uint32_t dm = (uint32_t)m->d_model, v = (uint32_t)m->vocab_size;
    if (n_vecs > 1 && m->has_mrope != 2 && m->dense_model && m->resident_mode && m->tp_size == 1 && !(m->n_layers > 0 && m->layers[0].is_mla)) {
        int chunk_sz = 64;
        cm_ensure_prefill_bufs(m, chunk_sz);
        int64_t off = 0;
        while (off < n_vecs) {
            uint32_t B = (uint32_t)((n_vecs - off > chunk_sz) ? chunk_sz : (n_vecs - off));
            g_cu.cuMemcpyHtoDAsync(m->d_px, vecs + (size_t)off * dm, (size_t)B * dm * 4u, m->compute_stream);
            for (int L = 0; L < m->n_layers; L++) {
                CmLayer* ly = &m->layers[L];
                launch_layer_attn_batch(m, m->compute_stream, L, (uint32_t)(start_pos + off), B, 0);
                launch_dense_ffn_batch(m, m->compute_stream, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                                       ly->ffn_up.dptr, ly->ffn_up.ggml_type, ly->ffn_down.dptr, ly->ffn_down.ggml_type,
                                       (uint32_t)m->d_ff_dense, B);
            }
            if (off + B == n_vecs) {
                CUdeviceptr lx = m->d_px + (uint64_t)(B - 1u) * dm * 4ull;
                if (m->d_last_hidden) g_cu.cuMemcpyDtoDAsync(m->d_last_hidden, lx, (size_t)dm * 4u, m->compute_stream);
                launch_rmsnorm(m->compute_stream, lx, &m->out_norm, m->d_xn, dm, 1, m->rms_eps, 0);
                launch_gemv(m->compute_stream, &m->lm_head, m->d_xn, m->d_logits, v, dm, 0);
                void* aa[] = { &m->d_logits, &m->d_argmax, &v };
                g_cu.cuLaunchKernel(g_cu.k_argmax, 1, 1, 1, 256, 1, 1, 0, m->compute_stream, aa, NULL);
                g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, m->compute_stream);
                g_cu.cuStreamSynchronize(m->compute_stream);
                last_tok = (int64_t)m->h_argmax[0];
            } else {
                g_cu.cuStreamSynchronize(m->compute_stream);
            }
            off += B;
        }
    } else {
        for (int64_t i = 0; i < n_vecs; i++) {
            g_cu.cuMemcpyHtoDAsync(m->d_x, vecs + (size_t)i * dm, (size_t)dm * 4u, m->compute_stream);
            for (int L = 0; L < m->n_layers; L++) {
                CmLayer* ly = &m->layers[L];
                launch_layer_attn(m, m->compute_stream, L, (uint32_t)(start_pos + i), 0);
                if (ly->is_dense) {
                    launch_dense_ffn_raw(m, m->compute_stream, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                                         ly->ffn_up.dptr, ly->ffn_up.ggml_type, ly->ffn_down.dptr, ly->ffn_down.ggml_type,
                                         (uint32_t)m->d_ff_dense);
                }
            }
            if (m->d_last_hidden) g_cu.cuMemcpyDtoDAsync(m->d_last_hidden, m->d_x, (size_t)dm * 4u, m->compute_stream);
            launch_final_lm_head(m, m->compute_stream);
            g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, m->compute_stream);
            g_cu.cuStreamSynchronize(m->compute_stream);
            last_tok = (int64_t)m->h_argmax[0];
        }
    }
    m->last_total_ms = cm_now_ms() - t0;
    return last_tok;
}

// =========================================================================
// Tenon v6: Native Multimodal Vision Encoder (`mmproj` CLIP/SigLIP/SmolVLM GGUF on CUDA GPU!)
// =========================================================================
#define CM_MMPROJ_MAX_MODELS 4
#define CM_MMPROJ_MAX_LAYERS 48

typedef struct {
    CmWeightRef ln1_w, ln1_b, ln2_w, ln2_b;
    CmWeightRef q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b;
    CmWeightRef fc1_w, fc1_b, fc2_w, fc2_b;
} CmMmprojLayer;

typedef struct {
    int active;
    CUstream stream;
    CUdeviceptr d_weights;
    int image_size, patch_size, d_vis, d_ff, n_layers, n_heads, proj_dim, scale_factor;
    int use_gelu, has_cls, is_idefics3;
    float eps, mean[3], std[3];
    CmWeightRef patch_w, patch_b, class_embd, pos_embd, pre_ln_w, pre_ln_b, post_ln_w, post_ln_b;
    CmWeightRef mm_fc_w, mm_fc_b, mm_0_w, mm_0_b, mm_2_w, mm_2_b;
    CmMmprojLayer layers[CM_MMPROJ_MAX_LAYERS];
    CUdeviceptr d_in, d_x, d_xn, d_q, d_k, d_v, d_ctx, d_ff_buf, d_ps, d_out, d_zero;
} CmMmprojModel;

static CmMmprojModel g_cm_mmproj[CM_MMPROJ_MAX_MODELS];

static void cm_mm_bind(CmMmprojModel* mm, CmWeightRef* r, const CmTensorInfo* ti) {
    cm_record_meta(r, ti);
    r->dptr = mm->d_weights + ti->rel_off;
}

static void launch_vit_ln(CUstream st, CUdeviceptr x, const CmWeightRef* w, const CmWeightRef* b,
                          CUdeviceptr out, uint32_t dim, uint32_t n_tok, float eps, CUdeviceptr zero) {
    CUdeviceptr wp = w->dptr, bp = b->dptr ? b->dptr : zero;
    uint32_t hb = b->dptr ? 1u : 0u;
    void* a[] = { &x, &wp, &bp, &out, &dim, &n_tok, &eps, &hb };
    g_cu.cuLaunchKernel(g_cu.k_vit_ln, n_tok, 1, 1, 256, 1, 1, 0, st, a, NULL);
}

int64_t tn_cumoe_mmproj_load(int64_t path_cstr) {
    if (!tn_cumoe_init() || !path_cstr) return 0;
    int slot = -1;
    for (int i = 1; i < CM_MMPROJ_MAX_MODELS; i++) if (!g_cm_mmproj[i].active) { slot = i; break; }
    if (slot < 0) return 0;
    CmMmprojModel* mm = &g_cm_mmproj[slot];
    memset(mm, 0, sizeof(CmMmprojModel));
    mm->image_size = 512; mm->patch_size = 16; mm->scale_factor = 1; mm->eps = 1e-6f;
    mm->mean[0] = mm->mean[1] = mm->mean[2] = 0.5f;
    mm->std[0]  = mm->std[1]  = mm->std[2]  = 0.5f;

    const char* path = (const char*)(uintptr_t)path_cstr;
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    CmGr gr = { fp };
    char magic[4];
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "GGUF", 4) != 0) { fclose(fp); return 0; }
    (void)gr_u32(&gr);
    uint64_t tcnt = gr_u64(&gr), kvcnt = gr_u64(&gr), align = 32;
    for (uint64_t i = 0; i < kvcnt; i++) {
        char* k = gr_str(&gr); uint32_t vt = gr_u32(&gr);
        if (strcmp(k, "general.alignment") == 0) align = gr_int(&gr, vt);
        else if (strcmp(k, "clip.vision.image_size") == 0) mm->image_size = (int)gr_int(&gr, vt);
        else if (strcmp(k, "clip.vision.patch_size") == 0) mm->patch_size = (int)gr_int(&gr, vt);
        else if (strcmp(k, "clip.vision.embedding_length") == 0) mm->d_vis = (int)gr_int(&gr, vt);
        else if (strcmp(k, "clip.vision.feed_forward_length") == 0) mm->d_ff = (int)gr_int(&gr, vt);
        else if (strcmp(k, "clip.vision.block_count") == 0) mm->n_layers = (int)gr_int(&gr, vt);
        else if (strcmp(k, "clip.vision.attention.head_count") == 0) mm->n_heads = (int)gr_int(&gr, vt);
        else if (strcmp(k, "clip.vision.projection_dim") == 0) mm->proj_dim = (int)gr_int(&gr, vt);
        else if (strcmp(k, "clip.vision.projector.scale_factor") == 0) mm->scale_factor = (int)gr_int(&gr, vt);
        else if (strcmp(k, "clip.use_gelu") == 0) mm->use_gelu = (int)gr_int(&gr, vt);
        else if (strcmp(k, "clip.vision.attention.layer_norm_epsilon") == 0 && vt == CM_GGUF_F32) mm->eps = gr_f32(&gr);
        else if (strcmp(k, "clip.projector_type") == 0 && vt == CM_GGUF_STR) {
            char* pt = gr_str(&gr); if (strstr(pt, "idefics3") != NULL) mm->is_idefics3 = 1; free(pt);
        } else if ((strcmp(k, "clip.vision.image_mean") == 0 || strcmp(k, "clip.vision.image_std") == 0) && vt == CM_GGUF_ARR) {
            uint32_t et = gr_u32(&gr); uint64_t n = gr_u64(&gr);
            float* dst = (k[18] == 'm') ? mm->mean : mm->std;
            for (uint64_t j = 0; j < n; j++) {
                if (et == CM_GGUF_F32) { float v = gr_f32(&gr); if (j < 3) dst[j] = v; }
                else gr_skip(&gr, et);
            }
        } else gr_skip(&gr, vt);
        free(k);
    }
    CmTensorInfo* tlist = (CmTensorInfo*)calloc((size_t)tcnt, sizeof(CmTensorInfo));
    for (uint64_t i = 0; i < tcnt; i++) {
        char* tn = gr_str(&gr); strncpy(tlist[i].name, tn, sizeof(tlist[i].name) - 1); free(tn);
        tlist[i].n_dims = gr_u32(&gr);
        uint64_t numel = 1;
        for (uint32_t d = 0; d < tlist[i].n_dims; d++) { uint64_t dm = gr_u64(&gr); if (d < 4) tlist[i].ne[d] = dm; numel *= dm; }
        tlist[i].ggml_type = gr_u32(&gr); tlist[i].rel_off = gr_u64(&gr);
        tlist[i].byte_size = cm_ggml_type_bytes(tlist[i].ggml_type, numel);
    }
    uint64_t hdr_end = (uint64_t)ftello(fp);
    uint64_t data_off = (hdr_end + align - 1) & ~(align - 1);
    fseeko(fp, 0, SEEK_END);
    uint64_t fsz = (uint64_t)ftello(fp);
    if (fsz <= data_off || mm->d_vis <= 0 || mm->n_layers <= 0 || mm->patch_size <= 0) { free(tlist); fclose(fp); return 0; }
    size_t data_bytes = (size_t)(fsz - data_off);
    uint8_t* raw = (uint8_t*)malloc(data_bytes);
    fseeko(fp, (off_t)data_off, SEEK_SET);
    if (!raw || fread(raw, 1, data_bytes, fp) != data_bytes) { free(raw); free(tlist); fclose(fp); return 0; }
    fclose(fp);

    g_cu.cuCtxSetCurrent(g_cu.ctx);
    if (g_cu.cuMemAlloc(&mm->d_weights, data_bytes) != CUDA_SUCCESS) { free(raw); free(tlist); return 0; }
    g_cu.cuMemcpyHtoD(mm->d_weights, raw, data_bytes);
    free(raw);
    g_cu.cuStreamCreate(&mm->stream, CU_STREAM_NON_BLOCKING);
    if (mm->n_layers > CM_MMPROJ_MAX_LAYERS) mm->n_layers = CM_MMPROJ_MAX_LAYERS;

    for (uint64_t i = 0; i < tcnt; i++) {
        const CmTensorInfo* ti = &tlist[i];
        if (strcmp(ti->name, "v.patch_embd.weight") == 0) cm_mm_bind(mm, &mm->patch_w, ti);
        else if (strcmp(ti->name, "v.patch_embd.bias") == 0) cm_mm_bind(mm, &mm->patch_b, ti);
        else if (strcmp(ti->name, "v.class_embd") == 0) { cm_mm_bind(mm, &mm->class_embd, ti); mm->has_cls = 1; }
        else if (strcmp(ti->name, "v.position_embd.weight") == 0) cm_mm_bind(mm, &mm->pos_embd, ti);
        else if (strcmp(ti->name, "v.pre_ln.weight") == 0) cm_mm_bind(mm, &mm->pre_ln_w, ti);
        else if (strcmp(ti->name, "v.pre_ln.bias") == 0) cm_mm_bind(mm, &mm->pre_ln_b, ti);
        else if (strcmp(ti->name, "v.post_ln.weight") == 0) cm_mm_bind(mm, &mm->post_ln_w, ti);
        else if (strcmp(ti->name, "v.post_ln.bias") == 0) cm_mm_bind(mm, &mm->post_ln_b, ti);
        else if (strcmp(ti->name, "mm.model.fc.weight") == 0) { cm_mm_bind(mm, &mm->mm_fc_w, ti); mm->is_idefics3 = 1; }
        else if (strcmp(ti->name, "mm.model.fc.bias") == 0) cm_mm_bind(mm, &mm->mm_fc_b, ti);
        else if (strcmp(ti->name, "mm.0.weight") == 0) cm_mm_bind(mm, &mm->mm_0_w, ti);
        else if (strcmp(ti->name, "mm.0.bias") == 0) cm_mm_bind(mm, &mm->mm_0_b, ti);
        else if (strcmp(ti->name, "mm.2.weight") == 0) cm_mm_bind(mm, &mm->mm_2_w, ti);
        else if (strcmp(ti->name, "mm.2.bias") == 0) cm_mm_bind(mm, &mm->mm_2_b, ti);
        else if (strncmp(ti->name, "v.blk.", 6) == 0) {
            int L = atoi(ti->name + 6);
            const char* dot = strchr(ti->name + 6, '.');
            if (!dot || L < 0 || L >= mm->n_layers) continue;
            const char* sub = dot + 1;
            CmMmprojLayer* ly = &mm->layers[L];
            if (strcmp(sub, "ln1.weight") == 0) cm_mm_bind(mm, &ly->ln1_w, ti);
            else if (strcmp(sub, "ln1.bias") == 0) cm_mm_bind(mm, &ly->ln1_b, ti);
            else if (strcmp(sub, "ln2.weight") == 0) cm_mm_bind(mm, &ly->ln2_w, ti);
            else if (strcmp(sub, "ln2.bias") == 0) cm_mm_bind(mm, &ly->ln2_b, ti);
            else if (strcmp(sub, "attn_q.weight") == 0) cm_mm_bind(mm, &ly->q_w, ti);
            else if (strcmp(sub, "attn_q.bias") == 0) cm_mm_bind(mm, &ly->q_b, ti);
            else if (strcmp(sub, "attn_k.weight") == 0) cm_mm_bind(mm, &ly->k_w, ti);
            else if (strcmp(sub, "attn_k.bias") == 0) cm_mm_bind(mm, &ly->k_b, ti);
            else if (strcmp(sub, "attn_v.weight") == 0) cm_mm_bind(mm, &ly->v_w, ti);
            else if (strcmp(sub, "attn_v.bias") == 0) cm_mm_bind(mm, &ly->v_b, ti);
            else if (strcmp(sub, "attn_out.weight") == 0) cm_mm_bind(mm, &ly->o_w, ti);
            else if (strcmp(sub, "attn_out.bias") == 0) cm_mm_bind(mm, &ly->o_b, ti);
            else if (strcmp(sub, "ffn_down.weight") == 0) {
                if ((int)ti->ne[1] == mm->d_ff) cm_mm_bind(mm, &ly->fc1_w, ti); else cm_mm_bind(mm, &ly->fc2_w, ti);
            } else if (strcmp(sub, "ffn_down.bias") == 0) {
                if ((int)ti->ne[0] == mm->d_ff) cm_mm_bind(mm, &ly->fc1_b, ti); else cm_mm_bind(mm, &ly->fc2_b, ti);
            } else if (strcmp(sub, "ffn_up.weight") == 0) {
                if ((int)ti->ne[1] == mm->d_ff) cm_mm_bind(mm, &ly->fc1_w, ti); else cm_mm_bind(mm, &ly->fc2_w, ti);
            } else if (strcmp(sub, "ffn_up.bias") == 0) {
                if ((int)ti->ne[0] == mm->d_ff) cm_mm_bind(mm, &ly->fc1_b, ti); else cm_mm_bind(mm, &ly->fc2_b, ti);
            }
        }
    }
    free(tlist);
    int grid = mm->image_size / mm->patch_size;
    int n_tok = grid * grid + (mm->has_cls ? 1 : 0);
    int patch_dim = 3 * mm->patch_size * mm->patch_size;
    int n_out = mm->is_idefics3 ? ((grid / mm->scale_factor) * (grid / mm->scale_factor)) : (grid * grid);
    int ps_dim = mm->is_idefics3 ? (mm->scale_factor * mm->scale_factor * mm->d_vis) : mm->d_vis;
    int pdim = (mm->proj_dim > 0) ? mm->proj_dim : mm->d_vis;

    g_cu.cuMemAlloc(&mm->d_in,     (size_t)n_tok * (size_t)patch_dim * 4u);
    g_cu.cuMemAlloc(&mm->d_x,      (size_t)n_tok * (size_t)mm->d_vis * 4u);
    g_cu.cuMemAlloc(&mm->d_xn,     (size_t)n_tok * (size_t)mm->d_vis * 4u);
    g_cu.cuMemAlloc(&mm->d_q,      (size_t)n_tok * (size_t)mm->d_vis * 4u);
    g_cu.cuMemAlloc(&mm->d_k,      (size_t)n_tok * (size_t)mm->d_vis * 4u);
    g_cu.cuMemAlloc(&mm->d_v,      (size_t)n_tok * (size_t)mm->d_vis * 4u);
    g_cu.cuMemAlloc(&mm->d_ctx,    (size_t)n_tok * (size_t)mm->d_vis * 4u);
    g_cu.cuMemAlloc(&mm->d_ff_buf, (size_t)n_tok * (size_t)mm->d_ff * 4u);
    g_cu.cuMemAlloc(&mm->d_ps,     (size_t)n_out * (size_t)ps_dim * 4u);
    g_cu.cuMemAlloc(&mm->d_out,    (size_t)n_out * (size_t)pdim * 4u);
    g_cu.cuMemAlloc(&mm->d_zero,   64u);
    g_cu.cuMemsetD8(mm->d_zero, 0, 64u);
    mm->active = 1;
    return (int64_t)slot;
}

int64_t tn_cumoe_mmproj_info(int64_t mm_h, int64_t field_id) {
    if (mm_h <= 0 || mm_h >= CM_MMPROJ_MAX_MODELS || !g_cm_mmproj[mm_h].active) return 0;
    CmMmprojModel* mm = &g_cm_mmproj[mm_h];
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

int64_t tn_cumoe_mmproj_encode_rgb(int64_t mm_h, int64_t rgb_u8_ptr, int64_t width, int64_t height, int64_t out_embd_ptr, int64_t max_vecs) {
    if (mm_h <= 0 || mm_h >= CM_MMPROJ_MAX_MODELS || !g_cm_mmproj[mm_h].active || !rgb_u8_ptr || !out_embd_ptr || width <= 0 || height <= 0) return 0;
    CmMmprojModel* mm = &g_cm_mmproj[mm_h];
    int sz = mm->image_size, ps = mm->patch_size, grid = sz / ps;
    int n_patches = grid * grid, n_tok = n_patches + (mm->has_cls ? 1 : 0);
    int pdim = 3 * ps * ps;
    int n_out = mm->is_idefics3 ? ((grid / mm->scale_factor) * (grid / mm->scale_factor)) : n_patches;
    if (max_vecs < n_out) return 0;

    const uint8_t* src = (const uint8_t*)(uintptr_t)rgb_u8_ptr;
    float* img = (float*)malloc((size_t)3 * (size_t)sz * (size_t)sz * sizeof(float));
    for (int y = 0; y < sz; y++) {
        float fy = ((float)y + 0.5f) * (float)height / (float)sz - 0.5f;
        int y0 = (int)floorf(fy); float dy = fy - (float)y0; int y1 = y0 + 1;
        if (y0 < 0) y0 = 0;
        if (y0 >= height) y0 = (int)height - 1;
        if (y1 < 0) y1 = 0;
        if (y1 >= height) y1 = (int)height - 1;
        for (int x = 0; x < sz; x++) {
            float fx = ((float)x + 0.5f) * (float)width / (float)sz - 0.5f;
            int x0 = (int)floorf(fx); float dx = fx - (float)x0; int x1 = x0 + 1;
            if (x0 < 0) x0 = 0;
            if (x0 >= width) x0 = (int)width - 1;
            if (x1 < 0) x1 = 0;
            if (x1 >= width) x1 = (int)width - 1;
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
    float* pin = (float*)malloc((size_t)n_patches * (size_t)pdim * sizeof(float));
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

    g_cu.cuCtxSetCurrent(g_cu.ctx);
    CUstream st = mm->stream;
    g_cu.cuMemcpyHtoDAsync(mm->d_in, pin, (size_t)n_patches * (size_t)pdim * sizeof(float), st);
    CUdeviceptr patch_out = mm->has_cls ? mm->d_xn : mm->d_x;
    launch_gemm_batch_raw(st, mm->patch_w.dptr, mm->patch_w.ggml_type, mm->d_in, patch_out, (uint32_t)mm->d_vis, (uint32_t)pdim, (uint32_t)n_patches, 0);
    if (mm->patch_b.dptr) launch_add_bias_batch(st, patch_out, &mm->patch_b, (uint32_t)mm->d_vis, (uint32_t)n_patches);
    if (mm->has_cls && mm->class_embd.dptr) {
        g_cu.cuMemcpyDtoDAsync(mm->d_x, mm->class_embd.dptr, (size_t)mm->d_vis * 4u, st);
        g_cu.cuMemcpyDtoDAsync(mm->d_x + (uint64_t)mm->d_vis * 4ull, mm->d_xn, (size_t)n_patches * (size_t)mm->d_vis * 4u, st);
    }
    if (mm->pos_embd.dptr) {
        uint32_t tot = (uint32_t)(n_tok * mm->d_vis);
        void* pa[] = { &mm->d_x, &mm->pos_embd.dptr, &tot };
        g_cu.cuLaunchKernel(g_cu.k_vit_add, (tot + 255u) / 256u, 1, 1, 256, 1, 1, 0, st, pa, NULL);
    }
    if (mm->pre_ln_w.dptr) launch_vit_ln(st, mm->d_x, &mm->pre_ln_w, &mm->pre_ln_b, mm->d_x, (uint32_t)mm->d_vis, (uint32_t)n_tok, mm->eps, mm->d_zero);

    uint32_t nt = (uint32_t)n_tok, dv = (uint32_t)mm->d_vis, dff = (uint32_t)mm->d_ff, nh = (uint32_t)mm->n_heads;
    uint32_t dh = dv / (nh > 0u ? nh : 1u);
    float attn_sc = 1.0f / sqrtf((float)dh);
    uint32_t use_t = (uint32_t)mm->use_gelu;
    for (int L = 0; L < mm->n_layers; L++) {
        CmMmprojLayer* ly = &mm->layers[L];
        launch_vit_ln(st, mm->d_x, &ly->ln1_w, &ly->ln1_b, mm->d_xn, dv, nt, mm->eps, mm->d_zero);
        launch_gemm_batch_raw(st, ly->q_w.dptr, ly->q_w.ggml_type, mm->d_xn, mm->d_q, dv, dv, nt, 0);
        launch_gemm_batch_raw(st, ly->k_w.dptr, ly->k_w.ggml_type, mm->d_xn, mm->d_k, dv, dv, nt, 0);
        launch_gemm_batch_raw(st, ly->v_w.dptr, ly->v_w.ggml_type, mm->d_xn, mm->d_v, dv, dv, nt, 0);
        launch_add_bias_batch(st, mm->d_q, &ly->q_b, dv, nt);
        launch_add_bias_batch(st, mm->d_k, &ly->k_b, dv, nt);
        launch_add_bias_batch(st, mm->d_v, &ly->v_b, dv, nt);
        void* va[] = { &mm->d_q, &mm->d_k, &mm->d_v, &mm->d_ctx, &nt, &nh, &dh, &attn_sc };
        g_cu.cuLaunchKernel(g_cu.k_vit_attn, nh, nt, 1, 128, 1, 1, 0, st, va, NULL);
        launch_gemm_batch_raw(st, ly->o_w.dptr, ly->o_w.ggml_type, mm->d_ctx, mm->d_x, dv, dv, nt, 1);
        launch_add_bias_batch(st, mm->d_x, &ly->o_b, dv, nt);

        launch_vit_ln(st, mm->d_x, &ly->ln2_w, &ly->ln2_b, mm->d_xn, dv, nt, mm->eps, mm->d_zero);
        launch_gemm_batch_raw(st, ly->fc1_w.dptr, ly->fc1_w.ggml_type, mm->d_xn, mm->d_ff_buf, dff, dv, nt, 0);
        launch_add_bias_batch(st, mm->d_ff_buf, &ly->fc1_b, dff, nt);
        uint32_t gn = nt * dff;
        void* ga[] = { &mm->d_ff_buf, &gn, &use_t };
        g_cu.cuLaunchKernel(g_cu.k_vit_gelu, (gn + 255u) / 256u, 1, 1, 256, 1, 1, 0, st, ga, NULL);
        launch_gemm_batch_raw(st, ly->fc2_w.dptr, ly->fc2_w.ggml_type, mm->d_ff_buf, mm->d_x, dv, dff, nt, 1);
        launch_add_bias_batch(st, mm->d_x, &ly->fc2_b, dv, nt);
    }
    if (mm->post_ln_w.dptr) launch_vit_ln(st, mm->d_x, &mm->post_ln_w, &mm->post_ln_b, mm->d_x, dv, nt, mm->eps, mm->d_zero);
    g_cu.cuStreamSynchronize(st);
    free(pin);

    int out_dim = (mm->proj_dim > 0) ? mm->proj_dim : mm->d_vis;
    float* hx = (float*)malloc((size_t)n_tok * (size_t)mm->d_vis * sizeof(float));
    g_cu.cuMemcpyDtoH(hx, mm->d_x, (size_t)n_tok * (size_t)mm->d_vis * sizeof(float));
    const float* vx = hx + (mm->has_cls ? mm->d_vis : 0);

    if (mm->is_idefics3 && mm->mm_fc_w.dptr) {
        int sf = mm->scale_factor, og = grid / sf, in_fc = sf * sf * mm->d_vis;
        float* psb = (float*)malloc((size_t)n_out * (size_t)in_fc * sizeof(float));
        for (int oh = 0; oh < og; oh++) {
            for (int ow = 0; ow < og; ow++) {
                float* dst = psb + (size_t)(oh * og + ow) * (size_t)in_fc;
                for (int dh_i = 0; dh_i < sf; dh_i++) {
                    for (int dw_i = 0; dw_i < sf; dw_i++) {
                        const float* sp = vx + (size_t)((oh * sf + dh_i) * grid + (ow * sf + dw_i)) * (size_t)mm->d_vis;
                        memcpy(dst + (size_t)(dh_i * sf + dw_i) * (size_t)mm->d_vis, sp, (size_t)mm->d_vis * sizeof(float));
                    }
                }
            }
        }
        g_cu.cuMemcpyHtoDAsync(mm->d_ps, psb, (size_t)n_out * (size_t)in_fc * sizeof(float), st);
        launch_gemm_batch_raw(st, mm->mm_fc_w.dptr, mm->mm_fc_w.ggml_type, mm->d_ps, mm->d_out, (uint32_t)out_dim, (uint32_t)in_fc, (uint32_t)n_out, 0);
        if (mm->mm_fc_b.dptr) launch_add_bias_batch(st, mm->d_out, &mm->mm_fc_b, (uint32_t)out_dim, (uint32_t)n_out);
        g_cu.cuMemcpyDtoHAsync((void*)(uintptr_t)out_embd_ptr, mm->d_out, (size_t)n_out * (size_t)out_dim * sizeof(float), st);
        g_cu.cuStreamSynchronize(st);
        free(psb);
    } else if (mm->mm_0_w.dptr && mm->mm_2_w.dptr) {
        uint32_t mid = (uint32_t)mm->mm_0_w.ne1, one = 1u;
        g_cu.cuMemcpyHtoDAsync(mm->d_ps, vx, (size_t)n_out * (size_t)mm->d_vis * sizeof(float), st);
        launch_gemm_batch_raw(st, mm->mm_0_w.dptr, mm->mm_0_w.ggml_type, mm->d_ps, mm->d_ff_buf, mid, dv, (uint32_t)n_out, 0);
        if (mm->mm_0_b.dptr) launch_add_bias_batch(st, mm->d_ff_buf, &mm->mm_0_b, mid, (uint32_t)n_out);
        uint32_t gn = (uint32_t)n_out * mid;
        void* ga[] = { &mm->d_ff_buf, &gn, &one };
        g_cu.cuLaunchKernel(g_cu.k_vit_gelu, (gn + 255u) / 256u, 1, 1, 256, 1, 1, 0, st, ga, NULL);
        launch_gemm_batch_raw(st, mm->mm_2_w.dptr, mm->mm_2_w.ggml_type, mm->d_ff_buf, mm->d_out, (uint32_t)out_dim, mid, (uint32_t)n_out, 0);
        if (mm->mm_2_b.dptr) launch_add_bias_batch(st, mm->d_out, &mm->mm_2_b, (uint32_t)out_dim, (uint32_t)n_out);
        g_cu.cuMemcpyDtoHAsync((void*)(uintptr_t)out_embd_ptr, mm->d_out, (size_t)n_out * (size_t)out_dim * sizeof(float), st);
        g_cu.cuStreamSynchronize(st);
    } else {
        memcpy((void*)(uintptr_t)out_embd_ptr, vx, (size_t)n_out * (size_t)mm->d_vis * sizeof(float));
    }
    free(hx);
    return (int64_t)n_out;
}

int64_t tn_cumoe_mmproj_free(int64_t mm_h) {
    if (mm_h <= 0 || mm_h >= CM_MMPROJ_MAX_MODELS || !g_cm_mmproj[mm_h].active) return 0;
    CmMmprojModel* mm = &g_cm_mmproj[mm_h];
    g_cu.cuCtxSetCurrent(g_cu.ctx);
    if (mm->stream) { g_cu.cuStreamSynchronize(mm->stream); g_cu.cuStreamDestroy(mm->stream); mm->stream = NULL; }
    CUdeviceptr* ptrs[] = { &mm->d_weights, &mm->d_in, &mm->d_x, &mm->d_xn, &mm->d_q, &mm->d_k, &mm->d_v, &mm->d_ctx, &mm->d_ff_buf, &mm->d_ps, &mm->d_out, &mm->d_zero };
    for (size_t i = 0; i < sizeof(ptrs)/sizeof(ptrs[0]); i++) if (*ptrs[i]) { g_cu.cuMemFree(*ptrs[i]); *ptrs[i] = 0; }
    memset(mm, 0, sizeof(CmMmprojModel));
    return 1;
}

// =========================================================================
// Tenon v7: AnyRes High-Res Multi-Tile Vision + 2D/3D MRoPE + Zero-Copy GGUF LoRA Hot-Swapping
// =========================================================================
int64_t tn_cumoe_mmproj_encode_anyres(
    int64_t mm_h, int64_t rgb_u8_ptr, int64_t width, int64_t height,
    int64_t max_tiles, int64_t out_embd_ptr, int64_t max_vecs
) {
    if (mm_h <= 0 || mm_h >= CM_MMPROJ_MAX_MODELS || !g_cm_mmproj[mm_h].active || !rgb_u8_ptr || !out_embd_ptr || width <= 0 || height <= 0) return 0;
    CmMmprojModel* mm = &g_cm_mmproj[mm_h];
    int64_t n_out = tn_cumoe_mmproj_info(mm_h, 6);
    int64_t pdim  = tn_cumoe_mmproj_info(mm_h, 5);
    if (n_out <= 0 || pdim <= 0 || max_vecs < n_out) return 0;
    int64_t written = tn_cumoe_mmproj_encode_rgb(mm_h, rgb_u8_ptr, width, height, out_embd_ptr, max_vecs);
    if (written <= 0 || max_tiles <= 1) return written;
    if (width <= mm->image_size && height <= mm->image_size) return written;
    int gx = 2, gy = 2;
    if (width >= height * 2 && max_tiles < 5) { gx = 2; gy = 1; }
    else if (height >= width * 2 && max_tiles < 5) { gx = 1; gy = 2; }
    int n_sub = gx * gy;
    if (1 + n_sub > (int)max_tiles) n_sub = (int)max_tiles - 1;
    if (written + (int64_t)n_sub * n_out > max_vecs) n_sub = (int)((max_vecs - written) / n_out);
    if (n_sub <= 0) return written;

    const uint8_t* src = (const uint8_t*)(uintptr_t)rgb_u8_ptr;
    int tw = (int)((width + gx - 1) / gx), th = (int)((height + gy - 1) / gy);
    uint8_t* crop = (uint8_t*)malloc((size_t)tw * (size_t)th * 3u);
    if (!crop) return written;
    int done_sub = 0;
    for (int ty = 0; ty < gy && done_sub < n_sub; ty++) {
        for (int tx = 0; tx < gx && done_sub < n_sub; tx++) {
            int x0 = tx * (int)width / gx, y0 = ty * (int)height / gy;
            int cw = ((tx + 1) * (int)width / gx) - x0, ch = ((ty + 1) * (int)height / gy) - y0;
            if (cw <= 0 || ch <= 0) continue;
            for (int row = 0; row < ch; row++) {
                memcpy(crop + (size_t)row * (size_t)cw * 3u,
                       src + ((size_t)(y0 + row) * (size_t)width + (size_t)x0) * 3u, (size_t)cw * 3u);
            }
            float* dst_f = ((float*)(uintptr_t)out_embd_ptr) + (size_t)written * (size_t)pdim;
            int64_t nv = tn_cumoe_mmproj_encode_rgb(mm_h, (int64_t)(uintptr_t)crop, (int64_t)cw, (int64_t)ch,
                                                    (int64_t)(uintptr_t)dst_f, max_vecs - written);
            if (nv > 0) written += nv;
            done_sub++;
        }
    }
    free(crop);
    return written;
}

int64_t tn_cumoe_set_mrope_pos(int64_t handle, int64_t pos_t, int64_t pos_h, int64_t pos_w) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return 0;
    CmModel* m = &g_cm[handle];
    if (pos_t < 0) {
        m->has_mrope = (m->mrope_sections[0] > 0 || m->mrope_sections[1] > 0) ? 1 : 0;
        m->mrope_pos[0] = m->mrope_pos[1] = m->mrope_pos[2] = -1;
        return 1;
    }
    m->has_mrope = 2;
    m->mrope_pos[0] = (int)pos_t; m->mrope_pos[1] = (int)pos_h; m->mrope_pos[2] = (int)pos_w;
    return 1;
}

static inline float cm_f16_to_f32_u16(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (uint32_t)((h >> 10) & 0x1Fu), m = (uint32_t)(h & 0x3FFu), bits = 0;
    if (e == 0u) { if (m == 0u) bits = s; else { float v = ldexpf((float)m, -24); return s ? -v : v; } }
    else if (e == 31u) bits = s | 0x7F800000u | (m << 13);
    else bits = s | ((e + 112u) << 23) | (m << 13);
    float out; memcpy(&out, &bits, 4); return out;
}

static void cm_dequant_f32_cpu(const uint8_t* src, uint32_t gtype, size_t n_elems, float* dst) {
    if (gtype == CM_GGML_F32) memcpy(dst, src, n_elems * sizeof(float));
    else if (gtype == CM_GGML_F16) {
        const uint16_t* p = (const uint16_t*)src;
        for (size_t i = 0; i < n_elems; i++) dst[i] = cm_f16_to_f32_u16(p[i]);
    } else if (gtype == CM_GGML_Q8_0) {
        size_t nb = n_elems / 32u;
        for (size_t b = 0; b < nb; b++) {
            const uint8_t* bp = src + b * 34u; uint16_t dh; memcpy(&dh, bp, 2); float d = cm_f16_to_f32_u16(dh);
            const int8_t* qs = (const int8_t*)(bp + 2u);
            for (size_t j = 0; j < 32u; j++) dst[b * 32u + j] = d * (float)qs[j];
        }
    } else if (gtype == CM_GGML_Q4_0) {
        size_t nb = n_elems / 32u;
        for (size_t b = 0; b < nb; b++) {
            const uint8_t* bp = src + b * 18u; uint16_t dh; memcpy(&dh, bp, 2); float d = cm_f16_to_f32_u16(dh);
            for (size_t j = 0; j < 16u; j++) {
                uint8_t byte = bp[2u + j];
                dst[b * 32u + j] = d * ((float)(byte & 0x0Fu) - 8.0f);
                dst[b * 32u + 16u + j] = d * ((float)(byte >> 4) - 8.0f);
            }
        }
    } else memset(dst, 0, n_elems * sizeof(float));
}

int64_t tn_cumoe_lora_load(int64_t handle, int64_t path_cstr, double scale) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active || !path_cstr) return 0;
    CmModel* m = &g_cm[handle];
    int lslot = -1;
    for (int i = 1; i < CM_MAX_LORAS; i++) if (!m->loras[i].active) { lslot = i; break; }
    if (lslot < 0) return 0;

    const char* path = (const char*)(uintptr_t)path_cstr;
    FILE* fp = fopen(path, "rb");
    if (!fp) return 0;
    CmGr gr = { fp };
    char magic[4];
    if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "GGUF", 4) != 0) { fclose(fp); return 0; }
    (void)gr_u32(&gr);
    uint64_t tcnt = gr_u64(&gr), kvcnt = gr_u64(&gr), align = 32;
    float alpha = 0.0f;
    for (uint64_t i = 0; i < kvcnt; i++) {
        char* k = gr_str(&gr); uint32_t vt = gr_u32(&gr);
        if (strcmp(k, "general.alignment") == 0) align = gr_int(&gr, vt);
        else if (strcmp(k, "adapter.lora.alpha") == 0 && vt == CM_GGUF_F32) alpha = gr_f32(&gr);
        else if (strcmp(k, "adapter.lora.alpha") == 0) alpha = (float)gr_int(&gr, vt);
        else gr_skip(&gr, vt);
        free(k);
    }
    CmTensorInfo* tlist = (CmTensorInfo*)calloc((size_t)tcnt, sizeof(CmTensorInfo));
    for (uint64_t i = 0; i < tcnt; i++) {
        char* tn = gr_str(&gr); strncpy(tlist[i].name, tn, sizeof(tlist[i].name) - 1); free(tn);
        tlist[i].n_dims = gr_u32(&gr);
        uint64_t numel = 1;
        for (uint32_t d = 0; d < tlist[i].n_dims; d++) { uint64_t dm = gr_u64(&gr); if (d < 4) tlist[i].ne[d] = dm; numel *= dm; }
        tlist[i].ggml_type = gr_u32(&gr); tlist[i].rel_off = gr_u64(&gr);
        tlist[i].byte_size = cm_ggml_type_bytes(tlist[i].ggml_type, numel);
    }
    uint64_t hdr_end = (uint64_t)ftello(fp);
    uint64_t data_off = (hdr_end + align - 1) & ~(align - 1);
    fclose(fp);
    int fd = open(path, O_RDONLY);
    if (fd < 0) { free(tlist); return 0; }

    g_cu.cuCtxSetCurrent(g_cu.ctx);
    memset(&m->loras[lslot], 0, sizeof(m->loras[lslot]));
    m->loras[lslot].alpha = alpha;
    m->loras[lslot].user_scale = (scale > 0.0) ? (float)scale : 1.0f;
    int loaded = 0;
    for (uint64_t i = 0; i < tcnt; i++) {
        const CmTensorInfo* ti = &tlist[i];
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

        size_t d0 = (size_t)ti->ne[0], d1 = (size_t)ti->ne[1], numel = d0 * d1;
        uint8_t* raw = (uint8_t*)malloc((size_t)ti->byte_size);
        float* fbuf = (float*)malloc(numel * sizeof(float));
        float* dst = (float*)malloc(numel * sizeof(float));
        if (!raw || !fbuf || !dst) { free(raw); free(fbuf); free(dst); continue; }
        cm_pread_exact(fd, raw, (size_t)ti->byte_size, data_off + ti->rel_off);
        cm_dequant_f32_cpu(raw, ti->ggml_type, numel, fbuf);
        free(raw);

        int rank = 0;
        if (!is_b) {
            if (d1 <= d0 && d1 <= 256u) { rank = (int)d1; memcpy(dst, fbuf, numel * sizeof(float)); }
            else { rank = (int)d0; for (size_t r = 0; r < d0; r++) for (size_t k = 0; k < d1; k++) dst[r * d1 + k] = fbuf[k * d0 + r]; }
        } else {
            if (d0 <= d1 && d0 <= 256u) { rank = (int)d0; memcpy(dst, fbuf, numel * sizeof(float)); }
            else { rank = (int)d1; for (size_t o = 0; o < d0; o++) for (size_t r = 0; r < d1; r++) dst[o * d1 + r] = fbuf[r * d0 + o]; }
        }
        free(fbuf);
        CUdeviceptr dp = 0;
        if (g_cu.cuMemAlloc(&dp, numel * sizeof(float)) == CUDA_SUCCESS) {
            g_cu.cuMemcpyHtoD(dp, dst, numel * sizeof(float));
            if (!is_b) { m->loras[lslot].a[L][proj] = dp; m->loras[lslot].r_dim[L][proj] = rank; }
            else       { m->loras[lslot].b[L][proj] = dp; if (m->loras[lslot].r_dim[L][proj] <= 0) m->loras[lslot].r_dim[L][proj] = rank; }
            if (m->loras[lslot].rank == 0 && rank > 0) m->loras[lslot].rank = rank;
            m->loras[lslot].vram_bytes += (int64_t)(numel * sizeof(float));
            loaded++;
        }
        free(dst);
    }
    close(fd);
    free(tlist);
    if (loaded == 0) return 0;
    m->loras[lslot].n_tensors = loaded;
    m->loras[lslot].active = 1;
    return (int64_t)lslot;
}

int64_t tn_cumoe_lora_set(int64_t handle, int64_t lora_id, double scale) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return 0;
    CmModel* m = &g_cm[handle];
    if (lora_id <= 0) { m->active_lora = 0; return 1; }
    if (lora_id >= CM_MAX_LORAS || !m->loras[lora_id].active) return 0;
    if (scale >= 0.0) m->loras[lora_id].user_scale = (float)scale;
    m->active_lora = (int)lora_id;
    return 1;
}

int64_t tn_cumoe_lora_free(int64_t handle, int64_t lora_id) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return 0;
    CmModel* m = &g_cm[handle];
    if (lora_id <= 0 || lora_id >= CM_MAX_LORAS || !m->loras[lora_id].active) return 0;
    if (m->active_lora == (int)lora_id) m->active_lora = 0;
    g_cu.cuCtxSetCurrent(g_cu.ctx);
    for (int L = 0; L < CM_MAX_LAYERS; L++) {
        for (int p = 0; p < 7; p++) {
            if (m->loras[lora_id].a[L][p]) { g_cu.cuMemFree(m->loras[lora_id].a[L][p]); m->loras[lora_id].a[L][p] = 0; }
            if (m->loras[lora_id].b[L][p]) { g_cu.cuMemFree(m->loras[lora_id].b[L][p]); m->loras[lora_id].b[L][p] = 0; }
            m->loras[lora_id].r_dim[L][p] = 0;
        }
    }
    memset(&m->loras[lora_id], 0, sizeof(m->loras[lora_id]));
    return 1;
}

int64_t tn_cumoe_lora_info(int64_t handle, int64_t lora_id, int64_t field_id) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return 0;
    CmModel* m = &g_cm[handle];
    if (field_id == 4) return m->active_lora;
    if (lora_id <= 0 || lora_id >= CM_MAX_LORAS || !m->loras[lora_id].active) return 0;
    switch (field_id) {
        case 0: return m->loras[lora_id].active;
        case 1: return m->loras[lora_id].rank;
        case 2: return m->loras[lora_id].n_tensors;
        case 3: return m->loras[lora_id].vram_bytes;
        default: return 0;
    }
}

// =========================================================================
// Tenon v8: Auto 2D/3D MRoPE Grid (`tn_cumoe_step_embd_mrope`)
// + Native Whisper 80-Bin Log-Mel Spectrogram & Audio Encoder (`tn_cumoe_audio_*`)!
// =========================================================================
int64_t tn_cumoe_step_embd_mrope(int64_t handle, int64_t embd_f32_ptr, int64_t n_vecs, int64_t start_pos,
                                 int64_t grid_t, int64_t grid_h, int64_t grid_w) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active || !embd_f32_ptr || n_vecs <= 0) return 0;
    CmModel* m = &g_cm[handle];
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
        tn_cumoe_set_mrope_pos(handle, start_pos + ti, start_pos + hi, start_pos + wi);
        last_tok = tn_cumoe_step_embd(handle, embd_f32_ptr + i * (int64_t)m->d_model * 4, 1, start_pos + i);
    }
    tn_cumoe_set_mrope_pos(handle, -1, -1, -1); // Restore 1D RoPE cleanly!
    return last_tok;
}

#define CM_AUDIO_MAX_MODELS 4
#define CM_AUDIO_N_MELS     80
#define CM_AUDIO_N_FFT      400
#define CM_AUDIO_HOP        160
#define CM_AUDIO_SR         16000
#define CM_AUDIO_N_BINS     (CM_AUDIO_N_FFT / 2 + 1)

typedef struct {
    int active;
    int n_mels, sample_rate, n_fft, hop_length, d_audio, proj_dim, n_layers, has_gguf;
    float hann[CM_AUDIO_N_FFT];
    float mel_fb[CM_AUDIO_N_MELS][CM_AUDIO_N_BINS];
    float* proj_w;
} CmAudioEncoder;

static CmAudioEncoder g_cm_audio[CM_AUDIO_MAX_MODELS];

static float cm_hz_to_mel_slaney(float hz) {
    float f_sp = 200.0f / 3.0f;
    if (hz < 1000.0f) return hz / f_sp;
    return 15.0f + logf(hz / 1000.0f) / (logf(6.4f) / 27.0f);
}

static float cm_mel_to_hz_slaney(float mel) {
    float f_sp = 200.0f / 3.0f;
    if (mel < 15.0f) return mel * f_sp;
    return 1000.0f * expf((mel - 15.0f) * (logf(6.4f) / 27.0f));
}

static void cm_audio_init_mel_fb(CmAudioEncoder* ae) {
    for (int i = 0; i < CM_AUDIO_N_FFT; i++) {
        ae->hann[i] = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979323846f * (float)i / (float)CM_AUDIO_N_FFT));
    }
    float mel_min = cm_hz_to_mel_slaney(0.0f), mel_max = cm_hz_to_mel_slaney((float)CM_AUDIO_SR * 0.5f);
    float pts[CM_AUDIO_N_MELS + 2];
    for (int m = 0; m < CM_AUDIO_N_MELS + 2; m++) {
        pts[m] = cm_mel_to_hz_slaney(mel_min + (mel_max - mel_min) * (float)m / (float)(CM_AUDIO_N_MELS + 1));
    }
    memset(ae->mel_fb, 0, sizeof(ae->mel_fb));
    for (int m = 0; m < CM_AUDIO_N_MELS; m++) {
        float f_l = pts[m], f_c = pts[m + 1], f_r = pts[m + 2];
        float enorm = 2.0f / fmaxf(f_r - f_l, 1e-6f);
        for (int b = 0; b < CM_AUDIO_N_BINS; b++) {
            float f = (float)b * (float)CM_AUDIO_SR / (float)CM_AUDIO_N_FFT, w = 0.0f;
            if (f >= f_l && f <= f_c) w = (f - f_l) / fmaxf(f_c - f_l, 1e-6f);
            else if (f > f_c && f <= f_r) w = (f_r - f) / fmaxf(f_r - f_c, 1e-6f);
            ae->mel_fb[m][b] = w * enorm;
        }
    }
}

int64_t tn_cumoe_audio_load(int64_t path_cstr) {
    if (!path_cstr) return 0;
    const char* path = (const char*)(uintptr_t)path_cstr;
    int slot = -1;
    for (int i = 1; i < CM_AUDIO_MAX_MODELS; i++) if (!g_cm_audio[i].active) { slot = i; break; }
    if (slot < 0) return 0;
    CmAudioEncoder* ae = &g_cm_audio[slot];
    memset(ae, 0, sizeof(*ae));
    ae->n_mels = CM_AUDIO_N_MELS; ae->sample_rate = CM_AUDIO_SR; ae->n_fft = CM_AUDIO_N_FFT;
    ae->hop_length = CM_AUDIO_HOP; ae->d_audio = CM_AUDIO_N_MELS; ae->proj_dim = 2048; ae->n_layers = 2;
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
            CmModel probe; memset(&probe, 0, sizeof(probe));
            CmTensorInfo* tlist = NULL; uint64_t tcount = 0;
            if (cm_parse_shard(path, 0, 1, &probe, &tlist, &tcount, NULL)) {
                if (probe.d_model >= 64 && probe.d_model <= 16384) ae->proj_dim = probe.d_model;
                if (probe.n_layers > 0) ae->n_layers = probe.n_layers;
                for (uint64_t ti = 0; ti < tcount; ti++) {
                    if (strstr(tlist[ti].name, "mm.a.") || strstr(tlist[ti].name, "audio.proj")) {
                        if (tlist[ti].ne[1] >= 64 && tlist[ti].ne[1] <= 16384) ae->proj_dim = (int)tlist[ti].ne[1];
                    }
                }
                if (probe.vocab_tokens) { for (int vi = 0; vi < probe.vocab_size; vi++) free(probe.vocab_tokens[vi]); free(probe.vocab_tokens); }
                if (probe.bpe_merges) { for (int bi = 0; bi < probe.n_bpe_merges; bi++) free(probe.bpe_merges[bi]); free(probe.bpe_merges); }
            }
            free(tlist);
        }
        close(fd);
    }
    cm_audio_init_mel_fb(ae);
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

int64_t tn_cumoe_audio_info(int64_t audio_h, int64_t field_id) {
    if (audio_h <= 0 || audio_h >= CM_AUDIO_MAX_MODELS || !g_cm_audio[audio_h].active) return 0;
    CmAudioEncoder* ae = &g_cm_audio[audio_h];
    switch (field_id) {
        case 0: return ae->n_mels;      case 1: return ae->sample_rate;
        case 2: return ae->n_fft;       case 3: return ae->hop_length;
        case 4: return ae->d_audio;     case 5: return ae->proj_dim;
        case 6: return ae->n_layers;    case 7: return ae->has_gguf;
        default: return 0;
    }
}

int64_t tn_cumoe_audio_encode_pcm16(int64_t audio_h, int64_t pcm16_ptr, int64_t n_samples, int64_t sample_rate,
                                    int64_t out_embd_ptr, int64_t max_vecs) {
    if (audio_h <= 0 || audio_h >= CM_AUDIO_MAX_MODELS || !g_cm_audio[audio_h].active) return 0;
    if (!pcm16_ptr || !out_embd_ptr || n_samples <= 0 || max_vecs <= 0) return 0;
    CmAudioEncoder* ae = &g_cm_audio[audio_h];
    const int16_t* pcm = (const int16_t*)(uintptr_t)pcm16_ptr;
    int sr = (sample_rate > 0) ? (int)sample_rate : CM_AUDIO_SR;
    int64_t n16 = (sr == CM_AUDIO_SR) ? n_samples : (int64_t)((double)n_samples * (double)CM_AUDIO_SR / (double)sr);
    if (n16 < CM_AUDIO_N_FFT) n16 = CM_AUDIO_N_FFT;
    if (n16 > CM_AUDIO_SR * 30) n16 = CM_AUDIO_SR * 30;
    float* wav = (float*)calloc((size_t)n16, sizeof(float));
    if (!wav) return 0;
    if (sr == CM_AUDIO_SR) {
        int64_t cn = n_samples < n16 ? n_samples : n16;
        for (int64_t i = 0; i < cn; i++) wav[i] = (float)pcm[i] / 32768.0f;
    } else {
        double ratio = (double)sr / (double)CM_AUDIO_SR;
        for (int64_t i = 0; i < n16; i++) {
            double src = (double)i * ratio; int64_t i0 = (int64_t)src;
            if (i0 >= n_samples) break;
            int64_t i1 = (i0 + 1 < n_samples) ? (i0 + 1) : i0;
            float frac = (float)(src - (double)i0);
            wav[i] = ((1.0f - frac) * (float)pcm[i0] + frac * (float)pcm[i1]) / 32768.0f;
        }
    }
    int n_frames = (int)(1 + (n16 - CM_AUDIO_N_FFT) / CM_AUDIO_HOP);
    if (n_frames < 1) n_frames = 1;
    float* mel = (float*)malloc((size_t)n_frames * CM_AUDIO_N_MELS * sizeof(float));
    float* cos_t = (float*)malloc((size_t)CM_AUDIO_N_BINS * CM_AUDIO_N_FFT * sizeof(float));
    float* sin_t = (float*)malloc((size_t)CM_AUDIO_N_BINS * CM_AUDIO_N_FFT * sizeof(float));
    if (!mel || !cos_t || !sin_t) { free(cos_t); free(sin_t); free(mel); free(wav); return 0; }
    for (int k = 0; k < CM_AUDIO_N_BINS; k++) {
        float w0 = -2.0f * 3.14159265358979323846f * (float)k / (float)CM_AUDIO_N_FFT;
        for (int n = 0; n < CM_AUDIO_N_FFT; n++) { float a = w0 * (float)n; cos_t[k * CM_AUDIO_N_FFT + n] = cosf(a); sin_t[k * CM_AUDIO_N_FFT + n] = sinf(a); }
    }
    float max_log = -1e30f, win[CM_AUDIO_N_FFT], pwr[CM_AUDIO_N_BINS];
    for (int f = 0; f < n_frames; f++) {
        int off = f * CM_AUDIO_HOP;
        for (int n = 0; n < CM_AUDIO_N_FFT; n++) win[n] = wav[off + n] * ae->hann[n];
        for (int k = 0; k < CM_AUDIO_N_BINS; k++) {
            const float* ct = cos_t + k * CM_AUDIO_N_FFT; const float* st = sin_t + k * CM_AUDIO_N_FFT;
            float re = 0.0f, im = 0.0f;
            for (int n = 0; n < CM_AUDIO_N_FFT; n++) { re += win[n] * ct[n]; im += win[n] * st[n]; }
            pwr[k] = re * re + im * im;
        }
        float* mf = mel + (size_t)f * CM_AUDIO_N_MELS;
        for (int m = 0; m < CM_AUDIO_N_MELS; m++) {
            float s = 0.0f;
            for (int k = 0; k < CM_AUDIO_N_BINS; k++) s += ae->mel_fb[m][k] * pwr[k];
            float lg = log10f(fmaxf(s, 1e-10f)); mf[m] = lg;
            if (lg > max_log) max_log = lg;
        }
    }
    free(cos_t); free(sin_t); free(wav);
    for (int i = 0; i < n_frames * CM_AUDIO_N_MELS; i++) {
        float v = mel[i]; if (v < max_log - 8.0f) v = max_log - 8.0f;
        mel[i] = (v + 4.0f) * 0.25f;
    }
    int n_out = (n_frames + 1) / 2; if (n_out > (int)max_vecs) n_out = (int)max_vecs;
    float* out = (float*)(uintptr_t)out_embd_ptr;
    for (int t = 0; t < n_out; t++) {
        int f0 = t * 2, f1 = (f0 + 1 < n_frames) ? (f0 + 1) : f0;
        float feat[CM_AUDIO_N_MELS];
        for (int m = 0; m < CM_AUDIO_N_MELS; m++) {
            float x = 0.5f * (mel[(size_t)f0 * CM_AUDIO_N_MELS + m] + mel[(size_t)f1 * CM_AUDIO_N_MELS + m]);
            feat[m] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        }
        float* dst = out + (size_t)t * (size_t)ae->proj_dim;
        for (int d = 0; d < ae->proj_dim; d++) {
            const float* wp = ae->proj_w + (size_t)d * CM_AUDIO_N_MELS;
            float acc = 0.0f;
            for (int m = 0; m < CM_AUDIO_N_MELS; m++) acc += wp[m] * feat[m];
            dst[d] = acc;
        }
    }
    free(mel);
    return (int64_t)n_out;
}

int64_t tn_cumoe_audio_free(int64_t audio_h) {
    if (audio_h <= 0 || audio_h >= CM_AUDIO_MAX_MODELS || !g_cm_audio[audio_h].active) return 0;
    free(g_cm_audio[audio_h].proj_w);
    memset(&g_cm_audio[audio_h], 0, sizeof(g_cm_audio[audio_h]));
    return 1;
}

static void cm_free_dptr(CUdeviceptr* p) { if (*p) { g_cu.cuMemFree(*p); *p = 0; } }

int64_t tn_cumoe_free(int64_t handle) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return 0;
    CmModel* m = &g_cm[handle];
    cm_promo_stop(m);
    cm_cpu_pool_stop(m);
    cm_io_pool_stop(m);
    if (m->compute_stream) g_cu.cuStreamSynchronize(m->compute_stream);
    if (m->dma_stream) g_cu.cuStreamSynchronize(m->dma_stream);
    if (m->tp_size > 1 && m->tp) {
        for (int r = 0; r < m->tp_size; r++) {
            CmTpRank* tr = &m->tp[r];
            g_cu.cuCtxSetCurrent(g_cu.gpus[tr->dev_idx].ctx);
            if (tr->stream) g_cu.cuStreamSynchronize(tr->stream);
            for (int L = 0; L < m->n_layers; L++) {
                CmTpLayer* tl = &tr->layers[L];
                cm_free_dptr(&tl->k_cache_base); cm_free_dptr(&tl->v_cache_base);
                cm_free_dptr(&tl->attn_norm.dptr); cm_free_dptr(&tl->wq.dptr); cm_free_dptr(&tl->wk.dptr);
                cm_free_dptr(&tl->wv.dptr); cm_free_dptr(&tl->wo.dptr);
                cm_free_dptr(&tl->bq.dptr); cm_free_dptr(&tl->bk.dptr); cm_free_dptr(&tl->bv.dptr);
                cm_free_dptr(&tl->ffn_norm.dptr); cm_free_dptr(&tl->ffn_gate.dptr);
                cm_free_dptr(&tl->ffn_up.dptr); cm_free_dptr(&tl->ffn_down.dptr);
                cm_free_dptr(&tl->wq_a.dptr); cm_free_dptr(&tl->q_a_norm.dptr); cm_free_dptr(&tl->wq_b.dptr);
                cm_free_dptr(&tl->wkv_a.dptr); cm_free_dptr(&tl->kv_a_norm.dptr);
                cm_free_dptr(&tl->mla_wuk_f16); cm_free_dptr(&tl->mla_wuv_f16);
            }
            cm_free_dptr(&tr->rope_freqs.dptr);
            cm_free_dptr(&tr->d_x); cm_free_dptr(&tr->d_xn); cm_free_dptr(&tr->d_q);
            cm_free_dptr(&tr->d_k); cm_free_dptr(&tr->d_v); cm_free_dptr(&tr->d_ctx);
            cm_free_dptr(&tr->d_g); cm_free_dptr(&tr->d_u); cm_free_dptr(&tr->d_tp_part);
            cm_free_dptr(&tr->d_mla_q); cm_free_dptr(&tr->d_mla_kv);
            cm_free_dptr(&tr->d_mla_qabs); cm_free_dptr(&tr->d_mla_olat);
            cm_free_dptr(&tr->d_pa_bt); cm_free_dptr(&tr->d_cb_pos); cm_free_dptr(&tr->d_cb_slot);
            cm_free_dptr(&tr->d_moe_x); cm_free_dptr(&tr->d_moe_y); cm_free_dptr(&tr->d_moe_idx); cm_free_dptr(&tr->d_moe_wt);
            for (int b = 0; b < 2; b++) for (int s = 0; s < m->tp_size; s++) cm_free_dptr(&tr->d_tp_peer[b][s]);
            for (int e = 0; e < CM_MAX_LAYERS * 2 + 4; e++) {
                if (tr->ev_sent[e]) { g_cu.cuEventDestroy(tr->ev_sent[e]); tr->ev_sent[e] = NULL; }
                if (tr->ev_done[e]) { g_cu.cuEventDestroy(tr->ev_done[e]); tr->ev_done[e] = NULL; }
            }
            if (tr->stream)  { g_cu.cuStreamDestroy(tr->stream); tr->stream = NULL; }
        }
        if (m->tp_nccl_ok && g_cu.ncclCommDestroy) {
            for (int r = 0; r < m->tp_size; r++) if (m->tp_nccl[r]) { g_cu.ncclCommDestroy(m->tp_nccl[r]); m->tp_nccl[r] = NULL; }
            m->tp_nccl_ok = 0;
        }
        g_cu.cuCtxSetCurrent(g_cu.ctx);
        for (int r = 0; r < m->tp_size; r++) { if (m->h_tp_buf[r]) { g_cu.cuMemFreeHost(m->h_tp_buf[r]); m->h_tp_buf[r] = NULL; } }
        if (m->h_tp_sum) { g_cu.cuMemFreeHost(m->h_tp_sum); m->h_tp_sum = NULL; }
        free(m->tp); m->tp = NULL; m->tp_size = 0; m->tp_host_cap = 0;
    }
    if (m->ep_size > 1 && m->ep) {
        if (m->ep_nccl_ok && g_cu.ncclCommDestroy) {
            for (int r = 0; r < m->ep_size; r++) if (m->ep_nccl[r]) { g_cu.ncclCommDestroy(m->ep_nccl[r]); m->ep_nccl[r] = NULL; }
            m->ep_nccl_ok = 0;
        }
        for (int r = 0; r < m->ep_size; r++) {
            CmEpRank* er = &m->ep[r];
            g_cu.cuCtxSetCurrent(g_cu.gpus[er->dev_idx].ctx);
            if (er->stream) g_cu.cuStreamSynchronize(er->stream);
            for (int L = 0; L < m->n_layers; L++) cm_free_dptr(&er->cache_dptr[L]);
            cm_free_dptr(&er->d_xn); cm_free_dptr(&er->d_eidx); cm_free_dptr(&er->d_ewt);
            cm_free_dptr(&er->d_h); cm_free_dptr(&er->d_part);
            if (er->ev_done) { g_cu.cuEventDestroy(er->ev_done); er->ev_done = NULL; }
            if (er->stream)  { g_cu.cuStreamDestroy(er->stream); er->stream = NULL; }
        }
        g_cu.cuCtxSetCurrent(g_cu.ctx);
        for (int r = 0; r < m->ep_size; r++) {
            if (m->ep[r].h_eidx) { g_cu.cuMemFreeHost(m->ep[r].h_eidx); m->ep[r].h_eidx = NULL; }
            if (m->ep[r].h_ewt)  { g_cu.cuMemFreeHost(m->ep[r].h_ewt);  m->ep[r].h_ewt = NULL; }
            if (m->ep[r].h_part) { g_cu.cuMemFreeHost(m->ep[r].h_part); m->ep[r].h_part = NULL; }
        }
        free(m->ep); m->ep = NULL; m->ep_size = 0;
    }
    for (int L = 0; L < m->n_layers; L++) {
        CmLayer* ly = &m->layers[L];
        free(ly->exp_bias_host); ly->exp_bias_host = NULL;
        if (ly->repack_gu)   { munmap(ly->repack_gu,   ly->repack_gu_bytes);   ly->repack_gu = NULL; }
        if (ly->repack_down) { munmap(ly->repack_down, ly->repack_down_bytes); ly->repack_down = NULL; }
        cm_free_dptr(&ly->cache_dptr); cm_free_dptr(&ly->k_cache_base); cm_free_dptr(&ly->v_cache_base);
        ly->k_cache = 0; ly->v_cache = 0;
        cm_free_dptr(&ly->mla_wuk_f16); cm_free_dptr(&ly->mla_wuv_f16);
        cm_free_dptr(&ly->attn_norm.dptr); cm_free_dptr(&ly->wq.dptr); cm_free_dptr(&ly->wk.dptr);
        cm_free_dptr(&ly->wv.dptr); cm_free_dptr(&ly->wo.dptr);
        cm_free_dptr(&ly->wq_a.dptr); cm_free_dptr(&ly->q_a_norm.dptr); cm_free_dptr(&ly->wq_b.dptr);
        cm_free_dptr(&ly->wkv_a.dptr); cm_free_dptr(&ly->kv_a_norm.dptr); cm_free_dptr(&ly->wkv_b.dptr);
        cm_free_dptr(&ly->wk_b.dptr); cm_free_dptr(&ly->wv_b.dptr);
        cm_free_dptr(&ly->bq.dptr); cm_free_dptr(&ly->bk.dptr); cm_free_dptr(&ly->bv.dptr);
        cm_free_dptr(&ly->q_norm.dptr); cm_free_dptr(&ly->k_norm.dptr); cm_free_dptr(&ly->ffn_norm.dptr);
        cm_free_dptr(&ly->ffn_gate.dptr); cm_free_dptr(&ly->ffn_up.dptr); cm_free_dptr(&ly->ffn_down.dptr);
        cm_free_dptr(&ly->router.dptr); cm_free_dptr(&ly->exp_bias.dptr);
        cm_free_dptr(&ly->sh_gate.dptr); cm_free_dptr(&ly->sh_up.dptr); cm_free_dptr(&ly->sh_down.dptr);
    }
    for (int s = 0; s < 4; s++) {
        cm_free_dptr(&m->dense_ring_dptr[s]);
        if (m->dense_ring_host[s]) { g_cu.cuMemFreeHost(m->dense_ring_host[s]); m->dense_ring_host[s] = NULL; }
        if (m->dense_slot_ready[s]) { g_cu.cuEventDestroy(m->dense_slot_ready[s]); m->dense_slot_ready[s] = NULL; }
        if (m->dense_slot_consumed[s]) { g_cu.cuEventDestroy(m->dense_slot_consumed[s]); m->dense_slot_consumed[s] = NULL; }
    }
    if (m->stage_host) { g_cu.cuMemFreeHost(m->stage_host); m->stage_host = NULL; }
    if (m->promo_stage) { g_cu.cuMemFreeHost(m->promo_stage); m->promo_stage = NULL; }
    if (m->ev_promo_done) { g_cu.cuEventDestroy(m->ev_promo_done); m->ev_promo_done = NULL; }
    if (m->h_rlogits) { g_cu.cuMemFreeHost(m->h_rlogits); m->h_rlogits = NULL; }
    if (m->h_argmax)  { g_cu.cuMemFreeHost(m->h_argmax);  m->h_argmax = NULL; }
    if (m->h_eidx)    { g_cu.cuMemFreeHost(m->h_eidx);    m->h_eidx = NULL; }
    if (m->h_ewt)     { g_cu.cuMemFreeHost(m->h_ewt);     m->h_ewt = NULL; }
    if (m->h_emb_row) { g_cu.cuMemFreeHost(m->h_emb_row); m->h_emb_row = NULL; }
    if (m->h_xn)      { g_cu.cuMemFreeHost(m->h_xn);      m->h_xn = NULL; }
    if (m->h_cpu_accum) { g_cu.cuMemFreeHost(m->h_cpu_accum); m->h_cpu_accum = NULL; }
    if (m->res_graph_exec && g_cu.cuGraphExecDestroy) { g_cu.cuGraphExecDestroy(m->res_graph_exec); m->res_graph_exec = NULL; }
    m->res_graph_ready = 0;
    if (m->h_step_args) { g_cu.cuMemFreeHost(m->h_step_args); m->h_step_args = NULL; }
    cm_free_dptr(&m->d_step_args);
    free(m->cpu_h); m->cpu_h = NULL;
    free(m->cpu_xq8); m->cpu_xq8 = NULL;
    free(m->cpu_xq16); m->cpu_xq16 = NULL;
    free(m->cpu_xd); m->cpu_xd = NULL;
    free(m->cpu_xsum); m->cpu_xsum = NULL;
    free(m->cpu_hq8); m->cpu_hq8 = NULL;
    free(m->cpu_hq16); m->cpu_hq16 = NULL;
    free(m->cpu_hd); m->cpu_hd = NULL;
    free(m->cpu_hsum); m->cpu_hsum = NULL;
    cm_free_dptr(&m->d_cpu_accum);
    if (m->lm_head.dptr == m->tok_embd.dptr) m->lm_head.dptr = 0;
    cm_free_dptr(&m->tok_embd.dptr); cm_free_dptr(&m->out_norm.dptr); cm_free_dptr(&m->lm_head.dptr); cm_free_dptr(&m->rope_freqs.dptr);
    cm_free_dptr(&m->d_emb_row); cm_free_dptr(&m->d_x); cm_free_dptr(&m->d_xn); cm_free_dptr(&m->d_q);
    cm_free_dptr(&m->d_k); cm_free_dptr(&m->d_v); cm_free_dptr(&m->d_ctx); cm_free_dptr(&m->d_rlogits);
    cm_free_dptr(&m->d_eidx); cm_free_dptr(&m->d_ewt); cm_free_dptr(&m->d_h); cm_free_dptr(&m->d_sh_g);
    cm_free_dptr(&m->d_sh_u); cm_free_dptr(&m->d_logits); cm_free_dptr(&m->d_argmax);
    cm_free_dptr(&m->d_px); cm_free_dptr(&m->d_pxn); cm_free_dptr(&m->d_pq); cm_free_dptr(&m->d_pk);
    cm_free_dptr(&m->d_pv); cm_free_dptr(&m->d_pctx); cm_free_dptr(&m->d_pg); cm_free_dptr(&m->d_pu);
    cm_free_dptr(&m->d_mla_q); cm_free_dptr(&m->d_mla_kv);
    cm_free_dptr(&m->d_mla_qabs); cm_free_dptr(&m->d_mla_olat);
    cm_free_dptr(&m->d_prlogits); cm_free_dptr(&m->d_moe_x); cm_free_dptr(&m->d_moe_y); cm_free_dptr(&m->d_moe_idx); cm_free_dptr(&m->d_moe_wt);
    if (m->h_prlogits) { g_cu.cuMemFreeHost(m->h_prlogits); m->h_prlogits = NULL; }
    if (m->h_moe_idx)  { g_cu.cuMemFreeHost(m->h_moe_idx);  m->h_moe_idx = NULL; }
    if (m->h_moe_wt)   { g_cu.cuMemFreeHost(m->h_moe_wt);   m->h_moe_wt = NULL; }
    cm_free_dptr(&m->mtp_enorm.dptr); cm_free_dptr(&m->mtp_hnorm.dptr); cm_free_dptr(&m->mtp_eh_proj.dptr);
    cm_free_dptr(&m->d_last_hidden); cm_free_dptr(&m->d_mtp_cat);
    if (m->mtp_layer) {
        CmLayer* ml = m->mtp_layer;
        cm_free_dptr(&ml->k_cache_base); cm_free_dptr(&ml->v_cache_base);
        cm_free_dptr(&ml->attn_norm.dptr); cm_free_dptr(&ml->ffn_norm.dptr);
        cm_free_dptr(&ml->wq.dptr); cm_free_dptr(&ml->wk.dptr); cm_free_dptr(&ml->wv.dptr); cm_free_dptr(&ml->wo.dptr);
        cm_free_dptr(&ml->wq_a.dptr); cm_free_dptr(&ml->q_a_norm.dptr); cm_free_dptr(&ml->wq_b.dptr);
        cm_free_dptr(&ml->wkv_a.dptr); cm_free_dptr(&ml->kv_a_norm.dptr); cm_free_dptr(&ml->wkv_b.dptr);
        cm_free_dptr(&ml->ffn_gate.dptr); cm_free_dptr(&ml->ffn_up.dptr); cm_free_dptr(&ml->ffn_down.dptr);
        cm_free_dptr(&ml->sh_gate.dptr); cm_free_dptr(&ml->sh_up.dptr); cm_free_dptr(&ml->sh_down.dptr);
        cm_free_dptr(&ml->router.dptr); cm_free_dptr(&ml->exp_bias.dptr); cm_free_dptr(&ml->cache_dptr);
        cm_free_dptr(&ml->mla_wuk_f16); cm_free_dptr(&ml->mla_wuv_f16);
        free(m->mtp_layer); m->mtp_layer = NULL; m->has_mtp_layer = 0;
    }
    for (int lid = 1; lid < CM_MAX_LORAS; lid++) if (m->loras[lid].active) tn_cumoe_lora_free(handle, lid);
    cm_free_dptr(&m->d_lora_tmp);
    cm_free_dptr(&m->d_argmax_batch);
    if (m->h_argmax_batch) { g_cu.cuMemFreeHost(m->h_argmax_batch); m->h_argmax_batch = NULL; }
    free(m->pa_refcnt); m->pa_refcnt = NULL;
    cm_free_dptr(&m->d_pa_bt); cm_free_dptr(&m->d_cb_pos); cm_free_dptr(&m->d_cb_slot); cm_free_dptr(&m->d_batch_logits);
    m->batch_logits_cap = 0;
    cm_free_dptr(&m->d_tree_depth); cm_free_dptr(&m->d_tree_mask);
    m->mla_buf_cap = 0;
    if (m->h_pxn) { g_cu.cuMemFreeHost(m->h_pxn); m->h_pxn = NULL; }
    if (m->h_pcpu_accum) { g_cu.cuMemFreeHost(m->h_pcpu_accum); m->h_pcpu_accum = NULL; }
    free(m->h_logits); m->h_logits = NULL;
    m->prefill_cap = 0;
    if (m->ev_dma_done) { g_cu.cuEventDestroy(m->ev_dma_done); m->ev_dma_done = NULL; }
    if (m->compute_stream) { g_cu.cuStreamDestroy(m->compute_stream); m->compute_stream = NULL; }
    if (m->dma_stream) { g_cu.cuStreamDestroy(m->dma_stream); m->dma_stream = NULL; }
    if (m->vocab_tokens) {
        for (int i = 0; i < m->vocab_size; i++) free(m->vocab_tokens[i]);
        free(m->vocab_tokens); m->vocab_tokens = NULL;
    }
    if (m->bpe_merges) {
        for (int i = 0; i < m->n_bpe_merges; i++) free(m->bpe_merges[i]);
        free(m->bpe_merges); m->bpe_merges = NULL; m->n_bpe_merges = 0;
    }
    free(m->bpe_tok_ht); m->bpe_tok_ht = NULL; m->bpe_tok_cap = 0;
    free(m->bpe_merge_ht); m->bpe_merge_ht = NULL; m->bpe_merge_cap = 0;
    for (int s = 0; s < m->n_shards; s++) {
        if (m->mmap_ptrs[s]) { munmap(m->mmap_ptrs[s], m->shard_sizes[s]); m->mmap_ptrs[s] = NULL; }
        if (m->fds[s] >= 0) { close(m->fds[s]); m->fds[s] = -1; }
    }
    m->active = 0;
    return 1;
}
