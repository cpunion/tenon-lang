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
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>
#endif

#define CM_MAX_MODELS   8
#define CM_MAX_SHARDS   16
#define CM_MAX_LAYERS   128
#define CM_MAX_EXPERTS  256
#define CM_MAX_TOPK     16
#define CM_MAX_SEQ      1024
#define CM_STAGE_SLOTS  16
#define CM_IO_WORKERS   8
#define CM_MAX_CPU_THR  64

#define CM_GGML_F32   0
#define CM_GGML_F16   1
#define CM_GGML_Q4_0  2
#define CM_GGML_Q4_1  3
#define CM_GGML_Q8_0  8
#define CM_GGML_Q4_K  12
#define CM_GGML_Q5_K  13
#define CM_GGML_Q6_K  14

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
typedef CUresult (*fn_cuCtxSetCurrent)(CUcontext);
typedef CUresult (*fn_cuMemAlloc_v2)(CUdeviceptr*, size_t);
typedef CUresult (*fn_cuMemFree_v2)(CUdeviceptr);
typedef CUresult (*fn_cuMemHostAlloc)(void**, size_t, unsigned int);
typedef CUresult (*fn_cuMemFreeHost)(void*);
typedef CUresult (*fn_cuMemcpyHtoD_v2)(CUdeviceptr, const void*, size_t);
typedef CUresult (*fn_cuMemcpyDtoH_v2)(void*, CUdeviceptr, size_t);
typedef CUresult (*fn_cuMemcpyHtoDAsync_v2)(CUdeviceptr, const void*, size_t, CUstream);
typedef CUresult (*fn_cuMemcpyDtoHAsync_v2)(void*, CUdeviceptr, size_t, CUstream);
typedef CUresult (*fn_cuMemsetD8_v2)(CUdeviceptr, unsigned char, size_t);
typedef CUresult (*fn_cuStreamCreate)(CUstream*, unsigned int);
typedef CUresult (*fn_cuStreamDestroy_v2)(CUstream);
typedef CUresult (*fn_cuStreamSynchronize)(CUstream);
typedef CUresult (*fn_cuStreamWaitEvent)(CUstream, CUevent, unsigned int);
typedef CUresult (*fn_cuEventCreate)(CUevent*, unsigned int);
typedef CUresult (*fn_cuEventDestroy_v2)(CUevent);
typedef CUresult (*fn_cuEventRecord)(CUevent, CUstream);
typedef CUresult (*fn_cuEventSynchronize)(CUevent);
typedef CUresult (*fn_cuEventQuery)(CUevent);
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

typedef struct {
    int ready;
    void* libcuda;
    void* libnvrtc;
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    char dev_name[128];
    int cc_major, cc_minor;
    size_t vram_bytes;

    fn_cuInit cuInit;
    fn_cuDeviceGetCount cuDeviceGetCount;
    fn_cuDeviceGet cuDeviceGet;
    fn_cuDeviceGetName cuDeviceGetName;
    fn_cuDeviceTotalMem_v2 cuDeviceTotalMem;
    fn_cuDeviceGetAttribute cuDeviceGetAttribute;
    fn_cuCtxCreate_v2 cuCtxCreate;
    fn_cuCtxSetCurrent cuCtxSetCurrent;
    fn_cuMemAlloc_v2 cuMemAlloc;
    fn_cuMemFree_v2 cuMemFree;
    fn_cuMemHostAlloc cuMemHostAlloc;
    fn_cuMemFreeHost cuMemFreeHost;
    fn_cuMemcpyHtoD_v2 cuMemcpyHtoD;
    fn_cuMemcpyDtoH_v2 cuMemcpyDtoH;
    fn_cuMemcpyHtoDAsync_v2 cuMemcpyHtoDAsync;
    fn_cuMemcpyDtoHAsync_v2 cuMemcpyDtoHAsync;
    fn_cuMemsetD8_v2 cuMemsetD8;
    fn_cuStreamCreate cuStreamCreate;
    fn_cuStreamDestroy_v2 cuStreamDestroy;
    fn_cuStreamSynchronize cuStreamSynchronize;
    fn_cuStreamWaitEvent cuStreamWaitEvent;
    fn_cuEventCreate cuEventCreate;
    fn_cuEventDestroy_v2 cuEventDestroy;
    fn_cuEventRecord cuEventRecord;
    fn_cuEventSynchronize cuEventSynchronize;
    fn_cuEventQuery cuEventQuery;
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

    CUfunction k_embed, k_rmsnorm, k_add_bias, k_swiglu_mul;
    CUfunction k_gemv_q4, k_gemv_q4_1, k_gemv_q8, k_gemv_q4k, k_gemv_q5k, k_gemv_q6k, k_gemv_f16, k_gemv_f32;
    CUfunction k_dense_gu_q4, k_rope_kv, k_flash_dec, k_router;
    CUfunction k_moe_gu_q4, k_moe_down_q4, k_moe_down_q4_1;
    CUfunction k_moe_slot_gu_q4, k_moe_slot_down_q4, k_moe_slot_down_q4_1;
    CUfunction k_argmax;
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
        case CM_GGML_F32:  return n_elems * 4ull;
        case CM_GGML_F16:  return n_elems * 2ull;
        case CM_GGML_Q4_0: return (n_elems / 32ull) * 18ull;
        case CM_GGML_Q4_1: return (n_elems / 32ull) * 20ull;
        case CM_GGML_Q8_0: return (n_elems / 32ull) * 34ull;
        case CM_GGML_Q4_K: return (n_elems / 256ull) * 144ull;
        case CM_GGML_Q5_K: return (n_elems / 256ull) * 176ull;
        case CM_GGML_Q6_K: return (n_elems / 256ull) * 210ull;
        default:           return n_elems * 2ull;
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
"__device__ __forceinline__ float dot_q4_k_block(const uint8_t* __restrict__ blk, const float* __restrict__ x) {\n"
"    float d = load_f16_le(blk), dmin = load_f16_le(blk + 2u);\n"
"    const uint8_t* sc = blk + 4u; const uint8_t* qs = blk + 16u;\n"
"    float sum = 0.0f; uint32_t is = 0u;\n"
"    for (uint32_t j = 0; j < 256u; j += 64u) {\n"
"        uint8_t sc0, m0, sc1, m1;\n"
"        if (is < 4u) { sc0 = sc[is] & 63u; m0 = sc[is + 4u] & 63u; sc1 = sc[is + 1u] & 63u; m1 = sc[is + 5u] & 63u; }\n"
"        else { sc0 = (sc[is + 4u] & 0xFu) | ((sc[is - 4u] >> 6) << 4); m0 = (sc[is + 4u] >> 4) | ((sc[is] >> 6) << 4);\n"
"               sc1 = (sc[is + 5u] & 0xFu) | ((sc[is - 3u] >> 6) << 4); m1 = (sc[is + 5u] >> 4) | ((sc[is + 1u] >> 6) << 4); }\n"
"        float d1 = d * (float)sc0, dm1 = dmin * (float)m0, d2 = d * (float)sc1, dm2 = dmin * (float)m1;\n"
"        const uint8_t* q = qs + (j >> 1);\n"
"        for (uint32_t l = 0; l < 32u; l++) {\n"
"            uint8_t qv = q[l];\n"
"            sum = fmaf(d1 * (float)(qv & 0xFu) - dm1, x[j + l], sum);\n"
"            sum = fmaf(d2 * (float)(qv >> 4)   - dm2, x[j + 32u + l], sum);\n"
"        }\n"
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
"    for (uint32_t n = 0; n < 256u; n += 128u) {\n"
"        const uint8_t* qlp = ql + (n >> 1); const uint8_t* qhp = qh + (n >> 2);\n"
"        const int8_t* scp = sc + (n >> 4); const float* xp = x + n;\n"
"        float s0 = (float)scp[is + 0], s2 = (float)scp[is + 2], s4 = (float)scp[is + 4], s6 = (float)scp[is + 6];\n"
"        float p0 = 0.0f, p2 = 0.0f, p4 = 0.0f, p6 = 0.0f;\n"
"        for (uint32_t l = l0; l < l0 + 8u; l++) {\n"
"            uint8_t ql_a = qlp[l], ql_b = qlp[l + 32u], qhv = qhp[l];\n"
"            int q1 = (int)((ql_a & 0xFu) | (((qhv >> 0) & 3u) << 4)) - 32;\n"
"            int q2 = (int)((ql_b & 0xFu) | (((qhv >> 2) & 3u) << 4)) - 32;\n"
"            int q3 = (int)((ql_a >> 4)   | (((qhv >> 4) & 3u) << 4)) - 32;\n"
"            int q4 = (int)((ql_b >> 4)   | (((qhv >> 6) & 3u) << 4)) - 32;\n"
"            p0 = fmaf((float)q1, xp[l + 0u],  p0); p2 = fmaf((float)q2, xp[l + 32u], p2);\n"
"            p4 = fmaf((float)q3, xp[l + 64u], p4); p6 = fmaf((float)q4, xp[l + 96u], p6);\n"
"        }\n"
"        sum += s0 * p0 + s2 * p2 + s4 * p4 + s6 * p6;\n"
"    }\n"
"    return d * sum;\n"
"}\n"
"\n"
"extern \"C\" __global__ void cm_embed_lookup(const uint8_t* emb, float* out, uint32_t tok_id, uint32_t d_model, uint32_t ggml_type) {\n"
"    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (gid >= d_model) return;\n"
"    if (ggml_type == 0u) out[gid] = ((const float*)emb)[(uint64_t)tok_id * d_model + gid];\n"
"    else if (ggml_type == 1u) out[gid] = __half2float(((const half*)emb)[(uint64_t)tok_id * d_model + gid]);\n"
"    else if (ggml_type == 2u) {\n"
"        uint64_t rb = (uint64_t)(d_model / 32u) * 18ull;\n"
"        uint32_t blk = gid >> 5, sub = gid & 31u;\n"
"        const uint8_t* bp = emb + (uint64_t)tok_id * rb + (uint64_t)blk * 18ull;\n"
"        float d = load_f16_le(bp); uint8_t bv = bp[2u + (sub & 15u)];\n"
"        int q = (sub < 16u) ? (int)(bv & 0xFu) - 8 : (int)(bv >> 4) - 8;\n"
"        out[gid] = d * (float)q;\n"
"    } else if (ggml_type == 3u) {\n"
"        uint64_t rb = (uint64_t)(d_model / 32u) * 20ull;\n"
"        uint32_t blk = gid >> 5, sub = gid & 31u;\n"
"        const uint8_t* bp = emb + (uint64_t)tok_id * rb + (uint64_t)blk * 20ull;\n"
"        float d = load_f16_le(bp), m = load_f16_le(bp + 2u); uint8_t bv = bp[4u + (sub & 15u)];\n"
"        uint32_t q = (sub < 16u) ? (bv & 0xFu) : (bv >> 4);\n"
"        out[gid] = fmaf(d, (float)q, m);\n"
"    } else if (ggml_type == 8u) {\n"
"        uint64_t rb = (uint64_t)(d_model / 32u) * 34ull;\n"
"        uint32_t blk = gid >> 5, sub = gid & 31u;\n"
"        const uint8_t* bp = emb + (uint64_t)tok_id * rb + (uint64_t)blk * 34ull;\n"
"        out[gid] = load_f16_le(bp) * (float)((const int8_t*)(bp + 2u))[sub];\n"
"    } else if (ggml_type == 14u) {\n"
"        uint64_t rb = (uint64_t)(d_model / 256u) * 210ull;\n"
"        uint32_t blk = gid >> 8, rem = gid & 255u;\n"
"        const uint8_t* bp = emb + (uint64_t)tok_id * rb + (uint64_t)blk * 210ull;\n"
"        uint32_t n = (rem >= 128u) ? 128u : 0u; uint32_t r = rem - n;\n"
"        uint32_t qpart = r >> 5, l = r & 31u;\n"
"        const uint8_t* qlp = bp + (n >> 1); const uint8_t* qhp = bp + 128u + (n >> 2);\n"
"        const int8_t* scp = (const int8_t*)(bp + 192u) + (n >> 4); float d = load_f16_le(bp + 208u);\n"
"        uint32_t is = l >> 4;\n"
"        uint8_t lo = (qpart == 0u) ? (qlp[l] & 0xFu) : (qpart == 1u) ? (qlp[l + 32u] & 0xFu) : (qpart == 2u) ? (qlp[l] >> 4) : (qlp[l + 32u] >> 4);\n"
"        uint8_t hi = (qhp[l] >> (qpart * 2u)) & 3u;\n"
"        out[gid] = d * (float)scp[is + qpart * 2u] * (float)((int)(lo | (hi << 4)) - 32);\n"
"    }\n"
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
"    uint32_t row0 = (blockIdx.x * 4u + wid) * 4u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 18ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb;\n"
"    const uint8_t* wp1 = (row0 + 1u < out_dim) ? (wp0 + rb) : wp0;\n"
"    const uint8_t* wp2 = (row0 + 2u < out_dim) ? (wp0 + rb * 2ull) : wp0;\n"
"    const uint8_t* wp3 = (row0 + 3u < out_dim) ? (wp0 + rb * 3ull) : wp0;\n"
"    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        float4 xv_lo[4], xv_hi[4]; float xs = load_x32_q4(x + (b << 5), xv_lo, xv_hi);\n"
"        uint64_t boff = (uint64_t)b * 18ull;\n"
"        a0 += dot_q4_0_fast(wp0 + boff, xv_lo, xv_hi, xs); a1 += dot_q4_0_fast(wp1 + boff, xv_lo, xv_hi, xs);\n"
"        a2 += dot_q4_0_fast(wp2 + boff, xv_lo, xv_hi, xs); a3 += dot_q4_0_fast(wp3 + boff, xv_lo, xv_hi, xs);\n"
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
"extern \"C\" __global__ void cm_gemv_q4_1(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u;\n"
"    uint32_t row0 = (blockIdx.x * 4u + wid) * 4u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 20ull;\n"
"    const uint8_t* wp0 = w + (uint64_t)row0 * rb;\n"
"    const uint8_t* wp1 = (row0 + 1u < out_dim) ? (wp0 + rb) : wp0;\n"
"    const uint8_t* wp2 = (row0 + 2u < out_dim) ? (wp0 + rb * 2ull) : wp0;\n"
"    const uint8_t* wp3 = (row0 + 3u < out_dim) ? (wp0 + rb * 3ull) : wp0;\n"
"    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        float4 xv_lo[4], xv_hi[4]; float sx = load_x32_q4(x + (b << 5), xv_lo, xv_hi);\n"
"        uint64_t boff = (uint64_t)b * 20ull;\n"
"        a0 += dot_q4_1_fast(wp0 + boff, xv_lo, xv_hi, sx); a1 += dot_q4_1_fast(wp1 + boff, xv_lo, xv_hi, sx);\n"
"        a2 += dot_q4_1_fast(wp2 + boff, xv_lo, xv_hi, sx); a3 += dot_q4_1_fast(wp3 + boff, xv_lo, xv_hi, sx);\n"
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
"extern \"C\" __global__ void cm_gemv_q4k(const uint8_t* w, const float* x, float* y, uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {\n"
"    uint32_t wid = threadIdx.x >> 5, lane = threadIdx.x & 31u, row = blockIdx.x * 4u + wid;\n"
"    if (row >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 8; const uint8_t* wp = w + (uint64_t)row * (uint64_t)nb * 144ull; float acc = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) acc += dot_q4_k_block(wp + (uint64_t)b * 144ull, x + (b << 8));\n"
"    acc = warp_sum(acc); if (lane == 0u) y[row] = add_res ? (y[row] + acc) : acc;\n"
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
"    uint32_t row0 = (blockIdx.x * 4u + wid) * 2u, row1 = row0 + 1u;\n"
"    if (row0 >= out_dim) return;\n"
"    uint32_t nb = in_dim >> 5; uint64_t rb = (uint64_t)nb * 18ull;\n"
"    const uint8_t* g0 = wg + (uint64_t)row0 * rb; const uint8_t* g1 = (row1 < out_dim) ? (g0 + rb) : g0;\n"
"    const uint8_t* u0 = wu + (uint64_t)row0 * rb; const uint8_t* u1 = (row1 < out_dim) ? (u0 + rb) : u0;\n"
"    float ag0 = 0.0f, ag1 = 0.0f, au0 = 0.0f, au1 = 0.0f;\n"
"    for (uint32_t b = lane; b < nb; b += 32u) {\n"
"        float4 xv_lo[4], xv_hi[4]; float xs = load_x32_q4(xn + (b << 5), xv_lo, xv_hi);\n"
"        uint64_t boff = (uint64_t)b * 18ull;\n"
"        ag0 += dot_q4_0_fast(g0 + boff, xv_lo, xv_hi, xs); au0 += dot_q4_0_fast(u0 + boff, xv_lo, xv_hi, xs);\n"
"        ag1 += dot_q4_0_fast(g1 + boff, xv_lo, xv_hi, xs); au1 += dot_q4_0_fast(u1 + boff, xv_lo, xv_hi, xs);\n"
"    }\n"
"    ag0 = warp_sum(ag0); au0 = warp_sum(au0); ag1 = warp_sum(ag1); au1 = warp_sum(au1);\n"
"    if (lane == 0u) {\n"
"        h_out[row0] = (ag0 / (1.0f + expf(-ag0))) * au0;\n"
"        if (row1 < out_dim) h_out[row1] = (ag1 / (1.0f + expf(-ag1))) * au1;\n"
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
"                                           uint32_t n_heads, uint32_t n_kv, uint32_t d_head, uint32_t kv_len) {\n"
"    uint32_t h = blockIdx.x; if (h >= n_heads) return;\n"
"    uint32_t lid = threadIdx.x, wid = lid >> 5, lane = lid & 31u;\n"
"    uint32_t grp = (n_heads / n_kv > 0u) ? (n_heads / n_kv) : 1u, kvh = h / grp, kv_stride = n_kv * d_head;\n"
"    float scale = rsqrtf((float)d_head);\n"
"    __shared__ float s_q[256], s_m[4], s_l[4], s_o[4][256];\n"
"    for (uint32_t d = lid; d < d_head; d += 128u) s_q[d] = q[h * d_head + d];\n"
"    __syncthreads();\n"
"    float m_run = -1e30f, l_run = 0.0f, o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};\n"
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
"}\n";

static void* cm_sym(void* lib, const char* a, const char* b) {
    if (!lib) return NULL;
    void* p = dlsym(lib, a);
    if (!p && b) p = dlsym(lib, b);
    return p;
}

static char* cm_compile_ptx(const char* src) {
    // Prefer nvcc --ptx (fastest PTX codegen on RunPod CUDA containers), fallback to libnvrtc.so!
    char cu_path[128], ptx_path[128], cmd[512];
    snprintf(cu_path, sizeof(cu_path), "/tmp/tenon_cumoe_%d.cu", (int)getpid());
    snprintf(ptx_path, sizeof(ptx_path), "/tmp/tenon_cumoe_%d.ptx", (int)getpid());
    FILE* f = fopen(cu_path, "wb");
    if (f) {
        fwrite(src, 1, strlen(src), f);
        fclose(f);
        snprintf(cmd, sizeof(cmd), "nvcc --ptx -O3 -use_fast_math -arch=sm_%d%d \"%s\" -o \"%s\" 2>/tmp/tenon_cumoe_nvcc.err",
                 g_cu.cc_major, g_cu.cc_minor, cu_path, ptx_path);
        if (system(cmd) == 0) {
            FILE* pf = fopen(ptx_path, "rb");
            if (pf) {
                fseek(pf, 0, SEEK_END);
                long sz = ftell(pf);
                fseek(pf, 0, SEEK_SET);
                char* ptx = (char*)malloc((size_t)sz + 1);
                if (ptx && sz > 0) {
                    size_t nr = fread(ptx, 1, (size_t)sz, pf);
                    ptx[nr] = 0;
                    fclose(pf);
                    unlink(cu_path); unlink(ptx_path);
                    return ptx;
                }
                free(ptx);
                fclose(pf);
            }
        }
        unlink(cu_path); unlink(ptx_path);
    }
    if (!g_cu.nvrtcCreateProgram) return NULL;
    nvrtcProgram prog = NULL;
    if (g_cu.nvrtcCreateProgram(&prog, src, "tenon_cumoe.cu", 0, NULL, NULL) != NVRTC_SUCCESS) return NULL;
    char arch_opt[32];
    snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d", g_cu.cc_major, g_cu.cc_minor);
    const char* opts[2] = { arch_opt, "--use_fast_math" };
    if (g_cu.nvrtcCompileProgram(prog, 2, opts) != NVRTC_SUCCESS) {
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
    g_cu.cuCtxSetCurrent      = (fn_cuCtxSetCurrent)cm_sym(g_cu.libcuda, "cuCtxSetCurrent", NULL);
    g_cu.cuMemAlloc           = (fn_cuMemAlloc_v2)cm_sym(g_cu.libcuda, "cuMemAlloc_v2", "cuMemAlloc");
    g_cu.cuMemFree            = (fn_cuMemFree_v2)cm_sym(g_cu.libcuda, "cuMemFree_v2", "cuMemFree");
    g_cu.cuMemHostAlloc       = (fn_cuMemHostAlloc)cm_sym(g_cu.libcuda, "cuMemHostAlloc", NULL);
    g_cu.cuMemFreeHost        = (fn_cuMemFreeHost)cm_sym(g_cu.libcuda, "cuMemFreeHost", NULL);
    g_cu.cuMemcpyHtoD         = (fn_cuMemcpyHtoD_v2)cm_sym(g_cu.libcuda, "cuMemcpyHtoD_v2", "cuMemcpyHtoD");
    g_cu.cuMemcpyDtoH         = (fn_cuMemcpyDtoH_v2)cm_sym(g_cu.libcuda, "cuMemcpyDtoH_v2", "cuMemcpyDtoH");
    g_cu.cuMemcpyHtoDAsync    = (fn_cuMemcpyHtoDAsync_v2)cm_sym(g_cu.libcuda, "cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync");
    g_cu.cuMemcpyDtoHAsync    = (fn_cuMemcpyDtoHAsync_v2)cm_sym(g_cu.libcuda, "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync");
    g_cu.cuMemsetD8           = (fn_cuMemsetD8_v2)cm_sym(g_cu.libcuda, "cuMemsetD8_v2", "cuMemsetD8");
    g_cu.cuStreamCreate       = (fn_cuStreamCreate)cm_sym(g_cu.libcuda, "cuStreamCreate", NULL);
    g_cu.cuStreamDestroy      = (fn_cuStreamDestroy_v2)cm_sym(g_cu.libcuda, "cuStreamDestroy_v2", "cuStreamDestroy");
    g_cu.cuStreamSynchronize  = (fn_cuStreamSynchronize)cm_sym(g_cu.libcuda, "cuStreamSynchronize", NULL);
    g_cu.cuStreamWaitEvent    = (fn_cuStreamWaitEvent)cm_sym(g_cu.libcuda, "cuStreamWaitEvent", NULL);
    g_cu.cuEventCreate        = (fn_cuEventCreate)cm_sym(g_cu.libcuda, "cuEventCreate", NULL);
    g_cu.cuEventDestroy       = (fn_cuEventDestroy_v2)cm_sym(g_cu.libcuda, "cuEventDestroy_v2", "cuEventDestroy");
    g_cu.cuEventRecord        = (fn_cuEventRecord)cm_sym(g_cu.libcuda, "cuEventRecord", NULL);
    g_cu.cuEventSynchronize   = (fn_cuEventSynchronize)cm_sym(g_cu.libcuda, "cuEventSynchronize", NULL);
    g_cu.cuEventQuery         = (fn_cuEventQuery)cm_sym(g_cu.libcuda, "cuEventQuery", NULL);
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
    if (g_cu.cuDeviceGet(&g_cu.dev, 0) != CUDA_SUCCESS) return 0;
    g_cu.cuDeviceGetName(g_cu.dev_name, sizeof(g_cu.dev_name), g_cu.dev);
    g_cu.cuDeviceTotalMem(&g_cu.vram_bytes, g_cu.dev);
    g_cu.cuDeviceGetAttribute(&g_cu.cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, g_cu.dev);
    g_cu.cuDeviceGetAttribute(&g_cu.cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, g_cu.dev);
    if (g_cu.cuCtxCreate(&g_cu.ctx, 0, g_cu.dev) != CUDA_SUCCESS) return 0;

    char* ptx = cm_compile_ptx(kCuMoeSrc);
    if (!ptx) { fprintf(stderr, "[tenon-cuda-stream] failed to compile CUDA kernels\n"); return 0; }
    CUresult mr = g_cu.cuModuleLoadData(&g_cu.mod, ptx);
    free(ptx);
    if (mr != CUDA_SUCCESS) { fprintf(stderr, "[tenon-cuda-stream] cuModuleLoadData failed (%d)\n", mr); return 0; }

    #define LOAD_K(dst, nm) if (g_cu.cuModuleGetFunction(&g_cu.dst, g_cu.mod, nm) != CUDA_SUCCESS) { fprintf(stderr, "[tenon-cuda-stream] missing %s\n", nm); return 0; }
    LOAD_K(k_embed, "cm_embed_lookup");
    LOAD_K(k_rmsnorm, "cm_rmsnorm");
    LOAD_K(k_add_bias, "cm_add_bias");
    LOAD_K(k_swiglu_mul, "cm_swiglu_mul");
    LOAD_K(k_gemv_q4, "cm_gemv_q4");
    LOAD_K(k_gemv_q4_1, "cm_gemv_q4_1");
    LOAD_K(k_gemv_q8, "cm_gemv_q8");
    LOAD_K(k_gemv_q4k, "cm_gemv_q4k");
    LOAD_K(k_gemv_q5k, "cm_gemv_q5k");
    LOAD_K(k_gemv_q6k, "cm_gemv_q6k");
    LOAD_K(k_gemv_f16, "cm_gemv_f16");
    LOAD_K(k_gemv_f32, "cm_gemv_f32");
    LOAD_K(k_dense_gu_q4, "cm_dense_gate_up_q4");
    LOAD_K(k_rope_kv, "cm_rope_and_kv");
    LOAD_K(k_flash_dec, "cm_flash_decode");
    LOAD_K(k_moe_slot_gu_q4, "cm_slot_gate_up_q4");
    LOAD_K(k_moe_slot_down_q4, "cm_slot_down_accum_q4");
    LOAD_K(k_moe_slot_down_q4_1, "cm_slot_down_accum_q4_1");
    LOAD_K(k_argmax, "cm_argmax_f32");
    #undef LOAD_K

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
    CUdeviceptr k_cache, v_cache;
} CmLayer;

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
    float expert_weights_scale, rope_theta, rms_eps;
    int cache_slots_per_layer;
    char** vocab_tokens;

    CmWeightRef tok_embd, out_norm, lm_head, rope_freqs;
    int has_rope_freqs;
    CmLayer layers[CM_MAX_LAYERS];

    // CUDA streams & events
    CUstream compute_stream;
    CUstream dma_stream;
    CUevent ev_dma_done;

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
    // Pre-quantized Q8_0 activation buffers (for 3x faster AVX2 vpmaddubsw INT8 SIMD!)
    int8_t*  cpu_xq8;
    float*   cpu_xd;
    float*   cpu_xsum;
    int8_t*  cpu_hq8;
    float*   cpu_hd;
    float*   cpu_hsum;

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
            else if (strstr(k, ".expert_count")) m->n_experts = (int)gr_int(&gr, vt);
            else if (strstr(k, ".expert_used_count")) m->top_k = (int)gr_int(&gr, vt);
            else if (strstr(k, ".expert_gating_func")) m->gating_func = (int)gr_int(&gr, vt);
            else if (strstr(k, ".expert_weights_norm")) m->expert_weights_norm = (int)gr_int(&gr, vt);
            else if (strstr(k, ".expert_weights_scale") && vt == CM_GGUF_F32) m->expert_weights_scale = gr_f32(&gr);
            else if (strstr(k, ".rope.freq_base") && vt == CM_GGUF_F32) m->rope_theta = gr_f32(&gr);
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
static void launch_gemv_raw(CUstream st, CUdeviceptr w, uint32_t gtype, CUdeviceptr x, CUdeviceptr y,
                            uint32_t out_dim, uint32_t in_dim, uint32_t add_res) {
    CUfunction fn = g_cu.k_gemv_q4;
    uint32_t rptg = 4;
    switch (gtype) {
        case CM_GGML_Q4_0: fn = g_cu.k_gemv_q4;   rptg = 16; break;
        case CM_GGML_Q4_1: fn = g_cu.k_gemv_q4_1; rptg = 16; break;
        case CM_GGML_Q8_0: fn = g_cu.k_gemv_q8;   rptg = 8;  break;
        case CM_GGML_Q4_K: fn = g_cu.k_gemv_q4k;  rptg = 4;  break;
        case CM_GGML_Q5_K: fn = g_cu.k_gemv_q5k;  rptg = 8;  break;
        case CM_GGML_Q6_K: fn = g_cu.k_gemv_q6k;  rptg = 8;  break;
        case CM_GGML_F16:  fn = g_cu.k_gemv_f16;  rptg = 4;  break;
        default:           fn = g_cu.k_gemv_f32;  rptg = 4;  break;
    }
    void* args[] = { &w, &x, &y, &out_dim, &in_dim, &add_res };
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

static void launch_dense_ffn_raw(CmModel* m, CUstream st,
                                 CUdeviceptr wg, uint32_t gt, CUdeviceptr wu, uint32_t ut,
                                 CUdeviceptr wd, uint32_t dt, uint32_t d_ff) {
    uint32_t dm = (uint32_t)m->d_model;
    if (gt == CM_GGML_Q4_0 && ut == CM_GGML_Q4_0) {
        void* args[] = { &wg, &wu, &m->d_xn, &m->d_sh_g, &d_ff, &dm };
        g_cu.cuLaunchKernel(g_cu.k_dense_gu_q4, (d_ff + 7) / 8, 1, 1, 128, 1, 1, 0, st, args, NULL);
    } else {
        launch_gemv_raw(st, wg, gt, m->d_xn, m->d_sh_g, d_ff, dm, 0);
        launch_gemv_raw(st, wu, ut, m->d_xn, m->d_sh_u, d_ff, dm, 0);
        void* args[] = { &m->d_sh_g, &m->d_sh_u, &d_ff };
        g_cu.cuLaunchKernel(g_cu.k_swiglu_mul, (d_ff + 255) / 256, 1, 1, 256, 1, 1, 0, st, args, NULL);
    }
    launch_gemv_raw(st, wd, dt, m->d_sh_g, m->d_x, dm, d_ff, 1);
}

static void launch_layer_attn(CmModel* m, CUstream st, int L, uint32_t pos, CUdeviceptr ring_base) {
    CmLayer* ly = &m->layers[L];
    uint32_t dm = (uint32_t)m->d_model, qdim = (uint32_t)(m->n_heads * m->d_head), kvdim = (uint32_t)(m->n_kv * m->d_head);
    launch_rmsnorm(st, m->d_x, &ly->attn_norm, m->d_xn, dm, 1, m->rms_eps, 0);
    if (ring_base) {
        launch_gemv_raw(st, ring_base + m->dense_off_wq, ly->wq.ggml_type, m->d_xn, m->d_q, qdim, dm, 0);
        launch_gemv_raw(st, ring_base + m->dense_off_wk, ly->wk.ggml_type, m->d_xn, m->d_k, kvdim, dm, 0);
        launch_gemv_raw(st, ring_base + m->dense_off_wv, ly->wv.ggml_type, m->d_xn, m->d_v, kvdim, dm, 0);
    } else {
        launch_gemv(st, &ly->wq, m->d_xn, m->d_q, qdim, dm, 0);
        launch_gemv(st, &ly->wk, m->d_xn, m->d_k, kvdim, dm, 0);
        launch_gemv(st, &ly->wv, m->d_xn, m->d_v, kvdim, dm, 0);
    }
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
    void* rargs[] = { &m->d_q, &m->d_k, &m->d_v, &ly->k_cache, &ly->v_cache, &rf, &nh, &nkv, &dh, &rd, &pos, &theta, &rn, &hf };
    uint32_t hd2 = dh >> 1;
    g_cu.cuLaunchKernel(g_cu.k_rope_kv, (hd2 + 31) / 32, nh, 1, 32, 1, 1, 0, st, rargs, NULL);

    uint32_t kv_len = pos + 1u;
    void* aargs[] = { &m->d_q, &ly->k_cache, &ly->v_cache, &m->d_ctx, &nh, &nkv, &dh, &kv_len };
    g_cu.cuLaunchKernel(g_cu.k_flash_dec, nh, 1, 1, 128, 1, 1, 0, st, aargs, NULL);

    if (ring_base) launch_gemv_raw(st, ring_base + m->dense_off_wo, ly->wo.ggml_type, m->d_ctx, m->d_x, dm, qdim, 1);
    else launch_gemv(st, &ly->wo, m->d_ctx, m->d_x, dm, qdim, 1);
    launch_rmsnorm(st, m->d_x, &ly->ffn_norm, m->d_xn, dm, 1, m->rms_eps, 0);
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
    void* gu_args[] = { &ly->cache_dptr, &m->d_xn, &m->d_eidx, &m->d_h, &dm, &df, &n_act, &gb, &sb };
    g_cu.cuLaunchKernel(g_cu.k_moe_slot_gu_q4, (df + 7) / 8, n_act, 1, 128, 1, 1, 0, st, gu_args, NULL);
    CUfunction dfn = (ly->down_exps.ggml_type == CM_GGML_Q4_1) ? g_cu.k_moe_slot_down_q4_1 : g_cu.k_moe_slot_down_q4;
    void* dn_args[] = { &ly->cache_dptr, &m->d_h, &m->d_eidx, &m->d_ewt, &m->d_x, &dm, &df, &n_act, &doff, &sb };
    g_cu.cuLaunchKernel(dfn, (dm + 15) / 16, 1, 1, 128, 1, 1, 0, st, dn_args, NULL);
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
#endif

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

static inline void cm_cpu_dot_2rows(const uint8_t* w0, const uint8_t* w1, uint32_t gtype,
                                    const float* x, const int8_t* xq8, const float* xd, const float* xsum,
                                    uint32_t in_dim, float* o0, float* o1) {
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
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

static void* cm_cpu_pool_worker(void* arg) {
    CpuPoolArg* a = (CpuPoolArg*)arg;
    CmModel* m = a->m;
    int tid = a->tid;
    free(a);
    uint64_t seen_gen = 0;
    for (;;) {
        uint64_t g = __atomic_load_n(&m->cpu_gen, __ATOMIC_ACQUIRE);
        if (g == seen_gen && !__atomic_load_n(&m->cpu_stop, __ATOMIC_ACQUIRE)) {
            for (int sp = 0; sp < 40000; sp++) {
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
            // DENSE UNPINNED FFN IN-PLACE AVX2 INT8 `vpmaddubsw` COMPUTE (4-row unrolled -> 180+ GB/s)!
            uint32_t df = (uint32_t)m->d_ff_dense;
            uint32_t gt = ly->ffn_gate.ggml_type, ut = ly->ffn_up.ggml_type, dt = ly->ffn_down.ggml_type;
            uint64_t grb = cm_ggml_type_bytes(gt, dm), urb = cm_ggml_type_bytes(ut, dm), drb = cm_ggml_type_bytes(dt, df);
            const uint8_t* g_base = m->mmap_ptrs[ly->ffn_gate.shard_idx] + ly->ffn_gate.file_off;
            const uint8_t* u_base = m->mmap_ptrs[ly->ffn_up.shard_idx]   + ly->ffn_up.file_off;
            const uint8_t* d_base = m->mmap_ptrs[ly->ffn_down.shard_idx] + ly->ffn_down.file_off;

            uint32_t df_blks = df >> 5;
            uint32_t b0 = (uint32_t)(((uint64_t)df_blks * (uint64_t)tid) / (uint64_t)n_thr);
            uint32_t b1 = (uint32_t)(((uint64_t)df_blks * (uint64_t)(tid + 1)) / (uint64_t)n_thr);
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
                cm_quantize_q8_0(hp, m->cpu_hq8 + (size_t)bi * 32u, &m->cpu_hd[bi], &m->cpu_hsum[bi], 32u);
#endif
            }

            __atomic_fetch_add(&m->cpu_phase1_done, 1, __ATOMIC_ACQ_REL);
            while (__atomic_load_n(&m->cpu_phase1_done, __ATOMIC_ACQUIRE) < n_thr) cm_cpu_pause();

            uint32_t dm_quads = dm >> 2;
            uint32_t d0 = (uint32_t)(((uint64_t)dm_quads * (uint64_t)tid) / (uint64_t)n_thr);
            uint32_t d1 = (uint32_t)(((uint64_t)dm_quads * (uint64_t)(tid + 1)) / (uint64_t)n_thr);
            for (uint32_t dq = d0; dq < d1; dq++) {
                uint32_t r = dq << 2;
                float ov[4];
                cm_cpu_dot_4rows(d_base + (uint64_t)r * drb, drb, dt, m->cpu_h,
                                 m->cpu_hq8, m->cpu_hd, m->cpu_hsum, df, ov);
                m->h_cpu_accum[r + 0] = ov[0]; m->h_cpu_accum[r + 1] = ov[1];
                m->h_cpu_accum[r + 2] = ov[2]; m->h_cpu_accum[r + 3] = ov[3];
            }
        }

        __atomic_fetch_add(&m->cpu_phase2_done, 1, __ATOMIC_ACQ_REL);
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

static void cm_cpu_pool_start(CmModel* m) {
    if (m->cpu_pool_init) return;
    int nt = 32;
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
    for (int t = 0; t < nt; t++) {
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
    for (int t = 0; t < m->cpu_n_thr; t++) pthread_join(m->cpu_threads[t], NULL);
    pthread_mutex_destroy(&m->cpu_mu);
    pthread_cond_destroy(&m->cpu_cv_go);
    m->cpu_pool_init = 0;
}

static void cm_cpu_pool_run(CmModel* m, int mode, int L, int n_misses, const int* eids, const float* ws) {
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    cm_quantize_q8_0(m->h_xn, m->cpu_xq8, m->cpu_xd, m->cpu_xsum, (uint32_t)m->d_model);
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
    int nt = m->cpu_n_thr;
    while (__atomic_load_n(&m->cpu_phase2_done, __ATOMIC_ACQUIRE) < nt) cm_cpu_pause();
}

static void cm_cpu_router_topk(const float* logits, const float* exp_bias, int n_exp, int top_k,
                               int gating_func, int norm_wt, float w_scale, int* out_e, float* out_w) {
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
    if (m->rope_dim <= 0) m->rope_dim = m->d_head;
    if (m->d_ff_exp <= 0) m->d_ff_exp = m->d_ff_dense;
    if (m->d_ff_dense <= 0) m->d_ff_dense = m->d_ff_exp;
    m->rope_neox = (strncmp(m->arch, "qwen", 4) == 0 || strstr(m->arch, "neox") != NULL) ? 1 : 0;
    m->dense_model = (m->n_experts <= 0) ? 1 : 0;

    if (m->dense_model) {
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

    for (uint64_t i = 0; i < tcount; i++) {
        const CmTensorInfo* ti = &tlist[i];
        if (strcmp(ti->name, "token_embd.weight") == 0) {
            if (m->vocab_size <= 0) m->vocab_size = (int)ti->ne[1];
            if (m->resident_mode) cm_load_pinned_vram(m, &m->tok_embd, ti);
            else cm_record_meta(&m->tok_embd, ti);
        } else if (strcmp(ti->name, "output_norm.weight") == 0) cm_load_pinned_vram(m, &m->out_norm, ti);
        else if (strcmp(ti->name, "output.weight") == 0) cm_load_pinned_vram(m, &m->lm_head, ti);
        else if (strcmp(ti->name, "rope_freqs.weight") == 0) { cm_load_pinned_vram(m, &m->rope_freqs, ti); m->has_rope_freqs = 1; }
        else if (strncmp(ti->name, "blk.", 4) == 0) {
            int L = atoi(ti->name + 4);
            const char* dot = strchr(ti->name + 4, '.');
            if (L < 0 || L >= m->n_layers || !dot) continue;
            const char* sub = dot + 1;
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

    g_cu.cuStreamCreate(&m->compute_stream, CU_STREAM_NON_BLOCKING);
    g_cu.cuStreamCreate(&m->dma_stream, CU_STREAM_NON_BLOCKING);
    g_cu.cuEventCreate(&m->ev_dma_done, CU_EVENT_DISABLE_TIMING);

    size_t kv_bytes = (size_t)CM_MAX_SEQ * (size_t)(m->n_kv * m->d_head) * 2;
    uint64_t max_wq = 0, max_wk = 0, max_wv = 0, max_wo = 0, max_fg = 0, max_fu = 0, max_fd = 0, max_exp_stride = 0;
    for (int L = 0; L < m->n_layers; L++) {
        CmLayer* ly = &m->layers[L];
        g_cu.cuMemAlloc(&ly->k_cache, kv_bytes);
        g_cu.cuMemAlloc(&ly->v_cache, kv_bytes);
        m->core_vram_bytes += (int64_t)(kv_bytes * 2);
        for (int e = 0; e < CM_MAX_EXPERTS; e++) ly->exp_to_slot[e] = -1;
        for (int s = 0; s < CM_MAX_EXPERTS; s++) { ly->slot_to_exp[s] = -1; ly->slot_lru[s] = 0; }

        if (m->dense_model && !ly->is_pinned_dense) {
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
            size_t cbytes = (size_t)m->cache_slots_per_layer * (size_t)ly->slot_stride_bytes;
            if (g_cu.cuMemAlloc(&ly->cache_dptr, cbytes) != CUDA_SUCCESS) {
                fprintf(stderr, "[tenon-cuda-stream] VRAM OOM allocating layer %d cache (%zu MiB)\n", L, cbytes >> 20);
                return 0;
            }
            m->cache_vram_bytes += (int64_t)cbytes;
        }
    }

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

    g_cu.cuMemHostAlloc((void**)&m->h_rlogits, (size_t)(m->n_experts > 16 ? m->n_experts : 16) * 4, CU_MEMHOSTALLOC_PORTABLE);
    g_cu.cuMemHostAlloc((void**)&m->h_argmax, 16, CU_MEMHOSTALLOC_PORTABLE);
    g_cu.cuMemHostAlloc((void**)&m->h_eidx, (size_t)(CM_MAX_TOPK * 2) * sizeof(uint32_t), CU_MEMHOSTALLOC_PORTABLE);
    g_cu.cuMemHostAlloc((void**)&m->h_ewt,  (size_t)(CM_MAX_TOPK * 2) * sizeof(float),    CU_MEMHOSTALLOC_PORTABLE);
    g_cu.cuMemHostAlloc((void**)&m->h_emb_row, 65536, CU_MEMHOSTALLOC_PORTABLE);
    g_cu.cuMemHostAlloc((void**)&m->h_xn, (size_t)m->d_model * 4, CU_MEMHOSTALLOC_PORTABLE);
    g_cu.cuMemHostAlloc((void**)&m->h_cpu_accum, (size_t)m->d_model * 4, CU_MEMHOSTALLOC_PORTABLE);
    m->cpu_h = (float*)malloc(h_elems * sizeof(float));
    size_t dm_blks = ((size_t)m->d_model + 31u) >> 5;
    size_t h_blks  = (h_elems + 31u) >> 5;
    m->cpu_xq8  = (int8_t*)malloc(dm_blks * 32u);
    m->cpu_xd   = (float*)malloc(dm_blks * sizeof(float));
    m->cpu_xsum = (float*)malloc(dm_blks * sizeof(float));
    m->cpu_hq8  = (int8_t*)malloc(h_blks * 32u);
    m->cpu_hd   = (float*)malloc(h_blks * sizeof(float));
    m->cpu_hsum = (float*)malloc(h_blks * sizeof(float));

    if (!m->resident_mode) {
        cm_io_pool_start(m);
        if (m->enable_overlap && !m->use_nocache && m->mmap_ptrs[0] != NULL) {
            cm_cpu_pool_start(m);
            if (!m->dense_model && m->promo_stage) cm_promo_start(m);
        }
    }

    m->active = 1;
    return (int64_t)slot;
}

int64_t tn_cumoe_step(int64_t handle, int64_t tok_id, int64_t pos) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return 0;
    CmModel* m = &g_cm[handle];
    double t0 = cm_now_ms(), io_acc = 0.0;

    // PATH A: FULL RESIDENT DENSE (100% GPU VRAM)
    if (m->dense_model && m->resident_mode) {
        launch_embed(m, m->compute_stream, (uint32_t)tok_id);
        for (int L = 0; L < m->n_layers; L++) {
            CmLayer* ly = &m->layers[L];
            launch_layer_attn(m, m->compute_stream, L, (uint32_t)pos, 0);
            launch_dense_ffn_raw(m, m->compute_stream, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                                 ly->ffn_up.dptr, ly->ffn_up.ggml_type,
                                 ly->ffn_down.dptr, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense);
            m->cache_hits++;
        }
        launch_final_lm_head(m, m->compute_stream);
        g_cu.cuMemcpyDtoHAsync(m->h_argmax, m->d_argmax, 4, m->compute_stream);
        g_cu.cuStreamSynchronize(m->compute_stream);
        m->last_total_ms = cm_now_ms() - t0;
        return (int64_t)m->h_argmax[0];
    }

    // PATH B: DENSE STREAMING (All-80-Attn + Prefix FFNs + In-Place AVX2/FMA DDR4 OR 4-Slot PCIe Ring!)
    if (m->dense_model) {
        int first_stream = m->cache_slots_per_layer;
        if (m->enable_overlap && !m->use_nocache && m->cpu_pool_init && m->dense_pin_all_attn) {
            // HETEROGENEOUS HYBRID DENSE: All-80-Attn + Prefix FFNs on 1,008 GB/s GPU,
            // remaining FFNs executed IN PLACE on 32-thread AVX2+FMA DDR4 (ZERO PCIe weight copy!)
            launch_embed(m, m->compute_stream, (uint32_t)tok_id);
            for (int L = 0; L < first_stream && L < m->n_layers; L++) {
                CmLayer* ly = &m->layers[L];
                launch_layer_attn(m, m->compute_stream, L, (uint32_t)pos, 0);
                launch_dense_ffn_raw(m, m->compute_stream, ly->ffn_gate.dptr, ly->ffn_gate.ggml_type,
                                     ly->ffn_up.dptr, ly->ffn_up.ggml_type,
                                     ly->ffn_down.dptr, ly->ffn_down.ggml_type, (uint32_t)m->d_ff_dense);
                m->cache_hits++;
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
        cm_cpu_router_topk(m->h_rlogits, ly->exp_bias_host, m->n_experts, m->top_k,
                           m->gating_func, m->expert_weights_norm, m->expert_weights_scale, top_e, top_w);

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
                launch_dense_ffn_raw(m, m->compute_stream, ly->sh_gate.dptr, ly->sh_gate.ggml_type,
                                     ly->sh_up.dptr, ly->sh_up.ggml_type, ly->sh_down.dptr, ly->sh_down.ggml_type, (uint32_t)m->d_ff_exp);
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
                launch_dense_ffn_raw(m, m->compute_stream, ly->sh_gate.dptr, ly->sh_gate.ggml_type,
                                     ly->sh_up.dptr, ly->sh_up.ggml_type, ly->sh_down.dptr, ly->sh_down.ggml_type, (uint32_t)m->d_ff_exp);
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
                launch_dense_ffn_raw(m, m->compute_stream, ly->sh_gate.dptr, ly->sh_gate.ggml_type,
                                     ly->sh_up.dptr, ly->sh_up.ggml_type, ly->sh_down.dptr, ly->sh_down.ggml_type, (uint32_t)m->d_ff_exp);
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
                launch_dense_ffn_raw(m, m->compute_stream, ly->sh_gate.dptr, ly->sh_gate.ggml_type,
                                     ly->sh_up.dptr, ly->sh_up.ggml_type, ly->sh_down.dptr, ly->sh_down.ggml_type, (uint32_t)m->d_ff_exp);
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
        default: return 0;
    }
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
    const unsigned char* s = (const unsigned char*)m->vocab_tokens[tok_id];
    size_t o = 0;
    for (size_t i = 0; s[i] && o + 2 < sizeof(g_tok); ) {
        if (s[i] == 0xC4 && s[i + 1] == 0xA0) { g_tok[o++] = ' '; i += 2; }
        else if (s[i] == 0xC4 && s[i + 1] == 0x8A) { g_tok[o++] = '\n'; i += 2; }
        else if (s[i] == 0xE2 && s[i + 1] == 0x96 && s[i + 2] == 0x81) { g_tok[o++] = ' '; i += 3; }
        else g_tok[o++] = (char)s[i++];
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

static void cm_free_dptr(CUdeviceptr* p) { if (*p) { g_cu.cuMemFree(*p); *p = 0; } }

int64_t tn_cumoe_free(int64_t handle) {
    if (handle <= 0 || handle >= CM_MAX_MODELS || !g_cm[handle].active) return 0;
    CmModel* m = &g_cm[handle];
    cm_promo_stop(m);
    cm_cpu_pool_stop(m);
    cm_io_pool_stop(m);
    if (m->compute_stream) g_cu.cuStreamSynchronize(m->compute_stream);
    if (m->dma_stream) g_cu.cuStreamSynchronize(m->dma_stream);
    for (int L = 0; L < m->n_layers; L++) {
        CmLayer* ly = &m->layers[L];
        free(ly->exp_bias_host); ly->exp_bias_host = NULL;
        cm_free_dptr(&ly->cache_dptr); cm_free_dptr(&ly->k_cache); cm_free_dptr(&ly->v_cache);
        cm_free_dptr(&ly->attn_norm.dptr); cm_free_dptr(&ly->wq.dptr); cm_free_dptr(&ly->wk.dptr);
        cm_free_dptr(&ly->wv.dptr); cm_free_dptr(&ly->wo.dptr);
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
    free(m->cpu_h); m->cpu_h = NULL;
    free(m->cpu_xq8); m->cpu_xq8 = NULL;
    free(m->cpu_xd); m->cpu_xd = NULL;
    free(m->cpu_xsum); m->cpu_xsum = NULL;
    free(m->cpu_hq8); m->cpu_hq8 = NULL;
    free(m->cpu_hd); m->cpu_hd = NULL;
    free(m->cpu_hsum); m->cpu_hsum = NULL;
    cm_free_dptr(&m->d_cpu_accum);
    if (m->lm_head.dptr == m->tok_embd.dptr) m->lm_head.dptr = 0;
    cm_free_dptr(&m->tok_embd.dptr); cm_free_dptr(&m->out_norm.dptr); cm_free_dptr(&m->lm_head.dptr); cm_free_dptr(&m->rope_freqs.dptr);
    cm_free_dptr(&m->d_emb_row); cm_free_dptr(&m->d_x); cm_free_dptr(&m->d_xn); cm_free_dptr(&m->d_q);
    cm_free_dptr(&m->d_k); cm_free_dptr(&m->d_v); cm_free_dptr(&m->d_ctx); cm_free_dptr(&m->d_rlogits);
    cm_free_dptr(&m->d_eidx); cm_free_dptr(&m->d_ewt); cm_free_dptr(&m->d_h); cm_free_dptr(&m->d_sh_g);
    cm_free_dptr(&m->d_sh_u); cm_free_dptr(&m->d_logits); cm_free_dptr(&m->d_argmax);
    if (m->ev_dma_done) { g_cu.cuEventDestroy(m->ev_dma_done); m->ev_dma_done = NULL; }
    if (m->compute_stream) { g_cu.cuStreamDestroy(m->compute_stream); m->compute_stream = NULL; }
    if (m->dma_stream) { g_cu.cuStreamDestroy(m->dma_stream); m->dma_stream = NULL; }
    if (m->vocab_tokens) {
        for (int i = 0; i < m->vocab_size; i++) free(m->vocab_tokens[i]);
        free(m->vocab_tokens); m->vocab_tokens = NULL;
    }
    for (int s = 0; s < m->n_shards; s++) {
        if (m->mmap_ptrs[s]) { munmap(m->mmap_ptrs[s], m->shard_sizes[s]); m->mmap_ptrs[s] = NULL; }
        if (m->fds[s] >= 0) { close(m->fds[s]); m->fds[s] = -1; }
    }
    m->active = 0;
    return 1;
}
