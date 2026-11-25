// std/gpu/cuda/cuda_shim.c — Zero-Link-Flag Dynamic NVIDIA CUDA Driver + NVRTC Runtime HAL
//
// Uses `dlopen("libcuda.so.1")` + `dlopen("libnvrtc.so")` (with automatic `nvcc --ptx` fallback)
// so Tenon binaries compile cleanly on ANY host (macOS, Linux, CI) without static CUDA SDK links,
// while running at 100% native NVIDIA GPU speed on RTX 4090 / L40S / A100 / H100 / H200 / B200!

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if !defined(_WIN32)
#include <dlfcn.h>
#include <unistd.h>
#endif

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef void* CUgraph;
typedef void* CUgraphExec;
typedef uint64_t CUdeviceptr;

typedef int nvrtcResult;
typedef void* nvrtcProgram;
typedef int cublasStatus_t;
typedef void* cublasHandle_t;

#define CUDA_SUCCESS 0
#define NVRTC_SUCCESS 0
#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_OP_N 0
#define CUBLAS_OP_T 1
#define CUDA_R_32F 0
#define CUDA_R_16F 2
#define CUDA_R_16BF 14
#define CUBLAS_COMPUTE_32F 68
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP 99
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE 36
#define CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH 37
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
#define CU_STREAM_CAPTURE_MODE_GLOBAL 0

typedef CUresult (*pfn_cuInit)(unsigned int);
typedef CUresult (*pfn_cuDeviceGetCount)(int*);
typedef CUresult (*pfn_cuDeviceGet)(CUdevice*, int);
typedef CUresult (*pfn_cuDeviceGetName)(char*, int, CUdevice);
typedef CUresult (*pfn_cuDeviceTotalMem_v2)(size_t*, CUdevice);
typedef CUresult (*pfn_cuDeviceGetAttribute)(int*, int, CUdevice);
typedef CUresult (*pfn_cuCtxCreate_v2)(CUcontext*, unsigned int, CUdevice);
typedef CUresult (*pfn_cuCtxSetCurrent)(CUcontext);
typedef CUresult (*pfn_cuCtxSynchronize)(void);
typedef CUresult (*pfn_cuMemAlloc_v2)(CUdeviceptr*, size_t);
typedef CUresult (*pfn_cuMemFree_v2)(CUdeviceptr);
typedef CUresult (*pfn_cuMemcpyHtoD_v2)(CUdeviceptr, const void*, size_t);
typedef CUresult (*pfn_cuMemcpyDtoH_v2)(void*, CUdeviceptr, size_t);
typedef CUresult (*pfn_cuMemcpyHtoDAsync_v2)(CUdeviceptr, const void*, size_t, CUstream);
typedef CUresult (*pfn_cuMemcpyDtoHAsync_v2)(void*, CUdeviceptr, size_t, CUstream);
typedef CUresult (*pfn_cuMemsetD8_v2)(CUdeviceptr, unsigned char, size_t);
typedef CUresult (*pfn_cuModuleLoadData)(CUmodule*, const void*);
typedef CUresult (*pfn_cuModuleLoadDataEx)(CUmodule*, const void*, unsigned int, int*, void**);
typedef CUresult (*pfn_cuModuleUnload)(CUmodule);
typedef CUresult (*pfn_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef CUresult (*pfn_cuLaunchKernel)(
    CUfunction, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int,
    unsigned int, CUstream, void**, void**);
typedef CUresult (*pfn_cuStreamCreate)(CUstream*, unsigned int);
typedef CUresult (*pfn_cuStreamSynchronize)(CUstream);
typedef CUresult (*pfn_cuStreamBeginCapture_v2)(CUstream, int);
typedef CUresult (*pfn_cuStreamEndCapture)(CUstream, CUgraph*);
typedef CUresult (*pfn_cuGraphInstantiateWithFlags)(CUgraphExec*, CUgraph, uint64_t);
typedef CUresult (*pfn_cuGraphInstantiate_v2)(CUgraphExec*, CUgraph, void*, char*, size_t);
typedef CUresult (*pfn_cuGraphLaunch)(CUgraphExec, CUstream);
typedef CUresult (*pfn_cuGraphExecDestroy)(CUgraphExec);
typedef CUresult (*pfn_cuGraphDestroy)(CUgraph);

typedef nvrtcResult (*pfn_nvrtcCreateProgram)(nvrtcProgram*, const char*, const char*, int, const char* const*, const char* const*);
typedef nvrtcResult (*pfn_nvrtcCompileProgram)(nvrtcProgram, int, const char* const*);
typedef nvrtcResult (*pfn_nvrtcGetPTXSize)(nvrtcProgram, size_t*);
typedef nvrtcResult (*pfn_nvrtcGetPTX)(nvrtcProgram, char*);
typedef nvrtcResult (*pfn_nvrtcGetProgramLogSize)(nvrtcProgram, size_t*);
typedef nvrtcResult (*pfn_nvrtcGetProgramLog)(nvrtcProgram, char*);
typedef nvrtcResult (*pfn_nvrtcDestroyProgram)(nvrtcProgram*);

typedef cublasStatus_t (*pfn_cublasCreate_v2)(cublasHandle_t*);
typedef cublasStatus_t (*pfn_cublasSetStream_v2)(cublasHandle_t, CUstream);
typedef cublasStatus_t (*pfn_cublasGemmEx)(
    cublasHandle_t, int, int, int, int, int,
    const void*, const void*, int, int,
    const void*, int, int,
    const void*, void*, int, int,
    int, int);

typedef struct {
    int initialized;
    void* libcuda;
    void* libnvrtc;
    void* libcublas;
    CUdevice dev;
    CUcontext ctx;
    CUstream stream;
    cublasHandle_t cublas;
    char dev_name[128];
    int cc_major;
    int cc_minor;
    int sm_count;
    size_t vram_bytes;
    double peak_gbs;

    pfn_cuInit cuInit;
    pfn_cuDeviceGetCount cuDeviceGetCount;
    pfn_cuDeviceGet cuDeviceGet;
    pfn_cuDeviceGetName cuDeviceGetName;
    pfn_cuDeviceTotalMem_v2 cuDeviceTotalMem;
    pfn_cuDeviceGetAttribute cuDeviceGetAttribute;
    pfn_cuCtxCreate_v2 cuCtxCreate;
    pfn_cuCtxSetCurrent cuCtxSetCurrent;
    pfn_cuCtxSynchronize cuCtxSynchronize;
    pfn_cuMemAlloc_v2 cuMemAlloc;
    pfn_cuMemFree_v2 cuMemFree;
    pfn_cuMemcpyHtoD_v2 cuMemcpyHtoD;
    pfn_cuMemcpyDtoH_v2 cuMemcpyDtoH;
    pfn_cuMemcpyHtoDAsync_v2 cuMemcpyHtoDAsync;
    pfn_cuMemcpyDtoHAsync_v2 cuMemcpyDtoHAsync;
    pfn_cuMemsetD8_v2 cuMemsetD8;
    pfn_cuModuleLoadData cuModuleLoadData;
    pfn_cuModuleLoadDataEx cuModuleLoadDataEx;
    pfn_cuModuleUnload cuModuleUnload;
    pfn_cuModuleGetFunction cuModuleGetFunction;
    pfn_cuLaunchKernel cuLaunchKernel;
    pfn_cuStreamCreate cuStreamCreate;
    pfn_cuStreamSynchronize cuStreamSynchronize;
    pfn_cuStreamBeginCapture_v2 cuStreamBeginCapture;
    pfn_cuStreamEndCapture cuStreamEndCapture;
    pfn_cuGraphInstantiateWithFlags cuGraphInstantiateWithFlags;
    pfn_cuGraphInstantiate_v2 cuGraphInstantiate;
    pfn_cuGraphLaunch cuGraphLaunch;
    pfn_cuGraphExecDestroy cuGraphExecDestroy;
    pfn_cuGraphDestroy cuGraphDestroy;

    pfn_nvrtcCreateProgram nvrtcCreateProgram;
    pfn_nvrtcCompileProgram nvrtcCompileProgram;
    pfn_nvrtcGetPTXSize nvrtcGetPTXSize;
    pfn_nvrtcGetPTX nvrtcGetPTX;
    pfn_nvrtcGetProgramLogSize nvrtcGetProgramLogSize;
    pfn_nvrtcGetProgramLog nvrtcGetProgramLog;
    pfn_nvrtcDestroyProgram nvrtcDestroyProgram;

    pfn_cublasCreate_v2 cublasCreate;
    pfn_cublasSetStream_v2 cublasSetStream;
    pfn_cublasGemmEx cublasGemmEx;
} TnCudaState;

/* Shared across cuda_shim.c and cuda_ultra_shim.c */
TnCudaState g_tn_cuda = {0};

typedef struct {
    CUdeviceptr dptr;
    size_t byte_len;
} TnCudaBuf;

typedef struct {
    CUmodule mod;
    CUfunction fn;
} TnCudaKernel;

#define TN_CUDA_MAX_BUFS 4096
#define TN_CUDA_MAX_KERNELS 256
static TnCudaBuf g_tn_cuda_bufs[TN_CUDA_MAX_BUFS];
static TnCudaKernel g_tn_cuda_kernels[TN_CUDA_MAX_KERNELS];

static void* tn_cuda_sym(void* lib, const char* primary, const char* fallback) {
    if (!lib) return NULL;
#if !defined(_WIN32)
    void* p = dlsym(lib, primary);
    if (!p && fallback) p = dlsym(lib, fallback);
    return p;
#else
    (void)primary; (void)fallback;
    return NULL;
#endif
}

int64_t tn_cuda_init(void) {
    if (g_tn_cuda.initialized) return 1;
    strncpy(g_tn_cuda.dev_name, "CUDA Unavailable", sizeof(g_tn_cuda.dev_name) - 1);
#if !defined(_WIN32)
    const char* cuda_libs[] = {
        "libcuda.so.1",
        "libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
        "/usr/local/cuda/lib64/stubs/libcuda.so",
        NULL
    };
    for (int i = 0; cuda_libs[i]; i++) {
        g_tn_cuda.libcuda = dlopen(cuda_libs[i], RTLD_LAZY | RTLD_GLOBAL);
        if (g_tn_cuda.libcuda) break;
    }
    if (!g_tn_cuda.libcuda) return 0;

    g_tn_cuda.cuInit = (pfn_cuInit)tn_cuda_sym(g_tn_cuda.libcuda, "cuInit", NULL);
    g_tn_cuda.cuDeviceGetCount = (pfn_cuDeviceGetCount)tn_cuda_sym(g_tn_cuda.libcuda, "cuDeviceGetCount", NULL);
    g_tn_cuda.cuDeviceGet = (pfn_cuDeviceGet)tn_cuda_sym(g_tn_cuda.libcuda, "cuDeviceGet", NULL);
    g_tn_cuda.cuDeviceGetName = (pfn_cuDeviceGetName)tn_cuda_sym(g_tn_cuda.libcuda, "cuDeviceGetName", NULL);
    g_tn_cuda.cuDeviceTotalMem = (pfn_cuDeviceTotalMem_v2)tn_cuda_sym(g_tn_cuda.libcuda, "cuDeviceTotalMem_v2", "cuDeviceTotalMem");
    g_tn_cuda.cuDeviceGetAttribute = (pfn_cuDeviceGetAttribute)tn_cuda_sym(g_tn_cuda.libcuda, "cuDeviceGetAttribute", NULL);
    g_tn_cuda.cuCtxCreate = (pfn_cuCtxCreate_v2)tn_cuda_sym(g_tn_cuda.libcuda, "cuCtxCreate_v2", "cuCtxCreate");
    g_tn_cuda.cuCtxSetCurrent = (pfn_cuCtxSetCurrent)tn_cuda_sym(g_tn_cuda.libcuda, "cuCtxSetCurrent", NULL);
    g_tn_cuda.cuCtxSynchronize = (pfn_cuCtxSynchronize)tn_cuda_sym(g_tn_cuda.libcuda, "cuCtxSynchronize", NULL);
    g_tn_cuda.cuMemAlloc = (pfn_cuMemAlloc_v2)tn_cuda_sym(g_tn_cuda.libcuda, "cuMemAlloc_v2", "cuMemAlloc");
    g_tn_cuda.cuMemFree = (pfn_cuMemFree_v2)tn_cuda_sym(g_tn_cuda.libcuda, "cuMemFree_v2", "cuMemFree");
    g_tn_cuda.cuMemcpyHtoD = (pfn_cuMemcpyHtoD_v2)tn_cuda_sym(g_tn_cuda.libcuda, "cuMemcpyHtoD_v2", "cuMemcpyHtoD");
    g_tn_cuda.cuMemcpyDtoH = (pfn_cuMemcpyDtoH_v2)tn_cuda_sym(g_tn_cuda.libcuda, "cuMemcpyDtoH_v2", "cuMemcpyDtoH");
    g_tn_cuda.cuMemcpyHtoDAsync = (pfn_cuMemcpyHtoDAsync_v2)tn_cuda_sym(g_tn_cuda.libcuda, "cuMemcpyHtoDAsync_v2", "cuMemcpyHtoDAsync");
    g_tn_cuda.cuMemcpyDtoHAsync = (pfn_cuMemcpyDtoHAsync_v2)tn_cuda_sym(g_tn_cuda.libcuda, "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoHAsync");
    g_tn_cuda.cuMemsetD8 = (pfn_cuMemsetD8_v2)tn_cuda_sym(g_tn_cuda.libcuda, "cuMemsetD8_v2", "cuMemsetD8");
    g_tn_cuda.cuModuleLoadData = (pfn_cuModuleLoadData)tn_cuda_sym(g_tn_cuda.libcuda, "cuModuleLoadData", NULL);
    g_tn_cuda.cuModuleLoadDataEx = (pfn_cuModuleLoadDataEx)tn_cuda_sym(g_tn_cuda.libcuda, "cuModuleLoadDataEx", NULL);
    g_tn_cuda.cuModuleUnload = (pfn_cuModuleUnload)tn_cuda_sym(g_tn_cuda.libcuda, "cuModuleUnload", NULL);
    g_tn_cuda.cuModuleGetFunction = (pfn_cuModuleGetFunction)tn_cuda_sym(g_tn_cuda.libcuda, "cuModuleGetFunction", NULL);
    g_tn_cuda.cuLaunchKernel = (pfn_cuLaunchKernel)tn_cuda_sym(g_tn_cuda.libcuda, "cuLaunchKernel", NULL);
    g_tn_cuda.cuStreamCreate = (pfn_cuStreamCreate)tn_cuda_sym(g_tn_cuda.libcuda, "cuStreamCreate", NULL);
    g_tn_cuda.cuStreamSynchronize = (pfn_cuStreamSynchronize)tn_cuda_sym(g_tn_cuda.libcuda, "cuStreamSynchronize", NULL);
    g_tn_cuda.cuStreamBeginCapture = (pfn_cuStreamBeginCapture_v2)tn_cuda_sym(g_tn_cuda.libcuda, "cuStreamBeginCapture_v2", "cuStreamBeginCapture");
    g_tn_cuda.cuStreamEndCapture = (pfn_cuStreamEndCapture)tn_cuda_sym(g_tn_cuda.libcuda, "cuStreamEndCapture", NULL);
    g_tn_cuda.cuGraphInstantiateWithFlags = (pfn_cuGraphInstantiateWithFlags)tn_cuda_sym(g_tn_cuda.libcuda, "cuGraphInstantiateWithFlags", NULL);
    g_tn_cuda.cuGraphInstantiate = (pfn_cuGraphInstantiate_v2)tn_cuda_sym(g_tn_cuda.libcuda, "cuGraphInstantiate_v2", "cuGraphInstantiate");
    g_tn_cuda.cuGraphLaunch = (pfn_cuGraphLaunch)tn_cuda_sym(g_tn_cuda.libcuda, "cuGraphLaunch", NULL);
    g_tn_cuda.cuGraphExecDestroy = (pfn_cuGraphExecDestroy)tn_cuda_sym(g_tn_cuda.libcuda, "cuGraphExecDestroy", NULL);
    g_tn_cuda.cuGraphDestroy = (pfn_cuGraphDestroy)tn_cuda_sym(g_tn_cuda.libcuda, "cuGraphDestroy", NULL);

    if (!g_tn_cuda.cuInit || !g_tn_cuda.cuDeviceGetCount || !g_tn_cuda.cuDeviceGet || !g_tn_cuda.cuCtxCreate) {
        return 0;
    }
    if (g_tn_cuda.cuInit(0) != CUDA_SUCCESS) return 0;
    int count = 0;
    if (g_tn_cuda.cuDeviceGetCount(&count) != CUDA_SUCCESS || count <= 0) return 0;
    if (g_tn_cuda.cuDeviceGet(&g_tn_cuda.dev, 0) != CUDA_SUCCESS) return 0;
    if (g_tn_cuda.cuCtxCreate(&g_tn_cuda.ctx, 0, g_tn_cuda.dev) != CUDA_SUCCESS) return 0;
    if (g_tn_cuda.cuStreamCreate) {
        g_tn_cuda.cuStreamCreate(&g_tn_cuda.stream, 0);
    }

    if (g_tn_cuda.cuDeviceGetName) {
        g_tn_cuda.cuDeviceGetName(g_tn_cuda.dev_name, sizeof(g_tn_cuda.dev_name) - 1, g_tn_cuda.dev);
    }
    if (g_tn_cuda.cuDeviceTotalMem) {
        g_tn_cuda.cuDeviceTotalMem(&g_tn_cuda.vram_bytes, g_tn_cuda.dev);
    }
    int mem_clk_khz = 0, bus_bits = 0;
    if (g_tn_cuda.cuDeviceGetAttribute) {
        g_tn_cuda.cuDeviceGetAttribute(&g_tn_cuda.cc_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, g_tn_cuda.dev);
        g_tn_cuda.cuDeviceGetAttribute(&g_tn_cuda.cc_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, g_tn_cuda.dev);
        g_tn_cuda.cuDeviceGetAttribute(&g_tn_cuda.sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, g_tn_cuda.dev);
        g_tn_cuda.cuDeviceGetAttribute(&mem_clk_khz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, g_tn_cuda.dev);
        g_tn_cuda.cuDeviceGetAttribute(&bus_bits, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, g_tn_cuda.dev);
    }
    g_tn_cuda.peak_gbs = 2.0 * ((double)mem_clk_khz * 1e3) * ((double)bus_bits / 8.0) / 1e9;

    /* Load NVRTC dynamically (optional — nvcc fallback also supported!) */
    const char* nvrtc_libs[] = {
        "libnvrtc.so",
        "libnvrtc.so.13",
        "libnvrtc.so.12",
        "/usr/local/cuda/lib64/libnvrtc.so",
        "/usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so",
        NULL
    };
    for (int i = 0; nvrtc_libs[i]; i++) {
        g_tn_cuda.libnvrtc = dlopen(nvrtc_libs[i], RTLD_LAZY | RTLD_GLOBAL);
        if (g_tn_cuda.libnvrtc) break;
    }
    if (g_tn_cuda.libnvrtc) {
        g_tn_cuda.nvrtcCreateProgram = (pfn_nvrtcCreateProgram)dlsym(g_tn_cuda.libnvrtc, "nvrtcCreateProgram");
        g_tn_cuda.nvrtcCompileProgram = (pfn_nvrtcCompileProgram)dlsym(g_tn_cuda.libnvrtc, "nvrtcCompileProgram");
        g_tn_cuda.nvrtcGetPTXSize = (pfn_nvrtcGetPTXSize)dlsym(g_tn_cuda.libnvrtc, "nvrtcGetPTXSize");
        g_tn_cuda.nvrtcGetPTX = (pfn_nvrtcGetPTX)dlsym(g_tn_cuda.libnvrtc, "nvrtcGetPTX");
        g_tn_cuda.nvrtcGetProgramLogSize = (pfn_nvrtcGetProgramLogSize)dlsym(g_tn_cuda.libnvrtc, "nvrtcGetProgramLogSize");
        g_tn_cuda.nvrtcGetProgramLog = (pfn_nvrtcGetProgramLog)dlsym(g_tn_cuda.libnvrtc, "nvrtcGetProgramLog");
        g_tn_cuda.nvrtcDestroyProgram = (pfn_nvrtcDestroyProgram)dlsym(g_tn_cuda.libnvrtc, "nvrtcDestroyProgram");
    }

    /* Load cuBLAS dynamically for Tensor Core FP16/BF16 GEMM (ZERO static link flags!) */
    const char* cublas_libs[] = {
        "libcublas.so",
        "libcublas.so.13",
        "libcublas.so.12",
        "/usr/local/cuda/lib64/libcublas.so",
        "/usr/local/cuda/targets/x86_64-linux/lib/libcublas.so",
        "/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib/libcublas.so.12",
        "/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib/libcublas.so.13",
        NULL
    };
    for (int i = 0; cublas_libs[i]; i++) {
        g_tn_cuda.libcublas = dlopen(cublas_libs[i], RTLD_LAZY | RTLD_GLOBAL);
        if (g_tn_cuda.libcublas) break;
    }
    if (g_tn_cuda.libcublas) {
        g_tn_cuda.cublasCreate = (pfn_cublasCreate_v2)dlsym(g_tn_cuda.libcublas, "cublasCreate_v2");
        g_tn_cuda.cublasSetStream = (pfn_cublasSetStream_v2)dlsym(g_tn_cuda.libcublas, "cublasSetStream_v2");
        g_tn_cuda.cublasGemmEx = (pfn_cublasGemmEx)dlsym(g_tn_cuda.libcublas, "cublasGemmEx");
        if (g_tn_cuda.cublasCreate && g_tn_cuda.cublasCreate(&g_tn_cuda.cublas) == CUBLAS_STATUS_SUCCESS) {
            if (g_tn_cuda.cublasSetStream && g_tn_cuda.stream) {
                g_tn_cuda.cublasSetStream(g_tn_cuda.cublas, g_tn_cuda.stream);
            }
        }
    }

    g_tn_cuda.initialized = 1;
    return 1;
#else
    return 0;
#endif
}

int64_t tn_cuda_device_name_cstr(void) {
    return (int64_t)(uintptr_t)g_tn_cuda.dev_name;
}
int64_t tn_cuda_compute_cap(void) {
    return (int64_t)(g_tn_cuda.cc_major * 10 + g_tn_cuda.cc_minor);
}
int64_t tn_cuda_sm_count(void) {
    return (int64_t)g_tn_cuda.sm_count;
}
int64_t tn_cuda_vram_mb(void) {
    return (int64_t)(g_tn_cuda.vram_bytes / (1024ULL * 1024ULL));
}
double tn_cuda_peak_gbs(void) {
    return g_tn_cuda.peak_gbs;
}

CUdeviceptr tn_cuda_buf_dptr(int64_t h) {
    if (h <= 0 || h >= TN_CUDA_MAX_BUFS) return 0;
    return g_tn_cuda_bufs[h].dptr;
}

int64_t tn_cuda_buf_uninit(int64_t byte_len) {
    if (!g_tn_cuda.initialized && !tn_cuda_init()) return 0;
    size_t sz = byte_len < 16 ? 16 : (size_t)byte_len;
    int slot = -1;
    for (int i = 1; i < TN_CUDA_MAX_BUFS; i++) {
        if (g_tn_cuda_bufs[i].dptr == 0) { slot = i; break; }
    }
    if (slot < 0) return 0;
    CUdeviceptr dp = 0;
    if (g_tn_cuda.cuMemAlloc(&dp, sz) != CUDA_SUCCESS) return 0;
    if (g_tn_cuda.cuMemsetD8) g_tn_cuda.cuMemsetD8(dp, 0, sz);
    g_tn_cuda_bufs[slot].dptr = dp;
    g_tn_cuda_bufs[slot].byte_len = sz;
    return (int64_t)slot;
}

int64_t tn_cuda_buf_create(int64_t data_ptr, int64_t byte_len) {
    int64_t h = tn_cuda_buf_uninit(byte_len);
    if (h <= 0) return 0;
    if (data_ptr != 0 && byte_len > 0) {
        if (g_tn_cuda.cuMemcpyHtoD(g_tn_cuda_bufs[h].dptr, (const void*)(uintptr_t)data_ptr, (size_t)byte_len) != CUDA_SUCCESS) {
            g_tn_cuda.cuMemFree(g_tn_cuda_bufs[h].dptr);
            g_tn_cuda_bufs[h].dptr = 0;
            g_tn_cuda_bufs[h].byte_len = 0;
            return 0;
        }
    }
    return h;
}

int64_t tn_cuda_buf_upload(int64_t handle, int64_t src_ptr, int64_t byte_len) {
    if (handle <= 0 || handle >= TN_CUDA_MAX_BUFS || g_tn_cuda_bufs[handle].dptr == 0 || src_ptr == 0 || byte_len <= 0) return 0;
    size_t n = (size_t)byte_len;
    if (n > g_tn_cuda_bufs[handle].byte_len) n = g_tn_cuda_bufs[handle].byte_len;
    if (g_tn_cuda.cuMemcpyHtoD(g_tn_cuda_bufs[handle].dptr, (const void*)(uintptr_t)src_ptr, n) != CUDA_SUCCESS) return 0;
    return (int64_t)n;
}

int64_t tn_cuda_buf_read(int64_t handle, int64_t dst_ptr, int64_t byte_len) {
    if (handle <= 0 || handle >= TN_CUDA_MAX_BUFS || g_tn_cuda_bufs[handle].dptr == 0 || dst_ptr == 0 || byte_len <= 0) return 0;
    size_t n = (size_t)byte_len;
    if (n > g_tn_cuda_bufs[handle].byte_len) n = g_tn_cuda_bufs[handle].byte_len;
    if (g_tn_cuda.cuMemcpyDtoH((void*)(uintptr_t)dst_ptr, g_tn_cuda_bufs[handle].dptr, n) != CUDA_SUCCESS) return 0;
    return (int64_t)n;
}

int64_t tn_cuda_buf_free(int64_t handle) {
    if (handle <= 0 || handle >= TN_CUDA_MAX_BUFS || g_tn_cuda_bufs[handle].dptr == 0) return 0;
    g_tn_cuda.cuMemFree(g_tn_cuda_bufs[handle].dptr);
    g_tn_cuda_bufs[handle].dptr = 0;
    g_tn_cuda_bufs[handle].byte_len = 0;
    return 1;
}

/* Compile CUDA C++ source -> PTX/CUBIN via nvcc (checking /usr/local/cuda/bin/nvcc) OR NVRTC! */
CUmodule tn_cuda_compile_module(const char* cu_src, int prefer_nvcc) {
    if (!g_tn_cuda.initialized && !tn_cuda_init()) return NULL;
    char* ptx = NULL;
    size_t ptx_len = 0;

    for (int pass = 0; pass < 2 && !ptx; pass++) {
        int try_nvcc = (pass == 0) ? prefer_nvcc : !prefer_nvcc;
        if (!try_nvcc && g_tn_cuda.nvrtcCreateProgram && g_tn_cuda.nvrtcCompileProgram) {
            nvrtcProgram prog = NULL;
            if (g_tn_cuda.nvrtcCreateProgram(&prog, cu_src, "tenon_kernel.cu", 0, NULL, NULL) == NVRTC_SUCCESS) {
                char arch_opt[64];
                snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d", g_tn_cuda.cc_major, g_tn_cuda.cc_minor);
                const char* opts[5] = {
                    arch_opt, "--use_fast_math",
                    "-I/usr/local/cuda/include",
                    "-I/usr/local/cuda/targets/x86_64-linux/include",
                    "-I/usr/include"
                };
                nvrtcResult cr = g_tn_cuda.nvrtcCompileProgram(prog, 5, opts);
                if (cr != NVRTC_SUCCESS) {
                    const char* opts2[4] = {
                        "--gpu-architecture=compute_80", "--use_fast_math",
                        "-I/usr/local/cuda/include",
                        "-I/usr/local/cuda/targets/x86_64-linux/include"
                    };
                    cr = g_tn_cuda.nvrtcCompileProgram(prog, 4, opts2);
                }
                if (cr == NVRTC_SUCCESS) {
                    g_tn_cuda.nvrtcGetPTXSize(prog, &ptx_len);
                    ptx = (char*)malloc(ptx_len + 1);
                    if (ptx) {
                        g_tn_cuda.nvrtcGetPTX(prog, ptx);
                        ptx[ptx_len] = '\0';
                    }
                } else {
                    size_t lsz = 0;
                    g_tn_cuda.nvrtcGetProgramLogSize(prog, &lsz);
                    if (lsz > 1) {
                        char* log = (char*)malloc(lsz + 1);
                        if (log) {
                            g_tn_cuda.nvrtcGetProgramLog(prog, log);
                            log[lsz] = '\0';
                            fprintf(stderr, "[tenon-cuda] NVRTC note: %s\n", log);
                            free(log);
                        }
                    }
                }
                g_tn_cuda.nvrtcDestroyProgram(&prog);
            }
        } else if (try_nvcc) {
#if !defined(_WIN32)
            const char* nvcc_bin = "nvcc";
            if (access("/usr/local/cuda/bin/nvcc", X_OK) == 0) nvcc_bin = "/usr/local/cuda/bin/nvcc";
            char cu_path[128], out_path[128], cmd[768];
            int pid = (int)getpid();
            static int seq = 0;
            int id = ++seq;
            snprintf(cu_path, sizeof(cu_path), "/tmp/tenon_cuda_%d_%d.cu", pid, id);
            snprintf(out_path, sizeof(out_path), "/tmp/tenon_cuda_%d_%d.cubin", pid, id);
            FILE* f = fopen(cu_path, "wb");
            if (f) {
                fwrite(cu_src, 1, strlen(cu_src), f);
                fclose(f);
                int arch_num = g_tn_cuda.cc_major * 10 + g_tn_cuda.cc_minor;
                snprintf(cmd, sizeof(cmd),
                    "%s -O3 --use_fast_math -arch=sm_%d --cubin %s -o %s 2>/tmp/tenon_nvcc.err || "
                    "%s -O3 --use_fast_math -arch=compute_%d --ptx %s -o %s 2>>/tmp/tenon_nvcc.err",
                    nvcc_bin, arch_num, cu_path, out_path, nvcc_bin, arch_num, cu_path, out_path);
                int rc = system(cmd);
                if (rc == 0) {
                    FILE* pf = fopen(out_path, "rb");
                    if (pf) {
                        fseek(pf, 0, SEEK_END);
                        long sz = ftell(pf);
                        fseek(pf, 0, SEEK_SET);
                        if (sz > 0) {
                            ptx = (char*)malloc((size_t)sz + 1);
                            if (ptx) {
                                ptx_len = fread(ptx, 1, (size_t)sz, pf);
                                ptx[ptx_len] = '\0';
                            }
                        }
                        fclose(pf);
                    }
                }
                unlink(cu_path);
                unlink(out_path);
            }
#endif
        }
    }

    if (!ptx) return NULL;
    CUmodule mod = NULL;
    CUresult r = g_tn_cuda.cuModuleLoadData(&mod, ptx);
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "[tenon-cuda] cuModuleLoadData failed: err=%d\n", r);
        free(ptx);
        return NULL;
    }
    free(ptx);
    return mod;
}

int64_t tn_cuda_compile_kernel(int64_t cu_cstr, int64_t fn_cstr) {
    if (!cu_cstr || !fn_cstr) return 0;
    if (!g_tn_cuda.initialized && !tn_cuda_init()) return 0;
    int slot = -1;
    for (int i = 1; i < TN_CUDA_MAX_KERNELS; i++) {
        if (!g_tn_cuda_kernels[i].mod) { slot = i; break; }
    }
    if (slot < 0) return 0;
    CUmodule mod = tn_cuda_compile_module((const char*)(uintptr_t)cu_cstr, 0);
    if (!mod) return 0;
    CUfunction fn = NULL;
    if (g_tn_cuda.cuModuleGetFunction(&fn, mod, (const char*)(uintptr_t)fn_cstr) != CUDA_SUCCESS) {
        g_tn_cuda.cuModuleUnload(mod);
        return 0;
    }
    g_tn_cuda_kernels[slot].mod = mod;
    g_tn_cuda_kernels[slot].fn = fn;
    return (int64_t)slot;
}

int64_t tn_cuda_kernel_free(int64_t k_h) {
    if (k_h <= 0 || k_h >= TN_CUDA_MAX_KERNELS || !g_tn_cuda_kernels[k_h].mod) return 0;
    g_tn_cuda.cuModuleUnload(g_tn_cuda_kernels[k_h].mod);
    g_tn_cuda_kernels[k_h].mod = NULL;
    g_tn_cuda_kernels[k_h].fn = NULL;
    return 1;
}

int64_t tn_cuda_launch_3buf_u32(
    int64_t k_h, int64_t b0, int64_t b1, int64_t b2, int64_t n_arg,
    int64_t gx, int64_t gy, int64_t gz, int64_t bx, int64_t by, int64_t bz
) {
    if (k_h <= 0 || k_h >= TN_CUDA_MAX_KERNELS || !g_tn_cuda_kernels[k_h].fn) return 0;
    CUdeviceptr p0 = tn_cuda_buf_dptr(b0);
    CUdeviceptr p1 = tn_cuda_buf_dptr(b1);
    CUdeviceptr p2 = tn_cuda_buf_dptr(b2);
    uint32_t nu = (uint32_t)n_arg;
    void* params[4] = { &p0, &p1, &p2, &nu };
    CUresult r = g_tn_cuda.cuLaunchKernel(
        g_tn_cuda_kernels[k_h].fn,
        (unsigned int)(gx < 1 ? 1 : gx), (unsigned int)(gy < 1 ? 1 : gy), (unsigned int)(gz < 1 ? 1 : gz),
        (unsigned int)(bx < 1 ? 1 : bx), (unsigned int)(by < 1 ? 1 : by), (unsigned int)(bz < 1 ? 1 : bz),
        0, g_tn_cuda.stream, params, NULL
    );
    return r == CUDA_SUCCESS ? 1 : 0;
}

int64_t tn_cuda_sync(void) {
    if (!g_tn_cuda.initialized) return 0;
    if (g_tn_cuda.stream && g_tn_cuda.cuStreamSynchronize) {
        return g_tn_cuda.cuStreamSynchronize(g_tn_cuda.stream) == CUDA_SUCCESS ? 1 : 0;
    }
    return g_tn_cuda.cuCtxSynchronize() == CUDA_SUCCESS ? 1 : 0;
}

/* ===== Shared Internal Helpers for cuda_ultra_shim.c (Zero Struct Duplication) ===== */
CUstream tn_cuda_get_stream(void) {
    return g_tn_cuda.stream;
}

int tn_cuda_mod_get_fn(CUmodule mod, const char* name, CUfunction* out) {
    if (!mod || !name || !out || !g_tn_cuda.cuModuleGetFunction) return 0;
    return g_tn_cuda.cuModuleGetFunction(out, mod, name) == CUDA_SUCCESS ? 1 : 0;
}

int tn_cuda_launch_raw(
    CUfunction fn,
    unsigned int gx, unsigned int gy, unsigned int gz,
    unsigned int bx, unsigned int by, unsigned int bz,
    unsigned int smem_bytes,
    void** params
) {
    if (!fn || !g_tn_cuda.cuLaunchKernel) return 0;
    CUresult r = g_tn_cuda.cuLaunchKernel(
        fn,
        gx < 1 ? 1 : gx, gy < 1 ? 1 : gy, gz < 1 ? 1 : gz,
        bx < 1 ? 1 : bx, by < 1 ? 1 : by, bz < 1 ? 1 : bz,
        smem_bytes, g_tn_cuda.stream, params, NULL
    );
    return r == CUDA_SUCCESS ? 1 : 0;
}

int64_t tn_cuda_buf_write_u32_pair(int64_t h, int64_t v0, int64_t v1) {
    CUdeviceptr dp = tn_cuda_buf_dptr(h);
    if (!dp) return 0;
    uint32_t pair[2] = { (uint32_t)v0, (uint32_t)v1 };
    if (g_tn_cuda.cuMemcpyHtoDAsync && g_tn_cuda.stream) {
        return g_tn_cuda.cuMemcpyHtoDAsync(dp, pair, sizeof(pair), g_tn_cuda.stream) == CUDA_SUCCESS ? 1 : 0;
    }
    return g_tn_cuda.cuMemcpyHtoD(dp, pair, sizeof(pair)) == CUDA_SUCCESS ? 1 : 0;
}

int64_t tn_cuda_buf_read_u32_0(int64_t h) {
    CUdeviceptr dp = tn_cuda_buf_dptr(h);
    if (!dp) return 0;
    uint32_t val = 0;
    if (g_tn_cuda.cuMemcpyDtoH(&val, dp, sizeof(val)) != CUDA_SUCCESS) return 0;
    return (int64_t)val;
}

#define TN_CUDA_MAX_GRAPHS 64
static CUgraphExec g_tn_cuda_graphs[TN_CUDA_MAX_GRAPHS];

int64_t tn_cuda_graph_begin(void) {
    if (!g_tn_cuda.initialized || !g_tn_cuda.stream || !g_tn_cuda.cuStreamBeginCapture) return 0;
    return g_tn_cuda.cuStreamBeginCapture(g_tn_cuda.stream, CU_STREAM_CAPTURE_MODE_GLOBAL) == CUDA_SUCCESS ? 1 : 0;
}

int64_t tn_cuda_graph_end(void) {
    if (!g_tn_cuda.initialized || !g_tn_cuda.stream || !g_tn_cuda.cuStreamEndCapture) return 0;
    CUgraph gr = NULL;
    if (g_tn_cuda.cuStreamEndCapture(g_tn_cuda.stream, &gr) != CUDA_SUCCESS || !gr) return 0;
    CUgraphExec gexec = NULL;
    CUresult ir = 1;
    if (g_tn_cuda.cuGraphInstantiateWithFlags) {
        ir = g_tn_cuda.cuGraphInstantiateWithFlags(&gexec, gr, 0);
    } else if (g_tn_cuda.cuGraphInstantiate) {
        ir = g_tn_cuda.cuGraphInstantiate(&gexec, gr, NULL, NULL, 0);
    }
    if (g_tn_cuda.cuGraphDestroy) g_tn_cuda.cuGraphDestroy(gr);
    if (ir != CUDA_SUCCESS || !gexec) return 0;
    for (int i = 1; i < TN_CUDA_MAX_GRAPHS; i++) {
        if (!g_tn_cuda_graphs[i]) {
            g_tn_cuda_graphs[i] = gexec;
            return (int64_t)i;
        }
    }
    if (g_tn_cuda.cuGraphExecDestroy) g_tn_cuda.cuGraphExecDestroy(gexec);
    return 0;
}

int64_t tn_cuda_graph_launch(int64_t gh) {
    if (gh <= 0 || gh >= TN_CUDA_MAX_GRAPHS || !g_tn_cuda_graphs[gh] || !g_tn_cuda.cuGraphLaunch) return 0;
    return g_tn_cuda.cuGraphLaunch(g_tn_cuda_graphs[gh], g_tn_cuda.stream) == CUDA_SUCCESS ? 1 : 0;
}

int64_t tn_cuda_graph_free(int64_t gh) {
    if (gh <= 0 || gh >= TN_CUDA_MAX_GRAPHS || !g_tn_cuda_graphs[gh]) return 0;
    if (g_tn_cuda.cuGraphExecDestroy) g_tn_cuda.cuGraphExecDestroy(g_tn_cuda_graphs[gh]);
    g_tn_cuda_graphs[gh] = NULL;
    return 1;
}

/* Tensor Core FP16 GEMM via dynamically loaded cuBLAS (`Y[M, N] = X[M, K] @ W[N, K]^T + beta * Y`) */
int tn_cuda_cublas_gemm_f16(
    CUdeviceptr x_ptr, CUdeviceptr w_ptr, CUdeviceptr y_ptr,
    int m, int n, int k, int add_res
) {
    if (!g_tn_cuda.cublas || !g_tn_cuda.cublasGemmEx) return 0;
    float alpha = 1.0f;
    float beta = add_res ? 1.0f : 0.0f;
    cublasStatus_t st = g_tn_cuda.cublasGemmEx(
        g_tn_cuda.cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        (const void*)(uintptr_t)w_ptr, CUDA_R_16F, k,
        (const void*)(uintptr_t)x_ptr, CUDA_R_16F, k,
        &beta,
        (void*)(uintptr_t)y_ptr, CUDA_R_16F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    return st == CUBLAS_STATUS_SUCCESS ? 1 : 0;
}

int64_t tn_cuda_has_cublas(void) {
    return (g_tn_cuda.cublas && g_tn_cuda.cublasGemmEx) ? 1 : 0;
}
