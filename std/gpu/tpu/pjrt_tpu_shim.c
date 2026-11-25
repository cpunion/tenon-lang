// std/gpu/tpu/pjrt_tpu_shim.c — Zero-Python, Zero-Link-Flag Native Google Cloud TPU PJRT C-API Runtime
//
// Directly `dlopen`s Google's official `libtpu.so` (or any OpenXLA PJRT plugin) via `GetPjrtApi()`,
// compiles Tenon's StableHLO MLIR modules via `PJRT_Client_Compile` straight through XLA's TPU
// compiler into native TPU TensorCore MXU machine code, and executes asynchronously on TPU HBM!

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include "pjrt_c_api.h"

typedef const PJRT_Api* (*pfn_GetPjrtApi)(void);

static int g_tpu_ready = 0;
static void* g_tpu_lib = NULL;
static const PJRT_Api* g_api = NULL;
static PJRT_Client* g_client = NULL;
static PJRT_Device* g_dev0 = NULL;
static int64_t g_num_devices = 0;
static char g_platform_name[64] = "TPU Unavailable";
static char g_device_kind[128] = "No PJRT TPU Plugin Found";
static char g_platform_ver[256] = "";
static char g_last_err[512] = "";

#define TN_TPU_MAX_BUFS 4096
#define TN_TPU_MAX_EXECS 256
static PJRT_Buffer* g_tpu_bufs[TN_TPU_MAX_BUFS];
typedef struct {
    PJRT_LoadedExecutable* exec;
    size_t num_outputs;
} TnTpuExec;
static TnTpuExec g_tpu_execs[TN_TPU_MAX_EXECS];

static int tn_pjrt_check(PJRT_Error* err, const char* ctx) {
    if (!err) return 1;
    if (g_api && g_api->PJRT_Error_Message) {
        PJRT_Error_Message_Args ma;
        memset(&ma, 0, sizeof(ma));
        ma.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
        ma.error = err;
        g_api->PJRT_Error_Message(&ma);
        size_t n = ma.message_size < sizeof(g_last_err) - 1 ? ma.message_size : sizeof(g_last_err) - 1;
        memcpy(g_last_err, ma.message, n);
        g_last_err[n] = '\0';
        fprintf(stderr, "[tenon-tpu] %s failed: %s\n", ctx, g_last_err);
    } else {
        fprintf(stderr, "[tenon-tpu] %s failed (unknown PJRT error)\n", ctx);
    }
    if (g_api && g_api->PJRT_Error_Destroy) {
        PJRT_Error_Destroy_Args da;
        memset(&da, 0, sizeof(da));
        da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
        da.error = err;
        g_api->PJRT_Error_Destroy(&da);
    }
    return 0;
}

static void tn_pjrt_await_and_destroy(PJRT_Event* ev, const char* ctx) {
    if (!ev || !g_api) return;
    if (g_api->PJRT_Event_Await) {
        PJRT_Event_Await_Args aa;
        memset(&aa, 0, sizeof(aa));
        aa.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
        aa.event = ev;
        PJRT_Error* e = g_api->PJRT_Event_Await(&aa);
        tn_pjrt_check(e, ctx);
    }
    if (g_api->PJRT_Event_Destroy) {
        PJRT_Event_Destroy_Args da;
        memset(&da, 0, sizeof(da));
        da.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
        da.event = ev;
        PJRT_Error* e = g_api->PJRT_Event_Destroy(&da);
        tn_pjrt_check(e, "PJRT_Event_Destroy");
    }
}

static PJRT_Buffer_Type tn_map_dtype(int64_t dt) {
    switch (dt) {
        case 0: return PJRT_Buffer_Type_F32;
        case 1: return PJRT_Buffer_Type_BF16;
        case 2: return PJRT_Buffer_Type_F16;
        case 3: return PJRT_Buffer_Type_S32;
        case 4: return PJRT_Buffer_Type_S8;
        default: return PJRT_Buffer_Type_F32;
    }
}

static size_t tn_dtype_bytes(int64_t dt) {
    switch (dt) {
        case 0: return 4;
        case 1: return 2;
        case 2: return 2;
        case 3: return 4;
        case 4: return 1;
        default: return 4;
    }
}

static int tn_tpu_alloc_buf_slot(PJRT_Buffer* b) {
    if (!b) return 0;
    for (int i = 1; i < TN_TPU_MAX_BUFS; i++) {
        if (!g_tpu_bufs[i]) {
            g_tpu_bufs[i] = b;
            return i;
        }
    }
    return 0;
}

int64_t tn_tpu_init(void) {
    if (g_tpu_ready) return 1;
#if !defined(_WIN32)
    const char* env_path = getenv("TPU_LIBRARY_PATH");
    const char* env_pjrt = getenv("PJRT_PLUGIN_PATH");
    const char* candidates[] = {
        env_path,
        env_pjrt,
        "/usr/local/lib/python3.13/dist-packages/libtpu/libtpu.so",
        "/usr/local/lib/python3.12/dist-packages/libtpu/libtpu.so",
        "/usr/local/lib/python3.11/dist-packages/libtpu/libtpu.so",
        "/usr/local/lib/python3.10/dist-packages/libtpu/libtpu.so",
        "/lib/libtpu.so",
        "/usr/lib/libtpu.so",
        "libtpu.so",
        "pjrt_c_api_tpu_plugin.so",
        "pjrt_c_api_cpu_plugin.so",
        NULL
    };
    for (int i = 0; candidates[i] != NULL || i < 2; i++) {
        if (i >= 11) break;
        const char* path = candidates[i];
        if (!path || path[0] == '\0') continue;
        g_tpu_lib = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
        if (g_tpu_lib) break;
    }
    if (!g_tpu_lib) return 0;

    pfn_GetPjrtApi get_api = (pfn_GetPjrtApi)dlsym(g_tpu_lib, "GetPjrtApi");
    if (!get_api) return 0;
    g_api = get_api();
    if (!g_api) return 0;

    if (g_api->PJRT_Plugin_Initialize) {
        PJRT_Plugin_Initialize_Args ia;
        memset(&ia, 0, sizeof(ia));
        ia.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
        if (!tn_pjrt_check(g_api->PJRT_Plugin_Initialize(&ia), "PJRT_Plugin_Initialize")) return 0;
    }

    PJRT_Client_Create_Args ca;
    memset(&ca, 0, sizeof(ca));
    ca.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    if (!tn_pjrt_check(g_api->PJRT_Client_Create(&ca), "PJRT_Client_Create")) return 0;
    g_client = ca.client;
    if (!g_client) return 0;

    if (g_api->PJRT_Client_PlatformName) {
        PJRT_Client_PlatformName_Args pa;
        memset(&pa, 0, sizeof(pa));
        pa.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
        pa.client = g_client;
        if (tn_pjrt_check(g_api->PJRT_Client_PlatformName(&pa), "PJRT_Client_PlatformName") && pa.platform_name) {
            size_t n = pa.platform_name_size < sizeof(g_platform_name) - 1 ? pa.platform_name_size : sizeof(g_platform_name) - 1;
            memcpy(g_platform_name, pa.platform_name, n);
            g_platform_name[n] = '\0';
        }
    }
    if (g_api->PJRT_Client_PlatformVersion) {
        PJRT_Client_PlatformVersion_Args va;
        memset(&va, 0, sizeof(va));
        va.struct_size = PJRT_Client_PlatformVersion_Args_STRUCT_SIZE;
        va.client = g_client;
        if (tn_pjrt_check(g_api->PJRT_Client_PlatformVersion(&va), "PJRT_Client_PlatformVersion") && va.platform_version) {
            size_t n = va.platform_version_size < sizeof(g_platform_ver) - 1 ? va.platform_version_size : sizeof(g_platform_ver) - 1;
            memcpy(g_platform_ver, va.platform_version, n);
            g_platform_ver[n] = '\0';
        }
    }

    PJRT_Client_AddressableDevices_Args da;
    memset(&da, 0, sizeof(da));
    da.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    da.client = g_client;
    if (!tn_pjrt_check(g_api->PJRT_Client_AddressableDevices(&da), "PJRT_Client_AddressableDevices")) return 0;
    g_num_devices = (int64_t)da.num_addressable_devices;
    if (da.num_addressable_devices == 0 || !da.addressable_devices) return 0;
    g_dev0 = da.addressable_devices[0];

    if (g_api->PJRT_Device_GetDescription && g_api->PJRT_DeviceDescription_Kind) {
        PJRT_Device_GetDescription_Args gda;
        memset(&gda, 0, sizeof(gda));
        gda.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
        gda.device = g_dev0;
        if (tn_pjrt_check(g_api->PJRT_Device_GetDescription(&gda), "PJRT_Device_GetDescription") && gda.device_description) {
            PJRT_DeviceDescription_Kind_Args ka;
            memset(&ka, 0, sizeof(ka));
            ka.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
            ka.device_description = gda.device_description;
            if (tn_pjrt_check(g_api->PJRT_DeviceDescription_Kind(&ka), "PJRT_DeviceDescription_Kind") && ka.device_kind) {
                size_t n = ka.device_kind_size < sizeof(g_device_kind) - 1 ? ka.device_kind_size : sizeof(g_device_kind) - 1;
                memcpy(g_device_kind, ka.device_kind, n);
                g_device_kind[n] = '\0';
            }
        }
    }

    g_tpu_ready = 1;
    return 1;
#else
    return 0;
#endif
}

int64_t tn_tpu_platform_name_cstr(void) { return (int64_t)(uintptr_t)g_platform_name; }
int64_t tn_tpu_device_kind_cstr(void)   { return (int64_t)(uintptr_t)g_device_kind; }
int64_t tn_tpu_platform_ver_cstr(void)  { return (int64_t)(uintptr_t)g_platform_ver; }
int64_t tn_tpu_device_count(void)       { return g_num_devices; }
int64_t tn_tpu_pjrt_major(void)         { return g_api ? (int64_t)g_api->pjrt_api_version.major_version : 0; }
int64_t tn_tpu_pjrt_minor(void)         { return g_api ? (int64_t)g_api->pjrt_api_version.minor_version : 0; }

static int64_t tn_tpu_buf_create_nd(const void* data, int64_t dtype, const int64_t* dims, size_t ndim) {
    if (!g_tpu_ready && !tn_tpu_init()) return 0;
    PJRT_Client_BufferFromHostBuffer_Args ba;
    memset(&ba, 0, sizeof(ba));
    ba.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    ba.client = g_client;
    ba.data = data;
    ba.type = tn_map_dtype(dtype);
    ba.dims = dims;
    ba.num_dims = ndim;
    ba.byte_strides = NULL;
    ba.num_byte_strides = 0;
    ba.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    ba.device = g_dev0;
    ba.memory = NULL;
    ba.device_layout = NULL;
    if (!tn_pjrt_check(g_api->PJRT_Client_BufferFromHostBuffer(&ba), "PJRT_Client_BufferFromHostBuffer")) return 0;
    if (ba.done_with_host_buffer) {
        tn_pjrt_await_and_destroy(ba.done_with_host_buffer, "BufferFromHostBuffer_Await");
    }
    return (int64_t)tn_tpu_alloc_buf_slot(ba.buffer);
}

int64_t tn_tpu_buf_create_1d(int64_t data_ptr, int64_t dtype, int64_t d0) {
    int64_t dims[1] = { d0 };
    return tn_tpu_buf_create_nd((const void*)(uintptr_t)data_ptr, dtype, dims, 1);
}

int64_t tn_tpu_buf_create_2d(int64_t data_ptr, int64_t dtype, int64_t d0, int64_t d1) {
    int64_t dims[2] = { d0, d1 };
    return tn_tpu_buf_create_nd((const void*)(uintptr_t)data_ptr, dtype, dims, 2);
}

int64_t tn_tpu_buf_create_3d(int64_t data_ptr, int64_t dtype, int64_t d0, int64_t d1, int64_t d2) {
    int64_t dims[3] = { d0, d1, d2 };
    return tn_tpu_buf_create_nd((const void*)(uintptr_t)data_ptr, dtype, dims, 3);
}

int64_t tn_tpu_buf_zeros_2d(int64_t dtype, int64_t d0, int64_t d1) {
    size_t nbytes = (size_t)d0 * (size_t)d1 * tn_dtype_bytes(dtype);
    void* tmp = calloc(1, nbytes > 0 ? nbytes : 4);
    if (!tmp) return 0;
    int64_t dims[2] = { d0, d1 };
    int64_t h = tn_tpu_buf_create_nd(tmp, dtype, dims, 2);
    free(tmp);
    return h;
}

int64_t tn_tpu_buf_zeros_3d(int64_t dtype, int64_t d0, int64_t d1, int64_t d2) {
    size_t nbytes = (size_t)d0 * (size_t)d1 * (size_t)d2 * tn_dtype_bytes(dtype);
    void* tmp = calloc(1, nbytes > 0 ? nbytes : 4);
    if (!tmp) return 0;
    int64_t dims[3] = { d0, d1, d2 };
    int64_t h = tn_tpu_buf_create_nd(tmp, dtype, dims, 3);
    free(tmp);
    return h;
}

int64_t tn_tpu_buf_read(int64_t handle, int64_t dst_ptr, int64_t byte_len) {
    if (!g_tpu_ready || handle <= 0 || handle >= TN_TPU_MAX_BUFS || !g_tpu_bufs[handle] || !dst_ptr || byte_len <= 0) return 0;
    PJRT_Buffer_ToHostBuffer_Args ta;
    memset(&ta, 0, sizeof(ta));
    ta.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    ta.src = g_tpu_bufs[handle];
    ta.host_layout = NULL;
    ta.dst = (void*)(uintptr_t)dst_ptr;
    ta.dst_size = (size_t)byte_len;
    if (!tn_pjrt_check(g_api->PJRT_Buffer_ToHostBuffer(&ta), "PJRT_Buffer_ToHostBuffer")) return 0;
    if (ta.event) {
        tn_pjrt_await_and_destroy(ta.event, "ToHostBuffer_Await");
    }
    return byte_len;
}

int64_t tn_tpu_buf_free(int64_t handle) {
    if (!g_tpu_ready || handle <= 0 || handle >= TN_TPU_MAX_BUFS || !g_tpu_bufs[handle]) return 0;
    PJRT_Buffer_Destroy_Args da;
    memset(&da, 0, sizeof(da));
    da.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    da.buffer = g_tpu_bufs[handle];
    tn_pjrt_check(g_api->PJRT_Buffer_Destroy(&da), "PJRT_Buffer_Destroy");
    g_tpu_bufs[handle] = NULL;
    return 1;
}

int64_t tn_tpu_compile_mlir(int64_t mlir_cstr) {
    if (!mlir_cstr) return 0;
    if (!g_tpu_ready && !tn_tpu_init()) return 0;
    const char* code = (const char*)(uintptr_t)mlir_cstr;
    int slot = -1;
    for (int i = 1; i < TN_TPU_MAX_EXECS; i++) {
        if (!g_tpu_execs[i].exec) { slot = i; break; }
    }
    if (slot < 0) return 0;

    PJRT_Program prog;
    memset(&prog, 0, sizeof(prog));
    prog.struct_size = PJRT_Program_STRUCT_SIZE;
    prog.code = (char*)code;
    prog.code_size = strlen(code);
    prog.format = "mlir";
    prog.format_size = 4;

    PJRT_Client_Compile_Args ca;
    memset(&ca, 0, sizeof(ca));
    ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
    ca.client = g_client;
    ca.program = &prog;
    ca.compile_options = "";
    ca.compile_options_size = 0;

    if (!tn_pjrt_check(g_api->PJRT_Client_Compile(&ca), "PJRT_Client_Compile")) return 0;
    if (!ca.executable) return 0;

    size_t n_out = 1;
    if (g_api->PJRT_LoadedExecutable_GetExecutable && g_api->PJRT_Executable_NumOutputs) {
        PJRT_LoadedExecutable_GetExecutable_Args gea;
        memset(&gea, 0, sizeof(gea));
        gea.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
        gea.loaded_executable = ca.executable;
        if (tn_pjrt_check(g_api->PJRT_LoadedExecutable_GetExecutable(&gea), "GetExecutable") && gea.executable) {
            PJRT_Executable_NumOutputs_Args noa;
            memset(&noa, 0, sizeof(noa));
            noa.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
            noa.executable = gea.executable;
            if (tn_pjrt_check(g_api->PJRT_Executable_NumOutputs(&noa), "NumOutputs")) {
                n_out = noa.num_outputs;
            }
            if (g_api->PJRT_Executable_Destroy) {
                PJRT_Executable_Destroy_Args eda;
                memset(&eda, 0, sizeof(eda));
                eda.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
                eda.executable = gea.executable;
                tn_pjrt_check(g_api->PJRT_Executable_Destroy(&eda), "Executable_Destroy");
            }
        }
    }

    g_tpu_execs[slot].exec = ca.executable;
    g_tpu_execs[slot].num_outputs = n_out > 0 ? n_out : 1;
    return (int64_t)slot;
}

int64_t tn_tpu_exec_free(int64_t exec_h) {
    if (!g_tpu_ready || exec_h <= 0 || exec_h >= TN_TPU_MAX_EXECS || !g_tpu_execs[exec_h].exec) return 0;
    PJRT_LoadedExecutable_Destroy_Args da;
    memset(&da, 0, sizeof(da));
    da.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    da.executable = g_tpu_execs[exec_h].exec;
    tn_pjrt_check(g_api->PJRT_LoadedExecutable_Destroy(&da), "PJRT_LoadedExecutable_Destroy");
    g_tpu_execs[exec_h].exec = NULL;
    g_tpu_execs[exec_h].num_outputs = 0;
    return 1;
}

/* Executes a compiled StableHLO module on Google TPU MXU with arbitrary inputs; returns output 0 handle
   (and writes additional output handles if out_handles_ptr != 0). Always awaits TPU completion! */
int64_t tn_tpu_exec_many(int64_t exec_h, int64_t in_handles_ptr, int64_t n_in, int64_t out_handles_ptr, int64_t max_out) {
    if (!g_tpu_ready || exec_h <= 0 || exec_h >= TN_TPU_MAX_EXECS || !g_tpu_execs[exec_h].exec) return 0;
    size_t nin = (size_t)(n_in < 0 ? 0 : n_in);
    size_t nout = g_tpu_execs[exec_h].num_outputs;
    PJRT_Buffer* in_arr[64];
    PJRT_Buffer* out_arr[64];
    if (nin > 64 || nout > 64) return 0;
    const int64_t* ih = (const int64_t*)(uintptr_t)in_handles_ptr;
    for (size_t i = 0; i < nin; i++) {
        int64_t h = ih[i];
        if (h <= 0 || h >= TN_TPU_MAX_BUFS || !g_tpu_bufs[h]) return 0;
        in_arr[i] = g_tpu_bufs[h];
    }
    memset(out_arr, 0, sizeof(out_arr));

    PJRT_Buffer* const* arg_lists[1] = { in_arr };
    PJRT_Buffer** out_lists[1] = { out_arr };
    PJRT_Event* done_events[1] = { NULL };

    PJRT_ExecuteOptions opts;
    memset(&opts, 0, sizeof(opts));
    opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;

    PJRT_LoadedExecutable_Execute_Args ea;
    memset(&ea, 0, sizeof(ea));
    ea.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ea.executable = g_tpu_execs[exec_h].exec;
    ea.options = &opts;
    ea.argument_lists = arg_lists;
    ea.num_devices = 1;
    ea.num_args = nin;
    ea.output_lists = out_lists;
    ea.device_complete_events = done_events;
    ea.execute_device = NULL;

    if (!tn_pjrt_check(g_api->PJRT_LoadedExecutable_Execute(&ea), "PJRT_LoadedExecutable_Execute")) return 0;
    if (done_events[0]) {
        tn_pjrt_await_and_destroy(done_events[0], "Execute_Await");
    }

    int64_t first_h = 0;
    int64_t* oh = out_handles_ptr ? (int64_t*)(uintptr_t)out_handles_ptr : NULL;
    for (size_t i = 0; i < nout; i++) {
        int64_t h = (int64_t)tn_tpu_alloc_buf_slot(out_arr[i]);
        if (i == 0) first_h = h;
        if (oh && (int64_t)i < max_out) oh[i] = h;
    }
    return first_h;
}

int64_t tn_tpu_exec_2(int64_t exec_h, int64_t b0, int64_t b1) {
    int64_t ins[2] = { b0, b1 };
    return tn_tpu_exec_many(exec_h, (int64_t)(uintptr_t)ins, 2, 0, 0);
}

int64_t tn_tpu_exec_3(int64_t exec_h, int64_t b0, int64_t b1, int64_t b2) {
    int64_t ins[3] = { b0, b1, b2 };
    return tn_tpu_exec_many(exec_h, (int64_t)(uintptr_t)ins, 3, 0, 0);
}

/* Host F32 <-> BF16 fast conversion helpers for TPU native BF16 MXU systolic arrays */
int64_t tn_tpu_f32_to_bf16(int64_t src_f32_ptr, int64_t dst_bf16_ptr, int64_t n) {
    if (!src_f32_ptr || !dst_bf16_ptr || n <= 0) return 0;
    const uint32_t* src = (const uint32_t*)(uintptr_t)src_f32_ptr;
    uint16_t* dst = (uint16_t*)(uintptr_t)dst_bf16_ptr;
    for (int64_t i = 0; i < n; i++) {
        uint32_t u = src[i];
        uint32_t rounding_bias = 0x7FFFu + ((u >> 16) & 1u);
        dst[i] = (uint16_t)((u + rounding_bias) >> 16);
    }
    return n;
}

int64_t tn_tpu_bf16_to_f32(int64_t src_bf16_ptr, int64_t dst_f32_ptr, int64_t n) {
    if (!src_bf16_ptr || !dst_f32_ptr || n <= 0) return 0;
    const uint16_t* src = (const uint16_t*)(uintptr_t)src_bf16_ptr;
    uint32_t* dst = (uint32_t*)(uintptr_t)dst_f32_ptr;
    for (int64_t i = 0; i < n; i++) {
        dst[i] = ((uint32_t)src[i]) << 16;
    }
    return n;
}
