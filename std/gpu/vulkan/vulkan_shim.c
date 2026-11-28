// std/gpu/vulkan/vulkan_shim.c — Tenon v18 Zero-Rust Cross-Vendor Vulkan 1.1+ Compute HAL (Linux / Windows / macOS MoltenVK)
//
// Zero compile-time Vulkan SDK dependency (`dlopen`/`LoadLibraryA`), probing real cross-vendor GPU capabilities:
//   - Vendor identification: Intel Arc (0x8086 XMX), AMD (0x1002 WMMA), NVIDIA (0x10DE TC), Apple (0x106B), Qualcomm (0x5143), ARM (0x13B5)
//   - Extension probing: `VK_KHR_cooperative_matrix` & `VK_KHR_shader_integer_dot_product` (cross-vendor DP4A/TC)
//   - Unified async storage IO backend via `tenon_io_hal.h`

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../common/tenon_io_hal.h"

#if defined(_WIN32)
#  define TN_VK_DLOPEN(n)    ((void*)LoadLibraryA(n))
#  define TN_VK_DLSYM(l, s)  ((void*)GetProcAddress((HMODULE)(l), (s)))
#else
#  include <dlfcn.h>
#  define TN_VK_DLOPEN(n)    dlopen((n), RTLD_LAZY)
#  define TN_VK_DLSYM(l, s)  dlsym((l), (s))
#endif

// Minimal ABI-stable Vulkan 1.1 structs for zero-SDK hardware capability probing
typedef void* TnVkInstance;
typedef void* TnVkPhysicalDevice;
typedef struct {
    uint32_t sType; const void* pNext; const char* pApplicationName; uint32_t applicationVersion;
    const char* pEngineName; uint32_t engineVersion; uint32_t apiVersion;
} TnVkAppInfo;
typedef struct {
    uint32_t sType; const void* pNext; uint32_t flags; const TnVkAppInfo* pApplicationInfo;
    uint32_t enabledLayerCount; const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount; const char* const* ppEnabledExtensionNames;
} TnVkInstanceCreateInfo;
typedef struct {
    uint32_t apiVersion; uint32_t driverVersion; uint32_t vendorID; uint32_t deviceID; uint32_t deviceType;
    char deviceName[256]; uint8_t pipelineCacheUUID[16]; uint8_t limits_pad[640];
} TnVkPhysDevProps;
typedef struct { char extensionName[256]; uint32_t specVersion; } TnVkExtProps;

typedef int  (*pfn_vkCreateInstance)(const TnVkInstanceCreateInfo*, const void*, TnVkInstance*);
typedef void (*pfn_vkDestroyInstance)(TnVkInstance, const void*);
typedef int  (*pfn_vkEnumeratePhysicalDevices)(TnVkInstance, uint32_t*, TnVkPhysicalDevice*);
typedef void (*pfn_vkGetPhysicalDeviceProperties)(TnVkPhysicalDevice, TnVkPhysDevProps*);
typedef int  (*pfn_vkEnumerateDeviceExtensionProperties)(TnVkPhysicalDevice, const char*, uint32_t*, TnVkExtProps*);

static void* g_vk_lib = NULL;
static char g_vk_device_name[256] = "Vulkan Unavailable (CPU Fallback Ready)";
static uint32_t g_vk_vendor_id = 0;
static int g_vk_has_coop_mat = 0;
static int g_vk_has_int_dot = 0;
static TnIoEngine g_vk_io;

typedef struct {
    void* data;
    size_t len;
} TnVkBuf;

#define VK_MAX_HANDLES 512
static TnVkBuf* g_vk_bufs[VK_MAX_HANDLES];

int64_t tn_vk_init(void) {
    if (g_vk_lib != NULL) return 1;
    tn_io_init(&g_vk_io, 1);
#if defined(_WIN32)
    g_vk_lib = TN_VK_DLOPEN("vulkan-1.dll");
#elif defined(__APPLE__)
    g_vk_lib = TN_VK_DLOPEN("libvulkan.1.dylib");
    if (!g_vk_lib) g_vk_lib = TN_VK_DLOPEN("libMoltenVK.dylib");
#elif defined(__linux__)
    g_vk_lib = TN_VK_DLOPEN("libvulkan.so.1");
#endif
    if (g_vk_lib == NULL) return 0;
    strncpy(g_vk_device_name, "Vulkan 1.1+ Compute Device (Zero-Rust HAL)", sizeof(g_vk_device_name) - 1);
    pfn_vkCreateInstance fn_ci = (pfn_vkCreateInstance)TN_VK_DLSYM(g_vk_lib, "vkCreateInstance");
    pfn_vkDestroyInstance fn_di = (pfn_vkDestroyInstance)TN_VK_DLSYM(g_vk_lib, "vkDestroyInstance");
    pfn_vkEnumeratePhysicalDevices fn_epd = (pfn_vkEnumeratePhysicalDevices)TN_VK_DLSYM(g_vk_lib, "vkEnumeratePhysicalDevices");
    pfn_vkGetPhysicalDeviceProperties fn_gpdp = (pfn_vkGetPhysicalDeviceProperties)TN_VK_DLSYM(g_vk_lib, "vkGetPhysicalDeviceProperties");
    pfn_vkEnumerateDeviceExtensionProperties fn_edep = (pfn_vkEnumerateDeviceExtensionProperties)TN_VK_DLSYM(g_vk_lib, "vkEnumerateDeviceExtensionProperties");
    if (fn_ci && fn_di && fn_epd && fn_gpdp) {
        TnVkAppInfo ai = { 0u, NULL, "tenon", 1u, "tenon-v18", 18u, (1u << 22) | (2u << 12) };
        TnVkInstanceCreateInfo ci = { 1u, NULL, 0u, &ai, 0u, NULL, 0u, NULL };
        TnVkInstance inst = NULL;
        if (fn_ci(&ci, NULL, &inst) == 0 && inst) {
            uint32_t nd = 0;
            if (fn_epd(inst, &nd, NULL) == 0 && nd > 0) {
                TnVkPhysicalDevice devs[16];
                if (nd > 16) nd = 16;
                if (fn_epd(inst, &nd, devs) == 0 && nd > 0) {
                    TnVkPhysDevProps pr;
                    memset(&pr, 0, sizeof(pr));
                    fn_gpdp(devs[0], &pr);
                    if (pr.deviceName[0]) {
                        strncpy(g_vk_device_name, pr.deviceName, sizeof(g_vk_device_name) - 1);
                        g_vk_device_name[sizeof(g_vk_device_name) - 1] = '\0';
                    }
                    g_vk_vendor_id = pr.vendorID;
                    if (fn_edep) {
                        uint32_t ne = 0;
                        if (fn_edep(devs[0], NULL, &ne, NULL) == 0 && ne > 0 && ne < 4096u) {
                            TnVkExtProps* exts = (TnVkExtProps*)calloc(ne, sizeof(TnVkExtProps));
                            if (exts && fn_edep(devs[0], NULL, &ne, exts) == 0) {
                                for (uint32_t i = 0; i < ne; i++) {
                                    if (strcmp(exts[i].extensionName, "VK_KHR_cooperative_matrix") == 0) g_vk_has_coop_mat = 1;
                                    if (strcmp(exts[i].extensionName, "VK_KHR_shader_integer_dot_product") == 0) g_vk_has_int_dot = 1;
                                }
                            }
                            free(exts);
                        }
                    }
                }
            }
            fn_di(inst, NULL);
        }
    }
    return 1;
}

int64_t tn_vk_device_name_cstr(void) { return (int64_t)(uintptr_t)g_vk_device_name; }

int64_t tn_vk_info(int64_t field_id) {
    switch (field_id) {
        case 0: return g_vk_lib != NULL ? 1 : 0;
        case 1: return (int64_t)g_vk_vendor_id;
        case 2: return g_vk_has_coop_mat;
        case 3: return g_vk_has_int_dot;
        case 64: return g_vk_io.backend_id;
        default: return 0;
    }
}

int64_t tn_vk_buf_create(int64_t data_ptr, int64_t byte_len) {
    if (byte_len <= 0) return 0;
    int slot = -1;
    for (int i = 1; i < VK_MAX_HANDLES; i++) {
        if (!g_vk_bufs[i]) { slot = i; break; }
    }
    if (slot < 0) return 0;
    TnVkBuf* b = (TnVkBuf*)calloc(1, sizeof(TnVkBuf));
    b->data = calloc(1, (size_t)byte_len);
    b->len = (size_t)byte_len;
    if (data_ptr != 0) memcpy(b->data, (const void*)(uintptr_t)data_ptr, (size_t)byte_len);
    g_vk_bufs[slot] = b;
    return (int64_t)slot;
}

int64_t tn_vk_buf_read(int64_t handle, int64_t dst_ptr, int64_t byte_len) {
    if (handle <= 0 || handle >= VK_MAX_HANDLES || !g_vk_bufs[handle] || dst_ptr == 0 || byte_len <= 0) return 0;
    size_t n = (size_t)byte_len;
    if (n > g_vk_bufs[handle]->len) n = g_vk_bufs[handle]->len;
    memcpy((void*)(uintptr_t)dst_ptr, g_vk_bufs[handle]->data, n);
    return (int64_t)n;
}

int64_t tn_vk_buf_free(int64_t handle) {
    if (handle <= 0 || handle >= VK_MAX_HANDLES || !g_vk_bufs[handle]) return 0;
    free(g_vk_bufs[handle]->data);
    free(g_vk_bufs[handle]);
    g_vk_bufs[handle] = NULL;
    return 1;
}
