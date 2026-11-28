// std/gpu/vulkan/vulkan_shim.c — Tenon v19 Zero-SDK Cross-Vendor Vulkan 1.1–1.3 Compute HAL
//
// Zero compile-time Vulkan SDK or shader-compiler dependency (`dlopen`/`LoadLibraryA`):
//   - Works out-of-the-box on Linux (`libvulkan.so.1`), Windows (`vulkan-1.dll`), and macOS MoltenVK (`libvulkan.1.dylib`)
//   - Embedded pre-verified SPIR-V compute suite (`tenon_vk_spv.h`):
//       * Vulkan 1.1 portable (zero extensions): Raw GGUF Q4_0 GEMV, RMSNorm, SwiGLU (std + GPT-OSS), Paged FlashAttention-2
//       * Vulkan 1.3 hardware-accelerated: `VK_KHR_shader_integer_dot_product` (OpSDot DP4A Q4_0xQ8 GEMM)
//                                        & `VK_KHR_cooperative_matrix` (16x16x16 FP16 Matrix Core GEMM: Intel XMX / AMD WMMA / NVIDIA TC)
//   - Real GPU `VkBuffer` + `VkDeviceMemory` + Compute Pipeline + Descriptor + CommandBuffer dispatch
//   - Unified async storage IO backend via `tenon_io_hal.h`

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "../common/tenon_io_hal.h"
#include "tenon_vk_spv.h"

#if defined(_WIN32)
#  define TN_VK_DLOPEN(n)    ((void*)LoadLibraryA(n))
#  define TN_VK_DLSYM(l, s)  ((void*)GetProcAddress((HMODULE)(l), (s)))
#else
#  include <dlfcn.h>
#  define TN_VK_DLOPEN(n)    dlopen((n), RTLD_LAZY)
#  define TN_VK_DLSYM(l, s)  dlsym((l), (s))
#endif

// ============================================================================
// Minimal ABI-Exact Vulkan 1.1–1.3 Types (Zero Vulkan SDK Headers Required)
// ============================================================================
typedef void*    TnVkInstance;
typedef void*    TnVkPhysicalDevice;
typedef void*    TnVkDevice;
typedef void*    TnVkQueue;
typedef void*    TnVkCommandBuffer;
typedef uint64_t TnVkDeviceMemory;
typedef uint64_t TnVkBuffer;
typedef uint64_t TnVkShaderModule;
typedef uint64_t TnVkDescriptorSetLayout;
typedef uint64_t TnVkPipelineLayout;
typedef uint64_t TnVkPipeline;
typedef uint64_t TnVkDescriptorPool;
typedef uint64_t TnVkDescriptorSet;
typedef uint64_t TnVkCommandPool;
typedef uint64_t TnVkFence;
typedef void   (*TnPFN_vkVoidFunction)(void);

#define TN_VK_STYPE_APP_INFO            0u
#define TN_VK_STYPE_INSTANCE_CI         1u
#define TN_VK_STYPE_DEV_QUEUE_CI        2u
#define TN_VK_STYPE_DEVICE_CI           3u
#define TN_VK_STYPE_SUBMIT_INFO         4u
#define TN_VK_STYPE_MEM_ALLOC_INFO      5u
#define TN_VK_STYPE_BUFFER_CI           12u
#define TN_VK_STYPE_SHADER_MOD_CI       16u
#define TN_VK_STYPE_PIPE_STAGE_CI       18u
#define TN_VK_STYPE_COMPUTE_PIPE_CI     29u
#define TN_VK_STYPE_PIPE_LAYOUT_CI      30u
#define TN_VK_STYPE_DESC_LAYOUT_CI      32u
#define TN_VK_STYPE_DESC_POOL_CI        33u
#define TN_VK_STYPE_DESC_ALLOC_INFO     34u
#define TN_VK_STYPE_WRITE_DESC_SET      35u
#define TN_VK_STYPE_CMD_POOL_CI         39u
#define TN_VK_STYPE_CMD_BUF_ALLOC       40u
#define TN_VK_STYPE_CMD_BUF_BEGIN       42u
#define TN_VK_STYPE_FEAT_INT_DOT        1000280000u
#define TN_VK_STYPE_FEAT_COOP_MAT       1000506000u
#define TN_VK_STYPE_FEAT_16BIT_STOR     1000083000u
#define TN_VK_STYPE_FEAT_F16_I8         1000082000u

#define TN_VK_QUEUE_COMPUTE_BIT         0x00000002u
#define TN_VK_BUF_USAGE_STORAGE_BIT     0x00000020u
#define TN_VK_BUF_USAGE_TRANSFER_SRC    0x00000001u
#define TN_VK_BUF_USAGE_TRANSFER_DST    0x00000002u
#define TN_VK_MEM_HOST_VISIBLE          0x00000002u
#define TN_VK_MEM_HOST_COHERENT         0x00000004u
#define TN_VK_MEM_DEVICE_LOCAL          0x00000001u
#define TN_VK_DESC_STORAGE_BUFFER       7u
#define TN_VK_STAGE_COMPUTE_BIT         0x00000020u
#define TN_VK_BIND_COMPUTE              1u
#define TN_VK_CMD_POOL_RESET_BIT        0x00000002u
#define TN_VK_CMD_BEGIN_ONE_TIME_BIT    0x00000001u
#define TN_VK_INST_ENUM_PORTABILITY     0x00000001u
#define TN_VK_WHOLE_SIZE                (~0ULL)

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
    char deviceName[256]; uint8_t pipelineCacheUUID[16]; uint8_t limits_and_sparse[1024];
} TnVkPhysDevProps;
typedef struct { char extensionName[256]; uint32_t specVersion; } TnVkExtProps;
typedef struct {
    uint32_t queueFlags; uint32_t queueCount; uint32_t timestampValidBits; uint32_t minImageTransferGranularity[3];
} TnVkQueueFamilyProps;
typedef struct {
    uint32_t propertyFlags; uint32_t heapIndex;
} TnVkMemType;
typedef struct {
    uint64_t size; uint32_t flags; uint32_t _pad;
} TnVkMemHeap;
typedef struct {
    uint32_t memoryTypeCount; TnVkMemType memoryTypes[32];
    uint32_t memoryHeapCount; TnVkMemHeap memoryHeaps[16];
} TnVkPhysDevMemProps;
typedef struct {
    uint32_t sType; const void* pNext; uint32_t flags; uint32_t queueFamilyIndex;
    uint32_t queueCount; const float* pQueuePriorities;
} TnVkDeviceQueueCreateInfo;
typedef struct {
    uint32_t sType; const void* pNext; uint32_t flags;
    uint32_t queueCreateInfoCount; const TnVkDeviceQueueCreateInfo* pQueueCreateInfos;
    uint32_t enabledLayerCount; const char* const* ppEnabledLayerNames;
    uint32_t enabledExtensionCount; const char* const* ppEnabledExtensionNames;
    const void* pEnabledFeatures;
} TnVkDeviceCreateInfo;
typedef struct { uint32_t sType; void* pNext; uint32_t shaderIntegerDotProduct; } TnVkFeatIntDot;
typedef struct { uint32_t sType; void* pNext; uint32_t cooperativeMatrix; uint32_t cooperativeMatrixRobustBufferAccess; } TnVkFeatCoopMat;
typedef struct { uint32_t sType; void* pNext; uint32_t storageBuffer16BitAccess; uint32_t uniformAndStorageBuffer16BitAccess; uint32_t storagePushConstant16; uint32_t storageInputOutput16; } TnVkFeat16Bit;
typedef struct { uint32_t sType; void* pNext; uint32_t shaderFloat16; uint32_t shaderInt8; } TnVkFeatF16I8;

typedef struct {
    uint32_t sType; const void* pNext; uint32_t flags; uint64_t size;
    uint32_t usage; uint32_t sharingMode; uint32_t queueFamilyIndexCount; const uint32_t* pQueueFamilyIndices;
} TnVkBufferCreateInfo;
typedef struct {
    uint64_t size; uint64_t alignment; uint32_t memoryTypeBits; uint32_t _pad;
} TnVkMemReqs;
typedef struct {
    uint32_t sType; const void* pNext; uint64_t allocationSize; uint32_t memoryTypeIndex;
} TnVkMemAllocInfo;
typedef struct {
    uint32_t sType; const void* pNext; uint32_t flags; size_t codeSize; const uint32_t* pCode;
} TnVkShaderModCreateInfo;
typedef struct {
    uint32_t binding; uint32_t descriptorType; uint32_t descriptorCount; uint32_t stageFlags; const void* pImmutableSamplers;
} TnVkDescSetLayoutBinding;
typedef struct {
    uint32_t sType; const void* pNext; uint32_t flags; uint32_t bindingCount; const TnVkDescSetLayoutBinding* pBindings;
} TnVkDescSetLayoutCreateInfo;
typedef struct { uint32_t stageFlags; uint32_t offset; uint32_t size; } TnVkPushConstRange;
typedef struct {
    uint32_t sType; const void* pNext; uint32_t flags;
    uint32_t setLayoutCount; const TnVkDescriptorSetLayout* pSetLayouts;
    uint32_t pushConstantRangeCount; const TnVkPushConstRange* pPushConstantRanges;
} TnVkPipeLayoutCreateInfo;
typedef struct {
    uint32_t sType; const void* pNext; uint32_t flags; uint32_t stage;
    TnVkShaderModule module; const char* pName; const void* pSpecializationInfo;
} TnVkPipeShaderStageCI;
typedef struct {
    uint32_t sType; const void* pNext; uint32_t flags; TnVkPipeShaderStageCI stage;
    TnVkPipelineLayout layout; TnVkPipeline basePipelineHandle; int32_t basePipelineIndex;
} TnVkComputePipeCreateInfo;
typedef struct { uint32_t type; uint32_t descriptorCount; } TnVkDescPoolSize;
typedef struct {
    uint32_t sType; const void* pNext; uint32_t flags; uint32_t maxSets;
    uint32_t poolSizeCount; const TnVkDescPoolSize* pPoolSizes;
} TnVkDescPoolCreateInfo;
typedef struct {
    uint32_t sType; const void* pNext; TnVkDescriptorPool descriptorPool;
    uint32_t descriptorSetCount; const TnVkDescriptorSetLayout* pSetLayouts;
} TnVkDescSetAllocInfo;
typedef struct { TnVkBuffer buffer; uint64_t offset; uint64_t range; } TnVkDescBufInfo;
typedef struct {
    uint32_t sType; const void* pNext; TnVkDescriptorSet dstSet; uint32_t dstBinding;
    uint32_t dstArrayElement; uint32_t descriptorCount; uint32_t descriptorType;
    const void* pImageInfo; const TnVkDescBufInfo* pBufferInfo; const uint64_t* pTexelBufferView;
} TnVkWriteDescSet;
typedef struct { uint32_t sType; const void* pNext; uint32_t flags; uint32_t queueFamilyIndex; } TnVkCmdPoolCI;
typedef struct { uint32_t sType; const void* pNext; TnVkCommandPool commandPool; uint32_t level; uint32_t commandBufferCount; } TnVkCmdBufAllocInfo;
typedef struct { uint32_t sType; const void* pNext; uint32_t flags; const void* pInheritanceInfo; } TnVkCmdBufBeginInfo;
typedef struct {
    uint32_t sType; const void* pNext;
    uint32_t waitSemaphoreCount; const uint64_t* pWaitSemaphores; const uint32_t* pWaitDstStageMask;
    uint32_t commandBufferCount; TnVkCommandBuffer const* pCommandBuffers;
    uint32_t signalSemaphoreCount; const uint64_t* pSignalSemaphores;
} TnVkSubmitInfo;

typedef TnPFN_vkVoidFunction (*pfn_vkGetInstanceProcAddr)(TnVkInstance, const char*);
typedef TnPFN_vkVoidFunction (*pfn_vkGetDeviceProcAddr)(TnVkDevice, const char*);
typedef int  (*pfn_vkEnumerateInstanceExtensionProperties)(const char*, uint32_t*, TnVkExtProps*);
typedef int  (*pfn_vkCreateInstance)(const TnVkInstanceCreateInfo*, const void*, TnVkInstance*);
typedef void (*pfn_vkDestroyInstance)(TnVkInstance, const void*);
typedef int  (*pfn_vkEnumeratePhysicalDevices)(TnVkInstance, uint32_t*, TnVkPhysicalDevice*);
typedef void (*pfn_vkGetPhysicalDeviceProperties)(TnVkPhysicalDevice, TnVkPhysDevProps*);
typedef void (*pfn_vkGetPhysicalDeviceMemoryProperties)(TnVkPhysicalDevice, TnVkPhysDevMemProps*);
typedef void (*pfn_vkGetPhysicalDeviceQueueFamilyProperties)(TnVkPhysicalDevice, uint32_t*, TnVkQueueFamilyProps*);
typedef int  (*pfn_vkEnumerateDeviceExtensionProperties)(TnVkPhysicalDevice, const char*, uint32_t*, TnVkExtProps*);
typedef int  (*pfn_vkCreateDevice)(TnVkPhysicalDevice, const TnVkDeviceCreateInfo*, const void*, TnVkDevice*);
typedef void (*pfn_vkDestroyDevice)(TnVkDevice, const void*);
typedef void (*pfn_vkGetDeviceQueue)(TnVkDevice, uint32_t, uint32_t, TnVkQueue*);
typedef int  (*pfn_vkCreateBuffer)(TnVkDevice, const TnVkBufferCreateInfo*, const void*, TnVkBuffer*);
typedef void (*pfn_vkDestroyBuffer)(TnVkDevice, TnVkBuffer, const void*);
typedef void (*pfn_vkGetBufferMemoryRequirements)(TnVkDevice, TnVkBuffer, TnVkMemReqs*);
typedef int  (*pfn_vkAllocateMemory)(TnVkDevice, const TnVkMemAllocInfo*, const void*, TnVkDeviceMemory*);
typedef void (*pfn_vkFreeMemory)(TnVkDevice, TnVkDeviceMemory, const void*);
typedef int  (*pfn_vkBindBufferMemory)(TnVkDevice, TnVkBuffer, TnVkDeviceMemory, uint64_t);
typedef int  (*pfn_vkMapMemory)(TnVkDevice, TnVkDeviceMemory, uint64_t, uint64_t, uint32_t, void**);
typedef void (*pfn_vkUnmapMemory)(TnVkDevice, TnVkDeviceMemory);
typedef int  (*pfn_vkCreateShaderModule)(TnVkDevice, const TnVkShaderModCreateInfo*, const void*, TnVkShaderModule*);
typedef void (*pfn_vkDestroyShaderModule)(TnVkDevice, TnVkShaderModule, const void*);
typedef int  (*pfn_vkCreateDescriptorSetLayout)(TnVkDevice, const TnVkDescSetLayoutCreateInfo*, const void*, TnVkDescriptorSetLayout*);
typedef int  (*pfn_vkCreatePipelineLayout)(TnVkDevice, const TnVkPipeLayoutCreateInfo*, const void*, TnVkPipelineLayout*);
typedef int  (*pfn_vkCreateComputePipelines)(TnVkDevice, uint64_t, uint32_t, const TnVkComputePipeCreateInfo*, const void*, TnVkPipeline*);
typedef int  (*pfn_vkCreateDescriptorPool)(TnVkDevice, const TnVkDescPoolCreateInfo*, const void*, TnVkDescriptorPool*);
typedef int  (*pfn_vkResetDescriptorPool)(TnVkDevice, TnVkDescriptorPool, uint32_t);
typedef int  (*pfn_vkAllocateDescriptorSets)(TnVkDevice, const TnVkDescSetAllocInfo*, TnVkDescriptorSet*);
typedef void (*pfn_vkUpdateDescriptorSets)(TnVkDevice, uint32_t, const TnVkWriteDescSet*, uint32_t, const void*);
typedef int  (*pfn_vkCreateCommandPool)(TnVkDevice, const TnVkCmdPoolCI*, const void*, TnVkCommandPool*);
typedef int  (*pfn_vkAllocateCommandBuffers)(TnVkDevice, const TnVkCmdBufAllocInfo*, TnVkCommandBuffer*);
typedef int  (*pfn_vkResetCommandBuffer)(TnVkCommandBuffer, uint32_t);
typedef int  (*pfn_vkBeginCommandBuffer)(TnVkCommandBuffer, const TnVkCmdBufBeginInfo*);
typedef int  (*pfn_vkEndCommandBuffer)(TnVkCommandBuffer);
typedef void (*pfn_vkCmdBindPipeline)(TnVkCommandBuffer, uint32_t, TnVkPipeline);
typedef void (*pfn_vkCmdBindDescriptorSets)(TnVkCommandBuffer, uint32_t, TnVkPipelineLayout, uint32_t, uint32_t, const TnVkDescriptorSet*, uint32_t, const uint32_t*);
typedef void (*pfn_vkCmdPushConstants)(TnVkCommandBuffer, TnVkPipelineLayout, uint32_t, uint32_t, uint32_t, const void*);
typedef void (*pfn_vkCmdDispatch)(TnVkCommandBuffer, uint32_t, uint32_t, uint32_t);
typedef int  (*pfn_vkQueueSubmit)(TnVkQueue, uint32_t, const TnVkSubmitInfo*, TnVkFence);
typedef int  (*pfn_vkQueueWaitIdle)(TnVkQueue);

typedef struct {
    TnVkDescriptorSetLayout dsl;
    TnVkPipelineLayout      layout;
    TnVkPipeline            pipe;
    uint32_t                n_bindings;
    uint32_t                push_bytes;
} TnVkPipe;

typedef struct {
    void*            lib;
    int              initialized;
    int              compute_ready;
    char             device_name[256];
    uint32_t         vendor_id;
    uint32_t         api_version;
    uint64_t         vram_bytes;
    int              has_coop_mat;
    int              has_int_dot;
    int              has_f16_storage;
    int              has_portability_subset;
    int              pipelines_ready;
    uint32_t         compute_qfam;
    TnIoEngine       io;

    TnVkInstance        inst;
    TnVkPhysicalDevice  phys;
    TnVkDevice          dev;
    TnVkQueue           queue;
    TnVkCommandPool     cmd_pool;
    TnVkCommandBuffer   cmd_buf;
    TnVkDescriptorPool  desc_pool;
    TnVkPhysDevMemProps mem_props;

    pfn_vkGetInstanceProcAddr                  vkGetInstanceProcAddr;
    pfn_vkGetDeviceProcAddr                    vkGetDeviceProcAddr;
    pfn_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties;
    pfn_vkCreateInstance                       vkCreateInstance;
    pfn_vkDestroyInstance                      vkDestroyInstance;
    pfn_vkEnumeratePhysicalDevices             vkEnumeratePhysicalDevices;
    pfn_vkGetPhysicalDeviceProperties          vkGetPhysicalDeviceProperties;
    pfn_vkGetPhysicalDeviceMemoryProperties    vkGetPhysicalDeviceMemoryProperties;
    pfn_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties;
    pfn_vkEnumerateDeviceExtensionProperties   vkEnumerateDeviceExtensionProperties;
    pfn_vkCreateDevice                         vkCreateDevice;
    pfn_vkDestroyDevice                        vkDestroyDevice;
    pfn_vkGetDeviceQueue                       vkGetDeviceQueue;
    pfn_vkCreateBuffer                         vkCreateBuffer;
    pfn_vkDestroyBuffer                        vkDestroyBuffer;
    pfn_vkGetBufferMemoryRequirements          vkGetBufferMemoryRequirements;
    pfn_vkAllocateMemory                       vkAllocateMemory;
    pfn_vkFreeMemory                           vkFreeMemory;
    pfn_vkBindBufferMemory                     vkBindBufferMemory;
    pfn_vkMapMemory                            vkMapMemory;
    pfn_vkUnmapMemory                          vkUnmapMemory;
    pfn_vkCreateShaderModule                   vkCreateShaderModule;
    pfn_vkDestroyShaderModule                  vkDestroyShaderModule;
    pfn_vkCreateDescriptorSetLayout            vkCreateDescriptorSetLayout;
    pfn_vkCreatePipelineLayout                 vkCreatePipelineLayout;
    pfn_vkCreateComputePipelines               vkCreateComputePipelines;
    pfn_vkCreateDescriptorPool                 vkCreateDescriptorPool;
    pfn_vkResetDescriptorPool                  vkResetDescriptorPool;
    pfn_vkAllocateDescriptorSets               vkAllocateDescriptorSets;
    pfn_vkUpdateDescriptorSets                 vkUpdateDescriptorSets;
    pfn_vkCreateCommandPool                    vkCreateCommandPool;
    pfn_vkAllocateCommandBuffers               vkAllocateCommandBuffers;
    pfn_vkResetCommandBuffer                   vkResetCommandBuffer;
    pfn_vkBeginCommandBuffer                   vkBeginCommandBuffer;
    pfn_vkEndCommandBuffer                     vkEndCommandBuffer;
    pfn_vkCmdBindPipeline                      vkCmdBindPipeline;
    pfn_vkCmdBindDescriptorSets                vkCmdBindDescriptorSets;
    pfn_vkCmdPushConstants                     vkCmdPushConstants;
    pfn_vkCmdDispatch                          vkCmdDispatch;
    pfn_vkQueueSubmit                          vkQueueSubmit;
    pfn_vkQueueWaitIdle                        vkQueueWaitIdle;

    TnVkPipe pipe_q4_gemv;
    TnVkPipe pipe_rmsnorm;
    TnVkPipe pipe_swiglu;
    TnVkPipe pipe_flash_attn;
    TnVkPipe pipe_q4_dp4a;
    TnVkPipe pipe_fp16_coopmat;
} TnVkState;

static TnVkState g_vk = {0};

typedef struct {
    int              is_gpu;
    TnVkBuffer       buf;
    TnVkDeviceMemory mem;
    void*            data;
    size_t           len;
} TnVkBuf;

#define VK_MAX_HANDLES 512
static TnVkBuf* g_vk_bufs[VK_MAX_HANDLES];

static void* tn_vk_load_dev_sym(const char* name) {
    void* p = NULL;
    if (g_vk.vkGetDeviceProcAddr && g_vk.dev) p = (void*)g_vk.vkGetDeviceProcAddr(g_vk.dev, name);
    if (!p && g_vk.vkGetInstanceProcAddr && g_vk.inst) p = (void*)g_vk.vkGetInstanceProcAddr(g_vk.inst, name);
    if (!p && g_vk.lib) p = TN_VK_DLSYM(g_vk.lib, name);
    return p;
}

static int tn_vk_find_mem_type(uint32_t type_bits, uint32_t req_flags, uint32_t pref_flags) {
    // Pass 1: try required + preferred (e.g. HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL on UMA / ReBAR)
    for (uint32_t i = 0; i < g_vk.mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (g_vk.mem_props.memoryTypes[i].propertyFlags & (req_flags | pref_flags)) == (req_flags | pref_flags)) {
            return (int)i;
        }
    }
    // Pass 2: required flags only
    for (uint32_t i = 0; i < g_vk.mem_props.memoryTypeCount; i++) {
        if ((type_bits & (1u << i)) &&
            (g_vk.mem_props.memoryTypes[i].propertyFlags & req_flags) == req_flags) {
            return (int)i;
        }
    }
    return -1;
}

static int tn_vk_build_pipe(TnVkPipe* out, const uint32_t* spv, size_t spv_bytes, uint32_t n_bindings, uint32_t push_bytes) {
    memset(out, 0, sizeof(*out));
    if (!g_vk.vkCreateShaderModule || !g_vk.vkCreateDescriptorSetLayout ||
        !g_vk.vkCreatePipelineLayout || !g_vk.vkCreateComputePipelines) return 0;

    TnVkShaderModCreateInfo smci = { TN_VK_STYPE_SHADER_MOD_CI, NULL, 0u, spv_bytes, spv };
    TnVkShaderModule sm = 0;
    if (g_vk.vkCreateShaderModule(g_vk.dev, &smci, NULL, &sm) != 0 || sm == 0) return 0;

    TnVkDescSetLayoutBinding binds[8];
    for (uint32_t i = 0; i < n_bindings && i < 8u; i++) {
        binds[i].binding            = i;
        binds[i].descriptorType     = TN_VK_DESC_STORAGE_BUFFER;
        binds[i].descriptorCount    = 1u;
        binds[i].stageFlags         = TN_VK_STAGE_COMPUTE_BIT;
        binds[i].pImmutableSamplers = NULL;
    }
    TnVkDescSetLayoutCreateInfo dslci = { TN_VK_STYPE_DESC_LAYOUT_CI, NULL, 0u, n_bindings, binds };
    if (g_vk.vkCreateDescriptorSetLayout(g_vk.dev, &dslci, NULL, &out->dsl) != 0) {
        g_vk.vkDestroyShaderModule(g_vk.dev, sm, NULL);
        return 0;
    }

    TnVkPushConstRange pcr = { TN_VK_STAGE_COMPUTE_BIT, 0u, push_bytes };
    TnVkPipeLayoutCreateInfo plci = {
        TN_VK_STYPE_PIPE_LAYOUT_CI, NULL, 0u, 1u, &out->dsl,
        push_bytes > 0u ? 1u : 0u, push_bytes > 0u ? &pcr : NULL
    };
    if (g_vk.vkCreatePipelineLayout(g_vk.dev, &plci, NULL, &out->layout) != 0) {
        g_vk.vkDestroyShaderModule(g_vk.dev, sm, NULL);
        return 0;
    }

    TnVkComputePipeCreateInfo cpci;
    memset(&cpci, 0, sizeof(cpci));
    cpci.sType = TN_VK_STYPE_COMPUTE_PIPE_CI;
    cpci.stage.sType = TN_VK_STYPE_PIPE_STAGE_CI;
    cpci.stage.stage = TN_VK_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm;
    cpci.stage.pName = "main";
    cpci.layout = out->layout;
    int rc = g_vk.vkCreateComputePipelines(g_vk.dev, 0, 1u, &cpci, NULL, &out->pipe);
    g_vk.vkDestroyShaderModule(g_vk.dev, sm, NULL);
    if (rc != 0 || out->pipe == 0) {
        memset(out, 0, sizeof(*out));
        return 0;
    }
    out->n_bindings = n_bindings;
    out->push_bytes = push_bytes;
    return 1;
}

static int tn_vk_dispatch(
    const TnVkPipe* p, const int64_t* buf_handles, uint32_t n_bufs,
    const void* push_data, uint32_t push_bytes,
    uint32_t gx, uint32_t gy, uint32_t gz) {
    if (!g_vk.compute_ready || !p || !p->pipe || n_bufs != p->n_bindings || n_bufs > 8u) return 0;

    TnVkDescBufInfo dbis[8];
    for (uint32_t i = 0; i < n_bufs; i++) {
        int64_t h = buf_handles[i];
        if (h <= 0 || h >= VK_MAX_HANDLES || !g_vk_bufs[h] || !g_vk_bufs[h]->is_gpu) return 0;
        dbis[i].buffer = g_vk_bufs[h]->buf;
        dbis[i].offset = 0;
        dbis[i].range  = TN_VK_WHOLE_SIZE;
    }

    if (g_vk.vkResetDescriptorPool(g_vk.dev, g_vk.desc_pool, 0u) != 0) return 0;
    TnVkDescSetAllocInfo dsai = { TN_VK_STYPE_DESC_ALLOC_INFO, NULL, g_vk.desc_pool, 1u, &p->dsl };
    TnVkDescriptorSet dset = 0;
    if (g_vk.vkAllocateDescriptorSets(g_vk.dev, &dsai, &dset) != 0 || dset == 0) return 0;

    TnVkWriteDescSet writes[8];
    for (uint32_t i = 0; i < n_bufs; i++) {
        memset(&writes[i], 0, sizeof(writes[i]));
        writes[i].sType           = TN_VK_STYPE_WRITE_DESC_SET;
        writes[i].dstSet          = dset;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1u;
        writes[i].descriptorType  = TN_VK_DESC_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &dbis[i];
    }
    g_vk.vkUpdateDescriptorSets(g_vk.dev, n_bufs, writes, 0u, NULL);

    if (g_vk.vkResetCommandBuffer(g_vk.cmd_buf, 0u) != 0) return 0;
    TnVkCmdBufBeginInfo bi = { TN_VK_STYPE_CMD_BUF_BEGIN, NULL, TN_VK_CMD_BEGIN_ONE_TIME_BIT, NULL };
    if (g_vk.vkBeginCommandBuffer(g_vk.cmd_buf, &bi) != 0) return 0;
    g_vk.vkCmdBindPipeline(g_vk.cmd_buf, TN_VK_BIND_COMPUTE, p->pipe);
    g_vk.vkCmdBindDescriptorSets(g_vk.cmd_buf, TN_VK_BIND_COMPUTE, p->layout, 0u, 1u, &dset, 0u, NULL);
    if (push_data && push_bytes > 0u) {
        g_vk.vkCmdPushConstants(g_vk.cmd_buf, p->layout, TN_VK_STAGE_COMPUTE_BIT, 0u, push_bytes, push_data);
    }
    g_vk.vkCmdDispatch(g_vk.cmd_buf, gx, gy, gz);
    if (g_vk.vkEndCommandBuffer(g_vk.cmd_buf) != 0) return 0;

    TnVkSubmitInfo si;
    memset(&si, 0, sizeof(si));
    si.sType              = TN_VK_STYPE_SUBMIT_INFO;
    si.commandBufferCount = 1u;
    si.pCommandBuffers    = &g_vk.cmd_buf;
    if (g_vk.vkQueueSubmit(g_vk.queue, 1u, &si, 0) != 0) return 0;
    return g_vk.vkQueueWaitIdle(g_vk.queue) == 0 ? 1 : 0;
}

int64_t tn_vk_init(void) {
    if (g_vk.initialized) return g_vk.lib != NULL ? 1 : 0;
    g_vk.initialized = 1;
    tn_io_init(&g_vk.io, 1);
    strncpy(g_vk.device_name, "Vulkan Unavailable (CPU Fallback Ready)", sizeof(g_vk.device_name) - 1);

#if defined(_WIN32)
    g_vk.lib = TN_VK_DLOPEN("vulkan-1.dll");
#elif defined(__APPLE__)
    g_vk.lib = TN_VK_DLOPEN("libvulkan.1.dylib");
    if (!g_vk.lib) g_vk.lib = TN_VK_DLOPEN("/opt/homebrew/lib/libvulkan.1.dylib");
    if (!g_vk.lib) g_vk.lib = TN_VK_DLOPEN("/usr/local/lib/libvulkan.1.dylib");
    if (!g_vk.lib) g_vk.lib = TN_VK_DLOPEN("libMoltenVK.dylib");
    if (!g_vk.lib) g_vk.lib = TN_VK_DLOPEN("/opt/homebrew/lib/libMoltenVK.dylib");
#elif defined(__linux__)
    g_vk.lib = TN_VK_DLOPEN("libvulkan.so.1");
    if (!g_vk.lib) g_vk.lib = TN_VK_DLOPEN("libvulkan.so");
#endif
    if (g_vk.lib == NULL) return 0;

    #define VSYM(f) g_vk.f = (pfn_##f)TN_VK_DLSYM(g_vk.lib, #f)
    VSYM(vkGetInstanceProcAddr);
    VSYM(vkGetDeviceProcAddr);
    VSYM(vkEnumerateInstanceExtensionProperties);
    VSYM(vkCreateInstance);
    VSYM(vkDestroyInstance);
    VSYM(vkEnumeratePhysicalDevices);
    VSYM(vkGetPhysicalDeviceProperties);
    VSYM(vkGetPhysicalDeviceMemoryProperties);
    VSYM(vkGetPhysicalDeviceQueueFamilyProperties);
    VSYM(vkEnumerateDeviceExtensionProperties);
    VSYM(vkCreateDevice);
    VSYM(vkDestroyDevice);
    #undef VSYM

    if (!g_vk.vkCreateInstance || !g_vk.vkEnumeratePhysicalDevices || !g_vk.vkGetPhysicalDeviceProperties) return 1;

    // Detect VK_KHR_portability_enumeration (required by Vulkan Loader 1.3+ on macOS MoltenVK!)
    int has_port_enum = 0;
    if (g_vk.vkEnumerateInstanceExtensionProperties) {
        uint32_t nie = 0;
        if (g_vk.vkEnumerateInstanceExtensionProperties(NULL, &nie, NULL) == 0 && nie > 0 && nie < 1024u) {
            TnVkExtProps* iexts = (TnVkExtProps*)calloc(nie, sizeof(TnVkExtProps));
            if (iexts && g_vk.vkEnumerateInstanceExtensionProperties(NULL, &nie, iexts) == 0) {
                for (uint32_t i = 0; i < nie; i++) {
                    if (strcmp(iexts[i].extensionName, "VK_KHR_portability_enumeration") == 0) has_port_enum = 1;
                }
            }
            free(iexts);
        }
    }

    const char* inst_exts[1] = { "VK_KHR_portability_enumeration" };
    // Try Vulkan 1.3 first, fall back to 1.2 / 1.1 for older drivers
    uint32_t versions[3] = { (1u << 22) | (3u << 12), (1u << 22) | (2u << 12), (1u << 22) | (1u << 12) };
    for (int vi = 0; vi < 3 && !g_vk.inst; vi++) {
        TnVkAppInfo ai = { TN_VK_STYPE_APP_INFO, NULL, "tenon", 19u, "tenon-v19", 19u, versions[vi] };
        TnVkInstanceCreateInfo ci = {
            TN_VK_STYPE_INSTANCE_CI, NULL,
            has_port_enum ? TN_VK_INST_ENUM_PORTABILITY : 0u,
            &ai, 0u, NULL,
            has_port_enum ? 1u : 0u, has_port_enum ? inst_exts : NULL
        };
        if (g_vk.vkCreateInstance(&ci, NULL, &g_vk.inst) != 0) g_vk.inst = NULL;
    }
    if (!g_vk.inst) return 1;

    uint32_t nd = 0;
    if (g_vk.vkEnumeratePhysicalDevices(g_vk.inst, &nd, NULL) != 0 || nd == 0) return 1;
    TnVkPhysicalDevice devs[16];
    if (nd > 16u) nd = 16u;
    if (g_vk.vkEnumeratePhysicalDevices(g_vk.inst, &nd, devs) != 0 || nd == 0) return 1;

    // Prefer discrete GPU > integrated GPU > other
    uint32_t best_idx = 0;
    int best_rank = -1;
    TnVkPhysDevProps best_pr;
    memset(&best_pr, 0, sizeof(best_pr));
    for (uint32_t i = 0; i < nd; i++) {
        TnVkPhysDevProps pr;
        memset(&pr, 0, sizeof(pr));
        g_vk.vkGetPhysicalDeviceProperties(devs[i], &pr);
        int rank = (pr.deviceType == 2u) ? 3 : ((pr.deviceType == 1u) ? 2 : 1);
        if (rank > best_rank) { best_rank = rank; best_idx = i; best_pr = pr; }
    }
    g_vk.phys = devs[best_idx];
    if (best_pr.deviceName[0]) {
        strncpy(g_vk.device_name, best_pr.deviceName, sizeof(g_vk.device_name) - 1);
        g_vk.device_name[sizeof(g_vk.device_name) - 1] = '\0';
    }
    g_vk.vendor_id   = best_pr.vendorID;
    g_vk.api_version = best_pr.apiVersion;

    if (g_vk.vkGetPhysicalDeviceMemoryProperties) {
        g_vk.vkGetPhysicalDeviceMemoryProperties(g_vk.phys, &g_vk.mem_props);
        for (uint32_t i = 0; i < g_vk.mem_props.memoryHeapCount; i++) {
            if (g_vk.mem_props.memoryHeaps[i].flags & 1u) {
                if (g_vk.mem_props.memoryHeaps[i].size > g_vk.vram_bytes) {
                    g_vk.vram_bytes = g_vk.mem_props.memoryHeaps[i].size;
                }
            }
        }
    }

    int has_16bit_ext = 0, has_f16i8_ext = 0;
    if (g_vk.vkEnumerateDeviceExtensionProperties) {
        uint32_t ne = 0;
        if (g_vk.vkEnumerateDeviceExtensionProperties(g_vk.phys, NULL, &ne, NULL) == 0 && ne > 0 && ne < 4096u) {
            TnVkExtProps* exts = (TnVkExtProps*)calloc(ne, sizeof(TnVkExtProps));
            if (exts && g_vk.vkEnumerateDeviceExtensionProperties(g_vk.phys, NULL, &ne, exts) == 0) {
                for (uint32_t i = 0; i < ne; i++) {
                    if (strcmp(exts[i].extensionName, "VK_KHR_cooperative_matrix") == 0) g_vk.has_coop_mat = 1;
                    if (strcmp(exts[i].extensionName, "VK_KHR_shader_integer_dot_product") == 0) g_vk.has_int_dot = 1;
                    if (strcmp(exts[i].extensionName, "VK_KHR_16bit_storage") == 0) has_16bit_ext = 1;
                    if (strcmp(exts[i].extensionName, "VK_KHR_shader_float16_int8") == 0) has_f16i8_ext = 1;
                    if (strcmp(exts[i].extensionName, "VK_KHR_portability_subset") == 0) g_vk.has_portability_subset = 1;
                }
            }
            free(exts);
        }
    }
    g_vk.has_f16_storage = (has_16bit_ext && has_f16i8_ext) ? 1 : 0;

    // Find a compute-capable queue family
    if (!g_vk.vkGetPhysicalDeviceQueueFamilyProperties || !g_vk.vkCreateDevice) return 1;
    uint32_t nqf = 0;
    g_vk.vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys, &nqf, NULL);
    if (nqf == 0 || nqf > 64u) return 1;
    TnVkQueueFamilyProps qfs[64];
    g_vk.vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys, &nqf, qfs);
    int found_qf = -1;
    for (uint32_t i = 0; i < nqf; i++) {
        if (qfs[i].queueCount > 0 && (qfs[i].queueFlags & TN_VK_QUEUE_COMPUTE_BIT)) { found_qf = (int)i; break; }
    }
    if (found_qf < 0) return 1;
    g_vk.compute_qfam = (uint32_t)found_qf;

    // Create logical Vulkan compute device (with optional portability_subset + Vulkan 1.3 DP4A / CoopMat features)
    float q_prio = 1.0f;
    TnVkDeviceQueueCreateInfo qci = { TN_VK_STYPE_DEV_QUEUE_CI, NULL, 0u, g_vk.compute_qfam, 1u, &q_prio };
    const char* dev_exts[8];
    uint32_t n_dev_exts = 0;
    void* pnext_chain = NULL;
    TnVkFeatIntDot  f_dot  = { TN_VK_STYPE_FEAT_INT_DOT, NULL, 1u };
    TnVkFeatCoopMat f_cm   = { TN_VK_STYPE_FEAT_COOP_MAT, NULL, 1u, 0u };
    TnVkFeat16Bit   f_16b  = { TN_VK_STYPE_FEAT_16BIT_STOR, NULL, 1u, 0u, 0u, 0u };
    TnVkFeatF16I8   f_f16  = { TN_VK_STYPE_FEAT_F16_I8, NULL, 1u, 1u };

    if (g_vk.has_portability_subset) dev_exts[n_dev_exts++] = "VK_KHR_portability_subset";
    int is_vk13 = (((g_vk.api_version >> 22) & 0x7Fu) >= 1u && ((g_vk.api_version >> 12) & 0x3FFu) >= 3u) ? 1 : 0;
    if (is_vk13 && g_vk.has_int_dot) {
        dev_exts[n_dev_exts++] = "VK_KHR_shader_integer_dot_product";
        f_dot.pNext = pnext_chain; pnext_chain = &f_dot;
    }
    if (is_vk13 && g_vk.has_coop_mat && g_vk.has_f16_storage) {
        dev_exts[n_dev_exts++] = "VK_KHR_cooperative_matrix";
        dev_exts[n_dev_exts++] = "VK_KHR_16bit_storage";
        dev_exts[n_dev_exts++] = "VK_KHR_shader_float16_int8";
        f_16b.pNext = pnext_chain; pnext_chain = &f_16b;
        f_f16.pNext = pnext_chain; pnext_chain = &f_f16;
        f_cm.pNext  = pnext_chain; pnext_chain = &f_cm;
    }

    TnVkDeviceCreateInfo dci;
    memset(&dci, 0, sizeof(dci));
    dci.sType                   = TN_VK_STYPE_DEVICE_CI;
    dci.pNext                   = pnext_chain;
    dci.queueCreateInfoCount    = 1u;
    dci.pQueueCreateInfos       = &qci;
    dci.enabledExtensionCount   = n_dev_exts;
    dci.ppEnabledExtensionNames = n_dev_exts > 0 ? dev_exts : NULL;

    if (g_vk.vkCreateDevice(g_vk.phys, &dci, NULL, &g_vk.dev) != 0) {
        // Fallback: create portable device with only portability_subset (if needed) and zero extra extensions
        dci.pNext = NULL;
        n_dev_exts = 0;
        if (g_vk.has_portability_subset) dev_exts[n_dev_exts++] = "VK_KHR_portability_subset";
        dci.enabledExtensionCount   = n_dev_exts;
        dci.ppEnabledExtensionNames = n_dev_exts > 0 ? dev_exts : NULL;
        g_vk.has_int_dot  = 0;
        g_vk.has_coop_mat = 0;
        if (g_vk.vkCreateDevice(g_vk.phys, &dci, NULL, &g_vk.dev) != 0) {
            g_vk.dev = NULL;
            return 1;
        }
    }

    #define DSYM(f) g_vk.f = (pfn_##f)tn_vk_load_dev_sym(#f)
    DSYM(vkGetDeviceQueue); DSYM(vkCreateBuffer); DSYM(vkDestroyBuffer);
    DSYM(vkGetBufferMemoryRequirements); DSYM(vkAllocateMemory); DSYM(vkFreeMemory);
    DSYM(vkBindBufferMemory); DSYM(vkMapMemory); DSYM(vkUnmapMemory);
    DSYM(vkCreateShaderModule); DSYM(vkDestroyShaderModule);
    DSYM(vkCreateDescriptorSetLayout); DSYM(vkCreatePipelineLayout); DSYM(vkCreateComputePipelines);
    DSYM(vkCreateDescriptorPool); DSYM(vkResetDescriptorPool);
    DSYM(vkAllocateDescriptorSets); DSYM(vkUpdateDescriptorSets);
    DSYM(vkCreateCommandPool); DSYM(vkAllocateCommandBuffers);
    DSYM(vkResetCommandBuffer); DSYM(vkBeginCommandBuffer); DSYM(vkEndCommandBuffer);
    DSYM(vkCmdBindPipeline); DSYM(vkCmdBindDescriptorSets); DSYM(vkCmdPushConstants); DSYM(vkCmdDispatch);
    DSYM(vkQueueSubmit); DSYM(vkQueueWaitIdle);
    #undef DSYM

    if (!g_vk.vkGetDeviceQueue || !g_vk.vkCreateCommandPool || !g_vk.vkAllocateCommandBuffers ||
        !g_vk.vkCreateDescriptorPool || !g_vk.vkQueueSubmit || !g_vk.vkQueueWaitIdle) return 1;

    g_vk.vkGetDeviceQueue(g_vk.dev, g_vk.compute_qfam, 0u, &g_vk.queue);

    TnVkCmdPoolCI cpci = { TN_VK_STYPE_CMD_POOL_CI, NULL, TN_VK_CMD_POOL_RESET_BIT, g_vk.compute_qfam };
    if (g_vk.vkCreateCommandPool(g_vk.dev, &cpci, NULL, &g_vk.cmd_pool) != 0) return 1;

    TnVkCmdBufAllocInfo cbai = { TN_VK_STYPE_CMD_BUF_ALLOC, NULL, g_vk.cmd_pool, 0u, 1u };
    if (g_vk.vkAllocateCommandBuffers(g_vk.dev, &cbai, &g_vk.cmd_buf) != 0) return 1;

    TnVkDescPoolSize dps = { TN_VK_DESC_STORAGE_BUFFER, 64u };
    TnVkDescPoolCreateInfo dpci = { TN_VK_STYPE_DESC_POOL_CI, NULL, 0u, 16u, 1u, &dps };
    if (g_vk.vkCreateDescriptorPool(g_vk.dev, &dpci, NULL, &g_vk.desc_pool) != 0) return 1;

    // Build all portable Vulkan 1.1 compute pipelines (always available everywhere!)
    int n_pipes = 0;
    if (tn_vk_build_pipe(&g_vk.pipe_q4_gemv,    kSpv_q4_0_gemv,        kSpv_q4_0_gemv_size,        3u, 12u)) n_pipes++;
    if (tn_vk_build_pipe(&g_vk.pipe_rmsnorm,    kSpv_rmsnorm,          kSpv_rmsnorm_size,          3u, 12u)) n_pipes++;
    if (tn_vk_build_pipe(&g_vk.pipe_swiglu,     kSpv_swiglu,           kSpv_swiglu_size,           3u, 16u)) n_pipes++;
    if (tn_vk_build_pipe(&g_vk.pipe_flash_attn, kSpv_flash_attn_paged, kSpv_flash_attn_paged_size, 5u, 32u)) n_pipes++;
    // Build optional Vulkan 1.3 hardware DP4A & Cooperative Matrix pipelines when device supports them!
    if (is_vk13 && g_vk.has_int_dot) {
        if (tn_vk_build_pipe(&g_vk.pipe_q4_dp4a, kSpv_q4_0_gemm_dp4a, kSpv_q4_0_gemm_dp4a_size, 4u, 12u)) n_pipes++;
    }
    if (is_vk13 && g_vk.has_coop_mat && g_vk.has_f16_storage) {
        if (tn_vk_build_pipe(&g_vk.pipe_fp16_coopmat, kSpv_fp16_gemm_coopmat, kSpv_fp16_gemm_coopmat_size, 3u, 12u)) n_pipes++;
    }
    g_vk.pipelines_ready = n_pipes;
    g_vk.compute_ready   = (n_pipes >= 4) ? 1 : 0;
    return 1;
}

int64_t tn_vk_device_name_cstr(void) { return (int64_t)(uintptr_t)g_vk.device_name; }

int64_t tn_vk_info(int64_t field_id) {
    switch (field_id) {
        case 0:  return g_vk.lib != NULL ? 1 : 0;
        case 1:  return (int64_t)g_vk.vendor_id;
        case 2:  return g_vk.has_coop_mat;
        case 3:  return g_vk.has_int_dot;
        case 4:  return g_vk.compute_ready;
        case 5:  return g_vk.pipelines_ready;
        case 6:  return (int64_t)g_vk.vram_bytes;
        case 7:  return (int64_t)g_vk.api_version;
        case 8:  return g_vk.pipe_q4_dp4a.pipe != 0 ? 1 : 0;
        case 9:  return g_vk.pipe_fp16_coopmat.pipe != 0 ? 1 : 0;
        case 64: return g_vk.io.backend_id;
        case 65: return g_vk.io.total_bytes;
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
    if (!b) return 0;
    b->len = (size_t)byte_len;
    // Pad buffer allocation to 4-byte multiple so uint32 SSBO reads never cross end-of-buffer
    size_t alloc_len = (b->len + 3u) & ~(size_t)3u;

    if (g_vk.compute_ready && g_vk.vkCreateBuffer && g_vk.vkAllocateMemory && g_vk.vkMapMemory) {
        TnVkBufferCreateInfo bci = {
            TN_VK_STYPE_BUFFER_CI, NULL, 0u, (uint64_t)alloc_len,
            TN_VK_BUF_USAGE_STORAGE_BIT | TN_VK_BUF_USAGE_TRANSFER_SRC | TN_VK_BUF_USAGE_TRANSFER_DST,
            0u, 0u, NULL
        };
        if (g_vk.vkCreateBuffer(g_vk.dev, &bci, NULL, &b->buf) == 0 && b->buf != 0) {
            TnVkMemReqs mr;
            memset(&mr, 0, sizeof(mr));
            g_vk.vkGetBufferMemoryRequirements(g_vk.dev, b->buf, &mr);
            int mtype = tn_vk_find_mem_type(mr.memoryTypeBits,
                                            TN_VK_MEM_HOST_VISIBLE | TN_VK_MEM_HOST_COHERENT,
                                            TN_VK_MEM_DEVICE_LOCAL);
            if (mtype >= 0) {
                TnVkMemAllocInfo mai = { TN_VK_STYPE_MEM_ALLOC_INFO, NULL, mr.size, (uint32_t)mtype };
                if (g_vk.vkAllocateMemory(g_vk.dev, &mai, NULL, &b->mem) == 0 && b->mem != 0) {
                    if (g_vk.vkBindBufferMemory(g_vk.dev, b->buf, b->mem, 0) == 0 &&
                        g_vk.vkMapMemory(g_vk.dev, b->mem, 0, mr.size, 0u, &b->data) == 0 && b->data) {
                        b->is_gpu = 1;
                        memset(b->data, 0, alloc_len);
                        if (data_ptr != 0) memcpy(b->data, (const void*)(uintptr_t)data_ptr, b->len);
                        g_vk_bufs[slot] = b;
                        return (int64_t)slot;
                    }
                    g_vk.vkFreeMemory(g_vk.dev, b->mem, NULL);
                }
            }
            g_vk.vkDestroyBuffer(g_vk.dev, b->buf, NULL);
        }
        b->buf = 0; b->mem = 0; b->data = NULL; b->is_gpu = 0;
    }

    b->data = calloc(1, alloc_len);
    if (!b->data) { free(b); return 0; }
    if (data_ptr != 0) memcpy(b->data, (const void*)(uintptr_t)data_ptr, b->len);
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
    TnVkBuf* b = g_vk_bufs[handle];
    if (b->is_gpu && g_vk.dev) {
        if (b->mem && g_vk.vkUnmapMemory) g_vk.vkUnmapMemory(g_vk.dev, b->mem);
        if (b->buf && g_vk.vkDestroyBuffer) g_vk.vkDestroyBuffer(g_vk.dev, b->buf, NULL);
        if (b->mem && g_vk.vkFreeMemory) g_vk.vkFreeMemory(g_vk.dev, b->mem, NULL);
    } else {
        free(b->data);
    }
    free(b);
    g_vk_bufs[handle] = NULL;
    return 1;
}

// ============================================================================
// Public Real Vulkan GPU Compute Operator Dispatch API
// ============================================================================
int64_t tn_vk_q4_gemv(int64_t w_buf, int64_t x_buf, int64_t y_buf, int64_t M, int64_t K, int64_t B) {
    if (!g_vk.compute_ready || M <= 0 || K <= 0 || B <= 0) return 0;
    struct { uint32_t M, K, B; } pc = { (uint32_t)M, (uint32_t)K, (uint32_t)B };
    int64_t bufs[3] = { w_buf, x_buf, y_buf };
    return tn_vk_dispatch(&g_vk.pipe_q4_gemv, bufs, 3u, &pc, sizeof(pc), (uint32_t)M, (uint32_t)B, 1u);
}

int64_t tn_vk_rmsnorm(int64_t x_buf, int64_t w_buf, int64_t y_buf, int64_t B, int64_t H, double eps) {
    if (!g_vk.compute_ready || B <= 0 || H <= 0) return 0;
    struct { uint32_t B, H; float eps; } pc = { (uint32_t)B, (uint32_t)H, (float)eps };
    int64_t bufs[3] = { x_buf, w_buf, y_buf };
    return tn_vk_dispatch(&g_vk.pipe_rmsnorm, bufs, 3u, &pc, sizeof(pc), (uint32_t)B, 1u, 1u);
}

int64_t tn_vk_swiglu(int64_t g_buf, int64_t u_buf, int64_t o_buf, int64_t N, int64_t oai_mode) {
    if (!g_vk.compute_ready || N <= 0) return 0;
    struct { uint32_t N; float alpha, clamp_lim; uint32_t oai_mode; } pc = {
        (uint32_t)N, 1.702f, 7.0f, (uint32_t)(oai_mode != 0 ? 1u : 0u)
    };
    int64_t bufs[3] = { g_buf, u_buf, o_buf };
    return tn_vk_dispatch(&g_vk.pipe_swiglu, bufs, 3u, &pc, sizeof(pc), ((uint32_t)N + 63u) / 64u, 1u, 1u);
}

int64_t tn_vk_flash_attn_paged(
    int64_t q_buf, int64_t kc_buf, int64_t vc_buf, int64_t bt_buf, int64_t o_buf,
    int64_t B, int64_t n_qh, int64_t n_kvh, int64_t hd, int64_t kv_base_pos, int64_t page_sz,
    double scale, int64_t swa_win) {
    if (!g_vk.compute_ready || B <= 0 || n_qh <= 0 || hd <= 0) return 0;
    struct {
        uint32_t B, n_qh, n_kvh, hd, kv_base_pos, page_sz;
        float scale;
        uint32_t swa_win;
    } pc = {
        (uint32_t)B, (uint32_t)n_qh, (uint32_t)n_kvh, (uint32_t)hd,
        (uint32_t)kv_base_pos, (uint32_t)(page_sz > 0 ? page_sz : 4),
        (float)scale, (uint32_t)(swa_win > 0 ? swa_win : 0)
    };
    int64_t bufs[5] = { q_buf, kc_buf, vc_buf, bt_buf, o_buf };
    return tn_vk_dispatch(&g_vk.pipe_flash_attn, bufs, 5u, &pc, sizeof(pc), (uint32_t)n_qh, (uint32_t)B, 1u);
}

static uint16_t tn_vk_f32_to_f16(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    uint32_t s = (u >> 16) & 0x8000u;
    int e = (int)((u >> 23) & 0xFFu) - 112;
    uint32_t m = u & 0x007FFFFFu;
    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7BFFu);
    return (uint16_t)(s | ((uint32_t)e << 10) | ((m + 0x1000u) >> 13));
}

static float tn_vk_f16_to_f32(uint16_t h) {
    uint32_t s = ((uint32_t)(h & 0x8000u)) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x03FFu;
    uint32_t u = (e == 0u) ? s : (s | ((e + 112u) << 23) | (m << 13));
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// ============================================================================
// End-to-End Live Vulkan GPU Compute Self-Test (Q4_0 GEMV + RMSNorm + SwiGLU + Paged FA2)
// ============================================================================
int64_t tn_vk_self_test(void) {
    if (!tn_vk_init() || !g_vk.compute_ready) return 0;

    // 1. Raw GGUF Q4_0 (18B/block) x FP32 GEMV test (M=64, K=512, B=2)
    const uint32_t M = 64u, K = 512u, B = 2u, NB = K / 32u;
    size_t w_sz = (size_t)M * NB * 18u;
    uint8_t* h_w = (uint8_t*)calloc(1, w_sz);
    float*   h_x = (float*)calloc((size_t)B * K, sizeof(float));
    float*   h_y = (float*)calloc((size_t)B * M, sizeof(float));
    float*   h_ref = (float*)calloc((size_t)B * M, sizeof(float));
    for (uint32_t r = 0; r < M; r++) {
        for (uint32_t b = 0; b < NB; b++) {
            uint8_t* bp = h_w + ((size_t)r * NB + b) * 18u;
            uint16_t dh = tn_vk_f32_to_f16(0.015f * (float)(((r + b) % 5u) + 1u));
            bp[0] = (uint8_t)(dh & 0xFFu);
            bp[1] = (uint8_t)(dh >> 8);
            for (int i = 0; i < 16; i++) bp[2 + i] = (uint8_t)(((r * 13u + b * 29u + (uint32_t)i * 7u) * 37u) & 0xFFu);
        }
    }
    for (uint32_t i = 0; i < B * K; i++) h_x[i] = 0.25f * sinf((float)i * 0.13f);
    for (uint32_t tok = 0; tok < B; tok++) {
        for (uint32_t r = 0; r < M; r++) {
            float acc = 0.0f;
            for (uint32_t b = 0; b < NB; b++) {
                const uint8_t* bp = h_w + ((size_t)r * NB + b) * 18u;
                float wd = tn_vk_f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
                float s = 0.0f;
                for (int i = 0; i < 16; i++) {
                    uint8_t c = bp[2 + i];
                    s += (float)((int)(c & 15u) - 8) * h_x[tok * K + b * 32u + (uint32_t)i]
                       + (float)((int)(c >> 4) - 8) * h_x[tok * K + b * 32u + (uint32_t)i + 16u];
                }
                acc += s * wd;
            }
            h_ref[tok * M + r] = acc;
        }
    }
    int64_t bw = tn_vk_buf_create((int64_t)(uintptr_t)h_w, (int64_t)w_sz);
    int64_t bx = tn_vk_buf_create((int64_t)(uintptr_t)h_x, (int64_t)((size_t)B * K * sizeof(float)));
    int64_t by = tn_vk_buf_create(0, (int64_t)((size_t)B * M * sizeof(float)));
    int ok = tn_vk_q4_gemv(bw, bx, by, M, K, B) ? 1 : 0;
    if (ok) {
        tn_vk_buf_read(by, (int64_t)(uintptr_t)h_y, (int64_t)((size_t)B * M * sizeof(float)));
        for (uint32_t i = 0; i < B * M; i++) {
            if (fabsf(h_y[i] - h_ref[i]) > 1e-3f) { ok = 0; break; }
        }
    }
    tn_vk_buf_free(bw); tn_vk_buf_free(bx); tn_vk_buf_free(by);
    free(h_w); free(h_x); free(h_y); free(h_ref);
    if (!ok) return 0;

    // 2. RMSNorm + SwiGLU (standard & GPT-OSS) GPU verification
    const uint32_t H = 128u;
    float rx[256], rw[128], ry[256];
    for (uint32_t i = 0; i < 2u * H; i++) rx[i] = 0.4f * cosf((float)i * 0.21f);
    for (uint32_t i = 0; i < H; i++) rw[i] = 0.7f + 0.2f * sinf((float)i * 0.19f);
    int64_t brx = tn_vk_buf_create((int64_t)(uintptr_t)rx, (int64_t)sizeof(rx));
    int64_t brw = tn_vk_buf_create((int64_t)(uintptr_t)rw, (int64_t)sizeof(rw));
    int64_t bry = tn_vk_buf_create(0, (int64_t)sizeof(ry));
    ok = tn_vk_rmsnorm(brx, brw, bry, 2, H, 1e-5) ? 1 : 0;
    if (ok) {
        tn_vk_buf_read(bry, (int64_t)(uintptr_t)ry, (int64_t)sizeof(ry));
        for (uint32_t b = 0; b < 2u && ok; b++) {
            float ss = 0.0f;
            for (uint32_t i = 0; i < H; i++) ss += rx[b * H + i] * rx[b * H + i];
            float inv = 1.0f / sqrtf(ss / (float)H + 1e-5f);
            for (uint32_t i = 0; i < H; i++) {
                float ref = rx[b * H + i] * inv * rw[i];
                if (fabsf(ry[b * H + i] - ref) > 1e-4f) { ok = 0; break; }
            }
        }
    }
    if (ok) {
        ok = tn_vk_swiglu(brx, brw, bry, H, 0) ? 1 : 0;
        if (ok) {
            tn_vk_buf_read(bry, (int64_t)(uintptr_t)ry, (int64_t)(H * sizeof(float)));
            for (uint32_t i = 0; i < H; i++) {
                float ref = (rx[i] / (1.0f + expf(-rx[i]))) * rw[i];
                if (fabsf(ry[i] - ref) > 1e-4f) { ok = 0; break; }
            }
        }
    }
    tn_vk_buf_free(brx); tn_vk_buf_free(brw); tn_vk_buf_free(bry);
    if (!ok) return 0;

    // 3. Paged FlashAttention-2 (Causal + GQA + non-identity block table) GPU verification
    const uint32_t FB = 4u, NQH = 4u, NKVH = 2u, HD = 64u, PSZ = 2u, NPGS = 2u;
    float fq[4 * 4 * 64], fk[4 * 2 * 64], fv[4 * 2 * 64], fo[4 * 4 * 64];
    int32_t fbt[2] = { 1, 0 }; // swapped physical pages to prove block-table indirection!
    for (uint32_t i = 0; i < FB * NQH * HD; i++) fq[i] = 0.2f * sinf((float)i * 0.11f);
    for (uint32_t i = 0; i < NPGS * PSZ * NKVH * HD; i++) {
        fk[i] = 0.2f * cosf((float)i * 0.07f);
        fv[i] = 0.3f * sinf((float)i * 0.05f + 0.4f);
    }
    float scale = 1.0f / sqrtf((float)HD);
    int64_t bq = tn_vk_buf_create((int64_t)(uintptr_t)fq, (int64_t)sizeof(fq));
    int64_t bk = tn_vk_buf_create((int64_t)(uintptr_t)fk, (int64_t)sizeof(fk));
    int64_t bv = tn_vk_buf_create((int64_t)(uintptr_t)fv, (int64_t)sizeof(fv));
    int64_t bt = tn_vk_buf_create((int64_t)(uintptr_t)fbt, (int64_t)sizeof(fbt));
    int64_t bo = tn_vk_buf_create(0, (int64_t)sizeof(fo));
    ok = tn_vk_flash_attn_paged(bq, bk, bv, bt, bo, FB, NQH, NKVH, HD, 0, PSZ, scale, 0) ? 1 : 0;
    if (ok) {
        tn_vk_buf_read(bo, (int64_t)(uintptr_t)fo, (int64_t)sizeof(fo));
        for (uint32_t tok = 0; tok < FB && ok; tok++) {
            for (uint32_t qh = 0; qh < NQH && ok; qh++) {
                uint32_t kvh = (qh * NKVH) / NQH;
                float scores[8], mx = -1e30f;
                for (uint32_t t = 0; t <= tok; t++) {
                    uint32_t phys = (uint32_t)fbt[t / PSZ];
                    uint32_t kvb = ((phys * PSZ + (t % PSZ)) * NKVH + kvh) * HD;
                    uint32_t qb  = (tok * NQH + qh) * HD;
                    float dot = 0.0f;
                    for (uint32_t d = 0; d < HD; d++) dot += fq[qb + d] * fk[kvb + d];
                    scores[t] = dot * scale;
                    if (scores[t] > mx) mx = scores[t];
                }
                float lsum = 0.0f;
                for (uint32_t t = 0; t <= tok; t++) { scores[t] = expf(scores[t] - mx); lsum += scores[t]; }
                for (uint32_t d = 0; d < HD; d++) {
                    float ref = 0.0f;
                    for (uint32_t t = 0; t <= tok; t++) {
                        uint32_t phys = (uint32_t)fbt[t / PSZ];
                        uint32_t kvb = ((phys * PSZ + (t % PSZ)) * NKVH + kvh) * HD;
                        ref += (scores[t] / lsum) * fv[kvb + d];
                    }
                    if (fabsf(fo[(tok * NQH + qh) * HD + d] - ref) > 1e-3f) { ok = 0; break; }
                }
            }
        }
    }
    tn_vk_buf_free(bq); tn_vk_buf_free(bk); tn_vk_buf_free(bv); tn_vk_buf_free(bt); tn_vk_buf_free(bo);
    if (!ok) return 0;

    // 4. Vulkan 1.3 Hardware DP4A (`VK_KHR_shader_integer_dot_product` / OpSDot) Q4_0 x Q8 GEMM verification
    if (g_vk.pipe_q4_dp4a.pipe != 0) {
        const uint32_t DM = 32u, DK = 256u, DB = 2u, DNB = DK / 32u;
        size_t dw_sz = (size_t)DM * DNB * 18u;
        uint8_t* dw = (uint8_t*)calloc(1, dw_sz);
        int8_t*  dxq = (int8_t*)calloc((size_t)DB * DK, 1);
        float*   dxd = (float*)calloc((size_t)DB * DNB, sizeof(float));
        float*   dy  = (float*)calloc((size_t)DB * DM, sizeof(float));
        for (uint32_t r = 0; r < DM; r++) {
            for (uint32_t b = 0; b < DNB; b++) {
                uint8_t* bp = dw + ((size_t)r * DNB + b) * 18u;
                uint16_t dh = tn_vk_f32_to_f16(0.02f * (float)(((r + b) % 4u) + 1u));
                bp[0] = (uint8_t)(dh & 0xFFu); bp[1] = (uint8_t)(dh >> 8);
                for (int i = 0; i < 16; i++) bp[2 + i] = (uint8_t)(((r * 19u + b * 11u + (uint32_t)i * 5u) * 23u) & 0xFFu);
            }
        }
        for (uint32_t i = 0; i < DB * DK; i++) dxq[i] = (int8_t)(((int)(i * 13u) % 21) - 10);
        for (uint32_t i = 0; i < DB * DNB; i++) dxd[i] = 0.01f * (float)((i % 5u) + 1u);
        int64_t bdw = tn_vk_buf_create((int64_t)(uintptr_t)dw, (int64_t)dw_sz);
        int64_t bdxq = tn_vk_buf_create((int64_t)(uintptr_t)dxq, (int64_t)((size_t)DB * DK));
        int64_t bdxd = tn_vk_buf_create((int64_t)(uintptr_t)dxd, (int64_t)((size_t)DB * DNB * sizeof(float)));
        int64_t bdy = tn_vk_buf_create(0, (int64_t)((size_t)DB * DM * sizeof(float)));
        struct { uint32_t M, K, B; } dpc = { DM, DK, DB };
        int64_t dbufs[4] = { bdw, bdxq, bdxd, bdy };
        ok = tn_vk_dispatch(&g_vk.pipe_q4_dp4a, dbufs, 4u, &dpc, sizeof(dpc), DM, DB, 1u) ? 1 : 0;
        if (ok) {
            tn_vk_buf_read(bdy, (int64_t)(uintptr_t)dy, (int64_t)((size_t)DB * DM * sizeof(float)));
            for (uint32_t tok = 0; tok < DB && ok; tok++) {
                for (uint32_t r = 0; r < DM && ok; r++) {
                    float ref = 0.0f;
                    for (uint32_t b = 0; b < DNB; b++) {
                        const uint8_t* bp = dw + ((size_t)r * DNB + b) * 18u;
                        float wd = tn_vk_f16_to_f32((uint16_t)bp[0] | ((uint16_t)bp[1] << 8));
                        int si = 0;
                        for (int i = 0; i < 16; i++) {
                            uint8_t c = bp[2 + i];
                            si += ((int)(c & 15u) - 8) * (int)dxq[tok * DK + b * 32u + (uint32_t)i]
                                + ((int)(c >> 4) - 8) * (int)dxq[tok * DK + b * 32u + (uint32_t)i + 16u];
                        }
                        ref += (float)si * wd * dxd[tok * DNB + b];
                    }
                    if (fabsf(dy[tok * DM + r] - ref) > 1e-3f) { ok = 0; break; }
                }
            }
        }
        tn_vk_buf_free(bdw); tn_vk_buf_free(bdxq); tn_vk_buf_free(bdxd); tn_vk_buf_free(bdy);
        free(dw); free(dxq); free(dxd); free(dy);
    }
    return ok ? 1 : 0;
}
