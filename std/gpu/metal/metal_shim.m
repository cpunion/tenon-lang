// std/gpu/metal_shim.m — Zero-Rust Native Apple Silicon macOS Metal Hardware Bridge (Objective-C ARC)
//
// Directly drives `MTLCreateSystemDefaultDevice()`, live MSL shader compilation, shared-memory
// compute pipelines, and instanced SDF rounded-quad offscreen rendering with ZERO Rust code!

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>

static id<MTLDevice> g_mtl_device = nil;
static id<MTLCommandQueue> g_mtl_queue = nil;
static id<MTLCommandQueue> g_mtl_copy_queue = nil;
static char g_device_name_buf[256] = "Unavailable";

#define MAX_HANDLES 4096
static id<MTLBuffer> g_buffers[MAX_HANDLES];
static id<MTLComputePipelineState> g_pipelines[MAX_HANDLES];
static id<MTLRenderPipelineState> g_quad_render_pipe = nil;

static int64_t ensure_metal_init(void) {
    if (g_mtl_device != nil) return 1;
    @autoreleasepool {
        g_mtl_device = MTLCreateSystemDefaultDevice();
        if (g_mtl_device == nil) return 0;
        g_mtl_queue = [g_mtl_device newCommandQueue];
        if (g_mtl_queue == nil) return 0;
        g_mtl_copy_queue = [g_mtl_device newCommandQueue];
        const char* nm = [[g_mtl_device name] UTF8String];
        if (nm != NULL) {
            strncpy(g_device_name_buf, nm, sizeof(g_device_name_buf) - 1);
            g_device_name_buf[sizeof(g_device_name_buf) - 1] = 0;
        }
    }
    return 1;
}

int64_t tn_metal_init(void) {
    return ensure_metal_init();
}

// Bridge accessors shared with `mps_shim.m` (MetalPerformanceShaders.framework)
void* tn_metal_raw_device(void) {
    ensure_metal_init();
    return (__bridge void*)g_mtl_device;
}

void* tn_metal_raw_queue(void) {
    ensure_metal_init();
    return (__bridge void*)g_mtl_queue;
}

void* tn_metal_raw_buffer(int64_t handle) {
    if (handle <= 0 || handle >= MAX_HANDLES) return NULL;
    return (__bridge void*)g_buffers[handle];
}

void* tn_metal_raw_pipeline(int64_t handle) {
    if (handle <= 0 || handle >= MAX_HANDLES) return NULL;
    return (__bridge void*)g_pipelines[handle];
}

int64_t tn_metal_device_name_cstr(void) {
    ensure_metal_init();
    return (int64_t)(uintptr_t)g_device_name_buf;
}

int64_t tn_metal_buf_create(int64_t data_ptr, int64_t byte_len) {
    if (!ensure_metal_init() || byte_len <= 0) return 0;
    @autoreleasepool {
        int slot = -1;
        for (int i = 1; i < MAX_HANDLES; i++) {
            if (g_buffers[i] == nil) { slot = i; break; }
        }
        if (slot < 0) return 0;
        id<MTLBuffer> buf = nil;
        if (data_ptr != 0) {
            buf = [g_mtl_device newBufferWithBytes:(const void*)(uintptr_t)data_ptr
                                            length:(NSUInteger)byte_len
                                           options:MTLResourceStorageModeShared];
        } else {
            buf = [g_mtl_device newBufferWithLength:(NSUInteger)byte_len
                                            options:MTLResourceStorageModeShared];
        }
        if (buf == nil) return 0;
        g_buffers[slot] = buf;
        return (int64_t)slot;
    }
}

/// Registers an externally-allocated `id<MTLBuffer>` (e.g., zero-copy `mmap` `newBufferWithBytesNoCopy`)
/// into the shared Tenon `@linear GpuBuffer` handle table (takes +1 ARC reference via `__bridge`).
int64_t tn_metal_wrap_buffer(void* mtl_buf_ptr) {
    if (!ensure_metal_init() || mtl_buf_ptr == NULL) return 0;
    @autoreleasepool {
        int slot = -1;
        for (int i = 1; i < MAX_HANDLES; i++) {
            if (g_buffers[i] == nil) { slot = i; break; }
        }
        if (slot < 0) return 0;
        g_buffers[slot] = (__bridge id<MTLBuffer>)mtl_buf_ptr;
        return (int64_t)slot;
    }
}

int64_t tn_metal_buf_read(int64_t handle, int64_t dst_ptr, int64_t byte_len) {
    if (handle <= 0 || handle >= MAX_HANDLES || g_buffers[handle] == nil || dst_ptr == 0 || byte_len <= 0) {
        return 0;
    }
    id<MTLBuffer> buf = g_buffers[handle];
    NSUInteger avail = [buf length];
    NSUInteger n = (NSUInteger)byte_len;
    if (n > avail) n = avail;
    memcpy((void*)(uintptr_t)dst_ptr, [buf contents], n);
    return (int64_t)n;
}

int64_t tn_metal_buf_free(int64_t handle) {
    if (handle <= 0 || handle >= MAX_HANDLES) return 0;
    g_buffers[handle] = nil;
    return 1;
}

int64_t tn_metal_compile_compute(int64_t msl_cstr, int64_t fn_cstr) {
    if (!ensure_metal_init() || msl_cstr == 0 || fn_cstr == 0) return 0;
    @autoreleasepool {
        NSString* src = [NSString stringWithUTF8String:(const char*)(uintptr_t)msl_cstr];
        NSString* fnName = [NSString stringWithUTF8String:(const char*)(uintptr_t)fn_cstr];
        NSError* err = nil;
        id<MTLLibrary> lib = [g_mtl_device newLibraryWithSource:src options:nil error:&err];
        if (lib == nil) {
            fprintf(stderr, "[tenon-metal] MSL compile error: %s\n", [[err localizedDescription] UTF8String]);
            return 0;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:fnName];
        if (fn == nil) {
            fprintf(stderr, "[tenon-metal] kernel function `%s` not found\n", [fnName UTF8String]);
            return 0;
        }
        id<MTLComputePipelineState> pso = [g_mtl_device newComputePipelineStateWithFunction:fn error:&err];
        if (pso == nil) {
            fprintf(stderr, "[tenon-metal] pipeline error: %s\n", [[err localizedDescription] UTF8String]);
            return 0;
        }
        int slot = -1;
        for (int i = 1; i < MAX_HANDLES; i++) {
            if (g_pipelines[i] == nil) { slot = i; break; }
        }
        if (slot < 0) return 0;
        g_pipelines[slot] = pso;
        return (int64_t)slot;
    }
}

int64_t tn_metal_pipe_free(int64_t handle) {
    if (handle <= 0 || handle >= MAX_HANDLES) return 0;
    g_pipelines[handle] = nil;
    return 1;
}

int64_t tn_metal_dispatch_compute4(
    int64_t pipe_handle,
    int64_t b0, int64_t b1, int64_t b2, int64_t b3,
    int64_t gx, int64_t gy, int64_t gz
) {
    if (pipe_handle <= 0 || pipe_handle >= MAX_HANDLES || g_pipelines[pipe_handle] == nil) return 0;
    if (gx <= 0 || gy <= 0 || gz <= 0) return 0;
    @autoreleasepool {
        id<MTLComputePipelineState> pso = g_pipelines[pipe_handle];
        id<MTLCommandBuffer> cmd = [g_mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        int64_t bs[4] = { b0, b1, b2, b3 };
        for (int i = 0; i < 4; i++) {
            if (bs[i] > 0 && bs[i] < MAX_HANDLES && g_buffers[bs[i]] != nil) {
                [enc setBuffer:g_buffers[bs[i]] offset:0 atIndex:(NSUInteger)i];
            }
        }
        MTLSize grid = MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, (NSUInteger)gz);
        NSUInteger maxT = [pso maxTotalThreadsPerThreadgroup];
        NSUInteger tx = (NSUInteger)gx < 16 ? (NSUInteger)gx : 16;
        NSUInteger ty = (NSUInteger)gy < 16 ? (NSUInteger)gy : 16;
        while (tx * ty > maxT && ty > 1) ty /= 2;
        MTLSize tg = MTLSizeMake(tx, ty, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 1 : 0;
    }
}

int64_t tn_metal_dispatch_threadgroups4(
    int64_t pipe_handle,
    int64_t b0, int64_t b1, int64_t b2, int64_t b3,
    int64_t tgx, int64_t tgy, int64_t tgz,
    int64_t tx, int64_t ty, int64_t tz
) {
    if (pipe_handle <= 0 || pipe_handle >= MAX_HANDLES || g_pipelines[pipe_handle] == nil) return 0;
    if (tgx <= 0 || tgy <= 0 || tgz <= 0 || tx <= 0 || ty <= 0 || tz <= 0) return 0;
    @autoreleasepool {
        id<MTLComputePipelineState> pso = g_pipelines[pipe_handle];
        id<MTLCommandBuffer> cmd = [g_mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];
        int64_t bs[4] = { b0, b1, b2, b3 };
        for (int i = 0; i < 4; i++) {
            if (bs[i] > 0 && bs[i] < MAX_HANDLES && g_buffers[bs[i]] != nil) {
                [enc setBuffer:g_buffers[bs[i]] offset:0 atIndex:(NSUInteger)i];
            }
        }
        MTLSize groups = MTLSizeMake((NSUInteger)tgx, (NSUInteger)tgy, (NSUInteger)tgz);
        MTLSize threads = MTLSizeMake((NSUInteger)tx, (NSUInteger)ty, (NSUInteger)tz);
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 1 : 0;
    }
}

static const char* kTenonGpuiQuadShaderMSL =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct QuadInstance {\n"
"    float x; float y; float w; float h;\n"
"    float r; float g; float b; float a;\n"
"    float radius; float border_w; float br; float bg;\n"
"};\n"
"struct Viewport { float width; float height; float _p0; float _p1; };\n"
"struct VSOut {\n"
"    float4 pos [[position]];\n"
"    float2 local_px;\n"
"    float2 size_px;\n"
"    float4 color;\n"
"    float radius;\n"
"    float border_w;\n"
"    float2 border_rg;\n"
"};\n"
"vertex VSOut quad_vs(uint vid [[vertex_id]], uint iid [[instance_id]],\n"
"                     device const QuadInstance* quads [[buffer(0)]],\n"
"                     constant Viewport& vp [[buffer(1)]]) {\n"
"    const float2 corners[6] = {\n"
"        float2(0.0, 0.0), float2(1.0, 0.0), float2(0.0, 1.0),\n"
"        float2(1.0, 0.0), float2(1.0, 1.0), float2(0.0, 1.0)\n"
"    };\n"
"    QuadInstance q = quads[iid];\n"
"    float2 uv = corners[vid];\n"
"    float2 px = float2(q.x + uv.x * q.w, q.y + uv.y * q.h);\n"
"    float ndc_x = (px.x / vp.width) * 2.0f - 1.0f;\n"
"    float ndc_y = 1.0f - (px.y / vp.height) * 2.0f;\n"
"    VSOut o;\n"
"    o.pos = float4(ndc_x, ndc_y, 0.0f, 1.0f);\n"
"    o.local_px = uv * float2(q.w, q.h);\n"
"    o.size_px = float2(q.w, q.h);\n"
"    o.color = float4(q.r, q.g, q.b, q.a);\n"
"    o.radius = q.radius;\n"
"    o.border_w = q.border_w;\n"
"    o.border_rg = float2(q.br, q.bg);\n"
"    return o;\n"
"}\n"
"fragment float4 quad_fs(VSOut in [[stage_in]]) {\n"
"    float2 half_sz = in.size_px * 0.5f;\n"
"    float2 p = in.local_px - half_sz;\n"
"    float rad = min(in.radius, min(half_sz.x, half_sz.y));\n"
"    float2 d = abs(p) - (half_sz - float2(rad));\n"
"    float sdf = length(max(d, 0.0f)) + min(max(d.x, d.y), 0.0f) - rad;\n"
"    if (sdf > 0.5f) discard_fragment();\n"
"    float aa = clamp(0.5f - sdf, 0.0f, 1.0f);\n"
"    float4 col = in.color;\n"
"    if (in.border_w > 0.0f && sdf > -in.border_w) {\n"
"        col = float4(in.border_rg.x, in.border_rg.y, 0.85f, 1.0f);\n"
"    }\n"
"    return float4(col.rgb, col.a * aa);\n"
"}\n";

int64_t tn_metal_render_quads_offscreen(
    int64_t quads_ptr,
    int64_t quad_count,
    int64_t width,
    int64_t height,
    int64_t out_rgba_ptr
) {
    if (!ensure_metal_init() || width <= 0 || height <= 0 || out_rgba_ptr == 0) return 0;
    @autoreleasepool {
        if (g_quad_render_pipe == nil) {
            NSError* err = nil;
            NSString* src = [NSString stringWithUTF8String:kTenonGpuiQuadShaderMSL];
            id<MTLLibrary> lib = [g_mtl_device newLibraryWithSource:src options:nil error:&err];
            if (lib == nil) {
                fprintf(stderr, "[tenon-metal] quad shader compile error: %s\n", [[err localizedDescription] UTF8String]);
                return 0;
            }
            MTLRenderPipelineDescriptor* desc = [[MTLRenderPipelineDescriptor alloc] init];
            desc.vertexFunction = [lib newFunctionWithName:@"quad_vs"];
            desc.fragmentFunction = [lib newFunctionWithName:@"quad_fs"];
            desc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
            desc.colorAttachments[0].blendingEnabled = YES;
            desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
            desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
            desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            g_quad_render_pipe = [g_mtl_device newRenderPipelineStateWithDescriptor:desc error:&err];
            if (g_quad_render_pipe == nil) {
                fprintf(stderr, "[tenon-metal] quad pipeline error: %s\n", [[err localizedDescription] UTF8String]);
                return 0;
            }
        }

        MTLTextureDescriptor* tdesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                         width:(NSUInteger)width
                                                                                        height:(NSUInteger)height
                                                                                     mipmapped:NO];
        tdesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        tdesc.storageMode = MTLStorageModeShared;
        id<MTLTexture> tex = [g_mtl_device newTextureWithDescriptor:tdesc];
        if (tex == nil) return 0;

        MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
        rpd.colorAttachments[0].texture = tex;
        rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        // Dark slate background (#0F172A -> RGB 15, 23, 42)
        rpd.colorAttachments[0].clearColor = MTLClearColorMake(15.0 / 255.0, 23.0 / 255.0, 42.0 / 255.0, 1.0);

        id<MTLCommandBuffer> cmd = [g_mtl_queue commandBuffer];
        id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rpd];
        if (quad_count > 0 && quads_ptr != 0) {
            id<MTLBuffer> qbuf = [g_mtl_device newBufferWithBytes:(const void*)(uintptr_t)quads_ptr
                                                           length:(NSUInteger)(quad_count * 48)
                                                          options:MTLResourceStorageModeShared];
            float vp[4] = { (float)width, (float)height, 0.0f, 0.0f };
            [enc setRenderPipelineState:g_quad_render_pipe];
            [enc setVertexBuffer:qbuf offset:0 atIndex:0];
            [enc setVertexBytes:vp length:sizeof(vp) atIndex:1];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle
                    vertexStart:0
                    vertexCount:6
                  instanceCount:(NSUInteger)quad_count];
        }
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) return 0;

        MTLRegion reg = MTLRegionMake2D(0, 0, (NSUInteger)width, (NSUInteger)height);
        [tex getBytes:(void*)(uintptr_t)out_rgba_ptr
          bytesPerRow:(NSUInteger)(width * 4)
           fromRegion:reg
          mipmapLevel:0];
        return 1;
    }
}

// =========================================================================
// Generic Op-Agnostic Metal Graph Plan Executor & Live PGO Auto-Tuner
// (Zero knowledge of MatMul/Norm/Attention/MXFP4 — 100% driven by Tenon's
//  compile-time `tile_ir` Execution Plan + MSL source!)
// =========================================================================
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_PLANS 64
#define MAX_PLAN_BUFS 128
#define MAX_PLAN_DISPATCHES 128
#define MAX_PGO_CANDS 16

typedef struct {
    char name[96];
    int kind; // 0=input, 1=output, 2=arena, 3=scratch, 4=zero_scratch
    int io_idx;
    int64_t rows;
    int64_t cols;
    size_t offset;
    size_t size;
    char repack[32];
    id<MTLBuffer> buf; // non-nil for arena/scratch/repacked input or bound io
} TnPlanBuf;

typedef struct {
    char kname[128];
    id<MTLComputePipelineState> pso;
    NSUInteger tgx, tgy, tgz;
    NSUInteger tx, ty, tz;
    NSUInteger smem;
    int n_args;
    int arg_idx[16]; // index into plan->bufs
    int has_post;
    char post_kname[128];
    id<MTLComputePipelineState> post_pso;
    NSUInteger post_tgx, post_tgy, post_tgz;
    NSUInteger post_tx, post_ty, post_tz;
    int post_n_args;
    int post_arg_idx[16];
} TnPlanStep;

typedef struct {
    char group_name[128];
    int step_idx; // slot in plan->steps replaced by winning candidate
    int n_cands;
    int winner;
    double winner_ms;
    TnPlanStep cands[MAX_PGO_CANDS];
} TnPgoGroup;

typedef struct {
    int dst_buf_idx;
    size_t k;
    size_t n_per;
    int n_srcs;
    int src_buf_idx[8];
    int built;
} TnConcatSpec;

typedef struct {
    int min_m;
    int max_m;
    int tgy_mode; // 0=DYN_M, 1=CEIL32, 2=CEIL64
    TnPlanStep step;
} TnDynBucket;

typedef struct {
    int active;
    char graph_name[96];
    char opts_tag[256];
    size_t arena_bytes;
    size_t naive_arena_bytes;
    size_t donated_bytes;
    int single_cb;
    int pgo_enabled;
    int pgo_tuned;
    int replay_enabled;
    int overlap_enabled;
    id<MTLLibrary> lib;
    id<MTLBuffer> arena_buf;
    id<MTLIndirectCommandBuffer> icb;
    int icb_steps;
    int n_bufs;
    TnPlanBuf bufs[MAX_PLAN_BUFS];
    int n_steps;
    TnPlanStep steps[MAX_PLAN_DISPATCHES];
    int n_pgo;
    TnPgoGroup pgo[16];
    // Facility #2: Horizontal Weight Packing (`CONCAT_NK`)
    int n_concats;
    TnConcatSpec concats[8];
    // Facility #3: Buffer Donation (`DONATE|out_idx|in_idx`)
    int n_donates;
    int donate_out[8];
    int donate_in[8];
    // Facility #7: Symbolic Dynamic-Shape Bucketing (`DYN_CTRL` + `DYN_BUCKET`)
    int dyn_ctrl_buf_idx;
    int dyn_m_actual;
    int dyn_step_idx;
    int active_bucket;
    int n_buckets;
    TnDynBucket buckets[4];
} TnMetalPlan;

static TnMetalPlan g_plans[MAX_PLANS];

static double tn_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

static int plan_find_or_add_buf(TnMetalPlan* p, const char* name) {
    for (int i = 0; i < p->n_bufs; i++) {
        if (strcmp(p->bufs[i].name, name) == 0) return i;
    }
    if (p->n_bufs >= MAX_PLAN_BUFS) return -1;
    int idx = p->n_bufs++;
    memset(&p->bufs[idx], 0, sizeof(TnPlanBuf));
    strncpy(p->bufs[idx].name, name, sizeof(p->bufs[idx].name) - 1);
    return idx;
}

static id<MTLComputePipelineState> plan_compile_fn_icb(TnMetalPlan* p, const char* fn_name, int for_icb) {
    NSString* ns = [NSString stringWithUTF8String:fn_name];
    id<MTLFunction> fn = [p->lib newFunctionWithName:ns];
    if (!fn) {
        fprintf(stderr, "[tenon-plan] kernel `%s` not found in compiled MSL\n", fn_name);
        return nil;
    }
    NSError* err = nil;
    id<MTLComputePipelineState> pso = nil;
    if (for_icb) {
        MTLComputePipelineDescriptor* desc = [[MTLComputePipelineDescriptor alloc] init];
        desc.computeFunction = fn;
        desc.supportIndirectCommandBuffers = YES;
        pso = [g_mtl_device newComputePipelineStateWithDescriptor:desc
                                                          options:MTLPipelineOptionNone
                                                       reflection:nil
                                                            error:&err];
    }
    if (!pso) {
        err = nil;
        pso = [g_mtl_device newComputePipelineStateWithFunction:fn error:&err];
    }
    if (!pso) {
        fprintf(stderr, "[tenon-plan] PSO error for `%s`: %s\n", fn_name, [[err localizedDescription] UTF8String]);
    }
    return pso;
}

static id<MTLComputePipelineState> plan_compile_fn(TnMetalPlan* p, const char* fn_name) {
    // Keep direct-dispatch PSOs at full Metal peak speed (do NOT set supportIndirectCommandBuffers
    // globally, which forces argument-buffer ABI and hurts AMX register allocation by ~35%!).
    return plan_compile_fn_icb(p, fn_name, 0);
}

static void parse_step_line(TnMetalPlan* p, TnPlanStep* st, const char* rest) {
    // Format: kname|tg=x,y,z|th=x,y,z|smem=S|args=a,b,c
    char copy[1024];
    strncpy(copy, rest, sizeof(copy) - 1);
    copy[sizeof(copy) - 1] = 0;
    char* save = NULL;
    char* tok = strtok_r(copy, "|", &save);
    if (!tok) return;
    strncpy(st->kname, tok, sizeof(st->kname) - 1);
    st->pso = plan_compile_fn(p, st->kname);
    while ((tok = strtok_r(NULL, "|", &save)) != NULL) {
        if (strncmp(tok, "tg=", 3) == 0) {
            unsigned long x = 1, y = 1, z = 1;
            sscanf(tok + 3, "%lu,%lu,%lu", &x, &y, &z);
            st->tgx = x; st->tgy = y; st->tgz = z;
        } else if (strncmp(tok, "th=", 3) == 0) {
            unsigned long x = 1, y = 1, z = 1;
            sscanf(tok + 3, "%lu,%lu,%lu", &x, &y, &z);
            st->tx = x; st->ty = y; st->tz = z;
        } else if (strncmp(tok, "smem=", 5) == 0) {
            st->smem = (NSUInteger)strtoull(tok + 5, NULL, 10);
        } else if (strncmp(tok, "args=", 5) == 0) {
            char* asave = NULL;
            char* at = strtok_r(tok + 5, ",", &asave);
            st->n_args = 0;
            while (at && st->n_args < 16) {
                st->arg_idx[st->n_args++] = plan_find_or_add_buf(p, at);
                at = strtok_r(NULL, ",", &asave);
            }
        }
    }
}

static void parse_post_line(TnMetalPlan* p, TnPlanStep* st, const char* rest) {
    char copy[1024];
    strncpy(copy, rest, sizeof(copy) - 1);
    copy[sizeof(copy) - 1] = 0;
    char* save = NULL;
    char* tok = strtok_r(copy, "|", &save);
    if (!tok) return;
    st->has_post = 1;
    strncpy(st->post_kname, tok, sizeof(st->post_kname) - 1);
    st->post_pso = plan_compile_fn(p, st->post_kname);
    while ((tok = strtok_r(NULL, "|", &save)) != NULL) {
        if (strncmp(tok, "tg=", 3) == 0) {
            unsigned long x = 1, y = 1, z = 1;
            sscanf(tok + 3, "%lu,%lu,%lu", &x, &y, &z);
            st->post_tgx = x; st->post_tgy = y; st->post_tgz = z;
        } else if (strncmp(tok, "th=", 3) == 0) {
            unsigned long x = 1, y = 1, z = 1;
            sscanf(tok + 3, "%lu,%lu,%lu", &x, &y, &z);
            st->post_tx = x; st->post_ty = y; st->post_tz = z;
        } else if (strncmp(tok, "args=", 5) == 0) {
            char* asave = NULL;
            char* at = strtok_r(tok + 5, ",", &asave);
            st->post_n_args = 0;
            while (at && st->post_n_args < 16) {
                st->post_arg_idx[st->post_n_args++] = plan_find_or_add_buf(p, at);
                at = strtok_r(NULL, ",", &asave);
            }
        }
    }
}

int64_t tn_metal_plan_create(int64_t msl_cstr, int64_t plan_cstr, int64_t graph_cstr) {
    if (!ensure_metal_init() || !msl_cstr || !plan_cstr || !graph_cstr) return 0;
    @autoreleasepool {
        int slot = -1;
        for (int i = 1; i < MAX_PLANS; i++) {
            if (!g_plans[i].active) { slot = i; break; }
        }
        if (slot < 0) return 0;
        TnMetalPlan* p = &g_plans[slot];
        memset(p, 0, sizeof(TnMetalPlan));
        p->dyn_ctrl_buf_idx = -1;
        p->dyn_step_idx = -1;

        NSString* src = [NSString stringWithUTF8String:(const char*)(uintptr_t)msl_cstr];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.fastMathEnabled = YES;
        NSError* err = nil;
        p->lib = [g_mtl_device newLibraryWithSource:src options:opts error:&err];
        if (!p->lib) {
            fprintf(stderr, "[tenon-plan] MSL compile error: %s\n", [[err localizedDescription] UTF8String]);
            return 0;
        }

        const char* want = (const char*)(uintptr_t)graph_cstr;
        const char* plan_txt = (const char*)(uintptr_t)plan_cstr;
        char* dup = strdup(plan_txt);
        int in_graph = 0;
        int cur_pgo = -1;

        char* lsave = NULL;
        char* line = strtok_r(dup, "\n", &lsave);
        while (line) {
            if (strncmp(line, "GRAPH|", 6) == 0) {
                char gname[96] = {0};
                const char* bar = strchr(line + 6, '|');
                size_t len = bar ? (size_t)(bar - (line + 6)) : strlen(line + 6);
                if (len >= sizeof(gname)) len = sizeof(gname) - 1;
                memcpy(gname, line + 6, len);
                if (strcmp(gname, want) == 0) {
                    in_graph = 1;
                    strncpy(p->graph_name, gname, sizeof(p->graph_name) - 1);
                    char* ksave = NULL;
                    char* k = strtok_r((char*)(bar ? bar + 1 : ""), "|", &ksave);
                    while (k) {
                        if (strncmp(k, "opts=", 5) == 0) strncpy(p->opts_tag, k + 5, sizeof(p->opts_tag) - 1);
                        else if (strncmp(k, "arena=", 6) == 0) p->arena_bytes = (size_t)strtoull(k + 6, NULL, 10);
                        else if (strncmp(k, "naive_arena=", 12) == 0) p->naive_arena_bytes = (size_t)strtoull(k + 12, NULL, 10);
                        else if (strncmp(k, "donated=", 8) == 0) p->donated_bytes = (size_t)strtoull(k + 8, NULL, 10);
                        else if (strncmp(k, "single_cb=", 10) == 0) p->single_cb = atoi(k + 10);
                        else if (strncmp(k, "pgo=", 4) == 0) p->pgo_enabled = atoi(k + 4);
                        else if (strncmp(k, "replay=", 7) == 0) p->replay_enabled = atoi(k + 7);
                        else if (strncmp(k, "overlap=", 8) == 0) p->overlap_enabled = atoi(k + 8);
                        k = strtok_r(NULL, "|", &ksave);
                    }
                    if (p->arena_bytes < 256) p->arena_bytes = 256;
                    if (p->single_cb) {
                        p->arena_buf = [g_mtl_device newBufferWithLength:p->arena_bytes options:MTLResourceStorageModeShared];
                    }
                }
            } else if (in_graph) {
                if (strcmp(line, "END_GRAPH") == 0) {
                    break;
                } else if (strncmp(line, "INPUT|", 6) == 0) {
                    int io_i = 0; char nm[96]={0}, dt[32]={0}, rp[32]={0}; long long r=0,c=0,sz=0;
                    sscanf(line + 6, "%d|%95[^|]|%lld|%lld|%31[^|]|%lld|%31s", &io_i, nm, &r, &c, dt, &sz, rp);
                    int bi = plan_find_or_add_buf(p, nm);
                    if (bi >= 0) {
                        p->bufs[bi].kind = 0; p->bufs[bi].io_idx = io_i;
                        p->bufs[bi].rows = r; p->bufs[bi].cols = c; p->bufs[bi].size = (size_t)sz;
                        strncpy(p->bufs[bi].repack, rp, sizeof(p->bufs[bi].repack) - 1);
                    }
                } else if (strncmp(line, "OUTPUT|", 7) == 0) {
                    int io_i = 0; char nm[96]={0}, dt[32]={0}; long long r=0,c=0,sz=0;
                    sscanf(line + 7, "%d|%95[^|]|%lld|%lld|%31[^|]|%lld", &io_i, nm, &r, &c, dt, &sz);
                    int bi = plan_find_or_add_buf(p, nm);
                    if (bi >= 0) {
                        p->bufs[bi].kind = 1; p->bufs[bi].io_idx = io_i;
                        p->bufs[bi].rows = r; p->bufs[bi].cols = c; p->bufs[bi].size = (size_t)sz;
                    }
                } else if (strncmp(line, "ARENA|", 6) == 0) {
                    char nm[96]={0}; unsigned long long off=0, sz=0;
                    sscanf(line + 6, "%95[^|]|%llu|%llu", nm, &off, &sz);
                    int bi = plan_find_or_add_buf(p, nm);
                    if (bi >= 0) {
                        p->bufs[bi].kind = 2; p->bufs[bi].size = (size_t)sz;
                        if (p->single_cb && p->arena_buf) {
                            p->bufs[bi].offset = (size_t)off;
                            p->bufs[bi].buf = p->arena_buf;
                        } else {
                            p->bufs[bi].offset = 0;
                            p->bufs[bi].buf = [g_mtl_device newBufferWithLength:(NSUInteger)(sz > 64 ? sz : 64)
                                                                       options:MTLResourceStorageModeShared];
                        }
                    }
                } else if (strncmp(line, "SCRATCH|", 8) == 0 || strncmp(line, "ZERO_SCRATCH|", 13) == 0) {
                    int is_z = (strncmp(line, "ZERO_SCRATCH|", 13) == 0);
                    char nm[96]={0}; unsigned long long sz=0;
                    sscanf(line + (is_z ? 13 : 8), "%95[^|]|%llu", nm, &sz);
                    int bi = plan_find_or_add_buf(p, nm);
                    if (bi >= 0) {
                        p->bufs[bi].kind = is_z ? 4 : 3;
                        p->bufs[bi].size = (size_t)(sz > 64 ? sz : 64);
                        p->bufs[bi].offset = 0;
                        p->bufs[bi].buf = [g_mtl_device newBufferWithLength:(NSUInteger)p->bufs[bi].size
                                                                   options:MTLResourceStorageModeShared];
                    }
                } else if (strncmp(line, "CONCAT_NK|", 10) == 0) {
                    // Format: CONCAT_NK|dst_name|k|n_per|w0,w1,w2
                    if (p->n_concats < 8) {
                        char copy[512];
                        strncpy(copy, line + 10, sizeof(copy) - 1);
                        copy[sizeof(copy) - 1] = 0;
                        char* cs = NULL;
                        char* dst_nm = strtok_r(copy, "|", &cs);
                        char* k_str = strtok_r(NULL, "|", &cs);
                        char* np_str = strtok_r(NULL, "|", &cs);
                        char* ws_str = strtok_r(NULL, "|", &cs);
                        if (dst_nm && k_str && np_str && ws_str) {
                            TnConcatSpec* sp = &p->concats[p->n_concats++];
                            memset(sp, 0, sizeof(TnConcatSpec));
                            sp->dst_buf_idx = plan_find_or_add_buf(p, dst_nm);
                            sp->k = (size_t)strtoull(k_str, NULL, 10);
                            sp->n_per = (size_t)strtoull(np_str, NULL, 10);
                            if (sp->dst_buf_idx >= 0) p->bufs[sp->dst_buf_idx].kind = 5;
                            char* ws = NULL;
                            char* wt = strtok_r(ws_str, ",", &ws);
                            while (wt && sp->n_srcs < 8) {
                                sp->src_buf_idx[sp->n_srcs++] = plan_find_or_add_buf(p, wt);
                                wt = strtok_r(NULL, ",", &ws);
                            }
                        }
                    }
                } else if (strncmp(line, "DONATE|", 7) == 0) {
                    int oi = 0, ii = 0;
                    if (sscanf(line + 7, "%d|%d", &oi, &ii) == 2 && p->n_donates < 8) {
                        p->donate_out[p->n_donates] = oi;
                        p->donate_in[p->n_donates] = ii;
                        p->n_donates++;
                    }
                } else if (strncmp(line, "DYN_CTRL|", 9) == 0) {
                    char nm[96] = {0}; int max_m = 1;
                    if (sscanf(line + 9, "%95[^|]|%d", nm, &max_m) == 2) {
                        int bi = plan_find_or_add_buf(p, nm);
                        if (bi >= 0) {
                            p->bufs[bi].kind = 3;
                            p->bufs[bi].size = 64;
                            p->bufs[bi].offset = 0;
                            p->bufs[bi].buf = [g_mtl_device newBufferWithLength:64 options:MTLResourceStorageModeShared];
                            uint32_t* cp = (uint32_t*)[p->bufs[bi].buf contents];
                            cp[0] = (uint32_t)max_m;
                            p->dyn_ctrl_buf_idx = bi;
                            p->dyn_m_actual = max_m;
                        }
                    }
                } else if (strncmp(line, "DYN_BUCKET|", 11) == 0) {
                    // Format: DYN_BUCKET|min_m|max_m|kname|tgx=N|tgy_mode=DYN_M/CEIL32/CEIL64|th=x,y,z|args=a,b,ctrl,out
                    if (p->n_buckets < 4) {
                        if (p->dyn_step_idx < 0 && p->n_steps < MAX_PLAN_DISPATCHES) {
                            p->dyn_step_idx = p->n_steps++;
                            memset(&p->steps[p->dyn_step_idx], 0, sizeof(TnPlanStep));
                        }
                        char copy[1024];
                        strncpy(copy, line + 11, sizeof(copy) - 1);
                        copy[sizeof(copy) - 1] = 0;
                        char* bs = NULL;
                        char* min_s = strtok_r(copy, "|", &bs);
                        char* max_s = strtok_r(NULL, "|", &bs);
                        char* kn_s = strtok_r(NULL, "|", &bs);
                        if (min_s && max_s && kn_s) {
                            int bi = p->n_buckets++;
                            TnDynBucket* bk = &p->buckets[bi];
                            memset(bk, 0, sizeof(TnDynBucket));
                            bk->min_m = atoi(min_s);
                            bk->max_m = atoi(max_s);
                            strncpy(bk->step.kname, kn_s, sizeof(bk->step.kname) - 1);
                            bk->step.pso = plan_compile_fn(p, bk->step.kname);
                            bk->step.tgx = 1; bk->step.tgy = 1; bk->step.tgz = 1;
                            bk->step.tx = 128; bk->step.ty = 1; bk->step.tz = 1;
                            char* tok = NULL;
                            while ((tok = strtok_r(NULL, "|", &bs)) != NULL) {
                                if (strncmp(tok, "tgx=", 4) == 0) {
                                    bk->step.tgx = (NSUInteger)strtoull(tok + 4, NULL, 10);
                                } else if (strncmp(tok, "tgy_mode=", 9) == 0) {
                                    if (strcmp(tok + 9, "CEIL32") == 0) bk->tgy_mode = 1;
                                    else if (strcmp(tok + 9, "CEIL64") == 0) bk->tgy_mode = 2;
                                    else bk->tgy_mode = 0;
                                } else if (strncmp(tok, "th=", 3) == 0) {
                                    unsigned long x = 128, y = 1, z = 1;
                                    sscanf(tok + 3, "%lu,%lu,%lu", &x, &y, &z);
                                    bk->step.tx = x; bk->step.ty = y; bk->step.tz = z;
                                } else if (strncmp(tok, "args=", 5) == 0) {
                                    char* as = NULL;
                                    char* at = strtok_r(tok + 5, ",", &as);
                                    bk->step.n_args = 0;
                                    while (at && bk->step.n_args < 16) {
                                        bk->step.arg_idx[bk->step.n_args++] = plan_find_or_add_buf(p, at);
                                        at = strtok_r(NULL, ",", &as);
                                    }
                                }
                            }
                            if (p->dyn_step_idx >= 0 && p->dyn_m_actual >= bk->min_m && p->dyn_m_actual <= bk->max_m) {
                                p->active_bucket = bi;
                                p->steps[p->dyn_step_idx] = bk->step;
                                int m_act = p->dyn_m_actual;
                                p->steps[p->dyn_step_idx].tgy = (bk->tgy_mode == 1) ? (NSUInteger)((m_act + 31) / 32)
                                                              : (bk->tgy_mode == 2) ? (NSUInteger)((m_act + 63) / 64)
                                                              : (NSUInteger)m_act;
                            }
                        }
                    }
                } else if (strncmp(line, "DISPATCH|", 9) == 0) {
                    if (p->n_steps < MAX_PLAN_DISPATCHES) {
                        TnPlanStep* st = &p->steps[p->n_steps++];
                        memset(st, 0, sizeof(TnPlanStep));
                        parse_step_line(p, st, line + 9);
                    }
                } else if (strncmp(line, "PGO_BEGIN|", 10) == 0) {
                    if (p->n_pgo < 16 && p->n_steps < MAX_PLAN_DISPATCHES) {
                        cur_pgo = p->n_pgo++;
                        TnPgoGroup* g = &p->pgo[cur_pgo];
                        memset(g, 0, sizeof(TnPgoGroup));
                        const char* bar = strchr(line + 10, '|');
                        size_t len = bar ? (size_t)(bar - (line + 10)) : strlen(line + 10);
                        if (len >= sizeof(g->group_name)) len = sizeof(g->group_name) - 1;
                        memcpy(g->group_name, line + 10, len);
                        g->step_idx = p->n_steps++;
                    }
                } else if (strncmp(line, "PGO_CAND|", 9) == 0 && cur_pgo >= 0) {
                    TnPgoGroup* g = &p->pgo[cur_pgo];
                    if (g->n_cands < MAX_PGO_CANDS) {
                        TnPlanStep* st = &g->cands[g->n_cands++];
                        memset(st, 0, sizeof(TnPlanStep));
                        parse_step_line(p, st, line + 9);
                    }
                } else if (strncmp(line, "PGO_POST|", 9) == 0 && cur_pgo >= 0) {
                    TnPgoGroup* g = &p->pgo[cur_pgo];
                    if (g->n_cands > 0) {
                        parse_post_line(p, &g->cands[g->n_cands - 1], line + 9);
                    }
                } else if (strcmp(line, "PGO_END") == 0 && cur_pgo >= 0) {
                    TnPgoGroup* g = &p->pgo[cur_pgo];
                    if (g->n_cands > 0) {
                        p->steps[g->step_idx] = g->cands[0];
                    }
                    cur_pgo = -1;
                }
            }
            line = strtok_r(NULL, "\n", &lsave);
        }
        free(dup);
        if (!in_graph) {
            p->lib = nil; p->arena_buf = nil;
            return 0;
        }
        p->active = 1;
        return (int64_t)slot;
    }
}

static inline float tn_f16_to_f32(uint16_t h) {
    __fp16 v;
    memcpy(&v, &h, 2);
    return (float)v;
}

static inline uint8_t tn_quant_e2m1(float x) {
    static const float pos[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    uint8_t sign = (x < 0.0f) ? 8u : 0u;
    float ax = fabsf(x);
    uint8_t best = 0;
    float best_d = 1e30f;
    for (uint8_t i = 0; i < 8; i++) {
        float d = fabsf(ax - pos[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return (uint8_t)(sign | best);
}

int64_t tn_metal_plan_set_input(int64_t plan_h, int64_t input_idx, int64_t buf_handle) {
    if (plan_h <= 0 || plan_h >= MAX_PLANS || !g_plans[plan_h].active) return 0;
    if (buf_handle <= 0 || buf_handle >= MAX_HANDLES || g_buffers[buf_handle] == nil) return 0;
    @autoreleasepool {
        TnMetalPlan* p = &g_plans[plan_h];
        for (int i = 0; i < p->n_bufs; i++) {
            if (p->bufs[i].kind == 0 && p->bufs[i].io_idx == (int)input_idx) {
                id<MTLBuffer> src = g_buffers[buf_handle];
                if (strcmp(p->bufs[i].repack, "TRANSPOSE_NK") == 0 && p->bufs[i].rows > 0 && p->bufs[i].cols > 0) {
                    // Generic load-time 2D transpose [K, N] -> [N, K] (128-bit coalesced weight layout!)
                    size_t K = (size_t)p->bufs[i].rows;
                    size_t N = (size_t)p->bufs[i].cols;
                    size_t bytes = [src length];
                    id<MTLBuffer> dst = [g_mtl_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
                    if (bytes == K * N * 2) {
                        const uint16_t* sp = (const uint16_t*)[src contents];
                        uint16_t* dp = (uint16_t*)[dst contents];
                        for (size_t k = 0; k < K; k++)
                            for (size_t n = 0; n < N; n++)
                                dp[n * K + k] = sp[k * N + n];
                    } else if (bytes == K * N * 4) {
                        const uint32_t* sp = (const uint32_t*)[src contents];
                        uint32_t* dp = (uint32_t*)[dst contents];
                        for (size_t k = 0; k < K; k++)
                            for (size_t n = 0; n < N; n++)
                                dp[n * K + k] = sp[k * N + n];
                    }
                    p->bufs[i].buf = dst;
                } else if (strcmp(p->bufs[i].repack, "PACK_MXFP4_NK_SB16") == 0 && p->bufs[i].rows > 0 && p->bufs[i].cols > 0) {
                    // Stage 3 Interleaved SuperBlock16 OCP MXFP4: 16 blocks (512 weights) = 256B uint4 nibbles + 16B uint4 scales = 272B (16B aligned, SINGLE DRAM stream!)
                    size_t K = (size_t)p->bufs[i].rows;
                    size_t N = (size_t)p->bufs[i].cols;
                    size_t n_sb = K / 512;
                    size_t total_bytes = N * n_sb * 272;
                    id<MTLBuffer> dst = [g_mtl_device newBufferWithLength:(total_bytes > 64 ? total_bytes : 64)
                                                               options:MTLResourceStorageModeShared];
                    const uint16_t* sp = (const uint16_t*)[src contents];
                    uint8_t* out_base = (uint8_t*)[dst contents];
                    for (size_t n = 0; n < N; n++) {
                        for (size_t sb = 0; sb < n_sb; sb++) {
                            uint8_t* sb_ptr = out_base + (n * n_sb + sb) * 272;
                            uint8_t* sc_ptr = sb_ptr + 256;
                            for (size_t bi = 0; bi < 16; bi++) {
                                size_t blk = sb * 16 + bi;
                                float vals[32];
                                float mx = 0.0f;
                                for (size_t j = 0; j < 32; j++) {
                                    float v = tn_f16_to_f32(sp[(blk * 32 + j) * N + n]);
                                    vals[j] = v;
                                    float av = fabsf(v);
                                    if (av > mx) mx = av;
                                }
                                int exp_i = (mx > 1e-12f) ? (int)ceilf(log2f(mx / 6.0f)) : -126;
                                if (exp_i < -126) exp_i = -126;
                                if (exp_i > 127) exp_i = 127;
                                uint8_t e8m0 = (uint8_t)(exp_i + 127);
                                uint32_t fbits = ((uint32_t)e8m0) << 23;
                                float sc = 1.0f;
                                memcpy(&sc, &fbits, 4);
                                float inv_sc = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
                                sc_ptr[bi] = e8m0;
                                uint8_t* bp = sb_ptr + bi * 16;
                                for (size_t j = 0; j < 16; j++) {
                                    uint8_t lo = tn_quant_e2m1(vals[j * 2] * inv_sc);
                                    uint8_t hi = tn_quant_e2m1(vals[j * 2 + 1] * inv_sc);
                                    bp[j] = (uint8_t)((hi << 4) | (lo & 0x0Fu));
                                }
                            }
                        }
                    }
                    p->bufs[i].buf = dst;
                } else if (strcmp(p->bufs[i].repack, "PACK_BSR32_NK") == 0 && p->bufs[i].rows > 0 && p->bufs[i].cols > 0) {
                    // Stage 3 True Compressed 32x32 Block Sparse Row (BSR-32) packing for ANY arbitrary sparse matrix!
                    // Layout (all 16B aligned): [4 u32 hdr] [row_ptr[n_tiles+1]] [k_idx[nnz]] [contiguous non-zero 32_n x 32_k FP16 blocks]
                    size_t K = (size_t)p->bufs[i].rows;
                    size_t N = (size_t)p->bufs[i].cols;
                    size_t n_tiles = N / 32;
                    size_t k_blocks = K / 32;
                    const uint16_t* sp = (const uint16_t*)[src contents];
                    uint32_t* row_ptr = (uint32_t*)calloc(n_tiles + 1, sizeof(uint32_t));
                    size_t max_blk = n_tiles * k_blocks;
                    uint32_t* k_idx = (uint32_t*)malloc((max_blk > 0 ? max_blk : 1) * sizeof(uint32_t));
                    uint32_t total_nnz = 0;
                    for (size_t tn = 0; tn < n_tiles; tn++) {
                        row_ptr[tn] = total_nnz;
                        for (size_t kb = 0; kb < k_blocks; kb++) {
                            int nz = 0;
                            for (size_t kr = 0; kr < 32 && !nz; kr++) {
                                const uint16_t* rowp = sp + (kb * 32 + kr) * N + tn * 32;
                                for (size_t nc = 0; nc < 32; nc++) {
                                    if ((rowp[nc] & 0x7FFFu) != 0) { nz = 1; break; }
                                }
                            }
                            if (nz) {
                                k_idx[total_nnz++] = (uint32_t)(kb * 32);
                            }
                        }
                    }
                    row_ptr[n_tiles] = total_nnz;
                    size_t rp_u32 = 4 + n_tiles + 1;
                    size_t kidx_off_u32 = (rp_u32 + 3) & ~((size_t)3);
                    size_t data_off_bytes = ((kidx_off_u32 + total_nnz) * 4 + 15) & ~((size_t)15);
                    size_t total_bytes = data_off_bytes + (size_t)total_nnz * 32 * 32 * 2;
                    id<MTLBuffer> dst = [g_mtl_device newBufferWithLength:(total_bytes > 64 ? total_bytes : 64)
                                                               options:MTLResourceStorageModeShared];
                    uint32_t* hdr = (uint32_t*)[dst contents];
                    memset(hdr, 0, data_off_bytes);
                    hdr[0] = (uint32_t)kidx_off_u32;
                    hdr[1] = (uint32_t)(data_off_bytes / 2);
                    hdr[2] = (uint32_t)n_tiles;
                    hdr[3] = total_nnz;
                    memcpy(hdr + 4, row_ptr, (n_tiles + 1) * sizeof(uint32_t));
                    if (total_nnz > 0) {
                        memcpy(hdr + kidx_off_u32, k_idx, total_nnz * sizeof(uint32_t));
                        uint16_t* data_out = ((uint16_t*)[dst contents]) + (data_off_bytes / 2);
                        for (size_t tn = 0; tn < n_tiles; tn++) {
                            for (uint32_t b_i = row_ptr[tn]; b_i < row_ptr[tn + 1]; b_i++) {
                                size_t kt = (size_t)k_idx[b_i];
                                uint16_t* blk_dst = data_out + (size_t)b_i * 1024;
                                // Store block transposed as [32_n][32_k] for zero-scatter 128-bit uint4 loads!
                                for (size_t kr = 0; kr < 32; kr++) {
                                    const uint16_t* sr = sp + (kt + kr) * N + tn * 32;
                                    for (size_t nc = 0; nc < 32; nc++) {
                                        blk_dst[nc * 32 + kr] = sr[nc];
                                    }
                                }
                            }
                        }
                    }
                    free(row_ptr);
                    free(k_idx);
                    p->bufs[i].buf = dst;
                } else if (strcmp(p->bufs[i].repack, "PACK_MXFP4_NK") == 0 && p->bufs[i].rows > 0 && p->bufs[i].cols > 0) {
                    // Generic load-time OCP MXFP4 packing: [K, N] FP16 -> [N, K/2] 128-bit aligned E2M1 nibbles + [N, K/32] E8M0 scales!
                    size_t K = (size_t)p->bufs[i].rows;
                    size_t N = (size_t)p->bufs[i].cols;
                    size_t n_blk = K / 32;
                    size_t nib_bytes = N * n_blk * 16;
                    size_t total_bytes = nib_bytes + N * n_blk;
                    id<MTLBuffer> dst = [g_mtl_device newBufferWithLength:(total_bytes > 64 ? total_bytes : 64)
                                                               options:MTLResourceStorageModeShared];
                    const uint16_t* sp = (const uint16_t*)[src contents];
                    uint8_t* nib_out = (uint8_t*)[dst contents];
                    uint8_t* sc_out = nib_out + nib_bytes;
                    for (size_t n = 0; n < N; n++) {
                        for (size_t blk = 0; blk < n_blk; blk++) {
                            float vals[32];
                            float mx = 0.0f;
                            for (size_t j = 0; j < 32; j++) {
                                float v = tn_f16_to_f32(sp[(blk * 32 + j) * N + n]);
                                vals[j] = v;
                                float av = fabsf(v);
                                if (av > mx) mx = av;
                            }
                            int exp_i = (mx > 1e-12f) ? (int)ceilf(log2f(mx / 6.0f)) : -126;
                            if (exp_i < -126) exp_i = -126;
                            if (exp_i > 127) exp_i = 127;
                            uint8_t e8m0 = (uint8_t)(exp_i + 127);
                            uint32_t fbits = ((uint32_t)e8m0) << 23;
                            float sc = 1.0f;
                            memcpy(&sc, &fbits, 4);
                            float inv_sc = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
                            sc_out[n * n_blk + blk] = e8m0;
                            uint8_t* bp = nib_out + (n * n_blk + blk) * 16;
                            for (size_t j = 0; j < 16; j++) {
                                uint8_t lo = tn_quant_e2m1(vals[j * 2] * inv_sc);
                                uint8_t hi = tn_quant_e2m1(vals[j * 2 + 1] * inv_sc);
                                bp[j] = (uint8_t)((hi << 4) | (lo & 0x0Fu));
                            }
                        }
                    }
                    p->bufs[i].buf = dst;
                } else if (strcmp(p->bufs[i].repack, "PACK_INT8_ROW_NK") == 0 && p->bufs[i].rows > 0 && p->bufs[i].cols > 0) {
                    // Facility #6: Dynamic Per-Row INT8 Quantization [K, N] FP16 -> [N, K] int8_t + aligned [N] FP32 scales (-50% weight DRAM!)
                    size_t K = (size_t)p->bufs[i].rows;
                    size_t N = (size_t)p->bufs[i].cols;
                    size_t q_bytes_aligned = ((N * K) + 15) & ~((size_t)15);
                    size_t total_bytes = q_bytes_aligned + N * sizeof(float);
                    id<MTLBuffer> dst = [g_mtl_device newBufferWithLength:(total_bytes > 64 ? total_bytes : 64)
                                                               options:MTLResourceStorageModeShared];
                    const uint16_t* sp = (const uint16_t*)[src contents];
                    int8_t* qp = (int8_t*)[dst contents];
                    float* scales = (float*)(((uint8_t*)[dst contents]) + q_bytes_aligned);
                    for (size_t n = 0; n < N; n++) {
                        float mx = 0.0f;
                        for (size_t k = 0; k < K; k++) {
                            float av = fabsf(tn_f16_to_f32(sp[k * N + n]));
                            if (av > mx) mx = av;
                        }
                        float sc = (mx > 1e-12f) ? (mx / 127.0f) : 1.0f;
                        float inv_sc = 1.0f / sc;
                        scales[n] = sc;
                        for (size_t k = 0; k < K; k++) {
                            float v = tn_f16_to_f32(sp[k * N + n]) * inv_sc;
                            long qv = lroundf(v);
                            if (qv < -127) qv = -127;
                            if (qv > 127) qv = 127;
                            qp[n * K + k] = (int8_t)qv;
                        }
                    }
                    p->bufs[i].buf = dst;
                } else {
                    p->bufs[i].buf = src;
                }
                p->bufs[i].offset = 0;
                p->icb = nil;
                for (int ci = 0; ci < p->n_concats; ci++) {
                    for (int si = 0; si < p->concats[ci].n_srcs; si++) {
                        if (p->concats[ci].src_buf_idx[si] == i) p->concats[ci].built = 0;
                    }
                }
                return 1;
            }
        }
        return 0;
    }
}

int64_t tn_metal_plan_set_output(int64_t plan_h, int64_t output_idx, int64_t buf_handle) {
    if (plan_h <= 0 || plan_h >= MAX_PLANS || !g_plans[plan_h].active) return 0;
    if (buf_handle <= 0 || buf_handle >= MAX_HANDLES || g_buffers[buf_handle] == nil) return 0;
    TnMetalPlan* p = &g_plans[plan_h];
    for (int i = 0; i < p->n_bufs; i++) {
        if (p->bufs[i].kind == 1 && p->bufs[i].io_idx == (int)output_idx) {
            p->bufs[i].buf = g_buffers[buf_handle];
            p->bufs[i].offset = 0;
            p->icb = nil;
            return 1;
        }
    }
    return 0;
}

static void plan_prepare_derived(TnMetalPlan* p) {
    // Facility #2: Build horizontal packed `[total_n, k]` weight buffer once from sibling `[k, n_per]` inputs!
    for (int ci = 0; ci < p->n_concats; ci++) {
        TnConcatSpec* c = &p->concats[ci];
        if (c->built || c->dst_buf_idx < 0 || c->n_srcs <= 0) continue;
        int ready = 1;
        for (int s = 0; s < c->n_srcs; s++) {
            int bi = c->src_buf_idx[s];
            if (bi < 0 || p->bufs[bi].buf == nil) { ready = 0; break; }
        }
        if (!ready) continue;
        size_t total_n = c->n_per * (size_t)c->n_srcs;
        size_t src_bytes = [p->bufs[c->src_buf_idx[0]].buf length];
        size_t elem_b = (src_bytes == c->k * c->n_per * 4) ? 4 : 2;
        size_t dst_bytes = total_n * c->k * elem_b;
        id<MTLBuffer> dst = [g_mtl_device newBufferWithLength:(dst_bytes > 64 ? dst_bytes : 64)
                                                   options:MTLResourceStorageModeShared];
        if (elem_b == 2) {
            uint16_t* dp = (uint16_t*)[dst contents];
            for (int s = 0; s < c->n_srcs; s++) {
                const uint16_t* sp = (const uint16_t*)[p->bufs[c->src_buf_idx[s]].buf contents];
                size_t n_off = (size_t)s * c->n_per;
                for (size_t kk = 0; kk < c->k; kk++) {
                    for (size_t nn = 0; nn < c->n_per; nn++) {
                        dp[(n_off + nn) * c->k + kk] = sp[kk * c->n_per + nn];
                    }
                }
            }
        } else {
            uint32_t* dp = (uint32_t*)[dst contents];
            for (int s = 0; s < c->n_srcs; s++) {
                const uint32_t* sp = (const uint32_t*)[p->bufs[c->src_buf_idx[s]].buf contents];
                size_t n_off = (size_t)s * c->n_per;
                for (size_t kk = 0; kk < c->k; kk++) {
                    for (size_t nn = 0; nn < c->n_per; nn++) {
                        dp[(n_off + nn) * c->k + kk] = sp[kk * c->n_per + nn];
                    }
                }
            }
        }
        p->bufs[c->dst_buf_idx].buf = dst;
        p->bufs[c->dst_buf_idx].offset = 0;
        p->bufs[c->dst_buf_idx].size = dst_bytes;
        c->built = 1;
    }
    // Facility #3: Zero-Copy Buffer Donation (`DONATE|out_idx|in_idx`)
    for (int di = 0; di < p->n_donates; di++) {
        int in_bi = -1, out_bi = -1;
        for (int i = 0; i < p->n_bufs; i++) {
            if (p->bufs[i].kind == 0 && p->bufs[i].io_idx == p->donate_in[di]) in_bi = i;
            if (p->bufs[i].kind == 1 && p->bufs[i].io_idx == p->donate_out[di]) out_bi = i;
        }
        if (in_bi >= 0 && out_bi >= 0 && p->bufs[in_bi].buf != nil) {
            p->bufs[out_bi].buf = p->bufs[in_bi].buf;
            p->bufs[out_bi].offset = p->bufs[in_bi].offset;
        }
    }
}

int64_t tn_metal_plan_set_dyn_m(int64_t plan_h, int64_t m_actual) {
    if (plan_h <= 0 || plan_h >= MAX_PLANS || !g_plans[plan_h].active) return 0;
    if (m_actual < 1) m_actual = 1;
    TnMetalPlan* p = &g_plans[plan_h];
    p->dyn_m_actual = (int)m_actual;
    if (p->dyn_ctrl_buf_idx >= 0 && p->bufs[p->dyn_ctrl_buf_idx].buf != nil) {
        uint32_t* cp = (uint32_t*)[p->bufs[p->dyn_ctrl_buf_idx].buf contents];
        cp[0] = (uint32_t)m_actual;
    }
    if (p->n_buckets > 0 && p->dyn_step_idx >= 0) {
        int chosen = p->n_buckets - 1;
        for (int b = 0; b < p->n_buckets; b++) {
            if ((int)m_actual >= p->buckets[b].min_m && (int)m_actual <= p->buckets[b].max_m) {
                chosen = b;
                break;
            }
        }
        TnDynBucket* bk = &p->buckets[chosen];
        p->active_bucket = chosen;
        p->steps[p->dyn_step_idx] = bk->step;
        p->steps[p->dyn_step_idx].tgy = (bk->tgy_mode == 1) ? (NSUInteger)((m_actual + 31) / 32)
                                      : (bk->tgy_mode == 2) ? (NSUInteger)((m_actual + 63) / 64)
                                      : (NSUInteger)m_actual;
        p->icb = nil;
    }
    return 1;
}

static void plan_encode_step(TnMetalPlan* p, id<MTLComputeCommandEncoder> enc, const TnPlanStep* st) {
    if (!st->pso) return;
    [enc setComputePipelineState:st->pso];
    for (int a = 0; a < st->n_args; a++) {
        int bi = st->arg_idx[a];
        if (bi >= 0 && bi < p->n_bufs && p->bufs[bi].buf != nil) {
            [enc setBuffer:p->bufs[bi].buf offset:p->bufs[bi].offset atIndex:(NSUInteger)a];
        }
    }
    if (st->smem > 0) {
        [enc setThreadgroupMemoryLength:st->smem atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(st->tgx, st->tgy, st->tgz)
        threadsPerThreadgroup:MTLSizeMake(st->tx, st->ty, st->tz)];
    if (st->has_post && st->post_pso) {
        [enc setComputePipelineState:st->post_pso];
        for (int a = 0; a < st->post_n_args; a++) {
            int bi = st->post_arg_idx[a];
            if (bi >= 0 && bi < p->n_bufs && p->bufs[bi].buf != nil) {
                [enc setBuffer:p->bufs[bi].buf offset:p->bufs[bi].offset atIndex:(NSUInteger)a];
            }
        }
        [enc dispatchThreadgroups:MTLSizeMake(st->post_tgx, st->post_tgy, st->post_tgz)
            threadsPerThreadgroup:MTLSizeMake(st->post_tx, st->post_ty, st->post_tz)];
    }
}

static void plan_run_pgo_autotune(TnMetalPlan* p) {
    if (!p->pgo_enabled || p->pgo_tuned || p->n_pgo == 0) return;
    p->pgo_tuned = 1;
    const char* cache_path = getenv("TENON_GPU_CACHE");
    if (!cache_path || !*cache_path) cache_path = "/tmp/.tenon-gpu-cache";
    int force_fresh = getenv("TENON_GPU_PGO_FRESH") != NULL;

    for (int gi = 0; gi < p->n_pgo; gi++) {
        TnPgoGroup* g = &p->pgo[gi];
        if (g->n_cands <= 1) continue;
        // Include full input shape signature + candidate count so shape/compiler changes never hit stale entries!
        uint64_t shape_sig = (uint64_t)g->n_cands * 1315423911ull;
        for (int bi = 0; bi < p->n_bufs; bi++) {
            if (p->bufs[bi].kind == 0) {
                shape_sig = shape_sig * 1099511628211ull ^ ((uint64_t)p->bufs[bi].rows << 16) ^ (uint64_t)p->bufs[bi].cols;
            }
        }
        char key[384];
        snprintf(key, sizeof(key), "%s|%s|%s|%016llx|%s", g_device_name_buf, p->graph_name, g->group_name, (unsigned long long)shape_sig, p->opts_tag);

        int cached_winner = -1;
        if (!force_fresh) {
            FILE* cf = fopen(cache_path, "r");
            if (cf) {
                char line[512];
                while (fgets(line, sizeof(line), cf)) {
                    char* eq = strchr(line, '=');
                    if (eq) {
                        *eq = 0;
                        if (strcmp(line, key) == 0) {
                            cached_winner = atoi(eq + 1); // keep scanning so latest entry wins
                        }
                    }
                }
                fclose(cf);
            }
        }

        if (cached_winner >= 0 && cached_winner < g->n_cands) {
            g->winner = cached_winner;
            p->steps[g->step_idx] = g->cands[cached_winner];
            continue;
        }

        // Live hardware micro-benchmark of EVERY compiler-emitted Tile-IR candidate!
        // First wake GPU clocks out of idle DVFS so candidate 0 isn't penalized vs candidate N!
        if (g->cands[0].pso) {
            double wt0 = tn_now_ms();
            int wakes = 0;
            while ((tn_now_ms() - wt0) < 12.0 && wakes < 16) {
                id<MTLCommandBuffer> wcmd = [g_mtl_queue commandBuffer];
                id<MTLComputeCommandEncoder> wenc = [wcmd computeCommandEncoder];
                for (int w = 0; w < 4; w++) plan_encode_step(p, wenc, &g->cands[0]);
                [wenc endEncoding];
                [wcmd commit];
                [wcmd waitUntilCompleted];
                wakes++;
            }
        }
        int best_c = 0;
        double best_ms = 1e30;
        for (int c = 0; c < g->n_cands; c++) {
            const TnPlanStep* cand = &g->cands[c];
            if (!cand->pso) continue;
            // Warmup 2
            id<MTLCommandBuffer> wcmd = [g_mtl_queue commandBuffer];
            id<MTLComputeCommandEncoder> wenc = [wcmd computeCommandEncoder];
            for (int w = 0; w < 2; w++) plan_encode_step(p, wenc, cand);
            [wenc endEncoding];
            [wcmd commit];
            [wcmd waitUntilCompleted];
            if ([wcmd status] != MTLCommandBufferStatusCompleted) continue;

            // Timed 12 back-to-back GPU runs inside ONE CB
            double t0 = tn_now_ms();
            id<MTLCommandBuffer> tcmd = [g_mtl_queue commandBuffer];
            id<MTLComputeCommandEncoder> tenc = [tcmd computeCommandEncoder];
            for (int it = 0; it < 12; it++) plan_encode_step(p, tenc, cand);
            [tenc endEncoding];
            [tcmd commit];
            [tcmd waitUntilCompleted];
            double dt = (tn_now_ms() - t0) / 12.0;
            if ([tcmd status] == MTLCommandBufferStatusCompleted && dt < best_ms) {
                best_ms = dt;
                best_c = c;
            }
        }
        g->winner = best_c;
        g->winner_ms = best_ms;
        p->steps[g->step_idx] = g->cands[best_c];

        // Persist empirical winner into `.tenon-gpu-cache`
        FILE* wf = fopen(cache_path, "a");
        if (wf) {
            fprintf(wf, "%s=%d\n", key, best_c);
            fclose(wf);
        }
    }
}

static void plan_exec_once_sync_per_op(TnMetalPlan* p) {
    for (int i = 0; i < p->n_bufs; i++) {
        if (p->bufs[i].kind == 4 && p->bufs[i].buf != nil) {
            memset([p->bufs[i].buf contents], 0, p->bufs[i].size);
        }
    }
    for (int s = 0; s < p->n_steps; s++) {
        id<MTLCommandBuffer> cmd = [g_mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        plan_encode_step(p, enc, &p->steps[s]);
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

static void plan_exec_batch_single_cb(TnMetalPlan* p, int64_t reps) {
    id<MTLCommandBuffer> cmd = [g_mtl_queue commandBuffer];
    int has_zero_scratch = 0;
    for (int i = 0; i < p->n_bufs; i++) {
        if (p->bufs[i].kind == 4 && p->bufs[i].buf != nil) { has_zero_scratch = 1; break; }
    }
    if (!has_zero_scratch) {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        for (int64_t r = 0; r < reps; r++) {
            for (int s = 0; s < p->n_steps; s++) {
                plan_encode_step(p, enc, &p->steps[s]);
            }
        }
        [enc endEncoding];
    } else {
        for (int64_t r = 0; r < reps; r++) {
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            for (int i = 0; i < p->n_bufs; i++) {
                if (p->bufs[i].kind == 4 && p->bufs[i].buf != nil) {
                    [blit fillBuffer:p->bufs[i].buf range:NSMakeRange(0, p->bufs[i].size) value:0];
                }
            }
            [blit endEncoding];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            for (int s = 0; s < p->n_steps; s++) {
                plan_encode_step(p, enc, &p->steps[s]);
            }
            [enc endEncoding];
        }
    }
    [cmd commit];
    [cmd waitUntilCompleted];
}

double tn_metal_plan_bench_ms(int64_t plan_h, int64_t warmup, int64_t iters) {
    if (plan_h <= 0 || plan_h >= MAX_PLANS || !g_plans[plan_h].active) return -1.0;
    if (iters <= 0) iters = 1;
    @autoreleasepool {
        TnMetalPlan* p = &g_plans[plan_h];
        plan_prepare_derived(p);
        plan_run_pgo_autotune(p);
        if (warmup > 0) {
            // Ensure Apple Silicon GPU exits idle DVFS P-state cleanly before timed runs
            // (identical to Triton do_bench ~15ms warmup so cached PGO & fresh PGO measure at peak clock!)
            double wt0 = tn_now_ms();
            int64_t w_done = 0;
            do {
                int64_t chunk = (w_done == 0) ? warmup : (warmup > 4 ? warmup : 4);
                if (p->single_cb) plan_exec_batch_single_cb(p, chunk);
                else for (int64_t w = 0; w < chunk; w++) plan_exec_once_sync_per_op(p);
                w_done += chunk;
            } while ((tn_now_ms() - wt0) < 30.0 && w_done < 256);
        }
        double t0 = tn_now_ms();
        if (p->single_cb) {
            plan_exec_batch_single_cb(p, iters);
        } else {
            for (int64_t i = 0; i < iters; i++) plan_exec_once_sync_per_op(p);
        }
        return (tn_now_ms() - t0) / (double)iters;
    }
}

static void plan_build_icb(TnMetalPlan* p) {
    if (p->icb != nil || !p->replay_enabled || p->n_steps <= 0) return;
    // Count total compute commands including any post-reduction steps
    int total_cmds = 0;
    for (int s = 0; s < p->n_steps; s++) {
        if (p->steps[s].pso) total_cmds++;
        if (p->steps[s].has_post && p->steps[s].post_pso) total_cmds++;
    }
    if (total_cmds <= 0) return;
    MTLIndirectCommandBufferDescriptor* desc = [[MTLIndirectCommandBufferDescriptor alloc] init];
    desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    desc.inheritBuffers = NO;
    desc.inheritPipelineState = NO;
    desc.maxKernelBufferBindCount = 16;
    id<MTLIndirectCommandBuffer> icb = [g_mtl_device newIndirectCommandBufferWithDescriptor:desc
                                                                            maxCommandCount:(NSUInteger)total_cmds
                                                                                    options:MTLResourceStorageModeShared];
    if (!icb) return;
    NSUInteger ci = 0;
    for (int s = 0; s < p->n_steps; s++) {
        const TnPlanStep* st = &p->steps[s];
        if (st->pso) {
            id<MTLComputePipelineState> icb_pso = plan_compile_fn_icb(p, st->kname, 1);
            id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:ci++];
            [cmd setComputePipelineState:(icb_pso ? icb_pso : st->pso)];
            for (int a = 0; a < st->n_args; a++) {
                int bi = st->arg_idx[a];
                if (bi >= 0 && bi < p->n_bufs && p->bufs[bi].buf != nil) {
                    [cmd setKernelBuffer:p->bufs[bi].buf offset:p->bufs[bi].offset atIndex:(NSUInteger)a];
                }
            }
            if (st->smem > 0) [cmd setThreadgroupMemoryLength:st->smem atIndex:0];
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(st->tgx, st->tgy, st->tgz)
                          threadsPerThreadgroup:MTLSizeMake(st->tx, st->ty, st->tz)];
        }
        if (st->has_post && st->post_pso) {
            id<MTLComputePipelineState> icb_post = plan_compile_fn_icb(p, st->post_kname, 1);
            id<MTLIndirectComputeCommand> cmd = [icb indirectComputeCommandAtIndex:ci++];
            [cmd setComputePipelineState:(icb_post ? icb_post : st->post_pso)];
            for (int a = 0; a < st->post_n_args; a++) {
                int bi = st->post_arg_idx[a];
                if (bi >= 0 && bi < p->n_bufs && p->bufs[bi].buf != nil) {
                    [cmd setKernelBuffer:p->bufs[bi].buf offset:p->bufs[bi].offset atIndex:(NSUInteger)a];
                }
            }
            [cmd concurrentDispatchThreadgroups:MTLSizeMake(st->post_tgx, st->post_tgy, st->post_tgz)
                          threadsPerThreadgroup:MTLSizeMake(st->post_tx, st->post_ty, st->post_tz)];
        }
    }
    p->icb = icb;
    p->icb_steps = (int)ci;
}

static void plan_exec_icb_steps(TnMetalPlan* p, int64_t steps) {
    id<MTLCommandBuffer> cmd = [g_mtl_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    for (int i = 0; i < p->n_bufs; i++) {
        if (p->bufs[i].buf != nil) {
            [enc useResource:p->bufs[i].buf usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
        }
    }
    NSRange rng = NSMakeRange(0, (NSUInteger)p->icb_steps);
    for (int64_t s = 0; s < steps; s++) {
        [enc executeCommandsInBuffer:p->icb withRange:rng];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
    }
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
}

static void plan_exec_host_eager_steps(TnMetalPlan* p, int64_t steps) {
    // Realistic per-step eager host command-buffer allocation + kernel bind + commit!
    id<MTLCommandBuffer> last_cmd = nil;
    for (int64_t s = 0; s < steps; s++) {
        id<MTLCommandBuffer> cmd = [g_mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        for (int k = 0; k < p->n_steps; k++) {
            plan_encode_step(p, enc, &p->steps[k]);
        }
        [enc endEncoding];
        [cmd commit];
        last_cmd = cmd;
    }
    if (last_cmd) [last_cmd waitUntilCompleted];
}

/// Facility #4: Device-Side Graph Replay (`MTLIndirectCommandBuffer` ICB vs eager per-step host submission)!
double tn_metal_plan_bench_replay_ms(int64_t plan_h, int64_t steps, int64_t reps) {
    if (plan_h <= 0 || plan_h >= MAX_PLANS || !g_plans[plan_h].active) return -1.0;
    if (steps <= 0) steps = 1;
    if (reps <= 0) reps = 1;
    @autoreleasepool {
        TnMetalPlan* p = &g_plans[plan_h];
        plan_prepare_derived(p);
        plan_run_pgo_autotune(p);
        if (p->replay_enabled) plan_build_icb(p);

        // Warmup 30ms to wake Apple Silicon GPU DVFS cleanly
        double wt0 = tn_now_ms();
        int w_it = 0;
        do {
            if (p->replay_enabled && p->icb != nil) plan_exec_icb_steps(p, steps);
            else plan_exec_host_eager_steps(p, steps);
            w_it++;
        } while ((tn_now_ms() - wt0) < 30.0 && w_it < 32);

        double t0 = tn_now_ms();
        for (int64_t r = 0; r < reps; r++) {
            if (p->replay_enabled && p->icb != nil) plan_exec_icb_steps(p, steps);
            else plan_exec_host_eager_steps(p, steps);
        }
        return (tn_now_ms() - t0) / (double)reps;
    }
}

/// Facility #9: Dual-Queue Async Blit-Prefetch || Compute Overlap (`g_mtl_copy_queue` + `MTLEvent` ping-pong staging)!
double tn_metal_plan_bench_stream_ms(int64_t plan_h, int64_t weight_input_idx, int64_t layers, int64_t overlap, int64_t iters) {
    if (plan_h <= 0 || plan_h >= MAX_PLANS || !g_plans[plan_h].active) return -1.0;
    if (layers <= 0) layers = 1;
    if (iters <= 0) iters = 1;
    @autoreleasepool {
        TnMetalPlan* p = &g_plans[plan_h];
        plan_prepare_derived(p);
        plan_run_pgo_autotune(p);

        int w_bi = -1;
        for (int i = 0; i < p->n_bufs; i++) {
            if (p->bufs[i].kind == 0 && p->bufs[i].io_idx == (int)weight_input_idx) { w_bi = i; break; }
        }
        if (w_bi < 0 || p->bufs[w_bi].buf == nil) return -1.0;

        id<MTLBuffer> orig_w = p->bufs[w_bi].buf;
        NSUInteger w_len = [orig_w length];
        if (layers > 16) layers = 16;

        // Allocate `layers` distinct shared host-staged weight copies + 2 Private GPU ping-pong buffers
        NSMutableArray<id<MTLBuffer>>* host_layers = [NSMutableArray arrayWithCapacity:(NSUInteger)layers];
        for (int64_t l = 0; l < layers; l++) {
            id<MTLBuffer> sb = [g_mtl_device newBufferWithLength:w_len options:MTLResourceStorageModeShared];
            memcpy([sb contents], [orig_w contents], w_len);
            [host_layers addObject:sb];
        }
        id<MTLBuffer> priv0 = [g_mtl_device newBufferWithLength:w_len options:MTLResourceStorageModePrivate];
        id<MTLBuffer> priv1 = [g_mtl_device newBufferWithLength:w_len options:MTLResourceStorageModePrivate];
        id<MTLEvent> ev_ready = [g_mtl_device newEvent];
        id<MTLEvent> ev_done = [g_mtl_device newEvent];
        __block uint64_t sig_val = 0;

        void (^run_once)(void) = ^{
            if (!overlap || !p->overlap_enabled || !g_mtl_copy_queue) {
                // Serialized baseline: blit layer L -> compute layer L strictly back-to-back on ONE queue!
                id<MTLCommandBuffer> cmd = [g_mtl_queue commandBuffer];
                for (int64_t l = 0; l < layers; l++) {
                    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
                    [blit copyFromBuffer:host_layers[(NSUInteger)l] sourceOffset:0 toBuffer:priv0 destinationOffset:0 size:w_len];
                    [blit endEncoding];
                    p->bufs[w_bi].buf = priv0;
                    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                    for (int s = 0; s < p->n_steps; s++) plan_encode_step(p, enc, &p->steps[s]);
                    [enc endEncoding];
                }
                [cmd commit];
                [cmd waitUntilCompleted];
            } else {
                // Async Dual-Queue Overlap: `g_mtl_copy_queue` blits layer `L+1` into `priv[(L+1)&1]`
                // IN PARALLEL with `g_mtl_queue` running AMX compute on `priv[L&1]`, synced purely on GPU via `MTLEvent`!
                uint64_t base = sig_val;
                sig_val += (uint64_t)layers + 2ull;
                id<MTLCommandBuffer> last_comp = nil;
                for (int64_t l = 0; l < layers; l++) {
                    id<MTLBuffer> pbuf = (l & 1) ? priv1 : priv0;
                    id<MTLCommandBuffer> bcmd = [g_mtl_copy_queue commandBuffer];
                    if (l >= 2) {
                        // Wait until compute finished consuming `pbuf` from layer `l - 2`!
                        [bcmd encodeWaitForEvent:ev_done value:(base + (uint64_t)(l - 1))];
                    }
                    id<MTLBlitCommandEncoder> blit = [bcmd blitCommandEncoder];
                    [blit copyFromBuffer:host_layers[(NSUInteger)l] sourceOffset:0 toBuffer:pbuf destinationOffset:0 size:w_len];
                    [blit endEncoding];
                    [bcmd encodeSignalEvent:ev_ready value:(base + (uint64_t)(l + 1))];
                    [bcmd commit];

                    id<MTLCommandBuffer> ccmd = [g_mtl_queue commandBuffer];
                    [ccmd encodeWaitForEvent:ev_ready value:(base + (uint64_t)(l + 1))];
                    p->bufs[w_bi].buf = pbuf;
                    id<MTLComputeCommandEncoder> enc = [ccmd computeCommandEncoder];
                    for (int s = 0; s < p->n_steps; s++) plan_encode_step(p, enc, &p->steps[s]);
                    [enc endEncoding];
                    [ccmd encodeSignalEvent:ev_done value:(base + (uint64_t)(l + 1))];
                    [ccmd commit];
                    last_comp = ccmd;
                }
                if (last_comp) [last_comp waitUntilCompleted];
            }
        };

        // Warmup 30ms for stable DVFS
        double wt0 = tn_now_ms();
        int w_it = 0;
        do { run_once(); w_it++; } while ((tn_now_ms() - wt0) < 30.0 && w_it < 16);

        double t0 = tn_now_ms();
        for (int64_t it = 0; it < iters; it++) run_once();
        double avg_ms = (tn_now_ms() - t0) / (double)iters;

        p->bufs[w_bi].buf = orig_w;
        return avg_ms;
    }
}

int64_t tn_metal_plan_info(int64_t plan_h, int64_t field_id) {
    if (plan_h <= 0 || plan_h >= MAX_PLANS || !g_plans[plan_h].active) return 0;
    TnMetalPlan* p = &g_plans[plan_h];
    switch (field_id) {
        case 0: return (int64_t)p->arena_bytes;
        case 1: return (int64_t)p->naive_arena_bytes;
        case 2: return (p->n_pgo > 0) ? (int64_t)p->pgo[0].n_cands : 1;
        case 3: return (p->n_pgo > 0) ? (int64_t)p->pgo[0].winner : 0;
        case 4: return (int64_t)p->n_steps;
        case 5: return (int64_t)p->donated_bytes;
        case 6: return (int64_t)p->active_bucket;
        case 7: return (p->icb != nil) ? 1 : 0;
        default: return 0;
    }
}

int64_t tn_metal_plan_free(int64_t plan_h) {
    if (plan_h <= 0 || plan_h >= MAX_PLANS || !g_plans[plan_h].active) return 0;
    TnMetalPlan* p = &g_plans[plan_h];
    p->lib = nil;
    p->arena_buf = nil;
    p->icb = nil;
    for (int i = 0; i < p->n_bufs; i++) p->bufs[i].buf = nil;
    for (int s = 0; s < p->n_steps; s++) { p->steps[s].pso = nil; p->steps[s].post_pso = nil; }
    for (int g = 0; g < p->n_pgo; g++) {
        for (int c = 0; c < p->pgo[g].n_cands; c++) {
            p->pgo[g].cands[c].pso = nil;
            p->pgo[g].cands[c].post_pso = nil;
        }
    }
    for (int b = 0; b < p->n_buckets; b++) {
        p->buckets[b].step.pso = nil;
        p->buckets[b].step.post_pso = nil;
    }
    p->active = 0;
    return 1;
}
