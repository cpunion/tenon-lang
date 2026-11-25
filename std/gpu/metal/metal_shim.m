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
static char g_device_name_buf[256] = "Unavailable";

#define MAX_HANDLES 1024
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
