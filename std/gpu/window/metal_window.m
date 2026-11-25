// std/gpu/window/metal_window.m — Zero-Rust Native macOS Cocoa + CAMetalLayer Desktop Window & Retina Text Compositor
//
// Opens a real hardware-accelerated macOS desktop window (`NSWindow` + `CAMetalLayer`),
// renders batched Tenon `@gpu_struct QuadInstance` SDF rounded panels in ONE instanced Metal draw call,
// composites crisp Retina CoreText typography on top via a Metal texture pass,
// delivers live mouse click & keyboard events back to Tenon, and optionally exports a PNG snapshot!

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreText/CoreText.h>
#import <ImageIO/ImageIO.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

// Pending input event queue (mouse clicks + key presses + window close)
typedef struct {
    int64_t kind; // 0=none, 1=key, 2=mouse_down, 3=quit
    int64_t a;    // keycode / mouse_x
    int64_t b;    // char / mouse_y
} TnWinEvent;

#define MAX_EVENTS 64
static TnWinEvent g_events[MAX_EVENTS];
static int g_ev_head = 0, g_ev_tail = 0;

static void push_event(int64_t kind, int64_t a, int64_t b) {
    int next = (g_ev_tail + 1) % MAX_EVENTS;
    if (next == g_ev_head) return;
    g_events[g_ev_tail].kind = kind;
    g_events[g_ev_tail].a = a;
    g_events[g_ev_tail].b = b;
    g_ev_tail = next;
}

#define TN_MAX_INFLIGHT 3

static NSWindow* g_window = nil;
static CAMetalLayer* g_layer = nil;
static id<MTLDevice> g_dev = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLRenderPipelineState> g_quad_pso = nil;
static id<MTLRenderPipelineState> g_blit_pso = nil;
static id<MTLTexture> g_text_tex[TN_MAX_INFLIGHT] = { nil, nil, nil };
static dispatch_semaphore_t g_frame_sem = nil;
static int g_frame_slot = 0;
static uint8_t* g_text_pixels = NULL;
static CGContextRef g_text_ctx = NULL;
static int g_win_w = 840;
static int g_win_h = 540;
static float g_scale = 2.0f;
static int g_should_close = 0;
static char g_win_dev_name[256] = "Apple Metal GPU";

@interface TenonMetalView : NSView
@end

@implementation TenonMetalView
- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)isFlipped { return YES; }

- (void)mouseDown:(NSEvent *)event {
    NSPoint pt = [self convertPoint:[event locationInWindow] fromView:nil];
    push_event(2, (int64_t)pt.x, (int64_t)pt.y);
}

- (void)keyDown:(NSEvent *)event {
    NSString *chars = [event charactersIgnoringModifiers];
    int64_t ch = ([chars length] > 0) ? (int64_t)[chars characterAtIndex:0] : 0;
    push_event(1, (int64_t)[event keyCode], ch);
}
@end

@interface TenonWinDelegate : NSObject <NSWindowDelegate>
@end

@implementation TenonWinDelegate
- (void)windowWillClose:(NSNotification *)notification {
    (void)notification;
    g_should_close = 1;
    push_event(3, 0, 0);
}
@end

static TenonWinDelegate* g_delegate = nil;

static const char* kWinShadersMSL =
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
"vertex VSOut wquad_vs(uint vid [[vertex_id]], uint iid [[instance_id]],\n"
"                      device const QuadInstance* quads [[buffer(0)]],\n"
"                      constant Viewport& vp [[buffer(1)]]) {\n"
"    const float2 corners[6] = {\n"
"        float2(0.0, 0.0), float2(1.0, 0.0), float2(0.0, 1.0),\n"
"        float2(1.0, 0.0), float2(1.0, 1.0), float2(0.0, 1.0)\n"
"    };\n"
"    QuadInstance q = quads[iid];\n"
"    float2 uv = corners[vid];\n"
"    float2 px = float2(q.x + uv.x * q.w, q.y + uv.y * q.h);\n"
"    VSOut o;\n"
"    o.pos = float4((px.x / vp.width) * 2.0f - 1.0f, 1.0f - (px.y / vp.height) * 2.0f, 0.0f, 1.0f);\n"
"    o.local_px = uv * float2(q.w, q.h);\n"
"    o.size_px = float2(q.w, q.h);\n"
"    o.color = float4(q.r, q.g, q.b, q.a);\n"
"    o.radius = q.radius;\n"
"    o.border_w = q.border_w;\n"
"    o.border_rg = float2(q.br, q.bg);\n"
"    return o;\n"
"}\n"
"fragment float4 wquad_fs(VSOut in [[stage_in]]) {\n"
"    float2 half_sz = in.size_px * 0.5f;\n"
"    float2 p = in.local_px - half_sz;\n"
"    float rad = min(in.radius, min(half_sz.x, half_sz.y));\n"
"    float2 d = abs(p) - (half_sz - float2(rad));\n"
"    float sdf = length(max(d, 0.0f)) + min(max(d.x, d.y), 0.0f) - rad;\n"
"    if (sdf > 0.75f) discard_fragment();\n"
"    float aa = clamp(0.75f - sdf, 0.0f, 1.0f);\n"
"    float4 col = in.color;\n"
"    if (in.border_w > 0.0f && sdf > -in.border_w) {\n"
"        col = float4(in.border_rg.x, in.border_rg.y, 0.92f, 1.0f);\n"
"    }\n"
"    return float4(col.rgb, col.a * aa);\n"
"}\n"
"struct BlitOut { float4 pos [[position]]; float2 uv; };\n"
"vertex BlitOut wblit_vs(uint vid [[vertex_id]]) {\n"
"    const float2 pos[6] = {\n"
"        float2(-1.0,  1.0), float2( 1.0,  1.0), float2(-1.0, -1.0),\n"
"        float2( 1.0,  1.0), float2( 1.0, -1.0), float2(-1.0, -1.0)\n"
"    };\n"
"    const float2 uv[6] = {\n"
"        float2(0.0, 0.0), float2(1.0, 0.0), float2(0.0, 1.0),\n"
"        float2(1.0, 0.0), float2(1.0, 1.0), float2(0.0, 1.0)\n"
"    };\n"
"    BlitOut o; o.pos = float4(pos[vid], 0.0f, 1.0f); o.uv = uv[vid]; return o;\n"
"}\n"
"fragment float4 wblit_fs(BlitOut in [[stage_in]], texture2d<float> tex [[texture(0)]]) {\n"
"    constexpr sampler s(mag_filter::linear, min_filter::linear);\n"
"    return tex.sample(s, in.uv);\n"
"}\n";

static int ensure_pipelines(MTLPixelFormat fmt) {
    if (g_quad_pso != nil && g_blit_pso != nil) return 1;
    NSError* err = nil;
    id<MTLLibrary> lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:kWinShadersMSL]
                                             options:nil
                                               error:&err];
    if (lib == nil) {
        fprintf(stderr, "[tenon-gpui] shader error: %s\n", [[err localizedDescription] UTF8String]);
        return 0;
    }
    MTLRenderPipelineDescriptor* qd = [[MTLRenderPipelineDescriptor alloc] init];
    qd.vertexFunction = [lib newFunctionWithName:@"wquad_vs"];
    qd.fragmentFunction = [lib newFunctionWithName:@"wquad_fs"];
    qd.colorAttachments[0].pixelFormat = fmt;
    qd.colorAttachments[0].blendingEnabled = YES;
    qd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    qd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    qd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    qd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    g_quad_pso = [g_dev newRenderPipelineStateWithDescriptor:qd error:&err];
    if (g_quad_pso == nil) return 0;

    MTLRenderPipelineDescriptor* bd = [[MTLRenderPipelineDescriptor alloc] init];
    bd.vertexFunction = [lib newFunctionWithName:@"wblit_vs"];
    bd.fragmentFunction = [lib newFunctionWithName:@"wblit_fs"];
    bd.colorAttachments[0].pixelFormat = fmt;
    bd.colorAttachments[0].blendingEnabled = YES;
    bd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorOne; // premultiplied CoreGraphics RGBA!
    bd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    bd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    bd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    g_blit_pso = [g_dev newRenderPipelineStateWithDescriptor:bd error:&err];
    return g_blit_pso != nil ? 1 : 0;
}

static void init_text_surface(int w, int h, float scale) {
    int pw = (int)(w * scale);
    int ph = (int)(h * scale);
    if (g_text_ctx != NULL) { CGContextRelease(g_text_ctx); g_text_ctx = NULL; }
    if (g_text_pixels != NULL) { free(g_text_pixels); g_text_pixels = NULL; }
    g_text_pixels = (uint8_t*)calloc((size_t)(pw * ph * 4), 1);
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    g_text_ctx = CGBitmapContextCreate(
        g_text_pixels, (size_t)pw, (size_t)ph, 8, (size_t)(pw * 4), cs,
        kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big
    );
    CGColorSpaceRelease(cs);
    // Flip into top-left logical coordinates scaled by Retina factor
    CGContextTranslateCTM(g_text_ctx, 0, ph);
    CGContextScaleCTM(g_text_ctx, scale, -scale);
    CGContextSetAllowsAntialiasing(g_text_ctx, true);
    CGContextSetShouldSmoothFonts(g_text_ctx, true);

    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                  width:(NSUInteger)pw
                                                                                 height:(NSUInteger)ph
                                                                              mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    for (int i = 0; i < TN_MAX_INFLIGHT; i++) {
        g_text_tex[i] = [g_dev newTextureWithDescriptor:td];
    }
    if (g_frame_sem == nil) {
        g_frame_sem = dispatch_semaphore_create(TN_MAX_INFLIGHT);
    }
    g_frame_slot = 0;
}

int64_t tn_win_open(int64_t title_cstr, int64_t width, int64_t height, int64_t headless) {
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (g_dev == nil) return 0;
        g_queue = [g_dev newCommandQueue];
        if (g_queue == nil) return 0;
        const char* nm = [[g_dev name] UTF8String];
        if (nm) {
            strncpy(g_win_dev_name, nm, sizeof(g_win_dev_name) - 1);
            g_win_dev_name[sizeof(g_win_dev_name) - 1] = 0;
        }
        g_win_w = (int)width;
        g_win_h = (int)height;
        g_should_close = 0;
        g_scale = 2.0f;

        if (!headless) {
            [NSApplication sharedApplication];
            [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
            [NSApp finishLaunching];
            NSRect rect = NSMakeRect(0, 0, (CGFloat)width, (CGFloat)height);
            NSWindowStyleMask style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable;
            g_window = [[NSWindow alloc] initWithContentRect:rect styleMask:style backing:NSBackingStoreBuffered defer:NO];
            NSString* t = title_cstr ? [NSString stringWithUTF8String:(const char*)(uintptr_t)title_cstr] : @"Tenon GPUI";
            [g_window setTitle:t];
            [g_window setBackgroundColor:[NSColor colorWithSRGBRed:11.0/255.0 green:17.0/255.0 blue:32.0/255.0 alpha:1.0]];
            [g_window center];

            g_delegate = [[TenonWinDelegate alloc] init];
            [g_window setDelegate:g_delegate];

            TenonMetalView* view = [[TenonMetalView alloc] initWithFrame:rect];
            [view setWantsLayer:YES];
            g_layer = [CAMetalLayer layer];
            g_layer.device = g_dev;
            g_layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
            g_layer.framebufferOnly = YES;
            g_layer.opaque = YES;
            g_layer.displaySyncEnabled = YES;
            g_layer.maximumDrawableCount = TN_MAX_INFLIGHT;
            CGFloat backing = [g_window backingScaleFactor];
            g_scale = backing > 0.5 ? (float)backing : 2.0f;
            g_layer.contentsScale = g_scale;
            g_layer.drawableSize = CGSizeMake(width * g_scale, height * g_scale);
            [view setLayer:g_layer];

            [g_window setContentView:view];
            [g_window makeFirstResponder:view];
            [g_window makeKeyAndOrderFront:nil];
            [NSApp activateIgnoringOtherApps:YES];
            if (!ensure_pipelines(MTLPixelFormatBGRA8Unorm)) return 0;
        } else {
            if (!ensure_pipelines(MTLPixelFormatRGBA8Unorm)) return 0;
        }
        init_text_surface(g_win_w, g_win_h, g_scale);
        return 1;
    }
}

int64_t tn_win_device_name(void) {
    return (int64_t)(uintptr_t)g_win_dev_name;
}

int64_t tn_win_begin_text(void) {
    if (g_text_pixels != NULL) {
        int pw = (int)(g_win_w * g_scale);
        int ph = (int)(g_win_h * g_scale);
        memset(g_text_pixels, 0, (size_t)(pw * ph * 4));
    }
    return 1;
}

int64_t tn_win_draw_text(
    int64_t text_cstr,
    int64_t x, int64_t y,
    int64_t font_px, int64_t bold,
    int64_t r255, int64_t g255, int64_t b255
) {
    if (g_text_ctx == NULL || text_cstr == 0) return 0;
    @autoreleasepool {
        NSString* str = [NSString stringWithUTF8String:(const char*)(uintptr_t)text_cstr];
        if ([str length] == 0) return 1;
        NSFont* font = bold
            ? [NSFont boldSystemFontOfSize:(CGFloat)font_px]
            : [NSFont systemFontOfSize:(CGFloat)font_px];
        NSColor* col = [NSColor colorWithSRGBRed:(CGFloat)r255/255.0
                                           green:(CGFloat)g255/255.0
                                            blue:(CGFloat)b255/255.0
                                           alpha:1.0];
        NSDictionary* attrs = @{
            NSFontAttributeName: font,
            NSForegroundColorAttributeName: col,
        };
        NSGraphicsContext* prev = [NSGraphicsContext currentContext];
        NSGraphicsContext* nsCtx = [NSGraphicsContext graphicsContextWithCGContext:g_text_ctx flipped:YES];
        [NSGraphicsContext setCurrentContext:nsCtx];
        [str drawAtPoint:NSMakePoint((CGFloat)x, (CGFloat)y) withAttributes:attrs];
        [NSGraphicsContext setCurrentContext:prev];
        return 1;
    }
}

static void encode_frame_into_tex(id<MTLTexture> target_tex, id<CAMetalDrawable> drawable_opt, int tex_slot, const void* quads_ptr, int64_t quad_count, double bg_r, double bg_g, double bg_b) {
    int pw = (int)(g_win_w * g_scale);
    int ph = (int)(g_win_h * g_scale);
    id<MTLTexture> text_tex = g_text_tex[tex_slot];
    if (text_tex != nil && g_text_pixels != NULL) {
        [text_tex replaceRegion:MTLRegionMake2D(0, 0, (NSUInteger)pw, (NSUInteger)ph)
                    mipmapLevel:0
                      withBytes:g_text_pixels
                    bytesPerRow:(NSUInteger)(pw * 4)];
    }

    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = target_tex;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(bg_r, bg_g, bg_b, 1.0);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rpd];
    if (quad_count > 0 && quads_ptr != NULL) {
        id<MTLBuffer> qbuf = [g_dev newBufferWithBytes:quads_ptr
                                                length:(NSUInteger)(quad_count * 48)
                                               options:MTLResourceStorageModeShared];
        float vp[4] = { (float)g_win_w, (float)g_win_h, 0.0f, 0.0f };
        [enc setRenderPipelineState:g_quad_pso];
        [enc setVertexBuffer:qbuf offset:0 atIndex:0];
        [enc setVertexBytes:vp length:sizeof(vp) atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6 instanceCount:(NSUInteger)quad_count];
    }
    if (text_tex != nil) {
        [enc setRenderPipelineState:g_blit_pso];
        [enc setFragmentTexture:text_tex atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
    }
    [enc endEncoding];
    if (drawable_opt != nil) {
        dispatch_semaphore_t sem = g_frame_sem;
        [cmd addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            (void)cb;
            if (sem != nil) dispatch_semaphore_signal(sem);
        }];
        [cmd presentDrawable:drawable_opt];
        [cmd commit];
    } else {
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

int64_t tn_win_present(int64_t quads_ptr, int64_t quad_count, int64_t bg_r255, int64_t bg_g255, int64_t bg_b255) {
    if (g_layer == nil) return 0;
    @autoreleasepool {
        if (g_frame_sem != nil) {
            dispatch_semaphore_wait(g_frame_sem, DISPATCH_TIME_FOREVER);
        }
        id<CAMetalDrawable> drawable = [g_layer nextDrawable];
        if (drawable == nil) {
            if (g_frame_sem != nil) dispatch_semaphore_signal(g_frame_sem);
            return 0;
        }
        int slot = g_frame_slot;
        g_frame_slot = (g_frame_slot + 1) % TN_MAX_INFLIGHT;
        encode_frame_into_tex(drawable.texture, drawable, slot, (const void*)(uintptr_t)quads_ptr, quad_count,
                              (double)bg_r255 / 255.0, (double)bg_g255 / 255.0, (double)bg_b255 / 255.0);
        return 1;
    }
}

int64_t tn_win_save_png(int64_t quads_ptr, int64_t quad_count, int64_t bg_r255, int64_t bg_g255, int64_t bg_b255, int64_t path_cstr) {
    if (g_dev == nil || path_cstr == 0) return 0;
    @autoreleasepool {
        int pw = (int)(g_win_w * g_scale);
        int ph = (int)(g_win_h * g_scale);
        MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                      width:(NSUInteger)pw
                                                                                     height:(NSUInteger)ph
                                                                                  mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> tex = [g_dev newTextureWithDescriptor:td];
        if (tex == nil) return 0;
        encode_frame_into_tex(tex, nil, 0, (const void*)(uintptr_t)quads_ptr, quad_count,
                              (double)bg_r255 / 255.0, (double)bg_g255 / 255.0, (double)bg_b255 / 255.0);
        uint8_t* px = (uint8_t*)malloc((size_t)(pw * ph * 4));
        if (!px) return 0;
        [tex getBytes:px bytesPerRow:(NSUInteger)(pw * 4) fromRegion:MTLRegionMake2D(0, 0, (NSUInteger)pw, (NSUInteger)ph) mipmapLevel:0];
        CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
        CGDataProviderRef prov = CGDataProviderCreateWithData(NULL, px, (size_t)(pw * ph * 4), NULL);
        CGImageRef img = CGImageCreate((size_t)pw, (size_t)ph, 8, 32, (size_t)(pw * 4), cs,
                                       kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big, prov, NULL, false, kCGRenderingIntentDefault);
        NSString* path = [NSString stringWithUTF8String:(const char*)(uintptr_t)path_cstr];
        CFURLRef url = (__bridge CFURLRef)[NSURL fileURLWithPath:path];
        CGImageDestinationRef dst = CGImageDestinationCreateWithURL(url, CFSTR("public.png"), 1, NULL);
        int64_t ok = 0;
        if (dst) {
            CGImageDestinationAddImage(dst, img, NULL);
            ok = CGImageDestinationFinalize(dst) ? 1 : 0;
            CFRelease(dst);
        }
        CGImageRelease(img);
        CGDataProviderRelease(prov);
        CGColorSpaceRelease(cs);
        free(px);
        return ok;
    }
}

static int64_t g_last_ev_a = 0;
static int64_t g_last_ev_b = 0;

int64_t tn_win_poll_event(int64_t timeout_ms) {
    @autoreleasepool {
        if (g_window != nil) {
            NSDate* until = (timeout_ms > 0)
                ? [NSDate dateWithTimeIntervalSinceNow:(double)timeout_ms / 1000.0]
                : [NSDate distantPast];
            NSEvent* ev = nil;
            while ((ev = [NSApp nextEventMatchingMask:NSEventMaskAny untilDate:until inMode:NSDefaultRunLoopMode dequeue:YES]) != nil) {
                [NSApp sendEvent:ev];
                until = [NSDate distantPast];
            }
            [NSApp updateWindows];
        }
        if (g_should_close) return 3;
        if (g_ev_head == g_ev_tail) return 0;
        TnWinEvent e = g_events[g_ev_head];
        g_ev_head = (g_ev_head + 1) % MAX_EVENTS;
        g_last_ev_a = e.a;
        g_last_ev_b = e.b;
        return e.kind;
    }
}

int64_t tn_win_event_a(void) { return g_last_ev_a; }
int64_t tn_win_event_b(void) { return g_last_ev_b; }
