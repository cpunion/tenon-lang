// std/gpu/window/metal_window.m — Zero-Rust Native macOS Cocoa + Metal GPU Window with Dynamic R8 Glyph Atlas (`tenon-gpui`)
//
// Zed/GPUI-grade zero-Rust rendering architecture:
//   1. Hardware-accelerated `NSWindow` (resizable!) + VSync triple-buffered `CAMetalLayer`
//   2. Pass 1: ALL SDF rounded UI quads rendered in ONE instanced Metal draw call (`wquad_vs`/`wquad_fs`)
//   3. Pass 2: Dynamic 1024x1024 R8Unorm GPU Glyph Atlas (shelf-packed on-demand CoreText rasterizer,
//      full UTF-8 ASCII + CJK Chinese support, cached per `(codepoint, font_px, bold)`) rendered in
//      ONE instanced Metal draw call (`wglyph_vs`/`wglyph_fs`) — ZERO per-frame full-screen bitmap upload!
//   4. Live mouse clicks, hover, wheel scroll, window resize, Unicode keyboard input, clipboard paste,
//      optional GPU scissor clipping, and Retina PNG snapshot export!

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreText/CoreText.h>
#import <ImageIO/ImageIO.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

// Pending input event queue
typedef struct {
    int64_t kind; // 0=none, 1=key, 2=mouse_down, 3=quit, 4=resize, 5=scroll, 6=hover
    int64_t a;    // keyCode / mouse_x / new_w / scroll_dx
    int64_t b;    // unicode codepoint / mouse_y / new_h / scroll_dy
} TnWinEvent;

#define MAX_EVENTS 128
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

// 48-byte GPU Glyph Instance matching MSL `GlyphInstance`
typedef struct {
    float x, y, w, h;
    float u0, v0, u1, v1;
    float r, g, b, a;
} TnGlyphInstance;

#define ATLAS_SIZE 1024
#define MAX_GLYPH_CACHE 2048
#define MAX_FRAME_GLYPHS 16384
#define TN_MAX_INFLIGHT 3

typedef struct {
    uint64_t key; // (codepoint << 16) | (font_px << 1) | bold
    float u0, v0, u1, v1;
    float w_pt, h_pt, adv_pt;
    uint8_t used;
} AtlasEntry;

static AtlasEntry g_atlas_cache[MAX_GLYPH_CACHE];
static int g_atlas_x = 1;
static int g_atlas_y = 1;
static int g_atlas_row_h = 0;
static id<MTLTexture> g_atlas_tex = nil;

static TnGlyphInstance g_glyphs[MAX_FRAME_GLYPHS];
static int g_glyph_count = 0;

static NSWindow* g_window = nil;
static CAMetalLayer* g_layer = nil;
static id<MTLDevice> g_dev = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLRenderPipelineState> g_quad_pso = nil;
static id<MTLRenderPipelineState> g_glyph_pso = nil;
static dispatch_semaphore_t g_frame_sem = nil;
static int g_win_w = 880;
static int g_win_h = 540;
static float g_scale = 2.0f;
static int g_should_close = 0;
static int g_scissor_on = 0;
static int g_sc_x = 0, g_sc_y = 0, g_sc_w = 0, g_sc_h = 0;
static char g_win_dev_name[256] = "Apple Metal GPU";
static char g_clip_buf[4096] = "";

@interface TenonMetalView : NSView
@end

@implementation TenonMetalView
- (BOOL)acceptsFirstResponder { return YES; }
- (BOOL)isFlipped { return YES; }

- (void)mouseDown:(NSEvent *)event {
    NSPoint pt = [self convertPoint:[event locationInWindow] fromView:nil];
    push_event(2, (int64_t)pt.x, (int64_t)pt.y);
}

- (void)mouseMoved:(NSEvent *)event {
    NSPoint pt = [self convertPoint:[event locationInWindow] fromView:nil];
    push_event(6, (int64_t)pt.x, (int64_t)pt.y);
}

- (void)scrollWheel:(NSEvent *)event {
    push_event(5, (int64_t)([event scrollingDeltaX]), (int64_t)([event scrollingDeltaY]));
}

- (void)keyDown:(NSEvent *)event {
    NSString *chars = [event characters];
    int64_t ch = 0;
    if ([chars length] > 0) {
        unichar c0 = [chars characterAtIndex:0];
        if (c0 >= 0xD800 && c0 <= 0xDBFF && [chars length] > 1) {
            unichar c1 = [chars characterAtIndex:1];
            ch = 0x10000 + (((int64_t)(c0 - 0xD800) << 10) | (int64_t)(c1 - 0xDC00));
        } else {
            ch = (int64_t)c0;
        }
    }
    int64_t kc = (int64_t)[event keyCode];
    // Encode Cmd modifier in bit 16 of keyCode (e.g. Cmd+V paste)
    if (([event modifierFlags] & NSEventModifierFlagCommand) != 0) {
        kc |= (1LL << 16);
    }
    push_event(1, kc, ch);
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

- (void)windowDidResize:(NSNotification *)notification {
    (void)notification;
    if (g_window == nil || g_layer == nil) return;
    NSRect content = [[g_window contentView] bounds];
    int nw = (int)content.size.width;
    int nh = (int)content.size.height;
    if (nw >= 240 && nh >= 160) {
        g_win_w = nw;
        g_win_h = nh;
        CGFloat backing = [g_window backingScaleFactor];
        g_scale = backing > 0.5 ? (float)backing : 2.0f;
        g_layer.contentsScale = g_scale;
        g_layer.drawableSize = CGSizeMake(nw * g_scale, nh * g_scale);
        push_event(4, (int64_t)nw, (int64_t)nh);
    }
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
"struct GlyphInstance {\n"
"    float x; float y; float w; float h;\n"
"    float u0; float v0; float u1; float v1;\n"
"    float r; float g; float b; float a;\n"
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
"struct GlyphOut {\n"
"    float4 pos [[position]];\n"
"    float2 uv;\n"
"    float4 color;\n"
"};\n"
"vertex GlyphOut wglyph_vs(uint vid [[vertex_id]], uint iid [[instance_id]],\n"
"                          device const GlyphInstance* glyphs [[buffer(0)]],\n"
"                          constant Viewport& vp [[buffer(1)]]) {\n"
"    const float2 corners[6] = {\n"
"        float2(0.0, 0.0), float2(1.0, 0.0), float2(0.0, 1.0),\n"
"        float2(1.0, 0.0), float2(1.0, 1.0), float2(0.0, 1.0)\n"
"    };\n"
"    GlyphInstance g = glyphs[iid];\n"
"    float2 c = corners[vid];\n"
"    float2 px = float2(g.x + c.x * g.w, g.y + c.y * g.h);\n"
"    GlyphOut o;\n"
"    o.pos = float4((px.x / vp.width) * 2.0f - 1.0f, 1.0f - (px.y / vp.height) * 2.0f, 0.0f, 1.0f);\n"
"    o.uv = float2(mix(g.u0, g.u1, c.x), mix(g.v0, g.v1, c.y));\n"
"    o.color = float4(g.r, g.g, g.b, g.a);\n"
"    return o;\n"
"}\n"
"fragment float4 wglyph_fs(GlyphOut in [[stage_in]], texture2d<float> atlas [[texture(0)]]) {\n"
"    constexpr sampler s(mag_filter::linear, min_filter::linear);\n"
"    float cov = atlas.sample(s, in.uv).r;\n"
"    if (cov < 0.01f) discard_fragment();\n"
"    return float4(in.color.rgb, in.color.a * cov);\n"
"}\n";

static int ensure_pipelines(MTLPixelFormat fmt) {
    if (g_quad_pso != nil && g_glyph_pso != nil) return 1;
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

    MTLRenderPipelineDescriptor* gd = [[MTLRenderPipelineDescriptor alloc] init];
    gd.vertexFunction = [lib newFunctionWithName:@"wglyph_vs"];
    gd.fragmentFunction = [lib newFunctionWithName:@"wglyph_fs"];
    gd.colorAttachments[0].pixelFormat = fmt;
    gd.colorAttachments[0].blendingEnabled = YES;
    gd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    gd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    gd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    gd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    g_glyph_pso = [g_dev newRenderPipelineStateWithDescriptor:gd error:&err];
    return g_glyph_pso != nil ? 1 : 0;
}

static void init_glyph_atlas(void) {
    memset(g_atlas_cache, 0, sizeof(g_atlas_cache));
    g_atlas_x = 1;
    g_atlas_y = 1;
    g_atlas_row_h = 0;
    MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR8Unorm
                                                                                  width:ATLAS_SIZE
                                                                                 height:ATLAS_SIZE
                                                                              mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    g_atlas_tex = [g_dev newTextureWithDescriptor:td];
    uint8_t* zeros = (uint8_t*)calloc(ATLAS_SIZE * ATLAS_SIZE, 1);
    if (zeros) {
        [g_atlas_tex replaceRegion:MTLRegionMake2D(0, 0, ATLAS_SIZE, ATLAS_SIZE)
                       mipmapLevel:0
                         withBytes:zeros
                       bytesPerRow:ATLAS_SIZE];
        free(zeros);
    }
    if (g_frame_sem == nil) {
        g_frame_sem = dispatch_semaphore_create(TN_MAX_INFLIGHT);
    }
}

static AtlasEntry* get_or_rasterize_glyph(uint32_t cp, int font_px, int bold) {
    uint64_t key = ((uint64_t)cp << 16) | ((uint64_t)(font_px & 0x7FFF) << 1) | (uint64_t)(bold ? 1 : 0);
    uint32_t h = (uint32_t)((key ^ (key >> 33)) * 0xff51afd7ed558ccdULL) & (MAX_GLYPH_CACHE - 1);
    for (int probe = 0; probe < MAX_GLYPH_CACHE; probe++) {
        uint32_t idx = (h + (uint32_t)probe) & (MAX_GLYPH_CACHE - 1);
        if (g_atlas_cache[idx].used && g_atlas_cache[idx].key == key) {
            return &g_atlas_cache[idx];
        }
        if (!g_atlas_cache[idx].used) {
            @autoreleasepool {
                unichar u16[2];
                NSUInteger u16_len = 0;
                if (cp <= 0xFFFF) {
                    u16[0] = (unichar)cp;
                    u16_len = 1;
                } else {
                    uint32_t v = cp - 0x10000;
                    u16[0] = (unichar)(0xD800 + (v >> 10));
                    u16[1] = (unichar)(0xDC00 + (v & 0x3FF));
                    u16_len = 2;
                }
                NSString* s = [NSString stringWithCharacters:u16 length:u16_len];
                NSFont* font = bold
                    ? [NSFont boldSystemFontOfSize:(CGFloat)font_px]
                    : [NSFont systemFontOfSize:(CGFloat)font_px];
                CTFontRef ctFont = (__bridge CTFontRef)font;
                CGFloat ascent = CTFontGetAscent(ctFont);
                CGFloat descent = CTFontGetDescent(ctFont);
                CGFloat leading = CTFontGetLeading(ctFont);
                float line_h_pt = (float)ceil(ascent + descent + leading);

                NSDictionary* attrs = @{
                    (id)kCTFontAttributeName: font,
                    (id)kCTForegroundColorFromContextAttributeName: @YES,
                };
                NSAttributedString* attrStr = [[NSAttributedString alloc] initWithString:s attributes:attrs];
                CTLineRef line = CTLineCreateWithAttributedString((__bridge CFAttributedStringRef)attrStr);
                double typo_w = CTLineGetTypographicBounds(line, NULL, NULL, NULL);
                float adv_pt = (float)typo_w;
                if (adv_pt < 1.0f) adv_pt = (float)font_px * 0.35f;
                if (line_h_pt < 2.0f) line_h_pt = (float)font_px * 1.25f;

                int gw = (int)ceilf(adv_pt * g_scale) + 6;
                int gh = (int)ceilf(line_h_pt * g_scale) + 4;
                if (gw > 128) gw = 128;
                if (gh > 128) gh = 128;

                // Shelf-pack into 1024x1024 R8Unorm GPU Glyph Atlas
                if (g_atlas_x + gw + 1 >= ATLAS_SIZE) {
                    g_atlas_x = 1;
                    g_atlas_y += g_atlas_row_h + 2;
                    g_atlas_row_h = 0;
                }
                if (g_atlas_y + gh + 1 >= ATLAS_SIZE) {
                    memset(g_atlas_cache, 0, sizeof(g_atlas_cache));
                    g_atlas_x = 1;
                    g_atlas_y = 1;
                    g_atlas_row_h = 0;
                    idx = h;
                }

                uint8_t* r8 = (uint8_t*)calloc((size_t)(gw * gh), 1);
                if (r8) {
                    CGColorSpaceRef gray = CGColorSpaceCreateDeviceGray();
                    CGContextRef ctx = CGBitmapContextCreate(
                        r8, (size_t)gw, (size_t)gh, 8, (size_t)gw, gray, (CGBitmapInfo)kCGImageAlphaNone
                    );
                    CGColorSpaceRelease(gray);
                    if (ctx) {
                        CGContextScaleCTM(ctx, g_scale, g_scale);
                        CGContextSetAllowsAntialiasing(ctx, true);
                        CGContextSetShouldAntialias(ctx, true);
                        CGContextSetAllowsFontSmoothing(ctx, false);
                        CGContextSetGrayFillColor(ctx, 1.0, 1.0);
                        // Exact shared baseline for every glyph in this font size (ASCII + CJK!)
                        CGContextSetTextPosition(ctx, 1.5, descent + 1.0);
                        CTLineDraw(line, ctx);
                        CGContextRelease(ctx);
                    }
                    [g_atlas_tex replaceRegion:MTLRegionMake2D((NSUInteger)g_atlas_x, (NSUInteger)g_atlas_y, (NSUInteger)gw, (NSUInteger)gh)
                                   mipmapLevel:0
                                     withBytes:r8
                                   bytesPerRow:(NSUInteger)gw];
                    free(r8);
                }
                CFRelease(line);

                AtlasEntry* e = &g_atlas_cache[idx];
                e->key = key;
                e->u0 = (float)g_atlas_x / (float)ATLAS_SIZE;
                e->v0 = (float)g_atlas_y / (float)ATLAS_SIZE;
                e->u1 = (float)(g_atlas_x + gw) / (float)ATLAS_SIZE;
                e->v1 = (float)(g_atlas_y + gh) / (float)ATLAS_SIZE;
                e->w_pt = (float)gw / g_scale;
                e->h_pt = (float)gh / g_scale;
                e->adv_pt = adv_pt;
                e->used = 1;

                g_atlas_x += gw + 1;
                if (gh > g_atlas_row_h) g_atlas_row_h = gh;
                return e;
            }
        }
    }
    return NULL;
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
        g_scissor_on = 0;

        if (!headless) {
            [NSApplication sharedApplication];
            [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
            [NSApp finishLaunching];
            NSRect rect = NSMakeRect(0, 0, (CGFloat)width, (CGFloat)height);
            NSWindowStyleMask style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                                    | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;
            g_window = [[NSWindow alloc] initWithContentRect:rect styleMask:style backing:NSBackingStoreBuffered defer:NO];
            NSString* t = title_cstr ? [NSString stringWithUTF8String:(const char*)(uintptr_t)title_cstr] : @"Tenon GPUI";
            [g_window setTitle:t];
            [g_window setBackgroundColor:[NSColor colorWithSRGBRed:11.0/255.0 green:17.0/255.0 blue:32.0/255.0 alpha:1.0]];
            [g_window setAcceptsMouseMovedEvents:YES];
            [g_window setMinSize:NSMakeSize(520, 360)];
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
        init_glyph_atlas();
        return 1;
    }
}

int64_t tn_win_device_name(void) {
    return (int64_t)(uintptr_t)g_win_dev_name;
}

int64_t tn_win_width(void) { return (int64_t)g_win_w; }
int64_t tn_win_height(void) { return (int64_t)g_win_h; }

int64_t tn_win_set_scissor(int64_t x, int64_t y, int64_t w, int64_t h) {
    g_scissor_on = (w > 0 && h > 0) ? 1 : 0;
    g_sc_x = (int)x; g_sc_y = (int)y; g_sc_w = (int)w; g_sc_h = (int)h;
    return 1;
}

int64_t tn_win_clear_scissor(void) {
    g_scissor_on = 0;
    return 1;
}

int64_t tn_win_clipboard_get(void) {
    @autoreleasepool {
        NSPasteboard* pb = [NSPasteboard generalPasteboard];
        NSString* s = [pb stringForType:NSPasteboardTypeString];
        if (s != nil) {
            const char* u8 = [s UTF8String];
            if (u8) {
                strncpy(g_clip_buf, u8, sizeof(g_clip_buf) - 1);
                g_clip_buf[sizeof(g_clip_buf) - 1] = 0;
                return (int64_t)(uintptr_t)g_clip_buf;
            }
        }
        g_clip_buf[0] = 0;
        return (int64_t)(uintptr_t)g_clip_buf;
    }
}

int64_t tn_win_begin_text(void) {
    g_glyph_count = 0;
    return 1;
}

int64_t tn_win_draw_text(
    int64_t text_cstr,
    int64_t x, int64_t y,
    int64_t font_px, int64_t bold,
    int64_t r255, int64_t g255, int64_t b255
) {
    if (g_atlas_tex == nil || text_cstr == 0) return 0;
    const uint8_t* p = (const uint8_t*)(uintptr_t)text_cstr;
    float pen_x = (float)x;
    float pen_y = (float)y;
    float cr = (float)r255 / 255.0f;
    float cg = (float)g255 / 255.0f;
    float cb = (float)b255 / 255.0f;

    while (*p && g_glyph_count < MAX_FRAME_GLYPHS) {
        uint32_t cp = 0;
        if ((*p & 0x80) == 0) {
            cp = *p++;
        } else if ((*p & 0xE0) == 0xC0 && p[1]) {
            cp = ((uint32_t)(p[0] & 0x1F) << 6) | (uint32_t)(p[1] & 0x3F);
            p += 2;
        } else if ((*p & 0xF0) == 0xE0 && p[1] && p[2]) {
            cp = ((uint32_t)(p[0] & 0x0F) << 12) | ((uint32_t)(p[1] & 0x3F) << 6) | (uint32_t)(p[2] & 0x3F);
            p += 3;
        } else if ((*p & 0xF8) == 0xF0 && p[1] && p[2] && p[3]) {
            cp = ((uint32_t)(p[0] & 0x07) << 18) | ((uint32_t)(p[1] & 0x3F) << 12)
               | ((uint32_t)(p[2] & 0x3F) << 6) | (uint32_t)(p[3] & 0x3F);
            p += 4;
        } else {
            p++;
            continue;
        }
        if (cp == 10) continue;
        AtlasEntry* e = get_or_rasterize_glyph(cp, (int)font_px, (int)bold);
        if (!e) continue;
        if (cp != 32) {
            TnGlyphInstance* gi = &g_glyphs[g_glyph_count++];
            gi->x = pen_x;
            gi->y = pen_y;
            gi->w = e->w_pt;
            gi->h = e->h_pt;
            gi->u0 = e->u0;
            gi->v0 = e->v0;
            gi->u1 = e->u1;
            gi->v1 = e->v1;
            gi->r = cr;
            gi->g = cg;
            gi->b = cb;
            gi->a = 1.0f;
        }
        pen_x += e->adv_pt;
    }
    return (int64_t)g_glyph_count;
}

static void encode_frame_into_tex(id<MTLTexture> target_tex, id<CAMetalDrawable> drawable_opt, const void* quads_ptr, int64_t quad_count, double bg_r, double bg_g, double bg_b) {
    MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = target_tex;
    rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.colorAttachments[0].clearColor = MTLClearColorMake(bg_r, bg_g, bg_b, 1.0);

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rpd];
    float vp[4] = { (float)g_win_w, (float)g_win_h, 0.0f, 0.0f };

    if (g_scissor_on) {
        int pw = (int)(g_win_w * g_scale);
        int ph = (int)(g_win_h * g_scale);
        int sx = (int)(g_sc_x * g_scale); if (sx < 0) sx = 0;
        int sy = (int)(g_sc_y * g_scale); if (sy < 0) sy = 0;
        int sw = (int)(g_sc_w * g_scale); if (sx + sw > pw) sw = pw - sx;
        int sh = (int)(g_sc_h * g_scale); if (sy + sh > ph) sh = ph - sy;
        if (sw > 0 && sh > 0) {
            MTLScissorRect sr = { (NSUInteger)sx, (NSUInteger)sy, (NSUInteger)sw, (NSUInteger)sh };
            [enc setScissorRect:sr];
        }
    }

    // Pass 1: ALL rounded SDF quads in ONE instanced Metal draw call!
    if (quad_count > 0 && quads_ptr != NULL) {
        id<MTLBuffer> qbuf = [g_dev newBufferWithBytes:quads_ptr
                                                length:(NSUInteger)(quad_count * 48)
                                               options:MTLResourceStorageModeShared];
        [enc setRenderPipelineState:g_quad_pso];
        [enc setVertexBuffer:qbuf offset:0 atIndex:0];
        [enc setVertexBytes:vp length:sizeof(vp) atIndex:1];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6 instanceCount:(NSUInteger)quad_count];
    }

    // Pass 2: ALL GPU Glyph Atlas text instances in ONE instanced Metal draw call!
    if (g_glyph_count > 0 && g_atlas_tex != nil) {
        id<MTLBuffer> gbuf = [g_dev newBufferWithBytes:g_glyphs
                                                length:(NSUInteger)(g_glyph_count * (int)sizeof(TnGlyphInstance))
                                               options:MTLResourceStorageModeShared];
        [enc setRenderPipelineState:g_glyph_pso];
        [enc setVertexBuffer:gbuf offset:0 atIndex:0];
        [enc setVertexBytes:vp length:sizeof(vp) atIndex:1];
        [enc setFragmentTexture:g_atlas_tex atIndex:0];
        [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6 instanceCount:(NSUInteger)g_glyph_count];
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
        encode_frame_into_tex(drawable.texture, drawable, (const void*)(uintptr_t)quads_ptr, quad_count,
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
        encode_frame_into_tex(tex, nil, (const void*)(uintptr_t)quads_ptr, quad_count,
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
