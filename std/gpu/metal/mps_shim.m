// std/gpu/metal/mps_shim.m — Zero-Rust Apple Metal Performance Shaders (MPS) Hardware Bridge
//
// Directly accelerates Tenon Transformer inference on Apple Silicon (M1/M2/M3/M4) using:
//   - `MPSMatrixMultiplication` (FP16 `MPSDataTypeFloat16` & FP32 `MPSDataTypeFloat32`,
//     strided leading dimensions, byte offsets, alpha/beta, kernel LRU cache)
//   - `MPSMatrixSoftMax` (FP16 & FP32 hardware row-wise softmax)
//   - Single-Command-Buffer batching (`tn_mps_batch_begin` / `tn_mps_batch_commit`) so an
//     ENTIRE multi-layer Transformer forward pass (MPS GEMM + MPS SoftMax + custom Tenon
//     `@kernel` RMSNorm/RoPE/SwiGLU/KV-Cache + Blit DMA) submits in ONE GPU command buffer
//     with ZERO intermediate CPU waits!

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

extern void* tn_metal_raw_device(void);
extern void* tn_metal_raw_queue(void);
extern void* tn_metal_raw_buffer(int64_t handle);
extern void* tn_metal_raw_pipeline(int64_t handle);

static id<MTLCommandBuffer> g_mps_batch_cmd = nil;

// Small fixed cache for MPSMatrixMultiplication & MPSMatrixVectorMultiplication objects
#define MPS_MM_CACHE_SIZE 128
typedef struct {
    int64_t m, n, k, batch;
    int64_t tr_a, tr_b;
    int64_t is_f16;
    uint64_t alpha_bits, beta_bits;
    __unsafe_unretained MPSMatrixMultiplication* obj;
} MpsMmCacheEntry;

typedef struct {
    int64_t rows, cols;
    int64_t is_f16;
    uint64_t alpha_bits, beta_bits;
    __unsafe_unretained MPSMatrixVectorMultiplication* obj;
} MpsMvCacheEntry;

static MpsMmCacheEntry g_mm_cache[MPS_MM_CACHE_SIZE];
static MpsMvCacheEntry g_mv_cache[MPS_MM_CACHE_SIZE];
static NSMutableArray* g_mm_retain_pool = nil;
static NSMutableArray* g_mv_retain_pool = nil;
static int g_mm_cache_next = 0;
static int g_mv_cache_next = 0;
static MPSMatrixSoftMax* g_mps_softmax = nil;

static id<MTLDevice> mps_dev(void) {
    void* p = tn_metal_raw_device();
    return p ? (__bridge id<MTLDevice>)p : nil;
}

static id<MTLCommandQueue> mps_queue(void) {
    void* p = tn_metal_raw_queue();
    return p ? (__bridge id<MTLCommandQueue>)p : nil;
}

static id<MTLBuffer> mps_buf(int64_t h) {
    void* p = tn_metal_raw_buffer(h);
    return p ? (__bridge id<MTLBuffer>)p : nil;
}

static id<MTLComputePipelineState> mps_pipe(int64_t h) {
    void* p = tn_metal_raw_pipeline(h);
    return p ? (__bridge id<MTLComputePipelineState>)p : nil;
}

static MPSMatrixMultiplication* get_cached_mm(
    id<MTLDevice> dev,
    int64_t m, int64_t n, int64_t k, int64_t batch,
    int64_t tr_a, int64_t tr_b,
    int64_t is_f16,
    double alpha, double beta
) {
    uint64_t ab = 0, bb = 0;
    memcpy(&ab, &alpha, 8);
    memcpy(&bb, &beta, 8);
    for (int i = 0; i < MPS_MM_CACHE_SIZE; i++) {
        MpsMmCacheEntry* e = &g_mm_cache[i];
        if (e->obj != nil && e->m == m && e->n == n && e->k == k && e->batch == batch &&
            e->tr_a == tr_a && e->tr_b == tr_b && e->is_f16 == is_f16 &&
            e->alpha_bits == ab && e->beta_bits == bb) {
            return e->obj;
        }
    }
    if (g_mm_retain_pool == nil) {
        g_mm_retain_pool = [[NSMutableArray alloc] initWithCapacity:MPS_MM_CACHE_SIZE];
    }
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:dev
        transposeLeft:(tr_a != 0)
        transposeRight:(tr_b != 0)
        resultRows:(NSUInteger)m
        resultColumns:(NSUInteger)n
        interiorColumns:(NSUInteger)k
        alpha:alpha
        beta:beta];
    if (batch > 1) {
        mm.batchSize = (NSUInteger)batch;
        mm.batchStart = 0;
    }
    int slot = g_mm_cache_next % MPS_MM_CACHE_SIZE;
    g_mm_cache_next++;
    if ((int)[g_mm_retain_pool count] <= slot) {
        [g_mm_retain_pool addObject:mm];
    } else {
        g_mm_retain_pool[(NSUInteger)slot] = mm;
    }
    g_mm_cache[slot].m = m;
    g_mm_cache[slot].n = n;
    g_mm_cache[slot].k = k;
    g_mm_cache[slot].batch = batch;
    g_mm_cache[slot].tr_a = tr_a;
    g_mm_cache[slot].tr_b = tr_b;
    g_mm_cache[slot].is_f16 = is_f16;
    g_mm_cache[slot].alpha_bits = ab;
    g_mm_cache[slot].beta_bits = bb;
    g_mm_cache[slot].obj = mm;
    return mm;
}

static MPSMatrixVectorMultiplication* get_cached_mv(
    id<MTLDevice> dev,
    int64_t rows, int64_t cols,
    int64_t is_f16,
    double alpha, double beta
) {
    uint64_t ab = 0, bb = 0;
    memcpy(&ab, &alpha, 8);
    memcpy(&bb, &beta, 8);
    for (int i = 0; i < MPS_MM_CACHE_SIZE; i++) {
        MpsMvCacheEntry* e = &g_mv_cache[i];
        if (e->obj != nil && e->rows == rows && e->cols == cols &&
            e->is_f16 == is_f16 && e->alpha_bits == ab && e->beta_bits == bb) {
            return e->obj;
        }
    }
    if (g_mv_retain_pool == nil) {
        g_mv_retain_pool = [[NSMutableArray alloc] initWithCapacity:MPS_MM_CACHE_SIZE];
    }
    MPSMatrixVectorMultiplication* mv = [[MPSMatrixVectorMultiplication alloc]
        initWithDevice:dev
        transpose:NO
        rows:(NSUInteger)rows
        columns:(NSUInteger)cols
        alpha:alpha
        beta:beta];
    int slot = g_mv_cache_next % MPS_MM_CACHE_SIZE;
    g_mv_cache_next++;
    if ((int)[g_mv_retain_pool count] <= slot) {
        [g_mv_retain_pool addObject:mv];
    } else {
        g_mv_retain_pool[(NSUInteger)slot] = mv;
    }
    g_mv_cache[slot].rows = rows;
    g_mv_cache[slot].cols = cols;
    g_mv_cache[slot].is_f16 = is_f16;
    g_mv_cache[slot].alpha_bits = ab;
    g_mv_cache[slot].beta_bits = bb;
    g_mv_cache[slot].obj = mv;
    return mv;
}

int64_t tn_mps_available(void) {
    id<MTLDevice> dev = mps_dev();
    if (dev == nil) return 0;
    return MPSSupportsMTLDevice(dev) ? 1 : 0;
}

int64_t tn_mps_batch_begin(void) {
    id<MTLCommandQueue> q = mps_queue();
    if (q == nil) return 0;
    if (g_mps_batch_cmd != nil) return 1;
    g_mps_batch_cmd = [q commandBuffer];
    return g_mps_batch_cmd != nil ? 1 : 0;
}

void* tn_mps_active_cmd_buffer(void) {
    return (__bridge void*)g_mps_batch_cmd;
}

int64_t tn_mps_batch_commit(void) {
    if (g_mps_batch_cmd == nil) return 0;
    @autoreleasepool {
        id<MTLCommandBuffer> cmd = g_mps_batch_cmd;
        g_mps_batch_cmd = nil;
        [cmd commit];
        [cmd waitUntilCompleted];
        return ([cmd status] == MTLCommandBufferStatusCompleted) ? 1 : 0;
    }
}

/// Hardware MPS Matrix Multiplication (`C = alpha * op(A) * op(B) + beta * C`)
/// `is_f16`: 1 => `MPSDataTypeFloat16` (2B/elem), 0 => `MPSDataTypeFloat32` (4B/elem).
/// `lda`, `ldb`, `ldc`: row strides in elements (allows zero-copy slices of KV-Cache & score buffers!).
int64_t tn_mps_gemm_ex(
    int64_t buf_a, int64_t off_a, int64_t lda,
    int64_t buf_b, int64_t off_b, int64_t ldb,
    int64_t buf_c, int64_t off_c, int64_t ldc,
    int64_t m, int64_t n, int64_t k,
    int64_t tr_a, int64_t tr_b,
    int64_t is_f16,
    double alpha, double beta
) {
    id<MTLDevice> dev = mps_dev();
    id<MTLCommandQueue> q = mps_queue();
    id<MTLBuffer> ma = mps_buf(buf_a);
    id<MTLBuffer> mb = mps_buf(buf_b);
    id<MTLBuffer> mc = mps_buf(buf_c);
    if (dev == nil || q == nil || ma == nil || mb == nil || mc == nil) return 0;
    if (m <= 0 || n <= 0 || k <= 0) return 0;

    @autoreleasepool {
        MPSDataType dt = is_f16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
        NSUInteger elem_sz = is_f16 ? 2 : 4;

        // Ultra-fast Apple Silicon GEMV path (`MPSMatrixVectorMultiplication`) for 1-token LLM decode!
        // When `m == 1`, `!tr_a`, `tr_b`: `C[1, n] = A[1, k] @ B[n, k]^T` == `B[n, k] @ A[k]`!
        if (m == 1 && !tr_a && tr_b) {
            NSUInteger b_rb = (NSUInteger)(ldb > 0 ? ldb : k) * elem_sz;
            MPSMatrixDescriptor* db = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)n
                                                                            columns:(NSUInteger)k
                                                                           rowBytes:b_rb
                                                                           dataType:dt];
            MPSVectorDescriptor* dvx = [MPSVectorDescriptor vectorDescriptorWithLength:(NSUInteger)k dataType:dt];
            MPSVectorDescriptor* dvy = [MPSVectorDescriptor vectorDescriptorWithLength:(NSUInteger)n dataType:dt];
            MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:mb offset:(NSUInteger)off_b descriptor:db];
            MPSVector* vecX = [[MPSVector alloc] initWithBuffer:ma offset:(NSUInteger)off_a descriptor:dvx];
            MPSVector* vecY = [[MPSVector alloc] initWithBuffer:mc offset:(NSUInteger)off_c descriptor:dvy];
            MPSMatrixVectorMultiplication* mv = get_cached_mv(dev, n, k, is_f16, alpha, beta);
            int local_cmd = (g_mps_batch_cmd == nil);
            id<MTLCommandBuffer> cmd = local_cmd ? [q commandBuffer] : g_mps_batch_cmd;
            [mv encodeToCommandBuffer:cmd inputMatrix:matB inputVector:vecX resultVector:vecY];
            if (local_cmd) {
                [cmd commit];
                [cmd waitUntilCompleted];
                return ([cmd status] == MTLCommandBufferStatusCompleted) ? 1 : 0;
            }
            return 1;
        }

        NSUInteger a_rows = (NSUInteger)(tr_a ? k : m);
        NSUInteger a_cols = (NSUInteger)(tr_a ? m : k);
        NSUInteger b_rows = (NSUInteger)(tr_b ? n : k);
        NSUInteger b_cols = (NSUInteger)(tr_b ? k : n);

        NSUInteger a_rb = (NSUInteger)(lda > 0 ? lda : (int64_t)a_cols) * elem_sz;
        NSUInteger b_rb = (NSUInteger)(ldb > 0 ? ldb : (int64_t)b_cols) * elem_sz;
        NSUInteger c_rb = (NSUInteger)(ldc > 0 ? ldc : n) * elem_sz;

        MPSMatrixDescriptor* da = [MPSMatrixDescriptor matrixDescriptorWithRows:a_rows
                                                                        columns:a_cols
                                                                       rowBytes:a_rb
                                                                       dataType:dt];
        MPSMatrixDescriptor* db = [MPSMatrixDescriptor matrixDescriptorWithRows:b_rows
                                                                        columns:b_cols
                                                                       rowBytes:b_rb
                                                                       dataType:dt];
        MPSMatrixDescriptor* dc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)m
                                                                        columns:(NSUInteger)n
                                                                       rowBytes:c_rb
                                                                       dataType:dt];

        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:ma offset:(NSUInteger)off_a descriptor:da];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:mb offset:(NSUInteger)off_b descriptor:db];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:mc offset:(NSUInteger)off_c descriptor:dc];

        MPSMatrixMultiplication* mm = get_cached_mm(dev, m, n, k, 1, tr_a, tr_b, is_f16, alpha, beta);

        int local_cmd = (g_mps_batch_cmd == nil);
        id<MTLCommandBuffer> cmd = local_cmd ? [q commandBuffer] : g_mps_batch_cmd;
        [mm encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        if (local_cmd) {
            [cmd commit];
            [cmd waitUntilCompleted];
            return ([cmd status] == MTLCommandBufferStatusCompleted) ? 1 : 0;
        }
        return 1;
    }
}

/// Hardware Apple MPS 3D Batched FP16 GEMM (`batch` heads executed in ONE single Metal dispatch!).
int64_t tn_mps_batched_gemm_f16(
    int64_t buf_a, int64_t off_a, int64_t lda, int64_t stride_a_bytes,
    int64_t buf_b, int64_t off_b, int64_t ldb, int64_t stride_b_bytes,
    int64_t buf_c, int64_t off_c, int64_t ldc, int64_t stride_c_bytes,
    int64_t batch, int64_t m, int64_t n, int64_t k,
    int64_t tr_b,
    double alpha, double beta
) {
    id<MTLDevice> dev = mps_dev();
    id<MTLCommandQueue> q = mps_queue();
    id<MTLBuffer> ma = mps_buf(buf_a);
    id<MTLBuffer> mb = mps_buf(buf_b);
    id<MTLBuffer> mc = mps_buf(buf_c);
    if (dev == nil || q == nil || ma == nil || mb == nil || mc == nil) return 0;
    if (batch <= 0 || m <= 0 || n <= 0 || k <= 0) return 0;

    @autoreleasepool {
        MPSDataType dt = MPSDataTypeFloat16;
        NSUInteger a_rb = (NSUInteger)lda * 2;
        NSUInteger b_rows = (NSUInteger)(tr_b ? n : k);
        NSUInteger b_cols = (NSUInteger)(tr_b ? k : n);
        NSUInteger b_rb = (NSUInteger)ldb * 2;
        NSUInteger c_rb = (NSUInteger)ldc * 2;

        MPSMatrixDescriptor* da = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)m
                                                                        columns:(NSUInteger)k
                                                                       matrices:(NSUInteger)batch
                                                                       rowBytes:a_rb
                                                                    matrixBytes:(NSUInteger)stride_a_bytes
                                                                       dataType:dt];
        MPSMatrixDescriptor* db = [MPSMatrixDescriptor matrixDescriptorWithRows:b_rows
                                                                        columns:b_cols
                                                                       matrices:(NSUInteger)batch
                                                                       rowBytes:b_rb
                                                                    matrixBytes:(NSUInteger)stride_b_bytes
                                                                       dataType:dt];
        MPSMatrixDescriptor* dc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)m
                                                                        columns:(NSUInteger)n
                                                                       matrices:(NSUInteger)batch
                                                                       rowBytes:c_rb
                                                                    matrixBytes:(NSUInteger)stride_c_bytes
                                                                       dataType:dt];
        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:ma offset:(NSUInteger)off_a descriptor:da];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:mb offset:(NSUInteger)off_b descriptor:db];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:mc offset:(NSUInteger)off_c descriptor:dc];

        MPSMatrixMultiplication* mm = get_cached_mm(dev, m, n, k, batch, 0, tr_b, 1, alpha, beta);
        int local_cmd = (g_mps_batch_cmd == nil);
        id<MTLCommandBuffer> cmd = local_cmd ? [q commandBuffer] : g_mps_batch_cmd;
        [mm encodeToCommandBuffer:cmd leftMatrix:matA rightMatrix:matB resultMatrix:matC];
        if (local_cmd) {
            [cmd commit];
            [cmd waitUntilCompleted];
            return ([cmd status] == MTLCommandBufferStatusCompleted) ? 1 : 0;
        }
        return 1;
    }
}

/// Hardware MPS Row-Wise SoftMax (`dst[r, :] = softmax(src[r, :])`) supporting FP16 & FP32 + byte offsets!
int64_t tn_mps_softmax_ex(
    int64_t buf_src, int64_t off_src, int64_t ld_src,
    int64_t buf_dst, int64_t off_dst, int64_t ld_dst,
    int64_t rows, int64_t cols,
    int64_t is_f16
) {
    id<MTLDevice> dev = mps_dev();
    id<MTLCommandQueue> q = mps_queue();
    id<MTLBuffer> ms = mps_buf(buf_src);
    id<MTLBuffer> md = mps_buf(buf_dst);
    if (dev == nil || q == nil || ms == nil || md == nil || rows <= 0 || cols <= 0) return 0;

    @autoreleasepool {
        if (g_mps_softmax == nil) {
            g_mps_softmax = [[MPSMatrixSoftMax alloc] initWithDevice:dev];
        }
        MPSDataType dt = is_f16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
        NSUInteger elem_sz = is_f16 ? 2 : 4;
        NSUInteger s_rb = (NSUInteger)(ld_src > 0 ? ld_src : cols) * elem_sz;
        NSUInteger d_rb = (NSUInteger)(ld_dst > 0 ? ld_dst : cols) * elem_sz;

        MPSMatrixDescriptor* ds = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)rows
                                                                        columns:(NSUInteger)cols
                                                                       rowBytes:s_rb
                                                                       dataType:dt];
        MPSMatrixDescriptor* dd = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)rows
                                                                        columns:(NSUInteger)cols
                                                                       rowBytes:d_rb
                                                                       dataType:dt];
        MPSMatrix* mSrc = [[MPSMatrix alloc] initWithBuffer:ms offset:(NSUInteger)off_src descriptor:ds];
        MPSMatrix* mDst = [[MPSMatrix alloc] initWithBuffer:md offset:(NSUInteger)off_dst descriptor:dd];

        int local_cmd = (g_mps_batch_cmd == nil);
        id<MTLCommandBuffer> cmd = local_cmd ? [q commandBuffer] : g_mps_batch_cmd;
        [g_mps_softmax encodeToCommandBuffer:cmd inputMatrix:mSrc resultMatrix:mDst];
        if (local_cmd) {
            [cmd commit];
            [cmd waitUntilCompleted];
            return ([cmd status] == MTLCommandBufferStatusCompleted) ? 1 : 0;
        }
        return 1;
    }
}

/// Dispatches any Tenon `@kernel` compute pipeline directly into the active MPS command buffer
/// with up to 6 `MTLBuffer` bindings and per-binding byte offsets (zero CPU sync!).
int64_t tn_mps_dispatch_kernel6(
    int64_t pipe_handle,
    int64_t b0, int64_t o0,
    int64_t b1, int64_t o1,
    int64_t b2, int64_t o2,
    int64_t b3, int64_t o3,
    int64_t b4, int64_t o4,
    int64_t b5, int64_t o5,
    int64_t gx, int64_t gy, int64_t gz
) {
    id<MTLComputePipelineState> pso = mps_pipe(pipe_handle);
    id<MTLCommandQueue> q = mps_queue();
    if (pso == nil || q == nil || gx <= 0 || gy <= 0 || gz <= 0) return 0;

    @autoreleasepool {
        int local_cmd = (g_mps_batch_cmd == nil);
        id<MTLCommandBuffer> cmd = local_cmd ? [q commandBuffer] : g_mps_batch_cmd;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pso];

        int64_t bs[6] = { b0, b1, b2, b3, b4, b5 };
        int64_t os[6] = { o0, o1, o2, o3, o4, o5 };
        for (int i = 0; i < 6; i++) {
            if (bs[i] > 0) {
                id<MTLBuffer> mb = mps_buf(bs[i]);
                if (mb != nil) {
                    [enc setBuffer:mb offset:(NSUInteger)(os[i] > 0 ? os[i] : 0) atIndex:(NSUInteger)i];
                }
            }
        }
        MTLSize grid = MTLSizeMake((NSUInteger)gx, (NSUInteger)gy, (NSUInteger)gz);
        NSUInteger maxT = [pso maxTotalThreadsPerThreadgroup];
        NSUInteger tx = (NSUInteger)gx < 64 ? (NSUInteger)gx : 64;
        NSUInteger ty = (NSUInteger)gy < 4 ? (NSUInteger)gy : 4;
        while (tx * ty > maxT && tx > 1) tx /= 2;
        MTLSize tg = MTLSizeMake(tx, ty, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];

        if (local_cmd) {
            [cmd commit];
            [cmd waitUntilCompleted];
            return ([cmd status] == MTLCommandBufferStatusCompleted) ? 1 : 0;
        }
        return 1;
    }
}

/// Hardware GPU-to-GPU Blit copy inside the active MPS command buffer (for KV-Cache / zero-sync moves).
int64_t tn_mps_blit_copy(
    int64_t src_h, int64_t src_off,
    int64_t dst_h, int64_t dst_off,
    int64_t nbytes
) {
    id<MTLCommandQueue> q = mps_queue();
    id<MTLBuffer> ms = mps_buf(src_h);
    id<MTLBuffer> md = mps_buf(dst_h);
    if (q == nil || ms == nil || md == nil || nbytes <= 0) return 0;

    @autoreleasepool {
        int local_cmd = (g_mps_batch_cmd == nil);
        id<MTLCommandBuffer> cmd = local_cmd ? [q commandBuffer] : g_mps_batch_cmd;
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        [blit copyFromBuffer:ms
                sourceOffset:(NSUInteger)src_off
                    toBuffer:md
           destinationOffset:(NSUInteger)dst_off
                        size:(NSUInteger)nbytes];
        [blit endEncoding];
        if (local_cmd) {
            [cmd commit];
            [cmd waitUntilCompleted];
            return ([cmd status] == MTLCommandBufferStatusCompleted) ? 1 : 0;
        }
        return 1;
    }
}

/// Direct unified-memory write/read at arbitrary byte offset (zero extra buffer allocation during decode!).
int64_t tn_mps_buf_write_offset(int64_t handle, int64_t dst_off, int64_t src_ptr, int64_t nbytes) {
    id<MTLBuffer> mb = mps_buf(handle);
    if (mb == nil || src_ptr == 0 || nbytes <= 0 || dst_off < 0) return 0;
    if ((NSUInteger)(dst_off + nbytes) > [mb length]) return 0;
    memcpy((uint8_t*)[mb contents] + (NSUInteger)dst_off, (const void*)(uintptr_t)src_ptr, (NSUInteger)nbytes);
    return nbytes;
}

int64_t tn_mps_buf_read_offset(int64_t handle, int64_t src_off, int64_t dst_ptr, int64_t nbytes) {
    id<MTLBuffer> mb = mps_buf(handle);
    if (mb == nil || dst_ptr == 0 || nbytes <= 0 || src_off < 0) return 0;
    if ((NSUInteger)(src_off + nbytes) > [mb length]) return 0;
    memcpy((void*)(uintptr_t)dst_ptr, (const uint8_t*)[mb contents] + (NSUInteger)src_off, (NSUInteger)nbytes);
    return nbytes;
}

/// Direct inline 4-word `UInt32` uniform update (zero heap allocation inside hot Transformer layer loops!).
int64_t tn_mps_write_u32_4(int64_t handle, int64_t off_bytes, int64_t v0, int64_t v1, int64_t v2, int64_t v3) {
    id<MTLBuffer> mb = mps_buf(handle);
    if (mb == nil || off_bytes < 0 || (NSUInteger)(off_bytes + 16) > [mb length]) return 0;
    uint32_t* p = (uint32_t*)((uint8_t*)[mb contents] + (NSUInteger)off_bytes);
    p[0] = (uint32_t)v0;
    p[1] = (uint32_t)v1;
    p[2] = (uint32_t)v2;
    p[3] = (uint32_t)v3;
    return 16;
}
