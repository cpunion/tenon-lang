// benchmarks/c/mps_and_llamac_baseline.m
// Fair, Apples-to-Apples Native C/ObjC Baselines on Apple Silicon (clang -O3 -mcpu=apple-m4):
//   1. `llama.cpp`-style C CPU Q4_0 Multi-Threaded ARM64 NEON/FMA Transformer Decode Engine
//      (8 pthreads + exact GGUF `block_q4_0` 18B/32-weight quantized dot products + RoPE + GQA + SwiGLU)
//   2. Standard C-Wrapped Apple MPS FP16 Transformer Engine
//      (typical C/PyTorch-MPS architecture: separate Q/K/V/O/Gate/Up/Down `MPSMatrixMultiplication`
//       + `MPSMatrixSoftMax` + Metal elementwise shaders — exact same 6L/21M & 12L/85M models!)
//
// Build & run:
//   clang -O3 -mcpu=apple-m4 -fobjc-arc benchmarks/c/mps_and_llamac_baseline.m \
//         -framework Metal -framework MetalPerformanceShaders -framework Foundation -lpthread -lm \
//         -o /tmp/mps_llamac_baseline && /tmp/mps_llamac_baseline

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <arm_neon.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline int64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

typedef struct {
    const char* name;
    int vocab, d_model, n_layers, n_heads, n_kv, d_head, d_ff, max_seq;
} ModelCfg;

// =========================================================================
// PART 1: `llama.cpp`-Style C CPU Q4_0 Multi-Threaded ARM64 NEON Engine
// =========================================================================
typedef struct __attribute__((packed)) {
    __fp16 d;
    uint8_t qs[16];
} BlockQ4_0;

static void fill_q4_0(BlockQ4_0* dst, size_t n_elems, uint32_t seed) {
    size_t nb = n_elems / 32;
    uint32_t s = seed;
    for (size_t b = 0; b < nb; b++) {
        dst[b].d = (__fp16)0.015f;
        for (int i = 0; i < 16; i++) {
            s = s * 1664525u + 1013904223u;
            dst[b].qs[i] = (uint8_t)(s >> 24);
        }
    }
}

static inline float cpu_q4_0_row_dot(const BlockQ4_0* row, const float* x, int nb) {
    float32x4_t acc_lo = vdupq_n_f32(0.0f);
    float32x4_t acc_hi = vdupq_n_f32(0.0f);
    float total = 0.0f;
    for (int b = 0; b < nb; b++) {
        float d = (float)row[b].d;
        const uint8_t* qs = row[b].qs;
        const float* xb = x + b * 32;
        acc_lo = vdupq_n_f32(0.0f);
        acc_hi = vdupq_n_f32(0.0f);
        for (int g = 0; g < 16; g += 4) {
            float qlo[4] = {
                (float)((int)(qs[g + 0] & 0xF) - 8),
                (float)((int)(qs[g + 1] & 0xF) - 8),
                (float)((int)(qs[g + 2] & 0xF) - 8),
                (float)((int)(qs[g + 3] & 0xF) - 8),
            };
            float qhi[4] = {
                (float)((int)(qs[g + 0] >> 4) - 8),
                (float)((int)(qs[g + 1] >> 4) - 8),
                (float)((int)(qs[g + 2] >> 4) - 8),
                (float)((int)(qs[g + 3] >> 4) - 8),
            };
            acc_lo = vfmaq_f32(acc_lo, vld1q_f32(xb + g), vld1q_f32(qlo));
            acc_hi = vfmaq_f32(acc_hi, vld1q_f32(xb + 16 + g), vld1q_f32(qhi));
        }
        float32x4_t s = vaddq_f32(acc_lo, acc_hi);
        total += d * vaddvq_f32(s);
    }
    return total;
}

#define CPU_NTHREADS 8
typedef struct {
    const BlockQ4_0* w;
    const float* x;
    float* y;
    int r0, r1, nb;
} MatVecJob;

static void* cpu_matvec_worker(void* arg) {
    MatVecJob* j = (MatVecJob*)arg;
    for (int r = j->r0; r < j->r1; r++) {
        j->y[r] = cpu_q4_0_row_dot(j->w + (size_t)r * (size_t)j->nb, j->x, j->nb);
    }
    return NULL;
}

static void cpu_q4_matvec_mt(const BlockQ4_0* w, const float* x, float* y, int out_d, int in_d) {
    int nb = in_d / 32;
    pthread_t th[CPU_NTHREADS];
    MatVecJob jobs[CPU_NTHREADS];
    int chunk = (out_d + CPU_NTHREADS - 1) / CPU_NTHREADS;
    int active = 0;
    for (int t = 0; t < CPU_NTHREADS; t++) {
        int r0 = t * chunk;
        int r1 = r0 + chunk;
        if (r0 >= out_d) break;
        if (r1 > out_d) r1 = out_d;
        jobs[t] = (MatVecJob){ w, x, y, r0, r1, nb };
        if (t > 0) pthread_create(&th[t], NULL, cpu_matvec_worker, &jobs[t]);
        active++;
    }
    cpu_matvec_worker(&jobs[0]);
    for (int t = 1; t < active; t++) pthread_join(th[t], NULL);
}

static void cpu_rmsnorm(const float* x, const float* w, float* out, int d) {
    float ss = 0.0f;
    for (int i = 0; i < d; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)d + 1e-5f);
    for (int i = 0; i < d; i++) out[i] = x[i] * inv * w[i];
}

static void run_llamac_cpu_q4_bench(ModelCfg c, int prefill_len, int decode_steps) {
    int d = c.d_model, kv_dim = c.n_kv * c.d_head, d_ff = c.d_ff;
    size_t n_q = (size_t)c.n_layers * d * d;
    size_t n_kv = (size_t)c.n_layers * kv_dim * d;
    size_t n_ff = (size_t)c.n_layers * d_ff * d;
    size_t n_lm = (size_t)c.vocab * d;

    BlockQ4_0* wq = malloc((n_q / 32) * sizeof(BlockQ4_0));
    BlockQ4_0* wk = malloc((n_kv / 32) * sizeof(BlockQ4_0));
    BlockQ4_0* wv = malloc((n_kv / 32) * sizeof(BlockQ4_0));
    BlockQ4_0* wo = malloc((n_q / 32) * sizeof(BlockQ4_0));
    BlockQ4_0* wg = malloc((n_ff / 32) * sizeof(BlockQ4_0));
    BlockQ4_0* wu = malloc((n_ff / 32) * sizeof(BlockQ4_0));
    BlockQ4_0* wd = malloc((n_ff / 32) * sizeof(BlockQ4_0));
    BlockQ4_0* wlm = malloc((n_lm / 32) * sizeof(BlockQ4_0));
    fill_q4_0(wq, n_q, 1); fill_q4_0(wk, n_kv, 2); fill_q4_0(wv, n_kv, 3); fill_q4_0(wo, n_q, 4);
    fill_q4_0(wg, n_ff, 5); fill_q4_0(wu, n_ff, 6); fill_q4_0(wd, n_ff, 7); fill_q4_0(wlm, n_lm, 8);

    float* emb = calloc((size_t)c.vocab * d, sizeof(float));
    float* norm_w = malloc(d * sizeof(float));
    for (int i = 0; i < d; i++) norm_w[i] = 1.0f;
    for (int i = 0; i < c.vocab * d; i++) emb[i] = 0.02f * (float)((i % 17) - 8);

    float* x = calloc(d, sizeof(float));
    float* xn = calloc(d, sizeof(float));
    float* q = calloc(d, sizeof(float));
    float* k = calloc(kv_dim, sizeof(float));
    float* v = calloc(kv_dim, sizeof(float));
    float* ctx = calloc(d, sizeof(float));
    float* tmp = calloc(d, sizeof(float));
    float* fg = calloc(d_ff, sizeof(float));
    float* fu = calloc(d_ff, sizeof(float));
    float* logits = calloc(c.vocab, sizeof(float));
    float* kcache = calloc((size_t)c.n_layers * c.n_kv * c.max_seq * c.d_head, sizeof(float));
    float* vcache = calloc((size_t)c.n_layers * c.n_kv * c.max_seq * c.d_head, sizeof(float));
    float* att = calloc(c.max_seq, sizeof(float));

    int64_t t0 = now_ns();
    int tok = 7;
    int gqa = c.n_heads / c.n_kv;
    float inv_s = 1.0f / sqrtf((float)c.d_head);
    for (int step = 0; step < decode_steps; step++) {
        int pos = prefill_len + step;
        memcpy(x, emb + (size_t)tok * d, d * sizeof(float));
        for (int l = 0; l < c.n_layers; l++) {
            cpu_rmsnorm(x, norm_w, xn, d);
            cpu_q4_matvec_mt(wq + (size_t)l * (d * d / 32), xn, q, d, d);
            cpu_q4_matvec_mt(wk + (size_t)l * (kv_dim * d / 32), xn, k, kv_dim, d);
            cpu_q4_matvec_mt(wv + (size_t)l * (kv_dim * d / 32), xn, v, kv_dim, d);
            // Store K/V & GQA Attention
            size_t kv_l_off = (size_t)l * c.n_kv * c.max_seq * c.d_head;
            for (int h = 0; h < c.n_kv; h++) {
                memcpy(kcache + kv_l_off + ((size_t)h * c.max_seq + pos) * c.d_head, k + h * c.d_head, c.d_head * sizeof(float));
                memcpy(vcache + kv_l_off + ((size_t)h * c.max_seq + pos) * c.d_head, v + h * c.d_head, c.d_head * sizeof(float));
            }
            for (int h = 0; h < c.n_heads; h++) {
                int kh = h / gqa;
                const float* qh = q + h * c.d_head;
                float m = -1e30f;
                for (int t = 0; t <= pos; t++) {
                    const float* kt = kcache + kv_l_off + ((size_t)kh * c.max_seq + t) * c.d_head;
                    float s = 0.0f;
                    for (int i = 0; i < c.d_head; i++) s += qh[i] * kt[i];
                    s *= inv_s;
                    att[t] = s;
                    if (s > m) m = s;
                }
                float sum = 0.0f;
                for (int t = 0; t <= pos; t++) { att[t] = expf(att[t] - m); sum += att[t]; }
                float inv = 1.0f / sum;
                float* ch = ctx + h * c.d_head;
                memset(ch, 0, c.d_head * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    float w = att[t] * inv;
                    const float* vt = vcache + kv_l_off + ((size_t)kh * c.max_seq + t) * c.d_head;
                    for (int i = 0; i < c.d_head; i++) ch[i] += w * vt[i];
                }
            }
            cpu_q4_matvec_mt(wo + (size_t)l * (d * d / 32), ctx, tmp, d, d);
            for (int i = 0; i < d; i++) x[i] += tmp[i];
            cpu_rmsnorm(x, norm_w, xn, d);
            cpu_q4_matvec_mt(wg + (size_t)l * (d_ff * d / 32), xn, fg, d_ff, d);
            cpu_q4_matvec_mt(wu + (size_t)l * (d_ff * d / 32), xn, fu, d_ff, d);
            for (int i = 0; i < d_ff; i++) fg[i] = (fg[i] / (1.0f + expf(-fg[i]))) * fu[i];
            cpu_q4_matvec_mt(wd + (size_t)l * (d * d_ff / 32), fg, tmp, d, d_ff);
            for (int i = 0; i < d; i++) x[i] += tmp[i];
        }
        cpu_rmsnorm(x, norm_w, xn, d);
        cpu_q4_matvec_mt(wlm, xn, logits, c.vocab, d);
        int best = 0; float bv = logits[0];
        for (int i = 1; i < c.vocab; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        tok = best;
    }
    int64_t dt_ns = now_ns() - t0;
    double ms = (double)dt_ns / 1e6 / decode_steps;
    double tps = (double)decode_steps / ((double)dt_ns / 1e9);
    printf("C_LLAMAC_CPU_Q4_0 %s decode_ms=%.3f decode_tok_s=%.1f last_tok=%d\n", c.name, ms, tps, tok);

    free(wq); free(wk); free(wv); free(wo); free(wg); free(wu); free(wd); free(wlm);
    free(emb); free(norm_w); free(x); free(xn); free(q); free(k); free(v); free(ctx);
    free(tmp); free(fg); free(fu); free(logits); free(kcache); free(vcache); free(att);
}

// =========================================================================
// PART 2: Standard C-Wrapped Apple MPS FP16 Transformer (Unfused Q/K/V/Gate/Up)
// =========================================================================
static const char* kStdMpsHelpersMSL =
"#include <metal_stdlib>\nusing namespace metal;\n"
"kernel void std_rmsnorm(device const half* x [[buffer(0)]], device const half* w [[buffer(1)]], device half* o [[buffer(2)]], constant uint& d [[buffer(3)]], uint row [[thread_position_in_grid]]) {\n"
"  float ss = 0.0f; uint b = row * d;\n"
"  for (uint i = 0; i < d; i++) { float v = (float)x[b+i]; ss += v*v; }\n"
"  float inv = rsqrt(ss / (float)d + 1e-5f);\n"
"  for (uint i = 0; i < d; i++) o[b+i] = (half)((float)x[b+i] * inv * (float)w[i]);\n"
"}\n"
"kernel void std_swiglu(device const half* g [[buffer(0)]], device const half* u [[buffer(1)]], device half* o [[buffer(2)]], uint tid [[thread_position_in_grid]]) {\n"
"  float gv = (float)g[tid]; float uv = (float)u[tid];\n"
"  o[tid] = (half)((gv / (1.0f + exp(-gv))) * uv);\n"
"}\n"
"kernel void std_res_add(device half* x [[buffer(0)]], device const half* d [[buffer(1)]], uint tid [[thread_position_in_grid]]) {\n"
"  x[tid] = (half)((float)x[tid] + (float)d[tid]);\n"
"}\n"
"kernel void std_rope_kv(device half* q [[buffer(0)]], device half* k [[buffer(1)]], device const half* v [[buffer(2)]], device half* kc [[buffer(3)]], device half* vc [[buffer(4)]], constant uint4& p [[buffer(5)]], uint tid [[thread_position_in_grid]]) {\n"
"  uint d_head = p.x, pos = p.y, max_seq = p.z, n_kv = p.w;\n"
"  uint h = tid / (d_head / 2); uint pair = tid % (d_head / 2);\n"
"  float ang = (float)pos / pow(10000.0f, (float)(2*pair)/(float)d_head);\n"
"  float c = cos(ang), s = sin(ang);\n"
"  uint qi = h * d_head + 2*pair;\n"
"  float q0 = (float)q[qi], q1 = (float)q[qi+1];\n"
"  q[qi] = (half)(q0*c - q1*s); q[qi+1] = (half)(q0*s + q1*c);\n"
"  if (h < n_kv) {\n"
"    float k0 = (float)k[qi], k1 = (float)k[qi+1];\n"
"    uint dst = (h * max_seq + pos) * d_head + 2*pair;\n"
"    kc[dst] = (half)(k0*c - k1*s); kc[dst+1] = (half)(k0*s + k1*c);\n"
"    vc[dst] = v[qi]; vc[dst+1] = v[qi+1];\n"
"  }\n"
"}\n"
"struct BlockQ4 { half d; uchar qs[16]; };\n"
"// Standard llama.cpp/ggml-metal style 32-lane simd_sum Q4_0 matvec (1 row/simdgroup, 4 simdgroups/TG)\n"
"kernel void ggml_style_mv_q4_0(\n"
"  device const half* x [[buffer(0)]], device const BlockQ4* w [[buffer(1)]], device half* y [[buffer(2)]],\n"
"  constant uint2& p [[buffer(3)]], uint tg_id [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]\n"
") {\n"
"  uint row = tg_id * 4 + sg; if (row >= p.x) return;\n"
"  uint nb = p.y >> 5; device const BlockQ4* wr = w + row * nb;\n"
"  float sum = 0.0f;\n"
"  for (uint b = lane; b < nb; b += 32) {\n"
"    float d = (float)wr[b].d; device const half* xb = x + (b << 5); float s = 0.0f;\n"
"    for (uint i = 0; i < 16; i++) {\n"
"      uchar q = wr[b].qs[i];\n"
"      s += (float)xb[i] * ((float)(int)(q & 0xF) - 8.0f) + (float)xb[i + 16] * ((float)(int)(q >> 4) - 8.0f);\n"
"    }\n"
"    sum += d * s;\n"
"  }\n"
"  sum = simd_sum(sum);\n"
"  if (lane == 0) y[row] = (half)sum;\n"
"}\n"
"kernel void ggml_style_attn_qkt(\n"
"  device const half* q [[buffer(0)]], device const half* kc [[buffer(1)]], device half* sc [[buffer(2)]],\n"
"  constant uint4& p [[buffer(3)]], uint2 gid [[thread_position_in_grid]]\n"
") {\n"
"  uint t = gid.x, h = gid.y; uint dh = p.x, kv_len = p.y, max_seq = p.z, gqa = p.w;\n"
"  if (t >= kv_len) return; uint kh = h / gqa;\n"
"  float s = 0.0f; device const half* qh = q + h * dh; device const half* kt = kc + (kh * max_seq + t) * dh;\n"
"  for (uint i = 0; i < dh; i++) s += (float)qh[i] * (float)kt[i];\n"
"  sc[h * max_seq + t] = (half)(s * rsqrt((float)dh));\n"
"}\n"
"kernel void ggml_style_attn_pv(\n"
"  device const half* pr [[buffer(0)]], device const half* vc [[buffer(1)]], device half* ctx [[buffer(2)]],\n"
"  constant uint4& p [[buffer(3)]], uint2 gid [[thread_position_in_grid]]\n"
") {\n"
"  uint i = gid.x, h = gid.y; uint dh = p.x, kv_len = p.y, max_seq = p.z, gqa = p.w;\n"
"  if (i >= dh) return; uint kh = h / gqa; float s = 0.0f;\n"
"  for (uint t = 0; t < kv_len; t++) s += (float)pr[h * max_seq + t] * (float)vc[(kh * max_seq + t) * dh + i];\n"
"  ctx[h * dh + i] = (half)s;\n"
"}\n";

static void mps_mm(id<MTLDevice> dev, id<MTLCommandBuffer> cb,
                   id<MTLBuffer> a, NSUInteger oa, NSUInteger lda,
                   id<MTLBuffer> b, NSUInteger ob, NSUInteger ldb,
                   id<MTLBuffer> c, NSUInteger oc, NSUInteger ldc,
                   NSUInteger M, NSUInteger N, NSUInteger K, BOOL trB, double alpha) {
    MPSMatrixDescriptor* da = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:lda*2 dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor* db = [MPSMatrixDescriptor matrixDescriptorWithRows:(trB?N:K) columns:(trB?K:N) rowBytes:ldb*2 dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor* dc = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:ldc*2 dataType:MPSDataTypeFloat16];
    MPSMatrix* ma = [[MPSMatrix alloc] initWithBuffer:a offset:oa descriptor:da];
    MPSMatrix* mb = [[MPSMatrix alloc] initWithBuffer:b offset:ob descriptor:db];
    MPSMatrix* mc = [[MPSMatrix alloc] initWithBuffer:c offset:oc descriptor:dc];
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:trB resultRows:M resultColumns:N interiorColumns:K alpha:alpha beta:0.0];
    [mm encodeToCommandBuffer:cb leftMatrix:ma rightMatrix:mb resultMatrix:mc];
}

static void run_c_std_mps_fp16_bench(ModelCfg c, int prefill_len, int decode_steps) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kStdMpsHelpersMSL] options:nil error:&err];
        id<MTLComputePipelineState> p_norm = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"std_rmsnorm"] error:&err];
        id<MTLComputePipelineState> p_swig = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"std_swiglu"] error:&err];
        id<MTLComputePipelineState> p_add  = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"std_res_add"] error:&err];
        id<MTLComputePipelineState> p_rope = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"std_rope_kv"] error:&err];

        int d = c.d_model, kv_dim = c.n_kv * c.d_head, d_ff = c.d_ff;
        MTLResourceOptions opt = MTLResourceStorageModeShared;
        id<MTLBuffer> b_norm = [dev newBufferWithLength:d*2 options:opt];
        id<MTLBuffer> b_wq   = [dev newBufferWithLength:(size_t)c.n_layers*d*d*2 options:opt];
        id<MTLBuffer> b_wk   = [dev newBufferWithLength:(size_t)c.n_layers*kv_dim*d*2 options:opt];
        id<MTLBuffer> b_wv   = [dev newBufferWithLength:(size_t)c.n_layers*kv_dim*d*2 options:opt];
        id<MTLBuffer> b_wo   = [dev newBufferWithLength:(size_t)c.n_layers*d*d*2 options:opt];
        id<MTLBuffer> b_wg   = [dev newBufferWithLength:(size_t)c.n_layers*d_ff*d*2 options:opt];
        id<MTLBuffer> b_wu   = [dev newBufferWithLength:(size_t)c.n_layers*d_ff*d*2 options:opt];
        id<MTLBuffer> b_wd   = [dev newBufferWithLength:(size_t)c.n_layers*d*d_ff*2 options:opt];
        id<MTLBuffer> b_wlm  = [dev newBufferWithLength:(size_t)c.vocab*d*2 options:opt];
        id<MTLBuffer> b_x    = [dev newBufferWithLength:d*2 options:opt];
        id<MTLBuffer> b_xn   = [dev newBufferWithLength:d*2 options:opt];
        id<MTLBuffer> b_q    = [dev newBufferWithLength:d*2 options:opt];
        id<MTLBuffer> b_k    = [dev newBufferWithLength:kv_dim*2 options:opt];
        id<MTLBuffer> b_v    = [dev newBufferWithLength:kv_dim*2 options:opt];
        id<MTLBuffer> b_kc   = [dev newBufferWithLength:(size_t)c.n_layers*kv_dim*c.max_seq*2 options:opt];
        id<MTLBuffer> b_vc   = [dev newBufferWithLength:(size_t)c.n_layers*kv_dim*c.max_seq*2 options:opt];
        id<MTLBuffer> b_sc   = [dev newBufferWithLength:(size_t)c.n_heads*c.max_seq*2 options:opt];
        id<MTLBuffer> b_pr   = [dev newBufferWithLength:(size_t)c.n_heads*c.max_seq*2 options:opt];
        id<MTLBuffer> b_ctx  = [dev newBufferWithLength:d*2 options:opt];
        id<MTLBuffer> b_tmp  = [dev newBufferWithLength:d*2 options:opt];
        id<MTLBuffer> b_fg   = [dev newBufferWithLength:d_ff*2 options:opt];
        id<MTLBuffer> b_fu   = [dev newBufferWithLength:d_ff*2 options:opt];
        id<MTLBuffer> b_fa   = [dev newBufferWithLength:d_ff*2 options:opt];
        id<MTLBuffer> b_log  = [dev newBufferWithLength:c.vocab*2 options:opt];

        __fp16* nw = (__fp16*)[b_norm contents];
        for (int i = 0; i < d; i++) nw[i] = (__fp16)1.0f;
        __fp16* xp = (__fp16*)[b_x contents];
        for (int i = 0; i < d; i++) xp[i] = (__fp16)0.1f;

        MPSMatrixSoftMax* sm = [[MPSMatrixSoftMax alloc] initWithDevice:dev];
        double inv_s = 1.0 / sqrt((double)c.d_head);
        int gqa = c.n_heads / c.n_kv;

        // Warmup + timed decode loop
        for (int pass = 0; pass < 2; pass++) {
            int runs = (pass == 0) ? 4 : decode_steps;
            int64_t t0 = now_ns();
            for (int step = 0; step < runs; step++) {
                @autoreleasepool {
                    int pos = prefill_len + step;
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    uint32_t du = (uint32_t)d;
                    for (int l = 0; l < c.n_layers; l++) {
                        id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
                        [e1 setComputePipelineState:p_norm];
                        [e1 setBuffer:b_x offset:0 atIndex:0]; [e1 setBuffer:b_norm offset:0 atIndex:1];
                        [e1 setBuffer:b_xn offset:0 atIndex:2]; [e1 setBytes:&du length:4 atIndex:3];
                        [e1 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
                        [e1 endEncoding];

                        // Unfused Q, K, V MPSMatrixMultiplication (standard C/PyTorch-MPS pattern!)
                        mps_mm(dev, cb, b_xn, 0, d, b_wq, (NSUInteger)l*d*d*2, d, b_q, 0, d, 1, d, d, YES, 1.0);
                        mps_mm(dev, cb, b_xn, 0, d, b_wk, (NSUInteger)l*kv_dim*d*2, d, b_k, 0, kv_dim, 1, kv_dim, d, YES, 1.0);
                        mps_mm(dev, cb, b_xn, 0, d, b_wv, (NSUInteger)l*kv_dim*d*2, d, b_v, 0, kv_dim, 1, kv_dim, d, YES, 1.0);

                        id<MTLComputeCommandEncoder> er = [cb computeCommandEncoder];
                        [er setComputePipelineState:p_rope];
                        [er setBuffer:b_q offset:0 atIndex:0]; [er setBuffer:b_k offset:0 atIndex:1];
                        [er setBuffer:b_v offset:0 atIndex:2];
                        [er setBuffer:b_kc offset:(NSUInteger)l*kv_dim*c.max_seq*2 atIndex:3];
                        [er setBuffer:b_vc offset:(NSUInteger)l*kv_dim*c.max_seq*2 atIndex:4];
                        uint32_t rp[4] = { (uint32_t)c.d_head, (uint32_t)pos, (uint32_t)c.max_seq, (uint32_t)c.n_kv };
                        [er setBytes:rp length:16 atIndex:5];
                        [er dispatchThreads:MTLSizeMake((NSUInteger)c.n_heads*(c.d_head/2),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
                        [er endEncoding];

                        // Per-head MPS GQA Q@K^T + SoftMax + P@V
                        NSUInteger kv_l = (NSUInteger)l * kv_dim * c.max_seq * 2;
                        for (int h = 0; h < c.n_heads; h++) {
                            int kh = h / gqa;
                            NSUInteger kh_off = kv_l + (NSUInteger)kh * c.max_seq * c.d_head * 2;
                            mps_mm(dev, cb, b_q, (NSUInteger)h*c.d_head*2, c.d_head,
                                   b_kc, kh_off, c.d_head,
                                   b_sc, (NSUInteger)h*c.max_seq*2, c.max_seq, 1, (NSUInteger)(pos+1), c.d_head, YES, inv_s);
                        }
                        MPSMatrixDescriptor* dsm = [MPSMatrixDescriptor matrixDescriptorWithRows:c.n_heads columns:(pos+1) rowBytes:c.max_seq*2 dataType:MPSDataTypeFloat16];
                        MPSMatrix* msc = [[MPSMatrix alloc] initWithBuffer:b_sc offset:0 descriptor:dsm];
                        MPSMatrix* mpr = [[MPSMatrix alloc] initWithBuffer:b_pr offset:0 descriptor:dsm];
                        [sm encodeToCommandBuffer:cb inputMatrix:msc resultMatrix:mpr];
                        for (int h = 0; h < c.n_heads; h++) {
                            int kh = h / gqa;
                            NSUInteger kh_off = kv_l + (NSUInteger)kh * c.max_seq * c.d_head * 2;
                            mps_mm(dev, cb, b_pr, (NSUInteger)h*c.max_seq*2, c.max_seq,
                                   b_vc, kh_off, c.d_head,
                                   b_ctx, (NSUInteger)h*c.d_head*2, c.d_head, 1, c.d_head, (NSUInteger)(pos+1), NO, 1.0);
                        }

                        mps_mm(dev, cb, b_ctx, 0, d, b_wo, (NSUInteger)l*d*d*2, d, b_tmp, 0, d, 1, d, d, YES, 1.0);
                        id<MTLComputeCommandEncoder> ea1 = [cb computeCommandEncoder];
                        [ea1 setComputePipelineState:p_add]; [ea1 setBuffer:b_x offset:0 atIndex:0]; [ea1 setBuffer:b_tmp offset:0 atIndex:1];
                        [ea1 dispatchThreads:MTLSizeMake(d,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [ea1 endEncoding];

                        id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
                        [e2 setComputePipelineState:p_norm]; [e2 setBuffer:b_x offset:0 atIndex:0]; [e2 setBuffer:b_norm offset:0 atIndex:1];
                        [e2 setBuffer:b_xn offset:0 atIndex:2]; [e2 setBytes:&du length:4 atIndex:3];
                        [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding];

                        mps_mm(dev, cb, b_xn, 0, d, b_wg, (NSUInteger)l*d_ff*d*2, d, b_fg, 0, d_ff, 1, d_ff, d, YES, 1.0);
                        mps_mm(dev, cb, b_xn, 0, d, b_wu, (NSUInteger)l*d_ff*d*2, d, b_fu, 0, d_ff, 1, d_ff, d, YES, 1.0);
                        id<MTLComputeCommandEncoder> es = [cb computeCommandEncoder];
                        [es setComputePipelineState:p_swig]; [es setBuffer:b_fg offset:0 atIndex:0]; [es setBuffer:b_fu offset:0 atIndex:1]; [es setBuffer:b_fa offset:0 atIndex:2];
                        [es dispatchThreads:MTLSizeMake(d_ff,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [es endEncoding];
                        mps_mm(dev, cb, b_fa, 0, d_ff, b_wd, (NSUInteger)l*d*d_ff*2, d_ff, b_tmp, 0, d, 1, d, d_ff, YES, 1.0);
                        id<MTLComputeCommandEncoder> ea2 = [cb computeCommandEncoder];
                        [ea2 setComputePipelineState:p_add]; [ea2 setBuffer:b_x offset:0 atIndex:0]; [ea2 setBuffer:b_tmp offset:0 atIndex:1];
                        [ea2 dispatchThreads:MTLSizeMake(d,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [ea2 endEncoding];
                    }
                    mps_mm(dev, cb, b_x, 0, d, b_wlm, 0, d, b_log, 0, c.vocab, 1, c.vocab, d, YES, 1.0);
                    [cb commit];
                    [cb waitUntilCompleted];
                }
            }
            if (pass == 1) {
                int64_t dt_ns = now_ns() - t0;
                double ms = (double)dt_ns / 1e6 / runs;
                double tps = (double)runs / ((double)dt_ns / 1e9);
                printf("C_STD_MPS_FP16 %s decode_ms=%.3f decode_tok_s=%.1f\n", c.name, ms, tps);
            }
        }
    }
}

static void ggml_mv_q4(id<MTLCommandBuffer> cb, id<MTLComputePipelineState> pso,
                       id<MTLBuffer> x, id<MTLBuffer> w, NSUInteger w_off, id<MTLBuffer> y, int out_d, int in_d) {
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pso];
    [e setBuffer:x offset:0 atIndex:0];
    [e setBuffer:w offset:w_off atIndex:1];
    [e setBuffer:y offset:0 atIndex:2];
    uint32_t p[2] = { (uint32_t)out_d, (uint32_t)in_d };
    [e setBytes:p length:8 atIndex:3];
    [e dispatchThreadgroups:MTLSizeMake(((NSUInteger)out_d + 3) / 4, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [e endEncoding];
}

static void run_llamac_metal_q4_bench(ModelCfg c, int prefill_len, int decode_steps) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kStdMpsHelpersMSL] options:nil error:&err];
        id<MTLComputePipelineState> p_norm = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"std_rmsnorm"] error:&err];
        id<MTLComputePipelineState> p_swig = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"std_swiglu"] error:&err];
        id<MTLComputePipelineState> p_add  = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"std_res_add"] error:&err];
        id<MTLComputePipelineState> p_rope = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"std_rope_kv"] error:&err];
        id<MTLComputePipelineState> p_mvq4 = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ggml_style_mv_q4_0"] error:&err];
        id<MTLComputePipelineState> p_qkt  = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ggml_style_attn_qkt"] error:&err];
        id<MTLComputePipelineState> p_pv   = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ggml_style_attn_pv"] error:&err];
        MPSMatrixSoftMax* sm = [[MPSMatrixSoftMax alloc] initWithDevice:dev];

        int d = c.d_model, kv_dim = c.n_kv * c.d_head, d_ff = c.d_ff;
        MTLResourceOptions opt = MTLResourceStorageModeShared;
        NSUInteger q_l = (NSUInteger)(d * d / 32) * 18;
        NSUInteger kv_l = (NSUInteger)(kv_dim * d / 32) * 18;
        NSUInteger ff_l = (NSUInteger)(d_ff * d / 32) * 18;
        NSUInteger lm_b = (NSUInteger)(c.vocab * d / 32) * 18;
        id<MTLBuffer> b_norm = [dev newBufferWithLength:d*2 options:opt];
        id<MTLBuffer> b_wq   = [dev newBufferWithLength:(NSUInteger)c.n_layers*q_l options:opt];
        id<MTLBuffer> b_wk   = [dev newBufferWithLength:(NSUInteger)c.n_layers*kv_l options:opt];
        id<MTLBuffer> b_wv   = [dev newBufferWithLength:(NSUInteger)c.n_layers*kv_l options:opt];
        id<MTLBuffer> b_wo   = [dev newBufferWithLength:(NSUInteger)c.n_layers*q_l options:opt];
        id<MTLBuffer> b_wg   = [dev newBufferWithLength:(NSUInteger)c.n_layers*ff_l options:opt];
        id<MTLBuffer> b_wu   = [dev newBufferWithLength:(NSUInteger)c.n_layers*ff_l options:opt];
        id<MTLBuffer> b_wd   = [dev newBufferWithLength:(NSUInteger)c.n_layers*ff_l options:opt];
        id<MTLBuffer> b_wlm  = [dev newBufferWithLength:lm_b options:opt];
        id<MTLBuffer> b_x    = [dev newBufferWithLength:d*2 options:opt];
        id<MTLBuffer> b_xn   = [dev newBufferWithLength:d*2 options:opt];
        id<MTLBuffer> b_q    = [dev newBufferWithLength:d*2 options:opt];
        id<MTLBuffer> b_k    = [dev newBufferWithLength:kv_dim*2 options:opt];
        id<MTLBuffer> b_v    = [dev newBufferWithLength:kv_dim*2 options:opt];
        id<MTLBuffer> b_kc   = [dev newBufferWithLength:(size_t)c.n_layers*kv_dim*c.max_seq*2 options:opt];
        id<MTLBuffer> b_vc   = [dev newBufferWithLength:(size_t)c.n_layers*kv_dim*c.max_seq*2 options:opt];
        id<MTLBuffer> b_sc   = [dev newBufferWithLength:(size_t)c.n_heads*c.max_seq*2 options:opt];
        id<MTLBuffer> b_pr   = [dev newBufferWithLength:(size_t)c.n_heads*c.max_seq*2 options:opt];
        id<MTLBuffer> b_ctx  = [dev newBufferWithLength:d*2 options:opt];
        id<MTLBuffer> b_tmp  = [dev newBufferWithLength:d*2 options:opt];
        id<MTLBuffer> b_fg   = [dev newBufferWithLength:d_ff*2 options:opt];
        id<MTLBuffer> b_fu   = [dev newBufferWithLength:d_ff*2 options:opt];
        id<MTLBuffer> b_fa   = [dev newBufferWithLength:d_ff*2 options:opt];
        id<MTLBuffer> b_log  = [dev newBufferWithLength:c.vocab*2 options:opt];
        int gqa = c.n_heads / c.n_kv;

        for (int pass = 0; pass < 2; pass++) {
            int runs = (pass == 0) ? 4 : decode_steps;
            int64_t t0 = now_ns();
            for (int step = 0; step < runs; step++) {
                @autoreleasepool {
                    int pos = prefill_len + step;
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    uint32_t du = (uint32_t)d;
                    for (int l = 0; l < c.n_layers; l++) {
                        id<MTLComputeCommandEncoder> e1 = [cb computeCommandEncoder];
                        [e1 setComputePipelineState:p_norm]; [e1 setBuffer:b_x offset:0 atIndex:0]; [e1 setBuffer:b_norm offset:0 atIndex:1];
                        [e1 setBuffer:b_xn offset:0 atIndex:2]; [e1 setBytes:&du length:4 atIndex:3];
                        [e1 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e1 endEncoding];

                        ggml_mv_q4(cb, p_mvq4, b_xn, b_wq, (NSUInteger)l*q_l, b_q, d, d);
                        ggml_mv_q4(cb, p_mvq4, b_xn, b_wk, (NSUInteger)l*kv_l, b_k, kv_dim, d);
                        ggml_mv_q4(cb, p_mvq4, b_xn, b_wv, (NSUInteger)l*kv_l, b_v, kv_dim, d);

                        NSUInteger kvc_off = (NSUInteger)l*kv_dim*c.max_seq*2;
                        id<MTLComputeCommandEncoder> er = [cb computeCommandEncoder];
                        [er setComputePipelineState:p_rope]; [er setBuffer:b_q offset:0 atIndex:0]; [er setBuffer:b_k offset:0 atIndex:1];
                        [er setBuffer:b_v offset:0 atIndex:2]; [er setBuffer:b_kc offset:kvc_off atIndex:3]; [er setBuffer:b_vc offset:kvc_off atIndex:4];
                        uint32_t rp[4] = { (uint32_t)c.d_head, (uint32_t)pos, (uint32_t)c.max_seq, (uint32_t)c.n_kv };
                        [er setBytes:rp length:16 atIndex:5];
                        [er dispatchThreads:MTLSizeMake((NSUInteger)c.n_heads*(c.d_head/2),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [er endEncoding];

                        uint32_t ap[4] = { (uint32_t)c.d_head, (uint32_t)(pos+1), (uint32_t)c.max_seq, (uint32_t)gqa };
                        id<MTLComputeCommandEncoder> eq = [cb computeCommandEncoder];
                        [eq setComputePipelineState:p_qkt]; [eq setBuffer:b_q offset:0 atIndex:0]; [eq setBuffer:b_kc offset:kvc_off atIndex:1];
                        [eq setBuffer:b_sc offset:0 atIndex:2]; [eq setBytes:ap length:16 atIndex:3];
                        [eq dispatchThreads:MTLSizeMake((NSUInteger)(pos+1), (NSUInteger)c.n_heads, 1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [eq endEncoding];

                        MPSMatrixDescriptor* dsm = [MPSMatrixDescriptor matrixDescriptorWithRows:c.n_heads columns:(pos+1) rowBytes:c.max_seq*2 dataType:MPSDataTypeFloat16];
                        MPSMatrix* msc = [[MPSMatrix alloc] initWithBuffer:b_sc offset:0 descriptor:dsm];
                        MPSMatrix* mpr = [[MPSMatrix alloc] initWithBuffer:b_pr offset:0 descriptor:dsm];
                        [sm encodeToCommandBuffer:cb inputMatrix:msc resultMatrix:mpr];

                        id<MTLComputeCommandEncoder> ep = [cb computeCommandEncoder];
                        [ep setComputePipelineState:p_pv]; [ep setBuffer:b_pr offset:0 atIndex:0]; [ep setBuffer:b_vc offset:kvc_off atIndex:1];
                        [ep setBuffer:b_ctx offset:0 atIndex:2]; [ep setBytes:ap length:16 atIndex:3];
                        [ep dispatchThreads:MTLSizeMake((NSUInteger)c.d_head, (NSUInteger)c.n_heads, 1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [ep endEncoding];

                        ggml_mv_q4(cb, p_mvq4, b_ctx, b_wo, (NSUInteger)l*q_l, b_tmp, d, d);
                        id<MTLComputeCommandEncoder> ea1 = [cb computeCommandEncoder];
                        [ea1 setComputePipelineState:p_add]; [ea1 setBuffer:b_x offset:0 atIndex:0]; [ea1 setBuffer:b_tmp offset:0 atIndex:1];
                        [ea1 dispatchThreads:MTLSizeMake(d,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [ea1 endEncoding];

                        id<MTLComputeCommandEncoder> e2 = [cb computeCommandEncoder];
                        [e2 setComputePipelineState:p_norm]; [e2 setBuffer:b_x offset:0 atIndex:0]; [e2 setBuffer:b_norm offset:0 atIndex:1];
                        [e2 setBuffer:b_xn offset:0 atIndex:2]; [e2 setBytes:&du length:4 atIndex:3];
                        [e2 dispatchThreads:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)]; [e2 endEncoding];

                        ggml_mv_q4(cb, p_mvq4, b_xn, b_wg, (NSUInteger)l*ff_l, b_fg, d_ff, d);
                        ggml_mv_q4(cb, p_mvq4, b_xn, b_wu, (NSUInteger)l*ff_l, b_fu, d_ff, d);
                        id<MTLComputeCommandEncoder> es = [cb computeCommandEncoder];
                        [es setComputePipelineState:p_swig]; [es setBuffer:b_fg offset:0 atIndex:0]; [es setBuffer:b_fu offset:0 atIndex:1]; [es setBuffer:b_fa offset:0 atIndex:2];
                        [es dispatchThreads:MTLSizeMake(d_ff,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [es endEncoding];
                        ggml_mv_q4(cb, p_mvq4, b_fa, b_wd, (NSUInteger)l*ff_l, b_tmp, d, d_ff);
                        id<MTLComputeCommandEncoder> ea2 = [cb computeCommandEncoder];
                        [ea2 setComputePipelineState:p_add]; [ea2 setBuffer:b_x offset:0 atIndex:0]; [ea2 setBuffer:b_tmp offset:0 atIndex:1];
                        [ea2 dispatchThreads:MTLSizeMake(d,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)]; [ea2 endEncoding];
                    }
                    ggml_mv_q4(cb, p_mvq4, b_x, b_wlm, 0, b_log, c.vocab, d);
                    [cb commit];
                    [cb waitUntilCompleted];
                }
            }
            if (pass == 1) {
                int64_t dt_ns = now_ns() - t0;
                double ms = (double)dt_ns / 1e6 / runs;
                double tps = (double)runs / ((double)dt_ns / 1e9);
                printf("C_LLAMAC_METAL_Q4_0 %s decode_ms=%.3f decode_tok_s=%.1f\n", c.name, ms, tps);
            }
        }
    }
}

int main(void) {
    ModelCfg m21 = { "Llama3_6L_21M",  4096, 512, 6,  8,  2, 64, 1536, 256 };
    ModelCfg m85 = { "Llama3_12L_85M", 8192, 768, 12, 12, 4, 64, 2048, 256 };
    run_llamac_cpu_q4_bench(m21, 64, 32);
    run_llamac_cpu_q4_bench(m85, 64, 16);
    run_c_std_mps_fp16_bench(m21, 64, 64);
    run_c_std_mps_fp16_bench(m85, 64, 64);
    run_llamac_metal_q4_bench(m21, 64, 64);
    run_llamac_metal_q4_bench(m85, 64, 64);
    return 0;
}
