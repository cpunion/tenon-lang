// benchmarks/gpu/cublas_ref.cu — Raw C-CUDA Reference Baseline (nvcc -O3 + cuBLAS + CUDA Graph)
// Measures identical workloads to Tenon Native CUDA on the exact same NVIDIA GPU:
//   1. FP16 Tensor Core GEMM (4096x4096x4096 via cublasGemmEx CUBLAS_COMPUTE_32F)
//   2. FP16 GEMV Bandwidth Roofline (8192x8192 via cuBLAS Gemv/GemmEx AND custom float4 kernel)
//   3. Llama-3.2-1B FP16 Prefill (S=512, 16 layers, cuBLAS Tensor Core + fused SiLU*Up + RMSNorm)
//   4. Llama-3.2-1B FP16 Decode (B=1, 16 layers, cuBLAS + CUDA Graph!)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>

static inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

__global__ void ref_rmsnorm_f16(const half* __restrict__ x, const half* __restrict__ w, half* __restrict__ out, int d) {
    int row = blockIdx.x;
    const half* xr = x + (size_t)row * d;
    half* orow = out + (size_t)row * d;
    __shared__ float sw[8];
    int tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < d; i += blockDim.x) { float v = __half2float(xr[i]); ss += v * v; }
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xFFFFFFFFu, ss, off);
    if ((tid & 31) == 0) sw[tid >> 5] = ss;
    __syncthreads();
    if (tid < 32) {
        float tot = (tid < (blockDim.x >> 5)) ? sw[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) tot += __shfl_down_sync(0xFFFFFFFFu, tot, off);
        if (tid == 0) sw[0] = rsqrtf(tot / (float)d + 1e-5f);
    }
    __syncthreads();
    float inv = sw[0];
    for (int i = tid; i < d; i += blockDim.x) orow[i] = __float2half(__half2float(xr[i]) * inv * __half2float(w[i]));
}

__global__ void ref_silu_mul_f16(const half* __restrict__ g, const half* __restrict__ u, half* __restrict__ o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float gv = __half2float(g[i]), uv = __half2float(u[i]);
        o[i] = __float2half((gv / (1.0f + expf(-gv))) * uv);
    }
}

__global__ void ref_gemv_f16_vec4(const half* __restrict__ x, const half* __restrict__ w, half* __restrict__ y, int out_dim, int in_dim) {
    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int row0 = (blockIdx.x * 4 + wid) * 2;
    if (row0 >= out_dim) return;
    int row1 = (row0 + 1 < out_dim) ? row0 + 1 : row0;
    int k8 = in_dim >> 3;
    const float4* x8 = (const float4*)x;
    const float4* w0 = (const float4*)(w + (size_t)row0 * in_dim);
    const float4* w1 = (const float4*)(w + (size_t)row1 * in_dim);
    float s0 = 0.0f, s1 = 0.0f;
    for (int i = lane; i < k8; i += 32) {
        float4 xv = __ldg(&x8[i]), a0 = __ldg(&w0[i]), a1 = __ldg(&w1[i]);
        const half2 *xh = (const half2*)&xv, *h0 = (const half2*)&a0, *h1 = (const half2*)&a1;
        #pragma unroll
        for (int g = 0; g < 4; g++) {
            float2 xf = __half22float2(xh[g]), f0 = __half22float2(h0[g]), f1 = __half22float2(h1[g]);
            s0 = fmaf(xf.x, f0.x, fmaf(xf.y, f0.y, s0));
            s1 = fmaf(xf.x, f1.x, fmaf(xf.y, f1.y, s1));
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        s0 += __shfl_down_sync(0xFFFFFFFFu, s0, off);
        s1 += __shfl_down_sync(0xFFFFFFFFu, s1, off);
    }
    if (lane == 0) {
        y[row0] = __float2half(s0);
        if (row0 + 1 < out_dim) y[row0 + 1] = __float2half(s1);
    }
}

static void cublas_mm_f16(cublasHandle_t h, const half* x, const half* w, half* y, int m, int n, int k, float beta) {
    float alpha = 1.0f;
    cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                 w, CUDA_R_16F, k, x, CUDA_R_16F, k, &beta, y, CUDA_R_16F, n,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int mem_clk_khz = 0;
    cudaDeviceGetAttribute(&mem_clk_khz, cudaDevAttrMemoryClockRate, 0);
    double peak_gbs = 2.0 * (double)mem_clk_khz * 1e3 * ((double)prop.memoryBusWidth / 8.0) / 1e9;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    cublasSetStream(cublas, stream);

    // 1. FP16 Tensor Core GEMM 4096^3
    {
        int M = 4096, N = 4096, K = 4096;
        half *x, *w, *y;
        cudaMalloc(&x, (size_t)M*K*2); cudaMalloc(&w, (size_t)N*K*2); cudaMalloc(&y, (size_t)M*N*2);
        cudaMemset(x, 0, (size_t)M*K*2); cudaMemset(w, 0, (size_t)N*K*2);
        for (int i = 0; i < 10; i++) cublas_mm_f16(cublas, x, w, y, M, N, K, 0.0f);
        cudaStreamSynchronize(stream);
        double t0 = now_ms();
        int iters = 40;
        for (int i = 0; i < iters; i++) cublas_mm_f16(cublas, x, w, y, M, N, K, 0.0f);
        cudaStreamSynchronize(stream);
        double ms = (now_ms() - t0) / iters;
        double tflops = (2.0 * M * N * K / (ms / 1e3)) / 1e12;
        printf("CCUDA_TC_FP16 4096x4096x4096 ms=%.4f TFLOPS=%.2f\n", ms, tflops);
        cudaFree(x); cudaFree(w); cudaFree(y);
    }

    // 2. FP16 GEMV 8192x8192 (cuBLAS vs hand-written float4)
    {
        int N = 8192, K = 8192;
        half *x, *w, *y;
        cudaMalloc(&x, (size_t)K*2); cudaMalloc(&w, (size_t)N*K*2); cudaMalloc(&y, (size_t)N*2);
        cudaMemset(x, 0, (size_t)K*2); cudaMemset(w, 0, (size_t)N*K*2);
        for (int i = 0; i < 10; i++) cublas_mm_f16(cublas, x, w, y, 1, N, K, 0.0f);
        cudaStreamSynchronize(stream);
        double t0 = now_ms();
        for (int i = 0; i < 100; i++) cublas_mm_f16(cublas, x, w, y, 1, N, K, 0.0f);
        cudaStreamSynchronize(stream);
        double ms_cb = (now_ms() - t0) / 100.0;
        double gbs_cb = ((double)N * K * 2.0 / 1e9) / (ms_cb / 1e3);

        for (int i = 0; i < 10; i++) ref_gemv_f16_vec4<<<(N+7)/8, 128, 0, stream>>>(x, w, y, N, K);
        cudaStreamSynchronize(stream);
        t0 = now_ms();
        for (int i = 0; i < 100; i++) ref_gemv_f16_vec4<<<(N+7)/8, 128, 0, stream>>>(x, w, y, N, K);
        cudaStreamSynchronize(stream);
        double ms_v4 = (now_ms() - t0) / 100.0;
        double gbs_v4 = ((double)N * K * 2.0 / 1e9) / (ms_v4 / 1e3);
        printf("CCUDA_GEMV_FP16 cublas_ms=%.4f cublas_GBs=%.1f vec4_ms=%.4f vec4_GBs=%.1f peak_GBs=%.1f\n",
               ms_cb, gbs_cb, ms_v4, gbs_v4, peak_gbs);
        cudaFree(x); cudaFree(w); cudaFree(y);
    }

    // 3. Llama-3.2-1B FP16 Prefill (S=512, 16 layers, cuBLAS Tensor Core)
    {
        int S = 512, D = 2048, F = 8192, QKV = 3072, L = 16;
        half *x, *xn, *nw, *wqkv, *qkv, *wo, *wg, *wu, *wd, *g, *u, *act;
        cudaMalloc(&x, S*D*2); cudaMalloc(&xn, S*D*2); cudaMalloc(&nw, D*2);
        cudaMalloc(&wqkv, QKV*D*2); cudaMalloc(&qkv, S*QKV*2); cudaMalloc(&wo, D*D*2);
        cudaMalloc(&wg, F*D*2); cudaMalloc(&wu, F*D*2); cudaMalloc(&wd, D*F*2);
        cudaMalloc(&g, S*F*2); cudaMalloc(&u, S*F*2); cudaMalloc(&act, S*F*2);
        auto run_once = [&]() {
            for (int l = 0; l < L; l++) {
                ref_rmsnorm_f16<<<S, 256, 0, stream>>>(x, nw, xn, D);
                cublas_mm_f16(cublas, xn, wqkv, qkv, S, QKV, D, 0.0f);
                cublas_mm_f16(cublas, xn, wo, x, S, D, D, 1.0f);
                ref_rmsnorm_f16<<<S, 256, 0, stream>>>(x, nw, xn, D);
                cublas_mm_f16(cublas, xn, wg, g, S, F, D, 0.0f);
                cublas_mm_f16(cublas, xn, wu, u, S, F, D, 0.0f);
                ref_silu_mul_f16<<<(S*F+255)/256, 256, 0, stream>>>(g, u, act, S*F);
                cublas_mm_f16(cublas, act, wd, x, S, D, F, 1.0f);
            }
        };
        for (int i = 0; i < 3; i++) run_once();
        cudaStreamSynchronize(stream);
        double t0 = now_ms();
        int iters = 15;
        for (int i = 0; i < iters; i++) run_once();
        cudaStreamSynchronize(stream);
        double ms = (now_ms() - t0) / iters;
        double toks = (S * 1000.0) / ms;
        double flops = L * 2.0 * S * ((double)QKV*D + (double)D*D + 3.0*D*F);
        printf("CCUDA_PREFILL_1B_FP16 ms=%.3f tok_s=%.1f TFLOPS=%.2f\n", ms, toks, (flops/(ms/1e3))/1e12);
        cudaFree(x); cudaFree(xn); cudaFree(nw); cudaFree(wqkv); cudaFree(qkv);
        cudaFree(wo); cudaFree(wg); cudaFree(wu); cudaFree(wd); cudaFree(g); cudaFree(u); cudaFree(act);
    }

    // 4. Llama-3.2-1B FP16 Decode (B=1, 16 layers + 128k LM Head, CUDA Graph + cuBLAS)
    {
        int D = 2048, F = 8192, QKV = 3072, L = 16, V = 128256;
        half *x, *xn, *nw, *wqkv, *qkv, *wo, *wg, *wu, *wd, *g, *u, *act, *lmh, *logits;
        cudaMalloc(&x, D*2); cudaMalloc(&xn, D*2); cudaMalloc(&nw, D*2);
        cudaMalloc(&wqkv, (size_t)L*QKV*D*2); cudaMalloc(&qkv, QKV*2); cudaMalloc(&wo, (size_t)L*D*D*2);
        cudaMalloc(&wg, (size_t)L*F*D*2); cudaMalloc(&wu, (size_t)L*F*D*2); cudaMalloc(&wd, (size_t)L*D*F*2);
        cudaMalloc(&g, F*2); cudaMalloc(&u, F*2); cudaMalloc(&act, F*2);
        cudaMalloc(&lmh, (size_t)V*D*2); cudaMalloc(&logits, V*2);

        cudaGraph_t graph;
        cudaGraphExec_t gexec;
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        for (int l = 0; l < L; l++) {
            ref_rmsnorm_f16<<<1, 256, 0, stream>>>(x, nw, xn, D);
            cublas_mm_f16(cublas, xn, wqkv + (size_t)l*QKV*D, qkv, 1, QKV, D, 0.0f);
            cublas_mm_f16(cublas, xn, wo + (size_t)l*D*D, x, 1, D, D, 1.0f);
            ref_rmsnorm_f16<<<1, 256, 0, stream>>>(x, nw, xn, D);
            cublas_mm_f16(cublas, xn, wg + (size_t)l*F*D, g, 1, F, D, 0.0f);
            cublas_mm_f16(cublas, xn, wu + (size_t)l*F*D, u, 1, F, D, 0.0f);
            ref_silu_mul_f16<<<(F+255)/256, 256, 0, stream>>>(g, u, act, F);
            cublas_mm_f16(cublas, act, wd + (size_t)l*D*F, x, 1, D, F, 1.0f);
        }
        ref_rmsnorm_f16<<<1, 256, 0, stream>>>(x, nw, xn, D);
        cublas_mm_f16(cublas, xn, lmh, logits, 1, V, D, 0.0f);
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0);
        cudaGraphDestroy(graph);

        for (int i = 0; i < 5; i++) cudaGraphLaunch(gexec, stream);
        cudaStreamSynchronize(stream);
        double t0 = now_ms();
        int steps = 32;
        for (int i = 0; i < steps; i++) cudaGraphLaunch(gexec, stream);
        cudaStreamSynchronize(stream);
        double ms = (now_ms() - t0) / steps;
        double toks = 1000.0 / ms;
        double wbytes = (double)(L * (QKV*D + D*D + 3*D*F) + (size_t)V*D) * 2.0;
        printf("CCUDA_DECODE_1B_FP16_CUBLAS_GRAPH ms=%.3f tok_s=%.1f eff_GBs=%.1f\n", ms, toks, (wbytes/1e9)*toks);
        cudaGraphExecDestroy(gexec);
        cudaFree(x); cudaFree(xn); cudaFree(nw); cudaFree(wqkv); cudaFree(qkv);
        cudaFree(wo); cudaFree(wg); cudaFree(wu); cudaFree(wd); cudaFree(g); cudaFree(u); cudaFree(act);
        cudaFree(lmh); cudaFree(logits);
    }

    cublasDestroy(cublas);
    cudaStreamDestroy(stream);
    return 0;
}
