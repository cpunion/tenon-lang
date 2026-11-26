#!/usr/bin/env python3
"""
Same-Machine NVIDIA CUDA (RTX 4090) Head-to-Head Benchmark Harness:
  Tenon Native CUDA (__dp4a + Dynamic cuBLAS + CUDA Graphs)
  vs Raw C-CUDA (nvcc -O3 + cuBLAS + CUDA Graph)
  vs PyTorch (Eager & torch.compile reduce-overhead)
  vs TensorFlow (if installed)
Anchored on Physical RTX 4090 Silicon Roofline Ceiling: 165.2 TFLOPS FP16 TC (FP32 accum) / 1,008.1 GB/s GDDR6X!
"""
import json, os, re, subprocess, sys, time

PEAK_TFLOPS = 165.2
PEAK_GBS = 1008.1

def bench_pytorch():
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:
        return {"error": f"PyTorch unavailable: {e}"}

    dev = torch.device("cuda:0")
    res_eager = {"framework": f"PyTorch {torch.__version__} (Eager)"}
    res_comp = {"framework": f"PyTorch {torch.__version__} (torch.compile)"}

    def timed(fn, warmup=10, iters=40):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter_ns() - t0) / (iters * 1e6)

    # 1. FP16 Tensor Core GEMM 4096^3
    a = torch.zeros((4096, 4096), dtype=torch.float16, device=dev)
    b = torch.zeros((4096, 4096), dtype=torch.float16, device=dev)
    ms = timed(lambda: torch.matmul(a, b), 10, 40)
    res_eager["tc_4096_tflops"] = (2.0 * 4096**3 / (ms / 1e3)) / 1e12
    res_comp["tc_4096_tflops"] = res_eager["tc_4096_tflops"]
    del a, b

    # 2. FP16 GEMV 8192x8192
    x = torch.zeros((1, 8192), dtype=torch.float16, device=dev)
    w = torch.zeros((8192, 8192), dtype=torch.float16, device=dev)
    ms_g = timed(lambda: F.linear(x, w), 10, 100)
    res_eager["gemv_fp16_gbs"] = (8192 * 8192 * 2.0 / 1e9) / (ms_g / 1e3)
    lin_comp = torch.compile(lambda x, w: F.linear(x, w), mode="reduce-overhead")
    ms_gc = timed(lambda: lin_comp(x, w), 10, 100)
    res_comp["gemv_fp16_gbs"] = (8192 * 8192 * 2.0 / 1e9) / (ms_gc / 1e3)
    del x, w

    # 3. Llama-3.2-1B FP16 Prefill (S=512, 16 layers)
    S, D, FF, QKV, L, V = 512, 2048, 8192, 3072, 16, 128256
    def rmsnorm(t, nw):
        tf = t.float()
        return (tf * torch.rsqrt(tf.pow(2).mean(-1, keepdim=True) + 1e-5) * nw.float()).half()

    nw = torch.ones((D,), dtype=torch.float16, device=dev)
    wqkv = torch.zeros((QKV, D), dtype=torch.float16, device=dev)
    wo = torch.zeros((D, D), dtype=torch.float16, device=dev)
    wg = torch.zeros((FF, D), dtype=torch.float16, device=dev)
    wu = torch.zeros((FF, D), dtype=torch.float16, device=dev)
    wd = torch.zeros((D, FF), dtype=torch.float16, device=dev)
    xp = torch.zeros((S, D), dtype=torch.float16, device=dev)

    def prefill_layer(x):
        xn = rmsnorm(x, nw)
        _ = F.linear(xn, wqkv)
        x = x + F.linear(xn, wo)
        xn2 = rmsnorm(x, nw)
        act = F.silu(F.linear(xn2, wg)) * F.linear(xn2, wu)
        return x + F.linear(act, wd)

    def prefill_16(x):
        for _ in range(L):
            x = prefill_layer(x)
        return x

    ms_pe = timed(lambda: prefill_16(xp), 3, 15)
    res_eager["prefill_1b_toks"] = (S * 1000.0) / ms_pe
    try:
        pf_comp = torch.compile(prefill_layer, mode="reduce-overhead")
        def prefill_16_c(x):
            for _ in range(L):
                x = pf_comp(x)
            return x
        ms_pc = timed(lambda: prefill_16_c(xp), 5, 15)
        res_comp["prefill_1b_toks"] = (S * 1000.0) / ms_pc
    except Exception:
        res_comp["prefill_1b_toks"] = None
    del xp, wqkv, wo, wg, wu, wd

    # 4. Llama-3.2-1B FP16 Decode (B=1, 16 distinct layer weights + 128k LM Head)
    xd = torch.zeros((1, D), dtype=torch.float16, device=dev)
    wqkv_l = [torch.zeros((QKV, D), dtype=torch.float16, device=dev) for _ in range(L)]
    wo_l = [torch.zeros((D, D), dtype=torch.float16, device=dev) for _ in range(L)]
    wg_l = [torch.zeros((FF, D), dtype=torch.float16, device=dev) for _ in range(L)]
    wu_l = [torch.zeros((FF, D), dtype=torch.float16, device=dev) for _ in range(L)]
    wd_l = [torch.zeros((D, FF), dtype=torch.float16, device=dev) for _ in range(L)]
    lmh = torch.zeros((V, D), dtype=torch.float16, device=dev)

    def decode_1b(x):
        for i in range(L):
            xn = rmsnorm(x, nw)
            _ = F.linear(xn, wqkv_l[i])
            x = x + F.linear(xn, wo_l[i])
            xn2 = rmsnorm(x, nw)
            act = F.silu(F.linear(xn2, wg_l[i])) * F.linear(xn2, wu_l[i])
            x = x + F.linear(act, wd_l[i])
        return F.linear(rmsnorm(x, nw), lmh)

    ms_de = timed(lambda: decode_1b(xd), 4, 24)
    wbytes = (L * (QKV * D + D * D + 3 * D * FF) + V * D) * 2.0
    res_eager["decode_1b_fp16_toks"] = 1000.0 / ms_de
    res_eager["decode_1b_fp16_gbs"] = (wbytes / 1e9) * res_eager["decode_1b_fp16_toks"]

    try:
        dec_comp = torch.compile(decode_1b, mode="reduce-overhead")
        ms_dc = timed(lambda: dec_comp(xd), 6, 32)
        res_comp["decode_1b_fp16_toks"] = 1000.0 / ms_dc
        res_comp["decode_1b_fp16_gbs"] = (wbytes / 1e9) * res_comp["decode_1b_fp16_toks"]
    except Exception:
        res_comp["decode_1b_fp16_toks"] = None
        res_comp["decode_1b_fp16_gbs"] = None

    return [res_eager, res_comp]

def bench_tensorflow():
    code = r"""
import json, os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("JSON_TF:" + json.dumps({"framework": f"TensorFlow {tf.__version__}", "error": "no GPU"}))
        raise SystemExit(0)
except Exception as e:
    print("JSON_TF:" + json.dumps({"framework": "TensorFlow", "error": str(e)}))
    raise SystemExit(0)
res = {"framework": f"TensorFlow {tf.__version__} (XLA)"}
with tf.device("/GPU:0"):
    @tf.function(jit_compile=True)
    def mm(a, b): return tf.linalg.matmul(a, b)
    a = tf.zeros((4096, 4096), dtype=tf.float16)
    b = tf.zeros((4096, 4096), dtype=tf.float16)
    for _ in range(5): _ = mm(a, b).numpy()[0, 0]
    t0 = time.perf_counter_ns()
    for _ in range(30): y = mm(a, b)
    _ = y.numpy()[0, 0]
    ms = (time.perf_counter_ns() - t0) / (30 * 1e6)
    res["tc_4096_tflops"] = (2.0 * 4096**3 / (ms / 1e3)) / 1e12
    x = tf.zeros((1, 8192), dtype=tf.float16)
    w = tf.zeros((8192, 8192), dtype=tf.float16)
    for _ in range(5): _ = mm(x, w).numpy()[0, 0]
    t0 = time.perf_counter_ns()
    for _ in range(80): y = mm(x, w)
    _ = y.numpy()[0, 0]
    mg = (time.perf_counter_ns() - t0) / (80 * 1e6)
    res["gemv_fp16_gbs"] = (8192*8192*2.0 / 1e9) / (mg / 1e3)
print("JSON_TF:" + json.dumps(res))
"""
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    for l in p.stdout.splitlines():
        if l.startswith("JSON_TF:"):
            return json.loads(l[len("JSON_TF:"):])
    return {"framework": "TensorFlow", "error": "not installed"}

def parse_tenon(out):
    r = {"framework": "Tenon CUDA (EpiNorm+cuBLASLt+Graph)"}
    for l in out.splitlines():
        m = re.search(r"ROOFLINE_TC_FP16 4096x4096x4096 ms=[0-9.]+ TFLOPS=([0-9.]+)", l)
        if m: r["tc_4096_tflops"] = float(m.group(1))
        m = re.search(r"ROOFLINE_TC_FP16_330T 4096x4096x4096 ms=[0-9.]+ TFLOPS=([0-9.]+)", l)
        if m: r["tc_4096_330t"] = float(m.group(1))
        m = re.search(r"ROOFLINE_TC_FP8_660T 4096x4096x4096 ms=[0-9.]+ TFLOPS=([0-9.]+)", l)
        if m: r["tc_4096_fp8"] = float(m.group(1))
        m = re.search(r"ROOFLINE_TC_FP8_660T_PTX 4096x4096x4096 ms=[0-9.]+ TFLOPS=([0-9.]+)", l)
        if m: r["tc_4096_fp8_ptx"] = float(m.group(1))
        m = re.search(r"ROOFLINE_GEMV 8192x8192_FP16 .*eff_GBs=([0-9.]+)", l)
        if m: r["gemv_fp16_gbs"] = float(m.group(1))
        m = re.search(r"ROOFLINE_GEMV 8192x8192_Q4_0_dp4a .*eff_GBs=([0-9.]+)", l)
        if m: r["gemv_q4_gbs"] = float(m.group(1))
        m = re.search(r"BENCH_CUDA_PREFILL Llama3.2_1B_Prefill_S512_FP16 seq=.*tok_s=([0-9.]+)", l)
        if m: r["prefill_1b_toks"] = float(m.group(1))
        m = re.search(r"BENCH_CUDA_PREFILL Llama3.2_1B_Prefill_S512_FP16_330T .*tok_s=([0-9.]+)", l)
        if m: r["prefill_1b_330t"] = float(m.group(1))
        m = re.search(r"BENCH_CUDA_PREFILL Llama3.2_1B_Prefill_S512_FP8_660T seq=.*tok_s=([0-9.]+)", l)
        if m: r["prefill_1b_fp8"] = float(m.group(1))
        m = re.search(r"BENCH_CUDA_PREFILL Llama3.2_1B_Prefill_S512_FP8_660T_PTX .*tok_s=([0-9.]+)", l)
        if m: r["prefill_1b_fp8_ptx"] = float(m.group(1))
        m = re.search(r"BENCH_CUDA_BATCH_1B Llama3.2_1B_Q4_0 B4_ms=[0-9.]+ B4_tok_s=([0-9.]+).*B8_tok_s=([0-9.]+).*B16_tok_s=([0-9.]+).*B32_tok_s=([0-9.]+)", l)
        if m:
            r["batch_b4_toks"], r["batch_b8_toks"], r["batch_b16_toks"], r["batch_b32_toks"] = (
                float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
            )
        m = re.search(r"BENCH_CUDA Llama3.2_1B_FP16 .*graph_tok_s=([0-9.]+) graph_GBs=([0-9.]+)", l)
        if m: r["decode_1b_fp16_toks"], r["decode_1b_fp16_gbs"] = float(m.group(1)), float(m.group(2))
        m = re.search(r"BENCH_CUDA Llama3.2_1B_Q4_0 .*graph_tok_s=([0-9.]+) graph_GBs=([0-9.]+)", l)
        if m: r["decode_1b_q4_toks"], r["decode_1b_q4_gbs"] = float(m.group(1)), float(m.group(2))
    return r

def parse_ccuda(out):
    r = {"framework": "Raw C-CUDA (nvcc+cuBLAS+Graph)"}
    for l in out.splitlines():
        m = re.search(r"CCUDA_TC_FP16 4096x4096x4096 ms=[0-9.]+ TFLOPS=([0-9.]+)", l)
        if m: r["tc_4096_tflops"] = float(m.group(1))
        m = re.search(r"CCUDA_GEMV_FP16 cublas_ms=[0-9.]+ cublas_GBs=([0-9.]+) vec4_ms=[0-9.]+ vec4_GBs=([0-9.]+)", l)
        if m:
            r["gemv_fp16_gbs"] = float(m.group(2))
            r["gemv_cublas_gbs"] = float(m.group(1))
        m = re.search(r"CCUDA_PREFILL_1B_FP16 ms=[0-9.]+ tok_s=([0-9.]+)", l)
        if m: r["prefill_1b_toks"] = float(m.group(1))
        m = re.search(r"CCUDA_DECODE_1B_FP16_CUBLAS_GRAPH ms=[0-9.]+ tok_s=([0-9.]+) eff_GBs=([0-9.]+)", l)
        if m: r["decode_1b_fp16_toks"], r["decode_1b_fp16_gbs"] = float(m.group(1)), float(m.group(2))
    return r

def fmt(v, unit="", pct=None):
    if v is None: return "-"
    s = f"{v:.1f}{unit}"
    if pct: s += f" ({v/pct*100:.0f}%)"
    return s

def main():
    tenon_bin = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tenon_cuda_bench"
    ccuda_bin = sys.argv[2] if len(sys.argv) > 2 else "/tmp/cublas_ref"

    print("=== [1/4] Running Tenon Native CUDA ===")
    tp = subprocess.run([tenon_bin], capture_output=True, text=True)
    print(tp.stdout)
    if tp.returncode != 0: print("TENON ERR:", tp.stderr[-400:])
    r_tenon = parse_tenon(tp.stdout)

    print("=== [2/4] Running Raw C-CUDA (nvcc -O3 + cuBLAS + CUDA Graph) ===")
    cp = subprocess.run([ccuda_bin], capture_output=True, text=True)
    print(cp.stdout)
    r_ccuda = parse_ccuda(cp.stdout)

    print("=== [3/4] Running PyTorch (Eager + torch.compile) ===")
    pt_rows = bench_pytorch()
    print("PyTorch:", pt_rows)

    print("=== [4/4] Checking TensorFlow ===")
    r_tf = bench_tensorflow()
    print("TF:", r_tf)

    rows = [r_tenon, r_ccuda] + (pt_rows if isinstance(pt_rows, list) else []) + [r_tf]
    print("\n================ SAME-MACHINE NVIDIA RTX 4090 HEAD-TO-HEAD (Silicon Ceiling: 165.2T / 1,008 GB/s) ================")
    hdr = f"{'Framework':<34} | {'TC 4096^3 TFLOPS':<19} | {'FP16 GEMV GB/s':<17} | {'1B S=512 Prefill':<16} | {'1B FP16 Decode':<16} | {'1B FP16 GB/s':<16} | {'1B Q4_0 Decode':<14}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['framework']:<34} | SKIPPED ({r['error'][:75]})")
            continue
        print(
            f"{r['framework']:<34} | "
            f"{fmt(r.get('tc_4096_tflops'), 'T', PEAK_TFLOPS):<19} | "
            f"{fmt(r.get('gemv_fp16_gbs'), '', PEAK_GBS):<17} | "
            f"{fmt(r.get('prefill_1b_toks'), ' t/s'):<16} | "
            f"{fmt(r.get('decode_1b_fp16_toks'), ' t/s'):<16} | "
            f"{fmt(r.get('decode_1b_fp16_gbs'), '', PEAK_GBS):<16} | "
            f"{fmt(r.get('decode_1b_q4_toks'), ' t/s'):<14}"
        )
    if r_tenon.get("tc_4096_330t") or r_tenon.get("tc_4096_fp8") or r_tenon.get("tc_4096_fp8_ptx"):
        print("\n--- Tenon Ada Lovelace sm_89 2x/4x Tensor Core Roofline (330.3T FP16-accum & Native PTX 660.6T FP8) ---")
        print(f"  4096^3 FP16-accum (330.3T ceiling):      {fmt(r_tenon.get('tc_4096_330t'), 'T', 330.3)} | 1B S=512 Prefill: {fmt(r_tenon.get('prefill_1b_330t'), ' tok/s')}")
        print(f"  4096^3 FP8 cublasLt (330.3T FP32 cap):   {fmt(r_tenon.get('tc_4096_fp8'), 'T', 330.3)} | 1B S=512 Prefill: {fmt(r_tenon.get('prefill_1b_fp8'), ' tok/s')}")
        print(f"  4096^3 FP8 Native PTX (660.6T FP16 acc): {fmt(r_tenon.get('tc_4096_fp8_ptx'), 'T', 660.6)} | 1B S=512 Prefill: {fmt(r_tenon.get('prefill_1b_fp8_ptx'), ' tok/s')}")
    if r_tenon.get("batch_b4_toks"):
        print("\n--- Tenon CUDA Multi-Token Speculative/Batch Decode (1.24B Q4_0, Register-Direct B<=4 + W4A16 MMA+SplitK B=5..32) ---")
        print(f"  B=4: {fmt(r_tenon.get('batch_b4_toks'), ' tok/s')} | B=8: {fmt(r_tenon.get('batch_b8_toks'), ' tok/s')} | B=16: {fmt(r_tenon.get('batch_b16_toks'), ' tok/s')} | B=32: {fmt(r_tenon.get('batch_b32_toks'), ' tok/s')}")

if __name__ == "__main__":
    main()
