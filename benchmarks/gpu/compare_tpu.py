#!/usr/bin/env python3
"""
Same-Machine Google Cloud TPU v5e-1 Head-to-Head Benchmark Harness:
  Tenon Direct PJRT (Zero Python) vs JAX (jax.jit + lax.scan) vs TensorFlow (XLA) vs PyTorch/XLA
Anchored on Physical TPU v5e-1 Silicon Roofline Ceiling: 197 TFLOPS BF16 MXU / 819 GB/s HBM!

Each framework runs in an isolated OS subprocess so /dev/vfio/0 (libtpu.so) has 100% exclusive ownership.
"""
import json, os, re, subprocess, sys, time

TPU_PEAK_TFLOPS = 197.0
TPU_PEAK_GBS = 819.0

JAX_WORKER = r"""
import json, time
import jax
import jax.numpy as jnp

dev = jax.devices()[0]
res = {"framework": f"JAX {jax.__version__}", "device": str(dev)}

def bench(fn, args, warmup=5, iters=30):
    for _ in range(warmup):
        y = fn(*args)
        jax.block_until_ready(y)
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        y = fn(*args)
        jax.block_until_ready(y)
    return (time.perf_counter_ns() - t0) / (iters * 1e6)

# 1. MXU BF16 4096^3
a = jnp.zeros((4096, 4096), dtype=jnp.bfloat16)
b = jnp.zeros((4096, 4096), dtype=jnp.bfloat16)
mm_jit = jax.jit(lambda x, y: jnp.matmul(x, y, preferred_element_type=jnp.float32).astype(jnp.bfloat16))
ms_mm = bench(mm_jit, (a, b), warmup=5, iters=50)
res["mxu_4096_ms"] = ms_mm
res["mxu_4096_tflops"] = (2.0 * 4096**3 / (ms_mm / 1e3)) / 1e12
del a, b

# 2. HBM BF16 GEMV 1x16384x16384
x = jnp.zeros((1, 16384), dtype=jnp.bfloat16)
w = jnp.zeros((16384, 16384), dtype=jnp.bfloat16)
ms_gemv = bench(mm_jit, (x, w), warmup=5, iters=50)
res["hbm_bf16_ms"] = ms_gemv
res["hbm_bf16_gbs"] = ((16384*16384*2.0) / 1e9) / (ms_gemv / 1e3)
del w

# 3. HBM W8A16 INT8 GEMV 1x16384x16384
wq = jnp.zeros((16384, 16384), dtype=jnp.int8)
sc = jnp.ones((16384,), dtype=jnp.bfloat16) * jnp.bfloat16(0.01)
w8_jit = jax.jit(lambda x, wq, sc: (jnp.matmul(x, wq.astype(jnp.bfloat16), preferred_element_type=jnp.float32) * sc.astype(jnp.float32)).astype(jnp.bfloat16))
ms_w8 = bench(w8_jit, (x, wq, sc), warmup=5, iters=50)
res["hbm_w8_ms"] = ms_w8
res["hbm_w8_gbs"] = ((16384*16384*1.0) / 1e9) / (ms_w8 / 1e3)
# 3B. HBM W4A16 Packed INT4 GEMV 1x16384x16384
wq4 = jnp.zeros((16384, 8192), dtype=jnp.int8)
sc4 = jnp.ones((16384,), dtype=jnp.bfloat16) * jnp.bfloat16(0.01)
def unpack_w4(wp):
    lo = jax.lax.shift_right_arithmetic(jax.lax.shift_left(wp, jnp.int8(4)), jnp.int8(4)).astype(jnp.bfloat16)
    hi = jax.lax.shift_right_arithmetic(wp, jnp.int8(4)).astype(jnp.bfloat16)
    return jnp.concatenate([lo, hi], axis=-1)
w4_jit = jax.jit(lambda x, wq, sc: (jnp.matmul(x, unpack_w4(wq), preferred_element_type=jnp.float32) * sc.astype(jnp.float32)).astype(jnp.bfloat16))
ms_w4 = bench(w4_jit, (x, wq4, sc4), warmup=5, iters=50)
res["hbm_w4_ms"] = ms_w4
res["hbm_w4_gbs"] = ((16384*8192*1.0) / 1e9) / (ms_w4 / 1e3)
del x, wq, sc, wq4, sc4

# 4. Statically-Unrolled + Horizontally-Fused 16-Layer Llama-3.2-1B (BF16 / W8A16 / W4A16 — exact match to Tenon!)
def run_llama16(seq, qbits, iters):
    L, D, F = 16, 2048, 8192
    wdtype = jnp.bfloat16 if qbits == 16 else jnp.int8
    qkv_c = (3 * D) // 2 if qbits == 4 else 3 * D
    wo_c = D // 2 if qbits == 4 else D
    gu_c = F if qbits == 4 else 2 * F
    wd_c = D // 2 if qbits == 4 else D
    x0 = jnp.zeros((seq, D), dtype=jnp.bfloat16)
    n1 = jnp.ones((L, D), dtype=jnp.bfloat16)
    n2 = jnp.ones((L, D), dtype=jnp.bfloat16)
    wqkv = jnp.zeros((L, D, qkv_c), dtype=wdtype)
    wo = jnp.zeros((L, D, wo_c), dtype=wdtype)
    wgu = jnp.zeros((L, D, gu_c), dtype=wdtype)
    wd = jnp.zeros((L, F, wd_c), dtype=wdtype)
    inv_sqrt_d = jnp.float32(1.0 / (D ** 0.5))

    def rmsnorm(x_f32, nw):
        ss = jnp.mean(x_f32 * x_f32, axis=-1, keepdims=True)
        return (x_f32 * jax.lax.rsqrt(ss + 1e-5) * nw.astype(jnp.float32)).astype(jnp.bfloat16)

    def deq(w):
        if qbits == 4: return unpack_w4(w)
        if qbits == 8: return w.astype(jnp.bfloat16)
        return w

    @jax.jit
    def fwd(x, n1, wqkv, wo, n2, wgu, wd):
        for i in range(L):
            xf = x.astype(jnp.float32)
            xn1 = rmsnorm(xf, n1[i])
            qkv = jnp.matmul(xn1, deq(wqkv[i]), preferred_element_type=jnp.float32).astype(jnp.bfloat16)
            q, k, v = qkv[:, :D], qkv[:, D:2*D], qkv[:, 2*D:]
            sc = jnp.matmul(q, k.T, preferred_element_type=jnp.float32) * inv_sqrt_d
            prob = jax.nn.softmax(sc, axis=-1).astype(jnp.bfloat16)
            ctx = jnp.matmul(prob, v, preferred_element_type=jnp.float32).astype(jnp.bfloat16)
            h1 = xf + jnp.matmul(ctx, deq(wo[i]), preferred_element_type=jnp.float32)
            xn2 = rmsnorm(h1, n2[i])
            gu = jnp.matmul(xn2, deq(wgu[i]), preferred_element_type=jnp.float32)
            act = (jax.nn.silu(gu[:, :F]) * gu[:, F:]).astype(jnp.bfloat16)
            x = (h1 + jnp.matmul(act, deq(wd[i]), preferred_element_type=jnp.float32)).astype(jnp.bfloat16)
        return x

    ms = bench(fwd, (x0, n1, wqkv, wo, n2, wgu, wd), warmup=3, iters=iters)
    return ms, (seq * 1000.0) / ms

for tag, seq, qbits, iters in [
    ("prefill_s512_bf16", 512, 16, 15),
    ("decode_b32_bf16", 32, 16, 20),
    ("decode_b32_w8", 32, 8, 20),
    ("decode_b32_w4", 32, 4, 20),
    ("decode_b1_bf16", 1, 16, 30),
    ("decode_b1_w8", 1, 8, 30),
    ("decode_b1_w4", 1, 4, 30),
]:
    ms, toks = run_llama16(seq, qbits, iters)
    res[f"{tag}_ms"] = ms
    res[f"{tag}_toks"] = toks

print("JSON_RESULT:" + json.dumps(res))
"""

TF_WORKER = r"""
import json, os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tpus = tf.config.list_logical_devices("TPU")
if not tpus:
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        tpus = tf.config.list_logical_devices("TPU")
    except Exception as e:
        print("JSON_RESULT:" + json.dumps({"framework": f"TensorFlow {tf.__version__}", "error": str(e)}))
        raise SystemExit(0)

res = {"framework": f"TensorFlow {tf.__version__}", "device": str(tpus[0])}
with tf.device("/TPU:0"):
    @tf.function(jit_compile=True)
    def mm_bf16(a, b):
        return tf.cast(tf.linalg.matmul(a, b, output_type=tf.float32), tf.bfloat16)

    def bench(fn, args, warmup=5, iters=30):
        for _ in range(warmup):
            y = fn(*args)
            _ = y[0, 0].numpy()
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            y = fn(*args)
        _ = y[0, 0].numpy()
        return (time.perf_counter_ns() - t0) / (iters * 1e6)

    a = tf.zeros((4096, 4096), dtype=tf.bfloat16)
    b = tf.zeros((4096, 4096), dtype=tf.bfloat16)
    ms_mm = bench(mm_bf16, (a, b), warmup=5, iters=50)
    res["mxu_4096_ms"] = ms_mm
    res["mxu_4096_tflops"] = (2.0 * 4096**3 / (ms_mm / 1e3)) / 1e12
    del a, b

    x = tf.zeros((1, 16384), dtype=tf.bfloat16)
    w = tf.zeros((16384, 16384), dtype=tf.bfloat16)
    ms_gemv = bench(mm_bf16, (x, w), warmup=5, iters=50)
    res["hbm_bf16_ms"] = ms_gemv
    res["hbm_bf16_gbs"] = ((16384*16384*2.0) / 1e9) / (ms_gemv / 1e3)
    del w

    wq = tf.zeros((16384, 16384), dtype=tf.int8)
    sc = tf.ones((16384,), dtype=tf.bfloat16) * tf.cast(0.01, tf.bfloat16)
    @tf.function(jit_compile=True)
    def w8_mm(x, wq, sc):
        wf = tf.cast(wq, tf.bfloat16)
        acc = tf.linalg.matmul(x, wf, output_type=tf.float32) * tf.cast(sc, tf.float32)
        return tf.cast(acc, tf.bfloat16)
    ms_w8 = bench(w8_mm, (x, wq, sc), warmup=5, iters=50)
    res["hbm_w8_ms"] = ms_w8
    res["hbm_w8_gbs"] = ((16384*16384*1.0) / 1e9) / (ms_w8 / 1e3)
    del x, wq, sc

    def run_llama16(seq, w8, iters):
        L, D, F = 16, 2048, 8192
        wdtype = tf.int8 if w8 else tf.bfloat16
        x0 = tf.zeros((seq, D), dtype=tf.bfloat16)
        n1 = tf.ones((L, D), dtype=tf.bfloat16)
        n2 = tf.ones((L, D), dtype=tf.bfloat16)
        wq = tf.zeros((L, D, D), dtype=wdtype)
        wk = tf.zeros((L, D, D), dtype=wdtype)
        wv = tf.zeros((L, D, D), dtype=wdtype)
        wo = tf.zeros((L, D, D), dtype=wdtype)
        wg = tf.zeros((L, D, F), dtype=wdtype)
        wu = tf.zeros((L, D, F), dtype=wdtype)
        wd = tf.zeros((L, F, D), dtype=wdtype)
        inv_sqrt_d = tf.constant(1.0 / (D ** 0.5), dtype=tf.float32)

        @tf.function(jit_compile=True)
        def fwd(x, n1, wq, wk, wv, wo, n2, wg, wu, wd):
            for i in tf.range(L):
                ln1, lq, lk, lv, lo = n1[i], wq[i], wk[i], wv[i], wo[i]
                ln2, lg, lu, ld = n2[i], wg[i], wu[i], wd[i]
                if w8:
                    lq, lk, lv, lo = tf.cast(lq, tf.bfloat16), tf.cast(lk, tf.bfloat16), tf.cast(lv, tf.bfloat16), tf.cast(lo, tf.bfloat16)
                    lg, lu, ld = tf.cast(lg, tf.bfloat16), tf.cast(lu, tf.bfloat16), tf.cast(ld, tf.bfloat16)
                xf = tf.cast(x, tf.float32)
                ss1 = tf.reduce_mean(xf * xf, axis=-1, keepdims=True)
                xn1 = tf.cast(xf * tf.math.rsqrt(ss1 + 1e-5) * tf.cast(ln1, tf.float32), tf.bfloat16)
                q = tf.cast(tf.linalg.matmul(xn1, lq, output_type=tf.float32), tf.bfloat16)
                k = tf.cast(tf.linalg.matmul(xn1, lk, output_type=tf.float32), tf.bfloat16)
                v = tf.cast(tf.linalg.matmul(xn1, lv, output_type=tf.float32), tf.bfloat16)
                sc = tf.linalg.matmul(q, k, transpose_b=True, output_type=tf.float32) * inv_sqrt_d
                prob = tf.cast(tf.nn.softmax(sc, axis=-1), tf.bfloat16)
                ctx = tf.cast(tf.linalg.matmul(prob, v, output_type=tf.float32), tf.bfloat16)
                h1 = xf + tf.linalg.matmul(ctx, lo, output_type=tf.float32)
                ss2 = tf.reduce_mean(h1 * h1, axis=-1, keepdims=True)
                xn2 = tf.cast(h1 * tf.math.rsqrt(ss2 + 1e-5) * tf.cast(ln2, tf.float32), tf.bfloat16)
                gate = tf.linalg.matmul(xn2, lg, output_type=tf.float32)
                up = tf.linalg.matmul(xn2, lu, output_type=tf.float32)
                act = tf.cast(tf.nn.silu(gate) * up, tf.bfloat16)
                x = tf.cast(h1 + tf.linalg.matmul(act, ld, output_type=tf.float32), tf.bfloat16)
            return x

        ms = bench(fwd, (x0, n1, wq, wk, wv, wo, n2, wg, wu, wd), warmup=3, iters=iters)
        return ms, (seq * 1000.0) / ms

    for tag, seq, w8, iters in [
        ("prefill_s512_bf16", 512, False, 15),
        ("decode_b32_bf16", 32, False, 20),
        ("decode_b32_w8", 32, True, 20),
        ("decode_b1_bf16", 1, False, 30),
        ("decode_b1_w8", 1, True, 30),
    ]:
        ms, toks = run_llama16(seq, w8, iters)
        res[f"{tag}_ms"] = ms
        res[f"{tag}_toks"] = toks

print("JSON_RESULT:" + json.dumps(res))
"""

TORCH_XLA_WORKER = r"""
import json, time
try:
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
except Exception as e:
    print("JSON_RESULT:" + json.dumps({"framework": "PyTorch/XLA", "error": f"not installed: {e}"}))
    raise SystemExit(0)

dev = xm.xla_device()
res = {"framework": f"PyTorch {torch.__version__} / XLA", "device": str(dev)}

def bench(fn, warmup=5, iters=30):
    for _ in range(warmup):
        y = fn()
        xm.mark_step()
    xm.wait_device_ops()
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        y = fn()
        xm.mark_step()
    xm.wait_device_ops()
    return (time.perf_counter_ns() - t0) / (iters * 1e6)

a = torch.zeros((4096, 4096), dtype=torch.bfloat16, device=dev)
b = torch.zeros((4096, 4096), dtype=torch.bfloat16, device=dev)
ms_mm = bench(lambda: torch.matmul(a, b), warmup=5, iters=50)
res["mxu_4096_ms"] = ms_mm
res["mxu_4096_tflops"] = (2.0 * 4096**3 / (ms_mm / 1e3)) / 1e12
del a, b

x = torch.zeros((1, 16384), dtype=torch.bfloat16, device=dev)
w = torch.zeros((16384, 16384), dtype=torch.bfloat16, device=dev)
ms_gemv = bench(lambda: torch.matmul(x, w), warmup=5, iters=50)
res["hbm_bf16_ms"] = ms_gemv
res["hbm_bf16_gbs"] = ((16384*16384*2.0) / 1e9) / (ms_gemv / 1e3)
print("JSON_RESULT:" + json.dumps(res))
"""

def run_py_worker(code, label):
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("JSON_RESULT:"):
            return json.loads(line[len("JSON_RESULT:"):])
    return {"framework": label, "error": (p.stderr or p.stdout)[-400:]}

def parse_tenon_output(stdout):
    res = {"framework": "Tenon Native PJRT (0-Python)"}
    for line in stdout.splitlines():
        m = re.search(r"TPU_MXU_BF16 4096x4096x4096 ms=([0-9.]+) TFLOPS=([0-9.]+)", line)
        if m:
            res["mxu_4096_ms"], res["mxu_4096_tflops"] = float(m.group(1)), float(m.group(2))
        m = re.search(r"TPU_MXU_BF16 1x16384x16384 ms=([0-9.]+).*eff_GBs=([0-9.]+)", line)
        if m:
            res["hbm_bf16_ms"], res["hbm_bf16_gbs"] = float(m.group(1)), float(m.group(2))
        m = re.search(r"TPU_HBM_W8A16_GEMV 1x16384x16384 ms=([0-9.]+) eff_GBs=([0-9.]+)", line)
        if m:
            res["hbm_w8_ms"], res["hbm_w8_gbs"] = float(m.group(1)), float(m.group(2))
        m = re.search(r"TPU_HBM_W4A16_GEMV 1x16384x16384 ms=([0-9.]+) eff_GBs=([0-9.]+)", line)
        if m:
            res["hbm_w4_ms"], res["hbm_w4_gbs"] = float(m.group(1)), float(m.group(2))
        m = re.search(r"DECLARATIVE_GPU_GRAPH_TPU llama3_ffn_b32 ms=([0-9.]+) eff_GBs=([0-9.]+)", line)
        if m:
            res["decl_ffn_ms"], res["decl_ffn_gbs"] = float(m.group(1)), float(m.group(2))
        for tag, pat in [
            ("prefill_s512_bf16", "Llama3.2_1B_Prefill_S512_BF16"),
            ("decode_b32_bf16", "Llama3.2_1B_Decode_B32_BF16"),
            ("decode_b32_w8", "Llama3.2_1B_Decode_B32_W8A16"),
            ("decode_b32_w4", "Llama3.2_1B_Decode_B32_W4A16"),
            ("decode_b1_bf16", "Llama3.2_1B_Decode_B1_BF16"),
            ("decode_b1_w8", "Llama3.2_1B_Decode_B1_W8A16"),
            ("decode_b1_w4", "Llama3.2_1B_Decode_B1_W4A16"),
        ]:
            m = re.search(rf"BENCH_TPU_(?:FULL16|FUSED16) {pat}.*full_ms=([0-9.]+) tok_s=([0-9.]+)", line)
            if m:
                res[f"{tag}_ms"], res[f"{tag}_toks"] = float(m.group(1)), float(m.group(2))
    return res

def fmt(v, unit="", pct_of=None):
    if v is None:
        return "N/A"
    s = f"{v:.1f}{unit}"
    if pct_of:
        s += f" ({v/pct_of*100:.0f}%)"
    return s

def main():
    tenon_bin = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tenon_tpu_bench"
    print("=== [1/4] Running Tenon Direct PJRT C-API (Zero Python) ===")
    tp = subprocess.run([tenon_bin], capture_output=True, text=True)
    print(tp.stdout)
    if tp.returncode != 0:
        print("TENON STDERR:", tp.stderr[-500:])
    r_tenon = parse_tenon_output(tp.stdout)

    print("=== [2/4] Running JAX (jax.jit unrolled+fused+W4A16) in isolated subprocess ===")
    r_jax = run_py_worker(JAX_WORKER, "JAX")
    print("JAX:", r_jax)

    print("=== [3/4] Running TensorFlow (@tf.function jit_compile=True) in isolated subprocess ===")
    r_tf = run_py_worker(TF_WORKER, "TensorFlow")
    print("TF:", r_tf)

    print("=== [4/4] Running PyTorch/XLA in isolated subprocess ===")
    r_pt = run_py_worker(TORCH_XLA_WORKER, "PyTorch/XLA")
    print("PT_XLA:", r_pt)

    rows = [r_tenon, r_jax, r_tf, r_pt]
    print("\n================ SAME-MACHINE TPU v5e-1 HEAD-TO-HEAD (Silicon Ceiling: 197 TFLOPS / 819 GB/s) ================")
    hdr = f"{'Framework':<28} | {'MXU 4096^3':<16} | {'HBM BF16':<14} | {'HBM W4':<14} | {'1B S=512':<11} | {'B=32 BF16':<10} | {'B=32 W8':<10} | {'B=32 W4':<10} | {'B=1 BF16':<9} | {'B=1 W8':<9} | {'B=1 W4':<9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['framework']:<28} | ERROR: {r['error'][:95]}")
            continue
        print(
            f"{r['framework']:<28} | "
            f"{fmt(r.get('mxu_4096_tflops'), 'T', TPU_PEAK_TFLOPS):<16} | "
            f"{fmt(r.get('hbm_bf16_gbs'), '', TPU_PEAK_GBS):<14} | "
            f"{fmt(r.get('hbm_w4_gbs'), '', TPU_PEAK_GBS):<14} | "
            f"{fmt(r.get('prefill_s512_bf16_toks')):<11} | "
            f"{fmt(r.get('decode_b32_bf16_toks')):<10} | "
            f"{fmt(r.get('decode_b32_w8_toks')):<10} | "
            f"{fmt(r.get('decode_b32_w4_toks')):<10} | "
            f"{fmt(r.get('decode_b1_bf16_toks')):<9} | "
            f"{fmt(r.get('decode_b1_w8_toks')):<9} | "
            f"{fmt(r.get('decode_b1_w4_toks')):<9}"
        )

if __name__ == "__main__":
    main()
