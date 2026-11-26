#!/usr/bin/env bash
# benchmarks/gpu/runpod_moe_showdown.sh
# All-in-One RunPod NVIDIA CUDA (1x RTX 4090 24GB VRAM + 64-128GB RAM) Oversized MoE & Dense Showdown:
#   1. Tenon Native CUDA Streaming Engine (VRAM-LRU + Pinned Dual-Stream PCIe 4.0 DMA Overlap)
#   2. kvcache-ai/ktransformers (★19.7k, AVX-512/AMX kt-kernels + CUDA)
#   3. ikawrakow/ik_llama.cpp (★3.4k, CUDA + AVX-512)
#   4. ggml-org/llama.cpp (★136k, CUDA + AVX-512)
#   5. Helldez/BigMoeOnEdge v0.50.1 (★1.15k, CUDA + NVMe/RAM streaming)
#   6. EricLBuehler/mistral.rs v0.9.5 (★7.9k, Rust CUDA)
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
MODELS="$WORKDIR/models"
COMP="$WORKDIR/competitors"
REPO="${REPO:-$WORKDIR/tenon-lang}"
PROMPT="Explain how Mixture of Experts (MoE) saves compute in two sentences."

mkdir -p "$MODELS/glm45-air" "$MODELS/llama33-70b" "$COMP"

echo "=== [0] Hardware Inventory ==="
nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv
lscpu | grep -E "Model name|CPU\(s\):|Flags" | head -4
free -h
df -h "$WORKDIR"

echo "=== [1] Download Models (GLM-4.5-Air-Q4_0 58.47 GiB + Llama-3.3-70B-Q4_0 37.36 GiB) ==="
pip install -q huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

GLM1="$MODELS/glm45-air/GLM-4.5-Air-Q4_0-00001-of-00002.gguf"
GLM2="$MODELS/glm45-air/GLM-4.5-Air-Q4_0-00002-of-00002.gguf"
L70="$MODELS/llama33-70b/Llama-3.3-70B-Instruct-Q4_0.gguf"

if [ ! -f "$GLM1" ] || [ ! -f "$GLM2" ]; then
    python3 -c "
from huggingface_hub import hf_hub_download
for fn in ['Q4_0/GLM-4.5-Air-Q4_0-00001-of-00002.gguf', 'Q4_0/GLM-4.5-Air-Q4_0-00002-of-00002.gguf']:
    p = hf_hub_download('unsloth/GLM-4.5-Air-GGUF', filename=fn, local_dir='$MODELS/glm45-air')
"
    mv "$MODELS/glm45-air/Q4_0/"*.gguf "$MODELS/glm45-air/" 2>/dev/null || true
fi

if [ ! -f "$L70" ]; then
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('bartowski/Llama-3.3-70B-Instruct-GGUF', filename='Llama-3.3-70B-Instruct-Q4_0.gguf', local_dir='$MODELS/llama33-70b')
"
fi

echo "=== [2] Build & Run Tenon Native CUDA Streaming MoE + Dense ==="
cd "$REPO"
if [ ! -x target/release/tenon ]; then
    cargo build --release --bin tenon
fi
./target/release/tenon build examples/showcase/tenon_cuda_moe_stream_bench.tn -o /tmp/tenon_cuda_moe_bench
TENON_GLM_PATH="$GLM1" TENON_LLAMA_PATH="$L70" /tmp/tenon_cuda_moe_bench

echo "=== [3] Competitor 1: BigMoeOnEdge v0.50.1 (Prebuilt Linux x86_64 + CUDA pack) ==="
mkdir -p "$COMP/bmoe"
if [ ! -x "$COMP/bmoe/bmoe-cli" ]; then
    curl -sL "https://github.com/Helldez/BigMoeOnEdge/releases/download/v0.50.1/bmoe-cli-v0.50.1-linux-x86_64.tar.gz" | tar -xz -C "$COMP/bmoe" --strip-components=1
    curl -sL "https://github.com/Helldez/BigMoeOnEdge/releases/download/v0.50.1/bmoe-pack-cuda-v0.50.1-linux-x86_64.tar.gz" | tar -xz -C "$COMP/bmoe" --strip-components=1 || true
fi
"$COMP/bmoe/bmoe-cli" -m "$GLM1" --auto --plan-explain --ctx-size 512 --cache-type-k f16 --cache-type-v f16 --chatml -p "$PROMPT" -n 32 || true
"$COMP/bmoe/bmoe-cli" -m "$GLM1" --moe-stream --no-overlap --gpu-layers 99 --tensor-overrides 'ffn_.*_exps\.' --device-cache-mb 16384 --cache-mb 0 --ctx-size 512 --cache-type-k f16 --cache-type-v f16 --chatml -p "$PROMPT" -n 32 || true

echo "=== [4] Competitor 2: ik_llama.cpp (CUDA + AVX-512) ==="
if [ ! -x "$COMP/ik_llama.cpp/build/bin/llama-bench" ]; then
    git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp.git "$COMP/ik_llama.cpp"
    cmake -S "$COMP/ik_llama.cpp" -B "$COMP/ik_llama.cpp/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$COMP/ik_llama.cpp/build" --target llama-bench -j "$(nproc)"
fi
"$COMP/ik_llama.cpp/build/bin/llama-bench" -m "$GLM1" -ngl 99 -ncmoe 32 -fmoe 1 -mmp 1 -p 0 -n 32 -r 1
"$COMP/ik_llama.cpp/build/bin/llama-bench" -m "$GLM1" -ngl 99 -cmoe -fmoe 1 -mmp 1 -p 0 -n 32 -r 1
"$COMP/ik_llama.cpp/build/bin/llama-bench" -m "$L70" -ngl 42 -mmp 1 -p 0 -n 16 -r 1

echo "=== [5] Competitor 3: upstream llama.cpp (CUDA + AVX-512) ==="
if [ ! -x "$COMP/llama.cpp/build/bin/llama-bench" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$COMP/llama.cpp"
    cmake -S "$COMP/llama.cpp" -B "$COMP/llama.cpp/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build "$COMP/llama.cpp/build" --target llama-bench -j "$(nproc)"
fi
"$COMP/llama.cpp/build/bin/llama-bench" -m "$GLM1" -ngl 99 -ncmoe 32 -p 0 -n 32 -r 1 || \
"$COMP/llama.cpp/build/bin/llama-bench" -m "$GLM1" -ngl 99 -ot 'exps=CPU' -p 0 -n 32 -r 1
"$COMP/llama.cpp/build/bin/llama-bench" -m "$L70" -ngl 42 -p 0 -n 16 -r 1

echo "=== [6] Competitor 4: kvcache-ai/ktransformers (AVX-512/AMX kt-kernels + CUDA) ==="
echo "Install & run ktransformers on same GLM-4.5-Air-Q4_0 GGUF:"
echo "  pip install ktransformers && python3 -m ktransformers.local_chat --gguf_path $MODELS/glm45-air --model_path zai-org/GLM-4.5-Air --cpu_infer \$(nproc)"
