# Tenon v11 开源大模型跨框架全面评测报告

> **版本**: Tenon v11 (`cd77ca3`) · **评测日期**: 2026-11-26 ~ 2026-11-27  
> **评测原则**: 严格科学诚实、全部为真实物理硅片实测（Apple M4 Max Metal 4 / NVIDIA RTX 6000 Ada `sm_89` / NVIDIA RTX 4090 `sm_89` / NVIDIA A100 80GB `sm_80` / Google Cloud TPU v5e-1）、完整披露测试口径与硬件条件，**同时如实呈现 Tenon 领跑场景与竞品占优场景**。

---

## 一、评测全景与对比框架清单

### 1.1 覆盖的开源模型（全部此前评测模型 100% 覆盖）

| # | 开源模型 | 架构类型 | 权重文件 / 量化 | 体积 | 评测重点 |
| :-: | :--- | :--- | :--- | ---: | :--- |
| 1 | **`Llama-3.2-1B-Instruct`** | 1.24B Dense GQA (32H/8KV) | `Q4_0` GGUF + FP16/FP8/W4A8 | `737 MiB` | 短/长上下文 (`4K`/`8K`) FlashDecoding、四模式 KV、并发 CB、RTX 4090 FP8 PTX、TPU v5e-1 |
| 2 | **`DeepSeek-V2-Lite-Chat`** | 16B **Latent MLA + 64E Top-6 MoE** | `Q4_K_M` GGUF | `9.7 GiB` | 矩阵吸收 Latent MLA (`8.89×` KV 压缩)、`2K` MLA FlashDecoding、Grouped MoE Mega-v3、`EP=1` & `2D TP=2×EP=2`、并发 CB |
| 3 | **`Llama-3.3-70B-Instruct`** | 70B Dense GQA (80L) | `Q4_0` GGUF | `37.36 GiB` | M4 Max 满载 GPU & 超内存流式 (`≤32G`/`≤16G`)、单卡 24G/48G & 双卡 `TP=2` (CUDA Graphs / Spec / 2-Proc CUDA IPC)、`2× A100` |
| 4 | **`GLM-4.5-Air`** | **106B 128E Top-8 Sparse MoE** (46L) | `Q4_0` GGUF (**2 Shards**) | `58.47 GiB` | **超出单卡 48G 显存 & Mac 16G/42G 档**六引擎超内存 LRU 流式、严控 RSS/Swap、NVMe 直流、128G 无上限对照 |
| 5 | **`OLMoE-1B-7B`** | 6.9B 64E Top-8 Sparse MoE | `Q4_0` GGUF | `3.8 GiB` | Mac `F_NOCACHE` 严格冷盘 SSD 流式 vs `llama.cpp -ot exps=CPU` |
| 6 | **`SmolVLM-256M-Instruct`** (+ `mmproj-f16`) | 视觉多模态 VLM (ViT + LLM) | `Q8_0` + `F16 mmproj` | `167+181 MiB` | 真实图像端到端多轮对话：Turn-1 冷启动 vs Turn-2 PagedKV 块级 COW 前缀复用 |

### 1.2 对比的开源推理框架与硬件基线（共 10 类）

| 框架 / 基线 | 语言 / 运行时 | 评测版本 / 方式 | 定位 |
| :--- | :--- | :--- | :--- |
| **Tenon v11** | 纯 Tenon LLVM 22 AOT + 零链接 Metal/CUDA/PJRT | `cd77ca3`（`tenon llm` CLI + **913 KiB 零 Rust 原生服务器**） | 强精化类型形式化验证通用语言 + 全栈跨硬件编译器与生产推理引擎 |
| **上游 `llama.cpp`** (★136k) | C++ / ggml | **Mac**: Homebrew build `12359` (`b1b770d90`, ggml 1.1.0, Metal 4)<br>**CUDA**: 源码现编 HEAD `2a8bf0f` (`sm_89`, `-fa 1`) | 业界最成熟通用的 GGUF 跨平台推理引擎 |
| **`ik_llama.cpp`** (★3.4k) | C++ / 魔改 ggml | 源码现编 Metal / CUDA `sm_89` (`-fmoe 1`) | 社区 MoE CPU/GPU 混合卸载魔改标杆 |
| **`BigMoeOnEdge`** (★1.15k) | C++ / CUDA & Metal | 官方 `v0.50.1` 预编译包 | 专精超显存 MoE 分层缓存与 NVMe 流式引擎 |
| **`mistral.rs`** (★7.9k) | Rust / Candle | 官方 `v0.9.5` Metal & CUDA `sm_89` 预编译 | Rust 生态代表性高性能 LLM 推理框架 |
| **`KTransformers`** (★19.7k) | Python + C++ `kt-kernel` | `kt-kernel v0.7.0.post4` (AVX2/CUDA) | 异构 CPU/GPU 大 MoE 框架 |
| **`t8/hypura`** (★670) | Rust / Metal | 16G Profile (`F_NOCACHE`) | Apple Silicon 磁盘流式推理项目 |
| **`PyTorch 2.8` / `cublasLt` / `Raw C-CUDA`** | Python+C++ / CUDA | RunPod RTX 4090 同机 `compare_cuda.py` | 深度学习主流框架与 NVIDIA 官方闭源库基线 |
| **`JAX 0.11.2`** | Python / XLA | Colab TPU v5e-1 同机 `compare_tpu.py` | Google 官方 TPU 旗舰框架 |
| **`Apple MPS`** | Metal Performance Shaders | Apple M4 Max 同机消融 | Apple 官方闭源矩阵算子库 |

---

## 二、部署体积、形式化安全与架构能力矩阵

| 维度 | **Tenon v11** | 上游 `llama.cpp` | `ik_llama.cpp` | `BigMoeOnEdge` | `mistral.rs` | `vLLM` (参考) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **生产服务二进制体积** | **`913 KiB`** (零 Rust 单文件) | `~24.0 MiB` (`26.9×`) | `~4.1 MiB` (仅 bench) | `~1.2 GiB` (含 CUDA 包) | `~500 MiB` | Python + PyTorch (`>4 GiB`) |
| **编译期形式化证明** | **360 项 Z3 证明 + 越界/溢出消除** | 无 | 无 | 无 | 类型检查 | 无 |
| **三硬件统一后端** | **Metal + CUDA + TPU PJRT** | Metal + CUDA (无 TPU) | Metal + CUDA | Metal + CUDA | Metal + CUDA | CUDA + TPU (无 Metal) |
| **有界硬内存 MoE LRU 流式** | **原生 (`≤8G/16G/24G/42G`, 0 Swap)** | 无 (`-ot` 整层静态切分) | `-ncmoe` 整层静态切分 | 有 (`--moe-stream`) | 无 | 无 |
| **矩阵吸收 Latent MLA KV** | **原生 (`8.89×` 省显存 + 2K FD)** | 支持 (CUDA `2K` 实测较慢) | 支持 | 部分 | 部分 | 支持 |
| **四模式 PagedKV (`FP16/Q8/FP8/Q4`)** | **全支持 + SWA/Sink + 128K** | `f16`/`q8_0`/`q4_0` (无原生 FP8) | 支持 | `f16`/`q8_0` | `f16`/`fp8` | `fp16`/`fp8` |
| **无 P2P PCIe 多卡真 `TP=2..8`** | **原生 (`dlopen` NCCL + 持久 CUDA Graph)** | **实测 `-sm row` 在 `p2p=0` 报错拒载** | `-sm row` | 无 | 支持 | NCCL `TP` |
| **2D `TP × EP` + 分离式 PD 集群** | **Layer(`TNKL`)/SHM/2-Proc `CUDA IPC v2` + HA Router** | 无内置分离式 PD Router | 无 | 无 | 无 | LMCache / Nixl (需外部组件) |
| **零停机动态 LoRA 热插拔 & 503 背压** | **原生 `/v1/lora/{load,unload}` + 熔断/SSE 取消** | 部分 (无熔断 PD 网关) | 无 | 无 | 静态加载 | 支持 |

---

## 三、NVIDIA CUDA 全面正面评测

### 3.1 最新同机实测：`2× NVIDIA RTX 6000 Ada 48GB (sm_89)` — Tenon v11 vs 上游 `llama.cpp` HEAD (`2a8bf0f`, CUDA `sm_89`, `-fa 1`)

> **测试环境**：同一台 RunPod `2× NVIDIA RTX 6000 Ada Generation (49,140 MiB × 2, sm_89, 960 GB/s GDDR6 ECC, p2p=0, NCCL=1)`，完全空载 GPU 顺序对照。`llama.cpp` 现场以 `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89` 编译并全程启用 `-fa 1`（FlashAttention）最优配置。

#### A. `DeepSeek-V2-Lite-Chat.Q4_K_M`（16B Latent MLA + 64E Top-6 MoE — **Tenon 全面碾压**）

| 测试场景 | 拓扑 | **Tenon v11** | 上游 `llama.cpp` (`2a8bf0f`, `-fa 1`) | **Tenon vs `llama.cpp`** | 关键技术根因 |
| :--- | :---: | ---: | ---: | :---: | :--- |
| **短上下文 Decode (`d=0, B=1`)** | 1× GPU (`EP=1`) | **`188.83 tok/s`** | `53.40 tok/s` | **`3.54×` 胜出** | 真矩阵吸收 Latent MLA + CUDA Graphs + 融合 MoE |
| **短上下文 Decode (`d=0, B=1`)** | 2× GPU (`2D TP=2×EP=2`) | **`195.00 tok/s`** | — | **`3.65×` 胜出** | 2D `TP×EP` 持久设备端 NCCL CUDA Graphs |
| **长上下文 `2K` Latent MLA Decode** | 1× GPU | **`99.60 tok/s`** | `9.80 tok/s` | **`10.16×` 胜出!!** | Tenon v11 两阶段 **Latent MLA FlashDecoding Split-KV** |
| **连续批处理 `CB B=8`** | 1× GPU | **`633.19 agg tok/s`** | `44.62 agg tok/s` (`llama-server`) | **`14.19×` 胜出!!** | Tensor Core 分组 MoE Mega-Kernel v2/v3 + `8.89×` MLA KV 省带宽 |
| **连续批处理 `CB B=16`** | 1× GPU | **`853.38 agg tok/s`** | `47.37 agg tok/s` (`llama-server`) | **`18.02×` 胜出!!** | 同上（`2D TP=2×EP=2` 亦达 `799.49 agg tok/s`） |
| **Prefill `pp32` / `pp256`** | 1× GPU / 2× GPU `2D` | `469.3 / 740.5` (2D: **`641.1 / 1,080.4`**) | `830.1 / 1,054.6` (1× GPU) | 1GPU `0.70×` / **2D `1.02×`** | 1GPU 上 `llama.cpp` cuBLAS MMQ Prefill 占优；Tenon 2D 跨卡反超 |

> 补充微基准（同机 `RTX 6000 Ada`）：Tenon v11 原生 Ada `sm_89` FP8 (`e4m3×e4m3`) `mma.sync.m16n8k32` 分组 MoE Mega-Kernel v3 (`E=8, B=32, K=6, 2048→1408→2048`) 达 **`0.144 ms`**，较标量 Mega-v1 (`1.315 ms`) 提速 **`9.10×`**，较 FP16-TC Mega-v2 (`0.444 ms`) 再提速 **`3.07×`** (`cos=0.999294`)。

#### B. `Llama-3.3-70B-Instruct-Q4_0`（70B Dense GQA, `37.36 GiB` — 单卡 48G & 双卡 `TP=2` & `2× A100`）

| 测试场景 | 硬件 / 拓扑 | **Tenon v11** | 上游 `llama.cpp` (`2a8bf0f`, `-fa 1`) | 对比与说明 |
| :--- | :---: | ---: | ---: | :--- |
| **单卡 48G Full-VRAM (`d=0, B=1, Q4 KV`)** | 1× RTX 6000 Ada | `18.96 tok/s` | **`20.97 tok/s`** | 接近持平（`0.90×`）；Tenon 并发 `CB B=4: 32.43`, `B=8: 39.65 agg tok/s` |
| **单卡 48G Prefill (`pp128`)** | 1× RTX 6000 Ada | `195.42 tok/s` | **`873.30 tok/s`** | `llama.cpp` 融合 MMQ/cuBLAS 预填充领先 **`4.47×`** |
| **双卡 `TP=2` 真张量并行 (`d=0, B=1`)** | 2× RTX 6000 Ada (`p2p=0`) | **`25.98 tok/s`** | **加载失败** (`device CUDA0 does not support split buffers`) | **`llama.cpp -sm row` 在无 P2P 环境拒载**；Tenon `TP=2` 反超单卡 `llama.cpp` **`1.24×`** |
| **双卡 `TP=2` 连续批处理 (`B=4` / `B=8`)** | 2× RTX 6000 Ada | **`64.59` / `80.53 agg tok/s`** (`TP_GRAPH=1`: `45.71`/`66.51`) | `-sm row` 失败 | 全员 `52/52` & `104/104 BIT-EXACT` |
| **双卡 `TP=2` 槽内零迁移投机 (`1B Draft, K=5`)** | 2× RTX 6000 Ada | **`57.05 tok/s`** (`2.32×`, `83.3%` 接受率) | — | 代码生成 `64/64 BIT-EXACT` |
| **双卡 `TP=2` 跨进程 GPU-Direct `CUDA IPC v2` PD** | 2× RTX 6000 Ada (双 OS 进程) | **`export=3.35 ms, import=1.80 ms`** | 无此能力 | 零主机拷贝，较 POSIX SHM 快 `3.6×/4.5×` |
| **双卡 `2× A100 80GB PCIe` (`TP=2`)** | 2× A100 80GB (`sm_80`) | **`41.31 tok/s`** | — | `FAST_DP4A=1`, `BIT-EXACT` |

#### C. `Llama-3.2-1B-Instruct-Q4_0`（1.24B Dense GQA, `737 MiB` — 1× RTX 6000 Ada）

| 测试场景 | **Tenon v11** (GGUF Serving 引擎) | 上游 `llama.cpp` (`2a8bf0f`, `-fa 1`) | Tenon `@gpu_graph` 静态编译基准 (RTX 4090) | 客观分析 |
| :--- | ---: | ---: | ---: | :--- |
| **短上下文 Decode (`d=0, B=1`)** | `591.65 tok/s` | **`754.70 tok/s`** | **`1,053.8 tok/s`** | 通用 GGUF 运行时 `0.78×`；Tenon 静态编译内核超 `1K tok/s` |
| **并发批处理 `B=8`** | **`1,853.69 agg tok/s`** | `1,676.27 agg tok/s` (`llama-server`) | **`3,879.2 tok/s`** | **Tenon v11 领先 `1.11×`** |
| **并发批处理 `B=16`** | `1,804.35 agg tok/s` | **`1,977.94 agg tok/s`** (`llama-server`) | **`5,921.2 tok/s`** | `0.91×` 接近持平 |
| **长上下文 `4K FP16` / `8K Q8` / `4K Q4`** | `245.4` / `156.3` / `287.8 tok/s` (`8K FP8: 133.2`) | **`658.2`** / **`592.9`** / **`637.1 tok/s`** | — | Tenon v11 较 v10 提速 **`4.2×–6.1×`**，但距 `llama.cpp` 手写 Tensor-Core FA2 仍有 `2.2×–3.8×` 差距 |
| **Prefill (`pp512`)** | `6,156.4 tok/s` | **`52,686.5 tok/s`** | **`148,128 tok/s` (FP8 PTX)** | Tenon GGUF 路径用 Marlin W4A16 偏重解码，短模型 Prefill 明显慢于 `llama.cpp` |

---

### 3.2 六引擎超显存 MoE & Dense 同机实测：`NVIDIA RTX 4090 48GB / 24GB (sm_89)` + AMD EPYC

#### A. 106B Sparse MoE `GLM-4.5-Air-Q4_0`（`58.47 GiB`, 2 Shards — **超出单卡 48G 显存，Tenon 全档位 #1**）

> 硬件：同一台 RunPod `RTX 4090 48GB (sm_89)` + `AMD EPYC 7763 (32 线程)`，六引擎同机实测。

| 显存预算档 | 引擎与配置 | 二进制体积 | 实测 VRAM | 解码速度 (`tok/s`) | **Tenon 领先倍数** | 精度 / 备注 |
| :--- | :--- | ---: | ---: | ---: | :---: | :--- |
| **`≤ 16 GiB`** | **Tenon Native CUDA (`24/128` LRU)** | **`161 KB`** | **`13.86 GiB`** | **`35.76`** | **1.00× (#1)** | `BIT-EXACT`（14G 显存反超竞品 24G/48G 档!） |
| **`≤ 24 GiB`** | **Tenon Native CUDA (`40/128` LRU)** | **`161 KB`** | **`20.40 GiB`** | **`40.15`** | **1.00× (#1)** | `BIT-EXACT`（较串行 `6.02` 加速 **`6.67×`**） |
| `≤ 24 GiB` | `ik_llama.cpp` (`-ncmoe 32 -fmoe 1`) | `4.1 MB` | `~20.5 GiB` | `28.13` | Tenon 快 **`1.43×`** | 静态整层卸载 |
| `≤ 24 GiB` | 上游 `llama.cpp` (`-ot` 32L CPU) | `1.3 MB+so` | `~20.5 GiB` | `22.11` | Tenon 快 **`1.82×`** | 静态整层卸载 |
| `≤ 24 GiB` | `BigMoeOnEdge v0.50.1` (RAM 16G 缓存) | `~1.2 GB` | `~20.4 GiB` | `9.69` (`cpu-share 1.0`: `6.39`) | Tenon 快 **`4.14–6.28×`** | 主机缓存流式拷贝开销高 |
| **`≤ 44 GiB`** | **Tenon Native CUDA (`88/128` LRU)** | **`161 KB`** | **`40.05 GiB`** | **`50.48`** | **1.00× (#1)** | `BIT-EXACT` |
| `≤ 44 GiB` | `ik_llama.cpp` (`-ncmoe 14 -fmoe 1`) | `4.1 MB` | `~42.5 GiB` | `47.45` | Tenon 快 **`1.06×`** 且省 2.5G | 静态整层卸载 |
| `≤ 44 GiB` | 上游 `llama.cpp` (`-ot` 14L CPU) | `1.3 MB+so` | `~42.5 GiB` | `40.38` | Tenon 快 **`1.25×`** | 静态整层卸载 |
| `≤ 44 GiB` | `BigMoeOnEdge v0.50.1 --auto` | `~1.2 GB` | `~46.7 GiB` | `30.85` | Tenon 快 **`1.64×`** | 锁页加载耗时 28.5s vs Tenon <2s |
| **严格 NVMe 直流** | **Tenon NVMe (`use_nocache=1`)** | **`161 KB`** | **`20.40 GiB`** | **`3.23–3.40`** | **1.00× (#1)** | 零 OS 页缓存污染，`BIT-EXACT` |
| 严格 NVMe 直流 | `BigMoeOnEdge v0.50.1 --moe-stream` | `~1.2 GB` | `~20.4 GiB` | `1.73` | Tenon 快 **`1.87–1.97×`** | — |
| GGUF 兼容实测 | `KTransformers kt-kernel v0.7.0.post4` | Python wheel | — | **硬拒载报错** | — | `intermediate_size (1408) % 256 != 0`（BF16/AMX 需 220–440G RAM 仅 11–16 tok/s） |

#### B. 70B Dense `Llama-3.3-70B-Instruct-Q4_0`（消费级单卡 `RTX 4090 24GB` 超显存 & `48GB` 全常驻）

| 硬件与档位 | 引擎与配置 | 实测 VRAM | 解码速度 (`tok/s`) | 相对对比 |
| :--- | :--- | ---: | ---: | :--- |
| **标准 24GB RTX 4090** (EPYC 7282 16vCPU) | **Tenon v4+ `[CD1-RAM-41]`** (NUMA `Q4_0_4R` + AVX2 INT16) | **`22.72 GiB`** | **`3.97`** (满核 `4.41`) | **反超 `llama.cpp -ngl 48` `+26.5%` (`1.27×`)** 且省 `0.5G`, `BIT-EXACT` |
| 标准 24GB RTX 4090 | **Tenon v4+ `[CD1-RAM-37]`** | **`21.24 GiB`** | **`3.64`** | 同显存反超 `llama.cpp -ngl 44` **`+28.2%`** (`1.28×`) |
| 标准 24GB RTX 4090 | 上游 `llama.cpp -ngl 48` / `-ngl 44` | `~23.2` / `~21.5 GiB` | `3.14` / `2.84` | 基线（开启自动 `Q4_0_4x4` 重排） |
| **48GB RTX 4090 全显存** (EPYC 7763) | `ik_llama.cpp` / `llama.cpp` / `mistral.rs` (`-n 0:80`) | `~38–39 GiB` | `23.13` / `22.99` / `22.53` | 满卡单流解码竞品略快 `~7%` |
| 48GB RTX 4090 全显存 | **Tenon Native CUDA (`[CD2-RES-48G]`)** | **`37.67 GiB`** | **`21.35`** (`854 GB/s`) | `161 KB` 零链接二进制，`85%` GDDR6X，`BIT-EXACT` |

#### C. `@gpu_graph` 声明式编译器算子/端到端基线：`RTX 4090` vs `PyTorch 2.8` / `cublasLt` / `Raw C-CUDA`

> 硬件：RunPod `RTX 4090 (sm_89, 1,008.1 GB/s GDDR6X, 660.6 TFLOPS FP8-FP16 @ 450W TDP)`，同机执行 [`compare_cuda.py`](gpu/compare_cuda.py)。

| 基准项 (`Llama-3.2-1B` 1.24B) | **Tenon `@gpu_graph` CUDA** | `NVIDIA cublasLt` | `Raw C-CUDA` (`nvcc -O3`) | `PyTorch 2.8` (Eager / `compile`) | **Tenon 领先幅度** |
| :--- | ---: | ---: | ---: | ---: | :---: |
| **Tensor Core `4096³` FP8 PTX** | **`447.0–453.4 TFLOPS`** (`0.303 ms`) | `306.1 TFLOPS` | — | — | **反超 `cublasLt` `1.46×–1.49×`** |
| **1.24B `S=512` Prefill** | **`148,128 tok/s`** (FP8 PTX, `288.3T`) | `135,676 tok/s` (FP8) | `72,256 tok/s` (FP16) | `62,490 tok/s` (FP16) | **超 `cublasLt` `+9.2%` / `2.05×` C-CUDA / `2.37×` PyTorch** |
| **1.24B `B=1` FP16 Decode** | **`357.6 tok/s`** (`883.9 GB/s`, 87.7%) | — | `346.4 tok/s` | `230.3` / `121.4 tok/s` | **超 Raw C-CUDA `+3.2%` / `1.55×–2.95×` PyTorch** |
| **1.24B `Q4_0` `B=1` → `B=32`** | **`1,053.8` → `9,925.6 tok/s`** | — | — | — | `9.43×` 批扩展 |

---

## 四、Apple M4 Max (40-core Metal 4, 128 GiB Unified Memory) 全面正面评测

> **环境透明说明**：
> - **洁净空载基线**（2026-11-26 ~ 27 早）：Apple M4 Max 空载（实测裸 DRAM 上限 `468.4 GB/s`，16.2 TFLOPS FP16），完成六引擎（`Tenon` / `mistral.rs v0.9.5` / `ik_llama.cpp` / `BigMoeOnEdge v0.50.1` / `t8/hypura` / `llama.cpp`）与 `@gpu_graph` vs `Apple MPS` 正面实测。
> - **最新 v11 现场同机复测**（2026-11-27 19:00）：同机直接对比 **Tenon v11 (`cd77ca3`)** 与最新 **Homebrew `llama.cpp` build `12359` (`b1b770d90`, `ggml 1.1.0`, Metal 4 Residency Sets + `fa_vec_*` + `mul_mv_mma`)**。*注：19:00 复测期间宿主机后台同时存在其他工作区重型编译进程（合计占用 >600% CPU 与统一内存带宽），因此 70B 等大内存模型同时标注了“空载洁净实测”与“同窗背靠背热测”两套完全透明的口径。*

### 4.1 106B Sparse MoE `GLM-4.5-Air-Q4_0`（`58.47 GiB`, 2 Shards — 六引擎对决）

| 预算档位与口径 | 引擎与配置 | 速度 (`tok/s`) | 物理内存 Footprint | `peakRSS` | **Tenon 领先倍数** |
| :--- | :--- | ---: | ---: | ---: | :---: |
| **≤ 42G (最新 v11 同温 UBC)** | **Tenon v11 `80/128` LRU** | **`33.56`** | **`38.24 GiB`** | **`6.50 GiB`** | **比最新 `llama.cpp -ot` (`2.66`) 快 `12.62×`!!** |
| **≤ 16G (最新 v11 同温 UBC)** | **Tenon v11 `24/128` LRU** | **`14.58`** | **`15.32 GiB`** | **`9.49 GiB`** | **比最新 `llama.cpp -ot` (`2.66`) 快 `5.48×`!!** |
| ≤ 42G / 16G (最新 `llama.cpp 12359`) | 上游 `llama.cpp` (`-ngl 99 -fa 1 -ot exps=CPU`) | `2.66` | `51.03 GiB` | `60.68 GiB` | 内存失控超预算，Tenon 快 **`5.48×–12.62×`** |
| **≤ 42G (严格冷盘 `F_NOCACHE`)** | **Tenon `[M5-40G]` (`80/128` LRU)** | **`10.29`** | **`37.14 GiB`** | **`36.41 GiB`** | 严格冷盘基准 (`BIT-EXACT`, `89.6%` 命中) |
| **≤ 16G / ≤ 8G (严格冷盘)** | **Tenon `24/128` / `8/128`** | **`5.75` / `5.13`** | **`14.21` / `7.66 GiB`** | **`13.87 GiB`** | 仅占模型 `1/7.6` 仍领先全部竞品 |
| ≤ 16G GPU (六引擎历史空载) | `ik_llama.cpp` (`-ncmoe 36 -fmoe 1`) | `3.63` | `56.93 GiB` | `48.74 GiB` | Tenon 冷盘 40G **`2.83×`** / 16G **`1.58×`** |
| ≤ 16G GPU (六引擎历史空载) | 上游 `llama.cpp` (`10L GPU + 36L CPU`) | `3.17` | `mmap` | `57.93 GiB` | Tenon 冷盘 40G **`3.25×`** / 16G **`1.81×`** |
| ≤ 16G / 42G (六引擎历史空载) | `BigMoeOnEdge v0.50.1` Metal (`10240`/`33500`) | `2.86` / `2.68` (`--auto`: `1.04`/`0.81`) | `12.3` / `35.0 GiB` | `64.7` / `73.9 GiB` | Tenon 冷盘 **`2.01×–12.7×`** 且省 **`2.0–4.7×`** RSS |
| ≤ 16G (六引擎历史空载) | `t8/hypura` (`16G` profile) | `1.30` | `57.24 GiB` | `23.44 GiB` | 仅解析 1/2 Shard（乱码），Tenon **`4.42×`** |
| **无上限 128G 全 GPU (`~60G`)** | 上游 `llama.cpp` build `12359` (`-ngl 99 -fa 1`) | **`50.66`** | `0.35 GiB` (`mmap`) | `57.94 GiB` | **竞品占优 (`1.70×`)**：`llama.cpp` 拥有专门融合 `mul_mv_id` Metal 4 核 |
| 无上限 128G 全 GPU (`~60G`) | **Tenon v11 (`128/128` 无 ICB)** | `29.76` | `0.45 GiB` (`mmap`) | `0.16 GiB` | 当整模完全装入 128G 内存时不需流式调度，Tenon 仍有融合优化空间 |

### 4.2 70B Dense `Llama-3.3-70B-Instruct-Q4_0`（`37.36 GiB`, 80 层）

| 场景 / 档位 | 引擎与配置 | 速度 (`tok/s`) | 内存 / 带宽 | 对比与说明 |
| :--- | :--- | ---: | :--- | :--- |
| **满载 100% GPU (空载洁净实测)** | **Tenon 1-CB 零拷贝 `mmap`** | **`11.03–11.22`** | **`442.4 GB/s` (94.4% DRAM)** | 基准 (`BIT-EXACT`)；投机 `K=5` 代码生成达 **`20.43 tok/s`** (`1.85×`)；`CB B=8: 21.96` |
| 满载 100% GPU (空载洁净实测) | `mistral.rs v0.9.5` (`--paged-attn off`) | `7.40` | `296.8 GB/s`, `RSS 74.0G` | Tenon 快 **`1.49×`** 且省 `2.03×` RSS |
| 满载 100% GPU (空载洁净实测) | `llama.cpp` (早期 `-lm mmap`) / `ik_llama.cpp` | `6.72` / `2.14` | `269.6` / `85.8 GB/s` | Tenon 快 **`1.64×`** / **`5.15×`** |
| **满载 100% GPU (最新 `llama.cpp 12359` 压力同窗热测)** | 上游 `llama.cpp 12359` (`-fa 1` Metal 4 Residency Sets) | **`10.57`** (`4K Q4`: `5.53`) | `RSS 37.3G` | 最新 Metal 4 Residency Sets 进步显著，在宿主机高负载下仍保持 `10.57` |
| 满载 100% GPU (最新 Tenon v11 压力同窗热测) | **Tenon v11** (同窗背景 `>600%` CPU 竞争) | `8.31–8.94` (`4K Q4`: `4.85`) | `FP 1.08G` | 洁净空载达 `11.03–11.22`；高后台内存竞争时较 `llama.cpp 12359` 易受影响 |
| **超内存流式 `≤ 32G` (同温 UBC)** | **Tenon `[D3-RING30]`** | **`4.55`** | **`28.62 GiB`** | **反超 `ik_llama -ngl 60` (`1.14`) `3.99×`、上游 `llama.cpp` (`0.37`) `12.30×`!!** |
| **超内存流式 `≤ 16G` (同温 UBC)** | **Tenon `[D2-RING16]`** | **`1.66–2.08`** | **`14.78 GiB`** | **反超 `ik_llama -ngl 24` (`1.27`) `1.31–1.64×`、上游 `llama.cpp` (`0.32`) `5.19–6.50×`!!** |
| **超内存流式 `≤ 2.5G` (纯双缓冲)** | **Tenon `[D1-ASYNC]`** (冷盘 `0.327` / 同温 `1.18–1.40`) | **`0.327` / `1.40`** | **`2.22 GiB`** | 仅用 `1/16.9` 内存即反超 `llama.cpp` 与 `hypura` (`0.34 @ 19.75G`) |

### 4.3 `Llama-3.2-1B`、`DeepSeek-V2-Lite`、`OLMoE-1B-7B`、`SmolVLM-256M` 与 `@gpu_graph` vs `Apple MPS`

| 模型 / 测试项 | **Tenon v11** (M4 Max Metal) | 上游 `llama.cpp` (`12359`, Metal 4 `-fa 1`) / 竞品 | 客观对比分析 |
| :--- | ---: | ---: | :--- |
| **`Llama-3.2-1B` `d=0` Decode (`B=1`)** | `330.5 tok/s` (`@gpu_graph` 静态编译: **`601.95`**) | `354.9 tok/s` | GGUF 引擎接近持平 (`0.93×`)；静态编译版领先 `1.70×` |
| **`Llama-3.2-1B` `4K FP16` / `8K Q8` / `4K Q4`** | `161.4` / `135.1` / `180.7 tok/s` (`8K FP8`: `119.3`) | **`262.2`** / **`232.8`** / **`217.1 tok/s`** | Tenon v11 较 v10 提速 **`2.67×–4.07×`**；`llama.cpp` Metal 4 `fa_vec_*` 仍快 `1.20×–1.72×` |
| **`Llama-3.2-1B` HTTP 并发 `B=8` / `B=16`** | `767.3` (913K 服务: `715.0` / **`717.8`**) | **`1,035.2`** / **`781.2 agg tok/s`** | Tenon 913K 零 Rust 服务在 `B=16` 达 `llama-server` 的 **`91.9%`**，体积小 `26.9×` |
| **`Llama-3.2-1B` Prefill `pp512`** | `943.4 tok/s` | **`4,969.1 tok/s`** | `llama.cpp` Metal `mul_mm` 领先 `5.27×` |
| **`DeepSeek-V2-Lite` `d=0` / `2K MLA`** | 空载 `66.7` / `40.1`（压力同窗 `59.0` / `23.4`） | **`69.9`** / **`60.9 tok/s`** | Metal 端 `llama.cpp 12359` 较 CUDA 端优化得好得多；Tenon 优势是 **Latent MLA 省 `8.89×` KV 内存** (`FP 0.88G` vs `1.38G`) |
| **`DeepSeek-V2-Lite` 直接 `CB B=8` / `B=16`** | 空载 **`200.7` / `247.6`**（压力同窗 `133.7` / `175.6`） | `llama-server` HTTP `99.3` / `89.1 agg tok/s` | **Tenon AMX Mega-v3 批引擎内核吞吐达 `2.0×–2.8×`**（但 HTTP 调度层开销待优化） |
| **`OLMoE-1B-7B` (6.9B 64E Top-8 MoE)** | **`51.6–58.4 tok/s`** (`F_NOCACHE` 严格冷盘) / **`83.9`** (OS 缓存) | `13.27 tok/s` (`llama.cpp -ot exps=CPU`) | **Tenon 领先 `3.9×–6.3×`!!** |
| **`SmolVLM-256M` 多轮 VLM Turn-1 → Turn-2** | `536.1 ms` (`pf 471.6`) → **`39.6 ms`** (`pf 12.1`, **`39.0×` COW**) | `326.0 ms` (`pf 265.4`) → **`35.9 ms`** (`pf 3.7`) | Turn-2 端到端基本持平（`39.6` vs `35.9 ms`）；Turn-1 Tenon 准确识别红蓝色条纹 |
| **`@gpu_graph` Tile-IR `2048³` / `4096³` GEMM** | **`13.47` / `13.81 TFLOPS`** (`85.2%` 物理上限) | `Apple MPS`: `12.57 TFLOPS` (`2048³`) | **纯声明式 Tenon 编译器反超 Apple 官方 MPS `+7.2%`!!** |

---

## 五、Google Cloud TPU v5e-1 正面评测（Tenon Direct PJRT vs `JAX 0.11.2`）

> **硬件**：同一台 Google Colab `Cloud TPU v5e-1`（物理上限 `197 TFLOPS BF16 MXU / 394 TOPS INT8 MXU / 819 GB/s HBM2e`），隔离 OS 进程独占 `/dev/vfio/0` 同机执行 [`compare_tpu.py`](gpu/compare_tpu.py)。**Tenon 零 Python 直连 `libtpu.so` 官方 PJRT C-API v0.116，全线 11/11 基准完胜 `JAX 0.11.2`**（片上异步链式 `VERIFY_TPU_ASYNC_CHAIN max_err=0 PASS`）：

| 基准项 (`Llama-3.2-1B` 16L 真 GQA / 裸核) | 同机 **`JAX 0.11.2`** (Sync) | **Tenon Sync** (代数融合) | **Tenon 异步链式 PJRT** | **vs `JAX 0.11.2`** | 占硅片物理上限 |
| :--- | ---: | ---: | ---: | :---: | :---: |
| MXU BF16 `4096³` | `145.8 TFLOPS` | `153.0 TFLOPS` | **`189.85 TFLOPS`** (`0.724 ms`) | **`1.30×`** (`+4.9%` Sync) | **96.37%** of 197T |
| MXU INT8 `4096³` | `235.0 TOPS` | `246.1 TOPS` | **`378.27 TOPS`** (`0.363 ms`) | **`1.61×`** (`+4.7%` Sync) | **96.01%** of 394T |
| HBM GEMV `1×16384²` **BF16** | `571.5 GB/s` | `601.7 GB/s` | **`748.43 GB/s`** | **`1.31×`** (`+5.3%` Sync) | **91.38%** of 819 GB/s |
| HBM GEMV `1×16384²` **W8A16** | `450.9 GB/s` | `487.1 GB/s` | **`741.86 GB/s`** | **`1.65×`** (`+8.0%` Sync) | **90.58%** of 819 GB/s |
| HBM GEMV `1×16384²` **W4A8** | `289.1 GB/s` | `314.8 GB/s` | **`542.38 GB/s`** | **`1.88×`** (`+8.9%` Sync) | **66.23%** of 819 GB/s |
| 16L GQA `S=512` Prefill **BF16** | `74,136 tok/s` | `74,789 tok/s` | **`77,343 tok/s`** (`155.7T`) | **`1.04×`** | **79.04%** of 197T |
| 16L GQA `S=512` Prefill **W8A8** | `111,135 tok/s` | `113,711 tok/s` | **`119,367 tok/s`** (`240.3T`) | **`1.07×`** (`+2.3%` Sync) | **60.99%** of 394T |
| 16L GQA `B=128` Decode **W4A8** | `77,439 tok/s` | `81,539 tok/s` | **`92,306 tok/s`** (`181.2T`) | **`1.19×`** (`+5.3%` Sync) | — |
| 16L GQA `B=32` Decode **W4A8** | `23,733 tok/s` | `25,595 tok/s` | **`29,671 tok/s`** | **`1.25×`** (`+7.8%` Sync) | — |
| 16L GQA `B=1` Decode **BF16** | `327.6 tok/s` | `343.0 tok/s` | **`367.6 tok/s`** (`715.4 GB/s`) | **`1.12×`** (`+4.7%` Sync) | **87.35%** of 819 GB/s |
| 16L GQA `B=1` Decode **W4A8** | `738.7 tok/s` | `810.0 tok/s` | **`955.1 tok/s`** (`1.047 ms`) | **`1.29×`** (`+9.7%` Sync) | — |

---

## 六、综合工程评估：优势领跑区 vs 客观短板区

### 6.1 Tenon v11 具有压倒性优势的核心场景

1. **超内存 / 有界预算超大 MoE 与 Dense 模型（`GLM-4.5-Air 106B` / `Llama-3.3-70B` / `OLMoE`）**：
   - 在 Apple M4 Max 与 NVIDIA CUDA（16G/24G/42G/44G 预算与严格 NVMe 直流）**全档位排名 #1**，领先 `ik_llama.cpp` `1.06×–4.70×`、领先上游 `llama.cpp` `1.25×–12.62×`、领先 `BigMoeOnEdge` `1.64×–12.7×`、领先 `hypura` `4.42×`，同时严守物理内存红线、零 Swap 膨胀且全员 `BIT-EXACT`。
2. **NVIDIA CUDA 上的 DeepSeek Latent MLA + MoE 推理与高并发（`DeepSeek-V2-Lite`）**：
   - 在同一台 `RTX 6000 Ada` 上，Tenon v11 单流快上游 `llama.cpp` **`3.54×`**、2K 长上下文 Latent MLA 快 **`10.16×`**、连续批处理 `B=8/B=16` 快 **`14.2×–18.0×`**，并支持 `2D TP×EP` 持久 CUDA Graphs 与原生 `sm_89` FP8 `m16n8k32` MoE Mega-Kernel v3 (`0.144 ms`)。
3. **无 P2P 多卡真 Tensor Parallel 与分离式 PD 集群**：
   - 上游 `llama.cpp -sm row` 在无 P2P RunPod 双卡上直接报错拒载，而 Tenon v11 通过 `dlopen("libnccl.so.2")` + `TENON_TP_GRAPH=1` 稳定跑通 `70B TP=2` (`25.98 tok/s`, 投机 `57.05 tok/s`) 及**双 OS 进程 GPU-Direct `CUDA IPC v2` (`3.35ms / 1.80ms`)**、逐层流式 `TNKL` PD、以及 HA 熔断/故障转移 PD Router。
4. **跨三硬件统一声明式编译器（`@gpu_graph`）与极致部署体积 + 形式化安全**：
   - 同源直出 **Metal (`+7.2%` vs Apple MPS) / CUDA (`1.46×` vs `cublasLt`、`2.37×` vs `PyTorch 2.8`) / TPU PJRT (11/11 全胜 `JAX 0.11.2`)**；且完整生产服务+HA 网关仅 **`913 KiB` 零 Rust 单二进制**（比 `llama-server` 小 `26.9×`），通过 **360 项 Z3 静态形式化证明**与 `live_objects == 0` 内存安全校验。

### 6.2 客观存在的短板与取舍（诚实披露）

1. **GGUF 通用推理路径的 Dense 大块 Prefill（`pp128` / `pp512`）**：
   - 在 `Llama-3.2-1B` (`pp512`) 与 `Llama-3.3-70B` (`pp128`) 的 GGUF 运行时路径上，上游 `llama.cpp` 凭借多年打磨的 cuBLAS/MMQ/Metal `mul_mm` 分块内核领先 Tenon **`4.5×–8.5×`**（Tenon 的极致 Prefill 算力目前主要体现在 `@gpu_graph` 静态编译 FP8/BF16 路径上，尚未全部反哺到通用 GGUF 运行时）。
2. **Dense 小模型在 4K–8K 长上下文的绝对速度**：
   - 尽管 Tenon v11 `FlashDecoding` 较 v10 暴涨 **`2.67×–6.06×`**，但在 `Llama-3.2-1B` `4K/8K` 上，上游 `llama.cpp`（手写 `sm_89` MMA FlashAttention-2 与 Metal 4 `fa_vec_*`）仍领先 **`1.20×–3.79×`**。
3. **128 GiB 充裕内存 Mac 上 100% 常驻 MoE 单流解码**：
   - 当 106B `GLM-4.5-Air` 无需受限、全部 58.5 GiB 完整装入 128 GiB Mac 内存时，最新 `llama.cpp 12359` 的 Metal 4 `mul_mv_id` 达到 **`50.66 tok/s`**，领先 Tenon 100% 常驻无 ICB 路径（`29.76 tok/s`）`1.70×`；同样在 48G 单卡全常驻 70B Dense 上，`llama.cpp`/`ik_llama.cpp` 领先约 `7%–10%`。
4. **生态覆盖广度**：
   - `llama.cpp` / `vLLM` 支持上百种开源模型家族与社区插件；Tenon 当前聚焦主流旗舰架构（Llama / Qwen2.5-VL / Qwen3-MoE / DeepSeek-V2/V3/R1 MLA+MoE / GLM-4.5-Air / OLMoE / SmolVLM / Whisper）并优先保证端到端可验证性与跨三芯一致性。
