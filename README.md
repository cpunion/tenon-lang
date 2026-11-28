# Tenon 语言设计与技术方案（已拍板定稿）

> **跨框架综合实测报告**: 详见 **[`benchmarks/TENON_V15_BENCHMARK_REPORT.md`](benchmarks/TENON_V15_BENCHMARK_REPORT.md)**（**Tenon v15 硅片 Roofline 终极版 + 2026 多卡前沿生态** —— 覆盖 **Apple M4 Max 9 模型** 与 **NVIDIA `2× RTX 6000 Ada / L40S` 7 模型 + 双卡 `TP=2`** 全量实测：**Apple M4 Max** 全部 5 个 0.8B–1B 全量化与前沿模型 `tg_b1` **100% 全面反超上游 `llama.cpp` 且 5/5 `exact=True`**（`1B Q4_0` **`378.3` `+7.5%`**、`Q4_K_M` **`343.9` `+7.5%` (较 v14 `+26.2%`)**、`Q8_0` **`267.9` `+8.0%`**、**`Gemma-3-1B` `273.8` `+6.6%` (较 v14 `+57.9%`) & `pp512 6,067.3` `+4.7%`**、**`Qwen3.5-0.8B` `336.4` `+14.9%` (较 v14 `+36.7%`)**），`1B 8K Q8 / 4K Q4` 双反超（`268.2 / 283.5`）、`DSV2 2K MLA` **`126.0` (`+19.5%`)**、**`995 KiB` 纯 Tenon 零 Rust v15 服务器（374 项 Z3 证明 + 136 项越界消除 + 7/7 HA E2E 全绿）HTTP `B=16` 创纪录 `1,914.8 tok/s`（`1.21×` `llama-server`）**；**NVIDIA `2× RTX 6000 Ada`** 4/5 模型 `tg_b1` 全面反超 `llama.cpp`（**`Qwen3.5-0.8B` `756.5` `1.53×`**、**`Gemma-3-1B` `546.8` `+15.6%` (较 v14 `+69.6%`!)**、`1B Q4_K_M` **`686.0` `+5.3%`**、`1B Q8_0` **`484.8` `+4.2%`**、`1B Q4_0` **`711.8` (`+21.9%`, `96.8%` 上限)**），`pp512` 全面碾压（`1B Q4_0` **`48.6K–56.1K` `1.25×–1.45×`**、`Q4_K_M` **`36.7K` `1.95×`**、`Q8_0` **`43.8K` `2.91×`**、`Gemma-3` **`32.0K` `2.30×`**、`Qwen3.5` **`25.0K` `1.73×`**），`CB B=16` 冲破 **`6,213 tok/s`（`4.25×` `llama-server`!!）**；**16B 对齐 `cm_gemm_q4_mmq_m128`** 令 `70B 1G pp128` 暴涨 **`+26.2%`** 至 **`582.1 tok/s`**（零溢出 `tok=14198`），无条件 PCIe `BF16` NCCL AllReduce 令 `70B TP=2 pp128` 暴涨 **`+71.7%`**（`2× L40S` `p2p=1` 达 `503.2` / `B=8 192.4`），`DSV2 2D TP=2×EP=2 pp256` 达 **`3,358.4` (`3.17×`)**；**256-Superblock 精确不等分 `Gemma-3-1B TP=2` 与 3-Slice `Qwen3.5-0.8B TP=2` 双双实机 `100% BIT-EXACT`**，双平台完整支持 `GDN+MoE`（`Qwen3.5-35B-A3B` / `Qwen3-Next`）与官方 17 字节 `MXFP4 (39)` / `gpt-oss`！历史基线见 [`TENON_V14_BENCHMARK_REPORT.md`](benchmarks/TENON_V14_BENCHMARK_REPORT.md)、[`TENON_V13_BENCHMARK_REPORT.md`](benchmarks/TENON_V13_BENCHMARK_REPORT.md)、[`TENON_V12_BENCHMARK_REPORT.md`](benchmarks/TENON_V12_BENCHMARK_REPORT.md) 与 [`TENON_V11_BENCHMARK_REPORT.md`](benchmarks/TENON_V11_BENCHMARK_REPORT.md)）。

> **Working Codename**: `Tenon`（榫卯）—— 寓意**强约束精化类型、分层形式化验证、无别名污染的内存模型、系统级版本演化与可验证分布式协议**严丝合缝。  
> **状态**: 核心架构已定稿（[ADR-0001](docs/adr/0001-architecture-decisions.md)、[ADR-0002](docs/adr/0002-interfaces-comptime-and-stdlib.md) & [ADR-0003](docs/adr/0003-declarative-gpu-and-unified-ui.md)），**极简正交语言核心（零新增硬关键字）→ 完整系统编程能力（S1/S2/S3）→ 声明式 CPU/GPU 融合（`@gpu_struct` / `@kernel` / `@gpu_graph` 真正 AST 级 MSL+WGSL 算子融合、`gpu_shared_f32` + `gpu_barrier()` 与 `E0842`/`E0843` 静态并发安全证明、`--emit-webgpu` 浏览器 WebGPU ESM 桥、零 Rust Metal/Vulkan HAL、Const-Generic 依赖维度 `Matrix[R, C]`）→ OpenUI Lang + 统一 React 风格 VDOM（真增量 Surgical VDOM Patcher + OpenAI/Ollama SSE 流式生成 + 双缓冲信号安全 TUI + 原生 Metal GPU `tenon-gpui` 动态 R8 中英文 Glyph Atlas / 缩放 / 滚轮 / 悬停 / 量化 K 线桌面工作台）→ 自动循环累积不变量推导 + `.tenon-cache` 持久化验证缓存 → Stage-3 纯 Tenon 零 Rust 自举固定点编译器（`sha256(stage3.ll) == sha256(stage4.ll)` & `sha256(stage3.o) == sha256(stage4.o)`）→ LSP 1.5 实时证明/GPU 融合 CodeLens 与反例一键转测试**（Rust 1.100.0 + Z3 + Cranelift Dev JIT + **LLVM 22 `-O3` AOT 独立原生二进制**）。
> **上游沉淀**: 综合 [`loom-lang`](../loom-lang)（984 PR 自举实践）、[`lang-research`](../lang-research)（2026-11 多语言可复现实验）与 [`dast-os-kernel`](../../dast.ai/dast-os-kernel/README.md)（227 轮分布式调和内核审计），并结合 2025–2026 全球最新编程语言与形式化验证进展形成的全新通用编程语言设计。

---

## 0. 快速体验可运行原型（Quick Start）

前置依赖：Rust `stable`（已验证 `rustc 1.100.0`）、`z3`、LLVM `clang`（已验证 LLVM 22.1.8；可选 `node` 验证编译出的 `.wasm`）。

```bash
# 0. 可视化交互体验与旗舰应用（零 `--c-src` 参数、零外部 Shader 文件，全量内置 `@c_src` 与 `gpu_embedded_msl()`！）
cargo build --release --bin tenon

# (A0) 依赖类型声明式数据流编译器 (`@gpu_graph("tensor_dataflow")`) + 三架构硬件 Roofline 逼近引擎（Apple Metal + NVIDIA CUDA + Google Cloud TPU！Tenon-Torch `std/nn`：AutogradTape + AMP `GradScaler` + 编译期 `Matrix[R, C, DType = Float16/BFloat16]` / 变长 `Tensor[...Shape]` + DLPack v1.0 / GGUF v3 `mmap` 零拷贝 / Safetensors）：
#   - 声明式自动融合 (`crates/tenon-codegen/src/tensor_graph.rs` + `std/gpu/tensor.tn`): 用户只写纯依赖类型矩阵数据流（支持 `Int4` / `FP8` / `Q4_0` / `Q8_0` 量化类型与 `tg_rope` + `tg_gqa_attention` + `tg_qmatmul`，零线程索引/零手写 SMEM/零后端代码），编译器自动做 Z3 维度/DType 证明、3-way 水平 QKV 与双路 FFN 融合 (`w_gate || w_up`) + 垂直片上 SRAM 融合 + 尾声 L2 原子 RMSNorm 融合 + 物理硅片 Roofline 上限估算 (`T_min`)，单源直出 **StableHLO MLIR (TPU W4A8/W8A8) / CUDA C++ (NVIDIA Q4_0_SB8 & FP8) / Metal MSL (Apple Q4_0_SB8 AMX)**！
#   - Apple M4 Max Metal (`std/gpu/metal/`, 硬件上限 `546 GB/s` 标称 / `468.4 GB/s` 实测裸 DRAM): 编译器自动 128-bit 对齐 `Q4_0_SB8` 超级块（与 GGUF v3 零拷贝 `mmap` AoS 比特一致 `sb8_vs_q4_diff=0.000488, mmap_max_diff=0`）+ Roofline 三级自动分派（`B<=4` 寄存器直通 `reg_sub16`、`B=5..16` `N=32×K=128` AMX+SplitK、`B=17..32` `32×32` AMX+SplitK + 单遍 `BatchRoPE+FlashGQA`）+ 尾声 L2 原子 RMSNorm & 融合 Argmax -> 裸核 `8192²` **479.5 GB/s (FP16, 87.8%) / 483.4 GB/s (Q8_0, 88.5%)**，`16384×8192` **469.1 GB/s (Q4_0_SB8, 85.9%)**，`32000×2048` Q4_0 **581.6 GB/s (106.5% SLC)**；1.24B Llama-3.2-1B 端到端 `B=1` Chained **FP16 205.3–213.3 tok/s (426.5–443.1 GB/s, 91.1%–94.6% 实测 DRAM 上限!)**、**Q8_0 370.5 tok/s (408.9 GB/s)**、**Q4_0 突破 601.95 tok/s (`1.661 ms`, 351.8 GB/s)** / **Q4_0_SB8 563.2 tok/s**；**多 Token 投机/批解码达 `B=4` 1,421.9 tok/s (`+58.7%`)、`B=8` 2,099.5 tok/s (`+26.6%`)、`B=16` 2,971.1 tok/s (`+20.0%`)、`B=32` 3,252.5 tok/s (7.18 TFLOPS)**！
#   - NVIDIA CUDA (`std/gpu/cuda/`, RunPod **RTX 4090 `sm_89`** 实测，硬件上限 `1,008.1 GB/s / 165.2T FP16-FP32 / 330.3T FP16-FP16 / 660.6T FP8-FP16`): 零链接参数 `dlopen("libcuda.so.1" + "libcublas.so" + "libcublasLt.so")` + 尾声 L2 原子 RMSNorm 融合 + **原生 Ada `sm_89` PTX 660.6T FP8 Tensor Core 内核**（`mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16` + 3-stage `cp.async` + 零 Bank 冲突 XOR Swizzle + `ldmatrix.x4` + SM 饱和 `64×64` Roofline 分派 + `float4` 向量化 `silu_mul_fp8`，`cos=0.999468 PASS`）+ 原生 **CUDA Graphs** & `B<=8` 寄存器直通 / `B=9..32` 双缓冲 `BN=32×BK=32` W4A16 MMA -> Tensor Core `4096³` **FP8 PTX 447.0–453.4 TFLOPS (`0.303–0.307 ms`, 67.7%–68.6% of 660.6T @ 450W TDP，反超 NVIDIA 官方 `cublasLt` 306.1T `1.46×–1.49×`!!)**；1.24B `S=512` Prefill **FP8 PTX 148,128 tok/s (`3.456 ms`, 288.3 TFLOPS，反超 `cublasLt` 135,676 `+9.2%`、2.05× Raw C-CUDA FP16 72,256、2.37× PyTorch 2.8 62,490!!)**；1.24B 尾声融合 Graph Decode `B=1` **FP16 357.6 tok/s @ 883.9 GB/s (87.7% GDDR6X，反超 Raw C-CUDA 346.4)**、**Q8_0 617.6–619.3 tok/s @ 811–813 GB/s (80.5%)**、**Q4_0 1,052.9–1,053.8 tok/s (`0.950 ms`, 732.0 GB/s)**；**Q4_0 投机/批解码 `B=4` 3,142.2 tok/s、`B=8` 3,879.2 tok/s、`B=16` 5,921.2 tok/s、`B=32` 9,925.6 tok/s (`9.43×` vs `B=1`)**！
#   - Google Cloud TPU (`std/gpu/tpu/`, Colab **v5e-1 TPU** 实测，硬件上限 `819 GB/s HBM2e / 197T BF16 / 394T INT8`): **零 Python** 直连 `dlopen("libtpu.so")` 官方 PJRT C-API v0.116 + 真实 Llama-3.2-1B **32H/8KV GQA** + **代数单遍融合 `RMSNorm+动态INT8量化`**（`r1` 代数消元并折入 1D 缩放）+ **常驻 `f32` 残差流** + **滑动窗口（depth=8）PJRT 异步命令队列流水线与片上 HBM 自回归链式执行**（消除 `0.21 ms` 主机 VFIO 气泡，`VERIFY_TPU_ASYNC_CHAIN max_err=0 PASS`）-> 硅片裸核 Roofline **MXU BF16 `4096³` 189.85 TFLOPS (96.37%!) / MXU INT8 `4096³` 378.27 TOPS (96.01%!) / HBM GEMV BF16 748.4 GB/s (91.4%) & W8A16 741.9 GB/s (90.6%) & W4A8 542.4 GB/s (66.2%)**；16 层真实 GQA 端到端片上链式 **`S=512` Prefill BF16 77,343 tok/s (155.7T, 79.0%) / `W8A8` 119,367 tok/s (240.3 TOPS, 4.289 ms)**、**`W4A8` Decode `B=128` 92,306 tok/s (1.387 ms, 181.2T) / `B=32` 29,671 tok/s (1.078 ms) / `B=1` 955.1 tok/s (1.047 ms)**、**`B=1` BF16 367.6 tok/s @ 715.4 GB/s (87.35% HBM2e)**；同机同步口径正面对比 **JAX 0.11.2** 全线胜出（`B=1 W4A8` 810.0 vs 738.7 `+9.7%`、`B=32` 25,595 vs 23,733 `+7.8%`、`B=128` 81,539 vs 77,439 `+5.3%`、`S=512 W8A8` 113,711 vs 111,135 `+2.3%`）（全平台 `live_objects == 0` 零泄漏！）
#   - 通用跨硬件六阶段 Tile-IR 编译器 (`crates/tenon-codegen/src/tile_ir/`，**全部 62 项通用优化默认开启 (`GpuOptConfig::default() == all()`)**，Metal/CUDA/StableHLO 全阶段零占位完整内核 + 100% 算子无关通用 Metal Plan 执行器): 覆盖 Stage 1–3（Tile-MMA + `A^T@B` 128-bit 硬件转置加载 / 通用融合与 AMX FlashAttention-2 / 全图 Layout 传播 / Stream-K / 静态 Arena+Single-CB / 精确 `SiluBwd`/`GeluBwd` 反向 Epilogue 融合 AD + SPMD / PGO 自动调优 / Warp 特化双缓冲 / E-Graph 矩阵链重写 / 动态 Ragged + 真压缩 BSR-32 持久工作窃取 / SuperBlock16 OCP MXFP4）、Stage 4（#1 2D 蛇形 Tile Swizzle、#2 水平权重打包 `Q||K||V`、#3 零拷贝 Buffer Donation、#4 硬件 ICB 图重放、#5 在线 Monoid Cross-Entropy、#6 AMP+动态 INT8 AMX、#7 符号动态形状三桶零重编译分派、#8 零-`im2col` 隐式 GEMM 3×3 Conv2d & `O(log L)` 并行前缀扫描、#9 双队列 `MTLEvent` 异步重叠）、**Stage 5 借鉴 `llm.c` 的 4 项默认开启通用设施**（#10 自动逻辑/物理维度对齐填充 `50257→50304` & `1003→1024`、#11 无状态 Philox 随机舍入无 FP32 主权重训练、#12 全图活性驱动流式缓存提示 CUDA `__ldcs`/`__stcs` + Metal 64B Burst + TPU `hbm_streaming`、#13 SPMD 自动 ZeRO-1 优化器分片 `ReduceScatter`+`1/D` AdamW+`AllGather` 与梯度分桶）与 **Stage 6 v11 通用异构超内存流式与分布式服务编译 Pass (`hetero_stream.rs`, 设施 #14–#51)**（#14–#46：硬显存预算规划器 + NUMA + 双流 DMA + HybridCpuInt16 + Marlin + Latent MLA + `TP=2..8` + `Q2_K`/`Q3_K`/`IQ4_XS` + 全域 PagedAttention COW/CB + `EP=2..8`/NCCL + 原生 MTP/Tree-Attention/槽内零迁移投机 + 递归 `json_schema`/GBNF + `mmproj`/AnyRes/MRoPE/Whisper Log-Mel/Embeddings/Rerank + Shared-X + `2D TP×EP` + Chunked Prefill + 零拷贝 LoRA `BGMV` + 原生 OCP `FP8_E4M3`/`MXFP4` + MoE Mega-Kernel v2 + 1× GPU ICB/CUDA Graphs + 4-Mode PagedKV + SWA/Sink + 零拷贝 PD Binary/SHM/IPC + Zero-Rust PD Router；**v11 新增 #47–#51 全芯片同步推进、零短板**：#47 **多卡 `TP=2..8` & `2D TP×EP` 持久设备端解码图（`TENON_TP_GRAPH=1` + NCCL 捕获）与 Apple Metal 无持有引用异步 ICB 流水线（`commandBufferWithUnretainedReferences` + `enqueue`）**、#48 **原生 Ada/Hopper `sm_89+` FP8 (`e4m3×e4m3`) `mma.sync.m16n8k32` 分组 MoE Mega-Kernel v3 & `half2` 向量化打包 + Apple Metal 寄存器直通 `half4` AMX MoE Mega-Kernel v3**、#49 **多卡 `TP=2..8` GPU-Direct `CUDA IPC` (`version=2` 跨进程多卡零主机拷贝) 分片 PD + 逐层流式流水线 PD（`TNKL` 头 + FNV-1a 校验 + `/v1/pd/layer`）**、#50 **硬件原生两阶段 `FlashDecoding` Split-KV 长上下文并行解码（`4K–128K` 四模式 `FP16/Q8/FP8/Q4` GQA + SWA/Sink + Latent MLA，兼容 ICB 与 CUDA Graphs）**、#51 **生产高可用与动态弹性治理（HA `PD Router` 健康检查/熔断器/自动故障转移/队列背压 `HTTP 503 + Retry-After` + 实时 SSE 断连取消与 `SO_NOSIGPIPE`/`MSG_NOSIGNAL` 保护 + 零停机动态 LoRA 热插拔 `/v1/lora/{load,unload,list}` + Prometheus v11 全指标）**）。Apple M4 Max (40-core, `16.2 TFLOPS / 468.4 GB/s` 实测上限) 纯声明式 `@gpu_graph` 22 项实机消融全绿：**`4096³` GEMM `13.81 TFLOPS` (85.2%) / `2048³` `13.47 TFLOPS` (83.1%, 超 Apple MPS `12.57T`)**、**6 算子融合 MLP `12.89 TFLOPS` (79.6%, `-71.4%` VRAM)**、**AMX FlashAttention-2 `S=1024` `5.29×` / `S=2048` `8.41T` (`51.9%`, `-100%` O(S²) 显存)**、**`B=1` Decode FP16 `468.9 GB/s` (100% DRAM) → SuperBlock16 MXFP4 `2.69–2.90×`**、**精确 AD 端到端 `11.78T` (`72.7%`)**、**BSR-32 块稀疏 `3.20×`**、**水平 `QKV` 打包 `1.13×` (`12.83T`)**、**Donation AdamW `1.09–1.11×` (`462–490 GB/s`)**、**硬件 ICB `1.13×`**、**在线 Monoid Cross-Entropy `3.33–3.42×` (`32 MB→0 KB`)**、**动态 INT8 AMX `1.12×` (`-50%` 权重 DRAM)**、**动态三桶分派 `32.04×` (`M=1`)**、**隐式 Conv2d `1.10×` (`2.3 MB→0 KB`)**、**并行 Scan `5.61–5.87×`**、**双队列异步重叠 `1.11–1.34×`**、**奇数维度自动对齐 `1003×2003` `1.46×` (`10.41T`) & GPT-2 `50257→50304` `1.41×` (`diff=2.27e-13`)**、**Philox 随机舍入无主权重 FP16 AdamW `3.24×` (`-50%` 显存，100 步亚 ULP 误差从 RTNE `0.081` 降至 `3.86e-5` 精度提升 2,097×!)**、**ZeRO-1 (`spmd=4`) `-75%` 优化器显存 + 本地 `5.90×` (`diff=0`)**！
#   - 硬内存封顶（`≤ 16 GiB` / `≤ 24 GiB` / `≤ 44 GiB`、`F_NOCACHE`/`POSIX_FADV_DONTNEED` 零 OS 页缓存污染）双平台（Apple Metal v11 + NVIDIA CUDA v11）多分片超内存动态加载释放、异构协同引擎与 **开箱即用 `tenon llm` v11 CLI / 生产 HA 分离式 PD 集群 / 零 Rust `913 KiB` 双平台原生服务器与网关** (`crates/tenon-cli/src/llm.rs` + `examples/showcase/tenon_llm_server.tn` **913 KiB 零 Rust 原生 Metal 服务器 + HA PD Router v11** & `examples/showcase/tenon_llm_server_cuda.tn` **CUDA 对等体**，内置 FlashDecoding Split-KV + 异步 ICB / 多卡 `TP` CUDA Graphs + 四模式 PagedKV `FP16/Q8/FP8/Q4` + SWA/Sink + 逐层流式 `TNKL`/SHM/Raw Binary PD + 生产 HA `TENON_PD_ROUTER=1`（健康检查/熔断器/自动故障转移/503 背压）+ 实时 SSE 断连取消 + 动态 `/v1/lora/{load,unload,list}` 热插拔 + Prometheus TTFT/TPOT 直方图与 v11 指标 + 优雅停机 `POST /shutdown`，**360 项形式化证明全绿!** + `examples/showcase/tenon_llm_cli.tn` 257 KB CLI + `std/gpu/metal/moe_stream.{tn,m}` & `std/gpu/cuda/moe_stream_cuda.{tn,c}`): 全面支持 **15 种 GGML 量化（含原生 Ada `sm_89` `FP8 m16n8k32` / Apple 寄存器直通 `half4` AMX 加速的 `FP8_E4M3` (`39`) & `MXFP4` (`40`) & `Q4_K`/`Q6_K`/`Q8_0`、`Q2_K`/`Q3_K`/`IQ4_XS`、`Q5_0`/`Q5_1`、`Q5_K`、`Q4_0`/`Q4_1`、`F16`/`F32`）**、多 Shard GGUF 自动发现、**`Llama-3` / `GLM-4.5-Air` / `OLMoE` / `Qwen3-MoE` / `DeepSeek-V2/V3/R1` (真矩阵吸收压缩 Latent MLA + YaRN `mscale²` + 完整 MLA+MoE MTP Block + MoE Mega-Kernel v3)**、**全域 PagedAttention 连续批处理（Dense `TP=1..8` + Latent MLA + MoE `EP=1` + `2D TP×EP`）+ 硬件两阶段 `FlashDecoding` Split-KV + 四模式 PagedKV (`--kv-quant fp16|q8|fp8|q4`) + 真 `O(W)` SWA & Attention Sink (`--sliding-window` / `--attn-sink`) + `128K` 上下文 + 槽内零迁移投机校验 (`paged_verify_batch`) + 多卡 `TP` GPU-Direct CUDA IPC (`version=2`) & 逐层流式 `TNKL` PD & 生产 HA `--pd-router`（`--max-queue` 背压/熔断/自动转移）**、**单次 `BGMV` 混合批异构多租户 LoRA + 零停机动态 `/v1/lora/{load,unload,list}` 热插拔**、**完整递归 `json_schema` + 原生 GBNF 下推自动机 + OpenAI `tools`/`tool_calls`**、**真实 `Qwen2.5-VL` 2×2 Spatial Merger ViT + `SmolVLM-256M` 多图 AnyRes + 自动 2D/3D MRoPE 网格 + PagedKV 块级多模态 COW 前缀缓存 + 真实 Whisper Transformer GGUF (`has_gguf=2`) 音频 (`--audio-mmproj` / OpenAI `input_audio`) + `/v1/embeddings` + `/v1/rerank` + Prometheus v11 全指标 `GET /metrics`**：
#     * **平台一：RunPod NVIDIA `2× RTX 6000 Ada 48GB` (`sm_89`, `NCCL=1`) & `2× A100 80GB PCIe` (`sm_80`) & RTX 4090 48GB + AMD EPYC 六引擎正面实机对决**（对比 **`ik_llama.cpp`** ★3.4k / **上游 `llama.cpp`** ★136k / **`BigMoeOnEdge v0.50.1`** ★1.15k / **`mistral.rs v0.9.5`** ★7.9k / **`KTransformers kt-kernel v0.7.0.post4`** ★19.7k，Tenon 全员 `BIT-EXACT`）：
#       - **Tenon v11 新特性 CUDA 实机验收（`2× RTX 6000 Ada 48GB`, `sm_89`, GCC `-Wall -Wextra` 零错误零警告, `NCCL=1`）**：
#         • **硬件原生两阶段 `FlashDecoding` Split-KV 长上下文解码（8 Splits + CUDA Graphs，`BIT-EXACT`）**：`Llama-3.2-1B` **`4K FP16` `46.2 → 219.0 tok/s (4.74×, 23/23 BIT-EXACT)`**、**`8K FP8_E4M3` `20.7 → 125.5 tok/s (6.06×, 23/23 BIT-EXACT)`**、**`4K Q4_0` `61.7 → 259.4 tok/s (4.20×, 23/23 BIT-EXACT)`**（`fd_launches=352, g_cap=1, g_rep=22`）；**`DeepSeek-V2-Lite` `2K` Latent MLA FlashDecoding `13.4 → 98.0 tok/s (7.34×!!, 19/19 BIT-EXACT, fd_launches=486)`**！
#         • **原生 Ada `sm_89` FP8 (`e4m3×e4m3`) `mma.sync.m16n8k32` 分组 MoE Mega-Kernel v3 (`E=8, B=32, K=6, 2048→1408→2048`)**：Mega-v1 Scalar `1.315 ms` → Mega-v2 FP16-TC (`m16n8k16`) `0.444 ms (2.96×)` → **Mega-v3 Native FP8-TC (`m16n8k32`) `0.144 ms`（较标量提速 `9.10×`、较 FP16-TC 再暴涨 `3.07×`!! `cos_v1=0.999294, cos_fp16=0.999294, fp8_tc_tiles=176`）**！
#         • **多卡 `TP=2` & `2D TP=2 × EP=2` 持久设备端 CUDA Graphs (`TENON_TP_GRAPH=1` + NCCL 捕获, `p2p=0, nccl=1`)**：`DeepSeek-V2-Lite 2D TP=2×EP=2` **`B=1` `119.0 → 176.0 tok/s (+47.9% / 1.48×!)`**、`B=16` `617.4 → 630.2 agg tok/s`（`tp_cap=3, tp_rep=35, 13/13 BIT-EXACT vs EP=1`）；`Llama-3.3-70B TP=2 (Q4_0 KV)` `B=1: 25.98 / B=4: 45.71 / B=8: 66.51 agg tok/s`（`tp_cap=3, tp_rep=35`, `13/13` & `52/52` & `104/104 BIT-EXACT`）！
#         • **多卡 `70B TP=2` 真实跨进程 GPU-Direct `CUDA IPC` (`version=2`) 分片 PD + 逐层流式 `TNKL` PD**：**双 OS 进程跨双卡 RTX 6000 Ada 零主机内存拷贝 `cuIpcGetMemHandle`/`cuIpcOpenMemHandle` (`meta=624B`) 仅 `export=3.35 ms, import=1.80 ms`（较 v10 POSIX SHM `12.08/8.13 ms` 提速 `3.6×` / `4.5×`!! `12/12 BIT-EXACT`）**；逐层流式 `TNKL` PD：`1B (16L, 4×4L)` `512.2 KiB / 3.79 ms`、`DeepSeek-V2-Lite MLA (27L, 3×9L)` `486.2 KiB / 3.84 ms`、**`70B TP=2 (80L, 5×16L Q4_0)` `1,440.3 KiB / 13.51 ms`（全员篡改拒收 + `BIT-EXACT`）**！
#       - **Tenon v10 实机基线（`2× RTX 6000 Ada 48GB`, `sm_89`）**：
#         • **自适应 Tensor Core `mma.sync.aligned.m16n8k16` 分组 MoE Mega-Kernel v2 + CUDA Graphs (`DeepSeek-V2-Lite-Chat.Q4_K_M`, `tc_tiles=17,966`)**：**`EP=1`** `PagedCB B=4: 318.1 agg tok/s (2.04×)` → **`B=8: 657.9 agg tok/s (4.22×, 较 v9 384.44 暴涨 +71.1%!!, 13/13 BIT-EXACT)`** → **`B=16: 872.2 agg tok/s (5.59×, 13/13 BIT-EXACT)`**（`CUDA Graphs cap=4, rep=60`，SHM MLA PD `972.1 KiB`, `exp=2.74ms, imp=2.07ms`, `13/13 BIT-EXACT`）；**`2D TP=2 × EP=2`** **`PagedPrefill-32` 达 `663.6 tok/s` (`48.2 ms`, 较 v9 `553.4` 再提速 `+19.9%`!)**、`PagedCB B=8: 443.2 agg tok/s (3.66×) → B=16: 796.2 agg tok/s (6.58×, 13/13 BIT-EXACT)`；跨拓扑 `EP=1 vs 2D TP=2×EP=2` **`13/13 BIT-EXACT`**！
#         • **硬件 `CUDA Graphs` 持久解码 + 四模式 PagedKV 压缩 + SWA/Sink + 单卡跨进程 `CUDA IPC` PD (`Llama-3.2-1B`)**：`CUDA_GRAPH=0→1` `B=1` **`505.8 → 569.1 tok/s (+12.5% / 1.13×)`**、`B=8` **`1,839.8 agg tok/s`**（`33/33 B=1 & 27/27 B=8 BIT-EXACT`）；四模式 PagedKV：`FP16` `512.1 KiB` / `Q8_0` `272.1 KiB (1.88×, cos=0.999977)` / **硬件 `sm_89` `FP8_E4M3` `272.1 KiB (1.88×, cos=0.999833)`** / **`Q4_0` `144.1 KiB (3.55×, cos=0.993181)`**；单卡双进程 `CUDA IPC` `exp=0.82ms, imp=0.66ms, 11/11 BIT-EXACT`！
#         • **70B Dense `Llama-3.3-70B-Instruct-Q4_0` `2× TP=2` 分片 `Q4_0` PagedKV + SHM PD + PagedCB + 槽内零迁移投机 (`2× RTX 6000 Ada`)**：`TP=2` SHM `Q4_0` PD `2.81 MiB` (`3.55×` 压缩, `12/12 BIT-EXACT`)；`TP=2 Q4_0` PagedCB `B=1: 25.95 → B=4: 64.59 (2.49×) → B=8: 80.53 agg tok/s (3.10×, 96/96 BIT-EXACT)`；**`TP=2` 槽内零迁移 `paged_verify_batch` + `1B` Draft 代码生成 `K=5` 达 `57.05 tok/s` (`2.32×`, `83.3%` 接受率, `64/64 BIT-EXACT`)**、Marlin Prefill-128 **`286.63–300.36 tok/s`**（`2× A100 80GB` `TP=2 + FAST_DP4A=1` Decode `41.31 tok/s`）！
#         • **原生 Ada `sm_89` 真 FP8 `m16n8k32` + 混合批异构 LoRA `BGMV` + 真实多模态 (`v9–v10`)**：**`FP8_E4M3` 原生 `m16n8k32` `12.83×`**、`Q4_K` `8.65×`、`MXFP4` `8.28×`、`Q6_K` `5.23×`；单 Launch 异构 LoRA `BGMV` `4/4 BIT-EXACT`；真实 `Qwen2.5-VL` 2×2 Spatial Merger ViT → PagedKV COW Fork；真实 Whisper Transformer GGUF (`has_gguf=2`) `8.2ms`；Shared-X PagedCB `B=8: 1,906.0 agg tok/s`！
#       - **106B Sparse MoE `GLM-4.5-Air-Q4_0` (58.47 GiB, 2 Shards, 128E Top-8 — 超出单卡 48 GiB 显存！)**：
#         • **`≤ 16 GiB` VRAM 档**：Tenon `24/128` (`13.86 GiB` VRAM, 161 KB 二进制) 达 **`35.76 tok/s`** (`BIT-EXACT`) — 用不到 14 GiB 显存已反超 `ik_llama.cpp` 24G 档 (`28.13`) 与 `BigMoeOnEdge` 48G 档 (`30.85`)！
#         • **`≤ 24 GiB` VRAM 档（单卡消费级 4090 24G 预算）**：Tenon `40/128` (`20.40 GiB` VRAM) 达 **`40.15 tok/s`** (`BIT-EXACT`, 较 Serial `6.02` 提速 **`6.67×`**) — 全面领先 `ik_llama.cpp -ncmoe 32 -fmoe 1` (`28.13 tok/s`, **`1.43×`**)、上游 `llama.cpp -ot` (`22.11 tok/s`, **`1.82×`**)、`BigMoeOnEdge v0.50.1` RAM 16G 缓存 (`9.69 tok/s`, **`4.14×`**; `--device-cache-cpu-share 1.0` `6.39 tok/s`, **`6.28×`**)！
#         • **`≤ 44 GiB` VRAM 档**：Tenon `88/128` (`40.05 GiB` VRAM) 达 **`50.48 tok/s`** (`BIT-EXACT`) — 领先 `ik_llama.cpp -ncmoe 14 -fmoe 1` (`47.45 tok/s` @ ~42.5G VRAM, **`1.06×`** 且省 2.5G 显存)、上游 `llama.cpp -ot` (`40.38 tok/s`, **`1.25×`**)、`BigMoeOnEdge --auto` (`30.85 tok/s` @ 46.7G VRAM / 28.5s 锁页加载, **`1.64×`**)！
#         • **严格 NVMe 直流档（主机内存 < 模型体积，`use_nocache=1`）**：Tenon `40/128` (`20.40 GiB` VRAM) 达 **`3.23–3.40 tok/s`** (`BIT-EXACT`)，反超 `BigMoeOnEdge --moe-stream` NVMe 直流 (`1.73 tok/s`) **`1.87–1.97×`**！
#         • **`KTransformers kt-kernel v0.7.0.post4`**：其 GGUF AVX2 `LlamafileMoEWrapper` 在加载 `GLM-4.5-Air-Q4_0` 时硬拒绝 `intermediate_size (1408) % QK_K (256) != 0`（官方 `Glm4Moe` 方案需 220–440 GiB DRAM 跑 BF16/AMX-INT8 约 11–16 tok/s）。
#       - **16B 生产级 MLA+MoE `DeepSeek-V2-Lite-Chat.Q4_K_M` (`2× RTX 6000 Ada 48GB`)**：`0.48s` 加载，4K 上下文下核心 VRAM 从 `2.04 GiB → 0.89 GiB`（净省 `1,174.5 MiB`，`8.89×` 压缩，`8/8 BIT-EXACT`），Tensor Core Mega-Kernel v2 Prefill **`477–491`** (`EP=1`) / **`663.6 tok/s`** (`2D TP=2×EP=2`)，Decode `B=8` **`657.9`** → `B=16` **`872.2 agg tok/s`** (`EP=1`) / **`796.2 agg tok/s`** (`2D TP=2×EP=2`)，2K Latent MLA FlashDecoding **`98.0 tok/s (7.34×)`**！
#       - **70B Dense `Llama-3.3-70B-Instruct-Q4_0` (`RTX 4090 / 2× RTX 6000 Ada 48GB`)**：单卡 4090 48GB FP32 `21.35 tok/s` (`854 GB/s`)；**单卡 24G 档超显存** NUMA `Q4_0_4R` + 8-Token AVX2 INT16 Decode 达 **`3.97–4.41 tok/s`**（超 `llama.cpp -ngl 48` `3.14` **`+26.5%`**），批 Prefill-32 达 **`11.85–16.49 tok/s` (`2.28×–3.17×`)**！
#     * **平台二：Apple M4 Max Metal v11 (`468.4 GB/s` 裸 DRAM) 六引擎正面实机对决 + 生产 HA 3 节点 PD 集群端到端实测**（对比 `BigMoeOnEdge v0.50.1` / `mistral.rs v0.9.5` / `ik_llama.cpp` / `t8/hypura` / `llama.cpp`，全档位领跑、全员 `BIT-EXACT` 且零 Swap 增量）：
#       - **Tenon v11 Apple M4 Max 突破性实机验收**：
#         • **硬件两阶段 `FlashDecoding` Split-KV + 无持有引用异步 Metal ICB 流水线（`BIT-EXACT`）**：`Llama-3.2-1B` **`4K FP16` `42.2 → 171.5 tok/s (4.07×!!, 23/23 BIT-EXACT)`**、**`8K FP8_E4M3` `26.7 → 80.4 tok/s (3.01×!!, 23/23 BIT-EXACT)`**、**`4K Q4_0` `48.3 → 129.0 tok/s (2.67×!!, 23/23 BIT-EXACT)`**（`fd_launches=352, icb_overlaps=22`）；**`DeepSeek-V2-Lite` `2K` Latent MLA FlashDecoding `9.2 → 40.1 tok/s (4.34×!!, 15/15 BIT-EXACT, fd_launches=378)`**！
#         • **寄存器直通 `half4` AMX MoE Mega-Kernel v3 + 逐层流式 `TNKL` PD (`DeepSeek-V2-Lite` & `Llama-3.2-1B`)**：`DeepSeek-V2-Lite` `PagedCB B=8: 200.68 agg tok/s (amx_v3_tiles=7,124, icb_overlaps=24, 13/13 BIT-EXACT)` → **`B=16: 247.62 agg tok/s (amx_v3_tiles=16,276, icb_overlaps=36, 13/13 BIT-EXACT)`**；逐层流式 `TNKL` PD：`1B (16L, 4×4L)` `272.2 KiB / 0.93 ms`、`DeepSeek-V2-Lite MLA (27L, 3×9L)` `972.2 KiB / 2.52 ms`（篡改拒收 + `13/13 BIT-EXACT`）！
#         • **生产高可用与动态弹性端到端验收（`tenon llm` CLI v11 与纯 Tenon 零 Rust `913 KiB` `tenon_llm_server.tn` v11 双端同步全绿，360 项形式化证明全绿！）**：**HA `PD Router`（主节点宕机自动熔断转移 `failovers=2`）下 `Layer-Stream (TNKL)` + `Binary` + `SHM` 三通道 100% `BIT-EXACT`**、篡改 `TNKL` 块拒收 `HTTP 400`；**实时 SSE 客户端断连（`RST` 中断）零 `SIGPIPE` 崩溃且瞬间回收全部 PagedKV 块（`stream_cancelled=1, kv_free 128→128`）**；**零停机动态 LoRA 热插拔（`POST /v1/lora/load` → 推理 → `POST /v1/lora/unload`, `lora_hot_swaps=2`）**；**队列背压（`max_queue=1`, 4 并发 -> `200=1, 503=3, Retry-After: 1`）**；Prometheus `/metrics` 全部 7 项 v11 指标全绿！
#       - **Tenon v10 实机基线（Apple M4 Max）**：
#         • **自适应 Apple FP16-AMX MoE Mega-Kernel v2 + Metal ICB (`DeepSeek-V2-Lite`)**：`PagedPrefill-32` **`153.3 ms (208.8 tok/s, +24.4%)`**；`PagedCB B=4: 134.26 → B=8: 199.87 → B=16: 248.55 agg tok/s`；SHM MLA PD `972.0 KiB (exp=1.20ms, imp=1.14ms, 11/11 BIT-EXACT)`！
#         • **四模式 PagedKV 压缩 + POSIX SHM PD + SWA/Sink + 真实开源多模态 (`SmolVLM-256M`)**：Metal ICB `B=1: 334.2 / B=8: 868.5 agg tok/s`；`FP16/Q8_0/FP8_E4M3/Q4_0` 全模式 `11/11 BIT-EXACT`；**`SmolVLM-256M` Turn-1 `398.3 ms` → Turn-2 PagedKV 块级多模态 COW 命中 `12.0 ms`（`33.28×` TTFT 加速!! `mm_kv_cow_hits=1`）**；FP16-AMX `B=32` `Q6_K 2.82× / MXFP4 1.98× / FP8 1.80× / Q8_0 1.76×`；70B + 1B Draft 代码生成 `K=5` `16.51–20.43 tok/s (1.85–2.59×)`！
#       - **106B Sparse MoE `GLM-4.5-Air-Q4_0` (58.47 GiB, 2 Shards)**：`≤ 42 GiB` 档 Tenon `80/128` (`37.14 GiB`) 达 **`10.29 tok/s`** / `64/128` (`30.59 GiB`) **`8.56 tok/s`**（反超 `ik_llama.cpp` 最优 `3.63` **`2.83×`**、`llama.cpp` 最优 `3.17` **`3.25×`**、`BigMoeOnEdge` Metal 最优 `2.68–2.86` / `--auto` `0.81–1.04` **`3.60–12.7×`** 且省 `2.0–4.7×` RSS）；`≤ 16 GiB` 档 Tenon `24/128` (`14.21 GiB`) 达 **`5.75 tok/s`** / 极限 `8/128` (`7.66 GiB`) **`5.13 tok/s`**（反超 `ik_llama.cpp` `1.58×`、`llama.cpp` `1.81×`、`BigMoeOnEdge` `2.01×`、`hypura` `1.30` `4.42×`）！
#       - **70B Dense `Llama-3.3-70B-Instruct-Q4_0` (37.36 GiB)**：满载 100% GPU Tenon 单 CB 零拷贝 `mmap` 达 **`11.03–11.22 tok/s`** (`442.4 GB/s` = **`94.4%` M4 Max 裸 DRAM 上限**)，反超 `mistral.rs v0.9.5` (`7.40`, **`1.49×`** 且省 `2.03×` RSS)、`llama.cpp -ngl 99` (`6.72`, **`1.64×`**) 与 `ik_llama.cpp -ngl 99` (`2.14`, **`5.15×`**)；AMX `B=32` Prefill **`63.88 tok/s`**；Radix 前缀树多轮对话 **`26.57×` TTFT 加速** (`29.69s → 1.12s`)；超内存流式 `≤ 32G` `[D3-RING30]` **`4.55 tok/s`**（`3.99×` `ik_llama` / `12.30×` `llama.cpp`），`≤ 16G` `[D2-RING16]` **`1.66–2.08 tok/s`**；OLMoE-1B-7B `F_NOCACHE` SSD 流式 **`51.6–58.4 tok/s`**（OS 缓存 `83.9 tok/s`）！
./target/release/tenon llm /path/to/model.gguf --serve 127.0.0.1:8080 --kv-quant fp8 --sliding-window 4096 --attn-sink 4 --max-queue 64 --prefill-chunk 128 --lora coder=/path/to/lora.gguf:1.0 --mmproj /path/to/mmproj.gguf  # 一键启动 v11 全域 CB + FlashDecoding Split-KV + 异步 ICB/CUDA Graphs + 四模式 KV + 逐层流式 PD + 动态 LoRA 热插拔 (`/v1/lora/{load,unload,list}`) + 实时 SSE 断连取消 + 503 背压 + Prometheus v11 指标
./target/release/tenon llm --pd-router --serve 127.0.0.1:8080 --prefill-workers 127.0.0.1:8081,127.0.0.1:8083 --decode-workers 127.0.0.1:8082,127.0.0.1:8084 --pd-transport layer --max-queue 64   # 一键启动生产 HA 分离式 PD 网关（健康检查/熔断器/自动故障转移/503 背压，支持 `--pd-transport layer|shm|binary`）
./target/release/tenon llm /path/to/70b.gguf --draft /path/to/1b.gguf --speculative draft --spec-k 5 --kv-quant q4 --tp 2 --ep 2                            # 命令行推理（支持 `--kv-quant fp16|q8|fp8|q4` / `--sliding-window` / `--attn-sink` / `--draft` / `--speculative` / `--lora` / `--mmproj` / `--audio` / 2D `--tp --ep`）
./target/release/tenon build examples/showcase/tenon_llm_server.tn -o /tmp/tenon_llm_server                                                                  # 零 Rust 913 KiB 独立原生 Metal 服务器 + 生产 HA PD Router v11（`TENON_PD_ROUTER=1` / `TENON_PD_TRANSPORT=layer|shm|binary` / `TENON_MAX_QUEUE=64` / FlashDecoding / Metal ICB / 动态 LoRA 热插拔 / SSE 断连取消 / Prometheus v11；CUDA 对等版：`examples/showcase/tenon_llm_server_cuda.tn`）
./target/release/tenon build examples/showcase/tenon_llm_cli.tn -o /tmp/tenon_llm_native                                     # 零 Rust 257 KB 独立原生 Metal LLM 命令行工具（含 AutoPlan / AMX Prefill / Q8_0 KV / BPE / 完整采样器）
./target/release/tenon build examples/showcase/tenon_compiler_ablation_bench.tn -o /tmp/tn_ablation && /tmp/tn_ablation  # 通用 Tile-IR Stage 1–3 (11 Pass) 实机消融与数值验收基准
./target/release/tenon build examples/showcase/tenon_stage4_facilities_bench.tn -o /tmp/tn_stage4 && /tmp/tn_stage4      # 通用 Tile-IR Stage 4 (9 大语言/设施级 Pass) 实机消融与数值验收基准
./target/release/tenon build examples/showcase/tenon_stage5_llmc_bench.tn -o /tmp/tn_stage5 && /tmp/tn_stage5            # 通用 Tile-IR Stage 5 (`llm.c` 4 大默认开启通用 Pass) 实机消融与数值验收基准
./target/release/tenon build examples/showcase/tenon_ondemand_moe_bench.tn -o /tmp/tn_stream && /tmp/tn_stream           # 超内存硬封顶(<=16 GiB) 106B MoE + 70B Dense 流式动态加载释放实机基准
./target/release/tenon test std/nn/nn_test.tn
./target/release/tenon check examples/showcase/bad_nn_shape.tn   # 负例：Matrix[4,8] @ Matrix[6,2] 编译期秒抓维度错误！
./target/release/tenon build examples/showcase/tenon_ultra_llm_bench.tn -o /tmp/tn_ultra_bench && /tmp/tn_ultra_bench  # Apple Silicon Metal 1.10B Llama-3.2-1B + Qwen2/Gemma-2/Mixtral-MoE
./target/release/tenon build examples/showcase/tenon_cuda_llm_bench.tn --emit-cuda /tmp/out.cu -o /tmp/tn_cuda_bench && /tmp/tn_cuda_bench  # NVIDIA CUDA (__dp4a + dynamic cuBLAS + CUDA Graphs; 同机对比 benchmarks/gpu/compare_cuda.py)
./target/release/tenon build examples/showcase/tenon_tpu_llm_bench.tn --emit-stablehlo /tmp/out.mlir -o /tmp/tn_tpu_bench && /tmp/tn_tpu_bench  # Google Cloud TPU (Declarative @gpu_graph + Direct PJRT + Full-16L while; 同机对比 benchmarks/gpu/compare_tpu.py)
./target/release/tenon build examples/showcase/tenon_transformer_mps.tn -o /tmp/tenon_mps_tf && /tmp/tenon_mps_tf
./target/release/tenon build examples/showcase/tenon_torch_studio.tn -o /tmp/tenon_torch_studio && /tmp/tenon_torch_studio
# 或无头截图预览（生成 /tmp/tenon_torch_studio.png）：TENON_TORCH_SMOKE=1 /tmp/tenon_torch_studio

# (A0b) 超越 PyO3 的一键全语言原生互操作 (`--export all`) + 类 Mojo 无缝调用并动态继承 Python/ObjC 对象 + 21 目标声明式交叉编译 (`tenon targets`) + Clang AST JSON `tenon bindgen` + 本地包注册表 `tenon pkg publish/search`
#   - Python 3 PEP 590 `METH_FASTCALL`（实测 64.7 ns/call = 15.45M/s！）+ PEP 3118 Buffer Protocol 零拷/批量 memcpy（100 万整数 `array('q')` 0.23ms，提速 8.1×！）+ `PyObject_Vectorcall`
#   - Node.js N-API `Float64Array`/`BigInt64Array`（实测 16.6 ns/call = 60.1M/s，100 万元素 `BigInt64Array` 0.41ms 提速 61.2× / `Float64Array` 往返 1.18ms 提速 25.8×！）
./target/release/tenon build examples/showcase/polyglot_quant_lib.tn --export all --export-dir target/export_quant   # 一键产出 Python .so/.pyi + Node.js 零 node-gyp .node/.d.ts + Swift 6 + Java/Kotlin JNI + C++17 + Rust + Zig + Go（全程 Perceus live_objects=0！）
./target/release/tenon build examples/showcase/python_and_objc_interop.tn -o /tmp/tn_py_objc && /tmp/tn_py_objc     # 零链接参数动态加载 CPython 3 + 零拷贝 memoryview + 动态继承 Python 类与 Apple NSObject 注入原生 LLVM Tenon 方法！
./target/release/tenon targets                                                                                      # 列出 21 个内置目标（macOS/Linux/Windows/iOS/Android/Wasm32/Cortex-M0/M4/M7/RP2040/STM32F4/ESP32-C3/AVR，支持 llgo 风格 JSON inherits！）
./target/release/tenon build examples/showcase/embedded_sensor_pid.tn --target esp32c3 --lib static -o /tmp/libpid_esp32c3.a
./target/release/tenon bindgen "$(xcrun --show-sdk-path)/usr/include/zlib.h" --lib z -o /tmp/zlib_bindings.tn       # Clang AST JSON 驱动自动绑定

# (A0c) P-Core SPMC 批量窃取 M:N 抢占式协程调度器（10K 任务 2.12ms 超越 Go、零 cgo 1.74ns C FFI 快 Go 9×+）+ Biased Non-Atomic Perceus RC（线程本地对象 0 原子指令 + 1-cycle `__thread` TLS 缓存，160 万并发堆分配低至 1.73ms 且 live_objects=0）+ `!llvm.loop` 自动向量化 + `--pgo-gen`/`--pgo-use` PGO 优化 + 多后端 GPU（MSL/WGSL/CUDA/SPIR-V GLSL）+ 完整 DWARF v4 调试信息
./target/release/tenon build examples/dogfood/mn_preempt_sched_demo.tn -o /tmp/tn_mn_bench && /tmp/tn_mn_bench
./target/release/tenon build examples/showcase/tenon_transformer_mps.tn --emit-cuda /tmp/out.cu --emit-spirv-glsl /tmp/out.comp --debug -o /tmp/tn_dbg

# (A1) 原生 macOS Apple Silicon Metal 量化 K 线 + 100 组 Metal GPU 并行参数网格热力图（点击热力图格子实时切换 MA 策略！）
./target/release/tenon build examples/showcase/quant_kline_gpui.tn -o /tmp/tenon_quant_kline && /tmp/tenon_quant_kline
# 或无头截图预览（生成 /tmp/tenon_quant_kline.png）：TENON_KLINE_SMOKE=1 /tmp/tenon_quant_kline

# (A2) 原生 macOS Apple Silicon Metal GPUI Studio（动态 R8 中英文 Glyph Atlas + 可缩放窗口 + 滚轮/悬停 + AI Prompt 输入框流式生成 OpenUI）
./target/release/tenon build examples/showcase/metal_gui_app.tn -o /tmp/tenon_metal_gui && /tmp/tenon_metal_gui
# 或无头截图预览（生成 /tmp/tenon_gpui_preview.png）：TENON_GUI_SMOKE=1 /tmp/tenon_metal_gui

# (B) 真实浏览器交互式 OpenUI Studio & 零安装 Web Playground（Wasm32 真增量 Surgical VDOM + OpenAI/Ollama 真 SSE + Tenon-Torch/WGSL 预览）
./examples/showcase/web_openui_studio/launch.sh
./examples/showcase/web_playground/launch.sh

# (C) 全屏交互式 ANSI 终端 TUI（OpenUI -> VDOM -> 双缓冲极简 ANSI Diff + 信号安全光标自动恢复；无需手动 `--c-src`！）
./target/release/tenon build examples/showcase/interactive_tui.tn -o /tmp/tenon_tui && /tmp/tenon_tui

# (D) 统一三端 AI Generative UI 客户端（可选 TENON_LLM_BASE_URL 对接 OpenAI/Ollama 真 SSE，一份 OpenUI 同时驱动 Web + TUI + Metal SceneGraph）
./target/release/tenon build examples/showcase/ai_openui_client.tn -o /tmp/ai_openui_client && /tmp/ai_openui_client

# (E) Tenon 编程体验导览 + 证明性能画像 `--profile-proofs`（跨函数摘要自动推导 + 零手写 invariant + 磁盘缓存 177x 加速）
./target/release/tenon check examples/showcase/tenon_torch_studio.tn --profile-proofs
./target/release/tenon check examples/showcase/language_tour.tn && ./target/release/tenon run examples/showcase/language_tour.tn

# (F) 零胶水声明式 GPU 分块矩阵乘、混合精度 FP16 (`Matrix[R, C, DType = Float16]` + 变长维度 `Tensor[1, 1, 2, 3, 4]`) & 256×256 Apple Silicon Metal GPU 加速基准（无需 `--emit-msl` / `--c-src`！）
./target/release/tenon build examples/dogfood/metal_fp16_tensor_demo.tn -o /tmp/metal_fp16 && /tmp/metal_fp16
./target/release/tenon build examples/dogfood/metal_tiled_matmul_demo.tn -o /tmp/metal_tiled && /tmp/metal_tiled
./target/release/tenon build benchmarks/tenon/gpu_matmul_bench.tn -o /tmp/gpu_matmul_bench && /tmp/gpu_matmul_bench

# (G) Stage-3/4 纯 Tenon 零 Rust 自举固定点证明（含 type/record/enum/match/assert，验证 sha256(stage3) == sha256(stage4) 比特一致！）
./examples/bootstrap/stage3/verify_fixed_point.sh

# (H) 脚手架模板一键创建项目（支持 cli / lib / web / gpui / openui-web / tui / gpu）
./target/release/tenon new /tmp/my_gpu_app --template gpu

# 1. 全工作区自动化测试（81 项全量回归；macOS 默认通过 real clang 原生链接校验 + LLVM 22 `lli` 并行执行，全量约 3 秒全绿；设置 `TENON_TEST_NATIVE_EXEC=1` 可切回逐二进制直接执行）
cargo test --workspace

# 2. 精化类型别名复用 + 高阶形式化验证闭环 + 独立证明导出与审计 + 可执行反例转测试
cargo run --bin tenon -- check examples/refinements/date_and_odd.tn
cargo run --bin tenon -- run   examples/refinements/date_and_odd.tn
cargo run --bin tenon -- check examples/refinements/bad_date.tn --emit-repro-test target/repro_test.tn
cargo run --bin tenon -- check examples/refinements/advanced_verification.tn --emit-proofs target/proof_certs
cargo run --bin tenon -- verify-proofs target/proof_certs --solver auto
cargo run --bin tenon -- build examples/refinements/advanced_verification.tn -o target/adv_ver_bin --opt O3 --checked-overflow --emit-mir target/adv.mir && ./target/adv_ver_bin

# 2b. 生产级 Dogfooding、声明式 GPU（Apple Silicon Metal）、OpenUI/VDOM/TUI/GPUI 统一 UI 与 Stage-2 自举编译器
cargo run --bin tenon -- build examples/dogfood/matching_engine.tn -o target/matching_bin --opt O3 && ./target/matching_bin
cargo run --bin tenon -- build examples/bootstrap/tenon_lex_fmt.tn -o target/lex_fmt_bin  --opt O3 && ./target/lex_fmt_bin
cargo run --bin tenon -- build examples/bootstrap/tenon_parser.tn  -o target/parser_bin   --opt O3 --emit-mir target/parser.mir && ./target/parser_bin
# Stage-2 纯 Tenon 自举编译器（0% Rust 编译链：Tenon -> LLVM IR -> clang -O3 原生可执行文件！）
cargo run --bin tenon -- test  examples/bootstrap/stage2/tenon_compiler_test.tn
cargo run --bin tenon -- build examples/bootstrap/stage2/tenon_compiler.tn -o target/stage2_compiler && ./target/stage2_compiler && clang -O3 /tmp/tenon_stage2_out.ll -o target/bootstrapped && ./target/bootstrapped
# OpenUI Lang + 统一 React 风格 VDOM（Web Wasm32 + TUI 双缓冲 ANSI + 原生 Metal GPU `tenon-gpui` 三端合一！）
cargo run --bin tenon -- test  std/ui/ui_test.tn && cargo run --bin tenon -- test std/ui/gpui_test.tn && cargo run --bin tenon -- test std/gpu/tensor_test.tn
cargo run --bin tenon -- build examples/dogfood/openui_wasm_kernel.tn --target wasm32 -o target/openui.wasm --emit-ts target/openui.d.ts --emit-mjs target/openui.mjs
cargo run --bin tenon -- build examples/dogfood/openui_tui_demo.tn -o target/openui_tui && ./target/openui_tui
cargo run --bin tenon -- build examples/dogfood/unified_ui_demo.tn --c-src std/gpu/metal/metal_shim.m -o target/unified_ui && ./target/unified_ui
# 零 Rust 声明式 GPU（@kernel/@gpu_graph/@gpu_struct -> MSL/WGSL + 真实 Apple Silicon Metal 硬件矩阵计算 & SDF 渲染）
cargo run --bin tenon -- build examples/dogfood/metal_tensor_demo.tn --emit-msl /tmp/tenon_mlp.metal --emit-wgsl /tmp/tenon_mlp.wgsl --c-src std/gpu/metal/metal_shim.m -o target/metal_tensor && ./target/metal_tensor
cargo run --bin tenon -- build examples/dogfood/metal_render_demo.tn --c-src std/gpu/metal/metal_shim.m -o target/metal_render && ./target/metal_render
# Go cgo 量化回测静态库 & SARIF v2.1.0 源码高亮诊断
cargo run --bin tenon -- check examples/refinements/bad_date.tn --format sarif
cargo run --bin tenon -- build examples/dogfood/quant_backtest.tn --lib static --emit-header examples/dogfood/go_cgo_demo/quant.h --emit-go examples/dogfood/go_cgo_demo/quant_cgo.go -o examples/dogfood/go_cgo_demo/libquant.a
# 标准库 / 工具链 CLI：`tenon test`, `tenon doc`, `tenon bench`, `tenon new`, `tenon pkg`
cargo run --bin tenon -- test std/net && cargo run --bin tenon -- test std/db && cargo run --bin tenon -- test std/sync && cargo run --bin tenon -- test std/stdlib_r3_test.tn && cargo run --bin tenon -- test examples/web_app
cargo run --bin tenon -- doc examples/dogfood/quant_backtest.tn -o target/tenon-doc --format both
cargo run --bin tenon -- bench benchmarks/tenon/07_simd_dot.tn --iters 3
cargo run --bin tenon -- evolve check examples/dast_os_kernel/kernel_v1.tn examples/dast_os_kernel/kernel_v2.tn --lock target/evolution.lock --emit-migration target/migration.sql

# 3. M1：Perceus 精确 RC 原地复用 + MVS COW + 自动派生 Date.parse(...) + 符号化 mutate Z3 证明
cargo run --bin tenon -- run   examples/m1_records_perceus/date_parse_and_mutate.tn
cargo run --bin tenon -- check examples/m1_records_perceus/bad_mutate.tn # 负例：漏调闰年 29->28 被 Z3 抓反例

# 4. M5 & M6：无栈 async 协程状态机 + @no_heap_alloc 契约 + LLVM 22 -O3 独立原生二进制编译
cargo run --bin tenon -- run   examples/m5_async_and_contracts/stackless_async.tn
cargo run --bin tenon -- check examples/m5_async_and_contracts/bad_effects.tn
cargo run --bin tenon -- build examples/m5_async_and_contracts/stackless_async.tn -o target/aot_async_bin --emit-llvm target/aot_async.ll --opt O3
./target/aot_async_bin

# 5. M7：单源 `dast-os-kernel` 全栈 Tenon 重写 — 7 把工具全链路验证同一份源码！
cargo run --bin tenon -- check        examples/dast_os_kernel/kernel_v2.tn
cargo run --bin tenon -- evolve check examples/dast_os_kernel/kernel_v1.tn examples/dast_os_kernel/kernel_v2.tn
cargo run --bin tenon -- modelcheck   examples/dast_os_kernel/kernel_v2.tn --export-tla target/tla_dast_kernel
cargo run --bin tenon -- run          examples/dast_os_kernel/kernel_v2.tn
cargo run --bin tenon -- build        examples/dast_os_kernel/kernel_v2.tn -o target/dast_kernel_bin --opt O3 && ./target/dast_kernel_bin
cargo run --bin tenon -- build-wasm   examples/dast_os_kernel/kernel_v2.tn -o target/dast_kernel_client.wasm
cargo run --bin tenon -- cluster-demo examples/dast_os_kernel/kernel_v2.tn

# 6. 4 层纯 Tenon 标准库与完备表达力自测（std/{core,text,collections,encoding,os}，含 Float/JFloat、闭包与高阶组合子、for/Range、`?` 错误传播、元组与泛型 enum 嵌套模式匹配、`f"..."`、@derive(Eq, Hash, ToJson, FromJson, JsonSchema)、Map[K,V]/Set[T]、SafeRelPath 边界证明，285 项静态证明，live_objects=0）
cargo run --bin tenon -- check examples/stdlib_test.tn
cargo run --bin tenon -- run   examples/stdlib_test.tn
cargo run --bin tenon -- build examples/stdlib_test.tn -o target/stdlib_test_bin --opt O3 && ./target/stdlib_test_bin

# 7. 纯 Tenon 编写的真实交互式 CLI Agent（examples/agent/cli_agent.tn + examples/agent/lib/{llm,tools}.tn，基于 dyn LlmProvider + List[dyn Tool] + @derive(FromJson, JsonSchema) + SafeRelPath + Z3 AgentLease + Protocol 模型检测，346 项静态证明，live_objects=0）
cargo run --bin tenon -- check      examples/agent/cli_agent.tn
cargo run --bin tenon -- modelcheck examples/agent/cli_agent.tn
# 方式 A：Cranelift JIT 直接运行（支持交互 REPL 或 TENON_PROMPT 单轮执行；自动统计 Perceus live_objects=0）
TENON_MODEL=luna cargo run --bin tenon -- run examples/agent/cli_agent.tn
# 方式 B：LLVM 22 -O3 AOT 编译为独立原生可执行文件并运行！
cargo run --bin tenon -- build examples/agent/cli_agent.tn -o target/tenon_agent --opt O3
TENON_MODEL=luna ./target/tenon_agent

# 8. 启动交互式 Web UI（自动加载 /tenon_client_store.wasm；可选环境变量对接真实 OpenAI 兼容 LLM 与 Webhook 渠道）
# TENON_LLM_BASE_URL=http://127.0.0.1:11434 TENON_LLM_MODEL=qwen2.5 TENON_CHANNEL_WEBHOOK_URL=http://127.0.0.1:9000 \
cargo run --bin tenon -- serve examples/dast_os_kernel/kernel_v2.tn --addr 127.0.0.1:8787

# 9. 多语言性能基准与编译回归套件（C / Rust / Go / Tenon LLVM 22 -O3 ThinLTO AOT；详见 benchmarks/README.md）
python3 benchmarks/run.py --check-only                                   # 编译 + 证明 + 跨语言校验和回归
python3 benchmarks/run.py --warmup 1 --reps 7 --save benchmarks/results.json

# 10. 完整系统编程能力端到端自测（S1+S2+S3：103 项 L0/L1 静态证明，JIT & LLVM 22 -O3 AOT 双端 live_objects=0）
cargo run --bin tenon -- check examples/systems_test.tn
cargo run --bin tenon -- run   examples/systems_test.tn
cargo run --bin tenon -- build examples/systems_test.tn -o target/systems_test_bin --opt O3 && ./target/systems_test_bin
# 双向 C FFI、静态/动态库与头文件导出、DWARF 调试信息与格式化器：
cargo run --bin tenon -- build examples/systems_test.tn -o target/systems_dbg -g --opt O0
cargo run --bin tenon -- fmt   examples/systems_test.tn --check
# 导出 C 静态库 + C99 头文件：tenon build mylib.tn --lib static --emit-header mylib.h -o libmylib.a
# 链接外部 C 源码/系统库：     tenon build app.tn --c-src helper.c -L /usr/local/lib -l z -o app

# 11. 模块组织（tenon.toml + Go/通配/选择性 import）、comptime/宏/反射 & Loom 风格 `tenon test`（逐测试 Perceus live=0 校验）
cargo run --bin tenon -- test -r examples/modules_demo                   # 递归发现包（遵守嵌套 tenon.toml 边界）
cargo run --bin tenon -- test examples/modules_demo/mathlib              # 测试嵌套模块包及同包私有函数共享
cargo run --bin tenon -- test examples/modules_demo/app --filter comptime
cargo run --bin tenon -- run  examples/modules_demo/app                  # 34.4 KB LLVM -O3 产物同样支持

# 12. 高性能网络库（kqueue/epoll Reactor + HTTP/1.1 Keep-Alive）与元编程 Web 框架（@get/@post + 自动 OpenAPI 3.1 + 精化校验）
cargo run --bin tenon -- test std/net                                    # kqueue + chunked + 2-worker HTTP/1.1 (live=0)
cargo run --bin tenon -- test examples/web_app                           # 208 项证明，全路由+中间件+OpenAPI (live=0)
cargo run --bin tenon -- build examples/web_app -o target/web_app_bin --opt O3 && ./target/web_app_bin  # 120 KB 独立原生 Web 服务
cargo run --bin tenon -- build benchmarks/tenon/http_rps_bench.tn -o target/http_rps_bin --opt O3 && ./target/http_rps_bin  # 59.6k RPS

# 13. 原生 LSP 语言服务器（`tenon lsp`）与 VS Code 扩展（editors/vscode/）
(cd editors/vscode && npm test)                                          # 验证语法高亮、配置并拉起真实 `tenon lsp` 全协议冒烟测试
```

### 四层分层架构（极简核心 → 4 层纯 Tenon 标准库 → 应用层 SDK & CLI Agent）

| 层级 / 模块 | 职责 |
| --- | --- |
| [`crates/tenon-syntax`](crates/tenon-syntax) | `loom/Go` 风格词法/语法分析器、**目录即包加载器（`project.rs` + `tenon.toml` + 未保存编辑器内存覆盖）与 4 种 `import` 形式（Go 式 `import pkg.subpkg [as alias]` / 通配 `import pkg.subpkg.*` / 选择性 `import pkg.subpkg.{A, B as C}` / 相对路径）**、**编译期求值与类型化宏/反射引擎（`comptime.rs`：`comptime { ... }` / `comptime fn` / `comptime if` / `comptime for` / `macro` / `gen!(...)` / `fields_of` / `routes_of` / `config`）**、精确宽度整数与 `Float32`/`Float64`、`@repr(C)`/`@packed`/`@align(N)` + `LayoutEngine`、`bitfield`、`union`、`unsafe`、`extern "C"`、`defer`、`[T; N]`、三引号字符串 `"""..."""`、精化/`mutate`/`async`/`evolution`/`protocol` |
| [`crates/tenon-rt`](crates/tenon-rt) | 极简语言核心运行时（`rlib` + `staticlib`，零 Cargo 外部依赖）：Perceus RC/COW/`rc_mask` + `Bytes`/`BytesSlice`/`TAG_CSTRUCT`/`TAG_SIMD`/`TAG_ATOMIC`/`TAG_MUTEX`/`TAG_CHANNEL` + 硬件位原语/原始内存与 volatile/OS 多线程与显式内存序原子/互斥锁/MPMC Channel/**OS `kqueue`/`epoll` 事件 Reactor 与 `TCP_NODELAY`/`SO_REUSEPORT` 非阻塞 TCP** + 单调高精度时钟 + 无栈协程调度 |
| [`crates/tenon-verify`](crates/tenon-verify) | `IncrementalDb` 签名早停增量缓存 + **常驻增量 Z3 进程 + 并行批量 SMT + 独立 SMT-LIB2 证明导出/多求解器审计（`tenon verify-proofs`）** + L0 区间/同余/八边形求解 + L1 Z3 SMT（`while` 不变量/终止性、`inout` `old()`、`ghost`/`lemma`、效应多态 `[effect E] / E`、ADT/Set 理论、**`@checkfree` 零运行时检查契约 `E0622`、`@linear`/`@must_use` 线性资源令牌 `E0821`、`inout` 独占定律 `E0820`、锁/`mutate` 临界区禁 `await` `E0603`、可执行反例转测试 `--emit-repro-test`**）+ `@evolve` + L2 协议模型检测（含活性 `leads_to` & `--replay`） |
| [`crates/tenon-mir`](crates/tenon-mir) | **统一类型化 SSA/CFG 中端 IR**（`MirProgram` / `BasicBlock` / `MirInst` / `--emit-mir`），承接 `VerifiedMir` 证明优化事实并执行后向活性 **Last-Use Move** 优化，在拥有态 RC 局部最后一次读取处消除 `rc_inc`/`rc_dec` |
| [`crates/tenon-codegen`](crates/tenon-codegen) | 统一前置降级 + Cranelift Dev JIT + **LLVM 22 AOT Release 后端（模块化 `llvm_aot.rs`/`llvm_aot_rt.rs`/`llvm_aot_tests.rs`，平面未装箱 `List[SimdVec]` & 4 字节 `List[Int32]`/`List[Float32]`、Tier-2 `alloca` 栈提升与 `@arena` 64 KiB Slab Arena、Last-Use Move 零 RC、双向 C FFI + `--lib static\|shared` + `--emit-header` + DWARF `-g`）** + **全功能 LLVM `wasm32` / `wasm32-wasip2` + TS/ESM Auto-Bridge (`--emit-mjs`)/Swift/Kotlin/Go `cgo` 绑定生成器** |
| [`std/`](std) | **100% 纯 Tenon 编写的 4 层正交标准库与生产网络/数据栈**：<br>• Layer 0 [`std/core/`](std/core)、[`std/math/decimal.tn`](std/math/decimal.tn)：精化类型、`Eq`/`Hash`/`Display`、`List`/`Option`/`Result`、**精确 10^8 定点数 `Decimal64`**<br>• Layer 1 [`std/text/`](std/text)、[`std/collections/`](std/collections)、[`std/simd/`](std/simd)、**[`std/regex/regex.tn`](std/regex/regex.tn)**（Thompson NFA 线性时间正则，抗 ReDoS）、**[`std/time/time.tn`](std/time/time.tn)**（Hinnant O(1) 日历 + RFC 3339）<br>• Layer 2 [`std/encoding/{json,dotenv,csv,url}.tn`](std/encoding) & **[`std/crypto/crypto.tn`](std/crypto/crypto.tn)**（SHA-256/SHA-1/HMAC/PBKDF2/Base64URL/恒定时间比较/JWT HS256）<br>• Layer 3 [`std/os/`](std/os)、[`std/sync/sync.tn`](std/sync/sync.tn)（含 **`TaskScope` 结构化并发**）、[`std/net/`](std/net)（Reactor + **C10K 单线程多路复用 HTTP/1.1 Server** + WebSocket + SSE + OpenSSL TLS 1.3）、[`std/db/`](std/db)（PostgreSQL SCRAM-SHA-256 + Extended Query + `SSLRequest` + 命名列 `PgRow` + `ConnPool` + SQLite CHECK）与 [`std/web/`](std/web)（Radix Trie 路由 + `@get`/`@post`/`@ws`/`@sse` + 自动 OpenAPI 3.1） |
| [`examples/`](examples) & [`editors/vscode/`](editors/vscode) | 完整示例矩阵（[`examples/modules_demo/`](examples/modules_demo) 多包与元编程、[`examples/web_app/`](examples/web_app) 120 KB 元编程 Web 服务、[`examples/agent/`](examples/agent) 交互式 CLI Agent、[`examples/systems_test.tn`](examples/systems_test.tn) 系统编程总测）与 **VS Code 扩展（TextMate 语法高亮 + 自动拉起 `tenon lsp`）** |
| [`crates/tenon-dist`](crates/tenon-dist) & [`crates/tenon-cli`](crates/tenon-cli) | SQLite WAL 分布式权威内核、真多进程 Authority/Worker/Fabric Sandbox、`tenon` 统一 CLI（含 `check`/`run`/`test`/`build`/`build-wasm`/`evolve`/`modelcheck`/`serve`/`fmt`/**`lsp`**） |

---

## 1. 核心目标与五大架构支柱

1. **一等强约束精化类型（延续 `loom-lang` / Go 简洁语法）**
   - 标量范围（`Amount > 0`）、数论同余（`Odd`/`Even`/`Aligned[N]`）、可验证域值（`Email`/`CardNumber`）、**跨字段依赖记录不变量**（如 `Date { year, month, day }` 含闰年合法校验、`DateRange`、复式记账借贷平衡）。
   - 外部边界自动派生结构化 `.parse()` / JSON / SQL `CHECK`（单源真相）；内部流敏感传播，静态证明后零开销消除冗余检查。
2. **三层内置形式化验证（L0 / L1 / L2）+ 一等协议模型检验**
   - **L0（<1ms 进程内）**：区间 + 同余 + 八边形 + 有界 CTFE，覆盖 85%–90% 日常精化约束；
   - **L1（增量 SMT + 内容缓存）**：处理 `requires`/`ensures`/`old()`、跨字段公式与版本蕴涵，**全模式严格 Fail-Closed**，提供人类与 AI 均可直读的最小反例；
   - **L2（双引擎协议验证 `tenon modelcheck` + Ghost/Lemma）**：内置轻快 Rust 有界状态探索器 + 一等 TLA+/Quint 导出 + 自动负控变异 + 实跑轨迹精化核验，直接发现多节点分布式协议交错缺陷。
3. **默认 MVS 值语义 + 无限制 Perceus 基座 + 透明区域优化**
   - 逻辑层采用**默认可变值语义（MVS）+ 局部 `inout`（零 `'a` 生命周期标注）**，彻底消灭普通值上的共享可变别名爆炸，使验证引擎复杂度直降一个数量级。
   - 运行时以 **Perceus 精确引用计数与唯一引用原地复用（COW/FBIP）** 为无限制语义基座（递归、逃逸闭包、Wasm、嵌入式全无限制，无 STW GC 停顿）；Goose 式区域降级为**透明 Tier-2 局部优化**（默认分块池化 Slab Arena），配强制报错的 `@no_heap_alloc` / `@fip` 契约；长期可变图走显式 `Store[T] + 不可回绕代次 Handle + phase`。
   - 并发采用**无栈协程（Stackless MIR 状态机）+ 显式 `async` 效应（配效应多态）**，原生服务端、Wasm 前端与无 MMU 嵌入式一套统一。
4. **版本优先的双层系统级演化与兼容性自动推导**
   - 保留普通文本文件，以版本对 `(v1, v2)` + 字段名为默认演化依据（无需在每个字段写 `@field(N)`）；仅对持久化/Wire/公开跨版本契约显式标注 `@evolve`；
   - 自动检测跨版本**字段类型变更**（如 `Int64 -> Text`）、**精化范围收窄/放宽**、**新增/删除/重命名字段**（`@from("old")` 或 `evolution V1 <-> V2` 双向透镜），融合 **SMT 精化蕴涵 + 残差域 $\Delta$ + 已证明双向透镜** 与 **按职责少版本化能力子集**，覆盖升级、灰度共存、降级保留数据与再升级全生命周期。
5. **极简通用核心 + 可验证标准库分布式微内核**
   - 语言核心只内建极简通用原语，三轴调和、锁序格、Attempt 双围栏、4 种可扩展 `SyncArchetype` 与响应式数据流/pacing 在标准库实现；
   - 面向 `dast-os` 级云端多端系统，提出**分片单写者 + 存储/代理双端硬 Fencing + 不可变 Turn DAG + 声明式响应式流**架构，并**单源直编**为服务端原生二进制 + Web Wasm（薄 TS 胶水）+ iOS/Android/Desktop Native C-ABI（薄 Swift/Kotlin 绑定）。

---

## 2. 一眼看懂 Tenon（`loom/Go` 简洁语法定稿）

```tenon
// 1. 强约束标量与数论同余精化（无冒号、无 ->）
pub type Amount = Int64 where self > 0
pub type Odd    = Int64 where self % 2 != 0
pub type Even   = Int64 where self % 2 == 0
pub type Email  = Text  where is_valid_email(self)

// L0 同余求解器 <1ms 静态证明，零 Z3 调用，运行时零检查
pub pure fn add_odds(a Odd, b Odd) Even {
    a + b
}

// 2. 跨字段依赖记录不变量：合法公历日期 Date
pub pure fn is_leap_year(y Int) Bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

pub pure fn days_in_month(y Int, m Int where 1 <= self && self <= 12) Int
    ensures 28 <= result && result <= 31
{
    match m {
        2 => if is_leap_year(y) { 29 } else { 28 }
        4 | 6 | 9 | 11 => 30
        _ => 31
    }
}

pub record Date {
    year  Int where 1 <= self && self <= 9999
    month Int where 1 <= self && self <= 12
    day   Int where 1 <= self && self <= days_in_month(year, month)
}

// 编译期 L0 局部求值直接通过（days_in_month(2024, 2) == 29）
let leap = Date { year = 2024, month = 2, day = 29 }

// 编译期直接报错（反例明确指出 29 <= 28 为假）
// let bad = Date { year = 2023, month = 2, day = 29 }

// 外部边界输入：自动生成结构化解析器
let parsed Result[Date, ValidationError] = Date.parse(raw_year, raw_month, raw_day)

// 多字段原子修改：默认 MVS 无共享别名，块退出点单点证明整体不变量
pub fn shift_to_prev_year_safe(inout d Date) {
    mutate d {
        d.year -= 1
        if d.month == 2 && d.day == 29 && !is_leap_year(d.year) {
            d.day = 28
        }
    }
}
```

### 系统级版本演化与双向透镜示例（版本优先，零 `@field(N)` 噪音）

```tenon
@evolve("billing.Order", storage = "orders")
pub record Order {
    id     Int64
    amount Int64 where self > 0       // 相对 v1 (self >= 0) 收紧！编译器算出残差域 Δ: amount == 0；若类型变化也会自动拦截
    note   Option[Text] = None        // 新增可选字段：后向兼容，降级保留
}

evolution Order: v1 <-> v2 {
    archive ZeroOrderArchive {
        order_id        Int64
        original_amount Int64 where self == 0
    }

    // 编译器静态证明：1) 全覆盖与目标合法 2) 回滚无损 down(up(x)) == x 3) 降级保留 note 后再升级无冲突
    up(old Order@v1) (Order@v2, Option[ZeroOrderArchive]) {
        if old.amount == 0 {
            (Order { id = old.id, amount = 1, note = None },
             Some(ZeroOrderArchive { order_id = old.id, original_amount = 0 }))
        } else {
            (Order { id = old.id, amount = old.amount, note = None }, None)
        }
    }

    down(new Order@v2, arch Option[ZeroOrderArchive]) Order@v1
        retain(new.note)
    {
        match arch {
            Some(a) if new.amount == 1 => Order@v1 { id = new.id, amount = a.original_amount }
            _ => Order@v1 { id = new.id, amount = new.amount }
        }
    }
}
```

---

## 3. 技术文档导航

| 文档 | 内容摘要 |
| --- | --- |
| **[ADR-0001 — 13 项核心架构决议总表](docs/adr/0001-architecture-decisions.md)** | **最高裁决真相源**：记录全部 13 项已拍板架构决议（战略定位、核心边界、MVS+Perceus、无栈协程、checked 整数+Fail-Closed、版本优先双层演化、分片单写者+Turn DAG、Wasm/Native 单源产物、双引擎协议验证、Rust+Salsa+Cranelift/LLVM、loom/Go 语法、M0–M4 节奏）。 |
| **[ADR-0002 — 接口/实现双轨、`@derive` 序列化、轻量 `@evolve` 与 4 层标准库](docs/adr/0002-interfaces-comptime-and-stdlib.md)** | 统一 `interface` + `impl` + 静态单态化 `[T I]` & 动态 `dyn I` Perceus 胖对象 + Liskov 契约检查、`@derive(ToJson, FromJson, JsonSchema)` 与精化边界单源真相、版本优先 `@evolve`、4 层正交 `std/` 与应用 SDK 解耦。 |
| [01 — 现状、竞品与近一年（2025–2026）新语言全景调研](docs/01-landscape-and-positioning.md) | 复盘 `loom-lang` 进展与结构性张力、`lang-research` 核心实验，评估近 1 年 20+ 新语言/验证体系（MoonBit 0.9、Ora、Bend 2、Gojo、Goose、Wyzer、Hylo、Ante、Zed、Blood、Ambient、Jacquard、Mog、Flux、Verus 等）的优缺点与定位空白。 |
| [02 — 强约束精化类型系统设计](docs/02-type-system-and-refinements.md) | 标量/同余/域值/跨字段依赖记录（`Date`、`DateRange`、集合、状态机）语法与语义；流敏感精化；原子 `mutate` 块；精化子类型格与零开销消除；单源真相自动派生边界解析器与 SQL/JSON 约束。 |
| [03 — 三层分层形式化验证引擎设计](docs/03-formal-verification-engine.md) | L0 进程内极速抽象域求解器（区间/同余/八边形/有界 CTFE）+ L1 增量 SMT（Z3/CVC5 + 缓存 + 可解释反例）+ L2 双引擎协议验证与 Ghost/Lemma；MVS 减负对比；AI Vericoding 协议。 |
| [04 — 系统级版本演化与兼容性推导引擎](docs/04-system-evolution-and-compatibility.md) | `@evolve` / `@field(N)` 双轨身份与 `evolution.lock`；API/Wire/Storage 四维 SMT 兼容性推导矩阵；一等双向演化透镜四大定律；在线 Expand-Contract 计划与 Append-Only 部署账本。 |
| [05 — 内存模型、并发与编译器实现路线](docs/05-memory-concurrency-and-compiler-architecture.md) | MVS + Perceus 无限制基座 + 透明 Tier-2 分块区域优化 + `@no_heap_alloc` + `Store[T]`；无栈协程 + 显式效应多态；Rust + Salsa + Cranelift(Dev/Wasm) + LLVM(Release) 架构与 M0–M4 门禁。 |
| [06 — 调和状态机形式化、分布式同步协议推导与通用微内核演化](docs/06-reconciliation-and-distributed-sync-calculus.md) | 对接 [`dast-os-kernel`](../../dast.ai/dast-os-kernel/README.md)：标准库三轴实体/锁序格/防 ABA 令牌/双围栏线性 Attempt；声明式约束求解绑定 1 套极简参数化通用 `SyncArchetype` 微内核；双层少版本化能力演进；`tenon modelcheck` 双引擎同源验证与轨迹精化。 |
| [07 — 重思 `dast-os-kernel`：全新架构提案与七维声明式响应式数据流编译](docs/07-dast-os-rethink-and-reactive-dataflow.md) | 剖析 `dast-os-kernel` 本质真理与偶然复杂度；提出**分片单写者 + 存储/代理双端硬 Fencing + 不可变 Turn DAG** 全新架构；总结七维完备声明模型；展示 `LLM 流 → msg_list（Anti-Join 无闪烁交接）→ 多端 Wasm/Native 33ms 帧合并 & 多渠道节流/死流消除` 的直编实现。 |

---

## 4. 下一步执行路线（M0 启动就绪）

依据 [ADR-0001 (D11 & D13)](docs/adr/0001-architecture-decisions.md)，接下来将按以下顺序分阶段推进实现与可证伪验收：

1. **M0 — 核心前端 + L0/L1 精化与契约 + Cranelift 纵向切片**：
   - Rust Cargo Workspace + Salsa 增量查询骨架 + `loom/Go` 风格 Lexer/Parser/HIR/THIR/VIR；
   - L0 进程内求解器（区间 / 同余 / 八边形 / 有界 CTFE）+ L1 增量 Z3 引擎 + 反例反演器；
   - Cranelift 原生执行后端，端到端跑通 `Amount > 0`、`Odd + Odd -> Even`、`Date`（含 `2024-02-29` 通过、`2023-02-29` 编译期反例拒绝）、`Date.parse` 与 `mutate` 块。
2. **M1 — MVS / Perceus 内存引擎 + 无栈协程 + 编译期只读反射与边界派生**；
3. **M2 — 双层演化引擎（`tenon evolve` + 双向透镜）+ 双引擎协议模型检验（`tenon modelcheck`）**；
4. **M3 — 透明 Tier-2 分块区域优化 + `@no_heap_alloc` + `Store[T]` + 标准库通用 `SyncKernel` / `SyncArchetype` + LLVM Release & 多端 Wasm/Native 单源编译**；
5. **M4 — 极小核验内核证书与自举评估**。
