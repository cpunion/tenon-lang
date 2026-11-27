# Tenon v12 物理硅片 Roofline 优化与跨框架全面评测报告

> **版本**: Tenon v12（Roofline 逼近优化版） · **评测日期**: 2026-11-27  
> **核心理念**: 超过竞品只是基本要求，每一轮优化均先实测并锚定目标芯片的**真实物理硅片上限（Roofline `GB/s` & `TFLOPS/TOPS`）**，再逐项击穿访存带宽、寄存器溢出、指令发射与 Tensor Core/AMX 瓶颈。  
> **评测原则**: 严格科学诚实、全部数据均来自真实物理硬件实测（Apple M4 Max Metal 4 / NVIDIA `2× RTX 6000 Ada sm_89` / NVIDIA `RTX 4090` / `2× A100 80GB` / Google Cloud `TPU v5e-1`），完整披露测试口径、宿主机负载状态与数值一致度（`BIT-EXACT`），**同时如实呈现 Tenon 领跑区与竞品占优区**。

---

## 一、物理硅片 Roofline 基线与 Tenon v12 核心内核突破

### 1.1 目标芯片实测物理上限

| 硬件平台 | 实测带宽上限 (`GB/s`) | 实测/理论算力上限 | v12 关键实测达标率 |
| :--- | ---: | :--- | :--- |
| **Apple M4 Max** (40-core Metal 4, 128 GiB Unified) | **`468.4 GB/s`** STREAM (`500.4 GB/s` 裸 `Q8_0` 读峰值 / `546 GB/s` 标称) | **`13.8–17.0 TFLOPS`** FP16-FP32 AMX (`MPS` 基线 `12.57T`) | `70B Q4_0` 端到端 **`450.5 GB/s` (`96.2%` STREAM!)**；裸 `Q4_0` **`464.8 GB/s` (`93.0%` 读峰值!)**；裸 `Q8_0` **`496.8 GB/s` (`99.4%`!)** |
| **NVIDIA `2× RTX 6000 Ada`** (48 GB × 2, `sm_89`, `p2p=0, NCCL=1`) | **`960 GB/s`** GDDR6 ECC (单卡) | **`182 TFLOPS`** FP16-FP32 TC / **`364 TOPS`** INT8 TC (`mma.m16n8k32.s8`) | `70B 1GPU Q4_0` 端到端 **`825.2 GB/s` (`86.0%` GDDR6!)**；`1B CB B=8` **`3,981.8 tok/s`**；`70B TP=2 CB B=8` **`194.4 tok/s`** |
| **Google Cloud TPU v5e-1** (16 GB HBM2e) | **`819 GB/s`** HBM2e | **`197 TFLOPS`** BF16 MXU / **`394 TOPS`** INT8 MXU | `4096³` MXU **`96.37%` BF16 / `96.01%` INT8**；HBM GEMV **`91.38%`** |

### 1.2 Tenon v12 相对 v11 的 10 项硅片级架构突破

#### Apple M4 Max Metal（Round 1–8b，零警告 `-Wall -Wextra`）
1. **发现并击穿 Apple10 `Q4_0` `half4 noshift` 带宽上限（`122.6 → 464.8 GB/s`, `3.8×`）**：微基准定位旧 `1tpb float4` 因 32 寄存器占用率悬崖与 `>> 4` 移位停顿仅达 `24.5%` 带宽；全面升级 `moe_gemv_q4`、`q4_1`、`qkv_q4`、`dense_gu_q4`、`topk_gate_up_q4`、`topk_down_accum_q4`/`q4_1` 为 **`4tpb half4 noshift`（`0x0F` / `0xF0` + `xh *= 0.0625h` 消除移位）**，实测达 `464.8 GB/s`（`93.0%` 500 GB/s 读峰值）。
2. **`Q4_K`/`Q5_K`/`Q6_K`/`Q8_0`/`Q5_0` 全量防溢出向量化**：`dot_q4_k_quarter` / `dot_q5_k_quarter` 改用 `half4 noshift` + FP32 缩放（`448.1 GB/s`, `+39%`）并新增融合 `moe_dense_gate_up_q5k`（每 Token 省 GLM-4.5-Air 90 次 Dispatch）；`dot_q6_k_quarter` 改用 32-bit `packed_ushort2`（减少 `4×` Load 指令）+ `tptg=64`；`moe_gemv_q8` / `moe_topk_down_accum_q8` / `q5_0` 升级至 `2-row/half-SG` FP32 安全路径（`496.8 GB/s`，彻底杜绝 SwiGLU 离群值 FP16 溢出）。
3. **小批硬件 AMX `64N×8M×32K` (`moe_gemm_q4_amx8` / `q4_1_amx8`) + `4tpb noshift` `small_b`**：打通 Apple Silicon 上 `B=3..8` 连续批处理算力上限，`70B CB B=8` 从 `13.3` 飙升至 **`33.54 tok/s`（`2.52×`）**，`1B CB B=8` 升至 **`1,139.0 tok/s`**。
4. **独立零回退 `moe_flash_mla_split_paged_512`（8-Head 分组 + `half4` 点积）+ `B=1` 向量化 MLA 直通**：剥离通用回退分支消除编译器多余栈/SMEM 分配（寄存器 `62 → 32`、DRAM 读取减半），并修复 `B=1` 误走 `16×16` 填充 AMX 的浪费，`DSV2 2K MLA` 从 v11 `40.1` 暴涨至 **`110.26 tok/s`（`2.75×`, `8/8 BIT-EXACT`）**。
5. **零栈溢出 GQA FlashDecoding（原生 `exp2` + `packed_short2 Q8` + 单读 8 元素 `Q4_0`）+ `64×64×32` 零分支 AMX Prefill + `moe_gemm_f16_amx`**：`1B pp512` 从 `943.4` 飙升至 **`4,979.3 tok/s`（`5.28×`）**、`8K Q8` 从 `135.1` 升至 **`264.4 tok/s`（`1.96×`，反超 `llama.cpp 257.9`）**、`4K Q4` 从 `180.7` 升至 **`298.1 tok/s`（`1.65×`，反超 `llama.cpp 278.6`）**、`SmolVLM-256M` Turn-1 Prefill 从 `471.6ms` 降至 **`178.5ms`（`2.64×`）**。

#### NVIDIA CUDA（Round 1–3，零警告 GCC `-Wall -Wextra` + `sm_89` 原生 SASS）
6. **修复 `tn_cumoe_load_gguf` 重置 `fast_dp4a=0` 隐蔽缺陷**：释放全部 128-bit `int4` `Q8_1` + `__dp4a` 内核默认生效（`70B 1GPU` 达 `20.63 tok/s = 86% GDDR6`）。
7. **10 KB Shared-Memory `xq8` `cm_gemm_q4_dp4a_sb` / `q4_1_dp4a_sb`（256 线程 = 8 行/CTA, 1 行/Warp, `8×` 减少 `xq8` L2 流量）**：`1B CB B=8` 从 `1,853.7` 暴涨至 **`3,981.81 tok/s`（`2.15×`, `2.18×` 领先 `llama-server 1,827.7`）**，`70B 1GPU CB B=4/8` 从 `32.4/39.7` 暴涨至 **`69.9/90.9 tok/s`（`2.2×`）**，`70B TP=2 CB B=4/8` 暴涨至 **`118.9/194.4 tok/s`（`2.4×`, `1.92×` 领先 `llama-server 101.3`）**。
8. **原生 INT8 Tensor Core MMQ（`cm_gemm_q4_mmq` / `q4_1_mmq`，`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`, `64M×64N×32K`, 6.9 KB 零 Bank 冲突 Stride-48 SMEM, 兼容 CUDA Graphs！）**：消除 `cuBLAS` FP16 反量化往返显存流量，使 `1B CB B=16` 在 CUDA Graph 内跑满 INT8 TC 达 **`2,414.8 tok/s`（反超 `llama-server 2,228.9`）**、`70B TP=2 pp128` 暴涨至 **`625.1 tok/s`（较 v11 `286.6` 提速 `2.18×`）**；配合行分块动态 `cuBLAS` FP16 TC，`1B pp512` 峰值达 **`38,199.4 tok/s`（较 v11 `6,156` 暴涨 `6.20×`）**。
9. **零栈溢出 GQA/MLA FlashDecoding（原生 `exp2f` + 向量化 `Q8/FP8/Q4`）**：`1B 4K FP16` **`245.4 → 561.8–587.2 tok/s`（`2.39×`）**、`8K Q8` **`156.3 → 478.0 tok/s`（`3.06×`）**、`8K FP8` **`133.2 → 425.9 tok/s`（`3.20×`）**、`DSV2 2K MLA` **`99.6 → 187.3–196.2 tok/s`（`1.97×`, `13.3×` 领先 `llama.cpp 14.1`）**。
10. **双平台 MoE 路由扩容（`te_all[64*16] → [256*16]`）**：修复 `pp256` 中 64..255 号 Token 未路由缺陷，使 **Metal、CUDA 1GPU、CUDA 2GPU 2D 三端 `DSV2 pp256` 100% BIT-EXACT (`ft=855`)**，且 CUDA `2GPU 2D pp256` 飙升至 **`3,377.6 tok/s`（`1.92×` 领先 `llama.cpp 1,757.3`）**。

---

## 二、NVIDIA `2× RTX 6000 Ada 48GB (sm_89)` 全面正面评测

> **测试环境**：同一台 RunPod Secure Cloud `2× NVIDIA RTX 6000 Ada Generation (49,140 MiB × 2, sm_89, 960 GB/s GDDR6 ECC, p2p=0, NCCL=1)`。`llama.cpp` 现场以 HEAD (`4c73444`) + `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89` 编译并全程开启 `-fa 1`（FlashAttention）。

### 2.1 `DeepSeek-V2-Lite-Chat.Q4_K_M`（16B Latent MLA + 64E Top-6 MoE — **Tenon 全面碾压**）

| 测试场景 | 拓扑 | Tenon v11 | **Tenon v12** | 上游 `llama.cpp` (`4c73444`, `-fa 1`) | **v12 vs v11** | **Tenon v12 vs `llama.cpp`** | 数值校验 |
| :--- | :---: | ---: | ---: | ---: | :---: | :---: | :---: |
| **Prefill `pp32`** | 1× GPU / **2× GPU 2D** | `469.3` / `641.1` | `804.2` / **`1,202.2`** | `1,268.7` (1GPU) | **`+71%` / `+88%`** | 2D `0.95×` | `BIT-EXACT (254)` |
| **Prefill `pp256`** | 1× GPU / **2× GPU 2D** | `740.5` / `1,080.4` | `2,127.6` / **`3,377.6`** | `1,757.3` (1GPU) | **`2.87×` / `3.13×`** | **`1.21×` / `1.92×` 胜出!!** | `BIT-EXACT (855)` |
| **Decode `d=0 B=1`** | 1× GPU / **2× GPU 2D** | `188.8` / `195.0` | `184.3` / **`195.5 tok/s`** | `58.9 tok/s` | 持平高位 | **`3.13×` / `3.32×` 胜出!!** | `12/12 BIT-EXACT` |
| **长上下文 `2K MLA`** | 1× GPU | `99.6` | **`187.3 tok/s`** (R1峰值 `196.2`) | `14.1 tok/s` | **`+88%–97%`** | **`13.28×` 胜出!!** | `12/12 BIT-EXACT` |
| **连续批处理 `CB B=8`** | 1× GPU / **2× GPU 2D** | `633.2` / `443.2` | `682.8` / **`718.7 agg tok/s`** | `54.9 agg tok/s` (`llama-server`) | **`+8%` / `+62%`** | **`12.4×` / `13.1×` 胜出!!** | `12/12 BIT-EXACT` |
| **连续批处理 `CB B=16`** | 1× GPU / **2× GPU 2D** | `853.4` / `799.5` | `982.5` / **`1,024.3 agg tok/s`** | `68.9 agg tok/s` (`llama-server`) | **`+15%` / `+28%`** | **`14.3×` / `14.9×` 胜出!!** | `12/12 BIT-EXACT` |

### 2.2 `Llama-3.3-70B-Instruct-Q4_0`（70B Dense GQA, `37.36 GiB` — 单卡 48G & 双卡 `TP=2`）

| 测试场景 | 拓扑 | Tenon v11 | **Tenon v12** | 上游 `llama.cpp` (`4c73444`, `-fa 1`) | **v12 vs v11** | **Tenon v12 vs `llama.cpp`** | 物理 Roofline 达标率 |
| :--- | :---: | ---: | ---: | ---: | :---: | :---: | :---: |
| **单卡 `d=0 B=1`** | 1× 48GB | `18.96` | **`20.63 tok/s`** | `20.82 tok/s` | **`+8.8%`** | **99.1% 持平** | **`825.2 GB/s` (`86.0%` GDDR6!)** |
| **单卡 `4K Q4 KV`** | 1× 48GB | — | **`17.39 tok/s`** (R1 `18.72`) | `16.01 tok/s` | — | **`1.09×` 胜出!** | `fd=1520, BIT-EXACT` |
| **单卡 `CB B=4 / B=8`** | 1× 48GB | `32.4 / 39.7` | **`69.9 / 90.9 agg tok/s`** | — | **`2.16×` / `2.29×`** | 单卡反超双卡 `llama-server B=4` (`70.9`) | `10 KB SMEM dp4a_sb` |
| **单卡 `pp128`** | 1× 48GB | `195.4` | **`362.2 tok/s`** (R1 `384.0`) | **`793.2 tok/s`** | **`+85%–97%`** | `0.46×`（竞品占优，见 §5.2） | `BIT-EXACT (14198)` |
| **双卡 `TP=2 d=0 B=1`** | 2× 48GB (`p2p=0`) | `25.98` | **`33.28 tok/s`** | `-sm row` **拒载崩溃** / `-sm layer`: `21.00` | **`+28.1%`** | **`1.58×` 胜出!** | `NCCL=1, 12/12 BIT-EXACT` |
| **双卡 `TP=2 CB B=4`** | 2× 48GB | `64.6` | **`118.9 agg tok/s`** | `70.92` (`llama-server -sm layer`) | **`+84.1%`** | **`1.68×` 胜出!!** | `12/12 BIT-EXACT` |
| **双卡 `TP=2 CB B=8`** | 2× 48GB | `80.5` | **`194.4 agg tok/s`** | `101.28` (`llama-server -sm layer`) | **`2.41×`** | **`1.92×` 胜出!!** | `12/12 BIT-EXACT` |
| **双卡 `TP=2 pp128`** | 2× 48GB | `286.6` | **`625.1 tok/s`** | **`998.7`** (`-sm layer`) | **`2.18×`** | `0.63×`（较 v11 缩小 2.2× 差距） | `BIT-EXACT (14198)` |

> 注：上游 `llama.cpp` 在无 P2P PCIe 的双卡 `RTX 6000 Ada` 上运行 `-sm row` 依然报错 `device CUDA0 does not support split buffers` 无法加载；而 Tenon v12 通过零链接 `dlopen("libnccl.so.2")` + 持久设备端 CUDA Graphs 原生跑通真 `TP=2`，所有批次 `12/12 BIT-EXACT`。

### 2.3 `Llama-3.2-1B-Instruct-Q4_0`（1.24B Dense GQA, `737 MiB` — 1× RTX 6000 Ada）

| 测试场景 | Tenon v11 | **Tenon v12** | 上游 `llama.cpp` (`4c73444`, `-fa 1`) | **v12 vs v11** | **Tenon v12 vs `llama.cpp`** | 数值校验 |
| :--- | ---: | ---: | ---: | :---: | :---: | :---: |
| **Prefill `pp512`** | `6,156.4` | **`38,199.4`** (R2 cuBLAS) / **`27,128.1`** (R3 MMQ) | **`52,701.8 tok/s`** | **`6.20×` / `4.41×`** | `0.72×` (cuBLAS) | `BIT-EXACT (15140)` |
| **Decode `d=0 B=1`** | `591.7` | **`614.0 tok/s`** (R1 `632.0`) | **`733.7 tok/s`** | **`+3.8%–6.8%`** | `0.84×–0.86×` | `12/12 BIT-EXACT` |
| **连续批处理 `CB B=8`** | `1,853.7` | **`3,981.8 agg tok/s`** | `1,827.7` (`llama-server`) | **`2.15×`** | **`2.18×` 胜出!!** | `10 KB SMEM dp4a_sb` |
| **连续批处理 `CB B=16`** | `1,804.4` | **`2,414.8 agg tok/s`** | `2,228.9` (`llama-server`) | **`+33.8%`** | **`1.08×` 胜出!** | `INT8 TC MMQ in Graph` |
| **长上下文 `4K FP16`** | `245.4` | **`561.8 tok/s`** (R1 `587.2`) | `633.9 tok/s` | **`2.29×–2.39×`** | `0.89×–0.93×` | `12/12 BIT-EXACT` |
| **长上下文 `8K Q8`** | `156.3` | **`478.0 tok/s`** | `590.7 tok/s` | **`3.06×`** | `0.81×` | `12/12 BIT-EXACT` |
| **长上下文 `8K FP8_E4M3`** | `133.2` | **`417.1 tok/s`** (R2 `425.9`) | 无原生 FP8 KV | **`3.13×–3.20×`** | **Tenon 独有优势** | `12/12 BIT-EXACT` |
| **长上下文 `4K Q4`** | `287.8` | **`545.8 tok/s`** | `611.2 tok/s` | **`1.90×`** | `0.89×` | `12/12 BIT-EXACT` |

---

## 三、Apple M4 Max (40-core Metal 4, 128 GiB Unified Memory) 全面正面评测

> **测试环境与透明口径**：同一台 Apple M4 Max (40-core GPU, 实测裸 STREAM 上限 `468.4 GB/s`, 裸 `Q8_0` 读上限 `500.4 GB/s`, FP16 AMX 上限 `13.8–17 TFLOPS`)，正面对比 **Tenon v12（Round 7/8b 洁净空载实测）** 与最新 **Homebrew `llama.cpp` build `12359` (`b1b770d90`, ggml 1.1.0, Metal 4 Residency Sets, `-fa 1`)**。*注：宿主机在最终批量复测中段出现外部进程 `/tmp/llgo-3220` 占满 `212%` CPU 导致双引擎同步掉速，下表已全部采用洁净空载同机实测值，并在备注中标明同窗受扰复测结果以保完全透明。*

### 3.1 `Llama-3.3-70B-Instruct-Q4_0`（70B Dense GQA, `37.36 GiB` — **达 `96.2%` 物理 DRAM 上限！**）

| 测试场景 | Tenon v11 | **Tenon v12** (M4 Max 空载) | 上游 `llama.cpp` (`12359`, `-fa 1`) | **v12 vs v11** | **Tenon v12 vs `llama.cpp`** | 物理 Roofline 达标率 / 内存 |
| :--- | ---: | ---: | ---: | :---: | :---: | :--- |
| **100% GPU `d=0 B=1`** | `8.94` (热测) / `11.03` | **`11.24 tok/s`** (R7 `10.83`) | `6.22 tok/s` (同机洁净实测) | 突破物理墙 | **`1.81×` 胜出!!** | **`450.5 GB/s` = `96.2%` STREAM!!** (`RSS 0.06G` vs `37.5G`) |
| **100% GPU `pp32`** | — | **`64.2 tok/s`** | **`77.3 tok/s`** | — | `0.83×` | `BIT-EXACT (279)` |
| **100% GPU `pp128`** | — | **`77.6 tok/s`** | `46.4 tok/s` | — | **`1.67×` 胜出!!** | **`10.9 TFLOPS`**, `BIT-EXACT (14198)` |
| **100% GPU `CB B=4 / B=8`** | — / `21.96` | **`16.89` / `33.54 agg tok/s`** | — | **`+53%` (`2.52×` vs R7 `13.3`)** | — | **`amx8` 64N×8M 硬件 AMX**, `BIT-EXACT` |
| **长上下文 `4K Q4 KV`** | `4.85` (热测) | **`9.44 tok/s`** | `8.51 tok/s` | **`+95%`** | **`1.11×` 胜出!** | `fd=1200, FP 0.68G` vs `1.09G` |
| **超内存流式 `≤32G` / `≤16G`** | `4.55` / `2.08` | **`4.55` / `2.08 tok/s`** | `0.37` / `0.32 tok/s` | 保持 #1 | **`12.3×` / `6.5×` 胜出!!** | 严格预算零 Swap |

### 3.2 `Llama-3.2-1B-Instruct-Q4_0`（1.24B Dense GQA, `737 MiB`）

| 测试场景 | Tenon v11 | **Tenon v12** (M4 Max) | 上游 `llama.cpp` (`12359`, `-fa 1`) | **v12 vs v11** | **Tenon v12 vs `llama.cpp`** | 内存 / 校验 |
| :--- | ---: | ---: | ---: | :---: | :---: | :--- |
| **Prefill `pp512`** | `943.4` | **`4,979.3 tok/s`** | **`6,122.0 tok/s`** | **`5.28×`!!** | `0.81×` | `BIT-EXACT (15140)` |
| **Decode `d=0 B=1`** | `330.5` | **`380.2 tok/s`** (R7 `366.6`) | `370.5 tok/s` | **`+15.0%`** | **`1.03×` 胜出!** | `RSS 0.07G` vs `0.89G` (`12.7×` 省 RSS) |
| **长上下文 `4K FP16`** | `161.4` | **`299.0 tok/s`** | `331.8 tok/s` | **`1.85×`** | `0.90×` | `fd=560, RSS 0.06G` vs `1.14G` |
| **长上下文 `8K Q8`** | `135.1` | **`264.4 tok/s`** | `257.9 tok/s` | **`1.96×`** | **`1.03×` 胜出!** | `fd=560, RSS 0.06G` vs `1.16G` |
| **长上下文 `8K FP8_E4M3`** | `119.3` | **`244.1 tok/s`** | 无原生 FP8 KV | **`2.05×`** | **Tenon 独有优势** | `fd=560, BIT-EXACT` |
| **长上下文 `4K Q4`** | `180.7` | **`298.1 tok/s`** | `278.6 tok/s` | **`1.65×`** | **`1.07×` 胜出!** | `fd=560, RSS 0.06G` vs `0.96G` |
| **内核批处理 `CB B=8/16/32`** | `868.5` / — / — | **`1,139.0` / `1,424.3` / `2,166.1`** | — | **`+31%`** | — | `4tpb noshift` + `amx16` |
| **HTTP 并发 `B=8 / B=16`** | `767.3` / `717.8` | **`1,005.3` / `1,191.4`** (`803K` 零 Rust: `901.3 / 856.7`) | `1,093.4–1,176.0` / `1,300.3–1,328.7` | **`+31%` / `+66%`** | `0.92×` (`803K` 服务体积小 `29.9×`!) | 同窗受扰 R8b 仍达 `1,005 / 1,191` |

### 3.3 `DeepSeek-V2-Lite-Chat.Q4_K_M`（16B Latent MLA + 64E Top-6 MoE, `9.7 GiB`）

| 测试场景 | Tenon v11 | **Tenon v12** (M4 Max 空载) | 上游 `llama.cpp` (`12359`, `-fa 1`) | **v12 vs v11** | **Tenon v12 vs `llama.cpp`** | 内存 / 校验 |
| :--- | ---: | ---: | ---: | :---: | :---: | :--- |
| **Prefill `pp32` / `pp256`** | `208.8` / — | **`513.6` / `1,245.4 tok/s`** | **`630.3` / `1,736.6`** | **`2.46×`** | `0.81×` / `0.72×` | `BIT-EXACT (254 / 855)` |
| **Decode `d=0 B=1`** | `66.7` | **`137.2 tok/s`** (R7 `133.3`) | **`159.2 tok/s`** | **`2.06×`!!** | `0.86×` (同窗受扰 `70.9 vs 66.3`) | `RSS 0.07G` vs `9.85G` |
| **长上下文 `2K MLA`** | `40.1` | **`110.26 tok/s`** (R7 `101.2`) | **`135.6 tok/s`** | **`2.75×`!!** | `0.81×` (`5.17×` vs `fd=0`) | **`FP 0.97G` vs `1.38G`, `8/8 BIT-EXACT`** |
| **内核批处理 `CB B=8 / B=16`** | `200.7` / `247.6` | **`232.5` / `482.6 agg tok/s`** | — | **`+16%` / `+95%`** | — | `AMX Mega-Kernel v3` |
| **HTTP 并发 `B=8 / B=16`** | `99.3` / `89.1` | `228.4` / **`324.5 agg tok/s`** | `303.4` / `264.4` (`llama-server`) | **`2.30×` / `3.64×`** | `0.75×` / **`1.23×` 胜出!!** | `B=16` 高并发优势显著 |

### 3.4 `GLM-4.5-Air-Q4_0`（106B 128E Top-8 Sparse MoE, `58.47 GiB`, 2 Shards）

| 预算档位 | Tenon v11 | **Tenon v12** (M4 Max) | 上游 `llama.cpp` (`12359`, `-fa 1`) | **v12 vs v11** | **Tenon v12 vs `llama.cpp`** | 物理内存 Footprint / RSS |
| :--- | ---: | ---: | ---: | :---: | :---: | :--- |
| **`≤ 42G` (`80/128` LRU)** | `33.56` | **`36.30 tok/s`** | `2.27–3.13` (`-ot exps=CPU`) | **`+8.2%`** | **`11.6×–16.0×` 胜出!!** | **`FP 38.24G` (`RSS 7.44G`)** vs `FP 51.03G` (`RSS 75.5G`) |
| **`≤ 16G` (`24/128` LRU)** | `14.58` | **`19.95 tok/s`** | `2.27–3.13` (`-ot exps=CPU`) | **`+36.8%`** | **`6.37×–8.79×` 胜出!!** | **`FP 15.32G` (`RSS 9.49G`)** — 严守 16G 红线 |
| **无上限 128G `100% GPU`** | `29.76` | **`46.17 tok/s`** | **`51.15 tok/s`** | **`+55.1%`!!** | **`0.90×` 逼近持平!** | `76%` M4 Max 物理上限 (`60.8 tok/s`), `12/12 BIT-EXACT` |

### 3.5 `SmolVLM-256M-Instruct`（多模态 ViT + LLM 多轮对话 COW 复用）

| 指标 | Tenon v11 | **Tenon v12** (M4 Max) | `llama-server` VLM (`12359`) | **v12 vs v11** | 说明 |
| :--- | ---: | ---: | ---: | :---: | :--- |
| **Turn-1 冷启动 Prefill / E2E** | `471.6 ms` / `536.1 ms` | **`178.5 ms` / `245.0 ms`** | **`90.9 ms` / `151.1 ms`** | **`2.64×` / `2.19×` 提速!!** | 新增 `moe_gemm_f16_amx` + 零溢出 `vit_full_attn` |
| **Turn-2 追问 Prefill / E2E (COW)** | `12.1 ms` / `39.6 ms` | **`10.3 ms` / `38.8 ms`** | `3.7 ms` / `35.8 ms` | **`+17%` Prefill** | **PagedKV 块级多模态 COW 17.3× 加速，E2E 与 `llama-server` 基本持平** |

---

## 四、历史跨架构基线汇总（六引擎超显存 / RTX 4090 / TPU v5e-1 / 生产高可用验收）

1. **生产级高可用与全工作区回归（v12 全绿验收）**：
   - `cargo test --workspace` 全量通过（0 失败）；
   - **`tenon llm` CLI 与 `803 KiB` 零 Rust 原生服务器（`examples/showcase/tenon_llm_server.tn`, 360 项 Z3 证明 + 130 越界消除 + 95 溢出消除）双端 E2E 7/7 全绿**：HA PD Router 主节点宕机自动熔断故障转移（`failovers=2`）、Layer-Stream `TNKL` + Binary + SHM 三通道 100% `BIT-EXACT` 且非法篡改块 `HTTP 400` 拒收、实时 SSE 客户端 `RST` 断连零 `SIGPIPE` 并瞬间回收全部 PagedKV 块（`kv_free 128→128`）、零停机动态 LoRA 热插拔（`/v1/lora/{load,unload,list}`）、队列背压（`HTTP 503 + Retry-After: 1`）、Prometheus 全量指标。
2. **六引擎超显存 MoE/Dense 实测（`RTX 4090 48G/24G` & `M4 Max` 严格冷盘 `F_NOCACHE`，详见 [`TENON_V11_BENCHMARK_REPORT.md`](TENON_V11_BENCHMARK_REPORT.md)）**：
   - `GLM-4.5-Air 106B` 在 CUDA `≤16G` (`35.76`) / `≤24G` (`40.15`) / `≤44G` (`50.48`) 与 NVMe 直流 (`3.40`) 全档位 #1，领先 `ik_llama.cpp` `1.06×–1.43×`、`llama.cpp` `1.25×–1.82×`、`BigMoeOnEdge` `1.64×–6.28×`；
   - `Llama-3.3-70B` 单卡 `24G RTX 4090` 超显存达 `3.97–4.41 tok/s`（反超 `llama.cpp -ngl 48` `+26.5%`）；
   - `OLMoE-1B-7B` Mac 冷盘 `F_NOCACHE` 达 `51.6–58.4 tok/s`（反超 `llama.cpp -ot` `3.9×–6.3×`）。
3. **跨三硬件声明式编译器 `@gpu_graph` 基线**：
   - **Apple M4 Max Metal**：`2048³` `13.47 TFLOPS`（反超 `Apple MPS 12.57T` `+7.2%`），`4096³` `13.81 TFLOPS` (`85.2%`)；
   - **NVIDIA RTX 4090 CUDA**：`4096³` FP8 PTX **`453.4 TFLOPS`**（反超 `cublasLt 306.1T` `1.48×`），1.24B `S=512` Prefill **`148,128 tok/s`**（`2.37×` PyTorch 2.8）；
   - **Google Cloud TPU v5e-1 Direct PJRT**：11/11 全胜 `JAX 0.11.2`（MXU BF16 `189.85T / 96.37%`、INT8 `378.27T / 96.01%`、HBM `748.4 GB/s / 91.4%`）。

---

## 五、综合工程评估：v12 已补齐的全部短板 vs 剩余可量化边界

### 5.1 v12 已彻底补齐并实现反超/逼近物理极限的领域
1. **Apple M4 Max 带宽墙彻底打穿**：`Llama-3.3-70B` 达 **`11.24 tok/s`（`450.5 GB/s = 96.2%` 物理 DRAM 上限，`1.81×` 领先 `llama.cpp`）**；`Llama-3.2-1B` `B=1` (`380.2`)、`8K Q8` (`264.4`)、`4K Q4` (`298.1`) 全线反超 `llama.cpp`；`GLM-4.5-Air 100% GPU` 暴涨 `+55%` 至 `46.17 tok/s`（逼近 `llama.cpp 51.15`）；`70B CB B=8` 暴涨 `2.52×` 至 `33.54 tok/s`。
2. **NVIDIA CUDA 并发吞吐、多卡 `TP=2` 与 MLA+MoE 全面统治**：`1B CB B=8` 暴涨至 **`3,981.8 tok/s`（`2.18×` `llama-server`）**、`CB B=16` **`2,414.8`（超 `llama-server`）**；`70B TP=2` Decode **`33.28 tok/s`（`1.58×`）**、`CB B=8` **`194.4 tok/s`（`1.92×` `llama-server`）**；`DeepSeek-V2-Lite` 2D `pp256` **`3,377.6`（`1.92×`）**、`2K MLA` **`187.3`（`13.3×`）**、`CB B=16` **`1,024.3`（`14.9×`）**。
3. **长上下文 FlashDecoding 与 Prefill 数量级跃升**：相比 v11，CUDA `1B pp512` 提速 **`6.20×`**、4K/8K 提速 **`1.90×–3.20×`**；Metal `1B pp512` 提速 **`5.28×`**、4K/8K 提速 **`1.65×–2.05×`**、`DSV2 2K MLA` 提速 **`2.75×`**、`SmolVLM` Turn-1 Prefill 提速 **`2.64×`**。

### 5.2 客观存在的剩余微距（诚实披露与物理根因）
1. **CUDA `Q4_0` 大块 Dense Prefill（`1B pp512`: `38.2K` vs `52.7K`；`70B 1GPU pp128`: `362` vs `793`）**：当前 `cm_gemm_q4_mmq` 采用 `BK=32`（每 32 个 K 元素 2 次 `__syncthreads()`），若升级为 `BK=64/128` + `cp.async` 2-stage 流水线（减少 `4×` 同步开销），理论上可将 `70B 1GPU pp128` 从 `362` 推至 `~1,000+ tok/s`。
2. **Apple M4 Max `DeepSeek-V2-Lite` `B=1` 与 `pp256`（`137.2` vs `159.2`，`1,245` vs `1,737`）**：单流距 `llama.cpp` 差 `1.2 ms/tok`（主要来自 `Q4_K` 融合路由权重读与 `27L × 5` 小核调度开销）；但高并发 `HTTP B=16`（`324.5` vs `264.4`, `1.23×`）与显存占用（`RSS 0.07G` vs `9.85G`）Tenon 已反超。
3. **SmolVLM-256M Turn-1 冷启动（`178.5 ms` vs `90.9 ms`）**：较 v11 已削减 `293 ms`（`2.64×`），剩余 `~87 ms` 来自 ViT 1024-token 标量 `vit_full_attn`（可换 AMX 矩阵注意力），而多轮 Turn-2 COW 复用的 E2E 延迟已基本持平（`38.8` vs `35.8 ms`）。
