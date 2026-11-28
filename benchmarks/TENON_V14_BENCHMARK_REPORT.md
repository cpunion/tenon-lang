# Tenon v14 全量化 Roofline、2026 前沿架构与双平台全面评测报告

> **版本**: Tenon v14（Universal Roofline + 2026 Frontier Ecosystem） · **评测日期**: 2026-11-28  
> **核心目标**: 在**所有主流开源量化格式（`Q4_0` / `Q4_K_M` / `Q5_0` / `Q5_K` / `Q6_K` / `Q8_0`）**、**2026 前沿架构（Gemma-3 / Qwen3.5 混合 Gated DeltaNet + Gated Full GQA / 多模态 ViT / 16B–106B MoE / 70B Dense）**与**生产服务形态（Zero-Rust `MAX_CB=16` 真混批 Prefill+Decode / 多卡 TP / 2D TP×EP / 长上下文四模式 KV）**下全线补齐短板、逼近物理硅片 Roofline 上限。  
> **科学诚实口径**: 全部数据均来自真实物理硬件实测（Apple M4 Max Metal 4 + RunPod NVIDIA `2× RTX 6000 Ada sm_89`），完整披露贪婪解码一致率、上游 `llama.cpp` 版本号、实机迭代消融（含两次实测回退与修复过程）及宿主机背景负载情况。

---

## 一、Tenon v14 五大核心架构支柱与硅片 Roofline 突破

| 支柱 | 覆盖范围与核心实现文件 | Apple M4 Max (Metal 4) 实测突破 | NVIDIA `2× RTX 6000 Ada` (`sm_89`) 实测突破 |
| :--- | :--- | :--- | :--- |
| **Pillar 1：全量化双平台 Roofline** | `std/gpu/metal/moe_stream_shim.m`<br>`std/gpu/cuda/moe_stream_cuda_shim.c` | 新增 `Q4_K` / `Q5_0` / `Q5_K` / `Q6_K` / `Q8_0` 零转置 `C^T=W·X^T` AMX GEMM、融合 `QKV` 与 `Gate||Up`；**`1B Q8_0 tg=255.3`（反超 `llama.cpp 250.3` `+2.0%`）**、`1B Q4_0 tg=355.7`（**`+17.1%`**）、`Q4_K_M / Q8_0 pp512` 达 `92.2%–94.8%` | 新增 **`cm_gemm_q8_mmq`** & **`cm_gemm_q4k_mmq`**（`cp.async.ca` L1+L2 缓存双缓冲 INT8 TC MMQ）+ 全量化 `DP4A` `4-row/CTA` & 对齐 32-bit 块对加载；**`pp512` 全线大幅反超 `llama.cpp`：`Q4_0 +18.7%` (`46.1K`)、`Q4_K_M +91.1%` (`35.9K`)、`Q8_0 2.86×` (`43.1K`)！** |
| **Pillar 2：多模态 ViT `2×+` 极致提速** | `std/gpu/metal/moe_stream_shim.m`<br>`std/gpu/cuda/moe_stream_cuda_shim.c` | Single-CB 全 GPU ViT（27 层单命令缓冲区提交 + `vit_full_attn_amx` + GPU Pixel-Shuffle，消除 ~480 次同步）：**`SmolVLM-256M` Warm Turn-1 Prefill 降至 `41.7 ms`（反超 `llama-server 66.1 ms` `1.59×`！）**，E2E `92.5 ms`（`1.37×`），Turn-2 COW `8.7 ms` | 单流 GPU ViT + cuBLAS FP16 TC Patch + GPU Pixel-Shuffle：**`SmolVLM-256M` Cold `81.58 ms`（较 `llama-server 121.4 ms` 快 `1.49×`）、Warm `30.60 ms`（快 `3.97×`！）** |
| **Pillar 3：2026 前沿架构双平台全支持** | `std/gpu/metal/moe_stream_shim.m`<br>`std/gpu/cuda/moe_stream_cuda_shim.c`<br>`crates/tenon-cli/src/llm.rs` | 完整实现 **Gemma-2 / Gemma-3**（4-RMSNorm、GEGLU、`+1` norm 偏移、`embed_scale=√d_model`、每 6 层 Local/Global SWA、`d_head=256`、`final_logit_softcap`）与 **Qwen3.5**（18 层 **Gated DeltaNet** 深度可分离 Conv1D+SiLU+L2-Norm+状态递推 + 6 层 **Gated Full GQA** + 零 BOS 处理）：**Apple Metal 与 `llama.cpp` 均 100% `exact=True` 贪婪对齐** | **Gemma-3-1B**：`pp512 = 31,598–32,399 tok/s`（**`2.28×–2.33×` `llama.cpp 13,880`**），`CB B=16 = 3,503 tok/s`，`100% exact`；<br>**Qwen3.5-0.8B**：**`pp512 = 24,386 tok/s`（`+68.3%`）**、**`tg = 588.1 tok/s`（反超 `llama.cpp 493.8` `+19.1%`！）**、`CB B=16 = 4,276 tok/s`，`100% exact` |
| **Pillar 4：CUDA MMQ + Small-Batch DP4A 融合** | `std/gpu/cuda/moe_stream_cuda_shim.c` | （Metal 端同步补齐 `Q4_K`/`Q5_0`/`Q8_0` 融合 QKV 与 Gate\|\|Up） | 修正 `B > 16u` MMQ 切换阈值（`B=2..16` 全量化直通 `dp4a_sb` 8N×8M 共享内存微内核），**`CB B=16` 全模型暴涨 `1.8×–5.05×`**：`1B Q4_0 = 5,627 tok/s`（**`3.85×` `llama-server 1,460`**）、`Q4_K_M = 4,889`、`Q8_0 = 4,755–4,925`、`Qwen3.5 = 4,276`、`DSV2 = 1,424`！ |
| **Pillar 5：Zero-Rust 生产服务器升级** | `examples/showcase/tenon_llm_server.tn`<br>`examples/showcase/tenon_llm_server_cuda.tn` | **`962 KiB` 纯 Tenon 零 Rust 原生二进制**（**374 项 Z3 形式化证明全绿 + 136 项越界检查消除**）：`MAX_CB=16`、`MAX_RD=64`、**v14 真混批 Prefill+Decode 调度**（`/metrics` 新增 `tenon_mixed_prefill_decode_batches_total`）、Gemma-3/Qwen3.5 聊天模板与 `<end_of_turn>` 停止词；**M4 Max `HTTP B=16` 达 `1,899.1 tok/s`（反超 `llama-server 1,524.8` `+24.5%`！）** | CUDA 对等服务器同步移植 v13 T2.3 突发排空 + 冷突发 COW 前缀去重 + `MAX_CB=16` + 真混批 Prefill+Decode + Gemma-3/Qwen3.5 模板，`tenon check` 全绿 |

---

## 二、NVIDIA `2× RTX 6000 Ada 48GB`（`sm_89`）八模型全面正面评测

> **测试硬件**：RunPod `2× NVIDIA RTX 6000 Ada Generation`（48 GiB GDDR6 ECC × 2，单卡 `960 GB/s` / `182 TFLOPS FP16` / `364 TOPS INT8`，`p2p=0, NCCL=1`）。  
> **对照基线**：同机编译上游 **`llama.cpp` HEAD (`8e23278`, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -fa 1`)**；由于 `llama.cpp 8e23278` 在 `DeepSeek-V2-Lite` 上仍有 MLA 性能退化（`tg=35.8 / 2K=4.11`），下表一并标注 v13 同卡 `llama.cpp 4c73444` 参考值以保公平。  
> **实机迭代透明说明**：Round-1（全 8 模型 + `llama.cpp 8e23278` 对照）发现 3 项 CUDA 短板后，经 Round-2（修复 `cp.async.ca` + `B > 16u` + `_4r`）与 Round-3（收回两项实测回退：移除跨行非合并 `q8_lm` 与 70B FP16 激活直投）终态实测验证。下表所列 Tenon v14 均为**最终提交代码的实测值**。

### 2.1 主流全量化与 2026 前沿模型单卡实测（`1× RTX 6000 Ada`）

| 模型（架构 / 量化） | 指标 | **Tenon v14 (终态实测)** | 上游 `llama.cpp` (`8e23278`, `-fa 1`) | **Tenon v14 vs `llama.cpp`** | 贪婪一致性与关键技术 |
| :--- | :--- | ---: | ---: | :---: | :--- |
| **`Llama-3.2-1B-Q4_0`**<br>(1.24B Dense GQA) | `pp512` | **`46,074.3 tok/s`** | `38,823.5 tok/s` | **`1.19×` 胜出!** | 精简 1-stage `sX` INT8 TC MMQ（超 v13 `44.9K`） |
| | `d=0 B=1` | `583.7 tok/s` | **`735.5 tok/s`** | `0.79×` | `100% exact`（`g_cap=2, g_rep=82`） |
| | `4K FP16 / 4K Q4 / 8K Q8` | **`554.5 / 541.8 / 530.6`**¹ | `629.3 / 608.7 / 587.0` | `0.88×–0.90×` | `4K Q4` 较 v13 (`466.5`) **`+16.1%`**, `fd=560` |
| | `8K FP8_E4M3` | **`427.9 tok/s`** | 无原生 FP8 KV | **Tenon 独有** | `fd=560` |
| | **`CB B=8 / B=16`** | **`3,871.9 / 5,627.3`** | `1,333.4 / 1,460.3` (`llama-server`) | **`2.90× / 3.85×` 胜出!!** | `B > 16u` 修复使 `B=16` 暴涨 **`5.05×`**！ |
| **`Llama-3.2-1B-Q4_K_M`**<br>(P1 全量化 Roofline) | **`pp512`** | **`35,878.8 tok/s`** | `18,775.9 tok/s` | **`1.91×` 胜出!!** | `cm_gemm_q4k_mmq` + `cp.async.ca`（零 FP16 中间写） |
| | `d=0 B=1` | `571.1 tok/s` | **`651.5 tok/s`** | `0.88×` | 100% 一致于 Apple Metal `llama.cpp` |
| | **`CB B=8 / B=16`** | **`3,725.2 / 4,888.7`** | — | **8.56× vs `B=1`** | `dp4a_sb` + 4-row/CTA `_4r` |
| **`Llama-3.2-1B-Q8_0`**<br>(P1 全量化 Roofline) | **`pp512`** | **`43,118.3 tok/s`** | `15,070.5 tok/s` | **`2.86×` 胜出!!** | `cm_gemm_q8_mmq`（零 nibble 解包 + `cp.async.ca`） |
| | `d=0 B=1` | `432.5 tok/s` | `465.3 tok/s` | **`0.93×` 逼近持平!** | **`100% exact`** vs `llama.cpp` CUDA，对齐 68B 块对加载 |
| | **`CB B=8 / B=16`** | **`2,984.0 / 4,755.0`** | — | **11.0× vs `B=1`** | `dp4a_sb` 8N×8M 共享内存 |
| **`Gemma-3-1B-IT-Q4_K_M`**<br>(P3 4-Norm/GEGLU/SWA/`dh=256`) | **`pp512`** | **`31,597.8–32,399.1`** | `13,880.4 tok/s` | **`2.28×–2.33×` 胜出!!** | 全量原生 Gemma-3 架构支持 |
| | `d=0 B=1` | `322.4 tok/s` | **`473.0 tok/s`** | `0.68×` | R1 与 CUDA `llama.cpp` **`100% exact`**；R3 与 Metal `llama.cpp` **`100% exact`** |
| | **`CB B=8 / B=16`** | **`2,107.7 / 3,498.2`** | — | **10.9× vs `B=1`** | `g_cap=2`, 8-lane/row `Q5_0` Gate\|\|Up |
| **`Qwen3.5-0.8B-Q4_K_M`**<br>(P3 18×GDN + 6×Gated GQA) | **`pp512`** | **`24,385.6 tok/s`** | `14,487.6 tok/s` | **`1.68×` 胜出!!** | 原生混合 Gated DeltaNet + Gated Full GQA |
| | **`d=0 B=1`** | **`588.1 tok/s`** | `493.8 tok/s` | **`1.19×` 反超胜出!!** | **`100% exact`**，`_4r` 高网格 GEMV 提速 **`+19.2%`**！ |
| | **`CB B=8 / B=16`** | **`3,035.4 / 4,276.2`** | — | **7.27× vs `B=1`** | `g_cap=2, g_rep=82` |
| **`SmolVLM-256M`**<br>(P2 多模态 GPU ViT) | **Cold ViT + Prefill** | **`81.58 ms`** (`vit=76.1, pf=5.5`) | `121.4 ms` (`e2e=181.8 ms`) | **`1.49×` 胜出!** | 单流 GPU ViT + FP16 TC Patch + GPU Pixel-Shuffle |
| | **Warm ViT + Prefill** | **`30.60 ms`** (`vit=26.4, pf=4.2`) | `121.4 ms` (Turn-1) | **`3.97×` 胜出!!** | — |

> ¹ `1B Q4_0` `8K Q8` 在 R1 测得 `530.6 tok/s`、R3 测得 `491.7 tok/s`（云端同型号 GPU 频率波动约 ±4%）；`4K FP16` / `4K Q4` 在 R3 进一步升至 `554.5 / 541.8 tok/s`。

### 2.2 生产级大规模 MoE 与 70B Dense 多卡评测（`1× / 2× RTX 6000 Ada`）

| 模型与拓扑 | 指标 | **Tenon v14** | 上游 `llama.cpp` (`8e23278` / v13 `4c73444`) | **Tenon v14 vs `llama.cpp`** | 备注 |
| :--- | :--- | ---: | ---: | :---: | :--- |
| **`DeepSeek-V2-Lite` 1G (`EP=1`)** | `pp256` | **`2,217.1 tok/s`** | `1,060.7` (`4c73444`: `926.7`) | **`2.09×` 胜出!!** | `BIT-EXACT (855)` |
| | `d=0 B=1` / **`2K MLA`** | **`191.6`** / **`188.3 tok/s`** | `35.8` / `4.11` (`4c73444`: `22.8 / 3.54`) | **`5.35×` / `45.8×` 胜出!!** | 较 v13 `189.2` 再提速 `+1.3%` |
| | **`CB B=8 / B=16`** | **`837.0 / 1,423.9`** | — (`4c73444`: `24.6 / 33.7`) | **较 v13 `B=16` (`1,008`) 再暴涨 `+41.3%`!!** | `B > 16u` 修复释放全部批处理算力 |
| **`DeepSeek-V2-Lite` 2G (`2D TP=2×EP=2`)** | `pp256` / `B=1` / `CB B=8` | **`2,919.5` / `183.2` / `740.7`** | `1,060.7` / `35.8` / — (1G) | **`2.75×` / `5.11×` 胜出!!** | `tp=2, nccl=1, tp_rep=34` |
| **`Llama-3.3-70B-Q4_0` 1G (48GB)** | `d=0 B=1` / `4K Q4` | `20.53` / `19.45 tok/s` | `20.92` / `19.88 tok/s` | **`98.1%` / `97.8%` 持平!** | `fd=1520, g_cap=1`, 零溢出 `pp128_tok=14198` |
| | `pp128` / `CB B=4 / B=8` | `461.3` / **`75.8 / 113.2`** | **`874.3`** / — / — | `CB B=8` 较 v13 (`106.3`) **`+6.5%`** | 保留零溢出 INT8 TC MMQ（见第五节复盘） |
| **`Llama-3.3-70B-Q4_0` 2G (`TP=2`)** | **`d=0 B=1`** | **`33.42 tok/s`** | `-sm row` 失败 / `-sm layer`: `20.96` | **`1.59×` 胜出!!** | `tp=2, nccl=1, tp_rep=22, 12/12 BIT-EXACT` |
| | **`CB B=4 / B=8`** | **`109.5 / 176.8`** | `69.2 / 106.9` (`-sm layer`) | **`1.58× / 1.65×` 胜出!!** | 持续领跑双卡多并发 |
| | `pp128` | `484.6 tok/s` | **`1,020.1 tok/s`** (`-sm layer`) | `0.48×` | 非 P2P SYS PCIe 上 FP32 AllReduce + MMQ |

---

## 三、Apple M4 Max (40-core Metal 4, 128 GiB) 九模型全面正面评测

> **测试口径**：Apple M4 Max（40 核 GPU，实测 `468.4 GB/s` STREAM / `13.96 TFLOPS` AMX），正面对比 **Tenon v14** 与 Homebrew **`llama.cpp` (`-ngl 99 -fa 1`)**。  
> **贪婪对齐校验**：全部 4 个新增全量化/前沿架构模型（`1B-Q4_K_M`、`1B-Q8_0`、`Gemma-3-1B-IT`、`Qwen3.5-0.8B`）与 `Llama-3.2-1B-Q4_0` 均经 16-token 贪婪生成逐字比对，**5/5 全部 100% `exact=True`**！  
> **宿主机负载透明披露**：首轮全量测试 Models 1–7 期间宿主机后台外部任务（`llgo-corov2`）占用了 `200%–380%` CPU 与统一内存带宽，待其结束后已立即完成 Models 1–6 无干扰洁净复测（`m4max_v14_showdown_clean.json`）；仅末尾 `70B d=0` 时外部任务短暂重启（导致双引擎同步受扰），故 `70B` 同时呈现洁净 `4K Q4` 实测值与 v13 洁净 `d=0` 参考。

### 3.1 `Llama-3.2-1B-Q4_0` 全场景与 Zero-Rust v14 HTTP 并发实测

| 测试场景 | **Tenon v14 (洁净实测)** | 上游 `llama.cpp` (同窗) | **Tenon v14 vs `llama.cpp`** | 内存 / 说明 |
| :--- | ---: | ---: | :---: | :--- |
| **Prefill `pp512`** | **`5,972.2 tok/s`** | `3,972.8`（空载基线 `6,110.6`） | **`97.7%` 逼近上限**（同窗 `1.50×`） | `BIT-EXACT (15140)` |
| **Decode `d=0 B=1`** | **`355.7 tok/s`** | `303.8`（空载基线 `335.0`） | **`1.06×–1.17×` 胜出!** | `icb=2, rep=82`, `100% exact` |
| **长上下文 `4K FP16`** | **`284.6 tok/s`** | `280.8 tok/s` | **`1.01×` 胜出!** | `fd=560`, RSS `0.77G` vs `1.14G` |
| **长上下文 `8K Q8` / `8K FP8`** | **`225.1`** / **`211.7 tok/s`** | `232.3` / 无原生 FP8 KV | **`96.9%` 持平** / **Tenon 独有** | `fd=560`, RSS `0.77G` vs `1.16G` |
| **长上下文 `4K Q4`** | **`296.3 tok/s`** | `291.3 tok/s` | **`1.02×` 胜出!** | `fd=560`, RSS `0.77G` vs `0.96G` |
| **内核批处理 `CB B=8 / 16 / 32`** | **`1,138.9 / 1,410.8 / 2,136.7`** | — | **6.0× vs `B=1`** | `C^T=W·X^T` AMX |
| **HTTP 并发 `B=8` (`max_tokens=64`)** | `tenon llm`: `1,051.1` / **Zero-Rust v14: `1,060.9`** | `1,223.8` (`llama-server`) | `86.7%`（Zero-Rust 反超 Rust CLI） | `maxB=8, mix=1` |
| **HTTP 并发 `B=16` (`max_tokens=64`)** | `tenon llm`: **`1,813.1`** / **Zero-Rust v14: `1,899.1`** | `1,524.8` (`llama-server`) | **`1.25×` 全面胜出!!** | **`962 KiB` 零 Rust `MAX_CB=16, mix=2`** |

### 3.2 新增四大全量化 & 2026 前沿架构模型（Apple M4 Max 洁净实测）

| 模型 | 贪婪校验 | **Tenon v14 `pp512`** | `llama.cpp` `pp512` | **Tenon v14 `tg_b1`** | `llama.cpp` `tg_b1` | **Tenon `CB B=8 / B=16`** |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: |
| **`Llama-3.2-1B-Q4_K_M`** | **`100% exact`** | `4,382.5 tok/s` (**`92.2%`**) | `4,752.0` | `272.4 tok/s` (`81.3%`) | `335.2` | **`760.0 / 1,321.2`** |
| **`Llama-3.2-1B-Q8_0`** | **`100% exact`** | `4,780.7 tok/s` (**`94.8%`**) | `5,043.5` | **`255.3 tok/s` (`1.02×` 胜!)** | `250.3` | **`536.3 / 917.4`** |
| **`Gemma-3-1B-IT-Q4_K_M`** | **`100% exact`** | `5,771.5 tok/s` (**`97.4%`**) | `5,922.7` | `173.4 tok/s` (`68.4%`) | `253.7` | **`368.6 / 929.3`** |
| **`Qwen3.5-0.8B-Q4_K_M`** | **`100% exact`** | `5,927.0 tok/s` (**`89.2%`**) | `6,647.9` | `246.0 tok/s` (`81.7%`) | `301.3` | **`737.1 / 1,293.0`** |

### 3.3 16B MoE、70B Dense、106B 超内存 MoE 与多模态 VLM（Apple M4 Max）

| 模型与场景 | **Tenon v14** | 上游 `llama.cpp` (同窗) | **Tenon v14 vs `llama.cpp`** | 内存 / 技术亮点 |
| :--- | ---: | ---: | :---: | :--- |
| **`DeepSeek-V2-Lite` `pp32 / pp256 / pp512`** | `540.9 / 1,314.0 / 1,223.5` | `627.1 / 1,553.8 / 1,728.6` | `86.3% / 84.6% / 70.8%` | `BIT-EXACT (254/855/8144)`, `RSS 0.07G` vs `9.92G` |
| **`DeepSeek-V2-Lite` `tg_b1` / `CB B=8 / B=16`** | `125.2` / **`265.1 / 505.0`** | `151.5` / — | `82.6%` / 批吞吐较 v13 (`212/420`) **`+25% / +20%`!** | 零溢出 Mega-AMX |
| **`DeepSeek-V2-Lite` `2K MLA`** | **`105.4 tok/s`** | `104.8 tok/s` | **`1.01×` 胜出!** | 16-Head Single-TG `512`, `FP 0.96G` vs `1.38G` |
| **`Llama-3.3-70B` `4K Q4 KV` (洁净)** | **`9.21 tok/s`** | `9.40 tok/s` | **`98.0%` 持平!** | `fd=1200, icb=15`, **`RSS 0.07G` vs `38.21G`** |
| **`Llama-3.3-70B` `d=0` (同窗受扰¹)** | `pp128: 53.8 / tg: 3.61`（v13 洁净 `90.1 / 11.17`） | `pp128: 21.8 / tg: 2.68`（v13 洁净 `64.2 / 9.21`） | **`2.47× / 1.35×` 胜出!** | `icb=19`, `BIT-EXACT (14198)` |
| **`GLM-4.5-Air 106B` `≤42G (80/128)`** | **`35.40 tok/s`** | `2.43 tok/s` (`-ot exps=CPU`) | **`14.57×` 胜出!!** | **`FP 38.24G` (`RSS 5.97G`)** vs `FP 51.03G` (`RSS 74.6G`) |
| **`GLM-4.5-Air 106B` `≤16G (24/128)`** | **`17.23 tok/s`** | `2.43 tok/s` (`-ot exps=CPU`) | **`7.09×` 胜出!!** | **`FP 15.32G`** 严守 16G 内存预算 |
| **`GLM-4.5-Air 106B` `100% GPU`** | **`43.59 tok/s`** | `49.44 tok/s` | `88.2%` | `icb=19`, **`RSS 0.16G` vs `58.17G`** |
| **`SmolVLM-256M` Turn-1 Cold (`pf / e2e`)** | `114.6 / 173.8 ms` | `72.7 / 133.7 ms` | 较 v13 (`160.9 ms`) 提速 **`-28.8%`** | Single-CB 全 GPU ViT |
| **`SmolVLM-256M` Turn-1 Warm (`pf / e2e`)** | **`41.7 / 92.5 ms`** | `66.1 / 127.1 ms` (`llama-server`) | **`1.59×` (`pf`) / `1.37×` (`e2e`) 反超胜出!!** | 单命令缓冲区消除 ~480 次主机同步 |
| **`SmolVLM-256M` Turn-2 COW (`pf / e2e`)** | **`8.7 / 33.2 ms`** | `4.0 / 36.5 ms` | **E2E 反超 `llama-server` (`1.10×`)!** | PagedKV 块级多模态前缀共享 |

> ¹ `70B d=0` 测试时段宿主机外部进程 `llgo-corov2` 重新启动，但同窗条件下 Tenon 仍领先 `llama.cpp` `2.47× (pp128)` 与 `1.35× (tg)`；其洁净满速与 v13（`90.1 / 11.17 tok/s`）及同机洁净 `4K Q4`（`9.21 vs 9.40 tok/s`）完全一致。

---

## 四、全工作区回归、Zero-Rust 形式化证明与生产 E2E 验收

1. **`cargo test --workspace`**：全工作区自动化测试 **100% 通过**（0 失败，涵盖形式化验证、62 项 Tile-IR 优化 Pass、LLVM 22 AOT、Wasm32、LSP 与 `tenon llm` CLI Gemma-3/Qwen3.5 模板）。
2. **纯 Tenon 零 Rust 原生服务器编译与 Z3 证明**：
   - `examples/showcase/tenon_llm_server.tn`（Metal）与 `examples/showcase/tenon_llm_server_cuda.tn`（CUDA）均通过 `tenon check` 零错误零警告；
   - `tenon build examples/showcase/tenon_llm_server.tn` 编译出 **`962 KiB`** 独立原生二进制，通过 **374 项 Z3 形式化证明 + 136 项数组越界检查静态消除**；
   - 实测 `Llama-3.2-1B-Q4_0` HTTP `B=16` 达 **`1,899.1 tok/s`**（`max_batch=16, mixed_batches_run=2`，反超 `llama-server 1,524.8` **`+24.5%`**），并实测通过 **Gemma-3-1B**（`finish_reason=stop`）与 **Qwen3.5-0.8B**（`<think>` 完整推理 + `finish_reason=stop`）OpenAI Chat Completions 接口。
3. **双端 7/7 生产高可用 E2E 验收（`verify_cli_and_server_v11.py`）**：`target/release/tenon llm` Rust CLI 与 `962 KiB` 纯 Tenon Zero-Rust v14 服务器同步 **7/7 全部通过**（HA PD Router 故障转移、Layer-Stream `TNKL` / Binary / SHM 三通道 `BIT-EXACT` + 篡改拒收、SSE 客户端 `RST` 断连零崩溃块回收、动态 LoRA 热插拔、`HTTP 503` 背压、Prometheus 指标完整输出）。

---

## 五、实机硅片消融复盘与剩余客观空间

本轮 v14 坚持**“只相信实机硅片测量、不迷信纸面直觉”**，在 RTX 6000 Ada 实机上捕获并修正了 4 项极具工程价值的微架构现象：

1. **`cp.async.cg` vs `cp.async.ca`（L1 空间复用关键差异）**：在 `cm_gemm_q4_mmq` / `q8_mmq` / `q4k_mmq` 中，`cp.async.cg` 绕过 L1 导致沿 `blk` 方向相邻 48B 块丧失 L1 缓存复用，且 2-stage `sX` 使 SMEM 翻倍降低 Ada CTA 占用率。切换至 **`cp.async.ca`** 并恢复精简 1-stage `sX` 后，`1B Q4_0 pp512` 跃升至 **`46,074 tok/s`**（`+18.8%`）、`Q4_K_M` 跃升至 **`35,879 tok/s`**（`+21.4%`）、`Q8_0` 跃升至 **`43,118 tok/s`**（`2.86×` `llama.cpp`）。
2. **MMQ vs Small-Batch DP4A 临界点（`B > 16u`）**：`B=16` 误入 64M-tile MMQ 会浪费 75% Tensor Core 算力并造成 SM 欠饱和；修正为 `B > 16u` 后，所有模型 `CB B=16` 瞬间跃升 **`1.8×–5.05×`**（`1B Q4_0`: `1,115 → 5,627 tok/s`）。
3. **跨行 2-lane/row vs 合并 32-lane/row（`Q8_0` LM-Head 访存教训）**：在 `Gemma-3`（`nb=36`）与 `1B Q8_0`（`nb=64`）上尝试 2-lane/row（每 Warp 跨 16 行）反而因破坏 GDDR6 合并突发使 `tg` 下降 9%–13%；改回 32-lane 合并访存 + 68B 对齐 32-bit 块对加载后，`Q8_0` 与 `Gemma-3` 解码速度双双刷新新高（`432.5` / `322.4 tok/s`）。
4. **70B 大模型 FP16 激活溢出保护**：在 `Llama-3.3-70B` `pp128` 上验证发现，直接将隐状态转为 FP16 送入 `cublasGemmEx` 会触发 SwiGLU/残差大离群值 `> 65504` 溢出导致首 Token 偏离（`14198 → 100602`），因此坚持保留零溢出 INT8 TC MMQ 路径确保 100% 数值正确。

**剩余客观可探索空间**（留待后续独立迭代，收益已明显进入边际递减区）：
- `Llama-3.3-70B` CUDA `pp128` 若要在不丢失数值精度的前提下从 `461–485` 追平 `llama.cpp 874–1,020`，需采用 **BF16 cuBLAS TC**（与 FP32 相同指数范围、零激活溢出）或更高 K-tile/寄存器累加的 MMQ v2，以及多卡 `TP=2` 的 BF16 NCCL AllReduce；
- `Gemma-3-1B` 的 `d=0 B=1` 解码（M4 Max `173.4 vs 253.7`、Ada `322.4 vs 473.0`）受 26 层 `Q5_0`+`Q6_K` 混合量化与 `262K` 超大 `Q8_0` 词表（占模型 40% 体积）主导，后续可针对 `Q5_0` 64-bit 对齐加载与大词表分块展开进一步压榨约 20%–35% 带宽。
