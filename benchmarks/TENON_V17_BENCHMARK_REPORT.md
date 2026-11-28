# Tenon v17 硅片极限 MoE Overdrive、13.5 KiB 零冲突 `ldmatrix.x4` MMQ、引擎级 Radix APC 与原生 Linux CUDA 交叉编译实测报告

> **版本**: Tenon v17（2026 Frontier MoE Overdrive + Bank-Conflict-Free `ldmatrix.x4` Q4 MMQ + Engine-Level Radix APC + Native `linux-x86_64-cuda` CLI + Metal 4 8-SG MXFP4 + DeepSeek-V3.2 DSA + Blackwell `sm_100`） · **评测日期**: 2026-11-28  
> **核心目标**: 面向 2026 真实大权重 MoE/混合架构模型、单/双卡 70B Roofline、多轮 Agent 生产服务与原生跨平台交付，全面落地四大支柱并完成 **Apple M4 Max (Metal 4)**、**RunPod `2× NVIDIA RTX 6000 Ada 48GB` (`sm_89`)** 与 **RunPod `2× NVIDIA L40S 48GB` (`sm_89`)** 三平台真实权重实机验收：
> 1. **Pillar 1（CUDA 2026 前沿 MoE Overdrive）**：`Q8_0` 奇数块（`in_dim % 32 == 0`）MMQ/LM-Head、`cm_gemm_q5k_mmq`、`B<8` 原生 `Q8_1 + DP4A` 分组 MoE（`MXFP4=39` 无分支 `prmt` 精确 `2×` E2M1 解包 `cm_mxfp4_unpack4_s8` + `Q4_K`/`Q5_K` `nb16` 100% Warp 满载）、`32M×64N×32K` 零 Bank 冲突 `ldmatrix.x4` INT8 TC 分组 MoE Mega-Kernels（覆盖 `MXFP4`/`Q4_K`/`Q5_K`/`Q4_0`/`Q4_1`/`Q5_0`/`Q8_0`）、路由排序提升至 `B*top_k<=4096` + `chunk_sz=512`，以及 `cm_gdn_prep_qkgb` + 8-col/CTA `cm_gdn_step_batch`（`1G` 与 `TP` 全路径）+ 内容哈希磁盘 Cubin 缓存（重复冷启动 `62–145s → 0.29–2.30s`）。
> 2. **Pillar 2（CUDA `70B` Bank-Conflict-Free `ldmatrix.x4` Q4_0 MMQ Roofline）**：16B XOR Swizzle 零 Bank 冲突 `ldmatrix.sync.aligned.m8n8.x4` `128M×64N`（`cm_gemm_q4_mmq_m128`, 13.5 KiB SMEM, 97 regs, 0 spill）与 `128M×128N`（`cm_gemm_q4_mmq_m128_128`, 18.0 KiB SMEM, 164 regs, 0 spill）+ 确定性 Split-K，经 `70B 1G/TP=2` 与 `1B` 实机 A/B 消融定型 `m128_64` 为默认主路径。
> 3. **Pillar 3（多轮 Agent 零 Rust 生产服务器 + 原生 `linux-x86_64-cuda` CLI 交叉编译）**：双平台（Metal & CUDA）引擎级内容哈希 Radix PagedKV 前缀缓存（**APC**，`refcnt==0` LRU 驻留、跨 24-slot 逐出存活）+ Prometheus `/metrics` 扩展（APC 全指标 + `TTFT`/`TPOT` `p50/p95/p99` 分位数）+ `/health` JSON，以及编译器原生 `tenon build --target linux-x86_64-cuda --opt O3` 一键输出 Linux CUDA 交叉包（`.o` + `_rt.c` + `_link.sh`）。
> 4. **Pillar 4（Apple M4 Max MoE 极限 + DeepSeek-V3.2 DSA + Blackwell `sm_100`）**：Metal 4 `32N×64M`/`64N×64M` 8-SG 零冲突填充 `MXFP4` AMX + `chunk_sz=512` + `DSV2` 融合 `wq||wkv_a` & `moe_mla_rope_store_b1` + 双平台 DeepSeek-V3.2 **DSA**（Lightning Indexer Top-K + Sparse MLA）+ NVIDIA Blackwell `sm_100`（`cc>=100` `-arch=sm_100a` 编译与探测）。
>
> **科学诚实口径**：全部数据均为物理硬件实测（Apple M4 Max 40-core Metal 4 + RunPod `2× RTX 6000 Ada 48GB` R1 + RunPod `2× L40S 48GB` R2），完整公开全部 16 个新核 `ptxas` 寄存器/零溢出审计表、A/B 消融胜负结论与 `apc_http` 逐出细节。

---

## 一、Tenon v17 四大支柱核心突破总览

| 支柱与核心实现 | Apple M4 Max（Metal 4, `468.4 GB/s` 实测） | NVIDIA `sm_89`（`2× RTX 6000 Ada 48GB` R1 & `2× L40S 48GB` R2） |
| :--- | :--- | :--- |
| **Pillar 1：2026 前沿 MoE Overdrive**<br>`std/gpu/cuda/moe_stream_cuda_shim.c` | `gpt-oss-20b-MXFP4` **`16/16 BIT-EXACT`**，**`tg_b1 = 133.87`**（超 `llama.cpp 114.34` **`+17.1%`**），**`pp128 = 926.9`**（`+10.3%`，达 `llama.cpp 931.9` 的 **`99.5%`**），**`pp512 = 1,227.1`**（**较 v16 `+20.7%`!**） | **真实大权重双双大幅跃升（`16/16` 一致）**：<br>• **OpenAI `gpt-oss-20b-MXFP4` (12.11 GB)**：RTX 6000 Ada **`tg_b1 171.1 → 229.69` (`+34.3%`!, 达 `llama.cpp 249.7` 的 `92.0%`!)**、`pp128 1,818 → 2,637` (`+45.0%`)、`pp512 2,488 → 3,704` (`+48.9%`)、`CB B=16 1,094 → 1,338` (`+22.3%`)；L40S（FP16 KV + `attn_sinks`）**`pp512 = 4,265.6` (`+71.5% / 1.71×` vs v16!)**、`load_s = 1.97s`<br>• **Alibaba `Qwen3.5-35B-A3B-Q4_K_M` (22.02 GB, 40L GDN+256E)**：RTX 6000 Ada **`16/16 BIT-EXACT`**，**`tg_b1 98.8 → 122.04` (`+23.5%`!)**、`pp128 1,604 → 2,214` (`+38.0%`)、**`pp512 2,150 → 3,587` (`+66.8%`!)**、`CB B=16 888 → 945` (`+6.4%`)；L40S `load_s = 2.30s`（磁盘 Cubin 缓存提速 `63×`） |
| **Pillar 2：`70B` 13.5 KiB `ldmatrix.x4` MMQ**<br>`std/gpu/cuda/moe_stream_cuda_shim.c` | `Llama-3.2-1B-Q4_0` `pp512 = 6,052.3`、`tg = 377.1`（零回归） | **`13.5 KiB` XOR-Swizzle `ldmatrix.x4` `m128_64`（97 regs, 0 spill）刷新 `1G` 与 `TP=2` 双纪录**：<br>• **`70B 1G`（RTX 6000 Ada）**：`pp128 581.3 →` **`629.25` (`+8.3%`)**、`pp256 549.6 →` **`625.39` (`+13.8%`)**、`CB B=8 109.3 →` **`113.78` (`+4.1%`)**（`16/16 BIT-EXACT`）<br>• **`70B 1G`（L40S）**：**`pp128 = 640.43` (`+10.2%`)**、**`pp256 = 639.17` (`+16.3%`)**、**`CB B=8 = 119.79` (`+9.6%`)**<br>• **`70B TP=2`（`2× L40S`, `p2p=0` 非 P2P）**：**`pp128 = 985.10`（较 v16 非 P2P `834.2` 再暴涨 `+18.1%`! 甚至超越 v16 `p2p=1` `943.8` 与 `llama.cpp 1G 905.0`!!）**、**`pp256 = 979.94` (`+17.6%`)**（`tok=14198/927` 全员比特一致）<br>• **`1B Q4_0`（L40S）**：**`pp512 = 46,859.0`**（`2.92×` `llama.cpp 16,072`） |
| **Pillar 3：引擎级 Radix APC + 原生 `linux-x86_64-cuda` CLI**<br>`crates/tenon-{codegen,cli}/`<br>`examples/showcase/tenon_llm_server*.tn` | **纯 Tenon 零 Rust Metal v17 服务器**（`1,101.6 KiB`，**378 项 Z3 证明 + 139 项越界消除 + 100 项零溢出**）：<br>• 25-slot 全量逐出后 HTTP Radix APC 命中 **`496` Token**，**`TTFT 95.45ms → 11.54ms` (`8.27×` 加速!, `BIT-EXACT=True`)**<br>• Prometheus `/metrics` `p50/p95/p99` 分位数与 APC 全指标全绿<br>• **双端 7/7 = 14/14 HA E2E 零回归全通过** | **原生 `tenon build --target linux-x86_64-cuda --opt O3`** 直出 Linux CUDA 交叉包（**378 项 Z3 证明**），链接出 **`1,366.2 KiB` stripped 原生 ELF**（`ldd` 仅 `libc.so.6` + `libm.so.6`）：<br>• **`HTTP B=8`**：**`3,420.94 tok/s`**（较 v16 `2,650.4` **`+29.1%`**，反超 `llama-server 2,249.2` **`1.52×`!**）<br>• **`HTTP B=16`**：**`4,638.09 tok/s`**（较 v16 `4,446.1` **`+4.3%`**，反超 `llama-server 2,647.2` **`1.75×`!!**）<br>• 引擎级 APC 逐出存活实测 **`bit_exact=true` (`64/64`)**；HTTP 77 并发下实测 `hits=6 / saved=768 / evictions=18`、`TTFT p50/p95/p99 = 5.75 / 25.94 / 45.19 ms`、`TPOT p50/p95/p99 = 2.78 / 4.78 / 4.96 ms` |
| **Pillar 4：M4 Max MoE + DeepSeek-V3.2 DSA + Blackwell `sm_100`**<br>`std/gpu/metal/moe_stream_shim.m`<br>`std/gpu/cuda/moe_stream_cuda_shim.c` | • `DeepSeek-V2-Lite` **`pp512 1,235.7 → 1,435.6` (`+16.2%`!)**、`tg_icb = 141.01`（`+7.5%` vs v16 稳态 `131.2`）<br>• Metal 4 **DeepSeek-V3.2 DSA**（Lightning Indexer Top-K + Sparse MLA）自检 vs CPU 参考 **`256×32 = PASS, 512×64 = PASS`** | • CUDA **DeepSeek-V3.2 DSA**（`cm_dsa_indexer_topk` + `cm_flash_mla_dsa_sparse`, 40 regs, 0 spill）自检 **`256×32 = PASS, 512×64 = PASS`**<br>• NVIDIA Blackwell `sm_100` 原生支持（`cc >= 100` 自动 `-arch=sm_100a` 含 `sm_100` 回退、`tn_cumoe_info(h, 61)` + `cm_mma_s8_m16n8k32` 钩子）<br>• `cargo test --workspace` **104/104 全绿**（新增 4 项 `linux-x86_64-cuda` 目标单元测试） |

---

## 二、NVIDIA `sm_89`（`2× RTX 6000 Ada` R1 & `2× L40S` R2）全面实测

> **硬件说明**：
> - **R1（主基线 & A/B 消融）**：RunPod **`2× NVIDIA RTX 6000 Ada Generation`**（`48 GiB GDDR6 ECC × 2`, `960 GB/s`, `sm_89`, `142 SMs`）。
> - **R2（终版洁净全测 + `ptxas` + Zero-Rust Linux CUDA v17 服务器 + 磁盘 Cubin 缓存）**：RunPod **`2× NVIDIA L40S`**（`48 GiB GDDR6 ECC × 2`, `864 GB/s`, `sm_89`, `142 SMs`, 更高 Boost 频率 `2.52 GHz`）。

### 2.1 2026 真实大权重前沿 MoE / 混合架构模型演进（`gpt-oss-20b-MXFP4` & `Qwen3.5-35B-A3B`）

| 真实 HuggingFace GGUF 模型 | 版本 / 硬件配置 | 一致性校验 | `pp128` (`tok/s`) | `pp512` (`tok/s`) | `tg_b1` (`tok/s`) | `CB B=8` | `CB B=16` | 冷启动 `load_s` |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **OpenAI `gpt-oss-20b-MXFP4.gguf`**<br>（12.11 GB, 24L, 32E Top-4,<br>官方 17B `MXFP4=39` + `attn_sinks` + `SWIGLU_OAI`） | Tenon v16 R4 (`RTX 6000 Ada`) | `16/16` | `1,818.0` | `2,487.7` | `171.05` | `604.5` | `1,093.6` | `~60s` |
| | **Tenon v17 R1 (`RTX 6000 Ada`, `Q8` KV)** | **`16/16` vs ref** | **`2,636.98`** (`+45.0%`) | **`3,704.14`** (`+48.9%`) | **`229.69`** (**`+34.3%`!!**) | **`735.49`** (`+21.7%`) | **`1,337.51`** (`+22.3%`) | `62.5s` |
| | **Tenon v17 R2 (`L40S`, FP16 KV + `attn_sinks`)** | `pp512_tok=279` exact | `2,495.99` (`+37.3%`) | **`4,265.56`** (**`+71.5% / 1.71×`!!**) | `199.96` (`864 GB/s`) | `696.29` | `1,259.62` | **`1.97s`** (`31.8×`!) |
| | 上游 `llama.cpp` (`8e23278`, `RTX 6000 Ada`) | 参考真值 | `5,075.2` | `10,987.4` | `249.71` | — | — | — |
| **Alibaba `Qwen3.5-35B-A3B-Q4_K_M.gguf`**<br>（22.02 GB, 40L, 30×GDN + 10×Gated GQA + 256E Top-8 MoE `Q4_K`/`Q5_K`） | Tenon v16 R4 (`RTX 6000 Ada`) | `16/16 BIT-EXACT` | `1,603.6` | `2,149.7` | `98.83` | `521.1` | `887.9` | `~140s` |
| | Tenon v17 R1 仅 GDN 优化（`MOE_MMQ=0` 消融） | `16/16 BIT-EXACT` | `1,694.2` | `2,702.0` | `115.57` (`+16.9%`) | — | — | — |
| | **Tenon v17 R1 (`RTX 6000 Ada`, 全量开启)** | **`16/16 BIT-EXACT`** | **`2,213.56`** (`+38.0%`) | **`3,586.65`** (**`+66.8%`!!**) | **`122.04`** (**`+23.5%`!!**) | **`557.44`** (`+7.0%`) | **`944.66`** (`+6.4%`) | `144.7s` |
| | **Tenon v17 R2 (`L40S`, FP16 KV + Cubin 缓存)** | **`16/16 BIT-EXACT`** | `2,059.81` | `3,511.93` | `110.40` (`864 GB/s`) | `511.01` | `855.72` | **`2.30s`** (`62.9×`!) |
| | 上游 `llama.cpp` (`8e23278`, `RTX 6000 Ada`) | 参考真值 | `4,150.9` | `9,078.6` | `190.67` | — | — | — |

### 2.2 `Llama-3.3-70B-Instruct-Q4_0`（40.12 GB, 80 层）13.5 KiB `ldmatrix.x4` MMQ 实测与消融

| 拓扑与配置（`sm_89`） | `pp128` (`tok/s`) | `pp256` (`tok/s`) | `tg_b1` (`tok/s`) | `CB B=8` (`tok/s`) | 一致性与说明 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Tenon v16 `1G` (`19.5 KiB m128`, `RTX 6000 Ada`) | `581.27` | `549.61` | `19.50` | `109.29` | `tok=14198 / 927` |
| Tenon v17 R1 `1G` `m128_128` 消融 (`18.0 KiB, 164 regs`, `RTX 6000 Ada`) | `609.38` | `608.64` | `19.52` | `113.78` | `16/16` 一致，但 164 regs 限制占空比 |
| **Tenon v17 R1 `1G` `m128_64` 默认 (`13.5 KiB, 97 regs`, `RTX 6000 Ada`)** | **`629.25`** (`+8.3%`) | **`625.39`** (`+13.8%`) | **`19.53`** | **`113.78`** (`+4.1%`) | **`16/16 BIT-EXACT`** (`tok=14198 / 927`) |
| **Tenon v17 R2 `1G` `m128_64` 默认 (`NVIDIA L40S`)** | **`640.43`** (`+10.2%`) | **`639.17`** (`+16.3%`) | `16.13` (`864 GB/s`) | **`119.79`** (`+9.6%`) | **`16/16 BIT-EXACT`** (`tok=14198 / 927`) |
| 上游 `llama.cpp 1G`（`RTX 6000 Ada`, `-sm none`） | `904.96` | `990.39` | `21.22` | — | 单卡满载基线 |
| Tenon v16 `TP=2` (`p2p=0`, `2× RTX 6000 Ada`) | `834.15` | `833.14` | `31.56` | `191.71` | v16 非 P2P 基线 |
| Tenon v16 `TP=2` (`p2p=1`, `2× RTX 6000 Ada`) | `943.84` | — | `33.16` | `206.14` | v16 P2P 基线 |
| **Tenon v17 R2 `TP=2` (`p2p=0` 非 P2P, `2× L40S`, `m128_64`)** | **`985.10`** (**`+18.1%` vs v16 非 P2P!!**) | **`979.94`** (**`+17.6%`!!**) | `27.56` (`864 GB/s`) | `180.20` | **`tok=14198 / 927` & `12/12 toks` 比特一致，反超 `llama.cpp 1G` (`+8.9%`)!** |

### 2.3 原生 `tenon build --target linux-x86_64-cuda` 零 Rust 服务器压测（`Llama-3.2-1B-Q4_0`, `L40S`）

| 指标 | **Tenon v17 Zero-Rust Linux CUDA Server** | Tenon v16 (`RTX 6000 Ada`) | 上游 **`llama-server`** (`8e23278`) | **v17 vs `llama-server`** |
| :--- | ---: | ---: | ---: | :---: |
| **构建方式 / ELF 体积 / 动态库** | **`tenon build --target linux-x86_64-cuda --opt O3`**<br>**`1,366.2 KiB` stripped**（仅 `libc.so.6` + `libm.so.6`，**378 项 Z3 证明**） | 手动交叉编译 `1,274.3 KiB` | `> 60 MiB`（含 `libcublas`/`libcudart`） | **原生 CLI 一键构建，`~45×` 更轻量** |
| **并发 `HTTP B=8`（`max_tokens=64`）** | **`3,420.94 tok/s`** | `2,650.41 tok/s` | `2,249.15 tok/s` | **`1.52×` (`+52.1%`!)** |
| **并发 `HTTP B=16`（`max_tokens=64`）** | **`4,638.09 tok/s`** | `4,446.10 tok/s` | `2,647.18 tok/s` | **`1.75×` (`+75.2%`!!)** |
| **底层内核 `pp512` / `load_s`** | **`46,859.04 tok/s`** (`tok=15140`, `16/16`) / **`0.288s`** | `46,928.2 tok/s` / `~30s` | `16,071.7 tok/s` | **`2.92×` (`pp512`)** |
| **引擎级 Radix APC（底层逐出存活）** | **`cached_blks=4, matched_after_free=64, ref_tok=6666==apc_tok=6666, bit_exact=true`** | 不支持跨 slot 逐出 | — | **100% 比特一致** |
| **HTTP 77 并发 APC + Prometheus v17** | `hits=6`, `saved_tokens=768`, `evictions=18`（25-slot 逐出后 `35.60 → 22.46ms, 1.59×`）<br>`TTFT p50/p95/p99 = 5.75 / 25.94 / 45.19 ms`<br>`TPOT p50/p95/p99 = 2.78 / 4.78 / 4.96 ms` | 仅直方图桶，无分位数/APC | — | **全指标实机跑通** |

### 2.4 全部 16 个 v17 新 CUDA 内核 `ptxas --ptxas-options=-v` 寄存器与零溢出审计（`sm_89`）

| CUDA 内核名 | 寄存器/线程 | 共享内存 (`smem`) | 栈帧 (`stack`) | Spill Stores / Loads | 结论 |
| :--- | ---: | ---: | ---: | :---: | :--- |
| `cm_gemm_q4_mmq_m128`（默认 `128M×64N`） | **97** | **13,824 B (13.5 KiB)** | `0 B` | **`0 / 0`** | 零溢出高占空比，全面胜出 |
| `cm_gemm_q4_mmq_m128_128`（可选 `128M×128N`） | 164 | 18,432 B (18.0 KiB) | `0 B` | **`0 / 0`** | 零溢出，但 164 regs 限制 CTA/SM |
| `cm_gemm_q5k_mmq` | **96** | 19,456 B | `0 B` | **`0 / 0`** | 零溢出 |
| `cm_gemm_q8_mmq`（支持奇数 `nb`） | **100** | 25,600 B | `0 B` | **`0 / 0`** | 零溢出 |
| `cm_moe_grouped_gu_swiglu_dp4a` | **64** | `0 B` | `0 B` | **`0 / 0`** | 满占空比零溢出 |
| `cm_moe_grouped_down_dp4a` | **68** | `0 B` | `0 B` | **`0 / 0`** | 满占空比零溢出 |
| `cm_moe_grouped_gu_oai_dp4a` | **71** | `0 B` | `0 B` | **`0 / 0`** | 无分支 `prmt` MXFP4 零溢出 |
| `cm_moe_grouped_down_bias_dp4a` | **70** | `0 B` | `0 B` | **`0 / 0`** | 零溢出 |
| `cm_moe_grouped_gu_swiglu_mmq` | **96** | 6,400 B | `48 B` | **`0 / 0`** | 零寄存器溢出 |
| `cm_moe_grouped_down_mmq` | **80** | 3,840 B | `32 B` | **`0 / 0`** | 零寄存器溢出 |
| `cm_moe_grouped_gu_oai_mmq` | **100** | 6,400 B | `48 B` | **`0 / 0`** | 零寄存器溢出 |
| `cm_moe_grouped_down_bias_mmq` | **86** | 3,840 B | `32 B` | **`0 / 0`** | 零寄存器溢出 |
| `cm_gdn_prep_qkgb` | **20** | 32 B | `0 B` | **`0 / 0`** | 极致轻量零溢出 |
| `cm_gdn_step_batch`（8-col/CTA） | **46** | 256 B | `0 B` | **`0 / 0`** | 零溢出 |
| `cm_dsa_indexer_topk` | **40** | 64 B | `0 B` | **`0 / 0`** | `256×32` & `512×64` PASS |
| `cm_flash_mla_dsa_sparse` | **40** | 16 B | `0 B` | **`0 / 0`** | `256×32` & `512×64` PASS |

---

## 三、Apple M4 Max（40-core Metal 4, 128 GiB）全面实测与双端生产 HA E2E

| 评测项（Apple M4 Max Metal 4） | **Tenon v16D 基线** | **Tenon v17 实测** | 上游 `llama.cpp` | **v17 提升与亮点** |
| :--- | ---: | ---: | ---: | :--- |
| **`gpt-oss-20b-MXFP4`（12.11 GB）** `tg_b1` / `pp128` / `pp512` / `CB32` | `133.86` / `840.2` / `1,016.8` / `637.2` | **`133.87` / `926.9` / `1,227.1` / `613.5`** | `114.34` / `931.9` / `1,619.0` | **`16/16 BIT-EXACT`**；`tg_b1` 领先 `llama.cpp` **`+17.1%`**，8-SG 零冲突填充 AMX 令 **`pp128 +10.3%`（达 `99.5%` `llama.cpp`!）**、**`pp512 +20.7%`!!** |
| **`DeepSeek-V2-Lite-Chat.Q4_K_M`（16B MLA+MoE）** `tg_icb` / `pp512` | `131.2`（稳态） / `1,235.7` | **`141.01` / `1,435.6`** | `159.9` / `1,726.8` | **`16/16 BIT-EXACT`**；融合 `wq||wkv_a` + `moe_mla_rope_store_b1` + `chunk_sz=512` 令 **`pp512 +16.2%`、`tg_icb +7.5%`** |
| **`Llama-3.2-1B-Q4_0`** `pp512` / `tg_b1` | `5,952.6` / `375.0` | **`6,052.3` / `377.1`** | `6,198.8` / `352.0` | 稳健小幅提升，零回归 |
| **底层 Radix APC 跨逐出存活** | 无 | **`cached=4, matched=64/72, ref=222==apc=222 (BIT-EXACT)`** | — | 释放全部 slot 后精确命中 |
| **Zero-Rust Metal v17 服务器 HTTP APC（25-slot 逐出后）** | 无 | **`TTFT 95.45ms → 11.54ms` (`8.27×`!!)**，`BIT-EXACT=True`，复用 `496` tok，`cached=81` | — | **`1,101.6 KiB`，378 项 Z3 证明 + 139 项越界消除 + 100 项零溢出** |
| **DeepSeek-V3.2 DSA 自检（Metal 4）** | 无 | **`256×32 = PASS, 512×64 = PASS`** | — | 精确匹配 CPU 参考真值 |
| **双端 7/7 HA 生产 E2E（`verify_cli_and_server_v11.py`）** | `14/14` | **`14/14` 全绿**（`tenon llm` CLI + Zero-Rust Metal v17） | — | HA 故障转移 `failovers=2`、TNKL 篡改 HTTP 400、SSE RST 零泄漏、LoRA 热插拔、503 背压全绿 |
| **`cargo test --workspace` + 零告警编译** | `100/100` | **`104/104` 全绿**，Metal & CUDA 双 Shim `clang -Wall -Wextra` **0 警告** | — | 新增 `linux-x86_64-cuda` 目标单元测试 |

---

## 四、v17 实机消融与工程复盘（科学诚实记录）

1. **`cm_gemm_q4_mmq_m128` (`128M×64N`) vs `cm_gemm_q4_mmq_m128_128` (`128M×128N`) 消融**：
   - `ptxas` 实测 `m128_128` 虽为 0 spill 但占用 **164 registers/thread**（仅 1 CTA/SM），且在 `1B`（`N=2048..8192`）上 Grid 减半令 `pp512` 跌至 `24,422`；而 13.5 KiB XOR-Swizzle `ldmatrix.x4` **`m128_64` 仅 97 registers/thread**，在 `70B 1G`（`629.3 vs 609.4`）、`70B TP=2`（`412.8 vs 406.5` R1 / `985.1` R2）与 `1B`（`46,859`）上全面胜出，故定型为默认主路径（保留 `TENON_Q4_MMQ_MODE=128` 供显式调优）。
2. **`gpt-oss-20b` `Q8_0` LM-Head 消融与 `attn_sinks` + `Q8_0 KV` 边界修复**：
   - R1 实测 `Q8_0` LM-Head：`q8_lm_mode=0`（默认 32-lane 4-rows/warp 共享 `xq8`，适配奇数 `nb`）达 **`229.69 tok/s`**，显著优于 `lm16 (211.7) > lm8 (208.5) > lm2 (173.4)`；
   - R1 中 `TENON_MAX_CTX=4096` 触发自动 `Q8_0 KV` 绕过了 FP16 `learn_sinks`，R2 修复为 `m->kv_quant_q8 = (m->max_ctx > 2048 && !m->is_gpt_oss) ? 1 : 0`，使 `gpt-oss` 在任意上下文下默认保持 `FP16 KV` + 完整 `attn_sinks` 并将 `pp512` 推高至 **`4,265.6 tok/s`**（`pp512_tok=279` 与 Metal 端完全一致）。
3. **Zero-Rust 服务器 APC + 尾部批预填充优化与 `apc_http` 透明说明**：
   - 发现聊天模板前 5 Token（`<|begin_of_text|><|start_header_id|>system...`）会令所有会话命中共用槽位 `best_lcp=5` 从而跳过 Engine-Level APC，且 `rem<=16` 在空闲服务（`n_dec==0`）下逐 Token 跑 16 次 `step_batch` 耗时 `58ms`；修复为优先比较 `apc_max > best_lcp` 并在预填充完成时全量注册所有 16-token 块、空闲时 `rem>1` 走单次 `paged_prefill`，使 Apple M4 Max 25-slot 逐出后 HTTP TTFT 从 `95.45ms` 骤降至 **`11.54ms` (`8.27×`, `BIT-EXACT=True`)**。
   - 在 L40S 77 并发 HTTP 压测中，`paged_kv_blocks=128` 在 25 个租户轮番压入后触发真实 LRU 逐出（`evictions=18`），故 `apc_http` 命中剩余前缀块加速 `1.59×`（非全量前缀复用 + FP16 TC 累加次序致 `bit_exact_reply=false`），而底层非逐出精确校验 `apc_eviction_survival`（`64/64`）与 Apple M4 Max HTTP 逐出存活均严格为 **`bit_exact=true`**。
4. **Metal 单 TG 融合路由核负增益回退**：
   - 实测 Metal 单 TG `moe_router_gemv_topk` 反而令 `DSV2 tg`（`141 → 106`）与 `gpt-oss tg`（`134 → 120`）回退，已果断关闭（`use_fused_rt = 0`），保留全部正增益路径。
5. **RunPod 成本与资源清理**：
   - R1（`2× RTX 6000 Ada`）+ R2（`2× L40S`）均已自动释放销毁，`Active pods: []`，剩余余额 `$19.2047`。
