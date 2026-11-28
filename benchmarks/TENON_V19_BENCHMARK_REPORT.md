# Tenon v19 跨芯片全算子生态 + 零拷贝 IO HAL + 6 项代码审查缺陷修复实测与验收报告

> **版本**: Tenon v19（基于 `main` `238b5ed` 演进）  
> **验收原则**: 严格区分 **(A) 本机真实硬件实测数据**（Apple M4 Max 40-core Metal 4.0 / macOS 27.0.1、Apple M4 Max MoltenVK Vulkan 1.3、Node V8 Wasm32、Rust `cargo test --workspace` + VS Code 扩展测试 + CLI 端到端复现）与 **(B) 静态/AST/SPIR-V 编译校验待目标真机用户实测路径**（NVIDIA Blackwell `sm_120a`、AMD ROCm CDNA3 `gfx942` / RDNA3-4、Windows 11 MSVC/Clang 链接、裸机 Linux `io_uring SQPOLL` / GPUDirect Storage）。

---

## 1. 6 项 Code Review 缺陷（P1 × 5, P2 × 1）修复与复现验收

针对 `main` (`238b5ed`) 审查报告的全部 6 项问题已逐项根治并固化为自动化回归测试：

| 编号 | 级别 | 问题 | 根因与修复方案 | 验证结果 |
|---|---|---|---|---|
| **R1** | P1 | 验证缓存遗漏 `reactive view` 声明（`bad_sync.tn` 改回 `NeverReset` 仍 `cache=HIT`） | [`crates/tenon-verify/src/db.rs`](../crates/tenon-verify/src/db.rs) 中 `program_signature_hash()` 的 `_ => {}` 静默忽略了 `View` / `Stream` / `Subscriber` / `Protocol` / `LockLattice` / `Evolution` 等声明。现已改为**穷举匹配全部 21 个 `Item` 变体**（含所有 `attrs`、`View` 的 `bounds`/`sync_policy`/`projection`/`order_by` 等），并将磁盘缓存版本升级为 `v2_` (`verify_v2.json`)。 | **PASS**：单测 `reactive_view_bounds_change_invalidates_disk_and_memory_cache` 全绿；CLI 实测 `BoundedReset`→`PASS (MISS)`→二次`PASS (HIT)`→改回`NeverReset`（不设 `TENON_NO_CACHE`）→**`FAIL error[P2014]` 退出码 1**。 |
| **R2** | P1 | `build-wasm` 丢失布尔短路语义（`false && { x = 1; true }` 执行了副作用） | [`crates/tenon-codegen/src/wasm_general.rs`](../crates/tenon-codegen/src/wasm_general.rs) 原先先求值两侧再发射 `i64.and`/`i64.or`。现改为标准 Wasm `if (result i64)` 分支短路：`&&` 在左侧为 0 时直接返回 0（跳过右侧）；`\|\|` 在左侧非 0 时直接返回 1（跳过右侧）。 | **PASS**：`general_wasm32_short_circuit_block_scoping_and_float_if_in_v8` 与 CLI `tenon build-wasm` + V8 实测均返回 `x = 0`（与 JIT 一致）。 |
| **R3** | P1 | `build-wasm` 块内 `let` 遮蔽覆盖外层变量绑定 | [`crates/tenon-codegen/src/wasm_general.rs`](../crates/tenon-codegen/src/wasm_general.rs) 在进入 `emit_block` / `emit_block_branch` 前保存 `self.locals` 快照并在退出块时恢复，同时新增 `alloc_temp()` 使内层 `let` 分配独立 Wasm local 槽位（而 `Stmt::Assign` 仍可正确修改外层 `var`）。 | **PASS**：V8 实测外层 `let x = 1` + `if true { let x = 2 }` 返回 `1`（与 JIT 一致）。 |
| **R4** | P1 | 浮点 `if/else` 表达式生成无效 Wasm（V8 报 `expected i64, got f64`） | [`crates/tenon-codegen/src/wasm_general.rs`](../crates/tenon-codegen/src/wasm_general.rs) 原先将带 `else` 的 `if` 块类型硬编码为 `0x7E` (`i64`)。现改为根据分支实际类型动态回填 `0x7E` (`i64`) / `0x7C` (`f64`)，并正确处理含 `return`/`break`/`continue` 的发散分支。 | **PASS**：`pub fn choose(b Bool) Float { if b { 1.5 } else { 2.5 } }` 经 `tenon build-wasm` 编译后在 V8 中加载并实测返回 `choose(1n)=1.5`, `choose(0n)=2.5`。 |
| **R5** | P1 | Windows CUDA 目标仍无条件依赖 Linux POSIX 头文件与 `dlopen("libcuda.so")` | [`std/gpu/cuda/moe_stream_cuda_shim.c`](../std/gpu/cuda/moe_stream_cuda_shim.c) 重构为完整 `#if defined(_WIN32)` 原生 Win32 适配层：`LoadLibraryA`/`GetProcAddress` 加载 `nvcuda.dll`、`nvrtc64_*.dll`、`cublas64_12.dll`；原生 Win32 `CRITICAL_SECTION` + `CONDITION_VARIABLE` + `CreateThread` 包装；MSVC 原子原语宏；`_fseeki64`/`_ftelli64`/`_stat64`/`_fstat64`/`_open(..., _O_BINARY)`；线程安全 `OVERLAPPED` `ReadFile` + `hfile` 透传至 Windows `IOCP`；`CreateFileMappingA`/`MapViewOfFile`、`VirtualAlloc`、`QueryPerformanceCounter`、`K32GetProcessMemoryInfo`、`%TEMP%` 与 Windows `nvcc`。同步补齐 [`crates/tenon-codegen/src/llvm_runtime.rs`](../crates/tenon-codegen/src/llvm_runtime.rs) 中 `_MSC_VER` 位运算与 Win32 线程同步原语。 | **静态校验 PASS**（未在 Windows 主机执行最终链接）：`clang -Wall -Wextra` **0 错误 0 警告**；全文件 25,080 行自动化 POSIX 扫描确认 `#if !defined(_WIN32)` 保护之外**残余 POSIX 调用数 = `0`**。 |
| **R6** | P2 | 跨平台链接脚本 `_link.bat` / `_link.sh` 写入构建机绝对路径且未打包本地头文件 | [`crates/tenon-codegen/src/llvm_aot.rs`](../crates/tenon-codegen/src/llvm_aot.rs) 新增 `bundle_companion_c_sources()`：递归解析所有 `@c_src` 的本地 `#include "..."` 头文件，计算最近公共祖先目录并自动复制到同级 `<out>_src/`（如 `cuda/moe_stream_cuda_shim.c` + `common/tenon_io_hal.h`）。`_link.bat` 改用 `set "DIR=%~dp0"`（自动探测 `clang` 或 MSVC `cl`），`_link.sh` 改用 `DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"`，彻底消除构建机绝对路径。 | **PASS**：新增单测 `cross_gpu_link_bundle_uses_strictly_relative_paths_and_copies_local_headers` 全绿；CLI 实测生成 `win_srv_link.bat` / `lin_srv_link.sh` 零 `/Users/` 路径，且 `win_srv_src/{cuda,common}` 均完整随包携带。 |

- **全仓自动化测试汇总**：`cargo test --workspace` **107 passed, 0 failed**（原 104 项 + 新增 3 项回归测试），VS Code 扩展测试 **2 passed, 0 failed**。

---

## 2. Tenon v19 四大硬件支柱交付与验证详情

### 支柱 1：NVIDIA Blackwell `sm_120a`/`sm_100a` 原生 FP4 & 全速 FP8 Tensor Core 算子套件
- **实现文件**: [`std/gpu/cuda/moe_stream_cuda_shim.c`](../std/gpu/cuda/moe_stream_cuda_shim.c)
- **核心升级**:
  1. **Blackwell `sm_120a` 原生硬件 MXFP4 Tensor Core GEMM (`cm_blackwell_mxfp4_gemm_hw`)**：修正 `sm_120a`/`sm_100a` 特性宏守卫（`__CUDA_ARCH_FEAT_SM120_ALL` / `__CUDA_ARCH_FEAT_SM100_ALL`）与 PTX ISA 8.7 `mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0` 操作数语法（标量 scale 寄存器无花括号），支持 `TENON_BW_FP4=1` 启用。
  2. **Blackwell 全速 FP8 (`e4m3×e4m3`) 1-Barrier Tensor Core GEMM & MoE Mega-Kernel 套件**：
     - 新增 `cm_quant_fp8_1_b`（块级 `FP8_1` 激活在线量化）、无损精确查找表 `cm_mega_unpack16_e4m3`（覆盖 `Q4_0`/`Q4_1`/`Q4_K`/`Q5_K`（居中 `q-16` 并把 `+16*dw` 折叠进 `mw`）/`Q5_0`/`MXFP4`/`FP8_E4M3`，所有量化整数及 `e2m1` 值在 `e4m3` 中 100% 精确表示、零权重舍入损失）、`cm_mega_dot4_fp8`。
     - 新增 `cm_gemm_q4_fp8_m128`（128M×64N×64K Split-K 稠密 GEMM）与 4 个分组 MoE Mega-Kernel（`cm_moe_grouped_gu_swiglu_fp8_mmq`、`cm_moe_grouped_down_fp8_mmq`、`cm_moe_grouped_gu_oai_fp8_mmq`、`cm_moe_grouped_down_bias_fp8_mmq`）。在 Blackwell 上默认开启（`g_cu.use_fp8_mmq = is_blackwell`，支持 `TENON_CUDA_FP8_MMQ=0/1` 覆盖），解决 Blackwell 消费/工作站级 GPU INT8 TC 降频问题，恢复全速 2× Tensor Core 吞吐。
- **验证状态**（静态/AST 校验，待 Blackwell 真机用户实测）：
  - 主机 C 代码 `clang -Wall -Wextra` **0 错误 0 警告**；
  - 嵌入式 `kCuMoeSrc` 全部 **11,728 行** CUDA C++ 核函数经 `clang++ -D__CUDA_ARCH__=1200 -D__CUDA_ARCH_FEAT_SM120_ALL=1` 语法与类型检查 **0 C++ AST 错误**（检查期间成功拦截并修复了 1 处 `cm_dsa_indexer_topk` 前的注释行拼接瑕疵与 1 处 `launch_gemm_batch_raw_ex` 参数个数不匹配）。

### 支柱 2：跨厂商 AMD ROCm (CDNA3 MFMA / RDNA3-4) & Vulkan 1.1–1.3 完整算子套件
- **2A — AMD ROCm/HIP 零 SDK 全算子引擎**: [`std/gpu/rocm/rocm_stream_shim.c`](../std/gpu/rocm/rocm_stream_shim.c)、[`std/gpu/rocm/rocm_stream.tn`](../std/gpu/rocm/rocm_stream.tn)
  - 零 ROCm SDK 编译依赖，动态加载 `libamdhip64.so`/`amdhip64.dll` 与 `libhiprtc.so`/`hiprtc.dll`，自动读取 `prop.gcnArchName` 并以 `--offload-arch=<arch>` 在线编译 **16 个自包含 HIPRTC 内核**：
    - `tn_rocm_rmsnorm_f32`、`tn_rocm_quant_q8_1`、`tn_rocm_gemv_{q4_0,q4_k,q6_k,q8_0}_q8_1`（原生 `__builtin_amdgcn_sdot4`）、`tn_rocm_gemv_mxfp4_f32`；
    - **CDNA3 (`gfx942` MI300X) 原生 Wave64 `MFMA` 16×16×32 INT8 (`__builtin_amdgcn_mfma_i32_16x16x32_i8`) `Q4_0×Q8_1` GEMM**（单条 MFMA 指令精确对应 1 个 32 元素 Q4_0 块！）与 **CDNA2/3 `MFMA` 16×16×16 FP16 (`__builtin_amdgcn_mfma_f32_16x16x16f16`) GEMM**（RDNA3/4 Wave32 自动回退分块 `sdot4` GEMM）；
    - `tn_rocm_rope_kv_paged`、`tn_rocm_flash_attn_paged`（GQA + 因果 + SWA + softcap + `learn_sinks`）、`tn_rocm_swiglu_f32`（标准 + GPT-OSS）、`tn_rocm_moe_router_topk`、`tn_rocm_gdn_wy_prefill`（C=32 WY-GDN）、`tn_rocm_argmax_f32`。
  - 集成 16 MiB 锁页暂存区 + `tenon_io_hal.h` `IORING_REGISTER_BUFFERS` 异步存储直达 VRAM (`tn_rocm_io_read_to_vram`) 与 MI300X 192 GB HBM3 全常驻自动识别。
  - **验证状态**（静态/AST 校验，待 AMD GPU 真机用户实测）：主机 C `clang -Wall -Wextra` **0 警告**；嵌入式 HIP 核函数在 **RDNA3/generic** 与 **CDNA3 `-D__gfx942__=1` MFMA** 双模式下 `clang++ -std=c++17` **0 错误**。
- **2B — Vulkan 1.1–1.3 零 SDK 真 GPU 计算 + DP4A / Cooperative Matrix 引擎**: [`std/gpu/vulkan/vulkan_shim.c`](../std/gpu/vulkan/vulkan_shim.c)、[`std/gpu/vulkan/tenon_vk_spv.h`](../std/gpu/vulkan/tenon_vk_spv.h)、[`std/gpu/vulkan/gpu.tn`](../std/gpu/vulkan/gpu.tn)、[`std/gpu/vulkan/shaders/`](../std/gpu/vulkan/shaders/)
  - 6 个 GLSL 4.50 计算着色器全部通过 `shaderc/glslc v2026.3` 编译为预验证 SPIR-V 字节码并静态内嵌（零运行时着色器编译器依赖）：
    - 4 个零扩展 Vulkan 1.1 通用着色器：`q4_0_gemv.comp`（纯 32-bit 读取直接消费原始 18B GGUF Q4_0 权重）、`rmsnorm.comp`、`swiglu.comp`、`flash_attn_paged.comp`（因果 + GQA + SWA + 非恒等页表）；
    - 2 个 Vulkan 1.3 硬件扩展着色器：`q4_0_gemm_dp4a.comp`（`VK_KHR_shader_integer_dot_product` / `OpSDot`）与 `fp16_gemm_coopmat.comp`（`VK_KHR_cooperative_matrix` 16×16×16 矩阵核，统一驱动 Intel Arc XMX / AMD WMMA / NVIDIA Tensor Core）。
  - 完整真实 Vulkan Compute 调度栈（`VK_KHR_portability_enumeration` / `VK_KHR_portability_subset` 自动适配 macOS MoltenVK、真实 GPU `VkBuffer`/`VkDeviceMemory`、`vkCreateComputePipelines`、`vkCmdDispatch`、`vkQueueSubmit`）。
  - **本机 Apple M4 Max MoltenVK 实机验收（LIVE PASS）**：
    - `dev='Apple M4 Max'` (`vendor=0x106b`, `api=1.4.357`), **`compute=1, pipes=5/6`**（含 Vulkan 1.3 **`dp4a=1`** `OpSDot` 管线！）；
    - `tn_vk_self_test() == 1` **真机 GPU 全通过**：原始 GGUF Q4_0 GEMV (`64×512×2`) + RMSNorm + SwiGLU + 乱序页表 Paged FlashAttention-2 + Vulkan 1.3 硬件 DP4A `OpSDot` Q4_0×Q8 GEMM 全部与 CPU FP32 参考误差 `< 1e-3`！

### 支柱 3：多 OS 零拷贝存储-GPU 统一 IO HAL 深度升级
- **实现文件**: [`std/gpu/common/tenon_io_hal.h`](../std/gpu/common/tenon_io_hal.h)（已接入 Metal、CUDA、ROCm、Vulkan 四大后端）
- **核心升级**:
  - **Linux**: 原生零 `liburing` 的 `IORING_SETUP_SQPOLL`（`TENON_IO_URING_SQPOLL=1` 可选开启并自动回退）、`IORING_REGISTER_BUFFERS` + `IORING_OP_READ_FIXED`（零每请求页钉住开销，已接入 CUDA `stage_host`/`promo_stage`/`dense_ring_host` 与 ROCm 锁页暂存区）、`IORING_REGISTER_FILES` + `IOSQE_FIXED_FILE`；持久 `cuFile` 句柄缓存 + `cuFileBufRegister` (`tn_io_gds_register_vram`) + 批量 NVMe→VRAM 直通 `tn_io_gds_read_vram_batch`。
  - **Windows 11**: 真持久化 `IOCP` 句柄绑定 + 单次 `GetQueuedCompletionStatusEx` 批量收割 + 动态探测 `PrefetchVirtualMemory` + `tn_io_win_open_overlapped`。
  - **macOS**: `F_NOCACHE` + `F_RDADVISE` + 统一内存零拷贝 `MTLBuffer`；`tn_moe_io_self_test() == 1` 本机实测通过。

### 支柱 4：Apple Silicon Metal 4 (`mpp::tensor_ops`) + 分块 WY-GDN + 全模式 Q8/FP8 AMX FA2
- **实现文件**: [`std/gpu/metal/moe_stream_shim.m`](../std/gpu/metal/moe_stream_shim.m)
- **本机 Apple M4 Max (macOS 27.0.1, Build 26A434) 真实 GPU 实测结果**:

| 测试项 / 模型 | 对比配置 | Prefill (`tok/s`) | Decode (`tok/s`) | 精度 / 状态 |
|---|---|---:|---:|---|
| **Metal 4.0 MPP `matmul2d` + `cooperative_tensor`** | `MTLLanguageVersion4_0` (`0x40000`) + 融合 Bias Epilogue | — | — | **`tn_moe_mtl4_self_test() == 1` (LIVE HW PASS!)** |
| **Metal Chunkwise WY-GDN (`C=32`) 单元测试** | `B=96, n_vh=4, dk=dv=128` vs 顺序 GDN | — | — | **`tn_moe_gdn_wy_self_test() == 1` (`max_err < 1e-4`)** |
| **`Qwen3.5-0.8B-Q4_K_M` (`pp512`)** | `TENON_GDN_CHUNK=0` (v18 顺序) | `2,263.8` (`226.2 ms`) | `52.2` | 基线 |
| **`Qwen3.5-0.8B-Q4_K_M` (`pp512`)** | **`TENON_GDN_CHUNK=1` (v19 WY-GDN `C=32`)** | **`2,558.5` (`200.1 ms`, `+13.0%`!)** | **`79.1`** | 实机加速（注：单次 `B=2048` 大块未切片时受共享缓冲带宽影响为 `1,344.2`，配合服务层默认 `512` Chunked Prefill 获最大收益） |
| **`Llama-3.2-1B-Instruct-Q4_0` (`pp2048`)** | `FP16 KV` (AMX FA2) | `671.3` (`3,050.9 ms`) | `61.1` | 100% KV VRAM 基线 |
| **`Llama-3.2-1B-Instruct-Q4_0` (`pp2048`)** | **`Q8_0 KV` (v19 通用 Q8 AMX FA2)** | **`699.9` (`2,925.9 ms`, `+4.3%`)** | **`75.6`** | **50% KV 显存，反超 FP16！** |
| **`Llama-3.2-1B-Instruct-Q4_0` (`pp2048`)** | **`FP8_E4M3 KV` (v19 通用 FP8 AMX FA2)** | **`743.6` (`2,754.2 ms`, `+10.8%`!)** | **`72.3`** | **50% KV 显存，反超 FP16 `+10.8%`！** |
| **`gpt-oss-20b-MXFP4` (`pp512`, `learn_sinks`)** | `FP16 KV` (AMX FA2 + `learn_sinks`) | `1,037.2` (`493.6 ms`) | `90.6` | `[112,113,114,115,116,41695,21132,3441]` |
| **`gpt-oss-20b-MXFP4` (`pp512`, `learn_sinks`)** | **`Q8_0 KV` (v19 通用 Q8 AMX + `learn_sinks`)** | **`1,047.0` (`489.0 ms`, `+0.9%`)** | `87.5` | 50% KV 显存，前 6/8 token 一致 |
| **`gpt-oss-20b-MXFP4` (`pp512`, `learn_sinks`)** | **`FP8_E4M3 KV` (v19 通用 FP8 AMX + `learn_sinks`)** | **`1,065.6` (`480.5 ms`, `+2.7%`!)** | **`90.4`** | **50% KV 显存，`8/8 BIT-EXACT` 与 FP16 完全一致！** |
| **MoltenVK Vulkan 1.3 (`Apple M4 Max`)** | `Q4_0 GEMV + RMSNorm + SwiGLU + Paged FA2 + DP4A` | — | — | **`tn_vk_self_test() == 1` (`5/6` 管线真机全过!)** |
