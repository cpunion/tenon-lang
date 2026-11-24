# Tenon Benchmarks & Compile Regression Suite (vs C / Rust / Go)

本目录维护 Tenon 语言核心与标准库的多语言性能基准及编译回归套件，覆盖纯标量计算、位运算循环、密集数组原地写、纯 Tenon O(1) 哈希表、纯 Tenon JSON 解析/序列化与动态 `dyn Interface` 虚表分发 6 类典型负载。

## 快速运行

```bash
# 1. 快速编译 + 静态形式化验证 + 跨语言校验和回归门禁（CI 推荐）
python3 benchmarks/run.py --check-only

# 2. 完整跨语言性能基准测试（逐轮交错采样 + os.wait4 进程 CPU 时间 + 零泄漏检查）
python3 benchmarks/run.py --warmup 1 --reps 7 --save benchmarks/results.json

# 3. 单独运行指定项
python3 benchmarks/run.py --only 04_hashmap 05_json
```

> 说明：`cargo test --workspace` 亦内置 `benchmarks_tenon_compile_and_verify_regression` 自动测试，确保每次修改编译器或 `std/` 时，所有 `benchmarks/tenon/*.tn` 均通过解析、导入与 L0/L1 静态证明。

## 基准环境与实测基线（Apple M4 Max, darwin/arm64）

- **C**: `Homebrew clang 22.1.8` (`clang -O3`)
- **Rust**: `rustc 1.100.0` (`rustc -C opt-level=3`)
- **Go**: `go 1.27.3 darwin/arm64` (`go build`)
- **Tenon**: `tenon check`（L0/L1 全证明）+ `tenon build --opt O3`（LLVM 22 `-O3` + ThinLTO + Perceus `live_objects == 0` 退出强制断言）
- **度量方式**：`os.wait4` 进程 CPU 时间（`ru_utime + ru_stime`，抗宿主并发负载抖动），`1 warmup + 7 interleaved reps` 取中位数，4 种语言严格校验比特一致的整数校验和。

| Benchmark | 负载说明 | C (`-O3`) | Rust (`-O3`) | Go (`1.27`) | Tenon (`-O3`) | Tenon/C | Tenon/Go | Tenon 体积 | Checksum |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `01_fib` | 递归 `fib(40)`（`@no_heap_alloc`） | 184.37 ms | 182.02 ms | 239.66 ms | **176.78 ms** | **0.96x** | **0.74x** | 33.1 KB | `102334155` |
| `02_compute_loop` | 1 亿次 SplitMix64 标量/位运算循环 | 66.10 ms | 66.77 ms | 93.69 ms | **66.49 ms** | **1.01x** | **0.71x** | 33.1 KB | `4378771939937074704` |
| `03_sieve` | 埃拉托斯特尼筛（1M × 5 次，原地数组写） | 8.27 ms | 8.35 ms | 13.58 ms | **8.92 ms** | **1.08x** | **0.66x** | 33.4 KB | `392490` |
| `04_hashmap` | 50k 字符串泛型 `Map[Text, Text]` 增/改/查/删 25k/复查 | 16.20 ms | 16.37 ms | 16.89 ms | **16.12 ms** | **1.00x** | **0.95x** | 34.3 KB | `708334` |
| `05_json` | 5,000 次嵌套 JSON 解析/取字段/重建/序列化 | 6.20 ms | 6.25 ms | 15.03 ms | **6.11 ms** | **0.99x** | **0.41x** | 50.6 KB | `483984` |
| `06_dyn_dispatch` | 500 万次异构 `dyn Op` 虚表动态分发 | 5.45 ms | 5.62 ms | 7.73 ms | **5.43 ms** | **1.00x** | **0.70x** | 33.7 KB | `840988148304462647` |

**几何平均（6 项汇总）**：**`1.006x` C (`-O3`)**、**`0.68x` Go（比 Go 快 ~47%）**，其中 5 项达到 `<= 1.01x C` 并全面领先 Rust/Go，全部 6 项在 L0/L1 静态证明通过且 Perceus `live_objects == 0` 严格零泄漏下运行。

### 二进制体积对比（含带 OS / HTTPS-SSE 的完整应用）
- **C (`-O3`)**：32.7–33.2 KB
- **Tenon (`-O3` ThinLTO，纯计算/标准库基准)**：**33.1–50.6 KB**（无 GC 运行时，内置 Perceus 零泄漏退出校验）
- **Tenon (`-O3` ThinLTO，全量 4 层标准库测试 `examples/stdlib_test.tn`)**：**118 KB**（优化前 25.8 MB，缩小 **218×**）
- **Tenon (`-O3` ThinLTO，完整 HTTPS-SSE Coding Agent `examples/agent/cli_agent.tn`)**：**154 KB**（优化前 25.8 MB，缩小 **167×**；按需链接系统 `libcurl` + `libSystem`，零 `tokio`/`reqwest`/`rustls` 膨胀）
- **Rust (`-O3`)**：460–481 KB
- **Go (`1.27`)**：2.37–3.68 MB

### 核心编译器与运行时优化架构
1. **模块内 `alwaysinline` LLVM IR 辅助函数 + TBAA 别名元数据 + `internal` 链接**：将 Perceus `rc_inc`/`rc_dec`/`ensure_unique`、记录字段读/写、`dyn` 虚表槽位读取、列表索引、`byte_at`、`buf_push_byte` 与 `memcmp` 文本比较全部内联至同一 `.ll` 模块，并为对象头字段（`rc`/`tag`/`len`/`rc_mask`）与载荷区（`fields[]`）注入独立 `!tbaa` 树，使 `06_dyn_dispatch` 从 `19.92ms` 降至 **`5.36ms`（1.00x C）**。
2. **静态唯一性数据流分析 + `list_filled` 长度后置条件（`@llvm.assume`）**：对函数内未发生别名的局部唯一对象跳过每次写入的 `ensure_unique` 与判空，并向 LLVM 声明 `list_filled(n, v)` 返回列表长度恒等于 `max(n, 0)`，解锁内层循环边界检查消除与 ARM64 NEON 自动向量化，将 `03_sieve` 从 `19.97ms` 降至 **`8.90ms`（1.09x C，领先 Go 34%）**。
3. **只读永生常量（`rc = -1`）与小对象 LIFO Freelist**：字符串字面量与无载荷枚举变体（`None`、`JNull` 等）直接下沉到 `.rodata` 零分配常量（LLVM 在编译期彻底折叠其 `rc_inc`/`rc_dec`）；AOT 运行时为 `cap <= 8` 的小对象提供 L1 热缓存 LIFO Freelist（仍严格维护 `g_live == 0` 泄漏断言）。
4. **高质量纯 Tenon 标准库**：`std/collections/map.tn` 提供真泛型单态化 `Map[K Hash + Eq, V]`（SoA 平行数组 `keys`/`vals`/`hashes` + O(1) swap-remove + 零条目拷贝 `rehash`），`std/text/text.tn` 的 `int_to_text` 用常量除 10 位打包消除全部硬件 `sdiv`，`std/encoding/json.tn` 采用零中间装箱游标解析 + `JsonValue.put(inout self, ...)` 零 COW 原地构建，使泛型 `04_hashmap` 达到 **`1.00x C`（16.12ms，快于 C/Rust/Go）**、`05_json` 达到 **`0.99x C`（6.11ms，比 Go 快 2.46×）**。
5. **零依赖 `tenon-rt` 与按需轻量 AOT C 运行时**：彻底移除 `tenon-rt` 的 `tokio`/`reqwest`/`futures-util` 依赖（`Cargo.toml` 0 第三方依赖，编译耗时 `0.31s`），JIT 通过 Rust `#[link(name = "curl")]` FFI、AOT 通过按需注入的自包含 C POSIX + `<curl/curl.h>` 运行时实现流式 HTTP/SSE，彻底消除 25.8 MB 静态库链接膨胀。
