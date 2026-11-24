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
| `01_fib` | 递归 `fib(40)`（`@no_heap_alloc`） | 187.28 ms | 185.11 ms | 239.66 ms | **187.19 ms** | **1.00x** | **0.78x** | 33.1 KB | `102334155` |
| `02_compute_loop` | 1 亿次 SplitMix64 标量/位运算循环 | 66.63 ms | 67.37 ms | 94.22 ms | **67.01 ms** | **1.01x** | **0.71x** | 33.1 KB | `4378771939937074704` |
| `03_sieve` | 埃拉托斯特尼筛（1M × 5 次，原地数组写） | 8.22 ms | 8.41 ms | 13.51 ms | **19.97 ms** | 2.43x | 1.48x | 33.6 KB | `392490` |
| `04_hashmap` | 50k 字符串 HashMap 增/改/查/删 25k/复查 | 16.32 ms | 15.01 ms | 16.47 ms | **27.59 ms** | **1.69x** | **1.68x** | 34.6 KB | `708334` |
| `05_json` | 5,000 次嵌套 JSON 解析/取字段/重建/序列化 | 6.57 ms | 6.54 ms | 15.72 ms | **19.89 ms** | 3.03x | **1.27x** | 50.9 KB | `483984` |
| `06_dyn_dispatch` | 500 万次异构 `dyn Op` 虚表动态分发 | 5.74 ms | 5.84 ms | 7.95 ms | **19.92 ms** | 3.47x | 2.51x | 34.0 KB | `840988148304462647` |

### 二进制体积对比
- **C**：32.7–33.2 KB
- **Tenon (`-O3` ThinLTO)**：**33.1–50.9 KB**（无 GC 运行时，且内置 Perceus 零泄漏退出校验）
- **Rust (`-O3`)**：460–481 KB
- **Go (`1.27`)**：2.37–3.68 MB

### 关键剖析与后续优化方向
1. **纯标量/计算密集型（`01_fib`、`02_compute_loop`）**：Tenon 已达到 **1.00x–1.01x C / Rust**，领先 Go **22%–29%**。
2. **纯 Tenon 标准库（`04_hashmap`、`05_json`）**：`TextMap` 在升级为开放寻址紧凑哈希表 + `int_to_text` 单缓冲 + `TenonBuf` SBO 后，跑出 **1.69x C / 1.68x Go**（零 COW 拷贝、零泄漏）；`std/encoding/json.tn` 达到 Go `encoding/json` 的 **1.27x**。主要差距来自 `JsonValue.set` 借用 `self` 触发的 COW 与每个短字符串的独立 RC 堆对象头。
3. **密集数组写（`03_sieve`, 2.43x C）**：仅 5 次堆分配、0 次 COW，主要开销是内层循环每次 `flags[m] = 0` 的 `rc == 1` 检查与边界检查；未来在 MIR 层加入**循环外唯一性提升（LICM uniqueness hoist）**与 L0 证明后的边界检查消除即可逼近 1.0x C。
4. **动态接口分发（`06_dyn_dispatch`, 3.47x C）**：当前 `dyn Op` 为 2 槽位堆胖对象（导致循环中多了 1 层堆解引用与字段边界检查）；未来升级为**非装箱栈/寄存器 2-word 胖指针 `(data_ptr, vtable_ptr)`** 可将开销压至 ~1.0x C/Rust。
