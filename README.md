# Tenon 语言设计与技术方案（已拍板定稿）

> **Working Codename**: `Tenon`（榫卯）—— 寓意**强约束精化类型、分层形式化验证、无别名污染的内存模型、系统级版本演化与可验证分布式协议**严丝合缝。  
> **状态**: 核心架构已定稿（[ADR-0001](docs/adr/0001-architecture-decisions.md) & [ADR-0002](docs/adr/0002-interfaces-comptime-and-stdlib.md)），**极简语言核心 → 完整系统编程能力（S1/S2/S3：精确宽度整数与 `Float32`、`@repr(C)`/`@packed`/`@align`、零分配 `bitfield`、非标签 `union`、`unsafe` 裸指针/volatile、双向 C FFI + `--lib static|shared` + `--emit-header`、真实 OS 多线程/原子/互斥锁/MPMC Channel + MVS `Send` 边界检查 `E0810` + 条件原子 RC、原生 LLVM SIMD 向量 IR、DWARF `-g`、`tenon fmt`）→ Loom 风格目录即包（`tenon.toml`）+ Go/通配/选择性 `import` + `tenon test` 零泄漏测试框架 → 强大元编程系统（`comptime { ... }` / `comptime fn` / `comptime if` / `comptime for` / 声明式与类型化生成宏 `macro` / `gen!(...)` / 编译期反射）→ 高性能网络库（OS `kqueue`/`epoll` Reactor + 多线程 Keep-Alive HTTP/1.1 Server & Client `std/net/`）→ 元编程 Web 框架（`std/web/` + `@get`/`@post` 编译期路由扫描 + 自动 OpenAPI 3.1 + 精化类型 JSON 校验 `examples/web_app/`）→ 原生 LSP 语言服务器（`tenon lsp`）与 VS Code 扩展（`editors/vscode/`）已全部贯通**（Rust 1.100.0 + Z3 + Cranelift Dev JIT + **LLVM 22 `-O3` AOT 独立原生二进制，核心与多线程基准 0.97x–1.01x C `-O3`，4-worker HTTP/1.1 Keep-Alive 实测 59.6k RPS**）。
> **上游沉淀**: 综合 [`loom-lang`](../loom-lang)（984 PR 自举实践）、[`lang-research`](../lang-research)（2026-11 多语言可复现实验）与 [`dast-os-kernel`](../../dast.ai/dast-os-kernel/README.md)（227 轮分布式调和内核审计），并结合 2025–2026 全球最新编程语言与形式化验证进展形成的全新通用编程语言设计。

---

## 0. 快速体验可运行原型（Quick Start）

前置依赖：Rust `stable`（已验证 `rustc 1.100.0`）、`z3`、LLVM `clang`（已验证 LLVM 22.1.8；可选 `node` 验证编译出的 `.wasm`）。

```bash
# 1. 全工作区 69 项自动化测试（含全部 Next P0–P3 能力：平面未装箱 `List[SimdVec]` 使 `07_simd_dot` 降至 1.02x C、`alloca` 栈提升与 `@arena` 64 KiB Slab Arena、签名早停增量编译 + 并行 Z3；OpenSSL TLS 1.3 双向 HTTPS、PostgreSQL SCRAM-SHA-256 + Extended Query + `ConnPool`、Radix Trie Web 路由 + `@ws`/`@sse` + 静态文件防穿越；全功能 LLVM `wasm32` + Perceus 线性内存运行时、TS/Swift/Kotlin/Go `cgo` 绑定、效应多态 `[effect E] / E`、Z3 ADT/Set 理论、`tenon verify-proofs` 独立审计器、`--emit-repro-test`、L2 活性与 `--replay`；`tenon new`/`doc`/`bench`/`pkg sync`、三后端差异化模糊测试、Stage-0 自举词法格式化器、Go `cgo` 量化回测库与 Wasm32 A2UI 核）
cargo test --workspace

# 2. 精化类型别名复用 + 高阶形式化验证闭环 + 独立证明导出与审计 + 反例转测试
cargo run --bin tenon -- check examples/refinements/date_and_odd.tn
cargo run --bin tenon -- run   examples/refinements/date_and_odd.tn
cargo run --bin tenon -- check examples/refinements/bad_date.tn --emit-repro-test target/repro_test.tn
cargo run --bin tenon -- check examples/refinements/advanced_verification.tn --emit-proofs target/proof_certs
cargo run --bin tenon -- verify-proofs target/proof_certs --solver auto
cargo run --bin tenon -- build examples/refinements/advanced_verification.tn -o target/adv_ver_bin --opt O3 --checked-overflow && ./target/adv_ver_bin

# 2b. 生产级 Dogfooding & 自举实战（撮合风控引擎、Go cgo 量化回测静态库、Wasm32 A2UI 响应式核、Stage-0 纯 Tenon 词法格式化器）
cargo run --bin tenon -- build examples/dogfood/matching_engine.tn -o target/matching_bin --opt O3 && ./target/matching_bin
cargo run --bin tenon -- build examples/bootstrap/tenon_lex_fmt.tn -o target/lex_fmt_bin  --opt O3 && ./target/lex_fmt_bin
cargo run --bin tenon -- build examples/dogfood/a2ui_wasm_kernel.tn --target wasm32 -o target/a2ui.wasm --emit-ts target/a2ui.d.ts
cargo run --bin tenon -- build examples/dogfood/quant_backtest.tn --lib static --emit-header examples/dogfood/go_cgo_demo/quant.h --emit-go examples/dogfood/go_cgo_demo/quant_cgo.go -o examples/dogfood/go_cgo_demo/libquant.a
# 标准库 / 工具链 CLI：`tenon test`, `tenon doc`, `tenon bench`, `tenon new`, `tenon pkg`
cargo run --bin tenon -- test std/db && cargo run --bin tenon -- test std/net && cargo run --bin tenon -- test examples/web_app
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
| [`crates/tenon-verify`](crates/tenon-verify) | `IncrementalDb` 增量缓存 + **持久化磁盘 VC 缓存（`.tenon-cache/vc_cache_v1.json`）** + L0 精化求解（精确整型区间边界、`Float` 区间精化、`len(xs)` 符号追踪、定长数组越界、`bitfield` 位宽校验 `E0610`、`offset_of` 字段校验、`SafeRelPath` 与递归嵌套穷尽性）+ L1 Z3 SMT（**完整 `while` 不变量基例/归纳/退出传播与 `decreases` 终止性 `E0431`–`E0433`、`inout` 通用 `old(expr)`、零开销 `ghost`/`lemma` 证明擦除 `E0434`/`E0702`、符号化 `Float` `Real` 理论、64 位 BitVector 位运算桥接、递归 `pure fn` 有界展开、证明 `0 <= i < len(xs)` 驱动 LLVM 消除越界检查、精化子类型依赖转换 `as`/`as?`/`refine_cast[T](x)` `E0410`/`E0416`**、`mutate`、接口 Liskov 契约 `E0500`–`E0506`、`unsafe` 强制边界 `E0800`、同包私有共享与跨包 `pub` 可见性 `E0105`、生产/测试隔离 `E0107`、MVS 跨线程 `Send` 检查 `E0810`、`@no_heap_alloc`、`async` 效应）+ **版本优先 `@evolve`** + L2-b 通用 AST 协议模型检测 |
| [`crates/tenon-codegen`](crates/tenon-codegen) | 统一前置降级（泛型 `comptime if`/`comptime for`/`T implements I` 单态化分支裁剪、`defer` LIFO 展开、零分配 `bitfield`、泛型方法显式类型实参 `ctx.json_body[T]()`、方法扁平化、单态化、闭包转换、`dyn` 虚表）+ Cranelift Dev JIT + **LLVM 22 AOT Release 后端（`tenon build -O3`，内联 `@__tn_cstruct_data_ptr` 与原生 SIMD 树形归约 IR、条件原子 RC、模块化 `kqueue`/`epoll` C 网络运行时、双向 C FFI `--lib static\|shared` + `--emit-header` + `--c-src`、DWARF `-g`）** + `.wasm` 客户端编译器 |
| [`std/`](std) | **100% 纯 Tenon 编写的 4 层正交标准库与网络/Web 框架**：<br>• Layer 0 [`std/core/{types,ops}.tn`](std/core)：`NonNegative`/`Positive`/`Float`、`Eq`/`Hash`/`Display`、`List[T]`/`Option`/`Result`/`Writer`<br>• Layer 1 [`std/text/`](std/text)、[`std/collections/{map,set,bytes,store}.tn`](std/collections) & [`std/simd/simd.tn`](std/simd)：`TextBuilder`、`Map[K,V]`/`Set[T]`、`Bytes`/`BytesSlice`、`Store[T]`、硬件向量 `Float32x4`/`Float64x4`/`Int32x4`/`Int64x4`/`UInt8x16`<br>• Layer 2 [`std/encoding/{json,dotenv}.tn`](std/encoding)：`JsonValue` ADT + `ToJson`/`FromJson`/`JsonSchema` + `Dotenv`<br>• Layer 3 [`std/os/`](std/os)、[`std/sync/sync.tn`](std/sync/sync.tn)、**[`std/net/{reactor,http,http_test}.tn`](std/net)** & **[`std/web/{router,middleware,app}.tn`](std/web)**：OS `kqueue`/`epoll` Reactor、多线程 Keep-Alive HTTP/1.1 Server & Client（支持 Chunked）、分段参数/通配路由、CORS/Timing/Security 中间件、`@get`/`@post`/`@put`/`@delete` 编译期路由挂载宏 `mount_annotated_routes!()` 与自动 OpenAPI 3.1 生成 |
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
