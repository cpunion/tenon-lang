# Tenon 语言设计与技术方案（已拍板定稿）

> **Working Codename**: `Tenon`（榫卯）—— 寓意**强约束精化类型、分层形式化验证、无别名污染的内存模型、系统级版本演化与可验证分布式协议**严丝合缝。  
> **状态**: 核心架构 13 项决策已定稿（[ADR-0001](docs/adr/0001-architecture-decisions.md)），**M0–M7 全栈原型已彻底贯通**（Rust 1.100.0 + Z3 + Cranelift Dev JIT + **LLVM 22 `-O3` AOT 原生后端** + Perceus RC/COW + 无栈 `async` 状态机 + `@no_heap_alloc` + 通用 `.tn` AST 协议检测 + 真多进程 Authority/Worker/Fabric Sandbox + `.tn` 直编二进制 `.wasm` 客户端 + OpenAI SSE LLM & Webhook 渠道 + **`dast-os-kernel` 全栈 Tenon 重写原型**）。  
> **上游沉淀**: 综合 [`loom-lang`](../loom-lang)（984 PR 自举实践）、[`lang-research`](../lang-research)（2026-11 多语言可复现实验）与 [`dast-os-kernel`](../../dast.ai/dast-os-kernel/README.md)（227 轮分布式调和内核审计），并结合 2025–2026 全球最新编程语言与形式化验证进展形成的全新通用编程语言设计。

---

## 0. 快速体验可运行原型（Quick Start）

前置依赖：Rust `stable`（已验证 `rustc 1.100.0`）、`z3`、LLVM `clang`（已验证 LLVM 22.1.8；可选 `node` 验证编译出的 `.wasm`）。

```bash
# 1. 全工作区 18 项自动化测试（含 Perceus 零泄漏、无栈 async、LLVM 22 AOT、V8 Wasm、OpenAI SSE + Webhook、Fabric Sandbox + 410 Reset）
cargo test --workspace

# 2. 精化类型别名复用（Year/Month/Odd/Even/Amount）+ 跨字段 Date + Cranelift 原生执行
cargo run --bin tenon -- check examples/refinements/date_and_odd.tn
cargo run --bin tenon -- run   examples/refinements/date_and_odd.tn
cargo run --bin tenon -- check examples/refinements/bad_date.tn          # 负例：2023-02-29 编译期拦截

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

# 6. 启动交互式 Web UI（自动加载 /tenon_client_store.wasm；可选环境变量对接真实 OpenAI 兼容 LLM 与 Webhook 渠道）
# TENON_LLM_BASE_URL=http://127.0.0.1:11434 TENON_LLM_MODEL=qwen2.5 TENON_CHANNEL_WEBHOOK_URL=http://127.0.0.1:9000 \
cargo run --bin tenon -- serve examples/dast_os_kernel/kernel_v2.tn --addr 127.0.0.1:8787
```

### Workspace 结构

| Crate | 职责 |
| --- | --- |
| [`crates/tenon-syntax`](crates/tenon-syntax) | `loom/Go` 风格词法/语法分析器与 AST（精化、`mutate`、`async`/`await`、`evolution`、`lock_lattice`、`protocol state_machine`、`stream`、`reactive view`、`subscriber`） |
| [`crates/tenon-verify`](crates/tenon-verify) | `IncrementalDb` 增量查询缓存 + L0 极速求解 + L1 Z3 SMT（含 `mutate` 符号后态、`async` 效应纪律、`@no_heap_alloc`、别名继承）+ 版本演化/透镜证明 + 分布式约束求解 + L2-b 通用 AST 模型检测/锁序格/Trace Refinement |
| [`crates/tenon-codegen`](crates/tenon-codegen) | Cranelift Dev JIT + **LLVM 22 AOT Release 后端（`tenon build -O3`）** + Perceus RC/COW 零泄漏运行时 + 无栈 `async` 协程状态机 + 直出二进制 WebAssembly (`.wasm`) 客户端编译器 |
| [`crates/tenon-dist`](crates/tenon-dist) | SQLite WAL 分片权威存储 + 硬 `exec_epoch` Fencing + Turn DAG + 独立 Worker RPC + Fabric Sandbox 子进程隔离 + Journal GC `410 reset_required` + OpenAI SSE 流式 LLM 客户端 + 节流/拆包/死流消除 Webhook 渠道适配器 + 多路复用 HTTP/SSE/Wasm 网关 |
| [`crates/tenon-cli`](crates/tenon-cli) | `tenon` 统一 CLI（`check` / `run` / `test` / `build` / `build-wasm` / `evolve` / `modelcheck` / `serve` / `authority` / `worker` / `cluster-demo`） |

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
4. **双层系统级版本演化与兼容性自动推导**
   - 保留普通文本文件，仅对持久化/Wire/公开跨版本契约显式标注 `@evolve` 与稳定槽位 `@field(N)`；
   - 融合 **SMT 精化蕴涵 + 残差域 $\Delta$ + 已证明双向透镜（`evolution V1 <-> V2`）** 与 **按职责少版本化能力子集（`required(record, op) ⊆ supported(consumer, op)`）**，覆盖升级、灰度共存、降级保留数据与再升级全生命周期。
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

### 系统级版本演化与双向透镜示例

```tenon
@evolve("billing.Order", storage = "orders")
pub record Order {
    @field(1) id     Int64
    @field(2) amount Int64 where self > 0       // 相对 v1 (self >= 0) 收紧！编译器算出残差域 Δ: amount == 0
    @field(3) note   Option[Text] = None        // 新增可选字段：后向兼容，降级保留
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
| **[ADR-0001 — 13 项核心架构决议总表](docs/adr/0001-architecture-decisions.md)** | **最高裁决真相源**：记录全部 13 项已拍板架构决议（战略定位、核心边界、MVS+Perceus、无栈协程、checked 整数+Fail-Closed、双层演化、分片单写者+Turn DAG、Wasm/Native 单源产物、双引擎协议验证、Rust+Salsa+Cranelift/LLVM、loom/Go 语法、M0–M4 节奏）。 |
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
