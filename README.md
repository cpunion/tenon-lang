# Tenon 语言设计与技术方案（讨论稿）

> **Working Codename**: `Tenon`（榫卯）—— 寓意**强约束类型、内置形式化验证、无别名污染的内存模型、系统级版本演化推导**四者如榫卯般严丝合缝。  
> **状态**: 架构设计与技术提案（2026-11-23），供团队深度讨论与决策。  
> **上游沉淀**: 综合 [`loom-lang`](../loom-lang)（984 PR 工程实践）与 [`lang-research`](../lang-research)（2026-11 多语言可复现实验），并结合 2025–2026 全球最新编程语言与形式化验证进展形成的新一代融合架构方案。

---

## 1. 核心目标：我们要造一门怎样的语言？

面向高可靠业务系统、基础设施与 AI 辅助工程（Vericoding）时代，Tenon 聚焦解决现代主流语言普遍割裂的四大问题：

1. **一等强约束类型系统（First-Class Refinement & Invariant Types）**
   - 从标量范围（正金额 `Amount > 0`）、数论同余（奇数 `Odd`、内存对齐 `Aligned[N]`）、可验证域值（`Email`、`NonEmptyList`），到**跨字段依赖结构体不变量**（如 `Date { year, month, day }` 自动校验闰年与每月合法天数、`DateRange { start <= end }`、复式记账借贷平衡）。
   - **边界即校验，内部零开销**：外部输入自动派生结构化校验器与编解码器；内部流敏感传播与编译期证明自动消除冗余运行时检查。
2. **分层内置形式化验证（Tiered Built-in Verification: L0 / L1 / L2）**
   - 日常 85%–90% 的精化约束（区间、奇偶/模运算、线性关系、有界纯函数求值）由进程内 **L0 毫秒级求解器**直接判定，不拖慢 IDE 与编译体验。
   - 函数契约（`requires` / `ensures` / `invariant`）与跨版本蕴涵由 **L1 增量 SMT（Z3/CVC5）** 自动证明，严格 **Fail-Closed**，并输出人类与 AI 都能直读的具体反例。
   - 复杂算法与关键基础设施按需启用 **L2 Ghost 引理 / 归纳证明 / 极小内核证书**。
3. **系统级版本演化与发布兼容性推导（System-Level Evolution Calculus）**
   - 不只是“代码能编译”，而是把 **API 契约、线协议（Wire/RPC）、持久化 Schema、在途状态、灰度共存、回滚与再升级**纳入编译器一等推导。
   - 编译器用 SMT 自动对比已部署基线 $V_1$ 与候选版本 $V_2$，精确推导是 **Hotfix-Safe（等价/兼容）**、**单向兼容**还是**破坏性收紧**，自动算出违约残差域 $\Delta = P_1 \land \neg P_2$，并静态证明双向演化透镜（`evolution V1 <-> V2`）的全覆盖性、回滚无损性与保留数据再升级收敛性。
4. **安全、低心智负担、可预测高性能（MVS + 多数据栈 + Perceus）**
   - 摒弃“默认共享可变 GC 堆”（它让形式化验证深陷别名干扰泥潭），也摒弃 Rust 式显式生命周期标注地狱 `'a`。
   - 采用 **默认可变值语义（Mutable Value Semantics, MVS）+ 局部 `inout` 借用**；运行时融合 **原生栈 + Goose 式编译器多数据栈 + Perceus 唯一引用原地复用（COW/FBIP）**；显式长期可变图由 `Store[T] + Handle + phase` 受控管理。
   - 编译速度对标 Go/Zig（Query 增量 + Cranelift/极速 Dev 后端），峰值执行性能对标 Rust/C（LLVM Release 后端）。

---

## 2. 为什么不在 `loom-lang` 现有架构上直接修补，而要提出新方案？

[`loom-lang`](../loom-lang) 已经走了非常远（984 个 PR、14.6 万行自举前端、三平台 Stage 2/3 字节一致自举、已能端到端用 Z3 证明排序有序性与全行置换），但把 `loom-lang` 的工程实践和 [`lang-research`](../lang-research) 的内存/性能实验放在一起审视，暴露出四个**架构级结构性张力**：

| 维度 | `loom-lang` 现状与痛点 | `Tenon` 新架构破局点 |
| --- | --- | --- |
| **语义与验证的耦合** | 默认“共享可变引用 + Moving GC”，导致编译器 14.6 万行中极大比例在和**别名干扰、堆版本失效、草案发布（draft publication）、回调逃逸**搏斗 | 采用**默认可变值语义（MVS）**，普通值无共享可变别名，精化不变量天然免疫外部别名破坏，验证器复杂度直降一个数量级 |
| **验证性能与分层** | 单层验证器直接压向外部 Z3 进程（重负载检查达 13.5s），日常精化校验与重型全函数证明未分层隔离 | **L0（<1ms 区间/同余/八边形/CTFE 局部求值）→ L1（增量 SMT + 内容缓存）→ L2（按需 Ghost/Lemma）** 三级解耦 |
| **系统级版本演化** | 割裂为三张皮：手工 `.loom-ids` sidecar（合并）、局部 `std.storage`（字段 CHECK）、硬编码 `orders`/`notes` 离线 SQLite 试炼 | 统一为**一等演化演算**：`@evolve` 稳定语义身份 + `evolution.lock` + SMT 自动兼容性矩阵与残差域推导 + 可证明的**双向演化透镜（Bidirectional Lens）** + 在线 Expand-Contract 计划 |
| **内存与编译效率** | STW Moving GC + 纯 LLVM 22 后端 + 过早自举（前端改动成本极高），离 Go/Zig 编译速度和无停顿系统编程距离远 | 融合 `lang-research` 结论：**数据栈/区域 + Perceus 复用 + 显式 `Store`**；前期用 **Rust + Query 增量架构 + Cranelift(Dev)/LLVM(Release)** 快速迭代核心语义，稳定后再自举 |

> [!NOTE]
> 这并非否定 `loom-lang` 的成果——恰恰相反，Loom 在精化构造、Fail-Closed 契约、证据分级（proof / check / test / assumption）、部署账本与降级保留数据等核心原则上踩通了极宝贵的路径。Tenon 完整继承这些原则，并把底层地基（内存语义、验证分层、演化演算、编译架构）一次性重构到位。

---

## 3. 一眼看懂 Tenon：从 `Date` 到系统级升级推导

下面是一段概念语法示例（语法细节待讨论，重点展示核心语义与推导能力）：

```tenon
// 1. 强约束标量与数论精化
pub type Amount  = Int64 where self > 0
pub type Odd     = Int64 where self % 2 != 0
pub type Even    = Int64 where self % 2 == 0
pub type Email   = Text  where is_valid_email(self)

// L0 求解器 <1ms 直接证明，无需启动 Z3，运行时零检查
pub pure fn add_odds(a: Odd, b: Odd) -> Even {
    a + b
}

// 2. 跨字段依赖结构体不变量：合法公历日期
pub pure fn is_leap_year(y: Int) -> Bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

pub pure fn days_in_month(y: Int, m: Int where 1 <= self && self <= 12) -> Int
    ensures 28 <= result && result <= 31
{
    match m {
        2 => if is_leap_year(y) { 29 } else { 28 },
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

pub struct Date {
    year:  Int where 1 <= self && self <= 9999,
    month: Int where 1 <= self && self <= 12,
    day:   Int where 1 <= self && self <= days_in_month(year, month),
}

// 编译期直接通过（L0 局部求值 days_in_month(2024, 2) == 29）
let leap = Date { year: 2024, month: 2, day: 29 }

// 编译期直接报错（反例明确指出 29 <= 28 不成立）
// let bad = Date { year: 2023, month: 2, day: 29 }

// 外部边界输入：自动生成结构化解析器，返回 Result[Date, ValidationError]
let parsed: Result[Date, ValidationError] = Date.parse(raw_year, raw_month, raw_day)

// 多字段原子修改：无共享别名，块退出时自动校验/证明整体不变量
pub fn shift_to_prev_year_safe(inout d: Date) {
    mutate d {
        d.year -= 1
        if d.month == 2 && d.day == 29 && !is_leap_year(d.year) {
            d.day = 28
        }
    } // 编译器自动证明退出后 d 满足 Date 不变量！
}
```

### 系统级版本演化与兼容性自动推导示例

假设已部署版本 `v1` 的 `Order` 金额为 `amount >= 0`，新版本 `v2` 收紧为 `amount > 0` 并新增可选备注字段 `note`：

```tenon
// v1 (记录在已部署基线的 tenon.evolution.lock 中)
@evolve("billing.Order", storage = "orders")
pub struct Order@v1 {
    @field(1) id: Int64,
    @field(2) amount: Int64 where self >= 0,
}

// v2 (当前候选源码)
@evolve("billing.Order", storage = "orders")
pub struct Order {
    @field(1) id: Int64,
    @field(2) amount: Int64 where self > 0,      // 收紧！编译器自动算出残差域 Δ: amount == 0
    @field(3) note: Option[Text] = None,         // 新增可选字段：后向兼容，降级需保留
}
```

执行 `tenon evolve check --against prod-v1.receipt` 时，编译器**自动推导**：
- 字段 `@field(3) note`：新增带默认值字段 → **前/后向兼容（降级时标记 `retain_on_downgrade`）**。
- 字段 `@field(2) amount`：`P_v1 (self >= 0) ⇏ P_v2 (self > 0)` → **破坏性收紧**，精确反例与残差域 $\Delta = \{ x \mid x = 0 \}$ → **阻塞发布，要求提供覆盖 $\Delta$ 的已证明迁移透镜**。

开发者声明一等双向演化透镜，由编译器静态验证：

```tenon
evolution Order: v1 <-> v2 {
    archive ZeroOrderArchive {
        order_id: Int64,
        original_amount: Int64 where self == 0,
    }

    // 编译器强制证明：
    // 1. Totality: 对所有满足 v1 的输入均产出合法 v2
    // 2. Round-trip: down(up(x)) == x（含 archive）
    // 3. Retained Re-upgrade: 降级回 v1 后保留 note 并允许 v1 继续写入，再次升级回 v2 不丢数据
    up(old: Order@v1) -> (Order@v2, Option[ZeroOrderArchive]) {
        if old.amount == 0 {
            (Order { id: old.id, amount: 1, note: None },
             Some(ZeroOrderArchive { order_id: old.id, original_amount: 0 }))
        } else {
            // 流敏感精化已知 old.amount > 0，直接零检查构造 Order@v2
            (Order { id: old.id, amount: old.amount, note: None }, None)
        }
    }

    down(new: Order@v2, arch: Option[ZeroOrderArchive]) -> Order@v1
        retain(new.note)  // 显式声明降级时物理保留 note，供再升级合并
    {
        match arch {
            Some(a) if new.amount == 1 => Order@v1 { id: new.id, amount: a.original_amount },
            _ => Order@v1 { id: new.id, amount: new.amount },
        }
    }
}
```

---

## 4. 技术文档导航

| 文档 | 内容摘要 |
| --- | --- |
| [01 — 现状、竞品与近一年（2025–2026）新语言全景调研](docs/01-landscape-and-positioning.md) | 深度复盘 `loom-lang` 进展与结构性瓶颈、`lang-research` 核心实验结论，系统评估近 1 年 20+ 门新语言/验证体系（MoonBit 0.9、Ora、Bend 2、Gojo、Goose、Wyzer、Hylo、Ante、Zed、Blood、Ambient、Jacquard、Mog、Flux、Verus 等）的优缺点与定位空白。 |
| [02 — 强约束精化类型系统设计](docs/02-type-system-and-refinements.md) | 标量/同余/域值/跨字段依赖结构体（`Date`、`Range`、集合、状态机）语法与语义；流敏感精化；原子 `mutate` 块；精化子类型格与零开销消除；从单一真相自动派生边界解析器与 SQL/JSON 约束。 |
| [03 — 三层分层形式化验证引擎设计](docs/03-formal-verification-engine.md) | L0 进程内极速抽象域求解器（区间/同余/八边形/有界 CTFE）+ L1 增量 SMT（Z3/CVC5 + 缓存 + 可解释反例）+ L2 Ghost/Lemma/极小证书；MVS 如何让验证器复杂度下降 10×；面向 AI Vericoding 的结构化验证协议。 |
| [04 — 系统级版本演化与兼容性推导引擎](docs/04-system-evolution-and-compatibility.md) | `@evolve` 双轨身份与 `evolution.lock`；API/Wire/Storage/In-Flight 四维 SMT 兼容性推导矩阵；一等双向演化透镜（Bidirectional Lens）与四大形式化证明定律；在线 Expand-Contract 滚动发布计划推导与 Append-Only 部署账本。 |
| [05 — 内存模型、并发与编译器实现路线](docs/05-memory-concurrency-and-compiler-architecture.md) | MVS + 原生栈 / Goose 多数据栈 / Perceus 原地复用三层内存引擎；`Store[T] + Handle + phase` 可控长期图；显式效应与结构化并发；Query 增量编译器架构、IR 分层、Cranelift(Dev) + LLVM(Release) 双后端、M0–M4 里程碑与可证伪验收门禁。 |

---

## 5. 待讨论的核心架构决策（Open Questions）

创建本设计仓库的核心目的是先在架构层面达成共识，再落第一行编译器代码。建议重点讨论以下 6 个决策点：

1. **与 `loom-lang` 的关系定位**：是将本方案作为 `Loom 2.0` 的彻底重构蓝图，还是作为全新独立语言仓库从零启动（复用 Loom 的验收故事集与测试思想）？
2. **默认语义选择**：是否坚定采用 **MVS（默认可变值语义，无隐式共享可变别名）+ 显式 `Store[T]`**？还是保留类似 Loom 的共享可变集合？*（这是决定验证器复杂度上限的头号分叉点）*
3. **L0 内置求解器的边界**：L0 应当固化哪些可判定理论（区间 + 同余 + Octagon + 有界纯函数展开）？对于整数算术，默认采用**精确无溢出检查（checked int）**还是任意精度语义模型？
4. **内存引擎的首版组合**：在 M1 原型阶段，是先做 **MVS + Perceus RC 复用** 再叠 **Goose 多数据栈**，还是像 `lang-research` 的 `goose-hybrid-gc` 一样保留非移动精准 GC 回退路径？
5. **演化身份的载体**：在普通文本文件中，持久化/协议类型是否接受显式 `@evolve("...")` + 字段编号 `@field(N)`（类似 Protobuf/Avro 但带精化谓词），而内部函数/重构演化靠自动生成的 `evolution.lock` 跟踪？
6. **初始编译器实现语言与后端**：采用 **Rust + Salsa（Query 增量）+ Cranelift（Dev）+ LLVM（Release）**，还是 **Zig + 自研/C 后端**？何时启动自举？
