# ADR-0001：Tenon 语言与编译器核心架构决议总表

- **状态**：已批准（Accepted — 2026-11-23）
- **决策方式**：基于 [`loom-lang`](../../../loom-lang)、[`lang-research`](../../../lang-research) 与 [`dast-os-kernel`](../../../../dast.ai/dast-os-kernel) 的一线工程复盘，逐分支收敛全部 13 项上游设计决策。
- **适用范围**：指导 Tenon M0–M4 的规范制定、编译器实现、标准库协议与验收门禁。

---

## 决议一览表

| 编号 | 决策主题 | 已拍板决议 | 被否决的备选方案 | 关键架构后果 |
| --- | --- | --- | --- | --- |
| **D01** | 战略定位与仓库归属 | **全新独立通用编程语言（Tenon）**：从零设计，融合精化类型、分层验证、系统演化与可验证分布式同步，暂不背自举与历史兼容包袱 | 在 `loom-lang` 上原地重构 Loom 2.0；或仅做非通用分布式 DSL | 可一次性重构底层内存与验证地基，摆脱 14.6 万行自举包袱 |
| **D02** | 语言核心 vs 标准库边界 | **极简通用语义核心 + 标准库可验证协议模板**：核心仅内建精化类型、MVS/线性令牌、显式效应、协议公理(`laws`)验证钩子、`comptime` 特化与 `@evolve` 演化演算；三轴实体、锁序格、Attempt 双围栏、`SyncArchetype` 与响应式流/pacing 均在标准库以可验证模板实现 | 将分布式调和/响应式流硬编码进编译器核心；或完全不要协议模板的纯普通泛型库 | 保持语言核心极简通用（可写系统、金融、编译器、AI OS），分布式协议像库一样可扩充自演化 |
| **D03** | 默认数据与别名语义 | **默认可变值语义（MVS）+ 局部 `inout`（零 `'a` 生命周期标注）+ 显式 `Store[T]/Handle/phase`**（用于长期可变图） | 默认共享可变 + GC（Loom 现方案）；或 Gojo 式无自动 COW 显式所有权 | 彻底消灭普通值上的共享可变别名爆炸，验证引擎复杂度下降 ~10×，天然无循环引用 |
| **D04** | 并发与异步模型 | **无栈协程（Stackless MIR 状态机）+ 显式 `async` 效应（配效应多态减轻高阶函数染色）** | 有栈混合协程（Go 式）；或不内建协程 | 原生高并发服务端、Wasm 前端协调器、无 MMU 嵌入式三端统一；编译器精确掌控挂起点，静态禁止在 `mutate`/锁/`phase` 内挂起 |
| **D05** | 整数算术与验证门禁 | **默认固定宽度 `checked` 整数**（`Int`/`Int64` 等，L0/L1 证明不溢出自动零开销消除检查；显式提供 `BigInt`/`Decimal`/`wrapping`）；**所有 `ensures` 与精化在全模式严格 Fail-Closed** | dev 模式将未证明 `ensures` 降级为运行时断言；或默认 tagged BigInt | 与 C/Rust 硬件标量 1:1 零开销对齐；保持绝对可信的证据诚实度（绝不把未知/超时当通过） |
| **D06** | 版本演化身份与双层兼容 | **双层融合演化 + 名称优先轻量标记**：普通内部代码零注解；持久化/Wire/公开契约仅需简洁 `@evolve`（默认实体 ID 即类型名、**字段身份默认按字段名匹配**，改名用 `@from("old_name")`，仅紧凑二进制编解码才可选加 `@field(N)`）；结合自动 `evolution.lock`、SMT 精化/双向透镜与按职责能力子集 `required(record, op) ⊆ supported(consumer, op)` | 强制在每个字段手写 `@field(N)` 噪声注解；或全源码声明级隐藏 ID 文件（如 `.loom-ids`） | 保持代码高可读性（99% 场景零字段 ID 噪声），同时完整保留 SMT 数学级残差推导、安全改名血缘追踪与无损回滚证明 |
| **D07** | `dast-os` 级分布式内核架构 | **全新“分片单写者 + 存储/代理双端硬 Fencing + 不可变 Turn DAG + 声明式响应式数据流”架构**（底层 Substrate 支持单机 SQLite / 云端 PG / Raft-KV 可插拔） | 沿用共享 SQL 表多写者抢 6 级行锁与原地修改上下文 | 彻底消解多写者锁争用与 20 项强插补丁；防双 Worker 执行由存储 + Proxy 双端硬 `exec_epoch` 保证；`Interrupt`/`Fork`/`Reset` 统一为 Turn DAG 分叉 |
| **D08** | 多端客户端编译策略 | **单源编译为统一二进制执行核**：Web 编译为 **Wasm + 自动生成薄 TS 类型胶水**（靠声明式 `33ms` 帧合并摊薄边界拷贝），iOS/Android/Desktop 共用 **Native C-ABI 静态库 + 自动生成薄 Swift/Kotlin 绑定** | Web 单独维护 JS 后端；或三端分别转译 TS/Swift/Kotlin 源码 | 后端维护成本最低，四端 100% 同构执行同一份已验证 `SyncCoordinator` 与增量叠加引擎 |
| **D09** | 应对 Goose 内存限制的主次定位 | **以 MVS + Perceus 精确引用计数与唯一引用原地复用（COW/FBIP）作为无限制语义基座**；将 Goose 式区域降级为**透明的 Tier-2 局部优化**（默认分块池化 Slab Arena 兼容 Wasm/嵌入式，不引入递归/闭包/VMA 限制），配强制编译报错的 `@no_heap_alloc` / `@fip` 契约 | 引入显式用户 `region` 块；或坚持 Goose 数据栈为主模型 | 100% 解除 Goose 的递归、逃逸闭包、Wasm 32 位地址与独立删除限制，同时在热点流水线保留 0-malloc 性能与可审计硬契约 |
| **D10** | 协议形式化建模与验证引擎 | **双引擎同源协议验证（`tenon modelcheck`）**：从 `VIR` 直接驱动**内置轻快 Rust 有界状态探索器**（秒级 IDE 反馈与源码级反例轨迹）+ **一等 TLA+/Quint 后端导出**（深度探索与时序活性复核），并内建自动负控变异与实跑轨迹精化核验（Trace Refinement） | 仅导出外部 TLA+/Quint；或纯自研不导出外部交叉复核 | 既具备开发期秒级协议排错体验，又保留 TLA+/Quint 工业级交叉核验与“实跑轨迹 ↔ 协议模型”一致性闭环 |
| **D11** | 编译器实现栈与自举策略 | **初始编译器采用 Rust + Salsa（Query 增量架构）+ Cranelift（Dev 极速构建 & Wasm）+ LLVM（Release 峰值优化）**；待 M3 核心语义稳定后再评估自举 | Zig 自研后端；或 M1 过早自举 | 开发期编译追平 Go/Zig，发布期执行追平 Rust/C，同时在探索期保持极高迭代效率 |
| **D12** | 源码语法风格 | **延续 `loom-lang` / Go 风格**：`name Type`（无冒号）、`[]` 泛型、`record` / `enum`、无返回箭头的简洁函数签名（无返回值省略返回类型）、极简关键字、换行分隔记录字段 | 类 Rust/Swift 的 `name: Type` + `struct` + `->`；或 Python 缩进风格 | 与 `loom-lang` 与 Go 审美无缝衔接，视觉噪声低、可读性强且 AI 编辑鲁棒 |
| **D13** | 实施与验收推进节奏 | **先完成 `tenon-lang` 设计定稿（ADR + 统一 `loom/Go` 语法 + 修正内存/协议验证设计），随后按 M0 → M1 → M2 → M3 → M4 稳步推进**：M0 精化+L0/L1+Cranelift → M1 MVS/Perceus → M2 演化透镜+协议模型检验 → M3 通用同步微内核与多端 Wasm/Native | M0 并行堆叠所有高风险模块；或暂不落代码 | 每一里程碑均有清晰、可证伪的端到端验收门禁与基准对照，风险逐层收敛 |

---

## 规范语法速查（基于 D12 定稿）

后续全部规范、示例代码与解析器均严格遵循以下 `loom/Go` 风格语法约定：

```tenon
// 1. 精化类型与名义类型
pub type Amount = Int64 where self > 0
pub distinct type OrderId = Int64 where self > 0

// 2. 记录与跨字段不变量（name Type，无冒号，换行分隔字段）
pub record Date {
    year  Int where 1 <= self && self <= 9999
    month Int where 1 <= self && self <= 12
    day   Int where 1 <= self && self <= days_in_month(year, month)
}

pub record DateRange {
    start Date
    end   Date
    where self.start <= self.end
}

// 3. 函数签名（无冒号、无 ->，无返回值省略返回类型；支持 pure / inout / requires / ensures）
pub pure fn add_odds(a Odd, b Odd) Even {
    a + b
}

pub fn shift_range(inout r DateRange, delta_days Int)
    requires delta_days >= -36500 && delta_days <= 36500
    ensures r.start <= r.end
{
    mutate r {
        r.start = add_days(r.start, delta_days)
        r.end   = add_days(r.end, delta_days)
    }
}
```
