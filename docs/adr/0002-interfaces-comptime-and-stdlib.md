# ADR-0002：接口与实现（静态/动态双轨）、`comptime` 派生序列化、轻量 `@evolve` 与标准库分层架构

- **状态**：已批准（Accepted — 2026-11-24）
- **关联文档**：[ADR-0001](0001-architecture-decisions.md)（延续 D02 极简核心、D03 MVS + Perceus、D05 Fail-Closed 精化契约、D12 `loom/Go` 简洁语法）

---

## 1. 接口与实现：统一 `interface` + 显式 `impl` + 静态/`dyn` 双轨

### 1.1 语法与定位
- **`interface`**：定义行为与形式化契约（区别于专用于分布式状态机的 `protocol state_machine`），允许方法携带 `pure`、`async`、`@no_heap_alloc`、`requires`、`ensures` 与关联类型。
- **显式 `impl Type { ... }`**：为任意类型（`record` / `enum` / 内置类型）提供固有方法（支持 `x.method(args)` 流畅调用）。
- **显式 `impl Interface for Type { ... }`**：显式声明实现并作为 L0/L1 契约证明（Liskov 行为子类型检查）的编译期锚点。

```tenon
pub interface Tool {
    pure fn name(self) Text
    pure fn schema(self) JsonValue
    fn execute(self, workspace Text, args JsonValue) Result[Text, Text]
}

pub record ShellTool {
    timeout_ms Positive
}

impl Tool for ShellTool {
    pure fn name(self) Text { "shell" }
    pure fn schema(self) JsonValue { ... }
    fn execute(self, workspace Text, args JsonValue) Result[Text, Text] { ... }
}
```

### 1.2 静态分发 vs 动态分发（同一份 `interface`，按需选择）
- **静态接口（零开销单态化）**：
  - 语法：泛型约束 `fn run[T Tool](t T)` 或 `fn write_json[W Writer](inout w W, v JsonValue)`
  - 语义：编译期单态化特化，0 虚表、0 间接调用、0 装箱，完全兼容 `@no_heap_alloc`。
- **动态接口（`dyn Interface`）**：
  - 语法：显式存在类型 `dyn Tool`、`List[dyn Tool]`、`dyn LlmProvider`、`dyn FileSystem`
  - 语义：底层表示为 **Perceus 2-field 胖对象 `(data_handle, vtable_ptr)`**（`rc_mask` 追踪 `data_handle` 是否为 RC 对象），完全遵循 MVS 值语义与 Perceus 自动零泄漏回收；静态静态检查对象安全性（`E0506`），并在 `@no_heap_alloc` 中禁止隐式装箱（`E0701`）。

### 1.3 编译期静态校验规则（`tenon-verify`）
- `E0501`：缺少 `interface` 必需方法
- `E0502`：`impl` 方法签名（参数个数、`inout`、返回类型）与 `interface` 不一致
- `E0503`：`impl` 违背 `interface` 的效应要求（如接口要求 `pure` 或 `@no_heap_alloc`）
- `E0504`：`impl` 违背 `interface` 的行为契约（L0/L1 证明 `ensures` / `requires` 不满足 Liskov 替换）
- `E0505`：类型未实现所要求的 `interface`（泛型约束或 `dyn I` 转换）
- `E0506`：`dyn I` 使用了非对象安全的 `interface`

---

## 2. 自定义结构序列化：`comptime` 元信息 + 显式 `@derive` + 精化单源真相

- **接口解耦**：`ToJson` / `FromJson` / `JsonSchema`（及通用 `Serialize`/`Deserialize` 协议）。
- **显式 `@derive(ToJson, FromJson, JsonSchema)`**：
  - 编译器基于记录/枚举的编译期字段与变体元信息（`fields_of` / `variants_of` 模型），自动生成标准 `impl`：
    - 支持 `@rename("...")`、`@skip`、`@skip_if_none`、字段默认值、`@tag("type")` 内部标签枚举；
    - 反序列化自动过精化不变量门禁（`Date`、`SliceRange`、`SafeRelPath`、`TokenUsage` 等字段与 `where` 约束），返回带路径的结构化错误（`JsonError`），绝不绕过精化类型；
    - `@derive(JsonSchema)` 从同一份 `record` 定义自动输出标准 JSON Schema（含从精化区间推导的 `minimum`/`maximum`），彻底消除 LLM Tool Schema 与参数解析之间的漂移。

---

## 3. 版本驱动的轻量 `@evolve`（修订 ADR-0001 D06）

- **以版本对 `(v1, v2)` 为演化依据**，记录定义本身保持 100% 纯净可读，无需在每个字段上写 `@field(N)`；
- `@evolve` 参数可选，缺省即以记录名作为实体 ID；
- 同名字段默认按字段名对齐，`tenon evolve check` 同时检查 **字段类型变更**（如 `Text -> Int64`、`Int64 -> Option[Int64]`）、**精化范围收窄/放宽**、**新增/删除字段**；
- 跨版本字段改类型、改范围、改名或拆分，统一收敛到独立的 `evolution T: v1 <-> v2 { up / down }` 双向透镜中声明并交由 Z3 证明（`@from("old")` 与 `@field(N)` 仅作为可选辅助）。

---

## 4. 四层正交标准库架构（`std/` 与应用/SDK 解耦）

```text
std/
├── core/         # Layer 0: 零 OS 依赖、纯计算（types, ops, option, result）
├── text/         # Layer 1: 文本方法、O(n) TextBuilder、数字解析
├── collections/  # Layer 1: 值语义容器（Map 等，Perceus COW）
├── encoding/     # Layer 2: json (JsonValue + ToJson/FromJson/JsonSchema + @derive), dotenv
└── os/           # Layer 3: io (Reader/Writer), path (SafeRelPath), fs (FileSystem/OsFs/FsError), env, process, http (通用 HTTP/SSE)
```

- 业务/AI SDK（`LlmProvider`/`OpenAIProvider`、`Tool`/`ShellTool`/文件工具）从 `std/` 移出，放入 `examples/agent/lib/`，保持语言标准库纯净通用。
- `tenon-rt` 将 `sys_http_sse_open(url, api_key, body)` 通用化为 `sys_http_open(method, url, headers, body)`，彻底移除运行时中任何特定厂商协议细节。
