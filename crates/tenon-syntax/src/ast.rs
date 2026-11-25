use crate::span::Span;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub items: Vec<Item>,
    #[serde(default)]
    pub source_files: Vec<String>,
    /// Maps `file_id` (index into `source_files`) -> `package_id`.
    /// Files within the same directory package share the same `package_id`.
    #[serde(default)]
    pub file_packages: Vec<u32>,
    /// File IDs that came from `*_test.tn` files (production files may never depend on them).
    #[serde(default)]
    pub test_file_ids: Vec<u32>,
    /// `[configuration]` key-value map from `tenon.toml` for compile-time queries.
    #[serde(default)]
    pub build_config: BTreeMap<String, String>,
    /// Preserved compile-time-only items (`macro`, `comptime fn`) after expansion for IDE/LSP navigation.
    #[serde(default)]
    pub comptime_items: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportNamed {
    pub name: String,
    #[serde(default)]
    pub alias: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImportKind {
    /// Legacy string path: `import "std/text/text.tn" [as t]`
    FilePath(String),
    /// Go-style dotted import: `import pkg.subpkg [as alias]` (or item import `import pkg.subpkg.Item`)
    Package(Vec<String>),
    /// Wildcard import: `import pkg.subpkg.*`
    Glob(Vec<String>),
    /// Selective import: `import pkg.subpkg.{Type1, Type2 as T2, fn1}`
    Selective {
        segments: Vec<String>,
        names: Vec<ImportNamed>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MacroDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub name: String,
    pub params: Vec<String>,
    pub body: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttrDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub name: String,
    pub params: Vec<ParamDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attr {
    pub name: String,
    pub args: Vec<AttrArg>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttrArg {
    pub key: Option<String>,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Item {
    Import {
        path: String,
        #[serde(default)]
        alias: Option<String>,
        #[serde(default)]
        kind: Option<ImportKind>,
        span: Span,
    },
    Macro(MacroDecl),
    AttrDecl(AttrDecl),
    ComptimeBlock(Block),
    MacroCall {
        name: String,
        args: Vec<Expr>,
        span: Span,
    },
    TypeAlias(TypeAliasDecl),
    Record(RecordDecl),
    Enum(EnumDecl),
    Interface(InterfaceDecl),
    Impl(ImplDecl),
    Fn(FnDecl),
    ExternFn(ExternFnDecl),
    Bitfield(BitfieldDecl),
    Union(UnionDecl),
    Const(ConstDecl),
    Evolution(EvolutionDecl),
    LockLattice(LockLatticeDecl),
    Protocol(ProtocolDecl),
    Stream(StreamDecl),
    View(ViewDecl),
    Subscriber(SubscriberDecl),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterfaceDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub name: String,
    pub type_params: Vec<String>,
    #[serde(default)]
    pub type_param_bounds: BTreeMap<String, Vec<String>>,
    pub assoc_types: Vec<String>,
    pub methods: Vec<InterfaceMethodDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterfaceMethodDecl {
    pub attrs: Vec<Attr>,
    pub is_pure: bool,
    pub is_async: bool,
    pub name: String,
    pub type_params: Vec<String>,
    #[serde(default)]
    pub type_param_bounds: BTreeMap<String, Vec<String>>,
    pub params: Vec<ParamDecl>,
    pub ret_ty: Option<TypeExpr>,
    pub requires: Vec<Expr>,
    pub ensures: Vec<Expr>,
    pub default_body: Option<Block>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImplDecl {
    pub attrs: Vec<Attr>,
    pub type_params: Vec<String>,
    #[serde(default)]
    pub type_param_bounds: BTreeMap<String, Vec<String>>,
    /// `None` for inherent `impl Target { ... }`, `Some(Iface)` for `impl Iface for Target { ... }`.
    pub interface: Option<TypeExpr>,
    pub target_ty: TypeExpr,
    pub assoc_bindings: Vec<(String, TypeExpr)>,
    pub methods: Vec<FnDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeAliasDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub is_distinct: bool,
    pub name: String,
    pub type_params: Vec<String>,
    #[serde(default)]
    pub type_param_defaults: BTreeMap<String, TypeExpr>,
    #[serde(default)]
    pub variadic_param: Option<String>,
    pub base: TypeExpr,
    pub predicate: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub name: String,
    pub type_params: Vec<String>,
    #[serde(default)]
    pub type_param_bounds: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub type_param_defaults: BTreeMap<String, TypeExpr>,
    pub fields: Vec<FieldDecl>,
    pub where_clauses: Vec<Expr>,
    pub invariants: Vec<NamedExpr>,
    pub methods: Vec<FnDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDecl {
    pub attrs: Vec<Attr>,
    pub axis: Option<String>,
    pub name: String,
    pub ty: TypeExpr,
    pub predicate: Option<Expr>,
    pub default: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnumDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub name: String,
    pub type_params: Vec<String>,
    #[serde(default)]
    pub type_param_bounds: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub type_param_defaults: BTreeMap<String, TypeExpr>,
    pub variants: Vec<VariantDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariantDecl {
    pub name: String,
    pub fields: Vec<FieldDecl>,
    pub predicate: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FnDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub is_pure: bool,
    pub is_async: bool,
    pub is_test: bool,
    #[serde(default)]
    pub is_unsafe: bool,
    #[serde(default)]
    pub is_comptime: bool,
    #[serde(default)]
    pub is_ghost: bool,
    pub name: String,
    pub type_params: Vec<String>,
    #[serde(default)]
    pub type_param_bounds: BTreeMap<String, Vec<String>>,
    pub params: Vec<ParamDecl>,
    pub ret_ty: Option<TypeExpr>,
    pub requires: Vec<Expr>,
    pub ensures: Vec<Expr>,
    pub body: Block,
    pub span: Span,
}

impl FnDecl {
    /// Returns effect polymorphism parameter names declared via `[..., effect E]` (ADR-0001 D04).
    pub fn effect_params(&self) -> Vec<String> {
        self.type_param_bounds
            .iter()
            .filter_map(|(k, bs)| {
                if bs.iter().any(|b| b == "__effect__") {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns declared effect names from trailing `/ E, io, async` clause on the function signature.
    pub fn declared_effects(&self) -> Vec<String> {
        let mut out = Vec::new();
        for a in &self.attrs {
            if a.name == "__effects" {
                for arg in &a.args {
                    if let ExprKind::Str(s) | ExprKind::Ident(s) = &arg.value.kind {
                        out.push(s.clone());
                    }
                }
            }
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExternFnDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub abi: String,
    pub name: String,
    pub params: Vec<ParamDecl>,
    pub ret_ty: Option<TypeExpr>,
    #[serde(default)]
    pub is_variadic: bool,
    pub link_lib: Option<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BitfieldFieldDecl {
    pub name: String,
    pub bits: u32,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BitfieldDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub name: String,
    pub backing_ty: TypeExpr,
    pub fields: Vec<BitfieldFieldDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnionDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub name: String,
    pub fields: Vec<FieldDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub name: String,
    pub ty: TypeExpr,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamDecl {
    pub is_inout: bool,
    pub is_comptime: bool,
    pub name: String,
    pub ty: TypeExpr,
    pub predicate: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvolutionDecl {
    pub attrs: Vec<Attr>,
    pub entity: String,
    pub from_ver: String,
    pub to_ver: String,
    pub policy: Vec<(String, Expr)>,
    pub archives: Vec<RecordDecl>,
    pub up: Option<FnDecl>,
    pub down: Option<FnDecl>,
    pub retain_fields: Vec<String>,
    pub allow_loss_fields: Vec<String>,
    pub reup: Option<FnDecl>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LockLatticeDecl {
    pub is_pub: bool,
    pub name: String,
    pub levels: Vec<String>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtocolVarDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub init: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtocolActionDecl {
    pub name: String,
    pub is_fault: bool,
    pub guard: Option<Expr>,
    pub updates: Vec<(String, Expr)>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtocolDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub kind: Option<String>,
    pub name: String,
    pub type_params: Vec<String>,
    pub satisfies: Vec<Expr>,
    pub faults: Vec<String>,
    pub state_vars: Vec<ProtocolVarDecl>,
    pub actions: Vec<ProtocolActionDecl>,
    pub fault_actions: Vec<ProtocolActionDecl>,
    pub invariants: Vec<NamedExpr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub name: String,
    pub params: Vec<ParamDecl>,
    pub item_ty: TypeExpr,
    pub entries: Vec<(String, Expr)>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub is_reactive: bool,
    pub name: String,
    pub params: Vec<ParamDecl>,
    pub source: Option<String>,
    pub properties: Vec<(String, Expr)>,
    pub bounds: Vec<(String, Expr)>,
    pub body: Option<Block>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubscriberDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub name: String,
    pub params: Vec<ParamDecl>,
    pub view_name: Option<String>,
    pub pacing: Vec<(String, Expr)>,
    pub storage: Vec<(String, Expr)>,
    pub properties: Vec<(String, Expr)>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedExpr {
    pub name: String,
    pub expr: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeExpr {
    Named {
        name: String,
        version: Option<String>,
        args: Vec<TypeExpr>,
        span: Span,
    },
    Dyn {
        interface: String,
        args: Vec<TypeExpr>,
        span: Span,
    },
    Tuple(Vec<TypeExpr>, Span),
    Fn {
        params: Vec<TypeExpr>,
        ret: Option<Box<TypeExpr>>,
        span: Span,
    },
    Array {
        elem: Box<TypeExpr>,
        len: usize,
        span: Span,
    },
}

impl TypeExpr {
    /// Constructs a keyword-named type argument (e.g. `DType = Float16` in `Matrix[4, 8, DType = Float16]`).
    pub fn named_arg(key: String, value: TypeExpr, span: Span) -> Self {
        TypeExpr::Named {
            name: format!("__named__:{}", key),
            version: None,
            args: vec![value],
            span,
        }
    }

    /// Extracts `(key, value)` if this `TypeExpr` is a keyword-named type argument (`DType = Float16`).
    pub fn as_named_arg(&self) -> Option<(&str, &TypeExpr)> {
        if let TypeExpr::Named { name, args, .. } = self {
            if let Some(key) = name.strip_prefix("__named__:") {
                if args.len() == 1 {
                    return Some((key, &args[0]));
                }
            }
        }
        None
    }

    pub fn display_name(&self) -> String {
        if let Some((k, v)) = self.as_named_arg() {
            return format!("{} = {}", k, v.display_name());
        }
        match self {
            TypeExpr::Named {
                name,
                version,
                args,
                ..
            } => {
                let mut out = name.clone();
                if let Some(v) = version {
                    out.push('@');
                    out.push_str(v);
                }
                if !args.is_empty() {
                    out.push('[');
                    for (i, a) in args.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        out.push_str(&a.display_name());
                    }
                    out.push(']');
                }
                out
            }
            TypeExpr::Dyn {
                interface, args, ..
            } => {
                let mut out = format!("dyn {}", interface);
                if !args.is_empty() {
                    out.push('[');
                    for (i, a) in args.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        out.push_str(&a.display_name());
                    }
                    out.push(']');
                }
                out
            }
            TypeExpr::Tuple(elems, _) => {
                let parts: Vec<_> = elems.iter().map(|e| e.display_name()).collect();
                format!("({})", parts.join(", "))
            }
            TypeExpr::Fn { params, ret, .. } => {
                let parts: Vec<_> = params.iter().map(|e| e.display_name()).collect();
                match ret {
                    Some(r) => format!("fn({}) {}", parts.join(", "), r.display_name()),
                    None => format!("fn({})", parts.join(", ")),
                }
            }
            TypeExpr::Array { elem, len, .. } => {
                format!("Array[{}, {}]", elem.display_name(), len)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub tail: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ForBinding {
    Ident(String),
    Tuple(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LambdaParam {
    pub name: String,
    pub ty: Option<TypeExpr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Stmt {
    Let {
        name: String,
        ty: Option<TypeExpr>,
        value: Expr,
        is_mut: bool,
        span: Span,
    },
    LetTuple {
        names: Vec<String>,
        ty: Option<TypeExpr>,
        value: Expr,
        is_mut: bool,
        span: Span,
    },
    Assign {
        target: Expr,
        op: AssignOp,
        value: Expr,
        span: Span,
    },
    Return {
        value: Option<Expr>,
        span: Span,
    },
    Assert {
        cond: Expr,
        span: Span,
    },
    Mutate {
        target: String,
        body: Block,
        span: Span,
    },
    While {
        cond: Expr,
        invariants: Vec<Expr>,
        decreases: Option<Expr>,
        body: Block,
        span: Span,
    },
    For {
        binding: ForBinding,
        iter: Expr,
        body: Block,
        span: Span,
    },
    Break {
        span: Span,
    },
    Continue {
        span: Span,
    },
    Defer {
        body: Block,
        span: Span,
    },
    Ghost {
        body: Block,
        span: Span,
    },
    Expr {
        expr: Expr,
        span: Span,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssignOp {
    Set,
    AddSet,
    SubSet,
    MulSet,
    DivSet,
    BitAndSet,
    BitOrSet,
    BitXorSet,
    ShlSet,
    ShrSet,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExprKind {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Ident(String),
    Binary {
        op: BinOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Unary {
        op: UnOp,
        expr: Box<Expr>,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<CallArg>,
    },
    GenericInst {
        expr: Box<Expr>,
        type_args: Vec<TypeExpr>,
    },
    Field {
        target: Box<Expr>,
        field: String,
    },
    TupleIndex {
        target: Box<Expr>,
        index: usize,
    },
    Index {
        target: Box<Expr>,
        index: Box<Expr>,
    },
    Try(Box<Expr>),
    Range {
        start: Box<Expr>,
        end: Box<Expr>,
    },
    Lambda {
        params: Vec<LambdaParam>,
        ret_ty: Option<TypeExpr>,
        body: Box<Expr>,
    },
    RecordLit {
        name: String,
        version: Option<String>,
        fields: Vec<(String, Expr)>,
    },
    Tuple(Vec<Expr>),
    List(Vec<Expr>),
    ArrayRepeat {
        value: Box<Expr>,
        len: usize,
    },
    Cast {
        expr: Box<Expr>,
        ty: TypeExpr,
    },
    TryCast {
        expr: Box<Expr>,
        ty: TypeExpr,
    },
    Unsafe(Block),
    Comptime(Block),
    ComptimeIf {
        cond: Box<Expr>,
        then_block: Block,
        else_branch: Option<Box<Expr>>,
    },
    ComptimeFor {
        binding: String,
        iter: Box<Expr>,
        body: Block,
    },
    MacroCall {
        name: String,
        args: Vec<Expr>,
    },
    TypeImplements {
        ty: TypeExpr,
        interface: String,
    },
    If {
        cond: Box<Expr>,
        then_block: Block,
        else_branch: Option<Box<Expr>>,
    },
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    Block(Block),
    Await(Box<Expr>),
    Old(Box<Expr>),
    Quantifier {
        is_forall: bool,
        var: String,
        lo: Box<Expr>,
        hi: Box<Expr>,
        body: Box<Expr>,
    },
    Forall {
        binders: Vec<(String, TypeExpr)>,
        guard: Option<Box<Expr>>,
        body: Box<Expr>,
    },
    UnitValue {
        value: i64,
        unit: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CallArg {
    pub name: Option<String>,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchArm {
    pub patterns: Vec<Pattern>,
    pub guard: Option<Expr>,
    pub body: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Pattern {
    Wildcard(Span),
    Int(i64, Span),
    Bool(bool, Span),
    Str(String, Span),
    Ident(String, Span),
    Variant {
        name: String,
        binding: Option<String>,
        span: Span,
    },
    VariantMulti {
        name: String,
        patterns: Vec<Pattern>,
        span: Span,
    },
    Tuple(Vec<Pattern>, Span),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinOp {
    Or,
    And,
    Implies,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Concat,
    BitOr,
    BitXor,
    BitAnd,
    Shl,
    Shr,
}

impl BinOp {
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Or => "||",
            BinOp::And => "&&",
            BinOp::Implies => "=>",
            BinOp::Eq => "==",
            BinOp::Ne => "!=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::Concat => "++",
            BinOp::BitOr => "|",
            BinOp::BitXor => "^",
            BinOp::BitAnd => "&",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnOp {
    Neg,
    Not,
    BitNot,
}

/// Resolves positional, keyword-named (`DType = Float16`), default (`DType = Float32`),
/// and variadic const-pack (`...Shape Int`) type arguments into `(param_bindings, variadic_pack)`.
pub fn resolve_type_args(
    type_params: &[String],
    defaults: &BTreeMap<String, TypeExpr>,
    variadic_param: Option<&str>,
    raw_args: &[TypeExpr],
) -> (BTreeMap<String, TypeExpr>, Vec<TypeExpr>) {
    let mut bindings: BTreeMap<String, TypeExpr> = BTreeMap::new();
    let mut positional: Vec<TypeExpr> = Vec::new();
    for arg in raw_args {
        if let Some((k, v)) = arg.as_named_arg() {
            bindings.insert(k.to_string(), v.clone());
        } else {
            positional.push(arg.clone());
        }
    }
    let mut variadic_pack: Vec<TypeExpr> = Vec::new();
    if let Some(vp) = variadic_param {
        let vp_idx = type_params.iter().position(|p| p == vp).unwrap_or(0);
        let mut pos_iter = positional.into_iter();
        // Bind fixed parameters appearing before `...vp`
        for p in type_params.iter().take(vp_idx) {
            if !bindings.contains_key(p) {
                if let Some(a) = pos_iter.next() {
                    bindings.insert(p.clone(), a);
                }
            }
        }
        // Remaining positional arguments form the variadic pack (e.g. `Tensor[1, 1, 2, 3, 4]`)
        variadic_pack.extend(pos_iter);
    } else {
        let mut pos_iter = positional.into_iter();
        for p in type_params {
            if !bindings.contains_key(p) {
                if let Some(a) = pos_iter.next() {
                    bindings.insert(p.clone(), a);
                }
            }
        }
    }
    // Fill any omitted parameters from defaults (`DType = Float32`)
    for p in type_params {
        if variadic_param != Some(p.as_str()) && !bindings.contains_key(p) {
            if let Some(def) = defaults.get(p) {
                bindings.insert(p.clone(), def.clone());
            }
        }
    }
    (bindings, variadic_pack)
}

/// Recursively substitutes type parameters inside a `TypeExpr` (e.g. `Tensor2D[DType]` -> `Tensor2D[Float16]`).
pub fn instantiate_type_expr(ty: &TypeExpr, subst: &BTreeMap<String, TypeExpr>) -> TypeExpr {
    if subst.is_empty() {
        return ty.clone();
    }
    if let Some((k, v)) = ty.as_named_arg() {
        let span = match ty {
            TypeExpr::Named { span, .. } => *span,
            _ => Span::default(),
        };
        return TypeExpr::named_arg(k.to_string(), instantiate_type_expr(v, subst), span);
    }
    match ty {
        TypeExpr::Named { name, version, args, span } => {
            if args.is_empty() && version.is_none() {
                if let Some(rep) = subst.get(name) {
                    return rep.clone();
                }
            }
            TypeExpr::Named {
                name: name.clone(),
                version: version.clone(),
                args: args.iter().map(|a| instantiate_type_expr(a, subst)).collect(),
                span: *span,
            }
        }
        TypeExpr::Dyn { interface, args, span } => TypeExpr::Dyn {
            interface: interface.clone(),
            args: args.iter().map(|a| instantiate_type_expr(a, subst)).collect(),
            span: *span,
        },
        TypeExpr::Tuple(es, span) => {
            TypeExpr::Tuple(es.iter().map(|e| instantiate_type_expr(e, subst)).collect(), *span)
        }
        TypeExpr::Fn { params, ret, span } => TypeExpr::Fn {
            params: params.iter().map(|p| instantiate_type_expr(p, subst)).collect(),
            ret: ret.as_ref().map(|r| Box::new(instantiate_type_expr(r, subst))),
            span: *span,
        },
        TypeExpr::Array { elem, len, span } => TypeExpr::Array {
            elem: Box::new(instantiate_type_expr(elem, subst)),
            len: *len,
            span: *span,
        },
    }
}

/// Instantiates the base type of a `TypeAliasDecl` using positional, keyword-named, and default arguments
/// (e.g. `Matrix[4, 8]` -> `Tensor2D[Float32]`, `Matrix[4, 8, DType = Float16]` -> `Tensor2D[Float16]`).
pub fn instantiate_alias_base(ta: &TypeAliasDecl, raw_args: &[TypeExpr]) -> TypeExpr {
    if ta.type_params.is_empty() {
        return ta.base.clone();
    }
    let (bindings, _) = resolve_type_args(
        &ta.type_params,
        &ta.type_param_defaults,
        ta.variadic_param.as_deref(),
        raw_args,
    );
    instantiate_type_expr(&ta.base, &bindings)
}

fn type_arg_to_const_expr(arg: &TypeExpr) -> Option<Expr> {
    if let TypeExpr::Named { name, args: inner, span, .. } = arg {
        if inner.is_empty() {
            if let Ok(iv) = name.parse::<i64>() {
                return Some(Expr { kind: ExprKind::Int(iv), span: *span });
            } else if let Ok(fv) = name.parse::<f64>() {
                return Some(Expr { kind: ExprKind::Float(fv), span: *span });
            } else if !name.is_empty() {
                return Some(Expr { kind: ExprKind::Ident(name.clone()), span: *span });
            }
        }
    }
    None
}

/// Instantiates const/value parameters & variadic dimension packs (`...Shape Int`) on a parameterized
/// refinement `TypeAliasDecl` (e.g. `Matrix[4, 8, DType = Float16]`, `Tensor[1, 1, 2, 3, 4]`).
pub fn instantiate_alias_predicate(ta: &TypeAliasDecl, args: &[TypeExpr]) -> Option<Expr> {
    let pred = ta.predicate.as_ref()?;
    if ta.type_params.is_empty() {
        return Some(pred.clone());
    }
    let (bindings, variadic_pack) = resolve_type_args(
        &ta.type_params,
        &ta.type_param_defaults,
        ta.variadic_param.as_deref(),
        args,
    );
    let mut const_map: BTreeMap<String, Expr> = BTreeMap::new();
    for (tp, arg) in &bindings {
        if let Some(ce) = type_arg_to_const_expr(arg) {
            if !matches!(&ce.kind, ExprKind::Ident(s) if s == tp) {
                const_map.insert(tp.clone(), ce);
            }
        }
    }
    let mut out = pred.clone();
    if let Some(vp) = &ta.variadic_param {
        let pack_exprs: Vec<Expr> = variadic_pack
            .iter()
            .filter_map(type_arg_to_const_expr)
            .collect();
        out = subst_variadic_pack_in_expr(&out, vp, &pack_exprs);
    }
    if const_map.is_empty() {
        Some(out)
    } else {
        Some(subst_const_idents_in_expr(&out, &const_map))
    }
}

fn subst_variadic_pack_in_expr(expr: &Expr, pack_name: &str, pack: &[Expr]) -> Expr {
    let sp = expr.span;
    match &expr.kind {
        // `len(Shape)` -> constant rank `Int(pack.len())`
        ExprKind::Call { callee, args }
            if args.len() == 1
                && matches!(&callee.kind, ExprKind::Ident(f) if f == "len")
                && matches!(&args[0].value.kind, ExprKind::Ident(s) if s == pack_name) =>
        {
            Expr { kind: ExprKind::Int(pack.len() as i64), span: sp }
        }
        // `prod(Shape)` -> folded integer product (or symbolic `d0 * d1 * ...` product!)
        ExprKind::Call { callee, args }
            if args.len() == 1
                && matches!(&callee.kind, ExprKind::Ident(f) if f == "prod")
                && matches!(&args[0].value.kind, ExprKind::Ident(s) if s == pack_name) =>
        {
            if pack.is_empty() {
                return Expr { kind: ExprKind::Int(1), span: sp };
            }
            let all_ints: Option<i64> = pack.iter().try_fold(1i64, |acc, e| match e.kind {
                ExprKind::Int(n) => acc.checked_mul(n),
                _ => None,
            });
            if let Some(p) = all_ints {
                return Expr { kind: ExprKind::Int(p), span: sp };
            }
            let mut acc = pack[0].clone();
            for item in &pack[1..] {
                acc = Expr {
                    kind: ExprKind::Binary {
                        op: BinOp::Mul,
                        left: Box::new(acc),
                        right: Box::new(item.clone()),
                    },
                    span: sp,
                };
            }
            acc
        }
        // `Shape[i]` with constant index -> `pack[i]`
        ExprKind::Index { target, index }
            if matches!(&target.kind, ExprKind::Ident(s) if s == pack_name) =>
        {
            if let ExprKind::Int(idx) = index.kind {
                if idx >= 0 && (idx as usize) < pack.len() {
                    return pack[idx as usize].clone();
                }
            }
            expr.clone()
        }
        // `Shape` as a list literal `[d0, d1, ...]`
        ExprKind::Ident(s) if s == pack_name => Expr {
            kind: ExprKind::List(pack.to_vec()),
            span: sp,
        },
        ExprKind::Binary { op, left, right } => Expr {
            kind: ExprKind::Binary {
                op: *op,
                left: Box::new(subst_variadic_pack_in_expr(left, pack_name, pack)),
                right: Box::new(subst_variadic_pack_in_expr(right, pack_name, pack)),
            },
            span: sp,
        },
        ExprKind::Unary { op, expr: e } => Expr {
            kind: ExprKind::Unary {
                op: *op,
                expr: Box::new(subst_variadic_pack_in_expr(e, pack_name, pack)),
            },
            span: sp,
        },
        ExprKind::Call { callee, args } => Expr {
            kind: ExprKind::Call {
                callee: Box::new(subst_variadic_pack_in_expr(callee, pack_name, pack)),
                args: args
                    .iter()
                    .map(|a| CallArg {
                        name: a.name.clone(),
                        value: subst_variadic_pack_in_expr(&a.value, pack_name, pack),
                    })
                    .collect(),
            },
            span: sp,
        },
        ExprKind::Field { target, field } => Expr {
            kind: ExprKind::Field {
                target: Box::new(subst_variadic_pack_in_expr(target, pack_name, pack)),
                field: field.clone(),
            },
            span: sp,
        },
        ExprKind::Index { target, index } => Expr {
            kind: ExprKind::Index {
                target: Box::new(subst_variadic_pack_in_expr(target, pack_name, pack)),
                index: Box::new(subst_variadic_pack_in_expr(index, pack_name, pack)),
            },
            span: sp,
        },
        _ => expr.clone(),
    }
}

fn subst_const_idents_in_expr(expr: &Expr, map: &BTreeMap<String, Expr>) -> Expr {
    let kind = match &expr.kind {
        ExprKind::Ident(s) => {
            if let Some(rep) = map.get(s) {
                return rep.clone();
            }
            expr.kind.clone()
        }
        ExprKind::Binary { op, left, right } => ExprKind::Binary {
            op: *op,
            left: Box::new(subst_const_idents_in_expr(left, map)),
            right: Box::new(subst_const_idents_in_expr(right, map)),
        },
        ExprKind::Unary { op, expr: e } => ExprKind::Unary {
            op: *op,
            expr: Box::new(subst_const_idents_in_expr(e, map)),
        },
        ExprKind::Call { callee, args } => ExprKind::Call {
            callee: Box::new(subst_const_idents_in_expr(callee, map)),
            args: args
                .iter()
                .map(|a| CallArg {
                    name: a.name.clone(),
                    value: subst_const_idents_in_expr(&a.value, map),
                })
                .collect(),
        },
        ExprKind::Field { target, field } => ExprKind::Field {
            target: Box::new(subst_const_idents_in_expr(target, map)),
            field: field.clone(),
        },
        ExprKind::Index { target, index } => ExprKind::Index {
            target: Box::new(subst_const_idents_in_expr(target, map)),
            index: Box::new(subst_const_idents_in_expr(index, map)),
        },
        ExprKind::Cast { expr: e, ty } => ExprKind::Cast {
            expr: Box::new(subst_const_idents_in_expr(e, map)),
            ty: ty.clone(),
        },
        ExprKind::Quantifier {
            is_forall,
            var,
            lo,
            hi,
            body,
        } => {
            let mut inner = map.clone();
            inner.remove(var);
            ExprKind::Quantifier {
                is_forall: *is_forall,
                var: var.clone(),
                lo: Box::new(subst_const_idents_in_expr(lo, map)),
                hi: Box::new(subst_const_idents_in_expr(hi, map)),
                body: Box::new(subst_const_idents_in_expr(body, &inner)),
            }
        }
        other => other.clone(),
    };
    Expr {
        kind,
        span: expr.span,
    }
}
