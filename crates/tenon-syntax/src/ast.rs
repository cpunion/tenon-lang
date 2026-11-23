use crate::span::Span;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub items: Vec<Item>,
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
    TypeAlias(TypeAliasDecl),
    Record(RecordDecl),
    Enum(EnumDecl),
    Fn(FnDecl),
    Evolution(EvolutionDecl),
    LockLattice(LockLatticeDecl),
    Protocol(ProtocolDecl),
    Stream(StreamDecl),
    View(ViewDecl),
    Subscriber(SubscriberDecl),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeAliasDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub is_distinct: bool,
    pub name: String,
    pub type_params: Vec<String>,
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
    pub name: String,
    pub type_params: Vec<String>,
    pub params: Vec<ParamDecl>,
    pub ret_ty: Option<TypeExpr>,
    pub requires: Vec<Expr>,
    pub ensures: Vec<Expr>,
    pub body: Block,
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
    Tuple(Vec<TypeExpr>, Span),
}

impl TypeExpr {
    pub fn display_name(&self) -> String {
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
            TypeExpr::Tuple(elems, _) => {
                let parts: Vec<_> = elems.iter().map(|e| e.display_name()).collect();
                format!("({})", parts.join(", "))
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
pub enum Stmt {
    Let {
        name: String,
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExprKind {
    Int(i64),
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
    Field {
        target: Box<Expr>,
        field: String,
    },
    Index {
        target: Box<Expr>,
        index: Box<Expr>,
    },
    RecordLit {
        name: String,
        version: Option<String>,
        fields: Vec<(String, Expr)>,
    },
    Tuple(Vec<Expr>),
    List(Vec<Expr>),
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
    BitAnd,
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
            BinOp::BitAnd => "&",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnOp {
    Neg,
    Not,
}
