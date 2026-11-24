//! Shared pre-codegen lowering pass for Interfaces, Methods, Static Monomorphization, and `dyn` Vtables.
//!
//! Runs identically before BOTH Cranelift JIT and LLVM 22 AOT so both backends share
//! 100% identical semantics for:
//! - Inherent methods (`impl Type { ... }` -> direct static functions `__m__Type__method`)
//! - Interface implementations (`impl Iface for Type { ... }` + default methods -> `__i__Type__Iface__method`)
//! - Static generic dispatch (`fn f[T Tool](t T)`, `decode[Order](raw)`, `opt.unwrap_or(...)` -> true monomorphization)
//! - Dynamic dispatch (`dyn Iface` -> Perceus 2-field fat object `(data, vtable_ptr)` with `__box_dyn` coercion)

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use tenon_syntax::*;

#[derive(Debug, Clone)]
pub struct VtableSpec {
    pub sym_name: String,
    pub interface: String,
    pub target_type: String,
    pub fn_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DynMethodLayout {
    pub name: String,
    pub slot_idx: usize,
    pub is_inout_self: bool,
    pub param_tys: Vec<TypeExpr>, // excludes `self`
    pub ret_ty: Option<TypeExpr>,
    pub returns_rc: bool,
}

#[derive(Debug, Clone)]
pub struct DynInterfaceLayout {
    pub name: String,
    pub methods: Vec<DynMethodLayout>,
}

#[derive(Debug, Clone)]
pub struct PreparedProgram {
    pub program: Program,
    pub vtables: Vec<VtableSpec>,
    pub dyn_interfaces: BTreeMap<String, DynInterfaceLayout>,
}

fn named_ty(name: &str) -> TypeExpr {
    TypeExpr::Named {
        name: name.to_string(),
        version: None,
        args: Vec::new(),
        span: Span::default(),
    }
}

fn generic_ty(name: &str, args: Vec<TypeExpr>) -> TypeExpr {
    TypeExpr::Named {
        name: name.to_string(),
        version: None,
        args,
        span: Span::default(),
    }
}

fn type_head_name(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Named { name, .. } => name.clone(),
        TypeExpr::Dyn { interface, .. } => format!("dyn {}", interface),
        TypeExpr::Tuple(_, _) => "Tuple".into(),
        TypeExpr::Fn { .. } => "Fn".into(),
        TypeExpr::Array { .. } => "Array".into(),
    }
}

fn mangle_ty(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Named { name, args, .. } => {
            if args.is_empty() {
                name.clone()
            } else {
                let parts: Vec<String> = args.iter().map(mangle_ty).collect();
                format!("{}__{}", name, parts.join("_"))
            }
        }
        TypeExpr::Dyn { interface, .. } => format!("dyn_{}", interface),
        TypeExpr::Tuple(es, _) => {
            let parts: Vec<String> = es.iter().map(mangle_ty).collect();
            format!("tup_{}", parts.join("_"))
        }
        TypeExpr::Fn { params, ret, .. } => {
            let parts: Vec<String> = params.iter().map(mangle_ty).collect();
            let r = ret
                .as_ref()
                .map(|t| mangle_ty(t))
                .unwrap_or_else(|| "Unit".into());
            format!("fn_{}__to_{}", parts.join("_"), r)
        }
        TypeExpr::Array { elem, len, .. } => {
            format!("arr_{}_{}", mangle_ty(elem), len)
        }
    }
}

fn subst_type(ty: &TypeExpr, subst: &BTreeMap<String, TypeExpr>) -> TypeExpr {
    match ty {
        TypeExpr::Named {
            name,
            version,
            args,
            span,
        } => {
            if args.is_empty() {
                if let Some(rep) = subst.get(name) {
                    return rep.clone();
                }
            }
            TypeExpr::Named {
                name: name.clone(),
                version: version.clone(),
                args: args.iter().map(|a| subst_type(a, subst)).collect(),
                span: *span,
            }
        }
        TypeExpr::Dyn {
            interface,
            args,
            span,
        } => TypeExpr::Dyn {
            interface: interface.clone(),
            args: args.iter().map(|a| subst_type(a, subst)).collect(),
            span: *span,
        },
        TypeExpr::Tuple(es, span) => TypeExpr::Tuple(
            es.iter().map(|e| subst_type(e, subst)).collect(),
            *span,
        ),
        TypeExpr::Fn { params, ret, span } => TypeExpr::Fn {
            params: params.iter().map(|p| subst_type(p, subst)).collect(),
            ret: ret.as_ref().map(|r| Box::new(subst_type(r, subst))),
            span: *span,
        },
        TypeExpr::Array { elem, len, span } => TypeExpr::Array {
            elem: Box::new(subst_type(elem, subst)),
            len: *len,
            span: *span,
        },
    }
}

fn subst_expr_types(expr: &Expr, subst: &BTreeMap<String, TypeExpr>) -> Expr {
    let kind = match &expr.kind {
        ExprKind::Field { target, field } => {
            // Rewrite static generic calls `T.from_json(v)` -> `Concrete.from_json(v)`
            let new_target = if let ExprKind::Ident(s) = &target.kind {
                if let Some(rep) = subst.get(s) {
                    Box::new(Expr {
                        kind: ExprKind::Ident(type_head_name(rep)),
                        span: target.span,
                    })
                } else {
                    Box::new(subst_expr_types(target, subst))
                }
            } else {
                Box::new(subst_expr_types(target, subst))
            };
            ExprKind::Field {
                target: new_target,
                field: field.clone(),
            }
        }
        ExprKind::GenericInst { expr: inner, type_args } => ExprKind::GenericInst {
            expr: Box::new(subst_expr_types(inner, subst)),
            type_args: type_args.iter().map(|t| subst_type(t, subst)).collect(),
        },
        ExprKind::Call { callee, args } => ExprKind::Call {
            callee: Box::new(subst_expr_types(callee, subst)),
            args: args
                .iter()
                .map(|a| CallArg {
                    name: a.name.clone(),
                    value: subst_expr_types(&a.value, subst),
                })
                .collect(),
        },
        ExprKind::Binary { op, left, right } => ExprKind::Binary {
            op: *op,
            left: Box::new(subst_expr_types(left, subst)),
            right: Box::new(subst_expr_types(right, subst)),
        },
        ExprKind::Unary { op, expr: inner } => ExprKind::Unary {
            op: *op,
            expr: Box::new(subst_expr_types(inner, subst)),
        },
        ExprKind::Index { target, index } => ExprKind::Index {
            target: Box::new(subst_expr_types(target, subst)),
            index: Box::new(subst_expr_types(index, subst)),
        },
        ExprKind::RecordLit {
            name,
            version,
            fields,
        } => {
            let new_name = subst
                .get(name)
                .map(type_head_name)
                .unwrap_or_else(|| name.clone());
            ExprKind::RecordLit {
                name: new_name,
                version: version.clone(),
                fields: fields
                    .iter()
                    .map(|(k, v)| (k.clone(), subst_expr_types(v, subst)))
                    .collect(),
            }
        }
        ExprKind::Tuple(es) => ExprKind::Tuple(es.iter().map(|e| subst_expr_types(e, subst)).collect()),
        ExprKind::TupleIndex { target, index } => ExprKind::TupleIndex {
            target: Box::new(subst_expr_types(target, subst)),
            index: *index,
        },
        ExprKind::Try(inner) => ExprKind::Try(Box::new(subst_expr_types(inner, subst))),
        ExprKind::Range { start, end } => ExprKind::Range {
            start: Box::new(subst_expr_types(start, subst)),
            end: Box::new(subst_expr_types(end, subst)),
        },
        ExprKind::Lambda {
            params,
            ret_ty,
            body,
        } => ExprKind::Lambda {
            params: params
                .iter()
                .map(|p| LambdaParam {
                    name: p.name.clone(),
                    ty: p.ty.as_ref().map(|t| subst_type(t, subst)),
                    span: p.span,
                })
                .collect(),
            ret_ty: ret_ty.as_ref().map(|t| subst_type(t, subst)),
            body: Box::new(subst_expr_types(body, subst)),
        },
        ExprKind::List(es) => ExprKind::List(es.iter().map(|e| subst_expr_types(e, subst)).collect()),
        ExprKind::If {
            cond,
            then_block,
            else_branch,
        } => ExprKind::If {
            cond: Box::new(subst_expr_types(cond, subst)),
            then_block: subst_block_types(then_block, subst),
            else_branch: else_branch
                .as_ref()
                .map(|e| Box::new(subst_expr_types(e, subst))),
        },
        ExprKind::Match { scrutinee, arms } => ExprKind::Match {
            scrutinee: Box::new(subst_expr_types(scrutinee, subst)),
            arms: arms
                .iter()
                .map(|a| MatchArm {
                    patterns: a.patterns.clone(),
                    guard: a.guard.as_ref().map(|g| subst_expr_types(g, subst)),
                    body: subst_expr_types(&a.body, subst),
                    span: a.span,
                })
                .collect(),
        },
        ExprKind::Block(b) => ExprKind::Block(subst_block_types(b, subst)),
        ExprKind::Unsafe(b) => ExprKind::Unsafe(subst_block_types(b, subst)),
        ExprKind::Comptime(b) => ExprKind::Comptime(subst_block_types(b, subst)),
        ExprKind::ComptimeIf {
            cond,
            then_block,
            else_branch,
        } => ExprKind::ComptimeIf {
            cond: Box::new(subst_expr_types(cond, subst)),
            then_block: subst_block_types(then_block, subst),
            else_branch: else_branch
                .as_ref()
                .map(|e| Box::new(subst_expr_types(e, subst))),
        },
        ExprKind::ComptimeFor {
            binding,
            iter,
            body,
        } => ExprKind::ComptimeFor {
            binding: binding.clone(),
            iter: Box::new(subst_expr_types(iter, subst)),
            body: subst_block_types(body, subst),
        },
        ExprKind::MacroCall { name, args } => ExprKind::MacroCall {
            name: name.clone(),
            args: args.iter().map(|a| subst_expr_types(a, subst)).collect(),
        },
        ExprKind::TypeImplements { ty, interface } => ExprKind::TypeImplements {
            ty: subst_type(ty, subst),
            interface: interface.clone(),
        },
        ExprKind::Ident(s) => {
            if let Some(rep) = subst.get(s) {
                ExprKind::Ident(type_head_name(rep))
            } else {
                ExprKind::Ident(s.clone())
            }
        }
        ExprKind::Cast { expr: inner, ty } => ExprKind::Cast {
            expr: Box::new(subst_expr_types(inner, subst)),
            ty: subst_type(ty, subst),
        },
        ExprKind::TryCast { expr: inner, ty } => ExprKind::TryCast {
            expr: Box::new(subst_expr_types(inner, subst)),
            ty: subst_type(ty, subst),
        },
        ExprKind::ArrayRepeat { value, len } => ExprKind::ArrayRepeat {
            value: Box::new(subst_expr_types(value, subst)),
            len: *len,
        },
        ExprKind::Await(inner) => ExprKind::Await(Box::new(subst_expr_types(inner, subst))),
        ExprKind::Old(inner) => ExprKind::Old(Box::new(subst_expr_types(inner, subst))),
        other => other.clone(),
    };
    Expr {
        kind,
        span: expr.span,
    }
}

fn subst_block_types(blk: &Block, subst: &BTreeMap<String, TypeExpr>) -> Block {
    Block {
        stmts: blk
            .stmts
            .iter()
            .map(|s| match s {
                Stmt::Let {
                    name,
                    ty,
                    value,
                    is_mut,
                    span,
                } => Stmt::Let {
                    name: name.clone(),
                    ty: ty.as_ref().map(|t| subst_type(t, subst)),
                    value: subst_expr_types(value, subst),
                    is_mut: *is_mut,
                    span: *span,
                },
                Stmt::LetTuple {
                    names,
                    ty,
                    value,
                    is_mut,
                    span,
                } => Stmt::LetTuple {
                    names: names.clone(),
                    ty: ty.as_ref().map(|t| subst_type(t, subst)),
                    value: subst_expr_types(value, subst),
                    is_mut: *is_mut,
                    span: *span,
                },
                Stmt::For {
                    binding,
                    iter,
                    body,
                    span,
                } => Stmt::For {
                    binding: binding.clone(),
                    iter: subst_expr_types(iter, subst),
                    body: subst_block_types(body, subst),
                    span: *span,
                },
                Stmt::Assign {
                    target,
                    op,
                    value,
                    span,
                } => Stmt::Assign {
                    target: subst_expr_types(target, subst),
                    op: *op,
                    value: subst_expr_types(value, subst),
                    span: *span,
                },
                Stmt::Return { value, span } => Stmt::Return {
                    value: value.as_ref().map(|e| subst_expr_types(e, subst)),
                    span: *span,
                },
                Stmt::Assert { cond, span } => Stmt::Assert {
                    cond: subst_expr_types(cond, subst),
                    span: *span,
                },
                Stmt::Mutate { target, body, span } => Stmt::Mutate {
                    target: target.clone(),
                    body: subst_block_types(body, subst),
                    span: *span,
                },
                Stmt::Defer { body, span } => Stmt::Defer {
                    body: subst_block_types(body, subst),
                    span: *span,
                },
                Stmt::Ghost { body, span } => Stmt::Ghost {
                    body: subst_block_types(body, subst),
                    span: *span,
                },
                Stmt::While {
                    cond,
                    invariants,
                    decreases,
                    body,
                    span,
                } => Stmt::While {
                    cond: subst_expr_types(cond, subst),
                    invariants: invariants
                        .iter()
                        .map(|e| subst_expr_types(e, subst))
                        .collect(),
                    decreases: decreases.as_ref().map(|e| subst_expr_types(e, subst)),
                    body: subst_block_types(body, subst),
                    span: *span,
                },
                Stmt::Expr { expr, span } => Stmt::Expr {
                    expr: subst_expr_types(expr, subst),
                    span: *span,
                },
                other => other.clone(),
            })
            .collect(),
        tail: blk
            .tail
            .as_ref()
            .map(|t| Box::new(subst_expr_types(t, subst))),
        span: blk.span,
    }
}

fn subst_fn_types(f: &FnDecl, subst: &BTreeMap<String, TypeExpr>) -> FnDecl {
    let mut out = f.clone();
    for p in &mut out.params {
        p.ty = subst_type(&p.ty, subst);
    }
    if let Some(rt) = &out.ret_ty {
        out.ret_ty = Some(subst_type(rt, subst));
    }
    out.body = subst_block_types(&f.body, subst);
    out
}

struct LowerCtx {
    alias_bases: BTreeMap<String, TypeExpr>,
    alias_preds: BTreeMap<String, Expr>,
    ghost_fns: BTreeSet<String>,
    infallible_casts: BTreeSet<usize>,
    orig_program: Program,
    records: BTreeMap<String, RecordDecl>,
    generic_records: BTreeMap<String, RecordDecl>,
    mono_records: Vec<RecordDecl>,
    enums: BTreeMap<String, EnumDecl>,
    generic_enums: BTreeMap<String, EnumDecl>,
    mono_enums: Vec<EnumDecl>,
    enum_names: BTreeSet<String>,
    enum_variants: BTreeMap<String, (String, Vec<TypeExpr>)>,
    unions: BTreeMap<String, UnionDecl>,
    bitfields: BTreeMap<String, BitfieldLayout>,
    extern_fns: BTreeMap<String, ExternFnDecl>,
    consts: BTreeMap<String, (TypeExpr, Expr)>,
    layout: LayoutEngine,
    defer_stack: Vec<Vec<Block>>,
    interfaces: BTreeMap<String, InterfaceDecl>,
    /// Inherent methods: `(TypeName, method_name) -> mangled_fn_name`
    inherent_methods: BTreeMap<(String, String), String>,
    /// Interface impl methods: `(TypeName, method_name) -> Vec<(IfaceName, mangled_fn_name)>`
    impl_methods: BTreeMap<(String, String), Vec<(String, String)>>,
    /// Exact `(IfaceName, TypeName) -> vtable_sym_name`
    vtable_syms: BTreeMap<(String, String), String>,
    /// Generic function templates (not emitted directly; monomorphized on demand)
    generic_templates: BTreeMap<String, FnDecl>,
    /// Concrete function signatures (including flattened methods & monomorphized copies)
    fn_sigs: BTreeMap<String, FnDecl>,
    /// Monomorphized queue & deduplication set
    mono_emitted: BTreeSet<String>,
    mono_queue: VecDeque<FnDecl>,
    /// Lifted top-level closure functions synthesized during lowering
    lifted_closures: Vec<FnDecl>,
    /// Counter for unique synthetic temporary / closure names
    next_tmp_id: usize,
    /// Interface dynamic layouts
    dyn_interfaces: BTreeMap<String, DynInterfaceLayout>,
    /// Comptime & reflection context for evaluating generic `comptime if` / `comptime for` on monomorphization
    comptime_ctx: ComptimeContext,
}

impl LowerCtx {
    fn fresh_id(&mut self) -> usize {
        let id = self.next_tmp_id;
        self.next_tmp_id += 1;
        id
    }

    fn resolve_alias(&self, ty: &TypeExpr) -> TypeExpr {
        let mut cur = ty.clone();
        for _ in 0..8 {
            if let TypeExpr::Named { name, args, .. } = &cur {
                if args.is_empty() {
                    if let Some(base) = self.alias_bases.get(name) {
                        cur = base.clone();
                        continue;
                    }
                }
            }
            break;
        }
        cur
    }

    fn is_float_ty(&self, ty: &TypeExpr) -> bool {
        matches!(
            self.resolve_alias(ty),
            TypeExpr::Named { ref name, .. } if matches!(name.as_str(), "Float" | "Float64" | "Float32")
        )
    }

    fn type_is_rc(&self, ty: &TypeExpr) -> bool {
        let resolved = self.resolve_alias(ty);
        match resolved {
            TypeExpr::Dyn { .. }
            | TypeExpr::Tuple(_, _)
            | TypeExpr::Fn { .. }
            | TypeExpr::Array { .. } => true,
            TypeExpr::Named { name, .. } => {
                matches!(
                    name.as_str(),
                    "Text"
                        | "List"
                        | "Option"
                        | "Result"
                        | "Range"
                        | "Bytes"
                        | "BytesSlice"
                        | "AtomicInt"
                        | "Mutex"
                        | "Channel"
                ) || SimdKind::from_name(&name).is_some()
                    || self.records.contains_key(&name)
                    || self.generic_records.contains_key(&name)
                    || self.unions.contains_key(&name)
                    || self.enum_names.contains(&name)
                    || self.generic_enums.contains_key(&name)
            }
        }
    }

    fn is_known_type(&self, name: &str) -> bool {
        self.records.contains_key(name)
            || self.generic_records.contains_key(name)
            || self.enum_names.contains(name)
            || self.generic_enums.contains_key(name)
            || self.bitfields.contains_key(name)
            || self.unions.contains_key(name)
            || self.alias_bases.contains_key(name)
            || ScalarKind::from_name(name).is_some()
            || SimdKind::from_name(name).is_some()
            || matches!(
                name,
                "Bool"
                    | "Text"
                    | "Bytes"
                    | "BytesSlice"
                    | "Option"
                    | "Result"
                    | "List"
                    | "Range"
                    | "Array"
                    | "Ptr"
                    | "MutPtr"
                    | "CStr"
                    | "CVoid"
                    | "AtomicInt"
                    | "Mutex"
                    | "Channel"
                    | "Simd"
            )
    }

    fn infer_expr_ty(&self, expr: &Expr, var_types: &BTreeMap<String, TypeExpr>) -> TypeExpr {
        self.infer_expr_ty_expected(expr, None, var_types)
    }

    fn infer_expr_ty_expected(
        &self,
        expr: &Expr,
        expected: Option<&TypeExpr>,
        var_types: &BTreeMap<String, TypeExpr>,
    ) -> TypeExpr {
        match &expr.kind {
            ExprKind::Int(_) => named_ty("Int"),
            ExprKind::Float(_) => named_ty("Float"),
            ExprKind::Bool(_) => named_ty("Bool"),
            ExprKind::Str(_) => named_ty("Text"),
            ExprKind::Ident(s) => {
                if s == "None" {
                    return generic_ty("Option", vec![named_ty("Any")]);
                }
                if let Some(t) = var_types.get(s) {
                    return t.clone();
                }
                if let Some((ename, _)) = self.enum_variants.get(s) {
                    if let Some(ge) = self.generic_enums.get(ename) {
                        return generic_ty(ename, vec![named_ty("Any"); ge.type_params.len()]);
                    }
                    return named_ty(ename);
                }
                named_ty("Any")
            }
            ExprKind::RecordLit { name, fields, .. } => {
                if let Some(rec) = self.generic_records.get(name) {
                    let mut subst = BTreeMap::new();
                    for (fname, fval) in fields {
                        if let Some(fd) = rec.fields.iter().find(|f| f.name == *fname) {
                            let aty = self.infer_expr_ty(fval, var_types);
                            self.unify_type_params(&fd.ty, &aty, &rec.type_params, &mut subst);
                        }
                    }
                    let targs: Vec<TypeExpr> = rec
                        .type_params
                        .iter()
                        .map(|tp| subst.get(tp).cloned().unwrap_or_else(|| named_ty("Any")))
                        .collect();
                    return generic_ty(name, targs);
                }
                named_ty(name)
            }
            ExprKind::List(es) => {
                if let Some(first) = es.first() {
                    generic_ty("List", vec![self.infer_expr_ty(first, var_types)])
                } else {
                    generic_ty("List", vec![named_ty("Any")])
                }
            }
            ExprKind::Tuple(es) => TypeExpr::Tuple(
                es.iter().map(|e| self.infer_expr_ty(e, var_types)).collect(),
                expr.span,
            ),
            ExprKind::TupleIndex { target, index } => {
                let t = self.resolve_alias(&self.infer_expr_ty(target, var_types));
                if let TypeExpr::Tuple(es, _) = t {
                    if let Some(et) = es.get(*index) {
                        return et.clone();
                    }
                }
                named_ty("Any")
            }
            ExprKind::Range { .. } => generic_ty("Range", vec![named_ty("Int")]),
            ExprKind::Try(inner) => {
                let t = self.resolve_alias(&self.infer_expr_ty(inner, var_types));
                if let TypeExpr::Named { name, args, .. } = t {
                    if (name == "Result" || name == "Option") && !args.is_empty() {
                        return args[0].clone();
                    }
                }
                named_ty("Any")
            }
            ExprKind::Lambda { params, ret_ty, body } => {
                let (exp_params, exp_ret): (Vec<TypeExpr>, Option<TypeExpr>) =
                    match expected.map(|t| self.resolve_alias(t)) {
                        Some(TypeExpr::Fn { params: ep, ret: er, .. }) => (ep, er.map(|r| *r)),
                        _ => (Vec::new(), None),
                    };
                let mut lam_vars = var_types.clone();
                let mut resolved_params = Vec::with_capacity(params.len());
                for (i, p) in params.iter().enumerate() {
                    let pty = p
                        .ty
                        .clone()
                        .or_else(|| exp_params.get(i).cloned())
                        .unwrap_or_else(|| named_ty("Any"));
                    if p.name != "_" {
                        lam_vars.insert(p.name.clone(), pty.clone());
                    }
                    resolved_params.push(pty);
                }
                let inferred_ret = if let Some(rt) = ret_ty {
                    Some(Box::new(rt.clone()))
                } else if matches!(&body.kind, ExprKind::Block(b) if b.tail.is_none()) {
                    None
                } else {
                    Some(Box::new(self.infer_expr_ty_expected(
                        body,
                        exp_ret.as_ref(),
                        &lam_vars,
                    )))
                };
                TypeExpr::Fn {
                    params: resolved_params,
                    ret: inferred_ret,
                    span: expr.span,
                }
            }
            ExprKind::Binary { op: BinOp::Concat, .. } => named_ty("Text"),
            ExprKind::Binary {
                op: BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::And | BinOp::Or | BinOp::Implies,
                ..
            } => named_ty("Bool"),
            ExprKind::Binary { op, left, right } => {
                let lt = self.infer_expr_ty(left, var_types);
                let rt = self.infer_expr_ty(right, var_types);
                if matches!(left.kind, ExprKind::Int(_) | ExprKind::Float(_))
                    && matches!(self.resolve_alias(&rt), TypeExpr::Named { ref name, .. } if ScalarKind::from_name(name).is_some())
                {
                    rt
                } else if matches!(self.resolve_alias(&lt), TypeExpr::Named { ref name, .. } if ScalarKind::from_name(name).is_some())
                {
                    lt
                } else if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod)
                    && (self.is_float_ty(&lt) || self.is_float_ty(&rt))
                {
                    named_ty("Float")
                } else {
                    named_ty("Int")
                }
            }
            ExprKind::Unary { op: UnOp::Not, .. } => named_ty("Bool"),
            ExprKind::Unary { op: UnOp::Neg | UnOp::BitNot, expr: inner } => {
                let it = self.infer_expr_ty(inner, var_types);
                if matches!(self.resolve_alias(&it), TypeExpr::Named { ref name, .. } if ScalarKind::from_name(name).is_some()) {
                    it
                } else if self.is_float_ty(&it) {
                    named_ty("Float")
                } else {
                    named_ty("Int")
                }
            }
            ExprKind::Cast { ty, .. } => ty.clone(),
            ExprKind::TryCast { ty, .. } => {
                generic_ty("Result", vec![ty.clone(), named_ty("Text")])
            }
            ExprKind::Unsafe(b) => {
                let mut local = var_types.clone();
                self.infer_block_ty(b, &mut local)
            }
            ExprKind::ArrayRepeat { value, len } => {
                let et = self.infer_expr_ty(value, var_types);
                TypeExpr::Array {
                    elem: Box::new(et),
                    len: *len,
                    span: expr.span,
                }
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    if let Some((en, _)) = self.enum_variants.get(&format!("{}.{}", ename, field)) {
                        if let Some(ge) = self.generic_enums.get(en) {
                            return generic_ty(en, vec![named_ty("Any"); ge.type_params.len()]);
                        }
                        return named_ty(en);
                    }
                }
                let t = self.resolve_alias(&self.infer_expr_ty(target, var_types));
                if let TypeExpr::Named { name: rname, args: rargs, .. } = t {
                    if let Some(rec) = self
                        .records
                        .get(&rname)
                        .or_else(|| self.generic_records.get(&rname))
                    {
                        if let Some(fl) = rec.fields.iter().find(|f| f.name == *field) {
                            if !rec.type_params.is_empty() && !rargs.is_empty() {
                                let mut fsubst = BTreeMap::new();
                                for (tp, ra) in rec.type_params.iter().zip(rargs.iter()) {
                                    fsubst.insert(tp.clone(), ra.clone());
                                }
                                return subst_type(&fl.ty, &fsubst);
                            }
                            return fl.ty.clone();
                        }
                    }
                    if let Some(bf) = self.bitfields.get(&rname) {
                        if bf.fields.iter().any(|f| f.name == *field) {
                            return named_ty(bf.backing.canonical_name());
                        }
                    }
                    if let Some(u) = self.unions.get(&rname) {
                        if let Some(fl) = u.fields.iter().find(|f| f.name == *field) {
                            return fl.ty.clone();
                        }
                    }
                }
                named_ty("Any")
            }
            ExprKind::Index { target, .. } => {
                let t = self.resolve_alias(&self.infer_expr_ty(target, var_types));
                match t {
                    TypeExpr::Array { elem, .. } => *elem,
                    TypeExpr::Named { name, args, .. } if name == "List" && !args.is_empty() => {
                        args[0].clone()
                    }
                    _ => named_ty("Any"),
                }
            }
            ExprKind::Call { callee, args } => match &callee.kind {
                ExprKind::Ident(fname) => {
                    if fname == "__make_closure" {
                        if let Some(a0) = args.first() {
                            if let ExprKind::Str(csym) = &a0.value.kind {
                                if let Some(sig) = self.fn_sigs.get(csym) {
                                    let ps: Vec<TypeExpr> =
                                        sig.params.iter().skip(1).map(|p| p.ty.clone()).collect();
                                    return TypeExpr::Fn {
                                        params: ps,
                                        ret: sig.ret_ty.clone().map(Box::new),
                                        span: expr.span,
                                    };
                                }
                            }
                        }
                    }
                    if fname == "__call_closure" {
                        if let Some(a0) = args.first() {
                            if let TypeExpr::Fn { ret, .. } =
                                self.resolve_alias(&self.infer_expr_ty(&a0.value, var_types))
                            {
                                return ret.map(|r| *r).unwrap_or_else(|| named_ty("Any"));
                            }
                        }
                    }
                    if fname == "__box_dyn" {
                        if let Some(a1) = args.get(1) {
                            if let ExprKind::Str(vsym) = &a1.value.kind {
                                for ((iface, _), sym) in &self.vtable_syms {
                                    if sym == vsym {
                                        return TypeExpr::Dyn {
                                            interface: iface.clone(),
                                            args: vec![],
                                            span: expr.span,
                                        };
                                    }
                                }
                            }
                        }
                    }
                    if let Some(vty) = var_types.get(fname) {
                        if let TypeExpr::Fn { ret, .. } = self.resolve_alias(vty) {
                            return ret.map(|r| *r).unwrap_or_else(|| named_ty("Any"));
                        }
                    }
                    if fname == "Ok" && args.len() == 1 {
                        return generic_ty(
                            "Result",
                            vec![self.infer_expr_ty(&args[0].value, var_types), named_ty("Any")],
                        );
                    }
                    if fname == "Err" && args.len() == 1 {
                        return generic_ty(
                            "Result",
                            vec![named_ty("Any"), self.infer_expr_ty(&args[0].value, var_types)],
                        );
                    }
                    if fname == "Some" && args.len() == 1 {
                        return generic_ty("Option", vec![self.infer_expr_ty(&args[0].value, var_types)]);
                    }
                    if fname == "push" && args.len() == 2 {
                        let lt = self.infer_expr_ty(&args[0].value, var_types);
                        if matches!(&lt, TypeExpr::Named { name, args: la, .. } if name == "List" && la.first().is_some_and(|x| x.display_name() != "Any")) {
                            return lt;
                        }
                        return generic_ty("List", vec![self.infer_expr_ty(&args[1].value, var_types)]);
                    }
                    if fname == "list_filled" && args.len() == 2 {
                        return generic_ty("List", vec![self.infer_expr_ty(&args[1].value, var_types)]);
                    }
                    if fname == "list_pop_shrink" && args.len() == 1 {
                        return self.infer_expr_ty(&args[0].value, var_types);
                    }
                    if let Some(sig) = self.fn_sigs.get(fname) {
                        return sig.ret_ty.clone().unwrap_or_else(|| named_ty("Any"));
                    }
                    if let Some(tmpl) = self.generic_templates.get(fname) {
                        if let Some(rt) = &tmpl.ret_ty {
                            let subst = self.infer_generic_subst(tmpl, &[], args, var_types);
                            return subst_type(rt, &subst);
                        }
                    }
                    if let Some((ename, ptys)) = self.enum_variants.get(fname) {
                        return self.infer_enum_ctor_ty(ename, ptys, args, var_types);
                    }
                    if let Some(ef) = self.extern_fns.get(fname) {
                        return ef.ret_ty.clone().unwrap_or_else(|| named_ty("Any"));
                    }
                    if (fname == "ptr_of" || fname == "mut_ptr_of") && args.len() == 1 {
                        let raw_inner = self.infer_expr_ty(&args[0].value, var_types);
                        let inner = match self.resolve_alias(&raw_inner) {
                            TypeExpr::Array { elem, .. } => *elem,
                            _ => raw_inner,
                        };
                        let pname = if fname == "mut_ptr_of" { "MutPtr" } else { "Ptr" };
                        return generic_ty(pname, vec![inner]);
                    }
                    if (fname == "ptr_add" || fname == "__ptr_add") && !args.is_empty() {
                        return self.infer_expr_ty(&args[0].value, var_types);
                    }
                    if (fname == "ptr_read" || fname == "volatile_read") && args.len() == 1 {
                        if let TypeExpr::Named { name, args: pa, .. } =
                            self.resolve_alias(&self.infer_expr_ty(&args[0].value, var_types))
                        {
                            if (name == "Ptr" || name == "MutPtr") && !pa.is_empty() {
                                return pa[0].clone();
                            }
                        }
                        return named_ty("Any");
                    }
                    if fname == "mutex_new" || fname == "__mutex_new" {
                        return generic_ty(
                            "Mutex",
                            vec![args
                                .first()
                                .map(|a| self.infer_expr_ty(&a.value, var_types))
                                .unwrap_or_else(|| named_ty("Any"))],
                        );
                    }
                    if fname == "mutex_get" && args.len() == 1 {
                        if let TypeExpr::Named { name, args: ma, .. } =
                            self.resolve_alias(&self.infer_expr_ty(&args[0].value, var_types))
                        {
                            if name == "Mutex" && !ma.is_empty() {
                                return ma[0].clone();
                            }
                        }
                        return named_ty("Any");
                    }
                    if (fname == "chan_recv" || fname == "chan_try_recv") && args.len() == 1 {
                        if let TypeExpr::Named { name, args: ca, .. } =
                            self.resolve_alias(&self.infer_expr_ty(&args[0].value, var_types))
                        {
                            if name == "Channel" && !ca.is_empty() {
                                return generic_ty("Option", vec![ca[0].clone()]);
                            }
                        }
                        return generic_ty("Option", vec![named_ty("Any")]);
                    }
                    if matches!(fname.as_str(), "int_to_float" | "parse_float" | "simd_f64x4_lane" | "simd_f64x4_reduce_sum" | "simd_f64x4_dot") {
                        return named_ty("Float");
                    }
                    if matches!(fname.as_str(), "parse_f32" | "simd_f32x4_lane" | "simd_f32x4_reduce_sum" | "simd_f32x4_dot") {
                        return named_ty("Float32");
                    }
                    if matches!(fname.as_str(), "bytes_get" | "slice_get" | "simd_u8x16_lane") {
                        return named_ty("UInt8");
                    }
                    if matches!(fname.as_str(), "bswap16" | "bytes_read_u16_le" | "bytes_read_u16_be") {
                        return named_ty("UInt16");
                    }
                    if matches!(fname.as_str(), "bswap32" | "bytes_read_u32_le" | "bytes_read_u32_be") {
                        return named_ty("UInt32");
                    }
                    if matches!(fname.as_str(), "bswap64" | "rotl64" | "rotr64" | "bytes_read_u64_le" | "bytes_read_u64_be") {
                        return named_ty("UInt64");
                    }
                    if matches!(fname.as_str(), "simd_i32x4_lane" | "simd_i32x4_reduce_sum") {
                        return named_ty("Int32");
                    }
                    if matches!(
                        fname.as_str(),
                        "bytes_new"
                            | "bytes_filled"
                            | "bytes_set"
                            | "bytes_push"
                            | "bytes_write_u16_le"
                            | "bytes_write_u16_be"
                            | "bytes_write_u32_le"
                            | "bytes_write_u32_be"
                            | "bytes_write_u64_le"
                            | "bytes_write_u64_be"
                            | "text_to_bytes"
                            | "net_read"
                    ) {
                        return named_ty("Bytes");
                    }
                    if fname == "bytes_slice" {
                        return named_ty("BytesSlice");
                    }
                    if fname == "atomic_new" {
                        return named_ty("AtomicInt");
                    }
                    if matches!(fname.as_str(), "bytes_ptr" | "text_cstr_ptr") {
                        return generic_ty("Ptr", vec![named_ty("UInt8")]);
                    }
                    if matches!(fname.as_str(), "raw_alloc" | "raw_memcpy" | "raw_memset") {
                        return generic_ty("MutPtr", vec![named_ty("UInt8")]);
                    }
                    if fname.starts_with("simd_f32x4_") {
                        return named_ty("Float32x4");
                    }
                    if fname.starts_with("simd_f64x4_") {
                        return named_ty("Float64x4");
                    }
                    if fname.starts_with("simd_i32x4_") {
                        return named_ty("Int32x4");
                    }
                    if fname.starts_with("simd_i64x4_") {
                        return named_ty("Int64x4");
                    }
                    if fname.starts_with("simd_u8x16_") {
                        return named_ty("UInt8x16");
                    }
                    if matches!(
                        fname.as_str(),
                        "str_slice"
                            | "byte_to_text"
                            | "str_concat"
                            | "buf_finish"
                            | "float_to_text"
                            | "f32_to_text"
                            | "bytes_to_text"
                            | "cstr_to_text"
                            | "sys_read_line"
                            | "sys_env"
                            | "sys_cwd"
                            | "sys_read_file"
                            | "sys_exec"
                            | "sys_http_sse_next_line"
                            | "sys_http_sse_error"
                    ) {
                        return named_ty("Text");
                    }
                    if matches!(
                        fname.as_str(),
                        "float_is_nan"
                            | "ptr_is_null"
                            | "atomic_cas"
                            | "chan_send"
                            | "net_set_nonblocking"
                            | "sys_is_eof"
                            | "sys_file_ok"
                            | "sys_write_file"
                            | "sys_http_sse_eof"
                    ) {
                        return named_ty("Bool");
                    }
                    named_ty("Int")
                }
                ExprKind::GenericInst { expr: inner, type_args } => {
                    if let ExprKind::Ident(fname) = &inner.kind {
                        if fname == "refine_cast" && type_args.len() == 1 && args.len() == 1 {
                            let dst_ty = &type_args[0];
                            let src_ty = self.infer_expr_ty(&args[0].value, var_types);
                            if self.is_refinement_cast_infallible(&src_ty, dst_ty, expr.span) {
                                return dst_ty.clone();
                            }
                            return generic_ty("Result", vec![dst_ty.clone(), named_ty("Text")]);
                        }
                        if let Some(tmpl) = self.generic_templates.get(fname) {
                            if let Some(rt) = &tmpl.ret_ty {
                                let subst = self.infer_generic_subst(tmpl, type_args, args, var_types);
                                return subst_type(rt, &subst);
                            }
                        }
                    } else if let ExprKind::Field { target, field } = &inner.kind {
                        if let ExprKind::Ident(tname) = &target.kind {
                            if !var_types.contains_key(tname) && self.is_known_type(tname) {
                                if let Some(mname) = self.lookup_concrete_method(tname, field) {
                                    if let Some(tmpl) = self.generic_templates.get(&mname) {
                                        if let Some(rt) = &tmpl.ret_ty {
                                            let subst = self.infer_generic_subst(tmpl, type_args, args, var_types);
                                            return subst_type(rt, &subst);
                                        }
                                    }
                                    if let Some(sig) = self.fn_sigs.get(&mname) {
                                        return sig.ret_ty.clone().unwrap_or_else(|| named_ty("Any"));
                                    }
                                }
                            }
                        }
                        let rty = self.infer_expr_ty(target, var_types);
                        let rbase = self.resolve_alias(&rty);
                        if let TypeExpr::Named { name, args: rargs, .. } = &rbase {
                            let lookup_name = if let TypeExpr::Named { name: orig, .. } = &rty {
                                if self.lookup_concrete_method(orig, field).is_some() {
                                    orig.clone()
                                } else {
                                    name.clone()
                                }
                            } else {
                                name.clone()
                            };
                            if let Some(mname) = self.lookup_concrete_method(&lookup_name, field) {
                                if let Some(tmpl) = self.generic_templates.get(&mname) {
                                    if let Some(rt) = &tmpl.ret_ty {
                                        let mut explicit_targs = rargs.clone();
                                        explicit_targs.extend(type_args.iter().cloned());
                                        let mut full_args = Vec::with_capacity(args.len() + 1);
                                        full_args.push(CallArg {
                                            name: None,
                                            value: (**target).clone(),
                                        });
                                        full_args.extend(args.iter().cloned());
                                        let subst = self.infer_generic_subst(
                                            tmpl,
                                            &explicit_targs,
                                            &full_args,
                                            var_types,
                                        );
                                        return subst_type(rt, &subst);
                                    }
                                }
                                if let Some(sig) = self.fn_sigs.get(&mname) {
                                    return sig.ret_ty.clone().unwrap_or_else(|| named_ty("Any"));
                                }
                            }
                        }
                    }
                    named_ty("Any")
                }
                ExprKind::Field { target, field } => {
                    if let ExprKind::GenericInst { expr: inner, type_args } = &target.kind {
                        if let ExprKind::Ident(tname) = &inner.kind {
                            if !var_types.contains_key(tname) && self.is_known_type(tname) {
                                if let Some(mname) = self.lookup_concrete_method(tname, field) {
                                    if let Some(tmpl) = self.generic_templates.get(&mname) {
                                        if let Some(rt) = &tmpl.ret_ty {
                                            let subst = self.infer_generic_subst(tmpl, type_args, args, var_types);
                                            return subst_type(rt, &subst);
                                        }
                                    }
                                    if let Some(sig) = self.fn_sigs.get(&mname) {
                                        return sig.ret_ty.clone().unwrap_or_else(|| named_ty("Any"));
                                    }
                                }
                            }
                        }
                    }
                    if field == "parse" {
                        if let ExprKind::Ident(rname) = &target.kind {
                            if self.records.contains_key(rname)
                                && self.lookup_concrete_method(rname, "parse").is_none()
                            {
                                return generic_ty("Result", vec![named_ty(rname), named_ty("Int")]);
                            }
                        }
                    }
                    if let ExprKind::Ident(ename) = &target.kind {
                        if let Some((en, ptys)) = self.enum_variants.get(&format!("{}.{}", ename, field)) {
                            return self.infer_enum_ctor_ty(en, ptys, args, var_types);
                        }
                        if !var_types.contains_key(ename) && self.is_known_type(ename) {
                            if let Some(mname) = self.lookup_concrete_method(ename, field) {
                                if let Some(sig) = self.fn_sigs.get(&mname) {
                                    return sig.ret_ty.clone().unwrap_or_else(|| named_ty("Any"));
                                }
                                if let Some(tmpl) = self.generic_templates.get(&mname) {
                                    if let Some(rt) = &tmpl.ret_ty {
                                        let subst = self.infer_generic_subst(tmpl, &[], args, var_types);
                                        return subst_type(rt, &subst);
                                    }
                                }
                            }
                        }
                    }
                    let rty = self.infer_expr_ty(target, var_types);
                    let rbase = self.resolve_alias(&rty);
                    match &rbase {
                        TypeExpr::Dyn { interface, .. } => {
                            if let Some(layout) = self.dyn_interfaces.get(interface) {
                                if let Some(dm) = layout.methods.iter().find(|m| m.name == *field) {
                                    return dm.ret_ty.clone().unwrap_or_else(|| named_ty("Any"));
                                }
                            }
                        }
                        TypeExpr::Named { name, args: rargs, .. } => {
                            if let Some(mname) = self.lookup_concrete_method(name, field) {
                                if let Some(sig) = self.fn_sigs.get(&mname) {
                                    return sig.ret_ty.clone().unwrap_or_else(|| named_ty("Any"));
                                }
                                if let Some(tmpl) = self.generic_templates.get(&mname) {
                                    if let Some(rt) = &tmpl.ret_ty {
                                        let mut full_args = Vec::with_capacity(args.len() + 1);
                                        full_args.push(CallArg {
                                            name: None,
                                            value: (**target).clone(),
                                        });
                                        full_args.extend(args.iter().cloned());
                                        let subst = self.infer_generic_subst(tmpl, rargs, &full_args, var_types);
                                        return subst_type(rt, &subst);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                    if let TypeExpr::Fn { ret, .. } =
                        self.resolve_alias(&self.infer_expr_ty(callee, var_types))
                    {
                        return ret.map(|r| *r).unwrap_or_else(|| named_ty("Any"));
                    }
                    named_ty("Any")
                }
                _ => {
                    if let TypeExpr::Fn { ret, .. } =
                        self.resolve_alias(&self.infer_expr_ty(callee, var_types))
                    {
                        ret.map(|r| *r).unwrap_or_else(|| named_ty("Any"))
                    } else {
                        named_ty("Any")
                    }
                }
            },
            ExprKind::If { then_block, else_branch, .. } => {
                let mut local = var_types.clone();
                let t = self.infer_block_ty(then_block, &mut local);
                if t.display_name() != "Any" {
                    return t;
                }
                if let Some(eb) = else_branch {
                    return self.infer_expr_ty_expected(eb, expected, var_types);
                }
                named_ty("Any")
            }
            ExprKind::Match { scrutinee, arms } => {
                let st = self.infer_expr_ty(scrutinee, var_types);
                for arm in arms {
                    let mut av = var_types.clone();
                    self.bind_arm_vars(&st, arm, &mut av);
                    let t = self.infer_expr_ty_expected(&arm.body, expected, &av);
                    if t.display_name() != "Any" {
                        return t;
                    }
                }
                named_ty("Any")
            }
            ExprKind::Block(b) => {
                let mut local = var_types.clone();
                self.infer_block_ty(b, &mut local)
            }
            ExprKind::Await(inner) => self.infer_expr_ty(inner, var_types),
            _ => named_ty("Any"),
        }
    }

    fn infer_enum_ctor_ty(
        &self,
        ename: &str,
        ptys: &[TypeExpr],
        args: &[CallArg],
        var_types: &BTreeMap<String, TypeExpr>,
    ) -> TypeExpr {
        if let Some(ge) = self.generic_enums.get(ename) {
            let mut subst = BTreeMap::new();
            for (pty, a) in ptys.iter().zip(args.iter()) {
                let aty = self.infer_expr_ty(&a.value, var_types);
                self.unify_type_params(pty, &aty, &ge.type_params, &mut subst);
            }
            let targs: Vec<TypeExpr> = ge
                .type_params
                .iter()
                .map(|tp| subst.get(tp).cloned().unwrap_or_else(|| named_ty("Any")))
                .collect();
            return generic_ty(ename, targs);
        }
        named_ty(ename)
    }

    fn infer_block_ty(&self, blk: &Block, var_types: &mut BTreeMap<String, TypeExpr>) -> TypeExpr {
        for s in &blk.stmts {
            match s {
                Stmt::Let { name, ty, value, .. } => {
                    let t = ty
                        .clone()
                        .unwrap_or_else(|| self.infer_expr_ty(value, var_types));
                    var_types.insert(name.clone(), t);
                }
                Stmt::LetTuple { names, ty, value, .. } => {
                    let t = ty
                        .clone()
                        .unwrap_or_else(|| self.infer_expr_ty(value, var_types));
                    if let TypeExpr::Tuple(es, _) = self.resolve_alias(&t) {
                        for (i, n) in names.iter().enumerate() {
                            if n != "_" {
                                let et = es.get(i).cloned().unwrap_or_else(|| named_ty("Any"));
                                var_types.insert(n.clone(), et);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        blk.tail
            .as_deref()
            .map(|t| self.infer_expr_ty(t, var_types))
            .unwrap_or_else(|| named_ty("Any"))
    }

    fn resolve_variant_payload_tys(&self, scrutinee_base: &TypeExpr, variant_name: &str) -> Vec<TypeExpr> {
        let short = variant_name.rsplit('.').next().unwrap_or(variant_name);
        if short == "Some" {
            if let TypeExpr::Named { name: tn, args, .. } = scrutinee_base {
                if tn == "Option" && !args.is_empty() {
                    return vec![args[0].clone()];
                }
            }
            return vec![named_ty("Any")];
        }
        if short == "Ok" {
            if let TypeExpr::Named { name: tn, args, .. } = scrutinee_base {
                if tn == "Result" && !args.is_empty() {
                    return vec![args[0].clone()];
                }
            }
            return vec![named_ty("Any")];
        }
        if short == "Err" {
            if let TypeExpr::Named { name: tn, args, .. } = scrutinee_base {
                if tn == "Result" && args.len() >= 2 {
                    return vec![args[1].clone()];
                }
            }
            return vec![named_ty("Any")];
        }
        if let Some((ename, ptys)) = self
            .enum_variants
            .get(variant_name)
            .or_else(|| self.enum_variants.get(short))
        {
            if let Some(ge) = self.generic_enums.get(ename) {
                if let TypeExpr::Named { args, .. } = scrutinee_base {
                    if !args.is_empty() {
                        let mut subst = BTreeMap::new();
                        for (tp, ta) in ge.type_params.iter().zip(args.iter()) {
                            subst.insert(tp.clone(), ta.clone());
                        }
                        return ptys.iter().map(|t| subst_type(t, &subst)).collect();
                    }
                }
            }
            return ptys.clone();
        }
        Vec::new()
    }

    fn bind_pattern_vars(&self, scrutinee_ty: &TypeExpr, pat: &Pattern, vars: &mut BTreeMap<String, TypeExpr>) {
        let base = self.resolve_alias(scrutinee_ty);
        match pat {
            Pattern::Ident(name, _) if name != "_" => {
                vars.insert(name.clone(), scrutinee_ty.clone());
            }
            Pattern::Tuple(pats, _) => {
                if let TypeExpr::Tuple(elems, _) = &base {
                    for (i, sp) in pats.iter().enumerate() {
                        let ety = elems.get(i).cloned().unwrap_or_else(|| named_ty("Any"));
                        self.bind_pattern_vars(&ety, sp, vars);
                    }
                } else {
                    for sp in pats {
                        self.bind_pattern_vars(&named_ty("Any"), sp, vars);
                    }
                }
            }
            Pattern::Variant { name, binding: Some(bn), .. } if bn != "_" => {
                let ptys = self.resolve_variant_payload_tys(&base, name);
                let bty = ptys.into_iter().next().unwrap_or_else(|| named_ty("Any"));
                vars.insert(bn.clone(), bty);
            }
            Pattern::VariantMulti { name, patterns, .. } => {
                let ptys = self.resolve_variant_payload_tys(&base, name);
                for (i, sp) in patterns.iter().enumerate() {
                    let pty = ptys.get(i).cloned().unwrap_or_else(|| named_ty("Any"));
                    self.bind_pattern_vars(&pty, sp, vars);
                }
            }
            _ => {}
        }
    }

    fn bind_arm_vars(&self, scrutinee_ty: &TypeExpr, arm: &MatchArm, vars: &mut BTreeMap<String, TypeExpr>) {
        for p in &arm.patterns {
            self.bind_pattern_vars(scrutinee_ty, p, vars);
        }
    }

    fn lookup_concrete_method(&self, ty_name: &str, method: &str) -> Option<String> {
        if let Some(m) = self.inherent_methods.get(&(ty_name.to_string(), method.to_string())) {
            return Some(m.clone());
        }
        if let Some(list) = self.impl_methods.get(&(ty_name.to_string(), method.to_string())) {
            if let Some((_, m)) = list.first() {
                return Some(m.clone());
            }
        }
        None
    }

    fn infer_generic_subst(
        &self,
        tmpl: &FnDecl,
        explicit_type_args: &[TypeExpr],
        args: &[CallArg],
        var_types: &BTreeMap<String, TypeExpr>,
    ) -> BTreeMap<String, TypeExpr> {
        let mut subst = BTreeMap::new();
        for (tp, ta) in tmpl.type_params.iter().zip(explicit_type_args.iter()) {
            if ta.display_name() != "Any" {
                subst.insert(tp.clone(), ta.clone());
            }
        }
        // Pass 1: non-lambda arguments
        for (p, a) in tmpl.params.iter().zip(args.iter()) {
            if !matches!(a.value.kind, ExprKind::Lambda { .. }) {
                let aty = self.infer_expr_ty(&a.value, var_types);
                self.unify_type_params(&p.ty, &aty, &tmpl.type_params, &mut subst);
            }
        }
        // Pass 2: lambda arguments (using expected fn type with Pass-1 substitutions!)
        for (p, a) in tmpl.params.iter().zip(args.iter()) {
            if matches!(a.value.kind, ExprKind::Lambda { .. }) {
                let expected_pty = subst_type(&p.ty, &subst);
                let aty = self.infer_expr_ty_expected(&a.value, Some(&expected_pty), var_types);
                self.unify_type_params(&p.ty, &aty, &tmpl.type_params, &mut subst);
            }
        }
        subst
    }

    fn unify_type_params(
        &self,
        pat: &TypeExpr,
        actual: &TypeExpr,
        type_params: &[String],
        subst: &mut BTreeMap<String, TypeExpr>,
    ) {
        match (pat, actual) {
            (TypeExpr::Named { name, args, .. }, _) if args.is_empty() && type_params.contains(name) => {
                if actual.display_name() != "Any" {
                    subst.entry(name.clone()).or_insert_with(|| actual.clone());
                }
            }
            (
                TypeExpr::Named { name: pn, args: pa, .. },
                TypeExpr::Named { name: an, args: aa, .. },
            ) if pn == an && pa.len() == aa.len() => {
                for (x, y) in pa.iter().zip(aa.iter()) {
                    self.unify_type_params(x, y, type_params, subst);
                }
            }
            (TypeExpr::Tuple(pe, _), TypeExpr::Tuple(ae, _)) if pe.len() == ae.len() => {
                for (x, y) in pe.iter().zip(ae.iter()) {
                    self.unify_type_params(x, y, type_params, subst);
                }
            }
            (
                TypeExpr::Fn { params: pp, ret: pr, .. },
                TypeExpr::Fn { params: ap, ret: ar, .. },
            ) if pp.len() == ap.len() => {
                for (x, y) in pp.iter().zip(ap.iter()) {
                    self.unify_type_params(x, y, type_params, subst);
                }
                if let (Some(pr_ty), Some(ar_ty)) = (pr, ar) {
                    self.unify_type_params(pr_ty, ar_ty, type_params, subst);
                }
            }
            _ => {}
        }
    }

    fn ensure_monomorphized(
        &mut self,
        tmpl_name: &str,
        subst: &BTreeMap<String, TypeExpr>,
    ) -> Option<String> {
        let tmpl = self.generic_templates.get(tmpl_name)?.clone();
        let suffix_parts: Vec<String> = tmpl
            .type_params
            .iter()
            .map(|tp| {
                subst
                    .get(tp)
                    .map(mangle_ty)
                    .unwrap_or_else(|| "Int".into())
            })
            .collect();
        let mono_name = format!("{}__mono__{}", tmpl_name, suffix_parts.join("__"));
        if self.mono_emitted.insert(mono_name.clone()) {
            let mut spec = subst_fn_types(&tmpl, subst);
            spec.name = mono_name.clone();
            spec.type_params.clear();
            spec.type_param_bounds.clear();
            let _ = eval_comptime_in_fn(&self.comptime_ctx, &mut spec);
            self.fn_sigs.insert(mono_name.clone(), spec.clone());
            self.mono_queue.push_back(spec);
        }
        Some(mono_name)
    }

    fn maybe_box_dyn(&self, expr: Expr, expected: &TypeExpr, actual: &TypeExpr) -> Expr {
        let TypeExpr::Dyn { interface, .. } = expected else {
            return expr;
        };
        if matches!(actual, TypeExpr::Dyn { .. }) {
            return expr;
        }
        let concrete_name = type_head_name(&self.resolve_alias(actual));
        let orig_name = type_head_name(actual);
        let vsym = self
            .vtable_syms
            .get(&(interface.clone(), orig_name.clone()))
            .or_else(|| self.vtable_syms.get(&(interface.clone(), concrete_name.clone())));
        let Some(vsym) = vsym else {
            return expr;
        };
        let is_rc = if self.type_is_rc(actual) { 1i64 } else { 0i64 };
        let sp = expr.span;
        Expr {
            kind: ExprKind::Call {
                callee: Box::new(Expr {
                    kind: ExprKind::Ident("__box_dyn".into()),
                    span: sp,
                }),
                args: vec![
                    CallArg { name: None, value: expr },
                    CallArg {
                        name: None,
                        value: Expr {
                            kind: ExprKind::Str(vsym.clone()),
                            span: sp,
                        },
                    },
                    CallArg {
                        name: None,
                        value: Expr {
                            kind: ExprKind::Int(is_rc),
                            span: sp,
                        },
                    },
                ],
            },
            span: sp,
        }
    }

    fn lower_fn(&mut self, f: &FnDecl) -> FnDecl {
        let mut var_types = BTreeMap::new();
        for p in &f.params {
            var_types.insert(p.name.clone(), p.ty.clone());
        }
        let mut out = f.clone();
        out.body = self.lower_block(&f.body, &mut var_types, f.ret_ty.as_ref());
        out
    }

    fn lower_block(
        &mut self,
        blk: &Block,
        var_types: &mut BTreeMap<String, TypeExpr>,
        ret_ty: Option<&TypeExpr>,
    ) -> Block {
        let saved = var_types.clone();
        self.defer_stack.push(Vec::new());
        let mut stmts = Vec::with_capacity(blk.stmts.len());
        for s in &blk.stmts {
            match s {
                Stmt::Defer { body, .. } => {
                    let mut dv = var_types.clone();
                    let lowered_defer = self.lower_block(body, &mut dv, None);
                    if let Some(top) = self.defer_stack.last_mut() {
                        top.push(lowered_defer);
                    }
                }
                Stmt::Ghost { .. } => {
                    // Ghost proof blocks are statically verified and erased at codegen (zero runtime overhead).
                }
                Stmt::Let {
                    name,
                    ty,
                    value,
                    is_mut,
                    span,
                } => {
                    let pre_ty = match &value.kind {
                        ExprKind::Range { .. } => {
                            Some(generic_ty("Range", vec![named_ty("Int")]))
                        }
                        ExprKind::RecordLit { name: rname, .. }
                            if self.bitfields.contains_key(rname) =>
                        {
                            Some(named_ty(rname))
                        }
                        ExprKind::Call { callee, .. } => match &callee.kind {
                            ExprKind::Field { target, field }
                                if field == "from_raw" =>
                            {
                                match &target.kind {
                                    ExprKind::Ident(bname)
                                        if self.bitfields.contains_key(bname) =>
                                    {
                                        Some(named_ty(bname))
                                    }
                                    _ => None,
                                }
                            }
                            _ => None,
                        },
                        _ => None,
                    };
                    let is_bf_pre = matches!(
                        &pre_ty,
                        Some(TypeExpr::Named { name: bn, .. }) if self.bitfields.contains_key(bn)
                    );
                    let mut lv = self.lower_expr(value, var_types, ret_ty, ty.as_ref());
                    let inferred = pre_ty.unwrap_or_else(|| self.infer_expr_ty(&lv, var_types));
                    let eff_ty = if let Some(t) = ty {
                        lv = self.maybe_box_dyn(lv, t, &inferred);
                        t.clone()
                    } else {
                        inferred
                    };
                    let emitted_ty = if ty.is_none() && is_bf_pre {
                        Some(eff_ty.clone())
                    } else {
                        ty.clone()
                    };
                    var_types.insert(name.clone(), eff_ty);
                    stmts.push(Stmt::Let {
                        name: name.clone(),
                        ty: emitted_ty,
                        value: lv,
                        is_mut: *is_mut,
                        span: *span,
                    });
                }
                Stmt::Assign {
                    target,
                    op,
                    value,
                    span,
                } => {
                    // Bitfield field assignment: `bf.flag = v` -> `bf = (bf & ~(mask << off)) | (((v as U) & mask) << off)`
                    if *op == AssignOp::Set {
                        if let ExprKind::Field { target: bf_tgt, field } = &target.kind {
                            let bty = self.resolve_alias(&self.infer_expr_ty(bf_tgt, var_types));
                            if let TypeExpr::Named { name: bname, .. } = &bty {
                                if let Some(bl) = self.bitfields.get(bname).cloned() {
                                    if let Some(fl) = bl.fields.iter().find(|f| f.name == *field) {
                                        let lbf = self.lower_expr(bf_tgt, var_types, ret_ty, None);
                                        let bty_expr = named_ty(bl.backing.canonical_name());
                                        let lv = self.lower_expr(value, var_types, ret_ty, Some(&bty_expr));
                                        let clear_mask = !(fl.mask << fl.bit_offset) as i64;
                                        let cleared = Expr {
                                            kind: ExprKind::Binary {
                                                op: BinOp::BitAnd,
                                                left: Box::new(lbf.clone()),
                                                right: Box::new(Expr {
                                                    kind: ExprKind::Cast {
                                                        expr: Box::new(Expr {
                                                            kind: ExprKind::Int(clear_mask),
                                                            span: *span,
                                                        }),
                                                        ty: bty_expr.clone(),
                                                    },
                                                    span: *span,
                                                }),
                                            },
                                            span: *span,
                                        };
                                        let masked_v = Expr {
                                            kind: ExprKind::Binary {
                                                op: BinOp::BitAnd,
                                                left: Box::new(Expr {
                                                    kind: ExprKind::Cast {
                                                        expr: Box::new(lv),
                                                        ty: bty_expr.clone(),
                                                    },
                                                    span: *span,
                                                }),
                                                right: Box::new(Expr {
                                                    kind: ExprKind::Cast {
                                                        expr: Box::new(Expr {
                                                            kind: ExprKind::Int(fl.mask as i64),
                                                            span: *span,
                                                        }),
                                                        ty: bty_expr.clone(),
                                                    },
                                                    span: *span,
                                                }),
                                            },
                                            span: *span,
                                        };
                                        let shifted_v = if fl.bit_offset == 0 {
                                            masked_v
                                        } else {
                                            Expr {
                                                kind: ExprKind::Binary {
                                                    op: BinOp::Shl,
                                                    left: Box::new(masked_v),
                                                    right: Box::new(Expr {
                                                        kind: ExprKind::Int(fl.bit_offset as i64),
                                                        span: *span,
                                                    }),
                                                },
                                                span: *span,
                                            }
                                        };
                                        let updated = Expr {
                                            kind: ExprKind::Binary {
                                                op: BinOp::BitOr,
                                                left: Box::new(cleared),
                                                right: Box::new(shifted_v),
                                            },
                                            span: *span,
                                        };
                                        stmts.push(Stmt::Assign {
                                            target: lbf,
                                            op: AssignOp::Set,
                                            value: updated,
                                            span: *span,
                                        });
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                    let lt = self.lower_expr(target, var_types, ret_ty, None);
                    let target_exp = self.infer_expr_ty(&lt, var_types);
                    let exp_opt = if target_exp.display_name() != "Any" {
                        Some(&target_exp)
                    } else {
                        None
                    };
                    let mut lv = self.lower_expr(value, var_types, ret_ty, exp_opt);
                    match &lt.kind {
                        ExprKind::Ident(vname) => {
                            if let Some(tty) = var_types.get(vname).cloned() {
                                let aty = self.infer_expr_ty(&lv, var_types);
                                lv = self.maybe_box_dyn(lv, &tty, &aty);
                            }
                        }
                        ExprKind::Index { target: list_expr, .. } => {
                            let lty = self.resolve_alias(&self.infer_expr_ty(list_expr, var_types));
                            if let TypeExpr::Named { name: ln, args: la, .. } = lty {
                                if ln == "List" && la.len() == 1 {
                                    let aty = self.infer_expr_ty(&lv, var_types);
                                    lv = self.maybe_box_dyn(lv, &la[0], &aty);
                                }
                            }
                        }
                        _ => {}
                    }
                    stmts.push(Stmt::Assign {
                        target: lt,
                        op: *op,
                        value: lv,
                        span: *span,
                    });
                }
                Stmt::Return { value, span } => {
                    let lv = value.as_ref().map(|e| {
                        let le = self.lower_expr(e, var_types, ret_ty, ret_ty);
                        if let Some(rt) = ret_ty {
                            let aty = self.infer_expr_ty(&le, var_types);
                            self.maybe_box_dyn(le, rt, &aty)
                        } else {
                            le
                        }
                    });
                    let has_defers = self.defer_stack.iter().any(|scope| !scope.is_empty());
                    if has_defers {
                        let final_val = if let Some(val_expr) = lv {
                            let tid = self.fresh_id();
                            let tname = format!("__defer_ret_{}", tid);
                            let vty = ret_ty
                                .cloned()
                                .unwrap_or_else(|| self.infer_expr_ty(&val_expr, var_types));
                            stmts.push(Stmt::Let {
                                name: tname.clone(),
                                ty: Some(vty),
                                value: val_expr,
                                is_mut: false,
                                span: *span,
                            });
                            Some(Expr {
                                kind: ExprKind::Ident(tname),
                                span: *span,
                            })
                        } else {
                            None
                        };
                        for scope in self.defer_stack.iter().rev() {
                            for db in scope.iter().rev() {
                                stmts.push(Stmt::Expr {
                                    expr: Expr {
                                        kind: ExprKind::Block(db.clone()),
                                        span: db.span,
                                    },
                                    span: db.span,
                                });
                            }
                        }
                        stmts.push(Stmt::Return {
                            value: final_val,
                            span: *span,
                        });
                    } else {
                        stmts.push(Stmt::Return {
                            value: lv,
                            span: *span,
                        });
                    }
                }
                Stmt::Assert { cond, span } => stmts.push(Stmt::Assert {
                    cond: self.lower_expr(cond, var_types, ret_ty, None),
                    span: *span,
                }),
                Stmt::Mutate { target, body, span } => stmts.push(Stmt::Mutate {
                    target: target.clone(),
                    body: self.lower_block(body, var_types, ret_ty),
                    span: *span,
                }),
                Stmt::While {
                    cond,
                    invariants,
                    decreases,
                    body,
                    span,
                } => stmts.push(Stmt::While {
                    cond: self.lower_expr(cond, var_types, ret_ty, None),
                    invariants: invariants.clone(),
                    decreases: decreases.clone(),
                    body: self.lower_block(body, var_types, ret_ty),
                    span: *span,
                }),
                Stmt::LetTuple {
                    names,
                    ty,
                    value,
                    is_mut,
                    span,
                } => {
                    let lv = self.lower_expr(value, var_types, ret_ty, ty.as_ref());
                    let inferred = self.infer_expr_ty(&lv, var_types);
                    let eff_ty = ty.clone().unwrap_or(inferred);
                    let tmp_id = self.fresh_id();
                    let tmp_name = format!("__tup_{}", tmp_id);
                    var_types.insert(tmp_name.clone(), eff_ty.clone());
                    stmts.push(Stmt::Let {
                        name: tmp_name.clone(),
                        ty: Some(eff_ty.clone()),
                        value: lv,
                        is_mut: false,
                        span: *span,
                    });
                    let elems = match self.resolve_alias(&eff_ty) {
                        TypeExpr::Tuple(es, _) => es,
                        _ => Vec::new(),
                    };
                    for (idx, n) in names.iter().enumerate() {
                        if n == "_" {
                            continue;
                        }
                        let ety = elems.get(idx).cloned().unwrap_or_else(|| named_ty("Any"));
                        var_types.insert(n.clone(), ety.clone());
                        stmts.push(Stmt::Let {
                            name: n.clone(),
                            ty: Some(ety),
                            value: Expr {
                                kind: ExprKind::TupleIndex {
                                    target: Box::new(Expr {
                                        kind: ExprKind::Ident(tmp_name.clone()),
                                        span: *span,
                                    }),
                                    index: idx,
                                },
                                span: *span,
                            },
                            is_mut: *is_mut,
                            span: *span,
                        });
                    }
                }
                Stmt::For {
                    binding,
                    iter,
                    body,
                    span,
                } => {
                    let id = self.fresh_id();
                    let sp = *span;
                    let idx_name = format!("__for_i_{}", id);
                    let end_name = format!("__for_end_{}", id);
                    let mut body_vars = var_types.clone();

                    // Case 1: Range literal `for i in start..end { body }`
                    if let ExprKind::Range { start, end } = &iter.kind {
                        let ls = self.lower_expr(start, var_types, ret_ty, None);
                        let le = self.lower_expr(end, var_types, ret_ty, None);
                        stmts.push(Stmt::Let {
                            name: end_name.clone(),
                            ty: Some(named_ty("Int")),
                            value: le,
                            is_mut: false,
                            span: sp,
                        });
                        stmts.push(Stmt::Let {
                            name: idx_name.clone(),
                            ty: Some(named_ty("Int")),
                            value: ls,
                            is_mut: true,
                            span: sp,
                        });
                        let mut loop_stmts = Vec::new();
                        if let ForBinding::Ident(v) = binding {
                            if v != "_" {
                                body_vars.insert(v.clone(), named_ty("Int"));
                                loop_stmts.push(Stmt::Let {
                                    name: v.clone(),
                                    ty: Some(named_ty("Int")),
                                    value: Expr {
                                        kind: ExprKind::Ident(idx_name.clone()),
                                        span: sp,
                                    },
                                    is_mut: false,
                                    span: sp,
                                });
                            }
                        }
                        loop_stmts.push(Stmt::Assign {
                            target: Expr {
                                kind: ExprKind::Ident(idx_name.clone()),
                                span: sp,
                            },
                            op: AssignOp::AddSet,
                            value: Expr {
                                kind: ExprKind::Int(1),
                                span: sp,
                            },
                            span: sp,
                        });
                        let lb = self.lower_block(body, &mut body_vars, ret_ty);
                        loop_stmts.extend(lb.stmts);
                        if let Some(t) = lb.tail {
                            loop_stmts.push(Stmt::Expr { expr: *t, span: sp });
                        }
                        stmts.push(Stmt::While {
                            cond: Expr {
                                kind: ExprKind::Binary {
                                    op: BinOp::Lt,
                                    left: Box::new(Expr {
                                        kind: ExprKind::Ident(idx_name),
                                        span: sp,
                                    }),
                                    right: Box::new(Expr {
                                        kind: ExprKind::Ident(end_name),
                                        span: sp,
                                    }),
                                },
                                span: sp,
                            },
                            invariants: vec![],
                            decreases: None,
                            body: Block {
                                stmts: loop_stmts,
                                tail: None,
                                span: body.span,
                            },
                            span: sp,
                        });
                    } else {
                        let li = self.lower_expr(iter, var_types, ret_ty, None);
                        let iter_ty = self.resolve_alias(&self.infer_expr_ty(&li, var_types));
                        match &iter_ty {
                            // Case 2: Runtime `Range[Int]` value (represented as 2-tuple `(start, end)`)
                            TypeExpr::Named { name, .. } if name == "Range" => {
                                let src_name = format!("__for_rng_{}", id);
                                stmts.push(Stmt::Let {
                                    name: src_name.clone(),
                                    ty: Some(TypeExpr::Tuple(vec![named_ty("Int"), named_ty("Int")], sp)),
                                    value: li,
                                    is_mut: false,
                                    span: sp,
                                });
                                stmts.push(Stmt::Let {
                                    name: end_name.clone(),
                                    ty: Some(named_ty("Int")),
                                    value: Expr {
                                        kind: ExprKind::TupleIndex {
                                            target: Box::new(Expr {
                                                kind: ExprKind::Ident(src_name.clone()),
                                                span: sp,
                                            }),
                                            index: 1,
                                        },
                                        span: sp,
                                    },
                                    is_mut: false,
                                    span: sp,
                                });
                                stmts.push(Stmt::Let {
                                    name: idx_name.clone(),
                                    ty: Some(named_ty("Int")),
                                    value: Expr {
                                        kind: ExprKind::TupleIndex {
                                            target: Box::new(Expr {
                                                kind: ExprKind::Ident(src_name),
                                                span: sp,
                                            }),
                                            index: 0,
                                        },
                                        span: sp,
                                    },
                                    is_mut: true,
                                    span: sp,
                                });
                                let mut loop_stmts = Vec::new();
                                if let ForBinding::Ident(v) = binding {
                                    if v != "_" {
                                        body_vars.insert(v.clone(), named_ty("Int"));
                                        loop_stmts.push(Stmt::Let {
                                            name: v.clone(),
                                            ty: Some(named_ty("Int")),
                                            value: Expr {
                                                kind: ExprKind::Ident(idx_name.clone()),
                                                span: sp,
                                            },
                                            is_mut: false,
                                            span: sp,
                                        });
                                    }
                                }
                                loop_stmts.push(Stmt::Assign {
                                    target: Expr {
                                        kind: ExprKind::Ident(idx_name.clone()),
                                        span: sp,
                                    },
                                    op: AssignOp::AddSet,
                                    value: Expr {
                                        kind: ExprKind::Int(1),
                                        span: sp,
                                    },
                                    span: sp,
                                });
                                let lb = self.lower_block(body, &mut body_vars, ret_ty);
                                loop_stmts.extend(lb.stmts);
                                if let Some(t) = lb.tail {
                                    loop_stmts.push(Stmt::Expr { expr: *t, span: sp });
                                }
                                stmts.push(Stmt::While {
                                    cond: Expr {
                                        kind: ExprKind::Binary {
                                            op: BinOp::Lt,
                                            left: Box::new(Expr {
                                                kind: ExprKind::Ident(idx_name),
                                                span: sp,
                                            }),
                                            right: Box::new(Expr {
                                                kind: ExprKind::Ident(end_name),
                                                span: sp,
                                            }),
                                        },
                                        span: sp,
                                    },
                                    invariants: vec![],
                                    decreases: None,
                                    body: Block {
                                        stmts: loop_stmts,
                                        tail: None,
                                        span: body.span,
                                    },
                                    span: sp,
                                });
                            }
                            // Case 3: `Map[K, V]` (dense SoA `keys` & `vals` lists!)
                            TypeExpr::Named { name, args, .. } if name == "Map" => {
                                let k_ty = args.first().cloned().unwrap_or_else(|| named_ty("Any"));
                                let v_ty = args.get(1).cloned().unwrap_or_else(|| named_ty("Any"));
                                let map_name = format!("__for_map_{}", id);
                                let keys_name = format!("__for_keys_{}", id);
                                let vals_name = format!("__for_vals_{}", id);
                                stmts.push(Stmt::Let {
                                    name: map_name.clone(),
                                    ty: Some(iter_ty.clone()),
                                    value: li,
                                    is_mut: false,
                                    span: sp,
                                });
                                stmts.push(Stmt::Let {
                                    name: keys_name.clone(),
                                    ty: Some(generic_ty("List", vec![k_ty.clone()])),
                                    value: Expr {
                                        kind: ExprKind::Field {
                                            target: Box::new(Expr {
                                                kind: ExprKind::Ident(map_name.clone()),
                                                span: sp,
                                            }),
                                            field: "keys".into(),
                                        },
                                        span: sp,
                                    },
                                    is_mut: false,
                                    span: sp,
                                });
                                let need_vals = matches!(binding, ForBinding::Tuple(ns) if ns.get(1).is_some_and(|v| v != "_"));
                                if need_vals {
                                    stmts.push(Stmt::Let {
                                        name: vals_name.clone(),
                                        ty: Some(generic_ty("List", vec![v_ty.clone()])),
                                        value: Expr {
                                            kind: ExprKind::Field {
                                                target: Box::new(Expr {
                                                    kind: ExprKind::Ident(map_name),
                                                    span: sp,
                                                }),
                                                field: "vals".into(),
                                            },
                                            span: sp,
                                        },
                                        is_mut: false,
                                        span: sp,
                                    });
                                }
                                stmts.push(Stmt::Let {
                                    name: end_name.clone(),
                                    ty: Some(named_ty("Int")),
                                    value: Expr {
                                        kind: ExprKind::Call {
                                            callee: Box::new(Expr {
                                                kind: ExprKind::Ident("len".into()),
                                                span: sp,
                                            }),
                                            args: vec![CallArg {
                                                name: None,
                                                value: Expr {
                                                    kind: ExprKind::Ident(keys_name.clone()),
                                                    span: sp,
                                                },
                                            }],
                                        },
                                        span: sp,
                                    },
                                    is_mut: false,
                                    span: sp,
                                });
                                stmts.push(Stmt::Let {
                                    name: idx_name.clone(),
                                    ty: Some(named_ty("Int")),
                                    value: Expr {
                                        kind: ExprKind::Int(0),
                                        span: sp,
                                    },
                                    is_mut: true,
                                    span: sp,
                                });
                                let mut loop_stmts = Vec::new();
                                match binding {
                                    ForBinding::Ident(k) => {
                                        if k != "_" {
                                            body_vars.insert(k.clone(), k_ty.clone());
                                            loop_stmts.push(Stmt::Let {
                                                name: k.clone(),
                                                ty: Some(k_ty),
                                                value: Expr {
                                                    kind: ExprKind::Index {
                                                        target: Box::new(Expr {
                                                            kind: ExprKind::Ident(keys_name),
                                                            span: sp,
                                                        }),
                                                        index: Box::new(Expr {
                                                            kind: ExprKind::Ident(idx_name.clone()),
                                                            span: sp,
                                                        }),
                                                    },
                                                    span: sp,
                                                },
                                                is_mut: false,
                                                span: sp,
                                            });
                                        }
                                    }
                                    ForBinding::Tuple(ns) => {
                                        if let Some(k) = ns.first() {
                                            if k != "_" {
                                                body_vars.insert(k.clone(), k_ty.clone());
                                                loop_stmts.push(Stmt::Let {
                                                    name: k.clone(),
                                                    ty: Some(k_ty),
                                                    value: Expr {
                                                        kind: ExprKind::Index {
                                                            target: Box::new(Expr {
                                                                kind: ExprKind::Ident(keys_name),
                                                                span: sp,
                                                            }),
                                                            index: Box::new(Expr {
                                                                kind: ExprKind::Ident(idx_name.clone()),
                                                                span: sp,
                                                            }),
                                                        },
                                                        span: sp,
                                                    },
                                                    is_mut: false,
                                                    span: sp,
                                                });
                                            }
                                        }
                                        if let Some(v) = ns.get(1) {
                                            if v != "_" {
                                                body_vars.insert(v.clone(), v_ty.clone());
                                                loop_stmts.push(Stmt::Let {
                                                    name: v.clone(),
                                                    ty: Some(v_ty),
                                                    value: Expr {
                                                        kind: ExprKind::Index {
                                                            target: Box::new(Expr {
                                                                kind: ExprKind::Ident(vals_name),
                                                                span: sp,
                                                            }),
                                                            index: Box::new(Expr {
                                                                kind: ExprKind::Ident(idx_name.clone()),
                                                                span: sp,
                                                            }),
                                                        },
                                                        span: sp,
                                                    },
                                                    is_mut: false,
                                                    span: sp,
                                                });
                                            }
                                        }
                                    }
                                }
                                loop_stmts.push(Stmt::Assign {
                                    target: Expr {
                                        kind: ExprKind::Ident(idx_name.clone()),
                                        span: sp,
                                    },
                                    op: AssignOp::AddSet,
                                    value: Expr {
                                        kind: ExprKind::Int(1),
                                        span: sp,
                                    },
                                    span: sp,
                                });
                                let lb = self.lower_block(body, &mut body_vars, ret_ty);
                                loop_stmts.extend(lb.stmts);
                                if let Some(t) = lb.tail {
                                    loop_stmts.push(Stmt::Expr { expr: *t, span: sp });
                                }
                                stmts.push(Stmt::While {
                                    cond: Expr {
                                        kind: ExprKind::Binary {
                                            op: BinOp::Lt,
                                            left: Box::new(Expr {
                                                kind: ExprKind::Ident(idx_name),
                                                span: sp,
                                            }),
                                            right: Box::new(Expr {
                                                kind: ExprKind::Ident(end_name),
                                                span: sp,
                                            }),
                                        },
                                        span: sp,
                                    },
                                    invariants: vec![],
                                    decreases: None,
                                    body: Block {
                                        stmts: loop_stmts,
                                        tail: None,
                                        span: body.span,
                                    },
                                    span: sp,
                                });
                            }
                            // Case 4: `Array[T, N]` or `List[T]` (default iterable)
                            _ => {
                                let (elem_ty, fixed_len) = match &iter_ty {
                                    TypeExpr::Array { elem, len, .. } => ((**elem).clone(), Some(*len)),
                                    TypeExpr::Named { name, args, .. } if name == "List" => (
                                        args.first().cloned().unwrap_or_else(|| named_ty("Any")),
                                        None,
                                    ),
                                    _ => (named_ty("Any"), None),
                                };
                                let src_name = format!("__for_src_{}", id);
                                stmts.push(Stmt::Let {
                                    name: src_name.clone(),
                                    ty: Some(iter_ty.clone()),
                                    value: li,
                                    is_mut: false,
                                    span: sp,
                                });
                                let end_expr = if let Some(n) = fixed_len {
                                    Expr {
                                        kind: ExprKind::Int(n as i64),
                                        span: sp,
                                    }
                                } else {
                                    Expr {
                                        kind: ExprKind::Call {
                                            callee: Box::new(Expr {
                                                kind: ExprKind::Ident("len".into()),
                                                span: sp,
                                            }),
                                            args: vec![CallArg {
                                                name: None,
                                                value: Expr {
                                                    kind: ExprKind::Ident(src_name.clone()),
                                                    span: sp,
                                                },
                                            }],
                                        },
                                        span: sp,
                                    }
                                };
                                stmts.push(Stmt::Let {
                                    name: end_name.clone(),
                                    ty: Some(named_ty("Int")),
                                    value: end_expr,
                                    is_mut: false,
                                    span: sp,
                                });
                                stmts.push(Stmt::Let {
                                    name: idx_name.clone(),
                                    ty: Some(named_ty("Int")),
                                    value: Expr {
                                        kind: ExprKind::Int(0),
                                        span: sp,
                                    },
                                    is_mut: true,
                                    span: sp,
                                });
                                let mut loop_stmts = Vec::new();
                                let idx_expr = Expr {
                                    kind: ExprKind::Index {
                                        target: Box::new(Expr {
                                            kind: ExprKind::Ident(src_name),
                                            span: sp,
                                        }),
                                        index: Box::new(Expr {
                                            kind: ExprKind::Ident(idx_name.clone()),
                                            span: sp,
                                        }),
                                    },
                                    span: sp,
                                };
                                match binding {
                                    ForBinding::Ident(v) => {
                                        if v != "_" {
                                            body_vars.insert(v.clone(), elem_ty.clone());
                                            loop_stmts.push(Stmt::Let {
                                                name: v.clone(),
                                                ty: Some(elem_ty),
                                                value: idx_expr,
                                                is_mut: false,
                                                span: sp,
                                            });
                                        }
                                    }
                                    ForBinding::Tuple(ns) => {
                                        let tup_name = format!("__for_tup_{}", id);
                                        loop_stmts.push(Stmt::Let {
                                            name: tup_name.clone(),
                                            ty: Some(elem_ty.clone()),
                                            value: idx_expr,
                                            is_mut: false,
                                            span: sp,
                                        });
                                        let tup_elems = match self.resolve_alias(&elem_ty) {
                                            TypeExpr::Tuple(es, _) => es,
                                            _ => Vec::new(),
                                        };
                                        for (ti, tn) in ns.iter().enumerate() {
                                            if tn != "_" {
                                                let tty = tup_elems
                                                    .get(ti)
                                                    .cloned()
                                                    .unwrap_or_else(|| named_ty("Any"));
                                                body_vars.insert(tn.clone(), tty.clone());
                                                loop_stmts.push(Stmt::Let {
                                                    name: tn.clone(),
                                                    ty: Some(tty),
                                                    value: Expr {
                                                        kind: ExprKind::TupleIndex {
                                                            target: Box::new(Expr {
                                                                kind: ExprKind::Ident(tup_name.clone()),
                                                                span: sp,
                                                            }),
                                                            index: ti,
                                                        },
                                                        span: sp,
                                                    },
                                                    is_mut: false,
                                                    span: sp,
                                                });
                                            }
                                        }
                                    }
                                }
                                loop_stmts.push(Stmt::Assign {
                                    target: Expr {
                                        kind: ExprKind::Ident(idx_name.clone()),
                                        span: sp,
                                    },
                                    op: AssignOp::AddSet,
                                    value: Expr {
                                        kind: ExprKind::Int(1),
                                        span: sp,
                                    },
                                    span: sp,
                                });
                                let lb = self.lower_block(body, &mut body_vars, ret_ty);
                                loop_stmts.extend(lb.stmts);
                                if let Some(t) = lb.tail {
                                    loop_stmts.push(Stmt::Expr { expr: *t, span: sp });
                                }
                                stmts.push(Stmt::While {
                                    cond: Expr {
                                        kind: ExprKind::Binary {
                                            op: BinOp::Lt,
                                            left: Box::new(Expr {
                                                kind: ExprKind::Ident(idx_name),
                                                span: sp,
                                            }),
                                            right: Box::new(Expr {
                                                kind: ExprKind::Ident(end_name),
                                                span: sp,
                                            }),
                                        },
                                        span: sp,
                                    },
                                    invariants: vec![],
                                    decreases: None,
                                    body: Block {
                                        stmts: loop_stmts,
                                        tail: None,
                                        span: body.span,
                                    },
                                    span: sp,
                                });
                            }
                        }
                    }
                }
                Stmt::Break { span } => {
                    if let Some(top) = self.defer_stack.last() {
                        for db in top.iter().rev() {
                            stmts.push(Stmt::Expr {
                                expr: Expr {
                                    kind: ExprKind::Block(db.clone()),
                                    span: db.span,
                                },
                                span: db.span,
                            });
                        }
                    }
                    stmts.push(Stmt::Break { span: *span });
                }
                Stmt::Continue { span } => {
                    if let Some(top) = self.defer_stack.last() {
                        for db in top.iter().rev() {
                            stmts.push(Stmt::Expr {
                                expr: Expr {
                                    kind: ExprKind::Block(db.clone()),
                                    span: db.span,
                                },
                                span: db.span,
                            });
                        }
                    }
                    stmts.push(Stmt::Continue { span: *span });
                }
                Stmt::Expr { expr, span } => {
                    if self.is_ghost_call_expr(expr) {
                        continue;
                    }
                    stmts.push(Stmt::Expr {
                        expr: self.lower_expr(expr, var_types, ret_ty, None),
                        span: *span,
                    });
                }
            }
        }
        let scope_defers = self.defer_stack.pop().unwrap_or_default();
        let tail = if scope_defers.is_empty() {
            blk.tail.as_ref().map(|t| {
                let lt = self.lower_expr(t, var_types, ret_ty, ret_ty);
                let boxed = if let Some(rt) = ret_ty {
                    let aty = self.infer_expr_ty(&lt, var_types);
                    self.maybe_box_dyn(lt, rt, &aty)
                } else {
                    lt
                };
                Box::new(boxed)
            })
        } else if let Some(t) = blk.tail.as_ref() {
            let lt = self.lower_expr(t, var_types, ret_ty, ret_ty);
            let boxed = if let Some(rt) = ret_ty {
                let aty = self.infer_expr_ty(&lt, var_types);
                self.maybe_box_dyn(lt, rt, &aty)
            } else {
                lt
            };
            let tid = self.fresh_id();
            let tname = format!("__defer_tail_{}", tid);
            let tty = ret_ty
                .cloned()
                .unwrap_or_else(|| self.infer_expr_ty(&boxed, var_types));
            stmts.push(Stmt::Let {
                name: tname.clone(),
                ty: Some(tty),
                value: boxed,
                is_mut: false,
                span: t.span,
            });
            for db in scope_defers.iter().rev() {
                stmts.push(Stmt::Expr {
                    expr: Expr {
                        kind: ExprKind::Block(db.clone()),
                        span: db.span,
                    },
                    span: db.span,
                });
            }
            Some(Box::new(Expr {
                kind: ExprKind::Ident(tname),
                span: t.span,
            }))
        } else {
            for db in scope_defers.iter().rev() {
                stmts.push(Stmt::Expr {
                    expr: Expr {
                        kind: ExprKind::Block(db.clone()),
                        span: db.span,
                    },
                    span: db.span,
                });
            }
            None
        };
        *var_types = saved;
        Block {
            stmts,
            tail,
            span: blk.span,
        }
    }

    fn lower_lambda(
        &mut self,
        params: &[LambdaParam],
        ret_ty_ann: Option<&TypeExpr>,
        body: &Expr,
        var_types: &BTreeMap<String, TypeExpr>,
        expected_ty: Option<&TypeExpr>,
        sp: Span,
    ) -> Expr {
        let (exp_params, exp_ret): (Vec<TypeExpr>, Option<TypeExpr>) =
            match expected_ty.map(|t| self.resolve_alias(t)) {
                Some(TypeExpr::Fn { params: ep, ret: er, .. }) => (ep, er.map(|r| *r)),
                _ => (Vec::new(), None),
            };

        // 1. Resolve concrete parameter types for the lambda
        let mut resolved_params: Vec<ParamDecl> = Vec::with_capacity(params.len());
        let mut bound = BTreeSet::new();
        for (i, p) in params.iter().enumerate() {
            let pty = p
                .ty
                .clone()
                .or_else(|| exp_params.get(i).cloned())
                .unwrap_or_else(|| named_ty("Int"));
            if p.name != "_" {
                bound.insert(p.name.clone());
            }
            resolved_params.push(ParamDecl {
                name: p.name.clone(),
                ty: pty,
                is_inout: false,
                is_comptime: false,
                predicate: None,
                span: p.span,
            });
        }

        // 2. Collect free variables from body that exist in enclosing `var_types`
        let mut free_names = BTreeSet::new();
        collect_free_vars_expr(body, &bound, &mut free_names);
        let captures: Vec<(String, TypeExpr)> = free_names
            .into_iter()
            .filter_map(|n| var_types.get(&n).map(|t| (n, t.clone())))
            .collect();

        // 3. Build lambda's inner `var_types` = captures + resolved params
        let mut lam_vars: BTreeMap<String, TypeExpr> = BTreeMap::new();
        for (cn, ct) in &captures {
            lam_vars.insert(cn.clone(), ct.clone());
        }
        for rp in &resolved_params {
            if rp.name != "_" {
                lam_vars.insert(rp.name.clone(), rp.ty.clone());
            }
        }

        let target_ret = ret_ty_ann.cloned().or(exp_ret);

        // 4. Lower lambda body in its own environment
        let mut fn_stmts: Vec<Stmt> = Vec::new();
        let mut rc_mask: i64 = 0;
        for (k, (cn, ct)) in captures.iter().enumerate() {
            if self.type_is_rc(ct) && k < 63 {
                rc_mask |= 1i64 << (k + 1);
            }
            fn_stmts.push(Stmt::Let {
                name: cn.clone(),
                ty: Some(ct.clone()),
                value: Expr {
                    kind: ExprKind::Call {
                        callee: Box::new(Expr {
                            kind: ExprKind::Ident("__closure_get".into()),
                            span: sp,
                        }),
                        args: vec![
                            CallArg {
                                name: None,
                                value: Expr {
                                    kind: ExprKind::Ident("__env".into()),
                                    span: sp,
                                },
                            },
                            CallArg {
                                name: None,
                                value: Expr {
                                    kind: ExprKind::Int((k + 1) as i64),
                                    span: sp,
                                },
                            },
                        ],
                    },
                    span: sp,
                },
                is_mut: false,
                span: sp,
            });
        }

        let (lowered_body_tail, inferred_ret) = if let ExprKind::Block(blk) = &body.kind {
            let mut bvars = lam_vars.clone();
            let lb = self.lower_block(blk, &mut bvars, target_ret.as_ref());
            let mut ivars = lam_vars.clone();
            let ret_t = if let Some(rt) = ret_ty_ann {
                Some(rt.clone())
            } else if lb.tail.is_none() {
                None
            } else {
                let it = self.infer_block_ty(&lb, &mut ivars);
                if it.display_name() == "Any" {
                    target_ret.clone()
                } else {
                    Some(it)
                }
            };
            fn_stmts.extend(lb.stmts);
            (lb.tail, ret_t)
        } else {
            let lb = self.lower_expr(body, &lam_vars, target_ret.as_ref(), target_ret.as_ref());
            let ret_t = if let Some(rt) = ret_ty_ann {
                Some(rt.clone())
            } else {
                let it = self.infer_expr_ty(&lb, &lam_vars);
                if it.display_name() == "Any" {
                    target_ret.clone()
                } else {
                    Some(it)
                }
            };
            (Some(Box::new(lb)), ret_t)
        };

        // 5. Synthesize top-level `__closure__N(__env Int, params...) -> ret_ty`
        let cid = self.fresh_id();
        let cname = format!("__closure__{}", cid);
        let mut full_params = Vec::with_capacity(resolved_params.len() + 1);
        full_params.push(ParamDecl {
            name: "__env".into(),
            ty: named_ty("Int"),
            is_inout: false,
            is_comptime: false,
            predicate: None,
            span: sp,
        });
        full_params.extend(resolved_params);

        let closure_fn = FnDecl {
            attrs: vec![],
            is_pub: false,
            is_pure: false,
            is_async: false,
            is_test: false,
            is_unsafe: false,
            is_comptime: false,
            is_ghost: false,
            name: cname.clone(),
            type_params: vec![],
            type_param_bounds: BTreeMap::new(),
            params: full_params,
            ret_ty: inferred_ret,
            requires: vec![],
            ensures: vec![],
            body: Block {
                stmts: fn_stmts,
                tail: lowered_body_tail,
                span: sp,
            },
            span: sp,
        };
        self.fn_sigs.insert(cname.clone(), closure_fn.clone());
        self.lifted_closures.push(closure_fn);

        // 6. Emit `__make_closure("__closure__N", rc_mask, cap_0, cap_1, ...)`
        let mut make_args = Vec::with_capacity(2 + captures.len());
        make_args.push(CallArg {
            name: None,
            value: Expr {
                kind: ExprKind::Str(cname),
                span: sp,
            },
        });
        make_args.push(CallArg {
            name: None,
            value: Expr {
                kind: ExprKind::Int(rc_mask),
                span: sp,
            },
        });
        for (cn, _) in captures {
            make_args.push(CallArg {
                name: None,
                value: Expr {
                    kind: ExprKind::Ident(cn),
                    span: sp,
                },
            });
        }
        Expr {
            kind: ExprKind::Call {
                callee: Box::new(Expr {
                    kind: ExprKind::Ident("__make_closure".into()),
                    span: sp,
                }),
                args: make_args,
            },
            span: sp,
        }
    }

    fn lower_args_two_pass(
        &mut self,
        args: &[CallArg],
        param_tys: &[TypeExpr],
        type_params: &[String],
        subst: &mut BTreeMap<String, TypeExpr>,
        var_types: &BTreeMap<String, TypeExpr>,
        ret_ty: Option<&TypeExpr>,
    ) -> Vec<CallArg> {
        let mut out: Vec<Option<CallArg>> = vec![None; args.len()];
        // Pass 1: lower non-lambda arguments and unify generic type parameters
        for (i, a) in args.iter().enumerate() {
            if !matches!(a.value.kind, ExprKind::Lambda { .. }) {
                let exp = param_tys.get(i).map(|pt| subst_type(pt, subst));
                let lv = self.lower_expr(&a.value, var_types, ret_ty, exp.as_ref());
                if !type_params.is_empty() {
                    if let Some(pt) = param_tys.get(i) {
                        let aty = self.infer_expr_ty(&lv, var_types);
                        self.unify_type_params(pt, &aty, type_params, subst);
                    }
                }
                out[i] = Some(CallArg {
                    name: a.name.clone(),
                    value: lv,
                });
            }
        }
        // Pass 2: lower lambda arguments with substituted expected `TypeExpr::Fn` and unify return type params (e.g. `U` in `map[U]`)!
        for (i, a) in args.iter().enumerate() {
            if matches!(a.value.kind, ExprKind::Lambda { .. }) {
                let exp = param_tys.get(i).map(|pt| subst_type(pt, subst));
                let lv = self.lower_expr(&a.value, var_types, ret_ty, exp.as_ref());
                if !type_params.is_empty() {
                    if let Some(pt) = param_tys.get(i) {
                        let aty = self.infer_expr_ty(&lv, var_types);
                        self.unify_type_params(pt, &aty, type_params, subst);
                    }
                }
                out[i] = Some(CallArg {
                    name: a.name.clone(),
                    value: lv,
                });
            }
        }
        out.into_iter().map(|x| x.unwrap()).collect()
    }

    fn lower_expr(
        &mut self,
        expr: &Expr,
        var_types: &BTreeMap<String, TypeExpr>,
        ret_ty: Option<&TypeExpr>,
        expected_ty: Option<&TypeExpr>,
    ) -> Expr {
        let sp = expr.span;
        let kind = match &expr.kind {
            ExprKind::Float(x) => {
                if let Some(exp) = expected_ty {
                    if matches!(self.resolve_alias(exp), TypeExpr::Named { ref name, .. } if name == "Float32") {
                        return Expr {
                            kind: ExprKind::Cast {
                                expr: Box::new(Expr {
                                    kind: ExprKind::Float(*x),
                                    span: sp,
                                }),
                                ty: named_ty("Float32"),
                            },
                            span: sp,
                        };
                    }
                }
                ExprKind::Float(*x)
            }
            ExprKind::Ident(s) => {
                if !var_types.contains_key(s) {
                    if let Some((_, cval)) = self.consts.get(s).cloned() {
                        return self.lower_expr(&cval, var_types, ret_ty, expected_ty);
                    }
                }
                ExprKind::Ident(s.clone())
            }
            ExprKind::Unsafe(b) => {
                let mut lv = var_types.clone();
                ExprKind::Block(self.lower_block(b, &mut lv, expected_ty))
            }
            ExprKind::Cast { expr: inner, ty } => ExprKind::Cast {
                expr: Box::new(self.lower_expr(inner, var_types, ret_ty, None)),
                ty: ty.clone(),
            },
            ExprKind::TryCast { expr: inner, ty } => {
                let src_ty = self.infer_expr_ty(inner, var_types);
                let lv = self.lower_expr(inner, var_types, ret_ty, None);
                if self.is_refinement_cast_infallible(&src_ty, ty, sp) {
                    let cast_v = if self.resolve_alias(&src_ty).display_name()
                        != self.resolve_alias(ty).display_name()
                    {
                        Expr {
                            kind: ExprKind::Cast {
                                expr: Box::new(lv),
                                ty: ty.clone(),
                            },
                            span: sp,
                        }
                    } else {
                        lv
                    };
                    return Expr {
                        kind: ExprKind::Call {
                            callee: Box::new(Expr {
                                kind: ExprKind::Ident("Ok".into()),
                                span: sp,
                            }),
                            args: vec![CallArg {
                                name: None,
                                value: cast_v,
                            }],
                        },
                        span: sp,
                    };
                }
                return self.lower_fallible_refinement_check(
                    lv, &src_ty, ty, var_types, ret_ty, sp,
                );
            }
            ExprKind::ArrayRepeat { value, len } => {
                let exp_elem = match expected_ty.map(|t| self.resolve_alias(t)) {
                    Some(TypeExpr::Array { elem, .. }) => Some(*elem),
                    _ => None,
                };
                ExprKind::ArrayRepeat {
                    value: Box::new(self.lower_expr(value, var_types, ret_ty, exp_elem.as_ref())),
                    len: *len,
                }
            }
            ExprKind::Binary { op, left, right } => {
                let arith_exp = if matches!(
                    op,
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod
                ) {
                    expected_ty
                } else {
                    None
                };
                let ll0 = self.lower_expr(left, var_types, ret_ty, arith_exp);
                let l_inferred = self.infer_expr_ty(&ll0, var_types);
                let r_exp = if matches!(self.resolve_alias(&l_inferred), TypeExpr::Named { ref name, .. } if name == "Float32") {
                    Some(named_ty("Float32"))
                } else {
                    arith_exp.cloned()
                };
                let lr = self.lower_expr(right, var_types, ret_ty, r_exp.as_ref());
                let r_inferred = self.infer_expr_ty(&lr, var_types);
                let ll = if matches!(left.kind, ExprKind::Float(_))
                    && matches!(self.resolve_alias(&r_inferred), TypeExpr::Named { ref name, .. } if name == "Float32")
                    && !matches!(self.resolve_alias(&l_inferred), TypeExpr::Named { ref name, .. } if name == "Float32")
                {
                    self.lower_expr(left, var_types, ret_ty, Some(&named_ty("Float32")))
                } else {
                    ll0
                };
                if matches!(op, BinOp::Eq | BinOp::Ne) {
                    let lty = self.infer_expr_ty(&ll, var_types);
                    let lbase = self.resolve_alias(&lty);
                    if let TypeExpr::Named { name, args: rargs, .. } = &lbase {
                        let is_prim = ScalarKind::from_name(name).is_some()
                            || self.bitfields.contains_key(name)
                            || matches!(
                                name.as_str(),
                                "Bool" | "Text" | "Any" | "Ptr" | "MutPtr" | "CStr"
                            );
                        if !is_prim {
                            let lookup_name = if let TypeExpr::Named { name: orig, .. } = &lty {
                                if self.lookup_concrete_method(orig, "eq").is_some() {
                                    orig.clone()
                                } else {
                                    name.clone()
                                }
                            } else {
                                name.clone()
                            };
                            if let Some(mname) = self.lookup_concrete_method(&lookup_name, "eq") {
                                let full_args = vec![
                                    CallArg { name: None, value: ll },
                                    CallArg { name: None, value: lr },
                                ];
                                let final_fn = if let Some(tmpl) = self.generic_templates.get(&mname).cloned() {
                                    let mut subst = BTreeMap::new();
                                    for (tp, ra) in tmpl.type_params.iter().zip(rargs.iter()) {
                                        if ra.display_name() != "Any" {
                                            subst.insert(tp.clone(), ra.clone());
                                        }
                                    }
                                    let inferred = self.infer_generic_subst(&tmpl, &[], &full_args, var_types);
                                    for (k, v) in inferred {
                                        subst.entry(k).or_insert(v);
                                    }
                                    self.ensure_monomorphized(&mname, &subst).unwrap_or(mname)
                                } else {
                                    mname
                                };
                                let call_expr = Expr {
                                    kind: ExprKind::Call {
                                        callee: Box::new(Expr {
                                            kind: ExprKind::Ident(final_fn),
                                            span: sp,
                                        }),
                                        args: full_args,
                                    },
                                    span: sp,
                                };
                                return if *op == BinOp::Ne {
                                    Expr {
                                        kind: ExprKind::Unary {
                                            op: UnOp::Not,
                                            expr: Box::new(call_expr),
                                        },
                                        span: sp,
                                    }
                                } else {
                                    call_expr
                                };
                            }
                        }
                    }
                }
                ExprKind::Binary {
                    op: *op,
                    left: Box::new(ll),
                    right: Box::new(lr),
                }
            }
            ExprKind::Unary { op, expr: inner } => {
                let inner_exp = if matches!(op, UnOp::Neg | UnOp::BitNot) {
                    expected_ty
                } else {
                    None
                };
                ExprKind::Unary {
                    op: *op,
                    expr: Box::new(self.lower_expr(inner, var_types, ret_ty, inner_exp)),
                }
            }
            ExprKind::Field { target, field } => {
                let lt = self.lower_expr(target, var_types, ret_ty, None);
                let tty = self.resolve_alias(&self.infer_expr_ty(&lt, var_types));
                if let TypeExpr::Named { name: bname, .. } = &tty {
                    if let Some(bl) = self.bitfields.get(bname).cloned() {
                        if let Some(fl) = bl.fields.iter().find(|f| f.name == *field) {
                            let bty = named_ty(bl.backing.canonical_name());
                            let shifted = if fl.bit_offset == 0 {
                                lt
                            } else {
                                Expr {
                                    kind: ExprKind::Binary {
                                        op: BinOp::Shr,
                                        left: Box::new(lt),
                                        right: Box::new(Expr {
                                            kind: ExprKind::Int(fl.bit_offset as i64),
                                            span: sp,
                                        }),
                                    },
                                    span: sp,
                                }
                            };
                            return Expr {
                                kind: ExprKind::Binary {
                                    op: BinOp::BitAnd,
                                    left: Box::new(shifted),
                                    right: Box::new(Expr {
                                        kind: ExprKind::Cast {
                                            expr: Box::new(Expr {
                                                kind: ExprKind::Int(fl.mask as i64),
                                                span: sp,
                                            }),
                                            ty: bty,
                                        },
                                        span: sp,
                                    }),
                                },
                                span: sp,
                            };
                        }
                    }
                }
                ExprKind::Field {
                    target: Box::new(lt),
                    field: field.clone(),
                }
            }
            ExprKind::Index { target, index } => ExprKind::Index {
                target: Box::new(self.lower_expr(target, var_types, ret_ty, None)),
                index: Box::new(self.lower_expr(index, var_types, ret_ty, None)),
            },
            ExprKind::TupleIndex { target, index } => ExprKind::TupleIndex {
                target: Box::new(self.lower_expr(target, var_types, ret_ty, None)),
                index: *index,
            },
            ExprKind::Range { start, end } => ExprKind::Tuple(vec![
                self.lower_expr(start, var_types, ret_ty, None),
                self.lower_expr(end, var_types, ret_ty, None),
            ]),
            ExprKind::Try(inner) => {
                let li = self.lower_expr(inner, var_types, ret_ty, None);
                let ity = self.resolve_alias(&self.infer_expr_ty(&li, var_types));
                let is_opt = matches!(&ity, TypeExpr::Named { name, .. } if name == "Option")
                    || (!matches!(&ity, TypeExpr::Named { name, .. } if name == "Result")
                        && ret_ty
                            .map(|t| self.resolve_alias(t))
                            .is_some_and(|t| matches!(t, TypeExpr::Named { name, .. } if name == "Option")));
                let id = self.fresh_id();
                let vname = format!("__try_v_{}", id);
                if is_opt {
                    ExprKind::Match {
                        scrutinee: Box::new(li),
                        arms: vec![
                            MatchArm {
                                patterns: vec![Pattern::Variant {
                                    name: "Some".into(),
                                    binding: Some(vname.clone()),
                                    span: sp,
                                }],
                                guard: None,
                                body: Expr {
                                    kind: ExprKind::Ident(vname),
                                    span: sp,
                                },
                                span: sp,
                            },
                            MatchArm {
                                patterns: vec![Pattern::Variant {
                                    name: "None".into(),
                                    binding: None,
                                    span: sp,
                                }],
                                guard: None,
                                body: Expr {
                                    kind: ExprKind::Block(Block {
                                        stmts: vec![Stmt::Return {
                                            value: Some(Expr {
                                                kind: ExprKind::Ident("None".into()),
                                                span: sp,
                                            }),
                                            span: sp,
                                        }],
                                        tail: None,
                                        span: sp,
                                    }),
                                    span: sp,
                                },
                                span: sp,
                            },
                        ],
                    }
                } else {
                    let ename = format!("__try_e_{}", id);
                    ExprKind::Match {
                        scrutinee: Box::new(li),
                        arms: vec![
                            MatchArm {
                                patterns: vec![Pattern::Variant {
                                    name: "Ok".into(),
                                    binding: Some(vname.clone()),
                                    span: sp,
                                }],
                                guard: None,
                                body: Expr {
                                    kind: ExprKind::Ident(vname),
                                    span: sp,
                                },
                                span: sp,
                            },
                            MatchArm {
                                patterns: vec![Pattern::Variant {
                                    name: "Err".into(),
                                    binding: Some(ename.clone()),
                                    span: sp,
                                }],
                                guard: None,
                                body: Expr {
                                    kind: ExprKind::Block(Block {
                                        stmts: vec![Stmt::Return {
                                            value: Some(Expr {
                                                kind: ExprKind::Call {
                                                    callee: Box::new(Expr {
                                                        kind: ExprKind::Ident("Err".into()),
                                                        span: sp,
                                                    }),
                                                    args: vec![CallArg {
                                                        name: None,
                                                        value: Expr {
                                                            kind: ExprKind::Ident(ename),
                                                            span: sp,
                                                        },
                                                    }],
                                                },
                                                span: sp,
                                            }),
                                            span: sp,
                                        }],
                                        tail: None,
                                        span: sp,
                                    }),
                                    span: sp,
                                },
                                span: sp,
                            },
                        ],
                    }
                }
            }
            ExprKind::Lambda {
                params,
                ret_ty: lam_ret,
                body,
            } => {
                return self.lower_lambda(params, lam_ret.as_ref(), body, var_types, expected_ty, sp);
            }
            ExprKind::RecordLit {
                name,
                version,
                fields,
            } => {
                // Bitfield literal -> packed unboxed unsigned integer!
                if let Some(bl) = self.bitfields.get(name).cloned() {
                    let bty = named_ty(bl.backing.canonical_name());
                    let mut acc = Expr {
                        kind: ExprKind::Cast {
                            expr: Box::new(Expr {
                                kind: ExprKind::Int(0),
                                span: sp,
                            }),
                            ty: bty.clone(),
                        },
                        span: sp,
                    };
                    for (fname, fval) in fields {
                        if let Some(fl) = bl.fields.iter().find(|f| f.name == *fname) {
                            let lv = self.lower_expr(fval, var_types, ret_ty, Some(&bty));
                            let masked = Expr {
                                kind: ExprKind::Binary {
                                    op: BinOp::BitAnd,
                                    left: Box::new(Expr {
                                        kind: ExprKind::Cast {
                                            expr: Box::new(lv),
                                            ty: bty.clone(),
                                        },
                                        span: sp,
                                    }),
                                    right: Box::new(Expr {
                                        kind: ExprKind::Cast {
                                            expr: Box::new(Expr {
                                                kind: ExprKind::Int(fl.mask as i64),
                                                span: sp,
                                            }),
                                            ty: bty.clone(),
                                        },
                                        span: sp,
                                    }),
                                },
                                span: sp,
                            };
                            let shifted = if fl.bit_offset == 0 {
                                masked
                            } else {
                                Expr {
                                    kind: ExprKind::Binary {
                                        op: BinOp::Shl,
                                        left: Box::new(masked),
                                        right: Box::new(Expr {
                                            kind: ExprKind::Int(fl.bit_offset as i64),
                                            span: sp,
                                        }),
                                    },
                                    span: sp,
                                }
                            };
                            acc = Expr {
                                kind: ExprKind::Binary {
                                    op: BinOp::BitOr,
                                    left: Box::new(acc),
                                    right: Box::new(shifted),
                                },
                                span: sp,
                            };
                        }
                    }
                    return acc;
                }
                let rec_opt = self
                    .records
                    .get(name)
                    .or_else(|| self.generic_records.get(name))
                    .cloned();
                let mut rec_subst = BTreeMap::new();
                if let (Some(rec), Some(TypeExpr::Named { name: en, args: ea, .. })) =
                    (&rec_opt, expected_ty)
                {
                    if en == name && !rec.type_params.is_empty() {
                        for (tp, ta) in rec.type_params.iter().zip(ea.iter()) {
                            rec_subst.insert(tp.clone(), ta.clone());
                        }
                    }
                }
                let union_opt = self.unions.get(name).cloned();
                let new_fields = fields
                    .iter()
                    .map(|(k, v)| {
                        let f_expected = rec_opt
                            .as_ref()
                            .and_then(|rec| {
                                rec.fields.iter().find(|f| f.name == *k).map(|fd| {
                                    if rec_subst.is_empty() {
                                        fd.ty.clone()
                                    } else {
                                        subst_type(&fd.ty, &rec_subst)
                                    }
                                })
                            })
                            .or_else(|| {
                                union_opt.as_ref().and_then(|u| {
                                    u.fields.iter().find(|f| f.name == *k).map(|fd| fd.ty.clone())
                                })
                            });
                        let mut lv = self.lower_expr(v, var_types, ret_ty, f_expected.as_ref());
                        if let Some(fty) = &f_expected {
                            let aty = self.infer_expr_ty(&lv, var_types);
                            lv = self.maybe_box_dyn(lv, fty, &aty);
                        }
                        (k.clone(), lv)
                    })
                    .collect();
                ExprKind::RecordLit {
                    name: name.clone(),
                    version: version.clone(),
                    fields: new_fields,
                }
            }
            ExprKind::Tuple(es) => ExprKind::Tuple(
                es.iter()
                    .map(|e| self.lower_expr(e, var_types, ret_ty, None))
                    .collect(),
            ),
            ExprKind::List(es) => ExprKind::List(
                es.iter()
                    .map(|e| self.lower_expr(e, var_types, ret_ty, None))
                    .collect(),
            ),
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                let lc = self.lower_expr(cond, var_types, ret_ty, None);
                let mut tv = var_types.clone();
                let lt = self.lower_block(then_block, &mut tv, expected_ty);
                let le = else_branch
                    .as_ref()
                    .map(|eb| Box::new(self.lower_expr(eb, var_types, ret_ty, expected_ty)));
                ExprKind::If {
                    cond: Box::new(lc),
                    then_block: lt,
                    else_branch: le,
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                let ls = self.lower_expr(scrutinee, var_types, ret_ty, None);
                let st = self.infer_expr_ty(&ls, var_types);
                let mut new_arms = Vec::with_capacity(arms.len());
                for arm in arms {
                    let mut av = var_types.clone();
                    self.bind_arm_vars(&st, arm, &mut av);
                    let lg = arm
                        .guard
                        .as_ref()
                        .map(|g| self.lower_expr(g, &av, ret_ty, None));
                    let lb = self.lower_expr(&arm.body, &av, ret_ty, expected_ty);
                    new_arms.push(MatchArm {
                        patterns: arm.patterns.clone(),
                        guard: lg,
                        body: lb,
                        span: arm.span,
                    });
                }
                ExprKind::Match {
                    scrutinee: Box::new(ls),
                    arms: new_arms,
                }
            }
            ExprKind::Block(b) => {
                let mut lv = var_types.clone();
                ExprKind::Block(self.lower_block(b, &mut lv, expected_ty))
            }
            ExprKind::Await(inner) => {
                ExprKind::Await(Box::new(self.lower_expr(inner, var_types, ret_ty, None)))
            }
            ExprKind::Call { callee, args } => {
                match &callee.kind {
                    // 1. Explicit generic call `decode[Order](raw)` or layout/ptr/chan intrinsics
                    ExprKind::GenericInst { expr: inner, type_args } => {
                        if let ExprKind::Ident(fname) = &inner.kind {
                            if fname == "refine_cast" && type_args.len() == 1 && args.len() == 1 {
                                let dst_ty = &type_args[0];
                                let src_ty = self.infer_expr_ty(&args[0].value, var_types);
                                let lv = self.lower_expr(&args[0].value, var_types, ret_ty, None);
                                if self.is_refinement_cast_infallible(&src_ty, dst_ty, sp) {
                                    if self.resolve_alias(&src_ty).display_name()
                                        != self.resolve_alias(dst_ty).display_name()
                                    {
                                        return Expr {
                                            kind: ExprKind::Cast {
                                                expr: Box::new(lv),
                                                ty: dst_ty.clone(),
                                            },
                                            span: sp,
                                        };
                                    }
                                    return lv;
                                }
                                return self.lower_fallible_refinement_check(
                                    lv, &src_ty, dst_ty, var_types, ret_ty, sp,
                                );
                            }
                            if fname == "size_of" && args.is_empty() {
                                let sz = type_args
                                    .first()
                                    .map(|t| self.layout.size_and_align(t).0 as i64)
                                    .unwrap_or(8);
                                return Expr {
                                    kind: ExprKind::Cast {
                                        expr: Box::new(Expr {
                                            kind: ExprKind::Int(sz),
                                            span: sp,
                                        }),
                                        ty: named_ty("USize"),
                                    },
                                    span: sp,
                                };
                            }
                            if fname == "align_of" && args.is_empty() {
                                let al = type_args
                                    .first()
                                    .map(|t| self.layout.size_and_align(t).1 as i64)
                                    .unwrap_or(8);
                                return Expr {
                                    kind: ExprKind::Cast {
                                        expr: Box::new(Expr {
                                            kind: ExprKind::Int(al),
                                            span: sp,
                                        }),
                                        ty: named_ty("USize"),
                                    },
                                    span: sp,
                                };
                            }
                            if fname == "null_ptr" && args.is_empty() {
                                let target_ty = generic_ty(
                                    "MutPtr",
                                    vec![type_args.first().cloned().unwrap_or_else(|| named_ty("CVoid"))],
                                );
                                return Expr {
                                    kind: ExprKind::Cast {
                                        expr: Box::new(Expr {
                                            kind: ExprKind::Int(0),
                                            span: sp,
                                        }),
                                        ty: target_ty,
                                    },
                                    span: sp,
                                };
                            }
                            if fname == "chan_new" && args.is_empty() {
                                let is_rc = type_args
                                    .first()
                                    .is_some_and(|t| self.type_is_rc(t));
                                return Expr {
                                    kind: ExprKind::Call {
                                        callee: Box::new(Expr {
                                            kind: ExprKind::Ident("chan_new".into()),
                                            span: callee.span,
                                        }),
                                        args: vec![CallArg {
                                            name: None,
                                            value: Expr {
                                                kind: ExprKind::Int(if is_rc { 1 } else { 0 }),
                                                span: sp,
                                            },
                                        }],
                                    },
                                    span: sp,
                                };
                            }
                            if let Some(tmpl) = self.generic_templates.get(fname).cloned() {
                                let mut subst = BTreeMap::new();
                                for (tp, ta) in tmpl.type_params.iter().zip(type_args.iter()) {
                                    if ta.display_name() != "Any" {
                                        subst.insert(tp.clone(), ta.clone());
                                    }
                                }
                                let param_tys: Vec<TypeExpr> = tmpl.params.iter().map(|p| p.ty.clone()).collect();
                                let mut lowered_args = self.lower_args_two_pass(
                                    args,
                                    &param_tys,
                                    &tmpl.type_params,
                                    &mut subst,
                                    var_types,
                                    ret_ty,
                                );
                                if let Some(mono) = self.ensure_monomorphized(fname, &subst) {
                                    self.box_call_args(&mono, &mut lowered_args, var_types);
                                    return Expr {
                                        kind: ExprKind::Call {
                                            callee: Box::new(Expr {
                                                kind: ExprKind::Ident(mono),
                                                span: callee.span,
                                            }),
                                            args: lowered_args,
                                        },
                                        span: sp,
                                    };
                                }
                            }
                        } else if let ExprKind::Field { target, field } = &inner.kind {
                            // 1a. Associated generic method `Type.method[T](args)`
                            if let ExprKind::Ident(tname) = &target.kind {
                                if !var_types.contains_key(tname) && self.is_known_type(tname) {
                                    if let Some(mname) = self.lookup_concrete_method(tname, field) {
                                        let (final_fn, mut lowered_args) =
                                            if let Some(tmpl) = self.generic_templates.get(&mname).cloned() {
                                                let mut subst = BTreeMap::new();
                                                for (tp, ta) in tmpl.type_params.iter().zip(type_args.iter()) {
                                                    if ta.display_name() != "Any" {
                                                        subst.insert(tp.clone(), ta.clone());
                                                    }
                                                }
                                                let ptys: Vec<TypeExpr> = tmpl.params.iter().map(|p| p.ty.clone()).collect();
                                                let largs = self.lower_args_two_pass(
                                                    args,
                                                    &ptys,
                                                    &tmpl.type_params,
                                                    &mut subst,
                                                    var_types,
                                                    ret_ty,
                                                );
                                                if let (Some(rt), Some(exp)) = (&tmpl.ret_ty, expected_ty) {
                                                    self.unify_type_params(rt, exp, &tmpl.type_params, &mut subst);
                                                }
                                                (self.ensure_monomorphized(&mname, &subst).unwrap_or(mname), largs)
                                            } else {
                                                let ptys: Vec<TypeExpr> = self
                                                    .fn_sigs
                                                    .get(&mname)
                                                    .map(|s| s.params.iter().map(|p| p.ty.clone()).collect())
                                                    .unwrap_or_default();
                                                let mut es = BTreeMap::new();
                                                let largs = self.lower_args_two_pass(args, &ptys, &[], &mut es, var_types, ret_ty);
                                                (mname, largs)
                                            };
                                        self.box_call_args(&final_fn, &mut lowered_args, var_types);
                                        return Expr {
                                            kind: ExprKind::Call {
                                                callee: Box::new(Expr {
                                                    kind: ExprKind::Ident(final_fn),
                                                    span: callee.span,
                                                }),
                                                args: lowered_args,
                                            },
                                            span: sp,
                                        };
                                    }
                                }
                            }
                            // 1b. Receiver generic method `ctx.json_body[T](args)`
                            let lowered_target = self.lower_expr(target, var_types, ret_ty, None);
                            let recv_ty = self.infer_expr_ty(target, var_types);
                            let recv_base = self.resolve_alias(&recv_ty);
                            if let TypeExpr::Named { name, args: rargs, .. } = &recv_base {
                                let lookup_name = if let TypeExpr::Named { name: orig, .. } = &recv_ty {
                                    if self.lookup_concrete_method(orig, field).is_some() {
                                        orig.clone()
                                    } else {
                                        name.clone()
                                    }
                                } else {
                                    name.clone()
                                };
                                if let Some(mname) = self.lookup_concrete_method(&lookup_name, field) {
                                    let (final_fn, lowered_args) =
                                        if let Some(tmpl) = self.generic_templates.get(&mname).cloned() {
                                            let mut subst = BTreeMap::new();
                                            for (tp, ra) in tmpl.type_params.iter().zip(rargs.iter()) {
                                                if ra.display_name() != "Any" {
                                                    subst.insert(tp.clone(), ra.clone());
                                                }
                                            }
                                            if let Some(self_p) = tmpl.params.first() {
                                                self.unify_type_params(&self_p.ty, &recv_base, &tmpl.type_params, &mut subst);
                                            }
                                            let unbound_tps: Vec<String> = tmpl
                                                .type_params
                                                .iter()
                                                .filter(|tp| !subst.contains_key(*tp))
                                                .cloned()
                                                .collect();
                                            for (tp, ta) in unbound_tps.iter().zip(type_args.iter()) {
                                                if ta.display_name() != "Any" {
                                                    subst.insert(tp.clone(), ta.clone());
                                                }
                                            }
                                            let explicit_ptys: Vec<TypeExpr> =
                                                tmpl.params.iter().skip(1).map(|p| p.ty.clone()).collect();
                                            let largs = self.lower_args_two_pass(
                                                args,
                                                &explicit_ptys,
                                                &tmpl.type_params,
                                                &mut subst,
                                                var_types,
                                                ret_ty,
                                            );
                                            if let (Some(rt), Some(exp)) = (&tmpl.ret_ty, expected_ty) {
                                                self.unify_type_params(rt, exp, &tmpl.type_params, &mut subst);
                                            }
                                            (self.ensure_monomorphized(&mname, &subst).unwrap_or(mname), largs)
                                        } else {
                                            let explicit_ptys: Vec<TypeExpr> = self
                                                .fn_sigs
                                                .get(&mname)
                                                .map(|s| s.params.iter().skip(1).map(|p| p.ty.clone()).collect())
                                                .unwrap_or_default();
                                            let mut es = BTreeMap::new();
                                            let largs = self.lower_args_two_pass(
                                                args,
                                                &explicit_ptys,
                                                &[],
                                                &mut es,
                                                var_types,
                                                ret_ty,
                                            );
                                            (mname, largs)
                                        };
                                    let mut full_args = Vec::with_capacity(lowered_args.len() + 1);
                                    full_args.push(CallArg {
                                        name: None,
                                        value: lowered_target,
                                    });
                                    full_args.extend(lowered_args);
                                    self.box_call_args(&final_fn, &mut full_args, var_types);
                                    return Expr {
                                        kind: ExprKind::Call {
                                            callee: Box::new(Expr {
                                                kind: ExprKind::Ident(final_fn),
                                                span: callee.span,
                                            }),
                                            args: full_args,
                                        },
                                        span: sp,
                                    };
                                }
                            }
                        }
                        let lowered_args: Vec<CallArg> = args
                            .iter()
                            .map(|a| CallArg {
                                name: a.name.clone(),
                                value: self.lower_expr(&a.value, var_types, ret_ty, None),
                            })
                            .collect();
                        ExprKind::Call {
                            callee: callee.clone(),
                            args: lowered_args,
                        }
                    }
                    // 2. Direct function call OR indirect closure variable call!
                    ExprKind::Ident(fname) => {
                        if fname == "offset_of" && args.len() == 2 {
                            let off = if let (ExprKind::Ident(tname), ExprKind::Str(field_name)) =
                                (&args[0].value.kind, &args[1].value.kind)
                            {
                                self.layout.offset_of(tname, field_name).unwrap_or(0) as i64
                            } else {
                                0
                            };
                            return Expr {
                                kind: ExprKind::Cast {
                                    expr: Box::new(Expr {
                                        kind: ExprKind::Int(off),
                                        span: sp,
                                    }),
                                    ty: named_ty("USize"),
                                },
                                span: sp,
                            };
                        }
                        if fname == "mutex_new" && args.len() == 1 {
                            let lv = self.lower_expr(&args[0].value, var_types, ret_ty, None);
                            let vty = self.infer_expr_ty(&lv, var_types);
                            let is_rc = if self.type_is_rc(&vty) { 1 } else { 0 };
                            return Expr {
                                kind: ExprKind::Call {
                                    callee: callee.clone(),
                                    args: vec![
                                        CallArg { name: None, value: lv },
                                        CallArg {
                                            name: None,
                                            value: Expr {
                                                kind: ExprKind::Int(is_rc),
                                                span: sp,
                                            },
                                        },
                                    ],
                                },
                                span: sp,
                            };
                        }
                        if fname == "chan_new" && args.is_empty() {
                            let is_rc = match expected_ty.map(|t| self.resolve_alias(t)) {
                                Some(TypeExpr::Named { name, args: ca, .. })
                                    if name == "Channel" && !ca.is_empty() =>
                                {
                                    if self.type_is_rc(&ca[0]) { 1 } else { 0 }
                                }
                                _ => 0,
                            };
                            return Expr {
                                kind: ExprKind::Call {
                                    callee: callee.clone(),
                                    args: vec![CallArg {
                                        name: None,
                                        value: Expr {
                                            kind: ExprKind::Int(is_rc),
                                            span: sp,
                                        },
                                    }],
                                },
                                span: sp,
                            };
                        }
                        if fname == "ptr_add" && args.len() == 2 {
                            let lp = self.lower_expr(&args[0].value, var_types, ret_ty, None);
                            let ln = self.lower_expr(&args[1].value, var_types, ret_ty, None);
                            let elem_sz = match self.resolve_alias(&self.infer_expr_ty(&lp, var_types)) {
                                TypeExpr::Named { name, args: pa, .. }
                                    if (name == "Ptr" || name == "MutPtr") && !pa.is_empty() =>
                                {
                                    self.layout.size_and_align(&pa[0]).0.max(1) as i64
                                }
                                _ => 1,
                            };
                            return Expr {
                                kind: ExprKind::Call {
                                    callee: Box::new(Expr {
                                        kind: ExprKind::Ident("__ptr_add".into()),
                                        span: callee.span,
                                    }),
                                    args: vec![
                                        CallArg { name: None, value: lp },
                                        CallArg { name: None, value: ln },
                                        CallArg {
                                            name: None,
                                            value: Expr {
                                                kind: ExprKind::Int(elem_sz),
                                                span: sp,
                                            },
                                        },
                                    ],
                                },
                                span: sp,
                            };
                        }
                        // 2a. Local variable / parameter of type `fn(...)` -> `__call_closure(fname, args...)`
                        if let Some(vty) = var_types.get(fname).cloned() {
                            if let TypeExpr::Fn { params: fparams, .. } = self.resolve_alias(&vty) {
                                let mut empty_subst = BTreeMap::new();
                                let lowered_args = self.lower_args_two_pass(
                                    args,
                                    &fparams,
                                    &[],
                                    &mut empty_subst,
                                    var_types,
                                    ret_ty,
                                );
                                let mut call_args = Vec::with_capacity(1 + lowered_args.len());
                                call_args.push(CallArg {
                                    name: None,
                                    value: (**callee).clone(),
                                });
                                call_args.extend(lowered_args);
                                return Expr {
                                    kind: ExprKind::Call {
                                        callee: Box::new(Expr {
                                            kind: ExprKind::Ident("__call_closure".into()),
                                            span: callee.span,
                                        }),
                                        args: call_args,
                                    },
                                    span: sp,
                                };
                            }
                        }
                        // 2b. Generic function call `apply(x, |v| v + 1)`
                        if let Some(tmpl) = self.generic_templates.get(fname).cloned() {
                            let mut subst = BTreeMap::new();
                            let param_tys: Vec<TypeExpr> = tmpl.params.iter().map(|p| p.ty.clone()).collect();
                            let mut lowered_args = self.lower_args_two_pass(
                                args,
                                &param_tys,
                                &tmpl.type_params,
                                &mut subst,
                                var_types,
                                ret_ty,
                            );
                            if let Some(mono) = self.ensure_monomorphized(fname, &subst) {
                                self.box_call_args(&mono, &mut lowered_args, var_types);
                                return Expr {
                                    kind: ExprKind::Call {
                                        callee: Box::new(Expr {
                                            kind: ExprKind::Ident(mono),
                                            span: callee.span,
                                        }),
                                        args: lowered_args,
                                    },
                                    span: sp,
                                };
                            }
                        }
                        // 2c. Concrete function / extern fn / builtin / enum variant call
                        let expected_ptys: Vec<TypeExpr> = if let Some(sig) = self.fn_sigs.get(fname) {
                            sig.params.iter().map(|p| p.ty.clone()).collect()
                        } else if let Some(ef) = self.extern_fns.get(fname) {
                            ef.params.iter().map(|p| p.ty.clone()).collect()
                        } else if fname == "simd_f32x4_new" {
                            vec![named_ty("Float32"); 4]
                        } else if fname == "simd_f32x4_splat" || fname == "f32_to_text" {
                            vec![named_ty("Float32")]
                        } else {
                            Vec::new()
                        };
                        let mut empty_subst = BTreeMap::new();
                        let mut lowered_args = self.lower_args_two_pass(
                            args,
                            &expected_ptys,
                            &[],
                            &mut empty_subst,
                            var_types,
                            ret_ty,
                        );
                        if fname == "push" && lowered_args.len() == 2 {
                            let lt = self.resolve_alias(&self.infer_expr_ty(&lowered_args[0].value, var_types));
                            if let TypeExpr::Named { name: ln, args: la, .. } = lt {
                                if ln == "List" && la.len() == 1 {
                                    let it = self.infer_expr_ty(&lowered_args[1].value, var_types);
                                    let val = lowered_args[1].value.clone();
                                    lowered_args[1].value = self.maybe_box_dyn(val, &la[0], &it);
                                }
                            }
                        }
                        self.box_call_args(fname, &mut lowered_args, var_types);
                        ExprKind::Call {
                            callee: callee.clone(),
                            args: lowered_args,
                        }
                    }
                    // 3. Associated call `Type.method(args)` OR receiver method call `x.method(args)` OR callable field `x.cb(args)`!
                    ExprKind::Field { target, field } => {
                        // Preserve auto-derived `Record.parse(...)` (unless explicit `impl Record::parse` exists)
                        if field == "parse" {
                            if let ExprKind::Ident(rname) = &target.kind {
                                if self.records.contains_key(rname)
                                    && self.lookup_concrete_method(rname, "parse").is_none()
                                {
                                    let lowered_args: Vec<CallArg> = args
                                        .iter()
                                        .map(|a| CallArg {
                                            name: a.name.clone(),
                                            value: self.lower_expr(&a.value, var_types, ret_ty, None),
                                        })
                                        .collect();
                                    return Expr {
                                        kind: ExprKind::Call {
                                            callee: callee.clone(),
                                            args: lowered_args,
                                        },
                                        span: sp,
                                    };
                                }
                            }
                        }
                        if let ExprKind::GenericInst { expr: inner, type_args } = &target.kind {
                            if let ExprKind::Ident(tname) = &inner.kind {
                                if !var_types.contains_key(tname) && self.is_known_type(tname) {
                                    if let Some(mname) = self.lookup_concrete_method(tname, field) {
                                        let (final_fn, mut lowered_args) =
                                            if let Some(tmpl) = self.generic_templates.get(&mname).cloned() {
                                                let mut subst = BTreeMap::new();
                                                for (tp, ta) in tmpl.type_params.iter().zip(type_args.iter()) {
                                                    if ta.display_name() != "Any" {
                                                        subst.insert(tp.clone(), ta.clone());
                                                    }
                                                }
                                                let ptys: Vec<TypeExpr> = tmpl.params.iter().map(|p| p.ty.clone()).collect();
                                                let largs = self.lower_args_two_pass(
                                                    args,
                                                    &ptys,
                                                    &tmpl.type_params,
                                                    &mut subst,
                                                    var_types,
                                                    ret_ty,
                                                );
                                                if let (Some(rt), Some(exp)) = (&tmpl.ret_ty, expected_ty) {
                                                    self.unify_type_params(rt, exp, &tmpl.type_params, &mut subst);
                                                }
                                                (self.ensure_monomorphized(&mname, &subst).unwrap_or(mname), largs)
                                            } else {
                                                let ptys: Vec<TypeExpr> = self
                                                    .fn_sigs
                                                    .get(&mname)
                                                    .map(|s| s.params.iter().map(|p| p.ty.clone()).collect())
                                                    .unwrap_or_default();
                                                let mut es = BTreeMap::new();
                                                let largs = self.lower_args_two_pass(args, &ptys, &[], &mut es, var_types, ret_ty);
                                                (mname, largs)
                                            };
                                        self.box_call_args(&final_fn, &mut lowered_args, var_types);
                                        return Expr {
                                            kind: ExprKind::Call {
                                                callee: Box::new(Expr {
                                                    kind: ExprKind::Ident(final_fn),
                                                    span: callee.span,
                                                }),
                                                args: lowered_args,
                                            },
                                            span: sp,
                                        };
                                    }
                                }
                            }
                        }
                        if let ExprKind::Ident(ename) = &target.kind {
                            if field == "from_raw" && args.len() == 1 {
                                if let Some(bl) = self.bitfields.get(ename).cloned() {
                                    let lv = self.lower_expr(&args[0].value, var_types, ret_ty, None);
                                    return Expr {
                                        kind: ExprKind::Cast {
                                            expr: Box::new(lv),
                                            ty: named_ty(bl.backing.canonical_name()),
                                        },
                                        span: sp,
                                    };
                                }
                            }
                            if self.enum_variants.contains_key(&format!("{}.{}", ename, field)) {
                                let lowered_args: Vec<CallArg> = args
                                    .iter()
                                    .map(|a| CallArg {
                                        name: a.name.clone(),
                                        value: self.lower_expr(&a.value, var_types, ret_ty, None),
                                    })
                                    .collect();
                                return Expr {
                                    kind: ExprKind::Call {
                                        callee: callee.clone(),
                                        args: lowered_args,
                                    },
                                    span: sp,
                                };
                            }
                            if !var_types.contains_key(ename) && self.is_known_type(ename) {
                                if let Some(mname) = self.lookup_concrete_method(ename, field) {
                                    let (final_fn, mut lowered_args) =
                                        if let Some(tmpl) = self.generic_templates.get(&mname).cloned() {
                                            let mut subst = BTreeMap::new();
                                            let ptys: Vec<TypeExpr> = tmpl.params.iter().map(|p| p.ty.clone()).collect();
                                            let largs = self.lower_args_two_pass(
                                                args,
                                                &ptys,
                                                &tmpl.type_params,
                                                &mut subst,
                                                var_types,
                                                ret_ty,
                                            );
                                            if let (Some(rt), Some(exp)) = (&tmpl.ret_ty, expected_ty) {
                                                self.unify_type_params(rt, exp, &tmpl.type_params, &mut subst);
                                            }
                                            (self.ensure_monomorphized(&mname, &subst).unwrap_or(mname), largs)
                                        } else {
                                            let ptys: Vec<TypeExpr> = self
                                                .fn_sigs
                                                .get(&mname)
                                                .map(|s| s.params.iter().map(|p| p.ty.clone()).collect())
                                                .unwrap_or_default();
                                            let mut es = BTreeMap::new();
                                            let largs = self.lower_args_two_pass(args, &ptys, &[], &mut es, var_types, ret_ty);
                                            (mname, largs)
                                        };
                                    self.box_call_args(&final_fn, &mut lowered_args, var_types);
                                    return Expr {
                                        kind: ExprKind::Call {
                                            callee: Box::new(Expr {
                                                kind: ExprKind::Ident(final_fn),
                                                span: callee.span,
                                            }),
                                            args: lowered_args,
                                        },
                                        span: sp,
                                    };
                                }
                            }
                        }

                        let lowered_target = self.lower_expr(target, var_types, ret_ty, None);
                        let recv_ty = self.infer_expr_ty(&lowered_target, var_types);
                        let recv_base = self.resolve_alias(&recv_ty);

                        match &recv_base {
                            TypeExpr::Dyn { interface, .. } => {
                                let param_tys = self
                                    .dyn_interfaces
                                    .get(interface)
                                    .and_then(|l| l.methods.iter().find(|m| m.name == *field))
                                    .map(|dm| dm.param_tys.clone())
                                    .unwrap_or_default();
                                let mut es = BTreeMap::new();
                                let mut lowered_args = self.lower_args_two_pass(
                                    args,
                                    &param_tys,
                                    &[],
                                    &mut es,
                                    var_types,
                                    ret_ty,
                                );
                                for (pty, a) in param_tys.iter().zip(lowered_args.iter_mut()) {
                                    let aty = self.infer_expr_ty(&a.value, var_types);
                                    let v = a.value.clone();
                                    a.value = self.maybe_box_dyn(v, pty, &aty);
                                }
                                ExprKind::Call {
                                    callee: Box::new(Expr {
                                        kind: ExprKind::Field {
                                            target: Box::new(lowered_target),
                                            field: field.clone(),
                                        },
                                        span: callee.span,
                                    }),
                                    args: lowered_args,
                                }
                            }
                            TypeExpr::Named { name, args: rargs, .. } => {
                                if field == "raw" && args.is_empty() {
                                    if let Some(bl) = self.bitfields.get(name).cloned() {
                                        return Expr {
                                            kind: ExprKind::Cast {
                                                expr: Box::new(lowered_target),
                                                ty: named_ty(bl.backing.canonical_name()),
                                            },
                                            span: sp,
                                        };
                                    }
                                }
                                let lookup_name = if let TypeExpr::Named { name: orig, .. } = &recv_ty {
                                    if self.lookup_concrete_method(orig, field).is_some() {
                                        orig.clone()
                                    } else {
                                        name.clone()
                                    }
                                } else {
                                    name.clone()
                                };
                                if let Some(mname) = self.lookup_concrete_method(&lookup_name, field) {
                                    let (final_fn, lowered_args) =
                                        if let Some(tmpl) = self.generic_templates.get(&mname).cloned() {
                                            let mut subst = BTreeMap::new();
                                            for (tp, ra) in tmpl.type_params.iter().zip(rargs.iter()) {
                                                if ra.display_name() != "Any" {
                                                    subst.insert(tp.clone(), ra.clone());
                                                }
                                            }
                                            if let Some(self_p) = tmpl.params.first() {
                                                self.unify_type_params(&self_p.ty, &recv_base, &tmpl.type_params, &mut subst);
                                            }
                                            let explicit_ptys: Vec<TypeExpr> =
                                                tmpl.params.iter().skip(1).map(|p| p.ty.clone()).collect();
                                            let largs = self.lower_args_two_pass(
                                                args,
                                                &explicit_ptys,
                                                &tmpl.type_params,
                                                &mut subst,
                                                var_types,
                                                ret_ty,
                                            );
                                            if let (Some(rt), Some(exp)) = (&tmpl.ret_ty, expected_ty) {
                                                self.unify_type_params(rt, exp, &tmpl.type_params, &mut subst);
                                            }
                                            (self.ensure_monomorphized(&mname, &subst).unwrap_or(mname), largs)
                                        } else {
                                            let explicit_ptys: Vec<TypeExpr> = self
                                                .fn_sigs
                                                .get(&mname)
                                                .map(|s| s.params.iter().skip(1).map(|p| p.ty.clone()).collect())
                                                .unwrap_or_default();
                                            let mut es = BTreeMap::new();
                                            let largs = self.lower_args_two_pass(
                                                args,
                                                &explicit_ptys,
                                                &[],
                                                &mut es,
                                                var_types,
                                                ret_ty,
                                            );
                                            (mname, largs)
                                        };
                                    let mut full_args = Vec::with_capacity(lowered_args.len() + 1);
                                    full_args.push(CallArg {
                                        name: None,
                                        value: lowered_target,
                                    });
                                    full_args.extend(lowered_args);
                                    self.box_call_args(&final_fn, &mut full_args, var_types);
                                    ExprKind::Call {
                                        callee: Box::new(Expr {
                                            kind: ExprKind::Ident(final_fn),
                                            span: callee.span,
                                        }),
                                        args: full_args,
                                    }
                                } else {
                                    // Callable record field `obj.callback(args)` -> `__call_closure(obj.callback, args...)`
                                    let field_expr = Expr {
                                        kind: ExprKind::Field {
                                            target: Box::new(lowered_target),
                                            field: field.clone(),
                                        },
                                        span: callee.span,
                                    };
                                    if let TypeExpr::Fn { params: fparams, .. } =
                                        self.resolve_alias(&self.infer_expr_ty(&field_expr, var_types))
                                    {
                                        let mut es = BTreeMap::new();
                                        let lowered_args = self.lower_args_two_pass(
                                            args,
                                            &fparams,
                                            &[],
                                            &mut es,
                                            var_types,
                                            ret_ty,
                                        );
                                        let mut call_args = Vec::with_capacity(1 + lowered_args.len());
                                        call_args.push(CallArg { name: None, value: field_expr });
                                        call_args.extend(lowered_args);
                                        ExprKind::Call {
                                            callee: Box::new(Expr {
                                                kind: ExprKind::Ident("__call_closure".into()),
                                                span: callee.span,
                                            }),
                                            args: call_args,
                                        }
                                    } else {
                                        let lowered_args: Vec<CallArg> = args
                                            .iter()
                                            .map(|a| CallArg {
                                                name: a.name.clone(),
                                                value: self.lower_expr(&a.value, var_types, ret_ty, None),
                                            })
                                            .collect();
                                        ExprKind::Call {
                                            callee: Box::new(field_expr),
                                            args: lowered_args,
                                        }
                                    }
                                }
                            }
                            _ => {
                                let lowered_args: Vec<CallArg> = args
                                    .iter()
                                    .map(|a| CallArg {
                                        name: a.name.clone(),
                                        value: self.lower_expr(&a.value, var_types, ret_ty, None),
                                    })
                                    .collect();
                                ExprKind::Call {
                                    callee: Box::new(Expr {
                                        kind: ExprKind::Field {
                                            target: Box::new(lowered_target),
                                            field: field.clone(),
                                        },
                                        span: callee.span,
                                    }),
                                    args: lowered_args,
                                }
                            }
                        }
                    }
                    // 4. Arbitrary expression callee (e.g. `(make_adder(10))(5)` or `fns[0](5)`)
                    _ => {
                        let lc = self.lower_expr(callee, var_types, ret_ty, None);
                        if let TypeExpr::Fn { params: fparams, .. } =
                            self.resolve_alias(&self.infer_expr_ty(&lc, var_types))
                        {
                            let mut es = BTreeMap::new();
                            let lowered_args = self.lower_args_two_pass(
                                args,
                                &fparams,
                                &[],
                                &mut es,
                                var_types,
                                ret_ty,
                            );
                            let mut call_args = Vec::with_capacity(1 + lowered_args.len());
                            call_args.push(CallArg { name: None, value: lc });
                            call_args.extend(lowered_args);
                            ExprKind::Call {
                                callee: Box::new(Expr {
                                    kind: ExprKind::Ident("__call_closure".into()),
                                    span: callee.span,
                                }),
                                args: call_args,
                            }
                        } else {
                            let lowered_args: Vec<CallArg> = args
                                .iter()
                                .map(|a| CallArg {
                                    name: a.name.clone(),
                                    value: self.lower_expr(&a.value, var_types, ret_ty, None),
                                })
                                .collect();
                            ExprKind::Call {
                                callee: Box::new(lc),
                                args: lowered_args,
                            }
                        }
                    }
                }
            }
            other => other.clone(),
        };
        Expr { kind, span: sp }
    }

    fn ensure_enum_monomorphized(&mut self, base_name: &str, type_args: &[TypeExpr]) -> String {
        let resolved_args: Vec<TypeExpr> = type_args
            .iter()
            .map(|a| self.rewrite_generic_record_ty(a))
            .collect();
        let mono_name = mangle_ty(&generic_ty(base_name, resolved_args.clone()));
        if !self.enums.contains_key(&mono_name) {
            if let Some(tmpl) = self.generic_enums.get(base_name).cloned() {
                let mut subst = BTreeMap::new();
                for (tp, ta) in tmpl.type_params.iter().zip(resolved_args.iter()) {
                    subst.insert(tp.clone(), ta.clone());
                }
                // Insert placeholder FIRST so recursive enums like `Tree[T]` don't infinite-loop!
                let placeholder = EnumDecl {
                    attrs: tmpl.attrs.clone(),
                    is_pub: tmpl.is_pub,
                    name: mono_name.clone(),
                    type_params: Vec::new(),
                    type_param_bounds: BTreeMap::new(),
                    variants: Vec::new(),
                    span: tmpl.span,
                };
                self.enums.insert(mono_name.clone(), placeholder);
                self.enum_names.insert(mono_name.clone());

                let mut spec = tmpl;
                spec.name = mono_name.clone();
                spec.type_params.clear();
                spec.type_param_bounds.clear();
                for v in &mut spec.variants {
                    let mangled_vname = format!("{}__{}", mono_name, v.name);
                    v.name = mangled_vname.clone();
                    for f in &mut v.fields {
                        let st = subst_type(&f.ty, &subst);
                        f.ty = self.rewrite_generic_record_ty(&st);
                    }
                    let ptys: Vec<TypeExpr> = v.fields.iter().map(|f| f.ty.clone()).collect();
                    self.enum_variants.insert(mangled_vname.clone(), (mono_name.clone(), ptys.clone()));
                    self.enum_variants.insert(format!("{}.{}", mono_name, mangled_vname), (mono_name.clone(), ptys));
                }
                self.enums.insert(mono_name.clone(), spec.clone());
                self.mono_enums.push(spec);
            }
        }
        mono_name
    }

    fn ensure_record_monomorphized(&mut self, base_name: &str, type_args: &[TypeExpr]) -> String {
        let resolved_args: Vec<TypeExpr> = type_args
            .iter()
            .map(|a| self.rewrite_generic_record_ty(a))
            .collect();
        let mono_name = mangle_ty(&generic_ty(base_name, resolved_args.clone()));
        if !self.records.contains_key(&mono_name) {
            if let Some(tmpl) = self.generic_records.get(base_name).cloned() {
                let mut subst = BTreeMap::new();
                for (tp, ta) in tmpl.type_params.iter().zip(resolved_args.iter()) {
                    subst.insert(tp.clone(), ta.clone());
                }
                let placeholder = RecordDecl {
                    attrs: tmpl.attrs.clone(),
                    is_pub: tmpl.is_pub,
                    name: mono_name.clone(),
                    type_params: Vec::new(),
                    type_param_bounds: BTreeMap::new(),
                    fields: Vec::new(),
                    where_clauses: Vec::new(),
                    invariants: Vec::new(),
                    methods: Vec::new(),
                    span: tmpl.span,
                };
                self.records.insert(mono_name.clone(), placeholder);
                let mut spec = tmpl;
                spec.name = mono_name.clone();
                spec.type_params.clear();
                spec.type_param_bounds.clear();
                for f in &mut spec.fields {
                    let st = subst_type(&f.ty, &subst);
                    f.ty = self.rewrite_generic_record_ty(&st);
                }
                self.records.insert(mono_name.clone(), spec.clone());
                self.mono_records.push(spec);
            }
        }
        mono_name
    }

    fn rewrite_generic_record_ty(&mut self, ty: &TypeExpr) -> TypeExpr {
        match ty {
            TypeExpr::Named {
                name,
                version,
                args,
                span,
            } => {
                if let Some(bl) = self.bitfields.get(name) {
                    return TypeExpr::Named {
                        name: bl.backing.canonical_name().to_string(),
                        version: None,
                        args: Vec::new(),
                        span: *span,
                    };
                }
                if self.generic_records.contains_key(name) && !args.is_empty() {
                    let mono_name = self.ensure_record_monomorphized(name, args);
                    return TypeExpr::Named {
                        name: mono_name,
                        version: version.clone(),
                        args: Vec::new(),
                        span: *span,
                    };
                }
                if self.generic_enums.contains_key(name) && !args.is_empty() {
                    let mono_name = self.ensure_enum_monomorphized(name, args);
                    return TypeExpr::Named {
                        name: mono_name,
                        version: version.clone(),
                        args: Vec::new(),
                        span: *span,
                    };
                }
                TypeExpr::Named {
                    name: name.clone(),
                    version: version.clone(),
                    args: args.iter().map(|a| self.rewrite_generic_record_ty(a)).collect(),
                    span: *span,
                }
            }
            TypeExpr::Array { elem, len, span } => TypeExpr::Array {
                elem: Box::new(self.rewrite_generic_record_ty(elem)),
                len: *len,
                span: *span,
            },
            TypeExpr::Dyn {
                interface,
                args,
                span,
            } => TypeExpr::Dyn {
                interface: interface.clone(),
                args: args.iter().map(|a| self.rewrite_generic_record_ty(a)).collect(),
                span: *span,
            },
            TypeExpr::Tuple(es, span) => TypeExpr::Tuple(
                es.iter().map(|e| self.rewrite_generic_record_ty(e)).collect(),
                *span,
            ),
            TypeExpr::Fn { params, ret, span } => TypeExpr::Fn {
                params: params.iter().map(|p| self.rewrite_generic_record_ty(p)).collect(),
                ret: ret.as_ref().map(|r| Box::new(self.rewrite_generic_record_ty(r))),
                span: *span,
            },
        }
    }

    fn rewrite_pattern_generic_enums(&mut self, pat: &mut Pattern, scrutinee_ty: &TypeExpr) {
        let base = self.resolve_alias(scrutinee_ty);
        match pat {
            Pattern::Tuple(pats, _) => {
                let elems = match &base {
                    TypeExpr::Tuple(es, _) => es.clone(),
                    _ => Vec::new(),
                };
                for (i, sp) in pats.iter_mut().enumerate() {
                    let ety = elems.get(i).cloned().unwrap_or_else(|| named_ty("Any"));
                    self.rewrite_pattern_generic_enums(sp, &ety);
                }
            }
            Pattern::Variant { name, .. } => {
                let short = name.rsplit('.').next().unwrap_or(name).to_string();
                if let Some((ename, _)) = self.enum_variants.get(&short).cloned() {
                    if self.generic_enums.contains_key(&ename) {
                        if let TypeExpr::Named { name: sn, args: sa, .. } = &base {
                            let mono = if sn == &ename && !sa.is_empty() {
                                self.ensure_enum_monomorphized(&ename, sa)
                            } else if self.enums.contains_key(sn) && sn.starts_with(&format!("{}__", ename)) {
                                sn.clone()
                            } else {
                                return;
                            };
                            *name = format!("{}__{}", mono, short);
                        }
                    }
                }
            }
            Pattern::VariantMulti { name, patterns, .. } => {
                let short = name.rsplit('.').next().unwrap_or(name).to_string();
                let mut child_tys = self.resolve_variant_payload_tys(&base, name);
                if let Some((ename, _)) = self.enum_variants.get(&short).cloned() {
                    if self.generic_enums.contains_key(&ename) {
                        if let TypeExpr::Named { name: sn, args: sa, .. } = &base {
                            let mono_opt = if sn == &ename && !sa.is_empty() {
                                Some(self.ensure_enum_monomorphized(&ename, sa))
                            } else if self.enums.contains_key(sn) && sn.starts_with(&format!("{}__", ename)) {
                                Some(sn.clone())
                            } else {
                                None
                            };
                            if let Some(mono) = mono_opt {
                                let mangled_v = format!("{}__{}", mono, short);
                                *name = mangled_v.clone();
                                if let Some((_, ptys)) = self.enum_variants.get(&mangled_v) {
                                    child_tys = ptys.clone();
                                }
                            }
                        }
                    }
                }
                for (i, sp) in patterns.iter_mut().enumerate() {
                    let pty = child_tys.get(i).cloned().unwrap_or_else(|| named_ty("Any"));
                    self.rewrite_pattern_generic_enums(sp, &pty);
                }
            }
            _ => {}
        }
    }

    fn rewrite_block_generic_records(
        &mut self,
        blk: &mut Block,
        var_types: &mut BTreeMap<String, TypeExpr>,
        ret_ty: Option<&TypeExpr>,
    ) {
        let saved = var_types.clone();
        for stmt in &mut blk.stmts {
            match stmt {
                Stmt::Let { name, ty, value, .. } => {
                    self.rewrite_expr_generic_records(value, var_types, ty.as_ref());
                    let eff = ty
                        .clone()
                        .unwrap_or_else(|| self.infer_expr_ty(value, var_types));
                    var_types.insert(name.clone(), eff);
                    if let Some(t) = ty {
                        *t = self.rewrite_generic_record_ty(t);
                    }
                }
                Stmt::Assign { target, value, .. } => {
                    self.rewrite_expr_generic_records(target, var_types, None);
                    let tty = self.infer_expr_ty(target, var_types);
                    let exp = if tty.display_name() != "Any" { Some(&tty) } else { None };
                    self.rewrite_expr_generic_records(value, var_types, exp);
                }
                Stmt::Return { value: Some(v), .. } => {
                    self.rewrite_expr_generic_records(v, var_types, ret_ty);
                }
                Stmt::Assert { cond, .. } => {
                    self.rewrite_expr_generic_records(cond, var_types, None);
                }
                Stmt::Mutate { body, .. } => {
                    self.rewrite_block_generic_records(body, var_types, ret_ty);
                }
                Stmt::While { cond, body, .. } => {
                    self.rewrite_expr_generic_records(cond, var_types, None);
                    self.rewrite_block_generic_records(body, var_types, ret_ty);
                }
                Stmt::Expr { expr, .. } => {
                    self.rewrite_expr_generic_records(expr, var_types, None);
                }
                _ => {}
            }
        }
        if let Some(t) = &mut blk.tail {
            self.rewrite_expr_generic_records(t, var_types, ret_ty);
        }
        *var_types = saved;
    }

    fn rewrite_expr_generic_records(
        &mut self,
        expr: &mut Expr,
        var_types: &BTreeMap<String, TypeExpr>,
        expected_ty: Option<&TypeExpr>,
    ) {
        match &mut expr.kind {
            ExprKind::RecordLit { name, fields, .. } => {
                if let Some(tmpl) = self.generic_records.get(name.as_str()).cloned() {
                    let mut subst = BTreeMap::new();
                    if let Some(TypeExpr::Named { name: en, args: ea, .. }) = expected_ty {
                        if en == name {
                            for (tp, ta) in tmpl.type_params.iter().zip(ea.iter()) {
                                if ta.display_name() != "Any" {
                                    subst.insert(tp.clone(), ta.clone());
                                }
                            }
                        }
                    }
                    for (fname, fval) in fields.iter() {
                        if let Some(fd) = tmpl.fields.iter().find(|f| f.name == *fname) {
                            let aty = self.infer_expr_ty(fval, var_types);
                            self.unify_type_params(&fd.ty, &aty, &tmpl.type_params, &mut subst);
                        }
                    }
                    let concrete_args: Vec<TypeExpr> = tmpl
                        .type_params
                        .iter()
                        .map(|tp| subst.get(tp).cloned().unwrap_or_else(|| named_ty("Int")))
                        .collect();
                    for (fname, fval) in fields.iter_mut() {
                        let f_exp = tmpl
                            .fields
                            .iter()
                            .find(|f| f.name == *fname)
                            .map(|fd| subst_type(&fd.ty, &subst));
                        self.rewrite_expr_generic_records(fval, var_types, f_exp.as_ref());
                    }
                    *name = self.ensure_record_monomorphized(name, &concrete_args);
                } else {
                    let rec_opt = self.records.get(name.as_str()).cloned();
                    for (fname, fval) in fields.iter_mut() {
                        let f_exp = rec_opt
                            .as_ref()
                            .and_then(|r| r.fields.iter().find(|f| f.name == *fname))
                            .map(|fd| fd.ty.clone());
                        self.rewrite_expr_generic_records(fval, var_types, f_exp.as_ref());
                    }
                }
            }
            ExprKind::Binary { left, right, .. } => {
                self.rewrite_expr_generic_records(left, var_types, None);
                self.rewrite_expr_generic_records(right, var_types, None);
            }
            ExprKind::Unary { expr: inner, .. }
            | ExprKind::Await(inner)
            | ExprKind::Old(inner) => {
                self.rewrite_expr_generic_records(inner, var_types, None);
            }
            ExprKind::Ident(s) => {
                if let Some((ename, _)) = self.enum_variants.get(s.as_str()).cloned() {
                    if self.generic_enums.contains_key(&ename) {
                        if let Some(exp) = expected_ty {
                            let eb = self.resolve_alias(exp);
                            if let TypeExpr::Named { name: en, args: ea, .. } = &eb {
                                if en == &ename && !ea.is_empty() {
                                    let mono = self.ensure_enum_monomorphized(&ename, ea);
                                    *s = format!("{}__{}", mono, s);
                                } else if self.enums.contains_key(en) && en.starts_with(&format!("{}__", ename)) {
                                    *s = format!("{}__{}", en, s);
                                }
                            }
                        }
                    }
                }
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    if self.generic_enums.contains_key(ename)
                        && self.enum_variants.contains_key(&format!("{}.{}", ename, field))
                    {
                        if let Some(exp) = expected_ty {
                            let eb = self.resolve_alias(exp);
                            let mono_opt = match &eb {
                                TypeExpr::Named { name: en, args: ea, .. } if en == ename && !ea.is_empty() => {
                                    Some(self.ensure_enum_monomorphized(ename, ea))
                                }
                                TypeExpr::Named { name: en, .. }
                                    if self.enums.contains_key(en) && en.starts_with(&format!("{}__", ename)) =>
                                {
                                    Some(en.clone())
                                }
                                _ => None,
                            };
                            if let Some(mono) = mono_opt {
                                let vname = format!("{}__{}", mono, field);
                                expr.kind = ExprKind::Ident(vname);
                                return;
                            }
                        }
                    }
                }
                self.rewrite_expr_generic_records(target, var_types, None);
            }
            ExprKind::TupleIndex { target, .. } => {
                self.rewrite_expr_generic_records(target, var_types, None);
            }
            ExprKind::Index { target, index } => {
                self.rewrite_expr_generic_records(target, var_types, None);
                self.rewrite_expr_generic_records(index, var_types, None);
            }
            ExprKind::Call { callee, args } => {
                // Check if this is a generic enum constructor call (`Leaf(1)` or `Tree.Node(l, v, r)`)!
                let gen_enum_hit: Option<(String, String)> = match &callee.kind {
                    ExprKind::Ident(vname) => self
                        .enum_variants
                        .get(vname.as_str())
                        .and_then(|(en, _)| {
                            if self.generic_enums.contains_key(en) {
                                Some((en.clone(), vname.clone()))
                            } else {
                                None
                            }
                        }),
                    ExprKind::Field { target, field } => {
                        if let ExprKind::Ident(en) = &target.kind {
                            if self.generic_enums.contains_key(en)
                                && self.enum_variants.contains_key(&format!("{}.{}", en, field))
                            {
                                Some((en.clone(), field.clone()))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                if let Some((ename, vname)) = gen_enum_hit {
                    if let Some(tmpl) = self.generic_enums.get(&ename).cloned() {
                        let vdecl = tmpl.variants.iter().find(|v| v.name == vname).cloned();
                        let mut subst = BTreeMap::new();
                        if let Some(exp) = expected_ty {
                            let eb = self.resolve_alias(exp);
                            if let TypeExpr::Named { name: en, args: ea, .. } = &eb {
                                if en == &ename {
                                    for (tp, ta) in tmpl.type_params.iter().zip(ea.iter()) {
                                        if ta.display_name() != "Any" {
                                            subst.insert(tp.clone(), ta.clone());
                                        }
                                    }
                                }
                            }
                        }
                        if let Some(vd) = &vdecl {
                            for (fd, a) in vd.fields.iter().zip(args.iter()) {
                                let aty = self.infer_expr_ty(&a.value, var_types);
                                self.unify_type_params(&fd.ty, &aty, &tmpl.type_params, &mut subst);
                            }
                        }
                        let concrete_args: Vec<TypeExpr> = tmpl
                            .type_params
                            .iter()
                            .map(|tp| subst.get(tp).cloned().unwrap_or_else(|| named_ty("Int")))
                            .collect();
                        let mono_ename = self.ensure_enum_monomorphized(&ename, &concrete_args);
                        if let Some(vd) = &vdecl {
                            for (fd, a) in vd.fields.iter().zip(args.iter_mut()) {
                                let f_exp = subst_type(&fd.ty, &subst);
                                self.rewrite_expr_generic_records(&mut a.value, var_types, Some(&f_exp));
                            }
                        } else {
                            for a in args.iter_mut() {
                                self.rewrite_expr_generic_records(&mut a.value, var_types, None);
                            }
                        }
                        callee.kind = ExprKind::Ident(format!("{}__{}", mono_ename, vname));
                        return;
                    }
                }
                self.rewrite_expr_generic_records(callee, var_types, None);
                let callee_ptys: Vec<TypeExpr> = match &callee.kind {
                    ExprKind::Ident(fname) => self
                        .fn_sigs
                        .get(fname)
                        .map(|s| s.params.iter().map(|p| p.ty.clone()).collect())
                        .unwrap_or_default(),
                    _ => Vec::new(),
                };
                for (i, a) in args.iter_mut().enumerate() {
                    let pexp = callee_ptys.get(i);
                    self.rewrite_expr_generic_records(&mut a.value, var_types, pexp);
                }
            }
            ExprKind::Tuple(es) | ExprKind::List(es) => {
                for e in es {
                    self.rewrite_expr_generic_records(e, var_types, None);
                }
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                self.rewrite_expr_generic_records(cond, var_types, None);
                let mut tv = var_types.clone();
                self.rewrite_block_generic_records(then_block, &mut tv, expected_ty);
                if let Some(eb) = else_branch {
                    self.rewrite_expr_generic_records(eb, var_types, expected_ty);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.rewrite_expr_generic_records(scrutinee, var_types, None);
                let st = self.infer_expr_ty(scrutinee, var_types);
                for arm in arms {
                    for p in &mut arm.patterns {
                        self.rewrite_pattern_generic_enums(p, &st);
                    }
                    let mut av = var_types.clone();
                    self.bind_arm_vars(&st, arm, &mut av);
                    if let Some(g) = &mut arm.guard {
                        self.rewrite_expr_generic_records(g, &av, None);
                    }
                    self.rewrite_expr_generic_records(&mut arm.body, &av, expected_ty);
                }
            }
            ExprKind::Block(b) | ExprKind::Unsafe(b) => {
                let mut bv = var_types.clone();
                self.rewrite_block_generic_records(b, &mut bv, expected_ty);
            }
            ExprKind::Cast { expr, ty } | ExprKind::TryCast { expr, ty } => {
                self.rewrite_expr_generic_records(expr, var_types, None);
                *ty = self.rewrite_generic_record_ty(ty);
            }
            ExprKind::ArrayRepeat { value, .. } => {
                self.rewrite_expr_generic_records(value, var_types, None);
            }
            _ => {}
        }
    }

    fn box_call_args(
        &self,
        fname: &str,
        args: &mut [CallArg],
        var_types: &BTreeMap<String, TypeExpr>,
    ) {
        let Some(sig) = self.fn_sigs.get(fname) else {
            return;
        };
        for (p, a) in sig.params.iter().zip(args.iter_mut()) {
            let aty = self.infer_expr_ty(&a.value, var_types);
            let v = a.value.clone();
            a.value = self.maybe_box_dyn(v, &p.ty, &aty);
        }
    }

    fn is_ghost_call_expr(&self, expr: &Expr) -> bool {
        if let ExprKind::Call { callee, .. } = &expr.kind {
            match &callee.kind {
                ExprKind::Ident(fname) => return self.ghost_fns.contains(fname),
                ExprKind::GenericInst { expr: inner, .. } => {
                    if let ExprKind::Ident(fname) = &inner.kind {
                        return self.ghost_fns.contains(fname);
                    }
                }
                _ => {}
            }
        }
        false
    }

    fn collect_type_predicates(&self, ty: &TypeExpr) -> Vec<Expr> {
        let mut preds = Vec::new();
        let mut cur = ty.clone();
        let mut guard = 0usize;
        while guard < 16 {
            guard += 1;
            match cur {
                TypeExpr::Named { ref name, .. } => {
                    if let Some(p) = self.alias_preds.get(name) {
                        preds.push(p.clone());
                    }
                    if let Some(base) = self.alias_bases.get(name) {
                        cur = base.clone();
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        preds
    }

    fn is_refinement_cast_infallible(&self, src_ty: &TypeExpr, dst_ty: &TypeExpr, span: Span) -> bool {
        if self.infallible_casts.contains(&span.start) {
            return true;
        }
        matches!(
            tenon_verify::check_type_subsumption(&self.orig_program, src_ty, dst_ty),
            tenon_verify::CastVerdict::Infallible
        )
    }

    fn lower_fallible_refinement_check(
        &mut self,
        lowered_val: Expr,
        src_ty: &TypeExpr,
        dst_ty: &TypeExpr,
        var_types: &BTreeMap<String, TypeExpr>,
        ret_ty: Option<&TypeExpr>,
        sp: Span,
    ) -> Expr {
        let dst_base = self.resolve_alias(dst_ty);
        let cast_val = if self.resolve_alias(src_ty).display_name() != dst_base.display_name() {
            Expr {
                kind: ExprKind::Cast {
                    expr: Box::new(lowered_val),
                    ty: dst_base.clone(),
                },
                span: sp,
            }
        } else {
            lowered_val
        };
        let preds = self.collect_type_predicates(dst_ty);
        if preds.is_empty() {
            return Expr {
                kind: ExprKind::Call {
                    callee: Box::new(Expr {
                        kind: ExprKind::Ident("Ok".into()),
                        span: sp,
                    }),
                    args: vec![CallArg {
                        name: None,
                        value: cast_val,
                    }],
                },
                span: sp,
            };
        }
        let tmp_name = format!("__rc_v_{}", self.fresh_id());
        let mut combined: Option<Expr> = None;
        for p in &preds {
            let substituted = subst_self_in_expr(p, &tmp_name);
            combined = Some(match combined {
                None => substituted,
                Some(acc) => Expr {
                    kind: ExprKind::Binary {
                        op: BinOp::And,
                        left: Box::new(acc),
                        right: Box::new(substituted),
                    },
                    span: sp,
                },
            });
        }
        let mut local_vt = var_types.clone();
        local_vt.insert(tmp_name.clone(), dst_base.clone());
        let lowered_cond = self.lower_expr(&combined.unwrap(), &local_vt, ret_ty, None);
        Expr {
            kind: ExprKind::Block(Block {
                stmts: vec![Stmt::Let {
                    name: tmp_name.clone(),
                    ty: Some(dst_base),
                    value: cast_val,
                    is_mut: false,
                    span: sp,
                }],
                tail: Some(Box::new(Expr {
                    kind: ExprKind::If {
                        cond: Box::new(lowered_cond),
                        then_block: Block {
                            stmts: vec![],
                            tail: Some(Box::new(Expr {
                                kind: ExprKind::Call {
                                    callee: Box::new(Expr {
                                        kind: ExprKind::Ident("Ok".into()),
                                        span: sp,
                                    }),
                                    args: vec![CallArg {
                                        name: None,
                                        value: Expr {
                                            kind: ExprKind::Ident(tmp_name),
                                            span: sp,
                                        },
                                    }],
                                },
                                span: sp,
                            })),
                            span: sp,
                        },
                        else_branch: Some(Box::new(Expr {
                            kind: ExprKind::Block(Block {
                                stmts: vec![],
                                tail: Some(Box::new(Expr {
                                    kind: ExprKind::Call {
                                        callee: Box::new(Expr {
                                            kind: ExprKind::Ident("Err".into()),
                                            span: sp,
                                        }),
                                        args: vec![CallArg {
                                            name: None,
                                            value: Expr {
                                                kind: ExprKind::Str(format!(
                                                    "refinement check failed for `{}`",
                                                    dst_ty.display_name()
                                                )),
                                                span: sp,
                                            },
                                        }],
                                    },
                                    span: sp,
                                })),
                                span: sp,
                            }),
                            span: sp,
                        })),
                    },
                    span: sp,
                })),
                span: sp,
            }),
            span: sp,
        }
    }
}

fn subst_self_in_expr(expr: &Expr, var: &str) -> Expr {
    let kind = match &expr.kind {
        ExprKind::Ident(s) if s == "self" => ExprKind::Ident(var.to_string()),
        ExprKind::Binary { op, left, right } => ExprKind::Binary {
            op: *op,
            left: Box::new(subst_self_in_expr(left, var)),
            right: Box::new(subst_self_in_expr(right, var)),
        },
        ExprKind::Unary { op, expr: inner } => ExprKind::Unary {
            op: *op,
            expr: Box::new(subst_self_in_expr(inner, var)),
        },
        ExprKind::Call { callee, args } => ExprKind::Call {
            callee: Box::new(subst_self_in_expr(callee, var)),
            args: args
                .iter()
                .map(|a| CallArg {
                    name: a.name.clone(),
                    value: subst_self_in_expr(&a.value, var),
                })
                .collect(),
        },
        ExprKind::Field { target, field } => ExprKind::Field {
            target: Box::new(subst_self_in_expr(target, var)),
            field: field.clone(),
        },
        ExprKind::Index { target, index } => ExprKind::Index {
            target: Box::new(subst_self_in_expr(target, var)),
            index: Box::new(subst_self_in_expr(index, var)),
        },
        ExprKind::Cast { expr: inner, ty } => ExprKind::Cast {
            expr: Box::new(subst_self_in_expr(inner, var)),
            ty: ty.clone(),
        },
        other => other.clone(),
    };
    Expr {
        kind,
        span: expr.span,
    }
}

fn expr_uses_refinement_casts(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::TryCast { .. } => true,
        ExprKind::Call { callee, args } => {
            if let ExprKind::GenericInst { expr: inner, .. } = &callee.kind {
                if matches!(&inner.kind, ExprKind::Ident(n) if n == "refine_cast") {
                    return true;
                }
            }
            expr_uses_refinement_casts(callee)
                || args.iter().any(|a| expr_uses_refinement_casts(&a.value))
        }
        ExprKind::Binary { left, right, .. } | ExprKind::Range { start: left, end: right } => {
            expr_uses_refinement_casts(left) || expr_uses_refinement_casts(right)
        }
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Field { target: inner, .. }
        | ExprKind::TupleIndex { target: inner, .. }
        | ExprKind::Try(inner)
        | ExprKind::Await(inner)
        | ExprKind::Old(inner)
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::ArrayRepeat { value: inner, .. }
        | ExprKind::GenericInst { expr: inner, .. } => expr_uses_refinement_casts(inner),
        ExprKind::Index { target, index } => {
            expr_uses_refinement_casts(target) || expr_uses_refinement_casts(index)
        }
        ExprKind::RecordLit { fields, .. } => {
            fields.iter().any(|(_, v)| expr_uses_refinement_casts(v))
        }
        ExprKind::List(es) | ExprKind::Tuple(es) => es.iter().any(expr_uses_refinement_casts),
        ExprKind::If { cond, then_block, else_branch } => {
            expr_uses_refinement_casts(cond)
                || block_uses_refinement_casts(then_block)
                || else_branch.as_deref().is_some_and(expr_uses_refinement_casts)
        }
        ExprKind::Match { scrutinee, arms } => {
            expr_uses_refinement_casts(scrutinee)
                || arms.iter().any(|a| {
                    a.guard.as_ref().is_some_and(expr_uses_refinement_casts)
                        || expr_uses_refinement_casts(&a.body)
                })
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) => block_uses_refinement_casts(b),
        ExprKind::Lambda { body, .. } => expr_uses_refinement_casts(body),
        _ => false,
    }
}

fn block_uses_refinement_casts(b: &Block) -> bool {
    b.stmts.iter().any(|s| match s {
        Stmt::Let { value, .. } | Stmt::LetTuple { value, .. } => expr_uses_refinement_casts(value),
        Stmt::Assign { target, value, .. } => {
            expr_uses_refinement_casts(target) || expr_uses_refinement_casts(value)
        }
        Stmt::Return { value: Some(e), .. } | Stmt::Expr { expr: e, .. } => {
            expr_uses_refinement_casts(e)
        }
        Stmt::Assert { cond, .. } => expr_uses_refinement_casts(cond),
        Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } | Stmt::Ghost { body, .. } => {
            block_uses_refinement_casts(body)
        }
        Stmt::While { cond, body, .. } => {
            expr_uses_refinement_casts(cond) || block_uses_refinement_casts(body)
        }
        Stmt::For { iter, body, .. } => {
            expr_uses_refinement_casts(iter) || block_uses_refinement_casts(body)
        }
        _ => false,
    }) || b.tail.as_deref().is_some_and(expr_uses_refinement_casts)
}

fn program_uses_refinement_casts(p: &Program) -> bool {
    p.items.iter().any(|it| match it {
        Item::Fn(f) => block_uses_refinement_casts(&f.body),
        Item::Impl(im) => im.methods.iter().any(|m| block_uses_refinement_casts(&m.body)),
        _ => false,
    })
}

fn collect_pattern_names(pat: &Pattern, out: &mut BTreeSet<String>) {
    match pat {
        Pattern::Ident(s, _) if s != "_" => {
            out.insert(s.clone());
        }
        Pattern::Variant { binding: Some(b), .. } if b != "_" => {
            out.insert(b.clone());
        }
        Pattern::VariantMulti { patterns, .. } | Pattern::Tuple(patterns, _) => {
            for p in patterns {
                collect_pattern_names(p, out);
            }
        }
        _ => {}
    }
}

fn collect_free_vars_block(blk: &Block, bound: &BTreeSet<String>, free: &mut BTreeSet<String>) {
    let mut local_bound = bound.clone();
    for s in &blk.stmts {
        match s {
            Stmt::Let { name, value, .. } => {
                collect_free_vars_expr(value, &local_bound, free);
                if name != "_" {
                    local_bound.insert(name.clone());
                }
            }
            Stmt::LetTuple { names, value, .. } => {
                collect_free_vars_expr(value, &local_bound, free);
                for n in names {
                    if n != "_" {
                        local_bound.insert(n.clone());
                    }
                }
            }
            Stmt::Assign { target, value, .. } => {
                collect_free_vars_expr(target, &local_bound, free);
                collect_free_vars_expr(value, &local_bound, free);
            }
            Stmt::Return { value: Some(e), .. } | Stmt::Expr { expr: e, .. } => {
                collect_free_vars_expr(e, &local_bound, free);
            }
            Stmt::Assert { cond, .. } => {
                collect_free_vars_expr(cond, &local_bound, free);
            }
            Stmt::Mutate { target, body, .. } => {
                if !local_bound.contains(target) {
                    free.insert(target.clone());
                }
                collect_free_vars_block(body, &local_bound, free);
            }
            Stmt::While { cond, body, .. } => {
                collect_free_vars_expr(cond, &local_bound, free);
                collect_free_vars_block(body, &local_bound, free);
            }
            Stmt::Defer { body, .. } => {
                collect_free_vars_block(body, &local_bound, free);
            }
            Stmt::For { binding, iter, body, .. } => {
                collect_free_vars_expr(iter, &local_bound, free);
                let mut fbound = local_bound.clone();
                match binding {
                    ForBinding::Ident(v) if v != "_" => {
                        fbound.insert(v.clone());
                    }
                    ForBinding::Tuple(ns) => {
                        for n in ns {
                            if n != "_" {
                                fbound.insert(n.clone());
                            }
                        }
                    }
                    _ => {}
                }
                collect_free_vars_block(body, &fbound, free);
            }
            _ => {}
        }
    }
    if let Some(t) = &blk.tail {
        collect_free_vars_expr(t, &local_bound, free);
    }
}

fn collect_free_vars_expr(expr: &Expr, bound: &BTreeSet<String>, free: &mut BTreeSet<String>) {
    match &expr.kind {
        ExprKind::Ident(s) => {
            if !bound.contains(s) {
                free.insert(s.clone());
            }
        }
        ExprKind::Binary { left, right, .. } | ExprKind::Range { start: left, end: right } => {
            collect_free_vars_expr(left, bound, free);
            collect_free_vars_expr(right, bound, free);
        }
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Field { target: inner, .. }
        | ExprKind::TupleIndex { target: inner, .. }
        | ExprKind::Try(inner)
        | ExprKind::Await(inner)
        | ExprKind::Old(inner)
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::TryCast { expr: inner, .. }
        | ExprKind::ArrayRepeat { value: inner, .. }
        | ExprKind::GenericInst { expr: inner, .. } => {
            collect_free_vars_expr(inner, bound, free);
        }
        ExprKind::Index { target, index } => {
            collect_free_vars_expr(target, bound, free);
            collect_free_vars_expr(index, bound, free);
        }
        ExprKind::Call { callee, args } => {
            collect_free_vars_expr(callee, bound, free);
            for a in args {
                collect_free_vars_expr(&a.value, bound, free);
            }
        }
        ExprKind::RecordLit { fields, .. } => {
            for (_, v) in fields {
                collect_free_vars_expr(v, bound, free);
            }
        }
        ExprKind::List(es) | ExprKind::Tuple(es) => {
            for e in es {
                collect_free_vars_expr(e, bound, free);
            }
        }
        ExprKind::If { cond, then_block, else_branch } => {
            collect_free_vars_expr(cond, bound, free);
            collect_free_vars_block(then_block, bound, free);
            if let Some(eb) = else_branch {
                collect_free_vars_expr(eb, bound, free);
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            collect_free_vars_expr(scrutinee, bound, free);
            for arm in arms {
                let mut abound = bound.clone();
                for p in &arm.patterns {
                    collect_pattern_names(p, &mut abound);
                }
                if let Some(g) = &arm.guard {
                    collect_free_vars_expr(g, &abound, free);
                }
                collect_free_vars_expr(&arm.body, &abound, free);
            }
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) => collect_free_vars_block(b, bound, free),
        ExprKind::Lambda { params, body, .. } => {
            let mut lbound = bound.clone();
            for p in params {
                if p.name != "_" {
                    lbound.insert(p.name.clone());
                }
            }
            collect_free_vars_expr(body, &lbound, free);
        }
        _ => {}
    }
}

pub fn prepare_program_for_codegen(program: &Program) -> PreparedProgram {
    let mut expanded = program.clone();
    // Idempotent check: expand `@derive` if the caller used `parse(...)` directly instead of `load_program_with_imports`
    let existing_impl_keys: BTreeSet<(String, String)> = expanded
        .items
        .iter()
        .filter_map(|it| match it {
            Item::Impl(imp) => Some((
                imp.interface.as_ref().map(type_head_name).unwrap_or_default(),
                type_head_name(&imp.target_ty),
            )),
            _ => None,
        })
        .collect();
    let has_unexpanded_derive = expanded.items.iter().any(|it| match it {
        Item::Record(r) => r.attrs.iter().any(|a| a.name == "derive")
            && !existing_impl_keys.contains(&("ToJson".into(), r.name.clone()))
            && !existing_impl_keys.contains(&("FromJson".into(), r.name.clone()))
            && !existing_impl_keys.contains(&("JsonSchema".into(), r.name.clone()))
            && !existing_impl_keys.contains(&("Eq".into(), r.name.clone()))
            && !existing_impl_keys.contains(&("Hash".into(), r.name.clone())),
        Item::Enum(e) => e.attrs.iter().any(|a| a.name == "derive")
            && !existing_impl_keys.contains(&("ToJson".into(), e.name.clone()))
            && !existing_impl_keys.contains(&("FromJson".into(), e.name.clone())),
        _ => false,
    });
    if has_unexpanded_derive {
        expand_derives(&mut expanded);
    }

    let layout = LayoutEngine::from_program(&expanded);
    let comptime_ctx = ComptimeContext::from_program(&expanded);
    let infallible_casts = if program_uses_refinement_casts(&expanded) {
        tenon_verify::check_program(&expanded).infallible_casts
    } else {
        BTreeSet::new()
    };
    let mut ctx = LowerCtx {
        alias_bases: BTreeMap::new(),
        alias_preds: BTreeMap::new(),
        ghost_fns: BTreeSet::new(),
        infallible_casts,
        orig_program: expanded.clone(),
        records: BTreeMap::new(),
        generic_records: BTreeMap::new(),
        mono_records: Vec::new(),
        enums: BTreeMap::new(),
        generic_enums: BTreeMap::new(),
        mono_enums: Vec::new(),
        enum_names: BTreeSet::new(),
        enum_variants: BTreeMap::new(),
        unions: BTreeMap::new(),
        bitfields: BTreeMap::new(),
        extern_fns: BTreeMap::new(),
        consts: BTreeMap::new(),
        layout,
        interfaces: BTreeMap::new(),
        inherent_methods: BTreeMap::new(),
        impl_methods: BTreeMap::new(),
        vtable_syms: BTreeMap::new(),
        generic_templates: BTreeMap::new(),
        fn_sigs: BTreeMap::new(),
        mono_emitted: BTreeSet::new(),
        mono_queue: VecDeque::new(),
        lifted_closures: Vec::new(),
        next_tmp_id: 0,
        dyn_interfaces: BTreeMap::new(),
        defer_stack: Vec::new(),
        comptime_ctx,
    };

    // Pass 1: collect types, interfaces, unions, bitfields, externs, consts, and top-level fns
    for item in &expanded.items {
        match item {
            Item::TypeAlias(ta) => {
                ctx.alias_bases.insert(ta.name.clone(), ta.base.clone());
                if let Some(pred) = &ta.predicate {
                    ctx.alias_preds.insert(ta.name.clone(), pred.clone());
                }
            }
            Item::Record(r) => {
                if r.type_params.is_empty() {
                    ctx.records.insert(r.name.clone(), r.clone());
                } else {
                    ctx.generic_records.insert(r.name.clone(), r.clone());
                }
            }
            Item::Enum(e) => {
                if e.type_params.is_empty() {
                    ctx.enums.insert(e.name.clone(), e.clone());
                } else {
                    ctx.generic_enums.insert(e.name.clone(), e.clone());
                }
                ctx.enum_names.insert(e.name.clone());
                for v in &e.variants {
                    let ptys: Vec<TypeExpr> = v.fields.iter().map(|f| f.ty.clone()).collect();
                    ctx.enum_variants.insert(v.name.clone(), (e.name.clone(), ptys.clone()));
                    ctx.enum_variants
                        .insert(format!("{}.{}", e.name, v.name), (e.name.clone(), ptys));
                }
            }
            Item::Union(u) => {
                ctx.unions.insert(u.name.clone(), u.clone());
            }
            Item::Bitfield(bf) => {
                if let Some(bl) = ctx.layout.bitfield_layout(&bf.name) {
                    ctx.bitfields.insert(bf.name.clone(), bl);
                }
            }
            Item::ExternFn(ef) => {
                ctx.extern_fns.insert(ef.name.clone(), ef.clone());
            }
            Item::Const(c) => {
                ctx.consts.insert(c.name.clone(), (c.ty.clone(), c.value.clone()));
            }
            Item::Interface(iface) => {
                ctx.interfaces.insert(iface.name.clone(), iface.clone());
            }
            Item::Fn(f) => {
                if f.is_ghost {
                    ctx.ghost_fns.insert(f.name.clone());
                } else if f.type_params.is_empty() {
                    ctx.fn_sigs.insert(f.name.clone(), f.clone());
                } else {
                    ctx.generic_templates.insert(f.name.clone(), f.clone());
                }
            }
            _ => {}
        }
    }

    // Build canonical DynInterfaceLayout for every object-safe interface
    for (iname, idecl) in &ctx.interfaces {
        let object_safe = idecl
            .methods
            .iter()
            .all(|m| m.params.first().is_some_and(|p| p.name == "self") && m.type_params.is_empty());
        if object_safe {
            let mut dmethods = Vec::new();
            for (slot_idx, m) in idecl.methods.iter().enumerate() {
                let is_inout_self = m.params.first().is_some_and(|p| p.is_inout);
                let param_tys: Vec<TypeExpr> = m.params.iter().skip(1).map(|p| p.ty.clone()).collect();
                let returns_rc = m.ret_ty.as_ref().is_some_and(|t| ctx.type_is_rc(t));
                dmethods.push(DynMethodLayout {
                    name: m.name.clone(),
                    slot_idx,
                    is_inout_self,
                    param_tys,
                    ret_ty: m.ret_ty.clone(),
                    returns_rc,
                });
            }
            ctx.dyn_interfaces.insert(
                iname.clone(),
                DynInterfaceLayout {
                    name: iname.clone(),
                    methods: dmethods,
                },
            );
        }
    }

    // Pass 2: flatten `impl` blocks (inherent + interface impls + default methods) & build vtables!
    let mut flattened_fns: Vec<FnDecl> = Vec::new();
    let mut vtables: Vec<VtableSpec> = Vec::new();

    for item in &expanded.items {
        if let Item::Impl(imp) = item {
            let tname = type_head_name(&imp.target_ty);
            let mut self_subst = BTreeMap::new();
            self_subst.insert("Self".into(), imp.target_ty.clone());

            if let Some(iface_ty) = &imp.interface {
                let iname = type_head_name(iface_ty);
                let mut method_fn_map: BTreeMap<String, String> = BTreeMap::new();
                for m in &imp.methods {
                    let mangled = format!("__i__{}__{}__{}", tname, iname, m.name);
                    let mut f = subst_fn_types(m, &self_subst);
                    f.name = mangled.clone();
                    if f.type_params.is_empty() && imp.type_params.is_empty() {
                        ctx.fn_sigs.insert(mangled.clone(), f.clone());
                        flattened_fns.push(f);
                    } else {
                        let mut combined_tps = Vec::new();
                        for tp in &imp.type_params {
                            if !combined_tps.contains(tp) {
                                combined_tps.push(tp.clone());
                            }
                        }
                        for tp in &f.type_params {
                            if !combined_tps.contains(tp) {
                                combined_tps.push(tp.clone());
                            }
                        }
                        f.type_params = combined_tps;
                        ctx.generic_templates.insert(mangled.clone(), f);
                    }
                    method_fn_map.insert(m.name.clone(), mangled.clone());
                    ctx.impl_methods
                        .entry((tname.clone(), m.name.clone()))
                        .or_default()
                        .push((iname.clone(), mangled));
                }
                // Synthesize any default methods from `interface` not overridden in this `impl`
                if let Some(idecl) = ctx.interfaces.get(&iname).cloned() {
                    for im in &idecl.methods {
                        if !method_fn_map.contains_key(&im.name) {
                            if let Some(body) = &im.default_body {
                                let mangled = format!("__i__{}__{}__{}", tname, iname, im.name);
                                let raw = FnDecl {
                                    attrs: im.attrs.clone(),
                                    is_pub: true,
                                    is_pure: im.is_pure,
                                    is_async: im.is_async,
                                    is_unsafe: false,
                                    is_comptime: false,
                                    is_ghost: false,
                                    is_test: false,
                                    name: mangled.clone(),
                                    type_params: imp.type_params.clone(),
                                    type_param_bounds: imp.type_param_bounds.clone(),
                                    params: im.params.clone(),
                                    ret_ty: im.ret_ty.clone(),
                                    requires: im.requires.clone(),
                                    ensures: im.ensures.clone(),
                                    body: body.clone(),
                                    span: im.span,
                                };
                                let f = subst_fn_types(&raw, &self_subst);
                                ctx.fn_sigs.insert(mangled.clone(), f.clone());
                                flattened_fns.push(f);
                                method_fn_map.insert(im.name.clone(), mangled.clone());
                                ctx.impl_methods
                                    .entry((tname.clone(), im.name.clone()))
                                    .or_default()
                                    .push((iname.clone(), mangled));
                            }
                        }
                    }
                    // Build vtable if interface is object-safe and impl is non-generic
                    if imp.type_params.is_empty() && ctx.dyn_interfaces.contains_key(&iname) {
                        let mut fn_names = Vec::new();
                        let mut complete = true;
                        for im in &idecl.methods {
                            if let Some(fn_name) = method_fn_map.get(&im.name) {
                                fn_names.push(fn_name.clone());
                            } else {
                                complete = false;
                            }
                        }
                        if complete {
                            let sym = format!("__vtable__{}__{}", tname, iname);
                            ctx.vtable_syms
                                .insert((iname.clone(), tname.clone()), sym.clone());
                            vtables.push(VtableSpec {
                                sym_name: sym,
                                interface: iname,
                                target_type: tname,
                                fn_names,
                            });
                        }
                    }
                }
            } else {
                // Inherent `impl Target { ... }`
                for m in &imp.methods {
                    let mangled = format!("__m__{}__{}", tname, m.name);
                    let mut f = subst_fn_types(m, &self_subst);
                    f.name = mangled.clone();
                    if f.type_params.is_empty() && imp.type_params.is_empty() {
                        ctx.fn_sigs.insert(mangled.clone(), f.clone());
                        flattened_fns.push(f);
                    } else {
                        let mut combined_tps = Vec::new();
                        for tp in &imp.type_params {
                            if !combined_tps.contains(tp) {
                                combined_tps.push(tp.clone());
                            }
                        }
                        for tp in &f.type_params {
                            if !combined_tps.contains(tp) {
                                combined_tps.push(tp.clone());
                            }
                        }
                        f.type_params = combined_tps;
                        ctx.generic_templates.insert(mangled.clone(), f);
                    }
                    ctx.inherent_methods
                        .insert((tname.clone(), m.name.clone()), mangled);
                }
            }
        }
    }

    // Pass 3: lower all concrete functions, drain monomorphization worklist & collect lifted closures!
    let mut final_items: Vec<Item> = Vec::new();
    for item in &expanded.items {
        match item {
            Item::Fn(f) => {
                if !f.is_ghost && f.type_params.is_empty() {
                    final_items.push(Item::Fn(ctx.lower_fn(f)));
                }
            }
            Item::Impl(_) | Item::Bitfield(_) | Item::Const(_) => {} // lowered/inlined
            other => final_items.push(other.clone()),
        }
    }
    for f in flattened_fns {
        final_items.push(Item::Fn(ctx.lower_fn(&f)));
    }
    while let Some(mono_f) = ctx.mono_queue.pop_front() {
        final_items.push(Item::Fn(ctx.lower_fn(&mono_f)));
    }
    for cf in ctx.lifted_closures.drain(..) {
        final_items.push(Item::Fn(cf));
    }

    // Pass 4: Monomorphize all concrete generic record & generic enum types, and lower bitfield types
    // (e.g. `Map[Text, Text]` -> `Map__Text_Text`, `Tree[Int]` -> `Tree__Int`, `Flags` -> `UInt8`)
    if !ctx.generic_records.is_empty()
        || !ctx.generic_enums.is_empty()
        || !ctx.bitfields.is_empty()
    {
        let mut rewritten_items: Vec<Item> = Vec::with_capacity(final_items.len());
        for mut item in final_items {
            match &mut item {
                Item::Record(r) => {
                    if !r.type_params.is_empty() {
                        continue;
                    }
                    for f in &mut r.fields {
                        f.ty = ctx.rewrite_generic_record_ty(&f.ty);
                    }
                }
                Item::Union(u) => {
                    for f in &mut u.fields {
                        f.ty = ctx.rewrite_generic_record_ty(&f.ty);
                    }
                }
                Item::ExternFn(ef) => {
                    for p in &mut ef.params {
                        p.ty = ctx.rewrite_generic_record_ty(&p.ty);
                    }
                    if let Some(rt) = &mut ef.ret_ty {
                        *rt = ctx.rewrite_generic_record_ty(rt);
                    }
                }
                Item::Enum(e) => {
                    if !e.type_params.is_empty() {
                        continue;
                    }
                    for v in &mut e.variants {
                        for f in &mut v.fields {
                            f.ty = ctx.rewrite_generic_record_ty(&f.ty);
                        }
                    }
                }
                Item::TypeAlias(ta) => {
                    ta.base = ctx.rewrite_generic_record_ty(&ta.base);
                }
                Item::Fn(f) => {
                    let mut vt = BTreeMap::new();
                    for p in &f.params {
                        vt.insert(p.name.clone(), p.ty.clone());
                    }
                    let orig_ret = f.ret_ty.clone();
                    ctx.rewrite_block_generic_records(&mut f.body, &mut vt, orig_ret.as_ref());
                    for p in &mut f.params {
                        p.ty = ctx.rewrite_generic_record_ty(&p.ty);
                    }
                    if let Some(rt) = &mut f.ret_ty {
                        *rt = ctx.rewrite_generic_record_ty(rt);
                    }
                }
                _ => {}
            }
            rewritten_items.push(item);
        }
        let mut out_items: Vec<Item> = Vec::with_capacity(
            rewritten_items.len() + ctx.mono_records.len() + ctx.mono_enums.len(),
        );
        for mr in ctx.mono_records.drain(..) {
            out_items.push(Item::Record(mr));
        }
        for me in ctx.mono_enums.drain(..) {
            out_items.push(Item::Enum(me));
        }
        out_items.extend(rewritten_items);
        final_items = out_items;
    }

    PreparedProgram {
        program: Program {
            items: final_items,
            source_files: expanded.source_files,
            file_packages: expanded.file_packages,
            test_file_ids: expanded.test_file_ids,
            build_config: expanded.build_config,
            comptime_items: expanded.comptime_items,
        },
        vtables,
        dyn_interfaces: ctx.dyn_interfaces,
    }
}
