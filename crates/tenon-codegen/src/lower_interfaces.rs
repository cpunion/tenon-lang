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
    records: BTreeMap<String, RecordDecl>,
    generic_records: BTreeMap<String, RecordDecl>,
    mono_records: Vec<RecordDecl>,
    enum_names: BTreeSet<String>,
    enum_variants: BTreeMap<String, (String, Vec<TypeExpr>)>,
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
    /// Interface dynamic layouts
    dyn_interfaces: BTreeMap<String, DynInterfaceLayout>,
}

impl LowerCtx {
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

    fn type_is_rc(&self, ty: &TypeExpr) -> bool {
        let resolved = self.resolve_alias(ty);
        match resolved {
            TypeExpr::Dyn { .. } => true,
            TypeExpr::Named { name, .. } => {
                matches!(name.as_str(), "Text" | "List" | "Option" | "Result")
                    || self.records.contains_key(&name)
                    || self.generic_records.contains_key(&name)
                    || self.enum_names.contains(&name)
            }
            TypeExpr::Tuple(_, _) => true,
        }
    }

    fn is_known_type(&self, name: &str) -> bool {
        self.records.contains_key(name)
            || self.generic_records.contains_key(name)
            || self.enum_names.contains(name)
            || self.alias_bases.contains_key(name)
            || matches!(
                name,
                "Int" | "Int64" | "Int32" | "UInt16" | "USize" | "Bool" | "Text" | "Option" | "Result" | "List"
            )
    }

    fn infer_expr_ty(&self, expr: &Expr, var_types: &BTreeMap<String, TypeExpr>) -> TypeExpr {
        match &expr.kind {
            ExprKind::Int(_) => named_ty("Int"),
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
            ExprKind::Binary { op: BinOp::Concat, .. } => named_ty("Text"),
            ExprKind::Binary {
                op: BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::And | BinOp::Or,
                ..
            } => named_ty("Bool"),
            ExprKind::Binary { .. } => named_ty("Int"),
            ExprKind::Unary { op: UnOp::Not, .. } => named_ty("Bool"),
            ExprKind::Unary { op: UnOp::Neg, .. } => named_ty("Int"),
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    if let Some((en, _)) = self.enum_variants.get(&format!("{}.{}", ename, field)) {
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
                }
                named_ty("Any")
            }
            ExprKind::Index { target, .. } => {
                let t = self.resolve_alias(&self.infer_expr_ty(target, var_types));
                if let TypeExpr::Named { name, args, .. } = t {
                    if name == "List" && !args.is_empty() {
                        return args[0].clone();
                    }
                }
                named_ty("Any")
            }
            ExprKind::Call { callee, args } => match &callee.kind {
                ExprKind::Ident(fname) => {
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
                    if let Some((ename, _)) = self.enum_variants.get(fname) {
                        return named_ty(ename);
                    }
                    if matches!(
                        fname.as_str(),
                        "str_slice" | "byte_to_text" | "str_concat" | "buf_finish"
                            | "sys_read_line" | "sys_env" | "sys_cwd" | "sys_read_file"
                            | "sys_exec" | "sys_http_sse_next_line" | "sys_http_sse_error"
                    ) {
                        return named_ty("Text");
                    }
                    if matches!(
                        fname.as_str(),
                        "sys_is_eof" | "sys_file_ok" | "sys_write_file" | "sys_http_sse_eof"
                    ) {
                        return named_ty("Bool");
                    }
                    named_ty("Int")
                }
                ExprKind::GenericInst { expr: inner, type_args } => {
                    if let ExprKind::Ident(fname) = &inner.kind {
                        if let Some(tmpl) = self.generic_templates.get(fname) {
                            if let Some(rt) = &tmpl.ret_ty {
                                let subst = self.infer_generic_subst(tmpl, type_args, args, var_types);
                                return subst_type(rt, &subst);
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
                        if let Some((en, _)) = self.enum_variants.get(&format!("{}.{}", ename, field)) {
                            return named_ty(en);
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
                                        let mut subst = BTreeMap::new();
                                        for (tp, ra) in tmpl.type_params.iter().zip(rargs.iter()) {
                                            if ra.display_name() != "Any" {
                                                subst.insert(tp.clone(), ra.clone());
                                            }
                                        }
                                        return subst_type(rt, &subst);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                    named_ty("Any")
                }
                _ => named_ty("Any"),
            },
            ExprKind::If { then_block, else_branch, .. } => {
                let mut local = var_types.clone();
                let t = self.infer_block_ty(then_block, &mut local);
                if t.display_name() != "Any" {
                    return t;
                }
                if let Some(eb) = else_branch {
                    return self.infer_expr_ty(eb, var_types);
                }
                named_ty("Any")
            }
            ExprKind::Match { scrutinee, arms } => {
                let st = self.infer_expr_ty(scrutinee, var_types);
                for arm in arms {
                    let mut av = var_types.clone();
                    self.bind_arm_vars(&st, arm, &mut av);
                    let t = self.infer_expr_ty(&arm.body, &av);
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
            _ => named_ty("Any"),
        }
    }

    fn infer_block_ty(&self, blk: &Block, var_types: &mut BTreeMap<String, TypeExpr>) -> TypeExpr {
        for s in &blk.stmts {
            if let Stmt::Let { name, ty, value, .. } = s {
                let t = ty
                    .clone()
                    .unwrap_or_else(|| self.infer_expr_ty(value, var_types));
                var_types.insert(name.clone(), t);
            }
        }
        blk.tail
            .as_deref()
            .map(|t| self.infer_expr_ty(t, var_types))
            .unwrap_or_else(|| named_ty("Any"))
    }

    fn bind_arm_vars(&self, scrutinee_ty: &TypeExpr, arm: &MatchArm, vars: &mut BTreeMap<String, TypeExpr>) {
        let base = self.resolve_alias(scrutinee_ty);
        for p in &arm.patterns {
            if let Pattern::Variant { name, binding: Some(bn), .. } = p {
                if bn == "_" {
                    continue;
                }
                let bty = if name == "Some" {
                    if let TypeExpr::Named { name: tn, args, .. } = &base {
                        if tn == "Option" && !args.is_empty() {
                            args[0].clone()
                        } else {
                            named_ty("Any")
                        }
                    } else {
                        named_ty("Any")
                    }
                } else if name == "Ok" {
                    if let TypeExpr::Named { name: tn, args, .. } = &base {
                        if tn == "Result" && !args.is_empty() {
                            args[0].clone()
                        } else {
                            named_ty("Any")
                        }
                    } else {
                        named_ty("Any")
                    }
                } else if name == "Err" {
                    if let TypeExpr::Named { name: tn, args, .. } = &base {
                        if tn == "Result" && args.len() >= 2 {
                            args[1].clone()
                        } else {
                            named_ty("Any")
                        }
                    } else {
                        named_ty("Any")
                    }
                } else if let Some((_, ptys)) = self.enum_variants.get(name) {
                    ptys.first().cloned().unwrap_or_else(|| named_ty("Any"))
                } else {
                    named_ty("Any")
                };
                vars.insert(bn.clone(), bty);
            }
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
            subst.insert(tp.clone(), ta.clone());
        }
        for (p, a) in tmpl.params.iter().zip(args.iter()) {
            let aty = self.infer_expr_ty(&a.value, var_types);
            self.unify_type_params(&p.ty, &aty, &tmpl.type_params, &mut subst);
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
        let mut stmts = Vec::with_capacity(blk.stmts.len());
        for s in &blk.stmts {
            match s {
                Stmt::Let {
                    name,
                    ty,
                    value,
                    is_mut,
                    span,
                } => {
                    let mut lv = self.lower_expr(value, var_types, ret_ty, ty.as_ref());
                    let inferred = self.infer_expr_ty(&lv, var_types);
                    let eff_ty = if let Some(t) = ty {
                        lv = self.maybe_box_dyn(lv, t, &inferred);
                        t.clone()
                    } else {
                        inferred
                    };
                    var_types.insert(name.clone(), eff_ty);
                    stmts.push(Stmt::Let {
                        name: name.clone(),
                        ty: ty.clone(),
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
                    stmts.push(Stmt::Return {
                        value: lv,
                        span: *span,
                    });
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
                Stmt::Expr { expr, span } => stmts.push(Stmt::Expr {
                    expr: self.lower_expr(expr, var_types, ret_ty, None),
                    span: *span,
                }),
                other => stmts.push(other.clone()),
            }
        }
        let tail = blk.tail.as_ref().map(|t| {
            let lt = self.lower_expr(t, var_types, ret_ty, ret_ty);
            let boxed = if let Some(rt) = ret_ty {
                let aty = self.infer_expr_ty(&lt, var_types);
                self.maybe_box_dyn(lt, rt, &aty)
            } else {
                lt
            };
            Box::new(boxed)
        });
        *var_types = saved;
        Block {
            stmts,
            tail,
            span: blk.span,
        }
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
            ExprKind::Binary { op, left, right } => ExprKind::Binary {
                op: *op,
                left: Box::new(self.lower_expr(left, var_types, ret_ty, None)),
                right: Box::new(self.lower_expr(right, var_types, ret_ty, None)),
            },
            ExprKind::Unary { op, expr: inner } => ExprKind::Unary {
                op: *op,
                expr: Box::new(self.lower_expr(inner, var_types, ret_ty, None)),
            },
            ExprKind::Field { target, field } => ExprKind::Field {
                target: Box::new(self.lower_expr(target, var_types, ret_ty, None)),
                field: field.clone(),
            },
            ExprKind::Index { target, index } => ExprKind::Index {
                target: Box::new(self.lower_expr(target, var_types, ret_ty, None)),
                index: Box::new(self.lower_expr(index, var_types, ret_ty, None)),
            },
            ExprKind::RecordLit {
                name,
                version,
                fields,
            } => {
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
                let new_fields = fields
                    .iter()
                    .map(|(k, v)| {
                        let f_expected = rec_opt.as_ref().and_then(|rec| {
                            rec.fields.iter().find(|f| f.name == *k).map(|fd| {
                                if rec_subst.is_empty() {
                                    fd.ty.clone()
                                } else {
                                    subst_type(&fd.ty, &rec_subst)
                                }
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
                let mut lowered_args: Vec<CallArg> = args
                    .iter()
                    .map(|a| CallArg {
                        name: a.name.clone(),
                        value: self.lower_expr(&a.value, var_types, ret_ty, None),
                    })
                    .collect();

                match &callee.kind {
                    // 1. Explicit generic call `decode[Order](raw)`
                    ExprKind::GenericInst { expr: inner, type_args } => {
                        if let ExprKind::Ident(fname) = &inner.kind {
                            if let Some(tmpl) = self.generic_templates.get(fname).cloned() {
                                let subst = self.infer_generic_subst(&tmpl, type_args, &lowered_args, var_types);
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
                        }
                        ExprKind::Call {
                            callee: callee.clone(),
                            args: lowered_args,
                        }
                    }
                    // 2. Direct function call (possibly implicit generic or `push` into `List[dyn I]`)
                    ExprKind::Ident(fname) => {
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
                        if let Some(tmpl) = self.generic_templates.get(fname).cloned() {
                            let subst = self.infer_generic_subst(&tmpl, &[], &lowered_args, var_types);
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
                        self.box_call_args(fname, &mut lowered_args, var_types);
                        ExprKind::Call {
                            callee: callee.clone(),
                            args: lowered_args,
                        }
                    }
                    // 3. Associated call `Type.method(args)` OR receiver method call `x.method(args)`!
                    ExprKind::Field { target, field } => {
                        // Preserve auto-derived `Record.parse(...)` (unless explicit `impl Record::parse` exists) and `Enum.Variant(...)`
                        if field == "parse" {
                            if let ExprKind::Ident(rname) = &target.kind {
                                if self.records.contains_key(rname)
                                    && self.lookup_concrete_method(rname, "parse").is_none()
                                {
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
                                        let final_fn = if let Some(tmpl) = self.generic_templates.get(&mname).cloned() {
                                            let mut subst = self.infer_generic_subst(&tmpl, type_args, &lowered_args, var_types);
                                            if let (Some(rt), Some(exp)) = (&tmpl.ret_ty, expected_ty) {
                                                self.unify_type_params(rt, exp, &tmpl.type_params, &mut subst);
                                            }
                                            self.ensure_monomorphized(&mname, &subst).unwrap_or(mname)
                                        } else {
                                            mname
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
                            if self.enum_variants.contains_key(&format!("{}.{}", ename, field)) {
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
                                    let final_fn = if let Some(tmpl) = self.generic_templates.get(&mname).cloned() {
                                        let mut subst = self.infer_generic_subst(&tmpl, &[], &lowered_args, var_types);
                                        if let (Some(rt), Some(exp)) = (&tmpl.ret_ty, expected_ty) {
                                            self.unify_type_params(rt, exp, &tmpl.type_params, &mut subst);
                                        }
                                        self.ensure_monomorphized(&mname, &subst).unwrap_or(mname)
                                    } else {
                                        mname
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
                                // Keep as dynamic dispatch call `d.method(args)`! Box any `dyn` args if needed.
                                if let Some(layout) = self.dyn_interfaces.get(interface).cloned() {
                                    if let Some(dm) = layout.methods.iter().find(|m| m.name == *field) {
                                        for (pty, a) in dm.param_tys.iter().zip(lowered_args.iter_mut()) {
                                            let aty = self.infer_expr_ty(&a.value, var_types);
                                            let v = a.value.clone();
                                            a.value = self.maybe_box_dyn(v, pty, &aty);
                                        }
                                    }
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
                                    let mut full_args = Vec::with_capacity(lowered_args.len() + 1);
                                    full_args.push(CallArg {
                                        name: None,
                                        value: lowered_target,
                                    });
                                    full_args.extend(lowered_args);
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
                                    self.box_call_args(&final_fn, &mut full_args, var_types);
                                    ExprKind::Call {
                                        callee: Box::new(Expr {
                                            kind: ExprKind::Ident(final_fn),
                                            span: callee.span,
                                        }),
                                        args: full_args,
                                    }
                                } else {
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
                            _ => ExprKind::Call {
                                callee: Box::new(Expr {
                                    kind: ExprKind::Field {
                                        target: Box::new(lowered_target),
                                        field: field.clone(),
                                    },
                                    span: callee.span,
                                }),
                                args: lowered_args,
                            },
                        }
                    }
                    _ => ExprKind::Call {
                        callee: callee.clone(),
                        args: lowered_args,
                    },
                }
            }
            other => other.clone(),
        };
        Expr { kind, span: sp }
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
                if self.generic_records.contains_key(name) && !args.is_empty() {
                    let mono_name = self.ensure_record_monomorphized(name, args);
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
            ExprKind::Field { target, .. } => {
                self.rewrite_expr_generic_records(target, var_types, None);
            }
            ExprKind::Index { target, index } => {
                self.rewrite_expr_generic_records(target, var_types, None);
                self.rewrite_expr_generic_records(index, var_types, None);
            }
            ExprKind::Call { callee, args } => {
                self.rewrite_expr_generic_records(callee, var_types, None);
                for a in args {
                    self.rewrite_expr_generic_records(&mut a.value, var_types, None);
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
                    let mut av = var_types.clone();
                    self.bind_arm_vars(&st, arm, &mut av);
                    if let Some(g) = &mut arm.guard {
                        self.rewrite_expr_generic_records(g, &av, None);
                    }
                    self.rewrite_expr_generic_records(&mut arm.body, &av, expected_ty);
                }
            }
            ExprKind::Block(b) => {
                let mut bv = var_types.clone();
                self.rewrite_block_generic_records(b, &mut bv, expected_ty);
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
            && !existing_impl_keys.contains(&("JsonSchema".into(), r.name.clone())),
        Item::Enum(e) => e.attrs.iter().any(|a| a.name == "derive")
            && !existing_impl_keys.contains(&("ToJson".into(), e.name.clone()))
            && !existing_impl_keys.contains(&("FromJson".into(), e.name.clone())),
        _ => false,
    });
    if has_unexpanded_derive {
        expand_derives(&mut expanded);
    }

    let mut ctx = LowerCtx {
        alias_bases: BTreeMap::new(),
        records: BTreeMap::new(),
        generic_records: BTreeMap::new(),
        mono_records: Vec::new(),
        enum_names: BTreeSet::new(),
        enum_variants: BTreeMap::new(),
        interfaces: BTreeMap::new(),
        inherent_methods: BTreeMap::new(),
        impl_methods: BTreeMap::new(),
        vtable_syms: BTreeMap::new(),
        generic_templates: BTreeMap::new(),
        fn_sigs: BTreeMap::new(),
        mono_emitted: BTreeSet::new(),
        mono_queue: VecDeque::new(),
        dyn_interfaces: BTreeMap::new(),
    };

    // Pass 1: collect types, interfaces, and top-level fns
    for item in &expanded.items {
        match item {
            Item::TypeAlias(ta) => {
                ctx.alias_bases.insert(ta.name.clone(), ta.base.clone());
            }
            Item::Record(r) => {
                if r.type_params.is_empty() {
                    ctx.records.insert(r.name.clone(), r.clone());
                } else {
                    ctx.generic_records.insert(r.name.clone(), r.clone());
                }
            }
            Item::Enum(e) => {
                ctx.enum_names.insert(e.name.clone());
                for v in &e.variants {
                    let ptys: Vec<TypeExpr> = v.fields.iter().map(|f| f.ty.clone()).collect();
                    ctx.enum_variants.insert(v.name.clone(), (e.name.clone(), ptys.clone()));
                    ctx.enum_variants
                        .insert(format!("{}.{}", e.name, v.name), (e.name.clone(), ptys));
                }
            }
            Item::Interface(iface) => {
                ctx.interfaces.insert(iface.name.clone(), iface.clone());
            }
            Item::Fn(f) => {
                if f.type_params.is_empty() {
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
                        for tp in &imp.type_params {
                            if !f.type_params.contains(tp) {
                                f.type_params.push(tp.clone());
                            }
                        }
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
                        for tp in &imp.type_params {
                            if !f.type_params.contains(tp) {
                                f.type_params.push(tp.clone());
                            }
                        }
                        ctx.generic_templates.insert(mangled.clone(), f);
                    }
                    ctx.inherent_methods
                        .insert((tname.clone(), m.name.clone()), mangled);
                }
            }
        }
    }

    // Pass 3: lower all concrete functions & drain monomorphization worklist!
    let mut final_items: Vec<Item> = Vec::new();
    for item in &expanded.items {
        match item {
            Item::Fn(f) => {
                if f.type_params.is_empty() {
                    final_items.push(Item::Fn(ctx.lower_fn(f)));
                }
            }
            Item::Impl(_) => {} // flattened above
            other => final_items.push(other.clone()),
        }
    }
    for f in flattened_fns {
        final_items.push(Item::Fn(ctx.lower_fn(&f)));
    }
    while let Some(mono_f) = ctx.mono_queue.pop_front() {
        final_items.push(Item::Fn(ctx.lower_fn(&mono_f)));
    }

    // Pass 4: Monomorphize all concrete generic record types (e.g. `Map[Text, Text]` -> `Map__Text_Text`)
    // and rewrite generic `RecordLit` / `TypeExpr` references so Cranelift JIT & LLVM AOT see concrete layouts!
    if !ctx.generic_records.is_empty() {
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
                Item::Enum(e) => {
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
        let mut out_items: Vec<Item> = Vec::with_capacity(rewritten_items.len() + ctx.mono_records.len());
        for mr in ctx.mono_records.drain(..) {
            out_items.push(Item::Record(mr));
        }
        out_items.extend(rewritten_items);
        final_items = out_items;
    }

    PreparedProgram {
        program: Program { items: final_items },
        vtables,
        dyn_interfaces: ctx.dyn_interfaces,
    }
}
