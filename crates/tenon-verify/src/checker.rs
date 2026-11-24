use crate::l0::{AbsInt, Interval, L0Env, L0Result};
use crate::smt::{SmtOutcome, SmtQuery};
use crate::sync_solver::{DistributedAppManifest, solve_distributed_manifest};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use tenon_syntax::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofTier {
    L0Fast,
    L1Smt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofRecord {
    pub obligation: String,
    pub tier: ProofTier,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: String,
    pub message: String,
    pub span: Span,
    pub notes: Vec<String>,
    pub counterexample: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckReport {
    pub ok: bool,
    pub proofs: Vec<ProofRecord>,
    pub diagnostics: Vec<Diagnostic>,
    pub sql_checks: BTreeMap<String, String>,
    pub manifest: Option<DistributedAppManifest>,
}

struct BuiltinSig {
    name: &'static str,
    arity: usize,
    is_pure: bool,
    allocates: bool,
    param_types: &'static [&'static str],
    ret_type: &'static str,
}

const BUILTIN_SIGS: &[BuiltinSig] = &[
    // Core builtins
    BuiltinSig { name: "print_int", arity: 1, is_pure: false, allocates: false, param_types: &["Int"], ret_type: "Int" },
    BuiltinSig { name: "len", arity: 1, is_pure: true, allocates: false, param_types: &["Any"], ret_type: "Int" },
    BuiltinSig { name: "push", arity: 2, is_pure: true, allocates: true, param_types: &["Any", "Any"], ret_type: "List[Any]" },
    BuiltinSig { name: "yield_val", arity: 1, is_pure: false, allocates: true, param_types: &["Any"], ret_type: "Any" },
    BuiltinSig { name: "block_on", arity: 1, is_pure: false, allocates: true, param_types: &["Any"], ret_type: "Any" },
    // Polymorphic Option / Result constructors
    BuiltinSig { name: "Ok", arity: 1, is_pure: true, allocates: true, param_types: &["Any"], ret_type: "Result[Any, Any]" },
    BuiltinSig { name: "Err", arity: 1, is_pure: true, allocates: true, param_types: &["Any"], ret_type: "Result[Any, Any]" },
    BuiltinSig { name: "Some", arity: 1, is_pure: true, allocates: true, param_types: &["Any"], ret_type: "Option[Any]" },
    // 4 Minimal core byte / text primitives (pure)
    BuiltinSig { name: "byte_at", arity: 2, is_pure: true, allocates: false, param_types: &["Text", "Int"], ret_type: "Int" },
    BuiltinSig { name: "str_slice", arity: 3, is_pure: true, allocates: true, param_types: &["Text", "Int", "Int"], ret_type: "Text" },
    BuiltinSig { name: "byte_to_text", arity: 1, is_pure: true, allocates: true, param_types: &["Int"], ret_type: "Text" },
    BuiltinSig { name: "str_concat", arity: 2, is_pure: true, allocates: true, param_types: &["Text", "Text"], ret_type: "Text" },
    // 4 O(n) Byte buffer primitives (pure)
    BuiltinSig { name: "buf_new", arity: 0, is_pure: true, allocates: true, param_types: &[], ret_type: "Int" },
    BuiltinSig { name: "buf_push_str", arity: 2, is_pure: true, allocates: true, param_types: &["Int", "Text"], ret_type: "Int" },
    BuiltinSig { name: "buf_push_byte", arity: 2, is_pure: true, allocates: true, param_types: &["Int", "Int"], ret_type: "Int" },
    BuiltinSig { name: "buf_finish", arity: 1, is_pure: true, allocates: true, param_types: &["Int"], ret_type: "Text" },
    // 10 Minimal OS / I/O / FS / Process syscalls (impure)
    BuiltinSig { name: "sys_write_stdout", arity: 1, is_pure: false, allocates: false, param_types: &["Text"], ret_type: "Int" },
    BuiltinSig { name: "sys_read_line", arity: 0, is_pure: false, allocates: true, param_types: &[], ret_type: "Text" },
    BuiltinSig { name: "sys_is_eof", arity: 0, is_pure: false, allocates: false, param_types: &[], ret_type: "Bool" },
    BuiltinSig { name: "sys_env", arity: 1, is_pure: false, allocates: true, param_types: &["Text"], ret_type: "Text" },
    BuiltinSig { name: "sys_cwd", arity: 0, is_pure: false, allocates: true, param_types: &[], ret_type: "Text" },
    BuiltinSig { name: "sys_read_file", arity: 1, is_pure: false, allocates: true, param_types: &["Text"], ret_type: "Text" },
    BuiltinSig { name: "sys_file_ok", arity: 0, is_pure: false, allocates: false, param_types: &[], ret_type: "Bool" },
    BuiltinSig { name: "sys_write_file", arity: 2, is_pure: false, allocates: false, param_types: &["Text", "Text"], ret_type: "Bool" },
    BuiltinSig { name: "sys_exec", arity: 1, is_pure: false, allocates: true, param_types: &["Text"], ret_type: "Text" },
    BuiltinSig { name: "sys_exec_exit_code", arity: 0, is_pure: false, allocates: false, param_types: &[], ret_type: "Int" },
    // 6 Minimal Raw HTTPS Line-Stream transport syscalls (impure)
    BuiltinSig { name: "sys_http_open", arity: 4, is_pure: false, allocates: false, param_types: &["Text", "Text", "Text", "Text"], ret_type: "Int" },
    BuiltinSig { name: "sys_http_sse_open", arity: 3, is_pure: false, allocates: false, param_types: &["Text", "Text", "Text"], ret_type: "Int" },
    BuiltinSig { name: "sys_http_sse_next_line", arity: 1, is_pure: false, allocates: true, param_types: &["Int"], ret_type: "Text" },
    BuiltinSig { name: "sys_http_sse_eof", arity: 1, is_pure: false, allocates: false, param_types: &["Int"], ret_type: "Bool" },
    BuiltinSig { name: "sys_http_sse_error", arity: 1, is_pure: false, allocates: true, param_types: &["Int"], ret_type: "Text" },
    BuiltinSig { name: "sys_http_sse_close", arity: 1, is_pure: false, allocates: false, param_types: &["Int"], ret_type: "Int" },
];

fn lookup_builtin(name: &str) -> Option<&'static BuiltinSig> {
    BUILTIN_SIGS.iter().find(|b| b.name == name)
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

pub fn check_program(program: &Program) -> CheckReport {
    let mut ctx = CheckerCtx::new(program);
    ctx.check_all(program);

    // Run Protocol Constraint Solver if any distributed views/streams/subscribers exist
    let manifest = match solve_distributed_manifest(program) {
        Ok(m) => m,
        Err(errs) => {
            for e in errs {
                ctx.diagnostics.push(e);
            }
            None
        }
    };

    // Derive SQL CHECK constraints for visible refined records
    let sql_checks = derive_sqlite_checks(program);

    CheckReport {
        ok: ctx.diagnostics.is_empty(),
        proofs: ctx.proofs,
        diagnostics: ctx.diagnostics,
        sql_checks,
        manifest,
    }
}

#[derive(Debug, Clone)]
struct VariantInfo {
    enum_name: String,
    payload_tys: Vec<TypeExpr>,
}

struct CheckerCtx {
    type_aliases: BTreeMap<String, TypeAliasDecl>,
    records: BTreeMap<String, RecordDecl>,
    enums: BTreeMap<String, EnumDecl>,
    enum_variants: BTreeMap<String, VariantInfo>,
    interfaces: BTreeMap<String, InterfaceDecl>,
    inherent_methods: BTreeMap<(String, String), FnDecl>,
    interface_impls: BTreeMap<(String, String), ImplDecl>,
    impl_methods: BTreeMap<(String, String), Vec<(String, FnDecl)>>,
    pure_fns: BTreeMap<String, FnDecl>,
    smt_pure_fns: BTreeMap<String, FnDecl>,
    no_heap_fns: BTreeSet<String>,
    all_fns: BTreeMap<String, FnDecl>,
    async_fns: BTreeSet<String>,
    cur_type_params: BTreeSet<String>,
    cur_type_param_bounds: BTreeMap<String, Vec<String>>,
    proofs: Vec<ProofRecord>,
    diagnostics: Vec<Diagnostic>,
}

fn type_head_name(ty: &TypeExpr) -> String {
    match ty {
        TypeExpr::Named { name, .. } => name.clone(),
        TypeExpr::Dyn { interface, .. } => format!("dyn {}", interface),
        TypeExpr::Tuple(_, _) => "Tuple".into(),
    }
}

fn subst_type_expr(ty: &TypeExpr, subst: &BTreeMap<String, TypeExpr>) -> TypeExpr {
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
                args: args.iter().map(|a| subst_type_expr(a, subst)).collect(),
                span: *span,
            }
        }
        TypeExpr::Dyn {
            interface,
            args,
            span,
        } => TypeExpr::Dyn {
            interface: interface.clone(),
            args: args.iter().map(|a| subst_type_expr(a, subst)).collect(),
            span: *span,
        },
        TypeExpr::Tuple(es, span) => TypeExpr::Tuple(
            es.iter().map(|e| subst_type_expr(e, subst)).collect(),
            *span,
        ),
    }
}

fn subst_fn_self(f: &FnDecl, self_ty: &TypeExpr) -> FnDecl {
    let mut subst = BTreeMap::new();
    subst.insert("Self".into(), self_ty.clone());
    let mut out = f.clone();
    for p in &mut out.params {
        p.ty = subst_type_expr(&p.ty, &subst);
    }
    if let Some(rt) = &out.ret_ty {
        out.ret_ty = Some(subst_type_expr(rt, &subst));
    }
    out
}

fn iface_method_to_fn_decl(m: &InterfaceMethodDecl, self_ty: &TypeExpr) -> FnDecl {
    let f = FnDecl {
        attrs: m.attrs.clone(),
        is_pub: true,
        is_pure: m.is_pure,
        is_async: m.is_async,
        is_test: false,
        name: m.name.clone(),
        type_params: m.type_params.clone(),
        type_param_bounds: m.type_param_bounds.clone(),
        params: m.params.clone(),
        ret_ty: m.ret_ty.clone(),
        requires: m.requires.clone(),
        ensures: m.ensures.clone(),
        body: m.default_body.clone().unwrap_or(Block {
            stmts: vec![],
            tail: None,
            span: m.span,
        }),
        span: m.span,
    };
    subst_fn_self(&f, self_ty)
}

impl CheckerCtx {
    fn new(program: &Program) -> Self {
        let mut type_aliases = BTreeMap::new();
        let mut records = BTreeMap::new();
        let mut enums = BTreeMap::new();
        let mut enum_variants = BTreeMap::new();
        let mut interfaces = BTreeMap::new();
        let mut inherent_methods = BTreeMap::new();
        let mut interface_impls = BTreeMap::new();
        let mut impl_methods: BTreeMap<(String, String), Vec<(String, FnDecl)>> = BTreeMap::new();
        let mut pure_fns = BTreeMap::new();
        let mut no_heap_fns = BTreeSet::new();
        let mut all_fns = BTreeMap::new();
        let mut async_fns = BTreeSet::new();
        async_fns.insert("yield_val".into());

        for item in &program.items {
            match item {
                Item::TypeAlias(t) => {
                    type_aliases.insert(t.name.clone(), t.clone());
                }
                Item::Record(r) => {
                    records.insert(r.name.clone(), r.clone());
                    for m in &r.methods {
                        let sub = subst_fn_self(m, &named_ty(&r.name));
                        inherent_methods.insert((r.name.clone(), m.name.clone()), sub.clone());
                        all_fns.insert(format!("{}.{}", r.name, m.name), sub);
                    }
                }
                Item::Enum(e) => {
                    enums.insert(e.name.clone(), e.clone());
                    for v in &e.variants {
                        let payload_tys: Vec<TypeExpr> =
                            v.fields.iter().map(|f| f.ty.clone()).collect();
                        let info = VariantInfo {
                            enum_name: e.name.clone(),
                            payload_tys,
                        };
                        enum_variants.insert(v.name.clone(), info.clone());
                        enum_variants.insert(format!("{}.{}", e.name, v.name), info);
                    }
                }
                Item::Interface(iface) => {
                    interfaces.insert(iface.name.clone(), iface.clone());
                }
                Item::Impl(_) => {}
                Item::Fn(f) => {
                    if f.is_pure {
                        pure_fns.insert(f.name.clone(), f.clone());
                    }
                    if f.attrs.iter().any(|a| a.name == "no_heap_alloc") {
                        no_heap_fns.insert(f.name.clone());
                    }
                    if f.is_async {
                        async_fns.insert(f.name.clone());
                    }
                    all_fns.insert(f.name.clone(), f.clone());
                }
                _ => {}
            }
        }

        // Second pass: index all `impl` blocks (inherent and interface impls, including default methods)
        for item in &program.items {
            if let Item::Impl(imp) = item {
                let tname = type_head_name(&imp.target_ty);
                if let Some(iface_ty) = &imp.interface {
                    let iname = type_head_name(iface_ty);
                    interface_impls.insert((iname.clone(), tname.clone()), imp.clone());
                    let mut provided = BTreeSet::new();
                    for m in &imp.methods {
                        provided.insert(m.name.clone());
                        let sub = subst_fn_self(m, &imp.target_ty);
                        impl_methods
                            .entry((tname.clone(), m.name.clone()))
                            .or_default()
                            .push((iname.clone(), sub));
                    }
                    // Inherit any default methods from the interface not overridden by this impl
                    if let Some(idecl) = interfaces.get(&iname) {
                        for im in &idecl.methods {
                            if !provided.contains(&im.name) && im.default_body.is_some() {
                                let sub = iface_method_to_fn_decl(im, &imp.target_ty);
                                impl_methods
                                    .entry((tname.clone(), im.name.clone()))
                                    .or_default()
                                    .push((iname.clone(), sub));
                            }
                        }
                    }
                } else {
                    for m in &imp.methods {
                        let sub = subst_fn_self(m, &imp.target_ty);
                        inherent_methods.insert((tname.clone(), m.name.clone()), sub);
                    }
                }
            }
        }

        // Compute fixpoint subset of `pure_fns` that are strictly scalar (Int/Bool) and SMT-encodable
        let is_scalar_smt_ty = |ty: &TypeExpr| -> bool {
            let mut cur = ty;
            for _ in 0..8 {
                if let TypeExpr::Named { name, args, .. } = cur {
                    if !args.is_empty() {
                        return false;
                    }
                    if matches!(
                        name.as_str(),
                        "Int" | "Int64" | "Int32" | "UInt16" | "USize" | "Bool"
                    ) {
                        return true;
                    }
                    if let Some(alias) = type_aliases.get(name) {
                        cur = &alias.base;
                        continue;
                    }
                    return false;
                }
                return false;
            }
            false
        };
        let mut smt_pure_fns: BTreeMap<String, FnDecl> = BTreeMap::new();
        loop {
            let mut added = false;
            for (name, f) in &pure_fns {
                if smt_pure_fns.contains_key(name) {
                    continue;
                }
                let ret_ok = f.ret_ty.as_ref().is_some_and(&is_scalar_smt_ty);
                let params_ok = f.params.iter().all(|p| is_scalar_smt_ty(&p.ty));
                if !ret_ok || !params_ok || !crate::smt::can_encode_pure_fn(f) {
                    continue;
                }
                let callees = crate::smt::pure_fn_callees(f);
                if callees.iter().all(|c| c != name && smt_pure_fns.contains_key(c)) {
                    smt_pure_fns.insert(name.clone(), f.clone());
                    added = true;
                }
            }
            if !added {
                break;
            }
        }

        Self {
            type_aliases,
            records,
            enums,
            enum_variants,
            interfaces,
            inherent_methods,
            interface_impls,
            impl_methods,
            pure_fns,
            smt_pure_fns,
            no_heap_fns,
            all_fns,
            async_fns,
            cur_type_params: BTreeSet::new(),
            cur_type_param_bounds: BTreeMap::new(),
            proofs: Vec::new(),
            diagnostics: Vec::new(),
        }
    }

    fn check_all(&mut self, program: &Program) {
        for item in &program.items {
            match item {
                Item::Fn(f) => self.check_fn(f),
                Item::Impl(imp) => self.check_impl(imp),
                _ => {}
            }
        }
    }

    fn type_implements(&self, ty: &TypeExpr, iface_name: &str) -> bool {
        let base = self.resolve_base_type(ty);
        match &base {
            TypeExpr::Dyn { interface, .. } => interface == iface_name,
            TypeExpr::Named { name, .. } => {
                if name == "Any" {
                    return true;
                }
                if self.interface_impls.contains_key(&(iface_name.to_string(), name.clone())) {
                    return true;
                }
                if let TypeExpr::Named { name: orig, .. } = ty {
                    if self.interface_impls.contains_key(&(iface_name.to_string(), orig.clone())) {
                        return true;
                    }
                }
                if let Some(bounds) = self.cur_type_param_bounds.get(name) {
                    if bounds.iter().any(|b| b == iface_name) {
                        return true;
                    }
                }
                false
            }
            _ => false,
        }
    }

    fn is_object_safe(&self, iface_name: &str) -> Result<(), String> {
        let Some(iface) = self.interfaces.get(iface_name) else {
            return Err(format!("unknown interface `{}`", iface_name));
        };
        for m in &iface.methods {
            if !m.params.first().is_some_and(|p| p.name == "self") {
                return Err(format!(
                    "interface `{}` is not object-safe: method `{}` has no `self` receiver",
                    iface_name, m.name
                ));
            }
            if !m.type_params.is_empty() {
                return Err(format!(
                    "interface `{}` is not object-safe: method `{}` has generic type parameters",
                    iface_name, m.name
                ));
            }
            for p in m.params.iter().skip(1) {
                if p.ty.display_name().contains("Self") {
                    return Err(format!(
                        "interface `{}` is not object-safe: parameter `{}` of `{}` mentions `Self`",
                        iface_name, p.name, m.name
                    ));
                }
            }
            if let Some(rt) = &m.ret_ty {
                if rt.display_name().contains("Self") {
                    return Err(format!(
                        "interface `{}` is not object-safe: return type of `{}` mentions `Self`",
                        iface_name, m.name
                    ));
                }
            }
        }
        Ok(())
    }

    fn check_dyn_type_valid(&mut self, ty: &TypeExpr, span: Span) {
        match ty {
            TypeExpr::Dyn { interface, args, .. } => {
                if !self.interfaces.contains_key(interface) {
                    self.diagnostics.push(Diagnostic {
                        code: "E0500".into(),
                        message: format!("unknown interface `{}` in `dyn {}`", interface, interface),
                        span,
                        notes: vec![],
                        counterexample: BTreeMap::new(),
                    });
                } else if let Err(reason) = self.is_object_safe(interface) {
                    self.diagnostics.push(Diagnostic {
                        code: "E0506".into(),
                        message: reason,
                        span,
                        notes: vec!["use static generic constraint `[T Interface]` instead of `dyn Interface`".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
                for a in args {
                    self.check_dyn_type_valid(a, span);
                }
            }
            TypeExpr::Named { args, .. } => {
                for a in args {
                    self.check_dyn_type_valid(a, span);
                }
            }
            TypeExpr::Tuple(es, _) => {
                for e in es {
                    self.check_dyn_type_valid(e, span);
                }
            }
        }
    }

    fn resolve_method_sig(&self, recv_ty: &TypeExpr, method_name: &str) -> Option<FnDecl> {
        let base = self.resolve_base_type(recv_ty);
        match &base {
            TypeExpr::Dyn { interface, .. } => {
                let idecl = self.interfaces.get(interface)?;
                let m = idecl.methods.iter().find(|m| m.name == method_name)?;
                Some(iface_method_to_fn_decl(m, recv_ty))
            }
            TypeExpr::Named { name, args, .. } => {
                // 1. Bounded generic type parameter `T` -> check bound interfaces
                if let Some(bounds) = self.cur_type_param_bounds.get(name) {
                    for b in bounds {
                        if let Some(idecl) = self.interfaces.get(b) {
                            if let Some(m) = idecl.methods.iter().find(|m| m.name == method_name) {
                                return Some(iface_method_to_fn_decl(m, recv_ty));
                            }
                        }
                    }
                }
                // 2. Inherent method on exact or base type name
                let lookup_names: Vec<String> = match recv_ty {
                    TypeExpr::Named { name: orig, .. } if orig != name => {
                        vec![orig.clone(), name.clone()]
                    }
                    _ => vec![name.clone()],
                };
                for tn in &lookup_names {
                    if let Some(m) = self.inherent_methods.get(&(tn.clone(), method_name.to_string())) {
                        return Some(self.instantiate_container_method(m, tn, args));
                    }
                }
                // 3. Interface impl method on concrete type
                for tn in &lookup_names {
                    if let Some(list) = self.impl_methods.get(&(tn.clone(), method_name.to_string())) {
                        if let Some((_, m)) = list.first() {
                            return Some(self.instantiate_container_method(m, tn, args));
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn instantiate_container_method(&self, m: &FnDecl, _tn: &str, type_args: &[TypeExpr]) -> FnDecl {
        if type_args.is_empty() || m.type_params.is_empty() {
            return m.clone();
        }
        let mut subst = BTreeMap::new();
        for (tp, ta) in m.type_params.iter().zip(type_args.iter()) {
            subst.insert(tp.clone(), ta.clone());
        }
        let mut out = m.clone();
        for p in &mut out.params {
            p.ty = subst_type_expr(&p.ty, &subst);
        }
        if let Some(rt) = &out.ret_ty {
            out.ret_ty = Some(subst_type_expr(rt, &subst));
        }
        out
    }

    fn resolve_assoc_method_sig(&self, type_name: &str, method_name: &str) -> Option<FnDecl> {
        // Check bounded generic type param `T.from_json(...)`
        if let Some(bounds) = self.cur_type_param_bounds.get(type_name) {
            for b in bounds {
                if let Some(idecl) = self.interfaces.get(b) {
                    if let Some(m) = idecl.methods.iter().find(|m| m.name == method_name) {
                        return Some(iface_method_to_fn_decl(m, &named_ty(type_name)));
                    }
                }
            }
        }
        if let Some(m) = self.inherent_methods.get(&(type_name.to_string(), method_name.to_string())) {
            return Some(m.clone());
        }
        if let Some(list) = self.impl_methods.get(&(type_name.to_string(), method_name.to_string())) {
            if let Some((_, m)) = list.first() {
                return Some(m.clone());
            }
        }
        None
    }

    fn is_known_type_name(&self, name: &str) -> bool {
        self.records.contains_key(name)
            || self.enums.contains_key(name)
            || self.type_aliases.contains_key(name)
            || self.cur_type_params.contains(name)
            || matches!(
                name,
                "Int" | "Int64" | "Int32" | "UInt16" | "USize" | "Bool" | "Text" | "Option" | "Result" | "List"
            )
    }

    fn check_impl(&mut self, imp: &ImplDecl) {
        let tname = type_head_name(&imp.target_ty);
        let prev_tp = self.cur_type_params.clone();
        let prev_bounds = self.cur_type_param_bounds.clone();
        for tp in &imp.type_params {
            self.cur_type_params.insert(tp.clone());
        }
        for (k, v) in &imp.type_param_bounds {
            self.cur_type_param_bounds.insert(k.clone(), v.clone());
        }

        if let Some(iface_ty) = &imp.interface {
            let iname = type_head_name(iface_ty);
            let Some(idecl) = self.interfaces.get(&iname).cloned() else {
                self.diagnostics.push(Diagnostic {
                    code: "E0500".into(),
                    message: format!("unknown interface `{}` in `impl {} for {}`", iname, iname, tname),
                    span: imp.span,
                    notes: vec![],
                    counterexample: BTreeMap::new(),
                });
                self.cur_type_params = prev_tp;
                self.cur_type_param_bounds = prev_bounds;
                return;
            };

            let provided_map: BTreeMap<String, FnDecl> = imp
                .methods
                .iter()
                .map(|m| (m.name.clone(), m.clone()))
                .collect();

            // 1. Verify all required interface methods are implemented & conform to signature, effects, and Liskov contracts!
            for im in &idecl.methods {
                let expected = iface_method_to_fn_decl(im, &imp.target_ty);
                let Some(impl_m) = provided_map.get(&im.name) else {
                    if im.default_body.is_none() {
                        self.diagnostics.push(Diagnostic {
                            code: "E0501".into(),
                            message: format!(
                                "missing method `{}` in `impl {} for {}`",
                                im.name, iname, tname
                            ),
                            span: imp.span,
                            notes: vec![format!("interface `{}` requires method `{}`", iname, im.name)],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    continue;
                };
                let mut concrete_m = subst_fn_self(impl_m, &imp.target_ty);

                // Signature compatibility: arity, `inout`, parameter types, return type
                if concrete_m.params.len() != expected.params.len() {
                    self.diagnostics.push(Diagnostic {
                        code: "E0502".into(),
                        message: format!(
                            "method `{}` in `impl {} for {}` expects {} parameter(s), found {}",
                            im.name,
                            iname,
                            tname,
                            expected.params.len(),
                            concrete_m.params.len()
                        ),
                        span: impl_m.span,
                        notes: vec![],
                        counterexample: BTreeMap::new(),
                    });
                } else {
                    for (ep, ap) in expected.params.iter().zip(concrete_m.params.iter()) {
                        if ep.is_inout != ap.is_inout {
                            self.diagnostics.push(Diagnostic {
                                code: "E0502".into(),
                                message: format!(
                                    "method `{}` in `impl {} for {}`: parameter `{}` `inout` mismatch with interface `{}`",
                                    im.name, iname, tname, ap.name, iname
                                ),
                                span: ap.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                        if !self.types_compatible(&ep.ty, &ap.ty) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0502".into(),
                                message: format!(
                                    "method `{}` in `impl {} for {}`: parameter `{}` expects `{}`, found `{}`",
                                    im.name,
                                    iname,
                                    tname,
                                    ap.name,
                                    ep.ty.display_name(),
                                    ap.ty.display_name()
                                ),
                                span: ap.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                }
                match (&expected.ret_ty, &concrete_m.ret_ty) {
                    (Some(ert), Some(art)) if !self.types_compatible(ert, art) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0502".into(),
                            message: format!(
                                "method `{}` in `impl {} for {}`: return type expects `{}`, found `{}`",
                                im.name,
                                iname,
                                tname,
                                ert.display_name(),
                                art.display_name()
                            ),
                            span: impl_m.span,
                            notes: vec![],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    (Some(ert), None) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0502".into(),
                            message: format!(
                                "method `{}` in `impl {} for {}` must return `{}`",
                                im.name,
                                iname,
                                tname,
                                ert.display_name()
                            ),
                            span: impl_m.span,
                            notes: vec![],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    _ => {}
                }

                // Effect discipline: interface `pure` / `@no_heap_alloc` / `async` MUST be honored!
                if expected.is_pure && !concrete_m.is_pure {
                    self.diagnostics.push(Diagnostic {
                        code: "E0503".into(),
                        message: format!(
                            "effect mismatch in `impl {} for {}`: interface method `{}` requires `pure`",
                            iname, tname, im.name
                        ),
                        span: impl_m.span,
                        notes: vec!["mark the implementation `pure fn`".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
                let iface_no_heap = expected.attrs.iter().any(|a| a.name == "no_heap_alloc");
                let impl_no_heap = concrete_m.attrs.iter().any(|a| a.name == "no_heap_alloc");
                if iface_no_heap && !impl_no_heap {
                    self.diagnostics.push(Diagnostic {
                        code: "E0503".into(),
                        message: format!(
                            "effect mismatch in `impl {} for {}`: interface method `{}` requires `@no_heap_alloc`",
                            iname, tname, im.name
                        ),
                        span: impl_m.span,
                        notes: vec!["annotate the implementation with `@no_heap_alloc`".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
                if expected.is_async != concrete_m.is_async {
                    self.diagnostics.push(Diagnostic {
                        code: "E0503".into(),
                        message: format!(
                            "effect mismatch in `impl {} for {}`: `async` modifier of `{}` does not match interface `{}`",
                            iname, tname, im.name, iname
                        ),
                        span: impl_m.span,
                        notes: vec![],
                        counterexample: BTreeMap::new(),
                    });
                }

                // Liskov behavioural subtyping: merge interface `requires` (available to impl)
                // and interface `ensures` (MUST be proved by impl body via L0/L1 Z3!)
                for req in &expected.requires {
                    concrete_m.requires.push(req.clone());
                }
                for ens in &expected.ensures {
                    concrete_m.ensures.push(ens.clone());
                }
                concrete_m.name = format!("{}::{}::{}", tname, iname, im.name);
                let diag_before = self.diagnostics.len();
                self.check_fn(&concrete_m);
                if self.diagnostics.len() == diag_before {
                    self.proofs.push(ProofRecord {
                        obligation: format!("impl `{}` for `{}` satisfies `{}` contract", iname, tname, im.name),
                        tier: ProofTier::L0Fast,
                        span: impl_m.span,
                    });
                }
            }
        } else {
            // Inherent `impl Target { ... }`
            for m in &imp.methods {
                let mut concrete_m = subst_fn_self(m, &imp.target_ty);
                concrete_m.name = format!("{}::{}", tname, m.name);
                self.check_fn(&concrete_m);
            }
        }

        self.cur_type_params = prev_tp;
        self.cur_type_param_bounds = prev_bounds;
    }

    fn resolve_base_type(&self, ty: &TypeExpr) -> TypeExpr {
        if let TypeExpr::Named { name, .. } = ty {
            if let Some(alias) = self.type_aliases.get(name) {
                return self.resolve_base_type(&alias.base);
            }
        }
        ty.clone()
    }

    fn is_int_type_name(&self, name: &str) -> bool {
        if matches!(name, "Int" | "Int64" | "Int32" | "UInt16" | "USize") {
            return true;
        }
        if let Some(alias) = self.type_aliases.get(name) {
            if let TypeExpr::Named { name: base, .. } = self.resolve_base_type(&alias.base) {
                return matches!(base.as_str(), "Int" | "Int64" | "Int32" | "UInt16" | "USize");
            }
        }
        false
    }

    fn expr_smt_encodable(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Int(_) | ExprKind::Bool(_) | ExprKind::Ident(_) => true,
            ExprKind::Field { target, .. } => matches!(&target.kind, ExprKind::Ident(_)),
            ExprKind::Unary { expr: inner, .. } => self.expr_smt_encodable(inner),
            ExprKind::Binary { op, left, right } => {
                !matches!(op, BinOp::Concat | BinOp::BitAnd | BinOp::BitOr)
                    && self.expr_smt_encodable(left)
                    && self.expr_smt_encodable(right)
            }
            ExprKind::Call { callee, args } => {
                let ExprKind::Ident(fname) = &callee.kind else {
                    return false;
                };
                self.smt_pure_fns.contains_key(fname)
                    && args.iter().all(|a| self.expr_smt_encodable(&a.value))
            }
            _ => false,
        }
    }

    fn types_compatible(&self, expected: &TypeExpr, actual: &TypeExpr) -> bool {
        let exp = self.resolve_base_type(expected);
        let act = self.resolve_base_type(actual);
        match (&exp, &act) {
            (TypeExpr::Named { name: en, .. }, _) if en == "Any" => true,
            (_, TypeExpr::Named { name: an, .. }) if an == "Any" => true,
            (TypeExpr::Dyn { interface: ei, .. }, TypeExpr::Dyn { interface: ai, .. }) => ei == ai,
            (TypeExpr::Dyn { interface: ei, .. }, _) => self.type_implements(actual, ei),
            (
                TypeExpr::Named { name: en, args: ea, .. },
                TypeExpr::Named { name: an, args: aa, .. },
            ) => {
                if self.cur_type_params.contains(en) || self.cur_type_params.contains(an) {
                    return true;
                }
                if self.is_int_type_name(en) && self.is_int_type_name(an) {
                    return true;
                }
                if en != an {
                    return false;
                }
                if ea.is_empty() || aa.is_empty() {
                    return true;
                }
                if ea.len() != aa.len() {
                    return false;
                }
                ea.iter()
                    .zip(aa.iter())
                    .all(|(x, y)| self.types_compatible(x, y))
            }
            (TypeExpr::Tuple(es, _), TypeExpr::Tuple(as_, _)) => {
                es.len() == as_.len()
                    && es
                        .iter()
                        .zip(as_.iter())
                        .all(|(x, y)| self.types_compatible(x, y))
            }
            _ => false,
        }
    }

    fn infer_expr_type(
        &self,
        expr: &Expr,
        var_types: &BTreeMap<String, TypeExpr>,
    ) -> TypeExpr {
        match &expr.kind {
            ExprKind::Int(_) => named_ty("Int"),
            ExprKind::Bool(_) => named_ty("Bool"),
            ExprKind::Str(_) => named_ty("Text"),
            ExprKind::Ident(s) => {
                if s == "None" {
                    return generic_ty("Option", vec![named_ty("Any")]);
                }
                if let Some(ty) = var_types.get(s) {
                    return ty.clone();
                }
                if let Some(vi) = self.enum_variants.get(s) {
                    return named_ty(&vi.enum_name);
                }
                named_ty("Any")
            }
            ExprKind::RecordLit { name, .. } => named_ty(name),
            ExprKind::List(elems) => {
                if let Some(first) = elems.first() {
                    generic_ty("List", vec![self.infer_expr_type(first, var_types)])
                } else {
                    generic_ty("List", vec![named_ty("Any")])
                }
            }
            ExprKind::Tuple(elems) => TypeExpr::Tuple(
                elems.iter().map(|e| self.infer_expr_type(e, var_types)).collect(),
                expr.span,
            ),
            ExprKind::Unary { op, .. } => match op {
                UnOp::Neg => named_ty("Int"),
                UnOp::Not => named_ty("Bool"),
            },
            ExprKind::Binary { op, .. } => match op {
                BinOp::Concat => named_ty("Text"),
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::BitAnd | BinOp::BitOr => named_ty("Int"),
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::And | BinOp::Or | BinOp::Implies => named_ty("Bool"),
            },
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    if let Some(vi) = self.enum_variants.get(&format!("{}.{}", ename, field)) {
                        return named_ty(&vi.enum_name);
                    }
                }
                let t = self.resolve_base_type(&self.infer_expr_type(target, var_types));
                if let TypeExpr::Named { name: rname, .. } = t {
                    if let Some(rec) = self.records.get(&rname) {
                        if let Some(f) = rec.fields.iter().find(|f| f.name == *field) {
                            return f.ty.clone();
                        }
                    }
                }
                named_ty("Any")
            }
            ExprKind::Index { target, .. } => {
                let t = self.resolve_base_type(&self.infer_expr_type(target, var_types));
                if let TypeExpr::Named { name, args, .. } = t {
                    if name == "List" && !args.is_empty() {
                        return args[0].clone();
                    }
                }
                named_ty("Any")
            }
            ExprKind::Call { callee, args } => match &callee.kind {
                ExprKind::Field { target, field } if field == "parse" => {
                    if let ExprKind::Ident(rname) = &target.kind {
                        if self.records.contains_key(rname) {
                            return generic_ty("Result", vec![named_ty(rname), named_ty("Int")]);
                        }
                    }
                    named_ty("Any")
                }
                ExprKind::Field { target, field } => {
                    if let ExprKind::Ident(ename) = &target.kind {
                        if let Some(vi) = self.enum_variants.get(&format!("{}.{}", ename, field)) {
                            return named_ty(&vi.enum_name);
                        }
                        if !var_types.contains_key(ename) && self.is_known_type_name(ename) {
                            if let Some(m) = self.resolve_assoc_method_sig(ename, field) {
                                return m.ret_ty.unwrap_or_else(|| named_ty("Any"));
                            }
                        }
                    }
                    let recv_ty = self.infer_expr_type(target, var_types);
                    if let Some(m) = self.resolve_method_sig(&recv_ty, field) {
                        return m.ret_ty.unwrap_or_else(|| named_ty("Any"));
                    }
                    named_ty("Any")
                }
                ExprKind::GenericInst { expr: inner, type_args } => {
                    if let ExprKind::Ident(fname) = &inner.kind {
                        if let Some(fdecl) = self.all_fns.get(fname) {
                            if let Some(rt) = &fdecl.ret_ty {
                                let mut subst = BTreeMap::new();
                                for (tp, ta) in fdecl.type_params.iter().zip(type_args.iter()) {
                                    subst.insert(tp.clone(), ta.clone());
                                }
                                return subst_type_expr(rt, &subst);
                            }
                        }
                    }
                    named_ty("Any")
                }
                ExprKind::Ident(fname) => {
                    if fname == "Ok" && args.len() == 1 {
                        return generic_ty(
                            "Result",
                            vec![self.infer_expr_type(&args[0].value, var_types), named_ty("Any")],
                        );
                    }
                    if fname == "Err" && args.len() == 1 {
                        return generic_ty(
                            "Result",
                            vec![named_ty("Any"), self.infer_expr_type(&args[0].value, var_types)],
                        );
                    }
                    if fname == "Some" && args.len() == 1 {
                        return generic_ty(
                            "Option",
                            vec![self.infer_expr_type(&args[0].value, var_types)],
                        );
                    }
                    if fname == "push" && args.len() == 2 {
                        let lt = self.infer_expr_type(&args[0].value, var_types);
                        if matches!(&lt, TypeExpr::Named { name, args: la, .. } if name == "List" && la.first().is_some_and(|x| x.display_name() != "Any")) {
                            return lt;
                        }
                        return generic_ty("List", vec![self.infer_expr_type(&args[1].value, var_types)]);
                    }
                    if fname == "block_on" && args.len() == 1 {
                        return self.infer_expr_type(&args[0].value, var_types);
                    }
                    if let Some(fdecl) = self.all_fns.get(fname) {
                        return fdecl.ret_ty.clone().unwrap_or_else(|| named_ty("Any"));
                    }
                    if let Some(vi) = self.enum_variants.get(fname) {
                        return named_ty(&vi.enum_name);
                    }
                    if let Some(b) = lookup_builtin(fname) {
                        return if b.ret_type == "List[Any]" {
                            generic_ty("List", vec![named_ty("Any")])
                        } else if b.ret_type == "Option[Any]" {
                            generic_ty("Option", vec![named_ty("Any")])
                        } else if b.ret_type == "Result[Any, Any]" {
                            generic_ty("Result", vec![named_ty("Any"), named_ty("Any")])
                        } else {
                            named_ty(b.ret_type)
                        };
                    }
                    named_ty("Any")
                }
                _ => named_ty("Any"),
            },
            ExprKind::If { then_block, else_branch, .. } => {
                if let Some(t) = &then_block.tail {
                    let ty = self.infer_expr_type(t, var_types);
                    if ty.display_name() != "Any" {
                        return ty;
                    }
                }
                if let Some(eb) = else_branch {
                    return self.infer_expr_type(eb, var_types);
                }
                named_ty("Any")
            }
            ExprKind::Match { scrutinee, arms } => {
                let st = self.infer_expr_type(scrutinee, var_types);
                for arm in arms {
                    let mut arm_vars = var_types.clone();
                    self.bind_match_arm_types(&st, arm, &mut arm_vars);
                    let aty = self.infer_expr_type(&arm.body, &arm_vars);
                    if aty.display_name() != "Any" {
                        return aty;
                    }
                }
                named_ty("Any")
            }
            ExprKind::Block(b) => {
                if let Some(t) = &b.tail {
                    self.infer_expr_type(t, var_types)
                } else {
                    named_ty("Any")
                }
            }
            ExprKind::Await(inner) => self.infer_expr_type(inner, var_types),
            _ => named_ty("Any"),
        }
    }

    fn bind_match_arm_types(
        &self,
        scrutinee_ty: &TypeExpr,
        arm: &MatchArm,
        arm_vars: &mut BTreeMap<String, TypeExpr>,
    ) {
        let base = self.resolve_base_type(scrutinee_ty);
        for p in &arm.patterns {
            if let Pattern::Variant { name, binding: Some(bname), .. } = p {
                if bname == "_" {
                    continue;
                }
                if name == "Some" {
                    if let TypeExpr::Named { name: tn, args, .. } = &base {
                        if tn == "Option" && !args.is_empty() {
                            arm_vars.insert(bname.clone(), args[0].clone());
                            continue;
                        }
                    }
                }
                if name == "Ok" {
                    if let TypeExpr::Named { name: tn, args, .. } = &base {
                        if tn == "Result" && !args.is_empty() {
                            arm_vars.insert(bname.clone(), args[0].clone());
                            continue;
                        }
                    }
                }
                if name == "Err" {
                    if let TypeExpr::Named { name: tn, args, .. } = &base {
                        if tn == "Result" && args.len() >= 2 {
                            arm_vars.insert(bname.clone(), args[1].clone());
                            continue;
                        }
                    }
                }
                if let Some(vi) = self.enum_variants.get(name) {
                    if let Some(pty) = vi.payload_tys.first() {
                        arm_vars.insert(bname.clone(), pty.clone());
                        continue;
                    }
                }
                arm_vars.insert(bname.clone(), named_ty("Any"));
            }
        }
    }

    fn check_fn(&mut self, f: &FnDecl) {
        let prev_tp = self.cur_type_params.clone();
        let prev_bounds = self.cur_type_param_bounds.clone();
        for tp in &f.type_params {
            self.cur_type_params.insert(tp.clone());
        }
        for (k, v) in &f.type_param_bounds {
            self.cur_type_param_bounds.insert(k.clone(), v.clone());
        }

        let mut env = L0Env {
            pure_fns: self.pure_fns.clone(),
            type_aliases: self.type_aliases.clone(),
            ..Default::default()
        };
        let mut smt_premises: Vec<(Expr, BTreeMap<String, String>)> = Vec::new();
        let mut smt_int_vars: BTreeSet<String> = BTreeSet::new();
        let mut var_types: BTreeMap<String, TypeExpr> = BTreeMap::new();
        let mut mutable_vars: BTreeSet<String> = BTreeSet::new();

        // 1. Register parameters, mutability, dyn object-safety, refinements & record invariants
        for p in &f.params {
            self.check_dyn_type_valid(&p.ty, p.span);
            var_types.insert(p.name.clone(), p.ty.clone());
            if p.is_inout {
                mutable_vars.insert(p.name.clone());
            }
            self.assume_type_facts(
                &p.name,
                &p.ty,
                &mut env,
                &mut smt_int_vars,
                &mut smt_premises,
            );
            if let Some(pred) = &p.predicate {
                env.assume_expr(pred, Some(&p.name));
                let mut subst = BTreeMap::new();
                subst.insert("self".into(), p.name.clone());
                smt_premises.push((pred.clone(), subst));
            }
        }
        if let Some(rt) = &f.ret_ty {
            self.check_dyn_type_valid(rt, f.span);
        }

        // 2. Assume function `requires`
        for req in &f.requires {
            env.assume_expr(req, None);
            smt_premises.push((req.clone(), BTreeMap::new()));
        }

        // 3. Static Effect, Purity & Memory Contract Checks (`async`, `pure`, `@no_heap_alloc`)
        let no_heap = f.attrs.iter().any(|a| a.name == "no_heap_alloc");
        self.check_block_effects(&f.body, f.is_async, f.is_pure, no_heap, &f.name, false);

        // 4. Walk statements & tail expression checking types, exhaustiveness, requires, div-by-zero, mutate & early returns
        let mut active_mutate: Option<String> = None;
        let mut saw_return = false;
        let tail_expr = self.check_block(
            &f.body,
            f,
            &mut env,
            &mut smt_int_vars,
            &mut smt_premises,
            &mut var_types,
            &mut mutable_vars,
            &mut active_mutate,
            &mut saw_return,
        );

        // 5. Verify return type & refinement on final tail expression (early `Stmt::Return` paths were already checked!)
        if let Some(ret_expr) = &tail_expr {
            if let Some(ret_ty) = &f.ret_ty {
                let actual_ty = self.infer_expr_type(ret_expr, &var_types);
                if !self.types_compatible(ret_ty, &actual_ty) {
                    self.diagnostics.push(Diagnostic {
                        code: "E0205".into(),
                        message: format!(
                            "return type mismatch in `{}`: expected `{}`, found `{}`",
                            f.name,
                            ret_ty.display_name(),
                            actual_ty.display_name()
                        ),
                        span: ret_expr.span,
                        notes: vec![],
                        counterexample: BTreeMap::new(),
                    });
                }
                self.check_expr_satisfies_type(
                    ret_expr,
                    ret_ty,
                    &env,
                    &smt_int_vars,
                    &smt_premises,
                    &format!("return value of `{}`", f.name),
                    ret_expr.span,
                );
            }
            for ens in &f.ensures {
                self.check_ensures(f, ens, Some(ret_expr), &env, &smt_int_vars, &smt_premises);
            }
        } else if !saw_return {
            for ens in &f.ensures {
                self.check_ensures(f, ens, None, &env, &smt_int_vars, &smt_premises);
            }
        }
        self.cur_type_params = prev_tp;
        self.cur_type_param_bounds = prev_bounds;
    }

    fn check_block_effects(
        &mut self,
        blk: &Block,
        in_async: bool,
        in_pure: bool,
        no_heap: bool,
        fn_name: &str,
        inside_block_on: bool,
    ) {
        for s in &blk.stmts {
            match s {
                Stmt::Let { value, .. } => {
                    self.check_expr_effects(value, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
                Stmt::Assign { value, .. } => {
                    self.check_expr_effects(value, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
                Stmt::Return { value: Some(e), .. } | Stmt::Expr { expr: e, .. } => {
                    self.check_expr_effects(e, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
                Stmt::Assert { cond, .. } => {
                    self.check_expr_effects(cond, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
                Stmt::Mutate { body, .. } => {
                    self.check_block_effects(body, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
                Stmt::While { cond, body, .. } => {
                    self.check_expr_effects(cond, in_async, in_pure, no_heap, fn_name, inside_block_on);
                    self.check_block_effects(body, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
                _ => {}
            }
        }
        if let Some(t) = &blk.tail {
            self.check_expr_effects(t, in_async, in_pure, no_heap, fn_name, inside_block_on);
        }
    }

    fn check_expr_effects(
        &mut self,
        expr: &Expr,
        in_async: bool,
        in_pure: bool,
        no_heap: bool,
        fn_name: &str,
        inside_block_on: bool,
    ) {
        match &expr.kind {
            ExprKind::Await(inner) => {
                if in_pure {
                    self.diagnostics.push(Diagnostic {
                        code: "E0602".into(),
                        message: format!("pure function `{}` cannot use `await`", fn_name),
                        span: expr.span,
                        notes: vec!["remove `pure` or move suspension to an `async fn`".into()],
                        counterexample: BTreeMap::new(),
                    });
                } else if !in_async {
                    self.diagnostics.push(Diagnostic {
                        code: "E0601".into(),
                        message: format!(
                            "async effect violation: `await` used inside synchronous function `{}`",
                            fn_name
                        ),
                        span: expr.span,
                        notes: vec![
                            "mark the enclosing function `async fn` or drive futures via `block_on(...)`".into(),
                        ],
                        counterexample: BTreeMap::new(),
                    });
                }
                self.check_expr_effects(inner, in_async, in_pure, no_heap, fn_name, true);
            }
            ExprKind::RecordLit { name, fields, .. } => {
                if no_heap {
                    self.diagnostics.push(Diagnostic {
                        code: "E0701".into(),
                        message: format!(
                            "`@no_heap_alloc` contract violated in `{}`: record allocation `{}`",
                            fn_name, name
                        ),
                        span: expr.span,
                        notes: vec!["functions annotated `@no_heap_alloc` must stay strictly stack/register-only".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
                for (_, v) in fields {
                    self.check_expr_effects(v, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
            }
            ExprKind::List(elems) => {
                if no_heap {
                    self.diagnostics.push(Diagnostic {
                        code: "E0701".into(),
                        message: format!(
                            "`@no_heap_alloc` contract violated in `{}`: heap list allocation",
                            fn_name
                        ),
                        span: expr.span,
                        notes: vec!["avoid dynamic `List` literals inside `@no_heap_alloc` functions".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
                for e in elems {
                    self.check_expr_effects(e, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
            }
            ExprKind::Str(_) => {
                if no_heap {
                    self.diagnostics.push(Diagnostic {
                        code: "E0701".into(),
                        message: format!(
                            "`@no_heap_alloc` contract violated in `{}`: heap Text allocation",
                            fn_name
                        ),
                        span: expr.span,
                        notes: vec!["avoid dynamic `Text` allocations inside `@no_heap_alloc` functions".into()],
                        counterexample: BTreeMap::new(),
                    });
                }
            }
            ExprKind::Call { callee, args } => {
                let is_block_on = matches!(&callee.kind, ExprKind::Ident(s) if s == "block_on");
                if let ExprKind::Ident(fname) = &callee.kind {
                    if self.async_fns.contains(fname) && !in_async && !inside_block_on {
                        self.diagnostics.push(Diagnostic {
                            code: "E0601".into(),
                            message: format!(
                                "async effect violation: `{}` is `async` and cannot be called directly from sync fn `{}`",
                                fname, fn_name
                            ),
                            span: expr.span,
                            notes: vec![format!("use `await {}(...)` inside an `async fn` or `block_on({}(...))`", fname, fname)],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    // Purity enforcement: `pure fn` cannot call impure user functions or impure OS syscalls!
                    if in_pure {
                        if let Some(b) = lookup_builtin(fname) {
                            if !b.is_pure {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0610".into(),
                                    message: format!(
                                        "purity violation: `pure fn {}` cannot call impure builtin/syscall `{}`",
                                        fn_name, fname
                                    ),
                                    span: expr.span,
                                    notes: vec!["remove `pure` or move I/O effects out of pure functions".into()],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        } else if let Some(target_fn) = self.all_fns.get(fname) {
                            if !target_fn.is_pure {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0610".into(),
                                    message: format!(
                                        "purity violation: `pure fn {}` cannot call non-pure function `{}`",
                                        fn_name, fname
                                    ),
                                    span: expr.span,
                                    notes: vec![format!("mark `{}` as `pure fn` or remove `pure` from `{}`", fname, fn_name)],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        }
                    }
                    // `@no_heap_alloc` enforcement on calls!
                    if no_heap {
                        if let Some(b) = lookup_builtin(fname) {
                            if b.allocates {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0701".into(),
                                    message: format!(
                                        "`@no_heap_alloc` contract violated in `{}`: `{}(...)` allocates on the heap",
                                        fn_name, fname
                                    ),
                                    span: expr.span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        } else if self.enum_variants.contains_key(fname) && !self.all_fns.contains_key(fname) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0701".into(),
                                message: format!(
                                    "`@no_heap_alloc` contract violated in `{}`: enum allocation `{}(...)`",
                                    fn_name, fname
                                ),
                                span: expr.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        } else if self.all_fns.contains_key(fname) && !self.no_heap_fns.contains(fname) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0701".into(),
                                message: format!(
                                    "`@no_heap_alloc` contract violated in `{}`: callee `{}` is not annotated `@no_heap_alloc`",
                                    fn_name, fname
                                ),
                                span: expr.span,
                                notes: vec![format!("annotate `{}` with `@no_heap_alloc` to call it from a no-alloc hot path", fname)],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                }
                if let ExprKind::Field { field, .. } = &callee.kind {
                    if no_heap && field == "parse" {
                        self.diagnostics.push(Diagnostic {
                            code: "E0701".into(),
                            message: format!(
                                "`@no_heap_alloc` contract violated in `{}`: `.parse(...)` allocates a Result/Record",
                                fn_name
                            ),
                            span: expr.span,
                            notes: vec![],
                            counterexample: BTreeMap::new(),
                        });
                    }
                }
                for a in args {
                    self.check_expr_effects(
                        &a.value,
                        in_async,
                        in_pure,
                        no_heap,
                        fn_name,
                        inside_block_on || is_block_on,
                    );
                }
            }
            ExprKind::Binary { op, left, right } => {
                if no_heap && matches!(op, BinOp::Concat) {
                    self.diagnostics.push(Diagnostic {
                        code: "E0701".into(),
                        message: format!(
                            "`@no_heap_alloc` contract violated in `{}`: string concatenation `++` allocates on the heap",
                            fn_name
                        ),
                        span: expr.span,
                        notes: vec![],
                        counterexample: BTreeMap::new(),
                    });
                }
                self.check_expr_effects(left, in_async, in_pure, no_heap, fn_name, inside_block_on);
                self.check_expr_effects(right, in_async, in_pure, no_heap, fn_name, inside_block_on);
            }
            ExprKind::Unary { expr: inner, .. } => {
                self.check_expr_effects(inner, in_async, in_pure, no_heap, fn_name, inside_block_on);
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                self.check_expr_effects(cond, in_async, in_pure, no_heap, fn_name, inside_block_on);
                self.check_block_effects(then_block, in_async, in_pure, no_heap, fn_name, inside_block_on);
                if let Some(eb) = else_branch {
                    self.check_expr_effects(eb, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.check_expr_effects(scrutinee, in_async, in_pure, no_heap, fn_name, inside_block_on);
                for a in arms {
                    self.check_expr_effects(&a.body, in_async, in_pure, no_heap, fn_name, inside_block_on);
                }
            }
            ExprKind::Block(b) => {
                self.check_block_effects(b, in_async, in_pure, no_heap, fn_name, inside_block_on);
            }
            _ => {}
        }
    }

    fn assume_type_facts(
        &self,
        var_name: &str,
        ty: &TypeExpr,
        env: &mut L0Env,
        smt_int_vars: &mut BTreeSet<String>,
        smt_premises: &mut Vec<(Expr, BTreeMap<String, String>)>,
    ) {
        if let TypeExpr::Named { name, .. } = ty {
            if matches!(
                name.as_str(),
                "Int" | "Int64" | "Int32" | "UInt16" | "USize"
            ) {
                smt_int_vars.insert(var_name.to_string());
                match name.as_str() {
                    "UInt16" => env.set_int(
                        var_name,
                        AbsInt::top().refine_interval(Interval { min: 0, max: 65535 }),
                    ),
                    "USize" => env.set_int(
                        var_name,
                        AbsInt::top().refine_interval(Interval {
                            min: 0,
                            max: i64::MAX as i128,
                        }),
                    ),
                    _ => {}
                }
            } else if let Some(alias) = self.type_aliases.get(name).cloned() {
                self.assume_type_facts(var_name, &alias.base, env, smt_int_vars, smt_premises);
                if let Some(pred) = &alias.predicate {
                    env.assume_expr(pred, Some(var_name));
                    let mut subst = BTreeMap::new();
                    subst.insert("self".into(), var_name.to_string());
                    smt_premises.push((pred.clone(), subst));
                }
            } else if let Some(rec) = self.records.get(name).cloned() {
                let mut field_subst = BTreeMap::new();
                field_subst.insert("self".into(), var_name.to_string());
                for field in &rec.fields {
                    let fvar = format!("{}.{}", var_name, field.name);
                    field_subst.insert(field.name.clone(), fvar.clone());
                    self.assume_type_facts(&fvar, &field.ty, env, smt_int_vars, smt_premises);
                    if let Some(fpred) = &field.predicate {
                        let rewritten = substitute_field_idents(fpred, &rec.fields, var_name);
                        env.assume_expr(&rewritten, Some(&fvar));
                        let mut s = field_subst.clone();
                        s.insert("self".into(), fvar.clone());
                        smt_premises.push((fpred.clone(), s));
                    }
                }
                for wc in &rec.where_clauses {
                    let rewritten = substitute_field_idents(wc, &rec.fields, var_name);
                    env.assume_expr(&rewritten, Some(var_name));
                    smt_premises.push((wc.clone(), field_subst.clone()));
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_block(
        &mut self,
        blk: &Block,
        cur_fn: &FnDecl,
        env: &mut L0Env,
        smt_int_vars: &mut BTreeSet<String>,
        smt_premises: &mut Vec<(Expr, BTreeMap<String, String>)>,
        var_types: &mut BTreeMap<String, TypeExpr>,
        mutable_vars: &mut BTreeSet<String>,
        active_mutate: &mut Option<String>,
        saw_return: &mut bool,
    ) -> Option<Expr> {
        for stmt in &blk.stmts {
            match stmt {
                Stmt::Let {
                    name,
                    ty,
                    value,
                    is_mut,
                    span,
                } => {
                    self.check_expr(
                        value,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                    let inferred = self.infer_expr_type(value, var_types);
                    let effective_ty = if let Some(t) = ty {
                        self.check_dyn_type_valid(t, *span);
                        if let TypeExpr::Dyn { interface, .. } = t {
                            if !self.type_implements(&inferred, interface) {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0505".into(),
                                    message: format!(
                                        "type `{}` does not implement interface `{}` required by `{}`",
                                        inferred.display_name(),
                                        interface,
                                        t.display_name()
                                    ),
                                    span: *span,
                                    notes: vec![format!("add `impl {} for {} {{ ... }}`", interface, inferred.display_name())],
                                    counterexample: BTreeMap::new(),
                                });
                            } else if !matches!(inferred, TypeExpr::Dyn { .. })
                                && cur_fn.attrs.iter().any(|a| a.name == "no_heap_alloc")
                            {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0701".into(),
                                    message: format!(
                                        "`@no_heap_alloc` contract violated in `{}`: boxing `{}` into `dyn {}` allocates",
                                        cur_fn.name,
                                        inferred.display_name(),
                                        interface
                                    ),
                                    span: *span,
                                    notes: vec!["use static generic parameter `[T Interface]` on `@no_heap_alloc` paths".into()],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        } else if !self.types_compatible(t, &inferred) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0205".into(),
                                message: format!(
                                    "type mismatch in binding `{}`: expected `{}`, found `{}`",
                                    name,
                                    t.display_name(),
                                    inferred.display_name()
                                ),
                                span: *span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                        self.check_expr_satisfies_type(
                            value,
                            t,
                            env,
                            smt_int_vars,
                            smt_premises,
                            &format!("binding `{}`", name),
                            *span,
                        );
                        t.clone()
                    } else {
                        inferred
                    };

                    var_types.insert(name.clone(), effective_ty.clone());
                    if *is_mut {
                        mutable_vars.insert(name.clone());
                    } else {
                        mutable_vars.remove(name);
                    }
                    self.assume_type_facts(name, &effective_ty, env, smt_int_vars, smt_premises);

                    // Propagate concrete RecordLit fields or copied Ident record fields into `env` & `smt_premises`!
                    match &value.kind {
                        ExprKind::RecordLit { fields, .. } => {
                            for (fname, fexpr) in fields {
                                let fkey = format!("{}.{}", name, fname);
                                let fabs = env.eval_abs_int(fexpr, None);
                                env.set_int(fkey.clone(), fabs);
                                smt_int_vars.insert(fkey.clone());
                                let eq = Expr {
                                    kind: ExprKind::Binary {
                                        op: BinOp::Eq,
                                        left: Box::new(Expr {
                                            kind: ExprKind::Ident(fkey),
                                            span: fexpr.span,
                                        }),
                                        right: Box::new(fexpr.clone()),
                                    },
                                    span: fexpr.span,
                                };
                                smt_premises.push((eq, BTreeMap::new()));
                            }
                        }
                        ExprKind::Ident(src) => {
                            if let TypeExpr::Named { name: rname, .. } = self.resolve_base_type(&effective_ty) {
                                if let Some(rec) = self.records.get(&rname).cloned() {
                                    for f in &rec.fields {
                                        let src_k = format!("{}.{}", src, f.name);
                                        let dst_k = format!("{}.{}", name, f.name);
                                        let fabs = env.get_int(&src_k);
                                        env.set_int(dst_k.clone(), fabs);
                                        smt_int_vars.insert(dst_k.clone());
                                        let eq = Expr {
                                            kind: ExprKind::Binary {
                                                op: BinOp::Eq,
                                                left: Box::new(Expr {
                                                    kind: ExprKind::Ident(dst_k),
                                                    span: *span,
                                                }),
                                                right: Box::new(Expr {
                                                    kind: ExprKind::Ident(src_k),
                                                    span: *span,
                                                }),
                                            },
                                            span: *span,
                                        };
                                        smt_premises.push((eq, BTreeMap::new()));
                                    }
                                }
                            }
                        }
                        _ => {}
                    }

                    let abs = env.eval_abs_int(value, None);
                    // Preserve refined type facts when `eval_abs_int` returns Top!
                    let existing = env.get_int(name);
                    let refined = AbsInt {
                        interval: existing.interval.intersect(abs.interval),
                        cong: existing.cong.meet(abs.cong),
                        non_zero: existing.non_zero || abs.non_zero,
                    };
                    env.set_int(name.clone(), refined);
                    if self.is_int_type_name(&effective_ty.display_name()) {
                        smt_int_vars.insert(name.clone());
                        if self.expr_smt_encodable(value) {
                            let eq = Expr {
                                kind: ExprKind::Binary {
                                    op: BinOp::Eq,
                                    left: Box::new(Expr {
                                        kind: ExprKind::Ident(name.clone()),
                                        span: *span,
                                    }),
                                    right: Box::new(value.clone()),
                                },
                                span: *span,
                            };
                            smt_premises.push((eq, BTreeMap::new()));
                        } else {
                            // Bridge sound L0 domain bounds (e.g. `len(x) >= 0`, `byte_at in [-1, 255]`,
                            // callee `ensures` bounds) into L1 SMT premises.
                            if refined.interval.min > i64::MIN as i128
                                && refined.interval.min <= i64::MAX as i128
                            {
                                let ge = Expr {
                                    kind: ExprKind::Binary {
                                        op: BinOp::Ge,
                                        left: Box::new(Expr {
                                            kind: ExprKind::Ident(name.clone()),
                                            span: *span,
                                        }),
                                        right: Box::new(Expr {
                                            kind: ExprKind::Int(refined.interval.min as i64),
                                            span: *span,
                                        }),
                                    },
                                    span: *span,
                                };
                                smt_premises.push((ge, BTreeMap::new()));
                            }
                            if refined.interval.max < i64::MAX as i128
                                && refined.interval.max >= i64::MIN as i128
                            {
                                let le = Expr {
                                    kind: ExprKind::Binary {
                                        op: BinOp::Le,
                                        left: Box::new(Expr {
                                            kind: ExprKind::Ident(name.clone()),
                                            span: *span,
                                        }),
                                        right: Box::new(Expr {
                                            kind: ExprKind::Int(refined.interval.max as i64),
                                            span: *span,
                                        }),
                                    },
                                    span: *span,
                                };
                                smt_premises.push((le, BTreeMap::new()));
                            }
                        }
                    }
                }
                Stmt::Assert { cond, .. } => {
                    self.check_expr(
                        cond,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                    env.assume_expr(cond, None);
                    smt_premises.push((cond.clone(), BTreeMap::new()));
                }
                Stmt::Assign {
                    target,
                    op,
                    value,
                    span,
                } => {
                    self.check_expr(
                        value,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                    match &target.kind {
                        ExprKind::Ident(vname) => {
                            if !mutable_vars.contains(vname) && var_types.contains_key(vname) {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0203".into(),
                                    message: format!(
                                        "cannot assign to immutable binding `{}`",
                                        vname
                                    ),
                                    span: *span,
                                    notes: vec![format!("declare `{}` with `var` (or `inout` parameter) to mutate it", vname)],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                            if let Some(expected_ty) = var_types.get(vname).cloned() {
                                let rhs_ty = self.infer_expr_type(value, var_types);
                                if !self.types_compatible(&expected_ty, &rhs_ty) {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0205".into(),
                                        message: format!(
                                            "type mismatch assigning to `{}`: expected `{}`, found `{}`",
                                            vname,
                                            expected_ty.display_name(),
                                            rhs_ty.display_name()
                                        ),
                                        span: *span,
                                        notes: vec![],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                            let new_expr = match op {
                                AssignOp::Set => value.clone(),
                                AssignOp::AddSet => Expr {
                                    kind: ExprKind::Binary {
                                        op: BinOp::Add,
                                        left: Box::new(target.clone()),
                                        right: Box::new(value.clone()),
                                    },
                                    span: *span,
                                },
                                AssignOp::SubSet => Expr {
                                    kind: ExprKind::Binary {
                                        op: BinOp::Sub,
                                        left: Box::new(target.clone()),
                                        right: Box::new(value.clone()),
                                    },
                                    span: *span,
                                },
                            };
                            let new_abs = env.eval_abs_int(&new_expr, None);
                            env.set_int(vname.clone(), new_abs);
                            if let Some(expected_ty) = var_types.get(vname).cloned() {
                                self.check_expr_satisfies_type(
                                    &new_expr,
                                    &expected_ty,
                                    env,
                                    smt_int_vars,
                                    smt_premises,
                                    &format!("assignment to `{}`", vname),
                                    *span,
                                );
                            }
                            // Remove stale equality premises referencing `vname`
                            smt_premises.retain(|(p, _)| !expr_mentions_ident(p, vname));
                        }
                        ExprKind::Field {
                            target: obj,
                            field,
                        } => {
                            if let ExprKind::Ident(var) = &obj.kind {
                                if !mutable_vars.contains(var) && var_types.contains_key(var) {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0203".into(),
                                        message: format!(
                                            "cannot mutate field `{}.{}` on immutable binding `{}`",
                                            var, field, var
                                        ),
                                        span: *span,
                                        notes: vec![format!("declare `{}` with `var` or `inout`", var)],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                                let is_inside_mutate = active_mutate.as_deref() == Some(var.as_str());
                                if let Some(ty) = var_types.get(var).cloned() {
                                    let rec_name = ty.display_name();
                                    if let Some(rec) = self.records.get(&rec_name).cloned() {
                                        let has_cross_inv = !rec.where_clauses.is_empty()
                                            || rec.fields.iter().any(|f| {
                                                f.predicate.is_some()
                                                    || self
                                                        .type_aliases
                                                        .get(&f.ty.display_name())
                                                        .is_some_and(|a| a.predicate.is_some())
                                            });
                                        if has_cross_inv && !is_inside_mutate {
                                            self.diagnostics.push(Diagnostic {
                                                code: "E0420".into(),
                                                message: format!(
                                                    "direct field mutation `{}.{}` on invariant record `{}` must be wrapped in `mutate {} {{ ... }}`",
                                                    var, field, rec_name, var
                                                ),
                                                span: *span,
                                                notes: vec![
                                                    "mutating one field directly could expose an intermediate invalid invariant state".into(),
                                                ],
                                                counterexample: BTreeMap::new(),
                                            });
                                        }
                                        let fkey = format!("{}.{}", var, field);
                                        let new_val_expr = match op {
                                            AssignOp::Set => value.clone(),
                                            AssignOp::AddSet => Expr {
                                                kind: ExprKind::Binary {
                                                    op: BinOp::Add,
                                                    left: Box::new(target.clone()),
                                                    right: Box::new(value.clone()),
                                                },
                                                span: *span,
                                            },
                                            AssignOp::SubSet => Expr {
                                                kind: ExprKind::Binary {
                                                    op: BinOp::Sub,
                                                    left: Box::new(target.clone()),
                                                    right: Box::new(value.clone()),
                                                },
                                                span: *span,
                                            },
                                        };
                                        let new_abs = env.eval_abs_int(&new_val_expr, None);
                                        env.set_int(fkey, new_abs);
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Stmt::Mutate { target, body, span } => {
                    let prev = active_mutate.replace(target.clone());
                    let _ = self.check_block(
                        body,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                    *active_mutate = prev;

                    if let Some(ty) = var_types.get(target).cloned() {
                        let rec_name = ty.display_name();
                        if let Some(rec) = self.records.get(&rec_name).cloned() {
                            self.verify_record_invariants_after_mutate(
                                target,
                                &rec,
                                body,
                                env,
                                smt_int_vars,
                                smt_premises,
                                *span,
                            );
                        }
                    }
                }
                Stmt::While {
                    cond,
                    invariants,
                    body,
                    ..
                } => {
                    // Soundly widen variables mutated inside the loop before checking cond & body!
                    let mut mutated = BTreeSet::new();
                    collect_mutated_vars_in_block(body, &mut mutated);
                    for mv in &mutated {
                        env.ints.remove(mv);
                        env.bools.remove(mv);
                        env.diffs.retain(|(a, b), _| a != mv && b != mv);
                        smt_premises.retain(|(p, _)| !expr_mentions_ident(p, mv));
                        if let Some(mty) = var_types.get(mv).cloned() {
                            self.assume_type_facts(mv, &mty, env, smt_int_vars, smt_premises);
                        }
                    }
                    self.check_expr(
                        cond,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                    let mut body_env = env.clone();
                    let mut body_vars = smt_int_vars.clone();
                    let mut body_premises = smt_premises.clone();
                    let mut body_types = var_types.clone();
                    let mut body_mut = mutable_vars.clone();
                    body_env.assume_expr(cond, None);
                    body_premises.push((cond.clone(), BTreeMap::new()));
                    for inv in invariants {
                        body_env.assume_expr(inv, None);
                        body_premises.push((inv.clone(), BTreeMap::new()));
                    }
                    let _ = self.check_block(
                        body,
                        cur_fn,
                        &mut body_env,
                        &mut body_vars,
                        &mut body_premises,
                        &mut body_types,
                        &mut body_mut,
                        active_mutate,
                        saw_return,
                    );
                }
                Stmt::Expr { expr, .. } => {
                    self.check_expr(
                        expr,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                    // Flow-sensitive continuation narrowing: if `if cond { ... return/break/continue }`
                    // has no `else`, then execution reaching the next statement guarantees `!cond`!
                    if let ExprKind::If {
                        cond,
                        then_block,
                        else_branch: None,
                    } = &expr.kind
                    {
                        if block_always_diverges(then_block) {
                            let neg = Expr {
                                kind: ExprKind::Unary {
                                    op: UnOp::Not,
                                    expr: cond.clone(),
                                },
                                span: cond.span,
                            };
                            env.assume_expr(&neg, None);
                            smt_premises.push((neg, BTreeMap::new()));
                        }
                    }
                }
                Stmt::Return { value, span } => {
                    *saw_return = true;
                    if let Some(e) = value {
                        self.check_expr(
                            e,
                            cur_fn,
                            env,
                            smt_int_vars,
                            smt_premises,
                            var_types,
                            mutable_vars,
                            active_mutate,
                            saw_return,
                        );
                        if let Some(ret_ty) = &cur_fn.ret_ty {
                            let actual_ty = self.infer_expr_type(e, var_types);
                            if !self.types_compatible(ret_ty, &actual_ty) {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0205".into(),
                                    message: format!(
                                        "return type mismatch in `{}`: expected `{}`, found `{}`",
                                        cur_fn.name,
                                        ret_ty.display_name(),
                                        actual_ty.display_name()
                                    ),
                                    span: *span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                            self.check_expr_satisfies_type(
                                e,
                                ret_ty,
                                env,
                                smt_int_vars,
                                smt_premises,
                                &format!("return value of `{}`", cur_fn.name),
                                e.span,
                            );
                        }
                        for ens in &cur_fn.ensures {
                            self.check_ensures(cur_fn, ens, Some(e), env, smt_int_vars, smt_premises);
                        }
                    }
                }
                Stmt::Break { .. } | Stmt::Continue { .. } => {}
            }
        }
        if let Some(tail) = &blk.tail {
            self.check_expr(
                tail,
                cur_fn,
                env,
                smt_int_vars,
                smt_premises,
                var_types,
                mutable_vars,
                active_mutate,
                saw_return,
            );
            return Some((**tail).clone());
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_record_invariants_after_mutate(
        &mut self,
        var: &str,
        rec: &RecordDecl,
        body: &Block,
        env: &L0Env,
        smt_int_vars: &BTreeSet<String>,
        smt_premises: &[(Expr, BTreeMap<String, String>)],
        span: Span,
    ) {
        let mut sym_fields: BTreeMap<String, String> = BTreeMap::new();
        for f in &rec.fields {
            sym_fields.insert(
                f.name.clone(),
                crate::smt::smt_ident(&format!("{}.{}", var, f.name)),
            );
        }
        let mut mutated_any = false;
        apply_mutate_block_symbolic(var, body, &mut sym_fields, &mut mutated_any);

        let mut subst = BTreeMap::new();
        subst.insert("self".into(), var.to_string());
        for f in &rec.fields {
            let post_expr = sym_fields
                .get(&f.name)
                .cloned()
                .unwrap_or_else(|| crate::smt::smt_ident(&format!("{}.{}", var, f.name)));
            subst.insert(f.name.clone(), post_expr.clone());
            subst.insert(format!("{}.{}", var, f.name), post_expr);
        }

        for f in &rec.fields {
            let mut field_preds: Vec<Expr> = Vec::new();
            if let Some(alias) = self.type_aliases.get(&f.ty.display_name()) {
                if let Some(ap) = &alias.predicate {
                    field_preds.push(ap.clone());
                }
            }
            if let Some(pred) = &f.predicate {
                field_preds.push(pred.clone());
            }
            for pred in &field_preds {
                let fkey = format!("{}.{}", var, f.name);
                let rewritten = substitute_field_idents(pred, &rec.fields, var);
                let l0_res = if mutated_any {
                    L0Result::Unknown
                } else {
                    env.check_predicate(&rewritten, Some(&fkey))
                };
                match l0_res {
                    L0Result::Proved => {
                        self.proofs.push(ProofRecord {
                            obligation: format!("mutate({}) invariant `{}.{}`", var, rec.name, f.name),
                            tier: ProofTier::L0Fast,
                            span,
                        });
                    }
                    L0Result::Disproved(reason) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0421".into(),
                            message: format!(
                                "mutate `{}` violates field invariant `{}.{}`",
                                var, rec.name, f.name
                            ),
                            span,
                            notes: vec![reason],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    L0Result::Unknown => {
                        let mut q = SmtQuery::new(self.smt_pure_fns.clone());
                        for v in smt_int_vars {
                            q.declare_int(v.clone());
                        }
                        for (p, s) in smt_premises {
                            let _ = q.add_premise_expr(p, s);
                        }
                        let mut fs = subst.clone();
                        fs.insert(
                            "self".into(),
                            sym_fields
                                .get(&f.name)
                                .cloned()
                                .unwrap_or_else(|| crate::smt::smt_ident(&fkey)),
                        );
                        if q.set_goal_expr(pred, &fs).is_ok() {
                            match q.solve() {
                                SmtOutcome::Proved => {
                                    self.proofs.push(ProofRecord {
                                        obligation: format!(
                                            "mutate({}) invariant `{}.{}`",
                                            var, rec.name, f.name
                                        ),
                                        tier: ProofTier::L1Smt,
                                        span,
                                    });
                                }
                                SmtOutcome::Counterexample(ce) => {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0421".into(),
                                        message: format!(
                                            "mutate `{}` may violate field invariant `{}.{}`",
                                            var, rec.name, f.name
                                        ),
                                        span,
                                        notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                                        counterexample: ce,
                                    });
                                }
                                SmtOutcome::Unknown(msg) => {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0422".into(),
                                        message: format!(
                                            "unproved mutate `{}` invariant `{}.{}` (Fail-Closed)",
                                            var, rec.name, f.name
                                        ),
                                        span,
                                        notes: vec![msg],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        for wc in &rec.where_clauses {
            let mut q = SmtQuery::new(self.smt_pure_fns.clone());
            for v in smt_int_vars {
                q.declare_int(v.clone());
            }
            for (p, s) in smt_premises {
                let _ = q.add_premise_expr(p, s);
            }
            if q.set_goal_expr(wc, &subst).is_ok() {
                match q.solve() {
                    SmtOutcome::Proved => {
                        self.proofs.push(ProofRecord {
                            obligation: format!("mutate({}) record invariant `{}`", var, rec.name),
                            tier: ProofTier::L1Smt,
                            span,
                        });
                    }
                    SmtOutcome::Counterexample(ce) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0421".into(),
                            message: format!(
                                "mutate `{}` may violate record invariant on `{}`",
                                var, rec.name
                            ),
                            span,
                            notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                            counterexample: ce,
                        });
                    }
                    SmtOutcome::Unknown(msg) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0422".into(),
                            message: format!(
                                "unproved mutate `{}` record invariant on `{}` (Fail-Closed)",
                                var, rec.name
                            ),
                            span,
                            notes: vec![msg],
                            counterexample: BTreeMap::new(),
                        });
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_expr(
        &mut self,
        expr: &Expr,
        cur_fn: &FnDecl,
        env: &mut L0Env,
        smt_int_vars: &mut BTreeSet<String>,
        smt_premises: &mut Vec<(Expr, BTreeMap<String, String>)>,
        var_types: &BTreeMap<String, TypeExpr>,
        mutable_vars: &BTreeSet<String>,
        active_mutate: &Option<String>,
        saw_return: &mut bool,
    ) {
        match &expr.kind {
            ExprKind::Binary { op, left, right } => {
                self.check_expr(
                    left,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                self.check_expr(
                    right,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                if matches!(op, BinOp::Div | BinOp::Mod) {
                    let denom = env.eval_abs_int(right, None);
                    if !denom.non_zero {
                        let ne_zero = Expr {
                            kind: ExprKind::Binary {
                                op: BinOp::Ne,
                                left: right.clone(),
                                right: Box::new(Expr {
                                    kind: ExprKind::Int(0),
                                    span: right.span,
                                }),
                            },
                            span: right.span,
                        };
                        if matches!(env.check_predicate(&ne_zero, None), L0Result::Proved) {
                            self.proofs.push(ProofRecord {
                                obligation: "division denominator != 0".into(),
                                tier: ProofTier::L0Fast,
                                span: expr.span,
                            });
                        } else {
                            self.diagnostics.push(Diagnostic {
                                code: "E0430".into(),
                                message: "potential division/modulo by zero: denominator is not proved NonZero".into(),
                                span: expr.span,
                                notes: vec!["refine denominator with `where self != 0` or guard with `if denom != 0`".into()],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    } else {
                        self.proofs.push(ProofRecord {
                            obligation: "division denominator != 0 (elided)".into(),
                            tier: ProofTier::L0Fast,
                            span: expr.span,
                        });
                    }
                }
            }
            ExprKind::Unary { expr: inner, .. } | ExprKind::Await(inner) | ExprKind::Old(inner) => {
                self.check_expr(
                    inner,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
            }
            ExprKind::List(elems) | ExprKind::Tuple(elems) => {
                for e in elems {
                    self.check_expr(
                        e,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                }
            }
            ExprKind::Index { target, index } => {
                self.check_expr(
                    target,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                self.check_expr(
                    index,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    if self.enum_variants.contains_key(&format!("{}.{}", ename, field)) {
                        return;
                    }
                }
                self.check_expr(
                    target,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                let tty = self.resolve_base_type(&self.infer_expr_type(target, var_types));
                if let TypeExpr::Named { name: rname, .. } = tty {
                    if let Some(rec) = self.records.get(&rname) {
                        if !rec.fields.iter().any(|f| f.name == *field) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0206".into(),
                                message: format!("unknown field `{}` on record `{}`", field, rname),
                                span: expr.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                }
            }
            ExprKind::RecordLit { name, fields, .. } => {
                for (_, val) in fields {
                    self.check_expr(
                        val,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                }
                if let Some(rec) = self.records.get(name).cloned() {
                    // Check unknown & missing required fields + field type compatibility!
                    let provided_names: BTreeSet<&str> =
                        fields.iter().map(|(n, _)| n.as_str()).collect();
                    for (fname, fval) in fields {
                        if let Some(fdecl) = rec.fields.iter().find(|f| f.name == *fname) {
                            let actual_ty = self.infer_expr_type(fval, var_types);
                            if !self.types_compatible(&fdecl.ty, &actual_ty) {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0205".into(),
                                    message: format!(
                                        "field `{}` of `{}` expects `{}`, found `{}`",
                                        fname,
                                        rec.name,
                                        fdecl.ty.display_name(),
                                        actual_ty.display_name()
                                    ),
                                    span: fval.span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        } else {
                            self.diagnostics.push(Diagnostic {
                                code: "E0204".into(),
                                message: format!(
                                    "unknown field `{}` in `{}` literal",
                                    fname, rec.name
                                ),
                                span: fval.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                    for fdecl in &rec.fields {
                        if fdecl.default.is_none() && !provided_names.contains(fdecl.name.as_str()) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0204".into(),
                                message: format!(
                                    "missing required field `{}` in `{}` literal",
                                    fdecl.name, rec.name
                                ),
                                span: expr.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                    self.check_record_construction(
                        &rec,
                        fields,
                        env,
                        smt_int_vars,
                        smt_premises,
                        expr.span,
                    );
                }
            }
            ExprKind::Call { callee, args } => {
                for a in args {
                    self.check_expr(
                        &a.value,
                        cur_fn,
                        env,
                        smt_int_vars,
                        smt_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                }
                match &callee.kind {
                    ExprKind::Field { target, field } if field == "parse" => {
                        if let ExprKind::Ident(rname) = &target.kind {
                            if let Some(rec) = self.records.get(rname) {
                                if args.len() != rec.fields.len() {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0202".into(),
                                        message: format!(
                                            "`{}.parse` expects {} argument(s), found {}",
                                            rname,
                                            rec.fields.len(),
                                            args.len()
                                        ),
                                        span: expr.span,
                                        notes: vec![],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                        }
                    }
                    ExprKind::Field { target, field } => {
                        let is_enum_ctor = matches!(&target.kind, ExprKind::Ident(ename)
                            if self.enum_variants.contains_key(&format!("{}.{}", ename, field)));
                        if is_enum_ctor {
                            if let ExprKind::Ident(ename) = &target.kind {
                                if let Some(vi) =
                                    self.enum_variants.get(&format!("{}.{}", ename, field)).cloned()
                                {
                                    if args.len() != vi.payload_tys.len() {
                                        self.diagnostics.push(Diagnostic {
                                            code: "E0202".into(),
                                            message: format!(
                                                "variant `{}.{}` expects {} argument(s), found {}",
                                                ename,
                                                field,
                                                vi.payload_tys.len(),
                                                args.len()
                                            ),
                                            span: expr.span,
                                            notes: vec![],
                                            counterexample: BTreeMap::new(),
                                        });
                                    }
                                }
                            }
                        } else if let ExprKind::Ident(tname) = &target.kind {
                            if !var_types.contains_key(tname) && self.is_known_type_name(tname) {
                                // Static associated call: `Type.method(args)` (e.g. `Order.from_json(v)`)
                                if let Some(mdecl) = self.resolve_assoc_method_sig(tname, field) {
                                    self.check_method_call_contract(
                                        &format!("{}.{}", tname, field),
                                        &mdecl,
                                        None,
                                        args,
                                        cur_fn,
                                        env,
                                        smt_int_vars,
                                        smt_premises,
                                        var_types,
                                        mutable_vars,
                                        active_mutate,
                                        expr.span,
                                    );
                                } else {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0207".into(),
                                        message: format!(
                                            "unknown associated method `{}` on type `{}`",
                                            field, tname
                                        ),
                                        span: expr.span,
                                        notes: vec![],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            } else {
                                // Receiver method call on variable `target.field(args)`
                                self.check_expr(
                                    target,
                                    cur_fn,
                                    env,
                                    smt_int_vars,
                                    smt_premises,
                                    var_types,
                                    mutable_vars,
                                    active_mutate,
                                    saw_return,
                                );
                                let recv_ty = self.infer_expr_type(target, var_types);
                                if let Some(mdecl) = self.resolve_method_sig(&recv_ty, field) {
                                    self.check_method_call_contract(
                                        &format!("{}.{}", recv_ty.display_name(), field),
                                        &mdecl,
                                        Some(target),
                                        args,
                                        cur_fn,
                                        env,
                                        smt_int_vars,
                                        smt_premises,
                                        var_types,
                                        mutable_vars,
                                        active_mutate,
                                        expr.span,
                                    );
                                } else if recv_ty.display_name() != "Any" {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0207".into(),
                                        message: format!(
                                            "unknown method `{}` on type `{}`",
                                            field,
                                            recv_ty.display_name()
                                        ),
                                        span: expr.span,
                                        notes: vec![],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                        } else {
                            // Receiver method call on arbitrary expression `(expr).field(args)`
                            self.check_expr(
                                target,
                                cur_fn,
                                env,
                                smt_int_vars,
                                smt_premises,
                                var_types,
                                mutable_vars,
                                active_mutate,
                                saw_return,
                            );
                            let recv_ty = self.infer_expr_type(target, var_types);
                            if let Some(mdecl) = self.resolve_method_sig(&recv_ty, field) {
                                self.check_method_call_contract(
                                    &format!("{}.{}", recv_ty.display_name(), field),
                                    &mdecl,
                                    Some(target),
                                    args,
                                    cur_fn,
                                    env,
                                    smt_int_vars,
                                    smt_premises,
                                    var_types,
                                    mutable_vars,
                                    active_mutate,
                                    expr.span,
                                );
                            } else if recv_ty.display_name() != "Any" {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0207".into(),
                                    message: format!(
                                        "unknown method `{}` on type `{}`",
                                        field,
                                        recv_ty.display_name()
                                    ),
                                    span: expr.span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        }
                    }
                    ExprKind::GenericInst { expr: inner, type_args } => {
                        if let ExprKind::Ident(fname) = &inner.kind {
                            if let Some(fdecl) = self.all_fns.get(fname).cloned() {
                                let mut tsubst = BTreeMap::new();
                                for (tp, ta) in fdecl.type_params.iter().zip(type_args.iter()) {
                                    tsubst.insert(tp.clone(), ta.clone());
                                    if let Some(bounds) = fdecl.type_param_bounds.get(tp) {
                                        for b in bounds {
                                            if !self.type_implements(ta, b) {
                                                self.diagnostics.push(Diagnostic {
                                                    code: "E0505".into(),
                                                    message: format!(
                                                        "type `{}` does not satisfy interface bound `{}` for type parameter `{}` of `{}`",
                                                        ta.display_name(),
                                                        b,
                                                        tp,
                                                        fname
                                                    ),
                                                    span: expr.span,
                                                    notes: vec![format!("add `impl {} for {} {{ ... }}`", b, ta.display_name())],
                                                    counterexample: BTreeMap::new(),
                                                });
                                            }
                                        }
                                    }
                                }
                                let mut inst_fn = fdecl.clone();
                                for p in &mut inst_fn.params {
                                    p.ty = subst_type_expr(&p.ty, &tsubst);
                                }
                                if let Some(rt) = &inst_fn.ret_ty {
                                    inst_fn.ret_ty = Some(subst_type_expr(rt, &tsubst));
                                }
                                self.check_method_call_contract(
                                    fname,
                                    &inst_fn,
                                    None,
                                    args,
                                    cur_fn,
                                    env,
                                    smt_int_vars,
                                    smt_premises,
                                    var_types,
                                    mutable_vars,
                                    active_mutate,
                                    expr.span,
                                );
                            }
                        }
                    }
                    ExprKind::Ident(fname) => {
                        if let Some(fdecl) = self.all_fns.get(fname).cloned() {
                            // Infer generic type parameter bindings & verify interface bounds `[T Tool]`!
                            let mut inferred_tp: BTreeMap<String, TypeExpr> = BTreeMap::new();
                            if !fdecl.type_params.is_empty() {
                                for (p, a) in fdecl.params.iter().zip(args.iter()) {
                                    if let TypeExpr::Named { name: pname, args: pargs, .. } = &p.ty {
                                        if pargs.is_empty() && fdecl.type_params.contains(pname) {
                                            let aty = self.infer_expr_type(&a.value, var_types);
                                            inferred_tp.entry(pname.clone()).or_insert(aty);
                                        }
                                    }
                                }
                                for (tp, concrete_ty) in &inferred_tp {
                                    if let Some(bounds) = fdecl.type_param_bounds.get(tp) {
                                        for b in bounds {
                                            if !self.type_implements(concrete_ty, b) {
                                                self.diagnostics.push(Diagnostic {
                                                    code: "E0505".into(),
                                                    message: format!(
                                                        "type `{}` does not implement interface `{}` required by `{}`",
                                                        concrete_ty.display_name(),
                                                        b,
                                                        fname
                                                    ),
                                                    span: expr.span,
                                                    notes: vec![format!("add `impl {} for {} {{ ... }}`", b, concrete_ty.display_name())],
                                                    counterexample: BTreeMap::new(),
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                            if args.len() != fdecl.params.len() {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0202".into(),
                                    message: format!(
                                        "function `{}` expects {} argument(s), found {}",
                                        fname,
                                        fdecl.params.len(),
                                        args.len()
                                    ),
                                    span: expr.span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                            let mut inout_vars_seen = BTreeSet::new();
                            let mut param_arg_map: BTreeMap<String, Expr> = BTreeMap::new();
                            let mut mutated_inout_targets: Vec<(String, TypeExpr)> = Vec::new();
                            for (p, a) in fdecl.params.iter().zip(args.iter()) {
                                param_arg_map.insert(p.name.clone(), a.value.clone());
                                let expected_pty = if inferred_tp.is_empty() {
                                    p.ty.clone()
                                } else {
                                    subst_type_expr(&p.ty, &inferred_tp)
                                };
                                let actual_ty = self.infer_expr_type(&a.value, var_types);
                                if let TypeExpr::Dyn { interface, .. } = &expected_pty {
                                    if !self.type_implements(&actual_ty, interface) {
                                        self.diagnostics.push(Diagnostic {
                                            code: "E0505".into(),
                                            message: format!(
                                                "argument `{}` of `{}`: type `{}` does not implement interface `{}`",
                                                p.name,
                                                fname,
                                                actual_ty.display_name(),
                                                interface
                                            ),
                                            span: a.value.span,
                                            notes: vec![format!("add `impl {} for {} {{ ... }}`", interface, actual_ty.display_name())],
                                            counterexample: BTreeMap::new(),
                                        });
                                    } else if !matches!(actual_ty, TypeExpr::Dyn { .. })
                                        && cur_fn.attrs.iter().any(|at| at.name == "no_heap_alloc")
                                    {
                                        self.diagnostics.push(Diagnostic {
                                            code: "E0701".into(),
                                            message: format!(
                                                "`@no_heap_alloc` contract violated in `{}`: boxing `{}` into `dyn {}` allocates",
                                                cur_fn.name,
                                                actual_ty.display_name(),
                                                interface
                                            ),
                                            span: a.value.span,
                                            notes: vec![],
                                            counterexample: BTreeMap::new(),
                                        });
                                    }
                                } else if !self.types_compatible(&expected_pty, &actual_ty) {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0205".into(),
                                        message: format!(
                                            "type mismatch for argument `{}` of `{}`: expected `{}`, found `{}`",
                                            p.name,
                                            fname,
                                            expected_pty.display_name(),
                                            actual_ty.display_name()
                                        ),
                                        span: a.value.span,
                                        notes: vec![],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                                if p.is_inout {
                                    if let ExprKind::Ident(v) = &a.value.kind {
                                        if !inout_vars_seen.insert(v.clone()) {
                                            self.diagnostics.push(Diagnostic {
                                                code: "E0501".into(),
                                                message: format!(
                                                    "MVS exclusivity violation: variable `{}` passed to multiple `inout` parameters of `{}`",
                                                    v, fname
                                                ),
                                                span: a.value.span,
                                                notes: vec!["Tenon forbids overlapping mutable aliases".into()],
                                                counterexample: BTreeMap::new(),
                                            });
                                        }
                                        if active_mutate.as_deref() == Some(v.as_str()) {
                                            self.diagnostics.push(Diagnostic {
                                                code: "E0502".into(),
                                                message: format!(
                                                    "cannot pass `{}` (currently unsealed inside `mutate {}`) to `{}`",
                                                    v, v, fname
                                                ),
                                                span: a.value.span,
                                                notes: vec!["record invariant is temporarily suspended inside `mutate`".into()],
                                                counterexample: BTreeMap::new(),
                                            });
                                        }
                                        if !mutable_vars.contains(v) && var_types.contains_key(v) {
                                            self.diagnostics.push(Diagnostic {
                                                code: "E0203".into(),
                                                message: format!(
                                                    "cannot pass immutable binding `{}` to `inout` parameter `{}` of `{}`",
                                                    v, p.name, fname
                                                ),
                                                span: a.value.span,
                                                notes: vec![format!("declare `{}` with `var`", v)],
                                                counterexample: BTreeMap::new(),
                                            });
                                        }
                                        mutated_inout_targets.push((v.clone(), p.ty.clone()));
                                    }
                                }
                                // Verify refined type alias on argument
                                self.check_expr_satisfies_type(
                                    &a.value,
                                    &p.ty,
                                    env,
                                    smt_int_vars,
                                    smt_premises,
                                    &format!("argument `{}` of `{}`", p.name, fname),
                                    a.value.span,
                                );
                                // Verify inline parameter `where` predicate at call site!
                                if let Some(pred) = &p.predicate {
                                    self.check_predicate_obligation(
                                        pred,
                                        Some(&a.value),
                                        env,
                                        smt_int_vars,
                                        smt_premises,
                                        &format!("parameter `{}` predicate of `{}`", p.name, fname),
                                        "E0440",
                                        "E0441",
                                        a.value.span,
                                    );
                                }
                            }
                            // Verify every `requires` precondition at the call site!
                            for req in &fdecl.requires {
                                let instantiated = substitute_expr_idents(req, &param_arg_map);
                                self.check_predicate_obligation(
                                    &instantiated,
                                    None,
                                    env,
                                    smt_int_vars,
                                    smt_premises,
                                    &format!("precondition `requires` of `{}`", fname),
                                    "E0440",
                                    "E0441",
                                    expr.span,
                                );
                            }
                            // After `inout` call, widen mutated target fields to their record invariants!
                            for (v, pty) in mutated_inout_targets {
                                let prefix = format!("{}.", v);
                                env.ints.retain(|k, _| k != &v && !k.starts_with(&prefix));
                                env.bools.retain(|k, _| k != &v && !k.starts_with(&prefix));
                                smt_premises.retain(|(p, _)| !expr_mentions_ident(p, &v));
                                self.assume_type_facts(&v, &pty, env, smt_int_vars, smt_premises);
                            }
                        } else if let Some(vi) = self.enum_variants.get(fname).cloned() {
                            if args.len() != vi.payload_tys.len() {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0202".into(),
                                    message: format!(
                                        "variant `{}` expects {} argument(s), found {}",
                                        fname,
                                        vi.payload_tys.len(),
                                        args.len()
                                    ),
                                    span: expr.span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                            for (pty, a) in vi.payload_tys.iter().zip(args.iter()) {
                                let actual_ty = self.infer_expr_type(&a.value, var_types);
                                if !self.types_compatible(pty, &actual_ty) {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0205".into(),
                                        message: format!(
                                            "variant `{}` payload expects `{}`, found `{}`",
                                            fname,
                                            pty.display_name(),
                                            actual_ty.display_name()
                                        ),
                                        span: a.value.span,
                                        notes: vec![],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                        } else if let Some(b) = lookup_builtin(fname) {
                            if fname == "push" && args.len() == 2 {
                                let list_ty = self.resolve_base_type(&self.infer_expr_type(&args[0].value, var_types));
                                if let TypeExpr::Named { name: ln, args: la, .. } = list_ty {
                                    if ln == "List" && la.len() == 1 {
                                        let item_ty = self.infer_expr_type(&args[1].value, var_types);
                                        if let TypeExpr::Dyn { interface, .. } = &la[0] {
                                            if !self.type_implements(&item_ty, interface) {
                                                self.diagnostics.push(Diagnostic {
                                                    code: "E0505".into(),
                                                    message: format!(
                                                        "cannot push `{}` into `List[dyn {}]`: `{}` does not implement `{}`",
                                                        item_ty.display_name(),
                                                        interface,
                                                        item_ty.display_name(),
                                                        interface
                                                    ),
                                                    span: args[1].value.span,
                                                    notes: vec![format!("add `impl {} for {} {{ ... }}`", interface, item_ty.display_name())],
                                                    counterexample: BTreeMap::new(),
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                            if args.len() != b.arity {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0202".into(),
                                    message: format!(
                                        "builtin `{}` expects {} argument(s), found {}",
                                        fname,
                                        b.arity,
                                        args.len()
                                    ),
                                    span: expr.span,
                                    notes: vec![],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                            for (pt, a) in b.param_types.iter().zip(args.iter()) {
                                let expected = named_ty(pt);
                                let actual = self.infer_expr_type(&a.value, var_types);
                                if !self.types_compatible(&expected, &actual) {
                                    self.diagnostics.push(Diagnostic {
                                        code: "E0205".into(),
                                        message: format!(
                                            "builtin `{}` argument expects `{}`, found `{}`",
                                            fname,
                                            pt,
                                            actual.display_name()
                                        ),
                                        span: a.value.span,
                                        notes: vec![],
                                        counterexample: BTreeMap::new(),
                                    });
                                }
                            }
                        } else {
                            self.diagnostics.push(Diagnostic {
                                code: "E0201".into(),
                                message: format!("call to undefined function `{}`", fname),
                                span: expr.span,
                                notes: vec![],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                    _ => {}
                }
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                self.check_expr(
                    cond,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                let mut then_env = env.clone();
                then_env.assume_expr(cond, None);
                let mut then_premises = smt_premises.clone();
                then_premises.push(((**cond).clone(), BTreeMap::new()));
                let mut then_vars = var_types.clone();
                let mut then_mut = mutable_vars.clone();
                let mut m = active_mutate.clone();
                self.check_block(
                    then_block,
                    cur_fn,
                    &mut then_env,
                    &mut smt_int_vars.clone(),
                    &mut then_premises,
                    &mut then_vars,
                    &mut then_mut,
                    &mut m,
                    saw_return,
                );
                if let Some(eb) = else_branch {
                    let neg_cond = Expr {
                        kind: ExprKind::Unary {
                            op: UnOp::Not,
                            expr: cond.clone(),
                        },
                        span: cond.span,
                    };
                    let mut else_env = env.clone();
                    else_env.assume_expr(&neg_cond, None);
                    let mut else_premises = smt_premises.clone();
                    else_premises.push((neg_cond, BTreeMap::new()));
                    self.check_expr(
                        eb,
                        cur_fn,
                        &mut else_env,
                        &mut smt_int_vars.clone(),
                        &mut else_premises,
                        var_types,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                }
            }
            ExprKind::Match { scrutinee, arms } => {
                self.check_expr(
                    scrutinee,
                    cur_fn,
                    env,
                    smt_int_vars,
                    smt_premises,
                    var_types,
                    mutable_vars,
                    active_mutate,
                    saw_return,
                );
                let st = self.infer_expr_type(scrutinee, var_types);
                self.check_match_exhaustiveness(&st, arms, expr.span);
                for arm in arms {
                    let mut arm_vars = var_types.clone();
                    let mut arm_env = env.clone();
                    let mut arm_smt_vars = smt_int_vars.clone();
                    let mut arm_premises = smt_premises.clone();
                    self.bind_match_arm_types(&st, arm, &mut arm_vars);
                    // Assume type facts for any bound pattern variable
                    for p in &arm.patterns {
                        if let Pattern::Variant { binding: Some(bname), .. } = p {
                            if bname != "_" {
                                if let Some(bty) = arm_vars.get(bname).cloned() {
                                    self.assume_type_facts(
                                        bname,
                                        &bty,
                                        &mut arm_env,
                                        &mut arm_smt_vars,
                                        &mut arm_premises,
                                    );
                                }
                            }
                        }
                    }
                    if let Some(g) = &arm.guard {
                        self.check_expr(
                            g,
                            cur_fn,
                            &mut arm_env,
                            &mut arm_smt_vars,
                            &mut arm_premises,
                            &arm_vars,
                            mutable_vars,
                            active_mutate,
                            saw_return,
                        );
                        arm_env.assume_expr(g, None);
                        arm_premises.push((g.clone(), BTreeMap::new()));
                    }
                    self.check_expr(
                        &arm.body,
                        cur_fn,
                        &mut arm_env,
                        &mut arm_smt_vars,
                        &mut arm_premises,
                        &arm_vars,
                        mutable_vars,
                        active_mutate,
                        saw_return,
                    );
                }
            }
            ExprKind::Block(blk) => {
                let mut b_env = env.clone();
                let mut b_vars = var_types.clone();
                let mut b_mut = mutable_vars.clone();
                let mut m = active_mutate.clone();
                self.check_block(
                    blk,
                    cur_fn,
                    &mut b_env,
                    &mut smt_int_vars.clone(),
                    &mut smt_premises.clone(),
                    &mut b_vars,
                    &mut b_mut,
                    &mut m,
                    saw_return,
                );
            }
            _ => {}
        }
    }

    fn check_match_exhaustiveness(&mut self, scrutinee_ty: &TypeExpr, arms: &[MatchArm], span: Span) {
        let base = self.resolve_base_type(scrutinee_ty);
        let TypeExpr::Named { name: tname, .. } = base else {
            return;
        };

        let required: Option<Vec<String>> = if tname == "Option" {
            Some(vec!["Some".into(), "None".into()])
        } else if tname == "Result" {
            Some(vec!["Ok".into(), "Err".into()])
        } else if tname == "Bool" {
            Some(vec!["true".into(), "false".into()])
        } else if let Some(ed) = self.enums.get(&tname) {
            Some(ed.variants.iter().map(|v| v.name.clone()).collect())
        } else {
            None
        };

        let Some(req_variants) = required else {
            return;
        };

        let mut covered: BTreeSet<String> = BTreeSet::new();
        let mut has_wildcard = false;
        for arm in arms {
            if arm.guard.is_some() {
                // Guarded arms cannot guarantee exhaustive coverage of a variant
                continue;
            }
            for p in &arm.patterns {
                match p {
                    Pattern::Wildcard(_) | Pattern::Ident(_, _) => {
                        has_wildcard = true;
                    }
                    Pattern::Bool(true, _) => {
                        covered.insert("true".into());
                    }
                    Pattern::Bool(false, _) => {
                        covered.insert("false".into());
                    }
                    Pattern::Variant { name, .. } => {
                        let short = name.rsplit('.').next().unwrap_or(name).to_string();
                        covered.insert(short);
                    }
                    _ => {}
                }
            }
        }

        if has_wildcard {
            self.proofs.push(ProofRecord {
                obligation: format!("exhaustive match on `{}` (wildcard)", tname),
                tier: ProofTier::L0Fast,
                span,
            });
            return;
        }

        let missing: Vec<String> = req_variants
            .into_iter()
            .filter(|v| !covered.contains(v))
            .collect();
        if missing.is_empty() {
            self.proofs.push(ProofRecord {
                obligation: format!("exhaustive match on `{}`", tname),
                tier: ProofTier::L0Fast,
                span,
            });
        } else {
            self.diagnostics.push(Diagnostic {
                code: "E0301".into(),
                message: format!(
                    "non-exhaustive `match` on `{}`: missing variant(s) {}",
                    tname,
                    missing.join(", ")
                ),
                span,
                notes: vec!["handle all variants explicitly or add a wildcard `_ => ...` arm".into()],
                counterexample: BTreeMap::new(),
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_predicate_obligation(
        &mut self,
        pred: &Expr,
        self_val_expr: Option<&Expr>,
        env: &L0Env,
        smt_int_vars: &BTreeSet<String>,
        smt_premises: &[(Expr, BTreeMap<String, String>)],
        context: &str,
        fail_code: &str,
        unknown_code: &str,
        span: Span,
    ) {
        let mut check_env = env.clone();
        let self_var_name = if let Some(se) = self_val_expr {
            let abs = env.eval_abs_int(se, None);
            check_env.set_int("__val", abs);
            Some("__val")
        } else {
            None
        };

        match check_env.check_predicate(pred, self_var_name) {
            L0Result::Proved => {
                self.proofs.push(ProofRecord {
                    obligation: context.to_string(),
                    tier: ProofTier::L0Fast,
                    span,
                });
            }
            L0Result::Disproved(reason) => {
                self.diagnostics.push(Diagnostic {
                    code: fail_code.into(),
                    message: format!("{} violated", context),
                    span,
                    notes: vec![reason],
                    counterexample: BTreeMap::new(),
                });
            }
            L0Result::Unknown => {
                let mut q = SmtQuery::new(self.smt_pure_fns.clone());
                for v in smt_int_vars {
                    q.declare_int(v.clone());
                }
                let mut subst = BTreeMap::new();
                if let Some(se) = self_val_expr {
                    q.declare_int("__val");
                    subst.insert("self".into(), "__val".into());
                    let eq = Expr {
                        kind: ExprKind::Binary {
                            op: BinOp::Eq,
                            left: Box::new(Expr {
                                kind: ExprKind::Ident("__val".into()),
                                span,
                            }),
                            right: Box::new(se.clone()),
                        },
                        span,
                    };
                    if q.add_premise_expr(&eq, &BTreeMap::new()).is_err() {
                        self.diagnostics.push(Diagnostic {
                            code: unknown_code.into(),
                            message: format!("unproved {} (Fail-Closed)", context),
                            span,
                            notes: vec!["expression outside supported SMT fragment".into()],
                            counterexample: BTreeMap::new(),
                        });
                        return;
                    }
                }
                for (p, s) in smt_premises {
                    let _ = q.add_premise_expr(p, s);
                }
                if let Err(e) = q.set_goal_expr(pred, &subst) {
                    self.diagnostics.push(Diagnostic {
                        code: unknown_code.into(),
                        message: format!("unproved {} (Fail-Closed)", context),
                        span,
                        notes: vec![e],
                        counterexample: BTreeMap::new(),
                    });
                    return;
                }
                match q.solve() {
                    SmtOutcome::Proved => self.proofs.push(ProofRecord {
                        obligation: context.to_string(),
                        tier: ProofTier::L1Smt,
                        span,
                    }),
                    SmtOutcome::Counterexample(ce) => self.diagnostics.push(Diagnostic {
                        code: fail_code.into(),
                        message: format!("{} may be violated", context),
                        span,
                        notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                        counterexample: ce,
                    }),
                    SmtOutcome::Unknown(msg) => self.diagnostics.push(Diagnostic {
                        code: unknown_code.into(),
                        message: format!("unproved {} (Fail-Closed)", context),
                        span,
                        notes: vec![msg],
                        counterexample: BTreeMap::new(),
                    }),
                }
            }
        }
    }

    fn check_record_construction(
        &mut self,
        rec: &RecordDecl,
        provided: &[(String, Expr)],
        env: &L0Env,
        smt_int_vars: &BTreeSet<String>,
        smt_premises: &[(Expr, BTreeMap<String, String>)],
        span: Span,
    ) {
        let mut local_env = env.clone();
        let mut local_vars = smt_int_vars.clone();
        let mut local_premises = smt_premises.to_vec();
        let mut field_exprs: BTreeMap<String, Expr> = BTreeMap::new();

        for (fname, fexpr) in provided {
            field_exprs.insert(fname.clone(), fexpr.clone());
        }

        for f in &rec.fields {
            let Some(val_expr) = field_exprs.get(&f.name) else {
                continue;
            };
            let abs = local_env.eval_abs_int(val_expr, None);
            local_env.set_int(f.name.clone(), abs);
            local_vars.insert(f.name.clone());
            let eq = Expr {
                kind: ExprKind::Binary {
                    op: BinOp::Eq,
                    left: Box::new(Expr {
                        kind: ExprKind::Ident(f.name.clone()),
                        span: val_expr.span,
                    }),
                    right: Box::new(val_expr.clone()),
                },
                span: val_expr.span,
            };
            local_premises.push((eq, BTreeMap::new()));

            self.check_expr_satisfies_type(
                val_expr,
                &f.ty,
                &local_env,
                &local_vars,
                &local_premises,
                &format!("field `{}` of `{}`", f.name, rec.name),
                val_expr.span,
            );

            if let Some(pred) = &f.predicate {
                match local_env.check_predicate(pred, Some(&f.name)) {
                    L0Result::Proved => {
                        self.proofs.push(ProofRecord {
                            obligation: format!("record `{}` field `{}` refinement", rec.name, f.name),
                            tier: ProofTier::L0Fast,
                            span: val_expr.span,
                        });
                    }
                    L0Result::Disproved(reason) => {
                        self.diagnostics.push(Diagnostic {
                            code: "E0412".into(),
                            message: format!(
                                "refinement constraint not satisfied for field `{}` of `{}`",
                                f.name, rec.name
                            ),
                            span: val_expr.span,
                            notes: vec![
                                reason,
                                format!("hint: use `{}.parse(...)` for unchecked runtime inputs", rec.name),
                            ],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    L0Result::Unknown => {
                        let mut q = SmtQuery::new(self.smt_pure_fns.clone());
                        for v in &local_vars {
                            q.declare_int(v.clone());
                        }
                        for (p, s) in &local_premises {
                            let _ = q.add_premise_expr(p, s);
                        }
                        let mut subst = BTreeMap::new();
                        subst.insert("self".into(), f.name.clone());
                        if let Err(e) = q.set_goal_expr(pred, &subst) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0413".into(),
                                message: format!(
                                    "cannot statically prove refinement for `{}.{}` (Fail-Closed)",
                                    rec.name, f.name
                                ),
                                span: val_expr.span,
                                notes: vec![e],
                                counterexample: BTreeMap::new(),
                            });
                            continue;
                        }
                        match q.solve() {
                            SmtOutcome::Proved => {
                                self.proofs.push(ProofRecord {
                                    obligation: format!(
                                        "record `{}` field `{}` refinement",
                                        rec.name, f.name
                                    ),
                                    tier: ProofTier::L1Smt,
                                    span: val_expr.span,
                                });
                            }
                            SmtOutcome::Counterexample(ce) => {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0412".into(),
                                    message: format!(
                                        "refinement constraint not satisfied for field `{}` of `{}`",
                                        f.name, rec.name
                                    ),
                                    span: val_expr.span,
                                    notes: vec![
                                        "counterexample found by L1 SMT (Z3)".into(),
                                        format!("hint: use `{}.parse(...)` for unchecked runtime inputs", rec.name),
                                    ],
                                    counterexample: ce,
                                });
                            }
                            SmtOutcome::Unknown(msg) => {
                                self.diagnostics.push(Diagnostic {
                                    code: "E0413".into(),
                                    message: format!(
                                        "unproved refinement for `{}.{}` (Fail-Closed)",
                                        rec.name, f.name
                                    ),
                                    span: val_expr.span,
                                    notes: vec![msg],
                                    counterexample: BTreeMap::new(),
                                });
                            }
                        }
                    }
                }
            }
        }

        for wc in &rec.where_clauses {
            let rewritten = substitute_self_fields(wc);
            match local_env.check_predicate(&rewritten, None) {
                L0Result::Proved => {
                    self.proofs.push(ProofRecord {
                        obligation: format!("record `{}` invariant", rec.name),
                        tier: ProofTier::L0Fast,
                        span,
                    });
                }
                L0Result::Disproved(reason) => {
                    self.diagnostics.push(Diagnostic {
                        code: "E0414".into(),
                        message: format!("record invariant not satisfied for `{}`", rec.name),
                        span,
                        notes: vec![reason],
                        counterexample: BTreeMap::new(),
                    });
                }
                L0Result::Unknown => {
                    let mut q = SmtQuery::new(self.smt_pure_fns.clone());
                    for v in &local_vars {
                        q.declare_int(v.clone());
                    }
                    for (p, s) in &local_premises {
                        let _ = q.add_premise_expr(p, s);
                    }
                    if q.set_goal_expr(&rewritten, &BTreeMap::new()).is_ok() {
                        match q.solve() {
                            SmtOutcome::Proved => self.proofs.push(ProofRecord {
                                obligation: format!("record `{}` invariant", rec.name),
                                tier: ProofTier::L1Smt,
                                span,
                            }),
                            SmtOutcome::Counterexample(ce) => self.diagnostics.push(Diagnostic {
                                code: "E0414".into(),
                                message: format!("record invariant not satisfied for `{}`", rec.name),
                                span,
                                notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                                counterexample: ce,
                            }),
                            SmtOutcome::Unknown(msg) => self.diagnostics.push(Diagnostic {
                                code: "E0415".into(),
                                message: format!("unproved record invariant for `{}` (Fail-Closed)", rec.name),
                                span,
                                notes: vec![msg],
                                counterexample: BTreeMap::new(),
                            }),
                        }
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_method_call_contract(
        &mut self,
        label: &str,
        mdecl: &FnDecl,
        recv: Option<&Expr>,
        args: &[CallArg],
        cur_fn: &FnDecl,
        env: &mut L0Env,
        smt_int_vars: &mut BTreeSet<String>,
        smt_premises: &mut Vec<(Expr, BTreeMap<String, String>)>,
        var_types: &BTreeMap<String, TypeExpr>,
        mutable_vars: &BTreeSet<String>,
        active_mutate: &Option<String>,
        span: Span,
    ) {
        if cur_fn.is_pure && !mdecl.is_pure {
            self.diagnostics.push(Diagnostic {
                code: "E0610".into(),
                message: format!(
                    "purity violation: `pure fn {}` cannot call non-pure method `{}`",
                    cur_fn.name, label
                ),
                span,
                notes: vec![],
                counterexample: BTreeMap::new(),
            });
        }
        if cur_fn.attrs.iter().any(|a| a.name == "no_heap_alloc")
            && !mdecl.attrs.iter().any(|a| a.name == "no_heap_alloc")
        {
            self.diagnostics.push(Diagnostic {
                code: "E0701".into(),
                message: format!(
                    "`@no_heap_alloc` contract violated in `{}`: method `{}` is not `@no_heap_alloc`",
                    cur_fn.name, label
                ),
                span,
                notes: vec![],
                counterexample: BTreeMap::new(),
            });
        }

        let has_self = recv.is_some() && mdecl.params.first().is_some_and(|p| p.name == "self");
        let explicit_params = if has_self {
            &mdecl.params[1..]
        } else {
            &mdecl.params[..]
        };

        if has_self {
            if let Some(r_expr) = recv {
                if mdecl.params[0].is_inout {
                    if let ExprKind::Ident(v) = &r_expr.kind {
                        if !mutable_vars.contains(v) && var_types.contains_key(v) {
                            self.diagnostics.push(Diagnostic {
                                code: "E0203".into(),
                                message: format!(
                                    "cannot call `inout self` method `{}` on immutable binding `{}`",
                                    label, v
                                ),
                                span: r_expr.span,
                                notes: vec![format!("declare `{}` with `var`", v)],
                                counterexample: BTreeMap::new(),
                            });
                        }
                    }
                }
            }
        }

        if args.len() != explicit_params.len() {
            self.diagnostics.push(Diagnostic {
                code: "E0202".into(),
                message: format!(
                    "method `{}` expects {} argument(s), found {}",
                    label,
                    explicit_params.len(),
                    args.len()
                ),
                span,
                notes: vec![],
                counterexample: BTreeMap::new(),
            });
        }

        let mut param_arg_map: BTreeMap<String, Expr> = BTreeMap::new();
        if let Some(r_expr) = recv {
            param_arg_map.insert("self".into(), r_expr.clone());
        }
        for (p, a) in explicit_params.iter().zip(args.iter()) {
            param_arg_map.insert(p.name.clone(), a.value.clone());
            let actual_ty = self.infer_expr_type(&a.value, var_types);
            if let TypeExpr::Dyn { interface, .. } = &p.ty {
                if !self.type_implements(&actual_ty, interface) {
                    self.diagnostics.push(Diagnostic {
                        code: "E0505".into(),
                        message: format!(
                            "argument `{}` of `{}`: type `{}` does not implement interface `{}`",
                            p.name,
                            label,
                            actual_ty.display_name(),
                            interface
                        ),
                        span: a.value.span,
                        notes: vec![],
                        counterexample: BTreeMap::new(),
                    });
                }
            } else if !self.types_compatible(&p.ty, &actual_ty) {
                self.diagnostics.push(Diagnostic {
                    code: "E0205".into(),
                    message: format!(
                        "type mismatch for argument `{}` of `{}`: expected `{}`, found `{}`",
                        p.name,
                        label,
                        p.ty.display_name(),
                        actual_ty.display_name()
                    ),
                    span: a.value.span,
                    notes: vec![],
                    counterexample: BTreeMap::new(),
                });
            }
            if p.is_inout {
                if let ExprKind::Ident(v) = &a.value.kind {
                    if active_mutate.as_deref() == Some(v.as_str()) {
                        self.diagnostics.push(Diagnostic {
                            code: "E0502".into(),
                            message: format!(
                                "cannot pass `{}` (currently unsealed inside `mutate {}`) to `{}`",
                                v, v, label
                            ),
                            span: a.value.span,
                            notes: vec![],
                            counterexample: BTreeMap::new(),
                        });
                    }
                    if !mutable_vars.contains(v) && var_types.contains_key(v) {
                        self.diagnostics.push(Diagnostic {
                            code: "E0203".into(),
                            message: format!(
                                "cannot pass immutable binding `{}` to `inout` parameter `{}` of `{}`",
                                v, p.name, label
                            ),
                            span: a.value.span,
                            notes: vec![format!("declare `{}` with `var`", v)],
                            counterexample: BTreeMap::new(),
                        });
                    }
                }
            }
            self.check_expr_satisfies_type(
                &a.value,
                &p.ty,
                env,
                smt_int_vars,
                smt_premises,
                &format!("argument `{}` of `{}`", p.name, label),
                a.value.span,
            );
            if let Some(pred) = &p.predicate {
                self.check_predicate_obligation(
                    pred,
                    Some(&a.value),
                    env,
                    smt_int_vars,
                    smt_premises,
                    &format!("parameter `{}` predicate of `{}`", p.name, label),
                    "E0440",
                    "E0441",
                    a.value.span,
                );
            }
        }
        for req in &mdecl.requires {
            let instantiated = substitute_expr_idents(req, &param_arg_map);
            self.check_predicate_obligation(
                &instantiated,
                None,
                env,
                smt_int_vars,
                smt_premises,
                &format!("precondition `requires` of `{}`", label),
                "E0440",
                "E0441",
                span,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn check_expr_satisfies_type(
        &mut self,
        expr: &Expr,
        ty: &TypeExpr,
        env: &L0Env,
        smt_int_vars: &BTreeSet<String>,
        smt_premises: &[(Expr, BTreeMap<String, String>)],
        context: &str,
        span: Span,
    ) {
        let TypeExpr::Named { name, .. } = ty else {
            return;
        };
        let Some(alias) = self.type_aliases.get(name).cloned() else {
            return;
        };
        let Some(pred) = &alias.predicate else {
            return;
        };

        let mut check_env = env.clone();
        let abs = env.eval_abs_int(expr, None);
        check_env.set_int("__val", abs);

        match check_env.check_predicate(pred, Some("__val")) {
            L0Result::Proved => {
                self.proofs.push(ProofRecord {
                    obligation: format!("{} satisfies `{}`", context, name),
                    tier: ProofTier::L0Fast,
                    span,
                });
            }
            L0Result::Disproved(reason) => {
                self.diagnostics.push(Diagnostic {
                    code: "E0410".into(),
                    message: format!("{} violates refinement type `{}`", context, name),
                    span,
                    notes: vec![reason],
                    counterexample: BTreeMap::new(),
                });
            }
            L0Result::Unknown => {
                let mut q = SmtQuery::new(self.smt_pure_fns.clone());
                for v in smt_int_vars {
                    q.declare_int(v.clone());
                }
                q.declare_int("__val");
                for (p, s) in smt_premises {
                    let _ = q.add_premise_expr(p, s);
                }
                let eq = Expr {
                    kind: ExprKind::Binary {
                        op: BinOp::Eq,
                        left: Box::new(Expr {
                            kind: ExprKind::Ident("__val".into()),
                            span,
                        }),
                        right: Box::new(expr.clone()),
                    },
                    span,
                };
                if q.add_premise_expr(&eq, &BTreeMap::new()).is_err() {
                    self.diagnostics.push(Diagnostic {
                        code: "E0411".into(),
                        message: format!("unproved refinement `{}` for {} (Fail-Closed)", name, context),
                        span,
                        notes: vec!["expression outside supported SMT fragment".into()],
                        counterexample: BTreeMap::new(),
                    });
                    return;
                }
                let mut subst = BTreeMap::new();
                subst.insert("self".into(), "__val".into());
                if let Err(e) = q.set_goal_expr(pred, &subst) {
                    self.diagnostics.push(Diagnostic {
                        code: "E0411".into(),
                        message: format!("unproved refinement `{}` for {} (Fail-Closed)", name, context),
                        span,
                        notes: vec![e],
                        counterexample: BTreeMap::new(),
                    });
                    return;
                }
                match q.solve() {
                    SmtOutcome::Proved => self.proofs.push(ProofRecord {
                        obligation: format!("{} satisfies `{}`", context, name),
                        tier: ProofTier::L1Smt,
                        span,
                    }),
                    SmtOutcome::Counterexample(ce) => self.diagnostics.push(Diagnostic {
                        code: "E0410".into(),
                        message: format!("{} may violate refinement type `{}`", context, name),
                        span,
                        notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                        counterexample: ce,
                    }),
                    SmtOutcome::Unknown(msg) => self.diagnostics.push(Diagnostic {
                        code: "E0411".into(),
                        message: format!("unproved refinement `{}` for {} (Fail-Closed)", name, context),
                        span,
                        notes: vec![msg],
                        counterexample: BTreeMap::new(),
                    }),
                }
            }
        }
    }

    fn check_ensures(
        &mut self,
        f: &FnDecl,
        ens: &Expr,
        tail_expr: Option<&Expr>,
        env: &L0Env,
        smt_int_vars: &BTreeSet<String>,
        smt_premises: &[(Expr, BTreeMap<String, String>)],
    ) {
        let mut check_env = env.clone();
        if let Some(te) = tail_expr {
            let abs = env.eval_abs_int(te, None);
            check_env.set_int("result", abs);
        }

        match check_env.check_predicate(ens, Some("result")) {
            L0Result::Proved => {
                self.proofs.push(ProofRecord {
                    obligation: format!("ensures on `{}`", f.name),
                    tier: ProofTier::L0Fast,
                    span: ens.span,
                });
                return;
            }
            L0Result::Disproved(reason) => {
                self.diagnostics.push(Diagnostic {
                    code: "V1004".into(),
                    message: format!("postcondition `ensures` violated on `{}`", f.name),
                    span: ens.span,
                    notes: vec![reason],
                    counterexample: BTreeMap::new(),
                });
                return;
            }
            L0Result::Unknown => {}
        }

        let mut q = SmtQuery::new(self.smt_pure_fns.clone());
        for v in smt_int_vars {
            q.declare_int(v.clone());
        }
        q.declare_int("result");
        for (p, s) in smt_premises {
            let _ = q.add_premise_expr(p, s);
        }
        if let Some(te) = tail_expr {
            if let Ok(te_smt) = crate::smt::encode_expr(te, &BTreeMap::new()) {
                q.premises.push(format!("(= result {})", te_smt));
            } else if let Ok(body_smt) = crate::smt::encode_block(&f.body, &BTreeMap::new()) {
                q.premises.push(format!("(= result {})", body_smt));
            }
        } else if let Ok(body_smt) = crate::smt::encode_block(&f.body, &BTreeMap::new()) {
            q.premises.push(format!("(= result {})", body_smt));
        }
        if let Err(e) = q.set_goal_expr(ens, &BTreeMap::new()) {
            self.diagnostics.push(Diagnostic {
                code: "V1005".into(),
                message: format!("unproved `ensures` on `{}` (Fail-Closed)", f.name),
                span: ens.span,
                notes: vec![e],
                counterexample: BTreeMap::new(),
            });
            return;
        }
        match q.solve() {
            SmtOutcome::Proved => self.proofs.push(ProofRecord {
                obligation: format!("ensures on `{}`", f.name),
                tier: ProofTier::L1Smt,
                span: ens.span,
            }),
            SmtOutcome::Counterexample(ce) => self.diagnostics.push(Diagnostic {
                code: "V1004".into(),
                message: format!("postcondition `ensures` may be violated on `{}`", f.name),
                span: ens.span,
                notes: vec!["counterexample found by L1 SMT (Z3)".into()],
                counterexample: ce,
            }),
            SmtOutcome::Unknown(msg) => self.diagnostics.push(Diagnostic {
                code: "V1005".into(),
                message: format!("unproved `ensures` on `{}` (Fail-Closed)", f.name),
                span: ens.span,
                notes: vec![msg],
                counterexample: BTreeMap::new(),
            }),
        }
    }
}

fn block_always_diverges(blk: &Block) -> bool {
    if blk.stmts.iter().any(|s| {
        matches!(
            s,
            Stmt::Return { .. } | Stmt::Break { .. } | Stmt::Continue { .. }
        )
    }) {
        return true;
    }
    if let Some(tail) = &blk.tail {
        if let ExprKind::If {
            then_block,
            else_branch: Some(eb),
            ..
        } = &tail.kind
        {
            return block_always_diverges(then_block) && expr_always_diverges(eb);
        }
    }
    false
}

fn expr_always_diverges(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::Block(b) => block_always_diverges(b),
        ExprKind::If {
            then_block,
            else_branch: Some(eb),
            ..
        } => block_always_diverges(then_block) && expr_always_diverges(eb),
        _ => false,
    }
}

fn collect_mutated_vars_in_block(blk: &Block, out: &mut BTreeSet<String>) {
    for stmt in &blk.stmts {
        match stmt {
            Stmt::Assign { target, .. } => match &target.kind {
                ExprKind::Ident(s) => {
                    out.insert(s.clone());
                }
                ExprKind::Field { target: obj, .. } => {
                    if let ExprKind::Ident(s) = &obj.kind {
                        out.insert(s.clone());
                    }
                }
                _ => {}
            },
            Stmt::Mutate { target, body, .. } => {
                out.insert(target.clone());
                collect_mutated_vars_in_block(body, out);
            }
            Stmt::While { body, .. } => collect_mutated_vars_in_block(body, out),
            Stmt::Expr { expr, .. } => collect_mutated_vars_in_expr(expr, out),
            _ => {}
        }
    }
    if let Some(tail) = &blk.tail {
        collect_mutated_vars_in_expr(tail, out);
    }
}

fn collect_mutated_vars_in_expr(expr: &Expr, out: &mut BTreeSet<String>) {
    match &expr.kind {
        ExprKind::If {
            then_block,
            else_branch,
            ..
        } => {
            collect_mutated_vars_in_block(then_block, out);
            if let Some(eb) = else_branch {
                collect_mutated_vars_in_expr(eb, out);
            }
        }
        ExprKind::Match { arms, .. } => {
            for a in arms {
                collect_mutated_vars_in_expr(&a.body, out);
            }
        }
        ExprKind::Block(b) => collect_mutated_vars_in_block(b, out),
        _ => {}
    }
}

fn expr_mentions_ident(expr: &Expr, ident: &str) -> bool {
    let prefix = format!("{}.", ident);
    match &expr.kind {
        ExprKind::Ident(s) => s == ident || s.starts_with(&prefix),
        ExprKind::Field { target, .. } => expr_mentions_ident(target, ident),
        ExprKind::Binary { left, right, .. } => {
            expr_mentions_ident(left, ident) || expr_mentions_ident(right, ident)
        }
        ExprKind::Unary { expr: inner, .. } => expr_mentions_ident(inner, ident),
        ExprKind::Call { args, .. } => args.iter().any(|a| expr_mentions_ident(&a.value, ident)),
        _ => false,
    }
}

fn substitute_expr_idents(expr: &Expr, map: &BTreeMap<String, Expr>) -> Expr {
    let kind = match &expr.kind {
        ExprKind::Ident(s) => {
            if let Some(repl) = map.get(s) {
                return repl.clone();
            }
            expr.kind.clone()
        }
        ExprKind::Field { target, field } => ExprKind::Field {
            target: Box::new(substitute_expr_idents(target, map)),
            field: field.clone(),
        },
        ExprKind::Binary { op, left, right } => ExprKind::Binary {
            op: *op,
            left: Box::new(substitute_expr_idents(left, map)),
            right: Box::new(substitute_expr_idents(right, map)),
        },
        ExprKind::Unary { op, expr: inner } => ExprKind::Unary {
            op: *op,
            expr: Box::new(substitute_expr_idents(inner, map)),
        },
        ExprKind::Call { callee, args } => ExprKind::Call {
            callee: callee.clone(),
            args: args
                .iter()
                .map(|a| CallArg {
                    name: a.name.clone(),
                    value: substitute_expr_idents(&a.value, map),
                })
                .collect(),
        },
        other => other.clone(),
    };
    Expr {
        kind,
        span: expr.span,
    }
}

fn substitute_field_idents(expr: &Expr, fields: &[FieldDecl], prefix: &str) -> Expr {
    let field_set: BTreeSet<&str> = fields.iter().map(|f| f.name.as_str()).collect();
    fn rewrite(e: &Expr, fields: &BTreeSet<&str>, prefix: &str) -> Expr {
        let kind = match &e.kind {
            ExprKind::Ident(s) if fields.contains(s.as_str()) => ExprKind::Field {
                target: Box::new(Expr {
                    kind: ExprKind::Ident(prefix.to_string()),
                    span: e.span,
                }),
                field: s.clone(),
            },
            ExprKind::Binary { op, left, right } => ExprKind::Binary {
                op: *op,
                left: Box::new(rewrite(left, fields, prefix)),
                right: Box::new(rewrite(right, fields, prefix)),
            },
            ExprKind::Unary { op, expr } => ExprKind::Unary {
                op: *op,
                expr: Box::new(rewrite(expr, fields, prefix)),
            },
            ExprKind::Call { callee, args } => ExprKind::Call {
                callee: callee.clone(),
                args: args
                    .iter()
                    .map(|a| CallArg {
                        name: a.name.clone(),
                        value: rewrite(&a.value, fields, prefix),
                    })
                    .collect(),
            },
            other => other.clone(),
        };
        Expr { kind, span: e.span }
    }
    rewrite(expr, &field_set, prefix)
}

fn substitute_self_fields(expr: &Expr) -> Expr {
    let kind = match &expr.kind {
        ExprKind::Field { target, field } => {
            if let ExprKind::Ident(s) = &target.kind {
                if s == "self" {
                    return Expr {
                        kind: ExprKind::Ident(field.clone()),
                        span: expr.span,
                    };
                }
            }
            expr.kind.clone()
        }
        ExprKind::Binary { op, left, right } => ExprKind::Binary {
            op: *op,
            left: Box::new(substitute_self_fields(left)),
            right: Box::new(substitute_self_fields(right)),
        },
        ExprKind::Unary { op, expr: inner } => ExprKind::Unary {
            op: *op,
            expr: Box::new(substitute_self_fields(inner)),
        },
        other => other.clone(),
    };
    Expr {
        kind,
        span: expr.span,
    }
}

/// Automatically derive SQLite CHECK clauses from direct scalar predicates on records.
pub fn derive_sqlite_checks(program: &Program) -> BTreeMap<String, String> {
    let mut aliases: BTreeMap<String, Expr> = BTreeMap::new();
    for item in &program.items {
        if let Item::TypeAlias(ta) = item {
            if let Some(p) = &ta.predicate {
                aliases.insert(ta.name.clone(), p.clone());
            }
        }
    }
    let mut checks = BTreeMap::new();
    for item in &program.items {
        if let Item::Record(r) = item {
            let mut clauses = Vec::new();
            for f in &r.fields {
                if let Some(ap) = aliases.get(&f.ty.display_name()) {
                    if let Some(sql) = expr_to_sql(ap, Some(&f.name)) {
                        clauses.push(format!("({})", sql));
                    }
                }
                if let Some(pred) = &f.predicate {
                    if let Some(sql) = expr_to_sql(pred, Some(&f.name)) {
                        clauses.push(format!("({})", sql));
                    }
                }
            }
            for wc in &r.where_clauses {
                if let Some(sql) = expr_to_sql(&substitute_self_fields(wc), None) {
                    clauses.push(format!("({})", sql));
                }
            }
            if !clauses.is_empty() {
                checks.insert(r.name.clone(), format!("CHECK (({}) IS 1)", clauses.join(" AND ")));
            }
        }
    }
    checks
}

fn expr_to_sql(expr: &Expr, self_col: Option<&str>) -> Option<String> {
    match &expr.kind {
        ExprKind::Int(n) => Some(n.to_string()),
        ExprKind::Bool(b) => Some(if *b { "1".into() } else { "0".into() }),
        ExprKind::Ident(s) => {
            let col = if s == "self" {
                self_col?
            } else {
                s.as_str()
            };
            Some(format!("\"{}\"", col))
        }
        ExprKind::Unary { op, expr } => {
            let inner = expr_to_sql(expr, self_col)?;
            match op {
                UnOp::Neg => Some(format!("(-{})", inner)),
                UnOp::Not => Some(format!("(NOT {})", inner)),
            }
        }
        ExprKind::Binary { op, left, right } => {
            let l = expr_to_sql(left, self_col)?;
            let r = expr_to_sql(right, self_col)?;
            let sym = match op {
                BinOp::And => "AND",
                BinOp::Or => "OR",
                BinOp::Eq => "=",
                BinOp::Ne => "!=",
                BinOp::Lt => "<",
                BinOp::Le => "<=",
                BinOp::Gt => ">",
                BinOp::Ge => ">=",
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Mod => "%",
                _ => return None,
            };
            Some(format!("({} {} {})", l, sym, r))
        }
        _ => None,
    }
}

fn apply_mutate_block_symbolic(
    var: &str,
    blk: &Block,
    sym_fields: &mut BTreeMap<String, String>,
    mutated_any: &mut bool,
) {
    for stmt in &blk.stmts {
        match stmt {
            Stmt::Assign {
                target, op, value, ..
            } => {
                if let ExprKind::Field { target: obj, field } = &target.kind {
                    if matches!(&obj.kind, ExprKind::Ident(s) if s == var) {
                        let mut subst = BTreeMap::new();
                        for (k, v) in sym_fields.iter() {
                            subst.insert(format!("{}.{}", var, k), v.clone());
                            subst.insert(k.clone(), v.clone());
                        }
                        if let Ok(rhs) = crate::smt::encode_expr(value, &subst) {
                            let cur = sym_fields
                                .get(field)
                                .cloned()
                                .unwrap_or_else(|| crate::smt::smt_ident(&format!("{}.{}", var, field)));
                            let updated = match op {
                                AssignOp::Set => rhs,
                                AssignOp::AddSet => format!("(+ {} {})", cur, rhs),
                                AssignOp::SubSet => format!("(- {} {})", cur, rhs),
                            };
                            sym_fields.insert(field.clone(), updated);
                            *mutated_any = true;
                        }
                    }
                }
            }
            Stmt::Expr { expr, .. } => {
                apply_mutate_expr_symbolic(var, expr, sym_fields, mutated_any);
            }
            _ => {}
        }
    }
    if let Some(tail) = &blk.tail {
        apply_mutate_expr_symbolic(var, tail, sym_fields, mutated_any);
    }
}

fn apply_mutate_expr_symbolic(
    var: &str,
    expr: &Expr,
    sym_fields: &mut BTreeMap<String, String>,
    mutated_any: &mut bool,
) {
    match &expr.kind {
        ExprKind::If {
            cond,
            then_block,
            else_branch,
        } => {
            let mut subst = BTreeMap::new();
            for (k, v) in sym_fields.iter() {
                subst.insert(format!("{}.{}", var, k), v.clone());
                subst.insert(k.clone(), v.clone());
            }
            let Ok(cond_smt) = crate::smt::encode_expr(cond, &subst) else {
                return;
            };
            let mut then_fields = sym_fields.clone();
            let mut else_fields = sym_fields.clone();
            let mut m1 = false;
            let mut m2 = false;
            apply_mutate_block_symbolic(var, then_block, &mut then_fields, &mut m1);
            if let Some(eb) = else_branch {
                apply_mutate_expr_symbolic(var, eb, &mut else_fields, &mut m2);
            }
            if m1 || m2 {
                *mutated_any = true;
                let keys: Vec<String> = sym_fields.keys().cloned().collect();
                for k in keys {
                    let t = &then_fields[&k];
                    let e = &else_fields[&k];
                    if t != e {
                        sym_fields.insert(k, format!("(ite {} {} {})", cond_smt, t, e));
                    }
                }
            }
        }
        ExprKind::Block(b) => apply_mutate_block_symbolic(var, b, sym_fields, mutated_any),
        _ => {}
    }
}
