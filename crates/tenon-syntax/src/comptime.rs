use crate::ast::*;
use crate::parser::{parse_expr_str, parse_with_file_id};
use crate::span::Span;
use std::collections::{BTreeMap, BTreeSet};

const MAX_STEPS: usize = 250_000;
const MAX_CALL_DEPTH: usize = 128;
const MAX_MACRO_DEPTH: usize = 32;

#[derive(Debug, Clone)]
pub enum CtVal {
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
    Unit,
    List(Vec<CtVal>),
    Tuple(Vec<CtVal>),
    Record {
        name: String,
        fields: BTreeMap<String, CtVal>,
    },
    Enum {
        name: String,
        variant: String,
        payload: Vec<CtVal>,
    },
    Type(TypeExpr),
}

enum Flow {
    Return(CtVal),
    Break,
    Continue,
    Err(String),
}

impl From<String> for Flow {
    fn from(s: String) -> Self {
        Flow::Err(s)
    }
}

pub struct ComptimeContext {
    pub fns: BTreeMap<String, FnDecl>,
    pub macros: BTreeMap<String, MacroDecl>,
    pub records: BTreeMap<String, RecordDecl>,
    pub enums: BTreeMap<String, EnumDecl>,
    pub bitfields: BTreeMap<String, BitfieldDecl>,
    pub unions: BTreeMap<String, UnionDecl>,
    pub aliases: BTreeMap<String, TypeAliasDecl>,
    pub consts: BTreeMap<String, Expr>,
    /// `(target_type_name, interface_name)` pairs from `impl` & `@derive`
    pub impls: BTreeSet<(String, String)>,
    pub build_config: BTreeMap<String, String>,
    pub all_fns_ordered: Vec<FnDecl>,
}

impl ComptimeContext {
    pub fn from_program(prog: &Program) -> Self {
        let mut ctx = Self {
            fns: BTreeMap::new(),
            macros: BTreeMap::new(),
            records: BTreeMap::new(),
            enums: BTreeMap::new(),
            bitfields: BTreeMap::new(),
            unions: BTreeMap::new(),
            aliases: BTreeMap::new(),
            consts: BTreeMap::new(),
            impls: BTreeSet::new(),
            build_config: prog.build_config.clone(),
            all_fns_ordered: Vec::new(),
        };
        // Built-in primitive conformances
        for prim in ["Int", "Int64", "Int32", "UInt8", "UInt16", "UInt32", "UInt64", "Bool", "Text", "Float", "Float32", "Float64"] {
            ctx.impls.insert((prim.into(), "Eq".into()));
            ctx.impls.insert((prim.into(), "Display".into()));
            if !prim.starts_with("Float") {
                ctx.impls.insert((prim.into(), "Hash".into()));
            }
        }
        ctx.index_items(&prog.items);
        ctx
    }

    fn index_items(&mut self, items: &[Item]) {
        for item in items {
            match item {
                Item::Fn(f) => {
                    self.fns.insert(f.name.clone(), f.clone());
                    self.all_fns_ordered.push(f.clone());
                }
                Item::Macro(m) => {
                    self.macros.insert(m.name.clone(), m.clone());
                }
                Item::Record(r) => {
                    for attr in &r.attrs {
                        if attr.name == "derive" {
                            for arg in &attr.args {
                                if let ExprKind::Ident(dname) = &arg.value.kind {
                                    self.impls.insert((r.name.clone(), dname.clone()));
                                }
                            }
                        }
                    }
                    self.records.insert(r.name.clone(), r.clone());
                }
                Item::Enum(e) => {
                    self.enums.insert(e.name.clone(), e.clone());
                }
                Item::Bitfield(b) => {
                    self.bitfields.insert(b.name.clone(), b.clone());
                }
                Item::Union(u) => {
                    self.unions.insert(u.name.clone(), u.clone());
                }
                Item::TypeAlias(t) => {
                    self.aliases.insert(t.name.clone(), t.clone());
                }
                Item::Const(c) => {
                    self.consts.insert(c.name.clone(), c.value.clone());
                }
                Item::Impl(imp) => {
                    if let Some(iface) = &imp.interface {
                        let tname = imp.target_ty.display_name();
                        let iname = iface.display_name();
                        self.impls.insert((tname.clone(), iname.clone()));
                        if let Some(base) = tname.split('[').next() {
                            self.impls.insert((base.to_string(), iname));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn type_implements(&self, ty: &TypeExpr, iface: &str) -> bool {
        let full = ty.display_name();
        if self.impls.contains(&(full.clone(), iface.to_string())) {
            return true;
        }
        if let TypeExpr::Named { name, .. } = ty {
            if self.impls.contains(&(name.clone(), iface.to_string())) {
                return true;
            }
            if let Some(ta) = self.aliases.get(name) {
                return self.type_implements(&ta.base, iface);
            }
        }
        false
    }

    pub fn type_kind_str(&self, ty: &TypeExpr) -> String {
        match ty {
            TypeExpr::Tuple(_, _) => "Tuple".into(),
            TypeExpr::Fn { .. } => "Fn".into(),
            TypeExpr::Dyn { .. } => "Dyn".into(),
            TypeExpr::Array { .. } => "Array".into(),
            TypeExpr::Named { name, .. } => match name.as_str() {
                "Int" | "Int8" | "Int16" | "Int32" | "Int64" | "ISize" | "CChar" | "CInt" => {
                    "Int".into()
                }
                "UInt8" | "UInt16" | "UInt32" | "UInt64" | "USize" | "Byte" | "CUInt" | "CSize" => {
                    "UInt".into()
                }
                "Float" | "Float32" | "Float64" => "Float".into(),
                "Bool" => "Bool".into(),
                "Text" => "Text".into(),
                "List" => "List".into(),
                "Option" => "Option".into(),
                "Result" => "Result".into(),
                "Ptr" | "MutPtr" | "CStr" => "Ptr".into(),
                other => {
                    if self.records.contains_key(other) {
                        "Record".into()
                    } else if self.enums.contains_key(other) {
                        "Enum".into()
                    } else if self.bitfields.contains_key(other) {
                        "Bitfield".into()
                    } else if self.unions.contains_key(other) {
                        "Union".into()
                    } else if let Some(ta) = self.aliases.get(other) {
                        if ta.predicate.is_some() {
                            "Refined".into()
                        } else {
                            self.type_kind_str(&ta.base)
                        }
                    } else {
                        "Unknown".into()
                    }
                }
            },
        }
    }

    fn fields_of_type(&self, ty: &TypeExpr) -> Vec<CtVal> {
        let TypeExpr::Named { name, .. } = ty else {
            return Vec::new();
        };
        let resolved_name = if let Some(ta) = self.aliases.get(name) {
            ta.base.display_name()
        } else {
            name.clone()
        };
        if let Some(rec) = self.records.get(&resolved_name) {
            return rec
                .fields
                .iter()
                .enumerate()
                .map(|(idx, f)| {
                    let mut m = BTreeMap::new();
                    m.insert("name".into(), CtVal::Text(f.name.clone()));
                    m.insert("ty".into(), CtVal::Text(f.ty.display_name()));
                    m.insert("index".into(), CtVal::Int(idx as i64));
                    m.insert(
                        "has_refinement".into(),
                        CtVal::Bool(
                            f.predicate.is_some()
                                || self
                                    .aliases
                                    .get(&f.ty.display_name())
                                    .is_some_and(|a| a.predicate.is_some()),
                        ),
                    );
                    CtVal::Record {
                        name: "FieldInfo".into(),
                        fields: m,
                    }
                })
                .collect();
        }
        Vec::new()
    }

    fn type_info_val(&self, ty: &TypeExpr) -> CtVal {
        let mut m = BTreeMap::new();
        m.insert("name".into(), CtVal::Text(ty.display_name()));
        m.insert("kind".into(), CtVal::Text(self.type_kind_str(ty)));
        let fields = self.fields_of_type(ty);
        m.insert("field_count".into(), CtVal::Int(fields.len() as i64));
        m.insert("fields".into(), CtVal::List(fields));
        CtVal::Record {
            name: "TypeInfo".into(),
            fields: m,
        }
    }

    fn routes_of_program(&self) -> Vec<CtVal> {
        let mut out = Vec::new();
        for f in &self.all_fns_ordered {
            for attr in &f.attrs {
                let method = match attr.name.as_str() {
                    "get" => "GET",
                    "post" => "POST",
                    "put" => "PUT",
                    "delete" => "DELETE",
                    "patch" => "PATCH",
                    "head" => "HEAD",
                    "ws" => "GET",
                    "sse" => "GET",
                    _ => continue,
                };
                let path = attr
                    .args
                    .first()
                    .and_then(|a| match &a.value.kind {
                        ExprKind::Str(s) => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| "/".into());
                let pnames: Vec<CtVal> = f
                    .params
                    .iter()
                    .map(|p| CtVal::Text(p.name.clone()))
                    .collect();
                let ptypes: Vec<CtVal> = f
                    .params
                    .iter()
                    .map(|p| CtVal::Text(p.ty.display_name()))
                    .collect();
                let ret_ty = f
                    .ret_ty
                    .as_ref()
                    .map(|t| t.display_name())
                    .unwrap_or_else(|| "Unit".into());
                let mut m = BTreeMap::new();
                m.insert("method".into(), CtVal::Text(method.into()));
                m.insert("path".into(), CtVal::Text(path));
                m.insert("handler".into(), CtVal::Text(f.name.clone()));
                m.insert("param_count".into(), CtVal::Int(f.params.len() as i64));
                m.insert("param_names".into(), CtVal::List(pnames));
                m.insert("param_types".into(), CtVal::List(ptypes));
                m.insert("ret_ty".into(), CtVal::Text(ret_ty));
                out.push(CtVal::Record {
                    name: "RouteInfo".into(),
                    fields: m,
                });
            }
        }
        out
    }

    /// Compile-time refinement subtype check `is_subtype[S, T]()`:
    /// returns `true` when `S == T`, or `T` is the unconstrained base of `S`,
    /// or both `S` and `T` have statically bounded integer intervals with `[S.min, S.max] ⊆ [T.min, T.max]`.
    fn is_refinement_subtype(&self, src: &TypeExpr, dst: &TypeExpr) -> bool {
        let sn = src.display_name();
        let dn = dst.display_name();
        if sn == dn {
            return true;
        }
        let (s_head, s_args): (&str, &[TypeExpr]) = match src {
            TypeExpr::Named { name, args, .. } => (name.as_str(), args.as_slice()),
            _ => (sn.as_str(), &[]),
        };
        let (d_head, d_args): (&str, &[TypeExpr]) = match dst {
            TypeExpr::Named { name, args, .. } => (name.as_str(), args.as_slice()),
            _ => (dn.as_str(), &[]),
        };
        let s_alias = self.aliases.get(s_head);
        let d_alias = self.aliases.get(d_head);
        let s_base = s_alias
            .map(|a| a.base.display_name())
            .unwrap_or_else(|| s_head.to_string());
        let d_base = d_alias
            .map(|a| a.base.display_name())
            .unwrap_or_else(|| d_head.to_string());
        if s_base != d_base {
            // Check exact-width scalar widening (e.g. UInt8 <: UInt16 <: UInt32 <: UInt64)
            if let (Some(sk), Some(dk)) = (
                crate::layout::ScalarKind::from_name(&s_base),
                crate::layout::ScalarKind::from_name(&d_base),
            ) {
                if let (Some(smin), Some(smax), Some(dmin), Some(dmax)) =
                    (sk.min_i128(), sk.max_i128(), dk.min_i128(), dk.max_i128())
                {
                    if smin >= dmin && smax <= dmax && d_alias.and_then(|a| a.predicate.as_ref()).is_none() {
                        return true;
                    }
                }
            }
            return false;
        }
        // Target has no refinement predicate -> widening to base always holds!
        let Some(d_pred) = d_alias.and_then(|a| crate::instantiate_alias_predicate(a, d_args)) else {
            return true;
        };
        let Some(s_pred) = s_alias.and_then(|a| crate::instantiate_alias_predicate(a, s_args)) else {
            return false;
        };
        if let (Some((smin, smax)), Some((dmin, dmax))) = (
            extract_simple_int_bounds(&s_pred),
            extract_simple_int_bounds(&d_pred),
        ) {
            return smin >= dmin && smax <= dmax;
        }
        false
    }

    fn annotated_fns_of_program(&self, target_attr: &str) -> Vec<CtVal> {
        let mut out = Vec::new();
        for f in &self.all_fns_ordered {
            for attr in &f.attrs {
                if attr.name == target_attr {
                    let arg0 = attr
                        .args
                        .first()
                        .and_then(|a| match &a.value.kind {
                            ExprKind::Str(s) => Some(s.clone()),
                            ExprKind::Ident(s) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    let mut m = BTreeMap::new();
                    m.insert("name".into(), CtVal::Text(f.name.clone()));
                    m.insert("attr".into(), CtVal::Text(attr.name.clone()));
                    m.insert("arg".into(), CtVal::Text(arg0));
                    m.insert("param_count".into(), CtVal::Int(f.params.len() as i64));
                    out.push(CtVal::Record {
                        name: "AnnotatedFn".into(),
                        fields: m,
                    });
                }
            }
        }
        out
    }
}

/// Expand all top-level `comptime { ... }`, top-level `macro!(...)`, expression-level `comptime { ... }`,
/// `comptime if`, `comptime for`, and `name!(...)` macro calls across `program`.
/// Generic `comptime if` / `comptime for` depending on unresolved type parameters are safely
/// preserved for expansion during generic monomorphization (`eval_comptime_in_fn`).
pub fn expand_comptime_and_macros(program: &mut Program) -> Result<(), String> {
    // Pass 1: expand top-level `comptime { ... }` and top-level `name!(...)` (up to 4 rounds for generated items)
    for _round in 0..4 {
        let ctx = ComptimeContext::from_program(program);
        let has_top_ct = program.items.iter().any(|i| {
            matches!(i, Item::ComptimeBlock(_) | Item::MacroCall { .. })
        });
        if !has_top_ct {
            break;
        }
        let old_items = std::mem::take(&mut program.items);
        let mut new_items = Vec::with_capacity(old_items.len());
        for item in old_items {
            match item {
                Item::ComptimeBlock(blk) => {
                    let fid = blk.span.file_id;
                    let mut eval = Evaluator::new(&ctx);
                    let val = eval.eval_block(&blk)?;
                    let generated = splice_items_from_ctval(&val, fid)?;
                    new_items.extend(generated);
                }
                Item::MacroCall { name, args, span } => {
                    let generated = expand_top_level_macro(&ctx, &name, &args, span)?;
                    new_items.extend(generated);
                }
                other => new_items.push(other),
            }
        }
        program.items = new_items;
    }

    // Pass 2: expand expressions/statements/predicates inside all items
    let ctx = ComptimeContext::from_program(program);
    for item in &mut program.items {
        expand_item(&ctx, item)?;
    }
    // Move declarative `Item::Macro` and compile-time-only `comptime fn` definitions out of
    // runtime `program.items` into `program.comptime_items` so codegen never emits them while
    // LSP / IDE navigation retains exact spans & signatures!
    let mut kept = Vec::with_capacity(program.items.len());
    for item in std::mem::take(&mut program.items) {
        match item {
            Item::Macro(_) => program.comptime_items.push(item),
            Item::Fn(ref f) if f.is_comptime => program.comptime_items.push(item),
            other => kept.push(other),
        }
    }
    program.items = kept;
    Ok(())
}

/// Expand any remaining `comptime if` / `comptime for` / `comptime { ... }` / `name!(...)` inside
/// a freshly monomorphized function (called from `tenon-codegen` after substituting generic `T`).
pub fn eval_comptime_in_fn(ctx: &ComptimeContext, f: &mut FnDecl) -> Result<(), String> {
    let type_params: BTreeSet<String> = f.type_params.iter().cloned().collect();
    let mut var_types = BTreeMap::new();
    for p in &f.params {
        var_types.insert(p.name.clone(), p.ty.clone());
    }
    for r in &mut f.requires {
        expand_expr(ctx, r, &type_params, &var_types, 0)?;
    }
    for e in &mut f.ensures {
        expand_expr(ctx, e, &type_params, &var_types, 0)?;
    }
    expand_block(ctx, &mut f.body, &type_params, &mut var_types, 0)
}

fn splice_items_from_ctval(val: &CtVal, file_id: u32) -> Result<Vec<Item>, String> {
    match val {
        CtVal::Text(src) => {
            if src.trim().is_empty() {
                return Ok(Vec::new());
            }
            let parsed = parse_with_file_id(src, file_id)
                .map_err(|e| format!("comptime generated invalid Tenon items ({}):\n{}", e, src))?;
            Ok(parsed.items)
        }
        CtVal::List(elems) => {
            let mut out = Vec::new();
            for e in elems {
                out.extend(splice_items_from_ctval(e, file_id)?);
            }
            Ok(out)
        }
        CtVal::Unit => Ok(Vec::new()),
        other => Err(format!(
            "top-level `comptime` block must return `Text` or `List[Text]` of generated code, got {:?}",
            other
        )),
    }
}

fn expand_top_level_macro(
    ctx: &ComptimeContext,
    name: &str,
    args: &[Expr],
    span: Span,
) -> Result<Vec<Item>, String> {
    if let Some(fdecl) = ctx.fns.get(name) {
        let mut eval = Evaluator::new(ctx);
        let mut ct_args = Vec::new();
        for a in args {
            ct_args.push(eval.eval_expr(a)?);
        }
        let val = eval.call_fn(fdecl, ct_args)?;
        return splice_items_from_ctval(&val, span.file_id);
    }
    if let Some(mdecl) = ctx.macros.get(name) {
        if mdecl.params.len() != args.len() {
            return Err(format!(
                "macro `{}` expects {} argument(s), got {}",
                name,
                mdecl.params.len(),
                args.len()
            ));
        }
        let mut subst = BTreeMap::new();
        for (p, a) in mdecl.params.iter().zip(args.iter()) {
            subst.insert(p.clone(), a.clone());
        }
        let mut body = mdecl.body.clone();
        subst_expr_params(&mut body, &subst);
        let mut eval = Evaluator::new(ctx);
        let val = eval.eval_expr(&body)?;
        return splice_items_from_ctval(&val, span.file_id);
    }
    Err(format!("unknown top-level macro `{}!` at {}", name, span))
}

fn expand_item(ctx: &ComptimeContext, item: &mut Item) -> Result<(), String> {
    let empty_tp = BTreeSet::new();
    match item {
        Item::Fn(f) => {
            let tp: BTreeSet<String> = f.type_params.iter().cloned().collect();
            let mut vt = BTreeMap::new();
            for p in &mut f.params {
                vt.insert(p.name.clone(), p.ty.clone());
                if let Some(pr) = &mut p.predicate {
                    expand_expr(ctx, pr, &tp, &vt, 0)?;
                }
            }
            for r in &mut f.requires {
                expand_expr(ctx, r, &tp, &vt, 0)?;
            }
            for e in &mut f.ensures {
                expand_expr(ctx, e, &tp, &vt, 0)?;
            }
            expand_block(ctx, &mut f.body, &tp, &mut vt, 0)?;
        }
        Item::TypeAlias(ta) => {
            let mut vt = BTreeMap::new();
            vt.insert("self".into(), ta.base.clone());
            if let Some(p) = &mut ta.predicate {
                expand_expr(ctx, p, &empty_tp, &vt, 0)?;
            }
        }
        Item::Record(r) => {
            let tp: BTreeSet<String> = r.type_params.iter().cloned().collect();
            let mut vt = BTreeMap::new();
            for f in &r.fields {
                vt.insert(f.name.clone(), f.ty.clone());
            }
            for f in &mut r.fields {
                let mut fvt = vt.clone();
                fvt.insert("self".into(), f.ty.clone());
                if let Some(p) = &mut f.predicate {
                    expand_expr(ctx, p, &tp, &fvt, 0)?;
                }
                if let Some(d) = &mut f.default {
                    expand_expr(ctx, d, &tp, &vt, 0)?;
                }
            }
            for w in &mut r.where_clauses {
                expand_expr(ctx, w, &tp, &vt, 0)?;
            }
            for m in &mut r.methods {
                let mut mtp = tp.clone();
                mtp.extend(m.type_params.iter().cloned());
                let mut mvt = BTreeMap::new();
                for p in &m.params {
                    mvt.insert(p.name.clone(), p.ty.clone());
                }
                expand_block(ctx, &mut m.body, &mtp, &mut mvt, 0)?;
            }
        }
        Item::Impl(imp) => {
            let mut tp: BTreeSet<String> = imp.type_params.iter().cloned().collect();
            for m in &mut imp.methods {
                let mut mtp = tp.clone();
                mtp.extend(m.type_params.iter().cloned());
                let mut mvt = BTreeMap::new();
                mvt.insert("self".into(), imp.target_ty.clone());
                for p in &m.params {
                    mvt.insert(p.name.clone(), p.ty.clone());
                }
                for r in &mut m.requires {
                    expand_expr(ctx, r, &mtp, &mvt, 0)?;
                }
                for e in &mut m.ensures {
                    expand_expr(ctx, e, &mtp, &mvt, 0)?;
                }
                expand_block(ctx, &mut m.body, &mtp, &mut mvt, 0)?;
            }
            let _ = &mut tp;
        }
        Item::Const(c) => {
            let vt = BTreeMap::new();
            expand_expr(ctx, &mut c.value, &empty_tp, &vt, 0)?;
        }
        _ => {}
    }
    Ok(())
}

fn expand_block(
    ctx: &ComptimeContext,
    blk: &mut Block,
    type_params: &BTreeSet<String>,
    var_types: &mut BTreeMap<String, TypeExpr>,
    depth: usize,
) -> Result<(), String> {
    let saved = var_types.clone();
    let old_stmts = std::mem::take(&mut blk.stmts);
    let mut new_stmts = Vec::with_capacity(old_stmts.len());
    for mut s in old_stmts {
        match &mut s {
            Stmt::Let { name, ty, value, .. } => {
                expand_expr(ctx, value, type_params, var_types, depth)?;
                let eff = ty
                    .clone()
                    .unwrap_or_else(|| quick_infer_expr_ty(ctx, value, var_types));
                var_types.insert(name.clone(), eff);
                new_stmts.push(s);
            }
            Stmt::LetTuple { names, ty, value, .. } => {
                expand_expr(ctx, value, type_params, var_types, depth)?;
                if let Some(TypeExpr::Tuple(ts, _)) = ty {
                    for (n, t) in names.iter().zip(ts.iter()) {
                        var_types.insert(n.clone(), t.clone());
                    }
                }
                new_stmts.push(s);
            }
            Stmt::Assign { target, value, .. } => {
                expand_expr(ctx, target, type_params, var_types, depth)?;
                expand_expr(ctx, value, type_params, var_types, depth)?;
                new_stmts.push(s);
            }
            Stmt::Return { value: Some(e), .. } | Stmt::Assert { cond: e, .. } => {
                expand_expr(ctx, e, type_params, var_types, depth)?;
                new_stmts.push(s);
            }
            Stmt::Expr { expr, span } => {
                // If `comptime for` is a statement, unroll into flat statements directly!
                if let ExprKind::ComptimeFor { binding, iter, body } = &expr.kind {
                    if !expr_mentions_type_params(iter, type_params) {
                        let unrolled =
                            expand_comptime_for_stmts(ctx, binding, iter, body, type_params, var_types, depth)?;
                        new_stmts.extend(unrolled);
                        continue;
                    }
                }
                expand_expr(ctx, expr, type_params, var_types, depth)?;
                // Flatten `Stmt::Expr(Block(stmts, None))` produced by `comptime if` / `comptime for`
                if let ExprKind::Block(inner) = &expr.kind {
                    if inner.tail.is_none() && inner.stmts.is_empty() {
                        continue;
                    }
                }
                let _ = span;
                new_stmts.push(s);
            }
            Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } | Stmt::Ghost { body, .. } => {
                expand_block(ctx, body, type_params, var_types, depth)?;
                new_stmts.push(s);
            }
            Stmt::While { cond, body, .. } => {
                expand_expr(ctx, cond, type_params, var_types, depth)?;
                expand_block(ctx, body, type_params, var_types, depth)?;
                new_stmts.push(s);
            }
            Stmt::For { iter, body, .. } => {
                expand_expr(ctx, iter, type_params, var_types, depth)?;
                expand_block(ctx, body, type_params, var_types, depth)?;
                new_stmts.push(s);
            }
            _ => new_stmts.push(s),
        }
    }
    blk.stmts = new_stmts;
    if let Some(tail) = &mut blk.tail {
        expand_expr(ctx, tail, type_params, var_types, depth)?;
    }
    *var_types = saved;
    Ok(())
}

fn expand_expr(
    ctx: &ComptimeContext,
    expr: &mut Expr,
    type_params: &BTreeSet<String>,
    var_types: &BTreeMap<String, TypeExpr>,
    depth: usize,
) -> Result<(), String> {
    if depth > MAX_MACRO_DEPTH {
        return Err(format!("comptime/macro expansion depth limit ({}) exceeded", MAX_MACRO_DEPTH));
    }
    // Check direct compile-time reflection & build-config calls before mutably borrowing `expr.kind`
    let should_fold_call = match &expr.kind {
        ExprKind::Call { callee, .. } => match &callee.kind {
            ExprKind::GenericInst {
                expr: inner,
                type_args,
            } => matches!(
                &inner.kind,
                ExprKind::Ident(fname)
                    if matches!(
                        fname.as_str(),
                        "type_name" | "type_kind" | "field_count" | "field_names" | "has_field"
                    )
            ) && !type_args.iter().any(|t| type_mentions_type_params(t, type_params)),
            ExprKind::Ident(fname) => {
                matches!(fname.as_str(), "config" | "target_os" | "target_arch")
            }
            _ => false,
        },
        _ => false,
    };
    if should_fold_call {
        let sp = expr.span;
        let mut eval = Evaluator::new(ctx);
        if let Ok(val) = eval.eval_expr(expr) {
            *expr = ctval_to_ast_expr(&val, sp)?;
            return Ok(());
        }
    }
    match &mut expr.kind {
        ExprKind::Comptime(blk) => {
            if block_mentions_type_params(blk, type_params) {
                return Ok(());
            }
            let sp = expr.span;
            let mut eval = Evaluator::new(ctx);
            let val = eval.eval_block(blk)?;
            *expr = ctval_to_ast_expr(&val, sp)?;
            return Ok(());
        }
        ExprKind::ComptimeIf {
            cond,
            then_block,
            else_branch,
        } => {
            if expr_mentions_type_params(cond, type_params) {
                // Defer to generic monomorphization! Do NOT expand unselected branches prematurely!
                return Ok(());
            }
            let sp = expr.span;
            let mut eval = Evaluator::new(ctx);
            let cond_val = eval.eval_expr(cond)?;
            let take_then = match cond_val {
                CtVal::Bool(b) => b,
                other => {
                    return Err(format!(
                        "`comptime if` condition must evaluate to `Bool`, got {:?} at {}",
                        other, cond.span
                    ));
                }
            };
            if take_then {
                let mut chosen = then_block.clone();
                let mut vt = var_types.clone();
                expand_block(ctx, &mut chosen, type_params, &mut vt, depth + 1)?;
                *expr = Expr {
                    kind: ExprKind::Block(chosen),
                    span: sp,
                };
            } else if let Some(eb) = else_branch.take() {
                let mut chosen = *eb;
                expand_expr(ctx, &mut chosen, type_params, var_types, depth + 1)?;
                *expr = chosen;
            } else {
                *expr = Expr {
                    kind: ExprKind::Block(Block {
                        stmts: Vec::new(),
                        tail: None,
                        span: sp,
                    }),
                    span: sp,
                };
            }
            return Ok(());
        }
        ExprKind::ComptimeFor {
            binding,
            iter,
            body,
        } => {
            if expr_mentions_type_params(iter, type_params) {
                return Ok(());
            }
            let sp = expr.span;
            let mut vt = var_types.clone();
            let stmts = expand_comptime_for_stmts(ctx, binding, iter, body, type_params, &mut vt, depth)?;
            *expr = Expr {
                kind: ExprKind::Block(Block {
                    stmts,
                    tail: None,
                    span: sp,
                }),
                span: sp,
            };
            return Ok(());
        }
        ExprKind::MacroCall { name, args } => {
            // First expand any nested macros/comptime inside args
            for a in args.iter_mut() {
                expand_expr(ctx, a, type_params, var_types, depth + 1)?;
            }
            if args.iter().any(|a| expr_mentions_type_params(a, type_params)) && ctx.fns.contains_key(name.as_str()) {
                // Typed generator macro invoked inside generic fn on type param `T`: defer until monomorphization!
                return Ok(());
            }
            let mname = name.clone();
            let margs = args.clone();
            let sp = expr.span;
            let mut expanded = expand_expr_macro(ctx, &mname, &margs, var_types, sp)?;
            expand_expr(ctx, &mut expanded, type_params, var_types, depth + 1)?;
            *expr = expanded;
            return Ok(());
        }
        ExprKind::TypeImplements { ty, interface } => {
            if !type_mentions_type_params(ty, type_params) {
                let b = ctx.type_implements(ty, interface);
                *expr = Expr {
                    kind: ExprKind::Bool(b),
                    span: expr.span,
                };
            }
            return Ok(());
        }
        ExprKind::Call { callee, args } => {
            expand_expr(ctx, callee, type_params, var_types, depth)?;
            for a in args {
                expand_expr(ctx, &mut a.value, type_params, var_types, depth)?;
            }
        }
        ExprKind::Binary { left, right, .. } => {
            expand_expr(ctx, left, type_params, var_types, depth)?;
            expand_expr(ctx, right, type_params, var_types, depth)?;
        }
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Await(inner)
        | ExprKind::Old(inner)
        | ExprKind::Try(inner)
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::TryCast { expr: inner, .. }
        | ExprKind::ArrayRepeat { value: inner, .. }
        | ExprKind::Field { target: inner, .. }
        | ExprKind::TupleIndex { target: inner, .. }
        | ExprKind::GenericInst { expr: inner, .. } => {
            expand_expr(ctx, inner, type_params, var_types, depth)?;
        }
        ExprKind::Range { start, end } => {
            expand_expr(ctx, start, type_params, var_types, depth)?;
            expand_expr(ctx, end, type_params, var_types, depth)?;
        }
        ExprKind::Lambda { body, .. } => {
            expand_expr(ctx, body, type_params, var_types, depth)?;
        }
        ExprKind::Index { target, index } => {
            expand_expr(ctx, target, type_params, var_types, depth)?;
            expand_expr(ctx, index, type_params, var_types, depth)?;
        }
        ExprKind::RecordLit { fields, .. } => {
            for (_, v) in fields {
                expand_expr(ctx, v, type_params, var_types, depth)?;
            }
        }
        ExprKind::Tuple(es) | ExprKind::List(es) => {
            for e in es {
                expand_expr(ctx, e, type_params, var_types, depth)?;
            }
        }
        ExprKind::If {
            cond,
            then_block,
            else_branch,
        } => {
            expand_expr(ctx, cond, type_params, var_types, depth)?;
            let mut vt = var_types.clone();
            expand_block(ctx, then_block, type_params, &mut vt, depth)?;
            if let Some(eb) = else_branch {
                expand_expr(ctx, eb, type_params, var_types, depth)?;
            }
        }
        ExprKind::Match { scrutinee, arms } => {
            expand_expr(ctx, scrutinee, type_params, var_types, depth)?;
            for a in arms {
                if let Some(g) = &mut a.guard {
                    expand_expr(ctx, g, type_params, var_types, depth)?;
                }
                expand_expr(ctx, &mut a.body, type_params, var_types, depth)?;
            }
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) => {
            let mut vt = var_types.clone();
            expand_block(ctx, b, type_params, &mut vt, depth)?;
        }
        _ => {}
    }
    Ok(())
}

fn expand_comptime_for_stmts(
    ctx: &ComptimeContext,
    binding: &str,
    iter: &Expr,
    body: &Block,
    type_params: &BTreeSet<String>,
    var_types: &mut BTreeMap<String, TypeExpr>,
    depth: usize,
) -> Result<Vec<Stmt>, String> {
    let mut eval = Evaluator::new(ctx);
    let iter_val = eval.eval_expr(iter)?;
    let items = match iter_val {
        CtVal::List(xs) => xs,
        other => {
            return Err(format!(
                "`comptime for` iterable must evaluate to a `List`, got {:?} at {}",
                other, iter.span
            ));
        }
    };
    let mut out = Vec::new();
    for item_val in items {
        let mut iter_blk = body.clone();
        subst_ctval_in_block(&mut iter_blk, binding, &item_val);
        expand_block(ctx, &mut iter_blk, type_params, var_types, depth + 1)?;
        out.extend(iter_blk.stmts);
        if let Some(tail) = iter_blk.tail {
            let sp = tail.span;
            out.push(Stmt::Expr {
                expr: *tail,
                span: sp,
            });
        }
    }
    Ok(out)
}

fn expand_expr_macro(
    ctx: &ComptimeContext,
    name: &str,
    args: &[Expr],
    var_types: &BTreeMap<String, TypeExpr>,
    span: Span,
) -> Result<Expr, String> {
    // 1. Declarative macro: `macro name(p1, p2) { body }`
    if let Some(mdecl) = ctx.macros.get(name) {
        if mdecl.params.len() != args.len() {
            return Err(format!(
                "macro `{}!` expects {} argument(s), got {} at {}",
                name,
                mdecl.params.len(),
                args.len(),
                span
            ));
        }
        let mut subst = BTreeMap::new();
        for (p, a) in mdecl.params.iter().zip(args.iter()) {
            subst.insert(p.clone(), a.clone());
        }
        let mut body = mdecl.body.clone();
        subst_expr_params(&mut body, &subst);
        return Ok(body);
    }
    // 2. Loom-style typed generator function macro:
    //    `fn twice(types List[TypeInfo]) Text { "$0 + $0" }` invoked as `twice!(x)`
    if let Some(fdecl) = ctx.fns.get(name) {
        let mut eval = Evaluator::new(ctx);
        let result_val = if fdecl.params.len() == 1
            && fdecl.params[0].ty.display_name().starts_with("List[")
        {
            let type_infos: Vec<CtVal> = args
                .iter()
                .map(|a| {
                    let ty = quick_infer_expr_ty(ctx, a, var_types);
                    ctx.type_info_val(&ty)
                })
                .collect();
            eval.call_fn(fdecl, vec![CtVal::List(type_infos)])?
        } else if fdecl.params.is_empty() && args.is_empty() {
            eval.call_fn(fdecl, Vec::new())?
        } else {
            let mut ct_args = Vec::with_capacity(args.len());
            for a in args {
                ct_args.push(eval.eval_expr(a)?);
            }
            eval.call_fn(fdecl, ct_args)?
        };
        let CtVal::Text(mut src) = result_val else {
            return ctval_to_ast_expr(&result_val, span);
        };
        // Splice `$0`, `$1`, ... placeholders with argument source strings (reverse order so `$10` precedes `$1`!)
        for idx in (0..args.len()).rev() {
            let placeholder = format!("${}", idx);
            if src.contains(&placeholder) {
                let arg_src = format_expr_src(&args[idx]);
                src = src.replace(&placeholder, &arg_src);
            }
        }
        return parse_expr_str(&src, span.file_id).map_err(|e| {
            format!(
                "macro `{}!` produced invalid Tenon expression ({}):\n{}",
                name, e, src
            )
        });
    }
    Err(format!("unknown macro `{}!` at {}", name, span))
}

// =========================================================================
// Pure Tenon Compile-Time Evaluator
// =========================================================================

struct Evaluator<'a> {
    ctx: &'a ComptimeContext,
    steps: usize,
    depth: usize,
}

impl<'a> Evaluator<'a> {
    fn new(ctx: &'a ComptimeContext) -> Self {
        Self {
            ctx,
            steps: 0,
            depth: 0,
        }
    }

    fn tick(&mut self) -> Result<(), Flow> {
        self.steps += 1;
        if self.steps > MAX_STEPS {
            return Err(Flow::Err(format!(
                "comptime evaluation exceeded step limit ({})",
                MAX_STEPS
            )));
        }
        Ok(())
    }

    fn eval_block(&mut self, blk: &Block) -> Result<CtVal, String> {
        let mut env = BTreeMap::new();
        match self.eval_block_env(blk, &mut env) {
            Ok(v) | Err(Flow::Return(v)) => Ok(v),
            Err(Flow::Err(e)) => Err(e),
            Err(Flow::Break) => Err("`break` outside loop in comptime".into()),
            Err(Flow::Continue) => Err("`continue` outside loop in comptime".into()),
        }
    }

    fn eval_expr(&mut self, expr: &Expr) -> Result<CtVal, String> {
        let mut env = BTreeMap::new();
        match self.eval_expr_env(expr, &mut env) {
            Ok(v) | Err(Flow::Return(v)) => Ok(v),
            Err(Flow::Err(e)) => Err(e),
            Err(Flow::Break | Flow::Continue) => Err("unexpected loop control in comptime expr".into()),
        }
    }

    fn call_fn(&mut self, f: &FnDecl, args: Vec<CtVal>) -> Result<CtVal, String> {
        if self.depth >= MAX_CALL_DEPTH {
            return Err(format!("comptime recursion depth limit ({}) exceeded", MAX_CALL_DEPTH));
        }
        if f.params.len() != args.len() {
            return Err(format!(
                "comptime call to `{}` expects {} arg(s), got {}",
                f.name,
                f.params.len(),
                args.len()
            ));
        }
        self.depth += 1;
        let mut env = BTreeMap::new();
        for (p, v) in f.params.iter().zip(args.into_iter()) {
            env.insert(p.name.clone(), v);
        }
        let res = match self.eval_block_env(&f.body, &mut env) {
            Ok(v) | Err(Flow::Return(v)) => Ok(v),
            Err(Flow::Err(e)) => Err(e),
            Err(Flow::Break | Flow::Continue) => Err("unexpected loop control in comptime fn".into()),
        };
        self.depth -= 1;
        res
    }

    fn eval_block_env(
        &mut self,
        blk: &Block,
        env: &mut BTreeMap<String, CtVal>,
    ) -> Result<CtVal, Flow> {
        let saved_keys: BTreeSet<String> = env.keys().cloned().collect();
        let mut last = CtVal::Unit;
        for stmt in &blk.stmts {
            self.tick()?;
            match stmt {
                Stmt::Let { name, value, .. } => {
                    let v = self.eval_expr_env(value, env)?;
                    env.insert(name.clone(), v);
                }
                Stmt::LetTuple { names, value, .. } => {
                    let v = self.eval_expr_env(value, env)?;
                    let CtVal::Tuple(elems) = v else {
                        return Err(Flow::Err("expected tuple in comptime destructuring".into()));
                    };
                    for (n, e) in names.iter().zip(elems.into_iter()) {
                        if n != "_" {
                            env.insert(n.clone(), e);
                        }
                    }
                }
                Stmt::Assign { target, op, value, .. } => {
                    let rhs = self.eval_expr_env(value, env)?;
                    match &target.kind {
                        ExprKind::Ident(name) => {
                            let new_v = if *op == AssignOp::Set {
                                rhs
                            } else {
                                let cur = env
                                    .get(name)
                                    .cloned()
                                    .ok_or_else(|| Flow::Err(format!("undefined var `{}` in comptime", name)))?;
                                apply_assign_op(cur, *op, rhs)?
                            };
                            env.insert(name.clone(), new_v);
                        }
                        ExprKind::Index { target: arr, index } => {
                            if let ExprKind::Ident(aname) = &arr.kind {
                                let idx = expect_int(&self.eval_expr_env(index, env)?)? as usize;
                                if let Some(CtVal::List(xs)) = env.get_mut(aname) {
                                    if idx < xs.len() {
                                        xs[idx] = rhs;
                                    }
                                }
                            }
                        }
                        ExprKind::Field { target: obj, field } => {
                            if let ExprKind::Ident(oname) = &obj.kind {
                                if let Some(CtVal::Record { fields, .. }) = env.get_mut(oname) {
                                    fields.insert(field.clone(), rhs);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Stmt::Return { value, .. } => {
                    let v = match value {
                        Some(e) => self.eval_expr_env(e, env)?,
                        None => CtVal::Unit,
                    };
                    return Err(Flow::Return(v));
                }
                Stmt::Break { .. } => return Err(Flow::Break),
                Stmt::Continue { .. } => return Err(Flow::Continue),
                Stmt::Assert { cond, span } => {
                    let v = self.eval_expr_env(cond, env)?;
                    if !matches!(v, CtVal::Bool(true)) {
                        return Err(Flow::Err(format!("comptime assertion failed at {}", span)));
                    }
                }
                Stmt::While { cond, body, .. } => loop {
                    self.tick()?;
                    let c = self.eval_expr_env(cond, env)?;
                    if !matches!(c, CtVal::Bool(true)) {
                        break;
                    }
                    match self.eval_block_env(body, env) {
                        Ok(_) | Err(Flow::Continue) => {}
                        Err(Flow::Break) => break,
                        Err(other) => return Err(other),
                    }
                },
                Stmt::For { binding, iter, body, .. } => {
                    let items: Vec<CtVal> = match &iter.kind {
                        ExprKind::Range { start, end } => {
                            let s = expect_int(&self.eval_expr_env(start, env)?)?;
                            let e = expect_int(&self.eval_expr_env(end, env)?)?;
                            (s..e).map(CtVal::Int).collect()
                        }
                        _ => match self.eval_expr_env(iter, env)? {
                            CtVal::List(xs) => xs,
                            other => {
                                return Err(Flow::Err(format!(
                                    "cannot iterate over {:?} in comptime `for`",
                                    other
                                )));
                            }
                        },
                    };
                    for item in items {
                        self.tick()?;
                        match binding {
                            ForBinding::Ident(n) if n != "_" => {
                                env.insert(n.clone(), item);
                            }
                            ForBinding::Tuple(ns) => {
                                if let CtVal::Tuple(es) = item {
                                    for (n, e) in ns.iter().zip(es.into_iter()) {
                                        if n != "_" {
                                            env.insert(n.clone(), e);
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                        match self.eval_block_env(body, env) {
                            Ok(_) | Err(Flow::Continue) => {}
                            Err(Flow::Break) => break,
                            Err(other) => return Err(other),
                        }
                    }
                }
                Stmt::Expr { expr, .. } => {
                    last = self.eval_expr_env(expr, env)?;
                }
                Stmt::Mutate { body, .. } => {
                    last = self.eval_block_env(body, env)?;
                }
                Stmt::Defer { .. } | Stmt::Ghost { .. } => {}
            }
        }
        if let Some(tail) = &blk.tail {
            last = self.eval_expr_env(tail, env)?;
        }
        // Remove block-local newly introduced bindings while keeping updates to outer vars!
        let to_remove: Vec<String> = env
            .keys()
            .filter(|k| !saved_keys.contains(*k))
            .cloned()
            .collect();
        for k in to_remove {
            env.remove(&k);
        }
        Ok(last)
    }

    fn eval_expr_env(
        &mut self,
        expr: &Expr,
        env: &mut BTreeMap<String, CtVal>,
    ) -> Result<CtVal, Flow> {
        self.tick()?;
        match &expr.kind {
            ExprKind::Int(n) => Ok(CtVal::Int(*n)),
            ExprKind::Float(x) => Ok(CtVal::Float(*x)),
            ExprKind::Bool(b) => Ok(CtVal::Bool(*b)),
            ExprKind::Str(s) => Ok(CtVal::Text(s.clone())),
            ExprKind::Ident(name) => {
                if let Some(v) = env.get(name) {
                    return Ok(v.clone());
                }
                if let Some(c) = self.ctx.consts.get(name) {
                    return self.eval_expr_env(c, env);
                }
                if name == "None" {
                    return Ok(CtVal::Enum {
                        name: "Option".into(),
                        variant: "None".into(),
                        payload: Vec::new(),
                    });
                }
                // Check if `name` is a type name (used in `T == Int`)
                if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
                    return Ok(CtVal::Type(TypeExpr::Named {
                        name: name.clone(),
                        version: None,
                        args: Vec::new(),
                        span: expr.span,
                    }));
                }
                Err(Flow::Err(format!(
                    "undefined identifier `{}` in comptime at {}",
                    name, expr.span
                )))
            }
            ExprKind::TypeImplements { ty, interface } => {
                Ok(CtVal::Bool(self.ctx.type_implements(ty, interface)))
            }
            ExprKind::Unary { op, expr: inner } => {
                let v = self.eval_expr_env(inner, env)?;
                match (op, v) {
                    (UnOp::Neg, CtVal::Int(n)) => Ok(CtVal::Int(-n)),
                    (UnOp::Neg, CtVal::Float(x)) => Ok(CtVal::Float(-x)),
                    (UnOp::Not, CtVal::Bool(b)) => Ok(CtVal::Bool(!b)),
                    (UnOp::BitNot, CtVal::Int(n)) => Ok(CtVal::Int(!n)),
                    other => Err(Flow::Err(format!("invalid comptime unary op: {:?}", other))),
                }
            }
            ExprKind::Binary { op: BinOp::And, left, right } => {
                let l = self.eval_expr_env(left, env)?;
                if !matches!(l, CtVal::Bool(true)) {
                    return Ok(CtVal::Bool(false));
                }
                let r = self.eval_expr_env(right, env)?;
                Ok(CtVal::Bool(matches!(r, CtVal::Bool(true))))
            }
            ExprKind::Binary { op: BinOp::Or, left, right } => {
                let l = self.eval_expr_env(left, env)?;
                if matches!(l, CtVal::Bool(true)) {
                    return Ok(CtVal::Bool(true));
                }
                let r = self.eval_expr_env(right, env)?;
                Ok(CtVal::Bool(matches!(r, CtVal::Bool(true))))
            }
            ExprKind::Binary { op, left, right } => {
                let l = self.eval_expr_env(left, env)?;
                let r = self.eval_expr_env(right, env)?;
                eval_binop(*op, l, r)
            }
            ExprKind::List(es) => {
                let mut xs = Vec::with_capacity(es.len());
                for e in es {
                    xs.push(self.eval_expr_env(e, env)?);
                }
                Ok(CtVal::List(xs))
            }
            ExprKind::Tuple(es) => {
                let mut xs = Vec::with_capacity(es.len());
                for e in es {
                    xs.push(self.eval_expr_env(e, env)?);
                }
                Ok(CtVal::Tuple(xs))
            }
            ExprKind::RecordLit { name, fields, .. } => {
                let mut m = BTreeMap::new();
                for (k, v) in fields {
                    m.insert(k.clone(), self.eval_expr_env(v, env)?);
                }
                Ok(CtVal::Record {
                    name: name.clone(),
                    fields: m,
                })
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    if let Some(ed) = self.ctx.enums.get(ename) {
                        if ed.variants.iter().any(|v| v.name == *field) {
                            return Ok(CtVal::Enum {
                                name: ename.clone(),
                                variant: field.clone(),
                                payload: Vec::new(),
                            });
                        }
                    }
                }
                let v = self.eval_expr_env(target, env)?;
                match v {
                    CtVal::Record { fields, .. } => fields
                        .get(field)
                        .cloned()
                        .ok_or_else(|| Flow::Err(format!("missing field `{}` in comptime record", field))),
                    other => Err(Flow::Err(format!(
                        "cannot access field `{}` on {:?} in comptime",
                        field, other
                    ))),
                }
            }
            ExprKind::TupleIndex { target, index } => {
                let v = self.eval_expr_env(target, env)?;
                match v {
                    CtVal::Tuple(es) => es
                        .get(*index)
                        .cloned()
                        .ok_or_else(|| Flow::Err(format!("tuple index {} out of bounds", index))),
                    other => Err(Flow::Err(format!("expected tuple, got {:?}", other))),
                }
            }
            ExprKind::Index { target, index } => {
                let tv = self.eval_expr_env(target, env)?;
                let iv = expect_int(&self.eval_expr_env(index, env)?)? as usize;
                match tv {
                    CtVal::List(xs) => xs
                        .get(iv)
                        .cloned()
                        .ok_or_else(|| Flow::Err(format!("comptime index {} out of bounds (len={})", iv, xs.len()))),
                    other => Err(Flow::Err(format!("cannot index {:?} in comptime", other))),
                }
            }
            ExprKind::Cast { expr: inner, ty } => {
                let v = self.eval_expr_env(inner, env)?;
                match (&v, ty.display_name().as_str()) {
                    (CtVal::Int(n), "Float" | "Float64" | "Float32") => Ok(CtVal::Float(*n as f64)),
                    (CtVal::Float(x), "Int" | "Int64" | "Int32" | "UInt8" | "UInt32" | "UInt64" | "USize") => {
                        Ok(CtVal::Int(*x as i64))
                    }
                    _ => Ok(v),
                }
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            }
            | ExprKind::ComptimeIf {
                cond,
                then_block,
                else_branch,
            } => {
                let c = self.eval_expr_env(cond, env)?;
                if matches!(c, CtVal::Bool(true)) {
                    self.eval_block_env(then_block, env)
                } else if let Some(eb) = else_branch {
                    self.eval_expr_env(eb, env)
                } else {
                    Ok(CtVal::Unit)
                }
            }
            ExprKind::Block(b) | ExprKind::Unsafe(b) | ExprKind::Comptime(b) => {
                self.eval_block_env(b, env)
            }
            ExprKind::Match { scrutinee, arms } => {
                let sv = self.eval_expr_env(scrutinee, env)?;
                for arm in arms {
                    let mut bindings = BTreeMap::new();
                    let matched = arm
                        .patterns
                        .iter()
                        .any(|p| match_ct_pattern(p, &sv, &mut bindings));
                    if matched {
                        let mut arm_env = env.clone();
                        for (k, v) in bindings {
                            arm_env.insert(k, v);
                        }
                        if let Some(g) = &arm.guard {
                            if !matches!(self.eval_expr_env(g, &mut arm_env)?, CtVal::Bool(true)) {
                                continue;
                            }
                        }
                        return self.eval_expr_env(&arm.body, &mut arm_env);
                    }
                }
                Err(Flow::Err("non-exhaustive match in comptime".into()))
            }
            ExprKind::Call { callee, args } => self.eval_call(callee, args, env),
            other => Err(Flow::Err(format!(
                "unsupported expression in comptime at {}: {:?}",
                expr.span, other
            ))),
        }
    }

    fn eval_call(
        &mut self,
        callee: &Expr,
        args: &[CallArg],
        env: &mut BTreeMap<String, CtVal>,
    ) -> Result<CtVal, Flow> {
        // 1. Generic reflection calls: `type_name[T]()`, `fields_of[T]()`, etc.
        if let ExprKind::GenericInst {
            expr: inner,
            type_args,
        } = &callee.kind
        {
            if let ExprKind::Ident(fname) = &inner.kind {
                if let Some(ty) = type_args.first() {
                    match fname.as_str() {
                        "type_name" => return Ok(CtVal::Text(ty.display_name())),
                        "type_kind" => return Ok(CtVal::Text(self.ctx.type_kind_str(ty))),
                        "fields_of" => return Ok(CtVal::List(self.ctx.fields_of_type(ty))),
                        "field_count" => {
                            return Ok(CtVal::Int(self.ctx.fields_of_type(ty).len() as i64));
                        }
                        "field_names" => {
                            let names = self
                                .ctx
                                .fields_of_type(ty)
                                .into_iter()
                                .filter_map(|v| match v {
                                    CtVal::Record { fields, .. } => fields.get("name").cloned(),
                                    _ => None,
                                })
                                .collect();
                            return Ok(CtVal::List(names));
                        }
                        "has_field" if args.len() == 1 => {
                            let target = expect_text(&self.eval_expr_env(&args[0].value, env)?)?;
                            let found = self.ctx.fields_of_type(ty).into_iter().any(|v| {
                                matches!(v, CtVal::Record { fields, .. } if matches!(fields.get("name"), Some(CtVal::Text(n)) if n == &target))
                            });
                            return Ok(CtVal::Bool(found));
                        }
                        "implements" if type_args.len() == 2 => {
                            let iface = type_args[1].display_name();
                            return Ok(CtVal::Bool(self.ctx.type_implements(ty, &iface)));
                        }
                        "is_subtype" if type_args.len() == 2 => {
                            let sub = self.ctx.is_refinement_subtype(ty, &type_args[1]);
                            return Ok(CtVal::Bool(sub));
                        }
                        _ => {}
                    }
                }
            }
        }

        // 2. Method calls on `Text` / `List` (e.g. `s.len()`, `s.contains(...)`, `xs.len()`)
        if let ExprKind::Field { target, field } = &callee.kind {
            if let ExprKind::Ident(ename) = &target.kind {
                if let Some(ed) = self.ctx.enums.get(ename) {
                    if ed.variants.iter().any(|v| v.name == *field) {
                        let mut vals = Vec::new();
                        for a in args {
                            vals.push(self.eval_expr_env(&a.value, env)?);
                        }
                        return Ok(CtVal::Enum {
                            name: ename.clone(),
                            variant: field.clone(),
                            payload: vals,
                        });
                    }
                }
            }
            let recv = self.eval_expr_env(target, env)?;
            let mut vals = Vec::new();
            for a in args {
                vals.push(self.eval_expr_env(&a.value, env)?);
            }
            match (&recv, field.as_str()) {
                (CtVal::Text(s), "len" | "byte_len") => return Ok(CtVal::Int(s.len() as i64)),
                (CtVal::List(xs), "len") => return Ok(CtVal::Int(xs.len() as i64)),
                (CtVal::Text(s), "contains") if vals.len() == 1 => {
                    return Ok(CtVal::Bool(s.contains(&expect_text(&vals[0])?)));
                }
                (CtVal::Text(s), "starts_with") if vals.len() == 1 => {
                    return Ok(CtVal::Bool(s.starts_with(&expect_text(&vals[0])?)));
                }
                (CtVal::Text(s), "ends_with") if vals.len() == 1 => {
                    return Ok(CtVal::Bool(s.ends_with(&expect_text(&vals[0])?)));
                }
                (CtVal::Text(s), "replace") if vals.len() == 2 => {
                    return Ok(CtVal::Text(
                        s.replace(&expect_text(&vals[0])?, &expect_text(&vals[1])?),
                    ));
                }
                (CtVal::Text(s), "trim") => return Ok(CtVal::Text(s.trim().to_string())),
                _ => {}
            }
        }

        // 3. Direct function & built-in calls
        let ExprKind::Ident(fname) = &callee.kind else {
            return Err(Flow::Err("unsupported callee in comptime call".into()));
        };
        let mut vals = Vec::with_capacity(args.len());
        for a in args {
            vals.push(self.eval_expr_env(&a.value, env)?);
        }

        match fname.as_str() {
            "routes_of" => return Ok(CtVal::List(self.ctx.routes_of_program())),
            "annotated_fns" if vals.len() == 1 => {
                let aname = expect_text(&vals[0])?;
                return Ok(CtVal::List(self.ctx.annotated_fns_of_program(&aname)));
            }
            "config" if vals.len() == 1 => {
                let k = expect_text(&vals[0])?;
                return Ok(CtVal::Text(
                    self.ctx.build_config.get(&k).cloned().unwrap_or_default(),
                ));
            }
            "target_os" => return Ok(CtVal::Text(std::env::consts::OS.to_string())),
            "target_arch" => return Ok(CtVal::Text(std::env::consts::ARCH.to_string())),
            "target_pointer_width" => {
                return Ok(CtVal::Int((std::mem::size_of::<usize>() * 8) as i64));
            }
            "len" if vals.len() == 1 => match &vals[0] {
                CtVal::Text(s) => return Ok(CtVal::Int(s.len() as i64)),
                CtVal::List(xs) => return Ok(CtVal::Int(xs.len() as i64)),
                _ => return Err(Flow::Err("len() expects Text or List in comptime".into())),
            },
            "push" if vals.len() == 2 => {
                let mut xs = match vals.remove(0) {
                    CtVal::List(l) => l,
                    _ => return Err(Flow::Err("push() expects List in comptime".into())),
                };
                xs.push(vals.remove(0));
                return Ok(CtVal::List(xs));
            }
            "int_to_text" if vals.len() == 1 => {
                return Ok(CtVal::Text(expect_int(&vals[0])?.to_string()));
            }
            "float_to_text" if vals.len() == 1 => {
                return Ok(CtVal::Text(expect_float(&vals[0])?.to_string()));
            }
            "bool_to_text" if vals.len() == 1 => {
                return Ok(CtVal::Text(if matches!(vals[0], CtVal::Bool(true)) {
                    "true".into()
                } else {
                    "false".into()
                }));
            }
            "str_slice" if vals.len() == 3 => {
                let s = expect_text(&vals[0])?;
                let a = (expect_int(&vals[1])?.max(0) as usize).min(s.len());
                let b = (expect_int(&vals[2])?.max(0) as usize).min(s.len()).max(a);
                return Ok(CtVal::Text(s[a..b].to_string()));
            }
            "byte_at" if vals.len() == 2 => {
                let s = expect_text(&vals[0])?;
                let idx = expect_int(&vals[1])?;
                if idx < 0 || (idx as usize) >= s.len() {
                    return Ok(CtVal::Int(-1));
                }
                return Ok(CtVal::Int(s.as_bytes()[idx as usize] as i64));
            }
            "contains" if vals.len() == 2 => {
                let a = expect_text(&vals[0])?;
                let b = expect_text(&vals[1])?;
                return Ok(CtVal::Bool(a.contains(&b)));
            }
            "replace" if vals.len() == 3 => {
                let s = expect_text(&vals[0])?;
                let from = expect_text(&vals[1])?;
                let to = expect_text(&vals[2])?;
                return Ok(CtVal::Text(s.replace(&from, &to)));
            }
            "Some" if vals.len() == 1 => {
                return Ok(CtVal::Enum {
                    name: "Option".into(),
                    variant: "Some".into(),
                    payload: vals,
                });
            }
            "Ok" if vals.len() == 1 => {
                return Ok(CtVal::Enum {
                    name: "Result".into(),
                    variant: "Ok".into(),
                    payload: vals,
                });
            }
            "Err" if vals.len() == 1 => {
                return Ok(CtVal::Enum {
                    name: "Result".into(),
                    variant: "Err".into(),
                    payload: vals,
                });
            }
            _ => {}
        }

        if let Some(fdecl) = self.ctx.fns.get(fname) {
            return self.call_fn(fdecl, vals).map_err(Flow::Err);
        }
        // Check enum variant constructor
        for ed in self.ctx.enums.values() {
            if ed.variants.iter().any(|v| v.name == *fname) {
                return Ok(CtVal::Enum {
                    name: ed.name.clone(),
                    variant: fname.clone(),
                    payload: vals,
                });
            }
        }
        Err(Flow::Err(format!(
            "unknown function `{}` in comptime evaluation",
            fname
        )))
    }
}

fn expect_int(v: &CtVal) -> Result<i64, Flow> {
    match v {
        CtVal::Int(n) => Ok(*n),
        other => Err(Flow::Err(format!("expected Int in comptime, got {:?}", other))),
    }
}

fn expect_float(v: &CtVal) -> Result<f64, Flow> {
    match v {
        CtVal::Float(x) => Ok(*x),
        CtVal::Int(n) => Ok(*n as f64),
        other => Err(Flow::Err(format!("expected Float in comptime, got {:?}", other))),
    }
}

fn expect_text(v: &CtVal) -> Result<String, Flow> {
    match v {
        CtVal::Text(s) => Ok(s.clone()),
        other => Err(Flow::Err(format!("expected Text in comptime, got {:?}", other))),
    }
}

fn apply_assign_op(cur: CtVal, op: AssignOp, rhs: CtVal) -> Result<CtVal, Flow> {
    match op {
        AssignOp::Set => Ok(rhs),
        AssignOp::AddSet => eval_binop(BinOp::Add, cur, rhs),
        AssignOp::SubSet => eval_binop(BinOp::Sub, cur, rhs),
        AssignOp::MulSet => eval_binop(BinOp::Mul, cur, rhs),
        AssignOp::DivSet => eval_binop(BinOp::Div, cur, rhs),
        AssignOp::BitAndSet => eval_binop(BinOp::BitAnd, cur, rhs),
        AssignOp::BitOrSet => eval_binop(BinOp::BitOr, cur, rhs),
        AssignOp::BitXorSet => eval_binop(BinOp::BitXor, cur, rhs),
        AssignOp::ShlSet => eval_binop(BinOp::Shl, cur, rhs),
        AssignOp::ShrSet => eval_binop(BinOp::Shr, cur, rhs),
    }
}

fn eval_binop(op: BinOp, l: CtVal, r: CtVal) -> Result<CtVal, Flow> {
    match (op, l, r) {
        (BinOp::Concat, CtVal::Text(a), CtVal::Text(b)) => Ok(CtVal::Text(format!("{}{}", a, b))),
        (BinOp::Add, CtVal::Int(a), CtVal::Int(b)) => Ok(CtVal::Int(a.wrapping_add(b))),
        (BinOp::Sub, CtVal::Int(a), CtVal::Int(b)) => Ok(CtVal::Int(a.wrapping_sub(b))),
        (BinOp::Mul, CtVal::Int(a), CtVal::Int(b)) => Ok(CtVal::Int(a.wrapping_mul(b))),
        (BinOp::Div, CtVal::Int(a), CtVal::Int(b)) if b != 0 => Ok(CtVal::Int(a / b)),
        (BinOp::Mod, CtVal::Int(a), CtVal::Int(b)) if b != 0 => Ok(CtVal::Int(a % b)),
        (BinOp::BitAnd, CtVal::Int(a), CtVal::Int(b)) => Ok(CtVal::Int(a & b)),
        (BinOp::BitOr, CtVal::Int(a), CtVal::Int(b)) => Ok(CtVal::Int(a | b)),
        (BinOp::BitXor, CtVal::Int(a), CtVal::Int(b)) => Ok(CtVal::Int(a ^ b)),
        (BinOp::Shl, CtVal::Int(a), CtVal::Int(b)) => Ok(CtVal::Int(a.wrapping_shl(b as u32))),
        (BinOp::Shr, CtVal::Int(a), CtVal::Int(b)) => Ok(CtVal::Int(a.wrapping_shr(b as u32))),
        (BinOp::Add, CtVal::Float(a), CtVal::Float(b)) => Ok(CtVal::Float(a + b)),
        (BinOp::Sub, CtVal::Float(a), CtVal::Float(b)) => Ok(CtVal::Float(a - b)),
        (BinOp::Mul, CtVal::Float(a), CtVal::Float(b)) => Ok(CtVal::Float(a * b)),
        (BinOp::Div, CtVal::Float(a), CtVal::Float(b)) => Ok(CtVal::Float(a / b)),
        (BinOp::Lt, CtVal::Int(a), CtVal::Int(b)) => Ok(CtVal::Bool(a < b)),
        (BinOp::Le, CtVal::Int(a), CtVal::Int(b)) => Ok(CtVal::Bool(a <= b)),
        (BinOp::Gt, CtVal::Int(a), CtVal::Int(b)) => Ok(CtVal::Bool(a > b)),
        (BinOp::Ge, CtVal::Int(a), CtVal::Int(b)) => Ok(CtVal::Bool(a >= b)),
        (BinOp::Lt, CtVal::Float(a), CtVal::Float(b)) => Ok(CtVal::Bool(a < b)),
        (BinOp::Le, CtVal::Float(a), CtVal::Float(b)) => Ok(CtVal::Bool(a <= b)),
        (BinOp::Gt, CtVal::Float(a), CtVal::Float(b)) => Ok(CtVal::Bool(a > b)),
        (BinOp::Ge, CtVal::Float(a), CtVal::Float(b)) => Ok(CtVal::Bool(a >= b)),
        (BinOp::Eq, a, b) => Ok(CtVal::Bool(ctval_eq(&a, &b))),
        (BinOp::Ne, a, b) => Ok(CtVal::Bool(!ctval_eq(&a, &b))),
        (op, a, b) => Err(Flow::Err(format!(
            "unsupported comptime binary op {:?} on ({:?}, {:?})",
            op, a, b
        ))),
    }
}

fn ctval_eq(a: &CtVal, b: &CtVal) -> bool {
    match (a, b) {
        (CtVal::Int(x), CtVal::Int(y)) => x == y,
        (CtVal::Float(x), CtVal::Float(y)) => x == y,
        (CtVal::Bool(x), CtVal::Bool(y)) => x == y,
        (CtVal::Text(x), CtVal::Text(y)) => x == y,
        (CtVal::Unit, CtVal::Unit) => true,
        (CtVal::Type(x), CtVal::Type(y)) => x.display_name() == y.display_name(),
        (CtVal::List(x), CtVal::List(y)) | (CtVal::Tuple(x), CtVal::Tuple(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(u, v)| ctval_eq(u, v))
        }
        (
            CtVal::Enum {
                variant: v1,
                payload: p1,
                ..
            },
            CtVal::Enum {
                variant: v2,
                payload: p2,
                ..
            },
        ) => v1 == v2 && p1.len() == p2.len() && p1.iter().zip(p2.iter()).all(|(u, v)| ctval_eq(u, v)),
        _ => false,
    }
}

fn match_ct_pattern(pat: &Pattern, val: &CtVal, bindings: &mut BTreeMap<String, CtVal>) -> bool {
    match (pat, val) {
        (Pattern::Wildcard(_), _) => true,
        (Pattern::Int(n, _), CtVal::Int(v)) => n == v,
        (Pattern::Bool(b, _), CtVal::Bool(v)) => b == v,
        (Pattern::Str(s, _), CtVal::Text(v)) => s == v,
        (Pattern::Ident(name, _), v) => {
            bindings.insert(name.clone(), v.clone());
            true
        }
        (Pattern::Tuple(ps, _), CtVal::Tuple(vs)) if ps.len() == vs.len() => ps
            .iter()
            .zip(vs.iter())
            .all(|(p, v)| match_ct_pattern(p, v, bindings)),
        (
            Pattern::Variant { name, binding, .. },
            CtVal::Enum {
                variant, payload, ..
            },
        ) => {
            let leaf = name.rsplit('.').next().unwrap_or(name);
            if leaf != variant {
                return false;
            }
            if let Some(b) = binding {
                if b != "_" {
                    if let Some(v0) = payload.first() {
                        bindings.insert(b.clone(), v0.clone());
                    }
                }
            }
            true
        }
        (
            Pattern::VariantMulti { name, patterns, .. },
            CtVal::Enum {
                variant, payload, ..
            },
        ) => {
            let leaf = name.rsplit('.').next().unwrap_or(name);
            leaf == variant
                && patterns.len() == payload.len()
                && patterns
                    .iter()
                    .zip(payload.iter())
                    .all(|(p, v)| match_ct_pattern(p, v, bindings))
        }
        _ => false,
    }
}

fn ctval_to_ast_expr(val: &CtVal, span: Span) -> Result<Expr, String> {
    let kind = match val {
        CtVal::Int(n) => ExprKind::Int(*n),
        CtVal::Float(x) => ExprKind::Float(*x),
        CtVal::Bool(b) => ExprKind::Bool(*b),
        CtVal::Text(s) => ExprKind::Str(s.clone()),
        CtVal::Unit => ExprKind::Int(0),
        CtVal::List(xs) => {
            let mut es = Vec::with_capacity(xs.len());
            for x in xs {
                es.push(ctval_to_ast_expr(x, span)?);
            }
            ExprKind::List(es)
        }
        CtVal::Tuple(xs) => {
            let mut es = Vec::with_capacity(xs.len());
            for x in xs {
                es.push(ctval_to_ast_expr(x, span)?);
            }
            ExprKind::Tuple(es)
        }
        CtVal::Record { name, fields } => {
            let mut fs = Vec::with_capacity(fields.len());
            for (k, v) in fields {
                fs.push((k.clone(), ctval_to_ast_expr(v, span)?));
            }
            ExprKind::RecordLit {
                name: name.clone(),
                version: None,
                fields: fs,
            }
        }
        CtVal::Enum {
            variant, payload, ..
        } => {
            if payload.is_empty() {
                ExprKind::Ident(variant.clone())
            } else {
                let mut args = Vec::with_capacity(payload.len());
                for p in payload {
                    args.push(CallArg {
                        name: None,
                        value: ctval_to_ast_expr(p, span)?,
                    });
                }
                ExprKind::Call {
                    callee: Box::new(Expr {
                        kind: ExprKind::Ident(variant.clone()),
                        span,
                    }),
                    args,
                }
            }
        }
        CtVal::Type(t) => ExprKind::Str(t.display_name()),
    };
    Ok(Expr { kind, span })
}

// =========================================================================
// Substitution & Type-Param Helpers
// =========================================================================

fn type_mentions_type_params(ty: &TypeExpr, tps: &BTreeSet<String>) -> bool {
    if tps.is_empty() {
        return false;
    }
    match ty {
        TypeExpr::Named { name, args, .. } => {
            tps.contains(name) || args.iter().any(|a| type_mentions_type_params(a, tps))
        }
        TypeExpr::Dyn { args, .. } => args.iter().any(|a| type_mentions_type_params(a, tps)),
        TypeExpr::Tuple(es, _) => es.iter().any(|a| type_mentions_type_params(a, tps)),
        TypeExpr::Fn { params, ret, .. } => {
            params.iter().any(|a| type_mentions_type_params(a, tps))
                || ret
                    .as_ref()
                    .is_some_and(|r| type_mentions_type_params(r, tps))
        }
        TypeExpr::Array { elem, .. } => type_mentions_type_params(elem, tps),
    }
}

fn block_mentions_type_params(blk: &Block, tps: &BTreeSet<String>) -> bool {
    if tps.is_empty() {
        return false;
    }
    blk.stmts.iter().any(|s| match s {
        Stmt::Let { ty, value, .. } | Stmt::LetTuple { ty, value, .. } => {
            ty.as_ref().is_some_and(|t| type_mentions_type_params(t, tps))
                || expr_mentions_type_params(value, tps)
        }
        Stmt::Assign { target, value, .. } => {
            expr_mentions_type_params(target, tps) || expr_mentions_type_params(value, tps)
        }
        Stmt::Return { value: Some(e), .. }
        | Stmt::Assert { cond: e, .. }
        | Stmt::Expr { expr: e, .. } => expr_mentions_type_params(e, tps),
        Stmt::While { cond, body, .. } => {
            expr_mentions_type_params(cond, tps) || block_mentions_type_params(body, tps)
        }
        Stmt::For { iter, body, .. } => {
            expr_mentions_type_params(iter, tps) || block_mentions_type_params(body, tps)
        }
        Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } | Stmt::Ghost { body, .. } => {
            block_mentions_type_params(body, tps)
        }
        _ => false,
    }) || blk
        .tail
        .as_ref()
        .is_some_and(|t| expr_mentions_type_params(t, tps))
}

fn expr_mentions_type_params(expr: &Expr, tps: &BTreeSet<String>) -> bool {
    if tps.is_empty() {
        return false;
    }
    match &expr.kind {
        ExprKind::Ident(s) => tps.contains(s),
        ExprKind::TypeImplements { ty, .. } => type_mentions_type_params(ty, tps),
        ExprKind::GenericInst { expr: inner, type_args } => {
            expr_mentions_type_params(inner, tps)
                || type_args.iter().any(|t| type_mentions_type_params(t, tps))
        }
        ExprKind::Call { callee, args } => {
            expr_mentions_type_params(callee, tps)
                || args.iter().any(|a| expr_mentions_type_params(&a.value, tps))
        }
        ExprKind::MacroCall { args, .. } => args.iter().any(|a| expr_mentions_type_params(a, tps)),
        ExprKind::Binary { left, right, .. } => {
            expr_mentions_type_params(left, tps) || expr_mentions_type_params(right, tps)
        }
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Field { target: inner, .. }
        | ExprKind::TupleIndex { target: inner, .. }
        | ExprKind::Await(inner)
        | ExprKind::Try(inner)
        | ExprKind::Old(inner) => expr_mentions_type_params(inner, tps),
        ExprKind::Cast { expr: inner, ty } | ExprKind::TryCast { expr: inner, ty } => {
            expr_mentions_type_params(inner, tps) || type_mentions_type_params(ty, tps)
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) | ExprKind::Comptime(b) => {
            block_mentions_type_params(b, tps)
        }
        ExprKind::If { cond, then_block, else_branch }
        | ExprKind::ComptimeIf { cond, then_block, else_branch } => {
            expr_mentions_type_params(cond, tps)
                || block_mentions_type_params(then_block, tps)
                || else_branch
                    .as_ref()
                    .is_some_and(|e| expr_mentions_type_params(e, tps))
        }
        ExprKind::ComptimeFor { iter, body, .. } => {
            expr_mentions_type_params(iter, tps) || block_mentions_type_params(body, tps)
        }
        _ => false,
    }
}

fn subst_expr_params(expr: &mut Expr, subst: &BTreeMap<String, Expr>) {
    if let ExprKind::Ident(s) = &expr.kind {
        if let Some(rep) = subst.get(s) {
            *expr = rep.clone();
            return;
        }
    }
    match &mut expr.kind {
        ExprKind::Binary { left, right, .. } => {
            subst_expr_params(left, subst);
            subst_expr_params(right, subst);
        }
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Await(inner)
        | ExprKind::Old(inner)
        | ExprKind::Try(inner)
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::TryCast { expr: inner, .. }
        | ExprKind::ArrayRepeat { value: inner, .. }
        | ExprKind::Field { target: inner, .. }
        | ExprKind::TupleIndex { target: inner, .. }
        | ExprKind::GenericInst { expr: inner, .. } => subst_expr_params(inner, subst),
        ExprKind::Call { callee, args } => {
            subst_expr_params(callee, subst);
            for a in args {
                subst_expr_params(&mut a.value, subst);
            }
        }
        ExprKind::MacroCall { args, .. } => {
            for a in args {
                subst_expr_params(a, subst);
            }
        }
        ExprKind::Range { start, end } => {
            subst_expr_params(start, subst);
            subst_expr_params(end, subst);
        }
        ExprKind::Index { target, index } => {
            subst_expr_params(target, subst);
            subst_expr_params(index, subst);
        }
        ExprKind::Tuple(es) | ExprKind::List(es) => {
            for e in es {
                subst_expr_params(e, subst);
            }
        }
        ExprKind::RecordLit { fields, .. } => {
            for (_, v) in fields {
                subst_expr_params(v, subst);
            }
        }
        ExprKind::Lambda { body, .. } => subst_expr_params(body, subst),
        ExprKind::If { cond, then_block, else_branch }
        | ExprKind::ComptimeIf { cond, then_block, else_branch } => {
            subst_expr_params(cond, subst);
            subst_block_params(then_block, subst);
            if let Some(eb) = else_branch {
                subst_expr_params(eb, subst);
            }
        }
        ExprKind::ComptimeFor { iter, body, .. } => {
            subst_expr_params(iter, subst);
            subst_block_params(body, subst);
        }
        ExprKind::Match { scrutinee, arms } => {
            subst_expr_params(scrutinee, subst);
            for a in arms {
                if let Some(g) = &mut a.guard {
                    subst_expr_params(g, subst);
                }
                subst_expr_params(&mut a.body, subst);
            }
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) | ExprKind::Comptime(b) => {
            subst_block_params(b, subst);
        }
        _ => {}
    }
}

fn subst_block_params(blk: &mut Block, subst: &BTreeMap<String, Expr>) {
    for s in &mut blk.stmts {
        match s {
            Stmt::Let { value, .. } | Stmt::LetTuple { value, .. } => subst_expr_params(value, subst),
            Stmt::Assign { target, value, .. } => {
                subst_expr_params(target, subst);
                subst_expr_params(value, subst);
            }
            Stmt::Return { value: Some(e), .. }
            | Stmt::Assert { cond: e, .. }
            | Stmt::Expr { expr: e, .. } => subst_expr_params(e, subst),
            Stmt::While { cond, body, .. } => {
                subst_expr_params(cond, subst);
                subst_block_params(body, subst);
            }
            Stmt::For { iter, body, .. } => {
                subst_expr_params(iter, subst);
                subst_block_params(body, subst);
            }
            Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } | Stmt::Ghost { body, .. } => {
                subst_block_params(body, subst)
            }
            _ => {}
        }
    }
    if let Some(t) = &mut blk.tail {
        subst_expr_params(t, subst);
    }
}

fn subst_ctval_in_block(blk: &mut Block, binding: &str, val: &CtVal) {
    for s in &mut blk.stmts {
        match s {
            Stmt::Let { value, .. } | Stmt::LetTuple { value, .. } => {
                subst_ctval_in_expr(value, binding, val)
            }
            Stmt::Assign { target, value, .. } => {
                subst_ctval_in_expr(target, binding, val);
                subst_ctval_in_expr(value, binding, val);
            }
            Stmt::Return { value: Some(e), .. }
            | Stmt::Assert { cond: e, .. }
            | Stmt::Expr { expr: e, .. } => subst_ctval_in_expr(e, binding, val),
            Stmt::While { cond, body, .. } => {
                subst_ctval_in_expr(cond, binding, val);
                subst_ctval_in_block(body, binding, val);
            }
            Stmt::For { iter, body, .. } => {
                subst_ctval_in_expr(iter, binding, val);
                subst_ctval_in_block(body, binding, val);
            }
            Stmt::Mutate { body, .. } | Stmt::Defer { body, .. } | Stmt::Ghost { body, .. } => {
                subst_ctval_in_block(body, binding, val)
            }
            _ => {}
        }
    }
    if let Some(t) = &mut blk.tail {
        subst_ctval_in_expr(t, binding, val);
    }
}

fn subst_ctval_in_expr(expr: &mut Expr, binding: &str, val: &CtVal) {
    // Support `get_field(obj, f)` -> `obj.<f.name>` inside `comptime for f in fields_of[T]()`!
    if let ExprKind::Call { callee, args } = &expr.kind {
        if let ExprKind::Ident(fname) = &callee.kind {
            if fname == "get_field" && args.len() == 2 {
                if let ExprKind::Ident(b) = &args[1].value.kind {
                    if b == binding {
                        if let CtVal::Record { fields, .. } = val {
                            if let Some(CtVal::Text(field_name)) = fields.get("name") {
                                let mut obj = args[0].value.clone();
                                subst_ctval_in_expr(&mut obj, binding, val);
                                expr.kind = ExprKind::Field {
                                    target: Box::new(obj),
                                    field: field_name.clone(),
                                };
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
    // Support `f.name`, `f.ty`, `f.index`, `f.has_refinement` -> direct constant literal!
    if let ExprKind::Field { target, field } = &expr.kind {
        if let ExprKind::Ident(b) = &target.kind {
            if b == binding {
                if let CtVal::Record { fields, .. } = val {
                    if let Some(fv) = fields.get(field) {
                        if let Ok(lit) = ctval_to_ast_expr(fv, expr.span) {
                            *expr = lit;
                            return;
                        }
                    }
                }
            }
        }
    }
    if let ExprKind::Ident(b) = &expr.kind {
        if b == binding {
            if let Ok(lit) = ctval_to_ast_expr(val, expr.span) {
                *expr = lit;
                return;
            }
        }
    }
    match &mut expr.kind {
        ExprKind::Binary { left, right, .. } => {
            subst_ctval_in_expr(left, binding, val);
            subst_ctval_in_expr(right, binding, val);
        }
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Await(inner)
        | ExprKind::Old(inner)
        | ExprKind::Try(inner)
        | ExprKind::Cast { expr: inner, .. }
        | ExprKind::TryCast { expr: inner, .. }
        | ExprKind::ArrayRepeat { value: inner, .. }
        | ExprKind::Field { target: inner, .. }
        | ExprKind::TupleIndex { target: inner, .. }
        | ExprKind::GenericInst { expr: inner, .. } => subst_ctval_in_expr(inner, binding, val),
        ExprKind::Call { callee, args } => {
            subst_ctval_in_expr(callee, binding, val);
            for a in args {
                subst_ctval_in_expr(&mut a.value, binding, val);
            }
        }
        ExprKind::MacroCall { args, .. } => {
            for a in args {
                subst_ctval_in_expr(a, binding, val);
            }
        }
        ExprKind::Index { target, index } => {
            subst_ctval_in_expr(target, binding, val);
            subst_ctval_in_expr(index, binding, val);
        }
        ExprKind::Tuple(es) | ExprKind::List(es) => {
            for e in es {
                subst_ctval_in_expr(e, binding, val);
            }
        }
        ExprKind::RecordLit { fields, .. } => {
            for (_, v) in fields {
                subst_ctval_in_expr(v, binding, val);
            }
        }
        ExprKind::If { cond, then_block, else_branch }
        | ExprKind::ComptimeIf { cond, then_block, else_branch } => {
            subst_ctval_in_expr(cond, binding, val);
            subst_ctval_in_block(then_block, binding, val);
            if let Some(eb) = else_branch {
                subst_ctval_in_expr(eb, binding, val);
            }
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) | ExprKind::Comptime(b) => {
            subst_ctval_in_block(b, binding, val)
        }
        ExprKind::Match { scrutinee, arms } => {
            subst_ctval_in_expr(scrutinee, binding, val);
            for a in arms {
                if let Some(g) = &mut a.guard {
                    subst_ctval_in_expr(g, binding, val);
                }
                subst_ctval_in_expr(&mut a.body, binding, val);
            }
        }
        _ => {}
    }
}

fn quick_infer_expr_ty(
    ctx: &ComptimeContext,
    expr: &Expr,
    var_types: &BTreeMap<String, TypeExpr>,
) -> TypeExpr {
    let mk = |n: &str| TypeExpr::Named {
        name: n.into(),
        version: None,
        args: Vec::new(),
        span: expr.span,
    };
    match &expr.kind {
        ExprKind::Int(_) => mk("Int"),
        ExprKind::Float(_) => mk("Float"),
        ExprKind::Bool(_) => mk("Bool"),
        ExprKind::Str(_) => mk("Text"),
        ExprKind::Ident(s) => {
            if let Some(t) = var_types.get(s) {
                return t.clone();
            }
            if ctx.records.contains_key(s)
                || ctx.enums.contains_key(s)
                || ctx.aliases.contains_key(s)
            {
                return mk(s);
            }
            mk("Int")
        }
        ExprKind::RecordLit { name, .. } => mk(name),
        ExprKind::Cast { ty, .. } => ty.clone(),
        ExprKind::TryCast { ty, .. } => TypeExpr::Named {
            name: "Result".into(),
            version: None,
            args: vec![ty.clone(), mk("Text")],
            span: expr.span,
        },
        ExprKind::List(es) => {
            let et = es
                .first()
                .map(|e| quick_infer_expr_ty(ctx, e, var_types))
                .unwrap_or_else(|| mk("Any"));
            TypeExpr::Named {
                name: "List".into(),
                version: None,
                args: vec![et],
                span: expr.span,
            }
        }
        ExprKind::Tuple(es) => TypeExpr::Tuple(
            es.iter()
                .map(|e| quick_infer_expr_ty(ctx, e, var_types))
                .collect(),
            expr.span,
        ),
        ExprKind::Call { callee, .. } => {
            if let ExprKind::Ident(fname) = &callee.kind {
                if let Some(f) = ctx.fns.get(fname) {
                    if let Some(rt) = &f.ret_ty {
                        return rt.clone();
                    }
                }
            }
            mk("Any")
        }
        _ => mk("Any"),
    }
}

fn extract_simple_int_bounds(pred: &Expr) -> Option<(i128, i128)> {
    let mut lo = i128::MIN / 2;
    let mut hi = i128::MAX / 2;
    fn visit(e: &Expr, lo: &mut i128, hi: &mut i128) -> bool {
        match &e.kind {
            ExprKind::Binary { op: BinOp::And, left, right } => {
                visit(left, lo, hi) && visit(right, lo, hi)
            }
            ExprKind::Binary { op, left, right } => {
                let const_val = |x: &Expr| -> Option<i128> {
                    match &x.kind {
                        ExprKind::Int(n) => Some(*n as i128),
                        ExprKind::Unary { op: UnOp::Neg, expr } => match &expr.kind {
                            ExprKind::Int(n) => Some(-(*n as i128)),
                            _ => None,
                        },
                        _ => None,
                    }
                };
                let is_self = |x: &Expr| matches!(&x.kind, ExprKind::Ident(s) if s == "self");
                if is_self(left) {
                    if let Some(c) = const_val(right) {
                        match op {
                            BinOp::Ge => *lo = (*lo).max(c),
                            BinOp::Gt => *lo = (*lo).max(c + 1),
                            BinOp::Le => *hi = (*hi).min(c),
                            BinOp::Lt => *hi = (*hi).min(c - 1),
                            BinOp::Eq => {
                                *lo = (*lo).max(c);
                                *hi = (*hi).min(c);
                            }
                            _ => return false,
                        }
                        return true;
                    }
                } else if is_self(right) {
                    if let Some(c) = const_val(left) {
                        match op {
                            BinOp::Le => *lo = (*lo).max(c),
                            BinOp::Lt => *lo = (*lo).max(c + 1),
                            BinOp::Ge => *hi = (*hi).min(c),
                            BinOp::Gt => *hi = (*hi).min(c - 1),
                            BinOp::Eq => {
                                *lo = (*lo).max(c);
                                *hi = (*hi).min(c);
                            }
                            _ => return false,
                        }
                        return true;
                    }
                }
                false
            }
            _ => false,
        }
    }
    if visit(pred, &mut lo, &mut hi) && lo <= hi {
        Some((lo, hi))
    } else {
        None
    }
}

fn format_expr_src(expr: &Expr) -> String {
    match &expr.kind {
        ExprKind::Int(n) => n.to_string(),
        ExprKind::Float(x) => {
            let s = format!("{}", x);
            if s.contains('.') { s } else { format!("{}.0", s) }
        }
        ExprKind::Bool(b) => if *b { "true".into() } else { "false".into() },
        ExprKind::Str(s) => format!("{:?}", s),
        ExprKind::Ident(s) => s.clone(),
        ExprKind::Field { target, field } => format!("{}.{}", format_expr_src(target), field),
        ExprKind::TupleIndex { target, index } => format!("{}.{}", format_expr_src(target), index),
        ExprKind::Index { target, index } => {
            format!("{}[{}]", format_expr_src(target), format_expr_src(index))
        }
        ExprKind::Unary { op, expr: inner } => {
            let sym = match op {
                UnOp::Neg => "-",
                UnOp::Not => "!",
                UnOp::BitNot => "~",
            };
            format!("({}{})", sym, format_expr_src(inner))
        }
        ExprKind::Binary { op, left, right } => {
            format!(
                "({} {} {})",
                format_expr_src(left),
                op.symbol(),
                format_expr_src(right)
            )
        }
        ExprKind::Call { callee, args } => {
            let a: Vec<String> = args
                .iter()
                .map(|ca| match &ca.name {
                    Some(k) => format!("{} = {}", k, format_expr_src(&ca.value)),
                    None => format_expr_src(&ca.value),
                })
                .collect();
            format!("{}({})", format_expr_src(callee), a.join(", "))
        }
        ExprKind::List(es) => {
            let parts: Vec<String> = es.iter().map(format_expr_src).collect();
            format!("[{}]", parts.join(", "))
        }
        ExprKind::Tuple(es) => {
            let parts: Vec<String> = es.iter().map(format_expr_src).collect();
            format!("({})", parts.join(", "))
        }
        ExprKind::Cast { expr: inner, ty } => {
            format!("({} as {})", format_expr_src(inner), ty.display_name())
        }
        ExprKind::TryCast { expr: inner, ty } => {
            format!("({} as? {})", format_expr_src(inner), ty.display_name())
        }
        ExprKind::RecordLit { name, fields, .. } => {
            let fs: Vec<String> = fields
                .iter()
                .map(|(k, v)| format!("{} = {}", k, format_expr_src(v)))
                .collect();
            format!("{} {{ {} }}", name, fs.join(", "))
        }
        _ => "0".into(),
    }
}
