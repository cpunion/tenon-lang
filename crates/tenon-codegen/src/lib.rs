pub mod perceus;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    AbiParam, InstBuilder, MemFlags, StackSlotData, StackSlotKind, Type, Value, types,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use perceus::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicI64, Ordering};
use tenon_syntax::*;

static LAST_ASSERT_FAILED: AtomicI64 = AtomicI64::new(0);
static LAST_PRINTED: AtomicI64 = AtomicI64::new(0);

extern "C" fn tenon_rt_assert_fail(line: i64) {
    LAST_ASSERT_FAILED.store(line, Ordering::SeqCst);
}

extern "C" fn tenon_rt_print_int(val: i64) -> i64 {
    LAST_PRINTED.store(val, Ordering::SeqCst);
    println!("{}", val);
    val
}

#[derive(Debug, Clone)]
struct RecordLayout {
    tag: i64,
    fields: Vec<(String, Option<Expr>)>,
    where_clauses: Vec<Expr>,
}

#[derive(Debug, Clone)]
struct FnMeta {
    params_inout: Vec<bool>,
    returns_rc: bool,
}

#[derive(Clone, Copy)]
struct RtFuncs {
    assert_fail: FuncId,
    rc_alloc: FuncId,
    rc_inc: FuncId,
    rc_dec: FuncId,
    rc_ensure_unique: FuncId,
    rc_get_field: FuncId,
    rc_set_field: FuncId,
    rc_get_tag: FuncId,
    rc_get_len: FuncId,
    text_new: FuncId,
    text_eq: FuncId,
    list_push: FuncId,
}

pub struct JitProgram {
    module: JITModule,
    func_ids: BTreeMap<String, FuncId>,
    pub compiled_fns: Vec<String>,
    pub derived_parsers: Vec<String>,
    _interned_strings: Vec<Box<[u8]>>,
}

impl JitProgram {
    pub fn compile(program: &Program) -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|e| e.to_string())?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| e.to_string())?;
        let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| e.to_string())?;

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        jit_builder.symbol("tenon_rt_assert_fail", tenon_rt_assert_fail as *const u8);
        jit_builder.symbol("tenon_rt_print_int", tenon_rt_print_int as *const u8);
        jit_builder.symbol("tenon_rc_alloc", tenon_rc_alloc as *const u8);
        jit_builder.symbol("tenon_rc_inc", tenon_rc_inc as *const u8);
        jit_builder.symbol("tenon_rc_dec", tenon_rc_dec as *const u8);
        jit_builder.symbol("tenon_rc_ensure_unique", tenon_rc_ensure_unique as *const u8);
        jit_builder.symbol("tenon_rc_get_field", tenon_rc_get_field as *const u8);
        jit_builder.symbol("tenon_rc_set_field", tenon_rc_set_field as *const u8);
        jit_builder.symbol("tenon_rc_get_tag", tenon_rc_get_tag as *const u8);
        jit_builder.symbol("tenon_rc_get_len", tenon_rc_get_len as *const u8);
        jit_builder.symbol("tenon_text_new", tenon_text_new as *const u8);
        jit_builder.symbol("tenon_text_eq", tenon_text_eq as *const u8);
        jit_builder.symbol("tenon_list_push", tenon_list_push as *const u8);

        let mut module = JITModule::new(jit_builder);

        let rt = declare_runtime_funcs(&mut module)?;
        let print_int_id = declare_sig(&mut module, "tenon_rt_print_int", 1, true)?;

        // Collect Records & Enums
        let mut records: BTreeMap<String, RecordLayout> = BTreeMap::new();
        let mut field_to_idx: BTreeMap<String, usize> = BTreeMap::new();
        let mut enum_tags: BTreeMap<String, i64> = BTreeMap::new();
        enum_tags.insert("Err".into(), TAG_ERR);
        enum_tags.insert("Ok".into(), TAG_OK);
        enum_tags.insert("None".into(), TAG_NONE);
        enum_tags.insert("Some".into(), TAG_SOME);

        let mut next_rec_tag = TAG_RECORD_BASE;
        let mut next_enum_tag = 10i64;
        for item in &program.items {
            match item {
                Item::Record(r) if r.type_params.is_empty() => {
                    let mut fields = Vec::new();
                    for (idx, f) in r.fields.iter().enumerate() {
                        fields.push((f.name.clone(), f.predicate.clone()));
                        field_to_idx.entry(f.name.clone()).or_insert(idx);
                    }
                    records.insert(
                        r.name.clone(),
                        RecordLayout {
                            tag: next_rec_tag,
                            fields,
                            where_clauses: r.where_clauses.clone(),
                        },
                    );
                    next_rec_tag += 1;
                }
                Item::Enum(e) => {
                    for v in &e.variants {
                        enum_tags.insert(v.name.clone(), next_enum_tag);
                        enum_tags.insert(format!("{}.{}", e.name, v.name), next_enum_tag);
                        next_enum_tag += 1;
                    }
                }
                _ => {}
            }
        }

        // Collect compilable functions
        let fns: Vec<&FnDecl> = program
            .items
            .iter()
            .filter_map(|i| match i {
                Item::Fn(f) if !f.is_async && f.type_params.is_empty() => Some(f),
                _ => None,
            })
            .collect();

        let mut func_ids = BTreeMap::new();
        let mut fn_metas: BTreeMap<String, FnMeta> = BTreeMap::new();
        func_ids.insert("print_int".into(), print_int_id);

        // Declare all user functions
        for f in &fns {
            let mut sig = module.make_signature();
            let mut params_inout = Vec::new();
            for p in &f.params {
                sig.params.push(AbiParam::new(types::I64));
                params_inout.push(p.is_inout);
            }
            sig.returns.push(AbiParam::new(types::I64));
            let id = module
                .declare_function(&f.name, Linkage::Export, &sig)
                .map_err(|e| e.to_string())?;
            func_ids.insert(f.name.clone(), id);
            fn_metas.insert(
                f.name.clone(),
                FnMeta {
                    params_inout,
                    returns_rc: f
                        .ret_ty
                        .as_ref()
                        .is_some_and(|t| is_rc_type(t, &records)),
                },
            );
        }

        // Declare auto-derived `<Record>__parse` boundary parsers for every record!
        let mut derived_parsers = Vec::new();
        for (rname, layout) in &records {
            if layout.fields.is_empty() {
                continue;
            }
            let pname = format!("{}__parse", rname);
            let mut sig = module.make_signature();
            for _ in &layout.fields {
                sig.params.push(AbiParam::new(types::I64));
            }
            sig.returns.push(AbiParam::new(types::I64));
            let id = module
                .declare_function(&pname, Linkage::Export, &sig)
                .map_err(|e| e.to_string())?;
            func_ids.insert(pname.clone(), id);
            fn_metas.insert(
                pname.clone(),
                FnMeta {
                    params_inout: vec![false; layout.fields.len()],
                    returns_rc: true,
                },
            );
            derived_parsers.push(pname);
        }

        let mut interned_strings: Vec<Box<[u8]>> = Vec::new();
        let mut ctx = module.make_context();
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut compiled_fns = Vec::new();

        // Compile all user functions
        for f in &fns {
            let id = func_ids[&f.name];
            ctx.func.signature = module
                .declarations()
                .get_function_decl(id)
                .signature
                .clone();

            {
                let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
                let entry_block = builder.create_block();
                builder.append_block_params_for_function_params(entry_block);
                builder.switch_to_block(entry_block);
                builder.seal_block(entry_block);

                let mut vars = BTreeMap::new();
                let mut var_records = BTreeMap::new();
                let mut inout_vars = BTreeSet::new();
                let mut text_vars = BTreeSet::new();

                for (idx, p) in f.params.iter().enumerate() {
                    let var = builder.declare_var(types::I64);
                    let val = builder.block_params(entry_block)[idx];
                    builder.def_var(var, val);
                    vars.insert(p.name.clone(), var);
                    let tname = p.ty.display_name();
                    if records.contains_key(&tname) {
                        var_records.insert(p.name.clone(), tname.clone());
                    }
                    if tname == "Text" {
                        text_vars.insert(p.name.clone());
                    }
                    if p.is_inout {
                        inout_vars.insert(p.name.clone());
                    }
                }

                let mut lower = FnLower {
                    module: &mut module,
                    builder: &mut builder,
                    rt,
                    func_ids: &func_ids,
                    fn_metas: &fn_metas,
                    records: &records,
                    field_to_idx: &field_to_idx,
                    enum_tags: &enum_tags,
                    interned_strings: &mut interned_strings,
                    vars,
                    var_records,
                    inout_vars,
                    owned_rc_vars: BTreeSet::new(),
                    text_vars,
                    self_var: None,
                    terminated: false,
                };
                let ret_expr_ident = f.body.tail.as_deref().and_then(|t| match &t.kind {
                    ExprKind::Ident(s) => Some(s.clone()),
                    _ => None,
                });
                let ret_val = lower.lower_block(&f.body)?;
                if !lower.terminated {
                    lower.emit_cleanup(ret_expr_ident.as_deref());
                    lower.builder.ins().return_(&[ret_val]);
                }
                builder.finalize();
            }

            module
                .define_function(id, &mut ctx)
                .map_err(|e| format!("Cranelift codegen failed for `{}`: {}", f.name, e))?;
            module.clear_context(&mut ctx);
            compiled_fns.push(f.name.clone());
        }

        // Compile all auto-derived `<Record>__parse` functions!
        for (rname, layout) in &records {
            if layout.fields.is_empty() {
                continue;
            }
            let pname = format!("{}__parse", rname);
            let id = func_ids[&pname];
            ctx.func.signature = module
                .declarations()
                .get_function_decl(id)
                .signature
                .clone();

            {
                let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
                let entry_block = builder.create_block();
                builder.append_block_params_for_function_params(entry_block);
                builder.switch_to_block(entry_block);
                builder.seal_block(entry_block);

                let mut vars = BTreeMap::new();
                for (idx, (fname, _)) in layout.fields.iter().enumerate() {
                    let var = builder.declare_var(types::I64);
                    let val = builder.block_params(entry_block)[idx];
                    builder.def_var(var, val);
                    vars.insert(fname.clone(), var);
                }

                let mut lower = FnLower {
                    module: &mut module,
                    builder: &mut builder,
                    rt,
                    func_ids: &func_ids,
                    fn_metas: &fn_metas,
                    records: &records,
                    field_to_idx: &field_to_idx,
                    enum_tags: &enum_tags,
                    interned_strings: &mut interned_strings,
                    vars,
                    var_records: BTreeMap::new(),
                    inout_vars: BTreeSet::new(),
                    owned_rc_vars: BTreeSet::new(),
                    text_vars: BTreeSet::new(),
                    self_var: None,
                    terminated: false,
                };

                // Check each field's `where` predicate in declaration order!
                for (idx, (fname, pred_opt)) in layout.fields.iter().enumerate() {
                    if let Some(pred) = pred_opt {
                        lower.self_var = Some(fname.clone());
                        let cond_val = lower.lower_expr(pred)?;
                        lower.self_var = None;
                        let zero = lower.builder.ins().iconst(types::I64, 0);
                        let ok = lower.builder.ins().icmp(IntCC::NotEqual, cond_val, zero);
                        let fail_b = lower.builder.create_block();
                        let next_b = lower.builder.create_block();
                        lower.builder.ins().brif(ok, next_b, &[], fail_b, &[]);

                        lower.builder.switch_to_block(fail_b);
                        lower.builder.seal_block(fail_b);
                        let err_obj = lower.call_rt_2(rt.rc_alloc, TAG_ERR, 1);
                        let code_val = lower.builder.ins().iconst(types::I64, (idx + 1) as i64);
                        lower.call_rt_3_void(rt.rc_set_field, err_obj, 0, code_val);
                        lower.builder.ins().return_(&[err_obj]);

                        lower.builder.switch_to_block(next_b);
                        lower.builder.seal_block(next_b);
                    }
                }

                // Check record-level `where` clauses!
                for wc in &layout.where_clauses {
                    let cond_val = lower.lower_expr(wc)?;
                    let zero = lower.builder.ins().iconst(types::I64, 0);
                    let ok = lower.builder.ins().icmp(IntCC::NotEqual, cond_val, zero);
                    let fail_b = lower.builder.create_block();
                    let next_b = lower.builder.create_block();
                    lower.builder.ins().brif(ok, next_b, &[], fail_b, &[]);

                    lower.builder.switch_to_block(fail_b);
                    lower.builder.seal_block(fail_b);
                    let err_obj = lower.call_rt_2(rt.rc_alloc, TAG_ERR, 1);
                    let code_val = lower.builder.ins().iconst(types::I64, 99);
                    lower.call_rt_3_void(rt.rc_set_field, err_obj, 0, code_val);
                    lower.builder.ins().return_(&[err_obj]);

                    lower.builder.switch_to_block(next_b);
                    lower.builder.seal_block(next_b);
                }

                // Construct Record and wrap in Ok(rec)!
                let rec_obj = lower.call_rt_2(rt.rc_alloc, layout.tag, layout.fields.len() as i64);
                for (idx, (fname, _)) in layout.fields.iter().enumerate() {
                    let v = lower.builder.use_var(lower.vars[fname]);
                    lower.call_rt_3_void(rt.rc_set_field, rec_obj, idx as i64, v);
                }
                let ok_obj = lower.call_rt_2(rt.rc_alloc, TAG_OK, 1);
                lower.call_rt_3_void(rt.rc_set_field, ok_obj, 0, rec_obj);
                lower.builder.ins().return_(&[ok_obj]);
                builder.finalize();
            }

            module
                .define_function(id, &mut ctx)
                .map_err(|e| format!("Cranelift codegen failed for derived `{}`: {}", pname, e))?;
            module.clear_context(&mut ctx);
            compiled_fns.push(pname);
        }

        module.finalize_definitions().map_err(|e| e.to_string())?;
        Ok(Self {
            module,
            func_ids,
            compiled_fns,
            derived_parsers,
            _interned_strings: interned_strings,
        })
    }

    pub fn call_0(&self, name: &str) -> Result<i64, String> {
        LAST_ASSERT_FAILED.store(0, Ordering::SeqCst);
        let id = *self
            .func_ids
            .get(name)
            .ok_or_else(|| format!("function `{}` not compiled", name))?;
        let ptr = self.module.get_finalized_function(id);
        let f: extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
        let res = f();
        let failed = LAST_ASSERT_FAILED.load(Ordering::SeqCst);
        if failed != 0 {
            return Err(format!("assertion failed at line {}", failed));
        }
        Ok(res)
    }

    pub fn call_2(&self, name: &str, a: i64, b: i64) -> Result<i64, String> {
        LAST_ASSERT_FAILED.store(0, Ordering::SeqCst);
        let id = *self
            .func_ids
            .get(name)
            .ok_or_else(|| format!("function `{}` not compiled", name))?;
        let ptr = self.module.get_finalized_function(id);
        let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        let res = f(a, b);
        let failed = LAST_ASSERT_FAILED.load(Ordering::SeqCst);
        if failed != 0 {
            return Err(format!("assertion failed at line {}", failed));
        }
        Ok(res)
    }

    pub fn last_printed() -> i64 {
        LAST_PRINTED.load(Ordering::SeqCst)
    }
}

fn declare_sig(
    module: &mut JITModule,
    name: &str,
    n_params: usize,
    has_ret: bool,
) -> Result<FuncId, String> {
    let mut sig = module.make_signature();
    for _ in 0..n_params {
        sig.params.push(AbiParam::new(types::I64));
    }
    if has_ret {
        sig.returns.push(AbiParam::new(types::I64));
    }
    module
        .declare_function(name, Linkage::Import, &sig)
        .map_err(|e| e.to_string())
}

fn declare_runtime_funcs(module: &mut JITModule) -> Result<RtFuncs, String> {
    Ok(RtFuncs {
        assert_fail: declare_sig(module, "tenon_rt_assert_fail", 1, false)?,
        rc_alloc: declare_sig(module, "tenon_rc_alloc", 2, true)?,
        rc_inc: declare_sig(module, "tenon_rc_inc", 1, true)?,
        rc_dec: declare_sig(module, "tenon_rc_dec", 1, false)?,
        rc_ensure_unique: declare_sig(module, "tenon_rc_ensure_unique", 1, true)?,
        rc_get_field: declare_sig(module, "tenon_rc_get_field", 2, true)?,
        rc_set_field: declare_sig(module, "tenon_rc_set_field", 3, false)?,
        rc_get_tag: declare_sig(module, "tenon_rc_get_tag", 1, true)?,
        rc_get_len: declare_sig(module, "tenon_rc_get_len", 1, true)?,
        text_new: declare_sig(module, "tenon_text_new", 2, true)?,
        text_eq: declare_sig(module, "tenon_text_eq", 2, true)?,
        list_push: declare_sig(module, "tenon_list_push", 2, true)?,
    })
}

fn is_rc_type(ty: &TypeExpr, records: &BTreeMap<String, RecordLayout>) -> bool {
    let name = ty.display_name();
    records.contains_key(&name)
        || name == "Text"
        || name.starts_with("List[")
        || name.starts_with("Option[")
        || name.starts_with("Result[")
}

struct FnLower<'a, 'b> {
    module: &'a mut JITModule,
    builder: &'a mut FunctionBuilder<'b>,
    rt: RtFuncs,
    func_ids: &'a BTreeMap<String, FuncId>,
    fn_metas: &'a BTreeMap<String, FnMeta>,
    records: &'a BTreeMap<String, RecordLayout>,
    field_to_idx: &'a BTreeMap<String, usize>,
    enum_tags: &'a BTreeMap<String, i64>,
    interned_strings: &'a mut Vec<Box<[u8]>>,
    vars: BTreeMap<String, Variable>,
    var_records: BTreeMap<String, String>,
    inout_vars: BTreeSet<String>,
    owned_rc_vars: BTreeSet<String>,
    text_vars: BTreeSet<String>,
    self_var: Option<String>,
    terminated: bool,
}

impl<'a, 'b> FnLower<'a, 'b> {
    fn declare_var(&mut self, name: &str, ty: Type) -> Variable {
        if let Some(&v) = self.vars.get(name) {
            return v;
        }
        let v = self.builder.declare_var(ty);
        self.vars.insert(name.to_string(), v);
        v
    }

    fn call_rt_1(&mut self, id: FuncId, a: Value) -> Value {
        let callee = self.module.declare_func_in_func(id, self.builder.func);
        let call = self.builder.ins().call(callee, &[a]);
        self.builder.inst_results(call)[0]
    }

    fn call_rt_1_void(&mut self, id: FuncId, a: Value) {
        let callee = self.module.declare_func_in_func(id, self.builder.func);
        self.builder.ins().call(callee, &[a]);
    }

    fn call_rt_2(&mut self, id: FuncId, a: i64, b: i64) -> Value {
        let va = self.builder.ins().iconst(types::I64, a);
        let vb = self.builder.ins().iconst(types::I64, b);
        self.call_rt_2_vals(id, va, vb)
    }

    fn call_rt_2_vals(&mut self, id: FuncId, va: Value, vb: Value) -> Value {
        let callee = self.module.declare_func_in_func(id, self.builder.func);
        let call = self.builder.ins().call(callee, &[va, vb]);
        self.builder.inst_results(call)[0]
    }

    fn call_rt_3_void(&mut self, id: FuncId, obj: Value, idx: i64, val: Value) {
        let vidx = self.builder.ins().iconst(types::I64, idx);
        let callee = self.module.declare_func_in_func(id, self.builder.func);
        self.builder.ins().call(callee, &[obj, vidx, val]);
    }

    fn emit_cleanup(&mut self, keep_var: Option<&str>) {
        let to_dec: Vec<String> = self
            .owned_rc_vars
            .iter()
            .filter(|v| keep_var != Some(v.as_str()) && !self.inout_vars.contains(*v))
            .cloned()
            .collect();
        for vname in to_dec {
            if let Some(&var) = self.vars.get(&vname) {
                let handle = self.builder.use_var(var);
                self.call_rt_1_void(self.rt.rc_dec, handle);
            }
        }
    }

    fn resolve_field_idx(&self, obj_expr: &Expr, field: &str) -> Result<usize, String> {
        if let ExprKind::Ident(vname) = &obj_expr.kind {
            if let Some(rname) = self.var_records.get(vname) {
                if let Some(layout) = self.records.get(rname) {
                    if let Some(pos) = layout.fields.iter().position(|(f, _)| f == field) {
                        return Ok(pos);
                    }
                }
            }
        }
        self.field_to_idx
            .get(field)
            .copied()
            .ok_or_else(|| format!("unknown record field `{}` in codegen", field))
    }

    fn expr_produces_rc(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::RecordLit { .. } | ExprKind::List(_) | ExprKind::Str(_) => true,
            ExprKind::Ident(s) => {
                s == "None" || self.owned_rc_vars.contains(s) || self.var_records.contains_key(s)
            }
            ExprKind::Call { callee, .. } => match &callee.kind {
                ExprKind::Ident(f) => {
                    matches!(f.as_str(), "Ok" | "Err" | "Some" | "push")
                        || self.enum_tags.contains_key(f)
                        || self.fn_metas.get(f).is_some_and(|m| m.returns_rc)
                }
                ExprKind::Field { field, .. } => field == "parse",
                _ => false,
            },
            _ => false,
        }
    }

    fn is_text_expr(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Str(_) => true,
            ExprKind::Ident(s) => self.text_vars.contains(s),
            _ => false,
        }
    }

    fn lower_block(&mut self, blk: &Block) -> Result<Value, String> {
        let mut last = self.builder.ins().iconst(types::I64, 0);
        for stmt in &blk.stmts {
            if self.terminated {
                break;
            }
            match stmt {
                Stmt::Let {
                    name, ty, value, ..
                } => {
                    let is_copy_of_rc = match &value.kind {
                        ExprKind::Ident(src) => {
                            self.owned_rc_vars.contains(src) || self.var_records.contains_key(src)
                        }
                        _ => false,
                    };
                    let mut v = self.lower_expr(value)?;
                    if is_copy_of_rc {
                        // MVS copy increments Perceus refcount; subsequent mutate triggers COW!
                        v = self.call_rt_1(self.rt.rc_inc, v);
                    }
                    let var = self.declare_var(name, types::I64);
                    self.builder.def_var(var, v);

                    if let Some(t) = ty {
                        let tname = t.display_name();
                        if self.records.contains_key(&tname) {
                            self.var_records.insert(name.clone(), tname);
                            self.owned_rc_vars.insert(name.clone());
                        } else if tname == "Text" {
                            self.text_vars.insert(name.clone());
                            self.owned_rc_vars.insert(name.clone());
                        } else if is_rc_type(t, self.records) {
                            self.owned_rc_vars.insert(name.clone());
                        }
                    } else {
                        if let ExprKind::RecordLit { name: rname, .. } = &value.kind {
                            self.var_records.insert(name.clone(), rname.clone());
                        } else if let ExprKind::Ident(src) = &value.kind {
                            if let Some(rname) = self.var_records.get(src).cloned() {
                                self.var_records.insert(name.clone(), rname);
                            }
                            if self.text_vars.contains(src) {
                                self.text_vars.insert(name.clone());
                            }
                        } else if matches!(&value.kind, ExprKind::Str(_)) {
                            self.text_vars.insert(name.clone());
                        }
                        if self.expr_produces_rc(value) {
                            self.owned_rc_vars.insert(name.clone());
                        }
                    }
                }
                Stmt::Assign {
                    target, op, value, ..
                } => match &target.kind {
                    ExprKind::Ident(name) => {
                        let consuming_push = matches!(
                            &value.kind,
                            ExprKind::Call { callee, args }
                            if matches!(&callee.kind, ExprKind::Ident(f) if f == "push")
                                && args.first().is_some_and(|a| matches!(&a.value.kind, ExprKind::Ident(s) if s == name))
                        );
                        let rhs = self.lower_expr(value)?;
                        let var = self.declare_var(name, types::I64);
                        if self.owned_rc_vars.contains(name) && !consuming_push {
                            let old = self.builder.use_var(var);
                            self.call_rt_1_void(self.rt.rc_dec, old);
                        }
                        let new_v = match op {
                            AssignOp::Set => rhs,
                            AssignOp::AddSet => {
                                let cur = self.builder.use_var(var);
                                self.builder.ins().iadd(cur, rhs)
                            }
                            AssignOp::SubSet => {
                                let cur = self.builder.use_var(var);
                                self.builder.ins().isub(cur, rhs)
                            }
                        };
                        self.builder.def_var(var, new_v);
                    }
                    ExprKind::Field { target: obj, field } => {
                        if let ExprKind::Ident(vname) = &obj.kind {
                            let fidx = self.resolve_field_idx(obj, field)?;
                            let rhs = self.lower_expr(value)?;
                            let is_inout = self.inout_vars.contains(vname);
                            let var = *self
                                .vars
                                .get(vname)
                                .ok_or_else(|| format!("undefined variable `{}`", vname))?;
                            let cur_handle = if is_inout {
                                let slot_ptr = self.builder.use_var(var);
                                self.builder
                                    .ins()
                                    .load(types::I64, MemFlags::new(), slot_ptr, 0)
                            } else {
                                self.builder.use_var(var)
                            };
                            // Perceus unique in-place reuse (rc == 1) OR MVS COW clone (rc > 1)!
                            let uniq_handle = self.call_rt_1(self.rt.rc_ensure_unique, cur_handle);
                            if is_inout {
                                let slot_ptr = self.builder.use_var(var);
                                self.builder
                                    .ins()
                                    .store(MemFlags::new(), uniq_handle, slot_ptr, 0);
                            } else {
                                self.builder.def_var(var, uniq_handle);
                            }
                            let new_field_val = match op {
                                AssignOp::Set => rhs,
                                AssignOp::AddSet => {
                                    let idx_v = self.builder.ins().iconst(types::I64, fidx as i64);
                                    let cur_f =
                                        self.call_rt_2_vals(self.rt.rc_get_field, uniq_handle, idx_v);
                                    self.builder.ins().iadd(cur_f, rhs)
                                }
                                AssignOp::SubSet => {
                                    let idx_v = self.builder.ins().iconst(types::I64, fidx as i64);
                                    let cur_f =
                                        self.call_rt_2_vals(self.rt.rc_get_field, uniq_handle, idx_v);
                                    self.builder.ins().isub(cur_f, rhs)
                                }
                            };
                            self.call_rt_3_void(
                                self.rt.rc_set_field,
                                uniq_handle,
                                fidx as i64,
                                new_field_val,
                            );
                        }
                    }
                    _ => {}
                },
                Stmt::Mutate { body, .. } => {
                    let _ = self.lower_block(body)?;
                }
                Stmt::Return { value, .. } => {
                    let keep = value.as_ref().and_then(|e| match &e.kind {
                        ExprKind::Ident(s) => Some(s.clone()),
                        _ => None,
                    });
                    let v = match value {
                        Some(e) => self.lower_expr(e)?,
                        None => self.builder.ins().iconst(types::I64, 0),
                    };
                    self.emit_cleanup(keep.as_deref());
                    self.builder.ins().return_(&[v]);
                    self.terminated = true;
                    return Ok(v);
                }
                Stmt::Assert { cond, span } => {
                    let c = self.lower_expr(cond)?;
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    let is_ok = self.builder.ins().icmp(IntCC::NotEqual, c, zero);
                    let fail_blk = self.builder.create_block();
                    let cont_blk = self.builder.create_block();
                    self.builder
                        .ins()
                        .brif(is_ok, cont_blk, &[], fail_blk, &[]);
                    self.builder.switch_to_block(fail_blk);
                    self.builder.seal_block(fail_blk);
                    let line_val = self.builder.ins().iconst(types::I64, span.line as i64);
                    self.call_rt_1_void(self.rt.assert_fail, line_val);
                    self.builder.ins().jump(cont_blk, &[]);
                    self.builder.switch_to_block(cont_blk);
                    self.builder.seal_block(cont_blk);
                }
                Stmt::While { cond, body, .. } => {
                    let hdr = self.builder.create_block();
                    let loop_blk = self.builder.create_block();
                    let exit_blk = self.builder.create_block();
                    self.builder.ins().jump(hdr, &[]);
                    self.builder.switch_to_block(hdr);
                    let c = self.lower_expr(cond)?;
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    let cmp = self.builder.ins().icmp(IntCC::NotEqual, c, zero);
                    self.builder
                        .ins()
                        .brif(cmp, loop_blk, &[], exit_blk, &[]);
                    self.builder.switch_to_block(loop_blk);
                    self.builder.seal_block(loop_blk);
                    let _ = self.lower_block(body)?;
                    if !self.terminated {
                        self.builder.ins().jump(hdr, &[]);
                    }
                    self.terminated = false;
                    self.builder.seal_block(hdr);
                    self.builder.switch_to_block(exit_blk);
                    self.builder.seal_block(exit_blk);
                }
                Stmt::Expr { expr, .. } => {
                    last = self.lower_expr(expr)?;
                }
            }
        }
        if !self.terminated {
            if let Some(tail) = &blk.tail {
                last = self.lower_expr(tail)?;
            }
        }
        Ok(last)
    }

    fn lower_expr(&mut self, expr: &Expr) -> Result<Value, String> {
        match &expr.kind {
            ExprKind::Int(n) => Ok(self.builder.ins().iconst(types::I64, *n)),
            ExprKind::Bool(b) => Ok(self
                .builder
                .ins()
                .iconst(types::I64, if *b { 1 } else { 0 })),
            ExprKind::Str(s) => {
                let boxed: Box<[u8]> = s.as_bytes().to_vec().into_boxed_slice();
                let ptr = boxed.as_ptr() as i64;
                let len = boxed.len() as i64;
                self.interned_strings.push(boxed);
                Ok(self.call_rt_2(self.rt.text_new, ptr, len))
            }
            ExprKind::Ident(s) => {
                if s == "self" {
                    if let Some(sv) = self.self_var.clone() {
                        let var = self.vars[&sv];
                        return Ok(self.builder.use_var(var));
                    }
                }
                if s == "None" {
                    return Ok(self.call_rt_2(self.rt.rc_alloc, TAG_NONE, 0));
                }
                if let Some(&etag) = self.enum_tags.get(s) {
                    if !self.vars.contains_key(s) {
                        return Ok(self.call_rt_2(self.rt.rc_alloc, etag, 0));
                    }
                }
                let var = *self
                    .vars
                    .get(s)
                    .ok_or_else(|| format!("undefined variable `{}` in codegen", s))?;
                if self.inout_vars.contains(s) {
                    let slot_ptr = self.builder.use_var(var);
                    Ok(self
                        .builder
                        .ins()
                        .load(types::I64, MemFlags::new(), slot_ptr, 0))
                } else {
                    Ok(self.builder.use_var(var))
                }
            }
            ExprKind::RecordLit { name, fields, .. } => {
                let layout = self
                    .records
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown record `{}` in codegen", name))?;
                let obj = self.call_rt_2(self.rt.rc_alloc, layout.tag, layout.fields.len() as i64);
                for (fname, fexpr) in fields {
                    let val = self.lower_expr(fexpr)?;
                    if let Some(idx) = layout.fields.iter().position(|(f, _)| f == fname) {
                        self.call_rt_3_void(self.rt.rc_set_field, obj, idx as i64, val);
                    }
                }
                Ok(obj)
            }
            ExprKind::List(elems) => {
                let obj = self.call_rt_2(self.rt.rc_alloc, TAG_LIST, elems.len() as i64);
                for (idx, e) in elems.iter().enumerate() {
                    let val = self.lower_expr(e)?;
                    self.call_rt_3_void(self.rt.rc_set_field, obj, idx as i64, val);
                }
                Ok(obj)
            }
            ExprKind::Index { target, index } => {
                let t = self.lower_expr(target)?;
                let i = self.lower_expr(index)?;
                Ok(self.call_rt_2_vals(self.rt.rc_get_field, t, i))
            }
            ExprKind::Field { target, field } => {
                // Handle Enum.Variant zero-arg expression
                if let ExprKind::Ident(ename) = &target.kind {
                    let key = format!("{}.{}", ename, field);
                    if let Some(&tag) = self.enum_tags.get(&key) {
                        return Ok(self.call_rt_2(self.rt.rc_alloc, tag, 0));
                    }
                }
                let idx = self.resolve_field_idx(target, field)?;
                let obj = self.lower_expr(target)?;
                let idx_v = self.builder.ins().iconst(types::I64, idx as i64);
                Ok(self.call_rt_2_vals(self.rt.rc_get_field, obj, idx_v))
            }
            ExprKind::Unary { op, expr } => {
                let v = self.lower_expr(expr)?;
                match op {
                    UnOp::Neg => Ok(self.builder.ins().ineg(v)),
                    UnOp::Not => {
                        let zero = self.builder.ins().iconst(types::I64, 0);
                        let eq = self.builder.ins().icmp(IntCC::Equal, v, zero);
                        Ok(self.builder.ins().uextend(types::I64, eq))
                    }
                }
            }
            ExprKind::Binary { op, left, right } => {
                let text_cmp = self.is_text_expr(left) || self.is_text_expr(right);
                let l = self.lower_expr(left)?;
                let r = self.lower_expr(right)?;
                if text_cmp && matches!(op, BinOp::Eq | BinOp::Ne) {
                    let eq = self.call_rt_2_vals(self.rt.text_eq, l, r);
                    if *op == BinOp::Eq {
                        return Ok(eq);
                    }
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    let ne = self.builder.ins().icmp(IntCC::Equal, eq, zero);
                    return Ok(self.builder.ins().uextend(types::I64, ne));
                }
                let v = match op {
                    BinOp::Add => self.builder.ins().iadd(l, r),
                    BinOp::Sub => self.builder.ins().isub(l, r),
                    BinOp::Mul => self.builder.ins().imul(l, r),
                    BinOp::Div => self.builder.ins().sdiv(l, r),
                    BinOp::Mod => self.builder.ins().srem(l, r),
                    BinOp::BitAnd => self.builder.ins().band(l, r),
                    BinOp::BitOr => self.builder.ins().bor(l, r),
                    BinOp::Eq => {
                        let c = self.builder.ins().icmp(IntCC::Equal, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::Ne => {
                        let c = self.builder.ins().icmp(IntCC::NotEqual, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::Lt => {
                        let c = self.builder.ins().icmp(IntCC::SignedLessThan, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::Le => {
                        let c = self.builder.ins().icmp(IntCC::SignedLessThanOrEqual, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::Gt => {
                        let c = self.builder.ins().icmp(IntCC::SignedGreaterThan, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::Ge => {
                        let c = self
                            .builder
                            .ins()
                            .icmp(IntCC::SignedGreaterThanOrEqual, l, r);
                        self.builder.ins().uextend(types::I64, c)
                    }
                    BinOp::And => {
                        let z = self.builder.ins().iconst(types::I64, 0);
                        let lb = self.builder.ins().icmp(IntCC::NotEqual, l, z);
                        let rb = self.builder.ins().icmp(IntCC::NotEqual, r, z);
                        let and = self.builder.ins().band(lb, rb);
                        self.builder.ins().uextend(types::I64, and)
                    }
                    BinOp::Or => {
                        let z = self.builder.ins().iconst(types::I64, 0);
                        let lb = self.builder.ins().icmp(IntCC::NotEqual, l, z);
                        let rb = self.builder.ins().icmp(IntCC::NotEqual, r, z);
                        let or = self.builder.ins().bor(lb, rb);
                        self.builder.ins().uextend(types::I64, or)
                    }
                    other => return Err(format!("unsupported binary op {:?} in codegen", other)),
                };
                Ok(v)
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                let c = self.lower_expr(cond)?;
                let z = self.builder.ins().iconst(types::I64, 0);
                let cmp = self.builder.ins().icmp(IntCC::NotEqual, c, z);
                let then_b = self.builder.create_block();
                let else_b = self.builder.create_block();
                let merge_b = self.builder.create_block();
                self.builder.append_block_param(merge_b, types::I64);
                self.builder
                    .ins()
                    .brif(cmp, then_b, &[], else_b, &[]);

                self.builder.switch_to_block(then_b);
                self.builder.seal_block(then_b);
                let tv = self.lower_block(then_block)?;
                if !self.terminated {
                    self.builder.ins().jump(merge_b, &[tv.into()]);
                }
                self.terminated = false;

                self.builder.switch_to_block(else_b);
                self.builder.seal_block(else_b);
                let ev = match else_branch {
                    Some(eb) => self.lower_expr(eb)?,
                    None => self.builder.ins().iconst(types::I64, 0),
                };
                if !self.terminated {
                    self.builder.ins().jump(merge_b, &[ev.into()]);
                }
                self.terminated = false;

                self.builder.switch_to_block(merge_b);
                self.builder.seal_block(merge_b);
                Ok(self.builder.block_params(merge_b)[0])
            }
            ExprKind::Match { scrutinee, arms } => {
                let is_temp_parse = matches!(
                    &scrutinee.kind,
                    ExprKind::Call { callee, .. }
                    if matches!(&callee.kind, ExprKind::Field { field, .. } if field == "parse")
                );
                let parse_record_name = match &scrutinee.kind {
                    ExprKind::Call { callee, .. } => match &callee.kind {
                        ExprKind::Field { target, field } if field == "parse" => {
                            match &target.kind {
                                ExprKind::Ident(rname) => Some(rname.clone()),
                                _ => None,
                            }
                        }
                        _ => None,
                    },
                    _ => None,
                };
                let is_variant_match = arms
                    .iter()
                    .any(|a| a.patterns.iter().any(|p| matches!(p, Pattern::Variant { .. })));

                let sv = self.lower_expr(scrutinee)?;
                let cmp_target = if is_variant_match {
                    self.call_rt_1(self.rt.rc_get_tag, sv)
                } else {
                    sv
                };
                let merge_b = self.builder.create_block();
                self.builder.append_block_param(merge_b, types::I64);

                for arm in arms {
                    let is_wild = arm
                        .patterns
                        .iter()
                        .any(|p| matches!(p, Pattern::Wildcard(_)));
                    if is_wild {
                        let v = self.lower_expr(&arm.body)?;
                        if is_temp_parse {
                            self.call_rt_1_void(self.rt.rc_dec, sv);
                        }
                        self.builder.ins().jump(merge_b, &[v.into()]);
                        self.builder.switch_to_block(merge_b);
                        self.builder.seal_block(merge_b);
                        return Ok(self.builder.block_params(merge_b)[0]);
                    }
                    let mut cond: Option<Value> = None;
                    let mut variant_binding: Option<(String, bool)> = None;
                    for p in &arm.patterns {
                        let pv = match p {
                            Pattern::Int(n, _) => self.builder.ins().iconst(types::I64, *n),
                            Pattern::Bool(b, _) => {
                                self.builder
                                    .ins()
                                    .iconst(types::I64, if *b { 1 } else { 0 })
                            }
                            Pattern::Variant { name, binding, .. } => {
                                let tag = *self.enum_tags.get(name).ok_or_else(|| {
                                    format!("unknown enum variant `{}` in match", name)
                                })?;
                                if let Some(bname) = binding {
                                    variant_binding = Some((bname.clone(), name == "Ok"));
                                }
                                self.builder.ins().iconst(types::I64, tag)
                            }
                            _ => continue,
                        };
                        let eq = self.builder.ins().icmp(IntCC::Equal, cmp_target, pv);
                        cond = Some(match cond {
                            Some(prev) => self.builder.ins().bor(prev, eq),
                            None => eq,
                        });
                    }
                    let matched_b = self.builder.create_block();
                    let next_b = self.builder.create_block();
                    let c = cond.unwrap_or_else(|| self.builder.ins().iconst(types::I8, 0));
                    self.builder
                        .ins()
                        .brif(c, matched_b, &[], next_b, &[]);

                    self.builder.switch_to_block(matched_b);
                    self.builder.seal_block(matched_b);
                    let mut bound_payload: Option<(Value, bool)> = None;
                    if let Some((bname, is_ok)) = variant_binding {
                        let zero = self.builder.ins().iconst(types::I64, 0);
                        let payload = self.call_rt_2_vals(self.rt.rc_get_field, sv, zero);
                        let bvar = self.declare_var(&bname, types::I64);
                        self.builder.def_var(bvar, payload);
                        if is_ok {
                            if let Some(rname) = &parse_record_name {
                                self.var_records.insert(bname, rname.clone());
                                bound_payload = Some((payload, true));
                            }
                        }
                    }
                    let v = self.lower_expr(&arm.body)?;
                    if is_temp_parse {
                        if let Some((inner_rec, true)) = bound_payload {
                            // Free inner parsed record if body evaluated to a scalar (e.g. `dt.day`)
                            if !self.expr_produces_rc(&arm.body) {
                                self.call_rt_1_void(self.rt.rc_dec, inner_rec);
                            }
                        }
                        self.call_rt_1_void(self.rt.rc_dec, sv);
                    }
                    self.builder.ins().jump(merge_b, &[v.into()]);

                    self.builder.switch_to_block(next_b);
                    self.builder.seal_block(next_b);
                }

                let zero = self.builder.ins().iconst(types::I64, 0);
                if is_temp_parse {
                    self.call_rt_1_void(self.rt.rc_dec, sv);
                }
                self.builder.ins().jump(merge_b, &[zero.into()]);
                self.builder.switch_to_block(merge_b);
                self.builder.seal_block(merge_b);
                Ok(self.builder.block_params(merge_b)[0])
            }
            ExprKind::Block(blk) => self.lower_block(blk),
            ExprKind::Call { callee, args } => match &callee.kind {
                ExprKind::Field { target, field } if field == "parse" => {
                    if let ExprKind::Ident(rname) = &target.kind {
                        let pname = format!("{}__parse", rname);
                        let fid = *self.func_ids.get(&pname).ok_or_else(|| {
                            format!("no derived parser `{}.parse` found", rname)
                        })?;
                        let mut arg_vals = Vec::new();
                        for a in args {
                            arg_vals.push(self.lower_expr(&a.value)?);
                        }
                        let local_callee = self.module.declare_func_in_func(fid, self.builder.func);
                        let call = self.builder.ins().call(local_callee, &arg_vals);
                        return Ok(self.builder.inst_results(call)[0]);
                    }
                    Err("unsupported `.parse()` target".into())
                }
                ExprKind::Field { target, field } => {
                    if let ExprKind::Ident(ename) = &target.kind {
                        let key = format!("{}.{}", ename, field);
                        if let Some(&tag) = self.enum_tags.get(&key) {
                            let obj = self.call_rt_2(self.rt.rc_alloc, tag, args.len() as i64);
                            for (idx, a) in args.iter().enumerate() {
                                let av = self.lower_expr(&a.value)?;
                                self.call_rt_3_void(self.rt.rc_set_field, obj, idx as i64, av);
                            }
                            return Ok(obj);
                        }
                    }
                    Err("unsupported method call in M1 codegen".into())
                }
                ExprKind::Ident(fname) => {
                    if fname == "len" && args.len() == 1 {
                        let v = self.lower_expr(&args[0].value)?;
                        return Ok(self.call_rt_1(self.rt.rc_get_len, v));
                    }
                    if fname == "push" && args.len() == 2 {
                        let xs = self.lower_expr(&args[0].value)?;
                        let val = self.lower_expr(&args[1].value)?;
                        return Ok(self.call_rt_2_vals(self.rt.list_push, xs, val));
                    }
                    if let Some(&tag) = self.enum_tags.get(fname) {
                        if !self.func_ids.contains_key(fname) {
                            let obj = self.call_rt_2(self.rt.rc_alloc, tag, args.len() as i64);
                            for (idx, a) in args.iter().enumerate() {
                                let av = self.lower_expr(&a.value)?;
                                self.call_rt_3_void(self.rt.rc_set_field, obj, idx as i64, av);
                            }
                            return Ok(obj);
                        }
                    }

                    let fid = *self
                        .func_ids
                        .get(fname)
                        .ok_or_else(|| format!("unknown function `{}` in codegen", fname))?;
                    let meta = self.fn_metas.get(fname).cloned();
                    let mut arg_vals = Vec::new();
                    let mut inout_writebacks: Vec<(String, cranelift_codegen::ir::StackSlot)> =
                        Vec::new();

                    for (idx, a) in args.iter().enumerate() {
                        let is_inout = meta
                            .as_ref()
                            .and_then(|m| m.params_inout.get(idx).copied())
                            .unwrap_or(false);
                        if is_inout {
                            if let ExprKind::Ident(vname) = &a.value.kind {
                                let cur_val = self.lower_expr(&a.value)?;
                                let ss = self.builder.create_sized_stack_slot(StackSlotData::new(
                                    StackSlotKind::ExplicitSlot,
                                    8,
                                    3,
                                ));
                                self.builder.ins().stack_store(cur_val, ss, 0);
                                let addr = self.builder.ins().stack_addr(types::I64, ss, 0);
                                arg_vals.push(addr);
                                inout_writebacks.push((vname.clone(), ss));
                            } else {
                                return Err(format!(
                                    "`inout` argument {} of `{}` must be a variable",
                                    idx, fname
                                ));
                            }
                        } else {
                            arg_vals.push(self.lower_expr(&a.value)?);
                        }
                    }
                    let local_callee = self.module.declare_func_in_func(fid, self.builder.func);
                    let call = self.builder.ins().call(local_callee, &arg_vals);
                    let ret = self.builder.inst_results(call)[0];

                    // Write back mutated `inout` handles to caller variables!
                    for (vname, ss) in inout_writebacks {
                        let updated = self.builder.ins().stack_load(types::I64, ss, 0);
                        let var = self.vars[&vname];
                        if self.inout_vars.contains(&vname) {
                            let slot_ptr = self.builder.use_var(var);
                            self.builder
                                .ins()
                                .store(MemFlags::new(), updated, slot_ptr, 0);
                        } else {
                            self.builder.def_var(var, updated);
                        }
                    }
                    Ok(ret)
                }
                _ => Err("unsupported indirect call in M1 codegen".into()),
            },
            _ => Ok(self.builder.ins().iconst(types::I64, 0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tenon_syntax::parse;

    #[test]
    fn cranelift_jits_and_runs_days_in_month_and_add_odds() {
        let src = r#"
pub type Odd = Int64 where self % 2 != 0
pub type Even = Int64 where self % 2 == 0

pub pure fn add_odds(a Odd, b Odd) Even {
    a + b
}

pub pure fn is_leap_year(y Int) Bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

pub pure fn days_in_month(y Int, m Int) Int {
    match m {
        2 => if is_leap_year(y) { 29 } else { 28 }
        4 | 6 | 9 | 11 => 30
        _ => 31
    }
}

pub fn main() Int {
    assert days_in_month(2024, 2) == 29
    assert days_in_month(2023, 2) == 28
    assert days_in_month(1900, 2) == 28
    assert days_in_month(2000, 2) == 29
    assert add_odds(3, 5) == 8
    days_in_month(2024, 2) + add_odds(3, 5)
}
"#;
        let prog = parse(src).unwrap();
        let jit = JitProgram::compile(&prog).unwrap();
        assert_eq!(jit.call_2("days_in_month", 2024, 2).unwrap(), 29);
        assert_eq!(jit.call_2("days_in_month", 1900, 2).unwrap(), 28);
        assert_eq!(jit.call_2("add_odds", 7, 9).unwrap(), 16);
        assert_eq!(jit.call_0("main").unwrap(), 37);
    }

    #[test]
    fn perceus_records_parse_mutate_cow_and_zero_leaks() {
        let src = r#"
pub pure fn is_leap_year(y Int) Bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

pub pure fn days_in_month(y Int, m Int where 1 <= self && self <= 12) Int {
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

pub fn shift_to_prev_year_safe(inout d Date)
    requires d.year > 1
{
    mutate d {
        d.year -= 1
        if d.month == 2 && d.day == 29 && !is_leap_year(d.year) {
            d.day = 28
        }
    }
}

pub fn main() Int {
    var d Date = Date { year = 2024, month = 2, day = 29 }
    shift_to_prev_year_safe(d)
    assert d.year == 2023
    assert d.day == 28

    let a Date = Date { year = 2024, month = 2, day = 29 }
    var b Date = a
    shift_to_prev_year_safe(b)
    assert a.year == 2024
    assert a.day == 29
    assert b.year == 2023
    assert b.day == 28

    let ok_day = match Date.parse(2024, 2, 29) {
        Ok(dt) => dt.day
        Err(code) => -code
    }
    let bad_code = match Date.parse(2023, 2, 29) {
        Ok(dt) => dt.day
        Err(code) => -code
    }
    assert ok_day == 29
    assert bad_code == -3

    let s = "hello"
    assert len(s) == 5
    var xs = [1, 2, 3]
    xs = push(xs, 4)
    assert len(xs) == 4
    assert xs[3] == 4

    ok_day + b.day + xs[3]
}
"#;
        perceus::reset_stats();
        let prog = parse(src).unwrap();
        let jit = JitProgram::compile(&prog).unwrap();
        assert!(jit.derived_parsers.contains(&"Date__parse".to_string()));
        let res = jit.call_0("main").unwrap();
        assert_eq!(res, 29 + 28 + 4);
        let stats = perceus::snapshot_stats();
        assert!(
            stats.reuse_hits >= 2,
            "expected Perceus unique in-place reuse hits, got {:?}",
            stats
        );
        assert!(
            stats.cow_clones >= 1,
            "expected MVS COW clone on shared Date, got {:?}",
            stats
        );
        assert_eq!(
            stats.live_objects, 0,
            "Perceus RC must leave 0 leaked heap objects, got {:?}",
            stats
        );
    }
}
