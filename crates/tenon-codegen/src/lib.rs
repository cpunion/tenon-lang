pub mod llvm_aot;
pub use tenon_rt::perceus;
pub mod wasm_client;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    AbiParam, Block as ClifBlock, InstBuilder, MemFlags, StackSlotData, StackSlotKind, TrapCode,
    Type, Value, types,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use perceus::*;
use std::collections::{BTreeMap, BTreeSet};
use tenon_rt::syscalls;
use tenon_syntax::*;

/// Minimal Tenon Core Byte & OS/Network Syscall Table (19 primitives).
/// Shared between Cranelift JIT (`lib.rs`) and LLVM 22 AOT (`llvm_aot.rs`).
pub(crate) struct CoreSyscallSpec {
    pub tenon_name: &'static str,
    pub c_symbol: &'static str,
    pub arity: usize,
    pub returns_rc: bool,
    pub returns_text: bool,
}

pub(crate) const CORE_SYSCALLS: &[CoreSyscallSpec] = &[
    // 1. Core byte / text primitives (4)
    CoreSyscallSpec { tenon_name: "byte_at", c_symbol: "tenon_byte_at", arity: 2, returns_rc: false, returns_text: false },
    CoreSyscallSpec { tenon_name: "str_slice", c_symbol: "tenon_str_slice", arity: 3, returns_rc: true, returns_text: true },
    CoreSyscallSpec { tenon_name: "byte_to_text", c_symbol: "tenon_byte_to_text", arity: 1, returns_rc: true, returns_text: true },
    CoreSyscallSpec { tenon_name: "str_concat", c_symbol: "tenon_str_concat", arity: 2, returns_rc: true, returns_text: true },
    // 2. Minimal OS / I/O / FS / Process syscalls (10)
    CoreSyscallSpec { tenon_name: "sys_write_stdout", c_symbol: "tenon_sys_write_stdout", arity: 1, returns_rc: false, returns_text: false },
    CoreSyscallSpec { tenon_name: "sys_read_line", c_symbol: "tenon_sys_read_line", arity: 0, returns_rc: true, returns_text: true },
    CoreSyscallSpec { tenon_name: "sys_is_eof", c_symbol: "tenon_sys_is_eof", arity: 0, returns_rc: false, returns_text: false },
    CoreSyscallSpec { tenon_name: "sys_env", c_symbol: "tenon_sys_env", arity: 1, returns_rc: true, returns_text: true },
    CoreSyscallSpec { tenon_name: "sys_cwd", c_symbol: "tenon_sys_cwd", arity: 0, returns_rc: true, returns_text: true },
    CoreSyscallSpec { tenon_name: "sys_read_file", c_symbol: "tenon_sys_read_file", arity: 1, returns_rc: true, returns_text: true },
    CoreSyscallSpec { tenon_name: "sys_file_ok", c_symbol: "tenon_sys_file_ok", arity: 0, returns_rc: false, returns_text: false },
    CoreSyscallSpec { tenon_name: "sys_write_file", c_symbol: "tenon_sys_write_file", arity: 2, returns_rc: false, returns_text: false },
    CoreSyscallSpec { tenon_name: "sys_exec", c_symbol: "tenon_sys_exec", arity: 1, returns_rc: true, returns_text: true },
    CoreSyscallSpec { tenon_name: "sys_exec_exit_code", c_symbol: "tenon_sys_exec_exit_code", arity: 0, returns_rc: false, returns_text: false },
    // 3. Minimal Raw HTTPS SSE line-stream transport syscalls (5)
    CoreSyscallSpec { tenon_name: "sys_http_sse_open", c_symbol: "tenon_sys_http_sse_open", arity: 3, returns_rc: false, returns_text: false },
    CoreSyscallSpec { tenon_name: "sys_http_sse_next_line", c_symbol: "tenon_sys_http_sse_next_line", arity: 1, returns_rc: true, returns_text: true },
    CoreSyscallSpec { tenon_name: "sys_http_sse_eof", c_symbol: "tenon_sys_http_sse_eof", arity: 1, returns_rc: false, returns_text: false },
    CoreSyscallSpec { tenon_name: "sys_http_sse_error", c_symbol: "tenon_sys_http_sse_error", arity: 1, returns_rc: true, returns_text: true },
    CoreSyscallSpec { tenon_name: "sys_http_sse_close", c_symbol: "tenon_sys_http_sse_close", arity: 1, returns_rc: false, returns_text: false },
];

#[derive(Debug, Clone)]
struct FieldLayout {
    name: String,
    preds: Vec<Expr>,
    is_rc: bool,
    is_text: bool,
    record_ty: Option<String>,
}

#[derive(Debug, Clone)]
struct RecordLayout {
    tag: i64,
    fields: Vec<FieldLayout>,
    rc_mask: i64,
    where_clauses: Vec<Expr>,
}

#[derive(Debug, Clone)]
struct FnMeta {
    params_inout: Vec<bool>,
    returns_rc: bool,
    returns_text: bool,
    ret_record: Option<String>,
}

#[derive(Clone, Copy)]
struct RtFuncs {
    assert_fail: FuncId,
    rc_alloc: FuncId,
    rc_set_rc_mask: FuncId,
    rc_inc: FuncId,
    rc_dec: FuncId,
    rc_ensure_unique: FuncId,
    rc_get_field: FuncId,
    rc_set_field: FuncId,
    rc_get_tag: FuncId,
    rc_get_len: FuncId,
    text_new: FuncId,
    text_eq: FuncId,
    str_concat: FuncId,
    list_push: FuncId,
    async_frame_alloc: FuncId,
    async_yield_create: FuncId,
    async_poll_frame: FuncId,
    async_block_on: FuncId,
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
        // Core Perceus / Async runtime symbols
        jit_builder.symbol("tenon_rt_assert_fail", tenon_rt_assert_fail as *const u8);
        jit_builder.symbol("tenon_rt_print_int", tenon_rt_print_int as *const u8);
        jit_builder.symbol("tenon_rc_alloc", tenon_rc_alloc as *const u8);
        jit_builder.symbol("tenon_rc_set_rc_mask", tenon_rc_set_rc_mask as *const u8);
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
        jit_builder.symbol("tenon_async_frame_alloc", tenon_async_frame_alloc as *const u8);
        jit_builder.symbol("tenon_async_yield_create", tenon_async_yield_create as *const u8);
        jit_builder.symbol("tenon_async_poll_frame", tenon_async_poll_frame as *const u8);
        jit_builder.symbol("tenon_async_block_on", tenon_async_block_on as *const u8);

        // Minimal 19 Core Byte & OS/Network Syscall Primitives
        jit_builder.symbol("tenon_byte_at", syscalls::tenon_byte_at as *const u8);
        jit_builder.symbol("tenon_str_slice", syscalls::tenon_str_slice as *const u8);
        jit_builder.symbol("tenon_byte_to_text", syscalls::tenon_byte_to_text as *const u8);
        jit_builder.symbol("tenon_str_concat", syscalls::tenon_str_concat as *const u8);
        jit_builder.symbol("tenon_sys_write_stdout", syscalls::tenon_sys_write_stdout as *const u8);
        jit_builder.symbol("tenon_sys_read_line", syscalls::tenon_sys_read_line as *const u8);
        jit_builder.symbol("tenon_sys_is_eof", syscalls::tenon_sys_is_eof as *const u8);
        jit_builder.symbol("tenon_sys_env", syscalls::tenon_sys_env as *const u8);
        jit_builder.symbol("tenon_sys_cwd", syscalls::tenon_sys_cwd as *const u8);
        jit_builder.symbol("tenon_sys_read_file", syscalls::tenon_sys_read_file as *const u8);
        jit_builder.symbol("tenon_sys_file_ok", syscalls::tenon_sys_file_ok as *const u8);
        jit_builder.symbol("tenon_sys_write_file", syscalls::tenon_sys_write_file as *const u8);
        jit_builder.symbol("tenon_sys_exec", syscalls::tenon_sys_exec as *const u8);
        jit_builder.symbol("tenon_sys_exec_exit_code", syscalls::tenon_sys_exec_exit_code as *const u8);
        jit_builder.symbol("tenon_sys_http_sse_open", syscalls::tenon_sys_http_sse_open as *const u8);
        jit_builder.symbol("tenon_sys_http_sse_next_line", syscalls::tenon_sys_http_sse_next_line as *const u8);
        jit_builder.symbol("tenon_sys_http_sse_eof", syscalls::tenon_sys_http_sse_eof as *const u8);
        jit_builder.symbol("tenon_sys_http_sse_error", syscalls::tenon_sys_http_sse_error as *const u8);
        jit_builder.symbol("tenon_sys_http_sse_close", syscalls::tenon_sys_http_sse_close as *const u8);

        let mut module = JITModule::new(jit_builder);

        let rt = declare_runtime_funcs(&mut module)?;
        let print_int_id = declare_sig(&mut module, "tenon_rt_print_int", 1, true)?;

        // Collect refined type alias predicates
        let mut alias_preds: BTreeMap<String, Expr> = BTreeMap::new();
        for item in &program.items {
            if let Item::TypeAlias(ta) = item {
                if let Some(p) = &ta.predicate {
                    alias_preds.insert(ta.name.clone(), p.clone());
                }
            }
        }

        // First collect all non-generic record names so nested record fields are recognized as RC!
        let record_names: BTreeSet<String> = program
            .items
            .iter()
            .filter_map(|i| match i {
                Item::Record(r) if r.type_params.is_empty() => Some(r.name.clone()),
                _ => None,
            })
            .collect();

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
                    let mut rc_mask = 0u64;
                    for (idx, f) in r.fields.iter().enumerate() {
                        let tname = f.ty.display_name();
                        let mut preds = Vec::new();
                        if let Some(ap) = alias_preds.get(&tname) {
                            preds.push(ap.clone());
                        }
                        if let Some(fp) = &f.predicate {
                            preds.push(fp.clone());
                        }
                        let is_text = tname == "Text";
                        let is_rc = is_rc_type_name(&tname, &record_names);
                        if is_rc && idx < 64 {
                            rc_mask |= 1u64 << idx;
                        }
                        let record_ty = if record_names.contains(&tname) {
                            Some(tname)
                        } else {
                            None
                        };
                        fields.push(FieldLayout {
                            name: f.name.clone(),
                            preds,
                            is_rc,
                            is_text,
                            record_ty,
                        });
                        field_to_idx.entry(f.name.clone()).or_insert(idx);
                    }
                    records.insert(
                        r.name.clone(),
                        RecordLayout {
                            tag: next_rec_tag,
                            fields,
                            rc_mask: rc_mask as i64,
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

        // Collect compilable sync AND async functions
        let fns: Vec<&FnDecl> = program
            .items
            .iter()
            .filter_map(|i| match i {
                Item::Fn(f) if !f.is_async && f.type_params.is_empty() => Some(f),
                _ => None,
            })
            .collect();
        let async_fns: Vec<&FnDecl> = program
            .items
            .iter()
            .filter_map(|i| match i {
                Item::Fn(f) if f.is_async && f.type_params.is_empty() => Some(f),
                _ => None,
            })
            .collect();

        let mut func_ids = BTreeMap::new();
        let mut fn_metas: BTreeMap<String, FnMeta> = BTreeMap::new();
        func_ids.insert("print_int".into(), print_int_id);
        declare_core_syscalls(&mut module, rt.str_concat, &mut func_ids, &mut fn_metas)?;

        // Declare stackless async functions
        for f in &async_fns {
            let mut create_sig = module.make_signature();
            for _ in &f.params {
                create_sig.params.push(AbiParam::new(types::I64));
            }
            create_sig.returns.push(AbiParam::new(types::I64));
            let create_id = module
                .declare_function(&f.name, Linkage::Export, &create_sig)
                .map_err(|e| e.to_string())?;
            func_ids.insert(f.name.clone(), create_id);
            fn_metas.insert(
                f.name.clone(),
                FnMeta {
                    params_inout: vec![false; f.params.len()],
                    returns_rc: true,
                    returns_text: false,
                    ret_record: None,
                },
            );

            let poll_name = format!("{}__poll", f.name);
            let mut poll_sig = module.make_signature();
            poll_sig.params.push(AbiParam::new(types::I64));
            poll_sig.returns.push(AbiParam::new(types::I64));
            let poll_id = module
                .declare_function(&poll_name, Linkage::Export, &poll_sig)
                .map_err(|e| e.to_string())?;
            func_ids.insert(poll_name, poll_id);
        }

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
            let ret_name = f.ret_ty.as_ref().map(|t| t.display_name());
            fn_metas.insert(
                f.name.clone(),
                FnMeta {
                    params_inout,
                    returns_rc: ret_name
                        .as_deref()
                        .is_some_and(|n| is_rc_type_name(n, &record_names)),
                    returns_text: ret_name.as_deref() == Some("Text"),
                    ret_record: ret_name.filter(|n| record_names.contains(n)),
                },
            );
        }

        // Declare auto-derived `<Record>__parse` boundary parsers
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
                    returns_text: false,
                    ret_record: None,
                },
            );
            derived_parsers.push(pname);
        }

        let mut interned_strings: Vec<Box<[u8]>> = Vec::new();
        let mut ctx = module.make_context();
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut compiled_fns = Vec::new();

        // Compile all user sync functions
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
                let mut borrowed_rc_vars = BTreeSet::new();

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
                    if is_rc_type_name(&tname, &record_names) {
                        borrowed_rc_vars.insert(p.name.clone());
                    }
                    if p.is_inout {
                        inout_vars.insert(p.name.clone());
                    }
                }

                let fn_returns_rc = fn_metas
                    .get(&f.name)
                    .is_some_and(|m| m.returns_rc);

                let mut lower = FnLower {
                    module: &mut module,
                    builder: &mut builder,
                    rt,
                    func_ids: &func_ids,
                    fn_metas: &fn_metas,
                    records: &records,
                    record_names: &record_names,
                    field_to_idx: &field_to_idx,
                    enum_tags: &enum_tags,
                    interned_strings: &mut interned_strings,
                    vars,
                    var_records,
                    inout_vars,
                    owned_rc_vars: BTreeSet::new(),
                    borrowed_rc_vars,
                    text_vars,
                    rc_scopes: Vec::new(),
                    temp_rc_scopes: Vec::new(),
                    loop_stack: Vec::new(),
                    fn_returns_rc,
                    self_var: None,
                    terminated: false,
                };
                let ret_val = lower.lower_block(&f.body, fn_returns_rc)?;
                if !lower.terminated {
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

        // Compile all auto-derived `<Record>__parse` functions
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
                let mut text_vars = BTreeSet::new();
                let mut borrowed_rc_vars = BTreeSet::new();
                for (idx, fl) in layout.fields.iter().enumerate() {
                    let var = builder.declare_var(types::I64);
                    let val = builder.block_params(entry_block)[idx];
                    builder.def_var(var, val);
                    vars.insert(fl.name.clone(), var);
                    if fl.is_text {
                        text_vars.insert(fl.name.clone());
                    }
                    if fl.is_rc {
                        borrowed_rc_vars.insert(fl.name.clone());
                    }
                }

                let mut lower = FnLower {
                    module: &mut module,
                    builder: &mut builder,
                    rt,
                    func_ids: &func_ids,
                    fn_metas: &fn_metas,
                    records: &records,
                    record_names: &record_names,
                    field_to_idx: &field_to_idx,
                    enum_tags: &enum_tags,
                    interned_strings: &mut interned_strings,
                    vars,
                    var_records: BTreeMap::new(),
                    inout_vars: BTreeSet::new(),
                    owned_rc_vars: BTreeSet::new(),
                    borrowed_rc_vars,
                    text_vars,
                    rc_scopes: Vec::new(),
                    temp_rc_scopes: Vec::new(),
                    loop_stack: Vec::new(),
                    fn_returns_rc: true,
                    self_var: None,
                    terminated: false,
                };

                for (idx, fl) in layout.fields.iter().enumerate() {
                    for pred in &fl.preds {
                        lower.self_var = Some(fl.name.clone());
                        let mut temps = Vec::new();
                        let cond_val = lower.lower_expr_borrowed(pred, &mut temps)?;
                        lower.flush_temps(&temps);
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

                for wc in &layout.where_clauses {
                    let mut temps = Vec::new();
                    let cond_val = lower.lower_expr_borrowed(wc, &mut temps)?;
                    lower.flush_temps(&temps);
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

                let rec_obj = lower.call_rt_2(rt.rc_alloc, layout.tag, layout.fields.len() as i64);
                if layout.rc_mask != 0 {
                    let mv = lower.builder.ins().iconst(types::I64, layout.rc_mask);
                    lower.call_rt_2_void(rt.rc_set_rc_mask, rec_obj, mv);
                }
                for (idx, fl) in layout.fields.iter().enumerate() {
                    let mut v = lower.builder.use_var(lower.vars[&fl.name]);
                    if fl.is_rc {
                        v = lower.call_rt_1(rt.rc_inc, v);
                    }
                    lower.call_rt_3_void(rt.rc_set_field, rec_obj, idx as i64, v);
                }
                let ok_obj = lower.call_rt_2(rt.rc_alloc, TAG_OK, 1);
                let one = lower.builder.ins().iconst(types::I64, 1);
                lower.call_rt_2_void(rt.rc_set_rc_mask, ok_obj, one);
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

        // Compile all stackless `async fn` state machines
        for f in &async_fns {
            let poll_name = format!("{}__poll", f.name);
            let create_id = func_ids[&f.name];
            let poll_id = func_ids[&poll_name];

            let mut frame_slots: BTreeMap<String, i64> = BTreeMap::new();
            let mut next_slot = 4i64;
            for p in &f.params {
                frame_slots.insert(p.name.clone(), next_slot);
                next_slot += 1;
            }
            for s in &f.body.stmts {
                if let Stmt::Let { name, .. } = s {
                    if !frame_slots.contains_key(name) {
                        frame_slots.insert(name.clone(), next_slot);
                        next_slot += 1;
                    }
                }
            }
            let segments = split_async_segments(&f.body, &mut frame_slots, &mut next_slot);
            let total_slots = next_slot;

            // 1) Constructor `<name>(params...) -> frame_ptr`
            ctx.func.signature = module
                .declarations()
                .get_function_decl(create_id)
                .signature
                .clone();
            {
                let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
                let entry = builder.create_block();
                builder.append_block_params_for_function_params(entry);
                builder.switch_to_block(entry);
                builder.seal_block(entry);

                let poll_local = module.declare_func_in_func(poll_id, builder.func);
                let poll_ptr = builder.ins().func_addr(types::I64, poll_local);
                let slots_v = builder.ins().iconst(types::I64, total_slots);
                let alloc_callee = module.declare_func_in_func(rt.async_frame_alloc, builder.func);
                let call = builder.ins().call(alloc_callee, &[slots_v, poll_ptr]);
                let frame = builder.inst_results(call)[0];

                let set_callee = module.declare_func_in_func(rt.rc_set_field, builder.func);
                for (idx, p) in f.params.iter().enumerate() {
                    let pv = builder.block_params(entry)[idx];
                    let slot = builder.ins().iconst(types::I64, frame_slots[&p.name]);
                    builder.ins().call(set_callee, &[frame, slot, pv]);
                }
                builder.ins().return_(&[frame]);
                builder.finalize();
            }
            module
                .define_function(create_id, &mut ctx)
                .map_err(|e| format!("Cranelift async ctor failed for `{}`: {}", f.name, e))?;
            module.clear_context(&mut ctx);

            // 2) Stackless State-Machine `<name>__poll(frame_ptr) -> i64`
            ctx.func.signature = module
                .declarations()
                .get_function_decl(poll_id)
                .signature
                .clone();
            {
                let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
                let entry = builder.create_block();
                builder.append_block_params_for_function_params(entry);
                builder.switch_to_block(entry);
                let frame = builder.block_params(entry)[0];

                let mut vars = BTreeMap::new();
                let mut lower = FnLower {
                    module: &mut module,
                    builder: &mut builder,
                    rt,
                    func_ids: &func_ids,
                    fn_metas: &fn_metas,
                    records: &records,
                    record_names: &record_names,
                    field_to_idx: &field_to_idx,
                    enum_tags: &enum_tags,
                    interned_strings: &mut interned_strings,
                    vars: BTreeMap::new(),
                    var_records: BTreeMap::new(),
                    inout_vars: BTreeSet::new(),
                    owned_rc_vars: BTreeSet::new(),
                    borrowed_rc_vars: BTreeSet::new(),
                    text_vars: BTreeSet::new(),
                    rc_scopes: Vec::new(),
                    temp_rc_scopes: Vec::new(),
                    loop_stack: Vec::new(),
                    fn_returns_rc: false,
                    self_var: None,
                    terminated: false,
                };

                let z0 = lower.builder.ins().iconst(types::I64, 0);
                let cur_state = lower.call_rt_2_vals(rt.rc_get_field, frame, z0);
                for (vname, &slot) in &frame_slots {
                    let sv = lower.builder.ins().iconst(types::I64, slot);
                    let val = lower.call_rt_2_vals(rt.rc_get_field, frame, sv);
                    let var = lower.builder.declare_var(types::I64);
                    lower.builder.def_var(var, val);
                    lower.vars.insert(vname.clone(), var);
                    vars.insert(vname.clone(), var);
                }

                let n_awaits = segments.await_steps.len();
                let mut start_blocks = Vec::new();
                let mut poll_blocks = Vec::new();
                for _ in 0..n_awaits {
                    start_blocks.push(lower.builder.create_block());
                    poll_blocks.push(lower.builder.create_block());
                }
                let final_block = lower.builder.create_block();

                for s in 0..n_awaits {
                    let st_start = lower.builder.ins().iconst(types::I64, (2 * s) as i64);
                    let cmp_s = lower.builder.ins().icmp(IntCC::Equal, cur_state, st_start);
                    let next_chk1 = lower.builder.create_block();
                    lower
                        .builder
                        .ins()
                        .brif(cmp_s, start_blocks[s], &[], next_chk1, &[]);
                    lower.builder.switch_to_block(next_chk1);

                    let st_poll = lower.builder.ins().iconst(types::I64, (2 * s + 1) as i64);
                    let cmp_p = lower.builder.ins().icmp(IntCC::Equal, cur_state, st_poll);
                    let next_chk2 = lower.builder.create_block();
                    lower
                        .builder
                        .ins()
                        .brif(cmp_p, poll_blocks[s], &[], next_chk2, &[]);
                    lower.builder.switch_to_block(next_chk2);
                }
                lower.builder.ins().jump(final_block, &[]);

                for (s, step) in segments.await_steps.iter().enumerate() {
                    lower.builder.switch_to_block(start_blocks[s]);
                    lower.terminated = false;
                    let pre_blk = Block {
                        stmts: step.pre_stmts.clone(),
                        tail: None,
                        span: f.span,
                    };
                    let _ = lower.lower_block(&pre_blk, false)?;
                    for (vname, &slot) in &frame_slots {
                        let cur_v = lower.builder.use_var(vars[vname]);
                        lower.call_rt_3_void(rt.rc_set_field, frame, slot, cur_v);
                    }
                    let child_frame = lower.lower_expr_owned(&step.child_expr)?;
                    lower.call_rt_3_void(rt.rc_set_field, frame, 3, child_frame);
                    let poll_state = lower.builder.ins().iconst(types::I64, (2 * s + 1) as i64);
                    lower.call_rt_3_void(rt.rc_set_field, frame, 0, poll_state);
                    lower.builder.ins().jump(poll_blocks[s], &[]);

                    lower.builder.switch_to_block(poll_blocks[s]);
                    lower.terminated = false;
                    let slot3 = lower.builder.ins().iconst(types::I64, 3);
                    let cframe = lower.call_rt_2_vals(rt.rc_get_field, frame, slot3);
                    let ready = lower.call_rt_1(rt.async_poll_frame, cframe);
                    let zero = lower.builder.ins().iconst(types::I64, 0);
                    let is_ready = lower.builder.ins().icmp(IntCC::NotEqual, ready, zero);
                    let resume_b = lower.builder.create_block();
                    let pending_b = lower.builder.create_block();
                    lower
                        .builder
                        .ins()
                        .brif(is_ready, resume_b, &[], pending_b, &[]);

                    lower.builder.switch_to_block(pending_b);
                    let pend_ret = lower.builder.ins().iconst(types::I64, 0);
                    lower.builder.ins().return_(&[pend_ret]);

                    lower.builder.switch_to_block(resume_b);
                    let slot2 = lower.builder.ins().iconst(types::I64, 2);
                    let child_res = lower.call_rt_2_vals(rt.rc_get_field, cframe, slot2);
                    lower.call_rt_1_void(rt.rc_dec, cframe);
                    let zclear = lower.builder.ins().iconst(types::I64, 0);
                    lower.call_rt_3_void(rt.rc_set_field, frame, 3, zclear);
                    if let Some(bname) = &step.bind_var {
                        let slot = frame_slots[bname];
                        lower.call_rt_3_void(rt.rc_set_field, frame, slot, child_res);
                        lower.builder.def_var(vars[bname], child_res);
                    }
                    let next_state = lower.builder.ins().iconst(types::I64, (2 * (s + 1)) as i64);
                    lower.call_rt_3_void(rt.rc_set_field, frame, 0, next_state);
                    let target_b = if s + 1 < n_awaits {
                        start_blocks[s + 1]
                    } else {
                        final_block
                    };
                    lower.builder.ins().jump(target_b, &[]);
                }

                lower.builder.switch_to_block(final_block);
                lower.terminated = false;
                let final_blk = Block {
                    stmts: segments.final_stmts.clone(),
                    tail: segments.final_tail.clone().map(Box::new),
                    span: f.span,
                };
                let final_val = lower.lower_block(&final_blk, true)?;
                lower.call_rt_3_void(rt.rc_set_field, frame, 2, final_val);
                let done_st = lower.builder.ins().iconst(types::I64, -1);
                lower.call_rt_3_void(rt.rc_set_field, frame, 0, done_st);
                let ready_ret = lower.builder.ins().iconst(types::I64, 1);
                lower.builder.ins().return_(&[ready_ret]);

                lower.builder.seal_all_blocks();
                builder.finalize();
            }
            module
                .define_function(poll_id, &mut ctx)
                .map_err(|e| format!("Cranelift async poll failed for `{}`: {}", f.name, e))?;
            module.clear_context(&mut ctx);
            compiled_fns.push(f.name.clone());
            compiled_fns.push(poll_name);
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
        perceus::clear_assert_failed();
        let id = *self
            .func_ids
            .get(name)
            .ok_or_else(|| format!("function `{}` not compiled", name))?;
        let ptr = self.module.get_finalized_function(id);
        let f: extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
        let res = f();
        let failed = perceus::last_assert_failed();
        if failed != 0 {
            return Err(format!("assertion failed at line {}", failed));
        }
        Ok(res)
    }

    pub fn call_2(&self, name: &str, a: i64, b: i64) -> Result<i64, String> {
        perceus::clear_assert_failed();
        let id = *self
            .func_ids
            .get(name)
            .ok_or_else(|| format!("function `{}` not compiled", name))?;
        let ptr = self.module.get_finalized_function(id);
        let f: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        let res = f(a, b);
        let failed = perceus::last_assert_failed();
        if failed != 0 {
            return Err(format!("assertion failed at line {}", failed));
        }
        Ok(res)
    }

    pub fn last_printed() -> i64 {
        perceus::last_printed()
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
        rc_set_rc_mask: declare_sig(module, "tenon_rc_set_rc_mask", 2, false)?,
        rc_inc: declare_sig(module, "tenon_rc_inc", 1, true)?,
        rc_dec: declare_sig(module, "tenon_rc_dec", 1, false)?,
        rc_ensure_unique: declare_sig(module, "tenon_rc_ensure_unique", 1, true)?,
        rc_get_field: declare_sig(module, "tenon_rc_get_field", 2, true)?,
        rc_set_field: declare_sig(module, "tenon_rc_set_field", 3, false)?,
        rc_get_tag: declare_sig(module, "tenon_rc_get_tag", 1, true)?,
        rc_get_len: declare_sig(module, "tenon_rc_get_len", 1, true)?,
        text_new: declare_sig(module, "tenon_text_new", 2, true)?,
        text_eq: declare_sig(module, "tenon_text_eq", 2, true)?,
        str_concat: declare_sig(module, "tenon_str_concat", 2, true)?,
        list_push: declare_sig(module, "tenon_list_push", 2, true)?,
        async_frame_alloc: declare_sig(module, "tenon_async_frame_alloc", 2, true)?,
        async_yield_create: declare_sig(module, "tenon_async_yield_create", 1, true)?,
        async_poll_frame: declare_sig(module, "tenon_async_poll_frame", 1, true)?,
        async_block_on: declare_sig(module, "tenon_async_block_on", 1, true)?,
    })
}

fn declare_core_syscalls(
    module: &mut JITModule,
    str_concat_id: FuncId,
    func_ids: &mut BTreeMap<String, FuncId>,
    fn_metas: &mut BTreeMap<String, FnMeta>,
) -> Result<(), String> {
    for spec in CORE_SYSCALLS {
        let fid = if spec.c_symbol == "tenon_str_concat" {
            str_concat_id
        } else {
            declare_sig(module, spec.c_symbol, spec.arity, true)?
        };
        func_ids.insert(spec.tenon_name.to_string(), fid);
        fn_metas.insert(
            spec.tenon_name.to_string(),
            FnMeta {
                params_inout: vec![false; spec.arity],
                returns_rc: spec.returns_rc,
                returns_text: spec.returns_text,
                ret_record: None,
            },
        );
    }
    Ok(())
}

pub(crate) struct AsyncAwaitStep {
    pub pre_stmts: Vec<Stmt>,
    pub bind_var: Option<String>,
    pub child_expr: Expr,
}

pub(crate) struct AsyncSegments {
    pub await_steps: Vec<AsyncAwaitStep>,
    pub final_stmts: Vec<Stmt>,
    pub final_tail: Option<Expr>,
}

pub(crate) fn split_async_segments(
    body: &Block,
    frame_slots: &mut BTreeMap<String, i64>,
    next_slot: &mut i64,
) -> AsyncSegments {
    let mut await_steps = Vec::new();
    let mut cur_stmts = Vec::new();

    for stmt in &body.stmts {
        match stmt {
            Stmt::Let {
                name,
                value:
                    Expr {
                        kind: ExprKind::Await(inner),
                        ..
                    },
                ..
            } => {
                if !frame_slots.contains_key(name) {
                    frame_slots.insert(name.clone(), *next_slot);
                    *next_slot += 1;
                }
                await_steps.push(AsyncAwaitStep {
                    pre_stmts: std::mem::take(&mut cur_stmts),
                    bind_var: Some(name.clone()),
                    child_expr: (**inner).clone(),
                });
            }
            Stmt::Expr {
                expr:
                    Expr {
                        kind: ExprKind::Await(inner),
                        ..
                    },
                ..
            } => {
                await_steps.push(AsyncAwaitStep {
                    pre_stmts: std::mem::take(&mut cur_stmts),
                    bind_var: None,
                    child_expr: (**inner).clone(),
                });
            }
            other => cur_stmts.push(other.clone()),
        }
    }

    let final_tail = match body.tail.as_deref() {
        Some(Expr {
            kind: ExprKind::Await(inner),
            span,
        }) => {
            let synth = "__await_tail".to_string();
            if !frame_slots.contains_key(&synth) {
                frame_slots.insert(synth.clone(), *next_slot);
                *next_slot += 1;
            }
            await_steps.push(AsyncAwaitStep {
                pre_stmts: std::mem::take(&mut cur_stmts),
                bind_var: Some(synth.clone()),
                child_expr: (**inner).clone(),
            });
            Some(Expr {
                kind: ExprKind::Ident(synth),
                span: *span,
            })
        }
        Some(e) => Some(e.clone()),
        None => None,
    };

    AsyncSegments {
        await_steps,
        final_stmts: cur_stmts,
        final_tail,
    }
}

pub(crate) fn is_rc_type_name(name: &str, record_names: &BTreeSet<String>) -> bool {
    record_names.contains(name)
        || name == "Text"
        || name.starts_with("List[")
        || name.starts_with("Option[")
        || name.starts_with("Result[")
}

struct LoopCtx {
    hdr_blk: ClifBlock,
    exit_blk: ClifBlock,
    rc_scope_depth: usize,
    temp_scope_depth: usize,
}

struct SavedLexical {
    vars: BTreeMap<String, Variable>,
    var_records: BTreeMap<String, String>,
    owned_rc_vars: BTreeSet<String>,
    borrowed_rc_vars: BTreeSet<String>,
    text_vars: BTreeSet<String>,
}

struct FnLower<'a, 'b> {
    module: &'a mut JITModule,
    builder: &'a mut FunctionBuilder<'b>,
    rt: RtFuncs,
    func_ids: &'a BTreeMap<String, FuncId>,
    fn_metas: &'a BTreeMap<String, FnMeta>,
    records: &'a BTreeMap<String, RecordLayout>,
    record_names: &'a BTreeSet<String>,
    field_to_idx: &'a BTreeMap<String, usize>,
    enum_tags: &'a BTreeMap<String, i64>,
    interned_strings: &'a mut Vec<Box<[u8]>>,
    vars: BTreeMap<String, Variable>,
    var_records: BTreeMap<String, String>,
    inout_vars: BTreeSet<String>,
    owned_rc_vars: BTreeSet<String>,
    borrowed_rc_vars: BTreeSet<String>,
    text_vars: BTreeSet<String>,
    rc_scopes: Vec<Vec<(String, Variable)>>,
    temp_rc_scopes: Vec<Vec<Value>>,
    loop_stack: Vec<LoopCtx>,
    fn_returns_rc: bool,
    self_var: Option<String>,
    terminated: bool,
}

impl<'a, 'b> FnLower<'a, 'b> {
    fn save_lexical(&self) -> SavedLexical {
        SavedLexical {
            vars: self.vars.clone(),
            var_records: self.var_records.clone(),
            owned_rc_vars: self.owned_rc_vars.clone(),
            borrowed_rc_vars: self.borrowed_rc_vars.clone(),
            text_vars: self.text_vars.clone(),
        }
    }

    fn restore_lexical(&mut self, saved: SavedLexical) {
        self.vars = saved.vars;
        self.var_records = saved.var_records;
        self.owned_rc_vars = saved.owned_rc_vars;
        self.borrowed_rc_vars = saved.borrowed_rc_vars;
        self.text_vars = saved.text_vars;
    }

    fn declare_fresh_var(&mut self, name: &str, ty: Type) -> Variable {
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

    fn call_rt_2_void(&mut self, id: FuncId, va: Value, vb: Value) {
        let callee = self.module.declare_func_in_func(id, self.builder.func);
        self.builder.ins().call(callee, &[va, vb]);
    }

    fn call_rt_3_void(&mut self, id: FuncId, obj: Value, idx: i64, val: Value) {
        let vidx = self.builder.ins().iconst(types::I64, idx);
        let callee = self.module.declare_func_in_func(id, self.builder.func);
        self.builder.ins().call(callee, &[obj, vidx, val]);
    }

    fn flush_temps(&mut self, temps: &[Value]) {
        for &v in temps.iter().rev() {
            self.call_rt_1_void(self.rt.rc_dec, v);
        }
    }

    fn emit_cleanup_scopes_from(&mut self, from_depth: usize, skip_var: Option<Variable>) {
        let scopes: Vec<Vec<(String, Variable)>> = self.rc_scopes[from_depth..].to_vec();
        for scope in scopes.iter().rev() {
            for (_, var) in scope.iter().rev() {
                if skip_var == Some(*var) {
                    continue;
                }
                let handle = self.builder.use_var(*var);
                self.call_rt_1_void(self.rt.rc_dec, handle);
            }
        }
    }

    fn emit_cleanup_temp_scopes_from(&mut self, from_depth: usize) {
        let scopes: Vec<Vec<Value>> = self.temp_rc_scopes[from_depth..].to_vec();
        for scope in scopes.iter().rev() {
            for &val in scope.iter().rev() {
                self.call_rt_1_void(self.rt.rc_dec, val);
            }
        }
    }

    fn find_owned_var_in_scopes(&self, var: Variable, from_depth: usize) -> bool {
        self.rc_scopes[from_depth..]
            .iter()
            .any(|sc| sc.iter().any(|(_, v)| *v == var))
    }

    fn resolve_field_layout(&self, obj_expr: &Expr, field: &str) -> Result<(usize, Option<FieldLayout>), String> {
        if let Some(rname) = self.expr_record_name(obj_expr) {
            if let Some(layout) = self.records.get(&rname) {
                if let Some((pos, fl)) = layout
                    .fields
                    .iter()
                    .enumerate()
                    .find(|(_, f)| f.name == field)
                {
                    return Ok((pos, Some(fl.clone())));
                }
            }
        }
        let idx = self
            .field_to_idx
            .get(field)
            .copied()
            .ok_or_else(|| format!("unknown record field `{}` in codegen", field))?;
        Ok((idx, None))
    }

    fn expr_record_name(&self, expr: &Expr) -> Option<String> {
        match &expr.kind {
            ExprKind::Ident(vname) => self.var_records.get(vname).cloned(),
            ExprKind::RecordLit { name, .. } => Some(name.clone()),
            ExprKind::Call { callee, .. } => match &callee.kind {
                ExprKind::Ident(f) => self.fn_metas.get(f).and_then(|m| m.ret_record.clone()),
                _ => None,
            },
            ExprKind::Field { target, field } => {
                if let Some(rname) = self.expr_record_name(target) {
                    if let Some(layout) = self.records.get(&rname) {
                        if let Some(fl) = layout.fields.iter().find(|f| f.name == *field) {
                            return fl.record_ty.clone();
                        }
                    }
                }
                None
            }
            ExprKind::Block(blk) => blk.tail.as_deref().and_then(|t| self.expr_record_name(t)),
            ExprKind::If { then_block, else_branch, .. } => then_block
                .tail
                .as_deref()
                .and_then(|t| self.expr_record_name(t))
                .or_else(|| else_branch.as_deref().and_then(|e| self.expr_record_name(e))),
            _ => None,
        }
    }

    fn is_rc_ident(&self, s: &str) -> bool {
        self.owned_rc_vars.contains(s)
            || self.borrowed_rc_vars.contains(s)
            || self.var_records.contains_key(s)
            || self.text_vars.contains(s)
    }

    fn expr_produces_rc(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::RecordLit { .. } | ExprKind::List(_) | ExprKind::Str(_) => true,
            ExprKind::Binary { op: BinOp::Concat, .. } => true,
            ExprKind::Ident(s) => s == "None" || self.is_rc_ident(s) || (self.enum_tags.contains_key(s) && !self.vars.contains_key(s)),
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    if self.enum_tags.contains_key(&format!("{}.{}", ename, field)) {
                        return true;
                    }
                }
                if let Ok((_, Some(fl))) = self.resolve_field_layout(target, field) {
                    return fl.is_rc;
                }
                false
            }
            ExprKind::Call { callee, .. } => match &callee.kind {
                ExprKind::Ident(f) => {
                    matches!(f.as_str(), "Ok" | "Err" | "Some" | "push" | "yield_val")
                        || (self.enum_tags.contains_key(f) && !self.func_ids.contains_key(f))
                        || self.fn_metas.get(f).is_some_and(|m| m.returns_rc)
                }
                ExprKind::Field { field, .. } => field == "parse",
                _ => false,
            },
            ExprKind::If { then_block, else_branch, .. } => {
                else_branch.is_some()
                    && (then_block.tail.as_deref().is_some_and(|t| self.expr_produces_rc(t))
                        || else_branch.as_deref().is_some_and(|e| self.expr_produces_rc(e)))
            }
            ExprKind::Match { arms, .. } => arms.iter().any(|a| self.expr_produces_rc(&a.body)),
            ExprKind::Block(blk) => blk.tail.as_deref().is_some_and(|t| self.expr_produces_rc(t)),
            _ => false,
        }
    }

    fn is_text_expr(&self, expr: &Expr) -> bool {
        match &expr.kind {
            ExprKind::Str(_) => true,
            ExprKind::Binary { op: BinOp::Concat, .. } => true,
            ExprKind::Ident(s) => self.text_vars.contains(s),
            ExprKind::Field { target, field } => {
                if let Ok((_, Some(fl))) = self.resolve_field_layout(target, field) {
                    return fl.is_text;
                }
                false
            }
            ExprKind::Call { callee, .. } => match &callee.kind {
                ExprKind::Ident(f) => self.fn_metas.get(f).is_some_and(|m| m.returns_text),
                _ => false,
            },
            ExprKind::If { then_block, else_branch, .. } => {
                then_block.tail.as_deref().is_some_and(|t| self.is_text_expr(t))
                    || else_branch.as_deref().is_some_and(|e| self.is_text_expr(e))
            }
            ExprKind::Match { arms, .. } => arms.iter().any(|a| self.is_text_expr(&a.body)),
            ExprKind::Block(blk) => blk.tail.as_deref().is_some_and(|t| self.is_text_expr(t)),
            _ => false,
        }
    }

    fn lower_block(&mut self, blk: &Block, want_owned_tail: bool) -> Result<Value, String> {
        let saved = self.save_lexical();
        let scope_idx = self.rc_scopes.len();
        self.rc_scopes.push(Vec::new());

        let mut last = self.builder.ins().iconst(types::I64, 0);
        for stmt in &blk.stmts {
            if self.terminated {
                break;
            }
            match stmt {
                Stmt::Let {
                    name, ty, value, ..
                } => {
                    let produces_rc = if let Some(t) = ty {
                        is_rc_type_name(&t.display_name(), self.record_names)
                    } else {
                        self.expr_produces_rc(value)
                    };
                    let v = if produces_rc {
                        self.lower_expr_owned(value)?
                    } else {
                        let mut temps = Vec::new();
                        let val = self.lower_expr_borrowed(value, &mut temps)?;
                        self.flush_temps(&temps);
                        val
                    };
                    let var = self.declare_fresh_var(name, types::I64);
                    self.builder.def_var(var, v);

                    // Update lexical type metadata for this new binding
                    self.var_records.remove(name);
                    self.text_vars.remove(name);
                    self.owned_rc_vars.remove(name);
                    self.borrowed_rc_vars.remove(name);

                    if let Some(t) = ty {
                        let tname = t.display_name();
                        if self.records.contains_key(&tname) {
                            self.var_records.insert(name.clone(), tname);
                            self.owned_rc_vars.insert(name.clone());
                            self.rc_scopes[scope_idx].push((name.clone(), var));
                        } else if tname == "Text" {
                            self.text_vars.insert(name.clone());
                            self.owned_rc_vars.insert(name.clone());
                            self.rc_scopes[scope_idx].push((name.clone(), var));
                        } else if is_rc_type_name(&tname, self.record_names) {
                            self.owned_rc_vars.insert(name.clone());
                            self.rc_scopes[scope_idx].push((name.clone(), var));
                        }
                    } else {
                        if let Some(rname) = self.expr_record_name(value) {
                            self.var_records.insert(name.clone(), rname);
                        }
                        if self.is_text_expr(value) {
                            self.text_vars.insert(name.clone());
                        }
                        if produces_rc {
                            self.owned_rc_vars.insert(name.clone());
                            self.rc_scopes[scope_idx].push((name.clone(), var));
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
                        let target_is_rc = self.owned_rc_vars.contains(name);
                        let rhs = if consuming_push {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(value, &mut temps)?;
                            // `push` consumes its first arg and returns a new owned list; do NOT dec `v`!
                            if temps.last().copied() == Some(v) {
                                temps.pop();
                            }
                            self.flush_temps(&temps);
                            v
                        } else if target_is_rc && *op == AssignOp::Set {
                            self.lower_expr_owned(value)?
                        } else {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(value, &mut temps)?;
                            self.flush_temps(&temps);
                            v
                        };

                        let var = *self
                            .vars
                            .get(name)
                            .ok_or_else(|| format!("undefined variable `{}` in assignment", name))?;
                        let is_inout = self.inout_vars.contains(name);
                        if target_is_rc && !consuming_push && !is_inout {
                            let old = self.builder.use_var(var);
                            self.builder.def_var(var, rhs);
                            self.call_rt_1_void(self.rt.rc_dec, old);
                        } else if is_inout {
                            let slot_ptr = self.builder.use_var(var);
                            let new_v = match op {
                                AssignOp::Set => rhs,
                                AssignOp::AddSet => {
                                    let cur = self.builder.ins().load(types::I64, MemFlags::new(), slot_ptr, 0);
                                    self.builder.ins().iadd(cur, rhs)
                                }
                                AssignOp::SubSet => {
                                    let cur = self.builder.ins().load(types::I64, MemFlags::new(), slot_ptr, 0);
                                    self.builder.ins().isub(cur, rhs)
                                }
                            };
                            self.builder.ins().store(MemFlags::new(), new_v, slot_ptr, 0);
                        } else {
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
                    }
                    ExprKind::Field { target: obj, field } => {
                        if let ExprKind::Ident(vname) = &obj.kind {
                            let (fidx, fl) = self.resolve_field_layout(obj, field)?;
                            let field_is_rc = fl.as_ref().is_some_and(|f| f.is_rc);
                            let rhs = if field_is_rc && *op == AssignOp::Set {
                                self.lower_expr_owned(value)?
                            } else {
                                let mut temps = Vec::new();
                                let v = self.lower_expr_borrowed(value, &mut temps)?;
                                self.flush_temps(&temps);
                                v
                            };
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
                    let _ = self.lower_block(body, false)?;
                }
                Stmt::Return { value, .. } => {
                    let mut move_skip_var: Option<Variable> = None;
                    let v = match value {
                        Some(e) => {
                            if self.fn_returns_rc {
                                if let ExprKind::Ident(rname) = &e.kind {
                                    if !self.inout_vars.contains(rname) {
                                        if let Some(&rv) = self.vars.get(rname) {
                                            if self.find_owned_var_in_scopes(rv, 0) {
                                                move_skip_var = Some(rv);
                                                self.builder.use_var(rv)
                                            } else {
                                                self.lower_expr_owned(e)?
                                            }
                                        } else {
                                            self.lower_expr_owned(e)?
                                        }
                                    } else {
                                        self.lower_expr_owned(e)?
                                    }
                                } else {
                                    self.lower_expr_owned(e)?
                                }
                            } else {
                                let mut temps = Vec::new();
                                let val = self.lower_expr_borrowed(e, &mut temps)?;
                                self.flush_temps(&temps);
                                val
                            }
                        }
                        None => self.builder.ins().iconst(types::I64, 0),
                    };
                    self.emit_cleanup_temp_scopes_from(0);
                    self.emit_cleanup_scopes_from(0, move_skip_var);
                    self.builder.ins().return_(&[v]);
                    self.terminated = true;
                    last = v;
                    break;
                }
                Stmt::Break { .. } => {
                    let (exit_blk, rc_depth, temp_depth) = self
                        .loop_stack
                        .last()
                        .map(|l| (l.exit_blk, l.rc_scope_depth, l.temp_scope_depth))
                        .ok_or_else(|| "`break` used outside of `while` loop".to_string())?;
                    self.emit_cleanup_temp_scopes_from(temp_depth);
                    self.emit_cleanup_scopes_from(rc_depth, None);
                    self.builder.ins().jump(exit_blk, &[]);
                    self.terminated = true;
                    break;
                }
                Stmt::Continue { .. } => {
                    let (hdr_blk, rc_depth, temp_depth) = self
                        .loop_stack
                        .last()
                        .map(|l| (l.hdr_blk, l.rc_scope_depth, l.temp_scope_depth))
                        .ok_or_else(|| "`continue` used outside of `while` loop".to_string())?;
                    self.emit_cleanup_temp_scopes_from(temp_depth);
                    self.emit_cleanup_scopes_from(rc_depth, None);
                    self.builder.ins().jump(hdr_blk, &[]);
                    self.terminated = true;
                    break;
                }
                Stmt::Assert { cond, span } => {
                    let mut temps = Vec::new();
                    let c = self.lower_expr_borrowed(cond, &mut temps)?;
                    self.flush_temps(&temps);
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
                    let mut cond_temps = Vec::new();
                    let c = self.lower_expr_borrowed(cond, &mut cond_temps)?;
                    self.flush_temps(&cond_temps);
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    let cmp = self.builder.ins().icmp(IntCC::NotEqual, c, zero);
                    self.builder
                        .ins()
                        .brif(cmp, loop_blk, &[], exit_blk, &[]);
                    self.builder.switch_to_block(loop_blk);
                    self.builder.seal_block(loop_blk);

                    self.loop_stack.push(LoopCtx {
                        hdr_blk: hdr,
                        exit_blk,
                        rc_scope_depth: self.rc_scopes.len(),
                        temp_scope_depth: self.temp_rc_scopes.len(),
                    });
                    let _ = self.lower_block(body, false)?;
                    self.loop_stack.pop();

                    if !self.terminated {
                        self.builder.ins().jump(hdr, &[]);
                    }
                    self.terminated = false;
                    self.builder.seal_block(hdr);
                    self.builder.switch_to_block(exit_blk);
                    self.builder.seal_block(exit_blk);
                }
                Stmt::Expr { expr, .. } => {
                    let mut temps = Vec::new();
                    last = self.lower_expr_borrowed(expr, &mut temps)?;
                    if !self.terminated {
                        self.flush_temps(&temps);
                    }
                }
            }
        }

        let mut move_skip_var: Option<Variable> = None;
        if !self.terminated {
            if let Some(tail) = &blk.tail {
                if want_owned_tail {
                    if let ExprKind::Ident(tname) = &tail.kind {
                        if !self.inout_vars.contains(tname) {
                            if let Some(&tv) = self.vars.get(tname) {
                                if self.find_owned_var_in_scopes(tv, scope_idx) {
                                    move_skip_var = Some(tv);
                                    last = self.builder.use_var(tv);
                                } else {
                                    last = self.lower_expr_owned(tail)?;
                                }
                            } else {
                                last = self.lower_expr_owned(tail)?;
                            }
                        } else {
                            last = self.lower_expr_owned(tail)?;
                        }
                    } else {
                        last = self.lower_expr_owned(tail)?;
                    }
                } else {
                    let mut temps = Vec::new();
                    last = self.lower_expr_borrowed(tail, &mut temps)?;
                    if !self.terminated {
                        self.flush_temps(&temps);
                    }
                }
            }
            if !self.terminated {
                self.emit_cleanup_scopes_from(scope_idx, move_skip_var);
            }
        }

        self.rc_scopes.pop();
        self.restore_lexical(saved);
        Ok(last)
    }

    fn lower_expr_owned(&mut self, expr: &Expr) -> Result<Value, String> {
        match &expr.kind {
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
                let val = self.load_ident_val(s)?;
                if self.is_rc_ident(s) {
                    Ok(self.call_rt_1(self.rt.rc_inc, val))
                } else {
                    Ok(val)
                }
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    let key = format!("{}.{}", ename, field);
                    if let Some(&tag) = self.enum_tags.get(&key) {
                        return Ok(self.call_rt_2(self.rt.rc_alloc, tag, 0));
                    }
                }
                let (idx, fl) = self.resolve_field_layout(target, field)?;
                let mut temps = Vec::new();
                let obj = self.lower_expr_borrowed(target, &mut temps)?;
                let idx_v = self.builder.ins().iconst(types::I64, idx as i64);
                let mut val = self.call_rt_2_vals(self.rt.rc_get_field, obj, idx_v);
                if fl.as_ref().is_some_and(|f| f.is_rc) {
                    val = self.call_rt_1(self.rt.rc_inc, val);
                }
                self.flush_temps(&temps);
                Ok(val)
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => self.lower_if(cond, then_block, else_branch.as_deref(), true),
            ExprKind::Match { scrutinee, arms } => self.lower_match(scrutinee, arms, true),
            ExprKind::Block(blk) => self.lower_block(blk, true),
            _ => self.lower_expr_fresh(expr),
        }
    }

    fn lower_expr_borrowed(&mut self, expr: &Expr, temps: &mut Vec<Value>) -> Result<Value, String> {
        match &expr.kind {
            ExprKind::Int(n) => Ok(self.builder.ins().iconst(types::I64, *n)),
            ExprKind::Bool(b) => Ok(self
                .builder
                .ins()
                .iconst(types::I64, if *b { 1 } else { 0 })),
            ExprKind::Ident(s) => {
                if s == "self" {
                    if let Some(sv) = self.self_var.clone() {
                        let var = self.vars[&sv];
                        return Ok(self.builder.use_var(var));
                    }
                }
                if s == "None" {
                    let v = self.call_rt_2(self.rt.rc_alloc, TAG_NONE, 0);
                    temps.push(v);
                    return Ok(v);
                }
                if let Some(&etag) = self.enum_tags.get(s) {
                    if !self.vars.contains_key(s) {
                        let v = self.call_rt_2(self.rt.rc_alloc, etag, 0);
                        temps.push(v);
                        return Ok(v);
                    }
                }
                self.load_ident_val(s)
            }
            ExprKind::Field { target, field } => {
                if let ExprKind::Ident(ename) = &target.kind {
                    let key = format!("{}.{}", ename, field);
                    if let Some(&tag) = self.enum_tags.get(&key) {
                        let v = self.call_rt_2(self.rt.rc_alloc, tag, 0);
                        temps.push(v);
                        return Ok(v);
                    }
                }
                let (idx, _) = self.resolve_field_layout(target, field)?;
                let obj = self.lower_expr_borrowed(target, temps)?;
                let idx_v = self.builder.ins().iconst(types::I64, idx as i64);
                Ok(self.call_rt_2_vals(self.rt.rc_get_field, obj, idx_v))
            }
            ExprKind::If {
                cond,
                then_block,
                else_branch,
            } => {
                let want_owned = self.expr_produces_rc(expr);
                let v = self.lower_if(cond, then_block, else_branch.as_deref(), want_owned)?;
                if want_owned && !self.terminated {
                    temps.push(v);
                }
                Ok(v)
            }
            ExprKind::Match { scrutinee, arms } => {
                let want_owned = self.expr_produces_rc(expr);
                let v = self.lower_match(scrutinee, arms, want_owned)?;
                if want_owned && !self.terminated {
                    temps.push(v);
                }
                Ok(v)
            }
            ExprKind::Block(blk) => {
                let want_owned = self.expr_produces_rc(expr);
                let v = self.lower_block(blk, want_owned)?;
                if want_owned && !self.terminated {
                    temps.push(v);
                }
                Ok(v)
            }
            _ => {
                let v = self.lower_expr_fresh(expr)?;
                if self.expr_produces_rc(expr) {
                    temps.push(v);
                }
                Ok(v)
            }
        }
    }

    fn load_ident_val(&mut self, s: &str) -> Result<Value, String> {
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

    fn lower_if(
        &mut self,
        cond: &Expr,
        then_block: &Block,
        else_branch: Option<&Expr>,
        want_owned: bool,
    ) -> Result<Value, String> {
        let mut ctemps = Vec::new();
        let c = self.lower_expr_borrowed(cond, &mut ctemps)?;
        self.flush_temps(&ctemps);

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
        let tv = self.lower_block(then_block, want_owned)?;
        let then_term = self.terminated;
        if !then_term {
            self.builder.ins().jump(merge_b, &[tv.into()]);
        }
        self.terminated = false;

        self.builder.switch_to_block(else_b);
        self.builder.seal_block(else_b);
        let ev = match else_branch {
            Some(eb) => {
                if want_owned {
                    self.lower_expr_owned(eb)?
                } else {
                    let mut etemps = Vec::new();
                    let v = self.lower_expr_borrowed(eb, &mut etemps)?;
                    if !self.terminated {
                        self.flush_temps(&etemps);
                    }
                    v
                }
            }
            None => self.builder.ins().iconst(types::I64, 0),
        };
        let else_term = self.terminated;
        if !else_term {
            self.builder.ins().jump(merge_b, &[ev.into()]);
        }

        self.builder.switch_to_block(merge_b);
        self.builder.seal_block(merge_b);
        if then_term && else_term {
            self.builder.ins().trap(TrapCode::unwrap_user(1));
            self.terminated = true;
            return Ok(self.builder.block_params(merge_b)[0]);
        }
        self.terminated = false;
        Ok(self.builder.block_params(merge_b)[0])
    }

    fn lower_match(
        &mut self,
        scrutinee: &Expr,
        arms: &[MatchArm],
        want_owned: bool,
    ) -> Result<Value, String> {
        let parse_record_name = match &scrutinee.kind {
            ExprKind::Call { callee, .. } => match &callee.kind {
                ExprKind::Field { target, field } if field == "parse" => match &target.kind {
                    ExprKind::Ident(rname) => Some(rname.clone()),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        };
        let is_variant_match = arms
            .iter()
            .any(|a| a.patterns.iter().any(|p| matches!(p, Pattern::Variant { .. })));
        let is_str_match = arms
            .iter()
            .any(|a| a.patterns.iter().any(|p| matches!(p, Pattern::Str(_, _))));

        let mut match_temps = Vec::new();
        let sv = self.lower_expr_borrowed(scrutinee, &mut match_temps)?;
        let temp_scope_idx = self.temp_rc_scopes.len();
        self.temp_rc_scopes.push(match_temps);

        let cmp_target = if is_variant_match {
            self.call_rt_1(self.rt.rc_get_tag, sv)
        } else {
            sv
        };
        let merge_b = self.builder.create_block();
        self.builder.append_block_param(merge_b, types::I64);
        let mut any_reachable_merge = false;

        for arm in arms {
            let is_wild = arm
                .patterns
                .iter()
                .any(|p| matches!(p, Pattern::Wildcard(_)));
            if is_wild {
                let v = if want_owned {
                    self.lower_expr_owned(&arm.body)?
                } else {
                    let mut atemps = Vec::new();
                    let av = self.lower_expr_borrowed(&arm.body, &mut atemps)?;
                    if !self.terminated {
                        self.flush_temps(&atemps);
                    }
                    av
                };
                if !self.terminated {
                    self.builder.ins().jump(merge_b, &[v.into()]);
                    any_reachable_merge = true;
                }
                self.terminated = false;
                self.builder.switch_to_block(merge_b);
                self.builder.seal_block(merge_b);
                let mt = self.temp_rc_scopes.pop().unwrap();
                if !any_reachable_merge {
                    self.builder.ins().trap(TrapCode::unwrap_user(2));
                    self.terminated = true;
                    return Ok(self.builder.block_params(merge_b)[0]);
                }
                self.flush_temps(&mt);
                return Ok(self.builder.block_params(merge_b)[0]);
            }

            let mut cond: Option<Value> = None;
            let mut variant_binding: Option<(String, bool)> = None;
            for p in &arm.patterns {
                let eq = match p {
                    Pattern::Int(n, _) => {
                        let pv = self.builder.ins().iconst(types::I64, *n);
                        self.builder.ins().icmp(IntCC::Equal, cmp_target, pv)
                    }
                    Pattern::Bool(b, _) => {
                        let pv = self.builder.ins().iconst(types::I64, if *b { 1 } else { 0 });
                        self.builder.ins().icmp(IntCC::Equal, cmp_target, pv)
                    }
                    Pattern::Str(s, _) if is_str_match => {
                        let boxed: Box<[u8]> = s.as_bytes().to_vec().into_boxed_slice();
                        let ptr = boxed.as_ptr() as i64;
                        let len = boxed.len() as i64;
                        self.interned_strings.push(boxed);
                        let ptxt = self.call_rt_2(self.rt.text_new, ptr, len);
                        let teq = self.call_rt_2_vals(self.rt.text_eq, cmp_target, ptxt);
                        self.call_rt_1_void(self.rt.rc_dec, ptxt);
                        let zero = self.builder.ins().iconst(types::I64, 0);
                        self.builder.ins().icmp(IntCC::NotEqual, teq, zero)
                    }
                    Pattern::Variant { name, binding, .. } => {
                        let tag = *self
                            .enum_tags
                            .get(name)
                            .ok_or_else(|| format!("unknown enum variant `{}` in match", name))?;
                        if let Some(bname) = binding {
                            variant_binding = Some((bname.clone(), name == "Ok"));
                        }
                        let pv = self.builder.ins().iconst(types::I64, tag);
                        self.builder.ins().icmp(IntCC::Equal, cmp_target, pv)
                    }
                    _ => continue,
                };
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
            let saved_arm = self.save_lexical();
            if let Some((bname, is_ok)) = variant_binding {
                let zero = self.builder.ins().iconst(types::I64, 0);
                let payload = self.call_rt_2_vals(self.rt.rc_get_field, sv, zero);
                let bvar = self.declare_fresh_var(&bname, types::I64);
                self.builder.def_var(bvar, payload);
                if is_ok {
                    if let Some(rname) = &parse_record_name {
                        self.var_records.insert(bname.clone(), rname.clone());
                        self.borrowed_rc_vars.insert(bname);
                    }
                }
            }
            let v = if want_owned {
                self.lower_expr_owned(&arm.body)?
            } else {
                let mut atemps = Vec::new();
                let av = self.lower_expr_borrowed(&arm.body, &mut atemps)?;
                if !self.terminated {
                    self.flush_temps(&atemps);
                }
                av
            };
            self.restore_lexical(saved_arm);
            if !self.terminated {
                self.builder.ins().jump(merge_b, &[v.into()]);
                any_reachable_merge = true;
            }
            self.terminated = false;

            self.builder.switch_to_block(next_b);
            self.builder.seal_block(next_b);
        }

        let zero = self.builder.ins().iconst(types::I64, 0);
        self.builder.ins().jump(merge_b, &[zero.into()]);
        self.builder.switch_to_block(merge_b);
        self.builder.seal_block(merge_b);
        let mt = self.temp_rc_scopes.remove(temp_scope_idx);
        self.flush_temps(&mt);
        Ok(self.builder.block_params(merge_b)[0])
    }

    fn lower_expr_fresh(&mut self, expr: &Expr) -> Result<Value, String> {
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
            ExprKind::RecordLit { name, fields, .. } => {
                let layout = self
                    .records
                    .get(name)
                    .cloned()
                    .ok_or_else(|| format!("unknown record `{}` in codegen", name))?;
                let obj = self.call_rt_2(self.rt.rc_alloc, layout.tag, layout.fields.len() as i64);
                if layout.rc_mask != 0 {
                    let mv = self.builder.ins().iconst(types::I64, layout.rc_mask);
                    self.call_rt_2_void(self.rt.rc_set_rc_mask, obj, mv);
                }
                for (fname, fexpr) in fields {
                    if let Some((idx, fl)) = layout
                        .fields
                        .iter()
                        .enumerate()
                        .find(|(_, f)| f.name == *fname)
                    {
                        let val = if fl.is_rc {
                            self.lower_expr_owned(fexpr)?
                        } else {
                            let mut temps = Vec::new();
                            let v = self.lower_expr_borrowed(fexpr, &mut temps)?;
                            self.flush_temps(&temps);
                            v
                        };
                        self.call_rt_3_void(self.rt.rc_set_field, obj, idx as i64, val);
                    }
                }
                Ok(obj)
            }
            ExprKind::List(elems) => {
                let obj = self.call_rt_2(self.rt.rc_alloc, TAG_LIST, elems.len() as i64);
                for (idx, e) in elems.iter().enumerate() {
                    let mut temps = Vec::new();
                    let val = self.lower_expr_borrowed(e, &mut temps)?;
                    self.flush_temps(&temps);
                    self.call_rt_3_void(self.rt.rc_set_field, obj, idx as i64, val);
                }
                Ok(obj)
            }
            ExprKind::Index { target, index } => {
                let mut temps = Vec::new();
                let t = self.lower_expr_borrowed(target, &mut temps)?;
                let i = self.lower_expr_borrowed(index, &mut temps)?;
                let res = self.call_rt_2_vals(self.rt.rc_get_field, t, i);
                self.flush_temps(&temps);
                Ok(res)
            }
            ExprKind::Unary { op, expr } => {
                let mut temps = Vec::new();
                let v = self.lower_expr_borrowed(expr, &mut temps)?;
                self.flush_temps(&temps);
                match op {
                    UnOp::Neg => Ok(self.builder.ins().ineg(v)),
                    UnOp::Not => {
                        let zero = self.builder.ins().iconst(types::I64, 0);
                        let eq = self.builder.ins().icmp(IntCC::Equal, v, zero);
                        Ok(self.builder.ins().uextend(types::I64, eq))
                    }
                }
            }
            ExprKind::Binary {
                op: BinOp::And,
                left,
                right,
            } => {
                let mut ltemps = Vec::new();
                let l = self.lower_expr_borrowed(left, &mut ltemps)?;
                self.flush_temps(&ltemps);
                let z = self.builder.ins().iconst(types::I64, 0);
                let lb = self.builder.ins().icmp(IntCC::NotEqual, l, z);
                let rhs_b = self.builder.create_block();
                let short_b = self.builder.create_block();
                let merge_b = self.builder.create_block();
                self.builder.append_block_param(merge_b, types::I64);
                self.builder.ins().brif(lb, rhs_b, &[], short_b, &[]);

                self.builder.switch_to_block(short_b);
                self.builder.seal_block(short_b);
                let fval = self.builder.ins().iconst(types::I64, 0);
                self.builder.ins().jump(merge_b, &[fval.into()]);

                self.builder.switch_to_block(rhs_b);
                self.builder.seal_block(rhs_b);
                let mut rtemps = Vec::new();
                let r = self.lower_expr_borrowed(right, &mut rtemps)?;
                self.flush_temps(&rtemps);
                let rz = self.builder.ins().iconst(types::I64, 0);
                let rb = self.builder.ins().icmp(IntCC::NotEqual, r, rz);
                let rval = self.builder.ins().uextend(types::I64, rb);
                self.builder.ins().jump(merge_b, &[rval.into()]);

                self.builder.switch_to_block(merge_b);
                self.builder.seal_block(merge_b);
                Ok(self.builder.block_params(merge_b)[0])
            }
            ExprKind::Binary {
                op: BinOp::Or,
                left,
                right,
            } => {
                let mut ltemps = Vec::new();
                let l = self.lower_expr_borrowed(left, &mut ltemps)?;
                self.flush_temps(&ltemps);
                let z = self.builder.ins().iconst(types::I64, 0);
                let lb = self.builder.ins().icmp(IntCC::NotEqual, l, z);
                let short_b = self.builder.create_block();
                let rhs_b = self.builder.create_block();
                let merge_b = self.builder.create_block();
                self.builder.append_block_param(merge_b, types::I64);
                self.builder.ins().brif(lb, short_b, &[], rhs_b, &[]);

                self.builder.switch_to_block(short_b);
                self.builder.seal_block(short_b);
                let tval = self.builder.ins().iconst(types::I64, 1);
                self.builder.ins().jump(merge_b, &[tval.into()]);

                self.builder.switch_to_block(rhs_b);
                self.builder.seal_block(rhs_b);
                let mut rtemps = Vec::new();
                let r = self.lower_expr_borrowed(right, &mut rtemps)?;
                self.flush_temps(&rtemps);
                let rz = self.builder.ins().iconst(types::I64, 0);
                let rb = self.builder.ins().icmp(IntCC::NotEqual, r, rz);
                let rval = self.builder.ins().uextend(types::I64, rb);
                self.builder.ins().jump(merge_b, &[rval.into()]);

                self.builder.switch_to_block(merge_b);
                self.builder.seal_block(merge_b);
                Ok(self.builder.block_params(merge_b)[0])
            }
            ExprKind::Binary { op, left, right } => {
                let text_cmp = self.is_text_expr(left) || self.is_text_expr(right);
                let mut temps = Vec::new();
                let l = self.lower_expr_borrowed(left, &mut temps)?;
                let r = self.lower_expr_borrowed(right, &mut temps)?;
                if *op == BinOp::Concat {
                    let res = self.call_rt_2_vals(self.rt.str_concat, l, r);
                    self.flush_temps(&temps);
                    return Ok(res);
                }
                if text_cmp && matches!(op, BinOp::Eq | BinOp::Ne) {
                    let eq = self.call_rt_2_vals(self.rt.text_eq, l, r);
                    self.flush_temps(&temps);
                    if *op == BinOp::Eq {
                        return Ok(eq);
                    }
                    let zero = self.builder.ins().iconst(types::I64, 0);
                    let ne = self.builder.ins().icmp(IntCC::Equal, eq, zero);
                    return Ok(self.builder.ins().uextend(types::I64, ne));
                }
                self.flush_temps(&temps);
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
                    other => return Err(format!("unsupported binary op {:?} in codegen", other)),
                };
                Ok(v)
            }
            ExprKind::Call { callee, args } => match &callee.kind {
                ExprKind::Field { target, field } if field == "parse" => {
                    if let ExprKind::Ident(rname) = &target.kind {
                        let pname = format!("{}__parse", rname);
                        let fid = *self.func_ids.get(&pname).ok_or_else(|| {
                            format!("no derived parser `{}.parse` found", rname)
                        })?;
                        let mut temps = Vec::new();
                        let mut arg_vals = Vec::new();
                        for a in args {
                            arg_vals.push(self.lower_expr_borrowed(&a.value, &mut temps)?);
                        }
                        let local_callee = self.module.declare_func_in_func(fid, self.builder.func);
                        let call = self.builder.ins().call(local_callee, &arg_vals);
                        let ret = self.builder.inst_results(call)[0];
                        self.flush_temps(&temps);
                        return Ok(ret);
                    }
                    Err("unsupported `.parse()` target".into())
                }
                ExprKind::Field { target, field } => {
                    if let ExprKind::Ident(ename) = &target.kind {
                        let key = format!("{}.{}", ename, field);
                        if let Some(&tag) = self.enum_tags.get(&key) {
                            let obj = self.call_rt_2(self.rt.rc_alloc, tag, args.len() as i64);
                            let mut rc_mask = 0u64;
                            for (idx, a) in args.iter().enumerate() {
                                let is_rc = self.expr_produces_rc(&a.value);
                                let av = if is_rc {
                                    if idx < 64 {
                                        rc_mask |= 1u64 << idx;
                                    }
                                    self.lower_expr_owned(&a.value)?
                                } else {
                                    let mut temps = Vec::new();
                                    let v = self.lower_expr_borrowed(&a.value, &mut temps)?;
                                    self.flush_temps(&temps);
                                    v
                                };
                                self.call_rt_3_void(self.rt.rc_set_field, obj, idx as i64, av);
                            }
                            if rc_mask != 0 {
                                let mv = self.builder.ins().iconst(types::I64, rc_mask as i64);
                                self.call_rt_2_void(self.rt.rc_set_rc_mask, obj, mv);
                            }
                            return Ok(obj);
                        }
                    }
                    Err("unsupported method call in codegen".into())
                }
                ExprKind::Ident(fname) => {
                    if fname == "len" && args.len() == 1 {
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let res = self.call_rt_1(self.rt.rc_get_len, v);
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    if fname == "push" && args.len() == 2 {
                        let mut temps = Vec::new();
                        let xs = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        let val = self.lower_expr_borrowed(&args[1].value, &mut temps)?;
                        let res = self.call_rt_2_vals(self.rt.list_push, xs, val);
                        self.flush_temps(&temps);
                        return Ok(res);
                    }
                    if fname == "yield_val" && args.len() == 1 {
                        let mut temps = Vec::new();
                        let v = self.lower_expr_borrowed(&args[0].value, &mut temps)?;
                        self.flush_temps(&temps);
                        return Ok(self.call_rt_1(self.rt.async_yield_create, v));
                    }
                    if fname == "block_on" && args.len() == 1 {
                        // `block_on` consumes and frees the async frame itself!
                        let fut = self.lower_expr_owned(&args[0].value)?;
                        return Ok(self.call_rt_1(self.rt.async_block_on, fut));
                    }
                    if let Some(&tag) = self.enum_tags.get(fname) {
                        if !self.func_ids.contains_key(fname) {
                            let obj = self.call_rt_2(self.rt.rc_alloc, tag, args.len() as i64);
                            let mut rc_mask = 0u64;
                            for (idx, a) in args.iter().enumerate() {
                                let is_rc = self.expr_produces_rc(&a.value);
                                let av = if is_rc {
                                    if idx < 64 {
                                        rc_mask |= 1u64 << idx;
                                    }
                                    self.lower_expr_owned(&a.value)?
                                } else {
                                    let mut temps = Vec::new();
                                    let v = self.lower_expr_borrowed(&a.value, &mut temps)?;
                                    self.flush_temps(&temps);
                                    v
                                };
                                self.call_rt_3_void(self.rt.rc_set_field, obj, idx as i64, av);
                            }
                            if rc_mask != 0 {
                                let mv = self.builder.ins().iconst(types::I64, rc_mask as i64);
                                self.call_rt_2_void(self.rt.rc_set_rc_mask, obj, mv);
                            }
                            return Ok(obj);
                        }
                    }

                    let fid = *self
                        .func_ids
                        .get(fname)
                        .ok_or_else(|| format!("unknown function `{}` in codegen", fname))?;
                    let meta = self.fn_metas.get(fname).cloned();
                    let mut temps = Vec::new();
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
                                let cur_val = self.load_ident_val(vname)?;
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
                            arg_vals.push(self.lower_expr_borrowed(&a.value, &mut temps)?);
                        }
                    }
                    let local_callee = self.module.declare_func_in_func(fid, self.builder.func);
                    let call = self.builder.ins().call(local_callee, &arg_vals);
                    let ret = self.builder.inst_results(call)[0];
                    self.flush_temps(&temps);

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
                _ => Err("unsupported indirect call in codegen".into()),
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

    #[test]
    fn stackless_async_state_machine_suspends_resumes_and_frees_zero_leaks() {
        let src = r#"
@no_heap_alloc
pub pure fn combine(a Int, b Int) Int { a * 2 + b }

pub async fn step(seed Int) Int {
    let first = await yield_val(seed + 10)
    let second = await yield_val(first + 5)
    second
}

pub async fn pipeline(base Int) Int {
    let x = await step(base)
    let y = await yield_val(base * 3)
    combine(x, y)
}

pub fn main() Int {
    block_on(pipeline(7))
}
"#;
        perceus::reset_stats();
        let prog = parse(src).unwrap();
        let jit = JitProgram::compile(&prog).unwrap();
        assert!(jit.compiled_fns.contains(&"pipeline__poll".to_string()));
        assert_eq!(jit.call_0("main").unwrap(), 65);
        let st = perceus::snapshot_stats();
        assert!(
            st.async_suspends >= 3,
            "expected >=3 stackless suspensions, got {:?}",
            st
        );
        assert!(
            st.async_polls >= 6,
            "expected >=6 cooperative polls, got {:?}",
            st
        );
        assert_eq!(
            st.live_objects, 0,
            "stackless async frames must leave 0 leaks, got {:?}",
            st
        );
    }

    #[test]
    fn core_byte_ops_concat_break_continue_text_records_zero_leaks() {
        let src = r#"
pub record Pair {
    key Text
    val Text
}

pub fn make_pair(k Text, v Text) Pair {
    Pair { key = k ++ "_k", val = v }
}

pub fn reverse_ascii(s Text) Text {
    var out = ""
    var i = len(s) - 1
    while i >= 0 {
        let b = byte_at(s, i)
        if b == 32 {
            i -= 1
            continue
        }
        out = out ++ byte_to_text(b)
        if len(out) == 5 {
            break
        }
        i -= 1
    }
    out
}

pub fn main() Int {
    let p = make_pair("foo", "bar")
    assert p.key == "foo_k"
    assert p.val == "bar"
    let r = reverse_ascii("h e l l o !")
    assert r == "!olle"
    let sl = str_slice("tenon-lang", 0, 5)
    assert sl == "tenon"
    let m = match sl {
        "rust" => 1
        "tenon" => 42
        _ => 0
    }
    assert m == 42
    m
}
"#;
        perceus::reset_stats();
        let prog = parse(src).unwrap();
        let jit = JitProgram::compile(&prog).unwrap();
        assert_eq!(jit.call_0("main").unwrap(), 42);
        let st = perceus::snapshot_stats();
        assert_eq!(
            st.live_objects, 0,
            "core text/record/loop ops must leave 0 leaks, got {:?}",
            st
        );
    }
}
