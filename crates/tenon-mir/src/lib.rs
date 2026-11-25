//! Unified Typed SSA/CFG Middle-End IR (`tenon-mir`) shared across Cranelift JIT,
//! LLVM 22 AOT, and LLVM Wasm32 backends.
//!
//! Responsibilities:
//! 1. Lowers verified Tenon AST + `VerifiedMir` facts into explicit SSA basic blocks.
//! 2. Preserves proof-driven optimization flags (`proved_no_overflow`, `proved_in_bounds`,
//!    `stack_promoted`, `@arena`, `@fip`, `@checkfree`).
//! 3. Runs sound backward liveness analysis (`run_last_use_move_pass`) to identify Last-Use Move
//!    sites where owned Perceus RC handles can be moved with zero `rc_inc` / `rc_dec` overhead.
//! 4. Pretty-prints inspectable SSA CFG IR (`--emit-mir <file.mir>`).

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use tenon_syntax::*;
use tenon_verify::VerifiedMir;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MirVal(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u32);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MirInst {
    ConstInt { dst: MirVal, val: i64 },
    ConstFloat { dst: MirVal, val: f64 },
    ConstBool { dst: MirVal, val: bool },
    ConstText { dst: MirVal, val: String },
    LoadLocal { dst: MirVal, name: String, is_last_use_move: bool, span_start: usize },
    StoreLocal { name: String, val: MirVal },
    BinOp { dst: MirVal, op: String, lhs: MirVal, rhs: MirVal, proved_no_overflow: bool },
    LoadIndex { dst: MirVal, target: MirVal, index: MirVal, proved_in_bounds: bool },
    StoreIndex { target: String, index: MirVal, val: MirVal, proved_in_bounds: bool },
    AllocRecord { dst: MirVal, record: String, fields: Vec<(String, MirVal)>, stack_promoted: bool, reuse_from: Option<String> },
    FieldLoad { dst: MirVal, target: MirVal, field: String },
    Call { dst: MirVal, callee: String, args: Vec<MirVal> },
    RcMove { var: String, span_start: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MirTerminator {
    Jump(BlockId),
    Branch { cond: MirVal, then_bb: BlockId, else_bb: BlockId },
    Return(Option<MirVal>),
    Unreachable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicBlock {
    pub id: BlockId,
    pub label: String,
    pub insts: Vec<MirInst>,
    pub term: MirTerminator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirFunction {
    pub name: String,
    pub params: Vec<(String, String)>,
    pub ret_ty: String,
    pub is_arena: bool,
    pub is_fip: bool,
    pub is_no_heap_alloc: bool,
    pub is_checkfree: bool,
    pub blocks: Vec<BasicBlock>,
    /// Span start offsets of identifier reads proved to be Last-Use Moves (zero `rc_inc`/`rc_dec`).
    pub last_use_move_spans: BTreeSet<usize>,
    pub stack_promoted_locals: BTreeSet<String>,
    /// FBIP drop-reuse pairs: `(block_idx, inst_idx) -> dropped_local_name` whose heap slot can be reused in-place.
    pub fbip_reuse_pairs: BTreeMap<(usize, usize), String>,
    pub folded_constants: usize,
    pub dce_removed: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MirProgram {
    pub functions: Vec<MirFunction>,
}

struct FnLowerer<'a> {
    verified: &'a VerifiedMir,
    next_val: u32,
    blocks: Vec<BasicBlock>,
    cur_bb: usize,
    stack_promoted: BTreeSet<String>,
    /// Variables conservatively excluded from Last-Use Move (e.g. used in `defer`, captured by
    /// closure, mutated by `mutate`, or present in a function with `?` early-error propagation).
    non_movable: BTreeSet<String>,
    has_try_or_closure_or_loop_break: bool,
    seen_spans: BTreeMap<usize, usize>,
}

impl<'a> FnLowerer<'a> {
    fn new(fn_name: &str, verified: &'a VerifiedMir) -> Self {
        let stack_promoted = verified
            .stack_promotable_vars
            .get(fn_name)
            .cloned()
            .unwrap_or_default();
        let entry = BasicBlock {
            id: BlockId(0),
            label: "entry".into(),
            insts: Vec::new(),
            term: MirTerminator::Unreachable,
        };
        Self {
            verified,
            next_val: 0,
            blocks: vec![entry],
            cur_bb: 0,
            stack_promoted,
            non_movable: BTreeSet::new(),
            has_try_or_closure_or_loop_break: false,
            seen_spans: BTreeMap::new(),
        }
    }

    fn fresh_val(&mut self) -> MirVal {
        let v = MirVal(self.next_val);
        self.next_val += 1;
        v
    }

    fn new_block(&mut self, label: impl Into<String>) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(BasicBlock {
            id,
            label: label.into(),
            insts: Vec::new(),
            term: MirTerminator::Unreachable,
        });
        id
    }

    fn emit(&mut self, inst: MirInst) {
        self.blocks[self.cur_bb].insts.push(inst);
    }

    fn set_term(&mut self, term: MirTerminator) {
        self.blocks[self.cur_bb].term = term;
    }

    fn record_load(&mut self, name: &str, span_start: usize) -> MirVal {
        let d = self.fresh_val();
        if span_start > 0 {
            *self.seen_spans.entry(span_start).or_insert(0) += 1;
        }
        self.emit(MirInst::LoadLocal {
            dst: d,
            name: name.to_string(),
            is_last_use_move: false,
            span_start,
        });
        d
    }

    fn lower_block(&mut self, blk: &Block) -> Option<MirVal> {
        for stmt in &blk.stmts {
            match stmt {
                Stmt::Let { name, value, .. } => {
                    let v = self.lower_expr(value, Some(name.as_str()));
                    self.emit(MirInst::StoreLocal { name: name.clone(), val: v });
                }
                Stmt::LetTuple { names, value, .. } => {
                    let v = self.lower_expr(value, None);
                    for n in names {
                        self.emit(MirInst::StoreLocal { name: n.clone(), val: v });
                    }
                }
                Stmt::Assign { target, op, value, .. } => {
                    // For compound assignments (`x += e`), `target` is also read before write.
                    if *op != AssignOp::Set {
                        let _ = self.lower_expr(target, None);
                    }
                    let v = self.lower_expr(value, None);
                    match &target.kind {
                        ExprKind::Ident(n) => {
                            self.emit(MirInst::StoreLocal { name: n.clone(), val: v });
                        }
                        ExprKind::Field { target: obj, .. } => {
                            let _ = self.lower_expr(obj, None);
                            if let ExprKind::Ident(n) = &obj.kind {
                                self.non_movable.insert(n.clone());
                            }
                        }
                        ExprKind::Index { target: arr, index } => {
                            let idx = self.lower_expr(index, None);
                            let aname = match &arr.kind {
                                ExprKind::Ident(s) => s.clone(),
                                _ => {
                                    let _ = self.lower_expr(arr, None);
                                    "<expr>".into()
                                }
                            };
                            let safe = self.verified.proved_safe_indices.contains(&target.span.start);
                            self.emit(MirInst::StoreIndex {
                                target: aname,
                                index: idx,
                                val: v,
                                proved_in_bounds: safe,
                            });
                        }
                        _ => {
                            let _ = self.lower_expr(target, None);
                        }
                    }
                }
                Stmt::Return { value, .. } => {
                    let rv = value.as_ref().map(|e| self.lower_expr(e, None));
                    self.set_term(MirTerminator::Return(rv));
                    let dead = self.new_block("after_ret");
                    self.cur_bb = dead.0 as usize;
                }
                Stmt::Expr { expr, .. } | Stmt::Assert { cond: expr, .. } => {
                    let _ = self.lower_expr(expr, None);
                }
                Stmt::While { cond, body, .. } => {
                    let hdr = self.new_block("while_hdr");
                    let bdy = self.new_block("while_body");
                    let exit = self.new_block("while_exit");
                    self.set_term(MirTerminator::Jump(hdr));
                    self.cur_bb = hdr.0 as usize;
                    let cv = self.lower_expr(cond, None);
                    self.set_term(MirTerminator::Branch { cond: cv, then_bb: bdy, else_bb: exit });
                    self.cur_bb = bdy.0 as usize;
                    let _ = self.lower_block(body);
                    if matches!(self.blocks[self.cur_bb].term, MirTerminator::Unreachable) {
                        self.set_term(MirTerminator::Jump(hdr));
                    }
                    self.cur_bb = exit.0 as usize;
                }
                Stmt::For { binding, iter, body, .. } => {
                    let _iv = self.lower_expr(iter, None);
                    let hdr = self.new_block("for_hdr");
                    let bdy = self.new_block("for_body");
                    let exit = self.new_block("for_exit");
                    self.set_term(MirTerminator::Jump(hdr));
                    self.cur_bb = hdr.0 as usize;
                    let cond = self.fresh_val();
                    self.emit(MirInst::ConstBool { dst: cond, val: true });
                    self.set_term(MirTerminator::Branch { cond, then_bb: bdy, else_bb: exit });
                    self.cur_bb = bdy.0 as usize;
                    let item_v = self.fresh_val();
                    self.emit(MirInst::ConstInt { dst: item_v, val: 0 });
                    match binding {
                        ForBinding::Ident(var) => {
                            self.emit(MirInst::StoreLocal { name: var.clone(), val: item_v });
                        }
                        ForBinding::Tuple(vars) => {
                            for v in vars {
                                self.emit(MirInst::StoreLocal { name: v.clone(), val: item_v });
                            }
                        }
                    }
                    let _ = self.lower_block(body);
                    if matches!(self.blocks[self.cur_bb].term, MirTerminator::Unreachable) {
                        self.set_term(MirTerminator::Jump(hdr));
                    }
                    self.cur_bb = exit.0 as usize;
                }
                Stmt::Break { .. } | Stmt::Continue { .. } => {
                    self.has_try_or_closure_or_loop_break = true;
                }
                Stmt::Mutate { target, body, .. } => {
                    self.non_movable.insert(target.clone());
                    let _ = self.record_load(target, 0);
                    let _ = self.lower_block(body);
                }
                Stmt::Defer { body, .. } => {
                    collect_idents_in_block(body, &mut self.non_movable);
                    let _ = self.lower_block(body);
                }
                Stmt::Ghost { .. } => {}
            }
        }
        blk.tail.as_ref().map(|t| self.lower_expr(t, None))
    }

    fn lower_expr(&mut self, expr: &Expr, bind_name: Option<&str>) -> MirVal {
        match &expr.kind {
            ExprKind::Int(n) => {
                let d = self.fresh_val();
                self.emit(MirInst::ConstInt { dst: d, val: *n });
                d
            }
            ExprKind::Float(x) => {
                let d = self.fresh_val();
                self.emit(MirInst::ConstFloat { dst: d, val: *x });
                d
            }
            ExprKind::Bool(b) => {
                let d = self.fresh_val();
                self.emit(MirInst::ConstBool { dst: d, val: *b });
                d
            }
            ExprKind::Str(s) => {
                let d = self.fresh_val();
                self.emit(MirInst::ConstText { dst: d, val: s.clone() });
                d
            }
            ExprKind::Ident(name) => self.record_load(name, expr.span.start),
            ExprKind::Binary { op, left, right } => {
                let l = self.lower_expr(left, None);
                let r = self.lower_expr(right, None);
                let d = self.fresh_val();
                let nsw = self.verified.proved_no_overflow.contains(&expr.span.start);
                self.emit(MirInst::BinOp {
                    dst: d,
                    op: format!("{:?}", op),
                    lhs: l,
                    rhs: r,
                    proved_no_overflow: nsw,
                });
                d
            }
            ExprKind::Index { target, index } => {
                let t = self.lower_expr(target, None);
                let i = self.lower_expr(index, None);
                let d = self.fresh_val();
                let safe = self.verified.proved_safe_indices.contains(&expr.span.start);
                self.emit(MirInst::LoadIndex {
                    dst: d,
                    target: t,
                    index: i,
                    proved_in_bounds: safe,
                });
                d
            }
            ExprKind::Field { target, field } => {
                let t = self.lower_expr(target, None);
                let d = self.fresh_val();
                self.emit(MirInst::FieldLoad { dst: d, target: t, field: field.clone() });
                d
            }
            ExprKind::TupleIndex { target, index } => {
                let t = self.lower_expr(target, None);
                let d = self.fresh_val();
                self.emit(MirInst::FieldLoad { dst: d, target: t, field: index.to_string() });
                d
            }
            ExprKind::RecordLit { name, fields, .. } => {
                let mut fvs = Vec::with_capacity(fields.len());
                for (fnm, fex) in fields {
                    fvs.push((fnm.clone(), self.lower_expr(fex, None)));
                }
                let d = self.fresh_val();
                let stk = bind_name.is_some_and(|bn| self.stack_promoted.contains(bn));
                self.emit(MirInst::AllocRecord {
                    dst: d,
                    record: name.clone(),
                    fields: fvs,
                    stack_promoted: stk,
                    reuse_from: None,
                });
                d
            }
            ExprKind::List(elems) | ExprKind::Tuple(elems) => {
                let avs: Vec<MirVal> = elems.iter().map(|e| self.lower_expr(e, None)).collect();
                let d = self.fresh_val();
                self.emit(MirInst::Call { dst: d, callee: "__lit".into(), args: avs });
                d
            }
            ExprKind::ArrayRepeat { value, .. } => {
                let v = self.lower_expr(value, None);
                let d = self.fresh_val();
                self.emit(MirInst::Call { dst: d, callee: "__arr_repeat".into(), args: vec![v] });
                d
            }
            ExprKind::Call { callee, args } => {
                let mut avs = Vec::with_capacity(args.len() + 1);
                let cname = match &callee.kind {
                    ExprKind::Ident(s) => s.clone(),
                    ExprKind::Field { target, field } => {
                        avs.push(self.lower_expr(target, None));
                        format!(".{}", field)
                    }
                    _ => {
                        avs.push(self.lower_expr(callee, None));
                        "<indirect>".into()
                    }
                };
                for a in args {
                    avs.push(self.lower_expr(&a.value, None));
                }
                let d = self.fresh_val();
                self.emit(MirInst::Call { dst: d, callee: cname, args: avs });
                d
            }
            ExprKind::If { cond, then_block, else_branch } => {
                let cv = self.lower_expr(cond, None);
                let then_bb = self.new_block("if_then");
                let else_bb = self.new_block("if_else");
                let merge_bb = self.new_block("if_merge");
                self.set_term(MirTerminator::Branch { cond: cv, then_bb, else_bb });
                self.cur_bb = then_bb.0 as usize;
                let tv = self.lower_block(then_block);
                if matches!(self.blocks[self.cur_bb].term, MirTerminator::Unreachable) {
                    self.set_term(MirTerminator::Jump(merge_bb));
                }
                self.cur_bb = else_bb.0 as usize;
                let _ev = else_branch.as_ref().map(|eb| self.lower_expr(eb, None));
                if matches!(self.blocks[self.cur_bb].term, MirTerminator::Unreachable) {
                    self.set_term(MirTerminator::Jump(merge_bb));
                }
                self.cur_bb = merge_bb.0 as usize;
                tv.unwrap_or_else(|| {
                    let d = self.fresh_val();
                    self.emit(MirInst::ConstInt { dst: d, val: 0 });
                    d
                })
            }
            ExprKind::Match { scrutinee, arms } => {
                let sv = self.lower_expr(scrutinee, None);
                let merge_bb = self.new_block("match_merge");
                let start_bb = self.cur_bb;
                let mut first_arm_bb = merge_bb;
                let mut prev_else_bb: Option<usize> = None;
                for (idx, arm) in arms.iter().enumerate() {
                    let arm_bb = self.new_block(format!("match_arm_{}", idx));
                    if idx == 0 {
                        first_arm_bb = arm_bb;
                    }
                    if let Some(pe) = prev_else_bb {
                        let next_chk = if idx + 1 < arms.len() {
                            self.new_block(format!("match_chk_{}", idx))
                        } else {
                            arm_bb
                        };
                        self.blocks[pe].term = MirTerminator::Branch {
                            cond: sv,
                            then_bb: arm_bb,
                            else_bb: next_chk,
                        };
                        prev_else_bb = if idx + 1 < arms.len() {
                            Some(next_chk.0 as usize)
                        } else {
                            None
                        };
                    } else if arms.len() == 1 {
                        self.blocks[start_bb].term = MirTerminator::Jump(arm_bb);
                    } else {
                        let chk = self.new_block("match_chk_0");
                        self.blocks[start_bb].term = MirTerminator::Branch {
                            cond: sv,
                            then_bb: arm_bb,
                            else_bb: chk,
                        };
                        prev_else_bb = Some(chk.0 as usize);
                    }
                    self.cur_bb = arm_bb.0 as usize;
                    if let Some(g) = &arm.guard {
                        let _ = self.lower_expr(g, None);
                    }
                    let _ = self.lower_expr(&arm.body, None);
                    if matches!(self.blocks[self.cur_bb].term, MirTerminator::Unreachable) {
                        self.set_term(MirTerminator::Jump(merge_bb));
                    }
                }
                if arms.is_empty() {
                    self.blocks[start_bb].term = MirTerminator::Jump(first_arm_bb);
                } else if let Some(pe) = prev_else_bb {
                    if matches!(self.blocks[pe].term, MirTerminator::Unreachable) {
                        self.blocks[pe].term = MirTerminator::Jump(merge_bb);
                    }
                }
                self.cur_bb = merge_bb.0 as usize;
                let d = self.fresh_val();
                self.emit(MirInst::ConstInt { dst: d, val: 0 });
                d
            }
            ExprKind::Lambda { body, .. } => {
                // Any variable captured by a lambda may execute multiple times.
                self.has_try_or_closure_or_loop_break = true;
                collect_idents_in_expr(body, &mut self.non_movable);
                let d = self.fresh_val();
                self.emit(MirInst::ConstInt { dst: d, val: 0 });
                d
            }
            ExprKind::Range { start, end } => {
                let _ = self.lower_expr(start, None);
                self.lower_expr(end, None)
            }
            ExprKind::Try(inner) => {
                self.has_try_or_closure_or_loop_break = true;
                self.lower_expr(inner, None)
            }
            ExprKind::Block(b) | ExprKind::Unsafe(b) | ExprKind::Comptime(b) => {
                self.lower_block(b).unwrap_or_else(|| {
                    let d = self.fresh_val();
                    self.emit(MirInst::ConstInt { dst: d, val: 0 });
                    d
                })
            }
            ExprKind::Unary { expr: inner, .. }
            | ExprKind::Await(inner)
            | ExprKind::Cast { expr: inner, .. }
            | ExprKind::TryCast { expr: inner, .. }
            | ExprKind::GenericInst { expr: inner, .. }
            | ExprKind::Old(inner) => self.lower_expr(inner, None),
            ExprKind::Forall { body, .. } | ExprKind::Quantifier { body, .. } => {
                self.lower_expr(body, None)
            }
            ExprKind::ComptimeIf { cond, then_block, else_branch } => {
                let _ = self.lower_expr(cond, None);
                let _ = self.lower_block(then_block);
                if let Some(eb) = else_branch {
                    let _ = self.lower_expr(eb, None);
                }
                let d = self.fresh_val();
                self.emit(MirInst::ConstInt { dst: d, val: 0 });
                d
            }
            ExprKind::ComptimeFor { iter, body, .. } => {
                let _ = self.lower_expr(iter, None);
                let _ = self.lower_block(body);
                let d = self.fresh_val();
                self.emit(MirInst::ConstInt { dst: d, val: 0 });
                d
            }
            ExprKind::MacroCall { args, .. } => {
                for a in args {
                    let _ = self.lower_expr(a, None);
                }
                let d = self.fresh_val();
                self.emit(MirInst::ConstInt { dst: d, val: 0 });
                d
            }
            ExprKind::TypeImplements { .. } | ExprKind::UnitValue { .. } => {
                let d = self.fresh_val();
                self.emit(MirInst::ConstInt { dst: d, val: 0 });
                d
            }
        }
    }
}

fn collect_idents_in_block(blk: &Block, out: &mut BTreeSet<String>) {
    for st in &blk.stmts {
        match st {
            Stmt::Let { value, .. } | Stmt::LetTuple { value, .. } => {
                collect_idents_in_expr(value, out)
            }
            Stmt::Assign { target, value, .. } => {
                collect_idents_in_expr(target, out);
                collect_idents_in_expr(value, out);
            }
            Stmt::Return { value: Some(v), .. } => collect_idents_in_expr(v, out),
            Stmt::Expr { expr, .. } | Stmt::Assert { cond: expr, .. } => {
                collect_idents_in_expr(expr, out)
            }
            Stmt::While { cond, body, .. } => {
                collect_idents_in_expr(cond, out);
                collect_idents_in_block(body, out);
            }
            Stmt::For { iter, body, .. } => {
                collect_idents_in_expr(iter, out);
                collect_idents_in_block(body, out);
            }
            Stmt::Mutate { target, body, .. } => {
                out.insert(target.clone());
                collect_idents_in_block(body, out);
            }
            Stmt::Defer { body, .. } | Stmt::Ghost { body, .. } => {
                collect_idents_in_block(body, out);
            }
            Stmt::Return { value: None, .. } | Stmt::Break { .. } | Stmt::Continue { .. } => {}
        }
    }
    if let Some(t) = &blk.tail {
        collect_idents_in_expr(t, out);
    }
}

fn collect_idents_in_expr(expr: &Expr, out: &mut BTreeSet<String>) {
    match &expr.kind {
        ExprKind::Ident(s) => {
            out.insert(s.clone());
        }
        ExprKind::Binary { left, right, .. } | ExprKind::Range { start: left, end: right } => {
            collect_idents_in_expr(left, out);
            collect_idents_in_expr(right, out);
        }
        ExprKind::Unary { expr: e, .. }
        | ExprKind::Try(e)
        | ExprKind::Await(e)
        | ExprKind::Cast { expr: e, .. }
        | ExprKind::TryCast { expr: e, .. }
        | ExprKind::GenericInst { expr: e, .. }
        | ExprKind::Old(e)
        | ExprKind::ArrayRepeat { value: e, .. }
        | ExprKind::Field { target: e, .. }
        | ExprKind::TupleIndex { target: e, .. } => collect_idents_in_expr(e, out),
        ExprKind::Index { target, index } => {
            collect_idents_in_expr(target, out);
            collect_idents_in_expr(index, out);
        }
        ExprKind::Call { callee, args } => {
            collect_idents_in_expr(callee, out);
            for a in args {
                collect_idents_in_expr(&a.value, out);
            }
        }
        ExprKind::RecordLit { fields, .. } => {
            for (_, v) in fields {
                collect_idents_in_expr(v, out);
            }
        }
        ExprKind::List(es) | ExprKind::Tuple(es) | ExprKind::MacroCall { args: es, .. } => {
            for e in es {
                collect_idents_in_expr(e, out);
            }
        }
        ExprKind::If { cond, then_block, else_branch }
        | ExprKind::ComptimeIf { cond, then_block, else_branch } => {
            collect_idents_in_expr(cond, out);
            collect_idents_in_block(then_block, out);
            if let Some(eb) = else_branch {
                collect_idents_in_expr(eb, out);
            }
        }
        ExprKind::ComptimeFor { iter, body, .. } => {
            collect_idents_in_expr(iter, out);
            collect_idents_in_block(body, out);
        }
        ExprKind::Match { scrutinee, arms } => {
            collect_idents_in_expr(scrutinee, out);
            for a in arms {
                if let Some(g) = &a.guard {
                    collect_idents_in_expr(g, out);
                }
                collect_idents_in_expr(&a.body, out);
            }
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) | ExprKind::Comptime(b) => {
            collect_idents_in_block(b, out)
        }
        ExprKind::Lambda { body, .. }
        | ExprKind::Forall { body, .. }
        | ExprKind::Quantifier { body, .. } => collect_idents_in_expr(body, out),
        ExprKind::Int(_)
        | ExprKind::Float(_)
        | ExprKind::Bool(_)
        | ExprKind::Str(_)
        | ExprKind::TypeImplements { .. }
        | ExprKind::UnitValue { .. } => {}
    }
}

/// Lowers a verified `Program` into SSA/CFG `MirProgram` and runs Last-Use Move analysis.
pub fn lower_program_to_ssa_mir(program: &Program, verified: &VerifiedMir) -> MirProgram {
    let mut functions = Vec::new();
    for item in &program.items {
        if let Item::Fn(f) = item {
            if f.is_ghost || !f.type_params.is_empty() {
                continue;
            }
            let mut lowerer = FnLowerer::new(&f.name, verified);
            for p in &f.params {
                if p.is_inout {
                    lowerer.non_movable.insert(p.name.clone());
                }
            }
            let tail = lowerer.lower_block(&f.body);
            if matches!(lowerer.blocks[lowerer.cur_bb].term, MirTerminator::Unreachable) {
                lowerer.set_term(MirTerminator::Return(tail));
            }
            let mut mf = MirFunction {
                name: f.name.clone(),
                params: f.params.iter().map(|p| (p.name.clone(), p.ty.display_name())).collect(),
                ret_ty: f.ret_ty.as_ref().map(|t| t.display_name()).unwrap_or_else(|| "Unit".into()),
                is_arena: verified.arena_verified_fns.contains(&f.name),
                is_fip: verified.fip_verified_fns.contains(&f.name),
                is_no_heap_alloc: f.attrs.iter().any(|a| a.name == "no_heap_alloc"),
                is_checkfree: f.attrs.iter().any(|a| a.name == "checkfree"),
                blocks: lowerer.blocks,
                last_use_move_spans: BTreeSet::new(),
                stack_promoted_locals: lowerer.stack_promoted,
                fbip_reuse_pairs: BTreeMap::new(),
                folded_constants: 0,
                dce_removed: 0,
            };
            if !lowerer.has_try_or_closure_or_loop_break {
                let dup_spans: BTreeSet<usize> = lowerer
                    .seen_spans
                    .into_iter()
                    .filter(|&(_, cnt)| cnt > 1)
                    .map(|(sp, _)| sp)
                    .collect();
                run_last_use_move_pass(&mut mf, &lowerer.non_movable, &dup_spans);
            }
            run_const_fold_pass(&mut mf);
            run_dce_pass(&mut mf);
            run_fbip_reuse_pass(&mut mf);
            functions.push(mf);
        }
    }
    MirProgram { functions }
}

/// Backward liveness & Last-Use Move pass over SSA basic blocks.
/// Marks any local variable read in a CFG basic block that has NO subsequent uses
/// in that block or any successor block as `is_last_use_move = true`!
pub fn run_last_use_move_pass(
    mf: &mut MirFunction,
    non_movable: &BTreeSet<String>,
    dup_spans: &BTreeSet<usize>,
) {
    let n = mf.blocks.len();
    if n == 0 {
        return;
    }
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut gen_sets: Vec<BTreeSet<String>> = vec![BTreeSet::new(); n];
    let mut kill_sets: Vec<BTreeSet<String>> = vec![BTreeSet::new(); n];
    for (bi, bb) in mf.blocks.iter().enumerate() {
        match &bb.term {
            MirTerminator::Jump(t) => succs[bi].push(t.0 as usize),
            MirTerminator::Branch { then_bb, else_bb, .. } => {
                succs[bi].push(then_bb.0 as usize);
                succs[bi].push(else_bb.0 as usize);
            }
            _ => {}
        }
        let mut killed = BTreeSet::new();
        for inst in &bb.insts {
            match inst {
                MirInst::LoadLocal { name, .. } => {
                    if !killed.contains(name) {
                        gen_sets[bi].insert(name.clone());
                    }
                }
                MirInst::StoreIndex { target, .. } => {
                    if !killed.contains(target) {
                        gen_sets[bi].insert(target.clone());
                    }
                }
                MirInst::StoreLocal { name, .. } => {
                    killed.insert(name.clone());
                }
                _ => {}
            }
        }
        kill_sets[bi] = killed;
    }

    let mut live_in: Vec<BTreeSet<String>> = vec![BTreeSet::new(); n];
    let mut live_out: Vec<BTreeSet<String>> = vec![BTreeSet::new(); n];
    let mut changed = true;
    while changed {
        changed = false;
        for bi in (0..n).rev() {
            let mut out = BTreeSet::new();
            for &s in &succs[bi] {
                if s < n {
                    out.extend(live_in[s].iter().cloned());
                }
            }
            let mut inn = gen_sets[bi].clone();
            for v in &out {
                if !kill_sets[bi].contains(v) {
                    inn.insert(v.clone());
                }
            }
            if out != live_out[bi] || inn != live_in[bi] {
                live_out[bi] = out;
                live_in[bi] = inn;
                changed = true;
            }
        }
    }

    let mut last_use_spans = BTreeSet::new();
    for (bi, bb) in mf.blocks.iter_mut().enumerate() {
        let mut live = live_out[bi].clone();
        for inst in bb.insts.iter_mut().rev() {
            match inst {
                MirInst::StoreLocal { name, .. } => {
                    live.remove(name);
                }
                MirInst::StoreIndex { target, .. } => {
                    live.insert(target.clone());
                }
                MirInst::LoadLocal { name, is_last_use_move, span_start, .. } => {
                    if !live.contains(name)
                        && *span_start > 0
                        && !non_movable.contains(name)
                        && !dup_spans.contains(span_start)
                    {
                        *is_last_use_move = true;
                        last_use_spans.insert(*span_start);
                    }
                    live.insert(name.clone());
                }
                _ => {}
            }
        }
    }
    mf.last_use_move_spans = last_use_spans;
}

/// SSA Constant Folding pass: folds pure integer arithmetic/comparison `BinOp`s on constant operands.
pub fn run_const_fold_pass(mf: &mut MirFunction) {
    let mut const_ints: BTreeMap<MirVal, i64> = BTreeMap::new();
    let mut folded = 0usize;
    for bb in &mut mf.blocks {
        for inst in &mut bb.insts {
            match inst {
                MirInst::ConstInt { dst, val } => {
                    const_ints.insert(*dst, *val);
                }
                MirInst::BinOp { dst, op, lhs, rhs, .. } => {
                    if let (Some(&a), Some(&b)) = (const_ints.get(lhs), const_ints.get(rhs)) {
                        let dst_v = *dst;
                        let folded_inst = match op.as_str() {
                            "Add" | "+" => a.checked_add(b).map(|v| MirInst::ConstInt { dst: dst_v, val: v }),
                            "Sub" | "-" => a.checked_sub(b).map(|v| MirInst::ConstInt { dst: dst_v, val: v }),
                            "Mul" | "*" => a.checked_mul(b).map(|v| MirInst::ConstInt { dst: dst_v, val: v }),
                            "Div" | "/" if b != 0 => a.checked_div(b).map(|v| MirInst::ConstInt { dst: dst_v, val: v }),
                            "Mod" | "%" if b != 0 => a.checked_rem(b).map(|v| MirInst::ConstInt { dst: dst_v, val: v }),
                            "Eq" | "==" => Some(MirInst::ConstBool { dst: dst_v, val: a == b }),
                            "Ne" | "!=" => Some(MirInst::ConstBool { dst: dst_v, val: a != b }),
                            "Lt" | "<" => Some(MirInst::ConstBool { dst: dst_v, val: a < b }),
                            "Le" | "<=" => Some(MirInst::ConstBool { dst: dst_v, val: a <= b }),
                            "Gt" | ">" => Some(MirInst::ConstBool { dst: dst_v, val: a > b }),
                            "Ge" | ">=" => Some(MirInst::ConstBool { dst: dst_v, val: a >= b }),
                            _ => None,
                        };
                        if let Some(new_inst) = folded_inst {
                            if let MirInst::ConstInt { val, .. } = new_inst {
                                const_ints.insert(dst_v, val);
                            }
                            *inst = new_inst;
                            folded += 1;
                        }
                    }
                }
                _ => {}
            }
        }
    }
    mf.folded_constants = folded;
}

/// SSA Dead Code Elimination pass: removes unused pure `Const*` / `BinOp` / `FieldLoad` instructions.
pub fn run_dce_pass(mf: &mut MirFunction) {
    let mut removed = 0usize;
    loop {
        let mut used: BTreeSet<MirVal> = BTreeSet::new();
        for bb in &mf.blocks {
            for inst in &bb.insts {
                match inst {
                    MirInst::StoreLocal { val, .. } => { used.insert(*val); }
                    MirInst::BinOp { lhs, rhs, .. } => { used.insert(*lhs); used.insert(*rhs); }
                    MirInst::LoadIndex { target, index, .. } => { used.insert(*target); used.insert(*index); }
                    MirInst::StoreIndex { index, val, .. } => { used.insert(*index); used.insert(*val); }
                    MirInst::AllocRecord { fields, .. } => { for (_, v) in fields { used.insert(*v); } }
                    MirInst::FieldLoad { target, .. } => { used.insert(*target); }
                    MirInst::Call { args, .. } => { for a in args { used.insert(*a); } }
                    _ => {}
                }
            }
            match &bb.term {
                MirTerminator::Branch { cond, .. } => { used.insert(*cond); }
                MirTerminator::Return(Some(v)) => { used.insert(*v); }
                _ => {}
            }
        }
        let mut pass_removed = 0usize;
        for bb in &mut mf.blocks {
            let before = bb.insts.len();
            bb.insts.retain(|inst| match inst {
                MirInst::ConstInt { dst, .. }
                | MirInst::ConstFloat { dst, .. }
                | MirInst::ConstBool { dst, .. }
                | MirInst::ConstText { dst, .. }
                | MirInst::BinOp { dst, .. }
                | MirInst::FieldLoad { dst, .. } => used.contains(dst),
                _ => true,
            });
            pass_removed += before - bb.insts.len();
        }
        removed += pass_removed;
        if pass_removed == 0 {
            break;
        }
    }
    mf.dce_removed = removed;
}

/// FBIP (Functional But In-Place) Drop-Reuse Pairing pass:
/// pairs heap `AllocRecord` instructions with a preceding Last-Use Move of a local variable
/// of the **exact same record type** in the same basic block so Perceus can reuse the allocation
/// slot in-place with zero `malloc`.
pub fn run_fbip_reuse_pass(mf: &mut MirFunction) {
    let mut local_record_ty: BTreeMap<String, String> = BTreeMap::new();
    for (pname, pty) in &mf.params {
        local_record_ty.insert(pname.clone(), pty.clone());
    }
    let mut val_record_ty: BTreeMap<MirVal, String> = BTreeMap::new();
    for bb in &mf.blocks {
        for inst in &bb.insts {
            match inst {
                MirInst::AllocRecord { dst, record, .. } => {
                    val_record_ty.insert(*dst, record.clone());
                }
                MirInst::StoreLocal { name, val } => {
                    if let Some(rt) = val_record_ty.get(val) {
                        local_record_ty.insert(name.clone(), rt.clone());
                    }
                }
                _ => {}
            }
        }
    }
    let mut pairs = BTreeMap::new();
    for (bi, bb) in mf.blocks.iter_mut().enumerate() {
        let mut free_by_record: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (ii, inst) in bb.insts.iter_mut().enumerate() {
            match inst {
                MirInst::LoadLocal { name, is_last_use_move: true, .. } => {
                    if let Some(rty) = local_record_ty.get(name) {
                        let q = free_by_record.entry(rty.clone()).or_default();
                        if !q.contains(name) {
                            q.push(name.clone());
                        }
                    }
                }
                MirInst::AllocRecord { record, stack_promoted: false, reuse_from, .. } => {
                    if let Some(q) = free_by_record.get_mut(record) {
                        if let Some(donor) = q.pop() {
                            *reuse_from = Some(donor.clone());
                            pairs.insert((bi, ii), donor);
                        }
                    }
                }
                _ => {}
            }
        }
    }
    mf.fbip_reuse_pairs = pairs;
}

/// Formats a `MirProgram` into human-readable SSA CFG text (`tenon build --emit-mir`).
pub fn format_mir(mir: &MirProgram) -> String {
    let mut out = String::from("; Tenon Unified Typed SSA/CFG MIR (tenon-mir)\n\n");
    for f in &mir.functions {
        let mut tags = Vec::new();
        if f.is_arena { tags.push("@arena"); }
        if f.is_fip { tags.push("@fip"); }
        if f.is_no_heap_alloc { tags.push("@no_heap_alloc"); }
        if f.is_checkfree { tags.push("@checkfree"); }
        let tag_str = if tags.is_empty() { String::new() } else { format!(" [{}]", tags.join(", ")) };
        let ps: Vec<String> = f.params.iter().map(|(n, t)| format!("{}: {}", n, t)).collect();
        out.push_str(&format!(
            "fn {}({}) -> {}{} ; last_use_moves={} stack_promoted={}\n",
            f.name,
            ps.join(", "),
            f.ret_ty,
            tag_str,
            f.last_use_move_spans.len(),
            f.stack_promoted_locals.len()
        ));
        for bb in &f.blocks {
            out.push_str(&format!("  bb{} ({}):\n", bb.id.0, bb.label));
            for inst in &bb.insts {
                match inst {
                    MirInst::ConstInt { dst, val } => out.push_str(&format!("    %{} = const.i64 {}\n", dst.0, val)),
                    MirInst::ConstFloat { dst, val } => out.push_str(&format!("    %{} = const.f64 {}\n", dst.0, val)),
                    MirInst::ConstBool { dst, val } => out.push_str(&format!("    %{} = const.bool {}\n", dst.0, val)),
                    MirInst::ConstText { dst, val } => out.push_str(&format!("    %{} = const.text {:?}\n", dst.0, val)),
                    MirInst::LoadLocal { dst, name, is_last_use_move, .. } => {
                        let op = if *is_last_use_move { "move.local" } else { "load.local" };
                        out.push_str(&format!("    %{} = {} ${}\n", dst.0, op, name));
                    }
                    MirInst::StoreLocal { name, val } => out.push_str(&format!("    store.local ${} = %{}\n", name, val.0)),
                    MirInst::BinOp { dst, op, lhs, rhs, proved_no_overflow } => {
                        let fl = if *proved_no_overflow { " nsw" } else { "" };
                        out.push_str(&format!("    %{} = binop.{}{} %{}, %{}\n", dst.0, op, fl, lhs.0, rhs.0));
                    }
                    MirInst::LoadIndex { dst, target, index, proved_in_bounds } => {
                        let fl = if *proved_in_bounds { " unchecked" } else { " checked" };
                        out.push_str(&format!("    %{} = idx.load{} %{}[%{}]\n", dst.0, fl, target.0, index.0));
                    }
                    MirInst::StoreIndex { target, index, val, proved_in_bounds } => {
                        let fl = if *proved_in_bounds { " unchecked" } else { " checked" };
                        out.push_str(&format!("    idx.store{} ${}[%{}] = %{}\n", fl, target, index.0, val.0));
                    }
                    MirInst::AllocRecord { dst, record, fields, stack_promoted, reuse_from } => {
                        let loc = if *stack_promoted {
                            "stack.alloca".to_string()
                        } else if let Some(donor) = reuse_from {
                            format!("reuse.rc(${})", donor)
                        } else {
                            "heap.rc".to_string()
                        };
                        let fs: Vec<String> = fields.iter().map(|(k, v)| format!("{}=%{}", k, v.0)).collect();
                        out.push_str(&format!("    %{} = alloc.{} {} {{ {} }}\n", dst.0, loc, record, fs.join(", ")));
                    }
                    MirInst::FieldLoad { dst, target, field } => out.push_str(&format!("    %{} = field %{}.{}\n", dst.0, target.0, field)),
                    MirInst::Call { dst, callee, args } => {
                        let av: Vec<String> = args.iter().map(|a| format!("%{}", a.0)).collect();
                        out.push_str(&format!("    %{} = call {}({})\n", dst.0, callee, av.join(", ")));
                    }
                    MirInst::RcMove { var, .. } => out.push_str(&format!("    rc.move ${}\n", var)),
                }
            }
            match &bb.term {
                MirTerminator::Jump(t) => out.push_str(&format!("    br bb{}\n", t.0)),
                MirTerminator::Branch { cond, then_bb, else_bb } => {
                    out.push_str(&format!("    br_if %{}, bb{}, bb{}\n", cond.0, then_bb.0, else_bb.0));
                }
                MirTerminator::Return(Some(v)) => out.push_str(&format!("    ret %{}\n", v.0)),
                MirTerminator::Return(None) => out.push_str("    ret void\n"),
                MirTerminator::Unreachable => out.push_str("    unreachable\n"),
            }
        }
        out.push('\n');
    }
    out
}

/// Extracts per-function Last-Use Move span map for downstream backends.
pub fn collect_last_use_move_spans(mir: &MirProgram) -> BTreeMap<String, BTreeSet<usize>> {
    mir.functions
        .iter()
        .map(|f| (f.name.clone(), f.last_use_move_spans.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_to_ssa_mir_with_proof_facts_and_last_use_move_liveness() {
        let src = r#"
pub record Pt {
    x Int
    y Int
}

@arena
pub fn score_loop(n Int where self >= 1 && self <= 16) Int {
    var xs = list_filled(n, 4)
    let p = Pt { x = 3, y = 7 }
    var acc = p.x + p.y
    var i = 0
    while i < n invariant 0 <= i && i <= n {
        acc = acc + xs[i]
        i += 1
    }
    acc
}
"#;
        let prog = tenon_syntax::parse(src).unwrap();
        let rep = tenon_verify::check_program(&prog);
        assert!(rep.ok, "verify failed: {:?}", rep.diagnostics);
        let mir = lower_program_to_ssa_mir(&prog, &rep.mir);
        assert_eq!(mir.functions.len(), 1);
        let f = &mir.functions[0];
        assert!(f.is_arena);
        assert!(f.stack_promoted_locals.contains("p"));
        assert!(!f.last_use_move_spans.is_empty(), "expected Last-Use Move spans detected");
        let txt = format_mir(&mir);
        assert!(txt.contains("alloc.stack.alloca Pt"));
        assert!(txt.contains("idx.load unchecked"));
        assert!(txt.contains("move.local"));
    }

    #[test]
    fn mir_constant_folding_dce_and_fbip_drop_reuse() {
        let src = r#"
pub record Node {
    v Int
}

pub fn transform_node(n Node) Node {
    let k = 10 + 20 * 3
    let dead = 999 + 1
    let nv = n.v + k
    Node { v = nv }
}
"#;
        let prog = tenon_syntax::parse(src).unwrap();
        let rep = tenon_verify::check_program(&prog);
        assert!(rep.ok, "verify failed: {:?}", rep.diagnostics);
        let mir = lower_program_to_ssa_mir(&prog, &rep.mir);
        let f = &mir.functions[0];
        assert!(f.folded_constants >= 2, "expected const folding of 20*3 and 10+60, got {}", f.folded_constants);
        assert!(f.dce_removed >= 1, "expected DCE to drop dead constants, got {}", f.dce_removed);
        assert!(!f.fbip_reuse_pairs.is_empty(), "expected FBIP drop-reuse pairing for Node");
        let txt = format_mir(&mir);
        assert!(txt.contains("reuse.rc($n)"), "expected reuse.rc($n) in MIR:\n{}", txt);
    }
}
