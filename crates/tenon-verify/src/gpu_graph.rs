//! Declarative `@gpu_graph` Static Compiler Reconciler (`tenon-verify`).
//!
//! Uses ZERO new keywords — analyzes ordinary `@gpu_graph` functions using Tenon's
//! orthogonal core (`let` SSA def-use edges + `inout` writes + escape analysis) to:
//! 1. Build a verified compute/render pass DAG and reject cycles (`E0840`) or RAW/WAW
//!    aliasing hazards (`E0841`).
//! 2. Synthesize minimal pipeline synchronization barriers between producer/consumer nodes.
//! 3. Compute transient VRAM aliasing slots so non-escaping intermediates with disjoint
//!    lifetimes share the same VRAM allocation offset (zero wasted GPU memory).
//! 4. Detect consecutive pure elementwise `@kernel` dispatches (`a -> b`) where the
//!    intermediate is single-use and non-escaping, emitting an automatic Kernel Fusion plan!

use crate::{Diagnostic, ProofRecord, ProofTier};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use tenon_syntax::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuGraphNode {
    pub id: usize,
    pub kernel_name: String,
    pub output_var: Option<String>,
    pub input_vars: Vec<String>,
    pub depends_on: Vec<usize>,
    pub is_elementwise: bool,
    pub barrier_before: bool,
    pub vram_slot: Option<usize>,
    pub fused_into: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuFusionPlan {
    pub fused_name: String,
    pub first_kernel: String,
    pub second_kernel: String,
    pub eliminated_buffer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuGraphPlan {
    pub fn_name: String,
    pub nodes: Vec<GpuGraphNode>,
    pub barriers: usize,
    pub vram_slots_used: usize,
    pub transient_buffers_aliased: usize,
    pub fusions: Vec<GpuFusionPlan>,
}

pub fn has_kernel_attr(attrs: &[Attr]) -> bool {
    attrs.iter().any(|a| a.name == "kernel")
}

pub fn has_gpu_graph_attr(attrs: &[Attr]) -> bool {
    attrs.iter().any(|a| a.name == "gpu_graph")
}

fn is_elementwise_kernel(f: &FnDecl) -> bool {
    if !has_kernel_attr(&f.attrs) {
        return false;
    }
    // Check explicit `@kernel("elementwise")` or heuristic: pure/no nested loop over K
    for a in &f.attrs {
        if a.name == "kernel" {
            for arg in &a.args {
                if matches!(&arg.value.kind, ExprKind::Str(s) | ExprKind::Ident(s) if s == "elementwise" || s == "fuse") {
                    return true;
                }
            }
        }
    }
    // Single-pass elementwise heuristic: no `while` loop inside the kernel body
    !block_has_while(&f.body)
}

fn block_has_while(blk: &Block) -> bool {
    for s in &blk.stmts {
        match s {
            Stmt::While { .. } | Stmt::For { .. } => return true,
            Stmt::Expr { expr, .. } | Stmt::Let { value: expr, .. } => {
                if expr_has_while(expr) {
                    return true;
                }
            }
            _ => {}
        }
    }
    blk.tail.as_deref().is_some_and(expr_has_while)
}

fn expr_has_while(e: &Expr) -> bool {
    match &e.kind {
        ExprKind::If { then_block, else_branch, .. } => {
            block_has_while(then_block) || else_branch.as_deref().is_some_and(expr_has_while)
        }
        ExprKind::Block(b) | ExprKind::Unsafe(b) => block_has_while(b),
        _ => false,
    }
}

fn collect_ident_refs(expr: &Expr, out: &mut Vec<String>) {
    match &expr.kind {
        ExprKind::Ident(s) => {
            if !out.contains(s) {
                out.push(s.clone());
            }
        }
        ExprKind::Field { target, .. }
        | ExprKind::TupleIndex { target, .. }
        | ExprKind::Unary { expr: target, .. }
        | ExprKind::Cast { expr: target, .. }
        | ExprKind::Try(target) => collect_ident_refs(target, out),
        ExprKind::Binary { left, right, .. } => {
            collect_ident_refs(left, out);
            collect_ident_refs(right, out);
        }
        ExprKind::Call { args, .. } => {
            for a in args {
                collect_ident_refs(&a.value, out);
            }
        }
        ExprKind::List(es) | ExprKind::Tuple(es) => {
            for e in es {
                collect_ident_refs(e, out);
            }
        }
        ExprKind::RecordLit { fields, .. } => {
            for (_, v) in fields {
                collect_ident_refs(v, out);
            }
        }
        _ => {}
    }
}

/// Analyzes a `@gpu_graph` function, verifying DAG acyclicity, RAW/WAW hazard safety,
/// synthesizing minimal barriers, aliasing transient VRAM buffers, and fusing pure elementwise chains.
pub fn analyze_gpu_graph_fn(
    f: &FnDecl,
    all_fns: &BTreeMap<String, FnDecl>,
) -> (GpuGraphPlan, Vec<Diagnostic>, Vec<ProofRecord>) {
    let mut diags = Vec::new();
    let mut proofs = Vec::new();
    let mut nodes: Vec<GpuGraphNode> = Vec::new();
    let mut producer_of: BTreeMap<String, usize> = BTreeMap::new();
    let mut last_use_of: BTreeMap<String, usize> = BTreeMap::new();
    let mut use_count_of: BTreeMap<String, usize> = BTreeMap::new();
    let mut escaping_vars: BTreeSet<String> = BTreeSet::new();

    if let Some(tail) = &f.body.tail {
        let mut refs = Vec::new();
        collect_ident_refs(tail, &mut refs);
        escaping_vars.extend(refs);
    }

    for stmt in &f.body.stmts {
        match stmt {
            Stmt::Return { value: Some(v), .. } => {
                let mut refs = Vec::new();
                collect_ident_refs(v, &mut refs);
                escaping_vars.extend(refs);
            }
            Stmt::Let { name, value, span, .. } => {
                if let ExprKind::Call { callee, args } = &value.kind {
                    if let ExprKind::Ident(kname) = &callee.kind {
                        if !all_fns.get(kname).is_some_and(|fd| has_kernel_attr(&fd.attrs)) {
                            continue;
                        }
                        let mut inputs = Vec::new();
                        for a in args {
                            collect_ident_refs(&a.value, &mut inputs);
                        }
                        // Check self-cycle (`let x = kernel(x)`) -> E0840
                        if inputs.contains(name) {
                            diags.push(Diagnostic {
                                code: "E0840".into(),
                                message: format!(
                                    "`@gpu_graph` cycle detected in `{}`: `{}` depends on itself via `{}`",
                                    f.name, name, kname
                                ),
                                span: *span,
                                notes: vec!["bind GPU passes to distinct SSA `let` variables to form a verified DAG".into()],
                                counterexample: BTreeMap::new(),
                            });
                        }
                        // Check WAW / duplicate alias hazard in arguments to an inout kernel -> E0841
                        if let Some(kdecl) = all_fns.get(kname) {
                            let mut written: BTreeSet<String> = BTreeSet::new();
                            let mut read: BTreeSet<String> = BTreeSet::new();
                            for (idx, a) in args.iter().enumerate() {
                                if let ExprKind::Ident(v) = &a.value.kind {
                                    if kdecl.params.get(idx).is_some_and(|p| p.is_inout) {
                                        if !written.insert(v.clone()) || read.contains(v) {
                                            diags.push(Diagnostic {
                                                code: "E0841".into(),
                                                message: format!(
                                                    "`@gpu_graph` data hazard in `{}`: buffer `{}` aliases multiple arguments of `{}`",
                                                    f.name, v, kname
                                                ),
                                                span: a.value.span,
                                                notes: vec!["separate read and `inout` write buffers to eliminate GPU RAW/WAW race".into()],
                                                counterexample: BTreeMap::new(),
                                            });
                                        }
                                    } else {
                                        if written.contains(v) {
                                            diags.push(Diagnostic {
                                                code: "E0841".into(),
                                                message: format!(
                                                    "`@gpu_graph` RAW hazard in `{}`: buffer `{}` is both written (`inout`) and read in `{}`",
                                                    f.name, v, kname
                                                ),
                                                span: a.value.span,
                                                notes: vec!["GPU kernels cannot alias an `inout` output buffer with a read-only input buffer".into()],
                                                counterexample: BTreeMap::new(),
                                            });
                                        }
                                        read.insert(v.clone());
                                    }
                                }
                            }
                        }

                        let node_id = nodes.len();
                        let mut deps = Vec::new();
                        for iv in &inputs {
                            if let Some(&pid) = producer_of.get(iv) {
                                if !deps.contains(&pid) {
                                    deps.push(pid);
                                }
                            }
                            last_use_of.insert(iv.clone(), node_id);
                            *use_count_of.entry(iv.clone()).or_insert(0) += 1;
                        }
                        let is_ew = all_fns
                            .get(kname)
                            .is_some_and(is_elementwise_kernel);
                        producer_of.insert(name.clone(), node_id);
                        nodes.push(GpuGraphNode {
                            id: node_id,
                            kernel_name: kname.clone(),
                            output_var: Some(name.clone()),
                            input_vars: inputs,
                            depends_on: deps,
                            is_elementwise: is_ew,
                            barrier_before: false,
                            vram_slot: None,
                            fused_into: None,
                        });
                    }
                }
            }
            Stmt::Expr { expr, .. } => {
                if let ExprKind::Call { callee, args } = &expr.kind {
                    if let ExprKind::Ident(kname) = &callee.kind {
                        if let Some(kdecl) = all_fns.get(kname) {
                            if !has_kernel_attr(&kdecl.attrs) {
                                continue;
                            }
                            let mut inputs = Vec::new();
                            let mut written: BTreeSet<String> = BTreeSet::new();
                            let mut read: BTreeSet<String> = BTreeSet::new();
                            let mut out_var: Option<String> = None;
                            for (idx, a) in args.iter().enumerate() {
                                if let ExprKind::Ident(v) = &a.value.kind {
                                    if kdecl.params.get(idx).is_some_and(|p| p.is_inout) {
                                        if !written.insert(v.clone()) || read.contains(v) {
                                            diags.push(Diagnostic {
                                                code: "E0841".into(),
                                                message: format!(
                                                    "`@gpu_graph` data hazard in `{}`: buffer `{}` aliases multiple arguments of `{}`",
                                                    f.name, v, kname
                                                ),
                                                span: a.value.span,
                                                notes: vec!["separate read and `inout` write buffers".into()],
                                                counterexample: BTreeMap::new(),
                                            });
                                        }
                                        out_var = Some(v.clone());
                                    } else {
                                        if written.contains(v) {
                                            diags.push(Diagnostic {
                                                code: "E0841".into(),
                                                message: format!(
                                                    "`@gpu_graph` RAW hazard in `{}`: buffer `{}` is both written and read in `{}`",
                                                    f.name, v, kname
                                                ),
                                                span: a.value.span,
                                                notes: vec!["cannot read and write the same GPU buffer in one dispatch".into()],
                                                counterexample: BTreeMap::new(),
                                            });
                                        }
                                        read.insert(v.clone());
                                        inputs.push(v.clone());
                                    }
                                }
                            }
                            let node_id = nodes.len();
                            let mut deps = Vec::new();
                            for iv in &inputs {
                                if let Some(&pid) = producer_of.get(iv) {
                                    if !deps.contains(&pid) {
                                        deps.push(pid);
                                    }
                                }
                                last_use_of.insert(iv.clone(), node_id);
                                *use_count_of.entry(iv.clone()).or_insert(0) += 1;
                            }
                            if let Some(ref ov) = out_var {
                                producer_of.insert(ov.clone(), node_id);
                            }
                            nodes.push(GpuGraphNode {
                                id: node_id,
                                kernel_name: kname.clone(),
                                output_var: out_var,
                                input_vars: inputs,
                                depends_on: deps,
                                is_elementwise: is_elementwise_kernel(kdecl),
                                barrier_before: false,
                                vram_slot: None,
                                fused_into: None,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // 1. Automatic Elementwise Kernel Fusion:
    // If node `i` produces a transient single-use buffer `v` consumed immediately by elementwise node `j`
    // and `i` is also elementwise, fuse `i` + `j` into a single GPU dispatch (eliminating VRAM round-trip)!
    let mut fusions = Vec::new();
    for i in 0..nodes.len() {
        if !nodes[i].is_elementwise || nodes[i].fused_into.is_some() {
            continue;
        }
        let Some(ref out_v) = nodes[i].output_var.clone() else {
            continue;
        };
        if escaping_vars.contains(out_v) {
            continue;
        }
        if use_count_of.get(out_v).copied().unwrap_or(0) != 1 {
            continue;
        }
        if let Some(&consumer_id) = last_use_of.get(out_v) {
            if consumer_id > i
                && consumer_id < nodes.len()
                && nodes[consumer_id].is_elementwise
                && nodes[consumer_id].fused_into.is_none()
            {
                let fused_name = format!(
                    "{}__fused__{}",
                    nodes[i].kernel_name, nodes[consumer_id].kernel_name
                );
                fusions.push(GpuFusionPlan {
                    fused_name,
                    first_kernel: nodes[i].kernel_name.clone(),
                    second_kernel: nodes[consumer_id].kernel_name.clone(),
                    eliminated_buffer: out_v.clone(),
                });
                nodes[i].fused_into = Some(consumer_id);
            }
        }
    }

    // 2. Minimal Barrier Synthesis: insert a barrier before a node iff it depends on an unfused prior node.
    let mut barriers = 0usize;
    for i in 0..nodes.len() {
        let needs_barrier = nodes[i]
            .depends_on
            .iter()
            .any(|&d| nodes[d].fused_into != Some(i));
        if needs_barrier {
            nodes[i].barrier_before = true;
            barriers += 1;
        }
    }

    // 3. Transient VRAM Aliasing (linear-scan interval coloring over non-escaping GPU buffers):
    // Disjoint lifetimes reuse the SAME VRAM buffer slot!
    let mut slot_free_at: Vec<usize> = Vec::new();
    let mut transient_count = 0usize;
    for i in 0..nodes.len() {
        if nodes[i].fused_into.is_some() {
            continue;
        }
        let Some(ref ov) = nodes[i].output_var else {
            continue;
        };
        if escaping_vars.contains(ov) {
            continue;
        }
        let end_use = last_use_of.get(ov).copied().unwrap_or(i);
        transient_count += 1;
        if let Some(pos) = slot_free_at.iter().position(|&free_idx| free_idx <= i) {
            slot_free_at[pos] = end_use;
            nodes[i].vram_slot = Some(pos);
        } else {
            let slot = slot_free_at.len();
            slot_free_at.push(end_use);
            nodes[i].vram_slot = Some(slot);
        }
    }
    let vram_slots_used = slot_free_at.len();
    let transient_buffers_aliased = transient_count.saturating_sub(vram_slots_used);

    let plan = GpuGraphPlan {
        fn_name: f.name.clone(),
        nodes,
        barriers,
        vram_slots_used,
        transient_buffers_aliased,
        fusions,
    };

    if diags.is_empty() {
        proofs.push(ProofRecord {
            obligation: format!(
                "@gpu_graph DAG proof for `{}` ({} pass(es), {} barrier(s), {} fused, {} VRAM slot(s))",
                f.name,
                plan.nodes.len(),
                plan.barriers,
                plan.fusions.len(),
                plan.vram_slots_used
            ),
            tier: ProofTier::L0Fast,
            span: f.span,
        });
    }

    (plan, diags, proofs)
}
