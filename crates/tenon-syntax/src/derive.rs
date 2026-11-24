//! Compile-time `@derive(ToJson, FromJson, JsonSchema)` expansion pass.
//!
//! Operates purely on AST structural metadata (`fields` / `variants` / attributes /
//! refinement predicates) to synthesize standard `impl ToJson for T`, `impl FromJson for T`,
//! and `impl JsonSchema for T` blocks — keeping the runtime 100% reflection-free while
//! guaranteeing that derived `from_json` passes Tenon's L0/L1 formal verifier!

use crate::ast::*;
use crate::span::Span;
use std::collections::BTreeMap;

fn sp() -> Span {
    Span::default()
}

fn named(s: &str) -> TypeExpr {
    TypeExpr::Named {
        name: s.to_string(),
        version: None,
        args: Vec::new(),
        span: sp(),
    }
}

fn generic(s: &str, args: Vec<TypeExpr>) -> TypeExpr {
    TypeExpr::Named {
        name: s.to_string(),
        version: None,
        args,
        span: sp(),
    }
}

fn ident(s: &str) -> Expr {
    Expr {
        kind: ExprKind::Ident(s.to_string()),
        span: sp(),
    }
}

fn str_lit(s: &str) -> Expr {
    Expr {
        kind: ExprKind::Str(s.to_string()),
        span: sp(),
    }
}

fn int_lit(n: i64) -> Expr {
    Expr {
        kind: ExprKind::Int(n),
        span: sp(),
    }
}

fn call(fname: &str, args: Vec<Expr>) -> Expr {
    Expr {
        kind: ExprKind::Call {
            callee: Box::new(ident(fname)),
            args: args
                .into_iter()
                .map(|value| CallArg { name: None, value })
                .collect(),
        },
        span: sp(),
    }
}

fn method_call(recv: Expr, method: &str, args: Vec<Expr>) -> Expr {
    Expr {
        kind: ExprKind::Call {
            callee: Box::new(Expr {
                kind: ExprKind::Field {
                    target: Box::new(recv),
                    field: method.to_string(),
                },
                span: sp(),
            }),
            args: args
                .into_iter()
                .map(|value| CallArg { name: None, value })
                .collect(),
        },
        span: sp(),
    }
}

fn field_access(recv: Expr, f: &str) -> Expr {
    Expr {
        kind: ExprKind::Field {
            target: Box::new(recv),
            field: f.to_string(),
        },
        span: sp(),
    }
}

fn let_stmt(name: &str, ty: Option<TypeExpr>, value: Expr, is_mut: bool) -> Stmt {
    Stmt::Let {
        name: name.to_string(),
        ty,
        value,
        is_mut,
        span: sp(),
    }
}

fn assign_ident(name: &str, value: Expr) -> Stmt {
    Stmt::Assign {
        target: ident(name),
        op: AssignOp::Set,
        value,
        span: sp(),
    }
}

fn return_expr(e: Expr) -> Stmt {
    Stmt::Return {
        value: Some(e),
        span: sp(),
    }
}

fn json_err(msg: &str) -> Expr {
    call(
        "Err",
        vec![Expr {
            kind: ExprKind::RecordLit {
                name: "JsonError".into(),
                version: None,
                fields: vec![
                    ("pos".into(), int_lit(0)),
                    ("message".into(), str_lit(msg)),
                ],
            },
            span: sp(),
        }],
    )
}

fn extract_derives(attrs: &[Attr]) -> Vec<String> {
    let mut out = Vec::new();
    for a in attrs {
        if a.name == "derive" {
            for arg in &a.args {
                if let ExprKind::Ident(name) = &arg.value.kind {
                    out.push(name.clone());
                } else if let ExprKind::Str(name) = &arg.value.kind {
                    out.push(name.clone());
                }
            }
        }
    }
    out
}

fn attr_str(attrs: &[Attr], name: &str) -> Option<String> {
    attrs.iter().find(|a| a.name == name).and_then(|a| {
        a.args.first().and_then(|arg| match &arg.value.kind {
            ExprKind::Str(s) => Some(s.clone()),
            _ => None,
        })
    })
}

fn has_attr(attrs: &[Attr], name: &str) -> bool {
    attrs.iter().any(|a| a.name == name)
}

fn resolve_base<'a>(
    ty: &'a TypeExpr,
    aliases: &'a BTreeMap<String, TypeAliasDecl>,
) -> &'a TypeExpr {
    let mut cur = ty;
    for _ in 0..8 {
        if let TypeExpr::Named { name, args, .. } = cur {
            if args.is_empty() {
                if let Some(a) = aliases.get(name) {
                    cur = &a.base;
                    continue;
                }
            }
        }
        break;
    }
    cur
}

fn is_int_name(name: &str) -> bool {
    matches!(name, "Int" | "Int64" | "Int32" | "UInt16" | "USize")
}

/// Expand `@derive(ToJson, FromJson, JsonSchema)` on records and enums into standard `impl` items.
pub fn expand_derives(program: &mut Program) {
    let mut aliases: BTreeMap<String, TypeAliasDecl> = BTreeMap::new();
    for item in &program.items {
        if let Item::TypeAlias(ta) = item {
            aliases.insert(ta.name.clone(), ta.clone());
        }
    }

    let mut generated: Vec<Item> = Vec::new();
    for item in &program.items {
        match item {
            Item::Record(rec) => {
                let derives = extract_derives(&rec.attrs);
                if derives.is_empty() {
                    continue;
                }
                if derives.iter().any(|d| d == "ToJson" || d == "Serialize") {
                    generated.push(Item::Impl(derive_record_to_json(rec, &aliases)));
                }
                if derives.iter().any(|d| d == "FromJson" || d == "Deserialize") {
                    generated.push(Item::Impl(derive_record_from_json(rec, &aliases)));
                }
                if derives.iter().any(|d| d == "JsonSchema") {
                    generated.push(Item::Impl(derive_record_json_schema(rec, &aliases)));
                }
            }
            Item::Enum(en) => {
                let derives = extract_derives(&en.attrs);
                if derives.is_empty() {
                    continue;
                }
                if derives.iter().any(|d| d == "ToJson" || d == "Serialize") {
                    generated.push(Item::Impl(derive_enum_to_json(en, &aliases)));
                }
                if derives.iter().any(|d| d == "FromJson" || d == "Deserialize") {
                    generated.push(Item::Impl(derive_enum_from_json(en, &aliases)));
                }
            }
            _ => {}
        }
    }
    program.items.extend(generated);
}

fn encode_value_expr(
    val: Expr,
    ty: &TypeExpr,
    aliases: &BTreeMap<String, TypeAliasDecl>,
    depth: usize,
) -> Expr {
    let base = resolve_base(ty, aliases);
    match base {
        TypeExpr::Named { name, args, .. } => {
            if is_int_name(name) {
                return call("JInt", vec![val]);
            }
            if name == "Bool" {
                return call("JBool", vec![val]);
            }
            if name == "Text" {
                return call("JStr", vec![val]);
            }
            if name == "JsonValue" {
                return val;
            }
            if name == "Option" && args.len() == 1 {
                let bind = format!("__opt_{}", depth);
                let inner = encode_value_expr(ident(&bind), &args[0], aliases, depth + 1);
                return Expr {
                    kind: ExprKind::Match {
                        scrutinee: Box::new(val),
                        arms: vec![
                            MatchArm {
                                patterns: vec![Pattern::Variant {
                                    name: "Some".into(),
                                    binding: Some(bind),
                                    span: sp(),
                                }],
                                guard: None,
                                body: inner,
                                span: sp(),
                            },
                            MatchArm {
                                patterns: vec![Pattern::Variant {
                                    name: "None".into(),
                                    binding: None,
                                    span: sp(),
                                }],
                                guard: None,
                                body: ident("JNull"),
                                span: sp(),
                            },
                        ],
                    },
                    span: sp(),
                };
            }
            if name == "List" && args.len() == 1 {
                let arr_v = format!("__arr_{}", depth);
                let src_v = format!("__src_{}", depth);
                let idx_v = format!("__i_{}", depth);
                let elem_expr = Expr {
                    kind: ExprKind::Index {
                        target: Box::new(ident(&src_v)),
                        index: Box::new(ident(&idx_v)),
                    },
                    span: sp(),
                };
                let enc_elem = encode_value_expr(elem_expr, &args[0], aliases, depth + 1);
                return Expr {
                    kind: ExprKind::Block(Block {
                        stmts: vec![
                            let_stmt(&src_v, Some(ty.clone()), val, false),
                            let_stmt(&arr_v, None, call("json_arr_new", vec![]), true),
                            let_stmt(&idx_v, None, int_lit(0), true),
                            Stmt::While {
                                cond: Expr {
                                    kind: ExprKind::Binary {
                                        op: BinOp::Lt,
                                        left: Box::new(ident(&idx_v)),
                                        right: Box::new(call("len", vec![ident(&src_v)])),
                                    },
                                    span: sp(),
                                },
                                invariants: vec![],
                                decreases: None,
                                body: Block {
                                    stmts: vec![
                                        assign_ident(
                                            &arr_v,
                                            call("json_arr_push", vec![ident(&arr_v), enc_elem]),
                                        ),
                                        Stmt::Assign {
                                            target: ident(&idx_v),
                                            op: AssignOp::AddSet,
                                            value: int_lit(1),
                                            span: sp(),
                                        },
                                    ],
                                    tail: None,
                                    span: sp(),
                                },
                                span: sp(),
                            },
                        ],
                        tail: Some(Box::new(ident(&arr_v))),
                        span: sp(),
                    }),
                    span: sp(),
                };
            }
            // Nested record/enum implementing ToJson
            method_call(val, "to_json", vec![])
        }
        _ => ident("JNull"),
    }
}

fn derive_record_to_json(
    rec: &RecordDecl,
    aliases: &BTreeMap<String, TypeAliasDecl>,
) -> ImplDecl {
    let mut stmts = vec![let_stmt(
        "__obj",
        None,
        call("json_obj_new", vec![]),
        true,
    )];
    for (idx, f) in rec.fields.iter().enumerate() {
        if has_attr(&f.attrs, "skip") {
            continue;
        }
        let key = attr_str(&f.attrs, "rename").unwrap_or_else(|| f.name.clone());
        let fexpr = field_access(ident("self"), &f.name);
        let skip_none = has_attr(&f.attrs, "skip_if_none");
        let base = resolve_base(&f.ty, aliases);
        if skip_none {
            if let TypeExpr::Named { name, args, .. } = base {
                if name == "Option" && args.len() == 1 {
                    let bind = format!("__v_{}", idx);
                    let enc = encode_value_expr(ident(&bind), &args[0], aliases, idx + 1);
                    stmts.push(Stmt::Expr {
                        expr: Expr {
                            kind: ExprKind::Match {
                                scrutinee: Box::new(fexpr),
                                arms: vec![
                                    MatchArm {
                                        patterns: vec![Pattern::Variant {
                                            name: "Some".into(),
                                            binding: Some(bind),
                                            span: sp(),
                                        }],
                                        guard: None,
                                        body: Expr {
                                            kind: ExprKind::Block(Block {
                                                stmts: vec![assign_ident(
                                                    "__obj",
                                                    call(
                                                        "json_obj_set",
                                                        vec![ident("__obj"), str_lit(&key), enc],
                                                    ),
                                                )],
                                                tail: None,
                                                span: sp(),
                                            }),
                                            span: sp(),
                                        },
                                        span: sp(),
                                    },
                                    MatchArm {
                                        patterns: vec![Pattern::Variant {
                                            name: "None".into(),
                                            binding: None,
                                            span: sp(),
                                        }],
                                        guard: None,
                                        body: Expr {
                                            kind: ExprKind::Block(Block {
                                                stmts: vec![],
                                                tail: None,
                                                span: sp(),
                                            }),
                                            span: sp(),
                                        },
                                        span: sp(),
                                    },
                                ],
                            },
                            span: sp(),
                        },
                        span: sp(),
                    });
                    continue;
                }
            }
        }
        let enc = encode_value_expr(fexpr, &f.ty, aliases, idx + 1);
        stmts.push(assign_ident(
            "__obj",
            call("json_obj_set", vec![ident("__obj"), str_lit(&key), enc]),
        ));
    }

    let method = FnDecl {
        attrs: vec![],
        is_pub: true,
        is_pure: true,
        is_async: false,
        is_test: false,
        name: "to_json".into(),
        type_params: vec![],
        type_param_bounds: BTreeMap::new(),
        params: vec![ParamDecl {
            is_inout: false,
            is_comptime: false,
            name: "self".into(),
            ty: named("Self"),
            predicate: None,
            span: sp(),
        }],
        ret_ty: Some(named("JsonValue")),
        requires: vec![],
        ensures: vec![],
        body: Block {
            stmts,
            tail: Some(Box::new(ident("__obj"))),
            span: sp(),
        },
        span: rec.span,
    };

    ImplDecl {
        attrs: vec![],
        type_params: vec![],
        type_param_bounds: BTreeMap::new(),
        interface: Some(named("ToJson")),
        target_ty: named(&rec.name),
        assoc_bindings: vec![],
        methods: vec![method],
        span: rec.span,
    }
}

fn decode_value_expr(
    jval: Expr,
    ty: &TypeExpr,
    aliases: &BTreeMap<String, TypeAliasDecl>,
    field_label: &str,
    depth: usize,
) -> Expr {
    let base = resolve_base(ty, aliases);
    match base {
        TypeExpr::Named { name, args, .. } => {
            if is_int_name(name) {
                return Expr {
                    kind: ExprKind::Match {
                        scrutinee: Box::new(call("json_as_int", vec![jval])),
                        arms: vec![
                            MatchArm {
                                patterns: vec![Pattern::Variant {
                                    name: "Some".into(),
                                    binding: Some(format!("__n_{}", depth)),
                                    span: sp(),
                                }],
                                guard: None,
                                body: ident(&format!("__n_{}", depth)),
                                span: sp(),
                            },
                            MatchArm {
                                patterns: vec![Pattern::Variant {
                                    name: "None".into(),
                                    binding: None,
                                    span: sp(),
                                }],
                                guard: None,
                                body: Expr {
                                    kind: ExprKind::Block(Block {
                                        stmts: vec![return_expr(json_err(&format!(
                                            "expected integer for `{}`",
                                            field_label
                                        )))],
                                        tail: None,
                                        span: sp(),
                                    }),
                                    span: sp(),
                                },
                                span: sp(),
                            },
                        ],
                    },
                    span: sp(),
                };
            }
            if name == "Bool" {
                return Expr {
                    kind: ExprKind::Match {
                        scrutinee: Box::new(call("json_as_bool", vec![jval])),
                        arms: vec![
                            MatchArm {
                                patterns: vec![Pattern::Variant {
                                    name: "Some".into(),
                                    binding: Some(format!("__b_{}", depth)),
                                    span: sp(),
                                }],
                                guard: None,
                                body: ident(&format!("__b_{}", depth)),
                                span: sp(),
                            },
                            MatchArm {
                                patterns: vec![Pattern::Variant {
                                    name: "None".into(),
                                    binding: None,
                                    span: sp(),
                                }],
                                guard: None,
                                body: Expr {
                                    kind: ExprKind::Block(Block {
                                        stmts: vec![return_expr(json_err(&format!(
                                            "expected boolean for `{}`",
                                            field_label
                                        )))],
                                        tail: None,
                                        span: sp(),
                                    }),
                                    span: sp(),
                                },
                                span: sp(),
                            },
                        ],
                    },
                    span: sp(),
                };
            }
            if name == "Text" {
                return Expr {
                    kind: ExprKind::Match {
                        scrutinee: Box::new(call("json_as_str", vec![jval])),
                        arms: vec![
                            MatchArm {
                                patterns: vec![Pattern::Variant {
                                    name: "Some".into(),
                                    binding: Some(format!("__s_{}", depth)),
                                    span: sp(),
                                }],
                                guard: None,
                                body: ident(&format!("__s_{}", depth)),
                                span: sp(),
                            },
                            MatchArm {
                                patterns: vec![Pattern::Variant {
                                    name: "None".into(),
                                    binding: None,
                                    span: sp(),
                                }],
                                guard: None,
                                body: Expr {
                                    kind: ExprKind::Block(Block {
                                        stmts: vec![return_expr(json_err(&format!(
                                            "expected string for `{}`",
                                            field_label
                                        )))],
                                        tail: None,
                                        span: sp(),
                                    }),
                                    span: sp(),
                                },
                                span: sp(),
                            },
                        ],
                    },
                    span: sp(),
                };
            }
            if name == "JsonValue" {
                return jval;
            }
            if name == "List" && args.len() == 1 {
                let items_v = format!("__jitems_{}", depth);
                let out_v = format!("__lout_{}", depth);
                let idx_v = format!("__li_{}", depth);
                let raw_elem = Expr {
                    kind: ExprKind::Index {
                        target: Box::new(ident(&items_v)),
                        index: Box::new(ident(&idx_v)),
                    },
                    span: sp(),
                };
                let dec_elem =
                    decode_value_expr(raw_elem, &args[0], aliases, field_label, depth + 1);
                return Expr {
                    kind: ExprKind::Match {
                        scrutinee: Box::new(jval),
                        arms: vec![
                            MatchArm {
                                patterns: vec![Pattern::Variant {
                                    name: "JArr".into(),
                                    binding: Some(items_v.clone()),
                                    span: sp(),
                                }],
                                guard: None,
                                body: Expr {
                                    kind: ExprKind::Block(Block {
                                        stmts: vec![
                                            let_stmt(
                                                &out_v,
                                                Some(ty.clone()),
                                                Expr {
                                                    kind: ExprKind::List(vec![]),
                                                    span: sp(),
                                                },
                                                true,
                                            ),
                                            let_stmt(&idx_v, None, int_lit(0), true),
                                            Stmt::While {
                                                cond: Expr {
                                                    kind: ExprKind::Binary {
                                                        op: BinOp::Lt,
                                                        left: Box::new(ident(&idx_v)),
                                                        right: Box::new(call(
                                                            "len",
                                                            vec![ident(&items_v)],
                                                        )),
                                                    },
                                                    span: sp(),
                                                },
                                                invariants: vec![],
                                                decreases: None,
                                                body: Block {
                                                    stmts: vec![
                                                        assign_ident(
                                                            &out_v,
                                                            call(
                                                                "push",
                                                                vec![ident(&out_v), dec_elem],
                                                            ),
                                                        ),
                                                        Stmt::Assign {
                                                            target: ident(&idx_v),
                                                            op: AssignOp::AddSet,
                                                            value: int_lit(1),
                                                            span: sp(),
                                                        },
                                                    ],
                                                    tail: None,
                                                    span: sp(),
                                                },
                                                span: sp(),
                                            },
                                        ],
                                        tail: Some(Box::new(ident(&out_v))),
                                        span: sp(),
                                    }),
                                    span: sp(),
                                },
                                span: sp(),
                            },
                            MatchArm {
                                patterns: vec![Pattern::Wildcard(sp())],
                                guard: None,
                                body: Expr {
                                    kind: ExprKind::Block(Block {
                                        stmts: vec![return_expr(json_err(&format!(
                                            "expected array for `{}`",
                                            field_label
                                        )))],
                                        tail: None,
                                        span: sp(),
                                    }),
                                    span: sp(),
                                },
                                span: sp(),
                            },
                        ],
                    },
                    span: sp(),
                };
            }
            // Nested record/enum implementing FromJson: `Type.from_json(jval)`
            let ok_v = format!("__sub_{}", depth);
            let err_v = format!("__sub_e_{}", depth);
            Expr {
                kind: ExprKind::Match {
                    scrutinee: Box::new(method_call(ident(name), "from_json", vec![jval])),
                    arms: vec![
                        MatchArm {
                            patterns: vec![Pattern::Variant {
                                name: "Ok".into(),
                                binding: Some(ok_v.clone()),
                                span: sp(),
                            }],
                            guard: None,
                            body: ident(&ok_v),
                            span: sp(),
                        },
                        MatchArm {
                            patterns: vec![Pattern::Variant {
                                name: "Err".into(),
                                binding: Some(err_v.clone()),
                                span: sp(),
                            }],
                            guard: None,
                            body: Expr {
                                kind: ExprKind::Block(Block {
                                    stmts: vec![return_expr(call("Err", vec![ident(&err_v)]))],
                                    tail: None,
                                    span: sp(),
                                }),
                                span: sp(),
                            },
                            span: sp(),
                        },
                    ],
                },
                span: sp(),
            }
        }
        _ => jval,
    }
}

fn subst_ident(expr: &Expr, replacements: &BTreeMap<String, String>) -> Expr {
    let kind = match &expr.kind {
        ExprKind::Ident(s) => ExprKind::Ident(
            replacements
                .get(s)
                .cloned()
                .unwrap_or_else(|| s.clone()),
        ),
        ExprKind::Field { target, field } => {
            if matches!(&target.kind, ExprKind::Ident(s) if s == "self") {
                if let Some(rep) = replacements.get(field) {
                    ExprKind::Ident(rep.clone())
                } else {
                    ExprKind::Field {
                        target: Box::new(subst_ident(target, replacements)),
                        field: field.clone(),
                    }
                }
            } else {
                ExprKind::Field {
                    target: Box::new(subst_ident(target, replacements)),
                    field: field.clone(),
                }
            }
        }
        ExprKind::Binary { op, left, right } => ExprKind::Binary {
            op: *op,
            left: Box::new(subst_ident(left, replacements)),
            right: Box::new(subst_ident(right, replacements)),
        },
        ExprKind::Unary { op, expr: inner } => ExprKind::Unary {
            op: *op,
            expr: Box::new(subst_ident(inner, replacements)),
        },
        ExprKind::Call { callee, args } => ExprKind::Call {
            callee: Box::new(subst_ident(callee, replacements)),
            args: args
                .iter()
                .map(|a| CallArg {
                    name: a.name.clone(),
                    value: subst_ident(&a.value, replacements),
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

fn emit_refinement_guard(pred: Expr, err_msg: &str) -> Stmt {
    Stmt::Expr {
        expr: Expr {
            kind: ExprKind::If {
                cond: Box::new(Expr {
                    kind: ExprKind::Unary {
                        op: UnOp::Not,
                        expr: Box::new(pred),
                    },
                    span: sp(),
                }),
                then_block: Block {
                    stmts: vec![return_expr(json_err(err_msg))],
                    tail: None,
                    span: sp(),
                },
                else_branch: None,
            },
            span: sp(),
        },
        span: sp(),
    }
}

fn derive_record_from_json(
    rec: &RecordDecl,
    aliases: &BTreeMap<String, TypeAliasDecl>,
) -> ImplDecl {
    let mut stmts = Vec::new();
    // 1. Ensure input is a JSON object
    stmts.push(Stmt::Expr {
        expr: Expr {
            kind: ExprKind::Match {
                scrutinee: Box::new(ident("v")),
                arms: vec![
                    MatchArm {
                        patterns: vec![Pattern::Variant {
                            name: "JObj".into(),
                            binding: Some("_".into()),
                            span: sp(),
                        }],
                        guard: None,
                        body: Expr {
                            kind: ExprKind::Block(Block {
                                stmts: vec![],
                                tail: None,
                                span: sp(),
                            }),
                            span: sp(),
                        },
                        span: sp(),
                    },
                    MatchArm {
                        patterns: vec![Pattern::Wildcard(sp())],
                        guard: None,
                        body: Expr {
                            kind: ExprKind::Block(Block {
                                stmts: vec![return_expr(json_err(&format!(
                                    "expected JSON object for `{}`",
                                    rec.name
                                )))],
                                tail: None,
                                span: sp(),
                            }),
                            span: sp(),
                        },
                        span: sp(),
                    },
                ],
            },
            span: sp(),
        },
        span: sp(),
    });

    // Map field names -> decoded local variable names (`__f_<name>`) for cross-field predicates
    let mut field_var_map: BTreeMap<String, String> = BTreeMap::new();
    for f in &rec.fields {
        field_var_map.insert(f.name.clone(), format!("__f_{}", f.name));
    }

    // 2. Decode every field + emit refinement guards (Parse, Don't Validate!)
    for (idx, f) in rec.fields.iter().enumerate() {
        let local_name = format!("__f_{}", f.name);
        let key = attr_str(&f.attrs, "rename").unwrap_or_else(|| f.name.clone());
        let label = format!("{}.{}", rec.name, key);
        let base = resolve_base(&f.ty, aliases);

        let is_opt = matches!(base, TypeExpr::Named { name, args, .. } if name == "Option" && args.len() == 1);
        let lookup_expr = call("json_get", vec![ident("v"), str_lit(&key)]);

        let decoded_expr = if is_opt {
            let TypeExpr::Named { args, .. } = base else {
                unreachable!()
            };
            let jv_bind = format!("__jf_{}", idx);
            let inner_dec = decode_value_expr(
                ident(&jv_bind),
                &args[0],
                aliases,
                &label,
                idx * 10 + 1,
            );
            Expr {
                kind: ExprKind::Match {
                    scrutinee: Box::new(lookup_expr),
                    arms: vec![
                        MatchArm {
                            patterns: vec![Pattern::Variant {
                                name: "None".into(),
                                binding: None,
                                span: sp(),
                            }],
                            guard: None,
                            body: ident("None"),
                            span: sp(),
                        },
                        MatchArm {
                            patterns: vec![Pattern::Variant {
                                name: "Some".into(),
                                binding: Some(jv_bind.clone()),
                                span: sp(),
                            }],
                            guard: None,
                            body: Expr {
                                kind: ExprKind::Match {
                                    scrutinee: Box::new(ident(&jv_bind)),
                                    arms: vec![
                                        MatchArm {
                                            patterns: vec![Pattern::Variant {
                                                name: "JNull".into(),
                                                binding: None,
                                                span: sp(),
                                            }],
                                            guard: None,
                                            body: ident("None"),
                                            span: sp(),
                                        },
                                        MatchArm {
                                            patterns: vec![Pattern::Wildcard(sp())],
                                            guard: None,
                                            body: call("Some", vec![inner_dec]),
                                            span: sp(),
                                        },
                                    ],
                                },
                                span: sp(),
                            },
                            span: sp(),
                        },
                    ],
                },
                span: sp(),
            }
        } else {
            let jv_bind = format!("__jf_{}", idx);
            let inner_dec =
                decode_value_expr(ident(&jv_bind), &f.ty, aliases, &label, idx * 10 + 1);
            let none_body = if let Some(def) = &f.default {
                def.clone()
            } else {
                Expr {
                    kind: ExprKind::Block(Block {
                        stmts: vec![return_expr(json_err(&format!(
                            "missing required field `{}` in `{}`",
                            key, rec.name
                        )))],
                        tail: None,
                        span: sp(),
                    }),
                    span: sp(),
                }
            };
            Expr {
                kind: ExprKind::Match {
                    scrutinee: Box::new(lookup_expr),
                    arms: vec![
                        MatchArm {
                            patterns: vec![Pattern::Variant {
                                name: "Some".into(),
                                binding: Some(jv_bind),
                                span: sp(),
                            }],
                            guard: None,
                            body: inner_dec,
                            span: sp(),
                        },
                        MatchArm {
                            patterns: vec![Pattern::Variant {
                                name: "None".into(),
                                binding: None,
                                span: sp(),
                            }],
                            guard: None,
                            body: none_body,
                            span: sp(),
                        },
                    ],
                },
                span: sp(),
            }
        };

        // Bind to base type first so refinement check runs cleanly before constructing `R { ... }`!
        stmts.push(let_stmt(&local_name, None, decoded_expr, false));

        // Check alias refinement predicate (e.g., `NonNegative`, `Positive`, `SafeRelPath`)
        if let TypeExpr::Named { name: tname, args, .. } = &f.ty {
            if args.is_empty() {
                if let Some(alias) = aliases.get(tname) {
                    if let Some(pred) = &alias.predicate {
                        let mut repl = BTreeMap::new();
                        repl.insert("self".into(), local_name.clone());
                        stmts.push(emit_refinement_guard(
                            subst_ident(pred, &repl),
                            &format!("field `{}` violates refinement `{}`", key, tname),
                        ));
                    }
                }
            }
        }

        // Check inline field `where` predicate (may reference previous fields, e.g., `Date.day`)
        if let Some(fpred) = &f.predicate {
            let mut repl = field_var_map.clone();
            repl.insert("self".into(), local_name.clone());
            stmts.push(emit_refinement_guard(
                subst_ident(fpred, &repl),
                &format!("field `{}` violates invariant on `{}`", key, rec.name),
            ));
        }
    }

    // 3. Check record-level `where` invariants (e.g., `start <= end`)
    for wc in &rec.where_clauses {
        let mut repl = field_var_map.clone();
        repl.insert("self".into(), "__self".into());
        stmts.push(emit_refinement_guard(
            subst_ident(wc, &repl),
            &format!("record invariant violated for `{}`", rec.name),
        ));
    }

    // 4. Construct verified record `Ok(R { f1 = __f_f1, ... })`
    let rec_lit = Expr {
        kind: ExprKind::RecordLit {
            name: rec.name.clone(),
            version: None,
            fields: rec
                .fields
                .iter()
                .map(|f| (f.name.clone(), ident(&format!("__f_{}", f.name))))
                .collect(),
        },
        span: sp(),
    };

    let method = FnDecl {
        attrs: vec![],
        is_pub: true,
        is_pure: true,
        is_async: false,
        is_test: false,
        name: "from_json".into(),
        type_params: vec![],
        type_param_bounds: BTreeMap::new(),
        params: vec![ParamDecl {
            is_inout: false,
            is_comptime: false,
            name: "v".into(),
            ty: named("JsonValue"),
            predicate: None,
            span: sp(),
        }],
        ret_ty: Some(generic("Result", vec![named(&rec.name), named("JsonError")])),
        requires: vec![],
        ensures: vec![],
        body: Block {
            stmts,
            tail: Some(Box::new(call("Ok", vec![rec_lit]))),
            span: sp(),
        },
        span: rec.span,
    };

    ImplDecl {
        attrs: vec![],
        type_params: vec![],
        type_param_bounds: BTreeMap::new(),
        interface: Some(named("FromJson")),
        target_ty: named(&rec.name),
        assoc_bindings: vec![],
        methods: vec![method],
        span: rec.span,
    }
}

fn extract_const_bounds(pred: &Expr) -> (Option<i64>, Option<i64>) {
    match &pred.kind {
        ExprKind::Binary {
            op: BinOp::And,
            left,
            right,
        } => {
            let (lmin, lmax) = extract_const_bounds(left);
            let (rmin, rmax) = extract_const_bounds(right);
            (
                match (lmin, rmin) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (a, b) => a.or(b),
                },
                match (lmax, rmax) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                },
            )
        }
        ExprKind::Binary { op, left, right } => {
            if matches!(&left.kind, ExprKind::Ident(s) if s == "self") {
                if let ExprKind::Int(n) = &right.kind {
                    return match op {
                        BinOp::Ge => (Some(*n), None),
                        BinOp::Gt => (Some(*n + 1), None),
                        BinOp::Le => (None, Some(*n)),
                        BinOp::Lt => (None, Some(*n - 1)),
                        _ => (None, None),
                    };
                }
            }
            if matches!(&right.kind, ExprKind::Ident(s) if s == "self") {
                if let ExprKind::Int(n) = &left.kind {
                    return match op {
                        BinOp::Le => (Some(*n), None),
                        BinOp::Lt => (Some(*n + 1), None),
                        BinOp::Ge => (None, Some(*n)),
                        BinOp::Gt => (None, Some(*n - 1)),
                        _ => (None, None),
                    };
                }
            }
            (None, None)
        }
        _ => (None, None),
    }
}

fn schema_for_type(
    ty: &TypeExpr,
    aliases: &BTreeMap<String, TypeAliasDecl>,
    doc: Option<&str>,
    field_pred: Option<&Expr>,
    depth: usize,
) -> Expr {
    let base = resolve_base(ty, aliases);
    let mut stmts = Vec::new();
    let var_name = format!("__sch_{}", depth);

    match base {
        TypeExpr::Named { name, args, .. } => {
            if name == "Option" && args.len() == 1 {
                return schema_for_type(&args[0], aliases, doc, field_pred, depth);
            }
            if is_int_name(name) {
                stmts.push(let_stmt(&var_name, None, call("json_obj_new", vec![]), true));
                stmts.push(assign_ident(
                    &var_name,
                    call(
                        "json_obj_set",
                        vec![ident(&var_name), str_lit("type"), call("JStr", vec![str_lit("integer")])],
                    ),
                ));
                let mut min_b = None;
                let mut max_b = None;
                if let TypeExpr::Named { name: tname, .. } = ty {
                    if let Some(al) = aliases.get(tname) {
                        if let Some(p) = &al.predicate {
                            let (a, b) = extract_const_bounds(p);
                            min_b = a;
                            max_b = b;
                        }
                    }
                }
                if let Some(fp) = field_pred {
                    let (a, b) = extract_const_bounds(fp);
                    if a.is_some() {
                        min_b = a;
                    }
                    if b.is_some() {
                        max_b = b;
                    }
                }
                if let Some(mn) = min_b {
                    stmts.push(assign_ident(
                        &var_name,
                        call(
                            "json_obj_set",
                            vec![ident(&var_name), str_lit("minimum"), call("JInt", vec![int_lit(mn)])],
                        ),
                    ));
                }
                if let Some(mx) = max_b {
                    stmts.push(assign_ident(
                        &var_name,
                        call(
                            "json_obj_set",
                            vec![ident(&var_name), str_lit("maximum"), call("JInt", vec![int_lit(mx)])],
                        ),
                    ));
                }
            } else if name == "Bool" {
                stmts.push(let_stmt(&var_name, None, call("json_obj_new", vec![]), true));
                stmts.push(assign_ident(
                    &var_name,
                    call(
                        "json_obj_set",
                        vec![ident(&var_name), str_lit("type"), call("JStr", vec![str_lit("boolean")])],
                    ),
                ));
            } else if name == "Text" {
                stmts.push(let_stmt(&var_name, None, call("json_obj_new", vec![]), true));
                stmts.push(assign_ident(
                    &var_name,
                    call(
                        "json_obj_set",
                        vec![ident(&var_name), str_lit("type"), call("JStr", vec![str_lit("string")])],
                    ),
                ));
            } else if name == "List" && args.len() == 1 {
                let item_sch = schema_for_type(&args[0], aliases, None, None, depth + 1);
                stmts.push(let_stmt(&var_name, None, call("json_obj_new", vec![]), true));
                stmts.push(assign_ident(
                    &var_name,
                    call(
                        "json_obj_set",
                        vec![ident(&var_name), str_lit("type"), call("JStr", vec![str_lit("array")])],
                    ),
                ));
                stmts.push(assign_ident(
                    &var_name,
                    call(
                        "json_obj_set",
                        vec![ident(&var_name), str_lit("items"), item_sch],
                    ),
                ));
            } else if name == "JsonValue" {
                stmts.push(let_stmt(&var_name, None, call("json_obj_new", vec![]), true));
            } else {
                // Nested record implementing JsonSchema
                stmts.push(let_stmt(
                    &var_name,
                    None,
                    method_call(ident(name), "json_schema", vec![]),
                    true,
                ));
            }
        }
        _ => {
            stmts.push(let_stmt(&var_name, None, call("json_obj_new", vec![]), true));
        }
    }

    if let Some(d) = doc {
        stmts.push(assign_ident(
            &var_name,
            call(
                "json_obj_set",
                vec![
                    ident(&var_name),
                    str_lit("description"),
                    call("JStr", vec![str_lit(d)]),
                ],
            ),
        ));
    }

    Expr {
        kind: ExprKind::Block(Block {
            stmts,
            tail: Some(Box::new(ident(&var_name))),
            span: sp(),
        }),
        span: sp(),
    }
}

fn derive_record_json_schema(
    rec: &RecordDecl,
    aliases: &BTreeMap<String, TypeAliasDecl>,
) -> ImplDecl {
    let mut stmts = vec![
        let_stmt("__props", None, call("json_obj_new", vec![]), true),
        let_stmt("__req", None, call("json_arr_new", vec![]), true),
    ];
    for (idx, f) in rec.fields.iter().enumerate() {
        if has_attr(&f.attrs, "skip") {
            continue;
        }
        let key = attr_str(&f.attrs, "rename").unwrap_or_else(|| f.name.clone());
        let doc = attr_str(&f.attrs, "doc");
        let fsch = schema_for_type(
            &f.ty,
            aliases,
            doc.as_deref(),
            f.predicate.as_ref(),
            idx + 1,
        );
        stmts.push(assign_ident(
            "__props",
            call("json_obj_set", vec![ident("__props"), str_lit(&key), fsch]),
        ));
        let base = resolve_base(&f.ty, aliases);
        let is_opt = matches!(base, TypeExpr::Named { name, args, .. } if name == "Option" && args.len() == 1);
        if !is_opt && f.default.is_none() {
            stmts.push(assign_ident(
                "__req",
                call(
                    "json_arr_push",
                    vec![ident("__req"), call("JStr", vec![str_lit(&key)])],
                ),
            ));
        }
    }
    stmts.push(let_stmt("__out", None, call("json_obj_new", vec![]), true));
    stmts.push(assign_ident(
        "__out",
        call(
            "json_obj_set",
            vec![ident("__out"), str_lit("type"), call("JStr", vec![str_lit("object")])],
        ),
    ));
    stmts.push(assign_ident(
        "__out",
        call("json_obj_set", vec![ident("__out"), str_lit("properties"), ident("__props")]),
    ));
    stmts.push(assign_ident(
        "__out",
        call("json_obj_set", vec![ident("__out"), str_lit("required"), ident("__req")]),
    ));

    let method = FnDecl {
        attrs: vec![],
        is_pub: true,
        is_pure: true,
        is_async: false,
        is_test: false,
        name: "json_schema".into(),
        type_params: vec![],
        type_param_bounds: BTreeMap::new(),
        params: vec![],
        ret_ty: Some(named("JsonValue")),
        requires: vec![],
        ensures: vec![],
        body: Block {
            stmts,
            tail: Some(Box::new(ident("__out"))),
            span: sp(),
        },
        span: rec.span,
    };

    ImplDecl {
        attrs: vec![],
        type_params: vec![],
        type_param_bounds: BTreeMap::new(),
        interface: Some(named("JsonSchema")),
        target_ty: named(&rec.name),
        assoc_bindings: vec![],
        methods: vec![method],
        span: rec.span,
    }
}

fn derive_enum_to_json(en: &EnumDecl, aliases: &BTreeMap<String, TypeAliasDecl>) -> ImplDecl {
    let tag_key = attr_str(&en.attrs, "tag").unwrap_or_else(|| "type".into());
    let mut arms = Vec::new();
    for (idx, v) in en.variants.iter().enumerate() {
        let vtag = v.name.clone();
        if v.fields.is_empty() {
            arms.push(MatchArm {
                patterns: vec![Pattern::Variant {
                    name: v.name.clone(),
                    binding: None,
                    span: sp(),
                }],
                guard: None,
                body: Expr {
                    kind: ExprKind::Block(Block {
                        stmts: vec![
                            let_stmt("__o", None, call("json_obj_new", vec![]), true),
                            assign_ident(
                                "__o",
                                call(
                                    "json_obj_set",
                                    vec![ident("__o"), str_lit(&tag_key), call("JStr", vec![str_lit(&vtag)])],
                                ),
                            ),
                        ],
                        tail: Some(Box::new(ident("__o"))),
                        span: sp(),
                    }),
                    span: sp(),
                },
                span: sp(),
            });
        } else {
            let bind = format!("__p_{}", idx);
            let enc = encode_value_expr(ident(&bind), &v.fields[0].ty, aliases, idx + 1);
            arms.push(MatchArm {
                patterns: vec![Pattern::Variant {
                    name: v.name.clone(),
                    binding: Some(bind),
                    span: sp(),
                }],
                guard: None,
                body: Expr {
                    kind: ExprKind::Block(Block {
                        stmts: vec![
                            let_stmt("__o", None, call("json_obj_new", vec![]), true),
                            assign_ident(
                                "__o",
                                call(
                                    "json_obj_set",
                                    vec![ident("__o"), str_lit(&tag_key), call("JStr", vec![str_lit(&vtag)])],
                                ),
                            ),
                            assign_ident(
                                "__o",
                                call("json_obj_set", vec![ident("__o"), str_lit("value"), enc]),
                            ),
                        ],
                        tail: Some(Box::new(ident("__o"))),
                        span: sp(),
                    }),
                    span: sp(),
                },
                span: sp(),
            });
        }
    }

    let method = FnDecl {
        attrs: vec![],
        is_pub: true,
        is_pure: true,
        is_async: false,
        is_test: false,
        name: "to_json".into(),
        type_params: vec![],
        type_param_bounds: BTreeMap::new(),
        params: vec![ParamDecl {
            is_inout: false,
            is_comptime: false,
            name: "self".into(),
            ty: named("Self"),
            predicate: None,
            span: sp(),
        }],
        ret_ty: Some(named("JsonValue")),
        requires: vec![],
        ensures: vec![],
        body: Block {
            stmts: vec![],
            tail: Some(Box::new(Expr {
                kind: ExprKind::Match {
                    scrutinee: Box::new(ident("self")),
                    arms,
                },
                span: sp(),
            })),
            span: sp(),
        },
        span: en.span,
    };

    ImplDecl {
        attrs: vec![],
        type_params: vec![],
        type_param_bounds: BTreeMap::new(),
        interface: Some(named("ToJson")),
        target_ty: named(&en.name),
        assoc_bindings: vec![],
        methods: vec![method],
        span: en.span,
    }
}

fn derive_enum_from_json(en: &EnumDecl, aliases: &BTreeMap<String, TypeAliasDecl>) -> ImplDecl {
    let tag_key = attr_str(&en.attrs, "tag").unwrap_or_else(|| "type".into());
    let mut stmts = vec![let_stmt(
        "__tag",
        None,
        Expr {
            kind: ExprKind::Match {
                scrutinee: Box::new(call("json_get_str", vec![ident("v"), str_lit(&tag_key)])),
                arms: vec![
                    MatchArm {
                        patterns: vec![Pattern::Variant {
                            name: "Some".into(),
                            binding: Some("__t".into()),
                            span: sp(),
                        }],
                        guard: None,
                        body: ident("__t"),
                        span: sp(),
                    },
                    MatchArm {
                        patterns: vec![Pattern::Variant {
                            name: "None".into(),
                            binding: None,
                            span: sp(),
                        }],
                        guard: None,
                        body: Expr {
                            kind: ExprKind::Block(Block {
                                stmts: vec![return_expr(json_err(&format!(
                                    "missing tag field `{}` for enum `{}`",
                                    tag_key, en.name
                                )))],
                                tail: None,
                                span: sp(),
                            }),
                            span: sp(),
                        },
                        span: sp(),
                    },
                ],
            },
            span: sp(),
        },
        false,
    )];

    for (idx, v) in en.variants.iter().enumerate() {
        let cond = Expr {
            kind: ExprKind::Binary {
                op: BinOp::Eq,
                left: Box::new(ident("__tag")),
                right: Box::new(str_lit(&v.name)),
            },
            span: sp(),
        };
        let then_stmts = if v.fields.is_empty() {
            vec![return_expr(call("Ok", vec![ident(&v.name)]))]
        } else {
            let jv = format!("__ev_{}", idx);
            let dec = decode_value_expr(
                ident(&jv),
                &v.fields[0].ty,
                aliases,
                &format!("{}.{}", en.name, v.name),
                idx + 1,
            );
            vec![
                let_stmt(
                    &format!("__pv_{}", idx),
                    None,
                    Expr {
                        kind: ExprKind::Match {
                            scrutinee: Box::new(call("json_get", vec![ident("v"), str_lit("value")])),
                            arms: vec![
                                MatchArm {
                                    patterns: vec![Pattern::Variant {
                                        name: "Some".into(),
                                        binding: Some(jv),
                                        span: sp(),
                                    }],
                                    guard: None,
                                    body: dec,
                                    span: sp(),
                                },
                                MatchArm {
                                    patterns: vec![Pattern::Variant {
                                        name: "None".into(),
                                        binding: None,
                                        span: sp(),
                                    }],
                                    guard: None,
                                    body: Expr {
                                        kind: ExprKind::Block(Block {
                                            stmts: vec![return_expr(json_err(&format!(
                                                "missing `value` for variant `{}.{}`",
                                                en.name, v.name
                                            )))],
                                            tail: None,
                                            span: sp(),
                                        }),
                                        span: sp(),
                                    },
                                    span: sp(),
                                },
                            ],
                        },
                        span: sp(),
                    },
                    false,
                ),
                return_expr(call(
                    "Ok",
                    vec![call(&v.name, vec![ident(&format!("__pv_{}", idx))])],
                )),
            ]
        };
        stmts.push(Stmt::Expr {
            expr: Expr {
                kind: ExprKind::If {
                    cond: Box::new(cond),
                    then_block: Block {
                        stmts: then_stmts,
                        tail: None,
                        span: sp(),
                    },
                    else_branch: None,
                },
                span: sp(),
            },
            span: sp(),
        });
    }

    let method = FnDecl {
        attrs: vec![],
        is_pub: true,
        is_pure: true,
        is_async: false,
        is_test: false,
        name: "from_json".into(),
        type_params: vec![],
        type_param_bounds: BTreeMap::new(),
        params: vec![ParamDecl {
            is_inout: false,
            is_comptime: false,
            name: "v".into(),
            ty: named("JsonValue"),
            predicate: None,
            span: sp(),
        }],
        ret_ty: Some(generic("Result", vec![named(&en.name), named("JsonError")])),
        requires: vec![],
        ensures: vec![],
        body: Block {
            stmts,
            tail: Some(Box::new(json_err(&format!(
                "unknown enum tag for `{}`",
                en.name
            )))),
            span: sp(),
        },
        span: en.span,
    };

    ImplDecl {
        attrs: vec![],
        type_params: vec![],
        type_param_bounds: BTreeMap::new(),
        interface: Some(named("FromJson")),
        target_ty: named(&en.name),
        assoc_bindings: vec![],
        methods: vec![method],
        span: en.span,
    }
}
