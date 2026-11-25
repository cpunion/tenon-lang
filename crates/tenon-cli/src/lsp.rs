//! Tenon Language Server Protocol (LSP) implementation over JSON-RPC 2.0 stdio.
//!
//! Zero external LSP dependencies — uses `tenon-syntax` (parser, project/module loader with
//! unsaved-buffer overlays, layout engine), `tenon-verify` (L0 + L1 Z3 Fail-Closed checker),
//! and `serde_json` already in the workspace.
//!
//! Supported LSP capabilities:
//! - `textDocument/didOpen`, `didChange`, `didSave`, `didClose`
//! - `textDocument/publishDiagnostics` (syntax + L0/L1 formal verification + counterexamples)
//! - `textDocument/hover` (signatures, `where`/`requires`/`ensures`, `@repr(C)`/`@packed` byte layout, proof counts)
//! - `textDocument/definition` (same-file and cross-package symbol definitions)
//! - `textDocument/completion` (keywords, primitive/SIMD types, in-scope items, `@attr`, and `.` member/package completion)
//! - `textDocument/documentSymbol` (outline hierarchy)
//! - `textDocument/formatting` (canonical `tenon fmt`)

use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tenon_syntax::{
    Attr, AttrDecl, BitfieldDecl, Block, ConstDecl, EnumDecl, Expr, ExprKind, FnDecl,
    InterfaceDecl, InterfaceMethodDecl, Item, LayoutEngine, LoadMode, MacroDecl, Program,
    RecordDecl, ScalarKind, Span, Stmt, TypeAliasDecl, TypeExpr, UnionDecl,
    load_package_with_overlays, parse_with_file_id,
};
use tenon_verify::{CheckReport, IncrementalDb, ProofTier};

pub fn run_lsp_stdio() -> ExitCode {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut reader = BufReader::new(stdin.lock());
    let mut writer = BufWriter::new(stdout.lock());
    let mut server = LspServer::new();
    let ok = server.run_loop(&mut reader, &mut writer);
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

#[derive(Debug, Clone)]
struct OpenDoc {
    path: Option<PathBuf>,
    text: String,
}

#[derive(Debug, Clone)]
struct AnalyzedDoc {
    raw_program: Option<Program>,
    full_program: Option<Program>,
    active_file_id: u32,
    report: Option<CheckReport>,
    parse_error: Option<(usize, usize, String)>,
}

struct LspServer {
    docs: BTreeMap<String, OpenDoc>,
    analysis: BTreeMap<String, AnalyzedDoc>,
    db: IncrementalDb,
    shutdown_requested: bool,
}

impl LspServer {
    fn new() -> Self {
        Self {
            docs: BTreeMap::new(),
            analysis: BTreeMap::new(),
            db: IncrementalDb::new(),
            shutdown_requested: false,
        }
    }

    fn run_loop<R: BufRead, W: Write>(&mut self, reader: &mut R, writer: &mut W) -> bool {
        loop {
            let msg = match read_lsp_message(reader) {
                Ok(Some(v)) => v,
                Ok(None) => return true,
                Err(_) => return false,
            };

            let method = msg.get("method").and_then(|m| m.as_str()).unwrap_or("");
            let id = msg.get("id").cloned();
            let params = msg.get("params").cloned().unwrap_or(Value::Null);

            match method {
                "initialize" => {
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "result": {
                            "capabilities": {
                                "textDocumentSync": {
                                    "openClose": true,
                                    "change": 1,
                                    "save": { "includeText": true }
                                },
                                "hoverProvider": true,
                                "definitionProvider": true,
                                "completionProvider": {
                                    "triggerCharacters": [".", "@", "!"]
                                },
                                "documentSymbolProvider": true,
                                "documentFormattingProvider": true,
                                "inlayHintProvider": true,
                                "codeActionProvider": true,
                                "referencesProvider": true,
                                "renameProvider": true,
                                "signatureHelpProvider": {
                                    "triggerCharacters": ["(", ","]
                                },
                                "workspaceSymbolProvider": true,
                                "codeLensProvider": {
                                    "resolveProvider": false
                                },
                                "semanticTokensProvider": {
                                    "legend": {
                                        "tokenTypes": [
                                            "keyword", "type", "function", "variable",
                                            "number", "string", "operator", "macro"
                                        ],
                                        "tokenModifiers": ["declaration", "readonly", "ghost"]
                                    },
                                    "full": true
                                }
                            },
                            "serverInfo": {
                                "name": "tenon-lsp",
                                "version": env!("CARGO_PKG_VERSION")
                            }
                        }
                    });
                    if write_lsp_message(writer, &resp).is_err() {
                        return false;
                    }
                }
                "initialized" => {}
                "shutdown" => {
                    self.shutdown_requested = true;
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "result": Value::Null
                    });
                    if write_lsp_message(writer, &resp).is_err() {
                        return false;
                    }
                }
                "exit" => {
                    return self.shutdown_requested;
                }
                "textDocument/didOpen" => {
                    if let Some(td) = params.get("textDocument") {
                        let uri = td.get("uri").and_then(|u| u.as_str()).unwrap_or("").to_string();
                        let text = td.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string();
                        let path = uri_to_path(&uri);
                        self.docs.insert(
                            uri.clone(),
                            OpenDoc { path, text },
                        );
                        self.analyze_and_publish(&uri, writer);
                    }
                }
                "textDocument/didChange" => {
                    if let Some(uri) = params
                        .get("textDocument")
                        .and_then(|t| t.get("uri"))
                        .and_then(|u| u.as_str())
                    {
                        if let Some(changes) = params.get("contentChanges").and_then(|c| c.as_array()) {
                            if let Some(last) = changes.last() {
                                if let Some(text) = last.get("text").and_then(|t| t.as_str()) {
                                    if let Some(doc) = self.docs.get_mut(uri) {
                                        doc.text = text.to_string();
                                    } else {
                                        self.docs.insert(
                                            uri.to_string(),
                                            OpenDoc {
                                                path: uri_to_path(uri),
                                                text: text.to_string(),
                                            },
                                        );
                                    }
                                    self.analyze_and_publish(uri, writer);
                                }
                            }
                        }
                    }
                }
                "textDocument/didSave" => {
                    if let Some(uri) = params
                        .get("textDocument")
                        .and_then(|t| t.get("uri"))
                        .and_then(|u| u.as_str())
                    {
                        if let Some(text) = params.get("text").and_then(|t| t.as_str()) {
                            if let Some(doc) = self.docs.get_mut(uri) {
                                doc.text = text.to_string();
                            }
                        }
                        self.analyze_and_publish(uri, writer);
                    }
                }
                "textDocument/didClose" => {
                    if let Some(uri) = params
                        .get("textDocument")
                        .and_then(|t| t.get("uri"))
                        .and_then(|u| u.as_str())
                    {
                        self.docs.remove(uri);
                        self.analysis.remove(uri);
                        let notif = json!({
                            "jsonrpc": "2.0",
                            "method": "textDocument/publishDiagnostics",
                            "params": { "uri": uri, "diagnostics": [] }
                        });
                        let _ = write_lsp_message(writer, &notif);
                    }
                }
                "textDocument/hover" => {
                    let result = self.handle_hover(&params);
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "result": result
                    });
                    if write_lsp_message(writer, &resp).is_err() {
                        return false;
                    }
                }
                "textDocument/definition" => {
                    let result = self.handle_definition(&params);
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "result": result
                    });
                    if write_lsp_message(writer, &resp).is_err() {
                        return false;
                    }
                }
                "textDocument/completion" => {
                    let result = self.handle_completion(&params);
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "result": result
                    });
                    if write_lsp_message(writer, &resp).is_err() {
                        return false;
                    }
                }
                "textDocument/documentSymbol" => {
                    let result = self.handle_document_symbol(&params);
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "result": result
                    });
                    if write_lsp_message(writer, &resp).is_err() {
                        return false;
                    }
                }
                "textDocument/formatting" => {
                    let result = self.handle_formatting(&params);
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "result": result
                    });
                    if write_lsp_message(writer, &resp).is_err() {
                        return false;
                    }
                }
                "textDocument/inlayHint" => {
                    let result = self.handle_inlay_hint(&params);
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "result": result
                    });
                    if write_lsp_message(writer, &resp).is_err() {
                        return false;
                    }
                }
                "textDocument/codeAction" => {
                    let result = self.handle_code_action(&params);
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "result": result
                    });
                    if write_lsp_message(writer, &resp).is_err() {
                        return false;
                    }
                }
                "textDocument/references" => {
                    let result = self.handle_references(&params);
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "result": result
                    });
                    if write_lsp_message(writer, &resp).is_err() {
                        return false;
                    }
                }
                "textDocument/rename" => {
                    let result = self.handle_rename(&params);
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "result": result
                    });
                    if write_lsp_message(writer, &resp).is_err() {
                        return false;
                    }
                }
                "textDocument/semanticTokens/full" => {
                    let result = self.handle_semantic_tokens(&params);
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "result": result
                    });
                    if write_lsp_message(writer, &resp).is_err() {
                        return false;
                    }
                }
                "textDocument/signatureHelp" => {
                    let result = self.handle_signature_help(&params);
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "result": result
                    });
                    if write_lsp_message(writer, &resp).is_err() {
                        return false;
                    }
                }
                "workspace/symbol" => {
                    let result = self.handle_workspace_symbol(&params);
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "result": result
                    });
                    if write_lsp_message(writer, &resp).is_err() {
                        return false;
                    }
                }
                "textDocument/codeLens" => {
                    let result = self.handle_code_lens(&params);
                    let resp = json!({
                        "jsonrpc": "2.0",
                        "id": id.unwrap_or(Value::Null),
                        "result": result
                    });
                    if write_lsp_message(writer, &resp).is_err() {
                        return false;
                    }
                }
                _ => {
                    // Respond with null/method-not-found for unknown requests that expect a response
                    if let Some(req_id) = id {
                        let resp = json!({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": Value::Null
                        });
                        if write_lsp_message(writer, &resp).is_err() {
                            return false;
                        }
                    }
                }
            }
        }
    }

    fn overlays(&self) -> BTreeMap<PathBuf, String> {
        let mut out = BTreeMap::new();
        for doc in self.docs.values() {
            if let Some(p) = &doc.path {
                out.insert(p.clone(), doc.text.clone());
            }
        }
        out
    }

    fn analyze_and_publish<W: Write>(&mut self, uri: &str, writer: &mut W) {
        let Some(doc) = self.docs.get(uri).cloned() else {
            return;
        };

        let mut analyzed = AnalyzedDoc {
            raw_program: None,
            full_program: None,
            active_file_id: 0,
            report: None,
            parse_error: None,
        };

        // 1. Always parse raw active buffer first for immediate syntax diagnostics & outline.
        match parse_with_file_id(&doc.text, 0) {
            Ok(raw) => {
                analyzed.raw_program = Some(raw);
            }
            Err(pe) => {
                analyzed.parse_error = Some((pe.span.line, pe.span.col, pe.message));
            }
        }

        // 2. If raw parse succeeded, attempt full package load (with unsaved overlays) + verification.
        if analyzed.parse_error.is_none() {
            let overlays = self.overlays();
            let loaded = if let Some(path) = &doc.path {
                if path.exists() || path.parent().is_some_and(|p| p.exists()) {
                    let mode = if tenon_syntax::is_test_source_file(path) {
                        LoadMode::Test
                    } else {
                        LoadMode::Production
                    };
                    load_package_with_overlays(path, mode, &BTreeMap::new(), &overlays)
                } else {
                    Err("virtual".into())
                }
            } else {
                Err("untitled".into())
            };

            match loaded {
                Ok(full) => {
                    let active_fid = doc
                        .path
                        .as_ref()
                        .and_then(|p| {
                            let canon = std::fs::canonicalize(p).unwrap_or_else(|_| p.clone());
                            let canon_str = canon.to_string_lossy();
                            full.source_files
                                .iter()
                                .position(|sf| sf == canon_str.as_ref())
                                .map(|idx| idx as u32)
                        })
                        .unwrap_or(0);
                    let rep = self.db.check_program_incremental(uri, &full);
                    analyzed.active_file_id = active_fid;
                    analyzed.full_program = Some(full);
                    analyzed.report = Some(rep);
                }
                Err(load_err) => {
                    // Fallback: run single-file expansion + check on raw_program
                    if let Some(mut single) = analyzed.raw_program.clone() {
                        tenon_syntax::eval_cfg_attrs(&mut single);
                        if let Err(e) = tenon_syntax::expand_comptime_and_macros(&mut single) {
                            analyzed.parse_error = Some((1, 1, e));
                        } else {
                            tenon_syntax::expand_derives(&mut single);
                            let rep = self.db.check_program_incremental(uri, &single);
                            analyzed.active_file_id = 0;
                            analyzed.full_program = Some(single);
                            analyzed.report = Some(rep);
                        }
                    } else if load_err != "virtual" && load_err != "untitled" {
                        analyzed.parse_error = Some((1, 1, load_err));
                    }
                }
            }
        }

        // 3. Build LSP Diagnostic[] payload for this document
        let mut lsp_diags: Vec<Value> = Vec::new();
        if let Some((line, col, msg)) = &analyzed.parse_error {
            let l0 = line.saturating_sub(1);
            let c0 = col.saturating_sub(1);
            lsp_diags.push(json!({
                "range": {
                    "start": { "line": l0, "character": c0 },
                    "end": { "line": l0, "character": c0 + 1 }
                },
                "severity": 1,
                "code": "E0001",
                "source": "tenon",
                "message": msg
            }));
        } else if let Some(rep) = &analyzed.report {
            for d in &rep.diagnostics {
                // Only attach diagnostics belonging to this file (or fallback file_id 0 when active_file_id == 0)
                if analyzed.active_file_id != 0 && d.span.file_id != 0 && d.span.file_id != analyzed.active_file_id {
                    continue;
                }
                let l0 = d.span.line.saturating_sub(1);
                let c0 = d.span.col.saturating_sub(1);
                let len = if d.span.end > d.span.start {
                    (d.span.end - d.span.start).min(80)
                } else {
                    1
                };
                let mut full_msg = d.message.clone();
                for note in &d.notes {
                    full_msg.push_str(&format!("\nnote: {}", note));
                }
                if !d.counterexample.is_empty() {
                    let ce: Vec<String> = d
                        .counterexample
                        .iter()
                        .map(|(k, v)| format!("{} = {}", k, v))
                        .collect();
                    full_msg.push_str(&format!("\nZ3 counterexample: {}", ce.join(", ")));
                }
                lsp_diags.push(json!({
                    "range": {
                        "start": { "line": l0, "character": c0 },
                        "end": { "line": l0, "character": c0 + len }
                    },
                    "severity": 1,
                    "code": d.code,
                    "source": "tenon-verify",
                    "message": full_msg
                }));
            }
        }

        self.analysis.insert(uri.to_string(), analyzed);

        let notif = json!({
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": uri,
                "diagnostics": lsp_diags
            }
        });
        let _ = write_lsp_message(writer, &notif);
    }

    fn handle_hover(&self, params: &Value) -> Value {
        let Some((uri, line, ch)) = extract_text_doc_pos(params) else {
            return Value::Null;
        };
        let Some(doc) = self.docs.get(&uri) else {
            return Value::Null;
        };
        let Some(tok) = word_at_position(&doc.text, line, ch) else {
            return Value::Null;
        };
        if tok.is_string_lit {
            return Value::Null;
        }
        let analyzed = self.analysis.get(&uri);
        let prog = analyzed.and_then(|a| a.full_program.as_ref().or(a.raw_program.as_ref()));
        let report = analyzed.and_then(|a| a.report.as_ref());

        if tok.is_attr {
            if let Some(prog) = prog {
                if let Some(ad) = find_attr_decl(prog, &tok.word) {
                    let md = format_attr_decl_hover(ad);
                    return json!({ "contents": { "kind": "markdown", "value": md } });
                }
            }
            if let Some(md) = attribute_hover(&tok.word) {
                return json!({
                    "contents": {
                        "kind": "markdown",
                        "value": md
                    }
                });
            }
            return json!({
                "contents": {
                    "kind": "markdown",
                    "value": format!("```tenon\n@{}\n```\n\nCompile-time metadata attribute.", tok.word)
                }
            });
        }

        if let Some(prog) = prog {
            let layout = LayoutEngine::from_program(prog);
            if let Some(pref) = &tok.qual_prefix {
                // 1. Qualified package symbol (`math.add` -> `math__add`)
                let mangled = format!("{}__{}", pref, tok.word);
                if let Some(md) = format_item_hover(prog, &layout, report, &mangled) {
                    return json!({ "contents": { "kind": "markdown", "value": md } });
                }
                // 2. Receiver type member (`ctx.json_raw`, `conn.get`, `HttpResponse.text`)
                let recv_ty = infer_receiver_type(&doc.text, pref, prog);
                if let Some(rty) = &recv_ty {
                    if let Some(md) = format_member_hover(prog, rty, &tok.word, report) {
                        return json!({ "contents": { "kind": "markdown", "value": md } });
                    }
                }
                // 3. Fallback: package symbol without prefix mangling, or any impl method matching name
                if let Some(md) = format_item_hover(prog, &layout, report, &tok.word) {
                    return json!({ "contents": { "kind": "markdown", "value": md } });
                }
                for item in &prog.items {
                    if let Item::Impl(im) = item {
                        let target_name = im.target_ty.display_name();
                        for m in &im.methods {
                            if m.name == tok.word {
                                let md = format_fn_hover(m, Some(&target_name), report);
                                return json!({ "contents": { "kind": "markdown", "value": md } });
                            }
                        }
                    }
                }
            } else {
                // Unqualified symbol: only match top-level items (never unrelated impl methods)
                if let Some(md) = format_item_hover(prog, &layout, report, &tok.word) {
                    return json!({ "contents": { "kind": "markdown", "value": md } });
                }
            }
        }

        // Subtype-dependent `refine_cast[T](x)` hover with statically resolved return type
        if tok.qual_prefix.is_none() && tok.word == "refine_cast" {
            if let Some(rep) = report {
                let line_start: usize = doc
                    .text
                    .split_inclusive('\n')
                    .take(line as usize)
                    .map(|s| s.len())
                    .sum();
                let line_end = line_start
                    + doc.text[line_start..]
                        .split('\n')
                        .next()
                        .map(|s| s.len())
                        .unwrap_or(0);
                let cursor_off = line_start + ch as usize;
                if let Some((_, resolved_ty)) = rep
                    .refine_cast_types
                    .iter()
                    .filter(|(off, _)| **off >= line_start && **off <= line_end)
                    .min_by_key(|(off, _)| (**off as isize - cursor_off as isize).abs())
                {
                    let md = format!(
                        "```tenon\nrefine_cast -> {}\n```\n\nSubtype-dependent refinement cast statically resolved by `tenon-verify` (L0 + Z3 SMT).",
                        resolved_ty
                    );
                    return json!({ "contents": { "kind": "markdown", "value": md } });
                }
            }
        }

        // Builtin / primitive hover fallback
        if tok.qual_prefix.is_none() {
            if let Some(md) = builtin_hover(&tok.word) {
                return json!({
                    "contents": {
                        "kind": "markdown",
                        "value": md
                    }
                });
            }
        }

        Value::Null
    }

    fn handle_definition(&self, params: &Value) -> Value {
        let Some((uri, line, ch)) = extract_text_doc_pos(params) else {
            return Value::Null;
        };
        let Some(doc) = self.docs.get(&uri) else {
            return Value::Null;
        };
        let Some(tok) = word_at_position(&doc.text, line, ch) else {
            return Value::Null;
        };
        if tok.is_string_lit {
            return Value::Null;
        }
        let Some(analyzed) = self.analysis.get(&uri) else {
            return Value::Null;
        };
        let Some(prog) = analyzed.full_program.as_ref().or(analyzed.raw_program.as_ref()) else {
            return Value::Null;
        };

        // Declared attributes (`pub attr get(path Text)`) jump straight to their `attr` declaration!
        // Built-in compiler directives (`@repr(C)`, `@packed`, `@derive`) return null (no source file).
        if tok.is_attr {
            if let Some(ad) = find_attr_decl(prog, &tok.word) {
                let target_uri = span_file_uri(prog, ad.span, &uri);
                let l0 = ad.span.line.saturating_sub(1);
                let c0 = ad.span.col.saturating_sub(1);
                return json!({
                    "uri": target_uri,
                    "range": {
                        "start": { "line": l0, "character": c0 },
                        "end": { "line": l0, "character": c0 + tok.word.len() }
                    }
                });
            }
            return Value::Null;
        }

        let found_span = if let Some(pref) = &tok.qual_prefix {
            let mangled = format!("{}__{}", pref, tok.word);
            let recv_ty = infer_receiver_type(&doc.text, pref, prog);
            find_top_level_item_span(prog, &mangled)
                .or_else(|| {
                    recv_ty
                        .as_deref()
                        .and_then(|rty| find_member_span(prog, rty, &tok.word))
                })
                .or_else(|| find_any_impl_method_span(prog, &tok.word))
                .or_else(|| find_top_level_item_span(prog, &tok.word))
        } else {
            // Unqualified symbol: only jump to top-level declarations (never unrelated impl methods)
            find_top_level_item_span(prog, &tok.word)
        };

        if let Some(span) = found_span {
            let target_uri = span_file_uri(prog, span, &uri);
            let l0 = span.line.saturating_sub(1);
            let c0 = span.col.saturating_sub(1);
            return json!({
                "uri": target_uri,
                "range": {
                    "start": { "line": l0, "character": c0 },
                    "end": { "line": l0, "character": c0 + tok.word.len() }
                }
            });
        }

        Value::Null
    }

    fn handle_completion(&self, params: &Value) -> Value {
        let Some((uri, line, ch)) = extract_text_doc_pos(params) else {
            return json!([]);
        };
        let Some(doc) = self.docs.get(&uri) else {
            return json!([]);
        };

        let analyzed = self.analysis.get(&uri);
        let prog = analyzed.and_then(|a| a.full_program.as_ref().or(a.raw_program.as_ref()));
        let raw_prog = analyzed.and_then(|a| a.raw_program.as_ref());

        let line_prefix = line_prefix_before_cursor(&doc.text, line, ch);
        let mut items: Vec<Value> = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();

        // 1. Attribute completion after `@`
        if line_prefix.trim_end().ends_with('@')
            || line_prefix
                .rsplit_once('@')
                .is_some_and(|(_, tail)| !tail.is_empty() && tail.chars().all(|c| c.is_alphanumeric() || c == '_'))
        {
            if let Some(prog) = prog {
                for it in &prog.items {
                    if let Item::AttrDecl(a) = it {
                        if seen.insert(a.name.clone()) {
                            items.push(json!({
                                "label": a.name,
                                "kind": 14,
                                "detail": format_attr_decl_sig(a)
                            }));
                        }
                    }
                }
            }
            for (attr, detail) in [
                ("repr(C)", "C-ABI compatible struct layout"),
                ("packed", "Unaligned packed struct layout (1-byte alignment)"),
                ("align(16)", "Explicit minimum byte alignment"),
                ("export", "Export symbol with C ABI for static/shared library FFI"),
                ("link(\"m\")", "Link external C/native library"),
                ("cfg(os = \"macos\")", "Conditional compilation attribute"),
                ("derive(ToJson, FromJson)", "Derive JSON serialization/deserialization"),
                ("evolve(\"entity.Name\")", "Versioned schema evolution entity"),
                ("field(1)", "Stable schema evolution field ID"),
            ] {
                let key = attr.split('(').next().unwrap_or(attr);
                if seen.insert(key.to_string()) {
                    items.push(json!({
                        "label": attr,
                        "kind": 14,
                        "detail": detail
                    }));
                }
            }
            return json!(items);
        }

        // 2. Member / package completion after `.`
        if let Some(dot_idx) = line_prefix.rfind('.') {
            let before_dot = &line_prefix[..dot_idx];
            let receiver_ident: String = before_dot
                .chars()
                .rev()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>()
                .chars()
                .rev()
                .collect();

            if !receiver_ident.is_empty() {
                if let Some(prog) = prog {
                    // 2a. Check if `receiver_ident` is a package namespace alias from raw imports
                    let mut pkg_aliases = BTreeSet::new();
                    if let Some(rp) = raw_prog {
                        for it in &rp.items {
                            if let Item::Import { path, alias, kind, .. } = it {
                                let ns = alias.clone().unwrap_or_else(|| {
                                    if let Some(tenon_syntax::ImportKind::Package(segs)) = kind {
                                        segs.last().cloned().unwrap_or_default()
                                    } else {
                                        path.rsplit('.').next().unwrap_or("").to_string()
                                    }
                                });
                                if !ns.is_empty() {
                                    pkg_aliases.insert(ns);
                                }
                            }
                        }
                    }

                    if pkg_aliases.contains(&receiver_ident) {
                        let prefix_mangled = format!("{}__", receiver_ident);
                        for it in &prog.items {
                            if let Some((name, kind_num, detail)) = item_completion_info(it) {
                                if let Some(short) = name.strip_prefix(&prefix_mangled) {
                                    if seen.insert(short.to_string()) {
                                        items.push(json!({
                                            "label": short,
                                            "kind": kind_num,
                                            "detail": detail
                                        }));
                                    }
                                } else if is_item_pub(it) && seen.insert(name.clone()) {
                                    items.push(json!({
                                        "label": name,
                                        "kind": kind_num,
                                        "detail": detail
                                    }));
                                }
                            }
                        }
                    }

                    // 2b. Check if `receiver_ident` is a type name (enum variants / assoc methods)
                    for it in &prog.items {
                        match it {
                            Item::Enum(e) if e.name == receiver_ident => {
                                for v in &e.variants {
                                    if seen.insert(v.name.clone()) {
                                        items.push(json!({
                                            "label": v.name,
                                            "kind": 20,
                                            "detail": format!("enum variant of {}", e.name)
                                        }));
                                    }
                                }
                            }
                            Item::Impl(im) if im.target_ty.display_name() == receiver_ident => {
                                for m in &im.methods {
                                    if seen.insert(m.name.clone()) {
                                        items.push(json!({
                                            "label": m.name,
                                            "kind": 2,
                                            "detail": format_fn_sig(m)
                                        }));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }

                    // 2c. Infer local variable type from text (e.g. `ctx Context`, `let u = User { ... }`)
                    if let Some(var_ty) = guess_var_type_in_source(&doc.text, &receiver_ident) {
                        for it in &prog.items {
                            match it {
                                Item::Record(r) if r.name == var_ty => {
                                    for f in &r.fields {
                                        if seen.insert(f.name.clone()) {
                                            items.push(json!({
                                                "label": f.name,
                                                "kind": 5,
                                                "detail": f.ty.display_name()
                                            }));
                                        }
                                    }
                                }
                                Item::Bitfield(b) if b.name == var_ty => {
                                    for f in &b.fields {
                                        if seen.insert(f.name.clone()) {
                                            items.push(json!({
                                                "label": f.name,
                                                "kind": 5,
                                                "detail": format!("{} bit(s)", f.bits)
                                            }));
                                        }
                                    }
                                    if seen.insert("raw".into()) {
                                        items.push(json!({
                                            "label": "raw",
                                            "kind": 2,
                                            "detail": format!("fn raw(self) {}", b.backing_ty.display_name())
                                        }));
                                    }
                                }
                                Item::Union(u) if u.name == var_ty => {
                                    for f in &u.fields {
                                        if seen.insert(f.name.clone()) {
                                            items.push(json!({
                                                "label": f.name,
                                                "kind": 5,
                                                "detail": f.ty.display_name()
                                            }));
                                        }
                                    }
                                }
                                Item::Impl(im) if im.target_ty.display_name() == var_ty => {
                                    for m in &im.methods {
                                        if seen.insert(m.name.clone()) {
                                            items.push(json!({
                                                "label": m.name,
                                                "kind": 2,
                                                "detail": format_fn_sig(m)
                                            }));
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                return json!(items);
            }
        }

        // 3. General identifier completion: in-scope items + keywords + primitive/SIMD types
        if let Some(prog) = prog {
            for it in &prog.items {
                if let Some((name, kind_num, detail)) = item_completion_info(it) {
                    if !name.starts_with("__") && seen.insert(name.clone()) {
                        items.push(json!({
                            "label": name,
                            "kind": kind_num,
                            "detail": detail
                        }));
                    }
                }
            }
        }

        for kw in [
            "fn", "pure", "comptime", "macro", "record", "enum", "interface", "impl", "type",
            "const", "union", "bitfield", "import", "pub", "unsafe", "defer", "test", "where",
            "requires", "ensures", "match", "if", "else", "while", "for", "in", "return", "let",
            "var", "inout", "dyn", "async", "await", "extern", "true", "false",
        ] {
            if seen.insert(kw.into()) {
                items.push(json!({ "label": kw, "kind": 14, "detail": "keyword" }));
            }
        }

        for ty in [
            "Int", "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64",
            "ISize", "USize", "Float", "Float16", "BFloat16", "Float32", "Float64", "Bool", "Text", "Bytes", "BytesSlice",
            "List", "Map", "Option", "Result", "Store", "Channel", "Mutex", "AtomicInt", "Ptr",
            "MutPtr", "Float16x8", "Float32x4", "Float64x4", "Int32x4", "Int64x4", "UInt8x16",
            "Tensor", "Matrix", "Tensor3D", "Tensor4D", "Shape",
        ] {
            if seen.insert(ty.into()) {
                items.push(json!({ "label": ty, "kind": 22, "detail": "built-in type" }));
            }
        }

        json!(items)
    }

    fn handle_document_symbol(&self, params: &Value) -> Value {
        let Some(uri) = params
            .get("textDocument")
            .and_then(|t| t.get("uri"))
            .and_then(|u| u.as_str())
        else {
            return json!([]);
        };
        let Some(analyzed) = self.analysis.get(uri) else {
            return json!([]);
        };
        let Some(prog) = analyzed.raw_program.as_ref().or(analyzed.full_program.as_ref()) else {
            return json!([]);
        };

        let mut syms = Vec::new();
        for item in &prog.items {
            match item {
                Item::Fn(f) => {
                    syms.push(make_doc_symbol(&f.name, 12, f.span, format_fn_sig(f), vec![]));
                }
                Item::ExternFn(ef) => {
                    syms.push(make_doc_symbol(
                        &ef.name,
                        12,
                        ef.span,
                        format!("extern \"C\" fn {}", ef.name),
                        vec![],
                    ));
                }
                Item::Record(r) => {
                    let children: Vec<Value> = r
                        .fields
                        .iter()
                        .map(|f| make_doc_symbol(&f.name, 8, f.span, f.ty.display_name(), vec![]))
                        .collect();
                    syms.push(make_doc_symbol(&r.name, 23, r.span, "record".into(), children));
                }
                Item::Union(u) => {
                    let children: Vec<Value> = u
                        .fields
                        .iter()
                        .map(|f| make_doc_symbol(&f.name, 8, f.span, f.ty.display_name(), vec![]))
                        .collect();
                    syms.push(make_doc_symbol(&u.name, 23, u.span, "union".into(), children));
                }
                Item::Bitfield(b) => {
                    let children: Vec<Value> = b
                        .fields
                        .iter()
                        .map(|f| {
                            make_doc_symbol(&f.name, 8, f.span, format!("{} bits", f.bits), vec![])
                        })
                        .collect();
                    syms.push(make_doc_symbol(
                        &b.name,
                        23,
                        b.span,
                        format!("bitfield({})", b.backing_ty.display_name()),
                        children,
                    ));
                }
                Item::Enum(e) => {
                    let children: Vec<Value> = e
                        .variants
                        .iter()
                        .map(|v| make_doc_symbol(&v.name, 22, v.span, "variant".into(), vec![]))
                        .collect();
                    syms.push(make_doc_symbol(&e.name, 10, e.span, "enum".into(), children));
                }
                Item::Interface(i) => {
                    let children: Vec<Value> = i
                        .methods
                        .iter()
                        .map(|m| make_doc_symbol(&m.name, 6, m.span, format_iface_method_sig(m), vec![]))
                        .collect();
                    syms.push(make_doc_symbol(&i.name, 11, i.span, "interface".into(), children));
                }
                Item::Impl(im) => {
                    let label = if let Some(iface) = &im.interface {
                        format!("impl {} for {}", iface.display_name(), im.target_ty.display_name())
                    } else {
                        format!("impl {}", im.target_ty.display_name())
                    };
                    let children: Vec<Value> = im
                        .methods
                        .iter()
                        .map(|m| make_doc_symbol(&m.name, 6, m.span, format_fn_sig(m), vec![]))
                        .collect();
                    syms.push(make_doc_symbol(&label, 5, im.span, "impl".into(), children));
                }
                Item::TypeAlias(ta) => {
                    syms.push(make_doc_symbol(
                        &ta.name,
                        26,
                        ta.span,
                        format!("type = {}", ta.base.display_name()),
                        vec![],
                    ));
                }
                Item::Const(c) => {
                    syms.push(make_doc_symbol(
                        &c.name,
                        14,
                        c.span,
                        c.ty.display_name(),
                        vec![],
                    ));
                }
                Item::Macro(m) => {
                    syms.push(make_doc_symbol(&m.name, 12, m.span, "macro".into(), vec![]));
                }
                Item::AttrDecl(a) => {
                    syms.push(make_doc_symbol(
                        &format!("@{}", a.name),
                        14,
                        a.span,
                        format_attr_decl_sig(a),
                        vec![],
                    ));
                }
                _ => {}
            }
        }
        json!(syms)
    }

    fn handle_formatting(&self, params: &Value) -> Value {
        let Some(uri) = params
            .get("textDocument")
            .and_then(|t| t.get("uri"))
            .and_then(|u| u.as_str())
        else {
            return json!([]);
        };
        let Some(doc) = self.docs.get(uri) else {
            return json!([]);
        };
        let formatted = crate::format_tenon_source(&doc.text);
        if formatted == doc.text {
            return json!([]);
        }
        let line_count = doc.text.lines().count() as u32;
        json!([
            {
                "range": {
                    "start": { "line": 0, "character": 0 },
                    "end": { "line": line_count + 1, "character": 0 }
                },
                "newText": formatted
            }
        ])
    }

    fn handle_inlay_hint(&self, params: &Value) -> Value {
        let Some(uri) = params
            .get("textDocument")
            .and_then(|t| t.get("uri"))
            .and_then(|u| u.as_str())
        else {
            return json!([]);
        };
        let Some(doc) = self.docs.get(uri) else {
            return json!([]);
        };
        let Some(analyzed) = self.analysis.get(uri) else {
            return json!([]);
        };
        let Some(rep) = analyzed.report.as_ref() else {
            return json!([]);
        };
        let mut hints: Vec<Value> = Vec::new();
        let lines: Vec<&str> = doc.text.lines().collect();

        // 1. Subtype-dependent `refine_cast[T](x)` resolved return type hints (`-> Range[2, 6]` vs `-> Result[Range[2, 4], Text]`)
        for (&byte_off, resolved_ty) in &rep.mir.refine_cast_types {
            let (l, _) = offset_to_line_col(&doc.text, byte_off);
            if let Some(line_str) = lines.get(l as usize) {
                let end_col = line_str.len() as u32;
                hints.push(json!({
                    "position": { "line": l, "character": end_col },
                    "label": format!(" -> {}", resolved_ty),
                    "kind": 1,
                    "paddingLeft": true,
                    "tooltip": format!("Subtype-dependent `refine_cast` statically resolved to `{}`", resolved_ty)
                }));
            }
        }

        // 2. Per-function static verification badges (`✓ L0/Z3`, `⚡ bounds`, `⚡ @fip`, `⚡ no-ov`)
        if let Some(prog) = analyzed.raw_program.as_ref() {
            for item in &prog.items {
                if let Item::Fn(f) = item {
                    let fn_line = f.span.line.saturating_sub(1) as u32;
                    let fn_l0 = rep
                        .proofs
                        .iter()
                        .filter(|p| {
                            p.span.start >= f.span.start
                                && p.span.end <= f.span.end
                                && p.tier == ProofTier::L0Fast
                        })
                        .count();
                    let fn_l1 = rep
                        .proofs
                        .iter()
                        .filter(|p| {
                            p.span.start >= f.span.start
                                && p.span.end <= f.span.end
                                && p.tier == ProofTier::L1Smt
                        })
                        .count();
                    let fn_bounds = rep
                        .mir
                        .proved_safe_indices
                        .iter()
                        .filter(|&&off| off >= f.span.start && off <= f.span.end)
                        .count();
                    let fn_no_ov = rep
                        .mir
                        .proved_no_overflow
                        .iter()
                        .filter(|&&off| off >= f.span.start && off <= f.span.end)
                        .count();
                    let is_fip = rep.mir.fip_verified_fns.contains(&f.name);
                    if fn_l0 + fn_l1 + fn_bounds + fn_no_ov > 0 || is_fip {
                        let mut parts = Vec::new();
                        if fn_l0 + fn_l1 > 0 {
                            parts.push(format!("✓ {} L0 / {} Z3", fn_l0, fn_l1));
                        }
                        if fn_bounds > 0 {
                            parts.push(format!("⚡ {} bounds", fn_bounds));
                        }
                        if fn_no_ov > 0 {
                            parts.push(format!("⚡ {} nsw", fn_no_ov));
                        }
                        if is_fip {
                            parts.push("⚡ @fip".into());
                        }
                        if let Some(line_str) = lines.get(fn_line as usize) {
                            hints.push(json!({
                                "position": { "line": fn_line, "character": line_str.len() as u32 },
                                "label": format!("  [{}]", parts.join(" · ")),
                                "kind": 2,
                                "paddingLeft": true
                            }));
                        }
                    }
                    // 3. LSP 2.0 Tensor/Matrix static shape & compact byte-size inlay hints!
                    //    e.g. `Tensor[1, 1, 2, 3, 4, DType = Float16]` -> ` // rank=5, numel=24, 48B Float16`
                    fn collect_tensor_hints(
                        blk: &Block,
                        lines: &[&str],
                        hints: &mut Vec<Value>,
                    ) {
                        for stmt in &blk.stmts {
                            if let Stmt::Let { ty: Some(t), span, .. } = stmt {
                                if let Some(summary) = format_tensor_shape_hint(t) {
                                    let ln = span.line.saturating_sub(1) as u32;
                                    if let Some(line_str) = lines.get(ln as usize) {
                                        hints.push(json!({
                                            "position": { "line": ln, "character": line_str.len() as u32 },
                                            "label": format!("  // {}", summary),
                                            "kind": 1,
                                            "paddingLeft": true,
                                            "tooltip": format!("Static dependent-shape tensor layout: {}", summary)
                                        }));
                                    }
                                }
                            }
                            if let Stmt::While { body, .. } | Stmt::For { body, .. } = stmt {
                                collect_tensor_hints(body, lines, hints);
                            }
                        }
                    }
                    collect_tensor_hints(&f.body, &lines, &mut hints);
                }
            }
        }
        json!(hints)
    }

    fn handle_code_action(&self, params: &Value) -> Value {
        let Some(uri) = params
            .get("textDocument")
            .and_then(|t| t.get("uri"))
            .and_then(|u| u.as_str())
        else {
            return json!([]);
        };
        let Some(doc) = self.docs.get(uri) else {
            return json!([]);
        };
        let Some(rep) = self.analysis.get(uri).and_then(|a| a.report.as_ref()) else {
            return json!([]);
        };
        let req_start_line = params
            .pointer("/range/start/line")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let req_end_line = params
            .pointer("/range/end/line")
            .and_then(|v| v.as_u64())
            .unwrap_or(u64::MAX) as usize;
        let lines: Vec<&str> = doc.text.lines().collect();
        let mut actions: Vec<Value> = Vec::new();

        for d in &rep.diagnostics {
            let dl = d.span.line.saturating_sub(1);
            if dl < req_start_line || dl > req_end_line {
                continue;
            }
            let Some(&line_str) = lines.get(dl) else {
                continue;
            };
            // Quick Fix 1: `E0410` narrowing cast `expr as T` -> `expr as? T` or `refine_cast[T](expr)`
            if d.code == "E0410" {
                if let Some(as_pos) = line_str.find(" as ") {
                    let fixed_try = format!(
                        "{} as? {}",
                        &line_str[..as_pos],
                        &line_str[as_pos + 4..]
                    );
                    actions.push(json!({
                        "title": "Convert narrowing cast to fallible `as?` (returns Result)",
                        "kind": "quickfix",
                        "isPreferred": true,
                        "edit": {
                            "changes": {
                                uri: [{
                                    "range": {
                                        "start": { "line": dl, "character": 0 },
                                        "end": { "line": dl, "character": line_str.len() }
                                    },
                                    "newText": fixed_try
                                }]
                            }
                        }
                    }));
                    let rhs_ty = line_str[as_pos + 4..].trim();
                    let lhs_part = &line_str[..as_pos];
                    let (prefix, expr_str) = if let Some(eq_idx) = lhs_part.rfind('=') {
                        (&lhs_part[..=eq_idx], lhs_part[eq_idx + 1..].trim())
                    } else {
                        ("", lhs_part.trim())
                    };
                    let fixed_rc = if prefix.is_empty() {
                        format!(
                            "{}refine_cast[{}]({})",
                            &line_str[..line_str.len() - line_str.trim_start().len()],
                            rhs_ty,
                            expr_str
                        )
                    } else {
                        format!("{} refine_cast[{}]({})", prefix, rhs_ty, expr_str)
                    };
                    actions.push(json!({
                        "title": format!("Convert to subtype-dependent `refine_cast[{}]({})`", rhs_ty, expr_str),
                        "kind": "quickfix",
                        "edit": {
                            "changes": {
                                uri: [{
                                    "range": {
                                        "start": { "line": dl, "character": 0 },
                                        "end": { "line": dl, "character": line_str.len() }
                                    },
                                    "newText": fixed_rc
                                }]
                            }
                        }
                    }));
                }
            }
            // Quick Fix 2: `E0420` direct field mutation on invariant record -> wrap in `mutate <var> { ... }`
            if d.code == "E0420" {
                let indent_len = line_str.len() - line_str.trim_start().len();
                let indent = &line_str[..indent_len];
                let trimmed = line_str.trim();
                if let Some(dot_idx) = trimmed.find('.') {
                    let var_name = trimmed[..dot_idx].trim();
                    if !var_name.is_empty() {
                        let wrapped = format!(
                            "{}mutate {} {{\n{}    {}\n{}}}",
                            indent, var_name, indent, trimmed, indent
                        );
                        actions.push(json!({
                            "title": format!("Wrap invariant mutation in `mutate {} {{ ... }}`", var_name),
                            "kind": "quickfix",
                            "isPreferred": true,
                            "edit": {
                                "changes": {
                                    uri: [{
                                        "range": {
                                            "start": { "line": dl, "character": 0 },
                                            "end": { "line": dl, "character": line_str.len() }
                                        },
                                        "newText": wrapped
                                    }]
                                }
                            }
                        }));
                    }
                }
            }
            // Quick Fix 3: Counterexample-to-Test — synthesize a runnable `test fn` from Z3 counterexample
            if !d.counterexample.is_empty() {
                let prog = self
                    .analysis
                    .get(uri)
                    .and_then(|a| a.full_program.as_ref().or(a.raw_program.as_ref()));
                let repro_src = tenon_verify::emit_repro_test_from_diagnostics(
                    uri,
                    prog,
                    std::slice::from_ref(d),
                );
                let end_line = lines.len();
                let insert_text = format!("\n{}", repro_src);
                actions.push(json!({
                    "title": format!("Generate Z3 counterexample unit test for `{}`", d.code),
                    "kind": "quickfix",
                    "edit": {
                        "changes": {
                            uri: [{
                                "range": {
                                    "start": { "line": end_line, "character": 0 },
                                    "end": { "line": end_line, "character": 0 }
                                },
                                "newText": insert_text
                            }]
                        }
                    }
                }));
            }
        }
        json!(actions)
    }

    fn handle_references(&self, params: &Value) -> Value {
        let Some((uri, line, ch)) = extract_text_doc_pos(params) else {
            return json!([]);
        };
        let Some(doc) = self.docs.get(&uri) else {
            return json!([]);
        };
        let Some(tok) = word_at_position(&doc.text, line, ch) else {
            return json!([]);
        };
        let mut locs: Vec<Value> = Vec::new();
        for (doc_uri, open) in &self.docs {
            for (l, start_col, end_col) in find_word_occurrences(&open.text, &tok.word) {
                locs.push(json!({
                    "uri": doc_uri,
                    "range": {
                        "start": { "line": l, "character": start_col },
                        "end": { "line": l, "character": end_col }
                    }
                }));
            }
        }
        json!(locs)
    }

    fn handle_rename(&self, params: &Value) -> Value {
        let Some((uri, line, ch)) = extract_text_doc_pos(params) else {
            return Value::Null;
        };
        let Some(new_name) = params.get("newName").and_then(|v| v.as_str()) else {
            return Value::Null;
        };
        let Some(doc) = self.docs.get(&uri) else {
            return Value::Null;
        };
        let Some(tok) = word_at_position(&doc.text, line, ch) else {
            return Value::Null;
        };
        let mut changes = serde_json::Map::new();
        for (doc_uri, open) in &self.docs {
            let edits: Vec<Value> = find_word_occurrences(&open.text, &tok.word)
                .into_iter()
                .map(|(l, start_col, end_col)| {
                    json!({
                        "range": {
                            "start": { "line": l, "character": start_col },
                            "end": { "line": l, "character": end_col }
                        },
                        "newText": new_name
                    })
                })
                .collect();
            if !edits.is_empty() {
                changes.insert(doc_uri.clone(), Value::Array(edits));
            }
        }
        json!({ "changes": Value::Object(changes) })
    }

    fn handle_signature_help(&self, params: &Value) -> Value {
        let Some(uri) = params.get("textDocument").and_then(|t| t.get("uri")).and_then(|u| u.as_str()) else {
            return Value::Null;
        };
        let Some(doc) = self.docs.get(uri) else {
            return Value::Null;
        };
        let line = params.get("position").and_then(|p| p.get("line")).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let ch = params.get("position").and_then(|p| p.get("character")).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let line_str = doc.text.lines().nth(line).unwrap_or("");
        let prefix: String = line_str.chars().take(ch).collect();

        // Walk backward from cursor to find the enclosing unclosed '(' and count commas at depth 0
        let bytes = prefix.as_bytes();
        let mut depth = 0i32;
        let mut active_param = 0usize;
        let mut open_paren = None;
        for idx in (0..bytes.len()).rev() {
            match bytes[idx] {
                b')' => depth += 1,
                b'(' => {
                    if depth == 0 {
                        open_paren = Some(idx);
                        break;
                    }
                    depth -= 1;
                }
                b',' if depth == 0 => active_param += 1,
                _ => {}
            }
        }
        let Some(lp) = open_paren else {
            return Value::Null;
        };
        let before = prefix[..lp].trim_end();
        let callee: String = before
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        if callee.is_empty() {
            return Value::Null;
        }
        let Some(an) = self.analysis.get(uri) else {
            return Value::Null;
        };
        let Some(prog) = an.full_program.as_ref().or(an.raw_program.as_ref()) else {
            return Value::Null;
        };

        let mut sigs = Vec::new();
        for item in &prog.items {
            if let Item::Fn(f) = item {
                if f.name == callee {
                    let param_labels: Vec<String> = f
                        .params
                        .iter()
                        .map(|p| {
                            let io = if p.is_inout { "inout " } else { "" };
                            format!("{}{} {}", io, p.name, p.ty.display_name())
                        })
                        .collect();
                    let ret = f
                        .ret_ty
                        .as_ref()
                        .map(|r| format!(" {}", r.display_name()))
                        .unwrap_or_default();
                    let label = format!("fn {}({}){}", f.name, param_labels.join(", "), ret);
                    let mut doc_lines = Vec::new();
                    for req in &f.requires {
                        doc_lines.push(format!("`requires` contract verified statically (line {})", req.span.line));
                    }
                    for ens in &f.ensures {
                        doc_lines.push(format!("`ensures` postcondition verified (line {})", ens.span.line));
                    }
                    let param_objs: Vec<Value> = param_labels
                        .iter()
                        .map(|pl| json!({ "label": pl }))
                        .collect();
                    sigs.push(json!({
                        "label": label,
                        "documentation": { "kind": "markdown", "value": doc_lines.join("\n\n") },
                        "parameters": param_objs
                    }));
                }
            }
        }
        if sigs.is_empty() {
            return Value::Null;
        }
        json!({
            "signatures": sigs,
            "activeSignature": 0,
            "activeParameter": active_param
        })
    }

    fn handle_workspace_symbol(&self, params: &Value) -> Value {
        let query = params
            .get("query")
            .and_then(|q| q.as_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let mut results = Vec::new();
        for (uri, an) in &self.analysis {
            let Some(prog) = an.raw_program.as_ref().or(an.full_program.as_ref()) else {
                continue;
            };
            for item in &prog.items {
                let (name, kind, span) = match item {
                    Item::Fn(f) => (f.name.as_str(), 12u32, f.span),
                    Item::Record(r) => (r.name.as_str(), 23u32, r.span),
                    Item::Enum(e) => (e.name.as_str(), 10u32, e.span),
                    Item::Interface(i) => (i.name.as_str(), 11u32, i.span),
                    Item::TypeAlias(t) => (t.name.as_str(), 5u32, t.span),
                    Item::Const(c) => (c.name.as_str(), 14u32, c.span),
                    _ => continue,
                };
                if query.is_empty() || name.to_ascii_lowercase().contains(&query) {
                    let ln = span.line.saturating_sub(1);
                    let col = span.col.saturating_sub(1);
                    results.push(json!({
                        "name": name,
                        "kind": kind,
                        "location": {
                            "uri": uri,
                            "range": {
                                "start": { "line": ln, "character": col },
                                "end": { "line": ln, "character": col + name.len() }
                            }
                        }
                    }));
                }
            }
        }
        Value::Array(results)
    }

    fn handle_code_lens(&self, params: &Value) -> Value {
        let Some(uri) = params.get("textDocument").and_then(|t| t.get("uri")).and_then(|u| u.as_str()) else {
            return Value::Array(Vec::new());
        };
        let Some(an) = self.analysis.get(uri) else {
            return Value::Array(Vec::new());
        };
        let Some(prog) = an.raw_program.as_ref() else {
            return Value::Array(Vec::new());
        };
        let file_path = self
            .docs
            .get(uri)
            .and_then(|d| d.path.as_ref())
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| uri.to_string());
        let mut lenses = Vec::new();
        for item in &prog.items {
            if let Item::Fn(f) = item {
                let ln = f.span.line.saturating_sub(1);
                let range = json!({
                    "start": { "line": ln, "character": 0 },
                    "end": { "line": ln, "character": 0 }
                });
                if f.is_test {
                    lenses.push(json!({
                        "range": range,
                        "command": {
                            "title": "▶ Run Test",
                            "command": "tenon.runTest",
                            "arguments": [file_path, f.name]
                        }
                    }));
                }
                if let Some(rep) = &an.report {
                    let fn_start = f.span.start;
                    let fn_end = f.body.span.end.max(f.span.end);
                    let mut l0_count = 0usize;
                    let mut l1_count = 0usize;
                    for p in &rep.proofs {
                        if p.span.file_id == an.active_file_id
                            && p.span.start >= fn_start
                            && p.span.start <= fn_end
                        {
                            match p.tier {
                                ProofTier::L0Fast => l0_count += 1,
                                ProofTier::L1Smt => l1_count += 1,
                            }
                        }
                    }
                    let proof_count = l0_count + l1_count;
                    let bounds_elided = rep
                        .mir
                        .proved_safe_indices
                        .iter()
                        .filter(|&&off| off >= fn_start && off <= fn_end)
                        .count();
                    let mut badges = Vec::new();
                    if proof_count > 0 {
                        badges.push(format!(
                            "✓ {} proof{} (L0:{}, L1:{})",
                            proof_count,
                            if proof_count == 1 { "" } else { "s" },
                            l0_count,
                            l1_count
                        ));
                    }
                    if bounds_elided > 0 {
                        badges.push(format!(
                            "⚡ {} bounds check{} elided",
                            bounds_elided,
                            if bounds_elided == 1 { "" } else { "s" }
                        ));
                    }
                    if f.attrs.iter().any(|a| a.name == "kernel") {
                        let has_gpu_err = rep.diagnostics.iter().any(|d| {
                            (d.code == "E0842" || d.code == "E0843")
                                && d.span.start >= fn_start
                                && d.span.start <= fn_end
                        });
                        if !has_gpu_err {
                            badges.push("⚡ @kernel: E0842 barrier-uniform · E0843 race-free (MSL + WGSL)".into());
                        }
                    }
                    if let Some(plan) = rep.mir.gpu_graphs.iter().find(|g| g.fn_name == f.name) {
                        let fused_summary = if plan.fusions.is_empty() {
                            "0 fused".to_string()
                        } else {
                            let names: Vec<&str> = plan.fusions.iter().map(|fu| fu.fused_name.as_str()).collect();
                            format!("{} fused ({})", plan.fusions.len(), names.join(", "))
                        };
                        badges.push(format!(
                            "⚡ @gpu_graph: {} pass(es) · {} · {} barrier(s) · {} VRAM slot(s)",
                            plan.nodes.len(),
                            fused_summary,
                            plan.barriers,
                            plan.vram_slots_used
                        ));
                    }
                    if rep.mir.checkfree_verified_fns.contains(&f.name) {
                        badges.push("⚡ @checkfree".into());
                    }
                    if rep.mir.fip_verified_fns.contains(&f.name) {
                        badges.push("♻ @fip".into());
                    }
                    if rep.mir.arena_verified_fns.contains(&f.name) {
                        badges.push("⛁ @arena".into());
                    }
                    if !badges.is_empty() {
                        lenses.push(json!({
                            "range": range,
                            "command": {
                                "title": badges.join(" · "),
                                "command": ""
                            }
                        }));
                    }
                }
            }
        }
        Value::Array(lenses)
    }

    fn handle_semantic_tokens(&self, params: &Value) -> Value {
        let Some(uri) = params
            .get("textDocument")
            .and_then(|t| t.get("uri"))
            .and_then(|u| u.as_str())
        else {
            return json!({ "data": [] });
        };
        let Some(doc) = self.docs.get(uri) else {
            return json!({ "data": [] });
        };
        // Token types: 0=keyword, 1=type, 2=function, 3=variable, 4=number, 5=string, 6=operator, 7=macro
        // Modifiers: bit 0=declaration, bit 1=readonly, bit 2=ghost
        let mut data: Vec<u32> = Vec::new();
        let mut prev_line: u32 = 0;
        let mut prev_col: u32 = 0;
        for (l_idx, line) in doc.text.lines().enumerate() {
            let bytes = line.as_bytes();
            let mut i = 0usize;
            while i < bytes.len() {
                if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
                    break;
                }
                if bytes[i] == b'"' {
                    let start = i;
                    i += 1;
                    while i < bytes.len() && bytes[i] != b'"' {
                        if bytes[i] == b'\\' {
                            i += 1;
                        }
                        i += 1;
                    }
                    if i < bytes.len() {
                        i += 1;
                    }
                    push_sem_tok(&mut data, &mut prev_line, &mut prev_col, l_idx as u32, start as u32, (i - start) as u32, 5, 0);
                    continue;
                }
                if bytes[i].is_ascii_digit() {
                    let start = i;
                    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'.') {
                        i += 1;
                    }
                    push_sem_tok(&mut data, &mut prev_line, &mut prev_col, l_idx as u32, start as u32, (i - start) as u32, 4, 0);
                    continue;
                }
                if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
                    let start = i;
                    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                        i += 1;
                    }
                    let word = &line[start..i];
                    let is_macro = i < bytes.len() && bytes[i] == b'!';
                    if is_macro {
                        i += 1;
                        push_sem_tok(&mut data, &mut prev_line, &mut prev_col, l_idx as u32, start as u32, (i - start) as u32, 7, 0);
                    } else if matches!(
                        word,
                        "pub" | "fn" | "pure" | "async" | "unsafe" | "comptime" | "macro" | "attr"
                            | "record" | "enum" | "interface" | "impl" | "type" | "const" | "let"
                            | "var" | "if" | "else" | "match" | "while" | "for" | "in" | "return"
                            | "break" | "continue" | "defer" | "assert" | "mutate" | "import" | "as"
                            | "where" | "requires" | "ensures" | "invariant" | "decreases" | "old"
                            | "forall" | "exists" | "inout" | "dyn"
                    ) {
                        push_sem_tok(&mut data, &mut prev_line, &mut prev_col, l_idx as u32, start as u32, word.len() as u32, 0, 0);
                    } else if matches!(word, "ghost" | "lemma") {
                        push_sem_tok(&mut data, &mut prev_line, &mut prev_col, l_idx as u32, start as u32, word.len() as u32, 0, 4);
                    } else if word.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
                        push_sem_tok(&mut data, &mut prev_line, &mut prev_col, l_idx as u32, start as u32, word.len() as u32, 1, 0);
                    } else if i < bytes.len() && (bytes[i] == b'(' || bytes[i] == b'[') {
                        push_sem_tok(&mut data, &mut prev_line, &mut prev_col, l_idx as u32, start as u32, word.len() as u32, 2, 0);
                    }
                    continue;
                }
                i += 1;
            }
        }
        json!({ "data": data })
    }
}

fn offset_to_line_col(text: &str, byte_off: usize) -> (u32, u32) {
    let clamped = byte_off.min(text.len());
    let prefix = &text[..clamped];
    let line = prefix.bytes().filter(|&b| b == b'\n').count() as u32;
    let col = prefix.rsplit('\n').next().map(|s| s.len()).unwrap_or(0) as u32;
    (line, col)
}

fn find_word_occurrences(text: &str, word: &str) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    if word.is_empty() {
        return out;
    }
    for (l_idx, line) in text.lines().enumerate() {
        let bytes = line.as_bytes();
        let wlen = word.len();
        let mut i = 0usize;
        while i + wlen <= bytes.len() {
            if &line[i..i + wlen] == word {
                let left_ok = i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
                let right_ok = i + wlen == bytes.len()
                    || !(bytes[i + wlen].is_ascii_alphanumeric() || bytes[i + wlen] == b'_');
                if left_ok && right_ok {
                    out.push((l_idx as u32, i as u32, (i + wlen) as u32));
                    i += wlen;
                    continue;
                }
            }
            i += 1;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn push_sem_tok(
    data: &mut Vec<u32>,
    prev_line: &mut u32,
    prev_col: &mut u32,
    line: u32,
    col: u32,
    len: u32,
    ty: u32,
    mods: u32,
) {
    let dl = line - *prev_line;
    let dc = if dl == 0 { col.saturating_sub(*prev_col) } else { col };
    data.extend_from_slice(&[dl, dc, len, ty, mods]);
    *prev_line = line;
    *prev_col = col;
}

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 Content-Length framing
// ---------------------------------------------------------------------------

fn read_lsp_message<R: BufRead>(reader: &mut R) -> io::Result<Option<Value>> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            return Ok(None);
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
            if let Ok(len) = rest.trim().parse::<usize>() {
                content_length = Some(len);
            }
        }
    }
    let Some(len) = content_length else {
        return Ok(None);
    };
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    let val: Value = serde_json::from_slice(&buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(Some(val))
}

fn write_lsp_message<W: Write>(writer: &mut W, msg: &Value) -> io::Result<()> {
    let body = serde_json::to_vec(msg)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
    writer.write_all(&body)?;
    writer.flush()
}

// ---------------------------------------------------------------------------
// Helpers: URI, cursor text extraction, hover formatting, symbol lookup
// ---------------------------------------------------------------------------

fn uri_to_path(uri: &str) -> Option<PathBuf> {
    let raw = uri.strip_prefix("file://")?;
    let mut bytes = Vec::with_capacity(raw.len());
    let src = raw.as_bytes();
    let mut i = 0;
    while i < src.len() {
        if src[i] == b'%' && i + 2 < src.len() {
            if let Ok(hex) = std::str::from_utf8(&src[i + 1..i + 3]) {
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    bytes.push(b);
                    i += 3;
                    continue;
                }
            }
        }
        bytes.push(src[i]);
        i += 1;
    }
    let s = String::from_utf8(bytes).ok()?;
    Some(PathBuf::from(s))
}

fn path_to_uri(path: &Path) -> String {
    let s = path.to_string_lossy();
    let encoded = s.replace(' ', "%20");
    if encoded.starts_with('/') {
        format!("file://{}", encoded)
    } else {
        format!("file:///{}", encoded)
    }
}

fn span_file_uri(prog: &Program, span: Span, fallback_uri: &str) -> String {
    if span.file_id == 0 {
        return fallback_uri.to_string();
    }
    if let Some(p) = prog.source_files.get(span.file_id as usize) {
        if !p.starts_with('<') {
            return path_to_uri(Path::new(p));
        }
    }
    fallback_uri.to_string()
}

fn extract_text_doc_pos(params: &Value) -> Option<(String, usize, usize)> {
    let uri = params
        .get("textDocument")?
        .get("uri")?
        .as_str()?
        .to_string();
    let pos = params.get("position")?;
    let line = pos.get("line")?.as_u64()? as usize;
    let ch = pos.get("character")?.as_u64()? as usize;
    Some((uri, line, ch))
}

fn line_prefix_before_cursor(text: &str, line: usize, ch: usize) -> String {
    let Some(ln) = text.lines().nth(line) else {
        return String::new();
    };
    ln.chars().take(ch).collect()
}

#[derive(Debug, Clone)]
struct CursorToken {
    word: String,
    qual_prefix: Option<String>,
    is_attr: bool,
    is_string_lit: bool,
}

fn word_at_position(text: &str, line: usize, ch: usize) -> Option<CursorToken> {
    let ln = text.lines().nth(line)?;
    let chars: Vec<char> = ln.chars().collect();
    if chars.is_empty() {
        return None;
    }
    let idx = ch.min(chars.len().saturating_sub(1));
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    let pivot = if is_ident(chars[idx]) {
        idx
    } else if idx > 0 && is_ident(chars[idx - 1]) {
        idx - 1
    } else {
        return None;
    };
    let mut start = pivot;
    while start > 0 && is_ident(chars[start - 1]) {
        start -= 1;
    }
    let mut end = pivot + 1;
    while end < chars.len() && is_ident(chars[end]) {
        end += 1;
    }
    let word: String = chars[start..end].iter().collect();

    // Check whether cursor is inside a string literal on this line
    let mut in_str = false;
    let mut i = 0;
    while i < start {
        if chars[i] == '\\' && in_str {
            i += 2;
            continue;
        }
        if chars[i] == '"' {
            in_str = !in_str;
        }
        i += 1;
    }

    let is_attr = !in_str && start > 0 && chars[start - 1] == '@';
    let qual_prefix = if !in_str && !is_attr && start >= 2 && chars[start - 1] == '.' {
        let mut pstart = start - 1;
        while pstart > 0 && is_ident(chars[pstart - 1]) {
            pstart -= 1;
        }
        if pstart < start - 1 {
            Some(chars[pstart..start - 1].iter().collect())
        } else {
            None
        }
    } else {
        None
    };
    Some(CursorToken {
        word,
        qual_prefix,
        is_attr,
        is_string_lit: in_str,
    })
}

fn make_doc_symbol(name: &str, kind: u32, span: Span, detail: String, children: Vec<Value>) -> Value {
    let l0 = span.line.saturating_sub(1);
    let c0 = span.col.saturating_sub(1);
    let range = json!({
        "start": { "line": l0, "character": c0 },
        "end": { "line": l0, "character": c0 + name.len().max(1) }
    });
    json!({
        "name": name,
        "detail": detail,
        "kind": kind,
        "range": range,
        "selectionRange": range,
        "children": children
    })
}

fn find_attr_decl<'a>(prog: &'a Program, name: &str) -> Option<&'a AttrDecl> {
    for item in &prog.items {
        if let Item::AttrDecl(a) = item {
            if a.name == name {
                return Some(a);
            }
        }
    }
    None
}

fn find_top_level_item_span(prog: &Program, name: &str) -> Option<Span> {
    for item in prog.items.iter().chain(prog.comptime_items.iter()) {
        match item {
            Item::Fn(f) if f.name == name => return Some(f.span),
            Item::ExternFn(f) if f.name == name => return Some(f.span),
            Item::Record(r) if r.name == name => return Some(r.span),
            Item::Union(u) if u.name == name => return Some(u.span),
            Item::Bitfield(b) if b.name == name => return Some(b.span),
            Item::Enum(e) if e.name == name => return Some(e.span),
            Item::Interface(i) if i.name == name => return Some(i.span),
            Item::TypeAlias(t) if t.name == name => return Some(t.span),
            Item::Const(c) if c.name == name => return Some(c.span),
            Item::Macro(m) if m.name == name => return Some(m.span),
            Item::AttrDecl(a) if a.name == name => return Some(a.span),
            _ => {}
        }
    }
    None
}

fn type_head_base(display: &str) -> &str {
    display.split('[').next().unwrap_or(display).trim()
}

fn find_member_span(prog: &Program, receiver_ty: &str, member: &str) -> Option<Span> {
    let base = type_head_base(receiver_ty);
    for item in &prog.items {
        match item {
            Item::Impl(im) if type_head_base(&im.target_ty.display_name()) == base => {
                for m in &im.methods {
                    if m.name == member {
                        return Some(m.span);
                    }
                }
            }
            Item::Record(r) if r.name == base => {
                for f in &r.fields {
                    if f.name == member {
                        return Some(f.span);
                    }
                }
            }
            Item::Union(u) if u.name == base => {
                for f in &u.fields {
                    if f.name == member {
                        return Some(f.span);
                    }
                }
            }
            Item::Bitfield(b) if b.name == base => {
                for f in &b.fields {
                    if f.name == member {
                        return Some(f.span);
                    }
                }
            }
            Item::Enum(e) if e.name == base => {
                for v in &e.variants {
                    if v.name == member {
                        return Some(v.span);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn find_any_impl_method_span(prog: &Program, member: &str) -> Option<Span> {
    for item in &prog.items {
        if let Item::Impl(im) = item {
            for m in &im.methods {
                if m.name == member {
                    return Some(m.span);
                }
            }
        }
    }
    None
}

fn format_member_hover(
    prog: &Program,
    receiver_ty: &str,
    member: &str,
    report: Option<&CheckReport>,
) -> Option<String> {
    let base = type_head_base(receiver_ty);
    for item in &prog.items {
        match item {
            Item::Impl(im) if type_head_base(&im.target_ty.display_name()) == base => {
                for m in &im.methods {
                    if m.name == member {
                        return Some(format_fn_hover(m, Some(&im.target_ty.display_name()), report));
                    }
                }
            }
            Item::Record(r) if r.name == base => {
                for f in &r.fields {
                    if f.name == member {
                        let pred = f
                            .predicate
                            .as_ref()
                            .map(|w| format!(" where {}", expr_brief(w)))
                            .unwrap_or_default();
                        return Some(format!(
                            "```tenon\n{}.{} {}{}\n```\n\nField of record `{}`.",
                            r.name,
                            f.name,
                            f.ty.display_name(),
                            pred,
                            r.name
                        ));
                    }
                }
            }
            Item::Bitfield(b) if b.name == base => {
                for f in &b.fields {
                    if f.name == member {
                        return Some(format!(
                            "```tenon\n{}.{}: {} bits\n```\n\nField of bitfield `{}` (`{}`).",
                            b.name,
                            f.name,
                            f.bits,
                            b.name,
                            b.backing_ty.display_name()
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn attribute_hover(name: &str) -> Option<String> {
    let desc = match name {
        "get" => "```tenon\n@get(\"/path\")\n```\n\nHTTP `GET` route handler attribute. Discovered at compile time by `routes_of()` and `mount_annotated_routes!()` with zero runtime reflection.",
        "post" => "```tenon\n@post(\"/path\")\n```\n\nHTTP `POST` route handler attribute. Discovered at compile time by `routes_of()` and `mount_annotated_routes!()`.",
        "put" => "```tenon\n@put(\"/path\")\n```\n\nHTTP `PUT` route handler attribute. Discovered at compile time by `routes_of()` and `mount_annotated_routes!()`.",
        "delete" => "```tenon\n@delete(\"/path\")\n```\n\nHTTP `DELETE` route handler attribute. Discovered at compile time by `routes_of()` and `mount_annotated_routes!()`.",
        "repr" => "```tenon\n@repr(C)\n```\n\nC-ABI compatible record memory layout for FFI and deterministic byte offsets (`size_of`, `align_of`, `offset_of`).",
        "packed" => "```tenon\n@packed\n```\n\nUnaligned packed struct layout (`align = 1`, zero inter-field padding).",
        "align" => "```tenon\n@align(N)\n```\n\nExplicit minimum power-of-two byte alignment for a record or union.",
        "export" => "```tenon\n@export\n```\n\nExport function with C ABI symbol name for `--lib static|shared` and `--emit-header`.",
        "link" => "```tenon\n@link(\"libname\")\n```\n\nLink an external C/native library for `extern \"C\"` declarations.",
        "cfg" => "```tenon\n@cfg(os = \"macos\")\n```\n\nCompile-time conditional compilation predicate (`os`, `arch`, `family`, `target_pointer_width`, `not(...)`).",
        "derive" => "```tenon\n@derive(Eq, Hash, ToJson, FromJson, JsonSchema)\n```\n\nCompile-time interface derivation with automatic refinement-preserving validation.",
        "evolve" => "```tenon\n@evolve(\"namespace.Entity\")\n```\n\nMarks a record for versioned schema evolution and SMT compatibility verification (`tenon evolve check`).",
        "field" => "```tenon\n@field(N)\n```\n\nOptional explicit stable wire/storage field ID for `@evolve` records.",
        "no_heap_alloc" => "```tenon\n@no_heap_alloc\n```\n\nCompile-time contract guaranteeing the function performs zero heap allocations (`E0701`).",
        _ => return None,
    };
    Some(desc.to_string())
}

fn infer_receiver_type(src: &str, receiver: &str, prog: &Program) -> Option<String> {
    // 1. Direct type name (e.g. `HttpResponse.text`, `Option.Some`)
    if receiver.chars().next().is_some_and(|c| c.is_uppercase()) {
        return Some(receiver.to_string());
    }
    // 2. Direct parameter / literal annotation from source text
    if let Some(ty) = guess_var_type_in_source(src, receiver) {
        return Some(ty);
    }
    // 3. Infer from function call initializer: `let app = build_app()` or `var conn = match http_connect(...) { ... }`
    for line in src.lines() {
        for pat in [
            format!("let {} = ", receiver),
            format!("var {} = ", receiver),
        ] {
            if let Some(idx) = line.find(&pat) {
                let mut rest = line[idx + pat.len()..].trim();
                let is_match = if let Some(after) = rest.strip_prefix("match ") {
                    rest = after.trim();
                    true
                } else {
                    false
                };
                let callee: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !callee.is_empty() {
                    for it in &prog.items {
                        if let Item::Fn(f) = it {
                            if f.name == callee {
                                if let Some(rt) = &f.ret_ty {
                                    if is_match {
                                        if let tenon_syntax::TypeExpr::Named { name, args, .. } = rt {
                                            if (name == "Result" || name == "Option") && !args.is_empty() {
                                                return Some(args[0].display_name());
                                            }
                                        }
                                    }
                                    return Some(rt.display_name());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

fn is_item_pub(item: &Item) -> bool {
    match item {
        Item::Fn(d) => d.is_pub,
        Item::Record(d) => d.is_pub,
        Item::Union(d) => d.is_pub,
        Item::Bitfield(d) => d.is_pub,
        Item::Enum(d) => d.is_pub,
        Item::Interface(d) => d.is_pub,
        Item::TypeAlias(d) => d.is_pub,
        Item::Const(d) => d.is_pub,
        Item::Macro(d) => d.is_pub,
        Item::ExternFn(d) => d.is_pub,
        _ => false,
    }
}

fn item_completion_info(item: &Item) -> Option<(String, u32, String)> {
    match item {
        Item::Fn(f) => Some((f.name.clone(), 3, format_fn_sig(f))),
        Item::ExternFn(ef) => Some((ef.name.clone(), 3, format!("extern \"C\" fn {}", ef.name))),
        Item::Record(r) => Some((r.name.clone(), 22, format!("record {}", r.name))),
        Item::Union(u) => Some((u.name.clone(), 22, format!("union {}", u.name))),
        Item::Bitfield(b) => Some((
            b.name.clone(),
            22,
            format!("bitfield {} : {}", b.name, b.backing_ty.display_name()),
        )),
        Item::Enum(e) => Some((e.name.clone(), 13, format!("enum {}", e.name))),
        Item::Interface(i) => Some((i.name.clone(), 8, format!("interface {}", i.name))),
        Item::TypeAlias(t) => Some((t.name.clone(), 22, format!("type {}", t.name))),
        Item::Const(c) => Some((c.name.clone(), 21, format!("const {}", c.name))),
        Item::Macro(m) => Some((format!("{}!", m.name), 15, format!("macro {}", m.name))),
        _ => None,
    }
}

fn format_attrs(attrs: &[Attr]) -> String {
    attrs
        .iter()
        .map(|a| {
            if a.args.is_empty() {
                format!("@{}", a.name)
            } else {
                let args: Vec<String> = a
                    .args
                    .iter()
                    .map(|arg| {
                        let v = expr_brief(&arg.value);
                        if let Some(k) = &arg.key {
                            format!("{} = {}", k, v)
                        } else {
                            v
                        }
                    })
                    .collect();
                format!("@{}({})", a.name, args.join(", "))
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_fn_sig(f: &FnDecl) -> String {
    let mut s = String::new();
    if f.is_pub {
        s.push_str("pub ");
    }
    if f.is_comptime {
        s.push_str("comptime ");
    }
    if f.is_test {
        s.push_str("test ");
    }
    if f.is_unsafe {
        s.push_str("unsafe ");
    }
    if f.is_async {
        s.push_str("async ");
    }
    if f.is_ghost {
        if !f.is_pure && f.ret_ty.is_none() {
            s.push_str("lemma ");
            s.push_str(&f.name);
        } else {
            s.push_str("ghost ");
            if f.is_pure {
                s.push_str("pure ");
            }
            s.push_str("fn ");
            s.push_str(&f.name);
        }
    } else {
        if f.is_pure {
            s.push_str("pure ");
        }
        s.push_str("fn ");
        s.push_str(&f.name);
    }
    if !f.type_params.is_empty() {
        s.push('[');
        s.push_str(&f.type_params.join(", "));
        s.push(']');
    }
    s.push('(');
    let ps: Vec<String> = f
        .params
        .iter()
        .map(|p| {
            let mut ps = String::new();
            if p.is_inout {
                ps.push_str("inout ");
            }
            ps.push_str(&p.name);
            if p.name != "self" {
                ps.push(' ');
                ps.push_str(&p.ty.display_name());
            }
            if let Some(w) = &p.predicate {
                ps.push_str(&format!(" where {}", expr_brief(w)));
            }
            ps
        })
        .collect();
    s.push_str(&ps.join(", "));
    s.push(')');
    if let Some(rt) = &f.ret_ty {
        s.push(' ');
        s.push_str(&rt.display_name());
    }
    s
}

fn format_iface_method_sig(m: &InterfaceMethodDecl) -> String {
    let mut s = String::new();
    if m.is_async {
        s.push_str("async ");
    }
    if m.is_pure {
        s.push_str("pure ");
    }
    s.push_str("fn ");
    s.push_str(&m.name);
    if !m.type_params.is_empty() {
        s.push('[');
        s.push_str(&m.type_params.join(", "));
        s.push(']');
    }
    s.push('(');
    let ps: Vec<String> = m
        .params
        .iter()
        .map(|p| {
            if p.name == "self" {
                if p.is_inout { "inout self".into() } else { "self".into() }
            } else {
                format!("{} {}", p.name, p.ty.display_name())
            }
        })
        .collect();
    s.push_str(&ps.join(", "));
    s.push(')');
    if let Some(rt) = &m.ret_ty {
        s.push(' ');
        s.push_str(&rt.display_name());
    }
    s
}

fn format_fn_hover(f: &FnDecl, impl_target: Option<&str>, report: Option<&CheckReport>) -> String {
    let mut out = String::new();
    out.push_str("```tenon\n");
    if !f.attrs.is_empty() {
        out.push_str(&format_attrs(&f.attrs));
        out.push('\n');
    }
    out.push_str(&format_fn_sig(f));
    for r in &f.requires {
        out.push_str(&format!("\n  requires {}", expr_brief(r)));
    }
    for e in &f.ensures {
        out.push_str(&format!("\n  ensures {}", expr_brief(e)));
    }
    out.push_str("\n```");
    if let Some(t) = impl_target {
        out.push_str(&format!("\n\nMethod on `{}`", t));
    }
    if let Some(rep) = report {
        let l0 = rep.proofs.iter().filter(|p| p.tier == ProofTier::L0Fast).count();
        let l1 = rep.proofs.iter().filter(|p| p.tier == ProofTier::L1Smt).count();
        if l0 + l1 > 0 {
            out.push_str(&format!(
                "\n\n*Verification*: {} L0-fast + {} L1-Z3 proofs checked ({})",
                l0,
                l1,
                if rep.ok { "PASS" } else { "FAIL" }
            ));
        }
    }
    out
}

fn format_record_hover(r: &RecordDecl, layout: &LayoutEngine) -> String {
    let mut out = String::from("```tenon\n");
    if !r.attrs.is_empty() {
        out.push_str(&format_attrs(&r.attrs));
        out.push('\n');
    }
    if r.is_pub {
        out.push_str("pub ");
    }
    out.push_str("record ");
    out.push_str(&r.name);
    if !r.type_params.is_empty() {
        out.push('[');
        out.push_str(&r.type_params.join(", "));
        out.push(']');
    }
    out.push_str(" {\n");
    for f in &r.fields {
        out.push_str(&format!("    {} {}", f.name, f.ty.display_name()));
        if let Some(w) = &f.predicate {
            out.push_str(&format!(" where {}", expr_brief(w)));
        }
        out.push('\n');
    }
    out.push_str("}\n```");

    if let Some(sl) = layout.struct_layout(&r.name) {
        let repr = if sl.is_packed {
            "@packed"
        } else if sl.is_c_repr {
            "@repr(C)"
        } else {
            "default"
        };
        out.push_str(&format!(
            "\n\n**Layout** (`{}`): `size_of = {} B`, `align_of = {} B`",
            repr, sl.size, sl.align
        ));
        if !sl.fields.is_empty() {
            out.push_str("\n\n| Field | Offset | Size | Align |\n|---|---:|---:|---:|");
            for fl in &sl.fields {
                out.push_str(&format!(
                    "\n| `{}` | `+{}` | `{} B` | `{} B` |",
                    fl.name, fl.byte_offset, fl.byte_size, fl.align
                ));
            }
        }
    }
    out
}

fn format_union_hover(u: &UnionDecl, layout: &LayoutEngine) -> String {
    let mut out = String::from("```tenon\n");
    if u.is_pub {
        out.push_str("pub ");
    }
    out.push_str(&format!("union {} {{\n", u.name));
    for f in &u.fields {
        out.push_str(&format!("    {} {}\n", f.name, f.ty.display_name()));
    }
    out.push_str("}\n```");
    if let Some(ul) = layout.union_layout(&u.name) {
        out.push_str(&format!(
            "\n\n**C Union Layout**: `size_of = {} B`, `align_of = {} B` (all fields at offset `+0`)",
            ul.size, ul.align
        ));
    }
    out
}

fn format_bitfield_hover(b: &BitfieldDecl, layout: &LayoutEngine) -> String {
    let mut out = String::from("```tenon\n");
    if b.is_pub {
        out.push_str("pub ");
    }
    out.push_str(&format!(
        "bitfield {} : {} {{\n",
        b.name,
        b.backing_ty.display_name()
    ));
    for f in &b.fields {
        out.push_str(&format!("    {}: {}\n", f.name, f.bits));
    }
    out.push_str("}\n```");
    if let Some(bl) = layout.bitfield_layout(&b.name) {
        out.push_str(&format!(
            "\n\n**Bitfield Layout**: backing `{}` ({} bits used / {} bits)\n\n| Field | Bits | Mask |\n|---|---|---|",
            bl.backing.canonical_name(),
            bl.total_bits,
            bl.backing.bit_width()
        ));
        for f in &bl.fields {
            out.push_str(&format!(
                "\n| `{}` | `[{}..{})` ({}b) | `0x{:x}` |",
                f.name,
                f.bit_offset,
                f.bit_offset + f.bit_width,
                f.bit_width,
                f.mask
            ));
        }
    }
    out
}

fn format_type_alias_hover(t: &TypeAliasDecl) -> String {
    let mut out = String::from("```tenon\n");
    if t.is_pub {
        out.push_str("pub ");
    }
    out.push_str(&format!("type {} = {}", t.name, t.base.display_name()));
    if let Some(w) = &t.predicate {
        out.push_str(&format!(" where {}", expr_brief(w)));
    }
    out.push_str("\n```");
    out
}

fn format_enum_hover(e: &EnumDecl) -> String {
    let mut out = String::from("```tenon\n");
    if e.is_pub {
        out.push_str("pub ");
    }
    out.push_str(&format!("enum {} {{\n", e.name));
    for v in &e.variants {
        if v.fields.is_empty() {
            out.push_str(&format!("    {}\n", v.name));
        } else {
            let ps: Vec<String> = v
                .fields
                .iter()
                .map(|f| format!("{} {}", f.name, f.ty.display_name()))
                .collect();
            out.push_str(&format!("    {}({})\n", v.name, ps.join(", ")));
        }
    }
    out.push_str("}\n```");
    out
}

fn format_interface_hover(i: &InterfaceDecl) -> String {
    let mut out = String::from("```tenon\n");
    if i.is_pub {
        out.push_str("pub ");
    }
    out.push_str(&format!("interface {} {{\n", i.name));
    for m in &i.methods {
        out.push_str(&format!("    {}\n", format_iface_method_sig(m)));
    }
    out.push_str("}\n```");
    out
}

fn format_const_hover(c: &ConstDecl) -> String {
    format!(
        "```tenon\n{}const {} {} = {}\n```",
        if c.is_pub { "pub " } else { "" },
        c.name,
        c.ty.display_name(),
        expr_brief(&c.value)
    )
}

fn format_attr_decl_sig(a: &AttrDecl) -> String {
    let mut s = String::new();
    if a.is_pub {
        s.push_str("pub ");
    }
    s.push_str("attr ");
    s.push_str(&a.name);
    if !a.params.is_empty() {
        s.push('(');
        let ps: Vec<String> = a
            .params
            .iter()
            .map(|p| format!("{} {}", p.name, p.ty.display_name()))
            .collect();
        s.push_str(&ps.join(", "));
        s.push(')');
    }
    s
}

fn format_attr_decl_hover(a: &AttrDecl) -> String {
    let mut out = format!("```tenon\n{}\n```", format_attr_decl_sig(a));
    if let Some(doc) = attribute_hover(&a.name) {
        if let Some((_, rest)) = doc.split_once("\n\n") {
            out.push_str("\n\n");
            out.push_str(rest);
        }
    } else {
        out.push_str("\n\nDeclared Tenon compile-time attribute.");
    }
    out
}

fn format_macro_hover(m: &MacroDecl) -> String {
    format!(
        "```tenon\n{}macro {}({})\n```\n\nCompile-time declarative AST macro.",
        if m.is_pub { "pub " } else { "" },
        m.name,
        m.params.join(", ")
    )
}

fn format_item_hover(
    prog: &Program,
    layout: &LayoutEngine,
    report: Option<&CheckReport>,
    name: &str,
) -> Option<String> {
    for item in prog.items.iter().chain(prog.comptime_items.iter()) {
        match item {
            Item::Fn(f) if f.name == name => return Some(format_fn_hover(f, None, report)),
            Item::Record(r) if r.name == name => return Some(format_record_hover(r, layout)),
            Item::Union(u) if u.name == name => return Some(format_union_hover(u, layout)),
            Item::Bitfield(b) if b.name == name => return Some(format_bitfield_hover(b, layout)),
            Item::Enum(e) if e.name == name => return Some(format_enum_hover(e)),
            Item::Interface(i) if i.name == name => return Some(format_interface_hover(i)),
            Item::TypeAlias(t) if t.name == name => return Some(format_type_alias_hover(t)),
            Item::Const(c) if c.name == name => return Some(format_const_hover(c)),
            Item::Macro(m) if m.name == name => return Some(format_macro_hover(m)),
            Item::AttrDecl(a) if a.name == name => return Some(format_attr_decl_hover(a)),
            _ => {}
        }
    }
    None
}

fn expr_brief(expr: &Expr) -> String {
    match &expr.kind {
        ExprKind::Int(n) => n.to_string(),
        ExprKind::Float(f) => f.to_string(),
        ExprKind::Bool(b) => b.to_string(),
        ExprKind::Str(s) => format!("{:?}", s),
        ExprKind::Ident(s) => s.clone(),
        ExprKind::Binary { op, left, right } => {
            let op_s = match op {
                tenon_syntax::BinOp::Add => "+",
                tenon_syntax::BinOp::Sub => "-",
                tenon_syntax::BinOp::Mul => "*",
                tenon_syntax::BinOp::Div => "/",
                tenon_syntax::BinOp::Mod => "%",
                tenon_syntax::BinOp::Eq => "==",
                tenon_syntax::BinOp::Ne => "!=",
                tenon_syntax::BinOp::Lt => "<",
                tenon_syntax::BinOp::Le => "<=",
                tenon_syntax::BinOp::Gt => ">",
                tenon_syntax::BinOp::Ge => ">=",
                tenon_syntax::BinOp::And => "&&",
                tenon_syntax::BinOp::Or => "||",
                tenon_syntax::BinOp::BitAnd => "&",
                tenon_syntax::BinOp::BitOr => "|",
                tenon_syntax::BinOp::BitXor => "^",
                tenon_syntax::BinOp::Shl => "<<",
                tenon_syntax::BinOp::Shr => ">>",
                tenon_syntax::BinOp::Implies => "==>",
                tenon_syntax::BinOp::Concat => "++",
            };
            format!("{} {} {}", expr_brief(left), op_s, expr_brief(right))
        }
        ExprKind::Field { target, field } => format!("{}.{}", expr_brief(target), field),
        ExprKind::Call { callee, args } => {
            let a: Vec<String> = args.iter().map(|x| expr_brief(&x.value)).collect();
            format!("{}({})", expr_brief(callee), a.join(", "))
        }
        _ => "...".into(),
    }
}

fn builtin_hover(word: &str) -> Option<String> {
    let desc = match word {
        "Int" | "Int64" => "`Int` (`Int64`): 64-bit signed integer (`i64`, 8 bytes).",
        "Int8" => "`Int8`: 8-bit signed integer (`i8`, 1 byte, `-128..=127`).",
        "Int16" => "`Int16`: 16-bit signed integer (`i16`, 2 bytes).",
        "Int32" => "`Int32`: 32-bit signed integer (`i32`, 4 bytes).",
        "UInt8" | "Byte" => "`UInt8` (`Byte`): 8-bit unsigned integer (`u8`, 1 byte, `0..=255`).",
        "UInt16" => "`UInt16`: 16-bit unsigned integer (`u16`, 2 bytes).",
        "UInt32" => "`UInt32`: 32-bit unsigned integer (`u32`, 4 bytes).",
        "UInt64" | "USize" => "`UInt64` / `USize`: 64-bit unsigned integer (`u64`, 8 bytes).",
        "Float" | "Float64" => "`Float` (`Float64`): IEEE-754 64-bit floating-point (`f64`).",
        "Float32" => "`Float32`: IEEE-754 32-bit floating-point (`f32`).",
        "Float32x4" => "`Float32x4`: 128-bit SIMD vector (`<4 x float>`, 16-byte aligned, LLVM native vector IR).",
        "Float64x4" => "`Float64x4`: 256-bit SIMD vector (`<4 x double>`, 32-byte aligned, LLVM native vector IR).",
        "Int32x4" => "`Int32x4`: 128-bit SIMD vector (`<4 x i32>`).",
        "UInt8x16" => "`UInt8x16`: 128-bit SIMD vector (`<16 x i8>`).",
        "Text" => "`Text`: UTF-8 immutable value-semantic string with Perceus RC & COW.",
        "Bytes" => "`Bytes`: Owned byte buffer with Perceus RC & zero-copy `BytesSlice` views.",
        "Channel" => "`Channel[T]`: Bounded lock-safe MPMC OS-thread channel.",
        "Mutex" => "`Mutex[T]`: OS mutex protecting value `T` across threads.",
        "Store" => "`Store[T]`: Generational arena with ABA-safe `Handle[T]`.",
        "comptime" => "`comptime`: Compile-time evaluation (`comptime { ... }`, `comptime fn`, `comptime if`, `comptime for`).",
        "size_of" => "`size_of[T]() USize`: Compile-time byte size of type `T`.",
        "align_of" => "`align_of[T]() USize`: Compile-time byte alignment of type `T`.",
        "offset_of" => "`offset_of(T, \"field\") USize`: Compile-time byte offset of field in `@repr(C)`/`@packed` record.",
        "refine_cast" => "`refine_cast[T](x)`: Subtype-dependent refinement cast — returns `T` (zero-cost) when `S <: T` is proved, `Result[T, Text]` when `S ∩ T ≠ ∅`, or compile error `E0416` when disjoint.",
        "is_subtype" => "`is_subtype[S, T]() Bool`: Compile-time predicate (in `comptime if`) testing whether refinement type `S` statically implies `T`.",
        "ghost" => "`ghost { ... }` / `ghost fn`: Statically verified proof code erased completely before codegen (zero runtime overhead).",
        "lemma" => "`lemma name(...) requires ... ensures ... { ... }`: Zero-overhead proof lemma whose `ensures` facts are injected into callers.",
        "old" => "`old(expr)`: Refers to the pre-state value of an `inout` parameter or field inside `ensures` postconditions.",
        "decreases" => "`decreases expr`: Loop termination variant proved `>= 0` and strictly decreasing on every `while` iteration (`E0433`).",
        _ => return None,
    };
    Some(desc.to_string())
}

fn guess_var_type_in_source(src: &str, var_name: &str) -> Option<String> {
    // 1. Parameter pattern: `var_name TypeName` inside `(...)`
    for line in src.lines() {
        let tokens: Vec<&str> = line
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|t| !t.is_empty())
            .collect();
        for w in tokens.windows(2) {
            if w[0] == var_name && w[1].chars().next().is_some_and(|c| c.is_uppercase()) {
                return Some(w[1].to_string());
            }
        }
        // 2. Binding pattern: `let var_name = TypeName {` or `var var_name = TypeName.new(`
        for prefix in [format!("let {} = ", var_name), format!("var {} = ", var_name)] {
            if let Some(idx) = line.find(&prefix) {
                let rest = line[idx + prefix.len()..].trim();
                let ty_tok: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
                if ty_tok.chars().next().is_some_and(|c| c.is_uppercase()) {
                    return Some(ty_tok);
                }
            }
        }
    }
    None
}

fn format_tensor_shape_hint(ty: &TypeExpr) -> Option<String> {
    let TypeExpr::Named { name, args, .. } = ty else {
        return None;
    };
    if !matches!(name.as_str(), "Tensor" | "Matrix" | "Tensor3D" | "Tensor4D") || args.is_empty() {
        return None;
    }
    let mut dims: Vec<i64> = Vec::new();
    let mut dtype = "Float32".to_string();
    for a in args {
        if let Some((k, v)) = a.as_named_arg() {
            if k == "DType" {
                dtype = v.display_name();
            }
            continue;
        }
        let d = a.display_name();
        if let Ok(n) = d.parse::<i64>() {
            dims.push(n);
        } else if ScalarKind::from_name(&d).is_some() {
            dtype = d;
        }
    }
    if dims.is_empty() {
        return None;
    }
    let numel: i64 = dims.iter().try_fold(1i64, |acc, &x| acc.checked_mul(x))?;
    let elem_bytes = ScalarKind::from_name(&dtype).map(|k| k.byte_size() as i64).unwrap_or(4);
    let total_bytes = numel.checked_mul(elem_bytes)?;
    let byte_str = if total_bytes >= 1024 * 1024 {
        format!("{}MB", total_bytes / (1024 * 1024))
    } else if total_bytes >= 1024 {
        format!("{}KB", total_bytes / 1024)
    } else {
        format!("{}B", total_bytes)
    };
    Some(format!(
        "rank={}, numel={}, {} {}",
        dims.len(),
        numel,
        byte_str,
        dtype
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn frame(msg: &Value) -> Vec<u8> {
        let body = serde_json::to_vec(msg).unwrap();
        let mut out = format!("Content-Length: {}\r\n\r\n", body.len()).into_bytes();
        out.extend_from_slice(&body);
        out
    }

    fn read_all_messages(bytes: &[u8]) -> Vec<Value> {
        let mut cur = Cursor::new(bytes);
        let mut msgs = Vec::new();
        while let Ok(Some(v)) = read_lsp_message(&mut cur) {
            msgs.push(v);
        }
        msgs
    }

    #[test]
    fn lsp_full_protocol_lifecycle_diagnostics_hover_def_completion_symbols_fmt() {
        let sample_src = r#"@repr(C)
pub record Packet {
    kind UInt8
    len  UInt16
    seq  UInt32
}

pub type Port = Int where 1 <= self && self <= 65535

pub pure fn double_port(p Port) Int
    ensures result >= 2
{
    p * 2
}

pub fn bad_call() Int {
    let pkt = Packet { kind = 1, len = 4, seq = 9 }
    let x = double_port(0)
    pkt.
    x
}
"#;
        // First open valid code (without `pkt.`) to test diagnostics + hover + definition + completion + symbols + fmt
        let valid_with_refinement_err = sample_src.replace("    pkt.\n", "");
        let uri = "file:///tmp/tenon_lsp_unit_test.tn";

        let mut input = Vec::new();
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "method": "initialized", "params": {}
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": { "textDocument": { "uri": uri, "languageId": "tenon", "version": 1, "text": valid_with_refinement_err } }
        })));
        // Hover over `Packet` (line 1, col 12) -> should include @repr(C) layout (`size_of = 8 B`)!
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 2, "method": "textDocument/hover",
            "params": { "textDocument": { "uri": uri }, "position": { "line": 1, "character": 12 } }
        })));
        // Hover over `double_port` at callsite (line 17, col 14)
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 3, "method": "textDocument/hover",
            "params": { "textDocument": { "uri": uri }, "position": { "line": 17, "character": 14 } }
        })));
        // Go to definition of `double_port` from callsite (line 17, col 14) -> line 9!
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 4, "method": "textDocument/definition",
            "params": { "textDocument": { "uri": uri }, "position": { "line": 17, "character": 14 } }
        })));
        // Document symbols
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 5, "method": "textDocument/documentSymbol",
            "params": { "textDocument": { "uri": uri } }
        })));
        // Completion after `pkt.` (update text to include `pkt.`)
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 6, "method": "textDocument/completion",
            "params": { "textDocument": { "uri": uri }, "position": { "line": 16, "character": 4 } }
        })));
        // Formatting
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 7, "method": "textDocument/formatting",
            "params": { "textDocument": { "uri": uri }, "options": { "tabSize": 4, "insertSpaces": true } }
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 8, "method": "shutdown", "params": null
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "method": "exit", "params": null
        })));

        let mut reader = Cursor::new(input);
        let mut output = Vec::new();
        let mut server = LspServer::new();
        assert!(server.run_loop(&mut reader, &mut output));

        let msgs = read_all_messages(&output);
        // 1. Initialize response advertises capabilities
        let init_resp = msgs.iter().find(|m| m.get("id") == Some(&json!(1))).unwrap();
        assert_eq!(init_resp["result"]["capabilities"]["hoverProvider"], json!(true));
        assert_eq!(init_resp["result"]["capabilities"]["definitionProvider"], json!(true));

        // 2. publishDiagnostics catches `double_port(0)` refinement violation!
        let diag_notif = msgs
            .iter()
            .find(|m| m.get("method") == Some(&json!("textDocument/publishDiagnostics")))
            .unwrap();
        let diags = diag_notif["params"]["diagnostics"].as_array().unwrap();
        assert!(!diags.is_empty(), "expected refinement diagnostic for double_port(0)");

        // 3. Hover over `@repr(C)` record `Packet` includes exact byte layout
        let hover_pkt = msgs.iter().find(|m| m.get("id") == Some(&json!(2))).unwrap();
        let hover_md = hover_pkt["result"]["contents"]["value"].as_str().unwrap();
        assert!(hover_md.contains("@repr(C)"), "hover missing @repr(C): {}", hover_md);
        assert!(hover_md.contains("size_of = 8 B"), "hover missing size_of = 8 B: {}", hover_md);

        // 4. Hover over `double_port` includes `ensures result >= 2`
        let hover_fn = msgs.iter().find(|m| m.get("id") == Some(&json!(3))).unwrap();
        let hover_fn_md = hover_fn["result"]["contents"]["value"].as_str().unwrap();
        assert!(hover_fn_md.contains("ensures result >= 2"), "hover_fn={}", hover_fn_md);

        // 5. Definition of `double_port` jumps to line 9 (0-based)
        let def_resp = msgs.iter().find(|m| m.get("id") == Some(&json!(4))).unwrap();
        assert_eq!(def_resp["result"]["range"]["start"]["line"], json!(9));

        // 6. Document symbols include Packet, Port, double_port, bad_call
        let sym_resp = msgs.iter().find(|m| m.get("id") == Some(&json!(5))).unwrap();
        let syms = sym_resp["result"].as_array().unwrap();
        assert!(syms.iter().any(|s| s["name"] == json!("Packet")));
        assert!(syms.iter().any(|s| s["name"] == json!("double_port")));

        // 7. Completion includes in-scope symbols & keywords
        let comp_resp = msgs.iter().find(|m| m.get("id") == Some(&json!(6))).unwrap();
        let comps = comp_resp["result"].as_array().unwrap();
        assert!(comps.iter().any(|c| c["label"] == json!("double_port")));
        assert!(comps.iter().any(|c| c["label"] == json!("Float32x4")));
    }

    #[test]
    fn lsp_hover_resolves_subtype_dependent_refine_cast_return_types() {
        let src = r#"pub type Range3to5 = Int where self >= 3 && self <= 5
pub type Range2to6 = Int where self >= 2 && self <= 6
pub type Range2to4 = Int where self >= 2 && self <= 4

pub pure fn demo(r Range3to5) Int {
    let a = refine_cast[Range2to6](r)
    let b = refine_cast[Range2to4](r)
    let _ = b
    a
}
"#;
        let uri = "file:///tmp/tenon_lsp_refine_cast.tn";
        let mut input = Vec::new();
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": { "textDocument": { "uri": uri, "languageId": "tenon", "version": 1, "text": src } }
        })));
        // Line 5: `    let a = refine_cast[Range2to6](r)` -> hover on `refine_cast` (col 15)
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 2, "method": "textDocument/hover",
            "params": { "textDocument": { "uri": uri }, "position": { "line": 5, "character": 15 } }
        })));
        // Line 6: `    let b = refine_cast[Range2to4](r)` -> hover on `refine_cast` (col 15)
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 3, "method": "textDocument/hover",
            "params": { "textDocument": { "uri": uri }, "position": { "line": 6, "character": 15 } }
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 4, "method": "shutdown", "params": null
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "method": "exit", "params": null
        })));

        let mut reader = Cursor::new(input);
        let mut output = Vec::new();
        let mut server = LspServer::new();
        assert!(server.run_loop(&mut reader, &mut output));

        let msgs = read_all_messages(&output);
        let h_widen = msgs.iter().find(|m| m.get("id") == Some(&json!(2))).unwrap();
        let md_widen = h_widen["result"]["contents"]["value"].as_str().unwrap();
        assert!(
            md_widen.contains("refine_cast -> Range2to6"),
            "expected widening hover `Range2to6`, got: {}",
            md_widen
        );
        let h_narrow = msgs.iter().find(|m| m.get("id") == Some(&json!(3))).unwrap();
        let md_narrow = h_narrow["result"]["contents"]["value"].as_str().unwrap();
        assert!(
            md_narrow.contains("refine_cast -> Result[Range2to4, Text]"),
            "expected narrowing hover `Result[Range2to4, Text]`, got: {}",
            md_narrow
        );
    }

    #[test]
    fn lsp_inlay_hints_code_actions_references_rename_and_semantic_tokens() {
        let src = r#"pub type Range[LO Int, HI Int] = Int where self >= LO && self <= HI
pub record Boxed { val Int where self >= 0 }

pub fn demo(r Range[3, 5], inout b Boxed) Int {
    let w = refine_cast[Range[2, 6]](r)
    let bad Range[2, 4] = r as Range[2, 4]
    b.val = 10
    w + bad
}
"#;
        let uri = "file:///tmp/tenon_lsp_killer.tn";
        let mut input = Vec::new();
        input.extend_from_slice(&frame(&json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {} })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": { "textDocument": { "uri": uri, "languageId": "tenon", "version": 1, "text": src } }
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 2, "method": "textDocument/inlayHint",
            "params": { "textDocument": { "uri": uri } }
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 3, "method": "textDocument/codeAction",
            "params": { "textDocument": { "uri": uri }, "range": { "start": { "line": 4, "character": 0 }, "end": { "line": 7, "character": 0 } } }
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 4, "method": "textDocument/references",
            "params": { "textDocument": { "uri": uri }, "position": { "line": 3, "character": 12 } }
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 5, "method": "textDocument/rename",
            "params": { "textDocument": { "uri": uri }, "position": { "line": 3, "character": 12 }, "newName": "val_r" }
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 6, "method": "textDocument/semanticTokens/full",
            "params": { "textDocument": { "uri": uri } }
        })));
        input.extend_from_slice(&frame(&json!({ "jsonrpc": "2.0", "id": 7, "method": "shutdown", "params": null })));
        input.extend_from_slice(&frame(&json!({ "jsonrpc": "2.0", "method": "exit", "params": null })));

        let mut reader = Cursor::new(input);
        let mut output = Vec::new();
        let mut server = LspServer::new();
        assert!(server.run_loop(&mut reader, &mut output));

        let msgs = read_all_messages(&output);
        let hints = msgs.iter().find(|m| m.get("id") == Some(&json!(2))).unwrap()["result"].as_array().unwrap();
        assert!(
            hints.iter().any(|h| h["label"].as_str().unwrap_or("").contains("-> Range[2, 6]")),
            "expected inlayHint for refine_cast -> Range[2, 6], got {:?}",
            hints
        );
        let actions = msgs.iter().find(|m| m.get("id") == Some(&json!(3))).unwrap()["result"].as_array().unwrap();
        assert!(
            actions.iter().any(|a| a["title"].as_str().unwrap_or("").contains("as?")),
            "expected E0410 quickfix to `as?`, got {:?}",
            actions
        );
        assert!(
            actions.iter().any(|a| a["title"].as_str().unwrap_or("").contains("mutate b")),
            "expected E0420 quickfix wrapping in `mutate b`, got {:?}",
            actions
        );
        let refs = msgs.iter().find(|m| m.get("id") == Some(&json!(4))).unwrap()["result"].as_array().unwrap();
        assert!(refs.len() >= 3, "expected >= 3 references for `r`, got {:?}", refs);
        let ren = &msgs.iter().find(|m| m.get("id") == Some(&json!(5))).unwrap()["result"]["changes"][uri];
        assert!(ren.as_array().unwrap().len() >= 3, "expected >= 3 rename edits");
        let sem = msgs.iter().find(|m| m.get("id") == Some(&json!(6))).unwrap()["result"]["data"].as_array().unwrap();
        assert!(sem.len() >= 25 && sem.len() % 5 == 0, "expected valid semantic token tuples");
    }

    #[test]
    fn lsp_signature_help_workspace_symbol_and_code_lens() {
        let src = r#"@checkfree
pub pure fn safe_add(a Int where 0 <= self && self <= 100, b Int where 0 <= self && self <= 100) Int {
    a + b
}

test fn test_safe_add() {
    let v = safe_add(10, 20)
    assert v == 30
}
"#;
        let uri = "file:///tmp/tenon_lsp_r4.tn";
        let mut input = Vec::new();
        input.extend_from_slice(&frame(&json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {} })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": { "textDocument": { "uri": uri, "languageId": "tenon", "version": 1, "text": src } }
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 2, "method": "textDocument/signatureHelp",
            "params": { "textDocument": { "uri": uri }, "position": { "line": 6, "character": 25 } }
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 3, "method": "workspace/symbol",
            "params": { "query": "safe_add" }
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 4, "method": "textDocument/codeLens",
            "params": { "textDocument": { "uri": uri } }
        })));
        input.extend_from_slice(&frame(&json!({ "jsonrpc": "2.0", "id": 5, "method": "shutdown", "params": null })));
        input.extend_from_slice(&frame(&json!({ "jsonrpc": "2.0", "method": "exit", "params": null })));

        let mut reader = Cursor::new(input);
        let mut output = Vec::new();
        let mut server = LspServer::new();
        assert!(server.run_loop(&mut reader, &mut output));

        let msgs = read_all_messages(&output);
        let init = msgs.iter().find(|m| m.get("id") == Some(&json!(1))).unwrap();
        assert!(init["result"]["capabilities"]["signatureHelpProvider"].is_object());
        assert_eq!(init["result"]["capabilities"]["workspaceSymbolProvider"], json!(true));
        assert!(init["result"]["capabilities"]["codeLensProvider"].is_object());

        let sig = &msgs.iter().find(|m| m.get("id") == Some(&json!(2))).unwrap()["result"];
        assert_eq!(sig["activeParameter"], json!(1));
        assert!(sig["signatures"][0]["label"].as_str().unwrap().contains("fn safe_add("));

        let wsyms = msgs.iter().find(|m| m.get("id") == Some(&json!(3))).unwrap()["result"].as_array().unwrap();
        assert!(wsyms.iter().any(|s| s["name"] == "safe_add") && wsyms.iter().any(|s| s["name"] == "test_safe_add"));

        let lenses = msgs.iter().find(|m| m.get("id") == Some(&json!(4))).unwrap()["result"].as_array().unwrap();
        assert!(
            lenses.iter().any(|l| l["command"]["title"].as_str().unwrap_or("").contains("▶ Run Test")),
            "expected Run Test CodeLens, got {:?}",
            lenses
        );
        assert!(
            lenses.iter().any(|l| l["command"]["title"].as_str().unwrap_or("").contains("@checkfree")),
            "expected @checkfree CodeLens badge, got {:?}",
            lenses
        );
    }

    #[test]
    fn lsp_h5_gpu_code_lens_and_counterexample_quickfix() {
        let gpu_src = r#"@kernel("elementwise")
pub pure fn add_bias(x List[Float32], b List[Float32], inout out List[Float32], n UInt32) {
    let i = gpu_global_id_x() as Int
    if i < (n as Int) {
        out[i] = x[i] + b[i]
    }
}

@kernel("elementwise")
pub pure fn relu_act(x List[Float32], inout out List[Float32], n UInt32) {
    let i = gpu_global_id_x() as Int
    if i < (n as Int) {
        let v = x[i]
        out[i] = if v > (0.0 as Float32) { v } else { 0.0 as Float32 }
    }
}

@gpu_graph
pub pure fn mlp_graph(x List[Float32], b List[Float32], n UInt32) List[Float32] {
    var biased List[Float32] = list_filled(16, 0.0 as Float32)
    var out List[Float32] = list_filled(16, 0.0 as Float32)
    add_bias(x, b, biased, n)
    relu_act(biased, out, n)
    out
}
"#;
        let bad_src = r#"pub type Pos = Int where self > 0
pub pure fn bad_pos(x Int where self >= 0) Pos {
    x
}
"#;
        let gpu_uri = "file:///tmp/tenon_lsp_h5_gpu.tn";
        let bad_uri = "file:///tmp/tenon_lsp_h5_bad.tn";
        let mut input = Vec::new();
        input.extend_from_slice(&frame(&json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {} })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": { "textDocument": { "uri": gpu_uri, "languageId": "tenon", "version": 1, "text": gpu_src } }
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 2, "method": "textDocument/codeLens",
            "params": { "textDocument": { "uri": gpu_uri } }
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": { "textDocument": { "uri": bad_uri, "languageId": "tenon", "version": 1, "text": bad_src } }
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 3, "method": "textDocument/codeAction",
            "params": {
                "textDocument": { "uri": bad_uri },
                "range": { "start": { "line": 0, "character": 0 }, "end": { "line": 3, "character": 0 } },
                "context": { "diagnostics": [] }
            }
        })));
        input.extend_from_slice(&frame(&json!({ "jsonrpc": "2.0", "id": 4, "method": "shutdown", "params": null })));
        input.extend_from_slice(&frame(&json!({ "jsonrpc": "2.0", "method": "exit", "params": null })));

        let mut reader = Cursor::new(input);
        let mut output = Vec::new();
        let mut server = LspServer::new();
        assert!(server.run_loop(&mut reader, &mut output));

        let msgs = read_all_messages(&output);
        let lenses = msgs.iter().find(|m| m.get("id") == Some(&json!(2))).unwrap()["result"].as_array().unwrap();
        assert!(
            lenses.iter().any(|l| {
                let t = l["command"]["title"].as_str().unwrap_or("");
                t.contains("@kernel") && t.contains("E0842") && t.contains("E0843")
            }),
            "expected @kernel safety CodeLens, got {:?}",
            lenses
        );
        assert!(
            lenses.iter().any(|l| {
                let t = l["command"]["title"].as_str().unwrap_or("");
                t.contains("@gpu_graph") && t.contains("add_bias__fused__relu_act")
            }),
            "expected @gpu_graph fusion CodeLens, got {:?}",
            lenses
        );

        let actions = msgs.iter().find(|m| m.get("id") == Some(&json!(3))).unwrap()["result"].as_array().unwrap();
        assert!(
            actions.iter().any(|a| {
                a["title"].as_str().unwrap_or("").contains("counterexample unit test")
                    && a["edit"]["changes"][bad_uri][0]["newText"]
                        .as_str()
                        .unwrap_or("")
                        .contains("test fn ")
            }),
            "expected Z3 counterexample unit test quickfix, got {:?}",
            actions
        );
    }

    #[test]
    fn lsp_tensor_shape_and_dtype_inlay_hints() {
        let src = r#"pub type Positive = Int where self >= 1
pub record NdTensor[DType = Float32] {
    shape List[Positive]
    data  List[DType]
}
pub type Tensor[...Shape Int, DType = Float32] =
    NdTensor[DType] where len(self.shape) == len(Shape) && len(self.data) == prod(Shape)

pub pure fn inspect_tensor(t Tensor[1, 1, 2, 3, 4, DType = Float16]) Int {
    let x Tensor[1, 1, 2, 3, 4, DType = Float16] = t
    len(x.data)
}
"#;
        let uri = "file:///tmp/tensor_lsp_test.tn";
        let mut input = Vec::new();
        input.extend_from_slice(&frame(&json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {} })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "method": "textDocument/didOpen",
            "params": { "textDocument": { "uri": uri, "languageId": "tenon", "version": 1, "text": src } }
        })));
        input.extend_from_slice(&frame(&json!({
            "jsonrpc": "2.0", "id": 2, "method": "textDocument/inlayHint",
            "params": { "textDocument": { "uri": uri } }
        })));
        input.extend_from_slice(&frame(&json!({ "jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": null })));
        input.extend_from_slice(&frame(&json!({ "jsonrpc": "2.0", "method": "exit", "params": null })));

        let mut reader = Cursor::new(input);
        let mut output = Vec::new();
        let mut server = LspServer::new();
        assert!(server.run_loop(&mut reader, &mut output));

        let msgs = read_all_messages(&output);
        let hints = msgs.iter().find(|m| m.get("id") == Some(&json!(2))).unwrap()["result"].as_array().unwrap();
        assert!(
            hints.iter().any(|h| {
                let lbl = h["label"].as_str().unwrap_or("");
                lbl.contains("rank=5") && lbl.contains("numel=24") && lbl.contains("48B Float16")
            }),
            "expected Tensor[1,1,2,3,4,DType=Float16] inlay hint (rank=5, numel=24, 48B Float16), got {:?}",
            hints
        );
    }
}
