use crate::ast::*;
use crate::lexer::{Token, TokenKind, lex};
use crate::span::{ParseError, Span};
use std::collections::BTreeMap;

pub fn parse(src: &str) -> Result<Program, ParseError> {
    let tokens = lex(src)?;
    let mut parser = Parser { tokens, pos: 0 };
    parser.parse_program()
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn peek_non_nl(&self) -> &Token {
        let mut p = self.pos;
        while p < self.tokens.len() && self.tokens[p].kind == TokenKind::Newline {
            p += 1;
        }
        &self.tokens[p.min(self.tokens.len() - 1)]
    }

    fn peek_ahead_non_nl(&self, skip: usize) -> &Token {
        let mut p = self.pos;
        let mut seen = 0;
        while p < self.tokens.len() {
            if self.tokens[p].kind != TokenKind::Newline {
                if seen == skip {
                    return &self.tokens[p];
                }
                seen += 1;
            }
            p += 1;
        }
        &self.tokens[self.tokens.len() - 1]
    }

    fn skip_newlines(&mut self) {
        while self.peek().kind == TokenKind::Newline {
            self.pos += 1;
        }
    }

    fn bump(&mut self) -> Token {
        let tok = self.peek().clone();
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn eat(&mut self, kind: &TokenKind) -> bool {
        self.skip_newlines();
        if &self.peek().kind == kind {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Token, ParseError> {
        self.skip_newlines();
        if self.peek().kind == kind {
            Ok(self.bump())
        } else {
            Err(ParseError {
                message: format!("expected {:?}, found {:?}", kind, self.peek().kind),
                span: self.peek().span,
            })
        }
    }

    fn expect_ident(&mut self) -> Result<(String, Span), ParseError> {
        self.skip_newlines();
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::Ident(s) => {
                self.bump();
                Ok((s, tok.span))
            }
            // Allow contextual keywords as field/attribute identifiers
            TokenKind::Output => {
                self.bump();
                Ok(("output".into(), tok.span))
            }
            TokenKind::Bounds => {
                self.bump();
                Ok(("bounds".into(), tok.span))
            }
            TokenKind::Policy => {
                self.bump();
                Ok(("policy".into(), tok.span))
            }
            TokenKind::View => {
                self.bump();
                Ok(("view".into(), tok.span))
            }
            TokenKind::Stream => {
                self.bump();
                Ok(("stream".into(), tok.span))
            }
            TokenKind::Pacing => {
                self.bump();
                Ok(("pacing".into(), tok.span))
            }
            TokenKind::From => {
                self.bump();
                Ok(("from".into(), tok.span))
            }
            TokenKind::Archive => {
                self.bump();
                Ok(("archive".into(), tok.span))
            }
            TokenKind::Retain => {
                self.bump();
                Ok(("retain".into(), tok.span))
            }
            TokenKind::Type => {
                self.bump();
                Ok(("type".into(), tok.span))
            }
            TokenKind::Bind => {
                self.bump();
                Ok(("bind".into(), tok.span))
            }
            TokenKind::Old => {
                self.bump();
                Ok(("old".into(), tok.span))
            }
            _ => Err(ParseError {
                message: format!("expected identifier, found {:?}", tok.kind),
                span: tok.span,
            }),
        }
    }

    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut items = Vec::new();
        loop {
            self.skip_newlines();
            if self.peek().kind == TokenKind::Eof {
                break;
            }
            items.push(self.parse_item()?);
        }
        Ok(Program { items })
    }

    fn parse_attrs(&mut self) -> Result<Vec<Attr>, ParseError> {
        let mut attrs = Vec::new();
        loop {
            self.skip_newlines();
            if self.peek().kind != TokenKind::At {
                break;
            }
            let start = self.bump().span;
            let (name, end) = self.expect_ident()?;
            let mut args = Vec::new();
            let mut full_span = start.merge(end);
            if self.peek().kind == TokenKind::LParen {
                self.bump();
                loop {
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::RParen {
                        full_span = full_span.merge(self.bump().span);
                        break;
                    }
                    // Check `key = expr`
                    let is_named = matches!(self.peek().kind, TokenKind::Ident(_) | TokenKind::Bounds | TokenKind::Policy | TokenKind::View | TokenKind::Stream | TokenKind::Pacing | TokenKind::Output | TokenKind::Type | TokenKind::From)
                        && self.peek_ahead_non_nl(1).kind == TokenKind::Eq;
                    if is_named {
                        let (k, _) = self.expect_ident()?;
                        self.expect(TokenKind::Eq)?;
                        let v = self.parse_expr()?;
                        args.push(AttrArg {
                            key: Some(k),
                            value: v,
                        });
                    } else {
                        let v = self.parse_expr()?;
                        args.push(AttrArg {
                            key: None,
                            value: v,
                        });
                    }
                    if !self.eat(&TokenKind::Comma) {
                        full_span = full_span.merge(self.expect(TokenKind::RParen)?.span);
                        break;
                    }
                }
            }
            attrs.push(Attr {
                name,
                args,
                span: full_span,
            });
        }
        Ok(attrs)
    }

    fn parse_item(&mut self) -> Result<Item, ParseError> {
        let attrs = self.parse_attrs()?;
        self.skip_newlines();
        if self.peek().kind == TokenKind::Import {
            let start = self.bump().span;
            self.skip_newlines();
            let tok = self.peek().clone();
            if let TokenKind::Str(path) = tok.kind {
                self.bump();
                let mut end = tok.span;
                let alias = if self.peek().kind == TokenKind::As {
                    self.bump();
                    let (a, aspan) = self.expect_ident()?;
                    end = aspan;
                    Some(a)
                } else {
                    None
                };
                return Ok(Item::Import {
                    path,
                    alias,
                    span: start.merge(end),
                });
            }
            return Err(ParseError {
                message: format!("expected string path after `import`, found {:?}", tok.kind),
                span: tok.span,
            });
        }
        let is_pub = self.eat(&TokenKind::Pub);
        self.skip_newlines();
        match self.peek().kind {
            TokenKind::Distinct | TokenKind::Type => {
                Ok(Item::TypeAlias(self.parse_type_alias(attrs, is_pub)?))
            }
            TokenKind::Record => Ok(Item::Record(self.parse_record(attrs, is_pub)?)),
            TokenKind::Enum => Ok(Item::Enum(self.parse_enum(attrs, is_pub)?)),
            TokenKind::Interface => Ok(Item::Interface(self.parse_interface(attrs, is_pub)?)),
            TokenKind::Impl => Ok(Item::Impl(self.parse_impl(attrs)?)),
            TokenKind::Pure | TokenKind::Async | TokenKind::Test | TokenKind::Fn => {
                Ok(Item::Fn(self.parse_fn(attrs, is_pub)?))
            }
            TokenKind::Evolution => Ok(Item::Evolution(self.parse_evolution(attrs)?)),
            TokenKind::LockLattice => Ok(Item::LockLattice(self.parse_lock_lattice(is_pub)?)),
            TokenKind::Protocol => Ok(Item::Protocol(self.parse_protocol(attrs, is_pub)?)),
            TokenKind::Stream => Ok(Item::Stream(self.parse_stream(attrs, is_pub)?)),
            TokenKind::Reactive | TokenKind::View => {
                Ok(Item::View(self.parse_view(attrs, is_pub)?))
            }
            TokenKind::Subscriber => Ok(Item::Subscriber(self.parse_subscriber(attrs, is_pub)?)),
            _ => Err(ParseError {
                message: format!("unexpected token at top-level: {:?}", self.peek().kind),
                span: self.peek().span,
            }),
        }
    }

    fn parse_type_params(&mut self) -> Result<Vec<String>, ParseError> {
        Ok(self.parse_type_params_with_bounds()?.0)
    }

    fn parse_type_params_with_bounds(
        &mut self,
    ) -> Result<(Vec<String>, BTreeMap<String, Vec<String>>), ParseError> {
        let mut params = Vec::new();
        let mut bounds: BTreeMap<String, Vec<String>> = BTreeMap::new();
        if self.peek().kind == TokenKind::LBracket {
            self.bump();
            loop {
                self.skip_newlines();
                if self.peek().kind == TokenKind::RBracket {
                    self.bump();
                    break;
                }
                let is_comptime = self.eat(&TokenKind::Comptime);
                let (name, _) = self.expect_ident()?;
                // Optional bound(s) e.g. `T Tool` or `K Hash + Eq` or `comptime N USize where ...`
                if !matches!(
                    self.peek_non_nl().kind,
                    TokenKind::Comma | TokenKind::RBracket
                ) {
                    let first_bound = self.parse_type_expr()?;
                    if !is_comptime {
                        let mut bvec = vec![first_bound.display_name()];
                        while self.eat(&TokenKind::Plus) {
                            let next_b = self.parse_type_expr()?;
                            bvec.push(next_b.display_name());
                        }
                        bounds.insert(name.clone(), bvec);
                    }
                    if self.eat(&TokenKind::Where) {
                        let _ = self.parse_expr()?;
                    }
                }
                params.push(name);
                if !self.eat(&TokenKind::Comma) {
                    self.expect(TokenKind::RBracket)?;
                    break;
                }
            }
        }
        Ok((params, bounds))
    }

    fn parse_interface(
        &mut self,
        attrs: Vec<Attr>,
        is_pub: bool,
    ) -> Result<InterfaceDecl, ParseError> {
        let start = self.expect(TokenKind::Interface)?.span;
        let (name, _) = self.expect_ident()?;
        let (type_params, type_param_bounds) = self.parse_type_params_with_bounds()?;
        self.expect(TokenKind::LBrace)?;
        let mut assoc_types = Vec::new();
        let mut methods = Vec::new();
        loop {
            self.skip_newlines();
            if self.peek().kind == TokenKind::RBrace {
                let end = self.bump().span;
                return Ok(InterfaceDecl {
                    attrs,
                    is_pub,
                    name,
                    type_params,
                    type_param_bounds,
                    assoc_types,
                    methods,
                    span: start.merge(end),
                });
            }
            let mattrs = self.parse_attrs()?;
            self.skip_newlines();
            if self.peek().kind == TokenKind::Type {
                self.bump();
                let (aname, _) = self.expect_ident()?;
                assoc_types.push(aname);
                self.eat_sep();
                continue;
            }
            let _mpub = self.eat(&TokenKind::Pub);
            let mstart = self.peek().span;
            let mut is_pure = false;
            let mut is_async = false;
            loop {
                self.skip_newlines();
                match self.peek().kind {
                    TokenKind::Pure => {
                        self.bump();
                        is_pure = true;
                    }
                    TokenKind::Async => {
                        self.bump();
                        is_async = true;
                    }
                    _ => break,
                }
            }
            self.expect(TokenKind::Fn)?;
            let (mname, mname_span) = self.expect_ident()?;
            let (mtype_params, mtype_param_bounds) = self.parse_type_params_with_bounds()?;
            let params = self.parse_params()?;
            // Return type on the same line before newline/brace/requires/ensures
            let ret_ty = if !matches!(
                self.peek().kind,
                TokenKind::Newline
                    | TokenKind::Semi
                    | TokenKind::LBrace
                    | TokenKind::RBrace
                    | TokenKind::Requires
                    | TokenKind::Ensures
            ) {
                Some(self.parse_type_expr()?)
            } else {
                None
            };
            let mut requires = Vec::new();
            let mut ensures = Vec::new();
            loop {
                let saved = self.pos;
                self.skip_newlines();
                if self.peek().kind == TokenKind::Requires {
                    self.bump();
                    requires.push(self.parse_expr()?);
                } else if self.peek().kind == TokenKind::Ensures {
                    self.bump();
                    ensures.push(self.parse_expr()?);
                } else {
                    self.pos = saved;
                    break;
                }
            }
            let saved_b = self.pos;
            self.skip_newlines();
            let default_body = if self.peek().kind == TokenKind::LBrace {
                Some(self.parse_block()?)
            } else {
                self.pos = saved_b;
                None
            };
            let mend = default_body
                .as_ref()
                .map(|b| b.span)
                .unwrap_or(mname_span);
            methods.push(InterfaceMethodDecl {
                attrs: mattrs,
                is_pure,
                is_async,
                name: mname,
                type_params: mtype_params,
                type_param_bounds: mtype_param_bounds,
                params,
                ret_ty,
                requires,
                ensures,
                default_body,
                span: mstart.merge(mend),
            });
            self.eat_sep();
        }
    }

    fn parse_impl(&mut self, attrs: Vec<Attr>) -> Result<ImplDecl, ParseError> {
        let start = self.expect(TokenKind::Impl)?.span;
        let (type_params, type_param_bounds) = self.parse_type_params_with_bounds()?;
        let first_ty = self.parse_type_expr()?;
        let (interface, target_ty) = if self.eat(&TokenKind::For) {
            let target = self.parse_type_expr()?;
            (Some(first_ty), target)
        } else {
            (None, first_ty)
        };
        self.expect(TokenKind::LBrace)?;
        let mut assoc_bindings = Vec::new();
        let mut methods = Vec::new();
        loop {
            self.skip_newlines();
            if self.peek().kind == TokenKind::RBrace {
                let end = self.bump().span;
                return Ok(ImplDecl {
                    attrs,
                    type_params,
                    type_param_bounds,
                    interface,
                    target_ty,
                    assoc_bindings,
                    methods,
                    span: start.merge(end),
                });
            }
            let mattrs = self.parse_attrs()?;
            self.skip_newlines();
            if self.peek().kind == TokenKind::Type {
                self.bump();
                let (aname, _) = self.expect_ident()?;
                self.expect(TokenKind::Eq)?;
                let aty = self.parse_type_expr()?;
                assoc_bindings.push((aname, aty));
                self.eat_sep();
                continue;
            }
            let mpub = self.eat(&TokenKind::Pub);
            methods.push(self.parse_fn(mattrs, mpub)?);
            self.eat_sep();
        }
    }

    fn parse_type_alias(
        &mut self,
        attrs: Vec<Attr>,
        is_pub: bool,
    ) -> Result<TypeAliasDecl, ParseError> {
        let start = self.peek().span;
        let is_distinct = self.eat(&TokenKind::Distinct);
        self.expect(TokenKind::Type)?;
        let (name, _) = self.expect_ident()?;
        let type_params = self.parse_type_params()?;
        self.expect(TokenKind::Eq)?;
        let base = self.parse_type_expr()?;
        let predicate = if self.peek().kind == TokenKind::Where
            || (self.peek_non_nl().kind == TokenKind::Where
                && self.peek_ahead_non_nl(0).kind == TokenKind::Where)
        {
            if self.eat(&TokenKind::Where) {
                Some(self.parse_expr()?)
            } else {
                None
            }
        } else {
            None
        };
        let end = predicate
            .as_ref()
            .map(|e| e.span)
            .unwrap_or(self.peek().span);
        Ok(TypeAliasDecl {
            attrs,
            is_pub,
            is_distinct,
            name,
            type_params,
            base,
            predicate,
            span: start.merge(end),
        })
    }

    fn parse_record(&mut self, attrs: Vec<Attr>, is_pub: bool) -> Result<RecordDecl, ParseError> {
        let start = self.expect(TokenKind::Record)?.span;
        let (name, _) = self.expect_ident()?;
        let (type_params, type_param_bounds) = self.parse_type_params_with_bounds()?;
        self.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();
        let mut where_clauses = Vec::new();
        let mut invariants = Vec::new();
        let mut methods = Vec::new();
        loop {
            self.skip_newlines();
            if self.peek().kind == TokenKind::RBrace {
                let end = self.bump().span;
                return Ok(RecordDecl {
                    attrs,
                    is_pub,
                    name,
                    type_params,
                    type_param_bounds,
                    fields,
                    where_clauses,
                    invariants,
                    methods,
                    span: start.merge(end),
                });
            }
            if self.peek().kind == TokenKind::Where {
                self.bump();
                where_clauses.push(self.parse_expr()?);
                self.eat_sep();
                continue;
            }
            if self.peek().kind == TokenKind::Invariant {
                let ispan = self.bump().span;
                let (iname, _) = self.expect_ident()?;
                self.expect(TokenKind::Colon)?;
                let expr = self.parse_expr()?;
                let span = ispan.merge(expr.span);
                invariants.push(NamedExpr {
                    name: iname,
                    expr,
                    span,
                });
                self.eat_sep();
                continue;
            }
            let f_attrs = self.parse_attrs()?;
            self.skip_newlines();
            // Handle `@axis(Intent) { ... }` block
            if let Some(axis_attr) = f_attrs.iter().find(|a| a.name == "axis") {
                if self.peek().kind == TokenKind::LBrace {
                    let axis_name = axis_attr
                        .args
                        .first()
                        .and_then(|a| match &a.value.kind {
                            ExprKind::Ident(s) => Some(s.clone()),
                            _ => None,
                        })
                        .unwrap_or_else(|| "Unknown".into());
                    self.bump(); // '{'
                    loop {
                        self.skip_newlines();
                        if self.peek().kind == TokenKind::RBrace {
                            self.bump();
                            break;
                        }
                        let sub_attrs = self.parse_attrs()?;
                        let mut f = self.parse_field(sub_attrs)?;
                        f.axis = Some(axis_name.clone());
                        fields.push(f);
                        self.eat_sep();
                    }
                    continue;
                }
            }
            // Handle method inside record
            if matches!(
                self.peek().kind,
                TokenKind::Pub | TokenKind::Pure | TokenKind::Async | TokenKind::Fn
            ) {
                let m_pub = self.eat(&TokenKind::Pub);
                methods.push(self.parse_fn(f_attrs, m_pub)?);
                continue;
            }
            fields.push(self.parse_field(f_attrs)?);
            self.eat_sep();
        }
    }

    fn parse_field(&mut self, attrs: Vec<Attr>) -> Result<FieldDecl, ParseError> {
        let (name, start) = self.expect_ident()?;
        let ty = self.parse_type_expr()?;
        let predicate = if self.peek().kind == TokenKind::Where {
            self.bump();
            Some(self.parse_expr()?)
        } else {
            None
        };
        let default = if self.peek().kind == TokenKind::Eq {
            self.bump();
            Some(self.parse_expr()?)
        } else {
            None
        };
        let end = default
            .as_ref()
            .map(|e| e.span)
            .or_else(|| predicate.as_ref().map(|e| e.span))
            .unwrap_or(start);
        Ok(FieldDecl {
            attrs,
            axis: None,
            name,
            ty,
            predicate,
            default,
            span: start.merge(end),
        })
    }

    fn eat_sep(&mut self) {
        while matches!(
            self.peek().kind,
            TokenKind::Newline | TokenKind::Semi | TokenKind::Comma
        ) {
            self.bump();
        }
    }

    fn parse_enum(&mut self, attrs: Vec<Attr>, is_pub: bool) -> Result<EnumDecl, ParseError> {
        let start = self.expect(TokenKind::Enum)?.span;
        let (name, _) = self.expect_ident()?;
        let type_params = self.parse_type_params()?;
        self.expect(TokenKind::LBrace)?;
        let mut variants = Vec::new();
        loop {
            self.skip_newlines();
            if self.peek().kind == TokenKind::RBrace {
                let end = self.bump().span;
                return Ok(EnumDecl {
                    attrs,
                    is_pub,
                    name,
                    type_params,
                    variants,
                    span: start.merge(end),
                });
            }
            let (vname, vstart) = self.expect_ident()?;
            let mut fields = Vec::new();
            if self.peek().kind == TokenKind::LBrace {
                self.bump();
                loop {
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::RBrace {
                        self.bump();
                        break;
                    }
                    fields.push(self.parse_field(Vec::new())?);
                    self.eat_sep();
                }
            } else if self.peek().kind == TokenKind::LParen {
                // Tuple-payload variant syntax: e.g. `JStr(Text)`, `JArr(List[JsonValue])`
                self.bump();
                let mut idx = 0usize;
                loop {
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::RParen {
                        self.bump();
                        break;
                    }
                    let fstart = self.peek().span;
                    let ty = self.parse_type_expr()?;
                    fields.push(FieldDecl {
                        attrs: Vec::new(),
                        axis: None,
                        name: format!("_{}", idx),
                        ty,
                        predicate: None,
                        default: None,
                        span: fstart,
                    });
                    idx += 1;
                    if !self.eat(&TokenKind::Comma) {
                        self.expect(TokenKind::RParen)?;
                        break;
                    }
                }
            }
            let predicate = if self.peek_non_nl().kind == TokenKind::Where {
                self.eat(&TokenKind::Where);
                Some(self.parse_expr()?)
            } else {
                None
            };
            let vend = predicate.as_ref().map(|e| e.span).unwrap_or(vstart);
            variants.push(VariantDecl {
                name: vname,
                fields,
                predicate,
                span: vstart.merge(vend),
            });
            self.eat_sep();
        }
    }

    fn parse_fn(&mut self, attrs: Vec<Attr>, is_pub: bool) -> Result<FnDecl, ParseError> {
        let start = self.peek().span;
        let mut is_pure = false;
        let mut is_async = false;
        let mut is_test = false;
        loop {
            self.skip_newlines();
            match self.peek().kind {
                TokenKind::Pure => {
                    self.bump();
                    is_pure = true;
                }
                TokenKind::Async => {
                    self.bump();
                    is_async = true;
                }
                TokenKind::Test => {
                    self.bump();
                    is_test = true;
                }
                _ => break,
            }
        }
        self.expect(TokenKind::Fn)?;
        let (name, _) = self.expect_ident()?;
        let (type_params, type_param_bounds) = self.parse_type_params_with_bounds()?;
        let params = self.parse_params()?;
        // Optional return type before `{`, `requires`, `ensures`, `retain`, `allow_loss`
        let ret_ty = if !matches!(
            self.peek_non_nl().kind,
            TokenKind::LBrace
                | TokenKind::Requires
                | TokenKind::Ensures
                | TokenKind::Retain
                | TokenKind::AllowLoss
        ) {
            Some(self.parse_type_expr()?)
        } else {
            None
        };
        let mut requires = Vec::new();
        let mut ensures = Vec::new();
        loop {
            self.skip_newlines();
            if self.peek().kind == TokenKind::Requires {
                self.bump();
                requires.push(self.parse_expr()?);
            } else if self.peek().kind == TokenKind::Ensures {
                self.bump();
                ensures.push(self.parse_expr()?);
            } else {
                break;
            }
        }
        let body = self.parse_block()?;
        let span = start.merge(body.span);
        Ok(FnDecl {
            attrs,
            is_pub,
            is_pure,
            is_async,
            is_test,
            name,
            type_params,
            type_param_bounds,
            params,
            ret_ty,
            requires,
            ensures,
            body,
            span,
        })
    }

    fn parse_params(&mut self) -> Result<Vec<ParamDecl>, ParseError> {
        self.expect(TokenKind::LParen)?;
        let mut params = Vec::new();
        loop {
            self.skip_newlines();
            if self.peek().kind == TokenKind::RParen {
                self.bump();
                break;
            }
            let pstart = self.peek().span;
            let is_inout = self.eat(&TokenKind::Inout);
            let is_comptime = self.eat(&TokenKind::Comptime);
            let (name, nspan) = self.expect_ident()?;
            let ty = if name == "self"
                && matches!(self.peek_non_nl().kind, TokenKind::Comma | TokenKind::RParen)
            {
                TypeExpr::Named {
                    name: "Self".into(),
                    version: None,
                    args: Vec::new(),
                    span: nspan,
                }
            } else {
                self.parse_type_expr()?
            };
            let predicate = if self.peek().kind == TokenKind::Where {
                self.bump();
                Some(self.parse_expr()?)
            } else {
                None
            };
            let pend = predicate.as_ref().map(|e| e.span).unwrap_or(nspan);
            params.push(ParamDecl {
                is_inout,
                is_comptime,
                name,
                ty,
                predicate,
                span: pstart.merge(pend),
            });
            if !self.eat(&TokenKind::Comma) {
                self.expect(TokenKind::RParen)?;
                break;
            }
        }
        Ok(params)
    }

    fn parse_evolution(&mut self, attrs: Vec<Attr>) -> Result<EvolutionDecl, ParseError> {
        let start = self.expect(TokenKind::Evolution)?.span;
        let (entity, _) = self.expect_ident()?;
        self.expect(TokenKind::Colon)?;
        let (from_ver, _) = self.expect_ident()?;
        self.expect(TokenKind::Bidirectional)?;
        let (to_ver, _) = self.expect_ident()?;

        let mut policy = Vec::new();
        if self.eat(&TokenKind::Policy) {
            self.expect(TokenKind::LBrace)?;
            loop {
                self.skip_newlines();
                if self.peek().kind == TokenKind::RBrace {
                    self.bump();
                    break;
                }
                let (k, _) = self.expect_ident()?;
                self.expect(TokenKind::Eq)?;
                let v = self.parse_expr()?;
                policy.push((k, v));
                self.eat_sep();
            }
        }

        self.expect(TokenKind::LBrace)?;
        let mut archives = Vec::new();
        let mut up = None;
        let mut down = None;
        let mut reup = None;
        let mut retain_fields = Vec::new();
        let mut allow_loss_fields = Vec::new();

        loop {
            self.skip_newlines();
            if self.peek().kind == TokenKind::RBrace {
                let end = self.bump().span;
                return Ok(EvolutionDecl {
                    attrs,
                    entity,
                    from_ver,
                    to_ver,
                    policy,
                    archives,
                    up,
                    down,
                    retain_fields,
                    allow_loss_fields,
                    reup,
                    span: start.merge(end),
                });
            }
            if self.peek().kind == TokenKind::Archive {
                let astart = self.bump().span;
                let (aname, _) = self.expect_ident()?;
                self.expect(TokenKind::LBrace)?;
                let mut fields = Vec::new();
                loop {
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::RBrace {
                        let aend = self.bump().span;
                        archives.push(RecordDecl {
                            attrs: Vec::new(),
                            is_pub: false,
                            name: aname,
                            type_params: Vec::new(),
                            type_param_bounds: BTreeMap::new(),
                            fields,
                            where_clauses: Vec::new(),
                            invariants: Vec::new(),
                            methods: Vec::new(),
                            span: astart.merge(aend),
                        });
                        break;
                    }
                    fields.push(self.parse_field(Vec::new())?);
                    self.eat_sep();
                }
                continue;
            }
            let (mname, mstart) = self.expect_ident()?;
            let params = self.parse_params()?;
            let ret_ty = Some(self.parse_type_expr()?);
            loop {
                self.skip_newlines();
                if self.peek().kind == TokenKind::Retain {
                    self.bump();
                    self.expect(TokenKind::LParen)?;
                    let expr = self.parse_expr()?;
                    self.expect(TokenKind::RParen)?;
                    if let ExprKind::Field { field, .. } = expr.kind {
                        retain_fields.push(field);
                    } else if let ExprKind::Ident(s) = expr.kind {
                        retain_fields.push(s);
                    }
                } else if self.peek().kind == TokenKind::AllowLoss {
                    self.bump();
                    self.expect(TokenKind::LParen)?;
                    let expr = self.parse_expr()?;
                    self.expect(TokenKind::RParen)?;
                    if let ExprKind::Field { field, .. } = expr.kind {
                        allow_loss_fields.push(field);
                    } else if let ExprKind::Ident(s) = expr.kind {
                        allow_loss_fields.push(s);
                    }
                } else {
                    break;
                }
            }
            let body = self.parse_block()?;
            let f = FnDecl {
                attrs: Vec::new(),
                is_pub: true,
                is_pure: true,
                is_async: false,
                is_test: false,
                name: mname.clone(),
                type_params: Vec::new(),
                type_param_bounds: BTreeMap::new(),
                params,
                ret_ty,
                requires: Vec::new(),
                ensures: Vec::new(),
                span: mstart.merge(body.span),
                body,
            };
            match mname.as_str() {
                "up" => up = Some(f),
                "down" => down = Some(f),
                "reup" => reup = Some(f),
                other => {
                    return Err(ParseError {
                        message: format!(
                            "unexpected clause `{}` in evolution block (expected up/down/reup)",
                            other
                        ),
                        span: mstart,
                    });
                }
            }
        }
    }

    fn parse_lock_lattice(&mut self, is_pub: bool) -> Result<LockLatticeDecl, ParseError> {
        let start = self.expect(TokenKind::LockLattice)?.span;
        let (name, _) = self.expect_ident()?;
        self.expect(TokenKind::LBrace)?;
        let mut levels = Vec::new();
        loop {
            self.skip_newlines();
            if self.peek().kind == TokenKind::RBrace {
                let end = self.bump().span;
                return Ok(LockLatticeDecl {
                    is_pub,
                    name,
                    levels,
                    span: start.merge(end),
                });
            }
            let (lvl, _) = self.expect_ident()?;
            levels.push(lvl);
            let _ = self.eat(&TokenKind::Lt);
        }
    }

    fn parse_protocol(
        &mut self,
        attrs: Vec<Attr>,
        is_pub: bool,
    ) -> Result<ProtocolDecl, ParseError> {
        let start = self.expect(TokenKind::Protocol)?.span;
        let (first, _) = self.expect_ident()?;
        let (kind, name) = if matches!(self.peek_non_nl().kind, TokenKind::Ident(_)) {
            let (n, _) = self.expect_ident()?;
            (Some(first), n)
        } else {
            (None, first)
        };
        let type_params = self.parse_type_params()?;
        let mut satisfies = Vec::new();
        let mut faults = Vec::new();
        let mut state_vars = Vec::new();
        let mut actions = Vec::new();
        let mut fault_actions = Vec::new();
        let mut invariants = Vec::new();
        loop {
            self.skip_newlines();
            if self.peek().kind == TokenKind::Satisfies {
                self.bump();
                self.expect(TokenKind::LBrace)?;
                loop {
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::RBrace {
                        self.bump();
                        break;
                    }
                    satisfies.push(self.parse_expr()?);
                    self.eat_sep();
                }
            } else if self.peek().kind == TokenKind::Laws {
                self.bump();
                self.expect(TokenKind::LBrace)?;
                loop {
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::RBrace {
                        self.bump();
                        break;
                    }
                    if self.peek().kind == TokenKind::FaultModel {
                        self.bump();
                        self.expect(TokenKind::LBrace)?;
                        loop {
                            self.skip_newlines();
                            if self.peek().kind == TokenKind::RBrace {
                                self.bump();
                                break;
                            }
                            let (f, _) = self.expect_ident()?;
                            faults.push(f);
                            self.eat_sep();
                        }
                    } else if self.peek().kind == TokenKind::Invariant {
                        let ispan = self.bump().span;
                        let (iname, _) = self.expect_ident()?;
                        self.expect(TokenKind::Colon)?;
                        let expr = self.parse_expr()?;
                        invariants.push(NamedExpr {
                            name: iname,
                            span: ispan.merge(expr.span),
                            expr,
                        });
                        self.eat_sep();
                    } else {
                        self.bump();
                    }
                }
            } else if self.peek().kind == TokenKind::LBrace {
                self.bump(); // '{'
                loop {
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::RBrace {
                        let end = self.bump().span;
                        return Ok(ProtocolDecl {
                            attrs,
                            is_pub,
                            kind,
                            name,
                            type_params,
                            satisfies,
                            faults,
                            state_vars,
                            actions,
                            fault_actions,
                            invariants,
                            span: start.merge(end),
                        });
                    }
                    if self.peek().kind == TokenKind::Laws {
                        self.bump();
                        self.expect(TokenKind::LBrace)?;
                        loop {
                            self.skip_newlines();
                            if self.peek().kind == TokenKind::RBrace {
                                self.bump();
                                break;
                            }
                            if self.peek().kind == TokenKind::Invariant {
                                let ispan = self.bump().span;
                                let (iname, _) = self.expect_ident()?;
                                self.expect(TokenKind::Colon)?;
                                let expr = self.parse_expr()?;
                                invariants.push(NamedExpr {
                                    name: iname,
                                    span: ispan.merge(expr.span),
                                    expr,
                                });
                                self.eat_sep();
                            } else {
                                self.bump();
                            }
                        }
                        continue;
                    }
                    if self.peek().kind == TokenKind::Invariant {
                        let ispan = self.bump().span;
                        let (iname, _) = self.expect_ident()?;
                        self.expect(TokenKind::Colon)?;
                        let expr = self.parse_expr()?;
                        invariants.push(NamedExpr {
                            name: iname,
                            span: ispan.merge(expr.span),
                            expr,
                        });
                        self.eat_sep();
                        continue;
                    }
                    if let TokenKind::Ident(kw) = self.peek().kind.clone() {
                        if kw == "state" && self.peek_ahead_non_nl(1).kind == TokenKind::LBrace {
                            self.bump();
                            self.expect(TokenKind::LBrace)?;
                            loop {
                                self.skip_newlines();
                                if self.peek().kind == TokenKind::RBrace {
                                    self.bump();
                                    break;
                                }
                                let (vname, vstart) = self.expect_ident()?;
                                let vty = self.parse_type_expr()?;
                                self.expect(TokenKind::Eq)?;
                                let init = self.parse_expr()?;
                                let span = vstart.merge(init.span);
                                state_vars.push(ProtocolVarDecl {
                                    name: vname,
                                    ty: vty,
                                    init,
                                    span,
                                });
                                self.eat_sep();
                            }
                            continue;
                        }
                        if kw == "action" || kw == "fault" {
                            let is_fault = kw == "fault";
                            let astart = self.bump().span;
                            let (aname, _) = self.expect_ident()?;
                            let guard = if matches!(&self.peek_non_nl().kind, TokenKind::Ident(w) if w == "when")
                            {
                                self.skip_newlines();
                                self.bump(); // 'when'
                                Some(self.parse_expr()?)
                            } else {
                                None
                            };
                            self.expect(TokenKind::LBrace)?;
                            let mut updates = Vec::new();
                            let aend = loop {
                                self.skip_newlines();
                                if self.peek().kind == TokenKind::RBrace {
                                    break self.bump().span;
                                }
                                let (target, _) = self.expect_ident()?;
                                self.expect(TokenKind::Eq)?;
                                let rhs = self.parse_expr()?;
                                updates.push((target, rhs));
                                self.eat_sep();
                            };
                            let act = ProtocolActionDecl {
                                name: aname,
                                is_fault,
                                guard,
                                updates,
                                span: astart.merge(aend),
                            };
                            if is_fault {
                                fault_actions.push(act);
                            } else {
                                actions.push(act);
                            }
                            continue;
                        }
                    }
                    // Skip opaque legacy protocol statements cleanly
                    if self.peek().kind == TokenKind::LBrace {
                        let _ = self.skip_balanced_braces()?;
                    } else {
                        self.bump();
                    }
                }
            } else {
                let end = self.peek().span;
                return Ok(ProtocolDecl {
                    attrs,
                    is_pub,
                    kind,
                    name,
                    type_params,
                    satisfies,
                    faults,
                    state_vars,
                    actions,
                    fault_actions,
                    invariants,
                    span: start.merge(end),
                });
            }
        }
    }

    fn skip_balanced_braces(&mut self) -> Result<Span, ParseError> {
        let mut depth = 0;
        let mut last = self.peek().span;
        loop {
            match self.peek().kind {
                TokenKind::LBrace => {
                    depth += 1;
                    last = self.bump().span;
                }
                TokenKind::RBrace => {
                    depth -= 1;
                    last = self.bump().span;
                    if depth == 0 {
                        return Ok(last);
                    }
                }
                TokenKind::Eof => {
                    return Err(ParseError {
                        message: "unclosed `{`".into(),
                        span: last,
                    });
                }
                _ => {
                    last = self.bump().span;
                }
            }
        }
    }

    fn parse_stream(&mut self, attrs: Vec<Attr>, is_pub: bool) -> Result<StreamDecl, ParseError> {
        let start = self.expect(TokenKind::Stream)?.span;
        let (name, _) = self.expect_ident()?;
        let params = if self.peek().kind == TokenKind::LParen {
            self.parse_params()?
        } else {
            Vec::new()
        };
        let item_ty = self.parse_type_expr()?;
        self.expect(TokenKind::LBrace)?;
        let mut entries = Vec::new();
        loop {
            self.skip_newlines();
            if self.peek().kind == TokenKind::RBrace {
                let end = self.bump().span;
                return Ok(StreamDecl {
                    attrs,
                    is_pub,
                    name,
                    params,
                    item_ty,
                    entries,
                    span: start.merge(end),
                });
            }
            let (k, _) = self.expect_ident()?;
            self.expect(TokenKind::Eq)?;
            let v = self.parse_expr()?;
            entries.push((k, v));
            self.eat_sep();
        }
    }

    fn parse_view(&mut self, attrs: Vec<Attr>, is_pub: bool) -> Result<ViewDecl, ParseError> {
        let start = self.peek().span;
        let is_reactive = self.eat(&TokenKind::Reactive);
        self.expect(TokenKind::View)?;
        let (name, _) = self.expect_ident()?;
        let params = if self.peek().kind == TokenKind::LParen {
            self.parse_params()?
        } else {
            Vec::new()
        };
        let source = if self.eat(&TokenKind::From) {
            Some(self.expect_ident()?.0)
        } else {
            None
        };
        let mut properties = Vec::new();
        let mut bounds = Vec::new();
        let mut body = None;
        self.expect(TokenKind::LBrace)?;
        loop {
            self.skip_newlines();
            if self.peek().kind == TokenKind::RBrace {
                let end = self.bump().span;
                return Ok(ViewDecl {
                    attrs,
                    is_pub,
                    is_reactive,
                    name,
                    params,
                    source,
                    properties,
                    bounds,
                    body,
                    span: start.merge(end),
                });
            }
            if self.peek().kind == TokenKind::Bounds {
                self.bump();
                let _ = self.eat(&TokenKind::Eq);
                self.expect(TokenKind::LBrace)?;
                loop {
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::RBrace {
                        self.bump();
                        break;
                    }
                    let (k, _) = self.expect_ident()?;
                    self.expect(TokenKind::Eq)?;
                    let v = self.parse_expr()?;
                    bounds.push((k, v));
                    self.eat_sep();
                }
                continue;
            }
            if matches!(self.peek().kind, TokenKind::Let | TokenKind::Output) {
                let mut stmts = Vec::new();
                let mut tail = None;
                let bstart = self.peek().span;
                while self.peek_non_nl().kind != TokenKind::RBrace {
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::Output {
                        self.bump();
                        tail = Some(Box::new(self.parse_expr()?));
                        self.eat_sep();
                    } else {
                        stmts.push(self.parse_stmt()?);
                        self.eat_sep();
                    }
                }
                body = Some(Block {
                    stmts,
                    tail,
                    span: bstart.merge(self.peek().span),
                });
                continue;
            }
            let (k, _) = self.expect_ident()?;
            self.expect(TokenKind::Eq)?;
            let v = self.parse_expr()?;
            properties.push((k, v));
            self.eat_sep();
        }
    }

    fn parse_subscriber(
        &mut self,
        attrs: Vec<Attr>,
        is_pub: bool,
    ) -> Result<SubscriberDecl, ParseError> {
        let start = self.expect(TokenKind::Subscriber)?.span;
        let (name, _) = self.expect_ident()?;
        let params = if self.peek().kind == TokenKind::LParen {
            self.parse_params()?
        } else {
            Vec::new()
        };
        let mut view_name = None;
        let mut pacing = Vec::new();
        let mut storage = Vec::new();
        let mut properties = Vec::new();
        self.expect(TokenKind::LBrace)?;
        loop {
            self.skip_newlines();
            if self.peek().kind == TokenKind::RBrace {
                let end = self.bump().span;
                return Ok(SubscriberDecl {
                    attrs,
                    is_pub,
                    name,
                    params,
                    view_name,
                    pacing,
                    storage,
                    properties,
                    span: start.merge(end),
                });
            }
            if self.peek().kind == TokenKind::Bind {
                self.bump();
                let _ = self.expect_ident()?;
                self.expect(TokenKind::Eq)?;
                let expr = self.parse_expr()?;
                view_name = match &expr.kind {
                    ExprKind::Call { callee, .. } => match &callee.kind {
                        ExprKind::Ident(s) => Some(s.clone()),
                        _ => None,
                    },
                    ExprKind::Ident(s) => Some(s.clone()),
                    _ => None,
                };
                self.eat_sep();
                continue;
            }
            if self.peek().kind == TokenKind::Pacing {
                self.bump();
                self.expect(TokenKind::LBrace)?;
                loop {
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::RBrace {
                        self.bump();
                        break;
                    }
                    let k = self.parse_dotted_key()?;
                    self.expect(TokenKind::Eq)?;
                    let v = self.parse_expr()?;
                    pacing.push((k, v));
                    self.eat_sep();
                }
                continue;
            }
            if self.peek().kind == TokenKind::LocalStorage {
                self.bump();
                self.expect(TokenKind::LBrace)?;
                loop {
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::RBrace {
                        self.bump();
                        break;
                    }
                    let (k, _) = self.expect_ident()?;
                    self.expect(TokenKind::Eq)?;
                    let v = self.parse_expr()?;
                    storage.push((k, v));
                    self.eat_sep();
                }
                continue;
            }
            let (k, _) = self.expect_ident()?;
            self.expect(TokenKind::Eq)?;
            let v = self.parse_expr()?;
            properties.push((k, v));
            self.eat_sep();
        }
    }

    fn parse_dotted_key(&mut self) -> Result<String, ParseError> {
        let (mut key, _) = self.expect_ident()?;
        while self.peek().kind == TokenKind::Dot || self.peek().kind == TokenKind::Pipe {
            if self.peek().kind == TokenKind::Dot {
                self.bump();
                let (part, _) = self.expect_ident()?;
                key.push('.');
                key.push_str(&part);
            } else {
                self.bump();
                let (part, _) = self.expect_ident()?;
                key.push('|');
                key.push_str(&part);
            }
        }
        Ok(key)
    }

    fn parse_type_expr(&mut self) -> Result<TypeExpr, ParseError> {
        self.skip_newlines();
        if self.peek().kind == TokenKind::Dyn {
            let start = self.bump().span;
            let (interface, ispan) = self.expect_ident()?;
            let mut end = ispan;
            let mut args = Vec::new();
            if self.peek().kind == TokenKind::LBracket {
                self.bump();
                loop {
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::RBracket {
                        end = self.bump().span;
                        break;
                    }
                    args.push(self.parse_type_expr()?);
                    if !self.eat(&TokenKind::Comma) {
                        end = self.expect(TokenKind::RBracket)?.span;
                        break;
                    }
                }
            }
            return Ok(TypeExpr::Dyn {
                interface,
                args,
                span: start.merge(end),
            });
        }
        if self.peek().kind == TokenKind::LParen {
            let start = self.bump().span;
            let mut elems = Vec::new();
            loop {
                self.skip_newlines();
                if self.peek().kind == TokenKind::RParen {
                    let end = self.bump().span;
                    return Ok(TypeExpr::Tuple(elems, start.merge(end)));
                }
                elems.push(self.parse_type_expr()?);
                if !self.eat(&TokenKind::Comma) {
                    let end = self.expect(TokenKind::RParen)?.span;
                    return Ok(TypeExpr::Tuple(elems, start.merge(end)));
                }
            }
        }
        let (name, start) = self.expect_ident()?;
        let mut end = start;
        let version = if self.peek().kind == TokenKind::At {
            self.bump();
            let (v, vspan) = self.expect_ident()?;
            end = vspan;
            Some(v)
        } else {
            None
        };
        let mut args = Vec::new();
        if self.peek().kind == TokenKind::LBracket {
            self.bump();
            loop {
                self.skip_newlines();
                if self.peek().kind == TokenKind::RBracket {
                    end = self.bump().span;
                    break;
                }
                if let TokenKind::Int(n) = self.peek().kind {
                    let sp = self.bump().span;
                    args.push(TypeExpr::Named {
                        name: n.to_string(),
                        version: None,
                        args: Vec::new(),
                        span: sp,
                    });
                } else {
                    args.push(self.parse_type_expr()?);
                }
                if !self.eat(&TokenKind::Comma) {
                    end = self.expect(TokenKind::RBracket)?.span;
                    break;
                }
            }
        }
        Ok(TypeExpr::Named {
            name,
            version,
            args,
            span: start.merge(end),
        })
    }

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let start = self.expect(TokenKind::LBrace)?.span;
        let mut stmts = Vec::new();
        let mut tail = None;
        loop {
            self.skip_newlines();
            if self.peek().kind == TokenKind::RBrace {
                let end = self.bump().span;
                return Ok(Block {
                    stmts,
                    tail,
                    span: start.merge(end),
                });
            }
            if matches!(
                self.peek().kind,
                TokenKind::Let
                    | TokenKind::Var
                    | TokenKind::Return
                    | TokenKind::Assert
                    | TokenKind::Mutate
                    | TokenKind::While
                    | TokenKind::Break
                    | TokenKind::Continue
            ) {
                stmts.push(self.parse_stmt()?);
                self.eat_sep();
                continue;
            }
            let expr = self.parse_expr()?;
            if matches!(
                self.peek().kind,
                TokenKind::Eq | TokenKind::PlusEq | TokenKind::MinusEq
            ) {
                let op = match self.bump().kind {
                    TokenKind::Eq => AssignOp::Set,
                    TokenKind::PlusEq => AssignOp::AddSet,
                    TokenKind::MinusEq => AssignOp::SubSet,
                    _ => unreachable!(),
                };
                let value = self.parse_expr()?;
                let span = expr.span.merge(value.span);
                stmts.push(Stmt::Assign {
                    target: expr,
                    op,
                    value,
                    span,
                });
                self.eat_sep();
                continue;
            }
            self.skip_newlines();
            if self.peek().kind == TokenKind::RBrace {
                tail = Some(Box::new(expr));
            } else {
                let span = expr.span;
                stmts.push(Stmt::Expr { expr, span });
                let _ = self.eat(&TokenKind::Semi);
            }
        }
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        self.skip_newlines();
        match self.peek().kind {
            TokenKind::Let | TokenKind::Var => {
                let start = self.peek().span;
                let is_mut = self.bump().kind == TokenKind::Var;
                let (name, _) = self.expect_ident()?;
                let ty = if self.peek().kind != TokenKind::Eq {
                    Some(self.parse_type_expr()?)
                } else {
                    None
                };
                self.expect(TokenKind::Eq)?;
                let value = self.parse_expr()?;
                let span = start.merge(value.span);
                Ok(Stmt::Let {
                    name,
                    ty,
                    value,
                    is_mut,
                    span,
                })
            }
            TokenKind::Return => {
                let start = self.bump().span;
                if matches!(
                    self.peek().kind,
                    TokenKind::Newline | TokenKind::Semi | TokenKind::RBrace
                ) {
                    Ok(Stmt::Return {
                        value: None,
                        span: start,
                    })
                } else {
                    let val = self.parse_expr()?;
                    let span = start.merge(val.span);
                    Ok(Stmt::Return {
                        value: Some(val),
                        span,
                    })
                }
            }
            TokenKind::Assert => {
                let start = self.bump().span;
                let cond = self.parse_expr()?;
                let span = start.merge(cond.span);
                Ok(Stmt::Assert { cond, span })
            }
            TokenKind::Mutate => {
                let start = self.bump().span;
                let (target, _) = self.expect_ident()?;
                let body = self.parse_block()?;
                let span = start.merge(body.span);
                Ok(Stmt::Mutate { target, body, span })
            }
            TokenKind::While => {
                let start = self.bump().span;
                let cond = self.parse_expr()?;
                let mut invariants = Vec::new();
                let mut decreases = None;
                loop {
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::Invariant {
                        self.bump();
                        invariants.push(self.parse_expr()?);
                    } else if self.peek().kind == TokenKind::Decreases {
                        self.bump();
                        decreases = Some(self.parse_expr()?);
                    } else {
                        break;
                    }
                }
                let body = self.parse_block()?;
                let span = start.merge(body.span);
                Ok(Stmt::While {
                    cond,
                    invariants,
                    decreases,
                    body,
                    span,
                })
            }
            TokenKind::Break => {
                let span = self.bump().span;
                Ok(Stmt::Break { span })
            }
            TokenKind::Continue => {
                let span = self.bump().span;
                Ok(Stmt::Continue { span })
            }
            _ => {
                let expr = self.parse_expr()?;
                let span = expr.span;
                Ok(Stmt::Expr { expr, span })
            }
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_implies()
    }

    fn parse_implies(&mut self) -> Result<Expr, ParseError> {
        let left = self.parse_or()?;
        if self.peek().kind == TokenKind::FatArrow {
            self.bump();
            let right = self.parse_implies()?;
            let span = left.span.merge(right.span);
            return Ok(Expr {
                kind: ExprKind::Binary {
                    op: BinOp::Implies,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            });
        }
        Ok(left)
    }

    fn parse_or(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_and()?;
        while self.peek().kind == TokenKind::PipePipe {
            self.bump();
            self.skip_newlines();
            let right = self.parse_and()?;
            let span = left.span.merge(right.span);
            left = Expr {
                kind: ExprKind::Binary {
                    op: BinOp::Or,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_bitor()?;
        while self.peek().kind == TokenKind::AmpAmp {
            self.bump();
            self.skip_newlines();
            let right = self.parse_bitor()?;
            let span = left.span.merge(right.span);
            left = Expr {
                kind: ExprKind::Binary {
                    op: BinOp::And,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_bitor(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_bitxor()?;
        while self.peek().kind == TokenKind::Pipe {
            self.bump();
            self.skip_newlines();
            let right = self.parse_bitxor()?;
            let span = left.span.merge(right.span);
            left = Expr {
                kind: ExprKind::Binary {
                    op: BinOp::BitOr,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_bitxor(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_bitand()?;
        while self.peek().kind == TokenKind::Caret {
            self.bump();
            self.skip_newlines();
            let right = self.parse_bitand()?;
            let span = left.span.merge(right.span);
            left = Expr {
                kind: ExprKind::Binary {
                    op: BinOp::BitXor,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_bitand(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_cmp()?;
        while self.peek().kind == TokenKind::Amp {
            self.bump();
            self.skip_newlines();
            let right = self.parse_cmp()?;
            let span = left.span.merge(right.span);
            left = Expr {
                kind: ExprKind::Binary {
                    op: BinOp::BitAnd,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_cmp(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_shift()?;
        while let Some(op) = match self.peek().kind {
            TokenKind::EqEq => Some(BinOp::Eq),
            TokenKind::BangEq => Some(BinOp::Ne),
            TokenKind::Lt => Some(BinOp::Lt),
            TokenKind::Le => Some(BinOp::Le),
            TokenKind::Gt => Some(BinOp::Gt),
            TokenKind::Ge => Some(BinOp::Ge),
            _ => None,
        } {
            self.bump();
            self.skip_newlines();
            let right = self.parse_shift()?;
            let span = left.span.merge(right.span);
            left = Expr {
                kind: ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_shift(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_add()?;
        while let Some(op) = match self.peek().kind {
            TokenKind::Shl => Some(BinOp::Shl),
            TokenKind::Shr => Some(BinOp::Shr),
            _ => None,
        } {
            self.bump();
            self.skip_newlines();
            let right = self.parse_add()?;
            let span = left.span.merge(right.span);
            left = Expr {
                kind: ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_add(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_mul()?;
        while let Some(op) = match self.peek().kind {
            TokenKind::Plus => Some(BinOp::Add),
            TokenKind::Minus => Some(BinOp::Sub),
            TokenKind::PlusPlus => Some(BinOp::Concat),
            _ => None,
        } {
            self.bump();
            self.skip_newlines();
            let right = self.parse_mul()?;
            let span = left.span.merge(right.span);
            left = Expr {
                kind: ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_mul(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.parse_unary()?;
        while let Some(op) = match self.peek().kind {
            TokenKind::Star => Some(BinOp::Mul),
            TokenKind::Slash => Some(BinOp::Div),
            TokenKind::Percent => Some(BinOp::Mod),
            _ => None,
        } {
            self.bump();
            self.skip_newlines();
            let right = self.parse_unary()?;
            let span = left.span.merge(right.span);
            left = Expr {
                kind: ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            };
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        self.skip_newlines();
        match self.peek().kind {
            TokenKind::Minus => {
                let start = self.bump().span;
                let expr = self.parse_unary()?;
                let span = start.merge(expr.span);
                Ok(Expr {
                    kind: ExprKind::Unary {
                        op: UnOp::Neg,
                        expr: Box::new(expr),
                    },
                    span,
                })
            }
            TokenKind::Bang => {
                let start = self.bump().span;
                let expr = self.parse_unary()?;
                let span = start.merge(expr.span);
                Ok(Expr {
                    kind: ExprKind::Unary {
                        op: UnOp::Not,
                        expr: Box::new(expr),
                    },
                    span,
                })
            }
            TokenKind::Move => {
                self.bump();
                self.parse_unary()
            }
            TokenKind::Await => {
                let start = self.bump().span;
                let inner = self.parse_unary()?;
                let span = start.merge(inner.span);
                Ok(Expr {
                    kind: ExprKind::Await(Box::new(inner)),
                    span,
                })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek().kind {
                TokenKind::Dot => {
                    self.bump();
                    if self.peek().kind == TokenKind::Await {
                        let aspan = self.bump().span;
                        let span = expr.span.merge(aspan);
                        expr = Expr {
                            kind: ExprKind::Await(Box::new(expr)),
                            span,
                        };
                        continue;
                    }
                    let (field, fspan) = self.expect_ident()?;
                    let span = expr.span.merge(fspan);
                    expr = Expr {
                        kind: ExprKind::Field {
                            target: Box::new(expr),
                            field,
                        },
                        span,
                    };
                }
                TokenKind::LParen => {
                    self.bump();
                    let mut args = Vec::new();
                    let end = loop {
                        self.skip_newlines();
                        if self.peek().kind == TokenKind::RParen {
                            break self.bump().span;
                        }
                        let is_named = matches!(
                            self.peek().kind,
                            TokenKind::Ident(_) | TokenKind::From | TokenKind::Output
                        ) && self.peek_ahead_non_nl(1).kind == TokenKind::Eq;
                        if is_named {
                            let (k, _) = self.expect_ident()?;
                            self.expect(TokenKind::Eq)?;
                            let v = self.parse_expr()?;
                            args.push(CallArg {
                                name: Some(k),
                                value: v,
                            });
                        } else {
                            let v = self.parse_expr()?;
                            args.push(CallArg {
                                name: None,
                                value: v,
                            });
                        }
                        if !self.eat(&TokenKind::Comma) {
                            break self.expect(TokenKind::RParen)?.span;
                        }
                    };
                    let span = expr.span.merge(end);
                    expr = Expr {
                        kind: ExprKind::Call {
                            callee: Box::new(expr),
                            args,
                        },
                        span,
                    };
                }
                TokenKind::LBracket => {
                    // Disambiguate generic call instantiation `decode[Order](raw)` vs indexing `arr[i]`
                    // by speculatively parsing `[TypeExpr, ...]` ONLY when immediately followed by `(`.
                    let saved = self.pos;
                    self.bump();
                    let mut parsed_type_args = Vec::new();
                    let mut ok_generic = false;
                    let mut generic_end = expr.span;
                    if self.peek().kind != TokenKind::RBracket {
                        let mut valid = true;
                        loop {
                            match self.parse_type_expr() {
                                Ok(t) => parsed_type_args.push(t),
                                Err(_) => {
                                    valid = false;
                                    break;
                                }
                            }
                            if !self.eat(&TokenKind::Comma) {
                                if self.peek().kind == TokenKind::RBracket {
                                    generic_end = self.bump().span;
                                } else {
                                    valid = false;
                                }
                                break;
                            }
                        }
                        let is_type_head = matches!(
                            &expr.kind,
                            ExprKind::Ident(s) if s.chars().next().is_some_and(|c| c.is_ascii_uppercase())
                        );
                        if valid
                            && (self.peek().kind == TokenKind::LParen
                                || (is_type_head && self.peek().kind == TokenKind::Dot))
                        {
                            ok_generic = true;
                        }
                    }
                    if ok_generic {
                        let span = expr.span.merge(generic_end);
                        expr = Expr {
                            kind: ExprKind::GenericInst {
                                expr: Box::new(expr),
                                type_args: parsed_type_args,
                            },
                            span,
                        };
                    } else {
                        self.pos = saved;
                        self.bump();
                        let idx = self.parse_expr()?;
                        let end = self.expect(TokenKind::RBracket)?.span;
                        let span = expr.span.merge(end);
                        expr = Expr {
                            kind: ExprKind::Index {
                                target: Box::new(expr),
                                index: Box::new(idx),
                            },
                            span,
                        };
                    }
                }
                _ => {
                    // Allow fluent multiline `.method()` chaining across newlines
                    let saved = self.pos;
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::Dot {
                        continue;
                    }
                    self.pos = saved;
                    break;
                }
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        self.skip_newlines();
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::Int(n) => {
                self.bump();
                // Support unit suffix like `64 KiB`, `33 ms`, `100 items`, `50 turns`
                if let TokenKind::Ident(u) = &self.peek().kind {
                    if matches!(
                        u.as_str(),
                        "B" | "KiB" | "MiB" | "GiB" | "ms" | "s" | "m" | "items" | "turns" | "batches" | "frames"
                    ) {
                        let unit = u.clone();
                        let uspan = self.bump().span;
                        return Ok(Expr {
                            kind: ExprKind::UnitValue { value: n, unit },
                            span: tok.span.merge(uspan),
                        });
                    }
                }
                Ok(Expr {
                    kind: ExprKind::Int(n),
                    span: tok.span,
                })
            }
            TokenKind::True => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::Bool(true),
                    span: tok.span,
                })
            }
            TokenKind::False => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::Bool(false),
                    span: tok.span,
                })
            }
            TokenKind::Str(s) => {
                self.bump();
                Ok(Expr {
                    kind: ExprKind::Str(s),
                    span: tok.span,
                })
            }
            TokenKind::Old => {
                self.bump();
                if self.peek().kind == TokenKind::LParen {
                    self.bump();
                    let inner = self.parse_expr()?;
                    let end = self.expect(TokenKind::RParen)?.span;
                    Ok(Expr {
                        kind: ExprKind::Old(Box::new(inner)),
                        span: tok.span.merge(end),
                    })
                } else {
                    Ok(Expr {
                        kind: ExprKind::Ident("old".into()),
                        span: tok.span,
                    })
                }
            }
            TokenKind::View
            | TokenKind::Stream
            | TokenKind::Policy
            | TokenKind::Bounds
            | TokenKind::From
            | TokenKind::Output
            | TokenKind::Pacing
            | TokenKind::Bind
            | TokenKind::Archive
            | TokenKind::Retain => {
                let (name, span) = self.expect_ident()?;
                Ok(Expr {
                    kind: ExprKind::Ident(name),
                    span,
                })
            }
            TokenKind::Forall => {
                self.bump();
                let mut binders = Vec::new();
                loop {
                    let (bname, _) = self.expect_ident()?;
                    let bty = self.parse_type_expr()?;
                    binders.push((bname, bty));
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                let guard = if self.eat(&TokenKind::Where) {
                    Some(Box::new(self.parse_or()?))
                } else {
                    None
                };
                self.expect(TokenKind::FatArrow)?;
                let body = self.parse_expr()?;
                let span = tok.span.merge(body.span);
                Ok(Expr {
                    kind: ExprKind::Forall {
                        binders,
                        guard,
                        body: Box::new(body),
                    },
                    span,
                })
            }
            TokenKind::If => {
                self.bump();
                let cond = self.parse_expr()?;
                let then_block = self.parse_block()?;
                let saved_pos = self.pos;
                let else_branch = if self.eat(&TokenKind::Else) {
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::If {
                        Some(Box::new(self.parse_primary()?))
                    } else {
                        let blk = self.parse_block()?;
                        let bspan = blk.span;
                        Some(Box::new(Expr {
                            kind: ExprKind::Block(blk),
                            span: bspan,
                        }))
                    }
                } else {
                    self.pos = saved_pos;
                    None
                };
                let end = else_branch
                    .as_ref()
                    .map(|e| e.span)
                    .unwrap_or(then_block.span);
                Ok(Expr {
                    kind: ExprKind::If {
                        cond: Box::new(cond),
                        then_block,
                        else_branch,
                    },
                    span: tok.span.merge(end),
                })
            }
            TokenKind::Match => {
                self.bump();
                let scrutinee = self.parse_expr()?;
                self.expect(TokenKind::LBrace)?;
                let mut arms = Vec::new();
                loop {
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::RBrace {
                        let end = self.bump().span;
                        return Ok(Expr {
                            kind: ExprKind::Match {
                                scrutinee: Box::new(scrutinee),
                                arms,
                            },
                            span: tok.span.merge(end),
                        });
                    }
                    let arm_start = self.peek().span;
                    let mut patterns = vec![self.parse_pattern()?];
                    while self.eat(&TokenKind::Pipe) {
                        patterns.push(self.parse_pattern()?);
                    }
                    let guard = if self.eat(&TokenKind::If) {
                        Some(self.parse_or()?)
                    } else {
                        None
                    };
                    self.expect(TokenKind::FatArrow)?;
                    let body = self.parse_expr()?;
                    let arm_span = arm_start.merge(body.span);
                    arms.push(MatchArm {
                        patterns,
                        guard,
                        body,
                        span: arm_span,
                    });
                    self.eat_sep();
                }
            }
            TokenKind::LParen => {
                self.bump();
                self.skip_newlines();
                if self.peek().kind == TokenKind::RParen {
                    let end = self.bump().span;
                    return Ok(Expr {
                        kind: ExprKind::Tuple(Vec::new()),
                        span: tok.span.merge(end),
                    });
                }
                let first = self.parse_expr()?;
                if self.eat(&TokenKind::Comma) {
                    let mut elems = vec![first];
                    loop {
                        self.skip_newlines();
                        if self.peek().kind == TokenKind::RParen {
                            let end = self.bump().span;
                            return Ok(Expr {
                                kind: ExprKind::Tuple(elems),
                                span: tok.span.merge(end),
                            });
                        }
                        elems.push(self.parse_expr()?);
                        if !self.eat(&TokenKind::Comma) {
                            let end = self.expect(TokenKind::RParen)?.span;
                            return Ok(Expr {
                                kind: ExprKind::Tuple(elems),
                                span: tok.span.merge(end),
                            });
                        }
                    }
                } else {
                    self.expect(TokenKind::RParen)?;
                    Ok(first)
                }
            }
            TokenKind::LBracket => {
                self.bump();
                let mut elems = Vec::new();
                loop {
                    self.skip_newlines();
                    if self.peek().kind == TokenKind::RBracket {
                        let end = self.bump().span;
                        return Ok(Expr {
                            kind: ExprKind::List(elems),
                            span: tok.span.merge(end),
                        });
                    }
                    elems.push(self.parse_expr()?);
                    if !self.eat(&TokenKind::Comma) {
                        let end = self.expect(TokenKind::RBracket)?.span;
                        return Ok(Expr {
                            kind: ExprKind::List(elems),
                            span: tok.span.merge(end),
                        });
                    }
                }
            }
            TokenKind::LBrace => {
                let blk = self.parse_block()?;
                let span = blk.span;
                Ok(Expr {
                    kind: ExprKind::Block(blk),
                    span,
                })
            }
            TokenKind::Ident(s) => {
                self.bump();
                // Check for versioned record literal `Order@v1 { ... }` or `Date { year = 2024, ... }`
                let version = if self.peek().kind == TokenKind::At {
                    self.bump();
                    Some(self.expect_ident()?.0)
                } else {
                    None
                };
                let is_upper = s.chars().next().is_some_and(|c| c.is_ascii_uppercase());
                if is_upper && self.peek().kind == TokenKind::LBrace {
                    self.bump();
                    let mut fields = Vec::new();
                    loop {
                        self.skip_newlines();
                        if self.peek().kind == TokenKind::RBrace {
                            let end = self.bump().span;
                            return Ok(Expr {
                                kind: ExprKind::RecordLit {
                                    name: s,
                                    version,
                                    fields,
                                },
                                span: tok.span.merge(end),
                            });
                        }
                        let (fname, _) = self.expect_ident()?;
                        self.expect(TokenKind::Eq)?;
                        let fval = self.parse_expr()?;
                        fields.push((fname, fval));
                        self.eat_sep();
                    }
                }
                Ok(Expr {
                    kind: ExprKind::Ident(s),
                    span: tok.span,
                })
            }
            _ => Err(ParseError {
                message: format!("expected expression, found {:?}", tok.kind),
                span: tok.span,
            }),
        }
    }

    fn parse_pattern(&mut self) -> Result<Pattern, ParseError> {
        self.skip_newlines();
        let tok = self.peek().clone();
        match tok.kind {
            TokenKind::Int(n) => {
                self.bump();
                Ok(Pattern::Int(n, tok.span))
            }
            TokenKind::Minus => {
                self.bump();
                if let TokenKind::Int(n) = self.peek().kind {
                    let nspan = self.bump().span;
                    Ok(Pattern::Int(-n, tok.span.merge(nspan)))
                } else {
                    Err(ParseError {
                        message: "expected integer after `-` in pattern".into(),
                        span: self.peek().span,
                    })
                }
            }
            TokenKind::True => {
                self.bump();
                Ok(Pattern::Bool(true, tok.span))
            }
            TokenKind::False => {
                self.bump();
                Ok(Pattern::Bool(false, tok.span))
            }
            TokenKind::Str(s) => {
                self.bump();
                Ok(Pattern::Str(s, tok.span))
            }
            TokenKind::Ident(s) => {
                self.bump();
                if s == "_" {
                    return Ok(Pattern::Wildcard(tok.span));
                }
                let mut full_name = s;
                let mut end = tok.span;
                while self.peek().kind == TokenKind::Dot {
                    self.bump();
                    let (part, pspan) = self.expect_ident()?;
                    full_name.push('.');
                    full_name.push_str(&part);
                    end = pspan;
                }
                if self.peek().kind == TokenKind::LParen {
                    self.bump();
                    let (bind, _) = self.expect_ident()?;
                    end = self.expect(TokenKind::RParen)?.span;
                    Ok(Pattern::Variant {
                        name: full_name,
                        binding: Some(bind),
                        span: tok.span.merge(end),
                    })
                } else if full_name.contains('.')
                    || full_name.chars().next().is_some_and(|c| c.is_uppercase())
                {
                    Ok(Pattern::Variant {
                        name: full_name,
                        binding: None,
                        span: tok.span.merge(end),
                    })
                } else {
                    Ok(Pattern::Ident(full_name, tok.span))
                }
            }
            _ => Err(ParseError {
                message: format!("expected pattern, found {:?}", tok.kind),
                span: tok.span,
            }),
        }
    }
}
