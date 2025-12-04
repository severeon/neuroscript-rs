//! NeuroScript Parser Core
//!
//! Recursive descent parser implementation with good error messages.

use crate::interfaces::*;
use crate::interfaces::{Lexer, Token, TokenKind};
use miette::SourceSpan;

impl ParseError {
    pub fn span(&self) -> SourceSpan {
        match self {
            ParseError::Expected { span, .. } => *span,
            ParseError::Unexpected { span, .. } => *span,
            ParseError::DuplicateNeuron { span, .. } => *span,
            ParseError::Lex(e) => e.span(),
        }
    }

    pub fn expected(&self) -> &str {
        match self {
            ParseError::Expected { expected, .. } => expected,
            ParseError::Unexpected { .. } => "",
            ParseError::DuplicateNeuron { .. } => "",
            ParseError::Lex(_) => "",
        }
    }

    pub fn found(&self) -> String {
        match self {
            ParseError::Expected { found, .. } => found.clone(),
            ParseError::Unexpected { found, .. } => found.clone(),
            ParseError::DuplicateNeuron { name, .. } => format!("neuron '{}'", name),
            ParseError::Lex(e) => e.to_string(),
        }
    }
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0, next_node_id: 0 }
    }

    fn next_id(&mut self) -> usize {
        let id = self.next_node_id;
        self.next_node_id += 1;
        id
    }

    pub fn parse(source: &str) -> Result<Program, ParseError> {
        let tokens = Lexer::new(source).tokenize()?;
        // print!("{:?}", tokens);
        let mut parser = Parser::new(tokens);
        parser.program()
    }

    // === Core parsing infrastructure ===

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&self.tokens[self.tokens.len() - 1])
    }

    fn peek_kind(&self) -> &TokenKind {
        &self.peek().kind
    }

    fn at(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(self.peek_kind()) == std::mem::discriminant(kind)
    }

    fn at_any(&self, kinds: &[TokenKind]) -> bool {
        kinds.iter().any(|k| self.at(k))
    }

    fn peek_at_offset(&self, offset: usize, kind: &TokenKind) -> bool {
        self.tokens
            .get(self.pos + offset)
            .map(|t| std::mem::discriminant(&t.kind) == std::mem::discriminant(kind))
            .unwrap_or(false)
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos];
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<&Token, ParseError> {
        if self.at(kind) {
            Ok(self.advance())
        } else {
            Err(ParseError::Expected {
                expected: format!("{:?}", kind),
                found: format!("{:?}", self.peek_kind()),
                span: self.peek().span.into(),
            })
        }
    }

    fn skip_newlines(&mut self) {
        while self.at(&TokenKind::Newline) {
            self.advance();
        }
    }

    // === Section parsing infrastructure ===

    /// Generic section parser that handles both inline and indented styles.
    ///
    /// Inline style:   `keyword: item`
    /// Indented style: `keyword:\n  item1\n  item2`
    ///
    /// Returns either a single item (inline) or multiple items (indented).
    fn parse_section<F, T>(
        &mut self,
        keyword: &TokenKind,
        mut parse_item: F,
    ) -> Result<Vec<T>, ParseError>
    where
        F: FnMut(&mut Self) -> Result<T, ParseError>,
    {
        self.expect(keyword)?;
        self.expect(&TokenKind::Colon)?;

        // Check for inline vs indented style
        if self.at(&TokenKind::Newline) {
            self.advance();

            // Empty section (no items)
            if !self.at(&TokenKind::Indent) {
                return Ok(vec![]);
            }

            // Indented style: parse multiple items
            self.expect(&TokenKind::Indent)?;
            let mut items = vec![];

            while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
                self.skip_newlines();
                if self.at(&TokenKind::Dedent) {
                    break;
                }
                items.push(parse_item(self)?);
            }

            if self.at(&TokenKind::Dedent) {
                self.advance();
            }

            Ok(items)
        } else {
            // Inline style: parse single item
            let item = parse_item(self)?;
            self.expect(&TokenKind::Newline)?;
            Ok(vec![item])
        }
    }

    /// Parse a section that should contain exactly one item.
    /// Used for sections like 'impl' where only one value is expected.
    fn parse_single_section<F, T>(
        &mut self,
        keyword: &TokenKind,
        parse_item: F,
    ) -> Result<T, ParseError>
    where
        F: FnMut(&mut Self) -> Result<T, ParseError>,
    {
        let items = self.parse_section(keyword, parse_item)?;

        if items.len() != 1 {
            return Err(ParseError::Unexpected {
                found: format!("{} items in section that expects exactly one", items.len()),
                span: self.peek().span.into(),
            });
        }

        Ok(items.into_iter().next().unwrap())
    }

    // === Grammar rules ===

    fn program(&mut self) -> Result<Program, ParseError> {
        let mut program = Program::new();

        self.skip_newlines();

        while !self.at(&TokenKind::Eof) {
            match self.peek_kind() {
                TokenKind::Use => {
                    let use_stmt = self.use_stmt()?;
                    program.uses.push(use_stmt);
                }
                TokenKind::Neuron => {
                    let neuron = self.neuron_def()?;
                    if program.neurons.contains_key(&neuron.name) {
                        return Err(ParseError::DuplicateNeuron {
                            name: neuron.name.clone(),
                            span: self.peek().span.into(),
                        });
                    }
                    program.neurons.insert(neuron.name.clone(), neuron);
                }
                TokenKind::Newline => {
                    self.advance();
                }
                _ => {
                    return Err(ParseError::Unexpected {
                        found: format!("{:?}", self.peek_kind()),
                        span: self.peek().span.into(),
                    });
                }
            }
        }

        Ok(program)
    }

    // use core,nn/*
    fn use_stmt(&mut self) -> Result<UseStmt, ParseError> {
        self.expect(&TokenKind::Use)?;

        let source = self.ident()?;
        self.expect(&TokenKind::Comma)?;

        let mut path = vec![];
        loop {
            if self.at(&TokenKind::Star) {
                self.advance();
                path.push("*".to_string());
            } else {
                path.push(self.ident()?);
            }

            if self.at(&TokenKind::Slash) {
                self.advance();
            } else {
                break;
            }
        }

        self.expect(&TokenKind::Newline)?;

        Ok(UseStmt { source, path })
    }

    // neuron Name(params): ...
    fn neuron_def(&mut self) -> Result<NeuronDef, ParseError> {
        self.expect(&TokenKind::Neuron)?;

        let name = self.ident()?;

        // Optional parameters
        let params = if self.at(&TokenKind::LParen) {
            self.params()?
        } else {
            vec![]
        };

        self.expect(&TokenKind::Colon)?;
        self.expect(&TokenKind::Newline)?;
        self.expect(&TokenKind::Indent)?;

        // Parse ports and body
        let mut inputs = vec![];
        let mut outputs = vec![];
        let mut let_bindings = vec![];
        let mut set_bindings = vec![];
        let mut graph_connections = None;
        let mut impl_ref = None;

        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            self.skip_newlines();

            match self.peek_kind() {
                TokenKind::In => {
                    // Support both old syntax (in name: [shape]) and new syntax (in: ...)
                    if self.peek_at_offset(1, &TokenKind::Colon) {
                        // New section-based syntax: in: ...
                        let ports = self.parse_section(&TokenKind::In, |p| p.port_item())?;
                        inputs.extend(ports);
                    } else {
                        // Old syntax: in name: [shape] (single port, no colon after keyword)
                        self.advance(); // consume 'in'
                        let port = self.port_item()?;
                        self.expect(&TokenKind::Newline)?;
                        inputs.push(port);
                    }
                }
                TokenKind::Out => {
                    // Support both old syntax (out name: [shape]) and new syntax (out: ...)
                    if self.peek_at_offset(1, &TokenKind::Colon) {
                        // New section-based syntax: out: ...
                        let ports = self.parse_section(&TokenKind::Out, |p| p.port_item())?;
                        outputs.extend(ports);
                    } else {
                        // Old syntax: out name: [shape] (single port, no colon after keyword)
                        self.advance(); // consume 'out'
                        let port = self.port_item()?;
                        self.expect(&TokenKind::Newline)?;
                        outputs.push(port);
                    }
                }
                TokenKind::Let => {
                    let bindings = self.parse_section(&TokenKind::Let, |p| p.parse_binding())?;
                    let_bindings.extend(bindings);
                }
                TokenKind::Set => {
                    let bindings = self.parse_section(&TokenKind::Set, |p| p.parse_binding())?;
                    set_bindings.extend(bindings);
                }
                TokenKind::Impl => {
                    let impl_ref_item = self.parse_single_section(&TokenKind::Impl, |p| p.impl_item())?;
                    impl_ref = Some(impl_ref_item);
                }
                TokenKind::Graph => {
                    graph_connections = Some(self.graph_body()?);
                }
                TokenKind::Dedent => break,
                _ => {
                    return Err(ParseError::Unexpected {
                        found: format!("{:?}", self.peek_kind()),
                        span: self.peek().span.into(),
                    });
                }
            }
        }

        if self.at(&TokenKind::Dedent) {
            self.advance();
        }

        // Construct body based on what we found
        let body = if let Some(impl_ref_item) = impl_ref {
            NeuronBody::Primitive(impl_ref_item)
        } else if let Some(connections) = graph_connections {
            NeuronBody::Graph {
                let_bindings,
                set_bindings,
                connections,
            }
        } else {
            // If no impl or graph but we have bindings, treat as graph with empty connections
            if !let_bindings.is_empty() || !set_bindings.is_empty() {
                NeuronBody::Graph {
                    let_bindings,
                    set_bindings,
                    connections: vec![],
                }
            } else {
                return Err(ParseError::Expected {
                    expected: "impl or graph".to_string(),
                    found: "end of neuron".to_string(),
                    span: self.peek().span.into(),
                });
            }
        };

        Ok(NeuronDef {
            name,
            params,
            inputs,
            outputs,
            body,
        })
    }

    // (param1, param2=default)
    fn params(&mut self) -> Result<Vec<Param>, ParseError> {
        self.expect(&TokenKind::LParen)?;

        let mut params = vec![];

        if !self.at(&TokenKind::RParen) {
            loop {
                let name = self.ident()?;
                let default = if self.at(&TokenKind::Assign) {
                    self.advance();
                    Some(self.expr()?)
                } else {
                    None
                };
                params.push(Param { name, default });

                if self.at(&TokenKind::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        self.expect(&TokenKind::RParen)?;
        Ok(params)
    }

    // Parse port content: [shape] or name: [shape]
    // Used by parse_section for both inline and indented styles
    // Note: The keyword and its colon are already consumed by parse_section
    // Note: The newline is handled by parse_section, not here
    fn port_item(&mut self) -> Result<Port, ParseError> {
        // Check if we have a named port (name: [shape]) or unnamed ([shape])
        let name = if self.at(&TokenKind::Ident("".into())) && self.peek_at_colon() {
            let n = self.ident()?;
            self.expect(&TokenKind::Colon)?;
            n
        } else {
            "default".to_string()
        };

        let shape = self.shape()?;

        Ok(Port { name, shape })
    }

    fn peek_at_colon(&self) -> bool {
        self.tokens.get(self.pos + 1).map(|t| t.kind == TokenKind::Colon).unwrap_or(false)
    }

    // [dim, dim, dim]
    fn shape(&mut self) -> Result<Shape, ParseError> {
        self.expect(&TokenKind::LBracket)?;

        let mut dims = vec![];

        if !self.at(&TokenKind::RBracket) {
            loop {
                dims.push(self.dim()?);
                if self.at(&TokenKind::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }

        self.expect(&TokenKind::RBracket)?;
        Ok(Shape { dims })
    }

    // *, *name, name, 512, expr
    fn dim(&mut self) -> Result<Dim, ParseError> {
        // Handle unary minus for negative numbers
        if self.at(&TokenKind::Minus) {
            self.advance();
            let inner = self.dim()?;
            return Ok(Dim::Expr(Box::new(DimExpr {
                op: BinOp::Mul,
                left: Dim::Literal(-1),
                right: inner,
            })));
        }

        if self.at(&TokenKind::Star) {
            self.advance();
            // Check for variadic: *name
            if self.at(&TokenKind::Ident("".into())) {
                let name = self.ident()?;
                Ok(Dim::Variadic(name))
            } else {
                Ok(Dim::Wildcard)
            }
        } else if self.at(&TokenKind::Int(0)) {
            let TokenKind::Int(n) = self.advance().kind.clone() else { unreachable!() };
            Ok(Dim::Literal(n))
        } else if self.at(&TokenKind::Ident("".into())) {
            let name = self.ident()?;

            // Check for binary op
            if self.at(&TokenKind::Star) || self.at(&TokenKind::Plus)
                || self.at(&TokenKind::Minus) || self.at(&TokenKind::Slash)
            {
                let op = self.binop()?;
                let right = self.dim()?;
                Ok(Dim::Expr(Box::new(DimExpr {
                    op,
                    left: Dim::Named(name),
                    right,
                })))
            } else {
                Ok(Dim::Named(name))
            }
        } else {
            Err(ParseError::Expected {
                expected: "dimension".to_string(),
                found: format!("{:?}", self.peek_kind()),
                span: self.peek().span.into(),
            })
        }
    }

    // Parse impl content: source,path or external(...)
    // Used by parse_section for both inline and indented styles
    // Note: The keyword and its colon are already consumed by parse_section
    // Note: The newline is handled by parse_section, not here
    fn impl_item(&mut self) -> Result<ImplRef, ParseError> {
        if self.at(&TokenKind::External) {
            self.advance();
            self.expect(&TokenKind::LParen)?;
            let (_, kwargs) = self.call_args()?;
            self.expect(&TokenKind::RParen)?;
            Ok(ImplRef::External { kwargs })
        } else {
            let source = self.ident()?;
            self.expect(&TokenKind::Comma)?;

            let mut path_parts = vec![];
            loop {
                if self.at(&TokenKind::Star) {
                    self.advance();
                    path_parts.push("*".to_string());
                } else {
                    path_parts.push(self.ident()?);
                }

                if self.at(&TokenKind::Slash) {
                    self.advance();
                } else {
                    break;
                }
            }

            Ok(ImplRef::Source {
                source,
                path: path_parts.join("/"),
            })
        }
    }

    // Parse graph content: one or more connections
    // Used by parse_section for both inline and indented styles
    fn graph_body(&mut self) -> Result<Vec<Connection>, ParseError> {
        // Use parse_section with connection as the item parser
        // This returns Vec<Vec<Connection>> since each connection is a pipeline
        let connection_groups = self.parse_section(&TokenKind::Graph, |p| p.connection())?;

        // Flatten the nested vectors
        Ok(connection_groups.into_iter().flatten().collect())
    }

    // Parse a binding: name = NeuronCall(args)
    fn parse_binding(&mut self) -> Result<Binding, ParseError> {
        let name = self.ident()?;
        self.expect(&TokenKind::Assign)?;

        // Parse the neuron call
        let call_name = self.ident()?;
        self.expect(&TokenKind::LParen)?;
        let (args, kwargs) = self.call_args()?;
        self.expect(&TokenKind::RParen)?;
        self.expect(&TokenKind::Newline)?;

        Ok(Binding {
            name,
            call_name,
            args,
            kwargs,
        })
    }

    // endpoint -> endpoint [-> endpoint...]
    fn connection(&mut self) -> Result<Vec<Connection>, ParseError> {
        let first = self.endpoint()?;
        self.expect(&TokenKind::Arrow)?;

        // Check for indented pipeline
        if self.at(&TokenKind::Newline) {
            self.advance();
            if self.at(&TokenKind::Indent) {
                self.advance();
                return self.indented_pipeline(first);
            }
        }

        // Inline pipeline
        self.inline_pipeline(first)
    }

    fn inline_pipeline(&mut self, first: Endpoint) -> Result<Vec<Connection>, ParseError> {
        let mut connections = vec![];
        let mut prev = first;

        loop {
            let next = self.endpoint()?;
            connections.push(Connection {
                source: prev.clone(),
                destination: next.clone(),
            });
            prev = next;

            if self.at(&TokenKind::Arrow) {
                self.advance();
            } else {
                break;
            }
        }

        // If the last endpoint was a match expression, it already consumed the newline/block
        if !matches!(prev, Endpoint::Match(_)) {
            self.expect(&TokenKind::Newline)?;
        }
        Ok(connections)
    }

    fn indented_pipeline(&mut self, first: Endpoint) -> Result<Vec<Connection>, ParseError> {
        let mut connections = vec![];
        let mut prev = first;

        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            self.skip_newlines();
            if self.at(&TokenKind::Dedent) {
                break;
            }

            let next = self.endpoint()?;
            connections.push(Connection {
                source: prev.clone(),
                destination: next.clone(),
            });
            prev = next;

            if self.at(&TokenKind::Newline) {
                self.advance();
            }
        }

        if self.at(&TokenKind::Dedent) {
            self.advance();
        }

        Ok(connections)
    }

    // name | name(args) | (a, b) | match: ...
    fn endpoint(&mut self) -> Result<Endpoint, ParseError> {
        if self.at(&TokenKind::LParen) {
            // Tuple
            self.advance();
            let mut refs = vec![];
            loop {
                refs.push(PortRef::new(self.ident()?));
                if self.at(&TokenKind::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect(&TokenKind::RParen)?;
            Ok(Endpoint::Tuple(refs))
        } else if self.at(&TokenKind::Match) {
            Ok(Endpoint::Match(self.match_expr()?))
        } else {
            // Accept 'in' and 'out' keywords as identifiers for port references
            let name = if self.at(&TokenKind::In) {
                self.advance();
                "in".to_string()
            } else if self.at(&TokenKind::Out) {
                self.advance();
                "out".to_string()
            } else {
                self.ident()?
            };

            // Check for port access: name.port
            let name = if self.at(&TokenKind::Dot) {
                self.advance();
                let port = self.ident()?;
                return Ok(Endpoint::Ref(PortRef::with_port(name, port)));
            } else {
                name
            };

            // Check for call: name(args)
            if self.at(&TokenKind::LParen) {
                self.advance();
                let (args, kwargs) = if self.at(&TokenKind::RParen) {
                    (vec![], vec![])
                } else {
                    self.call_args()?
                };
                self.expect(&TokenKind::RParen)?;
                Ok(Endpoint::Call { name, args, kwargs, id: self.next_id() })
            } else {
                Ok(Endpoint::Ref(PortRef::new(name)))
            }
        }
    }

    // match: arms...
    fn match_expr(&mut self) -> Result<MatchExpr, ParseError> {
        self.expect(&TokenKind::Match)?;
        self.expect(&TokenKind::Colon)?;
        
        self.expect(&TokenKind::Newline)?;
        self.expect(&TokenKind::Indent)?;

        let mut arms = vec![];

        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            self.skip_newlines();
            if self.at(&TokenKind::Dedent) {
                break;
            }

            let pattern = self.shape()?;

            let guard = if self.at(&TokenKind::Where) {
                self.advance();
                Some(self.expr()?)
            } else {
                None
            };

            self.expect(&TokenKind::Colon)?;

            // Parse pipeline for this arm
            let mut pipeline = vec![];
            loop {
                pipeline.push(self.endpoint()?);
                if self.at(&TokenKind::Arrow) {
                    self.advance();
                } else {
                    break;
                }
            }

            self.expect(&TokenKind::Newline)?;

            arms.push(MatchArm {
                pattern,
                guard,
                pipeline,
                is_reachable: true, // Default to reachable, validator will mark unreachable
            });
        }

        if self.at(&TokenKind::Dedent) {
            self.advance();
        }

        Ok(MatchExpr { arms })
    }

    // arg, arg, name=arg, ...
    fn call_args(&mut self) -> Result<(Vec<Value>, Vec<(String, Value)>), ParseError> {
        let mut args = vec![];
        let mut kwargs = vec![];

        loop {
            // Check for keyword arg
            if self.at(&TokenKind::Ident("".into())) && self.peek_at_assign() {
                let name = self.ident()?;
                self.expect(&TokenKind::Assign)?;
                let val = self.expr()?;
                kwargs.push((name, val));
            } else {
                args.push(self.expr()?);
            }

            if self.at(&TokenKind::Comma) {
                self.advance();
            } else {
                break;
            }
        }

        Ok((args, kwargs))
    }

    fn peek_at_assign(&self) -> bool {
        self.tokens.get(self.pos + 1).map(|t| t.kind == TokenKind::Assign).unwrap_or(false)
    }

    // Expression parsing
    fn expr(&mut self) -> Result<Value, ParseError> {
        self.comparison()
    }

    fn comparison(&mut self) -> Result<Value, ParseError> {
        let mut left = self.additive()?;
        
        while self.at_any(&[
            TokenKind::Lt, TokenKind::Gt, TokenKind::Le, 
            TokenKind::Ge, TokenKind::Eq, TokenKind::Ne
        ]) {
            let op = self.comparison_op()?;
            let right = self.additive()?;
            left = Value::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        
        Ok(left)
    }

    fn comparison_op(&mut self) -> Result<BinOp, ParseError> {
        let op = match self.peek_kind() {
            TokenKind::Lt => BinOp::Lt,
            TokenKind::Gt => BinOp::Gt,
            TokenKind::Le => BinOp::Le,
            TokenKind::Ge => BinOp::Ge,
            TokenKind::Eq => BinOp::Eq,
            TokenKind::Ne => BinOp::Ne,
            _ => return Err(ParseError::Unexpected {
                found: format!("{:?}", self.peek_kind()),
                span: self.peek().span.into(),
            }),
        };
        self.advance();
        Ok(op)
    }

    fn additive(&mut self) -> Result<Value, ParseError> {
        let mut left = self.multiplicative()?;

        while self.at(&TokenKind::Plus) || self.at(&TokenKind::Minus) {
            let op = self.binop()?;
            let right = self.multiplicative()?;
            left = Value::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn multiplicative(&mut self) -> Result<Value, ParseError> {
        let mut left = self.atom()?;

        while self.at(&TokenKind::Star) || self.at(&TokenKind::Slash) {
            let op = self.binop()?;
            let right = self.atom()?;
            left = Value::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn atom(&mut self) -> Result<Value, ParseError> {
        // Handle unary minus for negative numbers
        if self.at(&TokenKind::Minus) {
            self.advance();
            let inner = self.atom()?;
            return Ok(Value::BinOp {
                op: BinOp::Sub,
                left: Box::new(Value::Int(0)),
                right: Box::new(inner),
            });
        }

        match self.peek_kind().clone() {
            TokenKind::Int(n) => {
                self.advance();
                Ok(Value::Int(n))
            }
            TokenKind::Float(f) => {
                self.advance();
                Ok(Value::Float(f))
            }
            TokenKind::String(s) => {
                self.advance();
                Ok(Value::String(s))
            }
            TokenKind::True => {
                self.advance();
                Ok(Value::Bool(true))
            }
            TokenKind::False => {
                self.advance();
                Ok(Value::Bool(false))
            }
            TokenKind::Ident(name) => {
                self.advance();
                // Check for call
                if self.at(&TokenKind::LParen) {
                    self.advance();
                    let (args, kwargs) = if self.at(&TokenKind::RParen) {
                        (vec![], vec![])
                    } else {
                        self.call_args()?
                    };
                    self.expect(&TokenKind::RParen)?;
                    Ok(Value::Call { name, args, kwargs })
                } else {
                    Ok(Value::Name(name))
                }
            }
            TokenKind::LParen => {
                self.advance();
                let expr = self.expr()?;
                self.expect(&TokenKind::RParen)?;
                Ok(expr)
            }
            _ => Err(ParseError::Expected {
                expected: "expression".to_string(),
                found: format!("{:?}", self.peek_kind()),
                span: self.peek().span.into(),
            }),
        }
    }

    fn binop(&mut self) -> Result<BinOp, ParseError> {
        let op = match self.peek_kind() {
            TokenKind::Plus => BinOp::Add,
            TokenKind::Minus => BinOp::Sub,
            TokenKind::Star => BinOp::Mul,
            TokenKind::Slash => BinOp::Div,
            _ => {
                return Err(ParseError::Expected {
                    expected: "operator".to_string(),
                    found: format!("{:?}", self.peek_kind()),
                    span: self.peek().span.into(),
                })
            }
        };
        self.advance();
        Ok(op)
    }

    fn ident(&mut self) -> Result<String, ParseError> {
        match self.peek_kind().clone() {
            TokenKind::Ident(s) => {
                self.advance();
                Ok(s)
            }
            _ => Err(ParseError::Expected {
                expected: "identifier".to_string(),
                found: format!("{:?}", self.peek_kind()),
                span: self.peek().span.into(),
            }),
        }
    }
}
