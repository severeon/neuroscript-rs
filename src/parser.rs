//! NeuroScript Parser
//!
//! Recursive descent parser with good error messages.

use crate::interfaces::*;
use crate::interfaces::{Lexer, Token, TokenKind};
use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

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
        let mut body = None;

        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            self.skip_newlines();

            match self.peek_kind() {
                TokenKind::In => {
                    inputs.push(self.port(true)?);
                }
                TokenKind::Out => {
                    outputs.push(self.port(false)?);
                }
                TokenKind::Impl => {
                    body = Some(NeuronBody::Primitive(self.impl_body()?));
                }
                TokenKind::Graph => {
                    body = Some(NeuronBody::Graph(self.graph_body()?));
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

        let body = body.ok_or_else(|| ParseError::Expected {
            expected: "impl or graph".to_string(),
            found: "end of neuron".to_string(),
            span: self.peek().span.into(),
        })?;

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

    // in [name]: shape
    fn port(&mut self, is_input: bool) -> Result<Port, ParseError> {
        if is_input {
            self.expect(&TokenKind::In)?;
        } else {
            self.expect(&TokenKind::Out)?;
        }

        // Optional port name
        let name = if self.at(&TokenKind::Ident("".into())) && self.peek_at_colon() {
            let n = self.ident()?;
            n
        } else {
            "default".to_string()
        };

        self.expect(&TokenKind::Colon)?;
        let shape = self.shape()?;
        self.expect(&TokenKind::Newline)?;

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

    // impl: source,path
    fn impl_body(&mut self) -> Result<ImplRef, ParseError> {
        self.expect(&TokenKind::Impl)?;
        self.expect(&TokenKind::Colon)?;

        if self.at(&TokenKind::External) {
            self.advance();
            self.expect(&TokenKind::LParen)?;
            let (_, kwargs) = self.call_args()?;
            self.expect(&TokenKind::RParen)?;
            self.expect(&TokenKind::Newline)?;
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

            self.expect(&TokenKind::Newline)?;
            Ok(ImplRef::Source {
                source,
                path: path_parts.join("/"),
            })
        }
    }

    // graph: connections...
    fn graph_body(&mut self) -> Result<Vec<Connection>, ParseError> {
        self.expect(&TokenKind::Graph)?;
        self.expect(&TokenKind::Colon)?;
        self.expect(&TokenKind::Newline)?;
        self.expect(&TokenKind::Indent)?;

        let mut connections = vec![];

        while !self.at(&TokenKind::Dedent) && !self.at(&TokenKind::Eof) {
            self.skip_newlines();
            if self.at(&TokenKind::Dedent) {
                break;
            }

            let conns = self.connection()?;
            connections.extend(conns);
        }

        if self.at(&TokenKind::Dedent) {
            self.advance();
        }

        Ok(connections)
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

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Use Statement Tests =====

    #[test]
    fn test_parse_use_simple() {
        let source = "use core,nn/*\n";
        let program = Parser::parse(source).unwrap();
        assert_eq!(program.uses.len(), 1);
        assert_eq!(program.uses[0].source, "core");
        assert_eq!(program.uses[0].path, vec!["nn", "*"]);
    }

    #[test]
    fn test_parse_use_nested_path() {
        let source = "use stdlib,blocks/attention/MultiHead\n";
        let program = Parser::parse(source).unwrap();
        assert_eq!(program.uses.len(), 1);
        assert_eq!(program.uses[0].source, "stdlib");
        assert_eq!(program.uses[0].path, vec!["blocks", "attention", "MultiHead"]);
    }

    #[test]
    fn test_parse_multiple_uses() {
        let source = r#"
use core,nn/*
use stdlib,blocks/*
"#;
        let program = Parser::parse(source).unwrap();
        assert_eq!(program.uses.len(), 2);
    }

    // ===== Neuron Definition Tests =====

    #[test]
    fn test_parse_simple_neuron() {
        let source = r#"
neuron Linear(in_dim, out_dim):
  in: [*, in_dim]
  out: [*, out_dim]
  impl: core,nn/Linear
"#;
        let program = Parser::parse(source).unwrap();
        assert_eq!(program.neurons.len(), 1);
        assert!(program.neurons.contains_key("Linear"));
        
        let linear = &program.neurons["Linear"];
        assert_eq!(linear.name, "Linear");
        assert_eq!(linear.params.len(), 2);
        assert_eq!(linear.inputs.len(), 1);
        assert_eq!(linear.outputs.len(), 1);
    }

    #[test]
    fn test_parse_neuron_no_params() {
        let source = r#"
neuron Identity:
  in: [*shape]
  out: [*shape]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let identity = &program.neurons["Identity"];
        assert_eq!(identity.params.len(), 0);
    }

    #[test]
    fn test_parse_neuron_with_defaults() {
        let source = r#"
neuron Conv(in_ch, out_ch, kernel=3, stride=1):
  in: [*, in_ch, *, *]
  out: [*, out_ch, *, *]
  impl: core,nn/Conv2d
"#;
        let program = Parser::parse(source).unwrap();
        let conv = &program.neurons["Conv"];
        assert_eq!(conv.params.len(), 4);
        assert!(conv.params[0].default.is_none());
        assert!(conv.params[1].default.is_none());
        assert!(conv.params[2].default.is_some());
        assert!(conv.params[3].default.is_some());
    }

    #[test]
    fn test_duplicate_neuron_error() {
        let source = r#"
neuron Foo:
  in: [*x]
  out: [*x]
  impl: core,builtin/Identity

neuron Foo:
  in: [*y]
  out: [*y]
  impl: core,builtin/Identity
"#;
        let result = Parser::parse(source);
        assert!(result.is_err());
        match result.unwrap_err() {
            ParseError::DuplicateNeuron { name, .. } => assert_eq!(name, "Foo"),
            _ => panic!("Expected DuplicateNeuron error"),
        }
    }

    // ===== Port Tests =====

    #[test]
    fn test_parse_named_ports() {
        let source = r#"
neuron MultiPort:
  in query: [*, dim]
  in key: [*, dim]
  out attention: [*, dim]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["MultiPort"];
        assert_eq!(neuron.inputs.len(), 2);
        assert_eq!(neuron.inputs[0].name, "query");
        assert_eq!(neuron.inputs[1].name, "key");
        assert_eq!(neuron.outputs[0].name, "attention");
    }

    #[test]
    fn test_parse_default_port_name() {
        let source = r#"
neuron Simple:
  in: [*, dim]
  out: [*, dim]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Simple"];
        assert_eq!(neuron.inputs[0].name, "default");
        assert_eq!(neuron.outputs[0].name, "default");
    }

    // ===== Shape and Dimension Tests =====

    #[test]
    fn test_parse_shape_wildcard() {
        let source = r#"
neuron Test:
  in: [*, *, *]
  out: [*, *, *]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert_eq!(neuron.inputs[0].shape.dims.len(), 3);
        assert!(matches!(neuron.inputs[0].shape.dims[0], Dim::Wildcard));
    }

    #[test]
    fn test_parse_shape_literal() {
        let source = r#"
neuron Test:
  in: [32, 224, 224, 3]
  out: [32, 1000]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert_eq!(neuron.inputs[0].shape.dims.len(), 4);
        assert!(matches!(neuron.inputs[0].shape.dims[0], Dim::Literal(32)));
        assert!(matches!(neuron.inputs[0].shape.dims[1], Dim::Literal(224)));
    }

    #[test]
    fn test_parse_shape_named() {
        let source = r#"
neuron Test:
  in: [batch, height, width, channels]
  out: [batch, classes]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert!(matches!(&neuron.inputs[0].shape.dims[0], Dim::Named(s) if s == "batch"));
        assert!(matches!(&neuron.inputs[0].shape.dims[1], Dim::Named(s) if s == "height"));
    }

    #[test]
    fn test_parse_shape_variadic() {
        let source = r#"
neuron Test:
  in: [*shape]
  out: [*shape]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert!(matches!(&neuron.inputs[0].shape.dims[0], Dim::Variadic(s) if s == "shape"));
    }

    #[test]
    fn test_parse_shape_expr() {
        let source = r#"
neuron Test:
  in: [batch, dim]
  out: [batch, dim*2]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert!(matches!(&neuron.outputs[0].shape.dims[1], Dim::Expr(_)));
    }

    #[test]
    fn test_parse_negative_dimension() {
        let source = r#"
neuron Test:
  in: [batch, -1]
  out: [batch, dim]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        // Negative numbers are parsed as expressions
        assert!(matches!(&neuron.inputs[0].shape.dims[1], Dim::Expr(_)));
    }

    #[test]
    fn test_parse_empty_shape() {
        let source = r#"
neuron Scalar:
  in: []
  out: []
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Scalar"];
        assert_eq!(neuron.inputs[0].shape.dims.len(), 0);
    }

    // ===== Impl Body Tests =====

    #[test]
    fn test_parse_impl_source() {
        let source = r#"
neuron Test:
  in: [*x]
  out: [*x]
  impl: core,nn/Linear
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Primitive(ImplRef::Source { source, path }) => {
                assert_eq!(source, "core");
                assert_eq!(path, "nn/Linear");
            }
            _ => panic!("Expected primitive impl"),
        }
    }

    #[test]
    fn test_parse_impl_external() {
        let source = r#"
neuron Test:
  in: [*x]
  out: [*x]
  impl: external(module=`torch.nn`, class=`Linear`)
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Primitive(ImplRef::External { kwargs }) => {
                assert_eq!(kwargs.len(), 2);
            }
            _ => panic!("Expected external impl"),
        }
    }

    // ===== Graph Body Tests =====

    #[test]
    fn test_parse_simple_graph() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert!(neuron.is_composite());
        match &neuron.body {
            NeuronBody::Graph(connections) => {
                assert_eq!(connections.len(), 2);
            }
            _ => panic!("Expected graph body"),
        }
    }

    #[test]
    fn test_parse_indented_pipeline() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in ->
      Linear(dim, dim*2)
      ReLU()
      Linear(dim*2, dim)
      out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Graph(connections) => {
                assert_eq!(connections.len(), 4);
            }
            _ => panic!("Expected graph body"),
        }
    }

    #[test]
    fn test_parse_fork_and_tuple() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Fork() -> (a, b)
    (a, b) -> Add() -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Graph(connections) => {
                assert_eq!(connections.len(), 4);
                // Check tuple endpoint
                assert!(matches!(&connections[1].destination, Endpoint::Tuple(_)));
                assert!(matches!(&connections[2].source, Endpoint::Tuple(_)));
            }
            _ => panic!("Expected graph body"),
        }
    }

    #[test]
    fn test_parse_port_access() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> node.port -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Graph(connections) => {
                match &connections[0].destination {
                    Endpoint::Ref(port_ref) => {
                        assert_eq!(port_ref.node, "node");
                        assert_eq!(port_ref.port, "port");
                    }
                    _ => panic!("Expected port ref"),
                }
            }
            _ => panic!("Expected graph body"),
        }
    }

    #[test]
    fn test_parse_in_out_keywords_as_refs() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Graph(connections) => {
                assert!(matches!(&connections[0].source, Endpoint::Ref(r) if r.node == "in"));
                assert!(matches!(&connections[1].destination, Endpoint::Ref(r) if r.node == "out"));
            }
            _ => panic!("Expected graph body"),
        }
    }

    // ===== Match Expression Tests =====

    #[test]
    fn test_parse_match_simple() {
        let source = r#"
neuron Identity:
  in: [*shape]
  out: [*shape]
  impl: core,builtin/Identity

neuron Linear(in_dim, out_dim):
  in: [*, in_dim]
  out: [*, out_dim]
  impl: core,nn/Linear

neuron SimpleMatch:
  in: [*, dim]
  out: [*, 512]
  graph:
    in -> match:
      [*, 512]: Identity() -> out
      [*, 256]: Linear(256, 512) -> out
      [*, 1024]: Linear(1024, 512) -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["SimpleMatch"];
        match &neuron.body {
            NeuronBody::Graph(connections) => {
                match &connections[0].destination {
                    Endpoint::Match(match_expr) => {
                        assert_eq!(match_expr.arms.len(), 3);
                        // Check first arm
                        assert_eq!(match_expr.arms[0].pattern.dims.len(), 2);
                        assert_eq!(match_expr.arms[0].pipeline.len(), 2);
                    }
                    _ => panic!("Expected match endpoint"),
                }
            }
            _ => panic!("Expected graph body"),
        }
    }

    #[test]
    fn test_parse_match_with_wildcard() {
        let source = r#"
neuron Identity:
  in: [*shape]
  out: [*shape]
  impl: core,builtin/Identity

neuron Linear(in_dim, out_dim):
  in: [*, in_dim]
  out: [*, out_dim]
  impl: core,nn/Linear

neuron MatchWithWildcard:
  in: [*, dim]
  out: [*, 512]
  graph:
    in -> match:
      [*, 512]: Identity() -> out
      [*, d]: Linear(d, 512) -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["MatchWithWildcard"];
        match &neuron.body {
            NeuronBody::Graph(connections) => {
                match &connections[0].destination {
                    Endpoint::Match(match_expr) => {
                        assert_eq!(match_expr.arms.len(), 2);
                        // Second arm should have named dimension
                        assert!(matches!(&match_expr.arms[1].pattern.dims[1], Dim::Named(s) if s == "d"));
                    }
                    _ => panic!("Expected match endpoint"),
                }
            }
            _ => panic!("Expected graph body"),
        }
    }

    #[test]
    fn test_parse_match_with_guard() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> match:
      [*, d] where d > 512: Linear(d, 512) -> out
      [*, d]: Identity() -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Graph(connections) => {
                match &connections[0].destination {
                    Endpoint::Match(match_expr) => {
                        assert!(match_expr.arms[0].guard.is_some());
                        assert!(match_expr.arms[1].guard.is_none());
                    }
                    _ => panic!("Expected match endpoint"),
                }
            }
            _ => panic!("Expected graph body"),
        }
    }

    // ===== Expression Tests =====

    #[test]
    fn test_parse_expr_literals() {
        let source = r#"
neuron Test(a=42, b=3.14, c=`hello`, d=true, e=false):
  in: [*x]
  out: [*x]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert!(matches!(neuron.params[0].default, Some(Value::Int(42))));
        assert!(matches!(neuron.params[1].default, Some(Value::Float(_))));
        assert!(matches!(neuron.params[2].default, Some(Value::String(_))));
        assert!(matches!(neuron.params[3].default, Some(Value::Bool(true))));
        assert!(matches!(neuron.params[4].default, Some(Value::Bool(false))));
    }

    #[test]
    fn test_parse_expr_arithmetic() {
        let source = r#"
neuron Test(a=1+2, b=3*4, c=10-5, d=20/4):
  in: [*x]
  out: [*x]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert!(matches!(neuron.params[0].default, Some(Value::BinOp { .. })));
    }

    #[test]
    fn test_parse_expr_comparison() {
        let source = r#"
neuron Test(a=x>5, b=y<10, c=z==3, d=w!=0, e=a>=1, f=b<=2):
  in: [*x]
  out: [*x]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        for param in &neuron.params {
            assert!(matches!(param.default, Some(Value::BinOp { .. })));
        }
    }

    #[test]
    fn test_parse_expr_negative() {
        let source = r#"
neuron Test(a=-42, b=-3.14):
  in: [*x]
  out: [*x]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        // Negative numbers are parsed as binary operations (0 - value)
        assert!(matches!(neuron.params[0].default, Some(Value::BinOp { .. })));
    }

    #[test]
    fn test_parse_expr_function_call() {
        let source = r#"
neuron Test(a=sqrt(16), b=max(1, 2, 3)):
  in: [*x]
  out: [*x]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert!(matches!(neuron.params[0].default, Some(Value::Call { .. })));
        assert!(matches!(neuron.params[1].default, Some(Value::Call { .. })));
    }

    #[test]
    fn test_parse_expr_parentheses() {
        let source = r#"
neuron Test(a=(1+2)*3):
  in: [*x]
  out: [*x]
  impl: core,builtin/Identity
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        assert!(matches!(neuron.params[0].default, Some(Value::BinOp { .. })));
    }

    // ===== Call Arguments Tests =====

    #[test]
    fn test_parse_call_positional_args() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(256, 512) -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Graph(connections) => {
                match &connections[0].destination {
                    Endpoint::Call { args, kwargs, .. } => {
                        assert_eq!(args.len(), 2);
                        assert_eq!(kwargs.len(), 0);
                    }
                    _ => panic!("Expected call endpoint"),
                }
            }
            _ => panic!("Expected graph body"),
        }
    }

    #[test]
    fn test_parse_call_keyword_args() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Conv(in_ch=3, out_ch=64, kernel=3) -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Graph(connections) => {
                match &connections[0].destination {
                    Endpoint::Call { args, kwargs, .. } => {
                        assert_eq!(args.len(), 0);
                        assert_eq!(kwargs.len(), 3);
                        assert_eq!(kwargs[0].0, "in_ch");
                        assert_eq!(kwargs[1].0, "out_ch");
                        assert_eq!(kwargs[2].0, "kernel");
                    }
                    _ => panic!("Expected call endpoint"),
                }
            }
            _ => panic!("Expected graph body"),
        }
    }

    #[test]
    fn test_parse_call_mixed_args() {
        let source = r#"
neuron Test:
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Conv(3, 64, kernel=3, stride=1) -> out
"#;
        let program = Parser::parse(source).unwrap();
        let neuron = &program.neurons["Test"];
        match &neuron.body {
            NeuronBody::Graph(connections) => {
                match &connections[0].destination {
                    Endpoint::Call { args, kwargs, .. } => {
                        assert_eq!(args.len(), 2);
                        assert_eq!(kwargs.len(), 2);
                    }
                    _ => panic!("Expected call endpoint"),
                }
            }
            _ => panic!("Expected graph body"),
        }
    }

    // ===== Integration Tests =====

    #[test]
    fn test_parse_residual_example() {
        let source = std::fs::read_to_string("examples/residual.ns").unwrap();
        let program = Parser::parse(&source).unwrap();

        assert_eq!(program.neurons.len(), 6);
        assert_eq!(program.uses.len(), 1);

        // Get the Residual and MLP neurons
        let residual = &program.neurons["Residual"];
        let mlp = &program.neurons["MLP"];

        // Check that Residual and MLP are composite (have graphs)
        assert!(residual.is_composite());
        assert!(mlp.is_composite());

        // Extract connections from the graphs
        let NeuronBody::Graph(residual_connections) = &residual.body else {
            panic!("Expected graph body for Residual");
        };
        let NeuronBody::Graph(mlp_connections) = &mlp.body else {
            panic!("Expected graph body for MLP");
        };

        // Assert Residual has 6 connections
        assert_eq!(residual_connections.len(), 6);

        // Assert MLP has 4 connections
        assert_eq!(mlp_connections.len(), 4);

        // Assert the following 6 connections for Residual:
        // 1. Ref(PortRef { node: "in", port: "default" }) -> Call { name: "Fork", args: [], kwargs: [] }
        assert_eq!(residual_connections[0].source, Endpoint::Ref(PortRef::new("in")));
        assert_eq!(residual_connections[0].destination, Endpoint::Call {
            name: "Fork".to_string(),
            args: vec![],
            kwargs: vec![],
            id: 3
        });

        // 2. Call { name: "Fork", args: [], kwargs: [] } -> Tuple([PortRef { node: "main", port: "default" }, PortRef { node: "skip", port: "default" }])
        assert_eq!(residual_connections[1].source, Endpoint::Call {
            name: "Fork".to_string(),
            args: vec![],
            kwargs: vec![],
            id: 3
        });
        assert_eq!(residual_connections[1].destination, Endpoint::Tuple(vec![
            PortRef::new("main"),
            PortRef::new("skip")
        ]));

        // 3. Ref(PortRef { node: "main", port: "default" }) -> Call { name: "MLP", args: [Name("dim")], kwargs: [] }
        assert_eq!(residual_connections[2].source, Endpoint::Ref(PortRef::new("main")));
        assert_eq!(residual_connections[2].destination, Endpoint::Call {
            name: "MLP".to_string(),
            args: vec![Value::Name("dim".to_string())],
            kwargs: vec![],
            id: 4
        });

        // 4. Call { name: "MLP", args: [Name("dim")], kwargs: [] } -> Ref(PortRef { node: "processed", port: "default" })
        assert_eq!(residual_connections[3].source, Endpoint::Call {
            name: "MLP".to_string(),
            args: vec![Value::Name("dim".to_string())],
            kwargs: vec![],
            id: 4
        });
        assert_eq!(residual_connections[3].destination, Endpoint::Ref(PortRef::new("processed")));

        // 5. Tuple([PortRef { node: "processed", port: "default" }, PortRef { node: "skip", port: "default" }]) -> Call { name: "Add", args: [], kwargs: [] }
        assert_eq!(residual_connections[4].source, Endpoint::Tuple(vec![
            PortRef::new("processed"),
            PortRef::new("skip")
        ]));
        assert_eq!(residual_connections[4].destination, Endpoint::Call {
            name: "Add".to_string(),
            args: vec![],
            kwargs: vec![],
            id: 5
        });

        // 6. Call { name: "Add", args: [], kwargs: [] } -> Ref(PortRef { node: "out", port: "default" })
        assert_eq!(residual_connections[5].source, Endpoint::Call {
            name: "Add".to_string(),
            args: vec![],
            kwargs: vec![],
            id: 5
        });
        assert_eq!(residual_connections[5].destination, Endpoint::Ref(PortRef::new("out")));
    }

    // ===== Error Handling Tests =====

    #[test]
    fn test_error_unexpected_token() {
        let source = "invalid token sequence\n";
        let result = Parser::parse(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_missing_colon() {
        let source = r#"
neuron Test
  in: [*x]
  out: [*x]
  impl: core,builtin/Identity
"#;
        let result = Parser::parse(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_missing_impl() {
        let source = r#"
neuron Test:
  in: [*x]
  out: [*x]
"#;
        let result = Parser::parse(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unclosed_bracket() {
        let source = r#"
neuron Test:
  in: [*, dim
  out: [*, dim]
  impl: core,builtin/Identity
"#;
        let result = Parser::parse(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unclosed_paren() {
        let source = r#"
neuron Test(dim:
  in: [*, dim]
  out: [*, dim]
  impl: core,builtin/Identity
"#;
        let result = Parser::parse(source);
        assert!(result.is_err());
    }
}
