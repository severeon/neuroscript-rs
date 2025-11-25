//! NeuroScript Parser
//!
//! Recursive descent parser with good error messages.

use crate::ir::*;
use crate::lexer::{Lexer, Token, TokenKind};
use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

#[derive(Debug, Error, Diagnostic)]
pub enum ParseError {
    #[error("Expected {expected}, found {found}")]
    Expected {
        expected: String,
        found: String,
        #[label("here")]
        span: SourceSpan,
    },

    #[error("Unexpected token '{found}'")]
    Unexpected {
        found: String,
        #[label("unexpected")]
        span: SourceSpan,
    },

    #[error("Duplicate neuron definition '{name}'")]
    DuplicateNeuron {
        name: String,
        #[label("redefined here")]
        span: SourceSpan,
    },

    #[error("{0}")]
    Lex(#[from] crate::lexer::LexError),
}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
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
            println!("Expected {:?}, found {:?} as {:?}", kind, self.peek_kind(), self.peek());
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
                source: prev,
                destination: next.clone(),
            });
            prev = next;

            if self.at(&TokenKind::Arrow) {
                self.advance();
            } else {
                break;
            }
        }

        self.expect(&TokenKind::Newline)?;
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
                source: prev,
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
                Ok(Endpoint::Call { name, args, kwargs })
            } else {
                Ok(Endpoint::Ref(PortRef::new(name)))
            }
        }
    }

    // match: arms...
    fn match_expr(&mut self) -> Result<MatchExpr, ParseError> {
        self.expect(&TokenKind::Match)?;
        self.expect(&TokenKind::Colon)?;
        self.expect(&TokenKind::Dedent)?;
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
        self.additive()
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
    }

    #[test]
    fn test_parse_composite() {
        let source = r#"
neuron MLP(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Linear(dim, dim) -> out
"#;
        let program = Parser::parse(source).unwrap();
        assert_eq!(program.neurons.len(), 1);
        let mlp = &program.neurons["MLP"];
        assert!(mlp.is_composite());
    }

    #[test]
    fn test_parse_use() {
        let source = "use core,nn/*\n";
        let program = Parser::parse(source).unwrap();
        assert_eq!(program.uses.len(), 1);
        assert_eq!(program.uses[0].source, "core");
        assert_eq!(program.uses[0].path, vec!["nn", "*"]);
    }
}
