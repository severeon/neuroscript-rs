//! Core lexer implementation

use crate::interfaces::*;
use miette::SourceSpan;

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Lexer {
            source,
            chars: source.char_indices().peekable(),
            pos: 0,
            line: 1,
            col: 1,
            indent_stack: vec![0],
            pending_dedents: 0,
            at_line_start: true,
        }
    }

    pub fn tokenize(mut self) -> Result<Vec<Token>, LexError> {
        let mut tokens = Vec::new();

        loop {
            let tok = self.next_token()?;
            // println!("DEBUG: Token: {:?}", tok);
            let is_eof = tok.kind == TokenKind::Eof;
            tokens.push(tok);
            if is_eof {
                break;
            }
        }

        Ok(tokens)
    }

    fn next_token(&mut self) -> Result<Token, LexError> {
        // Handle pending dedents
        if self.pending_dedents > 0 {
            self.pending_dedents -= 1;
            return Ok(self.make_token(TokenKind::Dedent, ""));
        }

        // Handle indentation at line start
        if self.at_line_start {
            self.at_line_start = false;
            if let Some(tok) = self.handle_indent()? {
                return Ok(tok);
            }
        }

        // Skip whitespace (but not newlines)
        self.skip_horizontal_whitespace();

        // Skip comments
        if self.peek() == Some('#') {
            self.skip_line();
            return self.handle_newline();
        }

        let start_pos = self.pos;
        let start_line = self.line;
        let start_col = self.col;

        let Some(ch) = self.advance() else {
            // println!("DEBUG: Lexer reached EOF at pos {}", self.pos);
            // EOF - emit any remaining dedents
            while self.indent_stack.len() > 1 {
                self.indent_stack.pop();
                self.pending_dedents += 1;
            }
            if self.pending_dedents > 0 {
                self.pending_dedents -= 1;
                return Ok(self.make_token(TokenKind::Dedent, ""));
            }
            return Ok(Token {
                kind: TokenKind::Eof,
                span: Span {
                    start: start_pos,
                    end: start_pos,
                    line: start_line,
                    col: start_col,
                },
                text: String::new(),
            });
        };

        let kind = match ch {
            '\n' => return self.handle_newline(),

            '(' => TokenKind::LParen,
            ')' => TokenKind::RParen,
            '[' => TokenKind::LBracket,
            ']' => TokenKind::RBracket,
            ':' => TokenKind::Colon,
            ',' => TokenKind::Comma,
            '.' => TokenKind::Dot,
            '/' => TokenKind::Slash,
            '+' => TokenKind::Plus,

            '-' => {
                if self.peek() == Some('>') {
                    self.advance();
                    TokenKind::Arrow
                } else {
                    TokenKind::Minus
                }
            }

            '*' => TokenKind::Star,

            '=' => {
                if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::Eq
                } else {
                    TokenKind::Assign
                }
            }

            '!' => {
                if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::Ne
                } else {
                    return Err(LexError::UnexpectedChar {
                        ch,
                        span: self.make_span(start_pos).into(),
                    });
                }
            }

            '<' => {
                if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::Le
                } else {
                    TokenKind::Lt
                }
            }

            '>' => {
                if self.peek() == Some('=') {
                    self.advance();
                    TokenKind::Ge
                } else {
                    TokenKind::Gt
                }
            }

            '`' => self.string()?,

            '@' => {
                // Annotations: @static, @global, @lazy
                let mut s = String::new();
                while let Some(ch) = self.peek() {
                    if ch.is_ascii_alphabetic() {
                        s.push(ch);
                        self.advance();
                    } else {
                        break;
                    }
                }
                match s.as_str() {
                    "static" => TokenKind::AtStatic,
                    "global" => TokenKind::AtGlobal,
                    "lazy" => TokenKind::AtLazy,
                    _ => {
                        return Err(LexError::UnexpectedChar {
                            ch,
                            span: self.make_span(start_pos).into(),
                        })
                    }
                }
            }

            c if c.is_ascii_digit() => self.number(c)?,

            c if c.is_ascii_alphabetic() || c == '_' => self.ident_or_keyword(c),

            _ => {
                return Err(LexError::UnexpectedChar {
                    ch,
                    span: self.make_span(start_pos).into(),
                })
            }
        };

        let text = self.source[start_pos..self.pos].to_string();
        Ok(Token {
            kind,
            span: Span {
                start: start_pos,
                end: self.pos,
                line: start_line,
                col: start_col,
            },
            text,
        })
    }

    fn handle_newline(&mut self) -> Result<Token, LexError> {
        let tok = self.make_token(TokenKind::Newline, "\n");
        self.line += 1;
        self.col = 1;
        self.at_line_start = true;

        // Skip blank lines
        while self.peek() == Some('\n') {
            self.advance();
            self.line += 1;
        }

        Ok(tok)
    }

    fn handle_indent(&mut self) -> Result<Option<Token>, LexError> {
        let mut indent = 0;
        let start_pos = self.pos;

        while let Some(ch) = self.peek() {
            match ch {
                ' ' => {
                    indent += 1;
                    self.advance();
                }
                '\t' => {
                    indent += 2; // Treat tab as 2 spaces
                    self.advance();
                }
                '\n' => {
                    // Blank line, skip
                    self.advance();
                    self.line += 1;
                    indent = 0;
                }
                '#' => {
                    // Comment line, skip
                    self.skip_line();
                    if self.peek() == Some('\n') {
                        self.advance();
                        self.line += 1;
                    }
                    indent = 0;
                }
                _ => break,
            }
        }

        // Check if we're at EOF
        if self.peek().is_none() {
            return Ok(None);
        }

        let current_indent = *self.indent_stack.last().unwrap();

        if indent > current_indent {
            self.indent_stack.push(indent);
            return Ok(Some(self.make_token(TokenKind::Indent, "")));
        }

        if indent < current_indent {
            while let Some(&top) = self.indent_stack.last() {
                if top <= indent {
                    break;
                }
                self.indent_stack.pop();
                self.pending_dedents += 1;
            }

            // Check for inconsistent indentation
            if *self.indent_stack.last().unwrap() != indent {
                return Err(LexError::InconsistentIndent {
                    span: SourceSpan::new(start_pos.into(), indent),
                    expected: *self.indent_stack.last().unwrap(),
                    found: indent,
                });
            }

            if self.pending_dedents > 0 {
                self.pending_dedents -= 1;
                return Ok(Some(self.make_token(TokenKind::Dedent, "")));
            }
        }

        Ok(None)
    }

    fn string(&mut self) -> Result<TokenKind, LexError> {
        let start = self.pos - 1; // Include opening backtick
        let mut value = String::new();

        loop {
            match self.advance() {
                Some('`') => break,
                Some(ch) => value.push(ch),
                None => {
                    return Err(LexError::UnterminatedString {
                        span: SourceSpan::new(start.into(), self.pos - start),
                    })
                }
            }
        }

        Ok(TokenKind::String(value))
    }

    fn number(&mut self, first: char) -> Result<TokenKind, LexError> {
        let mut s = String::from(first);
        let mut is_float = false;

        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                s.push(ch);
                self.advance();
            } else if ch == '.' && !is_float {
                // Look ahead to make sure it's a decimal point, not method call
                let mut chars = self.chars.clone();
                chars.next(); // skip the dot
                if chars
                    .peek()
                    .map(|(_, c)| c.is_ascii_digit())
                    .unwrap_or(false)
                {
                    is_float = true;
                    s.push(ch);
                    self.advance();
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if is_float {
            s.parse()
                .map(TokenKind::Float)
                .map_err(|_| LexError::InvalidNumber {
                    span: SourceSpan::new(self.pos.into(), s.len()),
                })
        } else {
            s.parse()
                .map(TokenKind::Int)
                .map_err(|_| LexError::InvalidNumber {
                    span: SourceSpan::new(self.pos.into(), s.len()),
                })
        }
    }

    fn ident_or_keyword(&mut self, first: char) -> TokenKind {
        let mut s = String::from(first);

        while let Some(ch) = self.peek() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                s.push(ch);
                self.advance();
            } else {
                break;
            }
        }

        match s.as_str() {
            "neuron" => TokenKind::Neuron,
            "use" => TokenKind::Use,
            "in" => TokenKind::In,
            "out" => TokenKind::Out,
            "impl" => TokenKind::Impl,
            "graph" => TokenKind::Graph,
            "context" => TokenKind::Context,
            "match" => TokenKind::Match,
            "where" => TokenKind::Where,
            "external" => TokenKind::External,
            "and" => TokenKind::And,
            "or" => TokenKind::Or,
            "let" => TokenKind::Let,
            "set" => TokenKind::Set,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            _ => TokenKind::Ident(s),
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.chars.peek().map(|(_, ch)| *ch)
    }

    fn advance(&mut self) -> Option<char> {
        if let Some((pos, ch)) = self.chars.next() {
            self.pos = pos + ch.len_utf8();
            self.col += 1;
            Some(ch)
        } else {
            None
        }
    }

    fn skip_horizontal_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch == ' ' || ch == '\t' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn skip_line(&mut self) {
        while let Some(ch) = self.peek() {
            if ch == '\n' {
                break;
            }
            self.advance();
        }
    }

    fn make_token(&self, kind: TokenKind, text: &str) -> Token {
        Token {
            kind,
            span: Span {
                start: self.pos,
                end: self.pos,
                line: self.line,
                col: self.col,
            },
            text: text.to_string(),
        }
    }

    fn make_span(&self, start: usize) -> Span {
        Span {
            start,
            end: self.pos,
            line: self.line,
            col: self.col,
        }
    }
}
