//! NeuroScript Lexer
//!
//! Tokenizes source text into a stream of tokens.
//! Handles indentation-based scoping.

// Module organization
pub mod token;
pub mod core;

// Re-exports for public API
pub use crate::interfaces::Lexer;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interfaces::*;

    fn lex(s: &str) -> Vec<TokenKind> {
        Lexer::new(s)
            .tokenize()
            .unwrap()
            .into_iter()
            .map(|t| t.kind)
            .collect()
    }

    #[test]
    fn test_simple_tokens() {
        assert_eq!(
            lex("-> : , . + -"),
            vec![
                TokenKind::Arrow,
                TokenKind::Colon,
                TokenKind::Comma,
                TokenKind::Dot,
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_keywords() {
        assert_eq!(
            lex("neuron use in out impl graph"),
            vec![
                TokenKind::Neuron,
                TokenKind::Use,
                TokenKind::In,
                TokenKind::Out,
                TokenKind::Impl,
                TokenKind::Graph,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_string() {
        assert_eq!(
            lex("`hello world`"),
            vec![TokenKind::String("hello world".into()), TokenKind::Eof]
        );
    }

    #[test]
    fn test_indent() {
        let tokens = lex("a:\n  b\n  c\nd");
        assert!(tokens.contains(&TokenKind::Indent));
        assert!(tokens.contains(&TokenKind::Dedent));
    }
}
