//! Token-related implementations for the lexer

use crate::interfaces::*;
use miette::SourceSpan;

impl From<Span> for SourceSpan {
    fn from(s: Span) -> Self {
        SourceSpan::new(s.start.into(), s.end - s.start)
    }
}

impl LexError {
    pub fn span(&self) -> SourceSpan {
        match self {
            LexError::UnexpectedChar { span, .. } => *span,
            LexError::UnterminatedString { span, .. } => *span,
            LexError::InvalidNumber { span, .. } => *span,
            LexError::InconsistentIndent { span, .. } => *span,
        }
    }
}
