//! Error handling for the pest-based parser
//!
//! Converts pest errors to our ParseError type for consistent error reporting.

use miette::SourceSpan;
use pest::error::{Error as PestError, LineColLocation};

use crate::grammar::Rule;
use crate::interfaces::ParseError;

/// Convert a pest error to our ParseError type
pub fn from_pest_error(err: PestError<Rule>) -> ParseError {
    let (start, end) = match err.line_col {
        LineColLocation::Pos((line, col)) => {
            // Single position error - estimate span from location
            let start = estimate_offset(line, col);
            (start, start + 1)
        }
        LineColLocation::Span((start_line, start_col), (end_line, end_col)) => {
            let start = estimate_offset(start_line, start_col);
            let end = estimate_offset(end_line, end_col);
            (start, end)
        }
    };

    let span = SourceSpan::new(start.into(), (end - start).into());

    // Extract what was expected vs found from the pest error
    let message = err.to_string();
    
    ParseError::Expected {
        expected: extract_expected(&message),
        found: extract_found(&message),
        span,
    }
}

/// Estimate byte offset from line/column (approximate)
fn estimate_offset(line: usize, col: usize) -> usize {
    // Rough estimate: assume 80 chars per line on average
    // This is imprecise but gives a reasonable location for error reporting
    (line.saturating_sub(1)) * 80 + col.saturating_sub(1)
}

/// Extract expected items from pest error message
fn extract_expected(message: &str) -> String {
    if let Some(idx) = message.find("expected") {
        let rest = &message[idx + 8..];
        if let Some(end) = rest.find('\n') {
            rest[..end].trim().to_string()
        } else {
            rest.trim().to_string()
        }
    } else {
        "valid syntax".to_string()
    }
}

/// Extract found item from pest error message  
fn extract_found(message: &str) -> String {
    if let Some(idx) = message.find("found") {
        let rest = &message[idx + 5..];
        if let Some(end) = rest.find('\n') {
            rest[..end].trim().to_string()
        } else {
            rest.trim().to_string()
        }
    } else if let Some(idx) = message.find("unexpected") {
        let rest = &message[idx + 10..];
        if let Some(end) = rest.find('\n') {
            rest[..end].trim().to_string()
        } else {
            rest.trim().to_string()
        }
    } else {
        "unexpected input".to_string()
    }
}

/// Create a ParseError for duplicate neuron definitions
pub fn duplicate_neuron(name: &str, offset: usize) -> ParseError {
    ParseError::DuplicateNeuron {
        name: name.to_string(),
        span: SourceSpan::new(offset.into(), name.len().into()),
    }
}

/// Create a ParseError for unexpected content
pub fn unexpected(found: &str, offset: usize) -> ParseError {
    ParseError::Unexpected {
        found: found.to_string(),
        span: SourceSpan::new(offset.into(), found.len().max(1).into()),
    }
}

/// Create a ParseError for expected content
pub fn expected(expected: &str, found: &str, offset: usize) -> ParseError {
    ParseError::Expected {
        expected: expected.to_string(),
        found: found.to_string(),
        span: SourceSpan::new(offset.into(), found.len().max(1).into()),
    }
}
