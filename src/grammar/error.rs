//! Error handling for the pest-based parser
//!
//! Converts pest errors to our ParseError type for consistent error reporting.

use miette::SourceSpan;
use pest::error::{Error as PestError, InputLocation};

use crate::grammar::Rule;
use crate::interfaces::ParseError;

/// Convert a pest error to our ParseError type
pub fn from_pest_error(err: PestError<Rule>) -> ParseError {
    // Use pest's InputLocation which provides exact byte offsets,
    // rather than estimating from line/column numbers
    let (start, end) = match err.location {
        InputLocation::Pos(pos) => {
            (pos, pos + 1)
        }
        InputLocation::Span((start, end)) => {
            (start, end)
        }
    };

    let span = SourceSpan::new(start.into(), end - start);

    // Extract what was expected vs found from the pest error
    let message = err.to_string();

    ParseError::Expected {
        expected: extract_expected(&message),
        found: extract_found(&message),
        span,
    }
}

/// Compute byte offset from line/column using actual source text.
///
/// Lines and columns are 1-based (as reported by pest's `LineColLocation`).
/// Falls back to `estimate_offset` if the line number exceeds the source line count.
pub fn compute_offset(source: &str, line: usize, col: usize) -> usize {
    let mut offset = 0;
    for (i, src_line) in source.lines().enumerate() {
        if i + 1 == line {
            // Found the target line; add the column offset (1-based → 0-based)
            return offset + col.saturating_sub(1).min(src_line.len());
        }
        // +1 for the newline character
        offset += src_line.len() + 1;
    }
    // Fallback: line exceeds source — use estimate
    estimate_offset(line, col)
}

/// Estimate byte offset from line/column (approximate fallback)
fn estimate_offset(line: usize, col: usize) -> usize {
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
        span: SourceSpan::new(offset.into(), name.len()),
    }
}

/// Create a ParseError for unexpected content
pub fn unexpected(found: &str, offset: usize) -> ParseError {
    ParseError::Unexpected {
        found: found.to_string(),
        span: SourceSpan::new(offset.into(), found.len().max(1)),
    }
}

/// Create a ParseError for expected content
pub fn expected(expected: &str, found: &str, offset: usize) -> ParseError {
    ParseError::Expected {
        expected: expected.to_string(),
        found: found.to_string(),
        span: SourceSpan::new(offset.into(), found.len().max(1)),
    }
}
