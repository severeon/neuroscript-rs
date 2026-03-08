//! Grammar-based parser using pest
//!
//! This module replaces the handwritten lexer/parser with a PEG grammar.
//! Indentation is handled in the AST builder, not in the grammar itself.

pub mod ast;
pub mod error;

use pest::Parser as PestParser;
use pest_derive::Parser;

use crate::interfaces::{ParseError, Program};

#[derive(Parser)]
#[grammar = "grammar/neuroscript.pest"]
pub struct NeuroScriptParser;

impl NeuroScriptParser {
    /// Parse a NeuroScript source string into a Program using the pest grammar.
    ///
    /// This is the main entry point for the new parser.
    pub fn parse_program(source: &str) -> Result<Program, ParseError> {
        let pairs =
            NeuroScriptParser::parse(Rule::program, source).map_err(error::from_pest_error)?;

        let mut builder = ast::AstBuilder::new();
        builder.build_program(pairs.into_iter().next().expect("pest parse guarantees program rule"))
    }
}

#[cfg(test)]
mod tests;
