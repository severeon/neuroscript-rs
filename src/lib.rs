//! NeuroScript - Neural Architecture Composition Language
//!
//! A language for defining composable neural architectures.
//!
//! # Example
//!
//! ```neuroscript
//! neuron MLP(dim):
//!   in: [*, dim]
//!   out: [*, dim]
//!   graph:
//!     in ->
//!       Linear(dim, dim * 4)
//!       GELU()
//!       Linear(dim * 4, dim)
//!       out
//! ```

pub mod ir;
pub mod lexer;
pub mod parser;
pub mod validator;
pub mod codegen;

pub use ir::*;
pub use parser::Parser;
pub use validator::*;
pub use codegen::*;

/// Parse a NeuroScript source string into a Program.
pub fn parse(source: &str) -> Result<Program, parser::ParseError> {
    Parser::parse(source)
}

/// Validate a NeuroScript program for correctness:
/// 1. All referenced neurons exist
/// 2. Connection endpoints match (tuple arity, port names, shapes)
/// 3. No cycles in the dependency graph
pub fn validate(program: &Program) -> Result<(), Vec<validator::ValidationError>> {
    validator::Validator::validate(program)
}
