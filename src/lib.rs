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
pub mod shape_algebra;
pub mod stdlib_registry;
pub mod validator;

// Re-export main IR types (avoiding glob to prevent conflicts)
pub use ir::{
    BinOp, Connection, Dim, DimExpr, Endpoint, ImplRef as IrImplRef, MatchArm, MatchExpr,
    NeuronBody, NeuronDef, Param, Port, PortRef, Program, Shape as IrShape, UseStmt, Value,
};
pub use parser::Parser;
// Shape algebra and stdlib registry accessed via their modules to avoid conflicts
pub use validator::*;

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
