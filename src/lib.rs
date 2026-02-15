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

pub mod codegen;
pub mod doc_parser;
pub mod grammar;
pub mod interfaces;
pub mod ir;
pub mod optimizer;
#[cfg(feature = "cli")]
pub mod package;
pub mod shape;
pub mod stdlib;
pub mod stdlib_registry;
pub mod unroll;
pub mod validator;
#[cfg(feature = "wasm")]
pub mod wasm;

// Re-export main IR types (avoiding glob to prevent conflicts)
pub use codegen::{generate_pytorch, generate_pytorch_with_options, CodegenOptions};
pub use interfaces::*;
// Shape algebra and stdlib registry accessed via their modules to avoid conflicts
pub use validator::*;

/// Parse a NeuroScript source string into a Program using the pest grammar.
pub fn parse(source: &str) -> Result<Program, ParseError> {
    grammar::NeuroScriptParser::parse_program(source)
}

/// Validate a NeuroScript program for correctness:
/// 1. All referenced neurons exist
/// 2. Connection endpoints match (tuple arity, port names, shapes)
/// 3. No cycles in the dependency graph
/// 4. Shape compatibility for all connections
pub fn validate(program: &mut Program) -> Result<(), Vec<ValidationError>> {
    // Expand unroll constructs before any validation
    unroll::expand_unrolls(program)?;

    // First run basic validation
    validator::Validator::validate(program)?;

    // Then run shape inference validation
    let mut shape_engine = shape::ShapeInferenceEngine::new();
    match shape_engine.infer(program) {
        Ok(()) => Ok(()),
        Err(shape_errors) => {
            // Convert shape errors to validation errors
            let validation_errors = shape_errors
                .into_iter()
                .map(|e| ValidationError::Custom(format!("Shape error: {}", e)))
                .collect();
            Err(validation_errors)
        }
    }
}
