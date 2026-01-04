//! NeuroScript Graph Validator
//!
//! Validates that NeuroScript programs are well-formed:
//! 1. All referenced neurons exist
//! 2. Connection endpoints match (tuple arity, port names, shapes)
//! 3. No cycles in the dependency graph

pub mod core;

// Re-export public API
pub use core::Validator;

#[cfg(test)]
mod tests;
