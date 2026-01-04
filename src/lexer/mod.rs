//! NeuroScript Lexer
//!
//! Tokenizes source text into a stream of tokens.
//! Handles indentation-based scoping.

// Module organization
pub mod core;
pub mod token;

// Re-exports for public API
pub use crate::interfaces::Lexer;

#[cfg(test)]
mod tests;
