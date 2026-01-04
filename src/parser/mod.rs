//! NeuroScript Parser
//!
//! Recursive descent parser with good error messages.

// Module organization
pub mod core;

// Re-exports for public API
pub use crate::interfaces::Parser;

#[cfg(test)]
mod tests;
