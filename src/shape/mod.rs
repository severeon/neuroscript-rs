//! NeuroScript Shape System
//!
//! Provides shape algebra operations and shape inference for tensor
//! dimensions in neuron graphs.

pub mod algebra;
pub mod inference;

// Re-export public API
pub use algebra::*;
pub use inference::*;

#[cfg(test)]
mod tests;
