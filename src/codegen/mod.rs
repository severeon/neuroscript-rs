//! PyTorch code generation from NeuroScript IR
//!
//! This module generates Python code that implements NeuroScript neurons
//! as PyTorch nn.Module classes.
//!
//! # Example
//!
//! ```ignore
//! use neuroscript::parse;
//! use neuroscript::codegen::generate_pytorch;
//!
//! let program = parse("neuron MLP(dim): ...")?;
//! let code = generate_pytorch(&program, "MLP")?;
//! println!("{}", code);
//! ```

// Module organization
pub mod forward;
pub mod generator;
pub mod instantiation;
pub mod utils;

// Re-exports for public API
pub use generator::{
    generate_pytorch, generate_pytorch_with_options, CodeGenerator, CodegenError, CodegenOptions,
    ShapeCheckResult,
};

#[cfg(test)]
mod tests;
