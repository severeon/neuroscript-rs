//! Standard library registry - maps neuron names to implementation references.
//!
//! This module provides the mapping between NeuroScript primitive neuron names
//! and their Python/PyTorch implementations in the `neuroscript_runtime` package.
//!
//! When a neuron has `impl: <reference>`, that reference is looked up here to
//! determine what actual code to generate/import.
//!
//! Example:
//! ```neuroscript
//! neuron Linear(in_dim: int, out_dim: int):
//!     in: [*, in_dim]
//!     out: [*, out_dim]
//!     impl: neuroscript_runtime.primitives.Linear
//! ```
//!
//! The impl reference `neuroscript_runtime.primitives.Linear` maps to the
//! Python class in `neuroscript_runtime/primitives/linear.py`.

use crate::interfaces::*;
use std::collections::HashMap;

impl ImplRef {
    /// Create a new source-based implementation reference.
    pub fn new(module_path: impl Into<String>, class_name: impl Into<String>) -> Self {
        Self::Source {
            source: module_path.into(),
            path: class_name.into(),
        }
    }

    /// Create a source-based implementation reference with description.
    pub fn with_desc(
        module_path: impl Into<String>,
        class_name: impl Into<String>,
        _description: impl Into<String>,
    ) -> Self {
        Self::Source {
            source: module_path.into(),
            path: class_name.into(),
        }
    }

    /// Get the full qualified name (module.class).
    pub fn full_name(&self) -> String {
        match self {
            ImplRef::External { .. } => "external".to_string(),
            ImplRef::Source { source, path } => format!("{}.{}", source, path),
        }
    }

    /// Get Python import statement.
    pub fn import_statement(&self) -> String {
        match self {
            ImplRef::External { .. } => "external".to_string(),
            ImplRef::Source { source, path } => format!("from {} import {}", source, path),
        }
    }
}

impl StdlibRegistry {
    /// Create a new registry with all standard primitives registered.
    pub fn new() -> Self {
        let mut registry = Self {
            primitives: HashMap::new(),
        };

        registry.register_all_primitives();
        registry
    }

    /// Register all standard library primitives.
    fn register_all_primitives(&mut self) {
        // Level 0: Core Operations
        self.register(
            "Linear",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.linear",
                "Linear",
                "Dense/fully-connected layer with shape tracking",
            ),
        );

        self.register(
            "Bias",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.operations",
                "Bias",
                "Additive bias",
            ),
        );

        self.register(
            "Scale",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.operations",
                "Scale",
                "Multiplicative scale",
            ),
        );

        self.register(
            "MatMul",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.operations",
                "MatMul",
                "Matrix multiplication",
            ),
        );

        self.register(
            "Einsum",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.operations",
                "Einsum",
                "Einstein summation",
            ),
        );

        // Level 0: Activations
        self.register(
            "GELU",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.activations",
                "GELU",
                "Gaussian Error Linear Unit activation",
            ),
        );

        self.register(
            "ReLU",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.activations",
                "ReLU",
                "Rectified Linear Unit activation",
            ),
        );

        self.register(
            "Tanh",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.activations",
                "Tanh",
                "Hyperbolic tangent activation",
            ),
        );

        self.register(
            "Sigmoid",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.activations",
                "Sigmoid",
                "Sigmoid activation function",
            ),
        );

        self.register(
            "SiLU",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.activations",
                "SiLU",
                "Sigmoid Linear Unit (Swish) activation",
            ),
        );

        self.register(
            "Softmax",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.activations",
                "Softmax",
                "Softmax activation (normalizes to probability distribution)",
            ),
        );

        self.register(
            "Mish",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.activations",
                "Mish",
                "Mish activation",
            ),
        );

        self.register(
            "PReLU",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.activations",
                "PReLU",
                "Parametric ReLU",
            ),
        );

        self.register(
            "ELU",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.activations",
                "ELU",
                "Exponential Linear Unit",
            ),
        );

        // Level 0: Normalizations
        self.register(
            "LayerNorm",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.normalization",
                "LayerNorm",
                "Layer normalization (used in transformers)",
            ),
        );

        self.register(
            "RMSNorm",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.normalization",
                "RMSNorm",
                "Root Mean Square normalization (efficient variant)",
            ),
        );

        self.register(
            "GroupNorm",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.normalization",
                "GroupNorm",
                "Group normalization (works well with small batches)",
            ),
        );

        self.register(
            "BatchNorm",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.normalization",
                "BatchNorm",
                "Batch normalization",
            ),
        );

        self.register(
            "InstanceNorm",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.normalization",
                "InstanceNorm",
                "Instance normalization",
            ),
        );

        // Level 0: Regularization
        self.register(
            "Dropout",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.regularization",
                "Dropout",
                "Dropout regularization with training/eval modes",
            ),
        );

        self.register(
            "DropPath",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.regularization",
                "DropPath",
                "Stochastic depth / drop path regularization",
            ),
        );

        self.register(
            "DropConnect",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.regularization",
                "DropConnect",
                "Connection dropout (drops weights)",
            ),
        );

        // Level 0: Convolutions
        self.register(
            "Conv1d",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.convolutions",
                "Conv1d",
                "1D convolution layer",
            ),
        );

        self.register(
            "Conv2d",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.convolutions",
                "Conv2d",
                "2D convolution layer",
            ),
        );

        self.register(
            "Conv3d",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.convolutions",
                "Conv3d",
                "3D convolution layer",
            ),
        );

        self.register(
            "DepthwiseConv",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.convolutions",
                "DepthwiseConv",
                "Depthwise convolution layer",
            ),
        );

        self.register(
            "SeparableConv",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.convolutions",
                "SeparableConv",
                "Separable convolution layer",
            ),
        );

        self.register(
            "TransposedConv",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.convolutions",
                "TransposedConv",
                "Transposed convolution layer",
            ),
        );

        // Level 0: Pooling
        self.register(
            "MaxPool",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.pooling",
                "MaxPool",
                "Max pooling",
            ),
        );

        self.register(
            "AvgPool",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.pooling",
                "AvgPool",
                "Average pooling",
            ),
        );

        self.register(
            "AdaptiveAvgPool",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.pooling",
                "AdaptiveAvgPool",
                "Adaptive average pooling (output size fixed)",
            ),
        );

        self.register(
            "GlobalAvgPool",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.pooling",
                "GlobalAvgPool",
                "Global average pooling (reduces spatial dims to 1x1)",
            ),
        );

        self.register(
            "AdaptiveMaxPool",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.pooling",
                "AdaptiveMaxPool",
                "Adaptive max pooling (output size fixed)",
            ),
        );

        self.register(
            "GlobalMaxPool",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.pooling",
                "GlobalMaxPool",
                "Global max pooling (reduces spatial dims to 1x1)",
            ),
        );

        // Level 0: Embeddings
        self.register(
            "Embedding",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.embeddings",
                "Embedding",
                "Token embedding layer (discrete → dense)",
            ),
        );

        self.register(
            "PositionalEncoding",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.embeddings",
                "PositionalEncoding",
                "Sinusoidal positional encoding (Attention is All You Need)",
            ),
        );

        self.register(
            "LearnedPositionalEmbedding",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.embeddings",
                "LearnedPositionalEmbedding",
                "Learned positional embeddings (BERT/GPT style)",
            ),
        );

        self.register(
            "RotaryEmbedding",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.embeddings",
                "RotaryEmbedding",
                "Rotary Position Embedding (RoPE)",
            ),
        );

        // Level 0: Structural Operations
        self.register(
            "Identity",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.structural",
                "Identity",
                "Identity operation (pass-through, no-op)",
            ),
        );

        self.register(
            "Fork",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.structural",
                "Fork",
                "Split tensor into two references (multi-output for residual connections)",
            ),
        );

        self.register(
            "Fork3",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.structural",
                "Fork3",
                "Split tensor into three references (multi-output)",
            ),
        );

        self.register(
            "ForkN",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.structural",
                "ForkN",
                "Split tensor into N references (generic multi-output)",
            ),
        );

        self.register(
            "Add",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.structural",
                "Add",
                "Element-wise addition (multi-input for residual connections)",
            ),
        );

        self.register(
            "Multiply",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.structural",
                "Multiply",
                "Element-wise multiplication (for gating mechanisms)",
            ),
        );

        self.register(
            "Concat",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.structural",
                "Concat",
                "Concatenate tensors along a dimension",
            ),
        );

        self.register(
            "Reshape",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.structural",
                "Reshape",
                "Reshape tensor to a new shape (preserves element count)",
            ),
        );

        self.register(
            "Transpose",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.structural",
                "Transpose",
                "Permute tensor dimensions (transpose/permute operation)",
            ),
        );

        self.register(
            "Flatten",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.structural",
                "Flatten",
                "Flattens a contiguous range of dims into a tensor",
            ),
        );

        self.register(
            "Split",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.structural",
                "Split",
                "Split tensor into chunks",
            ),
        );

        self.register(
            "Slice",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.structural",
                "Slice",
                "Slice tensor along a dimension",
            ),
        );

        self.register(
            "Pad",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.structural",
                "Pad",
                "Pad tensor with value",
            ),
        );

        // Level 0: Attention
        self.register(
            "ScaledDotProductAttention",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.attention",
                "ScaledDotProductAttention",
                "Scaled dot-product attention (transformer building block)",
            ),
        );

        self.register(
            "MultiHeadSelfAttention",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.attention",
                "MultiHeadSelfAttention",
                "Multi-head self-attention (complete attention mechanism)",
            ),
        );

        // Level 0: Debug/Logging
        self.register(
            "Log",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.logging",
                "Log",
                "Debug logging with colored output and tensor info (pass-through)",
            ),
        );
    }

    /// Register a primitive neuron implementation.
    pub fn register(&mut self, name: impl Into<String>, impl_ref: ImplRef) {
        self.primitives.insert(name.into(), impl_ref);
    }

    /// Look up a primitive by name.
    pub fn lookup(&self, name: &str) -> Option<&ImplRef> {
        self.primitives.get(name)
    }

    /// Check if a primitive is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.primitives.contains_key(name)
    }

    /// Get all registered primitive names (sorted).
    pub fn primitives(&self) -> Vec<String> {
        let mut names: Vec<_> = self.primitives.keys().cloned().collect();
        names.sort();
        names
    }

    /// Get number of registered primitives.
    pub fn len(&self) -> usize {
        self.primitives.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.primitives.is_empty()
    }

    /// Generate Python imports for all used primitives.
    pub fn generate_imports(&self, used_primitives: &[String]) -> Vec<String> {
        let mut imports = Vec::new();
        let mut seen_modules = std::collections::HashSet::new();

        for name in used_primitives {
            if let Some(impl_ref) = self.lookup(name) {
                let import_stmt = impl_ref.import_statement();
                if seen_modules.insert(import_stmt.clone()) {
                    imports.push(import_stmt);
                }
            }
        }

        imports.sort();
        imports
    }
}

impl Default for StdlibRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
