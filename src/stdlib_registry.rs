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
//!     in: [*batch, in_dim]
//!     out: [*batch, out_dim]
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

    /// Register all standard library primitives using a table-driven approach.
    ///
    /// Each entry is `(name, module_path, class_name, description)`.
    /// Grouped by category for readability.
    fn register_all_primitives(&mut self) {
        /// Primitive registration table: (name, module, class, description).
        const PRIMITIVES: &[(&str, &str, &str, &str)] = &[
            // ── Core Operations ──────────────────────────────────────────
            ("Linear",     "neuroscript_runtime.primitives.linear",     "Linear",     "Dense/fully-connected layer with shape tracking"),
            ("Bias",       "neuroscript_runtime.primitives.operations", "Bias",       "Additive bias"),
            ("Scale",      "neuroscript_runtime.primitives.operations", "Scale",      "Multiplicative scale"),
            ("ConstScale", "neuroscript_runtime.primitives.operations", "ConstScale", "Constant scalar multiplication"),
            ("MatMul",     "neuroscript_runtime.primitives.operations", "MatMul",     "Matrix multiplication"),
            ("Einsum",     "neuroscript_runtime.primitives.operations", "Einsum",     "Einstein summation"),

            // ── Activations ──────────────────────────────────────────────
            ("GELU",    "neuroscript_runtime.primitives.activations", "GELU",    "Gaussian Error Linear Unit activation"),
            ("ReLU",    "neuroscript_runtime.primitives.activations", "ReLU",    "Rectified Linear Unit activation"),
            ("Tanh",    "neuroscript_runtime.primitives.activations", "Tanh",    "Hyperbolic tangent activation"),
            ("Sigmoid", "neuroscript_runtime.primitives.activations", "Sigmoid", "Sigmoid activation function"),
            ("SiLU",    "neuroscript_runtime.primitives.activations", "SiLU",    "Sigmoid Linear Unit (Swish) activation"),
            ("Softmax", "neuroscript_runtime.primitives.activations", "Softmax", "Softmax activation (normalizes to probability distribution)"),
            ("Mish",    "neuroscript_runtime.primitives.activations", "Mish",    "Mish activation"),
            ("PReLU",   "neuroscript_runtime.primitives.activations", "PReLU",   "Parametric ReLU"),
            ("ELU",     "neuroscript_runtime.primitives.activations", "ELU",     "Exponential Linear Unit"),

            // ── Normalizations ───────────────────────────────────────────
            ("LayerNorm",    "neuroscript_runtime.primitives.normalization", "LayerNorm",    "Layer normalization (used in transformers)"),
            ("RMSNorm",      "neuroscript_runtime.primitives.normalization", "RMSNorm",      "Root Mean Square normalization (efficient variant)"),
            ("GroupNorm",    "neuroscript_runtime.primitives.normalization", "GroupNorm",     "Group normalization (works well with small batches)"),
            ("BatchNorm",    "neuroscript_runtime.primitives.normalization", "BatchNorm",     "Batch normalization"),
            ("InstanceNorm", "neuroscript_runtime.primitives.normalization", "InstanceNorm",  "Instance normalization"),
            ("WeightNorm",   "neuroscript_runtime.primitives.normalization", "WeightNorm",    "Weight normalization (decouples magnitude from direction)"),

            // ── Regularization ───────────────────────────────────────────
            ("Dropout",     "neuroscript_runtime.primitives.regularization", "Dropout",     "Dropout regularization with training/eval modes"),
            ("DropPath",    "neuroscript_runtime.primitives.regularization", "DropPath",    "Stochastic depth / drop path regularization"),
            ("DropConnect", "neuroscript_runtime.primitives.regularization", "DropConnect", "Connection dropout (drops weights)"),
            ("Dropblock",   "neuroscript_runtime.primitives.regularization", "Dropblock",   "Structured dropout for CNNs (drops contiguous regions)"),
            ("SpecAugment", "neuroscript_runtime.primitives.regularization", "SpecAugment", "Frequency/time masking for audio spectrograms"),

            // ── Convolutions ─────────────────────────────────────────────
            ("Conv1d",         "neuroscript_runtime.primitives.convolutions", "Conv1d",         "1D convolution layer"),
            ("Conv2d",         "neuroscript_runtime.primitives.convolutions", "Conv2d",         "2D convolution layer"),
            ("Conv3d",         "neuroscript_runtime.primitives.convolutions", "Conv3d",         "3D convolution layer"),
            ("DepthwiseConv",  "neuroscript_runtime.primitives.convolutions", "DepthwiseConv",  "Depthwise convolution layer"),
            ("SeparableConv",  "neuroscript_runtime.primitives.convolutions", "SeparableConv",  "Separable convolution layer"),
            ("TransposedConv", "neuroscript_runtime.primitives.convolutions", "TransposedConv", "Transposed convolution layer"),
            ("DilatedConv",    "neuroscript_runtime.primitives.convolutions", "DilatedConv",    "Dilated (atrous) convolution for expanded receptive fields"),

            // ── Pooling ──────────────────────────────────────────────────
            ("MaxPool",         "neuroscript_runtime.primitives.pooling", "MaxPool",         "Max pooling"),
            ("AvgPool",         "neuroscript_runtime.primitives.pooling", "AvgPool",         "Average pooling"),
            ("AdaptiveAvgPool", "neuroscript_runtime.primitives.pooling", "AdaptiveAvgPool", "Adaptive average pooling (output size fixed)"),
            ("GlobalAvgPool",   "neuroscript_runtime.primitives.pooling", "GlobalAvgPool",   "Global average pooling (reduces spatial dims to 1x1)"),
            ("AdaptiveMaxPool", "neuroscript_runtime.primitives.pooling", "AdaptiveMaxPool", "Adaptive max pooling (output size fixed)"),
            ("GlobalMaxPool",   "neuroscript_runtime.primitives.pooling", "GlobalMaxPool",   "Global max pooling (reduces spatial dims to 1x1)"),

            // ── Embeddings ───────────────────────────────────────────────
            ("Embedding",                  "neuroscript_runtime.primitives.embeddings", "Embedding",                  "Token embedding layer (discrete → dense)"),
            ("PositionalEncoding",         "neuroscript_runtime.primitives.embeddings", "PositionalEncoding",         "Sinusoidal positional encoding (Attention is All You Need)"),
            ("LearnedPositionalEmbedding", "neuroscript_runtime.primitives.embeddings", "LearnedPositionalEmbedding", "Learned positional embeddings (BERT/GPT style)"),
            ("RotaryEmbedding",            "neuroscript_runtime.primitives.embeddings", "RotaryEmbedding",            "Rotary Position Embedding (RoPE)"),
            ("ALiBi",                      "neuroscript_runtime.primitives.embeddings", "ALiBi",                      "Attention with Linear Biases (length extrapolation)"),

            // ── Structural Operations ────────────────────────────────────
            ("Identity",  "neuroscript_runtime.primitives.structural", "Identity",  "Identity operation (pass-through, no-op)"),
            ("Fork",      "neuroscript_runtime.primitives.structural", "Fork",      "Split tensor into two references (multi-output for residual connections)"),
            ("Fork3",     "neuroscript_runtime.primitives.structural", "Fork3",     "Split tensor into three references (multi-output)"),
            ("ForkN",     "neuroscript_runtime.primitives.structural", "ForkN",     "Split tensor into N references (generic multi-output)"),
            ("Add",       "neuroscript_runtime.primitives.structural", "Add",       "Element-wise addition (multi-input for residual connections)"),
            ("Multiply",  "neuroscript_runtime.primitives.structural", "Multiply",  "Element-wise multiplication (for gating mechanisms)"),
            ("Subtract",  "neuroscript_runtime.primitives.structural", "Subtract",  "Element-wise subtraction (for complement/residual operations)"),
            ("Divide",    "neuroscript_runtime.primitives.structural", "Divide",    "Element-wise division (for normalization/scaling operations)"),
            ("Concat",    "neuroscript_runtime.primitives.structural", "Concat",    "Concatenate tensors along a dimension"),
            ("Reshape",   "neuroscript_runtime.primitives.structural", "Reshape",   "Reshape tensor to a new shape (preserves element count)"),
            ("Transpose", "neuroscript_runtime.primitives.structural", "Transpose", "Permute tensor dimensions (transpose/permute operation)"),
            ("Flatten",   "neuroscript_runtime.primitives.structural", "Flatten",   "Flattens a contiguous range of dims into a tensor"),
            ("Split",     "neuroscript_runtime.primitives.structural", "Split",     "Split tensor into chunks"),
            ("Slice",     "neuroscript_runtime.primitives.structural", "Slice",     "Slice tensor along a dimension"),
            ("Pad",       "neuroscript_runtime.primitives.structural", "Pad",       "Pad tensor with value"),
            ("Crop",      "neuroscript_runtime.primitives.structural", "Crop",      "Crop tensor to target spatial dimensions"),
            ("Cast",      "neuroscript_runtime.primitives.structural", "Cast",      "Convert tensor dtype"),
            ("Clone",     "neuroscript_runtime.primitives.structural", "Clone",     "Create independent copy of tensor"),

            // ── Attention ────────────────────────────────────────────────
            ("ScaledDotProductAttention", "neuroscript_runtime.primitives.attention", "ScaledDotProductAttention", "Scaled dot-product attention (transformer building block)"),
            ("MultiHeadSelfAttention",   "neuroscript_runtime.primitives.attention", "MultiHeadSelfAttention",   "Multi-head self-attention (complete attention mechanism)"),

            // ── Debug/Logging ────────────────────────────────────────────
            ("Log", "neuroscript_runtime.primitives.logging", "Log", "Debug logging with colored output and tensor info (pass-through)"),

            // ── Connections (Hyper-Connections) ──────────────────────────
            ("HyperExpand",   "neuroscript_runtime.primitives.connections", "HyperExpand",   "Expand single hidden to n copies for hyper-connections"),
            ("HyperCollapse", "neuroscript_runtime.primitives.connections", "HyperCollapse", "Collapse n copies via sum for hyper-connections"),
            ("HCWidth",       "neuroscript_runtime.primitives.connections", "HCWidth",       "Width connection for hyper-connections (mix hidden copies)"),
            ("HCDepth",       "neuroscript_runtime.primitives.connections", "HCDepth",       "Depth connection for hyper-connections (merge layer output)"),

            // ── Diffusion ────────────────────────────────────────────────
            ("DenoisingHead",           "neuroscript_runtime.primitives.diffusion", "DenoisingHead",           "MLM-style prediction head for masked diffusion (hidden → logits)"),
            ("MultiTokenPredictionHead","neuroscript_runtime.primitives.diffusion", "MultiTokenPredictionHead","Predicts N future tokens simultaneously (multi-token prediction)"),

            // ── Routing ──────────────────────────────────────────────────
            ("SigmoidMoERouter",        "neuroscript_runtime.primitives.routing",   "SigmoidMoERouter",        "Sigmoid-gated MoE router with auxiliary-loss-free load balancing (DeepSeek-V3)"),
        ];

        for &(name, module, class, desc) in PRIMITIVES {
            self.register(name, ImplRef::with_desc(module, class, desc));
        }
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

    /// Get unique module paths for a set of used primitives, sorted for deterministic output.
    pub fn modules_for_primitives(&self, used_primitives: &[String]) -> Vec<String> {
        let mut modules = std::collections::BTreeSet::new();
        for name in used_primitives {
            if let Some(impl_ref) = self.lookup(name) {
                if let ImplRef::Source { source, .. } = impl_ref {
                    modules.insert(source.clone());
                }
            }
        }
        modules.into_iter().collect()
    }
}

impl Default for StdlibRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
