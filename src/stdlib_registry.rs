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

use std::collections::HashMap;

/// Implementation reference for a primitive neuron.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImplRef {
    /// Full Python module path (e.g., "neuroscript_runtime.primitives.linear")
    pub module_path: String,

    /// Class name (e.g., "Linear")
    pub class_name: String,

    /// Short description for documentation
    pub description: String,
}

impl ImplRef {
    /// Create a new implementation reference.
    pub fn new(module_path: impl Into<String>, class_name: impl Into<String>) -> Self {
        Self {
            module_path: module_path.into(),
            class_name: class_name.into(),
            description: String::new(),
        }
    }

    /// Create an implementation reference with description.
    pub fn with_desc(
        module_path: impl Into<String>,
        class_name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            module_path: module_path.into(),
            class_name: class_name.into(),
            description: description.into(),
        }
    }

    /// Get the full qualified name (module.class).
    pub fn full_name(&self) -> String {
        format!("{}.{}", self.module_path, self.class_name)
    }

    /// Get Python import statement.
    pub fn import_statement(&self) -> String {
        format!("from {} import {}", self.module_path, self.class_name)
    }
}

/// Standard library registry - maps neuron names to implementations.
pub struct StdlibRegistry {
    primitives: HashMap<String, ImplRef>,
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

        // Level 0: Structural Operations
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
            "Add",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.structural",
                "Add",
                "Element-wise addition (multi-input for residual connections)",
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

        // Level 0: Attention
        self.register(
            "ScaledDotProductAttention",
            ImplRef::with_desc(
                "neuroscript_runtime.primitives.attention",
                "ScaledDotProductAttention",
                "Scaled dot-product attention (transformer building block)",
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
mod tests {
    use super::*;

    #[test]
    fn test_registry_basics() {
        let registry = StdlibRegistry::new();

        // Check Level 0 primitives are registered
        assert!(registry.contains("Linear"));
        assert!(registry.contains("GELU"));
        assert!(registry.contains("Dropout"));
        assert!(registry.contains("LayerNorm"));
        assert!(registry.contains("Embedding"));

        // Check non-existent primitive
        assert!(!registry.contains("NonExistent"));

        // Check we have the expected number of primitives
        assert_eq!(registry.len(), 21); // 21 primitives registered

        // Check lookup works
        let linear = registry.lookup("Linear").unwrap();
        assert_eq!(linear.class_name, "Linear");
        assert_eq!(
            linear.module_path,
            "neuroscript_runtime.primitives.linear"
        );
        assert_eq!(
            linear.full_name(),
            "neuroscript_runtime.primitives.linear.Linear"
        );
    }

    #[test]
    fn test_impl_ref() {
        let impl_ref = ImplRef::with_desc(
            "neuroscript_runtime.primitives.linear",
            "Linear",
            "Dense layer",
        );

        assert_eq!(
            impl_ref.full_name(),
            "neuroscript_runtime.primitives.linear.Linear"
        );
        assert_eq!(
            impl_ref.import_statement(),
            "from neuroscript_runtime.primitives.linear import Linear"
        );
        assert_eq!(impl_ref.description, "Dense layer");
    }

    #[test]
    fn test_generate_imports() {
        let registry = StdlibRegistry::new();

        let used = vec![
            "Linear".to_string(),
            "GELU".to_string(),
            "Dropout".to_string(),
        ];

        let imports = registry.generate_imports(&used);

        assert_eq!(imports.len(), 3);
        assert!(imports.contains(&"from neuroscript_runtime.primitives.linear import Linear".to_string()));
        assert!(imports.contains(&"from neuroscript_runtime.primitives.activations import GELU".to_string()));
        assert!(imports.contains(&"from neuroscript_runtime.primitives.regularization import Dropout".to_string()));
    }

    #[test]
    fn test_all_primitives() {
        let registry = StdlibRegistry::new();
        let primitives = registry.primitives();

        // Should be sorted
        assert_eq!(primitives[0], "Add");
        assert!(primitives.contains(&"Linear".to_string()));
        assert!(primitives.contains(&"GELU".to_string()));

        // All primitives should have valid impl refs
        for name in primitives {
            let impl_ref = registry.lookup(&name).unwrap();
            assert!(!impl_ref.module_path.is_empty());
            assert!(!impl_ref.class_name.is_empty());
        }
    }
}
