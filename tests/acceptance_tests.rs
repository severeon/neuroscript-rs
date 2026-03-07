//! Acceptance tests for GitHub issues.
//!
//! Each test targets a specific issue and is designed to fail until the
//! issue is implemented. These tests serve as automated acceptance criteria.

use neuroscript::interfaces::*;
use neuroscript::{parse, validate};
use std::fs;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// #117 — MutualLazyRecursion error lacks source span
// ---------------------------------------------------------------------------

/// The MutualLazyRecursion validation error should carry a source span so
/// miette can display inline diagnostics. Currently the Binding struct has
/// no span field, so the error always returns None.
#[test]
fn issue_117_mutual_lazy_recursion_error_has_span() {
    let source = r#"
neuron Helper(x):
    in: [*shape]
    out: [*shape]
    impl: core,nn/Identity

neuron MutualTest(dim):
    in: [*shape]
    out: [*shape]
    context:
        @lazy a = Helper(b)
        @lazy b = Helper(a)
    graph:
        in -> a -> out
"#;

    let mut program = parse(source).expect("should parse successfully");
    let result = validate(&mut program);
    assert!(
        result.is_err(),
        "Expected MutualLazyRecursion validation error"
    );
    let errors = result.unwrap_err();
    let mutual_error = errors
        .iter()
        .find(|e| matches!(e, ValidationError::MutualLazyRecursion { .. }))
        .expect("Expected a MutualLazyRecursion error");
    assert!(
        mutual_error.span().is_some(),
        "MutualLazyRecursion error should have a source span, but span() returned None"
    );
}

// ---------------------------------------------------------------------------
// #120 — Parse failure: embeddings.ns uses unsupported Reshape bracket syntax
// ---------------------------------------------------------------------------

/// Regression test: examples/primitives/embeddings.ns should parse
/// successfully. It previously used Reshape([...]) bracket syntax but was
/// updated to fat arrow (=>) syntax in PR #83.
#[test]
fn issue_120_embeddings_ns_parses_successfully() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("primitives")
        .join("embeddings.ns");
    let source = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    let program = parse(&source)
        .unwrap_or_else(|e| panic!("examples/primitives/embeddings.ns should parse: {:?}", e));
    assert!(
        program.neurons.contains_key("RoPEAttentionInput"),
        "Expected RoPEAttentionInput neuron (uses fat arrow syntax at previously-broken line 75)"
    );
}

// ---------------------------------------------------------------------------
// #116 — CLI tests: binary existence check
// ---------------------------------------------------------------------------

/// The neuroscript binary should exist after `cargo build`. This test
/// provides a clear error message if the binary is missing, rather than
/// letting Command::new() panic with a generic OS error.
#[test]
fn issue_116_binary_exists() {
    let mut bin_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    bin_path.push("target");
    bin_path.push("debug");
    bin_path.push("neuroscript");
    assert!(
        bin_path.exists(),
        "neuroscript binary not found at {:?} — run `cargo build` first",
        bin_path
    );
}
