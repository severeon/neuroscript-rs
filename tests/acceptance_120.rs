//! Regression test for #120: embeddings.ns parse failure.
//!
//! The file previously used Reshape([...]) bracket syntax at line 75
//! but was updated to use fat arrow (=>) syntax in PR #83.

use neuroscript::parse;
use std::fs;
use std::path::PathBuf;

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
