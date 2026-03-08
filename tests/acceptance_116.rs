//! Acceptance tests for #116: CLI binary existence check and error message stability.
//!
//! The neuroscript binary should exist after `cargo build`. This test provides
//! a clear, actionable error message if the binary is missing, rather than
//! letting Command::new() panic with a generic OS error.

use std::path::PathBuf;
use std::process::Command;

/// Returns the path to the compiled `neuroscript` binary.
fn neuroscript_bin() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");
    path.push("debug");
    path.push("neuroscript");
    path
}

#[test]
fn issue_116_binary_exists_before_cli_tests() {
    let bin = neuroscript_bin();
    assert!(
        bin.exists(),
        "neuroscript binary not found at {:?} — run `cargo build` first",
        bin
    );
}

/// Error message for missing file should be stable and use a single expected string.
/// This test will fail if the error message format changes, prompting an update.
#[test]
fn issue_116_stable_error_message_on_missing_file() {
    let bin = neuroscript_bin();
    if !bin.exists() {
        eprintln!("Skipping: neuroscript binary not built");
        return;
    }

    let output = Command::new(&bin)
        .args(["parse", "nonexistent_file_that_does_not_exist.ns"])
        .output()
        .expect("failed to execute neuroscript binary");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "should fail on missing file");

    // Pin to a single expected error message string.
    // If this fails, update the assertion to match the new stable message.
    assert!(
        stderr.contains("Failed to read"),
        "Expected stable error message containing 'Failed to read', got: {}",
        stderr
    );
}
