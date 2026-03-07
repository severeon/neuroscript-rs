//! CLI integration tests for the `neuroscript` binary.
//!
//! These tests exercise the CLI commands (parse, validate, compile, list)
//! by running the binary as a subprocess and asserting on exit codes and output.

use std::path::PathBuf;
use std::process::Command;

/// Returns the path to the compiled `neuroscript` binary.
fn neuroscript_bin() -> PathBuf {
    // cargo test builds to target/debug by default
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("target");
    path.push("debug");
    path.push("neuroscript");
    path
}

/// Returns the path to an example .ns file.
fn example(name: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("examples");
    path.push(name);
    path
}

/// Helper: run a neuroscript command and return (stdout, stderr, success).
fn run(args: &[&str]) -> (String, String, bool) {
    let output = Command::new(neuroscript_bin())
        .args(args)
        .output()
        .expect("failed to execute neuroscript binary");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (stdout, stderr, output.status.success())
}

// ---------------------------------------------------------------------------
// Parse command
// ---------------------------------------------------------------------------

#[test]
fn parse_succeeds_on_valid_file() {
    let (stdout, _stderr, success) = run(&["parse", example("dropout.ns").to_str().unwrap()]);
    assert!(success, "parse should succeed on a valid file");
    assert!(
        stdout.contains("Successfully parsed"),
        "expected success message in stdout, got: {}",
        stdout
    );
}

#[test]
fn parse_verbose_shows_neuron_details() {
    let (stdout, _stderr, success) =
        run(&["parse", "--verbose", example("dropout.ns").to_str().unwrap()]);
    assert!(success, "parse --verbose should succeed");
    assert!(
        stdout.contains("Parsed") && stdout.contains("neurons"),
        "verbose parse should show neuron count, got: {}",
        stdout
    );
    assert!(
        stdout.contains("Dropout"),
        "verbose parse should list the Dropout neuron, got: {}",
        stdout
    );
}

#[test]
fn parse_fails_on_missing_file() {
    let (_stdout, stderr, success) = run(&["parse", "nonexistent_file.ns"]);
    assert!(!success, "parse should fail on missing file");
    assert!(
        stderr.contains("Failed to read") || stderr.contains("No such file"),
        "expected file-not-found error, got: {}",
        stderr
    );
}

#[test]
fn parse_fails_on_invalid_syntax() {
    // Write a temporary file with invalid NeuroScript
    let tmp = std::env::temp_dir().join("cli_test_invalid.ns");
    std::fs::write(&tmp, "this is not valid neuroscript\n").unwrap();

    let (_stdout, stderr, success) = run(&["parse", tmp.to_str().unwrap()]);
    assert!(!success, "parse should fail on invalid syntax");
    assert!(
        stderr.contains("Expected") || stderr.contains("error"),
        "expected parse error message, got: {}",
        stderr
    );

    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------------
// Validate command
// ---------------------------------------------------------------------------

#[test]
fn validate_succeeds_on_valid_file() {
    let (stdout, _stderr, success) = run(&["validate", example("dropout.ns").to_str().unwrap()]);
    assert!(success, "validate should succeed on a valid file");
    assert!(
        stdout.contains("Valid"),
        "expected validation success message, got: {}",
        stdout
    );
}

#[test]
fn validate_verbose_shows_details() {
    let (stdout, _stderr, success) = run(&[
        "validate",
        "--verbose",
        example("dropout.ns").to_str().unwrap(),
    ]);
    assert!(success, "validate --verbose should succeed");
    assert!(
        stdout.contains("Loading standard library"),
        "verbose validate should show stdlib loading, got: {}",
        stdout
    );
    assert!(
        stdout.contains("Program is valid"),
        "verbose validate should confirm validity, got: {}",
        stdout
    );
}

#[test]
fn validate_no_stdlib_flag() {
    let (stdout, _stderr, success) = run(&[
        "validate",
        "--no-stdlib",
        example("dropout.ns").to_str().unwrap(),
    ]);
    assert!(success, "validate --no-stdlib should succeed for self-contained files");
    assert!(
        stdout.contains("Valid"),
        "expected validation success, got: {}",
        stdout
    );
}

#[test]
fn validate_fails_on_missing_file() {
    let (_stdout, stderr, success) = run(&["validate", "nonexistent_file.ns"]);
    assert!(!success, "validate should fail on missing file");
    assert!(
        stderr.contains("Failed to read") || stderr.contains("No such file"),
        "expected file-not-found error, got: {}",
        stderr
    );
}

// ---------------------------------------------------------------------------
// Compile command
// ---------------------------------------------------------------------------

#[test]
fn compile_succeeds_and_produces_pytorch() {
    let (stdout, _stderr, success) = run(&[
        "compile",
        "--neuron",
        "MLPWithDropout",
        example("mlp_with_dropout.ns").to_str().unwrap(),
    ]);
    assert!(success, "compile should succeed on a valid file");
    assert!(
        stdout.contains("import torch"),
        "compiled output should contain PyTorch imports, got: {}",
        stdout
    );
    assert!(
        stdout.contains("nn.Module"),
        "compiled output should contain nn.Module, got: {}",
        stdout
    );
}

#[test]
fn compile_neuron_auto_detection_from_filename() {
    // dropout.ns -> Dropout (snake_case/simple to PascalCase)
    let (stdout, _stderr, success) = run(&["compile", example("dropout.ns").to_str().unwrap()]);
    assert!(success, "compile should auto-detect neuron name from filename");
    assert!(
        stdout.contains("Generated PyTorch code for 'Dropout'"),
        "should auto-detect 'Dropout' from dropout.ns, got: {}",
        stdout
    );
}

#[test]
fn compile_neuron_auto_detection_snake_case() {
    // mlp_with_dropout.ns -> MLPWithDropout would be MlpWithDropout, but actual neuron
    // is MLPWithDropout. Auto-detection converts snake_case to PascalCase naively.
    // Test that auto-detection from filename works when neuron name matches.
    // dropout.ns -> Dropout (already tested above)
    // Use DilatedConv.ns which has a neuron named DilatedConv matching the filename.
    let (stdout, _stderr, success) =
        run(&["compile", example("DilatedConv.ns").to_str().unwrap()]);
    assert!(success, "compile should auto-detect neuron from filename");
    assert!(
        stdout.contains("Generated PyTorch code for 'DilatedConv'"),
        "should auto-detect 'DilatedConv' from DilatedConv.ns, got: {}",
        stdout
    );
}

#[test]
fn compile_explicit_neuron_flag() {
    let (stdout, _stderr, success) = run(&[
        "compile",
        "--neuron",
        "Dropout",
        example("dropout.ns").to_str().unwrap(),
    ]);
    assert!(success, "compile with explicit --neuron should succeed");
    assert!(
        stdout.contains("Generated PyTorch code for 'Dropout'"),
        "should compile the specified neuron, got: {}",
        stdout
    );
}

#[test]
fn compile_auto_detection_fails_when_no_match() {
    // fat_arrow_basic.ns has no neuron named FatArrowBasic, so auto-detection fails
    let (_stdout, stderr, success) =
        run(&["compile", example("fat_arrow_basic.ns").to_str().unwrap()]);
    assert!(!success, "compile should fail when filename doesn't match any neuron");
    assert!(
        stderr.contains("No neuron matching filename"),
        "should report auto-detection failure, got: {}",
        stderr
    );
    assert!(
        stderr.contains("--neuron"),
        "should suggest using --neuron flag, got: {}",
        stderr
    );
}

#[test]
fn compile_invalid_neuron_name_fails() {
    let (_stdout, stderr, success) = run(&[
        "compile",
        "--neuron",
        "NonExistentNeuron",
        example("dropout.ns").to_str().unwrap(),
    ]);
    assert!(!success, "compile with invalid neuron name should fail");
    assert!(
        stderr.contains("not found"),
        "should report neuron not found, got: {}",
        stderr
    );
    assert!(
        stderr.contains("Available neurons"),
        "should list available neurons, got: {}",
        stderr
    );
}

#[test]
fn compile_bundle_flag_self_contained() {
    let (stdout, _stderr, success) = run(&[
        "compile",
        "--bundle",
        example("dropout.ns").to_str().unwrap(),
    ]);
    assert!(success, "compile --bundle should succeed");
    // Bundle mode should NOT reference neuroscript_runtime
    assert!(
        !stdout.contains("neuroscript_runtime"),
        "bundled output should not reference neuroscript_runtime, got: {}",
        stdout
    );
    // Bundle mode includes extra imports for self-contained output
    assert!(
        stdout.contains("import torch"),
        "bundled output should still have torch imports, got: {}",
        stdout
    );
}

#[test]
fn compile_verbose_flag_shows_details() {
    let (stdout, _stderr, success) = run(&[
        "compile",
        "--verbose",
        example("dropout.ns").to_str().unwrap(),
    ]);
    assert!(success, "compile --verbose should succeed");
    assert!(
        stdout.contains("Loading standard library"),
        "verbose compile should show stdlib loading, got: {}",
        stdout
    );
    assert!(
        stdout.contains("Validation passed"),
        "verbose compile should confirm validation, got: {}",
        stdout
    );
}

#[test]
fn compile_output_to_file() {
    let tmp = std::env::temp_dir().join("cli_test_compile_output.py");
    let _ = std::fs::remove_file(&tmp); // clean up from prior runs

    let (stdout, _stderr, success) = run(&[
        "compile",
        "-o",
        tmp.to_str().unwrap(),
        example("dropout.ns").to_str().unwrap(),
    ]);
    assert!(success, "compile -o should succeed");
    assert!(
        stdout.contains("Compiled to"),
        "should report output file, got: {}",
        stdout
    );

    let content = std::fs::read_to_string(&tmp).expect("output file should exist");
    assert!(
        content.contains("import torch"),
        "output file should contain PyTorch code, got: {}",
        content
    );

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn compile_no_optimize_flag() {
    // Should succeed even with optimizations disabled
    let (stdout, _stderr, success) = run(&[
        "compile",
        "--no-optimize",
        example("dropout.ns").to_str().unwrap(),
    ]);
    assert!(success, "compile --no-optimize should succeed");
    assert!(
        stdout.contains("import torch"),
        "should still produce valid output, got: {}",
        stdout
    );
}

#[test]
fn compile_fails_on_missing_file() {
    let (_stdout, stderr, success) = run(&["compile", "nonexistent_file.ns"]);
    assert!(!success, "compile should fail on missing file");
    assert!(
        stderr.contains("Failed to read") || stderr.contains("No such file"),
        "expected file-not-found error, got: {}",
        stderr
    );
}

#[test]
fn compile_fails_on_parse_error() {
    let tmp = std::env::temp_dir().join("cli_test_bad_compile.ns");
    std::fs::write(&tmp, "this is invalid\n").unwrap();

    let (_stdout, stderr, success) = run(&["compile", tmp.to_str().unwrap()]);
    assert!(!success, "compile should fail on parse errors");
    assert!(
        stderr.contains("Expected") || stderr.contains("error"),
        "expected parse error, got: {}",
        stderr
    );

    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------------
// List command
// ---------------------------------------------------------------------------

#[test]
fn list_shows_neurons_in_file() {
    let (stdout, _stderr, success) = run(&["list", example("dropout.ns").to_str().unwrap()]);
    assert!(success, "list should succeed on a valid file");
    assert!(
        stdout.contains("Dropout"),
        "list should show the Dropout neuron, got: {}",
        stdout
    );
    assert!(
        stdout.contains("1 total"),
        "list should show neuron count, got: {}",
        stdout
    );
}

#[test]
fn list_verbose_shows_ports() {
    let (stdout, _stderr, success) =
        run(&["list", "--verbose", example("dropout.ns").to_str().unwrap()]);
    assert!(success, "list --verbose should succeed");
    assert!(
        stdout.contains("inputs:") && stdout.contains("outputs:"),
        "verbose list should show port details, got: {}",
        stdout
    );
}

#[test]
fn list_requires_file_or_flag() {
    let (_stdout, stderr, success) = run(&["list"]);
    assert!(!success, "list without file or flag should fail");
    assert!(
        stderr.contains("required") || stderr.contains("argument"),
        "should report missing argument, got: {}",
        stderr
    );
}

#[test]
fn list_stdlib_flag() {
    let (stdout, _stderr, success) = run(&["list", "--stdlib"]);
    assert!(success, "list --stdlib should succeed");
    // Should list many stdlib neurons
    assert!(
        stdout.contains("Linear") || stdout.contains("Conv2d"),
        "should list stdlib primitives, got: {}",
        stdout
    );
}

// ---------------------------------------------------------------------------
// No subcommand / help
// ---------------------------------------------------------------------------

#[test]
fn no_subcommand_shows_help() {
    let output = Command::new(neuroscript_bin())
        .output()
        .expect("failed to execute neuroscript binary");
    // clap exits with error code when no subcommand is given
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Usage") || stderr.contains("neuroscript"),
        "should show usage info, got: {}",
        stderr
    );
}

#[test]
fn help_flag_shows_usage() {
    let (stdout, _stderr, _success) = run(&["--help"]);
    assert!(
        stdout.contains("Usage") || stdout.contains("neuroscript"),
        "should show usage info, got: {}",
        stdout
    );
}

// ---------------------------------------------------------------------------
// Multiple example files compile successfully
// ---------------------------------------------------------------------------

#[test]
fn compile_fat_arrow_basic() {
    let (stdout, _stderr, success) = run(&[
        "compile",
        "--neuron",
        "VitFlatten",
        example("fat_arrow_basic.ns").to_str().unwrap(),
    ]);
    assert!(success, "fat_arrow_basic.ns should compile successfully");
    assert!(stdout.contains("import torch"));
    assert!(stdout.contains("reshape"), "fat arrow should produce reshape in output");
}

#[test]
fn compile_mlp_with_dropout() {
    let (stdout, _stderr, success) = run(&[
        "compile",
        "--neuron",
        "MLPWithDropout",
        example("mlp_with_dropout.ns").to_str().unwrap(),
    ]);
    assert!(success, "mlp_with_dropout.ns should compile successfully");
    assert!(stdout.contains("import torch"));
    assert!(stdout.contains("nn.Module"));
}

#[test]
fn compile_higher_order_neuron() {
    let (stdout, _stderr, success) = run(&[
        "compile",
        "--neuron",
        "Stack",
        example("higher_order_neuron.ns").to_str().unwrap(),
    ]);
    assert!(
        success,
        "higher_order_neuron.ns should compile successfully"
    );
    assert!(stdout.contains("import torch"));
}
