# NeuroScript Snapshot Testing

This directory contains comprehensive integration tests for NeuroScript using snapshot testing with the [`insta`](https://insta.rs/) crate.

## What is Snapshot Testing?

Snapshot testing captures the output of your code at a point in time and stores it as a "golden" snapshot. Future test runs compare new output against these snapshots to detect unintended changes.

### Benefits

- **Comprehensive Verification**: See entire outputs (IR structures, generated code) rather than checking individual fields
- **Regression Detection**: Automatically catch when refactoring changes behavior
- **Easy Maintenance**: Update multiple tests at once with `cargo insta review`
- **Documentation**: Snapshots serve as examples of expected compiler output
- **Better Refactoring Confidence**: Know exactly what changed and why

## Snapshot Categories

### 1. Parser IR Snapshots

Tests that parse `.ns` files and snapshot the complete AST structure.

**What we capture:**
- Complete `Program` structures with all neurons
- Connection graphs
- Shape annotations
- Port definitions
- Match expressions

**Test files:**
- `snapshot_parser_ir_*` - Individual notable examples
- `snapshot_all_examples` - All files in `examples/` and `stdlib/`

**Example snapshot location:** `tests/snapshots/integration_tests__parser_ir_residual.snap`

### 2. Codegen Output Snapshots

Tests that generate PyTorch code and snapshot the complete module.

**What we capture:**
- Generated Python class definitions
- Import statements
- `__init__` method with module instantiation
- `forward()` method with connection graph execution
- Match expression codegen with lazy instantiation

**Test files:**
- `snapshot_codegen_simple_linear`
- `snapshot_codegen_match_with_guards`
- `snapshot_codegen_residual_block`

**Example snapshot location:** `tests/snapshots/integration_tests__codegen_residual_block.snap`

### 3. Error Message Snapshots

Tests that verify error messages and diagnostics.

**What we capture:**
- Validation errors with context
- Parse errors with source spans
- Error message formatting

**Test files:**
- `snapshot_error_missing_neuron`
- `snapshot_error_arity_mismatch`
- `snapshot_error_parse_failure`

**Example snapshot location:** `tests/snapshots/integration_tests__error_missing_neuron.snap`

## Workflow

### Running Tests

```bash
# Run all tests (includes snapshot tests)
cargo test

# Run only snapshot tests
cargo test --test integration_tests

# Run with output for debugging
cargo test --test integration_tests -- --nocapture

# Run a specific snapshot test
cargo test snapshot_parser_ir_residual
```

### Reviewing Changes

When test output changes, `insta` will detect the difference and prompt you to review:

```bash
# Review all snapshot changes interactively
cargo insta review

# Accept all changes (use with caution!)
cargo insta accept

# Reject all changes
cargo insta reject

# Test and review in one step
cargo insta test --review
```

### Interactive Review UI

The `cargo insta review` command provides an interactive terminal UI:

- **Shows diffs**: See exactly what changed (old vs new)
- **Per-snapshot decisions**: Accept or reject each change individually
- **Context**: Understand why the snapshot changed

### Updating Snapshots After Intentional Changes

When you make intentional changes to the compiler:

1. **Run tests**: `cargo test` (tests will fail if snapshots don't match)
2. **Review changes**: `cargo insta review`
3. **Verify each diff**: Make sure changes are expected
4. **Accept valid changes**: Press `a` to accept in the review UI
5. **Commit snapshots**: `git add tests/snapshots/` and commit with your code changes

## Snapshot File Format

Snapshots are stored in `tests/snapshots/` with the naming convention:

```
<module_name>__<test_name>.snap
```

For example:
- `integration_tests__parser_ir_residual.snap`
- `integration_tests__codegen_simple_linear.snap`

Each snapshot file contains:
```
---
source: tests/integration_tests.rs
expression: formatted
---
<captured output>
```

## Custom IR Formatting

The tests use custom formatting functions (not raw `Debug` output) for readability:

- **Omits unstable data**: Skips spans, source text, internal IDs
- **Pretty-prints structures**: Indented nested elements
- **Focuses on semantics**: Shows neurons, connections, shapes clearly
- **Human-readable**: Easy to review diffs in `cargo insta review`

See `format_program_ir()` and related functions in `integration_tests.rs`.

## Adding New Snapshot Tests

### 1. Parser IR Test

```rust
#[test]
fn snapshot_parser_ir_my_feature() {
    let source = r#"
neuron MyFeature:
    graph:
        in -> out
"#;

    let program = parse(source).expect("Parse failed");
    let formatted = format_program_ir(&program);

    insta::assert_snapshot!("parser_ir_my_feature", formatted);
}
```

### 2. Codegen Test

```rust
#[test]
fn snapshot_codegen_my_feature() {
    let source = r#"
neuron MyFeature:
    graph:
        in -> Linear(512, 256) -> out

neuron Linear(in_dim, out_dim):
    impl: core,nn/Linear
"#;

    let mut program = parse(source).expect("Parse failed");
    validate(&mut program).expect("Validation failed");

    let code = generate_pytorch(&program, "MyFeature")
        .expect("Codegen failed");

    insta::assert_snapshot!("codegen_my_feature", code);
}
```

### 3. Error Test

```rust
#[test]
fn snapshot_error_my_case() {
    let source = r#"
neuron Broken:
    graph:
        in -> out -> in  // Creates cycle
"#;

    let mut program = parse(source).expect("Parse should succeed");
    let errors = validate(&mut program).unwrap_err();

    let error_text = errors.iter()
        .map(|e| format!("{:?}", e))
        .join("\n\n");

    insta::assert_snapshot!("error_my_case", error_text);
}
```

## Best Practices

### ✅ Do

- **Review all changes carefully**: Don't blindly accept snapshot updates
- **Commit snapshots with code**: Keep them in sync with implementation
- **Use descriptive test names**: Makes it easy to find relevant snapshots
- **Test both success and failure cases**: Verify errors are well-formatted
- **Run `cargo insta review` before committing**: Ensure no pending snapshot changes

### ❌ Don't

- **Don't use `cargo insta accept` without reviewing**: You might accept broken output
- **Don't commit `.snap.new` files**: These are temporary pending snapshots
- **Don't include unstable data in snapshots**: Avoid timestamps, random IDs, etc.
- **Don't make snapshots too large**: Break complex tests into focused cases
- **Don't ignore snapshot test failures**: They indicate behavior changes

## CI Integration

In continuous integration:

```bash
# Fail if snapshots don't match (no interactive review)
cargo test --test integration_tests

# Check for pending snapshot updates
cargo insta test --check
```

Snapshot files should be committed to the repository so CI can verify them.

## Troubleshooting

### "Snapshot does not match"

**Cause**: Your code output changed from the stored snapshot.

**Solution**:
1. Run `cargo insta review`
2. Review the diff to understand what changed
3. If expected, accept the change
4. If unexpected, fix your code

### "Snapshot not found"

**Cause**: First time running the test or snapshot file was deleted.

**Solution**:
1. Run `cargo test` to generate initial snapshot
2. Review with `cargo insta review`
3. Accept if output looks correct
4. Commit the new `.snap` file

### "Snapshots out of date after git checkout"

**Cause**: Switched branches with different snapshot versions.

**Solution**:
1. Run `cargo test` to detect differences
2. Run `cargo insta review` to see changes
3. Either accept changes or switch branches

## Useful Commands

```bash
# Run tests and generate snapshots
cargo test --test integration_tests

# Review changes interactively
cargo insta review

# Accept all pending snapshots (careful!)
cargo insta accept

# Reject all pending snapshots
cargo insta reject

# Delete all .snap.new files
cargo insta reject

# Test specific snapshot
cargo test snapshot_parser_ir_residual

# View snapshot contents
cat tests/snapshots/integration_tests__parser_ir_residual.snap

# Check for pending snapshots in CI
cargo insta test --check

# Update specific test's snapshot
cargo test snapshot_parser_ir_residual && cargo insta review
```

## Resources

- [insta documentation](https://insta.rs/)
- [insta GitHub](https://github.com/mitsuhiko/insta)
- [Snapshot Testing Guide](https://insta.rs/docs/snapshot-testing/)

## Summary

Snapshot testing with `insta` provides:
- ✅ Comprehensive output verification
- ✅ Easy maintenance with `cargo insta review`
- ✅ Automatic regression detection
- ✅ Clear diffs for understanding changes
- ✅ Better refactoring confidence

Run `cargo test` to execute tests, then `cargo insta review` to manage snapshot changes.
