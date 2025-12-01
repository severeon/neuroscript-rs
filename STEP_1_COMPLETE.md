# Step 1 Complete: Mark Unreachable Arms in Validator

## Changes

1. **`src/validator/core.rs`**:
    - Updated `Validator::validate` to take `&mut Program`.
    - Split validation into two passes:
        - **Pass 1 (Read-only)**: Validates graph structure, cycles, and missing neurons using `validate_neuron_graph`.
        - **Pass 2 (Mutable)**: Validates match expressions and updates `is_reachable` flag using `validate_match_expression`.
    - Updated `validate_match_expression` to:
        - Take `&mut MatchExpr`.
        - Detect shadowed arms.
        - Set `is_reachable = false` for shadowed arms.
        - No longer return `ValidationError::UnreachableMatchArm` (shadowing is now a warning/info, handled by the flag).
    - Updated `validate_neuron_graph` to remove the call to `validate_match_expression` (moved to Pass 2).

2. **`src/lib.rs`**:
    - Updated `validate` function signature to `pub fn validate(program: &mut Program)`.

3. **`src/main.rs`**:
    - Updated `main` function to pass `&mut program` to `validate`.

4. **`src/validator/mod.rs`**:
    - Updated all unit tests to pass `&mut program` to `Validator::validate`.
    - Updated `test_match_pattern_shadowing` to verify `is_reachable` flags instead of expecting a validation error. Added a catch-all arm to satisfy exhaustiveness check.

## Verification

- Ran `cargo test validator` and all tests passed.
- Verified that shadowing correctly marks arms as unreachable without causing validation failure.
