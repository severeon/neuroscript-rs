# Agent Guide: NeuroScript Compiler

## Build & Test
- **Build**: `cargo build --release` or `cargo check` (fast iteration)
- **Test all**: `cargo test` | **Single test**: `cargo test test_name` | **Module**: `cargo test lexer`
- **Snapshots**: `cargo test --test integration_tests && cargo insta review` (accept changes with interactive UI)
- **Integration**: `./test_examples.sh` (all .ns files) | **CLI**: `./target/release/neuroscript compile examples/residual.ns`
- **Python setup**: `source ~/.venv_ai/bin/activate && pip install -e .` (required for runtime)

## Code Style
- **Imports**: Group `std`, `crate`, external crates; use `crate::interfaces::*` for IR types
- **Errors**: Use `thiserror::Error` + `miette::Diagnostic` with source spans; always include context (what/where/why)
- **Types**: Algebraic enums in `src/interfaces.rs`; add `Display` impl for IR types; prefer `Result<T, E>` over panics
- **Naming**: Snake_case functions/vars, PascalCase types/enums, SCREAMING_SNAKE constants
- **Module structure**: Each module in subdirectory with `mod.rs` re-exports + `core.rs` implementation
- **Testing**: Inline `#[cfg(test)] mod tests` using `pretty_assertions::assert_eq`; snapshot changes affect 46+ files
- **Comments**: Doc comments (`///`) for public APIs; use `//!` for module-level docs; explain "why" not "what"
- **Parser patterns**: Never advance past EOF; use `at()` for lookahead; collect all errors, don't fail fast
- **Shape algebra**: Use `num_bigint::BigUint` for dimension arithmetic to prevent overflow
- **ImplRef format**: `<provider>,<library>/<neuron>` e.g., `core,attention/ScaledDotProductAttention` (NOT dot notation)

## Git Workflow
**Always commit completed tasks** (see `.agent/rules/always-commit.md`)
