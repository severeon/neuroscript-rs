# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuroScript is a neural architecture composition language implemented in Rust. It compiles neural network architectures into PyTorch modules (future: ONNX, JAX). The language treats "neurons" as first-class composable units with typed tensor shapes.

**Core philosophy**: Neurons all the way down - everything is a neuron, and neurons compose into neurons.

## Build and Test Commands

```bash
# Build the project
cargo build --release

# Run the parser on a file
./target/release/neuroscript examples/residual.ns

# Run all unit tests
cargo test

# Run unit tests with output
cargo test -- --nocapture

# Test all example files
./test_examples.sh

# Run a single test by name
cargo test test_name

# Check without building
cargo check
```

## Architecture: Three-Phase Compiler

```
Source (.ns) → Lexer → Tokens → Parser → IR → Validator → Codegen (future)
```

### 1. Lexer (`src/lexer.rs`)
- **Indentation-aware tokenization**: Tracks indent/dedent for Python-style blocks
- Produces tokens with span information for error reporting
- Handles keywords, operators, literals, and structural tokens (Indent/Dedent/Newline)
- **Key detail**: Indentation is significant - pipelines can be single-line (`a -> b -> c`) or multi-line with indentation

### 2. Parser (`src/parser.rs`)
- **Recursive descent** parser that converts tokens to IR
- Uses `miette` for diagnostic-quality error messages with source spans
- Returns `Result<Program, ParseError>` with structured error types
- **Important**: Parser tracks position in token stream; uses `peek()`, `at()`, `expect()` pattern

### 3. IR (`src/ir.rs`)
- **Algebraic data types** defining the full AST
- Key types:
  - `Program`: Top-level container with `uses` and `neurons` HashMap
  - `NeuronDef`: A neuron definition with params, inputs, outputs, and body
  - `NeuronBody`: Either `Primitive(ImplRef)` or `Graph(Vec<Connection>)`
  - `Connection`: Links `Endpoint` → `Endpoint` in a dataflow graph
  - `Endpoint`: Can be `Ref`, `Tuple`, `Call`, or `Match`
  - `Shape`: Tensor shapes like `[*, dim]` with dimension expressions
  - `PortRef`: References to ports (e.g., `in`, `out`, `fork.left`)

### 4. Validator (`src/validator.rs`)
- **Post-parse validation** of the IR graph
- Checks:
  1. All referenced neurons exist
  2. Connection endpoints match (tuple arity, port names)
  3. No cycles in dependency graph
- Returns `Result<(), Vec<ValidationError>>` - collects ALL errors rather than failing fast

## Key Language Concepts

### Neurons
Two types:
- **Primitive**: Has `impl:` reference to external code (e.g., `impl: core,nn/Linear`)
- **Composite**: Has `graph:` section with internal connections

### Ports
- Default port: `in` and `out` (named "default" internally)
- Named ports: `in left: [*shape]`, `out a: [*shape]`
- Port references: `in`, `out`, `fork.left`, `fork.a`

### Connections and Pipelines
- Simple: `in -> Linear(512, 256) -> out`
- Multi-line: Indentation creates pipeline continuation
- Tuple unpacking: `in -> Fork() -> (main, skip)` creates two named references
- Port access: `main -> MLP(dim) -> processed`

### Shapes
- Literal dimensions: `[512, 256]`
- Named dimensions: `[batch, seq, dim]`
- Wildcards: `[*, dim]` (single dimension), `[*shape]` (variadic)
- Expressions: `[dim * 4]`, `[seq - 1]`

## Critical Implementation Details

### Parser State Management
- Parser maintains `pos` index into token vector
- **Never advance past EOF**: `peek()` returns last token (EOF) when at end
- Use `at(&TokenKind)` for lookahead, not `peek().kind ==` (discriminant comparison)
- `expect()` consumes and validates, `eat()` optionally consumes

### Tuple Unpacking Grammar
```rust
// WRONG: Tuple unpacking only works for port references in connections
in -> Fork() -> (a, b, c)  // Creates references a, b, c

// RIGHT: Not for inline calls
Linear(dim, dim * 4)  // Call with args, not tuple
```

### Match Expressions
- Pattern match on tensor shapes
- Syntax: `match: [pattern]: pipeline`

### Error Handling Philosophy
- Use `miette::Diagnostic` for structured errors with source spans
- Prefer `thiserror::Error` for error types
- Always include context: what failed, where (span), and why
- The IR types use `Display` traits for debugging - shapes print as `[*, dim]`

## Testing Strategy

### Example Files (`examples/`)
Comprehensive test suite with numbered examples covering each language feature:
- `01-comments.ns` through `15-edge-cases.ns`: Individual features
- `comprehensive.ns`: Large integration test
- `residual.ns`: Real-world residual network example

### Unit Tests
Located inline in source files (Rust convention). Pattern:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    // ...
}
```

### Test Script
`./test_examples.sh` validates all `.ns` files parse successfully. Checks for "Parsed" output.

## Common Patterns

### Adding New Token Types
1. Add variant to `TokenKind` in `src/lexer.rs`
2. Add keyword mapping in lexer's keyword table if applicable
3. Update parser to handle new token in relevant `parse_*` methods

### Adding New IR Nodes
1. Add enum variant to appropriate IR type in `src/ir.rs`
2. Add parser logic in `src/parser.rs`
3. Add validation logic in `src/validator.rs` if needed
4. Add `Display` implementation for debugging

### Extending Validation
1. Add new `ValidationError` variant in `src/validator.rs`
2. Implement check in `Validator::validate()` or helper methods
3. Collect errors in `errors` vector (don't fail fast)

## Future Roadmap (from README.md)

1. Codegen: IR → PyTorch `nn.Module`
2. Shape inference and validation
3. PyO3 bindings for Python integration
4. LSP server for editor support

## Development Notes

- **Fast iteration**: `cargo check` is faster than `cargo build` for syntax checking
- **Error quality matters**: This is a language - users need clear diagnostics
- **Algebraic types are the architecture**: The IR perfectly maps to Rust enums, making pattern matching natural
- **Indentation is structural**: Like Python, indentation defines scope in pipelines
