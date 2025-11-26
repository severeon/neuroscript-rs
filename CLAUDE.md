# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuroScript is a neural architecture composition language implemented in Rust. It compiles neural network architectures into PyTorch modules (future: ONNX, JAX). The language treats "neurons" as first-class composable units with typed tensor shapes.

**Core philosophy**: Neurons all the way down - everything is a neuron, and neurons compose into neurons.

## Build and Test Commands

```bash
# Build the project
cargo build --release

# Parse a file (prints IR structure)
./target/release/neuroscript examples/residual.ns

# Parse and validate
./target/release/neuroscript --validate examples/residual.ns

# Generate PyTorch code for a specific neuron
./target/release/neuroscript --codegen ResidualBlock --output residual.py examples/residual.ns

# Run all unit tests
cargo test

# Run unit tests with output
cargo test -- --nocapture

# Run specific test module
cargo test lexer              # Lexer tests only
cargo test parser             # Parser tests only
cargo test validator          # Validation tests only
cargo test shape_algebra      # Shape algebra tests only
cargo test stdlib_registry    # Registry tests only

# Run a single test by name
cargo test test_name

# Test all example files (integration test)
./test_examples.sh

# Check without building (fast iteration)
cargo check
```

## Architecture: Five-Phase Compiler

```
Source (.ns) → Lexer → Tokens → Parser → IR → Validator → Codegen → PyTorch
                                              ↓
                                        Shape Algebra
                                        Stdlib Registry
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

### 5. Shape Algebra (`src/shape_algebra.rs`)
- **Tensor shape operations** using BigUint arithmetic to avoid overflow
- Provides pattern matching with wildcards and literals
- Key operations:
  - `Pattern::matches()`: Match shapes like `[*, 1, *]` against concrete shapes
  - `broadcastable()`: Check if two shapes can broadcast together
  - `refine_axis()` / `coarsen_axis()`: Split/merge dimensions
  - Axiswise operations: `axiswise_le()`, `axiswise_divides()`, `axiswise_gcd()`, `axiswise_lcm()`
- Used for shape validation and future inference

### 6. Stdlib Registry (`src/stdlib_registry.rs`)
- **Maps neuron names to implementation references**
- Tracks primitive neurons and their Python/PyTorch implementations
- `ImplRef` structure contains module path, class name, and description
- Used by codegen to generate correct imports

### 7. Codegen (`src/codegen.rs`)
- **Phase 0 implementation**: Direct lowering from IR to PyTorch
- Generates Python code with `nn.Module` classes
- Handles:
  - Import generation from stdlib_registry
  - `__init__` with module instantiation
  - `forward()` with connection graph execution
  - Parameter passing and shape comments
- **Future**: Shape inference integration and optimizations

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
Comprehensive test suite with 20 numbered examples covering each language feature:
- `01-comments.ns` through `15-edge-cases.ns`: Individual features
- `comprehensive.ns`: Large integration test
- `residual.ns`: Real-world residual network
- `codegen_demo.ns`: Codegen test cases
- **Total**: 20 example files

### Standard Library (`stdlib/`)
6 library files with composable neurons:
- `FFN.ns`: Feed-forward networks (3 variants)
- `Residual.ns`: Skip connections (5 variants)
- `MultiHeadAttention.ns`: Attention mechanisms (5 variants)
- `TransformerBlock.ns`: Complete transformer layers (5 variants)
- `TransformerStack.ns`: Stacked transformers (6 variants)
- `MetaNeurons.ns`: Routing and composition (16 neurons)

### Unit Tests (32 tests)
Located inline in source files (Rust convention):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    // ...
}
```

Organized by module:
- Lexer tests: Tokenization, indentation handling
- Parser tests: Grammar rules, error cases
- Validator tests: Existence, arity, cycles
- Shape algebra tests: Pattern matching, broadcasting, operations
- Stdlib registry tests: Mapping and lookups

### Integration Test Script
`./test_examples.sh` validates all 26 files (20 examples + 6 stdlib) parse successfully

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

### Adding Primitives
1. Register in `StdlibRegistry::new()` with `ImplRef` in `src/stdlib_registry.rs`
2. Implement Python class in `neuroscript_runtime/primitives/`
3. Add test case in appropriate example file
4. Verify codegen generates correct import

### Implementing Codegen Features
1. Extend `CodeGenerator` methods in `src/codegen.rs`
2. Handle new IR patterns in `generate_forward()` or `generate_init()`
3. Update variable name tracking if needed
4. Test with example file and verify generated Python

## Python Runtime Integration

NeuroScript compiles to PyTorch, requiring a Python runtime:
```bash
# Install the Python runtime package (in project root)
pip install -e .
```

The runtime provides:
- Primitive neuron implementations (`neuroscript_runtime.primitives.*`)
- Core utilities for shape handling
- Generated code imports from this package

Generated PyTorch modules are standalone after runtime is installed.

## Future Roadmap

### Phase 2: Codegen (In Progress)
- ✅ IR → PyTorch `nn.Module`
- ✅ Import generation from stdlib_registry
- ⏳ Shape inference integration
- ✅ Forward pass generation
- ✅ Parameter initialization

### Phase 3: Advanced Features
- Shape inference for dimension variables
- Loop constructs for repeated layers
- Higher-order neurons (neuron parameters)
- Optimization passes
- Multiple backends (ONNX, JAX)

### Phase 4: Tooling
- LSP server for editor support
- PyO3 bindings for Python integration
- Visualization of neuron graphs
- REPL for interactive development
- Package manager for sharing neurons

## Key Dependencies

### Rust Crates (from Cargo.toml)
- `thiserror` (1.0): Clean error type definitions with derive macros
- `miette` (7.x): Beautiful diagnostic error reporting with source spans and fancy formatting
- `num-bigint` (0.4): Arbitrary precision integers for shape algebra (prevents overflow)
- `num-integer` (0.1): Integer traits for gcd, lcm operations
- `num-traits` (0.2): Numeric traits (Zero, One) for generic arithmetic
- `pretty_assertions` (1.4): Enhanced test output with colored diffs (dev-only)

### Python Runtime (separate package)
- Located in project root with `setup.py` / `pyproject.toml`
- Install with `pip install -e .`
- Provides `neuroscript_runtime.primitives.*` modules for generated code

## Development Notes

- **Fast iteration**: `cargo check` is faster than `cargo build` for syntax checking
- **Error quality matters**: This is a language - users need clear diagnostics with miette spans
- **Algebraic types are the architecture**: The IR perfectly maps to Rust enums, making pattern matching natural
- **Indentation is structural**: Like Python, indentation defines scope in pipelines
- **Shape algebra uses BigUint**: Prevents overflow when computing tensor sizes (e.g., `[1000, 1000, 1000]` = 1 billion elements)
- **Codegen is string-based**: Phase 0 directly emits Python strings - future phases may use a PyTorch IR
