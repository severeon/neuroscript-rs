# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuroScript is a neural architecture composition language implemented in Rust. It compiles neural network architectures into PyTorch modules (future: ONNX, JAX). The language treats "neurons" as first-class composable units with typed tensor shapes.

**Core philosophy**: Neurons all the way down - everything is a neuron, and neurons compose into neurons.

## Build and Test Commands

```bash
# Build the project
cargo build --release

# Parse a file (prints IR structure: imports, neurons, connections)
./target/release/neuroscript examples/residual.ns

# Parse and validate (checks neuron existence, arity, cycles, shapes)
./target/release/neuroscript --validate examples/residual.ns

# Generate PyTorch code for a specific neuron
./target/release/neuroscript --codegen ResidualBlock examples/residual.ns
# ... or write to file
./target/release/neuroscript --codegen ResidualBlock --output residual.py examples/residual.ns

# CLI flags:
# --validate         Run validation checks on the parsed program
# --codegen <name>   Generate PyTorch code for the specified neuron
# --output <file>    Write codegen output to file (stdout if omitted)

# Run all unit tests
cargo test

# Run unit tests with output
cargo test -- --nocapture

# Run specific test module
cargo test lexer              # Lexer tests only
cargo test parser             # Parser tests only
cargo test validator          # Validation tests only
cargo test shape_algebra      # Shape algebra tests only
cargo test shape_inference    # Shape inference tests only
cargo test stdlib_registry    # Registry tests only
cargo test codegen            # Codegen tests only

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
                                        Shape Inference
                                        Shape Algebra
                                        Stdlib Registry
```

### Module Organization

The codebase is organized into modular subdirectories:

```
src/
├── lib.rs              # Public API (parse, validate, generate_pytorch)
├── main.rs             # CLI entry point
├── interfaces.rs       # Central type definitions (IR, errors, traits)
├── ir.rs               # (deprecated - types moved to interfaces.rs)
├── lexer/
│   ├── mod.rs         # Re-exports
│   ├── core.rs        # Lexer implementation
│   └── token.rs       # Token types
├── parser/
│   ├── mod.rs         # Re-exports
│   └── core.rs        # Parser implementation
├── validator/
│   ├── mod.rs         # Re-exports
│   └── core.rs        # Validation logic
├── shape/
│   ├── mod.rs         # Re-exports
│   ├── algebra.rs     # Shape pattern matching and operations
│   └── inference.rs   # Shape inference engine
├── codegen/
│   ├── mod.rs         # Re-exports
│   ├── generator.rs   # Main CodeGenerator struct
│   ├── instantiation.rs # Module instantiation (__init__)
│   ├── forward.rs     # Forward pass generation
│   └── utils.rs       # Helper functions
└── stdlib_registry.rs  # Primitive implementation registry
```

### 1. Lexer (`src/lexer/`)
- **Indentation-aware tokenization**: Tracks indent/dedent for Python-style blocks
- Produces tokens with span information for error reporting
- Handles keywords, operators, literals, and structural tokens (Indent/Dedent/Newline)
- **Key detail**: Indentation is significant - pipelines can be single-line (`a -> b -> c`) or multi-line with indentation

### 2. Parser (`src/parser/`)
- **Recursive descent** parser that converts tokens to IR
- Uses `miette` for diagnostic-quality error messages with source spans
- Returns `Result<Program, ParseError>` with structured error types
- **Important**: Parser tracks position in token stream; uses `peek()`, `at()`, `expect()` pattern

### 3. IR (`src/interfaces.rs`)
- **Algebraic data types** defining the full AST
- Key types:
  - `Program`: Top-level container with `uses` and `neurons` HashMap
  - `NeuronDef`: A neuron definition with params, inputs, outputs, and body
  - `NeuronBody`: Either `Primitive(ImplRef)` or `Graph(Vec<Connection>)`
  - `Connection`: Links `Endpoint` → `Endpoint` in a dataflow graph
  - `Endpoint`: Can be `Ref`, `Tuple`, `Call`, or `Match`
  - `Shape`: Tensor shapes like `[*, dim]` with dimension expressions
  - `PortRef`: References to ports (e.g., `in`, `out`, `fork.left`)
  - `InferenceContext`: Tracks resolved dimensions and node outputs during shape inference

### 4. Validator (`src/validator/`)
- **Post-parse validation** of the IR graph
- Checks:
  1. All referenced neurons exist
  2. Connection endpoints match (tuple arity, port names)
  3. No cycles in dependency graph
  4. Shape compatibility via shape inference engine
- Returns `Result<(), Vec<ValidationError>>` - collects ALL errors rather than failing fast
- **New**: Integrates shape inference for dimension variable resolution

### 5. Shape System (`src/shape/`)
- **Tensor shape operations** using BigUint arithmetic to avoid overflow
- **`algebra.rs`**: Pattern matching with wildcards and literals
  - `Pattern::matches()`: Match shapes like `[*, 1, *]` against concrete shapes
  - `broadcastable()`: Check if two shapes can broadcast together
  - `refine_axis()` / `coarsen_axis()`: Split/merge dimensions
  - Axiswise operations: `axiswise_le()`, `axiswise_divides()`, `axiswise_gcd()`, `axiswise_lcm()`
- **`inference.rs`**: Shape inference engine
  - Resolves dimension variables (e.g., `dim`, `batch`) across connections
  - Tracks equivalences and constraints
  - Validates shape compatibility throughout the graph

### 6. Stdlib Registry (`src/stdlib_registry.rs`)
- **Maps neuron names to implementation references**
- Tracks primitive neurons and their Python/PyTorch implementations
- `ImplRef` enum with two variants:
  - `External`: External implementations with kwargs
  - `Source`: Source-based implementations (module path + class name)
- Used by codegen to generate correct imports

### 7. Codegen (`src/codegen/`)
- **Phase 0 implementation**: Direct lowering from IR to PyTorch
- **`generator.rs`**: Main CodeGenerator with state tracking
- **`instantiation.rs`**: Generates `__init__` with module instantiation
- **`forward.rs`**: Generates `forward()` with connection graph execution
- **`utils.rs`**: Helper functions for code generation
- Handles:
  - Import generation from stdlib_registry
  - Match expression codegen with dimension binding
  - Parameter passing and shape comments
- **Future**: Optimizations and multiple backends

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
- Pattern match on tensor shapes with dimension capture
- **Basic syntax**: `match: [pattern]: pipeline`
- **With guards**: `match: [*, d] where d > 512: Linear(d, 512) -> out`
- **Dimension binding**: Captured dimensions (e.g., `d`, `seq`) can be:
  - Used in guard conditions (`where d > 512`)
  - Passed as arguments to neuron calls (`Linear(d, 512)`)
  - Referenced in pipeline expressions
- Compiler generates lazy instantiation for modules with captured dimensions

### Let Bindings
- Define reusable neuron instantiations within a neuron definition
- Syntax: `let: name = NeuronCall(args)`
- Enables recursion by binding to self with modified parameters
- Example:
  ```neuroscript
  neuron MyNeuron(depth):
    let:
      recurse = MyNeuron(depth - 1)
    graph:
      in -> match:
        [*] where depth > 0: recurse -> out
        [*]: Identity() -> out
  ```

### Error Handling Philosophy
- Use `miette::Diagnostic` for structured errors with source spans
- Prefer `thiserror::Error` for error types
- Always include context: what failed, where (span), and why
- The IR types use `Display` traits for debugging - shapes print as `[*, dim]`

## Testing Strategy

### Example Files (`examples/`)
Comprehensive test suite with 126+ `.ns` files covering language features:
- `01-comments.ns` through `17-match-dimension-binding.ns`: Core language features
- Many additional test cases for edge cases, patterns, and advanced features
- Real-world examples: `residual.ns`, `transformer_from_stdlib.ns`, etc.
- Codegen test cases: `codegen_demo.ns`

### Standard Library (`stdlib/`)
6 library files with composable neurons:
- `FFN.ns`: Feed-forward networks (3 variants)
- `Residual.ns`: Skip connections (5 variants)
- `MultiHeadAttention.ns`: Attention mechanisms (5 variants)
- `TransformerBlock.ns`: Complete transformer layers (5 variants)
- `TransformerStack.ns`: Stacked transformers (6 variants)
- `MetaNeurons.ns`: Routing and composition (16 neurons)

### Unit Tests
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
- `src/lexer/`: Tokenization, indentation handling
- `src/parser/`: Grammar rules, error cases
- `src/validator/`: Existence, arity, cycles
- `src/shape/algebra.rs`: Pattern matching, broadcasting, operations
- `src/shape/inference.rs`: Shape inference and dimension resolution
- `src/stdlib_registry.rs`: Primitive registry lookups
- `src/codegen/`: Code generation tests

### Integration Test Script
`./test_examples.sh` parses all `.ns` files in `examples/` and `stdlib/` directories

## Common Patterns

### Adding New Token Types
1. Add variant to `TokenKind` in `src/interfaces.rs`
2. Add keyword mapping in lexer's keyword table in `src/lexer/core.rs`
3. Update parser to handle new token in relevant `parse_*` methods in `src/parser/core.rs`

### Adding New IR Nodes
1. Add enum variant to appropriate IR type in `src/interfaces.rs`
2. Add parser logic in `src/parser/core.rs`
3. Add validation logic in `src/validator/core.rs` if needed
4. Add `Display` implementation for debugging in `src/interfaces.rs`

### Extending Validation
1. Add new `ValidationError` variant in `src/interfaces.rs`
2. Implement check in `Validator::validate()` or helper methods in `src/validator/core.rs`
3. Collect errors in `errors` vector (don't fail fast)
4. Update shape inference in `src/shape/inference.rs` if relevant

### Adding Primitives
1. Register in `StdlibRegistry::new()` with `ImplRef` in `src/stdlib_registry.rs`
2. Implement Python class in `neuroscript_runtime/primitives/`
3. Add test case in appropriate example file
4. Verify codegen generates correct import

### Implementing Codegen Features
1. Extend `CodeGenerator` in `src/codegen/generator.rs`
2. Handle new IR patterns in:
   - `src/codegen/forward.rs` for forward pass generation
   - `src/codegen/instantiation.rs` for module instantiation
3. Add helper functions to `src/codegen/utils.rs` if needed
4. Update variable name tracking and binding context if needed
5. Test with example file and verify generated Python

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

## Roadmap

### Phase 1: Core Language ✅ Complete
- ✅ Lexer with indent handling
- ✅ Parser with shape expressions
- ✅ IR with algebraic types (modularized to `interfaces.rs`)
- ✅ Validator (existence, arity, cycles)
- ✅ Shape algebra with pattern matching
- ✅ Python runtime package
- ✅ Standard library registry
- ✅ Comprehensive test suite (126+ examples, 6 stdlib files)

### Phase 2: Codegen ✅ Complete
- ✅ IR → PyTorch `nn.Module` generation
- ✅ Import generation from stdlib_registry
- ✅ Forward pass generation (connection graph traversal)
- ✅ Parameter initialization (`__init__`)
- ✅ Match expression codegen with dimension binding
- ✅ Let bindings for recursion
- ✅ Shape inference integration in validation

### Phase 3: Advanced Features (In Progress)
- ⏳ Full dimension variable type inference across programs
- ⏳ Loop constructs for repeated layers
- ⏳ Higher-order neurons (neuron parameters)
- ⏳ Optimization passes (graph simplification, fusion)
- ⏳ Multiple backends (ONNX, JAX, TorchScript)

### Phase 4: Tooling (Future)
- LSP server for editor support
- PyO3 bindings for Python integration
- Visualization of neuron graphs (GraphViz, D3.js)
- REPL for interactive development
- Package manager for sharing neurons
- Documentation generator from neuron definitions

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
