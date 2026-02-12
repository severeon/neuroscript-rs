# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Progress & Known Issues

See [docs/PROGRESS.md](docs/PROGRESS.md) for session-by-session progress, known bugs, key decisions, and exploration findings that reduce future ramp-up time.

## Ground Rules

* if implementing a feature: stash, checkout main, create a feature branch
* commit regularly with clear messages

## Project Overview

NeuroScript is a neural architecture composition language implemented in Rust. It compiles neural network architectures into PyTorch modules (future: ONNX, JAX). The language treats "neurons" as first-class composable units with typed tensor shapes.

**Core philosophy**: Neurons all the way down - everything is a neuron, and neurons compose into neurons.

## NeuroScript Language Constraints

When writing or generating NeuroScript (.ns) files:

* Binding blocks use `context:` keyword (not `let:`), with optional annotations: `@lazy`, `@static`, `@global`
* Recursive bindings require `@lazy` annotation with arguments that change (e.g., `depth - 1`)
* Only one variadic dimension per shape (e.g., `[*shape, dim]` works, `[*a, *b]` does not)
* Implicit fork is preferred for splitting: `in -> (a, b, c)` — any single output → N-way tuple. Explicit Fork/Fork3 only for named port access
* Shape dimension expressions support +, -, *, / but solver handles only simple single-unknown equations
* Always validate generated .ns files with the CLI before considering work complete

## Compilation & Validation

* Always run compilation/validation after any code changes to .ns files, Rust files, or playground code
* Do not consider a task complete until the build passes and tests run clean
* If a fix introduces new errors, fix those before reporting success

## Task Approach

* For implementation tasks, start coding promptly after brief exploration
* When using swarm/sub-agents for batch creation, target 1-2 items per agent to ensure quality
* When the user specifies a skill or tool to use, use it immediately without substituting your own approach

## Build and Test Commands

```bash
# Build the project
cargo build --release

# ============================================================================
# CLI: Parse, Validate, Compile, List, Package Management
# ============================================================================

# Parse a file and show structure (quiet mode by default)
./target/release/neuroscript parse examples/residual.ns

# Parse with detailed IR structure output
./target/release/neuroscript parse --verbose examples/residual.ns

# Validate a file (parse + validation checks)
./target/release/neuroscript validate examples/residual.ns

# Validate with detailed output
./target/release/neuroscript validate --verbose examples/residual.ns

# Compile to PyTorch
# (auto-detects neuron name from filename, e.g., residual.ns → Residual)
./target/release/neuroscript compile examples/residual.ns

# Compile specific neuron
./target/release/neuroscript compile examples/residual.ns --neuron ResidualBlock

# Compile with output to file
./target/release/neuroscript compile examples/residual.ns -o residual.py

# Compile with verbose output (shows optimization stats)
./target/release/neuroscript compile examples/residual.ns --verbose

# Compile without optimizations
./target/release/neuroscript compile examples/residual.ns --no-optimize

# Disable dead branch elimination only (but keep pattern reordering)
./target/release/neuroscript compile examples/residual.ns --no-dead-elim

# List all neurons in a file
./target/release/neuroscript list examples/residual.ns

# List with connection details
./target/release/neuroscript list --verbose examples/residual.ns

# Package management
./target/release/neuroscript init          # Initialize a new package (Axon.toml)
./target/release/neuroscript add <dep>     # Add a dependency
./target/release/neuroscript fetch         # Fetch all dependencies

# Get help for any command
./target/release/neuroscript --help
./target/release/neuroscript parse --help
./target/release/neuroscript compile --help

# ============================================================================
# Testing
# ============================================================================

# Run all unit tests
cargo test

# Run unit tests with output
cargo test -- --nocapture

# Run specific test module
cargo test grammar            # Grammar/parser tests only
cargo test validator          # Validation tests only
cargo test shape_algebra      # Shape algebra tests only
cargo test shape_inference    # Shape inference tests only
cargo test stdlib_registry    # Registry tests only
cargo test codegen            # Codegen tests only
cargo test optimizer          # Optimizer tests only

# Run a single test by name
cargo test test_name

# Test all example files (integration test)
./test_examples.sh

# Snapshot testing with insta
cargo test --test integration_tests  # Run integration snapshot tests
cargo insta review                   # Review snapshot changes interactively
cargo insta accept                   # Accept all snapshot changes
cargo insta test --review            # Test and review in one step

# Check without building (fast iteration)
cargo check
```

### CLI Subcommands

The CLI uses clap with seven subcommands:

**`parse <FILE>`**

* Parse NeuroScript file and display IR structure
* Quiet by default, use `-v/--verbose` to show detailed output
* Useful for understanding the program structure

**`validate <FILE>`**

* Parse and validate program
* Checks: neuron existence, connection arity, cycles, shape compatibility
* Use `-v/--verbose` to see all details

**`compile <FILE>`**

* Full compilation pipeline: parse → validate → optimize → codegen
* Auto-detects neuron name from filename (e.g., `residual.ns` → `Residual`)
* Requires explicit `--neuron` if filename-based detection fails
* Runs optimizations by default:
  * Pattern reordering (reorder match arms for efficiency)
  * Dead branch elimination (remove unreachable match arms)
* Options:
  * `-n, --neuron <NAME>`: Specify which neuron to compile
  * `-o, --output <FILE>`: Write to file instead of stdout
  * `--no-optimize`: Disable all optimizations
  * `--no-dead-elim`: Disable dead branch elimination only
  * `-v, --verbose`: Show optimization stats and detailed output

**`list <FILE>`**

* List all neurons in a file with signatures
* Shows: name, kind (primitive/composite), inputs, outputs
* Use `-v/--verbose` to see connection details

**`init`** / **`add <DEP>`** / **`fetch`**

* Package management commands for Axon.toml-based dependency management

## Architecture

```
Source (.ns) → Pest Grammar → AST Builder → IR → Validator → Optimizer → Codegen → PyTorch
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
├── ir.rs               # Legacy IR types (mostly moved to interfaces.rs)
├── grammar/
│   ├── mod.rs         # Pest parser entry point (NeuroScriptParser)
│   ├── neuroscript.pest # PEG grammar definition
│   ├── ast.rs         # AST builder (pest pairs → IR types)
│   ├── ast/           # AST helper modules (tests)
│   ├── error.rs       # Parse error conversion to miette diagnostics
│   └── tests.rs       # Grammar tests
├── validator/
│   ├── mod.rs         # Re-exports
│   ├── core.rs        # Core validation logic
│   ├── bindings.rs    # Context binding validation (recursion, annotations)
│   ├── cycles.rs      # Cycle detection in dependency graph
│   ├── shapes.rs      # Shape compatibility validation
│   ├── symbol_table.rs # Symbol resolution
│   └── tests/         # Validator test suite (7 modules)
├── shape/
│   ├── mod.rs         # Re-exports
│   ├── algebra.rs     # Shape pattern matching and operations
│   ├── inference.rs   # Shape inference engine (variadic support)
│   └── tests.rs       # Shape system tests
├── codegen/
│   ├── mod.rs         # Re-exports
│   ├── generator.rs   # Main CodeGenerator struct
│   ├── instantiation.rs # Module instantiation (__init__)
│   ├── forward.rs     # Forward pass generation
│   ├── utils.rs       # Helper functions
│   └── tests.rs       # Codegen tests
├── optimizer/
│   ├── mod.rs         # Re-exports
│   └── core.rs        # Match arm optimization (reordering, dead branch elim)
├── package/           # Package management (Axon.toml, registry, resolver)
│   ├── mod.rs
│   ├── init.rs        # Package initialization
│   ├── manifest.rs    # Axon.toml parsing
│   ├── lockfile.rs    # Axon.lock generation
│   ├── registry.rs    # Package registry
│   └── resolver.rs    # Dependency resolution
├── stdlib_registry.rs # Primitive implementation registry
├── stdlib.rs          # Standard library loading
├── doc_parser.rs      # Documentation extraction from .ns files
├── bin/
│   └── neuroscript-doc.rs # Documentation CLI tool
└── wasm.rs            # WebAssembly target support (conditional)
```

### 1. Grammar & Parser (`src/grammar/`)

* **PEG grammar** using the `pest` crate (`neuroscript.pest`)
* Grammar handles tokenization and parsing in one pass (no separate lexer)
* `AstBuilder` in `ast.rs` converts pest parse pairs into IR types
* Indentation handling done in AST builder, not grammar
* Error conversion via `error.rs` produces miette-compatible diagnostics
* Entry point: `NeuroScriptParser::parse_program(source) -> Result<Program, ParseError>`

### 2. IR (`src/interfaces.rs`)

* **Algebraic data types** defining the full AST
* Key types:
  * `Program`: Top-level container with `uses` and `neurons` HashMap
  * `NeuronDef`: A neuron definition with params, inputs, outputs, and body
  * `NeuronBody`: Either `Primitive(ImplRef)` or `Graph(Vec<Connection>)`
  * `Connection`: Links `Endpoint` → `Endpoint` in a dataflow graph
  * `Endpoint`: Can be `Ref`, `Tuple`, `Call`, or `Match`
  * `Shape`: Tensor shapes like `[*, dim]` with dimension expressions
  * `PortRef`: References to ports (e.g., `in`, `out`, `fork.left`)
  * `InferenceContext`: Tracks resolved dimensions and node outputs during shape inference

### 3. Validator (`src/validator/`)

* **Post-parse validation** of the IR graph
* Modular design: `core.rs`, `bindings.rs`, `cycles.rs`, `shapes.rs`, `symbol_table.rs`
* Checks:
  1. All referenced neurons exist (symbol resolution)
  2. Connection endpoints match (tuple arity, port names)
  3. No cycles in dependency graph
  4. Shape compatibility via shape inference engine
  5. Context binding validity (annotation correctness, recursion safety)
* Returns `Result<(), Vec<ValidationError>>` - collects ALL errors rather than failing fast
* Integrates shape inference for dimension variable resolution

### 4. Shape System (`src/shape/`)

* **Tensor shape operations** using BigUint arithmetic to avoid overflow
* **`algebra.rs`**: Pattern matching with wildcards and literals
  * `Pattern::matches()`: Match shapes like `[*, 1, *]` against concrete shapes
  * `broadcastable()`: Check if two shapes can broadcast together
  * `refine_axis()` / `coarsen_axis()`: Split/merge dimensions
  * Axiswise operations: `axiswise_le()`, `axiswise_divides()`, `axiswise_gcd()`, `axiswise_lcm()`
* **`inference.rs`**: Shape inference engine
  * Resolves dimension variables (e.g., `dim`, `batch`) across connections
  * Tracks equivalences and constraints
  * Validates shape compatibility throughout the graph
  * Supports variadic shape patterns

### 5. Optimizer (`src/optimizer/`)

* Match arm optimization passes
* Pattern reordering for efficiency
* Dead branch elimination for unreachable match arms

### 6. Stdlib Registry (`src/stdlib_registry.rs`)

* **Maps neuron names to implementation references**
* Tracks primitive neurons and their Python/PyTorch implementations
* `ImplRef` enum with two variants:
  * `External`: External implementations with kwargs
  * `Source`: Source-based implementations (module path + class name)
* Used by codegen to generate correct imports

### 7. Codegen (`src/codegen/`)

* Direct lowering from IR to PyTorch `nn.Module`
* **`generator.rs`**: Main CodeGenerator with state tracking
* **`instantiation.rs`**: Generates `__init__` with module instantiation
* **`forward.rs`**: Generates `forward()` with connection graph execution
* **`utils.rs`**: Helper functions for code generation
* Handles:
  * Import generation from stdlib_registry
  * Match expression codegen with dimension binding
  * Parameter passing and shape comments

### 8. Package Management (`src/package/`)

* Cargo-inspired package management with `Axon.toml` manifests
* Supports git, path, and registry dependencies
* Lockfile generation (`Axon.lock`) for reproducible builds
* CLI commands: `init`, `add`, `fetch`

## Key Language Concepts

### Neurons

Two types:

* **Primitive**: Has `impl:` reference to external code (e.g., `impl: core,nn/Linear`)
* **Composite**: Has `graph:` section with internal connections

### Ports

* Default port: `in` and `out` (named "default" internally)
* Named ports: `in left: [*shape]`, `out a: [*shape]`
* Variadic input port: `in *inputs: [*shape]` — accepts any number of inputs as a tuple
  - Only input ports can be variadic (not output)
  - A neuron with a variadic port must have exactly one `in` declaration
  - Must have an explicit name (e.g., `*inputs`, not unnamed)
  - Each tuple element is validated against the port's shape individually
  - In composite neurons, `in` carries the full tuple and passes it as-is
  - Used by Concat and similar N-ary neurons: `(a, b, c) -> Concat(1)`
* Port references: `in`, `out`, `fork.left`, `fork.a`

### Connections and Pipelines

* Simple: `in -> Linear(512, 256) -> out`
* Multi-line: Indentation creates pipeline continuation
* Implicit fork (preferred): `in -> (main, skip)` — single output auto-replicates to N tuple bindings
* Explicit Fork (for named ports only): `in -> Fork() -> f` then `f.left`, `f.right`
* Port access: `main -> MLP(dim) -> processed`

### Shapes

* Literal dimensions: `[512, 256]`
* Named dimensions: `[batch, seq, dim]`
* Wildcards: `[*, dim]` (single dimension), `[*shape]` (variadic)
* Expressions: `[dim * 4]`, `[seq - 1]`

## Critical Implementation Details

### PEG Grammar Notes

* The grammar is defined in `src/grammar/neuroscript.pest` using pest syntax
* `AstBuilder` in `src/grammar/ast.rs` walks pest `Pairs` to construct IR types
* Keywords are defined as pest rules with `!ident_cont` negative lookahead to prevent partial matches
* Indentation significance is handled during AST building, not in the grammar itself

### Tuple Unpacking & Implicit Fork

```rust
// Implicit fork (v0.3.0+) — preferred for splitting tensors
in -> (a, b, c)        // Single output auto-replicates to all bindings
in -> (main, skip)     // Any number of outputs supported

// Explicit Fork — only when you need named port access
in -> Fork() -> f
f.left -> ...
f.right -> ...

// NOT for inline calls
Linear(dim, dim * 4)  // Call with args, not tuple
```

### Match Expressions

* Pattern match on tensor shapes with dimension capture
* **Basic syntax**: `match: [pattern]: pipeline`
* **With guards**: `match: [*, d] where d > 512: Linear(d, 512) -> out`
* **Dimension binding**: Captured dimensions (e.g., `d`, `seq`) can be:
  * Used in guard conditions (`where d > 512`)
  * Passed as arguments to neuron calls (`Linear(d, 512)`)
  * Referenced in pipeline expressions
* Compiler generates lazy instantiation for modules with captured dimensions

### Context Bindings

* Define reusable neuron instantiations within a neuron definition
* Syntax: `context:` block with optional annotations (`@lazy`, `@static`, `@global`)
* Enables recursion via `@lazy` binding to self with modified parameters
* Example:

  ```neuroscript
  neuron MyNeuron(d_model, num_heads, d_ff, depth):
    in: [*, seq, d_model]
    out: [*, seq, d_model]
    context:
      @lazy recurse = MyNeuron(d_model, num_heads, d_ff, depth - 1)
    graph:
      in -> match:
        [*, seq, d_model] where depth > 0: recurse
        [*, seq, d_model]: Identity() -> out
  ```

### Error Handling Philosophy

* Use `miette::Diagnostic` for structured errors with source spans
* Prefer `thiserror::Error` for error types
* Always include context: what failed, where (span), and why
* The IR types use `Display` traits for debugging - shapes print as `[*, dim]`

## Testing Strategy

### Example Files (`examples/`)

Comprehensive test suite with 126+ `.ns` files covering language features:

* `01-comments.ns` through `17-match-dimension-binding.ns`: Core language features
* Many additional test cases for edge cases, patterns, and advanced features
* Real-world examples: `residual.ns`, `transformer_from_stdlib.ns`, etc.
* Codegen test cases: `codegen_demo.ns`

### Standard Library (`stdlib/`)

6 library files with composable neurons:

* `FFN.ns`: Feed-forward networks (3 variants)
* `Residual.ns`: Skip connections (5 variants)
* `MultiHeadAttention.ns`: Attention mechanisms (5 variants)
* `TransformerBlock.ns`: Complete transformer layers (5 variants)
* `TransformerStack.ns`: Stacked transformers (6 variants)
* `MetaNeurons.ns`: Routing and composition (16 neurons)

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

* `src/grammar/`: Grammar parsing, AST building
* `src/validator/tests/`: Existence, arity, cycles, shapes, bindings, match expressions
* `src/shape/`: Pattern matching, broadcasting, inference
* `src/stdlib_registry.rs`: Primitive registry lookups
* `src/codegen/`: Code generation tests
* `src/optimizer/`: Optimization pass tests

### Integration Test Script

`./test_examples.sh` parses all `.ns` files in `examples/` and `stdlib/` directories

### Snapshot Testing (`tests/integration_tests.rs`)

Comprehensive snapshot testing using the `insta` crate for regression detection:

**What we snapshot:**

* **Parser IR**: Complete AST structures for all example files (46+ snapshots)
* **Codegen output**: Generated PyTorch code for representative neurons
* **Error messages**: Formatted diagnostics for validation and parse errors

**Key features:**

* Custom IR formatting for readable snapshots (omits spans, focuses on semantics)
* All snapshots stored in `tests/snapshots/` directory
* Automatic detection of unintended changes during refactoring
* Interactive review workflow with `cargo insta review`

**Workflow:**

1. Run tests: `cargo test --test integration_tests`
2. Review changes: `cargo insta review` (interactive UI with diffs)
3. Accept valid changes, reject unexpected ones
4. Commit snapshots with code changes

**When to use:**

* After changing grammar, validator, or codegen logic
* Before committing refactoring changes
* When adding new language features
* For comprehensive regression testing

**Documentation:** See `tests/README.md` for detailed guide

## Common Patterns

### Adding New Grammar Rules

1. Add rule to `src/grammar/neuroscript.pest`
2. Handle the new pest pair in `src/grammar/ast.rs` (AstBuilder)
3. Add corresponding IR type in `src/interfaces.rs` if needed
4. Add tests in `src/grammar/tests.rs`

### Adding New IR Nodes

1. Add enum variant to appropriate IR type in `src/interfaces.rs`
2. Add AST builder logic in `src/grammar/ast.rs`
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
   * `src/codegen/forward.rs` for forward pass generation
   * `src/codegen/instantiation.rs` for module instantiation
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

* Primitive neuron implementations (`neuroscript_runtime.primitives.*`)
* Core utilities for shape handling
* Generated code imports from this package

Generated PyTorch modules are standalone after runtime is installed.

## Roadmap

### Phase 1: Core Language ✅ Complete

* ✅ PEG grammar with pest (replaced hand-written lexer/parser)
* ✅ IR with algebraic types (modularized to `interfaces.rs`)
* ✅ Validator (existence, arity, cycles)
* ✅ Shape algebra with pattern matching
* ✅ Python runtime package
* ✅ Standard library registry
* ✅ Comprehensive test suite (126+ examples, 6 stdlib files)

### Phase 2: Codegen ✅ Complete

* ✅ IR → PyTorch `nn.Module` generation
* ✅ Import generation from stdlib_registry
* ✅ Forward pass generation (connection graph traversal)
* ✅ Parameter initialization (`__init__`)
* ✅ Match expression codegen with dimension binding
* ✅ Context bindings for recursion
* ✅ Shape inference integration in validation
* ✅ Optimizer passes (pattern reordering, dead branch elimination)
* ✅ Package management (Axon.toml, dependency resolution)

### Phase 3: Advanced Features (In Progress)

* ⏳ Full dimension variable type inference across programs
* ⏳ Loop constructs for repeated layers
* ⏳ Higher-order neurons (neuron parameters)
* ⏳ Graph simplification and fusion optimizations
* ⏳ Multiple backends (ONNX, JAX, TorchScript)

### Phase 4: Tooling (Future)

* LSP server for editor support
* PyO3 bindings for Python integration
* Visualization of neuron graphs (GraphViz, D3.js)
* REPL for interactive development
* Documentation generator from neuron definitions

## Key Dependencies

### Rust Crates (from Cargo.toml)

* `pest` (2.7): PEG parser generator
* `pest_derive` (2.7): Derive macro for pest grammar compilation
* `thiserror` (1.0): Clean error type definitions with derive macros
* `miette` (7.x): Beautiful diagnostic error reporting with source spans and fancy formatting
* `num-bigint` (0.4): Arbitrary precision integers for shape algebra (prevents overflow)
* `num-integer` (0.1): Integer traits for gcd, lcm operations
* `num-traits` (0.2): Numeric traits (Zero, One) for generic arithmetic
* `pretty_assertions` (1.4): Enhanced test output with colored diffs (dev-only)
* `insta` (1.34): Snapshot testing for comprehensive regression detection (dev-only, with yaml feature)

### Python Runtime (separate package)

* Located in project root with `setup.py` / `pyproject.toml`
* Install with `pip install -e .`
* Provides `neuroscript_runtime.primitives.*` modules for generated code

## Development Notes

* **Fast iteration**: `cargo check` is faster than `cargo build` for syntax checking
* **Error quality matters**: This is a language - users need clear diagnostics with miette spans
* **Algebraic types are the architecture**: The IR perfectly maps to Rust enums, making pattern matching natural
* **Indentation is structural**: Like Python, indentation defines scope in pipelines
* **Shape algebra uses BigUint**: Prevents overflow when computing tensor sizes (e.g., `[1000, 1000, 1000]` = 1 billion elements)
* **Codegen is string-based**: Phase 0 directly emits Python strings - future phases may use a PyTorch IR
* we don't use dot notation in the impl field for neurons, it should be something like `core,attention/ScaledDotProductAttention` instead `<provider>,<library>/<neuron>`
* always run `source ~/.venv_ai/bin/activate` or the python code will fail
