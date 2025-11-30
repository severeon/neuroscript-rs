# NeuroScript

**Neural architecture composition language. Neurons all the way down.**

NeuroScript is a domain-specific language for defining composable neural network architectures. Everything is a neuron—primitives, layers, attention mechanisms, entire transformers. Neurons compose into neurons through declarative graph specifications with strong shape contracts.

## Status

**Current:** Parser + IR + Validation + Standard Library + Codegen (Phase 0)
**Next:** Shape Inference & Optimizations

## Quick Start

```bash
# Build
cargo build --release

# Parse and validate a file
./target/release/neuroscript --validate examples/residual.ns

# Install Python runtime
pip install -e .
```

## Language Overview

### Everything is a Neuron

```neuroscript
# Primitives wrap PyTorch implementations
neuron Linear(in_dim, out_dim):
  in: [*, in_dim]
  out: [*, out_dim]
  impl: neuroscript_runtime.primitives.Linear

# Composites define internal graphs
neuron MLP(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in ->
      Linear(dim, dim * 4)
      GELU()
      Linear(dim * 4, dim)
      out

# Multi-port neurons enable branching
neuron Residual(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> Fork() -> (main, skip)
    main -> MLP(dim) -> processed
    (processed, skip) -> Add() -> out
```

### Shape Algebra

Shapes are first-class with wildcards, variadics, and dimension expressions:

```neuroscript
neuron AttentionHead(d_model, d_k):
  in query: [batch, seq_q, d_model]
  in key: [batch, seq_k, d_model]
  in value: [batch, seq_k, d_model]
  out: [batch, seq_q, d_k]
  impl: ...

# Wildcards match any dimension
in: [*, 512]        # batch size is flexible
in: [*, *, dim]     # 2D batch dimensions

# Variadics capture zero or more dimensions
in: [*batch, seq, dim]   # any batch structure
in: [*shape]             # any shape at all

# Dimension expressions
in: [batch, dim]
out: [batch, dim * 4]     # expansion
out: [batch, dim / 2]     # reduction
```

**Full tensor shape operations with BigUint arithmetic to avoid overflow:**

```rust
use neuroscript::shape::*;

// Pattern matching with wildcards
let pattern = Pattern::from_tokens(vec![
    PatToken::Any,      // matches any dimension
    PatToken::Lit(1),   // must be exactly 1
    PatToken::Any,
]);
assert!(pattern.matches(&Shape::new(vec![32, 1, 256])));

// Broadcasting checks
let a = Shape::new(vec![32, 64, 56, 56]);
let b = Shape::new(vec![1, 64, 1, 1]);
assert!(broadcastable(&a, &b));

// Refine/coarsen operations
let shape = Shape::new(vec![64]);
let refined = refine_axis(&shape, 0, &[8, 8]).unwrap();
assert_eq!(refined, Shape::new(vec![8, 8]));
```

### Pipeline Syntax

Two styles for readability:

```neuroscript
# Inline (single line)
graph:
  in -> LayerNorm(dim) -> GELU() -> out

# Indented (multi-step, no intermediate arrows)
graph:
  in ->
    Linear(dim, dim * 4)
    GELU()
    Dropout(0.1)
    Linear(dim * 4, dim)
    out
```

### Tuple Unpacking

For multi-output neurons:

```neuroscript
neuron Fork:
  in: [*shape]
  out a: [*shape]
  out b: [*shape]
  impl: core,builtin/Fork

# Unpack outputs by name
graph:
  in -> Fork() -> (branch_a, branch_b)
  branch_a -> ProcessA() -> out_a
  branch_b -> ProcessB() -> out_b
```

### Pattern Matching

Route based on shape:

```neuroscript
neuron AdaptiveProjection:
  in: [*, dim]
  out: [*, 512]
  graph:
    in -> match:
      [*, 512]: Identity() -> out
      [*, 256]: Linear(256, 512) -> out
      [*, d]: Linear(d, 512) -> out
```

## Standard Library

### Level 0: Primitives (Python/PyTorch)

**Core:**

- `Linear` - Dense/fully-connected layer
- `Embedding` - Token → vector
- `PositionalEncoding` - Sinusoidal positions

**Activations:**

- `GELU`, `ReLU`, `Tanh`, `Sigmoid`, `SiLU`, `Softmax`

**Normalization:**

- `LayerNorm`, `RMSNorm`, `GroupNorm`

**Regularization:**

- `Dropout`, `DropPath`, `DropConnect`

### Level 1+: Composites (NeuroScript)

**Composition Patterns:**

- `FFN` - Feed-forward networks (3 variants)
- `Residual` - Skip connections (5 variants)
- `MultiHeadAttention` - Attention mechanisms (5 variants)
- `TransformerBlock` - Complete transformer layers (5 variants)
- `TransformerStack` - Stacked transformers (6 variants)
- `MetaNeurons` - Routing and composition (16 neurons)

See `stdlib/` directory for full definitions.

### Intermediate Representation (IR)

Algebraic types map neural architectures to Rust enums:

```rust
pub enum NeuronBody {
    Primitive(ImplRef),      // Wraps PyTorch/external
    Graph(Vec<Connection>),  // Composite definition
}

pub struct Connection {
    source: Endpoint,
    destination: Endpoint,
}

pub enum Endpoint {
    Ref(PortRef),           // Simple node reference
    Tuple(Vec<PortRef>),    // Multi-port unpacking
    Call { ... },           // Neuron instantiation
    Match(MatchExpr),       // Shape-based routing
}
```

### Validation

The validator ensures:

1. **Existence:** All referenced neurons are defined
2. **Arity:** Tuple unpacking matches port counts
3. **Shapes:** Connection endpoints are compatible
4. **Cycles:** No circular dependencies (except self-edges in single connections)

```bash
./target/release/neuroscript --validate examples/transformer_from_stdlib.ns
```

## Example: Building GPT-2 Small

```neuroscript
# Token embeddings + positional encoding
neuron TokenWithPosition(vocab_size, d_model, max_len):
    in: [batch, seq]
    out: [batch, seq, d_model]
    graph:
        in ->
            Embedding(vocab_size, d_model)
            PositionalEncoding(d_model, max_len=max_len)
            out

# GPT-2 small: 12 layers, 768 dim, 12 heads
neuron GPT2Small(vocab_size):
    in: [batch, seq]
    out: [batch, seq, vocab_size]
    graph:
        in ->
            TokenWithPosition(vocab_size, 768, 1024)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            TransformerBlock(768, 12, 3072)
            LayerNorm(768)
            Linear(768, vocab_size)
            out
```

See `examples/transformer_from_stdlib.ns` for complete working examples.

## Testing

### Unit Tests (32 tests)

```bash
cargo test                    # All tests
cargo test lexer              # Lexer tests
cargo test parser             # Parser tests
cargo test validator          # Validation tests
cargo test shape_algebra      # Shape algebra tests
cargo test stdlib_registry    # Registry tests
```

### Integration Tests (26 files)

```bash
./test_examples.sh            # Parse all examples + stdlib
```

All 26 files (20 examples + 6 stdlib) parse successfully with zero errors.

## Why Rust?

1. **Algebraic types** - IR maps perfectly to enums/structs
2. **Great errors** - `miette` provides beautiful diagnostics with source spans
3. **Fast** - Parses instantly, compiles to native ARM
4. **PyO3 ready** - Can expose to Python when needed
5. **Quality** - LLMs write excellent Rust with strong type guidance

## Syntax Summary

| Construct | Example |
|-----------|---------|
| Neuron definition | `neuron Name(p1, p2=default):` |
| Input port | `in [name]: [shape]` |
| Output port | `out [name]: [shape]` |
| Primitive impl | `impl: module.path.Class` |
| Graph body | `graph:` |
| Inline pipeline | `in -> A() -> B() -> out` |
| Indented pipeline | `in ->\n  A()\n  B()\n  out` |
| Tuple unpacking | `Fork() -> (a, b)` |
| Tuple packing | `(a, b) -> Add() -> out` |
| Function call | `Linear(512, 256)` |
| With kwargs | `Dropout(p=0.1)` |
| Shape literal | `[batch, 512]` |
| Wildcard | `[*, dim]` |
| Variadic | `[*batch, seq, dim]` |
| Dimension expr | `dim * 4`, `dim / 2` |
| Match expression | `match:\n  [*, 512]: Identity() -> out` |
| Import | `use core,nn/*` |
| Comment | `# comment` |
| String | `` `backtick string` `` |

## Roadmap

### Phase 1: Core Language ✅

- [x] Lexer with indent handling
- [x] Parser with shape expressions
- [x] IR with algebraic types
- [x] Validator (existence, arity, shapes, cycles)
- [x] Shape algebra with pattern matching
- [x] Python runtime package
- [x] Standard library registry
- [x] Comprehensive test suite

### Phase 2: Codegen (In Progress)

- [x] IR → PyTorch nn.Module
- [x] Import generation from stdlib_registry
- [ ] Shape inference integration
- [x] Forward pass generation
- [x] Parameter initialization

### Phase 3: Advanced Features

- [ ] Type inference for dimension variables
- [ ] Loop constructs for repeated layers
- [ ] Higher-order neurons (neuron parameters)
- [ ] Optimization passes
- [ ] Multiple backends (ONNX, JAX)

### Phase 4: Tooling

- [ ] LSP server for editor support
- [ ] PyO3 bindings for Python integration
- [ ] Visualization of neuron graphs
- [ ] REPL for interactive development
- [ ] Package manager for sharing neurons

## Contributing

This is a research project exploring compositional approaches to neural architecture design. The core insight: treating neural networks as a compositional algebra where neurons are first-class values enables powerful abstraction without sacrificing clarity.

### Key Design Principles

1. **Composition over configuration** - Build complex architectures by composing simple ones
2. **Shape contracts are explicit** - No silent broadcasting surprises
3. **Primitives are external** - Leverage existing frameworks (PyTorch, JAX)
4. **Graphs are declarative** - Describe *what*, not *how*
5. **Types guide correctness** - Catch errors at parse/validate time

## License

MIT

## References

- **Shape algebra:** Inspired by tensor type systems and array programming languages
- **Graph IR:** Influenced by ONNX, TorchScript, and JAX primitives
- **Syntax:** Python-like with significant whitespace, Rust-like expressions
- **Composition:** Category theory's compositional approach to systems
