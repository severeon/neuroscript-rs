# NeuroScript

**A compositional language for neural architectures. Neurons all the way down.**

NeuroScript is a domain-specific language for defining neural network architectures through composition. Everything is a neuron—primitives, layers, attention mechanisms, entire transformers. Neurons compose into neurons with strong shape contracts and zero boilerplate.

```neuroscript
neuron TransformerBlock(d_model, n_heads, d_ff):
  in: [batch, seq, d_model]
  out: [batch, seq, d_model]
  graph:
    in -> Fork() -> (attn_path, skip1)
    attn_path ->
      LayerNorm(d_model)
      MultiHeadAttention(d_model, n_heads)
      Add(skip1)
      Fork() -> (ffn_path, skip2)
    ffn_path ->
      LayerNorm(d_model)
      FFN(d_model, d_ff)
      Add(skip2)
      out
```

Compiles to clean PyTorch:

```bash
neuroscript compile transformer.ns > transformer.py
```

## Features

- **Compositional by design**: Build transformers from attention heads, attention from projections, projections from Linear
- **Shape contracts**: Tensor shapes are part of the signature—catch dimension mismatches at compile time
- **Zero boilerplate**: No classes, no `super().__init__()`, no forward method—just describe the graph
- **Pattern matching**: Route based on tensor shapes with dimension capture
- **Multi-backend ready**: Compiles to PyTorch today, ONNX and JAX tomorrow
- **Standard library**: Batteries-included components from Linear to TransformerStack

## Installation

### Prerequisites
- Rust toolchain (1.70+)
- Python 3.8+ with PyTorch

### Build from source

```bash
# Clone the repository
git clone https://github.com/yourusername/neuroscript-rs.git
cd neuroscript-rs

# Build the compiler
cargo build --release

# Install Python runtime
pip install -e .

# Verify installation
./target/release/neuroscript --help
```

## Quick Start

### 1. Write your first neuron

Create `mlp.ns`:

```neuroscript
neuron MLP(dim):
  in: [batch, dim]
  out: [batch, dim]
  graph:
    in ->
      Linear(dim, dim * 4)
      GELU()
      Linear(dim * 4, dim)
      out
```

### 2. Compile to PyTorch

```bash
./target/release/neuroscript compile mlp.ns
```

Generates a PyTorch module:

```python
import torch.nn as nn
from neuroscript_runtime.primitives import Linear, GELU

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_1 = Linear(dim, dim * 4)
        self.gelu_1 = GELU()
        self.linear_2 = Linear(dim * 4, dim)

    def forward(self, x0):
        x1 = self.linear_1(x0)
        x2 = self.gelu_1(x1)
        x3 = self.linear_2(x2)
        return x3
```

### 3. Use in Python

```python
import torch
from mlp import MLP

model = MLP(dim=512)
batch = torch.randn(32, 512)
output = model(batch)  # [32, 512]
```

## CLI Usage

```bash
# Validate a file (parse + type check + shape check)
neuroscript validate examples/transformer.ns

# Compile to PyTorch
neuroscript compile examples/transformer.ns

# Compile specific neuron
neuroscript compile examples/transformer.ns --neuron TransformerBlock

# Write output to file
neuroscript compile examples/transformer.ns -o transformer.py

# List all neurons in a file
neuroscript list examples/transformer.ns
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

The shape system uses BigUint arithmetic internally to prevent overflow when computing tensor sizes.

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

### State and Scoping

NeuroScript uses a unified `context` block for managing weight sharing and instantiation logic through explicit annotations:

```neuroscript
@global vocab_dim = 50257

neuron Transformer(d_model):
  in: [batch, seq]
  out: [batch, seq, @global vocab_dim]

  context:
    # Instance scope (default): Unique weights per instance
    embedding = Embedding(@global vocab_dim, d_model)

    # Static scope: Shared weights across ALL instances of this neuron type
    @static shared_norm = LayerNorm(d_model)

    # Lazy scope: Instantiated only when used (useful for conditional branches)
    @lazy extra_proj = Linear(d_model, d_model)

  graph:
    in -> embedding -> shared_norm -> out
```

| Scope | Annotation | Description |
| --- | --- | --- |
| Global | `@global` | Module-level constants or neurons shared across everything |
| Static | `@static` | Shared weights across all instances of a specific neuron type |
| Instance | (default) | Unique weights per instance (standard behavior) |
| Lazy | `@lazy` | Deferred instantiation (e.g., for conditional match arms) |

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

### Primitives (Python/PyTorch)

Core building blocks backed by PyTorch implementations:

**Core**: Linear, Embedding, PositionalEncoding

**Activations**: GELU, ReLU, Tanh, Sigmoid, SiLU, Softmax

**Normalization**: LayerNorm, RMSNorm, GroupNorm

**Regularization**: Dropout, DropPath, DropConnect

### Composites (NeuroScript)

Pre-built patterns written in NeuroScript:

- `FFN`: Feed-forward networks (3 variants)
- `Residual`: Skip connections (5 variants)
- `MultiHeadAttention`: Attention mechanisms (5 variants)
- `TransformerBlock`: Complete transformer layers (5 variants)
- `TransformerStack`: Stacked transformers (6 variants)
- `MetaNeurons`: Routing and composition (16 neurons)

See `stdlib/` directory for full definitions.

## Validation

NeuroScript validates your architectures before generating code:

1. All referenced neurons exist
2. Tuple unpacking matches port counts
3. Connection endpoints have compatible shapes
4. No circular dependencies (except self-edges)

```bash
neuroscript validate examples/transformer_from_stdlib.ns
```

Shape errors are caught at compile time with clear diagnostics pointing to the source location.

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

Run the test suite:

```bash
cargo test                    # All tests
cargo test lexer              # Lexer tests only
cargo test parser             # Parser tests only
cargo test validator          # Validation tests only
./test_examples.sh            # Integration tests (all example files)
```

## Syntax Summary

| Construct | Example |
| --- | --- |
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

### In Progress

- Loop constructs for repeated layers (`repeat 12: TransformerBlock(...)`)
- Higher-order neurons (neurons that take neurons as parameters)
- Compile-time recursion unrolling
- Optimization passes (fusion, dead code elimination)

### Future

- ONNX and JAX backends
- LSP server for editor support
- PyO3 bindings for tighter Python integration
- Graph visualization
- Package manager for sharing neurons

## Design Principles

1. Composition over configuration: Build complex architectures by composing simple ones
2. Explicit shape contracts: No silent broadcasting surprises
3. External primitives: Leverage existing frameworks (PyTorch, JAX)
4. Declarative graphs: Describe what, not how
5. Types guide correctness: Catch errors at compile time

## Contributing

NeuroScript explores compositional approaches to neural architecture design. The core insight is treating neural networks as a compositional algebra where neurons are first-class values.

Contributions welcome. See CLAUDE.md for development guidelines.

## References

NeuroScript draws inspiration from:

- Tensor type systems and array programming languages (shape algebra)
- ONNX, TorchScript, and JAX (graph IR)
- Python (significant whitespace) and Rust (expression syntax)
- Category theory (compositional approach to systems)

## License

MIT
