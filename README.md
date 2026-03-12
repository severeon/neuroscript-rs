# NeuroScript

**A compositional language for neural architectures. Neurons all the way down.**

[![CI](https://github.com/severeon/neuroscript-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/severeon/neuroscript-rs/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![GitHub Sponsors](https://img.shields.io/github/sponsors/severeon?label=Sponsor&logo=github)](https://github.com/sponsors/severeon)

NeuroScript is a domain-specific language for defining neural network architectures through composition. Everything is a neuron — primitives, layers, attention mechanisms, entire transformers. Neurons compose into neurons with typed shape contracts and zero boilerplate.

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
neuroscript compile transformer.ns
```

## Features

- **Compositional by design** — Build transformers from attention heads, attention from projections, projections from Linear. It's neurons all the way down.
- **Shape contracts** — Tensor shapes are part of the type signature. Catch dimension mismatches at compile time, not during training.
- **Zero boilerplate** — No classes, no `super().__init__()`, no forward method. Describe the dataflow graph and the compiler handles the rest.
- **Pattern matching** — Route tensors based on shape with dimension capture and guards.
- **Unroll & repeat** — Stack layers with `unroll()` for compile-time expansion into `nn.ModuleList`.
- **100-file standard library** — From `Linear` to `TransformerStack`, batteries included.

## Quick Start

### Prerequisites

- Rust toolchain (1.70+)
- Python 3.8+ with PyTorch

### Build from source

```bash
git clone https://github.com/severeon/neuroscript-rs.git
cd neuroscript-rs
cargo build --release
pip install -e .   # Install Python runtime
```

### Write your first neuron

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

### Compile and use

```bash
neuroscript compile mlp.ns -o mlp.py
```

```python
import torch
from mlp import MLP

model = MLP(dim=512)
output = model(torch.randn(32, 512))  # [32, 512]
```

## CLI

```bash
neuroscript parse examples/residual.ns        # Parse and show IR structure
neuroscript validate examples/transformer.ns   # Type check + shape check
neuroscript compile examples/transformer.ns    # Compile to PyTorch
neuroscript compile model.ns -o model.py       # Write to file
neuroscript compile model.ns --bundle          # Bundle primitives inline (no runtime dep)
neuroscript list examples/transformer.ns       # List all neurons with signatures
```

## Language Highlights

### Everything is a Neuron

Primitives wrap external implementations. Composites define dataflow graphs. Both share the same interface.

```neuroscript
# Primitive — backed by PyTorch
neuron Linear(in_dim, out_dim):
  in: [*, in_dim]
  out: [*, out_dim]
  impl: core,nn/Linear

# Composite — internal graph
neuron Residual(dim):
  in: [*, dim]
  out: [*, dim]
  graph:
    in -> (main, skip)
    main -> MLP(dim) -> processed
    (processed, skip) -> Add() -> out
```

### Shape Algebra

Shapes are first-class with wildcards, variadics, and dimension expressions:

```neuroscript
in: [*, 512]              # Wildcard — any leading dimensions
in: [*batch, seq, dim]    # Variadic — capture zero or more dimensions
out: [batch, dim * 4]     # Expressions — computed dimensions
```

Shape errors are caught at compile time with source-located diagnostics:

```text
  × Shape mismatch
   ╭─[model.ns:12:5]
12 │     Linear(256, 512) -> LayerNorm(768)
   │                         ^^^^^^^^^^^^^ expected [*, 512], got [*, 768]
   ╰────
```

### Pattern Matching

Route tensors based on shape, with dimension capture:

```neuroscript
neuron AdaptiveProjection:
  in: [*, dim]
  out: [*, 512]
  graph:
    in -> match:
      [*, 512]: Identity() -> out
      [*, d]:   Linear(d, 512) -> out
```

### Unroll for Repeated Layers

Stack identical layers without repetition:

```neuroscript
neuron GPT2(vocab_size, d_model=768, n_heads=12, d_ff=3072, layers=12):
  in: [batch, seq]
  out: [batch, seq, vocab_size]
  context:
    blocks = unroll(layers):
      block = TransformerBlock(d_model, n_heads, d_ff)
  graph:
    in ->
      Embedding(vocab_size, d_model)
      PositionalEncoding(d_model)
      blocks
      LayerNorm(d_model)
      Linear(d_model, vocab_size)
      out
```

### Fat Arrow Reshape

Inline shape transforms with `=>`:

```neuroscript
graph:
  in => [batch, seq, heads, dim_per_head] ->
    Transpose(1, 2) ->
    ScaledDotProductAttention(d_k) ->
    out
```

### Variadic Inputs

Neurons that accept any number of inputs:

```neuroscript
neuron Concat(axis):
  in *inputs: [*shape]
  out: [*shape]
  impl: core,nn/Concat

# Use with any arity
(a, b, c) -> Concat(1) -> out
```

## Standard Library

100 files covering common architectures:

| Category | Neurons |
|----------|---------|
| **Core** | Linear, Embedding, PositionalEncoding, FFN, GatedFFN, GLU, GeGLU, SwiGLU |
| **Activations** | GELU, ReLU, Tanh, Sigmoid, SiLU, Softmax, Mish |
| **Normalization** | LayerNorm, RMSNorm, GroupNorm, BatchNorm |
| **Residual** | Residual, PreNormResidual, PostNormResidual, DenseConnection, HighwayConnection |
| **Attention** | MultiHeadAttention, MultiQueryAttention, GroupedQueryAttention, CrossAttention |
| **Transformer** | TransformerBlock, TransformerEncoderBlock, TransformerDecoderBlock, TransformerStack |
| **Vision** | PatchEmbedding, ViTBlock, InceptionBlock, SEBlock |
| **ConvNets** | ResNetBasicBlock, BottleneckBlock, ResNeXtBlock, ConvNeXtBlock, MBConvBlock |
| **Routing** | MetaNeurons (16 composition/routing patterns), SigmoidMoERouter |
| **Diffusion** | DenoisingHead, MultiTokenPredictionHead |

See [`stdlib/`](stdlib/) for full definitions.

### Full Model Examples

| Model | Description | File |
|-------|-------------|------|
| **WedLM-DS** | 28-layer hybrid: Qwen2.5-7B backbone + DeepSeek-V3 MLA, learnable residuals, sigmoid MoE | [`examples/wedlm_ds.ns`](examples/wedlm_ds.ns) |

## Testing

```bash
cargo test                           # 329 unit tests
cargo test --test integration_tests  # 228 snapshot tests
./test_examples.sh                   # Parse all .ns files
```

Snapshot testing with [insta](https://insta.rs/) catches unintended changes to parser IR, codegen output, and error messages.

## Built by AI Agents

NeuroScript is developed with the help of autonomous AI agents — each named after a fictional AI character. They work in isolated git worktrees, implement GitHub issues, and submit PRs for human review.

See the **[Agent Scoreboard](docs/AGENT-SCOREBOARD.md)** for the full roster and their contributions.

> *Sprint 3 agents: Samantha, Sonny, TARS, Dolores, Ava, Vision, Roy, Bishop, Chappie*

## Documentation

- [Language Reference](website/docs/language-reference.md) — Complete syntax and semantics guide
- [Compiler Internals](website/docs/compiler.md) — Architecture and pipeline documentation
- [Tutorials](website/docs/tutorials/) — Step-by-step guides for language features
- [Standard Library Docs](website/docs/stdlib/) — Neuron signatures and usage
- [Agent Scoreboard](docs/AGENT-SCOREBOARD.md) — AI agent roster and contributions
- [Changelog](CHANGELOG.md) — Release history

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and workflow.

Check [open issues](https://github.com/severeon/neuroscript-rs/issues) for things to work on — look for `good first issue` and `help wanted` labels.

## Support

If you find NeuroScript useful, consider [sponsoring the project](https://github.com/sponsors/severeon) to support continued development.

## License

[MIT](LICENSE)
