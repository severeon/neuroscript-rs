---
sidebar_position: 2
---

# Standard Library

The NeuroScript standard library provides high-level, composable neural network components built from primitives. These neurons implement common architectural patterns and can be easily combined to create complex models.

## Categories

### Feed-Forward Networks
- FFN - Feed-forward network with configurable layers
- ParallelFFN - Parallel feed-forward processing

### Attention Mechanisms
- MultiHeadAttention - Multi-head self-attention
- ScaledDotProductAttention - Core attention computation

### Residual Connections
- Residual - Skip connection wrapper
- ResidualAdd - Addition-based residual
- ResidualConcat - Concatenation-based residual

### Transformer Components
- TransformerBlock - Complete transformer layer
- TransformerStack - Multi-layer transformer
- SequentialTransformer - Sequentially stacked transformer blocks

### Meta Neurons
- Fork - Split data into multiple paths
- Identity - Pass-through operation
- Freeze - Frozen (non-trainable) wrapper

## Using Standard Library Neurons

Import and use stdlib neurons just like primitives:

```neuroscript
use stdlib,attention/MultiHeadAttention

neuron MyTransformer(dim, heads):
    graph:
        in ->
            MultiHeadAttention(dim, heads)
            FFN(dim, dim * 4)
            out
```

## Composition Patterns

Standard library neurons are designed to compose naturally:

```neuroscript
neuron GPTBlock(dim, heads):
    in: [*batch, seq, dim]
    out: [*batch, seq, dim]
    graph:
        in ->
            Residual(
                LayerNorm(dim)
                MultiHeadAttention(dim, heads)
            )
            Residual(
                LayerNorm(dim)
                FFN(dim, dim * 4)
            )
            out
```

## Design Philosophy

- **Reusable**: Each component is self-contained and composable
- **Type-safe**: All shape contracts are validated at compile time
- **Flexible**: Parameters allow customization while maintaining correctness
- **Documented**: Every neuron includes comprehensive documentation
