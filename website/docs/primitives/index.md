---
sidebar_position: 1
---

# Primitives

Primitives are the fundamental building blocks of NeuroScript. They wrap PyTorch operations and provide the foundation for building complex neural architectures.

## Categories

### Layers
- [Linear](./linear) - Fully-connected transformation
- [Conv2d](./conv2d) - 2D convolutional layer
- [Embedding](./embedding) - Token embedding

### Activations
- [ReLU](./relu) - Rectified Linear Unit
- [GELU](./gelu) - Gaussian Error Linear Unit

### Normalization
- [LayerNorm](./layernorm) - Layer normalization
- [BatchNorm](./batchnorm) - Batch normalization

### Regularization
- [Dropout](./dropout) - Dropout regularization

### Structural
- [Flatten](./flatten) - Dimension flattening
- [Concat](./concat) - Tensor concatenation

## Usage

All primitives are automatically available in your NeuroScript programs. Simply use them by name:

```neuroscript
neuron MyModel:
    graph:
        in -> Linear(512, 256) -> GELU() -> out
```

## Shape Contracts

Every primitive has a well-defined shape contract that specifies:
- Input shapes (what tensor dimensions are expected)
- Output shapes (what dimensions are produced)
- Parameter constraints (what values are valid)

The NeuroScript compiler validates all shape contracts at compile time, catching dimensional errors before code generation.
