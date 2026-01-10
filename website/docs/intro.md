---
sidebar_position: 1
---

# Introduction to NeuroScript

NeuroScript is a neural architecture composition language that treats neurons as first-class composable units. It compiles to PyTorch (with future support for ONNX and JAX), enabling declarative definition of neural networks with strong shape guarantees.

## Core Philosophy

**Neurons all the way down** - Everything in NeuroScript is a neuron, and neurons compose into neurons. This uniform abstraction makes it easy to build complex architectures from simple, reusable components.

## Quick Example

```neuroscript
neuron FFN(dim, expansion):
    in: [*batch, seq, dim]
    out: [*batch, seq, dim]
    graph:
        in ->
            Linear(dim, dim * expansion)
            GELU()
            Linear(dim * expansion, dim)
            out
```

## Key Features

- **Shape-aware**: Tensor shapes are first-class citizens with compile-time validation
- **Composable**: Build complex networks from simple, reusable neurons
- **Type-safe**: Shape inference catches dimensional errors before runtime
- **Declarative**: Focus on *what* you want, not *how* to implement it

## Advanced Capabilities

### Compatibility & Shape Inference
NeuroScript ensures pipeline compatibility through its rigorous shape algebra system. Every connection is validated at compile time:

```neuroscript
# This will fail to compile if dimensions don't match
neuron Incompatible:
    in: [batch, 128]
    graph:
        in -> Linear(128, 64) -> Linear(32, 10) -> out
        # Error: Shape mismatch! Expected [*, 32], got [*, 64]
```

### Lazy Loading
Use the `@lazy` annotation to define components that are only instantiated when used. This is powerful for conditional architectures or dynamic routing:

```neuroscript
neuron DynamicBranch(dim):
    context:
        @lazy heavy_branch = StackedTransformer(dim, 12)
        light_branch = Linear(dim, dim)
    
    graph:
        in -> match:
            [*, 1, dim]: heavy_branch -> out
            [*, _, dim]: light_branch -> out
```

### Recursion
NeuroScript supports recursive neuron definitions, enabling fractal architectures and repeated structures:

```neuroscript
neuron FractalNet(dim, depth):
    in: [*, dim]
    out: [*, dim]
    
    graph:
        in -> match:
            # Base case
            if depth == 0:
                Linear(dim, dim) -> out
            
            # Recursive step
            else:
                Fork() -> (left, right)
                left -> FractalNet(dim, depth - 1) -> l_out
                right -> FractalNet(dim, depth - 1) -> r_out
                (l_out, r_out) -> Add() -> out
```

## Getting Started

1. **[Primitives](/docs/primitives)** - Low-level building blocks wrapping PyTorch operations
2. **[Standard Library](/docs/stdlib)** - High-level composable architectures

## Installation

```bash
# Install the NeuroScript compiler
cargo install neuroscript

# Install the Python runtime
pip install neuroscript-runtime
```

## Example Usage

```bash
# Compile a NeuroScript file to PyTorch
neuroscript compile my_model.ns -o model.py

# Validate shapes without generating code
neuroscript validate my_model.ns
```

## Learn More

- [Language Specification](https://github.com/neuroscript/neuroscript/blob/main/docs/language-spec.md)
- [Examples](https://github.com/neuroscript/neuroscript/tree/main/examples)
- [GitHub Repository](https://github.com/neuroscript/neuroscript)
