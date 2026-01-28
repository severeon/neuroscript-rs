# NeuroScript Examples

This directory contains comprehensive examples organized to mirror the [NeuroScript documentation site](../website/docs/) structure.

## Directory Structure

```
examples/
├── tutorials/          # Core language feature tutorials
├── primitives/         # All primitive neuron demonstrations
├── stdlib/             # Standard library usage patterns
├── real_world/         # Complete architecture examples
└── __scratch/          # Archive of old examples
```

## Quick Navigation

### Tutorials (Learn the Language)
Maps to: `website/docs/tutorials/`

| File | Demonstrates | Doc Link |
|------|--------------|----------|
| `01_shape_inference.ns` | Dimension variables, automatic shape inference | [Shape Inference](../website/docs/tutorials/shape-inference.mdx) |
| `02_fork_join.ns` | Fork primitive, tuple unpacking, residual patterns | [Fork & Join](../website/docs/tutorials/fork-join.mdx) |
| `03_match_guards.ns` | Match expressions, shape patterns, guards | [Match & Guards](../website/docs/tutorials/match-guards.mdx) |

### Primitives (Built-in Neurons)
Maps to: `website/docs/primitives/`

Each file demonstrates **all** primitives in that category with practical examples.

| File | Category | Primitives Demonstrated |
|------|----------|------------------------|
| `activations.ns` | Activations | ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, ELU, PReLU, Mish |
| `attention.ns` | Attention | ScaledDotProductAttention, MultiHeadSelfAttention |
| `basics.ns` | Basics | Linear |
| `convolutions.ns` | Convolutions | Conv1d, Conv2d, Conv3d, Depthwise, Separable, Transposed |
| `embeddings.ns` | Embeddings | Embedding, PositionalEncoding, LearnedPositionalEmbedding, RotaryEmbedding |
| `normalization.ns` | Normalization | LayerNorm, BatchNorm, GroupNorm, InstanceNorm, RMSNorm |
| `operations.ns` | Operations | Add, Multiply, MatMul, Einsum, Bias, Scale |
| `pooling.ns` | Pooling | MaxPool, AvgPool, GlobalMaxPool, GlobalAvgPool, AdaptiveMaxPool, AdaptiveAvgPool |
| `regularization.ns` | Regularization | Dropout, DropConnect, DropPath |
| `structural.ns` | Structural | Fork, Fork3, Split, Concat, Reshape, Flatten, Transpose, Slice, Pad, Identity |

### Standard Library (Reusable Components)
Maps to: `website/docs/stdlib/`

Usage patterns for composable neuron libraries.

| File | Components | Description |
|------|-----------|-------------|
| `feedforward.ns` | FFN, FFNWithHidden | Feed-forward network patterns |
| `attention.ns` | MultiHeadAttention | Attention mechanism usage |
| `residual.ns` | Residual, ResidualAdd | Skip connection patterns |
| `transformer_block.ns` | TransformerBlock | Complete transformer layers |
| `transformer_stack.ns` | TransformerStack | Multi-layer transformers |

### Real World (Complete Architectures)

Production-ready neural network architectures.

| File | Architecture | Description |
|------|--------------|-------------|
| `gpt2_small.ns` | GPT-2 | Decoder-only transformer (124M params) |
| `resnet.ns` | ResNet | Residual convolutional network |
| `vision_transformer.ns` | ViT | Vision Transformer with patch embeddings |
| `llama_rope.ns` | LLaMA | Decoder with RoPE and RMSNorm |

## How to Use

### Compile an Example
```bash
# Build the compiler
cargo build --release

# Validate an example
./target/release/neuroscript validate examples/tutorials/01_shape_inference.ns

# Compile to PyTorch
./target/release/neuroscript compile examples/real_world/gpt2_small.ns -o gpt2.py
```

### Learn the Language
1. Start with **tutorials/** to understand core concepts
2. Explore **primitives/** to see all available building blocks
3. Study **stdlib/** for reusable composition patterns
4. Examine **real_world/** for complete architecture examples

### Find a Specific Primitive
Each primitive category file (`primitives/*.ns`) contains:
- Individual primitive declarations
- Usage examples for each primitive
- Composition examples showing primitives working together

Use the table above to find which file contains the primitive you need.

## Archive (__scratch/)

Old examples and experimental files are preserved in `__scratch/`:
- `old_tutorials/`: Original 01-17.ns tutorial files
- `experiments/`: Advanced features and test files
- `configs/`: Training configurations and scripts

These are kept for reference but are not maintained.

## Documentation Links

- [NeuroScript Documentation](../website/docs/)
- [Primitives Reference](../website/docs/primitives/)
- [Standard Library](../website/docs/stdlib/)
- [Tutorials](../website/docs/tutorials/)

## Contributing Examples

When adding new examples:
1. Choose the appropriate category (tutorials/primitives/stdlib/real_world)
2. Follow the file structure pattern (see existing files)
3. Include comments explaining the purpose and doc references
4. Ensure the example compiles: `neuroscript validate <file>`
5. Update this README if adding new categories

## Self-Consistency Principle

This examples directory maintains **1:1 mapping** with documentation:
- Every doc category has a corresponding example file
- Every primitive mentioned in docs is demonstrated
- File organization mirrors doc site structure

If you update documentation, update the corresponding example file.
If you add a new primitive, add it to the appropriate category file and documentation.
