# NeuroScript Standard Library (Stdlib)

A collection of composable neural network neurons for building transformer and attention-based architectures.

## Contents

### Core Files

- **`primitives.ns`** - Definitions of all primitive neuron signatures (Linear, GELU, LayerNorm, Fork, Add, etc.)
- **`FFN.ns`** - Feed-forward networks (FFN, FFNWithHidden)
- **`MultiHeadAttention.ns`** - Attention primitives documentation
- **`TransformerBlock.ns`** - Simplified transformer blocks
- **`TransformerStack.ns`** - Stacked transformer layers
- **`Residual.ns`** - Residual connection documentation
- **`MetaNeurons.ns`** - Meta-neurons for composition (Identity, ParallelFFN)

## Available Neurons

### Feed-Forward Networks

- `FFN(dim, expansion)` - Standard 2-layer FFN with expansion/contraction
- `FFNWithHidden(in_dim, hidden_dim, out_dim)` - FFN with explicit dimensions

### Attention Mechanisms (Primitives)

- `MultiHeadSelfAttention(d_model, num_heads)` - Multi-head self-attention
- `ScaledDotProductAttention()` - Scaled dot-product attention

### Transformer Layers

- `SimpleTransformerBlock(dim)` - Minimal transformer layer (LayerNorm → Linear → Dropout)
- `TransformerStack2(d_model, num_heads, d_ff)` - 2-layer transformer stack
- `SequentialTransformer(d_model, num_heads, d_ff)` - Single transformer layer

### Structural Operations

- `Identity()` - Pass-through operation
- `Linear(in_dim, out_dim)` - Fully-connected layer
- `Fork()` - Split single input into two outputs (not yet fully supported with tuple unpacking)
- `Add()` - Element-wise addition (2 inputs)
- `Multiply()` - Element-wise multiplication (2 inputs)

### Activations

- `GELU()`, `ReLU()`, `Sigmoid()`, `Tanh()`, `SiLU()`, `Softmax()`

### Normalization

- `LayerNorm(dim)` - Layer normalization
- `RMSNorm(dim)` - Root mean square normalization
- `GroupNorm(num_groups, dim)` - Group normalization

### Regularization

- `Dropout(p)` - Dropout regularization
- `DropPath(p)` - Stochastic depth
- `DropConnect(p)` - Connection dropout

### Embeddings

- `Embedding(vocab_size, embedding_dim)` - Token embedding
- `PositionalEncoding(dim)` - Sinusoidal positional encoding

## Known Limitations

### Feature Limitations

1. **Fork() with Tuple Unpacking** - The Fork primitive is defined but tuple unpacking (e.g., `in -> Fork() -> (a, b)`) requires further validator enhancements. Neurons using this pattern are currently simplified or commented out.

2. **Neuron Parameters** - Passing neurons as parameters (e.g., `Residual(f)` where `f` is a neuron) is not yet implemented. Complex patterns that require this are simplified.

3. **Recursive Structures** - Recursive neuron definitions using `let:` bindings are not yet fully implemented. Complex stacks are provided as explicit unrolled variants.

4. **Conditional Routing** - `match:` expressions with complex routing are not fully supported for all use cases.

### Shape Constraints

All neurons in the stdlib follow these shape constraints:

- **Batch dimension**: Always variadic (`*shape` or `[*batch, ...]`)
- **Output shapes**: Derived from input shapes and operation parameters
- **Named dimensions**: Can use dimension expressions like `d_model`, `seq`, `batch`

### Primitive Port Signatures

All primitives have been defined with explicit port signatures in `primitives.ns`. When using stdlib neurons:

- Linear transformations expect correctly-sized inputs
- Element-wise operations (Add, Multiply) require same-shape inputs
- Activation functions preserve input shape
- Normalization functions require explicit dimension parameters

## Usage Examples

### Simple FFN Pipeline

```neuroscript
neuron MyFFN:
    in: [*, 512]
    out: [*, 512]
    graph:
        in ->
            FFN(512, 2048)
            out
```

### Transformer Stack

```neuroscript
neuron MyTransformer:
    in: [*, 768]
    out: [*, 768]
    graph:
        in ->
            PositionalEncoding(768)
            TransformerStack2(768, 12, 3072)
            LayerNorm(768)
            out
```

### Complex Pipeline

```neuroscript
neuron ComplexModel:
    in: [*, 512]
    out: [*, 512]
    graph:
        in ->
            FFN(512, 2048)
            LayerNorm(512)
            MultiHeadSelfAttention(512, 8)
            Dropout(0.1)
            out
```

## Testing

Run the stdlib test suite:

```bash
./target/release/neuroscript validate examples/test_stdlib.ns
```

This validates all stdlib neurons and their interactions.

## Future Enhancements

### Priority 1: Core Features

- [ ] Complete Fork() tuple unpacking support
- [ ] Implement neuron parameters (Residual(f), etc.)
- [ ] Fix Fork()/Add() shape inference for complex pipelines
- [ ] Remove cycle detection issues in transformer variants

### Priority 2: Advanced Neurons

- [ ] Full transformer encoder/decoder with cross-attention
- [ ] GQA (Grouped Query Attention) variants
- [ ] Gated FFN variants (SwiGLU, GeGLU, etc.)
- [ ] Multi-path networks with gating

### Priority 3: Optimization

- [ ] Recursive TransformerStack using `let:` bindings
- [ ] Dynamic stacking patterns
- [ ] Conditional routing for adaptive computation

## Requirements

- NeuroScript compiler with stdlib support
- Python runtime with PyTorch for execution
- All generated neurons work with PyTorch nn.Module interface

## Notes

- Primitive neurons (Linear, GELU, etc.) map to `neuroscript_runtime.primitives.*`
- Composite neurons are fully defined in .ns files
- All shapes support wildcard dimension matching and named dimension variables
- Shape inference validates connections during compilation
