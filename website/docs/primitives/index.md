---
sidebar_position: 1
---

# Primitives

Primitives are the fundamental building blocks of NeuroScript. They wrap PyTorch operations and provide the foundation for building complex neural architectures.

## Categories

### Basics

- [Linear](./Basics/linear) - Fully-connected linear transformation

### Activations

- [ReLU](./Activations/relu) - Rectified Linear Unit
- [GELU](./Activations/gelu) - Gaussian Error Linear Unit
- [SiLU](./Activations/silu) - Sigmoid Linear Unit (Swish)
- [Sigmoid](./Activations/sigmoid) - Sigmoid activation
- [Tanh](./Activations/tanh) - Hyperbolic tangent activation
- [ELU](./Activations/elu) - Exponential Linear Unit
- [Mish](./Activations/mish) - Mish activation
- [PReLU](./Activations/prelu) - Parametric ReLU with learnable slope
- [Softmax](./Activations/softmax) - Softmax normalization

### Attention

- [ScaledDotProductAttention](./Attention/scaleddotproductattention) - Core attention mechanism (Q, K, V)
- [MultiHeadSelfAttention](./Attention/multiheadselfattention) - Multi-head self-attention

### Normalization

- [LayerNorm](./Normalization/layernorm) - Layer normalization
- [RMSNorm](./Normalization/rmsnorm) - Root Mean Square normalization
- [BatchNorm](./Normalization/batchnorm) - Batch normalization
- [GroupNorm](./Normalization/groupnorm) - Group normalization
- [InstanceNorm](./Normalization/instancenorm) - Instance normalization

### Embeddings

- [Embedding](./Embeddings/embedding) - Token embedding (index → dense vector)
- [PositionalEncoding](./Embeddings/positionalencoding) - Fixed sinusoidal positional encoding
- [LearnedPositionalEmbedding](./Embeddings/learnedpositionalembedding) - Learnable positional embeddings
- [RotaryEmbedding](./Embeddings/rotaryembedding) - Rotary position embeddings (RoPE)

### Convolutions

- [Conv1d](./Convolutions/conv1d) - 1D convolution (sequences, audio)
- [Conv2d](./Convolutions/conv2d) - 2D convolution (images)
- [Conv3d](./Convolutions/conv3d) - 3D convolution (volumetric data, video)
- [DepthwiseConv](./Convolutions/depthwiseconv) - Depthwise separable convolution
- [SeparableConv](./Convolutions/separableconv) - Depthwise + pointwise convolution
- [TransposedConv](./Convolutions/transposedconv) - Transposed (deconvolutional) layer

### Pooling

- [MaxPool](./Pooling/maxpool) - 2D max pooling
- [AvgPool](./Pooling/avgpool) - 2D average pooling
- [GlobalAvgPool](./Pooling/globalavgpool) - Global average pooling
- [GlobalMaxPool](./Pooling/globalmaxpool) - Global max pooling
- [AdaptiveAvgPool](./Pooling/adaptiveavgpool) - Adaptive average pooling
- [AdaptiveMaxPool](./Pooling/adaptivemaxpool) - Adaptive max pooling

### Regularization

- [Dropout](./Regularization/dropout) - Random element dropout
- [DropConnect](./Regularization/dropconnect) - Random connection dropout
- [DropPath](./Regularization/droppath) - Stochastic depth (path dropout)

### Operations

- [MatMul](./Operations/matmul) - Batched matrix multiplication
- [Bias](./Operations/bias) - Learnable bias addition
- [Scale](./Operations/scale) - Element-wise scaling
- [Identity](./Operations/identity) - Pass-through (no-op)
- [Einsum](./Operations/einsum) - Einstein summation

### Structural

- [Fork](./Structural/fork) - Duplicate input to two outputs
- [Fork3](./Structural/fork3) - Duplicate input to three outputs
- [Concat](./Structural/concat) - Concatenate tensors along a dimension
- [Add](./Structural/add) - Element-wise addition
- [Multiply](./Structural/multiply) - Element-wise multiplication
- [Flatten](./Structural/flatten) - Flatten to 2D
- [Reshape](./Structural/reshape) - Reshape tensor dimensions
- [Transpose](./Structural/transpose) - Permute tensor dimensions
- [Slice](./Structural/slice) - Extract a portion of a tensor
- [Split](./Structural/split) - Split tensor into multiple pieces
- [Pad](./Structural/pad) - Pad tensor with zeros

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
