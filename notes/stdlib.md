## Level 1: Atomic Neurons (Primitives)

**The building Neurons of building Neurons.**

### Core Operations
- [ ] `Linear` - dense/fully-connected layer
- [ ] `Bias` - additive bias
- [ ] `Scale` - multiplicative scaling
- [ ] `MatMul` - matrix multiplication
- [ ] `Einsum` - Einstein summation (generalized tensor operations)

### Activations
- [ ] `ReLU` - rectified linear unit
- [ ] `GELU` - Gaussian error linear unit
- [ ] `SiLU` / `Swish` - sigmoid linear unit
- [ ] `Tanh` - hyperbolic tangent
- [ ] `Sigmoid` - logistic function
- [ ] `Softmax` - normalized exponential
- [ ] `Mish` - self-regularized non-monotonic activation
- [ ] `PReLU` - parametric ReLU
- [ ] `ELU` - exponential linear unit

### Normalizations
- [ ] `LayerNorm` - layer normalization
- [ ] `BatchNorm` - batch normalization
- [ ] `RMSNorm` - root mean square normalization
- [ ] `GroupNorm` - group normalization
- [ ] `InstanceNorm` - instance normalization
- [ ] `WeightNorm` - weight normalization

### Regularization
- [ ] `Dropout` - random neuron dropout
- [ ] `DropPath` - stochastic depth
- [ ] `Dropblock` - structured dropout for CNNs
- [ ] `DropConnect` - connection dropout
- [ ] `SpecAugment` - frequency/time masking (audio)

### Convolutions
- [ ] `Conv1d` - 1D convolution (sequences)
- [ ] `Conv2d` - 2D convolution (images)
- [ ] `Conv3d` - 3D convolution (video/volumetric)
- [ ] `DepthwiseConv` - channel-wise convolution
- [ ] `SeparableConv` - depthwise + pointwise
- [ ] `TransposedConv` / `Deconv` - upsampling convolution
- [ ] `DilatedConv` - atrous convolution

### Pooling
- [ ] `MaxPool` - max pooling
- [ ] `AvgPool` - average pooling
- [ ] `AdaptiveAvgPool` - output-size-adaptive pooling
- [ ] `AdaptiveMaxPool` - output-size-adaptive max pooling
- [ ] `GlobalAvgPool` - spatial averaging
- [ ] `GlobalMaxPool` - spatial max

### Embeddings
- [ ] `Embedding` - discrete token → dense vector
- [ ] `PositionalEncoding` - sinusoidal position embeddings
- [ ] `LearnedPositionalEmbedding` - trainable positions
- [ ] `RotaryEmbedding` (RoPE) - rotary position embeddings
- [ ] `ALiBi` - attention with linear biases

### Utility
- [ ] `Reshape` - tensor reshaping
- [ ] `Transpose` - dimension permutation
- [ ] `Concatenate` - tensor concatenation
- [ ] `Split` - tensor splitting
- [ ] `Slice` - tensor slicing
- [ ] `Pad` - tensor padding
- [ ] `Crop` - tensor cropping
- [ ] `Cast` - dtype conversion
- [ ] `Clone` - tensor duplication
- [ ] `Identity` - pass-through (useful for routing)

---
