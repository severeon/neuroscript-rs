# OpenCode Tasks

1. Update @website/docs/intro.md with a few more cool examples, including details about the shape system
    * explain how neuroscript ensures your pipeline is compatible
    * show off lazy loading
    * show off recursion
  
2. add in a side-by-side of the neuroscript source and the output py file
    * make a wasm build, run in the browser
    * have a "random model" pane that builds a random model from stdlib components every 10 seconds in left pane
    * animate the regeneration of the random neuroscript
    * have a button to build the ns -> py
    * show real py code in right pane

3. update @website/docs/stdlib/index.md to include the new neurons

4. Create categories for the neuron doc pages, it's a bit hard to use with all of them in primitives
    * Basics
      * ReLU
      * Linear
      * Embedding
      * Dropout
      * Fork
      * Add
      * Reshape
    * Actications
      * GELU
      * ReLU
      * Tanh
      * Sigmoid
      * SiLU
      * Softmax
      * Mish
      * PReLU
      * ELU
    * Attention
      * ScaledDotProductAttention
      * MultiHeadSelfAttention
    * Convolutions
      * Conv1d
      * Conv2d
      * Conv3d
      * DepthwiseConv
      * SeparableConv
      * TransposedConv
    * Embeddings
      * Embedding
      * PositionalEncoding
      * LearnedPositionalEmbedding
      * RotaryEmbedding
    * Normalization
      * LayerNorm
      * RMSNorm
      * GroupNorm
      * InstanceNorm
    * Operations
      * Bias
      * Scale
      * MatMul
      * Einsum
      * Identity
    * Pooling
      * MaxPool
      * AvgPool
      * AdaptiveAvgPool
      * AdaptiveMaxPool
      * GlobalAvgPool
      * GlobalMaxPool
    * Regularization
      * Dropout
      * DropPath
      * DropConnect
    * Structural
      * Fork
      * Fork3
      * Add
      * Multiply
      * Concat
      * Reshape
      * Transpose
      * Split
      * Slice
      * Pad

5. Add a universal search, similar to spotlight on a mac
    * Full-text search
