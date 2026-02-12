export const STDLIB_BUNDLE = `


neuron FFN(dim, expansion):
    in: [*shape, dim]
    out: [*shape, dim]
    graph:
        in ->
            Linear(dim, expansion)
            GELU()
            Linear(expansion, dim)
            out

neuron FFNWithHidden(in_dim, hidden_dim, out_dim):
    in: [*shape, in_dim]
    out: [*shape, out_dim]
    graph:
        in ->
            Linear(in_dim, hidden_dim)
            GELU()
            Linear(hidden_dim, out_dim)
            out

neuron ParallelFFN(dim):
    in: [*, dim]
    out: [*, dim]
    graph:
        in -> FFN(dim, dim * 2) -> out

neuron AdaptiveAvgPool(output_size):
    in: [batch, channels, *, *]
    out: [batch, channels, output_size, output_size]
    impl: core,pooling/AdaptiveAvgPool

neuron AdaptiveMaxPool(output_size):
    in: [batch, channels, *, *]
    out: [batch, channels, output_size, output_size]
    impl: core,pooling/AdaptiveMaxPool

neuron Add:
    in main: [*shape]
    in skip: [*shape]
    out: [*shape]
    impl: core,structural/Add

neuron AvgPool(kernel_size, stride=1, padding=0):
    in: [batch, channels, height, width]
    out: [batch, channels, *, *]
    impl: core,pooling/AvgPool

neuron BatchNorm(num_features):
    in: [*shape, num_features]
    out: [*shape, num_features]
    impl: core,normalization/BatchNorm

neuron Bias(dim):
    in: [*, dim]
    out: [*, dim]
    impl: core,operations/Bias

neuron Concat(dim):
    in *inputs: [*shape]
    out: [*shape_out]
    impl: core,structural/Concat

neuron Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=true):
    in: [batch, in_channels, length]
    out: [batch, out_channels, *]
    impl: core,convolutions/Conv1d

neuron Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=true):
    in: [batch, in_channels, height, width]
    out: [batch, out_channels, *, *]
    impl: core,convolutions/Conv2d

neuron Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=true):
    in: [batch, in_channels, depth, height, width]
    out: [batch, out_channels, *, *, *]
    impl: core,convolutions/Conv3d

neuron DepthwiseConv(channels, kernel_size, stride=1, padding=0, dilation=1, bias=true):
    in: [batch, channels, height, width]
    out: [batch, channels, *, *]
    impl: core,convolutions/DepthwiseConv

neuron DropConnect(drop_prob):
    in: [*shape]
    out: [*shape]
    impl: core,regularization/DropConnect

neuron Dropout(p):
    in: [*shape]
    out: [*shape]
    impl: core,regularization/Dropout

neuron DropPath(drop_prob):
    in: [*shape]
    out: [*shape]
    impl: core,regularization/DropPath

neuron Einsum(equation):
    in a: [*shape_a]
    in b: [*shape_b]
    out: [*shape_out]
    impl: core,operations/Einsum

neuron ELU(alpha=1.0):
    in: [*shape]
    out: [*shape]
    impl: core,activations/ELU

neuron Embedding(num_embeddings, embedding_dim):
    in: [*, seq_len]
    out: [*, seq_len, embedding_dim]
    impl: core,embeddings/Embedding

neuron Flatten(start_dim=1, end_dim=-1):
    in: [*shape_in]
    out: [*shape_out]
    impl: core,structural/Flatten

neuron Fork:
    in: [*shape]
    out main: [*shape]
    out skip: [*shape]
    impl: core,structural/Fork

neuron Fork3:
    in: [*shape]
    out a: [*shape]
    out b: [*shape]
    out c: [*shape]
    impl: core,structural/Fork3

neuron GELU:
    in: [*shape]
    out: [*shape]
    impl: core,activations/GELU

neuron GlobalAvgPool:
    in: [batch, channels, *, *]
    out: [batch, channels, 1, 1]
    impl: core,pooling/GlobalAvgPool

neuron GlobalMaxPool:
    in: [batch, channels, *, *]
    out: [batch, channels, 1, 1]
    impl: core,pooling/GlobalMaxPool

neuron GroupNorm(num_groups, num_channels):
    in: [*, num_channels, *, *]
    out: [*, num_channels, *, *]
    impl: core,normalization/GroupNorm

neuron Identity:
    in: [*shape]
    out: [*shape]
    impl: core,structural/Identity

neuron InstanceNorm(num_features, eps=0.00001, affine=true):
    in: [batch, num_features, *spatial]
    out: [batch, num_features, *spatial]
    impl: core,normalization/InstanceNorm

neuron LayerNorm(dim):
    in: [*shape, dim]
    out: [*shape, dim]
    impl: core,normalization/LayerNorm

neuron LearnedPositionalEmbedding(max_positions, embedding_dim):
    in: [*, seq_len, embedding_dim]
    out: [*, seq_len, embedding_dim]
    impl: core,embeddings/LearnedPositionalEmbedding

neuron Linear(in_dim, out_dim):
    in: [*, in_dim]
    out: [*, out_dim]
    impl: core,nn/Linear

neuron MatMul:
    in a: [*, n, m]
    in b: [*, m, p]
    out: [*, n, p]
    impl: core,operations/MatMul

neuron MaxPool(kernel_size, stride=1, padding=0, dilation=1):
    in: [batch, channels, height, width]
    out: [batch, channels, *, *]
    impl: core,pooling/MaxPool

neuron Mish:
    in: [*shape]
    out: [*shape]
    impl: core,activations/Mish

neuron MultiHeadSelfAttention(dim, num_heads):
    in: [*, seq_len, dim]
    out: [*, seq_len, dim]
    impl: core,attention/MultiHeadSelfAttention

neuron Multiply:
    in a: [*shape]
    in b: [*shape]
    out: [*shape]
    impl: core,structural/Multiply

neuron Pad(padding, value=0, mode=constant):
    in: [*shape_in]
    out: [*shape_out]
    impl: core,structural/Pad

neuron PositionalEncoding(dim, max_len):
    in: [*, seq_len, dim]
    out: [*, seq_len, dim]
    impl: core,embeddings/PositionalEncoding

neuron PReLU(num_parameters=1, init=0.25):
    in: [*shape]
    out: [*shape]
    impl: core,activations/PReLU

neuron ReLU:
    in: [*shape]
    out: [*shape]
    impl: core,activations/ReLU

neuron Reshape(target_shape):
    in: [*shape_in]
    out: [*shape_out]
    impl: core,structural/Reshape

neuron RMSNorm(dim):
    in: [*, dim]
    out: [*, dim]
    impl: core,normalization/RMSNorm

neuron RotaryEmbedding(dim, max_position_embeddings=2048, base=10000):
    in query: [*batch, seq, num_heads, dim]
    in key: [*batch, seq, num_heads, dim]
    out q_out: [*batch, seq, num_heads, dim]
    out k_out: [*batch, seq, num_heads, dim]
    impl: neuroscript,embeddings/RotaryEmbedding

neuron Scale(dim):
    in: [*, dim]
    out: [*, dim]
    impl: core,operations/Scale

neuron ScaledDotProductAttention(d_k):
    in query: [*, seq_q, d_k]
    in key: [*, seq_k, d_k]
    in value: [*, seq_v, d_v]
    out: [*, seq_q, d_v]
    impl: core,attention/ScaledDotProductAttention

neuron SeparableConv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=true):
    in: [batch, in_channels, height, width]
    out: [batch, out_channels, *, *]
    impl: core,convolutions/SeparableConv

neuron Sigmoid:
    in: [*shape]
    out: [*shape]
    impl: core,activations/Sigmoid

neuron SiLU:
    in: [*shape]
    out: [*shape]
    impl: core,activations/SiLU

neuron Slice(dim, start, end):
    in: [*shape_in]
    out: [*shape_out]
    impl: core,structural/Slice

neuron Softmax(dim):
    in: [*, dim]
    out: [*, dim]
    impl: core,activations/Softmax

neuron Split(num_splits, dim=-1):
    in: [*shape]
    out a: [*shape_a]
    out b: [*shape_b]
    impl: core,structural/Split

neuron Tanh:
    in: [*shape]
    out: [*shape]
    impl: core,activations/Tanh

neuron Transpose(dims):
    in: [*shape_in]
    out: [*shape_out]
    impl: core,structural/Transpose

neuron TransposedConv(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=true):
    in: [batch, in_channels, height, width]
    out: [batch, out_channels, *, *]
    impl: core,convolutions/TransposedConv

neuron SimpleTransformerBlock(dim):
    in: [*, dim]
    out: [*, dim]
    graph:
        in ->
            LayerNorm(dim)
            Linear(dim, dim)
            Dropout(0.1)
            out

neuron TransformerBlock(dim, num_heads, d_ff):
    in: [batch, seq, dim]
    out: [batch, seq, dim]
    graph:
        in -> Fork() -> (skip1, attn_path)
        attn_path ->
            LayerNorm(dim)
            MultiHeadSelfAttention(dim, num_heads)
            Dropout(0.1)
            attn_out
        (skip1, attn_out) -> Add() -> attn_residual
        attn_residual -> Fork() -> (skip2, ffn_path)
        ffn_path ->
            LayerNorm(dim)
            FFN(dim, d_ff)
            Dropout(0.1)
            ffn_out
        (skip2, ffn_out) -> Add() -> out

neuron TransformerStack2(d_model, num_heads, d_ff):
    in: [*, d_model]
    out: [*, d_model]
    graph:
        in ->
            SimpleTransformerBlock(d_model)
            SimpleTransformerBlock(d_model)
            out

neuron SequentialTransformer(d_model, num_heads, d_ff):
    in: [*, d_model]
    out: [*, d_model]
    graph:
        in ->
            SimpleTransformerBlock(d_model)
            out
`;
