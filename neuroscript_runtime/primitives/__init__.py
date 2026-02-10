"""
Primitive neurons - Level 0 building blocks.
"""

from neuroscript_runtime.primitives.linear import Linear

from neuroscript_runtime.primitives.activations import (
    GELU,
    ReLU,
    Tanh,
    Sigmoid,
    SiLU,
    Softmax,
    Mish,
    PReLU,
    ELU,
)

from neuroscript_runtime.primitives.normalization import (
    LayerNorm,
    RMSNorm,
    GroupNorm,
    InstanceNorm,
)

from neuroscript_runtime.primitives.regularization import (
    Dropout,
    DropPath,
    DropConnect,
)

from neuroscript_runtime.primitives.embeddings import (
    Embedding,
    PositionalEncoding,
    LearnedPositionalEmbedding,
    RotaryEmbedding,
)

from neuroscript_runtime.primitives.structural import (
    Fork,
    Fork3,
    Add,
    Multiply,
    Subtract,
    Divide,
    Concat,
    Reshape,
    Transpose,
    Split,
    Slice,
    Pad,
)

from neuroscript_runtime.primitives.attention import (
    ScaledDotProductAttention,
    MultiHeadSelfAttention,
)

from neuroscript_runtime.primitives.operations import (
    Bias,
    Scale,
    MatMul,
    Einsum,
    Identity,
)

from neuroscript_runtime.primitives.convolutions import (
    Conv1d,
    Conv2d,
    Conv3d,
    DepthwiseConv,
    SeparableConv,
    TransposedConv,
)

from neuroscript_runtime.primitives.pooling import (
    MaxPool,
    AvgPool,
    AdaptiveAvgPool,
    AdaptiveMaxPool,
    GlobalAvgPool,
    GlobalMaxPool,
)
# Debug/Logging
from neuroscript_runtime.primitives.logging import Log

__all__ = [
    "Linear",
    "GELU",
    "ReLU",
    "Tanh",
    "Sigmoid",
    "SiLU",
    "Softmax",
    "Mish",
    "PReLU",
    "ELU",
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "InstanceNorm",
    "Dropout",
    "DropPath",
    "DropConnect",
    "Embedding",
    "PositionalEncoding",
    "LearnedPositionalEmbedding",
    "RotaryEmbedding",
    "Fork",
    "Fork3",
    "Add",
    "Multiply",
    "Subtract",
    "Divide",
    "Concat",
    "Reshape",
    "Transpose",
    "Split",
    "Slice",
    "Pad",
    "ScaledDotProductAttention",
    "MultiHeadSelfAttention",
    "Bias",
    "Scale",
    "MatMul",
    "Einsum",
    "Identity",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "DepthwiseConv",
    "SeparableConv",
    "TransposedConv",
    "MaxPool",
    "AvgPool",
    "AdaptiveAvgPool",
    "AdaptiveMaxPool",
    "GlobalAvgPool",
    "GlobalMaxPool",
    # Debug/Logging
    "Log",
]
