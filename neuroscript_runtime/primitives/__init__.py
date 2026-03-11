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
    WeightNorm,
)

from neuroscript_runtime.primitives.regularization import (
    Dropout,
    DropPath,
    DropConnect,
    Dropblock,
    SpecAugment,
)

from neuroscript_runtime.primitives.embeddings import (
    Embedding,
    PositionalEncoding,
    LearnedPositionalEmbedding,
    RotaryEmbedding,
    ALiBi,
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
    Crop,
    Cast,
    Clone,
)

from neuroscript_runtime.primitives.attention import (
    ScaledDotProductAttention,
    MultiHeadSelfAttention,
    MultiHeadLatentAttention,
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
    DilatedConv,
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

# Connections (Hyper-Connections)
from neuroscript_runtime.primitives.connections import (
    HyperExpand,
    HyperCollapse,
    HCWidth,
    HCDepth,
    ManifoldHyperConnect,
    LearnableResidual,
    sinkhorn_knopp,
)

# Diffusion
from neuroscript_runtime.primitives.diffusion import (
    DenoisingHead,
    MultiTokenPredictionHead,
)

# Routing
from neuroscript_runtime.primitives.routing import (
    SigmoidMoERouter,
    MoERouter,
)

# Selective State Space Models
from neuroscript_runtime.primitives.ssm import MambaBlock

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
    "WeightNorm",
    "Dropout",
    "DropPath",
    "DropConnect",
    "Dropblock",
    "SpecAugment",
    "Embedding",
    "PositionalEncoding",
    "LearnedPositionalEmbedding",
    "RotaryEmbedding",
    "ALiBi",
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
    "Crop",
    "Cast",
    "Clone",
    "ScaledDotProductAttention",
    "MultiHeadSelfAttention",
    "MultiHeadLatentAttention",
    "Bias",
    "Scale",
    "MatMul",
    "Einsum",
    "Identity",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "DepthwiseConv",
    "DilatedConv",
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
    # Connections (Hyper-Connections)
    "HyperExpand",
    "HyperCollapse",
    "HCWidth",
    "HCDepth",
    "ManifoldHyperConnect",
    "LearnableResidual",
    # Diffusion
    "DenoisingHead",
    "MultiTokenPredictionHead",
    # Routing
    "SigmoidMoERouter",
    "MoERouter",
    # Selective State Space Models
    "MambaBlock",
]
