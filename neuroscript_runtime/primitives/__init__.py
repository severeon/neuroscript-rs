"""
Primitive neurons - Level 0 building blocks.

These are the atomic operations that all composite neurons are built from.
Each primitive maps to a NeuroScript neuron with `impl:` reference.
"""

# Core Operations
from neuroscript_runtime.primitives.linear import Linear
from neuroscript_runtime.primitives.arithmetic import Bias, Scale
from neuroscript_runtime.primitives.matrix import MatMul, Einsum

# Activations
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

# Normalization
from neuroscript_runtime.primitives.normalization import (
    LayerNorm,
    RMSNorm,
    GroupNorm,
    BatchNorm,
    InstanceNorm,
    WeightNorm,
)

# Regularization
from neuroscript_runtime.primitives.regularization import (
    Dropout,
    DropPath,
    DropConnect,
)

# Embeddings
from neuroscript_runtime.primitives.embeddings import (
    Embedding,
    PositionalEncoding,
    LearnedPositionalEmbedding,
)

# Structural Operations
from neuroscript_runtime.primitives.structural import (
    Fork,
    Fork3,
    Add,
    Concat,
    Reshape,
    Transpose,
)

# Attention
from neuroscript_runtime.primitives.attention import (
    ScaledDotProductAttention,
    MultiHeadSelfAttention,
)

__all__ = [
    # Core Operations
    "Linear",
    "Bias",
    "Scale",
    "MatMul",
    "Einsum",
    # Activations
    "GELU",
    "ReLU",
    "Tanh",
    "Sigmoid",
    "SiLU",
    "Softmax",
    "Mish",
    "PReLU",
    "ELU",
    # Normalization
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "BatchNorm",
    "InstanceNorm",
    "WeightNorm",
    # Regularization
    "Dropout",
    "DropPath",
    "DropConnect",
    # Embeddings
    "Embedding",
    "PositionalEncoding",
    "LearnedPositionalEmbedding",
    # Structural Operations
    "Fork",
    "Fork3",
    "Add",
    "Concat",
    "Reshape",
    "Transpose",
    # Attention
    "ScaledDotProductAttention",
    "MultiHeadSelfAttention",
]
