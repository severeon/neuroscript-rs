"""
Primitive neurons - Level 0 building blocks.

These are the atomic operations that all composite neurons are built from.
Each primitive maps to a NeuroScript neuron with `impl:` reference.
"""

# Core Operations
from neuroscript_runtime.primitives.linear import Linear

# Activations
from neuroscript_runtime.primitives.activations import (
    GELU,
    ReLU,
    Tanh,
    Sigmoid,
    SiLU,
    Softmax,
)

# Normalization
from neuroscript_runtime.primitives.normalization import (
    LayerNorm,
    RMSNorm,
    GroupNorm,
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

__all__ = [
    # Core Operations
    "Linear",
    # Activations
    "GELU",
    "ReLU",
    "Tanh",
    "Sigmoid",
    "SiLU",
    "Softmax",
    # Normalization
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    # Regularization
    "Dropout",
    "DropPath",
    "DropConnect",
    # Embeddings
    "Embedding",
    "PositionalEncoding",
    "LearnedPositionalEmbedding",
]
