"""
NeuroScript Runtime - Core primitives for compiled NeuroScript models.

This package provides PyTorch implementations of primitive neurons
that compiled NeuroScript code imports and uses.
"""

__version__ = "0.1.0"

from neuroscript_runtime.primitives import (
    Linear,
    GELU,
    Dropout,
    LayerNorm,
    Embedding,
)

__all__ = [
    "Linear",
    "GELU",
    "Dropout",
    "LayerNorm",
    "Embedding",
]
