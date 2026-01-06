"""
Arithmetic operations with learnable parameters.

Provides:
- Bias: Additive learnable bias
- Scale: Multiplicative learnable scaling
"""

import torch
import torch.nn as nn
from typing import Optional


class Bias(nn.Module):
    """
    Apply a learnable additive bias: y = x + b

    NeuroScript signature:
        neuron Bias(dim):
            in: [*shape, dim]
            out: [*shape, dim]
            impl: neuroscript_runtime.primitives.arithmetic.Bias

    Args:
        dim (int): Size of the bias vector (matching input last dimension)

    Shape:
        - Input: [*, dim]
        - Output: [*, dim]
    """

    def __init__(self, dim: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Add bias to input."""
        if input.size(-1) != self.dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.dim}, got {input.size(-1)}"
            )
        return input + self.bias

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Scale(nn.Module):
    """
    Apply a learnable multiplicative scale: y = x * s

    NeuroScript signature:
        neuron Scale(dim):
            in: [*shape, dim]
            out: [*shape, dim]
            impl: neuroscript_runtime.primitives.arithmetic.Scale

    Args:
        dim (int): Size of the scale vector (matching input last dimension)

    Shape:
        - Input: [*, dim]
        - Output: [*, dim]
    """

    def __init__(self, dim: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Scale input."""
        if input.size(-1) != self.dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.dim}, got {input.size(-1)}"
            )
        return input * self.scale

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
