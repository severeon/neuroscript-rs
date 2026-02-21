"""
Core operation primitives for NeuroScript.

Maps to NeuroScript neurons like:
    neuron Bias(dim):
        in: [*, dim]
        out: [*, dim]
        impl: core,operations/Bias
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class Bias(nn.Module):
    """
    Learnable additive bias.

    Adds a bias vector to the input tensor along the last dimension.

    Args:
        dim: Size of the bias vector

    Shape:
        - Input: [*, dim]
        - Output: [*, dim]
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.size(-1) != self.dim:
            raise ValueError(
                f"Input last dimension {input.size(-1)} doesn't match bias dim {self.dim}"
            )
        return input + self.bias

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Scale(nn.Module):
    """
    Learnable multiplicative scale.

    Multiplies input by a scale vector along the last dimension.

    Args:
        dim: Size of the scale vector

    Shape:
        - Input: [*, dim]
        - Output: [*, dim]
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.size(-1) != self.dim:
            raise ValueError(
                f"Input last dimension {input.size(-1)} doesn't match scale dim {self.dim}"
            )
        return input * self.scale

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class MatMul(nn.Module):
    """
    Matrix multiplication of two tensors.

    Performs batched matrix multiplication following PyTorch broadcasting rules.

    Shape:
        - Input a: [*, n, m]
        - Input b: [*, m, p]
        - Output: [*, n, p]
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        a, b = inputs
        if a.size(-1) != b.size(-2):
            raise ValueError(f"Inner dimensions don't match: {a.shape} x {b.shape}")
        return torch.matmul(a, b)


class Einsum(nn.Module):
    """
    Einstein summation for generalized tensor operations.

    Args:
        equation: Einsum equation string (e.g., "bij,bjk->bik")

    Shape:
        Depends on equation
    """

    def __init__(self, equation: str) -> None:
        super().__init__()
        if not equation or "->" not in equation:
            raise ValueError(f"Invalid einsum equation: {equation}")
        self.equation = equation

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return torch.einsum(self.equation, inputs)
        return torch.einsum(self.equation, *inputs)

    def extra_repr(self) -> str:
        return f"equation='{self.equation}'"


class ConstScale(nn.Module):
    """
    Constant (non-learnable) scalar multiplication.

    Multiplies input by a fixed scalar factor.

    Args:
        factor: The constant multiplier

    Shape:
        - Input: [*shape]
        - Output: [*shape]
    """

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.factor

    def extra_repr(self) -> str:
        return f"factor={self.factor}"


class Identity(nn.Module):
    """
    Identity operation - returns input unchanged.

    Shape:
        - Input: [*shape]
        - Output: [*shape]
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


__all__ = ["Bias", "Scale", "ConstScale", "MatMul", "Einsum", "Identity"]
