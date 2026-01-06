"""
Matrix and tensor contraction operations.

Provides:
- MatMul: Matrix multiplication between two tensors
- Einsum: Einstein summation for generalized tensor operations
"""

import torch
import torch.nn as nn
from typing import Tuple, Union


class MatMul(nn.Module):
    """
    Matrix multiplication: y = x1 @ x2

    NeuroScript signature:
        neuron MatMul:
            in left: [*shape, i, j]
            in right: [*shape, j, k]
            out: [*shape, i, k]
            impl: neuroscript_runtime.primitives.matrix.MatMul

    Shape:
        - Input: tuple of (left, right) tensors
        - Output: Result of matrix multiplication
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Multiply two matrices."""
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError(f"MatMul requires exactly 2 inputs, got {type(inputs)}")
        
        left, right = inputs
        try:
            return torch.matmul(left, right)
        except RuntimeError as e:
            raise ValueError(
                f"Cannot multiply tensors with shapes {left.shape} and {right.shape}: {e}"
            )


class Einsum(nn.Module):
    """
    Einstein summation: y = einsum(equation, *inputs)

    NeuroScript signature:
        neuron Einsum(equation):
            in: [*]
            out: [*]
            impl: neuroscript_runtime.primitives.matrix.Einsum

    Args:
        equation (str): Einstein summation equation (e.g., 'bij,bjk->bik')

    Shape:
        - Input: tuple of tensors
        - Output: Result of einsum operation
    """

    def __init__(self, equation: str):
        super().__init__()
        if not isinstance(equation, str):
            raise TypeError(f"equation must be a string, got {type(equation)}")
        self.equation = equation

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """Apply Einstein summation."""
        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)
        
        try:
            return torch.einsum(self.equation, *inputs)
        except RuntimeError as e:
            raise ValueError(
                f"Einsum failed with equation '{self.equation}': {e}"
            )

    def extra_repr(self) -> str:
        return f"equation='{self.equation}'"
