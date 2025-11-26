"""
Structural operations for tensor manipulation in NeuroScript.

Provides primitives for:
- Fork: Splitting tensor into multiple references (multi-output)
- Add: Element-wise addition (multi-input, for residual connections)
- Concat: Concatenating tensors along a dimension
"""

import torch
import torch.nn as nn
from typing import Tuple


class Fork(nn.Module):
    """
    Fork primitive: Split input tensor into multiple references.

    This is a multi-output neuron that returns a tuple of references to the
    same input tensor (not copies). Used for branching dataflow in residual
    connections and parallel processing paths.

    NeuroScript signature:
        neuron Fork:
            in: [*shape]
            out a: [*shape]
            out b: [*shape]
            impl: neuroscript_runtime.primitives.Fork

    Args:
        None

    Shape:
        - Input: [*] (any shape)
        - Output: tuple([*], [*]) where both elements reference the same tensor

    Example:
        >>> fork = Fork()
        >>> x = torch.randn(32, 512)
        >>> a, b = fork(x)
        >>> assert a.shape == b.shape == (32, 512)
        >>> assert a is b  # Same reference, not a copy
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fork the input tensor into two references.

        Args:
            input: Input tensor of any shape

        Returns:
            Tuple of two references to the input tensor
        """
        return (input, input)


class Fork3(nn.Module):
    """
    Fork3 primitive: Split input tensor into three references.

    Similar to Fork but returns three references instead of two.

    NeuroScript signature:
        neuron Fork3:
            in: [*shape]
            out a: [*shape]
            out b: [*shape]
            out c: [*shape]
            impl: neuroscript_runtime.primitives.Fork3

    Args:
        None

    Shape:
        - Input: [*] (any shape)
        - Output: tuple([*], [*], [*]) where all elements reference the same tensor

    Example:
        >>> fork3 = Fork3()
        >>> x = torch.randn(32, 512)
        >>> a, b, c = fork3(x)
        >>> assert a.shape == b.shape == c.shape == (32, 512)
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fork the input tensor into three references.

        Args:
            input: Input tensor of any shape

        Returns:
            Tuple of three references to the input tensor
        """
        return (input, input, input)


class Add(nn.Module):
    """
    Add primitive: Element-wise addition of two tensors.

    This is a multi-input neuron that performs element-wise addition.
    Primary use case is residual connections where a processed tensor
    is added back to the original (skip connection).

    NeuroScript signature:
        neuron Add:
            in left: [*shape]
            in right: [*shape]
            out: [*shape]
            impl: neuroscript_runtime.primitives.Add

    Args:
        None

    Shape:
        - Input: tuple([*shape], [*shape]) where both tensors must be broadcastable
        - Output: [*shape] (result of element-wise addition)

    Notes:
        The tensors must have compatible shapes for broadcasting according to
        PyTorch broadcasting rules. Typically used with tensors of identical shape.

    Example:
        >>> add = Add()
        >>> x = torch.randn(32, 512)
        >>> y = torch.randn(32, 512)
        >>> result = add((x, y))
        >>> assert result.shape == (32, 512)
        >>> assert torch.allclose(result, x + y)
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Add two tensors element-wise.

        Args:
            inputs: Tuple of (left, right) tensors to add

        Returns:
            Element-wise sum of the two input tensors

        Raises:
            ValueError: If inputs cannot be broadcast together
        """
        left, right = inputs

        try:
            result = left + right
        except RuntimeError as e:
            raise ValueError(
                f"Cannot add tensors with shapes {left.shape} and {right.shape}: {e}"
            )

        return result


class Concat(nn.Module):
    """
    Concat primitive: Concatenate tensors along a specified dimension.

    Multi-input neuron that concatenates two or more tensors along a given
    dimension. All tensors must have the same shape except in the concatenation
    dimension.

    NeuroScript signature:
        neuron Concat(dim):
            in a: [*shape]
            in b: [*shape]
            out: [*shape]
            impl: neuroscript_runtime.primitives.Concat

    Args:
        dim (int): Dimension along which to concatenate. Default: -1 (last dimension)

    Shape:
        - Input: tuple of tensors with compatible shapes
        - Output: Concatenated tensor with size increased in dimension `dim`

    Notes:
        - Supports concatenating 2 or more tensors
        - All tensors must have the same number of dimensions
        - All dimensions except `dim` must have the same size

    Example:
        >>> concat = Concat(dim=-1)
        >>> x = torch.randn(32, 10, 512)
        >>> y = torch.randn(32, 10, 512)
        >>> result = concat((x, y))
        >>> assert result.shape == (32, 10, 1024)  # Concatenated along last dim
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        Concatenate input tensors along the specified dimension.

        Args:
            inputs: Tuple of tensors to concatenate

        Returns:
            Concatenated tensor

        Raises:
            ValueError: If fewer than 2 tensors provided or shapes incompatible
        """
        if len(inputs) < 2:
            raise ValueError(
                f"Concat requires at least 2 tensors, got {len(inputs)}"
            )

        try:
            result = torch.cat(inputs, dim=self.dim)
        except RuntimeError as e:
            shapes = [t.shape for t in inputs]
            raise ValueError(
                f"Cannot concatenate tensors with shapes {shapes} along dim {self.dim}: {e}"
            )

        return result


__all__ = ["Fork", "Fork3", "Add", "Concat"]
