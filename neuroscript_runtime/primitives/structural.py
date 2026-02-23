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

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fork the input tensor into three references.

        Args:
            input: Input tensor of any shape

        Returns:
            Tuple of three references to the input tensor
        """
        return (input, input, input)


class ForkN(nn.Module):
    """Generic N-way fork for explicit use.

    Splits input tensor into N references. Unlike Fork (2-way) and Fork3 (3-way),
    this supports any number of outputs.

    Args:
        n (int): Number of output references.

    Shape:
        - Input: [*] (any shape)
        - Output: tuple of N references to the same tensor
    """

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(input for _ in range(self.n))


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


class Multiply(nn.Module):
    """
    Multiply primitive: Element-wise multiplication of two tensors.

    This is a multi-input neuron that performs element-wise multiplication (Hadamard product).
    Primary use case is gating mechanisms where one tensor modulates another,
    such as in GLU (Gated Linear Units) and attention mechanisms.

    NeuroScript signature:
        neuron Multiply:
            in left: [*shape]
            in right: [*shape]
            out: [*shape]
            impl: neuroscript_runtime.primitives.Multiply

    Args:
        None

    Shape:
        - Input: tuple([*shape], [*shape]) where both tensors must be broadcastable
        - Output: [*shape] (result of element-wise multiplication)

    Notes:
        The tensors must have compatible shapes for broadcasting according to
        PyTorch broadcasting rules. Typically used with tensors of identical shape.

    Example:
        >>> multiply = Multiply()
        >>> gate = torch.sigmoid(torch.randn(32, 512))
        >>> value = torch.randn(32, 512)
        >>> result = multiply((gate, value))
        >>> assert result.shape == (32, 512)
        >>> assert torch.allclose(result, gate * value)
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Multiply two tensors element-wise.

        Args:
            inputs: Tuple of (left, right) tensors to multiply

        Returns:
            Element-wise product of the two input tensors

        Raises:
            ValueError: If inputs cannot be broadcast together
        """
        left, right = inputs

        try:
            result = left * right
        except RuntimeError as e:
            raise ValueError(
                f"Cannot multiply tensors with shapes {left.shape} and {right.shape}: {e}"
            )

        return result


class Subtract(nn.Module):
    """
    Subtract primitive: Element-wise subtraction of two tensors.

    Computes left - right. Primary use cases include complement operations
    (1 - gate) for highway connections and residual difference computations.

    NeuroScript signature:
        neuron Subtract:
            in left: [*shape]
            in right: [*shape]
            out: [*shape]
            impl: neuroscript_runtime.primitives.Subtract

    Shape:
        - Input: tuple([*shape], [*shape]) where both tensors must be broadcastable
        - Output: [*shape] (result of element-wise subtraction)

    Example:
        >>> subtract = Subtract()
        >>> a = torch.ones(32, 512)
        >>> gate = torch.sigmoid(torch.randn(32, 512))
        >>> complement = subtract((a, gate))  # 1 - gate
        >>> assert complement.shape == (32, 512)
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Subtract right tensor from left tensor element-wise.

        Args:
            inputs: Tuple of (left, right) tensors

        Returns:
            Element-wise difference (left - right)

        Raises:
            ValueError: If inputs cannot be broadcast together
        """
        left, right = inputs

        try:
            result = left - right
        except RuntimeError as e:
            raise ValueError(
                f"Cannot subtract tensors with shapes {left.shape} and {right.shape}: {e}"
            )

        return result


class Divide(nn.Module):
    """
    Divide primitive: Element-wise division of two tensors.

    Computes left / right. Primary use cases include normalized attention
    weights and scaling operations.

    NeuroScript signature:
        neuron Divide:
            in left: [*shape]
            in right: [*shape]
            out: [*shape]
            impl: neuroscript_runtime.primitives.Divide

    Shape:
        - Input: tuple([*shape], [*shape]) where both tensors must be broadcastable
        - Output: [*shape] (result of element-wise division)

    Example:
        >>> divide = Divide()
        >>> scores = torch.randn(32, 8, 64, 64)
        >>> scale = torch.tensor(8.0)  # sqrt(d_k)
        >>> scaled = divide((scores, scale))
        >>> assert scaled.shape == (32, 8, 64, 64)
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Divide left tensor by right tensor element-wise.

        Args:
            inputs: Tuple of (left, right) tensors

        Returns:
            Element-wise quotient (left / right)

        Raises:
            ValueError: If inputs cannot be broadcast together
        """
        left, right = inputs

        try:
            result = left / right
        except RuntimeError as e:
            raise ValueError(
                f"Cannot divide tensors with shapes {left.shape} and {right.shape}: {e}"
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
            raise ValueError(f"Concat requires at least 2 tensors, got {len(inputs)}")

        try:
            result = torch.cat(inputs, dim=self.dim)
        except RuntimeError as e:
            shapes = [t.shape for t in inputs]
            raise ValueError(
                f"Cannot concatenate tensors with shapes {shapes} along dim {self.dim}: {e}"
            )

        return result


class Reshape(nn.Module):
    """
    Reshape primitive: Reshape tensor to a new shape.

    Changes the shape of a tensor without changing its data. The total number
    of elements must remain the same. Supports dynamic shapes with -1 for
    auto-inferred dimensions.

    NeuroScript signature:
        neuron Reshape(shape):
            in: [*input_shape]
            out: [*output_shape]
            impl: neuroscript_runtime.primitives.Reshape

    Args:
        shape (tuple of int): Target shape. Use -1 for one dimension to be inferred.

    Shape:
        - Input: [*input_shape] (any shape)
        - Output: [*output_shape] where product of dims equals input

    Notes:
        - Total number of elements must be preserved
        - One dimension can be -1 for automatic inference
        - Commonly used for multi-head attention to split/merge heads

    Example:
        >>> reshape = Reshape((32, 10, 8, 64))
        >>> x = torch.randn(32, 10, 512)  # batch, seq, d_model where d_model=512=8*64
        >>> result = reshape(x)
        >>> assert result.shape == (32, 10, 8, 64)  # batch, seq, num_heads, d_k
    """

    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        if not isinstance(shape, (tuple, list)):
            raise TypeError(f"shape must be tuple or list, got {type(shape)}")
        self.shape = tuple(shape)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Reshape the input tensor to the target shape.

        Args:
            input: Input tensor of any shape

        Returns:
            Reshaped tensor

        Raises:
            ValueError: If reshape is not possible (incompatible sizes)
        """
        try:
            # Handle dynamic batch dimensions by replacing -1 appropriately
            # and ensuring total element count matches
            result = input.view(*self.shape)
        except RuntimeError as e:
            raise ValueError(
                f"Cannot reshape tensor with shape {input.shape} to {self.shape}: {e}"
            )

        return result


class Transpose(nn.Module):
    """
    Transpose primitive: Permute dimensions of a tensor.

    Rearranges the dimensions of a tensor according to a permutation.
    Also called "permute" in PyTorch terminology.

    NeuroScript signature:
        neuron Transpose(dims):
            in: [*input_shape]
            out: [*output_shape]
            impl: neuroscript_runtime.primitives.Transpose

    Args:
        dims (tuple of int): Permutation of dimensions. For example,
            (0, 2, 1) swaps the last two dimensions for a 3D tensor.

    Shape:
        - Input: [d0, d1, ..., dn]
        - Output: [d_perm[0], d_perm[1], ..., d_perm[n]] where perm is the permutation

    Notes:
        - Number of dimensions in dims must match input tensor rank
        - Each dimension index must appear exactly once
        - Commonly used in multi-head attention to rearrange batch/head/seq dims

    Example:
        >>> transpose = Transpose((0, 2, 1, 3))
        >>> x = torch.randn(32, 10, 8, 64)  # batch, seq, num_heads, d_k
        >>> result = transpose(x)
        >>> assert result.shape == (32, 8, 10, 64)  # batch, num_heads, seq, d_k
    """

    def __init__(self, dims: Tuple[int, ...]):
        super().__init__()
        if not isinstance(dims, (tuple, list)):
            raise TypeError(f"dims must be tuple or list, got {type(dims)}")
        self.dims = tuple(dims)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Permute the dimensions of the input tensor.

        Args:
            input: Input tensor

        Returns:
            Transposed tensor with permuted dimensions

        Raises:
            ValueError: If permutation is invalid for the input shape
        """
        if len(self.dims) != input.ndim:
            raise ValueError(
                f"Permutation has {len(self.dims)} dimensions but input has {input.ndim} dimensions"
            )

        try:
            result = input.permute(*self.dims)
        except RuntimeError as e:
            raise ValueError(
                f"Cannot permute tensor with shape {input.shape} using dims {self.dims}: {e}"
            )

        return result


class Split(nn.Module):
    """Splits tensor into chunks along a dimension."""

    def __init__(self, num_splits: int, dim: int = -1) -> None:
        super().__init__()
        self.num_splits = num_splits
        self.dim = dim

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return tuple(torch.chunk(input, self.num_splits, dim=self.dim))


class Slice(nn.Module):
    """Extracts a slice from tensor along a dimension."""

    def __init__(self, dim: int, start: int, end: int) -> None:
        super().__init__()
        self.dim = dim
        self.start = start
        self.end = end if end != -1 else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        slices = [slice(None)] * input.ndim
        slices[self.dim] = slice(self.start, self.end)
        return input[tuple(slices)]


class Pad(nn.Module):
    """Pads tensor with specified value."""

    def __init__(
        self,
        padding: Tuple[int, ...],
        value: float = 0,
        mode: str = "constant",
    ) -> None:
        super().__init__()
        self.padding = padding
        self.value = value
        self.mode = mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        if self.mode == "constant":
            return F.pad(input, self.padding, mode=self.mode, value=self.value)
        return F.pad(input, self.padding, mode=self.mode)


class Crop(nn.Module):
    """
    Crop tensor to target spatial dimensions.

    Crops the input tensor's spatial dimensions (last N dims) to match
    specified target sizes, cropping symmetrically from edges.
    Useful for U-Net skip connections where sizes may not match exactly.

    NeuroScript signature:
        neuron Crop(target_size):
            in: [*batch, *spatial]
            out: [*batch, *target_size]
            impl: neuroscript_runtime.primitives.Crop

    Args:
        target_size (tuple of int): Target spatial size as tuple (e.g., (H, W) for 2D)

    Shape:
        - Input: [batch, channels, *spatial] where spatial dims >= target_size
        - Output: [batch, channels, *target_size]

    Example:
        >>> crop = Crop((256, 256))
        >>> x = torch.randn(1, 64, 260, 260)
        >>> result = crop(x)
        >>> assert result.shape == (1, 64, 256, 256)
    """

    def __init__(self, target_size: Tuple[int, ...]):
        super().__init__()
        if not isinstance(target_size, (tuple, list)):
            raise TypeError(f"target_size must be tuple or list, got {type(target_size)}")
        self.target_size = tuple(target_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Center-crop the input tensor's spatial dimensions.

        Args:
            input: Input tensor with spatial dims at least as large as target_size

        Returns:
            Cropped tensor with spatial dims matching target_size

        Raises:
            ValueError: If input spatial dims are smaller than target_size
        """
        n_spatial = len(self.target_size)
        spatial_dims = input.shape[-n_spatial:]

        slices = [slice(None)] * (input.ndim - n_spatial)
        for i in range(n_spatial):
            in_size = spatial_dims[i]
            out_size = self.target_size[i]
            if in_size < out_size:
                raise ValueError(
                    f"Input spatial dim {i} has size {in_size} but target is {out_size}"
                )
            offset = (in_size - out_size) // 2
            slices.append(slice(offset, offset + out_size))

        return input[tuple(slices)]

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"target_size={self.target_size}"


class Cast(nn.Module):
    """
    Convert tensor to specified dtype.

    Useful for mixed-precision training or when connecting layers
    that expect different dtypes.

    NeuroScript signature:
        neuron Cast(dtype):
            in: [*shape]
            out: [*shape]
            impl: neuroscript_runtime.primitives.Cast

    Args:
        dtype (str): Target dtype as string. One of: 'float16', 'float32', 'float64',
            'bfloat16', 'int32', 'int64', 'bool'

    Shape:
        - Input: [*shape] any shape
        - Output: [*shape] same shape, different dtype

    Example:
        >>> cast = Cast('float16')
        >>> x = torch.randn(32, 512)  # float32
        >>> result = cast(x)
        >>> assert result.dtype == torch.float16
        >>> assert result.shape == (32, 512)
    """

    DTYPE_MAP = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }

    def __init__(self, dtype: str):
        super().__init__()
        if dtype not in self.DTYPE_MAP:
            valid = ", ".join(sorted(self.DTYPE_MAP.keys()))
            raise ValueError(f"Unsupported dtype '{dtype}'. Valid options: {valid}")
        self.dtype_str = dtype
        self.dtype = self.DTYPE_MAP[dtype]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Cast the input tensor to the target dtype.

        Args:
            input: Input tensor of any shape and dtype

        Returns:
            Tensor with the same shape but target dtype
        """
        return input.to(self.dtype)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"dtype='{self.dtype_str}'"


class Clone(nn.Module):
    """
    Create a copy of the input tensor.

    Unlike Fork (which returns references to the same tensor),
    Clone creates an independent copy. Useful when downstream operations
    modify tensors in-place.

    NeuroScript signature:
        neuron Clone(detach=False):
            in: [*shape]
            out: [*shape]
            impl: neuroscript_runtime.primitives.Clone

    Args:
        detach (bool): If True, detaches from computation graph. Default: False

    Shape:
        - Input: [*shape] any shape
        - Output: [*shape] same shape (independent copy)

    Example:
        >>> clone = Clone()
        >>> x = torch.randn(32, 512, requires_grad=True)
        >>> y = clone(x)
        >>> assert y.shape == (32, 512)
        >>> assert y is not x
        >>> assert y.requires_grad  # gradients preserved

        >>> clone_detached = Clone(detach=True)
        >>> z = clone_detached(x)
        >>> assert not z.requires_grad  # detached from graph
    """

    def __init__(self, detach: bool = False):
        super().__init__()
        self.detach = detach

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Clone the input tensor.

        Args:
            input: Input tensor of any shape

        Returns:
            Independent copy of the input tensor
        """
        result = input.clone()
        if self.detach:
            result = result.detach()
        return result

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"detach={self.detach}"


__all__ = [
    "Fork",
    "Fork3",
    "ForkN",
    "Add",
    "Multiply",
    "Concat",
    "Reshape",
    "Transpose",
    "Split",
    "Slice",
    "Pad",
    "Crop",
    "Cast",
    "Clone",
]
