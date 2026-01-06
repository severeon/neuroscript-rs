"""
Activation function primitives.

Maps to NeuroScript neurons like:
    neuron GELU():
        in: [*shape]
        out: [*shape]
        impl: neuroscript_runtime.primitives.GELU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation function.

    GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution
    function of the standard normal distribution.

    This is the activation used in BERT, GPT-2, GPT-3, and many modern
    transformer architectures.

    Args:
        approximate: If 'tanh', use tanh approximation for faster computation.
                    If 'none', use exact erf-based computation.
                    Default: 'none'

    Shape:
        - Input: [*shape] arbitrary shape
        - Output: [*shape] same shape as input

    Examples:
        >>> gelu = GELU()
        >>> x = torch.randn(32, 512)
        >>> out = gelu(x)
        >>> out.shape
        torch.Size([32, 512])

    Notes:
        - Element-wise operation, preserves shape exactly
        - Smooth, non-monotonic activation
        - Provides better gradients than ReLU in deep networks
    """

    def __init__(self, approximate: str = "none") -> None:
        super().__init__()

        if approximate not in ("none", "tanh"):
            raise ValueError(
                f"approximate must be 'none' or 'tanh', got '{approximate}'"
            )

        self.approximate = approximate

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply GELU activation.

        Args:
            input: Input tensor of any shape

        Returns:
            Output tensor of same shape as input
        """
        return F.gelu(input, approximate=self.approximate)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"approximate='{self.approximate}'"


class ReLU(nn.Module):
    """
    Rectified Linear Unit activation: ReLU(x) = max(0, x)

    Args:
        inplace: If True, modifies input tensor in-place. Default: False

    Shape:
        - Input: [*shape] arbitrary shape
        - Output: [*shape] same shape as input

    Examples:
        >>> relu = ReLU()
        >>> x = torch.randn(32, 512)
        >>> out = relu(x)
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply ReLU activation."""
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"inplace={self.inplace}"


class Tanh(nn.Module):
    """
    Hyperbolic tangent activation: Tanh(x) = (e^x - e^-x) / (e^x + e^-x)

    Shape:
        - Input: [*shape] arbitrary shape
        - Output: [*shape] same shape as input, values in (-1, 1)

    Examples:
        >>> tanh = Tanh()
        >>> x = torch.randn(32, 512)
        >>> out = tanh(x)  # values in range (-1, 1)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply Tanh activation."""
        return torch.tanh(input)


class Sigmoid(nn.Module):
    """
    Sigmoid activation: Sigmoid(x) = 1 / (1 + e^-x)

    Shape:
        - Input: [*shape] arbitrary shape
        - Output: [*shape] same shape as input, values in (0, 1)

    Examples:
        >>> sigmoid = Sigmoid()
        >>> x = torch.randn(32, 512)
        >>> out = sigmoid(x)  # values in range (0, 1)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply Sigmoid activation."""
        return torch.sigmoid(input)


class SiLU(nn.Module):
    """
    Sigmoid Linear Unit (also known as Swish): SiLU(x) = x * sigmoid(x)

    Used in architectures like EfficientNet and some transformer variants.

    Shape:
        - Input: [*shape] arbitrary shape
        - Output: [*shape] same shape as input

    Examples:
        >>> silu = SiLU()
        >>> x = torch.randn(32, 512)
        >>> out = silu(x)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply SiLU activation."""
        return F.silu(input)


class Softmax(nn.Module):
    """
    Softmax activation: Softmax(x_i) = exp(x_i) / Σ_j exp(x_j)

    Args:
        dim: Dimension along which to apply softmax. Default: -1 (last dim)

    Shape:
        - Input: [*shape] arbitrary shape
        - Output: [*shape] same shape, values sum to 1 along dim

    Examples:
        >>> softmax = Softmax(dim=-1)
        >>> x = torch.randn(32, 10, 512)
        >>> out = softmax(x)  # out.sum(dim=-1) == 1.0
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply Softmax activation."""
        return F.softmax(input, dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Mish(nn.Module):
    """Mish(x) = x * tanh(softplus(x)) - self-regularized non-monotonic activation."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.mish(input)


class PReLU(nn.Module):
    """Parametric ReLU with learnable slope for negative values."""

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        super().__init__()
        self.prelu = nn.PReLU(num_parameters=num_parameters, init=init)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.prelu(input)


class ELU(nn.Module):
    """Exponential Linear Unit: ELU(x) = x if x > 0, else alpha * (exp(x) - 1)."""

    def __init__(self, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.elu(input, alpha=self.alpha, inplace=self.inplace)
