"""
Regularization primitives.

Maps to NeuroScript neurons like:
    neuron Dropout(p: float):
        in: [*shape]
        out: [*shape]
        impl: neuroscript_runtime.primitives.Dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Dropout(nn.Module):
    """
    Randomly zero out elements during training for regularization.

    During training, randomly sets elements to zero with probability p,
    and scales remaining elements by 1/(1-p). During evaluation, returns
    input unchanged.

    Args:
        p: Probability of an element being zeroed. Default: 0.5
        inplace: If True, modifies input tensor in-place. Default: False

    Shape:
        - Input: [*shape] arbitrary shape
        - Output: [*shape] same shape as input

    Examples:
        >>> dropout = Dropout(p=0.1)
        >>> x = torch.randn(32, 10, 512)
        >>> dropout.train()
        >>> out_train = dropout(x)  # ~10% of elements zeroed
        >>> dropout.eval()
        >>> out_eval = dropout(x)   # no dropout applied

    Notes:
        - Respects model.train() / model.eval() mode
        - Scaling by 1/(1-p) maintains expected value
        - Common values: 0.1 (attention), 0.5 (FFN)
    """

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()

        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1], got {p}")

        self.p = p
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply dropout.

        Args:
            input: Input tensor of any shape

        Returns:
            Output tensor of same shape (with dropout if training)
        """
        return F.dropout(input, p=self.p, training=self.training, inplace=self.inplace)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"p={self.p}, inplace={self.inplace}"


class DropPath(nn.Module):
    """
    Stochastic depth / drop path regularization.

    Randomly drops entire samples (along batch dimension) during training.
    Used in vision transformers and ResNets for regularization.

    Args:
        drop_prob: Probability of dropping a sample. Default: 0.1
        scale_by_keep: If True, scale by 1/(1-p). Default: True

    Shape:
        - Input: [batch, *shape]
        - Output: [batch, *shape] with some samples zeroed during training

    Examples:
        >>> drop_path = DropPath(drop_prob=0.1)
        >>> x = torch.randn(32, 512)
        >>> drop_path.train()
        >>> out = drop_path(x)  # ~3 samples completely zeroed

    Notes:
        - Different from Dropout: drops entire samples, not individual elements
        - Commonly used in skip connections
        - Only applies during training
    """

    def __init__(self, drop_prob: float = 0.1, scale_by_keep: bool = True) -> None:
        super().__init__()

        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError(f"Drop probability must be in [0, 1], got {drop_prob}")

        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply drop path.

        Args:
            input: Input tensor [batch, *shape]

        Returns:
            Output tensor with some samples dropped if training
        """
        # No dropout during eval or if drop_prob is 0
        if not self.training or self.drop_prob == 0.0:
            return input

        keep_prob = 1.0 - self.drop_prob
        batch_size = input.shape[0]

        # Create random tensor of shape [batch, 1, 1, ...] for broadcasting
        shape = (batch_size,) + (1,) * (input.ndim - 1)
        random_tensor = torch.rand(shape, dtype=input.dtype, device=input.device)
        random_tensor = (random_tensor > self.drop_prob).float()

        if self.scale_by_keep:
            random_tensor = random_tensor / keep_prob

        return input * random_tensor

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"drop_prob={self.drop_prob}, scale_by_keep={self.scale_by_keep}"


class DropConnect(nn.Module):
    """
    Randomly drop connections (weights) instead of activations.

    Similar to Dropout but drops weight connections rather than neuron outputs.
    Used in some mobile architectures.

    Args:
        p: Probability of dropping a connection. Default: 0.5

    Shape:
        - Input: [*shape] arbitrary shape
        - Output: [*shape] same shape as input

    Examples:
        >>> drop_connect = DropConnect(p=0.2)
        >>> x = torch.randn(32, 512)
        >>> out = drop_connect(x)

    Notes:
        - Primarily used in convolutional networks
        - Applies to weights, not activations
        - Only during training
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()

        if not 0.0 <= p <= 1.0:
            raise ValueError(f"DropConnect probability must be in [0, 1], got {p}")

        self.p = p

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply drop connect.

        Args:
            input: Input tensor of any shape

        Returns:
            Output tensor with connections randomly dropped if training
        """
        if not self.training or self.p == 0.0:
            return input

        keep_prob = 1.0 - self.p
        mask = torch.rand_like(input) > self.p
        return input * mask.float() / keep_prob

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"p={self.p}"
