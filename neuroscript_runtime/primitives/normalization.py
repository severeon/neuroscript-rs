"""
Normalization primitives.

Maps to NeuroScript neurons like:
    neuron LayerNorm(normalized_shape, eps):
        in: [*, normalized_shape]
        out: [*, normalized_shape]
        impl: core,nn/norm
"""

from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    Layer normalization.

    Normalizes across the last dimensions with learnable affine transformation.
    Commonly used in transformers instead of BatchNorm.

    Args:
        normalized_shape: Size of dimensions to normalize over (int or list)
                         If int, normalizes over last dimension only
        eps: Small constant for numerical stability. Default: 1e-5
        elementwise_affine: If True, learnable affine parameters. Default: True

    Shape:
        - Input: [*, normalized_shape]
        - Output: [*, normalized_shape] same shape as input

    Examples:
        >>> # Normalize over last dimension
        >>> layer_norm = LayerNorm(512)
        >>> x = torch.randn(32, 10, 512)
        >>> out = layer_norm(x)  # normalized over dim=512
        >>> out.shape
        torch.Size([32, 10, 512])

        >>> # Normalize over last two dimensions
        >>> layer_norm = LayerNorm([10, 512])
        >>> x = torch.randn(32, 10, 512)
        >>> out = layer_norm(x)  # normalized over [10, 512]

    Notes:
        - Mean and variance computed over normalized_shape dimensions
        - Affine transform: y = (x - mean) / sqrt(var + eps) * gamma + beta
        - gamma (weight) and beta (bias) are learnable if elementwise_affine=True
        - Independent of batch size (unlike BatchNorm)
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]

        if not normalized_shape:
            raise ValueError("normalized_shape cannot be empty")

        if any(d <= 0 for d in normalized_shape):
            raise ValueError(f"All dimensions must be positive, got {normalized_shape}")

        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # Use PyTorch's LayerNorm implementation
        self.layer_norm = nn.LayerNorm(
            normalized_shape=self.normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.

        Args:
            input: Input tensor [*, normalized_shape]

        Returns:
            Normalized tensor of same shape

        Raises:
            ValueError: If input shape doesn't match normalized_shape
        """
        # Validate input shape matches normalized_shape
        if input.shape[-len(self.normalized_shape):] != self.normalized_shape:
            raise ValueError(
                f"Input shape {list(input.shape)} doesn't match normalized_shape "
                f"{list(self.normalized_shape)}. Last {len(self.normalized_shape)} "
                f"dimensions must be {list(self.normalized_shape)}"
            )

        return self.layer_norm(input)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"normalized_shape={list(self.normalized_shape)}, "
            f"eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )

    @property
    def weight(self) -> torch.Tensor:
        """Learnable scale parameter (gamma)."""
        return self.layer_norm.weight

    @property
    def bias(self) -> torch.Tensor:
        """Learnable shift parameter (beta)."""
        return self.layer_norm.bias


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Simpler variant of LayerNorm that only normalizes by RMS (no mean centering).
    Used in some efficient transformer variants (e.g., LLaMA).

    Args:
        dim: Dimension to normalize over (last dimension)
        eps: Small constant for numerical stability. Default: 1e-6

    Shape:
        - Input: [*, dim]
        - Output: [*, dim] same shape as input

    Examples:
        >>> rms_norm = RMSNorm(512)
        >>> x = torch.randn(32, 10, 512)
        >>> out = rms_norm(x)

    Notes:
        - RMSNorm(x) = x / RMS(x) * gamma where RMS(x) = sqrt(mean(x^2) + eps)
        - Cheaper than LayerNorm (no mean computation)
        - gamma is learnable scale parameter
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.dim = dim
        self.eps = eps

        # Learnable scale parameter
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            input: Input tensor [*, dim]

        Returns:
            Normalized tensor of same shape

        Raises:
            ValueError: If input's last dimension doesn't match dim
        """
        if input.size(-1) != self.dim:
            raise ValueError(
                f"Input last dimension mismatch: expected {self.dim}, "
                f"got {input.size(-1)} (input shape: {list(input.shape)})"
            )

        # Compute RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(input ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        return input / rms * self.weight

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"dim={self.dim}, eps={self.eps}"


class GroupNorm(nn.Module):
    """
    Group Normalization.

    Divides channels into groups and normalizes within each group.
    Works well for small batch sizes (unlike BatchNorm).

    Args:
        num_groups: Number of groups to divide channels into
        num_channels: Number of channels (C)
        eps: Small constant for numerical stability. Default: 1e-5
        affine: If True, learnable affine parameters. Default: True

    Shape:
        - Input: [N, C, *] where C must be divisible by num_groups
        - Output: [N, C, *] same shape as input

    Examples:
        >>> # 32 channels, 8 groups (4 channels per group)
        >>> group_norm = GroupNorm(num_groups=8, num_channels=32)
        >>> x = torch.randn(16, 32, 64, 64)  # [batch, channels, H, W]
        >>> out = group_norm(x)

    Notes:
        - If num_groups == num_channels: equivalent to InstanceNorm
        - If num_groups == 1: equivalent to LayerNorm (for spatial dims)
        - Common for computer vision tasks
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()

        if num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {num_groups}")

        if num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}")

        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by "
                f"num_groups ({num_groups})"
            )

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        # Use PyTorch's GroupNorm implementation
        self.group_norm = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply group normalization.

        Args:
            input: Input tensor [N, C, *]

        Returns:
            Normalized tensor of same shape

        Raises:
            ValueError: If input channel dimension doesn't match num_channels
        """
        if input.dim() < 2:
            raise ValueError(
                f"Input must be at least 2D [N, C, ...], got shape {list(input.shape)}"
            )

        if input.size(1) != self.num_channels:
            raise ValueError(
                f"Input channel dimension mismatch: expected {self.num_channels}, "
                f"got {input.size(1)} (input shape: {list(input.shape)})"
            )

        return self.group_norm(input)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"num_groups={self.num_groups}, "
            f"num_channels={self.num_channels}, "
            f"eps={self.eps}, "
            f"affine={self.affine}"
        )

    @property
    def weight(self) -> torch.Tensor:
        """Learnable scale parameter."""
        return self.group_norm.weight

    @property
    def bias(self) -> torch.Tensor:
        """Learnable shift parameter."""
        return self.group_norm.bias


class BatchNorm(nn.Module):
    """
    Batch Normalization.

    Normalizes across the batch and spatial dimensions.
    Automatically handles 1D, 2D, or 3D spatial data based on input rank.

    Args:
        num_features: Number of features (C)
        eps: Small constant for numerical stability. Default: 1e-5
        momentum: Momentum for running stats. Default: 0.1
        affine: If True, learnable affine parameters. Default: True
        track_running_stats: If True, tracks running mean/var. Default: True

    Shape:
        - Input: [N, C, *]
        - Output: [N, C, *] same shape as input
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        # We'll use 2D as default, but it can work for 1D/3D if we're careful 
        # or if we wrap the specific versions. Many PyTorch BatchNorms share 
        # the same parameter structure.
        self.batch_norm = nn.BatchNorm2d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply batch normalization."""
        if input.dim() == 2 or input.dim() == 3:
            # Re-wrap if needed for 1D
            if not isinstance(self.batch_norm, nn.BatchNorm1d):
                self.batch_norm = nn.BatchNorm1d(
                    self.num_features, self.eps, self.momentum, self.affine, 
                    self.track_running_stats
                ).to(input.device).to(input.dtype)
        elif input.dim() == 4:
            if not isinstance(self.batch_norm, nn.BatchNorm2d):
                self.batch_norm = nn.BatchNorm2d(
                    self.num_features, self.eps, self.momentum, self.affine, 
                    self.track_running_stats
                ).to(input.device).to(input.dtype)
        elif input.dim() == 5:
            if not isinstance(self.batch_norm, nn.BatchNorm3d):
                self.batch_norm = nn.BatchNorm3d(
                    self.num_features, self.eps, self.momentum, self.affine, 
                    self.track_running_stats
                ).to(input.device).to(input.dtype)

        return self.batch_norm(input)


class InstanceNorm(nn.Module):
    """
    Instance Normalization.

    Normalizes across the spatial dimensions per channel and per instance.
    Common in style transfer and generative models.

    Args:
        num_features: Number of features (C)
        eps: Small constant for numerical stability. Default: 1e-5
        momentum: Momentum for running stats. Default: 0.1
        affine: If True, learnable affine parameters. Default: False
        track_running_stats: If True, tracks running mean/var. Default: False

    Shape:
        - Input: [N, C, *]
        - Output: [N, C, *] same shape as input
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.instance_norm = nn.InstanceNorm2d(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply instance normalization."""
        if input.dim() == 3:
            if not isinstance(self.instance_norm, nn.InstanceNorm1d):
                self.instance_norm = nn.InstanceNorm1d(
                    self.num_features, self.eps, self.momentum, self.affine, 
                    self.track_running_stats
                ).to(input.device).to(input.dtype)
        elif input.dim() == 4:
            if not isinstance(self.instance_norm, nn.InstanceNorm2d):
                self.instance_norm = nn.InstanceNorm2d(
                    self.num_features, self.eps, self.momentum, self.affine, 
                    self.track_running_stats
                ).to(input.device).to(input.dtype)
        elif input.dim() == 5:
            if not isinstance(self.instance_norm, nn.InstanceNorm3d):
                self.instance_norm = nn.InstanceNorm3d(
                    self.num_features, self.eps, self.momentum, self.affine, 
                    self.track_running_stats
                ).to(input.device).to(input.dtype)

        return self.instance_norm(input)


class WeightNorm(nn.Module):
    """
    Weight Normalization (activation-side).
    
    Normalizes the input tensor by its L2 norm along a given dimension.
    This is often used as a simpler alternative or component of weight norm 
    mechanisms that can be applied directly in the computation graph.

    Args:
        dim: Dimension to normalize over. Default: -1
        eps: Small constant for numerical stability. Default: 1e-12
    """

    def __init__(self, dim: int = -1, eps: float = 1e-12) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Normalize input by its L2 norm."""
        return input / (input.norm(p=2, dim=self.dim, keepdim=True) + self.eps)
