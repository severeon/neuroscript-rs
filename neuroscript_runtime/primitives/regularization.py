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


class Dropblock(nn.Module):
    """
    DropBlock: structured dropout for convolutional neural networks.

    Instead of dropping individual elements like standard dropout, drops
    contiguous square regions (blocks) of feature maps. This is more
    effective for convolutional layers where spatially correlated features
    can bypass element-wise dropout.

    Reference: Ghiasi et al. (2018), "DropBlock: A regularization technique
    for convolutional neural networks"

    Args:
        block_size: Size of the square block to drop. Default: 7
        drop_prob: Probability of dropping a block. Default: 0.1

    Shape:
        - Input: [batch, channels, height, width]
        - Output: [batch, channels, height, width] same shape as input

    Examples:
        >>> dropblock = Dropblock(block_size=7, drop_prob=0.1)
        >>> x = torch.randn(8, 64, 32, 32)
        >>> dropblock.train()
        >>> out_train = dropblock(x)  # contiguous regions zeroed
        >>> dropblock.eval()
        >>> out_eval = dropblock(x)   # no dropout applied

    Notes:
        - More effective than standard dropout for convolutional layers
        - block_size should be smaller than spatial dimensions
        - Output is rescaled to maintain expected values
        - Only applies during training
        - Implementation note: each (batch, channel) pair gets an independently
          sampled mask. The original paper (Ghiasi et al. 2018) drops the same
          spatial region across all channels for a given sample. This per-channel
          variant provides finer-grained regularization.
    """

    def __init__(self, block_size: int = 7, drop_prob: float = 0.1) -> None:
        super().__init__()

        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError(f"drop_prob must be in [0, 1], got {drop_prob}")

        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply DropBlock.

        Args:
            input: Input tensor [batch, channels, height, width]

        Returns:
            Output tensor with contiguous blocks dropped if training
        """
        if not self.training or self.drop_prob == 0.0:
            return input

        if input.ndim != 4:
            raise ValueError(
                f"Dropblock expects 4D input [batch, channels, height, width], got {input.ndim}D"
            )

        _, _, height, width = input.shape

        # Compute gamma: the probability of dropping each element in the
        # seed mask, adjusted so that the effective drop rate after block
        # expansion matches drop_prob.
        feat_area = height * width
        block_area = self.block_size * self.block_size
        # Clamp valid region to avoid division by zero for very small inputs
        valid_h = max(height - self.block_size + 1, 1)
        valid_w = max(width - self.block_size + 1, 1)
        gamma = (self.drop_prob * feat_area) / (block_area * valid_h * valid_w)

        # Sample seed mask (1 = keep, 0 = drop seed)
        mask = (torch.rand_like(input) >= gamma).float()

        # Expand drop regions using max pooling on the inverted mask
        # Invert: 0 becomes 1 (regions to expand), 1 stays 0
        block_mask = 1.0 - F.max_pool2d(
            1.0 - mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )

        # Crop to original spatial size in case padding changed dimensions
        block_mask = block_mask[:, :, :height, :width]

        # Scale output to maintain expected values
        count = block_mask.numel()
        count_ones = block_mask.sum().clamp(min=1.0)
        output = input * block_mask * (count / count_ones)

        return output

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"block_size={self.block_size}, drop_prob={self.drop_prob}"


class SpecAugment(nn.Module):
    """
    SpecAugment: frequency and time masking for audio spectrograms.

    Applies random frequency and time masks to spectrogram inputs,
    zeroing out contiguous bands along frequency and time axes. This
    is a simple but effective data augmentation for speech recognition.

    Reference: Park et al. (2019), "SpecAugment: A Simple Data Augmentation
    Method for ASR"

    Args:
        freq_mask_param: Maximum number of frequency channels to mask. Default: 27
        time_mask_param: Maximum number of time steps to mask. Default: 100
        num_freq_masks: Number of frequency masks to apply. Default: 1
        num_time_masks: Number of time masks to apply. Default: 1

    Shape:
        - Input: [batch, freq, time] or [batch, channels, freq, time]
        - Output: same shape as input with masked regions zeroed

    Examples:
        >>> spec_aug = SpecAugment(freq_mask_param=27, time_mask_param=100)
        >>> x = torch.randn(8, 80, 300)       # [batch, freq, time]
        >>> spec_aug.train()
        >>> out = spec_aug(x)                   # frequency/time bands zeroed
        >>> spec_aug.eval()
        >>> out_eval = spec_aug(x)              # no masking applied

    Notes:
        - Only applies during training
        - Supports both 3D (batch, freq, time) and 4D (batch, channels, freq, time) inputs
        - Mask widths are sampled uniformly from [0, param] (inclusive) for each mask
        - Each sample in the batch receives independently sampled masks
        - Does not depend on torchaudio; uses pure tensor operations
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 1,
        num_time_masks: int = 1,
    ) -> None:
        super().__init__()

        if freq_mask_param < 0:
            raise ValueError(f"freq_mask_param must be >= 0, got {freq_mask_param}")
        if time_mask_param < 0:
            raise ValueError(f"time_mask_param must be >= 0, got {time_mask_param}")
        if num_freq_masks < 0:
            raise ValueError(f"num_freq_masks must be >= 0, got {num_freq_masks}")
        if num_time_masks < 0:
            raise ValueError(f"num_time_masks must be >= 0, got {num_time_masks}")

        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment masking.

        Args:
            input: Input tensor [batch, freq, time] or [batch, channels, freq, time]

        Returns:
            Output tensor with frequency and time masks applied if training
        """
        if not self.training:
            return input

        if input.ndim == 3:
            # [batch, freq, time]
            freq_dim = 1
            time_dim = 2
        elif input.ndim == 4:
            # [batch, channels, freq, time]
            freq_dim = 2
            time_dim = 3
        else:
            raise ValueError(
                f"SpecAugment expects 3D or 4D input, got {input.ndim}D"
            )

        output = input.clone()
        batch_size = output.shape[0]
        freq_size = output.shape[freq_dim]
        time_size = output.shape[time_dim]

        # Apply masks independently per batch item (per-utterance augmentation)
        for b in range(batch_size):
            # Apply frequency masks
            for _ in range(self.num_freq_masks):
                f = int(torch.randint(0, self.freq_mask_param + 1, (1,)).item())
                f = min(f, freq_size)
                if f == 0:
                    continue
                f0 = int(torch.randint(0, max(freq_size - f, 1), (1,)).item())
                if input.ndim == 3:
                    output[b, f0 : f0 + f, :] = 0.0
                else:
                    output[b, :, f0 : f0 + f, :] = 0.0

            # Apply time masks
            for _ in range(self.num_time_masks):
                t = int(torch.randint(0, self.time_mask_param + 1, (1,)).item())
                t = min(t, time_size)
                if t == 0:
                    continue
                t0 = int(torch.randint(0, max(time_size - t, 1), (1,)).item())
                if input.ndim == 3:
                    output[b, :, t0 : t0 + t] = 0.0
                else:
                    output[b, :, :, t0 : t0 + t] = 0.0

        return output

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"freq_mask_param={self.freq_mask_param}, "
            f"time_mask_param={self.time_mask_param}, "
            f"num_freq_masks={self.num_freq_masks}, "
            f"num_time_masks={self.num_time_masks}"
        )
