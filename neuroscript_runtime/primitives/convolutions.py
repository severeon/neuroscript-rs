"""Convolutional layer primitives for NeuroScript."""

import torch
import torch.nn as nn
from typing import Union, Tuple

_size_1_t = Union[int, Tuple[int]]
_size_2_t = Union[int, Tuple[int, int]]
_size_3_t = Union[int, Tuple[int, int, int]]


class Conv1d(nn.Module):
    """1D convolution for sequences."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class Conv2d(nn.Module):
    """2D convolution for images."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class Conv3d(nn.Module):
    """3D convolution for volumetric data."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class DepthwiseConv(nn.Module):
    """Depthwise convolution - each channel convolved independently."""

    def __init__(
        self,
        channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class SeparableConv(nn.Module):
    """Separable convolution = depthwise + pointwise (1x1)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(input))


class TransposedConv(nn.Module):
    """Transposed convolution (deconvolution) for upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class DilatedConv(nn.Module):
    """Dilated (atrous) convolution for expanded receptive fields.

    A 2D convolution where the kernel is spread out by inserting gaps (dilation)
    between kernel elements. Increases receptive field without increasing parameters.
    Reference: Yu & Koltun (2016), "Multi-Scale Context Aggregation by Dilated Convolutions"

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        dilation: Dilation rate (spacing between kernel elements). Default: 2
        padding: Input padding. Default: 0
        stride: Convolution stride. Default: 1
        bias: If True, adds learnable bias. Default: True

    Shape:
        - Input: [batch, in_channels, height, width]
        - Output: [batch, out_channels, height', width']

    Examples:
        >>> conv = DilatedConv(64, 128, kernel_size=3, dilation=2, padding=2)
        >>> x = torch.randn(32, 64, 28, 28)
        >>> out = conv(x)
        >>> assert out.shape[1] == 128

    Notes:
        - Dilation=1 is equivalent to standard convolution
        - Effective kernel size = kernel_size + (kernel_size - 1) * (dilation - 1)
        - Commonly used in DeepLab, WaveNet, and semantic segmentation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        dilation: _size_2_t = 2,
        padding: _size_2_t = 0,
        stride: _size_2_t = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"in_channels={self.conv.in_channels}, out_channels={self.conv.out_channels}, "
            f"kernel_size={self.conv.kernel_size}, dilation={self.conv.dilation}, "
            f"stride={self.conv.stride}, padding={self.conv.padding}, "
            f"bias={self.conv.bias is not None}"
        )


__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "DepthwiseConv",
    "DilatedConv",
    "SeparableConv",
    "TransposedConv",
]
