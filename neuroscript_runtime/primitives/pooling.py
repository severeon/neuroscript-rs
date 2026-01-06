"""Pooling layer primitives for NeuroScript."""

import torch
import torch.nn as nn
from typing import Union, Tuple

_size_2_t = Union[int, Tuple[int, int]]


class MaxPool(nn.Module):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(
            kernel_size, stride=stride, padding=padding, dilation=dilation
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.pool(input)


class AvgPool(nn.Module):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
    ) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.pool(input)


class AdaptiveAvgPool(nn.Module):
    def __init__(self, output_size: _size_2_t) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.pool(input)


class AdaptiveMaxPool(nn.Module):
    def __init__(self, output_size: _size_2_t) -> None:
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.pool(input)


class GlobalAvgPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.pool(input)


class GlobalMaxPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.pool(input)


__all__ = [
    "MaxPool",
    "AvgPool",
    "AdaptiveAvgPool",
    "AdaptiveMaxPool",
    "GlobalAvgPool",
    "GlobalMaxPool",
]
