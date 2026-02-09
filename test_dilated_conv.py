#!/usr/bin/env python3
"""Test script for DilatedConv and ASPPBlock neurons."""

import torch
import sys

# Add the project root to Python path
sys.path.insert(0, '/Users/tquick/projects/neuroscript-rs')

from neuroscript_runtime.primitives.convolutions import Conv2d
from neuroscript_runtime.primitives.structural import Concat, Fork3


class ASPPBlock(torch.nn.Module):
    """Atrous Spatial Pyramid Pooling Block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.concat_4 = Concat(1)
        self.concat_5 = Concat(1)
        self.conv2d_1 = Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv2d_2 = Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv2d_3 = Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.fork3_0 = Fork3()

    def forward(self, x):
        x0 = self.fork3_0(x)
        x1, x2, x3 = x0
        x4 = self.conv2d_1(x1)
        x5 = self.conv2d_2(x2)
        x6 = self.conv2d_3(x3)
        x7 = self.concat_4((x4, x5))
        x8 = self.concat_5((x7, x6))
        return x8


def test_dilated_conv():
    """Test basic dilated convolution."""
    print("Testing DilatedConv (via Conv2d with dilation)...")

    # Create a dilated conv: dilation=2, kernel_size=3, padding=2
    conv = Conv2d(64, 64, kernel_size=3, dilation=2, padding=2)

    # Test forward pass
    x = torch.randn(2, 64, 32, 32)  # [batch, channels, height, width]
    y = conv(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Expected:     torch.Size([2, 64, 32, 32])")

    assert y.shape == torch.Size([2, 64, 32, 32]), "Output shape mismatch!"
    print("  ✓ DilatedConv test passed!\n")


def test_aspp_block():
    """Test ASPP block with multiple dilation rates."""
    print("Testing ASPPBlock...")

    # Create ASPP block
    aspp = ASPPBlock(in_channels=256, out_channels=64)

    # Test forward pass
    x = torch.randn(2, 256, 32, 32)  # [batch, channels, height, width]
    y = aspp(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Expected:     torch.Size([2, 192, 32, 32])  (64 * 3 branches)")

    assert y.shape == torch.Size([2, 192, 32, 32]), "Output shape mismatch!"
    print("  ✓ ASPPBlock test passed!\n")


def test_receptive_field():
    """Demonstrate receptive field increase with dilation."""
    print("Demonstrating receptive field expansion:")
    print("  Standard 3x3 conv (dilation=1): receptive field = 3x3")
    print("  Dilated 3x3 conv (dilation=2):  receptive field = 5x5")
    print("  Dilated 3x3 conv (dilation=4):  receptive field = 9x9")
    print("  Dilated 3x3 conv (dilation=6):  receptive field = 13x13")
    print("  Dilated 3x3 conv (dilation=12): receptive field = 25x25")
    print("  Dilated 3x3 conv (dilation=18): receptive field = 37x37\n")


if __name__ == "__main__":
    print("=" * 60)
    print("DilatedConv and ASPPBlock Test Suite")
    print("=" * 60 + "\n")

    test_dilated_conv()
    test_aspp_block()
    test_receptive_field()

    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
