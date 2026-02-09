# DilatedConv Neurons

This file contains two neurons related to dilated (atrous) convolutions:

## 1. DilatedConv (Primitive)

A simple wrapper around Conv2d that emphasizes the dilation parameter for creating atrous convolutions.

**Purpose**: Increase receptive field without increasing parameter count

**Parameters**:
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels
- `kernel_size`: Size of the convolving kernel
- `dilation`: Spacing between kernel elements (the key parameter!)
- `stride`: Stride of convolution (default: 1)
- `padding`: Zero-padding added to input (default: 0)

**Shape Contract**:
- Input: `[batch, in_channels, height, width]`
- Output: `[batch, out_channels, out_height, out_width]`

**Example Usage**:
```neuroscript
# Dilation rate 2 with 3x3 kernel = effective 5x5 receptive field
conv = DilatedConv(64, 64, kernel_size=3, dilation=2, padding=2)

# Dilation rate 4 with 3x3 kernel = effective 9x9 receptive field
conv = DilatedConv(64, 64, kernel_size=3, dilation=4, padding=4)
```

**Common Use Cases**:
- Semantic segmentation (DeepLab, PSPNet)
- Audio processing with WaveNet
- Capturing multi-scale context without downsampling

## 2. ASPPBlock (Composite)

Atrous Spatial Pyramid Pooling block that captures context at multiple scales using parallel dilated convolutions.

**Architecture**:
```
       input [batch, C_in, H, W]
          |
    +-----+-----+
    |     |     |
  rate6 rate12 rate18
    |     |     |
    +-----+-----+
          |
       concat
          |
   [batch, C_out * 3, H, W]
```

**Parameters**:
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels per branch

**Shape Contract**:
- Input: `[batch, in_channels, height, width]`
- Output: `[batch, out_channels * 3, height, width]`

**Design Details**:
- Three parallel branches with dilation rates: 6, 12, 18
- Each branch uses 3x3 convolution
- Padding matches dilation to preserve spatial dimensions
- Output channels are 3x the specified out_channels (one set per branch)

**Example Usage**:
```neuroscript
# Create ASPP block for semantic segmentation backbone
aspp = ASPPBlock(in_channels=256, out_channels=64)
# Input: [2, 256, 32, 32] → Output: [2, 192, 32, 32]
```

## Receptive Field Comparison

With a 3x3 kernel:

| Dilation | Effective Receptive Field | Use Case |
|----------|--------------------------|----------|
| 1        | 3×3                      | Standard convolution |
| 2        | 5×5                      | Slightly larger context |
| 4        | 9×9                      | Medium context |
| 6        | 13×13                    | Large context (ASPP) |
| 12       | 25×25                    | Very large context (ASPP) |
| 18       | 37×37                    | Extra large context (ASPP) |

## Testing

Run the test suite:
```bash
source ~/.venv_ai/bin/activate
python test_dilated_conv.py
```

## Validation

The neurons have been validated through all compilation stages:
```bash
✓ Parse successful
✓ Validation successful
✓ Compilation successful
✓ PyTorch forward pass works correctly
```

## References

- **DeepLab**: Introduced atrous convolutions for semantic segmentation
- **DeepLab v2**: Introduced ASPP (Atrous Spatial Pyramid Pooling)
- **WaveNet**: Used dilated convolutions for audio generation with exponentially growing receptive fields

## Implementation Notes

- The ASPP block uses pairwise concatenation (Concat takes 2 inputs at a time)
- Padding is set to match dilation rate to preserve spatial dimensions
- The primitive delegates to PyTorch's Conv2d with the dilation parameter
- All tests pass with correct shape propagation
