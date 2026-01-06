---
sidebar_label: Conv2d
---

# Conv2d

2D Convolutional Layer

Applies a 2D convolution over an input signal composed of several input planes.
Fundamental building block for computer vision models (CNNs).

Parameters:
- in_channels: Number of channels in the input image
- out_channels: Number of channels produced by the convolution
- kernel_size: Size of the convolving kernel (int or (height, width))
- stride: Stride of the convolution (default: 1)
- padding: Zero-padding added to both sides (default: 0)
- dilation: Spacing between kernel elements (default: 1)
- groups: Number of blocked connections (default: 1)
- bias: If true, adds learnable bias (default: true)

Shape Contract:
- Input: [batch, in_channels, height, width]
- Output: [batch, out_channels, out_height, out_width]

Notes:
- Output spatial size: (input_size + 2*padding - kernel_size) / stride + 1
- Common patterns: 3x3 kernel, stride=1, padding=1 (preserves size)
- Depthwise: groups=in_channels, separable convolutions

## Signature

```neuroscript
neuron Conv2d(in_channels, out_channels, kernel_size, stride=Int(1), padding=Int(0), dilation=Int(1), groups=Int(1), bias=Bool(true))
```

## Ports

**Inputs:**
- `default`: `[batch, in_channels, height, width]`

**Outputs:**
- `default`: `[batch, out_channels, *, *]`

## Implementation

```
Source { source: "core", path: "convolutions/Conv2d" }
```

