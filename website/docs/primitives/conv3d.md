---
sidebar_label: Conv3d
---

# Conv3d

3D Convolutional Layer

Applies 3D convolution over volumetric data (video, 3D medical imaging).

Parameters:
- in_channels: Number of channels in input
- out_channels: Number of channels produced by convolution
- kernel_size: Size of the convolving kernel (int or (d, h, w))
- stride: Stride of the convolution (default: 1)
- padding: Zero-padding added to all sides (default: 0)
- dilation: Spacing between kernel elements (default: 1)
- groups: Number of blocked connections (default: 1)
- bias: If true, adds learnable bias (default: true)

Shape Contract:
- Input: [batch, in_channels, depth, height, width]
- Output: [batch, out_channels, out_depth, out_height, out_width]

Notes:
- Output size per dim: (size + 2*padding - dilation*(kernel-1) - 1) / stride + 1
- Used for video understanding, 3D medical imaging, point clouds
- Memory intensive due to 5D tensors

## Signature

```neuroscript
neuron Conv3d(in_channels, out_channels, kernel_size, stride=Int(1), padding=Int(0), dilation=Int(1), groups=Int(1), bias=Bool(true))
```

## Ports

**Inputs:**
- `default`: `[batch, in_channels, depth, height, width]`

**Outputs:**
- `default`: `[batch, out_channels, *, *, *]`

## Implementation

```
Source { source: "core", path: "convolutions/Conv3d" }
```

