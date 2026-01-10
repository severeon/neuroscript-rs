---
sidebar_label: SeparableConv
---

# SeparableConv

Separable Convolution (Depthwise Separable)

Factorizes standard convolution into depthwise and pointwise operations.
Depthwise: per-channel spatial convolution. Pointwise: 1x1 channel mixing.

Parameters:
- in_channels: Number of input channels
- out_channels: Number of output channels
- kernel_size: Size of the depthwise kernel
- stride: Stride of the convolution (default: 1)
- padding: Zero-padding added to both sides (default: 0)
- dilation: Spacing between kernel elements (default: 1)
- bias: If true, adds learnable bias (default: true)

Shape Contract:
- Input: [batch, in_channels, height, width]
- Output: [batch, out_channels, out_height, out_width]

Notes:
- Reduces computation: from kernel^2 * in * out to kernel^2 * in + in * out
- Core building block of MobileNet, Xception, EfficientNet
- Combines DepthwiseConv + 1x1 Conv (pointwise)
- Similar accuracy to standard conv with far fewer parameters

## Signature

```neuroscript
neuron SeparableConv(in_channels, out_channels, kernel_size, stride=Int(1), padding=Int(0), dilation=Int(1), bias=Bool(true))
```

## Ports

**Inputs:**
- `default`: `[batch, in_channels, height, width]`

**Outputs:**
- `default`: `[batch, out_channels, *, *]`

## Implementation

```
"from core import convolutions/SeparableConv"
```

```
Source { source: "core", path: "convolutions/SeparableConv" }
```

