---
sidebar_label: DepthwiseConv
---

# DepthwiseConv

Depthwise Convolution

Applies separate convolution to each input channel independently.
Each channel is convolved with its own set of filters.

Parameters:
- channels: Number of input/output channels (same for depthwise)
- kernel_size: Size of the convolving kernel
- stride: Stride of the convolution (default: 1)
- padding: Zero-padding added to both sides (default: 0)
- dilation: Spacing between kernel elements (default: 1)
- bias: If true, adds learnable bias (default: true)

Shape Contract:
- Input: [batch, channels, height, width]
- Output: [batch, channels, out_height, out_width]

Notes:
- Equivalent to Conv2d with groups=in_channels
- Much fewer parameters than standard conv: kernel_size^2 * channels
- Used in MobileNet, EfficientNet for efficient computation
- Often followed by pointwise (1x1) convolution

## Signature

```neuroscript
neuron DepthwiseConv(channels, kernel_size, stride=Int(1), padding=Int(0), dilation=Int(1), bias=Bool(true))
```

## Ports

**Inputs:**
- `default`: `[batch, channels, height, width]`

**Outputs:**
- `default`: `[batch, channels, *, *]`

## Implementation

```
"from core import convolutions/DepthwiseConv"
```

```
Source { source: "core", path: "convolutions/DepthwiseConv" }
```

