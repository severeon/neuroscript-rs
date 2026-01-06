---
sidebar_label: TransposedConv
---

# TransposedConv

Transposed Convolution (Deconvolution)

Upsampling convolution that increases spatial dimensions.
Also known as deconvolution or fractionally-strided convolution.

Parameters:
- in_channels: Number of input channels
- out_channels: Number of output channels
- kernel_size: Size of the convolving kernel
- stride: Stride of the convolution (default: 1)
- padding: Zero-padding added to both sides (default: 0)
- output_padding: Additional size added to output (default: 0)
- dilation: Spacing between kernel elements (default: 1)
- groups: Number of blocked connections (default: 1)
- bias: If true, adds learnable bias (default: true)

Shape Contract:
- Input: [batch, in_channels, height, width]
- Output: [batch, out_channels, out_height, out_width]

Notes:
- Output size: (input - 1) * stride - 2*padding + kernel_size + output_padding
- Used in decoder networks, GANs, segmentation
- Learnable upsampling (vs interpolation)
- Can cause checkerboard artifacts if not used carefully

## Signature

```neuroscript
neuron TransposedConv(in_channels, out_channels, kernel_size, stride=Int(1), padding=Int(0), output_padding=Int(0), dilation=Int(1), groups=Int(1), bias=Bool(true))
```

## Ports

**Inputs:**
- `default`: `[batch, in_channels, height, width]`

**Outputs:**
- `default`: `[batch, out_channels, *, *]`

## Implementation

```
Source { source: "core", path: "convolutions/TransposedConv" }
```

