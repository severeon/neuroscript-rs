---
sidebar_label: Conv1d
---

# Conv1d

1D Convolutional Layer

Applies 1D convolution over an input signal (sequences, time series, audio).

Parameters:
- in_channels: Number of channels in input
- out_channels: Number of channels produced by convolution
- kernel_size: Size of the convolving kernel
- stride: Stride of the convolution (default: 1)
- padding: Zero-padding added to both sides (default: 0)
- dilation: Spacing between kernel elements (default: 1)
- groups: Number of blocked connections (default: 1)
- bias: If true, adds learnable bias (default: true)

Shape Contract:
- Input: [batch, in_channels, length]
- Output: [batch, out_channels, out_length]

Notes:
- Output length: (length + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
- Used for sequence modeling, audio processing, time series
- WaveNet, TCN, and other temporal architectures

## Signature

```neuroscript
neuron Conv1d(in_channels, out_channels, kernel_size, stride=Int(1), padding=Int(0), dilation=Int(1), groups=Int(1), bias=Bool(true))
```

## Ports

**Inputs:**
- `default`: `[batch, in_channels, length]`

**Outputs:**
- `default`: `[batch, out_channels, *]`

## Implementation

```
"from core import convolutions/Conv1d"
```

```
Source { source: "core", path: "convolutions/Conv1d" }
```

