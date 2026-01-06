---
sidebar_label: MaxPool
---

# MaxPool

2D Max Pooling

Downsamples by taking the maximum value in each pooling window.
Commonly used in CNNs to reduce spatial dimensions while preserving
the strongest activations.

Parameters:
- kernel_size: Size of the pooling window
- stride: Stride of the pooling window (default: 1)
- padding: Zero-padding added to input (default: 0)
- dilation: Spacing between kernel elements (default: 1)

Shape Contract:
- Input: [batch, channels, height, width]
- Output: [batch, channels, out_height, out_width]

Notes:
- Output size: floor((input + 2*padding - kernel_size) / stride) + 1
- Common pattern: kernel_size=2, stride=2 (halves spatial dimensions)
- Provides translation invariance
- No learnable parameters
- Preserves channel count

## Signature

```neuroscript
neuron MaxPool(kernel_size, stride=Int(1), padding=Int(0), dilation=Int(1))
```

## Ports

**Inputs:**
- `default`: `[batch, channels, height, width]`

**Outputs:**
- `default`: `[batch, channels, *, *]`

## Implementation

```
Source { source: "core", path: "pooling/MaxPool" }
```

