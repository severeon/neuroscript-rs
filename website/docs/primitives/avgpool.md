---
sidebar_label: AvgPool
---

# AvgPool

2D Average Pooling

Downsamples by computing the average value in each pooling window.
Provides smoother downsampling compared to max pooling.

Parameters:
- kernel_size: Size of the pooling window
- stride: Stride of the pooling window (default: 1)
- padding: Zero-padding added to input (default: 0)

Shape Contract:
- Input: [batch, channels, height, width]
- Output: [batch, channels, out_height, out_width]

Notes:
- Output size: floor((input + 2*padding - kernel_size) / stride) + 1
- Smoother than MaxPool, preserves more spatial information
- Used in ResNet final pooling and some attention mechanisms
- No learnable parameters
- Preserves channel count

## Signature

```neuroscript
neuron AvgPool(kernel_size, stride=Int(1), padding=Int(0))
```

## Ports

**Inputs:**
- `default`: `[batch, channels, height, width]`

**Outputs:**
- `default`: `[batch, channels, *, *]`

## Implementation

```
"from core import pooling/AvgPool"
```

```
Source { source: "core", path: "pooling/AvgPool" }
```

