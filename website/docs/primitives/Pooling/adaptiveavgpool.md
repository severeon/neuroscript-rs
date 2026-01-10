---
sidebar_label: AdaptiveAvgPool
---

# AdaptiveAvgPool

Adaptive Average Pooling

Pools input to a fixed output size regardless of input dimensions.
Automatically calculates kernel size and stride to achieve target output.

Parameters:
- output_size: Target spatial size (height and width will both be this value)

Shape Contract:
- Input: [batch, channels, height, width] (any spatial size)
- Output: [batch, channels, output_size, output_size]

Notes:
- Accepts any input spatial size, outputs fixed size
- output_size=1 is common for global average pooling
- Used before fully-connected layers to handle variable input sizes
- No learnable parameters
- Enables architectures that work with multiple resolutions

## Signature

```neuroscript
neuron AdaptiveAvgPool(output_size)
```

## Ports

**Inputs:**
- `default`: `[batch, channels, *, *]`

**Outputs:**
- `default`: `[batch, channels, output_size, output_size]`

## Implementation

```
Source { source: "core", path: "pooling/AdaptiveAvgPool" }
```

