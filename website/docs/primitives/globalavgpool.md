---
sidebar_label: GlobalAvgPool
---

# GlobalAvgPool

Global Average Pooling

Reduces each channel to a single value by averaging all spatial positions.
Commonly used as the final pooling before classification head.

Shape Contract:
- Input: [batch, channels, height, width] (any spatial size)
- Output: [batch, channels, 1, 1] single value per channel

Notes:
- Equivalent to AdaptiveAvgPool(1)
- Reduces spatial dimensions to 1x1
- No learnable parameters
- Reduces overfitting compared to large fully-connected layers
- Works with any input resolution
- Standard in ResNet, EfficientNet, and modern CNNs

## Signature

```neuroscript
neuron GlobalAvgPool()
```

## Ports

**Inputs:**
- `default`: `[batch, channels, *, *]`

**Outputs:**
- `default`: `[batch, channels, 1, 1]`

## Implementation

```
"from core import pooling/GlobalAvgPool"
```

```
Source { source: "core", path: "pooling/GlobalAvgPool" }
```

