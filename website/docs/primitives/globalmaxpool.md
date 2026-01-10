---
sidebar_label: GlobalMaxPool
---

# GlobalMaxPool

Global Max Pooling

Reduces each channel to a single value by taking the maximum across
all spatial positions.

Shape Contract:
- Input: [batch, channels, height, width] (any spatial size)
- Output: [batch, channels, 1, 1] single value per channel

Notes:
- Equivalent to AdaptiveMaxPool(1)
- Reduces spatial dimensions to 1x1
- No learnable parameters
- Preserves strongest activation in each channel
- Works with any input resolution
- Less common than GlobalAvgPool but useful for detection tasks

## Signature

```neuroscript
neuron GlobalMaxPool()
```

## Ports

**Inputs:**
- `default`: `[batch, channels, *, *]`

**Outputs:**
- `default`: `[batch, channels, 1, 1]`

## Implementation

```
"from core import pooling/GlobalMaxPool"
```

```
Source { source: "core", path: "pooling/GlobalMaxPool" }
```

