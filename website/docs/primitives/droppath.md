---
sidebar_label: DropPath
---

# DropPath

DropPath Regularization (Stochastic Depth)

Randomly drops entire residual paths during training. Enables training
of very deep networks by making depth stochastic during training.

Parameters:
- drop_prob: Probability of dropping the entire path (0 to 1)

Shape Contract:
- Input: [*shape] arbitrary shape
- Output: [*shape] same shape as input

Notes:
- Drops entire layer outputs, not individual neurons
- When dropped, input passes through unchanged (via residual)
- Drop probability often increases linearly with depth
- Key technique for training Vision Transformers (ViT)
- Only active during training
- Used in DeiT, Swin Transformer, and EfficientNet
- Also known as Stochastic Depth

## Signature

```neuroscript
neuron DropPath(drop_prob)
```

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Implementation

```
"from core import regularization/DropPath"
```

```
Source { source: "core", path: "regularization/DropPath" }
```

