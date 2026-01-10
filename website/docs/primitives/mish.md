---
sidebar_label: Mish
---

# Mish

Mish Activation

Self-regularized non-monotonic activation function.
Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))

Shape Contract:
- Input: [*shape] arbitrary shape
- Output: [*shape] same shape as input

Notes:
- Smooth, non-monotonic activation
- Self-regularizing properties
- Used in YOLOv4 and other modern architectures
- Slightly more expensive than ReLU but often better performance
- No vanishing gradient for negative inputs (unlike ReLU)
- Element-wise operation (preserves all dimensions)

## Signature

```neuroscript
neuron Mish()
```

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Implementation

```
"from core import activations/Mish"
```

```
Source { source: "core", path: "activations/Mish" }
```

