---
sidebar_label: ELU
---

# ELU

ELU (Exponential Linear Unit)

Smooth activation with negative saturation.
ELU(x) = x if x > 0, else alpha * (exp(x) - 1)

Parameters:
- alpha: Scale for negative part (default: 1.0)

Shape Contract:
- Input: [*shape] arbitrary shape
- Output: [*shape] same shape as input

Notes:
- Smooth everywhere, including at x=0
- Negative values saturate to -alpha
- Reduces bias shift (mean activations closer to zero)
- More expensive than ReLU due to exp computation
- Used in some image classification architectures
- Element-wise operation (preserves all dimensions)

## Signature

```neuroscript
neuron ELU(alpha=Float(1.0))
```

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Implementation

```
Source { source: "core", path: "activations/ELU" }
```

