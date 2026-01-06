---
sidebar_label: ReLU
---

# ReLU

Rectified Linear Unit (ReLU)

Simple non-linear activation that outputs max(0, x). One of the most
widely used activation functions in deep learning due to its simplicity
and effectiveness.

Shape Contract:
- Input: [*shape] arbitrary shape
- Output: [*shape] same shape as input

Notes:
- Element-wise operation: ReLU(x) = max(0, x)
- Non-differentiable at x=0 (subgradient is typically used)
- Can suffer from dying ReLU problem (neurons output 0 for all inputs)
- Fast to compute, no vanishing gradient for positive inputs

## Signature

```neuroscript
neuron ReLU()
```

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Implementation

```
Source { source: "core", path: "activations/ReLU" }
```

