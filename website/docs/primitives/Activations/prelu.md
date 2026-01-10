---
sidebar_label: PReLU
---

# PReLU

PReLU (Parametric ReLU)

ReLU with learnable slope for negative values.
PReLU(x) = max(0, x) + a * min(0, x) where a is learnable.

Parameters:
- num_parameters: Number of learnable parameters (1 or num_features)
- init: Initial value of a (default: 0.25)

Shape Contract:
- Input: [*shape] arbitrary shape
- Output: [*shape] same shape as input

Notes:
- If num_parameters=1, single shared slope for all channels
- If num_parameters=channels, per-channel slopes
- Learnable parameter a typically initialized to 0.25
- Reduces dying ReLU problem
- Used in image classification networks
- Element-wise operation (preserves all dimensions)

## Signature

```neuroscript
neuron PReLU(num_parameters=Int(1), init=Float(0.25))
```

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Implementation

```
Source { source: "core", path: "activations/PReLU" }
```

