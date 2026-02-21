---
sidebar_label: ConstScale
---

# ConstScale

ConstScale

Multiplies input by a **non-learnable** scale vector. Element-wise multiplicative
scaling applied along the last dimension.

Parameters:
- dim: Size of the scale vector (must match last dimension of input)

Shape Contract:
- Input: [*, dim] tensor with last dimension matching scale size
- Output: [*, dim] same shape as input

Notes:
- **Non-Learnable** parameter: scale vector of shape [dim], initialized to 1
- Applied element-wise: out = input * scale
- Used in normalization layers (gamma parameter)
- Can be used for learned feature weighting

## Signature

```neuroscript
neuron ConstScale(dim)
```

## Ports

**Inputs:**
- `default`: `[*, dim]`

**Outputs:**
- `default`: `[*, dim]`

## Implementation

```
Source { source: "core", path: "operations/ConstScale" }
```
