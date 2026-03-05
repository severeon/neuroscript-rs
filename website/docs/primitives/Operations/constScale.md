---
sidebar_label: ConstScale
---

# ConstScale

Multiplies input by a fixed **non-learnable** scalar factor. A pure constant
scaling operation with no trainable parameters.

Parameters:
- factor: The constant scalar multiplier

Shape Contract:
- Input: [*shape] any tensor
- Output: [*shape] same shape, scaled by factor

Notes:
- **No learnable parameters** — purely multiplies by a constant scalar
- Applied element-wise: out = input * factor
- Common use: half-step FFN in Conformer (factor=0.5)
- Useful for constant scaling in residual connections

## Signature

```neuroscript
neuron ConstScale(factor)
```

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Implementation

```
Source { source: "core", path: "operations/ConstScale" }
```
