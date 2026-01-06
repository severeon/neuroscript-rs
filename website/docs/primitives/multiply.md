---
sidebar_label: Multiply
---

# Multiply

Element-wise Multiplication

Multiplies two tensors element-by-element. Fundamental operation for
gating mechanisms, attention weighting, and feature modulation.

Shape Contract:
- Input a: [*shape] first tensor
- Input b: [*shape] second tensor (must match shape of a)
- Output: [*shape] element-wise product

Notes:
- No learnable parameters (pure element-wise operation)
- Both inputs must have identical shapes
- Used in: gating (LSTM, GRU), attention weighting, feature scaling
- Common pattern: x * sigmoid(gate) for learned gating
- Supports broadcasting if shapes are compatible

## Signature

```neuroscript
neuron Multiply()
```

## Ports

**Inputs:**
- `a`: `[*shape]`
- `b`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Implementation

```
Source { source: "core", path: "structural/Multiply" }
```

