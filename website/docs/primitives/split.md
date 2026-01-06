---
sidebar_label: Split
---

# Split

Split

Splits a tensor into multiple chunks along a specified dimension.

Parameters:
- num_splits: Number of equal-sized chunks to create
- dim: Dimension along which to split (default: -1)

Shape Contract:
- Input: [*shape] tensor to split
- Output a: [*shape_a] first chunk
- Output b: [*shape_b] second chunk (if num_splits >= 2)

Notes:
- Input size along dim must be divisible by num_splits
- Returns tuple of tensors
- No learnable parameters (pure structural operation)
- Inverse of Concat
- Commonly used to split heads in attention or features

## Signature

```neuroscript
neuron Split(num_splits, dim=BinOp { op: Sub, left: Int(0), right: Int(1) })
```

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `a`: `[*shape_a]`
- `b`: `[*shape_b]`

## Implementation

```
Source { source: "core", path: "structural/Split" }
```

