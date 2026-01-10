---
sidebar_label: Transpose
---

# Transpose

Transpose

Permutes the dimensions of a tensor according to the specified ordering.
Essential for reshaping data between different layer expectations.

Parameters:
- dims: Tuple/list specifying the new dimension order (e.g., (0, 2, 1))

Shape Contract:
- Input: [*shape_in] tensor with dimensions to permute
- Output: [*shape_out] tensor with reordered dimensions

Notes:
- No learnable parameters (pure structural operation)
- Common use: converting between channel-first and channel-last formats
- For attention: transposing keys for batched matrix multiplication
- Example: [batch, seq, dim] with dims=(0, 2, 1) gives [batch, dim, seq]
- All dimensions must be accounted for in the permutation

## Signature

```neuroscript
neuron Transpose(dims)
```

## Ports

**Inputs:**
- `default`: `[*shape_in]`

**Outputs:**
- `default`: `[*shape_out]`

## Implementation

```
Source { source: "core", path: "structural/Transpose" }
```

