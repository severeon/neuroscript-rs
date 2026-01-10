---
sidebar_label: Reshape
---

# Reshape

Reshape

Changes the shape of a tensor without changing its data.
Total element count must remain the same. Use -1 for one dimension to infer its size.

Parameters:
- target_shape: Tuple specifying the new shape (can include -1 for inference)

Shape Contract:
- Input: [*shape_in] tensor with original shape
- Output: [*shape_out] tensor with new shape (same total elements)

Notes:
- No learnable parameters (pure structural operation)
- Total elements must be preserved: prod(shape_in) == prod(shape_out)
- Use -1 for at most one dimension to auto-calculate
- Common use: flattening for FC layers, splitting/merging attention heads
- Data is not copied, only the view changes (memory efficient)

## Signature

```neuroscript
neuron Reshape(target_shape)
```

## Ports

**Inputs:**
- `default`: `[*shape_in]`

**Outputs:**
- `default`: `[*shape_out]`

## Implementation

```
Source { source: "core", path: "structural/Reshape" }
```

