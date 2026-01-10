---
sidebar_label: Concat
---

# Concat

Concatenate

Joins two or more tensors along an existing axis. All dimensions except
the concatenation dimension must match.

Parameters:
- dim: Dimension along which to concatenate (0-indexed)

Shape Contract:
- Input a: [*shape_a] first tensor
- Input b: [*shape_b] second tensor
- Output: [*shape_out] concatenated result

Notes:
- All dimensions except dim must be equal
- Example: [2, 3, 4] + [2, 5, 4] along dim=1 gives [2, 8, 4]
- No learnable parameters (pure structural operation)

## Signature

```neuroscript
neuron Concat(dim)
```

## Ports

**Inputs:**
- `a`: `[*shape_a]`
- `b`: `[*shape_b]`

**Outputs:**
- `default`: `[*shape_out]`

## Implementation

```
"from core import structural/Concat"
```

```
Source { source: "core", path: "structural/Concat" }
```

