---
sidebar_label: Concat
---

# Concat

Concatenate tensors along a specified dimension

Joins two or more tensors along an existing axis. All dimensions except
the concatenation dimension must match.

## Signature

```neuroscript
neuron Concat(dim)
```

## Parameters

- dim: Dimension along which to concatenate (0-indexed)

## Shape Contract

- Input a: [*shape_a] first tensor
- Input b: [*shape_b] second tensor
- Output: [*shape_out] concatenated result

## Ports

**Inputs:**
- `a`: `[*shape_a]`
- `b`: `[*shape_b]`

**Outputs:**
- `default`: `[*shape_out]`

## Example

```neuroscript
neuron ResidualConcat(dim):
graph:
in -> Fork() -> (main, skip)
main -> Linear(dim, dim) -> processed
(processed, skip) -> Concat(dim=-1) -> out
```

## Notes

- All dimensions except dim must be equal
- Example: [2, 3, 4] + [2, 5, 4] along dim=1 → [2, 8, 4]
- No learnable parameters (pure structural operation)

## Implementation

```
Source { source: "core", path: "structural/Concat" }
```

