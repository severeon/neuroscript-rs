---
sidebar_label: Flatten
---

# Flatten

Flatten

Flattens dimensions from start_dim to end_dim (inclusive) into a single dimension.
Commonly used to transition from convolutional layers to fully-connected layers.

Parameters:
- start_dim: First dimension to flatten (default: 1, preserving batch)
- end_dim: Last dimension to flatten (default: -1, all remaining dims)

Shape Contract:
- Input: [*shape_in] arbitrary input shape
- Output: [*shape_out] flattened output (depends on start_dim and end_dim)

Notes:
- Default behavior: Flatten(1, -1) preserves batch dimension
- Example: [32, 3, 224, 224] with default params gives [32, 150528]
- No learnable parameters (pure reshaping operation)

## Signature

```neuroscript
neuron Flatten(start_dim=Int(1), end_dim=BinOp { op: Sub, left: Int(0), right: Int(1) })
```

## Ports

**Inputs:**
- `default`: `[*shape_in]`

**Outputs:**
- `default`: `[*shape_out]`

## Implementation

```
Source { source: "core", path: "structural/Flatten" }
```

