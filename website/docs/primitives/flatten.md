---
sidebar_label: Flatten
---

# Flatten

Flatten tensor by collapsing contiguous dimensions

Flattens dimensions from start_dim to end_dim (inclusive) into a single dimension.
Commonly used to transition from convolutional layers to fully-connected layers.

## Signature

```neuroscript
neuron Flatten(start_dim=Int(1), end_dim=BinOp { op: Sub, left: Int(0), right: Int(1) })
```

## Parameters

- start_dim: First dimension to flatten (default: 1, preserving batch)
- end_dim: Last dimension to flatten (default: -1, all remaining dims)

## Shape Contract

- Input: [*shape_in] arbitrary input shape
- Output: [*shape_out] flattened output (depends on start_dim and end_dim)

## Ports

**Inputs:**
- `default`: `[*shape_in]`

**Outputs:**
- `default`: `[*shape_out]`

## Example

```neuroscript
neuron CNNClassifier:
graph:
in -> Conv2d(3, 64) -> Flatten() -> Linear(64 * 7 * 7, 10) -> out
```

## Notes

- Default behavior: Flatten(1, -1) preserves batch dimension
- Example: [32, 3, 224, 224] → [32, 150528] with default params
- No learnable parameters (pure reshaping operation)

## Implementation

```
Source { source: "core", path: "structural/Flatten" }
```

