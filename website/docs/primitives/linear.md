---
sidebar_label: Linear
---

# Linear

Fully-connected linear transformation layer

Applies a linear transformation to the incoming data: y = xW^T + b
This is a shape-aware wrapper around torch.nn.Linear that validates
tensor dimensions according to NeuroScript's shape algebra.

## Signature

```neuroscript
neuron Linear(in_dim, out_dim)
```

## Parameters

- in_dim: Size of each input sample (last dimension)
- out_dim: Size of each output sample (last dimension)

## Shape Contract

- Input: [*, in_dim] where * means any number of leading dimensions
- Output: [*, out_dim] with same leading dimensions as input

## Ports

**Inputs:**
- `default`: `[*, in_dim]`

**Outputs:**
- `default`: `[*, out_dim]`

## Example

```neuroscript
neuron MLP:
graph:
in -> Linear(512, 256) -> out
```

## Notes

- Preserves all leading dimensions (broadcasting-compatible)
- Weight shape: [out_dim, in_dim]
- Includes learnable bias by default

## Implementation

```
External { kwargs: [] }
```

