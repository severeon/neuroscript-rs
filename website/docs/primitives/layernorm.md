---
sidebar_label: LayerNorm
---

# LayerNorm

Layer Normalization

Normalizes inputs across the feature dimension, computing mean and
variance over the last dimension. Essential component in transformer architectures.

Parameters:
- dim: Normalization dimension (size of the feature axis)

Shape Contract:
- Input: [*shape, dim] where dim is the normalized dimension
- Output: [*shape, dim] same shape as input

Notes:
- Normalizes over last dimension only (per-example normalization)
- Includes learnable scale (gamma) and shift (beta) parameters
- More stable than BatchNorm for variable-length sequences
- Standard in transformers (BERT, GPT, etc.)

## Signature

```neuroscript
neuron LayerNorm(dim)
```

## Ports

**Inputs:**
- `default`: `[*shape, dim]`

**Outputs:**
- `default`: `[*shape, dim]`

## Implementation

```
Source { source: "core", path: "normalization/LayerNorm" }
```

