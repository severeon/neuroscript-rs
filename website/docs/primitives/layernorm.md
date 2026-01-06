---
sidebar_label: LayerNorm
---

# LayerNorm

Layer normalization

Normalizes inputs across the feature dimension, computing mean and
variance over the last dimension. Essential component in transformer architectures.

## Signature

```neuroscript
neuron LayerNorm(dim)
```

## Parameters

- dim: Normalization dimension (size of the feature axis)

## Shape Contract

- Input: [*shape, dim] where dim is the normalized dimension
- Output: [*shape, dim] same shape as input

## Ports

**Inputs:**
- `default`: `[*shape, dim]`

**Outputs:**
- `default`: `[*shape, dim]`

## Example

```neuroscript
neuron TransformerBlock(dim):
graph:
in -> LayerNorm(dim) -> Attention(dim) -> out
```

## Notes

- Normalizes over last dimension only (per-example normalization)
- Includes learnable scale (γ) and shift (β) parameters
- More stable than BatchNorm for variable-length sequences
- Standard in transformers (BERT, GPT, etc.)

## Implementation

```
Source { source: "core", path: "normalization/LayerNorm" }
```

