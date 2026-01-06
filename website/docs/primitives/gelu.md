---
sidebar_label: GELU
---

# GELU

Gaussian Error Linear Unit (GELU) activation function

A smooth approximation of ReLU that's commonly used in transformer models.
GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function
of the standard Gaussian distribution.

## Signature

```neuroscript
neuron GELU()
```

## Shape Contract

- Input: [*shape] arbitrary shape
- Output: [*shape] same shape as input

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Example

```neuroscript
neuron FFN(dim):
graph:
in -> Linear(dim, dim * 4) -> GELU() -> Linear(dim * 4, dim) -> out
```

## Notes

- Used in BERT, GPT-2, and many modern transformers
- Smoother than ReLU, which can help with gradient flow
- Element-wise operation (preserves all dimensions)

## Implementation

```
Source { source: "core", path: "activations/GELU" }
```

