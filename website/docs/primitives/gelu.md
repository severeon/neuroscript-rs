---
sidebar_label: GELU
---

# GELU

Gaussian Error Linear Unit (GELU)

A smooth approximation of ReLU commonly used in transformer models.
GELU(x) = x * CDF(x) where CDF is the cumulative distribution function
of the standard Gaussian distribution.

Shape Contract:
- Input: [*shape] arbitrary shape
- Output: [*shape] same shape as input

Notes:
- Used in BERT, GPT-2, and many modern transformers
- Smoother than ReLU, which can help with gradient flow
- Element-wise operation (preserves all dimensions)

## Signature

```neuroscript
neuron GELU()
```

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Implementation

```
"from core import activations/GELU"
```

```
Source { source: "core", path: "activations/GELU" }
```

