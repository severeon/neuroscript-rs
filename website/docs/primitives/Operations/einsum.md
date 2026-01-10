---
sidebar_label: Einsum
---

# Einsum

Einsum

Einstein summation for generalized tensor contractions. Expresses complex
tensor operations in a concise notation.

Parameters:
- equation: Einsum equation string (e.g., "bij,bjk->bik" for batched matmul)

Shape Contract:
- Input a: [*shape_a] first tensor
- Input b: [*shape_b] second tensor (optional, depends on equation)
- Output: [*shape_out] result shape determined by equation

Notes:
- No learnable parameters (pure operation)
- Supports 1-3 input tensors depending on equation
- Examples:
- "ij->ji" - transpose
- "bij,bjk->bik" - batched matrix multiply
- "bhqd,bhkd->bhqk" - attention scores
- Powerful but can be slower than specialized operations

## Signature

```neuroscript
neuron Einsum(equation)
```

## Ports

**Inputs:**
- `a`: `[*shape_a]`
- `b`: `[*shape_b]`

**Outputs:**
- `default`: `[*shape_out]`

## Implementation

```
Source { source: "core", path: "operations/Einsum" }
```

