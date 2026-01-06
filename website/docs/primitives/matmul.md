---
sidebar_label: MatMul
---

# MatMul

MatMul

Matrix multiplication of two input tensors. Performs batched matrix
multiplication following PyTorch broadcasting rules.

Shape Contract:
- Input a: [*, n, m] first matrix
- Input b: [*, m, p] second matrix (inner dimension must match)
- Output: [*, n, p] matrix product

Notes:
- No learnable parameters (pure operation)
- Supports batched operations with broadcasting
- Inner dimensions must match: a.shape[-1] == b.shape[-2]
- Core operation for attention mechanisms
- Uses torch.matmul for optimal performance

## Signature

```neuroscript
neuron MatMul()
```

## Ports

**Inputs:**
- `a`: `[*, n, m]`
- `b`: `[*, m, p]`

**Outputs:**
- `default`: `[*, n, p]`

## Implementation

```
Source { source: "core", path: "operations/MatMul" }
```

