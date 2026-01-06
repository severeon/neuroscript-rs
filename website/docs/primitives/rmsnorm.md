---
sidebar_label: RMSNorm
---

# RMSNorm

RMS Normalization

Root Mean Square Layer Normalization - efficient variant of LayerNorm.
Normalizes inputs using only the root mean square (no mean centering).

Parameters:
- dim: Size of the feature dimension to normalize

Shape Contract:
- Input: [*, dim] where dim is the normalized dimension
- Output: [*, dim] same shape as input

Notes:
- Formula: RMSNorm(x) = x / sqrt(mean(x*x) + eps) * gamma
- Omits mean subtraction from LayerNorm (only rescales by RMS)
- 10-15% faster than LayerNorm with similar performance
- Used in LLaMA, T5, and other modern transformers
- Includes learnable scale parameter (gamma)
- Particularly effective in large language models

## Signature

```neuroscript
neuron RMSNorm(dim)
```

## Ports

**Inputs:**
- `default`: `[*, dim]`

**Outputs:**
- `default`: `[*, dim]`

## Implementation

```
Source { source: "core", path: "normalization/RMSNorm" }
```

