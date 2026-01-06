---
sidebar_label: PositionalEncoding
---

# PositionalEncoding

Positional Encoding (Sinusoidal)

Adds fixed sinusoidal position information to input sequences.
From the original Transformer paper (Vaswani et al., 2017).

Parameters:
- dim: Embedding dimension (must match input)
- max_len: Maximum sequence length to precompute

Shape Contract:
- Input: [*, seq_len, dim] sequence of embeddings
- Output: [*, seq_len, dim] embeddings with positions added

Notes:
- Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d))
- Formula: PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
- No learnable parameters (fixed encoding)
- Can extrapolate to longer sequences than trained on
- Encodes both absolute and relative position information
- Used in original Transformer, still common in many architectures

## Signature

```neuroscript
neuron PositionalEncoding(dim, max_len)
```

## Ports

**Inputs:**
- `default`: `[*, seq_len, dim]`

**Outputs:**
- `default`: `[*, seq_len, dim]`

## Implementation

```
Source { source: "core", path: "embeddings/PositionalEncoding" }
```

