---
sidebar_label: RotaryEmbedding
---

# RotaryEmbedding

Rotary Position Embedding (RoPE)

Rotates query and key tensors to encode relative positional information.
Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding".

Parameters:
- dim: Embedding dimension (head_dim)
- max_position_embeddings: Maximum sequence length to pre-compute (default: 2048)
- base: Base for the geometric progression of frequencies (default: 10000.0)

## Signature

```neuroscript
neuron RotaryEmbedding(dim, max_position_embeddings=Int(2048), base=Int(10000))
```

## Ports

**Inputs:**
- `query`: `[*batch, seq, num_heads, dim]`
- `key`: `[*batch, seq, num_heads, dim]`

**Outputs:**
- `q_out`: `[*batch, seq, num_heads, dim]`
- `k_out`: `[*batch, seq, num_heads, dim]`

## Implementation

```
"from neuroscript import embeddings/RotaryEmbedding"
```

```
Source { source: "neuroscript", path: "embeddings/RotaryEmbedding" }
```

