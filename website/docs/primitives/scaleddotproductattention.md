---
sidebar_label: ScaledDotProductAttention
---

# ScaledDotProductAttention

Scaled Dot-Product Attention

Core attention mechanism: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
From the original Transformer paper (Vaswani et al., 2017).

Parameters:
- d_k: Dimension of keys (for scaling factor sqrt(d_k))

Shape Contract:
- Input query: [*, seq_q, d_k] query vectors
- Input key: [*, seq_k, d_k] key vectors
- Input value: [*, seq_v, d_v] value vectors (seq_v == seq_k)
- Output: [*, seq_q, d_v] attention-weighted values

Notes:
- Scaling by sqrt(d_k) prevents softmax saturation
- No learnable parameters (projections are in outer layer)
- Building block for multi-head attention
- Can include optional attention mask for causal/padding masking

## Signature

```neuroscript
neuron ScaledDotProductAttention(d_k)
```

## Ports

**Inputs:**
- `query`: `[*, seq_q, d_k]`
- `key`: `[*, seq_k, d_k]`
- `value`: `[*, seq_v, d_v]`

**Outputs:**
- `default`: `[*, seq_q, d_v]`

## Implementation

```
Source { source: "core", path: "attention/ScaledDotProductAttention" }
```

