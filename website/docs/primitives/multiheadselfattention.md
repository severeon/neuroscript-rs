---
sidebar_label: MultiHeadSelfAttention
---

# MultiHeadSelfAttention

Multi-Head Self-Attention

Complete multi-head self-attention mechanism where queries, keys, and values
all come from the same input. Core component of transformer architectures.

Parameters:
- dim: Model dimension (d_model)
- num_heads: Number of attention heads (dim must be divisible by num_heads)

Shape Contract:
- Input: [*, seq_len, dim] sequence of embeddings
- Output: [*, seq_len, dim] attended sequence (same shape)

Notes:
- Includes Q, K, V projections and output projection
- head_dim = dim / num_heads
- Each head attends independently, results are concatenated
- Self-attention: Q, K, V all derived from same input
- Can support causal masking for autoregressive models
- Used in BERT, GPT, and virtually all transformers

## Signature

```neuroscript
neuron MultiHeadSelfAttention(dim, num_heads)
```

## Ports

**Inputs:**
- `default`: `[*, seq_len, dim]`

**Outputs:**
- `default`: `[*, seq_len, dim]`

## Implementation

```
"from core import attention/MultiHeadSelfAttention"
```

```
Source { source: "core", path: "attention/MultiHeadSelfAttention" }
```

