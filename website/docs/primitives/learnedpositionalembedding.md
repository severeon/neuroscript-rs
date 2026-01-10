---
sidebar_label: LearnedPositionalEmbedding
---

# LearnedPositionalEmbedding

Learned Positional Embedding

Adds learnable position embeddings to input sequences. Unlike sinusoidal
encoding, positions are learned during training.

Parameters:
- max_positions: Maximum sequence length supported
- embedding_dim: Dimension of the embeddings (must match input)

Shape Contract:
- Input: [*, seq_len, embedding_dim] sequence of embeddings
- Output: [*, seq_len, embedding_dim] embeddings with positions added

Notes:
- Position embeddings are learned parameters (not fixed functions)
- Used in BERT, GPT-2, RoBERTa
- Cannot extrapolate beyond max_positions
- Embedding table shape: [max_positions, embedding_dim]
- Positions are added (not concatenated) to input embeddings
- More flexible than sinusoidal but requires training

## Signature

```neuroscript
neuron LearnedPositionalEmbedding(max_positions, embedding_dim)
```

## Ports

**Inputs:**
- `default`: `[*, seq_len, embedding_dim]`

**Outputs:**
- `default`: `[*, seq_len, embedding_dim]`

## Implementation

```
"from core import embeddings/LearnedPositionalEmbedding"
```

```
Source { source: "core", path: "embeddings/LearnedPositionalEmbedding" }
```

