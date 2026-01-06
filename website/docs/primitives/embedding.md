---
sidebar_label: Embedding
---

# Embedding

Token Embedding

Maps discrete token indices to dense continuous vectors. Fundamental
component for processing text, categorical data, or any discrete symbols.

Parameters:
- num_embeddings: Size of the vocabulary (number of unique tokens)
- embedding_dim: Dimension of the dense embedding vectors

Shape Contract:
- Input: [*, seq_len] integer token indices
- Output: [*, seq_len, embedding_dim] dense embeddings

Notes:
- Input indices must be in range [0, num_embeddings)
- Embeddings are learned parameters (initialized randomly)
- Common in NLP: word embeddings, position embeddings, token type embeddings

## Signature

```neuroscript
neuron Embedding(num_embeddings, embedding_dim)
```

## Ports

**Inputs:**
- `default`: `[*, seq_len]`

**Outputs:**
- `default`: `[*, seq_len, embedding_dim]`

## Implementation

```
Source { source: "core", path: "embeddings/Embedding" }
```

