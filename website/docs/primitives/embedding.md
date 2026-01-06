---
sidebar_label: Embedding
---

# Embedding

Token embedding layer

Maps discrete token indices to dense continuous vectors. Fundamental
component for processing text, categorical data, or any discrete symbols.

## Signature

```neuroscript
neuron Embedding(num_embeddings, embedding_dim)
```

## Parameters

- num_embeddings: Size of the vocabulary (number of unique tokens)
- embedding_dim: Dimension of the dense embedding vectors

## Shape Contract

- Input: [*, seq_len] integer token indices
- Output: [*, seq_len, embedding_dim] dense embeddings

## Ports

**Inputs:**
- `default`: `[*, seq_len]`

**Outputs:**
- `default`: `[*, seq_len, embedding_dim]`

## Example

```neuroscript
neuron LanguageModel:
graph:
in -> Embedding(50257, 768) -> TransformerStack(768) -> out
```

## Notes

- Input indices must be in range [0, num_embeddings)
- Embeddings are learned parameters (initialized randomly)
- Common in NLP: word embeddings, position embeddings, token type embeddings

## Implementation

```
Source { source: "core", path: "embeddings/Embedding" }
```

