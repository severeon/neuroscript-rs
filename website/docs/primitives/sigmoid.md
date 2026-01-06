---
sidebar_label: Sigmoid
---

# Sigmoid

Sigmoid Activation

Maps input values to the range (0, 1) using the logistic function.
Commonly used for binary classification outputs and gating mechanisms.

Shape Contract:
- Input: [*shape] arbitrary shape
- Output: [*shape] same shape as input

Formula: sigmoid(x) = 1 / (1 + exp(-x))

Notes:
- Output range: (0, 1), useful for probabilities
- Suffers from vanishing gradients for large absolute values
- Used in LSTMs, attention gates, and output layers
- Element-wise operation (preserves all dimensions)

## Signature

```neuroscript
neuron Sigmoid()
```

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Implementation

```
Source { source: "core", path: "activations/Sigmoid" }
```

