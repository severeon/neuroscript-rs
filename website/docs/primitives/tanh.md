---
sidebar_label: Tanh
---

# Tanh

Tanh Activation

Hyperbolic tangent - maps input values to the range (-1, 1).
Zero-centered output makes it preferable to sigmoid in hidden layers.

Shape Contract:
- Input: [*shape] arbitrary shape
- Output: [*shape] same shape as input

Notes:
- Formula: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
- Output range: (-1, 1), zero-centered
- Stronger gradients than sigmoid near zero
- Still suffers from vanishing gradients for large values
- Used in LSTMs, RNNs, and as output activation for bounded values
- Element-wise operation (preserves all dimensions)

## Signature

```neuroscript
neuron Tanh()
```

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Implementation

```
"from core import activations/Tanh"
```

```
Source { source: "core", path: "activations/Tanh" }
```

