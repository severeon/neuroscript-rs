---
sidebar_label: DropConnect
---

# DropConnect

DropConnect Regularization

Randomly zeros individual weights (connections) during training rather than
activations. Provides stronger regularization than standard Dropout.

Parameters:
- drop_prob: Probability of a connection being zeroed (0 to 1)

Shape Contract:
- Input: [*shape] arbitrary shape
- Output: [*shape] same shape as input

Notes:
- Drops weights, not activations (unlike Dropout)
- Applied to the connection weights during forward pass
- More fine-grained than Dropout
- Only active during training
- Used in EfficientNet and some attention mechanisms
- Outputs scaled to maintain expected values

## Signature

```neuroscript
neuron DropConnect(drop_prob)
```

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Implementation

```
"from core import regularization/DropConnect"
```

```
Source { source: "core", path: "regularization/DropConnect" }
```

