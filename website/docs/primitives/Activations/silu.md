---
sidebar_label: SiLU
---

# SiLU

SiLU Activation (Swish)

Sigmoid Linear Unit: x * sigmoid(x)
Self-gated activation that often outperforms ReLU in deep networks.

Shape Contract:
- Input: [*shape] arbitrary shape
- Output: [*shape] same shape as input

Notes:
- Formula: SiLU(x) = x * sigmoid(x)
- Also known as Swish activation (Google Brain, 2017)
- Smooth approximation with learnable properties
- Used in EfficientNet, GPT-NeoX, LLaMA, and modern architectures
- Slightly more expensive than ReLU but often better performance
- Element-wise operation (preserves all dimensions)

## Signature

```neuroscript
neuron SiLU()
```

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Implementation

```
Source { source: "core", path: "activations/SiLU" }
```

