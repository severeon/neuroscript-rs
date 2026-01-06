---
sidebar_label: ReLU
---

# ReLU

Rectified Linear Unit (ReLU) activation function

Simple non-linear activation that outputs max(0, x). One of the most
widely used activation functions in deep learning due to its simplicity
and effectiveness.

## Signature

```neuroscript
neuron ReLU()
```

## Shape Contract

- Input: [*shape] arbitrary shape
- Output: [*shape] same shape as input

## Ports

**Inputs:**
- `default`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Example

```neuroscript
neuron ConvBlock(channels):
graph:
in -> Conv2d(channels, channels) -> ReLU() -> out
```

## Notes

- Element-wise operation: ReLU(x) = max(0, x)
- Non-differentiable at x=0 (subgradient is typically used)
- Can suffer from "dying ReLU" problem (neurons output 0 for all inputs)
- Fast to compute, no vanishing gradient for positive inputs

## Implementation

```
Source { source: "core", path: "activations/ReLU" }
```

