---
sidebar_label: Add
---

# Add

Element-wise Addition

Adds two tensors element-by-element. Fundamental operation for residual
connections that enable training of very deep networks.

Shape Contract:
- Input main: [*shape] primary tensor
- Input skip: [*shape] tensor to add (must match shape of main)
- Output: [*shape] element-wise sum

Notes:
- No learnable parameters (pure element-wise operation)
- Both inputs must have identical shapes
- Named inputs (main, skip) follow residual connection convention
- Enables gradient flow through skip path (solves vanishing gradients)
- Central to ResNet, Transformers, and most modern architectures
- out = main + skip

## Signature

```neuroscript
neuron Add()
```

## Ports

**Inputs:**
- `main`: `[*shape]`
- `skip`: `[*shape]`

**Outputs:**
- `default`: `[*shape]`

## Implementation

```
"from core import structural/Add"
```

```
Source { source: "core", path: "structural/Add" }
```

