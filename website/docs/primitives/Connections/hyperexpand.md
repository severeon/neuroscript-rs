---
sidebar_label: HyperExpand
---

# HyperExpand

Expand a tensor into the n-wide Hyper-Connection residual stream.

Repeats the input along a new dimension to create n identical copies,
preparing it for processing by ManifoldHyperConnect layers.

Parameters:
- n: Number of stream copies (expansion factor)

Shape Contract:
- Input: [*batch, dim]
- Output: [*batch, n, dim]

Notes:
- Used at model entry to initialize the n-wide residual stream
- Paired with HyperCollapse at model exit

## Signature

```neuroscript
neuron HyperExpand(n)
```

## Ports

**Inputs:**
- `default`: `[*batch, dim]`

**Outputs:**
- `default`: `[*batch, n, dim]`

## Implementation

```
Source { source: "neuroscript", path: "connections/HyperExpand" }
```
