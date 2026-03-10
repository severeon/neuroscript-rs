---
sidebar_label: HyperCollapse
---

# HyperCollapse

Collapse the n-wide Hyper-Connection residual stream back to a single channel.

Takes the first stream (index 0) from the expanded residual, discarding
the auxiliary channels used for cross-layer communication.

Shape Contract:
- Input: [*batch, n, dim]
- Output: [*batch, dim]

Notes:
- Used at model exit to recover the original dimensionality
- Paired with HyperExpand at model entry
- Only the first channel carries the primary signal; others are mixing channels

## Signature

```neuroscript
neuron HyperCollapse()
```

## Ports

**Inputs:**
- `default`: `[*batch, n, dim]`

**Outputs:**
- `default`: `[*batch, dim]`

## Implementation

```
Source { source: "neuroscript", path: "connections/HyperCollapse" }
```
