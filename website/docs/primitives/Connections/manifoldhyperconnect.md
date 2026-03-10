---
sidebar_label: ManifoldHyperConnect
---

# ManifoldHyperConnect

Manifold-Constrained Hyper-Connection (mHC) wrapper.

Wraps any sublayer with an n-wide residual stream, constraining the residual
mixing matrix to the Birkhoff polytope (doubly stochastic) via Sinkhorn-Knopp
normalization. Guarantees spectral norm <= 1 for training stability at any depth.

Parameters:
- layer: The sublayer to wrap (Neuron type)
- n: Expansion factor for residual stream width (typically 4)
- dim: Model dimension
- layer_idx: Layer index for per-layer parameters
- alpha_init: Gating factor initialization (default: 0.01)
- sinkhorn_iters: Number of Sinkhorn-Knopp iterations (default: 20)

Shape Contract:
- Input: [*batch, n, dim] — expanded residual stream
- Output: [*batch, n, dim] — updated residual stream after layer + mixing

Notes:
- For n=1, degenerates to standard residual connection
- H_res is projected to doubly stochastic (all rows and columns sum to 1)
- H_pre and H_post use sigmoid to enforce non-negativity
- Adds ~0.1-0.2% parameter overhead to base model

Reference: Xie et al. (2026) arXiv:2512.24880v2 (DeepSeek-AI)

## Signature

```neuroscript
neuron ManifoldHyperConnect(layer: Neuron, n, dim, layer_idx, alpha_init=0.01, sinkhorn_iters=20)
```

## Ports

**Inputs:**
- `default`: `[*batch, n, dim]`

**Outputs:**
- `default`: `[*batch, n, dim]`

## Implementation

```
Source { source: "neuroscript", path: "connections/ManifoldHyperConnect" }
```
