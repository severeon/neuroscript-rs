---
sidebar_position: 1
title: Examples
---

# Real-World Examples

These examples show NeuroScript used to design production-quality model architectures, demonstrating how the language's composability handles real research-level complexity.

Each example includes:
- The NeuroScript architecture definition
- Design rationale and parameter choices
- A runnable PyTorch demo
- Performance measurements on Apple Silicon (MPS)

## Available Examples

- **[Manifold-Constrained Hyper-Connections](./mhc-transformer)** — Deep model stability via doubly stochastic residual mixing (DeepSeek-AI, arXiv:2512.24880v2). Includes a 40-layer transformer with n=4 stream expansion and fine-tuning adapter analysis.
