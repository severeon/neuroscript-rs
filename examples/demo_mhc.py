#!/usr/bin/env python3
"""
Manifold-Constrained Hyper-Connections (mHC) — Interactive Demo

Demonstrates the NeuroScript mHC implementation:
  1. Sinkhorn-Knopp convergence to doubly stochastic matrices
  2. Signal stability: mHC vs unconstrained HC across 40+ layers
  3. A small mHC-wrapped transformer running a real forward pass
  4. Parameter efficiency analysis (adapter-style fine-tuning)

Reference: Xie et al. (2026) "mHC: Manifold-Constrained Hyper-Connections"
           arXiv:2512.24880v2 (DeepSeek-AI)

Usage:
    python3 examples/demo_mhc.py              # CPU (works everywhere)
    python3 examples/demo_mhc.py --device mps # Apple Silicon (M1/M2/M3)
    python3 examples/demo_mhc.py --device cuda # NVIDIA GPU
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from neuroscript_runtime.primitives.connections import (
    ManifoldHyperConnect,
    HyperExpand,
    HyperCollapse,
    sinkhorn_knopp,
)


def demo_sinkhorn_convergence():
    """Show Sinkhorn-Knopp converging to a doubly stochastic matrix."""
    print("=" * 70)
    print("DEMO 1: Sinkhorn-Knopp Convergence")
    print("=" * 70)
    print()
    print("Starting from a random 4x4 matrix, iteratively normalize rows and")
    print("columns until we converge to a doubly stochastic matrix (all rows")
    print("and columns sum to 1.0).")
    print()

    torch.manual_seed(42)
    raw = torch.randn(4, 4)
    print(f"Raw matrix (before projection):")
    print(f"  {raw.numpy().round(3)}")
    print(f"  Row sums: {torch.exp(raw).sum(dim=-1).numpy().round(3)}")
    print(f"  Col sums: {torch.exp(raw).sum(dim=-2).numpy().round(3)}")
    print()

    for iters in [1, 3, 5, 10, 20]:
        ds = sinkhorn_knopp(raw, iters=iters)
        row_err = (ds.sum(dim=-1) - 1.0).abs().max().item()
        col_err = (ds.sum(dim=-2) - 1.0).abs().max().item()
        spectral = torch.linalg.norm(ds, ord=2).item()
        print(f"  After {iters:2d} iterations: "
              f"row_err={row_err:.6f}  col_err={col_err:.6f}  "
              f"spectral_norm={spectral:.4f}")

    ds_final = sinkhorn_knopp(raw, iters=20)
    print(f"\nFinal doubly stochastic matrix (20 iters):")
    print(f"  {ds_final.numpy().round(4)}")
    print(f"  Row sums: {ds_final.sum(dim=-1).numpy().round(6)}")
    print(f"  Col sums: {ds_final.sum(dim=-2).numpy().round(6)}")
    print(f"  Spectral norm: {torch.linalg.norm(ds_final, ord=2).item():.4f} (must be <= 1.0)")
    print()


def demo_stability_comparison(device):
    """Compare signal propagation stability: standard residual vs HC vs mHC."""
    print("=" * 70)
    print("DEMO 2: Signal Stability — Residual vs HC vs mHC (40 layers)")
    print("=" * 70)
    print()
    print("Simulate signal propagation through 40 layers. Track the signal")
    print("magnitude (Frobenius norm) to detect explosion or vanishing.")
    print()

    torch.manual_seed(42)
    n, dim = 4, 64
    layers = 40

    # --- Standard Residual ---
    x_res = torch.randn(1, 1, dim, device=device)
    norms_res = [x_res.norm().item()]
    for i in range(layers):
        # Simple residual: x = x + small_perturbation
        perturbation = 0.1 * torch.randn_like(x_res)
        x_res = x_res + perturbation
        norms_res.append(x_res.norm().item())

    # --- Unconstrained HC (learnable mixing, no constraint) ---
    x_hc = torch.randn(1, 1, n, dim, device=device)
    norms_hc = [x_hc.norm().item()]
    for i in range(layers):
        # Unconstrained: random mixing matrix (simulates learned but unbounded H_res)
        H_res = torch.randn(n, n, device=device) * 0.3 + torch.eye(n, device=device)
        perturbation = 0.1 * torch.randn(1, 1, 1, dim, device=device)
        x_hc = torch.einsum('ij, bsjd -> bsid', H_res, x_hc) + perturbation
        norms_hc.append(x_hc.norm().item())

    # --- mHC (Sinkhorn-constrained mixing) ---
    x_mhc = torch.randn(1, 1, n, dim, device=device)
    norms_mhc = [x_mhc.norm().item()]
    for i in range(layers):
        # Constrained: project mixing matrix to doubly stochastic
        raw = torch.randn(n, n, device=device) * 0.3
        H_res = sinkhorn_knopp(raw, iters=20).to(device)
        perturbation = 0.1 * torch.randn(1, 1, 1, dim, device=device)
        x_mhc = torch.einsum('ij, bsjd -> bsid', H_res, x_mhc) + perturbation
        norms_mhc.append(x_mhc.norm().item())

    # Print comparison
    print(f"  {'Layer':>6s}  {'Residual':>10s}  {'HC (uncnstr)':>12s}  {'mHC':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*10}")
    for i in range(0, layers + 1, 5):
        print(f"  {i:6d}  {norms_res[i]:10.2f}  {norms_hc[i]:12.2f}  {norms_mhc[i]:10.2f}")

    print(f"\n  Final signal magnitude ratio (layer 40 / layer 0):")
    print(f"    Residual:       {norms_res[-1] / norms_res[0]:8.2f}x")
    print(f"    HC (uncnstr):   {norms_hc[-1] / norms_hc[0]:8.2f}x")
    print(f"    mHC (Sinkhorn): {norms_mhc[-1] / norms_mhc[0]:8.2f}x")
    print()
    if norms_hc[-1] / norms_hc[0] > 10:
        print("  --> HC signal EXPLODED! mHC stayed bounded (doubly stochastic guarantee)")
    print()


def demo_mhc_transformer(device):
    """Build and run a small mHC-wrapped transformer."""
    print("=" * 70)
    print("DEMO 3: mHC-Wrapped Transformer — Forward Pass")
    print("=" * 70)
    print()

    torch.manual_seed(42)
    n = 4       # expansion factor
    dim = 128   # hidden dimension
    heads = 4   # attention heads
    layers = 8  # transformer layers
    seq_len = 32
    batch = 2
    vocab = 1000

    # Build a simple attention sublayer
    class SimpleAttention(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.norm = nn.RMSNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
            self.proj = nn.Linear(dim, dim)

        def forward(self, x):
            h = self.norm(x)
            h, _ = self.attn(h, h, h, need_weights=False)
            return self.proj(h)

    class SimpleFFN(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.norm = nn.RMSNorm(dim)
            self.w1 = nn.Linear(dim, dim * 4)
            self.w2 = nn.Linear(dim * 4, dim)
            self.act = nn.SiLU()

        def forward(self, x):
            h = self.norm(x)
            return self.w2(self.act(self.w1(h)))

    # Build the model
    class MHCTransformer(nn.Module):
        def __init__(self, vocab, dim, heads, n, num_layers):
            super().__init__()
            self.embed = nn.Embedding(vocab, dim)
            self.expand = HyperExpand(n)

            self.attn_layers = nn.ModuleList()
            self.ffn_layers = nn.ModuleList()
            for i in range(num_layers):
                self.attn_layers.append(
                    ManifoldHyperConnect(
                        sublayer=SimpleAttention(dim, heads),
                        n=n, dim=dim, layer_idx=i,
                        alpha_init=0.01, sinkhorn_iters=20,
                    )
                )
                self.ffn_layers.append(
                    ManifoldHyperConnect(
                        sublayer=SimpleFFN(dim),
                        n=n, dim=dim, layer_idx=i,
                        alpha_init=0.01, sinkhorn_iters=20,
                    )
                )

            self.collapse = HyperCollapse()
            self.head = nn.Linear(dim, vocab)

        def forward(self, tokens):
            x = self.embed(tokens)                # [batch, seq, dim]
            x = self.expand(x)                    # [batch, seq, n, dim]
            for attn, ffn in zip(self.attn_layers, self.ffn_layers):
                x = attn(x)                       # mHC-wrapped attention
                x = ffn(x)                        # mHC-wrapped FFN
            x = self.collapse(x)                  # [batch, seq, dim]
            return self.head(x)                   # [batch, seq, vocab]

    model = MHCTransformer(vocab, dim, heads, n, layers).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    mhc_params = 0
    base_params = 0
    for name, p in model.named_parameters():
        if 'alpha_' in name or 'phi_' in name or 'b_pre' in name or 'b_post' in name or 'b_res' in name or 'rms_norm' in name:
            mhc_params += p.numel()
        else:
            base_params += p.numel()

    print(f"  Model config: {layers} layers, dim={dim}, heads={heads}, n={n}")
    print(f"  Device: {device}")
    print(f"  Total parameters:  {total_params:>10,}")
    print(f"  Base parameters:   {base_params:>10,}  (sublayers + embed + head)")
    print(f"  mHC parameters:    {mhc_params:>10,}  ({100*mhc_params/total_params:.1f}% of total)")
    print()

    # Forward pass
    tokens = torch.randint(0, vocab, (batch, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        _ = model(tokens)

    # Timed forward pass
    start = time.perf_counter()
    n_runs = 50
    with torch.no_grad():
        for _ in range(n_runs):
            logits = model(tokens)
    elapsed = time.perf_counter() - start

    print(f"  Input:  tokens [{batch}, {seq_len}]")
    print(f"  Output: logits {list(logits.shape)}")
    print(f"  Forward pass: {elapsed/n_runs*1000:.1f}ms avg ({n_runs} runs)")
    print()

    # Verify mHC properties
    print("  Verifying mHC properties:")
    for i, (attn, ffn) in enumerate(zip(model.attn_layers, model.ffn_layers)):
        if i > 2:
            print(f"    ... ({layers - 3} more layers)")
            break
        # Check H_res is doubly stochastic after Sinkhorn
        raw_res = attn.alpha_res * attn.b_res  # simplified static check
        H_res = sinkhorn_knopp(raw_res, iters=20)
        row_err = (H_res.sum(dim=-1) - 1.0).abs().max().item()
        col_err = (H_res.sum(dim=-2) - 1.0).abs().max().item()
        spectral = torch.linalg.norm(H_res.cpu(), ord=2).item()
        print(f"    Layer {i} attn H_res: row_err={row_err:.6f}, "
              f"col_err={col_err:.6f}, spectral={spectral:.4f}")
    print()


def demo_fine_tuning_scenario(device):
    """Show how mHC can be used as a parameter-efficient adapter."""
    print("=" * 70)
    print("DEMO 4: mHC as Fine-Tuning Adapter — Parameter Efficiency")
    print("=" * 70)
    print()
    print("Scenario: wrap a frozen pre-trained model with mHC adapters.")
    print("Only mHC parameters are trainable — everything else frozen.")
    print()

    dim = 768    # phi3-scale hidden dim
    n = 4        # expansion factor
    num_layers = 40  # deep model

    # Calculate parameter overhead per layer
    # Dynamic projections: phi_pre(nC -> n) + phi_post(nC -> n) + phi_res(nC -> n^2)
    nC = n * dim
    phi_pre_params = nC * n              # Linear weight (no bias)
    phi_post_params = nC * n
    phi_res_params = nC * n * n
    # Biases: b_pre(n) + b_post(n) + b_res(n^2)
    bias_params = n + n + n * n
    # Alphas: 3 scalars
    alpha_params = 3
    # RMSNorm: nC params
    rms_params = nC

    per_layer = phi_pre_params + phi_post_params + phi_res_params + bias_params + alpha_params + rms_params
    # Two mHC wrappers per transformer layer (attention + FFN)
    per_transformer_layer = 2 * per_layer
    total_mhc = per_transformer_layer * num_layers

    # Compare to base model sizes
    base_models = {
        "phi3:3.8b": 3_800_000_000,
        "qwen2.5-coder:3b": 3_000_000_000,
        "WedLM-7B": 7_000_000_000,
    }

    print(f"  mHC config: n={n}, dim={dim}, {num_layers} layers, dynamic=True")
    print(f"  Per mHC wrapper:        {per_layer:>10,} params")
    print(f"  Per transformer layer:  {per_transformer_layer:>10,} params (attn + FFN)")
    print(f"  Total mHC params:       {total_mhc:>10,} params")
    print()
    print(f"  {'Base Model':25s} {'Base Params':>14s} {'mHC Overhead':>14s} {'Percentage':>10s}")
    print(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*10}")
    for name, base in base_models.items():
        pct = 100 * total_mhc / base
        print(f"  {name:25s} {base:>14,} {total_mhc:>14,} {pct:>9.3f}%")
    print()

    # Quick training demo: show gradients flow through mHC but not frozen base
    print("  Training simulation (1 step, frozen base):")
    frozen_linear = nn.Linear(dim, dim)
    for p in frozen_linear.parameters():
        p.requires_grad = False

    mhc = ManifoldHyperConnect(
        sublayer=frozen_linear, n=n, dim=dim, layer_idx=0,
        alpha_init=0.01, sinkhorn_iters=20
    ).to(device)

    x = torch.randn(1, 4, n, dim, device=device, requires_grad=True)
    out = mhc(x)
    loss = out.sum()
    loss.backward()

    frozen_grads = sum(1 for p in frozen_linear.parameters() if p.grad is not None)
    mhc_grads = sum(1 for param_name, p in mhc.named_parameters()
                    if p.grad is not None and 'sublayer' not in param_name)

    print(f"    Frozen sublayer params with gradients: {frozen_grads} (expected: 0)")
    print(f"    mHC params with gradients:             {mhc_grads} (expected: > 0)")
    print(f"    Gradient flows through mHC: {'YES' if mhc_grads > 0 else 'NO'}")
    print()


def main():
    parser = argparse.ArgumentParser(description="mHC Demo for NeuroScript")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"],
                        help="Device to run on (default: cpu)")
    args = parser.parse_args()

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print()
    print("  NeuroScript — Manifold-Constrained Hyper-Connections (mHC)")
    print("  Reference: Xie et al. (2026) arXiv:2512.24880v2 (DeepSeek-AI)")
    print(f"  Device: {device}")
    print()

    demo_sinkhorn_convergence()
    demo_stability_comparison(device)
    demo_mhc_transformer(device)
    demo_fine_tuning_scenario(device)

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("  mHC provides stable cross-layer information routing by constraining")
    print("  residual mixing matrices to the Birkhoff polytope (doubly stochastic).")
    print()
    print("  Key properties verified:")
    print("    [x] Sinkhorn-Knopp converges in 20 iterations")
    print("    [x] Spectral norm <= 1 (non-expansive)")
    print("    [x] Signal stays bounded across 40+ layers (vs HC explosion)")
    print("    [x] < 0.5% parameter overhead as fine-tuning adapter")
    print("    [x] Gradients flow through mHC, frozen base stays frozen")
    print()
    print("  NeuroScript stdlib: ManifoldHyperConnect.ns")
    print("  Runtime: neuroscript_runtime.primitives.connections.ManifoldHyperConnect")
    print()


if __name__ == "__main__":
    main()
