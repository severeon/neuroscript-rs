"""
NeuralBlock — banded circular block with recursive embedding.

This is an experimental architectural primitive. The block has three integer
parameters:

    N : node width    — rows/columns in the cell grid (and base I/O width)
    D : deviations    — kernel half-width for inter-column spreading (kernel
                        width = 2*D+1, with circular wraparound). Must satisfy
                        D <= max(0, (N // 2) - 1).
    R : recursion     — recursive embedding depth. At R=0 each cell holds a
                        single scalar weight; at R>=1 each cell is itself a
                        NeuralBlock(N, D, R-1).

Total parameter count: P = (N*N)^(R+1) + 1  (the +1 is the root scalar bias).

Forward semantics. NeuralBlock is a linear operator: NB(x) = W·x + bias for
some W ∈ R^{dim×dim} that depends only on the cell parameters. We materialize
W once via batched einsums and apply it as a single dense matmul.

Performance notes
-----------------
* Storage is **flat** — at R≥1 the entire cell tree collapses to a single
  ``nn.Parameter`` of shape ``[G_total, N, N]`` where ``G_total = N^(2R)``.
  Iterating column-by-column at materialization time uses simple reshape +
  batched einsum; there is no Python recursion through nn.ModuleList.
* W is **cached** when ``forward`` runs with autograd disabled (rollout time).
  Call ``cache_dense_weight()`` explicitly to pre-materialize, or rely on the
  first cached call. Cache is bypassed automatically when grad is enabled
  so PPO updates always see fresh autograd-tracked weights.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class NeuralBlock(nn.Module):
    """Banded circular block with recursive embedding.

    Args:
        n: Node width N (rows and columns). Must be >= 1.
        dim: Total I/O dimension. Must equal n ** (r + 1).
        d: Deviation half-width D. Must satisfy 0 <= d <= max(0, n // 2 - 1).
        r: Recursive embedding depth R. Must be >= 0.
        device, dtype: Standard PyTorch factory kwargs.

    Shape:
        Input  : [*, dim]
        Output : [*, dim]
    """

    def __init__(
        self,
        n: int,
        dim: int,
        d: int = 0,
        r: int = 0,
        *,
        _root: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if r < 0:
            raise ValueError(f"r must be >= 0, got {r}")
        if d < 0:
            raise ValueError(f"d must be >= 0, got {d}")
        max_d = max(0, (n // 2) - 1)
        if d > max_d:
            raise ValueError(
                f"d={d} exceeds maximum allowed (n // 2) - 1 = {max_d} for n={n}"
            )

        expected_dim = n ** (r + 1)
        if dim != expected_dim:
            raise ValueError(
                f"dim must equal n**(r+1) = {expected_dim} for n={n}, r={r}; got dim={dim}"
            )

        self.n = n
        self.d = d
        self.r = r
        self.dim = dim
        self.kernel_width = 2 * d + 1
        self._root = _root

        factory_kwargs = {"device": device, "dtype": dtype}

        # Init scaled so each per-column banded-sum step stays roughly
        # unit-variance: a sum of `kernel_width` neighbours times a weight,
        # so divide by sqrt(kernel_width).
        scale = 1.0 / math.sqrt(self.kernel_width)
        if r == 0:
            self.weights = nn.Parameter(torch.randn(n, n, **factory_kwargs) * scale)
        else:
            # G_total = N^(2R) leaves, each [N, N]. Depth-first cell-major
            # ordering: the (g)-th leaf belongs to outer-cell ``g // N^(2(R-1))``,
            # then to that cell's ``g // N^(2(R-2))``-th cell, etc. The
            # iterative composition in materialize_weight() folds this
            # ordering by simple reshape — no gather needed.
            g_total = n ** (2 * r)
            self.leaf_weights = nn.Parameter(
                torch.randn(g_total, n, n, **factory_kwargs) * scale
            )

        if _root:
            self.bias = nn.Parameter(torch.zeros(1, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # Materialized W cache. Populated by forward() when grad is disabled,
        # and explicitly invalidated by clear_dense_weight_cache(). Always
        # bypassed when grad is enabled so PPO updates see fresh autograd.
        self._W_cache: Optional[torch.Tensor] = None

    def num_block_parameters(self) -> int:
        """Return the total trainable parameter count for this block."""
        return sum(p.numel() for p in self.parameters())

    # ──────────────────────────────────────────────────────────────────────
    # Materialization: NB(x) = W·x + bias. We construct W from the flat
    # leaf-weights tensor in O(R) batched einsum levels.
    # ──────────────────────────────────────────────────────────────────────

    def _all_leaves(self) -> torch.Tensor:
        """Return the flat ``[G_total, N, N]`` leaf-weight tensor."""
        if self.r == 0:
            return self.weights.unsqueeze(0)            # [1, N, N]
        return self.leaf_weights                        # [G_total, N, N]

    def _banded_circular_adjacency(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """[N, N] {0,1}-matrix B with B[i, j] = 1 iff (i-j) mod N ∈ [-D, D]."""
        idx = torch.arange(self.n, device=device)
        diffs = (idx.unsqueeze(0) - idx.unsqueeze(1)) % self.n
        diffs_sym = torch.minimum(diffs, self.n - diffs)
        return (diffs_sym <= self.d).to(dtype)

    def _vectorized_r0_dense(self, leaf_weights: torch.Tensor) -> torch.Tensor:
        """Compose R=0 cells into [G, N, N] dense weight matrices.

        Args:
            leaf_weights: [G, N, N] — leaf_weights[g, c, i] is the scalar for
                column c, row i of leaf g.
        Returns: [G, N, N]
        """
        N = self.n
        device, dtype = leaf_weights.device, leaf_weights.dtype
        B = self._banded_circular_adjacency(device, dtype)
        eye = torch.eye(N, device=device, dtype=dtype)

        def batch_diag(v):
            return v.unsqueeze(-1) * eye

        W = batch_diag(leaf_weights[:, 0, :])
        for c in range(1, N):
            W = batch_diag(leaf_weights[:, c, :]) @ B @ W
        return W

    def _compose_one_level(self, cell_W: torch.Tensor, M: int) -> torch.Tensor:
        """Compose N² inner-cells (each M×M) into an outer NM×NM dense block.

        Args:
            cell_W: [G, N², M, M] — cell_W[:, c·N + i] is the (col=c, row=i)
                inner-cell's effective weight.
            M: inner sub-block dimension.
        Returns: [G, NM, NM]
        """
        N = self.n
        G = cell_W.shape[0]
        NM = N * M
        device, dtype = cell_W.device, cell_W.dtype

        cell_W = cell_W.reshape(G, N, N, M, M)          # [G, col, row, M_out, M_in]

        B = self._banded_circular_adjacency(device, dtype)
        eye_N = torch.eye(N, device=device, dtype=dtype)

        # W ∈ R^{G, i_out=N, m_out=M, j_in=N, m_in=M}.
        # Column 0: per-cell apply only (no banded sum).
        #   W_after0[g, i, o, j, n] = δ_ij · cell_W[g, 0, i, o, n]
        W = cell_W[:, 0].unsqueeze(-2) * eye_N.view(1, N, 1, N, 1)

        # Columns c = 1..N-1: banded sum over rows, then per-cell apply.
        for c in range(1, N):
            W = torch.einsum("ik, gkojn -> giojn", B, W)
            W = torch.einsum("giop, gipjn -> giojn", cell_W[:, c], W)

        return W.reshape(G, NM, NM)

    def materialize_weight(self) -> torch.Tensor:
        """Return W ∈ R^{dim × dim} such that this block computes W·x (without bias).

        Iterates from the leaves up: one batched R=0 dense-materialization
        followed by R rounds of ``_compose_one_level``.
        """
        cur = self._vectorized_r0_dense(self._all_leaves())             # [G_total, N, N]

        M = self.n
        N = self.n
        for _ in range(self.r):
            G = cur.shape[0] // (N * N)
            cur = cur.reshape(G, N * N, M, M)
            cur = self._compose_one_level(cur, M)                       # [G, NM, NM]
            M = M * N
        return cur.squeeze(0)                                           # [dim, dim]

    # ──────────────────────────────────────────────────────────────────────
    # Caching: rollout reuses one materialization across many batch=1
    # forwards. Bypassed when grad is enabled so backprop sees the autograd
    # graph through cells.
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def cache_dense_weight(self) -> None:
        """Materialize W and stash it for subsequent forward() calls.

        Call this at the start of a rollout. Subsequent forwards under
        ``torch.no_grad()`` reuse the cached W. Call ``clear_dense_weight_cache``
        before parameter updates to free memory; the cache is bypassed
        automatically when grad is enabled, so it cannot poison backprop.
        """
        self._W_cache = self.materialize_weight().detach()

    def clear_dense_weight_cache(self) -> None:
        self._W_cache = None

    def _resolve_weight(self) -> torch.Tensor:
        """Return cached W when usable, otherwise materialize fresh."""
        if self._W_cache is not None and not torch.is_grad_enabled():
            return self._W_cache
        return self.materialize_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"NeuralBlock expected last dim {self.dim} (n={self.n}, r={self.r}), "
                f"got {tuple(x.shape)}"
            )
        W = self._resolve_weight()
        out = torch.nn.functional.linear(x, W)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        return f"n={self.n}, d={self.d}, r={self.r}, dim={self.dim}"


# ──────────────────────────────────────────────────────────────────────────
# Module-tree helpers — call from training loops to manage NB caches across
# all NeuralBlock submodules of a larger model.
# ──────────────────────────────────────────────────────────────────────────


def cache_neural_block_weights(module: nn.Module) -> None:
    """Walk ``module`` and pre-materialize the dense W of every NeuralBlock.

    Call before a no-grad rollout where parameters won't change. Cuts the
    per-step cost from ``materialize + linear`` to just ``linear``.
    """
    for m in module.modules():
        if isinstance(m, NeuralBlock):
            m.cache_dense_weight()


def clear_neural_block_caches(module: nn.Module) -> None:
    """Drop cached W tensors on every NeuralBlock in ``module``."""
    for m in module.modules():
        if isinstance(m, NeuralBlock):
            m.clear_dense_weight_cache()
