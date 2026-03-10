"""Hyper-Connection primitives for NeuroScript.

References:
- Zhu et al., "Hyper-Connections" (ICLR 2025) -- arXiv:2409.19606v3
- Xie et al., "mHC: Manifold-Constrained Hyper-Connections" (2026) -- arXiv:2512.24880v2

These modules implement the core hyper-connection operations:
- HyperExpand: Expand single hidden to n copies (network entry)
- HyperCollapse: Collapse n copies via sum (network exit)
- HCWidth: Width connection (mix n hidden vectors, extract layer input + state)
- HCDepth: Depth connection (merge layer output back into hyper state)
- ManifoldHyperConnect: mHC wrapper with Sinkhorn-Knopp constrained residual mixing
"""

import math
from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperExpand(nn.Module):
    """Expand a single hidden vector to n copies along a new dimension.

    Input:  [*batch, dim]
    Output: [*batch, n, dim]
    """
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        # x: [*batch, dim] -> [*batch, 1, dim] -> [*batch, n, dim]
        return x.unsqueeze(-2).expand(*x.shape[:-1], self.n, x.shape[-1]).contiguous()


class HyperCollapse(nn.Module):
    """Collapse n copies by summing along the expansion dimension.

    Input:  [*batch, n, dim]
    Output: [*batch, dim]
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [*batch, n, dim] -> [*batch, dim]
        return x.sum(dim=-2)


class HCWidth(nn.Module):
    """Width connection: mix n hidden vectors and extract layer input + state.

    Implements Equations 10-13 (dynamic) or direct matrix multiply (static)
    from Algorithm 2 of the paper.

    Input:  [*batch, n, dim]
    Output: (layer_in: [*batch, dim], state: [*batch, n, dim])

    Args:
        n: Number of hidden copies (expansion rate)
        dim: Hidden dimension
        layer_idx: Layer index (for round-robin initialization per Eq. 14)
        dynamic: If True, use input-dependent mixing (Eqs. 10-13).
                 If False, use static learned matrix.
    """
    def __init__(self, n, dim, layer_idx, dynamic=True):
        super().__init__()
        self.n = n
        self.dim = dim
        self.layer_idx = layer_idx
        self.dynamic = dynamic

        if dynamic:
            # Dynamic: project input to mixing weights (Eq. 10-11)
            # alpha has shape (n+1, n) -- row 0 is for layer_in extraction,
            # rows 1..n+1 are for state pass-through.
            # Bias is used to initialize the round-robin pattern (Eq. 14)
            # so that tanh(bias) approximates the static identity init.
            self.proj = nn.Linear(dim, (n + 1) * n, bias=True)
        else:
            # Static: learnable (n+1) x n matrix
            self.alpha = nn.Parameter(torch.zeros(n + 1, n))

        # Initialize per Eq. 14: round-robin based on layer_idx
        self._init_weights()

    def _init_weights(self):
        """Initialize to Pre-Norm residual equivalence (Eq. 14)."""
        n = self.n
        # Am initialization: round-robin assignment
        # For layer m, the "active" copy index is m mod n
        active = self.layer_idx % n

        if self.dynamic:
            # Dynamic mode: zero-initialize weights so the projection
            # output depends only on the bias at init. The bias is set
            # so that tanh(bias) approximates the static round-robin
            # identity matrix from Eq. 14, ensuring the wrapped layer
            # receives a meaningful signal on the first forward pass.
            init_val = math.atanh(0.9)  # ~1.47, so tanh(init_val) ≈ 0.9
            with torch.no_grad():
                self.proj.weight.zero_()
                bias = torch.zeros((n + 1) * n)
                # Row 0 (layer input): extract from active copy
                bias[0 * n + active] = init_val
                # Rows 1..n+1 (state): identity pass-through
                for i in range(n):
                    bias[(i + 1) * n + i] = init_val
                self.proj.bias.copy_(bias)
        else:
            with torch.no_grad():
                self.alpha.zero_()
                # Row 0 (layer input extraction): take from active copy
                self.alpha[0, active] = 1.0
                # Rows 1..n+1 (state): identity pass-through
                for i in range(n):
                    self.alpha[i + 1, i] = 1.0

    def forward(self, x):
        # x: [*batch, n, dim]
        batch_shape = x.shape[:-2]
        n, dim = x.shape[-2], x.shape[-1]
        if n != self.n:
            raise ValueError(f"Expected n={self.n} but got {n}")
        if dim != self.dim:
            raise ValueError(f"Expected dim={self.dim} but got {dim}")

        if self.dynamic:
            # Eq. 10-11: compute input-dependent mixing weights
            # Average across copies to get a summary
            x_mean = x.mean(dim=-2)  # [*batch, dim]
            raw = self.proj(x_mean)  # [*batch, (n+1)*n]
            raw = raw.view(*batch_shape, n + 1, n)
            alpha = torch.tanh(raw)  # Eq. 12: stabilize with tanh
        else:
            alpha = self.alpha  # [n+1, n]

        # Eq. 13: apply mixing
        # alpha[0, :] extracts layer input, alpha[1:, :] produces state
        # layer_in = sum_j(alpha[0,j] * x[..., j, :])
        layer_in = torch.einsum('...jd, ...j -> ...d', x, alpha[..., 0, :])

        # state[i] = sum_j(alpha[i+1, j] * x[..., j, :])
        state = torch.einsum('...jd, ...ij -> ...id', x, alpha[..., 1:, :])

        return layer_in, state


class HCDepth(nn.Module):
    """Depth connection: merge layer output back into hyper hidden state.

    Implements Equation 5 from the paper.

    Input:  (layer_out: [*batch, dim], state: [*batch, n, dim])
    Output: [*batch, n, dim]

    Args:
        n: Number of hidden copies
        dim: Hidden dimension
        dynamic: If True, use input-dependent depth weights.
                 If False, use static learned weights.
    """
    def __init__(self, n, dim, dynamic=True):
        super().__init__()
        self.n = n
        self.dim = dim
        self.dynamic = dynamic

        if dynamic:
            self.proj = nn.Linear(dim, n, bias=True)
        else:
            # Static: learnable weight vector of length n
            self.beta = nn.Parameter(torch.ones(n))

        self._init_weights()

    def _init_weights(self):
        """Initialize to standard residual (add layer output equally).

        Static mode: beta = 1.0 for all copies.
        Dynamic mode: zero weights + atanh(0.9) bias so that
        tanh(bias) ≈ 0.9 ≈ 1.0, matching the static initialisation
        and ensuring the layer output contributes on the first forward pass.
        """
        if self.dynamic:
            init_val = math.atanh(0.9)  # ~1.47, so tanh(init_val) ≈ 0.9
            with torch.no_grad():
                self.proj.weight.zero_()
                self.proj.bias.fill_(init_val)
        else:
            with torch.no_grad():
                self.beta.fill_(1.0)

    def forward(self, inputs):
        layer_out, state = inputs
        # layer_out: [*batch, dim], state: [*batch, n, dim]

        if self.dynamic:
            beta = torch.tanh(self.proj(layer_out))  # [*batch, n]
        else:
            beta = self.beta  # [n]

        # Eq. 5: h_out[i] = state[i] + beta[i] * layer_out
        # Expand layer_out: [*batch, dim] -> [*batch, 1, dim]
        layer_expanded = layer_out.unsqueeze(-2)
        # Expand beta: [*batch, n] -> [*batch, n, 1]
        if self.dynamic:
            beta_expanded = beta.unsqueeze(-1)
        else:
            beta_expanded = beta.view(*([1] * (state.dim() - 2)), self.n, 1)

        return state + beta_expanded * layer_expanded


# ---------------------------------------------------------------------------
# Manifold-Constrained Hyper-Connections (mHC)
# ---------------------------------------------------------------------------
# Reference: Xie et al. (2026) arXiv:2512.24880v2 (DeepSeek-AI)
#
# Key innovation: project H_res onto the Birkhoff polytope (doubly stochastic
# matrices) via Sinkhorn-Knopp normalization. This guarantees:
#   1. Spectral norm <= 1 (non-expansive)
#   2. Closed under matrix multiplication (stability at any depth)
#   3. Signal mean preservation across layers
# ---------------------------------------------------------------------------


def sinkhorn_knopp(log_alpha: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Project a matrix onto the Birkhoff polytope (doubly stochastic matrices).

    Uses the Sinkhorn-Knopp algorithm: alternately normalize rows and columns
    of a positive matrix until convergence to a doubly stochastic matrix.

    Args:
        log_alpha: Raw unconstrained matrix [*, n, n]. We exponentiate first
                   to ensure positivity (Eq. 9 in paper).
        iters: Number of normalization iterations (paper uses 20).

    Returns:
        Doubly stochastic matrix [*, n, n] where rows and columns sum to 1.
    """
    # Eq. 9: M^(0) = exp(raw) — ensures all entries are positive
    M = torch.exp(log_alpha)

    for _ in range(iters):
        # Row normalization: T_r
        M = M / (M.sum(dim=-1, keepdim=True) + 1e-8)
        # Column normalization: T_c
        M = M / (M.sum(dim=-2, keepdim=True) + 1e-8)

    return M


class ManifoldHyperConnect(nn.Module):
    """Manifold-Constrained Hyper-Connection (mHC).

    Wraps a sublayer (attention or FFN) with an n-stream residual that uses
    doubly stochastic mixing matrices for provably stable signal propagation.

    The forward pass:
        1. Compute H_pre, H_post, H_res from the flattened input (Eq. 7-8)
        2. Extract layer input via H_pre: layer_in = H_pre @ x_streams
        3. Run sublayer: layer_out = sublayer(layer_in)
        4. Merge output via H_post: contribution = H_post^T @ layer_out
        5. Mix residual via H_res: x_out = H_res @ x_streams + contribution

    H_res is constrained to be doubly stochastic via Sinkhorn-Knopp (Eq. 8-9).
    H_pre and H_post are constrained to be non-negative via sigmoid (Eq. 8).

    Args:
        sublayer: The nn.Module to wrap (attention, FFN, etc.)
        n: Expansion factor for residual stream width (default: 4)
        dim: Model hidden dimension
        layer_idx: Layer index for per-layer parameter initialization
        alpha_init: Gating factor initialization (default: 0.01 per paper)
        sinkhorn_iters: Number of Sinkhorn-Knopp iterations (default: 20)
        dynamic: Use input-dependent mappings (default: True)

    Shape:
        Input:  [batch, seq, n, dim] — n-stream residual
        Output: [batch, seq, n, dim] — updated n-stream residual
    """

    def __init__(
        self,
        sublayer: nn.Module,
        n: int = 4,
        dim: int = 768,
        layer_idx: int = 0,
        alpha_init: float = 0.01,
        sinkhorn_iters: int = 20,
        dynamic: bool = True,
    ):
        super().__init__()
        self.sublayer = sublayer
        self.n = n
        self.dim = dim
        self.layer_idx = layer_idx
        self.sinkhorn_iters = sinkhorn_iters
        self.dynamic = dynamic

        nC = n * dim  # flattened stream dimension

        # --- Gating scalars (Eq. 7): alpha_pre, alpha_post, alpha_res ---
        # Initialized to alpha_init (small) so mHC starts near identity
        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init))

        if dynamic:
            # Dynamic projections (Eq. 7): input-dependent mappings
            # phi_pre: [nC -> n] for H_pre (aggregates n streams to 1)
            self.phi_pre = nn.Linear(nC, n, bias=False)
            # phi_post: [nC -> n] for H_post (distributes 1 output to n streams)
            self.phi_post = nn.Linear(nC, n, bias=False)
            # phi_res: [nC -> n*n] for H_res (n×n mixing matrix)
            self.phi_res = nn.Linear(nC, n * n, bias=False)
        else:
            # Static mappings: learnable vectors/matrices directly
            self.phi_pre = nn.Parameter(torch.zeros(n))
            self.phi_post = nn.Parameter(torch.zeros(n))
            self.phi_res = nn.Parameter(torch.zeros(n, n))

        # Static biases (Eq. 7): b_pre, b_post, b_res
        self.b_pre = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.zeros(n, n))

        # RMSNorm for the flattened input (Eq. 7)
        self.rms_norm = nn.RMSNorm(nC)

        self._init_weights()

    def _init_weights(self):
        """Initialize so mHC starts as near-identity residual connection."""
        with torch.no_grad():
            # H_pre bias: extract from stream (layer_idx % n) by default
            active = self.layer_idx % self.n
            self.b_pre[active] = 1.0

            # H_post bias: distribute back to all streams equally
            self.b_post.fill_(1.0 / self.n)

            # H_res bias: initialize near identity matrix
            # After exp + Sinkhorn, this converges to ~identity
            torch.nn.init.eye_(self.b_res)

            if self.dynamic:
                # Zero-init dynamic projections so initial behavior = static bias
                nn.init.zeros_(self.phi_pre.weight)
                nn.init.zeros_(self.phi_post.weight)
                nn.init.zeros_(self.phi_res.weight)

    def _compute_mappings(self, x_flat: torch.Tensor):
        """Compute H_pre, H_post, H_res from flattened input (Eq. 7-8).

        Args:
            x_flat: [batch, seq, n*dim] flattened n-stream input

        Returns:
            H_pre:  [batch, seq, n] — aggregation weights (non-negative)
            H_post: [batch, seq, n] — distribution weights (non-negative)
            H_res:  [batch, seq, n, n] — doubly stochastic mixing matrix
        """
        # Eq. 7: normalize input
        x_norm = self.rms_norm(x_flat)

        if self.dynamic:
            # Dynamic part: input-dependent projections
            raw_pre = self.alpha_pre * (x_norm @ self.phi_pre.weight.T) + self.b_pre
            raw_post = self.alpha_post * (x_norm @ self.phi_post.weight.T) + self.b_post
            raw_res_flat = self.alpha_res * (x_norm @ self.phi_res.weight.T)
            raw_res = raw_res_flat.view(*x_flat.shape[:-1], self.n, self.n) + self.b_res
        else:
            # Static: just bias terms scaled by alpha
            raw_pre = self.alpha_pre * self.phi_pre + self.b_pre
            raw_post = self.alpha_post * self.phi_post + self.b_post
            raw_res = self.alpha_res * self.phi_res + self.b_res

        # Eq. 8: apply manifold constraints
        H_pre = torch.sigmoid(raw_pre)          # non-negative
        H_post = 2.0 * torch.sigmoid(raw_post)  # non-negative, scaled by 2
        H_res = sinkhorn_knopp(raw_res, self.sinkhorn_iters)  # doubly stochastic

        return H_pre, H_post, H_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with manifold-constrained hyper-connections.

        Args:
            x: [batch, seq, n, dim] — n-stream residual input

        Returns:
            [batch, seq, n, dim] — updated n-stream residual
        """
        batch, seq, n, dim = x.shape
        assert n == self.n and dim == self.dim, \
            f"Expected [*, {self.n}, {self.dim}], got [*, {n}, {dim}]"

        # Flatten n streams: [batch, seq, n, dim] -> [batch, seq, n*dim]
        x_flat = x.reshape(batch, seq, n * dim)

        # Compute constrained mappings
        H_pre, H_post, H_res = self._compute_mappings(x_flat)

        # --- H_pre: aggregate n streams into layer input ---
        # H_pre: [batch, seq, n], x: [batch, seq, n, dim]
        # layer_in = sum_j(H_pre[j] * x[j])  ->  [batch, seq, dim]
        layer_in = torch.einsum('bsn, bsnd -> bsd', H_pre, x)

        # --- Run sublayer ---
        layer_out = self.sublayer(layer_in)  # [batch, seq, dim]

        # --- H_res: mix residual streams (doubly stochastic) ---
        # H_res: [batch, seq, n, n] or [n, n], x: [batch, seq, n, dim]
        if H_res.dim() == 2:
            # Static: [n, n]
            residual = torch.einsum('ij, bsjd -> bsid', H_res, x)
        else:
            # Dynamic: [batch, seq, n, n]
            residual = torch.einsum('bsij, bsjd -> bsid', H_res, x)

        # --- H_post: distribute layer output back to n streams ---
        # H_post: [batch, seq, n], layer_out: [batch, seq, dim]
        # contribution[i] = H_post[i] * layer_out  ->  [batch, seq, n, dim]
        contribution = torch.einsum('bsn, bsd -> bsnd', H_post, layer_out)

        # --- Combine: residual mixing + layer contribution ---
        return residual + contribution

    def extra_repr(self) -> str:
        return (
            f"n={self.n}, dim={self.dim}, layer_idx={self.layer_idx}, "
            f"sinkhorn_iters={self.sinkhorn_iters}, dynamic={self.dynamic}, "
            f"sublayer_params={sum(p.numel() for p in self.sublayer.parameters()):,}"
        )
