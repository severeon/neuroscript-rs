"""Hyper-Connection primitives for NeuroScript.

Reference: Zhu et al., "Hyper-Connections" (ICLR 2025) -- arXiv:2409.19606v3

These modules implement the core hyper-connection operations:
- HyperExpand: Expand single hidden to n copies (network entry)
- HyperCollapse: Collapse n copies via sum (network exit)
- HCWidth: Width connection (mix n hidden vectors, extract layer input + state)
- HCDepth: Depth connection (merge layer output back into hyper state)
"""

import torch
import torch.nn as nn


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
            # rows 1..n+1 are for state pass-through
            self.proj = nn.Linear(dim, (n + 1) * n, bias=False)
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
            # Dynamic mode: zero-initialize projection weights so initial
            # alpha = tanh(0) = 0 (uniform zero mixing). Training moves
            # weights away from zero to learn the optimal mixing pattern.
            with torch.no_grad():
                self.proj.weight.zero_()
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
        assert n == self.n, f"Expected n={self.n} but got {n}"
        assert dim == self.dim, f"Expected dim={self.dim} but got {dim}"

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
            self.proj = nn.Linear(dim, n, bias=False)
        else:
            # Static: learnable weight vector of length n
            self.beta = nn.Parameter(torch.ones(n))

        self._init_weights()

    def _init_weights(self):
        """Initialize to standard residual (add layer output equally)."""
        if self.dynamic:
            with torch.no_grad():
                self.proj.weight.zero_()
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
