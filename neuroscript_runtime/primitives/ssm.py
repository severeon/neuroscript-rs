"""Selective State Space Model (SSM) primitives for NeuroScript.

Provides:
- MambaBlock: Selective SSM layer (Mamba / Mamba-2)

Reference:
    Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    arXiv:2312.00752
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """Selective State Space Model (SSM) block.

    Processes sequences in O(n) time using a recurrent SSM formulation with
    input-dependent ("selective") state transitions, as introduced in the Mamba
    paper.  The key innovation over prior SSMs (S4, H3) is that A, B, C, and
    delta are all functions of the input, enabling content-based filtering.

    Internal structure (simplified Mamba-1):
        1. Input projection: Linear(dim, 2 * expand * dim) -> split into x, z
        2. Short convolution: Conv1d(d_conv) on x
        3. SSM: x -> (A, B, C, delta) via linear projections -> selective_scan -> y
        4. Gating: y * SiLU(z)
        5. Output projection: Linear(expand * dim, dim)

    .. note::
        This is a structurally-correct stub.  For a production-quality,
        hardware-efficient implementation see the official ``mamba-ssm`` package
        (``pip install mamba-ssm``), which ships a fused CUDA kernel for the
        selective scan.

    Args:
        dim (int): Model hidden dimension.
        d_state (int): SSM state dimension. Default: 128.
        d_conv (int): Local convolution width. Default: 4.
        num_heads (int): Number of parallel SSM heads (Mamba-2). Default: 1.
        expand (int): Inner expansion factor. Default: 2.

    Shape:
        - Input:  [batch, seq, dim]
        - Output: [batch, seq, dim]

    Reference:
        Gu & Dao (2023) arXiv:2312.00752;
        Dao & Gu (2024) "Transformers are SSMs" arXiv:2405.21060 (Mamba-2)
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 128,
        d_conv: int = 4,
        num_heads: int = 1,
        expand: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_heads = num_heads
        self.expand = expand
        self.d_inner = expand * dim

        # Input projection: projects to inner dimension (x) and gate (z)
        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # Short depthwise convolution on the x branch
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameter projections (input-dependent: "selective")
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + num_heads, bias=False)
        self.dt_proj = nn.Linear(num_heads, self.d_inner, bias=True)

        # State transition matrix A (log-parameterized for stability)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .repeat(num_heads, 1)
        )

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        # Output normalization
        self.norm = nn.RMSNorm(self.d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Selective SSM forward pass (simplified — no fused kernel).

        Args:
            x: [batch, seq, dim]

        Returns:
            [batch, seq, dim]
        """
        batch, seq, dim = x.shape

        # Input projection -> split into inner activations and gate
        xz = self.in_proj(x)                       # [batch, seq, 2*d_inner]
        x_inner, z = xz.chunk(2, dim=-1)           # each [batch, seq, d_inner]

        # Short depthwise convolution (causal: trim the acausal future padding)
        x_conv = x_inner.transpose(1, 2)           # [batch, d_inner, seq]
        x_conv = self.conv1d(x_conv)[..., :seq]    # [batch, d_inner, seq]
        x_conv = x_conv.transpose(1, 2)            # [batch, seq, d_inner]
        x_conv = F.silu(x_conv)

        # SSM parameter projections (input-dependent)
        x_dbl = self.x_proj(x_conv)                # [batch, seq, d_state*2 + num_heads]
        dt, B, C = x_dbl.split(
            [self.num_heads, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))          # [batch, seq, d_inner]

        # Simplified selective scan (sequential; production uses fused CUDA kernel)
        A = -torch.exp(self.A_log.float())         # [num_heads, d_state]
        y = self._selective_scan(x_conv, dt, A, B, C)

        # Gate and normalize
        y = self.norm(y)
        y = y * F.silu(z)

        return self.out_proj(y)

    def _selective_scan(
        self,
        u: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Naive sequential selective scan for correctness.

        Production implementations replace this with a parallel/fused scan.
        """
        batch, seq, d_inner = u.shape
        d_state = self.d_state

        # Zero initial state
        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []

        for t in range(seq):
            # Discretize: delta_A = exp(dt[t] * A), delta_B = dt[t] * B[t]
            dt_t = dt[:, t, :].unsqueeze(-1)           # [batch, d_inner, 1]
            A_broad = A.mean(0).unsqueeze(0)            # [1, d_state] (simplified)
            dA = torch.exp(dt_t * A_broad.unsqueeze(0)) # [batch, d_inner, d_state]

            B_t = B[:, t, :].unsqueeze(1)              # [batch, 1, d_state]
            dB = dt_t * B_t                            # [batch, d_inner, d_state]

            u_t = u[:, t, :].unsqueeze(-1)             # [batch, d_inner, 1]

            # State update: h = dA * h + dB * u
            h = dA * h + dB * u_t

            # Output: y = C @ h  (simplified)
            C_t = C[:, t, :]                           # [batch, d_state]
            y_t = (h * C_t.unsqueeze(1)).sum(-1)       # [batch, d_inner]
            ys.append(y_t)

        return torch.stack(ys, dim=1)                  # [batch, seq, d_inner]

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, d_state={self.d_state}, "
            f"d_conv={self.d_conv}, num_heads={self.num_heads}, expand={self.expand}"
        )


__all__ = ["MambaBlock"]
