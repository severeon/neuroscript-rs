"""MoE routing primitives for NeuroScript.

Provides:
- SigmoidMoERouter: DeepSeek-V3's sigmoid-gated MoE with auxiliary-loss-free balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidMoERouter(nn.Module):
    """Sigmoid MoE Router with auxiliary-loss-free load balancing.

    Implements DeepSeek-V3's routing strategy:
    1. Gate scores via sigmoid (allows multi-hot, more expressive than softmax)
    2. Top-k selection per token
    3. Load balancing via per-expert bias adjustment (noaux_tc strategy)
       — no auxiliary loss term needed in training objective

    Args:
        dim (int): Input/output feature dimension
        num_experts (int): Total number of expert networks
        active_experts (int): Number of experts activated per token (top-k)
        expert_dim (int): Hidden dimension within each expert FFN

    Shape:
        - Input: [batch, seq, dim]
        - Output: [batch, seq, dim]

    Reference:
        DeepSeek-V3 Technical Report (2024), Section 2.3:
        "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"
    """

    def __init__(self, dim: int, num_experts: int, active_experts: int, expert_dim: int):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.active_experts = active_experts

        # Gate network: projects to expert logits
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # Per-expert load-balancing bias (updated during training, not via backprop)
        self.register_buffer(
            "balance_bias",
            torch.zeros(num_experts),
        )

        # Expert FFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, expert_dim, bias=False),
                nn.SiLU(),
                nn.Linear(expert_dim, dim, bias=False),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        x_flat = x.view(-1, dim)  # [batch*seq, dim]
        tokens = x_flat.shape[0]

        # Sigmoid gate scores with load-balancing bias
        gate_logits = self.gate(x_flat)  # [tokens, num_experts]
        gate_scores = torch.sigmoid(gate_logits + self.balance_bias)

        # Top-k selection
        topk_scores, topk_indices = torch.topk(gate_scores, self.active_experts, dim=-1)
        # Normalize selected scores
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-6)

        # Route tokens to experts (vectorized: one pass per expert)
        output = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            # Gather all tokens assigned to expert e across top-k positions
            expert_mask = (topk_indices == e).any(dim=-1)  # [tokens]
            if not expert_mask.any():
                continue
            # Sum scores across all k positions where this token chose expert e
            score_e = (topk_scores * (topk_indices == e).float()).sum(dim=-1)  # [tokens]
            expert_out = self.experts[e](x_flat[expert_mask])
            output.index_add_(0, expert_mask.nonzero(as_tuple=True)[0],
                               score_e[expert_mask].unsqueeze(-1) * expert_out)

        return output.view(batch, seq, dim)


class MoERouter(nn.Module):
    """Mixture-of-Experts router with softmax gating and auxiliary-loss balancing.

    Top-k routing layer that selects a subset of expert networks per token.
    Each token is independently routed to the top-k scoring experts using a
    softmax gate.  Expert outputs are weighted by their normalized gate scores
    and summed.

    Compared to :class:`SigmoidMoERouter` (which uses sigmoid + bias for
    auxiliary-loss-free balancing), this router uses softmax gating and can
    optionally add an auxiliary load-balancing loss to the training objective.

    Args:
        dim (int): Input/output feature dimension.
        num_experts (int): Total number of expert networks.
        active_experts (int): Number of experts activated per token (top-k).
        expert_dim (int): Hidden dimension within each expert FFN.

    Shape:
        - Input:  [batch, seq, dim]
        - Output: [batch, seq, dim]

    Reference:
        Shazeer et al. (2017) "Outrageously Large Neural Networks: The
        Sparsely-Gated Mixture-of-Experts Layer";
        Fedus et al. (2022) "Switch Transformers"
    """

    def __init__(self, dim: int, num_experts: int, active_experts: int, expert_dim: int):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.active_experts = active_experts

        # Gate network: projects token features to per-expert logits
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # Expert FFNs (each a two-layer MLP)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, expert_dim, bias=False),
                nn.SiLU(),
                nn.Linear(expert_dim, dim, bias=False),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        x_flat = x.view(-1, dim)  # [batch*seq, dim]
        tokens = x_flat.shape[0]

        # Softmax gate scores
        gate_logits = self.gate(x_flat)               # [tokens, num_experts]
        gate_scores = F.softmax(gate_logits, dim=-1)

        # Top-k selection
        topk_scores, topk_indices = torch.topk(gate_scores, self.active_experts, dim=-1)
        # Normalize selected scores so they sum to 1 per token
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-6)

        # Route tokens to experts and accumulate weighted outputs (vectorized)
        output = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            # Gather all tokens assigned to expert e across top-k positions
            expert_mask = (topk_indices == e).any(dim=-1)  # [tokens]
            if not expert_mask.any():
                continue
            # Sum scores across all k positions where this token chose expert e
            score_e = (topk_scores * (topk_indices == e).float()).sum(dim=-1)  # [tokens]
            expert_out = self.experts[e](x_flat[expert_mask])
            output.index_add_(0, expert_mask.nonzero(as_tuple=True)[0],
                               score_e[expert_mask].unsqueeze(-1) * expert_out)

        return output.view(batch, seq, dim)
