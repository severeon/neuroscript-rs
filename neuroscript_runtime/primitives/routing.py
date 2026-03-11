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

        # Route tokens to experts
        output = torch.zeros_like(x_flat)
        for k in range(self.active_experts):
            expert_idx = topk_indices[:, k]  # [tokens]
            scores_k = topk_scores[:, k]     # [tokens]
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_out = self.experts[e](x_flat[mask])
                    output[mask] += scores_k[mask].unsqueeze(-1) * expert_out

        return output.view(batch, seq, dim)
