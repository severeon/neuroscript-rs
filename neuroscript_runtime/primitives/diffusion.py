"""Diffusion primitives for NeuroScript.

Provides primitives for masked diffusion language models:
- DenoisingHead: Maps hidden states to token logits for unmasking
- MultiTokenPredictionHead: Predicts N future tokens simultaneously
"""

import torch
import torch.nn as nn


class DenoisingHead(nn.Module):
    """Prediction head for masked diffusion language models.

    Maps hidden states [batch, seq, dim] to token logits [batch, seq, vocab_size].
    Used to predict the original tokens from noisy/masked hidden representations.

    Args:
        dim (int): Hidden dimension of input
        vocab_size (int): Vocabulary size for prediction

    Shape:
        - Input: [batch, seq, dim]
        - Output: [batch, seq, vocab_size]
    """

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MultiTokenPredictionHead(nn.Module):
    """Predicts N future tokens simultaneously.

    Shares a trunk projection then applies N independent prediction heads,
    enabling multi-token prediction objectives for faster training and inference.

    Args:
        dim (int): Input hidden dimension
        vocab_size (int): Vocabulary size
        num_tokens (int): Number of future tokens to predict. Default: 4

    Shape:
        - Input: [batch, seq, dim]
        - Output: [batch, seq, num_tokens, vocab_size]
    """

    def __init__(self, dim: int, vocab_size: int, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        # Independent projection per prediction position
        self.heads = nn.ModuleList([
            nn.Linear(dim, vocab_size, bias=False)
            for _ in range(num_tokens)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, dim]
        outputs = [head(x) for head in self.heads]  # each: [batch, seq, vocab_size]
        return torch.stack(outputs, dim=2)  # [batch, seq, num_tokens, vocab_size]
