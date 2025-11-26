"""
Attention mechanisms for NeuroScript.

Provides primitives for:
- ScaledDotProductAttention: Base attention operation used in transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention primitive.

    Implements the core attention mechanism:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    This is the fundamental building block of transformer architectures,
    computing attention weights between queries and keys, then using those
    weights to aggregate values.

    NeuroScript signature:
        neuron ScaledDotProductAttention:
            in query: [*batch, seq_q, d_k]
            in key: [*batch, seq_k, d_k]
            in value: [*batch, seq_k, d_v]
            out: [*batch, seq_q, d_v]
            impl: neuroscript_runtime.primitives.ScaledDotProductAttention

    Args:
        dropout_p (float, optional): Dropout probability for attention weights.
            Default: 0.0 (no dropout)
        scale (float, optional): Explicit scale factor. If None, uses 1/sqrt(d_k).
            Default: None

    Shape:
        - Query: [*batch, seq_q, d_k] where * means any number of batch dimensions
        - Key: [*batch, seq_k, d_k]
        - Value: [*batch, seq_k, d_v]
        - Output: [*batch, seq_q, d_v]

    Notes:
        - d_k is the key/query dimension (must match between Q and K)
        - d_v is the value dimension (can differ from d_k)
        - seq_k and seq_v must match (key and value sequence lengths)
        - Scaling by 1/sqrt(d_k) prevents dot products from growing too large
        - Handles arbitrary batch dimensions (single batch, multi-head, etc.)

    Example:
        >>> attn = ScaledDotProductAttention()
        >>> q = torch.randn(32, 10, 64)  # batch=32, seq_q=10, d_k=64
        >>> k = torch.randn(32, 20, 64)  # batch=32, seq_k=20, d_k=64
        >>> v = torch.randn(32, 20, 128) # batch=32, seq_k=20, d_v=128
        >>> output = attn((q, k, v))
        >>> assert output.shape == (32, 10, 128)  # [batch, seq_q, d_v]
    """

    def __init__(self, dropout_p: float = 0.0, scale: float = None):
        super().__init__()

        if dropout_p < 0.0 or dropout_p >= 1.0:
            raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")

        self.dropout_p = dropout_p
        self.scale = scale
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None

    def forward(
        self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention.

        Args:
            inputs: Tuple of (query, key, value) tensors

        Returns:
            Attention output tensor

        Raises:
            ValueError: If input shapes are incompatible
        """
        query, key, value = inputs

        # Validate input dimensions
        if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
            raise ValueError(
                f"Inputs must have at least 2 dimensions (seq, dim), "
                f"got query.ndim={query.ndim}, key.ndim={key.ndim}, value.ndim={value.ndim}"
            )

        # Extract dimensions
        # query: [*batch, seq_q, d_k]
        # key:   [*batch, seq_k, d_k]
        # value: [*batch, seq_k, d_v]
        d_k = query.shape[-1]
        d_k_key = key.shape[-1]
        seq_k = key.shape[-2]
        seq_v = value.shape[-2]

        # Validate query and key dimensions match
        if d_k != d_k_key:
            raise ValueError(
                f"Query and key must have the same last dimension (d_k), "
                f"got query: {query.shape}, key: {key.shape}"
            )

        # Validate key and value sequence lengths match
        if seq_k != seq_v:
            raise ValueError(
                f"Key and value must have the same sequence length, "
                f"got key: {key.shape[-2]}, value: {value.shape[-2]}"
            )

        # Validate batch dimensions are broadcastable
        try:
            # This will raise if batch dimensions aren't compatible
            torch.broadcast_shapes(query.shape[:-2], key.shape[:-2], value.shape[:-2])
        except RuntimeError as e:
            raise ValueError(
                f"Batch dimensions must be broadcastable, "
                f"got query: {query.shape}, key: {key.shape}, value: {value.shape}: {e}"
            )

        # Compute attention scores: Q @ K^T
        # query: [*batch, seq_q, d_k]
        # key^T: [*batch, d_k, seq_k]
        # scores: [*batch, seq_q, seq_k]
        scores = torch.matmul(query, key.transpose(-2, -1))

        # Scale by 1/sqrt(d_k) to prevent vanishing gradients
        scale_factor = self.scale if self.scale is not None else 1.0 / math.sqrt(d_k)
        scores = scores * scale_factor

        # Apply softmax to get attention weights
        # Softmax over the key dimension (last dimension)
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout to attention weights if specified
        if self.dropout is not None and self.training:
            attn_weights = self.dropout(attn_weights)

        # Apply attention weights to values
        # attn_weights: [*batch, seq_q, seq_k]
        # value: [*batch, seq_k, d_v]
        # output: [*batch, seq_q, d_v]
        output = torch.matmul(attn_weights, value)

        return output


__all__ = ["ScaledDotProductAttention"]
