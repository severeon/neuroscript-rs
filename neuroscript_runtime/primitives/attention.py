"""
Attention mechanisms for NeuroScript.

Provides primitives for:
- ScaledDotProductAttention: Base attention operation used in transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
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

    def __init__(self, dropout_p: float = 0.0, scale: Optional[float] = None):
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


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention primitive.

    Implements the multi-head attention mechanism from "Attention is All You Need".
    Projects input to Q, K, V, splits into multiple heads, applies attention per head,
    then concatenates and projects the output.

    NeuroScript signature:
        neuron MultiHeadSelfAttention(d_model, num_heads):
            in: [*batch, seq, d_model]
            out: [*batch, seq, d_model]
            impl: neuroscript_runtime.primitives.MultiHeadSelfAttention

    Args:
        d_model (int): Model dimension (must be divisible by num_heads)
        num_heads (int): Number of attention heads
        dropout_p (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to use bias in linear projections. Default: True

    Shape:
        - Input: [*batch, seq, d_model]
        - Output: [*batch, seq, d_model]

    Notes:
        - d_model must be divisible by num_heads
        - Each head has dimension d_k = d_model / num_heads
        - Uses learned linear projections for Q, K, V and output
        - Attention weights are computed per head independently

    Example:
        >>> mha = MultiHeadSelfAttention(d_model=512, num_heads=8)
        >>> x = torch.randn(32, 10, 512)  # batch=32, seq=10, d_model=512
        >>> output = mha(x)
        >>> assert output.shape == (32, 10, 512)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_p: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_p = dropout_p

        # Linear projections for Q, K, V (combined for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head self-attention.

        Args:
            input: Input tensor [*batch, seq, d_model]

        Returns:
            Attention output [*batch, seq, d_model]

        Raises:
            ValueError: If input shape is incompatible
        """
        if input.ndim < 2:
            raise ValueError(
                f"Input must have at least 2 dimensions (seq, d_model), got {input.ndim}"
            )

        if input.shape[-1] != self.d_model:
            raise ValueError(
                f"Input last dimension must be {self.d_model}, got {input.shape[-1]}"
            )

        # Extract dimensions
        # input: [*batch, seq, d_model]
        *batch_dims, seq_len, _ = input.shape
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim

        # Project to Q, K, V
        # qkv: [*batch, seq, 3 * d_model]
        qkv = self.qkv_proj(input)

        # Reshape and split into Q, K, V
        # qkv: [batch_prod, seq, 3, num_heads, d_k]
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.d_k)

        # Permute to [3, batch_prod, num_heads, seq, d_k]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # Split into Q, K, V: each [batch_prod, num_heads, seq, d_k]
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Compute attention scores: Q @ K^T
        # scores: [batch_prod, num_heads, seq_q, seq_k]
        scores = torch.matmul(query, key.transpose(-2, -1))

        # Scale by 1/sqrt(d_k)
        scores = scores / math.sqrt(self.d_k)

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout to attention weights
        if self.dropout is not None and self.training:
            attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # attn_weights: [batch_prod, num_heads, seq, seq]
        # value: [batch_prod, num_heads, seq, d_k]
        # attn_output: [batch_prod, num_heads, seq, d_k]
        attn_output = torch.matmul(attn_weights, value)

        # Transpose and reshape to concatenate heads
        # attn_output: [batch_prod, seq, num_heads, d_k]
        attn_output = attn_output.transpose(1, 2)

        # Concatenate heads: [batch_prod, seq, d_model]
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.d_model)

        # Restore original batch dimensions: [*batch, seq, d_model]
        if batch_dims:
            attn_output = attn_output.view(*batch_dims, seq_len, self.d_model)

        # Output projection
        output = self.out_proj(attn_output)

        return output


__all__ = ["ScaledDotProductAttention", "MultiHeadSelfAttention"]
