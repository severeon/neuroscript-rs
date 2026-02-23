"""
Embedding primitives.

Maps to NeuroScript neurons like:
    neuron Embedding(vocab_size: int, embedding_dim: int):
        in: [*, seq_len]  # integer indices
        out: [*, seq_len, embedding_dim]
        impl: neuroscript_runtime.primitives.Embedding
"""

from typing import Optional
import torch
import torch.nn as nn
import math


class Embedding(nn.Module):
    """
    Token embedding layer: discrete indices → dense vectors.

    Maps integer token IDs to dense continuous vectors. Fundamental building
    block for all sequence models (language, audio, etc.).

    Args:
        vocab_size: Size of vocabulary (number of unique tokens)
        embedding_dim: Dimension of embedding vectors
        padding_idx: If given, pads output with zeros at this index. Default: None
        max_norm: If given, renormalizes embeddings to have norm <= max_norm
        norm_type: Norm type for max_norm (2 = L2 norm). Default: 2.0
        scale_grad_by_freq: Scale gradients by token frequency. Default: False

    Shape:
        - Input: [*, seq_len] long tensor with token indices in [0, vocab_size)
        - Output: [*, seq_len, embedding_dim]

    Examples:
        >>> # 50,000 token vocabulary, 512-dimensional embeddings
        >>> embedding = Embedding(vocab_size=50000, embedding_dim=512)
        >>> tokens = torch.randint(0, 50000, (32, 128))  # [batch, seq_len]
        >>> embedded = embedding(tokens)  # [32, 128, 512]
        >>> embedded.shape
        torch.Size([32, 128, 512])

        >>> # With padding
        >>> embedding = Embedding(vocab_size=10000, embedding_dim=256, padding_idx=0)
        >>> tokens = torch.tensor([[1, 2, 0], [5, 0, 0]])  # 0 is padding
        >>> embedded = embedding(tokens)  # padding positions are zeros

    Notes:
        - Embedding matrix shape: [vocab_size, embedding_dim]
        - Input indices must be in range [0, vocab_size)
        - Typical vocab_size: 30k-100k for language models
        - Typical embedding_dim: 128-1024 depending on model size
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")

        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

        if padding_idx is not None:
            if not (0 <= padding_idx < vocab_size):
                raise ValueError(
                    f"padding_idx must be in [0, vocab_size), "
                    f"got {padding_idx} with vocab_size={vocab_size}"
                )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq

        # Use PyTorch's Embedding implementation
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Lookup embeddings for input token indices.

        Args:
            input: Long tensor [*, seq_len] with token indices

        Returns:
            Embedding tensor [*, seq_len, embedding_dim]

        Raises:
            ValueError: If input is not integer type
            RuntimeError: If input contains indices outside [0, vocab_size)
        """
        # Validate input dtype
        if not input.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            raise ValueError(
                f"Input must be integer type for embedding lookup, "
                f"got dtype={input.dtype}. Use .long() to convert."
            )

        # PyTorch will raise if indices are out of range, but we add
        # a clearer error message
        if input.numel() > 0:  # only check if not empty
            min_idx = input.min().item()
            max_idx = input.max().item()

            if min_idx < 0 or max_idx >= self.vocab_size:
                raise RuntimeError(
                    f"Input indices must be in [0, {self.vocab_size}), "
                    f"got range [{min_idx}, {max_idx}]"
                )

        return self.embedding(input)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        s = f"vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        if self.max_norm is not None:
            s += f", max_norm={self.max_norm}"
        return s

    @property
    def weight(self) -> torch.Tensor:
        """Embedding matrix [vocab_size, embedding_dim]."""
        return self.embedding.weight


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformers.

    Adds position information using sine/cosine functions of different frequencies.
    This is the "Attention is All You Need" (Vaswani et al. 2017) version.

    Args:
        d_model: Model dimension (must match embedding dimension)
        max_len: Maximum sequence length to precompute. Default: 5000
        dropout: Dropout probability. Default: 0.1

    Shape:
        - Input: [*, seq_len, d_model]
        - Output: [*, seq_len, d_model] (same shape, with positions added)

    Examples:
        >>> pos_enc = PositionalEncoding(d_model=512, max_len=1000)
        >>> x = torch.randn(32, 100, 512)  # [batch, seq_len, d_model]
        >>> x_with_pos = pos_enc(x)
        >>> x_with_pos.shape
        torch.Size([32, 100, 512])

    Notes:
        - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        - Precomputes encodings up to max_len
        - Registered as buffer (not a parameter, no gradients)
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # Precompute positional encodings [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer (moves with model to device, but not a parameter)
        self.register_buffer("pe", pe)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            input: Input tensor [*, seq_len, d_model]

        Returns:
            Output with positional encoding added [*, seq_len, d_model]

        Raises:
            ValueError: If last dim doesn't match d_model or seq_len > max_len
        """
        if input.size(-1) != self.d_model:
            raise ValueError(
                f"Input last dimension must be {self.d_model}, "
                f"got {input.size(-1)} (shape: {list(input.shape)})"
            )

        seq_len = input.size(-2)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.max_len}. "
                f"Increase max_len when creating PositionalEncoding."
            )

        # Add positional encoding (broadcasting over batch dims)
        # self.pe is [1, max_len, d_model], slice to [1, seq_len, d_model]
        x = input + self.pe[:, :seq_len, :]

        return self.dropout(x)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"d_model={self.d_model}, max_len={self.max_len}, dropout={self.dropout.p}"
        )


class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings.

    Alternative to sinusoidal encoding where positions are learned parameters.
    Used in BERT, GPT, and many modern transformers.

    Args:
        max_len: Maximum sequence length
        d_model: Model dimension

    Shape:
        - Input: [*, seq_len, d_model]
        - Output: [*, seq_len, d_model] (with learned positions added)

    Examples:
        >>> pos_emb = LearnedPositionalEmbedding(max_len=512, d_model=768)
        >>> x = torch.randn(32, 128, 768)  # [batch, seq, d_model]
        >>> x_with_pos = pos_emb(x)

    Notes:
        - Positions are learned during training
        - Requires seeing positions during training to generalize
        - Cannot extrapolate beyond max_len
    """

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()

        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")

        self.max_len = max_len
        self.d_model = d_model

        # Learnable position embeddings [max_len, d_model]
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional embeddings.

        Args:
            input: Input tensor [*, seq_len, d_model]

        Returns:
            Output with positions added [*, seq_len, d_model]

        Raises:
            ValueError: If seq_len > max_len or last dim != d_model
        """
        if input.size(-1) != self.d_model:
            raise ValueError(
                f"Input last dimension must be {self.d_model}, got {input.size(-1)}"
            )

        seq_len = input.size(-2)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.max_len}"
            )

        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.long, device=input.device)

        # Lookup position embeddings [seq_len, d_model]
        pos_emb = self.position_embeddings(positions)

        # Add to input (broadcasting over batch dims)
        return input + pos_emb.unsqueeze(0)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"max_len={self.max_len}, d_model={self.d_model}"


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Rotates query and key tensors to encode relative positional information.
    Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding".

    NeuroScript signature:
        neuron RotaryEmbedding(dim, max_position_embeddings=2048, base=10000.0):
            in query: [*batch, seq, num_heads, head_dim]
            in key: [*batch, seq, num_heads, head_dim]
            out q_out: [*batch, seq, num_heads, head_dim]
            out k_out: [*batch, seq, num_heads, head_dim]
            impl: neuroscript_runtime.primitives.embeddings.RotaryEmbedding

    Args:
        dim (int): Embedding dimension (head_dim).
        max_position_embeddings (int): Maximum sequence length to pre-compute. Default: 2048
        base (float): Base for the geometric progression of frequencies. Default: 10000.0
    """

    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Compute cos and sin cache
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build initial cache
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len, device=None, dtype=None):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # Different from paper, but common in implementations:
        # Concatenate freqs to match dimension
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key.

        Args:
            q: Query tensor [*batch, num_heads, seq_len, head_dim]
            k: Key tensor [*batch, num_heads, seq_len, head_dim]

        Returns:
            Tuple of rotated query and key
        """
        # q, k shape: [batch, num_heads, seq_len, head_dim]
        # or [batch, seq_len, num_heads, head_dim] depending on format
        # Standard implementation usually assumes [batch, num_heads, seq_len, head_dim]
        # But let's support generic last dim being head_dim

        seq_len = q.shape[-2]

        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=q.device, dtype=q.dtype)

        # Get cos and sin for current sequence length
        # cached shape: [1, 1, max_len, dim]
        # We slice to: [1, 1, seq_len, dim]
        cos = self.cos_cached[:, :, :seq_len, ...].to(dtype=q.dtype, device=q.device)
        sin = self.sin_cached[:, :, :seq_len, ...].to(dtype=q.dtype, device=q.device)

        return (
            self._apply_rotary_pos_emb(q, cos, sin),
            self._apply_rotary_pos_emb(k, cos, sin),
        )

    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb(self, x, cos, sin):
        """Applies Rotary Position Embedding to the input tensor."""
        # x: [batch, num_heads, seq_len, head_dim]
        # cos, sin: [1, 1, seq_len, head_dim]
        # Need to ensure broadcasting works.
        # If x is [batch, seq_len, num_heads, head_dim], we might need to unsqueeze differently.
        # Assuming standard [batch, num_heads, seq_len, head_dim] or [batch, seq_len, num_heads, head_dim]
        # where seq_len is -2.

        # If standard layout:
        # x: [..., seq_len, head_dim]
        # cos: [..., seq_len, head_dim]
        return (x * cos) + (self._rotate_half(x) * sin)


class ALiBi(nn.Module):
    """
    Attention with Linear Biases (ALiBi).

    Adds linear distance-based bias to attention scores instead of using
    positional embeddings. Each attention head gets a different slope.
    Enables length extrapolation beyond training sequence lengths.

    Reference: Press et al. (2022), "Train Short, Test Long"

    Args:
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length for precomputed bias matrix. Default: 2048

    Shape:
        - Input: [batch, num_heads, seq_len, seq_len] (attention scores)
        - Output: [batch, num_heads, seq_len, seq_len] (biased attention scores)

    Examples:
        >>> alibi = ALiBi(num_heads=8)
        >>> scores = torch.randn(2, 8, 128, 128)  # [batch, heads, seq, seq]
        >>> biased = alibi(scores)
        >>> biased.shape
        torch.Size([2, 8, 128, 128])

        >>> # Non-power-of-2 heads
        >>> alibi = ALiBi(num_heads=12, max_seq_len=512)
        >>> scores = torch.randn(4, 12, 64, 64)
        >>> biased = alibi(scores)

    Notes:
        - Operates on attention scores, not embeddings — placed in the embeddings
          module because it serves as a positional encoding mechanism
        - Slopes are geometric: 2^(-8/n * i) for power-of-2 n
        - Non-power-of-2 heads use interpolation between two closest powers
        - Bias is registered as a buffer (no gradients, moves with model)
        - Supports sequences up to max_seq_len without recomputation
    """

    def __init__(self, num_heads: int, max_seq_len: int = 2048) -> None:
        super().__init__()

        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")

        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Compute per-head slopes
        slopes = self._get_slopes(num_heads)
        # slopes shape: [num_heads]
        slopes = torch.tensor(slopes, dtype=torch.float32)

        # Precompute distance matrix [max_seq_len, max_seq_len]
        # distance[i, j] = |i - j|
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        distance = torch.abs(positions.unsqueeze(1) - positions.unsqueeze(0))

        # Compute bias: [1, num_heads, max_seq_len, max_seq_len]
        # bias[h, i, j] = -slope_h * |i - j|
        bias = -slopes.view(1, num_heads, 1, 1) * distance.view(
            1, 1, max_seq_len, max_seq_len
        )

        self.register_buffer("bias", bias)

    @staticmethod
    def _get_slopes(num_heads: int) -> list[float]:
        """
        Compute ALiBi slopes for each attention head.

        For power-of-2 num_heads: slopes = 2^(-8/n * (i+1)) for i in [0, n).
        For non-power-of-2: interpolate between two closest powers of 2.

        Args:
            num_heads: Number of attention heads

        Returns:
            List of slope values, one per head
        """

        def _get_slopes_power_of_2(n: int) -> list[float]:
            start = 2 ** (-(8.0 / n))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]

        # Check if num_heads is a power of 2
        if num_heads & (num_heads - 1) == 0:
            return _get_slopes_power_of_2(num_heads)

        # Non-power-of-2: use closest power of 2 and interpolate
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        slopes = _get_slopes_power_of_2(closest_power_of_2)

        # Get additional slopes by interpolating from 2x the closest power
        extra_base = 2 ** (-(8.0 / (2 * closest_power_of_2)))
        extra_slopes = [
            extra_base * (extra_base ** (2 * i))
            for i in range(num_heads - closest_power_of_2)
        ]
        slopes.extend(extra_slopes)

        return slopes

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Add ALiBi bias to attention scores.

        Args:
            input: Attention scores [batch, num_heads, seq_len, seq_len]

        Returns:
            Biased attention scores [batch, num_heads, seq_len, seq_len]

        Raises:
            ValueError: If input shape doesn't match expected dimensions
        """
        if input.dim() != 4:
            raise ValueError(
                f"Input must be 4D [batch, num_heads, seq_len, seq_len], "
                f"got {input.dim()}D with shape {list(input.shape)}"
            )

        if input.size(1) != self.num_heads:
            raise ValueError(
                f"Input has {input.size(1)} heads but ALiBi was created "
                f"with {self.num_heads}"
            )

        seq_len = input.size(-1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}. "
                f"Increase max_seq_len when creating ALiBi."
            )

        # Slice bias to match input sequence length
        # self.bias is [1, num_heads, max_seq_len, max_seq_len]
        bias = self.bias[:, :, :seq_len, :seq_len]

        return input + bias.to(dtype=input.dtype)

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"num_heads={self.num_heads}, max_seq_len={self.max_seq_len}"
