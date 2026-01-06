"""
Comprehensive test suite for NeuroScript primitives.

Tests all Level 0 primitives to ensure they work correctly.
"""

import pytest
import torch
import torch.nn as nn

from neuroscript_runtime.primitives import (
    # Core Operations
    Linear,
    Bias,
    Scale,
    MatMul,
    Einsum,
    # Activations
    GELU,
    ReLU,
    Tanh,
    Sigmoid,
    SiLU,
    Softmax,
    # Normalization
    LayerNorm,
    RMSNorm,
    GroupNorm,
    # Regularization
    Dropout,
    DropPath,
    DropConnect,
    # Embeddings
    Embedding,
    PositionalEncoding,
    LearnedPositionalEmbedding,
    # Structural Operations
    Fork,
    Fork3,
    Add,
    Concat,
    # Attention
    ScaledDotProductAttention,
)


class TestLinear:
    """Test Linear primitive."""

    def test_basic_forward(self):
        """Test basic linear transformation."""
        layer = Linear(512, 256)
        x = torch.randn(32, 512)
        out = layer(x)
        assert out.shape == (32, 256)

    def test_multi_dim_input(self):
        """Test with multi-dimensional input."""
        layer = Linear(512, 256)
        x = torch.randn(8, 16, 512)
        out = layer(x)
        assert out.shape == (8, 16, 256)

    def test_no_bias(self):
        """Test linear layer without bias."""
        layer = Linear(128, 64, bias=False)
        assert layer.bias is None
        x = torch.randn(16, 128)
        out = layer(x)
        assert out.shape == (16, 64)

    def test_invalid_input_dim(self):
        """Test error on dimension mismatch."""
        layer = Linear(512, 256)
        x = torch.randn(32, 256)  # Wrong dimension
        with pytest.raises(ValueError, match="Input feature dimension mismatch"):
            layer(x)


class TestActivations:
    """Test activation function primitives."""

    @pytest.fixture
    def sample_input(self):
        """Sample input tensor for testing."""
        return torch.randn(32, 512)

    def test_gelu(self, sample_input):
        """Test GELU activation."""
        gelu = GELU()
        out = gelu(sample_input)
        assert out.shape == sample_input.shape

    def test_gelu_approximate(self, sample_input):
        """Test GELU with tanh approximation."""
        gelu = GELU(approximate='tanh')
        out = gelu(sample_input)
        assert out.shape == sample_input.shape

    def test_relu(self, sample_input):
        """Test ReLU activation."""
        relu = ReLU()
        out = relu(sample_input)
        assert out.shape == sample_input.shape
        assert (out >= 0).all()  # All values should be non-negative

    def test_tanh(self, sample_input):
        """Test Tanh activation."""
        tanh = Tanh()
        out = tanh(sample_input)
        assert out.shape == sample_input.shape
        assert (out >= -1).all() and (out <= 1).all()  # Values in (-1, 1)

    def test_sigmoid(self, sample_input):
        """Test Sigmoid activation."""
        sigmoid = Sigmoid()
        out = sigmoid(sample_input)
        assert out.shape == sample_input.shape
        assert (out >= 0).all() and (out <= 1).all()  # Values in (0, 1)

    def test_silu(self, sample_input):
        """Test SiLU (Swish) activation."""
        silu = SiLU()
        out = silu(sample_input)
        assert out.shape == sample_input.shape

    def test_softmax(self):
        """Test Softmax activation."""
        softmax = Softmax(dim=-1)
        x = torch.randn(32, 10, 512)
        out = softmax(x)
        assert out.shape == x.shape
        # Check that probabilities sum to 1 along last dimension
        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


class TestNormalization:
    """Test normalization primitives."""

    def test_layer_norm(self):
        """Test LayerNorm."""
        layer_norm = LayerNorm(512)
        x = torch.randn(32, 10, 512)
        out = layer_norm(x)
        assert out.shape == x.shape

    def test_layer_norm_multi_dim(self):
        """Test LayerNorm with multiple normalized dimensions."""
        layer_norm = LayerNorm([10, 512])
        x = torch.randn(32, 10, 512)
        out = layer_norm(x)
        assert out.shape == x.shape

    def test_rms_norm(self):
        """Test RMSNorm."""
        rms_norm = RMSNorm(512)
        x = torch.randn(32, 10, 512)
        out = rms_norm(x)
        assert out.shape == x.shape

    def test_rms_norm_invalid_dim(self):
        """Test RMSNorm with dimension mismatch."""
        rms_norm = RMSNorm(512)
        x = torch.randn(32, 10, 256)  # Wrong dimension
        with pytest.raises(ValueError, match="Input last dimension mismatch"):
            rms_norm(x)

    def test_group_norm(self):
        """Test GroupNorm."""
        group_norm = GroupNorm(num_groups=8, num_channels=32)
        x = torch.randn(16, 32, 64, 64)  # [batch, channels, H, W]
        out = group_norm(x)
        assert out.shape == x.shape

    def test_group_norm_invalid_channels(self):
        """Test GroupNorm with invalid channel configuration."""
        with pytest.raises(ValueError, match="must be divisible"):
            GroupNorm(num_groups=8, num_channels=30)  # 30 not divisible by 8


class TestRegularization:
    """Test regularization primitives."""

    def test_dropout_train(self):
        """Test Dropout in training mode."""
        dropout = Dropout(p=0.5)
        dropout.train()
        x = torch.ones(1000, 512)
        out = dropout(x)
        assert out.shape == x.shape
        # Check that approximately half the elements are zeroed
        zero_ratio = (out == 0).float().mean()
        assert 0.4 < zero_ratio < 0.6  # Should be close to 0.5

    def test_dropout_eval(self):
        """Test Dropout in evaluation mode."""
        dropout = Dropout(p=0.5)
        dropout.eval()
        x = torch.ones(100, 512)
        out = dropout(x)
        assert torch.allclose(out, x)  # No dropout in eval mode

    def test_drop_path_train(self):
        """Test DropPath in training mode."""
        drop_path = DropPath(drop_prob=0.5)
        drop_path.train()
        x = torch.ones(100, 512)
        out = drop_path(x)
        assert out.shape == x.shape
        # Check that some samples are completely zeroed
        samples_zeroed = (out.sum(dim=1) == 0).float().mean()
        assert samples_zeroed > 0  # Some samples should be dropped

    def test_drop_path_eval(self):
        """Test DropPath in evaluation mode."""
        drop_path = DropPath(drop_prob=0.5)
        drop_path.eval()
        x = torch.ones(100, 512)
        out = drop_path(x)
        assert torch.allclose(out, x)  # No dropout in eval mode

    def test_drop_connect_train(self):
        """Test DropConnect in training mode."""
        drop_connect = DropConnect(p=0.5)
        drop_connect.train()
        x = torch.ones(1000, 512)
        out = drop_connect(x)
        assert out.shape == x.shape
        # Check that approximately half the connections are dropped
        zero_ratio = (out == 0).float().mean()
        assert 0.4 < zero_ratio < 0.6

    def test_drop_connect_eval(self):
        """Test DropConnect in evaluation mode."""
        drop_connect = DropConnect(p=0.5)
        drop_connect.eval()
        x = torch.ones(100, 512)
        out = drop_connect(x)
        assert torch.allclose(out, x)


class TestEmbeddings:
    """Test embedding primitives."""

    def test_embedding(self):
        """Test token Embedding."""
        embedding = Embedding(vocab_size=10000, embedding_dim=512)
        tokens = torch.randint(0, 10000, (32, 128))
        out = embedding(tokens)
        assert out.shape == (32, 128, 512)

    def test_embedding_with_padding(self):
        """Test Embedding with padding index."""
        embedding = Embedding(vocab_size=1000, embedding_dim=256, padding_idx=0)
        tokens = torch.tensor([[1, 2, 0], [5, 0, 0]])
        out = embedding(tokens)
        assert out.shape == (2, 3, 256)
        # Padding positions should be zeros
        assert torch.allclose(out[0, 2], torch.zeros(256))
        assert torch.allclose(out[1, 1], torch.zeros(256))
        assert torch.allclose(out[1, 2], torch.zeros(256))

    def test_embedding_invalid_indices(self):
        """Test Embedding with out-of-range indices."""
        embedding = Embedding(vocab_size=100, embedding_dim=128)
        tokens = torch.tensor([[1, 2, 150]])  # 150 is out of range
        with pytest.raises(RuntimeError, match="Input indices must be in"):
            embedding(tokens)

    def test_positional_encoding(self):
        """Test PositionalEncoding."""
        pos_enc = PositionalEncoding(d_model=512, max_len=1000, dropout=0.0)
        pos_enc.eval()  # No dropout for testing
        x = torch.zeros(32, 100, 512)
        out = pos_enc(x)
        assert out.shape == x.shape
        # Check that positions were added (output should differ from input)
        assert not torch.allclose(out, x)

    def test_positional_encoding_max_len(self):
        """Test PositionalEncoding exceeding max_len."""
        pos_enc = PositionalEncoding(d_model=512, max_len=100)
        x = torch.randn(32, 150, 512)  # seq_len > max_len
        with pytest.raises(ValueError, match="exceeds max_len"):
            pos_enc(x)

    def test_learned_positional_embedding(self):
        """Test LearnedPositionalEmbedding."""
        pos_emb = LearnedPositionalEmbedding(max_len=512, d_model=768)
        x = torch.zeros(32, 128, 768)
        out = pos_emb(x)
        assert out.shape == x.shape
        # Check that positions were added
        assert not torch.allclose(out, x)

    def test_learned_positional_embedding_max_len(self):
        """Test LearnedPositionalEmbedding exceeding max_len."""
        pos_emb = LearnedPositionalEmbedding(max_len=100, d_model=512)
        x = torch.randn(32, 150, 512)
        with pytest.raises(ValueError, match="exceeds max_len"):
            pos_emb(x)


class TestIntegration:
    """Integration tests combining multiple primitives."""

    def test_transformer_block_components(self):
        """Test components that might appear in a transformer block."""
        batch_size, seq_len, d_model = 16, 64, 512

        # Create components
        layer_norm = LayerNorm(d_model)
        linear = Linear(d_model, d_model * 4)
        gelu = GELU()
        dropout = Dropout(p=0.1)

        # Forward pass
        x = torch.randn(batch_size, seq_len, d_model)
        x = layer_norm(x)
        x = linear(x)
        x = gelu(x)
        x = dropout(x)

        assert x.shape == (batch_size, seq_len, d_model * 4)

    def test_embedding_pipeline(self):
        """Test embedding with positional encoding."""
        vocab_size = 10000
        d_model = 512
        max_len = 1000
        batch_size = 32
        seq_len = 128

        # Create components
        embedding = Embedding(vocab_size, d_model)
        pos_enc = PositionalEncoding(d_model, max_len, dropout=0.0)
        pos_enc.eval()

        # Forward pass
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        x = embedding(tokens)
        x = pos_enc(x)

        assert x.shape == (batch_size, seq_len, d_model)


class TestStructuralOperations:
    """Test structural operation primitives (Fork, Add, Concat)."""

    def test_fork_basic(self):
        """Test basic Fork operation."""
        fork = Fork()
        x = torch.randn(32, 512)
        a, b = fork(x)

        assert a.shape == (32, 512)
        assert b.shape == (32, 512)
        # Should be same reference, not a copy
        assert a is b
        assert torch.equal(a, x)

    def test_fork_multi_dim(self):
        """Test Fork with multi-dimensional input."""
        fork = Fork()
        x = torch.randn(8, 16, 10, 64)
        a, b = fork(x)

        assert a.shape == (8, 16, 10, 64)
        assert b.shape == (8, 16, 10, 64)
        assert a is b

    def test_fork3_basic(self):
        """Test basic Fork3 operation."""
        fork3 = Fork3()
        x = torch.randn(32, 512)
        a, b, c = fork3(x)

        assert a.shape == (32, 512)
        assert b.shape == (32, 512)
        assert c.shape == (32, 512)
        # All should be same reference
        assert a is b is c
        assert torch.equal(a, x)

    def test_fork3_multi_dim(self):
        """Test Fork3 with multi-dimensional input."""
        fork3 = Fork3()
        x = torch.randn(4, 8, 256)
        a, b, c = fork3(x)

        assert a.shape == (4, 8, 256)
        assert b.shape == (4, 8, 256)
        assert c.shape == (4, 8, 256)
        assert a is b is c

    def test_add_basic(self):
        """Test basic Add operation."""
        add = Add()
        x = torch.randn(32, 512)
        y = torch.randn(32, 512)
        result = add((x, y))

        assert result.shape == (32, 512)
        assert torch.allclose(result, x + y)

    def test_add_multi_dim(self):
        """Test Add with multi-dimensional inputs."""
        add = Add()
        x = torch.randn(8, 16, 10, 64)
        y = torch.randn(8, 16, 10, 64)
        result = add((x, y))

        assert result.shape == (8, 16, 10, 64)
        assert torch.allclose(result, x + y)

    def test_add_broadcasting(self):
        """Test Add with broadcasting."""
        add = Add()
        x = torch.randn(32, 512)
        y = torch.randn(512)  # Will broadcast
        result = add((x, y))

        assert result.shape == (32, 512)
        assert torch.allclose(result, x + y)

    def test_add_residual_connection(self):
        """Test Add in typical residual connection pattern."""
        add = Add()
        fork = Fork()

        # Simulate residual connection
        x = torch.randn(32, 512)
        main, skip = fork(x)

        # Process main path (simulate with simple scaling)
        processed = main * 2.0

        # Add back the skip connection
        result = add((processed, skip))

        assert result.shape == (32, 512)
        expected = x * 2.0 + x
        assert torch.allclose(result, expected)

    def test_add_incompatible_shapes(self):
        """Test Add error on incompatible shapes."""
        add = Add()
        x = torch.randn(32, 512)
        y = torch.randn(32, 256)  # Incompatible shape

        with pytest.raises(ValueError, match="Cannot add tensors"):
            add((x, y))

    def test_concat_basic(self):
        """Test basic Concat operation."""
        concat = Concat(dim=-1)
        x = torch.randn(32, 10, 512)
        y = torch.randn(32, 10, 512)
        result = concat((x, y))

        assert result.shape == (32, 10, 1024)  # Concatenated along last dim
        assert torch.equal(result[..., :512], x)
        assert torch.equal(result[..., 512:], y)

    def test_concat_three_tensors(self):
        """Test Concat with three tensors."""
        concat = Concat(dim=-1)
        x = torch.randn(16, 256)
        y = torch.randn(16, 128)
        z = torch.randn(16, 64)
        result = concat((x, y, z))

        assert result.shape == (16, 448)  # 256 + 128 + 64
        assert torch.equal(result[..., :256], x)
        assert torch.equal(result[..., 256:384], y)
        assert torch.equal(result[..., 384:], z)

    def test_concat_dim_0(self):
        """Test Concat along dimension 0."""
        concat = Concat(dim=0)
        x = torch.randn(32, 512)
        y = torch.randn(16, 512)
        result = concat((x, y))

        assert result.shape == (48, 512)  # Concatenated along batch dim
        assert torch.equal(result[:32], x)
        assert torch.equal(result[32:], y)

    def test_concat_multi_dim(self):
        """Test Concat with multi-dimensional inputs."""
        concat = Concat(dim=2)
        x = torch.randn(8, 16, 10, 64)
        y = torch.randn(8, 16, 20, 64)
        result = concat((x, y))

        assert result.shape == (8, 16, 30, 64)

    def test_concat_too_few_inputs(self):
        """Test Concat error with fewer than 2 tensors."""
        concat = Concat(dim=-1)
        x = torch.randn(32, 512)

        with pytest.raises(ValueError, match="at least 2 tensors"):
            concat((x,))

    def test_concat_incompatible_shapes(self):
        """Test Concat error on incompatible shapes."""
        concat = Concat(dim=-1)
        x = torch.randn(32, 512)
        y = torch.randn(16, 512)  # Different batch size

        with pytest.raises(ValueError, match="Cannot concatenate"):
            concat((x, y))


class TestAttention:
    """Test attention mechanism primitives."""

    def test_scaled_dot_product_basic(self):
        """Test basic scaled dot-product attention."""
        attn = ScaledDotProductAttention()
        q = torch.randn(32, 10, 64)  # batch=32, seq_q=10, d_k=64
        k = torch.randn(32, 20, 64)  # batch=32, seq_k=20, d_k=64
        v = torch.randn(32, 20, 128) # batch=32, seq_k=20, d_v=128

        output = attn((q, k, v))

        assert output.shape == (32, 10, 128)  # [batch, seq_q, d_v]

    def test_scaled_dot_product_self_attention(self):
        """Test self-attention (Q=K=V)."""
        attn = ScaledDotProductAttention()
        x = torch.randn(16, 8, 64)  # batch=16, seq=8, dim=64

        output = attn((x, x, x))

        assert output.shape == (16, 8, 64)

    def test_scaled_dot_product_multi_head(self):
        """Test with multi-head attention dimensions."""
        attn = ScaledDotProductAttention()
        # [batch, num_heads, seq, d_k]
        q = torch.randn(32, 8, 10, 64)  # 8 heads
        k = torch.randn(32, 8, 20, 64)
        v = torch.randn(32, 8, 20, 128)

        output = attn((q, k, v))

        assert output.shape == (32, 8, 10, 128)

    def test_scaled_dot_product_with_dropout(self):
        """Test attention with dropout."""
        attn = ScaledDotProductAttention(dropout_p=0.1)
        q = torch.randn(32, 10, 64)
        k = torch.randn(32, 20, 64)
        v = torch.randn(32, 20, 128)

        # In training mode, dropout should be applied
        attn.train()
        output = attn((q, k, v))
        assert output.shape == (32, 10, 128)

        # In eval mode, dropout should not be applied
        attn.eval()
        output_eval = attn((q, k, v))
        assert output_eval.shape == (32, 10, 128)

    def test_scaled_dot_product_custom_scale(self):
        """Test attention with custom scale factor."""
        attn = ScaledDotProductAttention(scale=0.5)
        q = torch.randn(16, 10, 64)
        k = torch.randn(16, 10, 64)
        v = torch.randn(16, 10, 64)

        output = attn((q, k, v))

        assert output.shape == (16, 10, 64)

    def test_scaled_dot_product_dimension_mismatch(self):
        """Test attention error on query/key dimension mismatch."""
        attn = ScaledDotProductAttention()
        q = torch.randn(32, 10, 64)
        k = torch.randn(32, 20, 128)  # Different d_k
        v = torch.randn(32, 20, 128)

        with pytest.raises(ValueError, match="same last dimension"):
            attn((q, k, v))

    def test_scaled_dot_product_sequence_mismatch(self):
        """Test attention error on key/value sequence mismatch."""
        attn = ScaledDotProductAttention()
        q = torch.randn(32, 10, 64)
        k = torch.randn(32, 20, 64)
        v = torch.randn(32, 15, 64)  # Different seq_k

        with pytest.raises(ValueError, match="same sequence length"):
            attn((q, k, v))

    def test_scaled_dot_product_invalid_dropout(self):
        """Test attention error on invalid dropout probability."""
        with pytest.raises(ValueError, match="dropout_p must be in"):
            ScaledDotProductAttention(dropout_p=1.5)

    def test_scaled_dot_product_attention_properties(self):
        """Test attention output properties."""
        attn = ScaledDotProductAttention()
        q = torch.randn(16, 10, 64)
        k = torch.randn(16, 10, 64)
        v = torch.randn(16, 10, 64)

        attn.eval()  # Disable dropout for deterministic output
        output = attn((q, k, v))

        # Output should be a weighted sum of values
        # So output magnitude should be bounded by value magnitudes
        assert output.shape == (16, 10, 64)
        assert torch.isfinite(output).all()


class TestArithmetic:
    """Test arithmetic primitives (Bias, Scale)."""

    def test_bias_basic(self):
        """Test basic Bias operation."""
        bias = Bias(512)
        x = torch.randn(32, 512)
        out = bias(x)
        assert out.shape == (32, 512)
        assert torch.allclose(out, x)  # Initial bias is zeros

    def test_bias_learnable(self):
        """Test that Bias is learnable."""
        bias = Bias(512)
        x = torch.randn(32, 512)
        # Update bias parameter manually
        with torch.no_grad():
            bias.bias.fill_(1.0)
        out = bias(x)
        assert torch.allclose(out, x + 1.0)

    def test_bias_invalid_dim(self):
        """Test Bias error on dimension mismatch."""
        bias = Bias(512)
        x = torch.randn(32, 256)
        with pytest.raises(ValueError, match="Input dimension mismatch"):
            bias(x)

    def test_scale_basic(self):
        """Test basic Scale operation."""
        scale = Scale(512)
        x = torch.randn(32, 512)
        out = scale(x)
        assert out.shape == (32, 512)
        assert torch.allclose(out, x)  # Initial scale is ones

    def test_scale_learnable(self):
        """Test that Scale is learnable."""
        scale = Scale(512)
        x = torch.randn(32, 512)
        # Update scale parameter manually
        with torch.no_grad():
            scale.scale.fill_(2.0)
        out = scale(x)
        assert torch.allclose(out, x * 2.0)

    def test_scale_invalid_dim(self):
        """Test Scale error on dimension mismatch."""
        scale = Scale(512)
        x = torch.randn(32, 256)
        with pytest.raises(ValueError, match="Input dimension mismatch"):
            scale(x)


class TestMatrix:
    """Test matrix primitives (MatMul, Einsum)."""

    def test_matmul_basic(self):
        """Test basic MatMul operation."""
        matmul = MatMul()
        x = torch.randn(32, 10, 64)
        y = torch.randn(32, 64, 128)
        out = matmul((x, y))
        assert out.shape == (32, 10, 128)
        assert torch.allclose(out, x @ y)

    def test_matmul_incompatible_shapes(self):
        """Test MatMul error on incompatible shapes."""
        matmul = MatMul()
        x = torch.randn(32, 10, 64)
        y = torch.randn(32, 32, 128)  # 64 != 32
        with pytest.raises(ValueError, match="Cannot multiply tensors"):
            matmul((x, y))

    def test_einsum_basic(self):
        """Test basic Einsum operation."""
        # Matrix multiplication using einsum
        einsum = Einsum("ij,jk->ik")
        x = torch.randn(10, 64)
        y = torch.randn(64, 128)
        out = einsum((x, y))
        assert out.shape == (10, 128)
        assert torch.allclose(out, x @ y)

    def test_einsum_batched(self):
        """Test batched Einsum."""
        einsum = Einsum("bij,bjk->bik")
        x = torch.randn(32, 10, 64)
        y = torch.randn(32, 64, 128)
        out = einsum((x, y))
        assert out.shape == (32, 10, 128)
        assert torch.allclose(out, x @ y)

    def test_einsum_transpose(self):
        """Test Einsum for transpose."""
        einsum = Einsum("ij->ji")
        x = torch.randn(10, 64)
        out = einsum(x)
        assert out.shape == (64, 10)
        assert torch.allclose(out, x.t())

    def test_einsum_invalid_equation(self):
        """Test Einsum error on invalid equation."""
        einsum = Einsum("ij,jk->il")  # Output 'l' is undefined
        x = torch.randn(10, 64)
        y = torch.randn(64, 128)
        with pytest.raises(ValueError, match="Einsum failed"):
            einsum((x, y))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
