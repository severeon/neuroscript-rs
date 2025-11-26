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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
