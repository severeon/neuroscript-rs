"""Tests for new Phase 0 primitives."""

import pytest
import torch

from neuroscript_runtime.primitives.operations import (
    Bias,
    Scale,
    MatMul,
    Einsum,
    Identity,
)
from neuroscript_runtime.primitives.activations import Mish, PReLU, ELU
from neuroscript_runtime.primitives.normalization import InstanceNorm
from neuroscript_runtime.primitives.convolutions import (
    Conv1d,
    Conv3d,
    DepthwiseConv,
    SeparableConv,
    TransposedConv,
)
from neuroscript_runtime.primitives.pooling import AdaptiveMaxPool, GlobalMaxPool
from neuroscript_runtime.primitives.structural import Split, Slice, Pad
from neuroscript_runtime.primitives.embeddings import ALiBi


class TestCoreOperations:
    def test_bias_forward(self):
        bias = Bias(dim=64)
        x = torch.randn(32, 64)
        out = bias(x)
        assert out.shape == (32, 64)
        assert bias.bias.shape == (64,)

    def test_scale_forward(self):
        scale = Scale(dim=64)
        x = torch.randn(32, 64)
        out = scale(x)
        assert out.shape == (32, 64)
        assert scale.scale.shape == (64,)

    def test_matmul_forward(self):
        matmul = MatMul()
        a = torch.randn(32, 10, 64)
        b = torch.randn(32, 64, 128)
        out = matmul((a, b))
        assert out.shape == (32, 10, 128)

    def test_einsum_forward(self):
        einsum = Einsum("bij,bjk->bik")
        a = torch.randn(32, 10, 64)
        b = torch.randn(32, 64, 128)
        out = einsum((a, b))
        assert out.shape == (32, 10, 128)

    def test_identity_forward(self):
        identity = Identity()
        x = torch.randn(32, 64)
        out = identity(x)
        assert torch.equal(out, x)


class TestActivations:
    def test_mish_forward(self):
        mish = Mish()
        x = torch.randn(32, 64)
        out = mish(x)
        assert out.shape == (32, 64)

    def test_prelu_forward(self):
        prelu = PReLU(num_parameters=1, init=0.25)
        x = torch.randn(32, 64)
        out = prelu(x)
        assert out.shape == (32, 64)

    def test_prelu_per_channel(self):
        prelu = PReLU(num_parameters=64, init=0.25)
        x = torch.randn(32, 64)
        out = prelu(x)
        assert out.shape == (32, 64)

    def test_elu_forward(self):
        elu = ELU(alpha=1.0)
        x = torch.randn(32, 64)
        out = elu(x)
        assert out.shape == (32, 64)


class TestNormalization:
    def test_instance_norm_forward(self):
        norm = InstanceNorm(num_features=64)
        x = torch.randn(32, 64, 28, 28)
        out = norm(x)
        assert out.shape == (32, 64, 28, 28)


class TestConvolutions:
    def test_conv1d_forward(self):
        conv = Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        x = torch.randn(32, 64, 100)
        out = conv(x)
        assert out.shape[0] == 32
        assert out.shape[1] == 128

    def test_conv3d_forward(self):
        conv = Conv3d(in_channels=3, out_channels=64, kernel_size=3)
        x = torch.randn(2, 3, 16, 32, 32)
        out = conv(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 64

    def test_depthwise_conv_forward(self):
        conv = DepthwiseConv(channels=64, kernel_size=3, padding=1)
        x = torch.randn(32, 64, 28, 28)
        out = conv(x)
        assert out.shape == (32, 64, 28, 28)

    def test_separable_conv_forward(self):
        conv = SeparableConv(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        x = torch.randn(32, 64, 28, 28)
        out = conv(x)
        assert out.shape[0] == 32
        assert out.shape[1] == 128

    def test_transposed_conv_forward(self):
        conv = TransposedConv(
            in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        x = torch.randn(32, 64, 14, 14)
        out = conv(x)
        assert out.shape[0] == 32
        assert out.shape[1] == 32
        assert out.shape[2] == 28


class TestPooling:
    def test_adaptive_max_pool_forward(self):
        pool = AdaptiveMaxPool(output_size=7)
        x = torch.randn(32, 64, 28, 28)
        out = pool(x)
        assert out.shape == (32, 64, 7, 7)

    def test_global_max_pool_forward(self):
        pool = GlobalMaxPool()
        x = torch.randn(32, 64, 28, 28)
        out = pool(x)
        assert out.shape == (32, 64, 1, 1)


class TestStructural:
    def test_split_forward(self):
        split = Split(num_splits=2, dim=-1)
        x = torch.randn(32, 64)
        chunks = split(x)
        assert len(chunks) == 2
        assert chunks[0].shape == (32, 32)
        assert chunks[1].shape == (32, 32)

    def test_slice_forward(self):
        slc = Slice(dim=1, start=10, end=20)
        x = torch.randn(32, 64)
        out = slc(x)
        assert out.shape == (32, 10)

    def test_slice_to_end(self):
        slc = Slice(dim=1, start=10, end=-1)
        x = torch.randn(32, 64)
        out = slc(x)
        assert out.shape == (32, 54)

    def test_pad_forward(self):
        pad = Pad(padding=(1, 1, 1, 1), value=0)
        x = torch.randn(32, 64, 28, 28)
        out = pad(x)
        assert out.shape == (32, 64, 30, 30)


class TestALiBiCausal:
    def test_causal_upper_triangular_is_neg_inf(self):
        alibi = ALiBi(num_heads=4, max_seq_len=8, causal=True)
        scores = torch.zeros(1, 4, 8, 8)
        biased = alibi(scores)
        # Upper triangular (j > i) should be -inf
        for i in range(8):
            for j in range(i + 1, 8):
                assert biased[0, 0, i, j] == float("-inf")

    def test_causal_lower_triangular_is_finite(self):
        alibi = ALiBi(num_heads=4, max_seq_len=8, causal=True)
        scores = torch.zeros(1, 4, 8, 8)
        biased = alibi(scores)
        # Diagonal and lower triangular (j <= i) should be finite
        for i in range(8):
            for j in range(i + 1):
                assert torch.isfinite(biased[0, 0, i, j])

    def test_causal_diagonal_is_zero(self):
        alibi = ALiBi(num_heads=4, max_seq_len=8, causal=True)
        scores = torch.zeros(1, 4, 8, 8)
        biased = alibi(scores)
        # Diagonal (i == j) means distance=0, so bias=0
        for i in range(8):
            for h in range(4):
                assert biased[0, h, i, i] == 0.0

    def test_extra_repr_includes_causal(self):
        alibi = ALiBi(num_heads=4, causal=True)
        assert "causal=True" in alibi.extra_repr()

    def test_causal_output_shape(self):
        alibi = ALiBi(num_heads=8, causal=True)
        scores = torch.randn(2, 8, 128, 128)
        biased = alibi(scores)
        assert biased.shape == torch.Size([2, 8, 128, 128])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
