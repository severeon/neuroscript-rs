"""Tests for the experimental NeuralBlock primitive."""

from __future__ import annotations

import pytest
import torch

from neuroscript_runtime.primitives.neural_block import (
    NeuralBlock,
    cache_neural_block_weights,
    clear_neural_block_caches,
)


class TestConstraints:
    def test_rejects_d_above_max(self):
        # n=4 → max d = (4 // 2) - 1 = 1
        with pytest.raises(ValueError, match=r"d=2 exceeds maximum"):
            NeuralBlock(n=4, dim=4, d=2, r=0)

    def test_rejects_dim_mismatch(self):
        # n=4, r=1 → expected dim = 16
        with pytest.raises(ValueError, match=r"dim must equal n\*\*\(r\+1\)"):
            NeuralBlock(n=4, dim=8, d=1, r=1)

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            NeuralBlock(n=0, dim=0, d=0, r=0)
        with pytest.raises(ValueError):
            NeuralBlock(n=4, dim=4, d=-1, r=0)
        with pytest.raises(ValueError):
            NeuralBlock(n=4, dim=4, d=0, r=-1)

    def test_d_zero_always_legal(self):
        # n=1 has max d = 0; n=2 has max d = 0; both should accept d=0.
        NeuralBlock(n=1, dim=1, d=0, r=0)
        NeuralBlock(n=2, dim=2, d=0, r=0)


class TestForwardSemantics:
    def test_d0_reproduces_row_product(self):
        torch.manual_seed(42)
        N = 4
        blk = NeuralBlock(n=N, dim=N, d=0, r=0)
        x = torch.randn(7, N)
        y = blk(x)
        # With d=0, output_i = x_i * prod_c W[c, i] + bias.
        expected = x * blk.weights.prod(dim=0) + blk.bias
        assert torch.allclose(y, expected, atol=1e-6)

    def test_d1_matches_hand_computation(self):
        """Reproduce the user-spec example structure for n=4, d=1, r=0."""
        torch.manual_seed(0)
        N, D = 4, 1
        blk = NeuralBlock(n=N, dim=N, d=D, r=0)
        with torch.no_grad():
            blk.bias.zero_()
        x = torch.randn(N)

        W = blk.weights
        v = x * W[0]
        for c in range(1, N):
            s = torch.zeros_like(v)
            for i in range(N):
                for k in (-1, 0, 1):
                    s[i] = s[i] + v[(i + k) % N]
            v = s * W[c]

        y = blk(x)
        assert torch.allclose(y, v, atol=1e-6)

    @pytest.mark.parametrize("n,d,r", [(4, 0, 0), (4, 1, 0), (4, 1, 1), (4, 1, 2), (5, 1, 0), (6, 2, 0)])
    def test_io_shape(self, n, d, r):
        dim = n ** (r + 1)
        blk = NeuralBlock(n=n, dim=dim, d=d, r=r)
        x = torch.randn(2, 3, dim)
        y = blk(x)
        assert y.shape == (2, 3, dim)

    def test_param_count_formula(self):
        # Total params at the root: (n*n)^(r+1) leaves + 1 bias.
        n, d, r = 4, 1, 2
        blk = NeuralBlock(n=n, dim=n ** (r + 1), d=d, r=r)
        total = sum(p.numel() for p in blk.parameters())
        assert total == (n * n) ** (r + 1) + 1  # 4097

    def test_param_count_at_each_r(self):
        # Spot-check the formula at several depths.
        for r in (0, 1, 2, 3):
            blk = NeuralBlock(n=4, dim=4 ** (r + 1), d=1, r=r)
            total = sum(p.numel() for p in blk.parameters())
            assert total == 16 ** (r + 1) + 1


# ─── Equivalence: dense materialization vs naive column-by-column reference ──


def _naive_apply(blk: NeuralBlock, x: torch.Tensor) -> torch.Tensor:
    """Reference implementation that walks the column-by-column algorithm
    directly on the leaf weights, recursing one level at a time.

    Computes NB(x) without materializing W. Used as a ground truth in the
    equivalence test below. Mathematically identical to the spec; slow.
    """
    leading = x.shape[:-1]
    N = blk.n
    D = blk.d
    leaves = blk._all_leaves().detach()  # [G_total, N, N]

    def banded_sum(v: torch.Tensor) -> torch.Tensor:
        # v: [..., N, M] — circular banded sum along dim=-2.
        if D == 0:
            return v
        out = v
        for k in range(1, D + 1):
            out = out + torch.roll(v, shifts=-k, dims=-2)
            out = out + torch.roll(v, shifts=+k, dims=-2)
        return out

    def apply_block(leaves_g: torch.Tensor, depth: int, x: torch.Tensor) -> torch.Tensor:
        # leaves_g: [G_g, N, N] — leaves belonging to one block at this depth.
        # depth: remaining recursion levels (0 means apply scalar weights).
        # x: [..., N**(depth+1)]
        unit_size = N ** depth
        v = x.reshape(*x.shape[:-1], N, unit_size)        # [..., N, M]

        if depth == 0:
            # leaves_g is [1, N, N]; weights[col, row].
            W = leaves_g[0]
            v = v * W[0].view(*([1] * (v.dim() - 2)), N, 1)
            for c in range(1, N):
                s = banded_sum(v)
                v = s * W[c].view(*([1] * (v.dim() - 2)), N, 1)
            return v.reshape(*x.shape[:-1], blk.n)

        # depth >= 1: leaves_g is [N² · N^(2(depth-1)), N, N].
        # Inner cells are arranged depth-first cell-major: outer cell index
        # c·N + i indexes a contiguous block of N^(2(depth-1)) leaves.
        leaves_per_inner = N ** (2 * (depth - 1))

        def cell_apply(c: int, i: int, x_row: torch.Tensor) -> torch.Tensor:
            start = (c * N + i) * leaves_per_inner
            sub = leaves_g[start:start + leaves_per_inner]
            return apply_block(sub, depth - 1, x_row)

        # Column 0: per-cell apply, no banded sum.
        rows = [cell_apply(0, i, v[..., i, :]) for i in range(N)]
        v = torch.stack(rows, dim=-2)
        for c in range(1, N):
            s = banded_sum(v)
            rows = [cell_apply(c, i, s[..., i, :]) for i in range(N)]
            v = torch.stack(rows, dim=-2)
        return v.reshape(*x.shape[:-1], N ** (depth + 1))

    out = apply_block(leaves, blk.r, x)
    if blk.bias is not None:
        out = out + blk.bias
    return out


class TestDenseEquivalence:
    """``forward`` (which uses materialize_weight + linear) must produce the
    same outputs as the naive column-by-column reference for the same params
    and input."""

    @pytest.mark.parametrize("n,d,r", [
        (4, 0, 0), (4, 1, 0), (5, 1, 0), (6, 2, 0),
        (4, 1, 1), (4, 0, 2), (4, 1, 2), (4, 1, 3),
    ])
    def test_dense_matches_naive(self, n, d, r):
        torch.manual_seed(7)
        dim = n ** (r + 1)
        blk = NeuralBlock(n=n, dim=dim, d=d, r=r)
        x = torch.randn(3, dim)
        y_dense = blk(x)
        y_naive = _naive_apply(blk, x)
        assert torch.allclose(y_dense, y_naive, atol=1e-5), \
            f"dense vs naive diverge at (n={n}, d={d}, r={r}): " \
            f"max-abs-diff {(y_dense - y_naive).abs().max().item()}"


class TestBackward:
    def test_all_params_receive_grad(self):
        n, d, r = 4, 1, 1
        blk = NeuralBlock(n=n, dim=n ** (r + 1), d=d, r=r)
        x = torch.randn(2, n ** (r + 1), requires_grad=False)
        y = blk(x).sum()
        y.backward()
        ungrad = [name for name, p in blk.named_parameters() if p.grad is None]
        assert ungrad == [], f"params without grad: {ungrad}"

    def test_finite_outputs_with_default_init(self):
        # Default init is scaled by 1/sqrt(2D+1); deeper blocks should not blow up.
        n, d, r = 4, 1, 2
        blk = NeuralBlock(n=n, dim=n ** (r + 1), d=d, r=r)
        x = torch.randn(8, n ** (r + 1))
        y = blk(x)
        assert torch.isfinite(y).all()
        assert y.std().item() < 50.0


class TestCache:
    def test_cache_matches_uncached(self):
        torch.manual_seed(3)
        blk = NeuralBlock(n=4, dim=64, d=1, r=2)
        x = torch.randn(5, 64)
        with torch.no_grad():
            y_uncached = blk(x).clone()
            blk.cache_dense_weight()
            y_cached = blk(x).clone()
        assert torch.allclose(y_uncached, y_cached, atol=1e-7)

    def test_cache_bypassed_in_grad_mode(self):
        """When grad is enabled (PPO update), cache must NOT be used —
        otherwise backprop would silently miss the cell parameters."""
        torch.manual_seed(11)
        blk = NeuralBlock(n=4, dim=16, d=1, r=1)
        blk.cache_dense_weight()
        x = torch.randn(2, 16, requires_grad=False)
        y = blk(x).sum()
        y.backward()
        assert blk.leaf_weights.grad is not None
        assert blk.leaf_weights.grad.abs().sum().item() > 0

    def test_module_helpers_walk_submodules(self):
        net = torch.nn.Sequential(
            NeuralBlock(n=4, dim=16, d=1, r=1),
            torch.nn.ReLU(),
            NeuralBlock(n=4, dim=16, d=1, r=1),
        )
        cache_neural_block_weights(net)
        for m in net.modules():
            if isinstance(m, NeuralBlock):
                assert m._W_cache is not None
        clear_neural_block_caches(net)
        for m in net.modules():
            if isinstance(m, NeuralBlock):
                assert m._W_cache is None
