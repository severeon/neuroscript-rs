import pytest
import torch
import torch.nn as nn


class ReferenceConformer(nn.Module):
    def __init__(self, dim, num_heads, ff_expansion, conv_kernel, dropout_rate):
        super().__init__()
        # Self-attention sub-block
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.attn_drop = nn.Dropout(dropout_rate)

        # Convolution sub-block
        self.conv_norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, conv_kernel, padding=conv_kernel // 2)
        self.conv_bn = nn.BatchNorm1d(dim)
        self.conv_act = nn.SiLU()
        self.conv_drop = nn.Dropout(dropout_rate)

        # FFN sub-block
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_expansion), nn.GELU(), nn.Linear(ff_expansion, dim)
        )
        self.ffn_drop = nn.Dropout(dropout_rate)

        # Final norm
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # Attention residual
        r = self.attn_norm(x)
        r, _ = self.attn(r, r, r)
        x = x + self.attn_drop(r)

        # Conv residual
        r = self.conv_norm(x)
        r = r.transpose(1, 2)
        r = self.conv_act(self.conv_bn(self.conv(r)))
        r = r.transpose(1, 2)
        x = x + self.conv_drop(r)

        # FFN residual
        r = self.ffn_norm(x)
        r = self.ffn(r)
        x = x + self.ffn_drop(r)

        return self.final_norm(x)


@pytest.mark.parametrize(
    "batch,seq",
    [(2, 16), (1, 64), (4, 8)],
    ids=["b2s16", "b1s64", "b4s8"],
)
def test_conformer_shapes(batch, seq):
    dim, heads, ff_exp, kernel, drop = 64, 4, 128, 7, 0.0
    model = ReferenceConformer(dim, heads, ff_exp, kernel, drop)

    x = torch.randn(batch, seq, dim)
    y = model(x)

    assert y.shape == x.shape, (
        f"Output shape {y.shape} does not match input shape {x.shape}"
    )


def test_conformer_gradient():
    dim, heads, ff_exp, kernel, drop = 32, 4, 64, 5, 0.0
    model = ReferenceConformer(dim, heads, ff_exp, kernel, drop)

    x = torch.randn(2, 16, dim, requires_grad=True)
    y = model(x)
    y.sum().backward()

    assert x.grad is not None, "Gradients did not flow back to input"
    assert x.grad.shape == x.shape
