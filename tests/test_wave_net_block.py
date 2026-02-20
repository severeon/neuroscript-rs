import pytest
import torch
import torch.nn as nn


class ReferenceWaveNetBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        filtered = self.conv(x)
        gated = torch.tanh(filtered) * torch.sigmoid(filtered)
        projection = self.proj(gated)
        return projection + x


@pytest.mark.parametrize(
    "batch,seq,dilation",
    [(2, 64, 1), (1, 128, 4), (4, 32, 8)],
    ids=["d1", "d4", "d8"],
)
def test_wave_net_block_shapes(batch, seq, dilation):
    channels, kernel = 32, 3
    model = ReferenceWaveNetBlock(channels, kernel, dilation)

    x = torch.randn(batch, channels, seq)
    y = model(x)

    assert y.shape == x.shape, (
        f"Output shape {y.shape} does not match input shape {x.shape}"
    )


def test_wave_net_block_gradient():
    channels, kernel, dilation = 16, 3, 2
    model = ReferenceWaveNetBlock(channels, kernel, dilation)

    x = torch.randn(2, channels, 64, requires_grad=True)
    y = model(x)
    y.sum().backward()

    assert x.grad is not None, "Gradients did not flow back to input"
    assert x.grad.shape == x.shape


def test_wave_net_block_increasing_dilation():
    """Stack blocks with increasing dilation to verify composability."""
    channels, kernel = 16, 3
    blocks = nn.ModuleList([
        ReferenceWaveNetBlock(channels, kernel, d) for d in [1, 2, 4, 8]
    ])

    x = torch.randn(2, channels, 64)
    y = x
    for block in blocks:
        y = block(y)

    assert y.shape == x.shape
