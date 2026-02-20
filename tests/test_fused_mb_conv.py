import pytest
import torch
import torch.nn as nn


class ReferenceSEBlock(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = torch.sigmoid(self.fc2(torch.relu(self.fc1(w))))
        return x * w.view(b, c, 1, 1)


class ReferenceFusedMBConv(nn.Module):
    def __init__(self, channels, expansion, reduction):
        super().__init__()
        mid = channels * expansion
        self.expand = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.SiLU()
        )
        self.se = ReferenceSEBlock(mid, reduction)
        self.project = nn.Sequential(
            nn.Conv2d(mid, channels, 1), nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        out = self.expand(x)
        out = self.se(out)
        out = self.project(out)
        return out + x


@pytest.mark.parametrize(
    "batch,h,w",
    [(2, 16, 16), (1, 32, 32), (4, 8, 8)],
    ids=["16x16", "32x32", "8x8"],
)
def test_fused_mb_conv_shapes(batch, h, w):
    channels, expansion, reduction = 32, 4, 4
    model = ReferenceFusedMBConv(channels, expansion, reduction)

    x = torch.randn(batch, channels, h, w)
    y = model(x)

    assert y.shape == x.shape, (
        f"Output shape {y.shape} does not match input shape {x.shape}"
    )


def test_fused_mb_conv_gradient():
    channels, expansion, reduction = 16, 4, 4
    model = ReferenceFusedMBConv(channels, expansion, reduction)

    x = torch.randn(2, channels, 8, 8, requires_grad=True)
    y = model(x)
    y.sum().backward()

    assert x.grad is not None, "Gradients did not flow back to input"
    assert x.grad.shape == x.shape
