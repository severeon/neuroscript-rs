import pytest
import torch
import torch.nn as nn


class ReferenceInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3, out_5x5, out_pool):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, 1), nn.BatchNorm2d(out_1x1), nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3, 3, padding=1), nn.BatchNorm2d(out_3x3), nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5, 5, padding=2), nn.BatchNorm2d(out_5x5), nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, 1), nn.BatchNorm2d(out_pool), nn.ReLU()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


@pytest.mark.parametrize(
    "batch,h,w",
    [(2, 16, 16), (1, 32, 32), (4, 8, 8)],
    ids=["16x16", "32x32", "8x8"],
)
def test_inception_block_shapes(batch, h, w):
    in_c, o1, o3, o5, op = 64, 16, 32, 8, 8
    model = ReferenceInceptionBlock(in_c, o1, o3, o5, op)

    x = torch.randn(batch, in_c, h, w)
    y = model(x)

    expected_channels = o1 + o3 + o5 + op
    assert y.shape == (batch, expected_channels, h, w), (
        f"Output shape {y.shape} != expected ({batch}, {expected_channels}, {h}, {w})"
    )


def test_inception_block_gradient():
    in_c, o1, o3, o5, op = 32, 8, 16, 4, 4
    model = ReferenceInceptionBlock(in_c, o1, o3, o5, op)

    x = torch.randn(2, in_c, 8, 8, requires_grad=True)
    y = model(x)
    y.sum().backward()

    assert x.grad is not None, "Gradients did not flow back to input"
    assert x.grad.shape == x.shape
