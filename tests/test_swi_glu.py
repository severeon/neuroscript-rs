import pytest
import torch
import torch.nn as nn


class ReferenceSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.value_proj = nn.Linear(dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.silu(self.gate_proj(x))
        value = self.value_proj(x)
        return self.out_proj(gate * value)


@pytest.mark.parametrize(
    "input_shape",
    [
        (2, 16, 64),       # [batch, seq, dim]
        (4, 64),            # [batch, dim]
        (1, 8, 32, 64),    # [extra, batch, seq, dim]
    ],
    ids=["3d", "2d", "4d"],
)
def test_swi_glu_shapes(input_shape):
    dim = 64
    hidden_dim = 128
    model = ReferenceSwiGLU(dim, hidden_dim)

    x = torch.randn(*input_shape)
    y = model(x)

    assert y.shape == x.shape, (
        f"Output shape {y.shape} does not match input shape {x.shape}"
    )


def test_swi_glu_gradient():
    dim = 64
    hidden_dim = 128
    model = ReferenceSwiGLU(dim, hidden_dim)

    x = torch.randn(2, 16, dim, requires_grad=True)
    y = model(x)
    y.sum().backward()

    assert x.grad is not None, "Gradients did not flow back to input"
    assert x.grad.shape == x.shape, (
        f"Gradient shape {x.grad.shape} does not match input shape {x.shape}"
    )
