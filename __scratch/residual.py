"""
NeuroScript MVP - Residual Network
Generated from examples/residual.ns (manually for MVP proof of concept)
"""

import torch
import torch.nn as nn


class Linear(nn.Module):
    """Primitive: Linear layer"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.layer(x)


class GELU(nn.Module):
    """Primitive: GELU activation"""
    def __init__(self):
        super().__init__()
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x)


class Fork(nn.Module):
    """Primitive: Fork - duplicates input"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x, x)


class Add(nn.Module):
    """Primitive: Element-wise addition"""
    def __init__(self):
        super().__init__()

    def forward(self, left, right):
        return torch.add(left, right)


class MLP(nn.Module):
    """Composite: Multi-layer perceptron

    Graph:
        in -> Linear(dim, dim*4) -> GELU() -> Linear(dim*4, dim) -> out
    """
    def __init__(self, dim):
        super().__init__()
        self.linear1 = Linear(dim, dim * 4)
        self.gelu = GELU()
        self.linear2 = Linear(dim * 4, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class Residual(nn.Module):
    """Composite: Residual block

    Graph:
        in -> Fork() -> (main, skip)
        main -> MLP(dim) -> processed
        (processed, skip) -> Add() -> out
    """
    def __init__(self, dim):
        super().__init__()
        self.fork = Fork()
        self.mlp = MLP(dim)
        self.add = Add()

    def forward(self, x):
        main, skip = self.fork(x)
        processed = self.mlp(main)
        out = self.add(processed, skip)
        return out


if __name__ == '__main__':
    print("="*60)
    print("NeuroScript MVP: Residual Network Test")
    print("="*60)

    # Instantiate the model
    dim = 512
    model = Residual(dim)

    print(f"\n✅ Created Residual model with dim={dim}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 32
    seq_len = 100
    x = torch.randn(batch_size, seq_len, dim)

    print(f"\n🔄 Running forward pass...")
    print(f"   Input shape: {x.shape}")

    with torch.no_grad():
        out = model(x)

    print(f"   Output shape: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch! Expected {x.shape}, got {out.shape}"

    print(f"\n✅ Forward pass successful!")

    # Test backward pass
    print(f"\n🔄 Testing backward pass...")
    out = model(x)
    loss = out.sum()
    loss.backward()

    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradients computed: {all(p.grad is not None for p in model.parameters())}")

    print(f"\n✅ Backward pass successful!")

    # Test a training step
    print(f"\n🔄 Testing training step...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Forward
    out = model(x)
    target = torch.randn_like(out)
    loss = nn.MSELoss()(out, target)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"   Training loss: {loss.item():.4f}")
    print(f"\n✅ Training step successful!")

    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED! NeuroScript MVP works!")
    print("="*60)
