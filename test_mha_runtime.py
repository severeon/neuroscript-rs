#!/usr/bin/env python3
"""Test the MultiHeadSelfAttention primitive at runtime."""

import torch
from neuroscript_runtime.primitives.attention import MultiHeadSelfAttention

def test_multi_head_self_attention():
    """Test MultiHeadSelfAttention with various batch and sequence sizes."""

    # Test parameters
    d_model = 512
    num_heads = 8
    batch_size = 32
    seq_len = 10

    # Create the module
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)

    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = mha(x)

    # Verify output shape
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Expected shape ({batch_size}, {seq_len}, {d_model}), got {output.shape}"

    print(f"✓ Test passed! Output shape: {output.shape}")

    # Test with different batch dimensions
    print("\nTesting with various input shapes:")

    test_cases = [
        (1, 5, d_model),      # Small batch
        (64, 128, d_model),   # Large batch and sequence
        (16, 32, d_model),    # Medium batch
    ]

    for shape in test_cases:
        x = torch.randn(*shape)
        output = mha(x)
        assert output.shape == shape, f"Shape mismatch: expected {shape}, got {output.shape}"
        print(f"  ✓ Input shape {shape} → Output shape {output.shape}")

    # Test parameter count
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Expected parameters:
    # - QKV projection: d_model * (3 * d_model) = 512 * 1536 = 786,432
    # - QKV bias: 3 * d_model = 1,536
    # - Output projection: d_model * d_model = 262,144
    # - Output bias: d_model = 512
    # Total: 1,050,624
    expected_params = (d_model * 3 * d_model) + (3 * d_model) + (d_model * d_model) + d_model
    assert total_params == expected_params, \
        f"Parameter count mismatch: expected {expected_params:,}, got {total_params:,}"
    print(f"✓ Parameter count correct: {total_params:,} parameters")

    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_multi_head_self_attention()
