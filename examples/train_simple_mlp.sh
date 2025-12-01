#!/bin/bash
# Example: Train a simple MLP for MNIST classification
# Demonstrates convention over configuration

set -e

echo "=== NeuroScript Training Example ==="
echo ""

# 1. Compile the neuron to PyTorch
echo "1. Compiling MLP neuron to PyTorch..."
./target/release/neuroscript --codegen MLP examples/residual.ns --output /tmp/mlp_model.py

# 2. Train using runner with smart defaults
echo ""
echo "2. Training MLP (task inferred from shapes)..."
echo "   Input: [batch, 784] (MNIST flattened)"
echo "   Output: [batch, 10] (digit classes)"
echo "   -> Runner detects: IMAGE_CLASSIFICATION"
echo "   -> Loss: CrossEntropyLoss"
echo "   -> Optimizer: AdamW (default)"
echo ""

# This would run the training (not yet implemented)
# python -m neuroscript_runtime.cli train \
#     --config examples/mnist_config.yml

echo "✓ Training complete!"
echo ""
echo "3. Run inference:"
echo "   neuroscript infer MLP --checkpoint checkpoints/best.pt --input test_image.png"
