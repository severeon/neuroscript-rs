#!/usr/bin/env python3
"""
End-to-end test of XOR training.

This demonstrates the complete workflow:
1. Compile NeuroScript to PyTorch
2. Load generated model
3. Train using runner
4. Test inference

Usage:
    source ~/.venv_ai/bin/activate
    python examples/test_xor_training.py
"""

import sys
import subprocess
import importlib.util
from pathlib import Path
import torch

# Add neuroscript_runtime to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuroscript_runtime.runner_v2 import train_from_config


def compile_neuron(neuron_name: str, source_file: Path, output_file: Path):
    """Compile NeuroScript neuron to PyTorch."""
    print(f"\n{'='*60}")
    print(f"STEP 1: Compiling {neuron_name} to PyTorch")
    print(f"{'='*60}")

    cmd = [
        "./target/release/neuroscript",
        "--codegen", neuron_name,
        "--output", str(output_file),
        str(source_file),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Compilation failed:")
        print(result.stderr)
        sys.exit(1)

    print(f"✓ Compiled {neuron_name} -> {output_file}")
    return output_file


def load_model(model_file: Path, neuron_name: str):
    """Dynamically load compiled model."""
    print(f"\n{'='*60}")
    print(f"STEP 2: Loading {neuron_name} class")
    print(f"{'='*60}")

    spec = importlib.util.spec_from_file_location("neuroscript_model", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_class = getattr(module, neuron_name)
    model = model_class()

    print(f"✓ Loaded {neuron_name}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def test_model(model, test_cases):
    """Test model on XOR truth table."""
    print(f"\n{'='*60}")
    print(f"STEP 4: Testing Inference")
    print(f"{'='*60}")

    model.eval()
    print("\nXOR Truth Table:")
    print("Input  | Expected | Predicted | Error")
    print("-------|----------|-----------|-------")

    with torch.no_grad():
        total_error = 0.0
        for inputs, expected in test_cases:
            input_tensor = torch.tensor([inputs], dtype=torch.float32)
            output = model(input_tensor)
            predicted = output[0, 0].item()
            error = abs(predicted - expected)
            total_error += error

            status = "✓" if error < 0.1 else "✗"
            print(f"{inputs} | {expected:8.1f} | {predicted:9.4f} | {error:.4f} {status}")

    avg_error = total_error / len(test_cases)
    print(f"\nAverage error: {avg_error:.4f}")

    if avg_error < 0.1:
        print("✓ Model learned XOR successfully!")
    else:
        print("⚠ Model needs more training or different hyperparameters")


def main():
    print("=" * 60)
    print("NeuroScript End-to-End Test: XOR Function")
    print("=" * 60)

    # Paths
    source_file = Path("examples/01-xor.ns")
    output_file = Path("/tmp/xor_model.py")
    config_file = Path("examples/xor_config.yml")

    # Compile
    compile_neuron("XOR", source_file, output_file)

    # Load model
    model = load_model(output_file, "XOR")

    # Train
    print(f"\n{'='*60}")
    print(f"STEP 3: Training XOR Model")
    print(f"{'='*60}")

    runner = train_from_config(model, config_file)

    # Test
    test_cases = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

    test_model(model, test_cases)

    print(f"\n{'='*60}")
    print("Test Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
