#!/usr/bin/env python3
"""
NeuroScript CLI Runner

Usage:
    # Compile NeuroScript to PyTorch
    neuroscript compile GPT2Small examples/transformer.ns --output models/gpt2.py

    # Train with config file
    neuroscript train --config training.yml

    # Train with CLI args (uses defaults)
    neuroscript train GPT2Small \
        --data train.jsonl \
        --batch-size 32 \
        --epochs 10

    # Inference
    neuroscript infer GPT2Small \
        --checkpoint checkpoints/best.pt \
        --input "Hello world"

    # Interactive inference (REPL)
    neuroscript repl GPT2Small --checkpoint checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path
import subprocess
import importlib.util
import torch
import yaml


def compile_neuron(args):
    """Compile NeuroScript neuron to PyTorch."""
    cmd = [
        "./target/release/neuroscript",
        "--codegen", args.neuron,
        "--output", args.output,
        args.file,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error compiling {args.neuron}:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    print(f"✓ Compiled {args.neuron} -> {args.output}")


def load_model_from_file(model_file: Path, neuron_name: str, **params):
    """Dynamically load a NeuroScript-generated model."""
    spec = importlib.util.spec_from_file_location("neuroscript_model", model_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_class = getattr(module, neuron_name)
    return model_class(**params)


def train_command(args):
    """Run training with smart defaults."""
    from neuroscript_runtime.runner import NeuroScriptRunner, TrainingConfig

    # Load config if provided
    if args.config:
        config_path = Path(args.config)
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        model_config = config_dict.get("model", {})
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = config_dict.get("data", {})
    else:
        # Use CLI args
        model_config = {
            "neuron": args.neuron,
            "file": args.file,
            "params": {},
        }
        training_config = TrainingConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
        )
        data_config = {
            "train": args.data,
        }

    # Compile NeuroScript if needed
    model_file = Path("./generated_model.py")
    compile_args = argparse.Namespace(
        neuron=model_config["neuron"],
        file=model_config["file"],
        output=str(model_file),
    )
    compile_neuron(compile_args)

    # Load model
    model = load_model_from_file(
        model_file,
        model_config["neuron"],
        **model_config.get("params", {})
    )

    print(f"✓ Loaded model: {model_config['neuron']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create runner
    runner = NeuroScriptRunner(model, training_config)

    # Infer task from model signature
    # TODO: Create dummy input from data config
    dummy_input = torch.randint(0, 50257, (training_config.batch_size, 128))
    task_type = runner.infer_task(dummy_input)

    print(f"✓ Detected task: {task_type.value}")

    # Setup training
    runner.setup_training(task_type)

    print(f"✓ Setup complete")
    print(f"  Optimizer: {training_config.optimizer}")
    print(f"  Learning rate: {training_config.lr}")
    print(f"  Batch size: {training_config.batch_size}")
    print(f"  Device: {training_config.device}")

    # TODO: Load data
    # TODO: Training loop
    # TODO: Evaluation
    # TODO: Checkpointing

    print("\n[Training loop not yet implemented - see neuroscript_runtime/runner.py]")


def infer_command(args):
    """Run inference on trained model."""
    from neuroscript_runtime.runner import NeuroScriptRunner

    print(f"Loading model from {args.checkpoint}...")

    # TODO: Load model architecture
    # TODO: Load checkpoint
    # TODO: Run inference

    print("[Inference not yet implemented]")


def repl_command(args):
    """Interactive REPL for model."""
    print("NeuroScript REPL")
    print(f"Model: {args.neuron}")
    print(f"Checkpoint: {args.checkpoint}")
    print("\nType 'exit' to quit")

    while True:
        try:
            user_input = input("\n> ")
            if user_input.strip().lower() in ("exit", "quit"):
                break

            # TODO: Run inference on user_input
            print(f"[Output for: {user_input}]")

        except (KeyboardInterrupt, EOFError):
            break

    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="NeuroScript - Neural Architecture Composition Language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Compile command
    compile_parser = subparsers.add_parser("compile", help="Compile NeuroScript to PyTorch")
    compile_parser.add_argument("neuron", help="Neuron name to compile")
    compile_parser.add_argument("file", help="NeuroScript file")
    compile_parser.add_argument("--output", "-o", required=True, help="Output Python file")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", "-c", help="Training config YAML")
    train_parser.add_argument("neuron", nargs="?", help="Neuron name (if not in config)")
    train_parser.add_argument("--file", "-f", help="NeuroScript file")
    train_parser.add_argument("--data", help="Training data file")
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--lr", type=float, default=3e-4)

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("neuron", help="Neuron name")
    infer_parser.add_argument("--checkpoint", "-ckpt", required=True, help="Model checkpoint")
    infer_parser.add_argument("--input", "-i", required=True, help="Input data")

    # REPL command
    repl_parser = subparsers.add_parser("repl", help="Interactive REPL")
    repl_parser.add_argument("neuron", help="Neuron name")
    repl_parser.add_argument("--checkpoint", "-ckpt", required=True, help="Model checkpoint")

    args = parser.parse_args()

    if args.command == "compile":
        compile_neuron(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "infer":
        infer_command(args)
    elif args.command == "repl":
        repl_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
