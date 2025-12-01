"""
Default implementations for all contracts.

This is the "batteries included" part - opinionated but reasonable defaults.
Each can be replaced by community implementations via the ContractRegistry.
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, Tuple
from datetime import datetime

from neuroscript_runtime.contracts import (
    DataLoaderContract,
    LossContract,
    OptimizerContract,
    CheckpointContract,
    LoggerContract,
    CheckpointMetadata,
    ContractRegistry,
)


# =============================================================================
# 1. DEFAULT DATA LOADER: JSONL
# =============================================================================

class JSONLDataLoader(DataLoaderContract):
    """
    Default data loader for .jsonl files.

    Expected format:
        {"input": [1.0, 2.0], "target": [3.0]}
        {"input": [0.0, 1.0], "target": [1.0]}

    Features:
    - Automatic batching
    - Shuffling (optional)
    - CPU/GPU device placement
    """

    def __init__(
        self,
        file_path: Path,
        batch_size: int = 32,
        shuffle: bool = True,
        device: str = "cpu",
    ):
        self.file_path = Path(file_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = torch.device(device)

        # Load all data into memory (simple for v1)
        self.samples = []
        with open(self.file_path) as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)

        if not self.samples:
            raise ValueError(f"No samples found in {file_path}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield batches."""
        indices = list(range(len(self.samples)))

        if self.shuffle:
            import random
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_samples = [self.samples[j] for j in batch_indices]

            # Convert to tensors
            inputs = torch.tensor(
                [s["input"] for s in batch_samples],
                dtype=torch.float32,
                device=self.device
            )
            targets = torch.tensor(
                [s["target"] for s in batch_samples],
                dtype=torch.float32,
                device=self.device
            )

            yield {"input": inputs, "target": targets}

    def __len__(self) -> int:
        """Number of batches."""
        return (len(self.samples) + self.batch_size - 1) // self.batch_size

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "JSONLDataLoader":
        """Create from config dict."""
        return cls(
            file_path=config["train"],
            batch_size=config.get("batch_size", 32),
            shuffle=config.get("shuffle", True),
            device=config.get("device", "cpu"),
        )

    @classmethod
    def can_handle(cls, config: Dict[str, Any]) -> bool:
        """Check if we can handle this config."""
        train_path = Path(config.get("train", ""))
        return train_path.suffix == ".jsonl"


# =============================================================================
# 2. DEFAULT LOSSES
# =============================================================================

class CrossEntropyLoss(LossContract):
    """Cross entropy loss for classification."""

    def __init__(self, ignore_index: int = -100):
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross entropy loss."""
        return self.loss_fn(predictions, targets)

    @classmethod
    def from_config(cls, loss_type: str, **kwargs) -> "CrossEntropyLoss":
        """Create from config."""
        return cls(**kwargs)


class BCELoss(LossContract):
    """Binary cross entropy loss."""

    def __init__(self):
        self.loss_fn = nn.BCELoss()

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss."""
        return self.loss_fn(predictions, targets)

    @classmethod
    def from_config(cls, loss_type: str, **kwargs) -> "BCELoss":
        """Create from config."""
        return cls(**kwargs)


class MSELoss(LossContract):
    """Mean squared error loss for regression."""

    def __init__(self):
        self.loss_fn = nn.MSELoss()

    def compute(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss."""
        return self.loss_fn(predictions, targets)

    @classmethod
    def from_config(cls, loss_type: str, **kwargs) -> "MSELoss":
        """Create from config."""
        return cls(**kwargs)


# =============================================================================
# 3. DEFAULT OPTIMIZERS
# =============================================================================

class AdamWOptimizer(OptimizerContract):
    """AdamW optimizer wrapper."""

    def __init__(self, optimizer: torch.optim.AdamW):
        self.optimizer = optimizer

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state)

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        optimizer_type: str,
        lr: float,
        **kwargs
    ) -> "AdamWOptimizer":
        """Create AdamW from config."""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=kwargs.get("weight_decay", 0.01),
            betas=kwargs.get("betas", (0.9, 0.999)),
        )
        return cls(optimizer)


class AdamOptimizer(OptimizerContract):
    """Adam optimizer wrapper."""

    def __init__(self, optimizer: torch.optim.Adam):
        self.optimizer = optimizer

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state)

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        optimizer_type: str,
        lr: float,
        **kwargs
    ) -> "AdamOptimizer":
        """Create Adam from config."""
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=kwargs.get("betas", (0.9, 0.999)),
        )
        return cls(optimizer)


# =============================================================================
# 4. DEFAULT CHECKPOINT: Torch Save/Load
# =============================================================================

class TorchCheckpoint(CheckpointContract):
    """Default checkpoint using torch.save/load."""

    def save(
        self,
        path: Path,
        model: nn.Module,
        optimizer: Optional[OptimizerContract],
        metadata: CheckpointMetadata
    ) -> None:
        """Save checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "metadata": {
                "step": metadata.step,
                "epoch": metadata.epoch,
                "loss": metadata.loss,
                "metrics": metadata.metrics,
                "config": metadata.config,
                "timestamp": datetime.now().isoformat(),
            }
        }

        torch.save(checkpoint, path)

    def load(
        self,
        path: Path,
        model: nn.Module,
        optimizer: Optional[OptimizerContract] = None
    ) -> CheckpointMetadata:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        meta = checkpoint["metadata"]
        return CheckpointMetadata(
            step=meta["step"],
            epoch=meta["epoch"],
            loss=meta["loss"],
            metrics=meta["metrics"],
            config=meta["config"],
        )

    def list_checkpoints(self, directory: Path) -> list[Tuple[Path, CheckpointMetadata]]:
        """List all checkpoints in directory."""
        if not directory.exists():
            return []

        checkpoints = []
        for path in directory.glob("*.pt"):
            try:
                checkpoint = torch.load(path, map_location="cpu")
                meta = checkpoint["metadata"]
                metadata = CheckpointMetadata(
                    step=meta["step"],
                    epoch=meta["epoch"],
                    loss=meta["loss"],
                    metrics=meta["metrics"],
                    config=meta["config"],
                )
                checkpoints.append((path, metadata))
            except Exception:
                # Skip corrupted checkpoints
                continue

        # Sort by step
        checkpoints.sort(key=lambda x: x[1].step)
        return checkpoints


# =============================================================================
# 5. DEFAULT LOGGER: Console
# =============================================================================

class ConsoleLogger(LoggerContract):
    """Simple console logger."""

    def __init__(self, log_file: Optional[Path] = None):
        self.log_file = log_file
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_metrics(
        self,
        step: int,
        metrics: Dict[str, float],
        phase: str = "train"
    ) -> None:
        """Log metrics to console."""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        message = f"[{phase.upper()}] Step {step} | {metrics_str}"

        print(message)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")

    def log_hyperparameters(self, config: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        print("\n=== Training Configuration ===")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("=" * 30 + "\n")

    def log_model(self, model: nn.Module) -> None:
        """Log model architecture."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("\n=== Model Architecture ===")
        print(model)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 30 + "\n")

    def finish(self) -> None:
        """Finish logging."""
        print("\n=== Training Complete ===\n")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConsoleLogger":
        """Create from config."""
        log_file = config.get("log_file")
        if log_file:
            log_file = Path(log_file)
        return cls(log_file=log_file)


# =============================================================================
# REGISTER ALL DEFAULTS
# =============================================================================

def register_defaults():
    """Register all default implementations."""
    # Data loaders
    ContractRegistry.register_dataloader("jsonl", JSONLDataLoader)

    # Losses
    ContractRegistry.register_loss("cross_entropy", CrossEntropyLoss)
    ContractRegistry.register_loss("bce", BCELoss)
    ContractRegistry.register_loss("mse", MSELoss)

    # Optimizers
    ContractRegistry.register_optimizer("adamw", AdamWOptimizer)
    ContractRegistry.register_optimizer("adam", AdamOptimizer)

    # Checkpoints
    ContractRegistry.register_checkpoint("torch", TorchCheckpoint)

    # Loggers
    ContractRegistry.register_logger("console", ConsoleLogger)


# Auto-register on import
register_defaults()
