"""
NeuroScript Model Runner V2 - Using Clean Contracts

Simplified runner that uses the contract system for extensibility.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import yaml

from neuroscript_runtime.contracts import (
    DataLoaderContract,
    LossContract,
    OptimizerContract,
    CheckpointContract,
    LoggerContract,
    CheckpointMetadata,
    ContractRegistry,
)


@dataclass
class TrainingConfig:
    """Training configuration with smart defaults."""

    # Basic
    batch_size: int = 32
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimizer
    optimizer: str = "adam"
    lr: float = 0.01
    weight_decay: float = 0.01

    # Loss (can be inferred or specified)
    loss: Optional[str] = None

    # Training
    max_grad_norm: float = 1.0

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 1000
    keep_last_n: int = 3

    # Logging
    log_every: int = 100
    eval_every: int = 500
    log_file: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("training", {}))


class NeuroScriptRunner:
    """
    Minimal viable runner for NeuroScript models.

    Uses contract system for extensibility:
    - DataLoaderContract for data loading
    - LossContract for loss computation
    - OptimizerContract for optimization
    - CheckpointContract for saving/loading
    - LoggerContract for logging
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoaderContract,
        val_loader: Optional[DataLoaderContract] = None,
        loss_fn: Optional[LossContract] = None,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Move model to device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Setup optimizer
        optimizer_cls = ContractRegistry.get_optimizer(config.optimizer)
        self.optimizer = optimizer_cls.from_config(
            model=self.model,
            optimizer_type=config.optimizer,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Setup loss function
        if loss_fn is None:
            # Use default from config or infer
            loss_type = config.loss or "mse"  # Default to MSE
            loss_cls = ContractRegistry.get_loss(loss_type)
            self.loss_fn = loss_cls.from_config(loss_type)
        else:
            self.loss_fn = loss_fn

        # Setup checkpoint manager
        checkpoint_cls = ContractRegistry.get_checkpoint("torch")
        self.checkpoint_manager = checkpoint_cls()

        # Setup logger
        logger_cls = ContractRegistry.get_logger("console")
        self.logger = logger_cls.from_config({
            "log_file": config.log_file
        })

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")

    def train(self):
        """
        Main training loop.

        Simple and readable - the heart of the runner.
        """
        # Log configuration
        self.logger.log_hyperparameters({
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "lr": self.config.lr,
            "optimizer": self.config.optimizer,
            "device": str(self.device),
        })

        # Log model
        self.logger.log_model(self.model)

        # Training loop
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            epoch_loss = 0.0
            num_batches = 0
            avg_loss = 0.0  # Initialize for final checkpoint

            # Training phase
            self.model.train()
            for batch in self.train_loader:
                loss = self._train_step(batch)

                epoch_loss += loss
                num_batches += 1
                self.global_step += 1

                # Log training metrics
                if self.global_step % self.config.log_every == 0:
                    self.logger.log_metrics(
                        step=self.global_step,
                        metrics={"loss": loss},
                        phase="train"
                    )

                # Validation
                if self.val_loader and self.global_step % self.config.eval_every == 0:
                    val_metrics = self._validate()
                    self.logger.log_metrics(
                        step=self.global_step,
                        metrics=val_metrics,
                        phase="val"
                    )

                    # Save if best
                    if val_metrics["loss"] < self.best_loss:
                        self.best_loss = val_metrics["loss"]
                        self._save_checkpoint("best.pt", val_metrics)

                # Regular checkpointing
                if self.global_step % self.config.save_every == 0:
                    self._save_checkpoint(f"step_{self.global_step}.pt", {"loss": loss})

            # Epoch summary
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f"\nEpoch {epoch + 1}/{self.config.epochs} - Avg Loss: {avg_loss:.4f}")

        # Final checkpoint
        final_loss = avg_loss if num_batches > 0 else 0.0
        self._save_checkpoint("final.pt", {"loss": final_loss})

        self.logger.finish()

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        # Get inputs and targets
        inputs = batch["input"]
        targets = batch["target"]

        # Forward pass
        outputs = self.model(inputs)

        # Compute loss
        loss = self.loss_fn.compute(outputs, targets)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

        # Optimizer step
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Validation phase."""
        if not self.val_loader:
            return {}

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            inputs = batch["input"]
            targets = batch["target"]

            outputs = self.model(inputs)
            loss = self.loss_fn.compute(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        return {"loss": avg_loss}

    def _save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)

        metadata = CheckpointMetadata(
            step=self.global_step,
            epoch=self.current_epoch,
            loss=metrics.get("loss", 0.0),
            metrics=metrics,
            config={
                "lr": self.config.lr,
                "batch_size": self.config.batch_size,
            }
        )

        self.checkpoint_manager.save(
            path=checkpoint_dir / filename,
            model=self.model,
            optimizer=self.optimizer,
            metadata=metadata
        )

    def load_checkpoint(self, path: Path):
        """Load checkpoint."""
        metadata = self.checkpoint_manager.load(
            path=path,
            model=self.model,
            optimizer=self.optimizer
        )

        self.global_step = metadata.step
        self.current_epoch = metadata.epoch
        self.best_loss = metadata.loss

        print(f"Loaded checkpoint from step {self.global_step}")

    @torch.no_grad()
    def infer(self, input_data: torch.Tensor) -> torch.Tensor:
        """Run inference."""
        self.model.eval()
        input_data = input_data.to(self.device)
        return self.model(input_data)


# =============================================================================
# HELPER FUNCTION FOR COMMON USAGE
# =============================================================================

def train_from_config(
    model: nn.Module,
    config_path: Path
) -> NeuroScriptRunner:
    """
    Convenience function to train a model from a config file.

    Args:
        model: NeuroScript-generated PyTorch model
        config_path: Path to training config YAML

    Returns:
        Trained runner (can be used for inference)

    Example:
        >>> from models import XOR  # Generated by NeuroScript
        >>> model = XOR()
        >>> runner = train_from_config(model, "config.yml")
        >>> runner.infer(torch.tensor([[1.0, 0.0]]))
    """
    # Load config
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    training_config = TrainingConfig(**config_dict.get("training", {}))
    data_config = config_dict.get("data", {})

    # Create data loaders
    # For v1, we just support JSONL
    train_loader_cls = ContractRegistry.get_dataloader("jsonl")
    train_loader = train_loader_cls.from_config({
        **data_config,
        "batch_size": training_config.batch_size,
        "device": training_config.device,
    })

    val_loader = None
    if "val" in data_config:
        val_loader = train_loader_cls.from_config({
            **data_config,
            "train": data_config["val"],  # Use val path
            "batch_size": training_config.batch_size,
            "device": training_config.device,
            "shuffle": False,  # Don't shuffle validation
        })

    # Infer loss from config
    loss_fn = None
    if "loss" in config_dict.get("training", {}):
        loss_type = config_dict["training"]["loss"]
        loss_cls = ContractRegistry.get_loss(loss_type)
        loss_fn = loss_cls.from_config(loss_type)

    # Create runner
    runner = NeuroScriptRunner(
        model=model,
        config=training_config,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
    )

    # Train
    runner.train()

    return runner


# Example usage
if __name__ == "__main__":
    # This demonstrates the usage pattern
    #
    # 1. Generate PyTorch from NeuroScript:
    #    neuroscript --codegen XOR examples/01-xor.ns --output xor_model.py
    #
    # 2. Import and train:
    #    from xor_model import XOR
    #    model = XOR()
    #    runner = train_from_config(model, "examples/xor_config.yml")
    #
    # 3. Inference:
    #    result = runner.infer(torch.tensor([[1.0, 0.0]]))
    #    print(result)  # Should be close to [1.0]

    print("NeuroScript Runner V2")
    print("See examples/xor_config.yml for usage")
