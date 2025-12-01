"""
NeuroScript Model Runner - Convention over Configuration

Usage:
    # Training
    python -m neuroscript_runtime.runner train \
        --model models.gpt2:GPT2Small \
        --data train.jsonl \
        --config training.yml

    # Inference
    python -m neuroscript_runtime.runner infer \
        --model models.gpt2:GPT2Small \
        --checkpoint checkpoints/best.pt \
        --input "Hello world"
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import json
from dataclasses import dataclass
from enum import Enum


class TaskType(Enum):
    """Inferred from model input/output shapes."""
    LANGUAGE_MODELING = "language_modeling"  # [batch, seq] -> [batch, seq, vocab]
    SEQUENCE_CLASSIFICATION = "seq_classification"  # [batch, seq] -> [batch, classes]
    IMAGE_CLASSIFICATION = "image_classification"  # [batch, C, H, W] -> [batch, classes]
    REGRESSION = "regression"  # [batch, features] -> [batch, 1]


@dataclass
class ModelSignature:
    """Extract model signature from forward pass inspection."""
    input_shape: tuple
    output_shape: tuple
    task_type: TaskType

    @classmethod
    def infer_from_model(cls, model: nn.Module, sample_input: torch.Tensor) -> "ModelSignature":
        """Infer model signature by running a forward pass."""
        model.eval()
        with torch.no_grad():
            output = model(sample_input)

        in_shape = sample_input.shape
        out_shape = output.shape

        # Infer task from shapes
        task = cls._infer_task(in_shape, out_shape)

        return cls(input_shape=in_shape, output_shape=out_shape, task_type=task)

    @staticmethod
    def _infer_task(in_shape: tuple, out_shape: tuple) -> TaskType:
        """Infer task type from input/output shapes."""
        # Language modeling: [batch, seq] -> [batch, seq, vocab]
        if len(in_shape) == 2 and len(out_shape) == 3:
            if out_shape[1] == in_shape[1]:  # seq length matches
                return TaskType.LANGUAGE_MODELING

        # Image classification: [batch, C, H, W] -> [batch, num_classes]
        if len(in_shape) == 4 and len(out_shape) == 2:
            return TaskType.IMAGE_CLASSIFICATION

        # Sequence classification: [batch, seq] -> [batch, num_classes]
        if len(in_shape) == 2 and len(out_shape) == 2:
            if out_shape[1] < 1000:  # heuristic: likely classes, not features
                return TaskType.SEQUENCE_CLASSIFICATION

        # Default to regression
        return TaskType.REGRESSION


@dataclass
class TrainingConfig:
    """Training configuration with smart defaults."""

    # Optimizer defaults
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)

    # Schedule defaults
    warmup_steps: int = 1000
    max_steps: Optional[int] = None

    # Training loop
    batch_size: int = 32
    epochs: int = 10
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 1000
    keep_last_n: int = 3

    # Logging
    log_every: int = 100
    eval_every: int = 1000

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # Use automatic mixed precision

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("training", {}))


class NeuroScriptRunner:
    """
    Opinionated runner for NeuroScript models.

    Conventions:
    - Infers task from model signature
    - Chooses appropriate loss and metrics
    - Provides sensible defaults for optimizers/schedules
    - Auto-checkpoints best model
    - Handles device placement
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)

        # Move model to device
        self.model.to(self.device)

        # Will be set during training
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.signature = None

    def infer_task(self, sample_input: torch.Tensor) -> TaskType:
        """Infer task type from model signature."""
        self.signature = ModelSignature.infer_from_model(self.model, sample_input)
        return self.signature.task_type

    def setup_training(self, task_type: TaskType):
        """Setup loss, optimizer, metrics based on task."""

        # Setup loss function
        if task_type == TaskType.LANGUAGE_MODELING:
            # CrossEntropy for next token prediction
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        elif task_type in (TaskType.SEQUENCE_CLASSIFICATION, TaskType.IMAGE_CLASSIFICATION):
            self.loss_fn = nn.CrossEntropyLoss()
        elif task_type == TaskType.REGRESSION:
            self.loss_fn = nn.MSELoss()

        # Setup optimizer (AdamW is good default for transformers)
        if self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                betas=self.config.betas,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                betas=self.config.betas,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        # Setup learning rate scheduler
        # Linear warmup + cosine decay is a good default
        from torch.optim.lr_scheduler import OneCycleLR

        if self.config.max_steps:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr,
                total_steps=self.config.max_steps,
                pct_start=self.config.warmup_steps / self.config.max_steps,
            )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Move batch to device
        inputs = batch["input"].to(self.device)
        targets = batch["target"].to(self.device)

        # Forward pass
        outputs = self.model(inputs)

        # Compute loss (reshape for language modeling)
        if self.signature.task_type == TaskType.LANGUAGE_MODELING:
            # outputs: [batch, seq, vocab]
            # targets: [batch, seq]
            loss = self.loss_fn(
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1)
            )
        else:
            loss = self.loss_fn(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.scheduler:
            self.scheduler.step()

        return {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single evaluation step."""
        self.model.eval()

        inputs = batch["input"].to(self.device)
        targets = batch["target"].to(self.device)

        outputs = self.model(inputs)

        # Compute loss
        if self.signature and self.loss_fn and self.signature.task_type == TaskType.LANGUAGE_MODELING:
            loss = self.loss_fn(
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1)
            )

            # Compute perplexity
            perplexity = torch.exp(loss)

            return {"loss": loss.item(), "perplexity": perplexity.item()}
        else:
            loss = self.loss_fn(outputs, targets)

            # Compute accuracy for classification
            if self.signature.task_type in (TaskType.SEQUENCE_CLASSIFICATION, TaskType.IMAGE_CLASSIFICATION):
                preds = outputs.argmax(dim=-1)
                accuracy = (preds == targets).float().mean()
                return {"loss": loss.item(), "accuracy": accuracy.item()}

            return {"loss": loss.item()}

    def save_checkpoint(self, path: Path, metadata: Optional[Dict[str, Any]] = None):
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "config": self.config.__dict__,
            "metadata": metadata or {},
        }

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Loaded checkpoint from {path}")
        return checkpoint.get("metadata", {})

    @torch.no_grad()
    def infer(self, input_data: torch.Tensor) -> torch.Tensor:
        """Run inference on input data."""
        self.model.eval()
        input_data = input_data.to(self.device)
        return self.model(input_data)


# Example usage
if __name__ == "__main__":
    # This would be called from CLI
    # neuroscript run --train GPT2Small --data train.jsonl

    # 1. Load generated model
    from models import GPT2Small  # Generated by NeuroScript codegen

    model = GPT2Small(vocab_size=50257)

    # 2. Create runner with default config
    config = TrainingConfig(
        batch_size=32,
        epochs=10,
        lr=3e-4,
    )
    runner = NeuroScriptRunner(model, config)

    # 3. Infer task from dummy input
    dummy_input = torch.randint(0, 50257, (32, 128))  # [batch, seq]
    task_type = runner.infer_task(dummy_input)
    print(f"Detected task: {task_type}")

    # 4. Setup training (loss, optimizer, etc.)
    runner.setup_training(task_type)

    # 5. Training loop (simplified)
    # for epoch in range(config.epochs):
    #     for batch in train_loader:
    #         metrics = runner.train_step(batch)
    #         if step % config.log_every == 0:
    #             print(f"Step {step}: {metrics}")
    #         if step % config.save_every == 0:
    #             runner.save_checkpoint(f"checkpoint_{step}.pt")
