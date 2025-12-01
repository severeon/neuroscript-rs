"""
Contracts between NeuroScript compiler and Python runner.

This module defines the interfaces that make the runner extensible.
Each contract has exactly ONE default implementation for v1,
but the design makes it easy for the community to add more.

Extension points:
1. DataLoader - how to load training data
2. Loss - how to compute error
3. Optimizer - how to update weights
4. Checkpoint - how to save/load models
5. Logger - how to track training progress
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, Optional, Tuple
from pathlib import Path
import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# 1. DATA LOADER CONTRACT
# =============================================================================

@dataclass
class TrainingSample:
    """A single training sample."""
    input: torch.Tensor
    target: torch.Tensor
    metadata: Optional[Dict[str, Any]] = None


class DataLoaderContract(ABC):
    """
    Contract for loading training data.

    v1 default: JSONLDataLoader (reads .jsonl files)
    Future: CSVDataLoader, ParquetDataLoader, HuggingFaceDataLoader, etc.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yield batches of data.

        Returns:
            Dict with keys 'input' and 'target', both tensors
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Number of batches."""
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "DataLoaderContract":
        """
        Factory method to create loader from config.

        Args:
            config: Data configuration from YAML

        Returns:
            Configured data loader
        """
        pass

    @classmethod
    @abstractmethod
    def can_handle(cls, config: Dict[str, Any]) -> bool:
        """
        Check if this loader can handle the given config.

        Args:
            config: Data configuration

        Returns:
            True if this loader can handle it
        """
        pass


# =============================================================================
# 2. LOSS FUNCTION CONTRACT
# =============================================================================

class LossType(Enum):
    """Supported loss functions."""
    CROSS_ENTROPY = "cross_entropy"
    BCE = "bce"  # Binary Cross Entropy
    MSE = "mse"  # Mean Squared Error
    MAE = "mae"  # Mean Absolute Error


class LossContract(ABC):
    """
    Contract for computing loss.

    v1 defaults: CrossEntropy, BCE, MSE
    Future: Custom losses, perceptual losses, contrastive losses, etc.
    """

    @abstractmethod
    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss between predictions and targets.

        Args:
            predictions: Model outputs
            targets: Ground truth

        Returns:
            Scalar loss tensor
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, loss_type: str, **kwargs) -> "LossContract":
        """Create loss from configuration."""
        pass


# =============================================================================
# 3. OPTIMIZER CONTRACT
# =============================================================================

class OptimizerType(Enum):
    """Supported optimizers."""
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class OptimizerContract(ABC):
    """
    Contract for optimizers.

    v1 defaults: AdamW, Adam, SGD
    Future: LAMB, Lion, custom optimizers, etc.
    """

    @abstractmethod
    def step(self) -> None:
        """Perform single optimization step."""
        pass

    @abstractmethod
    def zero_grad(self) -> None:
        """Zero out gradients."""
        pass

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state for checkpointing."""
        pass

    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load optimizer state from checkpoint."""
        pass

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        model: nn.Module,
        optimizer_type: str,
        lr: float,
        **kwargs
    ) -> "OptimizerContract":
        """Create optimizer from configuration."""
        pass


# =============================================================================
# 4. CHECKPOINT CONTRACT
# =============================================================================

@dataclass
class CheckpointMetadata:
    """Metadata saved with checkpoint."""
    step: int
    epoch: int
    loss: float
    metrics: Dict[str, float]
    config: Dict[str, Any]


class CheckpointContract(ABC):
    """
    Contract for saving/loading model checkpoints.

    v1 default: TorchCheckpoint (uses torch.save/load)
    Future: SafeTensorsCheckpoint, HuggingFaceCheckpoint, etc.
    """

    @abstractmethod
    def save(
        self,
        path: Path,
        model: nn.Module,
        optimizer: Optional[OptimizerContract],
        metadata: CheckpointMetadata
    ) -> None:
        """
        Save checkpoint to disk.

        Args:
            path: Where to save
            model: Model to save
            optimizer: Optimizer to save (optional)
            metadata: Training metadata
        """
        pass

    @abstractmethod
    def load(
        self,
        path: Path,
        model: nn.Module,
        optimizer: Optional[OptimizerContract] = None
    ) -> CheckpointMetadata:
        """
        Load checkpoint from disk.

        Args:
            path: Where to load from
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)

        Returns:
            Checkpoint metadata
        """
        pass

    @abstractmethod
    def list_checkpoints(self, directory: Path) -> list[Tuple[Path, CheckpointMetadata]]:
        """
        List all checkpoints in directory.

        Args:
            directory: Directory to search

        Returns:
            List of (path, metadata) tuples sorted by step
        """
        pass


# =============================================================================
# 5. LOGGER CONTRACT
# =============================================================================

class LoggerContract(ABC):
    """
    Contract for logging training progress.

    v1 default: ConsoleLogger (prints to stdout)
    Future: WandBLogger, TensorBoardLogger, MLFlowLogger, etc.
    """

    @abstractmethod
    def log_metrics(
        self,
        step: int,
        metrics: Dict[str, float],
        phase: str = "train"
    ) -> None:
        """
        Log metrics at a training step.

        Args:
            step: Global training step
            metrics: Metrics to log (loss, accuracy, etc.)
            phase: Training phase (train, val, test)
        """
        pass

    @abstractmethod
    def log_hyperparameters(self, config: Dict[str, Any]) -> None:
        """Log hyperparameters at start of training."""
        pass

    @abstractmethod
    def log_model(self, model: nn.Module) -> None:
        """Log model architecture."""
        pass

    @abstractmethod
    def finish(self) -> None:
        """Cleanup and finalize logging."""
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "LoggerContract":
        """Create logger from configuration."""
        pass


# =============================================================================
# REGISTRY - Where implementations are registered
# =============================================================================

class ContractRegistry:
    """
    Central registry for all contract implementations.

    This makes it easy for users to add their own implementations:

        from neuroscript_runtime.contracts import ContractRegistry
        from my_package import MyCustomDataLoader

        ContractRegistry.register_dataloader("custom", MyCustomDataLoader)
    """

    _dataloaders: Dict[str, type[DataLoaderContract]] = {}
    _losses: Dict[str, type[LossContract]] = {}
    _optimizers: Dict[str, type[OptimizerContract]] = {}
    _checkpoints: Dict[str, type[CheckpointContract]] = {}
    _loggers: Dict[str, type[LoggerContract]] = {}

    @classmethod
    def register_dataloader(cls, name: str, impl: type[DataLoaderContract]):
        """Register a data loader implementation."""
        cls._dataloaders[name] = impl

    @classmethod
    def register_loss(cls, name: str, impl: type[LossContract]):
        """Register a loss function implementation."""
        cls._losses[name] = impl

    @classmethod
    def register_optimizer(cls, name: str, impl: type[OptimizerContract]):
        """Register an optimizer implementation."""
        cls._optimizers[name] = impl

    @classmethod
    def register_checkpoint(cls, name: str, impl: type[CheckpointContract]):
        """Register a checkpoint implementation."""
        cls._checkpoints[name] = impl

    @classmethod
    def register_logger(cls, name: str, impl: type[LoggerContract]):
        """Register a logger implementation."""
        cls._loggers[name] = impl

    @classmethod
    def get_dataloader(cls, name: str) -> type[DataLoaderContract]:
        """Get data loader implementation by name."""
        if name not in cls._dataloaders:
            raise ValueError(f"Unknown data loader: {name}")
        return cls._dataloaders[name]

    @classmethod
    def get_loss(cls, name: str) -> type[LossContract]:
        """Get loss implementation by name."""
        if name not in cls._losses:
            raise ValueError(f"Unknown loss: {name}")
        return cls._losses[name]

    @classmethod
    def get_optimizer(cls, name: str) -> type[OptimizerContract]:
        """Get optimizer implementation by name."""
        if name not in cls._optimizers:
            raise ValueError(f"Unknown optimizer: {name}")
        return cls._optimizers[name]

    @classmethod
    def get_checkpoint(cls, name: str) -> type[CheckpointContract]:
        """Get checkpoint implementation by name."""
        if name not in cls._checkpoints:
            raise ValueError(f"Unknown checkpoint: {name}")
        return cls._checkpoints[name]

    @classmethod
    def get_logger(cls, name: str) -> type[LoggerContract]:
        """Get logger implementation by name."""
        if name not in cls._loggers:
            raise ValueError(f"Unknown logger: {name}")
        return cls._loggers[name]

    @classmethod
    def list_available(cls) -> Dict[str, list[str]]:
        """List all available implementations."""
        return {
            "dataloaders": list(cls._dataloaders.keys()),
            "losses": list(cls._losses.keys()),
            "optimizers": list(cls._optimizers.keys()),
            "checkpoints": list(cls._checkpoints.keys()),
            "loggers": list(cls._loggers.keys()),
        }


# Example usage by community:
#
# from neuroscript_runtime.contracts import DataLoaderContract, ContractRegistry
#
# class MyHuggingFaceLoader(DataLoaderContract):
#     def __iter__(self): ...
#     def __len__(self): ...
#     @classmethod
#     def from_config(cls, config): ...
#     @classmethod
#     def can_handle(cls, config): ...
#
# ContractRegistry.register_dataloader("huggingface", MyHuggingFaceLoader)
#
# Then in config:
# data:
#   format: huggingface
#   dataset: "wikitext"
