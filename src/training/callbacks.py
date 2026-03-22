"""Training callbacks: early stopping and model checkpointing."""

from pathlib import Path

import torch
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            return True
        return False


class ModelCheckpoint:
    """Save best model based on validation loss."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, model: nn.Module, val_loss: float, epoch: int) -> None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
            },
            self.path,
        )
        logger.info(f"Saved checkpoint to {self.path} (val_loss={val_loss:.6f})")

    def load(self, model: nn.Module) -> None:
        if self.path.exists():
            checkpoint = torch.load(self.path, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded checkpoint from {self.path}")
        else:
            logger.warning(f"No checkpoint found at {self.path}")
