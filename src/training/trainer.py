"""Config-driven training loop for anomaly detection models."""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.models.anomaly import AnomalyDetector
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.config import Config
from src.utils.logging import get_logger, get_writer

logger = get_logger(__name__)


class Trainer:
    """Training loop for anomaly detection models."""

    def __init__(
        self,
        model: AnomalyDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device
        if config.train.device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.train.device)

        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )

        # Callbacks
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        self.early_stopping = EarlyStopping(patience=config.train.patience)
        self.checkpoint = ModelCheckpoint(
            path=checkpoint_dir / f"{config.model.name}_best.pt"
        )

        # Logging
        self.writer = get_writer(f"runs/{config.model.name}")

    def train(self) -> dict:
        """Run full training loop. Returns best metrics."""
        logger.info(
            f"Training {self.config.model.name} for up to {self.config.train.max_epochs} epochs"
        )

        best_val_loss = float("inf")
        for epoch in range(self.config.train.max_epochs):
            t0 = time.time()
            train_loss = self._train_epoch()
            val_loss = self._validate()
            elapsed = time.time() - t0

            self.writer.add_scalar("loss/train", train_loss, epoch)
            self.writer.add_scalar("loss/val", val_loss, epoch)

            logger.info(
                f"Epoch {epoch+1}/{self.config.train.max_epochs} - "
                f"train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}, "
                f"time: {elapsed:.1f}s"
            )

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.checkpoint.save(self.model, val_loss, epoch)

            # Early stopping
            if self.early_stopping.step(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        self.writer.close()

        # Load best model
        self.checkpoint.load(self.model)
        logger.info(f"Best val_loss: {best_val_loss:.6f}")

        return {"best_val_loss": best_val_loss, "epochs_trained": epoch + 1}

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            batch = (batch[0].to(self.device), batch[1].to(self.device))
            self.optimizer.zero_grad()
            loss = self.model.training_step(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = (batch[0].to(self.device), batch[1].to(self.device))
                loss = self.model.training_step(batch)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)
