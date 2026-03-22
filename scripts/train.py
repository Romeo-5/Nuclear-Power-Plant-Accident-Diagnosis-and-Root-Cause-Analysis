"""CLI entry point for training anomaly detection models."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import NPPADWindowDataset
from src.models.anomaly import build_model
from src.training.trainer import Trainer
from src.utils.config import Config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.train.seed)

    # Load data
    processed_dir = Path(config.data.processed_dir)
    train_dataset = NPPADWindowDataset(processed_dir / "train.pt")
    val_dataset = NPPADWindowDataset(processed_dir / "val.pt")

    # For semi-supervised anomaly detection, train on normal data only
    train_normal = train_dataset.normal_only()
    logger.info(
        f"Training on {len(train_normal)} normal windows "
        f"(filtered from {len(train_dataset)} total)"
    )
    logger.info(f"Validation: {len(val_dataset)} windows")

    train_loader = DataLoader(
        train_normal,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        pin_memory=True,
    )

    # Build model
    model = build_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {config.model.name}, parameters: {n_params:,}")

    # Train
    trainer = Trainer(model, train_loader, val_loader, config)
    metrics = trainer.train()

    logger.info(f"Training complete: {metrics}")


if __name__ == "__main__":
    main()
