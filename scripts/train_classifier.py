"""CLI entry point for training accident classification models (Module 2)."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import NPPADWindowDataset
from src.models.diagnosis import build_classifier
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
    parser = argparse.ArgumentParser(description="Train accident classifier")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.train.seed)

    # Load data — use ALL data for classification (supervised)
    processed_dir = Path(config.data.processed_dir)
    train_dataset = NPPADWindowDataset(processed_dir / "train.pt")
    val_dataset = NPPADWindowDataset(processed_dir / "val.pt")

    # For classification, labels are accident_types (not binary anomaly labels)
    # Swap labels to accident_types
    train_dataset.labels = train_dataset.accident_types
    val_dataset.labels = val_dataset.accident_types

    logger.info(f"Training on {len(train_dataset)} windows ({config.model.n_classes} classes)")
    logger.info(f"Validation: {len(val_dataset)} windows")

    # Log class distribution
    unique, counts = torch.unique(train_dataset.labels, return_counts=True)
    for cls_id, count in zip(unique.tolist(), counts.tolist()):
        logger.info(f"  Class {cls_id}: {count} windows")

    train_loader = DataLoader(
        train_dataset,
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
    model = build_classifier(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {config.model.name}, parameters: {n_params:,}")

    # Train
    trainer = Trainer(model, train_loader, val_loader, config)
    metrics = trainer.train()

    logger.info(f"Training complete: {metrics}")


if __name__ == "__main__":
    main()
