"""Train the digital twin surrogate model."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset_twin import DigitalTwinDataset
from src.models.digital_twin import build_surrogate
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
    parser = argparse.ArgumentParser(description="Train digital twin surrogate")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    set_seed(config.train.seed)

    processed_dir = Path(config.data.processed_dir)

    # Load datasets
    train_dataset = DigitalTwinDataset(
        processed_dir / "train.pt",
        prediction_horizon=config.model.prediction_horizon,
    )
    val_dataset = DigitalTwinDataset(
        processed_dir / "val.pt",
        prediction_horizon=config.model.prediction_horizon,
    )

    logger.info(
        f"Train: {len(train_dataset)} windows, Val: {len(val_dataset)} windows, "
        f"Features: {train_dataset.n_features}, "
        f"Context: {train_dataset.context_size}, "
        f"Horizon: {config.model.prediction_horizon}"
    )

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

    # Build model with physics loss
    model = build_surrogate(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {config.model.name}, parameters: {n_params:,}")
    logger.info(
        f"Physics weights — data: {config.physics.lambda_data}, "
        f"physics: {config.physics.lambda_physics}, "
        f"conservation: {config.physics.lambda_conservation}, "
        f"bounds: {config.physics.lambda_bounds}"
    )

    # Train
    trainer = Trainer(model, train_loader, val_loader, config)
    metrics = trainer.train()
    logger.info(f"Training complete: {metrics}")


if __name__ == "__main__":
    main()
