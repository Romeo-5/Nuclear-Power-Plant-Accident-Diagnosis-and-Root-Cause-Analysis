"""Experiment logging helpers."""

import logging
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def get_writer(log_dir: str = "runs") -> SummaryWriter:
    """Get a TensorBoard writer."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)
