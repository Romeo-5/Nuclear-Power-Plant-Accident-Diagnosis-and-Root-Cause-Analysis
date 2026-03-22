"""Data transforms for time-series preprocessing."""

import pickle
from pathlib import Path

import numpy as np
import torch


class Normalizer:
    """Wraps a fitted sklearn scaler for use with torch tensors."""

    def __init__(self, scaler_path: str | Path):
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a (*, n_features) tensor."""
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).numpy()
        scaled = self.scaler.transform(flat).astype(np.float32)
        return torch.from_numpy(scaled).reshape(shape)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform a normalized tensor."""
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).numpy()
        unscaled = self.scaler.inverse_transform(flat).astype(np.float32)
        return torch.from_numpy(unscaled).reshape(shape)


def sliding_window(
    data: np.ndarray, window_size: int, stride: int = 1
) -> np.ndarray:
    """Create sliding windows from a 2D array.

    Args:
        data: (n_timesteps, n_features) array
        window_size: number of timesteps per window
        stride: step size between windows

    Returns:
        (n_windows, window_size, n_features) array
    """
    n_windows = (len(data) - window_size) // stride + 1
    windows = np.array([
        data[i * stride : i * stride + window_size]
        for i in range(n_windows)
    ])
    return windows
