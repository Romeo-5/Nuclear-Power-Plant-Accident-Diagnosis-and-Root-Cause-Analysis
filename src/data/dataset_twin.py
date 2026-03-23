"""Dataset for digital twin training: consecutive context-target window pairs."""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class DigitalTwinDataset(Dataset):
    """Dataset that returns (context_window, target_window) pairs.

    Splits each preprocessed window into a context portion (input to surrogate)
    and a target portion (ground truth future states to predict).

    context_window: (context_size, n_features) — input to the surrogate
    target_window: (prediction_horizon, n_features) — ground truth future
    """

    def __init__(self, path: str | Path, prediction_horizon: int = 10):
        data = torch.load(path, weights_only=False)
        self.windows: torch.Tensor = data["windows"]
        self.labels: torch.Tensor = data["labels"]
        self.accident_types: torch.Tensor = data["accident_types"]
        self.prediction_horizon = prediction_horizon

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.windows[idx]  # (window_size, n_features)
        split = window.shape[0] - self.prediction_horizon
        context = window[:split]  # (context_size, n_features)
        target = window[split:]  # (prediction_horizon, n_features)
        return context, target

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def n_features(self) -> int:
        return self.windows.shape[-1]

    @property
    def context_size(self) -> int:
        return self.windows.shape[1] - self.prediction_horizon
