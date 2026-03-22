"""PyTorch Dataset classes for NPPAD data."""

from pathlib import Path

import torch
from torch.utils.data import Dataset


class NPPADWindowDataset(Dataset):
    """Dataset of sliding windows from preprocessed NPPAD data.

    Each item is a (window, label) tuple where:
        window: (window_size, n_features) tensor
        label: 0 for normal, 1 for anomaly
    """

    def __init__(self, path: str | Path):
        data = torch.load(path, weights_only=False)
        self.windows: torch.Tensor = data["windows"]
        self.labels: torch.Tensor = data["labels"]
        self.accident_types: torch.Tensor = data["accident_types"]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def n_features(self) -> int:
        return self.windows.shape[-1]

    @property
    def window_size(self) -> int:
        return self.windows.shape[1]

    def normal_only(self) -> "NPPADWindowDataset":
        """Return a new dataset containing only normal windows."""
        mask = self.labels == 0
        subset = NPPADWindowDataset.__new__(NPPADWindowDataset)
        subset.windows = self.windows[mask]
        subset.labels = self.labels[mask]
        subset.accident_types = self.accident_types[mask]
        return subset
