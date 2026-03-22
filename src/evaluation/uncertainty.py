"""Uncertainty estimation for anomaly detection models."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.anomaly import AnomalyDetector


def mc_dropout_scores(
    model: AnomalyDetector,
    dataloader: DataLoader,
    n_samples: int = 20,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute anomaly scores with MC Dropout uncertainty.

    Args:
        model: Anomaly detection model with dropout layers
        dataloader: DataLoader for evaluation data
        n_samples: Number of stochastic forward passes
        device: Device to run inference on

    Returns:
        mean_scores: (N,) mean anomaly scores across MC samples
        std_scores: (N,) std of anomaly scores (uncertainty)
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    all_means = []
    all_stds = []

    for windows, _ in dataloader:
        windows = windows.to(device)
        mean, std = model.anomaly_score_with_uncertainty(windows, n_samples)
        all_means.append(mean.cpu().numpy())
        all_stds.append(std.cpu().numpy())

    return np.concatenate(all_means), np.concatenate(all_stds)


def ensemble_scores(
    models: list[AnomalyDetector],
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute anomaly scores from an ensemble of models.

    Returns:
        mean_scores: (N,) mean scores across ensemble
        std_scores: (N,) std across ensemble (epistemic uncertainty)
    """
    device = torch.device(device)
    all_model_scores = []

    for model in models:
        model.to(device)
        model.eval()
        scores_list = []

        with torch.no_grad():
            for windows, _ in dataloader:
                windows = windows.to(device)
                scores = model.anomaly_score(windows)
                scores_list.append(scores.cpu().numpy())

        all_model_scores.append(np.concatenate(scores_list))

    stacked = np.stack(all_model_scores, axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)
