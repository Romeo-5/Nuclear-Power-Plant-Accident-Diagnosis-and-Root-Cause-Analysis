"""Anomaly detection models for reactor sensor telemetry."""

import torch
import torch.nn as nn
from torch import Tensor


class AnomalyDetector(nn.Module):
    """Base class for anomaly detection models.

    All anomaly detectors follow the same interface:
    - forward(x) returns a reconstruction or prediction
    - anomaly_score(x) returns a per-window anomaly score
    - training_step(batch) returns the loss for a training batch
    """

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def anomaly_score(self, x: Tensor) -> Tensor:
        """Compute per-window anomaly score (higher = more anomalous)."""
        with torch.no_grad():
            x_hat = self.forward(x)
            # MSE per window
            return ((x - x_hat) ** 2).mean(dim=(1, 2))

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        """Compute loss for a training batch. Override for custom logic."""
        x, _ = batch
        x_hat = self.forward(x)
        return nn.functional.mse_loss(x_hat, x)

    @torch.no_grad()
    def anomaly_score_with_uncertainty(
        self, x: Tensor, n_samples: int = 20
    ) -> tuple[Tensor, Tensor]:
        """MC Dropout uncertainty estimation.

        Returns:
            mean_scores: (batch_size,) mean anomaly scores
            std_scores: (batch_size,) std of anomaly scores
        """
        # Enable dropout at inference
        dropout_modules = [m for m in self.modules() if isinstance(m, nn.Dropout)]
        for m in dropout_modules:
            m.train()

        scores = []
        for _ in range(n_samples):
            x_hat = self.forward(x)
            score = ((x - x_hat) ** 2).mean(dim=(1, 2))
            scores.append(score)

        # Restore eval mode
        for m in dropout_modules:
            m.eval()

        scores = torch.stack(scores)
        return scores.mean(dim=0), scores.std(dim=0)


from src.models.anomaly.autoencoder import Autoencoder
from src.models.anomaly.lstm_detector import LSTMDetector
from src.models.anomaly.transformer_detector import TransformerDetector

MODEL_REGISTRY = {
    "autoencoder": Autoencoder,
    "lstm": LSTMDetector,
    "transformer": TransformerDetector,
}


def build_model(config) -> AnomalyDetector:
    """Build an anomaly detection model from config."""
    cls = MODEL_REGISTRY[config.model.name]
    return cls.from_config(config)
