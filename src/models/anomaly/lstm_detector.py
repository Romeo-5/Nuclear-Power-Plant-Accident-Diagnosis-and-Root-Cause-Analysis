"""LSTM-based anomaly detector using next-step prediction."""

import torch
import torch.nn as nn
from torch import Tensor

from src.models.anomaly import AnomalyDetector


class LSTMDetector(AnomalyDetector):
    """Prediction-based LSTM anomaly detector.

    Predicts the next timestep given the sequence so far.
    High prediction error indicates anomalous behavior.
    Uses teacher forcing during training.
    """

    def __init__(
        self,
        n_features: int = 97,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_dim, n_features)

    def forward(self, x: Tensor) -> Tensor:
        """Predict next timestep for each position.

        x: (batch, window_size, n_features)
        Returns: (batch, window_size, n_features) shifted predictions
        """
        output, _ = self.lstm(x)
        output = self.dropout(output)
        predictions = self.projection(output)
        return predictions

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        """Teacher forcing: input t[0:T-1], predict t[1:T]."""
        x, _ = batch
        # Input is all but last timestep, target is all but first
        x_input = x[:, :-1, :]
        x_target = x[:, 1:, :]
        predictions = self.forward(x_input)
        return nn.functional.mse_loss(predictions, x_target)

    def anomaly_score(self, x: Tensor) -> Tensor:
        """Prediction error as anomaly score."""
        with torch.no_grad():
            x_input = x[:, :-1, :]
            x_target = x[:, 1:, :]
            predictions = self.forward(x_input)
            return ((predictions - x_target) ** 2).mean(dim=(1, 2))

    @classmethod
    def from_config(cls, config) -> "LSTMDetector":
        return cls(
            n_features=config.data.n_features,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
        )
