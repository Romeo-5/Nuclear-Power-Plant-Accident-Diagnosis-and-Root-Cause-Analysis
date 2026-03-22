"""Autoencoder-based anomaly detector for time-series sensor data."""

import torch.nn as nn
from torch import Tensor

from src.models.anomaly import AnomalyDetector


class Autoencoder(AnomalyDetector):
    """Per-timestep autoencoder with shared weights across the time dimension.

    Reconstructs each timestep independently using the same encoder/decoder.
    Anomaly score is the mean reconstruction error per window.
    """

    def __init__(
        self,
        n_features: int = 97,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        # Build encoder
        encoder_layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (mirror of encoder)
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, n_features))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: Tensor) -> Tensor:
        """Reconstruct input. x: (batch, window_size, n_features)."""
        batch_size, window_size, n_features = x.shape
        # Flatten time into batch for per-timestep processing
        x_flat = x.reshape(batch_size * window_size, n_features)
        z = self.encoder(x_flat)
        x_hat = self.decoder(z)
        return x_hat.reshape(batch_size, window_size, n_features)

    @classmethod
    def from_config(cls, config) -> "Autoencoder":
        return cls(
            n_features=config.data.n_features,
            hidden_dims=config.model.hidden_dims,
            latent_dim=config.model.latent_dim,
            dropout=config.model.dropout,
        )
