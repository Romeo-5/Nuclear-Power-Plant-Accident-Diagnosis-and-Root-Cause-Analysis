"""Transformer-based anomaly detector using masked reconstruction."""

import math

import torch
import torch.nn as nn
from torch import Tensor

from src.models.anomaly import AnomalyDetector


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerDetector(AnomalyDetector):
    """Masked reconstruction transformer for anomaly detection.

    During training, randomly masks a fraction of timesteps and reconstructs them.
    At inference, anomaly score is the reconstruction error across all positions.
    """

    def __init__(
        self,
        n_features: int = 97,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.n_features = n_features

        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.output_projection = nn.Linear(d_model, n_features)

    def forward(self, x: Tensor, mask_positions: Tensor | None = None) -> Tensor:
        """Reconstruct input, optionally with masked positions.

        x: (batch, window_size, n_features)
        mask_positions: (batch, window_size) boolean mask, True = masked
        """
        x_input = x.clone()
        if mask_positions is not None:
            x_input[mask_positions] = 0.0

        h = self.input_projection(x_input)
        h = self.pos_encoding(h)
        h = self.dropout(h)
        h = self.transformer_encoder(h)
        return self.output_projection(h)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        """Masked reconstruction loss."""
        x, _ = batch
        batch_size, window_size, _ = x.shape

        # Random mask
        mask = torch.rand(batch_size, window_size, device=x.device) < self.mask_ratio
        # Ensure at least one position is masked per sample
        if not mask.any():
            mask[:, 0] = True

        x_hat = self.forward(x, mask_positions=mask)

        # Loss only on masked positions
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        loss = nn.functional.mse_loss(x_hat[mask_expanded], x[mask_expanded])
        return loss

    def anomaly_score(self, x: Tensor) -> Tensor:
        """Full reconstruction error (no masking at inference)."""
        with torch.no_grad():
            x_hat = self.forward(x)
            return ((x - x_hat) ** 2).mean(dim=(1, 2))

    @classmethod
    def from_config(cls, config) -> "TransformerDetector":
        return cls(
            n_features=config.data.n_features,
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            num_layers=config.model.num_layers,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            mask_ratio=config.model.mask_ratio,
        )
