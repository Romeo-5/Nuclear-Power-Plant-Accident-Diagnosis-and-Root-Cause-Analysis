"""GRU encoder-decoder surrogate model for reactor dynamics prediction.

Predicts future sensor states from a context window, with optional
physics-informed loss constraints via an attached PhysicsInformedLoss module.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.models.digital_twin.loss import PhysicsInformedLoss
from src.utils.config import Config


class ReactorSurrogate(nn.Module):
    """GRU encoder-decoder with residual connection for reactor state prediction.

    Architecture:
        Encoder: GRU processes context window → hidden state
        Decoder: GRU autoregressively predicts future states
        Residual: predictions = last_observed + predicted_delta
    """

    def __init__(
        self,
        n_features: int = 96,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        prediction_horizon: int = 10,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon

        # Encoder GRU
        self.encoder = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder GRU
        self.decoder = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output projection: hidden → feature delta
        self.output_proj = nn.Linear(hidden_dim, n_features)

        # Step counter for physics loss warmup
        self._step = 0

        # Loss function (set via attach_loss or from_config)
        self.loss_fn: PhysicsInformedLoss | None = None

    def forward(self, context: Tensor) -> Tensor:
        """Predict future sensor states from context window.

        Args:
            context: (batch, context_len, n_features) — observed sensor data

        Returns:
            predictions: (batch, prediction_horizon, n_features)
        """
        batch_size = context.shape[0]

        # Encode context
        _, hidden = self.encoder(context)  # hidden: (layers, batch, hidden_dim)

        # Last observed state for residual connection
        last_state = context[:, -1:, :]  # (batch, 1, n_features)

        # Autoregressive decoding
        decoder_input = last_state  # Start with last observed
        outputs = []

        for _ in range(self.prediction_horizon):
            decoder_out, hidden = self.decoder(decoder_input, hidden)
            delta = self.output_proj(decoder_out)  # (batch, 1, n_features)
            prediction = decoder_input + delta  # Residual connection
            outputs.append(prediction)
            decoder_input = prediction  # Feed prediction as next input

        return torch.cat(outputs, dim=1)  # (batch, horizon, n_features)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        """Compute loss for a training batch.

        Args:
            batch: (context, target) tensors

        Returns:
            Scalar loss tensor
        """
        context, target = batch
        predictions = self.forward(context)

        self._step += 1

        if self.loss_fn is not None:
            warmup_frac = min(1.0, self._step / max(self.loss_fn.warmup_steps, 1))
            loss, _ = self.loss_fn(predictions, target, warmup_frac)
            return loss

        # Fallback: plain MSE if no physics loss attached
        return nn.functional.mse_loss(predictions, target)

    def attach_loss(self, loss_fn: PhysicsInformedLoss) -> None:
        """Attach a physics-informed loss function."""
        self.loss_fn = loss_fn

    @classmethod
    def from_config(cls, config: Config) -> "ReactorSurrogate":
        """Build surrogate model from config, optionally with physics loss."""
        model = cls(
            n_features=config.data.n_features,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            prediction_horizon=config.model.prediction_horizon,
        )

        # Try to load scaler and attach physics loss
        scaler_path = Path(config.data.processed_dir) / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            loss_fn = PhysicsInformedLoss(
                scaler_mean=scaler.mean_,
                scaler_std=scaler.scale_,
                lambda_data=config.physics.lambda_data,
                lambda_physics=config.physics.lambda_physics,
                lambda_conservation=config.physics.lambda_conservation,
                lambda_bounds=config.physics.lambda_bounds,
                warmup_steps=config.physics.warmup_steps,
                dt=config.physics.dt,
            )
            model.attach_loss(loss_fn)

        return model
