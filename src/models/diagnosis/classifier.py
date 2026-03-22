"""Multi-class accident classifiers for root cause diagnosis."""

import math

import torch
import torch.nn as nn
from torch import Tensor


class AccidentClassifier(nn.Module):
    """1D-CNN classifier for accident type identification.

    Uses temporal convolutions to extract patterns from sensor windows,
    followed by global average pooling and a classification head.
    """

    def __init__(
        self,
        n_features: int = 96,
        n_classes: int = 18,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]

        self.n_classes = n_classes

        # Build conv blocks: each is Conv1d + BatchNorm + ReLU + Dropout
        conv_layers = []
        in_ch = n_features
        for out_ch in channels:
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Classify accident type.

        x: (batch, window_size, n_features)
        Returns: (batch, n_classes) logits
        """
        # Conv1d expects (batch, channels, length)
        h = x.transpose(1, 2)
        h = self.conv(h)
        # Global average pooling over time
        h = h.mean(dim=2)
        return self.classifier(h)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        x, labels = batch
        logits = self.forward(x)
        return nn.functional.cross_entropy(logits, labels)

    def predict(self, x: Tensor) -> Tensor:
        """Return predicted class indices."""
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)

    def predict_proba(self, x: Tensor) -> Tensor:
        """Return class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    @classmethod
    def from_config(cls, config) -> "AccidentClassifier":
        return cls(
            n_features=config.data.n_features,
            n_classes=config.model.n_classes,
            channels=config.model.channels,
            dropout=config.model.dropout,
        )


class LSTMClassifier(nn.Module):
    """LSTM-based classifier for accident type identification."""

    def __init__(
        self,
        n_features: int = 96,
        n_classes: int = 18,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.n_classes = n_classes

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, window_size, n_features) -> (batch, n_classes) logits."""
        output, _ = self.lstm(x)
        # Use the last hidden state
        h = self.dropout(output[:, -1, :])
        return self.classifier(h)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        x, labels = batch
        logits = self.forward(x)
        return nn.functional.cross_entropy(logits, labels)

    def predict(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)

    def predict_proba(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)

    @classmethod
    def from_config(cls, config) -> "LSTMClassifier":
        return cls(
            n_features=config.data.n_features,
            n_classes=config.model.n_classes,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
        )


class TransformerClassifier(nn.Module):
    """Transformer encoder classifier for accident type identification.

    Uses a [CLS] token approach: prepends a learnable class token,
    encodes with transformer, and classifies from the class token output.
    """

    def __init__(
        self,
        n_features: int = 96,
        n_classes: int = 18,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.d_model = d_model

        self.input_projection = nn.Linear(n_features, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        max_len = 500
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, window_size, n_features) -> (batch, n_classes) logits."""
        batch_size = x.size(0)
        h = self.input_projection(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls_tokens, h], dim=1)

        # Add positional encoding
        h = h + self.pe[:, :h.size(1)]

        h = self.transformer(h)

        # Classify from CLS token
        cls_output = h[:, 0]
        return self.classifier(cls_output)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        x, labels = batch
        logits = self.forward(x)
        return nn.functional.cross_entropy(logits, labels)

    def predict(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self.forward(x).argmax(dim=1)

    def predict_proba(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)

    @classmethod
    def from_config(cls, config) -> "TransformerClassifier":
        return cls(
            n_features=config.data.n_features,
            n_classes=config.model.n_classes,
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            num_layers=config.model.num_layers,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
        )
