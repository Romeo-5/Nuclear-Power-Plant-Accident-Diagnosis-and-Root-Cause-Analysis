"""Configuration system using YAML files and dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    window_size: int = 30
    stride: int = 10
    eval_stride: int = 1
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    n_features: int = 97


@dataclass
class ModelConfig:
    name: str = "autoencoder"
    # Autoencoder
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    latent_dim: int = 16
    # LSTM
    hidden_dim: int = 128
    num_layers: int = 2
    # Transformer
    d_model: int = 128
    nhead: int = 8
    dim_feedforward: int = 256
    mask_ratio: float = 0.15
    # CNN Classifier
    channels: list[int] = field(default_factory=lambda: [64, 128, 256])
    # Classification
    n_classes: int = 18
    # Shared
    dropout: float = 0.2


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 10
    device: str = "auto"
    num_workers: int = 4
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load config from YAML file, merging with base config if specified."""
        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        # Handle base config inheritance
        base_name = raw.pop("_base_", None)
        if base_name:
            base_path = path.parent / base_name
            with open(base_path) as f:
                base_raw = yaml.safe_load(f)
            base_raw = _deep_merge(base_raw, raw)
            raw = base_raw

        return cls(
            data=_from_dict(DataConfig, raw.get("data", {})),
            model=_from_dict(ModelConfig, raw.get("model", {})),
            train=_from_dict(TrainConfig, raw.get("train", {})),
        )


def _from_dict(cls: type, d: dict[str, Any]) -> Any:
    """Create a dataclass instance from a dict, ignoring unknown keys."""
    valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in d.items() if k in valid_keys}
    return cls(**filtered)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
