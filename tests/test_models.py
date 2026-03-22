"""Tests for anomaly detection models."""

import pytest
import torch

from src.models.anomaly.autoencoder import Autoencoder
from src.models.anomaly.lstm_detector import LSTMDetector
from src.models.anomaly.transformer_detector import TransformerDetector


BATCH_SIZE = 4
WINDOW_SIZE = 30
N_FEATURES = 97


@pytest.fixture
def sample_input():
    return torch.randn(BATCH_SIZE, WINDOW_SIZE, N_FEATURES)


@pytest.fixture
def sample_batch(sample_input):
    labels = torch.zeros(BATCH_SIZE, dtype=torch.long)
    return sample_input, labels


class TestAutoencoder:
    def test_forward_shape(self, sample_input):
        model = Autoencoder(n_features=N_FEATURES)
        output = model(sample_input)
        assert output.shape == sample_input.shape

    def test_anomaly_score_shape(self, sample_input):
        model = Autoencoder(n_features=N_FEATURES)
        model.eval()
        scores = model.anomaly_score(sample_input)
        assert scores.shape == (BATCH_SIZE,)

    def test_training_step(self, sample_batch):
        model = Autoencoder(n_features=N_FEATURES)
        loss = model.training_step(sample_batch)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_custom_dims(self, sample_input):
        model = Autoencoder(
            n_features=N_FEATURES,
            hidden_dims=[48, 24],
            latent_dim=8,
        )
        output = model(sample_input)
        assert output.shape == sample_input.shape


class TestLSTMDetector:
    def test_forward_shape(self, sample_input):
        model = LSTMDetector(n_features=N_FEATURES)
        output = model(sample_input)
        assert output.shape == sample_input.shape

    def test_anomaly_score_shape(self, sample_input):
        model = LSTMDetector(n_features=N_FEATURES)
        model.eval()
        scores = model.anomaly_score(sample_input)
        assert scores.shape == (BATCH_SIZE,)

    def test_training_step(self, sample_batch):
        model = LSTMDetector(n_features=N_FEATURES)
        loss = model.training_step(sample_batch)
        assert loss.ndim == 0
        assert loss.item() > 0


class TestTransformerDetector:
    def test_forward_shape(self, sample_input):
        model = TransformerDetector(n_features=N_FEATURES)
        output = model(sample_input)
        assert output.shape == sample_input.shape

    def test_anomaly_score_shape(self, sample_input):
        model = TransformerDetector(n_features=N_FEATURES)
        model.eval()
        scores = model.anomaly_score(sample_input)
        assert scores.shape == (BATCH_SIZE,)

    def test_training_step(self, sample_batch):
        model = TransformerDetector(n_features=N_FEATURES)
        loss = model.training_step(sample_batch)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_masked_forward(self, sample_input):
        model = TransformerDetector(n_features=N_FEATURES)
        mask = torch.rand(BATCH_SIZE, WINDOW_SIZE) < 0.15
        output = model(sample_input, mask_positions=mask)
        assert output.shape == sample_input.shape


class TestMCDropout:
    def test_uncertainty_produces_variance(self, sample_input):
        model = Autoencoder(n_features=N_FEATURES, dropout=0.5)
        model.eval()
        mean_scores, std_scores = model.anomaly_score_with_uncertainty(
            sample_input, n_samples=10
        )
        assert mean_scores.shape == (BATCH_SIZE,)
        assert std_scores.shape == (BATCH_SIZE,)
        # With dropout=0.5, there should be some variance
        assert std_scores.sum() > 0


class TestSaveLoad:
    def test_save_and_load(self, sample_input, tmp_path):
        model = Autoencoder(n_features=N_FEATURES)
        model.eval()
        original_scores = model.anomaly_score(sample_input)

        # Save
        torch.save(model.state_dict(), tmp_path / "model.pt")

        # Load into new model
        model2 = Autoencoder(n_features=N_FEATURES)
        model2.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
        model2.eval()
        loaded_scores = model2.anomaly_score(sample_input)

        assert torch.allclose(original_scores, loaded_scores)
