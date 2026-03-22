"""Tests for accident classification models (Module 2)."""

import pytest
import torch

from src.models.diagnosis.classifier import (
    AccidentClassifier,
    LSTMClassifier,
    TransformerClassifier,
)

BATCH_SIZE = 4
WINDOW_SIZE = 30
N_FEATURES = 96
N_CLASSES = 18


@pytest.fixture
def sample_input():
    return torch.randn(BATCH_SIZE, WINDOW_SIZE, N_FEATURES)


@pytest.fixture
def sample_batch(sample_input):
    labels = torch.randint(0, N_CLASSES, (BATCH_SIZE,))
    return sample_input, labels


class TestAccidentClassifier:
    def test_forward_shape(self, sample_input):
        model = AccidentClassifier(n_features=N_FEATURES, n_classes=N_CLASSES)
        logits = model(sample_input)
        assert logits.shape == (BATCH_SIZE, N_CLASSES)

    def test_training_step(self, sample_batch):
        model = AccidentClassifier(n_features=N_FEATURES, n_classes=N_CLASSES)
        loss = model.training_step(sample_batch)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_predict(self, sample_input):
        model = AccidentClassifier(n_features=N_FEATURES, n_classes=N_CLASSES)
        model.eval()
        preds = model.predict(sample_input)
        assert preds.shape == (BATCH_SIZE,)
        assert (preds >= 0).all() and (preds < N_CLASSES).all()

    def test_predict_proba(self, sample_input):
        model = AccidentClassifier(n_features=N_FEATURES, n_classes=N_CLASSES)
        model.eval()
        probs = model.predict_proba(sample_input)
        assert probs.shape == (BATCH_SIZE, N_CLASSES)
        assert torch.allclose(probs.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-5)


class TestLSTMClassifier:
    def test_forward_shape(self, sample_input):
        model = LSTMClassifier(n_features=N_FEATURES, n_classes=N_CLASSES)
        logits = model(sample_input)
        assert logits.shape == (BATCH_SIZE, N_CLASSES)

    def test_training_step(self, sample_batch):
        model = LSTMClassifier(n_features=N_FEATURES, n_classes=N_CLASSES)
        loss = model.training_step(sample_batch)
        assert loss.ndim == 0
        assert loss.item() > 0


class TestTransformerClassifier:
    def test_forward_shape(self, sample_input):
        model = TransformerClassifier(n_features=N_FEATURES, n_classes=N_CLASSES)
        logits = model(sample_input)
        assert logits.shape == (BATCH_SIZE, N_CLASSES)

    def test_training_step(self, sample_batch):
        model = TransformerClassifier(n_features=N_FEATURES, n_classes=N_CLASSES)
        loss = model.training_step(sample_batch)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_cls_token(self, sample_input):
        model = TransformerClassifier(n_features=N_FEATURES, n_classes=N_CLASSES)
        model.eval()
        # Should work regardless of sequence length
        short_input = torch.randn(2, 10, N_FEATURES)
        logits = model(short_input)
        assert logits.shape == (2, N_CLASSES)
