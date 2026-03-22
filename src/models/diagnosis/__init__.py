"""Module 2: Accident classification and root cause analysis."""

from src.models.diagnosis.classifier import AccidentClassifier, LSTMClassifier, TransformerClassifier

MODEL_REGISTRY = {
    "cnn_classifier": AccidentClassifier,
    "lstm_classifier": LSTMClassifier,
    "transformer_classifier": TransformerClassifier,
}


def build_classifier(config):
    """Build a classification model from config."""
    cls = MODEL_REGISTRY[config.model.name]
    return cls.from_config(config)
