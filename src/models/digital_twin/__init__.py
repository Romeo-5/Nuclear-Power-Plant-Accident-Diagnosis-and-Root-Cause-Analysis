"""Digital twin module: physics-informed surrogate model for reactor dynamics."""

from src.models.digital_twin.surrogate import ReactorSurrogate

MODEL_REGISTRY = {
    "digital_twin": ReactorSurrogate,
}


def build_surrogate(config) -> ReactorSurrogate:
    """Build a digital twin surrogate model from config."""
    return ReactorSurrogate.from_config(config)
