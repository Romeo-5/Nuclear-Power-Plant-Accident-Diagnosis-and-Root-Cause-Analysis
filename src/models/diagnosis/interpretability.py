"""SHAP-based interpretability for accident classification.

Provides root cause analysis by identifying which sensors and time windows
contributed most to each accident diagnosis.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def compute_shap_values(
    model: nn.Module,
    data: torch.Tensor,
    background: torch.Tensor | None = None,
    n_background: int = 50,
    device: str = "cpu",
):
    """Compute SHAP values for accident classification predictions.

    Uses DeepExplainer for neural network models.

    Args:
        model: Trained classifier
        data: (N, window_size, n_features) windows to explain
        background: Background dataset for SHAP. If None, uses first n_background from data
        n_background: Number of background samples
        device: Device for computation

    Returns:
        shap_values: list of arrays, one per class, each (N, window_size, n_features)
    """
    import shap

    model.eval()
    model.to(device)

    if background is None:
        background = data[:n_background]

    background = background.to(device)
    data = data.to(device)

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(data)

    return shap_values


def feature_importance(
    shap_values: list[np.ndarray],
    feature_names: list[str],
    class_idx: int | None = None,
) -> dict[str, float]:
    """Compute per-feature importance from SHAP values.

    Args:
        shap_values: SHAP values per class from compute_shap_values
        feature_names: List of feature/sensor names
        class_idx: If given, importance for that class only. Otherwise aggregated.

    Returns:
        Dict mapping feature name to mean absolute SHAP value
    """
    if class_idx is not None:
        vals = np.abs(shap_values[class_idx])
    else:
        # Aggregate across all classes
        vals = np.mean([np.abs(sv) for sv in shap_values], axis=0)

    # Average across samples and time steps: (N, window_size, n_features) -> (n_features,)
    importance = vals.mean(axis=(0, 1))

    return {
        name: float(imp)
        for name, imp in sorted(
            zip(feature_names, importance), key=lambda x: -x[1]
        )
    }


def temporal_importance(
    shap_values: list[np.ndarray],
    class_idx: int,
) -> np.ndarray:
    """Compute per-timestep importance for a given accident class.

    Returns:
        (window_size,) array of mean absolute SHAP values per timestep
    """
    vals = np.abs(shap_values[class_idx])
    # Average across samples and features: (N, window_size, n_features) -> (window_size,)
    return vals.mean(axis=(0, 2))


def gradient_based_attribution(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Simple gradient-based feature attribution (Input x Gradient).

    Faster alternative to SHAP for quick analysis.

    Args:
        model: Trained classifier
        x: (batch, window_size, n_features) input
        target_class: Class to explain. If None, uses predicted class.
        device: Device

    Returns:
        attributions: (batch, window_size, n_features) importance scores
    """
    model.eval()
    model.to(device)
    x = x.to(device).requires_grad_(True)

    logits = model(x)

    if target_class is None:
        target_class_per_sample = logits.argmax(dim=1)
        # Gather the logit for each sample's predicted class
        target_logits = logits.gather(1, target_class_per_sample.unsqueeze(1)).squeeze(1)
    else:
        target_logits = logits[:, target_class]

    target_logits.sum().backward()

    # Input x Gradient
    attributions = (x * x.grad).detach().cpu()
    return attributions
