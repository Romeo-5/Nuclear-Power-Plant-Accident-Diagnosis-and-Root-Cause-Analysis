"""Anomaly detection evaluation metrics."""

import numpy as np
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from src.models.anomaly import AnomalyDetector


def compute_anomaly_scores(
    model: AnomalyDetector,
    dataloader: DataLoader,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute anomaly scores for all windows in the dataloader.

    Returns:
        scores: (N,) anomaly scores
        labels: (N,) binary labels (0=normal, 1=anomaly)
        accident_types: (N,) accident type indices
    """
    model.eval()
    device = torch.device(device)
    model.to(device)

    all_scores = []
    all_labels = []
    all_types = []

    with torch.no_grad():
        for i, (windows, labels) in enumerate(dataloader):
            windows = windows.to(device)
            scores = model.anomaly_score(windows)
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.numpy())
            all_types.append(dataloader.dataset.accident_types[
                i * dataloader.batch_size : (i + 1) * dataloader.batch_size
            ].numpy())

    return (
        np.concatenate(all_scores),
        np.concatenate(all_labels),
        np.concatenate(all_types),
    )


def optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Find the threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # Avoid division by zero
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0,
    )
    best_idx = np.argmax(f1_scores)
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    return float(thresholds[-1]) if len(thresholds) > 0 else 0.0


def evaluate(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute evaluation metrics at a given threshold."""
    predictions = (scores >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }

    # ROC and PR AUC (need more than one class present)
    if len(np.unique(labels)) > 1:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
        metrics["auprc"] = float(average_precision_score(labels, scores))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    return metrics


def per_accident_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    accident_types: np.ndarray,
    threshold: float,
) -> dict[int, dict[str, float]]:
    """Compute detection metrics broken down by accident type."""
    results = {}
    predictions = (scores >= threshold).astype(int)

    for atype in np.unique(accident_types):
        mask = accident_types == atype
        if mask.sum() == 0:
            continue

        type_labels = labels[mask]
        type_preds = predictions[mask]
        type_scores = scores[mask]

        result = {
            "n_samples": int(mask.sum()),
            "n_anomalies": int(type_labels.sum()),
            "detection_rate": float(type_preds[type_labels == 1].mean())
            if type_labels.sum() > 0
            else float("nan"),
            "mean_score": float(type_scores.mean()),
        }
        results[int(atype)] = result

    return results


def get_roc_curve(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve data for plotting."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return fpr, tpr, thresholds


def get_pr_curve(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall curve data for plotting."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    return precision, recall, thresholds
