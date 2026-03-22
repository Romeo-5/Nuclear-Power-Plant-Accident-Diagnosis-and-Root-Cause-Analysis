"""Multi-class classification evaluation metrics for accident diagnosis."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader


def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: list[str] | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """Evaluate a multi-class classifier.

    Args:
        model: Trained classifier
        dataloader: DataLoader with accident_type labels
        class_names: Optional list of class names for the report
        device: Device for inference

    Returns:
        Dict with accuracy, macro/weighted F1, per-class report, confusion matrix
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for windows, labels in dataloader:
            windows = windows.to(device)
            logits = model(windows)
            preds = logits.argmax(dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    # Only evaluate on classes present in labels
    present_classes = sorted(set(labels.tolist()) | set(preds.tolist()))

    target_names = None
    if class_names:
        target_names = [class_names[i] for i in present_classes if i < len(class_names)]

    report = classification_report(
        labels, preds,
        labels=present_classes,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(labels, preds, labels=present_classes)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": preds,
        "labels": labels,
    }
