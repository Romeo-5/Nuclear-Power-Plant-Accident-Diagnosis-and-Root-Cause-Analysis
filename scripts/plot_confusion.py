"""Plot the row-normalized confusion matrix for the stage-2 classifier.

Paper target: Figure 3 in `paper/sections/05_results.tex` (sec:results-clf).

Output: `paper/figures/fig3_confusion.pdf`

Usage:
    python scripts/plot_confusion.py \\
        --config configs/lstm_classifier.yaml \\
        --checkpoint checkpoints/lstm_classifier_best.pt \\
        --output paper/figures/fig3_confusion.pdf
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def load_accident_map() -> dict[int, str]:
    spec = importlib.util.spec_from_file_location(
        "preprocess", Path("data/scripts/preprocess.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return {v: k for k, v in module.ACCIDENT_TYPES.items()}


def load_checkpoint(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        state = blob["model_state_dict"]
    elif isinstance(blob, dict) and "state_dict" in blob:
        state = blob["state_dict"]
    else:
        state = blob
    model.load_state_dict(state)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="paper/figures/fig3_confusion.pdf")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    from src.data.dataset import NPPADWindowDataset
    from src.evaluation.classification_metrics import evaluate_classifier
    from src.models.diagnosis import build_classifier
    from src.utils.config import Config

    config = Config.from_yaml(args.config)
    processed_dir = Path(config.data.processed_dir)
    test_ds = NPPADWindowDataset(processed_dir / "test.pt")
    test_ds.labels = test_ds.accident_types

    model = build_classifier(config)
    model = load_checkpoint(model, Path(args.checkpoint))

    loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    cls_names_map = load_accident_map()
    n_classes = config.model.n_classes
    class_names = [cls_names_map.get(i, f"cls{i}") for i in range(n_classes)]

    metrics = evaluate_classifier(model, loader, class_names=class_names, device=args.device)
    cm = metrics["confusion_matrix"].astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    present_labels = sorted(set(metrics["labels"].tolist()) | set(metrics["predictions"].tolist()))
    tick_labels = [class_names[i] if i < len(class_names) else str(i) for i in present_labels]

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Stage-2 LSTM classifier — row-normalized confusion matrix")

    # Annotate cells with > 0.05 mass.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            v = cm_norm[i, j]
            if v >= 0.05:
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if v > 0.5 else "black",
                )

    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Row-normalized fraction")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"[fig3] wrote {out}")


if __name__ == "__main__":
    main()
