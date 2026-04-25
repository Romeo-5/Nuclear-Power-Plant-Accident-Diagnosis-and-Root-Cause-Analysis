"""Plot per-class F1 vs. number of training windows for the stage-2 classifier.

Paper target: Figure 2 in `paper/sections/05_results.tex` (sec:results-clf).

Output: `paper/figures/fig2_f1_vs_frequency.pdf`

Usage:
    python scripts/plot_per_class_f1.py \\
        --config configs/lstm_classifier.yaml \\
        --checkpoint checkpoints/lstm_classifier_best.pt \\
        --output paper/figures/fig2_f1_vs_frequency.pdf
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
    ap.add_argument("--output", default="paper/figures/fig2_f1_vs_frequency.pdf")
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

    train_ds = NPPADWindowDataset(processed_dir / "train.pt")
    test_ds = NPPADWindowDataset(processed_dir / "test.pt")
    test_ds.labels = test_ds.accident_types

    # Class -> n_train
    train_counts = {}
    for cls in train_ds.accident_types.unique().tolist():
        train_counts[int(cls)] = int((train_ds.accident_types == cls).sum())

    model = build_classifier(config)
    model = load_checkpoint(model, Path(args.checkpoint))

    loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    cls_names_map = load_accident_map()
    n_classes = config.model.n_classes
    class_names = [cls_names_map.get(i, f"cls{i}") for i in range(n_classes)]

    metrics = evaluate_classifier(model, loader, class_names=class_names, device=args.device)
    report = metrics["classification_report"]

    # Build (n_train, f1, label) tuples per class.
    points = []
    for cls_idx in range(n_classes):
        name = class_names[cls_idx]
        n_train = train_counts.get(cls_idx, 0)
        if name not in report:
            continue
        f1 = report[name]["f1-score"]
        if n_train == 0:
            continue
        points.append((n_train, f1, name))

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    ax.scatter(xs, ys, s=40)
    for x, y, name in points:
        ax.annotate(name, (x, y), xytext=(4, 2), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Training windows per class (log scale)")
    ax.set_ylabel("Test F1")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title("Per-class F1 vs. training-set frequency")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"[fig2] wrote {out}")


if __name__ == "__main__":
    main()
