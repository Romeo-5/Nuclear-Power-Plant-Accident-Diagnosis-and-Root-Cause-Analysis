"""Trajectory comparison plot: physics-informed vs. data-only vs. ground truth.

Paper target: Figure 5 in `paper/sections/05_results.tex` (sec:results-pinn).

Output: `paper/figures/fig5_trajectory_loca.pdf`

Loads both digital-twin checkpoints, picks one representative test window for
the requested accident class, runs both models, and overlays predictions
against ground truth for a small set of physically meaningful channels.

Usage:
    python scripts/plot_trajectory.py \\
        --config-physics configs/digital_twin.yaml \\
        --config-baseline configs/digital_twin_no_physics.yaml \\
        --checkpoint-physics checkpoints/digital_twin_best.pt \\
        --checkpoint-baseline checkpoints/digital_twin_no_physics_best.pt \\
        --class LOCA \\
        --output paper/figures/fig5_trajectory_loca.pdf
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path

import torch


CHANNELS_TO_PLOT = ["P", "TAVG", "WRCA", "PWR"]


def load_accident_map() -> dict[str, int]:
    spec = importlib.util.spec_from_file_location(
        "preprocess", Path("data/scripts/preprocess.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ACCIDENT_TYPES


def load_feature_names() -> list[str]:
    spec = importlib.util.spec_from_file_location(
        "preprocess", Path("data/scripts/preprocess.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return list(module.FEATURE_COLUMNS)


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


def load_scaler(processed_dir: Path) -> tuple:
    """Try a few likely names for the persisted scaler."""
    for name in ("scaler.pkl", "scaler.pt"):
        p = processed_dir / name
        if p.exists():
            if p.suffix == ".pkl":
                with open(p, "rb") as fh:
                    sc = pickle.load(fh)
                return sc.mean_, sc.scale_
            else:
                blob = torch.load(p, weights_only=False)
                return blob["mean"], blob["std"]
    raise FileNotFoundError(
        f"No scaler.pkl/scaler.pt under {processed_dir}; "
        "trajectory plot needs the scaler to invert normalization."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-physics", required=True)
    ap.add_argument("--config-baseline", required=True)
    ap.add_argument("--checkpoint-physics", required=True)
    ap.add_argument("--checkpoint-baseline", required=True)
    ap.add_argument("--class", dest="cls_name", default="LOCA")
    ap.add_argument("--sample-idx", type=int, default=0,
                    help="Which test window of the chosen class to plot.")
    ap.add_argument("--output", default="paper/figures/fig5_trajectory_loca.pdf")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    import numpy as np

    from src.data.dataset_twin import DigitalTwinDataset
    from src.models.digital_twin import build_surrogate
    from src.utils.config import Config

    cfg_p = Config.from_yaml(args.config_physics)
    cfg_b = Config.from_yaml(args.config_baseline)
    processed_dir = Path(cfg_p.data.processed_dir)

    horizon = getattr(cfg_p.model, "prediction_horizon", 10)
    test_ds = DigitalTwinDataset(processed_dir / "test.pt", prediction_horizon=horizon)

    accident_map = load_accident_map()
    if args.cls_name not in accident_map:
        raise SystemExit(f"Unknown class {args.cls_name}.")
    cls_idx = accident_map[args.cls_name]
    mask = (test_ds.accident_types == cls_idx).nonzero(as_tuple=True)[0]
    if len(mask) == 0:
        raise SystemExit(f"No test windows for class {args.cls_name}.")
    chosen = int(mask[min(args.sample_idx, len(mask) - 1)])

    context, target = test_ds[chosen]
    context = context.unsqueeze(0)
    target = target.unsqueeze(0)

    # Build models, load checkpoints
    m_p = build_surrogate(cfg_p)
    m_b = build_surrogate(cfg_b)
    load_checkpoint(m_p, Path(args.checkpoint_physics))
    load_checkpoint(m_b, Path(args.checkpoint_baseline))
    m_p.eval(); m_b.eval()

    with torch.no_grad():
        pred_p = m_p(context).numpy()[0]
        pred_b = m_b(context).numpy()[0]

    # Inverse transform via persisted scaler
    mean, std = load_scaler(processed_dir)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)

    target_phys = target.numpy()[0] * std + mean
    pred_p_phys = pred_p * std + mean
    pred_b_phys = pred_b * std + mean
    context_phys = context.numpy()[0] * std + mean

    feature_names = load_feature_names()
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    chans = [c for c in CHANNELS_TO_PLOT if c in name_to_idx]

    n = len(chans)
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.2 * n), sharex=True, constrained_layout=True)
    if n == 1:
        axes = [axes]

    ctx_t = np.arange(context_phys.shape[0])
    fut_t = np.arange(context_phys.shape[0], context_phys.shape[0] + horizon)

    for ax, ch in zip(axes, chans):
        j = name_to_idx[ch]
        ax.plot(ctx_t, context_phys[:, j], color="k", lw=1.0, label="Observed (context)")
        ax.plot(fut_t, target_phys[:, j], color="k", lw=1.5, label="Ground truth")
        ax.plot(fut_t, pred_p_phys[:, j], color="C0", lw=1.5, label="Physics-informed")
        ax.plot(fut_t, pred_b_phys[:, j], color="C3", lw=1.5, ls="--", label="Data-only")
        ax.set_ylabel(ch)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=8, frameon=False)
    axes[-1].set_xlabel("Timestep")
    fig.suptitle(f"{args.cls_name} trajectory prediction (sample idx={args.sample_idx})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"[fig5] wrote {out}")


if __name__ == "__main__":
    main()
