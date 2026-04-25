"""Histograms of per-window physics residuals: physics-informed vs. data-only.

Paper target: Figure 6 in `paper/sections/05_results.tex` (sec:results-pinn).

For each test window, run both surrogates, compute the point-kinetics residual
and the energy-conservation violation on each predicted trajectory, then
overlay the two distributions on a log-y axis.

Output: `paper/figures/fig6_residual_histograms.pdf`
        results/digital_twin/per_window_residuals.csv

Usage:
    python scripts/plot_residual_histograms.py \\
        --config-physics configs/digital_twin.yaml \\
        --config-baseline configs/digital_twin_no_physics.yaml \\
        --checkpoint-physics checkpoints/digital_twin_best.pt \\
        --checkpoint-baseline checkpoints/digital_twin_no_physics_best.pt
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


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


def load_scaler(processed_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    for name in ("scaler.pkl", "scaler.pt"):
        p = processed_dir / name
        if p.exists():
            if p.suffix == ".pkl":
                with open(p, "rb") as fh:
                    sc = pickle.load(fh)
                return np.asarray(sc.mean_, dtype=np.float32), np.asarray(sc.scale_, dtype=np.float32)
            else:
                blob = torch.load(p, weights_only=False)
                return np.asarray(blob["mean"], dtype=np.float32), np.asarray(blob["std"], dtype=np.float32)
    raise FileNotFoundError(f"No scaler under {processed_dir}.")


def per_window_residuals(model, dataloader, scaler_mean, scaler_std, device):
    """Run model on every batch; return per-window point-kinetics & conservation residuals."""
    from src.models.digital_twin.loss import PhysicsInformedLoss
    from src.models.digital_twin.physics import DEFAULT_PARAMS, FEATURE_MAP

    fm = FEATURE_MAP
    params = DEFAULT_PARAMS
    pk_list = []
    cons_list = []

    model.eval()
    model.to(device)
    mean_t = torch.tensor(scaler_mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(scaler_std, dtype=torch.float32, device=device)

    from src.models.digital_twin.physics import point_kinetics_rhs

    with torch.no_grad():
        for context, target in dataloader:
            context = context.to(device)
            pred = model(context)
            pred_phys = pred * std_t + mean_t

            n = pred_phys[:, :, fm.PWR]
            T_avg = pred_phys[:, :, fm.TAVG]
            ppm = pred_phys[:, :, fm.PPM]
            dn_dt_num = (n[:, 1:] - n[:, :-1]) / 1.0
            dn_dt_phys = point_kinetics_rhs(n[:, :-1], T_avg[:, :-1], ppm[:, :-1], params)
            n_scale = n[:, :-1].abs().mean().clamp(min=1.0)
            pk = ((dn_dt_num - dn_dt_phys) / n_scale).pow(2).mean(dim=(1,))

            Q = pred_phys[:, :, fm.QMWT]
            W = pred_phys[:, :, fm.WRCA] + pred_phys[:, :, fm.WRCB]
            T_hot = 0.5 * (pred_phys[:, :, fm.THA] + pred_phys[:, :, fm.THB])
            T_cold = 0.5 * (pred_phys[:, :, fm.TCA] + pred_phys[:, :, fm.TCB])
            Q_est = W * params.Cp * (T_hot - T_cold) / 1000.0
            cons = (Q - Q_est).abs().mean(dim=(1,))

            pk_list.append(pk.cpu().numpy())
            cons_list.append(cons.cpu().numpy())

    return np.concatenate(pk_list), np.concatenate(cons_list)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-physics", required=True)
    ap.add_argument("--config-baseline", required=True)
    ap.add_argument("--checkpoint-physics", required=True)
    ap.add_argument("--checkpoint-baseline", required=True)
    ap.add_argument("--output-fig", default="paper/figures/fig6_residual_histograms.pdf")
    ap.add_argument("--output-csv", default="results/digital_twin/per_window_residuals.csv")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    from src.data.dataset_twin import DigitalTwinDataset
    from src.models.digital_twin import build_surrogate
    from src.utils.config import Config

    cfg_p = Config.from_yaml(args.config_physics)
    cfg_b = Config.from_yaml(args.config_baseline)
    processed_dir = Path(cfg_p.data.processed_dir)
    horizon = getattr(cfg_p.model, "prediction_horizon", 10)

    test_ds = DigitalTwinDataset(processed_dir / "test.pt", prediction_horizon=horizon)
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    mean, std = load_scaler(processed_dir)

    m_p = build_surrogate(cfg_p)
    m_b = build_surrogate(cfg_b)
    load_checkpoint(m_p, Path(args.checkpoint_physics))
    load_checkpoint(m_b, Path(args.checkpoint_baseline))

    print("[fig6] running physics-informed model...")
    pk_p, cons_p = per_window_residuals(m_p, loader, mean, std, args.device)
    print("[fig6] running data-only baseline...")
    pk_b, cons_b = per_window_residuals(m_b, loader, mean, std, args.device)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["window_idx", "pk_physics", "pk_baseline", "cons_physics", "cons_baseline"])
        for i, (a, b, c, d) in enumerate(zip(pk_p, pk_b, cons_p, cons_b)):
            w.writerow([i, float(a), float(b), float(c), float(d)])
    print(f"[fig6] wrote per-window residuals -> {out_csv}")

    print(f"[fig6] mean PK residual: physics={pk_p.mean():.3e}  baseline={pk_b.mean():.3e}  "
          f"reduction={1 - pk_p.mean()/max(pk_b.mean(),1e-12):.3%}")
    print(f"[fig6] mean conservation: physics={cons_p.mean():.3e}  baseline={cons_b.mean():.3e}  "
          f"reduction={1 - cons_p.mean()/max(cons_b.mean(),1e-12):.3%}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    bins_pk = np.logspace(np.log10(max(min(pk_p.min(), pk_b.min()), 1e-9)),
                          np.log10(max(pk_p.max(), pk_b.max(), 1e-9)), 60)
    axes[0].hist(pk_b, bins=bins_pk, alpha=0.5, label="Data-only", color="C3")
    axes[0].hist(pk_p, bins=bins_pk, alpha=0.5, label="Physics-informed", color="C0")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("Point-kinetics residual (per window)")
    axes[0].set_ylabel("Windows (log)")
    axes[0].legend(frameon=False)
    axes[0].set_title("Point-kinetics residual")

    bins_c = np.logspace(np.log10(max(min(cons_p.min(), cons_b.min()), 1e-9)),
                         np.log10(max(cons_p.max(), cons_b.max(), 1e-9)), 60)
    axes[1].hist(cons_b, bins=bins_c, alpha=0.5, label="Data-only", color="C3")
    axes[1].hist(cons_p, bins=bins_c, alpha=0.5, label="Physics-informed", color="C0")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("Conservation violation (MW)")
    axes[1].set_title("Energy conservation")

    out_fig = Path(args.output_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig)
    print(f"[fig6] wrote {out_fig}")


if __name__ == "__main__":
    main()
