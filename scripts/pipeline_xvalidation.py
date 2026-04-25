"""End-to-end pipeline cross-validation: classifier + physics-informed surrogate.

For each test window:
  1. Run the stage-2 classifier; record predicted class and correct/incorrect.
  2. Run the stage-3 physics-informed surrogate; compute the physics-consistency
     score kappa = w_pk * pk_residual + w_cons * conservation_residual,
     normalized so the median kappa on correctly-classified validation windows
     is 1.0.
  3. Sweep kappa* on validation; pick the F1-optimal threshold for the binary
     task "flag = misclassification". Report precision/recall on test.

Paper targets: Figure 7, Table 5, Figure 8 in `paper/sections/05_results.tex`
(sec:results-pipeline).

Outputs:
    results/pipeline/kappa_distributions.pdf       -> Figure 7
    results/pipeline/xvalidation_table.tex         -> Table 5 fragment
    results/pipeline/case_studies.pdf              -> Figure 8
    results/pipeline/kappa_stats.json              -> KS statistic + p-value

Usage:
    python scripts/pipeline_xvalidation.py \\
        --classifier-config configs/lstm_classifier.yaml \\
        --classifier-checkpoint checkpoints/lstm_classifier_best.pt \\
        --twin-config configs/digital_twin.yaml \\
        --twin-checkpoint checkpoints/digital_twin_best.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
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


def compute_residuals(twin, contexts, mean_t, std_t):
    """Return (pk_per_window, cons_per_window) numpy arrays."""
    from src.models.digital_twin.physics import DEFAULT_PARAMS, FEATURE_MAP, point_kinetics_rhs
    fm = FEATURE_MAP
    params = DEFAULT_PARAMS
    with torch.no_grad():
        pred = twin(contexts)
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
    return pk.cpu().numpy(), cons.cpu().numpy()


def gather_predictions(classifier, twin, dataset, mean, std, device, horizon):
    """For each window: classifier prediction, true label, pk, cons residuals."""
    classifier.eval(); twin.eval()
    classifier.to(device); twin.to(device)
    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)
    loader = DataLoader(
        list(range(len(dataset))), batch_size=64, shuffle=False
    )
    preds, labels, pks, conses = [], [], [], []
    with torch.no_grad():
        for idx_batch in loader:
            idxs = idx_batch.tolist()
            windows = torch.stack([dataset.windows[i] for i in idxs]).to(device)
            atypes = torch.tensor([int(dataset.accident_types[i]) for i in idxs])
            logits = classifier(windows)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(atypes.numpy())
            ctx = windows[:, : windows.shape[1] - horizon, :]
            pk, cons = compute_residuals(twin, ctx, mean_t, std_t)
            pks.append(pk); conses.append(cons)
    return (np.concatenate(preds), np.concatenate(labels),
            np.concatenate(pks), np.concatenate(conses))


def best_f1_threshold(kappa_val, correct_val):
    """Find kappa* maximizing F1 on validation for the binary task
    'flag = misclassified' (i.e. positive when correct_val == False)."""
    misclass = ~correct_val
    if misclass.sum() == 0:
        return float(np.median(kappa_val)), {"note": "no misclassifications in val"}
    candidates = np.unique(np.quantile(kappa_val, np.linspace(0.5, 0.99, 50)))
    best = (-1.0, None, None, None)
    for thr in candidates:
        flag = kappa_val > thr
        tp = int((flag & misclass).sum())
        fp = int((flag & ~misclass).sum())
        fn = int((~flag & misclass).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        if f1 > best[0]:
            best = (f1, thr, prec, rec)
    return float(best[1]), {"val_f1": best[0], "val_precision": best[2], "val_recall": best[3]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier-config", required=True)
    ap.add_argument("--classifier-checkpoint", required=True)
    ap.add_argument("--twin-config", required=True)
    ap.add_argument("--twin-checkpoint", required=True)
    ap.add_argument("--w-pk", type=float, default=1.0)
    ap.add_argument("--w-cons", type=float, default=1.0)
    ap.add_argument("--output-dir", default="results/pipeline")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp

    from src.data.dataset import NPPADWindowDataset
    from src.models.diagnosis import build_classifier
    from src.models.digital_twin import build_surrogate
    from src.utils.config import Config

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg_clf = Config.from_yaml(args.classifier_config)
    cfg_twin = Config.from_yaml(args.twin_config)
    processed_dir = Path(cfg_clf.data.processed_dir)
    horizon = getattr(cfg_twin.model, "prediction_horizon", 10)

    classifier = build_classifier(cfg_clf)
    load_checkpoint(classifier, Path(args.classifier_checkpoint))
    twin = build_surrogate(cfg_twin)
    load_checkpoint(twin, Path(args.twin_checkpoint))

    val_ds = NPPADWindowDataset(processed_dir / "val.pt")
    test_ds = NPPADWindowDataset(processed_dir / "test.pt")
    mean, std = load_scaler(processed_dir)

    print("[xval] running validation pass...")
    p_v, l_v, pk_v, cons_v = gather_predictions(classifier, twin, val_ds, mean, std, args.device, horizon)
    print("[xval] running test pass...")
    p_t, l_t, pk_t, cons_t = gather_predictions(classifier, twin, test_ds, mean, std, args.device, horizon)

    # Build raw kappa as weighted sum.
    raw_v = args.w_pk * pk_v + args.w_cons * cons_v
    raw_t = args.w_pk * pk_t + args.w_cons * cons_t
    correct_v = (p_v == l_v)
    correct_t = (p_t == l_t)

    median_correct = float(np.median(raw_v[correct_v])) if correct_v.any() else 1.0
    if median_correct == 0:
        median_correct = 1.0
    kappa_v = raw_v / median_correct
    kappa_t = raw_t / median_correct

    thr, val_stats = best_f1_threshold(kappa_v, correct_v)

    # Test-set performance at thr
    flag_t = kappa_t > thr
    misc_t = ~correct_t
    tp = int((flag_t & misc_t).sum())
    fp = int((flag_t & ~misc_t).sum())
    fn = int((~flag_t & misc_t).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    ks = ks_2samp(kappa_t[correct_t], kappa_t[~correct_t]) if (~correct_t).any() else None

    stats = {
        "n_test": int(len(correct_t)),
        "n_misclass_test": int(misc_t.sum()),
        "kappa_threshold": float(thr),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "ks_statistic": float(ks.statistic) if ks else None,
        "ks_pvalue": float(ks.pvalue) if ks else None,
        "validation": val_stats,
    }
    (out / "kappa_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))

    # Figure 7: kappa distributions
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    bins = np.linspace(0, np.quantile(kappa_t, 0.99) * 1.1, 60)
    ax.hist(kappa_t[correct_t], bins=bins, alpha=0.5, label="Correctly classified", color="C0")
    ax.hist(kappa_t[~correct_t], bins=bins, alpha=0.5, label="Misclassified", color="C3")
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"$\\kappa^* = {thr:.2f}$")
    ax.set_xlabel(r"Physics-consistency score $\kappa$")
    ax.set_ylabel("Test windows")
    ax.legend(frameon=False)
    fig.savefig(out / "kappa_distributions.pdf")
    # Also drop a copy into paper/figures so it lands in the paper directly.
    paper_fig = Path("paper/figures/fig7_kappa_distributions.pdf")
    paper_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_fig)
    print(f"[xval] wrote {out / 'kappa_distributions.pdf'} (and {paper_fig})")

    # Table 5 fragment
    table_tex = (
        "% Auto-generated by scripts/pipeline_xvalidation.py\n"
        f"$\\kappa^*$ (val-optimal) & {prec:.3f} & {rec:.3f} & {f1:.3f} \\\\\n"
    )
    (out / "xvalidation_table.tex").write_text(table_tex)
    print(f"[xval] wrote {out / 'xvalidation_table.tex'}")

    # Figure 8 case studies: pick three windows
    accident_map = load_accident_map()
    case_indices = []
    # (i) correct LOCA with low kappa
    loca_idx = accident_map.get("LOCA")
    if loca_idx is not None:
        m = (correct_t & (l_t == loca_idx)).nonzero()[0]
        if len(m):
            case_indices.append(("Correct LOCA, low $\\kappa$", int(m[np.argmin(kappa_t[m])])))
    # (ii) LOCA misclassified as LOCAC with elevated kappa
    locac_idx = accident_map.get("LOCAC")
    if loca_idx is not None and locac_idx is not None:
        m = ((l_t == loca_idx) & (p_t == locac_idx)).nonzero()[0]
        if len(m):
            case_indices.append(("LOCA misclassified as LOCAC", int(m[np.argmax(kappa_t[m])])))
    # (iii) ATWS correct, surrogate-uncertain (high kappa among correct ATWS)
    atws_idx = accident_map.get("ATWS")
    if atws_idx is not None:
        m = (correct_t & (l_t == atws_idx)).nonzero()[0]
        if len(m):
            case_indices.append(("Correct ATWS, high $\\kappa$ (surrogate uncertain)", int(m[np.argmax(kappa_t[m])])))

    if case_indices:
        n_cases = len(case_indices)
        fig, axes = plt.subplots(n_cases, 1, figsize=(7, 2.4 * n_cases), constrained_layout=True)
        if n_cases == 1:
            axes = [axes]
        for ax, (title, idx) in zip(axes, case_indices):
            window = test_ds.windows[idx].numpy()
            ax.plot(window[:, 0], label="P (norm.)", color="C0")
            ax.plot(window[:, 57] if window.shape[1] > 57 else window[:, 0], label="PWR (norm.)", color="C1")
            ax.set_title(f"{title} | true={accident_map.get(int(l_t[idx]),'?')}, "
                         f"pred={accident_map.get(int(p_t[idx]),'?')}, "
                         f"$\\kappa$={kappa_t[idx]:.2f}")
            ax.legend(frameon=False, fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.savefig(out / "case_studies.pdf")
        fig.savefig("paper/figures/fig8_case_studies.pdf")
        print(f"[xval] wrote {out / 'case_studies.pdf'} (and paper/figures/fig8_case_studies.pdf)")


if __name__ == "__main__":
    main()
