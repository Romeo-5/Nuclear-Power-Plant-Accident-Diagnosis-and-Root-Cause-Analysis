"""SHAP physics-validation for the stage-2 accident classifier.

Computes SHAP attributions for a trained classifier on a chosen set of
accident classes, rolls the per-sensor importance up to physical parameter
groups (primary pressure, pressurizer, SG secondary, feedwater, ...), and
emits:

    results/shap/feature_importance_<CLASS>.csv
    results/shap/group_importance_<CLASS>.csv
    results/shap/temporal_importance_<CLASS>.csv
    results/shap/summary_table.tex          # LaTeX-ready Table 3 fragment
    results/shap/temporal_plot.pdf          # Figure 4 contents

Paper targets: Table 3 and Figure 4 in `paper/sections/05_results.tex`.

Usage:
    python scripts/shap_physics_validation.py \\
        --config configs/lstm_classifier.yaml \\
        --checkpoint checkpoints/lstm_classifier/best.pt \\
        --classes LOCA SGATR SGBTR FLB \\
        --n-samples 100 \\
        --n-background 100 \\
        --output-dir results/shap

Notes:
- Requires the `shap` package (pip install shap).
- `--checkpoint` should point to a PyTorch state_dict saved by the existing
  trainer. If the trainer wraps the model under a different key, adjust
  `load_checkpoint` below.
- Class names follow `ACCIDENT_TYPES` in `data/scripts/preprocess.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Physical parameter groupings for the 96 NPPAD sensors.
# ---------------------------------------------------------------------------
# These groupings are based on standard PWR instrumentation semantics. They
# are used to roll individual SHAP scores up into the physically meaningful
# categories referenced in the paper (Table 3). If your understanding of any
# channel differs from the grouping below, edit this dict.

PHYSICAL_GROUPS: dict[str, list[str]] = {
    "primary_pressure": ["P"],
    "primary_temperatures": ["TAVG", "THA", "THB", "TCA", "TCB", "TSAT"],
    "primary_flow": ["WRCA", "WRCB", "WLR", "WUP"],
    "pressurizer": ["VOL", "LVPZ", "WSPY", "WCSP", "HTR", "HUP", "HLW"],
    "sg_secondary": [
        "PSGA", "PSGB", "LSGA", "LSGB", "NSGA", "NSGB", "QMGA", "QMGB",
    ],
    "feedwater": ["WFWA", "WFWB"],
    "steam": ["WSTA", "WSTB", "STSG", "STTB", "STRB"],
    "reactor_power": ["QMWT", "PWR", "PWNT", "TFSB", "TFPK", "TF", "TPCT"],
    "safety_injection": ["WHPI", "WECS", "WLPI", "WCHG", "WCFT"],
    "boron": ["PPM"],
    "containment": [
        "PRB", "PRBA", "TRB", "LWRB", "CNH2", "MH2",
        "FRCL", "QFCL", "SCMA", "SCMB",
    ],
    "radiation": ["RM1", "RM2", "RM3", "RM4", "RC87", "RC131"],
    "control_rods": ["RRCA", "RRCB", "RRCO"],
    "residual_heat_removal": ["QRHR", "RHBR", "RHMT", "RHFL", "RHRD", "RH"],
    "break_flow": ["WBK", "WFLB", "MBK", "EBK", "DWB"],
}

# Physically expected top groups for the canonical accident classes the
# paper validates. These come from first-principles PWR accident analysis
# (see e.g. Duderstadt & Hamilton and the PCTRAN user manual).
EXPECTED_GROUPS: dict[str, list[str]] = {
    "LOCA":  ["primary_pressure", "pressurizer", "safety_injection", "containment"],
    "LOCAC": ["primary_pressure", "pressurizer", "safety_injection", "containment"],
    "SGATR": ["sg_secondary", "primary_pressure", "radiation"],
    "SGBTR": ["sg_secondary", "primary_pressure", "radiation"],
    "FLB":   ["feedwater", "sg_secondary", "steam"],
    "SLBIC": ["steam", "sg_secondary", "containment"],
    "SLBOC": ["steam", "sg_secondary"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_accident_type_map() -> dict[str, int]:
    """Import the canonical ACCIDENT_TYPES mapping from the preprocessing script."""
    import importlib.util

    pre_path = Path("data/scripts/preprocess.py")
    spec = importlib.util.spec_from_file_location("preprocess", pre_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.ACCIDENT_TYPES


def load_feature_names(processed_dir: Path) -> list[str]:
    """Read the feature-name list saved by preprocessing."""
    meta_path = processed_dir / "metadata.pt"
    if meta_path.exists():
        meta = torch.load(meta_path, weights_only=False)
        return list(meta["feature_names"])
    # Fall back to the list hard-coded in preprocessing (should be kept in sync).
    import importlib.util
    pre_path = Path("data/scripts/preprocess.py")
    spec = importlib.util.spec_from_file_location("preprocess", pre_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return list(module.FEATURE_COLUMNS)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> torch.nn.Module:
    """Load a state_dict into a freshly-built model.

    Handles both plain state_dicts and common trainer wrapper formats
    (`{"model_state_dict": ...}` or `{"state_dict": ...}`).
    """
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        state = blob["model_state_dict"]
    elif isinstance(blob, dict) and "state_dict" in blob:
        state = blob["state_dict"]
    else:
        state = blob
    model.load_state_dict(state)
    return model


def select_windows_for_class(
    dataset, class_idx: int, n_samples: int, rng: np.random.Generator
) -> torch.Tensor:
    """Return up to n_samples windows for a given accident class."""
    mask = (dataset.accident_types == class_idx).numpy()
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"No test windows found for class index {class_idx}.")
    chosen = rng.choice(idx, size=min(n_samples, len(idx)), replace=False)
    return dataset.windows[chosen]


def aggregate_per_feature(shap_class_array: np.ndarray) -> np.ndarray:
    """Mean absolute SHAP value per feature.

    shap_class_array: (N, T, F) -> returns (F,)
    """
    return np.mean(np.abs(shap_class_array), axis=(0, 1))


def aggregate_per_timestep(shap_class_array: np.ndarray) -> np.ndarray:
    """Mean absolute SHAP value per timestep.

    shap_class_array: (N, T, F) -> returns (T,)
    """
    return np.mean(np.abs(shap_class_array), axis=(0, 2))


def rollup_to_groups(
    feature_importance: dict[str, float],
) -> dict[str, float]:
    """Sum per-feature importance into physical-parameter groups."""
    group_totals: dict[str, float] = {g: 0.0 for g in PHYSICAL_GROUPS}
    assigned: set[str] = set()
    for group, members in PHYSICAL_GROUPS.items():
        for sensor in members:
            if sensor in feature_importance:
                group_totals[group] += feature_importance[sensor]
                assigned.add(sensor)
    other = sum(v for k, v in feature_importance.items() if k not in assigned)
    group_totals["other"] = other
    return dict(sorted(group_totals.items(), key=lambda kv: -kv[1]))


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# LaTeX emission
# ---------------------------------------------------------------------------

def format_group_name(group: str) -> str:
    return group.replace("_", " ")


def emit_latex_table(
    per_class: dict[str, dict[str, float]],
    expected: dict[str, list[str]],
    output_path: Path,
    top_k: int = 4,
) -> None:
    """Write a LaTeX fragment suitable for drop-in replacement of Table 3."""
    lines = [
        "% Auto-generated by scripts/shap_physics_validation.py",
        "% Drop the body rows below into Table 3 in sections/05_results.tex.",
        "",
    ]
    for cls, groups in per_class.items():
        top = [g for g, _ in list(groups.items())[:top_k] if g != "other"]
        observed = ", ".join(format_group_name(g) for g in top)
        exp = ", ".join(format_group_name(g) for g in expected.get(cls, []))
        overlap = set(top) & set(expected.get(cls, []))
        overlap_note = ""
        if expected.get(cls):
            overlap_note = (
                f" \\textit{{({len(overlap)}/{len(expected[cls])} expected groups "
                f"in top-{top_k})}}"
            )
        lines.append(
            f"{cls} & {exp} & {observed}{overlap_note} \\\\"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHAP physics-validation for the stage-2 classifier."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to classifier config YAML (e.g. configs/lstm_classifier.yaml).")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint.")
    parser.add_argument("--classes", type=str, nargs="+",
                        default=["LOCA", "SGATR", "SGBTR", "FLB"],
                        help="Accident class abbreviations to validate.")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of test windows per class.")
    parser.add_argument("--n-background", type=int, default=100,
                        help="Number of background windows for DeepExplainer.")
    parser.add_argument("--output-dir", type=str, default="results/shap")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu | cuda | mps.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Imports held here so that `--help` works without a full env.
    from src.data.dataset import NPPADWindowDataset
    from src.models.diagnosis import build_classifier
    from src.models.diagnosis.interpretability import compute_shap_values
    from src.utils.config import Config

    config = Config.from_yaml(args.config)
    processed_dir = Path(config.data.processed_dir)

    # Data
    test_dataset = NPPADWindowDataset(processed_dir / "test.pt")
    train_dataset = NPPADWindowDataset(processed_dir / "train.pt")
    feature_names = load_feature_names(processed_dir)
    assert len(feature_names) == test_dataset.windows.shape[-1], (
        "Feature-name list length does not match tensor feature dimension."
    )

    # Model
    model = build_classifier(config)
    model = load_checkpoint(model, Path(args.checkpoint))
    model.eval()

    # Background set: random windows from the training set.
    bg_idx = rng.choice(
        len(train_dataset.windows),
        size=min(args.n_background, len(train_dataset.windows)),
        replace=False,
    )
    background = train_dataset.windows[bg_idx]

    # Map class abbreviations to integer indices.
    accident_map = load_accident_type_map()
    unknown = [c for c in args.classes if c not in accident_map]
    if unknown:
        raise SystemExit(
            f"Unknown accident class(es): {unknown}. "
            f"Known: {sorted(accident_map)}"
        )

    # Gather windows per class.
    per_class_windows: dict[str, torch.Tensor] = {}
    per_class_indices: dict[str, int] = {}
    for cls in args.classes:
        idx = accident_map[cls]
        per_class_indices[cls] = idx
        per_class_windows[cls] = select_windows_for_class(
            test_dataset, idx, args.n_samples, rng
        )

    # Stack for a single SHAP call.
    concat = torch.cat(list(per_class_windows.values()), dim=0)
    print(
        f"[shap] Computing SHAP values on {concat.shape[0]} windows "
        f"({', '.join(f'{c}={t.shape[0]}' for c, t in per_class_windows.items())}) "
        f"with background size {background.shape[0]} on {args.device}."
    )
    shap_values = compute_shap_values(
        model, concat, background=background, device=args.device
    )

    # shap_values is a list of length n_classes; each entry has shape (N, T, F).
    # Slice it back out per input-class block so we can compute importance.
    per_class_summary: dict[str, dict[str, float]] = {}
    offset = 0
    temporal_by_class: dict[str, np.ndarray] = {}
    for cls, windows in per_class_windows.items():
        n = windows.shape[0]
        cls_idx = per_class_indices[cls]
        sv_block = shap_values[cls_idx][offset : offset + n]
        offset += n

        feat_imp = aggregate_per_feature(sv_block)
        feat_dict = dict(
            sorted(
                zip(feature_names, feat_imp.tolist()), key=lambda kv: -kv[1]
            )
        )
        group_dict = rollup_to_groups(feat_dict)
        temp_imp = aggregate_per_timestep(sv_block)

        # Write CSVs.
        write_csv(
            output_dir / f"feature_importance_{cls}.csv",
            ["sensor", "mean_abs_shap"],
            [[k, v] for k, v in feat_dict.items()],
        )
        write_csv(
            output_dir / f"group_importance_{cls}.csv",
            ["group", "mean_abs_shap"],
            [[k, v] for k, v in group_dict.items()],
        )
        write_csv(
            output_dir / f"temporal_importance_{cls}.csv",
            ["timestep", "mean_abs_shap"],
            [[t, float(v)] for t, v in enumerate(temp_imp)],
        )

        per_class_summary[cls] = group_dict
        temporal_by_class[cls] = temp_imp

    # Emit LaTeX Table 3 fragment.
    emit_latex_table(
        per_class_summary,
        EXPECTED_GROUPS,
        output_dir / "summary_table.tex",
    )

    # Emit temporal-importance plot (Figure 4).
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
        for cls, temp in temporal_by_class.items():
            ax.plot(temp / temp.max(), label=cls)
        ax.set_xlabel("Timestep in window")
        ax.set_ylabel("Mean |SHAP| (per-class normalized)")
        ax.set_title("Temporal SHAP importance by accident class")
        ax.legend(frameon=False)
        fig.savefig(output_dir / "temporal_plot.pdf")
        plt.close(fig)
    except ImportError:
        print("[shap] matplotlib not installed; skipping temporal_plot.pdf")

    # Also write a machine-readable summary for downstream tooling.
    summary_json = {
        cls: {
            "top_groups": list(groups.keys())[:5],
            "expected_groups": EXPECTED_GROUPS.get(cls, []),
            "group_scores": groups,
        }
        for cls, groups in per_class_summary.items()
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2) + "\n"
    )

    print(f"[shap] Wrote results to {output_dir}/")
    print(f"[shap] Drop {output_dir}/summary_table.tex rows into Table 3.")


if __name__ == "__main__":
    main()
