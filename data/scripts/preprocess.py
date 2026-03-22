"""Preprocess NPPAD dataset: normalize, create sliding windows, split."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.config import Config


# Mapping from actual directory names to accident type indices
ACCIDENT_TYPES = {
    "Normal": 0,
    "LOCA": 1,      # Loss of Coolant Accident
    "LOCAC": 2,     # Loss of Coolant Accident (Cold leg)
    "SLBIC": 3,     # Steam Line Break Inside Containment
    "SLBOC": 4,     # Steam Line Break Outside Containment
    "SGATR": 5,     # Steam Generator A Tube Rupture
    "SGBTR": 6,     # Steam Generator B Tube Rupture
    "FLB": 7,       # Feedwater Line Break
    "LLB": 8,       # Large Line Break
    "RW": 9,        # Rod Withdrawal
    "RI": 10,       # Rod Insertion
    "LR": 11,       # Load Rejection
    "TT": 12,       # Turbine Trip
    "MD": 13,       # Malfunction of equipment / Misdiagnosis
    "LOF": 14,      # Loss of Flow
    "LACP": 15,     # Loss of AC Power
    "ATWS": 16,     # Anticipated Transient Without Scram
    "SP": 17,       # Station Power
}

# Use the 96 known feature columns (excluding TIME) for consistency
FEATURE_COLUMNS = [
    "P", "TAVG", "THA", "THB", "TCA", "TCB", "WRCA", "WRCB",
    "PSGA", "PSGB", "WFWA", "WFWB", "WSTA", "WSTB", "VOL", "LVPZ",
    "VOID", "WLR", "WUP", "HUP", "HLW", "WHPI", "WECS", "QMWT",
    "LSGA", "LSGB", "QMGA", "QMGB", "NSGA", "NSGB", "TBLD", "WTRA",
    "WTRB", "TSAT", "QRHR", "LVCR", "SCMA", "SCMB", "FRCL", "PRB",
    "PRBA", "TRB", "LWRB", "DNBR", "QFCL", "WBK", "WSPY", "WCSP",
    "HTR", "MH2", "CNH2", "RHBR", "RHMT", "RHFL", "RHRD", "RH",
    "PWNT", "PWR", "TFSB", "TFPK", "TF", "TPCT", "WCFT", "WLPI",
    "WCHG", "RM1", "RM2", "RM3", "RM4", "RC87", "RC131", "STRB",
    "STSG", "STTB", "RBLK", "SGLK", "DTHY", "DWB", "WRLA", "WRLB",
    "WLD", "MBK", "EBK", "TKLV", "FRZR", "TDBR", "MDBR", "MCRT",
    "MGAS", "TCRT", "TSLP", "PPM", "RRCA", "RRCB", "RRCO", "WFLB",
]


def find_csv_files(raw_dir: Path) -> list[dict]:
    """Find all operational CSV files and extract metadata."""
    records = []
    nppad_dir = raw_dir / "NuclearPowerPlantAccidentData"

    if not nppad_dir.exists():
        raise FileNotFoundError(
            f"NPPAD data not found at {nppad_dir}. Run download.py first."
        )

    op_csv_dir = nppad_dir / "Operation_csv_data"
    if not op_csv_dir.exists():
        raise FileNotFoundError(
            f"Operation_csv_data not found at {op_csv_dir}"
        )

    for csv_path in sorted(op_csv_dir.rglob("*.csv")):
        accident_type = csv_path.parent.name
        if accident_type not in ACCIDENT_TYPES:
            print(f"Warning: unknown accident type directory '{accident_type}', skipping")
            continue
        records.append({
            "path": csv_path,
            "accident_type": accident_type,
            "scenario_id": f"{accident_type}_{csv_path.stem}",
        })

    types_found = set(r["accident_type"] for r in records)
    print(f"Found {len(records)} CSV files across {len(types_found)} accident types")
    print(f"Types: {sorted(types_found)}")
    return records


def load_and_concat(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all CSVs and concatenate with metadata, using consistent columns."""
    dfs = []
    metadata_rows = []
    skipped = 0

    for rec in records:
        try:
            df = pd.read_csv(rec["path"])
        except Exception as e:
            print(f"Warning: could not read {rec['path']}: {e}")
            skipped += 1
            continue

        # Use only the known feature columns that exist in this file
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        if len(available) < len(FEATURE_COLUMNS):
            # Add missing columns as zeros
            for c in FEATURE_COLUMNS:
                if c not in df.columns:
                    df[c] = 0.0

        df_features = df[FEATURE_COLUMNS]

        n_rows = len(df_features)
        meta = pd.DataFrame({
            "scenario_id": [rec["scenario_id"]] * n_rows,
            "accident_type": [rec["accident_type"]] * n_rows,
        })

        dfs.append(df_features)
        metadata_rows.append(meta)

    if skipped:
        print(f"Skipped {skipped} unreadable files")

    data = pd.concat(dfs, ignore_index=True)
    metadata = pd.concat(metadata_rows, ignore_index=True)

    print(f"Loaded {len(data)} total timesteps, {data.shape[1]} features")
    return data, metadata


def preprocess(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    config: Config,
) -> dict:
    """Normalize, create windows, and split the data."""
    processed_dir = Path(config.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    features = data.values.astype(np.float32)
    n_features = features.shape[1]
    print(f"Using {n_features} features")

    # Handle NaN/inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Fit scaler on normal data only
    is_normal = (metadata["accident_type"] == "Normal").values
    scaler = StandardScaler()
    if is_normal.any():
        scaler.fit(features[is_normal])
        print(f"Scaler fit on {is_normal.sum()} normal timesteps")
    else:
        print("Warning: no normal data found, fitting scaler on all data")
        scaler.fit(features)

    features_scaled = scaler.transform(features).astype(np.float32)

    # Save scaler
    with open(processed_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Split ACCIDENT scenarios into train/val/test (not the Normal scenario)
    scenarios = metadata["scenario_id"].unique()
    scenario_types = {
        sid: metadata[metadata["scenario_id"] == sid]["accident_type"].iloc[0]
        for sid in scenarios
    }

    normal_scenarios = [s for s in scenarios if scenario_types[s] == "Normal"]
    accident_scenarios = np.array([s for s in scenarios if scenario_types[s] != "Normal"])

    print(f"Normal scenarios: {len(normal_scenarios)}, Accident scenarios: {len(accident_scenarios)}")

    # Split accident scenarios
    train_scenarios, temp_scenarios = train_test_split(
        accident_scenarios,
        test_size=config.data.val_ratio + config.data.test_ratio,
        random_state=config.train.seed,
    )
    relative_test = config.data.test_ratio / (config.data.val_ratio + config.data.test_ratio)
    val_scenarios, test_scenarios = train_test_split(
        temp_scenarios,
        test_size=relative_test,
        random_state=config.train.seed,
    )

    # Add normal data to all splits
    train_scenarios = np.concatenate([train_scenarios, normal_scenarios])
    val_scenarios = np.concatenate([val_scenarios, normal_scenarios])
    test_scenarios = np.concatenate([test_scenarios, normal_scenarios])

    print(f"Split: {len(train_scenarios)} train, {len(val_scenarios)} val, {len(test_scenarios)} test scenarios")

    # Create windows for each split
    stride = config.data.stride  # Use same stride for all splits to keep sizes manageable
    splits = {}
    for split_name, split_scenarios in [
        ("train", train_scenarios),
        ("val", val_scenarios),
        ("test", test_scenarios),
    ]:
        windows = []
        labels = []
        accident_types_list = []

        for sid in split_scenarios:
            mask = (metadata["scenario_id"] == sid).values
            scenario_data = features_scaled[mask]
            atype = scenario_types[sid]
            is_anomaly = 0 if atype == "Normal" else 1

            for start in range(0, len(scenario_data) - config.data.window_size + 1, stride):
                window = scenario_data[start : start + config.data.window_size]
                windows.append(window)
                labels.append(is_anomaly)
                accident_types_list.append(ACCIDENT_TYPES.get(atype, -1))

        if windows:
            splits[split_name] = {
                "windows": torch.tensor(np.array(windows), dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
                "accident_types": torch.tensor(accident_types_list, dtype=torch.long),
            }
            n_normal = sum(1 for l in labels if l == 0)
            n_anomaly = sum(1 for l in labels if l == 1)
            print(
                f"{split_name}: {len(windows)} windows "
                f"({n_normal} normal, {n_anomaly} anomaly), "
                f"shape {splits[split_name]['windows'].shape}"
            )
        else:
            print(f"Warning: no windows created for {split_name} split")

    # Save splits
    for split_name in ("train", "val", "test"):
        if split_name in splits:
            save_path = processed_dir / f"{split_name}.pt"
            torch.save(splits[split_name], save_path)
            print(f"Saved {save_path}")

    # Save metadata
    torch.save(
        {"feature_names": FEATURE_COLUMNS, "n_features": n_features},
        processed_dir / "metadata.pt",
    )

    return splits


def main():
    parser = argparse.ArgumentParser(description="Preprocess NPPAD dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    raw_dir = Path(config.data.raw_dir)

    records = find_csv_files(raw_dir)
    data, metadata = load_and_concat(records)
    preprocess(data, metadata, config)
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
