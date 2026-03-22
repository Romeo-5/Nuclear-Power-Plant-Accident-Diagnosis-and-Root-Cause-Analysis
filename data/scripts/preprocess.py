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


# Mapping from directory/file naming conventions to accident types
ACCIDENT_TYPES = {
    "Normal": 0,
    "LOCA": 1,      # Loss of Coolant Accident
    "SLBIC": 2,     # Steam Line Break Inside Containment
    "SLBOC": 3,     # Steam Line Break Outside Containment
    "SGTR": 4,      # Steam Generator Tube Rupture
    "MSLB": 5,      # Main Steam Line Break
    "LOFW": 6,      # Loss of Feedwater
    "LOCV": 7,      # Loss of Condenser Vacuum
    "LOOP": 8,      # Loss of Offsite Power
    "RW": 9,        # Rod Withdrawal
    "RWAP": 10,     # Rod Withdrawal at Power
    "RWAS": 11,     # Rod Withdrawal at Startup
    "BE": 12,       # Boron Dilution Event
    "RI": 13,       # Rod Insertion
    "FLB": 14,      # Feedwater Line Break
    "LR": 15,       # Load Rejection
    "TT": 16,       # Turbine Trip
    "RCP": 17,      # Reactor Coolant Pump
}


def find_csv_files(raw_dir: Path) -> list[dict]:
    """Find all operational CSV files and extract metadata."""
    records = []
    nppad_dir = raw_dir / "NuclearPowerPlantAccidentData"

    if not nppad_dir.exists():
        raise FileNotFoundError(
            f"NPPAD data not found at {nppad_dir}. Run download.py first."
        )

    # Search for CSV files in the Operation_csv_data directory
    csv_dirs = list(nppad_dir.rglob("Operation_csv_data"))
    if not csv_dirs:
        # Fallback: search for any CSV files
        csv_dirs = [nppad_dir]

    for csv_dir in csv_dirs:
        for csv_path in sorted(csv_dir.rglob("*.csv")):
            # Extract accident type from path
            accident_type = _infer_accident_type(csv_path)
            records.append({
                "path": csv_path,
                "accident_type": accident_type,
                "scenario_id": csv_path.stem,
            })

    print(f"Found {len(records)} CSV files across {len(set(r['accident_type'] for r in records))} accident types")
    return records


def _infer_accident_type(csv_path: Path) -> str:
    """Infer accident type from file path components."""
    path_str = str(csv_path).upper()
    for accident_name in ACCIDENT_TYPES:
        if accident_name.upper() in path_str:
            return accident_name
    return "Normal"


def load_and_concat(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all CSVs and concatenate with metadata."""
    dfs = []
    metadata_rows = []

    for rec in records:
        try:
            df = pd.read_csv(rec["path"])
        except Exception as e:
            print(f"Warning: could not read {rec['path']}: {e}")
            continue

        # Add scenario tracking
        n_rows = len(df)
        meta = pd.DataFrame({
            "scenario_id": [rec["scenario_id"]] * n_rows,
            "accident_type": [rec["accident_type"]] * n_rows,
        })

        dfs.append(df)
        metadata_rows.append(meta)

    data = pd.concat(dfs, ignore_index=True)
    metadata = pd.concat(metadata_rows, ignore_index=True)

    print(f"Loaded {len(data)} total timesteps, {data.shape[1]} columns")
    return data, metadata


def preprocess(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    config: Config,
) -> dict:
    """Normalize, create windows, and split the data."""
    processed_dir = Path(config.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Drop non-numeric and time columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    time_cols = [c for c in numeric_cols if c.upper() in ("TIME", "T", "TIME(S)", "TIME (S)")]
    feature_cols = [c for c in numeric_cols if c not in time_cols]

    print(f"Using {len(feature_cols)} numeric feature columns")
    features = data[feature_cols].values.astype(np.float32)

    # Handle NaN/inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Fit scaler on normal data only
    is_normal = (metadata["accident_type"] == "Normal").values
    scaler = StandardScaler()
    if is_normal.any():
        scaler.fit(features[is_normal])
    else:
        print("Warning: no normal data found, fitting scaler on all data")
        scaler.fit(features)

    features_scaled = scaler.transform(features).astype(np.float32)

    # Save scaler
    with open(processed_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {processed_dir / 'scaler.pkl'}")

    # Create sliding windows per scenario
    scenarios = metadata["scenario_id"].unique()
    scenario_types = {
        sid: metadata[metadata["scenario_id"] == sid]["accident_type"].iloc[0]
        for sid in scenarios
    }

    # Split scenarios into train/val/test
    train_scenarios, temp_scenarios = train_test_split(
        scenarios,
        test_size=config.data.val_ratio + config.data.test_ratio,
        random_state=config.train.seed,
    )
    relative_test = config.data.test_ratio / (config.data.val_ratio + config.data.test_ratio)
    val_scenarios, test_scenarios = train_test_split(
        temp_scenarios,
        test_size=relative_test,
        random_state=config.train.seed,
    )

    print(f"Split: {len(train_scenarios)} train, {len(val_scenarios)} val, {len(test_scenarios)} test scenarios")

    n_features = features_scaled.shape[1]
    splits = {}
    for split_name, split_scenarios, stride in [
        ("train", train_scenarios, config.data.stride),
        ("val", val_scenarios, config.data.eval_stride),
        ("test", test_scenarios, config.data.eval_stride),
    ]:
        windows = []
        labels = []
        accident_types = []

        for sid in split_scenarios:
            mask = (metadata["scenario_id"] == sid).values
            scenario_data = features_scaled[mask]
            atype = scenario_types[sid]
            is_anomaly = 0 if atype == "Normal" else 1

            for start in range(0, len(scenario_data) - config.data.window_size + 1, stride):
                window = scenario_data[start : start + config.data.window_size]
                windows.append(window)
                labels.append(is_anomaly)
                accident_types.append(ACCIDENT_TYPES.get(atype, -1))

        if windows:
            splits[split_name] = {
                "windows": torch.tensor(np.array(windows), dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long),
                "accident_types": torch.tensor(accident_types, dtype=torch.long),
            }
            print(f"{split_name}: {len(windows)} windows, shape {splits[split_name]['windows'].shape}")
        else:
            print(f"Warning: no windows created for {split_name} split")

    # Save feature column names
    splits["feature_names"] = feature_cols
    splits["n_features"] = n_features

    # Save splits
    for split_name in ("train", "val", "test"):
        if split_name in splits:
            save_path = processed_dir / f"{split_name}.pt"
            torch.save(splits[split_name], save_path)
            print(f"Saved {save_path}")

    # Save metadata
    torch.save(
        {"feature_names": feature_cols, "n_features": n_features},
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
