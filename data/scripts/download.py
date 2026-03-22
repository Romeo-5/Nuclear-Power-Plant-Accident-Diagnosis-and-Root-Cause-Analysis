"""Download the NPPAD dataset from GitHub or Figshare."""

import argparse
import subprocess
import sys
from pathlib import Path

GITHUB_REPO = "https://github.com/thu-inet/NuclearPowerPlantAccidentData.git"
RAW_DIR = Path("data/raw")


def download_from_github(raw_dir: Path = RAW_DIR) -> None:
    """Clone the NPPAD GitHub repo containing pre-converted CSVs."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = raw_dir / "NuclearPowerPlantAccidentData"

    if clone_dir.exists():
        print(f"Dataset already exists at {clone_dir}")
        print("To re-download, remove the directory and run again.")
        return

    print(f"Cloning NPPAD dataset from {GITHUB_REPO}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", GITHUB_REPO, str(clone_dir)],
        check=True,
    )
    print(f"Dataset downloaded to {clone_dir}")

    # List available data
    csv_dirs = list(clone_dir.rglob("*.csv"))
    print(f"Found {len(csv_dirs)} CSV files")


def main():
    parser = argparse.ArgumentParser(description="Download NPPAD dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory to save raw data (default: data/raw)",
    )
    args = parser.parse_args()

    download_from_github(args.output_dir)


if __name__ == "__main__":
    main()
