"""Download the NPPAD dataset from GitHub (CSV files only, skipping large .mdb files)."""

import argparse
import subprocess
from pathlib import Path

GITHUB_REPO = "https://github.com/thu-inet/NuclearPowerPlantAccidentData.git"
RAW_DIR = Path("data/raw")


def download_from_github(raw_dir: Path = RAW_DIR) -> None:
    """Clone only CSV data from the NPPAD GitHub repo using sparse checkout.

    The full repo is ~4GB+ due to .mdb files. This downloads only the
    Operation_csv_data and Dose_csv_data directories (~200MB).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = raw_dir / "NuclearPowerPlantAccidentData"

    if clone_dir.exists():
        print(f"Dataset already exists at {clone_dir}")
        print("To re-download, remove the directory and run again.")
        return

    print(f"Cloning NPPAD dataset (CSV files only) from {GITHUB_REPO}...")

    # Initialize empty repo with sparse checkout
    clone_dir.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=clone_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", GITHUB_REPO],
        cwd=clone_dir, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "core.sparseCheckout", "true"],
        cwd=clone_dir, check=True, capture_output=True,
    )

    # Only checkout CSV directories and top-level files
    sparse_file = clone_dir / ".git" / "info" / "sparse-checkout"
    sparse_file.parent.mkdir(parents=True, exist_ok=True)
    sparse_file.write_text(
        "Operation_csv_data/\n"
        "Dose_csv_data/\n"
        "README.md\n"
        "*.csv\n"
    )

    # Pull only the latest commit
    subprocess.run(
        ["git", "pull", "--depth", "1", "origin", "main"],
        cwd=clone_dir, check=True,
    )

    # Clean up git objects to save space
    subprocess.run(
        ["git", "gc", "--aggressive"],
        cwd=clone_dir, capture_output=True,
    )

    # List what we got
    csv_files = list(clone_dir.rglob("*.csv"))
    print(f"Dataset downloaded to {clone_dir}")
    print(f"Found {len(csv_files)} CSV files")

    # Show breakdown by directory
    dirs = set(f.parent.relative_to(clone_dir) for f in csv_files)
    for d in sorted(dirs):
        n = len(list(clone_dir.joinpath(d).glob("*.csv")))
        print(f"  {d}: {n} files")


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
