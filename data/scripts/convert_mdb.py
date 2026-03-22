"""Convert .mdb (MS Access) files to CSV using mdbtools.

This is a fallback script for users who only have the .mdb files.
The preferred approach is to use the pre-converted CSVs from the GitHub repo.

Requirements:
    brew install mdbtools  (macOS)
    apt-get install mdbtools  (Ubuntu/Debian)
"""

import argparse
import subprocess
from pathlib import Path


def convert_mdb_to_csv(mdb_path: Path, output_dir: Path) -> list[Path]:
    """Extract all tables from an .mdb file to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # List tables in the .mdb file
    result = subprocess.run(
        ["mdb-tables", "-1", str(mdb_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    tables = [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]

    csv_paths = []
    for table in tables:
        csv_path = output_dir / f"{mdb_path.stem}_{table}.csv"
        with open(csv_path, "w") as f:
            subprocess.run(
                ["mdb-export", str(mdb_path), table],
                stdout=f,
                check=True,
            )
        csv_paths.append(csv_path)
        print(f"Exported {table} -> {csv_path}")

    return csv_paths


def main():
    parser = argparse.ArgumentParser(description="Convert .mdb files to CSV")
    parser.add_argument("input", type=Path, help="Path to .mdb file or directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/csv"),
        help="Output directory for CSVs",
    )
    args = parser.parse_args()

    if args.input.is_file():
        convert_mdb_to_csv(args.input, args.output_dir)
    elif args.input.is_dir():
        mdb_files = list(args.input.rglob("*.mdb"))
        print(f"Found {len(mdb_files)} .mdb files")
        for mdb_path in mdb_files:
            sub_dir = args.output_dir / mdb_path.parent.name
            convert_mdb_to_csv(mdb_path, sub_dir)
    else:
        print(f"Error: {args.input} not found")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
