"""Copy classified Casting_class1 train images into a pattern folder tree.

The script consumes ``casting_class1_train_pattern_assignments.csv`` generated
by ``classify_casting_class1_patterns.py`` and materializes a human-readable
folder layout under ``data/classified``.

Examples
--------
    python scripts/datasets/lineage/01_materialize_casting_class1_patterns.py
    python scripts/datasets/lineage/01_materialize_casting_class1_patterns.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from src.data.lineage import copy_file, project_relative, write_csv


PATTERN_DIRS = {
    "P1": "P1_piece_B_view_1_2",
    "P2": "P2_piece_B_view_1_3",
    "P3": "P3_piece_B_view_2_3",
    "P4": "P4_piece_A_single_2_3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy Casting_class1 train/good images into pattern-specific folders."
    )
    parser.add_argument(
        "--assignments-csv",
        type=Path,
        default=Path("reports/tables/summary/casting_class1_train_pattern_assignments.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/classified"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_assignments(assignments_csv: Path, output_root: Path, overwrite: bool) -> tuple[list[dict[str, str]], Counter[str]]:
    project_root = Path.cwd()
    target_root = output_root / "Casting_class1" / "train" / "good"
    target_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()

    with assignments_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pattern_id = row["pattern_id"]
            if pattern_id not in PATTERN_DIRS:
                raise ValueError(f"Unknown pattern_id={pattern_id!r} in {assignments_csv}")

            source = project_root / row["image_path"]
            if not source.exists():
                raise FileNotFoundError(source)

            destination_dir = target_root / PATTERN_DIRS[pattern_id]
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / source.name

            action = copy_file(source, destination, overwrite=overwrite)

            counts[pattern_id] += 1
            manifest_rows.append(
                {
                    "pattern_id": pattern_id,
                    "piece_family": row["piece_family"],
                    "pattern_view": row["pattern_view"],
                    "group_kind": row["group_kind"],
                    "source_path": row["image_path"],
                    "classified_path": project_relative(destination, project_root),
                    "action": action,
                }
            )

    manifest_path = target_root / "classified_manifest.csv"
    write_csv(manifest_path, manifest_rows)

    summary_path = target_root / "classified_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pattern_id", "folder", "images"])
        writer.writeheader()
        for pattern_id, folder in PATTERN_DIRS.items():
            writer.writerow({"pattern_id": pattern_id, "folder": folder, "images": counts[pattern_id]})

    return manifest_rows, counts


def main() -> None:
    args = parse_args()
    rows, counts = copy_assignments(args.assignments_csv, args.output_root, args.overwrite)
    print(f"Materialized {len(rows)} classified images under {args.output_root / 'Casting_class1' / 'train' / 'good'}")
    for pattern_id in sorted(PATTERN_DIRS):
        print(f"{pattern_id}: {counts[pattern_id]}")


if __name__ == "__main__":
    main()






