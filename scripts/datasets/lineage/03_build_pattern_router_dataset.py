"""Build the final pattern-router training dataset from classified folders."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path


DATASETS = {
    "C1": Path("data/classified/Casting_class1/train/good"),
    "C2": Path("data/classified/Casting_class2_cluster_aligned/train/good"),
    "C3": Path("data/classified/Casting_class3/train/good"),
}

FOLDER_RENAMES = {
    "C3/manual": "C3_pattern_bridge_two_holes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize final combined casting pattern classifier dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/classified/Casting_pattern_router/train/good"),
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("reports/tables/summary/casting_pattern_router_train_manifest.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("reports/tables/summary/casting_pattern_router_train_summary.csv"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_folder(prefix: str, folder_name: str) -> str:
    rename_key = f"{prefix}/{folder_name}"
    if rename_key in FOLDER_RENAMES:
        return FOLDER_RENAMES[rename_key]
    if folder_name.startswith(prefix + "_"):
        return folder_name
    return f"{prefix}_{folder_name}"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project_root = Path.cwd()
    rows: list[dict[str, str]] = []

    for prefix, source_root in DATASETS.items():
        for source_folder in sorted(path for path in source_root.iterdir() if path.is_dir()):
            route_folder = output_folder(prefix, source_folder.name)
            destination_dir = args.output_dir / route_folder
            destination_dir.mkdir(parents=True, exist_ok=True)
            for source in sorted(source_folder.glob("*.jpg")):
                destination = destination_dir / f"{prefix}_{source.name}"
                if destination.exists() and not args.overwrite:
                    action = "skipped_existing"
                else:
                    shutil.copy2(source, destination)
                    action = "copied"
                rows.append(
                    {
                        "source_prefix": prefix,
                        "source_folder": source_folder.name,
                        "route_folder": route_folder,
                        "source_path": source.resolve().relative_to(project_root.resolve()).as_posix(),
                        "classified_path": destination.resolve().relative_to(project_root.resolve()).as_posix(),
                        "action": action,
                    }
                )

    write_csv(args.manifest_csv, rows)
    write_csv(args.output_dir / "classified_manifest.csv", rows)
    counts = Counter(row["route_folder"] for row in rows)
    summary_rows = [
        {"route_folder": folder, "images": str(count)}
        for folder, count in sorted(counts.items())
    ]
    write_csv(args.summary_csv, summary_rows)
    write_csv(args.output_dir / "classified_summary.csv", summary_rows)

    print(f"Wrote {len(rows)} rows to {args.output_dir}")
    print(f"Routes: {len(summary_rows)}")


if __name__ == "__main__":
    main()





