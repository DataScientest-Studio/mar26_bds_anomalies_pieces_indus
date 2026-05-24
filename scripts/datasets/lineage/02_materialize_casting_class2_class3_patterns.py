"""Materialize classified train folders for Casting_class2 and Casting_class3.

Class2 and Class3 do not currently have hand-authored business pattern labels
like Casting_class1. This script uses the stable view key encoded at the end of
each filename as the routing class, then also builds a combined Casting_all
folder suitable for a single classifier over all Casting routes.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from pathlib import Path


FILENAME_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}_\d{3})(?P<sep>[-_])(?P<suffix>.+)$"
)

CLASS1_COMBINED_FOLDERS = {
    "P1": "C1_P1_piece_B_view_1_2",
    "P2": "C1_P2_piece_B_view_1_3",
    "P3": "C1_P3_piece_B_view_2_3",
    "P4": "C1_P4_piece_A_single_2_3",
}

CLASS_VIEW_FOLDERS = {
    "Casting_class2": {
        "1_1": "C2_view_1_1",
        "1_2": "C2_view_1_2",
        "1_3": "C2_view_1_3",
        "2_1": "C2_view_2_1",
        "2_2": "C2_view_2_2",
    },
    "Casting_class3": {
        "0_2": "C3_view_0_2",
        "2_1": "C3_view_2_1",
        "3_1": "C3_view_3_1",
        "3_2": "C3_view_3_2",
        "3_3": "C3_view_3_3",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify and copy Casting_class2/3 train images by view key.")
    parser.add_argument("--input-csv", type=Path, default=Path("data/processed/unified_dataset.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("data/classified"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/tables/summary"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_image_name(image_path: str) -> dict[str, str]:
    stem = Path(image_path).stem
    match = FILENAME_RE.match(stem)
    if not match:
        return {
            "base_timestamp": "",
            "acquisition_group": stem,
            "raw_suffix": stem,
            "view_key": "",
        }

    timestamp = match.group("timestamp")
    suffix = match.group("suffix")
    tokens = suffix.split("_")
    view_key = "_".join(tokens[-2:]) if len(tokens) >= 2 else suffix
    acquisition_group = f"{timestamp}-{tokens[0]}" if len(tokens) >= 3 else timestamp
    return {
        "base_timestamp": timestamp,
        "acquisition_group": acquisition_group,
        "raw_suffix": suffix,
        "view_key": view_key,
    }


def read_rows(input_csv: Path, category: str) -> list[dict[str, str]]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            row
            for row in reader
            if row["category"] == category and row["split"] == "train" and row["label"] == "good"
        ]
    for row in rows:
        row.update(parse_image_name(row["image_path"]))
        row["route_id"] = f"{category}_{row['view_key']}"
        row["route_folder"] = CLASS_VIEW_FOLDERS[category][row["view_key"]]
    return rows


def copy_rows(
    rows: list[dict[str, str]],
    output_root: Path,
    category: str,
    overwrite: bool,
) -> list[dict[str, str]]:
    project_root = Path.cwd()
    manifest_rows: list[dict[str, str]] = []
    for row in rows:
        source = project_root / row["image_path"]
        if not source.exists():
            raise FileNotFoundError(source)
        destination_dir = output_root / category / "train" / "good" / row["route_folder"]
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        if destination.exists() and not overwrite:
            action = "skipped_existing"
        else:
            shutil.copy2(source, destination)
            action = "copied"
        manifest_rows.append(
            {
                "dataset": row["dataset"],
                "category": row["category"],
                "split": row["split"],
                "label": row["label"],
                "image_path": row["image_path"],
                "base_timestamp": row["base_timestamp"],
                "acquisition_group": row["acquisition_group"],
                "raw_suffix": row["raw_suffix"],
                "view_key": row["view_key"],
                "route_id": row["route_id"],
                "route_folder": row["route_folder"],
                "classified_path": project_relative(destination, project_root),
                "action": action,
            }
        )
    return manifest_rows


def project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str]], folder_key: str = "route_folder") -> None:
    counts = Counter(row[folder_key] for row in rows)
    summary_rows = [
        {"route_folder": folder, "images": count}
        for folder, count in sorted(counts.items())
    ]
    write_csv(path, summary_rows)


def copy_to_combined_from_manifest(
    manifest_rows: list[dict[str, str]],
    output_root: Path,
    combined_folder_map: dict[str, str],
    route_key: str,
    overwrite: bool,
) -> list[dict[str, str]]:
    project_root = Path.cwd()
    combined_rows: list[dict[str, str]] = []
    combined_root = output_root / "Casting_all" / "train" / "good"
    for row in manifest_rows:
        key = row[route_key]
        folder = combined_folder_map[key]
        source = project_root / row["image_path"]
        destination_dir = combined_root / folder
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        if destination.exists() and not overwrite:
            action = "skipped_existing"
        else:
            shutil.copy2(source, destination)
            action = "copied"
        combined_rows.append(
            {
                "category": row["category"],
                "source_route": key,
                "combined_route_folder": folder,
                "source_path": row["image_path"],
                "combined_path": project_relative(destination, project_root),
                "action": action,
            }
        )
    return combined_rows


def load_class1_assignments(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    all_combined_rows: list[dict[str, str]] = []

    class1_path = args.reports_dir / "casting_class1_train_pattern_assignments.csv"
    if class1_path.exists():
        class1_rows = load_class1_assignments(class1_path)
        all_combined_rows.extend(
            copy_to_combined_from_manifest(
                class1_rows,
                args.output_root,
                CLASS1_COMBINED_FOLDERS,
                route_key="pattern_id",
                overwrite=args.overwrite,
            )
        )

    for category in ("Casting_class2", "Casting_class3"):
        rows = read_rows(args.input_csv, category)
        manifest_rows = copy_rows(rows, args.output_root, category, args.overwrite)
        assignments_path = args.reports_dir / f"{category.lower()}_train_pattern_assignments.csv"
        summary_path = args.reports_dir / f"{category.lower()}_train_pattern_assignment_summary.csv"
        write_csv(assignments_path, manifest_rows)
        write_summary(summary_path, manifest_rows)

        manifest_path = args.output_root / category / "train" / "good" / "classified_manifest.csv"
        summary_out_path = args.output_root / category / "train" / "good" / "classified_summary.csv"
        write_csv(manifest_path, manifest_rows)
        write_summary(summary_out_path, manifest_rows)

        all_combined_rows.extend(
            copy_to_combined_from_manifest(
                manifest_rows,
                args.output_root,
                CLASS_VIEW_FOLDERS[category],
                route_key="view_key",
                overwrite=args.overwrite,
            )
        )
        print(f"{category}: materialized {len(rows)} train/good images")

    if all_combined_rows:
        combined_manifest = args.output_root / "Casting_all" / "train" / "good" / "classified_manifest.csv"
        combined_summary = args.output_root / "Casting_all" / "train" / "good" / "classified_summary.csv"
        write_csv(combined_manifest, all_combined_rows)
        write_summary(combined_summary, all_combined_rows, folder_key="combined_route_folder")
        print(f"Casting_all: materialized {len(all_combined_rows)} train/good images")


if __name__ == "__main__":
    main()





