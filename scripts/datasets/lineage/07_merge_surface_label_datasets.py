"""Merge multiple functional-surface labels_index datasets into one trainable dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from src.config import PATHS
from src.models.baselines.patchcore import project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate functional-surface labels_index.csv files.")
    parser.add_argument("--base-dataset-dir", type=Path, required=True)
    parser.add_argument("--add-dataset-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-source-name", default="base")
    parser.add_argument("--add-source-name", action="append", default=[])
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    return project_path(str(path))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PATHS.root))
    except ValueError:
        return str(path)


def load_labels(dataset_dir: Path, source_name: str, source_rank: int) -> pd.DataFrame:
    labels_path = dataset_dir / "labels_index.csv"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    df = pd.read_csv(labels_path)
    df["dataset_source"] = source_name
    df["dataset_source_dir"] = rel(dataset_dir)
    df["dataset_source_rank"] = int(source_rank)
    return df


def main() -> None:
    args = parse_args()
    base_dir = resolve(args.base_dataset_dir)
    add_dirs = [resolve(path) for path in args.add_dataset_dir]
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    add_names = list(args.add_source_name)
    while len(add_names) < len(add_dirs):
        add_names.append(f"add_{len(add_names) + 1}")

    frames = [load_labels(base_dir, args.base_source_name, 0)]
    for idx, (dataset_dir, source_name) in enumerate(zip(add_dirs, add_names, strict=True), start=1):
        frames.append(load_labels(dataset_dir, source_name, idx))

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged["image_index"] = range(len(merged))
    merged.to_csv(output_dir / "labels_index.csv", index=False)

    copied = []
    for dataset_dir in [base_dir, *add_dirs]:
        for name in ["params.json", "crop_summary.csv", "prediction_label_correction_summary.json"]:
            source = dataset_dir / name
            if source.exists():
                target = output_dir / f"{dataset_dir.name}_{name}"
                shutil.copy2(source, target)
                copied.append(rel(target))

    summary = {
        "output_dir": rel(output_dir),
        "base_dataset_dir": rel(base_dir),
        "add_dataset_dirs": [rel(path) for path in add_dirs],
        "input_rows_by_source": merged.groupby("dataset_source").size().to_dict(),
        "output_rows": int(len(merged)),
        "copied_metadata": copied,
    }
    (output_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()





