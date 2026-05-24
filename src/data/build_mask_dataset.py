"""Build a trainable surface/landmark semantic dataset from multiclass predictions."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from src.config import PATHS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build labels_index from multiclass prediction_summary.csv.")
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-preview", action="append", default=[])
    parser.add_argument(
        "--balance-mode",
        choices=["sample_weight", "strict_subset", "none"],
        default="sample_weight",
        help="sample_weight keeps all rows and balances loss contribution by pattern.",
    )
    parser.add_argument("--label-source", default="pseudo_v21_epoch014_multiclass_refined")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PATHS.root / path


def copy_path(value: object, input_root: Path, output_root: Path) -> str:
    source = resolve(Path(str(value)))
    try:
        rel = source.relative_to(input_root)
    except ValueError:
        rel = Path(source.name)
    target = output_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target)


def main() -> None:
    args = parse_args()
    predictions_dir = resolve(args.predictions_dir)
    output_dir = resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(predictions_dir / "prediction_summary.csv")
    exclude_names = {Path(value).name for value in args.exclude_preview}
    df["_preview_name"] = df["preview_path"].map(lambda value: Path(str(value)).name)
    kept = df[~df["_preview_name"].isin(exclude_names)].copy()
    excluded = df[df["_preview_name"].isin(exclude_names)].copy()

    if args.balance_mode == "strict_subset":
        target_count = int(kept.groupby("pattern_id").size().min())
        kept = (
            kept.sort_values(["pattern_id", "surface_ratio", "landmark_ratio"])
            .groupby("pattern_id", group_keys=False)
            .head(target_count)
            .copy()
        )
    if args.balance_mode == "sample_weight":
        counts = kept.groupby("pattern_id").size().to_dict()
        mean_count = sum(counts.values()) / max(len(counts), 1)
        kept["sample_weight"] = kept["pattern_id"].map(lambda pattern: float(mean_count / max(counts.get(pattern, 1), 1)))
    else:
        kept["sample_weight"] = 1.0

    output_rows = []
    for idx, row in kept.reset_index(drop=True).iterrows():
        pattern = str(row["pattern_id"])
        stem = f"{idx:04d}_{pattern}_{Path(str(row['image_path'])).stem}"
        semantic_path = copy_path(row["semantic_mask_path"], predictions_dir, output_dir)
        surface_path = copy_path(row["surface_mask_path"], predictions_dir, output_dir)
        landmark_path = copy_path(row["landmark_mask_path"], predictions_dir, output_dir)
        preview_path = copy_path(row["preview_path"], predictions_dir, output_dir)
        prob_path = copy_path(row["prob_path"], predictions_dir, output_dir)
        output_rows.append(
            {
                "image_path": str(row["image_path"]),
                "category": "Casting_class1",
                "split": "train",
                "label": "good",
                "pattern_id": pattern,
                "base_pattern": pattern,
                "semantic_mask_path": semantic_path,
                "surface_mask_path": surface_path,
                "landmark_mask_path": landmark_path,
                "preview_path": preview_path,
                "prob_path": prob_path,
                "surface_ratio": float(row["surface_ratio"]),
                "landmark_ratio": float(row["landmark_ratio"]),
                "label_source": args.label_source,
                "sample_weight": float(row["sample_weight"]),
                "source_prediction_index": int(idx),
            }
        )

    labels = pd.DataFrame(output_rows)
    labels.to_csv(output_dir / "labels_index.csv", index=False)
    excluded.drop(columns=["_preview_name"], errors="ignore").to_csv(output_dir / "excluded_bad_predictions.csv", index=False)
    labels.groupby("pattern_id").agg(
        count=("image_path", "count"),
        sample_weight_mean=("sample_weight", "mean"),
        effective_weight=("sample_weight", "sum"),
        surface_ratio_mean=("surface_ratio", "mean"),
        landmark_ratio_mean=("landmark_ratio", "mean"),
    ).to_csv(output_dir / "pattern_balance_summary.csv")
    params = {
        "predictions_dir": str(predictions_dir),
        "output_dir": str(output_dir),
        "input_rows": int(len(df)),
        "excluded_rows": int(len(excluded)),
        "output_rows": int(len(labels)),
        "balance_mode": args.balance_mode,
        "counts_by_pattern": labels["pattern_id"].value_counts().sort_index().to_dict(),
        "effective_weight_by_pattern": labels.groupby("pattern_id")["sample_weight"].sum().round(6).to_dict(),
        "excluded_preview_names": sorted(exclude_names),
    }
    (output_dir / "params.json").write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")
    print(json.dumps(params, indent=2, default=str))


if __name__ == "__main__":
    main()





