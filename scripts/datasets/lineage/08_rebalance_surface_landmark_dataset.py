"""Filter a surface/landmark dataset from kept previews and rebalance patterns."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from src.config import PATHS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter deleted-preview rows and rebalance a semantic dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--group-column", default="pattern_id")
    parser.add_argument("--manual-source", default="manual")
    parser.add_argument("--pseudo-source", default="pseudo_v20_multiclass_best")
    parser.add_argument("--strict-total-balance", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PATHS.root / path


def copy_existing(path_value: object, src_root: Path, dst_root: Path) -> str:
    src = resolve(Path(str(path_value)))
    try:
        rel = src.relative_to(src_root)
    except ValueError:
        rel = Path(src.name)
    dst = dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def main() -> None:
    args = parse_args()
    dataset_dir = resolve(args.dataset_dir)
    output_dir = resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(dataset_dir / "labels_index.csv")
    kept_preview_names = {path.name for path in (dataset_dir / "previews").glob("*.png")}
    labels["_preview_name"] = labels["preview_path"].map(lambda value: Path(str(value)).name)
    kept = labels[labels["_preview_name"].isin(kept_preview_names)].copy()
    removed = labels[~labels["_preview_name"].isin(kept_preview_names)].copy()

    manual = kept[kept["label_source"] == args.manual_source].copy()
    pseudo = kept[kept["label_source"] == args.pseudo_source].copy()
    if pseudo.empty:
        selected = manual.copy()
        pseudo_per_pattern = 0
    else:
        pseudo_counts = pseudo.groupby(args.group_column).size()
        pseudo_per_pattern = int(pseudo_counts.min())
        sort_columns = [args.group_column]
        if "selection_score" in pseudo.columns:
            sort_columns.append("selection_score")
        selected_pseudo = (
            pseudo.sort_values(sort_columns, ascending=[True] * len(sort_columns))
            .groupby(args.group_column, group_keys=False)
            .head(pseudo_per_pattern)
        )
        selected = pd.concat([manual, selected_pseudo], ignore_index=True)

    if args.strict_total_balance and not selected.empty:
        target_total = int(selected.groupby(args.group_column).size().min())
        balanced_parts = []
        for _group, group in selected.groupby(args.group_column, sort=True):
            manual_group = group[group["label_source"] == args.manual_source]
            pseudo_group = group[group["label_source"] == args.pseudo_source]
            remaining = max(0, target_total - len(manual_group))
            sort_columns = ["selection_score"] if "selection_score" in pseudo_group.columns else []
            if sort_columns:
                pseudo_group = pseudo_group.sort_values(sort_columns)
            balanced_parts.append(pd.concat([manual_group, pseudo_group.head(remaining)], ignore_index=True))
        selected = pd.concat(balanced_parts, ignore_index=True)

    selected = selected.drop(columns=["_preview_name"], errors="ignore").reset_index(drop=True)
    removed = removed.drop(columns=["_preview_name"], errors="ignore").reset_index(drop=True)
    selected_source_paths = set(selected["preview_path"].astype(str))
    rebalanced_excluded = kept[~kept["preview_path"].astype(str).isin(selected_source_paths)].drop(columns=["_preview_name"], errors="ignore")
    excluded = pd.concat([removed, rebalanced_excluded], ignore_index=True)

    for column in [
        "semantic_mask_path",
        "surface_mask_path",
        "landmark_mask_path",
        "preview_path",
        "prob_path",
    ]:
        if column not in selected.columns:
            continue
        selected[column] = selected[column].map(
            lambda value: copy_existing(value, dataset_dir, output_dir) if pd.notna(value) and str(value).strip() else value
        )

    selected.to_csv(output_dir / "labels_index.csv", index=False)
    excluded.to_csv(output_dir / "excluded_labels_index.csv", index=False)
    selected.groupby([args.group_column, "label_source"]).size().reset_index(name="count").to_csv(output_dir / "source_counts.csv", index=False)
    params = {
        "input_dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "input_rows": int(len(labels)),
        "kept_after_preview_filter": int(len(kept)),
        "removed_by_deleted_preview": int(len(removed)),
        "removed_by_rebalance": int(len(rebalanced_excluded)),
        "output_rows": int(len(selected)),
        "pseudo_per_pattern": int(pseudo_per_pattern),
        "counts_by_pattern": selected[args.group_column].value_counts().sort_index().to_dict(),
        "counts_by_source": {
            f"{group}::{source}": int(count)
            for (group, source), count in selected.groupby([args.group_column, "label_source"]).size().to_dict().items()
        },
    }
    (output_dir / "params.json").write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")
    print(json.dumps(params, indent=2, default=str))


if __name__ == "__main__":
    main()





