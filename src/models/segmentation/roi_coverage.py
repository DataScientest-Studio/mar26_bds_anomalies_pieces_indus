"""Evaluate whether functional-surface ROI masks cover annotated defects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.config import PATHS
from src.features.functional_surface import preview_panel, safe_stem
from src.models.baselines.patchcore import project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure GT defect coverage by functional ROI predictions.")
    parser.add_argument("--category", action="append", required=True)
    parser.add_argument("--prediction-dir", type=Path, action="append", required=True)
    parser.add_argument("--run-label", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage-threshold", type=float, default=0.95)
    parser.add_argument("--defect-dilate", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=40)
    return parser.parse_args()


def resolve(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PATHS.root / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PATHS.root))
    except ValueError:
        return str(path)


def path_key(path: Path | str) -> str:
    resolved = resolve(path)
    try:
        resolved = resolved.resolve()
    except OSError:
        pass
    return str(resolved).replace("\\", "/").lower()


def load_binary(path: Path | str, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(resolve(path)).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 127


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if int(radius) <= 0 or not bool(mask.any()):
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    # PIL MaxFilter size must be odd.
    size = int(radius) * 2 + 1
    from PIL import ImageFilter

    return np.asarray(image.filter(ImageFilter.MaxFilter(size=size)), dtype=np.uint8) > 127


def prediction_lookup(prediction_dir: Path) -> dict[str, dict]:
    summary_path = prediction_dir / "prediction_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    df = pd.read_csv(summary_path)
    if "mask_path" not in df.columns and "surface_mask_path" in df.columns:
        df["mask_path"] = df["surface_mask_path"]
    if "pred_ratio" not in df.columns and "surface_ratio" in df.columns:
        df["pred_ratio"] = df["surface_ratio"]
    if "mask_path" not in df.columns:
        raise KeyError(f"{summary_path} must contain mask_path or surface_mask_path.")
    return {path_key(row["image_path"]): row.to_dict() for _, row in df.iterrows()}


def defect_rows(category: str) -> pd.DataFrame:
    df = pd.read_csv(PATHS.processed_dir / "unified_dataset.csv")
    rows = df[
        (df["category"].astype(str) == str(category))
        & (df["split"].astype(str) == "test")
        & (df["label"].astype(str) == "defective")
        & (df["has_mask"].astype(str).str.lower().isin(["true", "1"]))
        & df["mask_path"].notna()
    ].copy()
    if rows.empty:
        raise ValueError(f"No defective test masks found for category={category!r}")
    return rows.reset_index(drop=True)


def write_preview(row: dict, output_path: Path) -> None:
    image = Image.open(project_path(str(row["image_path"]))).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    defect = load_binary(row["defect_mask_path"], image.size)
    roi = load_binary(row["roi_mask_path"], image.size)
    missed = defect & ~roi
    covered = defect & roi

    roi_overlay = rgb.copy()
    roi_overlay[..., 0] = np.maximum(roi_overlay[..., 0], roi.astype(np.float32) * 0.90)
    roi_overlay[..., 1] = np.maximum(roi_overlay[..., 1], roi.astype(np.float32) * 0.55)

    defect_overlay = rgb.copy()
    defect_overlay[..., 1] = np.maximum(defect_overlay[..., 1], covered.astype(np.float32))
    defect_overlay[..., 0] = np.maximum(defect_overlay[..., 0], missed.astype(np.float32))

    preview_panel(
        [
            ("image", rgb),
            ("ROI", roi.astype(np.float32)),
            ("GT defect", defect.astype(np.float32)),
            ("missed defect", missed.astype(np.float32)),
            ("ROI overlay", roi_overlay),
            ("green=covered red=missed", defect_overlay),
        ],
        (
            f"{row['run_label']} | {row['category']} | coverage={row['defect_coverage']:.3f} "
            f"missed={row['missed_defect_pixels']}/{row['defect_pixels']}"
        ),
        output_path,
    )


def main() -> None:
    args = parse_args()
    if not (len(args.category) == len(args.prediction_dir) == len(args.run_label)):
        raise ValueError("--category, --prediction-dir and --run-label must have the same count.")

    output_dir = resolve(args.output_dir)
    preview_dir = output_dir / "low_coverage_previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for category, pred_dir_raw, run_label in zip(args.category, args.prediction_dir, args.run_label, strict=True):
        prediction_dir = resolve(pred_dir_raw)
        lookup = prediction_lookup(prediction_dir)
        for _, defect_row in defect_rows(category).iterrows():
            image_path = str(defect_row["image_path"])
            pred = lookup.get(path_key(image_path))
            if pred is None:
                raise KeyError(f"Image missing from prediction_summary.csv for {run_label}: {image_path}")
            image = Image.open(project_path(image_path)).convert("RGB")
            defect = load_binary(defect_row["mask_path"], image.size)
            defect_eval = dilate_mask(defect, int(args.defect_dilate))
            roi = load_binary(pred["mask_path"], image.size)
            defect_pixels = int(defect_eval.sum())
            covered_pixels = int((defect_eval & roi).sum())
            missed_pixels = int((defect_eval & ~roi).sum())
            coverage = float(covered_pixels / max(defect_pixels, 1))
            all_rows.append(
                {
                    "run_label": run_label,
                    "category": category,
                    "image_path": image_path,
                    "defect_mask_path": str(defect_row["mask_path"]),
                    "roi_mask_path": str(pred["mask_path"]),
                    "roi_pred_ratio": float(pred.get("pred_ratio", np.nan)),
                    "defect_pixels": defect_pixels,
                    "covered_defect_pixels": covered_pixels,
                    "missed_defect_pixels": missed_pixels,
                    "defect_coverage": coverage,
                    "below_threshold": bool(coverage < float(args.coverage_threshold)),
                }
            )

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(output_dir / "defect_coverage_by_image.csv", index=False)
    by_run = (
        metrics.groupby(["run_label", "category"])
        .agg(
            images=("image_path", "count"),
            defect_coverage_mean=("defect_coverage", "mean"),
            defect_coverage_min=("defect_coverage", "min"),
            below_threshold=("below_threshold", "sum"),
            missed_pixels_total=("missed_defect_pixels", "sum"),
            defect_pixels_total=("defect_pixels", "sum"),
        )
        .reset_index()
    )
    by_run["pixel_weighted_coverage"] = 1.0 - (
        by_run["missed_pixels_total"].astype(float) / by_run["defect_pixels_total"].clip(lower=1).astype(float)
    )
    by_run.to_csv(output_dir / "defect_coverage_summary.csv", index=False)

    low = metrics.sort_values(["below_threshold", "defect_coverage", "missed_defect_pixels"], ascending=[False, True, False])
    for idx, row in enumerate(low[low["below_threshold"]].head(int(args.preview_count)).to_dict("records")):
        write_preview(row, preview_dir / f"{idx:03d}_{row['run_label']}_{row['category']}_{safe_stem(row['image_path'])}.png")

    summary = {
        "output_dir": rel(output_dir),
        "rows": int(len(metrics)),
        "coverage_threshold": float(args.coverage_threshold),
        "defect_dilate": int(args.defect_dilate),
        "summary_csv": rel(output_dir / "defect_coverage_summary.csv"),
        "metrics_csv": rel(output_dir / "defect_coverage_by_image.csv"),
        "low_coverage_previews": rel(preview_dir),
    }
    (output_dir / "params.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(by_run.to_string(index=False))


if __name__ == "__main__":
    main()






