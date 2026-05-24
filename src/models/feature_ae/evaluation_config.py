"""CLI configuration for Feature AE evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PATHS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Feature-AE.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--layers", nargs="+", default=None)
    parser.add_argument("--teacher-backbone", default=None, choices=["resnet18"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=PATHS.root / "models" / "feature_ae_eval")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--preprocessing-mode", choices=["letterbox", "tile_256_overlap"], default="letterbox")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--context-tile-size", type=int, default=None)
    parser.add_argument("--tile-stride", type=int, default=128)
    parser.add_argument("--tile-aggregation", choices=["mean", "max"], default="mean")
    parser.add_argument("--cosine-weight", type=float, default=None)
    parser.add_argument("--calibrate-normal", action="store_true")
    parser.add_argument("--calibration-mode", choices=["fused", "per_layer"], default="fused")
    parser.add_argument("--calibration-max-images", type=int, default=60)
    parser.add_argument("--calibration-epsilon", type=float, default=1e-4)
    parser.add_argument("--layer-weights", nargs="*", default=["auto"])
    parser.add_argument(
        "--score-region",
        choices=[
            "full",
            "object_bbox",
            "toothbrush_head",
            "object_bbox_margin",
            "toothbrush_head_margin",
            "casting_surface",
            "casting_surface_margin",
            "functional_surface_prediction",
            "functional_surface_prediction_margin",
        ],
        default="full",
    )
    parser.add_argument("--roi-margin", type=float, default=0.10)
    parser.add_argument(
        "--roi-predictions-dir",
        type=Path,
        nargs="*",
        default=None,
        help="Functional-surface prediction dirs used by functional_surface_prediction score regions.",
    )
    parser.add_argument("--roi-threshold", type=float, default=0.30)
    parser.add_argument("--roi-dilate-radius", type=int, default=0)
    parser.add_argument("--apply-score-region-to-map", action="store_true")
    parser.add_argument("--score-smoothing", choices=["none", "median3", "gaussian"], default="none")
    parser.add_argument("--calibration-stat", choices=["mean_std", "median_mad"], default="mean_std")
    parser.add_argument("--object-threshold", type=float, default=0.08)
    parser.add_argument("--head-x-min", type=float, default=0.18)
    parser.add_argument("--head-x-max", type=float, default=0.82)
    parser.add_argument("--head-y-min", type=float, default=0.05)
    parser.add_argument("--head-y-max", type=float, default=0.72)
    parser.add_argument("--score-image", choices=["percentile99", "topk_mean", "max"], default="percentile99")
    parser.add_argument("--topk-fraction", type=float, default=0.01)
    parser.add_argument("--save-score-maps", action="store_true")
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--max-previews", type=int, default=30)
    parser.add_argument(
        "--preview-score-min-percentile",
        type=float,
        default=0.0,
        help="Preview-only lower percentile computed inside the ROI; values below it are shown as zero.",
    )
    parser.add_argument(
        "--preview-score-max-percentile",
        type=float,
        default=99.0,
        help="Preview-only upper percentile computed inside the ROI for heatmap clipping.",
    )
    parser.add_argument(
        "--preview-score-gamma",
        type=float,
        default=1.0,
        help="Preview-only gamma applied after percentile normalization.",
    )
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


