"""CLI configuration for Feature AE training."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PATHS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Feature-AE on normal images.")
    parser.add_argument("--category", default=None)
    parser.add_argument("--categories", nargs="+", default=None, help="Train on a selected list of normal categories.")
    parser.add_argument("--all-categories", action="store_true")
    parser.add_argument(
        "--model-type",
        default="feature_ae_resnet18",
        choices=[
            "feature_ae_resnet18",
            "feature_ae_resnet18_dual_context",
            "feature_ae_resnet18_dual_context_gated",
            "reverse_distill_resnet18",
            "reverse_distill_resnet18_dual_context_gated",
        ],
    )
    parser.add_argument("--teacher-backbone", default="resnet18", choices=["resnet18"])
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--preprocessing-mode", choices=["letterbox", "tile_256_overlap"], default="letterbox")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--context-tile-size", type=int, default=None)
    parser.add_argument("--tile-train-stride", type=int, default=256)
    parser.add_argument("--tile-train-sampling", choices=["all", "random"], default="all")
    parser.add_argument("--tile-train-max-tiles-per-image", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--init-checkpoint-path",
        type=Path,
        default=None,
        help="Optional Feature-AE checkpoint used to initialize the student weights before training.",
    )
    parser.add_argument("--loss", choices=["l2_cosine"], default="l2_cosine")
    parser.add_argument("--cosine-weight", type=float, default=0.5)
    parser.add_argument(
        "--layer-loss-weights",
        nargs="*",
        default=None,
        help="Optional per-layer training weights, e.g. layer2=0.65 layer3=0.35.",
    )
    parser.add_argument("--normalization", choices=["imagenet"], default="imagenet")
    parser.add_argument(
        "--augmentation-profile",
        choices=["none", "default", "toothbrush", "toothbrush_headprior", "casting_microdefect", "auto"],
        default="none",
    )
    parser.add_argument("--repeat-factor", default="auto")
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=0,
        help="Also save checkpoint_epoch_XXX.pt every N epochs; 0 disables periodic checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        nargs="*",
        default=None,
        help="Also save checkpoint_epoch_XXX.pt for these exact epochs.",
    )
    parser.add_argument(
        "--metric-eval-every-epochs",
        type=int,
        default=0,
        help="Evaluate anomaly metrics every N epochs and save metric-best checkpoints; 0 disables it.",
    )
    parser.add_argument("--metric-eval-start-epoch", type=int, default=1)
    parser.add_argument("--metric-eval-category", default=None)
    parser.add_argument(
        "--metric-eval-output-dir",
        type=Path,
        default=PATHS.root / "models" / "feature_ae_eval_during_training",
    )
    parser.add_argument("--metric-eval-device", default=None)
    parser.add_argument("--metric-eval-batch-size", type=int, default=None)
    parser.add_argument("--metric-eval-tile-stride", type=int, default=None)
    parser.add_argument("--metric-eval-tile-aggregation", choices=["mean", "max"], default="mean")
    parser.add_argument("--metric-eval-layer-weights", nargs="*", default=["auto"])
    parser.add_argument("--metric-eval-calibrate-normal", action="store_true")
    parser.add_argument("--metric-eval-calibration-mode", choices=["fused", "per_layer"], default="per_layer")
    parser.add_argument("--metric-eval-calibration-stat", choices=["mean_std", "median_mad"], default="median_mad")
    parser.add_argument("--metric-eval-calibration-max-images", type=int, default=120)
    parser.add_argument(
        "--metric-eval-score-region",
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
    parser.add_argument("--metric-eval-roi-predictions-dir", type=Path, nargs="*", default=None)
    parser.add_argument("--metric-eval-roi-threshold", type=float, default=0.30)
    parser.add_argument("--metric-eval-roi-dilate-radius", type=int, default=0)
    parser.add_argument("--metric-eval-apply-score-region-to-map", action="store_true")
    parser.add_argument("--metric-eval-score-smoothing", choices=["none", "median3", "gaussian"], default="none")
    parser.add_argument("--metric-eval-score-image", choices=["percentile99", "topk_mean", "max"], default="topk_mean")
    parser.add_argument("--metric-eval-topk-fraction", type=float, default=0.0005)
    parser.add_argument("--metric-eval-save-score-maps", action="store_true")
    parser.add_argument("--metric-eval-save-previews", action="store_true")
    parser.add_argument("--metric-eval-max-previews", type=int, default=30)
    parser.add_argument("--metric-eval-no-progress", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=["none", "plateau"], default="none")
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--roi-predictions-dir",
        type=Path,
        default=None,
        help="Optional functional-surface prediction directory used to weight Feature-AE loss.",
    )
    parser.add_argument("--roi-threshold", type=float, default=0.30)
    parser.add_argument("--roi-loss-weight", type=float, default=1.0)
    parser.add_argument("--background-loss-weight", type=float, default=0.02)
    parser.add_argument("--roi-dilate-radius", type=int, default=0)
    parser.add_argument("--min-roi-ratio", type=float, default=0.0)
    parser.add_argument(
        "--normal-structure-loss-weight",
        type=float,
        default=0.0,
        help="Extra normal-only loss weight on dark hole/thread-like structures inside the ROI.",
    )
    parser.add_argument("--normal-structure-dark-percentile", type=float, default=8.0)
    parser.add_argument("--normal-structure-min-area", type=int, default=50)
    parser.add_argument("--normal-structure-max-area-ratio", type=float, default=0.10)
    parser.add_argument("--normal-structure-dilate-radius", type=int, default=8)
    parser.add_argument(
        "--roi-border-loss-weight",
        type=float,
        default=0.0,
        help="Extra normal-only loss weight on the inner ROI border, useful for normal hole/thread edges.",
    )
    parser.add_argument("--roi-border-radius", type=int, default=12)
    return parser.parse_args()


