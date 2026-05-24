"""CLI configuration for functional-surface training."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import PATHS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multiclass functional-surface segmenter.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--semantic-mask-column", default="semantic_mask_path")
    parser.add_argument(
        "--model-type",
        choices=[
            "functional_unet_resnet18",
            "functional_unet_resnet18_det1",
            "functional_unet_resnet18_det1_context2b",
            "functional_unet_resnet18_det1_context2b_recon",
            "functional_unet_resnet18_det1_context_fpn",
            "functional_unet_resnet18_det1_context_fpn_light",
        ],
        default="functional_unet_resnet18_det1_context2b",
    )
    parser.add_argument(
        "--init-checkpoint-path",
        type=Path,
        default=None,
        help="Optional checkpoint used to initialize the direct segmenter before fine-tuning.",
    )
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument(
        "--context-size",
        type=int,
        default=None,
        help="Global context branch size. Defaults to --input-size; use 768 with --input-size 384 for V23.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--split-strategy", choices=["random", "stratified"], default="stratified")
    parser.add_argument("--split-column", default="pattern_id")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument(
        "--recon-weight",
        type=float,
        default=0.0,
        help="Auxiliary bottleneck reconstruction loss weight when the model exposes recon_logits.",
    )
    parser.add_argument("--dice-classes", default="1,2", help="Comma-separated classes included in Dice loss.")
    parser.add_argument("--class-weights", default="auto", help="'auto', 'none', or comma-separated weights.")
    parser.add_argument("--augmentation-profile", default="functional_surface_full_source_512_to_256_large_context")
    parser.add_argument("--repeat-factor", default="32")
    parser.add_argument("--context-crop-prob", type=float, default=1.0)
    parser.add_argument("--positive-crop-prob", type=float, default=0.30)
    parser.add_argument("--train-photometric-normalization-p", type=float, default=0.20)
    parser.add_argument("--train-photo-target-p05", type=float, default=0.03)
    parser.add_argument("--train-photo-target-p95", type=float, default=0.60)
    parser.add_argument("--synthetic-defect-p", type=float, default=0.0)
    parser.add_argument(
        "--synthetic-defect-mode",
        choices=["generic", "realistic", "mixed"],
        default="generic",
        help=(
            "generic draws synthetic blobs; realistic pastes true defect patches from "
            "--synthetic-defect-library-json; mixed combines both for large-defect invariance. "
            "Use shape weight 'machined' to draw smooth circular machining-like defects."
        ),
    )
    parser.add_argument(
        "--synthetic-defect-realistic-render",
        choices=["paste", "residual"],
        default="paste",
        help=(
            "How realistic defects are rendered. 'paste' copies RGB patches; "
            "'residual' uses real defect mask shapes but applies empirical "
            "contrast/noise to the target surface, avoiding copy-paste artifacts."
        ),
    )
    parser.add_argument("--synthetic-defect-library-json", type=Path, default=None)
    parser.add_argument(
        "--synthetic-defect-texture-library-json",
        type=Path,
        default=None,
        help="Optional extracted texture library used by procedural machined defects.",
    )
    parser.add_argument(
        "--synthetic-defect-photometric-library-json",
        type=Path,
        default=None,
        help="Optional real-defect lighting/contrast cluster library used to condition synthetic defect photometry.",
    )
    parser.add_argument(
        "--synthetic-defect-pattern-aware",
        action="store_true",
        help="Condition machined synthetic defect size/texture sampling on pattern_id. P4 gets larger round machined defects.",
    )
    parser.add_argument(
        "--synthetic-defect-p4-large-p",
        type=float,
        default=0.75,
        help="When pattern-aware, probability that P4 machined defects sample the largest P4 texture components.",
    )
    parser.add_argument("--synthetic-defect-max-blobs", type=int, default=5)
    parser.add_argument("--synthetic-defect-min-radius-frac", type=float, default=0.012)
    parser.add_argument("--synthetic-defect-max-radius-frac", type=float, default=0.055)
    parser.add_argument(
        "--synthetic-defect-shape-weights",
        default="hole:0.45,scratch:0.35,stain:0.20",
        help="Comma-separated procedural shape weights used by generic/mixed modes. Supported: hole,scratch,stain,machined.",
    )
    parser.add_argument("--synthetic-defect-scratch-min-length-frac", type=float, default=0.08)
    parser.add_argument("--synthetic-defect-scratch-max-length-frac", type=float, default=0.45)
    parser.add_argument("--synthetic-defect-scratch-p", type=float, default=0.35)
    parser.add_argument("--synthetic-defect-texture-strength", type=float, default=1.0)
    parser.add_argument(
        "--synthetic-defect-variant-strength",
        type=float,
        default=1.0,
        help="Shape/texture variation strength for realistic pasted defects. 0 keeps patches close to source; 1 is the default.",
    )
    parser.add_argument(
        "--synthetic-defect-large-p",
        type=float,
        default=0.0,
        help="Probability that a realistic pasted defect is sampled from the largest components.",
    )
    parser.add_argument("--synthetic-defect-large-quantile", type=float, default=0.75)
    parser.add_argument("--synthetic-defect-large-scale-min", type=float, default=1.15)
    parser.add_argument("--synthetic-defect-large-scale-max", type=float, default=2.10)
    parser.add_argument("--synthetic-defect-alpha-min", type=float, default=0.65)
    parser.add_argument("--synthetic-defect-alpha-max", type=float, default=1.0)
    parser.add_argument(
        "--synthetic-defect-bg-match-strength",
        type=float,
        default=0.45,
        help="How strongly pasted real defects are shifted toward the target local background. Lower keeps original contrast.",
    )
    parser.add_argument(
        "--synthetic-defect-min-surface-overlap",
        type=float,
        default=0.80,
        help=(
            "Minimum fraction of pasted defect alpha pixels that must fall on class-1 surface. "
            "Use a lower value for sparse whole-mask defect motifs."
        ),
    )
    parser.add_argument(
        "--synthetic-defect-context-consistent",
        action="store_true",
        help=(
            "Apply synthetic defects on the workspace image before local crop extraction, "
            "so local and global/context branches see the same simulated defect."
        ),
    )
    parser.add_argument(
        "--synthetic-defect-crop-localized",
        action="store_true",
        help=(
            "When --synthetic-defect-context-consistent is enabled, sample defect centers "
            "inside the selected local crop so the supervised crop reliably contains a defect."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument(
        "--external-monitor-labels-dir",
        type=Path,
        default=None,
        help="Optional corrected labels_index dataset evaluated at every epoch and never used for training.",
    )
    parser.add_argument(
        "--external-monitor-name",
        default="external",
        help="Prefix used in loss_history.csv for --external-monitor-labels-dir metrics.",
    )
    parser.add_argument(
        "--best-monitor",
        choices=["val_loss", "external_loss", "val_recon_loss", "external_recon_loss"],
        default="val_loss",
        help="Metric used for checkpoint_best.pt. external_* requires --external-monitor-labels-dir.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--lr-scheduler", choices=["none", "plateau"], default="plateau")
    parser.add_argument("--lr-patience", type=int, default=6)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=0,
        help="Save checkpoint_epoch_XXX.pt every N epochs. Use 1 to keep every epoch.",
    )
    parser.add_argument("--save-previews", action="store_true")
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument(
        "--preview-head",
        choices=["mask", "recon"],
        default="mask",
        help="Head used for saved validation previews.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=PATHS.root / "models" / "functional_surface")
    parser.add_argument("--overwrite-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


