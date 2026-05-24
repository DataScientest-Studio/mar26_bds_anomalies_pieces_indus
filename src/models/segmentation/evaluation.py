"""Evaluate a functional-surface checkpoint on a labels_index crop dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.mask_datasets import FunctionalSurfaceLabelDataset, collate
from src.config import PATHS
from src.features.functional_surface import preview_panel, safe_stem
from src.models.segmentation.models import build_segmentation_model
from src.models.baselines.patchcore import ResizeLetterbox, project_path, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate functional-surface checkpoint on labels_index rows.")
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preview-count", type=int, default=120)
    parser.add_argument("--worst-preview-count", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def mask_logits_from_model_output(output) -> torch.Tensor:
    if isinstance(output, dict):
        return output["mask_logits"]
    return output


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PATHS.root / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PATHS.root))
    except ValueError:
        return str(path)


def binary_metrics(prob: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float]:
    pred = prob >= float(threshold)
    true = target >= 0.5
    tp = float((pred & true).sum())
    fp = float((pred & ~true).sum())
    fn = float((~pred & true).sum())
    union = tp + fp + fn
    dice_den = 2.0 * tp + fp + fn
    return {
        "pred_ratio": float(pred.mean()),
        "target_ratio": float(true.mean()),
        "iou": float(tp / union) if union > 0 else 1.0,
        "dice": float((2.0 * tp) / dice_den) if dice_den > 0 else 1.0,
        "false_positive_ratio": float(fp / pred.size),
        "false_negative_ratio": float(fn / pred.size),
        "mean_abs_error": float(np.abs(prob - target).mean()),
    }


def write_preview(
    row: dict,
    prob: np.ndarray,
    target: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    metrics: dict[str, float],
    input_size: int,
    threshold: float,
    path: Path,
) -> None:
    raw_image = Image.open(project_path(row["image_path"])).convert("RGB")
    image = np.asarray(ResizeLetterbox(int(input_size))(raw_image), dtype=np.float32) / 255.0
    pred_binary = (prob >= float(threshold)).astype(np.float32)
    overlay = image.copy()
    overlay[..., 0] = np.maximum(overlay[..., 0], pred_binary * 0.95)
    overlay[..., 1] = np.maximum(overlay[..., 1], target * 0.75)
    preview_panel(
        [
            ("image", image),
            ("target", target),
            ("prediction", prob),
            ("pred binary", pred_binary),
            ("positive", positive),
            ("negative", negative),
            ("overlay red=pred green=target", overlay),
        ],
        f"iou={metrics['iou']:.3f} dice={metrics['dice']:.3f} pred={metrics['pred_ratio']:.3f} target={metrics['target_ratio']:.3f}",
        path,
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    labels_dir = resolve(args.labels_dir)
    output_dir = resolve(args.output_dir)
    previews_dir = output_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(labels_dir / "labels_index.csv")
    if args.limit is not None:
        labels = labels.head(int(args.limit)).reset_index(drop=True)

    checkpoint = torch.load(resolve(args.checkpoint_path), map_location="cpu", weights_only=False)
    model_type = checkpoint.get("model_type", "functional_unet_small")
    model = build_segmentation_model(model_type)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = resolve_device(args.device)
    model.to(device).eval()

    dataset = FunctionalSurfaceLabelDataset(labels, int(args.input_size), augmentation_profile=None)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate)
    rows = []
    preview_index = 0
    offset = 0
    for batch in tqdm(loader, desc="eval functional surface labels"):
        images = batch["image"].to(device)
        logits = mask_logits_from_model_output(model(images))
        probs = torch.sigmoid(logits)[:, 0].cpu().numpy()
        pseudo = batch["pseudo"][:, 0].cpu().numpy()
        positives = batch["positive"][:, 0].cpu().numpy()
        negatives = batch["negative"][:, 0].cpu().numpy()
        for item_idx in range(probs.shape[0]):
            row = labels.iloc[offset + item_idx].to_dict()
            prob = probs[item_idx]
            target = pseudo[item_idx]
            metrics = binary_metrics(prob, target, float(args.threshold))
            rows.append(
                {
                    "row_index": int(offset + item_idx),
                    "image_index": int(row.get("image_index", offset + item_idx)),
                    "image_path": row["image_path"],
                    "subpattern": row.get("subpattern", ""),
                    "meta_pattern": row.get("meta_pattern", ""),
                    "crop_kind": row.get("crop_kind", ""),
                    "crop_scale": row.get("crop_scale", ""),
                    **metrics,
                }
            )
            if preview_index < int(args.preview_count):
                write_preview(
                    row,
                    prob,
                    target,
                    positives[item_idx],
                    negatives[item_idx],
                    metrics,
                    int(args.input_size),
                    float(args.threshold),
                    previews_dir / f"{preview_index:04d}_{safe_stem(str(row['image_path']))}.png",
                )
                preview_index += 1
        offset += probs.shape[0]

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "crop_eval_metrics.csv", index=False)

    worst_preview_count = int(args.worst_preview_count)
    if worst_preview_count > 0 and len(metrics_df) > 0:
        worst_dir = output_dir / "worst_previews"
        worst_dir.mkdir(parents=True, exist_ok=True)
        worst_rows = metrics_df.sort_values("iou", ascending=True).head(worst_preview_count)
        for preview_rank, metric_row in enumerate(tqdm(worst_rows.itertuples(index=False), total=len(worst_rows), desc="write worst previews")):
            row_index = int(metric_row.row_index)
            sample = dataset[row_index]
            image_tensor = sample["image"].unsqueeze(0).to(device)
            prob = torch.sigmoid(mask_logits_from_model_output(model(image_tensor)))[0, 0].cpu().numpy()
            target = sample["pseudo"][0].cpu().numpy()
            positive = sample["positive"][0].cpu().numpy()
            negative = sample["negative"][0].cpu().numpy()
            source_row = labels.iloc[row_index].to_dict()
            metrics = binary_metrics(prob, target, float(args.threshold))
            write_preview(
                source_row,
                prob,
                target,
                positive,
                negative,
                metrics,
                int(args.input_size),
                float(args.threshold),
                worst_dir / f"{preview_rank:04d}_row{row_index:04d}_iou{metrics['iou']:.3f}_{safe_stem(str(source_row['image_path']))}.png",
            )

    summary = {
        "labels_dir": rel(labels_dir),
        "checkpoint_path": rel(resolve(args.checkpoint_path)),
        "output_dir": rel(output_dir),
        "rows": int(len(metrics_df)),
        "input_size": int(args.input_size),
        "threshold": float(args.threshold),
        "iou_mean": float(metrics_df["iou"].mean()),
        "dice_mean": float(metrics_df["dice"].mean()),
        "mean_abs_error": float(metrics_df["mean_abs_error"].mean()),
        "pred_ratio_mean": float(metrics_df["pred_ratio"].mean()),
        "target_ratio_mean": float(metrics_df["target_ratio"].mean()),
        "false_positive_ratio_mean": float(metrics_df["false_positive_ratio"].mean()),
        "false_negative_ratio_mean": float(metrics_df["false_negative_ratio"].mean()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def evaluate_checkpoint_from_args(args: argparse.Namespace) -> None:
    """Run checkpoint evaluation from an already parsed argparse namespace."""
    global parse_args
    original_parse_args = parse_args
    parse_args = lambda: args  # type: ignore[assignment]
    try:
        main()
    finally:
        parse_args = original_parse_args


if __name__ == "__main__":
    main()






