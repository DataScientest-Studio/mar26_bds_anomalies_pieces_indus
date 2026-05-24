"""Materialize one calibrated Feature-AE/RD-AE post-hoc configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import PATHS
from src.models.feature_ae.calibration import (
    image_metrics,
    load_roi_prob_maps,
    normalized_path,
    parse_weights,
    pixel_metrics,
    resolve,
    smooth_maps,
    topk_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save predictions.npz for one post-hoc calibration.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--roi-predictions-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--layer-weights", required=True)
    parser.add_argument("--smoothing", choices=["none", "median3", "gaussian"], default="median3")
    parser.add_argument("--roi-mode", choices=["full", "hard_map", "hard_score_only", "soft_map"], default="hard_map")
    parser.add_argument("--roi-threshold", type=float, default=0.30)
    parser.add_argument("--topk-fraction", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(resolve(args.predictions), allow_pickle=True)
    image_paths = data["image_path"]
    y_true = np.asarray(data["y_true"], dtype=np.int64)
    masks = np.asarray(data["masks"], dtype=np.uint8)
    fallback_roi = np.asarray(data["roi_masks"], dtype=np.float32) if "roi_masks" in data.files else np.ones_like(masks, dtype=np.float32)
    roi_prob = load_roi_prob_maps(image_paths, args.roi_predictions_dir, fallback_roi)
    weights = parse_weights(args.layer_weights, args.layers)
    layer_maps = {
        layer: np.asarray(data[f"layer_score_maps_{layer}"], dtype=np.float32)
        for layer in args.layers
    }
    fused = sum(layer_maps[layer] * weights[layer] for layer in args.layers).astype(np.float32, copy=False)
    smoothed = smooth_maps(fused, args.smoothing, int(args.batch_size))

    if args.roi_mode == "full":
        score_maps = smoothed
        roi_masks = np.ones_like(score_maps, dtype=np.float32)
        valid_masks = None
    else:
        hard_roi = (roi_prob >= float(args.roi_threshold)).astype(np.float32)
        roi_masks = hard_roi
        valid_masks = hard_roi
        if args.roi_mode == "hard_map":
            score_maps = smoothed * hard_roi
        elif args.roi_mode == "hard_score_only":
            score_maps = smoothed
        elif args.roi_mode == "soft_map":
            score_maps = smoothed * roi_prob
        else:
            raise ValueError(f"Unsupported ROI mode: {args.roi_mode}")

    image_score = topk_scores(score_maps, valid_masks, float(args.topk_fraction))
    metrics = {
        **image_metrics(y_true, image_score),
        **pixel_metrics(y_true, masks, score_maps),
    }
    payload = {
        "image_path": image_paths,
        "y_true": y_true,
        "image_score": image_score,
        "metrics": np.array([metrics], dtype=object),
        "score_maps": score_maps.astype(np.float32, copy=False),
        "masks": masks,
        "roi_masks": roi_masks.astype(np.float32, copy=False),
        "functional_surface_prob_maps": roi_prob.astype(np.float32, copy=False),
    }
    for layer, maps in layer_maps.items():
        payload[f"layer_score_maps_{layer}"] = maps
    np.savez_compressed(output_dir / "predictions.npz", **payload)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "params.json").write_text(
        json.dumps(
            {
                "source_predictions": str(resolve(args.predictions)),
                "roi_predictions_dir": str(resolve(args.roi_predictions_dir)) if args.roi_predictions_dir else None,
                "layers": args.layers,
                "layer_weights": weights,
                "smoothing": args.smoothing,
                "roi_mode": args.roi_mode,
                "roi_threshold": args.roi_threshold,
                "topk_fraction": args.topk_fraction,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), **metrics}, indent=2))


if __name__ == "__main__":
    main()






