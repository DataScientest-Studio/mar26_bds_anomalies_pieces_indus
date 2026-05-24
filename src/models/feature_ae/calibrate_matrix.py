"""Expanded post-hoc calibration matrix for saved Feature-AE/RD-AE maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PATHS
from src.models.feature_ae.scoring import (
    fuse_layer_maps,
    image_metrics,
    load_feature_ae_predictions,
    load_layer_maps,
    load_roi_prob_maps,
    parse_weights,
    pixel_metrics,
    smooth_maps,
    topk_scores,
)


DEFAULT_WEIGHTS = [
    "layer2=0.75,layer3=0.25",
    "layer2=0.70,layer3=0.30",
    "layer2=0.65,layer3=0.35",
    "layer2=0.60,layer3=0.40",
    "layer2=0.55,layer3=0.45",
    "layer2=0.50,layer3=0.50",
    "layer2=0.45,layer3=0.55",
    "layer2=0.40,layer3=0.60",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep RD-AE layer fusion, ROI, smoothing and image score calibration.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--roi-predictions-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--layer-weights", nargs="*", default=DEFAULT_WEIGHTS)
    parser.add_argument("--topk-fractions", type=float, nargs="+", default=[0.001, 0.002, 0.005, 0.01, 0.015, 0.02])
    parser.add_argument("--smoothing", nargs="+", choices=["none", "median3", "gaussian"], default=["none", "median3", "gaussian"])
    parser.add_argument(
        "--roi-modes",
        nargs="+",
        choices=["full", "hard_map", "hard_score_only", "soft_map"],
        default=["hard_map", "hard_score_only", "soft_map"],
    )
    parser.add_argument("--roi-thresholds", type=float, nargs="+", default=[0.20, 0.30, 0.40])
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PATHS.root / path


def best_row(rows: list[dict], key: str) -> dict | None:
    candidates = [row for row in rows if key in row and pd.notna(row[key])]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row[key]))


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_feature_ae_predictions(args.predictions)
    image_paths = data["image_path"]
    y_true = np.asarray(data["y_true"], dtype=np.int64)
    masks = np.asarray(data["masks"], dtype=np.uint8)
    fallback_roi = (
        np.asarray(data["roi_masks"], dtype=np.float32)
        if "roi_masks" in data.files
        else np.ones_like(masks, dtype=np.float32)
    )
    roi_prob = load_roi_prob_maps(image_paths, args.roi_predictions_dir, fallback_roi)
    layer_maps = load_layer_maps(data, args.layers, prefer_raw=False)

    rows: list[dict] = []
    for raw_weights in args.layer_weights:
        weights = parse_weights(raw_weights, args.layers)
        fused = fuse_layer_maps(layer_maps, args.layers, weights)
        for smoothing in args.smoothing:
            smoothed = smooth_maps(fused, smoothing, int(args.batch_size))
            for roi_mode in args.roi_modes:
                thresholds = [0.0] if roi_mode == "full" else args.roi_thresholds
                for threshold in thresholds:
                    hard_roi = (roi_prob >= float(threshold)).astype(np.float32) if roi_mode != "full" else None
                    if roi_mode == "full":
                        metric_map = smoothed
                        score_valid = None
                    elif roi_mode == "hard_map":
                        metric_map = smoothed * hard_roi
                        score_valid = hard_roi
                    elif roi_mode == "hard_score_only":
                        metric_map = smoothed
                        score_valid = hard_roi
                    elif roi_mode == "soft_map":
                        metric_map = smoothed * roi_prob
                        score_valid = hard_roi
                    else:
                        raise ValueError(f"Unsupported ROI mode: {roi_mode}")

                    pmetrics = pixel_metrics(y_true, masks, metric_map)
                    for topk in args.topk_fractions:
                        image_score = topk_scores(metric_map, score_valid, float(topk))
                        rows.append(
                            {
                                "layer_weights": raw_weights,
                                **{f"w_{layer}": weights[layer] for layer in args.layers},
                                "smoothing": smoothing,
                                "roi_mode": roi_mode,
                                "roi_threshold": "" if roi_mode == "full" else float(threshold),
                                "topk_fraction": float(topk),
                                **image_metrics(y_true, image_score),
                                **pmetrics,
                            }
                        )
        pd.DataFrame(rows).to_csv(output_dir / "calibration_matrix_partial.csv", index=False)
        print(json.dumps({"layer_weights": raw_weights, "rows": len(rows)}))

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "calibration_matrix.csv", index=False)
    summary = {
        "rows": len(table),
        "best_image_ap": best_row(rows, "image_ap"),
        "best_pixel_ap": best_row(rows, "pixel_ap"),
        "best_aupimo": best_row(rows, "pixel_aupimo_1e-5_1e-3"),
    }
    (output_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()






