"""Build standard Feature-AE preview panels from a saved predictions.npz file."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.config import PATHS
from src.features.tiling import safe_image_id
from src.visualization.previews import make_roi_preview_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-previews", type=int, default=80)
    parser.add_argument("--sort-by", choices=["score_desc", "original"], default="score_desc")
    parser.add_argument("--preview-score-min-percentile", type=float, default=65.0)
    parser.add_argument("--preview-score-max-percentile", type=float, default=99.5)
    parser.add_argument("--preview-score-gamma", type=float, default=0.8)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PATHS.root / path


def load_rgb(image_path: str, target_shape: tuple[int, int]) -> np.ndarray:
    path = resolve(Path(image_path))
    image = Image.open(path).convert("RGB")
    height, width = target_shape
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def main() -> None:
    args = parse_args()
    predictions_path = resolve(args.predictions_path)
    output_dir = resolve(args.output_dir)
    previews_dir = output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(predictions_path, allow_pickle=True)
    image_paths = np.asarray(data["image_path"]).astype(str)
    score_maps = np.asarray(data["score_maps"], dtype=np.float32)
    masks = np.asarray(data["masks"], dtype=np.float32)
    roi_masks = np.asarray(data["roi_masks"], dtype=np.float32) if "roi_masks" in data.files else None
    image_scores = np.asarray(data["image_score"], dtype=np.float32) if "image_score" in data.files else None
    y_true = np.asarray(data["y_true"], dtype=np.int32) if "y_true" in data.files else None

    order = np.arange(len(image_paths))
    if args.sort_by == "score_desc" and image_scores is not None:
        order = order[np.argsort(-image_scores[order])]
    order = order[: max(0, int(args.max_previews))]

    preview_args = SimpleNamespace(
        preview_score_min_percentile=float(args.preview_score_min_percentile),
        preview_score_max_percentile=float(args.preview_score_max_percentile),
        preview_score_gamma=float(args.preview_score_gamma),
    )

    for out_idx, pred_idx in enumerate(order):
        score_map = score_maps[pred_idx]
        rgb = load_rgb(image_paths[pred_idx], score_map.shape)
        roi = None if roi_masks is None else roi_masks[pred_idx]
        label = "unknown" if y_true is None else ("defective" if int(y_true[pred_idx]) else "normal")
        title = f"{Path(image_paths[pred_idx]).parent.name} | label={label} | score={float(image_scores[pred_idx]) if image_scores is not None else float('nan'):.6f}"
        panel = make_roi_preview_panel(
            rgb,
            score_map,
            masks[pred_idx],
            roi,
            title,
            preview_args,
        )
        panel.save(previews_dir / f"{out_idx:03d}_{safe_image_id(image_paths[pred_idx])}.png")

    print(f"Saved {len(order)} standard previews to: {previews_dir}")


if __name__ == "__main__":
    main()






