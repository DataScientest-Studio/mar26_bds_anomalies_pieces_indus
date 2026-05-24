"""Build a two-class semantic dataset: functional surface and landmarks.

Classes:
- 0: background / non-supervised outside surface
- 1: functional surface with landmarks subtracted
- 2: landmark/exclusion geometry

The input landmark labels are manually corrected landmark masks.  The surface
source should be a full train_good prediction set, typically the most recent
V18 ROI masks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.models.baselines.patchcore import project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Casting_class1 surface+landmark semantic labels.")
    parser.add_argument("--landmark-labels-dir", type=Path, required=True)
    parser.add_argument("--surface-predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--landmark-dilate-radius", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    return project_path(str(path))


def rel(path: Path) -> str:
    root = project_path(".")
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def path_key(path: object) -> str:
    return str(path).replace("\\", "/").lower()


def safe_stem(path: object) -> str:
    return Path(str(path).replace("\\", "/")).stem.replace(" ", "_")


def load_bool(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = Image.open(resolve(path)).convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 127


def save_bool(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def save_u8(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def preview(rgb: Image.Image, surface: np.ndarray, landmark: np.ndarray) -> Image.Image:
    arr = np.asarray(rgb.convert("RGB"), dtype=np.float32) / 255.0
    out = arr.copy()
    surface = surface.astype(bool)
    landmark = landmark.astype(bool)
    # Blue surface, orange landmarks. Landmarks are drawn after surface.
    out[..., 0] *= 1.0 - surface * 0.20
    out[..., 1] = np.maximum(out[..., 1], surface * 0.44)
    out[..., 2] = np.maximum(out[..., 2], surface * 0.78)
    out[..., 0] = np.maximum(out[..., 0], landmark * 1.00)
    out[..., 1] = np.maximum(out[..., 1], landmark * 0.48)
    out[..., 2] *= 1.0 - landmark * 0.60
    return Image.fromarray((np.clip(out, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")


def save_contact_sheet(rows: pd.DataFrame, output_path: Path) -> None:
    thumbs: list[tuple[str, Image.Image]] = []
    for _, row in rows.iterrows():
        image = Image.open(resolve(row["preview_path"])).convert("RGB").resize((220, 220), Image.Resampling.BILINEAR)
        label = (
            f"{row['base_pattern']} idx={int(row['image_index'])} "
            f"s={float(row['surface_ratio']):.2f} l={float(row['landmark_ratio']):.2f}"
        )
        thumbs.append((label, image))
    cols = 5
    tile_w, tile_h = 220, 248
    rows_count = int(np.ceil(len(thumbs) / cols))
    canvas = Image.new("RGB", (cols * tile_w, rows_count * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for idx, (label, image) in enumerate(thumbs):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        draw.text((x + 5, y + 5), label, fill=(0, 0, 0), font=font)
        canvas.paste(image, (x, y + 28))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    landmark_dir = resolve(args.landmark_labels_dir)
    surface_dir = resolve(args.surface_predictions_dir)
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Use --overwrite.")
        shutil.rmtree(output_dir)

    landmark_index = pd.read_csv(landmark_dir / "labels_index.csv")
    surface_summary = pd.read_csv(surface_dir / "prediction_summary.csv")
    surface_by_path = {path_key(row["image_path"]): row for _, row in surface_summary.iterrows()}

    masks_dir = output_dir / "masks"
    previews_dir = output_dir / "previews"
    masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    missing: list[str] = []
    for out_idx, row in landmark_index.reset_index(drop=True).iterrows():
        key = path_key(row["image_path"])
        if key not in surface_by_path:
            missing.append(str(row["image_path"]))
            continue
        surface_row = surface_by_path[key]
        rgb = Image.open(resolve(row["image_path"])).convert("RGB")
        size = rgb.size
        raw_surface = load_bool(surface_row["mask_path"], size)
        raw_landmark = load_bool(row["landmark_ellipse_exclusion_mask_path"], size)
        landmark = dilate(raw_landmark, int(args.landmark_dilate_radius))
        surface = raw_surface & ~landmark
        semantic = np.zeros(surface.shape, dtype=np.uint8)
        semantic[surface] = 1
        semantic[landmark] = 2

        stem = f"{out_idx:03d}_{int(row['image_index']):04d}_{row.get('base_pattern', row.get('pattern_id', 'P'))}_{safe_stem(row['image_path'])}"
        surface_path = masks_dir / f"{stem}_surface.png"
        landmark_path = masks_dir / f"{stem}_landmark.png"
        semantic_path = masks_dir / f"{stem}_semantic_012.png"
        preview_path = previews_dir / f"{stem}_preview.png"
        save_bool(surface, surface_path)
        save_bool(landmark, landmark_path)
        save_u8(semantic, semantic_path)
        preview(rgb, surface, landmark).save(preview_path)

        out_row = row.to_dict()
        out_row.update(
            {
                "task": "surface_landmark_semantic",
                "class_0": "background",
                "class_1": "functional_surface_minus_landmarks",
                "class_2": "landmark_exclusion",
                "surface_mask_path": rel(surface_path),
                "landmark_mask_path": rel(landmark_path),
                "semantic_mask_path": rel(semantic_path),
                "preview_path": rel(preview_path),
                "source_surface_mask_path": str(surface_row["mask_path"]),
                "source_surface_prob_path": str(surface_row.get("prob_path", "")),
                "surface_ratio": float(surface.mean()),
                "landmark_ratio": float(landmark.mean()),
                "overlap_removed_ratio": float((raw_surface & landmark).mean()),
                "raw_surface_ratio": float(raw_surface.mean()),
            }
        )
        rows.append(out_row)

    if missing:
        raise RuntimeError(f"Missing {len(missing)} surface predictions. First missing: {missing[:5]}")

    index = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(output_dir / "labels_index.csv", index=False)
    save_contact_sheet(index, output_dir / "contact_sheet.png")
    params = {
        "landmark_labels_dir": rel(landmark_dir),
        "surface_predictions_dir": rel(surface_dir),
        "output_dir": rel(output_dir),
        "landmark_dilate_radius": int(args.landmark_dilate_radius),
        "rows": int(len(index)),
        "counts_by_pattern": index["base_pattern"].value_counts().sort_index().to_dict(),
        "classes": {
            "0": "background",
            "1": "functional_surface_minus_landmarks",
            "2": "landmark_exclusion",
        },
        "mean_surface_ratio": float(index["surface_ratio"].mean()),
        "mean_landmark_ratio": float(index["landmark_ratio"].mean()),
        "mean_overlap_removed_ratio": float(index["overlap_removed_ratio"].mean()),
    }
    (output_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    print(json.dumps(params, indent=2))


if __name__ == "__main__":
    main()





