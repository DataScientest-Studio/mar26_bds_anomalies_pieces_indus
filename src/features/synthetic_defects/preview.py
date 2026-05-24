"""Preview synthetic defect families on all Casting_class1 piece patterns."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

from src.config import PATHS
from src.data.mask_datasets import SemanticSurfaceDataset
from src.models.baselines.patchcore import project_path


DEFAULT_LABELS_DIR = (
    PATHS.root
    / "data"
    / "processed"
    / "functional_surface_curated"
    / "Casting_class1_surface_landmark_semantic_v21_epoch014_full435_exclude5_weightbalanced_v1"
)


FAMILIES = [
    {
        "name": "machined_round",
        "mode": "generic",
        "shape_weights": "machined:1.0",
        "min_radius": 0.012,
        "max_radius": 0.045,
        "max_blobs": 3,
        "texture_strength": 0.55,
        "alpha_min": 0.55,
        "alpha_max": 0.88,
    },
    {
        "name": "scratch_like",
        "mode": "generic",
        "shape_weights": "scratch:1.0",
        "min_radius": 0.008,
        "max_radius": 0.028,
        "max_blobs": 2,
        "texture_strength": 0.45,
        "alpha_min": 0.45,
        "alpha_max": 0.82,
    },
    {
        "name": "speckle",
        "mode": "generic",
        "shape_weights": "hole:1.0",
        "min_radius": 0.004,
        "max_radius": 0.014,
        "max_blobs": 12,
        "texture_strength": 0.35,
        "alpha_min": 0.35,
        "alpha_max": 0.70,
    },
    {
        "name": "soft_stain",
        "mode": "generic",
        "shape_weights": "stain:1.0",
        "min_radius": 0.018,
        "max_radius": 0.060,
        "max_blobs": 2,
        "texture_strength": 0.35,
        "alpha_min": 0.25,
        "alpha_max": 0.62,
    },
    {
        "name": "empirical_residual",
        "mode": "realistic",
        "shape_weights": "hole:1.0",
        "min_radius": 0.012,
        "max_radius": 0.050,
        "max_blobs": 3,
        "texture_strength": 0.65,
        "alpha_min": 0.45,
        "alpha_max": 0.84,
        "render": "residual",
    },
    {
        "name": "mixed_hardening",
        "mode": "mixed",
        "shape_weights": "machined:0.55,scratch:0.20,hole:0.15,stain:0.10",
        "min_radius": 0.012,
        "max_radius": 0.050,
        "max_blobs": 4,
        "texture_strength": 0.65,
        "alpha_min": 0.45,
        "alpha_max": 0.86,
        "render": "residual",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument(
        "--defect-library-json",
        type=Path,
        default=PATHS.root / "reports" / "tables" / "summary" / "casting_all_defect_patch_library.json",
    )
    parser.add_argument(
        "--texture-library-json",
        type=Path,
        default=(
            PATHS.root
            / "reports"
            / "casting_surface_features"
            / "defect_synthetic_study"
            / "clustered_texture_library_casting_all"
            / "clustered_defect_texture_library.json"
        ),
    )
    parser.add_argument(
        "--photometric-library-json",
        type=Path,
        default=(
            PATHS.root
            / "reports"
            / "casting_surface_features"
            / "defect_synthetic_study"
            / "photometric_coherence_library.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.root / "reports" / "casting_surface_features" / "defect_synthetic_study" / "preview_v3_contextual",
    )
    parser.add_argument("--seed", type=int, default=292)
    parser.add_argument("--thumb-size", type=int, default=220)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PATHS.root / path


def choose_pattern_rows(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pattern_id in ["P1", "P2", "P3", "P4"]:
        sub = labels[labels["pattern_id"].astype(str) == pattern_id].copy()
        if sub.empty:
            continue
        sub["score"] = (
            pd.to_numeric(sub["surface_ratio"], errors="coerce").fillna(0.0)
            - 0.25 * pd.to_numeric(sub["landmark_ratio"], errors="coerce").fillna(0.0)
        )
        rows.append(sub.sort_values("score", ascending=False).iloc[0])
    if not rows:
        raise RuntimeError("No P1/P2/P3/P4 rows found in labels_index.csv")
    return pd.DataFrame(rows).reset_index(drop=True)


def make_dataset(row: pd.Series, family: dict, args: argparse.Namespace) -> SemanticSurfaceDataset:
    texture_path = resolve(args.texture_library_json)
    defect_path = resolve(args.defect_library_json)
    photometric_path = resolve(args.photometric_library_json)
    return SemanticSurfaceDataset(
        pd.DataFrame([row]),
        input_size=256,
        semantic_column="semantic_mask_path",
        synthetic_defect_p=1.0,
        synthetic_defect_mode=str(family["mode"]),
        synthetic_defect_realistic_render=str(family.get("render", "residual")),
        synthetic_defect_library_json=defect_path if defect_path.exists() else None,
        synthetic_defect_texture_library_json=texture_path if texture_path.exists() else None,
        synthetic_defect_photometric_library_json=photometric_path if photometric_path.exists() else None,
        synthetic_defect_pattern_aware=True,
        synthetic_defect_p4_large_p=0.80,
        synthetic_defect_max_blobs=int(family["max_blobs"]),
        synthetic_defect_min_radius_frac=float(family["min_radius"]),
        synthetic_defect_max_radius_frac=float(family["max_radius"]),
        synthetic_defect_shape_weights=str(family["shape_weights"]),
        synthetic_defect_texture_strength=float(family["texture_strength"]),
        synthetic_defect_alpha_min=float(family["alpha_min"]),
        synthetic_defect_alpha_max=float(family["alpha_max"]),
        synthetic_defect_bg_match_strength=0.65,
        synthetic_defect_min_surface_overlap=0.90,
    )


def surface_overlay(image: Image.Image, semantic: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    sem = np.asarray(semantic.resize(rgb.size, Image.Resampling.NEAREST), dtype=np.uint8)
    overlay = Image.new("RGB", rgb.size, (0, 0, 0))
    overlay_arr = np.array(overlay, dtype=np.uint8)
    overlay_arr[sem == 1] = (40, 180, 110)
    overlay_arr[sem == 2] = (230, 160, 40)
    overlay = Image.fromarray(overlay_arr, mode="RGB")
    mask = Image.fromarray(((sem > 0).astype(np.uint8) * 95), mode="L").filter(ImageFilter.GaussianBlur(radius=0.7))
    return Image.composite(overlay, rgb, mask)


def diff_panel(base: Image.Image, defect: Image.Image) -> Image.Image:
    diff = ImageChops.difference(base.convert("RGB"), defect.convert("RGB")).convert("L")
    diff = ImageOps.autocontrast(diff)
    return ImageOps.colorize(diff, black=(20, 20, 24), white=(255, 210, 80))


def diff_crop_box(base: Image.Image, defect: Image.Image, margin: int = 96) -> tuple[int, int, int, int] | None:
    diff = np.asarray(ImageChops.difference(base.convert("RGB"), defect.convert("RGB")).convert("L"), dtype=np.uint8)
    ys, xs = np.where(diff > 2)
    if len(xs) < 4:
        return None
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(base.width, int(xs.max()) + 1 + margin)
    y1 = min(base.height, int(ys.max()) + 1 + margin)
    size = max(x1 - x0, y1 - y0, 180)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    x0 = max(0, min(cx - size // 2, base.width - size))
    y0 = max(0, min(cy - size // 2, base.height - size))
    x1 = min(base.width, x0 + size)
    y1 = min(base.height, y0 + size)
    return int(x0), int(y0), int(x1), int(y1)


def labeled_thumb(image: Image.Image, title: str, thumb_size: int) -> Image.Image:
    canvas = Image.new("RGB", (thumb_size, thumb_size + 26), (245, 246, 248))
    thumb = image.convert("RGB")
    thumb.thumbnail((thumb_size, thumb_size), Image.Resampling.BICUBIC)
    canvas.paste(thumb, ((thumb_size - thumb.width) // 2, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((6, thumb_size + 7), title[:42], fill=(25, 30, 35), font=font)
    return canvas


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    labels_path = resolve(args.labels_dir) / "labels_index.csv"
    labels = pd.read_csv(labels_path)
    rows = choose_pattern_rows(labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tile_w = int(args.thumb_size)
    tile_h = int(args.thumb_size) + 26
    columns = 2 + len(FAMILIES) * 2
    sheet = Image.new("RGB", (columns * tile_w, len(rows) * tile_h), (236, 238, 242))
    metadata = {
        "seed": int(args.seed),
        "labels_index": str(labels_path),
        "defect_library_json": str(resolve(args.defect_library_json)),
        "texture_library_json": str(resolve(args.texture_library_json)),
        "families": FAMILIES,
        "rows": [],
    }

    for row_idx, row in rows.iterrows():
        image = Image.open(project_path(str(row["image_path"]))).convert("RGB")
        semantic = Image.open(project_path(str(row["semantic_mask_path"]))).convert("L").resize(
            image.size,
            Image.Resampling.NEAREST,
        )
        base_cells = [
            labeled_thumb(image, f"{row['pattern_id']} clean", tile_w),
            labeled_thumb(surface_overlay(image, semantic), "surface / landmarks", tile_w),
        ]
        for col_idx, cell in enumerate(base_cells):
            sheet.paste(cell, (col_idx * tile_w, row_idx * tile_h))

        metadata["rows"].append(
            {
                "pattern_id": str(row["pattern_id"]),
                "image_path": str(row["image_path"]),
                "semantic_mask_path": str(row["semantic_mask_path"]),
            }
        )
        col_idx = 2
        for family_idx, family in enumerate(FAMILIES):
            random.seed(int(args.seed) + row_idx * 101 + family_idx * 17)
            np.random.seed(int(args.seed) + row_idx * 101 + family_idx * 17)
            dataset = make_dataset(row, family, args)
            defect = dataset._apply_synthetic_defects(image, semantic, pattern_id=str(row["pattern_id"]))
            crop_box = diff_crop_box(image, defect)
            defect_view = defect.crop(crop_box) if crop_box is not None else defect
            delta = diff_panel(image, defect)
            delta_view = delta.crop(crop_box) if crop_box is not None else delta
            sheet.paste(labeled_thumb(defect_view, f"{family['name']} zoom", tile_w), (col_idx * tile_w, row_idx * tile_h))
            col_idx += 1
            sheet.paste(
                labeled_thumb(delta_view, f"delta {family['name']}", tile_w),
                (col_idx * tile_w, row_idx * tile_h),
            )
            col_idx += 1

    preview_path = args.output_dir / "casting_class1_all_patterns_all_defect_families_preview.png"
    metadata_path = args.output_dir / "casting_class1_all_patterns_all_defect_families_preview.json"
    sheet.save(preview_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"preview_path": str(preview_path), "metadata_path": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()






