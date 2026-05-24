"""Build a real-defect patch library for synthetic denoising augmentation.

The output JSON stores connected components from defective masks. The trainer
can then paste these real defect appearances onto functional surfaces while
keeping the semantic target unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.config import PATHS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Casting defect patch stats.")
    parser.add_argument(
        "--category",
        default="Casting_class1",
        help="Single category or comma-separated categories, e.g. Casting_class1,Casting_class2,Casting_class3.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional image directory for a single category. Defaults to data/raw/hss-iad/<category>/test/defective.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Optional mask directory for a single category. Defaults to data/raw/hss-iad/<category>/ground_truth/defective.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PATHS.root / "reports" / "tables" / "summary" / "casting_class1_defect_patch_library.json",
    )
    parser.add_argument("--min-area", type=int, default=12)
    parser.add_argument("--max-components-per-mask", type=int, default=12)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PATHS.root / path


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PATHS.root.resolve()))
    except ValueError:
        return str(path)


def parse_categories(raw: str) -> list[str]:
    categories = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not categories:
        raise ValueError("At least one category is required.")
    return categories


def default_image_dir(category: str) -> Path:
    return PATHS.root / "data" / "raw" / "hss-iad" / category / "test" / "defective"


def default_mask_dir(category: str) -> Path:
    return PATHS.root / "data" / "raw" / "hss-iad" / category / "ground_truth" / "defective"


def mask_to_components(mask: np.ndarray, min_area: int, max_components: int) -> list[dict]:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if int(area) < int(min_area):
            continue
        comp_mask = labels == label
        contours, _ = cv2.findContours(comp_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        circularity = float((4.0 * np.pi * float(area)) / max(perimeter * perimeter, 1e-6))
        comps.append(
            {
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area": int(area),
                "width": int(w),
                "height": int(h),
                "cx": float(centroids[label][0]),
                "cy": float(centroids[label][1]),
                "equiv_radius": float(np.sqrt(float(area) / np.pi)),
                "aspect": float(w / max(h, 1)),
                "circularity": circularity,
            }
        )
    comps.sort(key=lambda item: item["area"], reverse=True)
    return comps[: int(max_components)]


def component_intensity_stats(image: np.ndarray, mask: np.ndarray, bbox: list[int]) -> dict:
    x0, y0, x1, y1 = bbox
    h, w = mask.shape[:2]
    pad = max(6, int(max(x1 - x0, y1 - y0) * 0.35))
    rx0, ry0 = max(0, x0 - pad), max(0, y0 - pad)
    rx1, ry1 = min(w, x1 + pad), min(h, y1 + pad)
    region_mask = mask[ry0:ry1, rx0:rx1] > 0
    region = image[ry0:ry1, rx0:rx1].astype(np.float32)
    if region_mask.any():
        fg = region[region_mask]
    else:
        fg = region.reshape(-1, 3)
    ring = cv2.dilate(region_mask.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=2).astype(bool) & ~region_mask
    bg = region[ring] if ring.any() else region.reshape(-1, 3)
    return {
        "fg_rgb_mean": [float(v) for v in fg.mean(axis=0)],
        "fg_rgb_std": [float(v) for v in fg.std(axis=0)],
        "bg_rgb_mean": [float(v) for v in bg.mean(axis=0)],
        "contrast_luma": float(
            (0.299 * bg[:, 0] + 0.587 * bg[:, 1] + 0.114 * bg[:, 2]).mean()
            - (0.299 * fg[:, 0] + 0.587 * fg[:, 1] + 0.114 * fg[:, 2]).mean()
        ),
    }


def main() -> None:
    args = parse_args()
    categories = parse_categories(args.category)
    if len(categories) > 1 and (args.image_dir is not None or args.mask_dir is not None):
        raise ValueError("--image-dir/--mask-dir can only be used with a single category.")
    output_json = resolve(args.output_json)
    components = []
    source_dirs = []
    for category in categories:
        image_dir = resolve(args.image_dir) if args.image_dir is not None else default_image_dir(category)
        mask_dir = resolve(args.mask_dir) if args.mask_dir is not None else default_mask_dir(category)
        source_dirs.append({"category": category, "image_dir": safe_rel(image_dir), "mask_dir": safe_rel(mask_dir)})
        for mask_path in sorted(mask_dir.glob("*_mask.png")):
            stem = mask_path.name.removesuffix("_mask.png")
            image_path = image_dir / f"{stem}.jpg"
            if not image_path.exists():
                image_path = image_dir / f"{stem}.png"
            if not image_path.exists():
                continue
            image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            for component in mask_to_components(mask, args.min_area, args.max_components_per_mask):
                component.update(component_intensity_stats(image, mask, component["bbox"]))
                component["category"] = category
                component["image_path"] = safe_rel(image_path)
                component["mask_path"] = safe_rel(mask_path)
                component["source_stem"] = stem
                components.append(component)

    if not components:
        raise RuntimeError(f"No components extracted for {categories}")
    areas = np.array([item["area"] for item in components], dtype=np.float64)
    radii = np.array([item["equiv_radius"] for item in components], dtype=np.float64)
    contrasts = np.array([item["contrast_luma"] for item in components], dtype=np.float64)
    payload = {
        "category": categories[0] if len(categories) == 1 else categories,
        "source_dirs": source_dirs,
        "min_area": int(args.min_area),
        "components": components,
        "summary": {
            "component_count": int(len(components)),
            "component_count_by_category": {
                category: int(sum(1 for item in components if item.get("category") == category))
                for category in categories
            },
            "area_p10_p50_p90": [float(v) for v in np.percentile(areas, [10, 50, 90])],
            "equiv_radius_p10_p50_p90": [float(v) for v in np.percentile(radii, [10, 50, 90])],
            "contrast_luma_p10_p50_p90": [float(v) for v in np.percentile(contrasts, [10, 50, 90])],
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()





