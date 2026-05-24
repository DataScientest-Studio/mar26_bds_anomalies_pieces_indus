"""Build side-by-side heatmap previews for several Feature-AE evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.features.tiling import safe_image_id
from src.visualization.heatmaps import error_to_heatmap
from src.config import PATHS
from src.models.baselines.patchcore import project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Feature-AE heatmap runs on the same test images.")
    parser.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "PREDICTIONS_NPZ"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--panel-size", type=int, default=256)
    parser.add_argument("--score-min-percentile", type=float, default=80.0)
    parser.add_argument("--score-max-percentile", type=float, default=99.7)
    parser.add_argument("--score-gamma", type=float, default=0.85)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--blue-baseline-alpha", type=float, default=0.32)
    parser.add_argument(
        "--display-threshold",
        type=float,
        default=0.0,
        help="Inspector preview threshold in normalized heatmap units. Values below it are hidden.",
    )
    parser.add_argument("--heatmap-palette", choices=["orange", "blue_orange"], default="orange")
    parser.add_argument("--sheet-cols", type=int, default=1)
    parser.add_argument("--sheet-max-defective", type=int, default=31)
    parser.add_argument("--sheet-max-good", type=int, default=16)
    parser.add_argument(
        "--inspector-mode",
        action="store_true",
        help="Only show the raw image and clean heatmap overlays; hide ROI/GT debug panels.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PATHS.root / path


def load_prediction(path: Path) -> dict:
    data = np.load(resolve(path), allow_pickle=True)
    required = {"image_path", "score_maps", "masks", "roi_masks", "image_score", "y_true"}
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"{path} is missing required keys: {missing}")
    return {key: data[key] for key in data.files}


def normalized_path(path: object) -> str:
    return str(path).replace("\\", "/")


def resize_image(path: object, size: int) -> Image.Image:
    image = Image.open(project_path(str(path))).convert("RGB")
    return image.resize((size, size), Image.Resampling.BILINEAR)


def mask_overlay(raw: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> Image.Image:
    base = np.asarray(raw.convert("RGB"), dtype=np.float32)
    mask_img = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L").resize(raw.size, Image.Resampling.NEAREST)
    active = np.asarray(mask_img) > 0
    overlay = base.copy()
    overlay[active] = (1.0 - alpha) * overlay[active] + alpha * np.asarray(color, dtype=np.float32)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")


def normalize_score(
    score: np.ndarray,
    roi: np.ndarray,
    low: float,
    high: float,
    gamma: float,
    shared_limits: tuple[float, float] | None = None,
) -> np.ndarray:
    arr = np.asarray(score, dtype=np.float32)
    valid = np.asarray(roi) > 0
    values = arr[valid] if valid.any() else arr.reshape(-1)
    if shared_limits is None:
        lo = float(np.percentile(values, low))
        hi = float(np.percentile(values, high))
    else:
        lo, hi = shared_limits
    norm = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if gamma != 1.0:
        norm = norm ** float(gamma)
    return norm.astype(np.float32, copy=False)


def heat_overlay(
    raw: Image.Image,
    score: np.ndarray,
    roi: np.ndarray,
    shared_limits: tuple[float, float],
    args: argparse.Namespace,
) -> Image.Image:
    norm = normalize_score(
        score,
        roi,
        float(args.score_min_percentile),
        float(args.score_max_percentile),
        float(args.score_gamma),
        shared_limits=shared_limits,
    )
    if args.inspector_mode and args.heatmap_palette == "orange":
        heat = np.zeros((*norm.shape, 3), dtype=np.float32)
        heat[..., 0] = 255.0
        heat[..., 1] = np.clip(np.sqrt(norm) * 199.0, 0.0, 199.0)
        heat[..., 2] = 5.0
        heat = np.asarray(Image.fromarray(np.clip(heat, 0, 255).astype(np.uint8), mode="RGB").resize(raw.size, Image.Resampling.BILINEAR), dtype=np.float32)
    elif args.inspector_mode and args.heatmap_palette == "blue_orange":
        blue = np.array([20.0, 105.0, 230.0], dtype=np.float32)
        cyan = np.array([48.0, 170.0, 235.0], dtype=np.float32)
        amber = np.array([255.0, 152.0, 12.0], dtype=np.float32)
        yellow = np.array([255.0, 238.0, 70.0], dtype=np.float32)
        heat = np.zeros((*norm.shape, 3), dtype=np.float32)
        low = norm <= 0.58
        mid = (norm > 0.58) & (norm <= 0.86)
        high = norm > 0.86
        t_low = np.clip(norm / 0.58, 0.0, 1.0)[..., None]
        heat[low] = ((1.0 - t_low[low]) * blue + t_low[low] * cyan)
        t_mid = np.clip((norm - 0.58) / 0.28, 0.0, 1.0)[..., None]
        heat[mid] = ((1.0 - t_mid[mid]) * cyan + t_mid[mid] * amber)
        t_high = np.clip((norm - 0.86) / 0.14, 0.0, 1.0)[..., None]
        heat[high] = ((1.0 - t_high[high]) * amber + t_high[high] * yellow)
        heat = np.asarray(Image.fromarray(np.clip(heat, 0, 255).astype(np.uint8), mode="RGB").resize(raw.size, Image.Resampling.BILINEAR), dtype=np.float32)
    else:
        heat = np.asarray(error_to_heatmap(norm), dtype=np.uint8)
        heat = np.asarray(Image.fromarray(heat, mode="RGB").resize(raw.size, Image.Resampling.BILINEAR), dtype=np.float32)
    base = np.asarray(raw, dtype=np.float32)
    roi_resized = Image.fromarray((np.asarray(roi) > 0).astype(np.uint8) * 255, mode="L").resize(raw.size, Image.Resampling.NEAREST)
    active = np.asarray(roi_resized) > 0
    norm_resized = np.asarray(Image.fromarray((np.clip(norm, 0, 1) * 255).astype(np.uint8), mode="L").resize(raw.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    display_mask = norm_resized >= float(args.display_threshold)
    out = base.copy()
    if args.inspector_mode:
        if args.heatmap_palette == "blue_orange":
            alpha_strength = float(args.blue_baseline_alpha) + (1.0 - float(args.blue_baseline_alpha)) * np.clip(norm_resized, 0.0, 1.0)
            alpha = np.clip(float(args.overlay_alpha) * alpha_strength[..., None], 0.0, 1.0)
        else:
            alpha = np.clip(float(args.overlay_alpha) * norm_resized[..., None], 0.0, 1.0)
        alpha[~active] = 0.0
        alpha[~display_mask] = 0.0
        out = (1.0 - alpha) * base + alpha * heat
    else:
        out[active] = 0.48 * base[active] + 0.52 * heat[active]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def draw_label(image: Image.Image, title: str, subtitle: str = "") -> Image.Image:
    label_h = 44
    canvas = Image.new("RGB", (image.width, image.height + label_h), "white")
    canvas.paste(image, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((5, 5), title[:42], fill=(0, 0, 0), font=font)
    if subtitle:
        draw.text((5, 23), subtitle[:42], fill=(70, 70, 70), font=font)
    return canvas


def make_comparison_panel(index: int, rows: list[dict], predictions: dict[str, dict], args: argparse.Namespace) -> Image.Image:
    reference = rows[0]
    image_path = reference["image_path"]
    raw = resize_image(image_path, int(args.panel_size))
    mask = reference["mask"]
    roi = reference["roi"]
    y_true = int(reference["y_true"])
    all_roi_values: list[np.ndarray] = []
    for row in rows:
        valid = np.asarray(row["roi"]) > 0
        values = np.asarray(row["score"], dtype=np.float32)[valid]
        if len(values):
            all_roi_values.append(values)
    merged = np.concatenate(all_roi_values) if all_roi_values else np.concatenate([row["score"].reshape(-1) for row in rows])
    limits = (
        float(np.percentile(merged, float(args.score_min_percentile))),
        float(np.percentile(merged, float(args.score_max_percentile))),
    )

    panels = [draw_label(raw, "Image", Path(str(image_path)).name)]
    if not args.inspector_mode:
        panels.extend(
            [
                draw_label(mask_overlay(raw, roi, (40, 140, 230), 0.35), "ROI V36/V15", "zone score"),
                draw_label(mask_overlay(raw, mask, (255, 50, 50), 0.60), "GT defect" if y_true else "GT normal", f"y={y_true}"),
            ]
        )
    for row in rows:
        panel = heat_overlay(raw, row["score"], row["roi"], limits, args)
        title = row["label"] if not args.inspector_mode else f"Heatmap superposee - {row['label']}"
        subtitle = f"score={row['image_score']:.3f}" if not args.inspector_mode else ""
        panels.append(draw_label(panel, title, subtitle))

    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    return canvas


def make_contact_sheet(images: list[Image.Image], output_path: Path, cols: int) -> None:
    if not images:
        return
    cols = max(1, int(cols))
    rows = int(np.ceil(len(images) / cols))
    width = max(img.width for img in images)
    height = max(img.height for img in images)
    sheet = Image.new("RGB", (cols * width, rows * height), "white")
    for idx, image in enumerate(images):
        x = (idx % cols) * width
        y = (idx // cols) * height
        sheet.paste(image, (x, y))
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    preview_dir = output_dir / "per_image"
    preview_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, dict] = {label: load_prediction(Path(path)) for label, path in args.run}
    labels = [label for label, _path in args.run]
    first = loaded[labels[0]]
    path_order = [normalized_path(path) for path in first["image_path"]]
    indexes_by_run: dict[str, dict[str, int]] = {
        label: {normalized_path(path): idx for idx, path in enumerate(data["image_path"])}
        for label, data in loaded.items()
    }

    if args.max_items is not None:
        path_order = path_order[: int(args.max_items)]

    summary_rows = []
    defective_panels: list[Image.Image] = []
    good_panels: list[Image.Image] = []
    for out_index, path_key in enumerate(path_order):
        rows = []
        for label in labels:
            idx = indexes_by_run[label][path_key]
            data = loaded[label]
            rows.append(
                {
                    "label": label,
                    "image_path": data["image_path"][idx],
                    "score": np.asarray(data["score_maps"][idx], dtype=np.float32),
                    "mask": np.asarray(data["masks"][idx]),
                    "roi": np.asarray(data["roi_masks"][idx], dtype=np.float32),
                    "image_score": float(data["image_score"][idx]),
                    "y_true": int(data["y_true"][idx]),
                }
            )
        panel = make_comparison_panel(out_index, rows, loaded, args)
        image_path = rows[0]["image_path"]
        out_name = f"{out_index:03d}_{safe_image_id(str(image_path))}.png"
        panel.save(preview_dir / out_name)
        y_true = int(rows[0]["y_true"])
        if y_true and len(defective_panels) < int(args.sheet_max_defective):
            defective_panels.append(panel)
        if not y_true and len(good_panels) < int(args.sheet_max_good):
            good_panels.append(panel)
        row = {
            "image_path": str(image_path),
            "y_true": y_true,
            "preview_path": str(preview_dir / out_name),
        }
        for item in rows:
            row[f"{item['label']}_image_score"] = item["image_score"]
        summary_rows.append(row)

    make_contact_sheet(defective_panels, output_dir / "comparison_defective.png", int(args.sheet_cols))
    make_contact_sheet(good_panels, output_dir / "comparison_good.png", int(args.sheet_cols))
    pd.DataFrame(summary_rows).to_csv(output_dir / "comparison_index.csv", index=False)

    metrics = {}
    for label, data in loaded.items():
        raw_metrics = data.get("metrics")
        if raw_metrics is not None:
            try:
                metrics[label] = raw_metrics.item()
            except AttributeError:
                metrics[label] = str(raw_metrics)
    (output_dir / "params.json").write_text(json.dumps({"runs": args.run, "metrics": metrics}, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "previews": len(summary_rows)}, indent=2))


if __name__ == "__main__":
    main()






