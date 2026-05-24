"""Reusable preview-panel helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from src.visualization.heatmaps import rgb_array_to_image


def make_grid(
    images: Sequence[Image.Image],
    *,
    columns: int = 4,
    cell_size: tuple[int, int] | None = None,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    if not images:
        raise ValueError("images must not be empty.")
    columns = max(1, int(columns))
    if cell_size is None:
        widths, heights = zip(*(img.size for img in images), strict=False)
        cell_size = (max(widths), max(heights))
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell_size[0], rows * cell_size[1]), background)
    for idx, image in enumerate(images):
        row, col = divmod(idx, columns)
        canvas.paste(image.convert("RGB").resize(cell_size), (col * cell_size[0], row * cell_size[1]))
    return canvas


def label_panel(image: Image.Image, title: str, *, header_height: int = 28) -> Image.Image:
    base = image.convert("RGB")
    panel = Image.new("RGB", (base.width, base.height + header_height), (245, 247, 250))
    panel.paste(base, (0, header_height))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 7), title, fill=(30, 36, 48))
    return panel


def feature_ae_preview_heatmap(score_map: np.ndarray, roi: np.ndarray | None, args) -> Image.Image:
    array = np.asarray(score_map, dtype=np.float32)
    if roi is not None:
        array = array * np.asarray(roi, dtype=np.float32)
    lo = float(np.percentile(array, float(args.preview_score_min_percentile)))
    hi = float(np.percentile(array, float(args.preview_score_max_percentile)))
    norm = np.clip((array - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    norm = np.power(norm, float(args.preview_score_gamma))
    heat = np.zeros((*norm.shape, 3), dtype=np.float32)
    heat[..., 0] = norm
    heat[..., 1] = np.sqrt(norm) * 0.35
    heat[..., 2] = 1.0 - norm
    return rgb_array_to_image(heat)


def feature_ae_overlay_heatmap(
    rgb: np.ndarray,
    score_map: np.ndarray,
    roi: np.ndarray | None,
    args,
) -> Image.Image:
    base = np.asarray(rgb, dtype=np.float32)
    heat = np.asarray(feature_ae_preview_heatmap(score_map, roi, args), dtype=np.float32) / 255.0
    overlay = (0.55 * base + 0.45 * heat).clip(0.0, 1.0)
    return rgb_array_to_image(overlay)


def make_feature_ae_preview_panel(
    rgb: np.ndarray,
    score_map: np.ndarray,
    mask: np.ndarray,
    title: str,
    args,
) -> Image.Image:
    original = rgb_array_to_image(rgb)
    heat = feature_ae_preview_heatmap(score_map, None, args)
    overlay = feature_ae_overlay_heatmap(rgb, score_map, None, args)
    mask_img = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L").convert("RGB")
    panels = [original, heat, overlay, mask_img]
    labels = ["input", "score_map", "overlay", "mask"]
    width, height = original.size
    label_h = 42
    gap = 8
    canvas = Image.new("RGB", (width * len(panels) + gap * (len(panels) - 1), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, panel) in enumerate(zip(labels, panels, strict=True)):
        x = idx * (width + gap)
        canvas.paste(panel, (x, label_h))
        draw.text((x + 4, 8), label, fill=(0, 0, 0))
    draw.text((4, 24), title[:140], fill=(80, 80, 80))
    return canvas


def make_roi_preview_panel(
    rgb: np.ndarray,
    score_map: np.ndarray,
    mask: np.ndarray,
    roi: np.ndarray | None,
    title: str,
    args,
) -> Image.Image:
    if roi is None:
        return make_feature_ae_preview_panel(rgb, score_map, mask, title, args)
    original = rgb_array_to_image(rgb)
    heat = feature_ae_preview_heatmap(score_map, roi, args)
    overlay = feature_ae_overlay_heatmap(rgb, score_map, roi, args)
    mask_img = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L").convert("RGB")
    roi_img = Image.fromarray((np.clip(roi, 0, 1) * 255).astype(np.uint8), mode="L").convert("RGB")
    panels = [original, heat, overlay, mask_img, roi_img]
    labels = ["input", "score_map", "overlay", "mask", "score roi"]
    width, height = original.size
    label_h = 42
    gap = 8
    canvas = Image.new("RGB", (width * len(panels) + gap * (len(panels) - 1), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, panel) in enumerate(zip(labels, panels, strict=True)):
        x = idx * (width + gap)
        canvas.paste(panel, (x, label_h))
        draw.text((x + 4, 8), label, fill=(0, 0, 0))
    draw.text((4, 24), title[:140], fill=(80, 80, 80))
    return canvas





