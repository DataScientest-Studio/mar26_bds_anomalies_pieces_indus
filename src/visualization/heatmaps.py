"""Heatmap colorization helpers for inspection previews."""

from __future__ import annotations

import numpy as np
from PIL import Image


def normalize_map(score_map: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(score_map, dtype=np.float32)
    lo = float(np.nanmin(arr)) if arr.size else 0.0
    hi = float(np.nanmax(arr)) if arr.size else 1.0
    return np.clip((arr - lo) / max(hi - lo, eps), 0.0, 1.0)


def rgb_array_to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(array, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")


def error_to_heatmap(error: np.ndarray) -> Image.Image:
    arr = np.asarray(error, dtype=np.float32)
    high = float(np.percentile(arr, 99))
    if high <= 0:
        norm = np.zeros_like(arr, dtype=np.float32)
    else:
        norm = np.clip(arr / high, 0.0, 1.0)
    heatmap = np.zeros((*norm.shape, 3), dtype=np.float32)
    heatmap[..., 0] = norm
    heatmap[..., 1] = np.sqrt(norm) * 0.35
    heatmap[..., 2] = 1.0 - norm
    return rgb_array_to_image(heatmap)


def blue_orange_heatmap(score_map: np.ndarray) -> Image.Image:
    """Map low scores to blue and high scores to yellow/orange."""
    arr = normalize_map(score_map)
    r = np.clip(40 + 235 * arr, 0, 255)
    g = np.clip(90 + 140 * arr, 0, 255)
    b = np.clip(210 * (1.0 - arr) + 25 * arr, 0, 255)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def overlay_heatmap(
    image: Image.Image,
    score_map: np.ndarray,
    *,
    alpha: float = 0.45,
    threshold: float | None = None,
) -> Image.Image:
    base = image.convert("RGB")
    heatmap = blue_orange_heatmap(score_map).resize(base.size, Image.Resampling.BILINEAR)
    if threshold is None:
        return Image.blend(base, heatmap, alpha=float(alpha))
    mask = normalize_map(score_map) >= float(threshold)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(base.size, Image.Resampling.NEAREST)
    blended = Image.blend(base, heatmap, alpha=float(alpha))
    return Image.composite(blended, base, mask_img)





