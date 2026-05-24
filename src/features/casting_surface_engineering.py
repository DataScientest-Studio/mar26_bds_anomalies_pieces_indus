"""Feature engineering helpers for Casting functional-surface masks.

These functions are intentionally classical and inspectable. They expose the
signals we need before deciding any final rule: illumination-corrected gray,
local texture, gradients, directional texture coherence, edge maps and component
statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

from src.features.functional_surface import casting_part_mask, close_mask, dilate, remove_small_components


@dataclass(frozen=True)
class CastingSurfaceParams:
    object_threshold: float = 0.08
    illumination_sigma: float = 35.0
    local_window: int = 21
    canny_low: int = 30
    canny_high: int = 90
    min_component_area: int = 512
    close_radius: int = 5
    light_quantile: float = 0.45
    max_std_quantile: float = 0.78
    max_grad_quantile: float = 0.82
    max_texture_edge_density_quantile: float = 0.70
    min_coherence_quantile: float = 0.35
    score_threshold: float = 0.55
    filter_min_area: int = 8000
    filter_max_texture_edge_density: float = 0.09
    filter_max_std: float = 0.04
    filter_min_score: float = 0.74
    grow_regions: bool = False
    grow_iterations: int = 18
    grow_min_score: float = 0.50
    grow_max_texture_edge_density_quantile: float = 0.78
    grow_max_std_quantile: float = 0.82
    grow_edge_barrier_radius: int = 1
    filled_light_quantile: float = 0.58
    filled_score_threshold: float = 0.58
    filled_close_radius: int = 17
    filled_open_radius: int = 3
    filled_min_area: int = 3000
    filled_hole_dark_quantile: float = 0.22
    filled_hole_dark_threshold: float = 0.12
    filled_hole_dilate_radius: int = 2


def robust01(array: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32)
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0
    return image.mean(axis=2).astype(np.float32)


def illumination_correct(gray: np.ndarray, sigma: float) -> np.ndarray:
    smooth = ndimage.gaussian_filter(gray.astype(np.float32), sigma=float(sigma))
    corrected = gray / np.clip(smooth, 1e-4, None)
    return robust01(corrected)


def clahe_gray(gray: np.ndarray) -> np.ndarray:
    image_u8 = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_u8).astype(np.float32) / 255.0


def local_stats(gray: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    mean = cv2.blur(gray.astype(np.float32), (window, window))
    mean_sq = cv2.blur((gray.astype(np.float32) ** 2), (window, window))
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


def gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def structure_coherence(gray: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    jxx = ndimage.gaussian_filter(gx * gx, sigma=sigma)
    jyy = ndimage.gaussian_filter(gy * gy, sigma=sigma)
    jxy = ndimage.gaussian_filter(gx * gy, sigma=sigma)
    numerator = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy)
    denominator = jxx + jyy + 1e-6
    return np.clip(numerator / denominator, 0.0, 1.0).astype(np.float32)


def edge_map(gray: np.ndarray, low: int, high: int) -> np.ndarray:
    smooth = cv2.GaussianBlur((np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8), (5, 5), 0)
    edges = cv2.Canny(smooth, int(low), int(high))
    return (edges > 0).astype(np.uint8)


def boundary_edge_map(gray: np.ndarray, part: np.ndarray) -> np.ndarray:
    image_u8 = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    smooth = cv2.bilateralFilter(image_u8, d=9, sigmaColor=35, sigmaSpace=9)
    grad = gradient_magnitude(smooth.astype(np.float32) / 255.0)
    valid_grad = grad[part > 0]
    if len(valid_grad) == 0:
        return np.zeros_like(part, dtype=np.uint8)
    high = max(20, int(np.percentile(valid_grad, 93) * 255))
    low = max(5, int(0.45 * high))
    edges = cv2.Canny(smooth, low, high)
    return ((edges > 0) & (part > 0)).astype(np.uint8)


def casting_feature_maps(rgb: np.ndarray, params: CastingSurfaceParams) -> dict[str, np.ndarray]:
    gray = to_gray(rgb)
    corrected = illumination_correct(gray, params.illumination_sigma)
    equalized = clahe_gray(corrected)
    part = casting_part_mask(rgb, threshold=params.object_threshold)
    local_mean, local_std = local_stats(corrected, params.local_window)
    grad = gradient_magnitude(corrected)
    texture_grad = gradient_magnitude(equalized)
    coherence = structure_coherence(equalized)
    boundary_edges = boundary_edge_map(corrected, part)
    texture_edges = edge_map(equalized, params.canny_low, params.canny_high)
    texture_edge_density = cv2.blur(texture_edges.astype(np.float32), (params.local_window, params.local_window))
    valid = part > 0

    smoothness = 1.0 - robust01(local_std)
    low_gradient = 1.0 - robust01(grad)
    lightness = robust01(local_mean)
    score = (
        0.38 * lightness
        + 0.28 * smoothness
        + 0.20 * coherence
        + 0.14 * low_gradient
    ).astype(np.float32)
    score[~valid] = 0.0

    return {
        "gray": gray,
        "corrected": corrected,
        "equalized": equalized,
        "part": part.astype(np.uint8),
        "local_mean": local_mean,
        "local_std": local_std,
        "grad": grad,
        "texture_grad": texture_grad,
        "coherence": coherence,
        "edges": boundary_edges.astype(np.uint8),
        "texture_edges": texture_edges.astype(np.uint8),
        "texture_edge_density": texture_edge_density.astype(np.float32),
        "lightness": lightness,
        "smoothness": smoothness,
        "low_gradient": low_gradient,
        "score": score,
    }


def quantile_candidate(features: dict[str, np.ndarray], params: CastingSurfaceParams) -> np.ndarray:
    part = features["part"] > 0
    if part.sum() == 0:
        return np.zeros_like(features["part"], dtype=np.uint8)

    light = features["local_mean"]
    std = features["local_std"]
    grad = features["grad"]
    texture_edge_density = features["texture_edge_density"]
    coherence = features["coherence"]
    score = features["score"]

    light_thr = float(np.quantile(light[part], params.light_quantile))
    std_thr = float(np.quantile(std[part], params.max_std_quantile))
    grad_thr = float(np.quantile(grad[part], params.max_grad_quantile))
    texture_edge_density_thr = float(np.quantile(texture_edge_density[part], params.max_texture_edge_density_quantile))
    coherence_thr = float(np.quantile(coherence[part], params.min_coherence_quantile))
    candidate = (
        part
        & (score >= float(params.score_threshold))
        & (light >= light_thr)
        & (std <= std_thr)
        & (grad <= grad_thr)
        & (texture_edge_density <= texture_edge_density_thr)
        & (coherence >= coherence_thr)
    ).astype(np.uint8)
    candidate = close_mask(candidate, params.close_radius)
    candidate = remove_small_components(candidate, params.min_component_area)
    return candidate.astype(np.uint8)


def _morph(mask: np.ndarray, op: int, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0).astype(np.uint8)
    size = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return (cv2.morphologyEx((mask > 0).astype(np.uint8), op, kernel) > 0).astype(np.uint8)


def filled_surface_candidate(features: dict[str, np.ndarray], params: CastingSurfaceParams) -> np.ndarray:
    """Build reference-like filled functional surfaces with dark holes removed."""
    part = features["part"] > 0
    if part.sum() == 0:
        return np.zeros_like(features["part"], dtype=np.uint8)

    corrected_u8 = (np.clip(features["corrected"], 0.0, 1.0) * 255).astype(np.uint8)
    threshold, _ = cv2.threshold(corrected_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    seed = (corrected_u8 >= float(threshold)).astype(np.uint8)

    seed = _morph(seed, cv2.MORPH_OPEN, int(params.filled_open_radius))
    seed = _morph(seed, cv2.MORPH_CLOSE, int(params.filled_close_radius))
    seed = ndimage.binary_fill_holes(seed > 0).astype(np.uint8)
    seed = remove_small_components(seed, int(params.filled_min_area))

    dark_thr = float(params.filled_hole_dark_threshold)
    dark = (features["gray"] <= dark_thr).astype(np.uint8)
    dark = dilate(dark, int(params.filled_hole_dilate_radius))
    filled = ((seed > 0) & ~(dark > 0)).astype(np.uint8)
    filled = _morph(filled, cv2.MORPH_CLOSE, max(1, int(params.filled_open_radius)))
    filled = remove_small_components(filled, int(params.filled_min_area))
    return filled.astype(np.uint8)


def contour_mask_from_edges(edges: np.ndarray, part: np.ndarray, close_radius: int = 3) -> np.ndarray:
    contour = close_mask((edges > 0).astype(np.uint8), close_radius)
    contour = ((contour > 0) & (part > 0)).astype(np.uint8)
    return contour


def component_table(mask: np.ndarray, features: dict[str, np.ndarray]) -> pd.DataFrame:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rows = []
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        contour_u8 = component.astype(np.uint8)
        contours, _ = cv2.findContours(contour_u8 * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = float(sum(cv2.contourArea(c) for c in contours))
        hull_area = 0.0
        for contour in contours:
            hull_area += float(cv2.contourArea(cv2.convexHull(contour)))
        rows.append(
            {
                "component": label,
                "area": area,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "bbox_ratio": float(w / max(h, 1)),
                "fill_ratio": float(area / max(w * h, 1)),
                "solidity": float(contour_area / max(hull_area, 1.0)),
                "mean_gray": float(features["gray"][component].mean()),
                "mean_corrected": float(features["corrected"][component].mean()),
                "mean_std": float(features["local_std"][component].mean()),
                "mean_grad": float(features["grad"][component].mean()),
                "mean_texture_edge_density": float(features["texture_edge_density"][component].mean()),
                "mean_coherence": float(features["coherence"][component].mean()),
                "mean_score": float(features["score"][component].mean()),
            }
        )
    columns = [
        "component",
        "area",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "bbox_ratio",
        "fill_ratio",
        "solidity",
        "mean_gray",
        "mean_corrected",
        "mean_std",
        "mean_grad",
        "mean_texture_edge_density",
        "mean_coherence",
        "mean_score",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("area", ascending=False).reset_index(drop=True)


def filter_candidate_components(mask: np.ndarray, features: dict[str, np.ndarray], params: CastingSurfaceParams) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    keep = np.zeros_like(binary, dtype=np.uint8)
    table = component_table(binary, features)
    for _, row in table.iterrows():
        if int(row["area"]) < int(params.filter_min_area):
            continue
        if float(row["mean_texture_edge_density"]) > float(params.filter_max_texture_edge_density):
            continue
        if float(row["mean_std"]) > float(params.filter_max_std):
            continue
        if float(row["mean_score"]) < float(params.filter_min_score):
            continue
        keep[labels == int(row["component"])] = 1
    return keep.astype(np.uint8)


def grow_candidate_regions(seed: np.ndarray, features: dict[str, np.ndarray], params: CastingSurfaceParams) -> np.ndarray:
    part = features["part"] > 0
    if seed.sum() == 0 or part.sum() == 0:
        return (seed > 0).astype(np.uint8)

    texture_density = features["texture_edge_density"]
    local_std = features["local_std"]
    score = features["score"]
    texture_thr = float(np.quantile(texture_density[part], float(params.grow_max_texture_edge_density_quantile)))
    std_thr = float(np.quantile(local_std[part], float(params.grow_max_std_quantile)))
    support = (
        part
        & (score >= float(params.grow_min_score))
        & (texture_density <= texture_thr)
        & (local_std <= std_thr)
    )
    if params.grow_edge_barrier_radius > 0:
        barrier = dilate(features["edges"], int(params.grow_edge_barrier_radius)) > 0
        support &= ~barrier

    grown = (seed > 0) & support
    structure = np.ones((3, 3), dtype=bool)
    for _ in range(max(0, int(params.grow_iterations))):
        expanded = ndimage.binary_dilation(grown, structure=structure) & support
        if np.array_equal(expanded, grown):
            break
        grown = expanded

    grown = ndimage.binary_fill_holes(grown)
    grown = close_mask(grown.astype(np.uint8), max(1, params.close_radius))
    grown = remove_small_components(grown, max(1, params.min_component_area))
    return grown.astype(np.uint8)






