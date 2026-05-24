"""Utilities for functional-surface masks used before anomaly scoring."""

from __future__ import annotations

import hashlib
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage
from sklearn.cluster import KMeans

from src.config import PATHS
from src.models.baselines.patchcore import project_path


def safe_stem(path: str) -> str:
    raw = str(path).replace("\\", "/")
    stem = Path(raw).stem
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}"


def load_rgb(path: str | Path) -> np.ndarray:
    image = Image.open(project_path(str(path))).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    mask = Image.open(project_path(str(path))).convert("L")
    if size is not None and mask.size != size:
        mask = mask.resize(size, resample=Image.Resampling.NEAREST)
    return (np.asarray(mask) > 127).astype(np.uint8)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L").save(path)


def exterior_background_mask(rgb: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32)
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0
    gray = image.mean(axis=2)
    candidate = gray <= float(threshold)
    height, width = candidate.shape
    exterior = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        if candidate[0, x]:
            queue.append((0, x))
        if candidate[height - 1, x]:
            queue.append((height - 1, x))
    for y in range(height):
        if candidate[y, 0]:
            queue.append((y, 0))
        if candidate[y, width - 1]:
            queue.append((y, width - 1))
    while queue:
        y, x = queue.popleft()
        if exterior[y, x] or not candidate[y, x]:
            continue
        exterior[y, x] = True
        if y > 0:
            queue.append((y - 1, x))
        if y + 1 < height:
            queue.append((y + 1, x))
        if x > 0:
            queue.append((y, x - 1))
        if x + 1 < width:
            queue.append((y, x + 1))
    return exterior


def casting_part_mask(rgb: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    return (~exterior_background_mask(rgb, threshold)).astype(np.uint8)


def disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y <= radius * radius).astype(np.uint8)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0).astype(np.uint8)
    return ndimage.binary_dilation(mask > 0, structure=disk(radius)).astype(np.uint8)


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0).astype(np.uint8)
    return ndimage.binary_erosion(mask > 0, structure=disk(radius)).astype(np.uint8)


def close_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0).astype(np.uint8)
    closed = ndimage.binary_closing(mask > 0, structure=disk(radius))
    return closed.astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    binary = mask > 0
    if min_area <= 1:
        return binary.astype(np.uint8)
    labels, count = ndimage.label(binary)
    if count == 0:
        return binary.astype(np.uint8)
    areas = np.bincount(labels.reshape(-1))
    keep = np.zeros_like(areas, dtype=bool)
    keep[areas >= int(min_area)] = True
    keep[0] = False
    return keep[labels].astype(np.uint8)


def rgb_to_lab_hsv_features(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab[..., 0] /= 255.0
    lab[..., 1:] = (lab[..., 1:] - 128.0) / 128.0
    hsv[..., 0] /= 179.0
    hsv[..., 1:] /= 255.0
    return lab, hsv


def texture_features(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = (np.asarray(rgb, dtype=np.float32).mean(axis=2) * 255).astype(np.uint8)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    grad = grad / max(float(np.percentile(grad, 99)), 1e-6)
    mean = cv2.blur(gray.astype(np.float32) / 255.0, (7, 7))
    mean_sq = cv2.blur((gray.astype(np.float32) / 255.0) ** 2, (7, 7))
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    return grad.astype(np.float32), std.astype(np.float32)


def metal_feature_stack(rgb: np.ndarray, include_xy: bool = False) -> np.ndarray:
    lab, hsv = rgb_to_lab_hsv_features(rgb)
    grad, local_std = texture_features(rgb)
    features = [
        lab[..., 0],
        lab[..., 1],
        lab[..., 2],
        hsv[..., 1],
        hsv[..., 2],
        grad,
        local_std,
    ]
    if include_xy:
        height, width = lab.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        features.extend([0.15 * xx / max(width - 1, 1), 0.15 * yy / max(height - 1, 1)])
    return np.stack(features, axis=-1).astype(np.float32)


def weak_gt_ring_functional_masks(
    rgb: np.ndarray,
    defect_mask: np.ndarray,
    *,
    object_threshold: float,
    defect_dilate_inner: int,
    defect_dilate_outer: int,
    metal_similarity_threshold: float,
    min_component_area: int,
    closing_radius: int,
) -> dict[str, np.ndarray]:
    part = casting_part_mask(rgb, object_threshold)
    defect = (defect_mask > 0).astype(np.uint8)
    outer = dilate(defect, defect_dilate_outer)
    inner = dilate(defect, defect_dilate_inner)
    ring = ((outer > 0) & (inner == 0) & (part > 0)).astype(np.uint8)

    features = metal_feature_stack(rgb, include_xy=False)
    ring_values = features[ring > 0]
    if len(ring_values) == 0:
        pseudo = np.zeros(part.shape, dtype=np.uint8)
    else:
        center = np.median(ring_values, axis=0)
        distance = np.sqrt(np.mean((features - center[None, None, :]) ** 2, axis=-1))
        pseudo = ((distance <= float(metal_similarity_threshold)) & (part > 0)).astype(np.uint8)
    pseudo = close_mask(remove_small_components(pseudo, min_component_area), closing_radius)
    positive = np.maximum(pseudo, ring).astype(np.uint8)
    negative = ((part == 0) | ((rgb.mean(axis=2) < object_threshold * 1.5) & (positive == 0))).astype(np.uint8)
    ignore = ((positive == 0) & (negative == 0)).astype(np.uint8)
    return {
        "positive": positive,
        "negative": negative,
        "ignore": ignore,
        "pseudo": positive,
        "ring": ring,
        "part": part,
    }


def unsupervised_kmeans_functional_mask(
    rgb: np.ndarray,
    *,
    clusters: int,
    object_threshold: float,
    min_component_area: int,
    closing_radius: int,
    include_xy: bool = False,
    max_cluster_samples: int = 100_000,
) -> np.ndarray:
    part = casting_part_mask(rgb, object_threshold)
    features = metal_feature_stack(rgb, include_xy=include_xy)
    valid = part > 0
    if valid.sum() == 0:
        return np.zeros(part.shape, dtype=np.float32)
    sample = features[valid]
    if len(sample) > int(max_cluster_samples):
        rng = np.random.default_rng(42)
        sample = sample[rng.choice(len(sample), size=int(max_cluster_samples), replace=False)]
    kmeans = KMeans(n_clusters=int(clusters), random_state=42, n_init=10)
    kmeans.fit(sample)
    all_valid = features[valid]
    labels_valid = np.empty(len(all_valid), dtype=np.int32)
    chunk = 250_000
    for start in range(0, len(all_valid), chunk):
        labels_valid[start : start + chunk] = kmeans.predict(all_valid[start : start + chunk])
    labels = np.full(part.shape, -1, dtype=np.int32)
    labels[valid] = labels_valid
    lab, hsv = rgb_to_lab_hsv_features(rgb)
    best_label = 0
    best_score = -1e9
    for label in range(int(clusters)):
        cluster_mask = labels == label
        area = int(cluster_mask.sum())
        if area < int(min_component_area):
            continue
        lightness = float(lab[..., 0][cluster_mask].mean())
        saturation = float(hsv[..., 1][cluster_mask].mean())
        grad, local_std = texture_features(rgb)
        smoothness = 1.0 - float(local_std[cluster_mask].mean())
        score = lightness + 0.35 * smoothness - 0.45 * saturation + 0.05 * np.log1p(area)
        if score > best_score:
            best_score = score
            best_label = label
    mask = (labels == best_label).astype(np.uint8)
    mask = close_mask(remove_small_components(mask, min_component_area), closing_radius)
    return mask.astype(np.float32)


def contour_closed_functional_mask(
    rgb: np.ndarray,
    *,
    object_threshold: float,
    min_component_area: int,
    closing_radius: int,
    lightness_quantile: float = 0.58,
    min_lightness: float = 0.25,
    max_saturation: float = 0.55,
    min_value: float = 0.12,
    canny_low: int = 35,
    canny_high: int = 110,
    edge_dilate_radius: int = 1,
    min_edge_support: float = 0.02,
    min_solidity: float = 0.20,
    max_component_area_ratio: float = 0.70,
) -> np.ndarray:
    """Detect bright closed metal surfaces using color candidates plus contour support."""

    part = casting_part_mask(rgb, object_threshold)
    lab, hsv = rgb_to_lab_hsv_features(rgb)
    valid = part > 0
    if valid.sum() == 0:
        return np.zeros(part.shape, dtype=np.float32)

    light_values = lab[..., 0][valid]
    light_threshold = max(float(min_lightness), float(np.quantile(light_values, float(lightness_quantile))))
    candidate = (
        (part > 0)
        & (lab[..., 0] >= light_threshold)
        & (hsv[..., 1] <= float(max_saturation))
        & (hsv[..., 2] >= float(min_value))
    ).astype(np.uint8)
    candidate = close_mask(remove_small_components(candidate, min_component_area), closing_radius)
    candidate = ndimage.binary_fill_holes(candidate > 0).astype(np.uint8)
    candidate = remove_small_components(candidate, min_component_area)

    gray = (np.asarray(rgb, dtype=np.float32).mean(axis=2) * 255).astype(np.uint8)
    edges = cv2.Canny(gray, int(canny_low), int(canny_high))
    if edge_dilate_radius > 0:
        edges = dilate((edges > 0).astype(np.uint8), edge_dilate_radius) * 255
    edges = ((edges > 0) & (part > 0)).astype(np.uint8)

    contours, _hierarchy = cv2.findContours((candidate > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(part.shape, dtype=np.uint8)
    max_area = float(max_component_area_ratio) * float(max(int(part.sum()), 1))
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < float(min_component_area) or area > max_area:
            continue
        hull = cv2.convexHull(contour)
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        solidity = area / hull_area
        if solidity < float(min_solidity):
            continue
        boundary = np.zeros(part.shape, dtype=np.uint8)
        cv2.drawContours(boundary, [contour], -1, 1, thickness=2)
        boundary_count = max(int(boundary.sum()), 1)
        edge_support = float(((boundary > 0) & (edges > 0)).sum()) / float(boundary_count)
        if edge_support < float(min_edge_support):
            continue
        cv2.drawContours(filled, [contour], -1, 1, thickness=-1)

    filled = ((filled > 0) & (part > 0)).astype(np.uint8)
    filled = close_mask(remove_small_components(filled, min_component_area), closing_radius)
    filled = ndimage.binary_fill_holes(filled > 0).astype(np.uint8)
    return filled.astype(np.float32)


def contour_grabcut_functional_mask(
    rgb: np.ndarray,
    *,
    object_threshold: float,
    min_component_area: int,
    closing_radius: int,
    lightness_quantile: float = 0.58,
    min_lightness: float = 0.25,
    max_saturation: float = 0.55,
    min_value: float = 0.12,
    canny_low: int = 35,
    canny_high: int = 110,
    edge_dilate_radius: int = 1,
    min_edge_support: float = 0.02,
    min_solidity: float = 0.20,
    max_component_area_ratio: float = 0.70,
    grabcut_iterations: int = 3,
    sure_fg_erode_radius: int = 5,
    sure_bg_dilate_radius: int = 12,
) -> np.ndarray:
    """Refine closed-surface candidates by snapping boundaries to image gradients with GrabCut."""

    part = casting_part_mask(rgb, object_threshold)
    base = contour_closed_functional_mask(
        rgb,
        object_threshold=object_threshold,
        min_component_area=min_component_area,
        closing_radius=closing_radius,
        lightness_quantile=lightness_quantile,
        min_lightness=min_lightness,
        max_saturation=max_saturation,
        min_value=min_value,
        canny_low=canny_low,
        canny_high=canny_high,
        edge_dilate_radius=edge_dilate_radius,
        min_edge_support=min_edge_support,
        min_solidity=min_solidity,
        max_component_area_ratio=max_component_area_ratio,
    )
    base = (base > 0).astype(np.uint8)
    if base.sum() == 0:
        return base.astype(np.float32)

    image_u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    grab_mask = np.full(base.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    sure_bg = ((part == 0) | (dilate(base, sure_bg_dilate_radius) == 0)).astype(np.uint8)
    prob_fg = ((base > 0) & (sure_bg == 0)).astype(np.uint8)
    sure_fg = erode(prob_fg, sure_fg_erode_radius)
    if sure_fg.sum() == 0:
        sure_fg = prob_fg

    grab_mask[sure_bg > 0] = cv2.GC_BGD
    grab_mask[prob_fg > 0] = cv2.GC_PR_FGD
    grab_mask[sure_fg > 0] = cv2.GC_FGD
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(image_u8, grab_mask, None, bg_model, fg_model, int(grabcut_iterations), cv2.GC_INIT_WITH_MASK)
        refined = np.where((grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except cv2.error:
        refined = base

    refined = ((refined > 0) & (part > 0)).astype(np.uint8)
    refined = close_mask(remove_small_components(refined, min_component_area), closing_radius)
    refined = ndimage.binary_fill_holes(refined > 0).astype(np.uint8)
    return refined.astype(np.float32)


def load_functional_predictions(map_dir: Path) -> dict[str, np.ndarray]:
    npz_path = map_dir / "functional_surface_predictions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing functional surface predictions: {npz_path}")
    loaded = np.load(npz_path, allow_pickle=True)
    return {
        "image_path": loaded["image_path"],
        "prob_maps": loaded["prob_maps"],
        "binary_masks": loaded["binary_masks"],
    }


def functional_map_lookup(map_dir: Path) -> dict[str, np.ndarray]:
    predictions = load_functional_predictions(map_dir)
    return {
        str(path): np.asarray(prob, dtype=np.float32)
        for path, prob in zip(predictions["image_path"], predictions["prob_maps"], strict=True)
    }


def preview_panel(
    panels: list[tuple[str, Image.Image | np.ndarray]],
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared: list[tuple[str, Image.Image]] = []
    for label, panel in panels:
        if isinstance(panel, Image.Image):
            image = panel.convert("RGB")
        else:
            array = np.asarray(panel)
            if array.ndim == 2:
                image = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8), mode="L").convert("RGB")
            else:
                image = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8), mode="RGB")
        prepared.append((label, image))
    width, height = prepared[0][1].size
    gap = 8
    label_h = 44
    canvas = Image.new("RGB", (width * len(prepared) + gap * (len(prepared) - 1), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(prepared):
        x = idx * (width + gap)
        canvas.paste(image.resize((width, height)), (x, label_h))
        draw.text((x + 4, 8), label, fill=(0, 0, 0))
    draw.text((4, 26), title[:180], fill=(80, 80, 80))
    canvas.save(output_path)


def category_dataframe(category: str, split: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(PATHS.unified_csv)
    df = df[df["category"] == category].copy()
    if split is not None:
        df = df[df["split"] == split].copy()
    return df.reset_index(drop=True)





