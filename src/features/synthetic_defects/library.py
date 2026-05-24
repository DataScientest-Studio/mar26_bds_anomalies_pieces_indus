"""Shared builders for real-defect libraries and diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.config import PATHS
from src.models.baselines.patchcore import project_path

__all__ = [
    "categories",
    "connected_components",
    "family_from_geometry",
    "perimeter",
    "photometric_metrics",
    "rel",
    "write_json",
]


def categories(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PATHS.root.resolve()))
    except ValueError:
        return str(path)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    output = path if path.is_absolute() else PATHS.root / path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [labels == idx for idx in range(1, count) if int(stats[idx, cv2.CC_STAT_AREA]) > 0]


def perimeter(component: np.ndarray) -> float:
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def family_from_geometry(component: np.ndarray) -> str:
    ys, xs = np.where(component)
    area = len(xs)
    if area <= 0:
        return "irregular"
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    bw, bh = int(x1 - x0), int(y1 - y0)
    aspect = max(bw, bh) / max(float(min(bw, bh)), 1.0)
    circ = 4.0 * math.pi * float(area) / max(perimeter(component) ** 2, 1e-6)
    if area < 70 and max(bw, bh) <= 14:
        return "speckle"
    if aspect >= 2.3:
        return "scratch_like"
    if area >= 250 and circ >= 0.72 and aspect <= 1.55:
        return "machined_round"
    if circ >= 0.50 and aspect <= 2.1:
        return "blob_round"
    return "irregular"


def photometric_metrics(image: np.ndarray, component: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(component.astype(np.uint8), kernel, iterations=8).astype(bool)
    ring = dilated & ~component
    if not ring.any():
        ring = ~component
    fg = gray[component]
    bg = gray[ring]
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    gx = float(np.median(grad_x[ring])) if ring.any() else 0.0
    gy = float(np.median(grad_y[ring])) if ring.any() else 0.0
    ys, xs = np.where(component)
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    orientation = 0.0
    if len(coords) >= 3:
        vals, vecs = np.linalg.eigh(np.cov(coords, rowvar=False))
        vec = vecs[:, int(np.argmax(vals))]
        orientation = float(math.atan2(float(vec[1]), float(vec[0])))
    return {
        "fg_mean": float(fg.mean()) if len(fg) else 0.0,
        "fg_std": float(fg.std()) if len(fg) else 0.0,
        "bg_mean": float(bg.mean()) if len(bg) else 0.0,
        "bg_std": float(bg.std()) if len(bg) else 0.0,
        "contrast_luma": float(bg.mean() - fg.mean()) if len(fg) and len(bg) else 0.0,
        "light_angle_rad": float(math.atan2(gy, gx)),
        "light_grad_strength": float(np.hypot(gx, gy)),
        "component_orientation_rad": orientation,
    }


def load_component_table(path: str | Path, category: str, split: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(project_path(str(path)))
    return df[
        df["category"].isin(categories(category))
        & (df["split"] == split)
        & (df["label"] == label)
    ].copy()


def load_rgb_and_mask(image_path: str | Path, mask_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(Image.open(project_path(str(image_path))).convert("RGB"), dtype=np.uint8)
    mask = np.asarray(Image.open(project_path(str(mask_path))).convert("L"), dtype=np.uint8) > 0
    return image, mask





