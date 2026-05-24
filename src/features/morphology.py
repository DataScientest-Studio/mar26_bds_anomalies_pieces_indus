"""Binary morphology helpers used by surface-mask pipelines."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


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





