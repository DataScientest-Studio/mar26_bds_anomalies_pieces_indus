"""Post-processing des heatmaps : foreground mask, morphological cleanup.

Extrait du notebook 06 (transistor improvements) et 12 (hazelnut).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, grey_opening
from skimage.filters import threshold_otsu
from skimage.transform import resize as sk_resize


def foreground_mask_from_paths(
    train_paths: Iterable[str | Path],
    img_size: Tuple[int, int] = (504, 504),
    dilation_iters: int = 10,
    invert_if_majority_bg: bool = True,
) -> np.ndarray:
    """Calcule un masque foreground via Otsu sur l'image moyenne des train good.

    Args:
        train_paths : liste de paths vers les images train good.
        img_size    : (H, W) cible. Les images sont resize avant moyenne.
        dilation_iters : nombre d'itérations de dilation pour marge de sécurité.
        invert_if_majority_bg : inverse le masque si l'objet est sombre sur fond clair.

    Returns:
        Masque binaire (H, W) en float32.
    """
    imgs = []
    for p in train_paths:
        im = Image.open(p).convert("RGB").resize(img_size[::-1], Image.BILINEAR)
        imgs.append(np.array(im, dtype=np.float32) / 255.0)
    mean_img = np.stack(imgs).mean(axis=0)  # (H, W, 3)
    mean_gray = mean_img.mean(axis=-1)
    thr = threshold_otsu(mean_gray)
    fg = (mean_gray > thr).astype(np.float32)
    if invert_if_majority_bg and fg.mean() > 0.5:
        fg = 1.0 - fg
    if dilation_iters > 0:
        fg = binary_dilation(fg.astype(bool), iterations=dilation_iters).astype(np.float32)
    return fg


def apply_foreground_mask(heatmaps: np.ndarray, fg_mask: np.ndarray) -> np.ndarray:
    """Applique le mask à un batch de heatmaps. Resize le mask si nécessaire."""
    h_h, h_w = heatmaps.shape[-2:]
    if fg_mask.shape != (h_h, h_w):
        fg_mask = sk_resize(
            fg_mask, (h_h, h_w), order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.float32)
    if heatmaps.ndim == 2:
        return heatmaps * fg_mask
    return heatmaps * fg_mask[None, :, :]


def morphological_opening(heatmaps: np.ndarray, size: int = 5) -> np.ndarray:
    """Erosion + dilatation. Supprime les pixels isolés haut score."""
    if heatmaps.ndim == 2:
        return grey_opening(heatmaps, size=size)
    return np.stack([grey_opening(h, size=size) for h in heatmaps])
