"""Ensemble Mean/Max + 4 techniques de calibration de heatmaps.

Extrait des notebooks 08 (ensemble) et 11 (calibration).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import resize as sk_resize


# --- Normalisation -----------------------------------------------------------
def norm_global_minmax(h: np.ndarray) -> np.ndarray:
    """Normalisation global min-max sur toutes les images d'un modèle.

    Crucial pour l'ensemble : préserve les rangs cross-image (vs per-image qui
    force max=1 partout et tue l'AUROC).
    """
    mn, mx = float(h.min()), float(h.max())
    return ((h - mn) / (mx - mn + 1e-8)).astype(np.float32)


# --- Alignement résolutions --------------------------------------------------
def align_heatmaps(h: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Upsample/downsample (N, H, W) → (N, target_H, target_W) via sk_resize bilinéaire."""
    if h.shape[1:] == target_shape:
        return h.astype(np.float32)
    return np.stack(
        [
            sk_resize(x, target_shape, order=1, preserve_range=True, anti_aliasing=True)
            for x in h
        ]
    ).astype(np.float32)


# --- Ensembles ---------------------------------------------------------------
def ensemble_mean(h1: np.ndarray, h2: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Mean ensemble après normalisation global min-max par modèle."""
    if normalize:
        h1 = norm_global_minmax(h1)
        h2 = norm_global_minmax(h2)
    return ((h1 + h2) / 2).astype(np.float32)


def ensemble_max(h1: np.ndarray, h2: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Max ensemble après normalisation global min-max par modèle."""
    if normalize:
        h1 = norm_global_minmax(h1)
        h2 = norm_global_minmax(h2)
    return np.maximum(h1, h2).astype(np.float32)


# --- Calibration (4 techniques — notebook 11) --------------------------------
def calibrate_bg_subtract(
    test: np.ndarray, calib_mean: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """Background subtraction : `h_test - α × mean_calib`, clip à 0."""
    return np.clip(test - alpha * calib_mean, 0.0, None).astype(np.float32)


def calibrate_zscore(
    test: np.ndarray, calib_mean: np.ndarray, calib_std: np.ndarray
) -> np.ndarray:
    """Per-pixel z-score : `(h_test - μ_calib) / σ_calib`. ε ajouté pour stabilité."""
    return ((test - calib_mean) / (calib_std + 1e-6)).astype(np.float32)


def calibrate_p99(test: np.ndarray, calib_p99: np.ndarray) -> np.ndarray:
    """Per-pixel quantile p99 : `h_test / p99_calib`, clip à 0."""
    return np.clip(test / (calib_p99 + 1e-6), 0.0, None).astype(np.float32)


def calibrate_dog(
    test: np.ndarray, sigma1: float = 2, sigma2: float = 10
) -> np.ndarray:
    """Difference of Gaussians : passe-haut spatial, garde les pics locaux.

    Ne nécessite pas de stats calib — fonctionne image par image.
    """
    out = np.stack(
        [gaussian_filter(h, sigma=sigma1) - gaussian_filter(h, sigma=sigma2) for h in test]
    )
    return np.clip(out, 0.0, None).astype(np.float32)


def baseline_stats(calib_heatmaps: np.ndarray) -> dict:
    """Statistiques pour calibration sur un calib set (good unseen).

    Retourne dict avec mean / std / p99 par pixel.
    """
    return {
        "mean": calib_heatmaps.mean(axis=0),
        "std": calib_heatmaps.std(axis=0) + 1e-6,
        "p99": np.quantile(calib_heatmaps, 0.99, axis=0) + 1e-6,
    }
