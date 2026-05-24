"""Shared Feature-AE / RD-AE scoring, calibration, and rescoring helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import PATHS
from src.models.feature_ae.calibration import (
    apply_score_postprocessing,
    build_roi_masks_from_paths,
    calibration_stats,
    image_metrics,
    load_roi_prob_maps,
    parse_layer_weights,
    parse_weights,
    pixel_metrics,
    roi_mask_from_image_path,
    score_image,
    smooth_maps,
    topk_score,
    topk_scores,
)
from src.visualization.heatmaps import overlay_heatmap

__all__ = [
    "apply_display_threshold",
    "apply_score_postprocessing",
    "build_roi_masks",
    "calibrate_layer_maps",
    "calibration_stats",
    "fuse_layer_maps",
    "image_metrics",
    "materialize_calibrated_predictions",
    "compare_heatmap_runs",
    "load_feature_ae_predictions",
    "load_layer_maps",
    "load_roi_prob_maps",
    "parse_layer_weights",
    "parse_weights",
    "pixel_metrics",
    "roi_mask_from_image_path",
    "save_quality_heatmap_previews",
    "score_image",
    "score_images_topk",
    "smooth_maps",
    "topk_score",
    "topk_scores",
]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PATHS.root / path


def load_feature_ae_predictions(path: Path):
    """Load a saved Feature-AE ``predictions.npz`` with pickle enabled."""

    return np.load(resolve(path), allow_pickle=True)


def load_layer_maps(npz, layers: list[str], *, prefer_raw: bool = True) -> dict[str, np.ndarray]:
    """Load per-layer score maps from a Feature-AE prediction NPZ."""

    maps: dict[str, np.ndarray] = {}
    for layer in layers:
        candidates = (
            [f"raw_layer_score_maps_{layer}", f"layer_score_maps_{layer}"]
            if prefer_raw
            else [f"layer_score_maps_{layer}", f"raw_layer_score_maps_{layer}"]
        )
        for key in candidates:
            if key in npz:
                maps[layer] = np.asarray(npz[key], dtype=np.float32)
                break
        else:
            raise KeyError(f"Missing layer maps for {layer}: expected one of {candidates}.")
    return maps


def calibrate_layer_maps(
    npz,
    layer_maps: dict[str, np.ndarray],
    layers: list[str],
    *,
    stat: str = "mean_std",
    epsilon: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Apply saved per-layer calibration maps when present."""

    calibrated: dict[str, np.ndarray] = {}
    for layer in layers:
        if stat == "median_mad":
            center_key = f"calibration_median_map_{layer}"
            spread_key = f"calibration_mad_std_map_{layer}"
        else:
            center_key = f"calibration_mean_map_{layer}"
            spread_key = f"calibration_std_map_{layer}"
        if center_key in npz and spread_key in npz:
            center_map = np.asarray(npz[center_key], dtype=np.float32)
            spread_map = np.asarray(npz[spread_key], dtype=np.float32)
            calibrated[layer] = np.maximum(
                (layer_maps[layer] - center_map[None, ...]) / (spread_map[None, ...] + float(epsilon)),
                0.0,
            ).astype(np.float32, copy=False)
        else:
            if stat == "median_mad":
                raise KeyError(
                    f"Missing robust calibration maps for {layer}: expected {center_key} and {spread_key}."
                )
            calibrated[layer] = np.asarray(layer_maps[layer], dtype=np.float32)
    return calibrated


def fuse_layer_maps(layer_maps: dict[str, np.ndarray], layers: list[str], weights: dict[str, float]) -> np.ndarray:
    fused = None
    for layer in layers:
        current = np.asarray(layer_maps[layer], dtype=np.float32) * float(weights[layer])
        fused = current if fused is None else fused + current
    if fused is None:
        raise ValueError("No layer maps to fuse.")
    return fused.astype(np.float32, copy=False)


def apply_display_threshold(score_maps: np.ndarray, threshold: float) -> np.ndarray:
    """Mask weak display signal without changing upstream score calibration."""

    maps = np.asarray(score_maps, dtype=np.float32)
    return np.where(maps >= float(threshold), maps, 0.0).astype(np.float32, copy=False)


def build_roi_masks(
    predictions: dict,
    args,
) -> np.ndarray | None:
    """Build ROI masks aligned with Feature-AE prediction maps."""

    return build_roi_masks_from_paths(predictions, args)


def score_images_topk(
    score_maps: np.ndarray,
    *,
    roi_masks: np.ndarray | None = None,
    topk_fraction: float = 0.005,
) -> np.ndarray:
    """Compute one image score per map with the project-standard top-k rule."""

    maps = np.asarray(score_maps, dtype=np.float32)
    if roi_masks is None:
        return np.asarray(topk_scores(maps, None, fraction=float(topk_fraction)), dtype=np.float32)
    return np.asarray(
        [topk_score(m, fraction=float(topk_fraction), valid_mask=mask) for m, mask in zip(maps, roi_masks)],
        dtype=np.float32,
    )


def materialize_calibrated_predictions(
    output_dir: Path,
    *,
    score_maps: np.ndarray,
    image_scores: np.ndarray,
    image_paths: list[str] | np.ndarray,
    labels: np.ndarray | None = None,
    masks: np.ndarray | None = None,
    params: dict | None = None,
    metrics: dict | None = None,
) -> Path:
    """Write a compact calibrated Feature-AE prediction bundle."""

    out = resolve(Path(output_dir))
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "score_maps": np.asarray(score_maps, dtype=np.float32),
        "image_scores": np.asarray(image_scores, dtype=np.float32),
        "image_paths": np.asarray(list(image_paths), dtype=object),
    }
    if labels is not None:
        payload["labels"] = np.asarray(labels)
    if masks is not None:
        payload["masks"] = np.asarray(masks)
    np.savez_compressed(out / "predictions.npz", **payload)
    if params is not None:
        (out / "params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    if metrics is not None:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return out / "predictions.npz"


def save_quality_heatmap_previews(
    output_dir: Path,
    *,
    image_paths: list[str] | np.ndarray,
    score_maps: np.ndarray,
    max_previews: int = 32,
    display_threshold: float | None = None,
    alpha: float = 0.55,
) -> list[Path]:
    """Save inspector-facing heatmap overlays from calibrated score maps."""

    from PIL import Image

    out = resolve(Path(output_dir))
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    maps = np.asarray(score_maps, dtype=np.float32)
    if display_threshold is not None:
        maps = apply_display_threshold(maps, float(display_threshold))
    for idx, (image_path, score_map) in enumerate(zip(list(image_paths), maps)):
        if idx >= int(max_previews):
            break
        image = Image.open(resolve(Path(str(image_path)))).convert("RGB")
        overlay = overlay_heatmap(image, score_map, alpha=float(alpha))
        dst = out / f"{idx:03d}_{Path(str(image_path)).stem}_quality_heatmap.png"
        overlay.save(dst)
        saved.append(dst)
    return saved


def compare_heatmap_runs(run_dirs: dict[str, Path] | list[Path]) -> list[dict]:
    """Return compact metadata for calibrated heatmap runs."""

    if isinstance(run_dirs, dict):
        items = run_dirs.items()
    else:
        items = [(Path(p).name, Path(p)) for p in run_dirs]
    rows: list[dict] = []
    for name, run_dir in items:
        path = resolve(Path(run_dir))
        metrics_path = path / "metrics.json"
        predictions_path = path / "predictions.npz"
        metrics = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        n_predictions = None
        if predictions_path.exists():
            with np.load(predictions_path, allow_pickle=True) as npz:
                if "image_scores" in npz:
                    n_predictions = int(len(npz["image_scores"]))
                elif "score_maps" in npz:
                    n_predictions = int(len(npz["score_maps"]))
        rows.append(
            {
                "run": str(name),
                "path": str(path),
                "exists": path.exists(),
                "n_predictions": n_predictions,
                **metrics,
            }
        )
    return rows






