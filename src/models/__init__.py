"""Modèles d'anomaly detection : Dinomaly, PatchCore manuel, Ensemble."""
from src.models.dinomaly_wrapper import (
    build_dinomaly,
    ckpt_path,
    load_dinomaly_ckpt,
    score_image_manual,
    score_paths_manual,
)
from src.models.ensemble import (
    align_heatmaps,
    baseline_stats,
    calibrate_bg_subtract,
    calibrate_dog,
    calibrate_p99,
    calibrate_zscore,
    ensemble_max,
    ensemble_mean,
    norm_global_minmax,
)
from src.models.patchcore_manual import PatchCoreManual

__all__ = [
    "build_dinomaly",
    "ckpt_path",
    "load_dinomaly_ckpt",
    "score_paths_manual",
    "score_image_manual",
    "PatchCoreManual",
    "norm_global_minmax",
    "align_heatmaps",
    "ensemble_mean",
    "ensemble_max",
    "calibrate_bg_subtract",
    "calibrate_zscore",
    "calibrate_p99",
    "calibrate_dog",
    "baseline_stats",
]
