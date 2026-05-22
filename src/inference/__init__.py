"""Pipeline d'inférence haut-niveau pour la démo Streamlit."""
from src.inference.pipeline import AnomalyPipeline, PredictionResult
from src.inference.postproc import (
    apply_foreground_mask,
    foreground_mask_from_paths,
    morphological_opening,
)

__all__ = [
    "AnomalyPipeline",
    "PredictionResult",
    "foreground_mask_from_paths",
    "apply_foreground_mask",
    "morphological_opening",
]
