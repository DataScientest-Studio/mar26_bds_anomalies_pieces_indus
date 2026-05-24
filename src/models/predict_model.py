"""Navigation module for maintained prediction and evaluation entrypoints.

This file is not a CLI. Public commands are exposed through ``pyproject.toml``
entry points such as ``predict-roi`` and ``evaluate-rd-ae``.
"""

from __future__ import annotations

from src.models.feature_ae.models import feature_error_map
from src.models.segmentation.models import build_segmentation_model
from src.runtime import project_path, resolve_device


PUBLIC_PREDICTION_COMMANDS = {
    "segmentation": "uv run predict-roi",
    "feature_ae_eval": "uv run evaluate-rd-ae",
    "quality_heatmaps": "uv run materialize-quality-heatmaps",
}


__all__ = [
    "PUBLIC_PREDICTION_COMMANDS",
    "build_segmentation_model",
    "feature_error_map",
    "project_path",
    "resolve_device",
]


