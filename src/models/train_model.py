"""Navigation module for maintained training entrypoints.

This file is not a CLI. Public commands are exposed through ``pyproject.toml``
entry points such as ``train-roi`` and ``train-rd-ae``.
"""

from __future__ import annotations

from src.models.feature_ae.models import build_feature_autoencoder
from src.models.pixel_ae import build_pixel_autoencoder
from src.models.segmentation.models import build_segmentation_model


PUBLIC_TRAINING_COMMANDS = {
    "segmentation": "uv run train-roi",
    "feature_ae": "uv run train-rd-ae",
    "pixel_ae": "src.models.pixel_ae",
}


__all__ = [
    "PUBLIC_TRAINING_COMMANDS",
    "build_feature_autoencoder",
    "build_pixel_autoencoder",
    "build_segmentation_model",
]


