"""Official feature-building entrypoint.

This module is the stable import point for reusable feature construction.
The command-line scripts remain in `scripts/` and can progressively delegate
here as their internal logic is factored into importable functions.
"""

from __future__ import annotations

from src.features.casting_surface_engineering import CastingSurfaceParams, casting_feature_maps
from src.features.functional_surface import (
    category_dataframe,
    contour_closed_functional_mask,
    load_functional_predictions,
    preview_panel,
    safe_stem,
    unsupervised_kmeans_functional_mask,
)
from src.features.tiling import TileSpec, TiledImage, reconstruct_score_map, tile_image


__all__ = [
    "CastingSurfaceParams",
    "TileSpec",
    "TiledImage",
    "casting_feature_maps",
    "category_dataframe",
    "contour_closed_functional_mask",
    "load_functional_predictions",
    "preview_panel",
    "reconstruct_score_map",
    "safe_stem",
    "tile_image",
    "unsupervised_kmeans_functional_mask",
]
