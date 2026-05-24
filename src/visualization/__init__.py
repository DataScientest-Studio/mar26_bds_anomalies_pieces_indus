"""Visualization public API."""

from src.visualization.embeddings import plot_2d, reduce_2d, separability_score
from src.visualization.heatmaps import (
    blue_orange_heatmap,
    error_to_heatmap,
    normalize_map,
    overlay_heatmap,
    rgb_array_to_image,
)
from src.visualization.previews import label_panel, make_grid

__all__ = [
    "blue_orange_heatmap",
    "error_to_heatmap",
    "label_panel",
    "make_grid",
    "normalize_map",
    "overlay_heatmap",
    "rgb_array_to_image",
    "plot_2d",
    "reduce_2d",
    "separability_score",
]
