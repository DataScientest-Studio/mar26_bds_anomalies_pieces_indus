"""Official visualization entrypoint for previews, overlays and embeddings."""

from __future__ import annotations

from src.features.functional_surface import preview_panel
from src.visualization.embeddings import plot_2d, reduce_2d, separability_score
from src.visualization.heatmaps import blue_orange_heatmap, normalize_map, overlay_heatmap
from src.visualization.previews import label_panel, make_grid


__all__ = [
    "blue_orange_heatmap",
    "label_panel",
    "make_grid",
    "normalize_map",
    "overlay_heatmap",
    "plot_2d",
    "preview_panel",
    "reduce_2d",
    "separability_score",
]
