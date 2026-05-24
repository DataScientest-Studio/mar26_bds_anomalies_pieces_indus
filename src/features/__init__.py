"""Feature-building public API."""

from src.features.morphology import close_mask, dilate, disk, erode, remove_small_components
from src.features.roi import functional_map_lookup, load_functional_predictions
from src.features.surface_io import load_mask, load_rgb, safe_stem, save_mask
from src.features.tiling import TileSpec, TiledImage, reconstruct_score_map, tile_image
from src.features.transforms import crop_box_to_mask, resize_letterbox_pil, split_df

__all__ = [
    "TileSpec",
    "TiledImage",
    "close_mask",
    "crop_box_to_mask",
    "dilate",
    "disk",
    "erode",
    "functional_map_lookup",
    "load_functional_predictions",
    "load_mask",
    "load_rgb",
    "reconstruct_score_map",
    "remove_small_components",
    "resize_letterbox_pil",
    "safe_stem",
    "save_mask",
    "split_df",
    "tile_image",
]
