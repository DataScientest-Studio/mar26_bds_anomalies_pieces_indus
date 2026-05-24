"""Native-ratio tiling helpers for 256px inference windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TileSpec:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class TiledImage:
    tiles: list[Image.Image]
    specs: list[TileSpec]
    original_size: tuple[int, int]
    padded_size: tuple[int, int]
    tile_size: int
    stride: int


def safe_image_id(image_path: str | Path) -> str:
    text = str(image_path).replace("\\", "/")
    return (
        text.replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


def compute_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    padded_length = max(int(length), int(tile_size))
    if padded_length == tile_size:
        return [0]

    starts = list(range(0, padded_length - tile_size + 1, stride))
    last = padded_length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_image(
    image: Image.Image,
    *,
    tile_size: int = 256,
    stride: int = 128,
    fill: int | tuple[int, int, int] = 0,
) -> TiledImage:
    image = image.convert("RGB")
    width, height = image.size
    padded_width = max(width, tile_size)
    padded_height = max(height, tile_size)

    if (padded_width, padded_height) == (width, height):
        padded = image
    else:
        padded = Image.new("RGB", (padded_width, padded_height), fill)
        padded.paste(image, (0, 0))

    tiles: list[Image.Image] = []
    specs: list[TileSpec] = []
    for y0 in compute_starts(padded_height, tile_size, stride):
        for x0 in compute_starts(padded_width, tile_size, stride):
            x1 = x0 + tile_size
            y1 = y0 + tile_size
            tiles.append(padded.crop((x0, y0, x1, y1)))
            specs.append(TileSpec(x0=x0, y0=y0, x1=x1, y1=y1))

    return TiledImage(
        tiles=tiles,
        specs=specs,
        original_size=(width, height),
        padded_size=(padded_width, padded_height),
        tile_size=tile_size,
        stride=stride,
    )


def reconstruct_score_map(
    tile_score_maps: list[np.ndarray] | np.ndarray,
    tiled: TiledImage,
    *,
    aggregation: str = "mean",
) -> np.ndarray:
    if aggregation not in {"mean", "max"}:
        raise ValueError("aggregation must be 'mean' or 'max'.")
    if len(tile_score_maps) != len(tiled.specs):
        raise ValueError("tile_score_maps length must match tiled specs.")

    padded_width, padded_height = tiled.padded_size
    original_width, original_height = tiled.original_size
    if aggregation == "mean":
        canvas = np.zeros((padded_height, padded_width), dtype=np.float32)
        weights = np.zeros((padded_height, padded_width), dtype=np.float32)
        for score_map, spec in zip(tile_score_maps, tiled.specs, strict=True):
            arr = np.asarray(score_map, dtype=np.float32)
            canvas[spec.y0 : spec.y1, spec.x0 : spec.x1] += arr
            weights[spec.y0 : spec.y1, spec.x0 : spec.x1] += 1.0
        canvas = canvas / np.maximum(weights, 1.0)
    else:
        canvas = np.full((padded_height, padded_width), -np.inf, dtype=np.float32)
        for score_map, spec in zip(tile_score_maps, tiled.specs, strict=True):
            arr = np.asarray(score_map, dtype=np.float32)
            view = canvas[spec.y0 : spec.y1, spec.x0 : spec.x1]
            np.maximum(view, arr, out=view)
        canvas[~np.isfinite(canvas)] = 0.0

    return canvas[:original_height, :original_width].astype(np.float32, copy=False)





