"""Image and mask IO helpers for functional-surface workflows."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from src.runtime import project_path


def safe_stem(path: str | Path) -> str:
    raw = str(path).replace("\\", "/")
    stem = Path(raw).stem
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    return f"{stem}_{digest}"


def load_rgb(path: str | Path) -> np.ndarray:
    image = Image.open(project_path(path)).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    mask = Image.open(project_path(path)).convert("L")
    if size is not None and mask.size != size:
        mask = mask.resize(size, resample=Image.Resampling.NEAREST)
    return (np.asarray(mask) > 127).astype(np.uint8)


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    output = project_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L").save(output)





