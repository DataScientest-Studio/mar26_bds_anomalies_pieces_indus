"""Runtime helpers shared by command-line scripts and library modules."""

from __future__ import annotations

from pathlib import Path

import torch

from src.config import PATHS


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def project_path(path: str | Path) -> Path:
    """Resolve project-relative paths while tolerating Windows separators."""
    candidate = Path(str(path).replace("\\", "/"))
    if candidate.is_absolute():
        return candidate
    return PATHS.root / candidate


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve an explicit torch device or choose CUDA when available."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)





