"""Reusable synthetic-defect library and preview helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import PATHS
from src.data.mask_datasets import SemanticSurfaceDataset

DEFAULT_LABELS_DIR = (
    PATHS.root
    / "data"
    / "processed"
    / "functional_surface_curated"
    / "Casting_class1_surface_landmark_semantic_v21_epoch014_full435_exclude5_weightbalanced_v1"
)

FAMILIES = [
    {
        "name": "machined_round",
        "mode": "generic",
        "shape_weights": "machined:1.0",
        "min_radius": 0.012,
        "max_radius": 0.045,
        "max_blobs": 3,
        "texture_strength": 0.55,
        "alpha_min": 0.55,
        "alpha_max": 0.88,
    },
    {
        "name": "scratch_like",
        "mode": "generic",
        "shape_weights": "scratch:1.0",
        "min_radius": 0.008,
        "max_radius": 0.028,
        "max_blobs": 2,
        "texture_strength": 0.45,
        "alpha_min": 0.45,
        "alpha_max": 0.82,
    },
    {
        "name": "speckle",
        "mode": "generic",
        "shape_weights": "hole:1.0",
        "min_radius": 0.004,
        "max_radius": 0.014,
        "max_blobs": 12,
        "texture_strength": 0.35,
        "alpha_min": 0.35,
        "alpha_max": 0.70,
    },
    {
        "name": "soft_stain",
        "mode": "generic",
        "shape_weights": "stain:1.0",
        "min_radius": 0.018,
        "max_radius": 0.060,
        "max_blobs": 2,
        "texture_strength": 0.35,
        "alpha_min": 0.25,
        "alpha_max": 0.62,
    },
    {
        "name": "empirical_residual",
        "mode": "realistic",
        "shape_weights": "hole:1.0",
        "min_radius": 0.012,
        "max_radius": 0.050,
        "max_blobs": 3,
        "texture_strength": 0.65,
        "alpha_min": 0.45,
        "alpha_max": 0.84,
        "render": "residual",
    },
    {
        "name": "mixed_hardening",
        "mode": "mixed",
        "shape_weights": "machined:0.55,scratch:0.20,hole:0.15,stain:0.10",
        "min_radius": 0.012,
        "max_radius": 0.050,
        "max_blobs": 4,
        "texture_strength": 0.65,
        "alpha_min": 0.45,
        "alpha_max": 0.86,
        "render": "residual",
    },
]

__all__ = ["DEFAULT_LABELS_DIR", "FAMILIES", "make_dataset"]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PATHS.root / path


def make_dataset(row: pd.Series, family: dict, args: argparse.Namespace) -> SemanticSurfaceDataset:
    texture_path = _resolve(args.texture_library_json)
    defect_path = _resolve(args.defect_library_json)
    photometric_path = _resolve(args.photometric_library_json)
    return SemanticSurfaceDataset(
        pd.DataFrame([row]),
        input_size=256,
        semantic_column="semantic_mask_path",
        synthetic_defect_p=1.0,
        synthetic_defect_mode=str(family["mode"]),
        synthetic_defect_realistic_render=str(family.get("render", "residual")),
        synthetic_defect_library_json=defect_path if defect_path.exists() else None,
        synthetic_defect_texture_library_json=texture_path if texture_path.exists() else None,
        synthetic_defect_photometric_library_json=photometric_path if photometric_path.exists() else None,
        synthetic_defect_pattern_aware=True,
        synthetic_defect_p4_large_p=0.80,
        synthetic_defect_max_blobs=int(family["max_blobs"]),
        synthetic_defect_min_radius_frac=float(family["min_radius"]),
        synthetic_defect_max_radius_frac=float(family["max_radius"]),
        synthetic_defect_shape_weights=str(family["shape_weights"]),
        synthetic_defect_texture_strength=float(family["texture_strength"]),
        synthetic_defect_alpha_min=float(family["alpha_min"]),
        synthetic_defect_alpha_max=float(family["alpha_max"]),
        synthetic_defect_bg_match_strength=0.65,
        synthetic_defect_min_surface_overlap=0.90,
    )






