"""Train a multiclass functional-surface segmenter.

Classes expected in semantic masks:
0 = background
1 = functional surface
2 = landmark/exclusion surface
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.config import DATA, PATHS
from src.features.augmentation_profiles import AugmentationProfile, resolve_augmentation_profile
from src.features.transforms import (
    apply_flat_lighting,
    crop_box_to_mask,
    percentile_normalize_image,
    resize_letterbox_pil,
    resolve_repeat_factor,
    resolve_split_column,
    split_df,
)
from src.features.functional_surface import safe_stem
from src.models.segmentation.models import build_segmentation_model
from src.models.segmentation.checkpointing import save_functional_surface_checkpoint
from src.models.segmentation.metrics import prefixed_metrics
from src.models.segmentation.previews import save_functional_surface_previews
from src.models.segmentation.runtime import replace_segmentation_head
from src.models.segmentation.training_loop import run_epoch
from src.models.baselines.patchcore import IMAGENET_MEAN, IMAGENET_STD, project_path, resolve_device
from src.models.segmentation.config import parse_args


CLASS_NAMES = {
    0: "background",
    1: "functional_surface",
    2: "landmark_exclusion",
}




def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


class SemanticSurfaceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        input_size: int,
        semantic_column: str,
        *,
        context_size: int | None = None,
        augmentation_profile: AugmentationProfile | None = None,
        repeat_factor: int = 1,
        context_crop_prob: float = 1.0,
        positive_crop_prob: float = 0.30,
        train_photometric_normalization_p: float = 0.20,
        train_photo_target_p05: float = 0.03,
        train_photo_target_p95: float = 0.60,
        synthetic_defect_p: float = 0.0,
        synthetic_defect_mode: str = "generic",
        synthetic_defect_realistic_render: str = "paste",
        synthetic_defect_library_json: Path | None = None,
        synthetic_defect_texture_library_json: Path | None = None,
        synthetic_defect_photometric_library_json: Path | None = None,
        synthetic_defect_pattern_aware: bool = False,
        synthetic_defect_p4_large_p: float = 0.75,
        synthetic_defect_max_blobs: int = 5,
        synthetic_defect_min_radius_frac: float = 0.012,
        synthetic_defect_max_radius_frac: float = 0.055,
        synthetic_defect_shape_weights: str = "hole:0.45,scratch:0.35,stain:0.20",
        synthetic_defect_scratch_min_length_frac: float = 0.08,
        synthetic_defect_scratch_max_length_frac: float = 0.45,
        synthetic_defect_scratch_p: float = 0.35,
        synthetic_defect_texture_strength: float = 1.0,
        synthetic_defect_variant_strength: float = 1.0,
        synthetic_defect_large_p: float = 0.0,
        synthetic_defect_large_quantile: float = 0.75,
        synthetic_defect_large_scale_min: float = 1.15,
        synthetic_defect_large_scale_max: float = 2.10,
        synthetic_defect_alpha_min: float = 0.65,
        synthetic_defect_alpha_max: float = 1.0,
        synthetic_defect_bg_match_strength: float = 0.45,
        synthetic_defect_min_surface_overlap: float = 0.80,
        synthetic_defect_context_consistent: bool = False,
        synthetic_defect_crop_localized: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.input_size = int(input_size)
        self.context_size = int(context_size) if context_size is not None else int(input_size)
        self.semantic_column = str(semantic_column)
        self.augmentation_profile = augmentation_profile if augmentation_profile is not None and augmentation_profile.name != "none" else None
        self.repeat_factor = max(1, int(repeat_factor)) if self.augmentation_profile is not None else 1
        self.context_crop_prob = float(context_crop_prob)
        self.positive_crop_prob = float(positive_crop_prob)
        self.train_photometric_normalization_p = float(train_photometric_normalization_p)
        self.train_photo_target_p05 = float(train_photo_target_p05)
        self.train_photo_target_p95 = float(train_photo_target_p95)
        self.synthetic_defect_p = float(synthetic_defect_p)
        self.synthetic_defect_mode = str(synthetic_defect_mode)
        self.synthetic_defect_realistic_render = str(synthetic_defect_realistic_render)
        self.synthetic_defect_library = self._load_defect_library(synthetic_defect_library_json)
        self.synthetic_defect_texture_library = self._load_defect_texture_library(synthetic_defect_texture_library_json)
        self.synthetic_defect_photometric_library = self._load_defect_photometric_library(synthetic_defect_photometric_library_json)
        self.synthetic_defect_pattern_aware = bool(synthetic_defect_pattern_aware)
        self.synthetic_defect_p4_large_p = float(np.clip(synthetic_defect_p4_large_p, 0.0, 1.0))
        self._synthetic_defect_texture_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.synthetic_defect_max_blobs = max(0, int(synthetic_defect_max_blobs))
        self.synthetic_defect_min_radius_frac = float(synthetic_defect_min_radius_frac)
        self.synthetic_defect_max_radius_frac = float(synthetic_defect_max_radius_frac)
        self.synthetic_defect_shape_weights = self._parse_shape_weights(synthetic_defect_shape_weights)
        self.synthetic_defect_scratch_min_length_frac = float(synthetic_defect_scratch_min_length_frac)
        self.synthetic_defect_scratch_max_length_frac = float(synthetic_defect_scratch_max_length_frac)
        self.synthetic_defect_scratch_p = float(synthetic_defect_scratch_p)
        self.synthetic_defect_texture_strength = max(0.0, float(synthetic_defect_texture_strength))
        self.synthetic_defect_variant_strength = max(0.0, float(synthetic_defect_variant_strength))
        self.synthetic_defect_large_p = float(synthetic_defect_large_p)
        self.synthetic_defect_large_quantile = float(np.clip(synthetic_defect_large_quantile, 0.0, 1.0))
        self.synthetic_defect_large_scale_min = float(synthetic_defect_large_scale_min)
        self.synthetic_defect_large_scale_max = float(synthetic_defect_large_scale_max)
        alpha_min = float(np.clip(synthetic_defect_alpha_min, 0.0, 1.0))
        alpha_max = float(np.clip(synthetic_defect_alpha_max, 0.0, 1.0))
        self.synthetic_defect_alpha_min = min(alpha_min, alpha_max)
        self.synthetic_defect_alpha_max = max(alpha_min, alpha_max)
        self.synthetic_defect_bg_match_strength = float(np.clip(synthetic_defect_bg_match_strength, 0.0, 1.0))
        self.synthetic_defect_min_surface_overlap = float(np.clip(synthetic_defect_min_surface_overlap, 0.0, 1.0))
        self.synthetic_defect_context_consistent = bool(synthetic_defect_context_consistent)
        self.synthetic_defect_crop_localized = bool(synthetic_defect_crop_localized)
        self.synthetic_defect_large_library = self._make_large_defect_library()
        if self.semantic_column not in self.df.columns:
            raise KeyError(f"Missing semantic mask column {self.semantic_column!r}.")

    def _load_defect_library(self, path: Path | None) -> list[dict]:
        if path is None:
            return []
        resolved = project_path(str(path))
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        components = payload.get("components", [])
        if not isinstance(components, list):
            raise ValueError(f"Invalid defect library components in {resolved}")
        return components

    def _load_defect_texture_library(self, path: Path | None) -> list[dict]:
        if path is None:
            return []
        resolved = project_path(str(path))
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        textures = payload.get("textures", payload.get("components", []))
        if not isinstance(textures, list):
            raise ValueError(f"Invalid texture library in {resolved}")
        return textures

    def _load_defect_photometric_library(self, path: Path | None) -> list[dict]:
        if path is None:
            return []
        resolved = project_path(str(path))
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        profiles = payload.get("profiles", [])
        if not isinstance(profiles, list):
            raise ValueError(f"Invalid photometric profile library in {resolved}")
        return profiles

    @staticmethod
    def _local_photometry(target: np.ndarray, alpha: np.ndarray) -> dict[str, float]:
        gray = target.mean(axis=2).astype(np.float32)
        active = alpha > 0.08
        if not active.any():
            active = alpha > 0
        ring = (alpha > 0.005) & ~active
        if not ring.any():
            ring = ~active
        grad_y, grad_x = np.gradient(gray)
        gx = float(np.median(grad_x[ring])) if ring.any() else 0.0
        gy = float(np.median(grad_y[ring])) if ring.any() else 0.0
        bg = gray[ring] if ring.any() else gray.reshape(-1)
        return {
            "bg_mean": float(bg.mean()) if len(bg) else float(gray.mean()),
            "bg_std": float(bg.std()) if len(bg) else float(gray.std()),
            "light_angle_rad": float(math.atan2(gy, gx)),
            "light_grad_strength": float(np.hypot(gx, gy)),
        }

    def _sample_photometric_profile(
        self,
        target: np.ndarray,
        alpha: np.ndarray,
        *,
        family: str,
        pattern_id: str = "",
    ) -> dict | None:
        if not self.synthetic_defect_photometric_library:
            return None
        local = self._local_photometry(target, alpha)
        family = str(family).lower()
        if family == "machined":
            families = {"machined_round", "blob_round"}
        elif family == "scratch":
            families = {"scratch_like"}
        elif family == "stain":
            families = {"irregular", "blob_round"}
        elif family == "speckle":
            families = {"speckle", "blob_round"}
        else:
            families = {family, "blob_round", "irregular"}
        candidates = [
            profile for profile in self.synthetic_defect_photometric_library
            if str(profile.get("family", "")).lower() in families
        ]
        if not candidates:
            candidates = self.synthetic_defect_photometric_library
        local_bg = local["bg_mean"]
        local_grad = local["light_grad_strength"]
        def score(profile: dict) -> float:
            bg_score = abs(float(profile.get("bg_mean", local_bg)) - local_bg) / 32.0
            grad_score = abs(math.log1p(float(profile.get("light_grad_strength", local_grad))) - math.log1p(local_grad))
            return bg_score + 0.6 * grad_score + random.random() * 0.05
        ranked = sorted(candidates, key=score)
        profile = dict(random.choice(ranked[: min(12, len(ranked))]))
        profile["target_light_angle_rad"] = local["light_angle_rad"]
        profile["target_bg_mean"] = local["bg_mean"]
        return profile

    @staticmethod
    def _parse_shape_weights(raw: str) -> list[tuple[str, float]]:
        allowed = {"hole", "scratch", "stain", "machined"}
        weights: list[tuple[str, float]] = []
        for part in str(raw).split(","):
            if not part.strip():
                continue
            if ":" in part:
                name, value = part.split(":", 1)
                weight = float(value)
            else:
                name, weight = part, 1.0
            name = name.strip().lower()
            if name not in allowed or weight <= 0:
                continue
            weights.append((name, float(weight)))
        return weights or [("hole", 0.45), ("scratch", 0.35), ("stain", 0.20)]

    def _sample_defect_shape(self) -> str:
        total = sum(weight for _, weight in self.synthetic_defect_shape_weights)
        draw = random.random() * max(total, 1e-6)
        acc = 0.0
        for name, weight in self.synthetic_defect_shape_weights:
            acc += weight
            if draw <= acc:
                return name
        return self.synthetic_defect_shape_weights[-1][0]

    def _pattern_id_from_row(self, row: pd.Series) -> str:
        for column in ("pattern_id", "base_pattern", "meta_pattern"):
            value = row.get(column, "")
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
        return ""

    def _dominant_defect_shape(self) -> str:
        if not self.synthetic_defect_shape_weights:
            return "hole"
        return max(self.synthetic_defect_shape_weights, key=lambda item: item[1])[0]

    def _pattern_defect_profile(self, pattern_id: str, shape: str = "") -> dict[str, float]:
        profile = {
            "radius": 1.0,
            "length": 1.0,
            "count": 1.0,
            "contrast": 1.0,
            "texture": 1.0,
            "alpha": 1.0,
        }
        if not self.synthetic_defect_pattern_aware:
            return profile
        pattern = str(pattern_id).upper()
        shape = str(shape).lower()
        if pattern == "P4":
            profile.update({"radius": 2.75, "length": 1.70, "count": 1.05, "contrast": 3.20, "texture": 3.40, "alpha": 1.12})
            if shape in {"hole", "speckle"}:
                profile.update({"radius": 1.85, "count": 1.20, "contrast": 2.20, "texture": 2.55})
            elif shape == "scratch":
                profile.update({"radius": 2.10, "length": 1.55, "count": 0.90, "contrast": 2.00, "texture": 2.60})
            elif shape == "machined":
                profile.update({"radius": 3.35, "contrast": 4.20, "texture": 4.15, "alpha": 1.18})
        elif pattern == "P3":
            profile.update({"radius": 1.70, "length": 1.25, "count": 1.30, "contrast": 1.70, "texture": 2.00, "alpha": 1.05})
            if shape in {"hole", "speckle"}:
                profile.update({"radius": 1.25, "count": 4.50, "contrast": 1.45, "texture": 1.80})
        elif pattern == "P2":
            profile.update({"radius": 1.55, "length": 1.15, "count": 1.20, "contrast": 1.15, "texture": 1.70, "alpha": 1.00})
            if shape in {"hole", "speckle"}:
                profile.update({"radius": 1.18, "count": 4.30, "contrast": 1.05, "texture": 1.55})
        elif pattern == "P1":
            profile.update({"radius": 1.45, "length": 1.10, "count": 1.15, "contrast": 1.20, "texture": 1.55, "alpha": 1.00})
            if shape in {"hole", "speckle"}:
                profile.update({"radius": 1.15, "count": 3.00, "contrast": 1.15, "texture": 1.45})
        return profile

    def _machined_radius_bounds_for_pattern(self, pattern_id: str, min_radius: int, max_radius: int) -> tuple[int, int]:
        if not self.synthetic_defect_pattern_aware:
            return min_radius, max_radius
        pattern = str(pattern_id).upper()
        profile = self._pattern_defect_profile(pattern, "machined")
        scale_min = max(0.8, profile["radius"] * 0.92)
        scale_max = max(scale_min + 0.10, profile["radius"] * 1.12)
        lo = max(2, int(round(float(min_radius) * scale_min)))
        hi = max(lo + 1, int(round(float(max_radius) * scale_max)))
        return lo, hi

    def _texture_candidates_for_pattern(self, pattern_id: str) -> list[dict]:
        if not self.synthetic_defect_texture_library:
            return []
        if not self.synthetic_defect_pattern_aware:
            return self.synthetic_defect_texture_library
        pattern = str(pattern_id).upper()
        same = [
            item for item in self.synthetic_defect_texture_library
            if str(item.get("pattern_id", "")).upper() == pattern
        ]
        candidates = same if same else self.synthetic_defect_texture_library
        if pattern == "P4" and random.random() < self.synthetic_defect_p4_large_p and len(candidates) > 4:
            areas = np.asarray([float(item.get("area", 0.0)) for item in candidates], dtype=np.float32)
            threshold = float(np.quantile(areas, 0.60))
            large = [item for item in candidates if float(item.get("area", 0.0)) >= threshold]
            if large:
                return large
        return candidates

    def _texture_candidates_for_shape(self, shape: str, pattern_id: str = "") -> list[dict]:
        candidates = self._texture_candidates_for_pattern(pattern_id)
        if not candidates:
            return []
        shape = str(shape).lower()
        if shape == "machined":
            families = {"machined_round"}
        elif shape == "scratch":
            families = {"scratch_like"}
        elif shape == "stain":
            families = {"blob_round", "irregular"}
        else:
            families = {"blob_round", "speckle", "machined_round"}
        filtered = [
            item for item in candidates
            if str(item.get("family", "")).lower() in families
            or any(str(item.get("cluster_label", "")).lower().startswith(family) for family in families)
        ]
        return filtered if filtered else candidates

    def _make_large_defect_library(self) -> list[dict]:
        if not self.synthetic_defect_library:
            return []
        areas = np.asarray([float(item.get("area", 0.0)) for item in self.synthetic_defect_library], dtype=np.float64)
        if len(areas) == 0:
            return []
        threshold = float(np.quantile(areas, self.synthetic_defect_large_quantile))
        large = [item for item in self.synthetic_defect_library if float(item.get("area", 0.0)) >= threshold]
        return large if large else self.synthetic_defect_library

    def __len__(self) -> int:
        return len(self.df) * self.repeat_factor

    def _row(self, index: int) -> pd.Series:
        return self.df.iloc[index % len(self.df)]

    def _to_tensor_image(self, image: Image.Image) -> torch.Tensor:
        tensor = transforms.ToTensor()(image.convert("RGB"))
        return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(tensor)

    def _semantic_tensor(self, mask: Image.Image) -> torch.Tensor:
        arr = np.asarray(mask.convert("L"), dtype=np.int64)
        arr = np.where((arr >= 0) & (arr <= 2), arr, 0).astype(np.int64)
        return torch.from_numpy(arr)

    def _soft_tensor(self, mask: Image.Image) -> torch.Tensor:
        arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr[None, ...])

    def _augment_image(self, image: Image.Image) -> Image.Image:
        image = apply_flat_lighting(image.convert("RGB"), self.augmentation_profile)
        if self.train_photometric_normalization_p > 0 and random.random() < self.train_photometric_normalization_p:
            image = percentile_normalize_image(
                image,
                target_p05=self.train_photo_target_p05,
                target_p95=self.train_photo_target_p95,
            )
        return image

    def _crop_params(self, image: Image.Image, semantic: Image.Image) -> tuple[int, int, int, int]:
        profile = self.augmentation_profile
        assert profile is not None
        if self.positive_crop_prob <= 0 or random.random() >= self.positive_crop_prob:
            return transforms.RandomResizedCrop.get_params(image, scale=profile.scale, ratio=profile.ratio)
        arr = np.asarray(semantic.convert("L"), dtype=np.uint8) > 0
        ys, xs = np.where(arr)
        if len(xs) < 16:
            return transforms.RandomResizedCrop.get_params(image, scale=profile.scale, ratio=profile.ratio)
        width, height = image.size
        area = float(width * height)
        for _ in range(20):
            target_area = area * random.uniform(float(profile.scale[0]), float(profile.scale[1]))
            aspect_ratio = np.exp(random.uniform(np.log(float(profile.ratio[0])), np.log(float(profile.ratio[1]))))
            crop_w = int(round(np.sqrt(target_area * aspect_ratio)))
            crop_h = int(round(np.sqrt(target_area / aspect_ratio)))
            if crop_w <= 0 or crop_h <= 0 or crop_w > width or crop_h > height:
                continue
            point_idx = random.randrange(len(xs))
            center_x = int(xs[point_idx])
            center_y = int(ys[point_idx])
            left = max(0, min(center_x - crop_w // 2 + random.randint(-crop_w // 6, crop_w // 6), width - crop_w))
            top = max(0, min(center_y - crop_h // 2 + random.randint(-crop_h // 6, crop_h // 6), height - crop_h))
            return int(top), int(left), int(crop_h), int(crop_w)
        return transforms.RandomResizedCrop.get_params(image, scale=profile.scale, ratio=profile.ratio)

    def _paste_realistic_defect(
        self,
        image: Image.Image,
        surface: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        pattern_id: str = "",
    ) -> Image.Image:
        if not self.synthetic_defect_library:
            return self._draw_generic_defects(image, xs, ys, pattern_id=pattern_id)
        out = image.convert("RGB")
        width, height = out.size
        use_large = bool(self.synthetic_defect_large_library) and random.random() < self.synthetic_defect_large_p
        component = random.choice(self.synthetic_defect_large_library if use_large else self.synthetic_defect_library)
        image_path = project_path(str(component["image_path"]))
        mask_path = project_path(str(component["mask_path"]))
        x0, y0, x1, y1 = [int(v) for v in component["bbox"]]
        pad = max(4, int(max(x1 - x0, y1 - y0) * random.uniform(0.18, 0.35)))
        source_image = Image.open(image_path).convert("RGB")
        source_mask = Image.open(mask_path).convert("L")
        sx0, sy0 = max(0, x0 - pad), max(0, y0 - pad)
        sx1, sy1 = min(source_image.width, x1 + pad), min(source_image.height, y1 + pad)
        patch = source_image.crop((sx0, sy0, sx1, sy1))
        alpha = source_mask.crop((sx0, sy0, sx1, sy1))
        alpha = alpha.point(lambda value: 255 if value > 0 else 0).filter(ImageFilter.GaussianBlur(radius=1.2))
        strength = self.synthetic_defect_variant_strength
        if use_large:
            scale = random.uniform(self.synthetic_defect_large_scale_min, self.synthetic_defect_large_scale_max)
        else:
            scale = random.uniform(0.75, 1.25)
        aspect_jitter = random.uniform(1.0 - 0.28 * strength, 1.0 + 0.28 * strength)
        new_w = max(4, int(round(patch.width * scale * aspect_jitter)))
        new_h = max(4, int(round(patch.height * scale / max(aspect_jitter, 1e-3))))
        patch = patch.resize((new_w, new_h), Image.Resampling.BILINEAR)
        alpha = alpha.resize((new_w, new_h), Image.Resampling.BILINEAR)
        if random.random() < 0.5 * strength:
            patch = ImageOps.mirror(patch)
            alpha = ImageOps.mirror(alpha)
        if random.random() < 0.2 * strength:
            patch = ImageOps.flip(patch)
            alpha = ImageOps.flip(alpha)
        if random.random() < 0.5:
            angle = random.uniform(-25.0 * max(strength, 0.1), 25.0 * max(strength, 0.1))
            patch = patch.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
            alpha = alpha.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
        if strength > 0:
            if random.random() < 0.35 * strength:
                alpha = alpha.filter(ImageFilter.MaxFilter(size=3))
            if random.random() < 0.25 * strength:
                alpha = alpha.filter(ImageFilter.MinFilter(size=3))
            alpha_arr = np.asarray(alpha, dtype=np.float32)
            noise_w = max(2, alpha.width // 16)
            noise_h = max(2, alpha.height // 16)
            noise = np.random.uniform(0.65, 1.25, size=(noise_h, noise_w)).astype(np.float32)
            noise_img = Image.fromarray(np.uint8(np.clip(noise * 255.0, 0, 255)), mode="L").resize(alpha.size, Image.Resampling.BICUBIC)
            noise_arr = np.asarray(noise_img, dtype=np.float32) / 255.0
            alpha_arr = np.clip(alpha_arr * ((1.0 - 0.45 * strength) + 0.45 * strength * noise_arr), 0, 255)
            alpha = Image.fromarray(alpha_arr.astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(radius=random.uniform(0.6, 1.6)))
        if self.synthetic_defect_realistic_render == "paste" and random.random() < 0.7 * strength:
            patch = ImageEnhance.Contrast(patch).enhance(random.uniform(0.75, 1.35))
        if self.synthetic_defect_realistic_render == "paste" and random.random() < 0.7 * strength:
            patch = ImageEnhance.Brightness(patch).enhance(random.uniform(0.75, 1.20))

        for _ in range(24):
            point_idx = random.randrange(len(xs))
            cx = int(xs[point_idx])
            cy = int(ys[point_idx])
            px0 = int(cx - patch.width // 2)
            py0 = int(cy - patch.height // 2)
            px1 = px0 + patch.width
            py1 = py0 + patch.height
            if px0 < 0 or py0 < 0 or px1 > width or py1 > height:
                continue
            roi_surface = surface[py0:py1, px0:px1]
            alpha_arr = np.asarray(alpha, dtype=np.uint8) > 32
            if (
                alpha_arr.any()
                and float((roi_surface & alpha_arr).sum()) / float(alpha_arr.sum())
                >= self.synthetic_defect_min_surface_overlap
            ):
                break
        else:
            return out

        target = np.asarray(out.crop((px0, py0, px1, py1)), dtype=np.float32)
        alpha_arr_f = np.asarray(alpha, dtype=np.float32) / 255.0
        surface_alpha = roi_surface.astype(np.float32)
        alpha_arr_f = alpha_arr_f * surface_alpha
        family_name = str(component.get("family", "blob_round"))
        pattern_profile = self._pattern_defect_profile(pattern_id, family_name)
        if self.synthetic_defect_realistic_render == "residual":
            alpha_active = alpha_arr_f > 0.08
            if not alpha_active.any():
                return out
            photo_profile = self._sample_photometric_profile(
                target,
                alpha_arr_f,
                family=family_name,
                pattern_id=pattern_id,
            )
            contrast = float((photo_profile or component).get("contrast_luma", component.get("contrast_luma", 0.0)))
            # contrast_luma = bg - fg, therefore the target residual is fg - bg.
            delta = float(np.clip(-contrast, -36.0, 36.0))
            if abs(delta) < 6.0:
                delta = random.choice([-1.0, 1.0]) * random.uniform(4.0, 12.0)
            delta *= random.uniform(0.45, 0.90) * pattern_profile["contrast"]
            fg_std = component.get("fg_luma_std", None)
            if photo_profile is not None:
                fg_std = photo_profile.get("fg_std", fg_std)
            if fg_std is None:
                fg_std = np.mean(component.get("fg_rgb_std", [8.0, 8.0, 8.0]))
            noise_std = (
                float(np.clip(float(fg_std) * 0.45, 1.5, 18.0))
                * self.synthetic_defect_texture_strength
                * pattern_profile["texture"]
            )
            noise = np.random.normal(0.0, noise_std, size=target.shape[:2]).astype(np.float32)
            low_freq_w = max(2, target.shape[1] // 18)
            low_freq_h = max(2, target.shape[0] // 18)
            low_freq = np.random.normal(0.0, noise_std * 0.55, size=(low_freq_h, low_freq_w)).astype(np.float32)
            low_freq_img = Image.fromarray(np.uint8(np.clip(low_freq + 128.0, 0, 255)), mode="L").resize(
                (target.shape[1], target.shape[0]),
                Image.Resampling.BICUBIC,
            )
            low_freq_arr = np.asarray(low_freq_img, dtype=np.float32) - 128.0
            residual = delta + noise + low_freq_arr
            if photo_profile is not None:
                angle = float(photo_profile.get("light_angle_rad", photo_profile.get("target_light_angle_rad", 0.0)))
                yy, xx = np.mgrid[0:target.shape[0], 0:target.shape[1]].astype(np.float32)
                light_ramp = np.cos(xx * math.cos(angle) * 0.04 + yy * math.sin(angle) * 0.04)
                residual = residual + light_ramp * random.uniform(0.15, 0.35) * abs(delta)
            if random.random() < 0.35:
                # Fine scratches often have a brighter/darker core with a softer halo.
                core = alpha_arr_f > 0.45
                residual = residual + core.astype(np.float32) * delta * random.uniform(0.10, 0.35)
            patch_arr = np.clip(target + residual[..., None], 0, 255)
            alpha_arr_f = np.clip(
                alpha_arr_f
                * random.uniform(self.synthetic_defect_alpha_min, self.synthetic_defect_alpha_max)
                * pattern_profile["alpha"],
                0.0,
                1.0,
            )[..., None]
            blended = (target * (1.0 - alpha_arr_f) + patch_arr * alpha_arr_f).astype(np.uint8)
            out.paste(Image.fromarray(blended, mode="RGB"), (px0, py0))
            return out

        patch_arr = np.asarray(patch, dtype=np.float32)
        alpha_bin = alpha_arr_f > 0.15
        ring = alpha.filter(ImageFilter.MaxFilter(size=9))
        ring_arr = (np.asarray(ring, dtype=np.uint8) > 0) & ~alpha_bin
        if ring_arr.any() and alpha_bin.any():
            target_bg = target[ring_arr].mean(axis=0)
            patch_bg = patch_arr[ring_arr].mean(axis=0)
            patch_arr = np.clip(
                patch_arr + (target_bg - patch_bg) * self.synthetic_defect_bg_match_strength,
                0,
                255,
            )
        if strength > 0:
            texture_noise = np.random.normal(
                0.0,
                random.uniform(2.0, 10.0) * strength * pattern_profile["texture"],
                size=patch_arr.shape,
            ).astype(np.float32)
            patch_arr = np.clip(patch_arr + texture_noise, 0, 255)
        jitter = random.uniform(0.85, 1.12)
        patch_arr = np.clip(patch_arr * jitter, 0, 255)
        alpha_arr_f = np.clip(
            alpha_arr_f
            * random.uniform(self.synthetic_defect_alpha_min, self.synthetic_defect_alpha_max)
            * pattern_profile["alpha"],
            0.0,
            1.0,
        )[..., None]
        blended = (target * (1.0 - alpha_arr_f) + patch_arr * alpha_arr_f).astype(np.uint8)
        out.paste(Image.fromarray(blended, mode="RGB"), (px0, py0))
        return out

    def _load_texture_patch(self, item: dict) -> tuple[np.ndarray, np.ndarray] | None:
        patch_path = item.get("patch_path")
        alpha_path = item.get("alpha_path")
        if not patch_path or not alpha_path:
            return None
        cache_key = f"{patch_path}|{alpha_path}"
        cached = self._synthetic_defect_texture_cache.get(cache_key)
        if cached is not None:
            return cached
        patch = np.asarray(Image.open(project_path(str(patch_path))).convert("RGB"), dtype=np.float32)
        alpha = np.asarray(Image.open(project_path(str(alpha_path))).convert("L"), dtype=np.float32) / 255.0
        if patch.shape[:2] != alpha.shape or not np.any(alpha > 0.05):
            return None
        if len(self._synthetic_defect_texture_cache) > 64:
            self._synthetic_defect_texture_cache.clear()
        self._synthetic_defect_texture_cache[cache_key] = (patch, alpha)
        return patch, alpha

    def _machined_texture_from_library(
        self,
        base: np.ndarray,
        alpha: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        pattern_id: str = "",
    ) -> np.ndarray | None:
        if not self.synthetic_defect_texture_library:
            return None
        height, width = base.shape[:2]
        candidates = self._texture_candidates_for_pattern(pattern_id)
        if not candidates:
            return None
        item = random.choice(candidates)
        loaded = self._load_texture_patch(item)
        if loaded is None:
            return None
        patch, patch_alpha = loaded
        if self.synthetic_defect_pattern_aware and str(pattern_id).upper() == "P4":
            size = max(8, int(round(float(radius) * random.uniform(2.25, 3.05))))
        else:
            size = max(6, int(round(float(radius) * random.uniform(1.85, 2.55))))
        if size < 2 or cx - size // 2 < 0 or cy - size // 2 < 0 or cx + size // 2 >= width or cy + size // 2 >= height:
            return None
        patch_img = Image.fromarray(np.clip(patch, 0, 255).astype(np.uint8), mode="RGB")
        alpha_img = Image.fromarray(np.clip(patch_alpha * 255.0, 0, 255).astype(np.uint8), mode="L")
        if random.random() < 0.5:
            patch_img = ImageOps.mirror(patch_img)
            alpha_img = ImageOps.mirror(alpha_img)
        if random.random() < 0.5:
            angle = random.uniform(-35.0, 35.0)
            patch_img = patch_img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
            alpha_img = alpha_img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
        patch_img = patch_img.resize((size, size), Image.Resampling.BICUBIC)
        alpha_img = alpha_img.resize((size, size), Image.Resampling.BICUBIC).filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0.35, 0.85))
        )

        x0 = int(cx - size // 2)
        y0 = int(cy - size // 2)
        x1 = x0 + size
        y1 = y0 + size
        target = base[y0:y1, x0:x1].astype(np.float32)
        src = np.asarray(patch_img, dtype=np.float32)
        src_alpha = np.asarray(alpha_img, dtype=np.float32) / 255.0
        target_ring = (src_alpha > 0.02) & (src_alpha < 0.28)
        src_core = src_alpha > 0.35
        if not np.any(src_core):
            return None
        if np.any(target_ring):
            target_bg = target[target_ring].mean(axis=0)
            src_bg = src[target_ring].mean(axis=0)
            src = np.clip(src + (target_bg - src_bg) * self.synthetic_defect_bg_match_strength, 0, 255)
        target_low = np.asarray(
            Image.fromarray(np.clip(target, 0, 255).astype(np.uint8), mode="RGB").filter(
                ImageFilter.GaussianBlur(radius=random.uniform(1.2, 2.4))
            ),
            dtype=np.float32,
        )
        target_high = target - target_low
        src_low = np.asarray(
            Image.fromarray(np.clip(src, 0, 255).astype(np.uint8), mode="RGB").filter(
                ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.2))
            ),
            dtype=np.float32,
        )
        src_high = src - src_low
        photo_profile = self._sample_photometric_profile(target, src_alpha, family="machined", pattern_id=pattern_id)
        pattern_profile = self._pattern_defect_profile(pattern_id, "machined")
        high_gain = random.uniform(0.45, 0.95) * max(self.synthetic_defect_texture_strength, 0.1) * pattern_profile["texture"]
        src = np.clip(
            src_low
            + src_high * high_gain
            + target_high * random.uniform(0.35, 0.70) * pattern_profile["texture"],
            0,
            255,
        )
        contrast = float((photo_profile or item).get("contrast_luma", item.get("contrast_luma", 0.0)))
        if abs(contrast) > 1.0:
            # contrast_luma = bg - fg, so -contrast is the defect residual.
            residual_target = np.clip(-contrast, -42.0, 42.0) * random.uniform(0.30, 0.70) * pattern_profile["contrast"]
            current = float((src[src_core].mean() - target[src_core].mean()))
            src = np.clip(src + (residual_target - current) * src_alpha[..., None], 0, 255)
        alpha_local = src_alpha * alpha[y0:y1, x0:x1]
        if np.max(alpha_local) <= 0.05:
            return None
        out = base.copy()
        alpha_gain = random.uniform(self.synthetic_defect_alpha_min, self.synthetic_defect_alpha_max) * pattern_profile["alpha"]
        alpha3 = np.clip(alpha_local * alpha_gain, 0.0, 1.0)[..., None]
        out[y0:y1, x0:x1] = target * (1.0 - alpha3) + src * alpha3
        return out

    def _defect_texture_from_library(
        self,
        base: np.ndarray,
        alpha: np.ndarray,
        shape: str,
        pattern_id: str = "",
    ) -> np.ndarray | None:
        candidates = self._texture_candidates_for_shape(shape, pattern_id=pattern_id)
        if not candidates:
            return None
        item = random.choice(candidates)
        loaded = self._load_texture_patch(item)
        if loaded is None:
            return None
        patch, patch_alpha = loaded
        ys, xs = np.where(alpha > 0.04)
        if len(xs) < 3:
            return None
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        target_w = max(3, x1 - x0)
        target_h = max(3, y1 - y0)
        if x0 < 0 or y0 < 0 or x1 > base.shape[1] or y1 > base.shape[0]:
            return None
        patch_img = Image.fromarray(np.clip(patch, 0, 255).astype(np.uint8), mode="RGB")
        alpha_img = Image.fromarray(np.clip(patch_alpha * 255.0, 0, 255).astype(np.uint8), mode="L")
        if random.random() < 0.5:
            patch_img = ImageOps.mirror(patch_img)
            alpha_img = ImageOps.mirror(alpha_img)
        if random.random() < 0.35:
            patch_img = ImageOps.flip(patch_img)
            alpha_img = ImageOps.flip(alpha_img)
        if str(shape).lower() in {"scratch", "stain"} and random.random() < 0.45:
            angle = random.uniform(-25.0, 25.0)
            patch_img = patch_img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
            alpha_img = alpha_img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)
        patch_img = patch_img.resize((target_w, target_h), Image.Resampling.BICUBIC)
        alpha_img = alpha_img.resize((target_w, target_h), Image.Resampling.BICUBIC).filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0.35, 1.1))
        )
        target = base[y0:y1, x0:x1].astype(np.float32)
        src = np.asarray(patch_img, dtype=np.float32)
        src_alpha = np.asarray(alpha_img, dtype=np.float32) / 255.0
        dst_alpha = alpha[y0:y1, x0:x1]
        alpha_local = np.clip(src_alpha * dst_alpha, 0.0, 1.0)
        if np.max(alpha_local) <= 0.05:
            return None
        active = alpha_local > 0.12
        ring = (alpha_local > 0.02) & ~active
        if np.any(ring):
            target_bg = target[ring].mean(axis=0)
            src_bg = src[ring].mean(axis=0)
            src = np.clip(src + (target_bg - src_bg) * self.synthetic_defect_bg_match_strength, 0, 255)
        target_low = np.asarray(
            Image.fromarray(np.clip(target, 0, 255).astype(np.uint8), mode="RGB").filter(
                ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.2))
            ),
            dtype=np.float32,
        )
        src_low = np.asarray(
            Image.fromarray(np.clip(src, 0, 255).astype(np.uint8), mode="RGB").filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0.8, 1.8))
            ),
            dtype=np.float32,
        )
        target_high = target - target_low
        src_high = src - src_low
        photo_profile = self._sample_photometric_profile(target, alpha_local, family=shape, pattern_id=pattern_id)
        pattern_profile = self._pattern_defect_profile(pattern_id, shape)
        high_gain = random.uniform(0.50, 1.05) * max(self.synthetic_defect_texture_strength, 0.1) * pattern_profile["texture"]
        src = np.clip(
            src_low
            + src_high * high_gain
            + target_high * random.uniform(0.25, 0.60) * pattern_profile["texture"],
            0,
            255,
        )
        contrast = float((photo_profile or item).get("contrast_luma", item.get("contrast_luma", 0.0)))
        if abs(contrast) > 1.0 and np.any(active):
            residual_target = np.clip(-contrast, -42.0, 42.0) * random.uniform(0.30, 0.70) * pattern_profile["contrast"]
            current = float(src[active].mean() - target[active].mean())
            src = np.clip(src + (residual_target - current) * alpha_local[..., None], 0, 255)
        out = base.copy()
        alpha_gain = random.uniform(self.synthetic_defect_alpha_min, self.synthetic_defect_alpha_max) * pattern_profile["alpha"]
        alpha3 = np.clip(alpha_local * alpha_gain, 0.0, 1.0)[..., None]
        out[y0:y1, x0:x1] = target * (1.0 - alpha3) + src * alpha3
        return out

    def _paste_cluster_texture_defect(
        self,
        image: Image.Image,
        surface: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        shape: str,
        pattern_id: str = "",
    ) -> Image.Image | None:
        candidates = self._texture_candidates_for_shape(shape, pattern_id=pattern_id)
        if not candidates:
            return None
        out = image.convert("RGB")
        width, height = out.size
        item = random.choice(candidates)
        loaded = self._load_texture_patch(item)
        if loaded is None:
            return None
        patch, patch_alpha = loaded
        patch_img = Image.fromarray(np.clip(patch, 0, 255).astype(np.uint8), mode="RGB")
        alpha_img = Image.fromarray(np.clip(patch_alpha * 255.0, 0, 255).astype(np.uint8), mode="L")
        pattern_profile = self._pattern_defect_profile(pattern_id, shape)

        source_size = max(float(item.get("bbox_size", max(patch_img.size))), 1.0)
        if self.synthetic_defect_pattern_aware and str(pattern_id).upper() == "P4" and str(shape).lower() == "machined":
            scale = random.uniform(0.95, 1.35)
        elif str(shape).lower() == "scratch":
            scale = random.uniform(0.85, 1.20)
        elif str(shape).lower() == "stain":
            scale = random.uniform(0.80, 1.25)
        else:
            scale = random.uniform(0.70, 1.15)
        scale *= pattern_profile["radius"]
        target_size = max(4, int(round(source_size * scale)))
        patch_ratio = patch_img.width / max(float(patch_img.height), 1.0)
        if patch_ratio >= 1.0:
            new_w = target_size
            new_h = max(4, int(round(target_size / patch_ratio)))
        else:
            new_h = target_size
            new_w = max(4, int(round(target_size * patch_ratio)))
        if str(shape).lower() == "scratch":
            new_w = max(new_w, int(round(new_h * random.uniform(3.0, 7.0))))
        patch_img = patch_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        alpha_img = alpha_img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        if random.random() < 0.5:
            patch_img = ImageOps.mirror(patch_img)
            alpha_img = ImageOps.mirror(alpha_img)
        if random.random() < 0.35:
            patch_img = ImageOps.flip(patch_img)
            alpha_img = ImageOps.flip(alpha_img)
        if random.random() < 0.60:
            angle = random.uniform(-35.0, 35.0)
            if str(shape).lower() == "scratch":
                angle = random.uniform(-12.0, 12.0)
            patch_img = patch_img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
            alpha_img = alpha_img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.35, 1.0)))

        alpha_arr = np.asarray(alpha_img, dtype=np.float32) / 255.0
        alpha_bin = alpha_arr > 0.08
        if not alpha_bin.any():
            return None
        for _ in range(32):
            point_idx = random.randrange(len(xs))
            cx = int(xs[point_idx])
            cy = int(ys[point_idx])
            px0 = int(cx - patch_img.width // 2)
            py0 = int(cy - patch_img.height // 2)
            px1 = px0 + patch_img.width
            py1 = py0 + patch_img.height
            if px0 < 0 or py0 < 0 or px1 > width or py1 > height:
                continue
            roi_surface = surface[py0:py1, px0:px1]
            overlap = float((roi_surface & alpha_bin).sum()) / float(alpha_bin.sum())
            if overlap >= self.synthetic_defect_min_surface_overlap:
                break
        else:
            return None

        target = np.asarray(out.crop((px0, py0, px1, py1)), dtype=np.float32)
        src = np.asarray(patch_img, dtype=np.float32)
        alpha_arr = alpha_arr * roi_surface.astype(np.float32)
        active = alpha_arr > 0.12
        ring = (alpha_arr > 0.02) & ~active
        if np.any(ring):
            target_bg = target[ring].mean(axis=0)
            src_bg = src[ring].mean(axis=0)
            src = np.clip(src + (target_bg - src_bg) * self.synthetic_defect_bg_match_strength, 0, 255)

        target_low = np.asarray(
            Image.fromarray(np.clip(target, 0, 255).astype(np.uint8), mode="RGB").filter(
                ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.2))
            ),
            dtype=np.float32,
        )
        src_low = np.asarray(
            Image.fromarray(np.clip(src, 0, 255).astype(np.uint8), mode="RGB").filter(
                ImageFilter.GaussianBlur(radius=random.uniform(0.8, 1.6))
            ),
            dtype=np.float32,
        )
        photo_profile = self._sample_photometric_profile(target, alpha_arr, family=shape, pattern_id=pattern_id)
        src_high = src - src_low
        target_high = target - target_low
        src = np.clip(
            src_low
            + src_high * random.uniform(0.50, 1.05) * max(self.synthetic_defect_texture_strength, 0.1) * pattern_profile["texture"]
            + target_high * random.uniform(0.25, 0.60) * pattern_profile["texture"],
            0,
            255,
        )
        contrast = float((photo_profile or item).get("contrast_luma", item.get("contrast_luma", 0.0)))
        if abs(contrast) > 1.0 and np.any(active):
            residual_target = np.clip(-contrast, -48.0, 48.0) * random.uniform(0.35, 0.75) * pattern_profile["contrast"]
            current = float(src[active].mean() - target[active].mean())
            src = np.clip(src + (residual_target - current) * alpha_arr[..., None], 0, 255)
        alpha_gain = random.uniform(self.synthetic_defect_alpha_min, self.synthetic_defect_alpha_max) * pattern_profile["alpha"]
        alpha3 = np.clip(alpha_arr * alpha_gain, 0.0, 1.0)[..., None]
        blended = target * (1.0 - alpha3) + src * alpha3
        out.paste(Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB"), (px0, py0))
        return out

    def _draw_generic_defects(
        self,
        image: Image.Image,
        xs: np.ndarray,
        ys: np.ndarray,
        surface: np.ndarray | None = None,
        pattern_id: str = "",
    ) -> Image.Image:
        out = image.convert("RGB").copy()
        width, height = out.size
        if surface is None:
            surface = np.zeros((height, width), dtype=bool)
            surface[np.clip(ys, 0, height - 1), np.clip(xs, 0, width - 1)] = True
        else:
            surface = np.asarray(surface, dtype=bool)
        min_dim = min(width, height)
        min_radius = max(2, int(round(min_dim * self.synthetic_defect_min_radius_frac)))
        max_radius = max(min_radius + 1, int(round(min_dim * self.synthetic_defect_max_radius_frac)))
        min_scratch = max(8, int(round(min_dim * self.synthetic_defect_scratch_min_length_frac)))
        max_scratch = max(min_scratch + 1, int(round(min_dim * self.synthetic_defect_scratch_max_length_frac)))
        count_profile = self._pattern_defect_profile(pattern_id, self._dominant_defect_shape())
        base_count = max(1, self.synthetic_defect_max_blobs)
        count_hi = max(1, int(round(float(base_count) * count_profile["count"])))
        count_lo = max(1, int(round(float(count_hi) * 0.55)))
        blob_count = random.randint(count_lo, count_hi)

        for _ in range(blob_count):
            alpha = None
            shape = self._sample_defect_shape()
            pattern_profile = self._pattern_defect_profile(pattern_id, shape)
            if self.synthetic_defect_texture_library and random.random() < 0.88:
                textured_out = self._paste_cluster_texture_defect(
                    out,
                    surface,
                    xs,
                    ys,
                    shape=shape,
                    pattern_id=pattern_id,
                )
                if textured_out is not None:
                    out = textured_out
                    continue
            machined_params = None
            for _attempt in range(32):
                point_idx = random.randrange(len(xs))
                cx = int(xs[point_idx])
                cy = int(ys[point_idx])
                mask_img = Image.new("L", (width, height), 0)
                draw = ImageDraw.Draw(mask_img)
                if shape == "machined":
                    machined_min_radius, machined_max_radius = self._machined_radius_bounds_for_pattern(
                        pattern_id,
                        min_radius,
                        max_radius,
                    )
                    radius = random.randint(max(2, machined_min_radius), max(3, machined_max_radius))
                    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
                    machined_params = (cx, cy, radius)
                    blur_radius = random.uniform(0.65, 1.25)
                elif shape == "scratch" or (shape != "scratch" and random.random() < self.synthetic_defect_scratch_p * 0.25):
                    scratch_min = max(8, int(round(float(min_scratch) * pattern_profile["length"])))
                    scratch_max = max(scratch_min + 1, int(round(float(max_scratch) * pattern_profile["length"])))
                    length = random.randint(scratch_min, scratch_max)
                    angle = random.uniform(0.0, 2.0 * np.pi)
                    dx = int(np.cos(angle) * length * 0.5)
                    dy = int(np.sin(angle) * length * 0.5)
                    min_line = max(1, int(round(float(min_radius) * pattern_profile["radius"] / 3.0)))
                    max_line = max(2, int(round(float(max_radius) * pattern_profile["radius"] / 2.0)))
                    line_width = random.randint(min_line, max(min_line + 1, max_line))
                    draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=random.randint(155, 255), width=line_width)
                    if random.random() < 0.45:
                        branch_angle = angle + random.uniform(-0.55, 0.55)
                        branch_len = int(length * random.uniform(0.25, 0.65))
                        bx = int(np.cos(branch_angle) * branch_len)
                        by = int(np.sin(branch_angle) * branch_len)
                        draw.line((cx, cy, cx + bx, cy + by), fill=random.randint(105, 220), width=max(1, line_width // 2))
                    blur_radius = random.uniform(0.45, 1.4)
                elif shape == "stain":
                    shape_min_radius = max(2, int(round(float(min_radius) * pattern_profile["radius"])))
                    shape_max_radius = max(shape_min_radius + 1, int(round(float(max_radius) * pattern_profile["radius"])))
                    radius = random.randint(shape_min_radius, shape_max_radius)
                    for _part in range(random.randint(3, 8)):
                        ox = random.randint(-radius, radius)
                        oy = random.randint(-radius, radius)
                        rr = random.randint(max(2, radius // 4), max(3, radius))
                        rx = int(rr * random.uniform(0.45, 1.45))
                        ry = int(rr * random.uniform(0.45, 1.45))
                        draw.ellipse((cx + ox - rx, cy + oy - ry, cx + ox + rx, cy + oy + ry), fill=random.randint(70, 210))
                    blur_radius = random.uniform(1.0, 3.2)
                else:
                    shape_min_radius = max(2, int(round(float(min_radius) * pattern_profile["radius"])))
                    shape_max_radius = max(shape_min_radius + 1, int(round(float(max_radius) * pattern_profile["radius"])))
                    radius = random.randint(shape_min_radius, shape_max_radius)
                    rx = int(radius * random.uniform(0.75, 1.35))
                    ry = int(radius * random.uniform(0.75, 1.35))
                    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=random.randint(175, 255))
                    if random.random() < 0.65:
                        inner_rx = max(1, int(rx * random.uniform(0.35, 0.70)))
                        inner_ry = max(1, int(ry * random.uniform(0.35, 0.70)))
                        draw.ellipse((cx - inner_rx, cy - inner_ry, cx + inner_rx, cy + inner_ry), fill=255)
                    blur_radius = random.uniform(0.75, 2.0)

                if random.random() < 0.40:
                    mask_img = mask_img.filter(ImageFilter.MaxFilter(size=3))
                mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                alpha_arr = np.asarray(mask_img, dtype=np.float32) / 255.0
                alpha_bin = alpha_arr > 0.12
                if not alpha_bin.any():
                    continue
                overlap = float((surface & alpha_bin).sum()) / float(alpha_bin.sum())
                if overlap >= self.synthetic_defect_min_surface_overlap:
                    alpha = alpha_arr
                    break
            if alpha is None:
                continue

            arr = np.asarray(out, dtype=np.float32)
            alpha3 = alpha[..., None]
            active = alpha > 0.02
            if not active.any():
                continue
            texture_strength = self.synthetic_defect_texture_strength * pattern_profile["texture"]
            if shape == "machined" and machined_params is not None:
                cx, cy, radius = machined_params
                if random.random() < 0.85:
                    textured = self._machined_texture_from_library(arr, alpha, cx, cy, radius, pattern_id=pattern_id)
                    if textured is not None:
                        out = Image.fromarray(np.clip(textured, 0, 255).astype(np.uint8), mode="RGB")
                        continue
                yy, xx = np.mgrid[0:height, 0:width]
                dist = np.sqrt((xx.astype(np.float32) - float(cx)) ** 2 + (yy.astype(np.float32) - float(cy)) ** 2)
                rel_r = dist / max(float(radius), 1.0)
                disk = np.clip(1.0 - (dist / max(float(radius), 1.0)) ** 2, 0.0, 1.0)
                edge = np.exp(-((dist - float(radius) * 0.92) ** 2) / (2.0 * max(float(radius) * 0.07, 1.0) ** 2))
                photo_profile = self._sample_photometric_profile(arr, alpha, family="machined", pattern_id=pattern_id)
                light_angle = random.uniform(-2.6, -0.3)
                if photo_profile is not None:
                    local_angle = float(photo_profile.get("target_light_angle_rad", photo_profile.get("light_angle_rad", light_angle)))
                    real_angle = float(photo_profile.get("light_angle_rad", local_angle))
                    light_angle = 0.65 * local_angle + 0.35 * real_angle + random.uniform(-0.35, 0.35)
                hx = float(cx) + np.cos(light_angle) * float(radius) * random.uniform(0.20, 0.42)
                hy = float(cy) + np.sin(light_angle) * float(radius) * random.uniform(0.20, 0.42)
                hdist = np.sqrt((xx.astype(np.float32) - hx) ** 2 + (yy.astype(np.float32) - hy) ** 2)
                point_highlight = np.exp(
                    -(hdist**2) / (2.0 * max(float(radius) * random.uniform(0.16, 0.30), 1.0) ** 2)
                )
                theta = np.arctan2(yy.astype(np.float32) - float(cy), xx.astype(np.float32) - float(cx))
                angle_delta = np.arctan2(np.sin(theta - light_angle), np.cos(theta - light_angle))
                crescent = np.exp(-(angle_delta**2) / (2.0 * random.uniform(0.22, 0.48) ** 2))
                crescent *= np.exp(-((rel_r - random.uniform(0.55, 0.78)) ** 2) / (2.0 * random.uniform(0.11, 0.18) ** 2))
                bg = arr.copy()
                low_freq = np.asarray(
                    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB").filter(
                        ImageFilter.GaussianBlur(radius=random.uniform(1.4, 2.4))
                    ),
                    dtype=np.float32,
                )
                high_freq = arr - low_freq
                texture_keep = random.uniform(0.58, 0.84)
                machined_texture = low_freq + high_freq * random.uniform(0.85, 1.30) * max(texture_strength, 0.1)
                machined_texture = machined_texture * texture_keep + arr * (1.0 - texture_keep)
                recess_delta = random.uniform(-16.0, -3.0)
                if photo_profile is not None:
                    contrast = float(photo_profile.get("contrast_luma", 0.0))
                    if abs(contrast) > 1.0:
                        recess_delta = (
                            float(np.clip(-contrast, -30.0, 18.0))
                            * random.uniform(0.35, 0.75)
                            * pattern_profile["contrast"]
                        )
                if random.random() < 0.35:
                    recess_delta = random.uniform(2.0, 7.0) * pattern_profile["contrast"]
                rim_delta = random.uniform(0.5, 4.0) * min(pattern_profile["contrast"], 2.2)
                if random.random() < 0.45:
                    rim_delta *= -0.45
                point_spec_delta = random.uniform(8.0, 28.0) * min(pattern_profile["contrast"], 2.4)
                crescent_spec_delta = random.uniform(12.0, 45.0) * min(pattern_profile["contrast"], 2.4)
                side_shadow = np.clip(np.cos(theta - light_angle - np.pi) * 0.5 + 0.5, 0.0, 1.0) * disk
                residual = (
                    disk * recess_delta
                    + edge * rim_delta
                    - side_shadow * random.uniform(1.0, 6.0)
                    + point_highlight * point_spec_delta
                    + crescent * crescent_spec_delta
                )
                defect = machined_texture + residual[..., None]
                if texture_strength > 0:
                    fine_noise = np.random.normal(
                        0.0,
                        random.uniform(0.35, 1.8) * texture_strength,
                        size=arr.shape,
                    ).astype(np.float32)
                    line_mod = np.random.normal(
                        0.0,
                        random.uniform(1.5, 5.0) * texture_strength,
                        size=(1, width, 1),
                    ).astype(np.float32)
                    if random.random() < 0.55:
                        line_mod = np.random.normal(
                            0.0,
                            random.uniform(1.0, 4.0) * texture_strength,
                            size=(height, 1, 1),
                        ).astype(np.float32)
                    defect += fine_noise + line_mod
                defect = np.clip(defect, 0, 255)
                alpha_gain = random.uniform(self.synthetic_defect_alpha_min, self.synthetic_defect_alpha_max) * pattern_profile["alpha"]
                alpha3 = np.clip(alpha3 * alpha_gain, 0.0, 1.0)
                blended = bg * (1.0 - alpha3) + defect * alpha3
                out = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")
                continue

            if self.synthetic_defect_texture_library and random.random() < 0.80:
                textured = self._defect_texture_from_library(arr, alpha, shape=shape, pattern_id=pattern_id)
                if textured is not None:
                    out = Image.fromarray(np.clip(textured, 0, 255).astype(np.uint8), mode="RGB")
                    continue

            base_delta = (
                random.uniform(22.0, 90.0)
                * max(self.synthetic_defect_texture_strength, 0.1)
                * pattern_profile["contrast"]
                * math.sqrt(max(pattern_profile["texture"], 1.0))
            )
            base_delta = float(np.clip(base_delta, 8.0, 170.0))
            # Casting defects in the test masks are sometimes brighter and
            # sometimes darker than their local background, so sample both.
            if random.random() < 0.55:
                defect = arr + base_delta
            else:
                defect = arr - base_delta
            noise_std = random.uniform(4.0, 22.0) * texture_strength
            if noise_std > 0:
                defect = defect + np.random.normal(0.0, noise_std, size=arr.shape).astype(np.float32)
            if shape == "scratch" and random.random() < 0.55:
                stripe = np.random.uniform(0.75, 1.20, size=(height, 1, 1)).astype(np.float32)
                defect = defect * stripe
            defect = np.clip(defect, 0, 255)
            alpha_gain = random.uniform(self.synthetic_defect_alpha_min, self.synthetic_defect_alpha_max) * pattern_profile["alpha"]
            alpha3 = np.clip(alpha3 * alpha_gain, 0.0, 1.0)
            blended = arr * (1.0 - alpha3) + defect * alpha3
            out = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")
        return out

    def _apply_synthetic_defects(
        self,
        image: Image.Image,
        semantic: Image.Image,
        surface_mask: np.ndarray | None = None,
        pattern_id: str = "",
    ) -> Image.Image:
        if self.synthetic_defect_p <= 0 or random.random() >= self.synthetic_defect_p:
            return image
        if surface_mask is None:
            surface = np.asarray(semantic.convert("L"), dtype=np.uint8) == 1
        else:
            surface = np.asarray(surface_mask, dtype=bool)
            if surface.shape != (image.height, image.width):
                surface_image = Image.fromarray(surface.astype(np.uint8) * 255, mode="L").resize(
                    image.size,
                    Image.Resampling.NEAREST,
                )
                surface = np.asarray(surface_image, dtype=np.uint8) > 0
        ys, xs = np.where(surface)
        if len(xs) < 32:
            return image
        if self.synthetic_defect_mode in {"realistic", "mixed"}:
            out = image
            paste_count = random.randint(1, max(1, self.synthetic_defect_max_blobs))
            for _ in range(paste_count):
                out = self._paste_realistic_defect(out, surface, xs, ys, pattern_id=pattern_id)
            if self.synthetic_defect_mode == "mixed" and random.random() < 0.65:
                out = self._draw_generic_defects(out, xs, ys, surface=surface, pattern_id=pattern_id)
            return out
        return self._draw_generic_defects(image, xs, ys, surface=surface, pattern_id=pattern_id)

    def _context_bundle(self, source_image: Image.Image, crop_box: tuple[float, float, float, float]) -> tuple[torch.Tensor, torch.Tensor]:
        global_image = resize_letterbox_pil(source_image, self.context_size, mode="RGB")
        crop_box_mask = crop_box_to_mask(crop_box, self.context_size)
        return self._to_tensor_image(global_image), self._soft_tensor(crop_box_mask)

    def __getitem__(self, index: int) -> dict:
        row = self._row(index)
        pattern_id = self._pattern_id_from_row(row)
        image = Image.open(project_path(str(row["image_path"]))).convert("RGB")
        semantic = Image.open(project_path(str(row[self.semantic_column]))).convert("L")
        source_image = self._augment_image(image)
        if self.augmentation_profile is not None:
            profile = self.augmentation_profile
            workspace_size = int(getattr(profile, "workspace_size", None) or self.input_size)
            image_ws = resize_letterbox_pil(source_image, workspace_size, mode="RGB")
            semantic_ws = resize_letterbox_pil(semantic, workspace_size, mode="binary")
            i, j, h, w = self._crop_params(image_ws, semantic_ws)
            crop_box = (
                (float(j) + float(w) * 0.5) / float(workspace_size),
                (float(i) + float(h) * 0.5) / float(workspace_size),
                float(w) / float(workspace_size),
                float(h) / float(workspace_size),
            )
            context_source_image = source_image
            crop_source_image = image_ws
            if self.synthetic_defect_context_consistent:
                surface_region = np.asarray(semantic_ws.convert("L"), dtype=np.uint8) == 1
                if self.synthetic_defect_crop_localized:
                    localized = np.zeros_like(surface_region, dtype=bool)
                    localized[i : i + h, j : j + w] = surface_region[i : i + h, j : j + w]
                    surface_region = localized
                crop_source_image = self._apply_synthetic_defects(
                    image_ws,
                    semantic_ws,
                    surface_mask=surface_region,
                    pattern_id=pattern_id,
                )
                context_source_image = crop_source_image
            image_aug = TF.resized_crop(crop_source_image, i, j, h, w, (self.input_size, self.input_size), interpolation=TF.InterpolationMode.BILINEAR)
            semantic_aug = TF.resized_crop(semantic_ws, i, j, h, w, (self.input_size, self.input_size), interpolation=TF.InterpolationMode.NEAREST)
            if profile.horizontal_flip_p > 0 and random.random() < profile.horizontal_flip_p:
                image_aug = TF.hflip(image_aug)
                semantic_aug = TF.hflip(semantic_aug)
            if profile.vertical_flip_p > 0 and random.random() < profile.vertical_flip_p:
                image_aug = TF.vflip(image_aug)
                semantic_aug = TF.vflip(semantic_aug)
            if profile.rotation_degrees > 0:
                angle = random.uniform(-float(profile.rotation_degrees), float(profile.rotation_degrees))
                image_aug = TF.rotate(image_aug, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
                semantic_aug = TF.rotate(semantic_aug, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)
            if any(value > 0 for value in (profile.brightness, profile.contrast, profile.saturation, abs(profile.hue))):
                jitter = transforms.ColorJitter(
                    brightness=profile.brightness,
                    contrast=profile.contrast,
                    saturation=profile.saturation,
                    hue=profile.hue,
                )
                image_aug = jitter(image_aug)
            if profile.blur_p > 0 and random.random() < profile.blur_p:
                image_aug = TF.gaussian_blur(image_aug, kernel_size=3)
            if not self.synthetic_defect_context_consistent:
                image_aug = self._apply_synthetic_defects(image_aug, semantic_aug, pattern_id=pattern_id)
            global_image, crop_box_mask = self._context_bundle(
                context_source_image,
                crop_box if self.context_crop_prob > 0 else (0.5, 0.5, 1.0, 1.0),
            )
        else:
            image_aug = resize_letterbox_pil(source_image, self.input_size, mode="RGB")
            semantic_aug = resize_letterbox_pil(semantic, self.input_size, mode="binary")
            image_aug = self._apply_synthetic_defects(image_aug, semantic_aug, pattern_id=pattern_id)
            context_source_image = image_aug if self.synthetic_defect_context_consistent else source_image
            global_image, crop_box_mask = self._context_bundle(context_source_image, (0.5, 0.5, 1.0, 1.0))

        return {
            "image": self._to_tensor_image(image_aug),
            "semantic": self._semantic_tensor(semantic_aug),
            "global_image": global_image,
            "crop_box_mask": crop_box_mask,
            "image_path": str(row["image_path"]),
            "sample_weight": float(row.get("sample_weight", 1.0)),
        }


def collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "semantic": torch.stack([item["semantic"] for item in batch]),
        "global_image": torch.stack([item["global_image"] for item in batch]),
        "crop_box_mask": torch.stack([item["crop_box_mask"] for item in batch]),
        "image_path": [item["image_path"] for item in batch],
        "sample_weight": torch.tensor([item["sample_weight"] for item in batch], dtype=torch.float32),
    }


def compute_class_weights(df: pd.DataFrame, semantic_column: str, num_classes: int, raw: str) -> torch.Tensor | None:
    if str(raw).lower() == "none":
        return None
    if str(raw).lower() != "auto":
        values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
        if len(values) != int(num_classes):
            raise ValueError(f"Expected {num_classes} class weights, got {len(values)}.")
        return torch.tensor(values, dtype=torch.float32)
    counts = np.zeros(int(num_classes), dtype=np.float64)
    for path in df[semantic_column]:
        arr = np.asarray(Image.open(project_path(str(path))).convert("L"), dtype=np.uint8)
        for cls in range(int(num_classes)):
            counts[cls] += float((arr == cls).sum())
    freq = counts / max(float(counts.sum()), 1.0)
    weights = 1.0 / np.sqrt(np.maximum(freq, 1e-8))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def save_previews(run_dir: Path, model, dataset: SemanticSurfaceDataset, device, max_items: int, preview_head: str = "mask") -> None:
    save_functional_surface_previews(
        run_dir,
        model,
        dataset,
        device,
        max_items,
        preview_head=preview_head,
    )


def save_checkpoint(run_dir: Path, model, optimizer, args, history, best_epoch, best_val_loss, completed_epochs, save_as_best: bool) -> None:
    save_functional_surface_checkpoint(
        run_dir,
        model,
        optimizer,
        args,
        history,
        best_epoch,
        best_val_loss,
        completed_epochs,
        save_as_best=save_as_best,
        class_names=CLASS_NAMES,
    )


def main() -> None:
    args = parse_args()
    args.run_created_at = datetime.now(timezone.utc).isoformat()
    args.command_line = " ".join([sys.executable, *sys.argv])
    args.labels_dir = args.labels_dir if args.labels_dir.is_absolute() else PATHS.root / args.labels_dir
    if args.init_checkpoint_path is not None:
        args.init_checkpoint_path = (
            args.init_checkpoint_path
            if args.init_checkpoint_path.is_absolute()
            else PATHS.root / args.init_checkpoint_path
        )
    if args.external_monitor_labels_dir is not None:
        args.external_monitor_labels_dir = (
            args.external_monitor_labels_dir
            if args.external_monitor_labels_dir.is_absolute()
            else PATHS.root / args.external_monitor_labels_dir
        )
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else PATHS.root / args.output_dir
    args.context_size = int(args.context_size) if args.context_size is not None else int(args.input_size)
    run_dir = args.output_dir / args.category / args.run_name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite_run:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels_dir / "labels_index.csv")
    if args.limit_train is not None:
        labels = labels.head(int(args.limit_train)).reset_index(drop=True)
    if args.best_monitor == "external_loss" and args.external_monitor_labels_dir is None:
        raise ValueError("--best-monitor external_loss requires --external-monitor-labels-dir.")
    split_column = resolve_split_column(labels, args.split_column)
    train_df, val_df, split_info = split_df(
        labels,
        args.val_fraction,
        strategy=args.split_strategy,
        split_column=split_column,
        seed=args.split_seed,
    )
    split_info["train_count"] = int(len(train_df))
    split_info["val_count"] = int(len(val_df)) if val_df is not None else 0
    (run_dir / "split_info.json").write_text(json.dumps(split_info, indent=2, default=str), encoding="utf-8")
    train_df.to_csv(run_dir / "train_split_labels_index.csv", index=False)
    if val_df is not None:
        val_df.to_csv(run_dir / "val_split_labels_index.csv", index=False)
    external_monitor_df = None
    if args.external_monitor_labels_dir is not None:
        external_monitor_df = pd.read_csv(args.external_monitor_labels_dir / "labels_index.csv")
        external_monitor_df.to_csv(run_dir / f"{args.external_monitor_name}_monitor_labels_index.csv", index=False)
    print(
        "Split "
        f"{split_info.get('strategy')} | train={split_info['train_count']} | val={split_info['val_count']} "
        f"| column={split_info.get('split_column', split_column)}"
    )
    if external_monitor_df is not None:
        print(
            f"External monitor '{args.external_monitor_name}' | rows={len(external_monitor_df)} "
            f"| labels_dir={args.external_monitor_labels_dir}"
        )

    augmentation_profile = resolve_augmentation_profile(args.augmentation_profile, args.category)
    args.augmentation_profile_resolved = augmentation_profile.name
    args.augmentation_profile_params = augmentation_profile.to_dict()
    args.repeat_factor_resolved = resolve_repeat_factor(args.repeat_factor, augmentation_profile)

    train_dataset = SemanticSurfaceDataset(
        train_df,
        args.input_size,
        args.semantic_mask_column,
        context_size=args.context_size,
        augmentation_profile=augmentation_profile,
        repeat_factor=args.repeat_factor_resolved,
        context_crop_prob=args.context_crop_prob,
        positive_crop_prob=args.positive_crop_prob,
        train_photometric_normalization_p=args.train_photometric_normalization_p,
        train_photo_target_p05=args.train_photo_target_p05,
        train_photo_target_p95=args.train_photo_target_p95,
        synthetic_defect_p=args.synthetic_defect_p,
        synthetic_defect_mode=args.synthetic_defect_mode,
        synthetic_defect_realistic_render=args.synthetic_defect_realistic_render,
        synthetic_defect_library_json=args.synthetic_defect_library_json,
        synthetic_defect_texture_library_json=args.synthetic_defect_texture_library_json,
        synthetic_defect_photometric_library_json=args.synthetic_defect_photometric_library_json,
        synthetic_defect_pattern_aware=args.synthetic_defect_pattern_aware,
        synthetic_defect_p4_large_p=args.synthetic_defect_p4_large_p,
        synthetic_defect_max_blobs=args.synthetic_defect_max_blobs,
        synthetic_defect_min_radius_frac=args.synthetic_defect_min_radius_frac,
        synthetic_defect_max_radius_frac=args.synthetic_defect_max_radius_frac,
        synthetic_defect_shape_weights=args.synthetic_defect_shape_weights,
        synthetic_defect_scratch_min_length_frac=args.synthetic_defect_scratch_min_length_frac,
        synthetic_defect_scratch_max_length_frac=args.synthetic_defect_scratch_max_length_frac,
        synthetic_defect_scratch_p=args.synthetic_defect_scratch_p,
        synthetic_defect_texture_strength=args.synthetic_defect_texture_strength,
        synthetic_defect_variant_strength=args.synthetic_defect_variant_strength,
        synthetic_defect_large_p=args.synthetic_defect_large_p,
        synthetic_defect_large_quantile=args.synthetic_defect_large_quantile,
        synthetic_defect_large_scale_min=args.synthetic_defect_large_scale_min,
        synthetic_defect_large_scale_max=args.synthetic_defect_large_scale_max,
        synthetic_defect_alpha_min=args.synthetic_defect_alpha_min,
        synthetic_defect_alpha_max=args.synthetic_defect_alpha_max,
        synthetic_defect_bg_match_strength=args.synthetic_defect_bg_match_strength,
        synthetic_defect_min_surface_overlap=args.synthetic_defect_min_surface_overlap,
        synthetic_defect_context_consistent=args.synthetic_defect_context_consistent,
        synthetic_defect_crop_localized=args.synthetic_defect_crop_localized,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)

    val_loader = None
    val_dataset = None
    if val_df is not None:
        val_dataset = SemanticSurfaceDataset(val_df, args.input_size, args.semantic_mask_column, context_size=args.context_size)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    external_monitor_dataset = None
    external_monitor_loader = None
    if external_monitor_df is not None:
        external_monitor_dataset = SemanticSurfaceDataset(
            external_monitor_df,
            args.input_size,
            args.semantic_mask_column,
            context_size=args.context_size,
        )
        external_monitor_loader = DataLoader(
            external_monitor_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
        )

    device = resolve_device(args.device)
    model = build_segmentation_model(args.model_type)
    replace_segmentation_head(model, args.num_classes)
    if args.init_checkpoint_path is not None:
        init_checkpoint = torch.load(args.init_checkpoint_path, map_location="cpu", weights_only=False)
        init_model_type = str(init_checkpoint.get("model_type", ""))
        if init_model_type and init_model_type != str(args.model_type):
            raise ValueError(
                f"--init-checkpoint-path model_type={init_model_type!r} does not match --model-type={args.model_type!r}."
            )
        model.load_state_dict(init_checkpoint["model_state_dict"], strict=True)
        print(f"Initialized model from {args.init_checkpoint_path}")
    model = model.to(device)
    class_weights = compute_class_weights(train_df, args.semantic_mask_column, args.num_classes, args.class_weights)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    dice_classes = parse_int_list(args.dice_classes)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=args.lr_patience, factor=args.lr_factor) if args.lr_scheduler == "plateau" else None

    (run_dir / "training_config.json").write_text(
        json.dumps(
            {
                "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
                "dice_classes": dice_classes,
                "class_names": CLASS_NAMES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    history = []
    best_val_loss = None
    best_epoch = None
    epochs_without_improvement = 0
    show_progress = not args.no_progress
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, device, args, class_weights, dice_classes, epoch, "train", show_progress)
        val_loss = None
        val_metrics = {}
        if val_loader is not None:
            val_loss, val_metrics = run_epoch(model, val_loader, None, device, args, class_weights, dice_classes, epoch, "val", show_progress)
        external_loss = None
        external_metrics = {}
        if external_monitor_loader is not None:
            external_loss, external_metrics = run_epoch(
                model,
                external_monitor_loader,
                None,
                device,
                args,
                class_weights,
                dice_classes,
                epoch,
                args.external_monitor_name,
                show_progress,
            )
        if args.best_monitor == "external_loss":
            monitor = external_loss if external_loss is not None else train_loss
        elif args.best_monitor == "val_recon_loss":
            monitor = val_metrics.get("recon_loss", val_loss if val_loss is not None else train_loss)
        elif args.best_monitor == "external_recon_loss":
            monitor = external_metrics.get("recon_loss", external_loss if external_loss is not None else train_loss)
        else:
            monitor = val_loss if val_loss is not None else train_loss
        is_best = args.save_best and (best_val_loss is None or monitor < best_val_loss - float(args.min_delta))
        if is_best:
            best_val_loss = monitor
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if scheduler is not None:
            scheduler.step(monitor)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **prefixed_metrics(str(args.external_monitor_name), external_loss, external_metrics),
        }
        history.append(row)
        marker = " *" if is_best else ""
        external_text = ""
        if external_loss is not None:
            external_text = (
                f" {args.external_monitor_name}_loss={external_loss} "
                f"{args.external_monitor_name}_mfg_iou={external_metrics.get('mean_fg_iou', float('nan')):.4f} "
                f"{args.external_monitor_name}_landmark_dice={external_metrics.get('dice_class2', float('nan')):.4f}"
            )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"val_loss={val_loss if val_loss is not None else 'n/a'} "
            f"val_mfg_iou={val_metrics.get('mean_fg_iou', float('nan')):.4f} "
            f"val_landmark_dice={val_metrics.get('dice_class2', float('nan')):.4f} "
            f"val_recon_mfg_iou={val_metrics.get('recon_mean_fg_iou', float('nan')):.4f} "
            f"val_recon_landmark_dice={val_metrics.get('recon_dice_class2', float('nan')):.4f} "
            f"val_recon_loss={val_metrics.get('recon_loss', float('nan')):.5f}"
            f"{external_text}{marker}"
        )
        save_checkpoint(run_dir, model, optimizer, args, history, best_epoch, best_val_loss, epoch, save_as_best=is_best)
        if args.early_stopping_patience is not None and epochs_without_improvement >= int(args.early_stopping_patience):
            print(f"Early stopping at epoch={epoch:03d} after {epochs_without_improvement} epochs without improvement.")
            break

    if args.save_previews and val_dataset is not None:
        save_previews(run_dir, model, val_dataset, device, args.preview_count, preview_head=args.preview_head)
    if args.save_previews and external_monitor_dataset is not None:
        save_previews(
            run_dir / args.external_monitor_name,
            model,
            external_monitor_dataset,
            device,
            args.preview_count,
            preview_head=args.preview_head,
        )
    print(json.dumps({"saved": str(run_dir), "epochs": len(history), "best_epoch": best_epoch, "best_val_loss": best_val_loss}, indent=2))


def train_from_args(args: argparse.Namespace) -> None:
    """Run training from an already parsed argparse namespace."""
    global parse_args
    original_parse_args = parse_args
    parse_args = lambda: args  # type: ignore[assignment]
    try:
        main()
    finally:
        parse_args = original_parse_args


if __name__ == "__main__":
    main()

