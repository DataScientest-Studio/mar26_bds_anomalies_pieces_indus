"""
Image transforms pour l'extraction d'embeddings.

Deux modes :
- RGB + normalisation ImageNet (standard)
- Grayscale effectif : convertit en 1 canal, duplique sur 3, normalise avec Î¼ unique
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from torchvision import transforms

from src.config import DATA, FEATURES
from src.features.augmentation_profiles import AugmentationProfile


def get_transform(
    size: int = FEATURES.default_input_size,
    grayscale: bool = False,
    mean: tuple | None = None,
    std: tuple | None = None,
) -> transforms.Compose:
    """Construit la chaÃ®ne de transforms pour un batch d'images.

    Parameters
    ----------
    size : int
        Taille d'entrÃ©e carrÃ©e (ex: 224, 384).
    grayscale : bool
        Si True, convertit l'image en L puis duplique sur 3 canaux.
    mean, std : tuple de 3 floats, optionnel
        ParamÃ¨tres de normalisation. Par dÃ©faut :
        - RGB : ImageNet (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        - Grayscale : Î¼=Ïƒ=0.5 dupliquÃ© sur 3 canaux
    """
    if grayscale:
        mean = mean or (0.5, 0.5, 0.5)
        std = std or (0.5, 0.5, 0.5)
        pre = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((size, size)),
        ]
    else:
        mean = mean or FEATURES.imagenet_mean
        std = std or FEATURES.imagenet_std
        pre = [transforms.Resize((size, size))]

    return transforms.Compose([
        *pre,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def resolve_repeat_factor(raw: str, profile: AugmentationProfile) -> int:
    if str(raw).lower() == "auto":
        return max(1, int(profile.repeat_factor))
    return max(1, int(raw))


def apply_flat_lighting(image: Image.Image, profile: AugmentationProfile | None) -> Image.Image:
    """Simulate lifted blacks and compressed contrast for flat-lit Casting views."""
    if profile is None or float(getattr(profile, "flat_lighting_p", 0.0)) <= 0:
        return image
    if random.random() >= float(profile.flat_lighting_p):
        return image
    contrast_min, contrast_max = getattr(profile, "flat_lighting_contrast", (1.0, 1.0))
    lift_min, lift_max = getattr(profile, "flat_lighting_lift", (0.0, 0.0))
    contrast = random.uniform(float(contrast_min), float(contrast_max))
    lift = random.uniform(float(lift_min), float(lift_max))
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = np.clip(lift + contrast * arr, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")


def percentile_normalize_image(image: Image.Image, *, target_p05: float, target_p95: float) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    p05 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    denom = max(p95 - p05, 1e-6)
    normalized = (arr - p05) / denom
    normalized = normalized * (float(target_p95) - float(target_p05)) + float(target_p05)
    normalized = np.clip(normalized, 0.0, 1.0)
    return Image.fromarray((normalized * 255).astype(np.uint8), mode="RGB")


def image_luminance_descriptors(image: Image.Image) -> dict[str, float]:
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    small = np.asarray(gray.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    gx = np.diff(small, axis=1, append=small[:, -1:])
    gy = np.diff(small, axis=0, append=small[-1:, :])
    edge = np.sqrt(gx * gx + gy * gy)
    p05 = float(np.quantile(arr, 0.05))
    p95 = float(np.quantile(arr, 0.95))
    return {
        "lum_mean": float(arr.mean()),
        "lum_std": float(arr.std()),
        "lum_p05": p05,
        "lum_p95": p95,
        "dynamic_p95_p05": float(p95 - p05),
        "edge_mean": float(edge.mean()),
    }


def apply_photometric_normalization(
    image: Image.Image,
    args,
) -> tuple[Image.Image, dict[str, float | bool | str]]:
    mode = str(args.photometric_normalization)
    descriptors = image_luminance_descriptors(image)
    applied = False
    if mode == "percentile":
        applied = True
    elif mode == "conditional":
        applied = (
            descriptors["lum_mean"] >= float(args.photo_condition_lum_mean_min)
            and descriptors["lum_std"] <= float(args.photo_condition_lum_std_max)
            and descriptors["dynamic_p95_p05"] <= float(args.photo_condition_dynamic_max)
            and descriptors["edge_mean"] <= float(args.photo_condition_edge_max)
        )
    elif mode != "none":
        raise ValueError(f"Unsupported photometric normalization mode: {mode}")

    if not applied:
        return image, {"photometric_normalization": mode, "photometric_applied": False, **descriptors}
    normalized = percentile_normalize_image(
        image,
        target_p05=float(args.photo_target_p05),
        target_p95=float(args.photo_target_p95),
    )
    normalized_descriptors = image_luminance_descriptors(normalized)
    return normalized, {
        "photometric_normalization": mode,
        "photometric_applied": True,
        **descriptors,
        **{f"normalized_{key}": value for key, value in normalized_descriptors.items()},
    }


def resize_letterbox_pil(image: Image.Image, size: int, *, mode: str, fill: int = 0) -> Image.Image:
    if mode == "RGB":
        image = image.convert("RGB")
        resample = Image.Resampling.BILINEAR
        canvas_mode = "RGB"
        canvas_fill: int | tuple[int, int, int] = (fill, fill, fill)
    else:
        image = image.convert("L")
        resample = Image.Resampling.BILINEAR if mode == "soft" else Image.Resampling.NEAREST
        canvas_mode = "L"
        canvas_fill = fill
    width, height = image.size
    scale = int(size) / max(width, height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = image.resize((new_width, new_height), resample=resample)
    canvas = Image.new(canvas_mode, (int(size), int(size)), canvas_fill)
    left = (int(size) - new_width) // 2
    top = (int(size) - new_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def crop_box_to_mask(box: tuple[float, float, float, float], size: int) -> Image.Image:
    cx, cy, bw, bh = box
    x0 = max(0, min(int(round((cx - bw * 0.5) * size)), size))
    x1 = max(0, min(int(round((cx + bw * 0.5) * size)), size))
    y0 = max(0, min(int(round((cy - bh * 0.5) * size)), size))
    y1 = max(0, min(int(round((cy + bh * 0.5) * size)), size))
    mask = Image.new("L", (size, size), 0)
    if x1 > x0 and y1 > y0:
        draw = ImageDraw.Draw(mask)
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=255)
    return mask


def resolve_split_column(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise KeyError(f"Split column {requested!r} not found in labels_index.csv.")
        return requested
    for column in ("pattern_id", "base_pattern", "meta_pattern"):
        if column in df.columns:
            return column
    raise KeyError("No stratification column found. Use --split-column with an existing labels_index.csv column.")


def split_df(
    df: pd.DataFrame,
    val_fraction: float,
    *,
    strategy: str = "random",
    split_column: str = "auto",
    seed: int = DATA.random_seed,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    if val_fraction <= 0 or len(df) < 2:
        return df.reset_index(drop=True), None, {"strategy": strategy, "val_fraction": float(val_fraction)}
    if strategy == "stratified":
        column = resolve_split_column(df, split_column)
        rng = np.random.default_rng(int(seed))
        val_indices: list[int] = []
        group_counts: dict[str, dict[str, int]] = {}
        for group_value, group in df.groupby(column, dropna=False, sort=True):
            indices = group.index.to_numpy()
            rng.shuffle(indices)
            if len(indices) <= 1:
                val_count = 0
            else:
                val_count = int(round(len(indices) * float(val_fraction)))
                val_count = max(1, min(val_count, len(indices) - 1))
            selected = indices[:val_count].tolist()
            val_indices.extend(selected)
            group_counts[str(group_value)] = {
                "total": int(len(indices)),
                "train": int(len(indices) - val_count),
                "val": int(val_count),
            }
        if not val_indices:
            raise ValueError("Stratified split produced an empty validation set. Increase --val-fraction or dataset size.")
        val = df.loc[sorted(val_indices)]
        train = df.drop(val.index)
        info = {
            "strategy": "stratified",
            "split_column": column,
            "split_seed": int(seed),
            "val_fraction": float(val_fraction),
            "group_counts": group_counts,
        }
        return train.reset_index(drop=True), val.reset_index(drop=True), info
    val = df.sample(frac=val_fraction, random_state=int(seed))
    train = df.drop(val.index)
    return train.reset_index(drop=True), val.reset_index(drop=True), {
        "strategy": "random",
        "split_seed": int(seed),
        "val_fraction": float(val_fraction),
        "train_count": int(len(train)),
        "val_count": int(len(val)),
    }





