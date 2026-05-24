"""Reusable reconstruction-AE runtime helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision import transforms

from src.config import EDA
from src.models.baselines.patchcore import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ResizeLetterbox,
    _normalized_low_fpr_aupimo,
    load_unified_dataset,
    project_path,
    split_category_data,
)
from src.features.augmentation_profiles import AugmentationProfile

__all__ = [
    "build_tile_transform",
    "build_pixel_ae_transform",
    "evaluate_variable_predictions",
    "load_native_mask",
    "load_training_data",
    "maybe_limit",
    "repeat_training_rows",
    "run_pixel_ae_reconstruction",
    "resolve_repeat_factor",
    "split_train_val",
]


def build_pixel_ae_transform(
    input_size: int,
    *,
    phase: str,
    normalization: str,
    augmentation_policy: str,
    augmentation_profile: AugmentationProfile | None = None,
) -> transforms.Compose:
    if normalization == "ae":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif normalization == "imagenet":
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    else:
        raise ValueError(normalization)

    if phase == "train" and augmentation_profile is not None and augmentation_profile.name != "none":
        spatial = [
            ResizeLetterbox(input_size),
            transforms.RandomResizedCrop(
                input_size,
                scale=augmentation_profile.scale,
                ratio=augmentation_profile.ratio,
            ),
        ]
        if augmentation_profile.horizontal_flip_p > 0:
            spatial.append(transforms.RandomHorizontalFlip(p=augmentation_profile.horizontal_flip_p))
        if augmentation_profile.vertical_flip_p > 0:
            spatial.append(transforms.RandomVerticalFlip(p=augmentation_profile.vertical_flip_p))
        if augmentation_profile.rotation_degrees > 0:
            spatial.append(transforms.RandomRotation(degrees=augmentation_profile.rotation_degrees, fill=0))
        if any(
            value > 0
            for value in (
                augmentation_profile.brightness,
                augmentation_profile.contrast,
                augmentation_profile.saturation,
                abs(augmentation_profile.hue),
            )
        ):
            spatial.append(
                transforms.ColorJitter(
                    brightness=augmentation_profile.brightness,
                    contrast=augmentation_profile.contrast,
                    saturation=augmentation_profile.saturation,
                    hue=augmentation_profile.hue,
                )
            )
        if augmentation_profile.blur_p > 0:
            spatial.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3)],
                    p=augmentation_profile.blur_p,
                )
            )
        spatial_transform = transforms.Compose(spatial)
    elif phase == "train" and augmentation_policy == "random-resized-crop":
        spatial_transform = transforms.RandomResizedCrop(
            input_size,
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
        )
    else:
        spatial_transform = ResizeLetterbox(input_size)

    return transforms.Compose(
        [
            spatial_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_tile_transform(normalization: str) -> transforms.Compose:
    if normalization == "ae":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif normalization == "imagenet":
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    else:
        raise ValueError(normalization)
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def maybe_limit(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def resolve_repeat_factor(raw: str, profile: AugmentationProfile) -> int:
    if raw == "auto":
        return int(profile.repeat_factor)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("--repeat-factor must be 'auto' or a positive integer.") from exc
    if value < 1:
        raise ValueError("--repeat-factor must be >= 1.")
    return value


def repeat_training_rows(df: pd.DataFrame, repeat_factor: int) -> pd.DataFrame:
    if repeat_factor <= 1:
        return df.reset_index(drop=True)
    return pd.concat([df] * repeat_factor, ignore_index=True)


def load_training_data(args) -> pd.DataFrame:
    if args.all_categories:
        df = load_unified_dataset()
        train_df = df[(df["split"] == "train") & (~df["is_anomaly"])].copy()
    else:
        if args.category is None:
            raise ValueError("Use --category for category training or --all-categories.")
        train_df, _test_df = split_category_data(args.category)

    if train_df.empty:
        raise ValueError("No normal training images found.")
    if train_df["is_anomaly"].any():
        raise RuntimeError("Training data contains anomalies; this violates normal-only training.")
    return train_df.reset_index(drop=True)


def split_train_val(
    train_df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not 0 <= val_fraction < 1:
        raise ValueError("--val-fraction must be in [0, 1).")
    if val_fraction == 0 or len(train_df) < 2:
        return train_df.reset_index(drop=True), None

    val_count = max(1, int(round(len(train_df) * val_fraction)))
    val_count = min(val_count, len(train_df) - 1)
    val_df = train_df.sample(n=val_count, random_state=seed)
    fit_df = train_df.drop(index=val_df.index)
    return fit_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_native_mask(row: pd.Series, original_size: tuple[int, int]) -> np.ndarray:
    width, height = original_size
    if bool(row["has_mask"]) and pd.notna(row["mask_path"]):
        mask = Image.open(project_path(row["mask_path"])).convert("L")
        if mask.size != original_size:
            mask = mask.resize(original_size, resample=Image.Resampling.NEAREST)
        return (np.asarray(mask) > EDA.mask_threshold).astype(np.uint8)
    return np.zeros((height, width), dtype=np.uint8)


def is_mask_conditioned_model(model: nn.Module) -> bool:
    return bool(getattr(model, "mask_conditioned", False))


def run_pixel_ae_reconstruction(
    model: nn.Module,
    images: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if is_mask_conditioned_model(model):
        return model(images, mask)
    return model(images)


def evaluate_variable_predictions(predictions: dict) -> dict[str, float]:
    y_true = predictions["y_true"]
    image_score = predictions["image_score"]
    metrics: dict[str, float] = {}
    if len(np.unique(y_true)) == 2:
        metrics["image_auroc"] = float(roc_auc_score(y_true, image_score))
        metrics["image_ap"] = float(average_precision_score(y_true, image_score))
    if "mask" in predictions and "score_maps" in predictions:
        pixel_true = np.asarray(predictions["mask"]).reshape(-1).astype(np.uint8)
        pixel_score = np.asarray(predictions["score_maps"]).reshape(-1).astype(np.float32)
        if len(np.unique(pixel_true)) == 2:
            metrics["pixel_auroc"] = float(roc_auc_score(pixel_true, pixel_score))
            metrics["pixel_ap"] = float(average_precision_score(pixel_true, pixel_score))
            aupimo = _normalized_low_fpr_aupimo(
                np.asarray(y_true).astype(np.int64),
                np.asarray(predictions["mask"]),
                np.asarray(predictions["score_maps"]).astype(np.float32),
                fpr_low=1e-5,
                fpr_high=1e-3,
            )
            if aupimo is not None:
                metrics["pixel_aupimo_1e-5_1e-3"] = float(aupimo)
    return metrics





