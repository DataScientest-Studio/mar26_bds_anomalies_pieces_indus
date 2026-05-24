"""Masking and scoring-region helpers."""

from __future__ import annotations

import argparse

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def denormalize_for_masking(images: torch.Tensor, normalization: str) -> torch.Tensor:
    if normalization == "ae":
        return (images * 0.5 + 0.5).clamp(0.0, 1.0)
    if normalization == "imagenet":
        mean = torch.as_tensor(IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.as_tensor(IMAGENET_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        return (images * std + mean).clamp(0.0, 1.0)
    raise ValueError(normalization)


def object_bbox_from_sample(
    sample_rgb: torch.Tensor,
    *,
    threshold: float,
) -> tuple[int, int, int, int] | None:
    gray = sample_rgb.mean(dim=0)
    ys, xs = torch.where(gray > threshold)
    if ys.numel() == 0 or xs.numel() == 0:
        return None
    height, width = gray.shape
    x0 = max(0, int(xs.min().item()))
    x1 = min(width, int(xs.max().item()) + 1)
    y0 = max(0, int(ys.min().item()))
    y1 = min(height, int(ys.max().item()) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def mask_sampling_region(
    sample_rgb: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[int, int, int, int]:
    _channels, height, width = sample_rgb.shape
    full_region = (0, 0, width, height)
    if args.mask_sampling == "uniform":
        return full_region

    bbox = object_bbox_from_sample(
        sample_rgb,
        threshold=float(args.object_threshold),
    )
    if bbox is None:
        return full_region
    if args.mask_sampling == "object_bbox":
        return bbox

    x0, y0, x1, y1 = bbox
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    hx0 = x0 + int(round(box_w * float(args.head_x_min)))
    hx1 = x0 + int(round(box_w * float(args.head_x_max)))
    hy0 = y0 + int(round(box_h * float(args.head_y_min)))
    hy1 = y0 + int(round(box_h * float(args.head_y_max)))
    hx0 = max(0, min(width - 1, hx0))
    hy0 = max(0, min(height - 1, hy0))
    hx1 = max(hx0 + 1, min(width, hx1))
    hy1 = max(hy0 + 1, min(height, hy1))
    return hx0, hy0, hx1, hy1

