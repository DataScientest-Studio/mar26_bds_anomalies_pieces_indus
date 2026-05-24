"""Evaluate a Feature-AE as a complete anomaly detector."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from tqdm.auto import tqdm

from src.config import EDA, PATHS
from src.features.functional_surface import load_functional_predictions
from src.features.masking import denormalize_for_masking, mask_sampling_region
from src.features.tiling import reconstruct_score_map, safe_image_id, tile_image
from src.models.feature_ae.models import (
    ResNetTeacherFeatures,
    build_feature_autoencoder,
    feature_error_map,
)
from src.models.baselines.patchcore import (
    UnifiedAnomalyDataset,
    evaluate_predictions,
    make_dataloader,
    project_path,
    resolve_device,
    split_category_data,
)
from src.models.pixel_ae.runtime import (
    build_tile_transform,
    evaluate_variable_predictions,
    load_native_mask,
)
from src.models.pixel_ae.runtime import build_pixel_ae_transform
from src.visualization.heatmaps import error_to_heatmap, rgb_array_to_image
from src.models.feature_ae.evaluation_config import parse_args




def maybe_limit(df: pd.DataFrame, n: int | None, seed: int = 42) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def checkpoint_layers(checkpoint: dict, args: argparse.Namespace) -> list[str]:
    return list(args.layers or checkpoint.get("layers") or ["layer2", "layer3"])


def checkpoint_teacher(checkpoint: dict, args: argparse.Namespace) -> str:
    return str(args.teacher_backbone or checkpoint.get("teacher_backbone") or "resnet18")


def checkpoint_cosine_weight(checkpoint: dict, args: argparse.Namespace) -> float:
    if args.cosine_weight is not None:
        return float(args.cosine_weight)
    return float(checkpoint.get("cosine_weight", 0.5))


def build_run_name(args: argparse.Namespace, layers: list[str]) -> str:
    if args.run_name:
        return args.run_name
    layer_tag = "-".join(layers)
    name = f"{args.category}_feature_ae_resnet18_layers-{layer_tag}"
    if args.calibrate_normal:
        name += "_calibrated"
    name += f"_{args.preprocessing_mode}"
    return name


def parse_layer_weights(raw: list[str] | None, layers: list[str]) -> dict[str, float]:
    if raw is None or raw == [] or raw == ["auto"]:
        return {layer: 1.0 / len(layers) for layer in layers}
    weights: dict[str, float] = {}
    for item in raw:
        if item == "auto":
            if len(raw) > 1:
                raise ValueError("--layer-weights auto cannot be mixed with explicit weights.")
            return {layer: 1.0 / len(layers) for layer in layers}
        if "=" not in item:
            raise ValueError("--layer-weights expects entries like layer2=0.7.")
        layer_name, value_text = item.split("=", 1)
        if layer_name not in layers:
            raise ValueError(f"Layer weight provided for absent layer: {layer_name!r}.")
        value = float(value_text)
        if value < 0:
            raise ValueError("--layer-weights values must be >= 0.")
        weights[layer_name] = value
    missing = sorted(set(layers) - set(weights))
    if missing:
        raise ValueError(f"Missing layer weights for: {missing}.")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("--layer-weights must sum to a positive value.")
    return {layer: value / total for layer, value in weights.items()}


def score_image(score_map: np.ndarray, args: argparse.Namespace, roi_mask: np.ndarray | None = None) -> float:
    array = np.asarray(score_map, dtype=np.float32)
    if roi_mask is not None:
        valid = np.asarray(roi_mask) > 0
        values = array[valid] if np.any(valid) else array.reshape(-1)
    else:
        values = array.reshape(-1)
    if args.score_image == "max":
        return float(values.max(initial=0.0))
    if args.score_image == "topk_mean":
        k = max(1, int(round(len(values) * float(args.topk_fraction))))
        return float(np.partition(values, len(values) - k)[-k:].mean())
    return float(np.percentile(values, 99))


def _resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(np.float32, copy=False)
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
    resized = image.resize((target_shape[1], target_shape[0]), resample=Image.Resampling.NEAREST)
    return (np.asarray(resized) > 0).astype(np.float32)


def resize_tile_for_model(tile: Image.Image, input_size: int) -> Image.Image:
    if tile.size == (int(input_size), int(input_size)):
        return tile
    return tile.resize((int(input_size), int(input_size)), resample=Image.Resampling.BILINEAR)


def resize_score_map(score_map: np.ndarray, output_size: int) -> np.ndarray:
    arr = np.asarray(score_map, dtype=np.float32)
    if arr.shape == (int(output_size), int(output_size)):
        return arr.astype(np.float32, copy=False)
    image = Image.fromarray(arr, mode="F")
    resized = image.resize((int(output_size), int(output_size)), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def crop_centered_context_tile(image: Image.Image, spec, context_tile_size: int) -> Image.Image:
    size = int(context_tile_size)
    center_x = (int(spec.x0) + int(spec.x1)) // 2
    center_y = (int(spec.y0) + int(spec.y1)) // 2
    x0 = center_x - size // 2
    y0 = center_y - size // 2
    x1 = x0 + size
    y1 = y0 + size
    canvas = Image.new("RGB", (size, size), 0)
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(image.width, x1)
    src_y1 = min(image.height, y1)
    if src_x1 > src_x0 and src_y1 > src_y0:
        canvas.paste(image.crop((src_x0, src_y0, src_x1, src_y1)), (src_x0 - x0, src_y0 - y0))
    return canvas


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PATHS.root / path


def dilate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if int(radius) <= 0:
        return (np.asarray(mask) > 0).astype(np.float32)
    image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L")
    image = image.filter(ImageFilter.MaxFilter(int(radius) * 2 + 1))
    return (np.asarray(image) > 0).astype(np.float32)


def build_functional_roi_lookup(args: argparse.Namespace) -> dict[str, np.ndarray]:
    dirs = args.roi_predictions_dir or []
    lookup: dict[str, np.ndarray] = {}
    for raw_dir in dirs:
        predictions = load_functional_predictions(resolve_path(raw_dir))
        for image_path, prob_map in zip(predictions["image_path"], predictions["prob_maps"], strict=True):
            binary = np.asarray(prob_map, dtype=np.float32) >= float(args.roi_threshold)
            lookup[str(image_path)] = dilate_binary_mask(binary, int(args.roi_dilate_radius))
    return lookup


def casting_surface_mask_from_rgb(rgb: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Keep the casting surface and remove only dark exterior background connected to borders."""
    image = np.asarray(rgb, dtype=np.float32)
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0
    gray = image.mean(axis=2)
    exterior_candidate = gray <= float(args.object_threshold)
    height, width = exterior_candidate.shape
    exterior = np.zeros((height, width), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        if exterior_candidate[0, x]:
            queue.append((0, x))
        if exterior_candidate[height - 1, x]:
            queue.append((height - 1, x))
    for y in range(height):
        if exterior_candidate[y, 0]:
            queue.append((y, 0))
        if exterior_candidate[y, width - 1]:
            queue.append((y, width - 1))
    while queue:
        y, x = queue.popleft()
        if exterior[y, x] or not exterior_candidate[y, x]:
            continue
        exterior[y, x] = True
        if y > 0:
            queue.append((y - 1, x))
        if y + 1 < height:
            queue.append((y + 1, x))
        if x > 0:
            queue.append((y, x - 1))
        if x + 1 < width:
            queue.append((y, x + 1))
    mask = (~exterior).astype(np.float32)
    if args.score_region == "casting_surface_margin":
        kernel = max(3, int(round(min(height, width) * float(args.roi_margin))))
        if kernel % 2 == 0:
            kernel += 1
        tensor = torch.from_numpy(mask)[None, None]
        padded = F.pad(tensor, (kernel // 2,) * 4, mode="replicate")
        mask = F.max_pool2d(padded, kernel_size=kernel, stride=1)[0, 0].numpy().astype(np.float32)
    return mask


def roi_mask_from_image_path(image_path: str, target_shape: tuple[int, int], args: argparse.Namespace) -> np.ndarray:
    if args.score_region == "full":
        return np.ones(target_shape, dtype=np.float32)
    if args.score_region in {"functional_surface_prediction", "functional_surface_prediction_margin"}:
        lookup = getattr(args, "functional_roi_lookup", {})
        if image_path not in lookup:
            raise KeyError(
                f"Missing functional-surface ROI for {image_path!r}. "
                "Pass ROI prediction dirs covering all evaluated images."
            )
        roi = _resize_mask(np.asarray(lookup[image_path], dtype=np.float32), target_shape)
        if args.score_region == "functional_surface_prediction_margin":
            radius = max(1, int(round(min(target_shape) * float(args.roi_margin))))
            roi = dilate_binary_mask(roi, radius)
        return roi.astype(np.float32, copy=False)
    if args.score_region in {"casting_surface", "casting_surface_margin"}:
        image = Image.open(project_path(str(image_path))).convert("RGB")
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        return _resize_mask(casting_surface_mask_from_rgb(rgb, args), target_shape)
    raise ValueError(f"ROI from image path is not implemented for score-region={args.score_region!r}.")


def build_roi_masks_from_paths(predictions: dict, args: argparse.Namespace) -> np.ndarray | None:
    if args.score_region == "full":
        return None
    score_maps = predictions["score_maps"]
    masks = []
    for path, score_map in zip(predictions["image_path"], score_maps, strict=True):
        target_shape = tuple(np.asarray(score_map).shape[-2:])
        masks.append(roi_mask_from_image_path(str(path), target_shape, args))
    return np.stack(masks, axis=0).astype(np.float32, copy=False)


def smooth_score_map(score_map: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.asarray(score_map, dtype=np.float32)
    tensor = torch.from_numpy(np.asarray(score_map, dtype=np.float32))[None, None]
    if mode == "median3":
        padded = F.pad(tensor, (1, 1, 1, 1), mode="reflect")
        unfolded = F.unfold(padded, kernel_size=3).view(1, 9, *score_map.shape[-2:])
        return unfolded.median(dim=1).values[0].numpy().astype(np.float32, copy=False)
    if mode == "gaussian":
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        kernel = (kernel / kernel.sum()).view(1, 1, 3, 3)
        padded = F.pad(tensor, (1, 1, 1, 1), mode="reflect")
        return F.conv2d(padded, kernel)[0, 0].numpy().astype(np.float32, copy=False)
    raise ValueError(f"Unsupported score smoothing: {mode}")


def apply_score_postprocessing(predictions: dict, args: argparse.Namespace) -> dict:
    processed = dict(predictions)
    maps = np.asarray(processed["score_maps"], dtype=np.float32)
    if args.score_region != "full":
        roi_masks = processed.get("roi_masks")
        if roi_masks is None or len(roi_masks) != len(maps):
            roi_masks = build_roi_masks_from_paths(processed, args)
        processed["roi_masks"] = roi_masks
    else:
        roi_masks = None

    should_keep_raw = args.score_smoothing != "none" or args.apply_score_region_to_map
    if should_keep_raw and "raw_score_maps" not in processed:
        processed["raw_score_maps"] = maps.astype(np.float32, copy=False)

    if args.score_smoothing != "none":
        maps = np.stack([smooth_score_map(item, args.score_smoothing) for item in maps], axis=0)

    if args.apply_score_region_to_map and roi_masks is not None:
        maps = maps * np.asarray(roi_masks, dtype=np.float32)

    processed["score_maps"] = maps.astype(np.float32, copy=False)
    processed["image_score"] = np.array(
        [
            score_image(score_map, args, roi if roi_masks is not None else None)
            for score_map, roi in zip(maps, roi_masks if roi_masks is not None else [None] * len(maps), strict=True)
        ],
        dtype=np.float32,
    )
    return processed


def roi_masks_from_batch(images: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.score_region == "full":
        return torch.ones((images.shape[0], images.shape[2], images.shape[3]), device=images.device, dtype=images.dtype)
    if args.score_region in {"functional_surface_prediction", "functional_surface_prediction_margin"}:
        raise ValueError("functional_surface_prediction ROI is supported for tiled evaluation, not letterbox batches.")
    if args.score_region in {"casting_surface", "casting_surface_margin"}:
        denorm = denormalize_for_masking(images, "imagenet")
        masks = []
        for index in range(images.shape[0]):
            rgb = denorm[index].permute(1, 2, 0).detach().cpu().numpy()
            masks.append(casting_surface_mask_from_rgb(rgb, args))
        return torch.from_numpy(np.stack(masks, axis=0)).to(device=images.device, dtype=images.dtype)
    original_sampling = getattr(args, "mask_sampling", "uniform")
    args.normalization = "imagenet"
    if args.score_region in {"object_bbox", "object_bbox_margin"}:
        args.mask_sampling = "object_bbox"
    else:
        args.mask_sampling = "toothbrush_head"
    try:
        denorm = denormalize_for_masking(images, "imagenet")
        masks = torch.zeros((images.shape[0], images.shape[2], images.shape[3]), device=images.device, dtype=images.dtype)
        for index in range(images.shape[0]):
            x0, y0, x1, y1 = mask_sampling_region(denorm[index], args)
            if args.score_region.endswith("_margin"):
                width = images.shape[3]
                height = images.shape[2]
                margin_x = int(round((x1 - x0) * float(args.roi_margin)))
                margin_y = int(round((y1 - y0) * float(args.roi_margin)))
                x0 = max(0, x0 - margin_x)
                y0 = max(0, y0 - margin_y)
                x1 = min(width, x1 + margin_x)
                y1 = min(height, y1 + margin_y)
            masks[index, y0:y1, x0:x1] = 1.0
        return masks
    finally:
        args.mask_sampling = original_sampling


def normalize_layer_map(score_map: torch.Tensor) -> torch.Tensor:
    flat = score_map.flatten(start_dim=1)
    high = torch.quantile(flat, 0.99, dim=1).view(-1, 1, 1)
    return score_map / high.clamp_min(1e-6)


@torch.inference_mode()
def batch_score_maps(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    images: torch.Tensor,
    context_images: torch.Tensor | None = None,
    layers: list[str],
    input_size: int,
    cosine_weight: float,
    layer_weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    teacher_features = teacher(images)
    predicted_features = student(images, context_images) if context_images is not None else student(images)
    layer_maps: dict[str, torch.Tensor] = {}
    normalized_maps = []
    for layer_name in layers:
        layer_score = feature_error_map(
            predicted_features[layer_name],
            teacher_features[layer_name],
            cosine_weight=cosine_weight,
        )
        upsampled = F.interpolate(
            layer_score[:, None, :, :],
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        layer_maps[layer_name] = upsampled
        normalized_maps.append(normalize_layer_map(upsampled))
    weighted_maps = [
        normalized * float(layer_weights[layer_name])
        for normalized, layer_name in zip(normalized_maps, layers, strict=True)
    ]
    fused = torch.stack(weighted_maps, dim=0).sum(dim=0)
    return fused, layer_maps


@torch.inference_mode()
def predict_letterbox(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader,
    device: torch.device,
    layers: list[str],
    input_size: int,
    cosine_weight: float,
    layer_weights: dict[str, float],
    args: argparse.Namespace,
) -> dict:
    image_paths: list[str] = []
    y_true: list[int] = []
    image_scores: list[float] = []
    score_maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    roi_masks: list[np.ndarray] = []
    layer_score_maps = {layer: [] for layer in layers}

    iterator = tqdm(
        loader,
        desc="feature AE eval",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        fused, layer_maps = batch_score_maps(
            student=student,
            teacher=teacher,
            images=images,
            context_images=None,
            layers=layers,
            input_size=input_size,
            cosine_weight=cosine_weight,
            layer_weights=layer_weights,
        )
        roi = roi_masks_from_batch(images, args)
        fused_np = fused.cpu().numpy().astype(np.float32, copy=False)
        roi_np = roi.cpu().numpy().astype(np.float32, copy=False)
        score_maps.extend([item for item in fused_np])
        roi_masks.extend([item for item in roi_np])
        image_scores.extend(
            [
                score_image(item, args, roi_mask if args.score_region != "full" else None)
                for item, roi_mask in zip(fused_np, roi_np, strict=True)
            ]
        )
        for layer_name, layer_map in layer_maps.items():
            layer_np = layer_map.cpu().numpy().astype(np.float32, copy=False)
            layer_score_maps[layer_name].extend([item for item in layer_np])
        y_true.extend(batch["is_anomaly"].cpu().numpy().astype(int).tolist())
        image_paths.extend(batch["image_path"])
        if "mask" in batch:
            masks.extend(batch["mask"].cpu().numpy().astype(np.uint8))

    return {
        "image_path": np.array(image_paths, dtype=object),
        "y_true": np.array(y_true, dtype=np.int64),
        "image_score": np.array(image_scores, dtype=np.float32),
        "score_maps": np.stack(score_maps, axis=0) if score_maps else None,
        "masks": np.stack(masks, axis=0) if masks else None,
        "roi_masks": np.stack(roi_masks, axis=0) if roi_masks else None,
        "layer_score_maps": {
            layer: np.stack(values, axis=0) for layer, values in layer_score_maps.items()
        },
    }


@torch.inference_mode()
def predict_tiled(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    layers: list[str],
    cosine_weight: float,
    layer_weights: dict[str, float],
    args: argparse.Namespace,
) -> dict:
    transform = build_tile_transform("imagenet")
    image_paths: list[str] = []
    y_true: list[int] = []
    image_scores: list[float] = []
    score_maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    layer_score_maps = {layer: [] for layer in layers}

    iterator = tqdm(
        list(df.iterrows()),
        desc="feature AE tiled eval",
        unit="image",
        leave=False,
        dynamic_ncols=True,
        disable=args.no_progress,
    )
    for _idx, row in iterator:
        image = Image.open(project_path(row["image_path"])).convert("RGB")
        tiled = tile_image(image, tile_size=args.tile_size, stride=args.tile_stride)
        tile_maps: list[np.ndarray] = []
        layer_tile_maps: dict[str, list[np.ndarray]] = {layer: [] for layer in layers}
        for start in range(0, len(tiled.tiles), args.batch_size):
            batch_tiles = tiled.tiles[start : start + args.batch_size]
            batch_specs = tiled.specs[start : start + args.batch_size]
            images = torch.stack(
                [transform(resize_tile_for_model(tile, args.input_size)) for tile in batch_tiles]
            ).to(device)
            context_images = None
            if args.context_tile_size is not None:
                context_images = torch.stack(
                    [
                        transform(
                            resize_tile_for_model(
                                crop_centered_context_tile(image, spec, args.context_tile_size),
                                args.input_size,
                            )
                        )
                        for spec in batch_specs
                    ]
                ).to(device)
            fused, layer_maps = batch_score_maps(
                student=student,
                teacher=teacher,
                images=images,
                context_images=context_images,
                layers=layers,
                input_size=args.input_size,
                cosine_weight=cosine_weight,
                layer_weights=layer_weights,
            )
            tile_maps.extend(
                resize_score_map(item, args.tile_size)
                for item in fused.cpu().numpy().astype(np.float32, copy=False)
            )
            for layer_name, layer_map in layer_maps.items():
                layer_tile_maps[layer_name].extend(
                    resize_score_map(item, args.tile_size)
                    for item in layer_map.cpu().numpy().astype(np.float32, copy=False)
                )
        score_map = reconstruct_score_map(tile_maps, tiled, aggregation=args.tile_aggregation)
        for layer_name in layers:
            layer_score_maps[layer_name].append(
                reconstruct_score_map(
                    layer_tile_maps[layer_name],
                    tiled,
                    aggregation=args.tile_aggregation,
                )
            )
        mask = load_native_mask(row, tiled.original_size)
        image_paths.append(str(row["image_path"]))
        y_true.append(int(bool(row["is_anomaly"])))
        image_scores.append(score_image(score_map, args))
        score_maps.append(score_map)
        masks.append(mask)

    return {
        "image_path": np.array(image_paths, dtype=object),
        "y_true": np.array(y_true, dtype=np.int64),
        "image_score": np.array(image_scores, dtype=np.float32),
        "score_maps": score_maps,
        "masks": masks,
        "layer_score_maps": layer_score_maps,
    }


def calibration_stats(predictions: dict, stat: str = "mean_std") -> tuple[np.ndarray, np.ndarray]:
    maps = np.asarray(predictions["score_maps"], dtype=np.float32)
    if stat == "mean_std":
        center_map = maps.mean(axis=0).astype(np.float32, copy=False)
        spread_map = maps.std(axis=0).astype(np.float32, copy=False)
        return center_map, spread_map
    if stat == "median_mad":
        center_map = np.median(maps, axis=0).astype(np.float32, copy=False)
        spread_map = (1.4826 * np.median(np.abs(maps - center_map[None, ...]), axis=0)).astype(np.float32, copy=False)
        return center_map, spread_map
    raise ValueError(f"Unsupported calibration stat: {stat}")


def layer_calibration_stats(predictions: dict, layers: list[str], stat: str = "mean_std") -> dict[str, tuple[np.ndarray, np.ndarray]]:
    stats = {}
    layer_maps = predictions["layer_score_maps"]
    for layer_name in layers:
        maps = np.asarray(layer_maps[layer_name], dtype=np.float32)
        if stat == "mean_std":
            center_map = maps.mean(axis=0).astype(np.float32, copy=False)
            spread_map = maps.std(axis=0).astype(np.float32, copy=False)
        elif stat == "median_mad":
            center_map = np.median(maps, axis=0).astype(np.float32, copy=False)
            spread_map = (1.4826 * np.median(np.abs(maps - center_map[None, ...]), axis=0)).astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unsupported calibration stat: {stat}")
        stats[layer_name] = (center_map, spread_map)
    return stats


def fuse_layer_maps(
    layer_maps: dict[str, np.ndarray],
    layers: list[str],
    layer_weights: dict[str, float],
) -> np.ndarray:
    fused = None
    for layer_name in layers:
        maps = np.asarray(layer_maps[layer_name], dtype=np.float32)
        current = maps * float(layer_weights[layer_name])
        fused = current if fused is None else fused + current
    if fused is None:
        raise ValueError("No layer maps to fuse.")
    return fused.astype(np.float32, copy=False)


def apply_calibration(predictions: dict, mean_map: np.ndarray, std_map: np.ndarray, epsilon: float, args: argparse.Namespace) -> dict:
    calibrated = dict(predictions)
    raw_maps = np.asarray(predictions["score_maps"], dtype=np.float32)
    maps = np.maximum((raw_maps - mean_map[None, ...]) / (std_map[None, ...] + float(epsilon)), 0.0)
    calibrated["raw_score_maps"] = raw_maps
    calibrated["score_maps"] = maps.astype(np.float32, copy=False)
    roi_masks = calibrated.get("roi_masks")
    if roi_masks is None or args.score_region == "full":
        calibrated["image_score"] = np.array([score_image(item, args) for item in maps], dtype=np.float32)
    else:
        calibrated["image_score"] = np.array(
            [
                score_image(item, args, roi)
                for item, roi in zip(maps, roi_masks, strict=True)
            ],
            dtype=np.float32,
        )
    calibrated["calibration_mean_map"] = mean_map
    calibrated["calibration_std_map"] = std_map
    if args.calibration_stat == "median_mad":
        calibrated["calibration_median_map"] = mean_map
        calibrated["calibration_mad_std_map"] = std_map
    return calibrated


def apply_per_layer_calibration(
    predictions: dict,
    calibration_predictions: dict,
    *,
    layers: list[str],
    layer_weights: dict[str, float],
    epsilon: float,
    args: argparse.Namespace,
) -> dict:
    calibrated = dict(predictions)
    raw_layer_maps = predictions["layer_score_maps"]
    calibrated_layer_maps: dict[str, np.ndarray] = {}
    stats = layer_calibration_stats(calibration_predictions, layers, args.calibration_stat)
    for layer_name in layers:
        raw_maps = np.asarray(raw_layer_maps[layer_name], dtype=np.float32)
        mean_map, std_map = stats[layer_name]
        calibrated_maps = np.maximum(
            (raw_maps - mean_map[None, ...]) / (std_map[None, ...] + float(epsilon)),
            0.0,
        )
        calibrated_layer_maps[layer_name] = calibrated_maps.astype(np.float32, copy=False)
        calibrated[f"raw_layer_score_maps_{layer_name}"] = raw_maps
        if args.calibration_stat == "median_mad":
            calibrated[f"calibration_median_map_{layer_name}"] = mean_map
            calibrated[f"calibration_mad_std_map_{layer_name}"] = std_map
        else:
            calibrated[f"calibration_mean_map_{layer_name}"] = mean_map
            calibrated[f"calibration_std_map_{layer_name}"] = std_map
    fused = fuse_layer_maps(calibrated_layer_maps, layers, layer_weights)
    calibrated["raw_score_maps"] = predictions["score_maps"]
    calibrated["layer_score_maps"] = calibrated_layer_maps
    calibrated["score_maps"] = fused
    roi_masks = calibrated.get("roi_masks")
    if roi_masks is None or args.score_region == "full":
        calibrated["image_score"] = np.array([score_image(item, args) for item in fused], dtype=np.float32)
    else:
        calibrated["image_score"] = np.array(
            [
                score_image(item, args, roi)
                for item, roi in zip(fused, roi_masks, strict=True)
            ],
            dtype=np.float32,
        )
    return calibrated


def save_outputs(run_dir: Path, predictions: dict, metrics: dict, args: argparse.Namespace, checkpoint: dict, layers: list[str], cosine_weight: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_path": predictions["image_path"],
        "y_true": predictions["y_true"],
        "image_score": predictions["image_score"],
        "metrics": np.array([metrics], dtype=object),
    }
    if args.save_score_maps:
        payload["score_maps"] = predictions["score_maps"]
        payload["masks"] = predictions["masks"]
        if predictions.get("roi_masks") is not None:
            payload["roi_masks"] = predictions["roi_masks"]
        payload["layer_score_maps"] = np.array([predictions.get("layer_score_maps", {})], dtype=object)
        for layer_name, layer_maps in predictions.get("layer_score_maps", {}).items():
            payload[f"layer_score_maps_{layer_name}"] = layer_maps
        for key, value in predictions.items():
            if (
                key.startswith("raw_layer_score_maps_")
                or key.startswith("calibration_mean_map_")
                or key.startswith("calibration_std_map_")
                or key.startswith("calibration_median_map_")
                or key.startswith("calibration_mad_std_map_")
            ):
                payload[key] = value
        if "raw_score_maps" in predictions:
            payload["raw_score_maps"] = predictions["raw_score_maps"]
        if "calibration_mean_map" in predictions:
            payload["calibration_mean_map"] = predictions["calibration_mean_map"]
        if "calibration_std_map" in predictions:
            payload["calibration_std_map"] = predictions["calibration_std_map"]
        if "calibration_median_map" in predictions:
            payload["calibration_median_map"] = predictions["calibration_median_map"]
        if "calibration_mad_std_map" in predictions:
            payload["calibration_mad_std_map"] = predictions["calibration_mad_std_map"]
    np.savez_compressed(run_dir / "predictions.npz", **payload)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    params = {
        "category": args.category,
        "model_type": checkpoint.get("model_type", "feature_ae_resnet18"),
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_run_id": args.checkpoint_path.parent.name,
        "teacher_backbone": checkpoint_teacher(checkpoint, args),
        "teacher_weights": checkpoint.get("teacher_weights", "IMAGENET1K_V1"),
        "layers": layers,
        "loss": checkpoint.get("loss", "l2_cosine"),
        "cosine_weight": cosine_weight,
        "layer_weights": args.layer_weights_resolved,
        "normalization": "imagenet",
        "input_size": args.input_size,
        "preprocessing_mode": args.preprocessing_mode,
        "tile_size": args.tile_size if args.preprocessing_mode == "tile_256_overlap" else None,
        "context_tile_size": args.context_tile_size if args.preprocessing_mode == "tile_256_overlap" else None,
        "tile_stride": args.tile_stride if args.preprocessing_mode == "tile_256_overlap" else None,
        "tile_aggregation": args.tile_aggregation if args.preprocessing_mode == "tile_256_overlap" else None,
        "calibrate_normal": args.calibrate_normal,
        "calibration_mode": args.calibration_mode,
        "calibration_stat": args.calibration_stat if args.calibrate_normal else None,
        "calibration_max_images": args.calibration_max_images if args.calibrate_normal else None,
        "calibration_epsilon": args.calibration_epsilon if args.calibrate_normal else None,
        "score_region": args.score_region,
        "roi_margin": args.roi_margin,
        "roi_predictions_dir": [str(path) for path in (args.roi_predictions_dir or [])],
        "roi_threshold": args.roi_threshold,
        "roi_dilate_radius": args.roi_dilate_radius,
        "roi_applied_to": "score_map_and_image_score" if args.apply_score_region_to_map else "image_score_only",
        "apply_score_region_to_map": args.apply_score_region_to_map,
        "score_smoothing": args.score_smoothing,
        "object_threshold": args.object_threshold,
        "head_x_min": args.head_x_min,
        "head_x_max": args.head_x_max,
        "head_y_min": args.head_y_min,
        "head_y_max": args.head_y_max,
        "score_image": args.score_image,
        "topk_fraction": args.topk_fraction,
        "batch_size": args.batch_size,
        "limit_test": args.limit_test,
        "save_score_maps": args.save_score_maps,
        "save_previews": args.save_previews,
        "preview_score_min_percentile": args.preview_score_min_percentile,
        "preview_score_max_percentile": args.preview_score_max_percentile,
        "preview_score_gamma": args.preview_score_gamma,
        "source_training_scope": checkpoint.get("training_scope"),
    }
    (run_dir / "params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")


def load_letterbox_mask(row: pd.Series, input_size: int) -> np.ndarray:
    if bool(row["has_mask"]) and pd.notna(row["mask_path"]):
        mask = Image.open(project_path(row["mask_path"])).convert("L")
    else:
        mask = Image.new("L", (input_size, input_size), 0)
    from src.models.baselines.patchcore import ResizeMaskLetterbox

    return ResizeMaskLetterbox(input_size)(mask)


def tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor((0.485, 0.456, 0.406), device=image.device)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=image.device)[:, None, None]
    return (image * std + mean).clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()


def normalized_preview_score_map(score_map: np.ndarray, roi: np.ndarray | None, args: argparse.Namespace) -> np.ndarray:
    arr = np.asarray(score_map, dtype=np.float32)
    if roi is not None and np.asarray(roi).shape == arr.shape and np.any(np.asarray(roi) > 0):
        values = arr[np.asarray(roi) > 0]
    else:
        values = arr.reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    low_p = float(args.preview_score_min_percentile)
    high_p = float(args.preview_score_max_percentile)
    low = float(np.percentile(values, low_p)) if low_p > 0 else 0.0
    high = float(np.percentile(values, high_p))
    if high <= low:
        high = float(values.max(initial=low))
    if high <= low:
        return np.zeros_like(arr, dtype=np.float32)
    norm = np.clip((arr - low) / (high - low), 0.0, 1.0)
    gamma = float(args.preview_score_gamma)
    if gamma > 0 and gamma != 1.0:
        norm = np.power(norm, gamma)
    return norm.astype(np.float32, copy=False)


def preview_heatmap(score_map: np.ndarray, roi: np.ndarray | None, args: argparse.Namespace) -> Image.Image:
    norm = normalized_preview_score_map(score_map, roi, args)
    heatmap = np.zeros((*norm.shape, 3), dtype=np.float32)
    heatmap[..., 0] = norm
    heatmap[..., 1] = np.sqrt(norm) * 0.35
    heatmap[..., 2] = 1.0 - norm
    if roi is not None and np.asarray(roi).shape == norm.shape:
        heatmap[np.asarray(roi) <= 0] = 0.0
    return rgb_array_to_image(heatmap)


def overlay_heatmap(rgb: np.ndarray, score_map: np.ndarray, roi: np.ndarray | None, args: argparse.Namespace) -> Image.Image:
    base = np.clip(rgb, 0.0, 1.0)
    heat = np.asarray(preview_heatmap(score_map, roi, args)).astype(np.float32) / 255.0
    overlay = (0.55 * base + 0.45 * heat).clip(0.0, 1.0)
    return rgb_array_to_image(overlay)


def make_preview_panel(rgb: np.ndarray, score_map: np.ndarray, mask: np.ndarray, title: str, args: argparse.Namespace) -> Image.Image:
    original = rgb_array_to_image(rgb)
    heat = preview_heatmap(score_map, None, args)
    overlay = overlay_heatmap(rgb, score_map, None, args)
    mask_img = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L").convert("RGB")
    panels = [original, heat, overlay, mask_img]
    labels = ["input", "score_map", "overlay", "mask"]
    width, height = original.size
    label_h = 42
    gap = 8
    canvas = Image.new("RGB", (width * len(panels) + gap * (len(panels) - 1), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, panel) in enumerate(zip(labels, panels, strict=True)):
        x = idx * (width + gap)
        canvas.paste(panel, (x, label_h))
        draw.text((x + 4, 8), label, fill=(0, 0, 0))
    draw.text((4, 24), title[:140], fill=(80, 80, 80))
    return canvas


def make_roi_preview_panel(
    rgb: np.ndarray,
    score_map: np.ndarray,
    mask: np.ndarray,
    roi: np.ndarray | None,
    title: str,
    args: argparse.Namespace,
) -> Image.Image:
    if roi is None:
        return make_preview_panel(rgb, score_map, mask, title, args)
    original = rgb_array_to_image(rgb)
    heat = preview_heatmap(score_map, roi, args)
    overlay = overlay_heatmap(rgb, score_map, roi, args)
    mask_img = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L").convert("RGB")
    roi_img = Image.fromarray((np.clip(roi, 0, 1) * 255).astype(np.uint8), mode="L").convert("RGB")
    panels = [original, heat, overlay, mask_img, roi_img]
    labels = ["input", "score_map", "overlay", "mask", "score roi"]
    width, height = original.size
    label_h = 42
    gap = 8
    canvas = Image.new("RGB", (width * len(panels) + gap * (len(panels) - 1), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, panel) in enumerate(zip(labels, panels, strict=True)):
        x = idx * (width + gap)
        canvas.paste(panel, (x, label_h))
        draw.text((x + 4, 8), label, fill=(0, 0, 0))
    draw.text((4, 24), title[:140], fill=(80, 80, 80))
    return canvas


@torch.inference_mode()
def save_previews(
    run_dir: Path,
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    layers: list[str],
    cosine_weight: float,
    predictions: dict,
    args: argparse.Namespace,
) -> None:
    preview_dir = run_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    if args.preprocessing_mode == "tile_256_overlap":
        score_maps = predictions["score_maps"]
        masks = predictions["masks"]
        roi_maps = predictions.get("roi_masks")
        for idx, row in df.head(args.max_previews).reset_index(drop=True).iterrows():
            image = Image.open(project_path(row["image_path"])).convert("RGB")
            rgb = np.asarray(image, dtype=np.float32) / 255.0
            roi = None if roi_maps is None or args.score_region == "full" else roi_maps[idx]
            panel = make_roi_preview_panel(
                rgb,
                score_maps[idx],
                masks[idx],
                roi,
                f"{row['category']} | label={row['label']} | anomaly={bool(row['is_anomaly'])}",
                args,
            )
            panel.save(preview_dir / f"{idx:03d}_{safe_image_id(str(row['image_path']))}.png")
        return

    transform = build_pixel_ae_transform(
        args.input_size,
        phase="eval",
        normalization="imagenet",
        augmentation_policy="none",
    )
    score_maps = predictions["score_maps"]
    roi_maps = predictions.get("roi_masks")
    for idx, row in df.head(args.max_previews).reset_index(drop=True).iterrows():
        image = Image.open(project_path(row["image_path"])).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        rgb = tensor_to_rgb(tensor[0])
        score_map = score_maps[idx]
        mask = load_letterbox_mask(row, args.input_size)
        roi = None if roi_maps is None or args.score_region == "full" else roi_maps[idx]
        panel = make_roi_preview_panel(
            rgb,
            score_map,
            mask,
            roi,
            f"{row['category']} | label={row['label']} | anomaly={bool(row['is_anomaly'])}",
            args,
        )
        panel.save(preview_dir / f"{idx:03d}_{safe_image_id(str(row['image_path']))}.png")


def main() -> None:
    args = parse_args()
    if args.preprocessing_mode == "tile_256_overlap" and args.score_region not in {
        "full",
        "casting_surface",
        "casting_surface_margin",
        "functional_surface_prediction",
        "functional_surface_prediction_margin",
    }:
        raise ValueError(
            "Tiled Feature-AE evaluation supports --score-region full, casting_surface, "
            "casting_surface_margin, functional_surface_prediction or functional_surface_prediction_margin."
        )
    if args.score_region in {"functional_surface_prediction", "functional_surface_prediction_margin"}:
        if not args.roi_predictions_dir:
            raise ValueError("--score-region functional_surface_prediction requires --roi-predictions-dir.")
        args.functional_roi_lookup = build_functional_roi_lookup(args)
    else:
        args.functional_roi_lookup = {}
    if args.calibration_max_images < 1:
        raise ValueError("--calibration-max-images must be >= 1.")
    if args.calibration_epsilon <= 0:
        raise ValueError("--calibration-epsilon must be > 0.")
    if args.topk_fraction <= 0 or args.topk_fraction > 1:
        raise ValueError("--topk-fraction must be in (0, 1].")

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    layers = checkpoint_layers(checkpoint, args)
    teacher_backbone = checkpoint_teacher(checkpoint, args)
    cosine_weight = checkpoint_cosine_weight(checkpoint, args)
    layer_weights = parse_layer_weights(args.layer_weights, layers)
    args.layer_weights_resolved = layer_weights
    device = resolve_device(args.device)

    teacher = ResNetTeacherFeatures(teacher_backbone, layers).to(device).eval()
    student = build_feature_autoencoder(checkpoint.get("model_type", "feature_ae_resnet18"), layers)
    student.load_state_dict(checkpoint["model_state_dict"])
    student.to(device).eval()

    train_df, test_df = split_category_data(args.category)
    test_df = maybe_limit(test_df, args.limit_test)
    run_name = build_run_name(args, layers)
    run_dir = args.output_dir / args.category / run_name

    if args.preprocessing_mode == "letterbox":
        transform = build_pixel_ae_transform(
            args.input_size,
            phase="eval",
            normalization="imagenet",
            augmentation_policy="none",
        )
        test_loader = make_dataloader(
            UnifiedAnomalyDataset(test_df, input_size=args.input_size, include_masks=True, image_transform=transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        predictions = predict_letterbox(
            student=student,
            teacher=teacher,
            loader=test_loader,
            device=device,
            layers=layers,
            input_size=args.input_size,
            cosine_weight=cosine_weight,
            layer_weights=layer_weights,
            args=args,
        )
        if args.calibrate_normal:
            calibration_df = maybe_limit(train_df, args.calibration_max_images)
            calibration_loader = make_dataloader(
                UnifiedAnomalyDataset(calibration_df, input_size=args.input_size, include_masks=False, image_transform=transform),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            calibration_predictions = predict_letterbox(
                student=student,
                teacher=teacher,
                loader=calibration_loader,
                device=device,
                layers=layers,
                input_size=args.input_size,
                cosine_weight=cosine_weight,
                layer_weights=layer_weights,
                args=args,
            )
            if args.calibration_mode == "per_layer":
                predictions = apply_per_layer_calibration(
                    predictions,
                    calibration_predictions,
                    layers=layers,
                    layer_weights=layer_weights,
                    epsilon=args.calibration_epsilon,
                    args=args,
                )
            else:
                mean_map, std_map = calibration_stats(calibration_predictions, args.calibration_stat)
                predictions = apply_calibration(predictions, mean_map, std_map, args.calibration_epsilon, args)
        predictions = apply_score_postprocessing(predictions, args)
        metrics = evaluate_predictions(predictions)
    else:
        predictions = predict_tiled(
            student=student,
            teacher=teacher,
            df=test_df,
            device=device,
            layers=layers,
            cosine_weight=cosine_weight,
            layer_weights=layer_weights,
            args=args,
        )
        if args.calibrate_normal:
            calibration_df = maybe_limit(train_df, args.calibration_max_images)
            calibration_predictions = predict_tiled(
                student=student,
                teacher=teacher,
                df=calibration_df,
                device=device,
                layers=layers,
                cosine_weight=cosine_weight,
                layer_weights=layer_weights,
                args=args,
            )
            if args.calibration_mode == "per_layer":
                predictions = apply_per_layer_calibration(
                    predictions,
                    calibration_predictions,
                    layers=layers,
                    layer_weights=layer_weights,
                    epsilon=args.calibration_epsilon,
                    args=args,
                )
            else:
                mean_map, std_map = calibration_stats(calibration_predictions, args.calibration_stat)
                predictions = apply_calibration(predictions, mean_map, std_map, args.calibration_epsilon, args)
        predictions = apply_score_postprocessing(predictions, args)
        metrics = evaluate_variable_predictions(predictions)

    save_outputs(run_dir, predictions, metrics, args, checkpoint, layers, cosine_weight)
    if args.save_previews:
        save_previews(
            run_dir,
            student=student,
            teacher=teacher,
            df=test_df,
            device=device,
            layers=layers,
            cosine_weight=cosine_weight,
            predictions=predictions,
            args=args,
        )
    print(f"Saved Feature-AE evaluation to: {run_dir}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def evaluate_from_args(args: argparse.Namespace) -> None:
    """Run evaluation from an already parsed argparse namespace."""
    global parse_args
    original_parse_args = parse_args
    parse_args = lambda: args  # type: ignore[assignment]
    try:
        main()
    finally:
        parse_args = original_parse_args


if __name__ == "__main__":
    main()


