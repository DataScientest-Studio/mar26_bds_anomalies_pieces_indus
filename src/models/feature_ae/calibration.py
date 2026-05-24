"""Post-hoc calibration helpers for Feature AE / RD AE heatmaps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import PATHS
from src.features.functional_surface import load_functional_predictions
from src.models.baselines.patchcore import _normalized_low_fpr_aupimo

__all__ = [
    "calibration_stats",
    "apply_score_postprocessing",
    "build_functional_roi_lookup",
    "image_metrics",
    "load_roi_prob_maps",
    "normalized_path",
    "parse_layer_weights",
    "parse_weights",
    "pixel_metrics",
    "roi_mask_from_image_path",
    "score_image",
    "smooth_maps",
    "topk_score",
    "topk_scores",
]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PATHS.root / path


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


def parse_weights(raw: str, layers: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in raw.split(","):
        if "=" not in item:
            raise ValueError(f"Invalid layer weight {item!r}; expected layer=value.")
        key, value = item.split("=", 1)
        values[key.strip()] = float(value)
    missing = sorted(set(layers) - set(values))
    if missing:
        raise ValueError(f"Missing weights for layers: {missing}")
    total = sum(values[layer] for layer in layers)
    if total <= 0:
        raise ValueError(f"Weights must sum to > 0: {raw}")
    return {layer: float(values[layer]) / total for layer in layers}


def calibration_stats(predictions: dict, stat: str = "mean_std") -> tuple[np.ndarray, np.ndarray]:
    maps = np.asarray(predictions["score_maps"], dtype=np.float32)
    if stat == "mean_std":
        center_map = maps.mean(axis=0).astype(np.float32, copy=False)
        scale_map = maps.std(axis=0).astype(np.float32, copy=False)
    elif stat == "median_mad":
        center_map = np.median(maps, axis=0).astype(np.float32, copy=False)
        scale_map = (1.4826 * np.median(np.abs(maps - center_map[None]), axis=0)).astype(np.float32, copy=False)
    else:
        raise ValueError(stat)
    return center_map, scale_map


def normalized_path(path: object) -> str:
    return str(path).replace("\\", "/")


def load_roi_prob_maps(prediction_paths: np.ndarray, roi_dir: Path | None, fallback_roi: np.ndarray) -> np.ndarray:
    if roi_dir is None:
        return np.asarray(fallback_roi, dtype=np.float32)
    predictions = load_functional_predictions(resolve(roi_dir))
    lookup = {
        normalized_path(path): np.asarray(prob, dtype=np.float32)
        for path, prob in zip(predictions["image_path"], predictions["prob_maps"], strict=True)
    }
    maps = []
    for raw_path, fallback in zip(prediction_paths, fallback_roi, strict=True):
        prob = lookup.get(normalized_path(raw_path))
        if prob is None:
            prob = np.asarray(fallback, dtype=np.float32)
        maps.append(np.asarray(prob, dtype=np.float32))
    return np.stack(maps, axis=0).astype(np.float32, copy=False)


def smooth_maps(maps: np.ndarray, mode: str, batch_size: int) -> np.ndarray:
    if mode == "none":
        return np.asarray(maps, dtype=np.float32)
    outputs: list[np.ndarray] = []
    if mode == "median3":
        for start in range(0, len(maps), batch_size):
            batch = torch.from_numpy(np.asarray(maps[start : start + batch_size], dtype=np.float32))[:, None]
            padded = F.pad(batch, (1, 1, 1, 1), mode="reflect")
            unfolded = F.unfold(padded, kernel_size=3).view(batch.shape[0], 9, *batch.shape[-2:])
            outputs.append(unfolded.median(dim=1).values.numpy().astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0)
    if mode == "gaussian":
        kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        kernel = (kernel / kernel.sum()).view(1, 1, 3, 3)
        for start in range(0, len(maps), batch_size):
            batch = torch.from_numpy(np.asarray(maps[start : start + batch_size], dtype=np.float32))[:, None]
            padded = F.pad(batch, (1, 1, 1, 1), mode="reflect")
            outputs.append(F.conv2d(padded, kernel).squeeze(1).numpy().astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0)
    raise ValueError(f"Unknown smoothing mode: {mode}")


def topk_scores(score_maps: np.ndarray, valid_masks: np.ndarray | None, fraction: float) -> np.ndarray:
    scores: list[float] = []
    for index, score_map in enumerate(score_maps):
        if valid_masks is not None:
            valid = valid_masks[index] > 0
            values = score_map[valid] if valid.any() else score_map.reshape(-1)
        else:
            values = score_map.reshape(-1)
        k = max(1, int(round(len(values) * float(fraction))))
        scores.append(float(np.partition(values, len(values) - k)[-k:].mean()))
    return np.asarray(scores, dtype=np.float32)


def topk_score(score_map: np.ndarray, fraction: float, valid_mask: np.ndarray | None = None) -> float:
    maps = np.asarray(score_map, dtype=np.float32)[None, ...]
    masks = None if valid_mask is None else np.asarray(valid_mask, dtype=np.uint8)[None, ...]
    return float(topk_scores(maps, masks, fraction=fraction)[0])


def score_image(score_map: np.ndarray, args, roi_mask: np.ndarray | None = None) -> float:
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


def dilate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if int(radius) <= 0:
        return (np.asarray(mask) > 0).astype(np.float32)
    image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L")
    image = image.filter(ImageFilter.MaxFilter(int(radius) * 2 + 1))
    return (np.asarray(image) > 0).astype(np.float32)


def build_functional_roi_lookup(args) -> dict[str, np.ndarray]:
    dirs = args.roi_predictions_dir or []
    lookup: dict[str, np.ndarray] = {}
    for raw_dir in dirs:
        predictions = load_functional_predictions(resolve(raw_dir))
        for image_path, prob_map in zip(predictions["image_path"], predictions["prob_maps"], strict=True):
            binary = np.asarray(prob_map, dtype=np.float32) >= float(args.roi_threshold)
            lookup[str(image_path)] = dilate_binary_mask(binary, int(args.roi_dilate_radius))
    return lookup


def casting_surface_mask_from_rgb(rgb: np.ndarray, args) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32)
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0
    gray = image.mean(axis=2)
    exterior_candidate = gray <= float(args.object_threshold)
    height, width = exterior_candidate.shape
    exterior = np.zeros((height, width), dtype=bool)
    stack: list[tuple[int, int]] = []
    for x in range(width):
        stack.append((0, x))
        stack.append((height - 1, x))
    for y in range(height):
        stack.append((y, 0))
        stack.append((y, width - 1))
    while stack:
        y, x = stack.pop()
        if exterior[y, x] or not exterior_candidate[y, x]:
            continue
        exterior[y, x] = True
        if y > 0:
            stack.append((y - 1, x))
        if y + 1 < height:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x + 1 < width:
            stack.append((y, x + 1))
    mask = (~exterior).astype(np.float32)
    if args.score_region == "casting_surface_margin":
        kernel = max(3, int(round(min(height, width) * float(args.roi_margin))))
        if kernel % 2 == 0:
            kernel += 1
        tensor = torch.from_numpy(mask)[None, None]
        padded = F.pad(tensor, (kernel // 2,) * 4, mode="replicate")
        mask = F.max_pool2d(padded, kernel_size=kernel, stride=1)[0, 0].numpy().astype(np.float32)
    return mask


def roi_mask_from_image_path(image_path: str, target_shape: tuple[int, int], args) -> np.ndarray:
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
        image = Image.open(resolve(Path(str(image_path)))).convert("RGB")
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        return _resize_mask(casting_surface_mask_from_rgb(rgb, args), target_shape)
    raise ValueError(f"ROI from image path is not implemented for score-region={args.score_region!r}.")


def build_roi_masks_from_paths(predictions: dict, args) -> np.ndarray | None:
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


def apply_score_postprocessing(predictions: dict, args) -> dict:
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


def pixel_metrics(y_true: np.ndarray, masks: np.ndarray, score_maps: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    pixel_true = masks.reshape(-1).astype(np.uint8)
    pixel_score = score_maps.reshape(-1).astype(np.float32)
    if len(np.unique(pixel_true)) == 2:
        metrics["pixel_auroc"] = float(roc_auc_score(pixel_true, pixel_score))
        metrics["pixel_ap"] = float(average_precision_score(pixel_true, pixel_score))
        aupimo = _normalized_low_fpr_aupimo(
            y_true.astype(np.int64),
            masks,
            score_maps.astype(np.float32),
            fpr_low=1e-5,
            fpr_high=1e-3,
        )
        if aupimo is not None:
            metrics["pixel_aupimo_1e-5_1e-3"] = float(aupimo)
    return metrics


def image_metrics(y_true: np.ndarray, image_score: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_true)) != 2:
        return {}
    return {
        "image_auroc": float(roc_auc_score(y_true, image_score)),
        "image_ap": float(average_precision_score(y_true, image_score)),
    }






