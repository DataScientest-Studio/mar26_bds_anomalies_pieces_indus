"""Train a Feature-AE student to reconstruct frozen teacher features."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

try:
    import cv2
except ImportError:  # pragma: no cover - optional acceleration for structure priors.
    cv2 = None

from src.config import DATA, PATHS
from src.features.augmentation_profiles import resolve_augmentation_profile
from src.features.tiling import TileSpec, tile_image
from src.features.functional_surface import load_functional_predictions
from src.models.feature_ae.models import (
    ResNetTeacherFeatures,
    build_feature_autoencoder,
)
from src.models.feature_ae.training_loop import run_epoch
from src.models.baselines.patchcore import UnifiedAnomalyDataset, make_dataloader, project_path, resolve_device
from src.models.pixel_ae.runtime import (
    build_pixel_ae_transform,
    load_training_data,
    maybe_limit,
    repeat_training_rows,
    resolve_repeat_factor,
    split_train_val,
)
from src.models.baselines.patchcore import load_unified_dataset
from src.models.feature_ae.config import parse_args




def resolve_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else PATHS.root / path


def parse_layer_loss_weights(raw: list[str] | None, layers: list[str]) -> dict[str, float]:
    if not raw:
        return {layer: 1.0 for layer in layers}
    parsed: dict[str, float] = {}
    if all("=" not in item for item in raw):
        if len(raw) != len(layers):
            raise ValueError("--layer-loss-weights as bare values must match the number of --layers.")
        parsed = {layer: float(value) for layer, value in zip(layers, raw, strict=True)}
    else:
        for item in raw:
            if "=" not in item:
                raise ValueError("--layer-loss-weights must use either all bare values or all layer=value entries.")
            layer, value = item.split("=", 1)
            layer = layer.strip()
            if layer not in layers:
                raise ValueError(f"--layer-loss-weights contains unknown layer {layer!r}; expected one of {layers!r}.")
            parsed[layer] = float(value)
        missing = [layer for layer in layers if layer not in parsed]
        if missing:
            raise ValueError(f"--layer-loss-weights missing values for layers: {missing!r}.")
    if any(value < 0 for value in parsed.values()):
        raise ValueError("--layer-loss-weights values must be >= 0.")
    if sum(parsed.values()) <= 0:
        raise ValueError("At least one --layer-loss-weights value must be > 0.")
    return parsed


def dilate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if int(radius) <= 0:
        return (np.asarray(mask) > 0).astype(np.float32)
    image = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L")
    size = int(radius) * 2 + 1
    image = image.filter(ImageFilter.MaxFilter(size))
    return (np.asarray(image) > 0).astype(np.float32)


def load_roi_lookup(predictions_dir: Path | None, *, threshold: float, dilate_radius: int) -> dict[str, np.ndarray]:
    if predictions_dir is None:
        return {}
    predictions = load_functional_predictions(predictions_dir)
    lookup = {}
    for image_path, prob_map in zip(predictions["image_path"], predictions["prob_maps"], strict=True):
        binary = np.asarray(prob_map, dtype=np.float32) >= float(threshold)
        lookup[str(image_path)] = dilate_binary_mask(binary, int(dilate_radius))
    return lookup


def crop_centered_context_tile(image: Image.Image, spec: TileSpec, context_tile_size: int) -> Image.Image:
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


class TiledFeatureDataset(Dataset):
    """Virtual native-ratio tile dataset for Feature-AE training/validation."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        tile_size: int,
        stride: int,
        sampling: str,
        max_tiles_per_image: int | None,
        image_transform,
        model_input_size: int,
        seed: int,
        context_tile_size: int | None = None,
        context_transform=None,
        roi_lookup: dict[str, np.ndarray] | None = None,
        min_roi_ratio: float = 0.0,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tile_size = int(tile_size)
        self.model_input_size = int(model_input_size)
        self.context_tile_size = None if context_tile_size is None else int(context_tile_size)
        self.stride = int(stride)
        self.image_transform = image_transform
        self.context_transform = context_transform or image_transform
        self.roi_lookup = roi_lookup or {}
        self.roi_loss_weight = 1.0
        self.background_loss_weight = 0.02
        self.normal_structure_loss_weight = 0.0
        self.normal_structure_dark_percentile = 8.0
        self.normal_structure_min_area = 50
        self.normal_structure_max_area_ratio = 0.10
        self.normal_structure_dilate_radius = 8
        self.roi_border_loss_weight = 0.0
        self.roi_border_radius = 12
        self._normal_structure_prior_cache: dict[str, np.ndarray] = {}
        self.entries: list[tuple[str, TileSpec, str, bool, float]] = []
        rng = torch.Generator().manual_seed(int(seed))

        for _idx, row in self.df.iterrows():
            image_path = str(row["image_path"])
            category = str(row["category"])
            is_anomaly = bool(row.get("is_anomaly", False))
            image = Image.open(project_path(image_path)).convert("RGB")
            tiled = tile_image(image, tile_size=self.tile_size, stride=self.stride)
            specs = list(tiled.specs)
            if sampling == "random" or max_tiles_per_image is not None:
                sample_count = max_tiles_per_image if max_tiles_per_image is not None else 1
                sample_count = max(1, min(int(sample_count), len(specs)))
                indices = torch.randperm(len(specs), generator=rng)[:sample_count].tolist()
                specs = [specs[index] for index in sorted(indices)]
            roi = self.roi_lookup.get(image_path)
            for spec in specs:
                roi_ratio = 1.0
                if roi is not None:
                    roi_ratio = self._roi_ratio_for_spec(roi, spec, image.size)
                    if roi_ratio < float(min_roi_ratio):
                        continue
                self.entries.append((image_path, spec, category, is_anomaly, float(roi_ratio)))

    def _roi_ratio_for_spec(self, roi: np.ndarray, spec: TileSpec, image_size: tuple[int, int]) -> float:
        width, height = image_size
        padded_width = max(width, self.tile_size, self.context_tile_size or self.tile_size)
        padded_height = max(height, self.tile_size, self.context_tile_size or self.tile_size)
        if roi.shape != (padded_height, padded_width):
            roi_image = Image.fromarray((np.asarray(roi) > 0).astype(np.uint8) * 255, mode="L")
            resized = roi_image.resize((width, height), resample=Image.Resampling.NEAREST)
            canvas = Image.new("L", (padded_width, padded_height), 0)
            canvas.paste(resized, (0, 0))
            roi = (np.asarray(canvas) > 0).astype(np.float32)
        tile = roi[spec.y0 : spec.y1, spec.x0 : spec.x1]
        return float(np.asarray(tile, dtype=np.float32).mean())

    def _normal_structure_prior(self, image_path: str, image: Image.Image, roi_image: Image.Image) -> np.ndarray:
        if self.normal_structure_loss_weight <= 0:
            return np.zeros((image.height, image.width), dtype=np.uint8)
        cached = self._normal_structure_prior_cache.get(image_path)
        if cached is not None and cached.shape == (image.height, image.width):
            return cached
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        roi = np.asarray(roi_image, dtype=np.uint8) > 127
        values = gray[roi]
        if values.size == 0:
            prior = np.zeros((image.height, image.width), dtype=np.uint8)
            self._normal_structure_prior_cache[image_path] = prior
            return prior
        threshold = float(np.percentile(values, float(self.normal_structure_dark_percentile)))
        dark = (gray <= threshold) & roi
        labels, count = self._connected_components(dark)
        if count <= 1:
            prior = np.zeros((image.height, image.width), dtype=np.uint8)
            self._normal_structure_prior_cache[image_path] = prior
            return prior
        max_area = max(1, int(round(float(self.normal_structure_max_area_ratio) * roi.sum())))
        keep = np.zeros_like(dark, dtype=bool)
        min_area = max(1, int(self.normal_structure_min_area))
        for component_idx in range(1, count):
            area = int((labels == component_idx).sum())
            if min_area <= area <= max_area:
                keep |= labels == component_idx
        prior = (dilate_binary_mask(keep.astype(np.float32), int(self.normal_structure_dilate_radius)) > 0).astype(
            np.uint8
        )
        self._normal_structure_prior_cache[image_path] = prior
        return prior

    @staticmethod
    def _crop_mask(mask: np.ndarray, spec: TileSpec, model_input_size: int, tile_size: int) -> np.ndarray:
        tile = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255, mode="L").crop(
            (spec.x0, spec.y0, spec.x1, spec.y1)
        )
        if model_input_size != tile_size:
            tile = tile.resize((model_input_size, model_input_size), resample=Image.Resampling.NEAREST)
        return (np.asarray(tile, dtype=np.uint8) > 127).astype(np.float32)

    @staticmethod
    def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
        mask = np.asarray(mask, dtype=bool)
        if cv2 is not None:
            count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            return labels.astype(np.int32, copy=False), int(count)
        labels = np.zeros(mask.shape, dtype=np.int32)
        current = 0
        height, width = mask.shape
        for y in range(height):
            for x in range(width):
                if not mask[y, x] or labels[y, x] != 0:
                    continue
                current += 1
                stack = [(y, x)]
                labels[y, x] = current
                while stack:
                    cy, cx = stack.pop()
                    for ny in (cy - 1, cy, cy + 1):
                        for nx in (cx - 1, cx, cx + 1):
                            if (
                                ny < 0
                                or nx < 0
                                or ny >= height
                                or nx >= width
                                or labels[ny, nx] != 0
                                or not mask[ny, nx]
                            ):
                                continue
                            labels[ny, nx] = current
                            stack.append((ny, nx))
        return labels, current + 1

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        image_path, spec, category, is_anomaly, roi_ratio = self.entries[index]
        image = Image.open(project_path(image_path)).convert("RGB")
        width, height = image.size
        padded_width = max(width, self.tile_size, self.context_tile_size or self.tile_size)
        padded_height = max(height, self.tile_size, self.context_tile_size or self.tile_size)
        if (padded_width, padded_height) != (width, height):
            padded = Image.new("RGB", (padded_width, padded_height), 0)
            padded.paste(image, (0, 0))
            image = padded
        tile = image.crop((spec.x0, spec.y0, spec.x1, spec.y1))
        context_tile = None
        if self.context_tile_size is not None:
            context_tile = crop_centered_context_tile(image, spec, self.context_tile_size)
        roi = self.roi_lookup.get(image_path)
        if roi is None:
            roi_tile = Image.new("L", (self.tile_size, self.tile_size), 255)
        else:
            if roi.shape != (padded_height, padded_width):
                roi_image = Image.fromarray((np.asarray(roi) > 0).astype(np.uint8) * 255, mode="L")
                resized = roi_image.resize((width, height), resample=Image.Resampling.NEAREST)
                canvas = Image.new("L", (padded_width, padded_height), 0)
                canvas.paste(resized, (0, 0))
                roi_image = canvas
            else:
                roi_image = Image.fromarray((np.asarray(roi) > 0).astype(np.uint8) * 255, mode="L")
            roi_tile = roi_image.crop((spec.x0, spec.y0, spec.x1, spec.y1))
        if self.model_input_size != self.tile_size:
            roi_tile = roi_tile.resize(
                (self.model_input_size, self.model_input_size),
                resample=Image.Resampling.NEAREST,
            )
        roi_arr = (np.asarray(roi_tile, dtype=np.uint8) > 127).astype(np.float32)
        item = {
            "image": self.image_transform(tile),
            "roi_mask": torch.from_numpy(roi_arr[None, ...]),
            "roi_ratio": torch.tensor(roi_ratio, dtype=torch.float32),
            "is_anomaly": torch.tensor(is_anomaly, dtype=torch.long),
            "image_path": image_path,
            "category": category,
        }
        if self.normal_structure_loss_weight > 0:
            structure_prior = self._normal_structure_prior(image_path, image, roi_image)
            structure_mask = self._crop_mask(structure_prior, spec, self.model_input_size, self.tile_size)
            item["normal_structure_mask"] = torch.from_numpy(structure_mask[None, ...])
        if context_tile is not None:
            item["context_image"] = self.context_transform(context_tile)
        return item


def git_value(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=PATHS.root,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_trace() -> dict:
    status = git_value(["git", "status", "--short"])
    return {
        "git_commit": git_value(["git", "rev-parse", "HEAD"]),
        "git_dirty": None if status is None else bool(status),
    }


def feature_ae_root() -> Path:
    return PATHS.root / "models" / "feature_ae"


def training_scope(args: argparse.Namespace) -> str:
    if args.all_categories:
        return "global"
    if args.categories:
        return "selected"
    return "category"


def load_feature_ae_training_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.categories:
        df = load_unified_dataset()
        selected = set(str(category) for category in args.categories)
        train_df = df[(df["split"] == "train") & (~df["is_anomaly"]) & df["category"].astype(str).isin(selected)].copy()
        if train_df.empty:
            raise ValueError(f"No normal training images found for categories={sorted(selected)!r}.")
        if train_df["is_anomaly"].any():
            raise RuntimeError("Training data contains anomalies; this violates normal-only training.")
        return train_df.reset_index(drop=True)
    return load_training_data(args)


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    scope = training_scope(args)
    layer_tag = "-".join(args.layers)
    return (
        f"feature_ae_{args.teacher_backbone}_scope-{scope}"
        f"_s{int(args.input_size)}"
        f"_layers-{layer_tag}"
        f"_e{int(args.epochs)}"
        f"_loss-{args.loss}"
        f"_lr{args.learning_rate:g}"
    )


def run_dir_has_artifacts(run_dir: Path) -> bool:
    return any(
        (run_dir / name).exists()
        for name in {
            "params.json",
            "checkpoint.pt",
            "checkpoint_best.pt",
            "checkpoint_last.pt",
            "loss_history.csv",
        }
    )


def ensure_run_dir_writable(run_dir: Path, overwrite: bool) -> None:
    if run_dir_has_artifacts(run_dir) and not overwrite:
        raise FileExistsError(
            f"Run directory already contains artifacts: {run_dir}. "
            "Use a new --run-name or pass --overwrite-run explicitly."
        )
    run_dir.mkdir(parents=True, exist_ok=True)


def current_lr(optimizer: AdamW) -> float:
    return float(optimizer.param_groups[0]["lr"])


def make_checkpoint(
    *,
    student: torch.nn.Module,
    optimizer: AdamW,
    args: argparse.Namespace,
    history: list[dict],
    best_epoch: int | None,
    best_val_loss: float | None,
    completed_epochs: int,
    stopped_early: bool,
) -> dict:
    return {
        "model_type": args.model_type,
        "model_state_dict": student.state_dict(),
        "optimizer": "AdamW",
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "category": "global" if args.all_categories else ("selected" if args.categories else args.category),
        "categories": args.categories,
        "training_scope": training_scope(args),
        "teacher_backbone": args.teacher_backbone,
        "teacher_weights": "IMAGENET1K_V1",
        "layers": args.layers,
        "loss": args.loss,
        "cosine_weight": args.cosine_weight,
        "layer_loss_weights": args.layer_loss_weights_resolved,
        "layer_weights": args.layer_loss_weights_resolved,
        "normalization": args.normalization,
        "input_size": args.input_size,
        "preprocessing_mode": args.preprocessing_mode,
        "tile_size": args.tile_size if args.preprocessing_mode == "tile_256_overlap" else None,
        "context_tile_size": args.context_tile_size if args.preprocessing_mode == "tile_256_overlap" else None,
        "tile_train_stride": args.tile_train_stride if args.preprocessing_mode == "tile_256_overlap" else None,
        "tile_train_sampling": args.tile_train_sampling if args.preprocessing_mode == "tile_256_overlap" else None,
        "tile_train_max_tiles_per_image": args.tile_train_max_tiles_per_image
        if args.preprocessing_mode == "tile_256_overlap"
        else None,
        "roi_predictions_dir": str(args.roi_predictions_dir) if args.roi_predictions_dir is not None else None,
        "roi_threshold": args.roi_threshold,
        "roi_loss_weight": args.roi_loss_weight,
        "background_loss_weight": args.background_loss_weight,
        "roi_dilate_radius": args.roi_dilate_radius,
        "min_roi_ratio": args.min_roi_ratio,
        "normal_structure_loss_weight": args.normal_structure_loss_weight,
        "normal_structure_dark_percentile": args.normal_structure_dark_percentile,
        "normal_structure_min_area": args.normal_structure_min_area,
        "normal_structure_max_area_ratio": args.normal_structure_max_area_ratio,
        "normal_structure_dilate_radius": args.normal_structure_dilate_radius,
        "roi_border_loss_weight": args.roi_border_loss_weight,
        "roi_border_radius": args.roi_border_radius,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "checkpoint_every_epochs": args.checkpoint_every_epochs,
        "checkpoint_epochs": args.checkpoint_epochs,
        "learning_rate": current_lr(optimizer),
        "initial_learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "init_checkpoint_path": str(args.init_checkpoint_path) if args.init_checkpoint_path is not None else None,
        "augmentation_profile": args.augmentation_profile_resolved,
        "augmentation_profile_requested": args.augmentation_profile,
        "augmentation_profile_params": args.augmentation_profile_params,
        "repeat_factor": args.repeat_factor_resolved,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "completed_epochs": completed_epochs,
        "stopped_early": stopped_early,
        "run_created_at": args.run_created_at,
        "command_line": args.command_line,
        **args.git_trace,
    }


def save_state(
    *,
    run_dir: Path,
    student: torch.nn.Module,
    optimizer: AdamW,
    args: argparse.Namespace,
    history: list[dict],
    best_epoch: int | None,
    best_val_loss: float | None,
    completed_epochs: int,
    stopped_early: bool,
    save_as_best: bool,
) -> None:
    checkpoint = make_checkpoint(
        student=student,
        optimizer=optimizer,
        args=args,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        completed_epochs=completed_epochs,
        stopped_early=stopped_early,
    )
    torch.save(checkpoint, run_dir / "checkpoint_last.pt")
    if save_as_best:
        torch.save(checkpoint, run_dir / "checkpoint_best.pt")
        shutil.copy2(run_dir / "checkpoint_best.pt", run_dir / "checkpoint.pt")
    elif not args.save_best:
        shutil.copy2(run_dir / "checkpoint_last.pt", run_dir / "checkpoint.pt")
    elif not (run_dir / "checkpoint.pt").exists():
        shutil.copy2(run_dir / "checkpoint_last.pt", run_dir / "checkpoint.pt")

    pd.DataFrame(history).to_csv(run_dir / "loss_history.csv", index=False)
    params = {
        **checkpoint,
        "optimizer_state_dict": None,
        "model_state_dict": None,
        "checkpoint_path": str(run_dir / "checkpoint.pt"),
        "checkpoint_best_path": str(run_dir / "checkpoint_best.pt")
        if (run_dir / "checkpoint_best.pt").exists()
        else None,
        "checkpoint_last_path": str(run_dir / "checkpoint_last.pt"),
        "categories": args.categories,
        "limit_train": args.limit_train,
        "preprocessing_mode": args.preprocessing_mode,
        "tile_size": args.tile_size if args.preprocessing_mode == "tile_256_overlap" else None,
        "context_tile_size": args.context_tile_size if args.preprocessing_mode == "tile_256_overlap" else None,
        "tile_train_stride": args.tile_train_stride if args.preprocessing_mode == "tile_256_overlap" else None,
        "tile_train_sampling": args.tile_train_sampling if args.preprocessing_mode == "tile_256_overlap" else None,
        "tile_train_max_tiles_per_image": args.tile_train_max_tiles_per_image
        if args.preprocessing_mode == "tile_256_overlap"
        else None,
        "val_fraction": args.val_fraction,
        "save_best": args.save_best,
        "checkpoint_every_epochs": args.checkpoint_every_epochs,
        "checkpoint_epochs": args.checkpoint_epochs,
        "num_workers": args.num_workers,
        "lr_scheduler": args.lr_scheduler,
        "lr_patience": args.lr_patience,
        "lr_factor": args.lr_factor,
        "early_stopping_patience": args.early_stopping_patience,
        "min_delta": args.min_delta,
    }
    (run_dir / "params.json").write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_periodic_checkpoint(
    *,
    run_dir: Path,
    student: torch.nn.Module,
    optimizer: AdamW,
    args: argparse.Namespace,
    history: list[dict],
    best_epoch: int | None,
    best_val_loss: float | None,
    completed_epochs: int,
    stopped_early: bool,
) -> Path:
    checkpoint = make_checkpoint(
        student=student,
        optimizer=optimizer,
        args=args,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        completed_epochs=completed_epochs,
        stopped_early=stopped_early,
    )
    checkpoint_path = run_dir / f"checkpoint_epoch_{completed_epochs:03d}.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


METRIC_BEST_FILES = {
    "image_auroc": "checkpoint_best_image_auroc.pt",
    "image_ap": "checkpoint_best_image_ap.pt",
    "pixel_ap": "checkpoint_best_pixel_ap.pt",
    "pixel_aupimo_1e-5_1e-3": "checkpoint_best_pixel_aupimo_1e-5_1e-3.pt",
}


def metric_value(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_metric_evaluation(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    category = args.metric_eval_category or args.category
    if category is None:
        raise ValueError("--metric-eval-category is required when training on multiple categories.")

    output_root = resolve_path(args.metric_eval_output_dir) or (PATHS.root / "models" / "feature_ae_eval_during_training")
    eval_run_name = f"{run_dir.name}_epoch{epoch:03d}"
    metrics_path = output_root / category / eval_run_name / "metrics.json"
    device = args.metric_eval_device or args.device
    batch_size = args.metric_eval_batch_size or args.batch_size
    tile_stride = args.metric_eval_tile_stride or max(1, int(args.tile_size) // 2)

    cmd = [
        sys.executable,
        "-m",
        "src.models.feature_ae.evaluation",
        "--category",
        category,
        "--checkpoint-path",
        str(checkpoint_path),
        "--input-size",
        str(args.input_size),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--output-dir",
        str(output_root),
        "--run-name",
        eval_run_name,
        "--preprocessing-mode",
        args.preprocessing_mode,
        "--layers",
        *args.layers,
        "--teacher-backbone",
        args.teacher_backbone,
        "--layer-weights",
        *args.metric_eval_layer_weights,
        "--score-region",
        args.metric_eval_score_region,
        "--score-image",
        args.metric_eval_score_image,
        "--topk-fraction",
        str(args.metric_eval_topk_fraction),
        "--score-smoothing",
        args.metric_eval_score_smoothing,
        "--calibration-mode",
        args.metric_eval_calibration_mode,
        "--calibration-stat",
        args.metric_eval_calibration_stat,
        "--calibration-max-images",
        str(args.metric_eval_calibration_max_images),
        "--tile-aggregation",
        args.metric_eval_tile_aggregation,
    ]
    if args.preprocessing_mode == "tile_256_overlap":
        cmd.extend(["--tile-size", str(args.tile_size), "--tile-stride", str(tile_stride)])
        if args.context_tile_size is not None:
            cmd.extend(["--context-tile-size", str(args.context_tile_size)])
    if args.metric_eval_calibrate_normal:
        cmd.append("--calibrate-normal")
    if args.metric_eval_roi_predictions_dir:
        cmd.extend(
            [
                "--roi-predictions-dir",
                *[str(resolve_path(path)) for path in args.metric_eval_roi_predictions_dir],
            ]
        )
    cmd.extend(
        [
            "--roi-threshold",
            str(args.metric_eval_roi_threshold),
            "--roi-dilate-radius",
            str(args.metric_eval_roi_dilate_radius),
        ]
    )
    if args.metric_eval_apply_score_region_to_map:
        cmd.append("--apply-score-region-to-map")
    if args.metric_eval_save_score_maps:
        cmd.append("--save-score-maps")
    if args.metric_eval_save_previews:
        cmd.extend(["--save-previews", "--max-previews", str(args.metric_eval_max_previews)])
    if args.metric_eval_no_progress or args.no_progress:
        cmd.append("--no-progress")

    subprocess.run(cmd, cwd=PATHS.root, check=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {key: value for key, value in metrics.items() if metric_value(metrics, key) is not None}


def update_metric_best_checkpoints(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    epoch: int,
    metrics: dict[str, float],
    best_scores: dict[str, dict],
) -> list[str]:
    improved: list[str] = []
    for metric, filename in METRIC_BEST_FILES.items():
        value = metric_value(metrics, metric)
        if value is None:
            continue
        previous = best_scores.get(metric, {}).get("value")
        if previous is None or value > float(previous):
            best_scores[metric] = {
                "epoch": epoch,
                "value": value,
                "checkpoint_path": str(checkpoint_path),
                "copied_to": str(run_dir / filename),
            }
            shutil.copy2(checkpoint_path, run_dir / filename)
            improved.append(metric)

    if "image_ap" in improved:
        shutil.copy2(checkpoint_path, run_dir / "checkpoint_best_image.pt")
    elif "image_auroc" in improved and not (run_dir / "checkpoint_best_image.pt").exists():
        shutil.copy2(checkpoint_path, run_dir / "checkpoint_best_image.pt")

    if "pixel_aupimo_1e-5_1e-3" in improved:
        shutil.copy2(checkpoint_path, run_dir / "checkpoint_best_localization.pt")
    elif "pixel_ap" in improved and not (run_dir / "checkpoint_best_localization.pt").exists():
        shutil.copy2(checkpoint_path, run_dir / "checkpoint_best_localization.pt")

    (run_dir / "metric_eval_best.json").write_text(
        json.dumps(best_scores, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return improved


def main() -> None:
    args = parse_args()
    args.run_created_at = datetime.now(timezone.utc).isoformat()
    args.command_line = " ".join([sys.executable, *sys.argv])
    args.git_trace = git_trace()
    selected_modes = sum(
        [
            args.category is not None,
            bool(args.categories),
            bool(args.all_categories),
        ]
    )
    if selected_modes != 1:
        raise ValueError("Use exactly one of --category, --categories or --all-categories.")
    if args.save_best and args.val_fraction == 0:
        raise ValueError("--save-best requires --val-fraction > 0.")
    if args.cosine_weight < 0:
        raise ValueError("--cosine-weight must be >= 0.")
    if args.early_stopping_patience is not None and args.early_stopping_patience < 1:
        raise ValueError("--early-stopping-patience must be >= 1.")
    if args.checkpoint_every_epochs < 0:
        raise ValueError("--checkpoint-every-epochs must be >= 0.")
    if args.metric_eval_every_epochs < 0:
        raise ValueError("--metric-eval-every-epochs must be >= 0.")
    if args.metric_eval_start_epoch < 1:
        raise ValueError("--metric-eval-start-epoch must be >= 1.")
    if args.metric_eval_every_epochs > 0 and args.category is None and args.metric_eval_category is None:
        raise ValueError("--metric-eval-category is required for metric eval when training on multiple categories.")
    if args.checkpoint_epochs is not None:
        invalid_checkpoint_epochs = [epoch for epoch in args.checkpoint_epochs if epoch < 1]
        if invalid_checkpoint_epochs:
            raise ValueError(f"--checkpoint-epochs values must be >= 1: {invalid_checkpoint_epochs}")
        args.checkpoint_epochs = sorted(set(int(epoch) for epoch in args.checkpoint_epochs))
    if args.teacher_backbone == "resnet18" and not (
        args.model_type.startswith("feature_ae_resnet18") or args.model_type.startswith("reverse_distill_resnet18")
    ):
        raise ValueError("--teacher-backbone resnet18 requires a feature_ae_resnet18* or reverse_distill_resnet18* model type.")
    if not 0 < args.lr_factor < 1:
        raise ValueError("--lr-factor must be in (0, 1).")
    if args.tile_size <= 0 or args.tile_train_stride <= 0:
        raise ValueError("--tile-size and --tile-train-stride must be positive.")
    if args.context_tile_size is not None and args.context_tile_size <= 0:
        raise ValueError("--context-tile-size must be positive when provided.")
    if (
        args.model_type
        in {
            "feature_ae_resnet18_dual_context",
            "feature_ae_resnet18_dual_context_gated",
            "reverse_distill_resnet18_dual_context_gated",
        }
        and args.context_tile_size is None
    ):
        raise ValueError(f"--model-type {args.model_type} requires --context-tile-size.")
    if args.tile_train_max_tiles_per_image is not None and args.tile_train_max_tiles_per_image < 1:
        raise ValueError("--tile-train-max-tiles-per-image must be >= 1 when provided.")
    if args.roi_predictions_dir is not None and args.preprocessing_mode != "tile_256_overlap":
        raise ValueError("--roi-predictions-dir is currently supported for --preprocessing-mode tile_256_overlap.")
    if args.roi_loss_weight < 0 or args.background_loss_weight < 0:
        raise ValueError("--roi-loss-weight and --background-loss-weight must be >= 0.")
    if args.roi_loss_weight <= 0 and args.background_loss_weight <= 0:
        raise ValueError("At least one ROI/background loss weight must be > 0.")
    if args.min_roi_ratio < 0 or args.min_roi_ratio > 1:
        raise ValueError("--min-roi-ratio must be in [0, 1].")
    if args.normal_structure_loss_weight < 0 or args.roi_border_loss_weight < 0:
        raise ValueError("--normal-structure-loss-weight and --roi-border-loss-weight must be >= 0.")
    if not 0 <= args.normal_structure_dark_percentile <= 100:
        raise ValueError("--normal-structure-dark-percentile must be in [0, 100].")
    if args.normal_structure_min_area < 1:
        raise ValueError("--normal-structure-min-area must be >= 1.")
    if args.normal_structure_max_area_ratio <= 0 or args.normal_structure_max_area_ratio > 1:
        raise ValueError("--normal-structure-max-area-ratio must be in (0, 1].")
    if args.normal_structure_dilate_radius < 0 or args.roi_border_radius < 0:
        raise ValueError("--normal-structure-dilate-radius and --roi-border-radius must be >= 0.")
    args.layer_loss_weights_resolved = parse_layer_loss_weights(args.layer_loss_weights, list(args.layers))

    profile_category = args.category or ("Casting_multiclass" if args.categories else None)
    augmentation_profile = resolve_augmentation_profile(args.augmentation_profile, profile_category)
    args.augmentation_profile_resolved = augmentation_profile.name
    args.augmentation_profile_params = augmentation_profile.to_dict()
    args.repeat_factor_resolved = resolve_repeat_factor(args.repeat_factor, augmentation_profile)
    if (args.all_categories or args.categories) and args.repeat_factor_resolved != 1:
        raise ValueError("--repeat-factor > 1 is only supported for single-category fine-tuning.")

    category = "global" if args.all_categories else ("selected" if args.categories else str(args.category))
    run_name = build_run_name(args)
    run_dir = feature_ae_root() / category / run_name
    ensure_run_dir_writable(run_dir, args.overwrite_run)

    train_df = load_feature_ae_training_data(args)
    train_df = maybe_limit(train_df, args.limit_train, DATA.random_seed)
    fit_df, val_df = split_train_val(train_df, args.val_fraction, DATA.random_seed)
    fit_df_unique = len(fit_df)
    fit_df = repeat_training_rows(fit_df, args.repeat_factor_resolved)

    train_transform = build_pixel_ae_transform(
        args.input_size,
        phase="train",
        normalization=args.normalization,
        augmentation_policy="none",
        augmentation_profile=augmentation_profile,
    )
    eval_transform = build_pixel_ae_transform(
        args.input_size,
        phase="eval",
        normalization=args.normalization,
        augmentation_policy="none",
        augmentation_profile=None,
    )
    context_transform = eval_transform
    roi_predictions_dir = resolve_path(args.roi_predictions_dir)
    roi_lookup = load_roi_lookup(
        roi_predictions_dir,
        threshold=args.roi_threshold,
        dilate_radius=args.roi_dilate_radius,
    )
    if args.preprocessing_mode == "tile_256_overlap":
        train_dataset = TiledFeatureDataset(
            fit_df,
            tile_size=args.tile_size,
            stride=args.tile_train_stride,
            sampling=args.tile_train_sampling,
            max_tiles_per_image=args.tile_train_max_tiles_per_image,
            image_transform=train_transform,
            model_input_size=args.input_size,
            context_tile_size=args.context_tile_size,
            context_transform=context_transform,
            seed=DATA.random_seed,
            roi_lookup=roi_lookup,
            min_roi_ratio=args.min_roi_ratio,
        )
        train_dataset.roi_loss_weight = float(args.roi_loss_weight)
        train_dataset.background_loss_weight = float(args.background_loss_weight)
        train_dataset.normal_structure_loss_weight = float(args.normal_structure_loss_weight)
        train_dataset.normal_structure_dark_percentile = float(args.normal_structure_dark_percentile)
        train_dataset.normal_structure_min_area = int(args.normal_structure_min_area)
        train_dataset.normal_structure_max_area_ratio = float(args.normal_structure_max_area_ratio)
        train_dataset.normal_structure_dilate_radius = int(args.normal_structure_dilate_radius)
        train_dataset.roi_border_loss_weight = float(args.roi_border_loss_weight)
        train_dataset.roi_border_radius = int(args.roi_border_radius)
    else:
        train_dataset = UnifiedAnomalyDataset(
            fit_df,
            input_size=args.input_size,
            include_masks=False,
            image_transform=train_transform,
        )
    train_loader = make_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = None
    if val_df is not None:
        if args.preprocessing_mode == "tile_256_overlap":
            val_dataset = TiledFeatureDataset(
                val_df,
                tile_size=args.tile_size,
                stride=args.tile_train_stride,
                sampling="all",
                max_tiles_per_image=None,
                image_transform=eval_transform,
                model_input_size=args.input_size,
                context_tile_size=args.context_tile_size,
                context_transform=context_transform,
                seed=DATA.random_seed,
                roi_lookup=roi_lookup,
                min_roi_ratio=args.min_roi_ratio,
            )
            val_dataset.roi_loss_weight = float(args.roi_loss_weight)
            val_dataset.background_loss_weight = float(args.background_loss_weight)
            val_dataset.normal_structure_loss_weight = float(args.normal_structure_loss_weight)
            val_dataset.normal_structure_dark_percentile = float(args.normal_structure_dark_percentile)
            val_dataset.normal_structure_min_area = int(args.normal_structure_min_area)
            val_dataset.normal_structure_max_area_ratio = float(args.normal_structure_max_area_ratio)
            val_dataset.normal_structure_dilate_radius = int(args.normal_structure_dilate_radius)
            val_dataset.roi_border_loss_weight = float(args.roi_border_loss_weight)
            val_dataset.roi_border_radius = int(args.roi_border_radius)
        else:
            val_dataset = UnifiedAnomalyDataset(
                val_df,
                input_size=args.input_size,
                include_masks=False,
                image_transform=eval_transform,
            )
        val_loader = make_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    device = resolve_device(args.device)
    teacher = ResNetTeacherFeatures(args.teacher_backbone, args.layers).to(device).eval()
    student = build_feature_autoencoder(args.model_type, args.layers).to(device)
    if args.init_checkpoint_path is not None:
        init_path = resolve_path(args.init_checkpoint_path)
        assert init_path is not None
        checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
        checkpoint_model_type = checkpoint.get("model_type", args.model_type)
        checkpoint_layers = list(checkpoint.get("layers", args.layers))
        compatible_base_load = (
            args.model_type
            in {
                "feature_ae_resnet18_dual_context",
                "feature_ae_resnet18_dual_context_gated",
            }
            and checkpoint_model_type == "feature_ae_resnet18"
        )
        if checkpoint_model_type != args.model_type and not compatible_base_load:
            raise ValueError(
                f"Checkpoint model_type={checkpoint_model_type!r} does not match requested model_type={args.model_type!r}."
            )
        if checkpoint_layers != list(args.layers):
            raise ValueError(f"Checkpoint layers={checkpoint_layers!r} do not match requested layers={list(args.layers)!r}.")
        load_result = student.load_state_dict(checkpoint["model_state_dict"], strict=not compatible_base_load)
        print(f"Loaded Feature-AE student weights from: {init_path}")
        if compatible_base_load:
            print(
                "Loaded local Feature-AE weights into dual-context model "
                f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})."
            )
    optimizer = AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
        )

    print(
        f"Training {args.model_type} | category={args.category} | scope={training_scope(args)} | "
        f"train items={len(train_dataset)} | train unique={fit_df_unique} | "
        f"val items={0 if val_loader is None else len(val_loader.dataset)} | layers={tuple(args.layers)} | "
        f"layer_loss_weights={args.layer_loss_weights_resolved} | "
        f"batch_size={args.batch_size} | device={device} | profile={args.augmentation_profile_resolved} | "
        f"preprocessing={args.preprocessing_mode}"
    )

    history: list[dict] = []
    best_val_loss: float | None = None
    best_epoch: int | None = None
    best_stopping_loss: float | None = None
    epochs_without_improvement = 0
    stopped_early = False
    show_progress = not args.no_progress
    metric_best_scores: dict[str, dict] = {}
    metric_eval_history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        lr_before = current_lr(optimizer)
        train_loss, train_metrics = run_epoch(
            student=student,
            teacher=teacher,
            loader=train_loader,
            device=device,
            cosine_weight=args.cosine_weight,
            layer_loss_weights=args.layer_loss_weights_resolved,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=args.epochs,
            phase="train",
            show_progress=show_progress,
        )
        val_loss = None
        val_metrics: dict[str, float] = {}
        if val_loader is not None:
            with torch.inference_mode():
                val_loss, val_metrics = run_epoch(
                    student=student,
                    teacher=teacher,
                    loader=val_loader,
                    device=device,
                    cosine_weight=args.cosine_weight,
                    layer_loss_weights=args.layer_loss_weights_resolved,
                    optimizer=None,
                    epoch=epoch,
                    total_epochs=args.epochs,
                    phase="val",
                    show_progress=show_progress,
                )

        monitor_loss = val_loss if val_loss is not None else train_loss
        is_best = val_loss is not None and (
            best_val_loss is None or val_loss < best_val_loss
        )
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch

        if best_stopping_loss is None or monitor_loss < best_stopping_loss - args.min_delta:
            best_stopping_loss = monitor_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(monitor_loss)
        lr_after = current_lr(optimizer)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "monitor_loss": monitor_loss,
            "learning_rate": lr_after,
            "learning_rate_before_scheduler": lr_before,
            "is_best": is_best,
            "epochs_without_improvement": epochs_without_improvement,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)

        if val_loss is None:
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} lr={lr_after:.6g}")
        else:
            marker = " *" if is_best else ""
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} lr={lr_after:.6g}{marker}"
            )
        if lr_after < lr_before:
            print(f"LR reduced: {lr_before:.6g} -> {lr_after:.6g}", flush=True)

        save_state(
            run_dir=run_dir,
            student=student,
            optimizer=optimizer,
            args=args,
            history=history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            completed_epochs=epoch,
            stopped_early=stopped_early,
            save_as_best=args.save_best and is_best,
        )
        epoch_checkpoint_path: Path | None = None
        should_save_periodic = args.checkpoint_every_epochs > 0 and epoch % args.checkpoint_every_epochs == 0
        should_save_listed = args.checkpoint_epochs is not None and epoch in set(args.checkpoint_epochs)
        if should_save_periodic or should_save_listed:
            epoch_checkpoint_path = save_periodic_checkpoint(
                run_dir=run_dir,
                student=student,
                optimizer=optimizer,
                args=args,
                history=history,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                completed_epochs=epoch,
                stopped_early=stopped_early,
            )
            print(f"Saved periodic checkpoint: {epoch_checkpoint_path}")

        should_metric_eval = (
            args.metric_eval_every_epochs > 0
            and epoch >= args.metric_eval_start_epoch
            and epoch % args.metric_eval_every_epochs == 0
        )
        if should_metric_eval:
            if epoch_checkpoint_path is None:
                epoch_checkpoint_path = save_periodic_checkpoint(
                    run_dir=run_dir,
                    student=student,
                    optimizer=optimizer,
                    args=args,
                    history=history,
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    completed_epochs=epoch,
                    stopped_early=stopped_early,
                )
                print(f"Saved metric-eval checkpoint: {epoch_checkpoint_path}")
            metrics = run_metric_evaluation(
                run_dir=run_dir,
                checkpoint_path=epoch_checkpoint_path,
                args=args,
                epoch=epoch,
            )
            improved = update_metric_best_checkpoints(
                run_dir=run_dir,
                checkpoint_path=epoch_checkpoint_path,
                epoch=epoch,
                metrics=metrics,
                best_scores=metric_best_scores,
            )
            metric_row = {
                "epoch": epoch,
                "checkpoint_path": str(epoch_checkpoint_path),
                **{key: metric_value(metrics, key) for key in METRIC_BEST_FILES},
            }
            metric_eval_history.append(metric_row)
            for key, value in metric_row.items():
                if key not in {"epoch", "checkpoint_path"}:
                    row[f"metric_eval_{key}"] = value
            row["metric_eval_checkpoint_path"] = str(epoch_checkpoint_path)
            pd.DataFrame(history).to_csv(run_dir / "loss_history.csv", index=False)
            pd.DataFrame(metric_eval_history).to_csv(run_dir / "metric_eval_history.csv", index=False)
            metric_text = " ".join(
                f"{key}={metric_value(metrics, key):.6f}"
                for key in METRIC_BEST_FILES
                if metric_value(metrics, key) is not None
            )
            improved_text = f" improved={','.join(improved)}" if improved else ""
            print(f"metric_eval epoch={epoch:03d} {metric_text}{improved_text}")

        if (
            args.early_stopping_patience is not None
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            print(f"Early stopping at epoch {epoch}.")
            save_state(
                run_dir=run_dir,
                student=student,
                optimizer=optimizer,
                args=args,
                history=history,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                completed_epochs=epoch,
                stopped_early=stopped_early,
                save_as_best=False,
            )
            break

    print(f"Saved checkpoint to: {run_dir / 'checkpoint.pt'}")


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


