"""Minimal PatchCore-style baseline.

This module intentionally implements a small, explicit baseline:
- train on normal training images only;
- extract local features from intermediate CNN layers;
- score test images by nearest-neighbor distance to a memory bank;
- evaluate image-level and pixel-level metrics when masks are available.

It is not a full reproduction of the PatchCore paper, but it keeps the
important one-class behavior and is a good first benchmark for this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import models, transforms

from src.config import DATA, EDA, PATHS
from src.runtime import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    project_path as resolve_project_path,
    resolve_device as resolve_torch_device,
)
from src.models.pixel_ae import build_pixel_autoencoder, build_resnet


@dataclass(frozen=True)
class PatchCoreParams:
    category: str
    input_size: int = 256
    feature_extractor: str = "resnet_frozen"
    backbone: str = "resnet18"
    layers: tuple[str, ...] = ("layer2", "layer3")
    checkpoint_path: str | None = None
    batch_size: int = 16
    max_memory_embeddings: int = 100_000
    neighbor_batch_size: int = 4_096
    neighbor_memory_batch_size: int = 8_192
    num_neighbors: int = 1
    num_workers: int = 0
    device: str = "auto"
    seed: int = DATA.random_seed


def project_path(rel_path: str | Path) -> Path:
    """Resolve CSV paths that may contain Windows separators."""
    return resolve_project_path(rel_path)


def resolve_device(device: str = "auto") -> torch.device:
    return resolve_torch_device(device)


class ResizeLetterbox:
    """Resize while preserving aspect ratio, then pad to a square canvas."""

    def __init__(self, size: int, fill: int | tuple[int, int, int] = 0):
        self.size = int(size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        scale = self.size / max(width, height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resample = getattr(Image, "Resampling", Image).BILINEAR
        resized = image.resize((new_width, new_height), resample=resample)

        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2
        canvas.paste(resized, (left, top))
        return canvas


class ResizeMaskLetterbox:
    """Letterbox transform for binary masks, preserving nearest pixels."""

    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, mask: Image.Image) -> np.ndarray:
        mask = mask.convert("L")
        width, height = mask.size
        scale = self.size / max(width, height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resample = getattr(Image, "Resampling", Image).NEAREST
        resized = mask.resize((new_width, new_height), resample=resample)

        canvas = Image.new("L", (self.size, self.size), 0)
        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2
        canvas.paste(resized, (left, top))
        return (np.asarray(canvas) > EDA.mask_threshold).astype(np.uint8)


def build_image_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            ResizeLetterbox(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class UnifiedAnomalyDataset(Dataset):
    """Dataset backed by data/processed/unified_dataset.csv."""

    def __init__(
        self,
        df: pd.DataFrame,
        input_size: int,
        include_masks: bool = False,
        image_transform: transforms.Compose | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_transform = image_transform or build_image_transform(input_size)
        self.mask_transform = ResizeMaskLetterbox(input_size)
        self.include_masks = include_masks

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image = Image.open(project_path(row["image_path"])).convert("RGB")
        item = {
            "image": self.image_transform(image),
            "is_anomaly": torch.tensor(bool(row["is_anomaly"]), dtype=torch.long),
            "image_path": row["image_path"],
            "category": row["category"],
        }

        if self.include_masks:
            if bool(row["has_mask"]) and pd.notna(row["mask_path"]):
                mask = Image.open(project_path(row["mask_path"]))
                mask_arr = self.mask_transform(mask)
            else:
                mask_arr = np.zeros(
                    (self.mask_transform.size, self.mask_transform.size),
                    dtype=np.uint8,
                )
            item["mask"] = torch.from_numpy(mask_arr)

        return item


def collate_batch(batch: list[dict]) -> dict:
    out = {
        "image": torch.stack([x["image"] for x in batch]),
        "is_anomaly": torch.stack([x["is_anomaly"] for x in batch]),
        "image_path": [x["image_path"] for x in batch],
        "category": [x["category"] for x in batch],
    }
    if "mask" in batch[0]:
        out["mask"] = torch.stack([x["mask"] for x in batch])
    return out


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    sampler: Sampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )


def load_unified_dataset(category: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(PATHS.unified_csv)
    if category is not None:
        df = df[df["category"] == category].copy()
    return df.reset_index(drop=True)


def split_category_data(category: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_unified_dataset(category)
    train_normal = df[(df["split"] == "train") & (~df["is_anomaly"])].copy()
    test = df[df["split"] == "test"].copy()
    if train_normal.empty:
        raise ValueError(f"No normal training images found for category={category!r}")
    if test.empty:
        raise ValueError(f"No test images found for category={category!r}")
    return train_normal.reset_index(drop=True), test.reset_index(drop=True)


class FeatureExtractorProtocol(Protocol):
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Return local feature maps with shape (B, C, Hf, Wf)."""


class CNNFeatureExtractor(nn.Module):
    """Extract and concatenate local feature maps from selected layers."""

    def __init__(
        self,
        backbone: str,
        layers: Iterable[str],
        encoder_state_dict: dict | None = None,
    ):
        super().__init__()
        self.model = build_resnet(backbone).eval()
        if encoder_state_dict is not None:
            self.model.load_state_dict(encoder_state_dict)
        self.layers = tuple(layers)
        self._features: dict[str, torch.Tensor] = {}
        modules = dict(self.model.named_modules())

        for layer_name in self.layers:
            if layer_name not in modules:
                raise ValueError(f"Unknown layer {layer_name!r} for backbone {backbone!r}")
            modules[layer_name].register_forward_hook(self._make_hook(layer_name))

        for param in self.model.parameters():
            param.requires_grad_(False)

    def _make_hook(self, layer_name: str):
        def hook(_module, _inputs, output):
            self._features[layer_name] = output.detach()

        return hook

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self._features = {}
        _ = self.model(images)

        maps = [self._features[name] for name in self.layers]
        target_hw = maps[0].shape[-2:]
        resized = [
            fmap if fmap.shape[-2:] == target_hw
            else F.interpolate(fmap, size=target_hw, mode="bilinear", align_corners=False)
            for fmap in maps
        ]
        return torch.cat(resized, dim=1)


class CustomAutoencoderFeatureExtractor(nn.Module):
    """Feature extractor backed by a trained custom convolutional autoencoder."""

    def __init__(self, checkpoint_path: str | Path):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_type = checkpoint.get("model_type")
        if model_type is None:
            model_type = checkpoint.get("params", {}).get("model_type", "custom_autoencoder")
        if model_type not in {
            "custom_autoencoder",
            "custom_unet_autoencoder",
            "custom_constrained_unet_autoencoder",
        }:
            raise ValueError(
                "custom_autoencoder feature extraction expects a custom AE checkpoint, "
                f"got model_type={model_type!r}."
            )
        self.model = build_pixel_autoencoder(model_type)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.encode(images)


def build_feature_extractor(params: PatchCoreParams) -> nn.Module:
    """Create a local feature extractor for PatchCore."""
    method = params.feature_extractor
    if method == "resnet_frozen":
        return CNNFeatureExtractor(params.backbone, params.layers)

    if method == "custom_autoencoder":
        if params.checkpoint_path is None:
            raise ValueError("custom_autoencoder requires checkpoint_path.")
        return CustomAutoencoderFeatureExtractor(params.checkpoint_path)

    if method == "resnet_reconstruction_finetuned":
        if params.checkpoint_path is None:
            raise ValueError("resnet_reconstruction_finetuned requires checkpoint_path.")
        checkpoint = torch.load(params.checkpoint_path, map_location="cpu", weights_only=False)
        return CNNFeatureExtractor(
            params.backbone,
            params.layers,
            encoder_state_dict=checkpoint["encoder_state_dict"],
        )

    raise ValueError(
        "Unknown feature_extractor. Use resnet_frozen, custom_autoencoder or "
        "resnet_reconstruction_finetuned."
    )


class PatchCoreModel:
    """Nearest-neighbor PatchCore baseline."""

    def __init__(self, params: PatchCoreParams):
        self.params = params
        self.device = resolve_device(params.device)
        self.extractor = build_feature_extractor(params).to(self.device).eval()
        self.memory_bank: np.ndarray | None = None

    @staticmethod
    def _feature_map_to_embeddings(feature_map: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B*H*W, C)
        embeddings = feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.shape[1])
        return F.normalize(embeddings, p=2, dim=1)

    def _sample_memory_bank(self, dataloader: DataLoader) -> np.ndarray:
        max_embeddings = int(self.params.max_memory_embeddings)
        if max_embeddings <= 0:
            raise ValueError("max_memory_embeddings must be a positive integer.")

        rng = np.random.default_rng(self.params.seed)
        reservoir: np.ndarray | None = None
        fill_count = 0
        seen = 0

        for batch in dataloader:
            images = batch["image"].to(self.device, non_blocking=True)
            feature_map = self.extractor(images)
            embeddings = (
                self._feature_map_to_embeddings(feature_map)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

            batch_size = len(embeddings)
            if batch_size == 0:
                continue

            if reservoir is None:
                reservoir = np.empty((max_embeddings, embeddings.shape[1]), dtype=np.float32)

            if fill_count < max_embeddings:
                available = max_embeddings - fill_count
                take = min(batch_size, available)
                reservoir[fill_count : fill_count + take] = embeddings[:take]
                fill_count += take
                seen += take
            else:
                take = 0

            for batch_index in range(take, batch_size):
                seen += 1
                slot = int(rng.integers(0, seen))
                if slot < max_embeddings:
                    reservoir[slot] = embeddings[batch_index]

            del images, feature_map, embeddings
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if reservoir is None or len(reservoir) == 0:
            raise ValueError("No embeddings were extracted for the memory bank.")
        return reservoir[:fill_count].astype(np.float32, copy=False)

    def _kneighbors_chunked(self, embeddings: np.ndarray) -> np.ndarray:
        if self.memory_bank is None:
            raise RuntimeError("PatchCoreModel must be fitted before predict().")
        if self.params.num_neighbors != 1:
            raise ValueError("Only num_neighbors=1 is supported by the memory-bounded scorer.")

        chunk_size = int(self.params.neighbor_batch_size)
        if chunk_size <= 0:
            raise ValueError("neighbor_batch_size must be a positive integer.")
        memory_chunk_size = int(self.params.neighbor_memory_batch_size)
        if memory_chunk_size <= 0:
            raise ValueError("neighbor_memory_batch_size must be a positive integer.")

        distances: list[np.ndarray] = []
        memory = torch.from_numpy(self.memory_bank.astype(np.float32, copy=False))
        for start in range(0, len(embeddings), chunk_size):
            query = torch.from_numpy(
                embeddings[start : start + chunk_size].astype(np.float32, copy=False)
            )
            best_squared: torch.Tensor | None = None
            for mem_start in range(0, len(memory), memory_chunk_size):
                memory_chunk = memory[mem_start : mem_start + memory_chunk_size]
                similarity = query @ memory_chunk.T
                squared = (2.0 - 2.0 * similarity).clamp_min_(0.0)
                chunk_best = squared.min(dim=1).values
                if best_squared is None:
                    best_squared = chunk_best
                else:
                    best_squared = torch.minimum(best_squared, chunk_best)

            if best_squared is None:
                raise RuntimeError("Memory bank is empty.")
            distances.append(torch.sqrt(best_squared).numpy().astype(np.float32, copy=False))
        return np.concatenate(distances, axis=0)

    def fit(self, dataloader: DataLoader) -> None:
        self.memory_bank = self._sample_memory_bank(dataloader)

    @torch.inference_mode()
    def predict(self, dataloader: DataLoader, *, store_score_maps: bool = True) -> dict:
        if self.memory_bank is None:
            raise RuntimeError("PatchCoreModel must be fitted before predict().")

        image_scores: list[float] = []
        y_true: list[int] = []
        image_paths: list[str] = []
        score_maps: list[np.ndarray] = []
        masks: list[np.ndarray] = []

        for batch in dataloader:
            images = batch["image"].to(self.device, non_blocking=True)
            feature_map = self.extractor(images)
            batch_size, _channels, h_feat, w_feat = feature_map.shape
            embeddings = (
                self._feature_map_to_embeddings(feature_map)
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            distances = self._kneighbors_chunked(embeddings)
            patch_scores = distances.reshape(batch_size, h_feat, w_feat)

            score_tensor = torch.from_numpy(patch_scores[:, None, :, :]).float()
            score_tensor = F.interpolate(
                score_tensor,
                size=(self.params.input_size, self.params.input_size),
                mode="bilinear",
                align_corners=False,
            )
            upsampled = score_tensor[:, 0].numpy()

            if store_score_maps:
                score_maps.extend([m.astype(np.float32, copy=False) for m in upsampled])
            image_scores.extend([float(m.max()) for m in upsampled])
            y_true.extend(batch["is_anomaly"].cpu().numpy().astype(int).tolist())
            image_paths.extend(batch["image_path"])
            if store_score_maps and "mask" in batch:
                masks.extend(batch["mask"].cpu().numpy().astype(np.uint8))
            del images, feature_map, embeddings, distances, patch_scores, score_tensor
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return {
            "image_path": np.array(image_paths, dtype=object),
            "y_true": np.array(y_true, dtype=np.int64),
            "image_score": np.array(image_scores, dtype=np.float32),
            "score_maps": np.stack(score_maps, axis=0) if score_maps else None,
            "masks": np.stack(masks, axis=0) if masks else None,
        }

    def save_memory_bank(self, output_path: Path) -> None:
        if self.memory_bank is None:
            raise RuntimeError("No memory bank to save. Call fit() first.")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            memory_bank=self.memory_bank,
            params=np.array([self.params.__dict__], dtype=object),
        )


def evaluate_predictions(predictions: dict) -> dict[str, float]:
    y_true = predictions["y_true"]
    image_score = predictions["image_score"]
    metrics: dict[str, float] = {}

    if len(np.unique(y_true)) == 2:
        metrics["image_auroc"] = float(roc_auc_score(y_true, image_score))
        metrics["image_ap"] = float(average_precision_score(y_true, image_score))

    masks = predictions.get("masks")
    score_maps = predictions.get("score_maps")
    if masks is not None and score_maps is not None and masks.sum() > 0:
        pixel_true = masks.reshape(-1).astype(np.uint8)
        pixel_score = score_maps.reshape(-1).astype(np.float32)
        if len(np.unique(pixel_true)) == 2:
            metrics["pixel_auroc"] = float(roc_auc_score(pixel_true, pixel_score))
            metrics["pixel_ap"] = float(average_precision_score(pixel_true, pixel_score))
            aupimo = _normalized_low_fpr_aupimo(
                np.asarray(y_true, dtype=np.int64),
                np.asarray(masks),
                np.asarray(score_maps, dtype=np.float32),
                fpr_low=1e-5,
                fpr_high=1e-3,
            )
            if aupimo is not None:
                metrics["pixel_aupimo_1e-5_1e-3"] = float(aupimo)

    return metrics


def _normalized_low_fpr_aupimo(
    y_true: np.ndarray,
    masks: np.ndarray,
    score_maps: np.ndarray,
    *,
    fpr_low: float = 1e-5,
    fpr_high: float = 1e-3,
    num_thresholds: int = 200,
) -> float | None:
    """Approximate normalized AUPIMO on a low false-positive-rate interval.

    PIMO is computed by calibrating thresholds on normal-image pixels, then
    measuring per-anomalous-image pixel overlap at the same thresholds. The
    returned area is normalized by the log-FPR interval, so values stay in [0, 1].
    """
    normal_scores = np.asarray(score_maps[y_true == 0], dtype=np.float32)
    anomalous_scores = np.asarray(score_maps[y_true == 1], dtype=np.float32)
    anomalous_masks = np.asarray(masks[y_true == 1], dtype=bool)
    valid_anomaly = anomalous_masks.reshape(anomalous_masks.shape[0], -1).sum(axis=1) > 0
    anomalous_scores = anomalous_scores[valid_anomaly]
    anomalous_masks = anomalous_masks[valid_anomaly]
    if normal_scores.size == 0 or anomalous_scores.size == 0:
        return None

    fprs = np.geomspace(float(fpr_low), float(fpr_high), int(num_thresholds))
    normal_flat = normal_scores.reshape(-1)
    thresholds = np.quantile(normal_flat, 1.0 - fprs, method="higher")

    anomaly_flat = anomalous_scores.reshape(anomalous_scores.shape[0], -1)
    mask_flat = anomalous_masks.reshape(anomalous_masks.shape[0], -1)
    positives_per_image = mask_flat.sum(axis=1).astype(np.float32)
    pimo_values = []
    for threshold in thresholds:
        detected = (anomaly_flat >= threshold) & mask_flat
        per_image_overlap = detected.sum(axis=1).astype(np.float32) / positives_per_image
        pimo_values.append(float(per_image_overlap.mean()))

    log_fprs = np.log10(fprs)
    y_values = np.asarray(pimo_values, dtype=np.float32)
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(y_values, x=log_fprs)
    else:
        area = np.sum((y_values[1:] + y_values[:-1]) * np.diff(log_fprs) * 0.5)
    denom = float(log_fprs[-1] - log_fprs[0])
    if denom <= 0:
        return None
    return float(np.clip(area / denom, 0.0, 1.0))


def save_predictions(output_path: Path, predictions: dict, metrics: dict[str, float]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_path": predictions["image_path"],
        "y_true": predictions["y_true"],
        "image_score": predictions["image_score"],
        "metrics": np.array([metrics], dtype=object),
    }
    if predictions.get("score_maps") is not None:
        payload["score_maps"] = predictions["score_maps"]
    if predictions.get("masks") is not None:
        payload["masks"] = predictions["masks"]
    np.savez_compressed(output_path, **payload)






