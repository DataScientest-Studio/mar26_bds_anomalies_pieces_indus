"""Surface/landmark datasets and dataset API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.models.segmentation.training import SemanticSurfaceDataset, collate as semantic_collate
from src.models.baselines.patchcore import IMAGENET_MEAN, IMAGENET_STD, ResizeLetterbox, ResizeMaskLetterbox, project_path


def bbox_from_positive_mask(positive: torch.Tensor, min_positive_ratio: float = 0.005) -> tuple[torch.Tensor, torch.Tensor]:
    mask = positive.detach()
    if mask.ndim == 3:
        mask = mask[0]
    binary = mask > 0.5
    positive_ratio = float(binary.float().mean())
    if positive_ratio < float(min_positive_ratio) or not bool(binary.any()):
        return torch.zeros(4, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)
    ys, xs = torch.where(binary)
    height, width = binary.shape
    x0 = xs.float().min()
    x1 = xs.float().max() + 1.0
    y0 = ys.float().min()
    y1 = ys.float().max() + 1.0
    bbox = torch.tensor(
        [
            ((x0 + x1) * 0.5) / float(width),
            ((y0 + y1) * 0.5) / float(height),
            (x1 - x0) / float(width),
            (y1 - y0) / float(height),
        ],
        dtype=torch.float32,
    )
    return bbox.clamp(0.0, 1.0), torch.tensor(1.0, dtype=torch.float32)


def add_detection_targets(item: dict, min_positive_ratio: float = 0.005) -> dict:
    bbox, objectness = bbox_from_positive_mask(item["positive"], min_positive_ratio)
    item["bbox"] = bbox
    item["objectness"] = objectness
    return item


class FunctionalSurfaceLabelDataset(Dataset):
    """Binary functional-surface labels dataset used by evaluation and correction tools."""

    def __init__(
        self,
        df: pd.DataFrame,
        input_size: int,
        *,
        augmentation_profile=None,
        repeat_factor: int = 1,
        positive_mask_column: str = "positive_mask_path",
        ignore_mask_column: str = "ignore_mask_path",
        pseudo_mask_column: str = "pseudo_mask_path",
        negative_mask_column: str = "negative_mask_path",
        weight_map_column: str = "weight_map_path",
        min_bbox_positive_ratio: float = 0.005,
        **_unused,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.input_size = int(input_size)
        self.augmentation_profile = augmentation_profile
        self.repeat_factor = max(1, int(repeat_factor))
        self.positive_mask_column = positive_mask_column
        self.ignore_mask_column = ignore_mask_column
        self.pseudo_mask_column = pseudo_mask_column
        self.negative_mask_column = negative_mask_column
        self.weight_map_column = weight_map_column
        self.min_bbox_positive_ratio = float(min_bbox_positive_ratio)
        self.image_transform = transforms.Compose(
            [
                ResizeLetterbox(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.mask_transform = ResizeMaskLetterbox(self.input_size)

    def __len__(self) -> int:
        return len(self.df) * self.repeat_factor

    def _row(self, index: int) -> pd.Series:
        return self.df.iloc[index % len(self.df)]

    @staticmethod
    def _has_path(row: pd.Series, column: str) -> bool:
        return column in row and pd.notna(row.get(column, None)) and bool(str(row.get(column, "")).strip())

    def _empty_mask(self) -> torch.Tensor:
        return torch.zeros((1, self.input_size, self.input_size), dtype=torch.float32)

    def _mask(self, path: str | Path) -> torch.Tensor:
        image = Image.open(project_path(path)).convert("L")
        arr = self.mask_transform(image).astype(np.float32)
        return torch.from_numpy(arr[None, ...])

    def _soft_mask(self, path: str | Path) -> torch.Tensor:
        image = Image.open(project_path(path)).convert("L")
        width, height = image.size
        scale = self.input_size / max(width, height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = image.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
        canvas = Image.new("L", (self.input_size, self.input_size), 0)
        canvas.paste(resized, ((self.input_size - new_width) // 2, (self.input_size - new_height) // 2))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        return torch.from_numpy(arr[None, ...])

    def _optional_mask(self, row: pd.Series, column: str) -> torch.Tensor:
        if self._has_path(row, column):
            return self._mask(str(row[column]))
        return self._empty_mask()

    def _pseudo_mask(self, row: pd.Series) -> torch.Tensor:
        if self._has_path(row, "target_prior_path"):
            return self._soft_mask(str(row["target_prior_path"]))
        if not self._has_path(row, self.pseudo_mask_column):
            return self._optional_mask(row, self.positive_mask_column)
        if str(row.get("pseudo_is_soft", "")).lower() in {"1", "true", "yes", "y"}:
            return self._soft_mask(str(row[self.pseudo_mask_column]))
        return self._mask(str(row[self.pseudo_mask_column]))

    def _weight_map(self, row: pd.Series) -> torch.Tensor:
        if self._has_path(row, self.weight_map_column):
            return self._soft_mask(str(row[self.weight_map_column])).float()
        return torch.ones((1, self.input_size, self.input_size), dtype=torch.float32)

    def __getitem__(self, index: int) -> dict:
        row = self._row(index)
        image = Image.open(project_path(row["image_path"])).convert("RGB")
        item = {
            "image": self.image_transform(image),
            "positive": self._optional_mask(row, self.positive_mask_column).float(),
            "ignore": self._optional_mask(row, self.ignore_mask_column).float(),
            "negative": self._optional_mask(row, self.negative_mask_column).float(),
            "pseudo": self._pseudo_mask(row).float(),
            "weight": self._weight_map(row).float(),
            "row": row.to_dict(),
            "image_path": str(row["image_path"]),
        }
        return add_detection_targets(item, self.min_bbox_positive_ratio)


def collate(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    output: dict = {}
    for key in keys:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            output[key] = torch.stack(values)
        else:
            output[key] = values
    return output


__all__ = [
    "FunctionalSurfaceLabelDataset",
    "SemanticSurfaceDataset",
    "add_detection_targets",
    "bbox_from_positive_mask",
    "collate",
    "semantic_collate",
]







