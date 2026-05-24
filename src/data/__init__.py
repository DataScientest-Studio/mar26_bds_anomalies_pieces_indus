"""Data loading public API."""

from src.data.anomaly_dataset import UnifiedAnomalyDataset, make_dataloader, split_category_data
from src.features.augmentation_profiles import AugmentationProfile, resolve_augmentation_profile

__all__ = [
    "AugmentationProfile",
    "FunctionalSurfaceLabelDataset",
    "SemanticSurfaceDataset",
    "UnifiedAnomalyDataset",
    "make_dataloader",
    "resolve_augmentation_profile",
    "split_category_data",
]


def __getattr__(name: str):
    if name in {"FunctionalSurfaceLabelDataset", "SemanticSurfaceDataset"}:
        from src.data.mask_datasets import FunctionalSurfaceLabelDataset, SemanticSurfaceDataset

        return {
            "FunctionalSurfaceLabelDataset": FunctionalSurfaceLabelDataset,
            "SemanticSurfaceDataset": SemanticSurfaceDataset,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
