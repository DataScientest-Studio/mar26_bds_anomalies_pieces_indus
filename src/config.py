"""Central project configuration.

`data/` and `models/` are local workspaces and are intentionally ignored by
Git. Traceable registries for those assets live under `reports/registries/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    """Project path registry used by scripts and reusable modules."""

    root: Path = ROOT

    data_dir: Path = ROOT / "data"
    models_dir: Path = ROOT / "models"
    reports_dir: Path = ROOT / "reports"
    scripts_dir: Path = ROOT / "scripts"

    raw_dir: Path = ROOT / "data" / "raw"
    mvtec_dir: Path = ROOT / "data" / "raw" / "mvtec"
    hssiad_dir: Path = ROOT / "data" / "raw" / "hss-iad"

    processed_dir: Path = ROOT / "data" / "processed"
    classified_dir: Path = ROOT / "data" / "classified"
    unified_csv: Path = ROOT / "data" / "processed" / "unified_dataset.csv"
    resolutions_csv: Path = ROOT / "data" / "processed" / "resolutions_sample.csv"

    registries_dir: Path = ROOT / "reports" / "registries"
    evidence_dir: Path = ROOT / "reports" / "evidence"
    casting_surface_reports_dir: Path = ROOT / "reports" / "casting_surface_features"


PATHS = Paths()


@dataclass(frozen=True)
class DataConfig:
    """Dataset scanning defaults."""

    img_extensions: frozenset[str] = frozenset(
        {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    splits: tuple[str, ...] = ("train", "test")
    normal_label: str = "good"
    ground_truth_dirname: str = "ground_truth"

    mvtec_name: str = "mvtec"
    hssiad_name: str = "hss-iad"

    resolutions_sample_size: int = 2000
    random_seed: int = 42


DATA = DataConfig()


@dataclass(frozen=True)
class EDAConfig:
    """EDA and default visualization parameters."""

    sns_style: str = "whitegrid"
    sns_palette: str = "muted"
    sns_font_scale: float = 1.1
    figure_dpi: int = 120

    color_mvtec: str = "#4C72B0"
    color_hssiad: str = "#DD8452"
    color_normal: str = "#55A868"
    color_anomal: str = "#C44E52"

    n_example_categories: int = 4
    example_seed: int = 42

    mask_threshold: int = 127
    min_component_px: int = 5
    resize_targets: tuple[int, ...] = (224, 256, 384, 512)
    resize_loss_threshold_px: int = 5
    resize_risk_threshold_px: int = 10

    pixel_stats_per_category: int = 30
    mask_heatmap_size: int = 128


EDA = EDAConfig()


@dataclass(frozen=True)
class FeatureConfig:
    """Default feature-extraction parameters used by baseline feature extractors."""

    default_input_size: int = 224
    default_backbone: str = "resnet18"
    default_layer: str = "avgpool"
    default_pooling: str = "avg"
    batch_size: int = 32
    device: str = "auto"
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


FEATURES = FeatureConfig()


if __name__ == "__main__":
    print(f"ROOT        = {PATHS.root}")
    print(f"Data dir    = {PATHS.data_dir}")
    print(f"Models dir  = {PATHS.models_dir}")
    print(f"Reports dir = {PATHS.reports_dir}")
