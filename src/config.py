"""
Configuration centrale du projet.

Regroupe uniquement les paramètres actuellement utilisés par le code
(harmonisation + notebook EDA). À enrichir au fur et à mesure que le
projet avance (pipeline modèles, entraînement, etc.).

Usage:
    from src.config import PATHS, DATA, EDA
"""

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Racine du projet (src/config.py -> remonte d'un niveau)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    root: Path = ROOT

    # Données brutes
    mvtec_dir: Path = ROOT / "data" / "raw" / "mvtec"
    hssiad_dir: Path = ROOT / "data" / "raw" / "hss-iad"

    # Données traitées
    processed_dir: Path = ROOT / "data" / "processed"
    unified_csv: Path = ROOT / "data" / "processed" / "unified_dataset.csv"
    resolutions_csv: Path = ROOT / "data" / "processed" / "resolutions_sample.csv"


PATHS = Paths()


# ---------------------------------------------------------------------------
# Paramètres harmonisation (utilisés par src/data/harmonize.py)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    # Extensions d'images acceptées lors du scan
    img_extensions: frozenset = frozenset(
        {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )

    # Structure MVTec-style
    splits: tuple = ("train", "test")
    normal_label: str = "good"
    ground_truth_dirname: str = "ground_truth"

    # Noms utilisés dans la colonne "dataset" du DataFrame unifié
    mvtec_name: str = "mvtec"
    hssiad_name: str = "hss-iad"

    # Échantillonnage pour le relevé des résolutions
    resolutions_sample_size: int = 2000
    random_seed: int = 42


DATA = DataConfig()


# ---------------------------------------------------------------------------
# Paramètres EDA (utilisés par notebooks/01_eda_harmonisation.ipynb)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EDAConfig:
    # Style matplotlib / seaborn
    sns_style: str = "whitegrid"
    sns_palette: str = "muted"
    sns_font_scale: float = 1.1
    figure_dpi: int = 120

    # Couleurs datasets (MVTec / HSS-IAD)
    color_mvtec: str = "#4C72B0"
    color_hssiad: str = "#DD8452"

    # Couleurs status (Normal / Anomal)
    color_normal: str = "#55A868"
    color_anomal: str = "#C44E52"

    # Exemples visuels
    n_example_categories: int = 4
    example_seed: int = 42


EDA = EDAConfig()


if __name__ == "__main__":
    print(f"ROOT          = {PATHS.root}")
    print(f"MVTec dir     = {PATHS.mvtec_dir}")
    print(f"HSS-IAD dir   = {PATHS.hssiad_dir}")
    print(f"Unified CSV   = {PATHS.unified_csv}")
    print(f"Resolutions   = {PATHS.resolutions_csv}")
