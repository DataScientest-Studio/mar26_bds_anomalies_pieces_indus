"""Module partagé entre main.py et les pages : pipelines cachés + helpers.

`@st.cache_resource` est lié à l'identité de la fonction décorée — pour que main.py
et les pages partagent le MÊME cache, ils doivent tous appeler ces fonctions
définies ici (pas redéfinir leurs propres versions).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import PATHS
from src.inference.pipeline import AnomalyPipeline

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

# Catégories pré-chargées au boot pour la démo. Les autres restent disponibles
# dans le sélecteur de la page MVTec mais sont chargées paresseusement au 1er clic.
DEMO_CATEGORIES = ["cable", "leather", "transistor"]

MODEL_OPTIONS = {
    "Ensemble Mean": "ensemble_mean",
    "Dinomaly seul": "dinomaly",
    "PatchCore seul": "patchcore",
}


@st.cache_resource(show_spinner="Chargement du modèle…")
def get_pipeline(category: str, model_name: str) -> AnomalyPipeline:
    return AnomalyPipeline.from_category(category=category, model_name=model_name)


@st.cache_resource(show_spinner="Chargement ROI + RD AE…")
def get_casting_pipeline():
    from src.inference.casting_feature_ae import CastingFeatureAEPipeline
    return CastingFeatureAEPipeline(device="auto")


def list_available_mvtec_categories(filter_to: list[str] | None = None) -> list[str]:
    """Catégories MVTec dont le checkpoint Dinomaly existe localement.

    Si `filter_to` est fourni, ne retourne que les catégories de cette liste.
    """
    ckpt_dir = PATHS.root / "models"
    if not ckpt_dir.exists():
        return []
    pool = filter_to if filter_to is not None else MVTEC_CATEGORIES
    return [
        cat for cat in pool
        if any(ckpt_dir.glob(f"dinomaly_{cat}*ckpt"))
    ]


def warmup_all_models(categories: list[str] | None = None) -> tuple[int, int]:
    """Pré-charge les pipelines pour la liste `categories` + HSS-IAD Casting.

    Par défaut warm uniquement `DEMO_CATEGORIES` (cable, leather, transistor) pour
    rester rapide. Les autres catégories MVTec sont chargées paresseusement au 1er
    clic dans la page MVTec.

    Idempotent : grâce à `@st.cache_resource`, les appels successifs sont gratuits.
    """
    if categories is None:
        categories = DEMO_CATEGORIES
    available = list_available_mvtec_categories(filter_to=categories)
    total = len(available) + 1  # +1 pour HSS-IAD Casting
    loaded = 0

    progress = st.progress(0.0, text="Initialisation des modèles démo…")

    for i, cat in enumerate(available):
        progress.progress(
            i / total,
            text=f"MVTec ({i + 1}/{len(available)}) — `{cat}` (Dinomaly + PatchCore + calibration)",
        )
        try:
            get_pipeline(category=cat, model_name="ensemble_mean")
            loaded += 1
        except Exception as e:
            st.warning(f"⚠ MVTec `{cat}` ignoré : {e}")

    progress.progress(len(available) / total, text="HSS-IAD Casting (ROI + RD/AE)…")
    try:
        get_casting_pipeline()
        loaded += 1
    except Exception as e:
        st.warning(f"⚠ HSS-IAD Casting ignoré : {e}")

    progress.progress(1.0, text="✓ Pré-chargement terminé")
    progress.empty()
    return loaded, total
