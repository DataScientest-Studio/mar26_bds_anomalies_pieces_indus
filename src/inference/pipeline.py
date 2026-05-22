"""Pipeline d'inférence haut-niveau pour Streamlit / scripts.

Charge les modèles (cache via attribut), fait la prédiction sur une image,
retourne heatmap + score + overlay prêt à afficher.

Usage type :
    pipe = AnomalyPipeline.from_category('cable', model_name='ensemble_mean')
    result = pipe.predict(pil_image)
    st.image(result.overlay)
    st.metric("Score", result.score)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from skimage.transform import resize as sk_resize

from src.config import PATHS
from src.models.dinomaly_wrapper import (
    IMG_SIZE_DEFAULT,
    load_dinomaly_ckpt,
    score_image_manual,
    score_paths_manual,
)
from src.models.ensemble import (
    ensemble_max,
    ensemble_mean,
    norm_global_minmax,
)
from src.models.patchcore_manual import IMG_SIZE_DEFAULT as PC_IMG_SIZE_DEFAULT
from src.models.patchcore_manual import PatchCoreManual

ModelName = Literal["dinomaly", "patchcore", "ensemble_mean", "ensemble_max"]


@dataclass
class PredictionResult:
    """Résultat d'une inférence sur une image."""

    score: float  # score image-level (max de la heatmap normalisée)
    heatmap: np.ndarray  # (H, W) float32 — heatmap brute du modèle choisi
    heatmap_norm: np.ndarray  # (H, W) float32 — heatmap normalisée global-minmax [0,1]
    overlay: np.ndarray  # (H, W, 4) uint8 — heatmap colormap jet + alpha, prête à display
    image_resized: np.ndarray  # (H, W, 3) uint8 — image originale resize à la heatmap shape
    model_name: str
    category: str


def _to_jet_overlay(heatmap_norm: np.ndarray) -> np.ndarray:
    """Convertit une heatmap normalisée [0,1] en RGBA uint8 colormap jet."""
    import matplotlib.cm as cm

    cmap = cm.get_cmap("jet")
    rgba = (cmap(np.clip(heatmap_norm, 0, 1)) * 255).astype(np.uint8)
    return rgba


class AnomalyPipeline:
    """Pipeline unifié Dinomaly / PatchCore / Ensemble pour une catégorie donnée.

    Construction lente (chargement ckpt + memory bank), prédiction rapide.
    Cache à utiliser via Streamlit `@st.cache_resource`.
    """

    def __init__(
        self,
        category: str,
        model_name: ModelName = "ensemble_mean",
        dino_img_size: int = IMG_SIZE_DEFAULT,
        dino_epochs: int = 40,
        pc_img_size: int = PC_IMG_SIZE_DEFAULT,
        pc_coreset_ratio: float = 0.1,
        n_calib_for_stats: int = 20,
    ):
        self.category = category
        self.model_name = model_name
        self.dino_img_size = dino_img_size
        self.pc_img_size = pc_img_size

        self._dino = None
        self._patchcore = None
        # Stats de référence pour normalisation discriminative (option A)
        self.dino_min: float | None = None
        self.dino_max: float | None = None
        self.pc_min: float | None = None
        self.pc_max: float | None = None

        # Localiser les paths train good
        train_dir = self._train_dir()
        if not train_dir.exists():
            raise FileNotFoundError(
                f"Dossier train absent : {train_dir}\n"
                f"Run scripts/download_data.py au préalable."
            )
        train_paths = sorted(str(p) for p in train_dir.glob("*.png"))
        if not train_paths:
            train_paths = sorted(str(p) for p in train_dir.glob("*.jpg"))
        calib_paths = train_paths[:n_calib_for_stats]

        # Chargement Dinomaly si nécessaire
        if model_name in ("dinomaly", "ensemble_mean", "ensemble_max"):
            self._dino = load_dinomaly_ckpt(
                category=category, img_size=dino_img_size, epochs=dino_epochs
            )
            # Stats de référence : score sur n_calib train good → min/max
            h_calib_dino = score_paths_manual(
                self._dino, calib_paths, img_size=dino_img_size, batch_size=4
            )
            # Max par image puis stats sur les max → reflète la distribution des scores image
            per_image_max = h_calib_dino.reshape(h_calib_dino.shape[0], -1).max(axis=1)
            self.dino_min = float(per_image_max.min())
            self.dino_max = float(per_image_max.max())

        # Construction PatchCore si nécessaire
        if model_name in ("patchcore", "ensemble_mean", "ensemble_max"):
            self._patchcore = PatchCoreManual(
                img_size=pc_img_size, coreset_ratio=pc_coreset_ratio
            )
            self._patchcore.fit(train_paths)
            h_calib_pc = self._patchcore.score(calib_paths)
            per_image_max_pc = h_calib_pc.reshape(h_calib_pc.shape[0], -1).max(axis=1)
            self.pc_min = float(per_image_max_pc.min())
            self.pc_max = float(per_image_max_pc.max())

    def _train_dir(self) -> Path:
        """Localise le dossier train/good pour la catégorie (MVTec ou HSS-IAD)."""
        mvtec_path = PATHS.mvtec_dir / self.category / "train" / "good"
        if mvtec_path.exists():
            return mvtec_path
        # Fallback HSS-IAD
        return PATHS.hssiad_dir / f"Casting_class{self.category[-1]}" / "train" / "good"

    @classmethod
    def from_category(
        cls, category: str, model_name: ModelName = "ensemble_mean", **kwargs
    ) -> "AnomalyPipeline":
        """Factory raccourci."""
        return cls(category=category, model_name=model_name, **kwargs)

    def predict(self, image: Image.Image | str | Path) -> PredictionResult:
        """Inference sur une PIL Image ou un chemin. Retourne PredictionResult."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            raise TypeError(f"Type non supporté : {type(image)}")

        h_dino = None
        h_pc = None
        if self._dino is not None:
            h_dino = score_image_manual(self._dino, image, img_size=self.dino_img_size)
        if self._patchcore is not None:
            h_pc = self._patchcore.score_image(image)

        # --- Normalisation contre les stats de référence (train good baseline) ---
        # Score image = (max_heatmap - train_max) / (train_max - train_min)
        # → 0 si l'image ressemble au train good le plus anomal vu
        # → > 0 si l'image est plus anomale que le pire train good
        # Clip à [0, 1] pour usage UI (au-delà de 1 = très anomal mais cap visuel)
        def normed_score(h_raw, mn, mx):
            if h_raw is None or mn is None or mx is None:
                return None, None
            raw_max = float(h_raw.max())
            score = (raw_max - mx) / (mx - mn + 1e-8)
            # Heatmap normalisée pour affichage (référence stats train good)
            h_norm = (h_raw - mn) / (mx - mn + 1e-8)
            return score, h_norm

        score_dino, h_dino_n = normed_score(h_dino, self.dino_min, self.dino_max)
        score_pc, h_pc_n = normed_score(h_pc, self.pc_min, self.pc_max)

        # --- Sélection / combinaison ---
        if self.model_name == "dinomaly":
            heatmap = h_dino
            heatmap_norm = h_dino_n
            score = score_dino
        elif self.model_name == "patchcore":
            heatmap = h_pc
            heatmap_norm = h_pc_n
            score = score_pc
        else:
            # Ensemble : normaliser chaque modèle avec ses stats, PUIS combiner
            if h_pc_n.shape != h_dino_n.shape:
                h_pc_n = sk_resize(
                    h_pc_n, h_dino_n.shape, order=1, preserve_range=True, anti_aliasing=True
                ).astype(np.float32)
            if self.model_name == "ensemble_mean":
                heatmap_norm = (h_dino_n + h_pc_n) / 2
            else:  # ensemble_max
                heatmap_norm = np.maximum(h_dino_n, h_pc_n)
            heatmap = heatmap_norm  # raw n'a plus de sens en ensemble
            # Score = max de la heatmap combinée (déjà dans une échelle pseudo-[0,1])
            score = float(heatmap_norm.max())

        # Clip pour usage UI [0, 1]
        score_clipped = float(np.clip(score, 0.0, 1.0))
        # heatmap_norm peut dépasser 1 si défaut très net → clip pour le colormap jet
        heatmap_norm_display = np.clip(heatmap_norm, 0.0, 1.0).astype(np.float32)

        target_shape = heatmap_norm.shape
        img_resized = np.array(image.resize(target_shape[::-1], Image.BILINEAR))
        overlay = _to_jet_overlay(heatmap_norm_display)

        return PredictionResult(
            score=score_clipped,
            heatmap=(heatmap if heatmap is not None else heatmap_norm).astype(np.float32),
            heatmap_norm=heatmap_norm_display,
            overlay=overlay,
            image_resized=img_resized,
            model_name=self.model_name,
            category=self.category,
        )
