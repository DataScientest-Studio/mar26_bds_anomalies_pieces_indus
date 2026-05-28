"""Pipeline d'infÃ©rence haut-niveau pour Streamlit / scripts.

Charge les modÃ¨les (cache via attribut), fait la prÃ©diction sur une image,
retourne heatmap + score + overlay prÃªt Ã  afficher.

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
from src.models.dinomaly.wrapper import (
    IMG_SIZE_DEFAULT,
    load_dinomaly_ckpt,
    score_image_manual,
    score_paths_manual,
)
from src.models.ensemble.heatmap_fusion import (
    ensemble_max,
    ensemble_mean,
    norm_global_minmax,
)
from src.models.feature_ae.calibration import smooth_maps
from src.models.patchcore.manual import IMG_SIZE_DEFAULT as PC_IMG_SIZE_DEFAULT
from src.models.patchcore.manual import PatchCoreManual

# Agrégation pixel→image : max après lissage médian 3×3.
# Le médian 3×3 supprime les peaks pixels isolés (bruit), le max préserve
# les peaks régionaux même très localisés (petits défauts < 50 pixels).
SCORE_SMOOTHING = "median3"


def _aggregate_score(heatmap: np.ndarray) -> float:
    """Max après lissage médian 3×3 — sensible aux petits défauts, robuste au bruit pixel."""
    smoothed = smooth_maps(heatmap[None, ...], SCORE_SMOOTHING, batch_size=1)[0]
    return float(smoothed.max())

ModelName = Literal["dinomaly", "patchcore", "ensemble_mean", "ensemble_max"]


@dataclass
class PredictionResult:
    """RÃ©sultat d'une infÃ©rence sur une image."""

    score: float  # score image-level (max de la heatmap normalisÃ©e)
    heatmap: np.ndarray  # (H, W) float32 â€” heatmap brute du modÃ¨le choisi
    heatmap_norm: np.ndarray  # (H, W) float32 â€” heatmap normalisÃ©e global-minmax [0,1]
    overlay: np.ndarray  # (H, W, 4) uint8 â€” heatmap colormap jet + alpha, prÃªte Ã  display
    image_resized: np.ndarray  # (H, W, 3) uint8 â€” image originale resize Ã  la heatmap shape
    model_name: str
    category: str


def _to_jet_overlay(heatmap_norm: np.ndarray) -> np.ndarray:
    """Convertit une heatmap normalisÃ©e [0,1] en RGBA uint8 colormap jet."""
    import matplotlib.cm as cm

    cmap = cm.get_cmap("jet")
    rgba = (cmap(np.clip(heatmap_norm, 0, 1)) * 255).astype(np.uint8)
    return rgba


class AnomalyPipeline:
    """Pipeline unifiÃ© Dinomaly / PatchCore / Ensemble pour une catÃ©gorie donnÃ©e.

    Construction lente (chargement ckpt + memory bank), prÃ©diction rapide.
    Cache Ã  utiliser via Streamlit `@st.cache_resource`.
    """

    def __init__(
        self,
        category: str,
        model_name: ModelName = "ensemble_mean",
        dino_img_size: int = IMG_SIZE_DEFAULT,
        dino_epochs: int = 40,
        pc_img_size: int = PC_IMG_SIZE_DEFAULT,
        pc_coreset_ratio: float = 0.1,
        n_calib_for_stats: int = 100,
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
                f"Run scripts/download_data.py au prÃ©alable."
            )
        train_paths = sorted(str(p) for p in train_dir.glob("*.png"))
        if not train_paths:
            train_paths = sorted(str(p) for p in train_dir.glob("*.jpg"))
        calib_paths = train_paths[:n_calib_for_stats]

        # Chargement Dinomaly si nÃ©cessaire
        if model_name in ("dinomaly", "ensemble_mean", "ensemble_max"):
            self._dino = load_dinomaly_ckpt(
                category=category, img_size=dino_img_size, epochs=dino_epochs
            )
            # Stats de référence : score sur n_calib train good → min/max
            # Utilise l'agrégation top-k smoothée (idem prédiction) pour rester cohérent.
            h_calib_dino = score_paths_manual(
                self._dino, calib_paths, img_size=dino_img_size, batch_size=4
            )
            per_image_dino = np.array([_aggregate_score(m) for m in h_calib_dino])
            self.dino_min = float(per_image_dino.min())
            self.dino_max = float(per_image_dino.max())

        # Construction PatchCore si nÃ©cessaire
        if model_name in ("patchcore", "ensemble_mean", "ensemble_max"):
            self._patchcore = PatchCoreManual(
                img_size=pc_img_size, coreset_ratio=pc_coreset_ratio
            )
            self._patchcore.fit(train_paths)
            h_calib_pc = self._patchcore.score(calib_paths)
            per_image_pc = np.array([_aggregate_score(m) for m in h_calib_pc])
            self.pc_min = float(per_image_pc.min())
            self.pc_max = float(per_image_pc.max())

    def _train_dir(self) -> Path:
        """Localise le dossier train/good pour la catÃ©gorie (MVTec ou HSS-IAD)."""
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
            raise TypeError(f"Type non supportÃ© : {type(image)}")

        h_dino = None
        h_pc = None
        if self._dino is not None:
            h_dino = score_image_manual(self._dino, image, img_size=self.dino_img_size)
        if self._patchcore is not None:
            h_pc = self._patchcore.score_image(image)

        # --- Normalisation contre les stats de rÃ©fÃ©rence (train good baseline) ---
        # Score image = (max_heatmap - train_max) / (train_max - train_min)
        # â†’ 0 si l'image ressemble au train good le plus anomal vu
        # â†’ > 0 si l'image est plus anomale que le pire train good
        # Clip Ã  [0, 1] pour usage UI (au-delÃ  de 1 = trÃ¨s anomal mais cap visuel)
        def normed_score(h_raw, mn, mx):
            if h_raw is None or mn is None or mx is None:
                return None, None
            agg = _aggregate_score(h_raw)
            score = (agg - mx) / (mx - mn + 1e-8)
            # Heatmap normalisée pour affichage (référence stats train good)
            h_norm = (h_raw - mn) / (mx - mn + 1e-8)
            return score, h_norm

        score_dino, h_dino_n = normed_score(h_dino, self.dino_min, self.dino_max)
        score_pc, h_pc_n = normed_score(h_pc, self.pc_min, self.pc_max)

        # --- SÃ©lection / combinaison ---
        if self.model_name == "dinomaly":
            heatmap = h_dino
            heatmap_norm = h_dino_n
            score = score_dino
        elif self.model_name == "patchcore":
            heatmap = h_pc
            heatmap_norm = h_pc_n
            score = score_pc
        else:
            # Ensemble : normaliser chaque modÃ¨le avec ses stats, PUIS combiner
            if h_pc_n.shape != h_dino_n.shape:
                h_pc_n = sk_resize(
                    h_pc_n, h_dino_n.shape, order=1, preserve_range=True, anti_aliasing=True
                ).astype(np.float32)
            if self.model_name == "ensemble_mean":
                heatmap_norm = (h_dino_n + h_pc_n) / 2
                score = (score_dino + score_pc) / 2
            else:  # ensemble_max
                heatmap_norm = np.maximum(h_dino_n, h_pc_n)
                score = max(score_dino, score_pc)
            heatmap = heatmap_norm  # heatmap pixel-wise = affichage uniquement

        # Clip pour usage UI [0, 1]
        score_clipped = float(np.clip(score, 0.0, 1.0))
        # heatmap_norm peut dÃ©passer 1 si dÃ©faut trÃ¨s net â†’ clip pour le colormap jet
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

    def _build_result(
        self,
        image: Image.Image,
        heatmap: np.ndarray,
        heatmap_norm: np.ndarray,
        score: float,
        model_name: str,
    ) -> PredictionResult:
        """Construit un PredictionResult à partir d'une heatmap brute et normalisée."""
        score_clipped = float(np.clip(score, 0.0, 1.0))
        heatmap_norm_display = np.clip(heatmap_norm, 0.0, 1.0).astype(np.float32)
        target_shape = heatmap_norm.shape
        img_resized = np.array(image.resize(target_shape[::-1], Image.BILINEAR))
        overlay = _to_jet_overlay(heatmap_norm_display)
        return PredictionResult(
            score=score_clipped,
            heatmap=heatmap.astype(np.float32),
            heatmap_norm=heatmap_norm_display,
            overlay=overlay,
            image_resized=img_resized,
            model_name=model_name,
            category=self.category,
        )

    def predict_all_modes(self, image: Image.Image | str | Path) -> dict[str, PredictionResult]:
        """Inférence des 4 modes en réutilisant une SEULE passe par modèle.

        À appeler sur un pipeline construit avec `model_name="ensemble_mean"` (ou
        "ensemble_max") afin que Dinomaly ET PatchCore soient chargés et calibrés.

        ~3× plus rapide que 4 appels successifs à `predict()` en mode Compare.
        """
        if self._dino is None or self._patchcore is None:
            raise RuntimeError(
                "predict_all_modes nécessite Dinomaly ET PatchCore chargés. "
                "Instancier le pipeline avec model_name='ensemble_mean'."
            )

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        # 1 forward pass par modèle (au lieu de 3 dans l'ancien code).
        h_dino = score_image_manual(self._dino, image, img_size=self.dino_img_size)
        h_pc = self._patchcore.score_image(image)

        # Normalisation contre stats train good (idem predict()).
        agg_dino = _aggregate_score(h_dino)
        score_dino = (agg_dino - self.dino_max) / (self.dino_max - self.dino_min + 1e-8)
        h_dino_n = (h_dino - self.dino_min) / (self.dino_max - self.dino_min + 1e-8)

        agg_pc = _aggregate_score(h_pc)
        score_pc = (agg_pc - self.pc_max) / (self.pc_max - self.pc_min + 1e-8)
        h_pc_n = (h_pc - self.pc_min) / (self.pc_max - self.pc_min + 1e-8)

        # Pour les ensembles : aligner les shapes sans muter h_pc_n (réutilisé pour PC seul).
        if h_pc_n.shape != h_dino_n.shape:
            h_pc_n_aligned = sk_resize(
                h_pc_n, h_dino_n.shape, order=1, preserve_range=True, anti_aliasing=True
            ).astype(np.float32)
        else:
            h_pc_n_aligned = h_pc_n

        # Heatmaps pixel-wise = affichage uniquement.
        # Les SCORES ensemble sont combinés au niveau image (cohérent avec single mode).
        h_mean = (h_dino_n + h_pc_n_aligned) / 2
        h_max = np.maximum(h_dino_n, h_pc_n_aligned)
        score_mean = (score_dino + score_pc) / 2
        score_max = max(score_dino, score_pc)

        return {
            "dinomaly": self._build_result(image, h_dino, h_dino_n, score_dino, "dinomaly"),
            "patchcore": self._build_result(image, h_pc, h_pc_n, score_pc, "patchcore"),
            "ensemble_mean": self._build_result(image, h_mean, h_mean, score_mean, "ensemble_mean"),
            "ensemble_max": self._build_result(image, h_max, h_max, score_max, "ensemble_max"),
        }
