"""Direct Casting_class1 inference pipeline: ROI segmentation then RD/Feature-AE."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.config import PATHS
from src.features.tiling import reconstruct_score_map, tile_image
from src.features.transforms import crop_box_to_mask
from src.models.baselines.patchcore import resolve_device
from src.models.feature_ae.calibration import smooth_maps, topk_score
from src.models.feature_ae.evaluation import (
    batch_score_maps,
    crop_centered_context_tile,
    resize_score_map,
    resize_tile_for_model,
)
from src.models.feature_ae.models import ResNetTeacherFeatures, build_feature_autoencoder
from src.models.pixel_ae.runtime import build_tile_transform
from src.models.segmentation.models import build_segmentation_model
from src.models.segmentation.prediction import image_tensor, resize_probabilities
from src.models.segmentation.runtime import replace_segmentation_head


DEFAULT_SEGMENTER_CHECKPOINT = (
    PATHS.root
    / "models"
    / "functional_surface"
    / "Casting_class1"
    / "Casting_class1_v36_direct_imagenet_512_1024_synthv4_photometric_c123"
    / "checkpoint_epoch_028.pt"
)
DEFAULT_FEATURE_AE_CHECKPOINT = (
    PATHS.root
    / "models"
    / "feature_ae"
    / "Casting_class1"
    / "Casting_class1_reverse_distill_ms_dualcontext_v36roi_e28_layer23_metricbest_v1"
    / "checkpoint_epoch_002.pt"
)


@dataclass
class CastingFeatureAEResult:
    score: float
    raw_score_map: np.ndarray
    calibrated_score_map: np.ndarray
    display_map: np.ndarray
    surface_prob: np.ndarray
    surface_mask: np.ndarray
    semantic_mask: np.ndarray
    image_rgb: np.ndarray
    roi_overlay: np.ndarray
    heatmap_overlay: np.ndarray
    model_summary: dict


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PATHS.root / path


def _percentile_display_map(
    score_map: np.ndarray,
    roi_mask: np.ndarray,
    *,
    low_percentile: float,
    high_percentile: float,
    gamma: float,
    display_threshold: float,
) -> np.ndarray:
    score = np.asarray(score_map, dtype=np.float32)
    roi = np.asarray(roi_mask) > 0
    values = score[roi] if np.any(roi) else score.reshape(-1)
    low = float(np.percentile(values, low_percentile)) if low_percentile > 0 else 0.0
    high = float(np.percentile(values, high_percentile))
    if high <= low:
        high = low + 1e-6
    norm = np.clip((score - low) / (high - low), 0.0, 1.0)
    if gamma != 1.0:
        norm = np.power(norm, float(gamma))
    norm[~roi] = 0.0
    norm[norm < float(display_threshold)] = 0.0
    return norm.astype(np.float32, copy=False)


def _orange_overlay(image: Image.Image, display_map: np.ndarray, *, alpha: float) -> np.ndarray:
    display = np.asarray(display_map, dtype=np.float32)
    if display.shape != (image.height, image.width):
        display_img = Image.fromarray((np.clip(display, 0, 1) * 255).astype(np.uint8), mode="L")
        display = np.asarray(display_img.resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    heat = np.zeros_like(base, dtype=np.float32)
    heat[..., 0] = 255.0
    heat[..., 1] = np.clip(np.sqrt(display) * 199.0, 0.0, 199.0)
    heat[..., 2] = 5.0
    blend_alpha = float(alpha) * display[..., None]
    out = (1.0 - blend_alpha) * base + blend_alpha * heat
    return np.clip(out, 0, 255).astype(np.uint8)


def _roi_overlay(image: Image.Image, surface_prob: np.ndarray, semantic_mask: np.ndarray) -> np.ndarray:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    prob = np.asarray(surface_prob, dtype=np.float32)
    if prob.shape != (image.height, image.width):
        prob_img = Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8), mode="L")
        prob = np.asarray(prob_img.resize(image.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    semantic = np.asarray(semantic_mask)
    if semantic.shape != (image.height, image.width):
        semantic_img = Image.fromarray(semantic.astype(np.uint8), mode="L")
        semantic = np.asarray(semantic_img.resize(image.size, Image.Resampling.NEAREST))
    color = np.zeros_like(base)
    color[..., 0] = 40.0
    color[..., 1] = 140.0
    color[..., 2] = 230.0
    landmark = semantic == 2
    color[landmark] = np.array([255.0, 128.0, 20.0], dtype=np.float32)
    mask = (prob > 0.08) | landmark
    alpha = (0.48 * np.clip(prob, 0.0, 1.0))[..., None]
    alpha[landmark] = 0.55
    alpha[~mask] = 0.0
    out = (1.0 - alpha) * base + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


class CastingFeatureAEPipeline:
    """Single-image ROI + RD/Feature-AE inference used by Streamlit."""

    def __init__(
        self,
        *,
        segmenter_checkpoint: str | Path = DEFAULT_SEGMENTER_CHECKPOINT,
        feature_ae_checkpoint: str | Path = DEFAULT_FEATURE_AE_CHECKPOINT,
        device: str = "auto",
        seg_input_size: int = 512,
        seg_context_size: int = 1024,
        input_size: int = 384,
        tile_size: int = 384,
        context_tile_size: int = 768,
        tile_stride: int = 384,
        layer_weights: dict[str, float] | None = None,
        roi_threshold: float = 0.5,
        topk_fraction: float = 0.005,
        smoothing: str = "median3",
        display_threshold: float = 0.60,
        display_low_percentile: float = 85.0,
        display_high_percentile: float = 99.8,
        display_gamma: float = 1.4,
        overlay_alpha: float = 0.72,
        batch_size: int = 1,
    ) -> None:
        self.segmenter_checkpoint = _resolve(segmenter_checkpoint)
        self.feature_ae_checkpoint = _resolve(feature_ae_checkpoint)
        self.device = resolve_device(device)
        self.seg_input_size = int(seg_input_size)
        self.seg_context_size = int(seg_context_size)
        self.input_size = int(input_size)
        self.tile_size = int(tile_size)
        self.context_tile_size = int(context_tile_size)
        self.tile_stride = int(tile_stride)
        self.roi_threshold = float(roi_threshold)
        self.topk_fraction = float(topk_fraction)
        self.smoothing = str(smoothing)
        self.display_threshold = float(display_threshold)
        self.display_low_percentile = float(display_low_percentile)
        self.display_high_percentile = float(display_high_percentile)
        self.display_gamma = float(display_gamma)
        self.overlay_alpha = float(overlay_alpha)
        self.batch_size = int(batch_size)

        self.segmenter = self._load_segmenter()
        self.teacher, self.student, self.layers, self.cosine_weight = self._load_feature_ae()
        self.layer_weights = self._normalize_layer_weights(layer_weights)

    def _load_segmenter(self) -> torch.nn.Module:
        if not self.segmenter_checkpoint.exists():
            raise FileNotFoundError(self.segmenter_checkpoint)
        checkpoint = torch.load(self.segmenter_checkpoint, map_location="cpu", weights_only=False)
        model_type = str(checkpoint.get("model_type", "functional_unet_resnet18_det1_context2b"))
        num_classes = int(checkpoint.get("num_classes", 3))
        self.seg_context_size = int(checkpoint.get("context_size", self.seg_context_size))
        model = build_segmentation_model(model_type)
        replace_segmentation_head(model, num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(self.device).eval()

    def _load_feature_ae(self) -> tuple[torch.nn.Module, torch.nn.Module, list[str], float]:
        if not self.feature_ae_checkpoint.exists():
            raise FileNotFoundError(self.feature_ae_checkpoint)
        checkpoint = torch.load(self.feature_ae_checkpoint, map_location="cpu", weights_only=False)
        layers = list(checkpoint.get("layers") or ["layer2", "layer3"])
        teacher_backbone = str(checkpoint.get("teacher_backbone") or "resnet18")
        model_type = str(checkpoint.get("model_type") or "reverse_distill_resnet18_dual_context_gated")
        cosine_weight = float(checkpoint.get("cosine_weight", 0.5))
        teacher = ResNetTeacherFeatures(teacher_backbone, layers=layers).to(self.device).eval()
        student = build_feature_autoencoder(model_type, layers=layers)
        student.load_state_dict(checkpoint["model_state_dict"])
        student = student.to(self.device).eval()
        return teacher, student, layers, cosine_weight

    def _normalize_layer_weights(self, weights: dict[str, float] | None) -> dict[str, float]:
        raw = weights or {"layer2": 0.65, "layer3": 0.35}
        selected = {layer: float(raw.get(layer, 0.0)) for layer in self.layers}
        total = sum(selected.values())
        if total <= 0:
            return {layer: 1.0 / len(self.layers) for layer in self.layers}
        return {layer: value / total for layer, value in selected.items()}

    @torch.inference_mode()
    def _predict_roi(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tensor = image_tensor(image, self.seg_input_size)[None].to(self.device)
        global_image = image_tensor(image, self.seg_context_size)[None].to(self.device)
        crop_mask = transforms.ToTensor()(crop_box_to_mask((0.5, 0.5, 1.0, 1.0), self.seg_context_size))
        crop_mask = crop_mask[None].to(self.device)
        output = self.segmenter(tensor, global_image=global_image, crop_box_mask=crop_mask)
        logits = output["mask_logits"] if isinstance(output, dict) else output
        prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
        prob_original = resize_probabilities(prob, image.size)
        semantic = np.argmax(prob_original, axis=0).astype(np.uint8)
        surface_prob = prob_original[1].astype(np.float32, copy=False)
        surface_mask = (surface_prob >= self.roi_threshold).astype(np.uint8)
        return surface_prob, surface_mask, semantic

    @torch.inference_mode()
    def _predict_feature_maps(self, image: Image.Image) -> dict[str, np.ndarray]:
        transform = build_tile_transform("imagenet")
        tiled = tile_image(image, tile_size=self.tile_size, stride=self.tile_stride)
        layer_tile_maps: dict[str, list[np.ndarray]] = {layer: [] for layer in self.layers}
        tile_maps: list[np.ndarray] = []
        for start in range(0, len(tiled.tiles), self.batch_size):
            batch_tiles = tiled.tiles[start : start + self.batch_size]
            batch_specs = tiled.specs[start : start + self.batch_size]
            images = torch.stack(
                [transform(resize_tile_for_model(tile, self.input_size)) for tile in batch_tiles]
            ).to(self.device)
            context_images = torch.stack(
                [
                    transform(
                        resize_tile_for_model(
                            crop_centered_context_tile(image, spec, self.context_tile_size),
                            self.input_size,
                        )
                    )
                    for spec in batch_specs
                ]
            ).to(self.device)
            fused, layer_maps = batch_score_maps(
                student=self.student,
                teacher=self.teacher,
                images=images,
                context_images=context_images,
                layers=self.layers,
                input_size=self.input_size,
                cosine_weight=self.cosine_weight,
                layer_weights=self.layer_weights,
            )
            tile_maps.extend(
                resize_score_map(item, self.tile_size)
                for item in fused.cpu().numpy().astype(np.float32, copy=False)
            )
            for layer_name, layer_map in layer_maps.items():
                layer_tile_maps[layer_name].extend(
                    resize_score_map(item, self.tile_size)
                    for item in layer_map.cpu().numpy().astype(np.float32, copy=False)
                )
        maps = {
            layer: reconstruct_score_map(layer_tile_maps[layer], tiled, aggregation="mean")
            for layer in self.layers
        }
        maps["_normalized_fused"] = reconstruct_score_map(tile_maps, tiled, aggregation="mean")
        return maps

    def predict(self, image: Image.Image | str | Path) -> CastingFeatureAEResult:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        surface_prob, surface_mask, semantic = self._predict_roi(image)
        layer_maps = self._predict_feature_maps(image)
        raw_score = sum(layer_maps[layer] * self.layer_weights[layer] for layer in self.layers)
        smoothed = smooth_maps(raw_score[None, ...], self.smoothing, batch_size=1)[0]
        score_map = smoothed * surface_prob
        score = topk_score(score_map, fraction=self.topk_fraction, valid_mask=surface_mask)
        roi_overlay = _roi_overlay(image, surface_prob, semantic)
        result = CastingFeatureAEResult(
            score=float(score),
            raw_score_map=raw_score.astype(np.float32, copy=False),
            calibrated_score_map=score_map.astype(np.float32, copy=False),
            display_map=np.zeros_like(score_map, dtype=np.float32),
            surface_prob=surface_prob,
            surface_mask=surface_mask,
            semantic_mask=semantic,
            image_rgb=np.asarray(image, dtype=np.uint8),
            roi_overlay=roi_overlay,
            heatmap_overlay=np.asarray(image, dtype=np.uint8),
            model_summary={
                "segmenter": str(self.segmenter_checkpoint),
                "feature_ae": str(self.feature_ae_checkpoint),
                "layers": self.layers,
                "layer_weights": self.layer_weights,
                "roi_threshold": self.roi_threshold,
                "topk_fraction": self.topk_fraction,
                "display_threshold": self.display_threshold,
            },
        )
        return self.render(
            result,
            display_threshold=self.display_threshold,
            display_low_percentile=self.display_low_percentile,
            display_high_percentile=self.display_high_percentile,
            display_gamma=self.display_gamma,
            overlay_alpha=self.overlay_alpha,
            roi_threshold=self.roi_threshold,
            topk_fraction=self.topk_fraction,
        )

    def render(
        self,
        result: CastingFeatureAEResult,
        *,
        display_threshold: float,
        display_low_percentile: float,
        display_high_percentile: float,
        display_gamma: float,
        overlay_alpha: float,
        roi_threshold: float,
        topk_fraction: float,
    ) -> CastingFeatureAEResult:
        """Re-render a cached inference result without re-running the models."""

        image = Image.fromarray(result.image_rgb, mode="RGB")
        surface_mask = (np.asarray(result.surface_prob) >= float(roi_threshold)).astype(np.uint8)
        score = topk_score(
            result.calibrated_score_map,
            fraction=float(topk_fraction),
            valid_mask=surface_mask,
        )
        display_map = _percentile_display_map(
            result.calibrated_score_map,
            surface_mask,
            low_percentile=float(display_low_percentile),
            high_percentile=float(display_high_percentile),
            gamma=float(display_gamma),
            display_threshold=float(display_threshold),
        )
        heatmap_overlay = _orange_overlay(image, display_map, alpha=float(overlay_alpha))
        return replace(
            result,
            score=float(score),
            display_map=display_map,
            surface_mask=surface_mask,
            heatmap_overlay=heatmap_overlay,
            model_summary={
                **result.model_summary,
                "roi_threshold": float(roi_threshold),
                "topk_fraction": float(topk_fraction),
                "display_threshold": float(display_threshold),
            },
        )
