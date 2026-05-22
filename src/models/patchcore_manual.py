"""PatchCore from-scratch — WideResNet50_2 + memory bank + coreset.

Extrait des notebooks 07/08. Indépendant d'anomalib (anomalib 2.3.1 a un
RecursionError dans son implémentation PatchCore).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE_DEFAULT = 256
CORESET_RATIO_DEFAULT = 0.1
SMOOTH_SIGMA_DEFAULT = 4.0
BATCH_DEFAULT = 8


def _build_backbone():
    bb = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    bb.eval()
    for p in bb.parameters():
        p.requires_grad = False
    bb = bb.to(DEVICE)
    feats = {}

    def _hook(name):
        def fn(_m, _i, out):
            feats[name] = out

        return fn

    bb.layer2.register_forward_hook(_hook("layer2"))
    bb.layer3.register_forward_hook(_hook("layer3"))
    return bb, feats


def _img_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _extract_features(bb, feats_cache, batch_tensor):
    feats_cache.clear()
    with torch.no_grad():
        _ = bb(batch_tensor.to(DEVICE))
    f2, f3 = feats_cache["layer2"], feats_cache["layer3"]
    f2 = F.avg_pool2d(f2, kernel_size=3, stride=1, padding=1)
    f3 = F.avg_pool2d(f3, kernel_size=3, stride=1, padding=1)
    f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([f2, f3], dim=1)


class PatchCoreManual:
    """Implémentation manuelle de PatchCore.

    Usage :
        pc = PatchCoreManual(img_size=256, coreset_ratio=0.1)
        pc.fit(train_paths)
        heatmaps = pc.score(test_paths)
    """

    def __init__(
        self,
        img_size: int = IMG_SIZE_DEFAULT,
        coreset_ratio: float = CORESET_RATIO_DEFAULT,
        smooth_sigma: float = SMOOTH_SIGMA_DEFAULT,
        batch_size: int = BATCH_DEFAULT,
        seed: int = 42,
    ):
        self.img_size = img_size
        self.coreset_ratio = coreset_ratio
        self.smooth_sigma = smooth_sigma
        self.batch_size = batch_size
        self.seed = seed
        self.coreset: torch.Tensor | None = None
        self._bb = None
        self._feats = None
        self._tf = _img_transform(img_size)

    def _load_backbone(self):
        if self._bb is None:
            self._bb, self._feats = _build_backbone()

    def _load_img(self, p):
        return self._tf(Image.open(p).convert("RGB"))

    def fit(self, train_paths: Iterable[str | Path]) -> "PatchCoreManual":
        """Construit le memory bank + applique le coreset random."""
        self._load_backbone()
        train_paths = list(train_paths)
        all_patches = []
        for i in range(0, len(train_paths), self.batch_size):
            chunk = train_paths[i : i + self.batch_size]
            imgs = torch.stack([self._load_img(p) for p in chunk])
            feat = _extract_features(self._bb, self._feats, imgs)
            B, D, H, W = feat.shape
            all_patches.append(feat.permute(0, 2, 3, 1).reshape(-1, D).cpu())
        memory_bank = torch.cat(all_patches, dim=0)

        torch.manual_seed(self.seed)
        n_coreset = max(1, int(len(memory_bank) * self.coreset_ratio))
        idx = torch.randperm(len(memory_bank))[:n_coreset]
        self.coreset = memory_bank[idx].to(DEVICE)
        return self

    @torch.no_grad()
    def score(self, paths: Iterable[str | Path]) -> np.ndarray:
        """Score une liste de chemins. Retourne (N, img_size, img_size)."""
        if self.coreset is None:
            raise RuntimeError("Memory bank non construit — appelle fit() d'abord.")
        self._load_backbone()
        paths = list(paths)
        heatmaps = []
        for i in range(0, len(paths), self.batch_size):
            chunk = paths[i : i + self.batch_size]
            imgs = torch.stack([self._load_img(p) for p in chunk])
            feat = _extract_features(self._bb, self._feats, imgs)
            B, D, Hf, Wf = feat.shape
            patches = feat.permute(0, 2, 3, 1).reshape(-1, D)
            dists = torch.cdist(patches, self.coreset)
            nn_dist, _ = dists.min(dim=1)
            nn_dist = nn_dist.reshape(B, Hf, Wf).cpu().numpy()
            for j in range(B):
                h_im = Image.fromarray(nn_dist[j].astype(np.float32))
                h = np.array(
                    h_im.resize((self.img_size, self.img_size), Image.BILINEAR)
                )
                h = gaussian_filter(h, sigma=self.smooth_sigma)
                heatmaps.append(h)
        return np.stack(heatmaps)

    @torch.no_grad()
    def score_image(self, image: Image.Image) -> np.ndarray:
        """Score une PIL Image unique. Retourne (img_size, img_size)."""
        if self.coreset is None:
            raise RuntimeError("Memory bank non construit — appelle fit() d'abord.")
        self._load_backbone()
        img = self._tf(image.convert("RGB")).unsqueeze(0)
        feat = _extract_features(self._bb, self._feats, img)
        _, D, Hf, Wf = feat.shape
        patches = feat.permute(0, 2, 3, 1).reshape(-1, D)
        dists = torch.cdist(patches, self.coreset)
        nn_dist, _ = dists.min(dim=1)
        nn_dist = nn_dist.reshape(Hf, Wf).cpu().numpy()
        h_im = Image.fromarray(nn_dist.astype(np.float32))
        h = np.array(h_im.resize((self.img_size, self.img_size), Image.BILINEAR))
        return gaussian_filter(h, sigma=self.smooth_sigma)

    def save_coreset(self, path: str | Path) -> None:
        """Sauvegarde le coreset (tensor) pour réutilisation."""
        if self.coreset is None:
            raise RuntimeError("Rien à sauvegarder — coreset non construit.")
        torch.save(self.coreset.cpu(), path)

    def load_coreset(self, path: str | Path) -> "PatchCoreManual":
        """Charge un coreset pré-construit."""
        self._load_backbone()
        self.coreset = torch.load(path, map_location=DEVICE, weights_only=True)
        return self
