"""Wrapper Dinomaly : build_model + load_ckpt + manual scoring sans Engine anomalib.

Extrait des notebooks 04/08/11 — utilisable depuis Streamlit ou un script.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from anomalib.models import Dinomaly
from anomalib.pre_processing import PreProcessor
from torchvision.transforms import v2 as T

from src.config import PATHS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_DEFAULT = "dinov2reg_vit_base_14"
IMG_SIZE_DEFAULT = 504
EPOCHS_DEFAULT = 40


def build_dinomaly(
    img_size: int = IMG_SIZE_DEFAULT,
    encoder: str = ENCODER_DEFAULT,
    bottleneck_dropout: float = 0.2,
    decoder_depth: int = 8,
) -> Dinomaly:
    """Instancie un modèle Dinomaly avec PreProcessor custom (Resize+Normalize ImageNet)."""
    pp = PreProcessor(
        transform=T.Compose(
            [
                T.Resize(
                    (img_size, img_size),
                    interpolation=T.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    )
    return Dinomaly(
        encoder_name=encoder,
        bottleneck_dropout=bottleneck_dropout,
        decoder_depth=decoder_depth,
        pre_processor=pp,
    )


def ckpt_path(
    category: str,
    img_size: int = IMG_SIZE_DEFAULT,
    epochs: int = EPOCHS_DEFAULT,
    suffix: str = "",
) -> Path:
    """Chemin standardisé du checkpoint Dinomaly pour une catégorie."""
    name = f"dinomaly_{category}_v2_{img_size}_{epochs}ep{suffix}.ckpt"
    return (PATHS.root / "models" / name).resolve()


def load_dinomaly_ckpt(
    category: str,
    img_size: int = IMG_SIZE_DEFAULT,
    epochs: int = EPOCHS_DEFAULT,
    suffix: str = "",
) -> Dinomaly:
    """Charge un checkpoint existant. Raise FileNotFoundError si absent."""
    path = ckpt_path(category, img_size, epochs, suffix)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint absent : {path.name}\n"
            f"Run scripts/download_models.py ou entraîne le modèle au préalable."
        )
    model = build_dinomaly(img_size=img_size)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


@torch.no_grad()
def score_paths_manual(
    dino_module: Dinomaly,
    paths: Iterable[str | Path],
    img_size: int = IMG_SIZE_DEFAULT,
    batch_size: int = 4,
) -> np.ndarray:
    """Score une liste de chemins via dino_module.model (bypass Engine).

    Retourne un array (N, H, W) de heatmaps.
    """
    dino_module = dino_module.to(DEVICE).eval()
    pp = transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    paths = list(paths)
    heatmaps = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        imgs = torch.stack([pp(Image.open(p).convert("RGB")) for p in chunk]).to(DEVICE)
        out = dino_module.model(imgs)
        if hasattr(out, "anomaly_map"):
            h = out.anomaly_map
        elif isinstance(out, dict):
            h = out.get("anomaly_map", out.get("out_map"))
        elif isinstance(out, (tuple, list)):
            h = out[-1]
        else:
            h = out
        h = h.detach().cpu().numpy()
        if h.ndim == 4:
            h = h.squeeze(1)
        heatmaps.append(h)
    return np.concatenate(heatmaps)


@torch.no_grad()
def score_image_manual(
    dino_module: Dinomaly,
    image: Image.Image,
    img_size: int = IMG_SIZE_DEFAULT,
) -> np.ndarray:
    """Score une PIL Image unique. Retourne (H, W)."""
    dino_module = dino_module.to(DEVICE).eval()
    pp = transforms.Compose(
        [
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    x = pp(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    out = dino_module.model(x)
    if hasattr(out, "anomaly_map"):
        h = out.anomaly_map
    elif isinstance(out, dict):
        h = out.get("anomaly_map", out.get("out_map"))
    elif isinstance(out, (tuple, list)):
        h = out[-1]
    else:
        h = out
    h = h.detach().cpu().numpy()[0]
    if h.ndim == 3:
        h = h.squeeze(0)
    return h
