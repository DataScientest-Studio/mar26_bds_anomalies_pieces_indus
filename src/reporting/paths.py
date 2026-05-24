"""Canonical paths and run identifiers for experiment artifacts."""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import PATHS


@dataclass(frozen=True)
class ReportPaths:
    root: Path = PATHS.root / "reports"
    tables_raw: Path = PATHS.root / "reports" / "tables" / "raw"
    tables_summary: Path = PATHS.root / "reports" / "tables" / "summary"
    docs: Path = PATHS.root / "reports" / "docs"
    exports: Path = PATHS.root / "reports" / "exports"
    figures: Path = PATHS.root / "reports" / "figures"
    manifest: Path = PATHS.root / "reports" / "manifest.csv"

    patchcore_matrix_latest: Path = (
        PATHS.root / "reports" / "tables" / "raw" / "patchcore_matrix_all_latest.csv"
    )
    reconstruction_checkpoint_matrix: Path = (
        PATHS.root / "reports" / "tables" / "raw" / "reconstruction_matrix_checkpoints.csv"
    )


@dataclass(frozen=True)
class ModelArtifactPaths:
    root: Path = PATHS.root / "models"
    patchcore: Path = PATHS.root / "models" / "patchcore"
    reconstruction: Path = PATHS.root / "models" / "reconstruction"
    custom_autoencoder: Path = (
        PATHS.root / "models" / "reconstruction" / "custom_autoencoder"
    )
    resnet_finetuned: Path = PATHS.root / "models" / "reconstruction" / "resnet_finetuned"


REPORTS = ReportPaths()
MODEL_ARTIFACTS = ModelArtifactPaths()


def safe_tag(value: object) -> str:
    """Return a stable filesystem tag for compact experiment identifiers."""
    text = str(value).strip().replace("\\", "/")
    text = text.replace("+", "-").replace(",", "-").replace("/", "-")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-_.")
    return text or "unknown"


def compact_run_id(run_id: str, max_length: int = 150) -> str:
    """Keep run folders below Windows path limits while preserving readability."""
    if len(run_id) <= max_length:
        return run_id
    digest = hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:10]
    return f"{run_id[: max_length - 13].rstrip('-_.')}_h{digest}"


def lr_tag(value: object) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return safe_tag(value)


def normalize_scope(scope: str | None) -> str:
    if scope in {"global", "_global"}:
        return "global"
    if scope in {"globalft", "global_finetune"}:
        return "globalft"
    return "category"


def pixel_ae_root(model_type: str) -> Path:
    if model_type in {
        "custom_autoencoder",
        "custom_unet_autoencoder",
        "custom_constrained_unet_autoencoder",
        "custom_mask_conditioned_inpainting_autoencoder",
    }:
        return MODEL_ARTIFACTS.custom_autoencoder
    if model_type == "resnet_reconstruction_finetuned":
        return MODEL_ARTIFACTS.resnet_finetuned
    raise ValueError(f"Unknown pixel AE model type: {model_type}")


def pixel_ae_model_family(model_type: str) -> str:
    if model_type in {
        "custom_autoencoder",
        "custom_unet_autoencoder",
        "custom_constrained_unet_autoencoder",
        "custom_mask_conditioned_inpainting_autoencoder",
    }:
        return "custom_autoencoder"
    if model_type == "resnet_reconstruction_finetuned":
        return "resnet_finetuned"
    raise ValueError(f"Unknown pixel AE model type: {model_type}")


def canonical_pixel_ae_run_id(
    *,
    model_type: str,
    backbone: str,
    input_size: int,
    epochs: int,
    loss: str,
    learning_rate: object,
    scope: str = "category",
    limit_train: int | None = None,
    legacy_suffix: str | None = None,
) -> str:
    scope = normalize_scope(scope)
    if model_type == "custom_autoencoder":
        prefix = f"custom_ae_scope-{scope}"
    elif model_type == "custom_unet_autoencoder":
        prefix = f"custom_unet_ae_scope-{scope}"
    elif model_type == "custom_constrained_unet_autoencoder":
        prefix = f"custom_constrained_unet_ae_scope-{scope}"
    elif model_type == "custom_mask_conditioned_inpainting_autoencoder":
        prefix = f"custom_mask_conditioned_inpainting_ae_scope-{scope}"
    elif model_type == "resnet_reconstruction_finetuned":
        prefix = "resnet_recon"
    else:
        raise ValueError(f"Unknown pixel AE model type: {model_type}")

    run_id = (
        f"{prefix}"
        f"_bb-{safe_tag(backbone)}"
        f"_s{int(input_size)}"
        f"_e{int(epochs)}"
        f"_loss-{safe_tag(loss)}"
        f"_lr{lr_tag(learning_rate)}"
    )
    if limit_train is not None:
        run_id += f"_debug-train{int(limit_train)}"
    if legacy_suffix:
        run_id += f"_legacy-{safe_tag(legacy_suffix)}"
    return compact_run_id(run_id)


def canonical_patchcore_run_id(
    *,
    feature_extractor: str,
    backbone: str,
    input_size: int,
    layers: list[str] | tuple[str, ...],
    max_memory_embeddings: int,
    checkpoint_run_id: str | None = None,
    limit_train: int | None = None,
    limit_test: int | None = None,
    legacy_suffix: str | None = None,
) -> str:
    layer_tag = "-".join(safe_tag(layer) for layer in layers)
    run_id = (
        f"pcore_fx-{safe_tag(feature_extractor)}"
        f"_bb-{safe_tag(backbone)}"
        f"_s{int(input_size)}"
        f"_layers-{layer_tag}"
        f"_mem{int(max_memory_embeddings)}"
    )
    if checkpoint_run_id:
        run_id += f"_ckpt-{safe_tag(checkpoint_run_id)}"
    if limit_train is not None or limit_test is not None:
        train = "all" if limit_train is None else str(int(limit_train))
        test = "all" if limit_test is None else str(int(limit_test))
        run_id += f"_debug-train{train}-test{test}"
    if legacy_suffix:
        run_id += f"_legacy-{safe_tag(legacy_suffix)}"
    return compact_run_id(run_id)


def patchcore_run_dir(category: str, run_id: str, output_dir: Path | None = None) -> Path:
    root = MODEL_ARTIFACTS.patchcore if output_dir is None else Path(output_dir)
    return root / category / run_id


def read_manifest(path: Path = REPORTS.manifest) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "artifact_kind",
                "old_path",
                "new_path",
                "category",
                "model_family",
                "run_id_old",
                "run_id_new",
                "status",
                "notes",
            ]
        )
    return pd.read_csv(path)


def resolve_legacy_path(path: str | Path, manifest_path: Path = REPORTS.manifest) -> Path:
    """Resolve a historical artifact path through reports/manifest.csv when possible."""
    candidate = Path(path)
    manifest = read_manifest(manifest_path)
    if manifest.empty:
        return candidate

    normalized = str(candidate).replace("\\", "/")
    for old_path, new_path in zip(manifest["old_path"], manifest["new_path"], strict=False):
        if str(old_path).replace("\\", "/") == normalized:
            return Path(str(new_path))
    return candidate





