"""DINO/Dinomaly baseline record.

The Dinomaly experiments were run as an external Anomalib-style baseline.
This module keeps the baseline visible in the published code tree without
pretending that the external trainer is part of the retained pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DinomalyBaselineConfig:
    """Compact description of the published Dinomaly baseline."""

    encoder: str = "dinov2reg_vit_base_14"
    role: str = "External feature-reconstruction baseline kept for evidence"
    report: str = "reports/docs/final_feature_ae_synthesis.md"


DEFAULT_DINOMALY_BASELINE = DinomalyBaselineConfig()

__all__ = ["DinomalyBaselineConfig", "DEFAULT_DINOMALY_BASELINE"]



