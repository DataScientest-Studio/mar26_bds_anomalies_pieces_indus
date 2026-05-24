"""Functional-surface segmentation architectures."""

from __future__ import annotations

from src.models.segmentation.models import (
    FunctionalSurfaceUNetResNet18,
    FunctionalSurfaceUNetResNet18Det1,
    FunctionalSurfaceUNetResNet18Det1Context2B,
    FunctionalSurfaceUNetResNet18Det1Context2BRecon,
    FunctionalSurfaceUNetResNet18Det1ContextFPN,
    FunctionalSurfaceUNetResNet18Det1ContextFPNLight,
    FunctionalSurfaceUNetSmall,
)

__all__ = [name for name in globals() if name.startswith("FunctionalSurface")]






