"""Runtime helpers for functional-surface segmentation models.

This module is the stable ``src`` home for inference utilities that were
historically defined in training scripts. Script-level definitions can keep
re-exporting these names during the migration.
"""

from __future__ import annotations

import torch


def replace_segmentation_head(model: torch.nn.Module, num_classes: int) -> None:
    """Replace segmentation and optional reconstruction heads in-place."""
    if not hasattr(model, "out"):
        raise ValueError("Model does not expose an 'out' segmentation head.")
    old_head = model.out
    if not isinstance(old_head, torch.nn.Conv2d):
        raise ValueError(f"Unsupported segmentation head type: {type(old_head)}")
    model.out = torch.nn.Conv2d(old_head.in_channels, int(num_classes), kernel_size=old_head.kernel_size)
    if hasattr(model, "recon_out"):
        old_recon_head = model.recon_out
        if not isinstance(old_recon_head, torch.nn.Conv2d):
            raise ValueError(f"Unsupported reconstruction head type: {type(old_recon_head)}")
        model.recon_out = torch.nn.Conv2d(
            old_recon_head.in_channels,
            int(num_classes),
            kernel_size=old_recon_head.kernel_size,
        )


def mask_logits_from_output(output) -> torch.Tensor:
    if isinstance(output, dict):
        return output["mask_logits"]
    return output


def mask_logits_from_model_output(output) -> torch.Tensor:
    return mask_logits_from_output(output)


def model_output(model: torch.nn.Module, batch: dict, device) -> torch.Tensor | dict:
    images = batch["image"].to(device)
    kwargs = {}
    if "global_image" in batch:
        kwargs["global_image"] = batch["global_image"].to(device)
    if "crop_box_mask" in batch:
        kwargs["crop_box_mask"] = batch["crop_box_mask"].to(device)
    try:
        return model(images, **kwargs)
    except TypeError:
        return model(images)


def model_forward(model: torch.nn.Module, batch_or_images, device=None):
    """Forward helper for binary and multiclass functional-surface models."""
    if isinstance(batch_or_images, dict):
        return model_output(model, batch_or_images, device)
    return model(batch_or_images)


def model_mask_logits(model: torch.nn.Module, batch: dict, device) -> torch.Tensor:
    """Return only mask logits for multiclass training/evaluation loops."""
    return mask_logits_from_output(model_output(model, batch, device))






