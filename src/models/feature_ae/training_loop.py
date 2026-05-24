"""Train/validation epoch loop for Feature AE / RD AE models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm

from src.models.feature_ae.models import feature_reconstruction_loss


def run_epoch(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader,
    device: torch.device,
    cosine_weight: float,
    layer_loss_weights: dict[str, float],
    optimizer: AdamW | None,
    epoch: int,
    total_epochs: int,
    phase: str,
    show_progress: bool,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    student.train(is_train)
    teacher.eval()
    total_loss = 0.0
    total_items = 0
    metric_sums: dict[str, float] = {}
    iterator = tqdm(
        loader,
        desc=f"epoch {epoch:03d}/{total_epochs:03d} {phase}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        context_images = batch.get("context_image")
        if context_images is not None:
            context_images = context_images.to(device, non_blocking=True)
        pixel_weight = None
        if "roi_mask" in batch:
            roi_mask = batch["roi_mask"].to(device, non_blocking=True).float()
            pixel_weight = (
                roi_mask * float(getattr(loader.dataset, "roi_loss_weight", 1.0))
                + (1.0 - roi_mask) * float(getattr(loader.dataset, "background_loss_weight", 0.02))
            )
            roi_border_loss_weight = float(getattr(loader.dataset, "roi_border_loss_weight", 0.0))
            roi_border_radius = int(getattr(loader.dataset, "roi_border_radius", 0))
            if roi_border_loss_weight > 0 and roi_border_radius > 0:
                kernel_size = roi_border_radius * 2 + 1
                eroded_roi = -F.max_pool2d(-roi_mask, kernel_size=kernel_size, stride=1, padding=roi_border_radius)
                roi_border = (roi_mask - eroded_roi).clamp_min(0.0)
                pixel_weight = pixel_weight + roi_border * roi_border_loss_weight
            if "normal_structure_mask" in batch:
                normal_structure = batch["normal_structure_mask"].to(device, non_blocking=True).float()
                pixel_weight = pixel_weight + normal_structure * float(
                    getattr(loader.dataset, "normal_structure_loss_weight", 0.0)
                )
        with torch.no_grad():
            targets = teacher(images)
        with torch.set_grad_enabled(is_train):
            predictions = student(images, context_images) if context_images is not None else student(images)
            loss, metrics = feature_reconstruction_loss(
                predictions,
                targets,
                cosine_weight=cosine_weight,
                pixel_weight=pixel_weight,
                layer_weights=layer_loss_weights,
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
        for key, value in metrics.items():
            metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * batch_size
        iterator.set_postfix(loss=f"{total_loss / max(total_items, 1):.6f}")
    averaged = {key: value / max(total_items, 1) for key, value in metric_sums.items()}
    return total_loss / max(total_items, 1), averaged

