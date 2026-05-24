"""Metrics helpers for functional-surface segmentation training."""

from __future__ import annotations

import torch

__all__ = ["metrics_from_logits", "prefixed_metrics"]


def metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> dict[str, float]:
    with torch.no_grad():
        pred = torch.argmax(logits, dim=1)
        metrics = {}
        ious = []
        dices = []
        for cls in range(int(num_classes)):
            p = pred == cls
            t = target == cls
            tp = (p & t).sum().float()
            fp = (p & ~t).sum().float()
            fn = (~p & t).sum().float()
            iou = tp / (tp + fp + fn).clamp_min(1.0)
            dice = (2.0 * tp) / (2.0 * tp + fp + fn).clamp_min(1.0)
            metrics[f"iou_class{cls}"] = float(iou.cpu())
            metrics[f"dice_class{cls}"] = float(dice.cpu())
            if cls > 0:
                ious.append(iou)
                dices.append(dice)
        if ious:
            metrics["mean_fg_iou"] = float(torch.stack(ious).mean().cpu())
            metrics["mean_fg_dice"] = float(torch.stack(dices).mean().cpu())
        return metrics


def prefixed_metrics(prefix: str, loss: float | None, metrics: dict[str, float]) -> dict[str, float | None]:
    output: dict[str, float | None] = {f"{prefix}_loss": loss}
    output.update({f"{prefix}_{key}": value for key, value in metrics.items()})
    return output





