"""Train/validation epoch loop for functional-surface segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from src.models.segmentation.metrics import metrics_from_logits
from src.models.segmentation.runtime import mask_logits_from_output, model_output


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    classes: list[int],
    sample_weight: torch.Tensor | None = None,
    pixel_weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)
    if pixel_weight is not None:
        if pixel_weight.ndim == 4:
            pixel_weight = pixel_weight[:, 0]
        pixel_weight = pixel_weight.to(device=logits.device, dtype=prob.dtype).clamp_min(0.0)
    losses = []
    for cls in classes:
        pred = prob[:, cls]
        truth = (target == int(cls)).to(dtype=prob.dtype)
        if pixel_weight is not None:
            pred = pred * pixel_weight
            truth = truth * pixel_weight
        intersection = (pred * truth).sum(dim=(1, 2))
        denominator = pred.sum(dim=(1, 2)) + truth.sum(dim=(1, 2))
        losses.append(1.0 - (2.0 * intersection + eps) / (denominator + eps))
    if not losses:
        return logits.sum() * 0.0
    per_sample = torch.stack(losses, dim=0).mean(dim=0)
    if sample_weight is None:
        return per_sample.mean()
    weights = sample_weight.to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-6)


def run_epoch(model, loader, optimizer, device, args, class_weights, dice_classes, epoch: int, phase: str, show_progress: bool):
    is_train = optimizer is not None
    model.train(is_train)
    total = 0.0
    count = 0
    sums: dict[str, float] = {}
    iterator = tqdm(loader, desc=f"epoch {epoch:03d} {phase}", leave=False, disable=not show_progress)
    for batch in iterator:
        target = batch["semantic"].to(device)
        sample_weight = batch["sample_weight"].to(device)
        with torch.set_grad_enabled(is_train):
            output = model_output(model, batch, device)
            logits = mask_logits_from_output(output)
            ce_raw = F.cross_entropy(logits, target, weight=class_weights, reduction="none")
            ce_per_sample = ce_raw.mean(dim=(1, 2))
            weights = sample_weight.to(dtype=ce_per_sample.dtype).clamp_min(0.0)
            ce = (ce_per_sample * weights).sum() / weights.sum().clamp_min(1e-6)
            dice = multiclass_dice_loss(logits, target, dice_classes, sample_weight=sample_weight)
            direct_loss = float(args.ce_weight) * ce + float(args.dice_weight) * dice
            loss = direct_loss
            recon_loss = logits.sum() * 0.0
            if isinstance(output, dict) and "recon_logits" in output and float(args.recon_weight) > 0:
                recon_logits = output["recon_logits"]
                recon_ce_raw = F.cross_entropy(recon_logits, target, weight=class_weights, reduction="none")
                recon_ce_per_sample = recon_ce_raw.mean(dim=(1, 2))
                recon_weights = sample_weight.to(dtype=recon_ce_per_sample.dtype).clamp_min(0.0)
                recon_ce = (recon_ce_per_sample * recon_weights).sum() / recon_weights.sum().clamp_min(1e-6)
                recon_dice = multiclass_dice_loss(
                    recon_logits,
                    target,
                    dice_classes,
                    sample_weight=sample_weight,
                )
                recon_loss = float(args.ce_weight) * recon_ce + float(args.dice_weight) * recon_dice
                loss = loss + float(args.recon_weight) * recon_loss
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = target.shape[0]
        total += float(loss.detach().cpu()) * batch_size
        count += batch_size
        batch_metrics = metrics_from_logits(logits.detach(), target, args.num_classes)
        if isinstance(output, dict) and "recon_logits" in output:
            recon_metrics = metrics_from_logits(output["recon_logits"].detach(), target, args.num_classes)
            for key, value in recon_metrics.items():
                batch_metrics[f"recon_{key}"] = value
        batch_metrics["ce_loss"] = float(ce.detach().cpu())
        batch_metrics["dice_loss"] = float(dice.detach().cpu())
        batch_metrics["recon_loss"] = float(recon_loss.detach().cpu())
        for key, value in batch_metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value) * batch_size
        iterator.set_postfix(loss=f"{total / max(count, 1):.5f}")
    return total / max(count, 1), {key: value / max(count, 1) for key, value in sums.items()}

