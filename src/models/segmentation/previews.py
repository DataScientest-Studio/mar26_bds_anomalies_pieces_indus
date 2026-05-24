"""Validation preview helpers for functional-surface segmentation runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.features.functional_surface import safe_stem
from src.models.segmentation.runtime import mask_logits_from_output, model_output
from src.models.baselines.patchcore import IMAGENET_MEAN, IMAGENET_STD


def save_functional_surface_previews(
    run_dir: Path,
    model,
    dataset,
    device,
    max_items: int,
    *,
    preview_head: str = "mask",
) -> None:
    """Save compact image/GT/prediction panels for ROI validation."""

    preview_dir = run_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    colors = np.array([[0, 0, 0], [40, 140, 230], [255, 128, 20]], dtype=np.uint8)
    model.eval()
    with torch.inference_mode():
        for idx in range(min(int(max_items), len(dataset))):
            item = dataset[idx]
            batch = {
                "image": item["image"][None],
                "global_image": item["global_image"][None],
                "crop_box_mask": item["crop_box_mask"][None],
            }
            output = model_output(model, batch, device)
            if preview_head == "recon" and isinstance(output, dict) and "recon_logits" in output:
                logits = output["recon_logits"]
                pred_title = "reconstruction"
            else:
                logits = mask_logits_from_output(output)
                pred_title = "prediction"
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)
            target = item["semantic"].numpy().astype(np.uint8)
            image = item["image"].detach().cpu()
            mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
            std = torch.tensor(IMAGENET_STD)[:, None, None]
            rgb = ((image * std + mean).clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            panels = []
            for title, mask in [("image", None), ("GT", target), (pred_title, pred)]:
                panel = rgb.copy()
                if mask is not None:
                    overlay = colors[np.clip(mask, 0, 2)]
                    fg = mask > 0
                    panel[fg] = (0.55 * panel[fg] + 0.45 * overlay[fg]).astype(np.uint8)
                panels.append((title, Image.fromarray(panel)))
            canvas = Image.new("RGB", (rgb.shape[1] * 3, rgb.shape[0] + 24), "white")
            draw = ImageDraw.Draw(canvas)
            for panel_idx, (title, panel) in enumerate(panels):
                x = panel_idx * rgb.shape[1]
                draw.text((x + 4, 4), title, fill=(0, 0, 0))
                canvas.paste(panel, (x, 24))
            canvas.save(preview_dir / f"{idx:03d}_{safe_stem(item['image_path'])}.png")






