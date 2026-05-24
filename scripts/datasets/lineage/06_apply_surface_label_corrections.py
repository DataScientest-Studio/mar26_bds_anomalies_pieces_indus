"""Build a derived crop dataset by replacing selected masks with model predictions."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.config import PATHS
from src.data.mask_datasets import FunctionalSurfaceLabelDataset
from src.features.functional_surface import preview_panel, safe_stem
from src.models.segmentation.models import build_segmentation_model
from src.models.baselines.patchcore import ResizeLetterbox, project_path, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace manually selected crop masks with checkpoint predictions.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, default=None)
    parser.add_argument("--image-index", type=int, nargs="*", default=[])
    parser.add_argument("--row-index", type=int, nargs="*", default=[])
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target-mode", choices=["binary", "soft"], default="binary")
    parser.add_argument(
        "--preserve-ignore",
        action="store_true",
        help="Keep the existing ignore and weight maps, and recompute negative outside positive and ignore.",
    )
    parser.add_argument(
        "--clear-ignore-for-corrections",
        action="store_true",
        help="For corrected rows, write empty ignore maps and full weight maps.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--preview-count", type=int, default=200)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    return project_path(str(path))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PATHS.root))
    except ValueError:
        return str(path)


def read_selection(args: argparse.Namespace, labels: pd.DataFrame) -> pd.DataFrame:
    selected: list[int] = []
    if args.selection_csv is not None:
        selection = pd.read_csv(resolve(args.selection_csv))
        decision_col = None
        for candidate in ["decision", "action"]:
            if candidate in selection.columns:
                decision_col = candidate
                break
        if decision_col is not None:
            selection = selection[selection[decision_col].astype(str).str.lower().isin({"replace", "use_prediction", "prediction", "accept"})]
        if "image_index" in selection.columns:
            selected.extend(selection["image_index"].dropna().astype(int).tolist())
        elif "row_index" in selection.columns:
            row_ids = selection["row_index"].dropna().astype(int).tolist()
            selected.extend(labels.iloc[row_ids]["image_index"].astype(int).tolist())
        else:
            raise KeyError("Selection CSV must contain image_index or row_index.")
    selected.extend(int(value) for value in args.image_index)
    if args.row_index:
        selected.extend(labels.iloc[[int(value) for value in args.row_index]]["image_index"].astype(int).tolist())

    selected_ids = sorted(set(selected))
    if not selected_ids:
        raise ValueError("No rows selected. Provide --selection-csv, --image-index, or --row-index.")
    selected_df = labels[labels["image_index"].astype(int).isin(selected_ids)].copy()
    missing = sorted(set(selected_ids) - set(selected_df["image_index"].astype(int).tolist()))
    if missing:
        raise ValueError(f"Selected image_index values not found in labels_index.csv: {missing[:20]}")
    return selected_df


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_segmentation_model(checkpoint.get("model_type", "functional_unet_small"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


def mask_logits_from_model_output(output) -> torch.Tensor:
    if isinstance(output, dict):
        return output["mask_logits"]
    return output


def save_mask(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def load_mask_array(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(resolve(path)).convert("L")
    if mask.size != size:
        mask = mask.resize(size, resample=Image.Resampling.NEAREST)
    return (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)


def unletterbox_prediction(prob: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    width, height = original_size
    square_size = int(prob.shape[0])
    scale = square_size / max(width, height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    left = (square_size - resized_width) // 2
    top = (square_size - resized_height) // 2
    cropped = prob[top : top + resized_height, left : left + resized_width]
    image = Image.fromarray(np.clip(cropped * 255.0, 0, 255).astype(np.uint8), mode="L")
    image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def prediction_for_row(
    model,
    labels: pd.DataFrame,
    row_pos: int,
    input_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = FunctionalSurfaceLabelDataset(labels.iloc[[row_pos]].reset_index(drop=True), input_size, augmentation_profile=None)
    sample = dataset[0]
    image = sample["image"].unsqueeze(0).to(device)
    with torch.inference_mode():
        prob = torch.sigmoid(mask_logits_from_model_output(model(image)))[0, 0].cpu().numpy().astype(np.float32)
    target = sample["pseudo"][0].cpu().numpy().astype(np.float32)
    return prob, target


def write_correction_preview(
    row: pd.Series,
    prob: np.ndarray,
    previous: np.ndarray,
    new_target: np.ndarray,
    input_size: int,
    threshold: float,
    path: Path,
) -> None:
    raw_image = Image.open(resolve(row["image_path"])).convert("RGB")
    image = np.asarray(ResizeLetterbox(int(input_size))(raw_image), dtype=np.float32) / 255.0
    def mask_preview(mask: np.ndarray) -> np.ndarray:
        if mask.shape == (int(input_size), int(input_size)):
            return mask.astype(np.float32)
        mask_image = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), mode="L")
        arr = np.asarray(ResizeLetterbox(int(input_size))(mask_image), dtype=np.float32) / 255.0
        return arr[..., 0] if arr.ndim == 3 else arr

    previous = mask_preview(previous)
    new_target = mask_preview(new_target)
    pred_binary = (prob >= float(threshold)).astype(np.float32)
    overlay_before = image.copy()
    overlay_before[..., 1] = np.maximum(overlay_before[..., 1], previous * 0.85)
    overlay_after = image.copy()
    overlay_after[..., 0] = np.maximum(overlay_after[..., 0], pred_binary * 0.95)
    overlay_after[..., 1] = np.maximum(overlay_after[..., 1], new_target * 0.75)
    preview_panel(
        [
            ("image", image),
            ("old mask", previous),
            ("prediction", prob),
            ("pred binary", pred_binary),
            ("new target", new_target),
            ("old overlay green", overlay_before),
            ("new overlay red/green", overlay_after),
        ],
        f"image_index={int(row['image_index'])} mode={path.parent.name}",
        path,
    )


def main() -> None:
    args = parse_args()
    dataset_dir = resolve(args.dataset_dir)
    output_dir = resolve(args.output_dir)
    corrected_masks_dir = output_dir / "prediction_corrected_masks"
    previews_dir = output_dir / "prediction_correction_previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_masks_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(dataset_dir / "labels_index.csv")
    selected = read_selection(args, labels)
    device = resolve_device(args.device)
    model = load_model(resolve(args.checkpoint_path), device)

    output_labels = labels.copy()
    corrections = []
    preview_count = 0
    for row_pos, row in selected.iterrows():
        prob, previous = prediction_for_row(model, labels, int(row_pos), int(args.input_size), device)
        raw_size = Image.open(resolve(row["image_path"])).size
        prob_full = unletterbox_prediction(prob, raw_size)
        binary = (prob >= float(args.threshold)).astype(np.float32)
        binary_full = (prob_full >= float(args.threshold)).astype(np.float32)
        pseudo = prob_full if args.target_mode == "soft" else binary_full
        positive = binary_full
        if args.clear_ignore_for_corrections:
            ignore = np.zeros_like(binary_full, dtype=np.float32)
        elif args.preserve_ignore:
            ignore_path = row.get("ignore_mask_path", "")
            ignore = (
                load_mask_array(ignore_path, raw_size)
                if pd.notna(ignore_path) and str(ignore_path).strip()
                else np.zeros_like(binary_full, dtype=np.float32)
            )
            positive = positive * (1.0 - ignore)
            pseudo = pseudo * (1.0 - ignore)
        else:
            ignore = np.zeros_like(binary_full, dtype=np.float32)
        negative = ((positive < 0.5) & (ignore < 0.5)).astype(np.float32)

        stem = safe_stem(Path(str(row["image_path"])).stem)
        prefix = f"{int(row['image_index']):04d}_{stem}"
        pseudo_path = corrected_masks_dir / f"{prefix}_pseudo_{args.target_mode}.png"
        positive_path = corrected_masks_dir / f"{prefix}_positive.png"
        negative_path = corrected_masks_dir / f"{prefix}_negative.png"
        ignore_path = corrected_masks_dir / f"{prefix}_ignore.png"
        weight_path = corrected_masks_dir / f"{prefix}_weight.png"
        save_mask(pseudo, pseudo_path)
        save_mask(positive, positive_path)
        save_mask(negative, negative_path)
        if args.clear_ignore_for_corrections or not args.preserve_ignore:
            save_mask(ignore, ignore_path)
        if args.clear_ignore_for_corrections:
            save_mask(np.ones_like(ignore, dtype=np.float32), weight_path)

        row_filter = output_labels["image_index"].astype(int) == int(row["image_index"])
        output_labels.loc[row_filter, "pseudo_mask_path"] = rel(pseudo_path)
        output_labels.loc[row_filter, "positive_mask_path"] = rel(positive_path)
        output_labels.loc[row_filter, "negative_mask_path"] = rel(negative_path)
        if args.clear_ignore_for_corrections or not args.preserve_ignore:
            output_labels.loc[row_filter, "ignore_mask_path"] = rel(ignore_path)
        if args.clear_ignore_for_corrections:
            output_labels.loc[row_filter, "weight_map_path"] = rel(weight_path)
        output_labels.loc[row_filter, "pseudo_is_soft"] = args.target_mode == "soft"
        output_labels.loc[row_filter, "label_correction"] = "prediction"

        if preview_count < int(args.preview_count):
            write_correction_preview(
                row,
                prob,
                previous,
                (pseudo >= float(args.threshold)).astype(np.float32) if args.target_mode == "soft" else pseudo,
                int(args.input_size),
                float(args.threshold),
                previews_dir / f"{preview_count:04d}_{prefix}_preview.png",
            )
            preview_count += 1

        corrections.append(
            {
                "image_index": int(row["image_index"]),
                "row_index": int(row_pos),
                "old_positive_ratio": float((previous >= 0.5).mean()),
                "new_positive_ratio": float((positive >= 0.5).mean()),
                "prediction_mean": float(prob_full.mean()),
                "pseudo_mask_path": rel(pseudo_path),
                "positive_mask_path": rel(positive_path),
                "negative_mask_path": rel(negative_path),
                "preserve_ignore": bool(args.preserve_ignore),
                "clear_ignore_for_corrections": bool(args.clear_ignore_for_corrections),
                "target_mode": args.target_mode,
            }
        )

    output_labels.to_csv(output_dir / "labels_index.csv", index=False)
    pd.DataFrame(corrections).to_csv(output_dir / "prediction_label_corrections.csv", index=False)
    for name in ["params.json", "crop_summary.csv"]:
        source = dataset_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)

    summary = {
        "dataset_dir": rel(dataset_dir),
        "checkpoint_path": rel(resolve(args.checkpoint_path)),
        "output_dir": rel(output_dir),
        "input_rows": int(len(labels)),
        "corrected_rows": int(len(corrections)),
        "target_mode": args.target_mode,
        "threshold": float(args.threshold),
        "preserve_ignore": bool(args.preserve_ignore),
        "clear_ignore_for_corrections": bool(args.clear_ignore_for_corrections),
    }
    (output_dir / "prediction_label_correction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()






