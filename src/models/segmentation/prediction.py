"""Predict multiclass functional-surface masks and build readable previews."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

from src.config import PATHS
from src.features.functional_surface import safe_stem
from src.features.transforms import crop_box_to_mask, resize_letterbox_pil
from src.models.segmentation.models import build_segmentation_model
from src.models.segmentation.runtime import replace_segmentation_head
from src.models.baselines.patchcore import IMAGENET_MEAN, IMAGENET_STD, project_path, resolve_device


COLORS = np.array(
    [
        [0, 0, 0],
        [40, 140, 230],
        [255, 128, 20],
    ],
    dtype=np.uint8,
)

FILENAME_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}_\d{3})(?P<sep>[-_])(?P<suffix>.+)$"
)


def stack_or_object(arrays: list[np.ndarray], dtype) -> np.ndarray:
    try:
        return np.stack(arrays, axis=0).astype(dtype, copy=False)
    except ValueError:
        return np.array(arrays, dtype=object)


def project_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PATHS.root.resolve()))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict surface/landmark multiclass masks.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, default=PATHS.root / "data" / "classified" / "Casting_class1" / "train" / "good" / "classified_manifest.csv")
    parser.add_argument(
        "--image-dir",
        type=Path,
        action="append",
        default=None,
        help="Optional raw image directory. Can be passed multiple times. When set, a manifest is inferred from image filenames instead of --manifest-csv.",
    )
    parser.add_argument("--image-glob", default="*.jpg")
    parser.add_argument("--split", default=None, help="Split metadata used when --image-dir is set.")
    parser.add_argument("--label", default=None, help="Label metadata used when --image-dir is set, e.g. good or defective.")
    parser.add_argument("--exclude-labels-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument(
        "--context-size",
        type=int,
        default=None,
        help="Global context size. Defaults to checkpoint context_size when available, otherwise --input-size.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--mask-output-size",
        choices=["original", "input"],
        default="original",
        help="Save binary/semantic masks at original image size for review, or at network input size.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preview-per-pattern", type=int, default=40)
    parser.add_argument(
        "--write-labels-index",
        action="store_true",
        help="Also write labels_index.csv so the predictions can be corrected and used as an external monitor dataset.",
    )
    parser.add_argument("--label-source", default="prediction_seed_multiclass")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PATHS.root / path


def load_excluded_stems(labels_dir: Path | None) -> set[str]:
    if labels_dir is None:
        return set()
    labels_path = resolve(labels_dir) / "labels_index.csv"
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    labels = pd.read_csv(labels_path)
    return {Path(str(path)).stem for path in labels["image_path"]}


def parse_image_name(image_path: str | Path) -> dict[str, str]:
    stem = Path(str(image_path)).stem
    match = FILENAME_RE.match(stem)
    if not match:
        return {"acquisition_group": stem, "view_key": "", "group_index": ""}
    timestamp = match.group("timestamp")
    tokens = match.group("suffix").split("_")
    if len(tokens) >= 3:
        group_index = tokens[0]
        view_key = "_".join(tokens[-2:])
        acquisition_group = f"{timestamp}-{group_index}"
    else:
        group_index = ""
        view_key = "_".join(tokens)
        acquisition_group = timestamp
    return {"acquisition_group": acquisition_group, "view_key": view_key, "group_index": group_index}


def c1_pattern(view_key: str, group_views: set[str]) -> tuple[str, str, str]:
    if view_key == "1_2":
        return "P1", "piece_B_triplet_3views", "1_2"
    if view_key == "1_3":
        return "P2", "piece_B_triplet_3views", "1_3"
    if view_key == "2_3" and ("1_2" in group_views or "1_3" in group_views):
        return "P3", "piece_B_triplet_3views", "2_3"
    if view_key == "2_3":
        return "P4", "piece_A_single_2_3", "2_3"
    return "UNKNOWN", "unknown", view_key


def manifest_from_image_dirs(image_dirs: list[Path], image_glob: str, split: str | None, label: str | None) -> pd.DataFrame:
    all_rows: list[dict] = []
    for image_dir in image_dirs:
        all_rows.extend(manifest_rows_from_image_dir(image_dir, image_glob, split, label))
    return pd.DataFrame(all_rows)


def manifest_rows_from_image_dir(image_dir: Path, image_glob: str, split: str | None, label: str | None) -> list[dict]:
    root = resolve(image_dir)
    paths = sorted(root.glob(str(image_glob)))
    if not paths:
        raise FileNotFoundError(f"No images matched {root / image_glob}")
    rows = []
    for path in paths:
        info = parse_image_name(path)
        rows.append(
            {
                "source_path": str(path),
                "split": split or root.parent.name,
                "label": label or root.name,
                **info,
            }
        )
    by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_group[str(row["acquisition_group"])].add(str(row["view_key"]))
    for row in rows:
        pattern_id, piece_family, pattern_view = c1_pattern(str(row["view_key"]), by_group[str(row["acquisition_group"])])
        row.update(
            {
                "pattern_id": pattern_id,
                "base_pattern": pattern_id,
                "piece_family": piece_family,
                "pattern_view": pattern_view,
                "group_views": "|".join(sorted(by_group[str(row["acquisition_group"])])),
            }
        )
    return rows


def image_tensor(image: Image.Image, input_size: int) -> torch.Tensor:
    image = resize_letterbox_pil(image, input_size, mode="RGB")
    tensor = transforms.ToTensor()(image.convert("RGB"))
    return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(tensor)


def denormalized_letterbox(image: Image.Image, input_size: int) -> np.ndarray:
    return np.asarray(resize_letterbox_pil(image, input_size, mode="RGB"), dtype=np.uint8)


def overlay(rgb: np.ndarray, semantic: np.ndarray, alpha: float = 0.48) -> np.ndarray:
    out = rgb.copy()
    fg = semantic > 0
    color = COLORS[np.clip(semantic, 0, 2)]
    out[fg] = ((1.0 - alpha) * out[fg] + alpha * color[fg]).astype(np.uint8)
    return out


def make_preview(rgb: np.ndarray, semantic: np.ndarray, surface_prob: np.ndarray, landmark_prob: np.ndarray, title: str) -> Image.Image:
    h, w = rgb.shape[:2]
    panels = [
        ("image", rgb),
        ("classes", overlay(rgb, semantic)),
        ("surface p", ((np.clip(surface_prob, 0, 1) * 255).astype(np.uint8))),
        ("landmark p", ((np.clip(landmark_prob, 0, 1) * 255).astype(np.uint8))),
    ]
    canvas = Image.new("RGB", (w * len(panels), h + 44), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill=(0, 0, 0))
    for idx, (label, panel) in enumerate(panels):
        x = idx * w
        draw.text((x + 4, 24), label, fill=(0, 0, 0))
        if panel.ndim == 2:
            img = Image.fromarray(panel, mode="L").convert("RGB")
        else:
            img = Image.fromarray(panel, mode="RGB")
        canvas.paste(img, (x, 44))
    return canvas


def resize_probabilities(prob: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize CxHxW probabilities to PIL size=(W,H) before argmax."""
    width, height = size
    tensor = torch.from_numpy(prob.astype(np.float32))[None]
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return resized[0].numpy()


def make_contact_sheet(rows: list[dict], output_path: Path, *, max_items: int) -> None:
    selected = rows[: max(1, int(max_items))]
    if not selected:
        return
    thumbs = []
    for row in selected:
        image = Image.open(row["preview_path"]).convert("RGB")
        image.thumbnail((256, 160), Image.Resampling.LANCZOS)
        thumbs.append((row, image.copy()))
    cols = 4
    cell_w, cell_h = 256, 186
    rows_count = int(np.ceil(len(thumbs) / cols))
    canvas = Image.new("RGB", (cols * cell_w, rows_count * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (row, thumb) in enumerate(thumbs):
        x = (idx % cols) * cell_w
        y = (idx // cols) * cell_h
        draw.text((x + 3, y + 3), f"{row['pattern_id']} | s={row['surface_ratio']:.2f} l={row['landmark_ratio']:.2f}", fill=(0, 0, 0))
        canvas.paste(thumb, (x, y + 22))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve(args.checkpoint_path)
    output_dir = resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    masks_dir = output_dir / "masks"
    previews_dir = output_dir / "previews"
    prob_dir = output_dir / "prob_maps"
    for directory in (masks_dir, previews_dir, prob_dir):
        directory.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = str(checkpoint.get("model_type", "functional_unet_resnet18_det1_context2b"))
    num_classes = int(checkpoint.get("num_classes", 3))
    context_size = int(args.context_size) if args.context_size is not None else int(checkpoint.get("context_size", args.input_size))
    model = build_segmentation_model(model_type)
    replace_segmentation_head(model, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = resolve_device(args.device)
    model = model.to(device).eval()

    if args.image_dir is not None:
        manifest = manifest_from_image_dirs(args.image_dir, args.image_glob, args.split, args.label)
    else:
        manifest = pd.read_csv(resolve(args.manifest_csv))
    excluded = load_excluded_stems(args.exclude_labels_dir)
    rows = manifest[~manifest["source_path"].map(lambda p: Path(str(p)).stem in excluded)].copy()
    if args.limit is not None:
        rows = rows.head(int(args.limit)).copy()

    out_rows = []
    image_paths = []
    pattern_keys = []
    prob_maps = []
    binary_masks = []
    with torch.inference_mode():
        for idx, row in rows.reset_index(drop=True).iterrows():
            image_path = project_path(str(row["source_path"]))
            stored_image_path = project_relative_path(image_path)
            image = Image.open(image_path).convert("RGB")
            rgb_input = denormalized_letterbox(image, int(args.input_size))
            tensor = image_tensor(image, int(args.input_size))[None].to(device)
            global_image = image_tensor(image, context_size)[None].to(device)
            crop_box_mask = transforms.ToTensor()(crop_box_to_mask((0.5, 0.5, 1.0, 1.0), context_size))[None].to(device)
            output = model(tensor, global_image=global_image, crop_box_mask=crop_box_mask)
            logits = output["mask_logits"] if isinstance(output, dict) else output
            prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
            semantic_input = np.argmax(prob, axis=0).astype(np.uint8)
            if args.mask_output_size == "original":
                save_prob = resize_probabilities(prob, image.size)
                semantic = np.argmax(save_prob, axis=0).astype(np.uint8)
            else:
                save_prob = prob
                semantic = semantic_input

            stem = f"{idx:04d}_{row['pattern_id']}_{safe_stem(str(image_path))}"
            semantic_path = masks_dir / f"{stem}_semantic_012.png"
            surface_path = masks_dir / f"{stem}_surface.png"
            landmark_path = masks_dir / f"{stem}_landmark.png"
            negative_path = masks_dir / f"{stem}_negative.png"
            weight_path = masks_dir / f"{stem}_weight.png"
            preview_path = previews_dir / f"{stem}_preview.png"
            npz_path = prob_dir / f"{stem}_prob.npz"
            Image.fromarray(semantic, mode="L").save(semantic_path)
            Image.fromarray(((semantic == 1).astype(np.uint8) * 255), mode="L").save(surface_path)
            Image.fromarray(((semantic == 2).astype(np.uint8) * 255), mode="L").save(landmark_path)
            Image.fromarray(((semantic == 0).astype(np.uint8) * 255), mode="L").save(negative_path)
            Image.fromarray(((semantic != 2).astype(np.uint8) * 255), mode="L").save(weight_path)
            np.savez_compressed(npz_path, prob=prob.astype(np.float16))
            surface_prob = save_prob[1].astype(np.float32, copy=False)
            surface_binary = (semantic == 1).astype(np.uint8)
            image_paths.append(stored_image_path)
            pattern_keys.append(str(row["pattern_id"]))
            prob_maps.append(surface_prob)
            binary_masks.append(surface_binary)
            make_preview(
                rgb_input,
                semantic_input,
                prob[1],
                prob[2],
                f"{row['pattern_id']} | {image_path.name}",
            ).save(preview_path)
            out_rows.append(
                {
                    "image_index": int(idx),
                    "image_path": stored_image_path,
                    "pattern_id": row["pattern_id"],
                    "base_pattern": row.get("base_pattern", row["pattern_id"]),
                    "category": "Casting_class1",
                    "split": row.get("split", ""),
                    "label": row.get("label", ""),
                    "view_key": row.get("view_key", row.get("pattern_view", "")),
                    "piece_family": row.get("piece_family", ""),
                    "semantic_mask_path": str(semantic_path),
                    "surface_mask_path": str(surface_path),
                    "landmark_mask_path": str(landmark_path),
                    "positive_mask_path": str(surface_path),
                    "pseudo_mask_path": str(surface_path),
                    "ignore_mask_path": str(landmark_path),
                    "negative_mask_path": str(negative_path),
                    "weight_map_path": str(weight_path),
                    "prob_path": str(npz_path),
                    "preview_path": str(preview_path),
                    "surface_ratio": float((semantic == 1).mean()),
                    "landmark_ratio": float((semantic == 2).mean()),
                    "ignore_ratio": float((semantic == 2).mean()),
                    "mask_output_size": args.mask_output_size,
                    "mask_width": int(semantic.shape[1]),
                    "mask_height": int(semantic.shape[0]),
                    "label_source": args.label_source,
                    "sample_weight": 1.0,
                }
            )

    summary = pd.DataFrame(out_rows)
    summary.to_csv(output_dir / "prediction_summary.csv", index=False)
    np.savez_compressed(
        output_dir / "functional_surface_predictions.npz",
        image_path=np.array(image_paths, dtype=object),
        pattern_key=np.array(pattern_keys, dtype=object),
        prob_maps=stack_or_object(prob_maps, np.float32),
        binary_masks=stack_or_object(binary_masks, np.uint8),
        source=np.array(["multiclass_surface_class"], dtype=object),
        surface_class_index=np.array([1], dtype=np.int64),
    )
    if args.write_labels_index:
        summary.to_csv(output_dir / "labels_index.csv", index=False)
    for pattern_id, group in summary.groupby("pattern_id", sort=True):
        make_contact_sheet(group.to_dict("records"), output_dir / f"contact_sheet_{pattern_id}.png", max_items=args.preview_per_pattern)
    make_contact_sheet(summary.to_dict("records"), output_dir / "contact_sheet_all.png", max_items=args.preview_per_pattern * 4)
    (output_dir / "params.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_path),
                "model_type": model_type,
                "num_classes": num_classes,
                "input_size": int(args.input_size),
                "context_size": context_size,
                "mask_output_size": args.mask_output_size,
                "excluded_count": len(excluded),
                "predicted_count": len(summary),
                "output_dir": str(output_dir),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "predicted": len(summary)}, indent=2))


if __name__ == "__main__":
    main()






