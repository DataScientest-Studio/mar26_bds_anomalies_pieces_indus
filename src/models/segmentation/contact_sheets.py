"""Build contact sheets from functional-surface prediction previews."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.config import PATHS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact contact sheets for prediction preview PNGs.")
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, default=None)
    parser.add_argument("--group-column", default="pattern_id")
    parser.add_argument("--image-path-column", default="image_path")
    parser.add_argument("--preview-dirname", default="previews")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--thumb-width", type=int, default=240)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--max-per-sheet", type=int, default=140)
    parser.add_argument("--sort-by", default="image_path")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PATHS.root / path


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PATHS.root))
    except ValueError:
        return str(path)


def stem_key(path: str | Path) -> str:
    return Path(str(path)).stem


def preview_lookup(preview_dir: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    for path in sorted(preview_dir.glob("*.png")):
        name = path.stem
        # predict_functional_surface_tiled.py writes previews as:
        # 0000_<original_stem>_<hash>.png
        parts = name.split("_")
        key = "_".join(parts[1:-1]) if len(parts) > 2 and parts[0].isdigit() else name
        lookup[key] = path
    return lookup


def load_rows(args: argparse.Namespace, prediction_dir: Path) -> pd.DataFrame:
    summary_path = prediction_dir / "prediction_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    pred = pd.read_csv(summary_path)
    pred["stem_key"] = pred["image_path"].map(stem_key)
    if "preview_path" in pred.columns:
        pred["preview_path"] = pred["preview_path"].map(lambda path: resolve(Path(str(path))))
        missing_preview_path = ~pred["preview_path"].map(lambda path: Path(path).exists())
        if bool(missing_preview_path.any()):
            lookup = preview_lookup(prediction_dir / args.preview_dirname)
            pred.loc[missing_preview_path, "preview_path"] = pred.loc[missing_preview_path, "stem_key"].map(lookup)
    else:
        pred["preview_path"] = pred["stem_key"].map(preview_lookup(prediction_dir / args.preview_dirname))

    if args.labels_csv is not None:
        labels = pd.read_csv(resolve(args.labels_csv))
        labels["stem_key"] = labels[str(args.image_path_column)].map(stem_key)
        columns = ["stem_key", str(args.group_column)]
        if "meta_pattern" in labels.columns:
            columns.append("meta_pattern")
        pred = pred.merge(labels[columns].drop_duplicates("stem_key"), on="stem_key", how="left")
    else:
        pred[str(args.group_column)] = "all"

    missing = pred["preview_path"].isna().sum()
    if missing:
        raise FileNotFoundError(f"{missing} prediction previews could not be matched.")
    return pred


def draw_contact_sheet(rows: pd.DataFrame, output_path: Path, *, title: str, thumb_width: int, cols: int) -> None:
    paths = [Path(path) for path in rows["preview_path"].tolist()]
    if not paths:
        return

    title_h = 34
    label_h = 22
    margin = 10
    gutter = 8
    thumbs: list[Image.Image] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        ratio = thumb_width / image.width
        thumb = image.resize((thumb_width, max(1, int(round(image.height * ratio)))), Image.Resampling.LANCZOS)
        thumbs.append(thumb)

    thumb_height = max(t.height for t in thumbs)
    rows_count = int(math.ceil(len(thumbs) / cols))
    width = margin * 2 + cols * thumb_width + (cols - 1) * gutter
    height = margin * 2 + title_h + rows_count * (thumb_height + label_h) + max(0, rows_count - 1) * gutter
    sheet = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)

    for idx, (thumb, (_, row)) in enumerate(zip(thumbs, rows.iterrows(), strict=True)):
        r = idx // cols
        c = idx % cols
        x = margin + c * (thumb_width + gutter)
        y = margin + title_h + r * (thumb_height + label_h + gutter)
        sheet.paste(thumb, (x, y))
        label = f"{int(row.name):03d} ratio={float(row.get('pred_ratio', 0.0)):.3f}"
        draw.text((x, y + thumb_height + 3), label, fill=(35, 35, 35), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    prediction_dir = resolve(args.prediction_dir)
    output_dir = resolve(args.output_dir) if args.output_dir else prediction_dir / "contact_sheets"
    rows = load_rows(args, prediction_dir)
    sort_column = str(args.sort_by)
    if sort_column in rows.columns:
        rows = rows.sort_values(sort_column)

    group_column = str(args.group_column)
    summary_rows = []
    for group, group_rows in rows.groupby(group_column, dropna=False):
        group_label = "unknown" if pd.isna(group) else str(group)
        group_rows = group_rows.reset_index(drop=True)
        chunks = [
            group_rows.iloc[start : start + int(args.max_per_sheet)]
            for start in range(0, len(group_rows), int(args.max_per_sheet))
        ]
        for chunk_index, chunk in enumerate(chunks, start=1):
            suffix = f"_part{chunk_index:02d}" if len(chunks) > 1 else ""
            output_path = output_dir / f"{group_label}{suffix}_contact_sheet.png"
            title = f"{group_label} | {len(chunk)} previews | {rel(prediction_dir)}"
            draw_contact_sheet(chunk, output_path, title=title, thumb_width=int(args.thumb_width), cols=int(args.cols))
            summary_rows.append(
                {
                    "group": group_label,
                    "part": chunk_index,
                    "items": len(chunk),
                    "contact_sheet_path": rel(output_path),
                }
            )

    pd.DataFrame(summary_rows).to_csv(output_dir / "contact_sheet_summary.csv", index=False)
    print(f"Saved contact sheets to: {rel(output_dir)}")


if __name__ == "__main__":
    main()





