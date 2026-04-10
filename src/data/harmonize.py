"""
Harmonisation des datasets MVTec AD et HSS-IAD.

Scanne les deux datasets bruts et produit un DataFrame unifié sauvegardé
en CSV dans data/processed/unified_dataset.csv.

Usage:
    uv run python -m src.data.harmonize
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.config import DATA, PATHS


def scan_dataset(base_dir: Path, dataset_name: str) -> list[dict]:
    """Scan a dataset directory following the MVTec-style structure."""
    records = []

    for category_dir in sorted(base_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name

        for split in DATA.splits:
            split_dir = category_dir / split
            if not split_dir.exists():
                continue

            for label_dir in sorted(split_dir.iterdir()):
                if not label_dir.is_dir():
                    continue
                label = label_dir.name
                is_anomaly = label != DATA.normal_label

                for img_file in sorted(label_dir.iterdir()):
                    if img_file.suffix.lower() not in DATA.img_extensions:
                        continue

                    mask_path = _find_mask(category_dir, label, img_file) if is_anomaly else None

                    records.append({
                        "dataset": dataset_name,
                        "category": category,
                        "split": split,
                        "label": label,
                        "is_anomaly": is_anomaly,
                        "image_path": str(img_file.relative_to(PATHS.root)),
                        "mask_path": str(mask_path.relative_to(PATHS.root)) if mask_path else None,
                    })
    return records


def _find_mask(category_dir: Path, label: str, img_file: Path) -> Path | None:
    """Find the ground-truth mask corresponding to an anomalous image."""
    gt_dir = category_dir / DATA.ground_truth_dirname / label
    if not gt_dir.exists():
        # HSS-IAD: some categories use "ground_truth/defective" even for typed defects
        gt_dir = category_dir / DATA.ground_truth_dirname
        if not gt_dir.exists():
            return None

    stem = img_file.stem
    candidates = [
        gt_dir / f"{stem}_mask.png",
        gt_dir / f"{stem}_mask.PNG",
        gt_dir / f"{stem}.png",
        gt_dir / f"{stem}.PNG",
        gt_dir / f"{stem}.bmp",
        gt_dir / f"{stem}.jpg",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Fallback: search for file containing the stem
    for f in gt_dir.iterdir():
        if stem in f.stem and f.suffix.lower() in DATA.img_extensions:
            return f
    return None


def collect_resolutions(df: pd.DataFrame, sample_size: int | None = None) -> pd.DataFrame:
    """Sample images and collect width, height, channels."""
    if sample_size is None:
        sample_size = DATA.resolutions_sample_size

    np.random.seed(DATA.random_seed)
    n = min(sample_size, len(df))
    sample = df.sample(n=n, random_state=DATA.random_seed).copy()

    widths, heights, channels = [], [], []
    for path in sample["image_path"]:
        try:
            img = Image.open(PATHS.root / path)
            w, h = img.size
            c = len(img.getbands())
        except Exception:
            w, h, c = None, None, None
        widths.append(w)
        heights.append(h)
        channels.append(c)

    sample["width"] = widths
    sample["height"] = heights
    sample["channels"] = channels
    return sample


def harmonize() -> pd.DataFrame:
    """Main entry point: scan, merge, enrich, and save."""
    print(f"Scanning MVTec AD from {PATHS.mvtec_dir} ...")
    mvtec_records = scan_dataset(PATHS.mvtec_dir, DATA.mvtec_name)
    print(f"  -> {len(mvtec_records)} images")

    print(f"Scanning HSS-IAD from {PATHS.hssiad_dir} ...")
    hssiad_records = scan_dataset(PATHS.hssiad_dir, DATA.hssiad_name)
    print(f"  -> {len(hssiad_records)} images")

    df = pd.DataFrame(mvtec_records + hssiad_records)
    df["has_mask"] = df["mask_path"].notna()

    # Save
    PATHS.unified_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PATHS.unified_csv, index=False)
    print(f"\nUnified dataset saved to {PATHS.unified_csv}")
    print(f"Total: {len(df)} images | {df['category'].nunique()} categories | {df['dataset'].nunique()} datasets")

    # Also save resolution info
    print("\nCollecting image resolutions (sample) ...")
    df_res = collect_resolutions(df)
    df_res.to_csv(PATHS.resolutions_csv, index=False)
    print(f"Resolution sample saved to {PATHS.resolutions_csv}")

    return df


if __name__ == "__main__":
    harmonize()
