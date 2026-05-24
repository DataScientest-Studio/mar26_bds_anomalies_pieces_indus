"""Compare real Casting_class1 defects with current synthetic defect generation."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFont

from src.features.synthetic_defects.generation import DEFAULT_LABELS_DIR, FAMILIES, make_dataset
from src.config import PATHS
from src.models.baselines.patchcore import project_path


REAL_FAMILIES = ["machined_round", "scratch_like", "speckle", "blob_round", "irregular"]
COMPARE_FAMILIES = ["machined_round", "scratch_like", "speckle", "soft_stain", "empirical_residual", "mixed_hardening"]
FILENAME_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}_\d{3})(?P<sep>[-_])(?P<suffix>.+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--real-image-dir", type=Path, default=PATHS.root / "data/raw/hss-iad/Casting_class1/test/defective")
    parser.add_argument(
        "--real-mask-dir",
        type=Path,
        default=PATHS.root / "data/raw/hss-iad/Casting_class1/ground_truth/defective",
    )
    parser.add_argument(
        "--defect-library-json",
        type=Path,
        default=PATHS.root / "reports/tables/summary/casting_all_defect_patch_library.json",
    )
    parser.add_argument(
        "--texture-library-json",
        type=Path,
        default=(
            PATHS.root
            / "reports/casting_surface_features/defect_synthetic_study/clustered_texture_library_casting_all/clustered_defect_texture_library.json"
        ),
    )
    parser.add_argument(
        "--photometric-library-json",
        type=Path,
        default=PATHS.root / "reports/casting_surface_features/defect_synthetic_study/photometric_coherence_library.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.root / "reports/casting_surface_features/defect_synthetic_study/real_vs_synthetic_comparison_v3",
    )
    parser.add_argument("--samples-per-family", type=int, default=220)
    parser.add_argument("--seed", type=int, default=293)
    parser.add_argument("--min-component-area", type=int, default=8)
    parser.add_argument("--diff-threshold", type=int, default=2)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PATHS.root / path


def connected_components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_area):
            components.append(labels == label)
    return components


def perimeter(component: np.ndarray) -> float:
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def ring_mask(component: np.ndarray, iterations: int = 5) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(component.astype(np.uint8), kernel, iterations=int(iterations)).astype(bool)
    return dilated & ~component.astype(bool)


def edge_density(gray: np.ndarray, component: np.ndarray) -> float:
    ys, xs = np.where(component)
    if len(xs) < 3:
        return 0.0
    x0, x1 = max(0, xs.min() - 2), min(gray.shape[1], xs.max() + 3)
    y0, y1 = max(0, ys.min() - 2), min(gray.shape[0], ys.max() + 3)
    crop = gray[y0:y1, x0:x1]
    comp = component[y0:y1, x0:x1]
    edges = cv2.Canny(crop.astype(np.uint8), 40, 120) > 0
    return float((edges & comp).sum() / max(float(comp.sum()), 1.0))


def component_metrics(
    image: np.ndarray,
    component: np.ndarray,
    *,
    family: str,
    source: str,
    pattern_id: str,
    stem: str,
    sample_id: str | None = None,
) -> dict:
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    ys, xs = np.where(component)
    area = float(len(xs))
    height, width = component.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    peri = perimeter(component)
    circularity = float((4.0 * math.pi * area) / max(peri * peri, 1e-6))
    orientation_abs_deg = 0.0
    if len(xs) >= 3:
        coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
        cov = np.cov(coords, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        vec = eigvecs[:, int(np.argmax(eigvals))]
        angle = math.degrees(math.atan2(float(vec[1]), float(vec[0])))
        orientation_abs_deg = abs(((angle + 90.0) % 180.0) - 90.0)
    ring = ring_mask(component, iterations=5)
    fg = gray[component]
    bg = gray[ring] if ring.any() else gray.reshape(-1)
    return {
        "source": source,
        "family": family,
        "pattern_id": pattern_id,
        "stem": stem,
        "sample_id": sample_id or stem,
        "area": area,
        "area_frac": float(area / max(float(height * width), 1.0)),
        "diameter_256": float(2.0 * math.sqrt(area / math.pi) * 256.0 / max(float(min(height, width)), 1.0)),
        "bbox_w": bw,
        "bbox_h": bh,
        "aspect": float(max(bw, bh) / max(float(min(bw, bh)), 1.0)),
        "orientation_abs_deg": float(orientation_abs_deg),
        "circularity": circularity,
        "contrast_luma": float(bg.mean() - fg.mean()),
        "texture_std": float(fg.std()),
        "edge_density": edge_density(gray, component),
    }


def classify_real(component: np.ndarray) -> str:
    ys, xs = np.where(component)
    area = len(xs)
    if area <= 0:
        return "irregular"
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0
    aspect = max(bw, bh) / max(float(min(bw, bh)), 1.0)
    circ = (4.0 * math.pi * float(area)) / max(perimeter(component) ** 2, 1e-6)
    diameter_256 = 2.0 * math.sqrt(float(area) / math.pi) * 256.0 / max(float(min(component.shape)), 1.0)
    if area < 32 or diameter_256 < 3.5:
        return "speckle"
    if aspect >= 2.7:
        return "scratch_like"
    if circ >= 0.78 and aspect <= 1.55 and diameter_256 >= 4.0:
        return "machined_round"
    if circ >= 0.50 and aspect <= 2.1:
        return "blob_round"
    return "irregular"


def real_metrics(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    image_dir = resolve(args.real_image_dir)
    mask_dir = resolve(args.real_mask_dir)
    mask_paths = sorted(mask_dir.glob("*_mask.png"))
    pattern_by_stem = real_pattern_map(mask_paths)
    for mask_path in mask_paths:
        stem = mask_path.name.removesuffix("_mask.png")
        image_path = image_dir / f"{stem}.jpg"
        if not image_path.exists():
            continue
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
        pattern_id = pattern_by_stem.get(stem, pattern_from_stem(stem))
        for component in connected_components(mask, args.min_component_area):
            rows.append(
                component_metrics(
                    image,
                    component,
                    family=classify_real(component),
                    source="real",
                    pattern_id=pattern_id,
                    stem=stem,
                    sample_id=stem,
                )
            )
    return pd.DataFrame(rows)


def pattern_from_stem(stem: str) -> str:
    suffix = "_".join(stem.split("_")[-2:])
    if suffix == "1_2":
        return "P1"
    if suffix == "1_3":
        return "P2"
    if suffix == "2_3":
        return "P3/P4"
    return "UNKNOWN"


def parse_image_name(stem: str) -> dict[str, str]:
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


def c1_pattern(view_key: str, group_views: set[str]) -> str:
    if view_key == "1_2":
        return "P1"
    if view_key == "1_3":
        return "P2"
    if view_key == "2_3" and ("1_2" in group_views or "1_3" in group_views):
        return "P3"
    if view_key == "2_3":
        return "P4"
    return "UNKNOWN"


def real_pattern_map(mask_paths: list[Path]) -> dict[str, str]:
    parsed_by_stem = {}
    group_views: dict[str, set[str]] = defaultdict(set)
    for mask_path in mask_paths:
        stem = mask_path.name.removesuffix("_mask.png")
        parsed = parse_image_name(stem)
        parsed_by_stem[stem] = parsed
        group_views[str(parsed["acquisition_group"])].add(str(parsed["view_key"]))
    return {
        stem: c1_pattern(str(parsed["view_key"]), group_views[str(parsed["acquisition_group"])])
        for stem, parsed in parsed_by_stem.items()
    }


def sample_label_rows(labels: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    pattern_ids = [p for p in ["P1", "P2", "P3", "P4"] if (labels["pattern_id"].astype(str) == p).any()]
    per_pattern = max(1, int(math.ceil(n / max(len(pattern_ids), 1))))
    for pattern_id in pattern_ids:
        sub = labels[labels["pattern_id"].astype(str) == pattern_id]
        indices = rng.choice(sub.index.to_numpy(), size=per_pattern, replace=len(sub) < per_pattern)
        parts.append(labels.loc[indices])
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).head(n).reset_index(drop=True)
    return out


def synthetic_metrics(args: argparse.Namespace) -> pd.DataFrame:
    labels = pd.read_csv(resolve(args.labels_dir) / "labels_index.csv")
    rows = []
    family_by_name = {family["name"]: family for family in FAMILIES}
    selected_families = [family_by_name[name] for name in COMPARE_FAMILIES]
    for family_idx, family in enumerate(selected_families):
        sampled = sample_label_rows(labels, int(args.samples_per_family), int(args.seed) + family_idx * 997)
        for idx, row in sampled.iterrows():
            seed = int(args.seed) + family_idx * 100003 + idx
            random.seed(seed)
            np.random.seed(seed)
            dataset = make_dataset(row, family, args)
            image = Image.open(project_path(str(row["image_path"]))).convert("RGB")
            semantic = Image.open(project_path(str(row["semantic_mask_path"]))).convert("L").resize(
                image.size,
                Image.Resampling.NEAREST,
            )
            defect = dataset._apply_synthetic_defects(image, semantic, pattern_id=str(row["pattern_id"]))
            diff = np.asarray(ImageChops.difference(image, defect).convert("L"), dtype=np.uint8)
            diff_mask = diff > int(args.diff_threshold)
            image_arr = np.asarray(defect.convert("RGB"), dtype=np.uint8)
            stem = Path(str(row["image_path"])).stem
            sample_id = f"{family['name']}_{idx:04d}_{stem}"
            for component in connected_components(diff_mask, args.min_component_area):
                rows.append(
                    component_metrics(
                        image_arr,
                        component,
                        family=str(family["name"]),
                        source="synthetic",
                        pattern_id=str(row["pattern_id"]),
                        stem=stem,
                        sample_id=sample_id,
                    )
                )
    return pd.DataFrame(rows)


def summarize(real: pd.DataFrame, synth: pd.DataFrame, *, by_pattern: bool = False) -> pd.DataFrame:
    metrics = [
        "area_frac",
        "diameter_256",
        "aspect",
        "orientation_abs_deg",
        "circularity",
        "contrast_luma",
        "texture_std",
        "edge_density",
    ]
    rows = []
    comparison_map = {
        "machined_round": "machined_round",
        "scratch_like": "scratch_like",
        "speckle": "speckle",
        "soft_stain": "irregular",
        "empirical_residual": "blob_round",
        "mixed_hardening": "machined_round",
    }
    pattern_values = ["ALL"]
    if by_pattern:
        pattern_values = [pattern for pattern in ["P1", "P2", "P3", "P4"] if pattern in set(real["pattern_id"]) or pattern in set(synth["pattern_id"])]
    for pattern_id in pattern_values:
        real_scope = real if pattern_id == "ALL" else real[real["pattern_id"] == pattern_id]
        synth_scope = synth if pattern_id == "ALL" else synth[synth["pattern_id"] == pattern_id]
        for synth_family, real_family in comparison_map.items():
            real_sub = real_scope[real_scope["family"] == real_family]
            synth_sub = synth_scope[synth_scope["family"] == synth_family]
            if real_sub.empty or synth_sub.empty:
                continue
            row = {
                "pattern_id": pattern_id,
                "family": synth_family,
                "real_reference_family": real_family,
                "real_n": len(real_sub),
                "synthetic_n": len(synth_sub),
            }
            real_counts = real_sub.groupby("sample_id").size()
            synth_counts = synth_sub.groupby("sample_id").size()
            row["real_components_per_image_p50"] = float(real_counts.median())
            row["syn_components_per_image_p50"] = float(synth_counts.median())
            row["syn_over_real_components_per_image_p50"] = (
                float(synth_counts.median() / real_counts.median()) if float(real_counts.median()) > 0 else np.nan
            )
            for metric in metrics:
                real_p50 = float(real_sub[metric].median())
                synth_p50 = float(synth_sub[metric].median())
                denom = real_p50 if abs(real_p50) > 1e-9 else np.nan
                row[f"real_{metric}_p50"] = real_p50
                row[f"syn_{metric}_p50"] = synth_p50
                row[f"syn_over_real_{metric}_p50"] = float(synth_p50 / denom) if np.isfinite(denom) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def write_diagnostic(summary: pd.DataFrame, output_path: Path) -> None:
    lines = ["# Diagnostic real vs synthetic v3", ""]
    summary = summary.copy()
    if "pattern_id" not in summary.columns:
        summary["pattern_id"] = "ALL"
    for pattern_id, group in summary.groupby("pattern_id", sort=False):
        title = "Global" if pattern_id == "ALL" else f"Pattern {pattern_id}"
        lines += [f"# {title}", ""]
        for _, row in group.iterrows():
            lines += [f"## {row['family']} vs real {row['real_reference_family']}", ""]
            lines.append(
                f"- components_per_image: real p50={row['real_components_per_image_p50']:.4g}, "
                f"synth p50={row['syn_components_per_image_p50']:.4g}, "
                f"ratio={row['syn_over_real_components_per_image_p50']:.2f}"
            )
            for metric in [
                "area_frac",
                "diameter_256",
                "aspect",
                "orientation_abs_deg",
                "circularity",
                "contrast_luma",
                "texture_std",
                "edge_density",
            ]:
                lines.append(
                    f"- {metric}: real p50={row[f'real_{metric}_p50']:.4g}, "
                    f"synth p50={row[f'syn_{metric}_p50']:.4g}, "
                    f"ratio={row[f'syn_over_real_{metric}_p50']:.2f}"
                )
            lines.append("")
    lines += [
        "## Lecture rapide",
        "",
        "- Un ratio proche de 1 sur `diameter_256` et `area_frac` indique une taille coherente.",
        "- Un ratio de contraste superieur a 1 en valeur absolue indique un rendu encore trop visible.",
        "- `edge_density` reste le garde-fou contre les bords synthetiques trop nets.",
        "- `mixed_hardening` n'est pas cense matcher une famille unique : il sert surtout a auditer le regime d'entrainement combine.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def make_boxplots(metrics: pd.DataFrame, output_path: Path) -> None:
    plot_metrics = ["diameter_256", "aspect", "orientation_abs_deg", "circularity", "contrast_luma", "texture_std", "edge_density"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, metric in zip(axes.ravel(), plot_metrics):
        data = []
        labels = []
        for label, sub in metrics.groupby(["source", "family"]):
            if label[1] not in {"machined_round", "scratch_like", "speckle", "soft_stain", "empirical_residual", "mixed_hardening"}:
                continue
            data.append(sub[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy())
            labels.append(f"{label[0][0]}:{label[1][:8]}")
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=60)
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_atlas(metrics: pd.DataFrame, output_path: Path) -> None:
    # Lightweight atlas: use metric table only, shown as a heatmap-style summary.
    families = [family for family in COMPARE_FAMILIES if family in set(metrics["family"])]
    fig, ax = plt.subplots(figsize=(12, 4))
    table = metrics[metrics["family"].isin(families)].groupby(["source", "family"]).agg(
        n=("area", "size"),
        diameter=("diameter_256", "median"),
        aspect=("aspect", "median"),
        contrast=("contrast_luma", "median"),
        texture=("texture_std", "median"),
        edge=("edge_density", "median"),
    )
    ax.axis("off")
    ax.table(
        cellText=np.round(table.reset_index().drop(columns=["source", "family"]).to_numpy(), 3),
        rowLabels=[f"{idx[0]}:{idx[1]}" for idx in table.index],
        colLabels=list(table.columns),
        loc="center",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def make_pattern_family_counts(metrics: pd.DataFrame, output_path: Path) -> None:
    counts = (
        metrics.groupby(["source", "pattern_id", "family"], dropna=False)
        .size()
        .reset_index(name="components")
        .sort_values(["source", "pattern_id", "family"])
    )
    counts.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    real = real_metrics(args)
    synth = synthetic_metrics(args)
    metrics = pd.concat([real, synth], ignore_index=True)
    summary = summarize(real, synth)
    pattern_summary = summarize(real, synth, by_pattern=True)
    full_summary = pd.concat([summary, pattern_summary], ignore_index=True)

    metrics.to_csv(output_dir / "real_vs_synthetic_component_metrics.csv", index=False)
    full_summary.to_csv(output_dir / "real_vs_synthetic_summary.csv", index=False)
    pattern_summary.to_csv(output_dir / "real_vs_synthetic_summary_by_pattern.csv", index=False)
    make_pattern_family_counts(metrics, output_dir / "real_vs_synthetic_family_counts_by_pattern.csv")
    make_boxplots(metrics, output_dir / "real_vs_synthetic_metric_boxplots.png")
    make_atlas(metrics, output_dir / "real_vs_synthetic_metric_atlas.png")
    write_diagnostic(full_summary, output_dir / "diagnostic_real_vs_synthetic.md")
    (output_dir / "params.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "real_n": int(len(real)), "synthetic_n": int(len(synth))}, indent=2))


if __name__ == "__main__":
    main()






