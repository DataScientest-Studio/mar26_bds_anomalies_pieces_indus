"""Calibre les seuils de décision per-(catégorie, modèle) sur le test set MVTec.

Pour chaque catégorie :
  1. Build une pipeline `ensemble_mean` (charge Dinomaly + PatchCore, calibre sur train good).
  2. Score tout le test set (good + tous les types de défauts) via `predict_all_modes`
     -> 1 forward Dinomaly + 1 forward PatchCore par image, 4 scores extraits.
  3. Pour chacun des 4 modes (dinomaly, patchcore, ensemble_mean, ensemble_max) :
     - AUROC image-level (rank-based, indépendant du seuil)
     - threshold_good   = p90 des scores good     -> "bon si score ≤ threshold_good"
     - threshold_defect = p10 des scores défaut    -> "défectueux si score ≥ threshold_defect"
  4. Sauvegarde dans `reports/calibration/mvtec_thresholds.csv`.

Le CSV est ensuite chargé automatiquement par la page Streamlit `1_Live_inference`,
qui pré-remplit les sliders de décision avec les seuils calibrés par catégorie.

Usage :
    uv run --extra cu128 python scripts/calibrate_thresholds.py
    uv run --extra cu128 python scripts/calibrate_thresholds.py --categories cable bottle
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Forcer UTF-8 sur la console Windows (cp1252 par défaut) pour les chars unicode.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS  # noqa: E402
from src.inference.pipeline import AnomalyPipeline  # noqa: E402

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

MODELS = ["dinomaly", "patchcore", "ensemble_mean", "ensemble_max"]


def collect_test_images(category: str) -> tuple[list[Path], list[Path]]:
    """Retourne (good_paths, defective_paths) du test set MVTec de la catégorie."""
    test_dir = PATHS.mvtec_dir / category / "test"
    if not test_dir.exists():
        return [], []
    good, defective = [], []
    for subdir in sorted(test_dir.iterdir()):
        if not subdir.is_dir():
            continue
        imgs = sorted(subdir.glob("*.png")) + sorted(subdir.glob("*.jpg"))
        if subdir.name == "good":
            good.extend(imgs)
        else:
            defective.extend(imgs)
    return good, defective


def calibrate_category(category: str, n_calib: int) -> list[dict]:
    """Inference sur le test set de la catégorie. Retourne 1 dict par modèle."""
    print(f"\n=== {category} ===")
    good_paths, defective_paths = collect_test_images(category)
    if not good_paths or not defective_paths:
        print(f"  ⚠ Test data incomplet (good={len(good_paths)}, def={len(defective_paths)})")
        return []
    print(f"  test set : {len(good_paths)} good + {len(defective_paths)} défectueux")

    t0 = time.time()
    pipe = AnomalyPipeline(
        category=category, model_name="ensemble_mean", n_calib_for_stats=n_calib,
    )
    print(f"  pipeline construite en {time.time() - t0:.1f}s")

    scores_good = {m: [] for m in MODELS}
    scores_def = {m: [] for m in MODELS}

    t0 = time.time()
    for p in good_paths:
        res = pipe.predict_all_modes(Image.open(p).convert("RGB"))
        for m in MODELS:
            scores_good[m].append(res[m].score)
    for p in defective_paths:
        res = pipe.predict_all_modes(Image.open(p).convert("RGB"))
        for m in MODELS:
            scores_def[m].append(res[m].score)
    print(f"  inférence en {time.time() - t0:.1f}s")

    rows = []
    for m in MODELS:
        s_good = np.array(scores_good[m])
        s_def = np.array(scores_def[m])

        y_true = np.concatenate([np.zeros(len(s_good)), np.ones(len(s_def))])
        y_score = np.concatenate([s_good, s_def])
        auroc = float(roc_auc_score(y_true, y_score))

        th_good = float(np.percentile(s_good, 90))
        th_def = float(np.percentile(s_def, 10))

        # Garantir une zone "à vérifier" non vide : si overlap, on écarte légèrement.
        if th_def <= th_good:
            mid = (th_good + th_def) / 2
            th_good = max(0.0, mid - 0.05)
            th_def = min(1.0, mid + 0.05)

        print(f"  {m:15s} AUROC={auroc:.4f}  bon≤{th_good:.3f}  def≥{th_def:.3f}")
        rows.append({
            "category": category,
            "model": m,
            "threshold_good": round(th_good, 4),
            "threshold_defect": round(th_def, 4),
            "auroc": round(auroc, 4),
            "n_good": len(s_good),
            "n_defective": len(s_def),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--categories", nargs="+", default=MVTEC_CATEGORIES,
                        help="Catégories MVTec à calibrer (défaut : 15)")
    parser.add_argument("--n-calib", type=int, default=100,
                        help="Nombre d'images train good pour la calibration pipeline (défaut: 100)")
    parser.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "reports" / "calibration" / "mvtec_thresholds.csv",
        help="Chemin du CSV de sortie",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"# Calibration {len(args.categories)} catégories × {len(MODELS)} modèles")
    print(f"# n_calib_for_stats = {args.n_calib}")
    print(f"# Sortie : {args.out}")

    t_start = time.time()
    all_rows = []
    for cat in args.categories:
        try:
            all_rows.extend(calibrate_category(cat, args.n_calib))
        except FileNotFoundError as e:
            print(f"  ⚠ {cat} ignoré : {e}")
            continue
        except Exception as e:
            print(f"  ⚠ {cat} erreur : {e}")
            continue

    if not all_rows:
        print("\n⚠ Aucune ligne calibrée — vérifier que les checkpoints + data sont présents.")
        return

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✓ {len(all_rows)} lignes écrites dans {args.out}")
    print(f"✓ Temps total : {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
