"""Live inference — drag-drop image ou galerie d'exemples MVTec.

Deux modes :
- Single : un seul modèle, affichage détaillé.
- Compare : 4 modèles côte-à-côte (Dinomaly / PatchCore / Mean / Max).

Run :
    uv run streamlit run app/main.py
    → navigation vers "Live inference"
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import io
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

from src.config import PATHS
from src.inference.pipeline import AnomalyPipeline

st.set_page_config(page_title="Live inference", page_icon="🔍", layout="wide")

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

MODEL_OPTIONS = {
    "Ensemble Mean": "ensemble_mean",
    "Ensemble Max": "ensemble_max",
    "Dinomaly seul": "dinomaly",
    "PatchCore seul": "patchcore",
}


# --- Caches ----------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement du modèle…")
def get_pipeline(category: str, model_name: str) -> AnomalyPipeline:
    return AnomalyPipeline.from_category(category=category, model_name=model_name)


@st.cache_data
def list_test_examples(category: str, max_per_class: int = 4) -> dict:
    test_dir = PATHS.mvtec_dir / category / "test"
    if not test_dir.exists():
        return {}
    examples = {}
    for subdir in sorted(test_dir.iterdir()):
        if not subdir.is_dir():
            continue
        imgs = sorted(subdir.glob("*.png"))[:max_per_class]
        if imgs:
            examples[subdir.name] = [str(p) for p in imgs]
    return examples


def blend_overlay(image_resized: np.ndarray, overlay_rgba: np.ndarray, alpha: int = 160) -> Image.Image:
    base = Image.fromarray(image_resized).convert("RGBA")
    overlay = Image.fromarray(overlay_rgba).convert("RGBA")
    overlay.putalpha(alpha)
    return Image.alpha_composite(base, overlay)


def load_gt_mask(test_path: str | Path, target_shape: tuple[int, int]) -> Optional[np.ndarray]:
    """Charge le ground truth mask pour une image test MVTec.

    Convention : test/<defect_type>/<id>.png → ground_truth/<defect_type>/<id>_mask.png.
    Retourne None pour les images good ou si le mask n'existe pas.

    target_shape : (H, W) à laquelle resize le mask.
    """
    p = Path(test_path)
    if p.parent.name == "good":
        return np.zeros(target_shape, dtype=np.uint8)
    cat_dir = p.parent.parent.parent  # remonte à category/
    defect = p.parent.name
    mask_path = cat_dir / "ground_truth" / defect / f"{p.stem}_mask.png"
    if not mask_path.exists():
        return None
    mask = Image.open(mask_path).convert("L").resize(target_shape[::-1], Image.NEAREST)
    return (np.array(mask) > 0).astype(np.uint8) * 255


def mask_to_rgba(mask: np.ndarray, color: tuple = (255, 0, 0)) -> np.ndarray:
    """Convertit un mask binaire (H, W) en RGBA (H, W, 4) uint8 avec couleur + alpha."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    is_defect = mask > 0
    rgba[is_defect, 0] = color[0]
    rgba[is_defect, 1] = color[1]
    rgba[is_defect, 2] = color[2]
    rgba[is_defect, 3] = 200  # alpha pour mask défaut
    return rgba


@st.cache_data
def load_benchmark_csv() -> Optional["pd.DataFrame"]:
    """Charge les métriques benchmark du notebook 08 (AUROC img/pix + AUPIMO pour 15 cat)."""
    import pandas as pd
    csv_path = PATHS.root / "reports" / "mvtec_dinomaly_patchcore_ensemble.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


# Mapping interne model_name → préfixe CSV
MODEL_TO_PREFIX = {
    "dinomaly": "dino",
    "patchcore": "pc",
    "ensemble_mean": "mean",
    "ensemble_max": "max",
}


def per_image_auroc_pixel(heatmap: np.ndarray, gt_mask: np.ndarray) -> Optional[float]:
    """AUROC pixel calculé sur UNE image avec son GT mask.

    Mesure la qualité de localisation : à quel point la heatmap range les pixels défaut
    plus haut que les pixels normaux, au sein de cette image.

    Retourne None si l'image n'a pas de défaut (mask vide).
    """
    from sklearn.metrics import roc_auc_score
    gt_flat = (gt_mask > 0).astype(int).flatten()
    if gt_flat.sum() == 0 or gt_flat.sum() == len(gt_flat):
        return None  # 1 seule classe → AUROC indéfini
    # Resize heatmap si shape ≠ gt_mask shape
    if heatmap.shape != gt_mask.shape:
        from skimage.transform import resize as sk_resize
        heatmap = sk_resize(
            heatmap, gt_mask.shape, order=1, preserve_range=True, anti_aliasing=True
        )
    return float(roc_auc_score(gt_flat, heatmap.flatten()))


# --- UI -------------------------------------------------------------------
st.title("🔍 Inférence en direct")

with st.sidebar:
    st.header("Configuration")
    category = st.selectbox("Catégorie", MVTEC_CATEGORIES, index=1)
    mode = st.radio("Mode", ["Single model", "Compare 4 models"], index=1)
    if mode == "Single model":
        model_display = st.radio("Modèle", list(MODEL_OPTIONS.keys()), index=0)
        model_name = MODEL_OPTIONS[model_display]
    else:
        model_display = "Compare"
        model_name = None
    st.markdown("---")
    st.markdown("### Source de l'image")
    source = st.radio(
        "", ["📁 Upload", "🖼 Galerie d'exemples"], label_visibility="collapsed"
    )

# --- Vérification ckpt ----------------------------------------------------
ckpt_dir = PATHS.root / "models"
needs_dino = mode == "Compare 4 models" or model_name in ("dinomaly", "ensemble_mean", "ensemble_max")
has_dino_ckpt = ckpt_dir.exists() and any(ckpt_dir.glob(f"dinomaly_{category}*ckpt"))

if needs_dino and not has_dino_ckpt:
    st.error(
        f"⚠ Checkpoint Dinomaly absent pour `{category}`.\n\n"
        f"→ Entraîne : `uv run python scripts/train_all.py --cats {category}`\n"
        f"→ Ou bascule en mode **Single model → PatchCore seul**."
    )
    st.stop()

test_dir = PATHS.mvtec_dir / category / "test"
if not test_dir.exists():
    st.error(
        f"⚠ Dossier test MVTec absent : `{test_dir}`.\n\n"
        f"Run : `uv run python scripts/download_data.py --mvtec`"
    )
    st.stop()

# --- Sélection de l'image -------------------------------------------------
selected_image: Optional[Image.Image] = None
selected_path: Optional[str] = None

if source == "📁 Upload":
    uploaded = st.file_uploader(
        f"Image à analyser (catégorie : `{category}`)",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded is not None:
        selected_image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
else:
    examples = list_test_examples(category, max_per_class=4)
    if not examples:
        st.warning("Aucun exemple trouvé pour cette catégorie.")
    else:
        with st.expander("Galerie d'exemples MVTec (cliquer pour analyser)", expanded=True):
            for defect_type, paths in examples.items():
                st.markdown(f"**{defect_type}**")
                cols = st.columns(min(len(paths), 4))
                for i, p in enumerate(paths):
                    col = cols[i % len(cols)]
                    with col:
                        col.image(Image.open(p), use_container_width=True)
                        if col.button(f"Analyser", key=f"btn_{defect_type}_{i}"):
                            selected_path = p
                            selected_image = Image.open(p).convert("RGB")

# --- Inférence ------------------------------------------------------------
if selected_image is None:
    st.info("👆 Sélectionne une image pour lancer l'inférence.")
    st.stop()

st.markdown("---")

# --- Mode 1 : Single model ------------------------------------------------
if mode == "Single model":
    st.subheader(f"Résultat — {model_display}")
    pipe = get_pipeline(category=category, model_name=model_name)
    with st.spinner("Inférence…"):
        result = pipe.predict(selected_image)

    # Tente de charger le GT mask si l'image vient de la galerie
    gt_mask = None
    if selected_path is not None:
        gt_mask = load_gt_mask(selected_path, result.heatmap.shape)

    n_cols = 5 if gt_mask is not None else 3
    cols = st.columns(n_cols)
    cols[0].image(result.image_resized, caption="Image", use_container_width=True)
    if gt_mask is not None:
        cols[1].image(gt_mask, caption="GT mask", use_container_width=True)
        # Overlay GT (red) sur l'image
        gt_rgba = mask_to_rgba(gt_mask)
        gt_overlay = blend_overlay(result.image_resized, gt_rgba, alpha=200)
        cols[2].image(gt_overlay, caption="GT overlay", use_container_width=True)
        cols[3].image(result.overlay, caption="Heatmap pred", use_container_width=True)
        cols[4].image(
            blend_overlay(result.image_resized, result.overlay),
            caption="Pred overlay", use_container_width=True,
        )
    else:
        cols[1].image(result.overlay, caption="Heatmap pred", use_container_width=True)
        cols[2].image(
            blend_overlay(result.image_resized, result.overlay),
            caption="Pred overlay", use_container_width=True,
        )

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric(
        "Score anomalie [0, 1]", f"{result.score:.4f}",
        help="Normalisé contre 20 images train good. 0 = image ressemble au pire good vu pendant le fit. 1 = bien plus anomal que tout le train good.",
    )
    sc2.metric("Modèle", model_display)
    sc3.metric("Catégorie", category)

    # --- AUROC pixel sur CETTE image (si GT dispo) ---
    if gt_mask is not None and gt_mask.sum() > 0:
        auroc_pix_img = per_image_auroc_pixel(result.heatmap, gt_mask)
        if auroc_pix_img is not None:
            st.markdown("##### Métrique sur cette image (vs ground truth)")
            st.metric(
                "AUROC pixel (image)",
                f"{auroc_pix_img:.4f}",
                help="Mesure de localisation : à quel point la heatmap classe les pixels défaut plus haut que les normaux, au sein de cette seule image.",
            )

    # --- Métriques benchmark dataset-level (référence) ---
    bench = load_benchmark_csv()
    if bench is not None and category in bench["category"].values:
        st.markdown("##### Benchmark de référence (15 catégories, notebook 08)")
        row = bench[bench["category"] == category].iloc[0]
        prefix = MODEL_TO_PREFIX[model_name]
        b1, b2, b3 = st.columns(3)
        b1.metric("AUROC image", f"{row[f'{prefix}_img']:.4f}",
                  help="Calculé sur l'ensemble du test set MVTec de cette catégorie.")
        b2.metric("AUROC pixel", f"{row[f'{prefix}_pix']:.4f}",
                  help="Sur tous les pixels défaut/normal du test set.")
        b3.metric("AUPIMO", f"{row[f'{prefix}_aupimo']:.4f}",
                  help="Per-Image Overlap au FPR ultra-bas [1e-5, 1e-4].")

    st.caption(
        "ℹ️ **Score anomalie [0, 1]** : normalisé contre 20 train good au chargement du modèle. "
        "Une image good est proche de 0, une defective bien au-dessus. "
        "**AUROC/AUPIMO benchmark** = métriques pré-calculées sur le test set MVTec (notebook 08). "
        "**AUROC pixel sur cette image** = score local de localisation (vs GT mask)."
    )

# --- Mode 2 : Compare 4 modèles -------------------------------------------
else:
    st.subheader(f"Comparaison 4 modèles — `{category}`")

    pipes = {}
    results = {}
    for label, mn in MODEL_OPTIONS.items():
        with st.spinner(f"Inférence {label}…"):
            pipes[label] = get_pipeline(category=category, model_name=mn)
            results[label] = pipes[label].predict(selected_image)

    # Tente de charger le GT mask
    ref_result = results["Ensemble Mean"]
    gt_mask = None
    if selected_path is not None:
        gt_mask = load_gt_mask(selected_path, ref_result.heatmap.shape)

    # Image originale + GT mask (si dispo) en haut
    if gt_mask is not None:
        top_cols = st.columns(3)
        top_cols[0].image(ref_result.image_resized, caption=f"Image — {category}",
                          use_container_width=True)
        top_cols[1].image(gt_mask, caption="GT mask", use_container_width=True)
        gt_rgba = mask_to_rgba(gt_mask)
        gt_overlay = blend_overlay(ref_result.image_resized, gt_rgba, alpha=200)
        top_cols[2].image(gt_overlay, caption="GT overlay (rouge)",
                          use_container_width=True)
    else:
        img_col, _ = st.columns([1, 3])
        img_col.image(
            ref_result.image_resized,
            caption=f"Image — {category}",
            use_container_width=True,
        )

    # 4 colonnes : 1 par modèle
    st.markdown("#### Heatmaps par modèle")
    cols = st.columns(4)
    per_image_aurocs = {}
    for col, (label, result) in zip(cols, results.items()):
        col.markdown(f"**{label}**")
        col.image(
            blend_overlay(result.image_resized, result.overlay),
            use_container_width=True,
        )
        col.metric("Score", f"{result.score:.4f}")
        # AUROC pixel per-image si GT disponible
        if gt_mask is not None and gt_mask.sum() > 0:
            auroc = per_image_auroc_pixel(result.heatmap, gt_mask)
            if auroc is not None:
                per_image_aurocs[label] = auroc
                col.metric("AUROC pix (image)", f"{auroc:.4f}")

    # --- Tableau benchmark 4 modèles pour cette catégorie ---
    bench = load_benchmark_csv()
    if bench is not None and category in bench["category"].values:
        st.markdown("#### Benchmark dataset-level (test set complet)")
        row = bench[bench["category"] == category].iloc[0]
        import pandas as pd
        rows_table = []
        for label, mn in MODEL_OPTIONS.items():
            prefix = MODEL_TO_PREFIX[mn]
            rows_table.append({
                "Modèle": label,
                "AUROC image": round(row[f"{prefix}_img"], 4),
                "AUROC pixel": round(row[f"{prefix}_pix"], 4),
                "AUPIMO": round(row[f"{prefix}_aupimo"], 4),
            })
        bench_df = pd.DataFrame(rows_table).set_index("Modèle")
        st.dataframe(
            bench_df.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1).format(precision=4),
            use_container_width=True,
        )

    st.caption(
        "ℹ️ **Score [0, 1]** : normalisé contre 20 train good. Désormais comparable entre modèles. "
        "**AUROC pix (image)** = qualité de localisation sur cette image (si GT dispo). "
        "**Benchmark dataset-level** = métriques pré-calculées sur l'ensemble du test set MVTec (notebook 08)."
    )

    # Barchart des scores
    import altair as alt
    import pandas as pd
    df_scores = pd.DataFrame(
        [{"modèle": k, "score": v.score} for k, v in results.items()]
    )
    chart = (
        alt.Chart(df_scores)
        .mark_bar()
        .encode(
            x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("modèle:N", sort="-x"),
            color=alt.Color("modèle:N", legend=None),
            tooltip=["modèle", "score"],
        )
        .properties(width=600, height=180)
    )
    st.markdown("#### Scores image-level (normalisés [0, 1] per-image)")
    st.altair_chart(chart, use_container_width=True)

# --- Info ground truth si applicable --------------------------------------
if selected_path:
    defect = Path(selected_path).parent.name
    if defect == "good":
        st.success(f"✓ Image **good** (ground truth) — score idéalement bas.")
    else:
        st.info(f"⚠ Image **défective** ({defect}) — score idéalement haut.")
