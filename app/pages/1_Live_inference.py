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


def defect_only_overlay(
    image_resized: np.ndarray,
    heatmap_norm: np.ndarray,
    threshold: float = 0.4,
    max_alpha: int = 210,
) -> Image.Image:
    """Superpose UNIQUEMENT les régions de défaut sur l'image.

    Les pixels dont `heatmap_norm` < `threshold` restent totalement transparents
    (image originale visible tel quel). Au-delà du seuil, l'opacité monte
    linéairement avec l'intensité jusqu'à `max_alpha`.
    """
    import matplotlib.cm as cm
    cmap = cm.get_cmap("jet")
    rgba = (cmap(np.clip(heatmap_norm, 0, 1)) * 255).astype(np.uint8)
    above = np.clip(heatmap_norm - threshold, 0.0, 1.0) / max(1.0 - threshold, 1e-6)
    rgba[..., 3] = (above * max_alpha).astype(np.uint8)
    base = Image.fromarray(image_resized).convert("RGBA")
    return Image.alpha_composite(base, Image.fromarray(rgba))


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
    csv_path = PATHS.root / "reports" / "ensemble" / "mvtec_dinomaly_patchcore_ensemble.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


@st.cache_data
def load_calibrated_thresholds() -> dict[tuple[str, str], tuple[float, float, float]]:
    """Charge les seuils calibrés sur test set MVTec → {(category, model): (good, defect, auroc)}.

    CSV produit par `scripts/calibrate_thresholds.py`.
    """
    csv_path = PATHS.root / "reports" / "calibration" / "mvtec_thresholds.csv"
    if not csv_path.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(csv_path)
    return {
        (row["category"], row["model"]): (
            float(row["threshold_good"]),
            float(row["threshold_defect"]),
            float(row["auroc"]),
        )
        for _, row in df.iterrows()
    }


def get_default_thresholds(category: str, model_name: str) -> tuple[float, float, Optional[float]]:
    """Retourne (good_default, defect_default, auroc) — auroc=None si non calibré."""
    cal = load_calibrated_thresholds()
    if (category, model_name) in cal:
        g, d, auroc = cal[(category, model_name)]
        # Clip aux ranges des sliders ci-dessous
        return max(0.05, min(0.50, g)), max(0.40, min(0.95, d)), auroc
    return 0.30, 0.60, None


# Mapping interne model_name → préfixe CSV
MODEL_TO_PREFIX = {
    "dinomaly": "dino",
    "patchcore": "pc",
    "ensemble_mean": "mean",
    "ensemble_max": "max",
}


def classify_anomaly_score(
    score: float, good_threshold: float, defect_threshold: float
) -> tuple[str, str, str]:
    """3 classes inspirées de la page Casting_class1 : bonne / à vérifier / défectueuse."""
    if score <= good_threshold:
        confidence = min(1.0, (good_threshold - score) / max(good_threshold, 1e-6))
        return "Pièce bonne", "#17823b", f"Confiance bonne {100 * confidence:.0f}%"
    if score >= defect_threshold:
        confidence = min(1.0, (score - defect_threshold) / max(1.0 - defect_threshold, 1e-6))
        return "Pièce défectueuse", "#b42318", f"Confiance défaut {100 * confidence:.0f}%"
    span = max(defect_threshold - good_threshold, 1e-6)
    center = (good_threshold + defect_threshold) / 2.0
    ambiguity = 1.0 - min(1.0, abs(score - center) / (span / 2.0))
    return "À vérifier", "#b7791f", f"Zone mitigée {100 * ambiguity:.0f}%"


def render_decision_badge(score: float, good_threshold: float, defect_threshold: float) -> None:
    label, color, conf = classify_anomaly_score(score, good_threshold, defect_threshold)
    st.markdown(
        f"""
        <div style="border-left: 10px solid {color}; background: {color}18;
                    padding: 0.85rem 1rem; border-radius: 0.35rem; margin: 0.5rem 0 0.75rem 0;">
            <div style="font-size: 1.4rem; font-weight: 750; color: {color};">{label}</div>
            <div style="font-size: 0.95rem; color: #2f3b4a;">
                Score {score:.4f} · bon ≤ {good_threshold:.2f} · défectueux ≥ {defect_threshold:.2f} · {conf}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    # Seuils pré-remplis depuis la calibration test set MVTec (si disponible).
    th_model = model_name if mode == "Single model" else "ensemble_mean"
    default_good, default_defect, auroc_calib = get_default_thresholds(category, th_model)
    with st.expander("Décision anomalie", expanded=True):
        if auroc_calib is not None:
            st.caption(
                f"✓ Seuils calibrés sur test MVTec — `{th_model}` AUROC={auroc_calib:.3f} "
                f"(p90 good / p10 défectueux)."
            )
        else:
            st.caption(
                "Seuils par défaut (non calibrés pour cette catégorie). "
                "Lancer `scripts/calibrate_thresholds.py` pour activer la calibration."
            )
        good_threshold = st.slider(
            "Bon si score ≤", 0.05, 0.50, default_good, 0.01,
            key=f"good_thr_{category}_{th_model}",
        )
        defect_threshold = st.slider(
            "Défectueux si score ≥", 0.40, 0.95, default_defect, 0.01,
            key=f"defect_thr_{category}_{th_model}",
        )
        if defect_threshold <= good_threshold:
            st.warning("Le seuil défaut doit être strictement supérieur au seuil bon.")
            defect_threshold = good_threshold + 0.01

    st.markdown("---")
    st.markdown("### Source de l'image")
    source = st.radio(
        "Source de l'image", ["📁 Upload", "🖼 Galerie d'exemples"], label_visibility="collapsed"
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
                        col.image(Image.open(p), width="stretch")
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

    n_cols = 3 if gt_mask is not None else 2
    cols = st.columns(n_cols)
    cols[0].image(result.image_resized, caption="Image", width="stretch")
    if gt_mask is not None:
        gt_rgba = mask_to_rgba(gt_mask)
        gt_overlay = blend_overlay(result.image_resized, gt_rgba, alpha=200)
        cols[1].image(gt_overlay, caption="GT overlay", width="stretch")
        cols[2].image(
            defect_only_overlay(result.image_resized, result.heatmap_norm),
            caption="Pred overlay", width="stretch",
        )
    else:
        cols[1].image(
            defect_only_overlay(result.image_resized, result.heatmap_norm),
            caption="Pred overlay", width="stretch",
        )

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric(
        "Score anomalie [0, 1]", f"{result.score:.4f}",
        help="Max après lissage médian 3×3, normalisé contre 100 train good. 0 = image ressemble au pire good vu pendant le fit. 1 = bien plus anomal que tout le train good.",
    )
    sc2.metric("Modèle", model_display)
    sc3.metric("Catégorie", category)

    render_decision_badge(result.score, good_threshold, defect_threshold)

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
        "ℹ️ **Score anomalie [0, 1]** : normalisé contre 100 train good au chargement du modèle. "
        "Une image good est proche de 0, une defective bien au-dessus. "
        "**AUROC/AUPIMO benchmark** = métriques pré-calculées sur le test set MVTec (notebook 08). "
        "**AUROC pixel sur cette image** = score local de localisation (vs GT mask)."
    )

# --- Mode 2 : Compare 4 modèles -------------------------------------------
else:
    st.subheader(f"Comparaison 4 modèles — `{category}`")

    # Une seule pipeline (ensemble_mean charge Dinomaly + PatchCore), puis
    # predict_all_modes réutilise les heatmaps pour produire les 4 résultats.
    # ~3× plus rapide que 4 appels predict() séparés.
    pipe = get_pipeline(category=category, model_name="ensemble_mean")
    with st.spinner("Inférence 4 modèles…"):
        raw_results = pipe.predict_all_modes(selected_image)
    results = {label: raw_results[mn] for label, mn in MODEL_OPTIONS.items()}

    # Tente de charger le GT mask
    ref_result = results["Ensemble Mean"]
    gt_mask = None
    if selected_path is not None:
        gt_mask = load_gt_mask(selected_path, ref_result.heatmap.shape)

    # Image originale + GT overlay (si dispo) en haut
    if gt_mask is not None:
        top_cols = st.columns(2)
        top_cols[0].image(ref_result.image_resized, caption=f"Image — {category}",
                          width="stretch")
        gt_rgba = mask_to_rgba(gt_mask)
        gt_overlay = blend_overlay(ref_result.image_resized, gt_rgba, alpha=200)
        top_cols[1].image(gt_overlay, caption="GT overlay (rouge)",
                          width="stretch")
    else:
        img_col, _ = st.columns([1, 3])
        img_col.image(
            ref_result.image_resized,
            caption=f"Image — {category}",
            width="stretch",
        )

    # 4 colonnes : 1 par modèle
    st.markdown("#### Heatmaps par modèle")
    cols = st.columns(4)
    per_image_aurocs = {}
    for col, (label, result) in zip(cols, results.items()):
        col.markdown(f"**{label}**")
        col.image(
            defect_only_overlay(result.image_resized, result.heatmap_norm),
            width="stretch",
        )
        col.metric("Score", f"{result.score:.4f}")
        # AUROC pixel per-image si GT disponible
        if gt_mask is not None and gt_mask.sum() > 0:
            auroc = per_image_auroc_pixel(result.heatmap, gt_mask)
            if auroc is not None:
                per_image_aurocs[label] = auroc
                col.metric("AUROC pix (image)", f"{auroc:.4f}")

    st.markdown("#### Décision (basée sur Ensemble Mean)")
    render_decision_badge(
        results["Ensemble Mean"].score, good_threshold, defect_threshold
    )

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
            width="stretch",
        )

    st.caption(
        "ℹ️ **Score [0, 1]** : normalisé contre 100 train good. Désormais comparable entre modèles. "
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
    st.altair_chart(chart, width="stretch")

# --- Info ground truth si applicable --------------------------------------
if selected_path:
    defect = Path(selected_path).parent.name
    if defect == "good":
        st.success(f"✓ Image **good** (ground truth) — score idéalement bas.")
    else:
        st.info(f"⚠ Image **défective** ({defect}) — score idéalement haut.")
