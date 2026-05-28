"""Casting_class1 ROI + RD AE restitution page.

This page is intentionally lightweight: it reads prepared CSVs and figures, and
it does not run heavy model inference.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.casting_feature_ae import CastingFeatureAEPipeline

REPORT_DIR = PROJECT_ROOT / "reports" / "casting_roi_rd_ae"
METRICS_CSV = REPORT_DIR / "metrics" / "casting_class1_metric_history_for_soutenance.csv"
GAINS_CSV = REPORT_DIR / "metrics" / "casting_class1_metric_milestone_gains_for_soutenance.csv"
HEATMAP_DIR = REPORT_DIR / "heatmaps" / "quality_heatmaps_thresholded"
BLUE_ORANGE_DIR = REPORT_DIR / "heatmaps" / "blue_orange_quality_heatmaps"
TEST_DIR = PROJECT_ROOT / "data" / "raw" / "hss-iad" / "Casting_class1" / "test"

st.set_page_config(page_title="Casting_class1", layout="wide")


def classify_anomaly_score(score: float, good_threshold: float, defect_threshold: float) -> tuple[str, str, str]:
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

st.title("Casting_class1 - ROI + RD AE")
st.markdown(
    "Configuration finale de notre piste spécialisée Casting_class1 : "
    "segmentation de surface fonctionnelle, Reverse Distillation AE, calibration "
    "post-hoc et rendu qualité inspecteur."
)

cols = st.columns(4)
cols[0].metric("Image AP", "0.9221", help="Calibration soft ROI rerun")
cols[1].metric("Pixel AP", "0.3775", help="Calibration soft ROI rerun")
cols[2].metric("AUPIMO", "0.1415", help="Faible taux de faux positifs")
cols[3].metric("Seuil preview", "0.35", help="Rendu ajustable dans l'app")

st.markdown("---")


@st.cache_resource(show_spinner="Chargement ROI + RD AE...")
def get_casting_pipeline() -> CastingFeatureAEPipeline:
    return CastingFeatureAEPipeline(device="auto")


@st.cache_data
def list_casting_examples(max_per_class: int = 8) -> dict[str, list[str]]:
    if not TEST_DIR.exists():
        return {}
    examples: dict[str, list[str]] = {}
    for subdir in sorted(TEST_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        images = sorted(subdir.glob("*.jpg")) + sorted(subdir.glob("*.png"))
        if images:
            examples[subdir.name] = [str(path) for path in images[:max_per_class]]
    return examples


st.subheader("Inference directe ROI + RD AE")
with st.sidebar:
    st.markdown("### Casting_class1 inference")
    display_threshold = st.slider("Seuil d'affichage", 0.01, 0.80, 0.35, 0.01)
    with st.expander("Réglages avancés heatmap", expanded=True):
        display_low_percentile = st.slider("Percentile bas", 0.0, 95.0, 85.0, 1.0)
        display_high_percentile = st.slider("Percentile haut", 95.0, 100.0, 99.8, 0.1)
        display_gamma = st.slider("Gamma", 0.4, 2.5, 1.4, 0.1)
        overlay_alpha = st.slider("Opacité overlay", 0.10, 1.00, 0.72, 0.02)
    with st.expander("Réglages avancés modèle", expanded=False):
        roi_threshold = st.slider("Seuil ROI", 0.05, 0.90, 0.50, 0.05)
        topk_fraction = st.select_slider(
            "Fraction top-k score",
            options=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
            value=0.005,
            format_func=lambda value: f"{100 * value:.1f}%",
        )
    with st.expander("Décision anomalie", expanded=True):
        st.caption("Calibration test set : vert sans défaut connu, rouge sans bonne pièce connue, jaune pour la zone de recouvrement.")
        good_threshold = st.slider("Bon si score <=", 0.30, 0.60, 0.425, 0.005)
        defect_threshold = st.slider("Défectueux si score >=", 0.40, 0.80, 0.515, 0.005)
        if defect_threshold <= good_threshold:
            st.warning("Le seuil défaut doit être supérieur au seuil bon.")
    source = st.radio("Image", ["Upload", "Galerie test"], index=1)

selected_image: Image.Image | None = None
selected_path: str | None = None
selected_key: str | None = None

if source == "Upload":
    uploaded = st.file_uploader("Image Casting_class1", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        uploaded_bytes = uploaded.read()
        selected_key = f"upload:{uploaded.name}:{len(uploaded_bytes)}"
        selected_image = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
else:
    examples = list_casting_examples()
    if not examples:
        st.info(f"Aucun exemple trouve dans `{TEST_DIR}`.")
    else:
        tabs = st.tabs(list(examples.keys()))
        for tab, (label, paths) in zip(tabs, examples.items(), strict=True):
            with tab:
                cols_gallery = st.columns(4)
                for idx, path in enumerate(paths):
                    with cols_gallery[idx % 4]:
                        st.image(path, width="stretch")
                        if st.button("Analyser", key=f"casting_{label}_{idx}"):
                            selected_path = path
                            selected_key = f"path:{path}"
                            st.session_state["casting_selected_path"] = path
                            selected_image = Image.open(path).convert("RGB")

if selected_image is None and source == "Galerie test" and "casting_selected_path" in st.session_state:
    selected_path = st.session_state["casting_selected_path"]
    selected_key = f"path:{selected_path}"
    selected_image = Image.open(selected_path).convert("RGB")

if selected_image is not None:
    if defect_threshold <= good_threshold:
        defect_threshold = good_threshold + 0.01
    pipe = get_casting_pipeline()
    if (
        st.session_state.get("casting_last_key") != selected_key
        or "casting_last_raw_result" not in st.session_state
    ):
        with st.spinner("Inference segmentation + RD AE..."):
            st.session_state["casting_last_raw_result"] = pipe.predict(selected_image)
            st.session_state["casting_last_key"] = selected_key
    result = pipe.render(
        st.session_state["casting_last_raw_result"],
        display_threshold=display_threshold,
        display_low_percentile=display_low_percentile,
        display_high_percentile=display_high_percentile,
        display_gamma=display_gamma,
        overlay_alpha=overlay_alpha,
        roi_threshold=roi_threshold,
        topk_fraction=topk_fraction,
    )

    view_cols = st.columns(4)
    view_cols[0].image(result.image_rgb, caption="Image", width="stretch")
    view_cols[1].image(result.roi_overlay, caption="ROI fonctionnelle", width="stretch")
    view_cols[2].image(result.heatmap_overlay, caption="Heatmap anomalie", width="stretch")
    view_cols[3].image(
        (np.clip(result.display_map, 0, 1) * 255).astype(np.uint8),
        caption="Signal affiche",
        width="stretch",
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Score anomalie", f"{result.score:.4f}")
    m2.metric("ROI surface", f"{100 * float(result.surface_mask.mean()):.1f}%")
    m3.metric("Seuil affichage", f"{display_threshold:.2f}")
    m4.metric("Top-k", f"{100 * topk_fraction:.1f}%")
    decision_label, decision_color, confidence_label = classify_anomaly_score(
        result.score,
        good_threshold,
        defect_threshold,
    )
    st.markdown(
        f"""
        <div style="border-left: 10px solid {decision_color}; background: {decision_color}18;
                    padding: 0.85rem 1rem; border-radius: 0.35rem; margin: 0.5rem 0 0.75rem 0;">
            <div style="font-size: 1.4rem; font-weight: 750; color: {decision_color};">{decision_label}</div>
            <div style="font-size: 0.95rem; color: #2f3b4a;">
                Score {result.score:.4f} · bon ≤ {good_threshold:.2f} · défectueux ≥ {defect_threshold:.2f} · {confidence_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Contraste heatmap : p{display_low_percentile:.0f}-p{display_high_percentile:.1f}, "
        f"gamma {display_gamma:.1f}, alpha {overlay_alpha:.2f}, seuil ROI {roi_threshold:.2f}"
    )
    if selected_path is not None:
        st.caption(f"Image : `{selected_path}`")
else:
    st.info("Selectionne une image pour lancer l'inference directe.")

st.markdown("---")

left, right = st.columns(2)
with left:
    st.subheader("Rendu qualité seuillé")
    img = HEATMAP_DIR / "comparison_defective.png"
    if img.exists():
        st.image(str(img), caption="Défauts affichés après seuil qualité 0.38")
    else:
        st.info(f"Figure absente : {img}")

with right:
    st.subheader("Rendu bleu/orange calibré")
    img = BLUE_ORANGE_DIR / "comparison_defective.png"
    if img.exists():
        st.image(str(img), caption="Calibration soft ROI, sans masque ROI apparent")
    else:
        st.info(f"Figure absente : {img}")
        
st.markdown("---")
st.caption(
    "Les poids restent locaux et sont documentés par manifest. "
    "Cette page affiche les résultats matérialisés et ne lance pas d'inférence lourde."
)
