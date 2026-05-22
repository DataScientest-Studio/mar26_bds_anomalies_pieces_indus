"""Streamlit landing page — MVTec Anomaly Detection Demo.

Run :
    uv run streamlit run app/main.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import PATHS

st.set_page_config(
    page_title="MVTec Anomaly Detection — Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Header ---
st.title("🔍 Industrial Anomaly Detection")
st.markdown(
    "**Portfolio ML Engineer · MVTec AD + HSS-IAD · Ensemble Dinomaly + PatchCore**"
)
st.markdown("---")

# --- Stats hero ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("AUPIMO moyen", "0.843", help="15 catégories MVTec AD")
col2.metric("Catégories", "15", help="bottle → zipper")
col3.metric("Modèles testés", "5", help="PaDiM, DRAEM, Dinomaly, PatchCore, Ensemble")
col4.metric("Notebooks", "12", help="De l'EDA aux études négatives")

st.markdown("---")

# --- Project intro ---
st.markdown(
    """
## Le projet

Détection d'anomalies sur pièces industrielles (MVTec AD + HSS-IAD).
On part de **PaDiM from-scratch** (~0.30 AUPIMO) et on construit pas à pas un
**baseline ensemble** atteignant **0.843 AUPIMO** sur 15 catégories MVTec —
à portée du SOTA Dinomaly 2024 (~0.86).

## Architecture finale
"""
)

st.code(
    """
        Image industrielle
                │
       ┌────────┴────────┐
       ▼                 ▼
   Dinomaly V2       PatchCore manuel
   (504, 40ep)       (WRN50_2 + coreset 10%)
       │                 │
   norm_global_min_max par modèle
       │                 │
       └────────┬────────┘
                ▼
          Mean heatmap
                │
                ▼
      Score = max(heatmap)
""",
    language="text",
)

st.markdown(
    """
## Navigation

- **🔍 Live inference** — Drag-drop une image ou choisis dans la galerie d'exemples.
- **📊 Benchmark** — Tableau interactif des 15 catégories : Dinomaly seul, PatchCore seul, Ensemble Mean / Max.
- **📖 Journey** — L'histoire complète du projet : 6 succès + 4 études négatives.

## Stack
- Python 3.12 · uv (multi-CUDA) · PyTorch 2.11 CUDA 12.8
- anomalib 2.3.1 · DINOv2-B (Meta) · WideResNet50_2 (ImageNet)
- Streamlit · Matplotlib · scikit-learn

## Code & ressources
- Repo : [github.com/Missipsa06/...](https://github.com/Missipsa06)
- Journey complet : `reports/mvtec_journey.md`
- Notebooks : `notebooks/01_eda` → `12_hazelnut_improvements`
"""
)

# --- Footer ---
st.sidebar.markdown("### À propos")
st.sidebar.markdown(
    """
**Missipsa BENDJOUDI** · 2026
Portfolio ML Engineer

Co-réalisé avec des collègues.
"""
)
st.sidebar.markdown("---")
st.sidebar.info(
    f"📁 Projet root :\n`{PATHS.root.name}`\n\n"
    f"📦 Models dir :\n`{(PATHS.root / 'models').exists() and 'présent' or 'absent'}`\n\n"
    f"📊 Data dir :\n`{PATHS.mvtec_dir.exists() and 'présent' or 'absent'}`"
)
