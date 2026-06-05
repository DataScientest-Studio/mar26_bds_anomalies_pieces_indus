"""Streamlit landing page — Accueil / MVTec Anomaly Detection Demo.

Run :
    uv run streamlit run app/Accueil.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import PATHS

# Cache partagé : doit être importé depuis app/_shared.py pour que les pages
# voient les mêmes pipelines pré-chargés ici.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import warmup_all_models  # noqa: E402

st.set_page_config(
    page_title="Anomaly Detection - Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Pré-chargement de TOUS les modèles au boot --------------------------
# Une seule fois par session Streamlit grâce à st.session_state.
if not st.session_state.get("models_warmed_up"):
    with st.status(
        "🔥 Pré-chargement des modèles démo…", expanded=True
    ) as status:
        st.caption(
            "Construction des pipelines pour les **3 catégories MVTec de la démo "
            "(cable, leather, transistor) + HSS-IAD**. "
            "Cette étape n'est faite qu'**une seule fois par session**. "
            "Les autres catégories MVTec restent disponibles dans le sélecteur "
            "et seront chargées paresseusement au 1er clic."
        )
        loaded, total = warmup_all_models()
        status.update(
            label=f"✓ {loaded}/{total} pipelines en cache — démo prête",
            state="complete",
            expanded=False,
        )
    st.session_state["models_warmed_up"] = True

# --- Header ---
st.title("🔍 Industrial Anomaly Detection")
st.markdown(
    "**ML Engineer · Anomaly Detection · MVTec AD + HSS-IAD**"
)
st.markdown("---")

# --- Project intro ---
st.markdown(
    """
## Le projet

Détection d'anomalies sur pièces industrielles (MVTec AD + HSS-IAD).
On part de **PatchCore from-scratch** (~0.30 AUPIMO) et on construit pas à pas un
**baseline ensemble** atteignant **0.843 AUPIMO** sur 15 catégories MVTec -
à portée du SOTA Dinomaly 2024 (~0.86).

## Architecture finale
"""
)

st.code(
    """
                            ┌────────────────────────────┐
                            │    Image industrielle      │
                            └─────────────┬──────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
       ┌──────────────────────────┐               ┌──────────────────────────┐
       │      Dinomaly V2         │               │      PatchCore           │
       │   Reverse Distillation   │               │   Memory Bank + k-NN     │
       └────────────┬─────────────┘               └─────────────┬────────────┘
                    │                                           │
                    ▼                                           ▼
                heatmap                                     heatmap
                    │                                           │
                    ▼                                           ▼
       ┌──────────────────────────┐               ┌──────────────────────────┐
       │  Agrégation pixel→image  │               │  Agrégation pixel→image  │
       │   lissage + max          │               │   lissage + max          │
       └────────────┬─────────────┘               └─────────────┬────────────┘
                    │                                           │
                    ▼                                           ▼
       ┌──────────────────────────┐               ┌──────────────────────────┐
       │   Normalisation [0, 1]   │               │   Normalisation [0, 1]   │
       │   contre train good      │               │   contre train good      │
       └────────────┬─────────────┘               └─────────────┬────────────┘
                    │                                           │
              score_dino                                    score_pc
                    │                                           │
                    └────────────────────┬──────────────────────┘
                                         ▼
                          ┌──────────────────────────────┐
                          │       Ensemble Mean          │
                          │   (score_dino + score_pc)/2  │
                          └──────────────┬───────────────┘
                                         │
                                  score_final
                                         │
                                         ▼
              ┌──────────────────────────────────────────────────────┐
              │     Décision 3 classes — seuils calibrés             │
              ├──────────────────────────────────────────────────────┤
              │     score ≤ threshold_good     →   Pièce bonne       │
              │     score ≥ threshold_defect   →   Pièce défectueuse │
              │     entre les deux             →   À vérifier        │
              └──────────────────────────────────────────────────────┘
""",
    language="text",
)

st.markdown(
    """
## Stack
- Python 3.12 · uv (multi-CUDA) · PyTorch 2.11 CUDA 12.8
- anomalib 2.3.1 · DINOv2-B (Meta) · WideResNet50_2 (ImageNet)
- Streamlit · Matplotlib · scikit-learn

## Code & ressources
- Repo : [github.com/DataScientest-Studio/mar26_bds_anomalies_pieces_indus](https://github.com/DataScientest-Studio/mar26_bds_anomalies_pieces_indus)
- Dataset MVTec AD : [mvtec.com](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- Dataset HSS-IAD : [hss-iad.com](https://hss-iad.com/dataset)
"""
)

# --- Footer ---
st.sidebar.markdown("### À propos")
st.sidebar.markdown(
    """
2026 · ML Engineer

"""
)
st.sidebar.markdown("---")
st.sidebar.info(
    f"📁 Projet root :\n`{PATHS.root.name}`\n\n"
    f"📦 Models dir :\n`{(PATHS.root / 'models').exists() and 'présent' or 'absent'}`\n\n"
    f"📊 Data dir :\n`{PATHS.mvtec_dir.exists() and 'présent' or 'absent'}`"
)
