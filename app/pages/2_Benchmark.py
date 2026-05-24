"""Benchmark interactif des 15 catégories MVTec."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.config import PATHS

st.set_page_config(page_title="Benchmark", page_icon="📊", layout="wide")

st.title("📊 Benchmark MVTec AD — 15 catégories")
st.markdown(
    "Comparaison **Dinomaly seul · PatchCore seul · Ensemble Mean · Ensemble Max** "
    "sur AUROC image, AUROC pixel et AUPIMO."
)

# --- Load CSV ---
CSV_PATH = PATHS.root / "reports" / "ensemble" / "mvtec_dinomaly_patchcore_ensemble.csv"
if not CSV_PATH.exists():
    st.error(f"CSV absent : `{CSV_PATH.relative_to(PATHS.root)}`")
    st.stop()

df = pd.read_csv(CSV_PATH)

# --- Synthèse ---
st.markdown("### Moyennes globales (15 catégories)")
mean_cols = [c for c in df.columns if c.endswith(("_img", "_pix", "_aupimo"))]
means = df[mean_cols].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dinomaly AUPIMO", f"{means['dino_aupimo']:.4f}")
c2.metric("PatchCore AUPIMO", f"{means['pc_aupimo']:.4f}",
          delta=f"{means['pc_aupimo']-means['dino_aupimo']:+.4f}")
c3.metric("Ensemble Mean", f"{means['mean_aupimo']:.4f}",
          delta=f"{means['mean_aupimo']-means['dino_aupimo']:+.4f}")
c4.metric("Ensemble Max", f"{means['max_aupimo']:.4f}",
          delta=f"{means['max_aupimo']-means['dino_aupimo']:+.4f}")

# --- Tabs : tableau / barplot / per-defect ---
tab1, tab2 = st.tabs(["📋 Tableau", "📈 Barplot AUPIMO"])

with tab1:
    metric = st.selectbox(
        "Métrique à afficher",
        ["AUPIMO", "AUROC image", "AUROC pixel"],
        index=0,
    )
    metric_key = {
        "AUPIMO": "_aupimo",
        "AUROC image": "_img",
        "AUROC pixel": "_pix",
    }[metric]

    cols_show = ["category", "n_test", f"dino{metric_key}", f"pc{metric_key}",
                 f"mean{metric_key}", f"max{metric_key}"]
    df_show = df[cols_show].copy()
    df_show["best"] = df_show[[f"dino{metric_key}", f"pc{metric_key}",
                                 f"mean{metric_key}", f"max{metric_key}"]].idxmax(axis=1)
    df_show = df_show.sort_values(f"mean{metric_key}", ascending=False)

    st.dataframe(
        df_show.style.background_gradient(
            subset=[f"dino{metric_key}", f"pc{metric_key}", f"mean{metric_key}", f"max{metric_key}"],
            cmap="RdYlGn", vmin=0, vmax=1
        ).format(precision=4),
        width="stretch",
        hide_index=True,
    )

with tab2:
    import altair as alt

    df_long = df.melt(
        id_vars=["category"],
        value_vars=["dino_aupimo", "pc_aupimo", "mean_aupimo", "max_aupimo"],
        var_name="modèle",
        value_name="AUPIMO",
    )
    df_long["modèle"] = df_long["modèle"].replace({
        "dino_aupimo": "Dinomaly",
        "pc_aupimo": "PatchCore",
        "mean_aupimo": "Ensemble Mean",
        "max_aupimo": "Ensemble Max",
    })

    chart = (
        alt.Chart(df_long)
        .mark_bar()
        .encode(
            x=alt.X("AUPIMO:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("category:N", sort=alt.SortField("AUPIMO", order="ascending")),
            color=alt.Color("modèle:N", scale=alt.Scale(scheme="category10")),
            row=alt.Row("modèle:N", header=alt.Header(labelOrient="top")),
            tooltip=["category", "modèle", "AUPIMO"],
        )
        .properties(width=600, height=180)
    )
    st.altair_chart(chart, width="content")

# --- Catégories à explorer ---
st.markdown("---")
st.markdown("### Cas marquants")

best_mean = df.loc[df["mean_aupimo"].idxmax()]
worst_mean = df.loc[df["mean_aupimo"].idxmin()]
biggest_gain = df.assign(gain=df["mean_aupimo"] - df["dino_aupimo"]).sort_values("gain", ascending=False).iloc[0]

cc1, cc2, cc3 = st.columns(3)
cc1.markdown(f"**🏆 Meilleure catégorie**\n`{best_mean['category']}`\nAUPIMO {best_mean['mean_aupimo']:.4f}")
cc2.markdown(f"**🔻 Plus dure**\n`{worst_mean['category']}`\nAUPIMO {worst_mean['mean_aupimo']:.4f}")
cc3.markdown(f"**⚡ Plus gros gain Ensemble**\n`{biggest_gain['category']}`\nΔ {biggest_gain['gain']:+.3f}")
