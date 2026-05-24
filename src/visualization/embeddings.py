"""Visualisation d'embeddings 2D et scores de sÃ©parabilitÃ©."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def reduce_2d(embeddings: np.ndarray, method: str = "umap", seed: int = 42) -> np.ndarray:
    """RÃ©duit des embeddings (N, D) Ã  (N, 2).

    method : "umap" (nÃ©cessite umap-learn), "pca", "tsne"
    """
    if method == "umap":
        import umap  # import lazy â€” umap-learn est une dep optionnelle

        reducer = umap.UMAP(n_components=2, random_state=seed, n_jobs=1)
        return reducer.fit_transform(embeddings)
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, random_state=seed, init="pca").fit_transform(embeddings)
    raise ValueError(f"MÃ©thode inconnue : {method}")


def plot_2d(
    coords: np.ndarray,
    df: pd.DataFrame,
    color_by: str,
    ax: Axes,
    palette: dict | None = None,
    s: float = 10,
    alpha: float = 0.7,
    title: str | None = None,
) -> None:
    """Scatter 2D colorÃ© par une colonne du DataFrame.

    `coords` shape (N, 2) alignÃ© avec `df` (mÃªme ordre d'index).
    """
    values = df[color_by]
    categories = sorted(values.unique(), key=lambda x: (x is None, str(x)))

    for cat in categories:
        mask = (values == cat).to_numpy()
        color = palette.get(cat) if palette else None
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=str(cat), color=color, s=s, alpha=alpha,
        )
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    if title:
        ax.set_title(title)
    if len(categories) <= 10:
        ax.legend(fontsize=8, loc="best")


def separability_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Score de silhouette pour mesurer la sÃ©parabilitÃ© Normal/Anomal dans un espace.

    Plus c'est proche de 1 â†’ bien sÃ©parÃ©. Proche de 0 â†’ chevauchement.
    NÃ©gatif â†’ classes mal sÃ©parÃ©es (anomalies au milieu des normaux).
    """
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(embeddings, labels))





