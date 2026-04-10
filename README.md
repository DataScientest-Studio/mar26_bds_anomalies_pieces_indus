Project Name
==============================

This repo is a Starting Pack for DS projects. You can rearrange the structure to make it fits your project.

Project Organization
------------

    ├── LICENSE
    ├── README.md                  <- The top-level README for developers using this project.
    ├── pyproject.toml             <- Project metadata and dependencies (uv / hatch).
    ├── requirements.txt           <- Pip requirements for reproducing the analysis environment.
    ├── uv.lock                    <- Locked dependency versions (uv).
    │
    ├── data                       <- Datasets (local only, excluded from Git).
    │   ├── raw                    <- Original, immutable data dump.
    │   │   ├── mvtec              <- MVTec AD (15 catégories).
    │   │   └── hss-iad            <- HSS-IAD (7 catégories).
    │   └── processed              <- Données harmonisées prêtes à l'emploi.
    │       ├── unified_dataset.csv
    │       └── resolutions_sample.csv
    │
    ├── models                     <- Modèles entraînés, prédictions, checkpoints.
    │
    ├── notebooks                  <- Jupyter notebooks (numérotés pour l'ordre).
    │   ├── 01_eda_harmonisation.ipynb
    │   └── 01_eda_executed.ipynb  <- Version exécutée (via nbconvert).
    │
    ├── references                 <- Documentation, liens, papiers (ex. HSS-IAD).
    │
    ├── reports                    <- Rapports générés à partir des analyses.
    │   ├── eda_report.md
    │   └── figures
    │       └── eda                <- Figures extraites du notebook EDA.
    │
    └── src                        <- Code source du projet (package Python).
        ├── __init__.py
        ├── config.py              <- Configuration centrale (chemins, params EDA).
        │
        ├── data                   <- Pipeline de données.
        │   ├── __init__.py
        │   └── harmonize.py       <- Harmonisation MVTec AD + HSS-IAD → CSV unifié.
        │
        ├── features               <- (à venir) extraction de features.
        ├── models                 <- (à venir) entraînement et prédictions.
        ├── visualization          <- (à venir) visualisations.
        └── streamlit              <- (à venir) app de démo.

Harmonisation des données
-------------------------

Avant toute analyse, les deux datasets bruts doivent être téléchargés
manuellement et placés dans :

    data/raw/mvtec/      <- MVTec AD (15 catégories)
    data/raw/hss-iad/    <- HSS-IAD (7 catégories)

Lancer ensuite le script d'harmonisation depuis la racine du projet
(via **uv**, qui gère l'environnement et les dépendances) :

    uv run python -m src.data.harmonize

Le script scanne les deux arborescences (structure MVTec-style
`category/{train,test}/{good,defective,...}`), récupère les masques
correspondants dans `ground_truth/` et produit deux CSV dans
`data/processed/` :

- `unified_dataset.csv` — une ligne par image avec les colonnes
  `dataset`, `category`, `split`, `label`, `is_anomaly`, `image_path`,
  `mask_path`, `has_mask`.
- `resolutions_sample.csv` — échantillon de 2000 images avec `width`,
  `height`, `channels` (utilisé par le notebook EDA).

Tous les chemins et paramètres du scan (extensions d'images, splits,
label normal, taille d'échantillon, seed) sont centralisés dans
[`src/config.py`](src/config.py).

Une fois les CSV générés, ouvrir
[`notebooks/01_eda_harmonisation.ipynb`](notebooks/01_eda_harmonisation.ipynb)
pour l'analyse exploratoire.

--------

