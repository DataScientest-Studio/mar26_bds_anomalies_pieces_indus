Industrial Anomaly Detection — MVTec AD + HSS-IAD
==============================

Détection d'anomalies non supervisée sur pièces industrielles. Deux pistes
parallèles : un **modèle générique** validé sur les 15 catégories MVTec AD,
et un **modèle spécialisé** pour notre cas client (HSS-IAD Casting_class1).

Approche : entraînement uniquement sur images **good**, puis détection de
toute déviation à l'inférence. Combinaison de deux modèles SOTA — **Dinomaly
2024** (Reverse Distillation sur DINOv2) et **PatchCore** (memory bank +
k-NN) — agrégés en ensemble. Décision en 3 classes (bonne / à vérifier /
défectueuse) avec seuils calibrés per-catégorie sur le test set labélisé.

**Résultats** :

- **MVTec AD** (15 catégories, ensemble Dinomaly + PatchCore) : AUPIMO moyen
  **0.843** (à portée du SOTA Dinomaly 2024 ≈ 0.86), AUROC image **0.997**.
- **HSS-IAD Casting_class1** (pipeline spécialisée ROI + RD/Feature-AE +
  soft calibration) : Image AP **0.92**, Pixel AP **0.38**, AUPIMO **0.47**.

Project Organization
------------

    ├── LICENSE
    ├── README.md                  <- Top-level README for developers using this project.
    ├── pyproject.toml             <- Project metadata, deps, CLI entry points (uv / hatch).
    ├── requirements.txt           <- Pip requirements (regenerate via `uv export`).
    ├── uv.lock                    <- Locked dependency versions (uv).
    │
    ├── app                        <- Streamlit demo application.
    │   ├── Accueil.py             <- Landing page + auto-warmup des pipelines.
    │   ├── _shared.py             <- Cache partagé entre pages (pipelines, warmup).
    │   └── pages
    │       ├── 1_Benchmark.py     <- 15 catégories × 3 modèles, AUROC/AUPIMO.
    │       ├── 2_MVTec.py         <- Inférence live + Compare 3 modèles + décision 3 classes.
    │       └── 3_HSS-IAD.py       <- Pipeline ROI + RD/Feature-AE spécialisée Casting_class1.
    │
    ├── data                       <- Datasets (local only, excluded from Git).
    │   ├── raw
    │   │   ├── mvtec              <- MVTec AD (15 catégories).
    │   │   └── hss-iad            <- HSS-IAD (7 catégories).
    │   ├── processed              <- Harmonized data and curated masks.
    │   └── classified             <- Classified subsets used by experiments.
    │
    ├── models                     <- Trained models, checkpoints (local only).
    │
    ├── notebooks
    │   └── 01_eda_harmonisation.ipynb
    │
    ├── references                 <- Documentation, links, research papers.
    │
    ├── reports                    <- Reports, figures, manifests and experiment outputs.
    │   ├── calibration
    │   │   └── mvtec_thresholds.csv  <- Seuils décision per-(catégorie, modèle).
    │   ├── casting_roi_rd_ae      <- Casting_class1 ROI + RD/Feature-AE documentation.
    │   ├── dinomaly               <- Benchmarks Dinomaly per-catégorie.
    │   ├── ensemble               <- Benchmarks ensemble Dinomaly + PatchCore.
    │   ├── figures                <- Présentation et figures d'analyse.
    │   ├── registries             <- Manifests dataset et défauts.
    │   ├── tables                 <- Tableaux récapitulatifs.
    │   ├── Pitch_demo_5min.md     <- Script pas-à-pas pour la démo soutenance.
    │   ├── Slides_soutenance.md
    │   └── Slides_soutenance.pdf
    │
    ├── scripts                    <- Utility scripts, dataset lineage, calibration.
    │   ├── calibrate_thresholds.py  <- Calibre les seuils décision sur test set MVTec.
    │   ├── download_data.py         <- Téléchargement helpers MVTec/HSS-IAD.
    │   ├── install.py               <- Auto-détection CUDA pour uv sync.
    │   └── datasets                 <- Lineage CSVs.
    │
    └── src                        <- Project source code (Python package).
        ├── __init__.py
        ├── config.py              <- Paths, datasets, EDA parameters.
        │
        ├── data                   <- Data pipeline (harmonization, splits, masks).
        ├── features               <- Preprocessing, ROI tools, synthetic defects.
        ├── inference              <- Pipelines d'inférence runtime.
        │   ├── pipeline.py        <- AnomalyPipeline (Dinomaly + PatchCore + ensemble).
        │   └── casting_feature_ae.py  <- Pipeline HSS-IAD Casting.
        ├── models                 <- Wrappers training/eval (baselines, RD-AE, segmentation).
        ├── reporting              <- Shared report paths and manifests.
        └── visualization          <- Overlays, previews, heatmaps.

Data Harmonization
-------------------------

Avant toute analyse, télécharger les datasets manuellement et les placer dans :

    data/raw/mvtec/      <- MVTec AD (15 catégories)
    data/raw/hss-iad/    <- HSS-IAD (7 catégories)

Puis lancer l'harmonisation depuis la racine du projet (**uv** gère l'env
virtuel et les dépendances) :

    uv run python -m src.data.harmonize

Le script scanne les deux structures (MVTec-style:
`category/{train,test}/{good,defective,...}`), récupère les masques GT et
génère deux CSVs dans `data/processed/` :

- `unified_dataset.csv` — une ligne par image avec les colonnes
  `dataset`, `category`, `split`, `label`, `is_anomaly`, `image_path`,
  `mask_path`, `has_mask`.
- `resolutions_sample.csv` — échantillon de 2000 images avec `width`,
  `height`, `channels` (utilisé par l'EDA notebook).

Tous les chemins et paramètres sont centralisés dans
[`src/config.py`](src/config.py).

Ensuite, ouvrir
[`notebooks/01_eda_harmonisation.ipynb`](notebooks/01_eda_harmonisation.ipynb)
pour l'analyse exploratoire.

--------

Application Streamlit
-------------------------

L'app comporte 4 pages :

| Page | Rôle |
|---|---|
| **Accueil** | Présentation projet + schéma architecture + pré-chargement automatique. |
| **Benchmark** | Vue agrégée 15 catégories × 3 modèles (AUROC image/pixel, AUPIMO). |
| **MVTec** | Inférence live par catégorie, modes Single et Compare 3 modèles, badge décision 3 classes, sliders heatmap live. |
| **HSS-IAD** | Pipeline spécialisée Casting_class1 : ROI + RD/Feature-AE + calibration soft ROI. |

Lancement depuis la racine du projet :

    uv run --extra cu128 streamlit run app/Accueil.py

ou, avec un venv déjà créé :

    .\.venv\Scripts\python.exe -m streamlit run app\Accueil.py

⚠️ **Toujours préciser `--extra cu128`** (ou `cu121` / `cu124` / `cpu` selon
ton GPU) pour éviter que `uv run` re-synchronise et désinstalle torch CUDA
à chaque appel — cf. [`pyproject.toml`](pyproject.toml).

À l'ouverture, l'app **pré-charge automatiquement** les 3 catégories MVTec
de démo (cable, leather, transistor) + HSS-IAD (~20-30 min). Les autres
catégories MVTec sont chargées paresseusement au 1er clic.

Threshold Calibration
-------------------------

Pour calibrer les seuils de décision sur le test set MVTec labélisé :

    uv run --extra cu128 python scripts/calibrate_thresholds.py

Pour chaque (catégorie, modèle) le script calcule :

- AUROC image-level (rank-based, indépendant du seuil)
- `threshold_good` = p90 des scores des images bonnes
- `threshold_defect` = p10 des scores des images défectueuses

Sortie : [`reports/calibration/mvtec_thresholds.csv`](reports/calibration/mvtec_thresholds.csv),
chargé automatiquement par la page MVTec qui pré-remplit les sliders de
décision avec ces seuils per-catégorie.

Command-Line Pipelines
-------------------------

Les principales entrées CLI sont exposées dans `pyproject.toml` :

    uv run --extra cu128 train-roi --help              # Segmentation ROI fonctionnelle (Casting)
    uv run --extra cu128 predict-roi --help
    uv run --extra cu128 train-rd-ae --help            # Reverse Distillation / Feature-AE
    uv run --extra cu128 evaluate-rd-ae --help
    uv run --extra cu128 calibrate-rd-ae --help        # Calibration soft ROI
    uv run --extra cu128 materialize-quality-heatmaps --help
    uv run --extra cu128 build-defect-library --help   # Génère défauts synthétiques

HSS-IAD Casting_class1 Inspection Pipeline
-------------------------

Pour notre cas client, on est allé plus loin que la générique MVTec :

- **Segmentation ROI fonctionnelle** (U-Net dédié) qui isole la surface usinée.
- **Reverse Distillation Feature-AE** multi-couches sur features ResNet.
- **Calibration soft ROI** : la heatmap est pondérée continûment par la
  probabilité d'être dans la ROI fonctionnelle (pas un masque binaire dur).
- **Décision 3 classes** avec seuils calibrés sur le test set HSS-IAD.

Métriques : Image AP **0.92**, Pixel AP **0.38**, AUPIMO **0.14**.

Documentation détaillée :

- [`reports/casting_roi_rd_ae/README.md`](reports/casting_roi_rd_ae/README.md)
- [`reports/casting_roi_rd_ae/DATASCIENCE_PIPELINE.md`](reports/casting_roi_rd_ae/DATASCIENCE_PIPELINE.md)
- [`reports/casting_roi_rd_ae/DATASET_MASKS_AND_SYNTHETIC_DEFECTS.md`](reports/casting_roi_rd_ae/DATASET_MASKS_AND_SYNTHETIC_DEFECTS.md)
- [`reports/casting_roi_rd_ae/registries/selected_models.md`](reports/casting_roi_rd_ae/registries/selected_models.md)

Stack
-------------------------

- **Python 3.12** · **uv** (gestion multi-CUDA) · **PyTorch 2.11 CUDA 12.8**
- **anomalib 2.3.1** · **DINOv2-B** (Meta) · **WideResNet50_2** (ImageNet)
- **Streamlit** · **Matplotlib** · **scikit-learn** · **OpenCV**

