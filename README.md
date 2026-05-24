Project Name
==============================

This repo is a Starting Pack for DS projects. You can rearrange the structure to make it fits your project.

Project Organization
------------


    ├── LICENSE
    ├── README.md                  <- Top-level README for developers using this project.
    ├── pyproject.toml             <- Project metadata and dependencies (uv / hatch).
    ├── requirements.txt           <- Pip requirements to reproduce the analysis environment.
    ├── uv.lock                    <- Locked dependency versions (uv).
    │
    ├── app                        <- Streamlit demo application.
    │   ├── main.py
    │   └── pages                  <- Live inference, benchmarks, Casting_class1.
    │
    ├── data                       <- Datasets (local only, excluded from Git).
    │   ├── raw                    <- Original, immutable data dump.
    │   │   ├── mvtec              <- MVTec AD (15 categories).
    │   │   └── hss-iad            <- HSS-IAD (7 categories).
    │   ├── processed              <- Harmonized data and curated masks.
    │   └── classified             <- Classified subsets used by experiments.
    │
    ├── models                     <- Trained models, predictions, checkpoints.
    │
    ├── notebooks                  <- Jupyter notebooks (ordered numerically).
    │   ├── 01_eda_harmonisation.ipynb
    │
    ├── references                 <- Documentation, links, research papers (e.g., HSS-IAD).
    │
    ├── reports                    <- Reports, figures, manifests and experiment outputs.
    │   ├── baselines
    │   ├── casting_roi_rd_ae      <- Casting_class1 ROI + RD/Feature-AE documentation.
    │   ├── casting_surface_features
    │   ├── dinomaly
    │   ├── ensemble
    │   ├── figures                <- Presentation and analysis figures.
    │   ├── patchcore
    │   ├── registries
    │   └── tables
    │
    ├── scripts                    <- Utility scripts, dataset lineage and download helpers.
    │
    └── src                        <- Project source code (Python package).
        ├── __init__.py
        ├── config.py              <- Central configuration (paths, EDA parameters).
        │
        ├── data                   <- Data pipeline.
        │   ├── __init__.py
        │   └── harmonize.py       <- MVTec AD + HSS-IAD harmonization → unified CSV.
        │
        ├── features               <- Preprocessing, ROI tools and synthetic defects.
        ├── inference              <- Inference pipelines used by the demo application.
        ├── models                 <- Training, evaluation and model wrappers.
        │   ├── baselines
        │   ├── dinomaly
        │   ├── ensemble
        │   ├── feature_ae         <- RD/Feature-AE pipeline.
        │   ├── patchcore
        │   ├── pixel_ae
        │   └── segmentation       <- Functional-surface ROI segmentation.
        ├── reporting              <- Shared report paths and manifests.
        └── visualization          <- Overlays, previews and heatmaps.

Data Harmonization
-------------------------

Before any analysis, both raw datasets must be manually downloaded
and placed in the following directories:

    data/raw/mvtec/      <- MVTec AD (15 catégories)
    data/raw/hss-iad/    <- HSS-IAD (7 catégories)

Then run the harmonization script from the project root
(using **uv**, which manages the environment and dependencies) :

    uv run python -m src.data.harmonize

The script scans both directory structures (MVTec-style:
`category/{train,test}/{good,defective,...}`), retrieves the
corresponding masks from `ground_truth/` and generates two CSV files in
`data/processed/` :

- `unified_dataset.csv` -  one row per image with the columns
  `dataset`, `category`, `split`, `label`, `is_anomaly`, `image_path`,
  `mask_path`, `has_mask`.
- `resolutions_sample.csv` - sample of 2000 images with `width`,
  `height`, `channels` (used by the EDA notebook).

All paths and scan parameters (image extensions, splits,  
normal label, sample size, seed) are centralized in  
[`src/config.py`](src/config.py).

Once the CSV files are generated, open
[`notebooks/01_eda_harmonisation.ipynb`](notebooks/01_eda_harmonisation.ipynb)
for exploratory analysis.

--------

Application
-------------------------

The project includes a Streamlit demo application with pages for MVTec
benchmarking, live inference, and the specialized `Casting_class1` inspection
pipeline.

Streamlit can be launched from the project root with:

    uv run --extra cu128 streamlit run app/main.py

or, with an already-created virtual environment:

    .\.venv\Scripts\python.exe -m streamlit run app\main.py

Command-Line Pipelines
-------------------------

The main command-line entry points are exposed in `pyproject.toml`:

    uv run --extra cu128 train-roi --help
    uv run --extra cu128 predict-roi --help
    uv run --extra cu128 train-rd-ae --help
    uv run --extra cu128 evaluate-rd-ae --help
    uv run --extra cu128 calibrate-rd-ae --help
    uv run --extra cu128 materialize-quality-heatmaps --help

Casting_class1 Inspection Pipeline
-------------------------

The `Casting_class1` retained path combines:

- functional-surface ROI segmentation;
- RD/Feature-AE anomaly scoring;
- post-hoc soft ROI calibration;
- business decision thresholds: good / to review / defective.

Reference documentation:

- [`reports/casting_roi_rd_ae/README.md`](reports/casting_roi_rd_ae/README.md)
- [`reports/casting_roi_rd_ae/DATASCIENCE_PIPELINE.md`](reports/casting_roi_rd_ae/DATASCIENCE_PIPELINE.md)
- [`reports/casting_roi_rd_ae/DATASET_MASKS_AND_SYNTHETIC_DEFECTS.md`](reports/casting_roi_rd_ae/DATASET_MASKS_AND_SYNTHETIC_DEFECTS.md)
- [`reports/casting_roi_rd_ae/registries/selected_models.md`](reports/casting_roi_rd_ae/registries/selected_models.md)
