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
    ├── data                       <- Datasets (local only, excluded from Git).
    │   ├── raw                    <- Original, immutable data dump.
    │   │   ├── mvtec              <- MVTec AD (15 categories).
    │   │   └── hss-iad            <- HSS-IAD (7 categories).
    │   └── processed              <- Harmonized data ready for use.
    │       ├── unified_dataset.csv
    │       └── resolutions_sample.csv
    │
    ├── models                     <- Trained models, predictions, checkpoints.
    │
    ├── notebooks                  <- Jupyter notebooks (ordered numerically).
    │   ├── 01_eda_harmonisation.ipynb
    │
    ├── references                 <- Documentation, links, research papers (e.g., HSS-IAD).
    │
    ├── reports                    <- Reports generated from analyses.
    │   ├── eda_report.md
    │   └── figures
    │       └── eda                <- Figures extracted from the EDA notebook.
    │
    └── src                        <- Project source code (Python package).
        ├── __init__.py
        ├── config.py              <- Central configuration (paths, EDA parameters).
        │
        ├── data                   <- Data pipeline.
        │   ├── __init__.py
        │   └── harmonize.py       <- MVTec AD + HSS-IAD harmonization → unified CSV.
        │
        ├── features               <- (coming soon) feature extraction.
        ├── models                 <- (coming soon) training and prediction.
        ├── visualization          <- (coming soon) visualizations.
        └── streamlit              <- (coming soon) demo application.

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