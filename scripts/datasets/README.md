# Dataset Lineage CLI

Ce dossier ne contient plus l'atelier complet de construction des masques. Pour la publication, il garde seulement le lineage compact permettant de comprendre et reconstruire les etapes critiques :

- classification piece/pattern ;
- materialisation des datasets routeurs ;
- construction des masques semantic surface/landmark ;
- corrections et fusion des labels ;
- contact sheets de controle.

## Sequence Publiable

- `01_materialize_casting_class1_patterns.py`
- `02_materialize_casting_class2_class3_patterns.py`
- `03_build_pattern_router_dataset.py`
- `04_build_piece_router_dataset.py`
- `05_build_surface_landmark_semantic_dataset.py`
- `06_apply_surface_label_corrections.py`
- `07_merge_surface_label_datasets.py`
- `08_rebalance_surface_landmark_dataset.py`

Les explorations historiques, priors remplaces, filtres ponctuels et variantes V15/V18 sont archives dans `_archive/2026-05-23_repo_refinement/scripts/datasets_experimental/` et indexes dans `reports/registries/scripts_dataset_lineage_manifest.csv`.
