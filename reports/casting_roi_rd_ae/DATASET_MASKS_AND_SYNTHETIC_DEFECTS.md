# Casting_class1 - Constitution des masques et defauts synthetiques

Ce document decrit comment preparer les donnees qui alimentent le segmenteur de
surface fonctionnelle `Casting_class1`.

Le but est de produire :

- un dataset de masques semantiques `0/1/2` ;
- une librairie de defauts reels ;
- des previews de defauts synthetiques pour valider visuellement les
  augmentations utilisees pendant l'entrainement.

## 1. Convention des masques

Le segmenteur ROI utilise trois classes :

```text
0 = background / zone non supervisee
1 = surface fonctionnelle inspectable
2 = landmark / geometrie a exclure
```

La classe `1` correspond a la surface metier ou les anomalies sont pertinentes.
La classe `2` sert a retirer des formes structurelles normales qui ressemblent
parfois a des defauts.

## 2. Structure attendue d'un dataset de masques

Un dataset entrainable contient au minimum :

```text
labels_index.csv
masks/
previews/
```

Colonnes importantes de `labels_index.csv` :

```text
image_path
semantic_mask_path
surface_mask_path
landmark_mask_path
positive_mask_path
ignore_mask_path
negative_mask_path
weight_map_path
preview_path
pattern_id
label_source
surface_ratio
landmark_ratio
```

Les previews sont importantes : elles servent aussi de validation manuelle. Dans
le script de rebalancing, supprimer une preview revient a retirer la ligne du
dataset final.

## 3. Construire un dataset surface + landmarks

Point de depart :

- un dossier de labels landmarks corriges manuellement ;
- un dossier de predictions de surface fonctionnelle ;
- typiquement une ROI deja predite sur le train good.

Commande :

```powershell
.\.venv\Scripts\python.exe scripts\datasets\lineage\05_build_surface_landmark_semantic_dataset.py `
  --landmark-labels-dir "data\processed\functional_surface_curated\<landmark_labels_dir>" `
  --surface-predictions-dir "reports\casting_surface_features\<surface_prediction_run>" `
  --output-dir "data\processed\functional_surface_curated\<semantic_dataset_name>" `
  --landmark-dilate-radius 0 `
  --overwrite
```

Exemple de nom de dataset final utilise dans le projet :

```text
Casting_class1_surface_landmark_semantic_v21_epoch014_full435_exclude5_weightbalanced_v1
```

## 4. Corriger les masques avec un checkpoint

Quand certaines lignes sont mieux corrigees par une prediction de modele que par
le masque courant, on peut remplacer les masques selectionnes :

```powershell
.\.venv\Scripts\python.exe scripts\datasets\lineage\06_apply_surface_label_corrections.py `
  --dataset-dir "data\processed\functional_surface_curated\<semantic_dataset_name>" `
  --checkpoint-path "models\functional_surface\Casting_class1\<run_name>\checkpoint_epoch_XXX.pt" `
  --output-dir "data\processed\functional_surface_curated\<corrected_dataset_name>" `
  --selection-csv "reports\casting_surface_features\<selection_file>.csv" `
  --input-size 512 `
  --threshold 0.5 `
  --target-mode binary `
  --preserve-ignore `
  --device auto
```

Alternatives :

```powershell
--image-index 12 18 24
--row-index 3 7 9
```

pour corriger quelques lignes explicitement sans CSV de selection.

## 5. Fusionner plusieurs datasets de labels

Quand on a un dataset de base et un ou plusieurs ajouts :

```powershell
.\.venv\Scripts\python.exe scripts\datasets\lineage\07_merge_surface_label_datasets.py `
  --base-dataset-dir "data\processed\functional_surface_curated\<base_dataset>" `
  --add-dataset-dir "data\processed\functional_surface_curated\<add_dataset>" `
  --output-dir "data\processed\functional_surface_curated\<merged_dataset>" `
  --base-source-name manual `
  --add-source-name pseudo_v20_multiclass_best
```

Le script reconstruit un `labels_index.csv` unique et preserve la tracabilite via
les colonnes `dataset_source`, `dataset_source_dir`, `dataset_source_rank`.

## 6. Rebalancer le dataset

Le rebalancing filtre les lignes dont la preview a ete supprimee puis equilibre
les patterns.

```powershell
.\.venv\Scripts\python.exe scripts\datasets\lineage\08_rebalance_surface_landmark_dataset.py `
  --dataset-dir "data\processed\functional_surface_curated\<merged_dataset>" `
  --output-dir "data\processed\functional_surface_curated\<balanced_dataset>" `
  --group-column pattern_id `
  --manual-source manual `
  --pseudo-source pseudo_v20_multiclass_best `
  --strict-total-balance `
  --overwrite
```

Controle attendu :

```text
chaque pattern conserve un nombre comparable de lignes
les previews restantes correspondent aux masques acceptes
les landmarks ne mangent pas toute la surface fonctionnelle
```

## 7. Construire une librairie de defauts reels

La librairie extrait les composants connectes depuis les masques defectueux.
Elle sert ensuite aux augmentations synthetiques realistes.

Commande pour construire une librairie multi-classes Casting :

```powershell
.\.venv\Scripts\python.exe -m src.features.synthetic_defects.build_library `
  --category "Casting_class1,Casting_class2,Casting_class3" `
  --output-json "reports\tables\summary\casting_all_defect_patch_library.json" `
  --min-area 12 `
  --max-components-per-mask 12
```

Pour une seule classe :

```powershell
.\.venv\Scripts\python.exe -m src.features.synthetic_defects.build_library `
  --category Casting_class1 `
  --output-json "reports\tables\summary\casting_class1_defect_patch_library.json"
```

Chemins par defaut lus par le script :

```text
data/raw/hss-iad/<category>/test/defective
data/raw/hss-iad/<category>/ground_truth/defective
```

## 8. Librairies texture et photometrie

Le training V36 peut utiliser trois artefacts :

```text
reports/tables/summary/casting_all_defect_patch_library.json
reports/casting_surface_features/defect_synthetic_study/clustered_texture_library_casting_all/clustered_defect_texture_library.json
reports/casting_surface_features/defect_synthetic_study/photometric_coherence_library.json
```

La premiere librairie est generable par `build_library.py`.

Les deux autres servent a conditionner :

- la texture locale des defauts synthetiques ;
- la coherence lumiere/contraste avec les defauts reels.

Si elles ne sont pas presentes dans le repo officiel, il faut les rapatrier depuis
le workspace de recherche ou les regenerer avec les scripts d'etude dedies avant
de relancer strictement le training V36.

## 9. Previsualiser les defauts synthetiques

Avant entrainement, verifier visuellement que les defauts synthetiques sont
plausibles sur chaque pattern.

```powershell
.\.venv\Scripts\python.exe -m src.features.synthetic_defects.preview `
  --labels-dir "data\processed\functional_surface_curated\<balanced_dataset>" `
  --defect-library-json "reports\tables\summary\casting_all_defect_patch_library.json" `
  --texture-library-json "reports\casting_surface_features\defect_synthetic_study\clustered_texture_library_casting_all\clustered_defect_texture_library.json" `
  --photometric-library-json "reports\casting_surface_features\defect_synthetic_study\photometric_coherence_library.json" `
  --output-dir "reports\casting_surface_features\defect_synthetic_study\preview_v3_contextual" `
  --seed 292 `
  --thumb-size 220
```

Si les librairies texture/photometrie sont absentes, le preview peut encore aider
a verifier les familles procedurales, mais il ne represente plus exactement le
training V36.

## 10. Diagnostiquer real vs synthetic

Comparer statistiquement defauts reels et synthetiques :

```powershell
.\.venv\Scripts\python.exe -m src.features.synthetic_defects.diagnostics `
  --labels-dir "data\processed\functional_surface_curated\<balanced_dataset>" `
  --defect-library-json "reports\tables\summary\casting_all_defect_patch_library.json" `
  --texture-library-json "reports\casting_surface_features\defect_synthetic_study\clustered_texture_library_casting_all\clustered_defect_texture_library.json" `
  --photometric-library-json "reports\casting_surface_features\defect_synthetic_study\photometric_coherence_library.json" `
  --output-dir "reports\casting_surface_features\defect_synthetic_study\real_vs_synthetic_comparison_v3"
```

Sorties attendues :

```text
real_vs_synthetic_component_metrics.csv
real_vs_synthetic_summary.csv
real_vs_synthetic_summary_by_pattern.csv
real_vs_synthetic_metric_boxplots.png
real_vs_synthetic_metric_atlas.png
diagnostic_real_vs_synthetic.md
```

## 11. Utilisation dans le training du segmenteur

Les librairies sont passees au training avec :

```powershell
--synthetic-defect-library-json "reports\tables\summary\casting_all_defect_patch_library.json"
--synthetic-defect-texture-library-json "reports\casting_surface_features\defect_synthetic_study\clustered_texture_library_casting_all\clustered_defect_texture_library.json"
--synthetic-defect-photometric-library-json "reports\casting_surface_features\defect_synthetic_study\photometric_coherence_library.json"
```

Parametres importants :

```text
--synthetic-defect-p 0.6
--synthetic-defect-mode mixed
--synthetic-defect-realistic-render residual
--synthetic-defect-pattern-aware
--synthetic-defect-context-consistent
--synthetic-defect-crop-localized
--synthetic-defect-min-surface-overlap 0.86
```

L'objectif est que le modele voie des defauts plausibles dans l'image locale et
dans le contexte global, tout en gardant la cible de segmentation stable.

## 12. Checklist qualite

Avant entrainement :

```text
[ ] labels_index.csv existe
[ ] les chemins image_path et semantic_mask_path sont valides
[ ] les previews ont ete inspectees
[ ] les lignes douteuses ont ete supprimees ou corrigees
[ ] chaque pattern_id est represente
[ ] la librairie de defauts contient assez de composants
[ ] les previews synthetiques ne ressemblent pas a du bruit artificiel
[ ] les defauts synthetiques restent dans la surface fonctionnelle
```

