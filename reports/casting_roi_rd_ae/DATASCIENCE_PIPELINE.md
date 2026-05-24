# Casting_class1 - Guide datascience du pipeline ROI + RD Feature-AE

Ce guide regroupe les commandes utiles pour reconstruire, evaluer et calibrer le
pipeline `Casting_class1`.

## 1. Pre-requis

Depuis la racine du repo :

```powershell
cd "D:\repos_officiel\mar26_bds_anomalies_pieces_indus"
```

Verifier les poids locaux :

```powershell
Test-Path "models\functional_surface\Casting_class1\Casting_class1_v36_direct_imagenet_512_1024_synthv4_photometric_c123\checkpoint_epoch_028.pt"
Test-Path "models\feature_ae\Casting_class1\Casting_class1_reverse_distill_ms_dualcontext_v36roi_e28_layer23_metricbest_v1\checkpoint_epoch_002.pt"
```

Verifier les donnees utiles :

```powershell
Test-Path "data\processed\unified_dataset.csv"
Test-Path "data\raw\hss-iad\Casting_class1"
Test-Path "data\classified\Casting_class1"
```

Pour la constitution du dataset de masques et des librairies de defauts
synthetiques, voir aussi :

```text
reports/casting_roi_rd_ae/DATASET_MASKS_AND_SYNTHETIC_DEFECTS.md
```

## 2. Training du segmenteur ROI

Le segmenteur ROI apprend une carte semantique `0/1/2` :

```text
0 = background
1 = surface fonctionnelle
2 = landmark / exclusion
```

Commande representative de la configuration V36 :

```powershell
.\.venv\Scripts\python.exe -m src.models.segmentation.training `
  --category Casting_class1 `
  --labels-dir "data\processed\functional_surface_curated\Casting_class1_surface_landmark_semantic_v21_epoch014_full435_exclude5_weightbalanced_v1" `
  --semantic-mask-column semantic_mask_path `
  --model-type functional_unet_resnet18_det1_context2b `
  --num-classes 3 `
  --input-size 512 `
  --context-size 1024 `
  --epochs 100 `
  --batch-size 4 `
  --learning-rate 3e-05 `
  --weight-decay 0.0001 `
  --val-fraction 0.2 `
  --split-strategy stratified `
  --split-column pattern_id `
  --split-seed 42 `
  --ce-weight 1.0 `
  --dice-weight 1.0 `
  --dice-classes "1,2" `
  --class-weights auto `
  --augmentation-profile functional_surface_full_source_768_to_384_denoise `
  --repeat-factor 16 `
  --context-crop-prob 1.0 `
  --positive-crop-prob 0.5 `
  --train-photometric-normalization-p 0.25 `
  --train-photo-target-p05 0.03 `
  --train-photo-target-p95 0.6 `
  --synthetic-defect-p 0.6 `
  --synthetic-defect-mode mixed `
  --synthetic-defect-realistic-render residual `
  --synthetic-defect-library-json "reports\tables\summary\casting_all_defect_patch_library.json" `
  --synthetic-defect-texture-library-json "reports\casting_surface_features\defect_synthetic_study\clustered_texture_library_casting_all\clustered_defect_texture_library.json" `
  --synthetic-defect-photometric-library-json "reports\casting_surface_features\defect_synthetic_study\photometric_coherence_library.json" `
  --synthetic-defect-pattern-aware `
  --synthetic-defect-p4-large-p 0.88 `
  --synthetic-defect-max-blobs 4 `
  --synthetic-defect-min-radius-frac 0.012 `
  --synthetic-defect-max-radius-frac 0.055 `
  --synthetic-defect-shape-weights "machined:0.55,scratch:0.20,hole:0.15,stain:0.10" `
  --synthetic-defect-scratch-min-length-frac 0.08 `
  --synthetic-defect-scratch-max-length-frac 0.45 `
  --synthetic-defect-scratch-p 0.3 `
  --synthetic-defect-texture-strength 0.7 `
  --synthetic-defect-variant-strength 1.0 `
  --synthetic-defect-large-p 0.5 `
  --synthetic-defect-large-quantile 0.7 `
  --synthetic-defect-large-scale-min 1.15 `
  --synthetic-defect-large-scale-max 2.1 `
  --synthetic-defect-alpha-min 0.5 `
  --synthetic-defect-alpha-max 0.9 `
  --synthetic-defect-bg-match-strength 0.45 `
  --synthetic-defect-min-surface-overlap 0.86 `
  --synthetic-defect-context-consistent `
  --synthetic-defect-crop-localized `
  --num-workers 0 `
  --device auto `
  --save-best `
  --lr-scheduler plateau `
  --lr-patience 6 `
  --lr-factor 0.5 `
  --early-stopping-patience 14 `
  --min-delta 0.0001 `
  --checkpoint-every-epochs 1 `
  --save-previews `
  --preview-count 24 `
  --preview-head mask `
  --run-name "Casting_class1_v36_direct_imagenet_512_1024_synthv4_photometric_c123_rerun" `
  --output-dir "models\functional_surface"
```

### Options de donnees

```text
--category                 categorie entrainee
--labels-dir               dataset de masques semantiques
--semantic-mask-column     colonne du labels_index.csv contenant le masque 0/1/2
--limit-train              smoke test rapide sur N lignes
```

### Architecture

```text
--model-type               variante U-Net/ResNet18
--num-classes              3 pour background/surface/landmark
--input-size               taille locale du crop modele
--context-size             branche contexte globale
--init-checkpoint-path     fine-tuning depuis un checkpoint existant
```

### Split et pertes

```text
--val-fraction             proportion validation
--split-strategy           random ou stratified
--split-column             pattern_id pour garder les patterns equilibres
--ce-weight                poids cross-entropy
--dice-weight              poids Dice
--recon-weight             perte auxiliaire si le modele expose recon_logits
--dice-classes             classes incluses dans Dice, ex. "1,2"
--class-weights            auto, none ou poids explicites
```

### Augmentations spatiales et photometriques

```text
--augmentation-profile                 profil centralise dans src/features/augmentation_profiles.py
--repeat-factor                        repetition virtuelle du dataset
--context-crop-prob                    probabilite d'utiliser le crop contexte
--positive-crop-prob                   probabilite de forcer un crop contenant de la surface positive
--train-photometric-normalization-p    probabilite de normalisation photometrique
--train-photo-target-p05               percentile bas cible
--train-photo-target-p95               percentile haut cible
```

Le profil V36 `functional_surface_full_source_768_to_384_denoise` combine :

```text
workspace 768
crops synchronises image/masque
flips horizontal/vertical
rotations faibles
jitter luminosite/contraste/saturation
leger flou aleatoire
normalisation photometrique partielle
```

### Defauts synthetiques

```text
--synthetic-defect-p                         probabilite d'injecter un defaut
--synthetic-defect-mode                      generic, realistic ou mixed
--synthetic-defect-realistic-render          paste ou residual
--synthetic-defect-library-json              composants reels extraits des masques
--synthetic-defect-texture-library-json      textures clusterisees
--synthetic-defect-photometric-library-json  profils lumiere/contraste
--synthetic-defect-pattern-aware             adapte la generation au pattern_id
--synthetic-defect-p4-large-p                favorise les grands defauts sur P4
--synthetic-defect-max-blobs                 nombre max de composants synthetiques
--synthetic-defect-min/max-radius-frac       taille relative des blobs
--synthetic-defect-shape-weights             poids hole/scratch/stain/machined
--synthetic-defect-scratch-*                 longueur/probabilite des rayures
--synthetic-defect-texture-strength          intensite texture
--synthetic-defect-variant-strength          variation des defauts realistes
--synthetic-defect-large-*                   sampling des gros composants
--synthetic-defect-alpha-min/max             opacite du defaut
--synthetic-defect-bg-match-strength         adaptation au fond local
--synthetic-defect-min-surface-overlap       fraction minimale dans la surface fonctionnelle
--synthetic-defect-context-consistent        applique le defaut avant crop local/contexte
--synthetic-defect-crop-localized            centre le defaut dans le crop local supervise
```

### Monitoring et checkpoints

```text
--external-monitor-labels-dir     dataset externe jamais utilise pour train
--external-monitor-name           prefixe des colonnes de monitoring
--best-monitor                    val_loss ou external_loss
--save-best                       ecrit checkpoint_best.pt
--lr-scheduler                    none ou plateau
--lr-patience                     patience scheduler
--lr-factor                       facteur de reduction LR
--early-stopping-patience         arret anticipe
--min-delta                       gain minimal considere utile
--checkpoint-every-epochs         frequence checkpoint_epoch_XXX.pt
--save-previews                   sauvegarde previews validation
--preview-count                   nombre de previews
--preview-head                    mask ou recon
--overwrite-run                   autorise ecrasement dossier existant
--no-progress                     desactive tqdm
```

### Commande d'aide officielle

```powershell
.\.venv\Scripts\python.exe -m src.models.segmentation.training --help
```

## 3. Segmentation ROI

### Predict ROI sur le test set good

```powershell
.\.venv\Scripts\python.exe -m src.models.segmentation.prediction `
  --checkpoint-path "models\functional_surface\Casting_class1\Casting_class1_v36_direct_imagenet_512_1024_synthv4_photometric_c123\checkpoint_epoch_028.pt" `
  --image-dir "data\raw\hss-iad\Casting_class1\test\good" `
  --split test `
  --label good `
  --output-dir "reports\casting_surface_features\Casting_class1_v36_e28_test_good_previews" `
  --input-size 512 `
  --context-size 1024 `
  --device auto `
  --mask-output-size original `
  --preview-per-pattern 999 `
  --overwrite
```

### Predict ROI sur le test set defective

```powershell
.\.venv\Scripts\python.exe -m src.models.segmentation.prediction `
  --checkpoint-path "models\functional_surface\Casting_class1\Casting_class1_v36_direct_imagenet_512_1024_synthv4_photometric_c123\checkpoint_epoch_028.pt" `
  --image-dir "data\raw\hss-iad\Casting_class1\test\defective" `
  --split test `
  --label defective `
  --output-dir "reports\casting_surface_features\Casting_class1_v36_e28_test_defective_previews" `
  --input-size 512 `
  --context-size 1024 `
  --device auto `
  --mask-output-size original `
  --preview-per-pattern 999 `
  --overwrite
```

Le run complet deja utilise par le pipeline RD AE est :

```text
reports/casting_surface_features/Casting_class1_v36_e28_test_roi_full
```

Il doit contenir :

```text
functional_surface_predictions.npz
prediction_summary.csv
masks/
prob_maps/
previews/
```

## 4. Training RD/Feature-AE

Le modele RD/Feature-AE est entraine sur les images normales. La segmentation ROI
sert a concentrer la perte sur la surface fonctionnelle et a reduire le bruit du
fond ou des zones hors metier.

Commande representative du run champion conserve :

```powershell
.\.venv\Scripts\python.exe -m src.models.feature_ae.training `
  --category Casting_class1 `
  --model-type reverse_distill_resnet18_dual_context_gated `
  --teacher-backbone resnet18 `
  --layers layer2 layer3 `
  --input-size 384 `
  --preprocessing-mode tile_256_overlap `
  --tile-size 384 `
  --context-tile-size 768 `
  --tile-train-stride 192 `
  --tile-train-sampling all `
  --epochs 14 `
  --batch-size 16 `
  --learning-rate 5e-5 `
  --weight-decay 1e-4 `
  --loss l2_cosine `
  --cosine-weight 0.5 `
  --layer-loss-weights "layer2=0.65" "layer3=0.35" `
  --augmentation-profile none `
  --repeat-factor 2 `
  --val-fraction 0.15 `
  --roi-predictions-dir "reports\casting_surface_features\Casting_class1_v36_e28_train_good_roi_full" `
  --roi-threshold 0.30 `
  --roi-loss-weight 1.0 `
  --background-loss-weight 0.02 `
  --min-roi-ratio 0.03 `
  --lr-scheduler plateau `
  --lr-patience 4 `
  --lr-factor 0.5 `
  --early-stopping-patience 6 `
  --save-best `
  --checkpoint-every-epochs 1 `
  --metric-eval-every-epochs 1 `
  --metric-eval-start-epoch 1 `
  --metric-eval-category Casting_class1 `
  --metric-eval-device cuda `
  --metric-eval-batch-size 8 `
  --metric-eval-tile-stride 192 `
  --metric-eval-layer-weights "layer2=0.65" "layer3=0.35" `
  --metric-eval-calibrate-normal `
  --metric-eval-calibration-mode per_layer `
  --metric-eval-calibration-stat median_mad `
  --metric-eval-calibration-max-images 120 `
  --metric-eval-score-region functional_surface_prediction `
  --metric-eval-roi-predictions-dir "reports\casting_surface_features\Casting_class1_v36_e28_train_good_roi_full" "reports\casting_surface_features\Casting_class1_v36_e28_test_roi_full" `
  --metric-eval-roi-threshold 0.30 `
  --metric-eval-apply-score-region-to-map `
  --metric-eval-score-smoothing median3 `
  --metric-eval-score-image topk_mean `
  --metric-eval-topk-fraction 0.005 `
  --metric-eval-save-score-maps `
  --metric-eval-save-previews `
  --metric-eval-max-previews 31 `
  --device cuda `
  --run-name "Casting_class1_reverse_distill_ms_dualcontext_v36roi_e28_layer23_metricbest_v1" `
  --output-dir "models\feature_ae"
```

### Options de donnees

```text
--category / --categories / --all-categories  categorie(s) entrainees
--repeat-factor                                repetition virtuelle des normales
--val-fraction                                 split validation sur les normales
--init-checkpoint-path                         reprise/fine-tuning depuis un checkpoint
```

### Architecture et features

```text
--model-type          feature_ae ou reverse_distill, simple ou dual_context_gated
--teacher-backbone    backbone teacher, ici resnet18
--layers              couches distillees, ici layer2 et layer3
--input-size          taille locale modele
--normalization       normalisation image, par defaut imagenet
```

### Tiling et contexte

```text
--preprocessing-mode              letterbox ou tile_256_overlap
--tile-size                       taille du crop local
--context-tile-size               contexte large de la branche dual-context
--tile-train-stride               stride en training
--tile-train-sampling             all ou random
--tile-train-max-tiles-per-image  limite de tuiles par image si sampling contraint
```

### Pertes, couches et augmentations

```text
--loss                  perte principale, ici l2_cosine
--cosine-weight         poids du terme cosine
--layer-loss-weights    ponderation layer2/layer3
--augmentation-profile  none, default, toothbrush, toothbrush_headprior,
                        casting_microdefect ou auto
```

Pour le run champion, les augmentations RD/Feature-AE sont desactivees
(`--augmentation-profile none`) afin de garder une representation normale stable.
Les variations fortes sont portees par le segmenteur et par les defauts
synthetiques, pas par le detecteur d'anomalies.

### Perte guidee par ROI

```text
--roi-predictions-dir       predictions ROI sur le train good
--roi-threshold             seuil de binarisation ROI pour la perte
--roi-loss-weight           poids de la surface fonctionnelle
--background-loss-weight    poids residuel hors ROI
--roi-dilate-radius         dilatation optionnelle de la ROI
--min-roi-ratio             filtre les tuiles avec trop peu de surface utile
```

Le dossier ROI train attendu est :

```text
reports/casting_surface_features/Casting_class1_v36_e28_train_good_roi_full
```

Il doit contenir `functional_surface_predictions.npz`. Pour le monitoring
metrique pendant l'entrainement, le run utilise aussi la ROI test :

```text
reports/casting_surface_features/Casting_class1_v36_e28_test_roi_full
```

### Evaluation metrique pendant training

```text
--metric-eval-every-epochs              frequence evaluation complete
--metric-eval-start-epoch               premiere epoch evaluee
--metric-eval-category                  categorie evaluee
--metric-eval-layer-weights             poids layer2/layer3 pour scorer
--metric-eval-calibrate-normal          calibration par statistiques normales
--metric-eval-calibration-mode          global ou per_layer
--metric-eval-calibration-stat          median_mad ou autre statistique supportee
--metric-eval-score-region              region de score, ici functional_surface_prediction
--metric-eval-roi-predictions-dir       ROI train + ROI test
--metric-eval-apply-score-region-to-map applique la ROI aussi a la carte pixel
--metric-eval-score-smoothing           lissage, ici median3
--metric-eval-score-image               agrégation image, ici topk_mean
--metric-eval-topk-fraction             fraction des pixels les plus anormaux
--metric-eval-save-score-maps           sauvegarde des cartes score
--metric-eval-save-previews             sauvegarde des previews
```

Le checkpoint retenu est `checkpoint_epoch_002.pt`, selectionne sur les metriques
metier plutot que sur la seule `val_loss`. Sur le suivi historique, cette epoch
donne environ :

```text
image AP  = 0.9237
pixel AP  = 0.2860
AUPIMO    = 0.0967
```

### Monitoring et checkpoints

```text
--lr-scheduler              none ou plateau
--lr-patience               patience scheduler
--lr-factor                 facteur de reduction LR
--early-stopping-patience   arret anticipe
--save-best                 sauvegarde checkpoint_best.pt
--checkpoint-every-epochs   sauvegarde checkpoint_epoch_XXX.pt
--run-name                  nom du dossier de sortie
--output-dir                racine des poids Feature-AE
```

### Commande d'aide officielle

```powershell
.\.venv\Scripts\python.exe -m src.models.feature_ae.training --help
```

## 5. Evaluation RD/Feature-AE brute

```powershell
.\.venv\Scripts\python.exe -m src.models.feature_ae.evaluation `
  --category Casting_class1 `
  --checkpoint-path "models\feature_ae\Casting_class1\Casting_class1_reverse_distill_ms_dualcontext_v36roi_e28_layer23_metricbest_v1\checkpoint_epoch_002.pt" `
  --output-dir "reports\casting_roi_rd_ae\evaluation" `
  --run-name "Casting_class1_rd_ae_v36_test" `
  --input-size 384 `
  --preprocessing-mode tile_256_overlap `
  --tile-size 384 `
  --context-tile-size 768 `
  --tile-stride 384 `
  --layers layer2 layer3 `
  --layer-weights "layer2=0.65" "layer3=0.35" `
  --teacher-backbone resnet18 `
  --cosine-weight 0.5 `
  --score-region functional_surface_prediction `
  --roi-predictions-dir "reports\casting_surface_features\Casting_class1_v36_e28_test_roi_full" `
  --roi-threshold 0.5 `
  --score-smoothing median3 `
  --score-image topk_mean `
  --topk-fraction 0.005 `
  --batch-size 1 `
  --num-workers 0 `
  --device cpu `
  --save-score-maps
```

Sortie principale :

```text
reports/casting_roi_rd_ae/evaluation/Casting_class1/Casting_class1_rd_ae_v36_test/predictions.npz
```

## 6. Calibration post-hoc

```powershell
.\.venv\Scripts\python.exe -m src.models.feature_ae.calibrate_matrix `
  --predictions "reports\casting_roi_rd_ae\evaluation\Casting_class1\Casting_class1_rd_ae_v36_test\predictions.npz" `
  --roi-predictions-dir "reports\casting_surface_features\Casting_class1_v36_e28_test_roi_full" `
  --output-dir "reports\casting_roi_rd_ae\calibration\Casting_class1_rd_ae_v36_test" `
  --layers layer2 layer3 `
  --layer-weights "layer2=0.65,layer3=0.35" `
  --topk-fractions 0.005 `
  --smoothing median3 `
  --roi-modes soft_map `
  --roi-thresholds 0.5 `
  --batch-size 1
```

Attention : `calibrate_matrix` attend les poids de couches sous forme d'une seule chaine :

```powershell
--layer-weights "layer2=0.65,layer3=0.35"
```

## 7. Materialisation des heatmaps calibrees

```powershell
.\.venv\Scripts\python.exe -m src.models.feature_ae.materialize `
  --predictions "reports\casting_roi_rd_ae\evaluation\Casting_class1\Casting_class1_rd_ae_v36_test\predictions.npz" `
  --roi-predictions-dir "reports\casting_surface_features\Casting_class1_v36_e28_test_roi_full" `
  --output-dir "reports\casting_roi_rd_ae\heatmaps\Casting_class1_rd_ae_v36_test_calibrated" `
  --layers layer2 layer3 `
  --layer-weights "layer2=0.65,layer3=0.35" `
  --smoothing median3 `
  --roi-mode soft_map `
  --roi-threshold 0.5 `
  --topk-fraction 0.005 `
  --batch-size 1
```

## 8. Previews heatmap

Version stricte inspecteur :

```powershell
.\.venv\Scripts\python.exe -m src.models.feature_ae.compare_heatmaps `
  --run "RD AE V36 calibrated strict" "reports\casting_roi_rd_ae\heatmaps\Casting_class1_rd_ae_v36_test_calibrated\predictions.npz" `
  --output-dir "reports\casting_roi_rd_ae\heatmaps\Casting_class1_rd_ae_v36_test_strict_previews" `
  --max-items 40 `
  --panel-size 220 `
  --score-min-percentile 85 `
  --score-max-percentile 99.8 `
  --score-gamma 1.4 `
  --overlay-alpha 0.72 `
  --display-threshold 0.60 `
  --heatmap-palette orange `
  --sheet-cols 4 `
  --sheet-max-defective 20 `
  --sheet-max-good 12 `
  --inspector-mode
```

Pour afficher plus de signal, baisser :

```powershell
--display-threshold 0.35
```

## 9. Seuils de decision

Seuils calibres par defaut pour Streamlit :

```text
bonne       : score <= 0.425
a verifier  : 0.425 < score < 0.515
defectueuse : score >= 0.515
```

Trace :

```text
reports/casting_roi_rd_ae/calibration/Casting_class1_rd_ae_v36_test/decision_thresholds.json
```

## 10. Commandes de controle

Verifier les CLIs :

```powershell
.\.venv\Scripts\python.exe -m src.models.segmentation.prediction --help
.\.venv\Scripts\python.exe -m src.models.segmentation.training --help
.\.venv\Scripts\python.exe -m src.models.feature_ae.training --help
.\.venv\Scripts\python.exe -m src.models.feature_ae.evaluation --help
.\.venv\Scripts\python.exe -m src.models.feature_ae.calibrate_matrix --help
.\.venv\Scripts\python.exe -m src.models.feature_ae.materialize --help
.\.venv\Scripts\python.exe -m src.models.feature_ae.compare_heatmaps --help
```

Compiler rapidement les modules touches :

```powershell
.\.venv\Scripts\python.exe -m compileall -q src\inference src\models\feature_ae src\models\segmentation app
```
