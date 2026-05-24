# Casting_class1 - Pipeline ROI + RD Feature-AE

Ce dossier documente le pipeline specialise `Casting_class1` utilise pour l'inspection :

1. segmentation de la surface fonctionnelle ;
2. detection d'anomalies par RD/Feature-AE ;
3. heatmap inspecteur ;
4. decision metier : `bonne`, `a verifier`, `defectueuse`.

## Artefacts necessaires

Les poids sont locaux et ne sont pas versionnes par Git.

```text
models/functional_surface/Casting_class1/Casting_class1_v36_direct_imagenet_512_1024_synthv4_photometric_c123/checkpoint_epoch_028.pt
models/feature_ae/Casting_class1/Casting_class1_reverse_distill_ms_dualcontext_v36roi_e28_layer23_metricbest_v1/checkpoint_epoch_002.pt
```

Pour utiliser la galerie Streamlit, les images test doivent etre presentes ici :

```text
data/raw/hss-iad/Casting_class1/test/good
data/raw/hss-iad/Casting_class1/test/defective
```

## Lancement Streamlit

Depuis la racine du repo :

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\main.py
```

Ouvrir ensuite :

```text
http://localhost:8501
```

La page a utiliser est `Casting_class1`.

## Inference directe

La page Streamlit charge :

- le segmenteur ROI champion ;
- le RD/Feature-AE champion ;
- une image upload ou une image de la galerie test.

Elle affiche :

- image originale ;
- ROI fonctionnelle ;
- heatmap anomalie ;
- signal affiche ;
- score anomalie ;
- decision metier.

## Decision metier

Les seuils calibres par defaut sont :

```text
bonne       : score <= 0.425
a verifier  : 0.425 < score < 0.515
defectueuse : score >= 0.515
```

Ces seuils sont prudents : la zone jaune absorbe le recouvrement observe entre
scores `good` et `defective`.

Trace de calibration :

```text
reports/casting_roi_rd_ae/calibration/Casting_class1_rd_ae_v36_test/decision_thresholds.json
```

## Reglages Streamlit

Les reglages visuels sont recalcules sans relancer l'inference lourde :

```text
Seuil affichage      : quantite de signal visible
Percentile bas       : coupe le bruit faible avant normalisation
Percentile haut      : contraste des pics forts
Gamma                : progressivite de la heatmap
Opacite overlay      : intensite visuelle sur l'image
Seuil ROI            : surface fonctionnelle retenue
Fraction top-k score : pixels les plus suspects utilises pour le score image
```

## Metriques de reference

Calibration soft ROI recalculee :

```text
image AUROC : 0.8684
image AP    : 0.9221
pixel AUROC : 0.9814
pixel AP    : 0.3775
AUPIMO      : 0.1415
```

## Depannage

Si Streamlit semble garder une ancienne version du pipeline :

```powershell
Ctrl+C
.\.venv\Scripts\python.exe -m streamlit run app\main.py
```

Si une image ne s'affiche pas dans la galerie, verifier que les dossiers test
`data/raw/hss-iad/Casting_class1/test/...` existent bien.

