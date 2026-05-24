# Casting_class1 retained configuration

This note documents the retained configuration for the Casting_class1 ROI + RD AE path.
The wording is intentionally descriptive: these artefacts are the final retained configuration for the project story, not a claim of universal superiority.

## Functional-surface ROI

- Run: `Casting_class1_v36_direct_imagenet_512_1024_synthv4_photometric_c123`
- Local checkpoint: `checkpoint_epoch_028.pt`
- Role: high-resolution ROI for machined/functional surfaces.
- Coverage on test defects: defect coverage mean `0.9985`, pixel weighted coverage `0.9977`.

## RD AE

- Run: `Casting_class1_reverse_distill_ms_dualcontext_v36roi_e28_layer23_metricbest_v1`
- Local checkpoint: `checkpoint_epoch_002.pt`
- Layers: `layer2 + layer3`
- Raw test-set evaluation rerun: image AUROC `0.8114`, image AP `0.8789`.
- The raw run writes `predictions.npz`; localization metrics are reported after the post-hoc calibration step below.

## Post-hoc calibration and display

- Soft ROI calibration rerun: image AUROC `0.8684`, image AP `0.9221`, pixel AUROC `0.9814`, pixel AP `0.3775`, AUPIMO `0.1415`.
- Calibration parameters: `layer2=0.65`, `layer3=0.35`, `median3`, `soft_map`, ROI threshold `0.5`, top-k fraction `0.005`.
- Inspector display threshold `0.38` belongs to the previous rendered heatmap package and should be regenerated before reporting thresholded display metrics with the rerun calibration.
