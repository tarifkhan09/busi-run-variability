# Results — breast ultrasound explanation fidelity

- preprocessing variant: **rescale** (1/255 rescale, the draft setup)
- checkpoints: `/content/drive/MyDrive/BreastCancerNew/results/models`
- test set: 98 images (66 benign / 32 malignant)
- lesion area ratio: 38.43% benign / 42.49% malignant (matches the draft's 38.43 / 42.49)
- runtime: 16.4 min

## Table III — backbone ablation, fine-tune arm

| Backbone    |   AUROC | AUROC_CI         |   Acc |    Sen |    Spe |   FN |
|:------------|--------:|:-----------------|------:|-------:|-------:|-----:|
| DenseNet121 |  0.9995 | [0.9972, 1.0000] | 98.98 |  96.88 | 100    |    1 |
| VGG16       |  0.9995 | [0.9972, 1.0000] | 98.98 | 100    |  98.48 |    0 |
| Xception    |  0.9995 | [0.9972, 1.0000] | 96.94 | 100    |  95.45 |    0 |
| InceptionV3 |  0.9915 | [0.9768, 1.0000] | 96.94 |  93.75 |  98.48 |    2 |
| ResNet50    |  0.9792 | [0.9522, 0.9953] | 90.82 |  84.38 |  93.94 |    5 |

AUROC intervals are stratified bootstrap, 2000 resamples.

## Table IV — Grad-CAM++ fidelity, DenseNet121

| Configuration     |   Accuracy |   AIL_mean |   AIL_std |   IoU_mean |   IoU_std |    PG | PG_CI          |   PG_benign |   PG_malignant |   Oracle_AIL |   Oracle_IoU |   AIL_share |   IoU_share |
|:------------------|-----------:|-----------:|----------:|-----------:|----------:|------:|:---------------|------------:|---------------:|-------------:|-------------:|------------:|------------:|
| feature_extractor |     100    |      49.86 |      9.92 |      52.64 |     11.73 | 91.84 | [84.71, 95.81] |       89.39 |          96.88 |        80.61 |        89.59 |        61.9 |        58.8 |
| finetune          |      98.98 |      51.91 |      9.22 |      59.74 |      9.97 | 96.94 | [91.38, 98.95] |       98.48 |          93.75 |        80.61 |        89.59 |        64.4 |        66.7 |

Chance baseline for PG and AIL is the lesion area ratio, 40.46%. PG intervals are Wilson.

## Dissociation test

McNemar exact on pointing-game hits: b=7, c=2, p=1.797e-01

| | draft | this run |
|---|---|---|
| feature-extraction PG | 43.88% | 91.84% |
| fine-tune PG | 96.94% | 96.94% |
| McNemar p | < 1e-15 | 1.797e-01 |

**DOES NOT REPRODUCE - both arms localise alike. The draft's 53-point gap was dependent on something that changed; the headline claim needs reframing.**

## Scope

Table III uses the fine-tune arm of five backbones; Table IV uses both DenseNet121 arms. External validation, the ensemble and the remaining fidelity arms are out of scope for this run.

## Limitation stated up front

Preprocessing masks and crops each image using its ground-truth segmentation, so the pipeline assumes an oracle segmenter and is not a deployable end-to-end CAD system. It is a controlled protocol in which the reference region is known exactly, which is what makes the fidelity indices measurable. The lesion area ratio is therefore reported as the chance baseline for PG and AIL.