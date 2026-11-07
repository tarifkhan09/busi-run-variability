# A1 diagnostic result (2026-11-06, Colab T4, TF 2.21)

Run-B VGG16-PF checkpoint (`results/models/VGG16_feature_extractor_best.keras`, trained on [0,1] input),
scored on the same 98 test images:

| Test input | AUROC | Accuracy | FP | FN |
|---|---|---|---|---|
| rescaled 1/255 (correct) | 0.9991 | 97.96% | 0 | 2 |
| unscaled 0–255 | 0.7879 | 52.04% | 47 | 0 |

Run A reported VGG16-PF at 50.00% with 49 FP / 0 FN although its validation accuracy was 95.88%.
Feeding unscaled images reproduces that pattern (everything called malignant), so run A's VGG16-PF
(and likely Inc-FT and VGG-FT) test numbers are most plausibly an evaluation artifact, not training
variance. Full JSON on Drive: `BreastCancerNew/multiseed_v1/diagnostics/A1_scaling_check.json`.
