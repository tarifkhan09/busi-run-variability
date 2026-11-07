# Drive results snapshot (conf-1)

Copy of `MyDrive/BreastCancerNew/results/` as downloaded on 2026-11-06, kept here so it
does not have to be downloaded again. Model weights (`*.keras`, ~1.9 GB) are **not** in git
(`.gitignore` excludes them); their sizes and SHA-256 hashes are in `MODEL_CHECKSUMS.txt`,
so a later download can be checked against this snapshot.

The download was partial: several folders hold only some models (see "Missing" below).

## What maps to which run of the paper

| Path | Run | Contents |
|---|---|---|
| `final_results.csv`, `final_evaluation_results.csv` (26 Aug) | B ([0,1] scaling) | Test metrics at 0.5 for 8 of 10 configs (no Xception-PF, no ResNet50-FT). Match Table III run B. |
| `tonight_rescale/RESULTS.md` (31 Aug) | B | FT arm of all 5 backbones incl. retrained ResNet50-FT (90.82%, 4 FP / 5 FN), plus the Grad-CAM++ fidelity rerun that did **not** reproduce the draft's pointing-game gap. |
| `models/ResNet50_finetune_meta.json` | B | `lr 1e-5, freeze 0.4, rescale, 30 epochs`. |
| `history/*_history.json` dated 25 Aug | B | PF arm of Xception, VGG16, InceptionV3, ResNet50 (all start at lr 1e-4). |
| `history/ResNet50_finetune_history.json` (31 Aug) | B | 30 epochs at lr 1e-5. |
| `history/ResNet50_v2_*_history.json` (11 Apr) | A | ResNet50 PF (1e-4) and FT (5e-6, 50 epochs). |
| `paper_native/table1_internal_with_CI.csv` | C (native `preprocess_input`) | All 10 configs: AUROC, accuracy at 0.5, validation threshold, counts at that threshold. Match Table III/IV run C. |
| `models_native/*_meta.json` | C | lr 1e-4 (PF) / 1e-5 (FT), freeze 0.5/0.4 (VGG16-FT 0.5), 50 epochs. |
| `preds_native/*.npy` | C | Per-case val/test scores for some models only. |
| `paper_native/table3..5`, `external_validation_BUS_BRA/` | C | Ensemble and BUS-BRA external validation (not used in the current paper). |
| `paper_native/fidelity/*.npz`, `plots/gradcam/` | A/C | Grad-CAM++ fidelity arrays and figures (older XAI framing). |
| `models_corrupt/` | B | The lost Xception-PF checkpoint (0 bytes) and a broken ResNet50-FT (175 KB). Weights not in git. |

## Notes for the paper

- Run C validation thresholds in `table1_internal_with_CI.csv` differ from those in
  `analysis/three_runs.json` (e.g. DenseNet121-PF 0.608 vs 0.521, VGG16-PF 0.988 vs 0.984),
  but the FP/FN counts at the threshold are identical. The JSON values come from a later
  recomputation (median of tied Youden cut-points); the CSV used an earlier rule.
- Run B FT learning rate is confirmed (1e-5) only for ResNet50. No run-B meta or history
  file exists here for DenseNet121-FT or VGG16-FT, so the TODO in `main.tex` stays open.

## Missing (still needed from Drive / Chinmoy)

- Any training or rescoring **code** for runs B and C (no notebook in this folder).
- `threshold_rescoring/` is empty: run-B `threshold_rescoring.csv`, `per_case_predictions.csv`,
  `mcnemar_pf_vs_ft.csv`.
- Run-B meta/history for DenseNet121-FT, VGG16-FT, DenseNet121-PF, Xception-FT, InceptionV3-FT.
- Run-C history files, the remaining `preds_native` scores, and meta for
  Xception-PF, ResNet50-PF, VGG16-PF, DenseNet121-FT.
- Run A `final_results_improved.csv` (12 Apr) and the April 1e-5 retrain results.
- `HANDOFF.md`.
