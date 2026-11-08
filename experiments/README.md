# Multi-seed re-run (conf-1, audit items A1, A2, B2, B6)

- `build_multiseed_notebook.py` generates `MultiSeed_BUSI.ipynb` (standard library only).
- The notebook lives on Drive (appexistdesign account) at
  `MyDrive/BreastCancerNew/multiseed_v1/MultiSeed_BUSI.ipynb` (md5 95f83d1b4a3dade2a4e233bbb3e46e66).
- Ran 2026-11-06 to 2026-11-07 on a Colab T4 (all 58 runs of the main experiment finished; 9.6 GPU-hours; two RAM crashes, resumed by the skip logic) (TF 2.21, Keras 3.13). It ran inside the BreastCancerConf.ipynb runtime
  by exec-ing the notebook's code cells.
- Each run writes `result.json`, `history.json` and `scores.npz` (a few KB) to `multiseed_v1/runs/<tag>/`.
  Model checkpoints stay on Colab local disk and are deleted after evaluation, so Drive use is under 1 MB.
- **Resume after a disconnect:** open the notebook and run cells 1–4, then cell 6. Finished runs are skipped
  because they already have a `result.json`. Cell 7 prints the summary table.
- Plan: 5 seeds × 10 configs = 50 runs. Controls: a same-seed repeat for VGG16-PF and DenseNet121-PF, and
  shape-only DenseNet121-PF and VGG16-PF with seeds 0–2. That is 58 runs at about 10–12 minutes each.
- `A1_scaling_check.md`: result of the A1 diagnostic. It confirms that run A's VGG16-PF 50% matches unscaled
  test input.


## Follow-up runs

34 further runs (4.3 GPU-hours, 2026-11-07) were made with `FollowUp_BUSI.ipynb` (same code as `followup_cells.py`), after cells 1–4 of `MultiSeed_BUSI.ipynb`. Their outputs are in `multiseed_results/runs/` with the tag prefixes `native_`, `masksil_`, `whole_` and `det_`.
