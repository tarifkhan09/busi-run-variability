"""Builds FollowUp_BUSI.ipynb: the follow-up experiments requested by the second review.

Cells 1-4 are copied unchanged from MultiSeed_BUSI.ipynb (setup, data, helpers, run_one);
cell 5 is followup_cells.py. Run all cells in order on a Colab T4; finished runs are skipped.
"""
import json, os
here = os.path.dirname(os.path.abspath(__file__))
ms = json.load(open(os.path.join(here, "MultiSeed_BUSI.ipynb")))
code = [c for c in ms["cells"] if c["cell_type"] == "code"][:4]
intro = {"cell_type": "markdown", "metadata": {}, "source":
         "# Follow-up experiments (conf-1, second review)\n\n"
         "1. Native `preprocess_input` for VGG16 and ResNet50 (PF and FT, seeds 0-4).\n"
         "2. Mask-derived silhouette control and whole-image control (DenseNet121-PF, VGG16-PF, seeds 0-2).\n"
         "3. Determinism check: VGG16-PF seed 0 twice with op determinism enabled.\n\n"
         "Run cells 1-5 in order. Results go to `MyDrive/BreastCancerNew/multiseed_v1/runs/`."}
cell5 = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": open(os.path.join(here, "followup_cells.py")).read()}
nb = dict(ms, cells=[intro] + code + [cell5])
json.dump(nb, open(os.path.join(here, "FollowUp_BUSI.ipynb"), "w"), indent=1)
print("wrote FollowUp_BUSI.ipynb")
