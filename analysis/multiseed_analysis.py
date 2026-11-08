"""Analyse the multi-seed re-run (experiments/MultiSeed_BUSI.ipynb).

Usage:  python3 multiseed_analysis.py RUNS_DIR OUT_DIR
  RUNS_DIR  folder holding one sub-folder per run, each with result.json and scores.npz
            (Drive: MyDrive/BreastCancerNew/multiseed_v1/runs, or the copy in
            experiments/multiseed_results/runs)
  OUT_DIR   where summary_all_runs.csv, multiseed_stats.json and numbers_multiseed.txt are written

Needs only Python 3 and NumPy. Every number reported in the paper for the multi-seed
experiment comes from this script.

What it computes
  1. Per configuration (backbone x PF/FT, main runs only): mean, SD, min and max over seeds of
     test AUROC, errors at the fixed 0.5 threshold, and errors at the validation-tuned
     (Youden) threshold.
  2. Same-seed repeats: errors of the original and the repeat run, and the number of test
     cases whose predicted class differs between the two (GPU nondeterminism).
  3. Shape-only control: the same statistics for models trained on lesion silhouettes only.
  4. PF vs FT per backbone and seed: exact two-sided McNemar test on paired test-set
     correctness at 0.5 and at each model's validation-tuned threshold, with Holm correction.
  5. PF vs FT test AUROC, paired over seeds, with an exact sign-flip permutation test.
  6. Follow-up: native keras preprocess_input vs [0,1] scaling for all five backbones (paired over seeds).
  7. Follow-up input controls: silhouette from the expert mask and whole image without mask or crop,
     next to the non-zero-pixel silhouette and the lesion crop for the same seeds.
  8. Follow-up determinism check: VGG16-PF seed 0 twice with op determinism enabled.
"""
import csv
import glob
import json
import os
import sys
from math import comb

import numpy as np

BACKBONES = ["DenseNet121", "Xception", "InceptionV3", "ResNet50", "VGG16"]
STRATS = ["PF", "FT"]


def load(runs_dir):
    runs = {}
    for f in sorted(glob.glob(os.path.join(runs_dir, "*", "result.json"))):
        d = os.path.dirname(f)
        r = json.load(open(f))
        s = np.load(os.path.join(d, "scores.npz"), allow_pickle=False)
        r["_scores"] = {k: s[k] for k in ("val", "test", "y_val", "y_test")}
        runs[r["tag"]] = r
    return runs


def errs(c):
    return c["fp"] + c["fn"]


def stats(v):
    v = np.asarray(v, float)
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj



def main(runs_dir, out):
    os.makedirs(out, exist_ok=True)
    runs = load(runs_dir)
    if not runs:
        sys.exit(f"no runs found in {runs_dir}")
    n_test = len(next(iter(runs.values()))["_scores"]["y_test"])

    # 0. flat table of every run
    rows = []
    for t, r in runs.items():
        a, th = r["test_at_0.5"], r["test_at_thr"]
        rows.append({"tag": t, "kind": t.split("_")[0], "backbone": r["backbone"], "strategy": r["strategy"],
                     "seed": r["seed"], "val_auroc": round(r["val_auroc"], 4), "test_auroc": round(r["test_auroc"], 4),
                     "fp_05": a["fp"], "fn_05": a["fn"], "err_05": errs(a),
                     "acc_05": round(100 * (n_test - errs(a)) / n_test, 2),
                     "thr_val": round(r["thr_val"], 4), "fp_thr": th["fp"], "fn_thr": th["fn"], "err_thr": errs(th),
                     "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"], "minutes": r["minutes"]})
    with open(os.path.join(out, "summary_all_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: (x["kind"], x["strategy"], x["backbone"], x["seed"])))

    res = {"n_test": n_test, "n_runs": len(runs), "configs": {}, "repeats": [], "shape_only": {}, "pf_vs_ft": []}
    lines = [f"Multi-seed analysis: {len(runs)} runs, {n_test} test images", ""]

    # 1. per configuration, main runs
    lines.append("Per configuration (main runs): mean (SD) [min-max] over seeds")
    for s in STRATS:
        for b in BACKBONES:
            sel = [r for t, r in runs.items() if t.startswith("main_") and r["backbone"] == b and r["strategy"] == s]
            if not sel:
                continue
            e05 = [errs(r["test_at_0.5"]) for r in sel]
            eth = [errs(r["test_at_thr"]) for r in sel]
            auc = [r["test_auroc"] for r in sel]
            c = {"seeds": sorted(r["seed"] for r in sel), "auroc": stats(auc), "err_05": stats(e05),
                 "err_thr": stats(eth), "fp_05": stats([r["test_at_0.5"]["fp"] for r in sel]),
                 "fn_05": stats([r["test_at_0.5"]["fn"] for r in sel])}
            res["configs"][f"{b}|{s}"] = c
            lines.append(f"  {b:12s} {s}: n={len(sel)}  AUROC {c['auroc']['mean']:.4f} ({c['auroc']['sd']:.4f})"
                         f"  errors@0.5 {c['err_05']['mean']:.1f} ({c['err_05']['sd']:.1f}) [{c['err_05']['min']:.0f}-{c['err_05']['max']:.0f}]"
                         f"  errors@thr {c['err_thr']['mean']:.1f} ({c['err_thr']['sd']:.1f}) [{c['err_thr']['min']:.0f}-{c['err_thr']['max']:.0f}]")
    allmain = [r for t, r in runs.items() if t.startswith("main_")]
    if allmain:
        e05 = np.array([errs(r["test_at_0.5"]) for r in allmain])
        eth = np.array([errs(r["test_at_thr"]) for r in allmain])
        res["threshold_effect"] = {"runs": int(e05.size), "fewer_errors": int((eth < e05).sum()),
                                   "same": int((eth == e05).sum()), "more_errors": int((eth > e05).sum()),
                                   "total_err_05": int(e05.sum()), "total_err_thr": int(eth.sum())}
        te = res["threshold_effect"]
        lines += ["", f"Validation threshold vs 0.5 over {te['runs']} main runs: fewer errors in {te['fewer_errors']}, "
                      f"same in {te['same']}, more in {te['more_errors']}; total errors {te['total_err_05']} -> {te['total_err_thr']}"]

    # 2. same-seed repeats
    lines += ["", "Same-seed repeats (original vs repeat)"]
    for t, r in runs.items():
        if not t.startswith("repeat_"):
            continue
        o = runs.get(t.replace("repeat_", "main_", 1))
        if o is None:
            continue
        po = (o["_scores"]["test"] >= 0.5).astype(int)
        pr = (r["_scores"]["test"] >= 0.5).astype(int)
        rep = {"config": f"{r['backbone']}|{r['strategy']}", "seed": r["seed"],
               "orig": {"fp": o["test_at_0.5"]["fp"], "fn": o["test_at_0.5"]["fn"], "auroc": o["test_auroc"]},
               "repeat": {"fp": r["test_at_0.5"]["fp"], "fn": r["test_at_0.5"]["fn"], "auroc": r["test_auroc"]},
               "cases_flipped": int((po != pr).sum()),
               "max_abs_score_diff": float(np.abs(o["_scores"]["test"] - r["_scores"]["test"]).max())}
        res["repeats"].append(rep)
        lines.append(f"  {rep['config']} seed {rep['seed']}: FP/FN {rep['orig']['fp']}/{rep['orig']['fn']} vs "
                     f"{rep['repeat']['fp']}/{rep['repeat']['fn']}; {rep['cases_flipped']} of {n_test} test cases change class")

    # 3. shape-only control
    lines += ["", "Shape-only control (lesion silhouette input)"]
    for b in BACKBONES:
        sel = [r for t, r in runs.items() if t.startswith("shape_") and r["backbone"] == b]
        if not sel:
            continue
        c = {"seeds": sorted(r["seed"] for r in sel), "auroc": stats([r["test_auroc"] for r in sel]),
             "err_05": stats([errs(r["test_at_0.5"]) for r in sel]),
             "err_thr": stats([errs(r["test_at_thr"]) for r in sel])}
        res["shape_only"][b] = c
        lines.append(f"  {b:12s} PF: n={c['auroc']['n']}  AUROC {c['auroc']['mean']:.4f} [{c['auroc']['min']:.4f}-{c['auroc']['max']:.4f}]"
                     f"  errors@0.5 [{c['err_05']['min']:.0f}-{c['err_05']['max']:.0f}]  errors@thr [{c['err_thr']['min']:.0f}-{c['err_thr']['max']:.0f}]")

    # 4. PF vs FT, paired McNemar per backbone and seed, Holm-corrected
    pairs = []
    for b in BACKBONES:
        for seed in sorted({r["seed"] for r in runs.values()}):
            p, f = runs.get(f"main_{b}_PF_seed{seed}"), runs.get(f"main_{b}_FT_seed{seed}")
            if not (p and f):
                continue
            y = p["_scores"]["y_test"]
            cp = (p["_scores"]["test"] >= 0.5).astype(int) == y
            cf = (f["_scores"]["test"] >= 0.5).astype(int) == y
            bb, cc = int((cp & ~cf).sum()), int((~cp & cf).sum())
            # same comparison with each model at its own validation-tuned threshold
            tp = (p["_scores"]["test"] >= p["thr_val"]).astype(int) == y
            tf_ = (f["_scores"]["test"] >= f["thr_val"]).astype(int) == y
            bt, ct = int((tp & ~tf_).sum()), int((~tp & tf_).sum())
            pairs.append({"backbone": b, "seed": seed, "pf_only_correct": bb, "ft_only_correct": cc,
                          "p": mcnemar_exact(bb, cc), "thr_pf_only_correct": bt, "thr_ft_only_correct": ct,
                          "p_thr": mcnemar_exact(bt, ct)})
    for pr, a in zip(pairs, holm([x["p"] for x in pairs])):
        pr["p_holm"] = a
    for pr, a in zip(pairs, holm([x["p_thr"] for x in pairs])):
        pr["p_thr_holm"] = a
    res["pf_vs_ft"] = pairs
    if pairs:
        sig = [x for x in pairs if x["p_holm"] < 0.05]
        lines += ["", f"PF vs FT (exact McNemar at 0.5, Holm over {len(pairs)} comparisons): "
                      f"{len(sig)} significant at 0.05; at the validation-tuned thresholds "
                      f"{sum(x['p_thr_holm'] < 0.05 for x in pairs)} significant "
                      f"(smallest Holm p {min(x['p_thr_holm'] for x in pairs):.4f})"]
        for x in pairs:
            lines.append(f"  {x['backbone']:12s} seed {x['seed']}: discordant {x['pf_only_correct']}/{x['ft_only_correct']}"
                         f"  p={x['p']:.4f}  p_holm={x['p_holm']:.4f}"
                         f"  | at t: {x['thr_pf_only_correct']}/{x['thr_ft_only_correct']}  p_holm={x['p_thr_holm']:.4f}")

    # 5. PF vs FT discrimination: paired test-AUROC difference (FT - PF) over seeds,
    #    exact two-sided sign-flip permutation test (2^n_seeds sign patterns; minimum p with 5 seeds = 0.0625)
    res["auroc_pf_vs_ft"] = []
    lines += ["", "PF vs FT test AUROC, paired over seeds (FT - PF), exact sign-flip test"]
    for b in BACKBONES:
        d = [runs[f"main_{b}_FT_seed{s}"]["test_auroc"] - runs[f"main_{b}_PF_seed{s}"]["test_auroc"]
             for s in sorted({r["seed"] for r in runs.values()})
             if f"main_{b}_PF_seed{s}" in runs and f"main_{b}_FT_seed{s}" in runs]
        if not d:
            continue
        obs, n = abs(sum(d)), len(d)
        hits = sum(abs(sum(x if (m >> i) & 1 else -x for i, x in enumerate(d))) >= obs - 1e-12 for m in range(2 ** n))
        e = {"backbone": b, "n": n, "mean_diff": float(np.mean(d)), "min_diff": float(min(d)), "max_diff": float(max(d)),
             "p_signflip": hits / 2 ** n}
        res["auroc_pf_vs_ft"].append(e)
        lines.append(f"  {b:12s} mean {e['mean_diff']:+.4f} [{e['min_diff']:+.4f} to {e['max_diff']:+.4f}]  p={e['p_signflip']:.4f}")

    # 6. follow-up: native preprocess_input vs [0,1] scaling (same backbone, strategy and seed)
    res["native"] = {}
    lines += ["", "Native preprocess_input vs [0,1] scaling (paired over seeds; diff = native - [0,1])"]
    for s in STRATS:
        for b in BACKBONES:
            sel = sorted((r for t, r in runs.items() if t.startswith("native_") and r["backbone"] == b
                          and r["strategy"] == s), key=lambda r: r["seed"])
            if not sel:
                continue
            base = [runs.get(f"main_{b}_{s}_seed{r['seed']}") for r in sel]
            d_auc = [r["test_auroc"] - o["test_auroc"] for r, o in zip(sel, base) if o]
            d_err = [errs(r["test_at_0.5"]) - errs(o["test_at_0.5"]) for r, o in zip(sel, base) if o]
            obs, n = abs(sum(d_auc)), len(d_auc)
            hits = sum(abs(sum(x if (m >> i) & 1 else -x for i, x in enumerate(d_auc))) >= obs - 1e-12 for m in range(2 ** n))
            c = {"seeds": [r["seed"] for r in sel], "auroc": stats([r["test_auroc"] for r in sel]),
                 "err_05": stats([errs(r["test_at_0.5"]) for r in sel]),
                 "err_thr": stats([errs(r["test_at_thr"]) for r in sel]),
                 "main_auroc": stats([o["test_auroc"] for o in base if o]),
                 "main_err_05": stats([errs(o["test_at_0.5"]) for o in base if o]),
                 "auroc_diff_mean": float(np.mean(d_auc)), "err05_diff_mean": float(np.mean(d_err)),
                 "p_signflip_auroc": hits / 2 ** n}
            res["native"][f"{b}|{s}"] = c
            lines.append(f"  {b:9s} {s}: native AUROC {c['auroc']['mean']:.4f} [{c['auroc']['min']:.4f}-{c['auroc']['max']:.4f}]"
                         f" vs [0,1] {c['main_auroc']['mean']:.4f} [{c['main_auroc']['min']:.4f}-{c['main_auroc']['max']:.4f}]"
                         f" (p={c['p_signflip_auroc']:.4f});  errors@0.5 {c['err_05']['mean']:.1f} [{c['err_05']['min']:.0f}-{c['err_05']['max']:.0f}]"
                         f" vs {c['main_err_05']['mean']:.1f} [{c['main_err_05']['min']:.0f}-{c['main_err_05']['max']:.0f}];"
                         f"  errors@thr {c['err_thr']['mean']:.1f} [{c['err_thr']['min']:.0f}-{c['err_thr']['max']:.0f}]")

    # 7. follow-up input controls: mask-derived silhouette and whole B-mode image (PF, seeds 0-2)
    res["input_controls"] = {}
    lines += ["", "Input controls (PF): masksil = silhouette from the expert mask, whole = image without mask or crop,",
              "shape = silhouette from non-zero pixels, main = lesion crop (same seeds)"]
    for b in BACKBONES:
        for kind in ("masksil", "whole", "shape", "main"):
            sel = [r for t, r in runs.items() if t.startswith(kind + "_") and r["backbone"] == b and r["strategy"] == "PF"
                   and r["seed"] in (0, 1, 2)]
            if not sel or not any(t.startswith(("masksil_", "whole_")) and runs[t]["backbone"] == b for t in runs):
                continue
            c = {"seeds": sorted(r["seed"] for r in sel), "auroc": stats([r["test_auroc"] for r in sel]),
                 "err_05": stats([errs(r["test_at_0.5"]) for r in sel]),
                 "err_thr": stats([errs(r["test_at_thr"]) for r in sel])}
            res["input_controls"][f"{b}|{kind}"] = c
            lines.append(f"  {b:12s} {kind:7s}: n={c['auroc']['n']}  AUROC {c['auroc']['mean']:.4f} [{c['auroc']['min']:.4f}-{c['auroc']['max']:.4f}]"
                         f"  errors@0.5 [{c['err_05']['min']:.0f}-{c['err_05']['max']:.0f}]  errors@thr [{c['err_thr']['min']:.0f}-{c['err_thr']['max']:.0f}]")

    # 8. follow-up determinism check: same seed twice with tf.config.experimental.enable_op_determinism()
    a, b2 = runs.get("det_VGG16_PF_seed0a"), runs.get("det_VGG16_PF_seed0b")
    if a and b2:
        pa, pb = (a["_scores"]["test"] >= 0.5).astype(int), (b2["_scores"]["test"] >= 0.5).astype(int)
        res["determinism"] = {"config": "VGG16|PF", "seed": 0,
                              "a": {"fp": a["test_at_0.5"]["fp"], "fn": a["test_at_0.5"]["fn"], "auroc": a["test_auroc"]},
                              "b": {"fp": b2["test_at_0.5"]["fp"], "fn": b2["test_at_0.5"]["fn"], "auroc": b2["test_auroc"]},
                              "cases_flipped": int((pa != pb).sum()),
                              "max_abs_score_diff": float(np.abs(a["_scores"]["test"] - b2["_scores"]["test"]).max()),
                              "max_abs_val_score_diff": float(np.abs(a["_scores"]["val"] - b2["_scores"]["val"]).max())}
        d = res["determinism"]
        lines += ["", f"Op determinism, VGG16-PF seed 0 twice: FP/FN {d['a']['fp']}/{d['a']['fn']} vs {d['b']['fp']}/{d['b']['fn']};"
                      f" {d['cases_flipped']} of {n_test} test cases change class; max |score diff| {d['max_abs_score_diff']:.2e}"]

    json.dump(res, open(os.path.join(out, "multiseed_stats.json"), "w"), indent=1)
    open(os.path.join(out, "numbers_multiseed.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
