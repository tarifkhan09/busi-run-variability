"""Derive every number used in the three-run manuscript from three_runs.json.

Usage: python three_runs_analysis.py three_runs.json OUTDIR
Writes OUTDIR/numbers.txt (all derived quantities) and OUTDIR/fig_runs.pdf/.png.
"""
import json
import sys
from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

data = json.load(open(sys.argv[1]))
out = sys.argv[2]
NB, NM = data["n_benign"], data["n_malignant"]
N = NB + NM
RUNS = ["A", "B", "C"]
BACKBONES = ["DenseNet121", "Xception", "InceptionV3", "ResNet50", "VGG16"]
CONFIGS = [f"{b}|{s}" for s in ("PF", "FT") for b in BACKBONES]


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), min(100.0, 100 * (c + h))


def derived(fp, fn):
    err = fp + fn
    return {
        "acc": 100 * (N - err) / N,
        "sen": 100 * (NM - fn) / NM,
        "spe": 100 * (NB - fp) / NB,
        "err": err,
        "ci": wilson(N - err, N),
    }


lines = []
P = lines.append

P("=== Per-configuration metrics at 0.5 (and validation-tuned where available) ===")
for cfg in CONFIGS:
    for r in RUNS:
        e = data["runs"][r][cfg]
        if e is None:
            P(f"{cfg:16s} {r}: missing")
            continue
        d = derived(e["fp"], e["fn"])
        s = (f"{cfg:16s} {r}: AUROC {e['auc']:.4f} acc {d['acc']:.2f} "
             f"[{d['ci'][0]:.2f},{d['ci'][1]:.2f}] sen {d['sen']:.2f} spe {d['spe']:.2f} "
             f"FP {e['fp']} FN {e['fn']}")
        if "thr" in e:
            dv = derived(e["fpv"], e["fnv"])
            s += (f" | thr {e['thr']:.2f} acc {dv['acc']:.2f} sen {dv['sen']:.2f} "
                  f"spe {dv['spe']:.2f} FP {e['fpv']} FN {e['fnv']}")
        P(s)

P("\n=== Run summaries ===")
for r in RUNS:
    rows = {c: e for c, e in data["runs"][r].items() if e is not None}
    accs = {c: derived(e["fp"], e["fn"])["acc"] for c, e in rows.items()}
    best = max(accs.values())
    perfect = [c for c, e in rows.items() if e["auc"] >= 0.99995]
    perfect_err = [c for c in perfect if rows[c]["fp"] + rows[c]["fn"] > 0]
    P(f"Run {r}: n={len(rows)} best acc {best:.2f} by {[c for c in accs if accs[c] == best]}; "
      f"worst {min(accs.values()):.2f} by {[c for c in accs if accs[c] == min(accs.values())]}")
    P(f"   AUROC=1.0000 configs: {len(perfect)} {perfect}; with errors at 0.5: {perfect_err}")
    P(f"   total errors at 0.5: {sum(e['fp'] + e['fn'] for e in rows.values())}")
    if all("thr" in e for e in rows.values()):
        P(f"   total errors at val thr: {sum(e['fpv'] + e['fnv'] for e in rows.values())}")

P("\n=== Pure rerun: PF arm, run A vs run B (identical configuration) ===")
for b in BACKBONES:
    a, bb = data["runs"]["A"][f"{b}|PF"], data["runs"]["B"][f"{b}|PF"]
    if bb is None:
        P(f"{b}: B missing")
        continue
    P(f"{b}: A FP{a['fp']}/FN{a['fn']} ({derived(a['fp'], a['fn'])['acc']:.2f}) -> "
      f"B FP{bb['fp']}/FN{bb['fn']} ({derived(bb['fp'], bb['fn'])['acc']:.2f}); "
      f"shift {abs(derived(a['fp'], a['fn'])['acc'] - derived(bb['fp'], bb['fn'])['acc']):.2f} pts")

P("\n=== Per-configuration accuracy range across runs ===")
for cfg in CONFIGS:
    accs = [derived(e["fp"], e["fn"])["acc"] for r in RUNS
            if (e := data["runs"][r][cfg]) is not None]
    P(f"{cfg:16s} min {min(accs):.2f} max {max(accs):.2f} range {max(accs) - min(accs):.2f}")

P("\n=== DenseNet121 error profile per run (PF vs FT) ===")
for r in RUNS:
    pf, ft = data["runs"][r]["DenseNet121|PF"], data["runs"][r]["DenseNet121|FT"]
    P(f"Run {r}: PF FP{pf['fp']}/FN{pf['fn']}  FT FP{ft['fp']}/FN{ft['fn']}")

P("\n=== McNemar (exact, two-sided) PF vs FT ===")
for r, d in data["mcnemar_pf_vs_ft"].items():
    for b, v in d.items():
        for thr, (bb, cc, p) in v.items():
            pe = binomtest(bb, bb + cc, 0.5).pvalue if bb + cc else 1.0
            if p is not None:
                assert abs(pe - p) < 1e-3, (r, b, thr, pe, p)
            P(f"Run {r} {b:12s} {thr:4s} b={bb} c={cc} p={pe:.3g}")
P("Run A DenseNet121 (from confusion matrices): b=1 c=1 p=1.0")

open(f"{out}/numbers.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# --- Figure: errors per configuration in each run (stacked FP over FN) ---
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(3.45, 2.6))
colors = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}
labels = {"A": "Run A ([0,1], Apr)", "B": "Run B ([0,1], Aug)", "C": "Run C (native, Aug)"}
short = {"DenseNet121": "DN", "Xception": "Xc", "InceptionV3": "Inc",
         "ResNet50": "RN", "VGG16": "VGG"}
w = 0.27
for i, cfg in enumerate(CONFIGS):
    for j, r in enumerate(RUNS):
        e = data["runs"][r][cfg]
        x = i + (j - 1) * w
        if e is None:
            ax.text(x, 0.6, "×", ha="center", va="bottom", fontsize=6, color=colors[r])
            continue
        ax.bar(x, e["fn"], w, color=colors[r], edgecolor="black", linewidth=0.3)
        ax.bar(x, e["fp"], w, bottom=e["fn"], color=colors[r], alpha=0.45,
               edgecolor="black", linewidth=0.3, hatch="////")
ax.set_yscale("symlog", linthresh=5)
ax.set_yticks([0, 1, 2, 5, 10, 20, 50])
ax.set_yticklabels(["0", "1", "2", "5", "10", "20", "50"])
ax.set_ylim(0, 60)
ax.set_xticks(range(len(CONFIGS)))
ax.set_xticklabels([f"{short[c.split('|')[0]]}\n{c.split('|')[1]}" for c in CONFIGS], fontsize=6)
ax.axvline(4.5, color="grey", linewidth=0.5, linestyle=":")
ax.set_ylabel("Test errors at 0.5 (of 98)")
from matplotlib.patches import Patch
handles = [Patch(facecolor=colors[r], edgecolor="black", linewidth=0.3, label=labels[r]) for r in RUNS]
handles += [Patch(facecolor="white", edgecolor="black", linewidth=0.3, label="FN (solid)"),
            Patch(facecolor="white", edgecolor="black", linewidth=0.3, hatch="////", label="FP (hatched)")]
ax.legend(handles=handles, fontsize=5.5, ncol=3, frameon=False, loc="lower center",
          bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.2)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(f"{out}/fig_runs.pdf")
fig.savefig(f"{out}/fig_runs.png", dpi=300)
print("figure written")
