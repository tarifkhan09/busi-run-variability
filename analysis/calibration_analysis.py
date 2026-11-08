"""Calibration of every controlled run on the 98 test images.

Usage:  python3 calibration_analysis.py RESULTS_DIR
  RESULTS_DIR  folder with runs/<tag>/scores.npz (written by the Colab notebooks)

Writes RESULTS_DIR/calibration_runs.csv (one row per run) and
RESULTS_DIR/calibration_numbers.txt (mean [range] per configuration).

Brier score = mean squared difference between the malignancy score and the label.
ECE = expected calibration error with ten equal-width bins on [0, 1]
(weighted mean |mean score - fraction malignant| over non-empty bins).
"""
import collections
import csv
import os
import sys

import numpy as np


def ece(p, y, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def main(d):
    rows = []
    for tag in sorted(os.listdir(os.path.join(d, "runs"))):
        f = os.path.join(d, "runs", tag, "scores.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        p, y = z["test"].astype(float), z["y_test"].astype(float)
        rows.append(dict(tag=tag, group=tag.rsplit("_seed", 1)[0],
                         brier=float(np.mean((p - y) ** 2)), ece=ece(p, y),
                         median_benign=float(np.median(p[y == 0])),
                         median_malignant=float(np.median(p[y == 1]))))
    with open(os.path.join(d, "calibration_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    lines = ["group n brier_mean [min-max] ece_mean [min-max] median_benign_score median_malignant_score"]
    for g, rs in groups.items():
        b = np.array([r["brier"] for r in rs]); e = np.array([r["ece"] for r in rs])
        mb = np.mean([r["median_benign"] for r in rs]); mm = np.mean([r["median_malignant"] for r in rs])
        lines.append(f"{g} n={len(rs)} brier {b.mean():.3f} [{b.min():.3f}-{b.max():.3f}] "
                     f"ece {e.mean():.3f} [{e.min():.3f}-{e.max():.3f}] medB {mb:.2f} medM {mm:.2f}")
    for kind in ("main", "native"):
        rs = [r for r in rows if r["group"].startswith(kind + "_")]
        lines.append(f"ALL_{kind} n={len(rs)} ece_mean {np.mean([r['ece'] for r in rs]):.3f} "
                     f"brier_mean {np.mean([r['brier'] for r in rs]):.3f}")
    open(os.path.join(d, "calibration_numbers.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
