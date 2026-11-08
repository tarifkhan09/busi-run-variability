"""Duplicate-leakage sensitivity analysis (no retraining).

Usage:  python3 dedup_sensitivity.py BUSI_DIR RESULTS_DIR [HAMMING]
  BUSI_DIR     Dataset_BUSI_with_GT folder (benign/, malignant/ with the original file names)
  RESULTS_DIR  folder with split.csv and runs/<tag>/scores.npz
  HAMMING      maximum Hamming distance of the 64-bit difference hash (default 4)

A test image is "leaked" when a training or validation image lies within HAMMING bits of it
(difference hash of the grayscale image resized to 9x8, the rule used for the BSA study).
Every run is re-scored on the remaining test images with its stored scores and its
validation-tuned threshold. Writes dedup_pairs.csv, dedup_test_leaked.txt, dedup_runs.csv and
dedup_numbers.txt to RESULTS_DIR.
"""
import collections
import csv
import os
import sys

import numpy as np
from PIL import Image


def dhash(path):
    g = np.asarray(Image.open(path).convert("L").resize((9, 8), Image.BOX), np.int16)
    return int(np.packbits((g[:, 1:] > g[:, :-1]).flatten()).view(">u8")[0])


def auroc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def errors(p, y, t):
    pred = (p >= t).astype(int)
    return int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())


def main(busi, res, ham):
    rows = list(csv.DictReader(open(os.path.join(res, "split.csv"))))
    h = [dhash(os.path.join(busi, r["class"], r["filename"])) for r in rows]
    n = len(rows)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            k = bin(h[i] ^ h[j]).count("1")
            if k <= ham:
                pairs.append((i, j, k))
    with open(os.path.join(res, "dedup_pairs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file_a", "split_a", "class_a", "file_b", "split_b", "class_b", "hamming"])
        for i, j, k in pairs:
            w.writerow([rows[i]["filename"], rows[i]["split"], rows[i]["class"],
                        rows[j]["filename"], rows[j]["split"], rows[j]["class"], k])
    leaked = set()
    for i, j, _ in pairs:
        si, sj = rows[i]["split"], rows[j]["split"]
        if si == "test" and sj != "test":
            leaked.add(f"{rows[i]['class']}/{rows[i]['filename']}")
        if sj == "test" and si != "test":
            leaked.add(f"{rows[j]['class']}/{rows[j]['filename']}")
    open(os.path.join(res, "dedup_test_leaked.txt"), "w").write("\n".join(sorted(leaked)) + "\n")

    out = []
    for tag in sorted(os.listdir(os.path.join(res, "runs"))):
        f = os.path.join(res, "runs", tag, "scores.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        p, y = z["test"].astype(float), z["y_test"].astype(int)
        files = [str(s) for s in z["test_files"]]
        keep = np.array([s not in leaked for s in files])
        thr = float(__import__("json").load(open(os.path.join(res, "runs", tag, "result.json")))["thr_val"])
        fa, na = errors(p, y, 0.5)
        fc, nc = errors(p[keep], y[keep], 0.5)
        tc = errors(p[keep], y[keep], thr)
        out.append(dict(tag=tag, group=tag.rsplit("_seed", 1)[0], n_clean=int(keep.sum()),
                        auroc_all=auroc(p, y), auroc_clean=auroc(p[keep], y[keep]),
                        err05_all=fa + na, err05_clean=fc + nc, errt_clean=sum(tc),
                        acc05_all=100 * (1 - (fa + na) / len(y)),
                        acc05_clean=100 * (1 - (fc + nc) / keep.sum())))
    with open(os.path.join(res, "dedup_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    yb = sum(1 for s in leaked if s.startswith("benign"))
    cross = sum(1 for i, j, _ in pairs if rows[i]["split"] != rows[j]["split"])
    conflict = sum(1 for i, j, _ in pairs if rows[i]["class"] != rows[j]["class"])
    lines = [f"hamming<={ham}: {len(pairs)} near-duplicate pairs, {cross} across splits, {conflict} with conflicting labels",
             f"test images with a near-duplicate in train/val: {len(leaked)} ({yb} benign, {len(leaked) - yb} malignant); clean test set {n_test_clean(rows, leaked)}"]
    g = collections.defaultdict(list)
    for r in out:
        g[r["group"]].append(r)
    for k, rs in g.items():
        a = lambda key: np.array([r[key] for r in rs], float)
        lines.append(f"{k} n={len(rs)} AUROC all {a('auroc_all').mean():.4f} [{a('auroc_all').min():.4f}-{a('auroc_all').max():.4f}] "
                     f"clean {a('auroc_clean').mean():.4f} [{a('auroc_clean').min():.4f}-{a('auroc_clean').max():.4f}] | "
                     f"acc0.5 all {a('acc05_all').mean():.2f} [{a('acc05_all').min():.2f}-{a('acc05_all').max():.2f}] "
                     f"clean {a('acc05_clean').mean():.2f} [{a('acc05_clean').min():.2f}-{a('acc05_clean').max():.2f}] | "
                     f"err_t clean {a('errt_clean').mean():.1f}")
    for kind in ("main", "native"):
        rs = [r for r in out if r["group"].startswith(kind + "_")]
        a = lambda key: np.array([r[key] for r in rs], float)
        lines.append(f"ALL_{kind} n={len(rs)} AUROC all [{a('auroc_all').min():.4f}-{a('auroc_all').max():.4f}] mean {a('auroc_all').mean():.4f} "
                     f"clean [{a('auroc_clean').min():.4f}-{a('auroc_clean').max():.4f}] mean {a('auroc_clean').mean():.4f}; "
                     f"acc0.5 all [{a('acc05_all').min():.2f}-{a('acc05_all').max():.2f}] clean [{a('acc05_clean').min():.2f}-{a('acc05_clean').max():.2f}]; "
                     f"runs at AUROC 1 all {(a('auroc_all') >= 0.99995).sum()} clean {(a('auroc_clean') >= 0.99995).sum()}")
    open(os.path.join(res, "dedup_numbers.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def n_test_clean(rows, leaked):
    t = [r for r in rows if r["split"] == "test" and f"{r['class']}/{r['filename']}" not in leaked]
    return f"{len(t)} ({sum(r['class'] == 'benign' for r in t)} benign, {sum(r['class'] == 'malignant' for r in t)} malignant)"


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) == 4 else 4)
