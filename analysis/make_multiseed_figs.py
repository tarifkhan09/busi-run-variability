"""pgfplots figures for the multi-seed experiment.

Usage:  python3 make_multiseed_figs.py RESULTS_DIR FIG_DIR
  RESULTS_DIR  folder with summary_all_runs.csv (written by multiseed_analysis.py)
  FIG_DIR      where fig_seeds.tex and fig_auc_err.tex are written; compile each with
               tectonic (or pdflatex) and copy the PDFs next to main.tex.
Standard library only.

fig_seeds    test errors of every run, per configuration, at 0.5 (open) and at the
             validation-tuned threshold (filled); shape-only control in its own block.
fig_auc_err  test AUROC against errors at 0.5 for every main run: near-identical AUROC,
             widely different error counts.
"""
import csv
import math
import os
import sys

BACK = ["DenseNet121", "Xception", "InceptionV3", "ResNet50", "VGG16"]
SHORT = {"DenseNet121": "DenseNet121", "Xception": "Xception", "InceptionV3": "InceptionV3",
         "ResNet50": "ResNet50", "VGG16": "VGG16"}
BCOL = {"DenseNet121": "c1", "Xception": "c2", "InceptionV3": "c3", "ResNet50": "c4", "VGG16": "c5"}

PRE = r"""\documentclass[border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{c1}{RGB}{31,119,180}\definecolor{c2}{RGB}{255,127,14}\definecolor{c3}{RGB}{44,160,44}
\definecolor{c4}{RGB}{214,39,40}\definecolor{c5}{RGB}{148,103,189}
\begin{document}
"""
POST = "\\end{document}\n"


def lg(v):
    return math.log1p(v)


def ticks(vals):
    return ",".join(f"{lg(v):.4f}" for v in vals), ",".join(str(v) for v in vals)


def load(d):
    rows = list(csv.DictReader(open(os.path.join(d, "summary_all_runs.csv"))))
    for r in rows:
        for k in ("seed", "fp_05", "fn_05", "err_05", "fp_thr", "fn_thr", "err_thr"):
            r[k] = int(r[k])
        r["test_auroc"] = float(r["test_auroc"])
    return rows


def fig_seeds(rows):
    groups = [(f"{SHORT[b]} {s}", [r for r in rows if r["kind"] == "main" and r["backbone"] == b and r["strategy"] == s])
              for s in ("PF", "FT") for b in BACK]
    shape = [(f"{SHORT[b]} PF, shape", [r for r in rows if r["kind"] == "shape" and r["backbone"] == b]) for b in BACK]
    groups += [g for g in shape if g[1]]
    groups = [g for g in groups if g[1]]
    n = len(groups)
    ys = {name: n - 1 - i for i, (name, _) in enumerate(groups)}
    tv, tl = ticks([0, 1, 2, 5, 10, 20, 50, 98])
    out = [PRE, r"""\begin{tikzpicture}
\begin{axis}[width=8.6cm, height=""" + f"{0.36 * n + 1.4:.2f}" + r"""cm, ymin=-0.7, ymax=""" + f"{n - 0.3}" + r""",
  ytick={""" + ",".join(str(ys[g]) for g, _ in groups) + "}, yticklabels={" + ",".join("{" + g + "}" for g, _ in groups) + r"""},
  xmin=-0.15, xmax=""" + f"{lg(98) + 0.1:.4f}" + ", xtick={" + tv + "}, xticklabels={" + tl + r"""},
  ticklabel style={font=\scriptsize}, label style={font=\footnotesize}, xlabel={Test errors (FP + FN), log scale},
  xmajorgrids, grid style={black!10}, axis line style={black!70}, tick align=outside,
  legend cell align=left, legend style={font=\scriptsize, draw=none, legend columns=2, at={(0.5,1.01)}, anchor=south,
  /tikz/every even column/.append style={column sep=6pt}}]
"""]
    n_main = sum(1 for g, _ in groups if "shape" not in g)
    if n_main < n:
        out.append(f"\\draw[densely dotted, black!60] (axis cs:\\pgfkeysvalueof{{/pgfplots/xmin}},{n - n_main - 0.5}) -- "
                   f"(axis cs:\\pgfkeysvalueof{{/pgfplots/xmax}},{n - n_main - 0.5});")
    if n_main >= 10:
        out.append(f"\\draw[densely dotted, black!35] (axis cs:\\pgfkeysvalueof{{/pgfplots/xmin}},{n - 5.5}) -- "
                   f"(axis cs:\\pgfkeysvalueof{{/pgfplots/xmax}},{n - 5.5});")
    for name, rs in groups:
        y = ys[name]
        col = BCOL[rs[0]["backbone"]]
        lo = min(min(r["err_05"], r["err_thr"]) for r in rs)
        hi = max(max(r["err_05"], r["err_thr"]) for r in rs)
        out.append(f"\\draw[{col}!35, line width=3pt] (axis cs:{lg(lo):.4f},{y}) -- (axis cs:{lg(hi):.4f},{y});")
        k = len(rs)
        for j, r in enumerate(sorted(rs, key=lambda r: r["seed"])):
            dy = (j - (k - 1) / 2) * 0.09
            out.append(f"\\addplot[only marks, mark=o, mark size=1.9pt, {col}, forget plot] coordinates {{({lg(r['err_05']):.4f},{y + dy + 0.12:.3f})}};")
            out.append(f"\\addplot[only marks, mark=*, mark size=1.6pt, {col}, forget plot] coordinates {{({lg(r['err_thr']):.4f},{y + dy - 0.12:.3f})}};")
    out.append(r"\addlegendimage{only marks, mark=o, black}\addlegendentry{one run at 0.5}")
    out.append(r"\addlegendimage{only marks, mark=*, black}\addlegendentry{one run at validation-tuned $t$}")
    out.append(r"\end{axis}" + "\n" + r"\end{tikzpicture}" + "\n" + POST)
    return "\n".join(out)


def fig_auc_err(rows):
    main = [r for r in rows if r["kind"] == "main"]
    amin = min(r["test_auroc"] for r in main)
    xmin = math.floor((amin - 0.005) * 100) / 100
    tv, tl = ticks([0, 1, 2, 5, 10, 20, 50, 98])
    out = [PRE, r"""\begin{tikzpicture}
\begin{axis}[width=8.6cm, height=4.4cm, xmin=""" + f"{xmin}" + r""", xmax=1.002, ymin=-0.15, ymax=""" + f"{lg(98) + 0.1:.4f}" + r""",
  ytick={""" + tv + "}, yticklabels={" + tl + r"""}, xlabel={Test AUROC}, ylabel={Test errors at 0.5 (log scale)},
  xticklabel style={/pgf/number format/fixed, /pgf/number format/precision=3},
  ticklabel style={font=\scriptsize}, label style={font=\footnotesize}, grid=major, grid style={black!10},
  axis line style={black!70}, legend cell align=left,
  legend style={font=\scriptsize, draw=none, legend columns=4, at={(0.5,-0.34)}, anchor=north,
  /tikz/every even column/.append style={column sep=4pt}}]
"""]
    for b in BACK:
        for s, mk in (("PF", "*"), ("FT", "triangle*")):
            pts = " ".join(f"({r['test_auroc']:.4f},{lg(r['err_05']):.4f})" for r in main
                           if r["backbone"] == b and r["strategy"] == s)
            if pts:
                out.append(f"\\addplot[only marks, mark={mk}, mark size=2pt, {BCOL[b]}, mark options={{fill={BCOL[b]}!60, draw={BCOL[b]}}}, forget plot] coordinates {{{pts}}};")
    for b in BACK:
        out.append(f"\\addlegendimage{{only marks, mark=square*, {BCOL[b]}}}\\addlegendentry{{{b}}}")
    out.append(r"\addlegendimage{only marks, mark=*, black!60}\addlegendentry{PF}")
    out.append(r"\addlegendimage{only marks, mark=triangle*, black!60}\addlegendentry{FT}")
    out.append(r"\end{axis}" + "\n" + r"\end{tikzpicture}" + "\n" + POST)
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    rows = load(sys.argv[1])
    os.makedirs(sys.argv[2], exist_ok=True)
    for name, fn in (("fig_seeds", fig_seeds), ("fig_auc_err", fig_auc_err)):
        with open(os.path.join(sys.argv[2], name + ".tex"), "w") as f:
            f.write(fn(rows))
        print("wrote", name + ".tex")
