"""Generate pgfplots sources for the result figures from three_runs.json.

Usage (from conf-1/analysis):  python3 make_pgf_figs.py
Writes fig_spread.tex, fig_fpfn.tex and fig_thr.tex into the paper's figures_src/
folder; compile each with tectonic (or pdflatex) and copy the PDFs next to main.tex.
Standard library only.
"""
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.join(HERE, "..", "Quantifying_Visual_Explanation_Fidelity_in_Transfer_"
                     "Learning_Models_for_Breast_Ultrasound_Lesion_Classification_t_review")
OUT = os.path.join(PAPER, "figures_src")
D = json.load(open(os.path.join(HERE, "three_runs.json")))
N = D["n_benign"] + D["n_malignant"]

BACK = ["DenseNet121", "Xception", "InceptionV3", "ResNet50", "VGG16"]
SHORT = {"DenseNet121": "DN", "Xception": "Xc", "InceptionV3": "Inc", "ResNet50": "RN", "VGG16": "VGG"}
CONFIGS = [(b, s) for s in ("PF", "FT") for b in BACK]
RUNS = ["A", "B", "C"]
RUNSTYLE = {"A": "mark=*, color=cA", "B": "mark=square*, color=cB", "C": "mark=triangle*, color=cC"}
BCOL = {"DenseNet121": "c1", "Xception": "c2", "InceptionV3": "c3", "ResNet50": "c4", "VGG16": "c5"}

PRE = r"""\documentclass[border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\definecolor{cA}{RGB}{31,119,180}\definecolor{cB}{RGB}{230,120,30}\definecolor{cC}{RGB}{44,160,44}
\definecolor{c1}{RGB}{31,119,180}\definecolor{c2}{RGB}{255,127,14}\definecolor{c3}{RGB}{44,160,44}
\definecolor{c4}{RGB}{214,39,40}\definecolor{c5}{RGB}{148,103,189}
\begin{document}
"""
POST = "\\end{document}\n"


def rec(run, b, s):
    return D["runs"][run].get(f"{b}|{s}")


def lg(v):  # log(1+v) display transform for error counts
    return math.log1p(v)


def err_ticks():
    vals = [0, 1, 2, 5, 10, 20, 50]
    return ",".join(f"{lg(v):.4f}" for v in vals), ",".join(str(v) for v in vals)


def fig_spread():
    xt = ",".join(str(i) for i in range(len(CONFIGS)))
    xl = ",".join(f"{{{SHORT[b]}\\\\{s}}}" for b, s in CONFIGS)
    def panel(metric, ylab, ymin, ymax, ytick, last):
        out = [f"\\nextgroupplot[ylabel={{{ylab}}}, ymin={ymin}, ymax={ymax}, ytick={{{ytick}}}"
               + (", xticklabels={" + xl + "}" if last else ", xticklabels={}") + "]"]
        out.append(r"\draw[densely dotted, black!50] (axis cs:4.5,\pgfkeysvalueof{/pgfplots/ymin}) -- (axis cs:4.5,\pgfkeysvalueof{/pgfplots/ymax});")
        for i, (b, s) in enumerate(CONFIGS):
            vals = [metric(r) for r in (rec(R, b, s) for R in RUNS) if r]
            out.append(f"\\draw[black!35, line width=1.6pt] (axis cs:{i},{min(vals):.4f}) -- (axis cs:{i},{max(vals):.4f});")
        for R in RUNS:
            pts = " ".join(f"({i + {'A': -0.18, 'B': 0, 'C': 0.18}[R]},{metric(r):.4f})"
                           for i, (b, s) in enumerate(CONFIGS) if (r := rec(R, b, s)))
            leg = f"\\addlegendentry{{Run {R}}}" if last else ""
            out.append(f"\\addplot[only marks, {RUNSTYLE[R]}, mark size=1.7pt] coordinates {{{pts}}};{leg}")
        return "\n".join(out)
    acc = lambda r: 100 * (N - r["fp"] - r["fn"]) / N
    auc = lambda r: r["auc"]
    body = PRE + r"""\begin{tikzpicture}
\begin{groupplot}[group style={group size=1 by 2, vertical sep=6pt},
  width=8.9cm, height=3.6cm, xmin=-0.6, xmax=9.6, xtick={""" + xt + r"""},
  xticklabel style={align=center, font=\scriptsize}, yticklabel style={font=\scriptsize},
  ylabel style={font=\footnotesize}, tick align=outside, axis line style={black!70},
  ymajorgrids, grid style={black!10},
  legend style={font=\scriptsize, at={(0.5,-0.42)}, anchor=north, legend columns=3, draw=none}]
""" + panel(auc, "AUROC", 0.945, 1.003, "0.95,0.96,0.97,0.98,0.99,1.00", False).replace("\\nextgroupplot[", "\\nextgroupplot[yticklabel style={/pgf/number format/fixed, /pgf/number format/fixed zerofill, /pgf/number format/precision=2}, ", 1) + "\n" \
        + panel(acc, "Accuracy at 0.5 (\\%)", 45, 103, "50,60,70,80,90,100", True) + r"""
\end{groupplot}
\end{tikzpicture}
""" + POST
    return body


def fig_fpfn():
    tv, tl = err_ticks()
    out = [PRE, r"""\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1, horizontal sep=8pt, yticklabels at=edge left},
  width=4.75cm, height=4.75cm, xmin=-0.15, xmax=4.1, ymin=-0.15, ymax=3.2,
  xtick={""" + tv + "}, xticklabels={" + tl + "}, ytick={" + tv + "}, yticklabels={" + tl + r"""},
  ticklabel style={font=\scriptsize}, label style={font=\footnotesize}, title style={font=\footnotesize\bfseries, yshift=-4pt},
  grid=major, grid style={black!10}, axis line style={black!70}, xlabel={False positives},
  legend cell align=left, legend style={font=\scriptsize, draw=none, legend columns=4, at={(1.045,-0.27)}, anchor=north, /tikz/every even column/.append style={column sep=4pt}}]
"""]
    for k, s in enumerate(("PF", "FT")):
        out.append(f"\\nextgroupplot[title={{{'Partial freezing' if s == 'PF' else 'Fine-tuning'}}}"
                   + (", ylabel={False negatives}" if k == 0 else "") + "]")
        for b in BACK:
            pts = [(R, rec(R, b, s)) for R in RUNS if rec(R, b, s)]
            line = " ".join(f"({lg(r['fp']):.4f},{lg(r['fn']):.4f})" for _, r in pts)
            out.append(f"\\addplot[{BCOL[b]}, line width=0.5pt, opacity=0.55, forget plot] coordinates {{{line}}};")
            for R, r in pts:
                mk = {"A": "*", "B": "square*", "C": "triangle*"}[R]
                out.append(f"\\addplot[only marks, mark={mk}, mark size=2pt, color={BCOL[b]}, "
                           f"mark options={{fill={BCOL[b]}!70, draw=black!60}}, forget plot] "
                           f"coordinates {{({lg(r['fp']):.4f},{lg(r['fn']):.4f})}};")
        if k == 0:
            for b in BACK:
                out.append(f"\\addlegendimage{{only marks, mark=*, color={BCOL[b]}}}\\addlegendentry{{{b}}}")
            for R, mk in (("A", "*"), ("B", "square*"), ("C", "triangle*")):
                out.append(f"\\addlegendimage{{only marks, mark={mk}, color=black!60}}\\addlegendentry{{Run {R}}}")
    out.append(r"\end{groupplot}" + "\n" + r"\end{tikzpicture}" + "\n" + POST)
    return "\n".join(out)


def fig_thr():
    ylab = ",".join(f"{{{SHORT[b]}-{s}}}" for b, s in CONFIGS[::-1])
    yt = ",".join(str(i) for i in range(len(CONFIGS)))
    out = [PRE, r"""\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1, horizontal sep=8pt, yticklabels at=edge left},
  width=4.6cm, height=5.6cm, ymin=-0.6, ymax=9.6, ytick={""" + yt + "}, yticklabels={" + ylab + r"""},
  xmin=-1, xmax=23, xtick={0,5,10,15,20}, ticklabel style={font=\scriptsize}, label style={font=\footnotesize},
  title style={font=\footnotesize\bfseries, yshift=-4pt}, xmajorgrids, grid style={black!10},
  axis line style={black!70}, xlabel={Test errors (FP + FN)},
  legend cell align=left, legend style={font=\scriptsize, draw=none, legend columns=2, at={(1.045,-0.22)}, anchor=north, /tikz/every even column/.append style={column sep=6pt}}]
"""]
    for k, R in enumerate(("B", "C")):
        out.append(f"\\nextgroupplot[title={{Run {R}}}]")
        out.append(r"\draw[densely dotted, black!50] (axis cs:\pgfkeysvalueof{/pgfplots/xmin},4.5) -- (axis cs:\pgfkeysvalueof{/pgfplots/xmax},4.5);")
        for i, (b, s) in enumerate(CONFIGS[::-1]):
            r = rec(R, b, s)
            if not r:
                out.append(f"\\node[font=\\scriptsize, text=black!50] at (axis cs:11,{i}) {{checkpoint lost}};")
                continue
            e0, et = r["fp"] + r["fn"], r["fpv"] + r["fnv"]
            col = "black!70" if et == e0 else ("cC" if et < e0 else "c4")
            if et != e0:
                out.append(f"\\draw[-latex, {col}, line width=0.8pt] (axis cs:{e0},{i}) -- (axis cs:{et},{i});")
            out.append(f"\\addplot[only marks, mark=o, mark size=2pt, black, forget plot] coordinates {{({e0},{i})}};")
            out.append(f"\\addplot[only marks, mark=*, mark size=1.6pt, {col}, forget plot] coordinates {{({et},{i})}};")
        if k == 0:
            out.append(r"\addlegendimage{only marks, mark=o, black}\addlegendentry{at 0.5}")
            out.append(r"\addlegendimage{only marks, mark=*, black!70}\addlegendentry{at validation-tuned $t$}")
            out.append(r"\addlegendimage{-latex, cC, line width=0.8pt}\addlegendentry{fewer errors}")
            out.append(r"\addlegendimage{-latex, c4, line width=0.8pt}\addlegendentry{more errors}")
    out.append(r"\end{groupplot}" + "\n" + r"\end{tikzpicture}" + "\n" + POST)
    return "\n".join(out)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("fig_spread", fig_spread), ("fig_fpfn", fig_fpfn), ("fig_thr", fig_thr)):
        with open(os.path.join(OUT, name + ".tex"), "w") as f:
            f.write(fn())
        print("wrote", name + ".tex")
