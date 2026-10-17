#!/usr/bin/env python3
"""Round-2 edits, part 4 (2026-10-17): Section 3.6 (one-dimensional bar) recomputed in 077_klaim_rev2_1D_bar with
beta = 0.2, mu = 1, lam = 0, eta = 1e-5, eps = 0.1, M = 1000 (RESULTS.md there):
  sigma* 1.31 -> 1.28 (vM), 1.30 -> 1.28 (Eq sigma*), Eq J_c 1.72 unchanged; J_c of [Eq sigma*] 0.5679 -> 0.5523;
  eps* (strain at which the stress collapses) 1.18/1.08/0.82 -> 1.17/1.07/0.80; Pi* 2.16/2.22/1.29 -> 2.12/2.19/1.23;
  weak zone +-3 Delta h -> +-7.5 Delta h (= +-0.15 L, as in the scripts); figures replaced by the 077 versions.
Usage: python3 edits_round2_part4_1Dbar.py manuscript_revised_template.tex
       python3 edits_round2_part4_1Dbar.py response_to_reviewers_template.tex
"""
import re
import sys

path = sys.argv[1]
s = open(path).read()

if "manuscript" in path:
    E = [
        # parameter-set definition (Sec. 3.4)
        ("with fracture resistance $J_c=0.5679J_c^0$ and $\\lambda=\\mu_0$.",
         "with fracture resistance \\replaced{$J_c=0.5523J_c^0$}{$J_c=0.5679J_c^0$} and $\\lambda=\\mu_0$."),
        ("is chosen such that the one-dimensional tensile strength is close to that of \\vMLabel",
         "is chosen such that the one-dimensional tensile strength \\replaced{equals}{is close to} that of \\vMLabel"),
        # weak zone
        ("the fracture resistance $J_c$ was locally reduced by 1\\% within a region extending from $x=-3\\Delta h$ to $x=3\\Delta h$.",
         "the fracture resistance $J_c$ was locally reduced by 1\\% within a region extending from \\replaced{$x=-7.5\\Delta h$ to $x=7.5\\Delta h$, i.e.\\ $|x| \\le 0.15L$}{$x=-3\\Delta h$ to $x=3\\Delta h$}."),
        # fracture stresses and strains
        ("yielded a significantly higher fracture stress of $\\sigma^* = 1.72\\sigma_y$ than the von~Mises-type model \\vMLabel, which resulted in $\\sigma^* = 1.31\\sigma_y$. However, the fracture strain was smaller in the linear elastic case, with $\\varepsilon^* \\approx 1.08\\sqrt{\\nicefrac{J_c^0}{\\mu_0 L}}$ for \\EqJcLabel compared to $\\varepsilon^* \\approx 1.18\\sqrt{\\nicefrac{J_c^0}{\\mu_0 L}}$ for \\vMLabel, as shown in Fig.~\\ref{fig:reaction_forces}.",
         "yielded a significantly higher fracture stress of $\\sigma^* = 1.72\\sigma_y$ than the von~Mises-type model \\vMLabel, which resulted in \\replaced{$\\sigma^* = 1.28\\sigma_y$}{$\\sigma^* = 1.31\\sigma_y$}. However, the fracture strain\\added{ $\\varepsilon^*$, defined as the strain at which the stress drops to zero,} was smaller in the linear elastic case, with \\replaced{$\\varepsilon^* \\approx 1.07\\sqrt{\\nicefrac{J_c^0}{\\mu_0 L}}$}{$\\varepsilon^* \\approx 1.08\\sqrt{\\nicefrac{J_c^0}{\\mu_0 L}}$} for \\EqJcLabel compared to \\replaced{$\\varepsilon^* \\approx 1.17\\sqrt{\\nicefrac{J_c^0}{\\mu_0 L}}$}{$\\varepsilon^* \\approx 1.18\\sqrt{\\nicefrac{J_c^0}{\\mu_0 L}}$} for \\vMLabel, as shown in Fig.~\\ref{fig:reaction_forces}."),
        ("the fracture resistance was reduced to $J_c = 0.5679J_c^0$ in \\EqSigmaStarLabel. This adjustment resulted in an almost equivalent fracture stress of $\\sigma^* = 1.30\\sigma_y$ compared to the von~Mises model, but a significantly lower fracture strain of only $\\varepsilon^* \\approx 0.82\\sqrt{\\nicefrac{J_c^0}{\\mu_0 L}}$.",
         "the fracture resistance was reduced to \\replaced{$J_c = 0.5523J_c^0$}{$J_c = 0.5679J_c^0$} in \\EqSigmaStarLabel\\added{, since the fracture stress of the linear elastic model is proportional to $\\sqrt{J_c}$}. This adjustment resulted in \\replaced{the same fracture stress of $\\sigma^* = 1.28\\sigma_y$ as for the von~Mises model}{an almost equivalent fracture stress of $\\sigma^* = 1.30\\sigma_y$ compared to the von~Mises model}, but a significantly lower fracture strain of only \\replaced{$\\varepsilon^* \\approx 0.80\\sqrt{\\nicefrac{J_c^0}{\\mu_0 L}}$}{$\\varepsilon^* \\approx 0.82\\sqrt{\\nicefrac{J_c^0}{\\mu_0 L}}$}."),
        # dissipated energies
        ("with $\\Pi^* \\approx 2.22J_c^0 L$, followed by the von~Mises simulation \\vMLabel with $\\Pi^* \\approx 2.16J_c^0 L$. In contrast, the linear elastic simulation \\EqSigmaStarLabel dissipates only $\\Pi^* \\approx 1.29J_c^0 L$.",
         "with \\replaced{$\\Pi^* \\approx 2.19J_c^0 L$}{$\\Pi^* \\approx 2.22J_c^0 L$}, followed by the von~Mises simulation \\vMLabel with \\replaced{$\\Pi^* \\approx 2.12J_c^0 L$}{$\\Pi^* \\approx 2.16J_c^0 L$}. In contrast, the linear elastic simulation \\EqSigmaStarLabel dissipates only \\replaced{$\\Pi^* \\approx 1.23J_c^0 L$}{$\\Pi^* \\approx 1.29J_c^0 L$}."),
        # figures: 077 versions
        ("            \\input{pics/PAPER_reaction_forces_1D_EQ_ONLY.pgf}%",
         "            \\input{pics/PAPER_reaction_forces_1D_rev2.pgf}% rev2: recomputed in 077 (beta = 0.2, J_c = 0.5523 for [Eq sigma*])"),
        ("            \\input{pics/PAPER_energy_vs_displacement_1D_EQ_ONLY.pgf}%",
         "            \\input{pics/PAPER_energy_vs_displacement_1D_rev2.pgf}% rev2: recomputed in 077"),
    ]
else:
    E = [
        ("\\item The FEniCSx version is 0.7.3 (not 0.7.4).",
         """\\item The FEniCSx version is 0.7.3 (not 0.7.4).
\\item The one-dimensional bar of Section~3.6 was recomputed with the parameters now used throughout ($\\beta = 0.2$, $\\eta =
10^{-5}$, $\\epsilon = 0.1L$, $M = 1000$, $\\Delta h = 0.02L$). The fracture stress of the von~Mises bar is $\\sigma^* = 1.28\\sigma_y$
(previously $1.31\\sigma_y$, a value belonging to $\\beta = 0.1$); the linear elastic bar with $J_c = J_c^0$ is unchanged at
$1.72\\sigma_y$. Since the fracture stress of the linear elastic model is proportional to $\\sqrt{J_c}$, the parameter set
\\EqSigmaStarLabel{} now uses $J_c = 0.5523J_c^0$ (previously $0.5679J_c^0$, which gave $1.30\\sigma_y$) and has exactly the
fracture stress of the von~Mises bar. The fracture strains and dissipated energies were updated accordingly
($\\varepsilon^* = 1.17,\\ 1.07,\\ 0.80$ and $\\Pi^* = 2.12,\\ 2.19,\\ 1.23\\,J_c^0L$ for \\vMLabel, \\EqJcLabel, \\EqSigmaStarLabel),
the fracture strain is now defined explicitly as the strain at which the stress drops to zero, and the width of the
weakened zone that triggers nucleation is stated as implemented ($|x| \\le 0.15L$). The finite-element peaks agree with a
rate-independent homogeneous solution of the one-dimensional problem to within $0.1\\,\\%$, and halving the loading rate or
increasing the mobility tenfold changes them by less than $0.5\\,\\%$. Figures~5 and~6 were regenerated."""),
        ("%@quote env:eq:surfing-bcs",
         """%@quote env:eq:surfing-bcs
%@quote cmd:$J_c=0.5523J_c^0$
%@quote cmd:equals}{is close to}
%@quote cmd:$x=-7.5\\Delta h$
%@quote cmd:1.31\\sigma_y
%@quote cmd:defined as the strain at which the stress drops to zero
%@quote cmd:1.07\\sqrt
%@quote cmd:1.17\\sqrt
%@quote cmd:$J_c = 0.5523J_c^0$
%@quote cmd:proportional to $\\sqrt{J_c}$
%@quote cmd:the same fracture stress of
%@quote cmd:0.80\\sqrt
%@quote cmd:2.19J_c^0 L
%@quote cmd:2.12J_c^0 L
%@quote cmd:1.23J_c^0 L"""),
    ]


def find_once(text, old):
    if text.count(old) == 1:
        return old
    pat = "".join(("[ \t]*\n" if ch == "\n" else re.escape(ch)) for ch in old)
    hits = list(re.finditer(pat, text))
    return hits[0].group(0) if len(hits) == 1 else None


for old, new in E:
    hit = find_once(s, old)
    if hit is None:
        print("MATCH ERROR:", old[:100].replace("\n", " "), "count =", s.count(old))
        sys.exit(1)
    s = s.replace(hit, new)

open(path, "w").write(s)
print("applied", len(E), "edits to", path)
