#!/usr/bin/env python3
"""Round-2 edit script, part 10 (2026-10-28): the w_s = 2L placeholders of the 6 x 3 x 2 series are filled with the
numbers available after the evaluation of 2026-10-28, 08:10 (runs S1G_3D_vM_ws2.0 and S2G_3D_solid_vM_ws2.0 of 076).

Final (plain numbers):
  J_eff(2L) = 1.40 J_c^0 = 0.97 J_c^num (exact 1.4016 / 0.9697; the maximum of J_x in the evaluated cycle between the
  third and the fourth pore, 0.970 at x_ct = 11.8L, is passed, J_x has fallen to 0.78 at x_ct = 13L);
  J_c^num(2L domain) = 1.445 J_c^0 (plateau fully covered; was the preliminary 1.446).
Provisional (new macro \\prelim{...}, red without brackets; run so far up to x_ct = 13L):
  crack-front statistics for 2L: x_ct inside a pore's x-range in 17 % of the steps, maximum advance 0.11L per step.
Still missing (\\pending kept): energy partition per cycle and G_diss/J_eff for 2L (the cycle closes at x_ct = 13.5L),
  Fig. 19b; Fig. 18 shows the 2L curve up to x_ct = 13L only.
Four-pore check specimen vs 6 x 3 x 2 (normalised J_eff): +1.5/+0.9/+0.5/+0.4 % for w_s = 0.375L..L, -0.7 % at 2L
  (0.9625 vs 0.9697) -> "0.4 to 1.5 % above" restricted to w_s <= L, "at most 1.5 %" unchanged.

Files: manuscript_revised_template.tex, response_to_reviewers_template.tex (generated quote blocks of the letter are
rebuilt by build_response_quotes.py afterwards). One-shot: fails on the first anchor that is gone (already applied).
"""
HERE = "/Users/alexanderschluter/Work/Hypo/Hypo/Simulation/dolfinx_alex/shared/scripts/046_plasticity/KLAIM_2025/revision_2/"
MS = HERE + "manuscript_revised_template.tex"
LT = HERE + "response_to_reviewers_template.tex"


class Doc:
    def __init__(self, path):
        self.path = path
        self.t = open(path, encoding="utf-8").read()

    def rep(self, old, new, n=1):
        c = self.t.count(old)
        if c != n:
            raise SystemExit("%s: anchor found %d times (expected %d):\n%s" % (self.path, c, n, old[:200]))
        self.t = self.t.replace(old, new)

    def save(self):
        open(self.path, "w", encoding="utf-8").write(self.t)


ms = Doc(MS)
lt = Doc(LT)

# ============================================================================ manuscript
# ---------------------------------------------------------------- preamble: \pending comment, new \prelim macro
ms.rep(
    r"\newcommand{\pending}[1]{\textcolor{red}{[#1]}}% placeholder for numbers of the two 3D runs for w_s = 2L (6x3x2 series) still running on 2026-10-27; stays visible in the clean build" + "\n",
    r"\newcommand{\pending}[1]{\textcolor{red}{[#1]}}% placeholder: partition numbers and Fig. 19b of the w_s = 2L run still running on 2026-10-28; stays visible in the clean build" + "\n"
    + r"\newcommand{\prelim}[1]{\textcolor{red}{#1}}% provisional value of the running w_s = 2L run (red, no brackets), check when the run is finished" + "\n",
)

# ---------------------------------------------------------------- revision working note (removed in the clean build)
ms.rep(
    "Red bracketed placeholders (macro pending) mark every number that depends on the two $w_s = 2L$ runs still running.}",
    "Red bracketed placeholders (macro pending) mark every number that depends on the two $w_s = 2L$ runs still running. "
    "Part 10 (edits\\_round2\\_part10\\_ws2\\_current.py, 2026-10-28): $w_s = 2L$ filled with the evaluation of 2026-10-28 "
    "($J_{x,c}^{\\text{eff}} = 1.40J_c^0 = 0.97J_c^{\\text{num}}$, $J_c^{\\text{num}} = 1.445J_c^0$, final); crack-front "
    "statistics of the running $2L$ case provisional (macro prelim, red without brackets); placeholders remain for the "
    "energy partition of $2L$ (Section~5.2, Fig.~\\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}b) and for the incomplete $2L$ "
    "curve of Fig.~\\ref{fig:test12q2_3D}.}",
)

# ---------------------------------------------------------------- Fig. 18 caption
ms.rep(
    "normalized by $J_c^{\\text{num}}$ of the three-dimensional homogeneous material. \\pending{$w_s = 2L$ to be added}}{Section of the J-Integral",
    "normalized by $J_c^{\\text{num}}$ of the three-dimensional homogeneous material. \\pending{$w_s = 2L$ shown up to $x_{ct} = 13L$, to be completed}}{Section of the J-Integral",
)

# ---------------------------------------------------------------- Fig. 19 caption
ms.rep(
    "\\pending{$w_s = 2L$ of the $6 \\times 3 \\times 2$ specimen to be added in a) and b)}}{Effective crack resistance",
    "\\pending{$w_s = 2L$ to be added in b)}}{Effective crack resistance",
)

# ---------------------------------------------------------------- Section 5.2: crack-front statistics (provisional for 2L)
ms.rep(
    "The crack front advances by at most $0.13$ to $0.19L$ per time step, and $x_{ct}$ lies within the $x$-range of a pore in $38$ to $79\\,\\%$ of the time steps ($w_s = L$ to $0.375L$, \\pending{$2L$ to be added}), which is only possible if the front has advanced beside the sphere.",
    "The crack front advances by at most \\prelim{$0.11$} to $0.19L$ per time step, and $x_{ct}$ lies within the $x$-range of a pore in \\prelim{$17$} to $79\\,\\%$ of the time steps ($w_s = 2L$ to $0.375L$), which is only possible if the front has advanced beside the sphere.",
)

# ---------------------------------------------------------------- Section 5.2: solid reference of the 2L domain (final)
ms.rep(
    "It is $1.421J_c^0$ for the domain of $w_s = L$ and $\\pending{1.446}J_c^0$ for the larger domain of $w_s = 2L$, which agrees with the two-dimensional value $1.425J_c^0$",
    "It is $1.421J_c^0$ for the domain of $w_s = L$ and $1.445J_c^0$ for the larger domain of $w_s = 2L$, which agrees with the two-dimensional value $1.425J_c^0$",
)

# ---------------------------------------------------------------- Section 5.2: Fig. 18 text, J_eff values, four-pore comparison
ms.rep(
    "is shown for the wall widths in Fig.~\\ref{fig:test12q2_3D} \\pending{$w_s = 2L$ to be added}. $J_x$ varies smoothly",
    "is shown for the wall widths in Fig.~\\ref{fig:test12q2_3D}. $J_x$ varies smoothly",
)
ms.rep(
    "$J_{x,c}^{\\text{eff}} = 1.22$, $1.26$, $1.31$, $1.34$ and $\\pending{1.39}J_c^0$, i.e., $0.86$, $0.88$, $0.92$, $0.95$ and $\\pending{0.96}J_c^{\\text{num}}$ for $w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and $2L$.",
    "$J_{x,c}^{\\text{eff}} = 1.22$, $1.26$, $1.31$, $1.34$ and $1.40J_c^0$, i.e., $0.86$, $0.88$, $0.92$, $0.95$ and $0.97J_c^{\\text{num}}$ for $w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and $2L$.",
)
ms.rep(
    "i.e., $0.4$ to $1.5\\,\\%$ above the six-pore specimen for $w_s \\le L$, so that the number of pores in the crack-growth direction has little influence",
    "i.e., $0.4$ to $1.5\\,\\%$ above the six-pore specimen for $w_s \\le L$ and $0.7\\,\\%$ below it for $w_s = 2L$, so that the number of pores in the crack-growth direction has little influence",
)

# ---------------------------------------------------------------- Section 5.2: comparison with 2D (merged range)
ms.rep(
    "the effective crack resistance does not exceed the homogeneous value for any of the computed wall widths, $0.86$ to $0.95J_c^{\\text{num}}$ for $w_s = 0.375L$ to $L$ and $\\pending{0.96}J_c^{\\text{num}}$ for $w_s = 2L$, and the curve is nearly flat.",
    "the effective crack resistance does not exceed the homogeneous value for any of the computed wall widths, $0.86$ to $0.97J_c^{\\text{num}}$ for $w_s = 0.375L$ to $2L$, and the curve is nearly flat.",
)

# ---------------------------------------------------------------- conclusion (merged range)
ms.rep(
    "is $0.86$ to $0.95$ for $w_s = 0.375L$ to $L$ and $\\pending{0.96}$ for $w_s = 2L$, nearly independent of the wall width, does not exceed the homogeneous value",
    "is $0.86$ to $0.97$ for $w_s = 0.375L$ to $2L$, nearly independent of the wall width, does not exceed the homogeneous value",
)

ms.save()

# ============================================================================ response letter (hand-written parts only;
# the %@generated blocks are rebuilt by build_response_quotes.py)
# ---------------------------------------------------------------- preamble: \pending comment, new \prelim macro
lt.rep(
    r"\newcommand{\pending}[1]{\textcolor{red}{[#1]}}% placeholder for numbers of the two 3D runs for w_s = 2L (6x3x2 series) still running on 2026-10-27" + "\n",
    r"\newcommand{\pending}[1]{\textcolor{red}{[#1]}}% placeholder: partition numbers and Fig. 19b of the w_s = 2L run still running on 2026-10-28" + "\n"
    + r"\newcommand{\prelim}[1]{\textcolor{red}{#1}}% provisional value of the running w_s = 2L run (red, no brackets), check when the run is finished" + "\n",
)

# ---------------------------------------------------------------- minor 25: list of regenerated figures
lt.rep(
    "the sections between pores two and four for all wall widths \\pending{$w_s = 2L$ to be added}, and the effective crack resistance together with the energy partition of both models)",
    "the sections between pores two and four for all wall widths, and the effective crack resistance together with the energy partition of both models)",
)

# ---------------------------------------------------------------- R3-C3: 3D values (merged range)
lt.rep(
    "$J_{x,c}^{\\text{eff}} = 0.86$ to $0.95\\,J_c^{\\mathrm{num}}$ for $w_s = 0.375L$ to $L$ and $\\pending{0.96}\\,J_c^{\\mathrm{num}}$ at $w_s = 2L$\n",
    "$J_{x,c}^{\\text{eff}} = 0.86$ to $0.97\\,J_c^{\\mathrm{num}}$ for $w_s = 0.375L$ to $2L$\n",
)

# ---------------------------------------------------------------- Additional corrections, item 4 (results of the reruns)
lt.rep(
    "and $\\pending{1.39}\\,J_c^0$, i.e., $0.86$, $0.88$, $0.92$, $0.95$ and $\\pending{0.96}$ in units of the numerical fracture resistance of the\n",
    "and $1.40\\,J_c^0$, i.e., $0.86$, $0.88$, $0.92$, $0.95$ and $0.97$ in units of the numerical fracture resistance of the\n",
)
lt.rep(
    "von~Mises material above the homogeneous value ($\\pending{0.96}$ at $w_s = 2L$, against $1.34$ in two dimensions), whereas for thin\n",
    "von~Mises material above the homogeneous value ($0.97$ at $w_s = 2L$, against $1.34$ in two dimensions), whereas for thin\n",
)

# ---------------------------------------------------------------- Additional corrections, item 9 (mesh / four-pore specimen)
lt.rep(
    "as a check of the influence of the number of pores in the crack-growth direction: its normalised effective crack resistances\n"
    "lie $0.4$ to $1.5\\,\\%$ above those of the $6 \\times 3 \\times 2$ specimen (Sections~4.5 and 5.2, response to Reviewer~1,\n"
    "point~2).",
    "as a check of the influence of the number of pores in the crack-growth direction: its normalised effective crack resistances\n"
    "lie $0.4$ to $1.5\\,\\%$ above those of the $6 \\times 3 \\times 2$ specimen for $w_s \\le L$ and $0.7\\,\\%$ below at $w_s = 2L$,\n"
    "i.e., within $1.5\\,\\%$ (Sections~4.5 and 5.2, response to Reviewer~1, point~2).",
)

# ---------------------------------------------------------------- Additional corrections, item 11 (normalisation)
lt.rep(
    "domain of $w_s = L$, $\\pending{1.446}\\,J_c^0$ for $w_s = 2L$; $1.415$, $1.444$ and $1.351\\,J_c^0$ for the shorter four-pore\n",
    "domain of $w_s = L$, $1.445\\,J_c^0$ for $w_s = 2L$; $1.415$, $1.444$ and $1.351\\,J_c^0$ for the shorter four-pore\n",
)

lt.save()

print("part 10 applied:", MS, LT)
