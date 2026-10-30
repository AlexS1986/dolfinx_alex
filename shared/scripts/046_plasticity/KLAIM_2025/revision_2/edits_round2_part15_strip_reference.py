#!/usr/bin/env python3
"""Round-2 edit script, part 15 (2026-10-30): final 3D normalisation by the homogeneous reference WITH elastic side
layers ("strip" reference, runs S2G_3D_strip_vM_ws1.0 / ws2.0 of 076), all \\prelim removed.

Ground truth: 075_klaim_rev2_2D/evaluation_output/{master_table.csv, Jc_num_solid_3D.csv, energy_cycles.csv}.
  strip reference      1.268 J_c^0 (domain w_s = L)   1.343 J_c^0 (domain w_s = 2L)
  plastic through z    1.421 / 1.445 J_c^0 (agree with 2D 1.425 within 2 %; side layers lower the reference by 11 % / 7 %)
  6x3x2: J_eff/J_c^num = 0.964 0.989 1.035 1.060 1.044  (w_s = 0.375L 0.5L 0.75L L 2L)
  4x3x2 (own plastic reference x strip/plastic ratio): 0.978 0.998 1.040 1.064 1.036;  h = 0.07L: 1.074 (+0.9 %)
  cycle maxima of the 6x3x2 run w_s = L (energy_cycles.csv Jmax / 1.2684): 1.092 1.075 1.060 1.049 1.009

Files: manuscript_revised_template.tex (M1-M11), response_to_reviewers_template.tex (own prose, L1-L6),
response_to_editor_template.tex (E1), make_review_version.py (NOTE string, N1). One-shot: fails loudly on the first
anchor that is gone or not unique. Backups: scratchpad backup_pre_part15/.
"""
HERE = "/Users/alexanderschluter/Work/Hypo/Hypo/Simulation/dolfinx_alex/shared/scripts/046_plasticity/KLAIM_2025/revision_2/"
MS = HERE + "manuscript_revised_template.tex"
LT = HERE + "response_to_reviewers_template.tex"
ED = HERE + "response_to_editor_template.tex"
RV = HERE + "make_review_version.py"


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


PRELIM_DEF = (
    "\\newcommand{\\prelim}[1]{\\textcolor{red}{#1}}% provisional (red, no brackets): normalised 3D values, which "
    "await the homogeneous references with elastic side layers (strip runs submitted 2026-10-28)\n"
)

# ================================================================ manuscript
ms = Doc(MS)

# M1 macro definition removed
ms.rep(PRELIM_DEF, "")

# M2 abstract (new part of the \replaced block)
ms.rep(
    "and \\prelim{the effective crack resistance does not exceed the homogeneous value for wall widths up to $2L$}.}",
    "and the effective crack resistance stays within $-4\\,\\%$ and $+6\\,\\%$ of the homogeneous value for wall "
    "widths up to $2L$.}",
)

# M3 Revision Working Note (internal)
ms.rep(
    "(expected shift a few percent upward; caveat in Section~\\ref{sec:3D_fracture_behavior}).}",
    "(expected shift a few percent upward; caveat in Section~\\ref{sec:3D_fracture_behavior}). Part 15 "
    "(edits\\_round2\\_part15\\_strip\\_reference.py, 2026-10-30): 3D normalisation by the reference with elastic side "
    "layers ($1.268$ / $1.343J_c^0$), \\textbackslash prelim removed.}",
)

# M4 Section 3.5: three-dimensional reference = strip definition
ms.rep(
    "\\added{The corresponding three-dimensional homogeneous reference simulations, carried out for the parameter set "
    "\\vMLabel only, yield $J_c^{\\text{num}} = 1.42J_c^0$, and $1.445J_c^0$ for the longer specimen used for "
    "$w_s = 2L$, see Section~\\ref{sec:3D_fracture_behavior}.}",
    "\\added{The corresponding three-dimensional homogeneous reference simulations, carried out for the parameter set "
    "\\vMLabel only, use specimens without pores on the domains and meshes of $w_s = L$ and $2L$, with the von~Mises "
    "material assigned to the block and the same linear elastic layers on the $z$ faces as in the porous specimens, "
    "Section~\\ref{sec:3D_effective_medium}. They yield $J_c^{\\text{num}} = 1.27J_c^0$ for the domain of $w_s = L$ "
    "and $1.34J_c^0$ for the longer domain of $w_s = 2L$, see Section~\\ref{sec:3D_fracture_behavior}. A reference "
    "that is plastic through the whole thickness gives $1.42J_c^0$ and $1.445J_c^0$, in agreement with the "
    "two-dimensional value within $2\\,\\%$; the side layers, which fracture elastically, lower the reference by "
    "$11\\,\\%$ and $7\\,\\%$, respectively.}",
)

# M5 Section 5.1: definition and values of the reference, four-pore / finer-mesh references
ms.rep(
    "For the three-dimensional model, $J_c^{\\text{num}}$ is determined from a specimen without pores, with the yield "
    "stress assigned to the whole domain, constructed and meshed by the same rules as the porous specimens, as the mean "
    "of $J_x$ over the central part of the crack path. It is $1.421J_c^0$ for the domain of $w_s = L$ and $1.445J_c^0$ "
    "for the larger domain of $w_s = 2L$, which agrees with the two-dimensional value $1.425J_c^0$ of "
    "Section~\\ref{sec:steady_growth_griffith} within $2\\,\\%$. The simulations for $w_s \\le L$ are normalized by the "
    "former value and the simulation for $w_s = 2L$ by the latter. The four-pore specimen is normalized by its own "
    "references, $1.415J_c^0$ ($w_s \\le L$) and $1.444J_c^0$ ($w_s = 2L$), which differ from the above by less than "
    "$0.5\\,\\%$, and by $1.351J_c^0$ for the finer mesh, $h = 0.07L$.}",
    "For the three-dimensional model, $J_c^{\\text{num}}$ is determined from a specimen without pores on the domain and "
    "the mesh of the porous specimen, with the yield stress assigned to the block and the same linear elastic layers on "
    "the $z$ faces as in the porous specimens, Section~\\ref{sec:3D_effective_medium}, as the mean of $J_x$ over the "
    "central part of the crack path. It is $1.268J_c^0$ for the domain of $w_s = L$ and $1.343J_c^0$ for the larger "
    "domain of $w_s = 2L$; the simulations for $w_s \\le L$ are normalized by the former value and the simulation for "
    "$w_s = 2L$ by the latter. A reference that is plastic through the whole thickness gives $1.421J_c^0$ and "
    "$1.445J_c^0$, which agrees with the two-dimensional value $1.425J_c^0$ of "
    "Section~\\ref{sec:steady_growth_griffith} within $2\\,\\%$ and confirms the consistency of the two- and the "
    "three-dimensional setups; the elastic side layers lower the reference by $11\\,\\%$ and $7\\,\\%$. The four-pore "
    "specimen and the finer mesh, $h = 0.07L$, are normalized by the references of their own meshes (plastic through "
    "the thickness: $1.415J_c^0$ for $w_s \\le L$, $1.444J_c^0$ for $w_s = 2L$ and $1.351J_c^0$ for $h = 0.07L$), "
    "scaled by the ratio between the two reference types measured on the six-pore domains.}",
)

# M6 Section 5.1: the red caveat paragraph (an \added block that never existed in the submitted version) is removed
ms.rep(
    "\\prelim{The values normalized by $J_c^{\\text{num}}$ in this section are provisional: the homogeneous reference "
    "is plastic through the whole thickness, whereas in the porous specimens about $20\\,\\%$ of the thickness "
    "fractures within the elastic side layers of Section~\\ref{sec:3D_effective_medium}. A reference with identical "
    "side layers is being computed; the normalized values are expected to shift upward by a few percent.}\n\n",
    "",
)

# M7 Section 5.2, Fig. 17 paragraph: cycle maxima of w_s = L (energy_cycles.csv Jmax / 1.2684) and the drop of the
#    strip reference towards the end of the domain (last cycle window: 0.938 of the plateau)
ms.rep(
    "with maxima of \\prelim{$0.98$, $0.96$, $0.95$, $0.94$ and $0.90J_c^{\\text{num}}$} over the five pore-to-pore "
    "cycles, i.e., \\prelim{below the crack resistance of the homogeneous material}. Part of the decrease along the "
    "specimen is shared by the homogeneous reference, whose $J_x$ drops by about $5\\,\\%$ towards the end of the "
    "domain,",
    "with maxima of $1.09$, $1.07$, $1.06$, $1.05$ and $1.01J_c^{\\text{num}}$ over the five pore-to-pore cycles, "
    "i.e., $1$ to $9\\,\\%$ above the crack resistance of the homogeneous material. Part of the decrease along the "
    "specimen is shared by the homogeneous reference, whose $J_x$ drops by about $6\\,\\%$ towards the end of the "
    "domain,",
)

# M8 Section 5.2, Fig. 18 paragraph: normalised six-pore and four-pore values
ms.rep(
    "i.e., \\prelim{$0.86$, $0.88$, $0.92$, $0.95$ and $0.97J_c^{\\text{num}}$} for $w_s = 0.375L$, $0.5L$, $0.75L$, "
    "$L$ and $2L$.",
    "i.e., $0.96$, $0.99$, $1.03$, $1.06$ and $1.04J_c^{\\text{num}}$ for $w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and "
    "$2L$.",
)
ms.rep(
    "gives \\prelim{$0.87$, $0.89$, $0.93$, $0.95$ and $0.96J_c^{\\text{num}}$}, open triangles",
    "gives $0.98$, $1.00$, $1.04$, $1.06$ and $1.04J_c^{\\text{num}}$, open triangles",
)

# M9 Section 5.2, comparison paragraph
ms.rep(
    "In three dimensions, \\prelim{the effective crack resistance does not exceed the homogeneous value for any of the "
    "computed wall widths, $0.86$ to $0.97J_c^{\\text{num}}$ for $w_s = 0.375L$ to $2L$}, and the increase with the "
    "wall width is weak.",
    "In three dimensions, the effective crack resistance stays close to the homogeneous value, $0.96$ to "
    "$1.06J_c^{\\text{num}}$ for $w_s = 0.375L$ to $2L$; it lies $1$ to $4\\,\\%$ below it for $w_s \\le 0.5L$ and "
    "$3$ to $6\\,\\%$ above it for $w_s \\ge 0.75L$, and the increase with the wall width is weak.",
)
ms.rep(
    "\\prelim{$0.86$} against $0.50J_c^{\\text{num}}$ at $w_s = 0.375L$.",
    "$0.96$ against $0.50J_c^{\\text{num}}$ at $w_s = 0.375L$.",
)
ms.rep(
    "The two curves cross \\prelim{between $w_s = 0.75L$ ($0.92$ against $0.91J_c^{\\text{num}}$) and $w_s = L$ "
    "($0.95$ against $1.10J_c^{\\text{num}}$)}, the three-dimensional",
    "The two curves cross between $w_s = 0.75L$ ($1.03$ against $0.91J_c^{\\text{num}}$) and $w_s = L$ ($1.06$ "
    "against $1.10J_c^{\\text{num}}$), the three-dimensional",
)

# M10 conclusion
ms.rep(
    "normalized by the crack resistance of the three-dimensional homogeneous material, \\prelim{increases only from "
    "$0.86$ to $0.97$ for $w_s = 0.375L$ to $2L$ and does not exceed the homogeneous value}; the normalized value",
    "normalized by the crack resistance of the three-dimensional homogeneous material, increases only from $0.96$ at "
    "$w_s = 0.375L$ to $1.06$ at $w_s = L$ and is $1.04$ at $2L$, i.e., it stays within $-4\\,\\%$ and $+6\\,\\%$ of "
    "the homogeneous value; the normalized value",
)
ms.rep(
    "$0.86$ against $0.50$ at $w_s = 0.375L$, and below them for $w_s \\ge L$.",
    "$0.96$ against $0.50$ at $w_s = 0.375L$, and below them for $w_s \\ge L$.",
)

# M11 captions of Figs. 3, 17, 18, 19: they say "normalized by J_c^num" only -> nothing to change (checked).

assert "\\prelim{" not in ms.t, "prelim left in manuscript"
assert "\\pending{" not in ms.t, "pending left in manuscript"
ms.save()

# ================================================================ response to the reviewers (own prose only)
lt = Doc(LT)

# L1 macro definition removed
lt.rep(PRELIM_DEF, "")

# L2 summary list of changes
lt.rep(
    "and the effective crack resistance does not exceed that of the homogeneous material for any of the computed wall "
    "widths (Section~5.2).",
    "and the effective crack resistance stays within $-4$ and $+6\\,\\%$ of that of the homogeneous material for the "
    "computed wall widths (Section~5.2).",
)

# L3 R1 major 3, energy partition item: G_diss crossover (cycle-2 G_diss of [vM]/[Eq sigma*]: 0.85/0.96 at 0.5L)
lt.rep(
    "(smaller for $w_s \\le 0.375L$, equal at $0.6L$, larger for wide walls)",
    "(smaller for $w_s \\le 0.5L$, equal at $0.6L$, larger for wide walls)",
)

# L4 R3 C3 answer
lt.rep(
    "(Section~5.2), the pores do not raise the crack resistance of the von~Mises set above its homogeneous value at all:\n"
    "\\prelim{$J_{x,c}^{\\text{eff}} = 0.86$ to $0.97\\,J_c^{\\mathrm{num}}$} for $w_s = 0.375L$ to $2L$\n"
    "(with $J_c^{\\mathrm{num}}$ the value of the three-dimensional solid specimen), against $1.34\\,J_c^{\\mathrm{num}}$ "
    "in two dimensions.",
    "(Section~5.2), the pores change the crack resistance of the von~Mises set by at most $6\\,\\%$:\n"
    "$J_{x,c}^{\\text{eff}} = 0.96$ to $1.06\\,J_c^{\\mathrm{num}}$ for $w_s = 0.375L$ to $2L$\n"
    "(with $J_c^{\\mathrm{num}}$ the value of the three-dimensional homogeneous reference of Section~5.1), against "
    "$1.34\\,J_c^{\\mathrm{num}}$ in two dimensions.",
)

# L5 "Further changes by the authors"
lt.rep(
    "resistance of a three-dimensional homogeneous specimen computed with the same mesh rule. The two-dimensional",
    "resistance of a three-dimensional homogeneous specimen computed with the same mesh rule and the same elastic side "
    "layers. The two-dimensional",
)

# L6 "Notes on this document": no mention of red/provisional values (checked) -> nothing to change.

assert "\\prelim{" not in lt.t.split("%@generated-begin")[0], "prelim left in letter preamble"
lt.save()

# ================================================================ editor letter
ed = Doc(ED)
ed.rep(
    "all pores and normalized by a three-dimensional homogeneous reference, and its result changed: the crack front "
    "advances beside\nthe pores and the effective crack resistance no longer exceeds that of the homogeneous material. "
    "The abstract and the conclusion\nwere rewritten accordingly.",
    "all pores and normalized by a three-dimensional homogeneous reference with the same elastic side layers, and its "
    "result changed: the crack front\nadvances beside the pores and the effective crack resistance stays within a few "
    "percent of that of the homogeneous material\ninstead of the toughening reported before. The abstract and the "
    "conclusion were rewritten accordingly.",
)
# E2 the added words pushed the signature line onto a second page -> vertical spacing tightened (5mm -> 4mm twice,
#    4mm/6mm -> 3mm around "Sincerely"); the letter is one page again
ed.rep("\\vspace{4mm}\n\\noindent Sincerely,\n\n\\vspace{6mm}\n\\noindent Alexander",
       "\\vspace{3mm}\n\\noindent Sincerely,\n\n\\vspace{3mm}\n\\noindent Alexander")
ed.rep("\\vspace{5mm}", "\\vspace{4mm}", 2)
ed.save()

# ================================================================ make_review_version.py: NOTE string
rv = Doc(RV)
a = rv.t.find("One set of results is still provisional")
b = rv.t.find("the red marking will be removed before submission.\n")
assert 0 < a < b, "NOTE paragraph anchors not found"
b += len("the red marking will be removed before submission.\n")
NEW = (
    "Compared with the version of 28 October, the three-dimensional values normalized by the homogeneous reference\n"
    "changed: the reference now carries the same linear elastic side layers as the porous specimens\n"
    "(Sections~\\ref{sec:steady_growth_griffith} and~\\ref{sec:3D_effective_medium}), which lowers it by $11\\,\\%$\n"
    "($w_s = L$) and $7\\,\\%$ ($2L$). The normalized values are $0.96$ to $1.06$ instead of $0.86$ to $0.97$, so that\n"
    "the porous material now slightly exceeds the homogeneous value for $w_s \\ge 0.75L$\n"
    "(Section~\\ref{sec:3D_fracture_behavior}, abstract, conclusion). No value is provisional any more. This note will\n"
    "be removed before submission.\n"
)
rv.t = rv.t[:a] + NEW + rv.t[b:]
assert "textcolor" not in rv.t.split("NOTE = r")[1].split('"""')[1], "red mention left in NOTE"
rv.save()

print("part 15 applied: manuscript M1-M10 (15 anchors), letter L1-L5 (5 anchors), editor E1, review NOTE N1")
