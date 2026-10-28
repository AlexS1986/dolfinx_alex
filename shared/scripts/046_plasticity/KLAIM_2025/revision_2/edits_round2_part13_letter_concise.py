#!/usr/bin/env python3
"""Round-2 edit script, part 13 (2026-10-28): response letter only.

(1) Consistency of the letter prose with manuscript_revised_clean.tex:
    - R1 major 3, sensitivity: "-2 to +37 %" -> "-12 to +37 %" (Table 2: H = 0.11 mu_0 gives -12 % at w_s = 0.375L).
    - R1 major 4: 0.375L is not "the smallest wall width" (0.25L is, A = 0.870); it is the thinnest wall of the
      embedding test series. The old-mesh 3D cubic pair ("-0.22 %") was dropped from the manuscript in part 11
      (Section 4.5: "no separate three-dimensional test was performed") -> removed from the letter.
    - R1 major 3 / R3 comment 1: "yields across its (full) width" hedged as in Section 4.4 ("the recorded maxima
      indicate that ...").
    - relhead "Captions of Figs. 15 to 19" -> "15 to 20" (the block quotes Fig. 20).
    - five manuscript changes that answer reviewer points were not quoted: Section 2 (cycle definition of
      J_eff), Section 3.5 (3D reference), caption of Fig. 8 and Section 5.1 (isotropic approximation, side
      layers), caption of Fig. 17 -> %@quote lines added.
(2) Conciseness: \\small quote blocks, note on the --old-max 400 quote build, 11 duplicate quotes replaced by
    cross-references, four prose passages tightened (R1 point 1 reference list, R1 major 3 item 2, R1 minor 25,
    R3 comment 3 tail).

Only response_to_reviewers_template.tex is touched. Backup: scratchpad backup_pre_part13/. One-shot: stops at
the first anchor that is not found exactly n times.
"""
HERE = "/Users/alexanderschluter/Work/Hypo/Hypo/Simulation/dolfinx_alex/shared/scripts/046_plasticity/KLAIM_2025/revision_2/"
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

    def cut(self, start_anchor, end_anchor, new):
        """Replace everything from start_anchor (inclusive) up to end_anchor (exclusive); both unique."""
        for a in (start_anchor, end_anchor):
            c = self.t.count(a)
            if c != 1:
                raise SystemExit("%s: cut anchor found %d times (expected 1):\n%s" % (self.path, c, a[:200]))
        i = self.t.index(start_anchor)
        j = self.t.index(end_anchor)
        if j <= i:
            raise SystemExit("%s: cut anchors in wrong order" % self.path)
        self.t = self.t[:i] + new + self.t[j:]

    def save(self):
        open(self.path, "w", encoding="utf-8").write(self.t)


lt = Doc(LT)

# ============================================================================ Task 1: consistency
# R1 major 3, sensitivity range (Table 2: -12/+19 % for H, -2/+37 % for sigma_y at w_s = 0.375L)
lt.rep(
    "In the nucleation regime the changes are $-2$ to $+37\\,\\%$, in the propagation regime $-6$ to $+6\\,\\%$, which confirms the two regimes.",
    "In the nucleation regime the changes are $-12$ to $+37\\,\\%$, in the propagation regime $-6$ to $+6\\,\\%$, which confirms the two regimes.",
)
# R1 major 4: "smallest wall width" and the dropped old-mesh 3D pair
lt.rep(
    "For the two-dimensional unit cells, the Zener ratio is $0.89$ for the smallest wall width $w_s = 0.375L$, $0.96$ for $w_s = L$ and $1.00$ for $w_s = 4L$, i.e., the anisotropy is moderate and vanishes for widely spaced pores.",
    "For the two-dimensional unit cells, the Zener ratio is $0.89$ for $w_s = 0.375L$, the thinnest wall of the test series described next, $0.96$ for $w_s = L$ and $1.00$ for $w_s = 4L$, i.e., the anisotropy is moderate and vanishes for widely spaced pores.",
)
lt.rep(
    "i.e., the approximation has no influence on any conclusion of the paper, not even for the thinnest walls where the anisotropy is largest.",
    "i.e., the approximation has no influence on any conclusion of the paper, not even for the thinnest wall of the test series, for which the anisotropy is largest.",
)
lt.rep(
    "(Table~4 in Section~5.1, which also gives the effective Poisson's ratios, $0.23$ to $0.25$). An earlier three-dimensional run for $w_s = L$ on a coarser mesh with the full cubic stiffness in the embedding changed $J_{x,c}^{\\text{eff}}$ by $-0.22\\,\\%$ relative to its isotropic twin, which is within the numerical scatter of those runs. Section~5.1 refers to Section~4.5 for this test.",
    "(Table~4 in Section~5.1, which also gives the effective Poisson's ratios, $0.23$ to $0.25$). For this reason, no separate three-dimensional test of the approximation was performed, which is stated in Section~4.5, and Section~5.1 refers to Section~4.5 for the test.",
)
# R3 comment 1: hedge as in Section 4.4
lt.rep(
    "The thin ligament of the von~Mises material yields across its full width ($\\alpha_{\\max} \\approx 0.7$ at a triaxiality of $0.67$ to $0.74$, i.e., the state of the one-dimensional bar at fracture) and then opens at nearly constant stress,",
    "The recorded maxima indicate that the thin ligament of the von~Mises material yields across its width ($\\alpha_{\\max} \\approx 0.7$ at a triaxiality of $0.67$ to $0.74$, i.e., the state of the one-dimensional bar at fracture) and then opens at nearly constant stress,",
)
# relhead of the Section 5 captions block
lt.rep(
    "\\relhead{Captions of Figs.~15 to 19 (Section~5)}",
    "\\relhead{Captions of Figs.~15 to 20 (Section~5)}",
)
# unquoted manuscript changes that answer reviewer points
lt.rep(
    "%@quote cmd:Tetrahedral mesh of the $6 \\times 3 \\times 2$ specimen\n%@quote cmd:, three-dimensional model}\n",
    "%@quote cmd:Tetrahedral mesh of the $6 \\times 3 \\times 2$ specimen\n%@quote cmd:, three-dimensional model}\n%@quote cmd:Crack propagation through the central row of pores in the three-dimensional model\n",
)
lt.rep(
    "%@quote cmd:homogeneous matrix with the effective elastic stiffness of a unit cell\n%@quote cmd:The transition from the resolved region to the homogeneous embedding\n",
    "%@quote cmd:homogeneous matrix with the effective elastic stiffness of a unit cell\n%@quote cmd:The transition from the resolved region to the homogeneous embedding\n%@quote cmd:a rectangular domain (blue) that is assigned the isotropic approximation\n%@quote cmd:The two layers of thickness $w_c/4$ between the porous block\n",
)
lt.rep(
    "\\relhead{Harmonized equations and parameters}\n%@quote env:eq:psi_fr\n",
    "\\relhead{Harmonized equations and parameters}\n%@quote cmd:Hossain et al.\\ take the maximum over the complete separation process\n%@quote cmd:The corresponding three-dimensional homogeneous reference simulations\n%@quote env:eq:psi_fr\n",
)

# ============================================================================ Task 2: conciseness
# 2.1 note on the shortened quotes
lt.rep(
    "Each bullet starts with the location of the change in the revised manuscript. Equation, figure, table and reference\nnumbers in the quotes refer to the marked-up revised manuscript.",
    "Each bullet starts with the location of the change in the revised manuscript. For long rewritten or removed\npassages, only the new text is quoted; the removed text is shown struck out in the marked manuscript. Equation,\nfigure, table and reference numbers in the quotes refer to the marked-up revised manuscript.",
)
# 2.2 quote blocks in \small
lt.rep(
    "{\\par\\noindent\\textbf{Related changes in the manuscript.}\\begin{itemize}}",
    "{\\par\\noindent\\textbf{Related changes in the manuscript.}\\begin{itemize}\\small}",
)

# 2.3 duplicate quotes -> cross-references (first occurrence kept)
# R3 comment 2: both quotes are part of the rewritten introduction under R1 major 1
lt.cut(
    "\\relhead{Introduction}\n%@quote cmd:For elastic and plastic properties, the step from the microstructure\n",
    "\\end{relatedchanges}\n\n\\begin{reviewercomment}\nComment 3:",
    "\\relhead{Introduction}\n\\item \\msloc{Introduction} see the paragraphs ``For elastic and plastic properties, \\dots'' and ``Rigorous results exist \\dots'' of the rewritten introduction, quoted under Reviewer~1, major point~1.\n",
)
# R1 minor 2: "Our previous work ..." is quoted under major point 1
lt.cut(
    "\\relhead{Introduction}\n%@quote cmd:Our previous work considered only reversible elastic material behavior\n",
    "\\end{relatedchanges}\n\n\\begin{reviewercomment}\n3. The title of section",
    "\\relhead{Introduction}\n\\item \\msloc{Introduction} see the paragraph ``Our previous work \\dots'' quoted under major point~1.\n",
)
# R1 minor 3: title of Section 2 is quoted under major point 2
lt.cut(
    "\\relhead{Title of Section~2}\n%@quote cmd:A Numerical Experiment for the Crack Resistance of a Resolved Porous Region\n",
    "\\end{relatedchanges}\n\n\\begin{reviewercomment}\n4. equations are not words",
    "\\relhead{Title of Section~2}\n\\item \\msloc{Section~2} see the first quote under major point~2.\n",
)
# R1 minor 10 and 11: both sentences are quoted under major point 2
lt.cut(
    "\\relhead{Section~2}\n%@quote cmd:This definition does not imply that\n",
    "\\end{relatedchanges}\n\n\\begin{reviewercomment}\n11. ``Independence from the initial",
    "\\relhead{Section~2}\n\\item \\msloc{Section~2} see the quote ``This definition does not imply \\dots'' under major point~2.\n",
)
lt.cut(
    "\\relhead{Section~2}\n%@quote cmd:cannot be taken for granted and has to be tested\n",
    "\\end{relatedchanges}\n\n\\begin{reviewercomment}\n12. There is still text between section and subsection",
    "\\relhead{Section~2}\n\\item \\msloc{Section~2} see the quote ``Whether $J_{x,c}^{\\text{eff}}$ is independent \\dots'' under major point~2.\n",
)
# R1 minor 24: the Section 4.4 paragraph is quoted under major point 3
lt.rep(
    "%@quote cmd:Strictly speaking, these relations hold only in one dimension. In two and three dimensions\n%@quote cmd:For the two linear elastic sets, re-nucleation occurs\n",
    "%@quote cmd:Strictly speaking, these relations hold only in one dimension. In two and three dimensions\n",
)
lt.rep(
    "%@generated-end\n\\end{relatedchanges}\n\n\\begin{reviewercomment}\n25. The plotted figures",
    "%@generated-end\n\\item \\msloc{Section~4.4} the paragraph ``For the two linear elastic sets, re-nucleation occurs \\dots'' is quoted under major point~3.\n\\end{relatedchanges}\n\n\\begin{reviewercomment}\n25. The plotted figures",
)
# R3 comment 3: the two Section 4.2 paragraphs are quoted under R1 major point 3
lt.rep(
    "\\relhead{Section~4.2, new paragraph (scaling of the linear elastic sets), and Section~3.5}\n%@quote cmd:The ratio of the values of \\EqJcLabel and \\EqSigmaStarLabel, $1.088/0.604\n%@quote cmd:The two linear elastic parameter sets differ only in the value of $J_c$\n%@quote cmd:The three comparisons in Fig.\n%@quote cmd:The comparison with the two-dimensional model in Fig.\n",
    "\\relhead{Sections~3.5 and 5.2; the two new paragraphs of Section~4.2 (scaling of the linear elastic sets, the three comparisons of Fig.~12) are quoted under Reviewer~1, major point~3}\n%@quote cmd:The ratio of the values of \\EqJcLabel and \\EqSigmaStarLabel, $1.088/0.604\n%@quote cmd:The comparison with the two-dimensional model in Fig.\n",
)
# Further changes: Section 3.3 (minor 19) and caption of Fig. 10 (minor 25) are quoted there
lt.rep(
    "%@quote cmd:Kolosov constant\n%@quote cmd:The weak forms of the coupled boundary-value problems are implemented in FEniCSx using \\texttt{dolfinx} version 0.7.3\n",
    "%@quote cmd:Kolosov constant\n",
)
lt.rep(
    "%@quote cmd:It is trapped at each pore until re-nucleation\n%@quote cmd:and the three parameter sets. Dashed lines mark the pore edges\n",
    "%@quote cmd:It is trapped at each pore until re-nucleation\n",
)
lt.rep(
    "In three dimensions, the effective crack resistance is thus close to the energy dissipated per unit crack advance for the evaluation cycle, not pointwise along the crack path.}\n%@generated-end\n\\end{relatedchanges}\n\n\\end{document}",
    "In three dimensions, the effective crack resistance is thus close to the energy dissipated per unit crack advance for the evaluation cycle, not pointwise along the crack path.}\n%@generated-end\n\\item \\msloc{Section~3.3} the paragraph on the implementation and the time step control is quoted under Reviewer~1, minor point~19.\n\\item \\msloc{Caption of Fig.~10} quoted under Reviewer~1, minor point~25.\n\\end{relatedchanges}\n\n\\end{document}",
)

# 2.4 prose tightened where redundant
# R1 major 1: the ten added references are named in the preceding sentences (all but Francfort and Marigo)
lt.rep(
    "In total, ten references were added (Hill 1963; Braides et al.\\ 1996; Francfort and Marigo 1998; Cagnetti et al.\\ 2019; Gao and Rice 1989; Bower and Ortiz 1991; Lebihain et al.\\ 2020; Alessi et al.\\ 2014; Borden et al.\\ 2016; Ambati et al.\\ 2015), so that the manuscript now cites 44 instead of 34 references, and the reference list is ordered by first citation.",
    "In total, ten references were added (the ones named above and Francfort and Marigo 1998 for the variational formulation of brittle fracture), so that the manuscript now cites 44 instead of 34 references, and the reference list is ordered by first citation.",
)
# R1 major 3, item 2: the mechanism of the thin-wall regime is given in full under R3 comment 1
lt.rep(
    "For $w_s \\le 0.6L$ the ligament yields across its width ($\\alpha_{\\max} \\approx 0.7$, triaxiality $0.67$ to $0.74$) and re-nucleates at about the one-dimensional fracture stress after a large plastic strain, as the one-dimensional bar of Section~3.6, which is the reduced model for this regime. Once yielded, the tangent stiffness of the ligament drops to the hardening modulus, the ligament opens at nearly constant stress and $J_x$ at re-nucleation stays low, whereas the linear elastic ligament of equal strength has to be loaded to a higher driving force before it reaches $\\sigma^*$. The plastic work done on the ligament remains part of the energy balance; the decisive effect is that yielding lowers the peak stress ahead of the pore, so that re-nucleation is reached at a lower driving force, hence the reduction of the effective crack resistance below \\EqSigmaStarLabel. For $w_s \\ge L$",
    "For $w_s \\le 0.6L$ the recorded maxima indicate that the ligament yields across its width ($\\alpha_{\\max} \\approx 0.7$, triaxiality $0.67$ to $0.74$) and re-nucleates at about the one-dimensional fracture stress after a large plastic strain, as the one-dimensional bar of Section~3.6, which is the reduced model for this regime. Yielding lowers the peak stress ahead of the pore, so that re-nucleation is reached at a lower driving force than in the linear elastic ligament of equal strength, hence the reduction of the effective crack resistance below \\EqSigmaStarLabel (the mechanism is described in detail in the response to Reviewer~3, comment~1). For $w_s \\ge L$",
)
# R1 minor 25: the contents of Figs. 16 to 20 are given by the quoted captions
lt.rep(
    "The result figures of Section~5 were regenerated from the new three-dimensional simulations in the same way (Figs.~16 to 20: crack tip position vs.\\ pseudo-time, an illustration of the crack propagating through the central row of pores, the J-integral records of the two- and the three-dimensional model for $w_s = L$ in one plot, the sections between pores two and four for all wall widths, and the effective crack resistance together with the energy partition of both models). The former figure",
    "The result figures of Section~5 were regenerated from the new three-dimensional simulations in the same way (Figs.~16 to 20, see the quoted captions). The former figure",
)
# R3 comment 3, tail: paraphrased the Section 4.2 paragraph that is quoted under R1 major point 3
lt.cut(
    "an increased crack resistance. Section~4.2 now states which question each comparison of Fig.~12 answers: for equal\n",
    "\\end{response}\n\n\\begin{relatedchanges}\n\\relhead{Sections~3.5 and 5.2;",
    "an increased crack resistance. Section~4.2 now states which question each of the three comparisons of Fig.~12 answers.\nFor equal crack resistance of the homogeneous material, which is the comparison asked for by the reviewer, the\nvon~Mises set lies below the linear elastic curve for all wall widths (Fig.~12b). The gain of the von~Mises set over\n\\EqSigmaStarLabel in Fig.~12a stems from the material model, which already raises the crack resistance of the\nhomogeneous material ($+31\\,\\%$ against \\EqJcLabel with the same $J_c^0$, a factor $2.36$ against \\EqSigmaStarLabel),\nand the microstructure preserves only part of this gain: the pores raise the crack resistance of the linear elastic\nmaterials by a factor of about two relative to their homogeneous value, that of the plastic material only by $1.34$,\nand below $w_s \\approx 0.9L$ they even lower it. The mechanism (Sections~4.3 and 4.4, response to Comment~1) is that\ncrack trapping and forced re-nucleation at the blunt pore wall toughen a linear elastic material efficiently, at the\nprice of $47$ to $83\\,\\%$ of the dissipated energy going into the unstable snap-through, whereas the homogeneous\nplastic material already blunts its crack by the plastic zone, which is contained in $J_c^{\\mathrm{num}}$. The two\nparagraphs added at the end of Section~4.2 with this argument are quoted under Reviewer~1, major point~3.\n",
)

lt.save()
print("part 13 applied to response_to_reviewers_template.tex")
