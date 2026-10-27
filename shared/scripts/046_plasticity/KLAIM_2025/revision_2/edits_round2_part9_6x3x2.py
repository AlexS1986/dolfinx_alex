#!/usr/bin/env python3
"""Round-2 edit script, part 9 (2026-10-27): Section 5 switched to the final 6 x 3 x 2 series (076 runs S1G/S2G,
all 36 pores resolved as spheres), the intermediate 4 x 3 x 2 fixed-mesh series (S1F/S2F/S9F) is kept only as a check
(number of pores in the crack-growth direction, mesh convergence, Newton threshold). Two runs of the new series
(w_s = 2L porous and solid) were still running: every number that depends on them is a red \\pending{...} placeholder
(macro defined in the manuscript and the letter preamble, kept by make_clean_manuscript.py).

Files: manuscript_revised_template.tex, response_to_reviewers_template.tex, response_to_editor_template.tex.
One-shot: fails on the first anchor that is gone (already applied).
"""
HERE = "/Users/alexanderschluter/Work/Hypo/Hypo/Simulation/dolfinx_alex/shared/scripts/046_plasticity/KLAIM_2025/revision_2/"
MS = HERE + "manuscript_revised_template.tex"
LT = HERE + "response_to_reviewers_template.tex"
ED = HERE + "response_to_editor_template.tex"


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
ed = Doc(ED)

# ============================================================================ manuscript
# ---------------------------------------------------------------- preamble: \pending placeholder macro
ms.rep(
    r"\newcommand{\changed}[1]{\begingroup\color{blue}#1\endgroup}" + "\n",
    r"\newcommand{\changed}[1]{\begingroup\color{blue}#1\endgroup}" + "\n"
    + r"\newcommand{\pending}[1]{\textcolor{red}{[#1]}}% placeholder for numbers of the two 3D runs for w_s = 2L (6x3x2 series) still running on 2026-10-27; stays visible in the clean build" + "\n",
)

# ---------------------------------------------------------------- abstract (200 words, kept at 200)
ms.rep(
    "In three dimensions, with spherical pores, the crack front bypasses the pores without unstable ligament failure, and the effective crack resistance stays below the homogeneous value for all wall widths.}",
    "In three dimensions, with spherical pores, the crack front advances beside the pores without unstable ligament failure, and the effective crack resistance does not exceed the homogeneous value.}",
)

# ---------------------------------------------------------------- revision working note (removed in the clean build)
ms.rep(
    "old-mesh cubic pair at $w_s = 0.375L$ removed).}",
    "old-mesh cubic pair at $w_s = 0.375L$ removed). Part 9 (edits\\_round2\\_part9\\_6x3x2.py, 2026-10-27): Section~5 switched to the final "
    "$6 \\times 3 \\times 2$ series with all pores resolved (runs S1G/S2G of 076), the $4 \\times 3 \\times 2$ series is kept only as a check "
    "(number of pores in $x$, mesh convergence, Newton threshold), Fig.~\\ref{fig:mesh3D} = view and section of the $6 \\times 3 \\times 2$ mesh. "
    "Red bracketed placeholders (macro pending) mark every number that depends on the two $w_s = 2L$ runs still running.}",
)

# ---------------------------------------------------------------- Section 4.5: specimen length in x is now tested in 3D
ms.rep(
    "Not tested were the influence of the length of the resolved region in the crack-growth direction, of other crack-growth directions relative to the pore lattice, of the position of the initial crack relative to the pores, and of the microstructure itself, since one periodic arrangement of equal pores is considered and no statistics over microstructure realizations is attempted.",
    "The length of the resolved region in the crack-growth direction was varied only in three dimensions, Section~\\ref{sec:3D_fracture_behavior}, where four instead of six pores in this direction change the normalized effective crack resistance by at most $1.5\\,\\%$. Not tested were the influence of other crack-growth directions relative to the pore lattice, of the position of the initial crack relative to the pores, and of the microstructure itself, since one periodic arrangement of equal pores is considered and no statistics over microstructure realizations is attempted.",
)

# ---------------------------------------------------------------- Section 5.1: Fig. 15 (mesh of the 6x3x2 specimen)
ms.rep(
    "        \\includegraphics[width=0.75\\textwidth]{pics/PAPER_rev2_3D_mesh_section_ws0.375.pdf}% rev2: section of the mesh of the fixed-mesh run S1F_3D_vM_ws0.375 (076), replaces pics/mesh_with_cs.png (old w_s-scaled mesh of the v2.0 series, w_s = 0.25L)\n"
    "        \\caption{\\replaced{Section of the tetrahedral mesh of the three-dimensional specimen for $w_s = 0.375L$ ($h = 0.1L$ in the crack band, $85\\,681$ tetrahedra) through the plane of the pore centers, $z = -w_c/2$. a) Whole cross-section of the domain, $(6w_c) \\times 20L$, the rectangle marks the region shown in b). b) Resolved porous region with the refined band around the crack plane $y = 0$ (dashed line). The polygons are the intersections of the tetrahedra with the section plane.}{Three-dimensional mesh for $w_s=0.25L$.}}",
    "        \\includegraphics[width=0.9\\textwidth, trim=10 46 2 36, clip]{pics/PAPER_rev2_3D_mesh_6x3x2_ws1.0.pdf}% rev2 (2026-10-27): mesh of the 6x3x2 run S1G_3D_vM_ws1.0 (076, all 36 pores resolved), replaces the section of the 4x3x2 mesh (PAPER_rev2_3D_mesh_section_ws0.375.pdf) and pics/mesh_with_cs.png of v2.0; trim = white margins of the PDF (content bbox 15 52 450 241 of 454 x 281 bp)\n"
    "        \\caption{\\replaced{Tetrahedral mesh of the $6 \\times 3 \\times 2$ specimen for $w_s = L$ ($353\\,921$ tetrahedra). a) View of the domain with the surface mesh of the back face $z = z_{\\max}$, which shows the refined band around the crack plane, and the surfaces of all 36 pores. b) Section through the plane of the pore centers $z = -w_c/2$ with the crack plane $y = 0$ (dashed). The polygons are the intersections of the tetrahedra with the section plane.}{Three-dimensional mesh for $w_s=0.25L$.}}",
)

# ---------------------------------------------------------------- Section 5.1: geometry 6x3x2 in (8 w_c) x 20L x (2.5 w_c)
ms.rep(
    "A \\replaced{$4\\times3\\times2$}{$6\\times3\\times2$} array of these unit cells is assembled to represent a simplified porous material. This array is centered about the plane $y=0$ and embedded into a surrounding cuboidal domain of effective stiffness with dimensions \\replaced{$(6 w_c)\\times 20L \\times (2.5 w_c)$, i.e., one cell width of embedding before and after the porous block in the $x$-direction and $w_c/4$ on each side in the $z$-direction}{$(8 w_c)\\times 20L \\times (4 w_c)$}.",
    "A $6\\times3\\times2$ array of these unit cells is assembled to represent a simplified porous material. This array is centered about the plane $y=0$ and embedded into a surrounding cuboidal domain of effective stiffness with dimensions \\replaced{$(8 w_c)\\times 20L \\times (2.5 w_c)$, i.e., one cell width of embedding before and after the porous block in the $x$-direction and $w_c/4$ on each side in the $z$-direction}{$(8 w_c)\\times 20L \\times (4 w_c)$}.",
)

# ---------------------------------------------------------------- Section 5.1: mesh rule incl. pore surfaces, four-pore check specimen
ms.rep(
    "\\replaced{The mesh of tetrahedral finite elements is refined in a band around the expected fracture plane $y=0$, Fig.~\\ref{fig:mesh3D}, with a target element size of $h = 0.1L$ for $|y| < 0.35L$, $2h$ for $|y| < 0.65L$ and at most $L$ in the remaining domain, independent of the wall width. The median edge length of the elements in the crack band is $1.49\\epsilon$, and the meshes contain $85\\,000$ ($w_s = 0.375L$) to $416\\,000$ ($w_s = 2L$) elements. For $w_s = L$, the simulation was repeated with $h = 0.07L$ (median edge length $1.03\\epsilon$, $505\\,000$ elements) to check the mesh dependence.}",
    "\\replaced{The mesh of tetrahedral finite elements is refined in a band around the expected fracture plane $y=0$, Fig.~\\ref{fig:mesh3D}, with a target element size of $h = 0.1L$ for $|y| < 0.35L$, $2h$ for $|y| < 0.65L$ and at most $L$ in the remaining domain, independent of the wall width, and the surfaces of all 36 pores are resolved with $h = 0.1L$, from which the element size grows to the far-field value over a distance of $L$. The median edge length of the elements in the crack band is $1.49\\epsilon$, the mean edge length on the pore surfaces is $0.10L$ in the middle row and $0.11L$ in the outer rows, and the meshes contain $184\\,000$ ($w_s = 0.375L$) to $741\\,000$ ($w_s = 2L$) elements. The checks reported in Section~\\ref{sec:3D_fracture_behavior} use a shorter specimen with four instead of six pores in the $x$-direction, embedded in $(6 w_c)\\times 20L \\times (2.5 w_c)$ and meshed by the same rule in the crack band, with the outer pore rows resolved more coarsely, on which the simulation for $w_s = L$ was repeated with $h = 0.07L$ (median edge length $1.03\\epsilon$, $505\\,000$ elements).}",
)

# ---------------------------------------------------------------- Section 5.1: Newton threshold (7 for the whole 6x3x2 series, validation on the four-pore specimen)
ms.rep(
    "\\added{The time step is controlled as in Section~\\ref{sec:numerical_implementation}, except that it is increased when a step converges in fewer than five, for the longest runs seven, Newton iterations, since the plastic ligaments require five to six iterations per step. For $w_s = L$, a repetition with the threshold seven instead of five changes $J_{x,c}^{\\text{eff}}$ by $+0.5\\,\\%$ and the record of $J_x$ by less than $0.1\\,\\%$ in the median, with $37\\,\\%$ fewer time steps.}",
    "\\added{The time step is controlled as in Section~\\ref{sec:numerical_implementation}, except that it is increased when a step converges in fewer than seven Newton iterations, since the plastic ligaments require five to six iterations per step. On the four-pore specimen with $w_s = L$, the thresholds seven and five give values of $J_{x,c}^{\\text{eff}}$ that differ by $0.5\\,\\%$ and records of $J_x$ that agree within $0.1\\,\\%$ in the median, with $37\\,\\%$ fewer time steps for the higher threshold.}",
)

# ---------------------------------------------------------------- Section 5.2: captions of Figs. 17-19
ms.rep(
    "Dashed lines mark the pore edges of the three-dimensional specimen (four pores). The picture at the top shows the phase field $s$ of the middle pore row of the two-dimensional specimen after complete separation, Fig.~\\ref{fig:PAPER_02_all_Jx_vs_xct_pf_diamond_holes}, for orientation.}",
    "Dashed lines mark the pore edges, which are the same in both models (six pores). The picture at the top shows the phase field $s$ of the middle pore row of the two-dimensional specimen after complete separation, Fig.~\\ref{fig:PAPER_02_all_Jx_vs_xct_pf_diamond_holes}, for orientation.}",
)
ms.rep(
    "\\caption{\\replaced{J-Integral $J_x$ vs.\\ crack tip position $x_{ct}$ between pores one and three of the three-dimensional specimen for the parameter set \\vMLabel and varying $w_s$, normalized by $J_c^{\\text{num}}$ of the three-dimensional homogeneous material.}",
    "\\caption{\\replaced{J-Integral $J_x$ vs.\\ crack tip position $x_{ct}$ between pores two and four of the three-dimensional specimen, the same section as in Fig.~\\ref{fig:PAPER_rev2_Jx_sections_three_sets}, for the parameter set \\vMLabel and varying $w_s$, normalized by $J_c^{\\text{num}}$ of the three-dimensional homogeneous material. \\pending{$w_s = 2L$ to be added}}",
)
ms.rep(
    "The open triangle marks the three-dimensional simulation on the finer mesh, $h = 0.07L$. b) Fractions of the energy dissipated per pore-to-pore cycle, eq.~(\\ref{eq:D_tot_cycle}), that create crack surface (fracture) and that are dissipated by the viscous evolution of $s$ (viscous) in the two- and the three-dimensional model.}",
    "Filled triangles: $6 \\times 3 \\times 2$ specimen, open triangles: shorter specimen with four pores in the $x$-direction, open square: the latter on the finer mesh, $h = 0.07L$. b) Fractions of the energy dissipated per pore-to-pore cycle, eq.~(\\ref{eq:D_tot_cycle}), that create crack surface (fracture) and that are dissipated by the viscous evolution of $s$ (viscous) in the two- and the three-dimensional ($6 \\times 3 \\times 2$) model. \\pending{$w_s = 2L$ of the $6 \\times 3 \\times 2$ specimen to be added in a) and b)}}",
)

# ---------------------------------------------------------------- Section 5.2: Fig. 18 at the width of Fig. 17 (length)
ms.rep(
    "    \\includegraphics[width=0.8\\textwidth]{pics/PAPER_rev2_3D_Jx_vs_xct_sections.pdf}",
    "    \\includegraphics[width=0.72\\textwidth]{pics/PAPER_rev2_3D_Jx_vs_xct_sections.pdf}",
)

# ---------------------------------------------------------------- Section 5.2: crack-front behaviour (6x3x2 data)
ms.rep(
    "The crack front advances by at most about $0.2L$ per time step. Since $x_{ct}$ passes through the $x$-range of each pore without a jump, the front separates the material beside the sphere while it passes the pore, i.e., it bypasses the spherical pores instead of being trapped as a whole, and the ligament between two pores is separated progressively rather than at once.}",
    "The crack front advances by at most $0.13$ to $0.19L$ per time step, and $x_{ct}$ lies within the $x$-range of a pore in $38$ to $79\\,\\%$ of the time steps ($w_s = L$ to $0.375L$, \\pending{$2L$ to be added}), which is only possible if the front has advanced beside the sphere. The front thus advances beside the spherical pores instead of being trapped as a whole, and the ligament between two pores is separated progressively rather than at once.}",
)

# ---------------------------------------------------------------- Section 5.2: normalisation (6x3x2 references, four-pore references)
ms.rep(
    "It is $1.415J_c^0$ for the domain of $w_s = L$ and $1.444J_c^0$ for the larger domain of $w_s = 2L$, which agrees with the two-dimensional value $1.425J_c^0$ of Section~\\ref{sec:steady_growth_griffith} within $2\\,\\%$. The simulations for $w_s \\le L$ are normalized by the former value, the simulation for $w_s = 2L$ by the latter, and the simulation on the finer mesh by its own reference, $1.351J_c^0$.}",
    "It is $1.421J_c^0$ for the domain of $w_s = L$ and $\\pending{1.446}J_c^0$ for the larger domain of $w_s = 2L$, which agrees with the two-dimensional value $1.425J_c^0$ of Section~\\ref{sec:steady_growth_griffith} within $2\\,\\%$. The simulations for $w_s \\le L$ are normalized by the former value and the simulation for $w_s = 2L$ by the latter. The four-pore specimen is normalized by its own references, $1.415J_c^0$ ($w_s \\le L$), $1.444J_c^0$ ($w_s = 2L$) and $1.351J_c^0$ ($h = 0.07L$), which differ from the above by less than $0.5\\,\\%$.}",
)

# ---------------------------------------------------------------- Section 5.2: Fig. 17 discussion (cycle maxima at w_s = L)
ms.rep(
    "with peak values of $0.97$, $0.95$ and $0.90J_c^{\\text{num}}$ for the first three ligaments, i.e., below the crack resistance of the homogeneous material. The last pore is affected by the end of the resolved region.",
    "with maxima of $0.98$, $0.96$, $0.95$, $0.94$ and $0.90J_c^{\\text{num}}$ over the five pore-to-pore cycles, i.e., below the crack resistance of the homogeneous material. Part of the decrease along the specimen is shared by the homogeneous reference, whose $J_x$ drops by about $5\\,\\%$ towards the end of the domain, relative to which the inner cycles agree within about $1.5\\,\\%$.",
)

# ---------------------------------------------------------------- Section 5.2: sections, J_eff values, mesh check and four-pore comparison
ms.rep(
    "\\replaced{The pattern between pores one and three, which contains the ligament in which $J_{x,c}^{\\text{eff}}$ is evaluated, is shown for the five wall widths in Fig.~\\ref{fig:test12q2_3D}. $J_x$ varies smoothly for all wall widths, and the undulation becomes shallower as the pores move apart. For thin walls, the maximum of $J_x$ is attained while $x_{ct}$ is still within the $x$-range of the pore, since the front passes beside the sphere. The effective crack resistance, i.e., the maximum of $J_x$ over the pore-to-pore cycle that contains the evaluation ligament, Section~\\ref{sec:2D_fracture_behavior}, increases only weakly with the wall width, Fig.~\\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}a): $J_{x,c}^{\\text{eff}} = 1.24$, $1.26$, $1.31$, $1.34$ and $1.39J_c^0$, i.e., $0.87$, $0.89$, $0.93$, $0.95$ and $0.96J_c^{\\text{num}}$ for $w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and $2L$. On the finer mesh, $h = 0.07L$, the absolute values for $w_s = L$ decrease by $3.6\\,\\%$ ($J_{x,c}^{\\text{eff}}$) and $4.5\\,\\%$ ($J_c^{\\text{num}}$), whereas the normalized value changes by only $+0.9\\,\\%$, open triangle in Fig.~\\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}a). The normalized effective crack resistance is therefore the mesh-robust quantity, and it is the one compared with the two-dimensional results in the following.}",
    "\\replaced{The pattern between pores two and four, which contains the ligament in which $J_{x,c}^{\\text{eff}}$ is evaluated, is shown for the wall widths in Fig.~\\ref{fig:test12q2_3D} \\pending{$w_s = 2L$ to be added}. $J_x$ varies smoothly for all wall widths, and the undulation becomes shallower as the pores move apart. For thin walls, the maximum of $J_x$ is attained while $x_{ct}$ is still within the $x$-range of the pore, since the front passes beside the sphere. The effective crack resistance, i.e., the maximum of $J_x$ over the pore-to-pore cycle of the middle ligament between the third and the fourth pore, as in two dimensions, Section~\\ref{sec:2D_fracture_behavior}, increases only weakly with the wall width, Fig.~\\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}a): $J_{x,c}^{\\text{eff}} = 1.22$, $1.26$, $1.31$, $1.34$ and $\\pending{1.39}J_c^0$, i.e., $0.86$, $0.88$, $0.92$, $0.95$ and $\\pending{0.96}J_c^{\\text{num}}$ for $w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and $2L$. On the four-pore specimen of Section~\\ref{sec:3D_effective_medium}, refining the mesh to $h = 0.07L$ decreases the absolute values for $w_s = L$ by $3.6\\,\\%$ ($J_{x,c}^{\\text{eff}}$) and $4.5\\,\\%$ ($J_c^{\\text{num}}$), whereas the normalized value changes by only $+0.9\\,\\%$, open square in Fig.~\\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}a). The normalized effective crack resistance is therefore the mesh-robust quantity and is used in the comparison with the two-dimensional results. The four-pore specimen itself, with the maximum of $J_x$ taken over the cycle of its middle ligament between the second and the third pore, gives $0.87$, $0.89$, $0.93$, $0.95$ and $0.96J_c^{\\text{num}}$, open triangles in Fig.~\\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}a), i.e., $0.4$ to $1.5\\,\\%$ above the six-pore specimen for $w_s \\le L$, so that the number of pores in the crack-growth direction has little influence on the normalized effective crack resistance.}",
)

# ---------------------------------------------------------------- Section 5.2: comparison with 2D
ms.rep(
    "In three dimensions, the effective crack resistance stays below the homogeneous value for all wall widths, $0.87$ to $0.96J_c^{\\text{num}}$, and saturates at about $0.96J_c^{\\text{num}}$ for $w_s = 2L$. For thin walls, on the other hand, the three-dimensional specimen loses much less of its crack resistance than the two-dimensional one, $0.87$ against $0.50J_c^{\\text{num}}$ at $w_s = 0.375L$. The two curves cross at $w_s \\approx 0.75L$ ($0.93$ against $0.91J_c^{\\text{num}}$), the three-dimensional values lie above the two-dimensional ones for thinner and below them for wider walls.",
    "In three dimensions, the effective crack resistance does not exceed the homogeneous value for any of the computed wall widths, $0.86$ to $0.95J_c^{\\text{num}}$ for $w_s = 0.375L$ to $L$ and $\\pending{0.96}J_c^{\\text{num}}$ for $w_s = 2L$, and the curve is nearly flat. For thin walls, on the other hand, the three-dimensional specimen loses much less of its crack resistance than the two-dimensional one, $0.86$ against $0.50J_c^{\\text{num}}$ at $w_s = 0.375L$. The two curves cross between $w_s = 0.75L$ ($0.92$ against $0.91J_c^{\\text{num}}$) and $w_s = L$ ($0.95$ against $1.10J_c^{\\text{num}}$), the three-dimensional values lie above the two-dimensional ones for thinner and below them for wider walls.",
)
ms.rep(
    "In three dimensions, the front bypasses the spherical pores, so that this mechanism is absent.",
    "In three dimensions, the front advances beside the spherical pores, so that this mechanism is absent.",
)

# ---------------------------------------------------------------- Section 5.2: energy partition (cycle 2 of the 6x3x2 runs), energy balance
ms.rep(
    "In three dimensions, $92$ to $94\\,\\%$ of the dissipated energy create crack surface, $6$ to $8\\,\\%$ remain stored as plastic energy and less than $1\\,\\%$ is dissipated by the viscous evolution of the phase field, for all wall widths, compared with a viscous share of $12$ to $37\\,\\%$ in two dimensions, which collects the energy released in the unstable snap-through of the ligaments. The energy dissipated per unit crack advance, $G_{\\text{diss}} = 1.11$ to $1.44J_c^0$, is close to $J_{x,c}^{\\text{eff}}$, $G_{\\text{diss}}/J_{x,c}^{\\text{eff}} = 0.89$ to $1.04$ for $w_s = 0.375L$ to $2L$, whereas the ratio is $0.73$ to $0.93$ in two dimensions, where $J_{x,c}^{\\text{eff}}$ is a sharp peak and $G_{\\text{diss}}$ the average over the cycle.",
    "In three dimensions, $91$ to $93\\,\\%$ of the dissipated energy create crack surface, $6$ to $9\\,\\%$ remain stored as plastic energy and at most $1\\,\\%$ is dissipated by the viscous evolution of the phase field, for $w_s = 0.375L$ to $L$ \\pending{$2L$ to be added}, compared with a viscous share of $12$ to $37\\,\\%$ in two dimensions, which collects the energy released in the unstable snap-through of the ligaments. In the three-dimensional simulations, the external work and the sum of the stored and dissipated energies, eq.~(\\ref{eq:energy_balance}), agree to within $0.4\\,\\%$ ($0.1\\,\\%$ for $w_s = L$). The energy dissipated per unit crack advance, $G_{\\text{diss}} = 1.17$ to $1.31J_c^0$, is close to $J_{x,c}^{\\text{eff}}$, $G_{\\text{diss}}/J_{x,c}^{\\text{eff}} = 0.95$ to $0.97$ for $w_s = 0.375L$ to $L$ \\pending{$2L$ to be added}, whereas the ratio is $0.73$ to $0.93$ in two dimensions, where $J_{x,c}^{\\text{eff}}$ is a sharp peak and $G_{\\text{diss}}$ the average over the cycle.",
)

# ---------------------------------------------------------------- conclusion: 3D paragraph and limitations
ms.rep(
    "\\replaced{The three-dimensional model with spherical pores behaves differently. The crack front bypasses the pores in small increments, no unstable snap-through of a ligament occurs, and the viscous share of the dissipated energy is below $1\\,\\%$, so that neither the toughening by forced re-nucleation nor the strong weakening of thin ligaments of the two-dimensional model occurs. The effective crack resistance, normalized by the crack resistance of the three-dimensional homogeneous material, is $0.87$ to $0.96$, nearly independent of the wall width, never above the homogeneous value and robust against the mesh. It lies above the two-dimensional values for thin walls, $0.87$ against $0.50$ at $w_s = 0.375L$, and below them for $w_s \\ge L$.",
    "\\replaced{The three-dimensional model with spherical pores behaves differently. The crack front advances beside the pores in small increments, no unstable snap-through of a ligament occurs, and the viscous share of the dissipated energy is at most $1\\,\\%$, so that neither the toughening by forced re-nucleation nor the strong weakening of thin ligaments of the two-dimensional model occurs. The effective crack resistance, normalized by the crack resistance of the three-dimensional homogeneous material, is $0.86$ to $0.95$ for $w_s = 0.375L$ to $L$ and $\\pending{0.96}$ for $w_s = 2L$, nearly independent of the wall width, does not exceed the homogeneous value and is robust against the mesh and the number of pores in the crack-growth direction. It lies above the two-dimensional values for thin walls, $0.86$ against $0.50$ at $w_s = 0.375L$, and below them for $w_s \\ge L$.",
)
ms.rep(
    "Representativity in the sense of Hill is not established, since statistics over microstructure realizations, other crack-growth directions and the size of the specimen in the crack-growth direction were not tested.",
    "Representativity in the sense of Hill is not established, since statistics over microstructure realizations and other crack-growth directions were not tested, and the size of the specimen in the crack-growth direction only in three dimensions.",
)

ms.save()

# ============================================================================ response letter
# ---------------------------------------------------------------- preamble
lt.rep(
    r"\newcommand{\mycomment}[2]{\textbf{\textcolor{red}{[#1: #2]}}}" + "\n",
    r"\newcommand{\mycomment}[2]{\textbf{\textcolor{red}{[#1: #2]}}}" + "\n"
    + r"\newcommand{\pending}[1]{\textcolor{red}{[#1]}}% placeholder for numbers of the two 3D runs for w_s = 2L (6x3x2 series) still running on 2026-10-27" + "\n",
)

# ---------------------------------------------------------------- editor response, item 6
lt.rep(
    "with spherical pores, the crack front bypasses the pores, and the effective crack resistance stays below that of the homogeneous material for all wall widths (Section~5.2).",
    "with spherical pores, the crack front advances beside the pores, and the effective crack resistance does not exceed that of the homogeneous material for any of the computed wall widths (Section~5.2).",
)

# ---------------------------------------------------------------- Reviewer 1, major point 2: 4 vs 6 pores check
lt.rep(
    "Section~4.5 also states explicitly what was not tested (length of the resolved region in the crack-growth direction, other crack-growth directions and initial crack positions, other microstructures, statistics over realizations, amplitude of the K-field).",
    "In three dimensions, the length of the resolved region in the crack-growth direction was varied as well: a specimen with four instead of six pores in this direction changes the normalized effective crack resistance by at most $1.5\\,\\%$ (Section~5.2). Section~4.5 also states explicitly what was not tested (other crack-growth directions and initial crack positions, other microstructures, statistics over realizations, amplitude of the K-field, and the length of the resolved region in the crack-growth direction in two dimensions).",
)

# ---------------------------------------------------------------- Reviewer 1, minor point 25: Fig. 15, Fig. 18 range, quote anchors
lt.rep(
    "the sections between pores one and three for all wall widths, and the effective crack resistance together with the energy partition of both models). The former figure of the fields $s$ and $\\alpha$ for $w_s = 0.25L$ was removed, since this wall width is no longer part of the three-dimensional series, and Fig.~15 now shows a section through the mesh of the new series ($w_s = 0.375L$) instead of the mesh of the previous version.",
    "the sections between pores two and four for all wall widths \\pending{$w_s = 2L$ to be added}, and the effective crack resistance together with the energy partition of both models). The former figure of the fields $s$ and $\\alpha$ for $w_s = 0.25L$ was removed, since this wall width is no longer part of the three-dimensional series, and Fig.~15 now shows a three-dimensional view and a section of the mesh of the new series ($w_s = L$, all 36 pores resolved as spheres) instead of the mesh of the previous version.",
)
lt.rep("%@quote cmd:Section of the tetrahedral mesh of the three-dimensional specimen\n",
       "%@quote cmd:Tetrahedral mesh of the $6 \\times 3 \\times 2$ specimen\n")
lt.rep("%@quote cmd:between pores one and three of the three-dimensional specimen\n",
       "%@quote cmd:between pores two and four of the three-dimensional specimen\n")
lt.rep("%@quote cmd:The open triangle marks\n",
       "%@quote cmd:Filled triangles: $6 \\times 3 \\times 2$ specimen\n")

# ---------------------------------------------------------------- Reviewer 3, comment 3
lt.rep(
    "(Section~5.2), the pores do not raise the crack resistance of the von~Mises set above its homogeneous value at all:\n"
    "$J_{x,c}^{\\text{eff}}$ saturates at $0.96\\,J_c^{\\mathrm{num}}$ at $w_s = 2L$ (with $J_c^{\\mathrm{num}}$ the value of the\n"
    "three-dimensional solid specimen), against $1.34\\,J_c^{\\mathrm{num}}$ in two dimensions.",
    "(Section~5.2), the pores do not raise the crack resistance of the von~Mises set above its homogeneous value at all:\n"
    "$J_{x,c}^{\\text{eff}} = 0.86$ to $0.95\\,J_c^{\\mathrm{num}}$ for $w_s = 0.375L$ to $L$ and $\\pending{0.96}\\,J_c^{\\mathrm{num}}$ at $w_s = 2L$\n"
    "(with $J_c^{\\mathrm{num}}$ the value of the three-dimensional solid specimen), against $1.34\\,J_c^{\\mathrm{num}}$ in two dimensions.",
)

# ---------------------------------------------------------------- Additional corrections, item 4
lt.rep(
    "three-dimensional results of Section~5 were recomputed with these parameters and, in addition, with the geometry, mesh and\n"
    "normalisation of items~9 to~11. The effective crack resistances are now $J_{x,c}^{\\text{eff}} = 1.24$, $1.26$, $1.31$, $1.34$\n"
    "and $1.39\\,J_c^0$, i.e., $0.87$, $0.89$, $0.93$, $0.95$ and $0.96$ in units of the numerical fracture resistance of the\n"
    "three-dimensional solid, for $w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and $2L$.",
    "three-dimensional results of Section~5 were recomputed with these parameters and, in addition, with the mesh, thickness and\n"
    "normalisation of items~9 to~11. The effective crack resistances are now $J_{x,c}^{\\text{eff}} = 1.22$, $1.26$, $1.31$, $1.34$\n"
    "and $\\pending{1.39}\\,J_c^0$, i.e., $0.86$, $0.88$, $0.92$, $0.95$ and $\\pending{0.96}$ in units of the numerical fracture resistance of the\n"
    "three-dimensional solid, for $w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and $2L$.",
)
lt.rep(
    "von~Mises material above the homogeneous value ($0.96$ at $w_s = 2L$, against $1.34$ in two dimensions), whereas for thin\n"
    "walls the three-dimensional specimen loses much less than the two-dimensional one ($0.87$ against $0.50$ at $w_s = 0.375L$),",
    "von~Mises material above the homogeneous value ($\\pending{0.96}$ at $w_s = 2L$, against $1.34$ in two dimensions), whereas for thin\n"
    "walls the three-dimensional specimen loses much less than the two-dimensional one ($0.86$ against $0.50$ at $w_s = 0.375L$),",
)

# ---------------------------------------------------------------- Additional corrections, item 9 (mesh)
lt.rep(
    "at $w_s = 4L$) and the recorded $J_x$ increased with the wall width partly through the mesh. The new series uses a fixed\n"
    "element size $h = 0.1L$ in the crack band for all wall widths (median edge length $1.49\\epsilon$, between $85\\,000$ and\n"
    "$416\\,000$ tetrahedra), stated in Section~5.1, and the statement in Section~3.3 that the element size is at most half the\n"
    "regularization length is now restricted to two dimensions. A mesh-convergence check at $w_s = L$ with $h = 0.07L$\n"
    "($1.03\\epsilon$, $505\\,000$ tetrahedra) changes the absolute values by $-3.6\\,\\%$ ($J_{x,c}^{\\text{eff}}$) and $-4.5\\,\\%$\n"
    "(solid reference) and the normalised effective crack resistance by $+0.9\\,\\%$ (Section~5.2). Since fixed-size meshes for\n"
    "$w_s = 3L$ and $4L$ were out of reach, the three-dimensional series now covers $w_s = 0.375L$ to $2L$ (previously $0.25L$\n"
    "to $4L$).",
    "at $w_s = 4L$) and the recorded $J_x$ increased with the wall width partly through the mesh. The new series uses a fixed\n"
    "element size $h = 0.1L$ in the crack band for all wall widths (median edge length $1.49\\epsilon$) and resolves the surfaces\n"
    "of all 36 pores with the same element size (mean surface edge length $0.10L$ to $0.11L$, between $184\\,000$ and\n"
    "$741\\,000$ tetrahedra), stated in Section~5.1, and the statement in Section~3.3 that the element size is at most half the\n"
    "regularization length is now restricted to two dimensions. A mesh-convergence check at $w_s = L$ with $h = 0.07L$\n"
    "($1.03\\epsilon$, $505\\,000$ tetrahedra), carried out on a shorter specimen with four instead of six pores in the crack-growth\n"
    "direction (Section~5.1), changes the absolute values by $-3.6\\,\\%$ ($J_{x,c}^{\\text{eff}}$) and $-4.5\\,\\%$\n"
    "(solid reference) and the normalised effective crack resistance by $+0.9\\,\\%$ (Section~5.2). The same shorter specimen serves\n"
    "as a check of the influence of the number of pores in the crack-growth direction: its normalised effective crack resistances\n"
    "lie $0.4$ to $1.5\\,\\%$ above those of the $6 \\times 3 \\times 2$ specimen (Sections~4.5 and 5.2, response to Reviewer~1,\n"
    "point~2). Since fixed-size meshes for $w_s = 3L$ and $4L$ were out of reach, the three-dimensional series now covers\n"
    "$w_s = 0.375L$ to $2L$ (previously $0.25L$ to $4L$).",
)

# ---------------------------------------------------------------- Additional corrections, item 10 (thickness, Newton threshold; no 4x3x2 geometry claim)
lt.rep(
    "\\item The three-dimensional specimen now contains $4 \\times 3 \\times 2$ cells in an embedding of $(6w_c) \\times 20L \\times\n"
    "(2.5w_c)$ (previously $6 \\times 3 \\times 2$ cells in $(8w_c) \\times 20L \\times (4w_c)$), i.e., one cell width of embedding\n"
    "before and after the porous block and $w_c/4$ on each side in the $z$-direction, and the J-integral is normalised by the\n"
    "thickness $2.5w_c$ (Section~5.2). The time step control of the three-dimensional runs increases the step after fewer than\n"
    "five (for the longest runs seven) instead of four Newton iterations, which is stated in Section~5.1 together with a\n"
    "validation ($+0.5\\,\\%$ in $J_{x,c}^{\\text{eff}}$ for $w_s = L$).",
    "\\item The thickness of the three-dimensional specimen ($6 \\times 3 \\times 2$ cells, one cell width of embedding before and\n"
    "after the porous block) is $2.5w_c$, i.e., $w_c/4$ of embedding on each side of the two cell layers in the $z$-direction, and\n"
    "not $4w_c$ as stated in the previous version. The dimensions of the embedding are now given as $(8w_c) \\times 20L \\times\n"
    "(2.5w_c)$, and the J-integral is normalised by the thickness $2.5w_c$ (eq.~(40), Section~5.2). The time step control of the\n"
    "three-dimensional runs increases the step after fewer than seven instead of four Newton iterations, which is stated in\n"
    "Section~5.1 together with a validation on the shorter four-pore specimen ($0.5\\,\\%$ difference in $J_{x,c}^{\\text{eff}}$\n"
    "between the thresholds seven and five for $w_s = L$).",
)

# ---------------------------------------------------------------- Additional corrections, item 11 (normalisation)
lt.rep(
    "solid specimen without pores, computed with the same domain construction and the same mesh rule ($1.415\\,J_c^0$ for the\n"
    "domain of $w_s = L$, $1.444\\,J_c^0$ for $w_s = 2L$, $1.351\\,J_c^0$ on the finer mesh), which agrees with the two-dimensional\n"
    "value $1.425\\,J_c^0$ within $2\\,\\%$.",
    "solid specimen without pores, computed with the same domain construction and the same mesh rule ($1.421\\,J_c^0$ for the\n"
    "domain of $w_s = L$, $\\pending{1.446}\\,J_c^0$ for $w_s = 2L$; $1.415$, $1.444$ and $1.351\\,J_c^0$ for the shorter four-pore\n"
    "specimen with $h = 0.1L$, $w_s \\le L$ and $2L$, and with $h = 0.07L$), which agrees with the two-dimensional\n"
    "value $1.425\\,J_c^0$ within $2\\,\\%$.",
)
lt.rep(
    "The effective crack resistance is now evaluated as the maximum of $J_x$ over the\n"
    "pore-to-pore cycle that contains the evaluation ligament (stated in Section~4.2).",
    "The effective crack resistance is now evaluated as the maximum of $J_x$ over the\n"
    "pore-to-pore cycle of the middle ligament, between the third and the fourth pore (stated in Sections~4.2 and 5.2).",
)

# ---------------------------------------------------------------- quote anchors of the Section-5 block
lt.rep("%@quote cmd:$4\\times3\\times2$\n", "")
lt.rep("%@quote cmd:The pattern between pores one and three\n", "%@quote cmd:The pattern between pores two and four\n")

lt.save()

# ============================================================================ cover letter
ed.rep(
    "found in an audit of the code (listed at the end of the response letter). The two-dimensional conclusions are confirmed. The\n"
    "three-dimensional series was recomputed on meshes with a fixed element size and normalized by a three-dimensional homogeneous\n"
    "reference, and its result changed: with spherical pores, the crack front bypasses the pores and the effective crack resistance\n"
    "no longer exceeds that of the homogeneous material. The abstract and the conclusion were rewritten accordingly.",
    "found in an audit of the code (listed at the end of the response letter). The two-dimensional conclusions are confirmed. The\n"
    "three-dimensional series ($6 \\times 3 \\times 2$ spherical pores) was recomputed on meshes with a fixed element size that resolve\n"
    "all pores and normalized by a three-dimensional homogeneous reference, and its result changed: the crack front advances beside\n"
    "the pores and the effective crack resistance no longer exceeds that of the homogeneous material. The abstract and the conclusion\n"
    "were rewritten accordingly.",
)
ed.save()

print("part 9 applied:", MS, LT, ED)
