#!/usr/bin/env python3
"""Round-2 edit script, part 7b (2026-10-23, after part 7): decision of the first author that J_eff is the maximum of J_x over
the pore-to-pore cycle that contains the evaluation ligament (cycle definition of Section 4.3). 2D values unchanged (checked),
3D values for w_s = 0.375/0.5/0.75 L change (Fig. 19 regenerated with them, file names unchanged). The old-mesh cubic-embedding
pair at w_s = 0.375L (-0.37 %) is no longer comparable and is removed everywhere, only the pair at w_s = L (-0.22 %) remains.
One-shot: fails on the first anchor that is gone.
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
            raise SystemExit("%s: anchor found %d times (expected %d):\n%s" % (self.path, c, n, old[:160]))
        self.t = self.t.replace(old, new)

    def save(self):
        open(self.path, "w", encoding="utf-8").write(self.t)


ms = Doc(MS)
lt = Doc(LT)

# ---------------------------------------------------------------- Section 4.2: definition sentence
ms.rep(
    r"determines the effective \replaced{crack resistance}{strength} according to the definition in Section~\ref{sec:numericalExperiment}.",
    r"determines the effective \replaced{crack resistance}{strength} according to the definition in Section~\ref{sec:numericalExperiment}. \added{Throughout this work, $J_{x,c}^{\text{eff}}$ is evaluated as the maximum of $J_x$ over one pore-to-pore cycle in the sense of Section~\ref{sec:2D_energy_partition}, which in two dimensions coincides with the peak immediately before re-nucleation.}",
)

# ---------------------------------------------------------------- Section 4.5: only the w_s = L old-mesh pair
ms.rep(
    r"and two earlier three-dimensional runs on a coarser mesh with the full cubic stiffness changed $J_{x,c}^{\text{eff}}$ by $-0.37\,\%$ ($w_s = 0.375L$) and $-0.22\,\%$ ($w_s = L$).",
    r"and an earlier three-dimensional run on a coarser mesh with the full cubic stiffness changed $J_{x,c}^{\text{eff}}$ by $-0.22\,\%$ ($w_s = L$).",
)

# ---------------------------------------------------------------- Section 5.2: J_eff paragraph, red note removed
ms.rep(
    r"The effective crack resistance, taken as in two dimensions as the maximum of $J_x$ while the crack tip lies in the ligament ahead of a pore, here the third pore, increases with the wall width, Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}a): $J_{x,c}^{\text{eff}} = 1.11$, $1.12$, $1.30$, $1.34$ and $1.39J_c^0$, i.e., $0.79$, $0.79$, $0.92$, $0.95$ and $0.96J_c^{\text{num}}$ for $w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and $2L$.",
    r"For thin walls, the maximum of $J_x$ is attained while $x_{ct}$ is still within the $x$-range of the pore, since the front passes beside the sphere. The effective crack resistance, i.e., the maximum of $J_x$ over the pore-to-pore cycle that contains the evaluation ligament, Section~\ref{sec:2D_fracture_behavior}, increases only weakly with the wall width, Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}a): $J_{x,c}^{\text{eff}} = 1.24$, $1.26$, $1.31$, $1.34$ and $1.39J_c^0$, i.e., $0.87$, $0.89$, $0.93$, $0.95$ and $0.96J_c^{\text{num}}$ for $w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and $2L$.",
)
ms.rep(
    r"\mycomment{TODO}{Decision for the authors: for $w_s = 0.375L$ and $0.5L$ the window used for $J_{x,c}^{\text{eff}}$ (crack tip in the ligament ahead of the third pore) does not contain the maximum of $J_x$ of that pore-to-pore cycle, which occurs while the tracked tip is still within the $x$-range of the second pore, i.e., while the front passes beside the sphere: cycle maxima $0.873$, $0.891$, $0.928$, $0.950$ and $0.963J_c^{\text{num}}$ against the window values $0.786$, $0.791$, $0.919$, $0.950$ and $0.963$ (visible in Fig.~\ref{fig:test12q2_3D}). If $J_{x,c}^{\text{eff}}$ in 3D is defined as the cycle maximum, the 3D curve in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}a) is nearly flat ($0.87$ to $0.96$) and the figure, the numbers in this paragraph and in the letter must be regenerated (076/paper\_figures\_3D.py, evaluation\_rev2.py Jeff()). The qualitative conclusions (no toughening above $J_c^{\text{num}}$ in 3D, much smaller loss for thin walls than in 2D) are unchanged.}" "\n",
    "",
)

# ---------------------------------------------------------------- Section 5.2: comparison paragraph
ms.rep(
    r"In three dimensions, the effective crack resistance stays below the homogeneous value for all wall widths and saturates at about $0.96J_c^{\text{num}}$ for $w_s = 2L$. For thin walls, on the other hand, the three-dimensional specimen loses much less of its crack resistance than the two-dimensional one, $0.79$ against $0.50J_c^{\text{num}}$ at $w_s = 0.375L$. Both differences follow from the crack-front evolution described above.",
    r"In three dimensions, the effective crack resistance stays below the homogeneous value for all wall widths, $0.87$ to $0.96J_c^{\text{num}}$, and saturates at about $0.96J_c^{\text{num}}$ for $w_s = 2L$. For thin walls, on the other hand, the three-dimensional specimen loses much less of its crack resistance than the two-dimensional one, $0.87$ against $0.50J_c^{\text{num}}$ at $w_s = 0.375L$. The two curves cross at $w_s \approx 0.75L$ ($0.93$ against $0.91J_c^{\text{num}}$), the three-dimensional values lie above the two-dimensional ones for thinner and below them for wider walls. Both differences follow from the crack-front evolution described above.",
)

# ---------------------------------------------------------------- Section 5.2: energy partition, G_diss/J_eff
ms.rep(
    r"The energy dissipated per unit crack advance, $G_{\text{diss}} = 1.11$ to $1.44J_c^0$, agrees with $J_{x,c}^{\text{eff}}$ within $5\,\%$ for all wall widths, whereas $G_{\text{diss}}/J_{x,c}^{\text{eff}} = 0.73$ to $0.93$ in two dimensions, where $J_{x,c}^{\text{eff}}$ is a sharp peak and $G_{\text{diss}}$ the average over the cycle. In three dimensions, the driving force required for crack advance is thus nearly constant along the crack path, and the effective crack resistance coincides with the energy dissipated per unit crack advance.",
    r"The energy dissipated per unit crack advance, $G_{\text{diss}} = 1.11$ to $1.44J_c^0$, is close to $J_{x,c}^{\text{eff}}$, $G_{\text{diss}}/J_{x,c}^{\text{eff}} = 0.89$ to $1.04$ for $w_s = 0.375L$ to $2L$, whereas the ratio is $0.73$ to $0.93$ in two dimensions, where $J_{x,c}^{\text{eff}}$ is a sharp peak and $G_{\text{diss}}$ the average over the cycle. In three dimensions, the driving force required for crack advance thus varies only little along the crack path, and the effective crack resistance nearly coincides with the energy dissipated per unit crack advance.",
)

# ---------------------------------------------------------------- Working Note and conclusion TODO
ms.rep(
    r"rendering of a fixed-mesh 3D run for Fig.~\ref{fig:mesh3D}, decision on the $J_{x,c}^{\text{eff}}$ window for thin walls in 3D (red note in Section~\ref{sec:3D_fracture_behavior}).}",
    r"rendering of a fixed-mesh 3D run for Fig.~\ref{fig:mesh3D}. Part 7b (edits\_round2\_part7b\_Jeff\_cycle\_max.py): $J_{x,c}^{\text{eff}}$ = maximum of $J_x$ over the pore-to-pore cycle (definition sentence in Section~4.2, 3D values 0.87/0.89/0.93/0.95/0.96, old-mesh cubic pair at $w_s = 0.375L$ removed).}",
)
ms.rep(
    r"in 3D the pores never raise $J_{x,c}^{\text{eff}}$ above $J_c^{\text{num}}$ ($0.79$ to $0.96$, saturation at $0.96$ against $1.34$ in 2D)",
    r"in 3D the pores never raise $J_{x,c}^{\text{eff}}$ above $J_c^{\text{num}}$ ($0.87$ to $0.96$, nearly flat, saturation at $0.96$ against $1.34$ in 2D, 2D and 3D curves cross at $w_s \approx 0.75L$)",
)
ms.save()

# ============================================================================ letter
# ---------------------------------------------------------------- R1 major 4
lt.rep(
    r"Two earlier three-dimensional runs on a coarser mesh with the full cubic stiffness in the embedding changed $J_{x,c}^{\text{eff}}$ by $-0.37\,\%$ ($w_s = 0.375L$) and $-0.22\,\%$ ($w_s = L$) relative to their isotropic twins, which is within the numerical scatter of those runs.",
    r"An earlier three-dimensional run for $w_s = L$ on a coarser mesh with the full cubic stiffness in the embedding changed $J_{x,c}^{\text{eff}}$ by $-0.22\,\%$ relative to its isotropic twin, which is within the numerical scatter of those runs.",
)

# ---------------------------------------------------------------- Additional corrections, item 4
lt.rep(
    r"normalisation of items~9 to~11. The effective crack resistances are now $J_{x,c}^{\text{eff}} = 1.11$, $1.12$, $1.30$, $1.34$" "\n"
    r"and $1.39\,J_c^0$, i.e., $0.79$, $0.79$, $0.92$, $0.95$ and $0.96$ in units of the numerical fracture resistance of the",
    r"normalisation of items~9 to~11. The effective crack resistances are now $J_{x,c}^{\text{eff}} = 1.24$, $1.26$, $1.31$, $1.34$" "\n"
    r"and $1.39\,J_c^0$, i.e., $0.87$, $0.89$, $0.93$, $0.95$ and $0.96$ in units of the numerical fracture resistance of the",
)
lt.rep(
    r"walls the three-dimensional specimen loses much less than the two-dimensional one ($0.79$ against $0.50$ at $w_s = 0.375L$),",
    r"walls the three-dimensional specimen loses much less than the two-dimensional one ($0.87$ against $0.50$ at $w_s = 0.375L$),",
)

# ---------------------------------------------------------------- Additional corrections, item 11: evaluation as cycle maximum
lt.rep(
    r"the run for $w_s = L$ on the mesh of the previous version gave $0.93$ against $0.95$ on the fixed mesh, although the" "\n"
    r"absolute values differed by $14\,\%$.",
    r"the run for $w_s = L$ on the mesh of the previous version gave $0.93$ against $0.95$ on the fixed mesh, although the" "\n"
    r"absolute values differed by $14\,\%$. The effective crack resistance is now evaluated as the maximum of $J_x$ over the" "\n"
    r"pore-to-pore cycle that contains the evaluation ligament (stated in Section~4.2). In two dimensions this coincides with" "\n"
    r"the peak immediately before re-nucleation used in the previous versions (all two-dimensional values are unchanged), in" "\n"
    r"three dimensions it captures the peak that is attained while the crack front passes beside the spherical pore, which" "\n"
    r"a window on the ligament misses for thin walls.",
)
lt.rep(
    r"%@quote cmd:The pattern between pores one and three" "\n",
    r"%@quote cmd:The pattern between pores one and three" "\n" r"%@quote cmd:Throughout this work, $J_{x,c}^{\text{eff}}$ is evaluated as the maximum" "\n",
)
lt.save()
print("part 7b applied")
