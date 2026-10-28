#!/usr/bin/env python3
"""Round-2 edit script, part 11 (2026-10-28): Alex's 24 PDF annotations (A1-A23), the final w_s = 2L /
S5 values (B1-B5) and the corrections of the consistency review (C1-C17).

Sources: scratchpad brief_part11.md + annots_1028.txt of the session of 2026-10-28; every number verified
against 075_klaim_rev2_2D/evaluation_output/{master_table,Jeff_summary_2D,energy_cycles,cubic_vs_isotropic,
Jc_num_solid_3D}.csv (exception: the crack-front statistics of B5, which come from the run tables and were
taken from the brief).

Notes
- A23 uses pics/s_t_2.9998_wsteg_0.25.png (3D crack pattern of the previous version) instead of the
  pics/holes_s_middle_row.png named in the brief: the latter is the 2D middle-row picture and is already part
  of Figs. 10 and 17; Alex's annotation asks for the 3D crack pattern "from the previous version for w_s = 0.25L".
  The caption states neither w_s nor t nor the series (illustration only).
- A11 removes the redundant J_c^num([EqJc]) equation; all later equation numbers shift down by one, the
  hard-coded equation numbers in the letter's own prose are adjusted here, the generated quotes get the new
  numbers from the fresh .aux (build_response_quotes.py must run AFTER pdflatex of the template).
- The letter %@quote anchors that my manuscript edits invalidate are updated/removed here as well.
One-shot: fails loudly on the first anchor that is gone (already applied).
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

# ============================================================================ manuscript: preamble + working note
ms.rep(
    "% placeholder: partition numbers and Fig. 19b of the w_s = 2L run still running on 2026-10-28; stays visible in the clean build",
    "% no placeholders left after part 11 (all w_s = 2L numbers final); macro kept in case new data arrive",
)
ms.rep(
    "% provisional value of the running w_s = 2L run (red, no brackets), check when the run is finished",
    "% provisional (red, no brackets): normalised 3D values, which await the homogeneous references with elastic side layers (strip runs submitted 2026-10-28)",
)
ms.rep(
    "placeholders remain for the energy partition of $2L$ (Section~5.2, Fig.~\\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}b) and for the incomplete $2L$ curve of Fig.~\\ref{fig:test12q2_3D}.}",
    "placeholders remain for the energy partition of $2L$ (Section~5.2, Fig.~\\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}b) and for the incomplete $2L$ curve of Fig.~\\ref{fig:test12q2_3D}. "
    "Part 11 (edits\\_round2\\_part11\\_comments\\_review.py, 2026-10-28): Alex's 24 PDF annotations, the final $w_s = 2L$ values (partition, $G_{\\text{diss}}$, crack-front statistics up to the end of the evaluation cycle) "
    "and the corrections of the consistency review; no \\textbackslash pending left; \\textbackslash prelim now marks the normalised three-dimensional values, which await the homogeneous references with elastic side layers "
    "(expected shift a few percent upward; caveat in Section~\\ref{sec:3D_fracture_behavior}).}",
)

# ============================================================================ A. Alex's annotations
# ---------------------------------------------------------------- A1: Nemat-Nasser sentence out (intro RVE paragraph, p4)
ms.rep(
    "treated as a statistically uniform continuum. The effective properties of such a sample are independent of the boundary conditions applied to it, see also Nemat-Nasser and Hori~\\cite{nemat-nasser_micromechanics_2013}. Fracture does not fit",
    "treated as a statistically uniform continuum. Fracture does not fit",
)

# ---------------------------------------------------------------- A2: square (2D) / cubic (3D)
ms.rep(
    "this stiffness has square or cubic symmetry",
    "this stiffness has square (two dimensions) or cubic (three dimensions) symmetry",
)

# ---------------------------------------------------------------- A3 + C1c: corners paragraph kept, scope corrected
ms.rep(
    "In the simulations of Sections~\\ref{sec:2Dresults} and~\\ref{sec:3D_results}, the corners are located at least one cell width $w_c$ away from the crack plane and the embedding remains linear elastic, so that the J-integral evaluated on the outer boundary is not affected by the local stress concentration at the corners.",
    "In the two-dimensional simulations of Section~\\ref{sec:2Dresults}, the corners are located at least one cell width $w_c$ away from the crack plane; in the three-dimensional specimens of Section~\\ref{sec:3D_results}, thin layers of the embedding also lie on the crack plane, see Section~\\ref{sec:3D_effective_medium}. The embedding remains linear elastic, so that the J-integral evaluated on the outer boundary is not affected by the local stress concentration at the corners.",
)

# ---------------------------------------------------------------- A5: solver (MUMPS LU) + time-step thresholds (new text only)
ms.rep(
    "The nonlinear residual and its derivative are assembled by FEniCSx, and the resulting system is solved with the PETSc-based Newton solver provided by FEniCSx. "
    "The linear subproblems arising within Newton's method are solved with the default PETSc Krylov solver GMRES and the default PETSc ILU preconditioner used by this setup. "
    "The time step is adapted automatically during the simulation. The maximum number of Newton iterations per time step is set to eight. "
    "If a step does not converge within this limit, the time step is reduced by a factor of two and the step is repeated. "
    "If convergence is reached in fewer than four Newton iterations, the next time step is increased by a factor of two, subject to the prescribed upper bound for the respective simulation.",
    "The nonlinear residual and its derivative are assembled by FEniCSx, and the resulting system is solved with the PETSc-based Newton solver provided by FEniCSx. "
    "The linear subproblems arising within Newton's method are solved with a direct sparse LU factorization (MUMPS) through PETSc. "
    "The time step is adapted automatically during the simulation. The maximum number of Newton iterations per time step is set to eight. "
    "If a step does not converge within this limit, the time step is reduced by a factor of two and the step is repeated. "
    "If convergence is reached in fewer than a prescribed number of iterations (four in the two-dimensional simulations, five for $w_s = 4L$, and five to seven in the three-dimensional simulations), the next time step is doubled, subject to the prescribed upper bound for the respective simulation.",
)

# ---------------------------------------------------------------- A6 + A9: ", from an earlier simulation" out (Figs. 2 and 4)
ms.rep(
    ", from an earlier simulation in a homogeneous strip of size $16L \\times 2L$ with the same surfing boundary conditions",
    " in a homogeneous strip of size $16L \\times 2L$ with the same surfing boundary conditions",
    n=2,
)

# ---------------------------------------------------------------- A7: 3D reference values in Section 3.5
ms.rep(
    "whose energies scale with $J_c$, see Section~\\ref{sec:2D_fracture_behavior}.}\n",
    "whose energies scale with $J_c$, see Section~\\ref{sec:2D_fracture_behavior}.}\n"
    "\\added{The corresponding three-dimensional homogeneous reference simulations, carried out for the parameter set \\vMLabel only, yield $J_c^{\\text{num}} = 1.42J_c^0$, and $1.445J_c^0$ for the longer specimen used for $w_s = 2L$, see Section~\\ref{sec:3D_fracture_behavior}.}\n",
)

# ---------------------------------------------------------------- A8: Fig. 4 alpha legend between plot and caption
ms.rep(
    "    \\resizebox{0.85\\textwidth}{!}{\\includegraphics{pics/alpha_t_6.2495.png}}\n    \\vspace{0.2cm}",
    "    \\resizebox{0.85\\textwidth}{!}{\\includegraphics{pics/alpha_t_6.2495.png}}\\\\[4pt]\n"
    "    {\\small $\\alpha$}~\\resizebox{0.22\\textwidth}{!}{\\includegraphics{pics/legend_alpha.png}}\n"
    "    \\vspace{0.2cm}",
)
ms.rep(
    "    \\label{fig:plasticity_vm_jeff}\n\n    \\begin{picture}(0,0)\n        \\put(-50,25){%\n            \\resizebox{0.25\\textwidth}{!}{\\includegraphics{pics/legend_alpha.png}}\n        }\n        \\put(-5,55){$\\alpha$}\n    \\end{picture}\n\\end{figure}",
    "    \\label{fig:plasticity_vm_jeff}\n\\end{figure}",
)

# ---------------------------------------------------------------- A10: J_bc = 0.9375 J_c^0 clause out
ms.rep(
    "$K_I^{\\text{bc}} = \\sqrt{J_c^0 \\cdot 2.5\\mu_{0}}$\\added{, i.e., the energy release rate of the applied K-field is $J_{\\text{bc}} = 0.9375J_c^0$}, which enforce",
    "$K_I^{\\text{bc}} = \\sqrt{J_c^0 \\cdot 2.5\\mu_{0}}$, which enforce",
)

# ---------------------------------------------------------------- A11: redundant second J_c^num([EqJc]) statement out
ms.rep(
    "\\replaced{In Section~\\ref{sec:2Dresults}, the effective crack resistance of each parameter set is compared with the numerical fracture resistance $J_c^{\\text{num}}$ of the same set. For the linear elastic model with $J_c = J_c^0$, i.e., for the parameter set \\EqJcLabel, it is}{A reference numerical energetic fracture resistance is defined as the value obtained from parameter set \\EqJcLabel:}\n"
    "\\begin{equation}\n    J_c^{\\text{num}} \\approx \\changed{1.088J_c^0}.\n\\end{equation}\n",
    "\\replaced{In Section~\\ref{sec:2Dresults}, the effective crack resistance of each parameter set is compared with the numerical fracture resistance $J_c^{\\text{num}}$ of the same set.}{A reference numerical energetic fracture resistance is defined as the value obtained from parameter set \\EqJcLabel: $J_c^{\\text{num}} \\approx 1.12J_c^0$.}\n",
)

# ---------------------------------------------------------------- A12 + A13(b) + annotation 17: "an", usage statement
ms.rep(
    "The influence of the isotropic approximation on the effective crack resistance is tested in Section~\\ref{sec:2D_cubic_embedding}.}",
    "The influence of an isotropic approximation of the embedding on the effective crack resistance is tested in Section~\\ref{sec:2D_cubic_embedding}; unless stated otherwise, all simulations in this work use the isotropic approximation, and the full square-symmetric stiffness enters only these tests.}",
)

# ---------------------------------------------------------------- A13(a): cell has square symmetry, isotropy is the approximation
ms.rep(
    "The isotropic approximation of eq.~(\\ref{eq:isotropic_approximation_2D}) uses $C_{12}$ and $C_{33}$ and replaces $C_{11}$ by $C_{12} + 2C_{33}$.",
    "The isotropic approximation of eq.~(\\ref{eq:isotropic_approximation_2D}) uses $C_{12}$ and $C_{33}$ and replaces $C_{11}$ by $C_{12} + 2C_{33}$; the unit cell itself is therefore not isotropic, the isotropy is introduced as an approximation of its effective stiffness.",
)

# ---------------------------------------------------------------- A14: nu_0^eff column in Table 1 + text reference
ms.rep(
    "\\label{tab:effective_stiffness_2D}\n\\begin{tabular}{lccccccc}",
    "\\label{tab:effective_stiffness_2D}\n\\begin{tabular}{lcccccccc}",
)
ms.rep(
    "$w_s/L$ & $C_{11}$ & $C_{12}$ & $C_{33}$ & $\\lambda^{\\text{eff}}$ & $\\mu^{\\text{eff}}$ & $E_0^{\\text{eff}}$ & $A$ \\\\\n"
    "\\midrule\n"
    "0.25  & 0.869 & 0.177 & 0.301 & 0.177 & 0.301 & 0.713 & 0.870 \\\\\n"
    "0.375 & 1.087 & 0.241 & 0.376 & 0.241 & 0.376 & 0.900 & 0.890 \\\\\n"
    "1.0   & 1.814 & 0.520 & 0.621 & 0.520 & 0.621 & 1.524 & 0.960 \\\\\n"
    "2.0   & 2.361 & 0.755 & 0.794 & 0.755 & 0.794 & 1.975 & 0.989 \\\\\n"
    "4.0   & 2.740 & 0.907 & 0.915 & 0.907 & 0.915 & 2.285 & 0.998 \\\\",
    "$w_s/L$ & $C_{11}$ & $C_{12}$ & $C_{33}$ & $\\lambda^{\\text{eff}}$ & $\\mu^{\\text{eff}}$ & $E_0^{\\text{eff}}$ & $\\nu_0^{\\text{eff}}$ & $A$ \\\\\n"
    "\\midrule\n"
    "0.25  & 0.869 & 0.177 & 0.301 & 0.177 & 0.301 & 0.713 & 0.186 & 0.870 \\\\\n"
    "0.375 & 1.087 & 0.241 & 0.376 & 0.241 & 0.376 & 0.900 & 0.195 & 0.890 \\\\\n"
    "1.0   & 1.814 & 0.520 & 0.621 & 0.520 & 0.621 & 1.524 & 0.228 & 0.960 \\\\\n"
    "2.0   & 2.361 & 0.755 & 0.794 & 0.755 & 0.794 & 1.975 & 0.244 & 0.989 \\\\\n"
    "4.0   & 2.740 & 0.907 & 0.915 & 0.907 & 0.915 & 2.285 & 0.249 & 0.998 \\\\",
)
ms.rep(
    "and the effective Poisson's ratio of the isotropic approximation increases from $\\nu_0^{\\text{eff}} = 0.19$ to $0.25$.",
    "and the effective Poisson's ratio of the isotropic approximation increases from $\\nu_0^{\\text{eff}} = 0.19$ to $0.25$, Table~\\ref{tab:effective_stiffness_2D}.",
)

# ---------------------------------------------------------------- A15: name the embedding stiffness precisely
ms.rep(
    "The $6 \\times 3$ porous domain is embedded into a rectangular domain of effective stiffness (blue) with dimensions $(8 w_c) \\times 20 L$.",
    "The $6 \\times 3$ porous domain is embedded into \\replaced{a rectangular domain (blue) that is assigned the isotropic approximation of the effective stiffness, i.e., $E_0^{\\text{eff}}$ and $\\nu_0^{\\text{eff}}$,}{a rectangular domain of effective stiffness (blue)} with dimensions $(8 w_c) \\times 20 L$.",
)

# ---------------------------------------------------------------- A16: avoid "decouples"
ms.rep(
    "No linear elastic phase-field model has both the strength and the $J_c^0$ of \\vMLabel, since plasticity decouples the two.",
    "No linear elastic phase-field model has both the strength and the $J_c^0$ of \\vMLabel: in the linear elastic model the strength follows from $J_c$ and $\\epsilon$, eq.~(\\ref{eq:sig_c}), whereas the plastic model introduces $\\sigma_y$ and $H$ as additional parameters, so that strength and microscopic fracture resistance can be chosen independently.",
)

# ---------------------------------------------------------------- A17 + A18: "see Sections ...", clearer mechanism sentence
ms.rep(
    "The reason, Sections~\\ref{sec:2D_energy_partition} and~\\ref{sec:2D_mechanisms}, is that crack trapping and forced re-nucleation at the blunt pore wall toughens a linear elastic material efficiently, at the price of $47$ to $83\\,\\%$ of the dissipated energy going into the unstable snap-through of the ligament, whereas the homogeneous plastic material already blunts its crack by the plastic zone, which is contained in $J_c^{\\text{num}}$, and in thin ligaments the whole ligament yields, so that the plastic work is not available as crack resistance.}",
    "The reason is twofold, see Sections~\\ref{sec:2D_energy_partition} and~\\ref{sec:2D_mechanisms}. Trapping and forced re-nucleation at the blunt pore wall raise the peak driving force of a linear elastic material far above its $J_c^{\\text{num}}$, at the price of $47$ to $83\\,\\%$ of the dissipated energy going into the unstable snap-through of the ligament. The homogeneous plastic material, in contrast, already blunts its crack by the plastic zone, and this contribution is part of $J_c^{\\text{num}}$ itself, so re-nucleation at the pore wall raises the peak driving force above $J_c^{\\text{num}}$ by a smaller factor. For thin walls, yielding of the ligament keeps the peak driving force low, and it even falls below $J_c^{\\text{num}}$, Section~\\ref{sec:2D_mechanisms}.}",
)

# ---------------------------------------------------------------- A19: physics vs. numerics of the G_diss reduction (hedged)
ms.rep(
    "which is the quantity closest to an effective fracture energy in the sense of the energy dissipated along the crack path, see Michel and Suquet~\\cite{michel_merits_2022}. The mechanism is identified in the next section.}",
    "which is the quantity closest to an effective fracture energy in the sense of the energy dissipated along the crack path, see Michel and Suquet~\\cite{michel_merits_2022}. "
    "Part of this reduction must be attributed to the quasi-static model rather than to the physics. The energy released in the unstable snap-through of a ligament is absorbed by the viscous term of eq.~(\\ref{eq:evolution_s}), i.e., by the viscous shares quoted above ($12$ to $37\\,\\%$ for \\vMLabel, $47$ to $83\\,\\%$ for the linear elastic sets). "
    "In a dynamic computation, this energy would instead become kinetic energy and could drive further crack advance or additional plastic dissipation, so the reduction of $G_{\\text{diss}}$ by plasticity for thin walls is in part a property of the viscous regularization. "
    "The underlying mechanism, namely that the lower peak stress of the yielded ligament suppresses re-nucleation ahead of the pore and with it the snap-through, is physical, see Section~\\ref{sec:2D_mechanisms}. "
    "Resolving how the released energy would partition in reality requires dynamic simulations, which are beyond the scope of this work. The mechanism behind the crossover is identified in the next section.}",
)

# ---------------------------------------------------------------- A20 + C11: comparison added, mechanism corrected
ms.rep(
    "Once the ligament has yielded, its tangent stiffness drops to the hardening modulus $H$, so that its opening grows with the advancing K-field at nearly constant stress, and $J_x$ at re-nucleation stays low. The plastic work done on the ligament before re-nucleation is supplied by the boundary conditions, but it does not enter $J_x$, which measures the energy release rate of the elastic surroundings, and it is not available as crack resistance.",
    "Once the ligament has yielded, its tangent stiffness drops to the hardening modulus $H$, so that its opening grows with the advancing K-field at nearly constant stress, and $J_x$ at re-nucleation stays low compared with the linear elastic sets: their ligaments are loaded elastically up to the fracture stress, store correspondingly more energy, and release $47$ to $83\\,\\%$ of the dissipated energy per cycle in the unstable snap-through, against $12$ to $37\\,\\%$ for \\vMLabel, Section~\\ref{sec:2D_energy_partition}. Yielding lowers the peak stress ahead of the pore; it does not remove the plastic work from the energy balance, eq.~(\\ref{eq:energy_balance}).",
)

# ---------------------------------------------------------------- A21: regularization length in the length-scale list
ms.rep(
    "and the interaction length between neighboring pores is set by the size of the plastic zone rather than by the pore diameter, consistent with",
    "and the interaction length between neighboring pores is set by the size of the plastic zone rather than by the pore diameter or the regularization length $\\epsilon = 0.1L$, consistent with",
)

# ---------------------------------------------------------------- A22: lead-in sentences (4.3, 4.5, 5.2; 3.5/4.4/5.1 already have one)
ms.rep(
    "\\added{The model of Section~\\ref{sec:phase_fiel_model_ductile} obeys an exact energy balance,",
    "\\added{This section quantifies where the energy supplied in the numerical experiment goes, in order to identify the dissipation channels that distinguish the ductile from the linear elastic sets. "
    "The model of Section~\\ref{sec:phase_fiel_model_ductile} obeys an exact energy balance,",
)
ms.rep(
    "\\added{The embedding of the resolved porous region is assigned the isotropic approximation, eq.~(\\ref{eq:isotropic_approximation_2D}),",
    "\\added{This section tests how robust the effective crack resistance is with respect to modelling choices that are not part of the microstructure: the stiffness assigned to the embedding, the length of the initial crack and the size of the resolved region. "
    "The embedding of the resolved porous region is assigned the isotropic approximation, eq.~(\\ref{eq:isotropic_approximation_2D}),",
)
ms.rep(
    "\\replaced{In all simulations, the crack propagates through the central row of pores. Figure~\\ref{fig:position_3D} shows",
    "\\replaced{This section describes the crack patterns of the three-dimensional series, compares the effective crack resistance with the two-dimensional results and closes with the energy partition. "
    "In all simulations, the crack propagates through the central row of pores, Fig.~\\ref{fig:crack3D_illustration}. Figure~\\ref{fig:position_3D} shows",
)

# ---------------------------------------------------------------- A23: illustration figure in Section 5
ms.rep(
    "    \\label{fig:position_3D}\n\\end{figure}\n",
    "    \\label{fig:position_3D}\n\\end{figure}\n"
    "\n"
    "\\begin{figure}[htb]\n"
    "    \\centering\n"
    "    \\resizebox{0.45\\textwidth}{!}{\\includegraphics{pics/s_t_2.9998_wsteg_0.25.png}}\n"
    "    \\caption{\\added{Crack propagation through the central row of pores in the three-dimensional model (phase field $s$, illustration; red: intact material, blue: crack). The half of the specimen with $y > 0$ is indicated by its transparent surface mesh.}}\n"
    "    \\label{fig:crack3D_illustration}\n"
    "\\end{figure}\n",
)

# ============================================================================ B. final values
# ---------------------------------------------------------------- B1: Fig. 18 caption
ms.rep(
    " \\pending{$w_s = 2L$ shown up to $x_{ct} = 13L$, to be completed}}{Section of the J-Integral",
    "}{Section of the J-Integral",
)

# ---------------------------------------------------------------- B2: Fig. 19 caption
ms.rep(
    " \\pending{$w_s = 2L$ to be added in b)}}{Effective crack resistance",
    "}{Effective crack resistance",
)

# ---------------------------------------------------------------- B3: partition sentence complete (91-94 / 6-9 / <=1 %)
ms.rep(
    "In three dimensions, $91$ to $93\\,\\%$ of the dissipated energy create crack surface, $6$ to $9\\,\\%$ remain stored as plastic energy and at most $1\\,\\%$ is dissipated by the viscous evolution of the phase field, for $w_s = 0.375L$ to $L$ \\pending{$2L$ to be added}, compared with",
    "In three dimensions, $91$ to $94\\,\\%$ of the dissipated energy create crack surface, $6$ to $9\\,\\%$ remain stored as plastic energy and at most $1\\,\\%$ is dissipated by the viscous evolution of the phase field, for $w_s = 0.375L$ to $2L$, compared with",
)

# ---------------------------------------------------------------- B4: G_diss complete and weakened
ms.rep(
    "The energy dissipated per unit crack advance, $G_{\\text{diss}} = 1.17$ to $1.31J_c^0$, is close to $J_{x,c}^{\\text{eff}}$, $G_{\\text{diss}}/J_{x,c}^{\\text{eff}} = 0.95$ to $0.97$ for $w_s = 0.375L$ to $L$ \\pending{$2L$ to be added}, whereas the ratio is $0.73$ to $0.93$ in two dimensions, where $J_{x,c}^{\\text{eff}}$ is a sharp peak and $G_{\\text{diss}}$ the average over the cycle. In three dimensions, the driving force required for crack advance thus varies only little along the crack path, and the effective crack resistance nearly coincides with the energy dissipated per unit crack advance.}",
    "The energy dissipated per unit crack advance, evaluated for the same cycle as $J_{x,c}^{\\text{eff}}$, is $G_{\\text{diss}} = 1.17$ to $1.44J_c^0$, i.e., $G_{\\text{diss}}/J_{x,c}^{\\text{eff}} = 0.95$ to $1.03$ for $w_s = 0.375L$ to $2L$, whereas the ratio is $0.73$ to $0.93$ in two dimensions, where $J_{x,c}^{\\text{eff}}$ is a sharp peak and $G_{\\text{diss}}$ the average over the cycle. Across the other pore-to-pore cycles, the ratio varies between $0.82$ and $1.05$, since the stored elastic energy decreases during each cycle and $G_{\\text{diss}}$ therefore also contains released elastic energy. In three dimensions, the effective crack resistance is thus close to the energy dissipated per unit crack advance for the evaluation cycle, not pointwise along the crack path.}",
)

# ---------------------------------------------------------------- B5 + C14: crack-front statistics final, jump wording
ms.rep(
    "The crack front advances by at most \\prelim{$0.11$} to $0.19L$ per time step, and $x_{ct}$ lies within the $x$-range of a pore in \\prelim{$17$} to $79\\,\\%$ of the time steps ($w_s = 2L$ to $0.375L$), which is only possible",
    "The crack front advances by at most $0.11$ to $0.15L$ per time step, and $x_{ct}$ lies within the $x$-range of a pore in $21$ to $75\\,\\%$ of the time steps ($w_s = 2L$ to $0.375L$, counted up to the end of the evaluation cycle), which is only possible",
)
ms.rep(
    "with small steps at the pores instead of the jumps of about $1.2L$ across pore and ligament.",
    "with small steps at the pores instead of the jumps of about $1.2L$ across the pores.",
)
ms.rep(
    "{. It is trapped at each pore until re-nucleation and then crosses the pore and the following ligament in one jump.}",
    "{. It is trapped at each pore until re-nucleation and then jumps across the pore, after which the ligament ahead fails unstably.}",
)

# ============================================================================ C. consistency corrections
# ---------------------------------------------------------------- C1a: side layers and z-face BC stated in 5.1
ms.rep(
    "Thus, the unit cell used to determine the effective stiffness is cubic, whereas the surrounding homogeneous domain is a larger cuboidal embedding region assigned this effective stiffness.",
    "Thus, the unit cell used to determine the effective stiffness is cubic, whereas the surrounding homogeneous domain is a larger cuboidal embedding region assigned this effective stiffness."
    "\\added{ The two layers of thickness $w_c/4$ between the porous block and the planes $z = z_{\\rm min}$ and $z = z_{\\rm max}$ belong to the linear elastic embedding and lie on the crack plane $y = 0$ along the entire crack path, i.e., about $20\\,\\%$ of the specimen thickness fractures within embedding material with the same $J_c$.}",
)
ms.rep(
    "including the planes $z=z_{\\rm min}$ and $z=z_{\\rm max}$, which mimics}",
    "including the planes $z=z_{\\rm min}$ and $z=z_{\\rm max}$, for the porous specimens and the homogeneous reference alike, which mimics}",
)

# ---------------------------------------------------------------- C13: reference agreement scoped; C1b: caveat sentence
ms.rep(
    "The four-pore specimen is normalized by its own references, $1.415J_c^0$ ($w_s \\le L$), $1.444J_c^0$ ($w_s = 2L$) and $1.351J_c^0$ ($h = 0.07L$), which differ from the above by less than $0.5\\,\\%$.}",
    "The four-pore specimen is normalized by its own references, $1.415J_c^0$ ($w_s \\le L$) and $1.444J_c^0$ ($w_s = 2L$), which differ from the above by less than $0.5\\,\\%$, and by $1.351J_c^0$ for the finer mesh, $h = 0.07L$.}",
)
ms.rep(
    "$J_x^{\\text{num,pl}}\\approx1.50J_c^0$; see Section~\\ref{sec:steady_growth_griffith}.}",
    "$J_x^{\\text{num,pl}}\\approx1.50J_c^0$; see Section~\\ref{sec:steady_growth_griffith}.}\n"
    "\\prelim{The values normalized by $J_c^{\\text{num}}$ in this section are provisional: the homogeneous reference is plastic through the whole thickness, whereas in the porous specimens about $20\\,\\%$ of the thickness fractures within the elastic side layers of Section~\\ref{sec:3D_effective_medium}. A reference with identical side layers is being computed; the normalized values are expected to shift upward by a few percent.}",
)

# ---------------------------------------------------------------- C1b: prelim wraps of the normalised 3D values (5.2)
ms.rep(
    "with maxima of $0.98$, $0.96$, $0.95$, $0.94$ and $0.90J_c^{\\text{num}}$ over the five pore-to-pore cycles, i.e., below the crack resistance of the homogeneous material.",
    "with maxima of \\prelim{$0.98$, $0.96$, $0.95$, $0.94$ and $0.90J_c^{\\text{num}}$} over the five pore-to-pore cycles, i.e., \\prelim{below the crack resistance of the homogeneous material}.",
)
ms.rep(
    "$J_{x,c}^{\\text{eff}} = 1.22$, $1.26$, $1.31$, $1.34$ and $1.40J_c^0$, i.e., $0.86$, $0.88$, $0.92$, $0.95$ and $0.97J_c^{\\text{num}}$ for $w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and $2L$.",
    "$J_{x,c}^{\\text{eff}} = 1.22$, $1.25$, $1.31$, $1.34$ and $1.40J_c^0$, i.e., \\prelim{$0.86$, $0.88$, $0.92$, $0.95$ and $0.97J_c^{\\text{num}}$} for $w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and $2L$.",
)
ms.rep(
    "gives $0.87$, $0.89$, $0.93$, $0.95$ and $0.96J_c^{\\text{num}}$, open triangles",
    "gives \\prelim{$0.87$, $0.89$, $0.93$, $0.95$ and $0.96J_c^{\\text{num}}$}, open triangles",
)
ms.rep(
    "In three dimensions, the effective crack resistance does not exceed the homogeneous value for any of the computed wall widths, $0.86$ to $0.97J_c^{\\text{num}}$ for $w_s = 0.375L$ to $2L$, and the curve is nearly flat. For thin walls, on the other hand, the three-dimensional specimen loses much less of its crack resistance than the two-dimensional one, $0.86$ against $0.50J_c^{\\text{num}}$ at $w_s = 0.375L$. The two curves cross between $w_s = 0.75L$ ($0.92$ against $0.91J_c^{\\text{num}}$) and $w_s = L$ ($0.95$ against $1.10J_c^{\\text{num}}$),",
    "In three dimensions, \\prelim{the effective crack resistance does not exceed the homogeneous value for any of the computed wall widths, $0.86$ to $0.97J_c^{\\text{num}}$ for $w_s = 0.375L$ to $2L$}, and the increase with the wall width is weak. For thin walls, on the other hand, the three-dimensional specimen loses much less of its crack resistance than the two-dimensional one, \\prelim{$0.86$} against $0.50J_c^{\\text{num}}$ at $w_s = 0.375L$. The two curves cross \\prelim{between $w_s = 0.75L$ ($0.92$ against $0.91J_c^{\\text{num}}$) and $w_s = L$ ($0.95$ against $1.10J_c^{\\text{num}}$)},",
)

# ---------------------------------------------------------------- C1b + C7: abstract
ms.rep(
    "the crack front advances beside the pores without unstable ligament failure, and the effective crack resistance does not exceed the homogeneous value.}",
    "the crack front advances beside the pores without unstable ligament failure, and \\prelim{the effective crack resistance does not exceed the homogeneous value for wall widths up to $2L$}.}",
)

# ---------------------------------------------------------------- C1b + C15c: conclusion (prelim + measured deltas)
ms.rep(
    "The effective crack resistance, normalized by the crack resistance of the three-dimensional homogeneous material, is $0.86$ to $0.97$ for $w_s = 0.375L$ to $2L$, nearly independent of the wall width, does not exceed the homogeneous value and is robust against the mesh and the number of pores in the crack-growth direction.",
    "The effective crack resistance, normalized by the crack resistance of the three-dimensional homogeneous material, \\prelim{increases only from $0.86$ to $0.97$ for $w_s = 0.375L$ to $2L$ and does not exceed the homogeneous value}; the normalized value changes by $+0.9\\,\\%$ under a mesh refinement from $h = 0.1L$ to $0.07L$ and by at most $1.5\\,\\%$ with four instead of six pores in the crack-growth direction.",
)

# ---------------------------------------------------------------- C2: J_eff definition (cycle maximum) in Section 2
ms.rep(
    "\\begin{equation}\n    J_{x,c}^{\\text{eff}} = \\max_t J_x(t).\n\\end{equation}\nThe index $c$",
    "\\begin{equation}\n    J_{x,c}^{\\text{eff}} = \\max_t J_x(t).\n\\end{equation}\n"
    "\\added{Hossain et al.\\ take the maximum over the complete separation process. In this work, $J_{x,c}^{\\text{eff}}$ is evaluated as the maximum of $J_x$ over one pore-to-pore cycle in the interior of the specimen, see Sections~\\ref{sec:2D_fracture_behavior} and~\\ref{sec:2D_energy_partition}, which excludes the build-up of the K-field at the start and the influence of the end of the specimen.}\n"
    "The index $c$",
)

# ---------------------------------------------------------------- C3 + C10: regimes in Section 4.4
ms.rep(
    "For the von~Mises set, two regimes can be distinguished in Fig.~\\ref{fig:ligament_state}. For thin ligaments, $w_s \\le 0.6L$, the ligament yields across its full width before re-nucleation. The maximum of $\\alpha$ is $0.66$ to $0.73$, the triaxiality is $0.67$ to $0.74$, and re-nucleation occurs at",
    "For the von~Mises set, two regimes can be distinguished in Fig.~\\ref{fig:ligament_state}. For thin ligaments, $w_s \\le 0.6L$, the recorded maxima indicate that the ligament yields over its width before re-nucleation: the maximum of $\\alpha$ in the ligament reaches $0.66$ to $0.73$, the ratio $\\sigma_{h,\\max}/\\sigma_{\\text{vM},\\max}$ is $0.67$ to $0.74$, and re-nucleation occurs at",
)
ms.rep(
    "In this nucleation regime, the ductile material law reduces the effective crack resistance below that of \\EqSigmaStarLabel, whose linear elastic ligament has to be loaded to a higher driving force before it reaches $\\sigma^*$, Fig.~\\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets}.",
    "In this nucleation-controlled regime, which in terms of the effective crack resistance extends up to $w_s = 0.5L$ (the curves of \\vMLabel and \\EqSigmaStarLabel cross between $0.5L$ and $0.6L$, Section~\\ref{sec:2D_fracture_behavior}), the ductile material law reduces the effective crack resistance below that of \\EqSigmaStarLabel, whose linear elastic ligament has to be loaded to a higher driving force before it reaches $\\sigma^*$, Fig.~\\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets}.",
)
ms.rep(
    "For wide ligaments, $w_s \\ge L$, the plastic zone does not span the ligament, the maximum of $\\alpha$ drops to $0.44$ to $0.60$, and the trapped crack tip at the pore surface acts as a notch constrained by the surrounding elastic material. The triaxiality rises to $1.0$ to $1.1$, the von~Mises stress at the location of $\\sigma_{yy,\\max}$ remains at the yield stress, $0.98$ to $1.06\\,\\sigma_y$, and the normal stress reaches",
    "For wide ligaments, $w_s \\ge L$, the maximum of $\\alpha$ in the ligament drops to $0.44$ to $0.60$, which indicates that the plastic zone no longer spans the ligament, and the trapped crack tip at the pore surface acts as a notch constrained by the surrounding elastic material. The ratio $\\sigma_{h,\\max}/\\sigma_{\\text{vM},\\max}$ rises to $1.0$ to $1.1$, the recorded maximum of the von~Mises stress remains at the yield-stress level, $0.98$ to $1.06\\,\\sigma_y$, and the normal stress reaches",
)

# ---------------------------------------------------------------- C10: triaxiality = ratio of recorded maxima (4.4 text + caption)
ms.rep(
    "$\\sigma_y = \\sqrt{\\nicefrac{J_c^0\\mu_0}{L}}$, and the stress triaxiality $\\sigma_h/\\sigma_{\\text{vM}}$ at the location of $\\sigma_{yy,\\max}$.}",
    "$\\sigma_y = \\sqrt{\\nicefrac{J_c^0\\mu_0}{L}}$, and the triaxiality measure $\\sigma_{h,\\max}/\\sigma_{\\text{vM},\\max}$ formed from the recorded maxima, which are not necessarily attained at the same point.}",
)
ms.rep(
    "for all wall widths and both values of $J_c$, at a triaxiality of $0.49$ to $0.53$.",
    "for all wall widths and both values of $J_c$, at $\\sigma_{h,\\max}/\\sigma_{\\text{vM},\\max} = 0.49$ to $0.53$.",
)
ms.rep(
    "b) Stress triaxiality $\\sigma_h/\\sigma_{\\text{vM}}$ at the location of $\\sigma_{yy,\\max}$ and, dotted, the maximum of the plastic variable $\\alpha$ in the ligament for \\vMLabel.}",
    "b) Ratio $\\sigma_{h,\\max}/\\sigma_{\\text{vM},\\max}$ of the recorded maxima of the hydrostatic and the von~Mises stress in the ligament and, dotted, the maximum of the plastic variable $\\alpha$ in the ligament for \\vMLabel.}",
)

# ---------------------------------------------------------------- C4: first cycle / decreasing peaks / cause (4.5)
ms.rep(
    "In addition, the response along the crack path is periodic. The peaks of $J_x$ in successive ligaments, Fig.~\\ref{fig:PAPER_02_all_Jx_vs_xct_pf_diamond_holes}, and the energy dissipated per cycle, Section~\\ref{sec:2D_energy_partition}, are nearly constant from the second cycle on. The first cycle, during which the K-field builds up, is lower, the last cycle is affected by the end of the resolved region, and the inner cycles agree within about $7\\,\\%$ ($10\\,\\%$ for \\EqJcLabel at $w_s = 0.25L$).",
    "In addition, the response along the crack path is periodic. The energy dissipated per cycle, Section~\\ref{sec:2D_energy_partition}, is nearly constant from the second cycle on; the first cycle, during which the K-field builds up, dissipates less, the last cycle is affected by the end of the resolved region, and the inner cycles agree within about $7\\,\\%$ ($10\\,\\%$ for \\EqJcLabel at $w_s = 0.25L$). The peaks of $J_x$ in successive ligaments, Fig.~\\ref{fig:PAPER_02_all_Jx_vs_xct_pf_diamond_holes}, decrease slightly from cycle to cycle, e.g.\\ $1.59$, $1.58$, $1.56$ and $1.53J_c^0$ for \\vMLabel at $w_s = L$. The pore-free reference specimens show the same drift of $J_x$ along the crack path, by $1$ to $3\\,\\%$ over the same stretch, in two and in three dimensions, so this decrease is a property of the finite-domain surfing setup rather than of the pores; the averaging window of Section~\\ref{sec:steady_growth_griffith} covers it.",
)

# ---------------------------------------------------------------- C5: introduction promise reworded
ms.rep(
    "Both are used here to separate the contributions of crack nucleation, plastic dissipation and crack growth, and the sensitivity",
    "Both are used here to partition the energy dissipated per pore-to-pore cycle into crack surface energy, stored plastic energy and viscous dissipation and to characterize the stress state at re-nucleation, and the sensitivity",
)

# ---------------------------------------------------------------- C6: conclusion 31 % -> verified comparison
ms.rep(
    "which raises the crack resistance of the homogeneous material by $31\\,\\%$ at equal $J_c^0$, and the microstructure preserves only part of it.",
    "which raises the crack resistance of the homogeneous material by a factor of $2.36$ at equal strength ($31\\,\\%$ at equal $J_c^0$), and the microstructure preserves only part of it.",
)

# ---------------------------------------------------------------- C8: -0.22 % claim deleted, thinnest wall of the test series
ms.rep(
    "The effective crack resistance changes by at most $+0.18\\,\\%$, Table~\\ref{tab:robustness_2D}, even for the thinnest wall, for which the anisotropy is largest, since the embedding only transmits the K-field and the J-integral is evaluated on its outer boundary, whereas the re-nucleation that determines $J_{x,c}^{\\text{eff}}$ takes place in the resolved ligaments. The three-dimensional unit cells of Section~\\ref{sec:3D_effective_medium} are closer to isotropic, $A \\ge 0.978$ for $w_s \\ge 0.375L$, and an earlier three-dimensional run on a coarser mesh with the full cubic stiffness changed $J_{x,c}^{\\text{eff}}$ by $-0.22\\,\\%$ ($w_s = L$).",
    "The effective crack resistance changes by at most $+0.18\\,\\%$, Table~\\ref{tab:robustness_2D}, even for the thinnest wall of the test series, for which the anisotropy is largest, since the embedding only transmits the K-field and the J-integral is evaluated on its outer boundary, whereas the re-nucleation that determines $J_{x,c}^{\\text{eff}}$ takes place in the resolved ligaments. The three-dimensional unit cells of Section~\\ref{sec:3D_effective_medium} are closer to isotropic, $A \\ge 0.978$ for $w_s \\ge 0.375L$, so no separate three-dimensional test was performed.",
)

# ---------------------------------------------------------------- C9: initial-crack robustness (Table 3 rows + prose)
ms.rep(
    "initial crack $1.5w_c$     & $w_s = L$                   & $+0.02\\,\\%$ \\\\\n"
    "initial crack $2w_c$       & $w_s = L$                   & $+0.08\\,\\%$ \\\\\n",
    "initial crack $1.5w_c$     & $w_s = L$                   & $+0.02\\,\\%$ \\\\\n"
    "                           & $w_s = 0.375L$              & $+0.04\\,\\%$ \\\\\n"
    "initial crack $2w_c$       & $w_s = L$                   & $+0.08\\,\\%$ \\\\\n"
    "                           & $w_s = 0.375L$              & $-0.06\\,\\%$ \\\\\n",
)
ms.rep(
    "Increasing the length of the initial crack from $w_c$ to $1.5w_c$ and $2w_c$ changes $J_{x,c}^{\\text{eff}}$ by $+0.02\\,\\%$ and $+0.08\\,\\%$, and enlarging the resolved region from three to five rows of pores changes it by at most $1.1\\,\\%$.",
    "Increasing the length of the initial crack from $w_c$ to $1.5w_c$ and $2w_c$, which moves its tip from the edge of the first cell to the center of the first pore and to the middle of the first ligament, changes $J_{x,c}^{\\text{eff}}$ by at most $0.1\\,\\%$, both at $w_s = L$ and at $w_s = 0.375L$, i.e., in the propagation and in the nucleation regime, and enlarging the resolved region from three to five rows of pores changes it by at most $1.1\\,\\%$.",
)
ms.rep(
    "Not tested were the influence of other crack-growth directions relative to the pore lattice, of the position of the initial crack relative to the pores, and of the microstructure itself,",
    "Not tested were the influence of other crack-growth directions relative to the pore lattice and of the microstructure itself,",
)

# ---------------------------------------------------------------- C16i: Table 3 caption (last row is [Eq sigma*]; crack position)
ms.rep(
    "\\caption{\\added{Change of the effective crack resistance of the parameter set \\vMLabel by the full square-symmetric stiffness of the embedding (Zener ratio $A$ of Table~\\ref{tab:effective_stiffness_2D}), by a longer initial crack and by five instead of three resolved pore rows.}}",
    "\\caption{\\added{Change of the effective crack resistance of the parameter set \\vMLabel (last row: \\EqSigmaStarLabel) by the full square-symmetric stiffness of the embedding (Zener ratio $A$ of Table~\\ref{tab:effective_stiffness_2D}), by a longer initial crack, whose tip is thereby moved relative to the pores, and by five instead of three resolved pore rows.}}",
)

# ---------------------------------------------------------------- C12: partition built from the recorded sinks (4.3)
ms.rep(
    "All terms are recorded in every time step.}",
    "All terms are recorded in every time step. The partition below is built from the recorded sinks $E_{\\text{fr}}$, $E_{\\text{pl}}$, $D_{\\text{visc}}$ and the stored elastic energy; the recorded external work of the two-dimensional runs is not used. The balance of eq.~(\\ref{eq:energy_balance}) against the external work is verified for the three-dimensional simulations, Section~\\ref{sec:3D_fracture_behavior}.}",
)

# ---------------------------------------------------------------- C15: softening pass
ms.rep(
    "\\added{The linear elastic sets reach $95\\,\\%$ of their value at $w_s = 4L$ for $w_s \\approx 1.4L$, the von~Mises set for $w_s \\approx 2L$.}",
    "\\added{The linear elastic sets reach $95\\,\\%$ of their value at $w_s = 4L$ for $w_s \\approx 1.5L$, the von~Mises set between $w_s = 1.5L$ and $2L$.}",
)
ms.rep(
    "it is governed by the re-nucleation at the constrained notch and is much less sensitive to both.",
    "it is governed by the re-nucleation at the constrained notch and is less sensitive to both.",
)
ms.rep(
    "For the linear elastic sets, the corresponding sensitivity is the scaling with $J_c$ of Section~\\ref{sec:2D_fracture_behavior}.}",
    "For the linear elastic sets, the corresponding sensitivity is the scaling with $J_c$ of Section~\\ref{sec:2D_fracture_behavior}. The changes in Table~\\ref{tab:sensitivity_2D} are reported in units of $J_c^0$; the fracture resistance $J_c^{\\text{num}}$ of the homogeneous material was not recomputed for the varied parameters, so the table quantifies the sensitivity of the recorded value, not of the ratio $J_{x,c}^{\\text{eff}}/J_c^{\\text{num}}$.}",
)
ms.rep(
    "The normalized effective crack resistance is therefore the mesh-robust quantity and is used in the comparison with the two-dimensional results.",
    "The normalized effective crack resistance is therefore far less sensitive to the mesh than the absolute values and is used in the comparison with the two-dimensional results.",
)

# ---------------------------------------------------------------- C16ii: deformation-theory wording (introduction)
ms.rep(
    "The Ramberg--Osgood law is non-linear and coincides with von Mises plasticity under monotonic loading, but it does not capture unloading.",
    "The Ramberg--Osgood law is non-linear and corresponds to a deformation-theory description of plasticity, which agrees with von~Mises plasticity under monotonic, proportional loading, but it does not capture unloading.",
)

# ---------------------------------------------------------------- C16iii: Pi named consistently (work of the internal forces)
ms.rep(
    "The elastic energy dissipated due to fracture is defined as",
    "\\replaced{The work of the internal forces up to the strain $\\varepsilon$ is}{The elastic energy dissipated due to fracture is} defined as",
)
ms.rep(
    "Instead, the total dissipated energy $\\Pi^*$ observed in\nthe one-dimensional crack nucleation experiment \\replaced{",
    "Instead, \\replaced{the work of the internal forces until fracture, $\\Pi^*$, in}{the total dissipated energy $\\Pi^*$ observed in}\nthe one-dimensional crack nucleation experiment \\replaced{",
)

# ---------------------------------------------------------------- C16iv: same eps and degradation function
ms.rep(
    "\\added{, since the fracture stress of the linear elastic model is proportional to $\\sqrt{J_c}$}",
    "\\added{, since the fracture stress of the linear elastic model is proportional to $\\sqrt{J_c}$ for fixed $\\epsilon$ and degradation function}",
)

# ---------------------------------------------------------------- C16v: "steady" in the conclusion
ms.rep(
    "surfing boundary conditions to simulate steady macroscopic crack propagation in a \\replaced{resolved porous region}{representative",
    "surfing boundary conditions \\replaced{to enforce macroscopic crack propagation at a prescribed speed}{to simulate steady macroscopic crack propagation} in a \\replaced{resolved porous region}{representative",
)

ms.save()

# ============================================================================ response letter (own prose + quote anchors;
# the %@generated blocks are rebuilt by build_response_quotes.py AFTER pdflatex of the template)
# ---------------------------------------------------------------- preamble comments
lt.rep(
    "% placeholder: partition numbers and Fig. 19b of the w_s = 2L run still running on 2026-10-28",
    "% no placeholders left after part 11 (all w_s = 2L numbers final); macro kept in case new data arrive",
)
lt.rep(
    "% provisional value of the running w_s = 2L run (red, no brackets), check when the run is finished",
    "% provisional (red, no brackets): normalised 3D values, which await the homogeneous references with elastic side layers (strip runs submitted 2026-10-28)",
)

# ---------------------------------------------------------------- quote anchors invalidated by the manuscript edits
lt.rep("%@quote cmd:the energy release rate of the applied K-field is\n", "")                       # A10: sentence deleted
lt.rep("%@quote env:\\changed{1.088J_c^0}\n", "")                                                   # A11: equation deleted
lt.rep(
    "%@quote cmd:For wide ligaments, $w_s \\ge L$, the plastic zone does not span\n",
    "%@quote cmd:For wide ligaments, $w_s \\ge L$,\n",
)                                                                                                   # C10: wording changed

# ---------------------------------------------------------------- equation numbers shifted by A11 (letter's own prose)
lt.rep("the exact energy balance $W_{\\text{ext}} = E_{\\text{el}} + E_{\\text{pl}} + E_{\\text{fr}} + D_{\\text{visc}}$, new eq.~(37),",
       "the exact energy balance $W_{\\text{ext}} = E_{\\text{el}} + E_{\\text{pl}} + E_{\\text{fr}} + D_{\\text{visc}}$, new eq.~(36),")
lt.rep("The energy balance derived from eq.~(14) (now eq.~(37)) contains",
       "The energy balance derived from eq.~(14) (now eq.~(36)) contains")
lt.rep("stress~(26) given in the manuscript.", "stress~(25) given in the manuscript.")
lt.rep("the J-integral is normalised by the thickness $2.5w_c$ (eq.~(40), Section~5.2).",
       "the J-integral is normalised by the thickness $2.5w_c$ (eq.~(39), Section~5.2).")

# ---------------------------------------------------------------- C1c: corners answer scoped to 2D (R1 major 4)
lt.rep(
    "This is now stated in Section~2. The corners are located at least one cell width $w_c$ away from the crack plane and at least one cell width ahead of the first and behind the last pore, the embedding is linear elastic",
    "This is now stated in Section~2. In the two-dimensional specimens, the corners are located at least one cell width $w_c$ away from the crack plane and at least one cell width ahead of the first and behind the last pore; in the three-dimensional specimens, the thin elastic side layers of the embedding also lie on the crack plane, which is stated in Section~5.1. The embedding is linear elastic",
)

# ---------------------------------------------------------------- C9: initial-crack robustness in the R1 major-2 answer
lt.rep(
    "Doubling the length of the initial crack changes $J_{x,c}^{\\text{eff}}$ by $+0.08\\,\\%$ ($1.5w_c$: $+0.02\\,\\%$), five instead of three resolved pore rows change it by",
    "Doubling the length of the initial crack changes $J_{x,c}^{\\text{eff}}$ by $+0.08\\,\\%$ ($1.5w_c$: $+0.02\\,\\%$) at $w_s = L$ and by $-0.06\\,\\%$ ($1.5w_c$: $+0.04\\,\\%$) at $w_s = 0.375L$; these variations also move the crack tip from the edge of the first cell to the center of the first pore and to the middle of the first ligament, so the position of the initial crack relative to the pores is probed in both regimes. Five instead of three resolved pore rows change it by",
)
lt.rep(
    "Section~4.5 also states explicitly what was not tested (other crack-growth directions and initial crack positions, other microstructures, statistics over realizations, amplitude of the K-field, and the length of the resolved region in the crack-growth direction in two dimensions).",
    "Section~4.5 also states explicitly what was not tested (other crack-growth directions, other microstructures, statistics over realizations, amplitude of the K-field, and the length of the resolved region in the crack-growth direction in two dimensions).",
)

# ---------------------------------------------------------------- C10/C11/C15a consistency in the R1 major-3 answer
lt.rep(
    "The plastic work done on the ligament before re-nucleation is supplied by the boundary conditions but does not enter $J_x$ and is not available as crack resistance, hence the reduction of the effective crack resistance below \\EqSigmaStarLabel. For $w_s \\ge L$ the plastic zone does not span the ligament, the trapped tip acts as a constrained notch (triaxiality $1.0$ to $1.1$),",
    "The plastic work done on the ligament remains part of the energy balance; the decisive effect is that yielding lowers the peak stress ahead of the pore, so that re-nucleation is reached at a lower driving force, hence the reduction of the effective crack resistance below \\EqSigmaStarLabel. For $w_s \\ge L$ the recorded maxima indicate that the plastic zone does not span the ligament, the trapped tip acts as a constrained notch (triaxiality measure $1.0$ to $1.1$),",
)
lt.rep(
    "re-nucleation is delayed and the effective crack resistance is raised. This also explains the later saturation of the von~Mises set.",
    "re-nucleation is delayed and the effective crack resistance is raised. This is also consistent with the later saturation of the von~Mises set.",
)

# ---------------------------------------------------------------- C11/C15a consistency in the R3 comment-1 answer
lt.rep(
    "The plastic work spent on stretching the ligament is supplied by the boundary conditions but does not appear in $J_x$ and is not available as crack resistance.",
    "The plastic work spent on stretching the ligament remains part of the energy balance of the specimen; the decisive effect for the recorded maximum is that yielding lowers the peak stress ahead of the pore, so that re-nucleation is reached at a lower driving force.",
)
lt.rep(
    "and the plastic zone sets the interaction length between the pores, which explains the later saturation.",
    "and the plastic zone sets the interaction length between the pores, consistent with the later saturation.",
)

# ---------------------------------------------------------------- A16/A18 consistency in the R3 comment-3 answer
lt.rep(
    "No linear elastic phase-field model has both the strength and the $J_c^0$ of the von~Mises set, since\nplasticity decouples the two.",
    "No linear elastic phase-field model has both the strength and the $J_c^0$ of the von~Mises set, since the plastic\nmodel introduces $\\sigma_y$ and $H$ as parameters that are independent of $J_c$.",
)
lt.rep(
    "whereas the homogeneous plastic material already blunts its crack by the plastic zone, which is contained\nin $J_c^{\\mathrm{num}}$, and in thin ligaments the whole ligament yields, so that the plastic work is not available as\ncrack resistance. Two paragraphs with this argument were added at the end of Section~4.2.",
    "whereas the homogeneous plastic material already blunts its crack by the plastic zone, which is contained\nin $J_c^{\\mathrm{num}}$, and in thin ligaments yielding keeps the peak driving force low, so that the pores add little\ncrack resistance on top of the homogeneous value. Two paragraphs with this argument were added at the end of Section~4.2.",
)

# ---------------------------------------------------------------- C1b: prelim on the normalised 3D values (R3 comment-3 answer)
lt.rep(
    "$J_{x,c}^{\\text{eff}} = 0.86$ to $0.97\\,J_c^{\\mathrm{num}}$ for $w_s = 0.375L$ to $2L$\n(with $J_c^{\\mathrm{num}}$ the value of the three-dimensional solid specimen)",
    "\\prelim{$J_{x,c}^{\\text{eff}} = 0.86$ to $0.97\\,J_c^{\\mathrm{num}}$} for $w_s = 0.375L$ to $2L$\n(with $J_c^{\\mathrm{num}}$ the value of the three-dimensional solid specimen)",
)

# ---------------------------------------------------------------- C16vi + C1b: Additional corrections, item 4
lt.rep(
    "The effective crack resistances are now $J_{x,c}^{\\text{eff}} = 1.22$, $1.26$, $1.31$, $1.34$\nand $1.40\\,J_c^0$, i.e., $0.86$, $0.88$, $0.92$, $0.95$ and $0.97$ in units of the numerical fracture resistance of the\nthree-dimensional solid,",
    "The effective crack resistances are now $J_{x,c}^{\\text{eff}} = 1.22$, $1.25$, $1.31$, $1.34$\nand $1.40\\,J_c^0$, i.e., \\prelim{$0.86$, $0.88$, $0.92$, $0.95$ and $0.97$} in units of the numerical fracture resistance of the\nthree-dimensional solid,",
)
lt.rep(
    "von~Mises material above the homogeneous value ($0.97$ at $w_s = 2L$, against $1.34$ in two dimensions), whereas for thin\nwalls the three-dimensional specimen loses much less than the two-dimensional one ($0.86$ against $0.50$ at $w_s = 0.375L$),",
    "von~Mises material above the homogeneous value (\\prelim{$0.97$} at $w_s = 2L$, against $1.34$ in two dimensions), whereas for thin\nwalls the three-dimensional specimen loses much less than the two-dimensional one (\\prelim{$0.86$} against $0.50$ at $w_s = 0.375L$),",
)

# ---------------------------------------------------------------- C13 + C17: Additional corrections, item 11
lt.rep(
    "solid specimen without pores, computed with the same domain construction and the same mesh rule ($1.421\\,J_c^0$ for the\ndomain of $w_s = L$, $1.445\\,J_c^0$ for $w_s = 2L$; $1.415$, $1.444$ and $1.351\\,J_c^0$ for the shorter four-pore\nspecimen with $h = 0.1L$, $w_s \\le L$ and $2L$, and with $h = 0.07L$), which agrees with the two-dimensional\nvalue $1.425\\,J_c^0$ within $2\\,\\%$.",
    "solid specimen without pores, computed with the same domain construction and the same mesh rule ($1.421\\,J_c^0$ for the\ndomain of $w_s = L$, $1.445\\,J_c^0$ for $w_s = 2L$; $1.415$ and $1.444\\,J_c^0$ for the shorter four-pore\nspecimen with $h = 0.1L$, $w_s \\le L$ and $2L$). These $h = 0.1L$ references agree with the two-dimensional\nvalue $1.425\\,J_c^0$ within $1.5\\,\\%$; the four-pore reference on the finer mesh, $h = 0.07L$, is $1.351\\,J_c^0$,\nconsistent with the mesh dependence of the absolute values (item~9). The surfing boundary conditions, including the\ndisplacements prescribed on the planes $z = z_{\\rm min}$ and $z = z_{\\rm max}$, are applied by the same function to the\nporous and to the reference specimens, which is now also stated in Section~5.1.",
)

# ---------------------------------------------------------------- A23: figure numbering in the minor-25 answer
lt.rep(
    "The result figures of Section~5 were regenerated from the new three-dimensional simulations in the same way (Figs.~16 to 19: crack tip position vs.\\ pseudo-time, the J-integral records of the two- and the three-dimensional model for $w_s = L$ in one plot, the sections between pores two and four for all wall widths, and the effective crack resistance together with the energy partition of both models).",
    "The result figures of Section~5 were regenerated from the new three-dimensional simulations in the same way (Figs.~16 to 20: crack tip position vs.\\ pseudo-time, an illustration of the crack propagating through the central row of pores, the J-integral records of the two- and the three-dimensional model for $w_s = L$ in one plot, the sections between pores two and four for all wall widths, and the effective crack resistance together with the energy partition of both models).",
)
lt.rep(
    "The former figure of the fields $s$ and $\\alpha$ for $w_s = 0.25L$ was removed, since this wall width is no longer part of the three-dimensional series, and Fig.~15 now shows",
    "The former figure of the fields $s$ and $\\alpha$ for $w_s = 0.25L$ was replaced by a single illustration of the crack propagating through the central row of pores (Fig.~17), and Fig.~15 now shows",
)

lt.save()

print("part 11 applied:", MS, LT)
