#!/usr/bin/env python3
"""Round-2 edit script, part 6 (2026-10-22): condense Section 4 and align the letter.

One-shot script (fails on the first anchor that is gone). What it does:
  manuscript_revised_template.tex
    - Table 1 reduced to five rows (0.25, 0.375, 1, 2, 4 L), monotonic statement in text/caption
    - Figs. 11-13 (J_x sections per set) merged into one three-panel figure, stray label removed
    - Table 2 (J_eff), Table 3 (energy partition), Table 4 (ligament state) deleted, key numbers kept in the text
    - Sections 4.5 + 4.6 merged into "Robustness of the Effective Crack Resistance" with one small table
    - Section 4.2 interpretation of Fig. 14b (three comparisons), 4.4 nucleation-regime paragraph without contradiction
    - prose of 4.2-4.6 tightened
  response_to_reviewers_template.tex
    - figure/table/section numbers updated, quotes of deleted tables removed, deleted numbers moved into the answers
"""
import re
import sys

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
            raise SystemExit("%s: anchor found %d times (expected %d):\n%s" % (self.path, c, n, old[:120]))
        self.t = self.t.replace(old, new)

    def rep_line(self, prefix, new_line):
        """replace the unique line that starts with prefix"""
        lines = self.t.split("\n")
        hits = [i for i, l in enumerate(lines) if l.startswith(prefix)]
        if len(hits) != 1:
            raise SystemExit("%s: line prefix found %d times:\n%s" % (self.path, len(hits), prefix[:120]))
        lines[hits[0]] = new_line
        self.t = "\n".join(lines)

    def rep_block(self, first_line, last_line, new_block):
        """replace the block from the unique line first_line to the next line equal to last_line"""
        lines = self.t.split("\n")
        hits = [i for i, l in enumerate(lines) if l == first_line]
        if len(hits) != 1:
            raise SystemExit("%s: block start found %d times:\n%s" % (self.path, len(hits), first_line))
        a = hits[0]
        b = a
        while lines[b] != last_line:
            b += 1
        lines[a:b + 1] = new_block.split("\n")
        self.t = "\n".join(lines)

    def save(self):
        open(self.path, "w", encoding="utf-8").write(self.t)


# ====================================================================== manuscript
ms = Doc(MS)
MS_DONE = "tab:robustness_2D" in ms.t  # manuscript part already applied
FIG = "fig:PAPER_rev2_Jx_sections_three_sets"
JEFF = "fig:PAPER_06b_Jx_vs_wsteg_three_datasets"

# ---------------------------------------------------------------- working note
if not MS_DONE:  # one-shot manuscript part
    ms.rep(
        r"Tables~\ref{tab:effective_stiffness_2D}, \ref{tab:Jeff_2D}, \ref{tab:energy_partition_2D}--\ref{tab:sensitivity_2D}, new vector figures), $J_c^{\text{num}}$ in Section~3.5,",
        r"Tables~\ref{tab:effective_stiffness_2D}, \ref{tab:sensitivity_2D}, \ref{tab:robustness_2D}, new vector figures), $J_c^{\text{num}}$ in Section~3.5,",
    )
    ms.rep(
        r"$K_I^{\text{bc}}$ rule in Section~4.2. Open:",
        r"$K_I^{\text{bc}}$ rule in Section~4.2; condensed 2026-10-22 (edits\_round2\_part6\_condense.py: J$_{\text{eff}}$/partition/ligament tables removed, Figs.~11--13 merged, 4.5+4.6 merged). Open:",
    )

    # ---------------------------------------------------------------- 4.1 Table 1 and text
    ms.rep(
        r"Table~\ref{tab:effective_stiffness_2D} lists the effective constants and the Zener ratio for all wall widths considered. The anisotropy is largest for the smallest wall width and vanishes for widely spaced pores. The effective Poisson's ratio",
        r"Table~\ref{tab:effective_stiffness_2D} lists the effective constants and the Zener ratio for five of the eleven wall widths considered, the values of the other wall widths lie monotonically in between. The anisotropy decreases monotonically with the wall width, from $A = 0.870$ for $w_s = 0.25L$ to $0.998$ for $w_s = 4L$. The effective Poisson's ratio",
    )
    ms.rep(
        r"\caption{\added{Effective elastic constants of the two-dimensional unit cells (plane strain, in units of $\mu_0$, $\lambda_0 = \mu_0$), the isotropic approximation of eq.~(\ref{eq:isotropic_approximation_2D}) and the Zener ratio $A$.}}",
        r"\caption{\added{Effective elastic constants of the two-dimensional unit cells (plane strain, in units of $\mu_0$, $\lambda_0 = \mu_0$), the isotropic approximation of eq.~(\ref{eq:isotropic_approximation_2D}) and the Zener ratio $A$ for five of the eleven wall widths. The values of the intermediate wall widths lie monotonically in between.}}",
    )
    for row in ("0.4   & 1.126", "0.5   & 1.274", "0.6   & 1.406", "0.75  & 1.579", "1.4   & 2.091", "3.0   & 2.610"):
        lines = ms.t.split("\n")
        hits = [i for i, l in enumerate(lines) if l.startswith(row)]
        if len(hits) != 1:
            raise SystemExit("table row not unique: " + row)
        del lines[hits[0]]
        ms.t = "\n".join(lines)

    # ---------------------------------------------------------------- Figs. 11-13 -> one figure
    _a = ms.t.index("{pics/PAPER_rev2_Jx_vs_xct_pores2to4_vM.pdf}")
    _a = ms.t.rfind("    \\begin{minipage}{0.49\\textwidth}", 0, _a)
    _end = "    \\label{fig:side_by_side_plots}\n\\end{figure}"
    _b = ms.t.index(_end, _a) + len(_end)
    ms.t = ms.t[:_a] + (r"""    \includegraphics[width=\textwidth]{pics/PAPER_rev2_Jx_vs_xct_sections_three_sets.pdf}% rev2: one figure replaces PAPER_04_Jx_vs_xct_all_vonMises.png, PAPER_03A_Jx_vs_xct_all_linear_elastic_gc10.png and PAPER_04_Jx_vs_xct_all_SIG.png (Figs. 11-13 of v2.0)
        \caption{\replaced{J-Integral $J_x$ vs.\ crack tip position $x_{ct}$ between pores two and four for varying $w_s$ and the parameter sets a) \vMLabel, b) \EqJcLabel and c) \EqSigmaStarLabel.}{Section of the J-Integral $J_x$ vs.\ crack tip position $x_{ct}$ between pores two and four for the parameter set \vMLabel and varying $w_s$.}}
        \label{@FIG@}
    \end{figure}""").replace("@FIG@", FIG) + ms.t[_b:]
    # the "\begin{figure}[htb]\n    \centering" of the old minipage figure is kept

    # ---------------------------------------------------------------- 4.2 text
    ms.rep(
        r"This pattern is shown for varying $w_s$ in Fig.~\ref{fig:PAPER_04_Jx_vs_xct_all_von_Mises} for \vMLabel, Fig.~\ref{fig:PAPER_03A_Jx_vs_xct_all_linear_elastic_gc10} for \EqJcLabel, and Fig.~\ref{fig:PAPER_03_Jx_vs_xct_all_linear_elastic_gc05679.png} for \EqSigmaStarLabel.",
        r"This pattern is shown for varying $w_s$ in \replaced{Fig.~\ref{@FIG@}a) for \vMLabel, b) for \EqJcLabel and c) for \EqSigmaStarLabel}{Fig.~11 for \vMLabel, Fig.~12 for \EqJcLabel, and Fig.~13 for \EqSigmaStarLabel}.".replace("@FIG@", FIG),
    )
    ms.rep(
        r"\added{, normalized by $J_c^0$ in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets}a) and by the numerical fracture resistance $J_c^{\text{num}}$ of the respective parameter set in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets}b). The values are listed in Table~\ref{tab:Jeff_2D}}.",
        r"\added{, normalized by $J_c^0$ in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets}a) and by the numerical fracture resistance $J_c^{\text{num}}$ of the respective parameter set, Section~\ref{sec:steady_growth_griffith}, in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets}b)}.",
    )
    ms.rep(
        r"\replaced{The increase is steep for thin ligaments and levels off for wide ligaments. At $w_s = 4L$, the effective crack resistance is $1.90J_c^0 = 1.34J_c^{\text{num}}$ for \vMLabel, $2.21J_c^0 = 2.03J_c^{\text{num}}$ for \EqJcLabel and $1.22J_c^0 = 2.03J_c^{\text{num}}$ for \EqSigmaStarLabel. Thus, a pronounced toughening effect with respect to the fracture resistance of the homogeneous material is observed for all parameter sets. Relative to its own $J_c^{\text{num}}$, however, the von~Mises set is toughened less by the pores than the linear elastic sets for all wall widths, Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets}b).}",
        r"\replaced{The increase is steep for thin ligaments and levels off for wide ligaments. At $w_s = 4L$, the effective crack resistance is $1.90J_c^0 = 1.34J_c^{\text{num}}$ for \vMLabel, $2.21J_c^0 = 2.03J_c^{\text{num}}$ for \EqJcLabel and $1.22J_c^0 = 2.03J_c^{\text{num}}$ for \EqSigmaStarLabel. The pores thus toughen all three materials with respect to the homogeneous material, but relative to its own $J_c^{\text{num}}$ the von~Mises set is toughened less than the linear elastic sets for all wall widths, Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets}b).}",
    )
    # delete Table 2
    ms.rep_block(r"\caption{\added{Effective crack resistance $J_{x,c}^{\text{eff}}$ of the two-dimensional porous material in units of $J_c^0$ and of the numerical fracture resistance $J_c^{\text{num}}$ of the respective parameter set ($1.425$, $1.088$ and $0.604\,J_c^0$ for \vMLabel, \EqJcLabel and \EqSigmaStarLabel).}}",
                 r"\end{table}", "%%TAB2DELETED")
    ms.rep("\\begin{table}[htb]\n\\centering\n%%TAB2DELETED\n\n", "")

    ms.rep(
        r"\replaced{The ordering \EqJcLabel $>$ \vMLabel $>$ \EqSigmaStarLabel observed for $w_s = L$ holds for all wall widths $w_s \ge 0.6L$. For thin walls, $w_s \le 0.5L$, the ductile material law leads to a reduction of the effective crack resistance below the values obtained for \EqSigmaStarLabel. The curves of \vMLabel and \EqSigmaStarLabel cross between $w_s = 0.5L$ and $0.6L$. Neither the one-dimensional tensile strength, which is equal for \vMLabel and \EqSigmaStarLabel, nor the microscopic fracture resistance $J_c$, which is equal for \vMLabel and \EqJcLabel, is therefore a consistent indicator of the effective crack resistance. The two regimes are analyzed in Sections~\ref{sec:2D_energy_partition} and~\ref{sec:2D_mechanisms}.}",
        r"\replaced{The ordering \EqJcLabel $>$ \vMLabel $>$ \EqSigmaStarLabel observed for $w_s = L$ holds for all wall widths $w_s \ge 0.6L$. For thin walls, $w_s \le 0.5L$, the ductile material law reduces the effective crack resistance below the values of \EqSigmaStarLabel, the curves of \vMLabel and \EqSigmaStarLabel cross between $w_s = 0.5L$ and $0.6L$. Neither the one-dimensional tensile strength, which is equal for \vMLabel and \EqSigmaStarLabel, nor the microscopic fracture resistance $J_c$, which is equal for \vMLabel and \EqJcLabel, is therefore a consistent indicator of the effective crack resistance. The two regimes are analyzed in Sections~\ref{sec:2D_energy_partition} and~\ref{sec:2D_mechanisms}.}",
    )
    # scaling paragraph + interpretation of Fig. 14b (three comparisons)
    ms.rep_line(
        r"\added{The two linear elastic parameter sets differ only in the value of $J_c$.",
        r"""\added{The two linear elastic parameter sets differ only in the value of $J_c$. For the linear elastic phase-field model, $J_c$ is the only energy scale once the elastic constants and the lengths are fixed, and the amplitude of the K-field is proportional to $\sqrt{J_c}$ by the choice of $K_I^{\text{bc}}$. In the rate-independent limit $M \to \infty$, the displacement field scales with $\sqrt{J_c}$ and all energies, including $J_{x,c}^{\text{eff}}$, scale with $J_c$, while the crack pattern is unchanged. The simulations confirm this scaling. The ratio of the effective crack resistances of \EqJcLabel and \EqSigmaStarLabel lies between $1.809$ and $1.810$ for all wall widths, compared with $1/0.5523 = 1.811$ for the prescribed fracture resistances, and the finite mobility $M$ does not disturb the scaling noticeably. In Fig.~\ref{@J@}b), the two linear elastic sets consequently fall onto one curve.}

    \added{The three comparisons in Fig.~\ref{@J@} answer different questions. For equal microscopic fracture resistance $J_c^0$, \vMLabel lies below \EqJcLabel for all wall widths. For equal one-dimensional strength, \vMLabel lies above \EqSigmaStarLabel from $w_s \approx 0.6L$ on, Fig.~\ref{@J@}a). For equal crack resistance of the homogeneous material, which is the comparison of Fig.~\ref{@J@}b), \vMLabel lies below the linear elastic curve for all wall widths. No linear elastic phase-field model has both the strength and the fracture resistance $J_c^0$ of \vMLabel, since plasticity decouples the two. The toughness gain of \vMLabel over \EqSigmaStarLabel in Fig.~\ref{@J@}a) stems from the material model, which already raises the crack resistance of the homogeneous material: $J_c^{\text{num}} = 1.425J_c^0$ against $1.088J_c^0$ for \EqJcLabel with the same $J_c^0$, i.e., by $31\,\%$, and by a factor of $2.36$ against \EqSigmaStarLabel. Fig.~\ref{@J@}b) isolates the toughening by the microstructure. The pore arrangement raises the crack resistance of the linear elastic materials by a factor of about two relative to their homogeneous value, with a saturation value of $2.03J_c^{\text{num}}$, that of the plastic material only by $1.34$, and below $w_s \approx 0.9L$ it even lowers it, $J_{x,c}^{\text{eff}} < J_c^{\text{num}}$. The microstructure thus preserves only part of the gain that plasticity gives to the homogeneous material. By the scaling, a linear elastic set with $J_c^{\text{num}} = 1.425J_c^0$ would reach $2.03 \times 1.425J_c^0 = 2.89J_c^0$ at $w_s = 4L$ instead of the $1.90J_c^0$ of \vMLabel. The mechanism is identified in Sections~\ref{sec:2D_energy_partition} and~\ref{sec:2D_mechanisms}. Crack trapping and forced re-nucleation at the blunt pore wall is an efficient toughening mechanism for a linear elastic material, whose price is that $47$ to $83\,\%$ of the dissipated energy go into the unstable snap-through of the ligament. The homogeneous plastic material already blunts its crack by the plastic zone, which is contained in $J_c^{\text{num}}$, and in thin ligaments the whole ligament yields, so that the plastic work is not available as crack resistance.}""".replace("@J@", JEFF),
    )

    # ---------------------------------------------------------------- 4.3 energy partition
    ms.rep_line(
        r"\added{The model of Section~\ref{sec:phase_fiel_model_ductile} obeys an exact energy balance,",
        r"\added{The model of Section~\ref{sec:phase_fiel_model_ductile} obeys an exact energy balance, which follows from the time derivative of the total energy, eq.~(\ref{eq:E_ep_total}), with the equilibrium condition, eq.~(\ref{eq:equilibrium}), and the evolution equation of the phase field, eq.~(\ref{eq:evolution_s}). The work $W_{\text{ext}}$ of the tractions on the outer boundary, where the surfing displacements are prescribed, is}",
    )
    ms.rep_line(
        r"\added{where $E_{\text{el}}$, $E_{\text{pl}}$ and $E_{\text{fr}}$ denote",
        r"\added{where $E_{\text{el}}$, $E_{\text{pl}}$ and $E_{\text{fr}}$ denote the integrals of $\psi_{\text{el}}$, $\psi_{\text{pl}}$ and $\psi_{\text{fr}}$ over $\Omega$, i.e., the three contributions to $E_{\text{ep}}$ in eq.~(\ref{eq:E_ep_total}), and $D_{\text{visc}}$ is the energy dissipated by the viscous evolution of the phase field. The plastic work is stored in the degraded form $\psi_{\text{pl}} = g(s)(\nicefrac{1}{2}H\alpha^2 + \sigma_y\alpha)$, eq.~(\ref{eq:psi_pl}), and acts as a crack driving force through $g'(s)$ in eq.~(\ref{eq:evolution_s}). Plastic work done in material that fails later is released by the degradation and reappears in $E_{\text{fr}}$ and $D_{\text{visc}}$, whereas $E_{\text{pl}}$ records the plastic energy stored in intact material. The viscous term collects the energy released when a ligament fails in an unstable manner, which would become kinetic energy in a dynamic formulation. For the linear elastic model, $E_{\text{pl}} = 0$. All terms of eq.~(\ref{eq:energy_balance}) are recorded in every time step.}",
    )
    ms.rep_line(
        r"\added{The partition is evaluated per pore-to-pore cycle.",
        r"\added{The partition is evaluated per pore-to-pore cycle. A cycle starts when the crack enters a pore and ends when it enters the next pore, i.e., it comprises the loading of one ligament while the crack tip is trapped, the re-nucleation and the failure of this ligament. The energy dissipated during a cycle is}",
    )
    ms.rep_line(
        r"\added{and $G_{\text{diss}} = \Delta D_{\text{tot}}/w_c$ is the energy dissipated per unit crack advance, averaged over the cycle.",
        r"\added{and $G_{\text{diss}} = \Delta D_{\text{tot}}/w_c$ is the energy dissipated per unit crack advance, averaged over the cycle. Figure~\ref{fig:energy_partition_cycle} shows the three fractions of $\Delta D_{\text{tot}}$ and the ratio $G_{\text{diss}}/J_{x,c}^{\text{eff}}$ for the cycle that contains the ligament in which $J_{x,c}^{\text{eff}}$ is attained. The energy dissipated per cycle is nearly constant from cycle to cycle, see Section~\ref{sec:2D_representativity}. The stored elastic energy decreases over a cycle, by $0.05$ to $0.6\,\Delta D_{\text{tot}}$ depending on the case, because the K-field applied at $t = 0$ stores elastic energy that is drawn down as the crack advances. The specimen is therefore not in a strict steady state.}",
    )
    # delete Table 3
    ms.rep_block(r"\caption{\added{Energy partition of the cycle that contains the ligament in which $J_{x,c}^{\text{eff}}$ is attained: fractions of the dissipated energy $\Delta D_{\text{tot}}$, eq.~(\ref{eq:D_tot_cycle}), and energy dissipated per unit crack advance $G_{\text{diss}} = \Delta D_{\text{tot}}/w_c$. The fractions of \EqJcLabel agree with those of \EqSigmaStarLabel to two digits, and its $G_{\text{diss}}$ is larger by a factor of $1.75$ to $1.89$, close to the ratio $1/0.5523 = 1.81$ of the prescribed fracture resistances, see Section~\ref{sec:2D_fracture_behavior}.}}",
                 r"\end{table}", "%%TAB3DELETED")
    ms.rep("\\begin{table}[htb]\n\\centering\n%%TAB3DELETED\n\n", "")
    ms.rep_line(
        r"\added{For the two linear elastic sets, which give the same fractions to two digits, the share of the viscous term is large.",
        r"\added{For the two linear elastic sets, which give the same fractions to two digits, the viscous share is large. Between $83\,\%$ of the dissipated energy for $w_s = 0.25L$ and $47\,\%$ for $w_s = 4L$ are lost in the unstable snap-through of the ligament, and only $17$ to $53\,\%$ create crack surface. The elastic energy stored up to the moment of re-nucleation exceeds the fracture energy of the ligament by far, and the excess is released in the jump. For the von~Mises set, the viscous share is $12$ to $37\,\%$, the fracture share $49$ to $74\,\%$, and $12$ to $16\,\%$ of the dissipated energy remain stored as plastic energy in the intact material along the crack path. The difference between the material laws is largest for thin ligaments. The energy dissipated per unit crack advance shows the same crossover as $J_{x,c}^{\text{eff}}$. It is smaller for \vMLabel than for \EqSigmaStarLabel for thin walls, $0.48$ vs.\ $0.71\,J_c^0$ at $w_s = 0.25L$, equal at $w_s = 0.6L$ and larger for wide walls, $1.39$ vs.\ $0.89\,J_c^0$ at $w_s = 4L$. The ratio $G_{\text{diss}}/J_{x,c}^{\text{eff}}$, Fig.~\ref{fig:energy_partition_cycle}b), decreases from $0.93$ to $1.10$ for $w_s = 0.25L$ to $0.73$ for $w_s = 4L$, since $J_{x,c}^{\text{eff}}$ is the peak of $J_x$ in the cycle and $G_{\text{diss}}$ its average. For thin walls, the ductile material law thus reduces not only the peak driving force but also the energy dissipated per unit crack advance, which is the quantity closest to an effective fracture energy in the sense of the energy dissipated along the crack path, see Michel and Suquet~\cite{michel_merits_2022}. The mechanism is identified in the next section.}",
    )

    # ---------------------------------------------------------------- 4.4 mechanisms
    ms.rep_line(
        r"\added{To relate the re-nucleation in the porous specimen to the one-dimensional results",
        r"\added{To relate the re-nucleation in the porous specimen to the one-dimensional results of Section~\ref{sec:stress_driven_nucleation}, the maxima of the normal stress $\sigma_{yy}$, of the von~Mises stress $\sigma_{\text{vM}}$, of the hydrostatic stress $\sigma_h = \nicefrac{1}{3}\,\text{tr}(\boldsymbol{\sigma})$ and of the plastic variable $\alpha$ in the next intact ligament ahead of the trapped crack tip are recorded in every time step. Figure~\ref{fig:ligament_state} shows these values at the peak of $J_x$, i.e., immediately before re-nucleation. The maximum normal stress is normalized by the one-dimensional fracture stress $\sigma^*_{\text{1D}}$ of the respective parameter set, $1.28\sigma_y$ for \vMLabel and \EqSigmaStarLabel and $1.72\sigma_y$ for \EqJcLabel, with $\sigma_y = \sqrt{\nicefrac{J_c^0\mu_0}{L}}$, and the ratio $\sigma_h/\sigma_{\text{vM}}$ at the location of the maximum normal stress is the stress triaxiality.}",
    )
    # delete Table 4
    ms.rep_block(r"\caption{\added{Effective crack resistance and state of the next intact ligament immediately before re-nucleation for the parameter set \vMLabel: maxima of the normal stress, the von~Mises stress, the hydrostatic stress and the plastic variable in the ligament, and the triaxiality at the location of $\sigma_{yy,\max}$. Stresses in units of $\sqrt{\nicefrac{J_c^0\mu_0}{L}}$, the one-dimensional fracture stress is $\sigma^*_{\text{vM}} = 1.28$.}}",
                 r"\end{table}", "%%TAB4DELETED")
    ms.rep("\\begin{table}[htb]\n\\centering\n%%TAB4DELETED\n\n", "")
    ms.rep_line(
        r"\added{For the two linear elastic sets, re-nucleation occurs when the maximum normal stress",
        r"\added{For the two linear elastic sets, re-nucleation occurs when the maximum normal stress in the ligament reaches the one-dimensional fracture stress, $\sigma_{yy,\max} = 0.95$ to $1.02\,\sigma^*_{\text{1D}}$ for all wall widths and both values of $J_c$, at a triaxiality of $0.49$ to $0.53$. The nucleation in the ligament is thus stress controlled, and the one-dimensional value is a good estimate of the nucleation stress in the two-dimensional porous specimen. This confirms, for the present geometry and the cubic degradation function, the observations of Kuhn~\cite{kuhn_numerical_2013} and Schl\"uter~\cite{schlueter_phase_2018} referred to in Section~\ref{sec:stress_driven_nucleation}.}",
    )
    ms.rep_line(
        r"\added{For the von~Mises set, two regimes can be distinguished",
        r"\added{For the von~Mises set, two regimes can be distinguished in Fig.~\ref{fig:ligament_state}. For thin ligaments, $w_s \le 0.6L$, the ligament yields across its full width before re-nucleation. The maximum of $\alpha$ is $0.66$ to $0.73$, the triaxiality is $0.67$ to $0.74$, and re-nucleation occurs at $\sigma_{yy,\max} = 0.95$ to $1.11\,\sigma^*_{\text{vM}}$, i.e., at about the one-dimensional fracture stress after a large plastic strain, as in the bar of Section~\ref{sec:stress_driven_nucleation}. Once the ligament has yielded, its tangent stiffness drops to the hardening modulus $H$, so that its opening grows with the advancing K-field at nearly constant stress, and $J_x$ at re-nucleation stays low. The plastic work done on the ligament before re-nucleation is supplied by the boundary conditions, but it does not enter $J_x$, which measures the energy release rate of the elastic surroundings, and it is not available as crack resistance. In this nucleation regime, the ductile material law reduces the effective crack resistance below that of \EqSigmaStarLabel, whose linear elastic ligament has to be loaded to a higher driving force before it reaches $\sigma^*$, Fig.~\ref{@J@}. The energy dissipated per unit crack advance is reduced as well, Section~\ref{sec:2D_energy_partition}, because the linear elastic ligament stores more elastic energy up to failure, which is released in the snap-through.}".replace("@J@", JEFF),
    )
    ms.rep_line(
        r"\added{For wide ligaments, $w_s \ge L$, the plastic zone does not span the ligament.",
        r"\added{For wide ligaments, $w_s \ge L$, the plastic zone does not span the ligament. The maximum of $\alpha$ drops to $0.44$ to $0.60$, the trapped crack tip at the pore surface acts as a notch, and the material ahead of it is constrained by the surrounding elastic material. The triaxiality rises to $1.0$ to $1.1$, the von~Mises stress at the location of the maximum normal stress remains at the yield stress, $0.98$ to $1.06\,\sigma_y$, and the normal stress reaches $\sigma_{yy,\max} = 1.67$ to $1.71\sqrt{\nicefrac{J_c^0\mu_0}{L}} = 1.31$ to $1.34\,\sigma^*_{\text{vM}}$, i.e., the level of the one-dimensional fracture stress of \EqJcLabel. Since the hydrostatic stress does not enter the yield condition, the constrained ligament carries a normal stress well above the one-dimensional fracture stress before the driving force suffices for re-nucleation. Re-nucleation is delayed, and the plastic work ahead of the trapped tip has to be supplied before the crack can advance. In this propagation regime, the ductile material law raises the effective crack resistance above that of \EqSigmaStarLabel. The interaction length between neighboring pores is then set by the size of the plastic zone ahead of the trapped tip rather than by the pore diameter, consistent with the later saturation of the von~Mises set in Fig.~\ref{@J@}.}".replace("@J@", JEFF),
    )
    ms.rep_line(
        r"\added{The sensitivity of $J_{x,c}^{\text{eff}}$ with respect to the plastic parameters confirms the two regimes",
        r"\added{The sensitivity of $J_{x,c}^{\text{eff}}$ with respect to the plastic parameters confirms the two regimes, Table~\ref{tab:sensitivity_2D}. Halving or doubling the hardening modulus changes the effective crack resistance by $-12\,\%$ and $+19\,\%$ at $w_s = 0.375L$, but only by $-4\,\%$ and $+6\,\%$ at $w_s = L$. Increasing the yield stress by $50\,\%$ raises it by $37\,\%$ at $w_s = 0.375L$ and by $3\,\%$ at $w_s = L$, a reduction by $25\,\%$ changes it by $-2\,\%$ and $-6\,\%$. In the nucleation regime, the effective crack resistance is governed by the stress-strain response of the yielded ligament and is therefore sensitive to $\sigma_y$ and $H$. In the propagation regime, it is governed by the re-nucleation at the constrained notch and is much less sensitive to both. For the linear elastic sets, the corresponding sensitivity is the scaling with $J_c$ of Section~\ref{sec:2D_fracture_behavior}, which the simulations confirm to $0.1\,\%$.}",
    )

    # ---------------------------------------------------------------- 4.5 + 4.6 -> robustness
    ms.rep_block(
        r"\subsection{\added{Influence of the Isotropic Approximation of the Embedding}}",
        r"\added{These tests do not establish representativity in the sense of Hill, see Section~\ref{sec:numericalExperiment}. Not tested were the influence of the length of the resolved region in the crack-growth direction, of other crack-growth directions relative to the pore lattice, of the position of the initial crack relative to the pores, and of the microstructure itself, since one periodic arrangement of equal pores is considered and no statistics over microstructure realizations is attempted. The amplitude of the applied K-field is fixed by the rule of Section~\ref{sec:2D_fracture_behavior} and was not varied. The conclusions of this work are therefore restricted to the stated microstructure, crack-growth direction and loading.}",
        r"""\subsection{\added{Robustness of the Effective Crack Resistance}}
    \label{sec:2D_cubic_embedding}\label{sec:2D_representativity}

    \begin{table}[htb]
    \centering
    \caption{\added{Change of the effective crack resistance of the parameter set \vMLabel by the full square-symmetric stiffness of the embedding (Zener ratio $A$ of Table~\ref{tab:effective_stiffness_2D}), by a longer initial crack and by five instead of three resolved pore rows.}}
    \label{tab:robustness_2D}
    \begin{tabular}{llc}
    \toprule
    variation & case & change of $J_{x,c}^{\text{eff}}$ \\
    \midrule
    square-symmetric embedding & $w_s = 0.375L$, $A = 0.890$ & $+0.18\,\%$ \\
                               & $w_s = L$, $A = 0.960$      & $+0.08\,\%$ \\
                               & $w_s = 4L$, $A = 0.998$     & $+0.01\,\%$ \\
    initial crack $1.5w_c$     & $w_s = L$                   & $+0.02\,\%$ \\
    initial crack $2w_c$       & $w_s = L$                   & $+0.08\,\%$ \\
    five pore rows             & $w_s = L$                   & $-0.02\,\%$ \\
                               & $w_s = 0.375L$              & $+1.1\,\%$ \\
                               & $w_s = L$, \EqSigmaStarLabel & $-0.09\,\%$ \\
    \bottomrule
    \end{tabular}
    \end{table}

    \added{The embedding of the resolved porous region is assigned the isotropic approximation, eq.~(\ref{eq:isotropic_approximation_2D}), of the square-symmetric stiffness of Table~\ref{tab:effective_stiffness_2D}. To test this approximation, the simulations for $w_s = 0.375L$, $L$ and $4L$ with the parameter set \vMLabel were repeated with the full stiffness $C_{11}$, $C_{12}$, $C_{33}$ assigned to the embedding. The effective crack resistance changes by at most $+0.18\,\%$, Table~\ref{tab:robustness_2D}, even for the thinnest wall, for which the anisotropy is largest. This is plausible, since the embedding transmits the K-field and the J-integral is evaluated on its outer boundary, whereas the re-nucleation that determines $J_{x,c}^{\text{eff}}$ takes place in the resolved ligaments. The three-dimensional unit cells of Section~\ref{sec:3D_effective_medium} are closer to isotropic, $A \ge 0.978$ for $w_s \ge 0.375L$, and two earlier three-dimensional runs on a coarser mesh with the full cubic stiffness changed $J_{x,c}^{\text{eff}}$ by $-0.37\,\%$ ($w_s = 0.375L$) and $-0.22\,\%$ ($w_s = L$). The corners of the resolved region, where the material discontinuity meets a geometric corner, act as stress concentrators, see Section~\ref{sec:numericalExperiment}. In none of the simulations did a crack nucleate at a corner, the crack pattern remained confined to the central row of pores, Fig.~\ref{fig:setup_crack_growth}b), and since the embedding is linear elastic and the J-integral is evaluated on its outer boundary, the stress concentration at the corners does not enter the recorded driving force.}

    \added{The definition of $J_{x,c}^{\text{eff}}$ in Section~\ref{sec:numericalExperiment} refers to one initial crack and one resolved region. Two of the dependencies that can be tested within the present setup were probed, Table~\ref{tab:robustness_2D}. Increasing the length of the initial crack from $w_c$ to $1.5w_c$ and $2w_c$ changes $J_{x,c}^{\text{eff}}$ by $+0.02\,\%$ and $+0.08\,\%$, and enlarging the resolved region from three to five rows of pores changes it by at most $1.1\,\%$. In addition, the response along the crack path is periodic. The peaks of $J_x$ in successive ligaments, Fig.~\ref{fig:PAPER_02_all_Jx_vs_xct_pf_diamond_holes}, and the energy dissipated per cycle, Section~\ref{sec:2D_energy_partition}, are nearly constant from the second cycle on. The first cycle, during which the K-field builds up, is lower, the last cycle is affected by the end of the resolved region, and the inner cycles agree within about $7\,\%$ ($10\,\%$ for \EqJcLabel at $w_s = 0.25L$), e.g., $G_{\text{diss}} = 1.30$, $1.36$, $1.39$ and $1.37\,J_c^0$ in the first four cycles for \vMLabel and $w_s = L$.}

    \added{These tests do not establish representativity in the sense of Hill, see Section~\ref{sec:numericalExperiment}. Not tested were the influence of the length of the resolved region in the crack-growth direction, of other crack-growth directions relative to the pore lattice, of the position of the initial crack relative to the pores, and of the microstructure itself, since one periodic arrangement of equal pores is considered and no statistics over microstructure realizations is attempted. The amplitude of the applied K-field is fixed by the rule of Section~\ref{sec:2D_fracture_behavior} and was not varied. The conclusions of this work are therefore restricted to the stated microstructure, crack-growth direction and loading.}""",
    )

    for gone in ("tab:Jeff_2D", "tab:energy_partition_2D", "tab:ligament_state_vM",
                 "fig:PAPER_04_Jx_vs_xct_all_von_Mises", "fig:PAPER_03A_Jx_vs_xct_all_linear_elastic_gc10",
                 "fig:PAPER_03_Jx_vs_xct_all_linear_elastic_gc05679.png", "fig:side_by_side_plots"):
        body = ms.t[ms.t.index("\\begin{document}"):ms.t.index("\\section{Conclusion")]
        body = "\n".join(re.sub(r"(?<!\\)%.*", "", l) for l in body.split("\n"))  # ignore comments
        if "{%s}" % gone in body:
            raise SystemExit("label still referenced in the manuscript: " + gone)
    ms.save()

# ====================================================================== letter
lt = Doc(LT)

# ---------------------------------------------------------------- editor summary
lt.rep(
    r"\item a mechanism study was added to the two-dimensional results (new Sections~4.3 and 4.4, Figs.~15 and 16, Tables~3 to 5):",
    r"\item a mechanism study was added to the two-dimensional results (new Sections~4.3 and 4.4, Figs.~13 and 14, Table~2):",
)
lt.rep(
    r"\item the square resp.\ cubic symmetry of the effective stiffness of the unit cells is quantified (Zener ratio, Tables~1 and 6) and the influence of the isotropic approximation of the embedding is tested by simulations with the full anisotropic stiffness (new Section~4.5): the effective crack resistance changes by at most $0.2\,\%$;",
    r"\item the square resp.\ cubic symmetry of the effective stiffness of the unit cells is quantified (Zener ratio, Tables~1 and 4) and the influence of the isotropic approximation of the embedding is tested by simulations with the full anisotropic stiffness (new Section~4.5, Table~3): the effective crack resistance changes by at most $0.2\,\%$;",
)
lt.rep(
    r"\item the influence of the length of the initial crack and of the number of resolved pore rows on the effective crack resistance is tested (new Section~4.6): at most $1.1\,\%$;",
    r"\item the influence of the length of the initial crack and of the number of resolved pore rows on the effective crack resistance is tested (Section~4.5, Table~3): at most $1.1\,\%$;",
)

# ---------------------------------------------------------------- R1 major 2
lt.rep(
    r"the length of the initial crack and the number of resolved pore rows are varied in the new Section~4.6. Doubling",
    r"the length of the initial crack and the number of resolved pore rows are varied in the new Section~4.5 (Table~3). Doubling",
)
lt.rep(
    r"In addition, the energy dissipated per pore-to-pore cycle (new Section~4.3) is nearly constant from cycle to cycle (within a few per cent, except for the first cycle and the last cycle at the end of the resolved region), so that the recorded peak belongs to a periodic regime. Section~4.6 also states explicitly what was not tested",
    r"In addition, the energy dissipated per pore-to-pore cycle (new Section~4.3) is nearly constant from cycle to cycle (within about $7\,\%$, except for the first cycle and the last cycle at the end of the resolved region, Section~4.5), so that the recorded peak belongs to a periodic regime. Section~4.5 also states explicitly what was not tested",
)
lt.rep(
    r"\relhead{New Section~4.6 (influence of the initial crack and of the size of the resolved region)}",
    r"\relhead{New Section~4.5 (robustness of the effective crack resistance: initial crack, number of pore rows, periodicity, what was not tested), Table~3. The paragraphs on the embedding are quoted under major point~4.}",
)

# ---------------------------------------------------------------- R1 major 3
lt.rep(
    r"a mechanism study was added in the new Sections~4.3 and 4.4 (Figs.~15 and 16, Tables~3 to 5).",
    r"a mechanism study was added in the new Sections~4.3 and 4.4 (Figs.~13 and 14, Table~2). The complete numbers behind Figs.~12 to 14 are listed in this letter, since the corresponding tables were removed from the manuscript to keep it short.",
)
lt.rep(
    r"for the cycle that contains the ligament in which $J_{x,c}^{\text{eff}}$ is attained, Fig.~15 and Table~3. For the linear elastic sets,",
    r"for the cycle that contains the ligament in which $J_{x,c}^{\text{eff}}$ is attained, Fig.~13 and the following table (fractions of the dissipated energy $\Delta D_{\text{tot}}$ and $G_{\text{diss}} = \Delta D_{\text{tot}}/w_c$ in units of $J_c^0$, the two linear elastic sets give the same fractions to two digits, $G_{\text{diss}}$ of \EqJcLabel is larger than that of \EqSigmaStarLabel by a factor of $1.75$ to $1.89$). For the linear elastic sets,",
)
lt.rep(
    r"The energy dissipated per cycle is nearly constant from cycle to cycle (periodic regime), and the two linear elastic sets give the same fractions to two digits.",
    r"""The energy dissipated per cycle is nearly constant from cycle to cycle (periodic regime, Section~4.5).
\begin{center}\small
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{4}{c}{\vMLabel} & \multicolumn{3}{c}{\EqSigmaStarLabel} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-8}
$w_s/L$ & fracture & plastic & viscous & $G_{\text{diss}}$ & fracture & viscous & $G_{\text{diss}}$ \\
\midrule
0.25  & 0.535 & 0.138 & 0.327 & 0.479 & 0.173 & 0.827 & 0.709 \\
0.375 & 0.526 & 0.124 & 0.350 & 0.665 & 0.209 & 0.791 & 0.857 \\
0.6   & 0.493 & 0.141 & 0.366 & 0.989 & 0.235 & 0.765 & 0.989 \\
1.0   & 0.493 & 0.157 & 0.350 & 1.387 & 0.286 & 0.714 & 1.050 \\
2.0   & 0.589 & 0.147 & 0.264 & 1.517 & 0.395 & 0.605 & 0.998 \\
4.0   & 0.744 & 0.135 & 0.121 & 1.385 & 0.530 & 0.470 & 0.889 \\
\bottomrule
\end{tabular}
\end{center}""",
)
lt.rep(
    r"are recorded in every time step and evaluated at the peak of $J_x$, Fig.~16 and Table~4. The linear elastic sets",
    r"are recorded in every time step and evaluated at the peak of $J_x$, Fig.~14. The values for the von~Mises set are (stresses in units of $\sqrt{J_c^0\mu_0/L}$, one-dimensional fracture stress $\sigma^*_{\text{vM}} = 1.28$):"
    + r"""
\begin{center}\small
\begin{tabular}{lcccccc}
\toprule
$w_s/L$ & $J_{x,c}^{\text{eff}}/J_c^0$ & $\sigma_{yy,\max}$ & $\sigma_{\text{vM}}$ & $\sigma_h$ & $\sigma_h/\sigma_{\text{vM}}$ & $\alpha_{\max}$ \\
\midrule
0.25  & 0.514 & 1.218 & 1.095 & 0.732 & 0.67 & 0.656 \\
0.375 & 0.718 & 1.241 & 1.110 & 0.773 & 0.70 & 0.727 \\
0.6   & 1.084 & 1.427 & 1.112 & 0.826 & 0.74 & 0.708 \\
1.0   & 1.563 & 1.711 & 1.064 & 1.057 & 0.99 & 0.600 \\
2.0   & 1.830 & 1.693 & 1.015 & 1.076 & 1.06 & 0.442 \\
4.0   & 1.903 & 1.672 & 0.980 & 1.075 & 1.10 & 0.457 \\
\bottomrule
\end{tabular}
\end{center}
The linear elastic sets""",
)
lt.rep(
    r"re-nucleates at the one-dimensional fracture stress after a large plastic strain, i.e., it behaves like the one-dimensional bar of Section~3.6, which is the reduced model for this regime. Once yielded, the ligament opens at nearly constant stress, so that the macroscopic driving force at re-nucleation is close to the value at which the ligament yields and lower than for the linear elastic ligament of equal strength. The plastic work spent on stretching the ligament is supplied by the boundary conditions but does not enter $J_x$, hence the reduction of the effective crack resistance below \EqSigmaStarLabel.",
    r"re-nucleates at about the one-dimensional fracture stress after a large plastic strain, as the one-dimensional bar of Section~3.6, which is the reduced model for this regime. Once yielded, the tangent stiffness of the ligament drops to the hardening modulus, the ligament opens at nearly constant stress and $J_x$ at re-nucleation stays low, whereas the linear elastic ligament of equal strength has to be loaded to a higher driving force before it reaches $\sigma^*$. The plastic work done on the ligament before re-nucleation is supplied by the boundary conditions but does not enter $J_x$ and is not available as crack resistance, hence the reduction of the effective crack resistance below \EqSigmaStarLabel.",
)
lt.rep(
    r"at $w_s = 0.375L$ and $w_s = L$, Table~5. In the nucleation regime",
    r"at $w_s = 0.375L$ and $w_s = L$, Table~2. In the nucleation regime",
)
lt.rep(
    r"Normalized by their own numerical fracture resistance, the two linear elastic sets fall onto one curve (Fig.~14b), and the von~Mises set lies below it for all wall widths, i.e., relative to its own crack resistance in the homogeneous material, the ductile material is toughened less by the pores than the linear elastic material. The previous statement",
    r"Normalized by their own numerical fracture resistance, the two linear elastic sets fall onto one curve (Fig.~12b), and the von~Mises set lies below it for all wall widths, i.e., relative to its own crack resistance in the homogeneous material, the ductile material is toughened less by the pores than the linear elastic material. Section~4.2 now states explicitly which question each of the three comparisons of Fig.~12 answers (equal $J_c^0$, equal strength, equal crack resistance of the homogeneous material), that the toughness gain of the von~Mises set over \EqSigmaStarLabel in Fig.~12a stems from the material model, which already raises the crack resistance of the homogeneous material by $31\,\%$ relative to \EqJcLabel, and that Fig.~12b isolates the toughening by the microstructure (factor $2.03$ for the linear elastic materials, $1.34$ for the plastic material, below one for $w_s < 0.9L$). The previous statement",
)
lt.rep(
    r"the new Section~4.5 isolates the anisotropy of the embedding and Section~4.6 the initial crack and the size of the resolved region.",
    r"the new Section~4.5 isolates the anisotropy of the embedding, the initial crack and the size of the resolved region (Table~3).",
)
lt.rep(
    r"and the limitations are stated in Section~4.6.",
    r"and the limitations are stated in Section~4.5.",
)
lt.rep(
    "%@quote cmd:The two linear elastic parameter sets differ only in the value of $J_c$\n%@generated-begin\n\\item \\msloc{Section~4.2}\n\\replaced{The peak values",
    "%@quote cmd:The two linear elastic parameter sets differ only in the value of $J_c$\n%@quote cmd:The three comparisons in Fig.\n%@generated-begin\n\\item \\msloc{Section~4.2}\n\\replaced{The peak values",
)
lt.rep(
    r"\relhead{New Section~4.3 (energy partition), Fig.~15, Table~3}",
    r"\relhead{New Section~4.3 (energy partition), Fig.~13}",
)
lt.rep("%@quote cmd:Energy partition of the cycle that contains the ligament\n", "")
lt.rep(
    r"\relhead{New Section~4.4 (stress state and plastic zone before re-nucleation, sensitivity), Fig.~16, Tables~4 and 5}",
    r"\relhead{New Section~4.4 (stress state and plastic zone before re-nucleation, sensitivity), Fig.~14, Table~2}",
)
lt.rep("%@quote cmd:Effective crack resistance and state of the next intact ligament\n", "")

# ---------------------------------------------------------------- R1 major 4
lt.rep(
    r"reports the full effective stiffness of every unit cell in Sections~4.1 and 5.1 ($C_{11}$, $C_{12}$, $C_{33}$ resp.\ $C_{44}$, and the Zener ratio $A = 2C_{33}/(C_{11}-C_{12})$ resp.\ $2C_{44}/(C_{11}-C_{12})$, Tables~1 and 6)",
    r"reports the full effective stiffness of the unit cells in Sections~4.1 and 5.1 ($C_{11}$, $C_{12}$, $C_{33}$ resp.\ $C_{44}$, and the Zener ratio $A = 2C_{33}/(C_{11}-C_{12})$ resp.\ $2C_{44}/(C_{11}-C_{12})$, Tables~1 and 4; Table~1 lists five of the eleven two-dimensional wall widths, the anisotropy decreases monotonically in between, from $A = 0.870$ to $0.998$)",
)
lt.rep(
    r"The influence of the approximation on the result is tested directly in the new Section~4.5:",
    r"The influence of the approximation on the result is tested directly in the new Section~4.5 (Table~3):",
)
lt.rep(
    r"$0.75L$, $L$ and $2L$ (Table~6). Two earlier",
    r"$0.75L$, $L$ and $2L$ (Table~4). Two earlier",
)
lt.rep(
    r"\relhead{Table~1 (Section~4.1), new Section~4.5 (isotropic approximation of the embedding, corners)}",
    r"\relhead{Table~1 (Section~4.1), new Section~4.5 (isotropic approximation of the embedding, corners), Table~3}",
)
lt.rep("%@quote cmd:The corners of the resolved region, where the material discontinuity\n",
       "%@quote cmd:Change of the effective crack resistance of the parameter set\n")

# ---------------------------------------------------------------- minors 16, 24, 25
lt.rep(
    r"and $12$ to $37\,\%$ for the von~Mises set, Table~3).",
    r"and $12$ to $37\,\%$ for the von~Mises set, Fig.~13).",
)
lt.rep(
    r"which is compared with the one-dimensional values in the new Section~4.4 (Fig.~16).",
    r"which is compared with the one-dimensional values in the new Section~4.4 (Fig.~14).",
)
lt.rep(
    r"The result figures of Sections~3.6 and 4 were regenerated from the new simulations as vector graphics (pgf, Figs.~6, 7 and 9 to 16) with axis labels that start with an upper case letter. Fig.~9 (crack tip position vs.\ pseudo-time, mentioned by the reviewer) is now a vector graphic, and the J-integral records of Fig.~10 and Figs.~11 to 13 are plotted with the pore positions shaded resp.\ with the crack tip position measured from the center of pore two and normalized by the cell width, so that the curves for different wall widths can be compared directly.",
    r"The result figures of Sections~3.6 and 4 were regenerated from the new simulations as vector graphics (Figs.~6, 7 and 9 to 14) with axis labels that start with an upper case letter. Fig.~9 (crack tip position vs.\ pseudo-time, mentioned by the reviewer) is now a vector graphic. Fig.~10 shows the J-integral record for $w_s = L$ for all three parameter sets with the pore edges marked and the final crack pattern aligned above the plot, and the three separate figures of the previous version with the sections between pores two and four for the individual parameter sets (previous Figs.~11 to 13) were merged into the three panels of Fig.~11 with a common axis, so that the parameter sets can be compared directly.",
)
lt.rep(r"\relhead{Captions of Figs.~9 to 14}", r"\relhead{Captions of Figs.~9 to 12}")
lt.rep("%@quote cmd:and the three parameter sets. The shaded bands mark\n%@quote cmdall:measured from the center $x_{p2}$ of pore two\n",
       "%@quote cmd:and the three parameter sets. Dashed lines mark the pore edges\n%@quote cmd:J-Integral $J_x$ vs.\\ crack tip position $x_{ct}$ between pores two and four for varying $w_s$ and the parameter sets\n")

# ---------------------------------------------------------------- R3 C1, C3
lt.rep(
    r"It is evaluated per pore-to-pore cycle, Fig.~15 and Table~3.",
    r"It is evaluated per pore-to-pore cycle, Fig.~13 (numbers in the response to Reviewer~1, major point~3).",
)
lt.rep(
    r"The sensitivity study (Table~5) is consistent with this picture:",
    r"The sensitivity study (Table~2) is consistent with this picture:",
)
lt.rep(
    r"\relhead{New Sections~4.3 and 4.4, Figs.~15 and 16, Tables~3 to 5 (quoted under Reviewer~1, major point~3)}",
    r"\relhead{New Sections~4.3 and 4.4, Figs.~13 and 14, Table~2 (quoted under Reviewer~1, major point~3)}",
)
lt.rep(
    r"""(Fig.~14b and Table~2): the ratio of their effective crack resistances is $1.809$ to $1.810$ for all wall widths,
against $1/0.5523 = 1.811$ for the ratio of the prescribed $J_c$, and both reach $2.03\,J_c^{\mathrm{num}}$ at
$w_s = 4L$.""",
    r"""(Fig.~12b and the table below): the ratio of their effective crack resistances is $1.809$ to $1.810$ for all wall widths,
against $1/0.5523 = 1.811$ for the ratio of the prescribed $J_c$, and both reach $2.03\,J_c^{\mathrm{num}}$ at
$w_s = 4L$. The effective crack resistances of all wall widths are (in units of $J_c^0$ and of the numerical fracture
resistance $J_c^{\mathrm{num}} = 1.425$, $1.088$ and $0.604\,J_c^0$ of the respective set):
\begin{center}\small
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{3}{c}{$J_{x,c}^{\text{eff}}/J_c^0$} & \multicolumn{3}{c}{$J_{x,c}^{\text{eff}}/J_c^{\text{num}}$} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
$w_s/L$ & \vMLabel & \EqJcLabel & \EqSigmaStarLabel & \vMLabel & \EqJcLabel & \EqSigmaStarLabel \\
\midrule
0.25  & 0.514 & 1.219 & 0.674 & 0.361 & 1.121 & 1.116 \\
0.375 & 0.718 & 1.519 & 0.840 & 0.504 & 1.397 & 1.391 \\
0.4   & 0.759 & 1.568 & 0.866 & 0.532 & 1.441 & 1.435 \\
0.5   & 0.926 & 1.718 & 0.949 & 0.650 & 1.579 & 1.573 \\
0.6   & 1.084 & 1.825 & 1.009 & 0.761 & 1.678 & 1.671 \\
0.75  & 1.295 & 1.926 & 1.064 & 0.909 & 1.771 & 1.763 \\
1.0   & 1.563 & 2.020 & 1.116 & 1.097 & 1.858 & 1.849 \\
1.4   & 1.741 & 2.092 & 1.156 & 1.222 & 1.924 & 1.915 \\
2.0   & 1.830 & 2.143 & 1.184 & 1.284 & 1.971 & 1.961 \\
3.0   & 1.880 & 2.184 & 1.207 & 1.319 & 2.008 & 1.999 \\
4.0   & 1.903 & 2.212 & 1.222 & 1.336 & 2.034 & 2.025 \\
\bottomrule
\end{tabular}
\end{center}""",
)
lt.rep(
    r"""This strengthens the conclusion that the von~Mises material does not simply behave like a linear elastic material with
an increased crack resistance. The pores toughen it less, relative to its own crack resistance in the homogeneous
material, than they toughen a linear elastic material, because part of the work supplied before re-nucleation is spent
on plastic deformation of the ligament and does not enter the macroscopic driving force (see the response to Comment~1).
A paragraph with this argument was added at the end of Section~4.2.""",
    r"""This strengthens the conclusion that the von~Mises material does not simply behave like a linear elastic material with
an increased crack resistance. Section~4.2 now states which question each comparison of Fig.~12 answers: for equal
$J_c^0$ the von~Mises set lies below \EqJcLabel for all wall widths, for equal one-dimensional strength it lies above
\EqSigmaStarLabel from $w_s \approx 0.6L$ on (Fig.~12a), and for equal crack resistance of the homogeneous material,
which is the comparison asked for by the reviewer, it lies below the linear elastic curve for all wall widths
(Fig.~12b). No linear elastic phase-field model has both the strength and the $J_c^0$ of the von~Mises set, since
plasticity decouples the two. The toughness gain of the von~Mises set over \EqSigmaStarLabel in Fig.~12a stems from the
material model, which already raises the crack resistance of the homogeneous material ($J_c^{\mathrm{num}} = 1.425$
against $1.088\,J_c^0$ for \EqJcLabel with the same $J_c^0$, i.e., $+31\,\%$, and a factor $2.36$ against
\EqSigmaStarLabel). Fig.~12b isolates the toughening by the microstructure: the pore arrangement raises the crack
resistance of the linear elastic materials by a factor of about two relative to their homogeneous value, that of the
plastic material only by $1.34$, and below $w_s \approx 0.9L$ it even lowers it. The microstructure preserves only part
of the gain that plasticity gives to the homogeneous material. The mechanism (Sections~4.3 and 4.4, response to
Comment~1) is that crack trapping and forced re-nucleation at the blunt pore wall is an efficient toughening mechanism
for a linear elastic material, at the price of $47$ to $83\,\%$ of the dissipated energy going into the unstable
snap-through, whereas the homogeneous plastic material already blunts its crack by the plastic zone, which is contained
in $J_c^{\mathrm{num}}$, and in thin ligaments the whole ligament yields, so that the plastic work is not available as
crack resistance. Two paragraphs with this argument were added at the end of Section~4.2.""",
)
lt.rep(
    "%@quote cmd:The two linear elastic parameter sets differ only in the value of $J_c$\n%@generated-begin\n\\item \\msloc{Section~3.5}",
    "%@quote cmd:The two linear elastic parameter sets differ only in the value of $J_c$\n%@quote cmd:The three comparisons in Fig.\n%@generated-begin\n\\item \\msloc{Section~3.5}",
)

# ---------------------------------------------------------------- additional corrections
lt.rep(
    r"""The effective crack resistances of Section~4 are listed in
the new Table~2 for all wall widths. For $w_s \ge L$ they differ""",
    r"""The effective crack resistances of Section~4 are shown in
Fig.~12 for all wall widths and listed in the response to Reviewer~3, comment~3, of this letter. For $w_s \ge L$ they differ""",
)
lt.rep(
    r"stiffness are $A = 0.89,\ 0.96,\ 1.00$ (all values for $\lambda_0 = \mu_0$, complete table in Section~4.1).",
    r"stiffness are $A = 0.89,\ 0.96,\ 1.00$ (all values for $\lambda_0 = \mu_0$, Table~1 in Section~4.1).",
)
lt.rep("%@quote cmd:The shaded bands indicate the pore positions.\n",
       "%@quote cmd:and the three parameter sets. Dashed lines mark the pore edges\n")
lt.rep("%@quote cmd:Effective crack resistance $J_{x,c}^{\\text{eff}}$ of the two-dimensional porous material in units of\n", "")

lt.rep(r"\mycomment{TODO}{Section~5: state the Zener ratios in Table~6 and refer to Section~4.5", r"\mycomment{TODO}{Section~5: state the Zener ratios in Table~4 and refer to Section~4.5")
lt.rep(r"the figures of Section~5 (Figs.~17 to 22) follow", r"the figures of Section~5 (Figs.~15 to 20) follow")
for s in ("Table~5", "Table~6", "Fig.~15", "Fig.~16", "Figs.~15 and 16", "Section~4.6", "Tables~3 to 5", "Tables~4 and 5"):
    hand = "\n".join(l for l in lt.t.split("\n") if not l.startswith(("\\added", "\\replaced", "\\deleted", "\\item \\msloc")))
    if s in hand:
        raise SystemExit("stale reference left in the hand-written letter text: " + s)
lt.save()
print("part 6 applied")
