#!/usr/bin/env python3
"""Round-2 edit script, part 6b (2026-10-22): second tightening pass on the new prose of Section 4
(after edits_round2_part6_condense.py), figure widths reduced so that the floats pack better.
One-shot: fails on the first anchor that is gone. Letter anchors (%@quote) are preserved.
"""
HERE = "/Users/alexanderschluter/Work/Hypo/Hypo/Simulation/dolfinx_alex/shared/scripts/046_plasticity/KLAIM_2025/revision_2/"
MS = HERE + "manuscript_revised_template.tex"
J = "fig:PAPER_06b_Jx_vs_wsteg_three_datasets"


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
        lines = self.t.split("\n")
        hits = [i for i, l in enumerate(lines) if l.startswith(prefix)]
        if len(hits) != 1:
            raise SystemExit("%s: line prefix found %d times:\n%s" % (self.path, len(hits), prefix[:120]))
        lines[hits[0]] = new_line
        self.t = "\n".join(lines)

    def save(self):
        open(self.path, "w", encoding="utf-8").write(self.t)


ms = Doc(MS)

# ---------------------------------------------------------------- figure widths
ms.rep(r"\includegraphics[width=0.85\textwidth]{pics/PAPER_rev2_Jx_vs_xct_ws1.0_three_sets.pdf}",
       r"\includegraphics[width=0.72\textwidth]{pics/PAPER_rev2_Jx_vs_xct_ws1.0_three_sets.pdf}")
ms.rep(r"\resizebox{0.5\textwidth}{!}{\input{pics/PAPER_rev2_xct_xbc_vs_t_vM_ws1.0.pgf}}",
       r"\resizebox{0.45\textwidth}{!}{\input{pics/PAPER_rev2_xct_xbc_vs_t_vM_ws1.0.pgf}}")
ms.rep(r"\resizebox{\textwidth}{!}{\input{pics/PAPER_rev2_Jeff_vs_wsteg_three_sets.pgf}}",
       r"\resizebox{0.9\textwidth}{!}{\input{pics/PAPER_rev2_Jeff_vs_wsteg_three_sets.pgf}}")
ms.rep(r"\resizebox{\textwidth}{!}{\input{pics/PAPER_rev2_energy_partition_cycle.pgf}}",
       r"\resizebox{0.9\textwidth}{!}{\input{pics/PAPER_rev2_energy_partition_cycle.pgf}}")
ms.rep(r"\resizebox{\textwidth}{!}{\input{pics/PAPER_rev2_ligament_state_before_renucleation.pgf}}",
       r"\resizebox{0.9\textwidth}{!}{\input{pics/PAPER_rev2_ligament_state_before_renucleation.pgf}}")

# ---------------------------------------------------------------- 4.1
ms.rep(
    r"Table~\ref{tab:effective_stiffness_2D} lists the effective constants and the Zener ratio for five of the eleven wall widths considered, the values of the other wall widths lie monotonically in between. The anisotropy decreases monotonically with the wall width, from $A = 0.870$ for $w_s = 0.25L$ to $0.998$ for $w_s = 4L$. The effective Poisson's ratio of the isotropic approximation increases from $\nu_0^{\text{eff}} = 0.19$ for $w_s = 0.25L$ to $0.25$ for $w_s = 4L$. The influence of the isotropic approximation on the effective crack resistance is tested in Section~\ref{sec:2D_cubic_embedding} by simulations in which the embedding is assigned the full square-symmetric stiffness.}",
    r"Table~\ref{tab:effective_stiffness_2D} lists the effective constants and the Zener ratio for five of the eleven wall widths, the other values lie monotonically in between. The anisotropy decreases with the wall width, from $A = 0.870$ for $w_s = 0.25L$ to $0.998$ for $w_s = 4L$, and the effective Poisson's ratio of the isotropic approximation increases from $\nu_0^{\text{eff}} = 0.19$ to $0.25$. The influence of the isotropic approximation on the effective crack resistance is tested in Section~\ref{sec:2D_cubic_embedding}.}",
)

# ---------------------------------------------------------------- 4.2
ms.rep(
    r"\replaced{. It is trapped at each pore until re-nucleation and then crosses the pore and the following ligament in one jump, i.e., the ligament between two pores fails during the instantaneous crack growth phase that follows nucleation.}",
    r"\replaced{. It is trapped at each pore until re-nucleation and then crosses the pore and the following ligament in one jump.}",
)
ms.rep(
    r"\replaced{has the same ordering as $J_{x,c}^{\text{eff}}$ at this wall width, $\Pi^* = 2.19$, $2.12$ and $1.23\,J_c^0L$ for \EqJcLabel, \vMLabel and \EqSigmaStarLabel, i.e., the effective crack resistance of \vMLabel lies between those of \EqJcLabel and \EqSigmaStarLabel, see also Fig.~\ref{fig:energy_vs_displacement}. This correspondence is limited to the regime of wide ligaments, as shown below.}",
    r"\replaced{has the same ordering as $J_{x,c}^{\text{eff}}$ at this wall width, $\Pi^* = 2.19$, $2.12$ and $1.23\,J_c^0L$ for \EqJcLabel, \vMLabel and \EqSigmaStarLabel, see also Fig.~\ref{fig:energy_vs_displacement}. This correspondence is limited to wide ligaments, as shown below.}",
)
ms.rep_line(
    r"\added{The two linear elastic parameter sets differ only in the value of $J_c$.",
    r"\added{The two linear elastic parameter sets differ only in the value of $J_c$. For the linear elastic phase-field model, $J_c$ is the only energy scale once the elastic constants and the lengths are fixed, and the amplitude of the K-field is proportional to $\sqrt{J_c}$ by the choice of $K_I^{\text{bc}}$. In the rate-independent limit $M \to \infty$, all energies, including $J_{x,c}^{\text{eff}}$, therefore scale with $J_c$, while the crack pattern is unchanged. The simulations confirm this: the ratio of the effective crack resistances of \EqJcLabel and \EqSigmaStarLabel lies between $1.809$ and $1.810$ for all wall widths, compared with $1/0.5523 = 1.811$ for the prescribed fracture resistances, so that the two sets fall onto one curve in Fig.~\ref{@J@}b).}".replace("@J@", J),
)
ms.rep_line(
    r"\added{The three comparisons in Fig.",
    r"\added{The three comparisons in Fig.~\ref{@J@} answer different questions. For equal microscopic fracture resistance $J_c^0$, \vMLabel lies below \EqJcLabel for all wall widths. For equal one-dimensional strength, \vMLabel lies above \EqSigmaStarLabel from $w_s \approx 0.6L$ on, Fig.~\ref{@J@}a). For equal crack resistance of the homogeneous material, Fig.~\ref{@J@}b), \vMLabel lies below the linear elastic curve for all wall widths. No linear elastic phase-field model has both the strength and the $J_c^0$ of \vMLabel, since plasticity decouples the two. The gain of \vMLabel over \EqSigmaStarLabel in Fig.~\ref{@J@}a) stems from the material model, which already raises the crack resistance of the homogeneous material, $J_c^{\text{num}} = 1.425J_c^0$ against $1.088J_c^0$ for \EqJcLabel with the same $J_c^0$ ($+31\,\%$) and $0.604J_c^0$ for \EqSigmaStarLabel (factor $2.36$). Fig.~\ref{@J@}b) isolates the toughening by the microstructure. The pores raise the crack resistance of the linear elastic materials by a factor of up to $2.03$ relative to the homogeneous value, that of the plastic material only by $1.34$, and below $w_s \approx 0.9L$ they even lower it, $J_{x,c}^{\text{eff}} < J_c^{\text{num}}$. A linear elastic set with $J_c^{\text{num}} = 1.425J_c^0$ would reach $2.03 \times 1.425J_c^0 = 2.89J_c^0$ at $w_s = 4L$ instead of the $1.90J_c^0$ of \vMLabel. The microstructure thus preserves only part of the gain that plasticity gives to the homogeneous material. The reason, Sections~\ref{sec:2D_energy_partition} and~\ref{sec:2D_mechanisms}, is that crack trapping and forced re-nucleation at the blunt pore wall toughens a linear elastic material efficiently, at the price of $47$ to $83\,\%$ of the dissipated energy going into the unstable snap-through of the ligament, whereas the homogeneous plastic material already blunts its crack by the plastic zone, which is contained in $J_c^{\text{num}}$, and in thin ligaments the whole ligament yields, so that the plastic work is not available as crack resistance.}".replace("@J@", J),
)

# ---------------------------------------------------------------- 4.3
ms.rep_line(
    r"\added{where $E_{\text{el}}$, $E_{\text{pl}}$ and $E_{\text{fr}}$ denote",
    r"\added{where $E_{\text{el}}$, $E_{\text{pl}}$ and $E_{\text{fr}}$ denote the integrals of $\psi_{\text{el}}$, $\psi_{\text{pl}}$ and $\psi_{\text{fr}}$ over $\Omega$, i.e., the three contributions to $E_{\text{ep}}$ in eq.~(\ref{eq:E_ep_total}), and $D_{\text{visc}}$ is the energy dissipated by the viscous evolution of the phase field. The plastic work is stored in the degraded form $\psi_{\text{pl}} = g(s)(\nicefrac{1}{2}H\alpha^2 + \sigma_y\alpha)$, eq.~(\ref{eq:psi_pl}), and acts as a crack driving force through $g'(s)$ in eq.~(\ref{eq:evolution_s}). Plastic work done in material that fails later is released by the degradation and reappears in $E_{\text{fr}}$ and $D_{\text{visc}}$, whereas $E_{\text{pl}}$ records the plastic energy stored in intact material. The viscous term collects the energy released in the unstable failure of a ligament, which would become kinetic energy in a dynamic formulation. For the linear elastic model, $E_{\text{pl}} = 0$. All terms are recorded in every time step.}",
)
ms.rep_line(
    r"\added{and $G_{\text{diss}} = \Delta D_{\text{tot}}/w_c$ is the energy dissipated per unit crack advance, averaged over the cycle.",
    r"\added{and $G_{\text{diss}} = \Delta D_{\text{tot}}/w_c$ is the energy dissipated per unit crack advance, averaged over the cycle. Figure~\ref{fig:energy_partition_cycle} shows the three fractions of $\Delta D_{\text{tot}}$ and the ratio $G_{\text{diss}}/J_{x,c}^{\text{eff}}$ for the cycle that contains the ligament in which $J_{x,c}^{\text{eff}}$ is attained. The energy dissipated per cycle is nearly constant from cycle to cycle, see Section~\ref{sec:2D_representativity}. The stored elastic energy decreases over a cycle, by $0.05$ to $0.6\,\Delta D_{\text{tot}}$, because the K-field applied at $t = 0$ stores elastic energy that is drawn down as the crack advances, so that the specimen is not in a strict steady state.}",
)
ms.rep_line(
    r"\added{For the two linear elastic sets, which give the same fractions to two digits, the viscous share is large.",
    r"\added{For the two linear elastic sets, which give the same fractions to two digits, the viscous share is large. Between $83\,\%$ of the dissipated energy for $w_s = 0.25L$ and $47\,\%$ for $w_s = 4L$ are lost in the unstable snap-through of the ligament, and only $17$ to $53\,\%$ create crack surface, since the elastic energy stored up to re-nucleation exceeds the fracture energy of the ligament by far. For the von~Mises set, the viscous share is $12$ to $37\,\%$, the fracture share $49$ to $74\,\%$, and $12$ to $16\,\%$ remain stored as plastic energy in the intact material along the crack path. The difference is largest for thin ligaments. $G_{\text{diss}}$ shows the same crossover as $J_{x,c}^{\text{eff}}$: $0.48$ vs.\ $0.71\,J_c^0$ for \vMLabel and \EqSigmaStarLabel at $w_s = 0.25L$, equal values at $w_s = 0.6L$, and $1.39$ vs.\ $0.89\,J_c^0$ at $w_s = 4L$. The ratio $G_{\text{diss}}/J_{x,c}^{\text{eff}}$, Fig.~\ref{fig:energy_partition_cycle}b), decreases from $0.93$ to $1.10$ at $w_s = 0.25L$ to $0.73$ at $w_s = 4L$, since $J_{x,c}^{\text{eff}}$ is the peak of $J_x$ in the cycle and $G_{\text{diss}}$ its average. For thin walls, the ductile material law thus reduces not only the peak driving force but also the energy dissipated per unit crack advance, which is the quantity closest to an effective fracture energy in the sense of the energy dissipated along the crack path, see Michel and Suquet~\cite{michel_merits_2022}. The mechanism is identified in the next section.}",
)

# ---------------------------------------------------------------- 4.4
ms.rep_line(
    r"\added{To relate the re-nucleation in the porous specimen to the one-dimensional results",
    r"\added{To relate the re-nucleation in the porous specimen to the one-dimensional results of Section~\ref{sec:stress_driven_nucleation}, the maxima of the normal stress $\sigma_{yy}$, the von~Mises stress $\sigma_{\text{vM}}$, the hydrostatic stress $\sigma_h = \nicefrac{1}{3}\,\text{tr}(\boldsymbol{\sigma})$ and the plastic variable $\alpha$ in the next intact ligament ahead of the trapped crack tip are recorded in every time step. Figure~\ref{fig:ligament_state} shows these values at the peak of $J_x$, i.e., immediately before re-nucleation, with $\sigma_{yy,\max}$ normalized by the one-dimensional fracture stress $\sigma^*_{\text{1D}}$ of the respective parameter set, $1.28\sigma_y$ for \vMLabel and \EqSigmaStarLabel and $1.72\sigma_y$ for \EqJcLabel, $\sigma_y = \sqrt{\nicefrac{J_c^0\mu_0}{L}}$, and the stress triaxiality $\sigma_h/\sigma_{\text{vM}}$ at the location of $\sigma_{yy,\max}$.}",
)
ms.rep(
    r"The energy dissipated per unit crack advance is reduced as well, Section~\ref{sec:2D_energy_partition}, because the linear elastic ligament stores more elastic energy up to failure, which is released in the snap-through.}",
    r"The energy dissipated per unit crack advance is reduced as well, Section~\ref{sec:2D_energy_partition}, since the linear elastic ligament stores more elastic energy up to failure.}",
)
ms.rep_line(
    r"\added{For wide ligaments, $w_s \ge L$, the plastic zone does not span the ligament.",
    r"\added{For wide ligaments, $w_s \ge L$, the plastic zone does not span the ligament, the maximum of $\alpha$ drops to $0.44$ to $0.60$, and the trapped crack tip at the pore surface acts as a notch constrained by the surrounding elastic material. The triaxiality rises to $1.0$ to $1.1$, the von~Mises stress at the location of $\sigma_{yy,\max}$ remains at the yield stress, $0.98$ to $1.06\,\sigma_y$, and the normal stress reaches $\sigma_{yy,\max} = 1.67$ to $1.71\sqrt{\nicefrac{J_c^0\mu_0}{L}} = 1.31$ to $1.34\,\sigma^*_{\text{vM}}$, the level of the one-dimensional fracture stress of \EqJcLabel. Since the hydrostatic stress does not enter the yield condition, the constrained ligament carries a normal stress well above the one-dimensional fracture stress before the driving force suffices for re-nucleation, and the plastic work ahead of the trapped tip has to be supplied before the crack advances. In this propagation regime, the ductile material law raises the effective crack resistance above that of \EqSigmaStarLabel, and the interaction length between neighboring pores is set by the size of the plastic zone rather than by the pore diameter, consistent with the later saturation of the von~Mises set in Fig.~\ref{@J@}.}".replace("@J@", J),
)
ms.rep(
    r"For the linear elastic sets, the corresponding sensitivity is the scaling with $J_c$ of Section~\ref{sec:2D_fracture_behavior}, which the simulations confirm to $0.1\,\%$.}",
    r"For the linear elastic sets, the corresponding sensitivity is the scaling with $J_c$ of Section~\ref{sec:2D_fracture_behavior}.}",
)

# ---------------------------------------------------------------- 4.5
ms.rep_line(
    r"\added{The embedding of the resolved porous region is assigned the isotropic approximation",
    r"\added{The embedding of the resolved porous region is assigned the isotropic approximation, eq.~(\ref{eq:isotropic_approximation_2D}), of the square-symmetric stiffness of Table~\ref{tab:effective_stiffness_2D}. To test this approximation, the simulations for $w_s = 0.375L$, $L$ and $4L$ with the parameter set \vMLabel were repeated with the full stiffness $C_{11}$, $C_{12}$, $C_{33}$ assigned to the embedding. The effective crack resistance changes by at most $+0.18\,\%$, Table~\ref{tab:robustness_2D}, even for the thinnest wall, for which the anisotropy is largest, since the embedding only transmits the K-field and the J-integral is evaluated on its outer boundary, whereas the re-nucleation that determines $J_{x,c}^{\text{eff}}$ takes place in the resolved ligaments. The three-dimensional unit cells of Section~\ref{sec:3D_effective_medium} are closer to isotropic, $A \ge 0.978$ for $w_s \ge 0.375L$, and two earlier three-dimensional runs on a coarser mesh with the full cubic stiffness changed $J_{x,c}^{\text{eff}}$ by $-0.37\,\%$ ($w_s = 0.375L$) and $-0.22\,\%$ ($w_s = L$). The corners of the resolved region act as stress concentrators, see Section~\ref{sec:numericalExperiment}. In none of the simulations did a crack nucleate at a corner, the crack pattern remained confined to the central row of pores, Fig.~\ref{fig:setup_crack_growth}b), and since the embedding is linear elastic and the J-integral is evaluated on its outer boundary, the stress concentration at the corners does not enter the recorded driving force.}",
)
ms.rep(
    r"The first cycle, during which the K-field builds up, is lower, the last cycle is affected by the end of the resolved region, and the inner cycles agree within about $7\,\%$ ($10\,\%$ for \EqJcLabel at $w_s = 0.25L$), e.g., $G_{\text{diss}} = 1.30$, $1.36$, $1.39$ and $1.37\,J_c^0$ in the first four cycles for \vMLabel and $w_s = L$.}",
    r"The first cycle, during which the K-field builds up, is lower, the last cycle is affected by the end of the resolved region, and the inner cycles agree within about $7\,\%$ ($10\,\%$ for \EqJcLabel at $w_s = 0.25L$).}",
)
ms.save()
print("part 6b applied")
