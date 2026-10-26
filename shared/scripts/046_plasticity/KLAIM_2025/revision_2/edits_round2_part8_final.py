#!/usr/bin/env python3
"""Round-2 edit script, part 8 (2026-10-23, final writing pass after part 7b).

Section 3.5 (homogeneous reference = 16L x 20L box without pores, Fig. 3 regenerated as vector PDF from the S2
reference runs, captions of Figs. 2 and 4 qualified), abstract and conclusion rewritten from the results of the revised
Sections 4 and 5, consistency sweep (keyword), letter (minor 25, editor response, Additional corrections items 12/13,
quote anchors), cover letter to the editor (four TODOs filled). One-shot: fails on the first anchor that is gone.
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
            raise SystemExit("%s: anchor found %d times (expected %d):\n%s" % (self.path, c, n, old[:160]))
        self.t = self.t.replace(old, new)

    def save(self):
        open(self.path, "w", encoding="utf-8").write(self.t)


ms = Doc(MS)
lt = Doc(LT)
ed = Doc(ED)

# ============================================================================ manuscript
# ---------------------------------------------------------------- abstract (<= 200 words, template guide)
OLD_ABSTRACT = (
    "      In this work, we investigate the simulation-based determination of the effective crack resistance of porous materials while accounting for nonlinear, ductile material behavior.\n"
    "Following the approach of Hossain et al.~\\cite{hossain_effective_2014}, numerical experiments\n"
    "with surfing boundary conditions are conducted to identify the effective crack resistance. Crack evolution in the microstructure is modeled using a\n"
    "phase-field formulation without prescribing crack paths or growth continuity. The maximum\n"
    "value of the macroscopically acting J-integral defines the driving force required for sustained\n"
    "macroscopic crack advance and is taken as the effective crack resistance. In contrast to\n"
    "previous studies, a von~Mises plasticity model is employed to investigate ductile fracture in\n"
    "porous metals. The results show that the effective crack resistance is governed primarily\n"
    "by the energy dissipation capacity of the material during crack renucleation at pores rather\n"
    "than by tensile strength or microscopic fracture energy. A complementary three-dimensional\n"
    "model confirms these findings, showing similar saturation behavior of the effective crack resistance with respect to microstructural wall width and good quantitative agreement with\n"
    "the two-dimensional predictions.\n"
)
NEW_ABSTRACT = (
    "\\replaced{The effective crack resistance of a simplified porous material with ductile bulk behavior is determined in a numerical experiment. "
    "Following Hossain et al.~\\cite{hossain_effective_2014}, a resolved periodic arrangement of pores is subjected to surfing boundary conditions, the crack evolution is computed "
    "with a phase-field model without a prescribed crack path, and the maximum of the J-integral over one pore-to-pore cycle is taken as the effective crack resistance. "
    "A von~Mises plasticity model is compared with two linear elastic models with the same fracture energy or the same one-dimensional strength. "
    "In two dimensions, the effective crack resistance increases with the wall width and saturates. Relative to the homogeneous material, pores toughen the linear elastic materials "
    "by up to a factor of two, the plastic material by at most 1.34, and weaken it for thin walls. The energy balance and the ligament stress state before re-nucleation "
    "identify two regimes: thin ligaments yield across their width and re-nucleate at the one-dimensional strength, wide ligaments re-nucleate at a constrained notch at a "
    "higher stress. In three dimensions, with spherical pores, the crack front bypasses the pores without unstable ligament failure, and the effective crack resistance stays "
    "below the homogeneous value for all wall widths.}"
    "{In this work, we investigate the simulation-based determination of the effective crack resistance of porous materials while accounting for nonlinear, ductile material behavior. "
    "Following the approach of Hossain et al.~\\cite{hossain_effective_2014}, numerical experiments "
    "with surfing boundary conditions are conducted to identify the effective crack resistance. Crack evolution in the microstructure is modeled using a "
    "phase-field formulation without prescribing crack paths or growth continuity. The maximum "
    "value of the macroscopically acting J-integral defines the driving force required for sustained "
    "macroscopic crack advance and is taken as the effective crack resistance. In contrast to "
    "previous studies, a von~Mises plasticity model is employed to investigate ductile fracture in "
    "porous metals. The results show that the effective crack resistance is governed primarily "
    "by the energy dissipation capacity of the material during crack renucleation at pores rather "
    "than by tensile strength or microscopic fracture energy. A complementary three-dimensional "
    "model confirms these findings, showing similar saturation behavior of the effective crack resistance with respect to microstructural wall width and good quantitative agreement with "
    "the two-dimensional predictions.}\n"
)
ms.rep(OLD_ABSTRACT, NEW_ABSTRACT)

# ---------------------------------------------------------------- keywords (terminology: "fracture toughness" is reserved for the material property)
ms.rep(
    r"\keywords{Ductile Fracture, Porous Metals, Effective Fracture Toughness, Phase Field Model of Ductile Fracture, Numerical Experiment}",
    r"\keywords{Ductile Fracture, Porous Metals, \replaced{Effective Crack Resistance}{Effective Fracture Toughness}, Phase Field Model of Ductile Fracture, Numerical Experiment}",
)

# ---------------------------------------------------------------- Section 3.5: figures
ms.rep(
    r"""        Deformed contour plot of the phase field $s$ in a strip of size $16L \times 2L$ for the simulation \EqJcLabel. The deformation is scaled by a factor of 0.3. """,
    r"""        \replaced{Deformed contour plot of the phase field $s$ for the parameter set \EqJcLabel, from an earlier simulation in a homogeneous strip of size $16L \times 2L$ with the same surfing boundary conditions, shown to illustrate the enforced crack path. The deformation is scaled by a factor of 0.3.}{Deformed contour plot of the phase field $s$ in a strip of size $16L \times 2L$ for the simulation \EqJcLabel. The deformation is scaled by a factor of 0.3.}""",
)
ms.rep(
    r"""    \includegraphics[width=0.8\textwidth]{pics/PAPER_01_all_Jx_vs_xct_pf__vM.png}
    \caption{%
        J-Integral versus crack tip position $x_{ct}$ measured during a simulation of crack propagation in a homogeneous material strip of size $16L \times 2L$.
    }""",
    r"""    \includegraphics[width=0.7\textwidth]{pics/PAPER_rev2_Jx_vs_xct_homogeneous_references.pdf}% rev2: regenerated from 075 (S2_solid_vM, S2_strip_EqJc, S2_strip_EqSig), replaces PAPER_01_all_Jx_vs_xct_pf__vM.png (which contained a synthetic continuation of the final descent of the vM curve)
    \caption{%
        \replaced{J-Integral $J_x$ vs.\ crack tip position $x_{ct}$ for the three parameter sets during crack propagation through the homogeneous specimen of size $16L \times 20L$ without pores. The shaded band marks the central part of the crack path over which the numerical fracture resistance $J_c^{\text{num}}$ is averaged.}{J-Integral versus crack tip position $x_{ct}$ measured during a simulation of crack propagation in a homogeneous material strip of size $16L \times 2L$.}
    }""",
)
# (the caption line ends with a trailing blank, hence two anchors)
ms.rep(
    r"""        Plastic variable $\alpha$ at $t=6.25\nicefrac{L}{\dot{x}_{\rm bc}}$ for the simulation with the""",
    r"""        \replaced{Plastic variable $\alpha$ for the parameter set \vMLabel at $t = 6.25\nicefrac{L}{\dot{x}_{\rm bc}}$, from an earlier simulation in a homogeneous strip of size $16L \times 2L$ with the same surfing boundary conditions. Plastic deformation stays confined to the crack path.}{Plastic variable $\alpha$ at $t=6.25\nicefrac{L}{\dot{x}_{\rm bc}}$ for the simulation with the""",
)
ms.rep(
    "        \\vMLabel data set.\n    }\n    \\label{fig:plasticity_vm_jeff}",
    "        \\vMLabel data set.}\n    }\n    \\label{fig:plasticity_vm_jeff}",
)

ms.rep(r"\resizebox{0.8\textwidth}{!}{\includegraphics[]{pics/griffith_deformed.png}}",
       r"\resizebox{0.7\textwidth}{!}{\includegraphics[]{pics/griffith_deformed.png}}")
ms.rep(r"\resizebox{0.95\textwidth}{!}{\includegraphics{pics/alpha_t_6.2495.png}}",
       r"\resizebox{0.85\textwidth}{!}{\includegraphics{pics/alpha_t_6.2495.png}}")

# ---------------------------------------------------------------- Section 3.5: text
ms.rep(
    r"The computational domain consisted of a $16L \times 2L$ strip of homogeneous material under plane strain assumptions with the following parameters: $\lambda_0 = \mu_0$, $M = 1000.0 \nicefrac{\dot{x}_{\text{bc}}}{J_c^0}$, $\epsilon = 0.1L$, and element edge length $\Delta{h} = 0.02L$. The boundary condition parameters were $\nu_{\text{bc}} = 0.25$, $E_{\text{bc}} = 2.5\mu_{0}$, and $K_I^{\text{bc}} = \sqrt{J_c^0 \cdot 2.5\mu_{0}}$, which enforce crack propagation in the $x$-direction, as shown in Fig.~\ref{fig:griffith_deformed}.",
    r"\replaced{The computational domain is the simulation domain of the porous specimens of Section~\ref{sec:2Dresults} for the wall width $w_s = L$, i.e., a rectangle of size $(8w_c) \times 20L = 16L \times 20L$, but without pores and with an initial crack of length $2L$. It is meshed by the same rule as the porous specimens, with an element edge length of $\Delta h = 0.02L$ along the crack path and a coarser mesh in the far field, under plane strain assumptions with $\lambda_0 = \mu_0$, $M = 1000.0 \nicefrac{\dot{x}_{\text{bc}}}{J_c^0}$ and $\epsilon = 0.1L$. For the von~Mises set, the yield stress is assigned to the whole domain, so that the specimen is homogeneous for all three parameter sets.}{The computational domain consisted of a $16L \times 2L$ strip of homogeneous material under plane strain assumptions with the following parameters: $\lambda_0 = \mu_0$, $M = 1000.0 \nicefrac{\dot{x}_{\text{bc}}}{J_c^0}$, $\epsilon = 0.1L$, and element edge length $\Delta{h} = 0.02L$.} The boundary condition parameters were $\nu_{\text{bc}} = 0.25$, $E_{\text{bc}} = 2.5\mu_{0}$, and $K_I^{\text{bc}} = \sqrt{J_c^0 \cdot 2.5\mu_{0}}$\added{, i.e., the energy release rate of the applied K-field is $J_{\text{bc}} = 0.9375J_c^0$}, which enforce crack propagation in the $x$-direction, as shown in Fig.~\ref{fig:griffith_deformed}.",
)
ms.rep(
    r"The results show that the J-Integral remains nearly constant throughout the crack propagation process, as illustrated in Fig.~\ref{fig:PAPER_01_all_Jx_vs_xct_pf_2}.",
    r"\replaced{The J-Integral rises during a short transient at the start, remains nearly constant over the largest part of the crack path and drops when the crack approaches the end of the domain, as shown in Fig.~\ref{fig:PAPER_01_all_Jx_vs_xct_pf_2}.}{The results show that the J-Integral remains nearly constant throughout the crack propagation process, as illustrated in Fig.~\ref{fig:PAPER_01_all_Jx_vs_xct_pf_2}.}",
)
ms.rep(
    r"In the current setup, the following \replaced{values are obtained from the plateau of $J_x$}{approximate values were observed}:",
    r"In the current setup, the following \replaced{values are obtained from the plateau of $J_x$, evaluated as the mean over the central part of the crack path between $30$ and $70\,\%$ of the final position of the K-field, shaded in Fig.~\ref{fig:PAPER_01_all_Jx_vs_xct_pf_2}}{approximate values were observed}:",
)
ms.rep(
    "\\mycomment{TODO}{Fig.~\\ref{fig:PAPER_01_all_Jx_vs_xct_pf_2} to be regenerated from the S2 runs (S2\\_solid\\_vM, S2\\_strip\\_EqJc, S2\\_strip\\_EqSig); geometry description of the solid reference in Section~3.5 (box without pores) to be updated.}\n",
    "",
)

# ---------------------------------------------------------------- conclusion
ms.rep(
    "\\section{Conclusion and Outlook}\n\\mycomment{TODO}{Abstract and conclusion to be aligned with Sections~\\ref{sec:2D_fracture_behavior}--\\ref{sec:2D_representativity}: two regimes (nucleation regime $w_s \\le 0.5L$ with reduced crack resistance, propagation regime), toughening relative to the own $J_c^{\\text{num}}$ smaller for \\vMLabel than for the linear elastic sets, energy partition, ``predicted'' wording; and with the Section~5 results: in 3D the pores never raise $J_{x,c}^{\\text{eff}}$ above $J_c^{\\text{num}}$ ($0.87$ to $0.96$, nearly flat, saturation at $0.96$ against $1.34$ in 2D, 2D and 3D curves cross at $w_s \\approx 0.75L$), no re-nucleation peaks, the front bypasses the pores, $92$ to $94\\,\\%$ of the dissipated energy creates crack surface, $G_{\\text{diss}} \\approx J_{x,c}^{\\text{eff}}$. The statements ``good quantitative agreement between the two- and three-dimensional predictions'' (abstract, conclusion) and ``the pores still induce phases of stress-driven instantaneous crack growth'' (conclusion) no longer hold.}\n",
    "\\section{Conclusion and Outlook}\n",
)
OLD_P2 = (
    "As in previous work, the renucleation of crack tips that are trapped in pores is the determining\n"
    "process governing the effective crack resistance of the simplified porous material considered in these numerical experiments. Our results indicate that the\n"
    "energy dissipation capacity of the material law during this renucleation process plays a more\n"
    "critical role for the effective crack resistance than the tensile strength or the microscopic\n"
    "energetic fracture resistance. Material models that allow for greater energy dissipation during\n"
    "renucleation result in higher effective crack resistance, underscoring the importance of considering\n"
    "nonlinear constitutive effects when designing porous or additively manufactured materials.\n"
    "The comparison shows that matching this stress level alone does not reproduce the crack resistance obtained with the ductile material law, because the additional energy dissipated during plastic deformation and crack re-nucleation is decisive.\n"
)
OLD_P3 = (
    "The observed saturation behavior of the maximum J-integral with increasing wall width further\n"
    "highlights a scale-dependent toughening effect inherent to the microstructure. The saturation is\n"
    "observed at larger wall widths for the plastic material law than in the linear elastic case,\n"
    "highlighting the fact that ductile material laws enlarge the length scale at which neighboring pores\n"
    "can influence the renucleation process.\n"
)
OLD_P4 = (
    "A more realistic but still idealized three-dimensional model of porous metals showed very similar\n"
    "results. Although crack fronts are not fully trapped in pores in the three-dimensional case, the pores still induce\n"
    "phases of stress-driven instantaneous crack growth and a corresponding peak in the J-integral that\n"
    "defines the effective crack resistance. Qualitatively, the same saturation behavior of the\n"
    "effective crack resistance with respect to the wall width is observable in the three-dimensional case. However,\n"
    "the dependence on the wall width is less pronounced. In addition, a good quantitative agreement\n"
    "between the two- and three-dimensional predictions for the effective crack resistance is observed.\n"
)
OLD_P5 = (
    "Looking forward, several promising research directions arise from this study. \n"
    "First, more complex\n"
    "and statistically representative microstructures—such \n"
    "as those derived from real micro-CT scans of\n"
    "porous or additively manufactured components—should \n"
    "be considered in future work. Second, such studies would lend themselves to experimental\n"
    "verification in which the numerically determined effective crack resistance is compared to \n"
    "experimentally determined material parameters. Lastly, the influence of the \n"
    "different parameters of the employed material model such as the initial yield stress and the hardening\n"
    "parameter should be investigated.\n"
    "A further essential step is to investigate representativity more systematically, including variations of the initial crack position, crack-growth direction, microstructure realization, and domain size. Such studies are required before the operational effective crack resistance used here can be assessed as a material property in a stronger homogenization sense.\n"
    "A more detailed separation of the individual mechanisms, in particular stress-driven crack nucleation under plasticity, plastic dissipation during ligament failure, and subsequent crack growth between pores, should also be addressed in future work.\n"
)
OLD_P6 = (
    "Overall, this work provides a robust and extensible framework for assessing the fracture resistance\n"
    "of complex materials and opens the path for integrated computational design of toughened porous and\n"
    "heterogeneous structures.\n"
)
NEW_P2 = (
    "\\replaced{In the two-dimensional model with cylindrical pores, the effective crack resistance increases with the wall width between the pores and saturates for all three parameter sets. "
    "The von~Mises set lies below the linear elastic set with the same microscopic fracture resistance $J_c^0$ for all wall widths and above the linear elastic set with the same one-dimensional strength only from a wall width of about $0.6L$ on. "
    "Normalized by the crack resistance of the homogeneous material of the same set, the pores raise the crack resistance of the linear elastic materials by up to a factor of $2.03$, that of the plastic material only by $1.34$, and for wall widths below about $0.9L$ they lower it. "
    "The gain of the plastic material over the linear elastic material of equal strength thus stems from the material model, which raises the crack resistance of the homogeneous material by $31\\,\\%$ at equal $J_c^0$, and the microstructure preserves only part of it. "
    "The two linear elastic sets are exactly self-similar, their effective crack resistances scale with $J_c$ to within $0.1\\,\\%$.}"
    "{" + OLD_P2.rstrip("\n") + "}\n"
)
NEW_P3 = (
    "\\replaced{The energy balance of the model and the stress state in the ligament ahead of the trapped crack tip identify the mechanisms. "
    "In the linear elastic materials, $47$ to $83\\,\\%$ of the energy dissipated per pore-to-pore cycle are lost in the unstable snap-through of the ligament, in the plastic material only $12$ to $37\\,\\%$, whereas $49$ to $74\\,\\%$ create crack surface and $12$ to $16\\,\\%$ remain stored as plastic energy. "
    "The linear elastic ligaments re-nucleate at the one-dimensional fracture stress for all wall widths. Thin ligaments of the plastic material, $w_s \\le 0.6L$, yield across their width and reach this stress only after a large plastic strain, which keeps the driving force at re-nucleation low and makes the result sensitive to the yield stress and the hardening modulus. "
    "In wide ligaments, $w_s \\ge L$, the constraint ahead of the trapped crack raises the triaxiality to $1.0$ to $1.1$ and the normal stress at re-nucleation to $1.31$ to $1.34$ times the one-dimensional fracture stress, and the sensitivity to the plastic parameters is weak. "
    "The isotropic approximation of the embedding changes the effective crack resistance by at most $0.2\\,\\%$, the length of the initial crack and the number of resolved pore rows by at most $1.1\\,\\%$.}"
    "{" + OLD_P3.rstrip("\n") + "}\n"
)
NEW_P4 = (
    "\\replaced{The three-dimensional model with spherical pores behaves differently. The crack front bypasses the pores in small increments, no unstable snap-through of a ligament occurs, and the viscous share of the dissipated energy is below $1\\,\\%$, so that neither the toughening by forced re-nucleation nor the strong weakening of thin ligaments of the two-dimensional model occurs. "
    "The effective crack resistance, normalized by the crack resistance of the three-dimensional homogeneous material, is $0.87$ to $0.96$, nearly independent of the wall width, never above the homogeneous value and robust against the mesh. "
    "It lies above the two-dimensional values for thin walls, $0.87$ against $0.50$ at $w_s = 0.375L$, and below them for $w_s \\ge L$. "
    "The two-dimensional idealization of cylindrical pores thus does not carry over quantitatively to spherical pores.}"
    "{" + OLD_P4.rstrip("\n") + "}\n"
)
NEW_P5 = (
    "\\replaced{The effective crack resistance determined here is an operational quantity of the numerical experiment for the stated microstructure, crack-growth direction and loading, not a homogenized material property. "
    "Representativity in the sense of Hill is not established, since statistics over microstructure realizations, other crack-growth directions and the size of the specimen in the crack-growth direction were not tested. "
    "Such a study is the next step, followed by microstructures derived from micro-CT scans of porous or additively manufactured components and by a comparison with experimentally determined fracture resistances.}"
    "{" + OLD_P5.rstrip("\n") + "}\n"
)
NEW_P6 = "\\deleted{" + OLD_P6.rstrip("\n") + "}\n"
ms.rep(
    "influence of ductile material behavior using a phase-field modeling framework. By employing a\nvon~Mises-type constitutive law, we captured key aspects of ductile material behavior and\ncompared the results \\replaced{with}{against} multiple linear elastic reference models.",
    "influence of ductile material behavior using a phase-field modeling framework\\replaced{ with a von~Mises-type constitutive law and}{. By employing a\nvon~Mises-type constitutive law, we captured key aspects of ductile material behavior and}\ncompared the results \\replaced{with}{against} multiple linear elastic reference models.",
)
ms.rep(OLD_P2, NEW_P2)
ms.rep(OLD_P3, NEW_P3)
ms.rep(OLD_P4, NEW_P4)
ms.rep(OLD_P5, NEW_P5)
ms.rep(OLD_P6, NEW_P6)

# ---------------------------------------------------------------- Fig. 15 (3D mesh): rendering of the fixed-mesh run S1F w_s = 0.375L
ms.rep(
    r"""        \resizebox{0.42\textwidth}{!}{\includegraphics[]{pics/mesh_with_cs.png}}
        \begin{picture}(0,0)
                    \put(-209,157){$y$}
                    \put(-160,133){$x$}
                    \put(-170,87){$z$}
        \end{picture}
        \caption{\replaced{Three-dimensional mesh, illustration from a simulation on a coarser mesh for $w_s = 0.25L$, which is not part of the series evaluated below. The meshes of the present series use the same refinement band around the crack plane $y = 0$ with the fixed element size stated in the text.}{Three-dimensional mesh for $w_s=0.25L$.}}
        \mycomment{TODO}{replace by a rendering of the fixed-mesh run S1F, $w_s = 0.375L$ (mesh and crack surface $s$ around the pores, which would also illustrate the bypassing of the pores described in Section~\ref{sec:3D_fracture_behavior}).}
        \label{fig:mesh3D}""",
    r"""        \includegraphics[width=0.75\textwidth]{pics/PAPER_rev2_3D_mesh_section_ws0.375.pdf}% rev2: section of the mesh of the fixed-mesh run S1F_3D_vM_ws0.375 (076), replaces pics/mesh_with_cs.png (old w_s-scaled mesh of the v2.0 series, w_s = 0.25L)
        \caption{\replaced{Section of the tetrahedral mesh of the three-dimensional specimen for $w_s = 0.375L$ ($h = 0.1L$ in the crack band, $85\,681$ tetrahedra) through the plane of the pore centers, $z = -w_c/2$. a) Whole cross-section of the domain, $(6w_c) \times 20L$, the rectangle marks the region shown in b). b) Resolved porous region with the refined band around the crack plane $y = 0$ (dashed line). The polygons are the intersections of the tetrahedra with the section plane.}{Three-dimensional mesh for $w_s=0.25L$.}}
        \label{fig:mesh3D}""",
)

# ---------------------------------------------------------------- Working Note
ms.rep(
    "Open: Fig.~\\ref{fig:PAPER_01_all_Jx_vs_xct_pf_2} and the strip geometry of Section~3.5,\nabstract and conclusion update, rendering of a fixed-mesh 3D run for Fig.~\\ref{fig:mesh3D}.",
    "Part 8 (edits\\_round2\\_part8\\_final.py, 2026-10-23): Section~3.5 (homogeneous reference = $16L \\times 20L$ box without pores, Fig.~\\ref{fig:PAPER_01_all_Jx_vs_xct_pf_2} regenerated from the S2 runs as vector PDF, captions of Figs.~\\ref{fig:griffith_deformed} and~\\ref{fig:plasticity_vm_jeff} qualified), Fig.~\\ref{fig:mesh3D} replaced by a mesh section of the fixed-mesh run $w_s = 0.375L$, abstract and conclusion rewritten, keyword, cover letter. No open TODO.",
)
ms.save()

# ============================================================================ letter
# ---------------------------------------------------------------- editor response: item 6 + quotes of abstract and conclusion
lt.rep(
    "\\item all simulations recomputed with consistent parameters ($\\beta = 0.2$, $\\eta = 10^{-5}$ in two and three dimensions, linear elastic embedding); see ``Additional corrections by the authors''.\n\\end{enumerate}\n\\end{response}\n",
    "\\item all simulations recomputed with consistent parameters ($\\beta = 0.2$, $\\eta = 10^{-5}$ in two and three dimensions, linear elastic embedding); see ``Additional corrections by the authors'';\n"
    "\\item the abstract and the conclusion were rewritten to state the results of the revised Sections~4 and~5. In particular, the statement of the previous version that the two- and three-dimensional results agree quantitatively is withdrawn: with spherical pores, the crack front bypasses the pores, and the effective crack resistance stays below that of the homogeneous material for all wall widths (Section~5.2).\n"
    "\\end{enumerate}\n\\end{response}\n"
    "\n\\begin{relatedchanges}\n\\relhead{Abstract and conclusion, rewritten (quoted here once, referred to in the responses below)}\n"
    "%@quote cmd:The effective crack resistance of a simplified porous material with ductile bulk behavior is determined\n"
    "%@quote cmd:Effective Crack Resistance}{Effective Fracture Toughness}\n"
    "%@quote cmd: with a von~Mises-type constitutive law and}\n"
    "%@quote cmd:In the two-dimensional model with cylindrical pores, the effective crack resistance increases\n"
    "%@quote cmd:The energy balance of the model and the stress state in the ligament ahead of the trapped crack tip identify the mechanisms\n"
    "%@quote cmd:The three-dimensional model with spherical pores behaves differently\n"
    "%@quote cmd:The effective crack resistance determined here is an operational quantity\n"
    "%@quote cmd:Overall, this work provides a robust and extensible framework\n"
    "\\end{relatedchanges}\n",
)

# ---------------------------------------------------------------- minor 25: Fig. 3 TODO resolved, captions quoted
lt.rep(
    "The former figure of the fields $s$ and $\\alpha$ for $w_s = 0.25L$ was removed, since this wall width is no longer part of the three-dimensional series.\\mycomment{TODO}{Fig.~3 (Section~3.5) follows with the regenerated strip/solid figure.}",
    "The former figure of the fields $s$ and $\\alpha$ for $w_s = 0.25L$ was removed, since this wall width is no longer part of the three-dimensional series, and Fig.~15 now shows a section through the mesh of the new series ($w_s = 0.375L$) instead of the mesh of the previous version. Fig.~3 (Section~3.5) was regenerated as a vector graphic from the new homogeneous reference runs and shows solver output only, see items~12 and~13 of ``Additional corrections by the authors''. The captions of Figs.~2 and~4, which show field plots from the earlier strip simulations, were qualified accordingly.",
)
lt.rep(
    "%@quote cmd:illustration from a simulation on a coarser mesh\n",
    "%@quote cmd:Section of the tetrahedral mesh of the three-dimensional specimen\n",
)
lt.rep(
    "\\begin{relatedchanges}\n\\relhead{Captions of Figs.~9 to 12}\n",
    "\\begin{relatedchanges}\n\\relhead{Captions of Figs.~2 to 4 (Section~3.5)}\n"
    "%@quote cmd:shown to illustrate the enforced crack path\n"
    "%@quote cmd:The shaded band marks the central part of the crack path\n"
    "%@quote cmd:Plastic deformation stays confined to the crack path\n"
    "\\end{relatedchanges}\n\n"
    "\\begin{relatedchanges}\n\\relhead{Captions of Figs.~9 to 12}\n",
)

# ---------------------------------------------------------------- Additional corrections: items 12 and 13 + quotes of the Section 3.5 text
lt.rep(
    "three dimensions it captures the peak that is attained while the crack front passes beside the spherical pore, which\na window on the ligament misses for thin walls.\n\\end{enumerate}",
    "three dimensions it captures the peak that is attained while the crack front passes beside the spherical pore, which\na window on the ligament misses for thin walls.\n"
    "\\item The homogeneous reference of Section~3.5 is now the $16L \\times 20L$ simulation domain of the porous specimens ($w_s = L$)\n"
    "without pores, with the yield stress assigned to the whole domain for the von~Mises set, instead of the $16L \\times 2L$ strip of\n"
    "the previous version, and $J_c^{\\mathrm{num}}$ is defined as the mean of $J_x$ over the central part of the crack path.\n"
    "\\item The curve of the von~Mises set in Fig.~3 of the previous version ended in a synthetic continuation of the final descent of\n"
    "$J_x$ that had been appended for plotting purposes; the figure was regenerated from the new reference runs and shows solver\n"
    "output only.\n"
    "\\end{enumerate}",
)
lt.rep(
    "%@quote cmd:values are obtained from the plateau of $J_x$\n",
    "%@quote cmd:The computational domain is the simulation domain of the porous specimens\n"
    "%@quote cmd:the energy release rate of the applied K-field is\n"
    "%@quote cmd:The J-Integral rises during a short transient\n"
    "%@quote cmd:values are obtained from the plateau of $J_x$\n",
)
lt.save()

# ============================================================================ cover letter to the editor
ed.rep(r"\noindent Darmstadt, \mycomment{TODO}{date}", r"\noindent Darmstadt, \today")
ed.rep(
    r"\item \mycomment{TODO}{introduction and framing of the numerical experiment rewritten (Reviewer~1, points 1--2; Reviewer~3, comment 2): \dots}",
    r"\item The introduction and the framing of the numerical experiment were rewritten (Reviewer~1, points~1 and~2; Reviewer~3, comment~2). The manuscript now distinguishes the rigorous homogenization of surface energies of Griffith type (Braides et al., Cagnetti et al., Michel and Suquet), the crack-front analyses and the numerical experiment of Hossain et al., cites Hill's definition of a representative volume element, states why the resolved region is not called representative and names the limitations of the numerical experiment. Ten references were added.",
)
ed.rep(
    r"\item \mycomment{TODO}{mechanism study added (Reviewer~1, point 3; Reviewer~3, comment 1): energy partition, dissipation per re-nucleation event, ligament stress state, sensitivity with respect to yield stress and hardening, scaling argument}",
    r"\item A mechanism study was added to the two-dimensional results (Reviewer~1, point~3; Reviewer~3, comment~1): the energy partition per pore-to-pore cycle from the exact energy balance of the model, the stress state and the plastic zone in the ligament immediately before re-nucleation, the sensitivity with respect to the yield stress and the hardening modulus, and the scaling of the linear elastic reference sets with the fracture resistance. The influence of plasticity is explained by a nucleation regime for thin and a propagation regime for wide walls.",
)
ed.rep(
    r"\item \mycomment{TODO}{cubic symmetry of the effective stiffness quantified and its influence on the embedding tested (Reviewer~1, point 4)}",
    r"\item The square resp.\ cubic symmetry of the effective stiffness of the unit cells is quantified and the isotropic approximation of the embedding was tested with the full anisotropic stiffness (Reviewer~1, point~4): the effective crack resistance changes by at most $0.2\,\%$, and by at most $1.1\,\%$ with the length of the initial crack and the number of resolved pore rows.",
)
ed.rep(
    "\\item All simulations were recomputed with consistent parameters after an audit of the code against the manuscript;\nthe corrections are listed at the end of the response letter. The conclusions are unchanged.",
    "\\item All simulations were recomputed. Reviewer~3 requested no new simulations; they were carried out to answer points~3\n"
    "and~4 of Reviewer~1 with recorded observables and to correct inconsistencies between the simulation code and the manuscript\n"
    "found in an audit of the code (listed at the end of the response letter). The two-dimensional conclusions are confirmed. The\n"
    "three-dimensional series was recomputed on meshes with a fixed element size and normalized by a three-dimensional homogeneous\n"
    "reference, and its result changed: with spherical pores, the crack front bypasses the pores and the effective crack resistance\n"
    "no longer exceeds that of the homogeneous material. The abstract and the conclusion were rewritten accordingly.",
)
ed.rep(r"\usepackage[a4paper,margin=25mm]{geometry}", r"\usepackage[a4paper,margin=20mm]{geometry}")
ed.rep("\\vspace{8mm}\n\\noindent Prof.", "\\vspace{5mm}\n\\noindent Prof.")
ed.rep("\\vspace{8mm}\n\\noindent Darmstadt", "\\vspace{5mm}\n\\noindent Darmstadt")
ed.rep("\\vspace{6mm}\n\\noindent \\textbf{Resubmission", "\\vspace{4mm}\n\\noindent \\textbf{Resubmission")
ed.rep("\\vspace{6mm}\n\\noindent Sincerely", "\\vspace{4mm}\n\\noindent Sincerely")
ed.rep("\\vspace{10mm}\n\\noindent Alexander", "\\vspace{6mm}\n\\noindent Alexander")
ed.save()
print("part 8 applied")
