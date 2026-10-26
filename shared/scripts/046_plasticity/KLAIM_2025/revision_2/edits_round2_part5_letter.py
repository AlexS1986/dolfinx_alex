#!/usr/bin/env python3
"""Round-2 edits, part 5 (2026-10-22), letter: answers that use the Section-4 results
(R1 major 2/3/4, minors 16/24/25, R3 comments 1/3, editor summary, additional corrections).
One-shot: every anchor must occur exactly once in response_to_reviewers_template.tex.
Figure/table numbers refer to the marked manuscript built on 2026-10-22
(Figs. 9-16, Tables 1-5 in Section 4, Table 6 in Section 5).
"""
FN = "response_to_reviewers_template.tex"
t = open(FN, encoding="utf-8").read()
E = []


def rep(old, new):
    n = t.count(old)
    if n != 1:
        raise SystemExit("anchor occurs %d times:\n%s" % (n, old[:200]))
    E.append((old, new))


# ----------------------------------------------------------------- editor summary
rep(r"""\item \mycomment{TODO}{mechanism study: energy partition of the numerical experiment, dissipation per re-nucleation event, stress state in the ligament before re-nucleation (new Section~X / Figures~Y), sensitivity with respect to $\sigma_y$ and $H$, scaling argument for the linear elastic sets}
\item the square resp.\ cubic symmetry of the effective stiffness of the unit cells is quantified (Zener ratio) and the influence of the isotropic approximation of the embedding is tested by simulations with the full anisotropic stiffness\mycomment{TODO}{numbers from study S3};""",
    r"""\item a mechanism study was added to the two-dimensional results (new Sections~4.3 and 4.4, Figs.~15 and 16, Tables~3 to 5): the exact energy balance of the model is evaluated per pore-to-pore cycle (fracture, stored plastic and viscous shares of the dissipated energy, energy dissipated per unit crack advance), the stress state and the plastic zone in the ligament ahead of the trapped crack tip are recorded immediately before re-nucleation and compared with the one-dimensional nucleation stress, the sensitivity with respect to $\sigma_y$ and $H$ is quantified, and the scaling of the linear elastic sets with $J_c$ is derived and confirmed. The two regimes of the influence of plasticity (nucleation regime for thin walls, propagation regime for wide walls) are explained on this basis, and the interpretation in Section~4.2 was rewritten accordingly;
\item the square resp.\ cubic symmetry of the effective stiffness of the unit cells is quantified (Zener ratio, Tables~1 and 6) and the influence of the isotropic approximation of the embedding is tested by simulations with the full anisotropic stiffness (new Section~4.5): the effective crack resistance changes by at most $0.2\,\%$;
\item the influence of the length of the initial crack and of the number of resolved pore rows on the effective crack resistance is tested (new Section~4.6): at most $1.1\,\%$;""")

# ----------------------------------------------------------------- R1 major 2
rep(r"""the length of the initial crack and the number of resolved pore rows are varied in the new Section~\mycomment{TODO}{4.6 (study S5)}, and the resulting change of $J_{x,c}^{\text{eff}}$ is reported there.\mycomment{TODO}{numbers from study S5}""",
    r"""the length of the initial crack and the number of resolved pore rows are varied in the new Section~4.6. Doubling the length of the initial crack changes $J_{x,c}^{\text{eff}}$ by $+0.08\,\%$ ($1.5w_c$: $+0.02\,\%$), five instead of three resolved pore rows change it by $-0.02\,\%$ (\vMLabel, $w_s = L$), $+1.1\,\%$ (\vMLabel, $w_s = 0.375L$) and $-0.09\,\%$ (\EqSigmaStarLabel, $w_s = L$). In addition, the energy dissipated per pore-to-pore cycle (new Section~4.3) is constant from cycle to cycle except for the last cycle at the end of the resolved region, so that the recorded peak belongs to a periodic regime. Section~4.6 also states explicitly what was not tested (length of the resolved region in the crack-growth direction, other crack-growth directions and initial crack positions, other microstructures, statistics over realizations, amplitude of the K-field).""")

rep(r"""\begin{relatedchanges}
\relhead{\mycomment{TODO}{quotes: new Section 4.6 with the S5 results}}
\end{relatedchanges}""",
    r"""\begin{relatedchanges}
\relhead{New Section~4.6 (influence of the initial crack and of the size of the resolved region)}
%@quote cmd:The definition of $J_{x,c}^{\text{eff}}$ in Section~\ref{sec:numericalExperiment} refers to one initial crack
%@quote cmd:These tests do not establish representativity
\end{relatedchanges}""")

# ----------------------------------------------------------------- R1 major 3
rep(r"""\begin{response}
\mycomment{TODO}{answer: (a) energy partition of every simulation (exact balance $W = E_{\mathrm{el}} + E_{\mathrm{pl}} + E_{\mathrm{fr}} + D_{\mathrm{visc}}$, per re-nucleation event; the energy released in the unstable ligament failure is dissipated by the viscous regularisation and is the quantity that distinguishes the material laws); (b) stress state and plastic zone in the ligament immediately before re-nucleation (event snapshots) for the two regimes $w_s \lesssim 0.6L$ and $w_s \gtrsim 2L$; (c) sensitivity study $\sigma_y \in \{0.75, 1.5\}$, $H \in \{0.11, 0.44\}\mu_0$ at $w_s = 0.375L$ and $1.0L$ (study S4); (d) scaling argument: for the rate-independent linear elastic model $J_{x,c}^{\text{eff}} \propto J_c$ exactly, confirmed by \EqJcLabel vs.\ \EqSigmaStarLabel; (e) reduced one-dimensional ligament model for the nucleation regime (Sec.~3.6 extended). Numbers from 075/076 results.}
\end{response}

\begin{relatedchanges}
\relhead{\mycomment{TODO}{quotes: new mechanism section, new figures (energy partition vs.\ $w_s$, event energies, ligament stress), sensitivity paragraph, scaling paragraph}}
\end{relatedchanges}""",
    r"""\begin{response}
We accept this criticism. The two-dimensional results were recomputed with additional observables, and a mechanism study was added in the new Sections~4.3 and 4.4 (Figs.~15 and 16, Tables~3 to 5). The post-hoc interpretations of Section~4.2 of the previous version were removed or replaced by statements that are backed by these results. In detail:
\begin{enumerate}
\item \emph{Energy partition.} The model obeys the exact energy balance $W_{\text{ext}} = E_{\text{el}} + E_{\text{pl}} + E_{\text{fr}} + D_{\text{visc}}$, new eq.~(37), where $D_{\text{visc}} = \int\!\!\int \dot{s}^2/M$ is the energy dissipated by the viscous evolution of the phase field (the energy released in the unstable failure of a ligament, which would be kinetic energy in a dynamic model) and $E_{\text{pl}}$ is the plastic energy stored in the degraded form $g(s)\psi_{\text{pl}}$ of the model. All terms are recorded in every time step. The partition is evaluated per pore-to-pore cycle (loading of one ligament while the crack is trapped, re-nucleation, failure of the ligament) for the cycle that contains the ligament in which $J_{x,c}^{\text{eff}}$ is attained, Fig.~15 and Table~3. For the linear elastic sets, $47$ to $83\,\%$ of the energy dissipated per cycle are lost in the unstable snap-through of the ligament and only $17$ to $53\,\%$ create crack surface. For the von~Mises set, the viscous share is $12$ to $37\,\%$, the fracture share $49$ to $74\,\%$ and $12$ to $16\,\%$ remain stored as plastic energy. The energy dissipated per unit crack advance, $G_{\text{diss}}$, shows the same crossover between the von~Mises set and \EqSigmaStarLabel as the peak value $J_{x,c}^{\text{eff}}$ (smaller for $w_s \le 0.375L$, equal at $0.6L$, larger for wide walls), so that the reduction of the crack resistance by plasticity for thin walls is not an artefact of the peak definition. The energy dissipated per cycle is constant from cycle to cycle (periodic regime), and the two linear elastic sets give the same fractions to two to three digits.
\item \emph{Stress state and plastic zone before re-nucleation.} The maxima of $\sigma_{yy}$, $\sigma_{\text{vM}}$, $\sigma_h$ and $\alpha$ in the next intact ligament are recorded in every time step and evaluated at the peak of $J_x$, Fig.~16 and Table~4. The linear elastic sets re-nucleate when $\sigma_{yy,\max}$ reaches $0.95$ to $1.02$ times the one-dimensional fracture stress of Section~3.6, for all wall widths (stress-controlled nucleation, see also minor point~24). For the von~Mises set, two regimes are identified. For $w_s \le 0.6L$ the ligament yields across its width ($\alpha_{\max} \approx 0.7$, triaxiality $0.67$ to $0.74$) and re-nucleates at the one-dimensional fracture stress after a large plastic strain, i.e., it behaves like the one-dimensional bar of Section~3.6, which is the reduced model for this regime. Once yielded, the ligament opens at nearly constant stress, so that the macroscopic driving force at re-nucleation is close to the value at which the ligament yields and lower than for the linear elastic ligament of equal strength. The plastic work spent on stretching the ligament is supplied by the boundary conditions but does not enter $J_x$, hence the reduction of the effective crack resistance below \EqSigmaStarLabel. For $w_s \ge L$ the plastic zone does not span the ligament, the trapped tip acts as a constrained notch (triaxiality $1.0$ to $1.1$), the von~Mises stress stays at the yield stress while the normal stress reaches $1.31$ to $1.34$ times the one-dimensional fracture stress, re-nucleation is delayed and the effective crack resistance is raised. This also explains the later saturation of the von~Mises set.
\item \emph{Sensitivity study.} $J_{x,c}^{\text{eff}}$ was recomputed for $\sigma_y = 0.75$ and $1.5\sqrt{J_c^0\mu_0/L}$ and for $H = 0.11$ and $0.44\mu_0$ at $w_s = 0.375L$ and $w_s = L$, Table~5. In the nucleation regime the changes are $-2$ to $+37\,\%$, in the propagation regime $-6$ to $+6\,\%$, which confirms the two regimes.
\item \emph{Scaling argument.} For the linear elastic model, $J_c$ is the only energy scale, the amplitude of the K-field is proportional to $\sqrt{J_c}$ by the new choice of $K_I^{\text{bc}}$, and $J_{x,c}^{\text{eff}} \propto J_c$ in the rate-independent limit (Section~4.2). The simulations confirm this to $0.1\,\%$: the ratio of the effective crack resistances of \EqJcLabel and \EqSigmaStarLabel is $1.809$ to $1.810$ for all wall widths against $1/0.5523 = 1.811$. Normalized by their own numerical fracture resistance, the two linear elastic sets fall onto one curve (Fig.~14b), and the von~Mises set lies below it for all wall widths, i.e., relative to its own crack resistance in the homogeneous material, the ductile material is toughened less by the pores than the linear elastic material. The previous statement that ``a pronounced toughening effect is observed in all cases'' was qualified accordingly.
\item \emph{Isolation of competing effects.} The three parameter sets isolate the microscopic fracture resistance (\vMLabel vs.\ \EqJcLabel, same $J_c$) and the one-dimensional strength (\vMLabel vs.\ \EqSigmaStarLabel, same $\sigma^*$), the new Section~4.5 isolates the anisotropy of the embedding and Section~4.6 the initial crack and the size of the resolved region.
\end{enumerate}
No asymptotic analysis is attempted. The conclusions drawn from the results are now the two regimes and their mechanisms as stated above, and the limitations are stated in Section~4.6.
\end{response}

\begin{relatedchanges}
\relhead{Section~4.2, interpretation rewritten (the complete Section~4 was revised, only the interpretative changes are quoted here)}
%@quote cmd:The peak values are $J_{x,c}^{\text{eff}} = 2.02$
%@quote cmd:has the same ordering as $J_{x,c}^{\text{eff}}$ at this wall width
%@quote cmd:The increase is steep for thin ligaments
%@quote cmd:The ordering \EqJcLabel $>$ \vMLabel $>$ \EqSigmaStarLabel observed
%@quote cmd:The linear elastic sets reach $95
%@quote cmd:The two linear elastic parameter sets differ only in the value of $J_c$
\end{relatedchanges}
\begin{relatedchanges}
\relhead{New Section~4.3 (energy partition), Fig.~15, Table~3}
%@quote cmd:Energy partition of one pore-to-pore cycle vs.
%@quote cmd:The model of Section~\ref{sec:phase_fiel_model_ductile} obeys an exact energy balance
%@quote env:eq:energy_balance
%@quote cmd:where $E_{\text{el}}$, $E_{\text{pl}}$ and $E_{\text{fr}}$ denote
%@quote cmd:The partition is evaluated per pore-to-pore cycle
%@quote env:eq:D_tot_cycle
%@quote cmd:and $G_{\text{diss}} = \Delta D_{\text{tot}}/w_c$ is the energy dissipated per unit crack advance, averaged
%@quote cmd:Energy partition of the cycle that contains the ligament
%@quote cmd:For the two linear elastic sets, which give the same fractions
\end{relatedchanges}
\begin{relatedchanges}
\relhead{New Section~4.4 (stress state and plastic zone before re-nucleation, sensitivity), Fig.~16, Tables~4 and 5}
%@quote cmd:State of the next intact ligament ahead of the trapped crack tip at the peak
%@quote cmd:To relate the re-nucleation in the porous specimen
%@quote cmd:Effective crack resistance and state of the next intact ligament
%@quote cmd:For the two linear elastic sets, re-nucleation occurs
%@quote cmd:For the von~Mises set, two regimes can be distinguished
%@quote cmd:For wide ligaments, $w_s \ge L$, the plastic zone does not span
%@quote cmd:Sensitivity of the effective crack resistance of the parameter set
%@quote cmd:The sensitivity of $J_{x,c}^{\text{eff}}$ with respect to the plastic parameters confirms
\end{relatedchanges}""")

# ----------------------------------------------------------------- R1 major 4
rep(r"""Tables~\mycomment{TODO}{X and Y}), and explains which constants the isotropic approximation uses.""",
    r"""Tables~1 and 6), and explains which constants the isotropic approximation uses.""")
rep(r"""The influence of the approximation on the result is tested directly: the simulations of Section~\mycomment{TODO}{4.5 (study S3)} assign the full square-symmetric (cubic) stiffness to the embedding for $w_s = 0.375L$, $1.0L$ and $4.0L$ in two dimensions and for $w_s = 0.375L$ and $1.0L$ in three dimensions.\mycomment{TODO}{result: $J_{x,c}^{\text{eff}}$ changes by $y$\,\% (study S3), which justifies the approximation for the conclusions drawn.}""",
    r"""The influence of the approximation on the result is tested directly in the new Section~4.5: the simulations for $w_s = 0.375L$, $L$ and $4L$ (parameter set \vMLabel) were repeated with the full square-symmetric stiffness $C_{11}$, $C_{12}$, $C_{33}$ assigned to the embedding. The effective crack resistance changes by $+0.18\,\%$, $+0.08\,\%$ and $+0.01\,\%$, respectively, i.e., the approximation has no influence on any conclusion of the paper, not even for the thinnest walls where the anisotropy is largest. This is plausible because the embedding only transmits the K-field and the J-integral is evaluated on its outer boundary, whereas the re-nucleation that determines $J_{x,c}^{\text{eff}}$ takes place in the explicitly resolved ligaments. The three-dimensional unit cells are closer to isotropic than the two-dimensional ones: their Zener ratios are $A = 0.969$, $0.978$, $0.984$, $0.992$, $0.996$ and $1.000$ for $w_s = 0.25L$, $0.375L$, $0.5L$, $0.75L$, $L$ and $2L$ (Table~6). Two earlier three-dimensional runs on a coarser mesh with the full cubic stiffness in the embedding changed $J_{x,c}^{\text{eff}}$ by $-0.37\,\%$ ($w_s = 0.375L$) and $-0.22\,\%$ ($w_s = L$) relative to their isotropic twins, which is within the numerical scatter of those runs.\mycomment{TODO}{Section~5: state the Zener ratios in Table~6 and refer to Section~4.5 for the cubic-embedding test (3D numbers from the old-mesh pairs, or add a fixed-mesh S3F run).}""")
rep(r"""The local stress concentration at the corners therefore does not enter the recorded driving force, and it did not lead to crack nucleation at a corner in any of the simulations.\mycomment{TODO}{quantify the corner stress from the field output of study S1 and state it in Section~4.5}""",
    r"""The local stress concentration at the corners therefore does not enter the recorded driving force. In none of the simulations of this work did a crack nucleate at a corner of the resolved region, the crack pattern remained confined to the central row of pores. Both facts are now stated in Section~4.5.""")
rep(r"""\begin{relatedchanges}
\relhead{\mycomment{TODO}{quotes: Tables of the effective constants, new Section 4.5 with the S3 results}}
\end{relatedchanges}""",
    r"""\begin{relatedchanges}
\relhead{Table~1 (Section~4.1), new Section~4.5 (isotropic approximation of the embedding, corners)}
%@quote cmd:Effective elastic constants of the two-dimensional unit cells
%@quote cmd:The embedding of the resolved porous region is assigned the isotropic approximation
%@quote cmd:The corners of the resolved region, where the material discontinuity
\end{relatedchanges}""")

# ----------------------------------------------------------------- minor 16, 24, 25
rep(r"""refers to the energy partition of Section~\mycomment{TODO}{4.3}, where the energy dissipated by the viscous evolution of $s$ is quantified for the numerical experiments.""",
    r"""refers to the energy partition of Section~4.3, where the energy dissipated by the viscous evolution of $s$ is quantified for the numerical experiments (it is $47$ to $83\,\%$ of the energy dissipated per pore-to-pore cycle for the linear elastic sets and $12$ to $37\,\%$ for the von~Mises set, Table~3).""")

rep(r"""and the new simulations record the stress state in the ligament ahead of the trapped crack tip immediately before re-nucleation, which is compared with the one-dimensional values in Section~\mycomment{TODO}{4.4 (study S1 outputs; numbers to be inserted)}.
\end{response}
\begin{relatedchanges}
\relhead{Section~3.6}
%@quote cmd:Strictly speaking, these relations hold only in one dimension. In two and three dimensions""",
    r"""and the new simulations record the stress state in the ligament ahead of the trapped crack tip immediately before re-nucleation, which is compared with the one-dimensional values in the new Section~4.4 (Fig.~16). This provides the quantitative evidence for the present geometry: for the two linear elastic parameter sets, re-nucleation occurs when the maximum normal stress in the ligament reaches $0.95$ to $1.02$ times the one-dimensional fracture stress, for all wall widths from $0.25L$ to $4L$ and for both values of $J_c$. For the von~Mises set, the one-dimensional value holds for thin ligaments that yield across their width ($0.95$ to $1.11\,\sigma^*$ for $w_s \le 0.6L$), whereas the constraint ahead of the trapped crack tip raises the nucleation stress to $1.31$ to $1.34\,\sigma^*$ for $w_s \ge L$ (triaxiality $1.0$ to $1.1$), so that the one-dimensional estimate is not sufficient there. The text of Section~3.6 refers to this comparison.
\end{response}
\begin{relatedchanges}
\relhead{Section~3.6 and new Section~4.4}
%@quote cmd:Strictly speaking, these relations hold only in one dimension. In two and three dimensions
%@quote cmd:For the two linear elastic sets, re-nucleation occurs""")

rep(r"""\begin{response}\mycomment{TODO}{all result figures regenerated from the new data as vector graphics (pgf), axis labels capitalised.}\end{response}""",
    r"""\begin{response}
The result figures of Sections~3.6 and 4 were regenerated from the new simulations as vector graphics (pgf, Figs.~6, 7 and 9 to 16) with axis labels that start with an upper case letter. Fig.~9 (the former Fig.~9, crack tip position vs.\ pseudo-time) is now a vector graphic, and the J-integral records of Fig.~10 and Figs.~11 to 13 are plotted with the pore positions shaded resp.\ with the crack tip position measured from the center of pore two and normalized by the cell width, so that the curves for different wall widths can be compared directly.\mycomment{TODO}{Fig.~3 (Section~3.5) and the figures of Section~5 (Figs.~17 to 22) follow with the 076 results.}
\end{response}
\begin{relatedchanges}
\relhead{Captions of Figs.~9 to 14}
%@quote cmd:for $w_s = L$ and the parameter set \vMLabel}
%@quote cmd:and the three parameter sets. The shaded bands mark
%@quote cmdall:measured from the center $x_{p2}$ of pore two
%@quote cmd:: a) normalized by the reference fracture resistance
\end{relatedchanges}""")

# ----------------------------------------------------------------- R3 comment 1
rep(r"""\begin{response}
\mycomment{TODO}{same material as Reviewer 1, major point 3: energy partition based on eq.~(14) (now with the exact balance incl.\ the viscous term), stress/plastic-zone fields in the ligament immediately before re-nucleation for $w_s = 0.375L$ and $1.0L$ (vM vs.\ Eq$\sigma^*$), explanation of the small-$w_s$ regime (plastic zone spans the ligament, strain-driven nucleation at lower stress, the plastic work spent before nucleation is not available for the jump) vs.\ the large-$w_s$ regime (plastic dissipation ahead of the trapped tip delays re-nucleation).}
\end{response}

\begin{relatedchanges}
\relhead{\mycomment{TODO}{quotes as for R1 major point 3}}
\end{relatedchanges}""",
    r"""\begin{response}
We thank the reviewer for this suggestion, which we followed closely. Both requested elements were added to the two-dimensional results, without additional microstructures but with additional observables recorded in the recomputed simulations: an energy partition based on the exact energy balance of the model (new Section~4.3), and the stress state in the ligament ahead of the trapped crack tip immediately before re-nucleation (new Section~4.4). The energy balance derived from eq.~(14) (now eq.~(37)) contains, besides the stored elastic energy, the degraded plastic energy and the fracture energy, the energy dissipated by the viscous evolution of the phase field, which is the energy released in the unstable failure of a ligament. It is evaluated per pore-to-pore cycle, Fig.~15 and Table~3.

The answer to the reviewer's question, why the ductile material law \emph{reduces} the effective crack resistance below that of \EqSigmaStarLabel for small $w_s/L$, is the following. The two sets have the same one-dimensional nucleation stress, and the recorded ligament stresses show that both re-nucleate at this stress ($0.95$ to $1.11\,\sigma^*$ for $w_s \le 0.6L$). The difference lies in the way the ligament reaches it. The thin ligament of the von~Mises material yields across its full width ($\alpha_{\max} \approx 0.7$ at a triaxiality of $0.67$ to $0.74$, i.e., the state of the one-dimensional bar at fracture) and then opens at nearly constant stress, rising slowly from $\sigma_y$ to $\sigma^* = 1.28\sigma_y$ with the hardening modulus. The macroscopic driving force $J_x$, which is the energy release rate measured in the elastic surroundings, is therefore close to the value at which the ligament yields when re-nucleation occurs, whereas the linear elastic ligament of equal strength has to be loaded elastically up to $\sigma^*$, which requires a higher $J_x$. The plastic work spent on stretching the ligament is supplied by the boundary conditions but does not appear in $J_x$ and is not available as crack resistance. The energy partition confirms that this is not an artefact of the peak definition: the energy dissipated per unit crack advance is also smaller for the von~Mises set in this regime ($0.48$ vs.\ $0.71\,J_c^0$ at $w_s = 0.25L$), because the linear elastic structure stores much more elastic energy up to the moment of re-nucleation, which is then released in the snap-through ($83\,\%$ of the dissipated energy at $w_s = 0.25L$, against $33\,\%$ for the von~Mises set). The sensitivity study (Table~5) is consistent with this picture: at $w_s = 0.375L$, $J_{x,c}^{\text{eff}}$ responds strongly to the yield stress and the hardening modulus ($-12$ to $+37\,\%$), at $w_s = L$ only weakly ($-6$ to $+6\,\%$).

For large $w_s/L$ the mechanism suggested by the reviewer is confirmed and quantified: the plastic zone does not span the ligament ($\alpha_{\max} = 0.44$ to $0.60$), the trapped crack tip acts as a constrained notch with a triaxiality of $1.0$ to $1.1$, the von~Mises stress stays at the yield stress while the normal stress in the ligament reaches $1.31$ to $1.34\,\sigma^*$, i.e., the level of the one-dimensional fracture stress of \EqJcLabel, before the driving force suffices for re-nucleation. The plastic work ahead of the trapped tip delays the re-nucleation and raises the effective crack resistance above that of \EqSigmaStarLabel, and the plastic zone sets the interaction length between the pores, which explains the later saturation. The interpretation in Section~4.2 of the previous version (``the enhanced dissipation appears to play an important role'') was replaced by references to these results.
\end{response}

\begin{relatedchanges}
\relhead{New Sections~4.3 and 4.4, Figs.~15 and 16, Tables~3 to 5 (quoted under Reviewer~1, major point~3)}
\end{relatedchanges}""")

# ----------------------------------------------------------------- R3 comment 3
rep(r"""their own $J_c^{\mathrm{num}}$, \EqJcLabel and \EqSigmaStarLabel give the same curve $J_{x,c}^{\text{eff}}(w_s)$
\mycomment{TODO}{insert the numbers from the new S1 runs; v2.0 data: both $\approx 1.9$--$2.0\,J_c^{\mathrm{num}}$ in
the saturation regime}. A linear elastic set with $J_c^{\mathrm{num}} = J_c^{\mathrm{num,pl}}$ would therefore reach
$J_{x,c}^{\text{eff}} \approx 1.9\,J_c^{\mathrm{num,pl}}$, i.e.\ it would be \emph{tougher} than the von~Mises set
($\approx 1.25\,J_c^{\mathrm{num,pl}}$ \mycomment{TODO}{new value}) at large wall widths. This strengthens the
conclusion that the von~Mises material does not simply behave like a linear elastic material with an increased crack
resistance; the additional dissipation during re-nucleation is spent differently. A paragraph with this argument was
added. \mycomment{TODO}{quote}
\end{response}

\begin{relatedchanges}
\relhead{\mycomment{TODO}{quote: new paragraph at the end of Section~4.2}}
\end{relatedchanges}""",
    r"""their own $J_c^{\mathrm{num}}$, \EqJcLabel and \EqSigmaStarLabel give the same curve $J_{x,c}^{\text{eff}}(w_s)$
(Fig.~14b and Table~2): the ratio of their effective crack resistances is $1.809$ to $1.810$ for all wall widths,
against $1/0.5523 = 1.811$ for the ratio of the prescribed $J_c$, and both reach $2.03\,J_c^{\mathrm{num}}$ at
$w_s = 4L$. In the new simulations, the amplitude of the K-field is also proportional to $\sqrt{J_c}$ (Section~4.2),
so that the two sets are loaded self-similarly. A linear elastic set with $J_c^{\mathrm{num}} = J_c^{\mathrm{num,pl}} =
1.425\,J_c^0$ would therefore reach $J_{x,c}^{\text{eff}} = 2.03 \times 1.425\,J_c^0 = 2.89\,J_c^0$ at $w_s = 4L$, i.e.\
it would be \emph{tougher} than the von~Mises set ($1.90\,J_c^0 = 1.34\,J_c^{\mathrm{num,pl}}$). The same holds in
three dimensions\mycomment{TODO}{3D: insert $J_{x,c}^{\text{eff}}/J_c^{\mathrm{num,3D}}$ of the fixed-mesh runs (S1F/S2F, $\approx 0.95$ at $w_s = L$) once Section~5 is updated}.
This strengthens the conclusion that the von~Mises material does not simply behave like a linear elastic material with
an increased crack resistance. The pores toughen it less, relative to its own crack resistance in the homogeneous
material, than they toughen a linear elastic material, because part of the work supplied before re-nucleation is spent
on plastic deformation of the ligament and does not enter the macroscopic driving force (see the response to Comment~1).
A paragraph with this argument was added at the end of Section~4.2.
\end{response}

\begin{relatedchanges}
\relhead{Section~4.2, new paragraph (scaling of the linear elastic sets), and Section~3.5}
%@quote cmd:The ratio of the values of \EqJcLabel and \EqSigmaStarLabel, $1.088/0.604
%@quote cmd:The two linear elastic parameter sets differ only in the value of $J_c$
\end{relatedchanges}""")

# ----------------------------------------------------------------- additional corrections
rep(r"""linear elastic as described. All results in Sections~3.5, 4 and 5 were recomputed accordingly \mycomment{TODO}{(list the
numbers that changed: $J_c^{\mathrm{num}}$, saturation values, \dots)}.""",
    r"""linear elastic as described. All results in Sections~3.5, 4 and 5 were recomputed accordingly. The numerical fracture
resistances of Section~3.5 are now $J_c^{\mathrm{num}} = 1.425$, $1.088$ and $0.604\,J_c^0$ for \vMLabel, \EqJcLabel and
\EqSigmaStarLabel (previously $1.50$, $1.12$ and $0.64\,J_c^0$). The effective crack resistances of Section~4 are listed in
the new Table~2 for all wall widths. For $w_s \ge L$ they differ from the values of the previous version by at most
$3\,\%$ (\vMLabel and \EqJcLabel), for thinner walls by up to $11\,\%$, where also the amplitude of the K-field was
changed (item~8). The saturation values are now stated in units of $J_c^0$ and of the numerical fracture resistance of
the respective set ($1.34$, $2.03$ and $2.03\,J_c^{\mathrm{num}}$ at $w_s = 4L$ for \vMLabel, \EqJcLabel and
\EqSigmaStarLabel, previously $1.80$, $1.9$ and $1.12$ in units of the $J_c^{\mathrm{num}}$ of \EqJcLabel).\mycomment{TODO}{Section~5 numbers (076) to be added here once available.}""")

rep(r"""stiffness are $A = 0.89,\ 0.96,\ 1.00$ (all values for $\lambda_0 = \mu_0$, complete table in Section~4.1).
\end{enumerate}""",
    r"""stiffness are $A = 0.89,\ 0.96,\ 1.00$ (all values for $\lambda_0 = \mu_0$, complete table in Section~4.1).
\item The amplitude of the surfing K-field was $K_I^{\mathrm{bc}} = \sqrt{2.5J_c^0\mu_0}$ for all porous simulations of the
previous versions. Since the auxiliary moduli of the K-field are the effective moduli of the embedding (item~3), the energy
release rate $J_{\mathrm{bc}} = (1-\nu_{\mathrm{bc}}^2)(K_I^{\mathrm{bc}})^2/E_{\mathrm{bc}}$ applied by this K-field
depended on the wall width and reached $3.4\,J_c^0$ for the thinnest walls, whose embedding is soft. There the crack ran
far ahead of the fictitious crack tip and the recorded $J_x$ reflected the lag of the K-field rather than the resistance
of the ligament (the ratio of the effective crack resistances of the two linear elastic sets deviated from the ratio of
their $J_c$). In the new simulations, $K_I^{\mathrm{bc}}$ is chosen for every wall width and parameter set such that
$J_{\mathrm{bc}} = J_c$, i.e., $(K_I^{\mathrm{bc}})^2 = 2\mu_{\mathrm{bc}}J_c/(1-\nu_{\mathrm{bc}})$ (Section~4.2). The
applied load is then the same multiple of the microscopic fracture resistance for all wall widths, and the two linear
elastic sets are loaded self-similarly, which is the basis of the scaling argument of Section~4.2 (Reviewer~3,
comment~3). For $w_s \ge L$ the change has no noticeable effect on $J_{x,c}^{\text{eff}}$ (see item~4). The homogeneous
strip of Section~3.5 keeps $K_I^{\mathrm{bc}} = \sqrt{2.5J_c^0\mu_0}$.
\end{enumerate}""")

rep(r"""%@quote cmd:effective elastic constants of the surrounding material, $E_{\text{bc}} = E_0^{\text{eff}}$
%@quote cmd:as in the two-dimensional case, the effective elastic constants""",
    r"""%@quote cmd:effective elastic constants of the surrounding material, $E_{\text{bc}} = E_0^{\text{eff}}$
%@quote cmd:as in the two-dimensional case, the effective elastic constants
%@quote cmd:values are obtained from the plateau of $J_x$
%@quote cmd:$J_c^{\text{num}} \approx 1.425J_c^0$
%@quote cmd:$J_c^{\text{num}} \approx 1.088J_c^0$
%@quote cmd:$J_c^{\text{num}} \approx 0.604J_c^0$
%@quote cmd:In Section~\ref{sec:2Dresults}, the effective crack resistance of each parameter set is compared
%@quote env:\changed{1.088J_c^0}
%@quote cmd:a stress intensity factor $K_I^{\text{bc}}$ that is chosen for every wall width
%@quote cmd:With this choice, the applied load is the same multiple
%@quote cmd:$0.25L$}{$0.375L$}
%@quote cmd:crack resistance}{strength}
%@quote cmd:It is trapped at each pore until re-nucleation
%@quote cmd:The shaded bands indicate the pore positions.
%@quote cmd:, normalized by $J_c^0$ in Fig.
%@quote cmd:Effective crack resistance $J_{x,c}^{\text{eff}}$ of the two-dimensional porous material in units of
%@quote cmd:, see Section~\ref{sec:2D_mechanisms}}
%@quote cmd:, for $w_s \ge 0.6L$,}
%@quote cmd:equal}{almost identical}""")

for old, new in E:
    t = t.replace(old, new, 1)
open(FN, "w", encoding="utf-8").write(t)
print("applied %d edits" % len(E))
