#!/usr/bin/env python3
"""Round-2 edits, part 5 (2026-10-22): Section 4 (2D results) rewritten from the
075_klaim_rev2_2D results, Section 3.5 numbers, new figures/tables.
One-shot: every anchor must occur exactly once in manuscript_revised_template.tex.
"""
import sys

FN = "manuscript_revised_template.tex"
t = open(FN, encoding="utf-8").read()
E = []


def rep(old, new):
    if t.count(old) == 0:  # the editor strips trailing blanks; the manuscript has "observed in \n"
        old = old.replace("observed in \nthe", "observed in \nthe").replace("observed in\nthe", "observed in \nthe")
    n = t.count(old)
    if n != 1:
        raise SystemExit("anchor occurs %d times:\n%s" % (n, old[:200]))
    E.append((old, new))


# ----------------------------------------------------------------- working note
rep(r"""Open (need the 075/076 results): Sections~\ref{sec:2D_energy_partition}--\ref{sec:2D_representativity}, the two tables,
all numbers of Sections 3.5, 4, 5, new figures (minor 25), minor 24 numbers, abstract and conclusion update.}""",
    r"""Done 2026-10-22 (edits\_round2\_part5\_section4.py): Section~4 rewritten from the 075 results (Sections~\ref{sec:2D_energy_partition}--\ref{sec:2D_representativity},
Tables~\ref{tab:effective_stiffness_2D}, \ref{tab:Jeff_2D}, \ref{tab:energy_partition_2D}--\ref{tab:sensitivity_2D}, new vector figures), $J_c^{\text{num}}$ in Section~3.5,
$K_I^{\text{bc}}$ rule in Section~4.2. Open: Section~5 (076 results, table, figures), Fig.~\ref{fig:PAPER_01_all_Jx_vs_xct_pf_2} and the strip geometry of Section~3.5,
abstract and conclusion update.}""")

# ----------------------------------------------------------------- Section 3.5
rep(r"""In the current setup, the following approximate values were observed:
\begin{itemize}
    \item $J_c^{\text{num}} \approx 1.50J_c^0$ for the von Mises material model \vMLabel,
    \item $J_c^{\text{num}} \approx 1.12J_c^0$ for \EqJcLabel,
    \item $J_c^{\text{num}} \approx 0.64J_c^0$ for \EqSigmaStarLabel,
\end{itemize}
It is worth noting""",
    r"""In the current setup, the following \replaced{values are obtained from the plateau of $J_x$}{approximate values were observed}:
\begin{itemize}
    \item \replaced{$J_c^{\text{num}} \approx 1.425J_c^0$}{$J_c^{\text{num}} \approx 1.50J_c^0$} for the von Mises material model \vMLabel,
    \item \replaced{$J_c^{\text{num}} \approx 1.088J_c^0$}{$J_c^{\text{num}} \approx 1.12J_c^0$} for \EqJcLabel,
    \item \replaced{$J_c^{\text{num}} \approx 0.604J_c^0$}{$J_c^{\text{num}} \approx 0.64J_c^0$} for \EqSigmaStarLabel,
\end{itemize}
\mycomment{TODO}{Fig.~\ref{fig:PAPER_01_all_Jx_vs_xct_pf_2} to be regenerated from the S2 runs (S2\_solid\_vM, S2\_strip\_EqJc, S2\_strip\_EqSig); geometry description of the solid reference in Section~3.5 (box without pores) to be updated.}
\added{The ratio of the values of \EqJcLabel and \EqSigmaStarLabel, $1.088/0.604 = 1.80$, agrees with the ratio of the prescribed fracture resistances, $1/0.5523 = 1.81$, as expected for the linear elastic model, whose energies scale with $J_c$, see Section~\ref{sec:2D_fracture_behavior}.}
It is worth noting""")

rep(r"""A reference numerical energetic fracture resistance is defined as the value obtained from parameter set \EqJcLabel:
\begin{equation}
    J_c^{\text{num}} \approx 1.12J_c.
\end{equation}""",
    r"""\replaced{In Section~\ref{sec:2Dresults}, the effective crack resistance of each parameter set is compared with the numerical fracture resistance $J_c^{\text{num}}$ of the same set. For the linear elastic model with $J_c = J_c^0$, i.e., for the parameter set \EqJcLabel, it is}{A reference numerical energetic fracture resistance is defined as the value obtained from parameter set \EqJcLabel:}
\begin{equation}
    J_c^{\text{num}} \approx \changed{1.088J_c^0}.
\end{equation}""")

# label for the stress-driven nucleation subsection and the plastic energy
rep(r"""\subsection{Stress driven crack nucleation}
""", r"""\subsection{Stress driven crack nucleation}
\label{sec:stress_driven_nucleation}
""")
rep(r"""    \psi_{\rm{pl}}(\alpha,s) = g(s)\left(\dfrac{1}{2}H\alpha^2 + \sigma_y\alpha\right).
\end{equation}""", r"""    \psi_{\rm{pl}}(\alpha,s) = g(s)\left(\dfrac{1}{2}H\alpha^2 + \sigma_y\alpha\right).
    \label{eq:psi_pl}
\end{equation}""")

# ----------------------------------------------------------------- Section 4.1 figures
rep(r"""\begin{figure}[htb]
    \centering
    \resizebox{0.4\textwidth}{!}{\input{pics/PAPER_00_xct_pf_vs_xct_KI_vonMises.pgf}}
    \caption{%
        Position of the surfing boundary conditions $x_{\text{bc}}$ and the crack tip position $x_{\text{ct}}$.
        The horizontal axis denotes the pseudo-time $t$ normalized by the time scale $L/\dot{x}_{\mathrm{bc}}$.
    }
    \label{fig:PAPER_00_xct_pf_vs_xct_KI_diamond}
\end{figure}

\begin{figure}[htb]
    \centering
    \includegraphics[width=0.8\textwidth]{pics/PAPER_clean_Jx_vs_xct_SIG_Jc_vonMises.png}
    \caption{%
        J-Integral vs.\ crack tip position $x_{ct}$ for $w_s = 1.0L$.
    }
    \begin{picture}(0,0)
        \put(-89,70){\includegraphics[width=0.425\textwidth]{pics/holes_s_middle_row.png}}
    \end{picture}
    \label{fig:PAPER_02_all_Jx_vs_xct_pf_diamond_holes}
\end{figure}


\begin{figure}[htb]
    \centering
    \begin{minipage}{0.49\textwidth}
        \centering
        \resizebox{\textwidth}{!}{\includegraphics[width=\textwidth]{pics/PAPER_04_Jx_vs_xct_all_vonMises.png}}
        \caption{Section of the J-Integral $J_x$ vs.\ crack tip position $x_{ct}$ between pores two and four for the parameter set \vMLabel and varying $w_s$.}
        \label{fig:PAPER_04_Jx_vs_xct_all_von_Mises}
    \end{minipage}
    \hfill
    \begin{minipage}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{pics/PAPER_03A_Jx_vs_xct_all_linear_elastic_gc10.png}
        \caption{Section of the J-Integral $J_x$ vs.\ crack tip position $x_{ct}$ between pores two and four for the parameter set \EqJcLabel and varying $w_s$.}
        \label{fig:PAPER_03A_Jx_vs_xct_all_linear_elastic_gc10}
    \end{minipage}

    \vspace{3mm}

    \begin{minipage}{0.49\textwidth}
        \centering
        \includegraphics[width=\textwidth]{pics/PAPER_04_Jx_vs_xct_all_SIG.png}
        \caption{Section of the J-Integral $J_x$ vs.\ crack tip position $x_{ct}$ between pores two and four for the parameter set \EqSigmaStarLabel and varying $w_s$.}
        \label{fig:PAPER_03_Jx_vs_xct_all_linear_elastic_gc05679.png}
    \end{minipage}
    \label{fig:side_by_side_plots}
\end{figure}

\begin{figure}[htb]
    \centering
    \resizebox{0.6\textwidth}{!}{\includegraphics[]{pics/PAPER_06b_Jx_vs_wsteg_three_datasets.png}}
    \caption{Effective crack resistance $J_{x,c}^{\text{eff}}$ vs.\ wall width $w_s$.}
    \label{fig:PAPER_06b_Jx_vs_wsteg_three_datasets}
\end{figure}
""",
    r"""\begin{figure}[htb]
    \centering
    \resizebox{0.5\textwidth}{!}{\input{pics/PAPER_rev2_xct_xbc_vs_t_vM_ws1.0.pgf}}% rev2: regenerated from 075 (S1_vM_ws1.0), replaces PAPER_00_xct_pf_vs_xct_KI_vonMises
    \caption{%
        Position of the surfing boundary conditions $x_{\text{bc}}$ and the crack tip position $x_{\text{ct}}$\added{ for $w_s = L$ and the parameter set \vMLabel}.
        The horizontal axis denotes the pseudo-time $t$ normalized by the time scale $L/\dot{x}_{\mathrm{bc}}$.
    }
    \label{fig:PAPER_00_xct_pf_vs_xct_KI_diamond}
\end{figure}

\begin{figure}[htb]
    \centering
    \resizebox{0.85\textwidth}{!}{\input{pics/PAPER_rev2_Jx_vs_xct_ws1.0_three_sets.pgf}}% rev2: regenerated from 075, replaces PAPER_clean_Jx_vs_xct_SIG_Jc_vonMises.png (+ holes_s_middle_row.png inset)
    \caption{%
        J-Integral vs.\ crack tip position $x_{ct}$ for $w_s = 1.0L$\added{ and the three parameter sets. The shaded bands mark the positions of the pores along the crack path}.
    }
    \label{fig:PAPER_02_all_Jx_vs_xct_pf_diamond_holes}
\end{figure}


\begin{figure}[htb]
    \centering
    \begin{minipage}{0.49\textwidth}
        \centering
        \resizebox{\textwidth}{!}{\input{pics/PAPER_rev2_Jx_vs_xct_pores2to4_vM.pgf}}% rev2: replaces PAPER_04_Jx_vs_xct_all_vonMises.png
        \caption{\replaced{J-Integral $J_x$ vs.\ crack tip position $x_{ct}$ between pores two and four, measured from the center $x_{p2}$ of pore two and normalized by the cell width $w_c$,}{Section of the J-Integral $J_x$ vs.\ crack tip position $x_{ct}$ between pores two and four} for the parameter set \vMLabel and varying $w_s$.}
        \label{fig:PAPER_04_Jx_vs_xct_all_von_Mises}
    \end{minipage}
    \hfill
    \begin{minipage}{0.49\textwidth}
        \centering
        \resizebox{\textwidth}{!}{\input{pics/PAPER_rev2_Jx_vs_xct_pores2to4_EqJc.pgf}}% rev2: replaces PAPER_03A_Jx_vs_xct_all_linear_elastic_gc10.png
        \caption{\replaced{J-Integral $J_x$ vs.\ crack tip position $x_{ct}$ between pores two and four, measured from the center $x_{p2}$ of pore two and normalized by the cell width $w_c$,}{Section of the J-Integral $J_x$ vs.\ crack tip position $x_{ct}$ between pores two and four} for the parameter set \EqJcLabel and varying $w_s$.}
        \label{fig:PAPER_03A_Jx_vs_xct_all_linear_elastic_gc10}
    \end{minipage}

    \vspace{3mm}

    \begin{minipage}{0.49\textwidth}
        \centering
        \resizebox{\textwidth}{!}{\input{pics/PAPER_rev2_Jx_vs_xct_pores2to4_EqSig.pgf}}% rev2: replaces PAPER_04_Jx_vs_xct_all_SIG.png
        \caption{\replaced{J-Integral $J_x$ vs.\ crack tip position $x_{ct}$ between pores two and four, measured from the center $x_{p2}$ of pore two and normalized by the cell width $w_c$,}{Section of the J-Integral $J_x$ vs.\ crack tip position $x_{ct}$ between pores two and four} for the parameter set \EqSigmaStarLabel and varying $w_s$.}
        \label{fig:PAPER_03_Jx_vs_xct_all_linear_elastic_gc05679.png}
    \end{minipage}
    \label{fig:side_by_side_plots}
\end{figure}

\begin{figure}[htb]
    \centering
    \resizebox{\textwidth}{!}{\input{pics/PAPER_rev2_Jeff_vs_wsteg_three_sets.pgf}}% rev2: replaces PAPER_06b_Jx_vs_wsteg_three_datasets.png
    \caption{Effective crack resistance $J_{x,c}^{\text{eff}}$ vs.\ wall width $w_s$\added{: a) normalized by the reference fracture resistance $J_c^0$, b) normalized by the numerical fracture resistance $J_c^{\text{num}}$ of the respective parameter set, Section~\ref{sec:steady_growth_griffith}}.}
    \label{fig:PAPER_06b_Jx_vs_wsteg_three_datasets}
\end{figure}
""")

# ----------------------------------------------------------------- Section 4.1 table
rep(r"""The anisotropy is largest for the smallest wall width and vanishes for widely spaced pores. The influence of the isotropic approximation on the effective crack resistance is tested in Section~\ref{sec:2D_cubic_embedding} by simulations in which the embedding is assigned the full square-symmetric stiffness.}
\mycomment{TODO}{complete Table \ref{tab:effective_stiffness_2D} from effective\_stiffness\_table.csv of study S1 (all $w_s$); the three rows given are from the cluster test of 2026-10-15.}""",
    r"""The anisotropy is largest for the smallest wall width and vanishes for widely spaced pores. The effective Poisson's ratio of the isotropic approximation increases from $\nu_0^{\text{eff}} = 0.19$ for $w_s = 0.25L$ to $0.25$ for $w_s = 4L$. The influence of the isotropic approximation on the effective crack resistance is tested in Section~\ref{sec:2D_cubic_embedding} by simulations in which the embedding is assigned the full square-symmetric stiffness.}""")

rep(r"""0.375 & 1.087 & 0.241 & 0.376 & 0.241 & 0.376 & 0.899 & 0.890 \\
1.0   & 1.814 & 0.520 & 0.621 & 0.520 & 0.621 & 1.525 & 0.959 \\
4.0   & 2.740 & 0.907 & 0.915 & 0.907 & 0.915 & 2.285 & 0.998 \\""",
    r"""0.25  & 0.869 & 0.177 & 0.301 & 0.177 & 0.301 & 0.713 & 0.870 \\
0.375 & 1.087 & 0.241 & 0.376 & 0.241 & 0.376 & 0.900 & 0.890 \\
0.4   & 1.126 & 0.254 & 0.390 & 0.254 & 0.390 & 0.934 & 0.894 \\
0.5   & 1.274 & 0.305 & 0.441 & 0.305 & 0.441 & 1.063 & 0.910 \\
0.6   & 1.406 & 0.354 & 0.486 & 0.354 & 0.486 & 1.177 & 0.924 \\
0.75  & 1.579 & 0.422 & 0.544 & 0.422 & 0.544 & 1.326 & 0.941 \\
1.0   & 1.814 & 0.520 & 0.621 & 0.520 & 0.621 & 1.524 & 0.960 \\
1.4   & 2.091 & 0.639 & 0.709 & 0.639 & 0.709 & 1.754 & 0.977 \\
2.0   & 2.361 & 0.755 & 0.794 & 0.755 & 0.794 & 1.975 & 0.989 \\
3.0   & 2.610 & 0.857 & 0.873 & 0.857 & 0.873 & 2.178 & 0.996 \\
4.0   & 2.740 & 0.907 & 0.915 & 0.907 & 0.915 & 2.285 & 0.998 \\""")

# ----------------------------------------------------------------- Section 4.2 text
rep(r"""described in Section~\ref{sec:numericalExperiment}, with $K_I^{\text{bc}} = \sqrt{2.5 J_c^0 \mu_0}$\replaced{ and the effective elastic constants of the surrounding material, $E_{\text{bc}} = E_0^{\text{eff}}$ and $\nu_{\text{bc}} = \nu_0^{\text{eff}}$, as the auxiliary parameters of the prescribed K-field~(\ref{eq:surfing-bcs})}{, $E_{\text{bc}} = 2.5 \mu_0$, and $\nu_{\text{bc}} = 0.25$}. The phase-field parameters are $\epsilon = 0.1L$, $\eta = 10^{-5}$, $M = 1000\,\nicefrac{v_{\text{bc}}}{J_c^0}$, and $\lambda_0 = \mu_0$. Simulations are conducted with wall widths ranging from $0.375L$ to $4.0L$ using the parameter sets introduced in Section~\ref{sec:parameter_sets}.""",
    r"""described in Section~\ref{sec:numericalExperiment}, with \replaced{a stress intensity factor $K_I^{\text{bc}}$ that is chosen for every wall width such that the energy release rate of the prescribed K-field equals the microscopic fracture resistance of the respective parameter set, $J_{\text{bc}} = (1-\nu_{\text{bc}}^2)(K_I^{\text{bc}})^2/E_{\text{bc}} = J_c$, i.e., $(K_I^{\text{bc}})^2 = 2\mu_{\text{bc}}J_c/(1-\nu_{\text{bc}})$,}{$K_I^{\text{bc}} = \sqrt{2.5 J_c^0 \mu_0}$}\replaced{ and the effective elastic constants of the surrounding material, $E_{\text{bc}} = E_0^{\text{eff}}$ and $\nu_{\text{bc}} = \nu_0^{\text{eff}}$, as the auxiliary parameters of the prescribed K-field~(\ref{eq:surfing-bcs})}{, $E_{\text{bc}} = 2.5 \mu_0$, and $\nu_{\text{bc}} = 0.25$}. \added{With this choice, the applied load is the same multiple of the microscopic fracture resistance for all wall widths, and the two linear elastic parameter sets, which differ only in $J_c$, are loaded self-similarly.} The phase-field parameters are $\epsilon = 0.1L$, $\eta = 10^{-5}$, $M = 1000\,\nicefrac{v_{\text{bc}}}{J_c^0}$, and $\lambda_0 = \mu_0$. Simulations are conducted with wall widths ranging from \replaced{$0.25L$}{$0.375L$} to $4.0L$ using the parameter sets introduced in Section~\ref{sec:parameter_sets}.""")

rep(r"""The microscopic crack tip follows the macroscopic boundary condition in an unsteady manner, exhibiting six jumps of length $2L$. These jumps indicate that once a crack becomes trapped inside a pore and re-nucleates, it traverses the material ligament between pores during the instantaneous crack growth phase of nucleation.""",
    r"""The microscopic crack tip follows the macroscopic boundary condition in an unsteady manner\replaced{. It is trapped at each pore until re-nucleation and then crosses the pore and the following ligament in one jump, i.e., the ligament between two pores fails during the instantaneous crack growth phase that follows nucleation.}{, exhibiting six jumps of length $2L$. These jumps indicate that once a crack becomes trapped inside a pore and re-nucleates, it traverses the material ligament between pores during the instantaneous crack growth phase of nucleation.}""")

rep(r"""for a wall width of $1.0L$ and all three parameter sets. Vertical lines indicate pore positions. Aside from""",
    r"""for a wall width of $1.0L$ and all three parameter sets. \replaced{The shaded bands indicate the pore positions.}{Vertical lines indicate pore positions.} Aside from""")
rep(r"""The peak value of $J_x$ therefore occurs directly before re-nucleation and determines the effective strength according to the definition in Section~\ref{sec:numericalExperiment}.""",
    r"""The peak value of $J_x$ therefore occurs directly before re-nucleation and determines the effective \replaced{crack resistance}{strength} according to the definition in Section~\ref{sec:numericalExperiment}.""")

rep(r"""while the latter still yields higher $J_x$ values than the linear elastic simulation with equivalent fracture stress \EqSigmaStarLabel. For this fixed wall width, the ordering can be interpreted through the different resistance against crack re-nucleation and the additional energy dissipated during ligament failure\replaced{. In}{; in} particular, once the one-dimensional tensile strength is largely eliminated as the dominant difference between models, the enhanced dissipation appears to play an important role. The following parameter study shows that this ordering \replaced{reflects}{is representative of} the main trend, while the detailed pore interaction length additionally depends on $w_s$.

For a wall width of $1.0L$, neither the equivalent one-dimensional fracture stress of the bulk material nor the microscopic energetic fracture resistance $J_c$ provides a reliable indicator of the effective crack resistance. Instead, the total dissipated energy $\Pi^*$ observed in
the one-dimensional crack nucleation experiment correlates more closely with the $J_x$ response, correctly predicting that the effective crack resistance of \vMLabel lies between those of \EqJcLabel and \EqSigmaStarLabel\replaced{, see}{; see} also Fig.~\ref{fig:energy_vs_displacement}.""",
    r"""while the latter still yields higher $J_x$ values than the linear elastic simulation with equivalent fracture stress \EqSigmaStarLabel. \replaced{The peak values are $J_{x,c}^{\text{eff}} = 2.02$, $1.56$ and $1.12\,J_c^0$ for \EqJcLabel, \vMLabel and \EqSigmaStarLabel, respectively. The mechanisms behind this ordering, and behind its reversal for thin ligaments, are analyzed in Sections~\ref{sec:2D_energy_partition} and~\ref{sec:2D_mechanisms}.}{For this fixed wall width, the ordering can be interpreted through the different resistance against crack re-nucleation and the additional energy dissipated during ligament failure; in particular, once the one-dimensional tensile strength is largely eliminated as the dominant difference between models, the enhanced dissipation appears to play an important role. The following parameter study shows that this ordering is representative of the main trend, while the detailed pore interaction length additionally depends on $w_s$.}

For a wall width of $1.0L$, neither the equivalent one-dimensional fracture stress of the bulk material nor the microscopic energetic fracture resistance $J_c$ provides a reliable indicator of the effective crack resistance. Instead, the total dissipated energy $\Pi^*$ observed in
the one-dimensional crack nucleation experiment \replaced{has the same ordering as $J_{x,c}^{\text{eff}}$ at this wall width, $\Pi^* = 2.19$, $2.12$ and $1.23\,J_c^0L$ for \EqJcLabel, \vMLabel and \EqSigmaStarLabel, i.e., the effective crack resistance of \vMLabel lies between those of \EqJcLabel and \EqSigmaStarLabel, see also Fig.~\ref{fig:energy_vs_displacement}. This correspondence is limited to the regime of wide ligaments, as shown below.}{correlates more closely with the $J_x$ response, correctly predicting that the effective crack resistance of \vMLabel lies between those of \EqJcLabel and \EqSigmaStarLabel; see also Fig.~\ref{fig:energy_vs_displacement}.}""")

rep(r"""which shows the effective macroscopic crack resistance $J_{x,c}^{\text{eff}}$ versus $w_s$ for all parameter sets. Beyond a critical wall width, $J_{x,c}^{\text{eff}}$ no longer increases, indicating saturation. The saturation values are approximately $1.80 J_c^{\text{num}}$ for \vMLabel, $1.9 J_c^{\text{num}}$ for \EqJcLabel, and $1.12 J_c^{\text{num}}$ for \EqSigmaStarLabel. Thus, a pronounced toughening effect is observed in all cases for wall widths of at least $2L$.

The accumulated work of internal forces until fracture, $\Pi^*$, is again found to be the most reliable indicator of high effective crack resistance, particularly in the saturation regime at larger wall widths. In contrast, equivalent one-dimensional tensile strength or equivalent microscopic fracture energy $J_c$ does not consistently predict effective crack resistance. For wall widths up to $0.6L$, the ductile material law leads to a reduction in effective crack resistance below the values obtained for \EqSigmaStarLabel.

Finally, Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets} shows that saturation occurs at significantly larger wall widths for \vMLabel than for the linear elastic parameter sets. Since $J_{x,c}^{\text{eff}}$ is determined by the peak value reached immediately prior to re-nucleation, saturation implies that re-nucleation is no longer influenced by neighboring pores. The delayed saturation observed for the ductile material law indicates that plasticity enlarges the interaction zone over which neighboring pores affect re-nucleation, due to the formation of extended plastic deformation zones prior to crack nucleation.
The comparison of \vMLabel and \EqSigmaStarLabel in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets} shows that the ductile material law leads to a substantially higher effective crack resistance even though the one-dimensional tensile strengths are almost identical.""",
    r"""which shows the effective macroscopic crack resistance $J_{x,c}^{\text{eff}}$ versus $w_s$ for all parameter sets\added{, normalized by $J_c^0$ in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets}a) and by the numerical fracture resistance $J_c^{\text{num}}$ of the respective parameter set in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets}b). The values are listed in Table~\ref{tab:Jeff_2D}}. \replaced{The increase is steep for thin ligaments and levels off for wide ligaments. At $w_s = 4L$, the effective crack resistance is $1.90J_c^0 = 1.34J_c^{\text{num}}$ for \vMLabel, $2.21J_c^0 = 2.03J_c^{\text{num}}$ for \EqJcLabel and $1.22J_c^0 = 2.03J_c^{\text{num}}$ for \EqSigmaStarLabel. Thus, a pronounced toughening effect with respect to the fracture resistance of the homogeneous material is observed for all parameter sets. Relative to its own $J_c^{\text{num}}$, however, the von~Mises set is toughened less by the pores than the linear elastic sets for all wall widths, Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets}b).}{Beyond a critical wall width, $J_{x,c}^{\text{eff}}$ no longer increases, indicating saturation. The saturation values are approximately $1.80 J_c^{\text{num}}$ for \vMLabel, $1.9 J_c^{\text{num}}$ for \EqJcLabel, and $1.12 J_c^{\text{num}}$ for \EqSigmaStarLabel. Thus, a pronounced toughening effect is observed in all cases for wall widths of at least $2L$.}

\begin{table}[htb]
\centering
\caption{\added{Effective crack resistance $J_{x,c}^{\text{eff}}$ of the two-dimensional porous material in units of $J_c^0$ and of the numerical fracture resistance $J_c^{\text{num}}$ of the respective parameter set ($1.425$, $1.088$ and $0.604\,J_c^0$ for \vMLabel, \EqJcLabel and \EqSigmaStarLabel).}}
\label{tab:Jeff_2D}
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
\end{table}

\replaced{The ordering \EqJcLabel $>$ \vMLabel $>$ \EqSigmaStarLabel observed for $w_s = L$ holds for all wall widths $w_s \ge 0.6L$. For thin walls, $w_s \le 0.5L$, the ductile material law leads to a reduction of the effective crack resistance below the values obtained for \EqSigmaStarLabel. The curves of \vMLabel and \EqSigmaStarLabel cross between $w_s = 0.5L$ and $0.6L$. Neither the one-dimensional tensile strength, which is equal for \vMLabel and \EqSigmaStarLabel, nor the microscopic fracture resistance $J_c$, which is equal for \vMLabel and \EqJcLabel, is therefore a consistent indicator of the effective crack resistance. The two regimes are analyzed in Sections~\ref{sec:2D_energy_partition} and~\ref{sec:2D_mechanisms}.}{The accumulated work of internal forces until fracture, $\Pi^*$, is again found to be the most reliable indicator of high effective crack resistance, particularly in the saturation regime at larger wall widths. In contrast, equivalent one-dimensional tensile strength or equivalent microscopic fracture energy $J_c$ does not consistently predict effective crack resistance. For wall widths up to $0.6L$, the ductile material law leads to a reduction in effective crack resistance below the values obtained for \EqSigmaStarLabel.}

Finally, Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets} shows that saturation occurs at significantly larger wall widths for \vMLabel than for the linear elastic parameter sets. \added{The linear elastic sets reach $95\,\%$ of their value at $w_s = 4L$ for $w_s \approx 1.4L$, the von~Mises set for $w_s \approx 2L$.} Since $J_{x,c}^{\text{eff}}$ is determined by the peak value reached immediately prior to re-nucleation, saturation implies that re-nucleation is no longer influenced by neighboring pores. The delayed saturation observed for the ductile material law indicates that plasticity enlarges the interaction zone over which neighboring pores affect re-nucleation, due to the formation of extended plastic deformation zones prior to crack nucleation\added{, see Section~\ref{sec:2D_mechanisms}}.
The comparison of \vMLabel and \EqSigmaStarLabel in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets} shows that\added{, for $w_s \ge 0.6L$,} the ductile material law leads to a substantially higher effective crack resistance even though the one-dimensional tensile strengths are \replaced{equal}{almost identical}.

\added{The two linear elastic parameter sets differ only in the value of $J_c$. For the linear elastic phase-field model, $J_c$ is the only energy scale of the problem once the elastic constants and the lengths are fixed, and the amplitude of the K-field is proportional to $\sqrt{J_c}$ by the choice of $K_I^{\text{bc}}$. In the rate-independent limit $M \to \infty$, the displacement field therefore scales with $\sqrt{J_c}$ and all energies, including $J_{x,c}^{\text{eff}}$, scale with $J_c$, while the crack pattern is unchanged. The simulations confirm this scaling. The ratio of the effective crack resistances of \EqJcLabel and \EqSigmaStarLabel lies between $1.809$ and $1.810$ for all wall widths, compared with $1/0.5523 = 1.811$ for the prescribed fracture resistances, and the fractions of the energy partition of Section~\ref{sec:2D_energy_partition} agree to two to three digits. The finite mobility $M$ does not disturb the scaling noticeably. In Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets}b), the two linear elastic sets consequently fall onto one curve, which reaches $2.03J_c^{\text{num}}$ at $w_s = 4L$. The scaling also answers the question of how a linear elastic reference set whose numerical fracture resistance equals that of the von~Mises set, $J_c^{\text{num}} = 1.425J_c^0$, would behave. Such a set would reach $J_{x,c}^{\text{eff}} = 2.03 \times 1.425J_c^0 = 2.89J_c^0$ at $w_s = 4L$, i.e., it would be considerably tougher than the von~Mises set with $1.90J_c^0$. The von~Mises material therefore does not behave like a linear elastic material with an increased fracture resistance. The pores toughen it less, relative to its crack resistance in the homogeneous material, than they toughen a linear elastic material.}""")

# ----------------------------------------------------------------- Sections 4.3 - 4.6
rep(r"""\subsection{\added{Energy Partition of the Numerical Experiment}}
\label{sec:2D_energy_partition}
\mycomment{TODO}{R1-M3 / R3-C1: exact energy balance $W = E_{\mathrm{el}} + E_{\mathrm{pl}} + E_{\mathrm{fr}} + D_{\mathrm{visc}}$ of eq.~(\ref{eq:E_ep_total}) with the viscous term, partition vs.\ $w_s$ for the three parameter sets, energy dissipated per re-nucleation event (study S1 outputs: columns E\_el, E\_pl, E\_pl\_undeg, E\_frac, D\_visc, D\_pl\_deg, D\_pl\_undeg).}

\subsection{\added{Stress State and Plastic Zone before Re-Nucleation}}
\label{sec:2D_mechanisms}
\mycomment{TODO}{R1-M3 / R3-C1 / minor 24: ligament stress $\sigma_{yy}$, $\sigma_{\mathrm{vM}}$, hydrostatic stress and $\alpha$ ahead of the trapped crack tip immediately before re-nucleation (event snapshots) for $w_s = 0.375L$ and $1.0L$, \vMLabel vs.\ \EqSigmaStarLabel; comparison with the one-dimensional $\sigma^*$; explanation of the two regimes; sensitivity with respect to $\sigma_y$ and $H$ (study S4); scaling argument for the linear elastic sets.}

\subsection{\added{Influence of the Isotropic Approximation of the Embedding}}
\label{sec:2D_cubic_embedding}
\mycomment{TODO}{R1-M4: results of study S3 (full square-symmetric stiffness in the embedding, $w_s = 0.375L, 1.0L, 4.0L$) compared with the isotropic embedding; corner stresses.}

\subsection{\added{Influence of the Initial Crack and of the Size of the Resolved Region}}
\label{sec:2D_representativity}
\mycomment{TODO}{R1-M2 / minors 10, 11: results of study S5 (initial crack length $1.5 w_c$ and $2 w_c$, five instead of three pore rows) for $w_s = 1.0L$; statement of what was and what was not tested.}
""",
    r"""\subsection{\added{Energy Partition of the Numerical Experiment}}
\label{sec:2D_energy_partition}

\begin{figure}[htb]
    \centering
    \resizebox{\textwidth}{!}{\input{pics/PAPER_rev2_energy_partition_cycle.pgf}}% rev2: new figure from 075 (energy_cycles.csv)
    \caption{\added{Energy partition of one pore-to-pore cycle vs.\ wall width $w_s$. a) Fractions of the energy $\Delta D_{\text{tot}}$ dissipated per cycle, eq.~(\ref{eq:D_tot_cycle}), that create crack surface (fracture), remain stored as degraded plastic energy (plastic) and are dissipated by the viscous evolution of $s$ (viscous). The two linear elastic sets give the same fractions, only \EqSigmaStarLabel is shown. b) Energy dissipated per unit crack advance, $G_{\text{diss}} = \Delta D_{\text{tot}}/w_c$, relative to the effective crack resistance.}}
    \label{fig:energy_partition_cycle}
\end{figure}

\added{The model of Section~\ref{sec:phase_fiel_model_ductile} obeys an exact energy balance, which follows from the time derivative of the total energy, eq.~(\ref{eq:E_ep_total}), together with the equilibrium condition, eq.~(\ref{eq:equilibrium}), and the evolution equation of the phase field, eq.~(\ref{eq:evolution_s}). The work $W_{\text{ext}}$ done by the tractions on the outer boundary, where the surfing displacements are prescribed, is}
\begin{equation}
    \changed{W_{\text{ext}} = E_{\text{el}} + E_{\text{pl}} + E_{\text{fr}} + D_{\text{visc}},
    \quad
    D_{\text{visc}} = \int_0^t \int_\Omega \dfrac{\dot{s}^2}{M}\,\mathrm{d}V\,\mathrm{d}\tilde{t},}
    \label{eq:energy_balance}
\end{equation}
\added{where $E_{\text{el}}$, $E_{\text{pl}}$ and $E_{\text{fr}}$ denote the integrals of $\psi_{\text{el}}$, $\psi_{\text{pl}}$ and $\psi_{\text{fr}}$ over $\Omega$, i.e., the three contributions to $E_{\text{ep}}$ in eq.~(\ref{eq:E_ep_total}), and $D_{\text{visc}}$ is the energy dissipated by the viscous evolution of the phase field. In the present model, the plastic work is stored in the degraded form $\psi_{\text{pl}} = g(s)(\nicefrac{1}{2}H\alpha^2 + \sigma_y\alpha)$, eq.~(\ref{eq:psi_pl}), and acts as a crack driving force through $g'(s)$ in eq.~(\ref{eq:evolution_s}). Plastic work done in material that fails later is released by the degradation and reappears in $E_{\text{fr}}$ and $D_{\text{visc}}$, whereas $E_{\text{pl}}$ records the plastic energy that remains stored in intact material. The viscous term collects the energy that is released when a ligament fails in an unstable manner. In a dynamic formulation, this energy would be converted into kinetic energy. For the linear elastic model, $E_{\text{pl}} = 0$. All terms of eq.~(\ref{eq:energy_balance}) are recorded in every time step of the simulations.}

\added{The partition is evaluated per pore-to-pore cycle. A cycle starts when the crack enters a pore and ends when it enters the next pore. It thus comprises the loading of one ligament while the crack tip is trapped, the re-nucleation and the failure of this ligament. The energy dissipated during a cycle is}
\begin{equation}
    \changed{\Delta D_{\text{tot}} = \Delta E_{\text{fr}} + \Delta E_{\text{pl}} + \Delta D_{\text{visc}},}
    \label{eq:D_tot_cycle}
\end{equation}
\added{and $G_{\text{diss}} = \Delta D_{\text{tot}}/w_c$ is the energy dissipated per unit crack advance, averaged over the cycle. Figure~\ref{fig:energy_partition_cycle} and Table~\ref{tab:energy_partition_2D} show the three fractions of $\Delta D_{\text{tot}}$ and the ratio $G_{\text{diss}}/J_{x,c}^{\text{eff}}$ for the cycle that contains the ligament in which $J_{x,c}^{\text{eff}}$ is attained. The energy dissipated per cycle is nearly the same in all cycles except the last one, which is affected by the end of the resolved region. For \vMLabel and $w_s = L$, for example, $G_{\text{diss}} = 1.30$, $1.36$, $1.39$ and $1.37\,J_c^0$ in the first four cycles, for \vMLabel and $w_s = 0.375L$ it is $0.66$, $0.65$, $0.67$ and $0.65\,J_c^0$, and for \EqSigmaStarLabel and $w_s = L$ it is $0.95$, $1.03$, $1.05$ and $1.05\,J_c^0$. The periodic pattern of $J_x$ in Figs.~\ref{fig:PAPER_02_all_Jx_vs_xct_pf_diamond_holes} to~\ref{fig:PAPER_03_Jx_vs_xct_all_linear_elastic_gc05679.png} thus corresponds to a periodic regime of the energy balance as well. The stored elastic energy decreases over a cycle, by $0.05$ to $0.6\,\Delta D_{\text{tot}}$ depending on the case, because the K-field applied at $t = 0$ stores elastic energy in the domain that is drawn down as the crack advances. The specimen is therefore not in a strict steady state, and the energy dissipated during a cycle can exceed the work supplied by the boundary conditions during the cycle.}

\begin{table}[htb]
\centering
\caption{\added{Energy partition of the cycle that contains the ligament in which $J_{x,c}^{\text{eff}}$ is attained: fractions of the dissipated energy $\Delta D_{\text{tot}}$, eq.~(\ref{eq:D_tot_cycle}), and energy dissipated per unit crack advance $G_{\text{diss}} = \Delta D_{\text{tot}}/w_c$. The fractions of \EqJcLabel equal those of \EqSigmaStarLabel to the digits shown, and its $G_{\text{diss}}$ is larger by the factor $1/0.5523$, see Section~\ref{sec:2D_fracture_behavior}.}}
\label{tab:energy_partition_2D}
\begin{tabular}{lcccccccc}
\toprule
 & \multicolumn{4}{c}{\vMLabel} & \multicolumn{3}{c}{\EqSigmaStarLabel} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-8}
$w_s/L$ & fracture & plastic & viscous & $G_{\text{diss}}/J_c^0$ & fracture & viscous & $G_{\text{diss}}/J_c^0$ \\
\midrule
0.25  & 0.535 & 0.138 & 0.327 & 0.479 & 0.173 & 0.827 & 0.709 \\
0.375 & 0.526 & 0.124 & 0.350 & 0.665 & 0.209 & 0.791 & 0.857 \\
0.6   & 0.493 & 0.141 & 0.366 & 0.989 & 0.235 & 0.765 & 0.989 \\
1.0   & 0.493 & 0.157 & 0.350 & 1.387 & 0.286 & 0.714 & 1.050 \\
2.0   & 0.589 & 0.147 & 0.264 & 1.517 & 0.395 & 0.605 & 0.998 \\
4.0   & 0.744 & 0.135 & 0.121 & 1.385 & 0.530 & 0.470 & 0.889 \\
\bottomrule
\end{tabular}
\end{table}

\added{For the two linear elastic sets, which give the same fractions to two to three digits, the share of the viscous term is large. Between $83\,\%$ of the dissipated energy for $w_s = 0.25L$ and $47\,\%$ for $w_s = 4L$ are lost in the unstable snap-through of the ligament, and only $17$ to $53\,\%$ create crack surface. The elastic energy that the structure stores up to the moment of re-nucleation exceeds the fracture energy of the ligament by far, and the excess is released in the jump. For the von~Mises set, the viscous share is much smaller, between $37\,\%$ for $w_s = 0.6L$ and $12\,\%$ for $w_s = 4L$, the fracture share is $49$ to $74\,\%$, and $12$ to $16\,\%$ of the dissipated energy remain stored as plastic energy in the intact material along the crack path. The difference between the material laws is largest for thin ligaments. The energy dissipated per unit crack advance, $G_{\text{diss}}$, shows the same crossover as $J_{x,c}^{\text{eff}}$. It is smaller for \vMLabel than for \EqSigmaStarLabel for $w_s \le 0.375L$, with $0.479$ vs.\ $0.709\,J_c^0$ at $w_s = 0.25L$, equal at $w_s = 0.6L$, with $0.989\,J_c^0$, and larger for wide ligaments, with $1.39$ vs.\ $0.89\,J_c^0$ at $w_s = 4L$. For the von~Mises set, $G_{\text{diss}}$ reaches its maximum of $1.52\,J_c^0$ at $w_s = 2L$. The ratio $G_{\text{diss}}/J_{x,c}^{\text{eff}}$ in Fig.~\ref{fig:energy_partition_cycle}b) decreases from $1.10$ for $w_s = 0.25L$ to $0.73$ for $w_s = 4L$ for the linear elastic sets and from $0.93$ to $0.73$ for \vMLabel, since $J_{x,c}^{\text{eff}}$ is the peak value of $J_x$ in the cycle, whereas $G_{\text{diss}}$ corresponds to its average. Thus, for thin walls, the ductile material law reduces not only the peak driving force but also the energy dissipated per unit crack advance, which is the quantity closest to an effective fracture energy in the sense of the energy dissipated along the crack path, see Michel and Suquet~\cite{michel_merits_2022}. The mechanism behind this reduction is identified in the next section.}

\subsection{\added{Stress State and Plastic Zone before Re-Nucleation}}
\label{sec:2D_mechanisms}

\begin{figure}[htb]
    \centering
    \resizebox{\textwidth}{!}{\input{pics/PAPER_rev2_ligament_state_before_renucleation.pgf}}% rev2: new figure from 075 (ligament stress records at the J_eff peak)
    \caption{\added{State of the next intact ligament ahead of the trapped crack tip at the peak of $J_x$, i.e., immediately before re-nucleation, vs.\ wall width $w_s$. a) Maximum normal stress $\sigma_{yy,\max}$ in the ligament, normalized by the one-dimensional fracture stress $\sigma^*_{\text{1D}}$ of the respective parameter set, Section~\ref{sec:stress_driven_nucleation}. b) Stress triaxiality $\sigma_h/\sigma_{\text{vM}}$ at the location of $\sigma_{yy,\max}$ and, dotted, the maximum of the plastic variable $\alpha$ in the ligament for \vMLabel.}}
    \label{fig:ligament_state}
\end{figure}

\added{To relate the re-nucleation in the porous specimen to the one-dimensional results of Section~\ref{sec:stress_driven_nucleation}, the stress state in the next intact ligament ahead of the trapped crack tip is recorded in every time step, namely the maxima of the normal stress $\sigma_{yy}$, of the von~Mises stress $\sigma_{\text{vM}}$, of the hydrostatic stress $\sigma_h = \nicefrac{1}{3}\,\text{tr}(\boldsymbol{\sigma})$ and of the plastic variable $\alpha$ in the ligament. Figure~\ref{fig:ligament_state} shows these values at the moment at which $J_x$ attains its peak, i.e., immediately before re-nucleation. The maximum normal stress is normalized by the one-dimensional fracture stress $\sigma^*_{\text{1D}}$ of the respective parameter set, $1.28\sigma_y$ for \vMLabel and \EqSigmaStarLabel and $1.72\sigma_y$ for \EqJcLabel, with $\sigma_y = \sqrt{\nicefrac{J_c^0\mu_0}{L}}$. The ratio $\sigma_h/\sigma_{\text{vM}}$ at the location of the maximum normal stress is the stress triaxiality. The values for \vMLabel are listed in Table~\ref{tab:ligament_state_vM}.}

\begin{table}[htb]
\centering
\caption{\added{Effective crack resistance and state of the next intact ligament immediately before re-nucleation for the parameter set \vMLabel: maxima of the normal stress, the von~Mises stress, the hydrostatic stress and the plastic variable in the ligament, and the triaxiality at the location of $\sigma_{yy,\max}$. Stresses in units of $\sqrt{\nicefrac{J_c^0\mu_0}{L}}$, the one-dimensional fracture stress is $\sigma^*_{\text{vM}} = 1.28$.}}
\label{tab:ligament_state_vM}
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
\end{table}

\added{For the two linear elastic sets, re-nucleation occurs when the maximum normal stress in the ligament reaches the one-dimensional fracture stress, $\sigma_{yy,\max} = 0.95$ to $1.02\,\sigma^*_{\text{1D}}$ for all wall widths, at a triaxiality of $0.49$ to $0.53$. For \EqSigmaStarLabel, $\sigma_{yy,\max} = 1.22$, $1.27$ and $1.27\sqrt{\nicefrac{J_c^0\mu_0}{L}}$ at $w_s = 0.375L$, $L$ and $4L$, for \EqJcLabel, $\sigma_{yy,\max} = 1.65$ and $1.71\sqrt{\nicefrac{J_c^0\mu_0}{L}}$ at $w_s = 0.375L$ and $L$. The nucleation in the ligament is thus stress controlled, and the one-dimensional value is a good estimate of the nucleation stress in the two-dimensional porous specimen. This confirms, for the present geometry and the cubic degradation function, the observations of Kuhn~\cite{kuhn_numerical_2013} and Schl\"uter~\cite{schlueter_phase_2018} referred to in Section~\ref{sec:stress_driven_nucleation}.}

\added{For the von~Mises set, two regimes can be distinguished in Fig.~\ref{fig:ligament_state} and Table~\ref{tab:ligament_state_vM}. For thin ligaments, $w_s \le 0.6L$, the ligament yields across its full width before re-nucleation. The maximum of $\alpha$ is $0.66$ to $0.73$, the triaxiality is low, $0.67$ to $0.74$, and re-nucleation occurs at $\sigma_{yy,\max} = 0.95$ to $1.11\,\sigma^*_{\text{vM}}$, i.e., at the one-dimensional fracture stress after a large plastic strain, as in the one-dimensional bar. The ligament behaves like the bar of Section~\ref{sec:stress_driven_nucleation}, and its nucleation is driven by the accumulated plastic work in the same way. Once the thin ligament has yielded, its tangent stiffness drops to the hardening modulus, so that its opening grows rapidly with the advancing K-field while its stress rises only slowly from $\sigma_y$ to $\sigma^*_{\text{vM}} = 1.28\sigma_y$. The macroscopic driving force at re-nucleation is therefore close to the value at which the ligament yields, and it is lower than the value at which the linear elastic ligament of equal strength, \EqSigmaStarLabel, reaches $\sigma^*$ elastically. The plastic work spent on stretching the ligament to its nucleation state is supplied by the boundary conditions, but it does not enter $J_x$, which measures the energy release rate of the elastic surroundings, and it is not available as crack resistance. In this nucleation regime, the ductile material law reduces the effective crack resistance below that of \EqSigmaStarLabel, Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets}, and the energy dissipated per unit crack advance is reduced as well, Section~\ref{sec:2D_energy_partition}, because the yielded ligament stores less elastic energy up to failure than the linear elastic ligament, which is loaded to a higher driving force before it fails.}

\added{For wide ligaments, $w_s \ge L$, the plastic zone does not span the ligament. The maximum of $\alpha$ drops to $0.44$ to $0.60$, the trapped crack tip at the pore surface acts as a notch, and the material ahead of it is constrained by the surrounding elastic material. The triaxiality rises to $1.0$ to $1.1$, and the von~Mises stress at the location of the maximum normal stress remains at the yield stress, $\sigma_{\text{vM}} = 0.98$ to $1.06\,\sigma_y$, while the normal stress reaches $\sigma_{yy,\max} = 1.67$ to $1.71\sqrt{\nicefrac{J_c^0\mu_0}{L}} = 1.31$ to $1.34\,\sigma^*_{\text{vM}}$. Since the hydrostatic part of the stress does not contribute to the yield condition, the constrained ligament carries a normal stress well above the one-dimensional fracture stress before the crack driving force suffices for re-nucleation. The nucleation stress is at the level of the one-dimensional fracture stress of \EqJcLabel, $1.72\sigma_y$. Re-nucleation is delayed, and the plastic work ahead of the trapped tip has to be supplied before the crack can advance. In this propagation regime, the ductile material law raises the effective crack resistance above that of \EqSigmaStarLabel. The interaction length between neighboring pores is then set by the size of the plastic zone ahead of the trapped tip rather than by the pore diameter, which is consistent with the later saturation of the von~Mises set in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets}.}

\begin{table}[htb]
\centering
\caption{\added{Sensitivity of the effective crack resistance of the parameter set \vMLabel with respect to the yield stress $\sigma_y$ and the hardening modulus $H$, in the nucleation regime ($w_s = 0.375L$) and in the propagation regime ($w_s = L$). Reference values $\sigma_y = 1.0\sqrt{\nicefrac{J_c^0\mu_0}{L}}$ and $H = 0.22\mu_0$, all other parameters unchanged.}}
\label{tab:sensitivity_2D}
\begin{tabular}{lcccc}
\toprule
 & \multicolumn{2}{c}{$w_s = 0.375L$} & \multicolumn{2}{c}{$w_s = L$} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
 & $J_{x,c}^{\text{eff}}/J_c^0$ & change & $J_{x,c}^{\text{eff}}/J_c^0$ & change \\
\midrule
reference                                        & 0.718 & --        & 1.563 & --       \\
$\sigma_y = 0.75\sqrt{\nicefrac{J_c^0\mu_0}{L}}$ & 0.700 & $-2.4\,\%$  & 1.474 & $-5.7\,\%$ \\
$\sigma_y = 1.5\sqrt{\nicefrac{J_c^0\mu_0}{L}}$  & 0.981 & $+36.7\,\%$ & 1.608 & $+2.9\,\%$ \\
$H = 0.11\mu_0$                                  & 0.633 & $-11.9\,\%$ & 1.493 & $-4.4\,\%$ \\
$H = 0.44\mu_0$                                  & 0.853 & $+18.9\,\%$ & 1.658 & $+6.1\,\%$ \\
\bottomrule
\end{tabular}
\end{table}

\added{The sensitivity of $J_{x,c}^{\text{eff}}$ with respect to the plastic parameters confirms the two regimes, Table~\ref{tab:sensitivity_2D}. Halving or doubling the hardening modulus changes the effective crack resistance by $-12\,\%$ and $+19\,\%$ at $w_s = 0.375L$, but only by $-4\,\%$ and $+6\,\%$ at $w_s = L$. Increasing the yield stress by $50\,\%$ raises $J_{x,c}^{\text{eff}}$ by $37\,\%$ at $w_s = 0.375L$ and by $3\,\%$ at $w_s = L$, whereas a reduction of the yield stress by $25\,\%$ changes it by $-2\,\%$ and $-6\,\%$. In the nucleation regime, the effective crack resistance is governed by the stress-strain response of the yielded ligament up to its nucleation state and is therefore sensitive to $\sigma_y$ and $H$. In the propagation regime, it is governed by the re-nucleation at the constrained notch and is much less sensitive to both parameters. For the linear elastic sets, the corresponding sensitivity is the scaling with $J_c$ discussed in Section~\ref{sec:2D_fracture_behavior}, which is confirmed by the simulations to $0.1\,\%$.}

\subsection{\added{Influence of the Isotropic Approximation of the Embedding}}
\label{sec:2D_cubic_embedding}

\added{The embedding of the resolved porous region is assigned the isotropic approximation, eq.~(\ref{eq:isotropic_approximation_2D}), of the square-symmetric effective stiffness of Table~\ref{tab:effective_stiffness_2D}. To test the influence of this approximation, the simulations for $w_s = 0.375L$, $L$ and $4L$ with the parameter set \vMLabel were repeated with the full square-symmetric stiffness, $C_{11}$, $C_{12}$ and $C_{33}$, assigned to the embedding. The effective crack resistance changes by $+0.18\,\%$ for $w_s = 0.375L$ ($A = 0.890$), by $+0.08\,\%$ for $w_s = L$ ($A = 0.960$) and by $+0.01\,\%$ for $w_s = 4L$ ($A = 0.998$). The influence of the anisotropy of the embedding on $J_{x,c}^{\text{eff}}$ is thus negligible for all wall widths considered, including the thinnest walls, for which the anisotropy is largest. This is plausible, since the embedding transmits the K-field to the resolved region and the J-integral is evaluated on the outer boundary, whereas the re-nucleation that determines $J_{x,c}^{\text{eff}}$ takes place in the ligaments of the resolved region, whose stiffness is resolved explicitly. The three-dimensional unit cells of Section~\ref{sec:3D_effective_medium} are closer to isotropic than the two-dimensional ones, with Zener ratios $A \ge 0.978$ for $w_s \ge 0.375L$, so that the approximation is of even less consequence there.}

\added{The corners of the resolved region, where the material discontinuity between the porous region and the homogeneous embedding meets a geometric corner, act as stress concentrators, see Section~\ref{sec:numericalExperiment}. In none of the simulations of this work did a crack nucleate at a corner of the resolved region. The crack pattern remained confined to the central row of pores, Fig.~\ref{fig:setup_crack_growth}b). Since the J-integral is evaluated on the outer boundary of the embedding and the embedding is linear elastic, the local stress concentration at the corners does not enter the recorded driving force.}

\subsection{\added{Influence of the Initial Crack and of the Size of the Resolved Region}}
\label{sec:2D_representativity}

\added{The definition of $J_{x,c}^{\text{eff}}$ in Section~\ref{sec:numericalExperiment} refers to one initial crack and one resolved region. Two of the dependencies that can be tested within the present setup were probed. First, the length of the initial crack was increased from $w_c$ to $1.5w_c$ and $2w_c$ for \vMLabel and $w_s = L$, which changes $J_{x,c}^{\text{eff}}$ by $+0.02\,\%$ and $+0.08\,\%$. Second, the resolved region was enlarged from three to five rows of pores in the $y$-direction, which changes $J_{x,c}^{\text{eff}}$ by $-0.02\,\%$ for \vMLabel and $w_s = L$, by $+1.1\,\%$ for \vMLabel and $w_s = 0.375L$ and by $-0.09\,\%$ for \EqSigmaStarLabel and $w_s = L$. The effective crack resistance is thus independent of the length of the initial crack and of the number of resolved pore rows to within about one per cent. In addition, the periodicity of the response along the crack path supports the choice of the evaluation window. The peaks of $J_x$ in successive ligaments, Fig.~\ref{fig:PAPER_02_all_Jx_vs_xct_pf_diamond_holes}, and the energy dissipated per cycle, Section~\ref{sec:2D_energy_partition}, are constant from the second cycle on, apart from the last cycle at the end of the resolved region.}

\added{These tests do not establish representativity in the sense of Hill, see Section~\ref{sec:numericalExperiment}. Not tested were the influence of the length of the resolved region in the crack-growth direction, of other crack-growth directions relative to the pore lattice, of the position of the initial crack relative to the pores, and of the microstructure itself, since one periodic arrangement of equal pores is considered and no statistics over microstructure realizations is attempted. The amplitude of the applied K-field is fixed by the rule of Section~\ref{sec:2D_fracture_behavior} and was not varied. The conclusions of this work are therefore restricted to the stated microstructure, crack-growth direction and loading.}
""")

# ----------------------------------------------------------------- Section 5 / conclusion: short notes only
rep(r"""numerical fracture resistance of the von Mises-type material law $J_x^{\text{num,pl}}\approx1.50J_c^0$\replaced{, see}{; see} Section~\ref{sec:steady_growth_griffith}.""",
    r"""numerical fracture resistance of the von Mises-type material law $J_x^{\text{num,pl}}\approx1.50J_c^0$\replaced{, see}{; see} Section~\ref{sec:steady_growth_griffith}.\mycomment{TODO}{Section~5: normalize with the three-dimensional solid reference $J_c^{\text{num,3D}}$ of the same mesh (076 results), not with $1.50J_c^0$ (2D value now $1.425J_c^0$).}""")
rep(r"""\section{Conclusion and Outlook}

In this work, we investigated""",
    r"""\section{Conclusion and Outlook}
\mycomment{TODO}{Abstract and conclusion to be aligned with Sections~\ref{sec:2D_fracture_behavior}--\ref{sec:2D_representativity}: two regimes (nucleation regime $w_s \le 0.5L$ with reduced crack resistance, propagation regime), toughening relative to the own $J_c^{\text{num}}$ smaller for \vMLabel than for the linear elastic sets, energy partition, ``predicted'' wording; and with the Section~5 results.}

In this work, we investigated""")

for old, new in E:
    t = t.replace(old, new, 1)
open(FN, "w", encoding="utf-8").write(t)
print("applied %d edits" % len(E))
