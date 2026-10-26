#!/usr/bin/env python3
"""Round-2 edit script, part 7 (2026-10-23): Section 5 (3D results) rewritten from the fixed-mesh series of
076_klaim_rev2_3D (S1F porous h = 0.1L, S2F/S9F solid references, S9F h = 0.07L convergence check) and the 3D parts of the
response letter (R1 major 4, minor 25, R3 comment 3, "Additional corrections" items 4 and 9-11, quote anchors).
One-shot: fails on the first anchor that is gone. Letter anchors (%@quote) of earlier parts are preserved.
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

# ============================================================================ manuscript
# ---------------------------------------------------------------- Section 3.3: Delta h <= eps/2 holds in 2D only
ms.rep(
    r"the mesh is locally refined such that the element edge length $\Delta h$ is at most half the regularization length $\epsilon$, and it is usually chosen much smaller. The specific values of $\Delta h$ and $\epsilon$ used in the individual simulations are stated together with the corresponding numerical setups below.",
    r"the mesh is locally refined such that the element edge length $\Delta h$ is at most half the regularization length $\epsilon$ in two dimensions, and it is usually chosen much smaller. In three dimensions, the element size in the refined band is of the order of $\epsilon$, and the mesh dependence of the result is quantified in Section~\ref{sec:3D_fracture_behavior}. The specific values of $\Delta h$ and $\epsilon$ used in the individual simulations are stated together with the corresponding numerical setups below.",
)

# ---------------------------------------------------------------- Revision Working Note
ms.rep(
    r"Open: Section~5 (076 results, table, figures), Fig.~\ref{fig:PAPER_01_all_Jx_vs_xct_pf_2} and the strip geometry of Section~3.5,"
    "\n" r"abstract and conclusion update.}",
    r"Done 2026-10-23 (edits\_round2\_part7\_section5.py): Section~5 from the fixed-mesh 3D series of 076 (Table~\ref{tab:effective_stiffness_3D}, Figs.~\ref{fig:position_3D}--\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}, normalization by the 3D solid reference, 3D energy partition). Open: Fig.~\ref{fig:PAPER_01_all_Jx_vs_xct_pf_2} and the strip geometry of Section~3.5,"
    "\n" r"abstract and conclusion update, rendering of a fixed-mesh 3D run for Fig.~\ref{fig:mesh3D}, decision on the $J_{x,c}^{\text{eff}}$ window for thin walls in 3D (red note in Section~\ref{sec:3D_fracture_behavior}).}",
)

# ---------------------------------------------------------------- 5.1 figures: mesh (kept, qualified), fields (dropped)
ms.rep(
    r"        \caption{Three-dimensional mesh for $w_s=0.25L$.}" "\n"
    r"        \label{fig:mesh3D}",
    r"        \caption{\replaced{Three-dimensional mesh, illustration from a simulation on a coarser mesh for $w_s = 0.25L$, which is not part of the series evaluated below. The meshes of the present series use the same refinement band around the crack plane $y = 0$ with the fixed element size stated in the text.}{Three-dimensional mesh for $w_s=0.25L$.}}" "\n"
    r"        \mycomment{TODO}{replace by a rendering of the fixed-mesh run S1F, $w_s = 0.375L$ (mesh and crack surface $s$ around the pores, which would also illustrate the bypassing of the pores described in Section~\ref{sec:3D_fracture_behavior}).}" "\n"
    r"        \label{fig:mesh3D}",
)
OLD_FIELD_FIG = r"""\begin{figure}[t]
\centering
\begin{tabular}{c c}
\llap{{a)}\hspace{0mm}}%
\includegraphics[width=0.51\textwidth]{pics/s_t_2.9998_wsteg_0.25.png} &
\llap{{b)}\hspace{0mm}}%
\includegraphics[width=0.51\textwidth]{pics/alpha_t_2.9998_wsteg_0.25.png} \\
\multicolumn{2}{c}{\small $t = 2.998 \nicefrac{L}{\dot{x}_{\rm{bc}}}$}
\end{tabular}
\caption{a) Evolution of the phase field $s$ and b) evolution of the internal variable $\alpha$ for $w_s=0.25L$.}
\label{fig:one_row_two_cols}
\begin{picture}(0,0)
    \put(-33,186){\small$s$}
    \put(168,186){\small$\alpha$}
\end{picture}
\end{figure}
"""
NEW_FIELD_FIG = r"""% rev2 (2026-10-23): figure of the fields s and alpha for w_s = 0.25L removed. It came from a simulation of the previous
% version (coarser w_s-scaled mesh, eta = 1e-3, beta = 0.1, w_s = 0.25L), which is not part of the fixed-mesh series; the
% statement it illustrated (continuous crack advance around the pores) is now carried by Fig. position_3D and the text.
% Re-enable with renderings of the fixed-mesh run S1F if wanted (caption below).
% \begin{figure}[t]
% \centering
% \begin{tabular}{c c}
% \llap{{a)}\hspace{0mm}}%
% \includegraphics[width=0.51\textwidth]{pics/s_t_2.9998_wsteg_0.25.png} &
% \llap{{b)}\hspace{0mm}}%
% \includegraphics[width=0.51\textwidth]{pics/alpha_t_2.9998_wsteg_0.25.png} \\
% \multicolumn{2}{c}{\small $t = 2.998 \nicefrac{L}{\dot{x}_{\rm{bc}}}$}
% \end{tabular}
% \caption{a) Evolution of the phase field $s$ and b) evolution of the internal variable $\alpha$ for $w_s=0.25L$.}
% \label{fig:one_row_two_cols}
% \begin{picture}(0,0)
%     \put(-33,186){\small$s$}
%     \put(168,186){\small$\alpha$}
% \end{picture}
% \end{figure}
"""
ms.rep(OLD_FIELD_FIG, NEW_FIELD_FIG)

# ---------------------------------------------------------------- 5.1 -> 5.2: the four result figures (new files, new captions)
OLD_RESULT_FIGS = r"""\begin{figure}[htb]
    \centering
    \resizebox{0.4\textwidth}{!}{\input{pics/PAPER_VM3D_xct_vs_time.pgf}}
    \caption{Position of the surfing boundary conditions $x_{\text{bc}}$ and crack tip position $x_{\text{ct}}$ for $w_s=1.0L$ and the parameter set \vMLabel.}
    \label{fig:position_3D}
\end{figure}

\begin{figure}[htb]
    \centering
    \includegraphics[width=0.8\textwidth]{pics/PAPER_VM_Jx_vs_xct_2D_vs_3D.png}
    \caption{J-Integral vs.\ crack tip position $x_{ct}$ for $w_s=1.0L$.}
    \begin{picture}(0,0)
        \put(-85,60){\includegraphics[width=0.57\textwidth]{pics/holes_s_middle_row.png}}
    \end{picture}
    \label{fig:PAPER_VM_Jx_all_3D}
\end{figure}

\begin{figure}[htb]
    \centering
    \resizebox{0.95\textwidth}{!}{\includegraphics[]{pics/PAPER_VM3D_Jx_vs_xct_all_wsteg.png}}
    \caption{Section of the J-Integral $J_x$ vs.\ crack tip position $x_{ct}$ between pore two and four for the three-dimensional case using the parameter set \vMLabel and varying $w_s$.}
    \label{fig:test12q2_3D}
\end{figure}

\begin{figure}[htb]
    \centering
    \resizebox{0.6\textwidth}{!}{\includegraphics[]{pics/PAPER_VM_Jeff_vs_wsteg_2D_vs_3D.png}}
    \caption{Effective crack resistance $J_{x,c}^{\text{eff}}$ vs.\ wall width $w_s$.}
    \label{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}
\end{figure}

"""
ms.rep(OLD_RESULT_FIGS, "")

NEW_RESULT_FIGS = r"""
\begin{figure}[htb]
    \centering
    \resizebox{0.45\textwidth}{!}{\input{pics/PAPER_rev2_3D_xct_xbc_vs_t_ws1.0.pgf}}% rev2: regenerated from 076 (S1F_3D_vM_ws1.0), replaces PAPER_VM3D_xct_vs_time.pgf
    \caption{Position of the surfing boundary conditions $x_{\text{bc}}$ and crack tip position $x_{\text{ct}}$ for $w_s=1.0L$ and the parameter set \vMLabel\added{, three-dimensional model}.}
    \label{fig:position_3D}
\end{figure}

\begin{figure}[htb]
    \centering
    \includegraphics[width=0.72\textwidth]{pics/PAPER_rev2_3D_Jx_vs_xct_2D_vs_3D_ws1.0.pdf}% rev2: regenerated from 075/076, replaces PAPER_VM_Jx_vs_xct_2D_vs_3D.png (+ holes_s_middle_row.png overlay, now part of the figure)
    \caption{\replaced{J-Integral $J_x$ vs.\ crack tip position $x_{ct}$ for $w_s = L$ and \vMLabel in the two- and the three-dimensional model, each normalized by the numerical fracture resistance $J_c^{\text{num}}$ of the homogeneous material of the respective dimension. Dashed lines mark the pore edges of the three-dimensional specimen (four pores). The picture at the top shows the phase field $s$ of the middle pore row of the two-dimensional specimen after complete separation, Fig.~\ref{fig:PAPER_02_all_Jx_vs_xct_pf_diamond_holes}, for orientation.}{J-Integral vs.\ crack tip position $x_{ct}$ for $w_s=1.0L$.}}
    \label{fig:PAPER_VM_Jx_all_3D}
\end{figure}

\begin{figure}[htb]
    \centering
    \includegraphics[width=0.9\textwidth]{pics/PAPER_rev2_3D_Jx_vs_xct_sections.pdf}% rev2: regenerated from 076, replaces PAPER_VM3D_Jx_vs_xct_all_wsteg.png
    \caption{\replaced{J-Integral $J_x$ vs.\ crack tip position $x_{ct}$ between pores one and three of the three-dimensional specimen for the parameter set \vMLabel and varying $w_s$, normalized by $J_c^{\text{num}}$ of the three-dimensional homogeneous material.}{Section of the J-Integral $J_x$ vs.\ crack tip position $x_{ct}$ between pore two and four for the three-dimensional case using the parameter set \vMLabel and varying $w_s$.}}
    \label{fig:test12q2_3D}
\end{figure}

\begin{figure}[htb]
    \centering
    \resizebox{0.9\textwidth}{!}{\input{pics/PAPER_rev2_3D_Jeff_and_partition_2D_vs_3D.pgf}}% rev2: regenerated from 075/076, replaces PAPER_VM_Jeff_vs_wsteg_2D_vs_3D.png
    \caption{\replaced{a) Effective crack resistance $J_{x,c}^{\text{eff}}$ vs.\ wall width $w_s$ for the parameter set \vMLabel in the two- and the three-dimensional model, normalized by the numerical fracture resistance $J_c^{\text{num}}$ of the respective dimension. The open triangle marks the three-dimensional simulation on the finer mesh, $h = 0.07L$. b) Fractions of the energy dissipated per pore-to-pore cycle, eq.~(\ref{eq:D_tot_cycle}), that create crack surface (fracture) and that are dissipated by the viscous evolution of $s$ (viscous) in the two- and the three-dimensional model.}{Effective crack resistance $J_{x,c}^{\text{eff}}$ vs.\ wall width $w_s$.}}
    \label{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}
\end{figure}
"""
ms.rep(
    r"\subsection{Fracture Behavior and Comparison with the Two-Dimensional Model}" "\n" r"\label{sec:3D_fracture_behavior}" "\n",
    r"\subsection{Fracture Behavior and Comparison with the Two-Dimensional Model}" "\n" r"\label{sec:3D_fracture_behavior}" "\n" + NEW_RESULT_FIGS,
)

# ---------------------------------------------------------------- 5.1 text: Table 4, Zener paragraph
ms.rep(
    r"\mycomment{TODO}{complete Table \ref{tab:effective_stiffness_3D} from study S1\_3D; influence of the cubic embedding (study S3\_3D, $w_s = 0.375L, 1.0L$) to be reported in Section~\ref{sec:3D_fracture_behavior}.}",
    r"\added{Table~\ref{tab:effective_stiffness_3D} lists the effective constants for the five wall widths considered in this section. The three-dimensional cells are closer to isotropic than the two-dimensional ones, $A \ge 0.978$ compared with $A = 0.890$ for $w_s = 0.375L$ in Table~\ref{tab:effective_stiffness_2D}, and the effective Poisson's ratio of the isotropic approximation increases from $\nu_0^{\text{eff}} = 0.232$ to $0.249$. The influence of the isotropic approximation of the embedding on the effective crack resistance is discussed in Section~\ref{sec:2D_cubic_embedding}.}",
)
ms.rep(
    r"\multicolumn{8}{c}{\mycomment{TODO}{values from study S1\_3D}} \\",
    "0.375 & 2.046 & 0.608 & 0.703 & 0.608 & 0.703 & 1.732 & 0.978 \\\\\n"
    "0.5   & 2.230 & 0.682 & 0.762 & 0.682 & 0.762 & 1.884 & 0.984 \\\\\n"
    "0.75  & 2.483 & 0.787 & 0.841 & 0.787 & 0.841 & 2.090 & 0.992 \\\\\n"
    "1.0   & 2.640 & 0.853 & 0.890 & 0.853 & 0.890 & 2.216 & 0.996 \\\\\n"
    "2.0   & 2.889 & 0.955 & 0.966 & 0.955 & 0.966 & 2.413 & 1.000 \\\\",
)
ms.rep(
    r"\caption{\added{Effective elastic constants of the three-dimensional unit cells (in units of $\mu_0$, $\lambda_0 = \mu_0$), the isotropic approximation and the Zener ratio $A$.}}",
    r"\caption{\added{Effective elastic constants of the three-dimensional unit cells (in units of $\mu_0$, $\lambda_0 = \mu_0$), the isotropic approximation and the Zener ratio $A$ for the five wall widths of the three-dimensional series.}}",
)

# ---------------------------------------------------------------- 5.1 text: geometry, mesh, initial crack
ms.rep(
    r"A $6\times3\times2$ array of these unit cells is assembled to represent a simplified porous material. This array is centered about the plane $y=0$ and embedded into a surrounding cuboidal domain of effective stiffness with dimensions $(8 w_c)\times 20L \times (4 w_c)$. Thus, the unit cell used to determine the effective stiffness is cubic, whereas the surrounding homogeneous domain is a larger cuboidal embedding region assigned this effective stiffness. The mesh, consisting of bilinear tetrahedral finite elements, is locally refined around the expected fracture plane $y=0$, as shown in Fig.~\ref{fig:mesh3D}.",
    r"A \replaced{$4\times3\times2$}{$6\times3\times2$} array of these unit cells is assembled to represent a simplified porous material. This array is centered about the plane $y=0$ and embedded into a surrounding cuboidal domain of effective stiffness with dimensions \replaced{$(6 w_c)\times 20L \times (2.5 w_c)$, i.e., one cell width of embedding before and after the porous block in the $x$-direction and $w_c/4$ on each side in the $z$-direction}{$(8 w_c)\times 20L \times (4 w_c)$}. Thus, the unit cell used to determine the effective stiffness is cubic, whereas the surrounding homogeneous domain is a larger cuboidal embedding region assigned this effective stiffness. \replaced{The mesh of tetrahedral finite elements is refined in a band around the expected fracture plane $y=0$, Fig.~\ref{fig:mesh3D}, with a target element size of $h = 0.1L$ for $|y| < 0.35L$, $2h$ for $|y| < 0.65L$ and at most $L$ in the remaining domain, independent of the wall width. The median edge length of the elements in the crack band is $1.49\epsilon$, and the meshes contain $85\,000$ ($w_s = 0.375L$) to $416\,000$ ($w_s = 2L$) elements. For $w_s = L$, the simulation was repeated with $h = 0.07L$ (median edge length $1.03\epsilon$, $505\,000$ elements) to check the mesh dependence.}{The mesh, consisting of bilinear tetrahedral finite elements, is locally refined around the expected fracture plane $y=0$, as shown in Fig.~\ref{fig:mesh3D}.}",
)
ms.rep(
    r"An initial crack is prescribed in the plane $y=0$ from $x=x_{\rm min}$ to $x=x_{\rm min}+w_c$, where $x_{\rm min}$ denotes the minimum $x$-coordinate of the mesh. Along the expected crack path, the finite element edge length is set to $\Delta h = 0.02L$.",
    r"An initial crack is prescribed in the plane $y=0$ from $x=x_{\rm min}$ to $x=x_{\rm min}+w_c$, where $x_{\rm min}$ denotes the minimum $x$-coordinate of the mesh. \deleted{Along the expected crack path, the finite element edge length is set to $\Delta h = 0.02L$.}",
)

# ---------------------------------------------------------------- 5.1 text: K-field amplitude, wall widths, time stepping
ms.rep(
    r"$z=z_{\rm max}$ to mimic} plane strain while still accounting for the three-dimensional character of the geometry of the microstructure.",
    r"$z=z_{\rm max}$ to mimic} plane strain while still accounting for the three-dimensional character of the geometry of the microstructure. \added{The amplitude of the K-field is not adapted to the wall width here. Its energy release rate, $J_{\text{bc}} = (1-\nu_{\text{bc}}^2)(K_I^{\text{bc}})^2/E_{\text{bc}}$, is $1.37$, $1.25$, $1.13$, $1.06$ and $0.97\,J_c^0$ for $w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and $2L$, which is close to the effective crack resistances found below (ratio $0.7$ to $1.2$).}",
)
ms.rep(
    r"Simulations are conducted with wall widths ranging from $0.25L$ to $4.0L$ using the parameter set \vMLabel.",
    r"Simulations are conducted with wall widths \replaced{$w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and $2L$}{ranging from $0.25L$ to $4.0L$} using the parameter set \vMLabel. \added{The time step is controlled as in Section~\ref{sec:numerical_implementation}, except that it is increased when a step converges in fewer than five, for the longest runs seven, Newton iterations, since the plastic ligaments require five to six iterations per step. For $w_s = L$, a repetition with the threshold seven instead of five changes $J_{x,c}^{\text{eff}}$ by $+0.5\,\%$ and the record of $J_x$ by less than $0.1\,\%$ in the median, with $37\,\%$ fewer time steps.}",
)

# ---------------------------------------------------------------- 5.2 text
ms.rep(
    r"Figures~\ref{fig:one_row_two_cols}a) and b) illustrate the phase-field variable $s$ and the plastic variable $\alpha$, respectively, for $w_s=0.25L$. The crack propagates through the central row of pores and is able to grow continuously between pores, in contrast to the two-dimensional case. Consequently, the crack tip position $x_{ct}$, identified as the node with the largest $x$-coordinate that fulfills $s=0$, evolves more smoothly, as confirmed by Fig.~\ref{fig:position_3D}.",
    r"\replaced{In all simulations, the crack propagates through the central row of pores. Figure~\ref{fig:position_3D} shows the crack tip position $x_{ct}$, identified as the node with the largest $x$-coordinate that fulfills $s=0$, together with the position $x_{\text{bc}}$ of the K-field for $w_s = L$. In contrast to the two-dimensional case, Fig.~\ref{fig:PAPER_00_xct_pf_vs_xct_KI_diamond}, the crack tip follows the K-field nearly continuously, with small steps at the pores instead of the jumps of about $1.2L$ across pore and ligament. The crack front advances by at most about $0.2L$ per time step. Since $x_{ct}$ passes through the $x$-range of each pore without a jump, the front separates the material beside the sphere while it passes the pore, i.e., it bypasses the spherical pores instead of being trapped as a whole, and the ligament between two pores is separated progressively rather than at once.}{Figures~15a) and b) illustrate the phase-field variable $s$ and the plastic variable $\alpha$, respectively, for $w_s=0.25L$. The crack propagates through the central row of pores and is able to grow continuously between pores, in contrast to the two-dimensional case. Consequently, the crack tip position $x_{ct}$, identified as the node with the largest $x$-coordinate that fulfills $s=0$, evolves more smoothly, as confirmed by Fig.~\ref{fig:position_3D}.}",
)
ms.rep(
    r"    J_x = \dfrac{1}{4 w_c}\int_{\partial\Omega}\left(\psi_{\text{el}}(\boldsymbol{\varepsilon})\boldsymbol{1} - (\nabla\boldsymbol{u})^T\boldsymbol{\sigma}\right)\boldsymbol{n}\,\mathrm{d}A\cdot\boldsymbol{e}_x" "\n" r"\end{equation}",
    r"    J_x = \dfrac{1}{\changed{2.5 w_c}}\int_{\partial\Omega}\left(\psi_{\text{el}}(\boldsymbol{\varepsilon})\boldsymbol{1} - (\nabla\boldsymbol{u})^T\boldsymbol{\sigma}\right)\boldsymbol{n}\,\mathrm{d}A\cdot\boldsymbol{e}_x\changed{.}" "\n" r"    \label{eq:J_3D}" "\n" r"\end{equation}",
)
ms.rep(
    r"Since only the \vMLabel dataset is considered in this section and it is the intention of the following discussion" "\n"
    r"to highlight the effect of the geometric representation of microstructure and " "\n"
    r"not of the material law, the records of $J_x$ are normalized by the " "\n"
    r"numerical fracture resistance of the von Mises-type material law $J_x^{\text{num,pl}}\approx1.50J_c^0$\replaced{, see}{; see} Section~\ref{sec:steady_growth_griffith}.\mycomment{TODO}{Section~5: normalize with the three-dimensional solid reference $J_c^{\text{num,3D}}$ of the same mesh (076 results), not with $1.50J_c^0$ (2D value now $1.425J_c^0$).}",
    r"\replaced{Since only the parameter set \vMLabel is considered in this section, and the intention of the following discussion is to highlight the effect of the geometric representation of the microstructure and not of the material law, the records of $J_x$ are normalized by the numerical fracture resistance $J_c^{\text{num}}$ of the homogeneous material in the respective dimension. For the three-dimensional model, $J_c^{\text{num}}$ is determined from a specimen without pores, with the yield stress assigned to the whole domain, constructed and meshed by the same rules as the porous specimens, as the mean of $J_x$ over the central part of the crack path. It is $1.415J_c^0$ for the domain of $w_s = L$ and $1.444J_c^0$ for the larger domain of $w_s = 2L$, which agrees with the two-dimensional value $1.425J_c^0$ of Section~\ref{sec:steady_growth_griffith} within $2\,\%$. The simulations for $w_s \le L$ are normalized by the former value, the simulation for $w_s = 2L$ by the latter, and the simulation on the finer mesh by its own reference, $1.351J_c^0$.}{Since only the \vMLabel dataset is considered in this section and it is the intention of the following discussion to highlight the effect of the geometric representation of microstructure and not of the material law, the records of $J_x$ are normalized by the numerical fracture resistance of the von Mises-type material law $J_x^{\text{num,pl}}\approx1.50J_c^0$; see Section~\ref{sec:steady_growth_griffith}.}",
)
ms.rep(
    r"Figure~\ref{fig:PAPER_VM_Jx_all_3D} compares the evolution of $J_x$ with respect to $x_{ct}$ for the two- and three-dimensional models at $w_s=1.0L$. While the three-dimensional response is significantly smoother, a characteristic periodic pattern with pronounced peaks remains observable. The smoother behavior is consistent with the crack-front evolution shown in Fig.~\ref{fig:position_3D}: in contrast to the two-dimensional case in Fig.~\ref{fig:PAPER_00_xct_pf_vs_xct_KI_diamond}, the crack front can propagate between pores and is not trapped completely by a pore. These peaks correspond to phases of unsteady crack propagation induced by the pores and define the effective crack resistance. Nevertheless, Figs.~\ref{fig:position_3D} and~\ref{fig:PAPER_VM_Jx_all_3D} still show phases of nearly instantaneous crack advance and elevated values of the J-Integral when the crack front is forced to pass between pores. The comparison in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D} therefore indicates that this mechanism still produces a toughening effect of comparable magnitude to the two-dimensional case.",
    r"\replaced{Figure~\ref{fig:PAPER_VM_Jx_all_3D} compares $J_x$ as a function of $x_{ct}$ for the two- and the three-dimensional model at $w_s = L$. The two-dimensional record shows the sharp re-nucleation peaks of Section~\ref{sec:2D_fracture_behavior}: $J_x$ rises above $J_c^{\text{num}}$ while the crack tip is trapped at the pore wall and drops during the unstable failure of the ligament. The three-dimensional record is a smooth undulation with one maximum per cell. $J_x$ is lowest when the crack tip enters a pore and highest shortly after it has left the pore and entered the ligament, with peak values of $0.97$, $0.95$ and $0.90J_c^{\text{num}}$ for the first three ligaments, i.e., below the crack resistance of the homogeneous material. The last pore is affected by the end of the resolved region. Since the front is never trapped as a whole, the forced re-nucleation at a blunt notch, which produces the peaks of the two-dimensional model, does not occur, and since the ligament does not fail at once, there is no phase of unstable crack advance.}{Figure~\ref{fig:PAPER_VM_Jx_all_3D} compares the evolution of $J_x$ with respect to $x_{ct}$ for the two- and three-dimensional models at $w_s=1.0L$. While the three-dimensional response is significantly smoother, a characteristic periodic pattern with pronounced peaks remains observable. The smoother behavior is consistent with the crack-front evolution shown in Fig.~\ref{fig:position_3D}: in contrast to the two-dimensional case in Fig.~\ref{fig:PAPER_00_xct_pf_vs_xct_KI_diamond}, the crack front can propagate between pores and is not trapped completely by a pore. These peaks correspond to phases of unsteady crack propagation induced by the pores and define the effective crack resistance. Nevertheless, Figs.~\ref{fig:position_3D} and~\ref{fig:PAPER_VM_Jx_all_3D} still show phases of nearly instantaneous crack advance and elevated values of the J-Integral when the crack front is forced to pass between pores. The comparison in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D} therefore indicates that this mechanism still produces a toughening effect of comparable magnitude to the two-dimensional case.}",
)
ms.rep(
    r"The periodic pattern between pores one and three" "\n"
    r"% two and four" "\n"
    r" is considered \replaced{characteristic}{representative} of the microstructure and is shown for varying $w_s$ in Fig.~\ref{fig:test12q2_3D}. As in the two-dimensional case, $J_{x,c}^{\text{eff}}$ increases with increasing wall width until saturation is reached.",
    r"\replaced{The pattern between pores one and three, which contains the ligament in which $J_{x,c}^{\text{eff}}$ is evaluated, is shown for the five wall widths in Fig.~\ref{fig:test12q2_3D}. $J_x$ varies smoothly for all wall widths, and the undulation becomes shallower as the pores move apart. The effective crack resistance, taken as in two dimensions as the maximum of $J_x$ while the crack tip lies in the ligament ahead of a pore, here the third pore, increases with the wall width, Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}a): $J_{x,c}^{\text{eff}} = 1.11$, $1.12$, $1.30$, $1.34$ and $1.39J_c^0$, i.e., $0.79$, $0.79$, $0.92$, $0.95$ and $0.96J_c^{\text{num}}$ for $w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and $2L$. On the finer mesh, $h = 0.07L$, the absolute values for $w_s = L$ decrease by $3.6\,\%$ ($J_{x,c}^{\text{eff}}$) and $4.5\,\%$ ($J_c^{\text{num}}$), whereas the normalized value changes by only $+0.9\,\%$, open triangle in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}a). The normalized effective crack resistance is therefore the mesh-robust quantity, and it is the one compared with the two-dimensional results in the following.}{The periodic pattern between pores one and three is considered representative of the microstructure and is shown for varying $w_s$ in Fig.~\ref{fig:test12q2_3D}. As in the two-dimensional case, $J_{x,c}^{\text{eff}}$ increases with increasing wall width until saturation is reached.}" "\n"
    r"\mycomment{TODO}{Decision for the authors: for $w_s = 0.375L$ and $0.5L$ the window used for $J_{x,c}^{\text{eff}}$ (crack tip in the ligament ahead of the third pore) does not contain the maximum of $J_x$ of that pore-to-pore cycle, which occurs while the tracked tip is still within the $x$-range of the second pore, i.e., while the front passes beside the sphere: cycle maxima $0.873$, $0.891$, $0.928$, $0.950$ and $0.963J_c^{\text{num}}$ against the window values $0.786$, $0.791$, $0.919$, $0.950$ and $0.963$ (visible in Fig.~\ref{fig:test12q2_3D}). If $J_{x,c}^{\text{eff}}$ in 3D is defined as the cycle maximum, the 3D curve in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}a) is nearly flat ($0.87$ to $0.96$) and the figure, the numbers in this paragraph and in the letter must be regenerated (076/paper\_figures\_3D.py, evaluation\_rev2.py Jeff()). The qualitative conclusions (no toughening above $J_c^{\text{num}}$ in 3D, much smaller loss for thin walls than in 2D) are unchanged.}",
)
ms.rep(
    r"This trend is summarized in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}, where the effective crack resistance $J_{x,c}^{\text{eff}}$ is plotted as a function of $w_s$ for both the two- and three-dimensional models. The three-dimensional model exhibits a similar qualitative behavior, with a slightly reduced saturation value. This value is about $1.15J_c^{\text{num,pl}}$ in the three-dimensional model, whereas it is about $1.25J_c^{\text{num,pl}}$ in the two-dimensional model. The three-dimensional response also shows a weaker dependence on $w_s$, " "\n"
    r"reflecting the reduced crack trapping due to the additional degree of freedom for crack front propagation.",
    r"\replaced{The comparison with the two-dimensional model in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}a) shows two differences. In two dimensions, the pores raise the crack resistance of the von~Mises material above the homogeneous value for $w_s \gtrsim 0.9L$, up to $1.28J_c^{\text{num}}$ at $w_s = 2L$ and $1.34J_c^{\text{num}}$ at $w_s = 4L$. In three dimensions, the effective crack resistance stays below the homogeneous value for all wall widths and saturates at about $0.96J_c^{\text{num}}$ for $w_s = 2L$. For thin walls, on the other hand, the three-dimensional specimen loses much less of its crack resistance than the two-dimensional one, $0.79$ against $0.50J_c^{\text{num}}$ at $w_s = 0.375L$. Both differences follow from the crack-front evolution described above. The toughening of the two-dimensional specimen at large wall widths stems from the forced re-nucleation at the blunt pore wall, Section~\ref{sec:2D_mechanisms}, which requires the whole front to be trapped. In three dimensions, the front bypasses the spherical pores, so that this mechanism is absent. Equally, a thin ligament does not fail as a whole, which limits the loss of crack resistance in the nucleation regime of Section~\ref{sec:2D_mechanisms}.}{This trend is summarized in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}, where the effective crack resistance $J_{x,c}^{\text{eff}}$ is plotted as a function of $w_s$ for both the two- and three-dimensional models. The three-dimensional model exhibits a similar qualitative behavior, with a slightly reduced saturation value. This value is about $1.15J_c^{\text{num,pl}}$ in the three-dimensional model, whereas it is about $1.25J_c^{\text{num,pl}}$ in the two-dimensional model. The three-dimensional response also shows a weaker dependence on $w_s$, reflecting the reduced crack trapping due to the additional degree of freedom for crack front propagation.}" "\n\n"
    r"\added{The energy partition per pore-to-pore cycle, evaluated as in Section~\ref{sec:2D_energy_partition} for the cycle that contains the evaluation ligament, confirms the different mechanism, Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_2D_3D}b). In three dimensions, $92$ to $94\,\%$ of the dissipated energy create crack surface, $6$ to $8\,\%$ remain stored as plastic energy and less than $1\,\%$ is dissipated by the viscous evolution of the phase field, for all wall widths, compared with a viscous share of $12$ to $37\,\%$ in two dimensions, which collects the energy released in the unstable snap-through of the ligaments. The energy dissipated per unit crack advance, $G_{\text{diss}} = 1.11$ to $1.44J_c^0$, agrees with $J_{x,c}^{\text{eff}}$ within $5\,\%$ for all wall widths, whereas $G_{\text{diss}}/J_{x,c}^{\text{eff}} = 0.73$ to $0.93$ in two dimensions, where $J_{x,c}^{\text{eff}}$ is a sharp peak and $G_{\text{diss}}$ the average over the cycle. In three dimensions, the driving force required for crack advance is thus nearly constant along the crack path, and the effective crack resistance coincides with the energy dissipated per unit crack advance.}",
)

# ---------------------------------------------------------------- conclusion TODO: concrete 3D input for the next pass
ms.rep(
    r"energy partition, ``predicted'' wording; and with the Section~5 results.}",
    r"energy partition, ``predicted'' wording; and with the Section~5 results: in 3D the pores never raise $J_{x,c}^{\text{eff}}$ above $J_c^{\text{num}}$ ($0.79$ to $0.96$, saturation at $0.96$ against $1.34$ in 2D), no re-nucleation peaks, the front bypasses the pores, $92$ to $94\,\%$ of the dissipated energy creates crack surface, $G_{\text{diss}} \approx J_{x,c}^{\text{eff}}$. The statements ``good quantitative agreement between the two- and three-dimensional predictions'' (abstract, conclusion) and ``the pores still induce phases of stress-driven instantaneous crack growth'' (conclusion) no longer hold.}",
)

ms.save()

# ============================================================================ letter
# ---------------------------------------------------------------- R1 major 4
lt.rep(
    r"their Zener ratios are $A = 0.969$, $0.978$, $0.984$, $0.992$, $0.996$ and $1.000$ for $w_s = 0.25L$, $0.375L$, $0.5L$, $0.75L$, $L$ and $2L$ (Table~4). Two earlier three-dimensional runs on a coarser mesh with the full cubic stiffness in the embedding changed $J_{x,c}^{\text{eff}}$ by $-0.37\,\%$ ($w_s = 0.375L$) and $-0.22\,\%$ ($w_s = L$) relative to their isotropic twins, which is within the numerical scatter of those runs.\mycomment{TODO}{Section~5: state the Zener ratios in Table~4 and refer to Section~4.5 for the cubic-embedding test (3D numbers from the old-mesh pairs, or add a fixed-mesh S3F run).}",
    r"their Zener ratios are $A = 0.978$, $0.984$, $0.992$, $0.996$ and $1.000$ for $w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and $2L$ (Table~4 in Section~5.1, which also gives the effective Poisson's ratios, $0.23$ to $0.25$). Two earlier three-dimensional runs on a coarser mesh with the full cubic stiffness in the embedding changed $J_{x,c}^{\text{eff}}$ by $-0.37\,\%$ ($w_s = 0.375L$) and $-0.22\,\%$ ($w_s = L$) relative to their isotropic twins, which is within the numerical scatter of those runs. Section~5.1 refers to Section~4.5 for this test.",
)
lt.rep(
    r"%@quote cmd:The unit cell has cubic symmetry" "\n" r"%@generated-begin",
    r"%@quote cmd:The unit cell has cubic symmetry" "\n"
    r"%@quote cmd:Effective elastic constants of the three-dimensional unit cells" "\n"
    r"%@quote cmd:lists the effective constants for the five wall widths" "\n"
    r"%@generated-begin",
)

# ---------------------------------------------------------------- minor 25
lt.rep(
    r"were merged into the three panels of Fig.~11 with a common axis, so that the parameter sets can be compared directly.\mycomment{TODO}{Fig.~3 (Section~3.5) and the figures of Section~5 (Figs.~15 to 20) follow with the 076 results.}",
    r"were merged into the three panels of Fig.~11 with a common axis, so that the parameter sets can be compared directly. The result figures of Section~5 were regenerated from the new three-dimensional simulations in the same way (Figs.~16 to 19: crack tip position vs.\ pseudo-time, the J-integral records of the two- and the three-dimensional model for $w_s = L$ in one plot, the sections between pores one and three for all wall widths, and the effective crack resistance together with the energy partition of both models). The former figure of the fields $s$ and $\alpha$ for $w_s = 0.25L$ was removed, since this wall width is no longer part of the three-dimensional series.\mycomment{TODO}{Fig.~3 (Section~3.5) follows with the regenerated strip/solid figure.}",
)
lt.rep(
    "\\begin{reviewercomment}\n26. ",
    "\\begin{relatedchanges}\n"
    "\\relhead{Captions of Figs.~15 to 19 (Section~5)}\n"
    "%@quote cmd:illustration from a simulation on a coarser mesh\n"
    "%@quote cmd:, three-dimensional model}\n"
    "%@quote cmd:in the two- and the three-dimensional model, each normalized\n"
    "%@quote cmd:between pores one and three of the three-dimensional specimen\n"
    "%@quote cmd:The open triangle marks\n"
    "\\end{relatedchanges}\n\n"
    "\\begin{reviewercomment}\n26. ",
)

# ---------------------------------------------------------------- Reviewer 3, comment 3
lt.rep(
    r"it would be \emph{tougher} than the von~Mises set ($1.90\,J_c^0 = 1.34\,J_c^{\mathrm{num,pl}}$). The same holds in" "\n"
    r"three dimensions\mycomment{TODO}{3D: insert $J_{x,c}^{\text{eff}}/J_c^{\mathrm{num,3D}}$ of the fixed-mesh runs (S1F/S2F, $\approx 0.95$ at $w_s = L$) once Section~5 is updated}.",
    r"it would be \emph{tougher} than the von~Mises set ($1.90\,J_c^0 = 1.34\,J_c^{\mathrm{num,pl}}$). In three dimensions" "\n"
    r"(Section~5.2), the pores do not raise the crack resistance of the von~Mises set above its homogeneous value at all:" "\n"
    r"$J_{x,c}^{\text{eff}}$ saturates at $0.96\,J_c^{\mathrm{num}}$ at $w_s = 2L$ (with $J_c^{\mathrm{num}}$ the value of the" "\n"
    r"three-dimensional solid specimen), against $1.34\,J_c^{\mathrm{num}}$ in two dimensions.",
)
lt.rep(
    r"%@quote cmd:The ratio of the values of \EqJcLabel and \EqSigmaStarLabel, $1.088/0.604" "\n"
    r"%@quote cmd:The two linear elastic parameter sets differ only in the value of $J_c$" "\n"
    r"%@quote cmd:The three comparisons in Fig." "\n" r"%@generated-begin",
    r"%@quote cmd:The ratio of the values of \EqJcLabel and \EqSigmaStarLabel, $1.088/0.604" "\n"
    r"%@quote cmd:The two linear elastic parameter sets differ only in the value of $J_c$" "\n"
    r"%@quote cmd:The three comparisons in Fig." "\n" r"%@quote cmd:The comparison with the two-dimensional model in Fig." "\n" r"%@generated-begin",
)

# ---------------------------------------------------------------- Additional corrections, item 4 and new items 9-11
lt.rep(
    r"\EqSigmaStarLabel, previously $1.80$, $1.9$ and $1.12$ in units of the $J_c^{\mathrm{num}}$ of \EqJcLabel).\mycomment{TODO}{Section~5 numbers (076) to be added here once available.}",
    r"\EqSigmaStarLabel, previously $1.80$, $1.9$ and $1.12$ in units of the $J_c^{\mathrm{num}}$ of \EqJcLabel). The" "\n"
    r"three-dimensional results of Section~5 were recomputed with these parameters and, in addition, with the geometry, mesh and" "\n"
    r"normalisation of items~9 to~11. The effective crack resistances are now $J_{x,c}^{\text{eff}} = 1.11$, $1.12$, $1.30$, $1.34$" "\n"
    r"and $1.39\,J_c^0$, i.e., $0.79$, $0.79$, $0.92$, $0.95$ and $0.96$ in units of the numerical fracture resistance of the" "\n"
    r"three-dimensional solid, for $w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and $2L$. The statement of the previous version that the" "\n"
    r"three-dimensional saturation value ($\approx 1.15\,J_c^{\mathrm{num,pl}}$) is slightly below the two-dimensional one" "\n"
    r"($\approx 1.25$) is replaced by the finding that in three dimensions the pores do not raise the crack resistance of the" "\n"
    r"von~Mises material above the homogeneous value ($0.96$ at $w_s = 2L$, against $1.34$ in two dimensions), whereas for thin" "\n"
    r"walls the three-dimensional specimen loses much less than the two-dimensional one ($0.79$ against $0.50$ at $w_s = 0.375L$)," "\n"
    r"see Section~5.2.",
)
lt.rep(
    r"strip of Section~3.5 keeps $K_I^{\mathrm{bc}} = \sqrt{2.5J_c^0\mu_0}$." "\n" r"\end{enumerate}",
    r"strip of Section~3.5 keeps $K_I^{\mathrm{bc}} = \sqrt{2.5J_c^0\mu_0}$." "\n"
    r"\item The element size along the crack path of the three-dimensional simulations was stated as $\Delta h = 0.02L$, which is" "\n"
    r"the two-dimensional value. The three-dimensional meshes of the previous version were scaled with the domain, so that the" "\n"
    r"element size in the crack band grew with the wall width (median edge length $1.7\epsilon$ at $w_s = 0.5L$ to $6.4\epsilon$" "\n"
    r"at $w_s = 4L$) and the recorded $J_x$ increased with the wall width partly through the mesh. The new series uses a fixed" "\n"
    r"element size $h = 0.1L$ in the crack band for all wall widths (median edge length $1.49\epsilon$, between $85\,000$ and" "\n"
    r"$416\,000$ tetrahedra), stated in Section~5.1, and the statement in Section~3.3 that the element size is at most half the" "\n"
    r"regularization length is now restricted to two dimensions. A mesh-convergence check at $w_s = L$ with $h = 0.07L$" "\n"
    r"($1.03\epsilon$, $505\,000$ tetrahedra) changes the absolute values by $-3.6\,\%$ ($J_{x,c}^{\text{eff}}$) and $-4.5\,\%$" "\n"
    r"(solid reference) and the normalised effective crack resistance by $+0.9\,\%$ (Section~5.2). Since fixed-size meshes for" "\n"
    r"$w_s = 3L$ and $4L$ were out of reach, the three-dimensional series now covers $w_s = 0.375L$ to $2L$ (previously $0.25L$" "\n"
    r"to $4L$)." "\n"
    r"\item The three-dimensional specimen now contains $4 \times 3 \times 2$ cells in an embedding of $(6w_c) \times 20L \times" "\n"
    r"(2.5w_c)$ (previously $6 \times 3 \times 2$ cells in $(8w_c) \times 20L \times (4w_c)$), i.e., one cell width of embedding" "\n"
    r"before and after the porous block and $w_c/4$ on each side in the $z$-direction, and the J-integral is normalised by the" "\n"
    r"thickness $2.5w_c$ (Section~5.2). The time step control of the three-dimensional runs increases the step after fewer than" "\n"
    r"five (for the longest runs seven) instead of four Newton iterations, which is stated in Section~5.1 together with a" "\n"
    r"validation ($+0.5\,\%$ in $J_{x,c}^{\text{eff}}$ for $w_s = L$)." "\n"
    r"\item The three-dimensional records of $J_x$ were normalised by the two-dimensional value $J_c^{\mathrm{num,pl}} \approx" "\n"
    r"1.50\,J_c^0$ in the previous version. They are now normalised by the numerical fracture resistance of a three-dimensional" "\n"
    r"solid specimen without pores, computed with the same domain construction and the same mesh rule ($1.415\,J_c^0$ for the" "\n"
    r"domain of $w_s = L$, $1.444\,J_c^0$ for $w_s = 2L$, $1.351\,J_c^0$ on the finer mesh), which agrees with the two-dimensional" "\n"
    r"value $1.425\,J_c^0$ within $2\,\%$. Normalised in this way, the effective crack resistance is robust against the mesh:" "\n"
    r"the run for $w_s = L$ on the mesh of the previous version gave $0.93$ against $0.95$ on the fixed mesh, although the" "\n"
    r"absolute values differed by $14\,\%$." "\n"
    r"\end{enumerate}",
)
lt.rep(
    r"%@quote cmd:equal}{almost identical}" "\n" r"%@generated-begin",
    r"%@quote cmd:equal}{almost identical}" "\n"
    r"%@quote cmd:is at most half the regularization length $\epsilon$ in two dimensions" "\n"
    r"%@quote cmd:$4\times3\times2$" "\n"
    r"%@quote cmd:one cell width of embedding before and after" "\n"
    r"%@quote cmd:The mesh of tetrahedral finite elements is refined in a band" "\n"
    r"%@quote cmd:Along the expected crack path, the finite element edge length is set to" "\n"
    r"%@quote cmd:The amplitude of the K-field is not adapted to the wall width" "\n"
    r"%@quote cmd:$w_s = 0.375L$, $0.5L$, $0.75L$, $L$ and $2L$}{ranging from" "\n"
    r"%@quote cmd:The time step is controlled as in Section" "\n"
    r"%@quote cmd:In all simulations, the crack propagates through the central row of pores" "\n"
    r"%@quote env:eq:J_3D" "\n"
    r"%@quote cmd:the records of $J_x$ are normalized by the numerical fracture resistance $J_c^{\text{num}}$ of the homogeneous material in the respective dimension" "\n"
    r"%@quote cmd:compares $J_x$ as a function of $x_{ct}$ for the two- and the three-dimensional model" "\n"
    r"%@quote cmd:The pattern between pores one and three" "\n"
    r"%@quote cmd:The energy partition per pore-to-pore cycle, evaluated as in Section" "\n"
    r"%@generated-begin",
)

lt.save()

# ---------------------------------------------------------------- figure widths (applied after the first build, Section 5 ended on p. 32)
ms2 = Doc(MS)
ms2.rep(r"\resizebox{0.6\textwidth}{!}{\includegraphics[]{pics/mesh_with_cs.png}}", r"\resizebox{0.42\textwidth}{!}{\includegraphics[]{pics/mesh_with_cs.png}}")
ms2.rep(r"\includegraphics[width=0.9\textwidth]{pics/PAPER_rev2_3D_Jx_vs_xct_sections.pdf}", r"\includegraphics[width=0.8\textwidth]{pics/PAPER_rev2_3D_Jx_vs_xct_sections.pdf}")
ms2.rep(r"\resizebox{0.9\textwidth}{!}{\input{pics/PAPER_rev2_3D_Jeff_and_partition_2D_vs_3D.pgf}}", r"\resizebox{0.85\textwidth}{!}{\input{pics/PAPER_rev2_3D_Jeff_and_partition_2D_vs_3D.pgf}}")
ms2.save()
print("part 7 applied")
