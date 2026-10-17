#!/usr/bin/env python3
"""Round-2 edits, part 3: response letter -- answers and quote requests for the text changes made in parts 1 and 2.
Items that still depend on simulation results keep their TODO notes."""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "response_to_reviewers_template.tex"
s = open(path).read()
E = []

# ---------------------------------------------------------------- editor summary
E.append((r"""\item \mycomment{TODO}{introduction rewritten (problem $\to$ approaches and limitations $\to$ prior work $\to$ open questions), homogenization framing with Hill's RVE definition and the results of Braides et al.\ and Cagnetti et al.; term ``representative region'' dropped}""",
r"""\item the introduction was rewritten (engineering problem $\to$ approaches and their limitations $\to$ previous contributions $\to$ open questions), the relation of the numerical experiment to homogenization is stated with Hill's definition of a representative volume element and with the rigorous results on the homogenization of surface energies, the term ``representative region'' was dropped, and the number of references was increased from 34 to 44;"""))

E.append((r"""\item \mycomment{TODO}{cubic symmetry of the effective stiffness quantified (Zener ratio) and the influence of the isotropic approximation of the embedding tested by simulations with the full cubic stiffness}""",
r"""\item the square resp.\ cubic symmetry of the effective stiffness of the unit cells is quantified (Zener ratio) and the influence of the isotropic approximation of the embedding is tested by simulations with the full anisotropic stiffness\mycomment{TODO}{numbers from study S3};"""))

# ---------------------------------------------------------------- R1 major 1
E.append((r"""\mycomment{TODO}{answer: literature broadened in the rewritten introduction (homogenization of surface energies: Braides et al.\ 1996, Cagnetti et al.\ 2019; RVE: Hill 1963; numerical experiments for effective toughness: Hossain 2014, Hsueh 2018, Brach 2019, Lebihain 2021, Schneider 2020, Ernesti 2021/22, Michel \& Suquet 2022, Liu \& Shen 2025, Aranda \& Segurado 2025, Hao et al.\ 2026, \dots; ductile phase-field fracture: Miehe/Aldakheel 2016, Ambati 2015, Borden 2016, Kuhn 2016, Noll 2020, \dots; plasticity: Simo \& Hughes). Number of references $28 \to$ \dots. We also went through the whole manuscript once more for notation and consistency (see the additional corrections).}""",
r"""We accept this criticism. The introduction was rewritten from a broader view of the field. It now distinguishes three classes of approaches to the crack resistance of heterogeneous materials and states the limitations of each: the rigorous homogenization of surface energies of Griffith type (Braides, Defranceschi and Vitali 1996; Cagnetti, Dal Maso, Scardia and Zeppieri 2019; the variational definition of Schneider and of Ernesti and Schneider; the assessment by Michel and Suquet), the analyses of crack-front trapping by perturbation and numerical methods (Gao and Rice 1989; Bower and Ortiz 1991; Lebihain, Leblond and Ponson 2020), and the numerical experiment of Hossain et al.\ with its recent developments. The literature on phase-field models coupled with plasticity was extended (Alessi, Marigo and Vidoli 2014; Ambati, Gerasimov and De Lorenzis 2015; Borden et al.\ 2016, in addition to Miehe et al., Kuhn et al.\ and Noll et al.), and Hill's definition of a representative volume element is cited where the term is discussed. In total, ten references were added (Hill 1963; Braides et al.\ 1996; Francfort and Marigo 1998; Cagnetti et al.\ 2019; Gao and Rice 1989; Bower and Ortiz 1991; Lebihain et al.\ 2020; Alessi et al.\ 2014; Borden et al.\ 2016; Ambati et al.\ 2015), so that the manuscript now cites 44 instead of 34 references, and the reference list is ordered by first citation. Independently of the reviewer's comments, we also went through the whole manuscript once more for notation, consistency and the correspondence between the text and the simulation code. The corrections resulting from this audit are listed in the section ``Additional corrections by the authors'' at the end of this letter."""))

E.append((r"""\relhead{\mycomment{TODO}{quotes of the rewritten introduction and the new references}}""",
r"""\relhead{Introduction, rewritten (the new reference list entries are not quoted here, see the manuscript)}
%@quote cmd:Porous and other heterogeneous materials are increasingly used
%@quote cmd:For a heterogeneous material, the fracture resistance entering
%@quote cmd:For elastic and plastic properties, the step from the microstructure
%@quote cmd:Rigorous results exist for the homogenization of surface energies
%@quote cmd:A second class of approaches describes the interaction of a crack front
%@quote cmd:The third class is the numerical experiment of Hossain
%@quote cmd:A recent related development is the variable stiffness boundary condition of Hao et al.~\cite{hao_variable_2026}, which builds on the surfing boundary condition but translates a simple uniform remote displacement into a non-uniform load through an engineered stiffness gradient, thereby facilitating implementation in experiments and commercial finite-element software. The limitations
%@quote cmd:Phase-field models of fracture, see the review by Ambati
%@quote cmd:Our previous work considered only reversible elastic material behavior
%@quote cmd:The open questions addressed in this work are the following"""))

# ---------------------------------------------------------------- R1 major 2
E.append((r"""\mycomment{TODO}{answer along the agreed line: (i) the first sentence and the introduction are rewritten; homogenization of the surface energy exists under assumptions (Braides et al.\ 1996, Cagnetti et al.\ 2019, discussed by Michel \& Suquet 2022) and is not what we do; (ii) ``representative region'' replaced by ``resolved porous region''/``specimen'' everywhere, Hill's definition quoted, and the quantity is consistently called the effective crack resistance \emph{of the numerical experiment}; (iii) representativity is not assumed: new probes (shifted initial crack, five instead of three pore rows, study S5) show by how much $J_{x,c}^{\text{eff}}$ changes, and the text states what would be required for a representativity study; (iv) we do not claim that the microstructured material is described by the coupled model at the macroscale; the macroscopic use of $J_{x,c}^{\text{eff}}$ is restricted to a crack-growth criterion of the type of eq.~(1) under the loading of the experiment.}""",
r"""We agree with the reviewer on all four points and have changed the conceptual framing of the manuscript accordingly.
\begin{itemize}
\item The first sentence and the whole introduction were rewritten. The manuscript no longer speaks of a ``homogenization of fracture toughness''. It now states what homogenization provides for elastic and plastic properties (Hill's representative volume element, whose effective properties are independent of the boundary conditions), why fracture does not fit into this framework directly (localized, evolving, possibly unstable process; loss of the separation of length scales), and which rigorous results exist nevertheless for surface energies of Griffith type (Braides et al.\ 1996, Cagnetti et al.\ 2019, the cell formula used by Schneider and assessed by Michel and Suquet). The quantity determined in this work is consistently called the effective crack resistance of the numerical experiment and is explicitly distinguished from a fracture toughness in the sense of a material property.
\item ``Representative'' is no longer used for the sample. The region is called the resolved porous region (or resolved region), Hill's definition of a representative volume element is quoted in Section~2, and it is stated that the representativity of the resolved region in this sense is not established. The term ``representative'' was also removed from the figure caption of Fig.~1, from the conclusion and, in its colloquial sense (``representative of the main trend''), from Section~4.
\item The manuscript no longer suggests that representativity would be automatic. Section~2 now states that the independence of $J_{x,c}^{\text{eff}}$ of the initial crack and of the size of the resolved region cannot be taken for granted and has to be tested for every microstructure, that only one crack-growth direction is considered, and that a representativity study in the sense of Hill (statistics over microstructure realizations, sample sizes and loading conditions) is outside the scope of this work. What is done instead is a direct probe of the two dependencies that can be tested within the present setup: the length of the initial crack and the number of resolved pore rows are varied in the new Section~\mycomment{TODO}{4.6 (study S5)}, and the resulting change of $J_{x,c}^{\text{eff}}$ is reported there.\mycomment{TODO}{numbers from study S5}
\item We do not claim that the microstructured material is described by the coupled phase-field--plasticity model at the macroscopic scale. The macroscopic use of $J_{x,c}^{\text{eff}}$ is restricted to a crack-growth criterion of the type of eq.~(1) under the loading of the experiment, and the dependence of the recorded value on the loading of the experiment is stated as a limitation of the approach in the introduction and in Section~2.
\end{itemize}"""))

E.append((r"""\relhead{\mycomment{TODO}{quotes: first sentence, Sec.~2 framing, S5 results, conclusion}}""",
r"""\relhead{Section~2, framing of the numerical experiment (the rewritten introduction is quoted under major point~1)}
%@quote cmd:A Numerical Experiment for the Crack Resistance of a Resolved Porous Region
%@quote cmd:that resolves the}{containing representative}
%@quote cmd:finite region of the material
%@quote cmd:This region is referred to as the resolved porous region
%@quote cmd:This definition does not imply that
%@quote cmd:cannot be taken for granted and has to be tested
%@quote cmd:resolved porous region}{representative
\end{relatedchanges}
\begin{relatedchanges}
\relhead{\mycomment{TODO}{quotes: new Section 4.6 with the S5 results}}"""))

# ---------------------------------------------------------------- R1 major 4
E.append((r"""\mycomment{TODO}{answer: the reviewer is right; the unit cells have square/cubic symmetry. The full effective stiffness is now reported (Table~X: $C_{11}, C_{12}, C_{33}$ resp.\ $C_{44}$ and the Zener ratio $A = 2C_{44}/(C_{11}-C_{12})$ for all $w_s$); the isotropic approximation used in the paper takes $\lambda^{\mathrm{eff}} = C_{12}$, $\mu^{\mathrm{eff}} = C_{44}$ and thus misrepresents $C_{11}$ by $x$\,\%. Simulations with the full cubic stiffness in the embedding (study S3, 2D: $w_s = 0.375L, 1.0L, 4.0L$; 3D: $w_s = 0.375L, 1.0L$) change $J_{x,c}^{\text{eff}}$ by $y$\,\%, which justifies the approximation for the conclusions drawn. Corner/notch effects: the corners of the resolved porous block are at least $w_c$ from the crack plane and $2w_c$ from the first and last pore; the J-integral is path independent in the homogeneous elastic embedding, and the fields in the embedding stay elastic (yield stress $\to\infty$ there, now stated); the stress concentration at the corners is visible but $\dots$ (quantify from the field output).}""",
r"""The reviewer is right. The unit cells of both microstructures have square (two-dimensional) resp.\ cubic (three-dimensional) symmetry, and the previous versions assigned an isotropic stiffness to the embedding without saying so or quantifying the approximation. The revised manuscript states the symmetry in Section~2, reports the full effective stiffness of every unit cell in Sections~4.1 and 5.1 ($C_{11}$, $C_{12}$, $C_{33}$ resp.\ $C_{44}$, and the Zener ratio $A = 2C_{33}/(C_{11}-C_{12})$ resp.\ $2C_{44}/(C_{11}-C_{12})$, Tables~\mycomment{TODO}{X and Y}), and explains which constants the isotropic approximation uses. For the two-dimensional unit cells, the Zener ratio is $0.89$ for the smallest wall width $w_s = 0.375L$, $0.96$ for $w_s = L$ and $1.00$ for $w_s = 4L$, i.e., the anisotropy is moderate and vanishes for widely spaced pores. The influence of the approximation on the result is tested directly: the simulations of Section~\mycomment{TODO}{4.5 (study S3)} assign the full square-symmetric (cubic) stiffness to the embedding for $w_s = 0.375L$, $1.0L$ and $4.0L$ in two dimensions and for $w_s = 0.375L$ and $1.0L$ in three dimensions.\mycomment{TODO}{result: $J_{x,c}^{\text{eff}}$ changes by $y$\,\% (study S3), which justifies the approximation for the conclusions drawn.}

Regarding the corners: the transition from the resolved porous region to the homogeneous embedding is a material discontinuity and the corners of the resolved region are stress concentrators. This is now stated in Section~2. The corners are located at least one cell width $w_c$ away from the crack plane and at least one cell width ahead of the first and behind the last pore, the embedding is linear elastic (the yield stress is not assigned to it, which is now stated explicitly in Section~4.1), and the J-integral is evaluated on the outer boundary of the embedding. The local stress concentration at the corners therefore does not enter the recorded driving force, and it did not lead to crack nucleation at a corner in any of the simulations.\mycomment{TODO}{quantify the corner stress from the field output of study S1 and state it in Section~4.5}"""))

E.append((r"""\relhead{\mycomment{TODO}{quotes: Sec.~2 and 4.1 (embedding, Zener ratio, Table), new paragraph with the S3 results}}""",
r"""\relhead{Sections~2, 4.1 and 5.1 (embedding, symmetry of the effective stiffness)}
%@quote cmd:homogeneous matrix with the effective elastic stiffness of a unit cell
%@quote cmd:The transition from the resolved region to the homogeneous embedding
%@quote cmd:The unit cell has square symmetry
%@quote cmd:The embedding is therefore modelled as linear elastic
%@quote cmd:The unit cell has cubic symmetry
\end{relatedchanges}
\begin{relatedchanges}
\relhead{\mycomment{TODO}{quotes: Tables of the effective constants, new Section 4.5 with the S3 results}}"""))

# ---------------------------------------------------------------- R1 minor points
M = {
0: (r"""\begin{response}\mycomment{TODO}{all semicolons in running text replaced by separate sentences (grep for ';' outside math).}\end{response}""",
    r"""\begin{response}
All semicolons in running text were replaced by separate sentences (ten occurrences). Three examples are quoted below.
\end{response}
\begin{relatedchanges}
\relhead{Sections~2, 4.1 and 5.2 (examples)}
%@quote cmd:. Hence,}{; hence,}
%@quote cmd:. Several}{; several}
%@quote cmd:. It}{; it}
\end{relatedchanges}"""),
1: (r"""\begin{response}\mycomment{TODO}{introduction restructured exactly in this order; see major point 1 and 2.}\end{response}""",
    r"""\begin{response}
The introduction was restructured in exactly this order: the engineering problem (fracture resistance parameters are needed for heterogeneous, defect-rich materials), the approaches to the problem together with their limitations (rigorous homogenization of surface energies, crack-front perturbation analyses, the numerical experiment of Hossain et al.), previous contributions, and the open questions addressed in this work. The claim of the first sentence was removed. The rewritten introduction is quoted under major point~1.
\end{response}"""),
2: (r"""\begin{response}\mycomment{TODO}{sentence split (introduction, paragraph on the previous Ramberg--Osgood study).}\end{response}""",
    r"""\begin{response}
The sentence was split into three sentences.
\end{response}
\begin{relatedchanges}
\relhead{Introduction}
%@quote cmd:Our previous work considered only reversible elastic material behavior
\end{relatedchanges}"""),
3: (r"""\begin{response}\mycomment{TODO}{section title changed, e.g.\ ``A numerical experiment for the crack resistance of a resolved porous region''.}\end{response}""",
    r"""\begin{response}
The title of Section~2 was changed so that it refers to the resolved porous region and not to a material.
\end{response}
\begin{relatedchanges}
\relhead{Title of Section~2}
%@quote cmd:A Numerical Experiment for the Crack Resistance of a Resolved Porous Region
\end{relatedchanges}"""),
4: (r"""\begin{response}\mycomment{TODO}{all bare equation references prefixed with ``eq.''/``Eq.''.}\end{response}""",
    r"""\begin{response}
All references to equations are now written as ``eq.~(n)'' (eleven places were changed, one of which is quoted below).
\end{response}
\begin{relatedchanges}
\relhead{Section~2 (example)}
%@quote cmd:eq.~@1
\end{relatedchanges}"""),
5: (r"""\begin{response}\mycomment{TODO}{term removed everywhere; see major point 2.}\end{response}""",
    r"""\begin{response}
The term was removed everywhere, see major point~2. The region is called the resolved porous region.
\end{response}
\begin{relatedchanges}
\relhead{Section~2 and conclusion}
%@quote cmdall:resolved}{representative}
\end{relatedchanges}"""),
6: (r"""\begin{response}\mycomment{TODO}{sentence removed.}\end{response}""",
    r"""\begin{response}
The sentence was removed.
\end{response}
\begin{relatedchanges}
\relhead{Section~2}
%@quote cmd:Furthermore, the simulation must be based on physically consistent principles.
\end{relatedchanges}"""),
7: (r"""\begin{response}\mycomment{TODO}{done.}\end{response}

\begin{reviewercomment}
8.""",
    r"""\begin{response}
Done.
\end{response}
\begin{relatedchanges}
\relhead{Section~2}
%@quote cmd: assumption
\end{relatedchanges}

\begin{reviewercomment}
8."""),
8: (r"""\begin{response}\mycomment{TODO}{sentence rewritten (Sec.~2, paragraph on the J-integral at the boundary).}\end{response}""",
    r"""\begin{response}
The passage was rewritten as three sentences.
\end{response}
\begin{relatedchanges}
\relhead{Section~2}
%@quote cmd:The recorded value depends on the crack pattern
\end{relatedchanges}"""),
9: (r"""\begin{response}\mycomment{TODO}{adopted.}\end{response}""",
    r"""\begin{response}
The suggested wording was adopted.
\end{response}
\begin{relatedchanges}
\relhead{Section~2}
%@quote cmd:indicates that the crack-resistance value is obtained as a maximum
\end{relatedchanges}"""),
10: (r"""\begin{response}\mycomment{TODO}{reworded: ``whether $J_{x,c}^{\text{eff}}$ becomes independent of the size of the resolved region and of the position of the initial crack'' instead of ``convergence''.}\end{response}""",
    r"""\begin{response}
The word ``convergence'' was inappropriate. The sentence now states what is meant, namely whether $J_{x,c}^{\text{eff}}$ becomes independent of the size of the resolved region and of the position of the initial crack when the region is enlarged, and that this is a question of representativity which the definition does not settle. See also major point~2.
\end{response}
\begin{relatedchanges}
\relhead{Section~2}
%@quote cmd:This definition does not imply that
\end{relatedchanges}"""),
11: (r"""\begin{response}\mycomment{TODO}{sentence removed; replaced by the statement of what was tested (S5) and what was not.}\end{response}""",
    r"""\begin{response}
No, it is not. The sentence was removed and replaced by the statement that this independence cannot be taken for granted, by a description of what is tested in this work and of what is not, see major point~2.
\end{response}
\begin{relatedchanges}
\relhead{Section~2}
%@quote cmd:cannot be taken for granted and has to be tested
\end{relatedchanges}"""),
12: (r"""\begin{response}\mycomment{TODO}{the introductory sentence of Section~3 moved into Section~3.1 (and checked for Sections 4 and 5).}\end{response}""",
    r"""\begin{response}
We apologize. The sentence between the heading of Section~3 and that of Section~3.1 was moved into Section~3.1. Sections~4 and 5 start directly with their first subsection.
\end{response}
\begin{relatedchanges}
\relhead{Sections~3 and 3.1}
%@quote cmd:This section introduces the phase-field models and numerical ingredients
%@quote cmd:This section introduces the phase-field models and the numerical ingredients
\end{relatedchanges}"""),
13: (r"""\begin{response}\mycomment{TODO}{done.}\end{response}

\begin{reviewercomment}
14.""",
    r"""\begin{response}
Done.
\end{response}
\begin{relatedchanges}
\relhead{Section~3.1, eq.~(13)}
%@quote env:f\,\dot{\gamma} = 0\changed
\end{relatedchanges}

\begin{reviewercomment}
14."""),
14: (r"""\begin{response}\mycomment{TODO}{reworded: $\eta > 0$ keeps the elastic problem well-posed in fully fractured regions (positive definite stiffness); the numerical remark removed.}\end{response}""",
    r"""\begin{response}
We agree. The remark on the numerics was removed, and the text now states that a positive $\eta$ keeps the stiffness positive definite in fully fractured regions, so that the equilibrium problem remains well-posed, and that $\eta$ is a part of the model.
\end{response}
\begin{relatedchanges}
\relhead{Section~3.1, after eq.~(16)}
%@quote cmd:keeps the stiffness positive definite
\end{relatedchanges}"""),
15: (r"""\begin{response}\mycomment{TODO}{sentence rewritten.}\end{response}""",
    r"""\begin{response}
The sentence was rewritten.
\end{response}
\begin{relatedchanges}
\relhead{Section~3.1, after eq.~(18)}
%@quote cmd:No pointwise convergence of $s$ is implied
\end{relatedchanges}"""),
16: (r"""\begin{response}\mycomment{TODO}{reworded: $M$ is the kinetic coefficient of the evolution equation; large $M$ approximates rate-independent evolution; the viscous dissipation $D_{\mathrm{visc}}$ is now also quantified (energy partition).}\end{response}""",
    r"""\begin{response}
We agree. The text now introduces $M$ as the kinetic coefficient of the evolution equation~(19), states that large values of $M$ approximate a rate-independent evolution, and refers to the energy partition of Section~\mycomment{TODO}{4.3}, where the energy dissipated by the viscous evolution of $s$ is quantified for the numerical experiments.
\end{response}
\begin{relatedchanges}
\relhead{Section~3.1, after eq.~(19)}
%@quote cmd:kinetic coefficient of the evolution equation
\end{relatedchanges}"""),
17: (r"""\begin{response}\mycomment{TODO}{isotropy, $K$ and $\mu$ introduced with the elastic energy, eq.~(16).}\end{response}""",
    r"""\begin{response}
The isotropy of the bulk material is now stated where the elastic energy is introduced, eq.~(15), together with the bulk modulus $K$, the shear modulus $\mu$ and their relation to the Lam\'e parameters. Eq.~(21) is now introduced as the consequence of eq.~(15) for the isotropic stiffness.
\end{response}
\begin{relatedchanges}
\relhead{Section~3.1, eqs.~(15) and (21)}
%@quote cmd:characterized by the bulk modulus
%@quote cmd:follows from eq.~(\ref{eq:psi_el}) for the isotropic stiffness
\end{relatedchanges}"""),
18: (r"""\begin{response}\mycomment{TODO}{done.}\end{response}

\begin{reviewercomment}
19.""",
    r"""\begin{response}
Done, see the quote under the next point.
\end{response}

\begin{reviewercomment}
19."""),
19: (r"""\begin{response}\mycomment{TODO}{paragraph reordered.}\end{response}""",
    r"""\begin{response}
Section~3.3 was reordered accordingly: spatial and temporal discretization first, then the assembly of the monolithic system, then the solvers and the time step control.
\end{response}
\begin{relatedchanges}
\relhead{Section~3.3}
%@quote cmd:The spatial discretization uses first-order Lagrange finite elements on triangular meshes in two dimensions and tetrahedral meshes in three dimensions. Along the expected crack path, the mesh is locally refined such that the element edge length $\Delta h$ is at most half
%@quote cmd:The weak forms of the coupled boundary-value problems are implemented in FEniCSx using \texttt{dolfinx} version 0.7.3
\end{relatedchanges}"""),
20: (r"""\begin{response}\mycomment{TODO}{reworded in words.}\end{response}""",
    r"""\begin{response}
Reworded in words (``at most half the regularization length''), see the quote under point~19.
\end{response}"""),
21: (r"""\begin{response}\mycomment{TODO}{done.}\end{response}

\begin{reviewercomment}
22.""",
    r"""\begin{response}
Done.
\end{response}
\begin{relatedchanges}
\relhead{Section~3.4}
%@quote cmd:sections}{chapters}
\end{relatedchanges}

\begin{reviewercomment}
22."""),
22: (r"""\begin{response}\mycomment{TODO}{``predicted'' $\to$ ``modelled''; sentence qualified.}\end{response}""",
    r"""\begin{response}
We agree. ``Predicted'' was replaced by ``modelled'', and the sentence was qualified: whether the nucleation loads are quantitatively predictive is a separate question, and nucleation is used in this work as a model mechanism with analytically known characteristic stress and energy.
\end{response}
\begin{relatedchanges}
\relhead{Section~3.6}
%@quote cmd:modelled with}{predicted using}
%@quote cmd:Whether the nucleation loads obtained in this way
\end{relatedchanges}"""),
23: (r"""\begin{response}\mycomment{TODO}{``follows from eq.~(21); see [28]''.}\end{response}""",
    r"""\begin{response}
Corrected.
\end{response}
\begin{relatedchanges}
\relhead{Section~3.6}
%@quote cmd:, see Noll et al.}{, cf.}
\end{relatedchanges}"""),
24: (r"""\begin{response}\mycomment{TODO}{quantify with the new ligament-stress output: maximum stress in the ligament at re-nucleation vs.\ the one-dimensional $\sigma^*$ for all $w_s$ (new figure/table); references Kuhn 2013, Schl\"uter 2018.}\end{response}""",
    r"""\begin{response}
The statement was too broad. It now refers to the evidence that exists for two and three dimensions (Kuhn 2013 for the quadratic and Schl\"uter 2018 for the cubic degradation function: nucleation once a stress close to $\sigma_c$ is exceeded in a region of the size of several $\epsilon$), and the new simulations record the stress state in the ligament ahead of the trapped crack tip immediately before re-nucleation, which is compared with the one-dimensional values in Section~\mycomment{TODO}{4.4 (study S1 outputs; numbers to be inserted)}.
\end{response}
\begin{relatedchanges}
\relhead{Section~3.6}
%@quote cmd:Strictly speaking, these relations hold only in one dimension. In two and three dimensions
\end{relatedchanges}"""),
26: (r"""\begin{response}\mycomment{TODO}{done.}\end{response}

%==================================================================================================
\section*{Reviewer 2}""",
    r"""\begin{response}
Done.
\end{response}
\begin{relatedchanges}
\relhead{Conclusion}
%@quote cmd:with}{against}
\end{relatedchanges}

%==================================================================================================
\section*{Reviewer 2}"""),
}
for k in sorted(M):
    E.append(M[k])

# ---------------------------------------------------------------- R3 comment 2
E.append((r"""whether the maximum driving force recorded in a crack-growth experiment with an elasto-plastic bulk material and
unstable re-nucleation can be identified with such a homogenized surface energy. \mycomment{TODO}{quote the rewritten
paragraph of the introduction and the two new bibliography entries.}
\end{response}

\begin{relatedchanges}
\relhead{\mycomment{TODO}{quotes: introduction paragraph on the variational definition; bibliography}}
\end{relatedchanges}""",
r"""whether the maximum driving force recorded in a crack-growth experiment with an elasto-plastic bulk material and
unstable re-nucleation can be identified with such a homogenized surface energy. The references of Braides et al.\ (1996)
and Cagnetti et al.\ (2019) were added to the bibliography, together with Francfort and Marigo (1998) for the variational
formulation whose surface energy is homogenized.
\end{response}

\begin{relatedchanges}
\relhead{Introduction}
%@quote cmd:For elastic and plastic properties, the step from the microstructure
%@quote cmd:Rigorous results exist for the homogenization of surface energies
\end{relatedchanges}"""))


def find_once(text, old):
    if text.count(old) == 1:
        return old
    pat = "".join(("[ \t]*\n" if ch == "\n" else re.escape(ch)) for ch in old)
    hits = list(re.finditer(pat, text))
    return hits[0].group(0) if len(hits) == 1 else None


for old, new in E:
    hit = find_once(s, old)
    if hit is None:
        print("MATCH ERROR:", old[:100].replace("\n", " "))
        sys.exit(1)
    s = s.replace(hit, new)

open(path, "w").write(s)
print("applied", len(E), "edits to", path)
