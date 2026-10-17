#!/usr/bin/env python3
"""Round-2 edits, part 1: Sections 2-5 and conclusion (R1 minor points, R1-M2/M4 framing, semicolons).
Applied to manuscript_revised_template.tex with changes-package markup. Every edit must match exactly once."""
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "manuscript_revised_template.tex"
s = open(path).read()

E = []  # (old, new)

# ---------------------------------------------------------------- Section 2
E.append((r"""\section{A Numerical Experiment to Determine the Effective Crack Resistance
 of Materials with Complex Microstructure}
\label{sec:numericalExperiment}""",
r"""\section{\texorpdfstring{\replaced{A Numerical Experiment for the Crack Resistance of a Resolved Porous Region}{A Numerical Experiment to Determine the Effective Crack Resistance of Materials with Complex Microstructure}}{A Numerical Experiment for the Crack Resistance of a Resolved Porous Region}}
\label{sec:numericalExperiment}"""))

E.append((r"""Rather than conducting a macroscopic experiment, a region containing representative microstructural features is fractured in a numerical simulation while the energetic driving force required for fracture is measured.""",
r"""Rather than conducting a macroscopic experiment, a region \replaced{that resolves the}{containing representative} microstructural features is fractured in a numerical simulation while the energetic driving force required for fracture is measured."""))

E.append((r"""This quantity can subsequently be used as an input quantity in a macroscopic fracture criterion such as~(\ref{eq:fracture_criterion_J}).
Such a use is justified only to the extent that the fracture process analyzed by~(\ref{eq:fracture_criterion_J}) is representative of the specific material and loading situation considered in the numerical experiment.""",
r"""This quantity can subsequently be used as an input quantity in a macroscopic fracture criterion such as \added{eq.~}(\ref{eq:fracture_criterion_J}).
Such a use is justified only to the extent that the fracture process analyzed by \added{eq.~}(\ref{eq:fracture_criterion_J}) is representative of the specific material and loading situation considered in the numerical experiment."""))

E.append((r"""Instead of performing a physical experiment, a representative region
 is selected that explicitly resolves the relevant microstructural features and whose fracture behavior is to be captured by the effective energetic crack resistance.
The term representative volume element is deliberately avoided because the region no longer fulfills classical representativity requirements once fracture occurs, e.g., the separation of length scales.
This region is highlighted in dark gray in Fig.~\ref{fig:numerical_experiment}.""",
r"""Instead of performing a physical experiment, a \replaced{finite region of the material}{representative region}
 is selected that explicitly resolves the relevant microstructural features and whose fracture behavior is to be captured by the effective energetic crack resistance.
\replaced{This region is referred to as the resolved porous region, or simply the resolved region, in the following. The term representative volume element is deliberately avoided. Hill~\cite{hill_elastic_1963} defines a representative volume element as the smallest sample of a heterogeneous material that can be treated as a statistically uniform continuum, so that its effective properties are independent of the applied boundary conditions. The resolved region considered here is a sample of the microstructure whose representativity in this sense is not established, and the separation of length scales underlying Hill's definition is lost once a crack of the size of the region develops.}{The term representative volume element is deliberately avoided because the region no longer fulfills classical representativity requirements once fracture occurs, e.g., the separation of length scales.}
This region is highlighted in dark gray in Fig.~\ref{fig:numerical_experiment}."""))

E.append((r"""The representative  region  is embedded within a surrounding matrix characterized by effective elastic properties,
such as the effective Young's modulus $E^{\text{eff}}$ and Poisson's ratio $\nu^{\text{eff}}$ in the isotropic case.
This surrounding material facilitates the evaluation of surface integrals at the boundary and is shown in light gray in Fig.~\ref{fig:numerical_experiment}.
The union of the representative region and the surrounding matrix is considered as a continuum~$\Omega$ with boundary $\partial\Omega$.
Macroscopic crack growth, corresponding to complete separation of the representative region along a prescribed direction, is enforced through appropriate boundary conditions.""",
r"""The \replaced{resolved}{representative} region is embedded within a surrounding \replaced{homogeneous matrix with the effective elastic stiffness of a unit cell of the microstructure. For the periodic arrangements of circular or spherical pores considered in this work, this stiffness has square or cubic symmetry. In the simulations it is approximated by an isotropic stiffness with the effective Young's modulus $E^{\text{eff}}$ and Poisson's ratio $\nu^{\text{eff}}$. The degree of anisotropy of the unit cells and the influence of this approximation on the effective crack resistance are quantified in Sections~\ref{sec:2D_effective_medium} and~\ref{sec:3D_effective_medium}.}{matrix characterized by effective elastic properties,
such as the effective Young's modulus $E^{\text{eff}}$ and Poisson's ratio $\nu^{\text{eff}}$ in the isotropic case.}
This surrounding material facilitates the evaluation of surface integrals at the boundary and is shown in light gray in Fig.~\ref{fig:numerical_experiment}.
\added{The transition from the resolved region to the homogeneous embedding is a material discontinuity, and the corners of the resolved region act as geometric stress concentrators. In the simulations of Sections~\ref{sec:2Dresults} and~\ref{sec:3D_results}, the corners are located at least one cell width $w_c$ away from the crack plane and the embedding remains linear elastic, so that the J-integral evaluated on the outer boundary is not affected by the local stress concentration at the corners. The influence of the corners on the crack pattern is discussed together with the results.}
The union of the \replaced{resolved}{representative} region and the surrounding matrix is considered as a continuum~$\Omega$ with boundary $\partial\Omega$.
Macroscopic crack growth, corresponding to complete separation of the \replaced{resolved}{representative} region along a prescribed direction, is enforced through appropriate boundary conditions."""))

E.append((r"""The next step consists of simulating the evolution of the microscopic crack pattern until complete separation of the representative region is achieved.
The formulation must allow unrestricted crack evolution, since the interaction between microcracks and microstructural features governs the resulting fracture process, as indicated by the red crack paths in Fig.~\ref{fig:numerical_experiment}.
Furthermore, the simulation must be based on physically consistent principles.
An initial""",
r"""The next step consists of simulating the evolution of the microscopic crack pattern until complete separation of the \replaced{resolved}{representative} region is achieved.
The formulation must allow unrestricted crack evolution, since the interaction between microcracks and microstructural features governs the resulting fracture process, as indicated by the red crack paths in Fig.~\ref{fig:numerical_experiment}.
\deleted{Furthermore, the simulation must be based on physically consistent principles.}
An initial"""))

E.append((r"""where $\boldsymbol{e}_x$ denotes the unit vector in the macroscopic crack propagation direction; hence, the index $x$ indicates""",
r"""where $\boldsymbol{e}_x$ denotes the unit vector in the macroscopic crack propagation direction\replaced{. Hence,}{; hence,} the index $x$ indicates"""))

E.append((r"""Only small strain assumptions are considered in this work.
The recorded value is, of course, affected by the crack pattern inside the explicitly resolved microstructure; however, following the observation reported by Hossain et al.~\cite{hossain_effective_2014}, the additional homogeneous surrounding material is assumed not to change the value of the boundary J-Integral, so that evaluating it at the boundary of the resolved microstructure or at the boundary of an embedding region with representative effective properties gives the same macroscopic driving force.""",
r"""Only small strain\deleted{ assumption}s are considered in this work.
\replaced{The recorded value depends on the crack pattern inside the resolved microstructure. Following the observation reported by Hossain et al.~\cite{hossain_effective_2014}, the homogeneous surrounding material is assumed not to change the value of the boundary J-integral. Under this assumption, the J-integral evaluated at the outer boundary of the embedding equals the macroscopic driving force acting on the resolved region.}{The recorded value is, of course, affected by the crack pattern inside the explicitly resolved microstructure; however, following the observation reported by Hossain et al.~\cite{hossain_effective_2014}, the additional homogeneous surrounding material is assumed not to change the value of the boundary J-Integral, so that evaluating it at the boundary of the resolved microstructure or at the boundary of an embedding region with representative effective properties gives the same macroscopic driving force.}"""))

E.append((r"""during characteristic crack growth through the representative region,""",
r"""during characteristic crack growth through the \replaced{resolved}{representative} region,"""))

E.append((r"""The index $c$ denotes the critical crack-resistance value obtained from this maximum, whereas the index $x$ is retained to avoid implying isotropic crack resistance.""",
r"""The index $c$ \replaced{indicates that the crack-resistance value is obtained as a maximum, i.e., is \textsl{critical}}{denotes the critical crack-resistance value obtained from this maximum}, whereas the index $x$ is retained to avoid implying isotropic crack resistance."""))

E.append((r"""understood as full separation of the representative region.
This definition should not be confused with a proof that $J_{x,c}^{\text{eff}}$ converges to a material property under enlargement of the representative region. Establishing such convergence, if it exists for a given class of microstructures and loading paths, would require a separate study.""",
r"""understood as full separation of the \replaced{resolved}{representative} region.
\replaced{This definition does not imply that $J_{x,c}^{\text{eff}}$ becomes independent of the size of the resolved region and of the position of the initial crack when the region is enlarged. Whether this is the case for a given class of microstructures and loading paths is a question of representativity that is not settled by the definition.}{This definition should not be confused with a proof that $J_{x,c}^{\text{eff}}$ converges to a material property under enlargement of the representative region. Establishing such convergence, if it exists for a given class of microstructures and loading paths, would require a separate study.}"""))

E.append((r"""Independence from the initial microscopic crack configuration and adequate representation of fracture mechanisms can be ensured by selecting a sufficiently large representative region that contains an appropriate number of microstructural features.
This presumes a sufficiently homogeneous spatial distribution of microstructural features, analogous to assumptions made for representative volume elements in micromechanics.
In the present work, this requirement is not fully verified: only one prescribed macroscopic crack-growth direction and one initial crack location are considered, and no convergence study with respect to the size of the resolved porous region is performed. The results should therefore be read as a controlled numerical experiment on the influence of J2-plasticity for the selected simplified microstructure, rather than as the determination of a generally valid homogenized fracture toughness.""",
r"""\replaced{Whether $J_{x,c}^{\text{eff}}$ is independent of the initial microscopic crack configuration and of the size of the resolved region cannot be taken for granted and has to be tested for every microstructure. In the present work, only one macroscopic crack-growth direction is considered. The influence of the length of the initial crack and of the number of resolved pore rows on $J_{x,c}^{\text{eff}}$ is probed in Section~\ref{sec:2D_representativity}. A representativity study in the sense of Hill~\cite{hill_elastic_1963}, i.e., a statistical assessment over microstructure realizations, sample sizes and loading conditions, is outside the scope of this work.}{Independence from the initial microscopic crack configuration and adequate representation of fracture mechanisms can be ensured by selecting a sufficiently large representative region that contains an appropriate number of microstructural features.
This presumes a sufficiently homogeneous spatial distribution of microstructural features, analogous to assumptions made for representative volume elements in micromechanics.
In the present work, this requirement is not fully verified: only one prescribed macroscopic crack-growth direction and one initial crack location are considered, and no convergence study with respect to the size of the resolved porous region is performed.} The results should therefore be read as a controlled numerical experiment on the influence of J2-plasticity for the selected simplified microstructure, rather than as the determination of a generally valid homogenized fracture toughness."""))

# ---------------------------------------------------------------- Section 3
E.append((r"""\section{Simulation of fracture with phase field models}\label{sec:simulation_phase_field_models}

This section introduces the phase-field models and numerical ingredients used in the subsequent numerical experiments.

\subsection{A phase field model of ductile fracture}\label{sec:phase_fiel_model_ductile}

In order to model fracture in ductile materials under quasi-static loading conditions, we employ the model proposed in~\cite{noll_3d_2020}. """,
r"""\section{Simulation of fracture with phase field models}\label{sec:simulation_phase_field_models}

\deleted{This section introduces the phase-field models and numerical ingredients used in the subsequent numerical experiments.}

\subsection{A phase field model of ductile fracture}\label{sec:phase_fiel_model_ductile}

\added{This section introduces the phase-field models and the numerical ingredients used in the subsequent numerical experiments.}
In order to model fracture in ductile materials under quasi-static loading conditions, we employ the model proposed in~\cite{noll_3d_2020}. """))

E.append((r"""follow standard computational plasticity concepts; see, for example, Simo and Hughes~\cite{simo_computational_1998}.""",
r"""follow standard computational plasticity concepts\replaced{, see}{; see}, for example, Simo and Hughes~\cite{simo_computational_1998}."""))

E.append((r"""    \quad f\,\dot{\gamma} = 0
\end{equation}
must be satisfied""",
r"""    \quad f\,\dot{\gamma} = 0\changed{,}
\end{equation}
must be satisfied"""))

E.append((r"""    \left(\boldsymbol{\varepsilon}-\boldsymbol{\varepsilon}^p\right)\right).
\end{equation}
Here, $\boldsymbol{\mathbb{C}}$ denotes the elastic stiffness tensor, which is degraded in the presence of cracks through the degradation function""",
r"""    \left(\boldsymbol{\varepsilon}-\boldsymbol{\varepsilon}^p\right)\right).
    \label{eq:psi_el}
\end{equation}
Here, $\boldsymbol{\mathbb{C}}$ denotes the elastic stiffness tensor\added{ of the undamaged material. Throughout this work, the bulk material is assumed to be isotropic, so that $\boldsymbol{\mathbb{C}}$ is characterized by the bulk modulus $K$ and the shear modulus $\mu$, or equivalently by the Lam\'e parameters $\lambda = K - \nicefrac{2}{3}\mu$ and $\mu$. The stiffness}\deleted{, which} is degraded in the presence of cracks through the degradation function"""))

E.append((r"""with $\beta = 0.2$ controlling the slope of $g(s)$ at $s=1$, and $\eta = 10^{-5}$ serving as a residual stiffness parameter to avoid ill-conditioned tangents if the problem is solved using a Newton scheme.""",
r"""with $\beta = 0.2$ controlling the slope of $g(s)$ at $s=1$, and $\eta = 10^{-5}$ \replaced{a small residual stiffness. A positive value of $\eta$ keeps the stiffness positive definite in fully fractured regions, $s = 0$, so that the equilibrium problem remains well-posed there. It is therefore a part of the model and not a device of the numerical solution procedure.}{serving as a residual stiffness parameter to avoid ill-conditioned tangents if the problem is solved using a Newton scheme.}"""))

E.append((r"""In this sense, the regularized fields $(\boldsymbol{u},s)$ approximate the displacement field and crack set of the corresponding sharp-crack formulation, while the diffuse crack region represented by low values of $s$ approaches the crack set; no pointwise convergence statement for $s$ is intended here, see Kuhn~\cite{kuhn_numerical_2013} and Chambolle~\cite{chambolle_approximation_2004}.""",
r"""\replaced{In this sense, the regularized fields $(\boldsymbol{u},s)$ approximate the displacement field and the crack set of the corresponding sharp-crack formulation. The convergence is that of the energy functionals and of their minimizers, and the region of low values of $s$ concentrates on the crack set as $\epsilon \to 0$. No pointwise convergence of $s$ is implied, see Kuhn~\cite{kuhn_numerical_2013} and Chambolle~\cite{chambolle_approximation_2004}.}{In this sense, the regularized fields $(\boldsymbol{u},s)$ approximate the displacement field and crack set of the corresponding sharp-crack formulation, while the diffuse crack region represented by low values of $s$ approaches the crack set; no pointwise convergence statement for $s$ is intended here, see Kuhn~\cite{kuhn_numerical_2013} and Chambolle~\cite{chambolle_approximation_2004}.}"""))

E.append((r"""The mobility parameter $M$ arises from the introduction of a viscous regularization and facilitates the numerical solution of fracture problems under quasi-static loading conditions.
Accordingly, static equilibrium is assumed in the form""",
r"""\replaced{The parameter $M$ is the kinetic coefficient of the evolution equation~(\ref{eq:evolution_s}). It sets the time scale on which the phase field relaxes towards a stationary point of the energy, and for large values of $M$ the evolution of $s$ approximates a rate-independent process. The energy dissipated by this viscous evolution, $\int_\Omega \dot{s}^2/M\,\mathrm{d}V$ per unit time, is part of the energy balance of the model and is quantified for the numerical experiments in Section~\ref{sec:2D_energy_partition}.}{The mobility parameter $M$ arises from the introduction of a viscous regularization and facilitates the numerical solution of fracture problems under quasi-static loading conditions.}
\replaced{Static}{Accordingly, static} equilibrium is assumed in the form"""))

E.append((r"""where the stress tensor $\boldsymbol{\sigma}$ is computed under the assumption of isotropic linear elasticity as""",
r"""where the stress tensor $\boldsymbol{\sigma}$ \replaced{follows from eq.~(\ref{eq:psi_el}) for the isotropic stiffness as}{is computed under the assumption of isotropic linear elasticity as}"""))

E.append((r"""are very similar to (\ref{eq:evolution_s}) and (\ref{eq:equilibrium}) but don't have
any contributions of the plastic variables  """,
r"""are very similar to \added{eqs.~}(\ref{eq:evolution_s}) and (\ref{eq:equilibrium}) but \replaced{do not contain}{don't have}
any contributions of the plastic variables\replaced{:}{  }"""))

# Sec. 3.3: discretization -> assembly -> solver (minor 18, 19, 20)
E.append((r"""The weak forms of the coupled boundary-value problems are implemented in FEniCSx using \texttt{dolfinx} version \replaced{0.7.3}{0.7.4}~\cite{fenicsx_documentation}. The displacement field, the phase field, and, where applicable, the plastic internal variables are solved in a monolithic nonlinear system. The nonlinear residual and its Gateaux derivative are assembled and solved with the PETSc-based Newton solver provided by FEniCSx. The linear subproblems arising within Newton's method are solved with the default PETSc Krylov solver GMRES and the default PETSc ILU preconditioner used by this setup.

The evolution equation for the phase field, Eq.~(\ref{eq:evolution_s}), is discretized in pseudo-time by an implicit backward Euler scheme. The time step is adapted automatically during the simulation. The maximum number of Newton iterations per time step is set to eight. If a step does not converge within this limit, the time step is reduced by a factor of two and the step is repeated. If convergence is reached in fewer than four Newton iterations, the next time step is increased by a factor of two, subject to the prescribed upper bound for the respective simulation.

The spatial discretization uses first-order Lagrange finite elements on triangular meshes in two dimensions and tetrahedral meshes in three dimensions. Along the expected crack path, the mesh is locally refined such that the element edge length satisfies $\Delta h \leq \epsilon/2$ and is usually chosen much smaller; see the respective sections for the specific values. The specific values of $\Delta h$ and $\epsilon$ used in the individual simulations are stated together with the corresponding numerical setups below.""",
r"""\added{The spatial discretization uses first-order Lagrange finite elements on triangular meshes in two dimensions and tetrahedral meshes in three dimensions. Along the expected crack path, the mesh is locally refined such that the element edge length $\Delta h$ is at most half the regularization length $\epsilon$, and it is usually chosen much smaller. The specific values of $\Delta h$ and $\epsilon$ used in the individual simulations are stated together with the corresponding numerical setups below. The evolution equation for the phase field, eq.~(\ref{eq:evolution_s}), is discretized in pseudo-time by an implicit backward Euler scheme.}

\added{The weak forms of the coupled boundary-value problems are implemented in FEniCSx using \texttt{dolfinx} version 0.7.3~\cite{fenicsx_documentation}. The displacement field, the phase field, and, where applicable, the plastic internal variables form one monolithic nonlinear system per time step. The nonlinear residual and its derivative are assembled by FEniCSx, and the resulting system is solved with the PETSc-based Newton solver provided by FEniCSx. The linear subproblems arising within Newton's method are solved with the default PETSc Krylov solver GMRES and the default PETSc ILU preconditioner used by this setup. The time step is adapted automatically during the simulation. The maximum number of Newton iterations per time step is set to eight. If a step does not converge within this limit, the time step is reduced by a factor of two and the step is repeated. If convergence is reached in fewer than four Newton iterations, the next time step is increased by a factor of two, subject to the prescribed upper bound for the respective simulation.}

\deleted{The weak forms of the coupled boundary-value problems are implemented in FEniCSx using dolfinx version 0.7.4~\cite{fenicsx_documentation}. The displacement field, the phase field, and, where applicable, the plastic internal variables are solved in a monolithic nonlinear system. The nonlinear residual and its Gateaux derivative are assembled and solved with the PETSc-based Newton solver provided by FEniCSx. The linear subproblems arising within Newton's method are solved with the default PETSc Krylov solver GMRES and the default PETSc ILU preconditioner used by this setup.}

\deleted{The evolution equation for the phase field, Eq.~(\ref{eq:evolution_s}), is discretized in pseudo-time by an implicit backward Euler scheme. The time step is adapted automatically during the simulation. The maximum number of Newton iterations per time step is set to eight. If a step does not converge within this limit, the time step is reduced by a factor of two and the step is repeated. If convergence is reached in fewer than four Newton iterations, the next time step is increased by a factor of two, subject to the prescribed upper bound for the respective simulation.}

\deleted{The spatial discretization uses first-order Lagrange finite elements on triangular meshes in two dimensions and tetrahedral meshes in three dimensions. Along the expected crack path, the mesh is locally refined such that the element edge length satisfies $\Delta h \leq \epsilon/2$ and is usually chosen much smaller; see the respective sections for the specific values. The specific values of $\Delta h$ and $\epsilon$ used in the individual simulations are stated together with the corresponding numerical setups below.}"""))

E.append((r"""In the following chapters, the influence of the material law""",
r"""In the following \replaced{sections}{chapters}, the influence of the material law"""))

E.append((r"""In order to highlight the connection of the presented phase field model to the fracture criterion~(\ref{eq:fracture_criterion_J}), a simulation was performed""",
r"""In order to highlight the connection of the presented phase field model to the fracture criterion\added{, eq.~}(\ref{eq:fracture_criterion_J})\added{,} a simulation was performed"""))

E.append((r"""As discussed in section~\ref{sec:numericalExperiment}, the J-Integral~(\ref{eq:J-integral}) evaluated at the boundary""",
r"""As discussed in section~\ref{sec:numericalExperiment}, the J-Integral\added{, eq.~}(\ref{eq:J-integral})\added{,} evaluated at the boundary"""))

E.append((r"""so that the linear elastic formulation of the J-Integral (\ref{eq:J-integral}) remains a good approximation""",
r"""so that the linear elastic formulation of the J-Integral\added{, eq.~}(\ref{eq:J-integral})\added{,} remains a good approximation"""))

E.append((r"""In addition to satisfying the fracture criterion~(\ref{eq:fracture_criterion_J}) for pre-existing cracks, crack nucleation in previously undamaged material can also be predicted using phase-field models.""",
r"""In addition to satisfying the fracture criterion\added{, eq.~}(\ref{eq:fracture_criterion_J})\added{,} for pre-existing cracks, crack nucleation in previously undamaged material can also be \replaced{modelled with}{predicted using} phase-field models.\added{ Whether the nucleation loads obtained in this way are quantitatively predictive for a given material is a separate question, which depends on the regularization length and on the degradation function, see Tann\'e et al.~\cite{tanne_crack_2018}. In the present work, nucleation is used as a model mechanism whose characteristic stress and energy are known analytically.}"""))

E.append((r"""In the one-dimensional linear elastic case with the cubic degradation function~(\ref{eq:deg_grad}) and $\beta = 0$,""",
r"""In the one-dimensional linear elastic case with the cubic degradation function \added{of eq.~}(\ref{eq:deg_grad}) and $\beta = 0$,"""))

E.append((r"""For the ductile phase-field fracture model, also employing the cubic degradation function~(\ref{eq:deg_grad}) with $\beta = 0$,""",
r"""For the ductile phase-field fracture model, also employing the cubic degradation function \added{of eq.~}(\ref{eq:deg_grad}) with $\beta = 0$,"""))

E.append((r"""for $E = 2\mu_0$, and the corresponding critical stress then follows from~(\ref{eq:sigma_plast}), cf.~\cite{noll_3d_2020}. Strictly speaking, these relations hold only in one dimension, but they provide good estimates for two- and three-dimensional scenarios, where cracks tend to form spontaneously once a stress level close to $\sigma_c$ is exceeded in a sufficiently large region—typically on the order of several multiples of $\epsilon$, cf.~\cite{kuhn_numerical_2013}. Due to the relevance""",
r"""for $E = 2\mu_0$, and the corresponding critical stress then follows from \added{eq.~}(\ref{eq:sigma_plast})\replaced{, see Noll et al.}{, cf.}~\cite{noll_3d_2020}. \replaced{Strictly speaking, these relations hold only in one dimension. In two and three dimensions, cracks nucleate once a stress level close to $\sigma_c$ is exceeded in a region whose size is of the order of several multiples of $\epsilon$, as shown for the quadratic degradation function by Kuhn~\cite{kuhn_numerical_2013} and for the cubic degradation function by Schl\"uter~\cite{schlueter_phase_2018}. For the porous specimens studied in this work, the stress state in the ligament ahead of a trapped crack tip immediately before re-nucleation is recorded and compared with the one-dimensional values in Section~\ref{sec:2D_mechanisms}.}{Strictly speaking, these relations hold only in one dimension, but they provide good estimates for two- and three-dimensional scenarios, where cracks tend to form spontaneously once a stress level close to $\sigma_c$ is exceeded in a sufficiently large region—typically on the order of several multiples of $\epsilon$, cf.~\cite{kuhn_numerical_2013}.} Due to the relevance"""))

# ---------------------------------------------------------------- Section 4
E.append((r"""Here, $k$ denotes the load case, i.e., the prescribed macroscopic strain for which the linear displacement boundary condition is applied; several load cases are required to determine the full stiffness tensor.""",
r"""Here, $k$ denotes the load case, i.e., the prescribed macroscopic strain for which the linear displacement boundary condition is applied\replaced{. Several}{; several} load cases are required to determine the full stiffness tensor.\added{ The pore surface is traction free.}"""))

E.append((r"""    \nu_0^{\text{eff}} = \dfrac{\lambda^{\text{eff}}}{2 (\lambda^{\text{eff}} + \mu^{\text{eff}})}.
\end{equation}

The $6 \times 3$ porous domain""",
r"""    \nu_0^{\text{eff}} = \dfrac{\lambda^{\text{eff}}}{2 (\lambda^{\text{eff}} + \mu^{\text{eff}})}.
    \label{eq:isotropic_approximation_2D}
\end{equation}
\added{The unit cell has square symmetry, so that $\left< \underline{C} \right>_{11} = \left< \underline{C} \right>_{22}$ and the effective stiffness is characterized by the three independent constants $C_{11} = \left< \underline{C} \right>_{11}$, $C_{12} = \left< \underline{C} \right>_{12}$ and $C_{33} = \left< \underline{C} \right>_{33}$. The isotropic approximation of eq.~(\ref{eq:isotropic_approximation_2D}) uses $C_{12}$ and $C_{33}$ and replaces $C_{11}$ by $C_{12} + 2C_{33}$. The deviation from isotropy is measured by the Zener ratio $A = 2C_{33}/(C_{11} - C_{12})$, which equals one for an isotropic material. Table~\ref{tab:effective_stiffness_2D} lists the effective constants and the Zener ratio for all wall widths considered. The anisotropy is largest for the smallest wall width and vanishes for widely spaced pores. The influence of the isotropic approximation on the effective crack resistance is tested in Section~\ref{sec:2D_cubic_embedding} by simulations in which the embedding is assigned the full square-symmetric stiffness.}
\mycomment{TODO}{Table \ref{tab:effective_stiffness_2D} from effective\_stiffness\_table.csv of study S1 (C11, C12, C33, lam, mu, E, nu, Zener per $w_s$); first values: $A = 0.89$ ($w_s = 0.375L$), $0.96$ ($1.0L$), $1.00$ ($4.0L$).}

The $6 \times 3$ porous domain"""))

E.append((r"""The parameters $\sigma_y$ and $H$ entering the nonlinear part of the material law~(\ref{eq:sigma_plast}) are not homogenized.
Instead, it is assumed that nonlinear effects are locally relevant, while the effective behavior of each unit cell can be approximated as linear elastic with parameters $E_0^{\text{eff}}$ and $\nu_0^{\text{eff}}$. """,
r"""The parameters $\sigma_y$ and $H$ entering the nonlinear part of the material law\added{, eq.~}(\ref{eq:sigma_plast})\added{,} are not homogenized.
Instead, it is assumed that nonlinear effects are locally relevant, while the effective behavior of each unit cell can be approximated as linear elastic with parameters $E_0^{\text{eff}}$ and $\nu_0^{\text{eff}}$. \added{The embedding is therefore modelled as linear elastic, i.e., no yielding is admitted outside the resolved porous region.}"""))

E.append((r"""and the additional energy dissipated during ligament failure; in particular, once the one-dimensional tensile strength""",
r"""and the additional energy dissipated during ligament failure\replaced{. In}{; in} particular, once the one-dimensional tensile strength"""))

E.append((r"""The following parameter study shows that this ordering is representative of the main trend,""",
r"""The following parameter study shows that this ordering \replaced{reflects}{is representative of} the main trend,"""))

E.append((r"""lies between those of \EqJcLabel and \EqSigmaStarLabel; see also Fig.~\ref{fig:energy_vs_displacement}.""",
r"""lies between those of \EqJcLabel and \EqSigmaStarLabel\replaced{, see}{; see} also Fig.~\ref{fig:energy_vs_displacement}."""))

# placeholders for the new subsections (results of studies S3/S4/S5 and the energy partition)
E.append((r"""The comparison of \vMLabel and \EqSigmaStarLabel in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets} shows that the ductile material law leads to a substantially higher effective crack resistance even though the one-dimensional tensile strengths are almost identical.
""",
r"""The comparison of \vMLabel and \EqSigmaStarLabel in Fig.~\ref{fig:PAPER_06b_Jx_vs_wsteg_three_datasets} shows that the ductile material law leads to a substantially higher effective crack resistance even though the one-dimensional tensile strengths are almost identical.

\subsection{\added{Energy Partition of the Numerical Experiment}}
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
"""))

# ---------------------------------------------------------------- Section 5
E.append((r"""The J-integral is computed according to (\ref{eq:J-integral}) and normalized by the domain width in the $z$-direction for comparison with the two-dimensional plane strain results. This normalization does not change the definition of the J-integral; it converts the three-dimensional boundary integral to a value per unit out-of-plane thickness.""",
r"""The J-integral is computed according to \added{eq.~}(\ref{eq:J-integral}) and normalized by the domain width in the $z$-direction for comparison with the two-dimensional plane strain results. This normalization does not change the definition of the J-integral\replaced{. It}{; it} converts the three-dimensional boundary integral to a value per unit out-of-plane thickness."""))

E.append((r"""numerical fracture resistance of the von Mises-type material law $J_x^{\text{num,pl}}\approx1.50J_c^0$; see Section~\ref{sec:steady_growth_griffith}.""",
r"""numerical fracture resistance of the von Mises-type material law $J_x^{\text{num,pl}}\approx1.50J_c^0$\replaced{, see}{; see} Section~\ref{sec:steady_growth_griffith}."""))

E.append((r"""The periodic pattern between pores one and three
% two and four
 is considered representative of the microstructure and is shown""",
r"""The periodic pattern between pores one and three
% two and four
 is considered \replaced{characteristic}{representative} of the microstructure and is shown"""))

E.append((r"""Additionally, the displacement $u_z$ is constrained to zero on the planes $z=z_{\rm min}$ and
$z=z_{\rm max}$ to mimic plane strain while still accounting for the three-dimensional character of the geometry of the microstructure.""",
r"""\replaced{The surfing displacements with $u_z = 0$ are prescribed on the entire outer boundary, including the planes $z=z_{\rm min}$ and $z=z_{\rm max}$, which mimics}{Additionally, the displacement $u_z$ is constrained to zero on the planes $z=z_{\rm min}$ and
$z=z_{\rm max}$ to mimic} plane strain while still accounting for the three-dimensional character of the geometry of the microstructure."""))

E.append((r"""$\left< \underline{\boldsymbol{C}} \right>$ is of size $6\times6$. The effective Lam\'e parameters are determined as""",
r"""$\left< \underline{\boldsymbol{C}} \right>$ is of size $6\times6$. \added{The unit cell has cubic symmetry with the three independent constants $C_{11}$, $C_{12}$ and $C_{44}$ and the Zener ratio $A = 2C_{44}/(C_{11}-C_{12})$, which are listed in Table~\ref{tab:effective_stiffness_3D}.} The effective Lam\'e parameters \added{of the isotropic approximation} are determined as"""))

E.append((r"""    \mu^{\text{eff}} = \dfrac{\left< \underline{C} \right>_{44} + \left< \underline{C} \right>_{55} + \left< \underline{C} \right>_{66}}{3}.
\end{equation}
""",
r"""    \mu^{\text{eff}} = \dfrac{\left< \underline{C} \right>_{44} + \left< \underline{C} \right>_{55} + \left< \underline{C} \right>_{66}}{3}.
\end{equation}
\mycomment{TODO}{Table \ref{tab:effective_stiffness_3D} from study S1\_3D; influence of the cubic embedding (study S3\_3D, $w_s = 0.375L, 1.0L$) to be reported in Section~\ref{sec:3D_fracture_behavior}.}
"""))

# ---------------------------------------------------------------- Conclusion
E.append((r"""compared the results against multiple linear elastic reference models. The methodology followed
the numerical experiment proposed by Hossain et al.~\cite{hossain_effective_2014}, applying
surfing boundary conditions to simulate steady macroscopic crack propagation in a representative
microstructure.""",
r"""compared the results \replaced{with}{against} multiple linear elastic reference models. The methodology followed
the numerical experiment proposed by Hossain et al.~\cite{hossain_effective_2014}, applying
surfing boundary conditions to simulate steady macroscopic crack propagation in a \replaced{resolved porous region}{representative
microstructure}."""))

# total energy equation gets a label (referenced by the energy partition section)
E.append((r"""    + \psi_{\rm{pl}}(\alpha,s)
    \right] \, \mathrm{d}V,
\end{equation}""",
r"""    + \psi_{\rm{pl}}(\alpha,s)
    \right] \, \mathrm{d}V,
    \label{eq:E_ep_total}
\end{equation}"""))

import re


def find_once(text, old):
    """exact match, or a match that ignores trailing blanks before line breaks (the editor strips them)"""
    if text.count(old) == 1:
        return old
    pat = re.escape(old).replace(r"\\n", r"[ \t]*\n").replace("\\\n", "[ \t]*\n")
    pat = "".join(("[ \t]*\n" if ch == "\n" else re.escape(ch)) for ch in old)
    hits = list(re.finditer(pat, text))
    return hits[0].group(0) if len(hits) == 1 else None


for old, new in E:
    hit = find_once(s, old)
    if hit is None:
        print("MATCH ERROR:", old[:90].replace("\n", " "))
        sys.exit(1)
    s = s.replace(hit, new)

open(path, "w").write(s)
print("applied", len(E), "edits to", path)
