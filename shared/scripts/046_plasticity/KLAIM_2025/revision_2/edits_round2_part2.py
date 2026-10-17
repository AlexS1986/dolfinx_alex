#!/usr/bin/env python3
"""Round-2 edits, part 2: introduction restructured (R1 major 1 and 2, minors 1 and 2, R3 comment 2) and
ten new bibliography entries. Same markup conventions as part 1; every edit must match exactly once."""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "manuscript_revised_template.tex"
s = open(path).read()
E = []

# -------------------------------------------------------------------- paragraph 1 (problem, criteria)
E.append((r"""While homogenization techniques for elastic and plastic material properties are well established,
the homogenization of fracture toughness remains a comparatively young field of research.
Relevant quantities include the critical stress intensity factor \( K_{Ic} \),
appearing in Irwin’s criterion \( K_I = K_{Ic} \), and the critical energy release rate \( G_c \),
used in Griffith’s criterion \( G = G_c \).
Similarly,  more general fracture formulations based on the J-integral $J$ can be established, i.e.
\begin{equation}
    J = J_c,
    \label{eq:fracture_criterion_J}
\end{equation}
where \( J_c \) denotes the critical value of the J-integral and serves as a material parameter.
Such criteria
remain applicable up to the onset of crack growth under monotonic loading even
in the presence of plasticity, as discussed by Gross and Seelig~\cite{gross_fracture_2011}.



A reliable determination of the fracture resistance parameters which are properties of the
underlying material and microstructure is of significant
practical interest, particularly in light of the increasing use of
additive manufacturing technologies. For example, wire arc additive manufacturing (WAAM) enables the flexible combination of materials
and targeted microstructural design through the control of process parameters,
as demonstrated by Hauser et al.~\cite{hauser_porosity_2021}. Fracture-mechanics-based criteria are often the appropriate tools
for strength assessment in these defect-rich materials.
However, their application requires material- and microstructure-dependent
fracture resistance parameters as mentioned above.
""",
r"""\replaced{Porous and other heterogeneous materials are increasingly used in load-bearing components. Additive manufacturing processes such as wire arc additive manufacturing (WAAM) allow materials to be combined and microstructures to be designed through the process parameters, as demonstrated by Hauser et al.~\cite{hauser_porosity_2021}, and pores are a typical defect of these processes. The strength assessment of such defect-rich materials is often based on fracture mechanics, which requires a fracture resistance parameter of the material. Relevant quantities are the critical stress intensity factor $K_{Ic}$ of Irwin's criterion $K_I = K_{Ic}$, the critical energy release rate $G_c$ of Griffith's criterion $G = G_c$ and, more generally, a critical value $J_c$ of the J-integral $J$ in the criterion}{While homogenization techniques for elastic and plastic material properties are well established,
the homogenization of fracture toughness remains a comparatively young field of research.
Relevant quantities include the critical stress intensity factor \( K_{Ic} \),
appearing in Irwin’s criterion \( K_I = K_{Ic} \), and the critical energy release rate \( G_c \),
used in Griffith’s criterion \( G = G_c \).
Similarly,  more general fracture formulations based on the J-integral $J$ can be established, i.e.}
\begin{equation}
    J = J_c,
    \label{eq:fracture_criterion_J}
\end{equation}
where \( J_c \) denotes the critical value of the J-integral and serves as a material parameter.
Such criteria
remain applicable up to the onset of crack growth under monotonic loading even
in the presence of plasticity, as discussed by Gross and Seelig~\cite{gross_fracture_2011}.
\added{For a heterogeneous material, the fracture resistance entering eq.~(\ref{eq:fracture_criterion_J}) is not a property of the bulk material alone but depends on the microstructure. How it can be determined for a porous material with a ductile bulk material is the subject of this work.}

\deleted{A reliable determination of the fracture resistance parameters which are properties of the
underlying material and microstructure is of significant
practical interest, particularly in light of the increasing use of
additive manufacturing technologies. For example, wire arc additive manufacturing (WAAM) enables the flexible combination of materials
and targeted microstructural design through the control of process parameters,
as demonstrated by Hauser et al.~\cite{hauser_porosity_2021}. Fracture-mechanics-based criteria are often the appropriate tools
for strength assessment in these defect-rich materials.
However, their application requires material- and microstructure-dependent
fracture resistance parameters as mentioned above.}
"""))

# -------------------------------------------------------------------- approaches and limitations
E.append((r"""The existence of an effective fracture toughness for a given material
and the question of how it should be defined remain subjects
of active debate, as discussed by Lebihain~\cite{lebihain_brittle_2021},
with several approaches proposed in the literature.
In contrast to the effective elastic or plastic properties obtained in classical micromechanics, it is presently not established in general that fracture in a heterogeneous material gives rise to a homogenized macroscopic material model with a well-defined fracture toughness obtained in an RVE limit.
To avoid conflating two slightly different concepts, the term \textsl{effective crack resistance} is used in this work for the value obtained from a numerical experiment under specific less general conditions, i.e., the maximum energetic driving force required to force crack advance through the considered microstructure. In contrast, \textsl{fracture toughness} is reserved for a generalized material quantity in the sense of a macroscopic fracture property. Whether such a fracture toughness exists for heterogeneous materials and under which circumstances both concepts might coincide remains an open question of research.
Schneider~\cite{schneider_fftbased_2020} suggests a variational definition that
relies on rigorous mathematical results for the homogenization of
the energy functional occurring in Bourdin's popular variational formulation of brittle fracture~\cite{bourdin_variational_2008}.
This idea is further pursued by
Ernesti and Schneider~\cite{ernesti_fast_2021,ernesti_computing_2022} for anisotropic fracture resistance.

Michel and Suquet~\cite{michel_merits_2022}
assess essentially the same variational definition and discuss its merits and limitations. In particular, they relate the variational surface energy to limit-load problems and show that it provides a lower bound to a more physically motivated effective fracture energy based on the energy dissipated along the actual crack path through a unit cell.

Recent contributions underline that this remains an active field of research. Liu and Shen~\cite{liu_effective_2025} propose a phase-field formulation for anisotropic effective fracture toughness that accounts for a quasi-worst-case translation of a representative volume element. Aranda and Segurado~\cite{aranda_cracklength_2025} introduce a crack-length control technique for phase-field fracture in FFT homogenization and use it to estimate effective fracture properties of composites and porous materials. Man et al.~\cite{man_collaborative_2025} study how crack guiding and trapping in bioinspired interfaces control effective toughness, while Henry~\cite{henry_phasefield_2025} shows for dynamic crack propagation in disordered heterogeneous materials that the spatial distribution of soft inclusions can strongly affect the apparent fracture energy even when the effective elastic properties are similar.

In this work, we adopt the methodology introduced
by Hossain et al.~\cite{hossain_effective_2014} and further
developed and applied by Hsueh et al.~\cite{hsueh_stress_2018} for layered materials,
as well as by Brach et al.~\cite{brach_anisotropy_2019} for anisotropic effective crack resistance,
to determine the effective crack resistance of heterogeneous materials.
All of these contributions were limited to linear-elastic material behavior and two-dimensional problems.

The methodology is not used here as a proof of a homogenized fracture toughness in the mathematical sense. Rather, it is interpreted as a numerical counterpart of a fracture experiment.
In this experiment-like setting,
a representative material region containing
 microstructural features is subjected
 to so-called surfing boundary conditions that enforce complete separation, and the energetic driving force required for this process is recorded.
  A recent related development is the variable stiffness boundary condition of Hao et al.~\cite{hao_variable_2026}, which builds on the surfing boundary condition but translates a simple uniform remote displacement into a non-uniform load through an engineered stiffness gradient, thereby facilitating implementation in experiments and commercial finite-element software.
The resulting microstructural crack evolution is
then simulated using a phase-field model.
See, for example, the review by Ambati et al.~\cite{ambati_review_2015} for a comprehensive overview of phase-field models for fracture.
For ductile materials, phase-field fracture has also been coupled to plasticity in variational finite-strain formulations, for example by Miehe, Aldakheel and co-workers~\cite{miehe_phasefield_ductile_2016,miehe_phasefield_porous_2016}.
""",
r"""\added{For elastic and plastic properties, the step from the microstructure to a macroscopic material model is provided by homogenization. Its central concept is the representative volume element, defined by Hill~\cite{hill_elastic_1963} as the smallest sample of a heterogeneous material that can be treated as a statistically uniform continuum. The effective properties of such a sample are independent of the boundary conditions applied to it, see also Nemat-Nasser and Hori~\cite{nemat-nasser_micromechanics_2013}. Fracture does not fit into this framework in a direct way. A crack is a localized and evolving discontinuity, its growth is a non-stationary process that may be unstable, and the separation of length scales that underlies the notion of a representative volume element is lost once the crack interacts with individual microstructural features. For this reason, the term \textsl{fracture toughness} is used in this work only for a macroscopic material property in the sense of eq.~(\ref{eq:fracture_criterion_J}), whereas the value obtained from a numerical experiment under specific conditions, i.e., the maximum energetic driving force required to force crack advance through the considered microstructure, is called the \textsl{effective crack resistance}. Three classes of approaches to the latter quantity can be distinguished in the literature.}

\added{Rigorous results exist for the homogenization of surface energies of Griffith type. Braides, Defranceschi and Vitali~\cite{braides_homogenization_1996} proved that free-discontinuity functionals with a periodically oscillating surface energy, as they appear in the variational formulation of brittle fracture by Francfort and Marigo~\cite{francfort_revisiting_1998} and Bourdin et al.~\cite{bourdin_variational_2008}, $\Gamma$-converge to a functional with a homogenized and, in general, anisotropic surface energy. Cagnetti, Dal Maso, Scardia and Zeppieri~\cite{cagnetti_stochastic_2019} extended this result to stationary random microstructures. Schneider~\cite{schneider_fftbased_2020} and Ernesti and Schneider~\cite{ernesti_fast_2021,ernesti_computing_2022} use the resulting cell formula as a variational definition of an effective crack energy and compute it with FFT-based solvers, also for anisotropic fracture resistance. Michel and Suquet~\cite{michel_merits_2022} assess the merits and limitations of this definition. They relate it to limit-load problems and show that it is a lower bound of a more physically motivated effective fracture energy, namely the energy dissipated along the actual crack path through a unit cell. The homogenized surface energy is the energy of the minimal crack surface weighted by the local toughness. It does not account for the elastic interaction of the crack with the microstructure, for crack trapping and the unstable crack jumps that follow re-nucleation, or for inelastic bulk dissipation, which are the mechanisms that govern toughening in many materials, see the discussion by Lebihain~\cite{lebihain_brittle_2021}.}

\added{A second class of approaches describes the interaction of a crack front with the microstructure directly. Gao and Rice~\cite{gao_firstorder_1989} and Bower and Ortiz~\cite{bower_threedimensional_1991} analyzed the trapping of a crack front by arrays of tough obstacles with perturbation methods and with numerical methods for three-dimensional cracks, and Lebihain, Leblond and Ponson~\cite{lebihain_effective_2020} derived the effective toughness of periodic heterogeneous materials including out-of-plane excursions of the crack. These approaches are restricted to linear elastic bulk behavior and to small contrasts of the toughness or small perturbations of a planar crack.}

\added{The third class is the numerical experiment of Hossain et al.~\cite{hossain_effective_2014}. A region of the material that resolves the microstructure is subjected to so-called surfing boundary conditions, which enforce complete separation at a prescribed macroscopic crack speed, and the maximum of the energetic driving force recorded at the boundary is taken as the effective crack resistance. Since the crack evolution inside the region is computed with a phase-field model, crack trapping, re-nucleation, branching and unstable crack jumps are included without restriction. The method was developed further by Hsueh et al.~\cite{hsueh_stress_2018} for layered materials and by Brach et al.~\cite{brach_anisotropy_2019} for anisotropic effective crack resistance.}
Recent contributions underline that this remains an active field of research. Liu and Shen~\cite{liu_effective_2025} propose a phase-field formulation for anisotropic effective fracture toughness that accounts for a quasi-worst-case translation of a representative volume element. Aranda and Segurado~\cite{aranda_cracklength_2025} introduce a crack-length control technique for phase-field fracture in FFT homogenization and use it to estimate effective fracture properties of composites and porous materials. Man et al.~\cite{man_collaborative_2025} study how crack guiding and trapping in bioinspired interfaces control effective toughness, while Henry~\cite{henry_phasefield_2025} shows for dynamic crack propagation in disordered heterogeneous materials that the spatial distribution of soft inclusions can strongly affect the apparent fracture energy even when the effective elastic properties are similar.
\added{A recent related development is the variable stiffness boundary condition of Hao et al.~\cite{hao_variable_2026}, which builds on the surfing boundary condition but translates a simple uniform remote displacement into a non-uniform load through an engineered stiffness gradient, thereby facilitating implementation in experiments and commercial finite-element software. The limitations of this third class are of a different nature than those of the first two. The recorded value depends on the loading of the experiment and on the size and the content of the resolved region, and the representativity of this region in the sense of Hill is not guaranteed but has to be assessed for every microstructure. The result is therefore an operational quantity of the experiment and not a homogenized material property. With few exceptions, the contributions cited above are limited to linear elastic material behavior and to two-dimensional problems.}

\added{Phase-field models of fracture, see the review by Ambati et al.~\cite{ambati_review_2015}, have been coupled with plasticity in several ways. Gradient damage models coupled with plasticity and their relation to cohesive cracks were analyzed by Alessi, Marigo and Vidoli~\cite{alessi_gradient_2014}. Variational formulations at finite strains were proposed by Miehe, Aldakheel and co-workers~\cite{miehe_phasefield_ductile_2016,miehe_phasefield_porous_2016} and by Borden et al.~\cite{borden_phasefield_2016}, and small-strain formulations in which the plastic energy is degraded together with the elastic energy were proposed by Ambati, Gerasimov and De Lorenzis~\cite{ambati_ductile_2015} and by Kuhn et al.~\cite{kuhn_phase_2016} and Noll et al.~\cite{noll_3d_2020}. In the latter formulation, a cubic degradation function allows the yield stress and the hardening modulus to be prescribed directly, and it is the formulation used in this work.}

\deleted{The existence of an effective fracture toughness for a given material
and the question of how it should be defined remain subjects
of active debate, as discussed by Lebihain~\cite{lebihain_brittle_2021},
with several approaches proposed in the literature.
In contrast to the effective elastic or plastic properties obtained in classical micromechanics, it is presently not established in general that fracture in a heterogeneous material gives rise to a homogenized macroscopic material model with a well-defined fracture toughness obtained in an RVE limit.
To avoid conflating two slightly different concepts, the term effective crack resistance is used in this work for the value obtained from a numerical experiment under specific less general conditions, i.e., the maximum energetic driving force required to force crack advance through the considered microstructure. In contrast, fracture toughness is reserved for a generalized material quantity in the sense of a macroscopic fracture property. Whether such a fracture toughness exists for heterogeneous materials and under which circumstances both concepts might coincide remains an open question of research.
Schneider~\cite{schneider_fftbased_2020} suggests a variational definition that
relies on rigorous mathematical results for the homogenization of
the energy functional occurring in Bourdin's popular variational formulation of brittle fracture~\cite{bourdin_variational_2008}.
This idea is further pursued by
Ernesti and Schneider~\cite{ernesti_fast_2021,ernesti_computing_2022} for anisotropic fracture resistance.}

\deleted{Michel and Suquet~\cite{michel_merits_2022}
assess essentially the same variational definition and discuss its merits and limitations. In particular, they relate the variational surface energy to limit-load problems and show that it provides a lower bound to a more physically motivated effective fracture energy based on the energy dissipated along the actual crack path through a unit cell.}

\deleted{In this work, we adopt the methodology introduced
by Hossain et al.~\cite{hossain_effective_2014} and further
developed and applied by Hsueh et al.~\cite{hsueh_stress_2018} for layered materials,
as well as by Brach et al.~\cite{brach_anisotropy_2019} for anisotropic effective crack resistance,
to determine the effective crack resistance of heterogeneous materials.
All of these contributions were limited to linear-elastic material behavior and two-dimensional problems.}

\deleted{The methodology is not used here as a proof of a homogenized fracture toughness in the mathematical sense. Rather, it is interpreted as a numerical counterpart of a fracture experiment.
In this experiment-like setting,
a representative material region containing
 microstructural features is subjected
 to so-called surfing boundary conditions that enforce complete separation, and the energetic driving force required for this process is recorded.
  A recent related development is the variable stiffness boundary condition of Hao et al.~\cite{hao_variable_2026}, which builds on the surfing boundary condition but translates a simple uniform remote displacement into a non-uniform load through an engineered stiffness gradient, thereby facilitating implementation in experiments and commercial finite-element software.
The resulting microstructural crack evolution is
then simulated using a phase-field model.
See, for example, the review by Ambati et al.~\cite{ambati_review_2015} for a comprehensive overview of phase-field models for fracture.
For ductile materials, phase-field fracture has also been coupled to plasticity in variational finite-strain formulations, for example by Miehe, Aldakheel and co-workers~\cite{miehe_phasefield_ductile_2016,miehe_phasefield_porous_2016}.}
"""))

# -------------------------------------------------------------------- own previous work: long sentence split (minor 2)
E.append((r"""In the present work, we aim to extend this analysis
toward additively manufactured metals, which typically exhibit pronounced ductile behavior.
In contrast to our previous work, which only considered reversible elastic material behavior,
albeit non-linear and consistent with von Mises plasticity under monotonic loading,
but which does not accurately capture unloading behavior,
we here also study the effect of true energy dissipation due to plasticity in ductile materials.
Since local unloading inevitably occurs during crack propagation,
even under monotonically increasing external loads,
this more closely matches true ductile fracture behavior.

In addition, we extend the approach of Hossain et al.~\cite{hossain_effective_2014} to take into account the three-dimensional
character of real porous materials with spherical pores.

This paper is structured as follows.
Section~\ref{sec:numericalExperiment} introduces the numerical experiment
proposed by Hossain et al.~\cite{hossain_effective_2014}.
Section~\ref{sec:phase_fiel_model_ductile} presents the phase-field
fracture formulation for both linear-elastic and von Mises-type plasticity material models.
Section~\ref{sec:2Dresults} focuses on the computation
of the effective crack resistance for a simplified porous material
under two-dimensional plane strain assumptions
 using both constitutive descriptions.
Lastly, we extend the model to three dimensions and compare our findings for the von Mises-type plasticity model
for both the two-dimensional and three-dimensional cases.
""",
r"""In the present work, we aim to extend this analysis
toward additively manufactured metals, which typically exhibit pronounced ductile behavior.
\replaced{Our previous work considered only reversible elastic material behavior. The Ramberg--Osgood law is non-linear and coincides with von Mises plasticity under monotonic loading, but it does not capture unloading. Here, the effect of true energy dissipation due to plasticity is studied with a von Mises plasticity model.}{In contrast to our previous work, which only considered reversible elastic material behavior,
albeit non-linear and consistent with von Mises plasticity under monotonic loading,
but which does not accurately capture unloading behavior,
we here also study the effect of true energy dissipation due to plasticity in ductile materials.}
Since local unloading inevitably occurs during crack propagation,
even under monotonically increasing external loads,
this more closely matches true ductile fracture behavior.

\replaced{The open questions addressed in this work are the following. First, how does plastic dissipation change the effective crack resistance obtained in the numerical experiment, and by which mechanisms? Compared with a physical experiment, the simulation gives access to the complete energy balance of the specimen and to the stress state at the location of crack re-nucleation. Both are used here to separate the contributions of crack nucleation, plastic dissipation and crack growth, and the sensitivity of the result with respect to the yield stress and the hardening modulus is examined. Second, does the two-dimensional idealization of cylindrical pores carry over to spherical pores in three dimensions? Third, by how much does the result depend on the initial crack and on the size of the resolved region? The last point does not replace a representativity study, but it quantifies the dependence for the microstructure considered.}{In addition, we extend the approach of Hossain et al.~\cite{hossain_effective_2014} to take into account the three-dimensional
character of real porous materials with spherical pores.}

This paper is structured as follows.
Section~\ref{sec:numericalExperiment} introduces the numerical experiment
proposed by Hossain et al.~\cite{hossain_effective_2014}.
Section~\ref{sec:phase_fiel_model_ductile} presents the phase-field
fracture formulation for both linear-elastic and von Mises-type plasticity material models.
Section~\ref{sec:2Dresults} focuses on the computation
of the effective crack resistance for a simplified porous material
under two-dimensional plane strain assumptions
 using both constitutive descriptions\added{, followed by the energy partition of the experiment, the stress state before re-nucleation, the sensitivity with respect to the plasticity parameters, and the influence of the isotropic approximation of the embedding and of the size of the resolved region}.
Lastly, we extend the model to three dimensions and compare our findings for the von Mises-type plasticity model
for both the two-dimensional and three-dimensional cases.
"""))

# -------------------------------------------------------------------- bibliography: ten new entries
NEWBIB = r"""
%%% new in revision 2
\bibitem[\protect\citeauthoryear{Hill}{1963}]{hill_elastic_1963}
\begin{barticle}
\bauthor{\bsnm{Hill}, \binits{R.}}:
\batitle{Elastic properties of reinforced solids: Some theoretical principles}.
\bjtitle{Journal of the Mechanics and Physics of Solids}
\bvolume{11}(\bissue{5}),
\bfpage{357}--\blpage{372}
(\byear{1963})
\doiurl{10.1016/0022-5096(63)90036-X}
\end{barticle}
\endbibitem

\bibitem[\protect\citeauthoryear{Braides et~al.}{1996}]{braides_homogenization_1996}
\begin{barticle}
\bauthor{\bsnm{Braides}, \binits{A.}},
\bauthor{\bsnm{Defranceschi}, \binits{A.}},
\bauthor{\bsnm{Vitali}, \binits{E.}}:
\batitle{Homogenization of free discontinuity problems}.
\bjtitle{Archive for Rational Mechanics and Analysis}
\bvolume{135}(\bissue{4}),
\bfpage{297}--\blpage{356}
(\byear{1996})
\doiurl{10.1007/BF02198476}
\end{barticle}
\endbibitem

\bibitem[\protect\citeauthoryear{Cagnetti et~al.}{2019}]{cagnetti_stochastic_2019}
\begin{barticle}
\bauthor{\bsnm{Cagnetti}, \binits{F.}},
\bauthor{\bsnm{Dal Maso}, \binits{G.}},
\bauthor{\bsnm{Scardia}, \binits{L.}},
\bauthor{\bsnm{Zeppieri}, \binits{C.I.}}:
\batitle{Stochastic homogenisation of free-discontinuity problems}.
\bjtitle{Archive for Rational Mechanics and Analysis}
\bvolume{233}(\bissue{2}),
\bfpage{935}--\blpage{974}
(\byear{2019})
\doiurl{10.1007/s00205-019-01372-x}
\end{barticle}
\endbibitem

\bibitem[\protect\citeauthoryear{Francfort and Marigo}{1998}]{francfort_revisiting_1998}
\begin{barticle}
\bauthor{\bsnm{Francfort}, \binits{G.A.}},
\bauthor{\bsnm{Marigo}, \binits{J.-J.}}:
\batitle{Revisiting brittle fracture as an energy minimization problem}.
\bjtitle{Journal of the Mechanics and Physics of Solids}
\bvolume{46}(\bissue{8}),
\bfpage{1319}--\blpage{1342}
(\byear{1998})
\doiurl{10.1016/S0022-5096(98)00034-9}
\end{barticle}
\endbibitem

\bibitem[\protect\citeauthoryear{Gao and Rice}{1989}]{gao_firstorder_1989}
\begin{barticle}
\bauthor{\bsnm{Gao}, \binits{H.}},
\bauthor{\bsnm{Rice}, \binits{J.R.}}:
\batitle{A first-order perturbation analysis of crack trapping by arrays of obstacles}.
\bjtitle{Journal of Applied Mechanics}
\bvolume{56}(\bissue{4}),
\bfpage{828}--\blpage{836}
(\byear{1989})
\doiurl{10.1115/1.3176178}
\end{barticle}
\endbibitem

\bibitem[\protect\citeauthoryear{Bower and Ortiz}{1991}]{bower_threedimensional_1991}
\begin{barticle}
\bauthor{\bsnm{Bower}, \binits{A.F.}},
\bauthor{\bsnm{Ortiz}, \binits{M.}}:
\batitle{A three-dimensional analysis of crack trapping and bridging by tough particles}.
\bjtitle{Journal of the Mechanics and Physics of Solids}
\bvolume{39}(\bissue{6}),
\bfpage{815}--\blpage{858}
(\byear{1991})
\doiurl{10.1016/0022-5096(91)90026-K}
\end{barticle}
\endbibitem

\bibitem[\protect\citeauthoryear{Lebihain et~al.}{2020}]{lebihain_effective_2020}
\begin{barticle}
\bauthor{\bsnm{Lebihain}, \binits{M.}},
\bauthor{\bsnm{Leblond}, \binits{J.-B.}},
\bauthor{\bsnm{Ponson}, \binits{L.}}:
\batitle{Effective toughness of periodic heterogeneous materials: the effect of out-of-plane excursions of cracks}.
\bjtitle{Journal of the Mechanics and Physics of Solids}
\bvolume{137},
\bfpage{103876}
(\byear{2020})
\doiurl{10.1016/j.jmps.2020.103876}
\end{barticle}
\endbibitem

\bibitem[\protect\citeauthoryear{Alessi et~al.}{2014}]{alessi_gradient_2014}
\begin{barticle}
\bauthor{\bsnm{Alessi}, \binits{R.}},
\bauthor{\bsnm{Marigo}, \binits{J.-J.}},
\bauthor{\bsnm{Vidoli}, \binits{S.}}:
\batitle{Gradient damage models coupled with plasticity and nucleation of cohesive cracks}.
\bjtitle{Archive for Rational Mechanics and Analysis}
\bvolume{214}(\bissue{2}),
\bfpage{575}--\blpage{615}
(\byear{2014})
\doiurl{10.1007/s00205-014-0763-8}
\end{barticle}
\endbibitem

\bibitem[\protect\citeauthoryear{Borden et~al.}{2016}]{borden_phasefield_2016}
\begin{barticle}
\bauthor{\bsnm{Borden}, \binits{M.J.}},
\bauthor{\bsnm{Hughes}, \binits{T.J.R.}},
\bauthor{\bsnm{Landis}, \binits{C.M.}},
\bauthor{\bsnm{Anvari}, \binits{A.}},
\bauthor{\bsnm{Lee}, \binits{I.J.}}:
\batitle{A phase-field formulation for fracture in ductile materials: Finite deformation balance law derivation, plastic degradation, and stress triaxiality effects}.
\bjtitle{Computer Methods in Applied Mechanics and Engineering}
\bvolume{312},
\bfpage{130}--\blpage{166}
(\byear{2016})
\doiurl{10.1016/j.cma.2016.09.005}
\end{barticle}
\endbibitem

\bibitem[\protect\citeauthoryear{Ambati et~al.}{2015}]{ambati_ductile_2015}
\begin{barticle}
\bauthor{\bsnm{Ambati}, \binits{M.}},
\bauthor{\bsnm{Gerasimov}, \binits{T.}},
\bauthor{\bsnm{De Lorenzis}, \binits{L.}}:
\batitle{Phase-field modeling of ductile fracture}.
\bjtitle{Computational Mechanics}
\bvolume{55}(\bissue{5}),
\bfpage{1017}--\blpage{1040}
(\byear{2015})
\doiurl{10.1007/s00466-015-1151-4}
\end{barticle}
\endbibitem

\end{thebibliography}"""

E.append((r"""\end{thebibliography}""", NEWBIB))
E.append((r"""\begin{thebibliography}{28}""", r"""\begin{thebibliography}{38}"""))


def find_once(text, old):
    if text.count(old) == 1:
        return old
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
