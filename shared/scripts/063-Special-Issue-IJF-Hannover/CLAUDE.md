# CLAUDE.md — Paper A03: "Brittle fracture in topology-optimized structures"

Working notes for AI-assisted work on this manuscript and its research data.
Keep this file updated when decisions are made or workflows change.
Last updated: 2026-09-01 (evening: H-S issue + deadline extension added).

## 1. What this project is

- Manuscript: **"Brittle fracture in topology-optimized structures"** (working title; reviewers
  asked for a narrower one). Internal code: **A03** (TRR 375 project label; the earlier related
  manuscript is A02, hence the folder `Images_A02`).
- Authors: Alexander Schlüter (TU Darmstadt, corresponding), Ján Pravda, Dustin Roman Jantos,
  Philipp Junker (all LUH Hannover), Ralf Müller (TU Darmstadt).
- Journal: **Archive of Applied Mechanics** (Springer Nature, `sn-jnl` class,
  `sn-mathphys-num` bibliography style). Editor: Jörg Schröder.
  Submission ID `cc51f1c5-c888-45a4-8fae-ebefd5a3c8f7`. Support: manishasree.v@springernature.com
- Funding: DFG Project-ID 511263698 – TRR 375. Compute: Lichtenberg HPC, TU Darmstadt.
- The folder name says "IJF Special Issue Hannover" but the paper was submitted to
  *Archive of Applied Mechanics* — do not "correct" the journal name in the manuscript.

### Status (as of 2026-09-01)
- Submitted 2026-06-10/11 (snapshots in `submission_review_2026-06-1x/`).
- **Decision 2026-08-18: major revision.** Two reviewers + editor.
  **Recommended resubmission deadline: 15 Sep 2026.**
- Revision has *not* been started in the manuscript yet: `main_revised_template.tex` is a copy of
  `main.tex` plus the `changes` package and a "Revision Working Note" with one sample edit.
  `response_to_reviewers_template.tex` contains the verbatim reviewer text with empty
  response blocks. A *suggested* response draft exists in
  `review_feedback/response_to_reviewers_suggested_2026-08-18.tex` (NOT yet author-approved).
- Only uncommitted change in `main.tex` vs. git: an added `\bmhead{Funding}` block.
- **Deadline extended to 15 Oct 2026** (Alex, 2026-09-01). Editor-in-chief is well known to Junker;
  further extension considered plausible. Withdraw-and-resubmit is an option Pravda has used before.
- **Mobility side study done 2026-09-29** (R1 "artificial damping"; not in manuscript). **Read
  `research_data_.../MOBILITY_STUDY_FINDINGS.md` first** (conclusions, writing guidance); method, commands
  and full table in `MOBILITY_STUDY_README.md`, plots in `plots/mobility_study/` (single report:
  `plots/mobility_study/mobility_study_report.pdf`). β_s=0.03,
  βφ=0.01, spectral, M ∈ {100 (ref), 1000, 10000}; run locally in Docker, raw results in
  `/Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/mobility_study_260929/`. Script
  `000_template/01_phasefield_dcb_260504_folder.py` got `--mobility` (default 100 → unchanged names).
  **Outcome: M = 100 is NOT quasi-static for the peak load.** max R_y M=100→10000: ρ=0.3 E_var
  45.26→31.35 (−31 %), ρ=0.3 E_min 29.71→18.58 (−37 %), ρ=0.6 E_var 89.43→81.34 (−9 %, not yet
  converged in M). D_s/W at peak → ≤0.2 % at M=10000 (≈∝1/M above 1000); post-peak D_s stays ~Π_frac
  (unstable crack growth). E_max (G_c=1) is barely affected: ρ=0.3 47.29→44.20, ρ=0.6 85.57→79.58
  (−7 %). **Story at M=10000:** ρ=0.3 E_var vs E_min +52 %→+69 % (holds); ρ=0.6 E_var vs E_max
  +4.5 %→+2.2 % (holds, ρ=0.6 E_var not yet converged in M); **ρ=0.3 E_var vs E_max −4 %→−29 %
  (W at peak −11 %→−54 %) — changes.** Also found: post-peak energy balance open for E_min
  (Π_tot/W≈0.5), and the two-strip W (col. 4, `max_Work` in summary CSV) is 30–60 % below Π_el.
  **Follow-up 2026-09-30 (FINDINGS §8):** reported R_y (traction integral on strips, column 2 = left
  strip only) is only 62–76 % of the energy-consistent reaction 2Π_el/u, structure-dependent → fix via
  residual-based nodal reactions in any rerun; corrected, ρ=0.6 E_var vs E_max is ≈ +20 %, not +2 %.
  **Fix done:** `pp.reaction_force_from_residual` (both alex copies), test `13_test_reaction_force.py`,
  simulation script writes columns 14–17 (residual R_y left/right, dW, W); evaluation scripts not yet switched.
  **Corrected story without rerun (FINDINGS §9, `14_energy_consistent_metrics.py`, `plots/energy_consistent/`):**
  W_exact = Π_el+Π_frac+D_s, R = dW/du. M=100, βφ=0.01 over β_s: ρ=0.3 E_var vs E_min +50…+78 % (was +36…+64),
  ρ=0.3 E_var vs E_max −5…0 %, ρ=0.6 E_var vs E_max +26…+27 % (was +4…+6). Column 13 (whole-boundary W) is
  also unreliable (0.92–1.10 at peak). **Decisions Alex 2026-09-30: report R_y as total of both strips; keep W
  defined as surface integral in the text.** Peak-metrics panel (b) uses column 4, not Eq. total_boundary_work.
  **M=10⁵ convergence runs (FINDINGS §10, `mobility_convergence_260930/` on the external disk):** step 10⁴→10⁵ in
  first-peak load: ρ=0.6 E_max −0.1 %, ρ=0.6 E_var −1.1 %, ρ=0.3 E_min at β_s=0.06 −3.8 % (M=100 there is 1.9× too
  high; the flat E_min-vs-β_s trend in the manuscript is a rate artefact). ρ=0.6 E_var fails in two stages at
  high M: first peak 259 N/mm (total), 8.6 % drop, then re-rise to ≥279 N/mm until Newton failure → peak
  definition (first peak vs ultimate) to be decided. Recommended campaign mobility: M=10⁵.
  **Decisions Alex 2026-09-30 (evening): campaign mobility M=10⁵; peak = FIRST peak of the total reaction (first
  load drop > 1 %); rerun all 36 a=6 cases.** Campaign `260930_M1e5` started 2026-09-30 17:11 locally in Docker:
  `05_run_260930_mobility_1e5_campaign_local.sh` (restartable, 2 runs at a time, wall limit 3 h/run), results in
  `/Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/campaign_260930_M1e5/` (`campaign.log`, one `DONE`
  file per finished run). Evaluate with `python3 14_energy_consistent_metrics.py --extra-root <campaign root>`.
  **CAMPAIGN 260930_M1e5 COMPLETE 2026-10-02 (FINDINGS §12): 36/36 runs, M=10⁵, residual-based forces, first-peak
  metric, all to u_y=0.03 mm.** Histories in `results/campaign_260930_M1e5/`, evaluation in `plots/energy_consistent/`
  (`14_energy_consistent_metrics.py --extra-root results/campaign_260930_M1e5`); PDF report
  `plots/energy_consistent/energy_consistent_report.pdf`. Story at M=10⁵ (first peak, total of
  both strips, βφ=0.01, over β_s): ρ=0.3 E_var vs E_min +82…+92 %, ρ=0.3 E_var vs E_max −26…−32 %, ρ=0.6 E_var vs
  E_max +19…+21 %; all peaks fall ≈27 % from β_s=0.015 to 0.06; βφ: insensitive at ρ=0.3, at ρ=0.6 βφ=0.05 is
  7–14 % weaker than βφ=0.001. Post-peak all structures reload (E_min ρ=0.3 and E_var ρ=0.6 above first peak).
  **Phase 2 progress 2026-10-02 (10:25), read `research_data_.../HANDOFF_PROMPT_261002.md` first:** fracture part of the
  revision written with markup (Sec. 2.2 mobility/solver, Sec. 3.2 incl. new 3.2.1 secondary measures + 3.2.2 mobility,
  Sec. 3.3, conclusion, abstract results part) and the touched response items; all numbers independently verified.
  New scripts 15_ (figures/tables), 16_ (field snapshots from .h5 on the Mac, numpy+h5py), 17_ (phase field after first
  crack event, crack evolution, sigma_1, mesh stats); figures *_M1e5*.png in Images_A02. Open: phi/cost wording
  (Hannover), title, remaining R1/R2 TODO blocks, TO items, Overleaf push (Alex).
  **Phase-1 proposal done 2026-10-02:** `REVISION_PROPOSAL_phase1_261002.md` (sections A–E, decisions in F:
  story agreed, φ = material grade/cost, mobility paragraph at end of Sec. 3.2, secondary measures as table,
  peak-metrics figure keeps 4 panels, s-fields at the first peak, Hughes 2000 for nodal reactions). Overleaf
  pulled 2026-10-02: still `d00482b`.
  **Decision Alex 2026-10-02:** first peak (+ work up to it) = primary strength measure; max load and total work to
  u_y=0.03 mm as secondary measures (FINDINGS §13). Next step: manuscript revision proposal, prompt in
  `REVISION_PROMPT_claude_code_261002.md` (phase 1 = proposal only, no edits until green light).
  **Solver fix 2026-09-30 (FINDINGS §11):** the dt→1e-14 Newton failures were a tolerance artefact (atol 1e-10 below
  the residual round-off floor 1.15e-10; relative criterion scaled by the first update). Campaign restarted 22:05
  with NEWTON_ATOL=1e-8, NEWTON_MAX_IT=20, PHASEFIELD_DT_MIN=1e-10 (restart 2, 2026-10-01 00:01), PHASEFIELD_T_END=0.03
  (runs end at u_y=0.03 mm or at a solver failure, e.g. E_min at u_y≈0.026). First peak unchanged; ρ=0.6 E_var shows repeated partial crack events with reloading up to
  u_y=0.029 mm → first peak = strength measure. `alex/solution.py` (both copies) got NEWTON_LOG, NEWTON_TYPE=snes,
  NEWTON_FALLBACK=snes|damped (tested, not needed).
  M=10000 is ≈2–4 % above the M→∞ limit for ρ=0.3 (M^(−2/3) law), unproven for ρ=0.6 and other β_s.
  Any response to R1 on mobility must not claim M=100 is negligible for peak loads; decide whether
  to rerun the campaigns with M=10000 (≡ 100× slower loading).

### ⚠ Fundamental problem discovered 2026-08-19 (Hashin–Shtrikman bounds) — decision pending
- Hannover's TO paper (`pravda_macroscopic_2026`) received a review in July 2026 stating that
  **E₀/E₁ = 0.6 at φ_min = 0.5 violates the Hashin–Shtrikman bounds** for an isotropic porous
  material (a 50 % porous material cannot have 60 % of the solid stiffness). H-S-conform requires
  at least E₀/E₁ < φ_min; Hannover now uses **E₀/E₁ = φ_min² = 0.25** (Gibson–Ashby).
- **All TO structures in this manuscript were computed with the non-physical ratio 0.6.**
- Consequence (Jantos, 2026-09-01, "auch theoretisch nachweislich"): with H-S-conform material and a
  **single load case**, pure stiffness maximization yields only E_max / void — no porous material,
  regardless of total mass or BVP, given fine enough mesh. Graded porosity only appears with
  **multi-load-case** optimization (grey density → porosity with physical meaning). That is how the
  Hannover TO paper was resubmitted.
- Options on the table (email thread 19.08–01.09.2026):
  1. Keep results, reframe as a *purely numerical model study with artificial material interpolation*
     (least effort; hard to defend, several reviewer comments target exactly the physical motivation of
     the porosity/G_c mapping). Alex's reading: the study then investigates how phase-field fracture
     handles spatially varying E and G_c, without claiming a physically realizable porous material.
  2. Replace TO results by H-S/Gibson–Ashby-conform multi-load-case optimizations and recompute the
     fracture simulations (cleanest; large effort, convergence risk, Alex currently lacks capacity;
     manuscript changes substantially → consider withdraw + resubmit). Fracture load cases: apply each
     load case separately to the multi-load-optimized structure; weighted resistance measure (e.g.
     max R_y over all load cases) to be defined. Müller/Pravda/Schlüter lean towards this.
  3. Junker's variant: keep old results as Section XX.1 ("theoretical" single-load-case optimization)
     and add XX.2 with realistic multi-load-case structures. Avoids discarding positively reviewed
     results but old results must be decoupled from the "real porous material" derivation; scope grows
     massively; reviewers may still be confused.
  4. Side track: submit a shortened version (option-1 style) to WCCM-ECCOMAS 2026 proceedings
     (https://wccm-eccomas2026.org/conference_proceedings) as a HyPo (TRR 375) publication —
     unclear whether IKM does non-peer-reviewed contributions; content to be decided.
- Ján Pravda is out (accident, as of 2026-09-01). Viko planned: Wed 2.9. 11:00–12:00 (fits Müller
  "Mi ab 11", Jantos "Mi bis 12") or Fri 4.9. 12:00–15:30; Zoom link in Alex's email.
- **Until this is decided, do not start rewriting the results section or recomputing figures.**
  Text-only reviewer items that survive every option (phase-field limitations, notation, minor R2
  fixes, bib cleanup, title) can proceed in `main_revised_template.tex`.
- If option 1 is chosen: the "porous material" wording throughout (abstract, intro, Sec. 2.1, G_c
  mapping motivation) must be rewritten as "spatially graded stiffness/toughness fields"; E₀/E₁=0.6
  must be declared an artificial interpolation; the G_c(φ_r) relation becomes a model assumption.
- If option 2/3: new TO input from Hannover (Julia/Ferrite) → `resources/` → `04_mesh2dlfxmesh.py`
  → cluster campaigns (see §6). Budget several weeks; ask Hannover for E₀/E₁, φ_min, load cases first.

## 2. Folder layout

Two connected roots; the Overleaf folder appears twice (once nested, once as its own root):

```
063-Special-Issue-IJF-Hannover/
└── research_data_brittle_fracture_topology_optimized_structures/   # research compendium (Zenodo-ready)
    ├── README.md, DATA_DICTIONARY.md, SOFTWARE_ENVIRONMENT.md,
    │   ZENODO_UPLOAD_CHECKLIST.md, CITATION.cff, MANIFEST.csv, SHA256SUMS
    ├── results/new_W_whole_boundary/      # PRIMARY publication data (8 campaigns, 88 cases)
    ├── plots/new_W_whole_boundary/        # publication plots: beta001/, beta005/, beta_comparison/
    ├── resources/                         # meshes + TO fields (input to fracture sims)
    ├── code/vendor/alex/                  # vendored FEniCSx helper library (phasefield.py etc.)
    ├── 000_template/, 00_jobs/, 0N_*.sh   # HPC campaign prep/submit/eval scripts
    ├── 01_phasefield_dcb_260504_folder.py # main monolithic phase-field simulation (DOLFINx)
    ├── 04_mesh2dlfxmesh.py                # mesh conversion
    ├── 08_plot_phasefield_overview.py     # field overview PDFs (E, gc, sigma_c, sig_vol, sig_dev, s)
    ├── 09_evaluation_260504_parameter_space.py  # response/energy/peak-metric plots + summary CSV
    ├── 10_plot_rho_omega_constraint.py    # mass-constraint check on Omega_f
    ├── archive_tools/reproduce_manuscript_plots.sh, build_manifest.py
    ├── review_feedback/                   # decision letter (review_18.08.2026) + suggested response
    ├── submission_review_2026-06-10/, -11/  # frozen submission snapshots (do not edit)
    └── 68c3b8d0b7dca7b64b8b7a93/          # Overleaf project (git, see below)
        ├── main.tex                       # SUBMITTED version (keep as reference)
        ├── main_revised_template.tex      # REVISION working copy (edit here)
        ├── response_to_reviewers_template.tex  # point-by-point response (edit here)
        ├── bib/a02.bib, bib/a03.bib       # both are loaded by \bibliography
        ├── Images_A02/                    # fracture/evaluation figures (+ summary_260504.csv)
        ├── Images_A03/                    # TO result images (chi/phi/E, *_no_legend.png used)
        ├── manuscript_picture_list.txt    # source→figure mapping (plots/… → Images_A02/…)
        ├── failure_behavior_main_findings.txt  # bullet summary of results section
        ├── main_DRJ_comment_actions.txt   # log of DRJ's PDF-comment pass (2026-06-05)
        └── main_DRJ.pdf                   # co-author-annotated PDF (older than main.tex)
```

### Overleaf / git
- Remote: `https://git@sharelatex.tu-darmstadt.de/git/68c3b8d0b7dca7b64b8b7a93` (TU Darmstadt
  ShareLaTeX). Commits named "Update on Overleaf." are co-author edits made online.
- **Pull before editing; co-authors edit on Overleaf concurrently.** Do not force-push.
- Build artifacts (`.aux`, `.log`, `.fdb_latexmk`, `.pdf`) are tracked/untracked inconsistently;
  don't spend effort cleaning that unless asked.

## 3. Manuscript conventions (must-follow when editing .tex)

- Springer rule: **single `.tex` file, no `\input{}`** for text (the one `\input` of
  `Images_A02/cdb_setup.pdf_t` is an xfig figure and is fine).
- Build: `latexmk -pdf main.tex` (or `main_revised_template.tex`) in the Overleaf folder.
  `pdflatex`/`latexmk` exist in the local device shell; Overleaf builds it too.
- Macros: `\Evar`, `\Emin`, `\Emax` (bold E with subscript; note `main.tex` versions append `{\,}`,
  template versions use `\xspace`), `\Fig`, `\Section`, `\Table`, `\changed{}` (blue),
  `\mycomment{who}{text}` (red inline note), `\resultfigurediscussion` (empty placeholder).
- Revision markup (in `main_revised_template.tex` / response): `changes` package with
  `\added{}`, `\deleted{}`, `\replaced{new}{old}`; option `markup=underlined,authormarkup=none`.
  Every substantive edit must be visible so the response letter can point to it.
  Remove the "Revision Working Note" section before final submission.
- Notation decisions already settled (keep consistent):
  - TO domain `\Omega`, boundary `\partial\Omega`, outward normal **`\boldsymbol{N}`**
    (DRJ requested bold capital N to distinguish from fracture normal) — but note the *current*
    `main.tex` evolution equations still use `\boldsymbol{n}` in eqs. after (eq:evol_chi/phi)
    and Fig. `fig:omega_omega_f` also uses `n`; check and unify in the revision.
  - Fracture domain `\Omega_f = {x : \chi ≥ 0.5}`, `\partial\Omega_f^u`, `\partial\Omega_f^t`,
    normal `\boldsymbol{n}_f`, phase-field length `\beta_s`, residual `\kappa_s`.
  - TO: `\chi` density, `\phi` porosity design var (0..1; φ=1 solid), `\phi_p` true material
    density, `\phi_r = 1-\phi_p` true porosity, `\beta_\chi`, `\beta_\phi`, `\eta_\chi`, `\eta_\phi`,
    KKT `\gamma_\chi`, `\gamma_\phi`, Lagrange multiplier `\Lambda`, `\kappa=1e-9`, SIMP `p=3`, `q=2`.
  - "homogeneous" (never "homogenous"); "penalization exponent" (not "factor", per Reviewer 2);
    "first-order Lagrange elements on structured quadrilateral mesh" (TO) vs.
    "first-order Lagrange elements on triangular meshes" (fracture).
  - `n`, `m` are the TO element counts → the spectral-decomposition sum `\sum_{i=1}^{n}` must be
    renamed (use `d` or `2`) — Reviewer 2 item.
- Figures: use the `*_no_legend.png` TO images with `overpic` labels; fracture overview figures are
  `\textwidth` PNGs from `Images_A02/`. Don't regenerate figures unless asked; if regenerated,
  update `manuscript_picture_list.txt`.
- Bibliography: entries with "accessed" dates but no journal must be fixed (Reviewer 2).

## 4. Key facts and numbers (single source of truth for text consistency)

Benchmark: rectangular design domain a=6 mm × b=1 mm, fixed left/right edges, two point loads
F = −1 N e_y at (2,1) and (4,1) [TO]; fracture: displacement-controlled u_y* = −t·v₀, v₀=1 mm/s on
two strips of width w=0.075 mm centered at x=2 and x=4 (∂Ω_L: 1.9625≤x≤2.0375; ∂Ω_R: 3.9625≤x≤4.0375).

Topology optimization (Julia/Ferrite.jl, thermodynamic TO, Jantos/Junker/Pravda):
- mesh 600×100 quads (h = 0.01 mm, uniform), βχ=0.0009 mm², βφ=0.01 mm² (ref), ηχ=14 s, ηφ=150 s,
  φ₀=0.5, p=3, q=2, **E₀/E₁=0.6 (violates H-S bounds — see Status)**, φ_min=0.5, ρ∈{0.3,0.6}. Convergence: ΔΨ/Ψ ≤1e-4, L∞(Δχ),L∞(Δφ) ≤1e-3.
- E₁=210000 N/mm², E₀=126000 N/mm², ν=0.3 → μ₁=80769.23, λ₁=121153.85 N/mm² (Reviewer 2: round these).
- Sets: **E_var** (φ free), **E_min** (φ=0, max porosity), **E_max** (φ=1, solid); equal mass ρΩ.
  **E_min with ρ=0.6 is infeasible** (constraint cannot be met) → 5 fracture domains total.
- Compliance (Table tab:TO_Psi_comp, βφ=0.01): ρ=0.3: E_min 4.0665e-7, E_max 4.7695e-7, E_var 3.9618e-7;
  ρ=0.6: E_max 2.4374e-7, E_var 2.2436e-7. Stiffness gain of E_var ≈ 2.5 % (ρ=0.3), ≈ 8 % (ρ=0.6).
- βφ study: βφ ∈ {0.001, 0.01, 0.05} mm² (E_var only; E_min/E_max independent of βφ).

Fracture (FEniCSx/DOLFINx, monolithic Newton, spectral split of Miehe 2010, AT2-type, Kuhn–Müller model):
- G_c⁰ = 1 Nmm/mm²; G_c(φ_r)/G_c⁰ = clip(A − B·exp(−C[√(π/(4φ_r)) − 1]), 0.1, 1.0) with
  A=1.26, B=0.78, C=1.35 (from Schlüter & Müller 2025, circular pores). φ_r = 0.5·(1−φ).
- β_s (= "epsilon" in code/filenames) ∈ {0.015, 0.03, 0.045, 0.06} mm; κ_s = 0.001 (code: `eta`);
  mobility M = 100 mm³/(Nmm s); Δt_max = 0.001 s, halving on Newton failure, stop at Δt<1e-14 s.
- σ_c = (9/16)·sqrt(μ G_c/(3β_s)) (1-D nucleation stress, Kuhn 2013); volume-averaged variant eq:sigma_c_effective.
- **Irreversibility threshold (answer for Reviewer 1):** code `alex/phasefield.py::irreversibility_bc`
  fixes s=0 at all vertex dofs where `np.isclose(s, 0, atol=1e-3)` in the previous converged step,
  i.e. s ≤ 1e-3 → Dirichlet s=0. Not stated in the manuscript yet.
- Stress plots shown at t = 0.003 s (before fracture). Phase-field plots on deformed domain, scale ×10.
- Headline results: max R_y of E_var is 36–64 % above E_min for ρ=0.3 over β_s range; E_var ≈ E_max in
  R_y and exceeds E_max in work-at-peak for ρ=0.6; peak pairs (u_y, R_y) for βφ=0.001/0.01/0.05:
  ρ=0.3: (0.013500, 46.733), (0.014375, 45.256), (0.014375, 50.640) mm / N/mm;
  ρ=0.6: (0.016688, 95.744), (0.017313, 89.431), (0.017000, 93.176).
  All peak metrics per case are in `Images_A02/summary_260504.csv` (48 rows) — verify numbers there.
- Energy bookkeeping: W (total-boundary external work, trapezoidal), Π_el, Π_frac, D_s (mobility
  dissipation), Π_tot = Π_el + Π_frac + D_s ≈ W.

## 5. Revision to-do (from the 2026-08-18 decision letter)

Full text: `review_feedback/review_18.08.2026`; draft answers: `review_feedback/response_to_reviewers_suggested_2026-08-18.tex`.

### Needs new computation / figures (start these first — they gate the deadline)
1. **Principal tensile stress + direction plots** (replace or supplement σ_vol/σ_dev figures; R1 twice).
   → extend `08_plot_phasefield_overview.py` with a `sig_1`/principal-direction field.
2. **Quantify mobility dissipation**: D_s/(Π_frac+D_s) and D_s/W at peak and final state (R1).
   → from `result_graphs_*.txt` histories; small table or sentence.
3. **Mesh statistics table** for each Ω_f: #triangles, min/mean/max edge length, h/β_s (R1).
   → `mesh_properties.py` exists (reads `dlfx_mesh.xdmf`); loop over the 5 domains. Also state TO mesh h=0.01 mm.
4. **Crack-evolution snapshots** (first onset + propagation) for representative E_var cases (R2).
   → from `results_*.xdmf` time series.
5. **Redraw Fig. 4 (TO algorithm schematic)** to show the actual staggered loop: equilibrium →
   sensitivities → χ/φ evolution → Λ/KKT update → convergence checks (R1). TikZ in main.tex.
6. Optional: area fraction with intermediate density 0.1<χ<0.9 (R1, "intermediate-density regions").

### Text-only changes
- **Title**: narrow it. Proposed: "Post-optimization brittle-fracture assessment of topology-optimized
  graded porous structures" (or shorter "Brittle-fracture assessment of …"). Author decision pending.
- Distinguish more explicitly from the two DOIs Reviewer 1 named: 10.1016/j.cma.2018.10.010 =
  **Cheng, Bai, To 2019** (`cheng_functionally_2019`, FG-lattice TO with stress constraints) and
  10.1016/j.compstruc.2020.106205 = **Jansen & Pierard 2020** (`jansen_hybrid_2020`, hybrid
  density/level-set FG lattices). Both are already cited in the *first* intro paragraph but only as
  "graded porosity is realizable"; the reviewer wants the delta (they optimize graded lattices with
  stress constraints / two-scale description; we assess brittle fracture of stiffness-optimized
  graded structures with porosity-dependent G_c) stated in the "gap"/"purpose" paragraphs.
  Also condense the general intro part before "A more directly related line…" (R2).
- Phase-field section: external work in the functional vs. Griffith/Francfort–Marigo (cite
  10.1016/S0022-5096(98)00034-9, 10.1016/j.aml.2021.107437, 10.1016/j.jmps.2024.105625); rewrite the
  "crack growth is driven by minimization of elastic + fracture energy" sentence; acknowledge spectral-split
  limitations (shear, residual tractions; cite representative crack elements 10.1002/nme.6244); clarify
  κ_s statement (residual of degraded tensile part; ψ⁻ undegraded); state irreversibility threshold;
  say mobility is numerical regularization; diffuse "fat" cracks and non-local damage as limitation.
- G_c(φ_r) mapping: numerically motivated, microstructure-specific, no experimental validation;
  mention possible ligament-mechanism transitions.
- Temper "more efficient use of material" conclusion (sample wording already in the template's
  Revision Working Note). Say comparison focuses on pre-peak/peak measures, not full separation.
- βφ discussion: porosity fields are largely clustered at bounds → limits βφ–β_s interaction claims;
  temper the "internal length scale small vs. geometric features" sentence.
- Add why fracture wasn't put into the optimization (path dependence, irreversibility, cost, sensitivities) (R2).
- Explain E_min(ρ=0.3) ≈ E_max(ρ=0.6) similarity: φ_min=0.5 → E_min has half local mass density,
  so same occupied area; E_max(ρ=0.3) has half the mass → thinner truss (R2).
- Literature comparison for constant-porosity TO results (qualitative, compliance-minimization benchmarks).
- Conclusion: add quantitative highlights (36–64 %, 2.5 %/8 %, peak comparisons).
- Add limitations paragraphs (end of phase-field section + conclusion), per editor.
- Minor (R2): eqs. (5)–(7) index placement, missing brackets in (5),(8),(9); "penalization exponent";
  round μ₁, λ₁; rename summation index in spectral decomposition; remove lone subsubsection 2.2.1
  (make it a paragraph or add 2.2.2); comma→period after "same mass of material is employed";
  bib entries: add journal, drop access dates.

### Earlier co-author (DRJ) items still open
See "not safely resolved" list in `main_DRJ_comment_actions.txt`: italic t* vs upright t style,
κ_opt/κ_frac naming, a schematic comparing Ω and Ω_f (now exists as `fig:omega_omega_f`), "remove
paragraph?" on p. 26, legend placement in one plot.

## 6. Data / plotting workflow

- Regenerate all publication plots (needs Python 3 + numpy, matplotlib, h5py, LaTeX for `text.usetex`):
  `bash archive_tools/reproduce_manuscript_plots.sh` → `reproduced_plots/new_W_whole_boundary/`
  (never overwrites `plots/`). `OUTPUT_ROOT` env var changes destination.
- Individual scripts take `RESULT_ROOT` + `--fixed-beta {0.01|0.05}` `--splits spectral` `--a-values 6`;
  βφ=0.05 evaluations use `--shared-constant-beta 0.01` to borrow E_min/E_max references.
- Result naming: `results_<dataset>_<case>_<split>_eps<β_s>.xdmf/.h5`, `result_graphs_*.txt`
  (scalar histories), `vol_*.json` (volume + averaged E, G_c, μ), `convergence_log_*.txt`.
  Keep `.xdmf` next to its `.h5`. Open `.xdmf` in ParaView.
- Cases: `case ∈ {var, min, max}`, `split = spectral` (volumetric split exists in resources but is
  not used in the paper), `a=6`, `rho ∈ {0.3,0.6}`, `beta ∈ {0.001,0.01,0.05}`, `epsilon` = β_s.
- Re-running simulations needs DOLFINx + the Apptainer image `alex-dolfinx.sif` on the cluster
  (SLURM, 6 MPI tasks/case); see `SOFTWARE_ENVIRONMENT.md`. `export PYTHONPATH="$PWD/code/vendor:$PYTHONPATH"`.
- **Environment caveat:** the local device shell (Cowork VM) has `python3`, `latexmk`, `pdflatex` but
  **no h5py/numpy stack** → plotting scripts won't run there without a venv + `pip install -r
  requirements-postprocessing.txt` (network may be unavailable). HDF5 result files are large;
  avoid staging them into the cloud container.
- Zenodo deposit is prepared but blocked on author decisions (licenses, DOI) — `ZENODO_UPLOAD_CHECKLIST.md`.
  Data-availability statement in the paper says "published upon acceptance".

## 7. Learnings / working rules

- **Edit the revision in `main_revised_template.tex`, not `main.tex`.** `main.tex` = submitted
  version, needed for diffing and for the response letter.
- Every content change in the revision must be wrapped in `\added/\deleted/\replaced` and get a
  matching bullet in `response_to_reviewers_template.tex` ("Related changes in the manuscript").
- Numbers in the text (peak forces, percentages, compliance values) must be traceable to
  `Images_A02/summary_260504.csv` or the TO output; re-verify before changing any of them.
- When applying comments from an annotated PDF: the PDF may be older than the current `.tex`;
  only apply comments whose target can be located unambiguously, log the rest for manual review
  (that is how `main_DRJ_comment_actions.txt` was produced — reuse that format).
- Don't regenerate or replace figures silently; document new sources in `manuscript_picture_list.txt`
  and copy curated images into `Images_A02/` (fracture) or `Images_A03/` (TO).
- The reviewers were sensitive to overclaiming. Prefer "indicates", "within the investigated
  configuration", and explicitly name model limitations (spectral split, mobility, diffuse cracks,
  numerical G_c mapping, single benchmark) instead of general efficiency statements.
- Keep the paper's identity: a *sequential design-and-assessment* study; fracture is NOT optimized.
- The H-S issue is the dominant open question (see Status). Any reviewer response about the
  "physical motivation" of porosity/G_c must be consistent with whichever option the team picks —
  don't draft those paragraphs before the Viko decision.
- Reviewer 1's technical concerns come from the phase-field community (Larsen two-potential,
  representative crack elements, principal stresses) — address them with citations and honest
  limitations rather than new modelling.
- Co-author roles: TO part (Sections 2.1, 3.1, Fig. 4, Table 1/2, Images_A03) is Pravda/Jantos/Junker
  territory — coordinate before rewriting; fracture part is Schlüter/Müller.
- Language: manuscript and all notes in English; conversation with Alex may be in German.

## 8. Update 2026-10-05: terminology applied (Ján's proposal + φ_r solution)

Terms (from Ján's reply, φ_r kept; Alex forwarded the proposal 2026-10-05):

| symbol | new term | note |
|---|---|---|
| ρ | resource budget | "equal mass" → "same resource budget" |
| φ_p = (1−φ_min)φ + φ_min | local resource consumption | integrated in the budget constraint Eq. (9) / `eq:C` |
| φ_r = 1 − φ_p | local resource saving | share saved vs. the most expensive grade; x-axis of the G_c figure |
| φ | material design variable | not just "design variable" (χ is one too) |
| β_φ | material-gradient scale | "gradient scale of the material design field" at first mention |
| E_min/E_max refs | single-grade reference structures | |

G_c mapping text: G_c is a decreasing function of φ_r; form + parameters from Schlüter & Müller 2025
(circular pores, φ_r = porosity there); here a model assumption for a cheaper grade with lower
fracture resistance, no porous microstructure claimed; numerically motivated, microstructure-specific,
not validated, possible stretching→bending transition. This is the only place pores are mentioned.

Done 2026-10-05: 67 `\replaced`/direct edits in main_revised_template.tex (abstract, keywords,
intro, methods intro, Sec. 2.2 incl. G_c paragraph/heading/eq. 32/caption, 3.2, 3.3 heading +
text, conclusion), AS-wording comments removed. Response letter: editor item 4 written, R1 G_c
comment answered, related-changes bullets, "material budget" → "resource budget", title
proposal "Post-optimization brittle-fracture assessment of topology-optimized structures with
graded material" (to be agreed). Remaining "porosity" words in our sections: literature in the
intro (lattices, Schlüter & Müller) and internal labels only. Sec. 2.1/3.1 untouched (Hannover).
Check PDFs: review_feedback/*CHECK*2026-10-05.pdf (built in the cloud container with device ulem.sty).
Build note: the device TeX lacks algorithm.sty, the container lacks ulem.sty.
Not yet committed/pushed to Overleaf (Alex). German email draft to Ján given in chat.

### 8b. Update 2026-10-05 (later): G_c directly over φ, G_{c,1}, parameter fix
- Alex decided: no φ_r reinterpretation. G_c is written directly as G_c(φ)/G_{c,1} =
  clip(A − B exp(−C[√(π/(4(1−φ_min)(1−φ))) − 1]), 0.1, 1); "(1−φ_min)(1−φ) takes the role of the
  porosity" of the circular-pore study. φ_r ("local resource saving") is no longer used in our
  sections; Sec. 2.1 (Hannover) may drop φ_r, φ_p stays (budget constraint).
- Reference toughness renamed G_c^0 → **G_{c,1}** (value at φ = 1, analogous to E_1).
- **Parameter error fixed:** the manuscript (submitted and revised) printed A = 1.26, B = 0.78,
  C = 1.35. The code (`compute_jmax_grid_from_porosity`, 000_template/01_phasefield…py) and the
  submitted Fig. 2 use **A = 1.243657, B = 3.150239, C = 2.850765** (fit from
  07_fit_curve_to_Gc_eff_vs_wsteg.py). With these, G_c = 0.1 G_{c,1} for φ ≤ 0.145 and
  G_c = G_{c,1} for φ ≥ 0.564. Results unaffected; stated neutrally in the R1 G_c response.
- New Fig. 2: `Images_A02/Gc_E_vs_phi.png` (G_c/G_{c,1} and E/E_1 over φ), script
  `17_plot_gc_E_vs_phi.py` (cmr10 + mathtext, no usetex needed). Picture list updated.
- Check PDFs: review_feedback/*CHECK*2026-10-05b.pdf.
- Code column naming (for reference): cell_data.csv 'porosity raw' = φ, 'porosity' = φ_p;
  the code computes φ_r = 1 − φ_p.
- Heading (Alex, 2026-10-05): Sec. 2.2 is now "A phase field model of brittle fracture with spatially
  varying fracture toughness"; the G_c part has no own heading (former 2.2.1 merged, R2 item answered
  accordingly).
- Symmetry check (2026-10-05): TO fields χ, φ, E of all checked domains are exactly mirror-symmetric
  about x = 3 mm, strip centres exactly at 2/4 mm, BCs identical on both sides. Fracture meshes are
  unstructured (≈0.2 % mirrored nodes; strip/edge node sets differ, e.g. 8 vs 9 strip nodes).
  Left/right strip reactions differ by 0.04–0.7 % before the first crack (<1 % at the first peak);
  the side of the first crack varies between cases (bifurcation picked by mesh/interpolation/round-off).
  Two sentences added at the end of the crack-pattern paragraph in Sec. 3.2.
- Title (Alex, 2026-10-05, to be confirmed by Hannover): "From stiffness to strength: brittle fracture of
  graded structures obtained by topology optimization". In manuscript (\replaced) and response letter;
  TODO for Ján/Hannover to confirm.

### 8c. Update 2026-10-05 (midday): independent check + letter rebuilt
- Independent check (separate agent, all numbers recomputed from the 36 M=1e5 result_graphs):
  `review_feedback/independent_check_2026-10-05.md`. Fixed in our sections: H1 (at ρ=0.3 E_var is
  2.6 % stiffer than E_min and 17 % stiffer than E_max — the old "2.5 % vs stiffest grade" was wrong),
  β_φ qualifiers, G_c paragraph (fit data cover φ ≥ 0.30, bounds 0.1/1 are extra assumptions),
  σ_c overclaim, energy balance (up to ~7 % right after load drops), terminology (single-grade
  reference made of the cheapest / most expensive grade), outlook (shear-capable crack models,
  non-local damage), strong form Eq. (27) now 2β_s∇·(G_c∇s) (\changed), roadmap, first-peak
  definition (1 % below running maximum), post-peak D_s numbers in Sec. 3.2.2, macros use \xspace.
  Re-check by the same agent: all fixed; open = Hannover parts, abstract length (~390 words),
  working note / \mycomment removal before submission.
- Alex 2026-10-05: G_c depends abstractly on material grade; S&M 2025 cited only as source of the
  curve shape. H-S explanation goes into letter item 1 as TODO (Hannover) — deviates from the
  2 Sep agreement, must be agreed by all authors.
- Response letter restructured: item 1 = meaning of φ (H-S TODO), 2 = recomputed results,
  3 = revised/withdrawn statements, 4 = limitations, 5 = numerical details/corrections; pointer
  to the overview at the start of R1 and R2; template section removed.
- **Related changes are quoted verbatim** (marked-up text) by `18_build_response_quotes.py`
  (compendium root). In the letter, `%@quote cmd:<text>` / `cmdall:` / `env:<label>` lines request
  quotes; the script fills `%@generated-begin/end` blocks, resolves \ref/\cite from the manuscript
  .aux, and lists unquoted manuscript changes. Workflow: compile manuscript → run
  `python3 ../18_build_response_quotes.py main_revised_template.tex main_revised_template.aux
  response_to_reviewers_template.tex` in the Overleaf folder → compile letter. Needs ulem (device
  TeX) and algorithm.sty (container) — builds were done in the cloud container.
- Check PDFs: review_feedback/*CHECK*2026-10-05c.pdf (letter 43 pages).
- Brevity pass (Alex 2026-10-05: "only what is absolutely necessary", no boilerplate): 33 new passages
  shortened, new text 7273 → 5277 words (−27 %), manuscript 49 → 46 pages, letter 43 → 39. No number
  changed or added (checked by diff of numeric tokens). Ginzburg–Landau paragraph reduced to 3
  sentences. Keep new additions short from now on.
- Clean manuscript (2026-10-05): `68c3…/main_revised_clean.tex` is GENERATED from
  main_revised_template.tex by `19_make_clean_manuscript.py` (accepts all changes, removes \mycomment
  and the Revision Working Note, drops the changes package). Never edit it by hand; re-run the
  script after changes. Clean PDF: review_feedback/main_revised_CLEAN_2026-10-05.pdf (40 pages).

### 2026-10-06: Alex's annotated clean PDF worked in (review_feedback/*2026-10-06.pdf)
- Rules from Alex (apply to all future edits): be concise, no repetition, no forward references to
  simulation details the reader has not seen yet, no bracket formulations, no semicolons, numbers in
  the abstract not bold (fixed via \mathversion{normal} in \abstract).
- Sec. 2.2 reworked: limitations paragraph removed (spectral split stated once after its definition;
  diffuse zones/D_s where results are discussed; inertia in the conclusion limitations, one sentence,
  cites schlueter_phase_2018 without the 30 %); D_s equation moved from Sec. 3.2 to Sec. 2.2 (after the
  mobility paragraph); end displacement 0.03 mm and member-width/β_s statement moved to Sec. 3.2;
  Ginzburg–Landau paragraph without brackets, "all loads considered in this work"; G_c paragraph
  reduced to: assumption, parameters, thresholds, not validated — S&M 2025 no longer cited as origin in
  Sec. 2.2 or intro (only in the literature paragraph). Letter adjusted accordingly (G_c response gives
  the origin "for the reviewer's information").
- Further repetition removed: panel (d) sentence, mechanism sentence after "opposite outcome", 0.06 %
  in Sec. 3.2.2, closing-sentence numbers and mobility sentence in the conclusion.
- Pages: marked-up 45, clean 39, letter 39.

### 2026-10-06 (b): second annotation round (d5c07301-main_revised_clean.pdf)
- Sec. 3.2 now has 3.2.1 "Setup of the fracture simulations" and 3.2.2 "Crack patterns and structural response".
- Former 3.2.1 (secondary measures, Table 5) removed; one sentence after the first-peak comparison mentions the maximum load (confirms ranking, depends on end displacement/supports, not evaluated further). Conclusion/intro/letter adapted.
- Former 3.2.2 (mobility) moved to Appendix A (Fig. A1); `\begin{appendices}` before the bibliography. 18_build_response_quotes.py now reports locations inside appendices as "Appendix~A".
- W: Eqs. for W (integral) and trapezoidal sum removed; one sentence "computed approximately by trapezoidal integration of R_y over u_y^*". Energy-balance closure sentence moved to the Fig. 18 panel (d) discussion ("Panel (d) visualizes ...").
- Removed: "seven nodes" sentence, "mathematically", the peak-load remark in the crack-location paragraph (reader does not know the peak loads yet), the "specific to the chosen beta_s" clause.
- Symmetry: crack-pattern paragraph now explains the asymmetry (mesh perturbation selects the side); the "<1 % strip reactions" statement sits in the Fig. 18(a) discussion.
- Fig. 18(c): vertical axis limited to 3 Nmm/mm (15_manuscript_figures_M1e5.py patched: `ax_e.set_ylim(0.0, 3.0)`); caption notes the limit. Figure regenerated in the container (device LaTeX lacks cm-super for matplotlib usetex). Do not copy the regenerated Mobility figure from a run without --reference-root/--mobility-root.
- Table 2: arraystretch 1.3 and TODO (Hannover) for the unit of the compliance values.
- Letter: summary point 2 and the Fig. 18a / mobility answers refer to Appendix A, Fig. A1; summary point 5 notes the restructuring and that minor wording changes are marked but not quoted.
- Abstract shortened to 245 words (journal limit 150–250, 4–6 keywords; 5 keywords present). Cuts: first sentence, "In this work", benchmark/gradient-scale sentences merged, semicolon sentence rewritten, "peak reaction force, and work up to peak load" → "peak load". `\mycomment` on the word limit removed. Letter point 5 mentions the shortening.

### 2026-10-07: third annotation round (annotated main_revised_clean.pdf in the Overleaf folder) + sequential-reading check
- Intro: "unphysical crack growth", "predictions of crack nucleation", "crack surface density"; roadmap no longer mentions the mobility (undefined at that point).
- Sec. 2.1 (Lamé paragraph, ours): "material design variable", "modulus of the cheapest grade". Rest of 2.1/3.1 still says "porosity" -> Hannover.
- Sec. 2.2: "beta_s is varied in Section 3 to study its influence"; "spectral split introduced below" -> "spectral split"; G_c paragraph refers to Eq. (eq:E_phi), not eq:E_eff.
- Sec. 3.2.1: exception sentence (E_min, 0.0284 mm) removed (not visible anywhere; the letter keeps the caveat); "mean" bracket removed; W: trapezoidal formula is back as Eq. (eq:total_boundary_work_trapezoidal), "over the time increments n"; strip equation ends with a period.
- Sec. 3.2.2: "Figs. 13–16"; beta_s bold in the captions of Figs. 13–16 (\boldsymbol, marked with \changed); redundant Pi_tot sentence deleted.
- Figure placement: Figs. 9–11 at 0.72\textwidth, Fig. 12 at 0.85\textwidth, Fig. 12 environment moved before the field discussion, \FloatBarrier before 3.2.2 -> Figs. 9–12 on the pages of their text.
- Sec. 3.3 / conclusion: statements on the maximum load (24–40 % above E_max, "but not the maximum load") removed because the data are no longer reported; "(within 2.4 %)" bracket rewritten; "Figs. 21–24".
- Letter: anchors updated (W equation, Fig. 18 caption, Lamé wording). 38 pages. Check PDFs: review_feedback/*_2026-10-07.pdf.
