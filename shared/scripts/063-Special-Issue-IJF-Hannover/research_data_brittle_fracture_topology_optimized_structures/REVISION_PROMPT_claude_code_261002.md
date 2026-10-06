# Prompt: continue the revision of manuscript A03 with the new fracture results (2026-10-02)

Copy everything below the line into a new Claude Code session started in
`shared/scripts/063-Special-Issue-IJF-Hannover/research_data_brittle_fracture_topology_optimized_structures`.

---

Continue the major revision of the manuscript "Brittle fracture in topology-optimized structures" (A03, Archive of
Applied Mechanics, decision letter of 2026-08-18, deadline 15 Oct 2026). Read `../CLAUDE.md` first, then
`MOBILITY_STUDY_FINDINGS.md` (sections 8 to 13 are the new results), `review_feedback/review_18.08.2026`,
`review_feedback/response_to_reviewers_suggested_2026-08-18.tex`, and the manuscript
`68c3b8d0b7dca7b64b8b7a93/main_revised_template.tex` (results section, section on the phase-field model, figure
captions of the response/energy plots and of the peak-metric plots).

**Phase 1 of this task is a proposal only. Do not edit any .tex file, figure, or evaluation script until I give
the green light.** Pull the Overleaf repository (`git pull` in `68c3b8d0b7dca7b64b8b7a93`) before reading, because
co-authors edit online.

## What has changed since the submission (all verified, see the findings file)

1. **Reaction force and work.** The submitted reaction force was a traction integral over the strip facets,
   which for linear elements underestimates the resultant by 24 to 38 %, differently for each structure (`13_test_reaction_force.py`,
   findings section 8). The published "total-boundary work" (column 13) is off by −8 to +10 % at the peak and
   fails after cracking. The new simulation script computes the reaction from the assembled residual
   (energy-consistent nodal reactions, `pp.reaction_force_from_residual` in `code/vendor/alex/postprocessing.py`)
   and its work; energy balance closes to 0.2 % at the peak. Decisions already taken: report the **total of both
   strips**; keep the definition of W in the text as the surface integral of Eq. (total_boundary_work); only
   the sentence on its numerical evaluation (trapezoidal stress integral, Eq. total_boundary_work_trapezoidal)
   changes. The plotted R_y was the left strip only; the strips fail at slightly different times.
2. **Mobility.** M = 100 was not quasi-static (findings sections 1 to 2, 10). Peak loads follow R(M) ≈ R∞ + c·M^(−2/3).
   The whole study was rerun with **M = 10⁵** (within about 1 % of the rate-independent limit), see section 10.
   Pre-peak mobility dissipation is then below 0.05 % of the work; post-peak D_s is the energy released in
   unstable crack growth and does not vanish with M.
3. **Solver settings** (section 11): the former runs ended at Newton failures caused by an absolute tolerance
   (1e-10) below the round-off floor of the residual. New settings: Newton absolute tolerance 1e-8,
   20 iterations, smallest time step 1e-10 s, all runs to a common end point u_y = 0.03 mm. First peaks are
   unchanged by this to 6 digits. This has to be stated in the numerical-implementation part.
4. **Peak definition**: the strength measure is the **first peak** of the total reaction (first load drop larger
   than 1 %, i.e. the load at the first crack event) and the work up to it. Reason: after the first crack event
   every structure keeps carrying load because both ends are clamped and reloads; E_min (ρ = 0.3) and E_var
   (ρ = 0.6) even exceed their first peak again before u_y = 0.03 mm. Maximum load and total work up to
   u_y = 0.03 mm are reported as secondary measures with the end point and the supports named explicitly
   (section 13).
5. **New results** (`results/campaign_260930_M1e5/`, evaluation `14_energy_consistent_metrics.py --extra-root
   results/campaign_260930_M1e5` → `plots/energy_consistent/`, tables in `energy_consistent_summary.md`,
   figures `*_M_comparison.*` and `*_M100000.*`, sections 12 and 13 of the findings):
   - ρ = 0.3: E_var is 82 to 92 % above E_min (published 36 to 64 %) and 26 to 32 % **below** E_max (published
     "about equal"). ρ = 0.6: E_var is 19 to 21 % **above** E_max (published "about equal"). Work at the first
     peak: +219 to +260 %, −50 to −57 %, +42 to +52 %.
   - All first peaks fall by about 27 % from β_s = 0.015 to 0.06 mm, E_min included (the flat published E_min
     trend was a rate artefact).
   - β_φ: at ρ = 0.3 the three values agree within 2 % (β_φ = 0.001 up to 8 % lower at large β_s); at ρ = 0.6 the
     coarsest gradient β_φ = 0.05 is 7 to 14 % weaker in the first peak than β_φ = 0.001, with its margin over
     E_max down to +7 to +18 %; in the maximum load this difference disappears. The published "β_φ affects the
     peak only mildly" has to be qualified.
   - Secondary measures (section 13): max load E_var vs E_min +21 to +37 %, vs E_max −26 to −32 % (ρ = 0.3) and
     +28 to +33 % (ρ = 0.6); total work to 0.03 mm: −1 to +4 % (ρ = 0.3 vs E_max), +30 to +43 % (ρ = 0.6).
6. Also found: panel (b) of the peak-metrics figure was computed from the two-strip work (column 4) while the
   caption refers to the total-boundary equation; the loaded strips span 6 facets (0.06 mm), not 0.075 mm, on the
   0.01 mm mesh.
7. **Already in the revision (commit d00482b, 2026-09-29, written before these results):** the phase-field section
   of `main_revised_template.tex` (around line 987) and the response to Reviewer 1 on the mobility (around line 319
   of `response_to_reviewers_template.tex`) still state M = 100 and announce a table `tab:dissipation_ratios` with
   D_s ratios at M = 100 (TODO comment around line 1957). These must be revised to M = 10⁵ with the convergence
   argument, and the D_s table must use the new campaign. List them explicitly in the proposal.
8. Still pending: the Hashin–Shtrikman decision on the TO structures (CLAUDE.md, Status). Do not draft the
   "physical motivation" paragraphs; the proposal must say where it depends on that decision.

## What I want from you in phase 1 (proposal only, no edits)

A. **Story check.** Say plainly whether the paper's story still holds with the new results and where it changes.
   The old message was "graded porosity gives a stiffness gain of 2.5 / 8 % and a fracture resistance that is
   36 to 64 % above uniform porosity and about equal to the solid structure". With the new numbers the solid
   structure is 30 % stronger at ρ = 0.3 and the graded one 20 % stronger at ρ = 0.6. Is that still a consistent,
   defensible story, and what is the honest one-paragraph abstract-level statement? Name any claim in the
   current text that is no longer supported. Ask me explicitly whether I agree with the story before proposing
   the wording.
B. **Change proposal for the manuscript**, section by section, as a list of concrete edits (old statement → new
   statement, with the numbers taken from `energy_consistent_summary.md` and the findings file, not from
   memory): model and implementation section (reaction force and work evaluation, mobility value and its
   justification with the M^(−2/3) convergence, solver tolerances and time stepping, irreversibility threshold,
   common end point), results section (first-peak definition, every quantitative statement, the β_s and β_φ
   discussions, the post-peak reloading), figures to replace (which file from `plots/energy_consistent/` replaces
   which figure in `Images_A02/`, and which new figure is needed, e.g. the full curves to u_y = 0.03 mm and the
   mobility-convergence plot), tables, abstract and conclusion. Keep the sequential design-and-assessment
   identity of the paper. Mark which edits are text-only and which need a new figure.
C. **Change proposal for the response to reviewers** (`response_to_reviewers_template.tex` structure): for every
   reviewer item that the new work touches (Reviewer 1: artificial damping / mobility dissipation, principal
   stresses, mesh statistics; Reviewer 2: crack-evolution snapshots, quantitative conclusions) draft the
   argument, not yet the final text.
   **Framing rule for the changed numbers.** The response must say that all fracture results were recomputed and
   that the quantitative statements changed, and it must give the reasons, because the reviewers will compare
   the versions. But present the reasons as the methodological refinements they are, in a factual tone, not as
   confessions: (i) the reviewer's question on the mobility term led to a convergence study in M; it showed that
   the peak loads of the porous structures were still rate-affected at the submitted M, so the study was
   repeated at M = 10⁵, within about 1 % of the rate-independent limit, with the M^(−2/3) convergence documented
   (this is a direct, positive answer to the reviewer); (ii) in the revised evaluation the reaction forces and the
   external work are obtained from the energy-consistent nodal reactions of the discrete system instead of a
   surface integral of the interpolated stress, which closes the energy balance to 0.2 % at the peak and
   affects the absolute values and the structure-to-structure ratios; (iii) the time stepping and solver
   tolerances were adapted so that every simulation resolves the unstable crack growth up to a common end
   displacement, which made the full post-peak response and the secondary measures available. Do not use the
   words error, bug, mistake, wrong or incorrect; do not apologise; do not claim the submitted results were
   qualitatively right where they were not, and do not claim that M = 100 was negligible. Where a published
   claim is withdrawn, say which one and what replaces it. Use "indicates" and
   "within the investigated configuration".
D. **Work plan after the green light**: ordered list of edits with the files touched, which require coordination
   with the Hannover co-authors (TO sections are theirs), and the estimated effort. Everything in the manuscript
   must later be marked with the `changes` package (`\added`, `\deleted`, `\replaced`) and every substantive edit
   must get a bullet in the response letter.

Keep to the conventions in `../CLAUDE.md` (notation, "homogeneous", "penalization exponent", single .tex file,
numbers traceable to the summary files). Conversation with me may be in German; manuscript and notes in English.
End phase 1 with the questions you need answered, the story question first.
