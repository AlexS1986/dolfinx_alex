# Mobility study: findings for agents

Status: complete (2026-09-29). Side study for Reviewer 1 (mobility/"artificial damping").
**Not in the manuscript.** Decision pending (Alex): rerun the fracture campaigns with larger M or not.
Full method, commands and tables: `MOBILITY_STUDY_README.md`. This file holds the conclusions only.

## Setup (fixed for every number below)

- Spectral split, β_s = 0.03 mm, β_φ = 0.01 mm², a = 6, κ_s = 0.001, Δt_max = 0.001 s, v₀ = 1 mm/s.
- M in mm³/(Nmm s). M = 100 is the published campaign (`results/new_W_whole_boundary/`, not recomputed).
- New runs: M = 1000 and 10000 for ρ = 0.3 E_var, ρ = 0.3 E_min and ρ = 0.6 E_var; M = 10000 for ρ = 0.3/0.6 E_max.
  All runs passed their peak.
- W = column 13 of `result_graphs_*.txt` (total-boundary work, the manuscript W) unless stated otherwise.
- u_y in mm = t in s (v₀ = 1 mm/s).

## 1. Peak load max R_y [N/mm]

| case | M = 100 | M = 1000 | M = 10000 | change 100 → 10000 |
|---|---|---|---|---|
| ρ = 0.3 E_max | 47.29 | – | 44.20 | −6.5 % |
| ρ = 0.3 E_var | 45.26 | 33.73 | 31.35 | −30.7 % |
| ρ = 0.3 E_min | 29.71 | 20.93 | 18.58 | −37.4 % |
| ρ = 0.6 E_max | 85.57 | – | 79.58 | −7.0 % |
| ρ = 0.6 E_var | 89.43 | 84.81 | 81.34 | −9.0 % |

- **M = 100 is not quasi-static for the peak load of porous (low-G_c) structures.**
  The rate term delays damage in low-G_c regions (relaxation time τ ≈ β_s/(M G_c) ≈ 3·10⁻³ s at
  M = 100, G_c = 0.1; the time to peak is 0.009–0.017 s).
- Pre-peak stiffness is unchanged; the peak moves to smaller u_y. Post-peak branches of M = 1000 and 10000 nearly coincide.
- Convergence: ρ = 0.3 nearly converged (extrapolated M → ∞: E_var ≈ 30.7, E_min ≈ 17.7 N/mm).
  **ρ = 0.6 E_var not converged** (steps −4.6, −3.5 N/mm).

## 2. Consequences for the paper's comparisons

| comparison | max R_y M = 100 → 10000 | W at peak M = 100 → 10000 | verdict |
|---|---|---|---|
| ρ = 0.3: E_var vs E_min | +52 % → +69 % | +157 % → +208 % | holds, stronger |
| ρ = 0.6: E_var vs E_max | +4.5 % → +2.2 % | +88 % → +80 % | holds (margin not settled, see 1.) |
| ρ = 0.3: E_var vs E_max | −4 % → **−29 %** | −11 % → **−54 %** | **changes** |

- Robust message: graded porosity (E_var) clearly beats uniform max porosity (E_min), and E_var ≈ E_max at ρ = 0.6.
- Not robust: any claim that E_var at ρ = 0.3 comes close to E_max in peak load or work at peak.
- Headline "E_var is 36–64 % above E_min" (all β_s, M = 100) likely grows at larger M. It was only
  checked for β_s = 0.03 mm.

## 3. Size of D_s (mobility dissipation)

| case | D_s/W at peak, M = 100 | D_s/W at peak, M = 10000 | D_s/W final, M = 100 | D_s/Π_el final, M = 100 |
|---|---|---|---|---|
| ρ = 0.3 E_max | 0.5 % | 0.01 % | 55 % | 1.40 |
| ρ = 0.3 E_min | 4.7 % | 0.2 % | 11 %* | 0.26 |
| ρ = 0.3 E_var | 4.0 % | 0.1 % | 31 % | 0.58 |
| ρ = 0.6 E_max | 0.6 % | 0.01 % | 41 % | 0.80 |
| ρ = 0.6 E_var | 1.4 % | 0.04 % | 16 % | 0.27 |

\*E_min post-peak W unreliable (see 4.).

- At the peak, Π_el is 85–104 % of W and Π_frac only 2–6 %. D_s/Π_el ≈ D_s/W at the peak.
  At M = 100, D_s at the peak is comparable to Π_frac (D_s/(Π_frac+D_s) = 0.12–0.67).
- Pre-peak D_s falls ≈ ∝ 1/M only above M = 1000 (factor 7–10 from 1000 → 10000; only 3.5–4 from 100 → 1000).
- **Post-peak D_s does not vanish with M** (final D_s/W still 37–47 % for E_max at M = 10000).
  It is the energy released in unstable crack growth (snap-through under displacement control).
  In the quasi-static limit this energy is dissipated anyway. Do not describe it as an artefact that
  a larger M removes.

## 4. Data-quality issues found (affect the manuscript, not only this study)

- **Energy balance after the peak.** (Π_el + Π_frac + D_s)/W is 0.90–1.09 at the peak for all runs,
  but 0.44–0.58 at the final state for E_min: W keeps growing after cracking. Also visible in the published
  `plots/new_W_whole_boundary/beta001/evaluation/spectral/Response_energy_grid_vs_uy_*` panel (d).
  For post-peak ratios, use Π_el or Π_frac as the reference, not W.
- **The two-strip work W_strips (column 4) is 30–60 % below Π_el even before the peak.** It equals
  ∫R_y du, but it misses part of the work entering the body.
  Column 4 is `QUANTITIES["Work"]` in `09_evaluation_260504_parameter_space.py` and feeds `max_Work`
  in `Images_A02/summary_260504.csv` and the work-at-peak plots. **Check which W the manuscript's
  work-at-peak statements use before relying on them** (e.g. "E_var exceeds E_max in work at peak for ρ = 0.6").

## 5. Should M be increased beyond 10000?

- Not needed for pre-peak dissipation (already ≤ 0.2 % of W). It does not remove post-peak D_s (see 3.).
- Cost grows: M = 10000 needs about 2× the steps of M = 100, and runs stop earlier (Newton failure);
  ρ = 0.6 E_var at M = 10000 stopped right after the peak.
- Only M/v₀ matters (no inertia): M × k ≡ loading k× slower. This may be easier to argue in the paper
  ("quasi-static loading rate").
- **Recommendation:** rerun with M = 10000 if rerunning. Before that, one ρ = 0.6 E_var run at M = 10⁵ up to just
  past the peak (~1 h locally) to settle the ρ = 0.6 E_var vs E_max margin.

## 6. Guidance for writing (reviewer response / manuscript)

- Do not claim the mobility term is negligible for peak loads at M = 100.
- Do not draft ρ = 0.3 E_var vs E_max statements from the M = 100 results.
- Safe to state: pre-peak D_s at M = 100 is 0.5–4.7 % of W; it scales ∝ 1/M for large M; the E_var > E_min
  ranking is insensitive to M.
- Keep consistent with the pending Hashin–Shtrikman decision (CLAUDE.md, Status), which may replace the TO
  structures anyway.

## 7. Files

| what | where |
|---|---|
| **PDF report of the M = 10⁵ campaign** (9 pages: tables, curves, β_φ study, secondary measures, all 36 runs) | `plots/energy_consistent/energy_consistent_report.pdf` (source `.tex` next to it) |
| single PDF report (text, tables, all figures; 12 pages) | `plots/mobility_study/mobility_study_report.pdf` (source `.tex` next to it; build with `latexmk -pdf` in that folder) |
| method, commands, full table, caveats | `MOBILITY_STUDY_README.md` |
| summary table | `plots/mobility_study/mobility_study_summary.csv` / `.md` |
| force–displacement plots | `plots/mobility_study/Ry_vs_uy_mobility_all.png`, `Ry_vs_uy_mobility_<dataset>.png` |
| dissipation plots | `plots/mobility_study/Dissipation_vs_uy_mobility_all.png`, `Dissipation_grid_vs_uy_mobility_<dataset>.png` |
| per-case overview | `plots/mobility_study/mobility_<dataset>_<case>.png` |
| evaluation script (regenerates all of the above) | `12_mobility_study_eval.py` (numpy + matplotlib, LaTeX) |
| raw results (.h5/.xdmf/.txt, outside git) | `/Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/mobility_study_260929/` |
| M = 100 reference histories | `/Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/results/new_W_whole_boundary/simulation_20260526_122937_SPLITspectral_EPS0_03/` |
| simulation option | `000_template/01_phasefield_dcb_260504_folder.py --mobility M` (default 100; output suffix `_M<M>`) |
| campaign scripts (tag `260929_mobility`) | `01_create_directories_…`, `02_create_jobsa_…`, `04_submit_all_…`, `00_jobs/job_template_260929_mobility.sh`, `00_jobs/run_260929_mobility_local.sh` |

Local runs used Docker image `dolfinx_alex-alex-dolfinx` (DOLFINx 0.7.3) with 6 MPI tasks and `code/vendor/alex` first on
PYTHONPATH (exact command in the README). Runtime: 25 min to 1 h per run.

## 8. Follow-up analysis 2026-09-30 (from existing histories, no new runs)

**8.1 The reported reaction force R_y is not energy-consistent (affects the manuscript).**
`pp.reaction_force` integrates the interpolated stress over the strip facets. In the elastic range
(t = 0.003 s, Π_frac + D_s < 0.3 % of Π_el) the energy-consistent total reaction is 2Π_el/u_y. The
traction-integrated total (columns 2 + 7) is only a fraction of it, and the fraction depends on the structure
(independent of β_s and M):

| structure (β_φ = 0.01) | R_traction / R_energy, elastic | same, last 10 % before peak |
|---|---|---|
| ρ = 0.3 E_max | 0.668 | 0.65 |
| ρ = 0.3 E_var | 0.669 | 0.63–0.66 |
| ρ = 0.3 E_min | 0.723 | 0.67–0.70 |
| ρ = 0.6 E_max | 0.760 | 0.74 |
| ρ = 0.6 E_var | 0.651 | 0.61–0.62 |

(β_φ = 0.001: ρ = 0.3 E_var 0.713, ρ = 0.6 E_var 0.686; β_φ = 0.05: 0.703, 0.698.)
This is the cause of "two-strip W 30–60 % below Π_el" in section 4. Dividing the peak loads by these
factors changes the comparisons at M = 10000: ρ = 0.6 E_var vs E_max +2 % → about +20 %;
ρ = 0.3 E_var vs E_min +69 % → about +80 %; ρ = 0.3 E_var vs E_max stays about −29 %.
Also: the plotted R_y is column 2 = left strip only (48–51 % of the total).
**Fix for any rerun:** take the reaction from the assembled residual at the constrained y-dofs of the
strips (consistent nodal forces) and compute W = ∫R du from it. Check: R·u = 2Π_el in the first step.
For existing runs, the pre-peak consistent reaction can be recovered as d(Π_el + Π_frac + D_s)/du.

**8.2 Convergence in M.** Time discretization is not the limit: Δt at the peak is 1.6–6·10⁻⁵ s for the
M = 10000 runs, and R_y changes < 0.1 % over the last step. For ρ = 0.3 the peak load follows
R(M) ≈ R∞ + c·M^(−2/3) (rate-delayed loss of stability): fitted on M = 1000/10000 it predicts 44.8 N/mm at
M = 100 for E_var (actual 45.26). M = 10000 is then about 2 % (E_var) and 3.5 % (E_min) above the limit.
ρ = 0.6 E_var does not follow this law (limit between 71 and 80 N/mm depending on the extrapolation);
ρ = 0.6 and 0.3 E_max have only two points. Other β_s are untested; the relaxation time is ∝ β_s/(M G_c),
so β_s = 0.06 mm at M = 10000 corresponds to β_s = 0.03 mm at M = 5000 (worst case).

**8.3 Runs the paper needs (a = 6 only; the a = 3 cases are not used):** β_φ = 0.01: 5 structures × 4 β_s = 20
(5 exist at M = 10000); β_φ = 0.001 and 0.05: E_var, 2 ρ × 4 β_s each = 16. Total 36, 31 new.

**8.4 Reaction-force fix implemented and tested (2026-09-30).**
- New `pp.reaction_force_from_residual(residual, dofs, comm)` in `alex/postprocessing.py` (identical in
  `shared/utils/alex` and `code/vendor/alex`): sum of the assembled residual, without Dirichlet BCs, over the
  constrained dofs. `pp.reaction_force` (traction integral) is unchanged and got a docstring with its limits.
- Unit test `13_test_reaction_force.py` (linear elastic, reference 2Π_el/u₀; serial and 6 MPI processes
  give identical numbers): residual-based force matches to 1e-10. Traction integral over the strip facets:
  0.52 / 0.69 / 0.78 of the reference for h = 0.02 / 0.01 / 0.005 mm, identical for UFL stress and
  DP0-interpolated stress (P1 elements: the stress is already elementwise constant). Traction over the
  whole top edge for the strip problem: 0.93–0.97. Traction integral for a fully loaded top face:
  1.002–1.003 (rough) and 1.000 (homogeneous stress). So the traction integral is fine for whole faces and
  wrong for strips. At h = 0.01 mm each strip has only 6 loaded facets (0.06 mm instead of 0.075 mm).
- `000_template/01_phasefield_dcb_260504_folder.py` now writes columns 14–17 (residual reaction left,
  right, its work increment and accumulated work; see `DATA_DICTIONARY.md`). Columns 0–13 are unchanged.
  New optional environment variable `PHASEFIELD_T_END` overrides the end time (default unchanged).
- In-situ check on all five structures (β_s = 0.03 mm, M = 10000, t ≤ 0.003 s): residual reaction / (2Π_el/u)
  = 1.0000; old traction value 0.65–0.76. Test runs in
  `/Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/reaction_force_test_260930/`.
- Through the peak (ρ = 0.3 E_min, M = 10000, run to t = 0.0064 s, folder `peak_rho_0_3_min`): columns 0–13 are
  identical to the mobility-study run (the change does not alter the solution). Residual-based peak load
  26.85 N/mm (left strip) and 26.90 N/mm (right strip) instead of 18.58 N/mm. W_res/(Π_el + Π_frac + D_s) =
  0.998 at the peak and 0.96–0.98 after the two load drops; the old measures give 0.72 (strips) and
  0.90–0.99 (total boundary). The traction/residual ratio drifts from 0.71 to 0.62 after cracking, so old
  post-peak curves cannot be corrected by a constant factor. The two strips do not fail together: the
  left reaction drops at t = 0.00539 s, the right one at t = 0.00546 s (the paper plots the left strip only).
- **Evaluation scripts still read column 2 (left strip traction) and column 4.** For the new study switch
  `09_evaluation_260504_parameter_space.py` (`QUANTITIES`) and `12_mobility_study_eval.py` (`COL_RY`,
  `COL_W_*`) to columns 14/15 and 17; old result files have only 14 columns.

## 9. Corrected story from existing data (2026-09-30, no rerun)

Script `14_energy_consistent_metrics.py` → `plots/energy_consistent/` (`energy_consistent_summary.md`, two CSVs,
figures). Method: exact work W = Π_el + Π_frac + D_s; total reaction of both strips R = dW/du, evaluated up to the
peak as traction curve divided by a calibration factor from the energy balance (window 0.1·u_peak). Checked
against the residual-based run: peak load within 0.3 %. Post-peak curves cannot be reconstructed this way.

**Decisions (Alex, 2026-09-30):** report R_y as the **total of both strips**; keep the definition of W in the text
as the surface integral, Eq. (total_boundary_work) — only its numerical evaluation changes.

Whole-boundary traction work (column 13, the W of the manuscript) is not reliable either: W_col13/W_exact at the
peak is 0.93 (ρ = 0.3 E_max), 0.98 (E_min), 0.92 (ρ = 0.3 E_var), 0.95 (ρ = 0.6 E_max), 1.10 (ρ = 0.6 E_var), and
1.7–2.0 at the final state of E_min (spurious traction on free contours and at cracked boundary regions).

Comparisons at M = 100, β_φ = 0.01, range over the four β_s (published = manuscript values):

| comparison | max R_y published | max R_y corrected | W at peak published (col. 4) | W at peak corrected |
|---|---|---|---|---|
| ρ = 0.3: E_var vs E_min | +36 … +64 % | +50 … +78 % | +112 … +188 % | +129 … +210 % |
| ρ = 0.3: E_var vs E_max | −5 … −2 % | −5 … 0 % | −15 … −4 % | −15 … −3 % |
| ρ = 0.6: E_var vs E_max | +4 … +6 % | +26 … +27 % | +34 … +41 % | +60 … +68 % |

- β_φ study, total max R_y at β_s = 0.03 mm for β_φ = 0.001 / 0.01 / 0.05: ρ = 0.3: 131.8 / 146.5 / 145.2 N/mm
  (published, left strip: 46.73 / 45.26 / 50.64); ρ = 0.6: 285.6 / 290.6 / 286.0 N/mm (published 95.74 / 89.43 /
  93.18). The published ordering was an artefact: corrected, ρ = 0.6 is insensitive to β_φ (±1 %) and only
  β_φ = 0.001 at ρ = 0.3 is lower (−10 %). E_var (β_φ = 0.05) no longer exceeds E_max at ρ = 0.3 (published
  +6 … +12 %, corrected −5 … +1 %).
- M = 10000 (β_s = 0.03 mm, β_φ = 0.01), corrected: ρ = 0.3 E_var vs E_min +84 %, ρ = 0.3 E_var vs E_max −30 %,
  ρ = 0.6 E_var vs E_max +22 % (W at peak +225 %, −54 %, +55 %). The M effect itself is unchanged by the
  correction (ρ = 0.3 E_var −33 %, E_min −38 %, E_max −7 %, ρ = 0.6 E_var −10 % from M = 100 to 10000).
- **Manuscript inconsistency:** panel (b) of the peak-metrics figure is computed from column 4 (two-strip work,
  `work_at_peak_reaction` in `09_evaluation`), while caption and text refer to Eq. (total_boundary_work), column 13.
- Consequence: the reaction-force fix strengthens the ρ = 0.6 result and leaves ρ = 0.3 E_var ≈ E_max at M = 100.
  The rerun question is now only the mobility: at M = 10000, ρ = 0.3 E_var is 30 % below E_max.

## 10. Convergence runs at M = 10⁵ (2026-09-30)

Four runs with the fixed script (residual columns 14–17), 6 MPI tasks, stopped by a watcher once the total
reaction fell below 90 % of its maximum. Folder
`/Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/mobility_convergence_260930/` (`driver.sh`, `run_one.sh`).
Peak = first local maximum of the load. Changes are given for the traction-based total (available for all M);
the traction/residual ratio does not depend on M.

| case | M = 10² | M = 10³ | M = 10⁴ | M = 10⁵ | step 10⁴ → 10⁵ | residual total at 10⁵ |
|---|---|---|---|---|---|---|
| ρ = 0.6 E_max, β_s = 0.03 | 170.46 | – | 158.55 | 158.37 | −0.1 % | 213.76 N/mm |
| ρ = 0.6 E_var, β_s = 0.03 (first peak) | 176.93 | 167.82 | 161.15 | 159.36 | −1.1 % | 259.24 N/mm |
| ρ = 0.3 E_min, β_s = 0.06 | 61.65 | – | 32.67 | 31.44 | −3.8 % | 44.50 N/mm (46.35 at 10⁴) |
| ρ = 0.3 E_min, β_s = 0.03 | 60.40 | 42.61 | 37.86 | – | – | – |

- **E_max is converged at M = 10⁴** (0.1 %). ρ = 0.6 E_var first peak is within about 1.5 % of the limit at 10⁴.
- **E_min at β_s = 0.06 mm is the worst case:** M = 100 overestimates the peak by a factor of 1.9; M = 10⁴ is 3.8 %
  above M = 10⁵ and about 5 % above the M^(−2/3) limit; M = 10⁵ is about 1 % above it.
- **The flat E_min peak load over β_s in the manuscript (≈ 60 N/mm for all β_s) is a rate artefact.** At M = 10⁴ the
  traction total is 37.86 (β_s = 0.03) and 32.67 (β_s = 0.06). The published trend "E_var 36–64 % above E_min,
  decreasing with β_s" is therefore also affected.
- D_s/W at the peak at M = 10⁵: 5·10⁻⁶ to 5·10⁻⁴. Cost up to the peak is the same as at 10⁴ (3–5 min per run).
- **Two-stage failure of ρ = 0.6 E_var (new).** At M = 10⁵ the load peaks at u_y = 0.01531 mm (259.2 N/mm), drops by only
  8.6 % (to 236.9 N/mm, left strip 113.4, right 123.5; crack arrest) and then rises again to 279.4 N/mm (+7.8 % above the
  first peak) at u_y = 0.01886 mm, where Newton failed (Δt < 10⁻¹⁴ s) with the load still rising. The M = 10⁴ run had died
  at the bottom of the same partial drop (0.918 of the peak). So the ultimate load of ρ = 0.6 E_var is ≥ 279 N/mm and
  not known; at M = 100 and 1000 the delayed first event was the global maximum. **Decide: report the first peak
  (first crack event, well defined and converged) or the ultimate load.** With the first peak, ρ = 0.6 E_var vs E_max is
  +21.3 % at M = 10⁵ (259.24 vs 213.76 N/mm).
- The Newton failure during rising load shows that the run length is limited by the solver (non-smooth spectral
  split), not by the mobility.
- Recommendation: run the campaign at M = 10⁵ (≤ about 1 % from the limit everywhere, no extra cost up to the peak);
  M = 10⁴ leaves a 3–5 % bias for E_min that grows with β_s.

## 11. Newton convergence diagnosis (2026-09-30, campaign stopped after 5 runs to fix this first)

Case: ρ = 0.6 E_var, β_s = 0.03 mm, M = 10⁵ (fails at u_y = 0.01886 mm with the load still rising). Runs in
`/Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/solver_diagnostics_260930/` (`run_variant.sh NAME "ENV..."`),
per-iteration residuals via the new `NEWTON_LOG=1` (dolfinx log level INFO on rank 0).

**Two failure modes at the fatal step (variant A, plain Newton, 8 iterations):**
1. Δt ≥ 1.6·10⁻⁵ s: the residual diverges (2e2 → 8 → 3e2 → 6e3 → 5e10): genuine Newton overshoot at a crack event.
2. Δt ≤ 4·10⁻⁶ s: the residual converges in 3 iterations to 1.15·10⁻¹⁰ and stays there for all remaining
   iterations. That is the round-off floor of the assembled residual, just above atol = 10⁻¹⁰. The relative
   criterion of the dolfinx NewtonSolver is scaled by the norm of the first update (empirically r_rel = |r|/|Δx₀|;
   |Δx₀| = 221.6 at the first step matches the SNES step norm), which is tiny for small Δt. So no Δt can satisfy
   either criterion and the step size walks down to 10⁻¹⁴ s. Even the converged steps end at 1.1–1.2·10⁻¹⁰, i.e.
   they pass the relative criterion only marginally. **The dt → 10⁻¹⁴ walks of all runs are this artefact.**
   Variant C (50 iterations) confirms it: 50 iterations at 1.15·10⁻¹⁰, same failure point.
- Bad elements are not involved (uniform 0.01 mm triangulation, failure displacement reproducible to 5 digits).
- A pure SNES line-search solver (`NEWTON_TYPE=snes`, class `SNESNewtonSolver` in `alex/solution.py`) is NOT a
  drop-in replacement: at the first step the plain Newton residual rises from 0.11 to 0.63 before converging
  (kinks of |x|, sqrt and the eigenvector conditionals at u = 0), and a backtracking line search refuses such a
  step (reason −6 at every λ). It is kept as an optional fallback only.
- New in `alex/solution.py` (both copies): `NEWTON_LOG`, `NEWTON_TYPE=snes`, `NEWTON_FALLBACK=snes|damped`
  (retry a failed step at the same Δt with a line-search or damped Newton before halving Δt),
  `NEWTON_FALLBACK_MAX_IT`, `NEWTON_FALLBACK_RELAXATION`, `NEWTON_FALLBACK_LINESEARCH`; the existing
  `NEWTON_MAX_IT`, `NEWTON_ATOL`, `NEWTON_RTOL`, `NEWTON_RELAXATION`, `NEWTON_MIN_ITERS` from the main library are
  now also in the vendored copy. Script: `PHASEFIELD_DT_MIN` (default 10⁻¹⁴ s as before).
- **Fix (variant D, `NEWTON_ATOL=1e-8 NEWTON_MAX_IT=20 PHASEFIELD_DT_MIN=1e-8`):** the case runs through the old
  failure point. First peak identical to 6 digits (259.240 N/mm at u_y = 0.015305 mm), energy balance
  W_res/(Π_el+Π_frac+D_s) = 0.998 at u_y = 0.029 mm. What follows is a sequence of partial crack events with
  reloading: right strip at u_y = 0.0189 mm (279 → 242 N/mm total), left strip at 0.0191 mm (243 → 204), then
  reloading to 284 N/mm at u_y = 0.029 mm without a final drop. So the "ultimate load" of ρ = 0.6 E_var is not
  reached within 2× the first-peak displacement; the first peak stays the strength measure.
- Variant E (D + `NEWTON_FALLBACK=damped`, relaxation 0.5): no benefit (fallback needs 30–35 iterations, 30 of 51
  retries fail); not used. Variant C (50 iterations, atol 1e-10): same failure as A, confirms the tolerance cause.
- **Campaign restarted 2026-09-30 22:05 with these settings** (`05_run_…campaign_local.sh`: NEWTON_ATOL 1e-8,
  NEWTON_MAX_IT 20, PHASEFIELD_DT_MIN 1e-8, common end point PHASEFIELD_T_END 0.03 s, wall limit 4 h). The five
  runs of the first attempt (old solver settings) are kept in `campaign_260930_M1e5_oldsolver_partial/`.
- **Second pause (ρ = 0.3 E_min, β_s = 0.03, M = 10⁵):** with floor 10⁻⁸ s the run stopped inside the first drop
  after a Δt ratchet (every failure halves Δt, Δt grows only after steps with < 4 iterations). Variants on E_min:
  F (`NEWTON_MIN_ITERS=10`, floor 10⁻⁸): passes the drop but wastes 243 of 520 attempts on too-large steps, ends
  at u_y = 0.0259 at the floor; H (floor 10⁻¹⁴, default Δt policy): same history (first peak 52.63, global max 74.4
  at u_y = 0.0215), 815 converged steps of which 189 at Δt < 10⁻⁸ s; G (SNES bt fallback): 4 of 41 retries
  succeed, useless. Histories of F and H agree, so the Δt path does not affect the result.
- **Campaign restart 2 (2026-10-01 00:01):** NEWTON_ATOL 1e-8, NEWTON_MAX_IT 20, default Δt policy,
  PHASEFIELD_DT_MIN 1e-10, PHASEFIELD_T_END 0.03. E_min has a further hard event at u_y ≈ 0.026 mm (both F and H
  stall there); runs that end by solver failure before u_y = 0.03 mm are still valid up to their last state.

## 12. Campaign 260930_M1e5: main study complete (2026-10-01, 20 of 36 runs; β_φ block running)

All 20 runs (β_φ = 0.01, 5 structures × 4 β_s) reached u_y = 0.030 mm (E_min at β_s = 0.015 mm: 0.0284 mm, old 4 h
wall limit), no solver stops. Evaluation: `14_energy_consistent_metrics.py --extra-root <campaign root>` →
`plots/energy_consistent/` (tables `energy_consistent_summary.md`, figures `*_M_comparison.*`, `*_M100000.*`).
Total reaction of both strips at the FIRST peak in N/mm (in brackets: later maximum / first peak):

| β_s [mm] | ρ = 0.3 E_max | ρ = 0.3 E_var | ρ = 0.3 E_min | ρ = 0.6 E_max | ρ = 0.6 E_var |
|---|---|---|---|---|---|
| 0.015 | 161.7 (1.00) | 119.2 (1.00) | 62.1 (1.41) | 252.3 (1.00) | 301.1 (1.08) |
| 0.03 | 139.8 (1.00) | 97.0 (1.00) | 52.6 (1.41) | 213.8 (1.00) | 259.2 (1.09) |
| 0.045 | 127.2 (1.00) | 87.0 (1.00) | 47.7 (1.46) | 193.7 (1.00) | 233.9 (1.06) |
| 0.06 | 117.9 (1.00) | 81.0 (1.00) | 44.5 (1.50) | 180.7 (1.04) | 214.3 (1.13) |

| comparison (first peak) | β_s = 0.015 | 0.03 | 0.045 | 0.06 | W at first peak |
|---|---|---|---|---|---|
| ρ = 0.3: E_var vs E_min | +92 % | +84 % | +83 % | +82 % | +219 … +260 % |
| ρ = 0.3: E_var vs E_max | −26 % | −31 % | −32 % | −31 % | −50 … −57 % |
| ρ = 0.6: E_var vs E_max | +19 % | +21 % | +21 % | +19 % | +42 … +52 % |

- Compared with the published M = 100 results (corrected forces): E_var vs E_min at ρ = 0.3 +50…+78 % → +82…+92 %;
  E_var vs E_max at ρ = 0.3 −5…0 % → −26…−32 % (**changes**); E_var vs E_max at ρ = 0.6 +26…+27 % → +19…+21 %.
- All first peaks decrease monotonically with β_s (−27 % for E_max/E_var, −28 % for E_min from 0.015 to 0.06 mm);
  **Correction 2026-10-02:** per structure −27.1 % (ρ=0.3 E_max), −32.0 % (ρ=0.3 E_var), −28.3 % (E_min), −28.4 % (ρ=0.6 E_max), −28.8 % (ρ=0.6 E_var), i.e. 27–32 %;
  the flat E_min curve of the manuscript was the M = 100 rate artefact.
- Post-peak: every structure keeps carrying load after its main crack event (both ends clamped) and reloads;
  E_min (ρ = 0.3) and E_var (ρ = 0.6) exceed their first peak again before u_y = 0.03 mm (factors 1.4–1.5 and
  1.06–1.13). First peak = strength measure; the full curves to u_y = 0.03 mm are available for a figure.
- Run times (6 MPI tasks, 2 runs in parallel): E_max 0.3–1.5 h, ρ = 0.6 E_var 1.2–2 h, ρ = 0.3 E_var 1–4 h,
  E_min 1–4 h (longest at β_s = 0.015 mm).

**Campaign complete (2026-10-02 04:08): 36 of 36 runs**, 1 wall-limit stop (E_min, β_s = 0.015, at u_y = 0.0284 mm),
no solver stops. Raw results (7.2 GB) in `/Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/campaign_260930_M1e5/`;
the scalar histories, `vol_*.json`, convergence logs and run parameters are copied to `results/campaign_260930_M1e5/`
in this repository (evaluate with `python3 14_energy_consistent_metrics.py --extra-root results/campaign_260930_M1e5`).

β_φ study at M = 10⁵, E_var first peak in N/mm (relative to E_max of the same ρ and β_s):

| ρ | β_s | β_φ = 0.001 | β_φ = 0.01 | β_φ = 0.05 |
|---|---|---|---|---|
| 0.3 | 0.015 | 120.7 (−25 %) | 119.2 (−26 %) | 118.5 (−27 %) |
| 0.3 | 0.03 | 94.7 (−32 %) | 97.0 (−31 %) | 96.3 (−31 %) |
| 0.3 | 0.045 | 82.3 (−35 %) | 87.0 (−32 %) | 86.7 (−32 %) |
| 0.3 | 0.06 | 74.2 (−37 %) | 81.0 (−31 %) | 80.9 (−31 %) |
| 0.6 | 0.015 | 313.4 (+24 %) | 301.1 (+19 %) | 270.6 (+7 %) |
| 0.6 | 0.03 | 266.4 (+25 %) | 259.2 (+21 %) | 233.4 (+9 %) |
| 0.6 | 0.045 | 242.8 (+25 %) | 233.9 (+21 %) | 220.2 (+14 %) |
| 0.6 | 0.06 | 227.0 (+26 %) | 214.3 (+19 %) | 212.4 (+18 %) |

- ρ = 0.3: β_φ = 0.01 and 0.05 agree within 1 %; β_φ = 0.001 is equal at small β_s and up to 8 % lower at
  β_s = 0.06 mm (the finest porosity features are the ones a large β_s smears out).
- ρ = 0.6: systematic ordering β_φ = 0.001 > 0.01 > 0.05; the coarsest gradient is 7–14 % weaker than β_φ = 0.001
  at β_s ≤ 0.045 mm and its margin over E_max drops to +7 … +18 %. The published statement "β_φ affects the peak load
  only mildly" (M = 100, all three within 2–7 %) holds for ρ = 0.3 but not for ρ = 0.6.

## 13. Alternative metrics: maximum load and total work up to u_y = 0.03 mm (2026-10-02, M = 10⁵, β_φ = 0.01)

Computed from `results/campaign_260930_M1e5/` (E_min at β_s = 0.015 ends at 0.0284 mm, so its end-state values are
slightly low). "max load" = global maximum of the total reaction up to the end; "W" = residual-based external work
up to the end; "dissipated" = Π_frac + D_s at the end (W − Π_el).

| comparison | first peak | max load | W to 0.03 mm | dissipated |
|---|---|---|---|---|
| ρ = 0.3: E_var vs E_min | +82 … +92 % | +21 … +37 % | +23 … +36 % | +36 … +96 % |
| ρ = 0.3: E_var vs E_max | −26 … −32 % | −26 … −32 % | −1 … +4 % | −25 … −34 % |
| ρ = 0.6: E_var vs E_max | +19 … +21 % | +28 … +33 % | +30 … +43 % | +11 … +39 % |

- Max load = first peak for E_max and ρ = 0.3 E_var; for E_min it is the reloading maximum at u_y ≈ 0.02–0.03 mm
  (1.4–1.5× first peak, still rising at the end) and for ρ = 0.6 E_var 1.06–1.13× the first peak. So the max-load
  comparisons for exactly these two structures depend on the chosen end displacement and on the clamped supports.
- W to the end is dominated by the stored elastic energy of the residual structure (Π_el 55–75 % of W); E_var keeps
  more elastic energy and dissipates less than E_max at ρ = 0.3, which makes W equal but "dissipated" −30 %.
  Dissipated energy is a damage measure (less = fewer cracks), not a toughness measure.
- β_φ study with max load (relative to E_max): ρ = 0.3 −25 … −37 % (as first peak); ρ = 0.6 +28 … +40 % for all β_φ,
  i.e. the 7–14 % deficit of β_φ = 0.05 seen in the first peak disappears in the max load (it reloads more).
