# Mobility side study (campaign `260929_mobility`)

Side study for Reviewer 1 ("what portion of the dissipated energy comes from the
mobility term −ṡ/M, and is this artificial damping acceptable?"). **Not part of the
manuscript.** Nothing in `results/`, `plots/new_W_whole_boundary/`, the Overleaf
folder, the submission snapshots or the Zenodo files was changed.

## Design

| item | value |
|---|---|
| structures | primary: ρ = 0.3, E_var (`beta_0_01_a_6_rho_0_3_var`); optional: ρ = 0.3, E_min (`beta_0_01_a_6_rho_0_3_min`), ρ = 0.6, E_var (`beta_0_01_a_6_rho_0_6_var`); added on request: ρ = 0.3 and 0.6, E_max (`beta_0_01_a_6_rho_0_{3,6}_max`, M = 10000 only) |
| fixed | spectral split, β_s = 0.03 mm (`--epsilon 0.03`), β_φ = 0.01 mm², a = 6, κ_s = 0.001, Δt_max = 0.001 s, v₀ = 1 mm/s, all other parameters as in the publication |
| mobility M [mm³/(Nmm s)] | 100 (reference, publication campaign `simulation_20260526_122937_SPLITspectral_EPS0_03`, not recomputed), 1000, 10000 |
| parallelism | 6 MPI tasks per run (as in the publication campaigns) |

Inputs (`cell_data.csv`, `connectivity.csv`, `mesh.h5/.xdmf`, `node_coords.csv`,
`points_data.csv`, `active_cells_mapping`) are byte-identical to those of the
reference campaign, and `000_template/01_phasefield_dcb_260504_folder.py` was
identical to the script archived in the reference campaign folder before the
change below.

## Code changes

* `000_template/01_phasefield_dcb_260504_folder.py`: new CLI option
  `--mobility VALUE` / `--mobility=VALUE` / `mobility=VALUE` (default 100.0,
  `DEFAULT_MOBILITY`). `Mob` (rate dissipation) and `iMob = 1/Mob` (weak form)
  follow it. New `mobility_output_suffix = _M<label>` (empty for M = 100) is
  appended after `epsilon_output_suffix` in all output names (`results_*.xdmf`,
  `result_graphs_*.txt`, `vol_*.json`, `convergence_log_*.txt`, Newton log).
  The mobility is printed in the `[INFO]` startup line. With default arguments
  file names and numbers are unchanged (parser checked on sample argv lists).
* Campaign tooling (tag `260929_mobility`):
  `01_create_directories_for_260929_mobility_study.sh`,
  `02_create_jobsa_260929_mobility_study.sh` (also fills TIME/MEMORY/PROCESSOR, defaults 10080/4000/6),
  `04_submit_all_260929_mobility_study_jobs.sh`,
  `00_jobs/job_template_260929_mobility.sh` (`{MOBILITY}` → `--mobility`),
  `00_jobs/run_260929_mobility_local.sh` (4th positional argument = M).
* Evaluation: `12_mobility_study_eval.py` (numpy + matplotlib only) →
  `plots/mobility_study/`.

## Commands used

The runs were done locally (DOLFINx 0.7.3 in the Docker image
`dolfinx_alex-alex-dolfinx`, same base image `dolfinx/dolfinx:v0.7.3` as the
cluster Apptainer image; vendored `code/vendor/alex` first on `PYTHONPATH`).
Campaign root on the external disk (outside git):
`/Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/mobility_study_260929/`.

```bash
cd research_data_brittle_fracture_topology_optimized_structures
export HPC_SCRATCH=/Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/mobility_study_260929
bash 01_create_directories_for_260929_mobility_study.sh          # primary case, M = 1000, 10000
LEAVES="beta_phi_0_001-0_01_a_3-6_rho_0_3_min/beta_0_01_a_6_rho_0_3_min
        beta_phi_0_001-0_01_a_3-6_rho_0_3-0_6_var/beta_0_01_a_6_rho_0_6_var" \
  bash 01_create_directories_for_260929_mobility_study.sh         # optional cases
MOBILITIES=10000 LEAVES="beta_ph_0_001-0_01_a_3-6_rho_0_3-0_6_max/beta_0_01_a_6_rho_0_3_max
        beta_ph_0_001-0_01_a_3-6_rho_0_3-0_6_max/beta_0_01_a_6_rho_0_6_max" \
  bash 01_create_directories_for_260929_mobility_study.sh         # E_max cases
bash 02_create_jobsa_260929_mobility_study.sh                    # job scripts (cluster only)

# one local run (per simulation folder SIM and mobility M)
SHARED=$HOME/Work/Hypo/Hypo/Simulation/dolfinx_alex/shared
PROJ=/home/scripts/063-Special-Issue-IJF-Hannover/research_data_brittle_fracture_topology_optimized_structures
docker run --rm --cpus 6 -v "$SHARED":/home -v "$SIM":/work -w /work \
  -e PYTHONPATH=$PROJ/code/vendor:/home/utils:/usr/local/dolfinx-real/lib/python3.10/dist-packages:/usr/local/lib \
  -e PROCESSOR_NUMBER=6 -e PHASEFIELD_SCRIPT=/work/01_phasefield_dcb_260504_folder.py \
  -e MESH_SCRIPT=/work/04_mesh2dlfxmesh.py dolfinx_alex-alex-dolfinx \
  bash $PROJ/00_jobs/run_260929_mobility_local.sh \
       /work/resources/260504_dcb_beta_phi_a_rho_var_min_max spectral 0.03 $M > "$SIM/run_local.log" 2>&1

# evaluation (defaults point to the reference campaign and the study root above)
python3 12_mobility_study_eval.py
```

On the cluster (not used here): set `HPC_SCRATCH`, run 01 and 02, then
`bash 04_submit_all_260929_mobility_study_jobs.sh --dry-run` and without `--dry-run`.

## Results

All nine new runs reached and passed their peak load and ended with the usual
termination criterion (Newton failure, Δt < 1e-14 s). Wall time on 6 cores
(local Docker): 25 min (E_max) to about 1 h (M = 10000, E_min/E_var).
M = 300 / 3000 were therefore not needed.

Figures in `plots/mobility_study/` (PDF and PNG):
* In `09_evaluation` style (same fonts and colors per (ρ, case), line style = M, ✕ = peak load,
  top axis t = u_y/v₀ with v₀ = 1 mm/s, so u_y in mm = t in s):
  `Ry_vs_uy_mobility_all`, `Ry_vs_uy_mobility_<dataset>` (force–displacement),
  `Dissipation_vs_uy_mobility_all` (D_s of all cases) and
  `Dissipation_grid_vs_uy_mobility_<dataset>`, with panels (a) D_s, (b) Π_frac on the same scale,
  (c) dissipation rate Ḋ_s (column 8, log scale) and (d) D_s/W.
* Overview per case: `mobility_<dataset>_<case>`, with panels (a) R_y, (b) W, (c) Π_el, Π_frac, D_s,
  (d) D_s/W and D_s/(Π_frac+D_s).
* Table: `mobility_study_summary.csv/.md`.

W = column 13 (total-boundary work, the manuscript W); `W_strips` = column 4 (two load
strips), which is the quantity behind `max_Work` in `summary_260504.csv`. M in mm³/(Nmm s).

| dataset | case | M | max R_y [N/mm] | u_y at peak [mm] | W at peak [Nmm/mm] | W_strips at peak | Pi_frac at peak | D_s at peak | D_s/W peak | D_s/(Pi_frac+D_s) peak | D_s/W final | D_s/(Pi_frac+D_s) final | steps | final u_y [mm] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| beta_0_01_a_6_rho_0_3_max | max | 100 | 47.285 | 0.015875 | 1.1532 | 0.8198 | 0.0444 | 0.0060 | 0.0052 | 0.119 | 0.550 | 0.756 | 284 | 0.019419 |
| beta_0_01_a_6_rho_0_3_max | max | 10000 | 44.195 | 0.014594 | 0.9764 | 0.6955 | 0.0324 | 0.0000 | 0.0001 | 0.002 | 0.472 | 0.706 | 415 | 0.019425 |
| beta_0_01_a_6_rho_0_3_min | min | 100 | 29.710 | 0.008750 | 0.4007 | 0.2879 | 0.0094 | 0.0187 | 0.0467 | 0.667 | 0.109 | 0.673 | 534 | 0.029573 |
| beta_0_01_a_6_rho_0_3_min | min | 1000 | 20.929 | 0.006063 | 0.1915 | 0.1384 | 0.0074 | 0.0023 | 0.0122 | 0.240 | 0.075 | 0.486 | 811 | 0.024229 |
| beta_0_01_a_6_rho_0_3_min | min | 10000 | 18.584 | 0.005313 | 0.1471 | 0.1068 | 0.0062 | 0.0003 | 0.0017 | 0.039 | 0.051 | 0.361 | 1038 | 0.022728 |
| beta_0_01_a_6_rho_0_3_var | vary | 100 | 45.256 | 0.014375 | 1.0303 | 0.7352 | 0.0380 | 0.0411 | 0.0399 | 0.520 | 0.311 | 0.692 | 511 | 0.029584 |
| beta_0_01_a_6_rho_0_3_var | vary | 1000 | 33.732 | 0.010375 | 0.5410 | 0.3871 | 0.0269 | 0.0053 | 0.0099 | 0.166 | 0.182 | 0.471 | 670 | 0.025384 |
| beta_0_01_a_6_rho_0_3_var | vary | 10000 | 31.350 | 0.009469 | 0.4527 | 0.3241 | 0.0206 | 0.0005 | 0.0011 | 0.024 | 0.138 | 0.388 | 1150 | 0.025453 |
| beta_0_01_a_6_rho_0_6_max | max | 100 | 85.570 | 0.013750 | 1.5828 | 1.2471 | 0.0670 | 0.0100 | 0.0063 | 0.130 | 0.408 | 0.680 | 179 | 0.015678 |
| beta_0_01_a_6_rho_0_6_max | max | 10000 | 79.576 | 0.012625 | 1.3366 | 1.0554 | 0.0542 | 0.0001 | 0.0001 | 0.002 | 0.369 | 0.644 | 299 | 0.015679 |
| beta_0_01_a_6_rho_0_6_var | vary | 100 | 89.431 | 0.017313 | 2.9765 | 1.7002 | 0.1245 | 0.0405 | 0.0136 | 0.245 | 0.163 | 0.556 | 118 | 0.020344 |
| beta_0_01_a_6_rho_0_6_var | vary | 1000 | 84.812 | 0.016719 | 2.7625 | 1.5676 | 0.1664 | 0.0108 | 0.0039 | 0.061 | 0.076 | 0.345 | 355 | 0.031680 |
| beta_0_01_a_6_rho_0_6_var | vary | 10000 | 81.340 | 0.015594 | 2.4045 | 1.3774 | 0.1264 | 0.0010 | 0.0004 | 0.008 | 0.058 | 0.475 | 358 | 0.016081 |

### Peak load vs M

| case | M = 1000 | M = 10000 | 1000 → 10000 | M = 100 above M = 10000 |
|---|---|---|---|---|
| ρ = 0.3 E_max | – | −6.5 % | – | +7 % |
| ρ = 0.3 E_var | −25.5 % | −30.7 % | −7.1 % | +44 % |
| ρ = 0.3 E_min | −29.6 % | −37.4 % | −11.2 % | +60 % |
| ρ = 0.6 E_max | – | −7.0 % | – | +8 % |
| ρ = 0.6 E_var | −5.2 % | −9.0 % | −4.1 % | +10 % |

### Consequences for the comparisons in the paper (β_s = 0.03 mm, β_φ = 0.01 mm²)

| comparison | max R_y, M = 100 | max R_y, M = 10000 | W at peak (col. 13), M = 100 | W at peak (col. 13), M = 10000 |
|---|---|---|---|---|
| ρ = 0.3: E_var vs E_min | +52 % | +69 % | +157 % | +208 % |
| ρ = 0.3: E_var vs E_max | −4 % | **−29 %** | −11 % | **−54 %** |
| ρ = 0.6: E_var vs E_max | +4.5 % | +2.2 % | +88 % | +80 % |

(With the two-strip work, column 4, the W-at-peak comparisons are ρ = 0.3 E_var vs E_max
−10 % → −53 %, and ρ = 0.6 E_var vs E_max +36 % → +31 %.)

### Energy shares at the peak (M = 100 → 10000)

Π_el is 85–104 % of W, Π_frac 2–6 %. D_s/W is 0.5 → 0.01 % (E_max ρ = 0.3), 4.7 → 0.2 % (E_min),
4.0 → 0.1 % (E_var ρ = 0.3), 0.6 → 0.01 % (E_max ρ = 0.6) and 1.4 → 0.04 % (E_var ρ = 0.6).
D_s/Π_el is almost identical to D_s/W. At M = 100, D_s at the peak is comparable to Π_frac
(D_s/(Π_frac+D_s) = 0.12–0.67).

### Caveats

* **Final-state values** depend on where each run stops (solver failure), not on a
  common end point. ρ = 0.6 E_var at M = 10000 stopped at t = 0.0161 s inside the first
  load drop (R_y = 0.87 max R_y), so its final-state ratios are not comparable. The E_max
  runs stop at the same u_y for M = 100 and 10000. Peak values are unaffected.
* **Energy balance after the peak.** (Π_el + Π_frac + D_s)/W (column 13) is 0.90–1.09 at the
  peak for all runs, but only 0.44–0.58 at the final state for E_min (W keeps growing after
  cracking), so post-peak ratios relative to W are unreliable there; use D_s/Π_el or D_s/Π_frac.
  The same gap is visible in the published `Response_energy_grid…` figure, panel (d).
* **The two-strip work (column 4) is 30–60 % below Π_el even before the peak.** It equals
  ∫R_y du, but it misses part of the work entering the body. It is the quantity behind
  `max_Work` in `summary_260504.csv` and the work-at-peak plots of `09_evaluation`. Check which W
  the manuscript's work-at-peak statements use.
* An estimate of the phase-field relaxation time is τ ≈ β_s/(M G_c). With M = 100 and
  G_c = 0.1 (E_min, porous regions of E_var) this gives τ ≈ 3·10⁻³ s, a noticeable fraction of the
  time to peak (≈ 0.009–0.017 s); with G_c = 1 (E_max) it is 3·10⁻⁴ s. This fits the observed
  ordering of the sensitivity: E_min > E_var(ρ = 0.3) > E_var(ρ = 0.6) ≈ E_max.
* Without inertia only M/v₀ matters: increasing M by a factor k is equivalent to loading k
  times more slowly.

### Interpretation

**In three sentences:** max R_y does change with M. From M = 100 to 10000 it drops by 31 %
(ρ = 0.3 E_var) and 37 % (ρ = 0.3 E_min), but only 6.5–9 % for E_max and ρ = 0.6 E_var, because the
rate term mainly strengthens the low-G_c (porous) regions. D_s/W at peak falls roughly ∝ 1/M only
between M = 1000 and 10000 (factor 7–10), to ≤ 0.2 % of W, while the post-peak D_s stays of the order
of Π_frac for all M because it is the energy released in unstable crack growth. M = 100 is therefore
not in the quasi-static limit for the peak load of the porous structures: E_var > E_min at ρ = 0.3
survives and grows (+52 % → +69 %), and E_var ≈ E_max at ρ = 0.6 survives (+4.5 % → +2.2 %), but
**E_var at ρ = 0.3 drops from −4 % to −29 % relative to E_max** in max R_y (−11 % → −54 % in W at peak).

Details:

1. **max R_y is not M-independent.** The pre-peak stiffness is unchanged (ρ = 0.3 E_var:
   R_y within 3.5 % up to u_y = 0.009 mm), the peak moves to smaller u_y, and the post-peak
   residual branches of M = 1000 and 10000 nearly coincide.
2. **D_s/W at peak falls roughly ∝ 1/M only above M = 1000.** It drops by a factor of
   3.5–4 from M = 100 to 1000 (less than 10, because at M = 100 the damage evolution is
   itself rate-limited and the peak shifts) and by a factor of 7–10 from M = 1000 to
   10000 (0.01–0.17 % of W at M = 10000). After the peak, D_s stays of the order of Π_frac
   for every M (final D_s/W still 37–47 % for E_max at M = 10000): it is then the energy
   released in unstable crack growth (snap-through under displacement control), not an
   artefact that vanishes with M.
3. **M = 100 is not in the quasi-static limit for the peak load of the porous structures.**
   A rough three-point extrapolation (geometric convergence) gives M → ∞ limits of
   ≈ 30.7 N/mm (ρ = 0.3 E_var) and ≈ 17.7 N/mm (ρ = 0.3 E_min), i.e. about 2 % and 5 % below
   the M = 10000 values. For ρ = 0.6 E_var convergence in M is not yet evident (steps −4.6
   and −3.5 N/mm), so its margin over E_max (+2.2 %) is not yet settled.

Suggested follow-ups (not done): ρ = 0.6 E_var at M = 100000 up to just past the peak
(convergence of the ρ = 0.6 E_var vs E_max comparison); if the campaigns are rerun, M = 10000
(or equivalently a 100× slower loading rate) for all cases and β_s values.
