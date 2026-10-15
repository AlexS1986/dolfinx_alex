# 075_klaim_rev2_2D — all 2D simulations for revision 2 of the KLAIM 2025 paper

One template, one case list, one job per line. Sister folder for 3D: `076_klaim_rev2_3D` (same workflow).
Background, reviewer mapping and decisions: `../046_plasticity/CLAUDE.md` and `../046_plasticity/KLAIM_2025/revision2_plan.md`.

## What is different from the v2.0 runs (046 / 045 / 062)

| item | v2.0 runs | here | why |
|---|---|---|---|
| degradation function | `pf.degrad_cubic` (library default β = 0.1) | `pf.cubic_degradation(beta=0.2)` | manuscript states β = 0.2; 1-D bar (050) already used 0.2 |
| residual stiffness η | 1e-5 (2D), 1e-3 (3D) | 1e-5 everywhere | manuscript states 1e-5 |
| yield stress | global constant (embedding could yield) | cell function, embedding `sig_y = 1e6` (linear elastic) | manuscript: embedding is linear elastic with E_eff, ν_eff |
| elastic sets [Eq J_c], [Eq σ*] | separate elastic script (`StaticPhaseFieldProblem2D`, default quadrature) | same vM script with `--material elastic` (yield stress → ∞, no plastic update) | one formulation for all sets; same quadrature/measure; identical outputs |
| embedding stiffness | isotropic (λ_eff = C12, µ_eff = C33) | isotropic (default) **or** full cubic plane-strain stiffness `--embedding cubic` | R1 major point 4 |
| surfing BC moduli | (E_eff, ν_eff) of the embedding (code) vs. "E_bc = 2.5 µ0, ν_bc = 0.25" (text) | unchanged: `--bc_moduli effective` (final decision 2026-10-15); manuscript Secs. 4.2/5.1 corrected to E_bc = E^eff, ν_bc = ν^eff; `--bc_moduli fixed` available | continuity with the v2.0 runs; physically the K-field of the embedding medium |
| outputs | J, x_ct, W, A, E_el (+E_pl) | + fracture energy, stored plastic energy, viscous dissipation D_visc = ∫∫ṡ²/M, plastic work ∫g σ_Y α̇, energies split porous/embedding, max σ_vM / σ_h / σ_yy / α in the next intact ligament, Newton iterations | R3-C1, R1-M3 energy partition |
| field output | every 400 steps | every 400 steps **plus** the states immediately before and after every crack jump (`run_simulation_events.xdmf/.h5`, list in `run_simulation_events.txt`) | stress state at re-nucleation (R3-C1) |
| effective stiffness | only λ_eff, µ_eff stored | full matrix + Zener ratio in `parameters.txt` and `effective_stiffness_matrix.txt` | R1-M4 |
| crack-surface density | code: J_c[(1−s)²/(4ε) + ε|∇s|²] | unchanged | manuscript eq. (18) has to be corrected (factor 2), see corrections list |

Energy balance of the model (exact, see CLAUDE.md §6): `W = E_el + E_pl + E_frac + D_visc` (all relative to the
state after the initial relaxation of s). `D_visc` is the energy dissipated by the Allen–Cahn viscosity, i.e. the
energy released in unstable crack jumps (kinetic energy in a dynamic model). `D_pl_deg = Σ ∫ g(s) Δψ_pl` is the
plastic work done by the actual (degraded) stress and `E_pl` the part of it that is still "stored" as crack driving
force; their difference is plastic work in material that was fractured later.

## Studies in `cases_2D.list` (52 jobs × 32 cores, 7 d limit)

| study | cases | purpose |
|---|---|---|
| S1 | [vM], [Eq J_c], [Eq σ*] × w_s ∈ {0.25, 0.375, 0.4, 0.5, 0.6, 0.75, 1.0, 1.4, 2.0, 3.0, 4.0} (33) | all paper figures of Sec. 4 with corrected parameters and full energy outputs |
| S2 | homogeneous strip (no pores) for [vM], [Eq J_c], [Eq σ*] (3) | J_c^num calibration (Fig. 3) with β = 0.2 — needed because J_c^num changes with β |
| S3 | [vM] with cubic embedding, w_s ∈ {0.375, 1.0, 4.0} (3) | R1-M4: influence of the isotropic approximation |
| S4 | [vM] σ_y ∈ {0.75, 1.5}, H ∈ {0.11, 0.44} µ0 at w_s ∈ {0.375, 1.0} (8) | R1-M3: sensitivity study (also announced in the conclusion) |
| S5 | initial crack tip at 1.5 w_c and 2.0 w_c; 5 instead of 3 pore rows ([vM] w_s 1.0 and 0.375, [Eq σ*] w_s 1.0) (5) | R1-M2 / round-1 point 7: representativity probes |

Column meaning of `run_simulation_graphs.txt` is written to `run_simulation_graphs_columns.txt` in every run folder:
`t Jx Jy x_ct x_bc Rx_top Ry_top dW W A dt E_el E_el_micro E_pl E_pl_undeg E_frac E_frac_micro D_visc D_pl_deg D_pl_undeg sig_vM_max_lig sig_h_max_lig sig_yy_max_lig alpha_max_lig newton_iters`.

## Workflow

```bash
# 0. local smoke test (needs the dolfinx container with alex on the python path), a few minutes:
STOP_FRACTION=0.25 PROCESSOR_NUMBER=4 WSTEG=1.0 ./run_one_local.sh            # vM
STOP_FRACTION=0.25 PROCESSOR_NUMBER=4 WSTEG=1.0 EMBEDDING=cubic ./run_one_local.sh
STOP_FRACTION=0.25 PROCESSOR_NUMBER=4 WSTEG=1.0 MATERIAL=elastic GC=0.5679 ./run_one_local.sh
STOP_FRACTION=0.25 PROCESSOR_NUMBER=4 GEOMETRY=strip ./run_one_local.sh

# 1. push this folder to the cluster:  $HOME/dolfinx_alex/shared/scripts/075_klaim_rev2_2D
# 2. on the login node, inside the folder:
./01_create_directories.sh                 # $HPC_SCRATCH/075_klaim_rev2_2D/<case>/ + case.env   (--match S1_vM to restrict)
./02_create_jobs.sh                        # writes job_script.sh (global parameters at the top of this script)
./03_submit_all.sh --dry-run               # check, then
./03_submit_all.sh                         # or --match S1_ / --max 10 to stage the submission
./04_cancel_jobs.sh --dry-run              # if something is wrong
# 3. after completion:
./05_prepare_lightweight_results_on_cluster.sh          # tar.gz with tables, logs, event snapshots, w_s=1.0 fields
# 4. on the Mac:
./06_download_results_lightweight.sh                     # -> ./results/<case>/...
python3 evaluation_rev2.py --results results --results3d ../076_klaim_rev2_3D/results \
        --legacy ../046_plasticity/KLAIM_2025/submission_ready_20260525/KLAIM_2025_research_data/scripts
```

Suggested submission order if the queue is limited: S2 (3 strip runs, fast) and S1 w_s = 1.0 / 0.375 first (regression
check against the old curves via `evaluation_rev2.py --legacy`), then the rest of S1, then S3, S4, S5.

`evaluation_rev2.py` writes to `evaluation_output/`: `master_table.csv`, `Jeff_vs_wsteg.png`, `energy_fractions_vs_wsteg.png`,
`energy_partition_<case>.png`, `ligament_stress_<case>.png`, `events.csv` (energies per crack jump), `cubic_vs_isotropic.csv`,
`sensitivity.csv` + `sensitivity_sigy_H.png`, `representativity.csv`, `effective_stiffness_table.csv`, `Jc_num_strip.csv`,
`old_vs_new_*.png`. Paper-style figures (pgf, LaTeX labels) are produced afterwards from these tables with the old
`evaluation.py` conventions.

## Things to check in the first completed run

* `run_simulation_graphs_columns.txt` exists and the table has 25 columns.
* `parameters.txt` contains `C11_effective … zener_effective`, `beta_simulation=0.2`, `eta=1e-05`, `sig_y_embedding_simulation=1000000.0`.
* `run_simulation_events.txt` lists pre/post-jump pairs; `run_simulation_events.h5` is not empty.
* Energy balance residual (`evaluation_output/energy_partition_<case>.png`, grey dotted) is small compared with W.
* The [vM] J_x(x_ct) curve for w_s = 1.0 is close to the v2.0 curve (β 0.1 → 0.2 changes the numbers slightly; a
  qualitative change would indicate a setup error).
