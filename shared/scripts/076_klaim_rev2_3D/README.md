# 076_klaim_rev2_3D — 3D simulations for revision 2 of the KLAIM 2025 paper

Same workflow and scripts as `../075_klaim_rev2_2D` (read its README first); case list `cases_3D.txt`
(10 jobs × 64 cores, 5000 MB/core, 7 d). Template = `061-plasticity-fracture-noll-3D/000_template` with:

* `--eta` (default **1e-5**, old 3D runs used 1e-3), `--beta` (default **0.2**, old runs library default 0.1),
* yield stress as cell function (linear elastic embedding), `--material elastic` possible,
* `--embedding cubic` (uses C11, C12, C44 of the cubic unit cell; Δ = C11 − C12 − 2C44 on the normal strain components),
* the same instrumented outputs as in 2D (all energies, W, A, J divided by the thickness in z, i.e. per unit thickness
  like in the paper), event-triggered field snapshots before/after crack jumps,
* `run_effective_stiffness.py` stores the full 6×6 matrix and the Zener ratio,
* time step capped at `dt_max = 0.01` (2D value; the old 3D script had no cap).

Meshes are generated with `ronny.mesh` (unchanged scripts). Studies:

| study | cases | purpose |
|---|---|---|
| S1_3D | [vM] w_s ∈ {0.25, 0.375, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0} (8) | Sec. 5 figures with corrected η/β and energy outputs |
| S3_3D | [vM] cubic embedding at w_s = 1.0 and 0.375 (2) | R1-M4 in 3D (Zener ratio of the spherical-pore cell is larger than in 2D) |
| S6_3D | optional elastic sets at w_s = 1.0 (commented out) | only if the queue allows |

Evaluation: `python3 ../075_klaim_rev2_2D/evaluation_rev2.py --results ../075_klaim_rev2_2D/results --results3d results`.
The paper normalises the 3D curves by J_c^num of the vM strip (1.50 J_c^0 for β = 0.1); with the new S2 strip run the
value for β = 0.2 is taken from `evaluation_output/Jc_num_strip.csv`.
