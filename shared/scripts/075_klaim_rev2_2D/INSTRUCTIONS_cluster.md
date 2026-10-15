# Step-by-step: getting the revision-2 simulations onto Lichtenberg (as fast as possible)

All commands on the **cluster login node** are run inside the project folder
`$HOME/dolfinx_alex/shared/scripts/075_klaim_rev2_2D` (or `…/076_klaim_rev2_3D`). `$HPC_SCRATCH` must be set.

## 0. Mac → git → cluster (5 min)

```bash
cd ~/Work/Hypo/Hypo/Simulation/dolfinx_alex/shared          # git root is shared/ (contains utils/alex and scripts/)
rm -rf scripts/046_plasticity/_rev2_staging scripts/046_plasticity/revision2_scratch/*.png   # temporary files from the audit session
git add scripts/075_klaim_rev2_2D scripts/076_klaim_rev2_3D scripts/046_plasticity/CLAUDE.md scripts/046_plasticity/KLAIM_2025/*.md
git status --short | grep -v "^??"                           # make sure no unintended changes (e.g. the manuscript tex) go in unless wanted
git commit -m "KLAIM rev2: 2D/3D rerun studies (075/076), instrumented outputs, cubic embedding, sensitivity, representativity"
git push
# on the cluster
ssh as1vapa@lcluster14.hrz.tu-darmstadt.de
cd $HOME/dolfinx_alex/shared/scripts && git pull
chmod +x 075_klaim_rev2_2D/*.sh 076_klaim_rev2_3D/*.sh
```

## 1. Smoke + scaling test (submit today, ~1–3 h of queue+run)

```bash
cd $HOME/dolfinx_alex/shared/scripts/075_klaim_rev2_2D
./00_scaling_test.sh                       # 3 jobs: S1_vM_ws1.0 truncated at 20 % of the crack path with 16, 32, 48 cores, 6 h limit
cd ../076_klaim_rev2_3D
CORES="64 96" ./00_scaling_test.sh         # 2 jobs: S1_3D_vM_ws1.0 truncated, 64 and 96 cores (96 = one full i01 node)
```

While they queue, check the first job that starts:

```bash
./07_status.sh                             # steps / pseudo-time / x_ct per folder, SLURM state, last error line
tail -f $HPC_SCRATCH/075_klaim_rev2_2D/scaling_S1_vM_ws1.0_n32/*.out.*     # Ctrl-C to leave
```

What must be true after the first few minutes of a run (otherwise stop and send me the `.err` file):
* meshing + effective stiffness finished; `parameters.txt` contains `C11_effective`, `zener_effective`, `beta_simulation=0.2`,
  `eta=1e-05`, `bc_moduli=effective`, `mu_bc_simulation` = the `mue_effective` of the same file, `nu_bc_simulation` ≈ 0.25;
* `run_simulation_graphs.txt` grows and has 25 columns (`run_simulation_graphs_columns.txt` lists them);
* `run_simulation_events.txt` exists (it gets entries at the first crack jump, x_ct ≈ 2.5–3.5 for w_s = 1).

Also check once that the container's `alex` is the same as on the Mac (`cubic_degradation` must exist, it does in the May-2026
copy): `apptainer exec $HOME/dolfinx_alex/alex-dolfinx.sif python3 -c "import alex.phasefield as pf; print(pf.cubic_degradation)"`
run from `$HOME/dolfinx_alex/shared` with the usual `--bind $HOME/dolfinx_alex/shared:/home`.

When the test jobs are finished:

```bash
python3 scaling_report.py                  # wall time, s/step, extrapolated full-run time per core count
```

Decision rule for the production core count (2D): take the smallest core count whose "full run est." is below ~2 days;
if 16 and 32 cores differ by less than 30 % in s/step, use 16 (twice as many jobs fit on the i01 nodes and backfill
is easier). Typical expectation for ~0.5 M dofs: 16–32 cores. For 3D take 96 if s/step(96) < 0.75·s/step(64),
otherwise 64. Then set `PROCESSOR_NUMBER` in `02_create_jobs.sh` (both folders) and set `TIME` to about
1.5 × the longest extrapolated run (w_s = 3.0 and 2.0 have the most steps), rounded up to full days — shorter
limits are scheduled earlier than the old 7-day default.

Regression check of the physics (the only things that changed in S1 vs. v2.0 are β 0.1→0.2, η and the elastic embedding): the truncated test already gives J_x up to the first two pores — compare with the old curve:

```bash
# on the Mac, after downloading the test folder (06 script or scp of parameters.txt + run_simulation_graphs*.txt):
python3 evaluation_rev2.py --results results --legacy ../046_plasticity/KLAIM_2025/submission_ready_20260525/KLAIM_2025_research_data/scripts
# -> evaluation_output/old_vs_new_vM_ws1.0.png
```

## 2. Production submission (as soon as the core count is fixed)

```bash
cd $HOME/dolfinx_alex/shared/scripts/075_klaim_rev2_2D
./01_create_directories.sh                 # 52 folders (existing ones are skipped)
./02_create_jobs.sh                        # job_script.sh everywhere (PROCESSOR_NUMBER / TIME from step 1)
./03_submit_all.sh --dry-run               # sanity check
./03_submit_all.sh --match S2_strip        # 3 calibration runs first (cheapest, needed for every normalisation)
./03_submit_all.sh --match S1_             # 33 paper sweeps
./03_submit_all.sh --match S3_             # cubic embedding
./03_submit_all.sh --match S4_             # sensitivity
./03_submit_all.sh --match S5_             # representativity
cd ../076_klaim_rev2_3D
./01_create_directories.sh && ./02_create_jobs.sh && ./03_submit_all.sh
```

The scheduler decides the order anyway; submitting everything on day 1 maximises throughput. If the account's
job limit bites (`squeue -u $USER` shows many PENDING with reason `QOSMaxJobsPerUserLimit`/`AssocMaxJobsLimit`), it is
still better to have them queued. The submit script skips folders that already have SLURM output, so re-running
`03_submit_all.sh` after fixing something only submits what is missing (delete `*.out.*` in a folder to resubmit it).

Monitoring (daily):

```bash
./07_status.sh                             # or ./07_status.sh --match S1_vM
squeue -u $USER --format "%.10i %.30j %.8T %.10M %.6D %R" | sort -k2
./04_cancel_jobs.sh --dry-run              # lists the project's jobs; without --dry-run cancels them (--match to restrict)
```

A finished run prints `Simulation stopped early: specimen separated completely` and `finished with exit code 0`
in its `.out.*` file; a run killed by the time limit has no such line but its tables are still usable up to the last step.

## 3. Collect, download, evaluate

```bash
# cluster
cd $HOME/dolfinx_alex/shared/scripts/075_klaim_rev2_2D && ./05_prepare_lightweight_results_on_cluster.sh
cd ../076_klaim_rev2_3D && ./05_prepare_lightweight_results_on_cluster.sh
# Mac
cd ~/Work/Hypo/Hypo/Simulation/dolfinx_alex/shared/scripts/075_klaim_rev2_2D && ./06_download_results_lightweight.sh
cd ../076_klaim_rev2_3D && ./06_download_results_lightweight.sh
cd ../075_klaim_rev2_2D && python3 evaluation_rev2.py --results results --results3d ../076_klaim_rev2_3D/results \
      --legacy ../046_plasticity/KLAIM_2025/submission_ready_20260525/KLAIM_2025_research_data/scripts
```

`05_…` can be run at any time (also while jobs are running) to get partial results; add `--with-fields` to include all
h5 files (large).

## 4. Time budget (deadline Nov 1)

| when | what |
|---|---|
| Oct 15–16 | step 0 + 1 (test jobs), fix anything that fails, step 2 |
| Oct 17–24 | jobs run; text work in parallel (homogenisation framing, intro, minors, references) |
| Oct 24–27 | step 3, new figures, numbers into the manuscript, response letter |
| Oct 28–31 | final check by co-authors, submit |

If by Oct 22 the S1 sweeps are not through, reduce: keep S2, S1 at w_s ∈ {0.375, 0.6, 1.0, 2.0, 3.0} for the three sets,
S3 and S4 at w_s = 1.0 — cancel the rest with `./04_cancel_jobs.sh --match <name>`.
