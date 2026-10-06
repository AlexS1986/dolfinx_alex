# Claude Code prompt — mobility side study for Paper A03 (phase-field fracture)

Copy everything below the line into Claude Code, started in
`~/Work/Hypo/Hypo/Simulation/dolfinx_alex/shared/scripts/063-Special-Issue-IJF-Hannover/research_data_brittle_fracture_topology_optimized_structures`.

---

## Task

Set up, run and evaluate a small **mobility side study** for the phase-field fracture simulations
of the manuscript "Brittle fracture in topology-optimized structures" (folder
`research_data_brittle_fracture_topology_optimized_structures`, read its `CLAUDE.md` in the parent
folder first). The study is *not* part of the manuscript yet. Do not edit anything in
`68c3b8d0b7dca7b64b8b7a93/` (Overleaf), `results/`, `plots/`, `submission_review_*`,
`MANIFEST.csv`, `SHA256SUMS` or the Zenodo files. Everything new goes into new files or a new
side folder.

### Why

Reviewer 1 asked what portion of the dissipated energy comes from the mobility (rate) term
$-\dot s/M$ of the Ginzburg–Landau evolution equation and whether this "artificial damping" is
acceptable. From the existing `result_graphs_*.txt` histories (β_s = 0.03 mm, β_φ = 0.01 mm²,
spectral split, a = 6, M = 100 mm³/(Nmm s)) we already know:

| case | D_s/W at peak load | D_s/Π_frac at peak | D_s/Π_frac at final state | D_s/W at final state |
|---|---|---|---|---|
| ρ=0.3 E_max | 0.5 % | 0.13 | 3.1 | 55 % |
| ρ=0.3 E_min | 4.7 % | 2.0 | 2.1 | 11 % |
| ρ=0.3 E_var | 4.0 % | 1.08 | 2.3 | 31 % |
| ρ=0.6 E_max | 0.6 % | 0.15 | 2.1 | 41 % |
| ρ=0.6 E_var | 1.4 % | 0.33 | 1.3 | 16 % |

So D_s is small relative to the external work W up to the peak load, but it is of the order of
(and after the peak larger than) the fracture energy Π_frac. The side study must show how the
peak-load measures (max R_y, W at peak) and the dissipation ratios change when M is increased
by one and two orders of magnitude, i.e. how far the M = 100 results are from the quasi-static
limit M → ∞.

### Study design

- Structures / cases (input leaf folders below
  `resources/260504_dcb_beta_phi_a_rho_var_min_max/`):
  1. **primary:** `beta_phi_0_001-0_01_a_3-6_rho_0_3-0_6_var/beta_0_01_a_6_rho_0_3_var` (ρ = 0.3, E_var)
  2. optional, only if 1. runs through: `beta_phi_0_001-0_01_a_3-6_rho_0_3_min/beta_0_01_a_6_rho_0_3_min`
     (ρ = 0.3, E_min) and `beta_phi_0_001-0_01_a_3-6_rho_0_3-0_6_var/beta_0_01_a_6_rho_0_6_var` (ρ = 0.6, E_var)
- Fixed: split `spectral`, `--epsilon 0.03` (β_s = 0.03 mm), all other parameters unchanged.
- Mobility values: **M ∈ {1000, 10000}** (reference M = 100 already exists, see below).
  If M = 10000 does not converge past the peak load, add M = 300 and M = 3000 so that the trend
  is documented with at least two values above 100.
- Each run: 6 MPI tasks as in the publication campaigns (`PROCESSOR_NUMBER=6`).

Reference M = 100 histories for comparison (do not recompute, do not copy the large `.h5` files):
```
/Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/results/new_W_whole_boundary/simulation_20260526_122937_SPLITspectral_EPS0_03/resources/260504_dcb_beta_phi_a_rho_var_min_max/beta_phi_0_001-0_01_a_3-6_rho_0_3-0_6_var/beta_0_01_a_6_rho_0_3_var/result_graphs_beta_0_01_a_6_rho_0_3_var_vary_spectral_eps0_03.txt
/Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/results/new_W_whole_boundary/simulation_20260526_122937_SPLITspectral_EPS0_03/resources/260504_dcb_beta_phi_a_rho_var_min_max/beta_phi_0_001-0_01_a_3-6_rho_0_3_min/beta_0_01_a_6_rho_0_3_min/result_graphs_beta_0_01_a_6_rho_0_3_min_min_spectral_eps0_03.txt
/Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/results/new_W_whole_boundary/simulation_20260526_122937_SPLITspectral_EPS0_03/resources/260504_dcb_beta_phi_a_rho_var_min_max/beta_phi_0_001-0_01_a_3-6_rho_0_3-0_6_var/beta_0_01_a_6_rho_0_6_var/result_graphs_beta_0_01_a_6_rho_0_6_var_vary_spectral_eps0_03.txt
```

### Facts about the code you will touch

- The simulation script used by the campaigns is **`000_template/01_phasefield_dcb_260504_folder.py`**
  (the copy in the project root is an older version without the dissipation bookkeeping — leave it alone).
  In the template copy the mobility is hard-coded:
  `Mob = dlfx.fem.Constant(domain, 100.0)` followed by `iMob = dlfx.fem.Constant(domain, 1.0 / Mob.value)`.
  `iMob` enters the weak form, `Mob` enters `get_phasefield_rate_dissipation`. Both must follow the new value.
- The script has a hand-written token parser `parse_args(sys.argv)` that already understands
  `--epsilon VALUE`, `--epsilon=VALUE` and `epsilon=VALUE`. Add `--mobility VALUE`, `--mobility=VALUE`
  and `mobility=VALUE` in the same style. Default 100.0.
- Output file names carry `epsilon_output_suffix` (`_eps0_03` etc., empty for the default). Add a
  `mobility_output_suffix` = `_M<label>` (use the existing `safe_float_label`, e.g. `_M1000`) that is
  **empty for the default M = 100**, and append it wherever `epsilon_output_suffix` is used
  (results_*.xdmf, result_graphs_*.txt, vol_*.json, convergence_log_*.txt). This keeps all
  existing file names unchanged. Print the mobility in the `[INFO]` startup line.
- `result_graphs_*.txt` columns (0-based): 0 t, 1 u_y_top, 2 R_y_top_left, 3 dW, 4 W (two strips),
  5 Π_frac, 6 Π_el, 7 R_y_top_right, 8 ṡ-rate dissipation, 9 D_s (accumulated), 10/11 legacy work,
  12 dW total boundary, 13 **W total boundary** (this is the W used in the manuscript).
  Peak load = index of max |column 2|.
- Campaign tooling to mirror (read them first): `01_create_directories_for_260526_beta005_var_sweep.sh`,
  `02_create_jobsa_260526_beta005_var_sweep.sh`, `04_submit_all_260526_beta005_var_jobs.sh`,
  `00_jobs/job_template_260504_sweep.sh` (SLURM, Apptainer image `$HOME/dolfinx_alex/alex-dolfinx.sif`,
  passes `auto "$split" --epsilon "$epsilon"` to the script), and `00_jobs/run_260504_sweep_local.sh`
  (local `mpirun` runner). `HPC_SCRATCH` must be set for the cluster scripts. The job template loops
  over **every** `mesh.xdmf` leaf below the input root, so copy only the leaf folders listed above
  into the campaign input root.
- Evaluation conventions: `09_evaluation_260504_parameter_space.py` (matplotlib, `text.usetex`,
  colors per (ρ, case) in `ENERGY_COLORS`/`energy_color_for_record`, peak marker "X"). Its file-name
  regex does not know an `_M` suffix, so do not feed the new files into it. Write a separate small
  evaluation script instead.

### Execution environment (mandatory)

**DOLFINx is never installed natively on the Mac or on the cluster. Every DOLFINx invocation —
mesh conversion (`04_mesh2dlfxmesh.py`) and the phase-field solve
(`01_phasefield_dcb_260504_folder.py`) — must run inside the DOLFINx container.** Do not
`pip install`/`conda install` DOLFINx, do not try to run the scripts with the system `python3`.

- Cluster: Apptainer image `$HOME/dolfinx_alex/alex-dolfinx.sif`, invoked exactly as in
  `00_jobs/job_template_260504_sweep.sh`:
  `srun -n N apptainer exec --bind "$HOME/dolfinx_alex/shared:/home,$working_directory:/work" "$container" python3 /work/<script> …`
  (the bind of `dolfinx_alex/shared` to `/home` is what makes the vendored `alex` package importable).
- Locally on this Mac: find the local DOLFINx container first (`docker images | grep -i dolfinx`,
  `docker ps -a`, `ls ~/dolfinx_alex`, or an Apptainer/Singularity image if present) and **ask Alex
  which container/image to use if it is not obvious**. Then run the local runner with
  `PYTHON_BIN`/`MPIEXEC_BIN` pointing into the container, e.g. a wrapper along the lines of
  `docker run --rm -v "$HOME/dolfinx_alex/shared:/home" -v "$PWD:/work" -w /work <image> mpirun -n 6 python3 …`,
  mirroring the two bind mounts of the cluster job. Verify the setup with a mesh conversion of one
  leaf folder before starting a solve.
- If no container is reachable from this machine, stop at "campaign folders prepared" and print
  the cluster commands. Never fall back to a native Python environment.

### Steps

1. **Script change** in `000_template/01_phasefield_dcb_260504_folder.py` as described above.
   Keep the default behaviour byte-for-byte (same file names, same numbers) — verify by a dry
   `python3 -c` import test or by running the parser on sample argv lists.
2. **Campaign scripts** with tag `260929_mobility`:
   `01_create_directories_for_260929_mobility_study.sh` (one simulation folder per (split, epsilon,
   mobility, leaf), copying only the required leaf folders, writing `mobility=` into
   `run_parameters.txt`), `02_create_jobsa_260929_mobility_study.sh`,
   `04_submit_all_260929_mobility_study_jobs.sh`, and a job template
   `00_jobs/job_template_260929_mobility.sh` derived from the 260504 template with a `{MOBILITY}`
   placeholder that is passed as `--mobility`. Also add a `MOBILITY` argument to a copy of the local
   runner (`00_jobs/run_260929_mobility_local.sh`).
3. **Decide where to run** (see "Execution environment" above — container only). If a local
   DOLFINx container is available, run the primary case locally with the container-wrapped local
   runner, M = 1000 first. Otherwise prepare the campaign folders in `$HPC_SCRATCH` and print the
   exact `sbatch`/`rsync` commands (or run them if you have cluster access from this session) —
   ask before submitting anything to the cluster.
   Runs terminate by themselves when the Newton solver fails (Δt < 1e-14 s). Expect the M = 100
   reference to have taken 118–534 pseudo-time steps per case.
4. **Evaluation**: write `12_mobility_study_eval.py` (numpy + matplotlib, no h5py needed) that takes
   the reference M = 100 `result_graphs` files and the new ones and produces, per case, a figure with
   (a) R_y vs u_y, (b) W vs u_y, (c) Π_el, Π_frac and D_s vs u_y, (d) D_s/W and D_s/(Π_frac+D_s) vs u_y,
   one line style per M, peak-load markers, plus a CSV/markdown table with, per (case, M):
   max R_y, u_y at peak, W at peak, Π_frac at peak, D_s at peak, D_s/W at peak,
   D_s/(Π_frac+D_s) at peak, the same two ratios at the final state, number of steps, final u_y.
   Output to `plots/mobility_study/`.
5. **Write-up**: `MOBILITY_STUDY_README.md` in the project root with the design, the commands used,
   the table, and a three-sentence interpretation (does max R_y change with M? does D_s/W at peak
   fall roughly ∝ 1/M? where does M = 100 stand relative to the quasi-static limit?). Add a short
   entry to `CLAUDE.md` (Status section) with the outcome.

### Rules

- Nothing in this study may change the published campaign results or the manuscript files.
- Never copy the large `.h5`/`.xdmf` result files into the git-tracked folders.
- Language of all files and notes: English.
- If M = 1000 fails to reach the peak load, report immediately instead of tuning other parameters
  (Δt, κ_s, β_s must stay as in the publication).
