# Research data for "Brittle fracture in topology-optimized structures"

This archive is the research compendium for the manuscript **"Brittle
fracture in topology-optimized structures"** by Alexander Schlüter, Ján
Pravda, Dustin Roman Jantos, Philipp Junker, and Ralf Müller.

It contains the simulation inputs, source code, cluster job scripts, raw and
processed finite-element results, plotting code, generated manuscript figures,
and manuscript source needed to inspect and reproduce the reported numerical
results.

## Quick orientation

The data used for the current manuscript figures are primarily:

```text
results/new_W_whole_boundary/
```

The corresponding generated plots are:

```text
plots/new_W_whole_boundary/
```

The simulation input meshes and topology-optimization fields are under:

```text
resources/
```

The manuscript source and its curated figure list are under:

```text
68c3b8d0b7dca7b64b8b7a93/
```

## Directory map

| Path | Contents | Role |
|---|---|---|
| `results/new_W_whole_boundary/` | Eight spectral phase-field campaigns, including XDMF/HDF5 fields and scalar histories | **Primary publication data** |
| `plots/new_W_whole_boundary/` | Evaluation plots and field-overview PDFs generated from the primary results | **Publication plot outputs** |
| `resources/` | Meshes, material/topology fields, CSV connectivity, and campaign inputs | Simulation inputs |
| `code/vendor/alex/` | Snapshot of the local `alex` finite-element helper library imported by the scripts | Required source dependency |
| `000_template/` | Templates copied into simulation campaign folders | Simulation source |
| `00_jobs/` and numbered shell scripts | Local/HPC campaign preparation, submission, repair, and evaluation | Workflow source |
| `01_phasefield_dcb_260504_folder.py` | Main monolithic phase-field simulation | Simulation source |
| `04_mesh2dlfxmesh.py` | Converts archived mesh data to DOLFINx mesh files | Preprocessing source |
| `08_plot_phasefield_overview.py` | Creates field overview PDFs from XDMF/HDF5 results | Figure source |
| `09_evaluation_260504_parameter_space.py` | Creates response, energy, and parameter-summary plots | Figure source |
| `10_plot_rho_omega_constraint.py` | Evaluates and plots the material-volume constraint | Figure source |
| `archive_tools/` | Portable reproduction and archive inventory tools | Archive support |
| `68c3b8d0b7dca7b64b8b7a93/` | LaTeX manuscript, bibliography, and curated figure copies | Manuscript record |
| `submission_review_2026-06-11/` | Review-submission snapshot | Archival record |

Other folders and files are retained as provenance or earlier exploratory
work. The only result campaign currently present below `results/` is the
primary `new_W_whole_boundary` campaign.

## Primary result campaigns

`results/new_W_whole_boundary/` contains two related groups.

### Main beta_phi data

The four `simulation_20260526_1229*` folders contain spectral-split
simulations at phase-field regularization lengths:

```text
epsilon = 0.015, 0.030, 0.045, 0.060 mm
```

They contain the constant-minimum, constant-maximum, and spatially varying
Young's modulus cases for the archived combinations of `a`, `rho`, and
`beta_phi` (principally `beta_phi = 0.001` and `0.01`).

### beta_phi = 0.05 comparison data

The four folders containing `CAMPAIGN260526_beta005_var` hold the additional
spatially varying cases at:

```text
beta_phi = 0.05
a = 6
rho = 0.3 and 0.6
epsilon = 0.015, 0.030, 0.045, 0.060 mm
```

For comparisons at `beta_phi = 0.05`, the constant-minimum and
constant-maximum reference cases are shared from the main campaign because
those fields do not vary with `beta_phi`.

Every campaign folder includes its executed simulation and mesh-conversion
source, job script, scheduler output, and `run_parameters.txt`.

## Result file formats

Each result leaf normally contains:

| Pattern | Meaning |
|---|---|
| `mesh.xdmf`, `mesh.h5` | Original input mesh |
| `dlfx_mesh_1.xdmf`, `dlfx_mesh_1.h5` | Mesh converted for DOLFINx |
| `results_*.xdmf`, `results_*.h5` | Time-dependent finite-element fields |
| `result_graphs_*.txt` | Scalar time/load/displacement/energy history |
| `vol_*.json` | Volume and averaged material metadata |
| `convergence_log_*.txt` | Nonlinear solver convergence record |
| `cell_data.csv`, `connectivity.csv`, `node_coords.csv`, `points_data.csv` | Mesh and field source tables |

XDMF files are XML descriptors. They must remain beside their same-basename
HDF5 (`.h5`) companion because the XDMF contains relative HDF5 references.
Open the `.xdmf` file, not the `.h5` file, in ParaView.

The primary publication tree contains 88 simulation cases and, for each case,
one scalar-history file, one result XDMF/HDF5 pair, one volume JSON file, one
input mesh pair, and one converted DOLFINx mesh pair.

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for field and column details.

## Reproduce the manuscript plots

Plot regeneration does not require DOLFINx. It requires Python 3 with NumPy,
Matplotlib, h5py, and a working LaTeX installation because the plotting scripts
use Matplotlib's `text.usetex`.

From the archive root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-postprocessing.txt
bash archive_tools/reproduce_manuscript_plots.sh
```

By default, regenerated files are written to:

```text
reproduced_plots/new_W_whole_boundary/
```

The script does not overwrite the archived publication plots. Set
`OUTPUT_ROOT` to choose another output directory.

The manuscript's source-to-figure mapping is recorded in:

```text
68c3b8d0b7dca7b64b8b7a93/manuscript_picture_list.txt
```

## Re-run the simulations

Full simulation reproduction is substantially more demanding than plot
regeneration. The original calculations used:

- Python 3.10 bytecode is present in the archive and is the best available
  record of the Python generation used.
- DOLFINx, UFL, Basix, PETSc/petsc4py, and MPI/mpi4py.
- NumPy, pandas, SciPy, Matplotlib, meshio, and h5py.
- The archived `code/vendor/alex/` source snapshot.
- An Apptainer image named `alex-dolfinx.sif`.
- SLURM jobs, normally six MPI tasks per case.

The original container binary is **not** included. Exact package version
metadata was not embedded in the scheduler logs, so the archive cannot claim
bitwise reconstruction of the original software environment. The complete
application-level source, executed job scripts, inputs, and outputs are
included.

See `SOFTWARE_ENVIRONMENT.md` for the environment record and known limits.

To make the vendored helper library importable:

```bash
export PYTHONPATH="$PWD/code/vendor:$PYTHONPATH"
```

The original workflow was:

1. Prepare or select mesh leaves below `resources/`.
2. Run `04_mesh2dlfxmesh.py` once for each leaf.
3. Run `01_phasefield_dcb_260504_folder.py` with MPI for the selected split and
   epsilon.
4. Collect the created `results_*`, `result_graphs_*`, and `vol_*` files.
5. Run the postprocessing scripts described above.

The SLURM implementation is preserved in `00_jobs/job_template_260504_sweep.sh`
and in every campaign folder. Site-specific paths and account names in these
historical scripts must be adapted for another cluster.

## Verify and inventory the archive

Create a fresh CSV inventory:

```bash
python3 archive_tools/build_manifest.py
```

Create the inventory and compute SHA-256 checksums for every file:

```bash
python3 archive_tools/build_manifest.py --checksums
```

The checksum operation reads the entire archive and can take some time. The
generated `MANIFEST.csv` and `SHA256SUMS` are suitable for inclusion in the
Zenodo deposit.

## Manuscript

The current manuscript source is:

```text
68c3b8d0b7dca7b64b8b7a93/main.tex
```

Build it from that directory with:

```bash
latexmk -pdf main.tex
```

The manuscript directory also contains the bibliography, journal class, and
curated figure files used by the submitted document.

## Citation and licensing

Citation metadata are provided in `CITATION.cff`. Add the Zenodo DOI after
publication.

Before public deposit, the authors must select and add explicit licenses for:

1. the research data,
2. the original source code, and
3. any third-party manuscript template or other redistributed material.

No license is inferred by this README. See `ZENODO_UPLOAD_CHECKLIST.md`.

## Contact

Alexander Schlüter  
Institute for Mechanics, Technical University of Darmstadt  
`alexander.schlueter@tu-darmstadt.de`
