# Software environment

## Postprocessing environment

The figure-generation scripts require:

- Python 3
- NumPy
- Matplotlib
- h5py
- LaTeX, including a Computer Modern-compatible installation

Install the Python packages with:

```bash
python3 -m pip install -r requirements-postprocessing.txt
```

The scripts were validated in this archive by importing their command-line
interfaces and by parsing all 88 primary result records.

## Simulation environment

The simulation and mesh conversion source imports:

- DOLFINx
- UFL
- Basix
- PETSc and petsc4py
- MPI and mpi4py
- NumPy
- pandas
- SciPy
- Matplotlib
- meshio
- the archived `code/vendor/alex/` package

The original SLURM jobs executed an Apptainer image at:

```text
$HOME/dolfinx_alex/alex-dolfinx.sif
```

The image itself is not part of this archive. The historical source tree
contains Python 3.10 bytecode, indicating that Python 3.10 was used for at
least the archived production runs. Exact versions of DOLFINx and its compiled
dependencies were not written to the job logs.

## HPC execution record

The archived production job template requested:

- SLURM
- one node
- six MPI tasks by default
- 4000 MB per CPU by default
- an Apptainer-compatible compute environment

These are historical settings, not minimum hardware requirements. Site-specific
account names, partitions, scratch paths, and bind mounts in the archived job
scripts must be changed for another system.

## Reproducibility level

The archive supports:

- direct inspection of all reported fields and scalar histories,
- regeneration of plots from archived numerical outputs,
- source-level reconstruction of mesh conversion and phase-field solves, and
- comparison of newly computed results with the archived outputs.

Because the original container and exact dependency lockfile are unavailable,
bitwise reproduction of the finite-element runs is not guaranteed. Numerical
differences can also arise from MPI partitioning, PETSc solver versions,
linear-algebra libraries, and nonlinear convergence paths.

