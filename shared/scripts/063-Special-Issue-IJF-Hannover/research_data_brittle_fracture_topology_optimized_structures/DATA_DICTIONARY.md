# Data dictionary

## Naming parameters

Result filenames encode the principal simulation parameters:

| Token | Meaning |
|---|---|
| `beta_0_01` | Porosity-gradient parameter `beta_phi = 0.01` |
| `a_6` | Geometry/problem parameter `a = 6` |
| `rho_0_3` | Prescribed relative material amount `rho = 0.3` |
| `min` | Spatially constant minimum Young's modulus |
| `max` | Spatially constant maximum Young's modulus |
| `var` / `vary` | Spatially varying optimized Young's modulus |
| `spectral` | Spectral tension/compression energy split |
| `eps0_03` | Phase-field regularization length `epsilon = 0.03 mm` |

For the `epsilon = 0.015` campaign, older output names may omit the epsilon
suffix. The enclosing `simulation_*_EPS0_015` folder supplies that metadata.

## Scalar history files

`result_graphs_*.txt` is whitespace-delimited text. Comment lines begin with
`#`. The simulation writer records 14 numeric columns:

| Column (0-based) | Quantity used in this archive |
|---:|---|
| 0 | Simulation/load time parameter |
| 1 | Prescribed vertical displacement `u_y` |
| 2 | Vertical reaction force `R_y` |
| 3 | Displacement increment |
| 4 | External work `W` |
| 5 | Fracture energy |
| 6 | Elastic energy |
| 7 | Additional reaction/traction diagnostic |
| 8 | Crack-surface or dissipation diagnostic |
| 9 | Dissipation increment used by diagnostic scripts |
| 10 | Incremental work diagnostic |
| 11 | Accumulated work diagnostic |
| 12 | Boundary-work diagnostic |
| 13 | Accumulated boundary work used by energy checks |

Columns 1, 2, 4, 5, and 6 are the principal columns consumed by
`09_evaluation_260504_parameter_space.py`. The remaining columns are retained
for provenance and energy-balance diagnostics.

## Volume metadata

`vol_*.json` contains:

| Key | Meaning |
|---|---|
| `vol` | Integrated solid/design-domain volume or area measure |
| `E_average` | Domain-averaged Young's modulus |
| `porosity_average` | Domain-averaged porosity |

## XDMF/HDF5 result fields

`results_*.xdmf` describes datasets stored in the same-basename `.h5` file.
The archive plotting code supports these fields:

| Field | Meaning |
|---|---|
| `u` | Displacement vector |
| `s` | Phase-field/damage variable |
| `E` | Young's modulus field |
| `gc` | Fracture-toughness field |
| `sig` | Stress tensor |
| `sigma_c` | Derived critical stress |
| `sig_vol` | Volumetric stress contribution |
| `sig_dev` | Deviatoric stress contribution |

The available times are represented by temporal grids in XDMF and by keys
below `/Function/<field>/` in HDF5. Mesh geometry and triangle topology are
stored below `/Mesh/mesh/geometry` and `/Mesh/mesh/topology`.

## Mesh source tables

| File | Description |
|---|---|
| `node_coords.csv` | Mesh-node coordinates |
| `connectivity.csv` | Element connectivity |
| `points_data.csv` | Point-associated source fields |
| `cell_data.csv` | Cell-associated source fields |
| `active_cells_mapping` | Mapping used when selecting active mesh cells |

`04_mesh2dlfxmesh.py` converts these inputs and `mesh.xdmf/.h5` into the
`dlfx_mesh_1.xdmf/.h5` pair consumed by the simulation.

## Logs and provenance

| File | Description |
|---|---|
| `run_parameters.txt` | Split, epsilon, source input root, and campaign tag |
| `job_script_*.sh` | Exact SLURM/Apptainer script stored with a campaign |
| `*.out.<jobid>` | SLURM standard output |
| `*.err.<jobid>` | SLURM standard error |
| `convergence_log_*.txt` | Nonlinear-solver convergence events |
| `01_phasefield_*_log.txt` | Case-level application log |

