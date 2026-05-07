import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ufl
import basix.ufl
import basix
import dolfinx as dlfx
from mpi4py import MPI
from petsc4py import PETSc as petsc
from scipy.interpolate import griddata

import alex.homogenization
import alex.linearelastic as le
import alex.phasefield as pf
import alex.postprocessing as pp
import alex.util
import alex.os
import alex.boundaryconditions as bc
import alex.solution as sol
import json
from pathlib import Path

# ---------------------------
# CLI INPUT HANDLING
# ---------------------------
DEFAULT_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "resources",
    "260504_dcb_beta_phi_a_rho_var_min_max",
)
VALID_CASES = {"auto", "vary", "min", "max", "all", "fromfile"}
VALID_SPLITS = {"spectral", "volumetric", "all"}
DEFAULT_CASE = "auto"
DEFAULT_SPLIT = "spectral"
DEFAULT_EPSILON = 0.015
STATIC_OPTIMIZATION_PARAMS = {
    "E0toE1": 0.6,
    "E": 210000.0,
}



def read_params(file_path):
    def parse_value(value):
        value = value.strip()

        # range like 0.5:0.5:10.0
        if ":" in value:
            parts = value.split(":")
            try:
                start, step, end = map(float, parts)
                n = int((end - start) / step) + 1
                return [start + i * step for i in range(n)]
            except ValueError:
                pass

        # float
        try:
            return float(value)
        except ValueError:
            pass

        # boolean
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False

        # empty array like Any[]
        if value.endswith("[]"):
            return []

        # fallback (string)
        return value

    params = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue

            key, value = line.split("=", 1)
            params[key.strip()] = parse_value(value)

    return params

def parse_args(argv, rank=0):
    """
    Accepted forms:
      python script.py
      python script.py ROOT_OR_LEAF
      python script.py ROOT_OR_LEAF SPLIT
      python script.py ROOT_OR_LEAF CASE SPLIT
      python script.py ROOT_OR_LEAF START END CASE SPLIT
      python script.py ROOT_OR_LEAF CASE SPLIT --epsilon VALUE
      python script.py ROOT_OR_LEAF CASE SPLIT epsilon=VALUE
    CASE in {auto|vary|min|max|all|fromfile}; SPLIT in {spectral|volumetric|all}
    """
    folder = DEFAULT_FOLDER
    ds_start = None
    ds_end = None
    case = DEFAULT_CASE
    split = DEFAULT_SPLIT
    epsilon_value = DEFAULT_EPSILON
    used_defaults = []

    if len(argv) >= 2:
        folder = argv[1]
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Provided folder path does not exist: {folder}")
    else:
        used_defaults.append("folder")

    # Collect remaining tokens and try to interpret ints vs case/split/epsilon.
    tokens = argv[2:]
    cleaned_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in {"--epsilon", "-e"}:
            if i + 1 >= len(tokens):
                raise ValueError(f"{token} requires a numeric value")
            epsilon_value = float(tokens[i + 1])
            i += 2
            continue
        if token.startswith("--epsilon="):
            epsilon_value = float(token.split("=", 1)[1])
            i += 1
            continue
        if token.startswith("epsilon="):
            epsilon_value = float(token.split("=", 1)[1])
            i += 1
            continue
        cleaned_tokens.append(token)
        i += 1

    tokens = cleaned_tokens
    ints = []
    others = []
    for t in tokens:
        try:
            ints.append(int(t))
        except ValueError:
            others.append(t.lower())

    if len(ints) == 1:
        ds_start = ds_end = ints[0]
    elif len(ints) >= 2:
        ds_start, ds_end = ints[0], ints[1]
    else:
        used_defaults.append("dataset indices")

    for token in others:
        if token in VALID_CASES:
            case = token
        elif token in VALID_SPLITS:
            split = token
        else:
            raise ValueError(
                f"Unknown argument '{token}'. Valid cases: {sorted(VALID_CASES)}, "
                f"valid splits: {sorted(VALID_SPLITS)}"
            )

    if not any(token in VALID_CASES for token in others):
        used_defaults.append("case")
    if not any(token in VALID_SPLITS for token in others):
        used_defaults.append("split")
    if epsilon_value == DEFAULT_EPSILON:
        used_defaults.append("epsilon")

    # Print warnings for defaults if rank == 0
    if rank == 0 and used_defaults:
        print(f"[WARNING] Using default values for: {', '.join(used_defaults)}")

    if rank == 0:
        print(f"[INFO] Using folder/root: {folder}")
        print(
            f"[INFO] Dataset start: {ds_start}, end: {ds_end}, "
            f"case: {case}, split: {split}, epsilon: {epsilon_value}"
        )

    return folder, ds_start, ds_end, case, split, epsilon_value

folder_path, dataset_start, dataset_end, case_param, split_param, epsilon_param = parse_args(sys.argv)

# read optimization parameters
params_optimization_path = Path(folder_path).parent / "params.txt"
if params_optimization_path.exists():
    params_optimization = read_params(params_optimization_path)
else:
    params_optimization = STATIC_OPTIMIZATION_PARAMS

E0toE1=params_optimization["E0toE1"]
E_max = params_optimization["E"]
E_min = E_max * E0toE1
print(f"[INFO] Using folder/root: {folder_path}")

# ---------------------------
# AUTO-DETECT DATASET FOLDERS
# ---------------------------
def is_dataset_folder(path):
    required = ["node_coords.csv", "points_data.csv", "cell_data.csv", "connectivity.csv", "mesh.xdmf"]
    return all((Path(path) / name).is_file() for name in required)


def discover_dataset_folders(root):
    root = Path(root)
    if is_dataset_folder(root):
        return [root]
    return sorted(path for path in root.rglob("*") if path.is_dir() and is_dataset_folder(path))


def infer_case_from_folder(path):
    name = Path(path).name.lower()
    if name.endswith("_min"):
        return "min"
    if name.endswith("_max"):
        return "max"
    return "vary"


def infer_a_value_from_folder(path, fallback):
    match = re.search(r"_a_([0-9]+(?:_[0-9]+)?)_", Path(path).name)
    if match:
        return float(match.group(1).replace("_", "."))
    return float(fallback)


def safe_dataset_label(path):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(path).name)


def safe_float_label(value):
    return f"{value:g}".replace(".", "_").replace("-", "m")


epsilon_output_suffix = "" if np.isclose(epsilon_param, DEFAULT_EPSILON) else f"_eps{safe_float_label(epsilon_param)}"


all_dataset_folders = discover_dataset_folders(folder_path)

if not all_dataset_folders:
    raise FileNotFoundError(f"No 260504 dataset leaf folders found below {folder_path}")

if dataset_start is not None and dataset_end is not None:
    dataset_specs = [
        (idx, path)
        for idx, path in enumerate(all_dataset_folders, start=1)
        if dataset_start <= idx <= dataset_end
    ]
else:
    dataset_specs = list(enumerate(all_dataset_folders, start=1))

if not dataset_specs:
    raise ValueError(f"No dataset folders found in the specified range: {dataset_start} to {dataset_end}")



# ---------------------------
# MPI INITIALIZATION
# ---------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank == 0:
    print(f"[INFO] Detected dataset folders to process: {len(dataset_specs)}")

print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

# ---------------------------
# MAIN LOOP OVER SELECTED DATASET FOLDERS
# ---------------------------
splits_to_run = ["spectral", "volumetric"] if split_param == "all" else [split_param]

for dataset_index, dataset_folder in dataset_specs:
    folder_path = str(dataset_folder)
    dataset_label = safe_dataset_label(dataset_folder)
    x_value = dataset_index
    a_value = infer_a_value_from_folder(dataset_folder, dataset_index)
    convergence_log_path = os.path.join(folder_path, f"convergence_log_{split_param}{epsilon_output_suffix}.txt")
    if rank == 0:
        with open(convergence_log_path, "w") as f:
            f.write("dataset_index,dataset_label,case,split,status\n")
        print(f"[INFO] Processing dataset {dataset_index}: {folder_path}")

    # ---------------------------
    # BUILD FILE PATHS
    # ---------------------------
    node_file = os.path.join(folder_path, "node_coords.csv")
    point_data_file = os.path.join(folder_path, "points_data.csv")
    cell_data_file = os.path.join(folder_path, "cell_data.csv")
    connectivity_file = os.path.join(folder_path, "connectivity.csv")
    mesh_file = os.path.join(folder_path, "dlfx_mesh_1.xdmf")

    # case-agnostic figure (E-distribution); case-specific outputs will be below
    base_results_xdmf_path = os.path.join(folder_path, f"results_{dataset_label}.xdmf")  # kept for mesh write convenience
    base_output_graph_path = os.path.join(folder_path, f"result_graphs_{dataset_label}.txt")  # not used directly, but kept

    # ---------------------------
    # VALIDATE FILES
    # ---------------------------
    for fpath in [node_file, point_data_file, cell_data_file, connectivity_file, mesh_file]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Required file not found: {fpath}")

    if rank ==0:
        print(f"[INFO] All required files found for dataset {x_value}.")

    # ---------------------------
    # HELPER FUNCTIONS
    # ---------------------------
    def load_data(file_path):
        return pd.read_csv(file_path)

    def infer_mesh_dimensions_from_nodes(nodes_df):
        unique_y_coords = nodes_df['Points_1'].unique()
        unique_x_coords = nodes_df['Points_0'].unique()
        unique_y_coords.sort()
        unique_x_coords.sort()
        return len(unique_y_coords) - 1, len(unique_x_coords) - 1

    # def arrange_cells_2D(connectivity_df, mesh_dims):
    #     cell_grid = np.zeros(mesh_dims, dtype=int)
    #     for index, row in connectivity_df.iterrows():
    #         cell_id = row['Cell ID']
    #         row_idx = index // mesh_dims[1]
    #         col_idx = index % mesh_dims[1]
    #         cell_grid[row_idx, col_idx] = cell_id
    #     return cell_grid
    

    def arrange_cells_2D(connectivity_df, mesh_dims):
        """
        Arrange cell IDs into a 2D grid based on DataFrame order or cell IDs.

        Parameters
        ----------
        connectivity_df : pandas.DataFrame
            DataFrame containing cell connectivity information. 
            Optionally includes a 'Cell ID' column.
        mesh_dims : tuple of int
            The (rows, cols) dimensions of the desired mesh grid.

        Returns
        -------
        np.ndarray
            A 2D numpy array with cell IDs arranged in grid order.
        """
        cell_grid = np.zeros(mesh_dims, dtype=int)

        for index, row in connectivity_df.iterrows():
            # Use 'Cell ID' if present; otherwise default to index
            cell_id = row['Cell ID'] if 'Cell ID' in connectivity_df.columns else index

            # Compute 2D position
            row_idx = index // mesh_dims[1]
            col_idx = index % mesh_dims[1]

            # Assign to grid
            cell_grid[row_idx, col_idx] = cell_id

        return cell_grid


    def map_E_to_grid(cell_id_grid, cell_data_df):
        E_Grid = np.full(cell_id_grid.shape, np.nan)
        E_values = cell_data_df['E-Modul'].values
        for row in range(cell_id_grid.shape[0]):
            for col in range(cell_id_grid.shape[1]):
                cell_id = cell_id_grid[row, col]
                if cell_id < len(E_values):
                    E_Grid[row, col] = E_values[cell_id]
                else:
                    E_Grid[row, col] = np.nan
        return E_Grid
    
    def map_porosity_to_grid(cell_id_grid, cell_data_df):
        porosity_grid = np.full(cell_id_grid.shape, np.nan)
        porosity_values = cell_data_df['porosity'].values
        
        porosity_values = 1.0 - cell_data_df['porosity'].values # correction definition of porosity
        for row in range(cell_id_grid.shape[0]):
            for col in range(cell_id_grid.shape[1]):
                cell_id = cell_id_grid[row, col]
                if cell_id < len(porosity_values):
                    porosity_grid[row, col] = porosity_values[cell_id]
                else:
                    porosity_grid[row, col] = np.nan
                    
        return porosity_grid

    def calculate_element_size(nodes_df):
        x1, y1 = nodes_df.iloc[0]['Points_0'], nodes_df.iloc[0]['Points_1']
        x2, y2 = nodes_df.iloc[1]['Points_0'], nodes_df.iloc[1]['Points_1']
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def interpolate_pixel_data(data, element_size, x_coords, y_coords, method='linear'):
        grid_x, grid_y = np.meshgrid(
            (np.arange(data.shape[1]) + 0.5) * element_size, 
            (np.arange(data.shape[0]) + 0.5) * element_size
        )
        points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        values = data.ravel()
        query_points = np.column_stack((x_coords, y_coords))
        interpolated_values = griddata(points, values, query_points, method=method)
        nan_mask = np.isnan(interpolated_values)
        if np.any(nan_mask):
            interpolated_values[nan_mask] = griddata(points, values, query_points[nan_mask], method='nearest')
        return interpolated_values

    def create_emodulus_interpolator(nodes_df, E_grid):
        return lambda x: interpolate_pixel_data(E_grid, calculate_element_size(nodes_df), x[0], x[1])
    
    
    # --------------------------------------------------
# Generic pixel interpolation (UNCHANGED STYLE)
# --------------------------------------------------
    def interpolate_pixel_data(data, element_size, x_coords, y_coords, method='linear'):
        grid_x, grid_y = np.meshgrid(
            (np.arange(data.shape[1]) + 0.5) * element_size,
            (np.arange(data.shape[0]) + 0.5) * element_size
        )

        points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        values = data.ravel()
        query_points = np.column_stack((x_coords, y_coords))

        interpolated_values = griddata(points, values, query_points, method=method)

        # Fallback to nearest if NaN
        nan_mask = np.isnan(interpolated_values)
        if np.any(nan_mask):
            interpolated_values[nan_mask] = griddata(
                points, values, query_points[nan_mask], method='nearest'
            )

        return interpolated_values


    import numpy as np


    def compute_jmax_grid_from_porosity(
        porosity_grid,
        A, B, C,
        gc_min=0.1,
        gc_max=1.0
    ):
        """
        Converts a porosity grid into a Jmax grid using:

            x = sqrt(pi / (4 phi)) - 1
            Jmax = A - B exp(-C x)

        Parameters
        ----------
        porosity_grid : ndarray
        A, B, C : float
            Fit parameters
        gc_min : float or None
            Minimum allowed gc value
        gc_max : float or None
            Maximum allowed gc value

        Returns
        -------
        jmax_grid : ndarray
        """

        # Prevent division problems
        phi = np.clip(porosity_grid, 1e-12, None)

        # Convert porosity → ws/L
        x = np.sqrt(np.pi / (4.0 * phi)) - 1.0

        # Compute gc
        jmax_grid = A - B * np.exp(-C * x)

        # Apply bounds if requested
        if gc_min is not None or gc_max is not None:
            jmax_grid = np.clip(jmax_grid, gc_min, gc_max)

        return jmax_grid


    # --------------------------------------------------
    # GC interpolator factory (same structure as E-modulus)
    # --------------------------------------------------
    def create_gc_interpolator(
        nodes_df,
        porosity_grid,
        A, B, C,
        gc_min=None,
        gc_max=None
    ):

        element_size = calculate_element_size(nodes_df)

        jmax_grid = compute_jmax_grid_from_porosity(
            porosity_grid,
            A, B, C,
            gc_min=gc_min,
            gc_max=gc_max
        )

        return lambda x: interpolate_pixel_data(
            jmax_grid,
            element_size,
            x[0],
            x[1]
        )
        
        
    
    # Helper to read vol JSON produced previously.
    def read_E_average_from_vol_json(x_val):
        vol_filename = os.path.join(folder_path, f"vol_{dataset_label}_vary_{split_name}{epsilon_output_suffix}.json")
        if not os.path.exists(vol_filename):
            raise FileNotFoundError(f"Expected volume file for case 'fromfile' not found: {vol_filename}")
        with open(vol_filename, "r") as f:
            data = json.load(f)
        if "E_average" not in data:
            raise KeyError(f"'E_average' not found in {vol_filename}. File contents: {list(data.keys())}")
        return float(data["E_average"])
    
    def read_field_from_vol_json(x_val, field):
        vol_filename = os.path.join(folder_path, f"vol_{dataset_label}_vary_{split_name}{epsilon_output_suffix}.json")
        
        if not os.path.exists(vol_filename):
            raise FileNotFoundError(
                f"Expected volume file for case 'fromfile' not found: {vol_filename}"
            )

        with open(vol_filename, "r") as f:
            data = json.load(f)

        if field not in data:
            raise KeyError(
                f"'{field}' not found in {vol_filename}. File contents: {list(data.keys())}"
            )

        return float(data[field])
    

    def log_convergence_status(x_value, case, split, status):
        if rank == 0:
            with open(convergence_log_path, "a") as f:
                f.write(f"{x_value},{dataset_label},{case},{split},{status}\n")

    # ---------------------------
    # LOAD DATA
    # ---------------------------
    nodes_df = load_data(node_file)
    point_data_df = load_data(point_data_file)
    cell_data_df = load_data(cell_data_file)
    connectivity_df = load_data(connectivity_file)

    mesh_dims = infer_mesh_dimensions_from_nodes(nodes_df)
    cell_id_grid = arrange_cells_2D(connectivity_df, mesh_dims)
    E_grid = map_E_to_grid(cell_id_grid, cell_data_df)
    porosity_grid = map_porosity_to_grid(cell_id_grid,cell_data_df)
    #E_max, E_min = np.max(E_grid), 100000.0 #np.min(E_grid)

    # Plot E distribution (once per dataset index)
    plt.figure(figsize=(10, 8))
    plt.imshow(E_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='E')
    plt.title(f'E Distribution for dataset {x_value}')
    plt.savefig(os.path.join(folder_path, f'E_distribution_{dataset_label}.png'), dpi=300)
    plt.close()
    
    # Plot porosity distribution (once per dataset index)
    plt.figure(figsize=(10, 8))
    plt.imshow(porosity_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Porosity')
    plt.title(f'Porosity Distribution for dataset {x_value}')
    plt.savefig(os.path.join(folder_path, f'porosity_distribution_{dataset_label}.png'), dpi=300)
    plt.close()

    # ---------------------------
    # MPI + MESH LOADING
    # ---------------------------
    with dlfx.io.XDMFFile(comm, mesh_file, 'r') as mesh_inp:
        domain = mesh_inp.read_mesh()
        
    # with dlfx.io.XDMFFile(comm, os.path.join("/home/scripts/052-Special-Issue-IJF-Hannover/resources/310125_var_bcpos_rho_10_120_004","dlfx_mesh_20.xdmf"), 'r') as mesh_inp:
    #     domain = mesh_inp.read_mesh()

    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = pp.compute_bounding_box(comm, domain)
    if rank == 0:
        pp.print_bounding_box(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all)

    Ve = basix.ufl.element("P", domain.basix_cell(), 1, shape=(2,))
    Se = basix.ufl.element("P", domain.basix_cell(), 1, shape=())
    W = dlfx.fem.FunctionSpace(domain, basix.ufl.mixed_element([Ve, Se]))
    S = dlfx.fem.FunctionSpace(domain, Se)

    # ---------------------------
    # CASE LOOP
    # ---------------------------
    available_cases = ["vary", "min", "max", "fromfile"]
    if case_param == "auto":
        cases_to_run = [infer_case_from_folder(folder_path)]
    elif case_param is None or case_param == "all":
        cases_to_run = available_cases
    else:
        if case_param not in available_cases:
            if rank == 0:
                print(f"[WARNING] Unknown case '{case_param}'. Falling back to all cases {available_cases}.")
            cases_to_run = available_cases
        else:
            cases_to_run = [case_param]

    for split_name, case in [
        (split_name, case)
        for split_name in splits_to_run
        for case in cases_to_run
    ]:
        if rank == 0:
            print(f"[INFO] Running case '{case}' with split '{split_name}' for dataset {dataset_label}")

        # ---- Case-specific output paths to avoid overwrites
        results_xdmf_path = os.path.join(folder_path, f"results_{dataset_label}_{case}_{split_name}{epsilon_output_suffix}.xdmf")
        outputfile_graph_path = os.path.join(folder_path, f"result_graphs_{dataset_label}_{case}_{split_name}{epsilon_output_suffix}.txt")

        # ---- Material fields
        E = dlfx.fem.Function(S)
        porosity = dlfx.fem.Function(S)
        nu = dlfx.fem.Constant(domain=domain, c=0.3)

        # ---- Set E depending on case
        if case == "vary":
            E.interpolate(create_emodulus_interpolator(nodes_df, E_grid))
            porosity.interpolate(create_emodulus_interpolator(nodes_df, porosity_grid))
        elif case == "min":
            E.x.array[:] = np.full_like(E.x.array[:], E_min)
            porosity.interpolate(create_emodulus_interpolator(nodes_df, porosity_grid))
        elif case == "max":
            E.x.array[:] = np.full_like(E.x.array[:], E_max)
            porosity.interpolate(create_emodulus_interpolator(nodes_df, porosity_grid))
        elif case == "fromfile":
            
            # read vol_{x}_vary.json and get E_average
            try:
                E_average_value = read_field_from_vol_json(x_value,"E_average") #read_E_average_from_vol_json(x_value)
                porosity_average_value = read_field_from_vol_json(x_value,"porosity_average")
            except Exception as e:
                # ensure we report and stop this case cleanly
                if rank == 0:
                    print(f"[ERROR] Could not read E_average for x={x_value}: {e}")
                log_convergence_status(x_value, case, split_name, f"ErrorReadingVolJson: {e}")
                # skip this case and continue with next case
                continue
            # assign constant value
            E.x.array[:] = np.full_like(E.x.array[:], E_average_value)
            porosity.x.array[:] = np.full_like(porosity.x.array[:], porosity_average_value)
            if rank == 0:
                print(f"[INFO] For dataset {dataset_label} using E_average={E_average_value} from vary volume JSON")
                print(f"[INFO] For dataset {dataset_label} using porosity_average={porosity_average_value} from vary volume JSON")
        else:
            # should not happen, but guard
            raise ValueError(f"Unhandled case: {case}")

        lam = le.get_lambda(E, nu)
        mue = le.get_mu(E, nu)
        dim = domain.topology.dim
        alex.os.mpi_print('spatial dimensions: ' + str(dim), rank)

        # ---- Boundary dofs (top boundary, u_y)
        fdim = domain.topology.dim - 1
        atol = 1e-12
        atol_bc = 0.0
       
        increment_a = 0.5#0.5
        width_applied_load = 0.075 #+ increment_a * 0.2 # modification to stabilize numerical problems

        # ---- Simulation parameters
        dt_start = 0.001
        dt_global = dlfx.fem.Constant(domain, dt_start)
        dt_max = dlfx.fem.Constant(domain, dt_start)
        t_global = dlfx.fem.Constant(domain, 0.0000001)
        trestart_global = dlfx.fem.Constant(domain, t_global.value)
        Tend = 50.0 * dt_global.value * a_value
        
        # if case == "vary":
        #     # hard coded from fit 
        #     A = 1.243657
        #     B = 3.150239
        #     C = 2.850765
        #     gc = dlfx.fem.Function(S)
        #     gc.interpolate(create_gc_interpolator(nodes_df,porosity_grid,A,B,C))
        # else:
        #     gc = dlfx.fem.Constant(domain, 1.0)
        
        
        A = 1.243657
        B = 3.150239
        C = 2.850765
        gc = dlfx.fem.Function(S)
        gc.interpolate(create_gc_interpolator(nodes_df,porosity_grid,A,B,C,gc_min=0.1,gc_max=1.0))    
        
        eta = dlfx.fem.Constant(domain, 0.001)
        epsilon = dlfx.fem.Constant(domain, epsilon_param)
        Mob = dlfx.fem.Constant(domain, 100.0)
        iMob = dlfx.fem.Constant(domain, 1.0 / Mob.value)
        
       #λ_arc_length = dlfx.fem.Constant(domain, petsc.ScalarType(0.0000000001))

        # ---- Solution fields
        w = dlfx.fem.Function(W)
        u, s = w.split()
        wrestart = dlfx.fem.Function(W)
        wm1 = dlfx.fem.Function(W)
        um1, sm1 = ufl.split(wm1)
        dw = ufl.TestFunction(W)
        ddw = ufl.TrialFunction(W)

        phaseFieldProblem = pf.StaticPhaseFieldProblem2D_split(
            degradationFunction=pf.quadratic_degradation(),
            psisurf=pf.psisurf_from_function,
            split=split_name,
            geometric_nl=False
        )
        
        # phaseFieldProblem = pf.StaticPhaseFieldProblem2D(degradationFunction=pf.degrad_quadratic,
        #                                                  psisurf=pf.psisurf_from_function)

        timer = dlfx.common.Timer()

        # ---- Logs
        script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
        logfile_path = alex.os.logfile_full_path(folder_path, f"{script_name_without_extension}_{dataset_label}_{case}_{split_name}{epsilon_output_suffix}")

        # ---- Hooks
        def before_first_time_step():
            timer.start()
            wm1.sub(1).x.array[:] = np.ones_like(wm1.sub(1).x.array[:])
            wrestart.x.array[:] = wm1.x.array[:]
            if rank == 0:
                pp.prepare_graphs_output_file(outputfile_graph_path)
            # write mesh container once so XDMF exists
            pp.write_meshoutputfile(domain, results_xdmf_path, comm)

        def before_each_time_step(t, dt):
            if rank == 0:
                sol.print_time_and_dt(t, dt)

        def get_residuum_and_gateaux(delta_t):
            return phaseFieldProblem.prep_newton(
                w=w, wm1=wm1, dw=dw, ddw=ddw, lam=lam, mu=mue,
                Gc=gc, epsilon=epsilon, eta=eta, iMob=iMob, delta_t=delta_t
            )

        n = ufl.FacetNormal(domain)
        # external_surface_tag = 5
        # external_surface_tags = pp.tag_part_of_boundary(domain,bc.get_boundary_of_box_as_function(domain, comm,atol=atol*0.0),external_surface_tag)
        # ds = ufl.Measure('ds', domain=domain, subdomain_data=external_surface_tags)
        
        
        
        top_surface_tag = 9
        top_surface_tags = pp.tag_part_of_boundary(
            domain, bc.get_top_boundary_of_box_as_function(domain, comm, atol=atol*1.0), top_surface_tag
        )
        ds_top_tagged = ufl.Measure('ds', domain=domain, subdomain_data=top_surface_tags)

        success_timestep_counter = dlfx.fem.Constant(domain, 0.0)
        postprocessing_interval = dlfx.fem.Constant(domain, 50.0)


        
        load_left_bc_function = bc.get_x_range_at_top_of_box_as_function(domain,comm,width_applied_load,(x_max_all-x_min_all) / 3 + x_min_all,atol=atol_bc)
        load_right_bc_function = bc.get_x_range_at_top_of_box_as_function(domain,comm,width_applied_load,2*(x_max_all-x_min_all) / 3 + x_min_all,atol=atol_bc)

        facets_at_left_load = dlfx.mesh.locate_entities_boundary(
            domain, fdim, load_left_bc_function
        )
        facets_at_right_load = dlfx.mesh.locate_entities_boundary(
            domain, fdim, load_right_bc_function
        )
        dofs_at_left_load_y = dlfx.fem.locate_dofs_topological(
            W.sub(0).sub(1), fdim, facets_at_left_load
        )
        dofs_at_right_load_y = dlfx.fem.locate_dofs_topological(
            W.sub(0).sub(1), fdim, facets_at_right_load
        )
        dofs_at_loaded_y = np.unique(
            np.concatenate((dofs_at_left_load_y, dofs_at_right_load_y))
        )

        left_bc_tag = 1
        left_bc_surface_tags = pp.tag_part_of_boundary(
            domain, load_left_bc_function, left_bc_tag
        )
        ds_left_bc_tagged = ufl.Measure('ds', domain=domain, subdomain_data=left_bc_surface_tags)
        
        right_bc_tag = 1
        right_bc_surface_tags = pp.tag_part_of_boundary(
            domain, load_right_bc_function, right_bc_tag
        )
        ds_right_bc_tagged = ufl.Measure('ds', domain=domain, subdomain_data=right_bc_surface_tags)

        def get_bcs(t):
           
            
            bcs = [
                # left
                bc.define_dirichlet_bc_from_value(domain,0.0,0,
                                                  bc.get_left_boundary_of_box_as_function(domain,comm,atol=atol_bc),W,0),
                bc.define_dirichlet_bc_from_value(domain,0.0,1,
                                                  bc.get_left_boundary_of_box_as_function(domain,comm,atol=atol_bc),W,0),
                #right
                bc.define_dirichlet_bc_from_value(domain,0.0,0,
                                                  bc.get_right_boundary_of_box_as_function(domain,comm,atol=atol_bc),W,0),
                bc.define_dirichlet_bc_from_value(domain,0.0,1,
                                                  bc.get_right_boundary_of_box_as_function(domain,comm,atol=atol_bc),W,0),
                
                #dcb_force_1
                # bc.define_dirichlet_bc_from_value(domain, -t_global.value, 1,
                #                                    bc.get_x_range_at_top_of_box_as_function(domain,comm,width_applied_load,(float(x_value) * increment_a) / 3,atol=atol_bc), W, 0),
                # bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
                #                                    bc.get_x_range_at_top_of_box_as_function(domain,comm,width_applied_load,(float(x_value) * increment_a) / 3,atol=atol_bc), W, 0),
                
                bc.define_dirichlet_bc_from_value(domain, -t_global.value, 1,
                                                   load_left_bc_function, W, 0),
                bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
                                                   load_left_bc_function, W, 0),
                
                
                #dcb_force_2
                bc.define_dirichlet_bc_from_value(domain, -t_global.value, 1,
                                                   load_right_bc_function, W, 0),
                bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
                                                   load_right_bc_function, W, 0),
                
                
            ]

            
            # bcs = [
            #     bc.define_dirichlet_bc_from_value(domain, t_global.value, 1,
            #                                        bc.get_x_range_at_top_of_box_as_function(domain,comm,width_applied_load,width_applied_load/2.0,atol=atol_bc), W, 0),
            #     bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
            #                                        bc.get_x_range_at_top_of_box_as_function(domain,comm,width_applied_load,width_applied_load/2.0,atol=atol_bc), W, 0),
            #     bc.define_dirichlet_bc_from_value(domain, 0.0, 1,
            #                                       bc.get_x_range_at_bottom_of_box_as_function(domain,comm,width_applied_load,float(x_value) * increment_a - width_applied_load/2,atol=atol_bc), W, 0),
            #     bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
            #                                       bc.get_x_range_at_bottom_of_box_as_function(domain,comm,width_applied_load,float(x_value) * increment_a - width_applied_load/2,atol=atol_bc), W, 0),
            #     # bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
            #     #                                   bc.get_left_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0),
            #     # bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
            #     #                                    bc.get_right_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0),
                
            #     # bc.define_dirichlet_bc_from_value(domain, -t_global.value, 1,
            #     #                                   bc.get_top_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0),
            #     # bc.define_dirichlet_bc_from_value(domain, 0.0, 1,
            #     #                                   bc.get_bottom_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0),
            #     bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
            #                                       bc.get_left_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0),
            #     # bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
            #     #                                   bc.get_right_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0)
            # ]

            
            # sigma_at_surface.value =  np.array([[0.0, 0.0],
            #                                     [0.0, -sigma_amplitude* t_global.value ]])
            # phaseFieldProblem.set_traction_bc(sigma_at_surface=sigma_at_surface,w=w,N=n,ds=ds_top_tagged(top_surface_tag))
            
            if abs(t) > sys.float_info.epsilon * 5:
                bcs.append(pf.irreversibility_bc(domain, W, wm1))
            return bcs

        Work = dlfx.fem.Constant(domain,0.0)
        
       
        dx = ufl.Measure("dx", domain=domain)
        vol = alex.homogenization.get_filled_vol(dx=dx,comm=comm)
        E_average = pp.get_volume_average_of_field(E,vol,dx=ufl.dx,comm=comm)
        porosity_average = pp.get_volume_average_of_field(porosity,vol,dx=ufl.dx,comm=comm)
        
        
        def write_vol_data_to_file():
            if rank == 0:
                vol_path = os.path.join(folder_path, f"vol_{dataset_label}_{case}_{split_name}{epsilon_output_suffix}.json")
                volumes_data = {
                        "vol": vol,
                        "E_average": E_average,
                        "porosity_average": porosity_average,
                    }
                with open(vol_path, "w") as f:
                    json.dump(volumes_data, f, indent=4)
                print(f"Saved volume info to: {vol_path}")
        
        write_vol_data_to_file()
        
        
        TEN = dlfx.fem.functionspace(domain, ("DP", 0, (dim, dim)))
        sigma_interpolated = dlfx.fem.Function(TEN) 



        
        def after_timestep_success(t, dt, iters):
            sigma = phaseFieldProblem.sigma_degraded(u, s, lam, mue, eta)
            tensor_field_expression = dlfx.fem.Expression(sigma, 
                                                                TEN.element.interpolation_points())
            tensor_field_name = "sigma"
            sigma_interpolated.interpolate(tensor_field_expression)
            sigma_interpolated.name = tensor_field_name
            
            # Reaction force at top boundary
            Rx_top, Ry_top_left = pp.reaction_force(sigma_interpolated, n=n, ds=ds_left_bc_tagged(1), comm=comm)
            Rx_top, Ry_top_right = pp.reaction_force(sigma_interpolated, n=n, ds=ds_right_bc_tagged(1), comm=comm)
            #Rx_top, Ry_top = pp.reaction_force(sigma_interpolated, n=n, ds=ds_top_tagged(top_surface_tag), comm=comm)

            # Get vertical displacement from the same y-DOFs where the load BC is applied.
            if len(dofs_at_loaded_y) > 0:
                u_y_top_local = np.min(w.x.array[dofs_at_loaded_y])
            else:
                u_y_top_local = 1e10

            comm.barrier()
            u_y_top = comm.allreduce(u_y_top_local, MPI.MIN)
            comm.barrier()




            
            # dW = pp.work_increment_external_forces(sigma,u,um1,n,ds_top_tagged(top_surface_tag),comm=comm)
            #dW = pp.work_increment_external_forces(sigma_interpolated,u,um1,n,ds=ufl.ds,comm=comm)
            dW_left = pp.work_increment_external_forces(sigma_interpolated,u,um1,n,ds=ds_left_bc_tagged(left_bc_tag),comm=comm)
            dW_right = pp.work_increment_external_forces(sigma_interpolated,u,um1,n,ds=ds_right_bc_tagged(right_bc_tag),comm=comm)
            dW = dW_left + dW_right
            Work.value = Work.value +dW
    
            A = pf.get_surf_area(s,epsilon=epsilon,dx=ufl.dx, comm=comm)
    
            E_el = phaseFieldProblem.get_E_el_global(s,eta,u,lam,mue,dx=ufl.dx,comm=comm)
    
            if rank == 0:
                pp.write_to_graphs_output_file(outputfile_graph_path, t, u_y_top, Ry_top_left, dW, Work.value, A, E_el,Ry_top_right)

            if rank == 0:
                sol.write_to_newton_logfile(logfile_path, t, dt, iters)
                
            wm1.x.array[:] = w.x.array[:]
            wrestart.x.array[:] = w.x.array[:]
            
            
                

            success_timestep_counter.value = success_timestep_counter.value + 1.0
            if int(success_timestep_counter.value) % int(postprocessing_interval.value) == 0:
                pp.write_phasefield_mixed_solution(domain, results_xdmf_path, w, t, comm)
                E.name = "E"
                pp.write_scalar_fields(domain, comm, [E,gc], ["E","gc"], outputfile_xdmf_path=results_xdmf_path, t=t)
                #pp.write_scalar_fields(domain, comm, [E], ["E"], outputfile_xdmf_path=results_xdmf_path, t=t)
                pp.write_tensor_fields(domain, comm, [sigma], ["sig"], outputfile_xdmf_path=results_xdmf_path, t=t)

        def after_timestep_restart(t, dt, iters):
            # If global dt has shrunk beyond tolerance -> write what we have and skip this case
            if dt_global.value < 10.0 ** (-14):
                sigma = phaseFieldProblem.sigma_degraded(u, s, lam, mue, eta)
                pp.write_phasefield_mixed_solution(domain, results_xdmf_path, w, t, comm)
                E.name = "E"
                pp.write_scalar_fields(domain, comm, [E], ["E"], outputfile_xdmf_path=results_xdmf_path, t=t)
                pp.write_tensor_fields(domain, comm, [sigma], ["sig"], outputfile_xdmf_path=results_xdmf_path, t=t)
                if rank == 0:
                    print(f"[WARNING] NO CONVERGENCE (dt too small) in case '{case}' for dataset {x_value}. Skipping to next case.")
                # Signal to outer try/except to continue with next case
                raise RuntimeError("ConvergenceFailure")
            # Otherwise: restore previous state and let the solver retry with smaller dt
            w.x.array[:] = wrestart.x.array[:]
            # random perturbation
            # epsilon_num = 1e-8  # adjust as needed (e.g., 1e-3 for larger noise)
            # w.sub(1).x.array[:] += epsilon_num * np.random.randn(*w.sub(1).x.array.shape)

        def after_last_timestep():
            timer.stop()
            if rank == 0:
                runtime = timer.elapsed()
                sol.print_runtime(runtime)
                sol.write_runtime_to_newton_logfile(logfile_path, runtime)
                pp.print_graphs_plot(outputfile_graph_path, print_path=folder_path, legend_labels=["u_y_top", "R_y_top", "dW", "W","A", "E_el"])

            
                # vol_path = os.path.join(folder_path, f"vol_{x_value}_{case}.json")
                # volumes_data = {
                #     "vol": vol,
                # }
                # with open(vol_path, "w") as f:
                #     json.dump(volumes_data, f, indent=4)
                # print(f"Saved volume info to: {vol_path}")
        
            sigma = phaseFieldProblem.sigma_degraded(u, s, lam, mue, eta)
            pp.write_phasefield_mixed_solution(domain, results_xdmf_path, w, t_global.value, comm)
            E.name = "E"
            pp.write_scalar_fields(domain, comm, [E], ["E"], outputfile_xdmf_path=results_xdmf_path, t=t_global.value)
            pp.write_tensor_fields(domain, comm, [sigma], ["sig"], outputfile_xdmf_path=results_xdmf_path, t=t_global.value)

        # ---- Run solver, but keep going on convergence failure
        try:
            sol.solve_with_newton_adaptive_time_stepping(
                domain,
                w,
                Tend,
                dt_global,
                before_first_timestep_hook=before_first_time_step,
                after_last_timestep_hook=after_last_timestep,
                before_each_timestep_hook=before_each_time_step,
                get_residuum_and_gateaux=get_residuum_and_gateaux,
                get_bcs=get_bcs,
                after_timestep_restart_hook=after_timestep_restart,
                after_timestep_success_hook=after_timestep_success,
                comm=comm,
                print_bool=True,
                t=t_global,
                trestart=trestart_global,
                arc_length=False,
                arc_length_ds=0.01,
                λ_arc_length=t_global,
                dt_max=dt_max
            )
            log_convergence_status(x_value, case, split_name, "OK")
        except RuntimeError as e:
            if "ConvergenceFailure" in str(e):
                log_convergence_status(x_value, case, split_name, f"ConvergenceFailure at time {t_global.value}")
                pp.write_phasefield_mixed_solution(domain, results_xdmf_path, w, t_global.value+0.0001, comm)
                continue  # skip to next case
            else:
                log_convergence_status(x_value, case, split_name, f"RuntimeError: {str(e)}")
                raise
