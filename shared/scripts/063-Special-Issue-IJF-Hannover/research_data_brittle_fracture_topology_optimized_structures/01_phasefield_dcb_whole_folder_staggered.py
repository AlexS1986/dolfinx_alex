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

# ---------------------------
# CLI INPUT HANDLING
# ---------------------------
DEFAULT_FOLDER = os.path.join(os.path.dirname(__file__), "resources", "250925_TTO_mbb_festlager_var_a_E_var_min_max","mbb_festlager_var_a_E_var")
VALID_CASES = {"vary", "min", "max", "all", "fromfile"}
DEFAULT_CASE = "vary"

def parse_args(argv, rank=0):
    """
    Accepted forms:
      python script.py
      python script.py FOLDER
      python script.py FOLDER CASE
      python script.py FOLDER INDEX
      python script.py FOLDER START END
      python script.py FOLDER START END CASE
      python script.py FOLDER INDEX CASE
    CASE in {vary|min|max|all}
    """
    folder = DEFAULT_FOLDER
    ds_start = 4
    ds_end = 4
    case = None
    used_defaults = []

    if len(argv) >= 2:
        folder = argv[1]
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Provided folder path does not exist: {folder}")
    else:
        used_defaults.append("folder")

    # Collect remaining tokens and try to interpret ints vs case
    tokens = argv[2:]
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

    if others:
        last = others[-1]
        if last in VALID_CASES:
            case = last
        else:
            raise ValueError(f"Unknown case '{last}'. Valid: {sorted(VALID_CASES)}")
    else:
        case = DEFAULT_CASE
        used_defaults.append("case")

    # Print warnings for defaults if rank == 0
    if rank == 0 and used_defaults:
        print(f"[WARNING] Using default values for: {', '.join(used_defaults)}")

    if rank == 0:
        print(f"[INFO] Using folder: {folder}")
        print(f"[INFO] Dataset start: {ds_start}, end: {ds_end}, case: {case}")

    return folder, ds_start, ds_end, case

folder_path, dataset_start, dataset_end, case_param = parse_args(sys.argv)

print(f"[INFO] Using folder: {folder_path}")

# ---------------------------
# AUTO-DETECT INTEGER SUFFIXES
# ---------------------------
available_files = os.listdir(folder_path)
pattern = re.compile(r"cell_data_(\d+)\.csv")
all_x_candidates = sorted([int(pattern.match(f).group(1)) for f in available_files if pattern.match(f)])

if not all_x_candidates:
    raise FileNotFoundError(f"No 'cell_data_x.csv' file found in {folder_path}")

# Filter if range is specified
if dataset_start is not None and dataset_end is not None:
    x_candidates = [x for x in all_x_candidates if dataset_start <= x <= dataset_end]
else:
    x_candidates = all_x_candidates

if not x_candidates:
    raise ValueError(f"No dataset indices found in the specified range: {dataset_start} to {dataset_end}")



# ---------------------------
# MPI INITIALIZATION
# ---------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank == 0:
    print(f"[INFO] Detected dataset indices to process: {x_candidates}")

print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

# ---------------------------
# MAIN LOOP OVER SELECTED x_candidates
# ---------------------------
convergence_log_path = os.path.join(folder_path, "convergence_log.txt")
if rank == 0:
        # Start fresh for each run
        with open(convergence_log_path, "w") as f:
            f.write("x_value,case,status\n")

for x_value in x_candidates:
    if rank == 0:
        print(f"[INFO] Processing dataset index: {x_value}")

    # ---------------------------
    # BUILD FILE PATHS
    # ---------------------------
    node_file = os.path.join(folder_path, f"node_coords_{x_value}.csv")
    point_data_file = os.path.join(folder_path, f"points_data_{x_value}.csv")
    cell_data_file = os.path.join(folder_path, f"cell_data_{x_value}.csv")
    connectivity_file = os.path.join(folder_path, f"connectivity_{x_value}.csv")
    mesh_file = os.path.join(folder_path, f"dlfx_mesh_{x_value}.xdmf")

    # case-agnostic figure (E-distribution); case-specific outputs will be below
    base_results_xdmf_path = os.path.join(folder_path, f"results_{x_value}.xdmf")  # kept for mesh write convenience
    base_output_graph_path = os.path.join(folder_path, f"result_graphs_{x_value}.txt")  # not used directly, but kept

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
    
      # Helper to read vol JSON produced previously (vol_{x_value}_vary.json)
    def read_E_average_from_vol_json(x_val):
        vol_filename = os.path.join(folder_path, f"vol_{x_val}_vary.json")
        if not os.path.exists(vol_filename):
            raise FileNotFoundError(f"Expected volume file for case 'fromfile' not found: {vol_filename}")
        with open(vol_filename, "r") as f:
            data = json.load(f)
        if "E_average" not in data:
            raise KeyError(f"'E_average' not found in {vol_filename}. File contents: {list(data.keys())}")
        return float(data["E_average"])
    

    def log_convergence_status(x_value, case, status):
        if rank == 0:
            with open(convergence_log_path, "a") as f:
                f.write(f"{x_value},{case},{status}\n")

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
    E_max, E_min = np.max(E_grid), 100000.0 #np.min(E_grid)

    # Plot E distribution (once per dataset index)
    plt.figure(figsize=(10, 8))
    plt.imshow(E_grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='E')
    plt.title(f'E Distribution for dataset {x_value}')
    plt.savefig(os.path.join(folder_path, f'E_distribution_{x_value}.png'), dpi=300)
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
    V = dlfx.fem.FunctionSpace(domain, Ve)
    S = dlfx.fem.FunctionSpace(domain, Se)

    # ---------------------------
    # CASE LOOP
    # ---------------------------
    available_cases = ["vary", "min", "max", "fromfile"]
    if case_param is None or case_param == "all":
        cases_to_run = available_cases
    else:
        if case_param not in available_cases:
            if rank == 0:
                print(f"[WARNING] Unknown case '{case_param}'. Falling back to all cases {available_cases}.")
            cases_to_run = available_cases
        else:
            cases_to_run = [case_param]

    for case in cases_to_run:
        if rank == 0:
            print(f"[INFO] Running case '{case}' for dataset index {x_value}")

        # ---- Case-specific output paths to avoid overwrites
        results_xdmf_path = os.path.join(folder_path, f"results_{x_value}_{case}.xdmf")
        outputfile_graph_path = os.path.join(folder_path, f"result_graphs_{x_value}_{case}.txt")

        # ---- Material fields
        E = dlfx.fem.Function(S)
        nu = dlfx.fem.Constant(domain=domain, c=0.3)

        # ---- Set E depending on case
        if case == "vary":
            E.interpolate(create_emodulus_interpolator(nodes_df, E_grid))
        elif case == "min":
            E.x.array[:] = np.full_like(E.x.array[:], E_min)
        elif case == "max":
            E.x.array[:] = np.full_like(E.x.array[:], E_max)
        elif case == "fromfile":
            # read vol_{x}_vary.json and get E_average
            try:
                E_average_value = read_E_average_from_vol_json(x_value)
            except Exception as e:
                # ensure we report and stop this case cleanly
                if rank == 0:
                    print(f"[ERROR] Could not read E_average for x={x_value}: {e}")
                log_convergence_status(x_value, case, f"ErrorReadingVolJson: {e}")
                # skip this case and continue with next case
                continue
            # assign constant value
            E.x.array[:] = np.full_like(E.x.array[:], E_average_value)
            if rank == 0:
                print(f"[INFO] For dataset {x_value} using E_average={E_average_value} from vol_{x_value}_vary.json")
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
       
        increment_a = 0.5
        width_applied_load = 0.2 #+ increment_a * 0.2 # modification to stabilize numerical problems
        
        facets_at_boundary = dlfx.mesh.locate_entities_boundary(
            domain, fdim, bc.get_x_range_at_top_of_box_as_function(domain,comm,width_applied_load,width_applied_load/2.0,atol=atol_bc) 
        )
        dofs_at_boundary_y = dlfx.fem.locate_dofs_topological(V.sub(1), fdim, facets_at_boundary)

        # ---- Simulation parameters
        dt_start = 0.001
        dt_global = dlfx.fem.Constant(domain, dt_start)
        dt_max = dlfx.fem.Constant(domain, dt_start)
        t_global = dlfx.fem.Constant(domain, 0.000000)
        trestart_global = dlfx.fem.Constant(domain, t_global.value)
        Tend = 0.02 * x_value
        gc = dlfx.fem.Constant(domain, 1.0)
        eta = dlfx.fem.Constant(domain, 0.001)
        epsilon = dlfx.fem.Constant(domain, 0.1)
        Mob = dlfx.fem.Constant(domain, 10.0)
        iMob = dlfx.fem.Constant(domain, 1.0 / Mob.value)
        
       #λ_arc_length = dlfx.fem.Constant(domain, petsc.ScalarType(0.0000000001))

        # ---- Solution fields
        u = dlfx.fem.Function(V)
        s = dlfx.fem.Function(S)
        #w = dlfx.fem.Function(W)
        #u, s = w.split()
        urestart = dlfx.fem.Function(V)
        srestart = dlfx.fem.Function(S)
        # wrestart = dlfx.fem.Function(W)
        um1 = dlfx.fem.Function(V)
        sm1 = dlfx.fem.Function(S)
        # um1, sm1 = ufl.split(wm1)
        du = ufl.TestFunction(V)
        dphasefield = ufl.TestFunction(S)

        phaseFieldProblem = pf.StaticPhaseFieldProblem2D_split(
            degradationFunction=pf.degrad_quadratic,
            psisurf=pf.psisurf_from_function,
            split="spectral",
            geometric_nl=False
        )
        
        # phaseFieldProblem = pf.StaticPhaseFieldProblem2D(degradationFunction=pf.degrad_quadratic,
        #                                                  psisurf=pf.psisurf_from_function)

        timer = dlfx.common.Timer()

        # ---- Logs
        script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
        logfile_path = alex.os.logfile_full_path(folder_path, f"{script_name_without_extension}_{x_value}_{case}")

        # ---- Hooks
        def before_first_time_step():
            timer.start()
            sm1.x.array[:] = np.ones_like(sm1.x.array[:])
            um1.x.array[:] = np.zeros_like(u.x.array[:])
            urestart.x.array[:] = np.zeros_like(u.x.array[:])
            srestart.x.array[:] = np.zeros_like(s.x.array[:])
            if rank == 0:
                pp.prepare_graphs_output_file(outputfile_graph_path)
            # write mesh container once so XDMF exists
            pp.write_meshoutputfile(domain, results_xdmf_path, comm)

        def before_each_time_step(t, dt):
            if rank == 0:
                sol.print_time_and_dt(t, dt)

        def get_residuum_and_gateaux_u(delta_t):
            return phaseFieldProblem.prep_newton_staggered_u(u,s,du,lam=lam,mu=mue,Gc=gc,epsilon=epsilon,eta=eta)
        
        def get_residuum_and_gateaux_s(delta_t):
            return phaseFieldProblem.prep_newton_staggered_s(u,s,sm1, dphasefield ,lam,mue,gc,epsilon,eta,iMob,delta_t=delta_t)
        
        

        n = ufl.FacetNormal(domain)
        external_surface_tag = 5
        external_surface_tags = pp.tag_part_of_boundary(domain,bc.get_boundary_of_box_as_function(domain, comm,atol=atol*0.0),external_surface_tag)
        ds = ufl.Measure('ds', domain=domain, subdomain_data=external_surface_tags)
        
        
        
        top_surface_tag = 9
        top_surface_tags = pp.tag_part_of_boundary(
            domain, bc.get_top_boundary_of_box_as_function(domain, comm, atol=atol*1.0), top_surface_tag
        )
        ds_top_tagged = ufl.Measure('ds', domain=domain, subdomain_data=top_surface_tags)

        success_timestep_counter = dlfx.fem.Constant(domain, 0.0)
        postprocessing_interval = dlfx.fem.Constant(domain, 500.0)


        sigma_at_surface = dlfx.fem.Constant(domain, np.array([[0.0, 0.0],
                    [0.0, -1.0]]))
        sigma_amplitude = 1.0

        def get_bcs(t):
           
            bcs_u = [
                bc.define_dirichlet_bc_from_value(domain, -t_global.value, 1,
                                                  bc.get_top_boundary_of_box_as_function(domain, comm, atol=atol_bc), V, -1),
                bc.define_dirichlet_bc_from_value(domain, 0.0, 1,
                                                  bc.get_bottom_boundary_of_box_as_function(domain, comm, atol=atol_bc), V, -1),
                bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
                                                  bc.get_left_boundary_of_box_as_function(domain, comm, atol=atol_bc), V, -1),
                bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
                                                  bc.get_right_boundary_of_box_as_function(domain, comm, atol=atol_bc), V, -1)
            ]
            

            
            # bcs_u = [
            #     bc.define_dirichlet_bc_from_value(domain, -t_global.value, 1,
            #                                        bc.get_x_range_at_top_of_box_as_function(domain,comm,width_applied_load,width_applied_load/2.0,atol=atol_bc), V, -1),
            #     bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
            #                                        bc.get_x_range_at_top_of_box_as_function(domain,comm,width_applied_load,width_applied_load/2.0,atol=atol_bc), V, -1),
            #     bc.define_dirichlet_bc_from_value(domain, 0.0, 1,
            #                                       bc.get_x_range_at_bottom_of_box_as_function(domain,comm,width_applied_load,float(x_value) * increment_a - width_applied_load/2,atol=atol_bc), V, -1),
            #     bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
            #                                       bc.get_x_range_at_bottom_of_box_as_function(domain,comm,width_applied_load,float(x_value) * increment_a - width_applied_load/2,atol=atol_bc), V, -1),
            #     # bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
            #     #                                   bc.get_left_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0),
            #     # bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
            #     #                                    bc.get_right_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0),
                
            #     # bc.define_dirichlet_bc_from_value(domain, -t_global.value, 1,
            #     #                                   bc.get_top_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0),
            #     # bc.define_dirichlet_bc_from_value(domain, 0.0, 1,
            #     #                                   bc.get_bottom_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0),
            #     bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
            #                                       bc.get_left_boundary_of_box_as_function(domain, comm, atol=atol_bc), V, -1),
            #     # bc.define_dirichlet_bc_from_value(domain, 0.0, 0,
            #     #                                   bc.get_right_boundary_of_box_as_function(domain, comm, atol=atol_bc), W, 0)
            # ]

            
            # sigma_at_surface.value =  np.array([[0.0, 0.0],
            #                                     [0.0, -sigma_amplitude* t_global.value ]])
            # phaseFieldProblem.set_traction_bc(sigma_at_surface=sigma_at_surface,w=w,N=n,ds=ds_top_tagged(top_surface_tag))
            bcs_s = []
            if abs(t) > sys.float_info.epsilon * 5:
                bcs_s.append(pf.irreversibility_bc_staggered(domain,S,sm1))
            return bcs_u ,bcs_s

        Work = dlfx.fem.Constant(domain,0.0)
        
       
        dx = ufl.Measure("dx", domain=domain)
        vol = alex.homogenization.get_filled_vol(dx=dx,comm=comm)
        E_average = pp.get_volume_average_of_field(E,vol,dx=ufl.dx,comm=comm)
        
        def write_vol_data_to_file():
            if rank == 0:
                vol_path = os.path.join(folder_path, f"vol_{x_value}_{case}.json")
                volumes_data = {
                        "vol": vol,
                        "E_average": E_average,
                    }
                with open(vol_path, "w") as f:
                    json.dump(volumes_data, f, indent=4)
                print(f"Saved volume info to: {vol_path}")
        
        write_vol_data_to_file()
        
        def after_timestep_success(t, dt, iters):
            sigma = phaseFieldProblem.sigma_degraded(u, s, lam, mue, eta)
            # Reaction force at top boundary
            Rx_top, Ry_top = pp.reaction_force(sigma, n=n, ds=ds_top_tagged(top_surface_tag), comm=comm)

            # Get vertical displacement u_y at top boundary dofs
            if len(u.x.array[dofs_at_boundary_y]) > 0:
                u_y_top_local = u.x.array[dofs_at_boundary_y][0]
            else:
                u_y_top_local = 1e10

            comm.barrier()
            u_y_top = comm.allreduce(u_y_top_local, MPI.MIN)
            comm.barrier()




            
            # dW = pp.work_increment_external_forces(sigma,u,um1,n,ds_top_tagged(top_surface_tag),comm=comm)
            dW = pp.work_increment_external_forces(sigma,u,um1,n,ds,comm=comm)
            Work.value = Work.value + dW
    
            A = pf.get_surf_area(s,epsilon=epsilon,dx=ufl.dx, comm=comm)
    
            E_el = phaseFieldProblem.get_E_el_global(s,eta,u,lam,mue,dx=ufl.dx,comm=comm)
    
            if rank == 0:
                pp.write_to_graphs_output_file(outputfile_graph_path, t, u_y_top, Ry_top, dW, Work.value, A, E_el)

            if rank == 0:
                sol.write_to_newton_logfile(logfile_path, t, dt, iters)
            
            um1.x.array[:] = u.x.array[:] 
            sm1.x.array[:] = s.x.array[:]
            urestart.x.array[:] = u.x.array[:]
            srestart.x.array[:] = s.x.array[:]
            
                

            success_timestep_counter.value = success_timestep_counter.value + 1.0
            if int(success_timestep_counter.value) % int(postprocessing_interval.value) == 0:
                pp.write_vector_fields(domain,comm,[u],["u"],outputfile_xdmf_path=results_xdmf_path,t=t)
                #pp.write_phasefield_mixed_solution(domain, results_xdmf_path, w, t, comm)
                E.name = "E"
                pp.write_scalar_fields(domain, comm, [E,s], ["E","s"], outputfile_xdmf_path=results_xdmf_path, t=t)
                pp.write_tensor_fields(domain, comm, [sigma], ["sig"], outputfile_xdmf_path=results_xdmf_path, t=t)

        def after_timestep_restart(t, dt, iters):
            # If global dt has shrunk beyond tolerance -> write what we have and skip this case
            if dt_global.value < 10.0 ** (-14):
                if rank == 0:
                    print(f"[WARNING] NO CONVERGENCE (dt too small) in case '{case}' for dataset {x_value}. Skipping to next case.")
                # Signal to outer try/except to continue with next case
                raise RuntimeError("ConvergenceFailure")
            # Otherwise: restore previous state and let the solver retry with smaller dt
            u.x.array[:] = urestart.x.array[:]
            s.x.array[:] = srestart.x.array[:]
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
        # ---- Run solver, but keep going on convergence failure
        try:
            sol.solve_staggered(domain,
                                u,
                                s,
                                Tend,
                                dt_global,
                                before_first_timestep_hook=before_first_time_step,
                                before_each_timestep_hook=before_each_time_step,
                                after_last_timestep_hook=after_last_timestep,
                                get_residuum_and_gateaux_u=get_residuum_and_gateaux_u,
                                get_residuum_and_gateaux_s=get_residuum_and_gateaux_s,
                                get_bcs=get_bcs,
                                after_timestep_success_hook=after_timestep_success,
                                after_timestep_restart_hook=after_timestep_restart,
                                comm=comm,
                                print_bool=True,
                                trestart=trestart_global,
                                max_iters=20,
                                min_iters=4,
                                t=t_global,
                                dt_max=dt_max)
            log_convergence_status(x_value, case, "OK")
        except RuntimeError as e:
            if "ConvergenceFailure" in str(e):
                log_convergence_status(x_value, case, f"ConvergenceFailure at time {t_global.value}")
                pp.write_vector_fields(domain,comm,[u],["u"],outputfile_xdmf_path=results_xdmf_path,t=t_global.value+0.01)
                #pp.write_phasefield_mixed_solution(domain, results_xdmf_path, w, t, comm)
                E.name = "E"
                pp.write_scalar_fields(domain, comm, [E,s], ["E","s"], outputfile_xdmf_path=results_xdmf_path, t=t_global.value+0.01)
                continue  # skip to next case
            else:
                log_convergence_status(x_value, case, f"RuntimeError: {str(e)}")
                raise


