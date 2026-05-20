"""
Standalone version of run_effective_stiffness.py with all alex module dependencies inlined.
Original script: /home/scripts/046_plasticity/000_template/run_effective_stiffness.py
"""

import dolfinx as dlfx
from mpi4py import MPI
import ufl
import numpy as np
import os
import sys
import argparse
import glob
import basix
import gmsh
from dolfinx import io

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc

# ============================================================================
# UTILITIES (alex.util)
# ============================================================================

def get_dimension_of_function(f: dlfx.fem.Function) -> int:
    """Get the dimension of a function's value space."""
    return f.ufl_shape[0]


# ============================================================================
# OS UTILITIES (alex.os)
# ============================================================================

def logfile_full_path(script_path: str, script_name_without_extension: str) -> str:
    """Generate full path for log file."""
    return os.path.join(script_path, script_name_without_extension + "_log.txt")


def outputfile_graph_full_path(script_path: str, script_name_without_extension: str) -> str:
    """Generate full path for graphs output file."""
    return os.path.join(script_path, script_name_without_extension + "_graphs.txt")


def outputfile_xdmf_full_path(script_path: str, script_name_without_extension: str) -> str:
    """Generate full path for XDMF output file."""
    return os.path.join(script_path, script_name_without_extension + ".xdmf")


# ============================================================================
# LINEAR ELASTIC (alex.linearelastic)
# ============================================================================

def cmat_voigt_2D(lam: dlfx.fem.Constant, mu: dlfx.fem.Constant) -> any:
    """Constitutive matrix in Voigt notation for 2D."""
    return ufl.as_matrix([[lam+2*mu, lam, 0.0],
                          [lam, lam+2*mu, 0.0],
                          [0.0, 0.0, mu]])


def eps_voigt_2D(u: any) -> any:
    """Strain in Voigt notation for 2D."""
    return ufl.as_vector([u[0].dx(0), u[1].dx(1), u[0].dx(1)+u[1].dx(0)])


def cmat_voigt_3D(lam: dlfx.fem.Constant, mu: dlfx.fem.Constant) -> any:
    """Constitutive matrix in Voigt notation for 3D."""
    return ufl.as_matrix([[lam+2*mu, lam, lam, 0.0, 0.0, 0.0],
                          [lam, lam+2*mu, lam, 0.0, 0.0, 0.0],
                          [lam, lam, lam+2*mu, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, mu, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, mu, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, mu]])


def eps_voigt_3D(u: any) -> any:
    """Strain in Voigt notation for 3D."""
    return ufl.as_vector([u[0].dx(0), u[1].dx(1), u[2].dx(2), 
                          u[1].dx(2)+u[2].dx(1), u[2].dx(0)+u[0].dx(2), 
                          u[0].dx(1)+u[1].dx(0)])


def get_emod(lam: float, mu: float):
    """Calculate Young's modulus from Lamé parameters."""
    return mu * (3.0 * lam + 2.0 * mu) / (lam + mu)


def get_nu(lam: float, mu: float):
    """Calculate Poisson's ratio from Lamé parameters."""
    return lam / (2 * (lam + mu))


def sigma_as_tensor(u: dlfx.fem.Function, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant):
    """Stress tensor."""
    eps = ufl.sym(ufl.grad(u))
    dim = get_dimension_of_function(u)
    val = lam * ufl.tr(eps) * ufl.Identity(dim) + 2 * mu * eps
    return val


def sigma_as_voigt(u: dlfx.fem.Function, lam: dlfx.fem.Constant, mu: dlfx.fem.Constant):
    """Stress in Voigt notation."""
    dim = get_dimension_of_function(u)
    if dim == 3:  # 3D
        eps_voigt = eps_voigt_3D(u)
        sig_voigt = ufl.dot(cmat_voigt_3D(lam, mu), eps_voigt)
    elif dim == 2:  # 2D
        eps_voigt = eps_voigt_2D(u)
        sig_voigt = ufl.dot(cmat_voigt_2D(lam, mu), eps_voigt)
    return sig_voigt


def psiel(u: dlfx.fem.Function, sigma: any):
    """Elastic strain energy density."""
    return 0.5 * ufl.inner(sigma, ufl.sym(ufl.grad(u)))


class StaticLinearElasticProblem:
    """Defines a static linear elastic problem."""
    
    def __init__(self):
        self.traction = 0.0
    
    def prep_newton(self, u: any, du: ufl.TestFunction, ddu: ufl.TrialFunction, 
                   lam: dlfx.fem.Constant, mu: dlfx.fem.Constant, 
                   dx: ufl.Measure = ufl.dx):
        """Prepare residual and Gâteaux derivative for Newton solver."""
        pot = psiel(u, sigma_as_tensor(u, lam, mu)) * dx - self.traction
        equi = ufl.derivative(pot, u, du)
        Res = equi
        dResdw = ufl.derivative(Res, u, ddu)
        return [Res, dResdw]


# ============================================================================
# HOMOGENIZATION (alex.homogenization)
# ============================================================================

def unit_macro_strain_tensor_for_voigt_eps(domain: dlfx.mesh.Mesh, voigt_index: int):
    """Create unit macroscopic strain tensor for a given Voigt index."""
    if domain.topology.dim == 3:
        Eps_Voigt = np.zeros((6,))
        Eps_Voigt[voigt_index] = 1.0
        return dlfx.fem.Constant(domain, np.array([[Eps_Voigt[0], Eps_Voigt[5]/2.0, Eps_Voigt[4]/2.0],
                                                   [Eps_Voigt[5]/2.0, Eps_Voigt[1], Eps_Voigt[3]/2.0],
                                                   [Eps_Voigt[4]/2.0, Eps_Voigt[3]/2.0, Eps_Voigt[2]]]))
    elif domain.topology.dim == 2:
        Eps_Voigt = np.zeros((3,))
        Eps_Voigt[voigt_index] = 1.0
        return dlfx.fem.Constant(domain, np.array([[Eps_Voigt[0], Eps_Voigt[2]/2.0],
                                                   [Eps_Voigt[2]/2.0, Eps_Voigt[1]]]))


def compute_averaged_sigma(u, lam, mu, vol, dx: ufl.Measure = ufl.dx, 
                          comm: MPI.Intracomm = MPI.COMM_WORLD):
    """Compute averaged stress from displacement field."""
    if get_dimension_of_function(u) == 3:
        dim_voigt = 6
    else:
        dim_voigt = 3
    
    sigma_for_unit_strain = np.zeros((dim_voigt,))
    sigma_for_unit_strain_global = np.zeros((dim_voigt,))
    
    for k in range(len(sigma_for_unit_strain)):
        sigma_for_unit_strain[k] = dlfx.fem.assemble_scalar(
            dlfx.fem.form(sigma_as_voigt(u, lam, mu)[k] * dx)) / vol
        sigma_for_unit_strain_global[k] = comm.allreduce(
            sigma_for_unit_strain[k], op=MPI.SUM)
    
    return sigma_for_unit_strain_global


def lam_hom(cmat):
    """Extract homogenized Lamé parameter lambda."""
    if len(cmat[0]) == 6:
        data = np.array([cmat[0, 1], cmat[0, 2], cmat[1, 0], cmat[2, 0], cmat[1, 2], cmat[2, 1]])
    elif len(cmat[0]) == 3:
        data = np.array([cmat[0, 1], cmat[1, 0]])
    
    lam_hom = np.mean(data)
    lam_hom_std_dev = np.std(data, ddof=1)
    return lam_hom, lam_hom_std_dev


def mu_hom(cmat):
    """Extract homogenized shear modulus mu."""
    if len(cmat[0]) == 6:
        data = np.array([cmat[3, 3], cmat[4, 4], cmat[5, 5]])
    elif len(cmat[0]) == 3:
        data = np.array([cmat[2, 2]])
    
    mu_hom = np.mean(data)
    mu_hom_std_dev = np.std(data, ddof=1)
    return mu_hom, mu_hom_std_dev


def E_hom(cmat):
    """Compute homogenized Young's modulus."""
    lam_h, _ = lam_hom(cmat)
    mu_h, _ = mu_hom(cmat)
    return get_emod(lam_h, mu_h)


def nu_hom(cmat):
    """Compute homogenized Poisson's ratio."""
    lam_h, _ = lam_hom(cmat)
    mu_h, _ = mu_hom(cmat)
    return get_nu(lam_h, mu_h)


def average_of_values_that_should_be_zero_isotropic_hom(cmat):
    """Compute average of off-diagonal terms in homogenized stiffness."""
    if len(cmat[0]) == 6:
        data = np.array([[0.0, 0.0, 0.0, cmat[0, 3], cmat[0, 4], cmat[0, 5]],
                         [0.0, 0.0, 0.0, cmat[1, 3], cmat[1, 4], cmat[1, 5]],
                         [0.0, 0.0, 0.0, cmat[2, 3], cmat[2, 4], cmat[2, 5]],
                         [cmat[3, 0], cmat[3, 1], cmat[3, 2], 0.0, cmat[3, 4], cmat[3, 5]],
                         [cmat[4, 0], cmat[4, 1], cmat[4, 2], cmat[4, 3], 0.0, cmat[4, 5]],
                         [cmat[5, 0], cmat[5, 1], cmat[5, 2], cmat[5, 3], cmat[5, 4], 0.0]])
    elif len(cmat[0]) == 3:
        data = np.array([[0.0, 0.0, cmat[0, 2]],
                         [0.0, 0.0, cmat[1, 2]],
                         [cmat[2, 0], cmat[2, 1], 0.0]])
    
    values_that_should_be_zero_average = np.mean(data)
    values_that_should_be_zero_std = np.std(data, ddof=1)
    return values_that_should_be_zero_average, values_that_should_be_zero_std


def print_results(cmat):
    """Print homogenized material properties."""
    lam_h, lam_h_std_dev = lam_hom(cmat)
    mu_h, mu_h_std_dev = mu_hom(cmat)
    E_h = E_hom(cmat)
    nu_h = nu_hom(cmat)
    zero_avg, zero_std = average_of_values_that_should_be_zero_isotropic_hom(cmat)
    
    result_string = (
        f"Lam Hom: {lam_h:.4f}, Standard Deviation: {lam_h_std_dev:.4f}\n"
        f"Mu Hom: {mu_h:.4f}, Standard Deviation: {mu_h_std_dev:.4f}\n"
        f"E Hom: {E_h:.4f}\n"
        f"Nu Hom: {nu_h:.4f}\n"
        f"Average of Values That Should Be Zero: {zero_avg:.4f}, "
        f"Standard Deviation: {zero_std:.4f}"
    )
    return result_string


# ============================================================================
# BOUNDARY CONDITIONS (alex.boundaryconditions)
# ============================================================================

def get_dimensions(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm):
    """Get global min/max coordinates of domain."""
    gdim = domain.geometry.dim
    x_min = np.min(domain.geometry.x[:, 0])
    x_max = np.max(domain.geometry.x[:, 0])
    y_min = np.min(domain.geometry.x[:, 1])
    y_max = np.max(domain.geometry.x[:, 1])
    
    if gdim == 3:
        z_min = np.min(domain.geometry.x[:, 2])
        z_max = np.max(domain.geometry.x[:, 2])
    else:
        z_min = 0.0
        z_max = 0.0

    comm.Barrier()
    x_min_all = comm.allreduce(x_min, op=MPI.MIN)
    x_max_all = comm.allreduce(x_max, op=MPI.MAX)
    y_min_all = comm.allreduce(y_min, op=MPI.MIN)
    y_max_all = comm.allreduce(y_max, op=MPI.MAX)
    z_min_all = comm.allreduce(z_min, op=MPI.MIN)
    z_max_all = comm.allreduce(z_max, op=MPI.MAX)
    comm.Barrier()
    
    return x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all


def close_func(x, value, atol):
    """Check if values are close to target value."""
    if atol:
        return np.isclose(x, value, atol=atol)
    else:
        return np.isclose(x, value)


def linear_displacements(V: dlfx.fem.FunctionSpace, eps_mac: dlfx.fem.Constant):
    """Create linear displacement field from macroscopic strain."""
    u_D = dlfx.fem.Function(V)
    dim = get_dimension_of_function(u_D)
    if dim == 3:
        for k in range(dim):
            u_D.sub(k).interpolate(
                lambda x: eps_mac.value[k, 0]*x[0] + eps_mac.value[k, 1]*x[1] + eps_mac.value[k, 2]*x[2])
            u_D.x.scatter_forward()
    elif dim == 2:
        for k in range(dim):
            u_D.sub(k).interpolate(
                lambda x: eps_mac.value[k, 0]*x[0] + eps_mac.value[k, 1]*x[1])
            u_D.x.scatter_forward()
    return u_D


def define_dirichlet_bc_from_interpolated_function(domain: dlfx.mesh.Mesh,
                                                   desired_value_at_boundary_function: dlfx.fem.Function,
                                                   where_function,
                                                   functionSpace: dlfx.fem.FunctionSpace,
                                                   subspace_idx: int = -1) -> dlfx.fem.DirichletBC:
    """Define Dirichlet BC from interpolated function."""
    fdim = domain.topology.dim - 1
    facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, where_function)
    if subspace_idx < 0:
        dofs_at_boundary = dlfx.fem.locate_dofs_topological(functionSpace, fdim, facets_at_boundary)
    else:
        dofs_at_boundary = dlfx.fem.locate_dofs_topological(
            functionSpace.sub(subspace_idx), fdim, facets_at_boundary)
    bc: dlfx.fem.DirichletBC = dlfx.fem.dirichletbc(desired_value_at_boundary_function, dofs_at_boundary)
    return bc


def get_total_linear_displacement_boundary_condition_at_box(domain: dlfx.mesh.Mesh,
                                                           comm: MPI.Intercomm,
                                                           functionSpace: dlfx.fem.FunctionSpace,
                                                           eps_mac: dlfx.fem.Constant,
                                                           subspace_idx: int = -1,
                                                           atol: float = None):
    """Get complete set of linear displacement boundary conditions."""
    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    gdim = domain.geometry.dim

    def top(x):
        return close_func(x[1], y_max_all, atol)

    def bottom(x):
        return close_func(x[1], y_min_all, atol)

    def left(x):
        return close_func(x[0], x_min_all, atol)

    def right(x):
        return close_func(x[0], x_max_all, atol)

    def front(x):
        return close_func(x[2], z_max_all, atol)

    def back(x):
        return close_func(x[2], z_min_all, atol)

    bcs = []
    w_D = linear_displacements(V=functionSpace, eps_mac=eps_mac)

    # Apply boundaries based on mesh dimension
    boundary_functions = [top, bottom, left, right]
    if gdim == 3:
        boundary_functions.extend([front, back])

    for where_function in boundary_functions:
        bcs.append(define_dirichlet_bc_from_interpolated_function(
            domain, w_D, where_function, functionSpace, subspace_idx))

    return bcs


# ============================================================================
# POST-PROCESSING (alex.postprocessing)
# ============================================================================

def prepare_graphs_output_file(output_file_path: str):
    """Initialize graphs output file."""
    for file in glob.glob(output_file_path):
        os.remove(output_file_path)
    logfile = open(output_file_path, 'w')
    logfile.write('# This is a general outputfile for displaying scalar quantities vs time\n')
    logfile.close()
    return True


def write_meshoutputfile(domain: dlfx.mesh.Mesh, outputfile_path: str, comm: MPI.Intercomm, meshtags: any = None):
    """Write mesh to output file."""
    if outputfile_path.endswith(".xdmf"):
        with dlfx.io.XDMFFile(comm, outputfile_path, 'w') as xdmfout:
            xdmfout.write_mesh(domain)
            if meshtags is not None:
                xdmfout.write_meshtags(meshtags, domain.geometry)
    elif outputfile_path.endswith(".vtk"):
        with dlfx.io.VTKFile(comm, outputfile_path, 'w') as vtkout:
            vtkout.write_mesh(domain)
    else:
        return False
    return True


def write_vector_field(domain: dlfx.mesh.Mesh, outputfile_path: str, field: dlfx.fem.Function,
                      t: dlfx.fem.Constant, comm: MPI.Intercomm,
                      V: dlfx.fem.FunctionSpace = None, field_interp: dlfx.fem.Function = None):
    """Write vector field to output file."""
    if V is None and field_interp is None:
        Ve = basix.ufl.element("P", domain.basix_cell(), 1, shape=(domain.geometry.dim,))
        V = dlfx.fem.functionspace(domain, Ve)

    if field_interp is None:
        field_interp = dlfx.fem.Function(V)

    # Interpolate to vertices for output
    expr = dlfx.fem.Expression(field, V.element.interpolation_points())
    field_interp.interpolate(expr)
    field_interp.name = field.name

    # Write to file
    if outputfile_path.endswith(".xdmf"):
        with dlfx.io.XDMFFile(comm, outputfile_path, 'a') as xdmfout:
            xdmfout.write_function(field_interp, t)
    elif outputfile_path.endswith(".vtk"):
        with dlfx.io.VTKFile(comm, outputfile_path, 'a') as vtkout:
            vtkout.write_function(field_interp, t)


def write_field(domain: dlfx.mesh.Mesh, outputfile_path: str, field,
                t: dlfx.fem.Constant, comm: MPI.Intercomm,
                S: dlfx.fem.FunctionSpace = None,
                field_interp: dlfx.fem.Function = None):
    """Write a scalar expression/function to the output file."""
    if S is None and field_interp is None:
        Se = basix.ufl.element("P", domain.basix_cell(), 1, shape=())
        S = dlfx.fem.functionspace(domain, Se)

    if field_interp is None:
        field_interp = dlfx.fem.Function(S)

    expr = dlfx.fem.Expression(field, S.element.interpolation_points())
    field_interp.interpolate(expr)

    if hasattr(field, "name"):
        field_interp.name = field.name

    if outputfile_path.endswith(".xdmf"):
        with dlfx.io.XDMFFile(comm, outputfile_path, 'a') as xdmfout:
            xdmfout.write_function(field_interp, t)
    elif outputfile_path.endswith(".vtk"):
        with dlfx.io.VTKFile(comm, outputfile_path, 'a') as vtkout:
            vtkout.write_function(field_interp, t)


def write_tensor_fields(domain: dlfx.mesh.Mesh, comm: MPI.Intercomm,
                        tensor_fields_as_functions, tensor_field_names,
                        outputfile_xdmf_path: str, t: float):
    """Write tensor expressions/functions to XDMF."""
    dim = domain.topology.dim
    TEN = dlfx.fem.functionspace(domain, ("DP", 0, (dim, dim)))
    with dlfx.io.XDMFFile(comm, outputfile_xdmf_path, 'a') as xdmf_out:
        for tensor_field_function, tensor_field_name in zip(tensor_fields_as_functions, tensor_field_names):
            tensor_field_expression = dlfx.fem.Expression(
                tensor_field_function, TEN.element.interpolation_points())
            out_tensor_field = dlfx.fem.Function(TEN)
            out_tensor_field.interpolate(tensor_field_expression)
            out_tensor_field.name = tensor_field_name
            xdmf_out.write_function(out_tensor_field, t)


def compute_and_write_tensor_eigenvalue(domain: dlfx.mesh.Mesh,
                                        tensor,
                                        tensor_name: str,
                                        time: dlfx.fem.Constant,
                                        outputfile_path: str,
                                        comm: MPI.Intercomm):
    """Compute and write principal values of a symmetric 2D/3D tensor field."""
    dim = domain.topology.dim
    element = basix.ufl.element("P", domain.basix_cell(), 1, shape=())
    function_space = dlfx.fem.functionspace(domain, element)

    if dim == 2:
        a = tensor[0, 0]
        b = tensor[0, 1]
        c = tensor[1, 1]
        radius = 0.5 * ufl.sqrt(ufl.max_value((a - c)**2 + 4.0 * b**2, 0.0))
        center = 0.5 * (a + c)
        eigenvalues = [center + radius, center - radius]
    elif dim == 3:
        a = -1.0
        b = ufl.tr(tensor)
        c = -0.5 * (ufl.tr(tensor)**2 - ufl.tr(tensor * tensor))
        d = ufl.det(tensor)

        q = (3.0 * a * c - b**2) / (9.0 * a**2)
        r = (9.0 * a * b * c - 27.0 * a**2 * d - 2.0 * b**3) / (54.0 * a**3)

        eps = 1.0e-14
        sqrt_q = ufl.sqrt(ufl.max_value(-q, 0.0))
        cos_arg = r / ufl.sqrt(ufl.max_value(-q**3, eps))
        cos_arg = ufl.max_value(ufl.min_value(cos_arg, 1.0), -1.0)
        theta = ufl.acos(cos_arg)

        eigenvalues = [
            2.0 * sqrt_q * ufl.cos(theta / 3.0) - b / (3.0 * a),
            2.0 * sqrt_q * ufl.cos((theta + 2.0 * np.pi) / 3.0) - b / (3.0 * a),
            2.0 * sqrt_q * ufl.cos((theta + 4.0 * np.pi) / 3.0) - b / (3.0 * a),
        ]
    else:
        raise ValueError(f"Principal tensor output is only implemented for 2D/3D, got dim={dim}.")

    for i, eigenvalue in enumerate(eigenvalues, start=1):
        fn = dlfx.fem.Function(function_space)
        expr = dlfx.fem.Expression(eigenvalue, function_space.element.interpolation_points())
        fn.interpolate(expr)
        fn.name = f"{tensor_name}_principal_{i}"
        write_field(domain, outputfile_path, fn, time, comm, function_space)


def append_to_file(filename, parameters, comm=MPI.COMM_WORLD):
    """Append parameters to file."""
    if comm.Get_rank() == 0:
        with open(filename, 'a') as file:
            for key, value in parameters.items():
                file.write(f"{key}={value}\n")
                print(f"Appended to {filename}: {key}={value}")


# ============================================================================
# SOLUTION (alex.solution)
# ============================================================================

def prepare_newton_logfile(logfile_path: str):
    """Initialize Newton log file."""
    for file in glob.glob(logfile_path):
        os.remove(logfile_path)
    logfile = open(logfile_path, 'w')
    logfile.write('# time, dt, no. iterations (for convergence) \n')
    logfile.close()
    return True


def write_to_newton_logfile(logfile_path: str, t: float, dt: float, iters: int):
    """Write to Newton log file."""
    logfile = open(logfile_path, 'a')
    logfile.write(str(t) + '  ' + str(dt) + '  ' + str(iters) + '\n')
    logfile.close()
    return True


def write_runtime_to_newton_logfile(logfile_path: str, runtime: float):
    """Write runtime to Newton log file."""
    logfile = open(logfile_path, 'a')
    logfile.write('# \n')
    logfile.write('# elapsed time:  ' + str(runtime) + '\n')
    logfile.write('# \n')
    logfile.close()
    return True


def print_time_and_dt(t: float, dt: float):
    """Print current time and time step."""
    print(' ')
    print('==================================================')
    print('Computing solution at time = {0:.4e}'.format(t))
    print('==================================================')
    print('Current time step dt = {0:.4e}'.format(dt))
    print('==================================================')
    print(' ')
    sys.stdout.flush()
    return True


def print_runtime(runtime: float):
    """Print total runtime."""
    print('')
    print('-----------------------------')
    print('elapsed time:', runtime)
    print('-----------------------------')
    print('')
    sys.stdout.flush()
    return True


def get_solver(w, comm, max_iters, Res, dResdw, bcs):
    """Create Newton solver."""
    if dResdw is not None:
        problem = NonlinearProblem(Res, w, bcs, dResdw)
    else:
        problem = NonlinearProblem(Res, w, bcs)

    solver = NewtonSolver(comm, problem)
    solver.report = True
    solver.max_it = max_iters

    if comm.Get_rank() == 0:
        ksp = solver.krylov_solver
        print("Default KSP Type:", ksp.getType())
        print("Default PC Type:", ksp.getPC().getType())

    return solver, problem


def print_total_dofs(w, comm, rank):
    """Print total number of DOFs."""
    num_dofs = np.shape(w.x.array[:])[0]
    comm.Barrier()
    num_dofs_all = comm.allreduce(num_dofs, op=MPI.SUM)
    comm.Barrier()
    if rank == 0:
        print('solving fem problem with', num_dofs_all, 'dofs ...')
        sys.stdout.flush()


def default_hook():
    """Default empty hook."""
    return


def default_hook_tdt(t, dt):
    """Default empty hook with time and dt."""
    return


def default_hook_all(t, dt, iters):
    """Default empty hook with time, dt and iterations."""
    return


def print_timestep_overview(iters: int, converged: bool, restart_solution: bool):
    """Print timestep convergence overview."""
    print('-----------------------------')
    print(' No. of iterations: ', iters)
    print(' Converged:         ', converged)
    print(' Restarting:        ', restart_solution)
    print('-----------------------------')
    sys.stdout.flush()
    return True


def print_no_convergence(dt: float):
    """Print no convergence message."""
    print('-----------------------------')
    print('!!! NO CONVERGENCE => dt: ', dt)
    print('-----------------------------')
    sys.stdout.flush()
    return True


def print_increasing_dt(dt: float):
    """Print increasing dt message."""
    print('-----------------------------')
    print('!!! Increasing dt to: ', dt)
    print('-----------------------------')
    sys.stdout.flush()
    return True


def print_decreasing_dt(dt: float):
    """Print decreasing dt message."""
    print('-----------------------------')
    print('!!! Decreasing dt to: ', dt)
    print('-----------------------------')
    sys.stdout.flush()
    return True


def solve_with_newton_adaptive_time_stepping(
        domain: dlfx.mesh.Mesh,
        w: dlfx.fem.Function,
        Tend: float,
        dt: dlfx.fem.Constant,
        before_first_timestep_hook=default_hook,
        after_last_timestep_hook=default_hook,
        before_each_timestep_hook=default_hook_tdt,
        get_residuum_and_gateaux=None,
        get_bcs=None,
        after_timestep_success_hook=default_hook_all,
        after_timestep_restart_hook=default_hook_all,
        comm: MPI.Intercomm = MPI.COMM_WORLD,
        print_bool: bool = False,
        solver=None,
        t: dlfx.fem.Constant = None,
        dt_max: dlfx.fem.Constant = None,
        dt_never_scale_up: bool = False,
        trestart: dlfx.fem.Constant = None,
        max_iters=8,
        min_iters=4):
    """Solve with Newton method and adaptive time stepping."""
    rank = comm.Get_rank()

    if print_bool:
        print_total_dofs(w, comm, rank)

    dt_scale_down = 0.5
    dt_scale_up = 2.0

    if dt_max is not None:
        if dt.value >= dt_max.value:
            dt.value = dt_max.value

    if t is None:
        t = dlfx.fem.Constant(domain, 0.0)
    if trestart is None:
        trestart = dlfx.fem.Constant(domain, 0.0)

    before_first_timestep_hook()

    choose_default_solver_each_time_step = False
    if solver is None:
        choose_default_solver_each_time_step = True

    while t.value < Tend:
        before_each_timestep_hook(t.value, dt.value)

        [Res, dResdw] = get_residuum_and_gateaux(dt)
        bcs = get_bcs(t.value)

        if choose_default_solver_each_time_step:
            if comm.Get_rank() == 0:
                print(f"NO SOLVER PROVIDED. Default solver created each time step")
            solver, problem = get_solver(w, comm, max_iters, Res, dResdw, bcs)

        converged = False
        iters = 0
        try:
            (iters, converged) = solver.solve(w)
        except RuntimeError as e:
            if comm.Get_rank() == 0:
                print(e)
            dt.value = dt_scale_down * dt.value
            if rank == 0 and print_bool:
                print_no_convergence(dt.value)

        if converged:
            after_timestep_success_hook(t.value, dt.value, iters)

        if converged and iters < min_iters and t.value > np.finfo(float).eps and iters > 0:
            if not dt_never_scale_up:
                if dt_max is None:
                    dt.value = dt_scale_up * dt.value
                    if rank == 0 and print_bool:
                        print_increasing_dt(dt.value)
                else:
                    if not (dt_scale_up * dt.value > dt_max.value):
                        dt.value = dt_scale_up * dt.value
                        if rank == 0 and print_bool:
                            print_increasing_dt(dt.value)
                    else:
                        dt.value = dt_max.value

        restart_solution = False
        if converged:
            trestart.value = t.value
            t.value = t.value + dt.value
        else:
            restart_solution = True
            after_timestep_restart_hook(t.value, dt.value, iters)
            t.value = trestart.value + dt.value

        if rank == 0 and print_bool:
            print_timestep_overview(iters, converged, restart_solution)

    after_last_timestep_hook()


# ============================================================================
# MAIN SCRIPT
# ============================================================================

# Setup paths
script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = logfile_full_path(script_path, script_name_without_extension)
outputfile_graph_path = outputfile_graph_full_path(script_path, script_name_without_extension)
outputfile_xdmf_path = outputfile_xdmf_full_path(script_path, script_name_without_extension)
parameter_path = os.path.join(script_path, "parameters.txt")

# Remove existing parameters.txt if it exists
if os.path.exists(parameter_path):
    os.remove(parameter_path)

# Set and start stopwatch
timer = dlfx.common.Timer()
timer.start()

# Set MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print('MPI-STATUS: Process:', rank, 'of', size, 'processes.')
sys.stdout.flush()

# Parse arguments
parser = argparse.ArgumentParser(description="Run a simulation with specified parameters.")
try:
    parser.add_argument("--lam_micro_param", type=float, required=True, help="Lambda micro_parameter")
    parser.add_argument("--mue_micro_param", type=float, required=True, help="Mu micro_parameter")
    parser.add_argument("--mesh_size", type=float, default=0.01,
                        help="Target gmsh mesh size. Smaller values create a finer mesh.")
    args = parser.parse_args()
    lam_micro_param = args.lam_micro_param
    mue_micro_param = args.mue_micro_param
    mesh_size = args.mesh_size
    if mesh_size <= 0.0:
        raise ValueError("--mesh_size must be positive")
except:
    print("Could not parse arguments")
    lam_micro_param = 1.0
    mue_micro_param = 1.0
    mesh_size = 0.01

# Generate mesh using gmsh (same as fu_star.py)
msh_file = os.path.join(script_path, "fu_star.msh")

gmsh.initialize()
gmsh.model.add("fu_star")

#small star
hole_scale = 0.45
y_inner = gmsh.model.occ.addPoint(0, 0.2 * hole_scale, 0)
y_outer = gmsh.model.occ.addPoint(0.5 * hole_scale, 0.5 * hole_scale, 0)
x_inner = gmsh.model.occ.addPoint(0.2 * hole_scale, 0, 0)
origin = gmsh.model.occ.addPoint(0, 0, 0)

sl1 = gmsh.model.occ.addLine(origin, y_inner)
sl2 = gmsh.model.occ.addLine(y_inner, y_outer)
sl3 = gmsh.model.occ.addLine(y_outer, x_inner)
sl4 = gmsh.model.occ.addLine(x_inner, origin)

loop = gmsh.model.occ.addCurveLoop([sl1, sl2, sl3, sl4])
quarter = gmsh.model.occ.addPlaneSurface([loop])

# Mirror to create full small star
left = gmsh.model.occ.copy([(2, quarter)])
gmsh.model.occ.mirror(left, 1, 0, 0, 0)
half = gmsh.model.occ.copy([(2, quarter), left[0]])
gmsh.model.occ.mirror(half, 0, 1, 0, 0)

small_star_list = [(2, quarter), left[0], half[0], half[1]]

small_star_fused, _ = gmsh.model.occ.fuse([small_star_list[0]], small_star_list[1:], removeObject=True, removeTool=True)

#large star
t = 0.0625 #beam thickness

#points/parameters
hx1, vy2 = -0.5, 0.5
hx2, vy1 = -0.2, 0.2
s1_x, s1_y = -0.275, 0.225
s2_x, s2_y = -0.225, 0.275

#top left
h1 = gmsh.model.occ.addPoint(hx1, t, 0)
h2 = gmsh.model.occ.addPoint(hx2, t, 0)
v1 = gmsh.model.occ.addPoint(-t, vy1, 0)
v2 = gmsh.model.occ.addPoint(-t, vy2, 0)

slant1 = gmsh.model.occ.addPoint(s1_x, s1_y, 0)
slant2 = gmsh.model.occ.addPoint(s2_x, s2_y, 0)

L1 = gmsh.model.occ.addLine(h1, h2)
L2 = gmsh.model.occ.addLine(h2, slant1)
L3 = gmsh.model.occ.addLine(slant1, slant2)
L4 = gmsh.model.occ.addLine(slant2, v1)
L5 = gmsh.model.occ.addLine(v1, v2)

#top right
h1_m = gmsh.model.occ.addPoint(-hx1, t, 0)
h2_m = gmsh.model.occ.addPoint(-hx2, t, 0)
v1_m = gmsh.model.occ.addPoint(t, vy1, 0)
v2_m = gmsh.model.occ.addPoint(t, vy2, 0)

slant1_m = gmsh.model.occ.addPoint(-s1_x, s1_y, 0)
slant2_m = gmsh.model.occ.addPoint(-s2_x, s2_y, 0)

L1_m = gmsh.model.occ.addLine(h1_m, h2_m)
L2_m = gmsh.model.occ.addLine(h2_m, slant1_m)
L3_m = gmsh.model.occ.addLine(slant1_m, slant2_m)
L4_m = gmsh.model.occ.addLine(slant2_m, v1_m)
L5_m = gmsh.model.occ.addLine(v1_m, v2_m)

L6 = gmsh.model.occ.addLine(v2, v2_m) #top flat

#bottom right
h1_mr = gmsh.model.occ.addPoint(-hx1, -t, 0)
h2_mr = gmsh.model.occ.addPoint(-hx2, -t, 0)
v1_mr = gmsh.model.occ.addPoint(t, -vy1, 0)
v2_mr = gmsh.model.occ.addPoint(t, -vy2, 0)

slant1_mr = gmsh.model.occ.addPoint(-s1_x, -s1_y, 0)
slant2_mr = gmsh.model.occ.addPoint(-s2_x, -s2_y, 0)

L1_mr = gmsh.model.occ.addLine(h1_mr, h2_mr)
L2_mr = gmsh.model.occ.addLine(h2_mr, slant1_mr)
L3_mr = gmsh.model.occ.addLine(slant1_mr, slant2_mr)
L4_mr = gmsh.model.occ.addLine(slant2_mr, v1_mr)
L5_mr = gmsh.model.occ.addLine(v1_mr, v2_mr)

L9 = gmsh.model.occ.addLine(h1_m, h1_mr) #right flat

#bottom left
h1_ml = gmsh.model.occ.addPoint(hx1, -t, 0)
h2_ml = gmsh.model.occ.addPoint(hx2, -t, 0)
v1_ml = gmsh.model.occ.addPoint(-t, -vy1, 0)
v2_ml = gmsh.model.occ.addPoint(-t, -vy2, 0)

slant1_ml = gmsh.model.occ.addPoint(s1_x, -s1_y, 0)
slant2_ml = gmsh.model.occ.addPoint(s2_x, -s2_y, 0)

L1_ml = gmsh.model.occ.addLine(h1_ml, h2_ml)
L2_ml = gmsh.model.occ.addLine(h2_ml, slant1_ml)
L3_ml = gmsh.model.occ.addLine(slant1_ml, slant2_ml)
L4_ml = gmsh.model.occ.addLine(slant2_ml, v1_ml)
L5_ml = gmsh.model.occ.addLine(v1_ml, v2_ml)

L7 = gmsh.model.occ.addLine(v2_mr, v2_ml) #bottom flat
L8 = gmsh.model.occ.addLine(h1_ml, h1)    #left flat

curve = gmsh.model.occ.addCurveLoop([L1, L2, L3, L4, L5, L6,
                                     L5_m, L4_m, L3_m, L2_m, L1_m, L9,
                                     L1_mr, L2_mr, L3_mr, L4_mr, L5_mr, L7,
                                     L5_ml, L4_ml, L3_ml, L2_ml, L1_ml, L8])

surface = gmsh.model.occ.addPlaneSurface([curve])

final_shape, _ = gmsh.model.occ.cut([(2, surface)], small_star_fused, removeObject=True, removeTool=True)

gmsh.model.occ.synchronize()

final_surface_tag = final_shape[0][1]
gmsh.model.addPhysicalGroup(2, [final_surface_tag], 1)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
gmsh.model.mesh.generate(2)
gmsh.write(msh_file)

gmsh.finalize()

domain, cell_markers, facet_markers = io.gmshio.read_from_msh(msh_file, comm, gdim=2)

dt = dlfx.fem.Constant(domain, 0.05)
Tend = 16.0 * dt.value

# Elastic constants
lam = dlfx.fem.Constant(domain, lam_micro_param)
mu = dlfx.fem.Constant(domain, mue_micro_param)

E_mod = get_emod(lam.value, mu.value)

# Function space
Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
V = dlfx.fem.FunctionSpace(domain, Ve)

# Define solution fields
u = dlfx.fem.Function(V)
urestart = dlfx.fem.Function(V)
du = ufl.TestFunction(V)
ddu = ufl.TrialFunction(V)


def before_first_time_step():
    urestart.x.array[:] = np.ones_like(urestart.x.array[:])

    if rank == 0:
        prepare_newton_logfile(logfile_path)
        prepare_graphs_output_file(outputfile_graph_path)

    write_meshoutputfile(domain, outputfile_xdmf_path, comm)


def before_each_time_step(t, dt):
    if rank == 0:
        print_time_and_dt(t, dt)


linearElasticProblem = StaticLinearElasticProblem()


def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
    [Res, dResdw] = linearElasticProblem.prep_newton(u, du, ddu, lam, mu)
    return [Res, dResdw]


x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
atol = (x_max_all - x_min_all) * 0.00


def get_bcs(t):
    if column_of_cmat_computed[0] < 3:
        eps_mac = unit_macro_strain_tensor_for_voigt_eps(domain, column_of_cmat_computed[0])
    else:
        eps_mac = dlfx.fem.Constant(domain, np.array([[0.0, 0.0],
                                                       [0.0, 0.0]]))
    bcs = get_total_linear_displacement_boundary_condition_at_box(
        domain, comm, V, eps_mac=eps_mac, atol=atol)
    return bcs


simulation_result = np.array([0.0])
vol = (x_max_all - x_min_all) * (y_max_all - y_min_all)
Chom = np.zeros((3, 3))

column_of_cmat_computed = np.array([0])


def after_timestep_success(t, dt, iters):
    u.name = "u"
    write_vector_field(domain, outputfile_xdmf_path, u, t, comm)

    sigma = sigma_as_tensor(u, lam, mu)
    write_tensor_fields(domain, comm, [sigma], ["sigma"], outputfile_xdmf_path, t)
    compute_and_write_tensor_eigenvalue(domain, sigma, "sigma", t, outputfile_xdmf_path, comm)

    sigma_for_unit_strain = compute_averaged_sigma(u, lam, mu, vol)

    if rank == 0:
        if column_of_cmat_computed[0] < 3:
            Chom[column_of_cmat_computed[0], :] = sigma_for_unit_strain
        else:
            return

        print(column_of_cmat_computed[0])
        column_of_cmat_computed[0] = column_of_cmat_computed[0] + 1
        write_to_newton_logfile(logfile_path, t, dt, iters)

    urestart.x.array[:] = u.x.array[:]


def after_timestep_restart(t, dt, iters):
    u.x.array[:] = urestart.x.array[:]


def after_last_timestep():
    timer.stop()

    if rank == 0:
        print(np.array_str(Chom, precision=2))
        print(print_results(Chom))

        # Compute effective material properties
        lam_eff = lam_hom(Chom)[0]
        mu_eff = mu_hom(Chom)[0]
        E_eff = E_hom(Chom)
        nu_eff = nu_hom(Chom)

        # Print to console
        print('')
        print('=====================================')
        print('EFFECTIVE MATERIAL PROPERTIES:')
        print('=====================================')
        print(f'Young\'s Modulus (E): {E_eff:.6f}')
        print(f'Poisson\'s Ratio (ν): {nu_eff:.6f}')
        print(f'Lamé λ: {lam_eff:.6f}')
        print(f'Lamé μ: {mu_eff:.6f}')
        print('=====================================')

        parameters_to_write = {
            "mesh_size": mesh_size,
            "lam_effective": lam_eff,
            "mue_effective": mu_eff,
            "youngs_modulus_effective": E_eff,
            "poisson_ratio_effective": nu_eff
        }

        append_to_file(parameters=parameters_to_write, filename=parameter_path, comm=comm)

        runtime = timer.elapsed()
        print_runtime(runtime)
        write_runtime_to_newton_logfile(logfile_path, runtime)


# Main solver call
solve_with_newton_adaptive_time_stepping(
    domain,
    u,
    Tend,
    dt,
    before_first_timestep_hook=before_first_time_step,
    after_last_timestep_hook=after_last_timestep,
    before_each_timestep_hook=before_each_time_step,
    get_residuum_and_gateaux=get_residuum_and_gateaux,
    get_bcs=get_bcs,
    after_timestep_restart_hook=after_timestep_restart,
    after_timestep_success_hook=after_timestep_success,
    comm=comm,
    print_bool=True
)
