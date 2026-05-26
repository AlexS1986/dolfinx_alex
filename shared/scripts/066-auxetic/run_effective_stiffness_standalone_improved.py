#!/usr/bin/env python3
"""
Standalone effective stiffness computation for a 2D star-shaped cell.

Improvements:
- Prints Python, DOLFINx, PETSc, mpi4py, NumPy and Gmsh versions at startup.
- Works with old DOLFINx (`dolfinx.io.gmshio`) and newer DOLFINx (`dolfinx.io.gmsh`).
- Handles both old tuple return and new MeshData return from gmsh reader/model_to_mesh.
- Avoids required command-line parameters by using defaults, while still accepting overrides.
- Uses `basix.ufl.element` + `dolfinx.fem.functionspace`, which is compatible with newer DOLFINx.
- Builds the DOLFINx mesh directly from the Gmsh model when possible, avoiding a fragile .msh round trip.
- Builds the star from a centerline offset so `h` is its uniform full ligament thickness.
- Supports flat exterior star tips of length `b`.
"""

import os
import sys
import argparse
import glob
import platform
from packaging.version import Version

import numpy as np
import ufl
import basix
import gmsh

import dolfinx as dlfx
from dolfinx import io
from mpi4py import MPI
from petsc4py import PETSc
import petsc4py

from dolfinx.nls.petsc import NewtonSolver

try:
    from dolfinx.fem.petsc import NewtonSolverNonlinearProblem as _NonlinearProblem
except ImportError:
    # DOLFINx 0.7.x exposes the Newton-solver problem with this name.
    from dolfinx.fem.petsc import NonlinearProblem as _NonlinearProblem

import matplotlib.pyplot as plt
active_geometry_params = {}

from datetime import datetime

# ============================================================================
# VERSION / API COMPATIBILITY
# ============================================================================

def _version_of(module, fallback="unknown"):
    return getattr(module, "__version__", fallback)


def print_versions(comm=MPI.COMM_WORLD):
    if comm.rank != 0:
        return
    print("=====================================")
    print("RUNTIME VERSIONS")
    print("=====================================")
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.replace(os.linesep, ' ')}")
    print(f"Platform          : {platform.platform()}")
    print(f"DOLFINx version   : {_version_of(dlfx)}")
    print(f"UFL version       : {_version_of(ufl)}")
    print(f"Basix version     : {_version_of(basix)}")
    print(f"NumPy version     : {_version_of(np)}")
    print(f"mpi4py version    : {_version_of(MPI)}")
    print(f"petsc4py version  : {_version_of(petsc4py)}")
    print(f"PETSc version     : {PETSc.Sys.getVersion()}")
    print(f"Gmsh version      : {_version_of(gmsh)}")
    print("=====================================")
    sys.stdout.flush()


def get_gmshio_module():
    """
    DOLFINx changed `dolfinx.io.gmshio` to `dolfinx.io.gmsh` in newer releases.
    """
    try:
        from dolfinx.io import gmsh as gmshio
        return gmshio
    except Exception:
        from dolfinx.io import gmshio
        return gmshio


gmshio = get_gmshio_module()


def unpack_mesh_data(mesh_data):
    """
    Handles different DOLFINx gmsh reader return formats.

    Common cases:
      old: (domain, cell_tags, facet_tags)
      new: MeshData(mesh, cell_tags, facet_tags, ...)
      some builds: tuple length 6
    """

    print(f"gmsh reader returned type: {type(mesh_data)}")

    if isinstance(mesh_data, tuple):
        print(f"gmsh reader returned tuple length: {len(mesh_data)}")

        domain = mesh_data[0]
        cell_markers = mesh_data[1] if len(mesh_data) > 1 else None
        facet_markers = mesh_data[2] if len(mesh_data) > 2 else None

        return domain, cell_markers, facet_markers

    if hasattr(mesh_data, "mesh"):
        domain = mesh_data.mesh
        cell_markers = getattr(mesh_data, "cell_tags", None)
        facet_markers = getattr(mesh_data, "facet_tags", None)

        return domain, cell_markers, facet_markers

    raise RuntimeError(
        f"Unsupported gmsh reader return type: {type(mesh_data)}"
    )


def make_functionspace(domain, family="Lagrange", degree=1, shape=None):
    """
    Compatibility helper for old and new DOLFINx.
    """
    if shape is None:
        element = basix.ufl.element(family, domain.basix_cell(), degree)
    else:
        element = basix.ufl.element(family, domain.basix_cell(), degree, shape=shape)

    if hasattr(dlfx.fem, "functionspace"):
        return dlfx.fem.functionspace(domain, element)
    return dlfx.fem.FunctionSpace(domain, element)


def constant_float(c):
    return float(np.asarray(c.value))


# ============================================================================
# UTILITIES
# ============================================================================

def get_dimension_of_function(f: dlfx.fem.Function) -> int:
    return f.ufl_shape[0]


def logfile_full_path(script_path: str, script_name_without_extension: str) -> str:
    return os.path.join(script_path, script_name_without_extension + "_log.txt")


def outputfile_graph_full_path(script_path: str, script_name_without_extension: str) -> str:
    return os.path.join(script_path, script_name_without_extension + "_graphs.txt")


def outputfile_xdmf_full_path(script_path: str, script_name_without_extension: str) -> str:
    return os.path.join(script_path, script_name_without_extension + ".xdmf")


# ============================================================================
# LINEAR ELASTICITY
# ============================================================================

def cmat_voigt_2D(lam, mu):
    return ufl.as_matrix([[lam + 2 * mu, lam, 0.0],
                          [lam, lam + 2 * mu, 0.0],
                          [0.0, 0.0, mu]])


def eps_voigt_2D(u):
    return ufl.as_vector([u[0].dx(0), u[1].dx(1), u[0].dx(1) + u[1].dx(0)])


def cmat_voigt_3D(lam, mu):
    return ufl.as_matrix([[lam + 2 * mu, lam, lam, 0.0, 0.0, 0.0],
                          [lam, lam + 2 * mu, lam, 0.0, 0.0, 0.0],
                          [lam, lam, lam + 2 * mu, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, mu, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, mu, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, mu]])


def eps_voigt_3D(u):
    return ufl.as_vector([u[0].dx(0), u[1].dx(1), u[2].dx(2),
                          u[1].dx(2) + u[2].dx(1),
                          u[2].dx(0) + u[0].dx(2),
                          u[0].dx(1) + u[1].dx(0)])


def get_emod(lam: float, mu: float):
    return mu * (3.0 * lam + 2.0 * mu) / (lam + mu)


def get_nu(lam: float, mu: float):
    return lam / (2.0 * (lam + mu))


def sigma_as_tensor(u, lam, mu):
    eps = ufl.sym(ufl.grad(u))
    dim = get_dimension_of_function(u)
    return lam * ufl.tr(eps) * ufl.Identity(dim) + 2 * mu * eps


def sigma_as_voigt(u, lam, mu):
    dim = get_dimension_of_function(u)
    if dim == 3:
        return ufl.dot(cmat_voigt_3D(lam, mu), eps_voigt_3D(u))
    if dim == 2:
        return ufl.dot(cmat_voigt_2D(lam, mu), eps_voigt_2D(u))
    raise RuntimeError(f"Unsupported displacement dimension: {dim}")


def psiel(u, sigma):
    return 0.5 * ufl.inner(sigma, ufl.sym(ufl.grad(u)))


class StaticLinearElasticProblem:
    def __init__(self):
        self.traction = 0.0

    def prep_newton(self, u, du, ddu, lam, mu, dx=ufl.dx):
        pot = psiel(u, sigma_as_tensor(u, lam, mu)) * dx - self.traction
        res = ufl.derivative(pot, u, du)
        jac = ufl.derivative(res, u, ddu)
        return res, jac


# ============================================================================
# HOMOGENIZATION
# ============================================================================

def unit_macro_strain_tensor_for_voigt_eps(domain, voigt_index: int):
    if domain.topology.dim == 3:
        epsv = np.zeros(6)
        epsv[voigt_index] = 1.0
        return dlfx.fem.Constant(domain, np.array([[epsv[0], epsv[5] / 2.0, epsv[4] / 2.0],
                                                   [epsv[5] / 2.0, epsv[1], epsv[3] / 2.0],
                                                   [epsv[4] / 2.0, epsv[3] / 2.0, epsv[2]]], dtype=PETSc.ScalarType))
    if domain.topology.dim == 2:
        epsv = np.zeros(3)
        epsv[voigt_index] = 1.0
        return dlfx.fem.Constant(domain, np.array([[epsv[0], epsv[2] / 2.0],
                                                   [epsv[2] / 2.0, epsv[1]]], dtype=PETSc.ScalarType))
    raise RuntimeError(f"Unsupported mesh dimension: {domain.topology.dim}")


def compute_averaged_sigma(u, lam, mu, vol, dx=ufl.dx, comm=MPI.COMM_WORLD):
    dim_voigt = 6 if get_dimension_of_function(u) == 3 else 3
    local = np.zeros(dim_voigt)
    global_ = np.zeros(dim_voigt)

    for k in range(dim_voigt):
        local[k] = dlfx.fem.assemble_scalar(dlfx.fem.form(sigma_as_voigt(u, lam, mu)[k] * dx)) / vol
        global_[k] = comm.allreduce(local[k], op=MPI.SUM)
    return global_


def lam_hom(cmat):
    if cmat.shape[0] == 6:
        data = np.array([cmat[0, 1], cmat[0, 2], cmat[1, 0], cmat[2, 0], cmat[1, 2], cmat[2, 1]])
    else:
        data = np.array([cmat[0, 1], cmat[1, 0]])
    return np.mean(data), np.std(data, ddof=1) if len(data) > 1 else 0.0


def mu_hom(cmat):
    if cmat.shape[0] == 6:
        data = np.array([cmat[3, 3], cmat[4, 4], cmat[5, 5]])
    else:
        data = np.array([cmat[2, 2]])
    return np.mean(data), np.std(data, ddof=1) if len(data) > 1 else 0.0


def E_hom(cmat):
    lam_h, _ = lam_hom(cmat)
    mu_h, _ = mu_hom(cmat)
    return get_emod(lam_h, mu_h)


def nu_hom(cmat):
    lam_h, _ = lam_hom(cmat)
    mu_h, _ = mu_hom(cmat)
    return get_nu(lam_h, mu_h)


def average_of_values_that_should_be_zero_isotropic_hom(cmat):
    if cmat.shape[0] == 6:
        data = np.array([[0.0, 0.0, 0.0, cmat[0, 3], cmat[0, 4], cmat[0, 5]],
                         [0.0, 0.0, 0.0, cmat[1, 3], cmat[1, 4], cmat[1, 5]],
                         [0.0, 0.0, 0.0, cmat[2, 3], cmat[2, 4], cmat[2, 5]],
                         [cmat[3, 0], cmat[3, 1], cmat[3, 2], 0.0, cmat[3, 4], cmat[3, 5]],
                         [cmat[4, 0], cmat[4, 1], cmat[4, 2], cmat[4, 3], 0.0, cmat[4, 5]],
                         [cmat[5, 0], cmat[5, 1], cmat[5, 2], cmat[5, 3], cmat[5, 4], 0.0]])
    else:
        data = np.array([[0.0, 0.0, cmat[0, 2]],
                         [0.0, 0.0, cmat[1, 2]],
                         [cmat[2, 0], cmat[2, 1], 0.0]])
    flat = data.ravel()
    return np.mean(flat), np.std(flat, ddof=1)


def print_results(cmat):
    lam_h, lam_h_std = lam_hom(cmat)
    mu_h, mu_h_std = mu_hom(cmat)
    zero_avg, zero_std = average_of_values_that_should_be_zero_isotropic_hom(cmat)
    return (
        f"Lam Hom: {lam_h:.4f}, Standard Deviation: {lam_h_std:.4f}\n"
        f"Mu Hom: {mu_h:.4f}, Standard Deviation: {mu_h_std:.4f}\n"
        f"E Hom: {E_hom(cmat):.4f}\n"
        f"Nu Hom: {nu_hom(cmat):.4f}\n"
        f"Average of Values That Should Be Zero: {zero_avg:.4f}, Standard Deviation: {zero_std:.4f}"
    )


# ============================================================================
# BOUNDARY CONDITIONS
# ============================================================================

def get_dimensions(domain, comm):
    gdim = domain.geometry.dim
    x_min = np.min(domain.geometry.x[:, 0])
    x_max = np.max(domain.geometry.x[:, 0])
    y_min = np.min(domain.geometry.x[:, 1])
    y_max = np.max(domain.geometry.x[:, 1])
    z_min = np.min(domain.geometry.x[:, 2]) if gdim == 3 else 0.0
    z_max = np.max(domain.geometry.x[:, 2]) if gdim == 3 else 0.0

    return (
        comm.allreduce(x_min, op=MPI.MIN),
        comm.allreduce(x_max, op=MPI.MAX),
        comm.allreduce(y_min, op=MPI.MIN),
        comm.allreduce(y_max, op=MPI.MAX),
        comm.allreduce(z_min, op=MPI.MIN),
        comm.allreduce(z_max, op=MPI.MAX),
    )


def close_func(x, value, atol):
    return np.isclose(x, value, atol=atol) if atol is not None else np.isclose(x, value)


def linear_displacements(V, eps_mac):
    u_D = dlfx.fem.Function(V)
    dim = get_dimension_of_function(u_D)
    if dim == 3:
        for k in range(dim):
            u_D.sub(k).interpolate(lambda x, k=k: eps_mac.value[k, 0] * x[0] + eps_mac.value[k, 1] * x[1] + eps_mac.value[k, 2] * x[2])
            u_D.x.scatter_forward()
    elif dim == 2:
        for k in range(dim):
            u_D.sub(k).interpolate(lambda x, k=k: eps_mac.value[k, 0] * x[0] + eps_mac.value[k, 1] * x[1])
            u_D.x.scatter_forward()
    else:
        raise RuntimeError(f"Unsupported displacement dimension: {dim}")
    return u_D


def define_dirichlet_bc_from_interpolated_function(domain, desired_value_at_boundary_function, where_function, functionSpace, subspace_idx=-1):
    fdim = domain.topology.dim - 1
    facets = dlfx.mesh.locate_entities_boundary(domain, fdim, where_function)
    if subspace_idx < 0:
        dofs = dlfx.fem.locate_dofs_topological(functionSpace, fdim, facets)
    else:
        dofs = dlfx.fem.locate_dofs_topological(functionSpace.sub(subspace_idx), fdim, facets)
    return dlfx.fem.dirichletbc(desired_value_at_boundary_function, dofs)


def get_total_linear_displacement_boundary_condition_at_box(domain, comm, functionSpace, eps_mac, subspace_idx=-1, atol=None):
    x_min, x_max, y_min, y_max, z_min, z_max = get_dimensions(domain, comm)
    gdim = domain.geometry.dim

    boundary_functions = [
        lambda x: close_func(x[1], y_max, atol),
        lambda x: close_func(x[1], y_min, atol),
        lambda x: close_func(x[0], x_min, atol),
        lambda x: close_func(x[0], x_max, atol),
    ]
    if gdim == 3:
        boundary_functions += [
            lambda x: close_func(x[2], z_max, atol),
            lambda x: close_func(x[2], z_min, atol),
        ]

    w_D = linear_displacements(functionSpace, eps_mac)
    return [
        define_dirichlet_bc_from_interpolated_function(domain, w_D, where_function, functionSpace, subspace_idx)
        for where_function in boundary_functions
    ]


# ============================================================================
# POST-PROCESSING / SOLVER HELPERS
# ============================================================================

def prepare_graphs_output_file(output_file_path):
    for file in glob.glob(output_file_path):
        os.remove(file)
    with open(output_file_path, "w") as logfile:
        logfile.write("# This is a general outputfile for displaying scalar quantities vs time\n")
    return True


def write_meshoutputfile(domain, outputfile_path, comm, meshtags=None):
    if outputfile_path.endswith(".xdmf"):
        with dlfx.io.XDMFFile(comm, outputfile_path, "w") as xdmfout:
            xdmfout.write_mesh(domain)
            if meshtags is not None:
                xdmfout.write_meshtags(meshtags, domain.geometry)
    elif outputfile_path.endswith(".vtk"):
        with dlfx.io.VTKFile(comm, outputfile_path, "w") as vtkout:
            vtkout.write_mesh(domain)
    else:
        return False
    return True


def write_vector_field(domain, outputfile_path, field, t, comm, V=None, field_interp=None):
    if V is None and field_interp is None:
        V = make_functionspace(domain, "Lagrange", 1, shape=(domain.geometry.dim,))
    if field_interp is None:
        field_interp = dlfx.fem.Function(V)

    points = V.element.interpolation_points
    if callable(points):
        points = points()

    expr = dlfx.fem.Expression(field, points)
    field_interp.interpolate(expr)
    field_interp.name = field.name

    time_value = float(t)
    if outputfile_path.endswith(".xdmf"):
        with dlfx.io.XDMFFile(comm, outputfile_path, "a") as xdmfout:
            xdmfout.write_function(field_interp, time_value)
    elif outputfile_path.endswith(".vtk"):
        with dlfx.io.VTKFile(comm, outputfile_path, "a") as vtkout:
            vtkout.write_function(field_interp, time_value)


def write_field(domain, outputfile_path, field,
                t, comm,
                S=None,
                field_interp=None):

    if S is None and field_interp is None:
        S = make_functionspace(domain, "Lagrange", 1)

    if field_interp is None:
        field_interp = dlfx.fem.Function(S)

    points = S.element.interpolation_points
    if callable(points):
        points = points()

    expr = dlfx.fem.Expression(field, points)
    field_interp.interpolate(expr)

    if hasattr(field, "name"):
        field_interp.name = field.name

    time_value = float(t)

    if outputfile_path.endswith(".xdmf"):
        with dlfx.io.XDMFFile(comm, outputfile_path, "a") as xdmfout:
            xdmfout.write_function(field_interp, time_value)

    elif outputfile_path.endswith(".vtk"):
        with dlfx.io.VTKFile(comm, outputfile_path, "a") as vtkout:
            vtkout.write_function(field_interp, time_value)


def write_tensor_fields(domain, comm,
                        tensor_fields_as_functions,
                        tensor_field_names,
                        outputfile_xdmf_path,
                        t):

    dim = domain.topology.dim

    TEN = make_functionspace(domain, "DG", 0, shape=(dim, dim))

    with dlfx.io.XDMFFile(comm, outputfile_xdmf_path, "a") as xdmf_out:

        for tensor_field_function, tensor_field_name in zip(
                tensor_fields_as_functions,
                tensor_field_names):

            points = TEN.element.interpolation_points
            if callable(points):
                points = points()

            tensor_expr = dlfx.fem.Expression(
                tensor_field_function,
                points
            )

            out_tensor_field = dlfx.fem.Function(TEN)

            out_tensor_field.interpolate(tensor_expr)

            out_tensor_field.name = tensor_field_name

            xdmf_out.write_function(out_tensor_field, float(t))


def compute_and_write_tensor_eigenvalue(domain,
                                        tensor,
                                        tensor_name,
                                        time,
                                        outputfile_path,
                                        comm):

    dim = domain.topology.dim

    S = make_functionspace(domain, "Lagrange", 1)

    if dim == 2:

        a = tensor[0, 0]
        b = tensor[0, 1]
        c = tensor[1, 1]

        radius = 0.5 * ufl.sqrt(
            ufl.max_value((a - c)**2 + 4.0 * b**2, 0.0)
        )

        center = 0.5 * (a + c)

        eigenvalues = [
            center + radius,
            center - radius
        ]

    else:
        raise RuntimeError("Only 2D implemented")

    for i, eigenvalue in enumerate(eigenvalues):

        fn = dlfx.fem.Function(S)

        points = S.element.interpolation_points
        if callable(points):
            points = points()

        expr = dlfx.fem.Expression(
            eigenvalue,
            points
        )

        fn.interpolate(expr)

        fn.name = f"{tensor_name}_principal_{i+1}"

        write_field(
            domain,
            outputfile_path,
            fn,
            time,
            comm,
            S
        )



def append_to_file(filename, parameters, comm=MPI.COMM_WORLD):
    if comm.rank == 0:
        with open(filename, "a") as file:
            for key, value in parameters.items():
                file.write(f"{key}={value}\n")
                print(f"Appended to {filename}: {key}={value}")


def prepare_newton_logfile(logfile_path):
    for file in glob.glob(logfile_path):
        os.remove(file)
    with open(logfile_path, "w") as logfile:
        logfile.write("# time, dt, no. iterations (for convergence)\n")
    return True


def write_to_newton_logfile(logfile_path, t, dt, iters):
    with open(logfile_path, "a") as logfile:
        logfile.write(f"{t}  {dt}  {iters}\n")
    return True


def write_runtime_to_newton_logfile(logfile_path, runtime):
    with open(logfile_path, "a") as logfile:
        logfile.write("#\n")
        logfile.write(f"# elapsed time: {runtime}\n")
        logfile.write("#\n")
    return True


def print_time_and_dt(t, dt):
    print(" ")
    print("==================================================")
    print(f"Computing solution at time = {t:.4e}")
    print("==================================================")
    print(f"Current time step dt = {dt:.4e}")
    print("==================================================")
    print(" ")
    sys.stdout.flush()
    return True


def print_runtime(runtime):
    print("")
    print("-----------------------------")
    print("elapsed time:", runtime)
    print("-----------------------------")
    print("")
    sys.stdout.flush()


def get_solver(w, comm, max_iters, Res, dResdw, bcs):
    """Create Newton solver compatible with dolfinx.nls.petsc.NewtonSolver."""

    problem = _NonlinearProblem(
        Res,
        w,
        bcs=bcs,
        J=dResdw
    )

    solver = NewtonSolver(comm, problem)
    solver.report = True
    solver.max_it = max_iters

   # if comm.Get_rank() == 0:
   #     ksp = solver.krylov_solver
   #     print("Default KSP Type:", ksp.getType())
   #     print("Default PC Type:", ksp.getPC().getType())

    return solver, problem


def print_total_dofs(w, comm):
    num_dofs = np.shape(w.x.array[:])[0]
    num_dofs_all = comm.allreduce(num_dofs, op=MPI.SUM)
    if comm.rank == 0:
        print("solving fem problem with", num_dofs_all, "dofs ...")
        sys.stdout.flush()


def solve_with_newton_adaptive_time_stepping(
        domain, w, Tend, dt,
        before_first_timestep_hook=lambda: None,
        after_last_timestep_hook=lambda: None,
        before_each_timestep_hook=lambda t, dt: None,
        get_residuum_and_gateaux=None,
        get_bcs=None,
        after_timestep_success_hook=lambda t, dt, iters: None,
        after_timestep_restart_hook=lambda t, dt, iters: None,
        comm=MPI.COMM_WORLD,
        print_bool=False,
        solver=None,
        t=None,
        dt_max=None,
        dt_never_scale_up=False,
        trestart=None,
        max_iters=8,
        min_iters=4):

    rank = comm.rank
    if print_bool:
        print_total_dofs(w, comm)

    dt_scale_down = 0.5
    dt_scale_up = 2.0

    if t is None:
        t = dlfx.fem.Constant(domain, PETSc.ScalarType(0.0))
    if trestart is None:
        trestart = dlfx.fem.Constant(domain, PETSc.ScalarType(0.0))

    if dt_max is not None and constant_float(dt) >= constant_float(dt_max):
        dt.value = dt_max.value

    before_first_timestep_hook()

    choose_default_solver_each_time_step = solver is None

    while constant_float(t) < Tend - 1.0e-14:
        before_each_timestep_hook(constant_float(t), constant_float(dt))

        Res, dResdw = get_residuum_and_gateaux(dt)
        bcs = get_bcs(constant_float(t))

        if choose_default_solver_each_time_step:
            #if rank == 0:
           #     print("NO SOLVER PROVIDED. Default solver created each time step")
            solver, _ = get_solver(w, comm, max_iters, Res, dResdw, bcs)

        converged = False
        iters = 0
        try:
            iters, converged = solver.solve(w)
            w.x.scatter_forward()
        except RuntimeError as e:
            if rank == 0:
                print(e)
            dt.value = PETSc.ScalarType(dt_scale_down * constant_float(dt))
            if rank == 0 and print_bool:
                print("-----------------------------")
                print("!!! NO CONVERGENCE => dt:", constant_float(dt))
                print("-----------------------------")

        if converged:
            after_timestep_success_hook(constant_float(t), constant_float(dt), iters)

        if converged and iters < min_iters and constant_float(t) > np.finfo(float).eps and iters > 0:
            if not dt_never_scale_up:
                proposed = dt_scale_up * constant_float(dt)
                if dt_max is None or proposed <= constant_float(dt_max):
                    dt.value = PETSc.ScalarType(proposed)
                    if rank == 0 and print_bool:
                        print("-----------------------------")
                        print("!!! Increasing dt to:", constant_float(dt))
                        print("-----------------------------")
                elif dt_max is not None:
                    dt.value = dt_max.value

        restart_solution = False
        if converged:
            trestart.value = t.value
            t.value = PETSc.ScalarType(constant_float(t) + constant_float(dt))
        else:
            restart_solution = True
            after_timestep_restart_hook(constant_float(t), constant_float(dt), iters)
            t.value = PETSc.ScalarType(constant_float(trestart) + constant_float(dt))

      #  if rank == 0 and print_bool:
      #      print("-----------------------------")
      #      print(" No. of iterations: ", iters)
      #      print(" Converged:         ", converged)
      #      print(" Restarting:        ", restart_solution)
      #      print("-----------------------------")
      #      sys.stdout.flush()

    after_last_timestep_hook()


# ============================================================================
# MESH GENERATION
# ============================================================================

def offset_closed_polygon(vertices, distance):
    """Return mitered vertices at signed normal distance from a CCW polygon."""
    offset_vertices = []
    for i, vertex in enumerate(vertices):
        p_prev = vertices[i - 1]
        p_next = vertices[(i + 1) % len(vertices)]
        d_prev = vertex - p_prev
        d_next = p_next - vertex
        d_prev /= np.linalg.norm(d_prev)
        d_next /= np.linalg.norm(d_next)
        n_prev = np.array([d_prev[1], -d_prev[0]])
        n_next = np.array([d_next[1], -d_next[0]])
        line_prev = vertex + distance * n_prev
        line_next = vertex + distance * n_next
        cross = d_prev[0] * d_next[1] - d_prev[1] * d_next[0]
        step = line_next - line_prev
        parameter = (step[0] * d_next[1] - step[1] * d_next[0]) / cross
        offset_vertices.append(line_prev + parameter * d_prev)
    return np.asarray(offset_vertices)


def flatten_polygon_corners(vertices, corner_indices, flat_length):
    """Replace selected polygon vertices with centered line caps of a set length."""
    if flat_length <= 0.0:
        return vertices

    corner_indices = set(corner_indices)
    flattened = []
    for i, vertex in enumerate(vertices):
        if i not in corner_indices:
            flattened.append(vertex)
            continue
        to_prev = vertices[i - 1] - vertex
        to_next = vertices[(i + 1) % len(vertices)] - vertex
        to_prev /= np.linalg.norm(to_prev)
        to_next /= np.linalg.norm(to_next)
        sine_half_angle = np.sqrt((1.0 - np.dot(to_prev, to_next)) / 2.0)
        trim = flat_length / (2.0 * sine_half_angle)
        if trim >= min(
            np.linalg.norm(vertices[i - 1] - vertex),
            np.linalg.norm(vertices[(i + 1) % len(vertices)] - vertex),
        ):
            raise ValueError("b is too large for the selected star dimensions")
        flattened.extend([vertex + trim * to_prev, vertex + trim * to_next])
    return np.asarray(flattened)


def add_polygon_surface(vertices, holes=()):
    loops = []
    for polygon in (vertices, *holes):
        points = [gmsh.model.occ.addPoint(float(x), float(y), 0.0) for x, y in polygon]
        lines = [
            gmsh.model.occ.addLine(points[i], points[(i + 1) % len(points)])
            for i in range(len(points))
        ]
        loops.append(gmsh.model.occ.addCurveLoop(lines))
    return gmsh.model.occ.addPlaneSurface(loops)


def build_gmsh_star_model(d1=0.20, L=0.1, h=0.2, b=0.04, mesh_size=0.02):
    global active_geometry_params

    if not (0.0 < d1 < 0.5):
        raise ValueError("d1 must be between 0 and 0.5")
    if L <= 0.0 or h <= 0.0 or mesh_size <= 0.0 or b < 0.0:
        raise ValueError("L, h and mesh_size must be positive, and b non-negative")

    gmsh.model.add("fu_star")
    d2 = d1 + L / np.sqrt(2.0)
    if d2 >= 0.5:
        raise ValueError("d1 + L / sqrt(2) must be below 0.5")

    # CHANGED FROM run_effective_stiffness_standalone.py:
    # The original star was assembled from hand-positioned inner/outer points,
    # so its local wall thickness was not consistently h. This version offsets
    # one centerline by +/-h/2 for uniform ligament thickness; b only creates
    # the four flat exterior tip caps and does not alter the ligament width.
    # The ring is centered on this eight-segment star. Offsetting each side by
    # h / 2 gives exactly h thickness along all straight star ligaments.
    centerline = np.array([
        [d1, 0.0], [d2, d2], [0.0, d1], [-d2, d2],
        [-d1, 0.0], [-d2, -d2], [0.0, -d1], [d2, -d2],
    ])
    outer = offset_closed_polygon(centerline, h / 2.0)
    outer = flatten_polygon_corners(outer, (1, 3, 5, 7), b)
    inner = offset_closed_polygon(centerline, -h / 2.0)
    ring = add_polygon_surface(outer, holes=(inner,))

    # The four connections to the representative-cell boundary are beams with
    # the same full thickness h as the central star ring.
    half_h = h / 2.0
    arms = [
        gmsh.model.occ.addRectangle(d1, -half_h, 0.0, 0.5 - d1, h),
        gmsh.model.occ.addRectangle(-0.5, -half_h, 0.0, 0.5 - d1, h),
        gmsh.model.occ.addRectangle(-half_h, d1, 0.0, h, 0.5 - d1),
        gmsh.model.occ.addRectangle(-half_h, -0.5, 0.0, h, 0.5 - d1),
    ]
    final_shape, _ = gmsh.model.occ.fuse(
        [(2, ring)], [(2, arm) for arm in arms], removeObject=True, removeTool=True
    )
    gmsh.model.occ.synchronize()

    surfaces = [tag for dim, tag in final_shape if dim == 2]
    gmsh.model.addPhysicalGroup(2, surfaces, 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    gmsh.model.mesh.generate(2)

    active_geometry_params = {
        "d1": d1,
        "d2": d2,
        "h": h,
        "b": b,
        "L": L,
        "mesh_size": mesh_size,
    }


def create_domain_from_gmsh(comm, script_path, d1=0.20, L=0.3, h=0.030, b=0.04,
                            mesh_size=0.01, write_msh=True):
    msh_file = os.path.join(script_path, "fu_star.msh")
    gmsh.initialize()
    try:
        build_gmsh_star_model(d1=d1, L=L, h=h, b=b, mesh_size=mesh_size)

        if write_msh and comm.rank == 0:
            gmsh.write(msh_file)

        # Best path: direct model_to_mesh; no .msh import API differences.
        if hasattr(gmshio, "model_to_mesh"):
            data = gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=2)
            return unpack_mesh_data(data)

        # Fallback for older versions with only read_from_msh.
        if comm.rank == 0:
            gmsh.write(msh_file)
        comm.barrier()
        if hasattr(gmshio, "read_from_msh"):
            try:
                data = gmshio.read_from_msh(msh_file, comm, rank=0, gdim=2)
            except TypeError:
                data = gmshio.read_from_msh(msh_file, comm, gdim=2)
            return unpack_mesh_data(data)

        raise RuntimeError("Neither gmshio.model_to_mesh nor gmshio.read_from_msh is available in this DOLFINx install.")
    finally:
        gmsh.finalize()


# ============================================================================
# MAIN
# ============================================================================

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    parser = argparse.ArgumentParser(description="Run effective stiffness homogenization.")
    parser.add_argument("--lam_micro_param", type=float, default=1.0, help="Microscopic Lame lambda. Default: 1.0")
    parser.add_argument("--mue_micro_param", type=float, default=1.0, help="Microscopic Lame mu. Default: 1.0")
    parser.add_argument("--d1", type=float, default=0.25, help="Axis radius of the star centerline")
    parser.add_argument("--L", type=float, default=0.15, help="Parameter defining the diagonal star radius")
    parser.add_argument("--h", type=float, default=0.03, help="Uniform full star-ligament thickness")
    parser.add_argument("--b", type=float, default=0.04, help="Length of each flat exterior star tip")
    parser.add_argument("--mesh_size", type=float, default=0.01, help="Target mesh size")
    args = parser.parse_args()

    lam_micro_param = args.lam_micro_param
    mue_micro_param = args.mue_micro_param
    d1 = args.d1
    L = args.L
    h = args.h
    b = args.b
    mesh_size = args.mesh_size

    print_versions(comm)

    script_path = os.path.dirname(os.path.abspath(__file__))
    script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]

    logfile_path = logfile_full_path(script_path, script_name_without_extension)
    outputfile_graph_path = outputfile_graph_full_path(script_path, script_name_without_extension)
    outputfile_xdmf_path = outputfile_xdmf_full_path(script_path, script_name_without_extension)
    parameter_path = os.path.join(script_path, "parameters.txt")

    if rank == 0 and os.path.exists(parameter_path):
        os.remove(parameter_path)

    timer = dlfx.common.Timer()
    timer.start()

    print("MPI-STATUS: Process:", rank, "of", comm.size, "processes.")
    sys.stdout.flush()

    if rank == 0:
        print(f"Using lam_micro_param = {lam_micro_param}")
        print(f"Using mue_micro_param = {mue_micro_param}")
        print(f"Using uniform star thickness h = {h}")
        print(f"Using flat exterior tip length b = {b}")

    domain, cell_markers, facet_markers = create_domain_from_gmsh(
        comm, script_path, d1=d1, L=L, h=h, b=b, mesh_size=mesh_size, write_msh=True
    )

    dt = dlfx.fem.Constant(domain, PETSc.ScalarType(0.05))
    Tend = 16.0 * constant_float(dt)

    lam = dlfx.fem.Constant(domain, PETSc.ScalarType(lam_micro_param))
    mu = dlfx.fem.Constant(domain, PETSc.ScalarType(mue_micro_param))

    E_mod = get_emod(constant_float(lam), constant_float(mu))
    if rank == 0:
        print(f"Microscopic Young's modulus E = {E_mod:.6f}")

    V = make_functionspace(domain, "Lagrange", 1, shape=(domain.geometry.dim,))

    u = dlfx.fem.Function(V)
    urestart = dlfx.fem.Function(V)
    du = ufl.TestFunction(V)
    ddu = ufl.TrialFunction(V)

    x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = get_dimensions(domain, comm)
    atol = max(x_max_all - x_min_all, y_max_all - y_min_all) * 1.0e-10

    vol = (x_max_all - x_min_all) * (y_max_all - y_min_all)
    Chom = np.zeros((3, 3))
    column_of_cmat_computed = np.array([0], dtype=np.int64)

    linearElasticProblem = StaticLinearElasticProblem()

    def before_first_time_step():
        urestart.x.array[:] = u.x.array[:]
        if rank == 0:
            prepare_newton_logfile(logfile_path)
            prepare_graphs_output_file(outputfile_graph_path)
        write_meshoutputfile(domain, outputfile_xdmf_path, comm)

    def before_each_time_step(t, dt_value):
        pass
        #if rank == 0:
       #     print_time_and_dt(t, dt_value)

    def get_residuum_and_gateaux(delta_t):
        return linearElasticProblem.prep_newton(u, du, ddu, lam, mu)

    def get_bcs(t):
        if column_of_cmat_computed[0] < 3:
            eps_mac = unit_macro_strain_tensor_for_voigt_eps(domain, int(column_of_cmat_computed[0]))
        else:
            eps_mac = dlfx.fem.Constant(domain, np.array([[0.0, 0.0],
                                                          [0.0, 0.0]], dtype=PETSc.ScalarType))
        return get_total_linear_displacement_boundary_condition_at_box(domain, comm, V, eps_mac=eps_mac, atol=atol)

    def after_timestep_success(t, dt_value, iters):
        u.name = "u"
        write_vector_field(domain, outputfile_xdmf_path, u, t, comm)

        sigma = sigma_as_tensor(u, lam, mu)

        write_tensor_fields(domain, comm, [sigma], ["sigma"], outputfile_xdmf_path, t)

        compute_and_write_tensor_eigenvalue(domain, sigma, "sigma", t, outputfile_xdmf_path, comm)       

        sigma_for_unit_strain = compute_averaged_sigma(u, lam, mu, vol, comm=comm)

        if rank == 0:
            if column_of_cmat_computed[0] < 3:
                # The loop index represents the applied strain case. Store as a COLUMN.
                Chom[:, column_of_cmat_computed[0]] = sigma_for_unit_strain
            else:
                return

           # print("Computed stiffness column:", column_of_cmat_computed[0])
            column_of_cmat_computed[0] += 1
            write_to_newton_logfile(logfile_path, t, dt_value, iters)

        urestart.x.array[:] = u.x.array[:]

    def after_timestep_restart(t, dt_value, iters):
        u.x.array[:] = urestart.x.array[:]
        u.x.scatter_forward()

    def after_last_timestep():
        timer.stop()
        if rank == 0:
            print(np.array_str(Chom, precision=6))
            print(print_results(Chom))

            lam_eff = lam_hom(Chom)[0]
            mu_eff = mu_hom(Chom)[0]
            E_eff = E_hom(Chom)
            nu_eff = nu_hom(Chom)

            if isinstance(active_geometry_params, dict) and 'd1' in active_geometry_params:
                g_param = active_geometry_params
            else:
                g_param = {
                    "d1": 0.20,
                    "d2": 0.20 + 0.3 / np.sqrt(2.0),
                    "h": 0.030,
                    "b": 0.04,
                    "L": 0.3,
                    "mesh_size": mesh_size
                    }
            table_csv_path = os.path.join(script_path, "results_table.csv")
            file_exists = os.path.isfile(table_csv_path)
            with open(table_csv_path, "a") as f:
                if not file_exists: 
                    headers = ["Timestamp", "d1", "d2", "h", "b", "L", "mesh_size", 
                        "lam_micro", "mu_micro",
                        "C11", "C12", "C13", "C21", "C22", "C23", "C31", "C32", "C33",
                        "E_effective", "nu_effective", "lam_effective", "mu_effective"]
                    f.write(",".join(headers) + "\n")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                row_data = [timestamp,
                    f"{g_param['d1']}", f"{g_param['d2']}", f"{g_param['h']}", f"{g_param['b']}", f"{g_param['L']}", f"{g_param['mesh_size']}",
                    f"{lam_micro_param}", f"{mue_micro_param}",
                    f"{Chom[0,0]:.6f}", f"{Chom[0,1]:.6f}", f"{Chom[0,2]:.6f}",
                    f"{Chom[1,0]:.6f}", f"{Chom[1,1]:.6f}", f"{Chom[1,2]:.6f}",
                    f"{Chom[2,0]:.6f}", f"{Chom[2,1]:.6f}", f"{Chom[2,2]:.6f}",
                    f"{E_eff:.6f}", f"{nu_eff:.6f}", f"{lam_eff:.6f}", f"{mu_eff:.6f}"
                ]
                f.write(",".join(row_data) + "\n")
            
            print(f"Success: Parameter run logged cleanly to table at: {table_csv_path}")

            variable_name = "h" #change variable depending on study
            try:
                with open(table_csv_path, "r") as f:
                    lines = [line.strip().split(",") for line in f.readlines()]
                if len(lines) > 1:
                    headers = lines[0]
                    data_rows = np.array(lines[1:])
                    
                    x_col_idx = headers.index(variable_name)
                    nu_col_idx = headers.index("nu_effective")
                    
                    x_values = data_rows[:, x_col_idx].astype(float)
                    poisson_ratios = data_rows[:, nu_col_idx].astype(float)

                    sort_idx = np.argsort(x_values)
                    x_values = x_values[sort_idx]
                    poisson_ratios = poisson_ratios[sort_idx]

                    plt.figure(figsize=(7, 4.5))
                    plt.plot(x_values, poisson_ratios, marker='o', linestyle='-', color='indigo', linewidth=2, label=r'Effective $\nu$')
                    plt.xlabel(f"Parametric Star Variable ({variable_name})", fontsize=11)
                    plt.ylabel("Effective Poisson's Ratio", fontsize=11)
                    plt.title(f"Parametric Study: Poisson's Ratio vs. {variable_name}", fontsize=12, fontweight='bold')
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.legend(loc="best")
                    plt.tight_layout()

                    trend_png_path = os.path.join(script_path, f"poisson_vs_{variable_name}.png")
                    plt.savefig(trend_png_path, dpi=300)
                    plt.close()
                    print(f"Success: Parametric trend plot saved directly to: {trend_png_path}")
            except Exception as e:
                print(f"Note: Could not generate trend plot for {variable_name}. Error details: {e}")
                

            runtime = timer.elapsed()
            print_runtime(runtime)
            write_runtime_to_newton_logfile(logfile_path, runtime)

            print("")
            print("=====================================")
            print("EFFECTIVE MATERIAL PROPERTIES:")
            print("=====================================")
            print(f"Young's Modulus (E): {E_eff:.6f}")
            print(f"Poisson's Ratio (nu): {nu_eff:.6f}")
            print(f"Lamé lambda: {lam_eff:.6f}")
            print(f"Lamé mu: {mu_eff:.6f}")
            print("=====================================")

            append_to_file(
                filename=parameter_path,
                parameters={
                    "lam_effective": lam_eff,
                    "mue_effective": mu_eff,
                    "youngs_modulus_effective": E_eff,
                    "poisson_ratio_effective": nu_eff,
                },
                comm=comm,
            )

            runtime = timer.elapsed()
            print_runtime(runtime)
            write_runtime_to_newton_logfile(logfile_path, runtime)

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
        print_bool=True,
    )


if __name__ == "__main__":
    main()
