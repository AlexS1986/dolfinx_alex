"""Test of the reaction-force evaluations in alex.postprocessing.

Linear-elastic plane-strain rectangle 6 mm x 1 mm, P1 triangles, displacement-controlled. For a linear
problem with one prescribed displacement u0 the exact reaction of the discrete model is R = 2*Pi_el/u0.
Compared methods:
  residual          pp.reaction_force_from_residual  (sum of the assembled residual at the constrained dofs)
  sigma (UFL)       pp.reaction_force with the stress as UFL expression (not interpolated), loaded facets
  sigma (DP0)       pp.reaction_force with the stress interpolated to DP0 (as in 01_phasefield_dcb_260504_folder.py)
  sigma (UFL) top   pp.reaction_force with the UFL stress over the whole top boundary

Load cases:
  strips    left/right edge clamped, two strips of width 0.075 mm at x = 2 and x = 4 on the top edge with
            u_y = -u0, u_x = 0 (the benchmark of the manuscript, homogeneous material)
  top_rough bottom edge clamped, whole top edge with u_y = -u0, u_x = 0
  top_free  bottom edge u_y = 0 (one node u_x = 0), whole top edge u_y = -u0 (homogeneous stress state)

Usage:  python3 13_test_reaction_force.py            (serial)
        mpirun -n 4 python3 13_test_reaction_force.py
"""
import sys

import numpy as np
import ufl
import dolfinx as dlfx
import dolfinx.fem.petsc
import dolfinx.nls.petsc
from mpi4py import MPI

import alex.postprocessing as pp
import alex.boundaryconditions as bc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

E_MOD, NU = 210000.0, 0.3
LAM = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))
MU = E_MOD / (2 * (1 + NU))
U0 = 0.001
WIDTH = 0.075
TOL_RESIDUAL = 1e-8


def run(load_case, h):
    nx, ny = int(round(6.0 / h)), int(round(1.0 / h))
    domain = dlfx.mesh.create_rectangle(comm, [[0.0, 0.0], [6.0, 1.0]], [nx, ny], dlfx.mesh.CellType.triangle)
    fdim = domain.topology.dim - 1
    V = dlfx.fem.functionspace(domain, ("P", 1, (2,)))
    u = dlfx.fem.Function(V)
    du = ufl.TestFunction(V)
    n = ufl.FacetNormal(domain)

    def sigma_of(v):
        eps = ufl.sym(ufl.grad(v))
        return LAM * ufl.tr(eps) * ufl.Identity(2) + 2.0 * MU * eps

    sigma = sigma_of(u)
    residual = ufl.inner(sigma, ufl.sym(ufl.grad(du))) * ufl.dx

    def dirichlet(value, component, where):
        facets = dlfx.mesh.locate_entities_boundary(domain, fdim, where)
        dofs = dlfx.fem.locate_dofs_topological(V.sub(component), fdim, facets)
        return dlfx.fem.dirichletbc(dlfx.default_scalar_type(value), dofs, V.sub(component)), dofs

    top = bc.get_top_boundary_of_box_as_function(domain, comm, atol=0.0)
    bcs, loaded = [], {}
    if load_case == "strips":
        for where in (bc.get_left_boundary_of_box_as_function(domain, comm, atol=0.0),
                      bc.get_right_boundary_of_box_as_function(domain, comm, atol=0.0)):
            bcs += [dirichlet(0.0, 0, where)[0], dirichlet(0.0, 1, where)[0]]
        for name, center in (("left", 2.0), ("right", 4.0)):
            where = bc.get_x_range_at_top_of_box_as_function(domain, comm, WIDTH, center, atol=0.0)
            bc_y, dofs_y = dirichlet(-U0, 1, where)
            bcs += [bc_y, dirichlet(0.0, 0, where)[0]]
            loaded[name] = (where, dofs_y)
    else:
        bottom = bc.get_bottom_boundary_of_box_as_function(domain, comm, atol=0.0)
        bc_y, dofs_y = dirichlet(-U0, 1, top)
        bcs += [bc_y, dirichlet(0.0, 1, bottom)[0]]
        if load_case == "top_rough":
            bcs += [dirichlet(0.0, 0, top)[0], dirichlet(0.0, 0, bottom)[0]]
        else:
            corner = dlfx.mesh.locate_entities_boundary(
                domain, 0, lambda x: np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)))
            dofs_c = dlfx.fem.locate_dofs_topological(V.sub(0), 0, corner)
            bcs.append(dlfx.fem.dirichletbc(dlfx.default_scalar_type(0.0), dofs_c, V.sub(0)))
        loaded["top"] = (top, dofs_y)

    a = ufl.derivative(residual, u, ufl.TrialFunction(V))
    problem = dlfx.fem.petsc.NonlinearProblem(residual, u, bcs, a)
    solver = dlfx.nls.petsc.NewtonSolver(comm, problem)
    solver.rtol, solver.atol = 1e-12, 1e-10
    ksp = solver.krylov_solver
    ksp.setType("preonly"); ksp.getPC().setType("lu"); ksp.getPC().setFactorSolverType("mumps")
    solver.solve(u)
    u.x.scatter_forward()

    Pi_el = comm.allreduce(dlfx.fem.assemble_scalar(dlfx.fem.form(0.5 * ufl.inner(sigma, ufl.sym(ufl.grad(u))) * ufl.dx)), op=MPI.SUM)
    R_energy = -2.0 * Pi_el / U0  # compressive, same sign convention as the traction integral

    TEN = dlfx.fem.functionspace(domain, ("DP", 0, (2, 2)))
    sigma_dp0 = dlfx.fem.Function(TEN)
    sigma_dp0.interpolate(dlfx.fem.Expression(sigma, TEN.element.interpolation_points()))

    R_res = pp.reaction_force_from_residual(dlfx.fem.form(residual), [d for _, d in loaded.values()], comm)
    R_residual = sum(R_res)
    R_ufl = R_dp0 = 0.0
    n_facets = 0
    for where, _ in loaded.values():
        tags = pp.tag_part_of_boundary(domain, where, 1)
        ds_tagged = ufl.Measure("ds", domain=domain, subdomain_data=tags)
        R_ufl += pp.reaction_force(sigma, n=n, ds=ds_tagged(1), comm=comm)[1]
        R_dp0 += pp.reaction_force(sigma_dp0, n=n, ds=ds_tagged(1), comm=comm)[1]
        n_facets += comm.allreduce(int(np.sum(tags.indices < domain.topology.index_map(fdim).size_local)), op=MPI.SUM)
    top_tags = pp.tag_part_of_boundary(domain, top, 9)
    ds_top = ufl.Measure("ds", domain=domain, subdomain_data=top_tags)
    R_ufl_top = pp.reaction_force(sigma, n=n, ds=ds_top(9), comm=comm)[1]

    if rank == 0:
        print(f"\nload case {load_case:9s}  h = {h:g} mm  loaded facets = {n_facets}  processes = {comm.size}")
        print(f"  reference  2*Pi_el/u0      R = {R_energy: .6e} N/mm")
        for label, value in (("residual", R_residual), ("sigma (UFL), loaded facets", R_ufl),
                             ("sigma (DP0), loaded facets", R_dp0), ("sigma (UFL), whole top", R_ufl_top)):
            print(f"  {label:27s}R = {value: .6e} N/mm   ratio to reference = {value / R_energy:.6f}")
    return abs(R_residual / R_energy - 1.0)


if __name__ == "__main__":
    worst = 0.0
    for load_case, sizes in (("strips", (0.02, 0.01, 0.005)), ("top_rough", (0.02, 0.01)), ("top_free", (0.02,))):
        for h in sizes:
            worst = max(worst, run(load_case, h))
    if rank == 0:
        ok = worst < TOL_RESIDUAL
        print(f"\nresidual-based reaction: max relative deviation from 2*Pi_el/u0 = {worst:.2e} -> {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
