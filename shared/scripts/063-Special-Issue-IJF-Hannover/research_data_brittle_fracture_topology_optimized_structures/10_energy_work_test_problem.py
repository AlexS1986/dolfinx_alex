#!/usr/bin/env python3
"""Small phase-field energy/work consistency test.

The test is a square specimen with a sharp triangular notch at the
left edge. The bottom edge is fixed, while the complete top edge is
loaded by a prescribed vertical displacement. It writes XDMF
fields and a graph file containing work, elastic energy, fracture
energy, total energy, and the instantaneous viscous phase-field
dissipation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import basix
import basix.ufl
import dolfinx as dlfx
import matplotlib.pyplot as plt
import numpy as np
import ufl
from mpi4py import MPI


ROOT = Path(__file__).resolve().parents[2]
UTILS = ROOT / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

import alex.boundaryconditions as bc
import alex.linearelastic as le
import alex.phasefield as pf
import alex.postprocessing as pp
import alex.solution as sol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a square energy/work phase-field check."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--L", type=float, default=1.0)
    parser.add_argument("--nx", type=int, default=150)
    parser.add_argument("--ny", type=int, default=150)
    parser.add_argument("--notch-depth", type=float, default=0.18)
    parser.add_argument("--notch-height", type=float, default=0.25)
    parser.add_argument("--mesh-size", type=float, default=None)
    parser.add_argument("--E", type=float, default=210000.0)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--Gc", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--eta", type=float, default=1.0e-5)
    parser.add_argument("--M", type=float, default=100.0)
    parser.add_argument("--dt", type=float, default=1.0e-3)
    parser.add_argument("--Tend", type=float, default=1.0)
    parser.add_argument("--v0", type=float, default=0.01)
    parser.add_argument("--load-width", type=float, default=0.2)
    parser.add_argument("--write-every", type=int, default=50)
    parser.add_argument(
        "--split",
        choices=("spectral", "volumetric"),
        default="spectral",
    )
    return parser.parse_args()


def assemble_global(form: ufl.Form, comm: MPI.Intracomm) -> float:
    value_local = dlfx.fem.assemble_scalar(dlfx.fem.form(form))
    return comm.allreduce(value_local, op=MPI.SUM)


def trapezoidal_work_increment_from_stress_expression(
    sigma,
    sigma_m1,
    u,
    um1,
    n: ufl.FacetNormal,
    ds: ufl.Measure,
    comm: MPI.Intracomm,
) -> float:
    du = u - um1
    traction = 0.5 * ufl.dot(sigma + sigma_m1, n)
    return assemble_global(ufl.inner(traction, du) * ds, comm)


def create_left_notched_square_mesh(
    comm: MPI.Intracomm,
    L: float,
    notch_depth: float,
    notch_height: float,
    mesh_size: float,
) -> dlfx.mesh.Mesh:
    import gmsh
    from dolfinx.io import gmshio

    y_mid = 0.5 * L
    y_low = y_mid - 0.5 * notch_height
    y_high = y_mid + 0.5 * notch_height

    if notch_depth <= 0.0:
        raise ValueError("--notch-depth must be positive")
    if notch_height <= 0.0:
        raise ValueError("--notch-height must be positive")
    if notch_depth >= L:
        raise ValueError("--notch-depth must be smaller than L")
    if y_low <= 0.0 or y_high >= L:
        raise ValueError("--notch-height must fit inside the square")

    gmsh.initialize()
    try:
        gmsh.model.add("left_notched_square")

        if comm.rank == 0:
            points = [
                (0.0, 0.0, 0.0),
                (L, 0.0, 0.0),
                (L, L, 0.0),
                (0.0, L, 0.0),
                (0.0, y_high, 0.0),
                (notch_depth, y_mid, 0.0),
                (0.0, y_low, 0.0),
            ]
            point_tags = [
                gmsh.model.geo.addPoint(x, y, z, mesh_size)
                for x, y, z in points
            ]
            line_tags = []
            for i, point_tag in enumerate(point_tags):
                next_point_tag = point_tags[(i + 1) % len(point_tags)]
                line_tags.append(
                    gmsh.model.geo.addLine(point_tag, next_point_tag)
                )

            curve_loop = gmsh.model.geo.addCurveLoop(line_tags)
            surface = gmsh.model.geo.addPlaneSurface([curve_loop])
            gmsh.model.geo.synchronize()
            gmsh.model.addPhysicalGroup(2, [surface], 1)
            gmsh.model.setPhysicalName(2, 1, "domain")
            gmsh.model.mesh.generate(2)

        domain, _, _ = gmshio.model_to_mesh(
            gmsh.model,
            comm,
            0,
            gdim=2,
        )
    finally:
        gmsh.finalize()

    return domain


def reference_cell_jacobian_determinants(
    cell_coordinates: np.ndarray,
    quadrature_points: np.ndarray,
) -> np.ndarray:
    if cell_coordinates.shape[0] == 4:
        x0, x1, x2, x3 = cell_coordinates[:, :2]
        xi = quadrature_points[:, 0]
        eta = quadrature_points[:, 1]
        dx_dxi = (
            -(1.0 - eta)[:, None] * x0
            + (1.0 - eta)[:, None] * x1
            - eta[:, None] * x2
            + eta[:, None] * x3
        )
        dx_deta = (
            -(1.0 - xi)[:, None] * x0
            - xi[:, None] * x1
            + (1.0 - xi)[:, None] * x2
            + xi[:, None] * x3
        )
        return np.abs(
            dx_dxi[:, 0] * dx_deta[:, 1]
            - dx_dxi[:, 1] * dx_deta[:, 0]
        )

    if cell_coordinates.shape[0] == 3:
        x0, x1, x2 = cell_coordinates[:, :2]
        dx_dxi = x1 - x0
        dx_deta = x2 - x0
        det = abs(dx_dxi[0] * dx_deta[1] - dx_dxi[1] * dx_deta[0])
        return np.full(len(quadrature_points), det)

    raise NotImplementedError(
        f"Unsupported cell with {cell_coordinates.shape[0]} nodes"
    )


def compute_quadrature_measures(
    cell_coordinates: np.ndarray,
    quadrature_points: np.ndarray,
    quadrature_weights: np.ndarray,
) -> np.ndarray:
    if cell_coordinates.shape[1] == 3:
        x0 = cell_coordinates[:, 0, :2]
        x1 = cell_coordinates[:, 1, :2]
        x2 = cell_coordinates[:, 2, :2]
        dx_dxi = x1 - x0
        dx_deta = x2 - x0
        det_j = np.abs(
            dx_dxi[:, 0] * dx_deta[:, 1]
            - dx_dxi[:, 1] * dx_deta[:, 0]
        )
        return det_j[:, None] * quadrature_weights[None, :]

    if cell_coordinates.shape[1] == 4:
        x0 = cell_coordinates[:, 0, :2]
        x1 = cell_coordinates[:, 1, :2]
        x2 = cell_coordinates[:, 2, :2]
        x3 = cell_coordinates[:, 3, :2]
        xi = quadrature_points[:, 0]
        eta = quadrature_points[:, 1]
        dx_dxi = (
            -(1.0 - eta)[None, :, None] * x0[:, None, :]
            + (1.0 - eta)[None, :, None] * x1[:, None, :]
            - eta[None, :, None] * x2[:, None, :]
            + eta[None, :, None] * x3[:, None, :]
        )
        dx_deta = (
            -(1.0 - xi)[None, :, None] * x0[:, None, :]
            - xi[None, :, None] * x1[:, None, :]
            + (1.0 - xi)[None, :, None] * x2[:, None, :]
            + xi[None, :, None] * x3[:, None, :]
        )
        det_j = np.abs(
            dx_dxi[:, :, 0] * dx_deta[:, :, 1]
            - dx_dxi[:, :, 1] * dx_deta[:, :, 0]
        )
        return det_j * quadrature_weights[None, :]

    raise NotImplementedError(
        f"Unsupported cell with {cell_coordinates.shape[1]} nodes"
    )


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    script_dir = Path(__file__).resolve().parent
    output_dir = args.output_dir or script_dir / "energy_work_test_output"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    graph_path = output_dir / "energy_work_test_graphs.txt"
    xdmf_path = output_dir / "energy_work_test.xdmf"
    plot_path = output_dir / "energy_work_test_plot.png"
    logfile_path = output_dir / "energy_work_test_newton_log.txt"

    mesh_size = args.mesh_size
    if mesh_size is None:
        mesh_size = args.L / max(args.nx, args.ny)

    domain = create_left_notched_square_mesh(
        comm,
        args.L,
        args.notch_depth * args.L,
        args.notch_height * args.L,
        mesh_size,
    )
    dim = domain.topology.dim
    fdim = dim - 1
    domain.topology.create_connectivity(fdim, dim)
    basix_celltype = getattr(
        basix.CellType,
        domain.topology.cell_types[0].name,
    )
    quadrature_points, quadrature_weights = basix.make_quadrature(
        basix_celltype,
        2,
        rule=basix.quadrature.string_to_type("default"),
    )
    cell_map = domain.topology.index_map(dim)
    cells = np.arange(cell_map.size_local, dtype=np.int32)
    geometry_dofmap = domain.geometry.dofmap
    geometry_x = domain.geometry.x
    cell_coordinates = np.asarray(
        [geometry_x[geometry_dofmap[cell]] for cell in cells]
    )
    quadrature_measures = compute_quadrature_measures(
        cell_coordinates,
        quadrature_points,
        quadrature_weights,
    )

    Ve = basix.ufl.element("P", domain.basix_cell(), 1, shape=(dim,))
    Se = basix.ufl.element("P", domain.basix_cell(), 1, shape=())
    W = dlfx.fem.FunctionSpace(
        domain,
        basix.ufl.mixed_element([Ve, Se]),
    )
    S = dlfx.fem.FunctionSpace(domain, Se)

    w = dlfx.fem.Function(W)
    wm1 = dlfx.fem.Function(W)
    wrestart = dlfx.fem.Function(W)
    u, s = w.split()
    um1, sm1 = ufl.split(wm1)
    dw = ufl.TestFunction(W)
    ddw = ufl.TrialFunction(W)

    E = dlfx.fem.Constant(domain, args.E)
    nu = dlfx.fem.Constant(domain, args.nu)
    lam = le.get_lambda(E, nu)
    mu = le.get_mu(E, nu)
    eta = dlfx.fem.Constant(domain, args.eta)
    epsilon = dlfx.fem.Constant(domain, args.epsilon)
    Gc = dlfx.fem.Function(S)
    Gc.x.array[:] = args.Gc
    Mob = dlfx.fem.Constant(domain, args.M)
    iMob = dlfx.fem.Constant(domain, 1.0 / args.M)

    t_global = dlfx.fem.Constant(domain, 0.0)
    trestart_global = dlfx.fem.Constant(domain, 0.0)
    dt_global = dlfx.fem.Constant(domain, args.dt)
    dt_max = dlfx.fem.Constant(domain, 10.0 * args.dt)
    Work = dlfx.fem.Constant(domain, 0.0)
    Work_top_sigma = dlfx.fem.Constant(domain, 0.0)
    Work_boundary_interpolated = dlfx.fem.Constant(domain, 0.0)
    Work_boundary = dlfx.fem.Constant(domain, 0.0)
    Dissipation = dlfx.fem.Constant(domain, 0.0)
    step_counter = dlfx.fem.Constant(domain, 0.0)

    phase_problem = pf.StaticPhaseFieldProblem2D_split(
        degradationFunction=pf.quadratic_degradation(),
        psisurf=pf.psisurf_from_function,
        split=args.split,
        geometric_nl=False,
    )

    apply_initial_crack = False
    bccrack = None
    crack_dofs = np.array([], dtype=np.int32)
    # crack_y = 0.5 * args.L
    # crack_x_tip = 0.5 * args.L
    # crack_atol = 0.51 * args.L / args.ny
    #
    # def initial_crack(x):
    #     on_center = np.isclose(x[1], crack_y, atol=crack_atol)
    #     left_half = x[0] <= crack_x_tip + crack_atol
    #     return np.logical_and(on_center, left_half)
    #
    # if apply_initial_crack:
    #     crack_facets = dlfx.mesh.locate_entities(
    #         domain,
    #         fdim,
    #         initial_crack,
    #     )
    #     crack_dofs = dlfx.fem.locate_dofs_topological(
    #         W.sub(1),
    #         fdim,
    #         crack_facets,
    #     )
    #     bccrack = dlfx.fem.dirichletbc(
    #         0.0,
    #         crack_dofs,
    #         W.sub(1),
    #     )

    def bottom_bc_function(x):
        return np.isclose(x[1], 0.0)

    def load_top_bc_function(x):
        return np.isclose(x[1], args.L)

    top_tag = 1
    top_tags = pp.tag_part_of_boundary(
        domain,
        load_top_bc_function,
        top_tag,
    )
    ds_boundary = ufl.Measure("ds", domain=domain)
    ds_top = ufl.Measure("ds", domain=domain, subdomain_data=top_tags)
    facets_at_load = dlfx.mesh.locate_entities_boundary(
        domain,
        fdim,
        load_top_bc_function,
    )
    dofs_at_loaded_y = dlfx.fem.locate_dofs_topological(
        W.sub(0).sub(1),
        fdim,
        facets_at_load,
    )
    n = ufl.FacetNormal(domain)
    dx = ufl.Measure("dx", domain=domain)

    TEN = dlfx.fem.functionspace(domain, ("DP", 0, (dim, dim)))
    sigma_interpolated = dlfx.fem.Function(TEN)
    sigma_m1_interpolated = dlfx.fem.Function(TEN)

    def get_bcs(t):
        uy_top = args.v0 * t_global.value
        bcs = [
            bc.define_dirichlet_bc_from_value(
                domain, 0.0, 0, bottom_bc_function, W, 0
            ),
            bc.define_dirichlet_bc_from_value(
                domain, 0.0, 1, bottom_bc_function, W, 0
            ),
            bc.define_dirichlet_bc_from_value(
                domain, uy_top, 1, load_top_bc_function, W, 0
            ),
        ]
        if apply_initial_crack and bccrack is not None:
            bcs.append(bccrack)
        return bcs

    def get_residuum_and_gateaux(delta_t):
        return phase_problem.prep_newton(
            w=w,
            wm1=wm1,
            dw=dw,
            ddw=ddw,
            lam=lam,
            mu=mu,
            Gc=Gc,
            epsilon=epsilon,
            eta=eta,
            iMob=iMob,
            delta_t=delta_t,
        )

    def fracture_energy() -> float:
        _, s_now = ufl.split(w)
        return assemble_global(phase_problem.psisurf(s_now, Gc, epsilon) * dx, comm)

    def phasefield_rate_power(dt: float) -> float:
        _, s_now = ufl.split(w)
        _, s_old = ufl.split(wm1)
        s_dot_expr = (s_now - s_old) / dt
        s_dot_eval = dlfx.fem.Expression(
            s_dot_expr,
            quadrature_points,
        ).eval(domain, cells)
        s_dot_values = np.asarray(s_dot_eval).reshape(len(cells), -1)
        dissipation_local = float(
            np.sum((s_dot_values**2 / args.M) * quadrature_measures)
        )
        return comm.allreduce(dissipation_local, op=MPI.SUM)

    def write_xdmf(t: float, sigma):
        pp.write_phasefield_mixed_solution(domain, str(xdmf_path), w, t, comm)
        Gc.name = "Gc"
        pp.write_field(domain, str(xdmf_path), Gc, t, comm, S=S)
        pp.write_tensor_fields(
            domain,
            comm,
            [sigma],
            ["sig"],
            outputfile_xdmf_path=str(xdmf_path),
            t=t,
        )

    def before_first_time_step():
        wm1.x.array[:] = 0.0
        wm1.sub(1).x.array[:] = np.ones_like(wm1.sub(1).x.array[:])
        if apply_initial_crack:
            wm1.x.array[crack_dofs] = 0.0
        w.x.array[:] = wm1.x.array[:]
        wrestart.x.array[:] = wm1.x.array[:]
        if rank == 0:
            pp.prepare_graphs_output_file(str(graph_path))
            sol.prepare_newton_logfile(str(logfile_path))
        pp.write_meshoutputfile(domain, str(xdmf_path), comm)
        sigma = phase_problem.sigma_degraded(u, s, lam, mu, eta)
        initial_output_time = -max(abs(float(dt_global.value)), 1.0e-12)
        write_xdmf(initial_output_time, sigma)

    def before_each_time_step(t, dt):
        if rank == 0:
            sol.print_time_and_dt(t, dt)

    def after_timestep_success(t, dt, iters):
        sigma = phase_problem.sigma_degraded(u, s, lam, mu, eta)
        sigma_m1 = phase_problem.sigma_degraded(um1, sm1, lam, mu, eta)
        sigma_expr = dlfx.fem.Expression(
            sigma,
            TEN.element.interpolation_points(),
        )
        sigma_interpolated.interpolate(sigma_expr)
        sigma_interpolated.name = "sig"
        sigma_m1_expr = dlfx.fem.Expression(
            sigma_m1,
            TEN.element.interpolation_points(),
        )
        sigma_m1_interpolated.interpolate(sigma_m1_expr)

        _, Ry_top = pp.reaction_force(
            sigma_interpolated,
            n=n,
            ds=ds_top(top_tag),
            comm=comm,
        )
        dW_top_interpolated = pp.work_increment_external_forces(
            sigma_interpolated,
            u,
            um1,
            n,
            ds_top(top_tag),
            comm=comm,
        )
        dW_top_interpolated_m1 = pp.work_increment_external_forces(
            sigma_m1_interpolated,
            u,
            um1,
            n,
            ds_top(top_tag),
            comm=comm,
        )
        dW_top_interpolated = 0.5 * (
            dW_top_interpolated + dW_top_interpolated_m1
        )
        Work.value = Work.value + dW_top_interpolated
        dW_top_sigma = trapezoidal_work_increment_from_stress_expression(
            sigma,
            sigma_m1,
            u,
            um1,
            n,
            ds_top(top_tag),
            comm,
        )
        Work_top_sigma.value = Work_top_sigma.value + dW_top_sigma
        dW_boundary_interpolated = pp.work_increment_external_forces(
            sigma_interpolated,
            u,
            um1,
            n,
            ds_boundary,
            comm=comm,
        )
        dW_boundary_interpolated_m1 = pp.work_increment_external_forces(
            sigma_m1_interpolated,
            u,
            um1,
            n,
            ds_boundary,
            comm=comm,
        )
        dW_boundary_interpolated = 0.5 * (
            dW_boundary_interpolated + dW_boundary_interpolated_m1
        )
        Work_boundary_interpolated.value = (
            Work_boundary_interpolated.value + dW_boundary_interpolated
        )
        dW_boundary_sigma = trapezoidal_work_increment_from_stress_expression(
            sigma,
            sigma_m1,
            u,
            um1,
            n,
            ds_boundary,
            comm,
        )
        Work_boundary.value = Work_boundary.value + dW_boundary_sigma

        Pi_el = phase_problem.get_E_el_global(
            s,
            eta,
            u,
            lam,
            mu,
            dx=dx,
            comm=comm,
        )
        Pi_frac = fracture_energy()
        Pi_total = Pi_el + Pi_frac
        D_rate = phasefield_rate_power(dt)
        Dissipation.value = Dissipation.value + D_rate * dt
        if len(dofs_at_loaded_y) > 0:
            uy_top_local = np.max(w.x.array[dofs_at_loaded_y])
        else:
            uy_top_local = -1.0e20
        uy_top = comm.allreduce(uy_top_local, op=MPI.MAX)

        if rank == 0:
            pp.write_to_graphs_output_file(
                str(graph_path),
                t,
                uy_top,
                Ry_top,
                dW_top_interpolated,
                Work.value,
                dW_top_sigma,
                Work_top_sigma.value,
                dW_boundary_interpolated,
                Work_boundary_interpolated.value,
                dW_boundary_sigma,
                Work_boundary.value,
                Pi_el,
                Pi_frac,
                Pi_total,
                D_rate,
                Dissipation.value,
            )
            sol.write_to_newton_logfile(str(logfile_path), t, dt, iters)

        wm1.x.array[:] = w.x.array[:]
        wrestart.x.array[:] = w.x.array[:]
        step_counter.value = step_counter.value + 1.0
        if int(step_counter.value) % max(args.write_every, 1) == 0:
            write_xdmf(t, sigma)

    def after_timestep_restart(t, dt, iters):
        w.x.array[:] = wrestart.x.array[:]

    def after_last_timestep():
        if rank == 0:
            create_energy_plot(graph_path, plot_path)
            print(f"Wrote graph data to {graph_path}")
            print(f"Wrote XDMF output to {xdmf_path}")
            print(f"Wrote energy plot to {plot_path}")

    sol.solve_with_newton_adaptive_time_stepping(
        domain,
        w,
        args.Tend,
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
        dt_max=dt_max,
    )


def create_energy_plot(graph_path: Path, plot_path: Path) -> None:
    data = np.loadtxt(graph_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    uy = data[:, 1]
    work_top_interpolated = data[:, 4]
    work_top_sigma = data[:, 6]
    work_boundary_interpolated = data[:, 8]
    work_boundary_sigma = data[:, 10]
    pi_el = data[:, 11]
    pi_frac = data[:, 12]
    pi_total = data[:, 13]
    dissipation = data[:, 15]
    pi_total_with_dissipation = pi_total + dissipation

    y_min = min(
        float(np.min(work_top_interpolated)),
        float(np.min(work_top_sigma)),
        float(np.min(work_boundary_interpolated)),
        float(np.min(work_boundary_sigma)),
        float(np.min(pi_el)),
        float(np.min(pi_frac)),
        float(np.min(pi_total)),
        float(np.min(dissipation)),
        float(np.min(pi_total_with_dissipation)),
    )
    y_max = max(
        float(np.max(work_top_interpolated)),
        float(np.max(work_top_sigma)),
        float(np.max(work_boundary_interpolated)),
        float(np.max(work_boundary_sigma)),
        float(np.max(pi_el)),
        float(np.max(pi_frac)),
        float(np.max(pi_total)),
        float(np.max(dissipation)),
        float(np.max(pi_total_with_dissipation)),
    )
    y_padding = 0.05 * max(y_max - y_min, 1.0)
    y_lower = y_min - y_padding
    y_upper = y_max + y_padding

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(
        uy,
        work_top_interpolated,
        label=r"$W_{\Gamma_\mathrm{top}}$ interp. trap.",
        linewidth=2.0,
    )
    ax.plot(
        uy,
        work_top_sigma,
        label=r"$W_{\Gamma_\mathrm{top}}$ $\sigma$ trap.",
        linewidth=2.0,
        linestyle="--",
    )
    ax.plot(
        uy,
        work_boundary_interpolated,
        label=r"$W_{\partial\Omega}$ interp. trap.",
        linewidth=2.0,
        linestyle="-.",
    )
    ax.plot(
        uy,
        work_boundary_sigma,
        label=r"$W_{\partial\Omega}$ $\sigma$ trap.",
        linewidth=2.0,
        linestyle=":",
    )
    ax.plot(uy, pi_el, label=r"$\Pi_\mathrm{el}$", linewidth=1.8)
    ax.plot(uy, pi_frac, label=r"$\Pi_\mathrm{frac}$", linewidth=1.8)
    ax.plot(uy, pi_total, label=r"$\Pi_\mathrm{el}+\Pi_\mathrm{frac}$", linewidth=2.0)
    ax.plot(
        uy,
        pi_total_with_dissipation,
        label=r"$\Pi_\mathrm{el}+\Pi_\mathrm{frac}+D_s$",
        linewidth=2.2,
        linestyle="--",
    )
    ax.plot(
        uy,
        dissipation,
        label=r"$D_s$",
        linewidth=1.6,
        linestyle=(0, (1, 2)),
    )
    ax.set_xlabel(r"$u_y$")
    ax.set_ylabel("energy / work")
    ax.set_ylim(y_lower, y_upper)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
