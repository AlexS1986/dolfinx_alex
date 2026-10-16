"""
3D surfing-boundary-condition fracture simulation (spherical pores, 6x3x2 cells) for the KLAIM 2025
revision-2 reruns. Based on 061-plasticity-fracture-noll-3D/000_template/run_simulation.py with

  * eta and beta as command line arguments (manuscript values 1e-5 and 0.2 are the defaults;
    the old 3D runs used eta = 1e-3 and the library default beta),
  * yield stress as a cell function (linear elastic embedding, as described in the manuscript),
  * optional cubic (non-isotropic) effective stiffness of the embedding (--embedding cubic),
  * energy partition / dissipation outputs identical to the 2D script (see README.md),
  * field snapshots before/after crack jumps.
"""
import alex.linearelastic
import alex.phasefield
import dolfinx as dlfx
from mpi4py import MPI
from petsc4py import PETSc as petsc

import ufl
import numpy as np
import os
import sys
import math
import shutil
from datetime import datetime

import alex.os
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol
import alex.linearelastic as le
import alex.phasefield as pf
import basix
import alex.plasticity
import alex.heterogeneous as het
import alex.util
import argparse


class StopSimulation(Exception):
    pass


script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path, script_name_without_extension)
outputfile_graph_path = alex.os.outputfile_graph_full_path(script_path, script_name_without_extension)
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path, script_name_without_extension)
outputfile_events_xdmf_path = os.path.join(script_path, script_name_without_extension + "_events.xdmf")
events_log_path = os.path.join(script_path, script_name_without_extension + "_events.txt")
columns_path = os.path.join(script_path, script_name_without_extension + "_graphs_columns.txt")
parameter_path = os.path.join(script_path, "parameters.txt")

comm, rank, size = alex.os.set_mpi()
alex.os.print_mpi_status(rank, size)
dlfx.log.set_log_level(dlfx.log.LogLevel.INFO)
if rank == 0:
    alex.util.print_dolfinx_version()

parser = argparse.ArgumentParser(description="3D surfing-BC phase-field fracture simulation (KLAIM rev. 2).")
parser.add_argument("--mesh_file", type=str, default="mesh_fracture_adaptive.xdmf")
parser.add_argument("--in_crack_length", type=float, required=True)
parser.add_argument("--lam_micro_param", type=float, default=1.0)
parser.add_argument("--mue_micro_param", type=float, default=1.0)
parser.add_argument("--gc_micro_param", type=float, default=1.0)
parser.add_argument("--eps_param", type=float, default=0.1)
parser.add_argument("--element_order", type=int, default=1)
parser.add_argument("--material", type=str, default="vonmises", choices=["vonmises", "elastic"])
parser.add_argument("--sig_y", type=float, default=1.0)
parser.add_argument("--hard", type=float, default=0.2222222)
parser.add_argument("--sig_y_embedding", type=float, default=1.0e6)
parser.add_argument("--beta", type=float, default=0.2)
parser.add_argument("--eta", type=float, default=1.0e-5)
parser.add_argument("--mob", type=float, default=1000.0)
parser.add_argument("--embedding", type=str, default="isotropic", choices=["isotropic", "cubic"])
parser.add_argument("--bc_moduli", type=str, default="effective", choices=["effective", "fixed"],
                    help="fixed: K-field with E_bc, nu_bc (manuscript: 2.5 mu0, 0.25); effective: (E_eff, nu_eff) of the embedding (old runs)")
parser.add_argument("--E_bc", type=float, default=2.5)
parser.add_argument("--nu_bc", type=float, default=0.25)
parser.add_argument("--K_bc", type=float, default=math.sqrt(2.5))
parser.add_argument("--postprocessing_interval", type=int, default=200)
parser.add_argument("--event_snapshots", type=int, default=1)
parser.add_argument("--jump_threshold_factor", type=float, default=4.0)
parser.add_argument("--stop_fraction", type=float, default=0.9)
parser.add_argument("--dt_start", type=float, default=1.0e-4)
parser.add_argument("--dt_max", type=float, default=0.01)
args = parser.parse_args()

mesh_file = args.mesh_file
in_crack_length = args.in_crack_length
la_micro = args.lam_micro_param
mu_micro = args.mue_micro_param
gc_micro = args.gc_micro_param
eps_param = args.eps_param

parameters = pp.read_parameters_file(parameter_path)
la_effective = parameters["lam_effective"]
mu_effective = parameters["mue_effective"]
wsteg = parameters["wsteg"]
dhole = parameters["dhole"]
w_cell = wsteg + dhole
if args.embedding == "cubic":
    for key in ("C11_effective", "C12_effective", "C44_effective"):
        if key not in parameters:
            raise RuntimeError(f"--embedding cubic requires {key} in parameters.txt")
    la_effective = parameters["C12_effective"]
    mu_effective = parameters["C44_effective"]
    delta_cubic = parameters["C11_effective"] - parameters["C12_effective"] - 2.0 * parameters["C44_effective"]
else:
    delta_cubic = 0.0
if rank == 0:
    print(f"Initial crack length: {in_crack_length}; material={args.material} beta={args.beta} eta={args.eta} "
          f"sig_y={args.sig_y} hard={args.hard} embedding={args.embedding} delta_cubic={delta_cubic:.4e}")

timer = dlfx.common.Timer()
timer.start()

with dlfx.io.XDMFFile(comm, os.path.join(script_path, mesh_file), 'r') as mesh_inp:
    domain = mesh_inp.read_mesh(name="mesh")
    mesh_tags = mesh_inp.read_meshtags(domain, name="Cell tags")

t_global = dlfx.fem.Constant(domain, 0.0)
dt_global = dlfx.fem.Constant(domain, args.dt_start)
dt_max = dlfx.fem.Constant(domain, args.dt_max)
trestart_global = dlfx.fem.Constant(domain, 0.0)

deg_quad = 1
Ve = basix.ufl.element("P", domain.basix_cell(), deg_quad, shape=(domain.geometry.dim,))
Te = basix.ufl.element("P", domain.basix_cell(), deg_quad, shape=())
W = dlfx.fem.FunctionSpace(domain, ufl.MixedElement([Ve, Te]))
S = dlfx.fem.functionspace(domain, Te)
S0e = basix.ufl.element("DP", domain.basix_cell(), 0, shape=())
S0 = dlfx.fem.functionspace(domain, S0e)

dim = domain.topology.dim
tdim = dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
alex.os.mpi_print('spatial dimensions: ' + str(dim), rank)

x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = pp.compute_bounding_box(comm, domain)
thickness_z = z_max_all - z_min_all
pp.print_bounding_box(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all)

v_crack = 1.0
Tend = (x_max_all - x_min_all) * 2.0 / v_crack

micro_material_marker = 1
effective_material_marker = 0
micro_material_cells = mesh_tags.find(micro_material_marker)
effective_material_cells = mesh_tags.find(effective_material_marker)

la = het.set_cell_function_heterogeneous_material(domain, la_micro, la_effective, micro_material_cells, effective_material_cells)
mu = het.set_cell_function_heterogeneous_material(domain, mu_micro, mu_effective, micro_material_cells, effective_material_cells)
gc = het.set_cell_function_heterogeneous_material(domain, gc_micro, gc_micro, micro_material_cells, effective_material_cells)
marker_micro = het.set_cell_function_heterogeneous_material(domain, 1.0, 0.0, micro_material_cells, effective_material_cells)
marker_eff = het.set_cell_function_heterogeneous_material(domain, 0.0, 1.0, micro_material_cells, effective_material_cells)
if args.material == "elastic":
    sig_y_micro_value = 1.0e6
    sig_y_eff_value = 1.0e6
else:
    sig_y_micro_value = args.sig_y
    sig_y_eff_value = args.sig_y_embedding
sig_y = het.set_cell_function_heterogeneous_material(domain, sig_y_micro_value, sig_y_eff_value, micro_material_cells, effective_material_cells)
hard = dlfx.fem.Constant(domain, args.hard)

eta = dlfx.fem.Constant(domain, args.eta)
epsilon = dlfx.fem.Constant(domain, eps_param)
Mob = dlfx.fem.Constant(domain, args.mob)
iMob = dlfx.fem.Constant(domain, 1.0 / Mob.value)
delta_cubic_const = dlfx.fem.Constant(domain, delta_cubic)

(alpha_n, alpha_tmp,
 e_p_11_n, e_p_22_n, e_p_33_n,
 e_p_12_n, e_p_13_n, e_p_23_n) = alex.plasticity.define_internal_state_variables_basix_3D_wo_tmp(domain, deg_quad, quad_scheme="default")
# own copies of the previous plastic strain (for the pre-jump stress reconstruction)
Wq = alpha_n.function_space
e_p_prev = [dlfx.fem.Function(Wq) for _ in range(6)]  # 11,22,33,12,13,23

dx = alex.plasticity.define_custom_integration_measure_that_matches_quadrature_degree_and_scheme(domain, deg_quad, "default")
dx_tagged = ufl.Measure("dx", domain=domain, subdomain_data=mesh_tags,
                        metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"})
quadrature_points, cells = alex.plasticity.get_quadraturepoints_and_cells_for_inter_polation_at_gauss_points(domain, deg_quad)

e_p_n_3D = ufl.as_tensor([[e_p_11_n, e_p_12_n, e_p_13_n],
                          [e_p_12_n, e_p_22_n, e_p_23_n],
                          [e_p_13_n, e_p_23_n, e_p_33_n]])
e_p_prev_3D = ufl.as_tensor([[e_p_prev[0], e_p_prev[3], e_p_prev[4]],
                             [e_p_prev[3], e_p_prev[1], e_p_prev[5]],
                             [e_p_prev[4], e_p_prev[5], e_p_prev[2]]])


class PhaseFieldProblemCubicEmbedding(pf.StaticPhaseFieldProblem_plasticity_noll):
    """adds delta * marker_eff * diag(eps11, eps22, eps33) in the embedding cells (cubic symmetry:
    C11 = lam + 2 mu + delta, C12 = lam, C44 = mu); delta = 0 -> isotropic"""

    def __init__(self, delta, marker_eff, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta
        self.marker_eff = marker_eff

    def sigma_undegraded(self, u, lam, mu):
        sig = super().sigma_undegraded(u, lam, mu)
        eps3 = ufl.sym(ufl.grad(u))
        corr = self.delta * self.marker_eff * ufl.as_tensor([[eps3[0, 0], 0.0, 0.0], [0.0, eps3[1, 1], 0.0], [0.0, 0.0, eps3[2, 2]]])
        return sig + corr

    def psiel_undegraded(self, u, la, mu):
        eps3 = ufl.sym(ufl.grad(u))
        return super().psiel_undegraded(u, la, mu) + 0.5 * self.delta * self.marker_eff * (eps3[0, 0] ** 2 + eps3[1, 1] ** 2 + eps3[2, 2] ** 2)


degradation = pf.cubic_degradation(beta=args.beta)
phaseFieldProblem = PhaseFieldProblemCubicEmbedding(
    delta=delta_cubic_const, marker_eff=marker_eff,
    degradationFunction=degradation, psisurf=pf.psisurf_from_function, dx=dx,
    sig_y=sig_y, hard=hard.value, alpha_n=alpha_n, e_p_n=e_p_n_3D)


def psi_pl_undegraded(alpha):
    return 0.5 * hard * alpha ** 2 + sig_y * alpha


def update_plastic_state_3D(u_):
    """radial-return update of alpha and e_p with sig_y as a cell function (alex' 3D helper expects a Constant)."""
    for f_prev, f_cur in zip(e_p_prev, [e_p_11_n, e_p_22_n, e_p_33_n, e_p_12_n, e_p_13_n, e_p_23_n]):
        f_prev.x.array[:] = f_cur.x.array[:]
    alpha_tmp.x.array[:] = alpha_n.x.array[:]
    alpha_expr = alex.plasticity.update_alpha(u_, e_p_n=e_p_prev_3D, alpha_n=alpha_tmp, sig_y=sig_y, hard=hard, mu=mu)
    alpha_n.x.array[:] = alex.plasticity.interpolate_quadrature(domain, cells, quadrature_points, alpha_expr)
    e_p_np1_expr = alex.plasticity.update_e_p(u_, e_p_n=e_p_prev_3D, alpha_n=alpha_tmp, sig_y=sig_y, hard=hard, mu=mu)
    for f_cur, (i, j) in zip([e_p_11_n, e_p_22_n, e_p_33_n, e_p_12_n, e_p_13_n, e_p_23_n], [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]):
        f_cur.x.array[:] = alex.plasticity.interpolate_quadrature(domain, cells, quadrature_points, e_p_np1_expr[i, j])


# ---------------------------------------------------------------- crack and BCs (as in 061)
K1 = dlfx.fem.Constant(domain, args.K_bc)
crack_tip_start_location_x = in_crack_length + x_min_all
crack_tip_start_location_y = (y_max_all + y_min_all) / 2.0


def crack(x):
    x_log = x[0] < (crack_tip_start_location_x)
    y_log = np.isclose(x[1], crack_tip_start_location_y, atol=(0.5 * epsilon.value))
    return np.logical_and(y_log, x_log)


crackfacets = dlfx.mesh.locate_entities(domain, fdim, crack)
crackdofs = dlfx.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)
bccrack = dlfx.fem.dirichletbc(0.0, crackdofs, W.sub(1))

w = dlfx.fem.Function(W)
u, s = w.split()
wrestart = dlfx.fem.Function(W)
wm1 = dlfx.fem.Function(W)
um1, sm1 = ufl.split(wm1)
dw = ufl.TestFunction(W)
ddw = ufl.TrialFunction(W)

vcrack_const = dlfx.fem.Constant(domain, np.array([v_crack, 0.0, 0.0], dtype=dlfx.default_scalar_type))
crack_start = dlfx.fem.Constant(domain, np.array([x_min_all, crack_tip_start_location_y, 0.0], dtype=dlfx.default_scalar_type))
w_D = dlfx.fem.Function(W)
xxK1 = dlfx.fem.Constant(domain, np.array([0.0, 0.0, 0.0], dtype=dlfx.default_scalar_type))

if args.bc_moduli == "fixed":
    mu_bc_value = args.E_bc / (2.0 * (1.0 + args.nu_bc))
    nu_bc_value = args.nu_bc
else:
    mu_bc_value = mu_effective
    nu_bc_value = alex.linearelastic.get_nu(lam=la_effective, mu=mu_effective)
mu_bc = dlfx.fem.Constant(domain, mu_bc_value)
nu_bc = dlfx.fem.Constant(domain, nu_bc_value)


def compute_surf_displacement():
    x = ufl.SpatialCoordinate(domain)
    dxx = x[0] - xxK1[0]
    dyy = x[1] - xxK1[1]
    r = ufl.sqrt(ufl.inner(dxx, dxx) + ufl.inner(dyy, dyy))
    theta = ufl.atan2(dyy, dxx)
    kappa = 3.0 - 4.0 * nu_bc
    u_x = K1 / (2.0 * mu_bc * math.sqrt(2.0 * math.pi)) * ufl.sqrt(r) * (kappa - ufl.cos(theta)) * ufl.cos(0.5 * theta)
    u_y = K1 / (2.0 * mu_bc * math.sqrt(2.0 * math.pi)) * ufl.sqrt(r) * (kappa - ufl.cos(theta)) * ufl.sin(0.5 * theta)
    return ufl.as_vector([u_x, u_y, 0.0])  # u_z = 0 on all outer faces (mimics plane strain, as in the paper)


atol = (x_max_all - x_min_all) * 0.02
boundary_surfing_bc = bc.get_boundary_of_box_as_function(domain, comm, atol=atol, epsilon=2.0 * epsilon.value)
bc_expression = dlfx.fem.Expression(compute_surf_displacement(), W.sub(0).element.interpolation_points())
facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, boundary_surfing_bc)
dofs_at_boundary = dlfx.fem.locate_dofs_topological(W.sub(0), fdim, facets_at_boundary)


def get_bcs(t):
    xxK1.value = np.array([crack_start.value[0] + vcrack_const.value[0] * t_global.value,
                           crack_start.value[1] + vcrack_const.value[1] * t_global.value, 0.0], dtype=dlfx.default_scalar_type)
    bcs = []
    w_D.sub(0).interpolate(bc_expression)
    bc_surf = dlfx.fem.dirichletbc(w_D, dofs_at_boundary)
    if abs(t) > sys.float_info.epsilon * 5:
        bcs.append(pf.irreversibility_bc(domain, W, wm1))
    bcs.append(bccrack)
    bcs.append(bc_surf)
    return bcs


def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
    [Res, dResdw] = phaseFieldProblem.prep_newton(w=w, wm1=wm1, dw=dw, ddw=ddw, lam=la, mu=mu,
                                                  Gc=gc, epsilon=epsilon, eta=eta, iMob=iMob, delta_t=delta_t)
    return [Res, dResdw]


# ---------------------------------------------------------------- post-processing setup
n = ufl.FacetNormal(domain)
external_surface_tag = 5
external_surface_tags = pp.tag_part_of_boundary(domain, boundary_surfing_bc, external_surface_tag)
ds = ufl.Measure('ds', domain=domain, subdomain_data=external_surface_tags, metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"})
top_surface_tag = 9
top_surface_tags = pp.tag_part_of_boundary(domain, bc.get_top_boundary_of_box_as_function(domain, comm, atol=atol * 0.0), top_surface_tag)
ds_top_tagged = ufl.Measure('ds', domain=domain, subdomain_data=top_surface_tags, metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"})

s_zero_for_tracking_at_nodes = dlfx.fem.Function(S)
c = dlfx.fem.Constant(domain, petsc.ScalarType(1))
s_zero_for_tracking_at_nodes.interpolate(dlfx.fem.Expression(c, S.element.interpolation_points()))

TEN = dlfx.fem.functionspace(domain, ("DP", 0, (dim, dim)))
sigma_interpolated = dlfx.fem.Function(TEN); sigma_interpolated.name = "sigma"
eshelby_interpolated = dlfx.fem.Function(TEN); eshelby_interpolated.name = "eshelby"
sig_vm0 = dlfx.fem.Function(S0); sig_vm0.name = "sigma_vM"
sig_h0 = dlfx.fem.Function(S0); sig_h0.name = "sigma_h"
sig_yy0 = dlfx.fem.Function(S0); sig_yy0.name = "sigma_yy"
s0 = dlfx.fem.Function(S0); s0.name = "s_cell"
alpha0 = dlfx.fem.Function(S0); alpha0.name = "alpha"
alpha_prev0 = dlfx.fem.Function(S0); alpha_prev0.name = "alpha"
# plastic strain as a cell-wise (DG0) tensor: deg_quad = 1 -> one quadrature point per cell, so this is exact
ep_interpolated = dlfx.fem.Function(TEN); ep_interpolated.name = "eps_p"
ep_expr = dlfx.fem.Expression(e_p_n_3D, TEN.element.interpolation_points())
ep_prev_interpolated = dlfx.fem.Function(TEN); ep_prev_interpolated.name = "eps_p"
ep_prev_expr = dlfx.fem.Expression(e_p_prev_3D, TEN.element.interpolation_points())
w_prev_fields = dlfx.fem.Function(W)
sigma_prev_interpolated = dlfx.fem.Function(TEN); sigma_prev_interpolated.name = "sigma"
u_prev, s_prev = ufl.split(w_prev_fields)

num_local_cells = domain.topology.index_map(tdim).size_local
midpoints = dlfx.mesh.compute_midpoints(domain, tdim, np.arange(num_local_cells, dtype=np.int32))
micro_mask_local = marker_micro.x.array[:num_local_cells] > 0.5

Work = dlfx.fem.Constant(domain, 0.0)
D_visc = dlfx.fem.Constant(domain, 0.0)
D_pl_deg = dlfx.fem.Constant(domain, 0.0)
D_pl_undeg = dlfx.fem.Constant(domain, 0.0)
success_timestep_counter = dlfx.fem.Constant(domain, 0.0)
postprocessing_interval = dlfx.fem.Constant(domain, float(args.postprocessing_interval))
dt_used = dlfx.fem.Constant(domain, 1.0)

form_D_visc_inc = dlfx.fem.form(((s - sm1) ** 2) * iMob / dt_used * ufl.dx)
form_D_pl_deg_inc = dlfx.fem.form(degradation(s, eta) * (psi_pl_undegraded(alpha_n) - psi_pl_undegraded(alpha_tmp)) * dx)
form_D_pl_undeg_inc = dlfx.fem.form((psi_pl_undegraded(alpha_n) - psi_pl_undegraded(alpha_tmp)) * dx)
form_E_pl = dlfx.fem.form(degradation(s, eta) * psi_pl_undegraded(alpha_n) * dx)
form_E_pl_undeg = dlfx.fem.form(psi_pl_undegraded(alpha_n) * dx)
form_E_frac = dlfx.fem.form(pf.psisurf_from_function(s, gc, epsilon) * ufl.dx)
form_E_frac_micro = dlfx.fem.form(pf.psisurf_from_function(s, gc, epsilon) * dx_tagged(micro_material_marker))
form_E_el = dlfx.fem.form(phaseFieldProblem.psiel_degraded(s, eta, u, la, mu) * dx)
form_E_el_micro = dlfx.fem.form(phaseFieldProblem.psiel_degraded(s, eta, u, la, mu) * dx_tagged(micro_material_marker))
form_A = dlfx.fem.form(pf.surf(s, epsilon) * ufl.dx)


def assemble(form):
    return comm.allreduce(dlfx.fem.assemble_scalar(form), MPI.SUM)


sigma_expr = dlfx.fem.Expression(phaseFieldProblem.sigma_degraded(u, s, la, mu, eta), TEN.element.interpolation_points())
eshelby_expr = dlfx.fem.Expression(phaseFieldProblem.getEshelbyELastic(w, eta, la, mu), TEN.element.interpolation_points())
sig_vm_expr = dlfx.fem.Expression(le.sigvM(sigma_interpolated), S0.element.interpolation_points())
sig_h_expr = dlfx.fem.Expression(ufl.tr(sigma_interpolated) / 3.0, S0.element.interpolation_points())
sig_yy_expr = dlfx.fem.Expression(sigma_interpolated[1, 1], S0.element.interpolation_points())
s0_expr = dlfx.fem.Expression(s, S0.element.interpolation_points())
alpha0_expr = dlfx.fem.Expression(alpha_n, S0.element.interpolation_points())
alpha_prev0_expr = dlfx.fem.Expression(alpha_tmp, S0.element.interpolation_points())
sigma_prev_expr = dlfx.fem.Expression(
    degradation(s_prev, eta) * alex.plasticity.sig_plasticity(u_prev, e_p_n=e_p_prev_3D, alpha_n=alpha_tmp, sig_y=sig_y, hard=hard, lam=la, mu=mu),
    TEN.element.interpolation_points())

state = {"x_ct_prev": in_crack_length + x_min_all, "t_prev": 0.0, "Jx_prev": 0.0, "last_event_step": -10 ** 9, "has_prev": False}
COLUMNS = ["t", "Jx", "Jy", "Jz", "x_ct", "x_bc", "Rx_top", "Ry_top", "Rz_top", "dW", "W", "A", "dt",
           "E_el", "E_el_micro", "E_pl", "E_pl_undeg", "E_frac", "E_frac_micro", "D_visc", "D_pl_deg", "D_pl_undeg",
           "sig_vM_max_lig", "sig_h_max_lig", "sig_yy_max_lig", "alpha_max_lig", "newton_iters"]
# NOTE: J, W, A and all energies/dissipations are divided by the thickness in z (values per unit thickness, as in the paper)


def ligament_window_max(field0, x_ct):
    vals = field0.x.array[:num_local_cells]
    mask = (micro_mask_local & (midpoints[:, 0] >= x_ct - 0.5 * epsilon.value) & (midpoints[:, 0] <= x_ct + 1.5 * w_cell)
            & (np.abs(midpoints[:, 1] - crack_tip_start_location_y) <= 0.6 * dhole) & (s0.x.array[:num_local_cells] > 0.9))
    local_max = np.max(vals[mask]) if np.any(mask) else -1.0e30
    return comm.allreduce(local_max, MPI.MAX)


def write_event_snapshot(xdmf_path, time_value, w_fields, sigma_fields, alpha_field, label, ep_fields=None):
    pp.write_tensor_fields(domain, comm, [sigma_fields], ["sigma"], outputfile_xdmf_path=xdmf_path, t=time_value)
    pp.write_field(domain, xdmf_path, alpha_field, time_value, comm, S=S0)
    if ep_fields is not None:
        pp.write_tensor_fields(domain, comm, [ep_fields], ["eps_p"], outputfile_xdmf_path=xdmf_path, t=time_value)
    pp.write_phasefield_mixed_solution(domain, xdmf_path, w_fields, time_value, comm)
    if rank == 0:
        with open(events_log_path, "a") as f:
            f.write(f"{time_value:.8e} {label}\n")


def before_first_time_step():
    wm1.sub(1).x.array[:] = np.ones_like(wm1.sub(1).x.array[:])
    wrestart.x.array[:] = wm1.x.array[:]
    if rank == 0:
        sol.prepare_newton_logfile(logfile_path)
        pp.prepare_graphs_output_file(outputfile_graph_path)
        with open(columns_path, "w") as f:
            f.write(" ".join(f"{i}:{name}" for i, name in enumerate(COLUMNS)) + "\n")
        with open(events_log_path, "w") as f:
            f.write("# t label(pre_jump|post_jump) x_ct Jx\n")
    pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)
    if args.event_snapshots:
        pp.write_meshoutputfile(domain, outputfile_events_xdmf_path, comm)


def before_each_time_step(t, dt):
    if rank == 0:
        sol.print_time_and_dt(t, dt)


def after_timestep_success(t, dt, iters):
    dt_used.value = dt
    dD_visc = assemble(form_D_visc_inc)
    D_visc.value = D_visc.value + dD_visc

    if args.material == "vonmises":
        update_plastic_state_3D(u)
        dD_pl_deg = assemble(form_D_pl_deg_inc)
        dD_pl_undeg = assemble(form_D_pl_undeg_inc)
    else:
        dD_pl_deg = 0.0
        dD_pl_undeg = 0.0
    D_pl_deg.value = D_pl_deg.value + dD_pl_deg
    D_pl_undeg.value = D_pl_undeg.value + dD_pl_undeg

    sigma_interpolated.interpolate(sigma_expr)
    Rx_top, Ry_top, Rz_top = pp.reaction_force(sigma_interpolated, n=n, ds=ds_top_tagged(top_surface_tag), comm=comm)
    dW = pp.work_increment_external_forces(sigma_interpolated, u, um1, n, ds, comm=comm)
    Work.value = Work.value + dW

    A = assemble(form_A)
    E_el = assemble(form_E_el)
    E_el_micro = assemble(form_E_el_micro)
    E_pl = assemble(form_E_pl)
    E_pl_undeg = assemble(form_E_pl_undeg)
    E_frac = assemble(form_E_frac)
    E_frac_micro = assemble(form_E_frac_micro)

    if rank == 0:
        sol.write_to_newton_logfile(logfile_path, t, dt, iters)

    eshelby_interpolated.interpolate(eshelby_expr)
    Jx, Jy, Jz = alex.linearelastic.get_J_3D(eshelby_interpolated, n, ds=ds(external_surface_tag), comm=comm)
    Jx, Jy, Jz = Jx / thickness_z, Jy / thickness_z, Jz / thickness_z

    s_zero_for_tracking_at_nodes.interpolate(s)
    x_ct, max_y, max_z, min_x, min_y, min_z = pp.crack_bounding_box_3D(domain, pf.get_dynamic_crack_locator_function(wm1, s_zero_for_tracking_at_nodes), comm)

    s0.interpolate(s0_expr)
    sig_vm0.interpolate(sig_vm_expr)
    sig_h0.interpolate(sig_h_expr)
    sig_yy0.interpolate(sig_yy_expr)
    alpha0.interpolate(alpha0_expr)
    sig_vm_max = ligament_window_max(sig_vm0, x_ct)
    sig_h_max = ligament_window_max(sig_h0, x_ct)
    sig_yy_max = ligament_window_max(sig_yy0, x_ct)
    alpha_max = ligament_window_max(alpha0, x_ct)

    tz = thickness_z
    if rank == 0:
        print(pp.getJString(Jx, Jy, Jz))
        print("Crack tip position x: " + str(x_ct))
        pp.write_to_graphs_output_file(outputfile_graph_path, t, Jx, Jy, Jz, x_ct, xxK1.value[0], Rx_top, Ry_top, Rz_top,
                                       dW / tz, Work.value / tz, A / tz, dt,
                                       E_el / tz, E_el_micro / tz, E_pl / tz, E_pl_undeg / tz, E_frac / tz, E_frac_micro / tz,
                                       D_visc.value / tz, D_pl_deg.value / tz, D_pl_undeg.value / tz,
                                       sig_vm_max, sig_h_max, sig_yy_max, alpha_max, float(iters))

    step = int(success_timestep_counter.value)
    if args.event_snapshots:
        jumped = (x_ct - state["x_ct_prev"]) > args.jump_threshold_factor * epsilon.value
        if jumped and state["has_prev"] and (step - state["last_event_step"]) > 5:
            w_prev_fields.x.array[:] = wm1.x.array[:]
            sigma_prev_interpolated.interpolate(sigma_prev_expr)
            alpha_prev0.interpolate(alpha_prev0_expr)
            ep_prev_interpolated.interpolate(ep_prev_expr)
            ep_interpolated.interpolate(ep_expr)
            write_event_snapshot(outputfile_events_xdmf_path, state["t_prev"], w_prev_fields, sigma_prev_interpolated, alpha_prev0,
                                 f"pre_jump {state['x_ct_prev']:.6e} {state['Jx_prev']:.6e}", ep_fields=ep_prev_interpolated)
            write_event_snapshot(outputfile_events_xdmf_path, t, w, sigma_interpolated, alpha0, f"post_jump {x_ct:.6e} {Jx:.6e}", ep_fields=ep_interpolated)
            state["last_event_step"] = step
    state["x_ct_prev"] = x_ct
    state["t_prev"] = t
    state["Jx_prev"] = Jx
    state["has_prev"] = True

    if x_ct >= x_min_all + args.stop_fraction * (x_max_all - x_min_all):
        wm1.x.array[:] = w.x.array[:]
        wrestart.x.array[:] = w.x.array[:]
        raise StopSimulation("specimen separated completely")

    wm1.x.array[:] = w.x.array[:]
    wrestart.x.array[:] = w.x.array[:]
    success_timestep_counter.value = success_timestep_counter.value + 1.0
    if not int(success_timestep_counter.value) % int(postprocessing_interval.value) == 0:
        return
    pp.write_tensor_fields(domain, comm, [sigma_interpolated], ["sigma"], outputfile_xdmf_path=outputfile_xdmf_path, t=t)
    pp.write_field(domain, outputfile_xdmf_path, alpha_n, t, comm, S=S0)
    pp.write_field(domain, outputfile_xdmf_path, sig_vm0, t, comm, S=S0)
    pp.write_field(domain, outputfile_xdmf_path, sig_h0, t, comm, S=S0)
    ep_interpolated.interpolate(ep_expr)
    pp.write_tensor_fields(domain, comm, [ep_interpolated], ["eps_p"], outputfile_xdmf_path=outputfile_xdmf_path, t=t)
    pp.write_phasefield_mixed_solution(domain, outputfile_xdmf_path, w, t, comm)


def after_timestep_restart(t, dt, iters):
    w.x.array[:] = wrestart.x.array[:]


def after_last_timestep():
    pp.write_phasefield_mixed_solution(domain, outputfile_xdmf_path, w, t_global.value, comm)
    timer.stop()
    if rank == 0:
        runtime = timer.elapsed()
        sol.print_runtime(runtime)
        sol.write_runtime_to_newton_logfile(logfile_path, runtime)
        try:
            pp.print_graphs_plot(outputfile_graph_path, script_path, legend_labels=COLUMNS[1:])
        except Exception as e:
            print("print_graphs_plot failed (non-fatal):", e)


parameters_to_write = {
    'mesh_file': mesh_file, 'material': args.material,
    'lam_eff_simulation': la_effective, 'mue_eff_simulation': mu_effective, 'delta_cubic_simulation': delta_cubic, 'embedding': args.embedding,
    'lam_micro_simulation': la_micro, 'mue_micro_simulation': mu_micro, 'Gc_simulation': gc_micro,
    'sig_y_simulation': sig_y_micro_value, 'sig_y_embedding_simulation': sig_y_eff_value, 'hard_simulation': args.hard,
    'beta_simulation': args.beta, 'eps_simulation': eps_param, 'eps': epsilon.value, 'eta': eta.value, 'mob': Mob.value,
    'bc_moduli': args.bc_moduli, 'mu_bc_simulation': mu_bc_value, 'nu_bc_simulation': nu_bc_value, 'K_bc_simulation': args.K_bc,
    'element_order': 1, 'in_crack_length': in_crack_length, 'thickness_z': thickness_z,
    'dt_start': args.dt_start, 'dt_max': args.dt_max, 'postprocessing_interval': args.postprocessing_interval,
}
if rank == 0:
    pp.append_to_file(parameters=parameters_to_write, filename=parameter_path, comm=comm)


def create_timestamped_directory(base_dir="."):
    directory_name = os.path.join(base_dir, f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(directory_name, exist_ok=True)
    return directory_name


try:
    sol.solve_with_newton_adaptive_time_stepping(
        domain=domain, w=w, Tend=Tend,
        before_first_timestep_hook=before_first_time_step, after_last_timestep_hook=after_last_timestep,
        before_each_timestep_hook=before_each_time_step, get_residuum_and_gateaux=get_residuum_and_gateaux,
        get_bcs=get_bcs, after_timestep_restart_hook=after_timestep_restart, after_timestep_success_hook=after_timestep_success,
        comm=comm, print_bool=True, t=t_global, dt=dt_global, dt_max=dt_max, trestart=trestart_global,
    )
except StopSimulation as e:
    if rank == 0:
        print(f"Simulation stopped early: {e}")
    after_last_timestep()
finally:
    if rank == 0:
        target_directory = create_timestamped_directory(base_dir=script_path)
        for file in [parameter_path, outputfile_graph_path, columns_path, events_log_path, logfile_path,
                     os.path.join(script_path, script_name_without_extension + ".py"), os.path.join(script_path, "graphs.png")]:
            if os.path.exists(file):
                shutil.copy(file, target_directory)
        print(f"Light-weight results copied to {target_directory}")
