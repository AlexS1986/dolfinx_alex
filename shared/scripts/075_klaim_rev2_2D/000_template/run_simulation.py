"""
Unified 2D surfing-boundary-condition fracture simulation for the KLAIM 2025 revision-2 reruns.

One script for all 2D studies of the paper
    "Influence of J2-plasticity on the effective crack resistance of a simplified porous material
     in a numerical experiment"  (Schlueter, Medda, Mueller)

Physics is identical to 046_plasticity/000_template/run_simulation.py (von Mises, Noll et al. 2020
formulation, cubic degradation, Allen-Cahn-type viscous regularisation, surfing BCs), with the
following switches (all default to the settings stated in the manuscript):

  --material vonmises|elastic   elastic = same formulation with yield stress -> infinity (no plastic
                                update), i.e. the brittle model used for the [Eq Jc] / [Eq sigma*] sets
  --beta                        slope parameter of the cubic degradation function (manuscript: 0.2)
  --eta                         residual stiffness (manuscript: 1e-5)
  --sig_y, --hard               yield stress and hardening modulus of the porous (micro) region
  --sig_y_embedding             yield stress in the homogeneous embedding (default 1e6 = linear elastic
                                embedding as described in the manuscript; set equal to --sig_y to
                                reproduce the behaviour of the old scripts, where sig_y was global)
  --embedding isotropic|cubic   constitutive law of the embedding region: isotropic (lam_eff, mu_eff as in
                                the paper) or full plane-strain cubic stiffness (C11, C12, C33 from
                                run_effective_stiffness.py)
  --bc_moduli effective|fixed   moduli used in the surfing K-field displacement: 'effective' = (E_eff, nu_eff)
                                of the embedding (what all previous runs did); 'fixed' = (E_bc, nu_bc)

Additional outputs (energy partition, dissipation, ligament stress state, event-triggered field snapshots)
are documented in README.md of the project folder and in run_simulation_graphs_columns.txt.
"""
import argparse
import dolfinx as dlfx
import os
from mpi4py import MPI
import numpy as np
import ufl

import alex.heterogeneous as het
import alex.os
import alex.phasefield as pf
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol
import alex.linearelastic
import alex.plasticity
import alex.util
import math

from petsc4py import PETSc as petsc
import sys
import basix

import shutil
from datetime import datetime


class StopSimulation(Exception):
    pass


dlfx.log.set_log_level(dlfx.log.LogLevel.INFO)

script_path = os.path.dirname(__file__)
script_name_without_extension = os.path.splitext(os.path.basename(__file__))[0]
logfile_path = alex.os.logfile_full_path(script_path, script_name_without_extension)
outputfile_graph_path = alex.os.outputfile_graph_full_path(script_path, script_name_without_extension)
outputfile_xdmf_path = alex.os.outputfile_xdmf_full_path(script_path, script_name_without_extension)
outputfile_events_xdmf_path = os.path.join(script_path, script_name_without_extension + "_events.xdmf")
events_log_path = os.path.join(script_path, script_name_without_extension + "_events.txt")
columns_path = os.path.join(script_path, script_name_without_extension + "_graphs_columns.txt")
parameter_path = os.path.join(script_path, "parameters.txt")

# set MPI environment
comm, rank, size = alex.os.set_mpi()
alex.os.print_mpi_status(rank, size)

if rank == 0:
    alex.util.print_dolfinx_version()

# ----------------------------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="2D surfing-BC phase-field fracture simulation (KLAIM rev. 2).")
parser.add_argument("--mesh_file", type=str, default="mesh_fracture_adaptive.xdmf")
parser.add_argument("--in_crack_length", type=float, required=True, help="Initial crack length (tip x-coordinate)")
parser.add_argument("--lam_micro_param", type=float, default=1.0)
parser.add_argument("--mue_micro_param", type=float, default=1.0)
parser.add_argument("--gc_micro_param", type=float, default=1.0)
parser.add_argument("--eps_param", type=float, default=0.1)
parser.add_argument("--element_order", type=int, default=1)
parser.add_argument("--material", type=str, default="vonmises", choices=["vonmises", "elastic"])
parser.add_argument("--sig_y", type=float, default=1.0, help="initial yield stress (micro region)")
parser.add_argument("--hard", type=float, default=0.2222222, help="hardening modulus H (micro region)")
parser.add_argument("--sig_y_embedding", type=float, default=1.0e6, help="yield stress in the embedding region")
parser.add_argument("--beta", type=float, default=0.2, help="beta of the cubic degradation function")
parser.add_argument("--eta", type=float, default=1.0e-5, help="residual stiffness")
parser.add_argument("--mob", type=float, default=1000.0, help="mobility M")
parser.add_argument("--embedding", type=str, default="isotropic", choices=["isotropic", "cubic"])
parser.add_argument("--bc_moduli", type=str, default="effective", choices=["effective", "fixed"],
                    help="fixed: K-field with E_bc, nu_bc (manuscript: 2.5 mu0, 0.25); effective: (E_eff, nu_eff) of the embedding (old runs)")
parser.add_argument("--E_bc", type=float, default=2.5, help="only for --bc_moduli fixed")
parser.add_argument("--nu_bc", type=float, default=0.25, help="only for --bc_moduli fixed")
parser.add_argument("--K_bc", type=float, default=math.sqrt(2.5), help="K_I^bc, default sqrt(2.5 Jc0 mu0)")
parser.add_argument("--postprocessing_interval", type=int, default=400, help="field output every N steps")
parser.add_argument("--event_snapshots", type=int, default=1, help="1 = write fields before/after crack jumps")
parser.add_argument("--jump_threshold_factor", type=float, default=4.0, help="crack jump = dx_ct > factor*eps")
parser.add_argument("--stop_fraction", type=float, default=0.9, help="stop when x_ct >= fraction * x_max")
parser.add_argument("--dt_max", type=float, default=0.01)
parser.add_argument("--dt_start", type=float, default=2.0e-7)
args = parser.parse_args()

mesh_file = args.mesh_file
in_crack_length = args.in_crack_length
la_micro = args.lam_micro_param
mu_micro = args.mue_micro_param
gc_micro = args.gc_micro_param
gc_matrix = gc_micro
eps_param = args.eps_param

parameters = pp.read_parameters_file(parameter_path)
la_effective = parameters["lam_effective"]
mu_effective = parameters["mue_effective"]
wsteg = parameters["wsteg"]
dhole = parameters["dhole"]
w_cell = wsteg + dhole

if args.embedding == "cubic":
    for key in ("C11_effective", "C12_effective", "C33_effective"):
        if key not in parameters:
            raise RuntimeError(f"--embedding cubic requires {key} in parameters.txt (run the new run_effective_stiffness.py)")
    C11_eff = parameters["C11_effective"]
    C12_eff = parameters["C12_effective"]
    C33_eff = parameters["C33_effective"]
    # isotropic part: lam = C12, mu = C33 (as in the paper); cubic correction on the normal components
    la_effective = C12_eff
    mu_effective = C33_eff
    delta_cubic = C11_eff - C12_eff - 2.0 * C33_eff
else:
    delta_cubic = 0.0

if rank == 0:
    print(f"Initial crack length: {in_crack_length}")
    print(f"material={args.material} beta={args.beta} eta={args.eta} sig_y={args.sig_y} hard={args.hard} "
          f"embedding={args.embedding} (delta_cubic={delta_cubic:.4e}) bc_moduli={args.bc_moduli}")

# ----------------------------------------------------------------------------------------------
# Mesh and material regions
# ----------------------------------------------------------------------------------------------
with dlfx.io.XDMFFile(comm, os.path.join(script_path, mesh_file), 'r') as mesh_inp:
    domain = mesh_inp.read_mesh(name="Grid")
    mesh_tags = mesh_inp.read_meshtags(domain, name="Grid")

dim = domain.topology.dim
tdim = dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
alex.os.mpi_print('spatial dimensions: ' + str(dim), rank)

x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all = pp.compute_bounding_box(comm, domain)
if rank == 0:
    pp.print_bounding_box(rank, x_min_all, x_max_all, y_min_all, y_max_all, z_min_all, z_max_all)

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

# Simulation parameters
dt_global = dlfx.fem.Constant(domain, args.dt_start)
t_global = dlfx.fem.Constant(domain, 0.0)
trestart_global = dlfx.fem.Constant(domain, 0.0)
dt_max = dlfx.fem.Constant(domain, args.dt_max)

# ----------------------------------------------------------------------------------------------
# Function spaces
# ----------------------------------------------------------------------------------------------
Ve = basix.ufl.element("P", domain.basix_cell(), 1, shape=(domain.geometry.dim,))  # displacements
Se = basix.ufl.element("P", domain.basix_cell(), 1, shape=())  # fracture field
W = dlfx.fem.functionspace(domain, basix.ufl.mixed_element([Ve, Se]))

w = dlfx.fem.Function(W)
u, s = w.split()
wrestart = dlfx.fem.Function(W)
wm1 = dlfx.fem.Function(W)
um1, sm1 = ufl.split(wm1)
dw = ufl.TestFunction(W)
ddw = ufl.TrialFunction(W)

S = dlfx.fem.functionspace(domain, Se)
S0e = basix.ufl.element("DP", domain.basix_cell(), 0, shape=())
S0 = dlfx.fem.functionspace(domain, S0e)

deg_quad = 1  # quadrature degree for internal state variable representation
(_, alpha_n, alpha_tmp,
 e_p_11_n, e_p_22_n, e_p_12_n, e_p_33_n,
 e_p_11_n_tmp, e_p_22_n_tmp, e_p_12_n_tmp, e_p_33_n_tmp) = alex.plasticity.define_internal_state_variables_basix_2D(domain, deg_quad, quad_scheme="default")
dx = alex.plasticity.define_custom_integration_measure_that_matches_quadrature_degree_and_scheme(domain, deg_quad, "default")
dx_tagged = ufl.Measure("dx", domain=domain, subdomain_data=mesh_tags,
                        metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"})
quadrature_points, cells = alex.plasticity.get_quadraturepoints_and_cells_for_inter_polation_at_gauss_points(domain, deg_quad)

e_p_n_3D = ufl.as_tensor([[e_p_11_n, e_p_12_n, 0.0],
                          [e_p_12_n, e_p_22_n, 0.0],
                          [0.0, 0.0, e_p_33_n]])
e_p_n_3D_tmp = ufl.as_tensor([[e_p_11_n_tmp, e_p_12_n_tmp, 0.0],
                              [e_p_12_n_tmp, e_p_22_n_tmp, 0.0],
                              [0.0, 0.0, e_p_33_n_tmp]])


# ----------------------------------------------------------------------------------------------
# Phase-field problem (Noll et al. formulation); optional cubic correction in the embedding
# ----------------------------------------------------------------------------------------------
class PhaseFieldProblemCubicEmbedding(pf.StaticPhaseFieldProblem_plasticity_noll):
    """Adds  delta * marker_eff * diag(eps11, eps22)  to the stress of the embedding cells, i.e.
    C11 = lam + 2 mu + delta, C12 = lam, C33 = mu  (plane-strain cubic symmetry). delta = 0 -> isotropic."""

    def __init__(self, delta, marker_eff, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta
        self.marker_eff = marker_eff

    def sigma_undegraded(self, u, lam, mu):
        sig = super().sigma_undegraded(u, lam, mu)
        eps2 = ufl.sym(ufl.grad(u))
        corr = self.delta * self.marker_eff * ufl.as_tensor([[eps2[0, 0], 0.0], [0.0, eps2[1, 1]]])
        return sig + corr

    def psiel_undegraded(self, u, la, mu):
        w_el = super().psiel_undegraded(u, la, mu)
        eps2 = ufl.sym(ufl.grad(u))
        return w_el + 0.5 * self.delta * self.marker_eff * (eps2[0, 0] ** 2 + eps2[1, 1] ** 2)


degradation = pf.cubic_degradation(beta=args.beta)
phaseFieldProblem = PhaseFieldProblemCubicEmbedding(
    delta=delta_cubic_const, marker_eff=marker_eff,
    degradationFunction=degradation, psisurf=pf.psisurf_from_function, dx=dx,
    sig_y=sig_y, hard=hard.value, alpha_n=alpha_n, e_p_n=e_p_n_3D)


def psi_pl_undegraded(alpha):
    return 0.5 * hard * alpha ** 2 + sig_y * alpha


# ----------------------------------------------------------------------------------------------
# Initial crack and surfing boundary conditions
# ----------------------------------------------------------------------------------------------
crack_tip_start_location_x = in_crack_length
crack_tip_start_location_y = 0.0


def crack(x):
    x_log = x[0] < (crack_tip_start_location_x)
    y_log = np.isclose(x[1], crack_tip_start_location_y, atol=0.01 * dhole)
    return np.logical_and(y_log, x_log)


v_crack = 1.0
Tend = (x_max_all - 0.0) * 2.0 / v_crack

crackfacets = dlfx.mesh.locate_entities(domain, fdim, crack)
crackdofs = dlfx.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)
bccrack = dlfx.fem.dirichletbc(0.0, crackdofs, W.sub(1))

K1 = dlfx.fem.Constant(domain, args.K_bc)
vcrack_const = dlfx.fem.Constant(domain, np.array([v_crack, 0.0, 0.0], dtype=dlfx.default_scalar_type))
crack_start = dlfx.fem.Constant(domain, np.array([0.0, crack_tip_start_location_y, 0.0], dtype=dlfx.default_scalar_type))
w_D = dlfx.fem.Function(W)
xxK1 = dlfx.fem.Constant(domain, np.array([0.0, 0.0, 0.0], dtype=dlfx.default_scalar_type))

if args.bc_moduli == "fixed":
    mu_bc_value = args.E_bc / (2.0 * (1.0 + args.nu_bc))
    nu_bc_value = args.nu_bc
else:  # effective moduli of the embedding (behaviour of all previous runs)
    mu_bc_value = mu_effective
    nu_bc_value = alex.linearelastic.get_nu(lam=la_effective, mu=mu_effective)
mu_bc = dlfx.fem.Constant(domain, mu_bc_value)
nu_bc = dlfx.fem.Constant(domain, nu_bc_value)


def compute_surf_displacement():
    # plane-strain mode-I K-field, kappa = 3 - 4 nu
    x = ufl.SpatialCoordinate(domain)
    dxx = x[0] - xxK1[0]
    dyy = x[1] - xxK1[1]
    r = ufl.sqrt(ufl.inner(dxx, dxx) + ufl.inner(dyy, dyy))
    theta = ufl.atan2(dyy, dxx)
    kappa = 3.0 - 4.0 * nu_bc
    u_x = K1 / (2.0 * mu_bc * math.sqrt(2.0 * math.pi)) * ufl.sqrt(r) * (kappa - ufl.cos(theta)) * ufl.cos(0.5 * theta)
    u_y = K1 / (2.0 * mu_bc * math.sqrt(2.0 * math.pi)) * ufl.sqrt(r) * (kappa - ufl.cos(theta)) * ufl.sin(0.5 * theta)
    return ufl.as_vector([u_x, u_y])


atol = (x_max_all - x_min_all) * 0.000
bc_expression = dlfx.fem.Expression(compute_surf_displacement(), W.sub(0).element.interpolation_points())
boundary_surfing_bc = bc.get_2D_boundary_of_box_as_function(domain, comm, atol=atol * 0.0, epsilon=epsilon.value)
facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, boundary_surfing_bc)
dofs_at_boundary = dlfx.fem.locate_dofs_topological(W.sub(0), fdim, facets_at_boundary)


def get_bcs(t):
    xxK1.value = np.array([crack_start.value[0] + vcrack_const.value[0] * t_global.value,
                           crack_start.value[1] + vcrack_const.value[1] * t_global.value,
                           0.0], dtype=dlfx.default_scalar_type)
    bcs = []
    w_D.sub(0).interpolate(bc_expression)
    bc_surf = dlfx.fem.dirichletbc(w_D, dofs_at_boundary)
    if abs(t) > sys.float_info.epsilon * 5:  # irreversibility, not before first step
        bcs.append(pf.irreversibility_bc(domain, W, wm1))
    bcs.append(bccrack)
    bcs.append(bc_surf)
    return bcs


def get_residuum_and_gateaux(delta_t: dlfx.fem.Constant):
    [Res, dResdw] = phaseFieldProblem.prep_newton(
        w=w, wm1=wm1, dw=dw, ddw=ddw, lam=la, mu=mu,
        Gc=gc, epsilon=epsilon, eta=eta,
        iMob=iMob, delta_t=delta_t)
    return [Res, dResdw]


# ----------------------------------------------------------------------------------------------
# Post-processing setup
# ----------------------------------------------------------------------------------------------
n = ufl.FacetNormal(domain)
external_surface_tag = 5
external_surface_tags = pp.tag_part_of_boundary(domain, boundary_surfing_bc, external_surface_tag)
ds = ufl.Measure('ds', domain=domain, subdomain_data=external_surface_tags,
                 metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"})
top_surface_tag = 9
top_surface_tags = pp.tag_part_of_boundary(domain, bc.get_top_boundary_of_box_as_function(domain, comm, atol=atol * 0.0), top_surface_tag)
ds_top_tagged = ufl.Measure('ds', domain=domain, subdomain_data=top_surface_tags,
                            metadata={"quadrature_degree": deg_quad, "quadrature_scheme": "default"})

s_zero_for_tracking_at_nodes = dlfx.fem.Function(S)
c = dlfx.fem.Constant(domain, petsc.ScalarType(1))
s_zero_for_tracking_at_nodes.interpolate(dlfx.fem.Expression(c, S.element.interpolation_points()))

TEN = dlfx.fem.functionspace(domain, ("DP", 0, (dim, dim)))
sigma_interpolated = dlfx.fem.Function(TEN)
sigma_interpolated.name = "sigma"
eshelby_interpolated = dlfx.fem.Function(TEN)
eshelby_interpolated.name = "eshelby"
sig_vm0 = dlfx.fem.Function(S0); sig_vm0.name = "sigma_vM"
sig_h0 = dlfx.fem.Function(S0); sig_h0.name = "sigma_h"
sig_yy0 = dlfx.fem.Function(S0); sig_yy0.name = "sigma_yy"
s0 = dlfx.fem.Function(S0); s0.name = "s_cell"
alpha0 = dlfx.fem.Function(S0); alpha0.name = "alpha"

# cell midpoints (local cells) for the ligament window evaluation
num_local_cells = domain.topology.index_map(tdim).size_local
midpoints = dlfx.mesh.compute_midpoints(domain, tdim, np.arange(num_local_cells, dtype=np.int32))
micro_mask_local = marker_micro.x.array[:num_local_cells] > 0.5

# cumulative quantities
Work = dlfx.fem.Constant(domain, 0.0)
D_visc = dlfx.fem.Constant(domain, 0.0)
D_pl_deg = dlfx.fem.Constant(domain, 0.0)
D_pl_undeg = dlfx.fem.Constant(domain, 0.0)
success_timestep_counter = dlfx.fem.Constant(domain, 0.0)
postprocessing_interval = dlfx.fem.Constant(domain, float(args.postprocessing_interval))

# precompiled forms (quantities that do not depend on t explicitly)
dt_used = dlfx.fem.Constant(domain, 1.0)  # set to the accepted time step before assembling the viscous increment
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
# Full plane-strain stress (incl. sigma_zz) directly from the constitutive law, degraded, as a 3x3 DG0 tensor
TEN3 = dlfx.fem.functionspace(domain, ("DP", 0, (3, 3)))
sigma3_interpolated = dlfx.fem.Function(TEN3)
sigma3_interpolated.name = "sigma3D"


def sigma3D_plane_strain(u_, s_, e_p_tensor, alpha_field):
    """3x3 degraded stress for plane strain (sigma_zz from the 3D plasticity stress), isotropic part only.
    The cubic correction (embedding cells only) is irrelevant for the ligament evaluation (micro cells)."""
    eps_np1 = alex.plasticity.assemble_3D_representation_of_eps(u_)
    e_np1 = ufl.dev(eps_np1)
    s_tr = 2.0 * mu * (e_np1 - e_p_tensor)
    norm_s_tr_val = ufl.sqrt(ufl.inner(s_tr, s_tr))
    norm_s_tr = ufl.conditional(ufl.lt(norm_s_tr_val, 1.0e-12), 1.0e-12, norm_s_tr_val)
    f_tr = norm_s_tr_val - np.sqrt(2.0 / 3.0) * (sig_y + hard * alpha_field)
    dgamma = f_tr / (2.0 * (mu + hard / 3.0))
    N_np1 = s_tr / norm_s_tr
    s_np1 = ufl.conditional(ufl.le(f_tr, 0.0), s_tr, s_tr - 2.0 * mu * dgamma * N_np1)
    K = alex.linearelastic.get_K(lam=la, mu=mu)
    return degradation(s_, eta) * (K * ufl.tr(eps_np1) * ufl.Identity(3) + s_np1)


sigma3_expr = dlfx.fem.Expression(sigma3D_plane_strain(u, s, e_p_n_3D, alpha_n), TEN3.element.interpolation_points())
sig_vm_expr = dlfx.fem.Expression(alex.linearelastic.sigvM(sigma3_interpolated), S0.element.interpolation_points())
sig_h_expr = dlfx.fem.Expression(ufl.tr(sigma3_interpolated) / 3.0, S0.element.interpolation_points())
sig_yy_expr = dlfx.fem.Expression(sigma_interpolated[1, 1], S0.element.interpolation_points())
s0_expr = dlfx.fem.Expression(s, S0.element.interpolation_points())
alpha0_expr = dlfx.fem.Expression(alpha_n, S0.element.interpolation_points())

# pre-jump state helpers (for event snapshots)
w_prev_fields = dlfx.fem.Function(W)
sigma_prev_interpolated = dlfx.fem.Function(TEN)
sigma_prev_interpolated.name = "sigma"
u_prev, s_prev = ufl.split(w_prev_fields)
sigma_prev_expr = dlfx.fem.Expression(
    degradation(s_prev, eta) * alex.plasticity.sig_plasticity(u_prev, e_p_n=e_p_n_3D_tmp, alpha_n=alpha_tmp, sig_y=sig_y, hard=hard, lam=la, mu=mu),
    TEN.element.interpolation_points())
alpha_prev0 = dlfx.fem.Function(S0); alpha_prev0.name = "alpha"
alpha_prev0_expr = dlfx.fem.Expression(alpha_tmp, S0.element.interpolation_points())
# plastic strain as a 3x3 cell-wise (DG0) tensor: deg_quad = 1 -> one quadrature point per cell, so this is exact
ep_interpolated = dlfx.fem.Function(TEN3); ep_interpolated.name = "eps_p"
ep_expr = dlfx.fem.Expression(e_p_n_3D, TEN3.element.interpolation_points())
ep_prev_interpolated = dlfx.fem.Function(TEN3); ep_prev_interpolated.name = "eps_p"
ep_prev_expr = dlfx.fem.Expression(e_p_n_3D_tmp, TEN3.element.interpolation_points())

timer = dlfx.common.Timer()
state = {"x_ct_prev": in_crack_length, "t_prev": 0.0, "Jx_prev": 0.0, "Jx_running_max": -1.0e30,
         "n_event_snapshots": 0, "last_event_step": -10 ** 9, "has_prev": False}

COLUMNS = ["t", "Jx", "Jy", "x_ct", "x_bc", "Rx_top", "Ry_top", "dW", "W", "A", "dt",
           "E_el", "E_el_micro", "E_pl", "E_pl_undeg", "E_frac", "E_frac_micro",
           "D_visc", "D_pl_deg", "D_pl_undeg",
           "sig_vM_max_lig", "sig_h_max_lig", "sig_yy_max_lig", "alpha_max_lig", "newton_iters"]


def ligament_window_max(field0: dlfx.fem.Function, x_ct: float):
    """max of a DG0 field over intact micro cells in the window ahead of the crack tip."""
    vals = field0.x.array[:num_local_cells]
    xw0 = x_ct - 0.5 * epsilon.value
    xw1 = x_ct + 1.5 * w_cell
    mask = (micro_mask_local & (midpoints[:, 0] >= xw0) & (midpoints[:, 0] <= xw1)
            & (np.abs(midpoints[:, 1]) <= 0.6 * dhole) & (s0.x.array[:num_local_cells] > 0.9))
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
    timer.start()
    wm1.sub(1).x.array[:] = np.ones_like(wm1.sub(1).x.array[:])
    wrestart.x.array[:] = wm1.x.array[:]
    if rank == 0:
        sol.prepare_newton_logfile(logfile_path)
        pp.prepare_graphs_output_file(outputfile_graph_path)
        with open(columns_path, "w") as f:
            f.write(" ".join(f"{i}:{name}" for i, name in enumerate(COLUMNS)) + "\n")
        with open(events_log_path, "w") as f:
            f.write("# t label(pre_jump|post_jump|new_Jx_max) x_ct Jx\n")
    pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)
    if args.event_snapshots:
        pp.write_meshoutputfile(domain, outputfile_events_xdmf_path, comm)


def before_each_time_step(t, dt):
    if rank == 0:
        sol.print_time_and_dt(t, dt)


def after_timestep_success(t, dt, iters):
    # --- viscous dissipation increment (uses s of this step and s of the previous converged step)
    dt_used.value = dt
    dD_visc = assemble(form_D_visc_inc)
    D_visc.value = D_visc.value + dD_visc

    # --- plastic update (alpha_tmp / e_p_*_tmp keep the previous state afterwards)
    if args.material == "vonmises":
        alex.plasticity.update_e_p_n_and_alpha_arrays_2D(u, e_p_11_n_tmp, e_p_22_n_tmp, e_p_12_n_tmp, e_p_33_n_tmp,
                                                         e_p_11_n, e_p_22_n, e_p_12_n, e_p_33_n,
                                                         alpha_tmp, alpha_n, domain, cells, quadrature_points, sig_y, hard, mu)
        dD_pl_deg = assemble(form_D_pl_deg_inc)
        dD_pl_undeg = assemble(form_D_pl_undeg_inc)
    else:
        dD_pl_deg = 0.0
        dD_pl_undeg = 0.0
    D_pl_deg.value = D_pl_deg.value + dD_pl_deg
    D_pl_undeg.value = D_pl_undeg.value + dD_pl_undeg

    # --- stresses, reaction forces, external work
    sigma_interpolated.interpolate(sigma_expr)
    Rx_top, Ry_top = pp.reaction_force(sigma_interpolated, n=n, ds=ds_top_tagged(top_surface_tag), comm=comm)
    dW = pp.work_increment_external_forces(sigma_interpolated, u, um1, n, ds, comm=comm)
    Work.value = Work.value + dW

    # --- energies
    A = assemble(form_A)
    E_el = assemble(form_E_el)
    E_el_micro = assemble(form_E_el_micro)
    E_pl = assemble(form_E_pl)
    E_pl_undeg = assemble(form_E_pl_undeg)
    E_frac = assemble(form_E_frac)
    E_frac_micro = assemble(form_E_frac_micro)

    if rank == 0:
        sol.write_to_newton_logfile(logfile_path, t, dt, iters)

    # --- J-integral at the outer boundary (elastic Eshelby tensor, as in the paper)
    eshelby_interpolated.interpolate(eshelby_expr)
    Jx, Jy = alex.linearelastic.get_J_2D(eshelby_interpolated, n, ds=ds(external_surface_tag), comm=comm)

    # --- crack tip position
    s_zero_for_tracking_at_nodes.interpolate(s)
    max_x, max_y, min_x, min_y = pp.crack_bounding_box_2D(domain, pf.get_dynamic_crack_locator_function(wm1, s_zero_for_tracking_at_nodes), comm)
    x_ct = max_x

    # --- ligament stress state ahead of the crack tip (intact micro cells only)
    s0.interpolate(s0_expr)
    sigma3_interpolated.interpolate(sigma3_expr)
    sig_vm0.interpolate(sig_vm_expr)
    sig_h0.interpolate(sig_h_expr)
    sig_yy0.interpolate(sig_yy_expr)
    alpha0.interpolate(alpha0_expr)
    sig_vm_max = ligament_window_max(sig_vm0, x_ct)
    sig_h_max = ligament_window_max(sig_h0, x_ct)
    sig_yy_max = ligament_window_max(sig_yy0, x_ct)
    alpha_max = ligament_window_max(alpha0, x_ct)

    if rank == 0:
        print("Crack tip position x: " + str(x_ct))
        pp.write_to_graphs_output_file(outputfile_graph_path, t, Jx, Jy, x_ct, xxK1.value[0], Rx_top, Ry_top, dW, Work.value, A, dt,
                                       E_el, E_el_micro, E_pl, E_pl_undeg, E_frac, E_frac_micro,
                                       D_visc.value, D_pl_deg.value, D_pl_undeg.value,
                                       sig_vm_max, sig_h_max, sig_yy_max, alpha_max, float(iters))

    # --- event-triggered field snapshots: state just before and just after a crack jump
    step = int(success_timestep_counter.value)
    if args.event_snapshots:
        jumped = (x_ct - state["x_ct_prev"]) > args.jump_threshold_factor * epsilon.value
        if jumped and state["has_prev"] and (step - state["last_event_step"]) > 5:
            # pre-jump state: previous converged step (wm1 still holds it, alpha_tmp/e_p_*_tmp hold previous plastic state)
            w_prev_fields.x.array[:] = wm1.x.array[:]
            sigma_prev_interpolated.interpolate(sigma_prev_expr)
            alpha_prev0.interpolate(alpha_prev0_expr)
            ep_prev_interpolated.interpolate(ep_prev_expr)
            ep_interpolated.interpolate(ep_expr)
            write_event_snapshot(outputfile_events_xdmf_path, state["t_prev"], w_prev_fields, sigma_prev_interpolated, alpha_prev0,
                                 f"pre_jump {state['x_ct_prev']:.6e} {state['Jx_prev']:.6e}", ep_fields=ep_prev_interpolated)
            write_event_snapshot(outputfile_events_xdmf_path, t, w, sigma_interpolated, alpha0,
                                 f"post_jump {x_ct:.6e} {Jx:.6e}", ep_fields=ep_interpolated)
            state["n_event_snapshots"] += 1
            state["last_event_step"] = step
    state["x_ct_prev"] = x_ct
    state["t_prev"] = t
    state["Jx_prev"] = Jx
    state["has_prev"] = True
    if Jx > state["Jx_running_max"]:
        state["Jx_running_max"] = Jx

    if x_ct >= args.stop_fraction * x_max_all:
        # update before stopping so that after_last_timestep writes a consistent final state
        wm1.x.array[:] = w.x.array[:]
        wrestart.x.array[:] = w.x.array[:]
        raise StopSimulation("specimen separated completely")

    # update
    wm1.x.array[:] = w.x.array[:]
    wrestart.x.array[:] = w.x.array[:]
    success_timestep_counter.value = success_timestep_counter.value + 1.0
    if not int(success_timestep_counter.value) % int(postprocessing_interval.value) == 0:
        return

    pp.write_tensor_fields(domain, comm, [sigma_interpolated], ["sigma"], outputfile_xdmf_path=outputfile_xdmf_path, t=t)
    pp.write_field(domain, outputfile_xdmf_path, alpha_n, t, comm, S=S0)
    pp.write_field(domain, outputfile_xdmf_path, sig_vm0, t, comm, S=S0)
    pp.write_field(domain, outputfile_xdmf_path, sig_h0, t, comm, S=S0)
    if args.material == "vonmises":
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
    'mesh_file': mesh_file,
    'material': args.material,
    'lam_eff_simulation': la_effective,
    'mue_eff_simulation': mu_effective,
    'delta_cubic_simulation': delta_cubic,
    'embedding': args.embedding,
    'lam_micro_simulation': la_micro,
    'mue_micro_simulation': mu_micro,
    'Gc_simulation': gc_matrix,
    'sig_y_simulation': sig_y_micro_value,
    'sig_y_embedding_simulation': sig_y_eff_value,
    'hard_simulation': args.hard,
    'beta_simulation': args.beta,
    'eps_simulation': eps_param,
    'eps': epsilon.value,
    'eta': eta.value,
    'mob': Mob.value,
    'bc_moduli': args.bc_moduli,
    'mu_bc_simulation': mu_bc_value,
    'nu_bc_simulation': nu_bc_value,
    'K_bc_simulation': args.K_bc,
    'element_order': 1,
    'in_crack_length': in_crack_length,
    'dt_max': args.dt_max,
    'postprocessing_interval': args.postprocessing_interval,
}


def create_timestamped_directory(base_dir="."):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory_name = os.path.join(base_dir, f"simulation_{timestamp}")
    os.makedirs(directory_name, exist_ok=True)
    return directory_name


def copy_files_to_directory(files, target_directory):
    for file in files:
        if os.path.exists(file):
            shutil.copy(file, target_directory)
        else:
            print(f"Warning: File '{file}' does not exist and will not be copied.")


if rank == 0:
    pp.append_to_file(parameters=parameters_to_write, filename=parameter_path, comm=comm)

try:
    sol.solve_with_newton_adaptive_time_stepping(
        domain, w, Tend, dt_global,
        before_first_timestep_hook=before_first_time_step,
        after_last_timestep_hook=after_last_timestep,
        before_each_timestep_hook=before_each_time_step,
        get_residuum_and_gateaux=get_residuum_and_gateaux,
        get_bcs=get_bcs,
        after_timestep_restart_hook=after_timestep_restart,
        after_timestep_success_hook=after_timestep_success,
        comm=comm, print_bool=True, t=t_global, dt_max=dt_max, trestart=trestart_global,
    )
except StopSimulation as e:
    if rank == 0:
        print(f"Simulation stopped early: {e}")
    after_last_timestep()
except Exception as e:
    if rank == 0:
        print("Unhandled exception:", e)
    raise
finally:
    if rank == 0:
        files_to_copy = [
            parameter_path, outputfile_graph_path, columns_path, events_log_path,
            os.path.join(script_path, script_name_without_extension + ".py"),
            os.path.join(script_path, "graphs.png"),
            logfile_path,
        ]
        target_directory = create_timestamped_directory(base_dir=script_path)
        print(f"Created directory: {target_directory}")
        copy_files_to_directory(files_to_copy, target_directory)
        print("Files copied successfully.")
