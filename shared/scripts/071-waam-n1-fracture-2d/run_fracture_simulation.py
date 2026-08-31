#!/usr/bin/env python3
"""
071 - 2D phase-field crack propagation through the explicitly modelled WAAM
N=1 microstructure, with J-integral evaluation and effective toughness.

Model
-----
* Elasticity: per cell a full 3x3 plane stiffness (Voigt [xx,yy,xy]).
  Inside the microstructure patch every GRAIN carries its own rotated
  single-crystal tensor (identical construction to the tensile load case of
  project 070, including the transition-zone prefactor s(x)); outside the
  patch the domain is homogeneous isotropic steel with ASSUMED constants.
* Fracture: `phasefield_anisotropic.StaticPhaseFieldProblem2D_anisotropic`,
  sigma = g(s,eta) C(x) eps, psi_el = 1/2 sigma:eps.
  Gc is CONSTANT everywhere - the elastic field is the only heterogeneity.
* Loading: surfing boundary condition (as in 067). A mode-I K-field with
  K = K_SCALE * sqrt(Gc * E'_emb) is prescribed on the outer boundary and its
  centre translates with v_crack along +x, driving quasi-steady crack growth.
* Evaluation: J = contour integral of the elastic Eshelby tensor over the
  outer boundary. In a heterogeneous solid that far-field J contains the
  crack-tip driving force AND the configurational forces of the microstructure
  inside the contour - which is exactly the effective (macroscopic) energy
  release rate. Its plateau during steady growth is the effective toughness
  Gc_eff (post-processed by `evaluate_gc_eff.py`).

Units: GPa, um.  1 GPa*um = 1 kJ/m^2.  K in GPa*sqrt(um)
(1 GPa*sqrt(um) = 31.62 MPa*sqrt(m)).

Run (inside the dolfinx container):
    mpirun -np 6 python3 run_fracture_simulation.py \
        --mesh_file mesh_fracture_micro.xdmf --micro micro_long.npz --tag long
"""
import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import ufl
import dolfinx as dlfx
from mpi4py import MPI
from petsc4py import PETSc as petsc

import alex.os
import alex.phasefield as pf
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol

import dolfinx_compat as cmp
import materials_fracture_2d as MF
import phasefield_anisotropic as pfa


class StopSimulation(Exception):
    pass


script_path = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]

comm, rank, size = alex.os.set_mpi()
alex.os.print_mpi_status(rank, size)

# ------------------------------------------------------------------ input ---
ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
ap.add_argument('--mesh_file', default='mesh_fracture_micro.xdmf')
ap.add_argument('--micro', default=None,
                help='micro_<tag>.npz; omit for a purely homogeneous run (verification)')
ap.add_argument('--config', default=None)
ap.add_argument('--tag', default='run')
ap.add_argument('--sfun', default='1.0',
                help='transition-zone stiffness prefactor s(x_map) [um], numpy expression')
ap.add_argument('--rotate_ccw90', action='store_true',
                help='rotate the microstructure patch by +90 deg (transverse crack case)')
ap.add_argument('--Gc', type=float, default=None, help='override config fracture.Gc_GPa_um')
ap.add_argument('--epsilon', type=float, default=None, help='override config fracture.epsilon_um')
ap.add_argument('--K_scale', type=float, default=1.0,
                help='K = K_scale * sqrt(Gc*E_emb_prime); >1 over-drives the crack')
ap.add_argument('--v_crack', type=float, default=None,
                help='speed of the K-field centre [um per unit time] (default Lx_dom/20)')
ap.add_argument('--a0', type=float, default=None,
                help='initial crack length measured from the left domain edge [um]. '
                     'Overrides --tip_setback.')
ap.add_argument('--k_start_x', type=float, default=None,
                help='x where the CENTRE of the applied K field starts [um]. '
                     'Default: the left domain edge x_min (as in 067). The field '
                     'then sweeps forward and only starts driving once it reaches '
                     'the real crack tip - a gentle ramp-in instead of full drive '
                     'at t=0. Set equal to the crack tip to drive immediately.')
ap.add_argument('--tip_setback', type=float, default=None,
                help='distance the initial crack tip is placed BEFORE the left '
                     'edge of the microstructure patch [um]. The crack then '
                     'starts entirely in the homogeneous embedding and runs into '
                     'the microstructure, which (a) keeps the whole process zone '
                     'out of the patch at t=0 and (b) produces a homogeneous J '
                     'plateau in the same run, right before the grains take over '
                     '- a per-run reference that cancels systematic bias. '
                     'Default: min(10*epsilon, 0.4 * left margin).')
ap.add_argument('--dt_start', type=float, default=None,
                help='initial time step (default dt_max/200 - as in 067 it must '
                     'start SMALL, the stepper only ever grows it)')
ap.add_argument('--dt_max', type=float, default=None,
                help='largest time step. Default 0.33*epsilon/v_crack, i.e. the '
                     'K field advances a third of the phase-field length per '
                     'step. Deliberately independent of Tend.')
ap.add_argument('--min_iters', type=int, default=5,
                help='dt is doubled when Newton needs FEWER iterations than '
                     'this. alex defaults to 4; with 4-5 iterations per step '
                     'dt then never grows again and the run stalls.')
ap.add_argument('--max_iters', type=int, default=8,
                help='more iterations than this -> the step is rejected and dt halved')
ap.add_argument('--Tend', type=float, default=None,
                help='end time. Only ever cuts a run short - separation ends it '
                     'by itself. Default: 3*Lx_domain/v_crack.')
ap.add_argument('--separation_frac', type=float, default=0.95,
                help='the run stops when the crack tip has crossed this fraction '
                     'of the domain ("specimen separated completely"). Going much '
                     'above 0.95 runs the tip into the surfing boundary condition.')
ap.add_argument('--postprocessing_interval', type=int, default=25)
ap.add_argument('--max_steps', type=int, default=0, help='0 = unlimited (smoke tests)')
ap.add_argument('--no_init_profile', dest='init_profile', action='store_false',
                help='start from a SHARP notch (s=1 everywhere, s=0 only on the '
                     'crack Dirichlet nodes) instead of the analytic phase-field '
                     'profile s = 1-exp(-d/2eps). Only for comparison - the sharp '
                     'notch wastes the first ~40 steps relaxing into the profile.')
ap.add_argument('--no_init_u', dest='init_u', action='store_false',
                help='start from u = 0 instead of the analytic surfing K-field. '
                     'Only for comparison - u = 0 makes the first Newton solve '
                     'jump to the full boundary condition in one step and is how '
                     'the first production run collapsed to dt = 9e-15.')
ap.add_argument('--u_degree', type=int, default=1, choices=(1, 2),
                help='polynomial degree of the displacement field')
ap.add_argument('--s_degree', type=int, default=1, choices=(1, 2),
                help='polynomial degree of the phase field. Safe here because '
                     '071 uses its own degree-agnostic irreversibility BC and '
                     'crack tracker instead of the vertex-based ones in `alex`.')
ap.add_argument('--quad_degree', type=int, default=None,
                help='cap the quadrature degree of the volume forms (the UFL '
                     'estimate gets high for quadratic elements x cubic '
                     'degradation). Try 4-6 with --u_degree 2.')
ap.add_argument('--stiffness_average', default='hill',
                choices=('none', 'voigt', 'reuss', 'hill'),
                help='how each element gets its stiffness. "none" = nearest-'
                     'pixel draw at the centroid (only valid while the element '
                     'is smaller than a grain - here the median 17-4PH grain is '
                     '5.4 um, so for h > ~5 um this produces element-scale '
                     'stiffness NOISE). "hill" (default) averages the pixels the '
                     'element actually covers.')
ap.add_argument('--crack_width', type=float, default=None,
                help='FULL width of the initial crack [um]: inside |y| < W/2 the '
                     'material is fully broken (s = 0, held by Dirichlet), '
                     'outside it the analytic profile continues. W = 0 is the '
                     'pure exponential profile on a single node row. A blunt '
                     'notch is closer to a machined starter crack and eases the '
                     'first steps. Default: 0 (or config fracture.crack_width_um).')
ap.add_argument('--eta', type=float, default=None,
                help='residual stiffness of fully broken material, g(0) = eta. '
                     'Larger = better conditioned, slightly stiffer crack.')
ap.add_argument('--degradation', default=None, choices=('quadratic', 'cubic'),
                help='override config fracture.degradation. quadratic = AT2, '
                     'g(s) = s^2 + eta (default here); cubic = the form used by '
                     '067. At the same Gc and epsilon the cubic one gives a '
                     'roughly 1.8x higher critical stress.')
ap.add_argument('--mobility', type=float, default=None,
                help='override config fracture.mobility. SMALLER = more viscous '
                     'regularisation of the phase field = more robust Newton at '
                     'the onset of propagation, at the price of rate dependence.')
ap.set_defaults(init_profile=True, init_u=True)
ap.add_argument('--outdir', default=None)
args = ap.parse_args()

outdir = args.outdir or script_path
os.makedirs(outdir, exist_ok=True)
logfile_path = os.path.join(outdir, f'{script_name}_{args.tag}_log.txt')
graph_path = os.path.join(outdir, f'{script_name}_{args.tag}_graphs.txt')
xdmf_path = os.path.join(outdir, f'{script_name}_{args.tag}.xdmf')
param_path = os.path.join(outdir, f'parameters_{args.tag}.txt')

cfg = MF.load_config(args.config, script_path)
if args.Gc is not None:
    cfg['fracture']['Gc_GPa_um'] = args.Gc
if args.epsilon is not None:
    cfg['fracture']['epsilon_um'] = args.epsilon
if args.mobility is not None:
    cfg['fracture']['mobility'] = args.mobility
if args.degradation is not None:
    cfg['fracture']['degradation'] = args.degradation
if args.eta is not None:
    cfg['fracture']['eta'] = args.eta
CRACK_W = (args.crack_width if args.crack_width is not None
           else float(cfg['fracture'].get('crack_width_um', 0.0)))
frac, emb = cfg['fracture'], cfg['embedding']
if rank == 0:
    print(MF.describe(cfg))

# ------------------------------------------------------------------- mesh ---
mesh_path = args.mesh_file if os.path.isabs(args.mesh_file) \
    else os.path.join(script_path, args.mesh_file)
with dlfx.io.XDMFFile(comm, mesh_path, 'r') as fh:
    domain = fh.read_mesh(name='Grid')

dim = domain.topology.dim
alex.os.mpi_print(f'spatial dimensions: {dim}', rank)
x_min, x_max, y_min, y_max, z_min, z_max = pp.compute_bounding_box(comm, domain)
if rank == 0:
    pp.print_bounding_box(rank, x_min, x_max, y_min, y_max, z_min, z_max)

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
cmap = domain.topology.index_map(tdim)
ncells_all = cmap.size_local + cmap.num_ghosts
cell_ids = np.arange(ncells_all, dtype=np.int32)
mid = dlfx.mesh.compute_midpoints(domain, tdim, cell_ids)


def cell_area(dom, ids):
    """Area of each cell, for the element-size-aware stiffness average."""
    v = dom.geometry.x[dom.geometry.dofmap[ids]][:, :, :2]
    return 0.5 * ((v[:, 1, 0] - v[:, 0, 0]) * (v[:, 2, 1] - v[:, 0, 1])
                  - (v[:, 2, 0] - v[:, 0, 0]) * (v[:, 1, 1] - v[:, 0, 1]))

# ------------------------------------------------- material fields per cell --
# Patch is centred on the crack line: origin = (0, -Ly/2) in the FE frame,
# exactly matching mesh_fracture_micro.py.
micro = None
if args.micro:
    micro_path = args.micro if os.path.isabs(args.micro) else os.path.join(script_path, args.micro)
    micro = MF.Microstructure(micro_path, cfg, sfun=MF.make_sfun(args.sfun),
                              rotate_ccw90=args.rotate_ccw90,
                              verbose=(rank == 0)).place(x_left=0.0, y_center=0.0)
    # per-cell size, for the element-wise stiffness average
    h_cell = np.sqrt(2.0 * np.abs(cell_area(domain, cell_ids)))
    sample = micro.sample_averaged(mid[:, 0], mid[:, 1], h_cell, cfg,
                                   scheme=args.stiffness_average)
    if rank == 0:
        Ei = sample['Ex'][sample['inside'] > 0]
        print(f'Steifigkeit: Schema "{args.stiffness_average}", '
              f'im Mittel {sample["n_pixels_averaged"].mean():.1f} EBSD-Pixel je '
              f'Element, E_x {Ei.min():.0f}...{Ei.max():.0f} GPa '
              f'(Streuung {Ei.std()/Ei.mean():.3f})')
else:
    Cemb = MF.embedding_C2D(cfg)
    n = ncells_all
    sample = dict(C=np.repeat(Cemb[None], n, axis=0),
                  Ex=np.full(n, MF.X.E_directional(Cemb)),
                  region=np.full(n, -1.0), phase=np.zeros(n),
                  grain_id=np.full(n, -1.0), s=np.ones(n), inside=np.zeros(n))
    if rank == 0:
        print('kein --micro: homogen-isotroper Verifikationslauf')

Cf = pfa.make_cell_tensor_function(domain, 'C2D_GPa')
pfa.set_cell_tensor(Cf, sample['C'], ncells_all)

def cell_scalar(name, values):
    f = pfa.make_cell_scalar_function(domain, name)
    pfa.set_cell_scalar(f, values, ncells_all)
    return f

# Gc is CONSTANT everywhere by design of this study, but kept as a field so
# that a Gc heterogeneity can be switched on later without touching the model.
gc = cell_scalar('gc_GPa_um', np.full(ncells_all, frac['Gc_GPa_um']))
Ex_f = cell_scalar('E_x_local_GPa', sample['Ex'])
region_f = cell_scalar('region', sample['region'])
phase_f = cell_scalar('phase_fcc1_bcc2', sample['phase'])
grain_f = cell_scalar('grain_id', sample['grain_id'])
s_pref_f = cell_scalar('s_prefactor', sample['s'])
patch_f = cell_scalar('in_patch', sample['inside'])

# --------------------------------------------------------------- constants --
epsilon = dlfx.fem.Constant(domain, frac['epsilon_um'])
eta = dlfx.fem.Constant(domain, frac['eta'])
Mob = dlfx.fem.Constant(domain, frac['mobility'])
iMob = dlfx.fem.Constant(domain, 1.0 / Mob.value)

E_emb, nu_emb = emb['E_GPa'], emb['nu']
mu_emb = E_emb / (2.0 * (1.0 + nu_emb))
if frac['plane_state'] == 'stress':
    kappa_emb = (3.0 - nu_emb) / (1.0 + nu_emb)
    E_prime = E_emb
else:
    kappa_emb = 3.0 - 4.0 * nu_emb
    E_prime = E_emb / (1.0 - nu_emb ** 2)

Gc_val = frac['Gc_GPa_um']
K1_val = args.K_scale * math.sqrt(Gc_val * E_prime)
K1 = dlfx.fem.Constant(domain, K1_val)
K1_MPa_sqrt_m = K1_val * MF.GPA_SQRT_UM_TO_MPA_SQRT_M
if rank == 0:
    print(f'Surfing-K: K = {args.K_scale} * sqrt(Gc*E\') = {K1_val:.4f} GPa*sqrt(um) '
          f'= {K1_MPa_sqrt_m:.2f} MPa*sqrt(m);  J_ref = K^2/E\' = '
          f'{K1_val**2/E_prime:.4f} GPa*um')

Lx_dom = x_max - x_min
v_crack = args.v_crack if args.v_crack is not None else Lx_dom / 20.0
tip_setback = 0.0
if args.a0 is not None:
    a0 = args.a0
elif micro is not None:
    # The patch starts at x = 0, so the left margin is -x_min. Place the tip
    # `tip_setback` BEFORE the patch: the initial crack and its whole process
    # zone are then in the homogeneous embedding, the crack reaches a steady
    # state there, and only afterwards enters the grains. The old default
    # (0.25*Lx_dom) instead put the tip 18 % INSIDE the patch.
    tip_setback = (args.tip_setback if args.tip_setback is not None
                   else min(10.0 * epsilon.value, 0.4 * (-x_min)))
    a0 = -x_min - tip_setback
else:
    a0 = 0.25 * Lx_dom
# Tend can only ever CUT THE RUN SHORT - a run that separates the specimen ends
# on the separation criterion below, whatever Tend is. So budget generously:
# 3 x the domain length of K-field travel. (Measured: the crack follows the
# K field at ~0.9 v_crack, so 1.6 was already enough, but the margin is free.)
Tend = args.Tend if args.Tend is not None else (Lx_dom * 3.0) / v_crack

# Centre of the applied K field at t = 0. It starts at the LEFT DOMAIN EDGE
# (067 does the same: crack_start = [0, ...] with a domain starting at x = 0),
# NOT on the crack tip. Consequence: at t = 0 the applied field belongs to a
# shorter crack than the real one, so the tip is under-driven and does not move
# until the field has swept forward to it. That is the ramp-in of the surfing
# method; centring it on the tip would apply the full drive instantly.
K_START_X = args.k_start_x if args.k_start_x is not None else x_min

# ------------------------------------------------------- spaces / functions --
# Element degrees are free (--u_degree / --s_degree). This project does NOT use
# the two vertex-based helpers of `alex` that would otherwise force P1:
#   * alex.phasefield.irreversibility_bc locates s dofs via
#     locate_entities(domain, 0, ...) - vertices only, so P2 mid-edge dofs
#     would never be pinned and a broken crack could heal there;
#   * alex.postprocessing.crack_bounding_box_2D indexes a mask of length
#     "number of s dofs" against domain.geometry.x (number of vertices).
# Both are replaced below by `irreversibility_bc_local` and `crack_tip_x`,
# which work off the collapsed s dofmap and its dof coordinates and are
# therefore degree-agnostic (and simpler).
Ve = cmp.vector_element(domain, degree=args.u_degree)
Se = cmp.scalar_element(domain, degree=args.s_degree)
W = cmp.functionspace(domain, cmp.mixed_element([Ve, Se]))

# volume measure for every form (potential, energies, surface area)
dxm = (ufl.dx if args.quad_degree is None else
       ufl.Measure('dx', domain=domain,
                   metadata={'quadrature_degree': args.quad_degree}))

w = dlfx.fem.Function(W)
u, s = w.split()
wrestart = dlfx.fem.Function(W)
wm1 = dlfx.fem.Function(W)
dw = ufl.TestFunction(W)
ddw = ufl.TrialFunction(W)

# dof indices of the two sub-fields inside the mixed vector. NEEDED: in dolfinx
# `w.sub(1).x.array` is NOT a view of the s-dofs only - it is the WHOLE mixed
# vector. Writing `w.sub(1).x.array[:] = 1.0` would therefore also set u = 1
# everywhere (that is what 067 does; harmless there, but wrong). With the
# collapsed dofmaps below we can address u and s separately and exactly.
SUB_U, DOFS_U = W.sub(0).collapse()
SUB_S, DOFS_S = W.sub(1).collapse()
DOFS_U = np.asarray(DOFS_U, dtype=np.int32)
DOFS_S = np.asarray(DOFS_S, dtype=np.int32)
S_DOF_XY = SUB_S.tabulate_dof_coordinates()[:, :2]   # node coords of the s dofs


def irreversibility_bc_local():
    """Pin every s dof that is ALREADY broken. Degree-agnostic replacement for
    `alex.phasefield.irreversibility_bc`, which searches vertices only and
    would therefore leave P2 mid-edge dofs free to heal."""
    wm1.x.scatter_forward()
    broken = np.isclose(wm1.x.array[DOFS_S], 0.0, atol=1.0e-3)
    return dlfx.fem.dirichletbc(dlfx.default_scalar_type(0.0),
                                DOFS_S[broken].astype(np.int32), W.sub(1))


def crack_tip_x():
    """Largest x among the broken s dofs. Degree-agnostic replacement for
    `alex.postprocessing.crack_bounding_box_2D` + the dynamic locator, which
    index an s-dof-length mask against the vertex coordinates."""
    m = w.x.array[DOFS_S] < 0.05
    loc = float(S_DOF_XY[m, 0].max()) if np.any(m) else -np.inf
    return comm.allreduce(loc, op=MPI.MAX)


def set_intact_state(fun, s_values=None, u_values=None):
    """Initial state of the mixed field: u (0 or the analytic K-field) and
    s (1 = intact, or the analytic crack profile)."""
    fun.x.array[DOFS_U] = 0.0 if u_values is None else u_values
    fun.x.array[DOFS_S] = 1.0 if s_values is None else s_values
    fun.x.scatter_forward()

problem = pfa.StaticPhaseFieldProblem2D_anisotropic(
    degradationFunction=pfa.get_degradation(frac['degradation'],
                                            frac['degradation_beta']),
    psisurf=pfa.psisurf_from_function, dx=dxm)

# ------------------------------------------------------- crack + surfing BC --
crack_tip_y = 0.0
crack_tip_x0 = x_min + a0
h_crack = 0.5 * frac['epsilon_um']


def crack(x):
    """Facets of the initial crack: |y - y_crack| <= W/2 and x < tip.
    W = 0 falls back to the single node row."""
    tol = 0.5 * CRACK_W + 1e-6 * (y_max - y_min) + 1e-9
    return np.logical_and(np.abs(x[1] - crack_tip_y) <= tol, x[0] < crack_tip_x0)


crackfacets = dlfx.mesh.locate_entities(domain, fdim, crack)
crackdofs = dlfx.fem.locate_dofs_topological(W.sub(1), fdim, crackfacets)
bccrack = dlfx.fem.dirichletbc(0.0, crackdofs, W.sub(1))

def crack_distance(px, py):
    """Distance to the initial crack segment: y = crack_tip_y, x from x_min to
    crack_tip_x0."""
    dy = np.asarray(py) - crack_tip_y
    px = np.asarray(px)
    dx = np.where(px > crack_tip_x0, px - crack_tip_x0,
                  np.where(px < x_min, px - x_min, 0.0))
    return np.hypot(dx, dy)


def initial_phasefield(px, py):
    """s(d) = 1 - exp(-d / (2*epsilon)).

    This is the exact 1D minimiser of the surface term
    Gc*[(1-s)^2/(4 eps) + eps |grad s|^2] (v = 1-s solves v'' = v/(4 eps^2),
    and the integral across the profile is 1 per unit crack length). Starting
    from it means the regularised crack already HAS its equilibrium width:
    A = crack length from the first step.

    Without it the initial crack is only the one node row where the Dirichlet
    condition sets s = 0. That sharp notch carries A/length = eps/h instead of
    1 - with eps = 8 um and h = 2 um that is 4x too much (the observed
    A: 860 -> 220 relaxation over the first ~40 steps, all of it wasted work).
    """
    d = np.maximum(crack_distance(px, py) - 0.5 * CRACK_W, 0.0)
    return 1.0 - np.exp(-d / (2.0 * epsilon.value))


n_crack_facets = comm.allreduce(len(crackfacets), MPI.SUM)
if rank == 0:
    print(f'Anfangsriss: y=0, x < {crack_tip_x0:.2f} um  ({n_crack_facets} Facetten), '
          f'v_crack={v_crack:.4g} um/t, Tend={Tend:.4g}')
    if n_crack_facets == 0:
        print('WARNUNG: keine Rissfacetten gefunden - liegt die Risslinie im Netz?')

t_global = dlfx.fem.Constant(domain, 0.0)
trestart_global = dlfx.fem.Constant(domain, 0.0)
# The time step is tied to the PHYSICS, not to Tend: per step the K field must
# not travel more than a fraction of the phase-field length, otherwise the
# crack cannot follow it. dt_max = DT_EPS_FRAC * epsilon / v_crack.
# (Coupling dt to Tend, as an earlier version did, meant that merely giving the
# run a longer time budget silently changed the time stepping - the working
# dt_max = 0.032 would have become 0.060.)
DT_EPS_FRAC = 0.33
dt_max_default = DT_EPS_FRAC * epsilon.value / v_crack
dt_start = args.dt_start if args.dt_start is not None else dt_max_default / 200.0
dt_max = dlfx.fem.Constant(domain, args.dt_max if args.dt_max is not None
                           else dt_max_default)
dt_global = dlfx.fem.Constant(domain, dt_start)
dt_min_stop = 1.0e-14

if rank == 0:
    # The viscous term iMob*(s-s_n)/dt competes with the phase-field driving
    # force ~ Gc/(4*epsilon). The ratio depends on the (arbitrary) time unit
    # set by v_crack, so it must be reported, not assumed.
    visc = (1.0 / Mob.value) / dt_max.value
    drive = Gc_val / (4.0 * epsilon.value)
    print(f'Ansatz: u = P{args.u_degree}, s = P{args.s_degree}, Quadratur = '
          f'{args.quad_degree if args.quad_degree else "UFL-Schaetzung"}')
    _x_stop = x_min + args.separation_frac * Lx_dom
    _t_lead = (crack_tip_x0 - K_START_X) / v_crack
    _t_need = (_x_stop - K_START_X) / v_crack
    print(f'K-Feld-Zentrum startet bei x = {K_START_X:.0f} um (linker Gebietsrand), '
          f'Rissspitze bei {crack_tip_x0:.0f} um')
    print(f'  -> Vorlaufzeit bis das Feld die Spitze erreicht: t = {_t_lead:.2f} '
          f'(solange ist der Riss unterbelastet und bewegt sich nicht)')
    print(f'Trennung bei x_ct >= {_x_stop:.0f} um -> {_x_stop-crack_tip_x0:.0f} um '
          f'Risswachstum, ~{_t_need:.1f} Zeiteinheiten bei v_crack={v_crack:.0f} um/t. '
          f'Tend={Tend:.0f} (Reserve x{Tend/max(_t_need,1e-9):.1f})')
    print(f'Zeitschritt: dt_start={dt_start:.3e}, dt_max={dt_max.value:.3e}, '
          f'Tend={Tend:.4g}  (dt waechst nur bei iters < {args.min_iters})')
    print(f'  bei dt_max wandert das K-Feld {v_crack*dt_max.value:.2f} um je Schritt '
          f'= {v_crack*dt_max.value/epsilon.value:.2f} * epsilon')
    print(f'Viskositaet: iMob/dt_max = {visc:.4g} gegen Gc/(4eps) = {drive:.4g} '
          f'-> Verhaeltnis {visc/drive:.3g} (klein = quasistatisch)')

xxK1 = dlfx.fem.Constant(domain, np.array([K_START_X, crack_tip_y, 0.0],
                                          dtype=dlfx.default_scalar_type))
w_D = dlfx.fem.Function(W)


def compute_surf_displacement():
    x = ufl.SpatialCoordinate(domain)
    dx_ = x[0] - xxK1[0]
    dy_ = x[1] - xxK1[1]
    r = ufl.sqrt(ufl.inner(dx_, dx_) + ufl.inner(dy_, dy_))
    theta = ufl.atan2(dy_, dx_)
    pref = K1 / (2.0 * mu_emb * math.sqrt(2.0 * math.pi)) * ufl.sqrt(r)
    u_x = pref * (kappa_emb - ufl.cos(theta)) * ufl.cos(0.5 * theta)
    u_y = pref * (kappa_emb - ufl.cos(theta)) * ufl.sin(0.5 * theta)
    return ufl.as_vector([u_x, u_y])


bc_expression = dlfx.fem.Expression(compute_surf_displacement(),
                                    W.sub(0).element.interpolation_points())


def initial_displacement(s_values):
    """Exact elastic equilibrium for the INITIAL phase field: one linear solve
    of  div( g(s0,eta) C(x) : eps(u) ) = 0  with the surfing K-field on the
    outer boundary.

    Why not simply interpolate the analytic K-field over the domain (which is
    what this function did first): the analytic mode-I field has its BRANCH CUT
    exactly on the crack line. A phase-field mesh has only ONE node per position
    there, and atan2(0, negative) = +pi hands that node the UPPER crack face
    displacement while its neighbour a few um below gets the lower one. For
    K = 47 GPa*sqrt(um) that is a 7.5 um jump across 4 um, i.e. a spurious
    strain of ~1.9 along the whole crack. The run then starts with J = 26.5
    instead of K^2/E' = 11.0, the crack is 2.4x over-driven, and Newton only
    converges once dt is small enough (7.8e-8) to freeze s completely.

    The linear solve has none of that: it is the true equilibrium for the given
    s, it accounts for the heterogeneous C, and the analytic field enters only
    on the outer boundary - where the crack-mouth band is excluded from the
    Dirichlet set anyway, so the branch cut is never evaluated.
    """
    from dolfinx.fem.petsc import LinearProblem
    s_fun = dlfx.fem.Function(SUB_S)
    s_fun.x.array[:] = 1.0 if s_values is None else s_values
    s_fun.x.scatter_forward()

    uu, vv = ufl.TrialFunction(SUB_U), ufl.TestFunction(SUB_U)
    g0 = problem.degradation_function(s_fun, eta)
    a = ufl.inner(ufl.dot(g0 * Cf, pfa.eps_voigt(uu)), pfa.eps_voigt(vv)) * dxm
    L = ufl.inner(dlfx.fem.Constant(domain, np.zeros(dim, dtype=dlfx.default_scalar_type)),
                  vv) * dxm

    u_bc = dlfx.fem.Function(SUB_U)
    u_bc.interpolate(dlfx.fem.Expression(compute_surf_displacement(),
                                         SUB_U.element.interpolation_points()))
    dofs_b = dlfx.fem.locate_dofs_topological(SUB_U, fdim, facets_at_boundary)
    lp = LinearProblem(a, L, bcs=[dlfx.fem.dirichletbc(u_bc, dofs_b)],
                       petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'})
    uh0 = lp.solve()
    if rank == 0:
        print('Startzustand: elastischer Vorabsolve fuer u durchgefuehrt '
              f'({SUB_U.dofmap.index_map.size_global * SUB_U.dofmap.index_map_bs} DOF)')
    return uh0.x.array
boundary_surfing = bc.get_2D_boundary_of_box_as_function(domain, comm, atol=0.0,
                                                         epsilon=epsilon.value)
facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, boundary_surfing)
dofs_at_boundary = dlfx.fem.locate_dofs_topological(W.sub(0), fdim, facets_at_boundary)


def get_bcs(t):
    xxK1.value = np.array([K_START_X + v_crack * t_global.value,
                           crack_tip_y, 0.0], dtype=dlfx.default_scalar_type)
    bcs = []
    w_D.sub(0).interpolate(bc_expression)
    bcs.append(dlfx.fem.dirichletbc(w_D, dofs_at_boundary))
    if abs(t) > sys.float_info.epsilon * 5:
        bcs.append(irreversibility_bc_local())
    bcs.append(bccrack)
    return bcs


def get_residuum_and_gateaux(delta_t):
    return problem.prep_newton(w=w, wm1=wm1, dw=dw, ddw=ddw, C=Cf, Gc=gc,
                               epsilon=epsilon, eta=eta, iMob=iMob,
                               delta_t=delta_t)


# ------------------------------------------------------------- measures ------
n = ufl.FacetNormal(domain)
external_tag = 5
external_tags = pp.tag_part_of_boundary(domain, boundary_surfing, external_tag)
ds = ufl.Measure('ds', domain=domain, subdomain_data=external_tags)
top_tag = 9
top_tags = pp.tag_part_of_boundary(
    domain, bc.get_top_boundary_of_box_as_function(domain, comm, atol=0.0), top_tag)
ds_top = ufl.Measure('ds', domain=domain, subdomain_data=top_tags)

S0 = cmp.functionspace(domain, cmp.scalar_element(domain, degree=0, family='DP'))
S = cmp.functionspace(domain, Se)
TEN = cmp.functionspace(domain, cmp.tensor_element(domain, shape=(dim, dim),
                                                   degree=0, family='DP'))

Work = dlfx.fem.Constant(domain, 0.0)
success_counter = dlfx.fem.Constant(domain, 0.0)
A_history = [0.0]                      # largest surface area seen so far
timer = dlfx.common.Timer()


def dt_as_float(dt):
    return float(np.asarray(getattr(dt, 'value', dt)).reshape(-1)[0])


def stop_if_dt_too_small(dt):
    v = dt_as_float(dt)
    if v < dt_min_stop:
        raise StopSimulation(f'time step dt={v:.3e} below {dt_min_stop:.1e}')


def assemble_J(eshelby, measure):
    trac = ufl.dot(eshelby, n)
    Jx = dlfx.fem.assemble_scalar(dlfx.fem.form(trac[0] * measure))
    Jy = dlfx.fem.assemble_scalar(dlfx.fem.form(trac[1] * measure))
    return comm.allreduce(Jx, MPI.SUM), comm.allreduce(Jy, MPI.SUM)


def write_material_fields(t):
    """Cell-wise material bookkeeping.

    Written at EVERY output time, not only at t=0. `alex.postprocessing` puts
    each field into its own temporal collection in the XDMF, so a field only
    exists at the times it was written - if the material fields sat at t=0
    alone, ParaView would show them at t=0 and nothing else, while u/s/sigma
    would be missing exactly there. Writing them every time makes each time
    step self-contained.

    The 3x3 stiffness Cf itself is not written (ParaView has no use for a
    non-square-tensor cell field); its two readable projections are
    `E_x_local_GPa` (grain-wise modulus along the crack direction) and
    `s_prefactor`. The grain structure is visible in `grain_id`.
    """
    for f in (gc, Ex_f, region_f, phase_f, grain_f, s_pref_f, patch_f):
        # pass an explicitly NAMED target: pp.write_field would otherwise build
        # an unnamed Function(S0) and every field would land in the XDMF under
        # the same default name.
        interp = dlfx.fem.Function(S0)
        interp.name = f.name
        pp.write_field(domain, xdmf_path, f, t, comm, S=S0, field_interp=interp)


def write_solution_fields(t, sig_int=None):
    """sigma, u and s at time t. Every call writes the SAME set of names, so
    the XDMF time series stays uniform."""
    if sig_int is None:
        sig_int = dlfx.fem.Function(TEN, name='sigma')
        sig_int.interpolate(dlfx.fem.Expression(
            problem.sigma_degraded(u, s, Cf, eta), TEN.element.interpolation_points()))
    pp.write_tensor_fields(domain, comm, [sig_int], ['sigma'], xdmf_path, t)
    pp.write_phasefield_mixed_solution(domain, xdmf_path, w, t, comm)


def before_first_time_step():
    timer.start()
    # Initial state in wm1 AND in w: the first Newton solve must start from
    # intact material, not from the default all-zero vector (s = 0 = fully
    # broken everywhere), which costs many iterations and shows up as a
    # DECREASING surface area A while the solver heals the domain back.
    s0 = initial_phasefield(S_DOF_XY[:, 0], S_DOF_XY[:, 1]) \
        if args.init_profile else None
    u0 = initial_displacement(s0) if args.init_u else None
    set_intact_state(wm1, s0, u0)
    set_intact_state(w, s0, u0)
    wrestart.x.array[:] = wm1.x.array[:]

    # self-check: with the analytic profile A must already equal the crack
    # length; with a sharp notch it is about eps/h times too large.
    _, s_now = ufl.split(w)
    A0 = pfa.get_surf_area(s_now, epsilon=epsilon, dx=dxm, comm=comm)
    L0 = crack_tip_x0 - x_min
    if rank == 0:
        how = 'analytisches Profil' if args.init_profile else 'scharfer Kerb'
        howu = 'K-Feld' if args.init_u else 'u=0'
        # Expected surface measure of the initial crack:
        #   flanks   L * (1 + W/(4 eps))          (exact, per unit length)
        #   tip cap  int f(r) * pi * r dr         (length-independent, and 17 %
        #                                          of the total at L = 14*eps)
        # The flank term alone is what the 1D formula gives - it is NOT the
        # value to compare against for a short crack.
        _e, _W = epsilon.value, CRACK_W
        _d = np.linspace(0.0, 0.5 * _W + 40.0 * _e, 20001)
        _t = np.maximum(_d - 0.5 * _W, 0.0)
        _s = 1.0 - np.exp(-_t / (2.0 * _e))
        _f = (1.0 - _s) ** 2 / (4.0 * _e) + _e * (np.exp(-_t / (2.0 * _e)) /
                                                  (2.0 * _e) * (_d > 0.5 * _W)) ** 2
        _g = _f * np.pi * _d          # trapezoid by hand: np.trapz is gone in
        A_tip = float(np.sum(0.5 * (_g[1:] + _g[:-1]) * np.diff(_d)))   # numpy 2.x, np.trapezoid absent in old ones
        A_soll = L0 * (1.0 + _W / (4.0 * _e)) + A_tip
        soll = A_soll / max(L0, 1e-9)
        print(f'Startzustand (s: {how}, u: {howu}): A = {A0:.1f}, '
              f'Risslaenge = {L0:.1f} um, Rissbreite W = {CRACK_W:.1f} um '
              f'-> A/L = {A0/max(L0,1e-9):.3f} (Soll {soll:.3f}: '
              f'Flanken {1.0 + _W/(4.0*_e):.3f} + Spitzenkappe {A_tip/max(L0,1e-9):.3f})')
        if micro is None:
            print(f'Rissspitze startet bei x = {crack_tip_x0:.1f} um')
        else:
            print(f'Rissspitze startet bei x = {crack_tip_x0:.1f} um, '
                  f'Patch 0 ... {micro.Lx:.0f} um')
            if crack_tip_x0 < -1e-9:
                print(f'  -> {-crack_tip_x0:.1f} um Vorlauf im homogenen '
                      f'Material ({-crack_tip_x0/epsilon.value:.1f} x epsilon), '
                      f'dann {micro.Lx:.0f} um Mikrostruktur')
                if -crack_tip_x0 < 2.0 * epsilon.value:
                    print('  WARNUNG: Vorlauf < 2*epsilon - die Prozesszone '
                          'ragt beim Start schon in den Patch')
            else:
                print(f'  WARNUNG: Spitze liegt {crack_tip_x0:.1f} um IM Patch, '
                      f'{100*crack_tip_x0/micro.Lx:.0f} % der Mikrostruktur '
                      f'werden uebersprungen')
            print(f'  Anfangsrisslaenge {L0:.1f} um = '
                  f'{L0/epsilon.value:.1f} x epsilon')
    if rank == 0:
        sol.prepare_newton_logfile(logfile_path)
        pp.prepare_graphs_output_file(graph_path)
    pp.write_meshoutputfile(domain, xdmf_path, comm)
    # t = 0 output: material fields AND the initial (u, s) state, so the first
    # time step in ParaView already carries every field.
    write_material_fields(0.0)
    write_solution_fields(0.0)


def before_each_time_step(t, dt):
    stop_if_dt_too_small(dt)
    if rank == 0:
        sol.print_time_and_dt(t, dt)


def after_timestep_success(t, dt, iters):
    um1, _ = ufl.split(wm1)
    stop_if_dt_too_small(dt)

    sigma = problem.sigma_degraded(u, s, Cf, eta)
    sig_int = dlfx.fem.Function(TEN, name='sigma')
    sig_int.interpolate(dlfx.fem.Expression(sigma, TEN.element.interpolation_points()))

    Rx_top, Ry_top = pp.reaction_force(sig_int, n=n, ds=ds_top(top_tag), comm=comm)
    dW = pp.work_increment_external_forces(sig_int, u, um1, n, ds, comm=comm)
    Work.value = Work.value + dW

    A = pfa.get_surf_area(s, epsilon=epsilon, dx=dxm, comm=comm)
    E_el = problem.get_E_el_global(s, eta, u, Cf, dxm, comm)
    E_surf = problem.get_E_surf_global(s, gc, epsilon, dxm, comm)

    Jx, Jy = assemble_J(problem.getEshelby(w, eta, Cf), ds(external_tag))

    if rank == 0:
        sol.write_to_newton_logfile(logfile_path, t, dt, iters)

    x_ct = crack_tip_x()

    s_min = comm.allreduce(float(np.min(w.x.array[DOFS_S])), MPI.MIN)
    if rank == 0:
        # A is the regularised crack surface; it must grow monotonically. A
        # falling A means the phase field is still relaxing out of a bad
        # initial state rather than propagating a crack.
        flag = '' if A >= A_history[0] - 1e-9 else '   <-- A faellt (Relaxation!)'
        A_history[0] = max(A_history[0], A)
        print(f'x_tip = {x_ct:9.3f}  Jx = {Jx:10.5f} (Gc = {Gc_val:.4g})  '
              f'A = {A:9.2f}  s_min = {s_min:6.4f}  iters = {iters}{flag}')
        pp.write_to_graphs_output_file(graph_path, t, Jx, Jy, x_ct,
                                       xxK1.value[0], Rx_top, Ry_top, dW,
                                       Work.value, A, dt, E_el, E_surf,
                                       E_el + E_surf, float(iters), s_min)

    if x_ct >= x_min + args.separation_frac * Lx_dom:
        raise StopSimulation('specimen separated completely')

    wm1.x.array[:] = w.x.array[:]
    wrestart.x.array[:] = w.x.array[:]
    success_counter.value = success_counter.value + 1.0
    if args.max_steps and success_counter.value >= args.max_steps:
        raise StopSimulation(f'max_steps={args.max_steps} reached (smoke test)')
    if int(success_counter.value) % max(int(args.postprocessing_interval), 1) != 0:
        return
    write_material_fields(t)
    write_solution_fields(t, sig_int)


def after_timestep_restart(t, dt, iters):
    stop_if_dt_too_small(dt)
    w.x.array[:] = wrestart.x.array[:]


GRAPH_LABELS = ['Jx', 'Jy', 'x_crack_tip', 'x_K_field', 'Rx_top', 'Ry_top',
                'dW', 'W', 'A_surf', 'dt', 'E_el', 'E_surf', 'E_total',
                'newton_iters', 's_min']


def after_last_timestep():
    write_material_fields(t_global.value)
    write_solution_fields(t_global.value)
    timer.stop()
    if rank == 0:
        runtime = timer.elapsed()
        sol.print_runtime(runtime)
        sol.write_runtime_to_newton_logfile(logfile_path, runtime)
        try:
            pp.print_graphs_plot(graph_path, outdir, legend_labels=GRAPH_LABELS)
        except Exception as exc:                       # plotting must never kill a run
            print(f'graph plot skipped: {exc}')


# ---------------------------------------------------------------- provenance -
def json_safe(o):
    """Recursively turn numpy scalars/arrays into plain Python types.

    Needed because the bounding box comes from `pp.compute_bounding_box` and
    the time steps from `dlfx.fem.Constant.value` - both numpy, both not JSON
    serialisable.
    """
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.item() if o.size == 1 else o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (str, bool, int, float)) or o is None:
        return o
    return str(o)


def write_run_meta(meta):
    with open(os.path.join(outdir, f'run_meta_{args.tag}.json'), 'w') as fh:
        json.dump(json_safe(meta), fh, indent=2)


run_meta = {
    'tag': args.tag, 'mesh_file': args.mesh_file, 'micro': args.micro,
    'sfun': args.sfun, 'rotate_ccw90': bool(args.rotate_ccw90),
    'Gc_GPa_um': Gc_val, 'epsilon_um': frac['epsilon_um'], 'eta': frac['eta'],
    'mobility': frac['mobility'], 'degradation': frac['degradation'],
    'degradation_beta': frac['degradation_beta'],
    'plane_state': frac['plane_state'],
    'embedding_E_GPa': E_emb, 'embedding_nu': nu_emb, 'E_prime_GPa': E_prime,
    'kappa': kappa_emb, 'K_scale': args.K_scale, 'K1_GPa_sqrt_um': K1_val,
    'K1_MPa_sqrt_m': K1_MPa_sqrt_m, 'J_reference_GPa_um': K1_val ** 2 / E_prime,
    'init_profile': bool(args.init_profile), 'init_u': bool(args.init_u),
    'stiffness_average': args.stiffness_average, 'crack_width_um': CRACK_W,
    'u_degree': args.u_degree, 's_degree': args.s_degree,
    'quad_degree': args.quad_degree,
    'min_iters': args.min_iters, 'max_iters': args.max_iters,
    'max_steps': args.max_steps, 'postprocessing_interval': args.postprocessing_interval,
    'stop_reason': 'laeuft noch / abgebrochen',
    'v_crack': v_crack, 'a0_um': a0, 'crack_tip_x0_um': crack_tip_x0,
    'separation_frac': args.separation_frac, 'k_start_x_um': K_START_X,
    'tip_setback_um': tip_setback, 'patch_x0_um': 0.0 if micro is not None else None,
    'Tend': Tend, 'dt_start': dt_start, 'dt_max': dt_max.value,
    'domain_um': [x_min, y_min, x_max, y_max],
    'graph_columns': ['t'] + GRAPH_LABELS,
    'mpi_ranks': size, 'started': datetime.now().isoformat(timespec='seconds'),
}
if micro is not None:
    run_meta['microstructure'] = micro.info
    # needed by evaluate_gc_eff.py to split J(x_tip) into the three regions
    run_meta['roi_um'] = micro.meta.get('roi_um')
    run_meta['zones_um'] = micro.meta.get('zones')
if rank == 0:
    write_run_meta(run_meta)
    pp.append_to_file(parameters={k: json_safe(v) for k, v in run_meta.items()
                                  if not isinstance(v, (dict, list))},
                      filename=param_path, comm=comm)

# ------------------------------------------------------------------- solve --
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
        comm=comm, print_bool=True, t=t_global, dt_max=dt_max,
        trestart=trestart_global,
        min_iters=args.min_iters, max_iters=args.max_iters)
    run_meta['stop_reason'] = f'Tend = {Tend:.6g} erreicht'
except StopSimulation as exc:
    run_meta['stop_reason'] = str(exc)
    if rank == 0:
        print(f'Simulation stopped early: {exc}')
    after_last_timestep()
except Exception as exc:
    run_meta['stop_reason'] = f'{type(exc).__name__}: {exc}'
    if rank == 0:
        print('Unhandled exception:', exc)
        write_run_meta(run_meta)
    raise
finally:
    if rank == 0:
        run_meta['finished'] = datetime.now().isoformat(timespec='seconds')
        run_meta['t_reached'] = float(t_global.value)
        run_meta['successful_steps'] = int(success_counter.value)
        run_meta['dt_final'] = dt_as_float(dt_global)
        write_run_meta(run_meta)
        print(f'Abbruchgrund: {run_meta["stop_reason"]}  '
              f'(t = {run_meta["t_reached"]:.6g} von {Tend:.6g}, '
              f'{run_meta["successful_steps"]} Schritte, '
              f'dt zuletzt {run_meta["dt_final"]:.3g})')
        print(f'Ergebnisse: {graph_path}\n            {xdmf_path}')
