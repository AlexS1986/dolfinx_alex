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
import dolfinx.fem.petsc
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
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
ap.add_argument('--max_iters', type=int, default=12,
                help='more iterations than this -> the step is rejected and dt '
                     'halved. Was 8; with the absolute tolerance below a few '
                     'more iterations are cheaper than a rejected step.')
# --- Newton tolerances -------------------------------------------------------
# WHY (2026-09-02): all four production runs (s1 AND sgauss, both directions)
# died the same death: the crack tip stalls for a moment (stick-slip at a grain
# boundary, or leaving the stiff Gaussian patch), Newton fails once at
# dt ~ 5e-4, and from then on EVERY smaller dt fails too, down to 1e-14.
# That cascade is not physics. The dolfinx NewtonSolver was rebuilt each step
# with its defaults (rtol = 1e-9 RELATIVE to the initial residual of the step,
# atol = 1e-10). The initial residual of a step is essentially the boundary
# increment v_crack*dt, i.e. it shrinks WITH dt. Once dt is small, the target
# 1e-9 * r0 lies below the round-off floor of the LU solve on a 600k-DOF
# ill-conditioned mixed system, so no iteration count can ever reach it -
# the smaller dt gets, the more certain the failure.
# FIX: one solver, built once and reused, with an ABSOLUTE tolerance tied to
# the problem scale: atol = newton_atol_rel * R_ref, where R_ref is the largest
# initial step residual seen so far in the run (a step at dt_max). A tiny step
# then converges to the same absolute equilibrium accuracy as a normal one,
# instead of chasing an ever-smaller relative target.
ap.add_argument('--newton_rtol', type=float, default=1.0e-9,
                help='relative residual tolerance of Newton (dolfinx default)')
ap.add_argument('--newton_atol', type=float, default=None,
                help='ABSOLUTE residual tolerance [GPa*um]. Default: adaptive, '
                     'newton_atol_rel * (largest initial step residual so far).')
ap.add_argument('--newton_atol_rel', type=float, default=1.0e-8,
                help='adaptive absolute tolerance = this * R_ref (see above). '
                     '1e-8 of the largest step imbalance of the run is far below '
                     'anything J or the reaction forces can resolve, but safely '
                     'above the LU round-off floor.')
ap.add_argument('--newton_relax', type=float, default=1.0,
                help='Newton relaxation (damping) parameter, 1 = full step '
                     '(--solver newton only)')
ap.add_argument('--solver', default='newton', choices=('newton', 'snes'),
                help='"newton": dolfinx NewtonSolver (plain Newton, optional '
                     'constant relaxation). "snes": PETSc SNES newtonls with '
                     'backtracking line search - the standard remedy when '
                     'plain Newton diverges at a crack jump (the energy is '
                     'non-convex there; a full Newton step overshoots, the '
                     'line search does not). Same tolerances, same LU/MUMPS.')
ap.add_argument('--snes_monitor', action='store_true',
                help='print the SNES residual per iteration (rank 0)')
ap.add_argument('--dt_regrow_steps', type=int, default=5,
                help='after a rejected step, dt may not exceed the halved value '
                     'until this many steps in a row have converged; then the '
                     'ceiling doubles (up to dt_max). Stops the ping-pong '
                     '"fail at 2dt, converge at dt in 3 iterations, alex doubles, '
                     'fail at 2dt, ..." in which every second solve is wasted. '
                     '0 = alex behaviour (double immediately).')
ap.add_argument('--checkpoint_interval', type=int, default=50,
                help='write a binary restart checkpoint (w, wm1, t, dt, work, '
                     'counters; one .npz per rank) every N successful steps into '
                     '<outdir>/ckpt_<tag>/. 0 = off. A checkpoint is also written '
                     'when the run ends.')
ap.add_argument('--restart', action='store_true',
                help='continue from <outdir>/ckpt_<tag>/ instead of t = 0. Needs '
                     'the SAME mesh, the same number of MPI ranks and the same '
                     'model parameters (the partition fingerprint is checked). '
                     'graphs/log files are truncated to the checkpoint time and '
                     'continued; fields go to a new <tag>_restartN.xdmf.')
ap.add_argument('--restart_from_xdmf', default=None, metavar='H5[:TIME]',
                help='continue from the nodal u and s stored in the .h5 of an '
                     'EARLIER run that has no checkpoint (the four 2026-09-01 '
                     'production runs). Default TIME = last one in the file, '
                     'which is the last CONVERGED state (after_last_timestep '
                     'writes wrestart). Nodes are matched by coordinates, so '
                     'the mesh must be the same but -np may differ. W, A and the '
                     'step count are taken from the graphs file at that time; '
                     'dt restarts at --dt_start. Self-check: the first printed '
                     'x_tip/J must continue the old graphs seamlessly.')
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
CKPT_DIR = os.path.join(outdir, f'ckpt_{args.tag}')
if args.restart or args.restart_from_xdmf:
    # never append to / overwrite the XDMF of the previous leg - a new file
    # per restart leg, numbered
    _k = 1
    while os.path.exists(os.path.join(outdir, f'{script_name}_{args.tag}_restart{_k}.xdmf')):
        _k += 1
    xdmf_path = os.path.join(outdir, f'{script_name}_{args.tag}_restart{_k}.xdmf')

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
    # statistics over ALL ranks - a single rank may own no patch cell at all
    Ei = sample['Ex'][sample['inside'] > 0]
    _n = comm.allreduce(int(Ei.size), MPI.SUM)
    _mn = comm.allreduce(float(Ei.min()) if Ei.size else np.inf, MPI.MIN)
    _mx = comm.allreduce(float(Ei.max()) if Ei.size else -np.inf, MPI.MAX)
    _s1 = comm.allreduce(float(Ei.sum()), MPI.SUM)
    _s2 = comm.allreduce(float((Ei ** 2).sum()), MPI.SUM)
    _npx = comm.allreduce(float(sample['n_pixels_averaged'].sum()), MPI.SUM)
    _nc = comm.allreduce(int(sample['n_pixels_averaged'].size), MPI.SUM)
    if rank == 0 and _n > 0:
        _mean = _s1 / _n
        _std = math.sqrt(max(_s2 / _n - _mean ** 2, 0.0))
        print(f'Steifigkeit: Schema "{args.stiffness_average}", '
              f'im Mittel {_npx/max(_nc,1):.1f} EBSD-Pixel je '
              f'Element, E_x {_mn:.0f}...{_mx:.0f} GPa '
              f'(Streuung {_std/_mean:.3f})')
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
    newton_prepare_step(bcs)      # solver holds NEWTON_BCS, refreshed in place
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


# ------------------------------------------------ Newton solver (built once) --
# One NonlinearProblem + NewtonSolver for the whole run instead of a fresh
# default solver every step (see the --newton_* help text for the reason).
# The residual/Jacobian forms depend on dt only through the Constant
# `dt_global`, so they can be compiled once. The boundary conditions change
# every step (surfing field, irreversibility set): NonlinearProblem keeps a
# reference to the list object it was given, so `NEWTON_BCS` is refreshed IN
# PLACE by get_bcs() right before each solve.
NEWTON_BCS = []
_Res_form, _Jac_form = get_residuum_and_gateaux(dt_global)
nl_problem = NonlinearProblem(_Res_form, w, NEWTON_BCS, _Jac_form)


class SNESNewton:
    """PETSc SNES (newtonls + backtracking line search) behind the same
    `solve(w) -> (iters, converged)` interface alex expects from a dolfinx
    NewtonSolver; raises RuntimeError on non-convergence like it does.
    Residual/Jacobian assembly is delegated to the dolfinx NonlinearProblem
    (same forms, same in-place bc list, same Dirichlet lifting)."""

    def __init__(self, comm, problem, w, max_it, rtol, atol, monitor=False):
        self.problem, self.w = problem, w
        self.max_it, self.rtol, self.atol = max_it, rtol, atol
        self.A = dlfx.fem.petsc.create_matrix(problem.a)
        self.b = dlfx.fem.petsc.create_vector(problem.L)
        self.snes = petsc.SNES().create(comm)
        self.snes.setOptionsPrefix('pf_')
        self.snes.setFunction(self._F, self.b)
        self.snes.setJacobian(self._J, self.A)
        self.snes.setType('newtonls')
        self.snes.getLineSearch().setType('bt')
        ksp = self.snes.getKSP()
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')
        if monitor and comm.Get_rank() == 0:
            self.snes.setMonitor(lambda snes, it, r: print(f'    SNES {it:2d}: |R| = {r:.3e}'))
        self.snes.setFromOptions()

    def _sync(self, x):
        x.ghostUpdate(addv=petsc.InsertMode.INSERT, mode=petsc.ScatterMode.FORWARD)
        x.copy(_petsc_vec(self.w))
        self.w.x.scatter_forward()

    def _F(self, snes, x, F):
        self._sync(x)
        self.problem.F(x, F)

    def _J(self, snes, x, J, P):
        self._sync(x)
        self.problem.J(x, J)

    def solve(self, w):
        self.snes.setTolerances(rtol=self.rtol, atol=self.atol, max_it=self.max_it)
        x = _petsc_vec(w).copy()
        self.snes.solve(None, x)
        its = self.snes.getIterationNumber()
        reason = self.snes.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(f'SNES did not converge (reason {reason}) '
                               f'after {its} iterations')
        self._sync(x)
        return its, True


def _petsc_vec(fun):
    return fun.x.petsc_vec if hasattr(fun.x, 'petsc_vec') else fun.vector


if args.solver == 'snes':
    newton = SNESNewton(comm, nl_problem, w, args.max_iters, args.newton_rtol,
                        args.newton_atol if args.newton_atol is not None else 0.0,
                        monitor=args.snes_monitor)
else:
    newton = NewtonSolver(comm, nl_problem)
    newton.convergence_criterion = 'residual'
    newton.max_it = args.max_iters
    newton.rtol = args.newton_rtol
    newton.atol = args.newton_atol if args.newton_atol is not None else 0.0
    newton.relaxation_parameter = args.newton_relax
    newton.error_on_nonconvergence = True     # alex catches the RuntimeError -> dt/2
    newton.report = True
res_vec = dlfx.fem.petsc.create_vector(nl_problem.L)
NEWTON_STATE = {'R_ref': 0.0, 'r0': 0.0, 'atol': 0.0, 'last_iters': -1, 't_ok': None,
                'dt_ceiling': None, 'ok_in_row': 0, 'steps_at_start': 0.0}


def residual_norm():
    """||R(w)|| with the CURRENT boundary conditions applied - exactly the
    quantity the NewtonSolver tests against its tolerances."""
    w.x.scatter_forward()
    nl_problem.F(_petsc_vec(w), res_vec)
    return float(res_vec.norm())


def newton_prepare_step(bcs):
    """Refresh the bc list the solver holds, measure the initial residual of
    the step and set the absolute tolerance from the run's residual scale."""
    NEWTON_BCS[:] = bcs
    r0 = residual_norm()
    NEWTON_STATE['r0'] = r0
    NEWTON_STATE['R_ref'] = max(NEWTON_STATE['R_ref'], r0)
    if args.newton_atol is None:
        NEWTON_STATE['atol'] = args.newton_atol_rel * NEWTON_STATE['R_ref']
        newton.atol = NEWTON_STATE['atol']
    else:
        NEWTON_STATE['atol'] = args.newton_atol


# ---------------------------------------------------------- checkpointing --
def _ckpt_signature():
    """Partition fingerprint: a checkpoint may only be loaded onto the very
    same dof layout (same mesh file, same number of ranks)."""
    return np.array([size, rank, w.x.array.size, DOFS_S.size, DOFS_U.size,
                     float(S_DOF_XY[:, 0].sum()), float(S_DOF_XY[:, 1].sum()),
                     float(np.abs(S_DOF_XY).sum())])


def _ckpt_file(r=None):
    return os.path.join(CKPT_DIR, f'state_rank{(rank if r is None else r):04d}.npz')


def write_checkpoint(t_now):
    if rank == 0:
        os.makedirs(CKPT_DIR, exist_ok=True)
    comm.Barrier()
    fn = _ckpt_file()
    tmp = fn[:-4] + '_tmp.npz'
    np.savez(tmp, w=w.x.array, wm1=wm1.x.array,
             t=float(t_now), trestart=float(trestart_global.value),
             dt=dt_as_float(dt_global), Work=float(Work.value),
             A_hist=float(A_history[0]), steps=float(success_counter.value),
             R_ref=float(NEWTON_STATE['R_ref']), sig=_ckpt_signature(),
             xdmf=os.path.basename(xdmf_path))
    os.replace(tmp, fn)
    comm.Barrier()
    if rank == 0:
        print(f'Checkpoint geschrieben: {CKPT_DIR} (t = {t_now:.6g}, '
              f'{int(success_counter.value)} Schritte, dt = {dt_as_float(dt_global):.3g})')


def _truncate_table(path, t_max):
    """Keep header/comment lines and rows with t <= t_max (first column)."""
    if not os.path.exists(path):
        return
    keep = []
    with open(path) as fh:
        for line in fh:
            st = line.strip()
            if not st or st.startswith('#'):
                keep.append(line)
                continue
            try:
                tv = float(st.split()[0])
            except ValueError:
                keep.append(line)
                continue
            if tv <= t_max * (1.0 + 1e-12) + 1e-300:
                keep.append(line)
    with open(path, 'w') as fh:
        fh.writelines(keep)
        fh.write(f'# restart {datetime.now().isoformat(timespec="seconds")} '
                 f'from checkpoint t = {t_max:.10g}\n')


def _graphs_row_at(path, t_target):
    """(W, A, n_rows) of the graphs file at the row closest to t_target."""
    rows = []
    with open(path) as fh:
        for line in fh:
            st = line.strip()
            if not st or st.startswith('#'):
                continue
            rows.append([float(v) for v in st.split()])
    a = np.asarray(rows)
    k = int(np.argmin(np.abs(a[:, 0] - t_target)))
    # columns: t Jx Jy x_ct x_K Rx Ry dW W A dt ...
    return float(a[k, 8]), float(a[k, 9]), k + 1, float(a[k, 0])


def load_from_xdmf(spec):
    """Restart from nodal u, s of an earlier run's .h5 (see --restart_from_xdmf)."""
    import h5py
    if ':' in spec and not os.path.exists(spec):
        h5path, t_spec = spec.rsplit(':', 1)
        t_spec = float(t_spec)
    else:
        h5path, t_spec = spec, None
    if not os.path.isabs(h5path):
        h5path = os.path.join(outdir, h5path)
    with h5py.File(h5path, 'r') as f:
        keys = list(f['Function/s'].keys())
        times = {k: float(k.replace('_', '.')) for k in keys}
        if t_spec is None:
            key = max(keys, key=lambda k: times[k])
        else:
            key = min(keys, key=lambda k: abs(times[k] - t_spec))
        t_ck = times[key]
        xyz = np.asarray(f['Mesh/Grid/geometry'])[:, :2]
        s_h5 = np.asarray(f['Function/s'][key]).reshape(-1)
        u_h5 = np.asarray(f['Function/u'][key])
        u_h5 = u_h5.reshape(u_h5.shape[0], -1)[:, :2]
    if xyz.shape[0] != s_h5.size or xyz.shape[0] != u_h5.shape[0]:
        raise RuntimeError('--restart_from_xdmf: node count of geometry and '
                           'fields differ - not a P1 nodal dataset?')
    # match nodes by (rounded) coordinates - same mesh file, any partition
    scale = 1e-6 * max(x_max - x_min, y_max - y_min)
    def keyof(arr):
        q = np.round(arr / scale).astype(np.int64)
        return q[:, 0] * np.int64(1 << 31) + q[:, 1]
    lut = dict(zip(keyof(xyz).tolist(), range(xyz.shape[0])))
    def lookup(coords):
        idx = np.empty(coords.shape[0], dtype=np.int64)
        for i, k in enumerate(keyof(coords).tolist()):
            j = lut.get(k)
            if j is None:
                raise RuntimeError(f'--restart_from_xdmf: node {coords[i]} not found in h5 geometry')
            idx[i] = j
        return idx
    js = lookup(S_DOF_XY)
    ju = lookup(SUB_U.tabulate_dof_coordinates()[:, :2])
    w.x.array[DOFS_S] = s_h5[js]
    uvals = u_h5[ju]                          # (n_nodes, 2) -> interleaved
    w.x.array[DOFS_U] = uvals.reshape(-1)
    w.x.scatter_forward()
    wm1.x.array[:] = w.x.array[:]
    wrestart.x.array[:] = w.x.array[:]
    for f_ in (wm1, wrestart):
        f_.x.scatter_forward()
    W_old, A_old, n_rows, t_row = _graphs_row_at(graph_path, t_ck)
    Work.value = W_old
    A_history[0] = A_old
    success_counter.value = float(n_rows)
    NEWTON_STATE['steps_at_start'] = float(success_counter.value)
    NEWTON_STATE['t_ok'] = t_ck
    dt_global.value = dt_start
    trestart_global.value = t_ck
    # seed the residual scale with what a step at dt_max would see, so that
    # the absolute tolerance is meaningful from the first (small) step on
    t_global.value = t_ck + dt_max.value
    get_bcs(t_global.value)
    NEWTON_STATE['R_ref'] = NEWTON_STATE['r0']
    t_global.value = t_ck + dt_start
    if rank == 0:
        print(f'RESTART aus {h5path} @ t = {t_ck:.6g} (Datensatz {key}): '
              f'W = {W_old:.6g}, A = {A_old:.2f} (graphs-Zeile t = {t_row:.6g}, '
              f'{n_rows} Schritte), dt = {dt_start:.3g}, R_ref = {NEWTON_STATE["R_ref"]:.4g}')
        _truncate_table(graph_path, t_ck)
        _truncate_table(logfile_path, t_ck)
    comm.Barrier()
    return t_ck


def load_checkpoint():
    fn = _ckpt_file()
    if not os.path.exists(fn):
        raise RuntimeError(f'--restart: no checkpoint {fn}')
    d = np.load(fn)
    sig_now, sig_ck = _ckpt_signature(), d['sig']
    ok = (sig_now[:5] == sig_ck[:5]).all() and np.allclose(sig_now[5:], sig_ck[5:],
                                                            rtol=1e-10, atol=1e-6)
    ok = comm.allreduce(bool(ok), MPI.LAND)
    if not ok:
        raise RuntimeError('--restart: checkpoint does not match this mesh '
                           'partition (same mesh file and same -np needed)')
    w.x.array[:] = d['w']
    wm1.x.array[:] = d['wm1']
    wrestart.x.array[:] = d['wm1']
    for f in (w, wm1, wrestart):
        f.x.scatter_forward()
    Work.value = float(d['Work'])
    A_history[0] = float(d['A_hist'])
    success_counter.value = float(d['steps'])
    NEWTON_STATE['steps_at_start'] = float(success_counter.value)
    NEWTON_STATE['R_ref'] = float(d['R_ref'])
    t_ck = float(d['t'])
    NEWTON_STATE['t_ok'] = t_ck
    # the checkpoint is the converged state AT t_ck; the next solve is at t_ck+dt
    dt_global.value = min(float(d['dt']), dt_max.value)
    trestart_global.value = t_ck
    t_global.value = t_ck + dt_as_float(dt_global)
    if rank == 0:
        print(f'RESTART aus {CKPT_DIR}: t = {t_ck:.6g}, dt = {dt_global.value:.3g}, '
              f'{int(success_counter.value)} Schritte bisher, W = {Work.value:.6g}, '
              f'R_ref = {NEWTON_STATE["R_ref"]:.4g}')
        _truncate_table(graph_path, t_ck)
        _truncate_table(logfile_path, t_ck)
    comm.Barrier()
    return t_ck


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
    if args.restart or args.restart_from_xdmf:
        t_ck = load_checkpoint() if args.restart else load_from_xdmf(args.restart_from_xdmf)
        pp.write_meshoutputfile(domain, xdmf_path, comm)
        write_material_fields(t_ck)
        write_solution_fields(t_ck)
        return
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
    # With an absolute tolerance a very small step can be accepted with ZERO
    # Newton iterations (the initial residual is already below atol). alex
    # only grows dt for 0 < iters < min_iters, so such a step would repeat
    # forever at the same tiny dt. Grow it here instead - and move t along
    # with it, so the surfing field, the viscous rate term and the recorded
    # time all belong to the same dt.
    if NEWTON_STATE['last_iters'] == 0:
        new_dt = min(2.0 * dt_as_float(dt_global), dt_max.value)
        dt_global.value = new_dt
        t_global.value = trestart_global.value + new_dt
        NEWTON_STATE['last_iters'] = -1
        t, dt = t_global.value, new_dt
        if rank == 0:
            print(f'0-Iterationen-Schritt -> dt verdoppelt auf {new_dt:.3e}')
    # dt ceiling after a rejected step (see --dt_regrow_steps): alex has
    # possibly just doubled dt again - cap it and move t along consistently.
    ceil = NEWTON_STATE['dt_ceiling']
    if ceil is not None and dt_as_float(dt_global) > ceil * (1.0 + 1e-12):
        dt_global.value = ceil
        t_global.value = trestart_global.value + ceil
        t, dt = t_global.value, ceil
        if rank == 0:
            print(f'dt auf Deckel {ceil:.3e} gehalten '
                  f'({NEWTON_STATE["ok_in_row"]}/{args.dt_regrow_steps} Schritte konvergiert)')
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
    NEWTON_STATE['last_iters'] = int(iters)
    r_end = residual_norm()
    if NEWTON_STATE['dt_ceiling'] is not None:
        NEWTON_STATE['ok_in_row'] += 1
        if NEWTON_STATE['ok_in_row'] >= max(args.dt_regrow_steps, 1):
            NEWTON_STATE['ok_in_row'] = 0
            NEWTON_STATE['dt_ceiling'] = min(2.0 * NEWTON_STATE['dt_ceiling'],
                                             float(dt_max.value))
            if NEWTON_STATE['dt_ceiling'] >= float(dt_max.value):
                NEWTON_STATE['dt_ceiling'] = None

    s_min = comm.allreduce(float(np.min(w.x.array[DOFS_S])), MPI.MIN)
    if rank == 0:
        # A is the regularised crack surface; it must grow monotonically. A
        # falling A means the phase field is still relaxing out of a bad
        # initial state rather than propagating a crack.
        flag = '' if A >= A_history[0] - 1e-9 else '   <-- A faellt (Relaxation!)'
        A_history[0] = max(A_history[0], A)
        print(f'x_tip = {x_ct:9.3f}  Jx = {Jx:10.5f} (Gc = {Gc_val:.4g})  '
              f'A = {A:9.2f}  s_min = {s_min:6.4f}  iters = {iters}  '
              f'|R| {NEWTON_STATE["r0"]:.2e} -> {r_end:.2e} '
              f'(atol {NEWTON_STATE["atol"]:.1e}, R_ref {NEWTON_STATE["R_ref"]:.2e}){flag}')
        pp.write_to_graphs_output_file(graph_path, t, Jx, Jy, x_ct,
                                       xxK1.value[0], Rx_top, Ry_top, dW,
                                       Work.value, A, dt, E_el, E_surf,
                                       E_el + E_surf, float(iters), s_min)

    if x_ct >= x_min + args.separation_frac * Lx_dom:
        raise StopSimulation('specimen separated completely')

    wm1.x.array[:] = w.x.array[:]
    wrestart.x.array[:] = w.x.array[:]
    success_counter.value = success_counter.value + 1.0
    # checkpoint AFTER wm1 is updated: the state is the converged solution at
    # time t (alex hands the hook the time the step was solved at)
    NEWTON_STATE['t_ok'] = float(t)
    if args.checkpoint_interval and \
            int(success_counter.value) % int(args.checkpoint_interval) == 0:
        write_checkpoint(t)
    # max_steps counts the steps of THIS leg, not the restored total
    if args.max_steps and success_counter.value - NEWTON_STATE['steps_at_start'] >= args.max_steps:
        raise StopSimulation(f'max_steps={args.max_steps} reached (smoke test)')
    if int(success_counter.value) % max(int(args.postprocessing_interval), 1) != 0:
        return
    write_material_fields(t)
    write_solution_fields(t, sig_int)


def after_timestep_restart(t, dt, iters):
    stop_if_dt_too_small(dt)
    NEWTON_STATE['last_iters'] = -1
    if args.dt_regrow_steps > 0:
        # dt is already halved by alex; hold it there for a few steps
        NEWTON_STATE['dt_ceiling'] = dt_as_float(dt_global)
        NEWTON_STATE['ok_in_row'] = 0
    w.x.array[:] = wrestart.x.array[:]


GRAPH_LABELS = ['Jx', 'Jy', 'x_crack_tip', 'x_K_field', 'Rx_top', 'Ry_top',
                'dW', 'W', 'A_surf', 'dt', 'E_el', 'E_surf', 'E_total',
                'newton_iters', 's_min']


def after_last_timestep():
    write_material_fields(t_global.value)
    write_solution_fields(t_global.value)
    if args.checkpoint_interval and NEWTON_STATE.get('t_ok') is not None:
        # wm1/wrestart hold the last CONVERGED state, at time t_ok - not w,
        # which may be a half-finished attempt after a dt collapse
        w.x.array[:] = wm1.x.array[:]
        write_checkpoint(NEWTON_STATE['t_ok'])
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
    'newton_rtol': args.newton_rtol, 'newton_atol': args.newton_atol,
    'newton_atol_rel': args.newton_atol_rel, 'newton_relax': args.newton_relax,
    'solver': args.solver, 'dt_regrow_steps': args.dt_regrow_steps,
    'checkpoint_interval': args.checkpoint_interval, 'restart': bool(args.restart),
    'restart_from_xdmf': args.restart_from_xdmf,
    'xdmf_file': os.path.basename(xdmf_path),
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
        trestart=trestart_global, solver=newton,
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
