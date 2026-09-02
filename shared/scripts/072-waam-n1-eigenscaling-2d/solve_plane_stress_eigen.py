#!/usr/bin/env python3
"""
2D plane-stress uniaxial tension on the EBSD-reconstructed WAAM N=1
transition-region microstructure with EIGEN-SCALED transition stiffness
(dolfinx v0.7.3). Project 072 — generalisation of 070's scalar s(x):

Inside the transition zone (region 1) the three irreducible parts of the
cubic single-crystal tensor are scaled separately, in the crystal frame,
before rotation and plane-stress condensation:

    C_cell = P( R(g_grain) . [ aK(x)*Ch + aCp(x)*Ct + aC44(x)*Cs ] )

with K bulk, C' = (C11-C12)/2 tetragonal shear, C44 trigonal shear
(--aK/--aCp/--aC44, numpy expressions in x [um], default "1.0";
--sfun sets all three at once = exact 070 behaviour).

Everything else (mesh, BCs, outputs, frames) is identical to
070/solve_plane_stress.py:

Loading (x = horizontal map axis = load axis):
    u_x = 0 on x = 0,  u_x = eps0*Lx on x = Lx,
    u_y = 0 at corner node (0,0); top/bottom traction-free.

Outputs: E_<tag>.json, fields_<tag>.npz, ps_<tag>.xdmf with cell fields
    grain_id, E_x_local_GPa, region, phase_fcc1_bcc2,
    aK_factor, aCp_factor, aC44_factor, u, sig_xx/yy/xy, eps_xx.

Frame handling as in 070: npz Bunge angles are TSL MAP frame (y down); mesh
is built y-UP; orientations go through plane_stress_crystal.FLIP_X180.

Runs standalone (no `alex` package) in any dolfinx 0.7.x environment:
    python3 solve_plane_stress_eigen.py --micro <path>/micro_roi.npz --tag base
"""
import argparse, json, os

import numpy as np
from mpi4py import MPI
import dolfinx as dlfx
from dolfinx.fem.petsc import LinearProblem
import ufl

import materials_eigen_2d as M2

ap = argparse.ArgumentParser()
ap.add_argument('--micro', required=True, help='micro_<tag>.npz (z.B. ../070-waam-n1-transition-2d/micro_roi.npz)')
ap.add_argument('--config', default=None, help='config.json (crystal constants)')
ap.add_argument('--tag', default='eigen2d')
ap.add_argument('--strain', type=float, default=1e-3)
ap.add_argument('--aK', default='1.0',
                help='bulk factor aK(x) in the transition zone; numpy expr in x [um]')
ap.add_argument('--aCp', default='1.0',
                help="tetragonal-shear factor aCp(x) on C'=(C11-C12)/2")
ap.add_argument('--aC44', default='1.0',
                help='trigonal-shear factor aC44(x) on C44')
ap.add_argument('--sfun', default=None,
                help='overrides all three factors (aK=aCp=aC44=s(x)); '
                     'exact 070 compatibility mode')
args = ap.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
here = os.path.dirname(os.path.abspath(__file__))

# ---- microstructure grid ---------------------------------------------------
d = np.load(args.micro if os.path.isabs(args.micro) else os.path.join(here, args.micro))
euler = d['euler_deg']; phase = d['phase']; gid = d['grain_id']; zone = d['zone']
meta = json.loads(str(d['meta']))
ny, nx = phase.shape
step = float(meta['step_um'])
Lx, Ly = nx * step, ny * step

cfg = M2.load_config(args.config, here)
afuns = M2.make_factor_funs(args.aK, args.aCp, args.aC44, sfun=args.sfun)
factor_exprs = ({n: args.sfun for n in M2.FACTOR_NAMES} if args.sfun is not None
                else {'aK': args.aK, 'aCp': args.aCp, 'aC44': args.aC44})
if rank == 0:
    print(M2.describe(cfg))
    print('Faktoren (nur region 1): ' + ', '.join(
        f'{n}(x) = {e}' for n, e in factor_exprs.items()))

# per-cell tensors on the microstructure grid (shared with the numpy reference)
Ccell, Excell, a_maps, minfo = M2.build_cell_tensors(
    euler, phase, gid, zone, (np.arange(nx) + 0.5) * step, cfg, afuns,
    verbose=(rank == 0))

# ---- mesh (quadrilaterals, one cell per grid cell), FE frame y-up ----------
domain = dlfx.mesh.create_rectangle(
    comm, [np.array([0.0, 0.0]), np.array([Lx, Ly])], [nx, ny],
    cell_type=dlfx.mesh.CellType.quadrilateral)
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, tdim)

# ---- per-cell plane-stress stiffness ---------------------------------------
ncells = domain.topology.index_map(tdim).size_local
mid = dlfx.mesh.compute_midpoints(domain, tdim, np.arange(ncells, dtype=np.int32))
ci = np.clip((mid[:, 0] / step).astype(int), 0, nx - 1)
cj = np.clip(ny - 1 - (mid[:, 1] / step).astype(int), 0, ny - 1)   # row 0 = top

Qe = ufl.TensorElement('DG', domain.ufl_cell(), 0, shape=(3, 3))
Q = dlfx.fem.FunctionSpace(domain, Qe)
Cf = dlfx.fem.Function(Q, name='C2D')
Cflat = Cf.x.array.reshape(-1, 9)
Ze = ufl.FiniteElement('DG', domain.ufl_cell(), 0)
Z = dlfx.fem.FunctionSpace(domain, Ze)
zonef = dlfx.fem.Function(Z, name='region')
phasef = dlfx.fem.Function(Z, name='phase_fcc1_bcc2')
grainf = dlfx.fem.Function(Z, name='grain_id')
exf = dlfx.fem.Function(Z, name='E_x_local_GPa')
afs = {n: dlfx.fem.Function(Z, name=f'{n}_factor') for n in M2.FACTOR_NAMES}

# vectorised gather: every cell picks its own grain's tensor
Cflat[:ncells] = Ccell[cj, ci].reshape(-1, 9)
zonef.x.array[:ncells] = zone[cj, ci]
phasef.x.array[:ncells] = phase[cj, ci]
grainf.x.array[:ncells] = gid[cj, ci]
exf.x.array[:ncells] = Excell[cj, ci]
for n in M2.FACTOR_NAMES:
    afs[n].x.array[:ncells] = a_maps[n][cj, ci]
for f in (Cf, zonef, phasef, grainf, exf, *afs.values()):
    f.x.scatter_forward()
if rank == 0:
    print(f'grid {ny}x{nx}, {ncells} lokale Zellen, '
          f'{minfo["n_distinct_grain_tensors"]} verschiedene Korn-Orientierungen')

# ---- variational problem ----------------------------------------------------
Ve = ufl.VectorElement('Lagrange', domain.ufl_cell(), 1)
V = dlfx.fem.FunctionSpace(domain, Ve)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

def eps2(w):
    return ufl.as_vector([w[0].dx(0), w[1].dx(1), w[0].dx(1) + w[1].dx(0)])

a_form = ufl.inner(ufl.dot(Cf, eps2(u)), eps2(v)) * ufl.dx
L_form = ufl.inner(dlfx.fem.Constant(domain, np.zeros(2)), v) * ufl.dx

fdim = tdim - 1
delta = args.strain * Lx
f_left = lambda x: np.isclose(x[0], 0.0)
f_right = lambda x: np.isclose(x[0], Lx)
dofs_l = dlfx.fem.locate_dofs_topological(
    V.sub(0), fdim, dlfx.mesh.locate_entities_boundary(domain, fdim, f_left))
dofs_r = dlfx.fem.locate_dofs_topological(
    V.sub(0), fdim, dlfx.mesh.locate_entities_boundary(domain, fdim, f_right))
V1, _ = V.sub(1).collapse()
corner = dlfx.fem.locate_dofs_geometrical(
    (V.sub(1), V1), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
uy0 = dlfx.fem.Function(V1); uy0.x.array[:] = 0.0
bcs = [dlfx.fem.dirichletbc(0.0, dofs_l, V.sub(0)),
       dlfx.fem.dirichletbc(delta, dofs_r, V.sub(0)),
       dlfx.fem.dirichletbc(uy0, corner, V.sub(1))]

problem = LinearProblem(a_form, L_form, bcs=bcs,
                        petsc_options={'ksp_type': 'cg', 'pc_type': 'gamg',
                                       'ksp_rtol': 1e-10, 'ksp_max_it': 5000})
uh = problem.solve()
uh.name = 'u'
niter = problem.solver.getIterationNumber()
if rank == 0:
    print(f'solved: {niter} CG iterations')

# ---- postprocessing ----------------------------------------------------------
sig = ufl.dot(Cf, eps2(uh))
area = Lx * Ly

def integrate(expr):
    loc = dlfx.fem.assemble_scalar(dlfx.fem.form(expr * ufl.dx))
    return comm.allreduce(loc, op=MPI.SUM)

eps0 = args.strain
res = {'tag': args.tag, 'units': 'GPa, um', 'load_axis': 'x',
       'applied_eps_xx': eps0, 'factors': factor_exprs,
       'grid': [ny, nx], 'Lx_um': Lx, 'Ly_um': Ly, 'materials': minfo}
res['E_apparent_GPa'] = integrate(sig[0]) / area / eps0
res['nu_xy_apparent'] = -(integrate(uh[1].dx(1)) / area) / eps0

# per-zone: local modulus like a DIC extensometer, E = <sig_xx>/<eps_xx>
names = {0: '17-4PH', 1: 'transition', 2: '316L'}
for z, nm in names.items():
    ind = ufl.conditional(ufl.eq(zonef, float(z)), 1.0, 0.0)
    az = integrate(ind)
    if az < 1e-12:
        continue
    s_xx = integrate(sig[0] * ind) / az
    e_xx = integrate(uh[0].dx(0) * ind) / az
    res[f'zone_{nm}'] = {'area_frac': az / area, 'avg_sigma_xx_GPa': s_xx,
                         'avg_eps_xx': e_xx, 'E_local_GPa': s_xx / e_xx}
# local E(x) profile: ~100 um bins along x, E = sum(sig_xx)/sum(eps_xx) per bin
S0 = dlfx.fem.FunctionSpace(domain, Ze)
sx_f = dlfx.fem.Function(S0)
sx_f.interpolate(dlfx.fem.Expression(sig[0], S0.element.interpolation_points()))
ex_f = dlfx.fem.Function(S0)
ex_f.interpolate(dlfx.fem.Expression(uh[0].dx(0), S0.element.interpolation_points()))
nbin = max(int(round(Lx / 100.0)), 10)
edges = np.linspace(0.0, Lx, nbin + 1)
ib = np.clip(np.digitize(mid[:, 0], edges) - 1, 0, nbin - 1)
ssum = np.bincount(ib, weights=sx_f.x.array[:ncells], minlength=nbin)
esum = np.bincount(ib, weights=ex_f.x.array[:ncells], minlength=nbin)
ssum = comm.allreduce(ssum, op=MPI.SUM)
esum = comm.allreduce(esum, op=MPI.SUM)
res['E_profile'] = {'x_um': (0.5 * (edges[:-1] + edges[1:])).tolist(),
                    'E_GPa': (ssum / esum).tolist(),
                    'bin_um': float(Lx / nbin)}

if rank == 0:
    print(json.dumps({k: v for k, v in res.items() if k != 'E_profile'}, indent=2))
    with open(os.path.join(here, f'E_{args.tag}.json'), 'w') as fh:
        json.dump(res, fh, indent=2)

# ---- grid-shaped field dump (fields_<tag>.npz) ------------------------------
_comp = {'sig_xx': sig[0], 'sig_yy': sig[1], 'sig_xy': sig[2],
         'eps_xx': uh[0].dx(0), 'eps_yy': uh[1].dx(1)}
_local = {'cj': cj, 'ci': ci}
for _name, _expr in _comp.items():
    _f = dlfx.fem.Function(S0)
    _f.interpolate(dlfx.fem.Expression(_expr, S0.element.interpolation_points()))
    _local[_name] = _f.x.array[:ncells].copy()
_local['E_x'] = Excell[cj, ci]
_local['grain'] = gid[cj, ci]
_local['region'] = zone[cj, ci]
_local['phase'] = phase[cj, ci]
for n in M2.FACTOR_NAMES:
    _local[n] = a_maps[n][cj, ci]
_gathered = comm.gather(_local, root=0)
if rank == 0:
    _out = {}
    for _key in [k for k in _local if k not in ('cj', 'ci')]:
        _arr = np.zeros((ny, nx))
        for _g in _gathered:
            _arr[_g['cj'], _g['ci']] = _g[_key]
        _out[_key] = _arr
    _out['meta'] = json.dumps({'tag': args.tag, 'factors': factor_exprs,
                               'Lx_um': Lx, 'Ly_um': Ly, 'step_um': step,
                               'solver': 'dolfinx', 'units': 'GPa, um',
                               'row0': 'top of the EBSD map (y_fe = Ly)'})
    np.savez_compressed(os.path.join(here, f'fields_{args.tag}.npz'), **_out)
    print(f'wrote fields_{args.tag}.npz')

# ---- fields to XDMF ----------------------------------------------------------
S = dlfx.fem.FunctionSpace(domain, Ze)
out = dlfx.io.XDMFFile(comm, os.path.join(here, f'ps_{args.tag}.xdmf'), 'w')
out.write_mesh(domain)
out.write_function(uh, 0.0)
for k, name in [(0, 'sig_xx'), (1, 'sig_yy'), (2, 'sig_xy')]:
    f = dlfx.fem.Function(S, name=name)
    f.interpolate(dlfx.fem.Expression(sig[k], S.element.interpolation_points()))
    out.write_function(f, 0.0)
fe = dlfx.fem.Function(S, name='eps_xx')
fe.interpolate(dlfx.fem.Expression(uh[0].dx(0), S.element.interpolation_points()))
out.write_function(fe, 0.0)
out.write_function(zonef, 0.0)       # 0=17-4PH 1=Uebergang 2=316L
out.write_function(phasef, 0.0)      # 1=fcc 2=bcc (Kristallsystem, NICHT das Korn)
out.write_function(grainf, 0.0)      # Korn-ID -> Kornstruktur sichtbar
out.write_function(exf, 0.0)         # E_x des Korns/der Zelle aus ihrem Tensor
for n in M2.FACTOR_NAMES:
    out.write_function(afs[n], 0.0)  # angewandte Faktoren aK/aCp/aC44
out.close()
if rank == 0:
    print(f'wrote ps_{args.tag}.xdmf')
