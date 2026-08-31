"""
Numerical uniaxial tensile test on a directional WAAM specimen (dolfinx v0.7.x)
to determine the apparent Young's modulus in the loading direction.

The Neper "specimen" bars (spec_<MAT>_V/H/45deg) have the load axis = x and the
build direction at 0 deg (V) / 90 deg (H) / 45 deg to x. Loading each along x
therefore yields the directional modulus E_V (load || build), E_H (load _|_
build), E_45.

Setup (standard symmetric uniaxial test, lateral faces free to contract):
  u_x = 0        on x = x_min          (symmetry)
  u_y = 0        on y = y_min          (symmetry, removes rigid body)
  u_z = 0        on z = z_min          (symmetry, removes rigid body)
  u_x = delta    on x = x_max          (applied, delta = eps0 * Lx)
Per-grain anisotropic stiffness (rotated cubic single crystal), linear elastic.

Apparent modulus:  E = <sigma_xx> / eps_xx ,  eps_xx = delta / Lx
(<.> = volume average; for a free-lateral bar <sigma_xx> = F/A by equilibrium).

Run inside the dolfinx container, e.g.:
  python3 uniaxial_tension.py --mesh inputs/spec_316L_V.xdmf \
                              --ori  inputs/grain_ori_316L_V.txt --tag 316L_V
Outputs: Emodul_<tag>.json + uniaxial_<tag>.xdmf (displacement field).
Units: single-crystal C in GPa -> E in GPa.
"""
import argparse
import json
import os
import sys

import numpy as np
from mpi4py import MPI
import dolfinx as dlfx
import ufl

import alex.os
import alex.boundaryconditions as bc
import alex.postprocessing as pp
import alex.solution as sol

import waam_crystal as wc

parser = argparse.ArgumentParser(description="WAAM directional uniaxial tensile test")
parser.add_argument("--mesh", required=True, help="specimen mesh .xdmf (with 'grain' cell tag)")
parser.add_argument("--ori", required=True, help="grain_ori_<MAT>_<orient>.txt")
parser.add_argument("--config", default=None, help="config.json (single-crystal constants)")
parser.add_argument("--tag", default="spec", help="output tag, e.g. 316L_V")
parser.add_argument("--strain", type=float, default=1.0e-3, help="applied eps_xx (linear)")
args = parser.parse_args()

script_path = os.path.dirname(os.path.abspath(__file__))
config_path = args.config or os.path.join(script_path, "config.json")
outputfile_xdmf_path = os.path.join(script_path, f"uniaxial_{args.tag}.xdmf")
logfile_path = alex.os.logfile_full_path(script_path, f"uniaxial_{args.tag}")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
timer = dlfx.common.Timer(); timer.start()

def resolve(p):
    return p if os.path.isabs(p) else os.path.join(script_path, p)

# ---- mesh + per-grain anisotropic stiffness --------------------------------
domain, grain_mt = wc.read_mesh_and_grains(comm, resolve(args.mesh))
cfg = wc.load_config(config_path)
ori_map = wc.read_grain_ori(resolve(args.ori))
Cf, info = wc.build_cell_stiffness(domain, grain_mt, ori_map, cfg)
if rank == 0:
    print(f"[{args.tag}] grains={len(ori_map)} cells_tagged={info['cells_tagged']} "
          f"crystals={info['crystal_grain_counts']}")
    sys.stdout.flush()

Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
V = dlfx.fem.FunctionSpace(domain, Ve)
u = dlfx.fem.Function(V, name="u")
urestart = dlfx.fem.Function(V)
du = ufl.TestFunction(V)
ddu = ufl.TrialFunction(V)

x_min, x_max, y_min, y_max, z_min, z_max = bc.get_dimensions(domain, comm)
Lx = x_max - x_min
vol = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
dx = ufl.dx
delta = args.strain * Lx
eps_xx = args.strain
fdim = domain.topology.dim - 1

sx = 0.02 * Lx
sy = 0.02 * (y_max - y_min)
sz = 0.02 * (z_max - z_min)
f_xmin = lambda x: np.isclose(x[0], x_min, atol=sx)
f_xmax = lambda x: np.isclose(x[0], x_max, atol=sx)
f_ymin = lambda x: np.isclose(x[1], y_min, atol=sy)
f_zmin = lambda x: np.isclose(x[2], z_min, atol=sz)

def dofs_on(sub, marker):
    facets = dlfx.mesh.locate_entities_boundary(domain, fdim, marker)
    return dlfx.fem.locate_dofs_topological(V.sub(sub), fdim, facets)

bcs = [
    dlfx.fem.dirichletbc(0.0,   dofs_on(0, f_xmin), V.sub(0)),   # u_x=0 at x_min
    dlfx.fem.dirichletbc(0.0,   dofs_on(1, f_ymin), V.sub(1)),   # u_y=0 at y_min
    dlfx.fem.dirichletbc(0.0,   dofs_on(2, f_zmin), V.sub(2)),   # u_z=0 at z_min
    dlfx.fem.dirichletbc(delta, dofs_on(0, f_xmax), V.sub(0)),   # u_x=delta at x_max
]

dt = dlfx.fem.Constant(domain, 1.0)
t = dlfx.fem.Constant(domain, 0.0)
Tend = 1.0 * dt.value

def before_first_time_step():
    urestart.x.array[:] = np.zeros_like(urestart.x.array[:])
    if rank == 0:
        sol.prepare_newton_logfile(logfile_path)
    pp.write_meshoutputfile(domain, outputfile_xdmf_path, comm)

def before_each_time_step(t, dt):
    if rank == 0:
        sol.print_time_and_dt(t, dt)

def get_residuum_and_gateaux(delta_t):
    pot = wc.strain_energy_density(Cf, u) * dx
    Res = ufl.derivative(pot, u, du)
    dResdw = ufl.derivative(Res, u, ddu)
    return [Res, dResdw]

def get_bcs(t):
    return bcs

def avg(expr):
    loc = dlfx.fem.assemble_scalar(dlfx.fem.form(expr * dx)) / vol
    return comm.allreduce(loc, op=MPI.SUM)

def after_timestep_success(t, dt, iters):
    pp.write_vector_field(domain, outputfile_xdmf_path, u, t, comm)
    sig = wc.averaged_sigma_voigt(Cf, u, vol, dx, comm)
    eyy = avg(u[1].dx(1)); ezz = avg(u[2].dx(2))
    comm.barrier()
    if rank == 0:
        E_app = sig[0] / eps_xx
        nu_xy = -eyy / eps_xx
        nu_xz = -ezz / eps_xx
        print(f"\n=== {args.tag}: uniaxial (load along x) ===")
        print(f"  applied eps_xx = {eps_xx:.3e}")
        print(f"  <sigma_xx>     = {sig[0]:.4f} GPa")
        print(f"  E_apparent     = {E_app:.2f} GPa")
        print(f"  nu_xy={nu_xy:.3f}  nu_xz={nu_xz:.3f}")
        out = {"tag": args.tag, "units": "GPa", "load_axis": "x",
               "applied_eps_xx": eps_xx, "avg_sigma_voigt": sig.tolist(),
               "E_apparent_GPa": E_app, "nu_xy": nu_xy, "nu_xz": nu_xz,
               "volume": vol, "crystal_grain_counts": info["crystal_grain_counts"]}
        with open(os.path.join(script_path, f"Emodul_{args.tag}.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"  saved Emodul_{args.tag}.json")
        sol.write_to_newton_logfile(logfile_path, t, dt, iters)
    urestart.x.array[:] = u.x.array[:]

def after_timestep_restart(t, dt, iters):
    raise RuntimeError("Linear computation - NO RESTART NECESSARY")

def after_last_timestep():
    timer.stop()
    if rank == 0:
        sol.print_runtime(timer.elapsed())

sol.solve_with_newton_adaptive_time_stepping(
    domain, u, Tend, dt,
    before_first_timestep_hook=before_first_time_step,
    after_last_timestep_hook=after_last_timestep,
    before_each_timestep_hook=before_each_time_step,
    get_residuum_and_gateaux=get_residuum_and_gateaux,
    get_bcs=get_bcs,
    after_timestep_restart_hook=after_timestep_restart,
    after_timestep_success_hook=after_timestep_success,
    comm=comm, print_bool=True, t=t, dt_never_scale_up=True)
