"""
Anisotropic elastic homogenization of a WAAM polycrystal RVE (dolfinx v0.7.x).

KUBC (kinematic uniform / linear-displacement boundary conditions): apply the 6
unit macro-strains (Voigt) as a linear displacement field u = eps_mac . x on the
whole outer boundary, solve linear elasticity with a PER-GRAIN anisotropic
stiffness (rotated cubic single crystal), and read off the columns of the
effective 6x6 stiffness Chom from the volume-averaged stress.

Same solver/BC structure as
  Meshing/pygalmesh/data/scripts/009-Binning-Variation-CT-Stiffness/00_template/linearelastic.py
but with the isotropic material replaced by the Neper grain structure.

Run inside the dolfinx container, e.g.:
  python3 homogenize_rve.py --mesh inputs/waam_316L_n300.xdmf \
                            --ori  inputs/grain_ori_316L.txt \
                            --tag  316L
Outputs (next to this script): Chom_<tag>.json, homogenize_rve_<tag>.xdmf (fields),
and a console summary (effective E, nu, anisotropy).

Units: single-crystal C in GPa (config.json) -> Chom in GPa. Mesh length unit is
irrelevant for elastic moduli.
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
import alex.homogenization as hom

import waam_crystal as wc

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="WAAM polycrystal KUBC homogenization")
parser.add_argument("--mesh", required=True, help="RVE mesh .xdmf (with 'grain' cell tag)")
parser.add_argument("--ori", required=True, help="grain_ori_<MAT>.txt")
parser.add_argument("--config", default=None, help="config.json (single-crystal constants)")
parser.add_argument("--tag", default="rve", help="output tag, e.g. material name")
args = parser.parse_args()

script_path = os.path.dirname(os.path.abspath(__file__))
config_path = args.config or os.path.join(script_path, "config.json")
outputfile_xdmf_path = os.path.join(script_path, f"homogenize_rve_{args.tag}.xdmf")
logfile_path = alex.os.logfile_full_path(script_path, f"homogenize_rve_{args.tag}")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

timer = dlfx.common.Timer()
timer.start()

# ---- mesh + per-grain anisotropic stiffness --------------------------------
def resolve(p):
    return p if os.path.isabs(p) else os.path.join(script_path, p)

domain, grain_mt = wc.read_mesh_and_grains(comm, resolve(args.mesh))
cfg = wc.load_config(config_path)
ori_map = wc.read_grain_ori(resolve(args.ori))
Cf, info = wc.build_cell_stiffness(domain, grain_mt, ori_map, cfg)
if rank == 0:
    print(f"[{args.tag}] grains={len(ori_map)}  cells_tagged={info['cells_tagged']}  "
          f"crystals={info['crystal_grain_counts']}  no_ori={info['grains_without_ori']}")
    sys.stdout.flush()

# ---- function space, fields ------------------------------------------------
Ve = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
V = dlfx.fem.FunctionSpace(domain, Ve)
u = dlfx.fem.Function(V, name="u")
urestart = dlfx.fem.Function(V)
du = ufl.TestFunction(V)
ddu = ufl.TrialFunction(V)

# ---- time / load-case bookkeeping (t = Voigt index 0..5) -------------------
dt = dlfx.fem.Constant(domain, 1.0)
t = dlfx.fem.Constant(domain, 0.0)
column = dlfx.fem.Constant(domain, 0.0)
Tend = 6.0 * dt.value

x_min, x_max, y_min, y_max, z_min, z_max = bc.get_dimensions(domain, comm)
vol = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
dx = ufl.dx

atol_scal = 0.04
atol_x = (x_max - x_min) * atol_scal
atol_y = (y_max - y_min) * atol_scal
atol_z = (z_max - z_min) * atol_scal
fdim = domain.topology.dim - 1
boundary = bc.get_boundary_of_box_as_function(domain, comm, atol_x=atol_x, atol_y=atol_y, atol_z=atol_z)
facets_at_boundary = dlfx.mesh.locate_entities_boundary(domain, fdim, boundary)
dofs_at_boundary = dlfx.fem.locate_dofs_topological(V, fdim, facets_at_boundary)

eps_mac = dlfx.fem.Constant(domain, np.zeros((3, 3)))
u_D = dlfx.fem.Function(V)
Chom = np.zeros((6, 6))

# ---- hooks -----------------------------------------------------------------
def before_first_time_step():
    urestart.x.array[:] = np.ones_like(urestart.x.array[:])
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
    Eps_Voigt = np.zeros((6,))
    if int(t) < 6:
        Eps_Voigt[int(t)] = 1.0
    eps_mac.value = np.array([
        [Eps_Voigt[0], Eps_Voigt[5] / 2.0, Eps_Voigt[4] / 2.0],
        [Eps_Voigt[5] / 2.0, Eps_Voigt[1], Eps_Voigt[3] / 2.0],
        [Eps_Voigt[4] / 2.0, Eps_Voigt[3] / 2.0, Eps_Voigt[2]]])
    comm.barrier()
    x = ufl.SpatialCoordinate(domain)
    u_lin = ufl.as_vector([
        eps_mac.value[0, 0] * x[0] + eps_mac.value[0, 1] * x[1] + eps_mac.value[0, 2] * x[2],
        eps_mac.value[1, 0] * x[0] + eps_mac.value[1, 1] * x[1] + eps_mac.value[1, 2] * x[2],
        eps_mac.value[2, 0] * x[0] + eps_mac.value[2, 1] * x[1] + eps_mac.value[2, 2] * x[2]])
    u_D.interpolate(dlfx.fem.Expression(u_lin, V.element.interpolation_points()))
    return [dlfx.fem.dirichletbc(u_D, dofs_at_boundary)]

def after_timestep_success(t, dt, iters):
    pp.write_vector_field(domain, outputfile_xdmf_path, u, t, comm)
    sigma_avg = wc.averaged_sigma_voigt(Cf, u, vol, dx, comm)
    comm.barrier()
    if rank == 0:
        col = int(column.value)
        if col < 6:
            Chom[col] = sigma_avg
        column.value = col + 1
        sol.write_to_newton_logfile(logfile_path, t, dt, iters)
    urestart.x.array[:] = u.x.array[:]

def after_timestep_restart(t, dt, iters):
    raise RuntimeError("Linear computation - NO RESTART NECESSARY")

def after_last_timestep():
    timer.stop()
    if rank == 0:
        Csym = 0.5 * (Chom + Chom.T)     # symmetrize (numerical)
        print("\n=== Effective stiffness Chom [GPa] (Voigt xx,yy,zz,yz,xz,xy) ===")
        print(np.array_str(Chom, precision=2, suppress_small=True))
        try:
            print("\n" + hom.print_results(Chom))
        except Exception as exc:
            print("iso-equivalent summary skipped:", exc)
        # directional Young's moduli from the compliance S = Chom^-1
        S = np.linalg.inv(Csym)
        E = {"E_x": 1.0 / S[0, 0], "E_y": 1.0 / S[1, 1], "E_z": 1.0 / S[2, 2]}
        print("\nDirectional Young's moduli [GPa]: "
              + ", ".join(f"{k}={v:.1f}" for k, v in E.items()))
        out = {"tag": args.tag, "units": "GPa", "voigt_order": "xx,yy,zz,yz,xz,xy",
               "Chom": Chom.tolist(), "Chom_sym": Csym.tolist(),
               "E_directional": E, "volume": vol,
               "crystal_grain_counts": info["crystal_grain_counts"],
               "single_crystal_cubic_GPa": cfg["single_crystal_cubic_GPa"]}
        with open(os.path.join(script_path, f"Chom_{args.tag}.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved Chom_{args.tag}.json")
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
