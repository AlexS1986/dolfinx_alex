"""
Shared helpers for the WAAM polycrystal anisotropy computations (dolfinx).

Both the KUBC homogenization (homogenize_rve.py) and the uniaxial tensile test
(uniaxial_tension.py) need the SAME thing: a per-cell anisotropic elastic
stiffness built from the Neper grain structure, where each grain gets the cubic
single-crystal stiffness of its phase (FCC austenite / BCC martensite) rotated
into the sample frame by the grain's measured Bunge orientation.

Inputs (produced by the Neper pipeline in
  .../Meshing/Neper/data/04_anisotropy_waam/neper_pipeline):
  * <name>.xdmf/.h5   - tet mesh with a single cell tag "grain" (= grain id)
  * grain_ori_<MAT>.txt - lines: "grain_id phi1 Phi phi2 crystal"
                          (Bunge Euler in degrees; crystal in {fcc,bcc,...})

Conventions:
  * Voigt order [xx, yy, zz, yz, xz, xy] with ENGINEERING shear strains,
    matching alex.linearelastic.eps_voigt_3D / cmat_voigt_3D.
  * Stiffness rotation via the 6x6 stress Bond matrix M:  C_sample = M C_xtal M^T,
    with M built from a = g^T (sample<-crystal), g = crystal<-sample (Bunge).
  * Single-crystal constants are in GPa; the mesh length unit (um) is irrelevant
    for the resulting elastic moduli (strain is dimensionless).

The numpy-only helpers at the top are unit-testable without dolfinx.
"""
import json
import os

import numpy as np


# ===========================================================================
# Pure-numpy crystal elasticity (no dolfinx needed) -- unit-testable.
# ===========================================================================
def cubic_C(C11, C12, C44):
    """Cubic single-crystal stiffness as a 6x6 Voigt matrix (order xx,yy,zz,yz,xz,xy)."""
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    C[0, 0] = C[1, 1] = C[2, 2] = C11
    C[3, 3] = C[4, 4] = C[5, 5] = C44
    return C


def bunge_to_g(phi1, Phi, phi2):
    """Bunge Euler (rad) -> rotation g with v_crystal = g @ v_sample."""
    c1, s1, c, s, c2, s2 = (np.cos(phi1), np.sin(phi1), np.cos(Phi),
                            np.sin(Phi), np.cos(phi2), np.sin(phi2))
    return np.array([
        [c1 * c2 - s1 * s2 * c,  s1 * c2 + c1 * s2 * c, s2 * s],
        [-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s],
        [s1 * s,                 -c1 * s,                c]])


def bond_matrix(a):
    """6x6 stress Bond matrix M for C_sample = M @ C_crystal @ M.T, with a the
    rotation mapping crystal->sample (a = g^T). Voigt order xx,yy,zz,yz,xz,xy."""
    M = np.zeros((6, 6))
    M[:3, :3] = a ** 2
    M[:3, 3:] = 2 * np.array([
        [a[0, 1] * a[0, 2], a[0, 2] * a[0, 0], a[0, 0] * a[0, 1]],
        [a[1, 1] * a[1, 2], a[1, 2] * a[1, 0], a[1, 0] * a[1, 1]],
        [a[2, 1] * a[2, 2], a[2, 2] * a[2, 0], a[2, 0] * a[2, 1]]])
    M[3:, :3] = np.array([
        [a[1, 0] * a[2, 0], a[1, 1] * a[2, 1], a[1, 2] * a[2, 2]],
        [a[2, 0] * a[0, 0], a[2, 1] * a[0, 1], a[2, 2] * a[0, 2]],
        [a[0, 0] * a[1, 0], a[0, 1] * a[1, 1], a[0, 2] * a[1, 2]]])
    M[3:, 3:] = np.array([
        [a[1, 1] * a[2, 2] + a[1, 2] * a[2, 1], a[1, 2] * a[2, 0] + a[1, 0] * a[2, 2], a[1, 0] * a[2, 1] + a[1, 1] * a[2, 0]],
        [a[2, 1] * a[0, 2] + a[2, 2] * a[0, 1], a[2, 2] * a[0, 0] + a[2, 0] * a[0, 2], a[2, 0] * a[0, 1] + a[2, 1] * a[0, 0]],
        [a[0, 1] * a[1, 2] + a[0, 2] * a[1, 1], a[0, 2] * a[1, 0] + a[0, 0] * a[1, 2], a[0, 0] * a[1, 1] + a[0, 1] * a[1, 0]]])
    return M


def rotated_cubic_C(C11, C12, C44, phi1_deg, Phi_deg, phi2_deg):
    """Single-crystal cubic stiffness rotated into the sample frame (6x6, GPa)."""
    g = bunge_to_g(*np.deg2rad([phi1_deg, Phi_deg, phi2_deg]))
    M = bond_matrix(g.T)                       # a = g^T (sample<-crystal)
    return M @ cubic_C(C11, C12, C44) @ M.T


DEFAULT_CUBIC = {                              # literature placeholders [GPa]
    "fcc": {"C11": 204.6, "C12": 137.7, "C44": 126.2},   # ~316L austenite
    "bcc": {"C11": 231.4, "C12": 134.7, "C44": 116.4},   # ~alpha-Fe / martensite
}


def load_config(path=None):
    """Load config.json (single-crystal constants etc.); fall back to defaults."""
    cfg = {"single_crystal_cubic_GPa": {k: dict(v) for k, v in DEFAULT_CUBIC.items()},
           "crystal_fallback": "fcc"}
    if path and os.path.isfile(path):
        with open(path) as fh:
            user = json.load(fh)
        cfg.update(user)
        # merge crystal table so partial overrides work
        table = {k: dict(v) for k, v in DEFAULT_CUBIC.items()}
        table.update(user.get("single_crystal_cubic_GPa", {}))
        cfg["single_crystal_cubic_GPa"] = table
    return cfg


def read_grain_ori(path):
    """Parse grain_ori_<MAT>.txt -> {grain_id: (phi1, Phi, phi2, crystal)} (deg)."""
    ori = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            gid = int(p[0])
            crystal = p[4] if len(p) > 4 else "unknown"
            ori[gid] = (float(p[1]), float(p[2]), float(p[3]), crystal)
    return ori


def crystal_C_table(cfg):
    """Precompute unrotated cubic C (6x6) for every crystal system in config."""
    tab = {}
    for name, c in cfg["single_crystal_cubic_GPa"].items():
        tab[name] = cubic_C(c["C11"], c["C12"], c["C44"])
    return tab


# ===========================================================================
# dolfinx-dependent helpers (imported lazily so the numpy helpers above can be
# unit-tested in an environment without dolfinx).
# ===========================================================================
def read_mesh_and_grains(comm, mesh_path, grid_name="Grid"):
    """Read the tet mesh and the per-cell grain id (meshtags) from an XDMF that
    carries a single 'grain' cell tag (as written by meshio in the Neper
    pipeline step 04). Returns (domain, grain_meshtags)."""
    import dolfinx as dlfx
    with dlfx.io.XDMFFile(comm, mesh_path, "r") as xf:
        domain = xf.read_mesh(name=grid_name)
        tdim = domain.topology.dim
        domain.topology.create_connectivity(tdim, tdim)
        grain_mt = xf.read_meshtags(domain, name=grid_name)
    return domain, grain_mt


def build_cell_stiffness(domain, grain_mt, ori_map, cfg):
    """Build a DG-0 (6x6) stiffness Function: each cell gets its grain's cubic
    single-crystal stiffness (by phase) rotated by the grain's Bunge angles.

    Returns (Cf, info) where info has per-crystal cell counts. Robust in
    parallel: owned cells filled from meshtags, ghosts synced via scatter."""
    import dolfinx as dlfx
    import ufl
    tdim = domain.topology.dim
    Ctab = crystal_C_table(cfg)
    fallback = cfg.get("crystal_fallback", "fcc")

    # cache rotated C per grain id (many cells share a grain)
    C_by_grain = {}
    counts = {}
    for gid, (phi1, Phi, phi2, crystal) in ori_map.items():
        Cx = Ctab.get(crystal, Ctab[fallback])
        g = bunge_to_g(*np.deg2rad([phi1, Phi, phi2]))
        M = bond_matrix(g.T)
        C_by_grain[gid] = (M @ Cx @ M.T).flatten()
        counts[crystal] = counts.get(crystal, 0) + 1

    # dolfinx v0.7.x API: TensorElement + FunctionSpace (row-major (6,6) blocks)
    Qe = ufl.TensorElement("DG", domain.ufl_cell(), 0, shape=(6, 6))
    Q = dlfx.fem.FunctionSpace(domain, Qe)
    Cf = dlfx.fem.Function(Q, name="stiffness")
    Cflat = Cf.x.array.reshape(-1, 36)
    # default (mean of table) for any unmapped/ghost cell
    C_default = np.mean([C for C in Ctab.values()], axis=0).flatten()
    Cflat[:] = C_default
    missing = 0
    for local_cell, gid in zip(grain_mt.indices, grain_mt.values):
        row = C_by_grain.get(int(gid))
        if row is None:
            missing += 1
            continue
        Cflat[local_cell] = row
    Cf.x.scatter_forward()
    info = {"crystal_grain_counts": counts, "cells_tagged": int(len(grain_mt.indices)),
            "grains_without_ori": missing}
    return Cf, info


def eps_voigt_3D(u):
    """Engineering-strain Voigt vector [xx,yy,zz,yz,xz,xy] (matches alex)."""
    import ufl
    return ufl.as_vector([u[0].dx(0), u[1].dx(1), u[2].dx(2),
                          u[1].dx(2) + u[2].dx(1),
                          u[2].dx(0) + u[0].dx(2),
                          u[0].dx(1) + u[1].dx(0)])


def sigma_voigt(Cf, u):
    """Anisotropic stress in Voigt notation: sigma = C(x) . eps(u)."""
    import ufl
    return ufl.dot(Cf, eps_voigt_3D(u))


def voigt_to_tensor(sig6):
    """UFL 3x3 symmetric stress tensor from a Voigt-6 stress vector."""
    import ufl
    return ufl.as_matrix([[sig6[0], sig6[5], sig6[4]],
                          [sig6[5], sig6[1], sig6[3]],
                          [sig6[4], sig6[3], sig6[2]]])


def strain_energy_density(Cf, u):
    """0.5 * eps^T C eps  (anisotropic elastic energy density)."""
    import ufl
    e = eps_voigt_3D(u)
    return 0.5 * ufl.dot(e, ufl.dot(Cf, e))


def averaged_sigma_voigt(Cf, u, vol, dx, comm):
    """Volume-averaged stress (Voigt-6): (1/vol) integral of C.eps(u)."""
    import dolfinx as dlfx
    from mpi4py import MPI
    sig = sigma_voigt(Cf, u)
    out = np.zeros(6)
    for k in range(6):
        local = dlfx.fem.assemble_scalar(dlfx.fem.form(sig[k] * dx)) / vol
        out[k] = comm.allreduce(local, op=MPI.SUM)
    return out
