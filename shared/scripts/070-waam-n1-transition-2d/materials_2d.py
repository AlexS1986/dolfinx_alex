#!/usr/bin/env python3
"""
Material assignment for the 2D transition model — shared by BOTH solvers
(`solve_plane_stress.py`, dolfinx; `reference_solver_numpy.py`), so the two
differ only in the FE machinery, never in the material law.

Assignment rule (per microstructure cell):

    region (zone)  ->  material          ->  crystal constants of that material
    0              ->  "17-4PH"              indexed by the GRAIN's crystal
    1              ->  "trans"               system (fcc / bcc) from EBSD
    2              ->  "316L"

    C_cell = s(x) * P( R(g_grain) . C_cubic[material][crystal] )

with R the Bond rotation by that grain's own Bunge angles, P the exact
plane-stress condensation, and s(x) the transition-zone prefactor (only
applied in region 1, default 1.0).

So EVERY GRAIN gets its own tensor: same material + same crystal system still
give different C because the orientation g differs per grain. The cell fields
`phase` (2 values: fcc/bcc) and `region` (3 values) are only bookkeeping; the
per-grain variation is visible in `grain_id` and `E_x_local` (see solvers).

All constants live in `config.json` next to this file — that is the single
place to edit them.
"""
import json
import os

import numpy as np

from plane_stress_crystal import (cubic_C, bunge_to_g, rotate_C,
                                  plane_stress_condense, FLIP_X180)

# Literature PLACEHOLDERS [GPa] — same values as 069/config.json.
# Replace with your own; every result scales with them.
DEFAULT_CUBIC = {
    "17-4PH": {                                   # martensitic, region 0
        "bcc": {"C11": 231.4, "C12": 134.7, "C44": 116.4},   # martensite matrix
        "fcc": {"C11": 204.6, "C12": 137.7, "C44": 126.2},   # retained austenite
    },
    "316L": {                                     # austenitic, region 2
        "fcc": {"C11": 204.6, "C12": 137.7, "C44": 126.2},
        "bcc": {"C11": 231.4, "C12": 134.7, "C44": 116.4},   # stray ferrite
    },
    "trans": {                                    # transition zone, region 1
        # default = the parent-phase constants (composition unknown); edit here
        # to give the transition its OWN crystal constants. The scalar
        # prefactor s(x) is the separate, spatially varying knob.
        "fcc": {"C11": 204.6, "C12": 137.7, "C44": 126.2},
        "bcc": {"C11": 231.4, "C12": 134.7, "C44": 116.4},
    },
}
REGION_MATERIAL = {0: "17-4PH", 1: "trans", 2: "316L"}
PHASE_CRYSTAL = {1: "fcc", 2: "bcc"}
SCALED_REGION = 1                      # only the transition zone gets s(x)


def load_config(path=None, here=None):
    """Read config.json; missing entries fall back to DEFAULT_CUBIC.

    Accepts both the region-aware layout
        {"single_crystal_cubic_GPa": {"316L": {"fcc": {...}}, ...}}
    and the older flat layout {"single_crystal_cubic_GPa": {"fcc": {...}}},
    which is then applied to every material.
    """
    here = here or os.path.dirname(os.path.abspath(__file__))
    cfg = {"single_crystal_cubic_GPa": {m: {c: dict(v) for c, v in d.items()}
                                        for m, d in DEFAULT_CUBIC.items()},
           "region_material": dict(REGION_MATERIAL)}
    path = path or os.path.join(here, "config.json")
    if not os.path.isfile(path):
        return cfg
    with open(path) as fh:
        user = json.load(fh)
    tab = user.get("single_crystal_cubic_GPa", {})
    flat = {k: v for k, v in tab.items() if k in ("fcc", "bcc", "hcp")}
    for mat, per_crystal in cfg["single_crystal_cubic_GPa"].items():
        src = tab.get(mat, flat)
        for crystal, const in src.items():
            # skip "_comment"-style annotations, keep only {C11, C12, C44} dicts
            if not isinstance(const, dict) or "C11" not in const:
                continue
            if crystal in per_crystal:
                per_crystal[crystal].update(const)
            else:
                per_crystal[crystal] = dict(const)
    if "region_material" in user:
        cfg["region_material"] = {int(k): v for k, v in user["region_material"].items()}
    return cfg


def constants_table(cfg):
    """{material: {crystal: 6x6 unrotated cubic C}} from a config dict."""
    return {mat: {cry: cubic_C(c["C11"], c["C12"], c["C44"])
                  for cry, c in per.items()}
            for mat, per in cfg["single_crystal_cubic_GPa"].items()}


def E_directional(C2, theta_deg=0.0):
    """Young's modulus of a 3x3 plane-stress stiffness in direction theta
    (deg from x). E = 1 / S'_11 with S' the rotated compliance."""
    S = np.linalg.inv(C2)
    c, s = np.cos(np.deg2rad(theta_deg)), np.sin(np.deg2rad(theta_deg))
    # rotate compliance: S'_11 = sum over the transformed strain/stress basis
    s11 = (S[0, 0] * c**4 + S[1, 1] * s**4
           + (2 * S[0, 1] + S[2, 2]) * c**2 * s**2
           + 2 * (S[0, 2] * c**2 + S[1, 2] * s**2) * c * s)
    return 1.0 / s11


def build_cell_tensors(euler, phase, gid, zone, x_um, cfg, sfun=None,
                       verbose=True):
    """Per-cell plane-stress stiffness for the whole microstructure grid.

    Parameters
    ----------
    euler : (ny,nx,3) Bunge angles [deg], TSL MAP frame
    phase : (ny,nx)   1=fcc, 2=bcc
    gid   : (ny,nx)   grain id
    zone  : (ny,nx)   0/1/2 region tag
    x_um  : (nx,)     cell-centre x coordinates [um] (load axis)
    cfg   : dict from load_config()
    sfun  : callable s(x) applied in region SCALED_REGION only (default 1)

    Returns
    -------
    C     : (ny,nx,3,3) stiffness [GPa]
    Ex    : (ny,nx)     directional Young's modulus along x of each cell
    s_map : (ny,nx)     the applied prefactor (1 outside the transition zone)
    info  : dict with grain counts and the constants actually used
    """
    ny, nx = phase.shape
    tab = constants_table(cfg)
    rmat = cfg["region_material"]
    sfun = sfun or (lambda x: 1.0)

    C = np.empty((ny, nx, 3, 3))
    Ex = np.empty((ny, nx))
    s_map = np.ones((ny, nx))
    cache = {}
    counts = {}
    for j in range(ny):
        zrow = zone[j]
        for i in range(nx):
            z = int(zrow[i])
            mat = rmat.get(z, rmat.get(str(z), "316L"))
            cry = PHASE_CRYSTAL.get(int(phase[j, i]), "fcc")
            key = (int(gid[j, i]), mat, cry)
            hit = cache.get(key)
            if hit is None:
                g = bunge_to_g(*np.deg2rad(euler[j, i]))
                C0 = tab[mat][cry] if cry in tab[mat] else next(iter(tab[mat].values()))
                C2 = plane_stress_condense(rotate_C(C0, g, pre_rot=FLIP_X180))
                hit = (C2, E_directional(C2))
                cache[key] = hit
                counts[(mat, cry)] = counts.get((mat, cry), 0) + 1
            s = float(np.asarray(sfun(x_um[i]))) if z == SCALED_REGION else 1.0
            C[j, i] = s * hit[0]
            Ex[j, i] = s * hit[1]
            s_map[j, i] = s

    info = {"n_distinct_grain_tensors": len(cache),
            "grains_per_material_crystal": {f"{m}/{c}": n
                                            for (m, c), n in sorted(counts.items())},
            "region_material": {str(k): v for k, v in rmat.items()},
            "single_crystal_cubic_GPa": cfg["single_crystal_cubic_GPa"],
            "E_x_range_GPa": [float(Ex.min()), float(Ex.max())]}
    if verbose:
        print(f'Materialzuordnung: {info["grains_per_material_crystal"]}')
        print(f'  {len(cache)} verschiedene Korn-Steifigkeitstensoren; '
              f'E_x pro Zelle {Ex.min():.0f}...{Ex.max():.0f} GPa')
    return C, Ex, s_map, info


def describe(cfg):
    """Human-readable dump of the constants in use (printed by the solvers)."""
    lines = ['Einkristall-Konstanten [GPa] (aus config.json, LITERATUR-PLATZHALTER):']
    for z, mat in sorted(cfg["region_material"].items()):
        for cry, c in cfg["single_crystal_cubic_GPa"][mat].items():
            lines.append(f'  region {z} = {mat:8s} {cry}: '
                         f'C11={c["C11"]:.1f} C12={c["C12"]:.1f} C44={c["C44"]:.1f}')
    return '\n'.join(lines)


if __name__ == '__main__':
    print(describe(load_config()))
